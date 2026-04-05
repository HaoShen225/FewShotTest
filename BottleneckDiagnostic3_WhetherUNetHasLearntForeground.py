from __future__ import annotations

import csv
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Config
# =========================================================
@dataclass(frozen=True)
class Variant:
    name: str
    model_type: str
    base_ch: int
    latent_ch: int
    weight_decay: float
    use_early_stopping: bool = False
    es_patience: int = 20
    es_min_delta: float = 1e-4


@dataclass
class Cfg:
    # -------- data --------
    domain_root: str = "SPIDER_domain_strict/SIEMENS_SymphonyTim_37743_T1-TSE"
    image_dir: str = "images"
    mask_dir: str = "masks"

    out_dir: str = "UNet_singleDomainTest/LayeredSplit_TargetedCoverage"

    n_train_patients: int = 5
    split_seed: int = 0
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    hard_outer_patient_ids: Tuple[str, ...] = ("222", "190", "180", "23", "179")
    targeted_patient_ids: Tuple[str, ...] = ("190", "222", "180")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- slice selection --------
    num_middle_slices: int = 9

    # -------- image size --------
    resize_hw: Tuple[int, int] = (224, 224)

    # -------- intensity --------
    input_clip_min: float = -3.0
    input_clip_max: float = 3.0

    # -------- foreground labels --------
    fg_labels: Optional[Tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)

    # -------- optimization --------
    epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-3

    # -------- segmentation loss --------
    lambda_bce: float = 1.0
    lambda_dice: float = 0.0

    # -------- prediction --------
    pred_thr: float = 0.5

    # -------- region stability analysis --------
    coarse_thresholds: Tuple[float, ...] = (0.2, 0.3, 0.4)
    main_region_thr: float = 0.3
    fg_roi_dilate_px: int = 8
    failure_recall_good: float = 0.85
    failure_recall_bad: float = 0.60
    failure_bbox_good: float = 0.90
    failure_bbox_bad: float = 0.70

    # -------- latent / sparsity analysis --------
    latent_zero_thr: float = 1e-6
    targeted_channel_topk: int = 16
    sparsity_layers: Tuple[str, ...] = ("enc1", "enc2", "enc3", "z")
    sparsity_threshold_quantile: float = 95.0
    sparsity_threshold_ratio: float = 0.10
    sparsity_region_types: Tuple[str, ...] = ("all", "fg", "roi")

    # -------- visualization / reports --------
    save_all_eval_vis: bool = True
    save_all_test_vis: bool = True
    topk_patients_report: int = 10
    topk_slices_report: int = 30

    # -------- checkpoint --------
    save_checkpoint: bool = True

    # -------- complete experiment variants --------
    variants: Tuple[Variant, ...] = (
        Variant(
            name="unet_b16_l64_wd1e-4",
            model_type="unet",
            base_ch=16,
            latent_ch=64,
            weight_decay=1e-4,
            use_early_stopping=False,
        ),
        Variant(
            name="unet_b16_l64_wd3e-4",
            model_type="unet",
            base_ch=16,
            latent_ch=64,
            weight_decay=3e-4,
            use_early_stopping=False,
        ),
        Variant(
            name="unet_b8_l32_wd1e-4",
            model_type="unet",
            base_ch=8,
            latent_ch=32,
            weight_decay=1e-4,
            use_early_stopping=False,
        ),
        Variant(
            name="unet_b16_l64_es_pat20_wd1e-4",
            model_type="unet",
            base_ch=16,
            latent_ch=64,
            weight_decay=1e-4,
            use_early_stopping=True,
            es_patience=20,
            es_min_delta=1e-4,
        ),
        Variant(
            name="attention_unet_b16_l64_wd1e-4",
            model_type="attention_unet",
            base_ch=16,
            latent_ch=64,
            weight_decay=1e-4,
            use_early_stopping=False,
        ),
    )


cfg = Cfg()
print(cfg)


# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_log_csv(rows: List[Dict], path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["empty"])
        return

    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_patient_id_from_stem(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return stem


def resize_np_2d(arr: np.ndarray, size_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    t = torch.from_numpy(arr[None, None, ...]).float()
    mode = "nearest" if is_mask else "bilinear"
    out = F.interpolate(t, size=size_hw, mode=mode, align_corners=False if mode == "bilinear" else None)
    return out[0, 0].cpu().numpy()


def normalize_slice_to_01(img2d: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    img2d = np.nan_to_num(img2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    img2d = np.clip(img2d, clip_min, clip_max)
    img2d = (img2d - clip_min) / max(clip_max - clip_min, 1e-8)
    img2d = np.clip(img2d, 0.0, 1.0)
    return img2d.astype(np.float32)


def select_middle_sagittal_indices(mask_zyx: np.ndarray, num_slices: int = 5) -> List[int]:
    x_any = np.where(mask_zyx.sum(axis=(0, 1)) > 0)[0]
    x_dim = mask_zyx.shape[2]

    if len(x_any) > 0:
        x_min = int(x_any[0])
        x_max = int(x_any[-1])
        x_center = int(round((x_min + x_max) / 2.0))
    else:
        x_center = x_dim // 2

    if num_slices <= 1:
        idxs = [x_center]
    else:
        half = num_slices // 2
        idxs = list(range(x_center - half, x_center + half + 1))
        if len(idxs) > num_slices:
            idxs = idxs[:num_slices]

    idxs = [min(max(x, 0), x_dim - 1) for x in idxs]
    return idxs


def binarize_mask(mask2d: np.ndarray, fg_labels: Optional[Tuple[int, ...]]) -> np.ndarray:
    if fg_labels is None:
        return (mask2d > 0).astype(np.float32)
    keep = np.isin(mask2d.astype(np.int32), np.asarray(fg_labels, dtype=np.int32))
    return keep.astype(np.float32)


def iou_and_dice(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter) / float(union + 1e-9)
    dice = float(2 * inter) / float(pred.sum() + gt.sum() + 1e-9)
    return iou, dice


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def bbox_to_mask(bbox, shape_hw):
    h, w = shape_hw
    out = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        return out
    y0, x0, y1, x1 = bbox
    out[y0:y1 + 1, x0:x1 + 1] = 1
    return out


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    best = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            comp = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and (not visited[ny, nx]) and mask[ny, nx] > 0:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > len(best):
                best = comp

    out = np.zeros_like(mask)
    for y, x in best:
        out[y, x] = 1
    return out


def center_of_mass(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return np.array([h / 2.0, w / 2.0], dtype=np.float32)
    return np.array([float(np.mean(ys)), float(np.mean(xs))], dtype=np.float32)


def normalized_com_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ca = center_of_mass(mask_a)
    cb = center_of_mass(mask_b)
    h, w = mask_a.shape
    denom = max((h ** 2 + w ** 2) ** 0.5, 1e-6)
    return float(np.linalg.norm(ca - cb) / denom)


def dilate_binary(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    for y, x in zip(ys, xs):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        out[y0:y1, x0:x1] = 1
    return out


def classify_failure_type(recall_t03: float, bbox_cover_t03: float) -> str:
    if recall_t03 >= cfg.failure_recall_good and bbox_cover_t03 >= cfg.failure_bbox_good:
        return "position_correct_boundary_rough"
    if recall_t03 < cfg.failure_recall_bad or bbox_cover_t03 < cfg.failure_bbox_bad:
        return "main_region_missed"
    return "intermediate"


def resize_mask_to_feature(mask_hw: np.ndarray, feat_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask_hw[None, None, ...]).float()
    out = F.interpolate(t, size=feat_hw, mode="nearest")
    return (out[0, 0].numpy() > 0.5).astype(np.uint8)


def compute_layer_sparsity_stats(
    feat_chw: np.ndarray,
    fg_mask_hw: np.ndarray,
    roi_mask_hw: np.ndarray,
    layer_name: str,
) -> List[Dict]:
    feat = np.abs(feat_chw.astype(np.float32))
    c, h, w = feat.shape

    q = np.percentile(feat.reshape(-1), cfg.sparsity_threshold_quantile)
    tau = float(cfg.sparsity_threshold_ratio * q)

    fg_small = resize_mask_to_feature(fg_mask_hw, (h, w))
    roi_small = resize_mask_to_feature(roi_mask_hw, (h, w))
    all_small = np.ones((h, w), dtype=np.uint8)

    region_map = {
        "all": all_small,
        "fg": fg_small,
        "roi": roi_small,
    }

    spatial_resp = feat.max(axis=0)  # [H, W]
    active_spatial = (spatial_resp > tau).astype(np.uint8)

    rows = []
    for region_type in cfg.sparsity_region_types:
        region = region_map[region_type] > 0
        n_region = int(region.sum())

        if n_region == 0:
            rows.append({
                "layer": layer_name,
                "region_type": region_type,
                "space_support_ratio": 0.0,
                "space_sparsity": 1.0,
                "fg_energy_ratio": 0.0,
                "channel_support_ratio": 0.0,
                "channel_sparsity": 1.0,
                "effective_rank": 0.0,
            })
            continue

        space_support_ratio = float(active_spatial[region].mean())
        space_sparsity = 1.0 - space_support_ratio

        energy_total = float(spatial_resp.sum()) + 1e-8
        fg_energy_ratio = float(spatial_resp[region].sum() / energy_total)

        active_ch = (feat > tau).astype(np.uint8)
        ch_count = active_ch[:, region].sum(axis=0)
        channel_support_ratio = float(np.mean(ch_count / max(c, 1)))
        channel_sparsity = 1.0 - channel_support_ratio

        x = feat.reshape(c, -1).T  # [HW, C]
        x = x - np.mean(x, axis=0, keepdims=True)
        cov = (x.T @ x) / max(x.shape[0], 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        if eigvals.sum() <= 1e-12:
            erank = 0.0
        else:
            erank = float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))

        rows.append({
            "layer": layer_name,
            "region_type": region_type,
            "space_support_ratio": space_support_ratio,
            "space_sparsity": space_sparsity,
            "fg_energy_ratio": fg_energy_ratio,
            "channel_support_ratio": channel_support_ratio,
            "channel_sparsity": channel_sparsity,
            "effective_rank": erank,
        })
    return rows


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def summarize_array_stats(arr: np.ndarray, prefix: str, zero_thr: Optional[float] = None) -> Dict[str, float]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_abs_mean": 0.0,
            f"{prefix}_max_abs": 0.0,
            f"{prefix}_l2": 0.0,
            f"{prefix}_p01": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_zero_ratio": 0.0,
        }

    out = {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(flat))),
        f"{prefix}_max_abs": float(np.max(np.abs(flat))),
        f"{prefix}_l2": float(np.sqrt(np.mean(flat ** 2))),
        f"{prefix}_p01": float(np.percentile(flat, 1)),
        f"{prefix}_p05": float(np.percentile(flat, 5)),
        f"{prefix}_p25": float(np.percentile(flat, 25)),
        f"{prefix}_p50": float(np.percentile(flat, 50)),
        f"{prefix}_p75": float(np.percentile(flat, 75)),
        f"{prefix}_p95": float(np.percentile(flat, 95)),
        f"{prefix}_p99": float(np.percentile(flat, 99)),
    }
    thr = cfg.latent_zero_thr if zero_thr is None else zero_thr
    out[f"{prefix}_zero_ratio"] = float(np.mean(np.abs(flat) <= thr))
    return out


def aggregate_metric_rows(rows: List[Dict], group_keys: Tuple[str, ...], sort_key: Optional[str] = None) -> List[Dict]:
    if not rows:
        return []
    numeric_keys = set()
    for r in rows:
        for k, v in r.items():
            if k in group_keys:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_keys.add(k)
    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        grouped[key].append(r)

    out_rows = []
    for key, items in grouped.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_items"] = len(items)
        for nk in sorted(numeric_keys):
            vals = [float(it[nk]) for it in items if nk in it]
            if vals:
                row[f"{nk}_mean"] = float(np.mean(vals))
                row[f"{nk}_std"] = float(np.std(vals))
        out_rows.append(row)
    if sort_key is not None:
        out_rows = sorted(out_rows, key=lambda x: x.get(sort_key, 0.0))
    return out_rows


# =========================================================
# Dataset
# =========================================================
def collect_volume_records(cfg: Cfg) -> List[Dict]:
    img_root = Path(cfg.domain_root) / cfg.image_dir
    msk_root = Path(cfg.domain_root) / cfg.mask_dir

    records = []
    for img_path in sorted(img_root.glob("*.mha")):
        stem = img_path.stem
        msk_path = msk_root / f"{stem}.mha"
        if not msk_path.exists():
            continue

        patient_id = parse_patient_id_from_stem(stem)
        records.append({
            "patient_id": patient_id,
            "stem": stem,
            "img_path": str(img_path),
            "mask_path": str(msk_path),
        })
    return records


def split_patients(records: List[Dict], n_train_patients: int, split_seed: int) -> Tuple[List[str], List[str]]:
    patient_ids = sorted(list({r["patient_id"] for r in records}))
    rng = random.Random(split_seed)
    rng.shuffle(patient_ids)
    train_patients = sorted(patient_ids[:n_train_patients])
    heldout_patients = sorted(patient_ids[n_train_patients:])
    return train_patients, heldout_patients


def split_inner_and_hard_outer(heldout_patients: List[str], hard_outer_ids: Tuple[str, ...]) -> Tuple[List[str], List[str], List[str]]:
    hard_outer_set = set(hard_outer_ids)
    hard_outer_patients = sorted([pid for pid in heldout_patients if pid in hard_outer_set])
    inner_patients = sorted([pid for pid in heldout_patients if pid not in hard_outer_set])
    missing = sorted([pid for pid in hard_outer_ids if pid not in heldout_patients])
    return inner_patients, hard_outer_patients, missing


def load_middle_sagittal_slices_from_volume(record: Dict, cfg: Cfg) -> List[Dict]:
    img = sitk.ReadImage(record["img_path"])
    msk = sitk.ReadImage(record["mask_path"])

    img_zyx = sitk.GetArrayFromImage(img).astype(np.float32)   # [Z, Y, X]
    msk_zyx = sitk.GetArrayFromImage(msk).astype(np.int32)     # [Z, Y, X]

    sagittal_indices = select_middle_sagittal_indices(msk_zyx, num_slices=cfg.num_middle_slices)

    out = []
    for x_idx in sagittal_indices:
        img2d = img_zyx[:, :, x_idx]
        msk2d = msk_zyx[:, :, x_idx]

        msk_bin = binarize_mask(msk2d, cfg.fg_labels)
        img2d = normalize_slice_to_01(img2d, cfg.input_clip_min, cfg.input_clip_max)

        if cfg.resize_hw is not None:
            img2d = resize_np_2d(img2d, cfg.resize_hw, is_mask=False)
            msk_bin = resize_np_2d(msk_bin, cfg.resize_hw, is_mask=True)
            msk_bin = (msk_bin > 0.5).astype(np.float32)

        out.append({
            "patient_id": record["patient_id"],
            "stem": record["stem"],
            "sagittal_x_index": int(x_idx),
            "img": img2d.astype(np.float32),
            "mask": msk_bin.astype(np.float32),
        })
    return out


def build_slice_records(records: List[Dict], kept_patient_ids: List[str], cfg: Cfg) -> List[Dict]:
    out = []
    keep_set = set(kept_patient_ids)
    for r in records:
        if r["patient_id"] not in keep_set:
            continue
        out.extend(load_middle_sagittal_slices_from_volume(r, cfg))
    return out


class SpiderMiddleSliceDataset(Dataset):
    def __init__(self, slice_records: List[Dict]):
        self.items = slice_records

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_t = torch.from_numpy(item["img"])[None, ...].float()
        msk_t = torch.from_numpy(item["mask"])[None, ...].float()
        meta = {
            "patient_id": item["patient_id"],
            "stem": item["stem"],
            "sagittal_x_index": item["sagittal_x_index"],
        }
        return img_t, msk_t, meta


# =========================================================
# Model
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: Optional[int] = None):
        super().__init__()
        inter = inter_ch if inter_ch is not None else max(min(g_ch, x_ch) // 2, 1)
        self.g_proj = nn.Conv2d(g_ch, inter, kernel_size=1, bias=True)
        self.x_proj = nn.Conv2d(x_ch, inter, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.psi(self.g_proj(g) + self.x_proj(x))
        return x * alpha, alpha


class BaseSegNet(nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.model_type = model_type

    def collect_feature_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        out = dict(kwargs)
        if "bottleneck" in out and "z" not in out:
            out["z"] = out["bottleneck"]
        if "dec1" in out and "feat" not in out:
            out["feat"] = out["dec1"]
        return out


class UNet3Level(BaseSegNet):
    def __init__(self, base_ch: int = 16, latent_ch: int = 64):
        super().__init__(model_type="unet")
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.enc1 = ConvBlock(1, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c3, latent_ch)

        self.up3 = nn.ConvTranspose2d(latent_ch, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c2)
        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        logits = self.seg_head(dec1)

        if return_features:
            return {
                "logits": logits,
                "features": self.collect_feature_dict(
                    enc1=enc1,
                    enc2=enc2,
                    enc3=enc3,
                    bottleneck=bottleneck,
                    dec3=dec3,
                    dec2=dec2,
                    dec1=dec1,
                ),
            }
        return logits


class AttentionUNet3Level(BaseSegNet):
    def __init__(self, base_ch: int = 16, latent_ch: int = 64):
        super().__init__(model_type="attention_unet")
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.enc1 = ConvBlock(1, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c3, latent_ch)

        self.up3 = nn.ConvTranspose2d(latent_ch, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(g_ch=c3, x_ch=c3)
        self.dec3 = ConvBlock(c3 + c3, c2)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(g_ch=c2, x_ch=c2)
        self.dec2 = ConvBlock(c2 + c2, c1)

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(g_ch=c1, x_ch=c1)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        up3 = self.up3(bottleneck)
        gated3, att3 = self.att3(up3, enc3)
        dec3 = self.dec3(torch.cat([up3, gated3], dim=1))

        up2 = self.up2(dec3)
        gated2, att2 = self.att2(up2, enc2)
        dec2 = self.dec2(torch.cat([up2, gated2], dim=1))

        up1 = self.up1(dec2)
        gated1, att1 = self.att1(up1, enc1)
        dec1 = self.dec1(torch.cat([up1, gated1], dim=1))

        logits = self.seg_head(dec1)

        if return_features:
            return {
                "logits": logits,
                "features": self.collect_feature_dict(
                    enc1=enc1,
                    enc2=enc2,
                    enc3=enc3,
                    bottleneck=bottleneck,
                    dec3=dec3,
                    dec2=dec2,
                    dec1=dec1,
                    att3=att3,
                    att2=att2,
                    att1=att1,
                ),
            }
        return logits


def build_model(variant: Variant) -> nn.Module:
    if variant.model_type == "unet":
        return UNet3Level(base_ch=variant.base_ch, latent_ch=variant.latent_ch)
    if variant.model_type == "attention_unet":
        return AttentionUNet3Level(base_ch=variant.base_ch, latent_ch=variant.latent_ch)
    raise ValueError(f"Unsupported model_type: {variant.model_type}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


# =========================================================
# Train / Eval
# =========================================================
def use_dice_loss() -> bool:
    return cfg.lambda_dice > 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    n_seen = 0

    grad_norm_values = []

    _use_dice = use_dice_loss()

    for img, mask, _meta in loader:
        img = img.to(device)
        mask = mask.to(device)

        seg_logits = model(img)
        loss_bce = F.binary_cross_entropy_with_logits(seg_logits, mask)

        if _use_dice:
            loss_dice = dice_loss_from_logits(seg_logits, mask)
            loss = cfg.lambda_bce * loss_bce + cfg.lambda_dice * loss_dice
            loss_dice_value = float(loss_dice.item())
        else:
            loss = cfg.lambda_bce * loss_bce
            loss_dice_value = 0.0

        optimizer.zero_grad()
        loss.backward()

        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += float(torch.sum(p.grad.detach() ** 2).item())
        grad_norm_global = grad_sq ** 0.5
        grad_norm_values.append(grad_norm_global)

        optimizer.step()

        bs = img.shape[0]
        n_seen += bs
        total_loss += float(loss.item()) * bs
        total_bce += float(loss_bce.item()) * bs
        total_dice += loss_dice_value * bs

    param_sq = 0.0
    for p in model.parameters():
        param_sq += float(torch.sum(p.detach() ** 2).item())
    param_l2 = param_sq ** 0.5

    denom = max(n_seen, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "dice_loss": total_dice / denom,
        "grad_norm_global": float(np.mean(grad_norm_values)) if grad_norm_values else 0.0,
        "param_l2": float(param_l2),
    }


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    n_seen = 0
    dices = []
    ious = []
    pred_fg_ratios = []

    _use_dice = use_dice_loss()

    for img, mask, _meta in loader:
        img = img.to(device)
        mask = mask.to(device)

        seg_logits = model(img)
        loss_bce = F.binary_cross_entropy_with_logits(seg_logits, mask)

        if _use_dice:
            loss_dice = dice_loss_from_logits(seg_logits, mask)
            loss = cfg.lambda_bce * loss_bce + cfg.lambda_dice * loss_dice
            loss_dice_value = float(loss_dice.item())
        else:
            loss = cfg.lambda_bce * loss_bce
            loss_dice_value = 0.0

        prob = torch.sigmoid(seg_logits)
        pred = (prob >= cfg.pred_thr).float()

        for b in range(img.shape[0]):
            pred_np = pred[b, 0].cpu().numpy()
            gt_np = mask[b, 0].cpu().numpy()
            iou, dice = iou_and_dice(pred_np, gt_np)
            ious.append(iou)
            dices.append(dice)
            pred_fg_ratios.append(float(np.mean(pred_np > 0.5)))

        bs = img.shape[0]
        n_seen += bs
        total_loss += float(loss.item()) * bs
        total_bce += float(loss_bce.item()) * bs
        total_dice_loss += loss_dice_value * bs

    denom = max(n_seen, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "dice_loss": total_dice_loss / denom,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "std_dice": float(np.std(dices)) if dices else 0.0,
        "pred_fg_ratio": float(np.mean(pred_fg_ratios)) if pred_fg_ratios else 0.0,
        "n_slices": len(dices),
    }


@torch.no_grad()
def infer_one(model: nn.Module, img: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    img = img.to(device)
    out = model(img, return_features=True)
    seg_logits = out["logits"]
    prob = torch.sigmoid(seg_logits)
    pred = (prob >= cfg.pred_thr).float()
    features_cpu = {k: v.cpu() for k, v in out["features"].items()}
    ret = {
        "prob": prob.cpu(),
        "pred": pred.cpu(),
        **features_cpu,
    }
    if "z" not in ret and "bottleneck" in ret:
        ret["z"] = ret["bottleneck"]
    if "feat" not in ret and "dec1" in ret:
        ret["feat"] = ret["dec1"]
    return ret


# =========================================================
# Visualization
# =========================================================
def save_case_vis(case_name: str, img: np.ndarray, gt: np.ndarray, prob: np.ndarray, pred: np.ndarray, meta: Dict, out_path: str):
    iou, dice = iou_and_dice(pred, gt)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(prob, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pred prob")
    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(pred == 0, pred), alpha=0.50)
    axes[3].set_title(f"Pred mask\nDice={dice:.3f}")

    for ax in axes:
        ax.axis("off")

    title = (
        f"{case_name} | pid={meta.get('patient_id', '?')} | "
        f"stem={meta.get('stem', '?')} | sagittal_x={meta.get('sagittal_x_index', '?')} | "
        f"IoU={iou:.3f} Dice={dice:.3f}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_seed_curves(history: List[Dict[str, float]], out_path: str):
    xs = np.arange(1, len(history) + 1)

    train_loss = np.array([h["train_loss"] for h in history], dtype=np.float32)
    inner_loss = np.array([h["inner_loss"] for h in history], dtype=np.float32)
    hard_loss = np.array([h["hard_outer_loss"] for h in history], dtype=np.float32)

    train_dice = np.array([h["train_dice"] for h in history], dtype=np.float32)
    inner_dice = np.array([h["inner_dice"] for h in history], dtype=np.float32)
    hard_dice = np.array([h["hard_outer_dice"] for h in history], dtype=np.float32)

    train_iou = np.array([h["train_iou"] for h in history], dtype=np.float32)
    inner_iou = np.array([h["inner_iou"] for h in history], dtype=np.float32)
    hard_iou = np.array([h["hard_outer_iou"] for h in history], dtype=np.float32)

    gap_inner_dice = train_dice - inner_dice
    gap_hard_dice = train_dice - hard_dice
    gap_inner_loss = inner_loss - train_loss
    gap_hard_loss = hard_loss - train_loss

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(xs, train_dice, label="train_dice")
    axes[0, 0].plot(xs, inner_dice, label="inner_dice")
    axes[0, 0].plot(xs, hard_dice, label="hard_outer_dice")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Dice")
    axes[0, 0].set_title("Train / Inner / Hard-Outer Dice")
    axes[0, 0].legend()

    axes[0, 1].plot(xs, train_loss, label="train_loss")
    axes[0, 1].plot(xs, inner_loss, label="inner_loss")
    axes[0, 1].plot(xs, hard_loss, label="hard_outer_loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Train / Inner / Hard-Outer Loss")
    axes[0, 1].legend()

    axes[1, 0].plot(xs, train_iou, label="train_iou")
    axes[1, 0].plot(xs, inner_iou, label="inner_iou")
    axes[1, 0].plot(xs, hard_iou, label="hard_outer_iou")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].set_title("Train / Inner / Hard-Outer IoU")
    axes[1, 0].legend()

    axes[1, 1].plot(xs, gap_inner_dice, label="train_dice - inner_dice")
    axes[1, 1].plot(xs, gap_hard_dice, label="train_dice - hard_outer_dice")
    axes[1, 1].plot(xs, gap_inner_loss, label="inner_loss - train_loss")
    axes[1, 1].plot(xs, gap_hard_loss, label="hard_outer_loss - train_loss")
    axes[1, 1].axhline(0.0, color="k", linewidth=0.8)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_title("Generalization Gaps")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all_seeds_overlay_curves(histories_by_seed: Dict[int, List[Dict[str, float]]], out_path: str):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))

    for seed, history in sorted(histories_by_seed.items()):
        xs = np.arange(1, len(history) + 1)
        train_dice = np.array([h["train_dice"] for h in history], dtype=np.float32)
        train_loss = np.array([h["train_loss"] for h in history], dtype=np.float32)
        inner_dice = np.array([h["inner_dice"] for h in history], dtype=np.float32)
        inner_loss = np.array([h["inner_loss"] for h in history], dtype=np.float32)
        hard_dice = np.array([h["hard_outer_dice"] for h in history], dtype=np.float32)
        hard_loss = np.array([h["hard_outer_loss"] for h in history], dtype=np.float32)

        axes[0, 0].plot(xs, train_dice, alpha=0.8, label=f"seed{seed}")
        axes[0, 1].plot(xs, train_loss, alpha=0.8, label=f"seed{seed}")
        axes[1, 0].plot(xs, inner_dice, alpha=0.8, label=f"seed{seed}")
        axes[1, 1].plot(xs, inner_loss, alpha=0.8, label=f"seed{seed}")
        axes[2, 0].plot(xs, hard_dice, alpha=0.8, label=f"seed{seed}")
        axes[2, 1].plot(xs, hard_loss, alpha=0.8, label=f"seed{seed}")

    titles = [
        "Train Dice (all seeds)", "Train Loss (all seeds)",
        "Inner Dice (all seeds)", "Inner Loss (all seeds)",
        "Hard-Outer Dice (all seeds)", "Hard-Outer Loss (all seeds)",
    ]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all_seeds_mean_std_curves(histories_by_seed: Dict[int, List[Dict[str, float]]], out_path: str):
    seeds = sorted(histories_by_seed.keys())
    if not seeds:
        return

    max_epochs = max(len(histories_by_seed[s]) for s in seeds)
    if max_epochs == 0:
        return

    def stack(key: str) -> np.ndarray:
        arr = np.full((len(seeds), max_epochs), np.nan, dtype=np.float32)
        for i, seed in enumerate(seeds):
            history = histories_by_seed[seed]
            vals = []
            for h in history:
                v = h.get(key, np.nan)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            if vals:
                arr[i, :len(vals)] = np.asarray(vals, dtype=np.float32)
        return arr

    xs = np.arange(1, max_epochs + 1, dtype=np.int32)

    tr_dice = stack("train_dice")
    in_dice = stack("inner_dice")
    ho_dice = stack("hard_outer_dice")
    tr_loss = stack("train_loss")
    in_loss = stack("inner_loss")
    ho_loss = stack("hard_outer_loss")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))

    plot_items = [
        (tr_dice, "train_dice", axes[0, 0], "Train Dice (mean ± std)"),
        (tr_loss, "train_loss", axes[0, 1], "Train Loss (mean ± std)"),
        (in_dice, "inner_dice", axes[1, 0], "Inner Dice (mean ± std)"),
        (in_loss, "inner_loss", axes[1, 1], "Inner Loss (mean ± std)"),
        (ho_dice, "hard_outer_dice", axes[2, 0], "Hard-Outer Dice (mean ± std)"),
        (ho_loss, "hard_outer_loss", axes[2, 1], "Hard-Outer Loss (mean ± std)"),
    ]

    for arr, label, ax, title in plot_items:
        valid_counts = np.sum(~np.isnan(arr), axis=0)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)

        valid_mask = valid_counts > 0
        if np.any(valid_mask):
            ax.plot(xs[valid_mask], mu[valid_mask], label=label)
            ax.fill_between(
                xs[valid_mask],
                (mu - sd)[valid_mask],
                (mu + sd)[valid_mask],
                alpha=0.20,
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================
# Final eval on train / inner / hard-outer
# =========================================================
@torch.no_grad()
def evaluate_and_optionally_save_split_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    out_dir: Path,
    seed: int,
    split_name: str,
    checkpoint_tag: str,
    save_details: bool = True,
):
    split_dir = out_dir / f"{checkpoint_tag}" / f"eval_{split_name}"
    ensure_dir(split_dir)
    save_split_vis = cfg.save_all_eval_vis or (split_name == "test" and cfg.save_all_test_vis)
    vis_dir = split_dir / f"vis_{split_name}_all"
    if save_split_vis:
        ensure_dir(vis_dir)

    slice_rows = []
    patient_to_dices = defaultdict(list)
    patient_to_ious = defaultdict(list)

    latent_case_rows = []
    targeted_channel_rows = []
    layer_sparsity_rows = []

    target_pid_set = set(cfg.targeted_patient_ids)

    for i in range(len(dataset)):
        img_t, gt_t, meta = dataset[i]
        out = infer_one(model, img_t.unsqueeze(0), device)

        img = img_t.squeeze(0).numpy().astype(np.float32)
        gt = gt_t.squeeze(0).numpy().astype(np.float32)
        prob = out["prob"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        pred = out["pred"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        feature_np = extract_case_features_from_infer_output(out)
        z_np = feature_np.get("z", np.zeros((1, 1, 1), dtype=np.float32))
        feat_np = feature_np.get("feat", np.zeros((1, 1, 1), dtype=np.float32))

        iou, dice = iou_and_dice(pred, gt)
        row = {
            "seed": seed,
            "split": split_name,
            "checkpoint_tag": checkpoint_tag,
            "case_id": i,
            "patient_id": meta["patient_id"],
            "stem": meta["stem"],
            "sagittal_x_index": meta["sagittal_x_index"],
            "iou": iou,
            "dice": dice,
        }
        slice_rows.append(row)
        patient_to_dices[meta["patient_id"]].append(dice)
        patient_to_ious[meta["patient_id"]].append(iou)

        gt_mask = gt > 0.5
        bg_mask = ~gt_mask
        case_row = {
            "seed": seed,
            "split": split_name,
            "checkpoint_tag": checkpoint_tag,
            "case_id": i,
            "patient_id": meta["patient_id"],
            "stem": meta["stem"],
            "sagittal_x_index": meta["sagittal_x_index"],
            "dice": dice,
            "iou": iou,
            "gt_fg_ratio": float(np.mean(gt)),
            "pred_fg_ratio": float(np.mean(pred)),
            "prob_mean": float(np.mean(prob)),
            "prob_std": float(np.std(prob)),
            "prob_on_gt_fg": float(np.mean(prob[gt_mask])) if np.any(gt_mask) else 0.0,
            "prob_on_gt_bg": float(np.mean(prob[bg_mask])) if np.any(bg_mask) else 0.0,
            "prob_gap_fg_minus_bg": (
                float(np.mean(prob[gt_mask])) - float(np.mean(prob[bg_mask]))
            ) if np.any(gt_mask) and np.any(bg_mask) else 0.0,
        }
        case_row.update(summarize_array_stats(z_np, "z"))
        case_row.update(summarize_array_stats(feat_np, "feat"))
        case_row.update(summarize_feature_tensors(feature_np))

        # --- region stability analysis ---
        gt_bin = (gt > 0.5).astype(np.uint8)

        coarse_rows = {}
        main_lcc = np.zeros_like(gt_bin, dtype=np.uint8)
        main_bbox_mask = np.zeros_like(gt_bin, dtype=np.uint8)

        for thr in cfg.coarse_thresholds:
            coarse = (prob >= thr).astype(np.uint8)
            lcc = largest_connected_component(coarse)
            bbox = mask_bbox(lcc)
            bbox_mask = bbox_to_mask(bbox, coarse.shape)

            recall_t = float((coarse & gt_bin).sum() / (gt_bin.sum() + 1e-8))
            lcc_cover_t = float((lcc & gt_bin).sum() / (gt_bin.sum() + 1e-8))
            bbox_cover_t = float((bbox_mask & gt_bin).sum() / (gt_bin.sum() + 1e-8))
            com_dist_t = normalized_com_distance(lcc, gt_bin)

            tag = int(thr * 100)
            coarse_rows[f"recall_t{tag:02d}"] = recall_t
            coarse_rows[f"roi_cover_lcc_t{tag:02d}"] = lcc_cover_t
            coarse_rows[f"roi_cover_bbox_t{tag:02d}"] = bbox_cover_t
            coarse_rows[f"com_dist_t{tag:02d}"] = com_dist_t

            if abs(thr - cfg.main_region_thr) < 1e-8:
                main_lcc = lcc
                main_bbox_mask = bbox_mask

        failure_type = classify_failure_type(
            coarse_rows.get("recall_t30", 0.0),
            coarse_rows.get("roi_cover_bbox_t30", 0.0),
        )

        case_row.update(coarse_rows)
        case_row["failure_type"] = failure_type

        latent_case_rows.append(case_row)

        if meta["patient_id"] in target_pid_set:
            z_ch_absmean = np.mean(np.abs(z_np), axis=(1, 2))
            z_ch_mean = np.mean(z_np, axis=(1, 2))
            z_ch_std = np.std(z_np, axis=(1, 2))

            order = np.argsort(-z_ch_absmean)
            topk = min(cfg.targeted_channel_topk, len(order))
            topk_set = {int(order[r]) for r in range(topk)}

            for c in range(len(z_ch_absmean)):
                targeted_channel_rows.append({
                    "seed": seed,
                    "split": split_name,
                    "checkpoint_tag": checkpoint_tag,
                    "case_id": i,
                    "patient_id": meta["patient_id"],
                    "stem": meta["stem"],
                    "sagittal_x_index": meta["sagittal_x_index"],
                    "dice": dice,
                    "iou": iou,
                    "channel_index": int(c),
                    "z_ch_absmean": float(z_ch_absmean[c]),
                    "z_ch_mean": float(z_ch_mean[c]),
                    "z_ch_std": float(z_ch_std[c]),
                    "is_topk_channel": int(c in topk_set),
                    "channel_rank_by_absmean": int(np.where(order == c)[0][0] + 1),
                })

        # --- layer sparsity analysis ---
        roi_mask = dilate_binary(main_bbox_mask.astype(np.uint8), radius=cfg.fg_roi_dilate_px)

        for layer_name in cfg.sparsity_layers:
            if layer_name not in feature_np:
                continue
            rows_layer = compute_layer_sparsity_stats(
                feat_chw=feature_np[layer_name],
                fg_mask_hw=gt_bin,
                roi_mask_hw=roi_mask,
                layer_name=layer_name,
            )
            for rr in rows_layer:
                rr.update({
                    "seed": seed,
                    "split": split_name,
                    "checkpoint_tag": checkpoint_tag,
                    "patient_id": meta["patient_id"],
                    "stem": meta["stem"],
                    "sagittal_x_index": meta["sagittal_x_index"],
                    "iou": iou,
                    "dice": dice,
                    "failure_type": failure_type,
                })
                layer_sparsity_rows.append(rr)

        if save_split_vis:
            save_case_vis(
                case_name=f"case_{i:03d}",
                img=img,
                gt=gt,
                prob=prob,
                pred=pred,
                meta=meta,
                out_path=str(vis_dir / f"{checkpoint_tag}_case_{i:03d}_pid{meta['patient_id']}_x{meta['sagittal_x_index']}.png"),
            )

    patient_rows = []
    for pid in sorted(patient_to_dices.keys()):
        patient_rows.append({
            "seed": seed,
            "split": split_name,
            "checkpoint_tag": checkpoint_tag,
            "patient_id": pid,
            "n_slices": len(patient_to_dices[pid]),
            "mean_dice": float(np.mean(patient_to_dices[pid])),
            "std_dice": float(np.std(patient_to_dices[pid])),
            "mean_iou": float(np.mean(patient_to_ious[pid])),
            "std_iou": float(np.std(patient_to_ious[pid])),
        })

    patient_rows = sorted(patient_rows, key=lambda x: x["mean_dice"])
    slice_rows_sorted = sorted(slice_rows, key=lambda x: x["dice"])

    latent_patient_rows = aggregate_metric_rows(
        latent_case_rows,
        group_keys=("seed", "split", "checkpoint_tag", "patient_id"),
        sort_key="dice_mean",
    )

    if save_details:
        write_log_csv(slice_rows, split_dir / f"{split_name}_{checkpoint_tag}_slice_metrics.csv")
        write_log_csv(patient_rows, split_dir / f"{split_name}_{checkpoint_tag}_patient_metrics.csv")
        write_log_csv(patient_rows[:cfg.topk_patients_report], split_dir / f"{split_name}_{checkpoint_tag}_hard_negative_patients_topk.csv")
        write_log_csv(slice_rows_sorted[:cfg.topk_slices_report], split_dir / f"{split_name}_{checkpoint_tag}_hard_negative_slices_topk.csv")
        write_log_csv(latent_case_rows, split_dir / f"{split_name}_{checkpoint_tag}_latent_case_stats.csv")
        write_log_csv(latent_patient_rows, split_dir / f"{split_name}_{checkpoint_tag}_latent_patient_stats.csv")
        write_log_csv(targeted_channel_rows, split_dir / f"{split_name}_{checkpoint_tag}_targeted_channel_signatures.csv")
        write_log_csv(layer_sparsity_rows, split_dir / f"{split_name}_{checkpoint_tag}_layer_sparsity_stats.csv")

    mean_iou = float(np.mean([r["iou"] for r in slice_rows])) if slice_rows else 0.0
    std_iou = float(np.std([r["iou"] for r in slice_rows])) if slice_rows else 0.0
    mean_dice = float(np.mean([r["dice"] for r in slice_rows])) if slice_rows else 0.0
    std_dice = float(np.std([r["dice"] for r in slice_rows])) if slice_rows else 0.0
    mean_patient_dice = float(np.mean([r["mean_dice"] for r in patient_rows])) if patient_rows else 0.0
    std_patient_dice = float(np.std([r["mean_dice"] for r in patient_rows])) if patient_rows else 0.0

    summary_row = {
        "seed": seed,
        "split": split_name,
        "checkpoint_tag": checkpoint_tag,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_patient_dice": mean_patient_dice,
        "std_patient_dice": std_patient_dice,
        "n_slices": len(dataset),
        "n_patients": len(patient_rows),
    }

    with open(split_dir / f"{split_name}_{checkpoint_tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_row, f, ensure_ascii=False, indent=2)

    return summary_row




def summarize_feature_tensors(feature_map: Dict[str, np.ndarray]) -> Dict[str, float]:
    stats = {}
    for name, arr in feature_map.items():
        stats.update(summarize_array_stats(arr, name))
    return stats


def extract_case_features_from_infer_output(out: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    exclude_keys = {"prob", "pred"}
    feats = {}
    for k, v in out.items():
        if k in exclude_keys:
            continue
        if isinstance(v, torch.Tensor):
            feats[k] = v.squeeze(0).numpy().astype(np.float32)
    return feats

# =========================================================
# Aggregate across seeds / variants
# =========================================================
# Aggregate across seeds / variants
# =========================================================
def maybe_read_csv_rows(path: Path) -> List[Dict]:
    if (not path.exists()) or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if "empty" not in r]


def normalize_pid(x) -> str:
    """
    Normalize patient_id to a canonical string like '190'.
    Handles '190', 190, 190.0, np types, etc.
    """
    if x is None:
        return ""
    try:
        fx = float(x)
        if fx.is_integer():
            return str(int(fx))
        return str(fx)
    except Exception:
        return str(x).strip()


def aggregate_split_reports_across_seeds(
    variant_root: Path,
    split_names: Tuple[str, ...],
    checkpoint_tags: Tuple[str, ...],
):
    all_patient_rows = []
    all_slice_rows = []
    all_latent_case_rows = []
    all_latent_patient_rows = []
    all_targeted_channel_rows = []
    all_layer_sparsity_rows = []

    string_keys = {"patient_id", "stem", "split", "checkpoint_tag", "case_key"}

    def _convert_row_values(r: Dict) -> Dict:
        out = {}
        for k, v in r.items():
            if v is None:
                out[k] = v
                continue

            # 强制这些字段保持字符串，不做 float 化
            if k in string_keys:
                if k == "patient_id":
                    out[k] = normalize_pid(v)
                else:
                    out[k] = str(v)
                continue

            try:
                fv = float(v)
                if fv.is_integer():
                    out[k] = int(fv)
                else:
                    out[k] = fv
            except Exception:
                out[k] = v
        return out

    # -----------------------------------------------------
    # 1) 读取所有 seed / split / checkpoint 的原始 CSV
    # -----------------------------------------------------
    for seed in cfg.seeds:
        seed_dir = variant_root / f"seed_{seed}"
        for checkpoint_tag in checkpoint_tags:
            for split_name in split_names:
                split_dir = seed_dir / checkpoint_tag / f"eval_{split_name}"

                pat_csv = split_dir / f"{split_name}_{checkpoint_tag}_patient_metrics.csv"
                sl_csv = split_dir / f"{split_name}_{checkpoint_tag}_slice_metrics.csv"
                latent_case_csv = split_dir / f"{split_name}_{checkpoint_tag}_latent_case_stats.csv"
                latent_patient_csv = split_dir / f"{split_name}_{checkpoint_tag}_latent_patient_stats.csv"
                targeted_channel_csv = split_dir / f"{split_name}_{checkpoint_tag}_targeted_channel_signatures.csv"

                for r in maybe_read_csv_rows(pat_csv):
                    rr = _convert_row_values(r)
                    all_patient_rows.append({
                        "seed": int(rr["seed"]),
                        "split": rr["split"],
                        "checkpoint_tag": rr["checkpoint_tag"],
                        "patient_id": normalize_pid(rr["patient_id"]),
                        "mean_dice": float(rr["mean_dice"]),
                        "mean_iou": float(rr["mean_iou"]),
                        "n_slices": int(rr["n_slices"]),
                    })

                for r in maybe_read_csv_rows(sl_csv):
                    rr = _convert_row_values(r)
                    pid = normalize_pid(rr["patient_id"])
                    case_key = f"{pid}__{rr['stem']}__x{rr['sagittal_x_index']}"
                    all_slice_rows.append({
                        "seed": int(rr["seed"]),
                        "split": rr["split"],
                        "checkpoint_tag": rr["checkpoint_tag"],
                        "case_key": case_key,
                        "patient_id": pid,
                        "stem": rr["stem"],
                        "sagittal_x_index": int(rr["sagittal_x_index"]),
                        "dice": float(rr["dice"]),
                        "iou": float(rr["iou"]),
                    })

                for r in maybe_read_csv_rows(latent_case_csv):
                    rr = _convert_row_values(r)
                    rr["patient_id"] = normalize_pid(rr.get("patient_id"))
                    all_latent_case_rows.append(rr)

                for r in maybe_read_csv_rows(latent_patient_csv):
                    rr = _convert_row_values(r)
                    rr["patient_id"] = normalize_pid(rr.get("patient_id"))
                    all_latent_patient_rows.append(rr)

                for r in maybe_read_csv_rows(targeted_channel_csv):
                    rr = _convert_row_values(r)
                    rr["patient_id"] = normalize_pid(rr.get("patient_id"))
                    all_targeted_channel_rows.append(rr)

                layer_sparsity_csv = split_dir / f"{split_name}_{checkpoint_tag}_layer_sparsity_stats.csv"
                for r in maybe_read_csv_rows(layer_sparsity_csv):
                    rr = _convert_row_values(r)
                    rr["patient_id"] = normalize_pid(rr.get("patient_id"))
                    all_layer_sparsity_rows.append(rr)

    # 原始汇总先写出
    write_log_csv(all_patient_rows, variant_root / "all_seeds_patient_rows.csv")
    write_log_csv(all_slice_rows, variant_root / "all_seeds_slice_rows.csv")
    write_log_csv(all_latent_case_rows, variant_root / "all_seeds_latent_case_rows.csv")
    write_log_csv(all_latent_patient_rows, variant_root / "all_seeds_latent_patient_rows.csv")
    write_log_csv(all_targeted_channel_rows, variant_root / "all_seeds_targeted_channel_rows.csv")

    # -----------------------------------------------------
    # 2) hard-negative patient / slice 汇总
    # -----------------------------------------------------
    patient_summary_rows = aggregate_metric_rows(
        all_patient_rows,
        group_keys=("checkpoint_tag", "split", "patient_id"),
        sort_key="mean_dice_mean",
    )
    patient_summary_rows = sorted(
        patient_summary_rows,
        key=lambda x: (x["checkpoint_tag"], x["split"], x["mean_dice_mean"])
    )

    slice_summary_rows = aggregate_metric_rows(
        all_slice_rows,
        group_keys=("checkpoint_tag", "split", "case_key", "patient_id", "stem", "sagittal_x_index"),
        sort_key="dice_mean",
    )
    slice_summary_rows = sorted(
        slice_summary_rows,
        key=lambda x: (x["checkpoint_tag"], x["split"], x["dice_mean"])
    )

    write_log_csv(patient_summary_rows, variant_root / "hard_negative_patients_across_5seeds.csv")
    write_log_csv(slice_summary_rows, variant_root / "hard_negative_slices_across_5seeds.csv")

    # -----------------------------------------------------
    # 3) latent patient 汇总
    # -----------------------------------------------------
    latent_patient_summary_rows = aggregate_metric_rows(
        all_latent_patient_rows,
        group_keys=("checkpoint_tag", "split", "patient_id"),
        sort_key="dice_mean_mean",
    )
    write_log_csv(latent_patient_summary_rows, variant_root / "latent_patient_summary_across_5seeds.csv")

    # -----------------------------------------------------
    # 4) targeted checkpoint compare（修复 pid 类型问题）
    # -----------------------------------------------------
    target_set = {normalize_pid(x) for x in cfg.targeted_patient_ids}

    lp_map = {
        (str(r["checkpoint_tag"]), str(r["split"]), normalize_pid(r["patient_id"])): r
        for r in latent_patient_summary_rows
    }

    targeted_compare_rows = []
    for split_name in split_names:
        split_name = str(split_name)
        for pid in sorted(target_set):
            bi = lp_map.get(("best_inner", split_name, pid))
            bh = lp_map.get(("best_hard_outer", split_name, pid))
            if bi is None or bh is None:
                continue

            targeted_compare_rows.append({
                "split": split_name,
                "patient_id": pid,
                "best_inner_dice_mean": bi.get("dice_mean_mean", np.nan),
                "best_hard_outer_dice_mean": bh.get("dice_mean_mean", np.nan),
                "best_inner_iou_mean": bi.get("iou_mean_mean", np.nan),
                "best_hard_outer_iou_mean": bh.get("iou_mean_mean", np.nan),
                "best_inner_z_abs_mean": bi.get("z_abs_mean_mean", np.nan),
                "best_hard_outer_z_abs_mean": bh.get("z_abs_mean_mean", np.nan),
                "best_inner_z_std_mean": bi.get("z_std_mean", np.nan),
                "best_hard_outer_z_std_mean": bh.get("z_std_mean", np.nan),
                "best_inner_z_zero_ratio_mean": bi.get("z_zero_ratio_mean", np.nan),
                "best_hard_outer_z_zero_ratio_mean": bh.get("z_zero_ratio_mean", np.nan),
                "best_inner_z_max_abs_mean": bi.get("z_max_abs_mean", np.nan),
                "best_hard_outer_z_max_abs_mean": bh.get("z_max_abs_mean", np.nan),
                "best_inner_prob_gap_mean": bi.get("prob_gap_fg_minus_bg_mean", np.nan),
                "best_hard_outer_prob_gap_mean": bh.get("prob_gap_fg_minus_bg_mean", np.nan),
                "delta_hard_minus_inner_dice": float(
                    bh.get("dice_mean_mean", np.nan) - bi.get("dice_mean_mean", np.nan)
                ),
                "delta_hard_minus_inner_z_abs_mean": float(
                    bh.get("z_abs_mean_mean", np.nan) - bi.get("z_abs_mean_mean", np.nan)
                ),
                "delta_hard_minus_inner_z_zero_ratio": float(
                    bh.get("z_zero_ratio_mean", np.nan) - bi.get("z_zero_ratio_mean", np.nan)
                ),
                "delta_hard_minus_inner_prob_gap": float(
                    bh.get("prob_gap_fg_minus_bg_mean", np.nan) - bi.get("prob_gap_fg_minus_bg_mean", np.nan)
                ),
            })

    write_log_csv(targeted_compare_rows, variant_root / "targeted_checkpoint_compare_across_5seeds.csv")

    # -----------------------------------------------------
    # 5) targeted channel summary（修复 pid 类型问题）
    # -----------------------------------------------------
    filtered_targeted_channel_rows = [
        r for r in all_targeted_channel_rows
        if normalize_pid(r.get("patient_id")) in target_set
    ]

    targeted_channel_summary_rows = aggregate_metric_rows(
        filtered_targeted_channel_rows,
        group_keys=("checkpoint_tag", "split", "patient_id", "channel_index"),
        sort_key="z_ch_absmean_mean",
    )
    write_log_csv(
        targeted_channel_summary_rows,
        variant_root / "targeted_channel_summary_across_5seeds.csv",
    )

    # -----------------------------------------------------
    # 5b) layer sparsity aggregation
    # -----------------------------------------------------
    write_log_csv(all_layer_sparsity_rows, variant_root / "all_seeds_layer_sparsity_rows.csv")

    layer_sparsity_summary_rows = aggregate_metric_rows(
        all_layer_sparsity_rows,
        group_keys=("checkpoint_tag", "split", "layer", "region_type"),
        sort_key="space_support_ratio_mean",
    )
    write_log_csv(
        layer_sparsity_summary_rows,
        variant_root / "layer_sparsity_summary_across_5seeds.csv",
    )

    for r in all_layer_sparsity_rows:
        r["difficulty_group"] = "hard" if float(r.get("dice", 0.0)) < 0.60 else "easy"

    layer_sparsity_by_diff_rows = aggregate_metric_rows(
        all_layer_sparsity_rows,
        group_keys=("checkpoint_tag", "split", "layer", "region_type", "difficulty_group"),
        sort_key="space_support_ratio_mean",
    )
    write_log_csv(
        layer_sparsity_by_diff_rows,
        variant_root / "layer_sparsity_easy_vs_hard_across_5seeds.csv",
    )

    # -----------------------------------------------------
    # 6) 简短文字报告
    # -----------------------------------------------------
    with open(variant_root / "global_report.txt", "w", encoding="utf-8") as f:
        f.write("Layered split aggregated report across 5 seeds\n")
        f.write(f"targeted_patient_ids = {list(cfg.targeted_patient_ids)}\n")
        f.write(f"n_all_patient_rows = {len(all_patient_rows)}\n")
        f.write(f"n_all_slice_rows = {len(all_slice_rows)}\n")
        f.write(f"n_all_latent_case_rows = {len(all_latent_case_rows)}\n")
        f.write(f"n_all_latent_patient_rows = {len(all_latent_patient_rows)}\n")
        f.write(f"n_all_targeted_channel_rows = {len(all_targeted_channel_rows)}\n")
        f.write(f"n_all_layer_sparsity_rows = {len(all_layer_sparsity_rows)}\n")
        f.write(f"n_targeted_compare_rows = {len(targeted_compare_rows)}\n")
        f.write(f"n_targeted_channel_summary_rows = {len(targeted_channel_summary_rows)}\n")
        f.write(f"n_layer_sparsity_summary_rows = {len(layer_sparsity_summary_rows)}\n")
        f.write(f"n_layer_sparsity_by_diff_rows = {len(layer_sparsity_by_diff_rows)}\n")

    # -----------------------------------------------------
    # 7) experiment report markdown
    # -----------------------------------------------------
    raw_results_path = variant_root / "raw_results.csv"
    raw_results = maybe_read_csv_rows(raw_results_path)

    report_path = variant_root / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment report: {variant_root.name}\n\n")
        f.write("## Core outputs\n")
        f.write("- raw_results.csv\n")
        f.write("- hard_negative_patients_across_5seeds.csv\n")
        f.write("- hard_negative_slices_across_5seeds.csv\n")
        f.write("- latent_patient_summary_across_5seeds.csv\n")
        f.write("- layer_sparsity_summary_across_5seeds.csv\n")
        f.write("- layer_sparsity_easy_vs_hard_across_5seeds.csv\n\n")

        if raw_results:
            f.write("## Raw results (first rows)\n\n")
            for row in raw_results[:12]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        f.write("\n## Notes\n")
        f.write("- hard_outer_sacrifice_dice_when_select_inner = best_hard_ckpt_hard_outer_dice - best_inner_ckpt_hard_outer_dice\n")
        f.write("- inner_sacrifice_dice_when_select_hard = best_inner_ckpt_inner_dice - best_hard_ckpt_inner_dice\n")
        f.write("- Higher hard_outer_sacrifice means selecting by inner costs more hard-outer performance.\n")
        f.write("- layer_sparsity_summary_across_5seeds.csv is used to check enc1/enc2/enc3/z spatial/channel sparsity.\n")
        f.write("- slice_metrics.csv now includes recall_t20/t30/t40, roi_cover_lcc, roi_cover_bbox, com_dist, failure_type.\n")


def summarize_variant_results(variant_root: Path):
    raw_path = variant_root / "raw_results.csv"
    rows = maybe_read_csv_rows(raw_path)
    if not rows:
        return

    # convert numeric fields
    numeric_fields = set()
    for r in rows:
        for k, v in r.items():
            try:
                float(v)
                numeric_fields.add(k)
            except Exception:
                pass

    summary_rows = []
    for key in sorted(numeric_fields):
        vals = [float(r[key]) for r in rows]
        summary_rows.append({
            "metric": key,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    write_log_csv(summary_rows, variant_root / "summary_mean_std.csv")


def aggregate_all_variants(root: Path, variant_names: List[str]):
    all_rows = []
    for variant_name in variant_names:
        variant_root = root / variant_name
        raw_path = variant_root / "raw_results.csv"
        for r in maybe_read_csv_rows(raw_path):
            rr = dict(r)
            rr["variant"] = variant_name
            all_rows.append(rr)

    write_log_csv(all_rows, root / "all_variants_seed_results.csv")

    # group by variant
    grouped = defaultdict(list)
    for r in all_rows:
        grouped[r["variant"]].append(r)

    summary_rows = []
    important_metrics = [
        "last_inner_mean_dice",
        "last_hard_outer_mean_dice",
        "best_inner_ckpt_inner_mean_dice",
        "best_inner_ckpt_hard_outer_mean_dice",
        "best_hard_ckpt_inner_mean_dice",
        "best_hard_ckpt_hard_outer_mean_dice",
        "hard_outer_sacrifice_dice_when_select_inner",
        "inner_sacrifice_dice_when_select_hard",
        "best_inner_epoch",
        "best_hard_epoch",
        "stop_epoch",
    ]
    for variant_name, rows in grouped.items():
        for metric in important_metrics:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[metric]))
                except Exception:
                    pass
            if vals:
                summary_rows.append({
                    "variant": variant_name,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                })
    write_log_csv(summary_rows, root / "all_variants_summary_mean_std.csv")

    with open(root / "global_report.txt", "w", encoding="utf-8") as f:
        f.write("Layered split model-selection + anti-overfitting controls\n")
        f.write("Key metrics are aggregated across seeds.\n")
        f.write("hard_outer_sacrifice_dice_when_select_inner = best_hard_ckpt_hard_outer_dice - best_inner_ckpt_hard_outer_dice\n")
        f.write("inner_sacrifice_dice_when_select_hard = best_inner_ckpt_inner_dice - best_hard_ckpt_inner_dice\n")
        f.write("Higher hard_outer_sacrifice means selecting by inner costs more hard-outer performance.\n")


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)

    records = collect_volume_records(cfg)
    if len(records) == 0:
        raise RuntimeError(f"No .mha volumes found under: {cfg.domain_root}")

    train_patients, heldout_patients = split_patients(records=records, n_train_patients=cfg.n_train_patients, split_seed=cfg.split_seed)
    inner_patients, hard_outer_patients, missing_hard = split_inner_and_hard_outer(heldout_patients, cfg.hard_outer_patient_ids)

    train_slice_records = build_slice_records(records, train_patients, cfg)
    inner_slice_records = build_slice_records(records, inner_patients, cfg)
    hard_outer_slice_records = build_slice_records(records, hard_outer_patients, cfg)
    test_slice_records = inner_slice_records + hard_outer_slice_records

    train_set = SpiderMiddleSliceDataset(train_slice_records)
    inner_set = SpiderMiddleSliceDataset(inner_slice_records)
    hard_outer_set = SpiderMiddleSliceDataset(hard_outer_slice_records)
    test_set = SpiderMiddleSliceDataset(test_slice_records)

    print("=" * 100)
    print("3-level U-Net / Attention U-Net layered-validation experiment with targeted coverage analysis")
    print(f"Domain root         : {cfg.domain_root}")
    print(f"Train patients      : {train_patients}")
    print(f"Held-out patients   : {heldout_patients}")
    print(f"Inner patients      : {inner_patients}")
    print(f"Hard-outer patients : {hard_outer_patients}")
    print(f"Missing hard IDs    : {missing_hard}")
    print(f"Train slices        : {len(train_set)}")
    print(f"Inner slices        : {len(inner_set)}")
    print(f"Hard-outer slices   : {len(hard_outer_set)}")
    print(f"Test slices         : {len(test_set)}")
    print("=" * 100)

    split_rows = []
    for x in train_slice_records:
        split_rows.append({"split": "train", "patient_id": x["patient_id"], "stem": x["stem"], "sagittal_x_index": x["sagittal_x_index"]})
    for x in inner_slice_records:
        split_rows.append({"split": "inner", "patient_id": x["patient_id"], "stem": x["stem"], "sagittal_x_index": x["sagittal_x_index"]})
    for x in hard_outer_slice_records:
        split_rows.append({"split": "hard_outer", "patient_id": x["patient_id"], "stem": x["stem"], "sagittal_x_index": x["sagittal_x_index"]})
    for x in test_slice_records:
        split_rows.append({"split": "test", "patient_id": x["patient_id"], "stem": x["stem"], "sagittal_x_index": x["sagittal_x_index"]})
    write_log_csv(split_rows, Path(cfg.out_dir) / "split_manifest.csv")

    train_loader = DataLoader(train_set, batch_size=min(cfg.batch_size, max(len(train_set), 1)), shuffle=True, num_workers=0, drop_last=False)
    train_eval_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    inner_eval_loader = DataLoader(inner_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    hard_outer_eval_loader = DataLoader(hard_outer_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    test_eval_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    variant_names = []

    for variant in cfg.variants:
        print("\n" + "#" * 100)
        print(f"Variant = {variant.name} | model_type = {variant.model_type}")
        print("#" * 100)

        variant_root = Path(cfg.out_dir) / variant.name
        ensure_dir(variant_root)
        variant_names.append(variant.name)

        template_model = build_model(variant).to(device)
        param_info = count_parameters(template_model)
        del template_model

        histories_by_seed: Dict[int, List[Dict[str, float]]] = {}
        raw_rows = []
        train_param_rows = []

        for seed in cfg.seeds:
            print("-" * 100)
            print(f"[Variant={variant.name}] Seed={seed}")
            print("-" * 100)
            set_seed(seed)

            seed_dir = variant_root / f"seed_{seed}"
            ensure_dir(seed_dir)

            model = build_model(variant).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=variant.weight_decay)

            train_param_rows.append({
                "variant": variant.name,
                "model_type": variant.model_type,
                "seed": seed,
                "base_ch": variant.base_ch,
                "latent_ch": variant.latent_ch,
                "weight_decay": variant.weight_decay,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "params_total": param_info["params_total"],
                "params_trainable": param_info["params_trainable"],
                "train_patients": " ".join(train_patients),
                "inner_patients": " ".join(inner_patients),
                "hard_outer_patients": " ".join(hard_outer_patients),
            })

            history: List[Dict[str, float]] = []
            best_inner_dice = -1.0
            best_hard_dice = -1.0
            best_inner_epoch = -1
            best_hard_epoch = -1
            best_inner_state = None
            best_hard_state = None

            best_inner_epoch_for_es = -1
            early_stop_triggered = False
            stop_epoch = cfg.epochs

            for epoch in range(cfg.epochs):
                train_stats = train_one_epoch(model, train_loader, optimizer, device)
                train_eval = evaluate_loader(model, train_eval_loader, device)
                inner_eval = evaluate_loader(model, inner_eval_loader, device)
                hard_eval = evaluate_loader(model, hard_outer_eval_loader, device)
                test_eval = evaluate_loader(model, test_eval_loader, device)

                row = {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "variant": variant.name,
                    "model_type": variant.model_type,
                    "optim_loss": train_stats["loss"],
                    "optim_bce": train_stats["bce"],
                    "optim_dice_loss": train_stats["dice_loss"],
                    "train_grad_norm_global": train_stats["grad_norm_global"],
                    "train_param_l2": train_stats["param_l2"],
                    "train_loss": train_eval["loss"],
                    "train_dice": train_eval["mean_dice"],
                    "train_iou": train_eval["mean_iou"],
                    "train_pred_fg_ratio": train_eval["pred_fg_ratio"],
                    "inner_loss": inner_eval["loss"],
                    "inner_dice": inner_eval["mean_dice"],
                    "inner_iou": inner_eval["mean_iou"],
                    "inner_pred_fg_ratio": inner_eval["pred_fg_ratio"],
                    "hard_outer_loss": hard_eval["loss"],
                    "hard_outer_dice": hard_eval["mean_dice"],
                    "hard_outer_iou": hard_eval["mean_iou"],
                    "hard_outer_pred_fg_ratio": hard_eval["pred_fg_ratio"],
                    "test_loss": test_eval["loss"],
                    "test_dice": test_eval["mean_dice"],
                    "test_iou": test_eval["mean_iou"],
                    "gap_inner_dice": train_eval["mean_dice"] - inner_eval["mean_dice"],
                    "gap_hard_outer_dice": train_eval["mean_dice"] - hard_eval["mean_dice"],
                    "gap_test_dice": train_eval["mean_dice"] - test_eval["mean_dice"],
                    "gap_inner_loss": inner_eval["loss"] - train_eval["loss"],
                    "gap_hard_outer_loss": hard_eval["loss"] - train_eval["loss"],
                    "gap_test_loss": test_eval["loss"] - train_eval["loss"],
                }
                history.append(row)

                if inner_eval["mean_dice"] > best_inner_dice:
                    best_inner_dice = inner_eval["mean_dice"]
                    best_inner_epoch = epoch + 1
                    best_inner_epoch_for_es = epoch + 1
                    if cfg.save_checkpoint:
                        best_inner_state = copy.deepcopy(model.state_dict())

                if hard_eval["mean_dice"] > best_hard_dice:
                    best_hard_dice = hard_eval["mean_dice"]
                    best_hard_epoch = epoch + 1
                    if cfg.save_checkpoint:
                        best_hard_state = copy.deepcopy(model.state_dict())

                if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
                    print(
                        f"[{variant.name}] [seed={seed}] [Epoch {epoch + 1:03d}/{cfg.epochs}] "
                        f"optim_bce={train_stats['bce']:.4f} "
                        f"optim_loss={train_stats['loss']:.4f} "
                        f"train_dice={train_eval['mean_dice']:.4f} "
                        f"inner_dice={inner_eval['mean_dice']:.4f} "
                        f"hard_dice={hard_eval['mean_dice']:.4f} "
                        f"test_dice={test_eval['mean_dice']:.4f} "
                        f"best_inner={best_inner_dice:.4f}@{best_inner_epoch} "
                        f"best_hard={best_hard_dice:.4f}@{best_hard_epoch}"
                    )

                if variant.use_early_stopping:
                    if (epoch + 1) - best_inner_epoch_for_es >= variant.es_patience:
                        early_stop_triggered = True
                        stop_epoch = epoch + 1
                        print(
                            f"[{variant.name}] [seed={seed}] Early stopping triggered at epoch {stop_epoch} "
                            f"(best inner at epoch {best_inner_epoch_for_es}, patience={variant.es_patience})"
                        )
                        break

            histories_by_seed[seed] = history
            write_log_csv(history, seed_dir / "history.csv")
            save_seed_curves(history, str(seed_dir / "train_inner_hardouter_curves.png"))

            if cfg.save_checkpoint:
                torch.save(model.state_dict(), seed_dir / "last_model.pt")
                if best_inner_state is not None:
                    torch.save(best_inner_state, seed_dir / "best_inner_model.pt")
                if best_hard_state is not None:
                    torch.save(best_hard_state, seed_dir / "best_hard_outer_model.pt")

            checkpoint_states = {
                "last": copy.deepcopy(model.state_dict()),
                "best_inner": best_inner_state if best_inner_state is not None else copy.deepcopy(model.state_dict()),
                "best_hard_outer": best_hard_state if best_hard_state is not None else copy.deepcopy(model.state_dict()),
            }

            eval_rows = []
            for checkpoint_tag, state_dict in checkpoint_states.items():
                eval_model = build_model(variant).to(device)
                eval_model.load_state_dict(state_dict)

                train_summary = evaluate_and_optionally_save_split_predictions(
                    model=eval_model,
                    dataset=train_set,
                    device=device,
                    out_dir=seed_dir,
                    seed=seed,
                    split_name="train",
                    checkpoint_tag=checkpoint_tag,
                    save_details=True,
                )
                inner_summary = evaluate_and_optionally_save_split_predictions(
                    model=eval_model,
                    dataset=inner_set,
                    device=device,
                    out_dir=seed_dir,
                    seed=seed,
                    split_name="inner",
                    checkpoint_tag=checkpoint_tag,
                    save_details=True,
                )
                hard_summary = evaluate_and_optionally_save_split_predictions(
                    model=eval_model,
                    dataset=hard_outer_set,
                    device=device,
                    out_dir=seed_dir,
                    seed=seed,
                    split_name="hard_outer",
                    checkpoint_tag=checkpoint_tag,
                    save_details=True,
                )
                test_summary = evaluate_and_optionally_save_split_predictions(
                    model=eval_model,
                    dataset=test_set,
                    device=device,
                    out_dir=seed_dir,
                    seed=seed,
                    split_name="test",
                    checkpoint_tag=checkpoint_tag,
                    save_details=True,
                )

                merged = {
                    "variant": variant.name,
                    "model_type": variant.model_type,
                    "seed": seed,
                    "checkpoint_tag": checkpoint_tag,
                    "best_inner_epoch": best_inner_epoch,
                    "best_hard_epoch": best_hard_epoch,
                    "stop_epoch": stop_epoch,
                    "early_stop_triggered": int(early_stop_triggered),
                    **{f"train_{k}": v for k, v in train_summary.items() if k not in {"seed", "split", "checkpoint_tag"}},
                    **{f"inner_{k}": v for k, v in inner_summary.items() if k not in {"seed", "split", "checkpoint_tag"}},
                    **{f"hard_outer_{k}": v for k, v in hard_summary.items() if k not in {"seed", "split", "checkpoint_tag"}},
                    **{f"test_{k}": v for k, v in test_summary.items() if k not in {"seed", "split", "checkpoint_tag"}},
                }
                eval_rows.append(merged)

            write_log_csv(eval_rows, seed_dir / "checkpoint_selection_reval.csv")

            by_tag = {r["checkpoint_tag"]: r for r in eval_rows}
            last_r = by_tag["last"]
            bi_r = by_tag["best_inner"]
            bh_r = by_tag["best_hard_outer"]

            raw_rows.append({
                "variant": variant.name,
                "model_type": variant.model_type,
                "seed": seed,
                "base_ch": variant.base_ch,
                "latent_ch": variant.latent_ch,
                "weight_decay": variant.weight_decay,
                "use_early_stopping": int(variant.use_early_stopping),
                "es_patience": variant.es_patience if variant.use_early_stopping else 0,
                "early_stop_triggered": int(early_stop_triggered),
                "stop_epoch": stop_epoch,
                "best_inner_epoch": best_inner_epoch,
                "best_hard_epoch": best_hard_epoch,
                "params_total": param_info["params_total"],
                "params_trainable": param_info["params_trainable"],
                "last_train_mean_dice": last_r["train_mean_dice"],
                "last_inner_mean_dice": last_r["inner_mean_dice"],
                "last_hard_outer_mean_dice": last_r["hard_outer_mean_dice"],
                "last_test_mean_dice": last_r["test_mean_dice"],
                "last_train_mean_iou": last_r["train_mean_iou"],
                "last_inner_mean_iou": last_r["inner_mean_iou"],
                "last_hard_outer_mean_iou": last_r["hard_outer_mean_iou"],
                "last_test_mean_iou": last_r["test_mean_iou"],
                "last_train_loss": history[-1]["train_loss"],
                "last_inner_loss": history[-1]["inner_loss"],
                "last_hard_outer_loss": history[-1]["hard_outer_loss"],
                "last_test_loss": history[-1]["test_loss"],
                "best_inner_ckpt_train_mean_dice": bi_r["train_mean_dice"],
                "best_inner_ckpt_inner_mean_dice": bi_r["inner_mean_dice"],
                "best_inner_ckpt_hard_outer_mean_dice": bi_r["hard_outer_mean_dice"],
                "best_inner_ckpt_test_mean_dice": bi_r["test_mean_dice"],
                "best_inner_ckpt_train_mean_iou": bi_r["train_mean_iou"],
                "best_inner_ckpt_inner_mean_iou": bi_r["inner_mean_iou"],
                "best_inner_ckpt_hard_outer_mean_iou": bi_r["hard_outer_mean_iou"],
                "best_inner_ckpt_test_mean_iou": bi_r["test_mean_iou"],
                "best_hard_ckpt_train_mean_dice": bh_r["train_mean_dice"],
                "best_hard_ckpt_inner_mean_dice": bh_r["inner_mean_dice"],
                "best_hard_ckpt_hard_outer_mean_dice": bh_r["hard_outer_mean_dice"],
                "best_hard_ckpt_test_mean_dice": bh_r["test_mean_dice"],
                "best_hard_ckpt_train_mean_iou": bh_r["train_mean_iou"],
                "best_hard_ckpt_inner_mean_iou": bh_r["inner_mean_iou"],
                "best_hard_ckpt_hard_outer_mean_iou": bh_r["hard_outer_mean_iou"],
                "best_hard_ckpt_test_mean_iou": bh_r["test_mean_iou"],
                "hard_outer_sacrifice_dice_when_select_inner": bh_r["hard_outer_mean_dice"] - bi_r["hard_outer_mean_dice"],
                "hard_outer_sacrifice_iou_when_select_inner": bh_r["hard_outer_mean_iou"] - bi_r["hard_outer_mean_iou"],
                "inner_sacrifice_dice_when_select_hard": bi_r["inner_mean_dice"] - bh_r["inner_mean_dice"],
                "inner_sacrifice_iou_when_select_hard": bi_r["inner_mean_iou"] - bh_r["inner_mean_iou"],
                "test_sacrifice_dice_when_select_inner": bh_r["test_mean_dice"] - bi_r["test_mean_dice"],
                "test_sacrifice_iou_when_select_inner": bh_r["test_mean_iou"] - bi_r["test_mean_iou"],
                "gap_train_minus_inner_at_best_inner": bi_r["train_mean_dice"] - bi_r["inner_mean_dice"],
                "gap_train_minus_hard_at_best_inner": bi_r["train_mean_dice"] - bi_r["hard_outer_mean_dice"],
                "gap_train_minus_test_at_best_inner": bi_r["train_mean_dice"] - bi_r["test_mean_dice"],
                "gap_train_minus_inner_at_best_hard": bh_r["train_mean_dice"] - bh_r["inner_mean_dice"],
                "gap_train_minus_hard_at_best_hard": bh_r["train_mean_dice"] - bh_r["hard_outer_mean_dice"],
                "gap_train_minus_test_at_best_hard": bh_r["train_mean_dice"] - bh_r["test_mean_dice"],
            })

        write_log_csv(train_param_rows, variant_root / "training_params.csv")
        write_log_csv(raw_rows, variant_root / "raw_results.csv")
        write_log_csv([row for history in histories_by_seed.values() for row in history], variant_root / "all_seeds_history.csv")
        save_all_seeds_overlay_curves(histories_by_seed, str(variant_root / "all_seeds_overlay_curves.png"))
        save_all_seeds_mean_std_curves(histories_by_seed, str(variant_root / "all_seeds_mean_std_curves.png"))
        aggregate_split_reports_across_seeds(
            variant_root,
            split_names=("train", "inner", "hard_outer", "test"),
            checkpoint_tags=("last", "best_inner", "best_hard_outer"),
        )
        summarize_variant_results(variant_root)

        with open(variant_root / "run_report.txt", "w", encoding="utf-8") as f:
            f.write("3-level U-Net / Attention U-Net layered split experiment\n")
            f.write(f"Variant             : {variant.name}\n")
            f.write(f"Model type          : {variant.model_type}\n")
            f.write(f"Base channels       : {variant.base_ch}\n")
            f.write(f"Latent channels     : {variant.latent_ch}\n")
            f.write(f"Weight decay        : {variant.weight_decay}\n")
            f.write(f"LR                  : {cfg.lr}\n")
            f.write(f"Epochs              : {cfg.epochs}\n")
            f.write(f"Batch size          : {cfg.batch_size}\n")
            f.write(f"Lambda BCE          : {cfg.lambda_bce}\n")
            f.write(f"Lambda Dice         : {cfg.lambda_dice}\n")
            f.write("Training objective  : BCE only\n" if cfg.lambda_dice == 0 else "Training objective  : BCE + Dice\n")
            f.write("Dice usage          : monitoring / checkpoint selection metric\n")
            f.write(f"Params total        : {param_info['params_total']}\n")
            f.write(f"Params trainable    : {param_info['params_trainable']}\n")
            f.write(f"Early stopping      : {variant.use_early_stopping}\n")
            f.write(f"ES patience         : {variant.es_patience if variant.use_early_stopping else 'N/A'}\n")
            f.write(f"Domain root         : {cfg.domain_root}\n")
            f.write(f"Train patients      : {' '.join(train_patients)}\n")
            f.write(f"Held-out patients   : {' '.join(heldout_patients)}\n")
            f.write(f"Inner patients      : {' '.join(inner_patients)}\n")
            f.write(f"Hard-outer patients : {' '.join(hard_outer_patients)}\n")
            f.write(f"Missing hard IDs    : {' '.join(missing_hard) if missing_hard else 'None'}\n")
            f.write(f"Train slices        : {len(train_set)}\n")
            f.write(f"Inner slices        : {len(inner_set)}\n")
            f.write(f"Hard-outer slices   : {len(hard_outer_set)}\n")
            f.write(f"Test slices         : {len(test_set)}\n")
            f.write("Checkpoints re-evaluated: last / best_inner / best_hard_outer\n")
            f.write("Selection sacrifice metric:\n")
            f.write("  hard_outer_sacrifice_dice_when_select_inner = best_hard_ckpt_hard_outer_dice - best_inner_ckpt_hard_outer_dice\n")
            f.write("  inner_sacrifice_dice_when_select_hard = best_inner_ckpt_inner_dice - best_hard_ckpt_inner_dice\n")
            f.write("  test_sacrifice_dice_when_select_inner = best_hard_ckpt_test_dice - best_inner_ckpt_test_dice\n")

    aggregate_all_variants(Path(cfg.out_dir), variant_names)

    # experiment manifest JSON
    experiment_manifest = {
        "domain_root": cfg.domain_root,
        "train_patients": train_patients,
        "heldout_patients": heldout_patients,
        "inner_patients": inner_patients,
        "hard_outer_patients": hard_outer_patients,
        "missing_hard_ids": missing_hard,
        "n_train_slices": len(train_set),
        "n_inner_slices": len(inner_set),
        "n_hard_outer_slices": len(hard_outer_set),
        "n_test_slices": len(test_set),
        "variants": [v.name for v in cfg.variants],
        "seeds": list(cfg.seeds),
        "checkpoint_tags": ["last", "best_inner", "best_hard_outer"],
        "coarse_thresholds": list(cfg.coarse_thresholds),
        "sparsity_layers": list(cfg.sparsity_layers),
    }
    with open(Path(cfg.out_dir) / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(experiment_manifest, f, ensure_ascii=False, indent=2)

    with open(Path(cfg.out_dir) / "experiment_manifest.txt", "w", encoding="utf-8") as f:
        f.write("Current 3-level U-Net / Attention U-Net experiment variants:\n")
        for v in cfg.variants:
            f.write(
                f"{v.name}: model_type={v.model_type}, base_ch={v.base_ch}, latent_ch={v.latent_ch}, "
                f"wd={v.weight_decay}, early_stopping={v.use_early_stopping}, patience={v.es_patience}\n"
            )

    print("=" * 100)
    print("3-level U-Net / Attention U-Net experiments finished.")
    print(f"Output dir -> {Path(cfg.out_dir).resolve()}")
    print("=" * 100)


if __name__ == "__main__":
    main()