
from __future__ import annotations

import os
import csv
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.nn.utils.parametrizations import spectral_norm as apply_spectral_norm
except Exception:
    from torch.nn.utils import spectral_norm as apply_spectral_norm

# 当前版本的问题：
# I. 14层卷积太深了，救不了。最多3层卷积+三层反转卷积，以此基础做incremental design。
# II. L1正则化做的太粗了，对enc block整体施加稀疏约束，而不是每个位置的character vector，不符合LeCun提出的‘向量稀疏’理念。
# 尝试的改进：
# I. 砍掉一层enc block.
# II. 将L1改为每个(C*H*W)特征图中每个(H,W)空间位置上C维向量的稀疏惩罚。

# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    # -------- data --------
    domain_root: str = "SPIDER_domain_strict/SIEMENS_SymphonyTim_37743_T1-TSE"
    image_dir: str = "images"
    mask_dir: str = "masks"

    # 输出目录
    out_dir: str = "SparseAE_singleDomainTest/L1_AE_v1"

    # 单域 5-shot：5 个训练病人，其余全部测试
    n_train_patients: int = 5
    split_seed: int = 0
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- slice selection --------
    # 每个病人只取矢状面前景范围中部三张
    num_middle_slices: int = 3

    # -------- image size --------
    resize_hw: Tuple[int, int] = (224, 224)

    # 若 preprocessing 后是 robust z-score，可先 clip 再映射到 [0,1]
    input_clip_min: float = -3.0
    input_clip_max: float = 3.0

    # 前景标签；分割椎体
    fg_labels: Optional[Tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)

    # -------- model --------
    latent_ch: int = 64
    base_ch: int = 16

    # -------- optimization --------
    epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # -------- loss weights --------
    lambda_bce: float = 1.0
    lambda_dice: float = 1.0
    lambda_sparse: float = 2e-4
    lambda_sparse_enc1: float = 1e-4
    lambda_sparse_enc2: float = 5e-4

    lambda_lowrank_dec3: float = 0.0
    lambda_lowrank_dec2: float = 0.0
    lambda_lowrank_feat: float = 0.0

    # -------- prediction --------
    pred_thr: float = 0.5
    bottleneck_zero_thr: float = 1e-3

    # -------- save --------
    save_checkpoint: bool = True
    save_all_test_vis: bool = True


cfg = Cfg()
print(cfg)


# =========================================================
# Ablation / normalization settings
# =========================================================
@dataclass(frozen=True)
class AblationSetting:
    name: str
    lambda_sparse_z: float
    lambda_sparse_enc1: float
    lambda_sparse_enc2: float


@dataclass(frozen=True)
class NormSetting:
    name: str
    use_sn_prev_conv: bool
    use_sn_bottleneck: bool


def get_ablation_settings() -> List[AblationSetting]:
    return [
        AblationSetting(
            name="baseline_no_sparse",
            lambda_sparse_z=0.0,
            lambda_sparse_enc1=0.0,
            lambda_sparse_enc2=0.0,
        ),
        AblationSetting(
            name="z_only_sparse",
            lambda_sparse_z=cfg.lambda_sparse,
            lambda_sparse_enc1=0.0,
            lambda_sparse_enc2=0.0,
        ),
        AblationSetting(
            name="full_sparse",
            lambda_sparse_z=cfg.lambda_sparse,
            lambda_sparse_enc1=cfg.lambda_sparse_enc1,
            lambda_sparse_enc2=cfg.lambda_sparse_enc2,
        ),
    ]


def get_norm_settings() -> List[NormSetting]:
    return [
        NormSetting(
            name="no_sn",
            use_sn_prev_conv=False,
            use_sn_bottleneck=False,
        ),
        NormSetting(
            name="sn_bottleneck_only",
            use_sn_prev_conv=False,
            use_sn_bottleneck=True,
        ),
        NormSetting(
            name="sn_prev_and_bottleneck",
            use_sn_prev_conv=True,
            use_sn_bottleneck=True,
        ),
    ]


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


def iou_and_dice(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter) / float(union + 1e-9)
    dice = float(2 * inter) / float(pred.sum() + gt.sum() + 1e-9)
    return iou, dice


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def low_rank_loss(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, C, H, W]
    """
    b, c, h, w = feat.shape
    mats = feat.reshape(b, c, h * w)
    loss = 0.0
    for i in range(b):
        s = torch.linalg.svdvals(mats[i])
        loss = loss + s.sum() / (min(c, h * w) + 1e-6)
    return loss / b


def write_log_csv(rows: List[Dict], path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
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


def write_dict_of_lists_csv(data: Dict[str, List[float]], path: str | Path):
    rows = []
    if not data:
        write_log_csv(rows, path)
        return
    max_len = max(len(v) for v in data.values())
    pids = sorted(data.keys())
    for epoch_idx in range(max_len):
        row = {"epoch": epoch_idx + 1}
        for pid in pids:
            vals = data[pid]
            row[pid] = vals[epoch_idx] if epoch_idx < len(vals) else ""
        rows.append(row)
    write_log_csv(rows, path)


def parse_patient_id_from_stem(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return stem


def resize_np_2d(arr: np.ndarray, size_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    t = torch.from_numpy(arr[None, None, ...]).float()
    mode = "nearest" if is_mask else "bilinear"
    out = F.interpolate(
        t,
        size=size_hw,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return out[0, 0].cpu().numpy()


def normalize_slice_to_01(img2d: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    img2d = np.nan_to_num(img2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    img2d = np.clip(img2d, clip_min, clip_max)
    img2d = (img2d - clip_min) / max(clip_max - clip_min, 1e-8)
    img2d = np.clip(img2d, 0.0, 1.0)
    return img2d.astype(np.float32)


def select_middle_sagittal_indices(mask_zyx: np.ndarray, num_slices: int = 3) -> List[int]:
    """
    mask_zyx: [Z, Y, X]
    取矢状面中部三张：沿 X 轴选切片。
    若 mask 非空，则按前景所在 X 范围取中部三张；
    若 mask 为空，则退化为几何中心三张。
    """
    x_any = np.where(mask_zyx.sum(axis=(0, 1)) > 0)[0]
    x_dim = mask_zyx.shape[2]

    if len(x_any) > 0:
        x_min = int(x_any[0])
        x_max = int(x_any[-1])
        x_center = int(round((x_min + x_max) / 2.0))
    else:
        x_center = x_dim // 2

    if num_slices == 1:
        idxs = [x_center]
    elif num_slices == 3:
        idxs = [x_center - 1, x_center, x_center + 1]
    else:
        half = num_slices // 2
        idxs = list(range(x_center - half, x_center + half + 1))

    idxs = [min(max(x, 0), x_dim - 1) for x in idxs]
    return idxs


def binarize_mask(mask2d: np.ndarray, fg_labels: Optional[Tuple[int, ...]]) -> np.ndarray:
    if fg_labels is None:
        out = (mask2d > 0).astype(np.float32)
    else:
        keep = np.isin(mask2d.astype(np.int32), np.asarray(fg_labels, dtype=np.int32))
        out = keep.astype(np.float32)
    return out


def make_patient_curve_store(patient_ids: List[str]) -> Dict[str, List[float]]:
    return {str(pid): [] for pid in sorted(patient_ids)}


# =========================================================
# Dataset building
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
    test_patients = sorted(patient_ids[n_train_patients:])
    return train_patients, test_patients


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
    for r in records:
        if r["patient_id"] not in kept_patient_ids:
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
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SparseAENet(nn.Module):
    def __init__(self, base_ch: int = 16, latent_ch: int = 64, norm_setting: Optional[NormSetting] = None):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.norm_setting = norm_setting or NormSetting("no_sn", False, False)

        self.enc1 = ConvBlock(1, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = BottleneckBlock(c3, latent_ch)

        self.up3 = nn.ConvTranspose2d(latent_ch, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3, c2)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2, c1)

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1, c1)

        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

        self._apply_requested_spectral_norms()

    def _apply_requested_spectral_norms(self):
        if self.norm_setting.use_sn_prev_conv:
            self.enc3.conv2 = apply_spectral_norm(self.enc3.conv2)
        if self.norm_setting.use_sn_bottleneck:
            self.bottleneck.conv = apply_spectral_norm(self.bottleneck.conv)

    def forward(self, x: torch.Tensor):
        f1 = self.enc1(x)
        x = self.pool1(f1)

        f2 = self.enc2(x)
        x = self.pool2(f2)

        f3 = self.enc3(x)
        x = self.pool3(f3)

        z = self.bottleneck(x)

        d3 = self.up3(z)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        feat = self.dec1(d1)

        seg_logits = self.seg_head(feat)

        aux = {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "d3": d3,
            "d2": d2,
            "feat": feat,
        }
        return seg_logits, z, aux


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    setting: AblationSetting,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_sparse = 0.0
    total_sparse_z = 0.0
    total_sparse_f1 = 0.0
    total_sparse_f2 = 0.0
    total_lowrank = 0.0
    total_lowrank_dec3 = 0.0
    total_lowrank_dec2 = 0.0
    total_lowrank_feat = 0.0
    n_seen = 0

    for img, mask, _meta in loader:
        img = img.to(device)
        mask = mask.to(device)

        seg_logits, z, aux = model(img)

        loss_bce = F.binary_cross_entropy_with_logits(seg_logits, mask)
        loss_dice = dice_loss_from_logits(seg_logits, mask)

        loss_sparse_z = z.abs().mean()
        loss_sparse_f1 = aux["f1"].abs().mean()
        loss_sparse_f2 = aux["f2"].abs().mean()

        loss_lowrank_dec3 = low_rank_loss(aux["d3"])
        loss_lowrank_dec2 = low_rank_loss(aux["d2"])
        loss_lowrank_feat = low_rank_loss(aux["feat"])

        loss_sparse = (
            setting.lambda_sparse_z * loss_sparse_z
            + setting.lambda_sparse_enc1 * loss_sparse_f1
            + setting.lambda_sparse_enc2 * loss_sparse_f2
        )

        loss_lowrank = (
            cfg.lambda_lowrank_dec3 * loss_lowrank_dec3
            + cfg.lambda_lowrank_dec2 * loss_lowrank_dec2
            + cfg.lambda_lowrank_feat * loss_lowrank_feat
        )

        loss = (
            cfg.lambda_bce * loss_bce
            + cfg.lambda_dice * loss_dice
            + loss_sparse
            + loss_lowrank
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = img.shape[0]
        n_seen += bs
        total_loss += float(loss.item()) * bs
        total_bce += float(loss_bce.item()) * bs
        total_dice += float(loss_dice.item()) * bs
        total_sparse += float(loss_sparse.item()) * bs
        total_sparse_z += float(loss_sparse_z.item()) * bs
        total_sparse_f1 += float(loss_sparse_f1.item()) * bs
        total_sparse_f2 += float(loss_sparse_f2.item()) * bs
        total_lowrank += float(loss_lowrank.item()) * bs
        total_lowrank_dec3 += float(loss_lowrank_dec3.item()) * bs
        total_lowrank_dec2 += float(loss_lowrank_dec2.item()) * bs
        total_lowrank_feat += float(loss_lowrank_feat.item()) * bs

    denom = max(n_seen, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "dice_loss": total_dice / denom,
        "sparse": total_sparse / denom,
        "sparse_z_raw": total_sparse_z / denom,
        "sparse_f1_raw": total_sparse_f1 / denom,
        "sparse_f2_raw": total_sparse_f2 / denom,
        "lowrank": total_lowrank / denom,
        "lowrank_dec3": total_lowrank_dec3 / denom,
        "lowrank_dec2": total_lowrank_dec2 / denom,
        "lowrank_feat": total_lowrank_feat / denom,
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    setting: AblationSetting,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_sparse = 0.0
    total_sparse_z = 0.0
    total_sparse_f1 = 0.0
    total_sparse_f2 = 0.0
    total_lowrank = 0.0

    dices = []
    ious = []
    z_abs_list = []
    z_zero_list = []
    patient_to_dices = defaultdict(list)
    n_seen = 0

    for img, mask, meta in loader:
        img = img.to(device)
        mask = mask.to(device)

        seg_logits, z, aux = model(img)

        loss_bce = F.binary_cross_entropy_with_logits(seg_logits, mask)
        loss_dice = dice_loss_from_logits(seg_logits, mask)

        loss_sparse_z = z.abs().mean()
        loss_sparse_f1 = aux["f1"].abs().mean()
        loss_sparse_f2 = aux["f2"].abs().mean()

        loss_sparse = (
            setting.lambda_sparse_z * loss_sparse_z
            + setting.lambda_sparse_enc1 * loss_sparse_f1
            + setting.lambda_sparse_enc2 * loss_sparse_f2
        )

        loss_lowrank = (
            cfg.lambda_lowrank_dec3 * low_rank_loss(aux["d3"])
            + cfg.lambda_lowrank_dec2 * low_rank_loss(aux["d2"])
            + cfg.lambda_lowrank_feat * low_rank_loss(aux["feat"])
        )

        loss = (
            cfg.lambda_bce * loss_bce
            + cfg.lambda_dice * loss_dice
            + loss_sparse
            + loss_lowrank
        )

        prob = torch.sigmoid(seg_logits)
        pred = (prob >= cfg.pred_thr).float()

        for b in range(img.shape[0]):
            pred_np = pred[b, 0].cpu().numpy()
            gt_np = mask[b, 0].cpu().numpy()
            iou, dice = iou_and_dice(pred_np, gt_np)
            ious.append(iou)
            dices.append(dice)
            pid = str(meta["patient_id"][b] if isinstance(meta["patient_id"], list) else meta["patient_id"])
            patient_to_dices[pid].append(dice)

        z_abs = z.abs()
        z_zero_ratio = float((z_abs < cfg.bottleneck_zero_thr).float().mean().item())
        z_mean_abs = float(z_abs.mean().item())

        z_abs_list.append(z_mean_abs)
        z_zero_list.append(z_zero_ratio)

        bs = img.shape[0]
        n_seen += bs
        total_loss += float(loss.item()) * bs
        total_bce += float(loss_bce.item()) * bs
        total_dice_loss += float(loss_dice.item()) * bs
        total_sparse += float(loss_sparse.item()) * bs
        total_sparse_z += float(loss_sparse_z.item()) * bs
        total_sparse_f1 += float(loss_sparse_f1.item()) * bs
        total_sparse_f2 += float(loss_sparse_f2.item()) * bs
        total_lowrank += float(loss_lowrank.item()) * bs

    patient_dice_by_pid = {pid: float(np.mean(vals)) for pid, vals in patient_to_dices.items()}

    denom = max(n_seen, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "dice_loss": total_dice_loss / denom,
        "sparse": total_sparse / denom,
        "sparse_z_raw": total_sparse_z / denom,
        "sparse_f1_raw": total_sparse_f1 / denom,
        "sparse_f2_raw": total_sparse_f2 / denom,
        "lowrank": total_lowrank / denom,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "std_dice": float(np.std(dices)) if dices else 0.0,
        "mean_z_abs": float(np.mean(z_abs_list)) if z_abs_list else 0.0,
        "mean_z_zero_ratio": float(np.mean(z_zero_list)) if z_zero_list else 0.0,
        "patient_dice_by_pid": patient_dice_by_pid,
    }


@torch.no_grad()
def infer_one(model: nn.Module, img: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor | float]:
    model.eval()
    img = img.to(device)

    seg_logits, z, _ = model(img)
    prob = torch.sigmoid(seg_logits)
    pred = (prob >= cfg.pred_thr).float()

    z_abs = z.abs()
    z_zero_ratio = float((z_abs < cfg.bottleneck_zero_thr).float().mean().item())
    z_mean_abs = float(z_abs.mean().item())

    return {
        "prob": prob.cpu(),
        "pred": pred.cpu(),
        "z": z.cpu(),
        "z_zero_ratio": z_zero_ratio,
        "z_mean_abs": z_mean_abs,
    }


# =========================================================
# Visualization
# =========================================================
def save_case_vis(
    case_name: str,
    img: np.ndarray,
    gt: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    z: np.ndarray,
    meta: Dict,
    out_path: str,
):
    z_map = np.mean(np.abs(z), axis=0)
    z_map = (z_map - z_map.min()) / (z_map.max() - z_map.min() + 1e-9)

    iou, dice = iou_and_dice(pred, gt)

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")

    axes[2].imshow(prob, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pred prob")

    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(pred == 0, pred), alpha=0.50)
    axes[3].set_title(f"Pred mask\nDice={dice:.3f}")

    axes[4].imshow(z_map, cmap="magma", vmin=0, vmax=1)
    axes[4].set_title("mean |z|")

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


def save_training_curves(history: List[Dict[str, float]], out_path: str):
    xs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].plot(xs, [h["train_dice"] for h in history], label="train_dice")
    axes[0].plot(xs, [h["test_dice"] for h in history], label="test_dice")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice")
    axes[0].set_title("Train/Test Dice Curves")
    axes[0].legend()

    axes[1].plot(xs, [h["train_loss"] for h in history], label="train_loss")
    axes[1].plot(xs, [h["test_loss"] for h in history], label="test_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train/Test Loss Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_per_patient_dice_curves(
    patient_curve_store: Dict[str, List[float]],
    out_path: str,
    title: str,
):
    if not patient_curve_store:
        return

    xs = np.arange(1, max(len(v) for v in patient_curve_store.values()) + 1)
    plt.figure(figsize=(10, 6))
    for pid in sorted(patient_curve_store.keys()):
        ys = patient_curve_store[pid]
        if len(ys) == 0:
            continue
        plt.plot(xs[:len(ys)], ys, label=f"pid={pid}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(title)
    if len(patient_curve_store) <= 12:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================================================
# Final evaluation on test set
# =========================================================
@torch.no_grad()
def evaluate_and_save_predictions(
    model: nn.Module,
    test_set: Dataset,
    device: torch.device,
    exp_dir: str,
    setting: AblationSetting,
    norm_setting: NormSetting,
) -> Dict[str, float]:
    ensure_dir(exp_dir)
    vis_dir = os.path.join(exp_dir, "vis_test")
    ensure_dir(vis_dir)

    rows = []
    ious = []
    dices = []
    patient_to_dices = defaultdict(list)

    print("=" * 80)
    print(f"Testing [{norm_setting.name} | {setting.name}] ...")
    print("=" * 80)

    for i in range(len(test_set)):
        img_t, gt_t, meta = test_set[i]
        out = infer_one(model, img_t.unsqueeze(0), device)

        img = img_t.squeeze(0).numpy().astype(np.float32)
        gt = gt_t.squeeze(0).numpy().astype(np.float32)
        prob = out["prob"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        pred = out["pred"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        z = out["z"].squeeze(0).numpy().astype(np.float32)

        iou, dice = iou_and_dice(pred, gt)
        ious.append(iou)
        dices.append(dice)
        patient_to_dices[meta["patient_id"]].append(dice)

        row = {
            "case_id": i,
            "patient_id": meta["patient_id"],
            "stem": meta["stem"],
            "sagittal_x_index": meta["sagittal_x_index"],
            "iou": iou,
            "dice": dice,
            "z_mean_abs": float(out["z_mean_abs"]),
            "z_zero_ratio": float(out["z_zero_ratio"]),
        }
        rows.append(row)

        print(
            f"[{i + 1:03d}/{len(test_set)}] "
            f"pid={meta['patient_id']} stem={meta['stem']} sagittal_x={meta['sagittal_x_index']} "
            f"IoU={iou:.4f} Dice={dice:.4f} "
            f"|z|={out['z_mean_abs']:.4f} zero(z)={out['z_zero_ratio']:.3f}"
        )

        if cfg.save_all_test_vis:
            save_case_vis(
                case_name=f"case_{i:03d}",
                img=img,
                gt=gt,
                prob=prob,
                pred=pred,
                z=z,
                meta=meta,
                out_path=os.path.join(vis_dir, f"case_{i:03d}.png"),
            )

    mean_iou = float(np.mean(ious)) if ious else 0.0
    std_iou = float(np.std(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    std_dice = float(np.std(dices)) if dices else 0.0

    patient_mean_dices = [float(np.mean(v)) for v in patient_to_dices.values()]
    mean_patient_dice = float(np.mean(patient_mean_dices)) if patient_mean_dices else 0.0
    std_patient_dice = float(np.std(patient_mean_dices)) if patient_mean_dices else 0.0

    mean_z = float(np.mean([r["z_mean_abs"] for r in rows])) if rows else 0.0
    mean_zero = float(np.mean([r["z_zero_ratio"] for r in rows])) if rows else 0.0

    print("=" * 80)
    print(f"[{norm_setting.name} | {setting.name}] Slice mean IoU      = {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"[{norm_setting.name} | {setting.name}] Slice mean Dice     = {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"[{norm_setting.name} | {setting.name}] Patient mean Dice   = {mean_patient_dice:.4f} ± {std_patient_dice:.4f}")
    print(f"[{norm_setting.name} | {setting.name}] Mean |z|            = {mean_z:.4f}")
    print(f"[{norm_setting.name} | {setting.name}] Mean zero(z)        = {mean_zero:.4f}")
    print("=" * 80)

    metrics_path = os.path.join(exp_dir, "test_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else ["case_id", "iou", "dice"]
        )
        writer.writeheader()
        writer.writerows(rows)

    patient_metrics_rows = []
    for pid, vals in sorted(patient_to_dices.items()):
        patient_metrics_rows.append({
            "patient_id": pid,
            "mean_dice": float(np.mean(vals)),
            "std_dice": float(np.std(vals)),
            "n_slices": len(vals),
        })
    write_log_csv(patient_metrics_rows, os.path.join(exp_dir, "test_patient_metrics.csv"))

    summary_path = os.path.join(exp_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Norm setting         = {norm_setting.name}\n")
        f.write(f"Experiment           = {setting.name}\n")
        f.write(f"lambda_sparse_z      = {setting.lambda_sparse_z}\n")
        f.write(f"lambda_sparse_enc1   = {setting.lambda_sparse_enc1}\n")
        f.write(f"lambda_sparse_enc2   = {setting.lambda_sparse_enc2}\n")
        f.write(f"Slice Mean IoU       = {mean_iou:.6f} +- {std_iou:.6f}\n")
        f.write(f"Slice Mean Dice      = {mean_dice:.6f} +- {std_dice:.6f}\n")
        f.write(f"Patient Mean Dice    = {mean_patient_dice:.6f} +- {std_patient_dice:.6f}\n")
        f.write(f"Mean |z|             = {mean_z:.6f}\n")
        f.write(f"Mean zero(z)         = {mean_zero:.6f}\n")
        f.write("NOTE                 = test curve is for diagnosis only; do not use it for formal model selection.\n")

    return {
        "norm_name": norm_setting.name,
        "experiment": setting.name,
        "lambda_sparse_z": setting.lambda_sparse_z,
        "lambda_sparse_enc1": setting.lambda_sparse_enc1,
        "lambda_sparse_enc2": setting.lambda_sparse_enc2,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_patient_dice": mean_patient_dice,
        "std_patient_dice": std_patient_dice,
        "mean_z_abs": mean_z,
        "mean_z_zero_ratio": mean_zero,
    }


def save_grouped_results(raw_rows: List[Dict], out_path: str):
    grouped = defaultdict(list)

    for r in raw_rows:
        key = (r["norm_name"], r["experiment"])
        grouped[key].append(r)

    summary_rows = []
    for (norm_name, experiment), rows in grouped.items():
        final_test_dice = np.array([r["final_test_dice"] for r in rows], dtype=np.float32)
        best_test_dice = np.array([r["best_test_dice"] for r in rows], dtype=np.float32)
        final_patient_dice = np.array([r["mean_patient_dice"] for r in rows], dtype=np.float32)
        final_iou = np.array([r["mean_iou"] for r in rows], dtype=np.float32)

        summary_rows.append({
            "norm_name": norm_name,
            "experiment": experiment,
            "n_seeds": len(rows),
            "final_test_dice_mean": float(final_test_dice.mean()),
            "final_test_dice_std": float(final_test_dice.std()),
            "best_test_dice_mean": float(best_test_dice.mean()),
            "best_test_dice_std": float(best_test_dice.std()),
            "final_patient_dice_mean": float(final_patient_dice.mean()),
            "final_patient_dice_std": float(final_patient_dice.std()),
            "final_iou_mean": float(final_iou.mean()),
            "final_iou_std": float(final_iou.std()),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "norm_name",
                "experiment",
                "n_seeds",
                "final_test_dice_mean",
                "final_test_dice_std",
                "best_test_dice_mean",
                "best_test_dice_std",
                "final_patient_dice_mean",
                "final_patient_dice_std",
                "final_iou_mean",
                "final_iou_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)

    # 1) collect volumes and split by patient
    records = collect_volume_records(cfg)
    if len(records) == 0:
        raise RuntimeError(f"No .mha volumes found under: {cfg.domain_root}")

    train_patients, test_patients = split_patients(
        records=records,
        n_train_patients=cfg.n_train_patients,
        split_seed=cfg.split_seed,
    )

    train_slice_records = build_slice_records(records, train_patients, cfg)
    test_slice_records = build_slice_records(records, test_patients, cfg)

    train_set = SpiderMiddleSliceDataset(train_slice_records)
    test_set = SpiderMiddleSliceDataset(test_slice_records)

    print("Train slice example:")
    img_t, msk_t, meta = train_set[0]
    print("img shape =", img_t.shape)
    print("mask shape =", msk_t.shape)
    print("meta =", meta)
    print("=" * 100)
    print("Single-domain experiment")
    print(f"Domain root      : {cfg.domain_root}")
    print(f"Train patients   : {train_patients}")
    print(f"Test patients    : {test_patients}")
    print(f"Train slices     : {len(train_set)}")
    print(f"Test slices      : {len(test_set)}")
    print("=" * 100)

    # 保存切分信息
    split_rows = []
    for x in train_slice_records:
        split_rows.append({
            "split": "train",
            "patient_id": x["patient_id"],
            "stem": x["stem"],
            "sagittal_x_index": x["sagittal_x_index"],
        })
    for x in test_slice_records:
        split_rows.append({
            "split": "test",
            "patient_id": x["patient_id"],
            "stem": x["stem"],
            "sagittal_x_index": x["sagittal_x_index"],
        })
    write_log_csv(split_rows, Path(cfg.out_dir) / "split_manifest.csv")

    ablation_settings = get_ablation_settings()
    norm_settings = get_norm_settings()
    raw_rows = []

    # 固定数据集；只变 seed 初始化
    for seed in cfg.seeds:
        print("#" * 100)
        print(f"Seed = {seed}")
        print("#" * 100)
        set_seed(seed)

        seed_dir = Path(cfg.out_dir) / f"seed_{seed}"
        ensure_dir(seed_dir)

        train_loader = DataLoader(
            train_set,
            batch_size=min(cfg.batch_size, max(len(train_set), 1)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        train_eval_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        test_eval_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        for norm_setting in norm_settings:
            norm_dir = seed_dir / norm_setting.name
            ensure_dir(norm_dir)

            for setting in ablation_settings:
                exp_dir = norm_dir / setting.name
                ensure_dir(exp_dir)

                set_seed(seed)

                model = SparseAENet(
                    base_ch=cfg.base_ch,
                    latent_ch=cfg.latent_ch,
                    norm_setting=norm_setting,
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )

                print("-" * 100)
                print(f"Running: seed={seed}, norm={norm_setting.name}, exp={setting.name}")
                print(f"lambda_sparse_z    = {setting.lambda_sparse_z}")
                print(f"lambda_sparse_enc1 = {setting.lambda_sparse_enc1}")
                print(f"lambda_sparse_enc2 = {setting.lambda_sparse_enc2}")
                print("-" * 100)

                history: List[Dict[str, float]] = []
                train_patient_curves = make_patient_curve_store(train_patients)
                test_patient_curves = make_patient_curve_store(test_patients)

                best_test_dice = -1.0
                best_epoch = -1
                best_state = None

                for epoch in range(cfg.epochs):
                    train_stats = train_one_epoch(model, train_loader, optimizer, device, setting)
                    train_eval = evaluate_loader(model, train_eval_loader, device, setting)
                    test_eval = evaluate_loader(model, test_eval_loader, device, setting)

                    for pid in train_patient_curves.keys():
                        train_patient_curves[pid].append(train_eval["patient_dice_by_pid"].get(pid, np.nan))
                    for pid in test_patient_curves.keys():
                        test_patient_curves[pid].append(test_eval["patient_dice_by_pid"].get(pid, np.nan))

                    row = {
                        "epoch": epoch + 1,
                        "train_loss": train_eval["loss"],
                        "test_loss": test_eval["loss"],
                        "train_dice": train_eval["mean_dice"],
                        "test_dice": test_eval["mean_dice"],
                        "train_iou": train_eval["mean_iou"],
                        "test_iou": test_eval["mean_iou"],
                        "train_z_abs": train_eval["mean_z_abs"],
                        "test_z_abs": test_eval["mean_z_abs"],
                        "train_zero_ratio": train_eval["mean_z_zero_ratio"],
                        "test_zero_ratio": test_eval["mean_z_zero_ratio"],
                        "optim_loss": train_stats["loss"],
                        "optim_bce": train_stats["bce"],
                        "optim_dice_loss": train_stats["dice_loss"],
                        "optim_sparse": train_stats["sparse"],
                        "optim_sparse_z_raw": train_stats["sparse_z_raw"],
                        "optim_sparse_f1_raw": train_stats["sparse_f1_raw"],
                        "optim_sparse_f2_raw": train_stats["sparse_f2_raw"],
                    }
                    history.append(row)

                    if test_eval["mean_dice"] > best_test_dice:
                        best_test_dice = test_eval["mean_dice"]
                        best_epoch = epoch + 1
                        if cfg.save_checkpoint:
                            best_state = copy.deepcopy(model.state_dict())

                    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
                        print(
                            f"[seed={seed}] [{norm_setting.name}] [{setting.name}] [Epoch {epoch + 1:03d}/{cfg.epochs}] "
                            f"optim_loss={train_stats['loss']:.4f} "
                            f"train_dice={train_eval['mean_dice']:.4f} "
                            f"test_dice={test_eval['mean_dice']:.4f} "
                            f"|z|_train={train_eval['mean_z_abs']:.4f} "
                            f"|z|_test={test_eval['mean_z_abs']:.4f}"
                        )

                # 保存整体 history
                hist_csv = exp_dir / "history.csv"
                write_log_csv(history, hist_csv)
                save_training_curves(history, str(exp_dir / "train_test_dice_curves.png"))

                # 保存 per-patient 曲线
                write_dict_of_lists_csv(train_patient_curves, exp_dir / "per_patient_train_dice_history.csv")
                write_dict_of_lists_csv(test_patient_curves, exp_dir / "per_patient_test_dice_history.csv")
                save_per_patient_dice_curves(
                    train_patient_curves,
                    str(exp_dir / "per_patient_train_dice_curves.png"),
                    title="Per-patient Train Dice Curves",
                )
                save_per_patient_dice_curves(
                    test_patient_curves,
                    str(exp_dir / "per_patient_test_dice_curves.png"),
                    title="Per-patient Test Dice Curves",
                )

                # 保存最后模型 & 最佳模型（仅诊断）
                if cfg.save_checkpoint:
                    torch.save(model.state_dict(), exp_dir / "last_model.pt")
                    if best_state is not None:
                        torch.save(best_state, exp_dir / "best_test_dice_model_for_diagnosis_only.pt")

                # 最终用最后一个 epoch 的模型做正式测试汇总
                result = evaluate_and_save_predictions(
                    model=model,
                    test_set=test_set,
                    device=device,
                    exp_dir=str(exp_dir),
                    setting=setting,
                    norm_setting=norm_setting,
                )

                result.update({
                    "seed": seed,
                    "norm_name": norm_setting.name,
                    "final_train_dice": history[-1]["train_dice"],
                    "final_test_dice": history[-1]["test_dice"],
                    "best_test_dice": best_test_dice,
                    "best_epoch": best_epoch,
                    "train_patients": " ".join(train_patients),
                    "test_patients": " ".join(test_patients),
                    "n_train_slices": len(train_set),
                    "n_test_slices": len(test_set),
                })
                raw_rows.append(result)

    raw_csv = Path(cfg.out_dir) / "raw_results.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "norm_name",
                "experiment",
                "lambda_sparse_z",
                "lambda_sparse_enc1",
                "lambda_sparse_enc2",
                "n_train_slices",
                "n_test_slices",
                "train_patients",
                "test_patients",
                "mean_iou",
                "std_iou",
                "mean_dice",
                "std_dice",
                "mean_patient_dice",
                "std_patient_dice",
                "mean_z_abs",
                "mean_z_zero_ratio",
                "final_train_dice",
                "final_test_dice",
                "best_test_dice",
                "best_epoch",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped_csv = Path(cfg.out_dir) / "grouped_results.csv"
    save_grouped_results(raw_rows, str(grouped_csv))

    report_txt = Path(cfg.out_dir) / "run_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Single-domain Sparse AE test on SIEMENS__SymphonyTim__37743__T1-TSE\n")
        f.write(f"Domain root          : {cfg.domain_root}\n")
        f.write(f"Train patients       : {' '.join(train_patients)}\n")
        f.write(f"Test patients        : {' '.join(test_patients)}\n")
        f.write(f"Train slices         : {len(train_set)}\n")
        f.write(f"Test slices          : {len(test_set)}\n")
        f.write("Slice policy         : middle three sagittal slices within mask-positive x range\n")
        f.write("SN scan              : no_sn / sn_bottleneck_only / sn_prev_and_bottleneck\n")
        f.write("Visualization        : save all test cases\n")
        f.write("Model selection note : test curves are diagnostic only; do not use them for formal model selection\n")

    print("=" * 100)
    print("All experiments finished.")
    print(f"Output dir      -> {Path(cfg.out_dir).resolve()}")
    print(f"Raw results     -> {raw_csv}")
    print(f"Grouped results -> {grouped_csv}")
    print(f"Report          -> {report_txt}")
    print("=" * 100)


if __name__ == "__main__":
    main()
