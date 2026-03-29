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

# 实验方案
# I. 训练不加稀疏正则的 baseline_no_sparse，作为表征基线。
# II. 用训练好的 baseline 编码 train/test 全部病例，在 f1/f2/f3/z/d3/d2/feat 各层提取前景/背景 token。
# III. 以“训练 5 例张成的子空间能否覆盖测试域”为核心，分别统计：
#     1) 测试前景到训练前景子空间的重构残差比；
#     2) 有效秩 / 稳定秩；
#     3) train-test 子空间主角度；
#     4) train-LOPO 与 test patient-level 覆盖差异。

# 实验结果
# I. 稀疏没有提升泛化，反而降低性能：
#    baseline_no_sparse > full_sparse > z_only_sparse。
# II. 加稀疏后 |z| 明显变小、zero(z) 明显升高，说明主要效果是压窄表示，而不是扩大测试域覆盖。
# III. baseline 的 coverage 分析显示：主要失配集中在前景的中高层表示，而不是背景整体漂移；
#      部分测试病人持续成为困难对象，train-test gap 明显。

# 分析
# I. 当前主瓶颈不是“稀疏不够”或“背景建模不足”，而是 5 例训练病人的前景特征覆盖不全。
# II. 这种覆盖不足在 f3 / z 等中高层被放大，表现为测试前景对子空间重构变差、主角度增大、病人级泛化缺口稳定存在。
# III. 因此，后续重点应放在扩大前景表示覆盖与跨病人泛化，而不是继续单纯增强 L1 稀疏。
# 实验方案
# # I. 训练不加稀疏正则的 baseline_no_sparse，作为表征基线。
# # II. 用训练好的 baseline 编码 train/test 全部病例，在 f1/f2/f3/z/d3/d2/feat 各层提取前景/背景 token。
# # III. 以“训练 5 例张成的子空间能否覆盖测试域”为核心，分别统计：
# #     1) 测试前景到训练前景子空间的重构残差比；
# #     2) 有效秩 / 稳定秩；
# #     3) train-test 子空间主角度；
# #     4) train-LOPO 与 test patient-level 覆盖差异。
#
# # 实验结果
# # I. 稀疏没有提升泛化，反而降低性能：
# #    baseline_no_sparse > full_sparse > z_only_sparse。
# # II. 加稀疏后 |z| 明显变小、zero(z) 明显升高，说明主要效果是压窄表示，而不是扩大测试域覆盖。
# # III. baseline 的 coverage 分析显示：主要失配集中在前景的中高层表示，而不是背景整体漂移；
# #      部分测试病人持续成为困难对象，train-test gap 明显。
#
# # 分析
# # I. 当前主瓶颈不是“稀疏不够”或“背景建模不足”，而是 5 例训练病人的前景特征覆盖不全。
# # II. 这种覆盖不足在 f3 / z 等中高层被放大，表现为测试前景对子空间重构变差、主角度增大、病人级泛化缺口稳定存在。
# # III. 因此，后续重点应放在扩大前景表示覆盖与跨病人泛化，而不是继续单纯增强 L1 稀疏。
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
    out_dir: str = "SparseAE_singleDomainTest/L1_AE"

    # 单域 5-shot：5 个训练病人，其余全部测试
    n_train_patients: int = 5
    split_seed: int = 0          # 固定病人划分
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)  # 只变模型初始化

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- slice selection --------
    # 每个病人只取前景 z 范围中部三张
    num_middle_slices: int = 5

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
    save_vis_n: int = 12
    save_checkpoint: bool = True


cfg = Cfg()
print(cfg)


# =========================================================
# Ablation settings
# =========================================================
@dataclass(frozen=True)
class AblationSetting:
    name: str
    lambda_sparse_z: float
    lambda_sparse_enc1: float
    lambda_sparse_enc2: float


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


def parse_patient_id_from_stem(stem: str) -> str:
    # 例如: 1_t1 -> 1
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


def select_middle_sagittal_indices(mask_zyx: np.ndarray, num_slices: int = 3) -> List[int]:
    """
    mask_zyx: [Z, Y, X]
    取矢状面中部三张：沿 X 轴选切片。
    若 mask 非空，则按前景所在 X 范围取中部三张；
    若 mask 为空，则退化为几何中心三张。
    """
    # 在每个 x 位置上，看整张 sagittal plane 是否有前景
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

    sagittal_indices = select_middle_sagittal_indices(
        msk_zyx,
        num_slices=cfg.num_middle_slices
    )

    out = []
    for x_idx in sagittal_indices:
        # 固定 x，取 sagittal plane: [Z, Y]
        img2d = img_zyx[:, :, x_idx]
        msk2d = msk_zyx[:, :, x_idx]

        msk_bin = binarize_mask(msk2d, cfg.fg_labels)
        img2d = normalize_slice_to_01(
            img2d,
            cfg.input_clip_min,
            cfg.input_clip_max
        )

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
        img_t = torch.from_numpy(item["img"])[None, ...].float()   # [1,H,W]
        msk_t = torch.from_numpy(item["mask"])[None, ...].float()
        meta = {
            "patient_id": item["patient_id"],
            "stem": item["stem"],
            "sagittal_x_index": item["sagittal_x_index"],
        }
        return img_t, msk_t, meta


# =========================================================
# Model: 保留当前 Sparse AE 分割架构
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


class SparseAENet(nn.Module):
    def __init__(self, base_ch: int = 16, latent_ch: int = 64):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.enc1 = ConvBlock(1, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, latent_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(latent_ch, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3, c2)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2, c1)

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1, c1)

        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

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

    # Dice curves
    axes[0].plot(xs, [h["train_dice"] for h in history], label="train_dice")
    axes[0].plot(xs, [h["test_dice"] for h in history], label="test_dice")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice")
    axes[0].set_title("Train/Test Dice Curves")
    axes[0].legend()

    # Loss curves
    axes[1].plot(xs, [h["train_loss"] for h in history], label="train_loss")
    axes[1].plot(xs, [h["test_loss"] for h in history], label="test_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train/Test Loss Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
) -> Dict[str, float]:
    ensure_dir(exp_dir)
    vis_dir = os.path.join(exp_dir, "vis_test")
    ensure_dir(vis_dir)

    rows = []
    ious = []
    dices = []
    patient_to_dices = defaultdict(list)

    print("=" * 80)
    print(f"Testing [{setting.name}] ...")
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

        if i < cfg.save_vis_n:
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
    print(f"[{setting.name}] Slice mean IoU      = {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"[{setting.name}] Slice mean Dice     = {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"[{setting.name}] Patient mean Dice   = {mean_patient_dice:.4f} ± {std_patient_dice:.4f}")
    print(f"[{setting.name}] Mean |z|            = {mean_z:.4f}")
    print(f"[{setting.name}] Mean zero(z)        = {mean_zero:.4f}")
    print("=" * 80)

    metrics_path = os.path.join(exp_dir, "test_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else ["case_id", "iou", "dice"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(exp_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment          = {setting.name}\n")
        f.write(f"lambda_sparse_z     = {setting.lambda_sparse_z}\n")
        f.write(f"lambda_sparse_enc1  = {setting.lambda_sparse_enc1}\n")
        f.write(f"lambda_sparse_enc2  = {setting.lambda_sparse_enc2}\n")
        f.write(f"Slice Mean IoU      = {mean_iou:.6f} +- {std_iou:.6f}\n")
        f.write(f"Slice Mean Dice     = {mean_dice:.6f} +- {std_dice:.6f}\n")
        f.write(f"Patient Mean Dice   = {mean_patient_dice:.6f} +- {std_patient_dice:.6f}\n")
        f.write(f"Mean |z|            = {mean_z:.6f}\n")
        f.write(f"Mean zero(z)        = {mean_zero:.6f}\n")
        f.write("NOTE                = test curve is for diagnosis only; do not use it for formal model selection.\n")

    return {
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
        key = r["experiment"]
        grouped[key].append(r)

    summary_rows = []
    for experiment, rows in grouped.items():
        final_test_dice = np.array([r["final_test_dice"] for r in rows], dtype=np.float32)
        best_test_dice = np.array([r["best_test_dice"] for r in rows], dtype=np.float32)
        final_patient_dice = np.array([r["mean_patient_dice"] for r in rows], dtype=np.float32)
        final_iou = np.array([r["mean_iou"] for r in rows], dtype=np.float32)

        summary_rows.append({
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

    # -----------------------------------------------------
    # 1) collect volumes and split by patient
    # -----------------------------------------------------
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
    print("=======================================================")
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

        for setting in ablation_settings:
            exp_dir = seed_dir / setting.name
            ensure_dir(exp_dir)

            set_seed(seed)  # 保证相同 seed 下三种配置有相同初始化

            model = SparseAENet(base_ch=cfg.base_ch, latent_ch=cfg.latent_ch).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

            print("-" * 100)
            print(f"Running: seed={seed}, exp={setting.name}")
            print(f"lambda_sparse_z    = {setting.lambda_sparse_z}")
            print(f"lambda_sparse_enc1 = {setting.lambda_sparse_enc1}")
            print(f"lambda_sparse_enc2 = {setting.lambda_sparse_enc2}")
            print("-" * 100)

            history: List[Dict[str, float]] = []
            best_test_dice = -1.0
            best_epoch = -1
            best_state = None

            for epoch in range(cfg.epochs):
                train_stats = train_one_epoch(model, train_loader, optimizer, device, setting)
                train_eval = evaluate_loader(model, train_eval_loader, device, setting)
                test_eval = evaluate_loader(model, test_eval_loader, device, setting)

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
                        f"[seed={seed}] [{setting.name}] [Epoch {epoch + 1:03d}/{cfg.epochs}] "
                        f"optim_loss={train_stats['loss']:.4f} "
                        f"train_dice={train_eval['mean_dice']:.4f} "
                        f"test_dice={test_eval['mean_dice']:.4f} "
                        f"|z|_train={train_eval['mean_z_abs']:.4f} "
                        f"|z|_test={test_eval['mean_z_abs']:.4f}"
                    )

            # 保存 history
            hist_csv = exp_dir / "history.csv"
            write_log_csv(history, hist_csv)
            save_training_curves(history, str(exp_dir / "train_test_dice_curves.png"))

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
            )

            result.update({
                "seed": seed,
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

    # 保存总说明
    report_txt = Path(cfg.out_dir) / "run_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Single-domain Sparse AE test on SIEMENS__SymphonyTim__37743__T1-TSE\n")
        f.write(f"Domain root          : {cfg.domain_root}\n")
        f.write(f"Train patients       : {' '.join(train_patients)}\n")
        f.write(f"Test patients        : {' '.join(test_patients)}\n")
        f.write(f"Train slices         : {len(train_set)}\n")
        f.write(f"Test slices          : {len(test_set)}\n")
        f.write("Slice policy         : middle three slices within mask-positive z range\n")
        f.write("Model selection note : test curves are diagnostic only, not for formal model selection\n")

    print("=" * 100)
    print("All experiments finished.")
    print(f"Output dir      -> {Path(cfg.out_dir).resolve()}")
    print(f"Raw results     -> {raw_csv}")
    print(f"Grouped results -> {grouped_csv}")
    print(f"Report          -> {report_txt}")
    print("=" * 100)

# =========================================================
# Coverage analysis for baseline_no_sparse
# 粘贴位置：放在现有 if __name__ == "__main__": main() 之前
# =========================================================

@dataclass
class CoverageDiagCfg:
    enabled: bool = True

    # 只分析 baseline 模型
    experiment_name: str = "baseline_no_sparse"

    # "last" 或 "best"
    checkpoint_kind: str = "last"

    # 全层覆盖
    layer_names: Tuple[str, ...] = ("f1", "f2", "f3", "z", "d3", "d2", "feat")

    # 前景 / 背景双对照
    token_classes: Tuple[str, ...] = ("fg", "bg")

    # 每张切片每个类别保留多少 token
    # 数值越大越全，但内存越高
    max_fg_tokens_per_case: int = 256
    max_bg_tokens_per_case: int = 256

    # 子空间配置
    channel_standardize: bool = True
    energy_threshold: float = 0.95
    max_components: int = 32
    angle_components: int = 8
    eps: float = 1e-8

    # 保存项
    save_sampled_tokens_npz: bool = True
    save_subspace_npz: bool = True

    # 输出目录
    out_dir_name: str = "coverage_analysis"

    # 随机种子偏移
    rng_offset: int = 12345

    verbose: bool = True


coverage_cfg = CoverageDiagCfg()


# ---------------------------------------------------------
# Small utils
# ---------------------------------------------------------
def _stable_hash_u32(*parts) -> int:
    """
    稳定 32-bit hash，避免 Python 内建 hash 的跨进程随机性。
    """
    s = "||".join(map(str, parts))
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _get_checkpoint_path_for_seed(seed: int) -> Path:
    exp_dir = Path(cfg.out_dir) / f"seed_{seed}" / coverage_cfg.experiment_name
    if coverage_cfg.checkpoint_kind == "best":
        return exp_dir / "best_test_dice_model_for_diagnosis_only.pt"
    return exp_dir / "last_model.pt"


def _select_layer_tensor(layer_name: str, z: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    if layer_name == "z":
        return z
    if layer_name in aux:
        return aux[layer_name]
    raise KeyError(f"Unknown layer_name: {layer_name}")


def _stack_feature_list(arr_list: List[np.ndarray], feat_dim: int) -> np.ndarray:
    if len(arr_list) == 0:
        return np.zeros((0, feat_dim), dtype=np.float32)
    return np.concatenate([a.astype(np.float32, copy=False) for a in arr_list], axis=0)


def _effective_rank_from_singulars(s: np.ndarray, eps: float = 1e-8) -> float:
    if s.size == 0:
        return 0.0
    p = s / (np.sum(s) + eps)
    return float(np.exp(-np.sum(p * np.log(p + eps))))


def _stable_rank_from_singulars(s: np.ndarray, eps: float = 1e-8) -> float:
    if s.size == 0:
        return 0.0
    return float(np.sum(s ** 2) / (float(s[0] ** 2) + eps))


def _choose_k_from_singulars(
    s: np.ndarray,
    energy_threshold: float = 0.95,
    max_components: int = 32,
    eps: float = 1e-8,
) -> int:
    if s.size == 0:
        return 0
    e = s ** 2
    cum = np.cumsum(e) / (np.sum(e) + eps)
    k = int(np.searchsorted(cum, energy_threshold) + 1)
    k = min(k, max_components, s.size)
    return int(k)


def _matrix_rank_from_singulars(s: np.ndarray, eps: float = 1e-8) -> int:
    if s.size == 0:
        return 0
    thr = float(np.max(s)) * eps
    return int(np.sum(s > thr))


def _pad_angles(angles_deg: np.ndarray, k: int) -> np.ndarray:
    out = np.full((k,), np.nan, dtype=np.float32)
    if angles_deg.size > 0:
        m = min(k, angles_deg.size)
        out[:m] = angles_deg[:m].astype(np.float32)
    return out


# ---------------------------------------------------------
# Data rebuild (analysis only)
# ---------------------------------------------------------
def rebuild_train_test_sets_for_analysis() -> Tuple[Dataset, Dataset, List[str], List[str]]:
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

    return train_set, test_set, train_patients, test_patients


# ---------------------------------------------------------
# Feature collection: one pass gets all layers
# ---------------------------------------------------------
@torch.no_grad()
def collect_all_layer_tokens_from_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    split_name: str,
    seed: int,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    一次遍历 dataset，一次 forward，拿到所有层的前景/背景 token。
    返回:
      storage[layer] = {
          "feat_dim": int,
          "fg_all": [np.ndarray, ...],
          "bg_all": [np.ndarray, ...],
          "fg_by_patient": defaultdict(list),
          "bg_by_patient": defaultdict(list),
      }
      manifest_rows: per-case per-layer counts
    """
    model.eval()

    storage = {}
    for layer_name in coverage_cfg.layer_names:
        storage[layer_name] = {
            "feat_dim": None,
            "fg_all": [],
            "bg_all": [],
            "fg_by_patient": defaultdict(list),
            "bg_by_patient": defaultdict(list),
        }

    manifest_rows: List[Dict] = []

    for i in range(len(dataset)):
        img_t, mask_t, meta = dataset[i]
        img = img_t.unsqueeze(0).to(device)    # [1,1,H,W]
        mask = mask_t.unsqueeze(0).to(device)  # [1,1,H,W]

        seg_logits, z, aux = model(img)

        patient_id = meta["patient_id"]
        stem = meta["stem"]
        sx = meta["sagittal_x_index"]

        for layer_name in coverage_cfg.layer_names:
            feat = _select_layer_tensor(layer_name, z, aux)  # [1,C,h,w]
            _, C, h, w = feat.shape

            if storage[layer_name]["feat_dim"] is None:
                storage[layer_name]["feat_dim"] = int(C)

            # 下采样 GT mask 到该层分辨率
            mask_small = F.interpolate(mask, size=(h, w), mode="nearest")
            fg_idx = (mask_small[0, 0] > 0.5).reshape(-1).cpu().numpy()
            bg_idx = (~(mask_small[0, 0] > 0.5)).reshape(-1).cpu().numpy()

            # [1,C,h,w] -> [h*w, C]
            feat_hw_c = feat[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()

            # ---------- FG ----------
            fg_all_idx = np.where(fg_idx)[0]
            fg_total = int(fg_all_idx.size)

            if fg_total > 0:
                fg_keep = min(fg_total, coverage_cfg.max_fg_tokens_per_case)
                fg_rng = np.random.default_rng(
                    _stable_hash_u32(
                        coverage_cfg.rng_offset,
                        seed,
                        split_name,
                        "fg",
                        layer_name,
                        patient_id,
                        stem,
                        sx,
                    )
                )
                if fg_total > fg_keep:
                    fg_sel = fg_rng.choice(fg_all_idx, size=fg_keep, replace=False)
                else:
                    fg_sel = fg_all_idx
                fg_feat = feat_hw_c[fg_sel].astype(np.float16, copy=False)
                storage[layer_name]["fg_all"].append(fg_feat)
                storage[layer_name]["fg_by_patient"][patient_id].append(fg_feat)
                fg_kept = int(fg_feat.shape[0])
            else:
                fg_kept = 0

            # ---------- BG ----------
            bg_all_idx = np.where(bg_idx)[0]
            bg_total = int(bg_all_idx.size)

            if bg_total > 0:
                bg_keep = min(bg_total, coverage_cfg.max_bg_tokens_per_case)
                bg_rng = np.random.default_rng(
                    _stable_hash_u32(
                        coverage_cfg.rng_offset,
                        seed,
                        split_name,
                        "bg",
                        layer_name,
                        patient_id,
                        stem,
                        sx,
                    )
                )
                if bg_total > bg_keep:
                    bg_sel = bg_rng.choice(bg_all_idx, size=bg_keep, replace=False)
                else:
                    bg_sel = bg_all_idx
                bg_feat = feat_hw_c[bg_sel].astype(np.float16, copy=False)
                storage[layer_name]["bg_all"].append(bg_feat)
                storage[layer_name]["bg_by_patient"][patient_id].append(bg_feat)
                bg_kept = int(bg_feat.shape[0])
            else:
                bg_kept = 0

            manifest_rows.append({
                "seed": seed,
                "split": split_name,
                "patient_id": patient_id,
                "stem": stem,
                "sagittal_x_index": sx,
                "layer": layer_name,
                "feature_dim": int(C),
                "fg_total_tokens": fg_total,
                "fg_kept_tokens": fg_kept,
                "bg_total_tokens": bg_total,
                "bg_kept_tokens": bg_kept,
            })

        if coverage_cfg.verbose and ((i + 1) % 20 == 0 or i == len(dataset) - 1):
            print(f"[coverage][seed={seed}] [{split_name}] collected {i + 1}/{len(dataset)} cases")

    return storage, manifest_rows


# ---------------------------------------------------------
# Subspace fit / eval
# ---------------------------------------------------------
def fit_subspace_from_train_matrix(X_train: np.ndarray) -> Dict:
    """
    用 train 矩阵拟合子空间，返回:
      mean, scale, basis, singular_values, k, erank, srank, exact_rank
    """
    eps = coverage_cfg.eps
    out = {
        "valid": False,
        "mean": None,
        "scale": None,
        "basis": np.zeros((0, 0), dtype=np.float32),
        "singular_values": np.zeros((0,), dtype=np.float32),
        "k": 0,
        "erank": 0.0,
        "srank": 0.0,
        "exact_rank": 0,
        "feature_dim": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
    }

    if X_train.ndim != 2 or X_train.shape[0] < 2 or X_train.shape[1] < 1:
        return out

    mu = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = X_train - mu

    if coverage_cfg.channel_standardize:
        sc = Xc.std(axis=0, keepdims=True).astype(np.float32)
        sc = np.where(sc < eps, 1.0, sc)
    else:
        sc = np.ones((1, X_train.shape[1]), dtype=np.float32)

    Xn = Xc / sc

    # N x C, C <= 64，直接 full_matrices=False 即可
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)

    exact_rank = _matrix_rank_from_singulars(S, eps=eps)
    k = _choose_k_from_singulars(
        S,
        energy_threshold=coverage_cfg.energy_threshold,
        max_components=min(coverage_cfg.max_components, Xn.shape[1]),
        eps=eps,
    )

    basis = Vt[:k].T.astype(np.float32, copy=False) if k > 0 else np.zeros((Xn.shape[1], 0), dtype=np.float32)

    out.update({
        "valid": True,
        "mean": mu,
        "scale": sc.astype(np.float32),
        "basis": basis,
        "singular_values": S.astype(np.float32),
        "k": int(k),
        "erank": _effective_rank_from_singulars(S, eps=eps),
        "srank": _stable_rank_from_singulars(S, eps=eps),
        "exact_rank": int(exact_rank),
        "feature_dim": int(X_train.shape[1]),
    })
    return out


def evaluate_matrix_against_ref_subspace(X: np.ndarray, ref_fit: Dict) -> Dict:
    """
    用 ref_fit (train 子空间) 去投影 X，返回残差/解释率。
    """
    eps = coverage_cfg.eps
    out = {
        "n_tokens": int(X.shape[0]) if X.ndim == 2 else 0,
        "mean_total_energy": np.nan,
        "mean_residual": np.nan,
        "median_residual": np.nan,
        "std_residual": np.nan,
        "explained_fraction": np.nan,
    }

    if X.ndim != 2 or X.shape[0] < 1 or not ref_fit["valid"]:
        return out

    mu = ref_fit["mean"]
    sc = ref_fit["scale"]
    B = ref_fit["basis"]   # [C, k]

    Xn = (X - mu) / sc
    total_energy = np.sum(Xn ** 2, axis=1)

    if B.shape[1] == 0:
        residual = total_energy.copy()
    else:
        proj = (Xn @ B) @ B.T
        residual = np.sum((Xn - proj) ** 2, axis=1)

    mean_total = float(np.mean(total_energy)) if total_energy.size > 0 else np.nan
    mean_resid = float(np.mean(residual)) if residual.size > 0 else np.nan
    explained = 1.0 - mean_resid / (mean_total + eps) if np.isfinite(mean_total) else np.nan

    out.update({
        "mean_total_energy": mean_total,
        "mean_residual": mean_resid,
        "median_residual": float(np.median(residual)) if residual.size > 0 else np.nan,
        "std_residual": float(np.std(residual)) if residual.size > 0 else np.nan,
        "explained_fraction": float(explained),
    })
    return out


def fit_subspace_for_angle(X: np.ndarray) -> Dict:
    """
    为主角度准备的独立子空间拟合：各自按自身均值/方差中心化。
    """
    return fit_subspace_from_train_matrix(X)


def principal_angles_from_two_fits(fit_a: Dict, fit_b: Dict, k_angle: int) -> np.ndarray:
    """
    返回 degree，长度固定为 k_angle，缺的用 nan 补齐。
    """
    if (not fit_a["valid"]) or (not fit_b["valid"]):
        return np.full((k_angle,), np.nan, dtype=np.float32)

    Ba = fit_a["basis"]
    Bb = fit_b["basis"]

    if Ba.shape[1] == 0 or Bb.shape[1] == 0:
        return np.full((k_angle,), np.nan, dtype=np.float32)

    m = min(Ba.shape[1], Bb.shape[1], k_angle)
    if m <= 0:
        return np.full((k_angle,), np.nan, dtype=np.float32)

    Ma = Ba[:, :m]
    Mb = Bb[:, :m]

    s = np.linalg.svd(Ma.T @ Mb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s) * 180.0 / np.pi
    return _pad_angles(angles.astype(np.float32), k_angle)


# ---------------------------------------------------------
# Global metrics per layer / class
# ---------------------------------------------------------
def compute_global_metrics_for_one_layer_class(
    seed: int,
    layer_name: str,
    token_class: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[Dict, Dict, Dict, np.ndarray]:
    """
    token_class: "fg" or "bg"
    返回:
      global_row, fit_train, fit_test_for_angle, angles
    """
    row = {
        "seed": seed,
        "layer": layer_name,
        "token_class": token_class,
        "feature_dim": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
        "n_train_tokens": int(X_train.shape[0]) if X_train.ndim == 2 else 0,
        "n_test_tokens": int(X_test.shape[0]) if X_test.ndim == 2 else 0,
        "status": "ok",
    }

    for i in range(1, coverage_cfg.angle_components + 1):
        row[f"angle_deg_{i}"] = np.nan

    if (
        X_train.ndim != 2
        or X_test.ndim != 2
        or X_train.shape[0] < 2
        or X_test.shape[0] < 2
        or X_train.shape[1] < 1
    ):
        row["status"] = "insufficient_tokens"
        return row, {"valid": False}, {"valid": False}, np.full((coverage_cfg.angle_components,), np.nan, dtype=np.float32)

    # 训练子空间
    fit_train = fit_subspace_from_train_matrix(X_train)

    # 测试子空间（只用于主角度）
    fit_test = fit_subspace_for_angle(X_test)

    # train -> train
    train_eval = evaluate_matrix_against_ref_subspace(X_train, fit_train)

    # train -> test
    test_eval = evaluate_matrix_against_ref_subspace(X_test, fit_train)

    # test 自己的谱，仅做对照
    fit_test_self = fit_subspace_from_train_matrix(X_test)

    # 主角度
    angles = principal_angles_from_two_fits(
        fit_train,
        fit_test,
        k_angle=coverage_cfg.angle_components,
    )

    # 残差比
    eps = coverage_cfg.eps
    resid_ratio_mean = (
        test_eval["mean_residual"] / (train_eval["mean_residual"] + eps)
        if np.isfinite(train_eval["mean_residual"]) and np.isfinite(test_eval["mean_residual"])
        else np.nan
    )

    resid_ratio_median = (
        test_eval["median_residual"] / (train_eval["median_residual"] + eps)
        if np.isfinite(train_eval["median_residual"]) and np.isfinite(test_eval["median_residual"])
        else np.nan
    )

    row.update({
        "train_exact_rank": fit_train["exact_rank"],
        "train_erank": fit_train["erank"],
        "train_srank": fit_train["srank"],
        "train_k95": fit_train["k"],

        "test_exact_rank": fit_test_self["exact_rank"] if fit_test_self["valid"] else 0,
        "test_erank": fit_test_self["erank"] if fit_test_self["valid"] else np.nan,
        "test_srank": fit_test_self["srank"] if fit_test_self["valid"] else np.nan,
        "test_k95": fit_test_self["k"] if fit_test_self["valid"] else 0,

        "train_mean_total_energy": train_eval["mean_total_energy"],
        "train_mean_residual": train_eval["mean_residual"],
        "train_median_residual": train_eval["median_residual"],
        "train_std_residual": train_eval["std_residual"],
        "train_explained_fraction": train_eval["explained_fraction"],

        "test_mean_total_energy_to_train_subspace": test_eval["mean_total_energy"],
        "test_mean_residual_to_train_subspace": test_eval["mean_residual"],
        "test_median_residual_to_train_subspace": test_eval["median_residual"],
        "test_std_residual_to_train_subspace": test_eval["std_residual"],
        "test_explained_fraction_to_train_subspace": test_eval["explained_fraction"],

        "residual_ratio_mean_test_over_train": resid_ratio_mean,
        "residual_ratio_median_test_over_train": resid_ratio_median,

        "angle_mean_deg": float(np.nanmean(angles)) if np.any(np.isfinite(angles)) else np.nan,
        "angle_max_deg": float(np.nanmax(angles)) if np.any(np.isfinite(angles)) else np.nan,
    })

    for i in range(coverage_cfg.angle_components):
        row[f"angle_deg_{i + 1}"] = float(angles[i]) if np.isfinite(angles[i]) else np.nan

    return row, fit_train, fit_test, angles


# ---------------------------------------------------------
# Patient metrics
# ---------------------------------------------------------
def compute_patient_metrics_against_train_subspace(
    seed: int,
    split_name: str,
    layer_name: str,
    token_class: str,
    patient_feature_dict: Dict[str, List[np.ndarray]],
    feat_dim: int,
    train_fit: Dict,
    train_global_mean_residual: float,
) -> List[Dict]:
    rows = []

    for patient_id in sorted(patient_feature_dict.keys()):
        Xp = _stack_feature_list(patient_feature_dict[patient_id], feat_dim)

        row = {
            "seed": seed,
            "split": split_name,
            "layer": layer_name,
            "token_class": token_class,
            "patient_id": patient_id,
            "feature_dim": feat_dim,
            "n_tokens": int(Xp.shape[0]),
            "status": "ok",
        }
        for i in range(1, coverage_cfg.angle_components + 1):
            row[f"angle_deg_{i}"] = np.nan

        if Xp.shape[0] < 2 or feat_dim < 1 or not train_fit["valid"]:
            row["status"] = "insufficient_tokens"
            rows.append(row)
            continue

        evalp = evaluate_matrix_against_ref_subspace(Xp, train_fit)
        fitp = fit_subspace_for_angle(Xp)
        angles = principal_angles_from_two_fits(
            train_fit,
            fitp,
            k_angle=coverage_cfg.angle_components,
        )

        eps = coverage_cfg.eps
        resid_ratio = (
            evalp["mean_residual"] / (train_global_mean_residual + eps)
            if np.isfinite(evalp["mean_residual"]) and np.isfinite(train_global_mean_residual)
            else np.nan
        )

        row.update({
            "exact_rank_self": fitp["exact_rank"] if fitp["valid"] else 0,
            "erank_self": fitp["erank"] if fitp["valid"] else np.nan,
            "srank_self": fitp["srank"] if fitp["valid"] else np.nan,
            "k95_self": fitp["k"] if fitp["valid"] else 0,

            "mean_total_energy_to_train_subspace": evalp["mean_total_energy"],
            "mean_residual_to_train_subspace": evalp["mean_residual"],
            "median_residual_to_train_subspace": evalp["median_residual"],
            "std_residual_to_train_subspace": evalp["std_residual"],
            "explained_fraction_to_train_subspace": evalp["explained_fraction"],

            "residual_ratio_vs_train_global_mean": resid_ratio,

            "angle_mean_deg": float(np.nanmean(angles)) if np.any(np.isfinite(angles)) else np.nan,
            "angle_max_deg": float(np.nanmax(angles)) if np.any(np.isfinite(angles)) else np.nan,
        })

        for i in range(coverage_cfg.angle_components):
            row[f"angle_deg_{i + 1}"] = float(angles[i]) if np.isfinite(angles[i]) else np.nan

        rows.append(row)

    return rows


def compute_train_lopo_metrics(
    seed: int,
    layer_name: str,
    token_class: str,
    patient_feature_dict: Dict[str, List[np.ndarray]],
    feat_dim: int,
) -> List[Dict]:
    """
    训练病人 leave-one-patient-out:
      用其余 train 病人的 token 拟合 ref 子空间，
      看被留出的 train 病人能否被覆盖。
    """
    rows = []
    all_train_pids = sorted(patient_feature_dict.keys())

    for held_pid in all_train_pids:
        X_hold = _stack_feature_list(patient_feature_dict[held_pid], feat_dim)

        ref_list = []
        for pid in all_train_pids:
            if pid == held_pid:
                continue
            ref_list.extend(patient_feature_dict[pid])

        X_ref = _stack_feature_list(ref_list, feat_dim)

        row = {
            "seed": seed,
            "split": "train_lopo",
            "layer": layer_name,
            "token_class": token_class,
            "patient_id": held_pid,
            "feature_dim": feat_dim,
            "n_hold_tokens": int(X_hold.shape[0]),
            "n_ref_tokens": int(X_ref.shape[0]),
            "status": "ok",
        }
        for i in range(1, coverage_cfg.angle_components + 1):
            row[f"angle_deg_{i}"] = np.nan

        if X_hold.shape[0] < 2 or X_ref.shape[0] < 2 or feat_dim < 1:
            row["status"] = "insufficient_tokens"
            rows.append(row)
            continue

        fit_ref = fit_subspace_from_train_matrix(X_ref)
        eval_hold = evaluate_matrix_against_ref_subspace(X_hold, fit_ref)
        fit_hold = fit_subspace_for_angle(X_hold)

        angles = principal_angles_from_two_fits(
            fit_ref,
            fit_hold,
            k_angle=coverage_cfg.angle_components,
        )

        ref_self_eval = evaluate_matrix_against_ref_subspace(X_ref, fit_ref)
        ref_mean_resid = ref_self_eval["mean_residual"]

        eps = coverage_cfg.eps
        resid_ratio = (
            eval_hold["mean_residual"] / (ref_mean_resid + eps)
            if np.isfinite(eval_hold["mean_residual"]) and np.isfinite(ref_mean_resid)
            else np.nan
        )

        row.update({
            "ref_exact_rank": fit_ref["exact_rank"] if fit_ref["valid"] else 0,
            "ref_erank": fit_ref["erank"] if fit_ref["valid"] else np.nan,
            "ref_srank": fit_ref["srank"] if fit_ref["valid"] else np.nan,
            "ref_k95": fit_ref["k"] if fit_ref["valid"] else 0,

            "hold_exact_rank": fit_hold["exact_rank"] if fit_hold["valid"] else 0,
            "hold_erank": fit_hold["erank"] if fit_hold["valid"] else np.nan,
            "hold_srank": fit_hold["srank"] if fit_hold["valid"] else np.nan,
            "hold_k95": fit_hold["k"] if fit_hold["valid"] else 0,

            "ref_mean_residual_self": ref_mean_resid,

            "hold_mean_total_energy_to_ref_subspace": eval_hold["mean_total_energy"],
            "hold_mean_residual_to_ref_subspace": eval_hold["mean_residual"],
            "hold_median_residual_to_ref_subspace": eval_hold["median_residual"],
            "hold_std_residual_to_ref_subspace": eval_hold["std_residual"],
            "hold_explained_fraction_to_ref_subspace": eval_hold["explained_fraction"],

            "residual_ratio_hold_vs_ref_mean": resid_ratio,

            "angle_mean_deg": float(np.nanmean(angles)) if np.any(np.isfinite(angles)) else np.nan,
            "angle_max_deg": float(np.nanmax(angles)) if np.any(np.isfinite(angles)) else np.nan,
        })

        for i in range(coverage_cfg.angle_components):
            row[f"angle_deg_{i + 1}"] = float(angles[i]) if np.isfinite(angles[i]) else np.nan

        rows.append(row)

    return rows


# ---------------------------------------------------------
# Save NPZ
# ---------------------------------------------------------
def save_sampled_tokens_npz(
    train_storage: Dict[str, Dict],
    test_storage: Dict[str, Dict],
    out_path: str,
):
    payload = {}

    for split_name, storage in [("train", train_storage), ("test", test_storage)]:
        for layer_name in coverage_cfg.layer_names:
            feat_dim = int(storage[layer_name]["feat_dim"] or 0)
            for cls_name in coverage_cfg.token_classes:
                X = _stack_feature_list(storage[layer_name][f"{cls_name}_all"], feat_dim)
                payload[f"{split_name}_{layer_name}_{cls_name}_X"] = X.astype(np.float16)

    np.savez_compressed(out_path, **payload)


def save_subspace_npz(
    global_fit_payload: Dict[str, Dict],
    angle_payload: Dict[str, np.ndarray],
    out_path: str,
):
    payload = {}

    for key, fit_dict in global_fit_payload.items():
        # key = "seed{seed}_{layer}_{cls}_{train/test}"
        if fit_dict.get("valid", False):
            payload[f"{key}_mean"] = fit_dict["mean"].astype(np.float32)
            payload[f"{key}_scale"] = fit_dict["scale"].astype(np.float32)
            payload[f"{key}_basis"] = fit_dict["basis"].astype(np.float32)
            payload[f"{key}_singular_values"] = fit_dict["singular_values"].astype(np.float32)

    for key, angles in angle_payload.items():
        payload[f"{key}_angles_deg"] = angles.astype(np.float32)

    np.savez_compressed(out_path, **payload)


# ---------------------------------------------------------
# Seed-level analysis
# ---------------------------------------------------------
def run_baseline_coverage_analysis_for_one_seed(
    seed: int,
    train_set: Dataset,
    test_set: Dataset,
    train_patients: List[str],
    test_patients: List[str],
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    ckpt_path = _get_checkpoint_path_for_seed(seed)
    if not ckpt_path.exists():
        print(f"[coverage][seed={seed}] checkpoint not found: {ckpt_path}")
        return [], [], []

    seed_out_dir = Path(cfg.out_dir) / coverage_cfg.out_dir_name / coverage_cfg.experiment_name / f"seed_{seed}"
    ensure_dir(seed_out_dir)

    # 载入 baseline 模型
    model = SparseAENet(base_ch=cfg.base_ch, latent_ch=cfg.latent_ch).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("=" * 100)
    print(f"[coverage] seed={seed} | ckpt={ckpt_path}")
    print(f"[coverage] train_patients={train_patients}")
    print(f"[coverage] test_patients ={test_patients}")
    print("=" * 100)

    # 一次遍历 train / test，各 split 只 forward 一遍
    train_storage, train_manifest_rows = collect_all_layer_tokens_from_dataset(
        model=model,
        dataset=train_set,
        device=device,
        split_name="train",
        seed=seed,
    )
    test_storage, test_manifest_rows = collect_all_layer_tokens_from_dataset(
        model=model,
        dataset=test_set,
        device=device,
        split_name="test",
        seed=seed,
    )

    # 保存 token 清单
    case_manifest_rows = train_manifest_rows + test_manifest_rows
    write_log_csv(case_manifest_rows, seed_out_dir / "case_token_manifest.csv")

    if coverage_cfg.save_sampled_tokens_npz:
        save_sampled_tokens_npz(
            train_storage=train_storage,
            test_storage=test_storage,
            out_path=str(seed_out_dir / "sampled_tokens_all_layers.npz"),
        )

    global_rows: List[Dict] = []
    patient_rows: List[Dict] = []
    lopo_rows: List[Dict] = []

    global_fit_payload = {}
    angle_payload = {}

    for layer_name in coverage_cfg.layer_names:
        feat_dim = int(train_storage[layer_name]["feat_dim"] or 0)

        for cls_name in coverage_cfg.token_classes:
            X_train = _stack_feature_list(
                train_storage[layer_name][f"{cls_name}_all"],
                feat_dim,
            )
            X_test = _stack_feature_list(
                test_storage[layer_name][f"{cls_name}_all"],
                feat_dim,
            )

            global_row, fit_train, fit_test, angles = compute_global_metrics_for_one_layer_class(
                seed=seed,
                layer_name=layer_name,
                token_class=cls_name,
                X_train=X_train,
                X_test=X_test,
            )
            global_rows.append(global_row)

            key_prefix = f"seed{seed}_{layer_name}_{cls_name}"
            global_fit_payload[f"{key_prefix}_train"] = fit_train
            global_fit_payload[f"{key_prefix}_test"] = fit_test
            angle_payload[f"{key_prefix}_train_vs_test"] = angles

            # 测试病人逐病人
            if fit_train.get("valid", False):
                patient_rows.extend(
                    compute_patient_metrics_against_train_subspace(
                        seed=seed,
                        split_name="test",
                        layer_name=layer_name,
                        token_class=cls_name,
                        patient_feature_dict=test_storage[layer_name][f"{cls_name}_by_patient"],
                        feat_dim=feat_dim,
                        train_fit=fit_train,
                        train_global_mean_residual=global_row.get("train_mean_residual", np.nan),
                    )
                )

                # 训练病人逐病人（对 train 子空间）
                patient_rows.extend(
                    compute_patient_metrics_against_train_subspace(
                        seed=seed,
                        split_name="train",
                        layer_name=layer_name,
                        token_class=cls_name,
                        patient_feature_dict=train_storage[layer_name][f"{cls_name}_by_patient"],
                        feat_dim=feat_dim,
                        train_fit=fit_train,
                        train_global_mean_residual=global_row.get("train_mean_residual", np.nan),
                    )
                )

            # 训练病人 LOPO
            lopo_rows.extend(
                compute_train_lopo_metrics(
                    seed=seed,
                    layer_name=layer_name,
                    token_class=cls_name,
                    patient_feature_dict=train_storage[layer_name][f"{cls_name}_by_patient"],
                    feat_dim=feat_dim,
                )
            )

    # 保存 CSV
    write_log_csv(global_rows, seed_out_dir / "global_metrics.csv")
    write_log_csv(patient_rows, seed_out_dir / "patient_metrics.csv")
    write_log_csv(lopo_rows, seed_out_dir / "train_lopo_metrics.csv")

    # 保存谱 / basis / 主角度
    if coverage_cfg.save_subspace_npz:
        save_subspace_npz(
            global_fit_payload=global_fit_payload,
            angle_payload=angle_payload,
            out_path=str(seed_out_dir / "subspace_payload.npz"),
        )

    # seed 级说明
    with open(seed_out_dir / "analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(f"seed                 = {seed}\n")
        f.write(f"experiment           = {coverage_cfg.experiment_name}\n")
        f.write(f"checkpoint_kind      = {coverage_cfg.checkpoint_kind}\n")
        f.write(f"checkpoint_path      = {str(ckpt_path)}\n")
        f.write(f"train_patients       = {' '.join(train_patients)}\n")
        f.write(f"test_patients        = {' '.join(test_patients)}\n")
        f.write(f"layer_names          = {' '.join(coverage_cfg.layer_names)}\n")
        f.write(f"max_fg_tokens_case   = {coverage_cfg.max_fg_tokens_per_case}\n")
        f.write(f"max_bg_tokens_case   = {coverage_cfg.max_bg_tokens_per_case}\n")
        f.write(f"energy_threshold     = {coverage_cfg.energy_threshold}\n")
        f.write(f"max_components       = {coverage_cfg.max_components}\n")
        f.write(f"angle_components     = {coverage_cfg.angle_components}\n")
        f.write(f"channel_standardize  = {coverage_cfg.channel_standardize}\n")

    return global_rows, patient_rows, lopo_rows


# ---------------------------------------------------------
# Group summary
# ---------------------------------------------------------
def group_numeric_rows(
    rows: List[Dict],
    group_keys: List[str],
) -> List[Dict]:
    if len(rows) == 0:
        return []

    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, None) for k in group_keys)
        grouped[key].append(r)

    # 找数值字段
    numeric_keys = set()
    for r in rows:
        for k, v in r.items():
            if k in group_keys:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_keys.add(k)

    numeric_keys = sorted(list(numeric_keys))

    out_rows = []
    for gkey, rs in grouped.items():
        row = {}
        for k, v in zip(group_keys, gkey):
            row[k] = v
        row["n_rows"] = len(rs)

        for nk in numeric_keys:
            vals = []
            for r in rs:
                if nk in r:
                    vals.append(r[nk])
            vals = np.asarray(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[f"{nk}_mean"] = np.nan
                row[f"{nk}_std"] = np.nan
            else:
                row[f"{nk}_mean"] = float(np.mean(vals))
                row[f"{nk}_std"] = float(np.std(vals))

        out_rows.append(row)

    return out_rows


# ---------------------------------------------------------
# Main analysis over all seeds
# ---------------------------------------------------------
def run_full_baseline_coverage_analysis():
    if not coverage_cfg.enabled:
        print("[coverage] disabled.")
        return

    device = torch.device(cfg.device)
    train_set, test_set, train_patients, test_patients = rebuild_train_test_sets_for_analysis()

    analysis_root = Path(cfg.out_dir) / coverage_cfg.out_dir_name / coverage_cfg.experiment_name
    ensure_dir(analysis_root)

    all_global_rows: List[Dict] = []
    all_patient_rows: List[Dict] = []
    all_lopo_rows: List[Dict] = []

    for seed in cfg.seeds:
        global_rows, patient_rows, lopo_rows = run_baseline_coverage_analysis_for_one_seed(
            seed=seed,
            train_set=train_set,
            test_set=test_set,
            train_patients=train_patients,
            test_patients=test_patients,
            device=device,
        )
        all_global_rows.extend(global_rows)
        all_patient_rows.extend(patient_rows)
        all_lopo_rows.extend(lopo_rows)

    # 保存 all-seed
    write_log_csv(all_global_rows, analysis_root / "global_metrics_all_seeds.csv")
    write_log_csv(all_patient_rows, analysis_root / "patient_metrics_all_seeds.csv")
    write_log_csv(all_lopo_rows, analysis_root / "train_lopo_metrics_all_seeds.csv")

    # 全局 group summary：按 layer + token_class 汇总
    global_grouped = group_numeric_rows(
        all_global_rows,
        group_keys=["layer", "token_class"],
    )
    write_log_csv(global_grouped, analysis_root / "global_metrics_grouped_by_layer_class.csv")

    # 病人 group summary：按 split + layer + token_class + patient_id
    patient_grouped = group_numeric_rows(
        all_patient_rows,
        group_keys=["split", "layer", "token_class", "patient_id"],
    )
    write_log_csv(patient_grouped, analysis_root / "patient_metrics_grouped.csv")

    # LOPO group summary：按 layer + token_class + patient_id
    lopo_grouped = group_numeric_rows(
        all_lopo_rows,
        group_keys=["layer", "token_class", "patient_id"],
    )
    write_log_csv(lopo_grouped, analysis_root / "train_lopo_metrics_grouped.csv")

    # 总报告
    with open(analysis_root / "coverage_overall_report.txt", "w", encoding="utf-8") as f:
        f.write("Baseline coverage analysis completed.\n")
        f.write(f"experiment_name       = {coverage_cfg.experiment_name}\n")
        f.write(f"checkpoint_kind       = {coverage_cfg.checkpoint_kind}\n")
        f.write(f"train_patients        = {' '.join(train_patients)}\n")
        f.write(f"test_patients         = {' '.join(test_patients)}\n")
        f.write(f"layer_names           = {' '.join(coverage_cfg.layer_names)}\n")
        f.write(f"max_fg_tokens_case    = {coverage_cfg.max_fg_tokens_per_case}\n")
        f.write(f"max_bg_tokens_case    = {coverage_cfg.max_bg_tokens_per_case}\n")
        f.write(f"energy_threshold      = {coverage_cfg.energy_threshold}\n")
        f.write(f"max_components        = {coverage_cfg.max_components}\n")
        f.write(f"angle_components      = {coverage_cfg.angle_components}\n")
        f.write(f"channel_standardize   = {coverage_cfg.channel_standardize}\n")
        f.write(f"n_global_rows         = {len(all_global_rows)}\n")
        f.write(f"n_patient_rows        = {len(all_patient_rows)}\n")
        f.write(f"n_lopo_rows           = {len(all_lopo_rows)}\n")

    print("=" * 100)
    print("[coverage] all seeds finished.")
    print(f"[coverage] output -> {analysis_root.resolve()}")
    print("=" * 100)


# ---------------------------------------------------------
# New entry
# 你需要把原来的:
# if __name__ == "__main__":
#     main()
# 替换成下面这个版本
# ---------------------------------------------------------
def main_train_then_coverage():
    if RUN_TRAINING_MAIN:
        main()
    if RUN_COVERAGE_ANALYSIS:
        run_full_baseline_coverage_analysis()
RUN_TRAINING_MAIN = True
RUN_COVERAGE_ANALYSIS = True
if __name__ == "__main__":
    main_train_then_coverage()