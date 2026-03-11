#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_attention_unet_spider.py

用途：
- 从 SPIDER_448_25d 读取 2.5D 输入数据
- 按病人划分
- 随机选择 5 个病人作为训练集
- 其余病人作为测试集
- 使用 Attention U-Net 完成二分类骨分割
- 保存 best / last 模型、训练日志和数据划分

数据目录结构：
ROOT/
├── images/
│   ├── 1_t1_axis0_slice0003.npy / .npz
│   ├── ...
├── masks/
│   ├── 1_t1_axis0_slice0003.npy / .npz
│   ├── ...
└── meta.csv (可选)

输入 image:
- shape = (3, 448, 448)

输入 mask:
- shape = (448, 448)

运行示例：
python train_attention_unet_spider.py \
  --root ./SPIDER_448_25d \
  --out_dir ./attunet_spider_run1 \
  --train_patients 5 \
  --epochs 80 \
  --batch_size 2 \
  --lr 3e-4 \
  --seed 42
"""

import os
import re
import csv
import json
import math
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Utils
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_case_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    m = re.match(r"^([^_]+)_", stem)
    return m.group(1) if m else "unknown"


def load_array(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        if len(z.files) == 1:
            return z[z.files[0]]
        if "arr_0" in z.files:
            return z["arr_0"]
        raise ValueError(f"Cannot infer array key in npz: {path}, keys={z.files}")
    else:
        raise ValueError(f"Unsupported file type: {path}")


def dice_coefficient(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6):
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return dice


def iou_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6):
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou


def precision_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6):
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1.0 - target_bin)).sum(dim=(1, 2, 3))
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6):
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_bin) * target_bin).sum(dim=(1, 2, 3))
    return (tp + eps) / (tp + fn + eps)


# =========================
# Dataset
# =========================

class SpiderSliceDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        # 轻量增强
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()   # W
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()   # H
            mask = np.flip(mask, axis=0).copy()

        # 0/90/180/270 旋转
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # 轻微强度扰动
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            bias = random.uniform(-0.05, 0.05)
            img = np.clip(img * scale + bias, 0.0, 1.0)

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = load_array(img_path).astype(np.float32)
        mask = load_array(mask_path).astype(np.float32)

        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Expected image shape (3,H,W), got {img.shape} | {img_path}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H,W), got {mask.shape} | {mask_path}")

        mask = (mask > 0).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask[None, ...]).float()  # (1,H,W)

        return img, mask, os.path.basename(img_path)


def collect_pairs(root: str):
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".npy") or f.endswith(".npz")
    ])
    mask_files = set([
        f for f in os.listdir(masks_dir)
        if f.endswith(".npy") or f.endswith(".npz")
    ])

    pairs = []
    for f in image_files:
        if f in mask_files:
            pairs.append((os.path.join(images_dir, f), os.path.join(masks_dir, f)))

    if len(pairs) == 0:
        raise RuntimeError("No paired image/mask files found.")
    return pairs


def split_by_patient(pairs, train_patients=5, seed=42):
    case_to_pairs = defaultdict(list)
    for img_path, mask_path in pairs:
        case_id = parse_case_id(os.path.basename(img_path))
        case_to_pairs[case_id].append((img_path, mask_path))

    patient_ids = sorted(case_to_pairs.keys())
    if len(patient_ids) < train_patients + 1:
        raise ValueError(f"Not enough patients: total={len(patient_ids)}, requested train={train_patients}")

    rng = random.Random(seed)
    train_ids = sorted(rng.sample(patient_ids, train_patients))
    test_ids = sorted([x for x in patient_ids if x not in train_ids])

    train_pairs = []
    test_pairs = []

    for pid in train_ids:
        train_pairs.extend(case_to_pairs[pid])
    for pid in test_ids:
        test_pairs.extend(case_to_pairs[pid])

    return train_ids, test_ids, train_pairs, test_pairs


# =========================
# Model: Attention U-Net
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlockAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 1))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip_att = self.att(x, skip)
        x = torch.cat([skip_att, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(base_ch * 8, base_ch * 16)

        self.dec4 = UpBlockAtt(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = UpBlockAtt(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = UpBlockAtt(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = UpBlockAtt(base_ch * 2, base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        c = self.center(self.pool4(e4))

        d4 = self.dec4(c, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out_conv(d1)
        return out


# =========================
# Loss
# =========================

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


# =========================
# Train / Eval
# =========================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    losses = []
    dices = []
    ious = []
    precs = []
    recs = []

    criterion = BCEDiceLoss()

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        losses.append(loss.item())
        dices.extend(dice_coefficient(preds, masks).cpu().numpy().tolist())
        ious.extend(iou_score(preds, masks).cpu().numpy().tolist())
        precs.extend(precision_score(preds, masks).cpu().numpy().tolist())
        recs.extend(recall_score(preds, masks).cpu().numpy().tolist())

    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "precision": float(np.mean(precs)) if precs else float("nan"),
        "recall": float(np.mean(recs)) if recs else float("nan"),
    }
    return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# Main
# =========================

def main(args):
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")

    pairs = collect_pairs(args.root)
    print(f"[INFO] Total paired slices: {len(pairs)}")

    train_ids, test_ids, train_pairs, test_pairs = split_by_patient(
        pairs,
        train_patients=args.train_patients,
        seed=args.seed
    )

    print(f"[INFO] Train patients ({len(train_ids)}): {train_ids}")
    print(f"[INFO] Test patients  ({len(test_ids)}): {test_ids}")
    print(f"[INFO] Train slices: {len(train_pairs)}")
    print(f"[INFO] Test slices : {len(test_pairs)}")

    split_info = {
        "train_patients": train_ids,
        "test_patients": test_ids,
        "num_train_slices": len(train_pairs),
        "num_test_slices": len(test_pairs),
        "seed": args.seed,
    }
    save_json(split_info, os.path.join(args.out_dir, "split.json"))

    train_ds = SpiderSliceDataset(train_pairs, augment=True)
    test_ds = SpiderSliceDataset(test_pairs, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = AttentionUNet(in_ch=3, out_ch=1, base_ch=args.base_ch).to(device)
    criterion = BCEDiceLoss(bce_weight=args.w_bce, dice_weight=args.w_dice)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    best_model_path = os.path.join(args.out_dir, "best_model.pt")
    last_model_path = os.path.join(args.out_dir, "last_model.pt")

    best_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, device)

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_metrics["loss"],
            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "time_sec": elapsed,
        }
        history.append(row)

        print(
            f"[Ep {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_dice={test_metrics['dice']:.4f} "
            f"test_iou={test_metrics['iou']:.4f} "
            f"test_prec={test_metrics['precision']:.4f} "
            f"test_rec={test_metrics['recall']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if test_metrics["dice"] > best_dice:
            best_dice = test_metrics["dice"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
                "split": split_info,
            }, best_model_path)

        with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_dice": best_dice,
        "args": vars(args),
        "split": split_info,
    }, last_model_path)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Best Dice     : {best_dice:.4f}")
    print(f"Best model    : {best_model_path}")
    print(f"Last model    : {last_model_path}")
    print(f"Metrics CSV   : {metrics_csv}")
    print(f"Split JSON    : {os.path.join(args.out_dir, 'split.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./SPIDER_448_25d", help="preprocessed dataset root")
    parser.add_argument("--out_dir", type=str, default="./attunet_spider_run1", help="output dir")
    parser.add_argument("--train_patients", type=int, default=5, help="number of training patients")
    parser.add_argument("--epochs", type=int, default=80, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--base_ch", type=int, default=32, help="base channels")
    parser.add_argument("--w_bce", type=float, default=1.0, help="BCE weight")
    parser.add_argument("--w_dice", type=float, default=1.0, help="Dice weight")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--cpu", action="store_true", help="force cpu")
    args = parser.parse_args()
    main(args)