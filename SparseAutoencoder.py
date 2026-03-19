from __future__ import annotations

import os
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # image sizes
    out_size: int = 224
    digit_size: int = 28

    # MNIST bin threshold for GT / occlusion
    thr_bin: float = 0.15

    # dataset sizes
    n_train: int = 5
    n_test: int = 100

    # train/test distractors
    # 设定2：训练集干净；测试集有4个干扰数字
    n_distractors_train: int = 0
    n_distractors_test: int = 4

    # perturbations (test only)
    angle_max: float = 10.0
    scale_min: float = 0.75
    scale_max: float = 1.25
    translate_ratio: float = 0.08
    brightness_max: float = 0.25
    contrast_min: float = 0.6
    contrast_max: float = 1.6
    noise_std: float = 0.30

    # model
    latent_ch: int = 64
    base_ch: int = 16

    # optimization
    epochs: int = 200
    batch_size: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # loss weights
    lambda_bce: float = 1.0
    lambda_dice: float = 1.0
    lambda_recon: float = 0.5
    lambda_sparse: float = 2e-4

    # prediction threshold
    pred_thr: float = 0.5

    # sparsity stats
    bottleneck_zero_thr: float = 1e-3

    # outputs
    out_dir: str = "mnist_domain_shift_sparse_ae_out"
    save_vis_n: int = 12


cfg = Cfg()


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


# =========================================================
# MNIST sampling
# =========================================================
class MnistIndex:
    def __init__(self, mnist: datasets.MNIST):
        labels = mnist.targets.cpu().numpy()
        self.mnist = mnist
        self.by_label = {k: np.where(labels == k)[0].tolist() for k in range(10)}

    def sample(self, digit: int) -> Image.Image:
        idx = random.choice(self.by_label[digit])
        img, _ = self.mnist[idx]
        return img


# =========================================================
# Synthesis
# =========================================================
def pil_to_f32(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_f32(x: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize(hw[::-1], Image.BILINEAR),
        dtype=np.float32
    ) / 255.0


def overlay_digit_max(comp: np.ndarray, digit_img: np.ndarray, top: int, left: int):
    h, w = digit_img.shape
    comp[top:top + h, left:left + w] = np.maximum(
        comp[top:top + h, left:left + w], digit_img
    )


def make_synth_sample(mi: MnistIndex, is_train: bool) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    设定2：
      - train: 干净大8，无干扰数字
      - test : 大8 + 仿射/亮度/对比度/噪声 + 随机位置干扰数字
    """
    img8 = mi.sample(8)
    base = np.array(
        img8.resize((cfg.out_size, cfg.out_size), resample=Image.BILINEAR),
        dtype=np.float32
    ) / 255.0
    mask = (base > cfg.thr_bin).astype(np.uint8)

    meta: Dict = {
        "is_train": is_train,
        "angle": 0.0,
        "scale": 1.0,
        "translate_y": 0,
        "translate_x": 0,
        "brightness": 0.0,
        "contrast": 1.0,
        "pasted_digits": [],
        "distractor_boxes": [],
    }

    # 测试集才做 domain shift
    if not is_train:
        x = torch.from_numpy(base)[None, None, ...]
        m = torch.from_numpy(mask.astype(np.float32))[None, None, ...]

        angle = random.uniform(-cfg.angle_max, cfg.angle_max)
        scale = random.uniform(cfg.scale_min, cfg.scale_max)
        max_t = int(cfg.translate_ratio * cfg.out_size)
        translate = (
            random.randint(-max_t, max_t),  # x
            random.randint(-max_t, max_t),  # y
        )

        x2 = TF.affine(
            x,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        m2 = TF.affine(
            m,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )

        base = x2[0, 0].clamp(0, 1).numpy().astype(np.float32)
        mask = (m2[0, 0].numpy() > 0.5).astype(np.uint8)

        contrast = random.uniform(cfg.contrast_min, cfg.contrast_max)
        brightness = random.uniform(-cfg.brightness_max, cfg.brightness_max)
        noise = np.random.randn(cfg.out_size, cfg.out_size).astype(np.float32) * cfg.noise_std

        base = np.clip(base * contrast + brightness + noise, 0.0, 1.0)

        meta["angle"] = float(angle)
        meta["scale"] = float(scale)
        meta["translate_x"] = int(translate[0])
        meta["translate_y"] = int(translate[1])
        meta["brightness"] = float(brightness)
        meta["contrast"] = float(contrast)

    comp = base.copy()

    corners = [
        (0, 0),
        (0, cfg.out_size - cfg.digit_size),
        (cfg.out_size - cfg.digit_size, 0),
        (cfg.out_size - cfg.digit_size, cfg.out_size - cfg.digit_size),
    ]

    n_distractors = cfg.n_distractors_train if is_train else cfg.n_distractors_test

    for k in range(n_distractors):
        d = random.choice([i for i in range(10) if i != 8])
        od = mi.sample(d)
        o = pil_to_f32(od)
        if cfg.digit_size != 28:
            o = resize_f32(o, (cfg.digit_size, cfg.digit_size))

        if is_train:
            top, left = corners[k % 4]
        else:
            top = random.randint(0, cfg.out_size - cfg.digit_size)
            left = random.randint(0, cfg.out_size - cfg.digit_size)

        overlay_digit_max(comp, o, top, left)

        occ = (o > cfg.thr_bin).astype(np.uint8)
        mask[top:top + cfg.digit_size, left:left + cfg.digit_size] *= (1 - occ)

        meta["pasted_digits"].append(int(d))
        meta["distractor_boxes"].append((int(top), int(left), int(cfg.digit_size), int(cfg.digit_size)))

    return comp.astype(np.float32), mask.astype(np.uint8), meta


class DomainShiftSynthDataset(Dataset):
    def __init__(self, mi: MnistIndex, n: int, is_train: bool):
        self.items = [make_synth_sample(mi, is_train=is_train) for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, msk, meta = self.items[idx]
        img_t = torch.from_numpy(img)[None, ...].float()
        msk_t = torch.from_numpy(msk)[None, ...].float()
        return img_t, msk_t, meta


# =========================================================
# Model: 3-layer sparse AE segmentation network
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
    """
    Encoder: 3 levels
    Bottleneck: sparse latent z
    Decoder: 3 levels
    Heads:
      - recon head
      - segmentation head
    """
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

        self.recon_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.enc1(x))   # 224 -> 112
        x = self.pool2(self.enc2(x))   # 112 -> 56
        x = self.pool3(self.enc3(x))   # 56 -> 28

        z = self.bottleneck(x)

        x = self.up3(z)                # 28 -> 56
        x = self.dec3(x)

        x = self.up2(x)                # 56 -> 112
        x = self.dec2(x)

        x = self.up1(x)                # 112 -> 224
        feat = self.dec1(x)

        recon = torch.sigmoid(self.recon_head(feat))
        seg_logits = self.seg_head(feat)
        return seg_logits, recon, z


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_recon = 0.0
    total_sparse = 0.0
    n_seen = 0

    for img, mask, _meta in loader:
        img = img.to(device)
        mask = mask.to(device)

        seg_logits, recon, z = model(img)

        loss_bce = F.binary_cross_entropy_with_logits(seg_logits, mask)
        loss_dice = dice_loss_from_logits(seg_logits, mask)
        loss_recon = F.mse_loss(recon, img)
        loss_sparse = z.abs().mean()

        loss = (
            cfg.lambda_bce * loss_bce
            + cfg.lambda_dice * loss_dice
            + cfg.lambda_recon * loss_recon
            + cfg.lambda_sparse * loss_sparse
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = img.shape[0]
        n_seen += bs
        total_loss += float(loss.item()) * bs
        total_bce += float(loss_bce.item()) * bs
        total_dice += float(loss_dice.item()) * bs
        total_recon += float(loss_recon.item()) * bs
        total_sparse += float(loss_sparse.item()) * bs

    return {
        "loss": total_loss / max(n_seen, 1),
        "bce": total_bce / max(n_seen, 1),
        "dice": total_dice / max(n_seen, 1),
        "recon": total_recon / max(n_seen, 1),
        "sparse": total_sparse / max(n_seen, 1),
    }


@torch.no_grad()
def infer_one(model: nn.Module, img: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor | float]:
    model.eval()
    img = img.to(device)

    seg_logits, recon, z = model(img)
    prob = torch.sigmoid(seg_logits)
    pred = (prob >= cfg.pred_thr).float()

    z_abs = z.abs()
    z_zero_ratio = float((z_abs < cfg.bottleneck_zero_thr).float().mean().item())
    z_mean_abs = float(z_abs.mean().item())

    return {
        "prob": prob.cpu(),
        "pred": pred.cpu(),
        "recon": recon.cpu(),
        "z": z.cpu(),
        "z_zero_ratio": z_zero_ratio,
        "z_mean_abs": z_mean_abs,
    }


# =========================================================
# Visualization
# =========================================================
def save_case_vis(
    case_id: int,
    img: np.ndarray,
    gt: np.ndarray,
    recon: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    z: np.ndarray,
    meta: Dict,
    out_path: str,
):
    z_map = np.mean(np.abs(z), axis=0)
    z_map = (z_map - z_map.min()) / (z_map.max() - z_map.min() + 1e-9)

    iou, dice = iou_and_dice(pred, gt)

    fig, axes = plt.subplots(1, 6, figsize=(19, 3.6))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")

    axes[2].imshow(recon, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("AE recon")

    axes[3].imshow(prob, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Pred prob")

    axes[4].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(np.ma.masked_where(pred == 0, pred), alpha=0.50)
    axes[4].set_title(f"Pred mask\nIoU={iou:.3f}")

    axes[5].imshow(z_map, cmap="magma", vmin=0, vmax=1)
    axes[5].set_title("mean |z|")

    for ax in axes:
        ax.axis("off")

    title = (
        f"id={case_id} | "
        f"angle={meta.get('angle', 0):.1f}, "
        f"scale={meta.get('scale', 1):.2f}, "
        f"brightness={meta.get('brightness', 0):.2f}, "
        f"contrast={meta.get('contrast', 1):.2f}, "
        f"distractors={len(meta.get('pasted_digits', []))} | "
        f"Dice={dice:.3f}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_train_curve(history: List[Dict[str, float]], out_path: str):
    xs = np.arange(1, len(history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [h["loss"] for h in history], label="total")
    plt.plot(xs, [h["bce"] for h in history], label="bce")
    plt.plot(xs, [h["dice"] for h in history], label="dice")
    plt.plot(xs, [h["recon"] for h in history], label="recon")
    plt.plot(xs, [h["sparse"] for h in history], label="sparse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    vis_dir = os.path.join(cfg.out_dir, "vis")
    ensure_dir(vis_dir)

    device = torch.device(cfg.device)

    mnist = datasets.MNIST(root="./mnist_data", train=True, download=True)
    mi = MnistIndex(mnist)

    train_set = DomainShiftSynthDataset(mi, n=cfg.n_train, is_train=True)
    test_set = DomainShiftSynthDataset(mi, n=cfg.n_test, is_train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=min(cfg.batch_size, len(train_set)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = SparseAENet(base_ch=cfg.base_ch, latent_ch=cfg.latent_ch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    print("=" * 80)
    print("MNIST domain-shift benchmark with 3-layer sparse AE segmentation")
    print(f"device              = {cfg.device}")
    print(f"train samples       = {cfg.n_train} (clean, no distractors)")
    print(f"test samples        = {cfg.n_test} (shifted + random distractors)")
    print(f"epochs              = {cfg.epochs}")
    print(f"digit_size          = {cfg.digit_size}")
    print(f"noise_std(test)     = {cfg.noise_std}")
    print(f"angle_max(test)     = {cfg.angle_max}")
    print(f"scale_range(test)   = [{cfg.scale_min}, {cfg.scale_max}]")
    print(f"n_distractors train = {cfg.n_distractors_train}")
    print(f"n_distractors test  = {cfg.n_distractors_test}")
    print(f"lambda_sparse       = {cfg.lambda_sparse}")
    print("=" * 80)

    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        stats = train_one_epoch(model, train_loader, optimizer, device)
        history.append(stats)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            print(
                f"[Epoch {epoch + 1:03d}/{cfg.epochs}] "
                f"loss={stats['loss']:.4f} "
                f"bce={stats['bce']:.4f} "
                f"dice={stats['dice']:.4f} "
                f"recon={stats['recon']:.4f} "
                f"sparse={stats['sparse']:.4f}"
            )

    save_train_curve(history, os.path.join(cfg.out_dir, "train_curve.png"))

    # test
    rows = []
    ious = []
    dices = []

    print("=" * 80)
    print("Testing...")
    print("=" * 80)

    for i in range(len(test_set)):
        img_t, gt_t, meta = test_set[i]
        out = infer_one(model, img_t.unsqueeze(0), device)

        img = img_t.squeeze(0).numpy().astype(np.float32)
        gt = gt_t.squeeze(0).numpy().astype(np.float32)
        prob = out["prob"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        pred = out["pred"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        recon = out["recon"].squeeze(0).squeeze(0).numpy().astype(np.float32)
        z = out["z"].squeeze(0).numpy().astype(np.float32)   # (C,H,W)

        iou, dice = iou_and_dice(pred, gt)
        ious.append(iou)
        dices.append(dice)

        row = {
            "case_id": i,
            "iou": iou,
            "dice": dice,
            "z_mean_abs": float(out["z_mean_abs"]),
            "z_zero_ratio": float(out["z_zero_ratio"]),
            "angle": meta["angle"],
            "scale": meta["scale"],
            "translate_x": meta["translate_x"],
            "translate_y": meta["translate_y"],
            "brightness": meta["brightness"],
            "contrast": meta["contrast"],
            "n_distractors": len(meta["pasted_digits"]),
            "pasted_digits": " ".join(map(str, meta["pasted_digits"])),
        }
        rows.append(row)

        print(
            f"[{i + 1:03d}/{len(test_set)}] "
            f"IoU={iou:.4f} Dice={dice:.4f} "
            f"z_mean_abs={out['z_mean_abs']:.4f} "
            f"z_zero={out['z_zero_ratio']:.3f}"
        )

        if i < cfg.save_vis_n:
            save_case_vis(
                case_id=i,
                img=img,
                gt=gt,
                recon=recon,
                prob=prob,
                pred=pred,
                z=z,
                meta=meta,
                out_path=os.path.join(vis_dir, f"case_{i:03d}.png"),
            )

    mean_iou = float(np.mean(ious))
    std_iou = float(np.std(ious))
    mean_dice = float(np.mean(dices))
    std_dice = float(np.std(dices))
    mean_z = float(np.mean([r["z_mean_abs"] for r in rows]))
    mean_zero = float(np.mean([r["z_zero_ratio"] for r in rows]))

    print("=" * 80)
    print(f"Mean IoU      = {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Mean Dice     = {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean |z|      = {mean_z:.4f}")
    print(f"Mean zero(z)  = {mean_zero:.4f}")
    print("=" * 80)

    # save metrics
    metrics_path = os.path.join(cfg.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id", "iou", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    # save summary
    summary_path = os.path.join(cfg.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MNIST domain-shift benchmark with 3-layer sparse AE segmentation\n")
        f.write(f"device              = {cfg.device}\n")
        f.write(f"train samples       = {cfg.n_train} (clean)\n")
        f.write(f"test samples        = {cfg.n_test} (shifted + distractors)\n")
        f.write(f"epochs              = {cfg.epochs}\n")
        f.write(f"digit_size          = {cfg.digit_size}\n")
        f.write(f"noise_std(test)     = {cfg.noise_std}\n")
        f.write(f"angle_max(test)     = {cfg.angle_max}\n")
        f.write(f"scale_min(test)     = {cfg.scale_min}\n")
        f.write(f"scale_max(test)     = {cfg.scale_max}\n")
        f.write(f"n_distractors_train = {cfg.n_distractors_train}\n")
        f.write(f"n_distractors_test  = {cfg.n_distractors_test}\n")
        f.write(f"lambda_sparse       = {cfg.lambda_sparse}\n")
        f.write(f"Mean IoU            = {mean_iou:.6f} +- {std_iou:.6f}\n")
        f.write(f"Mean Dice           = {mean_dice:.6f} +- {std_dice:.6f}\n")
        f.write(f"Mean |z|            = {mean_z:.6f}\n")
        f.write(f"Mean zero(z)        = {mean_zero:.6f}\n")

    # save train samples preview
    preview_dir = os.path.join(cfg.out_dir, "train_preview")
    ensure_dir(preview_dir)
    for i in range(len(train_set)):
        img_t, gt_t, meta = train_set[i]
        img = img_t.squeeze(0).numpy()
        gt = gt_t.squeeze(0).numpy()
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title("Train image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(gt, cmap="gray", vmin=0, vmax=1)
        plt.title("Train mask")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(preview_dir, f"train_{i:02d}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()