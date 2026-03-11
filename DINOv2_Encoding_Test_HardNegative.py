#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINOv2_Encoding_AllSlices_3Class.py

三类 token 评估版：
- bone
- near-soft-tissue (hard negative ring around bone)
- far-background

对整个 SPIDER_448_25d 的所有切片做 DINOv2 特征可分性评估。

输出:
out/
├── all_slice_metrics_3class.csv
├── summary_by_modality_3class.csv
├── summary_by_patient_modality_3class.csv
├── bar_auc_bone_near.png
├── bar_auc_bone_far.png
├── bar_auc_near_far.png
├── bar_fisher_bone_near.png
├── bar_fisher_bone_far.png
├── bar_gap_bone_near.png
├── bar_gap_bone_far.png
├── bar_bone_purity.png
├── bar_bone_contam_near.png
├── bar_bone_contam_far.png
└── bar_all_metrics_3class.png

依赖:
pip install torch torchvision numpy pandas matplotlib

用法:
python DINOv2_Encoding_AllSlices_3Class.py
或
python DINOv2_Encoding_AllSlices_3Class.py --root ./SPIDER_448_25d --out ./dinov2_all_diag_3class --leave_one_out
"""

import os
import re
import math
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


FILE_RE = re.compile(
    r"^(?P<case>[^_]+)_(?P<modality>.+?)_axis(?P<axis>\d+)_slice(?P<slice>\d+)\.(?P<ext>npy|npz)$"
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# =========================================================
# 基础
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_filename(fname: str):
    m = FILE_RE.match(fname)
    if m is None:
        return None
    d = m.groupdict()
    d["axis"] = int(d["axis"])
    d["slice"] = int(d["slice"])
    return d


def list_data_files(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([f for f in os.listdir(folder) if f.endswith(".npy") or f.endswith(".npz")])


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        if len(z.files) == 0:
            raise ValueError(f"No arrays found in npz: {path}")
        key = "arr_0" if "arr_0" in z.files else z.files[0]
        return z[key]
    raise ValueError(f"Unsupported file type: {path}")


def collect_all_groups(root: str):
    """
    groups[(patient_id, modality)] = [item1, item2, ...]
    """
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    image_files = list_data_files(images_dir)
    mask_files = set(list_data_files(masks_dir))

    groups = defaultdict(list)
    for fname in image_files:
        meta = parse_filename(fname)
        if meta is None:
            continue
        if fname not in mask_files:
            continue

        patient_id = str(meta["case"])
        modality = meta["modality"]

        groups[(patient_id, modality)].append({
            "fname": fname,
            "slice": meta["slice"],
            "axis": meta["axis"],
            "image_path": os.path.join(images_dir, fname),
            "mask_path": os.path.join(masks_dir, fname),
            "patient_id": patient_id,
            "modality": modality,
        })

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x["slice"])

    return groups


# =========================================================
# DINOv2
# =========================================================
def load_dinov2(model_name: str, device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_patch_tokens(model, image_chw: np.ndarray, device: torch.device):
    """
    image_chw: (3,H,W), float32 in [0,1]
    return:
      feat: (N,C) on CPU
      gh, gw
    """
    x = torch.from_numpy(image_chw).float().unsqueeze(0).to(device)
    x = x.clamp(0.0, 1.0)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

    out = model.forward_features(x)

    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            feat = out["x_norm_patchtokens"]
        elif "x_prenorm" in out:
            feat = out["x_prenorm"][:, 1:, :]
        else:
            raise KeyError(f"Unexpected forward_features keys: {list(out.keys())}")
    else:
        raise TypeError("forward_features output is not a dict.")

    feat = feat[0].detach().cpu()
    n = feat.shape[0]
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Patch token number {n} is not square.")
    return feat, side, side


# =========================================================
# 数据转换
# =========================================================
def to_image_three_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 3:
        x = arr.astype(np.float32)
    elif arr.ndim == 2:
        x = np.stack([arr, arr, arr], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def to_binary_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {arr.shape}")
    return (arr > 0).astype(np.uint8)


def downsample_mask_to_tokens(mask_hw: np.ndarray, gh: int, gw: int) -> np.ndarray:
    m = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    cov = F.adaptive_avg_pool2d(m, output_size=(gh, gw))[0, 0].numpy()
    return cov


def dilate_mask(mask_hw: np.ndarray, radius_px: int) -> np.ndarray:
    """
    用 max_pool2d 做二维膨胀
    """
    if radius_px <= 0:
        return mask_hw.astype(np.uint8)
    k = radius_px * 2 + 1
    x = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius_px)
    return (y[0, 0].numpy() > 0).astype(np.uint8)


def build_three_class_token_masks(
    mask_hw: np.ndarray,
    gh: int,
    gw: int,
    fg_cov_thr: float,
    bg_cov_thr: float,
    near_inner_px: int,
    near_outer_px: int,
    near_cov_thr: float,
    far_cov_thr: float,
):
    """
    返回:
      bone_token: (N,) bool
      near_token: (N,) bool
      far_token:  (N,) bool
      cov_bone:   (gh,gw)
      cov_near:   (gh,gw)
      cov_far:    (gh,gw)
    """
    bone_mask = (mask_hw > 0).astype(np.uint8)

    dil_in = dilate_mask(bone_mask, near_inner_px)
    dil_out = dilate_mask(bone_mask, near_outer_px)
    near_ring = ((dil_out > 0) & (dil_in == 0)).astype(np.uint8)
    far_region = ((dil_out == 0) & (bone_mask == 0)).astype(np.uint8)

    cov_bone = downsample_mask_to_tokens(bone_mask, gh, gw)
    cov_near = downsample_mask_to_tokens(near_ring, gh, gw)
    cov_far = downsample_mask_to_tokens(far_region, gh, gw)

    bone_token = (cov_bone >= fg_cov_thr).reshape(-1)

    # near: 本身不是 bone，且在 near ring 中占比高
    near_token = ((cov_bone <= bg_cov_thr) & (cov_near >= near_cov_thr)).reshape(-1)

    # far: 既不是 bone，也不是 near，并且在 far region 中占比高
    far_token = ((cov_bone <= bg_cov_thr) & (cov_far >= far_cov_thr) & (~near_token.reshape(gh, gw))).reshape(-1)

    return bone_token, near_token, far_token, cov_bone, cov_near, cov_far


# =========================================================
# 指标
# =========================================================
def roc_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores).astype(np.float64)
    labels = np.asarray(labels).astype(np.int64)

    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        for uidx, c in enumerate(cnt):
            if c > 1:
                idx = np.where(inv == uidx)[0]
                ranks[idx] = ranks[idx].mean()

    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fisher_ratio(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) == 0 or len(scores_b) == 0:
        return np.nan
    mu_a = float(np.mean(scores_a))
    mu_b = float(np.mean(scores_b))
    var_a = float(np.var(scores_a))
    var_b = float(np.var(scores_b))
    return float((mu_a - mu_b) ** 2 / (var_a + var_b + 1e-12))


def score_gap(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) == 0 or len(scores_b) == 0:
        return np.nan
    return float(np.mean(scores_a) - np.mean(scores_b))


def knn_neighbor_stats(features: np.ndarray, labels: np.ndarray, k: int = 7):
    """
    labels:
      0 = far
      1 = near
      2 = bone

    返回:
      bone_purity
      bone_contam_near
      bone_contam_far
    """
    x = torch.from_numpy(features.astype(np.float32))
    x = F.normalize(x, dim=1)
    sim = x @ x.T
    n = sim.shape[0]
    if n <= 1:
        return np.nan, np.nan, np.nan

    kk = min(k, n - 1)
    sim.fill_diagonal_(-1e9)
    nn_idx = torch.topk(sim, k=kk, dim=1).indices

    y = torch.from_numpy(labels.astype(np.int64))
    nn_labels = y[nn_idx]

    bone_mask = (y == 2)
    if bone_mask.sum().item() == 0:
        return np.nan, np.nan, np.nan

    bone_nn = nn_labels[bone_mask]  # (n_bone, k)

    bone_purity = (bone_nn == 2).float().mean().item()
    bone_contam_near = (bone_nn == 1).float().mean().item()
    bone_contam_far = (bone_nn == 0).float().mean().item()

    return float(bone_purity), float(bone_contam_near), float(bone_contam_far)


def pairwise_auc_from_score(score: np.ndarray, mask_pos: np.ndarray, mask_neg: np.ndarray) -> float:
    if mask_pos.sum() == 0 or mask_neg.sum() == 0:
        return np.nan
    s = np.concatenate([score[mask_pos], score[mask_neg]], axis=0)
    y = np.concatenate([
        np.ones(mask_pos.sum(), dtype=np.int64),
        np.zeros(mask_neg.sum(), dtype=np.int64)
    ], axis=0)
    return roc_auc_from_scores(s, y)


# =========================================================
# prototype 构造
# =========================================================
def build_bone_prototype(
    all_feats: List[torch.Tensor],
    all_bone_masks: List[np.ndarray],
    exclude_idx: Optional[int] = None,
):
    feats = []
    for i, feat in enumerate(all_feats):
        if exclude_idx is not None and i == exclude_idx:
            continue
        m = all_bone_masks[i]
        if m.any():
            feats.append(feat[m])

    if len(feats) == 0:
        return None

    x = torch.cat(feats, dim=0)
    p = F.normalize(x.mean(dim=0, keepdim=True), dim=1)[0]
    return p


# =========================================================
# 主诊断逻辑
# =========================================================
def diagnose_group(
    model,
    device: torch.device,
    patient_id: str,
    modality: str,
    items: List[Dict],
    fg_cov_thr: float,
    bg_cov_thr: float,
    near_inner_px: int,
    near_outer_px: int,
    near_cov_thr: float,
    far_cov_thr: float,
    k: int,
    leave_one_out: bool,
):
    """
    对一个 (patient_id, modality) 组内所有切片做三类评估
    """
    all_feats = []
    all_items = []
    all_bone_masks = []
    all_near_masks = []
    all_far_masks = []

    for item in items:
        image_arr = load_array(item["image_path"])
        mask_arr = load_array(item["mask_path"])

        image_3ch = to_image_three_channel(image_arr)
        mask_hw = to_binary_mask(mask_arr)

        feat, gh, gw = extract_patch_tokens(model, image_3ch, device=device)

        bone_token, near_token, far_token, _, _, _ = build_three_class_token_masks(
            mask_hw=mask_hw,
            gh=gh,
            gw=gw,
            fg_cov_thr=fg_cov_thr,
            bg_cov_thr=bg_cov_thr,
            near_inner_px=near_inner_px,
            near_outer_px=near_outer_px,
            near_cov_thr=near_cov_thr,
            far_cov_thr=far_cov_thr,
        )

        all_feats.append(feat)
        all_bone_masks.append(bone_token)
        all_near_masks.append(near_token)
        all_far_masks.append(far_token)
        all_items.append({
            **item,
            "grid_h": gh,
            "grid_w": gw,
        })

    rows = []

    for idx, feat in enumerate(all_feats):
        p_bone = build_bone_prototype(
            all_feats, all_bone_masks,
            exclude_idx=idx if leave_one_out else None
        )
        if p_bone is None:
            continue

        x = F.normalize(feat, dim=1)
        score = (x @ p_bone).numpy()  # 用 bone prototype 相似度做三类排序

        bone_mask = all_bone_masks[idx]
        near_mask = all_near_masks[idx]
        far_mask = all_far_masks[idx]

        n_bone = int(bone_mask.sum())
        n_near = int(near_mask.sum())
        n_far = int(far_mask.sum())

        # 至少要能完成两个关键对比
        if n_bone == 0 or n_near == 0 or n_far == 0:
            continue

        scores_bone = score[bone_mask]
        scores_near = score[near_mask]
        scores_far = score[far_mask]

        auc_bone_near = pairwise_auc_from_score(score, bone_mask, near_mask)
        auc_bone_far = pairwise_auc_from_score(score, bone_mask, far_mask)
        auc_near_far = pairwise_auc_from_score(score, near_mask, far_mask)

        fisher_bone_near = fisher_ratio(scores_bone, scores_near)
        fisher_bone_far = fisher_ratio(scores_bone, scores_far)

        gap_bone_near = score_gap(scores_bone, scores_near)
        gap_bone_far = score_gap(scores_bone, scores_far)

        # kNN 统计：只在三类有效 token 上做
        valid = bone_mask | near_mask | far_mask
        valid_feats = feat.numpy()[valid]

        valid_labels = np.full(valid.sum(), -1, dtype=np.int64)
        # 约定 0=far, 1=near, 2=bone
        valid_labels[np.where(far_mask[valid])[0]] = 0
        valid_labels[np.where(near_mask[valid])[0]] = 1
        valid_labels[np.where(bone_mask[valid])[0]] = 2

        bone_purity, bone_contam_near, bone_contam_far = knn_neighbor_stats(
            valid_feats, valid_labels, k=k
        )

        # 难度比
        auc_ratio_bone_near_over_far = (
            auc_bone_near / auc_bone_far if np.isfinite(auc_bone_near) and np.isfinite(auc_bone_far) and abs(auc_bone_far) > 1e-12 else np.nan
        )
        gap_ratio_bone_near_over_far = (
            gap_bone_near / gap_bone_far if np.isfinite(gap_bone_near) and np.isfinite(gap_bone_far) and abs(gap_bone_far) > 1e-12 else np.nan
        )
        fisher_ratio_bone_near_over_far = (
            fisher_bone_near / fisher_bone_far if np.isfinite(fisher_bone_near) and np.isfinite(fisher_bone_far) and abs(fisher_bone_far) > 1e-12 else np.nan
        )

        item = all_items[idx]
        slice_key = f"{patient_id}_{modality}_s{item['slice']:04d}"

        rows.append({
            "patient_id": patient_id,
            "modality": modality,
            "fname": item["fname"],
            "slice_index": item["slice"],
            "slice_key": slice_key,
            "axis": item["axis"],
            "grid_h": item["grid_h"],
            "grid_w": item["grid_w"],

            "n_bone_tokens": n_bone,
            "n_near_tokens": n_near,
            "n_far_tokens": n_far,

            "auc_bone_near": auc_bone_near,
            "auc_bone_far": auc_bone_far,
            "auc_near_far": auc_near_far,

            "fisher_bone_near": fisher_bone_near,
            "fisher_bone_far": fisher_bone_far,

            "gap_bone_near": gap_bone_near,
            "gap_bone_far": gap_bone_far,

            "bone_purity": bone_purity,
            "bone_contam_near": bone_contam_near,
            "bone_contam_far": bone_contam_far,

            "auc_ratio_bone_near_over_far": auc_ratio_bone_near_over_far,
            "gap_ratio_bone_near_over_far": gap_ratio_bone_near_over_far,
            "fisher_ratio_bone_near_over_far": fisher_ratio_bone_near_over_far,

            "fg_cov_thr": fg_cov_thr,
            "bg_cov_thr": bg_cov_thr,
            "near_inner_px": near_inner_px,
            "near_outer_px": near_outer_px,
            "near_cov_thr": near_cov_thr,
            "far_cov_thr": far_cov_thr,
        })

        print(
            f"[{patient_id}][{modality}] slice={item['slice']:04d} "
            f"AUC(B,N)={auc_bone_near:.4f} AUC(B,F)={auc_bone_far:.4f} "
            f"Gap(B,N)={gap_bone_near:.4f} Gap(B,F)={gap_bone_far:.4f} "
            f"ContamNear={bone_contam_near:.4f}"
        )

    return rows


# =========================================================
# 作图
# =========================================================
def plot_metric_bar(df: pd.DataFrame, metric: str, out_path: str, title: str):
    df = df.sort_values(["patient_id", "modality", "slice_index"]).reset_index(drop=True)
    labels = df["slice_key"].tolist()
    values = df[metric].tolist()
    colors = ["tab:blue" if m == "t1" else "tab:orange" for m in df["modality"].tolist()]

    width = max(14, len(df) * 0.22)
    plt.figure(figsize=(width, 5))
    plt.bar(np.arange(len(df)), values, color=colors, width=0.8)
    plt.xticks(np.arange(len(df)), labels, rotation=90, fontsize=7)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_all_metrics(df: pd.DataFrame, out_path: str):
    df = df.sort_values(["patient_id", "modality", "slice_index"]).reset_index(drop=True)
    labels = df["slice_key"].tolist()
    x = np.arange(len(df))
    colors = ["tab:blue" if m == "t1" else "tab:orange" for m in df["modality"].tolist()]
    width = max(16, len(df) * 0.22)

    metrics = [
        ("auc_bone_near", "AUC(Bone,Near)"),
        ("auc_bone_far", "AUC(Bone,Far)"),
        ("gap_bone_near", "Gap(Bone,Near)"),
        ("gap_bone_far", "Gap(Bone,Far)"),
        ("bone_purity", "Bone purity"),
        ("bone_contam_near", "Bone->Near contam"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(width, 18), sharex=True)

    for ax, (col, title) in zip(axes, metrics):
        ax.bar(x, df[col].tolist(), color=colors, width=0.8)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_title("DINOv2 3-class token separability across all slices")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 主函数
# =========================================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(BASE_DIR, "SPIDER_448_25d"),
        help="e.g. ./SPIDER_448_25d"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(BASE_DIR, "dinov2_all_diag_3class"),
        help="output folder"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]
    )
    parser.add_argument("--fg_cov_thr", type=float, default=0.30)
    parser.add_argument("--bg_cov_thr", type=float, default=0.01)

    # near / far 三类划分参数
    parser.add_argument("--near_inner_px", type=int, default=3,
                        help="inner dilation radius in pixels")
    parser.add_argument("--near_outer_px", type=int, default=15,
                        help="outer dilation radius in pixels")
    parser.add_argument("--near_cov_thr", type=float, default=0.20,
                        help="token near-ring coverage threshold")
    parser.add_argument("--far_cov_thr", type=float, default=0.80,
                        help="token far-region coverage threshold")

    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--leave_one_out", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out)
    device = torch.device(args.device)

    print(f"[INFO] root  = {args.root}")
    print(f"[INFO] out   = {args.out}")
    print(f"[INFO] model = {args.model}")

    groups = collect_all_groups(args.root)
    if len(groups) == 0:
        raise RuntimeError(f"No matched image/mask groups found under: {args.root}")

    print(f"[INFO] total (patient, modality) groups = {len(groups)}")
    model = load_dinov2(args.model, device=device)

    all_rows = []
    for (patient_id, modality), items in sorted(groups.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        rows = diagnose_group(
            model=model,
            device=device,
            patient_id=patient_id,
            modality=modality,
            items=items,
            fg_cov_thr=args.fg_cov_thr,
            bg_cov_thr=args.bg_cov_thr,
            near_inner_px=args.near_inner_px,
            near_outer_px=args.near_outer_px,
            near_cov_thr=args.near_cov_thr,
            far_cov_thr=args.far_cov_thr,
            k=args.k,
            leave_one_out=args.leave_one_out,
        )
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise RuntimeError("No valid slice metrics were produced.")

    df = pd.DataFrame(all_rows).sort_values(["patient_id", "modality", "slice_index"]).reset_index(drop=True)

    all_csv = os.path.join(args.out, "all_slice_metrics_3class.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    summary_modality = (
        df.groupby("modality")[
            [
                "auc_bone_near", "auc_bone_far", "auc_near_far",
                "fisher_bone_near", "fisher_bone_far",
                "gap_bone_near", "gap_bone_far",
                "bone_purity", "bone_contam_near", "bone_contam_far",
                "auc_ratio_bone_near_over_far",
                "gap_ratio_bone_near_over_far",
                "fisher_ratio_bone_near_over_far",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_modality.columns = [
        "modality",
        "auc_bone_near_mean", "auc_bone_near_std", "auc_bone_near_count",
        "auc_bone_far_mean", "auc_bone_far_std", "auc_bone_far_count",
        "auc_near_far_mean", "auc_near_far_std", "auc_near_far_count",
        "fisher_bone_near_mean", "fisher_bone_near_std", "fisher_bone_near_count",
        "fisher_bone_far_mean", "fisher_bone_far_std", "fisher_bone_far_count",
        "gap_bone_near_mean", "gap_bone_near_std", "gap_bone_near_count",
        "gap_bone_far_mean", "gap_bone_far_std", "gap_bone_far_count",
        "bone_purity_mean", "bone_purity_std", "bone_purity_count",
        "bone_contam_near_mean", "bone_contam_near_std", "bone_contam_near_count",
        "bone_contam_far_mean", "bone_contam_far_std", "bone_contam_far_count",
        "auc_ratio_bone_near_over_far_mean", "auc_ratio_bone_near_over_far_std", "auc_ratio_bone_near_over_far_count",
        "gap_ratio_bone_near_over_far_mean", "gap_ratio_bone_near_over_far_std", "gap_ratio_bone_near_over_far_count",
        "fisher_ratio_bone_near_over_far_mean", "fisher_ratio_bone_near_over_far_std", "fisher_ratio_bone_near_over_far_count",
    ]
    summary_modality_csv = os.path.join(args.out, "summary_by_modality_3class.csv")
    summary_modality.to_csv(summary_modality_csv, index=False, encoding="utf-8-sig")

    summary_pm = (
        df.groupby(["patient_id", "modality"])[
            [
                "auc_bone_near", "auc_bone_far", "auc_near_far",
                "fisher_bone_near", "fisher_bone_far",
                "gap_bone_near", "gap_bone_far",
                "bone_purity", "bone_contam_near", "bone_contam_far",
                "auc_ratio_bone_near_over_far",
                "gap_ratio_bone_near_over_far",
                "fisher_ratio_bone_near_over_far",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_pm.columns = [
        "patient_id", "modality",
        "auc_bone_near_mean", "auc_bone_near_std", "auc_bone_near_count",
        "auc_bone_far_mean", "auc_bone_far_std", "auc_bone_far_count",
        "auc_near_far_mean", "auc_near_far_std", "auc_near_far_count",
        "fisher_bone_near_mean", "fisher_bone_near_std", "fisher_bone_near_count",
        "fisher_bone_far_mean", "fisher_bone_far_std", "fisher_bone_far_count",
        "gap_bone_near_mean", "gap_bone_near_std", "gap_bone_near_count",
        "gap_bone_far_mean", "gap_bone_far_std", "gap_bone_far_count",
        "bone_purity_mean", "bone_purity_std", "bone_purity_count",
        "bone_contam_near_mean", "bone_contam_near_std", "bone_contam_near_count",
        "bone_contam_far_mean", "bone_contam_far_std", "bone_contam_far_count",
        "auc_ratio_bone_near_over_far_mean", "auc_ratio_bone_near_over_far_std", "auc_ratio_bone_near_over_far_count",
        "gap_ratio_bone_near_over_far_mean", "gap_ratio_bone_near_over_far_std", "gap_ratio_bone_near_over_far_count",
        "fisher_ratio_bone_near_over_far_mean", "fisher_ratio_bone_near_over_far_std", "fisher_ratio_bone_near_over_far_count",
    ]
    summary_pm_csv = os.path.join(args.out, "summary_by_patient_modality_3class.csv")
    summary_pm.to_csv(summary_pm_csv, index=False, encoding="utf-8-sig")

    # 作图
    plot_metric_bar(df, "auc_bone_near", os.path.join(args.out, "bar_auc_bone_near.png"), "AUC(Bone, Near)")
    plot_metric_bar(df, "auc_bone_far", os.path.join(args.out, "bar_auc_bone_far.png"), "AUC(Bone, Far)")
    plot_metric_bar(df, "auc_near_far", os.path.join(args.out, "bar_auc_near_far.png"), "AUC(Near, Far)")
    plot_metric_bar(df, "fisher_bone_near", os.path.join(args.out, "bar_fisher_bone_near.png"), "Fisher(Bone, Near)")
    plot_metric_bar(df, "fisher_bone_far", os.path.join(args.out, "bar_fisher_bone_far.png"), "Fisher(Bone, Far)")
    plot_metric_bar(df, "gap_bone_near", os.path.join(args.out, "bar_gap_bone_near.png"), "Gap(Bone, Near)")
    plot_metric_bar(df, "gap_bone_far", os.path.join(args.out, "bar_gap_bone_far.png"), "Gap(Bone, Far)")
    plot_metric_bar(df, "bone_purity", os.path.join(args.out, "bar_bone_purity.png"), "Bone purity")
    plot_metric_bar(df, "bone_contam_near", os.path.join(args.out, "bar_bone_contam_near.png"), "Bone -> Near contamination")
    plot_metric_bar(df, "bone_contam_far", os.path.join(args.out, "bar_bone_contam_far.png"), "Bone -> Far contamination")
    plot_all_metrics(df, os.path.join(args.out, "bar_all_metrics_3class.png"))

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved: {all_csv}")
    print(f"Saved: {summary_modality_csv}")
    print(f"Saved: {summary_pm_csv}")
    print("\n[By modality]")
    print(summary_modality.to_string(index=False))

    summary_json = {
        "num_total_slices": int(len(df)),
        "num_groups": int(len(groups)),
        "args": vars(args),
        "modalities": summary_modality.to_dict(orient="records"),
    }
    with open(os.path.join(args.out, "summary_3class.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()