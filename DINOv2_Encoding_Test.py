#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINOv2_Encoding_AllSlices.py

对 SPIDER_448_25d 中所有切片测试 DINOv2 输出特征的前景/背景可分性，
并用柱状图呈现 AUC / Fisher ratio / Gap / kNN purity。

输出:
out/
├── all_slice_metrics.csv
├── summary_by_modality.csv
├── summary_by_patient_modality.csv
├── bar_auc.png
├── bar_fisher.png
├── bar_gap.png
├── bar_knn_purity.png
└── bar_all_metrics.png

依赖:
pip install torch torchvision numpy pandas matplotlib

用法:
python DINOv2_Encoding_AllSlices.py
或
python DINOv2_Encoding_AllSlices.py --root ./SPIDER_448_25d --out ./dinov2_all_diag
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


# =========================================================
# 基础工具
# =========================================================
FILE_RE = re.compile(
    r"^(?P<case>[^_]+)_(?P<modality>.+?)_axis(?P<axis>\d+)_slice(?P<slice>\d+)\.(?P<ext>npy|npz)$"
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    else:
        raise ValueError(f"Unsupported file type: {path}")


def collect_all_groups(root: str):
    """
    返回:
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
    image_chw: (3,H,W) float32 in [0,1]
    return:
      feat: (N,C) torch.Tensor on CPU
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
def to_image_center_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr[1].astype(np.float32)
    elif arr.ndim == 2:
        return arr.astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")


def to_image_three_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 3:
        x = arr.astype(np.float32)
    elif arr.ndim == 2:
        x = np.stack([arr, arr, arr], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x


def to_binary_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {arr.shape}")
    return (arr > 0).astype(np.uint8)


def downsample_mask_to_tokens(mask_hw: np.ndarray, gh: int, gw: int) -> np.ndarray:
    m = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    cov = F.adaptive_avg_pool2d(m, output_size=(gh, gw))[0, 0].numpy()
    return cov


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


def fisher_ratio(scores_fg: np.ndarray, scores_bg: np.ndarray) -> float:
    if len(scores_fg) == 0 or len(scores_bg) == 0:
        return np.nan
    mu_f = float(np.mean(scores_fg))
    mu_b = float(np.mean(scores_bg))
    var_f = float(np.var(scores_fg))
    var_b = float(np.var(scores_bg))
    return float((mu_f - mu_b) ** 2 / (var_f + var_b + 1e-12))


def score_gap(scores_fg: np.ndarray, scores_bg: np.ndarray) -> float:
    if len(scores_fg) == 0 or len(scores_bg) == 0:
        return np.nan
    return float(np.mean(scores_fg) - np.mean(scores_bg))


def knn_purity_cosine(features: np.ndarray, labels: np.ndarray, k: int = 7) -> float:
    x = torch.from_numpy(features.astype(np.float32))
    x = F.normalize(x, dim=1)
    sim = x @ x.T
    n = sim.shape[0]
    if n <= 1:
        return np.nan
    kk = min(k, n - 1)
    sim.fill_diagonal_(-1e9)
    nn_idx = torch.topk(sim, k=kk, dim=1).indices

    y = torch.from_numpy(labels.astype(np.int64))
    nn_labels = y[nn_idx]
    purity = (nn_labels == y[:, None]).float().mean(dim=1).mean().item()
    return float(purity)


# =========================================================
# prototype 构造
# =========================================================
def build_global_prototypes(
    all_feats: List[torch.Tensor],
    all_fg_masks: List[np.ndarray],
    all_bg_masks: List[np.ndarray],
    exclude_idx: Optional[int] = None,
):
    fg_list = []
    bg_list = []

    for i, feat in enumerate(all_feats):
        if exclude_idx is not None and i == exclude_idx:
            continue
        fg_mask = all_fg_masks[i]
        bg_mask = all_bg_masks[i]

        if fg_mask.any():
            fg_list.append(feat[fg_mask])
        if bg_mask.any():
            bg_list.append(feat[bg_mask])

    if len(fg_list) == 0 or len(bg_list) == 0:
        return None, None

    fg = torch.cat(fg_list, dim=0)
    bg = torch.cat(bg_list, dim=0)

    p_f = F.normalize(fg.mean(dim=0, keepdim=True), dim=1)[0]
    p_b = F.normalize(bg.mean(dim=0, keepdim=True), dim=1)[0]
    return p_f, p_b


# =========================================================
# 核心诊断
# =========================================================
def diagnose_group(
    model,
    device: torch.device,
    patient_id: str,
    modality: str,
    items: List[Dict],
    fg_cov_thr: float,
    bg_cov_thr: float,
    k: int,
    leave_one_out: bool,
):
    """
    对一个 (patient_id, modality) 组内全部切片做诊断，返回该组所有 slice 的 metrics rows
    """
    all_feats = []
    all_fg_masks = []
    all_bg_masks = []
    all_items = []

    for item in items:
        image_arr = load_array(item["image_path"])
        mask_arr = load_array(item["mask_path"])

        image_3ch = to_image_three_channel(image_arr)
        mask_hw = to_binary_mask(mask_arr)

        feat, gh, gw = extract_patch_tokens(model, image_3ch, device=device)
        cov = downsample_mask_to_tokens(mask_hw, gh, gw)

        fg_token = (cov >= fg_cov_thr).reshape(-1)
        bg_token = (cov <= bg_cov_thr).reshape(-1)

        all_feats.append(feat)
        all_fg_masks.append(fg_token)
        all_bg_masks.append(bg_token)
        all_items.append({
            **item,
            "grid_h": gh,
            "grid_w": gw,
        })

    rows = []
    for idx, feat in enumerate(all_feats):
        p_f, p_b = build_global_prototypes(
            all_feats, all_fg_masks, all_bg_masks,
            exclude_idx=idx if leave_one_out else None
        )

        if p_f is None or p_b is None:
            continue

        x = F.normalize(feat, dim=1)
        score = (x @ p_f) - (x @ p_b)
        score_np = score.numpy()

        fg_mask = all_fg_masks[idx]
        bg_mask = all_bg_masks[idx]
        valid = fg_mask | bg_mask

        n_fg = int(fg_mask.sum())
        n_bg = int(bg_mask.sum())
        if n_fg == 0 or n_bg == 0:
            continue

        labels = np.full(score_np.shape, -1, dtype=np.int64)
        labels[bg_mask] = 0
        labels[fg_mask] = 1

        valid_scores = score_np[valid]
        valid_labels = labels[valid]
        valid_feats = feat.numpy()[valid]

        scores_fg = valid_scores[valid_labels == 1]
        scores_bg = valid_scores[valid_labels == 0]

        auc = roc_auc_from_scores(valid_scores, valid_labels)
        fisher = fisher_ratio(scores_fg, scores_bg)
        gap = score_gap(scores_fg, scores_bg)
        purity = knn_purity_cosine(valid_feats, valid_labels, k=k)

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
            "fg_cov_thr": fg_cov_thr,
            "bg_cov_thr": bg_cov_thr,
            "n_fg_tokens": n_fg,
            "n_bg_tokens": n_bg,
            "n_valid_tokens": int(valid.sum()),
            "auc": auc,
            "fisher": fisher,
            "gap": gap,
            "knn_purity": purity,
        })

        print(
            f"[{patient_id}][{modality}] slice={item['slice']:04d} "
            f"AUC={auc:.4f} Fisher={fisher:.4f} Gap={gap:.4f} Purity={purity:.4f}"
        )

    return rows


# =========================================================
# 作图
# =========================================================
def plot_metric_bar(df: pd.DataFrame, metric: str, out_path: str, title: str):
    """
    对所有切片画柱状图
    """
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
    fig, axes = plt.subplots(4, 1, figsize=(width, 14), sharex=True)

    metrics = [
        ("auc", "AUC"),
        ("fisher", "Fisher ratio"),
        ("gap", "Gap"),
        ("knn_purity", "kNN purity"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        ax.bar(x, df[col].tolist(), color=colors, width=0.8)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_title("DINOv2 separability metrics across all slices")
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
        default=os.path.join(BASE_DIR, "dinov2_all_diag"),
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
            k=args.k,
            leave_one_out=args.leave_one_out,
        )
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise RuntimeError("No valid slice metrics were produced.")

    df = pd.DataFrame(all_rows).sort_values(["patient_id", "modality", "slice_index"]).reset_index(drop=True)

    # 保存逐切片 CSV
    all_csv = os.path.join(args.out, "all_slice_metrics.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    # 按模态汇总
    summary_modality = (
        df.groupby("modality")[["auc", "fisher", "gap", "knn_purity"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_modality.columns = [
        "modality",
        "auc_mean", "auc_std", "auc_count",
        "fisher_mean", "fisher_std", "fisher_count",
        "gap_mean", "gap_std", "gap_count",
        "knn_purity_mean", "knn_purity_std", "knn_purity_count",
    ]
    summary_modality_csv = os.path.join(args.out, "summary_by_modality.csv")
    summary_modality.to_csv(summary_modality_csv, index=False, encoding="utf-8-sig")

    # 按病人+模态汇总
    summary_pm = (
        df.groupby(["patient_id", "modality"])[["auc", "fisher", "gap", "knn_purity"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_pm.columns = [
        "patient_id", "modality",
        "auc_mean", "auc_std", "auc_count",
        "fisher_mean", "fisher_std", "fisher_count",
        "gap_mean", "gap_std", "gap_count",
        "knn_purity_mean", "knn_purity_std", "knn_purity_count",
    ]
    summary_pm_csv = os.path.join(args.out, "summary_by_patient_modality.csv")
    summary_pm.to_csv(summary_pm_csv, index=False, encoding="utf-8-sig")

    # 作图
    plot_metric_bar(df, "auc", os.path.join(args.out, "bar_auc.png"), "AUC across all slices")
    plot_metric_bar(df, "fisher", os.path.join(args.out, "bar_fisher.png"), "Fisher ratio across all slices")
    plot_metric_bar(df, "gap", os.path.join(args.out, "bar_gap.png"), "Gap across all slices")
    plot_metric_bar(df, "knn_purity", os.path.join(args.out, "bar_knn_purity.png"), "kNN purity across all slices")
    plot_all_metrics(df, os.path.join(args.out, "bar_all_metrics.png"))

    # 控制台摘要
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved: {all_csv}")
    print(f"Saved: {summary_modality_csv}")
    print(f"Saved: {summary_pm_csv}")
    print("\n[By modality]")
    print(summary_modality.to_string(index=False))

    # 同时保存 json 摘要
    summary_json = {
        "num_total_slices": int(len(df)),
        "num_groups": int(len(groups)),
        "modalities": summary_modality.to_dict(orient="records"),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()