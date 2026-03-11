#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_spider_448_25d_samples.py

功能:
- 从 SPIDER_448_25d/images 和 SPIDER_448_25d/masks 读取预处理后的数据
- 随机挑选 5 个病人
- 每个病人画一张 4x4 图表，展示前 16 张切片
- 在切片上以透明色叠加 mask

数据要求:
ROOT/
├── images/
│   ├── 1_t1_axis0_slice0003.npy
│   ├── ...
├── masks/
│   ├── 1_t1_axis0_slice0003.npy
│   ├── ...
└── meta.csv   (可有可无)

image .npy shape:
(3, 448, 448)

mask .npy shape:
(448, 448)

用法:
python visualize_spider_448_25d_samples.py --root ./SPIDER_448_25d --out ./vis_samples
"""

import os
import re
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_case_and_slice(filename: str):
    """
    文件名示例:
    1_t1_axis0_slice0003.npy

    返回:
    case_id, slice_idx
    """
    stem = os.path.splitext(filename)[0]

    m_case = re.match(r"^([^_]+)_", stem)
    if m_case is None:
        case_id = "unknown"
    else:
        case_id = m_case.group(1)

    m_slice = re.search(r"_slice(\d+)$", stem)
    if m_slice is None:
        slice_idx = -1
    else:
        slice_idx = int(m_slice.group(1))

    return case_id, slice_idx


def collect_cases(root: str):
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
    mask_set = set([f for f in os.listdir(masks_dir) if f.endswith(".npy")])

    cases = defaultdict(list)

    for fname in image_files:
        if fname not in mask_set:
            continue
        case_id, slice_idx = parse_case_and_slice(fname)
        cases[case_id].append((slice_idx, fname))

    # 每个病例按 slice index 排序
    for case_id in cases:
        cases[case_id] = sorted(cases[case_id], key=lambda x: x[0])

    return cases


def load_image_and_mask(root: str, fname: str):
    img_path = os.path.join(root, "images", fname)
    mask_path = os.path.join(root, "masks", fname)

    image = np.load(img_path)
    mask = np.load(mask_path)

    # image: (3, H, W)，取中间通道作为当前 slice
    if image.ndim == 3 and image.shape[0] == 3:
        image2d = image[1]
    elif image.ndim == 2:
        image2d = image
    else:
        raise ValueError(f"Unexpected image shape for {fname}: {image.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape for {fname}: {mask.shape}")

    image2d = image2d.astype(np.float32)
    mask = (mask > 0).astype(np.uint8)

    return image2d, mask


def draw_case_grid(root: str, case_id: str, items, out_dir: str):
    """
    items: [(slice_idx, fname), ...] 已排序
    """
    n_show = min(16, len(items))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(16):
        ax = axes[i]
        ax.axis("off")

        if i >= n_show:
            continue

        slice_idx, fname = items[i]
        image2d, mask = load_image_and_mask(root, fname)

        ax.imshow(image2d, cmap="gray")
        ax.imshow(mask, cmap="Reds", alpha=0.28, vmin=0, vmax=1)
        ax.set_title(f"slice {slice_idx}", fontsize=9)

    fig.suptitle(f"Case {case_id} | first {n_show} slices", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, f"case_{case_id}_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main(args):
    ensure_dir(args.out)

    if args.seed is not None:
        random.seed(args.seed)

    cases = collect_cases(args.root)
    case_ids = sorted(list(cases.keys()))

    if len(case_ids) == 0:
        raise RuntimeError("No valid paired .npy image/mask files found.")

    n_pick = min(args.num_cases, len(case_ids))
    selected = random.sample(case_ids, n_pick)

    print(f"[INFO] Total cases found: {len(case_ids)}")
    print(f"[INFO] Selected cases: {selected}")

    saved_paths = []
    for case_id in selected:
        out_path = draw_case_grid(args.root, case_id, cases[case_id], args.out)
        saved_paths.append(out_path)
        print(f"[Saved] {out_path}")

    # 可选：再做一个总览页，把 5 个病人的图拼成一个清单
    overview_txt = os.path.join(args.out, "selected_cases.txt")
    with open(overview_txt, "w", encoding="utf-8") as f:
        for case_id in selected:
            f.write(f"{case_id}\n")
    print(f"[Saved] {overview_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./SPIDER_448_25d", help="preprocessed dataset root")
    parser.add_argument("--out", type=str, default="./vis_samples", help="output folder")
    parser.add_argument("--num_cases", type=int, default=5, help="number of random cases to visualize")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args)