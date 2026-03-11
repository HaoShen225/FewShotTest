#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_array(path: str):
    """
    兼容 .npy / .npz
    """
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        if len(z.files) == 1:
            return z[z.files[0]]
        elif "arr_0" in z.files:
            return z["arr_0"]
        else:
            raise ValueError(f"无法确定 npz 中的数据键: {path}, keys={z.files}")
    else:
        raise ValueError(f"不支持的文件类型: {path}")


def parse_filename(fname: str):
    """
    解析类似:
      1_t1_axis0_slice0003.npy
      12_t2_axis1_slice0015.npz

    返回:
      case_id, seq, slice_idx
    """
    stem = os.path.splitext(fname)[0]

    m = re.match(r"^(\d+)_(t1|t2)_axis\d+_slice(\d+)$", stem, flags=re.IGNORECASE)
    if m is None:
        return None

    case_id = m.group(1)
    seq = m.group(2).lower()
    slice_idx = int(m.group(3))
    return case_id, seq, slice_idx


def get_middle_three(items):
    """
    items: [(slice_idx, fname), ...]，已按 slice_idx 排序
    返回中间三张；不足三张则返回全部
    """
    n = len(items)
    if n <= 3:
        return items

    mid = n // 2
    start = max(0, mid - 1)
    end = min(n, start + 3)

    # 防止末尾越界导致不足3张
    if end - start < 3:
        start = max(0, end - 3)

    return items[start:end]


def resize_image_to_224(img2d: np.ndarray) -> np.ndarray:
    """
    灰度图: 双线性缩放到 224x224
    输入可为 float 或 uint8
    输出 float32
    """
    x = img2d.astype(np.float32)

    # 如果像素值看起来已经在 [0,1]，按 float 图像处理
    if x.max() <= 1.0 + 1e-6:
        pil_img = Image.fromarray(x, mode="F")
        pil_img = pil_img.resize((224, 224), resample=Image.BILINEAR)
        out = np.array(pil_img, dtype=np.float32)
    else:
        # 否则按 0~255 处理
        x = np.clip(x, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(x, mode="L")
        pil_img = pil_img.resize((224, 224), resample=Image.BILINEAR)
        out = np.array(pil_img, dtype=np.float32)

    return out


def resize_mask_to_224(mask2d: np.ndarray) -> np.ndarray:
    """
    mask: 最近邻缩放到 224x224
    输出 uint8 二值图
    """
    m = (mask2d > 0).astype(np.uint8) * 255
    pil_mask = Image.fromarray(m, mode="L")
    pil_mask = pil_mask.resize((224, 224), resample=Image.NEAREST)
    out = (np.array(pil_mask, dtype=np.uint8) > 0).astype(np.uint8)
    return out


def collect_groups(images_dir: str, masks_dir: str):
    """
    按 (case_id, seq) 分组，保存匹配上的 image/mask 文件
    """
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".npy") or f.endswith(".npz")
    ])
    mask_files = set([
        f for f in os.listdir(masks_dir)
        if f.endswith(".npy") or f.endswith(".npz")
    ])

    groups = defaultdict(list)

    for fname in image_files:
        if fname not in mask_files:
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            continue

        case_id, seq, slice_idx = parsed
        groups[(case_id, seq)].append((slice_idx, fname))

    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x[0])

    return groups


def process_one_file(img_path: str, mask_path: str):
    """
    读取一个 image/mask 对
    image:
      - 若 shape=(3,H,W)，取中间通道 image[1]
      - 若 shape=(H,W)，直接使用
    mask:
      - 期望 shape=(H,W)
    """
    img = load_array(img_path)
    mask = load_array(mask_path)

    if img.ndim == 3:
        if img.shape[0] == 3:
            img2d = img[1]
        else:
            raise ValueError(f"不支持的 image shape: {img.shape} | {img_path}")
    elif img.ndim == 2:
        img2d = img
    else:
        raise ValueError(f"不支持的 image shape: {img.shape} | {img_path}")

    if mask.ndim != 2:
        raise ValueError(f"不支持的 mask shape: {mask.shape} | {mask_path}")

    img224 = resize_image_to_224(img2d)
    mask224 = resize_mask_to_224(mask)

    return img224, mask224


def main(args):
    root_448 = os.path.join(args.base_dir, "SPIDER_448_25d")
    images_dir = os.path.join(root_448, "images")
    masks_dir = os.path.join(root_448, "masks")

    out_root = os.path.join(args.base_dir, "SPIDER_224_2d")
    out_img_t1 = os.path.join(out_root, "images_t1")
    out_img_t2 = os.path.join(out_root, "images_t2")
    out_mask_t1 = os.path.join(out_root, "masks_t1")
    out_mask_t2 = os.path.join(out_root, "masks_t2")

    for p in [out_img_t1, out_img_t2, out_mask_t1, out_mask_t2]:
        ensure_dir(p)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"找不到图像目录: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"找不到掩码目录: {masks_dir}")

    groups = collect_groups(images_dir, masks_dir)

    if len(groups) == 0:
        raise RuntimeError("没有找到可匹配的 image/mask 文件对。")

    total_saved = 0

    for (case_id, seq), items in sorted(groups.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        chosen = get_middle_three(items)

        print(f"[INFO] case={case_id}, seq={seq}, total={len(items)}, selected={[x[0] for x in chosen]}")

        for slice_idx, fname in chosen:
            img_path = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir, fname)

            img224, mask224 = process_one_file(img_path, mask_path)

            stem = os.path.splitext(fname)[0] + "_224"
            img_save_name = stem + ".npy"
            mask_save_name = stem + ".npy"

            if seq == "t1":
                img_save_path = os.path.join(out_img_t1, img_save_name)
                mask_save_path = os.path.join(out_mask_t1, mask_save_name)
            elif seq == "t2":
                img_save_path = os.path.join(out_img_t2, img_save_name)
                mask_save_path = os.path.join(out_mask_t2, mask_save_name)
            else:
                continue

            np.save(img_save_path, img224.astype(np.float32))
            np.save(mask_save_path, mask224.astype(np.uint8))
            total_saved += 1

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"输出目录: {out_root}")
    print(f"总共保存切片数: {total_saved}")
    print(f"每个病人每个序列最多 3 张，因此理论上每组保存 3 张。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="SPIDER_Test1 根目录；若你就在 SPIDER_Test1 中运行，保持默认即可"
    )
    args = parser.parse_args()
    main(args)