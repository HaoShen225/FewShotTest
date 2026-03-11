#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_spider_to_dinov2_448_25d.py

流程:
1. 读取 image / mask
2. 跳过 _SPACE
3. 检查 image-mask header 一致性
4. 根据 spacing 选 slice axis
5. 提取 2D slice
6. 二值化 mask
7. 去掉全背景或极小前景 slice
8. 做 body-aware 或 ROI-aware 强度归一化
9. 根据 mask 做 ROI crop
10. 等比例 resize + pad 到 448x448
11. 构造 3 通道输入（默认 2.5D）
12. 保存 image / mask / meta

输出:
OUT/
├── images/   # 3x448x448 float32 .npy
├── masks/    # 448x448 uint8 .npy
└── meta.csv

可选:
--export_png 导出中间预览 png（取3通道中间通道）
"""

import os
import csv
import argparse
from typing import Tuple, List, Dict, Any

import numpy as np
import SimpleITK as sitk
from PIL import Image


EPS = 1e-6


def is_valid_case(filename: str) -> bool:
    name = filename.lower()
    return name.endswith(".mha") and ("_space.mha" not in name)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_case_id_and_modality(filename: str) -> Tuple[str, str]:
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    case_id = parts[0]
    modality = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
    return case_id, modality


def tuple_close(a, b, eps=EPS) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


def read_sitk_image(path: str):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    size = tuple(int(v) for v in img.GetSize())  # (x, y, z)
    spacing = tuple(float(v) for v in img.GetSpacing())  # (x, y, z)
    origin = tuple(float(v) for v in img.GetOrigin())
    direction = tuple(float(v) for v in img.GetDirection())
    return img, arr, size, spacing, origin, direction


def check_header_consistency(
    img_size, img_spacing, img_origin, img_direction,
    mask_size, mask_spacing, mask_origin, mask_direction,
) -> Tuple[bool, str]:
    reasons = []

    if img_size != mask_size:
        reasons.append("size_diff")
    if not tuple_close(img_spacing, mask_spacing):
        reasons.append("spacing_diff")
    if not tuple_close(img_origin, mask_origin):
        reasons.append("origin_diff")
    if not tuple_close(img_direction, mask_direction):
        reasons.append("direction_diff")

    ok = len(reasons) == 0
    return ok, ";".join(reasons)


def choose_slice_axis_from_spacing(spacing_xyz: Tuple[float, float, float]) -> int:
    """
    spacing 最大的轴视为厚层轴，并把它作为切片轴。
    返回:
      0 -> x
      1 -> y
      2 -> z
    """
    return int(np.argmax(np.asarray(spacing_xyz)))


def extract_slices_by_axis(arr_zyx: np.ndarray, axis_xyz: int) -> List[np.ndarray]:
    """
    arr_zyx: (z, y, x)
    axis_xyz:
      0 -> 固定 x, 切 arr[:, :, i], slice=(z, y)
      1 -> 固定 y, 切 arr[:, i, :], slice=(z, x)
      2 -> 固定 z, 切 arr[i, :, :], slice=(y, x)
    """
    out = []
    if axis_xyz == 0:
        for i in range(arr_zyx.shape[2]):
            out.append(arr_zyx[:, :, i])
    elif axis_xyz == 1:
        for i in range(arr_zyx.shape[1]):
            out.append(arr_zyx[:, i, :])
    elif axis_xyz == 2:
        for i in range(arr_zyx.shape[0]):
            out.append(arr_zyx[i, :, :])
    else:
        raise ValueError(f"Invalid axis_xyz: {axis_xyz}")
    return out


def binarize_mask(mask2d: np.ndarray) -> np.ndarray:
    return (mask2d > 0).astype(np.uint8)


def is_small_foreground(mask2d: np.ndarray, min_fg_pixels: int) -> bool:
    return int(mask2d.sum()) < int(min_fg_pixels)


def body_mask_from_image(img2d: np.ndarray, threshold_percentile: float = 10.0) -> np.ndarray:
    """
    粗 body mask：对非零强度区域做一个较稳健阈值
    """
    x = img2d.astype(np.float32)
    nz = x[x > 0]
    if nz.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    thr = np.percentile(nz, threshold_percentile)
    body = (x > thr).astype(np.uint8)

    # 若 body 太小，则退化为非零区域
    if body.sum() < max(16, 0.01 * x.size):
        body = (x > 0).astype(np.uint8)
    return body.astype(np.uint8)


def robust_minmax_from_region(img2d: np.ndarray, region_mask: np.ndarray,
                              q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    """
    在指定区域内统计分位数，再映射到 [0,1]
    """
    x = img2d.astype(np.float32)
    vals = x[region_mask > 0]
    if vals.size == 0:
        vals = x[x > 0]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo = np.percentile(vals, q_low)
    hi = np.percentile(vals, q_high)
    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def normalize_slice(img2d: np.ndarray, mask2d: np.ndarray, norm_mode: str = "roi") -> np.ndarray:
    """
    norm_mode:
      roi  -> ROI-aware，用 mask 区域统计
      body -> body-aware，用 body mask 统计
    """
    if norm_mode == "roi":
        region = (mask2d > 0).astype(np.uint8)
        return robust_minmax_from_region(img2d, region)
    elif norm_mode == "body":
        region = body_mask_from_image(img2d)
        return robust_minmax_from_region(img2d, region)
    else:
        raise ValueError(f"Unsupported norm_mode: {norm_mode}")


def bbox_from_mask(mask2d: np.ndarray, margin_ratio: float = 0.15) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask2d > 0)
    H, W = mask2d.shape

    if len(ys) == 0 or len(xs) == 0:
        return 0, H, 0, W

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    bh = y2 - y1
    bw = x2 - x1
    my = max(1, int(round(bh * margin_ratio)))
    mx = max(1, int(round(bw * margin_ratio)))

    y1 = max(0, y1 - my)
    y2 = min(H, y2 + my)
    x1 = max(0, x1 - mx)
    x2 = min(W, x2 + mx)
    return y1, y2, x1, x2


def square_expand_bbox(y1: int, y2: int, x1: int, x2: int, H: int, W: int,
                       square_expand_ratio: float = 1.15) -> Tuple[int, int, int, int]:
    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0
    bh = y2 - y1
    bw = x2 - x1
    side = int(np.ceil(max(bh, bw) * square_expand_ratio))
    side = max(side, 8)

    ny1 = int(round(cy - side / 2))
    ny2 = ny1 + side
    nx1 = int(round(cx - side / 2))
    nx2 = nx1 + side

    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny2 > H:
        shift = ny2 - H
        ny1 -= shift
        ny2 = H
    if nx2 > W:
        shift = nx2 - W
        nx1 -= shift
        nx2 = W

    ny1 = max(0, ny1)
    nx1 = max(0, nx1)
    ny2 = min(H, ny2)
    nx2 = min(W, nx2)
    return ny1, ny2, nx1, nx2


def crop_pair_by_mask(img2d: np.ndarray, mask2d: np.ndarray,
                      margin_ratio: float = 0.15,
                      square_expand_ratio: float = 1.15):
    H, W = mask2d.shape
    y1, y2, x1, x2 = bbox_from_mask(mask2d, margin_ratio=margin_ratio)
    y1, y2, x1, x2 = square_expand_bbox(y1, y2, x1, x2, H, W, square_expand_ratio=square_expand_ratio)

    img_crop = img2d[y1:y2, x1:x2]
    mask_crop = mask2d[y1:y2, x1:x2]
    return img_crop, mask_crop, (y1, y2, x1, x2)


def resize_keep_aspect_and_pad(img2d: np.ndarray, out_size: int, is_mask: bool):
    H, W = img2d.shape
    scale = float(out_size) / max(H, W)
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))

    if is_mask:
        src = Image.fromarray(img2d.astype(np.uint8), mode="L")
        dst = src.resize((new_w, new_h), resample=Image.NEAREST)
        arr = np.array(dst, dtype=np.uint8)
    else:
        src = Image.fromarray(img2d.astype(np.float32), mode="F")
        dst = src.resize((new_w, new_h), resample=Image.BILINEAR)
        arr = np.array(dst, dtype=np.float32)

    pad_top = (out_size - new_h) // 2
    pad_bottom = out_size - new_h - pad_top
    pad_left = (out_size - new_w) // 2
    pad_right = out_size - new_w - pad_left

    if is_mask:
        out = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode="constant", constant_values=0).astype(np.uint8)
    else:
        out = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode="constant", constant_values=0.0).astype(np.float32)

    info = {
        "orig_h": H, "orig_w": W,
        "new_h": new_h, "new_w": new_w,
        "scale": scale,
        "pad_top": pad_top, "pad_bottom": pad_bottom,
        "pad_left": pad_left, "pad_right": pad_right,
    }
    return out, info


def get_neighbor_index(center: int, delta: int, n: int) -> int:
    idx = center + delta
    idx = max(0, min(n - 1, idx))
    return idx


def build_25d_triplet(processed_slices: List[np.ndarray], idx: int) -> np.ndarray:
    """
    processed_slices: List of 2D float32 arrays, each (448,448)
    return: (3,448,448)
    """
    n = len(processed_slices)
    i0 = get_neighbor_index(idx, -1, n)
    i1 = idx
    i2 = get_neighbor_index(idx, +1, n)
    triplet = np.stack([
        processed_slices[i0],
        processed_slices[i1],
        processed_slices[i2],
    ], axis=0).astype(np.float32)
    return triplet


def save_png_image(img01: np.ndarray, path: str):
    x = np.clip(img01, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    Image.fromarray(x, mode="L").save(path)


def save_png_mask(mask: np.ndarray, path: str):
    x = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(x, mode="L").save(path)


def collect_pairs(root: str) -> List[Tuple[str, str, str]]:
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    image_files = sorted([f for f in os.listdir(images_dir) if is_valid_case(f)])
    mask_files = sorted([f for f in os.listdir(masks_dir) if is_valid_case(f)])
    common = sorted(set(image_files) & set(mask_files))

    return [
        (fname, os.path.join(images_dir, fname), os.path.join(masks_dir, fname))
        for fname in common
    ]


def process_case(
    fname: str,
    img_path: str,
    mask_path: str,
    out_root: str,
    out_size: int,
    min_fg_pixels: int,
    norm_mode: str,
    margin_ratio: float,
    square_expand_ratio: float,
    export_png: bool,
) -> List[Dict[str, Any]]:

    case_id, modality = infer_case_id_and_modality(fname)

    _, img_arr, img_size, img_spacing, img_origin, img_direction = read_sitk_image(img_path)
    _, mask_arr, mask_size, mask_spacing, mask_origin, mask_direction = read_sitk_image(mask_path)

    header_ok, header_reason = check_header_consistency(
        img_size, img_spacing, img_origin, img_direction,
        mask_size, mask_spacing, mask_origin, mask_direction,
    )

    # 尺寸不一致直接报错
    if img_arr.shape != mask_arr.shape:
        raise RuntimeError(f"Array shape mismatch: image={img_arr.shape}, mask={mask_arr.shape}")
    if img_size != mask_size:
        raise RuntimeError(f"Image/mask size mismatch: image={img_size}, mask={mask_size}")

    # 4. 根据 spacing 选 slice axis
    slice_axis_xyz = choose_slice_axis_from_spacing(img_spacing)

    # 5. 提取 2D slice
    img_slices_raw = extract_slices_by_axis(img_arr, slice_axis_xyz)
    mask_slices_raw = extract_slices_by_axis(mask_arr, slice_axis_xyz)

    # 先逐 slice 做到 448x448 的单通道处理结果
    processed_img_slices = []
    processed_mask_slices = []
    slice_meta = []

    for idx, (img2d_raw, mask2d_raw) in enumerate(zip(img_slices_raw, mask_slices_raw)):
        # 6. 二值化 mask
        mask2d = binarize_mask(mask2d_raw)

        # 7. 去掉全背景或极小前景 slice
        if is_small_foreground(mask2d, min_fg_pixels=min_fg_pixels):
            continue

        # 8. body-aware 或 ROI-aware 强度归一化
        img2d_norm = normalize_slice(img2d_raw, mask2d, norm_mode=norm_mode)

        # 9. 根据 mask 做 ROI crop
        img_crop, mask_crop, bbox = crop_pair_by_mask(
            img2d_norm, mask2d,
            margin_ratio=margin_ratio,
            square_expand_ratio=square_expand_ratio
        )

        # 10. 等比例 resize + pad 到 448x448
        img448, img_resize_info = resize_keep_aspect_and_pad(img_crop, out_size=out_size, is_mask=False)
        mask448, mask_resize_info = resize_keep_aspect_and_pad(mask_crop, out_size=out_size, is_mask=True)

        processed_img_slices.append(img448.astype(np.float32))
        processed_mask_slices.append(mask448.astype(np.uint8))
        slice_meta.append({
            "orig_index": idx,
            "raw_shape": tuple(img2d_raw.shape),
            "crop_bbox": bbox,
            "crop_shape": tuple(img_crop.shape),
            "resize_info": img_resize_info,
            "fg_before": int(mask2d.sum()),
            "fg_after": int(mask448.sum()),
        })

    if len(processed_img_slices) == 0:
        return []

    # 11. 构造 3 通道输入（2.5D）
    img_out_dir = os.path.join(out_root, "images")
    mask_out_dir = os.path.join(out_root, "masks")
    ensure_dir(img_out_dir)
    ensure_dir(mask_out_dir)

    if export_png:
        img_png_dir = os.path.join(out_root, "images_png")
        mask_png_dir = os.path.join(out_root, "masks_png")
        ensure_dir(img_png_dir)
        ensure_dir(mask_png_dir)

    rows = []
    for new_idx in range(len(processed_img_slices)):
        triplet = build_25d_triplet(processed_img_slices, new_idx)  # (3,H,W)
        mask448 = processed_mask_slices[new_idx]
        meta = slice_meta[new_idx]
        orig_index = meta["orig_index"]

        stem = f"{case_id}_{modality}_axis{slice_axis_xyz}_slice{orig_index:04d}"

        img_npy_path = os.path.join(img_out_dir, stem + ".npy")
        mask_npy_path = os.path.join(mask_out_dir, stem + ".npy")

        np.save(img_npy_path, triplet.astype(np.float32))
        np.save(mask_npy_path, mask448.astype(np.uint8))

        if export_png:
            save_png_image(triplet[1], os.path.join(img_png_dir, stem + ".png"))
            save_png_mask(mask448, os.path.join(mask_png_dir, stem + ".png"))

        y1, y2, x1, x2 = meta["crop_bbox"]
        rows.append({
            "filename": fname,
            "case_id": case_id,
            "modality": modality,
            "header_ok": header_ok,
            "header_reason": header_reason,
            "input_volume_shape_zyx": str(tuple(img_arr.shape)),
            "input_size_xyz": str(tuple(img_size)),
            "input_spacing_xyz": str(tuple(img_spacing)),
            "slice_axis_xyz": slice_axis_xyz,
            "slice_index_raw": orig_index,
            "slice_raw_shape": str(meta["raw_shape"]),
            "crop_bbox_y1y2x1x2": str((y1, y2, x1, x2)),
            "crop_shape": str(meta["crop_shape"]),
            "out_shape_image": str(tuple(triplet.shape)),
            "out_shape_mask": str(tuple(mask448.shape)),
            "foreground_pixels_before": meta["fg_before"],
            "foreground_pixels_after": meta["fg_after"],
            "norm_mode": norm_mode,
            "image_npy": img_npy_path,
            "mask_npy": mask_npy_path,
        })

    return rows


def main(args):
    ensure_dir(args.out)
    pairs = collect_pairs(args.root)

    print(f"[INFO] Root: {args.root}")
    print(f"[INFO] Out : {args.out}")
    print(f"[INFO] Paired non-SPACE cases: {len(pairs)}")
    print(f"[INFO] Norm mode: {args.norm_mode}")
    print(f"[INFO] Min foreground pixels per slice: {args.min_fg_pixels}")

    all_rows = []
    ok_cases = 0
    fail_cases = 0

    for i, (fname, img_path, mask_path) in enumerate(pairs, 1):
        print(f"[{i:03d}/{len(pairs)}] {fname}")
        try:
            rows = process_case(
                fname=fname,
                img_path=img_path,
                mask_path=mask_path,
                out_root=args.out,
                out_size=args.out_size,
                min_fg_pixels=args.min_fg_pixels,
                norm_mode=args.norm_mode,
                margin_ratio=args.margin_ratio,
                square_expand_ratio=args.square_expand_ratio,
                export_png=args.export_png,
            )
            all_rows.extend(rows)
            ok_cases += 1
        except Exception as e:
            fail_cases += 1
            print(f"  [ERROR] {fname}: {e}")

    meta_csv = os.path.join(args.out, "meta.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(meta_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Processed cases : {ok_cases}")
    print(f"Failed cases    : {fail_cases}")
    print(f"Exported slices : {len(all_rows)}")
    print(f"Meta CSV        : {meta_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./SPIDER", help="SPIDER root")
    parser.add_argument("--out", type=str, default="./SPIDER_448_25d", help="output root")
    parser.add_argument("--out_size", type=int, default=448, help="output size")
    parser.add_argument("--min_fg_pixels", type=int, default=10, help="skip slice if fg pixels < this")
    parser.add_argument("--norm_mode", type=str, default="roi", choices=["roi", "body"],
                        help="normalization mode: roi or body")
    parser.add_argument("--margin_ratio", type=float, default=0.15, help="bbox margin ratio")
    parser.add_argument("--square_expand_ratio", type=float, default=1.15, help="expand bbox toward square")
    parser.add_argument("--export_png", action="store_true", help="also export png previews")
    args = parser.parse_args()
    main(args)