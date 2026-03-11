#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_and_standardize_spider.py

功能：
1. 扫描 SPIDER/images 与 SPIDER/masks
2. 跳过 *_SPACE.mha
3. 检查每个 image-mask 对的空间一致性
4. 输出：
   - image size / spacing / origin / direction
   - mask size / spacing / origin / direction
   - sitk.GetArrayFromImage(img).shape
   - sitk.GetArrayFromImage(mask).shape
   - 对于 448x448 输入是否需要重采样
   - 是否疑似 header 不一致
5. 保存详细结果到 CSV

用法：
python check_and_standardize_spider.py --root ./SPIDER
"""

import os
import csv
import argparse
from typing import Tuple, Dict, Any, List

import SimpleITK as sitk


EPS = 1e-6


def is_valid_case(filename: str) -> bool:
    name = filename.lower()
    return name.endswith(".mha") and ("_space.mha" not in name)


def tuple_close(a, b, eps=EPS) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


def read_meta(path: str) -> Dict[str, Any]:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)

    meta = {
        "path": path,
        "size": tuple(int(v) for v in img.GetSize()),              # (x, y, z)
        "spacing": tuple(float(v) for v in img.GetSpacing()),
        "origin": tuple(float(v) for v in img.GetOrigin()),
        "direction": tuple(float(v) for v in img.GetDirection()),
        "array_shape": tuple(int(v) for v in arr.shape),           # (z, y, x)
    }
    return meta


def infer_case_id_and_modality(filename: str) -> Tuple[str, str]:
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    case_id = parts[0]
    modality = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
    return case_id, modality


def needs_448_resample_from_array_shape(arr_shape: Tuple[int, ...]) -> bool:
    """
    arr_shape = (z, y, x)
    对于 2D/2.5D slice 输入 DINOv2 448x448，
    只关心每张 slice 的 in-plane 是否已经是 448x448。
    """
    if len(arr_shape) != 3:
        return True
    _, h, w = arr_shape
    return not (h == 448 and w == 448)


def detect_header_inconsistency(img_meta: Dict[str, Any], mask_meta: Dict[str, Any]) -> Tuple[bool, str]:
    """
    判断是否疑似 header 不一致：
    - size 一样，但 spacing/origin/direction 不同
    - 或 array shape 一样，但空间 header 不同
    """
    reasons = []

    size_same = img_meta["size"] == mask_meta["size"]
    arr_same = img_meta["array_shape"] == mask_meta["array_shape"]
    spacing_same = tuple_close(img_meta["spacing"], mask_meta["spacing"])
    origin_same = tuple_close(img_meta["origin"], mask_meta["origin"])
    direction_same = tuple_close(img_meta["direction"], mask_meta["direction"])

    if size_same and not spacing_same:
        reasons.append("same_size_but_spacing_diff")
    if size_same and not origin_same:
        reasons.append("same_size_but_origin_diff")
    if size_same and not direction_same:
        reasons.append("same_size_but_direction_diff")
    if arr_same and (not spacing_same or not origin_same or not direction_same):
        reasons.append("same_array_shape_but_header_diff")

    suspicious = len(reasons) > 0
    return suspicious, ";".join(reasons) if reasons else ""


def collect_valid_pairs(root: str) -> List[Tuple[str, str, str]]:
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    image_files = sorted([f for f in os.listdir(images_dir) if is_valid_case(f)])
    mask_files = sorted([f for f in os.listdir(masks_dir) if is_valid_case(f)])

    image_set = set(image_files)
    mask_set = set(mask_files)
    common = sorted(image_set & mask_set)

    pairs = []
    for fname in common:
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        pairs.append((fname, img_path, mask_path))
    return pairs


def pretty_tuple(x) -> str:
    return "(" + ", ".join(str(v) for v in x) + ")"


def main(root: str):
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    print(f"[INFO] Root:   {root}")
    print(f"[INFO] Images: {images_dir}")
    print(f"[INFO] Masks:  {masks_dir}")

    pairs = collect_valid_pairs(root)
    print(f"[INFO] Valid non-SPACE paired cases: {len(pairs)}")

    out_rows = []

    header_diff_count = 0
    need_448_resample_count = 0
    exact_space_match_count = 0

    for idx, (fname, img_path, mask_path) in enumerate(pairs, 1):
        case_id, modality = infer_case_id_and_modality(fname)

        try:
            img_meta = read_meta(img_path)
            mask_meta = read_meta(mask_path)
        except Exception as e:
            print(f"\n[ERROR] {fname}: {e}")
            continue

        size_match = img_meta["size"] == mask_meta["size"]
        array_shape_match = img_meta["array_shape"] == mask_meta["array_shape"]
        spacing_match = tuple_close(img_meta["spacing"], mask_meta["spacing"])
        origin_match = tuple_close(img_meta["origin"], mask_meta["origin"])
        direction_match = tuple_close(img_meta["direction"], mask_meta["direction"])

        exact_space_match = size_match and spacing_match and origin_match and direction_match
        if exact_space_match:
            exact_space_match_count += 1

        need_448_resample = needs_448_resample_from_array_shape(img_meta["array_shape"])
        if need_448_resample:
            need_448_resample_count += 1

        suspicious_header, suspicious_reason = detect_header_inconsistency(img_meta, mask_meta)
        if suspicious_header:
            header_diff_count += 1

        print("\n" + "=" * 100)
        print(f"[{idx:03d}/{len(pairs)}] {fname} | case={case_id} | modality={modality}")
        print("-" * 100)
        print(f"image size      : {pretty_tuple(img_meta['size'])}")
        print(f"image spacing   : {pretty_tuple(img_meta['spacing'])}")
        print(f"image origin    : {pretty_tuple(img_meta['origin'])}")
        print(f"image direction : {pretty_tuple(img_meta['direction'])}")
        print(f"image array     : {pretty_tuple(img_meta['array_shape'])}")

        print(f"mask  size      : {pretty_tuple(mask_meta['size'])}")
        print(f"mask  spacing   : {pretty_tuple(mask_meta['spacing'])}")
        print(f"mask  origin    : {pretty_tuple(mask_meta['origin'])}")
        print(f"mask  direction : {pretty_tuple(mask_meta['direction'])}")
        print(f"mask  array     : {pretty_tuple(mask_meta['array_shape'])}")

        print(f"size match              : {size_match}")
        print(f"array shape match       : {array_shape_match}")
        print(f"spacing match           : {spacing_match}")
        print(f"origin match            : {origin_match}")
        print(f"direction match         : {direction_match}")
        print(f"need resample to 448x448: {need_448_resample}")
        print(f"suspicious header diff  : {suspicious_header}")
        if suspicious_header:
            print(f"header diff reason      : {suspicious_reason}")

        out_rows.append({
            "filename": fname,
            "case_id": case_id,
            "modality": modality,

            "image_path": img_path,
            "image_size": str(img_meta["size"]),
            "image_spacing": str(img_meta["spacing"]),
            "image_origin": str(img_meta["origin"]),
            "image_direction": str(img_meta["direction"]),
            "image_array_shape": str(img_meta["array_shape"]),

            "mask_path": mask_path,
            "mask_size": str(mask_meta["size"]),
            "mask_spacing": str(mask_meta["spacing"]),
            "mask_origin": str(mask_meta["origin"]),
            "mask_direction": str(mask_meta["direction"]),
            "mask_array_shape": str(mask_meta["array_shape"]),

            "size_match": size_match,
            "array_shape_match": array_shape_match,
            "spacing_match": spacing_match,
            "origin_match": origin_match,
            "direction_match": direction_match,
            "exact_space_match": exact_space_match,

            "need_resample_to_448x448": need_448_resample,
            "suspicious_header_inconsistency": suspicious_header,
            "suspicious_reason": suspicious_reason,
        })

    out_csv = os.path.join(root, "spider_space_check.csv")
    fieldnames = list(out_rows[0].keys()) if out_rows else []

    if out_rows:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    print("\n" + "#" * 100)
    print("SUMMARY")
    print("#" * 100)
    print(f"Total valid paired non-SPACE cases : {len(out_rows)}")
    print(f"Exact image-mask space match       : {exact_space_match_count}")
    print(f"Need resample to 448x448           : {need_448_resample_count}")
    print(f"Suspicious header inconsistency    : {header_diff_count}")
    print(f"Saved CSV                          : {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./SPIDER_448_25d",
        help="SPIDER dataset root, e.g. ./SPIDER"
    )
    args = parser.parse_args()
    main(args.root)