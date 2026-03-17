
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


# =========================================================
# 数据清理：
# 剔除所有后缀为_SPACE的样本
# 仅保留标签 1–7 和 201–207 全部都存在的样本 (掩码仅保留椎骨/椎间盘，其他视作杂类，暂不处理。)
# 剔除所有压扁的（人眼过滤）T1={133,174,186,221,227,253} T2={35,58,133,174,186,221,227,253}
# 用最大 spacing 对应厚层轴
# 朝向统一
# 物理空间统一
# 解剖范围统一
# 数组尺寸统一
# 消除场强偏移 (N4 bias correction, ROI 内 robust z-score)
# 将清洗好的样本放入SPIDER_normalized文件夹。
# =========================================================
@dataclass
class Cfg:
    src_root: str = "SPIDER"
    dst_root: str = "SPIDER_normalized"

    image_dir: str = "images"
    mask_dir: str = "masks"

    # keep only these labels, and require all of them to exist
    keep_labels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 201, 202, 203, 204, 205, 206, 207)

    # manual flattened cases to remove
    flattened_t1: Tuple[int, ...] = (133, 174, 186, 221, 227, 253)
    flattened_t2: Tuple[int, ...] = (35, 58, 133, 174, 186, 221, 227, 253)

    # canonical orientation after reorientation
    canonical_orient: str = "LPS"

    # target spacing will be estimated from retained cohort after reorientation
    # if not None, it overrides the auto-estimated spacing
    force_target_spacing: Optional[Tuple[float, float, float]] = None

    # crop size is estimated from the retained cohort after resampling
    # if not None, it overrides auto-estimation
    force_crop_size_xyz: Optional[Tuple[int, int, int]] = None

    # crop-size estimation
    crop_bbox_percentile: float = 95.0
    crop_margin_xyz: Tuple[int, int, int] = (16, 16, 8)  # in voxels after resampling

    # N4 + normalization
    n4_shrink_factor: int = 2
    robust_clip_percentiles: Tuple[float, float] = (1.0, 99.0)

    # for body mask estimation
    body_mask_closing_radius: int = 2

    # image interpolation for continuous images
    image_interp = sitk.sitkBSpline
    label_interp = sitk.sitkNearestNeighbor


# =========================================================
# Utilities
# =========================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_case_and_modality(stem: str) -> Tuple[str, str]:
    """
    Expect filenames like:
        1_t1.mha
        1_t2.mha
        12_SPACE_t1.mha
    Returns:
        case_id, modality
    """
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename stem: {stem}")
    case_id, modality = parts[0], parts[1].lower()
    if modality not in ("t1", "t2"):
        raise ValueError(f"Unexpected modality in filename stem: {stem}")
    return case_id, modality


def is_space_variant(case_id: str) -> bool:
    return case_id.endswith("_SPACE")


def maybe_int_case_id(case_id: str) -> Optional[int]:
    try:
        if case_id.endswith("_SPACE"):
            return int(case_id[:-6])
        return int(case_id)
    except Exception:
        return None


def should_drop_flattened(case_id: str, modality: str, cfg: Cfg) -> bool:
    case_num = maybe_int_case_id(case_id)
    if case_num is None:
        return False
    if modality == "t1" and case_num in set(cfg.flattened_t1):
        return True
    if modality == "t2" and case_num in set(cfg.flattened_t2):
        return True
    return False


def sitk_to_np(img: sitk.Image, dtype=np.float32) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(dtype, copy=False)  # z, y, x


def np_to_sitk(arr: np.ndarray, ref: sitk.Image, is_label: bool = False) -> sitk.Image:
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(ref.GetSpacing())
    out.SetDirection(ref.GetDirection())
    out.SetOrigin(ref.GetOrigin())
    if is_label:
        out = sitk.Cast(out, sitk.sitkUInt16)
    else:
        out = sitk.Cast(out, sitk.sitkFloat32)
    return out


def orient_to_canonical(img: sitk.Image, orient: str = "LPS") -> sitk.Image:
    # Reorders/flips axes while preserving physical positions
    return sitk.DICOMOrient(img, orient)


def infer_thick_axis_xyz(spacing_xyz: Sequence[float]) -> int:
    return int(np.argmax(np.asarray(spacing_xyz, dtype=np.float64)))


def is_direction_close_to_identity(direction: Sequence[float], atol: float = 1e-5) -> bool:
    direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return np.allclose(direction, np.eye(3), atol=atol)


def filter_mask_labels(mask: sitk.Image, keep_labels: Sequence[int]) -> sitk.Image:
    arr = sitk_to_np(mask, dtype=np.int32)
    keep = np.isin(arr, np.asarray(keep_labels, dtype=np.int32))
    arr = np.where(keep, arr, 0).astype(np.uint16)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(mask)
    return sitk.Cast(out, sitk.sitkUInt16)


def mask_has_all_required_labels(mask: sitk.Image, required_labels: Sequence[int]) -> bool:
    arr = sitk_to_np(mask, dtype=np.int32)
    uniq = set(np.unique(arr).tolist())
    return set(required_labels).issubset(uniq)


def safe_percentile(vals: np.ndarray, q: float, default: float = 0.0) -> float:
    vals = np.asarray(vals)
    if vals.size == 0:
        return float(default)
    return float(np.percentile(vals, q))


def compute_body_mask(img: sitk.Image, closing_radius: int = 2) -> sitk.Image:
    """
    Estimate foreground/body mask from image only (no GT leakage).
    """
    arr = sitk_to_np(img, dtype=np.float32)
    nz = arr[np.isfinite(arr)]
    if nz.size == 0:
        mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(img)
        return mask

    p1 = safe_percentile(nz, 1.0, float(arr.min()))
    p99 = safe_percentile(nz, 99.0, float(arr.max()))
    if not np.isfinite(p99 - p1) or (p99 - p1) < 1e-6:
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        scaled = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)

    scaled_img = sitk.GetImageFromArray(scaled)
    scaled_img.CopyInformation(img)

    # Otsu on scaled image
    body = sitk.OtsuThreshold(scaled_img, 0, 1, 128)
    body = sitk.Cast(body, sitk.sitkUInt8)

    if closing_radius > 0:
        body = sitk.BinaryMorphologicalClosing(body, [closing_radius] * 3)
    body = sitk.BinaryFillhole(body)

    cc = sitk.ConnectedComponent(body)
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    body = sitk.Cast(relabel == 1, sitk.sitkUInt8)

    arr_body = sitk_to_np(body, dtype=np.uint8)
    if arr_body.sum() == 0:
        # fallback: everything non-min intensity is foreground
        arr2 = sitk_to_np(img, dtype=np.float32)
        thr = safe_percentile(arr2[np.isfinite(arr2)], 5.0, float(arr2.min()))
        arr_body = (arr2 > thr).astype(np.uint8)
        body = sitk.GetImageFromArray(arr_body)
        body.CopyInformation(img)
        body = sitk.Cast(body, sitk.sitkUInt8)

    return body


def n4_bias_correct(img: sitk.Image, mask: Optional[sitk.Image], shrink_factor: int = 2) -> sitk.Image:
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    if mask is None:
        mask = compute_body_mask(img_f)
    else:
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    if shrink_factor > 1:
        img_small = sitk.Shrink(img_f, [shrink_factor] * 3)
        mask_small = sitk.Shrink(mask, [shrink_factor] * 3)
        corrected_small = corrector.Execute(img_small, mask_small)
        log_bias = corrector.GetLogBiasFieldAsImage(img_f)
        corrected = img_f / sitk.Exp(log_bias)
    else:
        corrected = corrector.Execute(img_f, mask)

    return sitk.Cast(corrected, sitk.sitkFloat32)


def robust_zscore(img: sitk.Image,
                  roi_mask: Optional[sitk.Image],
                  clip_percentiles: Tuple[float, float] = (1.0, 99.0)) -> sitk.Image:
    arr = sitk_to_np(img, dtype=np.float32)

    if roi_mask is None:
        roi = np.isfinite(arr)
    else:
        roi = sitk_to_np(roi_mask, dtype=np.uint8) > 0
        if roi.sum() == 0:
            roi = np.isfinite(arr)

    vals = arr[roi]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        out = sitk.GetImageFromArray(np.zeros_like(arr, dtype=np.float32))
        out.CopyInformation(img)
        return sitk.Cast(out, sitk.sitkFloat32)

    p_low, p_high = clip_percentiles
    lo = safe_percentile(vals, p_low, float(np.min(vals)))
    hi = safe_percentile(vals, p_high, float(np.max(vals)))
    if hi <= lo:
        hi = lo + 1e-6

    arr = np.clip(arr, lo, hi)

    vals = arr[roi]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(np.std(vals))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = 1.0

    arr = (arr - med) / sigma
    arr[~roi] = 0.0

    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.CopyInformation(img)
    return sitk.Cast(out, sitk.sitkFloat32)


def resample_to_spacing(img: sitk.Image,
                        target_spacing_xyz: Sequence[float],
                        is_label: bool = False,
                        default_value: float = 0.0) -> sitk.Image:
    in_spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    in_size = np.asarray(img.GetSize(), dtype=np.int64)
    target_spacing = np.asarray(target_spacing_xyz, dtype=np.float64)

    out_size = np.maximum(np.round(in_size * (in_spacing / target_spacing)).astype(np.int64), 1)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(x) for x in target_spacing))
    resampler.SetSize([int(x) for x in out_size.tolist()])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(Cfg.label_interp if is_label else Cfg.image_interp)
    return resampler.Execute(img)


def bbox_from_binary_mask(mask_img: sitk.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = sitk_to_np(mask_img, dtype=np.uint8) > 0  # z, y, x
    idx = np.argwhere(arr)
    if idx.size == 0:
        return None
    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0)
    # convert to x, y, z
    lo_xyz = np.array([x0, y0, z0], dtype=np.int64)
    hi_xyz = np.array([x1, y1, z1], dtype=np.int64)
    return lo_xyz, hi_xyz


def center_from_binary_mask(mask_img: sitk.Image) -> Optional[np.ndarray]:
    arr = sitk_to_np(mask_img, dtype=np.uint8) > 0
    idx = np.argwhere(arr)
    if idx.size == 0:
        return None
    center_zyx = idx.mean(axis=0)
    center_xyz = np.array([center_zyx[2], center_zyx[1], center_zyx[0]], dtype=np.float64)
    return center_xyz


def crop_or_pad_np(arr_zyx: np.ndarray,
                   center_xyz: Sequence[float],
                   out_size_xyz: Sequence[int],
                   pad_value: float = 0.0) -> np.ndarray:
    """
    Crop/pad numpy array in z,y,x layout around center specified in x,y,z.
    """
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    out_size_xyz = np.asarray(out_size_xyz, dtype=np.int64)

    # Convert to z, y, x
    center_zyx = np.array([center_xyz[2], center_xyz[1], center_xyz[0]], dtype=np.float64)
    out_size_zyx = np.array([out_size_xyz[2], out_size_xyz[1], out_size_xyz[0]], dtype=np.int64)

    in_size_zyx = np.array(arr_zyx.shape, dtype=np.int64)
    start = np.floor(center_zyx - out_size_zyx / 2.0).astype(np.int64)
    end = start + out_size_zyx

    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(end, in_size_zyx)

    dst_lo = np.maximum(-start, 0)
    dst_hi = dst_lo + (src_hi - src_lo)

    out = np.full(tuple(out_size_zyx.tolist()), pad_value, dtype=arr_zyx.dtype)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        arr_zyx[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def write_log_csv(rows: List[Dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
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

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Main pipeline
# =========================================================
def collect_all_pairs(cfg: Cfg) -> List[Tuple[Path, Path, str, str]]:
    img_root = Path(cfg.src_root) / cfg.image_dir
    msk_root = Path(cfg.src_root) / cfg.mask_dir

    pairs: List[Tuple[Path, Path, str, str]] = []
    for img_path in sorted(img_root.glob("*.mha")):
        stem = img_path.stem
        try:
            case_id, modality = parse_case_and_modality(stem)
        except Exception:
            continue

        msk_path = msk_root / f"{stem}.mha"
        if not msk_path.exists():
            continue

        pairs.append((img_path, msk_path, case_id, modality))
    return pairs


def first_pass_filter_and_stats(cfg: Cfg):
    pairs = collect_all_pairs(cfg)
    kept = []
    logs = []

    for img_path, msk_path, case_id, modality in pairs:
        row = {
            "file": img_path.name,
            "case_id": case_id,
            "modality": modality,
            "status": "drop",
            "reason": "",
        }

        if is_space_variant(case_id):
            row["reason"] = "SPACE_variant"
            logs.append(row)
            continue

        if should_drop_flattened(case_id, modality, cfg):
            row["reason"] = "manual_flattened"
            logs.append(row)
            continue

        try:
            img = sitk.ReadImage(str(img_path))
            msk = sitk.ReadImage(str(msk_path))
        except Exception as e:
            row["reason"] = f"read_error: {e}"
            logs.append(row)
            continue

        try:
            # Reorient first
            img = orient_to_canonical(img, cfg.canonical_orient)
            msk = orient_to_canonical(msk, cfg.canonical_orient)

            # Keep only target labels
            msk_keep = filter_mask_labels(msk, cfg.keep_labels)

            # Require all labels to be present
            if not mask_has_all_required_labels(msk_keep, cfg.keep_labels):
                row["reason"] = "missing_required_labels"
                logs.append(row)
                continue

            spacing_xyz = tuple(float(x) for x in img.GetSpacing())
            thick_axis = infer_thick_axis_xyz(spacing_xyz)

            row.update({
                "status": "keep",
                "reason": "ok",
                "spacing_x": spacing_xyz[0],
                "spacing_y": spacing_xyz[1],
                "spacing_z": spacing_xyz[2],
                "thick_axis_xyz": thick_axis,
                "direction_is_identity": is_direction_close_to_identity(img.GetDirection()),
            })
            logs.append(row)

            kept.append({
                "img_path": img_path,
                "msk_path": msk_path,
                "case_id": case_id,
                "modality": modality,
                "spacing_xyz": spacing_xyz,
                "thick_axis": thick_axis,
            })

        except Exception as e:
            row["reason"] = f"precheck_error: {e}"
            logs.append(row)
            continue

    return kept, logs


def estimate_target_spacing(cfg: Cfg, kept_rows: List[Dict]) -> Tuple[float, float, float]:
    if cfg.force_target_spacing is not None:
        return tuple(float(x) for x in cfg.force_target_spacing)

    spacings = np.asarray([r["spacing_xyz"] for r in kept_rows], dtype=np.float64)
    if spacings.size == 0:
        raise RuntimeError("No retained cases available to estimate target spacing.")

    # Median spacing per axis after canonical reorientation
    target = np.median(spacings, axis=0)

    # Use the axis with maximum spacing as thick-axis QA anchor
    thick_axes = np.asarray([r["thick_axis"] for r in kept_rows], dtype=np.int64)
    thick_axis_mode = int(np.bincount(thick_axes, minlength=3).argmax())
    target[thick_axis_mode] = np.median(spacings[:, thick_axis_mode])

    return tuple(float(x) for x in target.tolist())


def second_pass_estimate_crop(cfg: Cfg,
                              kept_rows: List[Dict],
                              target_spacing_xyz: Sequence[float]) -> Tuple[int, int, int]:
    if cfg.force_crop_size_xyz is not None:
        return tuple(int(x) for x in cfg.force_crop_size_xyz)

    bbox_sizes = []

    for r in kept_rows:
        img = sitk.ReadImage(str(r["img_path"]))
        img = orient_to_canonical(img, cfg.canonical_orient)
        img = n4_bias_correct(img, None, cfg.n4_shrink_factor)
        img = resample_to_spacing(img, target_spacing_xyz, is_label=False, default_value=0.0)
        body = compute_body_mask(img, cfg.body_mask_closing_radius)
        bbox = bbox_from_binary_mask(body)
        if bbox is None:
            continue
        lo, hi = bbox
        size_xyz = hi - lo + 1
        bbox_sizes.append(size_xyz)

    if len(bbox_sizes) == 0:
        raise RuntimeError("Failed to estimate crop size from body masks.")

    bbox_sizes = np.asarray(bbox_sizes, dtype=np.float64)
    crop_size = np.percentile(bbox_sizes, cfg.crop_bbox_percentile, axis=0)
    crop_size = np.ceil(crop_size).astype(np.int64)
    crop_size += np.asarray(cfg.crop_margin_xyz, dtype=np.int64)

    # make sizes even
    crop_size = np.where(crop_size % 2 == 1, crop_size + 1, crop_size)
    crop_size = np.maximum(crop_size, 16)
    return tuple(int(x) for x in crop_size.tolist())


def process_and_save_one(cfg: Cfg,
                         row: Dict,
                         target_spacing_xyz: Sequence[float],
                         crop_size_xyz: Sequence[int],
                         out_img_dir: Path,
                         out_msk_dir: Path) -> Dict:
    img = sitk.ReadImage(str(row["img_path"]))
    msk = sitk.ReadImage(str(row["msk_path"]))

    img = orient_to_canonical(img, cfg.canonical_orient)
    msk = orient_to_canonical(msk, cfg.canonical_orient)

    msk = filter_mask_labels(msk, cfg.keep_labels)

    # intensity correction
    body0 = compute_body_mask(img, cfg.body_mask_closing_radius)
    img = n4_bias_correct(img, body0, cfg.n4_shrink_factor)
    body1 = compute_body_mask(img, cfg.body_mask_closing_radius)
    img = robust_zscore(img, body1, cfg.robust_clip_percentiles)

    # physical resampling
    img = resample_to_spacing(img, target_spacing_xyz, is_label=False, default_value=0.0)
    msk = resample_to_spacing(msk, target_spacing_xyz, is_label=True, default_value=0)

    # anatomy crop from image only
    body = compute_body_mask(img, cfg.body_mask_closing_radius)
    center_xyz = center_from_binary_mask(body)
    if center_xyz is None:
        center_xyz = np.array(img.GetSize(), dtype=np.float64) / 2.0

    arr_img = sitk_to_np(img, dtype=np.float32)
    arr_msk = sitk_to_np(msk, dtype=np.uint16)

    arr_img = crop_or_pad_np(arr_img, center_xyz=center_xyz, out_size_xyz=crop_size_xyz, pad_value=0.0)
    arr_msk = crop_or_pad_np(arr_msk, center_xyz=center_xyz, out_size_xyz=crop_size_xyz, pad_value=0)

    out_img = sitk.GetImageFromArray(arr_img.astype(np.float32))
    out_msk = sitk.GetImageFromArray(arr_msk.astype(np.uint16))

    # final geometry: identity direction, zero origin, common spacing
    out_img.SetSpacing(tuple(float(x) for x in target_spacing_xyz))
    out_msk.SetSpacing(tuple(float(x) for x in target_spacing_xyz))
    out_img.SetDirection((1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))
    out_msk.SetDirection((1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))
    out_img.SetOrigin((0.0, 0.0, 0.0))
    out_msk.SetOrigin((0.0, 0.0, 0.0))

    stem = Path(row["img_path"]).stem
    out_img_path = out_img_dir / f"{stem}.mha"
    out_msk_path = out_msk_dir / f"{stem}.mha"

    sitk.WriteImage(sitk.Cast(out_img, sitk.sitkFloat32), str(out_img_path), useCompression=True)
    sitk.WriteImage(sitk.Cast(out_msk, sitk.sitkUInt16), str(out_msk_path), useCompression=True)

    uniq = np.unique(arr_msk).tolist()
    return {
        "file": out_img_path.name,
        "case_id": row["case_id"],
        "modality": row["modality"],
        "saved": True,
        "out_size_x": int(crop_size_xyz[0]),
        "out_size_y": int(crop_size_xyz[1]),
        "out_size_z": int(crop_size_xyz[2]),
        "spacing_x": float(target_spacing_xyz[0]),
        "spacing_y": float(target_spacing_xyz[1]),
        "spacing_z": float(target_spacing_xyz[2]),
        "labels_after_save": uniq,
    }


def main():
    cfg = Cfg()

    src_root = Path(cfg.src_root)
    dst_root = Path(cfg.dst_root)
    out_img_dir = dst_root / "images"
    out_msk_dir = dst_root / "masks"
    log_dir = dst_root / "logs"

    ensure_dir(out_img_dir)
    ensure_dir(out_msk_dir)
    ensure_dir(log_dir)

    kept_rows, first_logs = first_pass_filter_and_stats(cfg)
    write_log_csv(first_logs, log_dir / "first_pass_filter_log.csv")

    if len(kept_rows) == 0:
        raise RuntimeError("No cases left after first-pass filtering.")

    target_spacing_xyz = estimate_target_spacing(cfg, kept_rows)
    crop_size_xyz = second_pass_estimate_crop(cfg, kept_rows, target_spacing_xyz)

    summary_rows = []
    save_logs = []

    for row in kept_rows:
        try:
            save_row = process_and_save_one(
                cfg=cfg,
                row=row,
                target_spacing_xyz=target_spacing_xyz,
                crop_size_xyz=crop_size_xyz,
                out_img_dir=out_img_dir,
                out_msk_dir=out_msk_dir,
            )
            save_logs.append(save_row)

            summary_rows.append({
                "file": Path(row["img_path"]).name,
                "case_id": row["case_id"],
                "modality": row["modality"],
                "kept": True,
                "reason": "saved",
            })
            print(f"[OK] {Path(row['img_path']).name}")
        except Exception as e:
            summary_rows.append({
                "file": Path(row["img_path"]).name,
                "case_id": row["case_id"],
                "modality": row["modality"],
                "kept": False,
                "reason": f"save_error: {e}",
            })
            print(f"[FAIL] {Path(row['img_path']).name}: {e}")

    write_log_csv(save_logs, log_dir / "saved_cases_log.csv")
    write_log_csv(summary_rows, log_dir / "final_summary.csv")

    # Optional: copy raw CSV files for reference only (not filtered)
    for csv_name in ("overview.csv", "radiological_gradings.csv"):
        src_csv = src_root / csv_name
        if src_csv.exists():
            shutil.copy2(src_csv, dst_root / csv_name)

    # Also save preprocessing config
    with (log_dir / "preprocess_meta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_orient", cfg.canonical_orient])
        writer.writerow(["target_spacing_x", target_spacing_xyz[0]])
        writer.writerow(["target_spacing_y", target_spacing_xyz[1]])
        writer.writerow(["target_spacing_z", target_spacing_xyz[2]])
        writer.writerow(["crop_size_x", crop_size_xyz[0]])
        writer.writerow(["crop_size_y", crop_size_xyz[1]])
        writer.writerow(["crop_size_z", crop_size_xyz[2]])
        writer.writerow(["keep_labels", " ".join(map(str, cfg.keep_labels))])

    print("\nDone.")
    print(f"Saved normalized images to: {out_img_dir}")
    print(f"Saved normalized masks  to: {out_msk_dir}")
    print(f"Logs                  to: {log_dir}")


if __name__ == "__main__":
    main()
