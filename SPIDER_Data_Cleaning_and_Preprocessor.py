from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    src_root: str = "SPIDER"
    dst_root: str = "SPIDER_normalized_repaired"

    image_dir: str = "images"
    mask_dir: str = "masks"

    # keep only these labels, and require all of them to exist
    keep_labels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 201, 202, 203, 204, 205, 206, 207)

    # manual flattened cases to remove
    flattened_t1: Tuple[int, ...] = (133, 174, 186, 221, 227, 253)
    flattened_t2: Tuple[int, ...] = (35, 58, 133, 174, 186, 221, 227, 253)

    # canonical orientation after reorientation
    canonical_orient: str = "LPS"

    # target spacing estimated from retained cohort after reorientation
    force_target_spacing: Optional[Tuple[float, float, float]] = None

    # crop size estimated from retained cohort after resampling
    force_crop_size_xyz: Optional[Tuple[int, int, int]] = None

    # crop-size estimation from foreground support bbox after resampling
    crop_bbox_percentile: float = 97.5
    crop_margin_xyz: Tuple[int, int, int] = (20, 20, 10)

    # N4 + normalization
    n4_shrink_factor: int = 2
    robust_clip_percentiles: Tuple[float, float] = (0.5, 99.5)

    # foreground support mask estimation from raw image support, NOT Otsu body mask
    support_mask_closing_radius: int = 2
    support_min_component_ratio: float = 0.01
    support_min_voxels: int = 1024
    background_corner_width: int = 8
    background_tol_abs: float = 1e-6
    background_tol_rel: float = 1e-3

    # if True, only use mask for statistics; do not overwrite voxels outside support
    zero_outside_support_after_norm: bool = False

    # interpolation
    image_interp = sitk.sitkLinear
    label_interp = sitk.sitkNearestNeighbor


# =========================================================
# Utilities
# =========================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_case_and_modality(stem: str) -> Tuple[str, str]:
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


def orient_to_canonical(img: sitk.Image, orient: str = "LPS") -> sitk.Image:
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


def _corner_background_samples(arr: np.ndarray, width: int) -> np.ndarray:
    z, y, x = arr.shape
    wz = max(1, min(width, z))
    wy = max(1, min(width, y))
    wx = max(1, min(width, x))
    corners = [
        arr[:wz, :wy, :wx],
        arr[:wz, :wy, x - wx:],
        arr[:wz, y - wy:, :wx],
        arr[:wz, y - wy:, x - wx:],
        arr[z - wz:, :wy, :wx],
        arr[z - wz:, :wy, x - wx:],
        arr[z - wz:, y - wy:, :wx],
        arr[z - wz:, y - wy:, x - wx:],
    ]
    vals = np.concatenate([c[np.isfinite(c)].ravel() for c in corners])
    return vals


def _keep_large_components(mask: sitk.Image,
                           min_ratio: float,
                           min_voxels: int) -> sitk.Image:
    cc = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        out = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(mask)
        return out

    sizes = {lab: int(stats.GetNumberOfPixels(lab)) for lab in labels}
    largest = max(sizes.values())
    keep = [lab for lab, sz in sizes.items() if sz >= max(min_voxels, int(round(largest * min_ratio)))]

    out = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    out.CopyInformation(mask)
    for lab in keep:
        out = out | sitk.Cast(cc == lab, sitk.sitkUInt8)
    return sitk.Cast(out > 0, sitk.sitkUInt8)


def compute_support_mask(img: sitk.Image, cfg: Cfg) -> sitk.Image:
    """
    Build a conservative non-background support mask from the raw image.
    This is intentionally safer than Otsu+largest-component body masking.
    """
    arr = sitk_to_np(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        out = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(img)
        return out

    bg_vals = _corner_background_samples(arr, cfg.background_corner_width)
    bg_ref = float(np.median(bg_vals)) if bg_vals.size > 0 else float(np.min(arr[finite]))

    p1 = safe_percentile(arr[finite], 1.0, float(np.min(arr[finite])))
    p99 = safe_percentile(arr[finite], 99.0, float(np.max(arr[finite])))
    dyn = max(p99 - p1, 0.0)
    tol = max(cfg.background_tol_abs, cfg.background_tol_rel * max(dyn, 1.0))

    mask_np = finite & (np.abs(arr - bg_ref) > tol)

    # fallback 1: exact nonzero support
    if int(mask_np.sum()) < cfg.support_min_voxels:
        mask_np = finite & (np.abs(arr) > cfg.background_tol_abs)

    # fallback 2: broad percentile threshold above background floor
    if int(mask_np.sum()) < cfg.support_min_voxels:
        thr = safe_percentile(arr[finite], 5.0, bg_ref)
        mask_np = finite & (arr > thr)

    mask = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    mask.CopyInformation(img)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    if cfg.support_mask_closing_radius > 0:
        mask = sitk.BinaryMorphologicalClosing(mask, [cfg.support_mask_closing_radius] * 3)
    mask = sitk.BinaryFillhole(mask)
    mask = _keep_large_components(mask, cfg.support_min_component_ratio, cfg.support_min_voxels)

    arr_mask = sitk_to_np(mask, dtype=np.uint8)
    if int(arr_mask.sum()) == 0:
        arr_mask = finite.astype(np.uint8)
        mask = sitk.GetImageFromArray(arr_mask)
        mask.CopyInformation(img)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    return mask


def n4_bias_correct(img: sitk.Image,
                    mask: Optional[sitk.Image],
                    shrink_factor: int = 2) -> sitk.Image:
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.Cast(compute_support_mask(img_f, Cfg()) > 0, sitk.sitkUInt8)
    else:
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    if shrink_factor > 1:
        img_small = sitk.Shrink(img_f, [shrink_factor] * 3)
        mask_small = sitk.Shrink(mask, [shrink_factor] * 3)
        _ = corrector.Execute(img_small, mask_small)
        log_bias = corrector.GetLogBiasFieldAsImage(img_f)
        corrected = img_f / sitk.Exp(log_bias)
    else:
        corrected = corrector.Execute(img_f, mask)

    return sitk.Cast(corrected, sitk.sitkFloat32)


def robust_zscore(img: sitk.Image,
                  support_mask: Optional[sitk.Image],
                  clip_percentiles: Tuple[float, float] = (0.5, 99.5),
                  zero_outside_support: bool = False) -> sitk.Image:
    arr = sitk_to_np(img, dtype=np.float32)

    if support_mask is None:
        roi = np.isfinite(arr)
    else:
        roi = sitk_to_np(support_mask, dtype=np.uint8) > 0
        roi = roi & np.isfinite(arr)
        if int(roi.sum()) == 0:
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

    out_arr = (arr - med) / sigma
    if zero_outside_support and support_mask is not None:
        out_arr[~roi] = 0.0

    out = sitk.GetImageFromArray(out_arr.astype(np.float32))
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
    lo_xyz = np.array([x0, y0, z0], dtype=np.int64)
    hi_xyz = np.array([x1, y1, z1], dtype=np.int64)
    return lo_xyz, hi_xyz


def center_from_bbox(lo_xyz: Sequence[int], hi_xyz: Sequence[int]) -> np.ndarray:
    lo_xyz = np.asarray(lo_xyz, dtype=np.float64)
    hi_xyz = np.asarray(hi_xyz, dtype=np.float64)
    return 0.5 * (lo_xyz + hi_xyz)


def crop_or_pad_np(arr_zyx: np.ndarray,
                   center_xyz: Sequence[float],
                   out_size_xyz: Sequence[int],
                   pad_value: float = 0.0) -> np.ndarray:
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    out_size_xyz = np.asarray(out_size_xyz, dtype=np.int64)

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


def normalize_volume_to_uint8(arr: np.ndarray,
                              clip_percentiles: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    p_low, p_high = clip_percentiles
    lo = safe_percentile(vals, p_low, float(np.min(vals)))
    hi = safe_percentile(vals, p_high, float(np.max(vals)))
    if hi <= lo:
        hi = lo + 1e-6

    arr01 = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(arr01 * 255.0).astype(np.uint8)


def choose_center_sagittal_indices(mask_zyx: np.ndarray, num_slices: int = 3) -> List[int]:
    if mask_zyx.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got shape={mask_zyx.shape}")

    x_dim = int(mask_zyx.shape[2])
    valid_x = np.where((mask_zyx > 0).any(axis=(0, 1)))[0]

    if valid_x.size > 0:
        center_x = float(np.argwhere(mask_zyx > 0)[:, 2].mean())
        ordered = sorted(valid_x.tolist(), key=lambda x: (abs(x - center_x), x))
    else:
        center_x = 0.5 * (x_dim - 1)
        ordered = []

    chosen: List[int] = []
    for x in ordered:
        if x not in chosen:
            chosen.append(int(x))
        if len(chosen) >= num_slices:
            break

    if len(chosen) < num_slices:
        center_round = int(round(center_x))
        fallback = [center_round, center_round - 1, center_round + 1, center_round - 2, center_round + 2]
        for x in fallback:
            if 0 <= x < x_dim and x not in chosen:
                chosen.append(int(x))
            if len(chosen) >= num_slices:
                break

    chosen = sorted(chosen[:num_slices])
    return chosen


def save_center_sagittal_pngs(stem: str,
                              arr_img_zyx: np.ndarray,
                              arr_msk_zyx: np.ndarray,
                              out_png_dir: Path,
                              clip_percentiles: Tuple[float, float] = (0.5, 99.5)) -> List[int]:
    ensure_dir(out_png_dir)

    vol_u8 = normalize_volume_to_uint8(arr_img_zyx, clip_percentiles=clip_percentiles)
    sagittal_indices = choose_center_sagittal_indices(arr_msk_zyx, num_slices=3)

    for rank, x_idx in enumerate(sagittal_indices, start=1):
        slice_u8 = vol_u8[:, :, x_idx].T  # (y, z), easier to browse visually
        out_path = out_png_dir / f"{stem}_sag_center{rank}_x{x_idx:03d}.png"
        out_img2d = sitk.GetImageFromArray(slice_u8.astype(np.uint8))
        sitk.WriteImage(out_img2d, str(out_path), useCompression=True)

    return sagittal_indices


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
            img = orient_to_canonical(img, cfg.canonical_orient)
            msk = orient_to_canonical(msk, cfg.canonical_orient)
            msk_keep = filter_mask_labels(msk, cfg.keep_labels)

            if not mask_has_all_required_labels(msk_keep, cfg.keep_labels):
                row["reason"] = "missing_required_labels"
                logs.append(row)
                continue

            spacing_xyz = tuple(float(x) for x in img.GetSpacing())
            thick_axis = infer_thick_axis_xyz(spacing_xyz)
            support = compute_support_mask(img, cfg)
            bbox = bbox_from_binary_mask(support)

            row.update({
                "status": "keep",
                "reason": "ok",
                "spacing_x": spacing_xyz[0],
                "spacing_y": spacing_xyz[1],
                "spacing_z": spacing_xyz[2],
                "thick_axis_xyz": thick_axis,
                "direction_is_identity": is_direction_close_to_identity(img.GetDirection()),
                "support_voxels": int(sitk_to_np(support, dtype=np.uint8).sum()),
            })
            if bbox is not None:
                lo, hi = bbox
                row.update({
                    "bbox_x": int(hi[0] - lo[0] + 1),
                    "bbox_y": int(hi[1] - lo[1] + 1),
                    "bbox_z": int(hi[2] - lo[2] + 1),
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

    target = np.median(spacings, axis=0)
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
        support0 = compute_support_mask(img, cfg)
        img = n4_bias_correct(img, support0, cfg.n4_shrink_factor)
        img = robust_zscore(
            img,
            support0,
            cfg.robust_clip_percentiles,
            zero_outside_support=cfg.zero_outside_support_after_norm,
        )

        img = resample_to_spacing(img, target_spacing_xyz, is_label=False, default_value=0.0)
        support = resample_to_spacing(support0, target_spacing_xyz, is_label=True, default_value=0)
        bbox = bbox_from_binary_mask(support)
        if bbox is None:
            continue
        lo, hi = bbox
        size_xyz = hi - lo + 1
        bbox_sizes.append(size_xyz)

    if len(bbox_sizes) == 0:
        raise RuntimeError("Failed to estimate crop size from support masks.")

    bbox_sizes = np.asarray(bbox_sizes, dtype=np.float64)
    crop_size = np.percentile(bbox_sizes, cfg.crop_bbox_percentile, axis=0)
    crop_size = np.ceil(crop_size).astype(np.int64)
    crop_size += np.asarray(cfg.crop_margin_xyz, dtype=np.int64)

    crop_size = np.where(crop_size % 2 == 1, crop_size + 1, crop_size)
    crop_size = np.maximum(crop_size, 16)
    return tuple(int(x) for x in crop_size.tolist())


def process_and_save_one(cfg: Cfg,
                         row: Dict,
                         target_spacing_xyz: Sequence[float],
                         crop_size_xyz: Sequence[int],
                         out_img_dir: Path,
                         out_msk_dir: Path,
                         out_png_dir: Path) -> Dict:
    img = sitk.ReadImage(str(row["img_path"]))
    msk = sitk.ReadImage(str(row["msk_path"]))

    img = orient_to_canonical(img, cfg.canonical_orient)
    msk = orient_to_canonical(msk, cfg.canonical_orient)
    msk = filter_mask_labels(msk, cfg.keep_labels)

    support0 = compute_support_mask(img, cfg)

    # safer intensity correction: use support only for stats/N4, not for destructive masking
    img = n4_bias_correct(img, support0, cfg.n4_shrink_factor)
    img = robust_zscore(
        img,
        support0,
        cfg.robust_clip_percentiles,
        zero_outside_support=cfg.zero_outside_support_after_norm,
    )

    # physical resampling
    img = resample_to_spacing(img, target_spacing_xyz, is_label=False, default_value=0.0)
    msk = resample_to_spacing(msk, target_spacing_xyz, is_label=True, default_value=0)
    support = resample_to_spacing(support0, target_spacing_xyz, is_label=True, default_value=0)

    bbox = bbox_from_binary_mask(support)
    if bbox is None:
        center_xyz = np.array(img.GetSize(), dtype=np.float64) / 2.0
    else:
        lo, hi = bbox
        center_xyz = center_from_bbox(lo, hi)

    arr_img = sitk_to_np(img, dtype=np.float32)
    arr_msk = sitk_to_np(msk, dtype=np.uint16)
    arr_sup = sitk_to_np(support, dtype=np.uint8)

    arr_img = crop_or_pad_np(arr_img, center_xyz=center_xyz, out_size_xyz=crop_size_xyz, pad_value=0.0)
    arr_msk = crop_or_pad_np(arr_msk, center_xyz=center_xyz, out_size_xyz=crop_size_xyz, pad_value=0)
    arr_sup = crop_or_pad_np(arr_sup, center_xyz=center_xyz, out_size_xyz=crop_size_xyz, pad_value=0)

    out_img = sitk.GetImageFromArray(arr_img.astype(np.float32))
    out_msk = sitk.GetImageFromArray(arr_msk.astype(np.uint16))

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

    sagittal_png_indices = save_center_sagittal_pngs(
        stem=stem,
        arr_img_zyx=arr_img,
        arr_msk_zyx=arr_msk,
        out_png_dir=out_png_dir,
        clip_percentiles=cfg.robust_clip_percentiles,
    )

    uniq = np.unique(arr_msk).tolist()
    sup_bbox = bbox_from_binary_mask(sitk.GetImageFromArray(arr_sup.astype(np.uint8)))
    log = {
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
        "support_voxels_after_crop": int(arr_sup.sum()),
        "labels_after_save": uniq,
        "png_sagittal_indices": " ".join(map(str, sagittal_png_indices)),
    }
    if sup_bbox is not None:
        lo, hi = sup_bbox
        log.update({
            "support_bbox_x_after_crop": int(hi[0] - lo[0] + 1),
            "support_bbox_y_after_crop": int(hi[1] - lo[1] + 1),
            "support_bbox_z_after_crop": int(hi[2] - lo[2] + 1),
        })
    return log


def main():
    cfg = Cfg()

    src_root = Path(cfg.src_root)
    dst_root = Path(cfg.dst_root)
    out_img_dir = dst_root / "images"
    out_msk_dir = dst_root / "masks"
    out_png_dir = dst_root / "png_sagittal_center3"
    log_dir = dst_root / "logs"

    ensure_dir(out_img_dir)
    ensure_dir(out_msk_dir)
    ensure_dir(out_png_dir)
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
                out_png_dir=out_png_dir,
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

    for csv_name in ("overview.csv", "radiological_gradings.csv"):
        src_csv = src_root / csv_name
        if src_csv.exists():
            shutil.copy2(src_csv, dst_root / csv_name)

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
        writer.writerow(["clip_percentiles", f"{cfg.robust_clip_percentiles[0]} {cfg.robust_clip_percentiles[1]}"])
        writer.writerow(["zero_outside_support_after_norm", int(cfg.zero_outside_support_after_norm)])

    print("\nDone.")
    print(f"Saved normalized images to: {out_img_dir}")
    print(f"Saved normalized masks  to: {out_msk_dir}")
    print(f"Saved PNG slices       to: {out_png_dir}")
    print(f"Logs                  to: {log_dir}")


if __name__ == "__main__":
    main()
