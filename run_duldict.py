from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 允许从 eval/ 导入
THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from evaluate_run import evaluate_run  # noqa: E402


# ============================================================
# Utility
# ============================================================

def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_log(log_path: Path, text: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def zscore_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    mean = float(img.mean())
    std = float(img.std())
    return (img - mean) / (std + eps)


def minmax_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(img.min())
    mx = float(img.max())
    return (img - mn) / (mx - mn + eps)


def clip_percentile(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = np.percentile(img, lo)
    b = np.percentile(img, hi)
    return np.clip(img, a, b)


def simple_gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Minimal pure-numpy separable Gaussian blur to avoid extra dependencies.
    If sigma <= 0, returns img unchanged.
    """
    if sigma <= 0:
        return img

    radius = max(1, int(round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    # horizontal
    pad = radius
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[1]):
        out[:, i] = np.sum(tmp[:, i:i + 2 * radius + 1] * kernel[None, :], axis=1)

    # vertical
    tmp = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
    out2 = np.zeros_like(out, dtype=np.float32)
    for i in range(img.shape[0]):
        out2[i, :] = np.sum(tmp[i:i + 2 * radius + 1, :] * kernel[:, None], axis=0)

    return out2


def remove_small_components(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    Very simple connected-component filtering using BFS, 4-connectivity.
    """
    if min_size <= 0:
        return mask.astype(np.uint8)

    h, w = mask.shape
    mask = mask.astype(np.uint8)
    visited = np.zeros_like(mask, dtype=np.uint8)
    out = np.zeros_like(mask, dtype=np.uint8)

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y < h - 1:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x < w - 1:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue

            queue = [(y, x)]
            visited[y, x] = 1
            comp = [(y, x)]

            head = 0
            while head < len(queue):
                cy, cx = queue[head]
                head += 1
                for ny, nx in neighbors(cy, cx):
                    if mask[ny, nx] == 1 and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        queue.append((ny, nx))
                        comp.append((ny, nx))

            if len(comp) >= min_size:
                for cy, cx in comp:
                    out[cy, cx] = 1

    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    mask = mask.astype(np.uint8)
    visited = np.zeros_like(mask, dtype=np.uint8)
    comps: List[List[Tuple[int, int]]] = []

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y < h - 1:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x < w - 1:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue

            queue = [(y, x)]
            visited[y, x] = 1
            comp = [(y, x)]

            head = 0
            while head < len(queue):
                cy, cx = queue[head]
                head += 1
                for ny, nx in neighbors(cy, cx):
                    if mask[ny, nx] == 1 and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        queue.append((ny, nx))
                        comp.append((ny, nx))

            comps.append(comp)

    if not comps:
        return np.zeros_like(mask, dtype=np.uint8)

    largest = max(comps, key=len)
    out = np.zeros_like(mask, dtype=np.uint8)
    for y, x in largest:
        out[y, x] = 1
    return out


# ============================================================
# Data I/O
# ============================================================

def load_case_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one case from npz.

    Expected keys (priority order):
      image: "image", "img", "arr_0"
      mask : "mask", "gt", "label", "arr_1"

    Modify this function if your SPIDER npz format is different.
    """
    data = np.load(npz_path)

    image = None
    mask = None

    for k in ["image", "img", "arr_0"]:
        if k in data:
            image = data[k]
            break

    for k in ["mask", "gt", "label", "arr_1"]:
        if k in data:
            mask = data[k]
            break

    if image is None:
        raise KeyError(f"No image array found in {npz_path}")
    if mask is None:
        raise KeyError(f"No mask array found in {npz_path}")

    image = np.asarray(image).astype(np.float32)
    mask = np.asarray(mask)

    # squeeze common singleton dims
    image = np.squeeze(image)
    mask = np.squeeze(mask)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape} from {npz_path}")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape} from {npz_path}")

    mask = (mask > 0).astype(np.uint8)
    return image, mask


def resolve_case_path(case_id: str, images_dir: Path, masks_dir: Path, split_mode: str = "combined") -> Tuple[Path, Optional[Path]]:
    """
    split_mode:
      - "combined": images_dir/{case_id}.npz contains both image and mask
      - "separate": images_dir/{case_id}.npz and masks_dir/{case_id}.npz
    """
    if split_mode == "combined":
        return images_dir / f"{case_id}.npz", None
    if split_mode == "separate":
        return images_dir / f"{case_id}.npz", masks_dir / f"{case_id}.npz"
    raise ValueError(f"Unknown split_mode: {split_mode}")


def load_case(case_id: str, images_dir: Path, masks_dir: Path, split_mode: str = "combined") -> Tuple[np.ndarray, np.ndarray]:
    img_path, mask_path = resolve_case_path(case_id, images_dir, masks_dir, split_mode)

    if split_mode == "combined":
        return load_case_npz(img_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    img_data = np.load(img_path)
    mask_data = np.load(mask_path)

    image = None
    for k in ["image", "img", "arr_0"]:
        if k in img_data:
            image = img_data[k]
            break
    if image is None:
        raise KeyError(f"No image found in {img_path}")

    mask = None
    for k in ["mask", "gt", "label", "arr_0"]:
        if k in mask_data:
            mask = mask_data[k]
            break
    if mask is None:
        raise KeyError(f"No mask found in {mask_path}")

    image = np.squeeze(np.asarray(image).astype(np.float32))
    mask = (np.squeeze(np.asarray(mask)) > 0).astype(np.uint8)
    return image, mask


# ============================================================
# Patch extraction and features
# ============================================================

@dataclass
class PatchInfo:
    y: int
    x: int
    fg_ratio: float


def iter_patch_coords(h: int, w: int, patch_size: int, stride: int):
    ys = list(range(0, max(1, h - patch_size + 1), stride))
    xs = list(range(0, max(1, w - patch_size + 1), stride))

    if len(ys) == 0 or ys[-1] != h - patch_size:
        ys.append(max(0, h - patch_size))
    if len(xs) == 0 or xs[-1] != w - patch_size:
        xs.append(max(0, w - patch_size))

    for y in ys:
        for x in xs:
            yield y, x


def patch_feature_raw(img_patch: np.ndarray) -> np.ndarray:
    return img_patch.astype(np.float32).reshape(-1)


def patch_feature_gradient(img_patch: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(img_patch, dtype=np.float32)
    gx = np.zeros_like(img_patch, dtype=np.float32)
    gy[1:-1, :] = (img_patch[2:, :] - img_patch[:-2, :]) * 0.5
    gx[:, 1:-1] = (img_patch[:, 2:] - img_patch[:, :-2]) * 0.5
    grad = np.sqrt(gx * gx + gy * gy)
    return grad.reshape(-1).astype(np.float32)


def extract_patch_features(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    stride: int,
    min_fg_ratio_for_fg_patch: float,
    max_fg_ratio_for_bg_patch: float,
    feature_type: str,
    use_gradient: bool,
) -> Tuple[np.ndarray, np.ndarray, List[PatchInfo]]:
    """
    Returns:
      fg_features: [Nf, D]
      bg_features: [Nb, D]
      patch_infos: all patch coords + fg_ratio (for debug)
    """
    h, w = img.shape
    fg_feats: List[np.ndarray] = []
    bg_feats: List[np.ndarray] = []
    infos: List[PatchInfo] = []

    for y, x in iter_patch_coords(h, w, patch_size, stride):
        img_patch = img[y:y + patch_size, x:x + patch_size]
        m_patch = mask[y:y + patch_size, x:x + patch_size]

        fg_ratio = float(m_patch.mean())
        infos.append(PatchInfo(y=y, x=x, fg_ratio=fg_ratio))

        feat_parts = [patch_feature_raw(img_patch)]
        if use_gradient:
            feat_parts.append(patch_feature_gradient(img_patch))
        feat = np.concatenate(feat_parts, axis=0)

        if fg_ratio >= min_fg_ratio_for_fg_patch:
            fg_feats.append(feat)
        elif fg_ratio <= max_fg_ratio_for_bg_patch:
            bg_feats.append(feat)

    fg_arr = np.stack(fg_feats, axis=0) if len(fg_feats) > 0 else np.zeros((0, 0), dtype=np.float32)
    bg_arr = np.stack(bg_feats, axis=0) if len(bg_feats) > 0 else np.zeros((0, 0), dtype=np.float32)
    return fg_arr, bg_arr, infos


def extract_query_patch_features(
    img: np.ndarray,
    patch_size: int,
    stride: int,
    use_gradient: bool,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    h, w = img.shape
    feats: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []

    for y, x in iter_patch_coords(h, w, patch_size, stride):
        img_patch = img[y:y + patch_size, x:x + patch_size]
        feat_parts = [patch_feature_raw(img_patch)]
        if use_gradient:
            feat_parts.append(patch_feature_gradient(img_patch))
        feat = np.concatenate(feat_parts, axis=0)
        feats.append(feat)
        coords.append((y, x))

    arr = np.stack(feats, axis=0).astype(np.float32)
    return arr, coords


# ============================================================
# PCA / subspace dictionary
# ============================================================

@dataclass
class PCATransform:
    mean: np.ndarray
    components: np.ndarray  # [K, D]


def fit_pca(X: np.ndarray, out_dim: int) -> PCATransform:
    if X.ndim != 2:
        raise ValueError(f"fit_pca expects 2D array, got {X.shape}")

    mean = X.mean(axis=0, keepdims=False)
    Xc = X - mean[None, :]
    # economical SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(out_dim, Vt.shape[0])
    comps = Vt[:k].astype(np.float32)
    return PCATransform(mean=mean.astype(np.float32), components=comps)


def apply_pca(X: np.ndarray, pca: PCATransform) -> np.ndarray:
    Xc = X - pca.mean[None, :]
    return Xc @ pca.components.T


@dataclass
class SubspaceModel:
    mean: np.ndarray       # [D]
    basis: np.ndarray      # [D, r]


def fit_subspace(X: np.ndarray, rank: int) -> SubspaceModel:
    if X.ndim != 2:
        raise ValueError(f"fit_subspace expects 2D array, got {X.shape}")
    if len(X) == 0:
        raise ValueError("fit_subspace received empty data.")

    mean = X.mean(axis=0, keepdims=False)
    Xc = X - mean[None, :]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(rank, Vt.shape[0])
    basis = Vt[:r].T.astype(np.float32)  # [D, r]
    return SubspaceModel(mean=mean.astype(np.float32), basis=basis)


def subspace_residual(X: np.ndarray, model: SubspaceModel) -> np.ndarray:
    """
    X: [N, D]
    returns residual norm per sample: [N]
    """
    Xc = X - model.mean[None, :]
    proj = (Xc @ model.basis) @ model.basis.T
    res = Xc - proj
    return np.sqrt(np.sum(res * res, axis=1) + 1e-8)


# ============================================================
# Inference
# ============================================================

def patch_scores_to_map(
    scores: np.ndarray,
    coords: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    h, w = image_shape
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for s, (y, x) in zip(scores, coords):
        score_map[y:y + patch_size, x:x + patch_size] += float(s)
        count_map[y:y + patch_size, x:x + patch_size] += 1.0

    score_map = score_map / np.maximum(count_map, 1e-8)
    return score_map


def threshold_score_map(score_map: np.ndarray, threshold: float) -> np.ndarray:
    return (score_map >= threshold).astype(np.uint8)


def postprocess_mask(
    mask: np.ndarray,
    score_map: Optional[np.ndarray],
    gaussian_sigma: float,
    remove_small_objects_size: int,
    largest_component_only: bool,
    threshold: float,
) -> np.ndarray:
    if score_map is not None and gaussian_sigma > 0:
        smoothed = simple_gaussian_blur(score_map.astype(np.float32), sigma=gaussian_sigma)
        mask = (smoothed >= threshold).astype(np.uint8)

    if remove_small_objects_size > 0:
        mask = remove_small_components(mask, min_size=remove_small_objects_size)

    if largest_component_only:
        mask = keep_largest_component(mask)

    return mask.astype(np.uint8)


# ============================================================
# Main pipeline
# ============================================================

def preprocess_image(img: np.ndarray, preprocess_cfg: dict) -> np.ndarray:
    out = img.astype(np.float32)

    clip_cfg = preprocess_cfg.get("clip_percentile", None)
    if clip_cfg is not None and len(clip_cfg) == 2:
        out = clip_percentile(out, float(clip_cfg[0]), float(clip_cfg[1]))

    norm = preprocess_cfg.get("normalize", "zscore")
    if norm == "zscore":
        out = zscore_normalize(out)
    elif norm == "minmax":
        out = minmax_normalize(out)
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {norm}")

    return out


def build_dictionary_from_support(
    support_ids: List[str],
    images_dir: Path,
    masks_dir: Path,
    split_mode: str,
    config: dict,
    log_path: Path,
) -> Tuple[Optional[PCATransform], SubspaceModel, SubspaceModel, dict]:
    patch_cfg = config["patch"]
    feat_cfg = config["feature"]
    dict_cfg = config["dictionary"]

    all_fg = []
    all_bg = []

    support_stats = {
        "num_support_cases": len(support_ids),
        "fg_patch_count": 0,
        "bg_patch_count": 0,
    }

    for case_id in support_ids:
        img, mask = load_case(case_id, images_dir, masks_dir, split_mode=split_mode)
        img = preprocess_image(img, config["preprocess"])

        fg_feats, bg_feats, _ = extract_patch_features(
            img=img,
            mask=mask,
            patch_size=int(patch_cfg["patch_size"]),
            stride=int(patch_cfg["stride"]),
            min_fg_ratio_for_fg_patch=float(patch_cfg["min_fg_ratio_for_fg_patch"]),
            max_fg_ratio_for_bg_patch=float(patch_cfg["max_fg_ratio_for_bg_patch"]),
            feature_type=feat_cfg["type"],
            use_gradient=bool(feat_cfg.get("use_gradient", False)),
        )

        if fg_feats.size > 0:
            all_fg.append(fg_feats)
            support_stats["fg_patch_count"] += int(fg_feats.shape[0])
        if bg_feats.size > 0:
            all_bg.append(bg_feats)
            support_stats["bg_patch_count"] += int(bg_feats.shape[0])

        write_log(
            log_path,
            f"[SUPPORT] {case_id}: fg_patches={0 if fg_feats.size == 0 else fg_feats.shape[0]}, "
            f"bg_patches={0 if bg_feats.size == 0 else bg_feats.shape[0]}"
        )

    if len(all_fg) == 0:
        raise RuntimeError("No foreground patches extracted from support set.")
    if len(all_bg) == 0:
        raise RuntimeError("No background patches extracted from support set.")

    X_fg = np.concatenate(all_fg, axis=0).astype(np.float32)
    X_bg = np.concatenate(all_bg, axis=0).astype(np.float32)

    # optional PCA before subspace learning
    pca = None
    if feat_cfg["type"] == "raw_pca":
        pca_dim = int(feat_cfg["pca_dim"])
        X_all = np.concatenate([X_fg, X_bg], axis=0)
        pca = fit_pca(X_all, out_dim=pca_dim)
        X_fg = apply_pca(X_fg, pca)
        X_bg = apply_pca(X_bg, pca)
        write_log(log_path, f"[PCA] fitted pca_dim={pca_dim}, original_dim={X_all.shape[1]}")

    fg_model = fit_subspace(X_fg, rank=int(dict_cfg["fg_rank"]))
    bg_model = fit_subspace(X_bg, rank=int(dict_cfg["bg_rank"]))

    write_log(log_path, f"[DICT] FG subspace rank={fg_model.basis.shape[1]}, dim={fg_model.basis.shape[0]}")
    write_log(log_path, f"[DICT] BG subspace rank={bg_model.basis.shape[1]}, dim={bg_model.basis.shape[0]}")

    return pca, fg_model, bg_model, support_stats


def infer_case(
    case_id: str,
    images_dir: Path,
    masks_dir: Path,
    split_mode: str,
    config: dict,
    pca: Optional[PCATransform],
    fg_model: SubspaceModel,
    bg_model: SubspaceModel,
    pred_dir: Path,
    visual_dir: Optional[Path],
    log_path: Path,
) -> None:
    patch_cfg = config["patch"]
    feat_cfg = config["feature"]
    infer_cfg = config["inference"]
    post_cfg = config["postprocess"]

    img, gt = load_case(case_id, images_dir, masks_dir, split_mode=split_mode)
    img = preprocess_image(img, config["preprocess"])

    X_q, coords = extract_query_patch_features(
        img=img,
        patch_size=int(patch_cfg["patch_size"]),
        stride=int(patch_cfg["stride"]),
        use_gradient=bool(feat_cfg.get("use_gradient", False)),
    )

    if pca is not None:
        X_q = apply_pca(X_q, pca)

    ef = subspace_residual(X_q, fg_model)
    eb = subspace_residual(X_q, bg_model)

    score_formula = infer_cfg.get("score_formula", "eb-ef")
    if score_formula == "eb-ef":
        scores = eb - ef
    elif score_formula == "ef-eb":
        scores = ef - eb
    else:
        raise ValueError(f"Unknown score_formula: {score_formula}")

    score_map = patch_scores_to_map(
        scores=scores,
        coords=coords,
        image_shape=img.shape,
        patch_size=int(patch_cfg["patch_size"]),
    )

    threshold = float(infer_cfg["threshold"])
    pred_mask = threshold_score_map(score_map, threshold=threshold)

    pred_mask = postprocess_mask(
        mask=pred_mask,
        score_map=score_map,
        gaussian_sigma=float(post_cfg.get("gaussian_sigma", 0.0)) if post_cfg.get("enabled", True) else 0.0,
        remove_small_objects_size=int(post_cfg.get("remove_small_objects", 0)) if post_cfg.get("enabled", True) else 0,
        largest_component_only=bool(post_cfg.get("largest_component_only", False)) if post_cfg.get("enabled", True) else False,
        threshold=threshold,
    )

    np.save(pred_dir / f"{case_id}.npy", pred_mask.astype(np.uint8))

    if visual_dir is not None:
        # Save lightweight npz visualization bundle
        np.savez_compressed(
            visual_dir / f"{case_id}.npz",
            image=img.astype(np.float32),
            gt=gt.astype(np.uint8),
            score_map=score_map.astype(np.float32),
            pred=pred_mask.astype(np.uint8),
        )

    write_log(
        log_path,
        f"[QUERY] {case_id}: num_patches={len(coords)}, "
        f"score_min={scores.min():.4f}, score_max={scores.max():.4f}, score_mean={scores.mean():.4f}"
    )


# ============================================================
# Runner
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    config = load_json(args.config)
    split = load_json(config["split_file"])

    exp_name = config["exp_name"]
    runs_root = Path(config["paths"]["runs_root"])
    run_dir = ensure_dir(runs_root / exp_name)
    pred_dir = ensure_dir(run_dir / "predictions")
    visual_dir = ensure_dir(run_dir / "visuals") if config["evaluation"].get("save_visuals", True) else None
    log_path = run_dir / "log.txt"

    # Save copies for reproducibility
    save_json(config, run_dir / "config.json")
    save_json(split, run_dir / "split.json")

    write_log(log_path, f"Experiment: {exp_name}")
    write_log(log_path, f"Timestamp: {datetime.now().isoformat()}")
    write_log(log_path, f"Method: {config['method']}")
    write_log(log_path, f"Split: {split['split_name']}")
    write_log(log_path, f"Support IDs: {split['support_ids']}")
    write_log(log_path, f"Query IDs: {split['query_ids']}")

    images_dir = Path(config["paths"]["images_dir"])
    masks_dir = Path(config["paths"]["masks_dir"])
    split_mode = config["paths"].get("case_storage_mode", "combined")  # combined / separate

    # Build support dictionaries
    pca, fg_model, bg_model, support_stats = build_dictionary_from_support(
        support_ids=split["support_ids"],
        images_dir=images_dir,
        masks_dir=masks_dir,
        split_mode=split_mode,
        config=config,
        log_path=log_path,
    )

    write_log(log_path, f"[SUPPORT_STATS] {support_stats}")

    # Run query inference
    for case_id in split["query_ids"]:
        infer_case(
            case_id=case_id,
            images_dir=images_dir,
            masks_dir=masks_dir,
            split_mode=split_mode,
            config=config,
            pca=pca,
            fg_model=fg_model,
            bg_model=bg_model,
            pred_dir=pred_dir,
            visual_dir=visual_dir,
            log_path=log_path,
        )

    # Unified evaluation
    summary = evaluate_run(
        run_dir=str(run_dir),
        gt_dir=str(masks_dir),
        threshold=0.5,  # predictions are already binary
    )

    # Merge metadata into summary
    summary["exp_name"] = exp_name
    summary["method"] = config["method"]
    summary["split_name"] = split["split_name"]
    summary["seed"] = split.get("seed", -1)
    summary["support_count"] = len(split["support_ids"])
    summary["query_count"] = len(split["query_ids"])
    summary["fg_patch_count"] = support_stats["fg_patch_count"]
    summary["bg_patch_count"] = support_stats["bg_patch_count"]
    summary["feature_type"] = config["feature"]["type"]
    summary["patch_size"] = config["patch"]["patch_size"]
    summary["stride"] = config["patch"]["stride"]
    summary["fg_rank"] = config["dictionary"]["fg_rank"]
    summary["bg_rank"] = config["dictionary"]["bg_rank"]

    save_json(summary, run_dir / "metrics.json")

    write_log(log_path, "[DONE] Experiment finished successfully.")
    write_log(log_path, json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()