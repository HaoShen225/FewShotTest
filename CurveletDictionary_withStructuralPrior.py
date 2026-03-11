from __future__ import annotations

import os
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import pywt

try:
    from curvelets.numpy import UDCT
except ImportError as e:
    raise SystemExit(
        "Missing package 'curvelets'. Please install with: pip install curvelets"
    ) from e

# 小波曲波双字典+亮暗块结构场先验
# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    root: str = "SPIDER_224_2d_png"
    img_dir: str = "images_t1"
    mask_dir: str = "masks_t1"
    out_dir: str = "CurveletWavelet_Method/exp_centered_aug_piecewise"

    seed: int = 0

    # dataset split
    n_support_patients: int = 5

    # image
    image_size: int = 224

    # curvelet
    num_scales: int = 4
    wedges_per_direction: int = 6   # must be divisible by 3

    # wavelet
    wavelet: str = "db2"
    wavelet_level: int = 3

    # support foreground augmentation for curvelet dictionary
    n_aug_per_support: int = 6
    max_translate_x: int = 10
    max_translate_y: int = 6
    max_rotate_deg: float = 10.0

    # dictionary selection
    fg_sample_top_ratio: float = 0.010
    bg_sample_top_ratio: float = 0.015
    fg_topk_atoms: int = 12000
    bg_topk_atoms: int = 18000

    # score / segmentation
    alpha_bg: float = 0.80
    score_scale: float = 7.0
    prob_thr: float = 0.55

    # simple refine
    use_postprocess: bool = True

    # save
    save_vis_n: int = 20
    n_vis_atoms: int = 16

    # -------------------------
    # piecewise candidate config
    # -------------------------
    w_score: float = 0.75
    w_img: float = 0.25
    score_quantiles: Tuple[float, ...] = (0.78, 0.84, 0.88, 0.92)
    score_sigma: float = 1.0

    cand_area_min: int = 80
    cand_area_max: int = 4200
    cand_aspect_min: float = 0.45   # width / height
    cand_aspect_max: float = 2.40   # width / height
    cand_solidity_min: float = 0.35
    cand_pca_ratio_max: float = 4.80
    cand_chain_dist_max: float = 48.0
    cand_center_dedup: float = 12.0
    cand_iou_dedup: float = 0.55
    cand_max_keep: int = 40

    edge_max_dy: float = 70.0
    edge_min_dy: float = 8.0
    edge_max_dx: float = 42.0
    edge_max_scale_ratio: float = 2.20
    chain_min_len: int = 3
    chain_gap_penalty: float = 0.65
    chain_dx_penalty: float = 0.06
    chain_dy_penalty: float = 0.015
    chain_scale_penalty: float = 0.45
    chain_angle_penalty: float = 0.20

    sdf_margin_ratio: float = 0.35
    sdf_margin_min: int = 6
    sdf_sigma: float = 1.25
    sdf_tau_soft: float = 5.0
    sdf_min_fill_hole: int = 12


cfg = Cfg()


# =========================================================
# Data structures
# =========================================================
@dataclass
class Candidate:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]   # y0, x0, y1, x1 (y1/x1 exclusive)
    center: Tuple[float, float]       # cy, cx
    area: int
    mean_score: float
    mean_img: float
    fused_score: float
    solidity: float
    aspect_ratio: float
    pca_ratio: float
    angle_deg: float
    chain_dist: float


# =========================================================
# Basic utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_gray_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_mask(path: str, mask01: np.ndarray):
    Image.fromarray((mask01 * 255).astype(np.uint8)).save(path)


def save_overlay(path: str, img01: np.ndarray, pred01: np.ndarray, gt01: np.ndarray):
    base = np.stack([(img01 * 255).astype(np.uint8)] * 3, axis=-1)
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    vis = base.copy()
    vis[gt, 1] = 255
    vis[pred, 0] = 255
    Image.fromarray(vis).save(path)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


def robust_norm(x: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(x, p1))
    hi = float(np.percentile(x, p99))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    dice = 2.0 * inter / (pred.sum() + gt.sum() + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def parse_patient_id(filename_stem: str) -> str:
    return filename_stem.split("_")[0]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, x0, y1, x1


def crop_box(y0: int, x0: int, y1: int, x1: int, H: int, W: int) -> Tuple[int, int, int, int]:
    return max(0, y0), max(0, x0), min(H, y1), min(W, x1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    return inter / (union + 1e-8)


def fill_small_holes(mask: np.ndarray, area_thr: int) -> np.ndarray:
    inv = ~mask
    lbl, n = ndi.label(inv)
    if n == 0:
        return mask
    sizes = ndi.sum(inv, lbl, index=np.arange(1, n + 1))
    out = mask.copy()
    H, W = mask.shape
    for lab, size in enumerate(sizes, start=1):
        if size > area_thr:
            continue
        ys, xs = np.where(lbl == lab)
        if len(xs) == 0:
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
            continue
        out[ys, xs] = True
    return out


# =========================================================
# Dataset indexing
# =========================================================
def build_case_list(root: str, img_dir: str, mask_dir: str) -> List[Dict]:
    img_root = Path(root) / img_dir
    mask_root = Path(root) / mask_dir

    items = []
    for img_path in sorted(img_root.glob("*.png")):
        stem = img_path.stem
        mask_path = mask_root / f"{stem}.png"
        if not mask_path.exists():
            continue
        pid = parse_patient_id(stem)
        items.append(
            {
                "pid": pid,
                "name": stem,
                "img_path": str(img_path),
                "mask_path": str(mask_path),
            }
        )
    return items


def split_support_query(items: List[Dict], n_support_patients: int, seed: int):
    patient_ids = sorted(list({x["pid"] for x in items}))
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    support_pids = sorted(patient_ids[:n_support_patients])
    query_pids = sorted(patient_ids[n_support_patients:])

    support_items = [x for x in items if x["pid"] in support_pids]
    query_items = [x for x in items if x["pid"] in query_pids]

    return support_pids, query_pids, support_items, query_items


# =========================================================
# Horizontal centering by mask centroid
# =========================================================
def horizontal_shift_no_wrap(arr: np.ndarray, shift_x: int) -> np.ndarray:
    """
    shift_x > 0: move right
    shift_x < 0: move left
    zero-fill, no wrap
    """
    h, w = arr.shape
    out = np.zeros_like(arr)

    if shift_x == 0:
        out[:] = arr
        return out

    if shift_x > 0:
        out[:, shift_x:] = arr[:, :w - shift_x]
    else:
        sx = -shift_x
        out[:, :w - sx] = arr[:, sx:]
    return out


def center_horizontally_by_mask(img01: np.ndarray, mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0:
        return img01.copy(), mask01.copy(), 0

    cx = float(xs.mean())
    target_x = (img01.shape[1] - 1) / 2.0
    shift_x = int(round(target_x - cx))

    img_c = horizontal_shift_no_wrap(img01, shift_x)
    mask_c = horizontal_shift_no_wrap(mask01, shift_x)
    mask_c = (mask_c > 0.5).astype(np.float32)
    return img_c.astype(np.float32), mask_c, shift_x


# =========================================================
# Support augmentation for foreground
# =========================================================
def random_transform_fg(
    img01: np.ndarray,
    mask01: np.ndarray,
    max_tx: int,
    max_ty: int,
    max_rot_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    tx = random.randint(-max_tx, max_tx)
    ty = random.randint(-max_ty, max_ty)
    ang = random.uniform(-max_rot_deg, max_rot_deg)

    img_t = ndi.shift(img01, shift=(ty, tx), order=1, mode="constant", cval=0.0)
    msk_t = ndi.shift(mask01, shift=(ty, tx), order=0, mode="constant", cval=0.0)

    img_r = ndi.rotate(img_t, angle=ang, reshape=False, order=1, mode="constant", cval=0.0)
    msk_r = ndi.rotate(msk_t, angle=ang, reshape=False, order=0, mode="constant", cval=0.0)

    msk_r = (msk_r > 0.5).astype(np.float32)
    img_r = img_r.astype(np.float32) * msk_r
    return img_r.astype(np.float32), msk_r


# =========================================================
# Load dataset tensors / arrays
# =========================================================
def load_items_centered(items: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
    imgs = []
    masks = []
    names = []
    shifts = []

    for item in items:
        img = read_gray_png(item["img_path"])
        msk = read_gray_png(item["mask_path"])
        msk = (msk > 0.5).astype(np.float32)

        img_c, msk_c, sx = center_horizontally_by_mask(img, msk)

        imgs.append(img_c)
        masks.append(msk_c)
        names.append(item["name"])
        shifts.append(sx)

    return imgs, masks, names, shifts


# =========================================================
# Curvelet helpers
# =========================================================
def flatten_curvelet_coeffs(coeffs) -> np.ndarray:
    parts = []

    def rec(node):
        if isinstance(node, np.ndarray):
            parts.append(node.reshape(-1))
        elif isinstance(node, (list, tuple)):
            for x in node:
                rec(x)
        else:
            raise TypeError(f"Unsupported coeff node type: {type(node)}")

    rec(coeffs)
    if len(parts) == 0:
        return np.zeros((0,), dtype=np.complex64)
    return np.concatenate(parts).astype(np.complex64, copy=False)


def mask_curvelet_coeffs(coeffs, select_mask: np.ndarray):
    cursor = 0

    def rec(node):
        nonlocal cursor
        if isinstance(node, np.ndarray):
            n = node.size
            keep = select_mask[cursor:cursor + n].reshape(node.shape)
            out = np.zeros_like(node)
            out[keep] = node[keep]
            cursor += n
            return out
        items = [rec(x) for x in node]
        return tuple(items) if isinstance(node, tuple) else items

    return rec(coeffs)


def single_curvelet_atom_like(template_coeffs, atom_index: int, value: complex = 1.0 + 0.0j):
    cursor = 0

    def rec(node):
        nonlocal cursor
        if isinstance(node, np.ndarray):
            n = node.size
            out = np.zeros_like(node)
            if cursor <= atom_index < cursor + n:
                flat = out.reshape(-1)
                flat[atom_index - cursor] = value
            cursor += n
            return out
        items = [rec(x) for x in node]
        return tuple(items) if isinstance(node, tuple) else items

    return rec(template_coeffs)


# =========================================================
# Wavelet helpers (stationary wavelet transform)
# =========================================================
def swt2_detail_coeffs(img01: np.ndarray, wavelet: str, level: int):
    return pywt.swt2(img01, wavelet=wavelet, level=level, start_level=0, norm=True)


def flatten_swt_detail_coeffs(coeffs) -> np.ndarray:
    parts = []
    for cA, (cH, cV, cD) in coeffs:
        parts.extend([cH.reshape(-1), cV.reshape(-1), cD.reshape(-1)])
    return np.concatenate(parts).astype(np.float32, copy=False)


def mask_swt_detail_coeffs(coeffs, select_mask: np.ndarray):
    out = []
    cursor = 0
    for cA, (cH, cV, cD) in coeffs:
        zA = np.zeros_like(cA)
        bands = []
        for band in (cH, cV, cD):
            n = band.size
            keep = select_mask[cursor:cursor + n].reshape(band.shape)
            arr = np.zeros_like(band)
            arr[keep] = band[keep]
            bands.append(arr)
            cursor += n
        out.append((zA, tuple(bands)))
    return out


def topk_mask_from_vector(v_abs: np.ndarray, topk: int) -> np.ndarray:
    mask = np.zeros_like(v_abs, dtype=bool)
    topk = max(1, min(topk, v_abs.size))
    ids = np.argpartition(v_abs, -topk)[-topk:]
    mask[ids] = True
    return mask


# =========================================================
# Dictionary construction
# =========================================================
def build_foreground_curvelet_dictionary(cfg: Cfg, support_imgs, support_masks, curvelet_op: UDCT):
    zero_coeffs = curvelet_op.forward(np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32))
    n_atoms = flatten_curvelet_coeffs(zero_coeffs).size

    freq = np.zeros(n_atoms, dtype=np.float32)
    mean_abs = np.zeros(n_atoms, dtype=np.float64)
    aug_examples = []

    for img, msk in zip(support_imgs, support_masks):
        fg = img * msk

        coeffs = curvelet_op.forward(fg.astype(np.float32))
        v = np.abs(flatten_curvelet_coeffs(coeffs)).astype(np.float64)
        mean_abs += v
        k = max(1, int(cfg.fg_sample_top_ratio * v.size))
        active = topk_mask_from_vector(v, k)
        freq += active.astype(np.float32)

        for _ in range(cfg.n_aug_per_support):
            aug_img, aug_msk = random_transform_fg(
                fg,
                msk,
                cfg.max_translate_x,
                cfg.max_translate_y,
                cfg.max_rotate_deg,
            )
            aug_fg = aug_img * aug_msk
            aug_examples.append(aug_fg)

            coeffs = curvelet_op.forward(aug_fg.astype(np.float32))
            v = np.abs(flatten_curvelet_coeffs(coeffs)).astype(np.float64)
            mean_abs += v
            k = max(1, int(cfg.fg_sample_top_ratio * v.size))
            active = topk_mask_from_vector(v, k)
            freq += active.astype(np.float32)

    n_total = len(support_imgs) * (1 + cfg.n_aug_per_support)
    mean_abs /= max(1, n_total)

    rank = freq * (mean_abs / (mean_abs.max() + 1e-8))
    select = topk_mask_from_vector(rank, cfg.fg_topk_atoms)
    top_ids = np.argsort(-rank)[:cfg.n_vis_atoms]

    return select, rank, top_ids, aug_examples, zero_coeffs


def build_background_wavelet_dictionary(cfg: Cfg, support_imgs, support_masks):
    coeffs0 = swt2_detail_coeffs(
        np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32),
        cfg.wavelet,
        cfg.wavelet_level,
    )
    n_atoms = flatten_swt_detail_coeffs(coeffs0).size

    freq = np.zeros(n_atoms, dtype=np.float32)
    mean_abs = np.zeros(n_atoms, dtype=np.float64)

    for img, msk in zip(support_imgs, support_masks):
        bg = img * (1.0 - msk)

        coeffs = swt2_detail_coeffs(bg.astype(np.float32), cfg.wavelet, cfg.wavelet_level)
        v = np.abs(flatten_swt_detail_coeffs(coeffs)).astype(np.float64)
        mean_abs += v

        k = max(1, int(cfg.bg_sample_top_ratio * v.size))
        active = topk_mask_from_vector(v, k)
        freq += active.astype(np.float32)

    mean_abs /= max(1, len(support_imgs))
    rank = freq * (mean_abs / (mean_abs.max() + 1e-8))
    select = topk_mask_from_vector(rank, cfg.bg_topk_atoms)

    return select, rank


# =========================================================
# Segmentation
# =========================================================
def largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, lbl, index=np.arange(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return (lbl == best).astype(np.uint8)


def postprocess_mask(mask01: np.ndarray) -> np.ndarray:
    mask = mask01 > 0
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    mask = largest_component(mask.astype(np.uint8)) > 0
    return mask.astype(np.uint8)


def segment_one_image(
    img01: np.ndarray,
    cfg: Cfg,
    curvelet_op: UDCT,
    fg_select_mask: np.ndarray,
    bg_select_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # foreground by curvelet selected atoms
    curv_coeffs = curvelet_op.forward(img01.astype(np.float32))
    curv_masked = mask_curvelet_coeffs(curv_coeffs, fg_select_mask)
    fg_rec = np.real(curvelet_op.backward(curv_masked)).astype(np.float32)

    # background by selected wavelet atoms
    wav_coeffs = swt2_detail_coeffs(
        img01.astype(np.float32),
        cfg.wavelet,
        cfg.wavelet_level,
    )
    wav_masked = mask_swt_detail_coeffs(wav_coeffs, bg_select_mask)
    bg_rec = np.real(pywt.iswt2(wav_masked, cfg.wavelet, norm=True)).astype(np.float32)

    fg_map = robust_norm(np.abs(fg_rec))
    bg_map = robust_norm(np.abs(bg_rec))
    img_map = robust_norm(img01)

    score = fg_map - cfg.alpha_bg * bg_map
    prob = sigmoid(cfg.score_scale * score)
    prob = prob * (0.35 + 0.65 * img_map)

    pred = (prob > cfg.prob_thr).astype(np.uint8)
    if cfg.use_postprocess:
        pred = postprocess_mask(pred)

    return pred, score, prob, fg_map, bg_map


# =========================================================
# Piecewise candidate utils
# =========================================================
def center_distance(a: Candidate, b: Candidate) -> float:
    dy = a.center[0] - b.center[0]
    dx = a.center[1] - b.center[1]
    return float(math.hypot(dy, dx))


def pca_shape_stats(mask: np.ndarray) -> Tuple[float, float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return 0.0, 0.0, 999.0, 0.0, 0.0

    cy = float(ys.mean())
    cx = float(xs.mean())
    pts = np.stack([ys - cy, xs - cx], axis=1).astype(np.float64)
    cov = (pts.T @ pts) / max(1, pts.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    major = float(math.sqrt(max(evals[0], 1e-8)))
    minor = float(math.sqrt(max(evals[1], 1e-8)))
    ratio = major / (minor + 1e-8)
    v = evecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(v[0], v[1])))
    return cy, cx, ratio, major, angle_deg


def _points_in_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = poly.shape[0]
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for i in range(n):
        cond = ((y0[i] > y) != (y1[i] > y))
        xint = (x1[i] - x0[i]) * (y - y0[i]) / (y1[i] - y0[i] + 1e-12) + x0[i]
        inside ^= cond & (x < xint)
    return inside


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return mask.copy()

    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return mask.copy()
    pts = np.stack([xs, ys], axis=1)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return mask.copy()

    poly = pts[hull.vertices]
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    q = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = _points_in_poly(q, poly).reshape(mask.shape)
    return inside


def chain_distance_map(H: int, W: int, chain_hint: Optional[Sequence[Tuple[float, float]]]) -> np.ndarray:
    if not chain_hint:
        return np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([yy, xx], axis=-1).astype(np.float32)
    d2 = np.full((H, W), np.inf, dtype=np.float32)
    hint = np.array(chain_hint, dtype=np.float32)
    for p in hint:
        diff = pts - p[None, None, :]
        cur = diff[..., 0] ** 2 + diff[..., 1] ** 2
        d2 = np.minimum(d2, cur)
    return np.sqrt(d2)


def generate_vertebral_candidates(
    img01: np.ndarray,
    score: np.ndarray,   # 这里直接用 fg_map
    cfg: Cfg,
    chain_hint: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Candidate]:
    H, W = img01.shape
    score_n = robust_norm(score)
    img_n = robust_norm(img01)
    dist_map = chain_distance_map(H, W, chain_hint)

    fused = cfg.w_score * score_n + cfg.w_img * img_n
    fused = ndi.gaussian_filter(fused, sigma=cfg.score_sigma)

    masks: List[np.ndarray] = []
    for q in cfg.score_quantiles:
        thr = float(np.quantile(fused, q))
        m = fused >= thr
        m = ndi.binary_opening(m, structure=np.ones((3, 3), dtype=bool))
        m = ndi.binary_closing(m, structure=np.ones((5, 5), dtype=bool))
        m = fill_small_holes(m, area_thr=cfg.sdf_min_fill_hole)
        masks.append(m)

    cands: List[Candidate] = []
    for m in masks:
        lbl, n = ndi.label(m)
        if n == 0:
            continue
        objs = ndi.find_objects(lbl)
        for lab_idx, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            comp = lbl[sl] == lab_idx
            area = int(comp.sum())
            if area < cfg.cand_area_min or area > cfg.cand_area_max:
                continue

            full = np.zeros((H, W), dtype=bool)
            full[sl][comp] = True

            y0, x0, y1, x1 = bbox_from_mask(full)
            h = max(1, y1 - y0)
            w = max(1, x1 - x0)

            # 宽高比，不再用永远 >=1 的 elongation 写法
            aspect = float(w / (h + 1e-8))
            if aspect < cfg.cand_aspect_min or aspect > cfg.cand_aspect_max:
                continue

            cy, cx, pca_ratio, _, angle_deg = pca_shape_stats(full)
            if pca_ratio > cfg.cand_pca_ratio_max:
                continue

            hull = convex_hull_mask(full)
            solidity = float(area / (hull.sum() + 1e-8))
            if solidity < cfg.cand_solidity_min:
                continue

            mean_score = float(score_n[full].mean())
            mean_img = float(img_n[full].mean())
            chain_dist = float(dist_map[int(round(cy)), int(round(cx))]) if chain_hint else 0.0
            if chain_hint and chain_dist > cfg.cand_chain_dist_max:
                continue

            fused_score = (
                1.40 * mean_score
                + 0.35 * mean_img
                + 0.60 * solidity
                - 0.10 * max(0.0, pca_ratio - 1.8)
                - 0.015 * chain_dist
            )

            cands.append(
                Candidate(
                    mask=full.astype(np.uint8),
                    bbox=(y0, x0, y1, x1),
                    center=(cy, cx),
                    area=area,
                    mean_score=mean_score,
                    mean_img=mean_img,
                    fused_score=float(fused_score),
                    solidity=float(solidity),
                    aspect_ratio=float(aspect),
                    pca_ratio=float(pca_ratio),
                    angle_deg=float(angle_deg),
                    chain_dist=float(chain_dist),
                )
            )

    cands.sort(key=lambda c: c.fused_score, reverse=True)
    kept: List[Candidate] = []
    for c in cands:
        duplicate = False
        for k in kept:
            if center_distance(c, k) < cfg.cand_center_dedup:
                duplicate = True
                break
            if mask_iou(c.mask, k.mask) > cfg.cand_iou_dedup:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
        if len(kept) >= cfg.cand_max_keep:
            break
    return kept


def _edge_score(a: Candidate, b: Candidate, cfg: Cfg) -> Optional[float]:
    dy = b.center[0] - a.center[0]
    dx = abs(b.center[1] - a.center[1])
    if dy < cfg.edge_min_dy or dy > cfg.edge_max_dy:
        return None
    if dx > cfg.edge_max_dx:
        return None

    area_ratio = max(a.area, b.area) / (min(a.area, b.area) + 1e-8)
    if area_ratio > cfg.edge_max_scale_ratio:
        return None

    dtheta = abs(b.angle_deg - a.angle_deg)
    dtheta = min(dtheta, 180.0 - dtheta)

    return float(
        0.35
        - cfg.chain_dx_penalty * dx
        - cfg.chain_dy_penalty * dy
        - cfg.chain_scale_penalty * abs(math.log(area_ratio + 1e-8))
        - cfg.chain_angle_penalty * (dtheta / 45.0)
    )


def select_candidate_chain(cands: Sequence[Candidate], cfg: Cfg) -> List[Candidate]:
    if len(cands) == 0:
        return []

    nodes = sorted(cands, key=lambda c: (c.center[0], c.center[1]))
    n = len(nodes)

    dp = np.full(n, -1e18, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    length = np.ones(n, dtype=np.int32)

    for i, ci in enumerate(nodes):
        dp[i] = ci.fused_score
        prev[i] = -1
        for j in range(i):
            ej = _edge_score(nodes[j], ci, cfg)
            if ej is None:
                continue
            val = dp[j] + ci.fused_score + ej - cfg.chain_gap_penalty
            if val > dp[i]:
                dp[i] = val
                prev[i] = j
                length[i] = length[j] + 1

    valid_ids = [i for i in range(n) if length[i] >= cfg.chain_min_len]
    end = int(max(valid_ids, key=lambda i: dp[i])) if valid_ids else int(np.argmax(dp))

    chain_rev = []
    cur = end
    while cur >= 0:
        chain_rev.append(nodes[cur])
        cur = int(prev[cur])
    chain = list(reversed(chain_rev))

    pruned: List[Candidate] = []
    for c in chain:
        if not pruned:
            pruned.append(c)
            continue
        last = pruned[-1]
        if mask_iou(last.mask, c.mask) > 0.35:
            if c.fused_score > last.fused_score:
                pruned[-1] = c
            continue
        pruned.append(c)
    return pruned


def _repair_candidate_mask(mask: np.ndarray, hole_thr: int) -> np.ndarray:
    m = mask > 0
    m = ndi.binary_opening(m, structure=np.ones((3, 3), dtype=bool))
    m = ndi.binary_closing(m, structure=np.ones((5, 5), dtype=bool))
    m = fill_small_holes(m, area_thr=hole_thr)
    lbl, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lbl, index=np.arange(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        m = lbl == keep
    return m


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    outside = ndi.distance_transform_edt(~mask)
    inside = ndi.distance_transform_edt(mask)
    return (outside - inside).astype(np.float32)


def build_piecewise_sdf(
    chain_cands: Sequence[Candidate],
    H: int,
    W: int,
    cfg: Cfg,
):
    gamma_list: List[np.ndarray] = []
    window_list: List[Tuple[int, int, int, int]] = []
    soft_union = np.zeros((H, W), dtype=np.float32)
    hard_union = np.zeros((H, W), dtype=np.uint8)

    for c in chain_cands:
        repaired = _repair_candidate_mask(c.mask.astype(bool), cfg.sdf_min_fill_hole)
        y0, x0, y1, x1 = bbox_from_mask(repaired)
        hh = max(1, y1 - y0)
        ww = max(1, x1 - x0)
        margin = max(cfg.sdf_margin_min, int(round(cfg.sdf_margin_ratio * max(hh, ww))))
        wy0, wx0, wy1, wx1 = crop_box(y0 - margin, x0 - margin, y1 + margin, x1 + margin, H, W)

        local = repaired[wy0:wy1, wx0:wx1]
        if local.sum() == 0:
            continue

        phi = _signed_distance(local)
        gamma_local = ndi.gaussian_filter(phi, sigma=cfg.sdf_sigma).astype(np.float32)

        gamma_full = np.zeros((H, W), dtype=np.float32)
        gamma_full[wy0:wy1, wx0:wx1] = gamma_local
        gamma_list.append(gamma_full)
        window_list.append((wy0, wx0, wy1, wx1))

        soft_local = np.exp(-(phi ** 2) / (2.0 * cfg.sdf_tau_soft ** 2)).astype(np.float32)
        soft_union[wy0:wy1, wx0:wx1] = np.maximum(soft_union[wy0:wy1, wx0:wx1], soft_local)
        hard_union[wy0:wy1, wx0:wx1] |= local.astype(np.uint8)

    return gamma_list, window_list, np.clip(soft_union, 0.0, 1.0), hard_union

def build_chain_band(
    chain: Sequence[Candidate],
    H: int,
    W: int,
    extra_margin: int = 8,
    min_half_width: int = 12,
) -> np.ndarray:
    if len(chain) == 0:
        return np.ones((H, W), dtype=np.float32)

    ys = np.array([c.center[0] for c in chain], dtype=np.float32)
    xs = np.array([c.center[1] for c in chain], dtype=np.float32)
    ws = np.array([c.bbox[3] - c.bbox[1] for c in chain], dtype=np.float32)

    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]
    ws = ws[order]

    yq = np.arange(H, dtype=np.float32)

    if len(chain) == 1:
        x_line = np.full(H, xs[0], dtype=np.float32)
        w_line = np.full(H, ws[0], dtype=np.float32)
    else:
        x_line = np.interp(yq, ys, xs, left=xs[0], right=xs[-1]).astype(np.float32)
        w_line = np.interp(yq, ys, ws, left=ws[0], right=ws[-1]).astype(np.float32)

    half_width = np.maximum(min_half_width, 0.75 * w_line + extra_margin)

    band = np.zeros((H, W), dtype=np.float32)
    xx = np.arange(W, dtype=np.float32)
    for y in range(H):
        band[y, :] = (np.abs(xx - x_line[y]) <= half_width[y]).astype(np.float32)

    band = ndi.binary_dilation(band > 0, structure=np.ones((5, 5), dtype=bool)).astype(np.float32)
    return band

def build_chain_band(
    chain: Sequence[Candidate],
    H: int,
    W: int,
    extra_margin: int = 8,
    min_half_width: int = 12,
) -> np.ndarray:
    """
    根据 selected chain 构造一条纵向窄带先验 band。
    band=1 表示允许前景出现的区域，band=0 表示强抑制区域。
    """
    if len(chain) == 0:
        return np.ones((H, W), dtype=np.float32)

    ys = np.array([c.center[0] for c in chain], dtype=np.float32)
    xs = np.array([c.center[1] for c in chain], dtype=np.float32)
    ws = np.array([c.bbox[3] - c.bbox[1] for c in chain], dtype=np.float32)

    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]
    ws = ws[order]

    yq = np.arange(H, dtype=np.float32)

    if len(chain) == 1:
        x_line = np.full(H, xs[0], dtype=np.float32)
        w_line = np.full(H, ws[0], dtype=np.float32)
    else:
        x_line = np.interp(yq, ys, xs, left=xs[0], right=xs[-1]).astype(np.float32)
        w_line = np.interp(yq, ys, ws, left=ws[0], right=ws[-1]).astype(np.float32)

    half_width = np.maximum(min_half_width, 0.75 * w_line + extra_margin)

    band = np.zeros((H, W), dtype=np.float32)
    xx = np.arange(W, dtype=np.float32)
    for y in range(H):
        band[y, :] = (np.abs(xx - x_line[y]) <= half_width[y]).astype(np.float32)

    band = ndi.binary_dilation(band > 0, structure=np.ones((5, 5), dtype=bool)).astype(np.float32)
    return band

# =========================================================
# Visualization
# =========================================================
def save_atom_visual(cfg: Cfg, curvelet_op: UDCT, template_coeffs, atom_ids: np.ndarray):
    n = len(atom_ids)
    rows = int(math.ceil(n / 4))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, atom_id in zip(axes, atom_ids):
        atom_coeffs = single_curvelet_atom_like(template_coeffs, int(atom_id), value=1.0 + 0.0j)
        atom = np.real(curvelet_op.backward(atom_coeffs))
        atom = robust_norm(np.abs(atom))
        ax.imshow(atom, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"atom #{int(atom_id)}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "top_curvelet_atoms.png"), dpi=150)
    plt.close(fig)


def save_aug_examples(cfg: Cfg, aug_examples: List[np.ndarray]):
    import matplotlib.pyplot as plt
    n = min(12, len(aug_examples))
    if n == 0:
        return
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.ravel()
    for i in range(n):
        axes[i].imshow(aug_examples[i], cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Aug {i}")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "support_aug_examples.png"), dpi=150)
    plt.close(fig)


def save_piecewise_vis(
    path: str,
    img01: np.ndarray,
    fg_map: np.ndarray,
    cands: Sequence[Candidate],
    chain: Sequence[Candidate],
    cand_soft: np.ndarray,
    cand_union: np.ndarray,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].imshow(img01, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("img")
    axes[0].axis("off")

    axes[1].imshow(fg_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("fg_map")
    axes[1].axis("off")

    axes[2].imshow(img01, cmap="gray", vmin=0, vmax=1)
    for c in cands:
        y0, x0, y1, x1 = c.bbox
        axes[2].add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.0)
        )
    axes[2].set_title(f"candidates: {len(cands)}")
    axes[2].axis("off")

    axes[3].imshow(img01, cmap="gray", vmin=0, vmax=1)
    for c in chain:
        y0, x0, y1, x1 = c.bbox
        axes[3].add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2.0)
        )
        axes[3].plot(c.center[1], c.center[0], "ro", markersize=3)
    axes[3].set_title(f"selected chain: {len(chain)}")
    axes[3].axis("off")

    axes[4].imshow(cand_soft, cmap="hot", vmin=0, vmax=1)
    axes[4].set_title("cand_soft")
    axes[4].axis("off")

    axes[5].imshow(cand_union, cmap="gray", vmin=0, vmax=1)
    axes[5].set_title("cand_union")
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    set_seed(cfg.seed)

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "vis"))

    # 所有样本的 piecewise 总目录（中间目录）
    vis_piecewise_all_dir = os.path.join(cfg.out_dir, "vis_piecewise_all")
    ensure_dir(vis_piecewise_all_dir)

    # 最终筛选结果目录
    vis_piecewise_top30_dir = os.path.join(cfg.out_dir, "vis_piecewise_top30")
    vis_piecewise_bottom30_dir = os.path.join(cfg.out_dir, "vis_piecewise_bottom30")
    ensure_dir(vis_piecewise_top30_dir)
    ensure_dir(vis_piecewise_bottom30_dir)

    items = build_case_list(cfg.root, cfg.img_dir, cfg.mask_dir)
    if len(items) == 0:
        raise RuntimeError("没有找到任何图像/掩码，请检查目录。")

    support_pids, query_pids, support_items, query_items = split_support_query(
        items, cfg.n_support_patients, cfg.seed
    )

    print("=" * 70)
    print("Support patient IDs:", support_pids)
    print("Query   patient IDs:", query_pids)
    print(f"#support slices = {len(support_items)}")
    print(f"#query   slices = {len(query_items)}")
    print("=" * 70)

    with open(os.path.join(cfg.out_dir, "split.txt"), "w", encoding="utf-8") as f:
        f.write("Support patient IDs:\n")
        f.write(",".join(support_pids) + "\n")
        f.write("Query patient IDs:\n")
        f.write(",".join(query_pids) + "\n")
        f.write(f"#support slices={len(support_items)}\n")
        f.write(f"#query slices={len(query_items)}\n")

    print("Loading and horizontally centering support/query by mask centroid...")
    sup_imgs, sup_masks, sup_names, sup_shifts = load_items_centered(support_items)
    qry_imgs, qry_masks, qry_names, qry_shifts = load_items_centered(query_items)

    with open(os.path.join(cfg.out_dir, "centering_shifts.txt"), "w", encoding="utf-8") as f:
        f.write("=== Support shifts ===\n")
        for name, sx in zip(sup_names, sup_shifts):
            f.write(f"{name}: shift_x={sx}\n")
        f.write("\n=== Query shifts ===\n")
        for name, sx in zip(qry_names, qry_shifts):
            f.write(f"{name}: shift_x={sx}\n")

    print("Initializing curvelet transform...")
    curvelet_op = UDCT(
        shape=(cfg.image_size, cfg.image_size),
        num_scales=cfg.num_scales,
        wedges_per_direction=cfg.wedges_per_direction,
    )

    print("Building foreground curvelet dictionary with augmentation...")
    fg_select_mask, fg_rank, top_atom_ids, aug_examples, template_coeffs = build_foreground_curvelet_dictionary(
        cfg, sup_imgs, sup_masks, curvelet_op
    )
    save_aug_examples(cfg, aug_examples)

    print("Building background wavelet dictionary...")
    bg_select_mask, bg_rank = build_background_wavelet_dictionary(cfg, sup_imgs, sup_masks)

    print("Segmenting queries + piecewise candidate analysis...")
    rows = []
    dices = []
    ious = []

    for idx, (img, gt, name, item) in enumerate(zip(qry_imgs, qry_masks, qry_names, query_items)):
        pred, score, prob, fg_map, bg_map = segment_one_image(
            img01=img,
            cfg=cfg,
            curvelet_op=curvelet_op,
            fg_select_mask=fg_select_mask,
            bg_select_mask=bg_select_mask,
        )

        # -------------------------
        # piecewise candidate branch
        # score 直接用 fg_map
        # -------------------------
        cands = generate_vertebral_candidates(
            img01=img,
            score=fg_map,
            cfg=cfg,
            chain_hint=None,
        )
        chain = select_candidate_chain(cands, cfg)
        gamma_list, window_list, cand_soft, cand_union = build_piecewise_sdf(
            chain_cands=chain,
            H=img.shape[0],
            W=img.shape[1],
            cfg=cfg,
        )

        # -------------------------
        # prob gating by piecewise candidates
        # -------------------------
        band = build_chain_band(
            chain=chain,
            H=img.shape[0],
            W=img.shape[1],
            extra_margin=8,
            min_half_width=12,
        )

        cand_prior = np.maximum(
            cand_soft,
            ndi.gaussian_filter(cand_union.astype(np.float32), sigma=2.0)
        )

        prob_gated = prob * (0.15 + 0.85 * cand_prior) * (0.25 + 0.75 * band)
        prob_gated = np.clip(prob_gated, 0.0, 1.0)

        vals = prob_gated[band > 0.5]
        if vals.size > 0:
            thr = float(np.clip(vals.mean() + 0.15 * vals.std(), 0.35, 0.65))
        else:
            thr = cfg.prob_thr

        pred = (prob_gated > thr).astype(np.uint8)
        if cfg.use_postprocess:
            pred = postprocess_mask(pred)

        dice, iou = dice_iou(pred, gt.astype(np.uint8))
        dices.append(dice)
        ious.append(iou)

        piecewise_path = os.path.join(vis_piecewise_all_dir, f"{name}_piecewise.png")

        rows.append({
            "name": name,
            "pid": item["pid"],
            "dice": dice,
            "iou": iou,
            "score_mean": float(score.mean()),
            "score_std": float(score.std()),
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "n_cands": len(cands),
            "n_chain": len(chain),
            "cand_union_area": int(cand_union.sum()),
            "n_gamma": len(gamma_list),
            "piecewise_path": piecewise_path,
        })
        save_piecewise_vis(
            piecewise_path,
            img01=img,
            fg_map=fg_map,
            cands=cands,
            chain=chain,
            cand_soft=cand_soft,
            cand_union=cand_union,
        )
        if idx < cfg.save_vis_n:
            save_mask(os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_pred.png"), pred)
            save_mask(os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_gt.png"), gt)

            save_overlay(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_overlay.png"),
                img, pred, gt
            )

            Image.fromarray((normalize01(score) * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_score.png")
            )
            Image.fromarray((prob * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_prob.png")
            )
            Image.fromarray((prob_gated * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_prob_gated.png")
            )
            Image.fromarray((fg_map * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_fgmap.png")
            )
            Image.fromarray((bg_map * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_bgmap.png")
            )
            Image.fromarray((cand_soft * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_candsoft.png")
            )
            Image.fromarray((cand_union * 255).astype(np.uint8)).save(
                os.path.join(cfg.out_dir, "vis", f"{idx:03d}_{name}_candunion.png")
            )


        print(
            f"[{idx+1:04d}/{len(query_items):04d}] {name}  "
            f"Dice={dice:.4f}  IoU={iou:.4f}  "
            f"cands={len(cands)}  chain={len(chain)}  cand_area={int(cand_union.sum())}"
        )
    # =====================================================
    # Save top-30 / bottom-30 piecewise visualizations by Dice
    # =====================================================
    rows_sorted_desc = sorted(rows, key=lambda r: r["dice"], reverse=True)
    rows_sorted_asc = sorted(rows, key=lambda r: r["dice"])

    top_k = min(30, len(rows_sorted_desc))
    bottom_k = min(30, len(rows_sorted_asc))

    # 清空旧结果，避免重复残留
    for d in [vis_piecewise_top30_dir, vis_piecewise_bottom30_dir]:
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp):
                os.remove(fp)

    # 复制 Dice 最高 30
    for rank, row in enumerate(rows_sorted_desc[:top_k], start=1):
        src = row["piecewise_path"]
        dst = os.path.join(
            vis_piecewise_top30_dir,
            f"{rank:02d}_dice_{row['dice']:.4f}_iou_{row['iou']:.4f}_{row['name']}.png"
        )
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # 复制 Dice 最低 30
    for rank, row in enumerate(rows_sorted_asc[:bottom_k], start=1):
        src = row["piecewise_path"]
        dst = os.path.join(
            vis_piecewise_bottom30_dir,
            f"{rank:02d}_dice_{row['dice']:.4f}_iou_{row['iou']:.4f}_{row['name']}.png"
        )
        if os.path.exists(src):
            shutil.copy2(src, dst)
    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    std_dice = float(np.std(dices)) if len(dices) > 0 else 0.0
    std_iou = float(np.std(ious)) if len(ious) > 0 else 0.0

    print("=" * 70)
    print(f"Mean Dice = {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean IoU  = {mean_iou:.4f} ± {std_iou:.4f}")
    print("=" * 70)

    with open(os.path.join(cfg.out_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("name,pid,dice,iou,score_mean,score_std,score_min,score_max,n_cands,n_chain,cand_union_area,n_gamma\n")
        for row in rows:
            f.write(
                f"{row['name']},{row['pid']},"
                f"{row['dice']:.6f},{row['iou']:.6f},"
                f"{row['score_mean']:.6f},{row['score_std']:.6f},"
                f"{row['score_min']:.6f},{row['score_max']:.6f},"
                f"{row['n_cands']},{row['n_chain']},{row['cand_union_area']},{row['n_gamma']}\n"
            )

    with open(os.path.join(cfg.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Curvelet + Wavelet segmentation + piecewise candidates ===\n")
        f.write(f"num_scales = {cfg.num_scales}\n")
        f.write(f"wedges_per_direction = {cfg.wedges_per_direction}\n")
        f.write(f"wavelet = {cfg.wavelet}\n")
        f.write(f"wavelet_level = {cfg.wavelet_level}\n")
        f.write(f"n_aug_per_support = {cfg.n_aug_per_support}\n")
        f.write(f"max_translate_x = {cfg.max_translate_x}\n")
        f.write(f"max_translate_y = {cfg.max_translate_y}\n")
        f.write(f"max_rotate_deg = {cfg.max_rotate_deg}\n")
        f.write(f"fg_topk_atoms = {cfg.fg_topk_atoms}\n")
        f.write(f"bg_topk_atoms = {cfg.bg_topk_atoms}\n")
        f.write(f"alpha_bg = {cfg.alpha_bg}\n")
        f.write(f"score_scale = {cfg.score_scale}\n")
        f.write(f"prob_thr = {cfg.prob_thr}\n")
        f.write("\n")
        f.write("=== piecewise candidate cfg ===\n")
        f.write(f"score_quantiles = {cfg.score_quantiles}\n")
        f.write(f"cand_area_min = {cfg.cand_area_min}\n")
        f.write(f"cand_area_max = {cfg.cand_area_max}\n")
        f.write(f"cand_aspect_min = {cfg.cand_aspect_min}\n")
        f.write(f"cand_aspect_max = {cfg.cand_aspect_max}\n")
        f.write(f"cand_solidity_min = {cfg.cand_solidity_min}\n")
        f.write(f"cand_pca_ratio_max = {cfg.cand_pca_ratio_max}\n")
        f.write(f"chain_min_len = {cfg.chain_min_len}\n")
        f.write(f"sdf_sigma = {cfg.sdf_sigma}\n")
        f.write(f"sdf_tau_soft = {cfg.sdf_tau_soft}\n")
        f.write("\n")
        f.write(f"Support patient IDs: {support_pids}\n")
        f.write(f"Query patient IDs: {query_pids}\n")
        f.write(f"Support slices: {len(support_items)}\n")
        f.write(f"Query slices: {len(query_items)}\n")
        f.write(f"\nMean Dice: {mean_dice:.6f}\n")
        f.write(f"Std Dice: {std_dice:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")
        f.write(f"Std IoU: {std_iou:.6f}\n")
        f.write(f"\nTop-30 piecewise dir: {vis_piecewise_top30_dir}\n")
        f.write(f"Bottom-30 piecewise dir: {vis_piecewise_bottom30_dir}\n")

    print("Saving top activated curvelet atoms...")
    save_atom_visual(cfg, curvelet_op, template_coeffs, top_atom_ids)

    print("Saved to:", cfg.out_dir)


if __name__ == "__main__":
    main()