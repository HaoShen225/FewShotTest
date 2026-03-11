from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    root: str = "SPIDER_224_2d_png"
    img_dir: str = "images_t1"
    mask_dir: str = "masks_t1"
    out_dir: str = "Dictionary_Method/piecewise_candidates"

    out_size: int = 224

    # candidate score fusion
    w_score: float = 0.75
    w_img: float = 0.25
    score_quantiles: Tuple[float, ...] = (0.78, 0.84, 0.88, 0.92)
    score_sigma: float = 1.0

    # candidate filters
    cand_area_min: int = 80
    cand_area_max: int = 4200
    cand_aspect_min: float = 0.45
    cand_aspect_max: float = 2.40
    cand_solidity_min: float = 0.35
    cand_pca_ratio_max: float = 4.80
    cand_chain_dist_max: float = 48.0
    cand_center_dedup: float = 12.0
    cand_iou_dedup: float = 0.55
    cand_max_keep: int = 40

    # chain selection
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

    # piecewise sdf
    sdf_margin_ratio: float = 0.35
    sdf_margin_min: int = 6
    sdf_sigma: float = 1.25
    sdf_tau_soft: float = 5.0
    sdf_min_fill_hole: int = 12


# =========================================================
# Data structures
# =========================================================
@dataclass
class Candidate:
    mask: np.ndarray               # full-res binary mask (H,W)
    bbox: Tuple[int, int, int, int]  # y0, x0, y1, x1 inclusive-exclusive
    center: Tuple[float, float]    # cy, cx
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
# Utils
# =========================================================
def robust_norm(x: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(x, p1))
    hi = float(np.percentile(x, p99))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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


def center_distance(a: Candidate, b: Candidate) -> float:
    dy = a.center[0] - b.center[0]
    dx = a.center[1] - b.center[1]
    return float(math.hypot(dy, dx))


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
        # ignore background-connected holes touching image boundary
        if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
            continue
        out[ys, xs] = True
    return out


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


def _points_in_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    # ray casting
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


def chain_distance_map(H: int, W: int, chain_hint: Optional[Sequence[Tuple[float, float]]]) -> np.ndarray:
    if not chain_hint:
        return np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([yy, xx], axis=-1).astype(np.float32)
    d2 = np.full((H, W), np.inf, dtype=np.float32)
    hint = np.array(chain_hint, dtype=np.float32)
    for p in hint:
        diff = pts - p[None, None, :]
        cur = (diff[..., 0] ** 2 + diff[..., 1] ** 2)
        d2 = np.minimum(d2, cur)
    return np.sqrt(d2)


# =========================================================
# Score map from previous curvelet+wavelet branch
# =========================================================
def curvelet_wavelet_score_to_prob(score: np.ndarray, img01: np.ndarray, score_scale: float = 7.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses the same downstream score->prob idea as the earlier curvelet+wavelet branch:
    prob = sigmoid(score_scale * score) * (0.35 + 0.65 * img_map)
    Here score is assumed to already be fg_map - alpha_bg * bg_map.
    """
    img_map = robust_norm(img01)
    prob = sigmoid(score_scale * score)
    prob = prob * (0.35 + 0.65 * img_map)
    return score.astype(np.float32), np.clip(prob, 0.0, 1.0).astype(np.float32)


# =========================================================
# 1) Candidate generation from previous score
# =========================================================
def generate_vertebral_candidates(
    img01: np.ndarray,
    score: np.ndarray,
    cfg: Cfg,
    chain_hint: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Candidate]:
    """
    Generate block-like vertebral candidates directly from the previous curvelet+wavelet score.

    Args:
        img01: grayscale image in [0,1], shape (H,W)
        score: score map from previous branch, same shape (H,W)
        cfg: configuration
        chain_hint: optional list of (cy,cx) points for weak distance gating

    Returns:
        List[Candidate], sorted by fused_score descending.
    """
    H, W = img01.shape
    score_n = robust_norm(score)
    img_n = robust_norm(img01)
    dist_map = chain_distance_map(H, W, chain_hint)

    fused = cfg.w_score * score_n + cfg.w_img * img_n
    fused = ndi.gaussian_filter(fused, sigma=cfg.score_sigma)

    # multi-threshold proposal extraction
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

            # basic bbox / aspect
            y0, x0, y1, x1 = bbox_from_mask(full)
            h = max(1, y1 - y0)
            w = max(1, x1 - x0)
            aspect = float(w / h) if w >= h else float(h / w)
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

            # node score: emphasize score, reward blockiness/solidity, lightly reward image brightness
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

    # deduplicate near-identical proposals
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


# =========================================================
# 2) Candidate chain selection by DAG dynamic programming
# =========================================================
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
    """
    Select a vertebral main chain from the candidate pool using DP on a DAG.
    Nodes are sorted by vertical location; edges prefer moderate dy, small dx, stable size/angle.
    """
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

    # choose best sufficiently long path; if none, fall back to best node
    valid_ids = [i for i in range(n) if length[i] >= cfg.chain_min_len]
    end = int(max(valid_ids, key=lambda i: dp[i])) if valid_ids else int(np.argmax(dp))

    chain_rev = []
    cur = end
    while cur >= 0:
        chain_rev.append(nodes[cur])
        cur = int(prev[cur])
    chain = list(reversed(chain_rev))

    # prune strong overlaps inside chain, keep monotonic order
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


# =========================================================
# 3) Piecewise SDF construction
# =========================================================
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
    # negative inside, positive outside
    outside = ndi.distance_transform_edt(~mask)
    inside = ndi.distance_transform_edt(mask)
    return (outside - inside).astype(np.float32)


def build_piecewise_sdf(
    chain_cands: Sequence[Candidate],
    H: int,
    W: int,
    cfg: Cfg,
):
    """
    Build piecewise structural fields from selected vertebral candidates.

    Returns:
        gamma_list: list of smoothed local SDF maps, each full-size (H,W) with zeros outside local window
        window_list: list of (y0,x0,y1,x1)
        cand_soft: soft union map from all candidate-local SDFs
        cand_union: hard union mask of all candidate regions/windows
    """
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


__all__ = [
    "Cfg",
    "Candidate",
    "curvelet_wavelet_score_to_prob",
    "generate_vertebral_candidates",
    "select_candidate_chain",
    "build_piecewise_sdf",
]
