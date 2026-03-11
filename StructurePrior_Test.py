import os
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    root: str = 'SPIDER_224_2d_png'
    img_dir: str = 'images_t1'
    mask_dir: str = 'masks_t1'
    out_dir: str = 'SPIDER_224_2d_png/spider_t1_patch_cosine_out'

    patch: int = 14          # 224 / 14 = 16
    image_size: int = 224
    eps: float = 1e-6

    # patch class by mask coverage
    fg_thr: float = 0.50
    bg_thr: float = 0.05

    # structure feature config
    hog_bins: int = 8
    dct_keep: int = 4
    edge_thr: float = 0.12

    # dictionary experiment (unused here, kept for compatibility)
    dict_atoms: int = 24
    dict_iters: int = 20
    dict_topk: int = 4
    dict_train_max: int = 4096
    dict_seed: int = 0

    # candidate generation
    max_candidates_per_image: int = 12
    min_component_area: int = 6
    max_component_area: int = 1200
    bright_quantile: float = 0.93
    center_sigma: float = 1.2
    ring_inner: int = 10
    ring_outer: int = 22
    nms_radius: int = 18

    # local crop
    crop_size: int = 96

    # visualization
    dpi: int = 160
    cmap: str = 'magma'

    # experiment scope
    max_images: int = 20


# =========================
# Basic IO
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def robust_normalize(x: np.ndarray, q1: float = 0.01, q2: float = 0.99) -> np.ndarray:
    lo = np.quantile(x, q1)
    hi = np.quantile(x, q2)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0.0, 1.0)


def list_pairs(img_root: Path, mask_root: Path) -> List[Tuple[Path, Path]]:
    img_paths = sorted(img_root.glob('*.png'))
    pairs = []
    missing = []

    mask_map = {p.name: p for p in mask_root.glob('*.png')}

    for ip in img_paths:
        mp = mask_map.get(ip.name, None)
        if mp is None:
            missing.append(ip.name)
            continue
        pairs.append((ip, mp))

    if missing:
        print(f'[WARN] Missing masks for {len(missing)} images. First few: {missing[:5]}')
    return pairs


# =========================
# Image ops
# =========================
def gaussian_blur_np(x: np.ndarray, radius: float) -> np.ndarray:
    """Use PIL GaussianBlur for simplicity."""
    img = Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def sobel_grad(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple Sobel gradient using manual padding.
    """
    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    ky = kx.T

    xp = np.pad(x, ((1, 1), (1, 1)), mode='reflect')
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)

    H, W = x.shape
    for i in range(H):
        for j in range(W):
            patch = xp[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * kx)
            gy[i, j] = np.sum(patch * ky)

    gm = np.sqrt(gx * gx + gy * gy)
    gm = gm / (gm.max() + 1e-6)
    return gx, gy, gm


def make_ring_masks(h: int, w: int, cx: float, cy: float,
                    r_in: int, r_out: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    center = dist2 <= (r_in ** 2)
    ring = (dist2 > (r_in ** 2)) & (dist2 <= (r_out ** 2))
    return center, ring


# =========================
# Connected Components
# =========================
def connected_components(binary: np.ndarray) -> List[np.ndarray]:
    """
    8-connected components. Returns list of arrays of coordinates (N,2) as (y,x).
    """
    H, W = binary.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    comps = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    ys, xs = np.where(binary)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue

        stack = [(sy, sx)]
        visited[sy, sx] = 1
        pts = []

        while stack:
            y, x = stack.pop()
            pts.append((y, x))

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))

        comps.append(np.array(pts, dtype=np.int32))

    return comps


# =========================
# Candidate generation
# =========================
def component_to_candidate(
    comp: np.ndarray,
    score_map: np.ndarray,
    img: np.ndarray,
    mask: np.ndarray,
    edge_map: np.ndarray,
    cfg: Cfg
) -> Dict:
    ys = comp[:, 0]
    xs = comp[:, 1]

    area = len(comp)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    comp_score = float(score_map[ys, xs].mean())
    comp_bright = float(img[ys, xs].mean())

    center_mask, ring_mask = make_ring_masks(
        img.shape[0], img.shape[1], cx, cy, cfg.ring_inner, cfg.ring_outer
    )

    center_mean = float(img[center_mask].mean()) if center_mask.any() else 0.0
    ring_mean = float(img[ring_mask].mean()) if ring_mask.any() else 0.0
    ring_edge = float(edge_map[ring_mask].mean()) if ring_mask.any() else 0.0

    # foreground fraction inside patch-sized neighborhood
    half = cfg.patch // 2
    yy0 = max(0, int(round(cy)) - half)
    yy1 = min(img.shape[0], int(round(cy)) + half)
    xx0 = max(0, int(round(cx)) - half)
    xx1 = min(img.shape[1], int(round(cx)) + half)
    local_mask = mask[yy0:yy1, xx0:xx1]
    mask_frac = float(local_mask.mean()) if local_mask.size > 0 else 0.0

    # candidate score:
    # bright center + strong component score + darker ring + some edge texture on ring
    final_score = (
        1.20 * center_mean
        + 1.00 * comp_score
        - 0.80 * ring_mean
        + 0.35 * ring_edge
    )

    if mask_frac >= cfg.fg_thr:
        label = 'fg'
    elif mask_frac <= cfg.bg_thr:
        label = 'bg'
    else:
        label = 'mid'

    return {
        'cx': cx,
        'cy': cy,
        'area': area,
        'bbox_x0': x0,
        'bbox_y0': y0,
        'bbox_x1': x1,
        'bbox_y1': y1,
        'comp_score': comp_score,
        'comp_bright': comp_bright,
        'center_mean': center_mean,
        'ring_mean': ring_mean,
        'ring_edge': ring_edge,
        'mask_frac': mask_frac,
        'label': label,
        'score': float(final_score),
    }


def nms_candidates(cands: List[Dict], radius: int) -> List[Dict]:
    """
    Simple distance-based NMS on candidate centers.
    """
    if not cands:
        return []

    cands = sorted(cands, key=lambda d: d['score'], reverse=True)
    keep = []

    for c in cands:
        ok = True
        for k in keep:
            dx = c['cx'] - k['cx']
            dy = c['cy'] - k['cy']
            if dx * dx + dy * dy < radius * radius:
                ok = False
                break
        if ok:
            keep.append(c)

    return keep


def disk_ring_score_map(
    img: np.ndarray,
    edge_map: np.ndarray,
    r_disk: int = 5,
    r_in: int = 8,
    r_out: int = 16,
    alpha_ring: float = 1.0,
    beta_edge: float = 0.35,
) -> Dict[str, np.ndarray]:
    """
    Compute disk-minus-ring score map.

    score = mean(disk) - alpha * mean(ring) - beta * mean(edge on disk)

    Intuition:
      - prefer locally bright blob centers
      - prefer darker outer ring
      - suppress thin strong edges a bit
    """
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]

    disk_offsets = []
    ring_offsets = []

    for dy in range(-r_out, r_out + 1):
        for dx in range(-r_out, r_out + 1):
            d2 = dx * dx + dy * dy
            if d2 <= r_disk * r_disk:
                disk_offsets.append((dy, dx))
            elif r_in * r_in < d2 <= r_out * r_out:
                ring_offsets.append((dy, dx))

    disk_sum = np.zeros_like(img, dtype=np.float32)
    ring_sum = np.zeros_like(img, dtype=np.float32)
    edge_disk_sum = np.zeros_like(img, dtype=np.float32)

    for dy, dx in disk_offsets:
        shifted_img = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        shifted_edge = np.roll(np.roll(edge_map, dy, axis=0), dx, axis=1)
        disk_sum += shifted_img
        edge_disk_sum += shifted_edge

    for dy, dx in ring_offsets:
        shifted_img = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        ring_sum += shifted_img

    disk_mean = disk_sum / max(len(disk_offsets), 1)
    ring_mean = ring_sum / max(len(ring_offsets), 1)
    edge_disk_mean = edge_disk_sum / max(len(disk_offsets), 1)

    # kill wrap-around artifacts from np.roll by zeroing boundary margins
    margin = r_out + 1
    valid = np.zeros_like(img, dtype=np.float32)
    valid[margin:H - margin, margin:W - margin] = 1.0

    score = disk_mean - alpha_ring * ring_mean - beta_edge * edge_disk_mean
    score *= valid

    return {
        'score_raw': score.astype(np.float32),
        'disk_mean': disk_mean.astype(np.float32),
        'ring_mean': ring_mean.astype(np.float32),
        'edge_disk_mean': edge_disk_mean.astype(np.float32),
        'valid': valid.astype(np.float32),
    }


def local_max_binary(score_map: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Simple local maxima detector.
    A pixel is kept if it equals the max in its local window.
    """
    H, W = score_map.shape
    out = np.zeros((H, W), dtype=bool)

    for y in range(H):
        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)
        for x in range(W):
            x0 = max(0, x - radius)
            x1 = min(W, x + radius + 1)
            v = score_map[y, x]
            if v >= score_map[y0:y1, x0:x1].max():
                out[y, x] = True
    return out


# =========================
# Crop utilities
# =========================
def safe_crop(arr: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    """
    Center crop with zero padding if near border.
    """
    H, W = arr.shape
    hs = crop_size // 2
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    x0 = cx_i - hs
    x1 = x0 + crop_size
    y0 = cy_i - hs
    y1 = y0 + crop_size

    out = np.zeros((crop_size, crop_size), dtype=arr.dtype)

    sx0 = max(0, x0)
    sx1 = min(W, x1)
    sy0 = max(0, y0)
    sy1 = min(H, y1)

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def make_candidate_ring_maps(h: int, w: int, cx: float, cy: float, cfg: Cfg) -> Tuple[np.ndarray, np.ndarray]:
    center_mask, ring_mask = make_ring_masks(h, w, cx, cy, cfg.ring_inner, cfg.ring_outer)
    return center_mask.astype(np.float32), ring_mask.astype(np.float32)


# =========================
# Visualization
# =========================
def draw_overview(
    img: np.ndarray,
    mask: np.ndarray,
    aux: Dict[str, np.ndarray],
    cands: List[Dict],
    out_path: Path,
    title: str,
    cfg: Cfg
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=cfg.dpi)

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(aux['score_map'], cmap=cfg.cmap)
    axes[1].set_title('Candidate score map')
    axes[1].axis('off')

    axes[2].imshow(aux['edge_map'], cmap=cfg.cmap)
    axes[2].set_title('Edge map')
    axes[2].axis('off')

    axes[3].imshow(img, cmap='gray')
    axes[3].imshow(mask, cmap='Blues', alpha=0.25)
    axes[3].set_title('Candidates on image+mask')
    axes[3].axis('off')

    for i, c in enumerate(cands):
        cx, cy = c['cx'], c['cy']
        rect = plt.Rectangle(
            (c['bbox_x0'], c['bbox_y0']),
            c['bbox_x1'] - c['bbox_x0'] + 1,
            c['bbox_y1'] - c['bbox_y0'] + 1,
            edgecolor='lime', facecolor='none', linewidth=1.2
        )
        axes[3].add_patch(rect)
        axes[3].scatter([cx], [cy], s=18, c='red')
        axes[3].text(cx + 2, cy + 2, str(i), color='yellow', fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    if len(cands) >= 2 and 'line_x' in cands[0]:
        ys_line = np.linspace(0, img.shape[0] - 1, 100)
        # refit from selected chain for display
        ys_sel = np.array([c['cy'] for c in cands], dtype=np.float32)
        xs_sel = np.array([c['cx'] for c in cands], dtype=np.float32)
        if len(ys_sel) >= 2:
            p = np.polyfit(ys_sel, xs_sel, deg=1)
            xs_line = p[0] * ys_line + p[1]
            axes[3].plot(xs_line, ys_line, color='cyan', linewidth=1.2)


def draw_candidate_crop(
    img: np.ndarray,
    mask: np.ndarray,
    edge_map: np.ndarray,
    cand: Dict,
    out_path: Path,
    cfg: Cfg
):
    cx, cy = cand['cx'], cand['cy']

    crop_img = safe_crop(img, cx, cy, cfg.crop_size)
    crop_mask = safe_crop(mask, cx, cy, cfg.crop_size)
    crop_edge = safe_crop(edge_map, cx, cy, cfg.crop_size)

    center_map, ring_map = make_candidate_ring_maps(img.shape[0], img.shape[1], cx, cy, cfg)
    crop_center = safe_crop(center_map, cx, cy, cfg.crop_size)
    crop_ring = safe_crop(ring_map, cx, cy, cfg.crop_size)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=cfg.dpi)

    axes[0].imshow(crop_img, cmap='gray')
    axes[0].set_title('Local image')
    axes[0].axis('off')

    axes[1].imshow(crop_img, cmap='gray')
    axes[1].imshow(crop_mask, cmap='Blues', alpha=0.35)
    axes[1].set_title(f'Mask overlay ({cand["label"]})')
    axes[1].axis('off')

    axes[2].imshow(crop_edge, cmap=cfg.cmap)
    axes[2].set_title('Edge map')
    axes[2].axis('off')

    axes[3].imshow(crop_center, cmap='gray')
    axes[3].set_title('Center disk')
    axes[3].axis('off')

    axes[4].imshow(crop_ring, cmap=cfg.cmap)
    axes[4].set_title('Outer ring')
    axes[4].axis('off')

    txt = (
        f'idx?  score={cand["score"]:.3f} | '
        f'center={cand["center_mean"]:.3f} | '
        f'ring={cand["ring_mean"]:.3f} | '
        f'ring_edge={cand["ring_edge"]:.3f} | '
        f'mask_frac={cand["mask_frac"]:.3f}'
    )
    fig.suptitle(txt)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# =========================
# CSV
# =========================
def save_csv(rows: List[Dict], out_csv: Path):
    if not rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name'])
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def grow_region_from_seed(
    score_map: np.ndarray,
    seed_y: int,
    seed_x: int,
    rel_thr: float = 0.82,
    abs_thr: float = 0.55,
    max_radius: int = 20,
) -> np.ndarray:
    """
    Region growing on score_map from a seed point.

    A pixel is accepted if:
      score >= max(abs_thr, rel_thr * seed_score)
    and it stays within max_radius from seed.

    Returns a boolean mask of the grown region.
    """
    H, W = score_map.shape
    seed_score = float(score_map[seed_y, seed_x])
    thr = max(abs_thr, rel_thr * seed_score)

    visited = np.zeros((H, W), dtype=np.uint8)
    region = np.zeros((H, W), dtype=bool)

    stack = [(seed_y, seed_x)]
    visited[seed_y, seed_x] = 1

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    while stack:
        y, x = stack.pop()

        if (y - seed_y) * (y - seed_y) + (x - seed_x) * (x - seed_x) > max_radius * max_radius:
            continue

        if score_map[y, x] < thr:
            continue

        region[y, x] = True

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if not visited[ny, nx]:
                    visited[ny, nx] = 1
                    stack.append((ny, nx))

    return region


def mask_to_component(mask: np.ndarray) -> np.ndarray:
    """
    Convert a boolean mask to component coordinates array (N,2) as (y,x).
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([ys, xs], axis=1).astype(np.int32)

def component_shape_stats(comp: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute simple shape stats for a connected component.

    comp: (N,2) array of (y,x)

    Returns:
      bbox_w, bbox_h, bbox_aspect,
      pca_major, pca_minor, pca_aspect
    """
    if comp.shape[0] == 0:
        return {
            'bbox_w': 0.0,
            'bbox_h': 0.0,
            'bbox_aspect': 1e9,
            'pca_major': 0.0,
            'pca_minor': 0.0,
            'pca_aspect': 1e9,
        }

    ys = comp[:, 0].astype(np.float32)
    xs = comp[:, 1].astype(np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    bbox_w = float(x1 - x0 + 1.0)
    bbox_h = float(y1 - y0 + 1.0)
    bbox_aspect = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + eps)

    if len(xs) < 2:
        return {
            'bbox_w': bbox_w,
            'bbox_h': bbox_h,
            'bbox_aspect': bbox_aspect,
            'pca_major': 0.0,
            'pca_minor': 0.0,
            'pca_aspect': 1e9,
        }

    pts = np.stack([xs, ys], axis=1)
    pts = pts - pts.mean(axis=0, keepdims=True)

    cov = (pts.T @ pts) / max(len(pts) - 1, 1)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]  # descending

    major = float(np.sqrt(eigvals[0] + eps))
    minor = float(np.sqrt(eigvals[1] + eps)) if len(eigvals) > 1 else 0.0
    pca_aspect = major / (minor + eps)

    return {
        'bbox_w': bbox_w,
        'bbox_h': bbox_h,
        'bbox_aspect': bbox_aspect,
        'pca_major': major,
        'pca_minor': minor,
        'pca_aspect': pca_aspect,
    }

def generate_candidates(img: np.ndarray, mask: np.ndarray, cfg: Cfg) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Candidate generation by:
      1) disk-minus-ring score map
      2) local maxima detection
      3) region growing from each peak on score_map
      4) shape filtering by bbox aspect ratio + PCA aspect ratio
    """
    img_n = robust_normalize(img)

    # edge map
    _, _, edge_map = sobel_grad(img_n)

    # disk-ring response
    maps = disk_ring_score_map(
        img=img_n,
        edge_map=edge_map,
        r_disk=5,
        r_in=8,
        r_out=16,
        alpha_ring=1.00,
        beta_edge=0.35,
    )

    score_raw = maps['score_raw']
    score_map = robust_normalize(score_raw)

    valid_scores = score_map[maps['valid'] > 0]
    thr = float(np.quantile(valid_scores, cfg.bright_quantile))

    # local maxima on score map
    lmax = local_max_binary(score_map, radius=4)
    peak_mask = (score_map >= thr) & lmax

    ys, xs = np.where(peak_mask)
    raw_cands = []
    grown_union = np.zeros_like(peak_mask, dtype=np.float32)

    for sy, sx in zip(ys, xs):
        region = grow_region_from_seed(
            score_map=score_map,
            seed_y=int(sy),
            seed_x=int(sx),
            rel_thr=0.82,
            abs_thr=thr * 0.92,
            max_radius=18,
        )

        comp = mask_to_component(region)
        area = len(comp)
        if area < cfg.min_component_area or area > cfg.max_component_area:
            continue

        # -------- shape stats & filtering --------
        shp = component_shape_stats(comp, eps=cfg.eps)

        # basic size rejection
        if shp['bbox_w'] < 6 or shp['bbox_h'] < 6:
            continue

        # bbox aspect ratio: remove thin long components
        if shp['bbox_aspect'] > 2.2:
            continue

        # PCA aspect ratio: remove strongly elongated components
        if shp['pca_aspect'] > 2.8:
            continue
        # ----------------------------------------

        cand = component_to_candidate(comp, score_map, img_n, mask, edge_map, cfg)

        # attach shape stats for csv / debug
        cand['bbox_w'] = shp['bbox_w']
        cand['bbox_h'] = shp['bbox_h']
        cand['bbox_aspect'] = shp['bbox_aspect']
        cand['pca_major'] = shp['pca_major']
        cand['pca_minor'] = shp['pca_minor']
        cand['pca_aspect'] = shp['pca_aspect']

        # overwrite / enrich with disk-ring stats at grown-region center
        cx_i = int(round(cand['cx']))
        cy_i = int(round(cand['cy']))
        if 0 <= cy_i < img_n.shape[0] and 0 <= cx_i < img_n.shape[1]:
            cand['disk_mean'] = float(maps['disk_mean'][cy_i, cx_i])
            cand['ring_mean_diskring'] = float(maps['ring_mean'][cy_i, cx_i])
            cand['edge_disk_mean'] = float(maps['edge_disk_mean'][cy_i, cx_i])
            cand['score_diskring'] = float(score_map[cy_i, cx_i])
        else:
            cand['disk_mean'] = 0.0
            cand['ring_mean_diskring'] = 0.0
            cand['edge_disk_mean'] = 0.0
            cand['score_diskring'] = 0.0

        # final candidate score
        cand['score'] = (
            1.50 * cand['disk_mean']
            - 1.20 * cand['ring_mean_diskring']
            - 0.45 * cand['edge_disk_mean']
            + 0.20 * cand['center_mean']
            - 0.10 * cand['ring_mean']
        )

        raw_cands.append(cand)
        grown_union = np.maximum(grown_union, region.astype(np.float32))

    cands = nms_candidates(raw_cands, cfg.nms_radius)
    cands = sorted(cands, key=lambda d: d['score'], reverse=True)
    cands = cands[:cfg.max_candidates_per_image]

    aux = {
        'img_n': img_n,
        'score_map': score_map,
        'score_raw': score_raw,
        'disk_mean': maps['disk_mean'],
        'ring_mean_diskring': maps['ring_mean'],
        'edge_disk_mean': maps['edge_disk_mean'],
        'edge_map': edge_map,
        'binary': grown_union.astype(np.float32),
        'binary_peaks': peak_mask.astype(np.float32),
        'valid': maps['valid'],
    }
    return cands, aux
def select_spine_chain(
    cands: List[Dict],
    image_h: int,
    image_w: int,
    max_keep: int = 8,
    line_dist_thr: float = 18.0,
    max_dx_step: float = 18.0,
    max_dy_step: float = 45.0,
    min_dy_step: float = 6.0,
    score_weight_power: float = 1.5,
) -> List[Dict]:
    """
    Fit a spine main column x(y)=a*y+b from current candidates,
    then keep the most column-consistent chain.

    Args:
        cands: list of candidate dicts, each must contain:
               'cx', 'cy', 'score'
        image_h, image_w: image size
        max_keep: maximum number of candidates to keep in final chain
        line_dist_thr: max horizontal distance to fitted line
        max_dx_step: max x drift between adjacent chain nodes
        max_dy_step: max y gap between adjacent chain nodes
        min_dy_step: min y gap between adjacent chain nodes
        score_weight_power: weighting power for line fitting

    Returns:
        selected_chain: filtered candidates, sorted by y ascending
    """
    if len(cands) <= 1:
        return cands

    # -------------------------
    # Step 1: weighted line fit x = a*y + b
    # -------------------------
    ys = np.array([c['cy'] for c in cands], dtype=np.float32)
    xs = np.array([c['cx'] for c in cands], dtype=np.float32)
    ss = np.array([max(c['score'], 0.0) for c in cands], dtype=np.float32)

    # avoid zero weights
    ws = (ss - ss.min() + 1e-3) ** score_weight_power
    ws = ws / (ws.sum() + 1e-6)

    y_bar = float((ws * ys).sum())
    x_bar = float((ws * xs).sum())

    var_y = float((ws * (ys - y_bar) ** 2).sum())
    if var_y < 1e-6:
        a = 0.0
        b = x_bar
    else:
        cov_xy = float((ws * (ys - y_bar) * (xs - x_bar)).sum())
        a = cov_xy / (var_y + 1e-6)
        b = x_bar - a * y_bar

    # predicted x on line
    x_line = a * ys + b
    line_dist = np.abs(xs - x_line)

    # -------------------------
    # Step 2: keep line-consistent candidates
    # -------------------------
    line_inliers = []
    for i, c in enumerate(cands):
        c2 = dict(c)
        c2['line_x'] = float(x_line[i])
        c2['line_dist'] = float(line_dist[i])
        if line_dist[i] <= line_dist_thr:
            line_inliers.append(c2)

    # fallback: if too few survive, keep nearest-to-line candidates
    if len(line_inliers) < 3:
        order = np.argsort(line_dist)
        keep_n = min(max_keep, len(cands))
        line_inliers = []
        for idx in order[:keep_n]:
            c2 = dict(cands[idx])
            c2['line_x'] = float(x_line[idx])
            c2['line_dist'] = float(line_dist[idx])
            line_inliers.append(c2)

    # sort by y
    line_inliers = sorted(line_inliers, key=lambda d: d['cy'])

    # -------------------------
    # Step 3: dynamic programming for best vertical chain
    # -------------------------
    n = len(line_inliers)
    dp = np.full(n, -1e18, dtype=np.float32)
    prev = np.full(n, -1, dtype=np.int32)

    def node_score(c):
        # favor high local score + close to fitted line
        return (
            2.0 * float(c['score'])
            - 0.08 * float(c['line_dist'])
        )

    def trans_score(ci, cj):
        # ci -> cj, require y increasing
        dy = float(cj['cy'] - ci['cy'])
        dx = float(cj['cx'] - ci['cx'])
        if dy <= 0:
            return -1e18
        if dy < min_dy_step or dy > max_dy_step:
            return -1e18
        if abs(dx) > max_dx_step:
            return -1e18

        # prefer moderate dy and small dx
        return (
            -0.06 * abs(dx)
            -0.03 * abs(dy - 22.0)
        )

    for i in range(n):
        dp[i] = node_score(line_inliers[i])

    for j in range(n):
        for i in range(j):
            ts = trans_score(line_inliers[i], line_inliers[j])
            if ts < -1e17:
                continue
            cand_score = dp[i] + ts + node_score(line_inliers[j])
            if cand_score > dp[j]:
                dp[j] = cand_score
                prev[j] = i

    # traceback
    end_idx = int(np.argmax(dp))
    chain_idx = []
    cur = end_idx
    while cur != -1:
        chain_idx.append(cur)
        cur = int(prev[cur])
    chain_idx = chain_idx[::-1]

    chain = [line_inliers[i] for i in chain_idx]

    # if chain too long, keep top max_keep by y continuity order
    if len(chain) > max_keep:
        chain = chain[:max_keep]

    return chain
# =========================
# Main
# =========================
def main():
    cfg = Cfg()

    root = Path(cfg.root)
    img_root = root / cfg.img_dir
    mask_root = root / cfg.mask_dir
    out_root = Path(cfg.out_dir)

    ensure_dir(out_root)
    ensure_dir(out_root / 'overview')
    ensure_dir(out_root / 'crops')
    ensure_dir(out_root / 'score_maps')

    pairs = list_pairs(img_root, mask_root)[:20]
    print(f'[INFO] Found {len(pairs)} image-mask pairs.')

    all_rows = []

    for idx, (img_path, mask_path) in enumerate(pairs):
        img = load_gray(img_path)
        mask = load_gray(mask_path)
        mask = (mask > 0.5).astype(np.float32)

        cands, aux = generate_candidates(img, mask, cfg)

        cands = select_spine_chain(
            cands=cands,
            image_h=img.shape[0],
            image_w=img.shape[1],
            max_keep=8,
            line_dist_thr=18.0,
            max_dx_step=18.0,
            max_dy_step=45.0,
            min_dy_step=6.0,
            score_weight_power=1.5,
        )

        stem = img_path.stem
        print(f'[{idx+1}/{len(pairs)}] {stem}: {len(cands)} candidates')

        # save score maps
        score_fig = plt.figure(figsize=(8, 3), dpi=cfg.dpi)
        ax1 = score_fig.add_subplot(1, 3, 1)
        ax2 = score_fig.add_subplot(1, 3, 2)
        ax3 = score_fig.add_subplot(1, 3, 3)

        ax1.imshow(aux['img_n'], cmap='gray')
        ax1.set_title('norm image')
        ax1.axis('off')

        ax2.imshow(aux['score_map'], cmap=cfg.cmap)
        ax2.set_title('score map')
        ax2.axis('off')

        ax3.imshow(aux['binary'], cmap='gray')
        ax3.set_title('binary peaks')
        ax3.axis('off')

        score_fig.tight_layout()
        score_fig.savefig(out_root / 'score_maps' / f'{stem}_scoremap.png', bbox_inches='tight')
        plt.close(score_fig)

        # overview figure
        draw_overview(
            img=aux['img_n'],
            mask=mask,
            aux=aux,
            cands=cands,
            out_path=out_root / 'overview' / f'{stem}_overview.png',
            title=f'{stem} | {len(cands)} candidates',
            cfg=cfg
        )

        # per-candidate crops
        for cid, cand in enumerate(cands):
            crop_name = f'{stem}_cand{cid:02d}.png'
            draw_candidate_crop(
                img=aux['img_n'],
                mask=mask,
                edge_map=aux['edge_map'],
                cand=cand,
                out_path=out_root / 'crops' / crop_name,
                cfg=cfg
            )

            row = {
                'image_name': img_path.name,
                'mask_name': mask_path.name,
                'candidate_id': cid,
                'cx': round(cand['cx'], 3),
                'cy': round(cand['cy'], 3),
                'area': cand['area'],
                'bbox_x0': cand['bbox_x0'],
                'bbox_y0': cand['bbox_y0'],
                'bbox_x1': cand['bbox_x1'],
                'bbox_y1': cand['bbox_y1'],
                'score': round(cand['score'], 6),
                'comp_score': round(cand['comp_score'], 6),
                'comp_bright': round(cand['comp_bright'], 6),
                'center_mean': round(cand['center_mean'], 6),
                'ring_mean': round(cand['ring_mean'], 6),
                'ring_edge': round(cand['ring_edge'], 6),
                'mask_frac': round(cand['mask_frac'], 6),
                'label': cand['label'],
            }
            all_rows.append(row)

    save_csv(all_rows, out_root / 'candidates.csv')
    print(f'[DONE] Results saved to: {out_root}')


if __name__ == '__main__':
    main()