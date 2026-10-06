# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
from tqdm import tqdm

# ============================================================
# 02. M_Pred Pred_Mask -> R_Pred Operation-binarization by Weighted KDE-GMM (Bin-OTP)
# ------------------------------------------------------------
# This script is model-agnostic. Any segmentation model can be used upstream
# as long as it outputs full-size binary Pred_Mask files.
#
# Required inputs:
#   IMAGE_DIR      full-size RGB image folder, e.g. 01 output/Image
#   PRED_MASK_DIR  M_Pred = inference-original/predicted ROI, e.g. Pred_Mask
# Optional:
#   GT_MASK_DIR    M_GT = GT-original, used only for six-panel visualization and diagnostics
#
# Main outputs:
#   03_final_mask/        R_Pred masks. Both parent.png and parent__final_mask.png are saved.
#   all_output_info.csv   object-level thresholds, method branch, gray statistics
#   image_level_summary.csv
#   04_intermediate_masks/ ring/object/method/threshold maps for later analysis
# ============================================================

IMAGE_DIR = Path(r"")  # TODO: set path
PRED_MASK_DIR = Path(r"")  # TODO: set path
GT_MASK_DIR = Path(r"")  # TODO: set path
OUTPUT_ROOT = Path(r"")  # TODO: set path
RUN_NAME = "MitB5_wkdegmm_from_predmask-1"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_THRESHOLD = 127
NUM_RANDOM_PARENTS = -1
RANDOM_SEED = 42
RESIZE_MASKS_TO_IMAGE = False
APPLY_PRED_OPENING = False

# Object extraction
PRED_OPEN_KSIZE = 3
SEED_ERODE_KSIZE = 5
MIN_OBJECT_AREA = 40
CORE_ERODE_KSIZE = 5
MIN_CORE_PIXELS = 25

# Ring/context
RING_CONTEXT_PAD = 48
RING_INNER_KSIZE = 5
RING_OUTER_CANDIDATES = [21, 31, 41]
MIN_RING_PIXELS = 30
MIN_CONTEXT_BG_PIXELS = 80

# Weighted KDE-GMM decision logic
KDE_BANDWIDTH_SIGMA = 15.0
KDE_PEAK_PROMINENCE = 0.015
OBJECT_UNIFORM_P95_P05_MAX = 80.0
EM_MAX_ITER = 80
EM_TOL = 1e-6
VAR_FLOOR = 9.0
GMM_RESCUE_MU_GAP_MIN = 45.0
GMM_RESCUE_OBJ_RANGE_MIN = 26.0
GMM_RESCUE_PI_MIN = 0.03

# Fallback thresholds
RING_CONTRAST_STRICT_MAX = 100.0
FIXED_THRESHOLD_STRICT = 205.0
FIXED_THRESHOLD_LOOSE = 165.0
BIN_OTP_TRANSITION_CENTER = RING_CONTRAST_STRICT_MAX
BIN_OTP_TRANSITION_TAU = 12.0

# Guarded local fallback for over-strict Bin-OTP thresholds.
# This is a fixed decision-tree branch, not a swept parameter.
ENABLE_GUARDED_LOCAL_FALLBACK = True
FALLBACK_GUARD_RING_P90_MARGIN = 35.0
FALLBACK_GUARD_DARK_RING_MARGIN = 50.0
FALLBACK_GUARD_OBJ_P50_OFFSET = 20.0
FALLBACK_GUARD_OBJ_P90_EPS = 5.0
FALLBACK_GUARD_MIN_THRESHOLD = 110.0
FALLBACK_GUARD_MAX_THRESHOLD = FIXED_THRESHOLD_LOOSE
FALLBACK_GUARD_HIGH_CONTRAST_MIN = 100.0
FALLBACK_GUARD_DARK_RING_P90_MAX = 90.0
FALLBACK_GUARD_MAX_KDE_PEAKS = 1

# Shadow compensation
ENABLE_SHADOW_COMPENSATION = True
ILLUM_INPAINT_RADIUS = 5
ILLUM_BLUR_SIGMA = 21.0
SHADOW_OBJ_ILLUM_RANGE_MIN = 18.0
SHADOW_OBJ_GRAD_P95_MIN = 1.8
SHADOW_RING_IQR_MIN = 12.0

# Visualization
CURVES_PER_PAGE = 12
# Windows long-path friendly settings.
# KDE/GMM curve saving is useful but should never block R_Pred final-mask export.
SAVE_KDE_CURVES = True
CURVE_FILENAME_SHORT = True
STRIP_TITLE_H = 56
STRIP_LABEL_H = 48
STRIP_BG_COLOR = (255, 255, 255)
STRIP_TEXT_COLOR = (0, 0, 0)

METHOD_ID = {
    "weighted_kdegmm": 1,
    "fixed_threshold_strict": 3,
    "fixed_threshold_loose": 4,
    "fixed_threshold_strict_unreliable_bg": 5,
    "bin_otp_smooth_strict_loose": 6,
    "guarded_local_fallback": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted KDE-GMM binarization from precomputed predicted masks")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--pred-mask-dir", type=str, default=str(PRED_MASK_DIR))
    parser.add_argument("--gt-mask-dir", type=str, default=str(GT_MASK_DIR), help="Optional GT mask folder. Empty string disables GT panel.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--num-random-parents", type=int, default=NUM_RANDOM_PARENTS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--resize-masks-to-image", action="store_true", default=RESIZE_MASKS_TO_IMAGE)
    parser.add_argument("--apply-pred-opening", action="store_true", default=APPLY_PRED_OPENING)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files(folder: Path) -> List[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask01(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > MASK_THRESHOLD).astype(np.uint8)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def save_mask(mask01: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray((mask01.astype(np.uint8) * 255)).save(path)


def save_u16(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    arr_u16 = np.clip(arr, 0, 65535).astype(np.uint16)
    ok = cv2.imwrite(str(path), arr_u16)
    if not ok:
        raise IOError(f"Failed to write uint16 image: {path}")


def save_npy(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    np.save(path, arr)


def normalize_stem(stem: str) -> str:
    suffixes = [
        "__final_mask", "_final_mask", "__binary", "_binary", "__final", "_final",
        "__pred_mask", "_pred_mask", "__Pred_Mask", "_Pred_Mask", "__mask", "_mask", "__pred", "_pred",
        "_overlay",
    ]
    out = stem
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if out.endswith(suf):
                out = out[:-len(suf)]
                changed = True
    return out


def find_by_stem(folder: Path, stem: str) -> Optional[Path]:
    if folder is None or not folder.exists():
        return None
    candidates = []
    stems = [stem, normalize_stem(stem)]
    stems += [f"{normalize_stem(stem)}__final_mask", f"{normalize_stem(stem)}_final_mask"]
    seen = set()
    for s in stems:
        if s in seen:
            continue
        seen.add(s)
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidates.append(folder / f"{s}{ext}")
    for p in candidates:
        if p.exists():
            return p
    norm = normalize_stem(stem)
    for p in list_files(folder):
        if normalize_stem(p.stem) == norm:
            return p
    return None


def gray_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)


def smooth_binary(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.uint8)


def erode_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def dilate_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def make_ring(mask: np.ndarray, inner_ksize: int, outer_ksize: int) -> np.ndarray:
    inner_ksize = max(1, int(inner_ksize))
    outer_ksize = max(inner_ksize + 2, int(outer_ksize))
    if inner_ksize % 2 == 0:
        inner_ksize += 1
    if outer_ksize % 2 == 0:
        outer_ksize += 1
    kin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_ksize, inner_ksize))
    kout = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_ksize, outer_ksize))
    din = cv2.dilate(mask.astype(np.uint8), kin, iterations=1)
    dout = cv2.dilate(mask.astype(np.uint8), kout, iterations=1)
    return ((dout > 0) & (din == 0)).astype(np.uint8)


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask.size and (mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()))


def safe_percentile(arr: np.ndarray, q: float, default: float = np.nan) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def extract_objects_with_eroded_seed(full_mask: np.ndarray, erode_ksize: int, min_area: int) -> List[Dict]:
    full_mask = full_mask.astype(np.uint8)
    seed_mask = erode_mask(full_mask, erode_ksize)
    n_full, labels_full, _, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
    n_seed, labels_seed, stats_seed, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
    objects, used_full_labels, obj_id = [], set(), 0
    for sid in range(1, n_seed):
        _, _, _, _, sarea = stats_seed[sid]
        if int(sarea) < min_area:
            continue
        seed_comp = labels_seed == sid
        overlapped_full = [int(v) for v in np.unique(labels_full[seed_comp]) if int(v) != 0]
        overlapped_full = [v for v in overlapped_full if v not in used_full_labels]
        if not overlapped_full:
            continue
        full_comp = np.isin(labels_full, overlapped_full).astype(np.uint8)
        ys, xs = np.where(full_comp > 0)
        if xs.size == 0:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        area = int(full_comp.sum())
        if area < min_area:
            continue
        obj_id += 1
        objects.append({
            "object_id": obj_id,
            "bbox_x": x0, "bbox_y": y0, "bbox_w": x1 - x0 + 1, "bbox_h": y1 - y0 + 1,
            "area": area,
            "full_mask_crop": full_comp[y0:y1+1, x0:x1+1].astype(np.uint8),
            "seed_mask_crop": seed_comp[y0:y1+1, x0:x1+1].astype(np.uint8),
        })
        used_full_labels.update(overlapped_full)
    return objects


def build_local_object_context(big_rgb: np.ndarray, x: int, y: int, w: int, h: int, objmask: np.ndarray, pad: int) -> Dict:
    H, W = big_rgb.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    local_rgb = big_rgb[y0:y1, x0:x1]
    local_gray = gray_u8(local_rgb)
    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    ox, oy = x - x0, y - y0
    local_mask[oy:oy+h, ox:ox+w] = objmask.astype(np.uint8)
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "local_rgb": local_rgb, "local_gray": local_gray, "local_mask": local_mask,
        "ox": ox, "oy": oy,
        "context_truncated_by_image_border": int(x0 == 0 or y0 == 0 or x1 == W or y1 == H),
        "object_touches_image_border": int(x == 0 or y == 0 or (x + w) == W or (y + h) == H),
    }


def select_background_region(local_mask: np.ndarray, local_gray: np.ndarray) -> Dict:
    outer_candidates = sorted({int(v) for v in RING_OUTER_CANDIDATES if int(v) > RING_INNER_KSIZE})
    if not outer_candidates:
        outer_candidates = [RING_INNER_KSIZE + 16]
    ring_viz_local = make_ring(local_mask, RING_INNER_KSIZE, outer_candidates[-1])
    for outer_ksize in outer_candidates:
        ring = make_ring(local_mask, RING_INNER_KSIZE, outer_ksize)
        vals = local_gray[ring > 0]
        if vals.size >= MIN_RING_PIXELS:
            return {
                "bg_mask_local": ring, "bg_source": "ring", "bg_pixels_used": int(vals.size),
                "ring_pixels": int(ring.sum()), "ring_touch_context_border": int(touches_border(ring)),
                "selected_outer_ksize": int(outer_ksize), "ring_viz_local": ring_viz_local,
            }
    exclusion = dilate_mask(local_mask, outer_candidates[-1])
    context_bg = (exclusion == 0).astype(np.uint8)
    vals = local_gray[context_bg > 0]
    if vals.size >= MIN_CONTEXT_BG_PIXELS:
        return {
            "bg_mask_local": context_bg, "bg_source": "context_bg", "bg_pixels_used": int(vals.size),
            "ring_pixels": int(ring_viz_local.sum()), "ring_touch_context_border": int(touches_border(ring_viz_local)),
            "selected_outer_ksize": int(outer_candidates[-1]), "ring_viz_local": ring_viz_local,
        }
    return {
        "bg_mask_local": None, "bg_source": "unreliable", "bg_pixels_used": 0,
        "ring_pixels": int(ring_viz_local.sum()), "ring_touch_context_border": int(touches_border(ring_viz_local)),
        "selected_outer_ksize": int(outer_candidates[-1]), "ring_viz_local": ring_viz_local,
    }


def estimate_local_illumination(local_gray: np.ndarray, local_mask: np.ndarray) -> Dict[str, np.ndarray]:
    src = local_gray.astype(np.uint8)
    mask_u8 = (local_mask > 0).astype(np.uint8) * 255
    filled = cv2.inpaint(src, mask_u8, float(max(1, ILLUM_INPAINT_RADIUS)), cv2.INPAINT_NS) if mask_u8.any() else src.copy()
    illum = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), sigmaX=ILLUM_BLUR_SIGMA, sigmaY=ILLUM_BLUR_SIGMA)
    gx = cv2.Sobel(illum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(illum, cv2.CV_32F, 0, 1, ksize=3)
    return {"filled_gray": filled, "illumination": illum, "illum_grad_mag": np.sqrt(gx ** 2 + gy ** 2)}


def detect_shadow_crossing_and_compensate(local_gray: np.ndarray, local_mask: np.ndarray, ring_viz_local: np.ndarray, bg_mask_local: Optional[np.ndarray]) -> Dict:
    est = estimate_local_illumination(local_gray, local_mask)
    illum, grad = est["illumination"], est["illum_grad_mag"]
    obj_vals, obj_grad_vals = illum[local_mask > 0], grad[local_mask > 0]
    ring_probe = bg_mask_local if bg_mask_local is not None and np.any(bg_mask_local > 0) else ring_viz_local
    ring_vals = illum[ring_probe > 0] if ring_probe is not None and np.any(ring_probe > 0) else np.array([], dtype=np.float32)
    obj_range = safe_percentile(obj_vals, 90, 0.0) - safe_percentile(obj_vals, 10, 0.0)
    obj_grad_p95 = safe_percentile(obj_grad_vals, 95, 0.0)
    ring_iqr = safe_percentile(ring_vals, 75, 0.0) - safe_percentile(ring_vals, 25, 0.0)
    shadow_by_grad = bool(obj_range >= SHADOW_OBJ_ILLUM_RANGE_MIN and obj_grad_p95 >= SHADOW_OBJ_GRAD_P95_MIN)
    shadow_by_ring = bool(ring_iqr >= SHADOW_RING_IQR_MIN)
    shadow = bool(ENABLE_SHADOW_COMPENSATION and (shadow_by_grad or shadow_by_ring))
    if bg_mask_local is not None and np.any(bg_mask_local > 0):
        ref_vals = illum[bg_mask_local > 0]
    elif ring_probe is not None and np.any(ring_probe > 0):
        ref_vals = illum[ring_probe > 0]
    else:
        ref_vals = illum.reshape(-1)
    illum_ref = float(np.median(ref_vals)) if ref_vals.size else float(np.median(illum))
    comp = np.clip(local_gray.astype(np.float32) - illum + illum_ref, 0, 255).astype(np.uint8)
    reasons = []
    if shadow_by_grad:
        reasons.append("strong_object_illum_gradient")
    if shadow_by_ring:
        reasons.append("ring_brightness_inconsistency")
    return {
        "compensated_local_gray": comp,
        "shadow_crossing_detected": int(shadow),
        "shadow_reason": "|".join(reasons) if reasons else "none",
        "shadow_by_grad": int(shadow_by_grad),
        "shadow_by_ring": int(shadow_by_ring),
        "obj_illum_range_p90_p10": float(obj_range),
        "obj_illum_grad_p95": float(obj_grad_p95),
        "ring_illum_iqr": float(ring_iqr),
        "illumination_ref_value": float(illum_ref),
    }


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0, 1))
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx]
    cdf = np.cumsum(ws)
    if cdf[-1] <= 0:
        return float(xs[len(xs) // 2])
    cdf = cdf / cdf[-1]
    return float(xs[np.searchsorted(cdf, q, side="left")])


def smooth_histogram_kde(values_u8: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    hist = np.bincount(values_u8.astype(np.int32), minlength=256).astype(np.float64)
    x = np.arange(256, dtype=np.float64)
    radius = max(1, int(math.ceil(4 * sigma)))
    kx = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kx / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(hist, kernel, mode="same")
    density = density / max(density.sum(), 1e-12)
    return x, density


def gaussian_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    denom = np.sqrt(2.0 * np.pi * max(var, VAR_FLOOR))
    return np.exp(-0.5 * ((x - mu) ** 2) / max(var, VAR_FLOOR)) / max(denom, 1e-12)


def weighted_gmm_fit_1d(x: np.ndarray, w: np.ndarray) -> Dict:
    w = np.clip(np.asarray(w, dtype=np.float64), 0, None)
    w = w / max(w.sum(), 1e-12)
    mu = np.array([weighted_percentile(x, w, 0.25), weighted_percentile(x, w, 0.75)], dtype=np.float64)
    var_global = max(float(np.sum(w * (x - np.sum(w * x)) ** 2)), VAR_FLOOR)
    var = np.array([var_global, var_global], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)
    prev_ll = None
    for it in range(EM_MAX_ITER):
        pdfs = np.stack([gaussian_pdf(x, mu[k], var[k]) for k in range(2)], axis=1)
        weighted_pdf = pdfs * pi[None, :]
        resp = weighted_pdf / np.clip(weighted_pdf.sum(axis=1, keepdims=True), 1e-12, None)
        Nk = np.sum(w[:, None] * resp, axis=0)
        pi = Nk / max(Nk.sum(), 1e-12)
        mu = np.sum((w[:, None] * resp) * x[:, None], axis=0) / np.clip(Nk, 1e-12, None)
        var = np.sum((w[:, None] * resp) * (x[:, None] - mu[None, :]) ** 2, axis=0) / np.clip(Nk, 1e-12, None)
        var = np.maximum(var, VAR_FLOOR)
        mix_pdf = np.clip(np.sum(pdfs * pi[None, :], axis=1), 1e-12, None)
        ll = float(np.sum(w * np.log(mix_pdf)))
        if prev_ll is not None and abs(ll - prev_ll) < EM_TOL:
            return {"mu": mu, "var": var, "pi": pi, "iters": it + 1, "ll": ll}
        prev_ll = ll
    return {"mu": mu, "var": var, "pi": pi, "iters": EM_MAX_ITER, "ll": float(prev_ll or 0.0)}


def weighted_kdegmm_threshold_from_core(values_u8: np.ndarray) -> Dict:
    x_grid, kde = smooth_histogram_kde(values_u8, KDE_BANDWIDTH_SIGMA)
    kde_norm = kde / max(kde.sum(), 1e-12)
    kde_display = kde / max(kde.max(), 1e-12)
    peaks, _ = find_peaks(kde_display, prominence=KDE_PEAK_PROMINENCE)
    fit = weighted_gmm_fit_1d(x_grid, kde_norm)
    mu, var, pi = np.asarray(fit["mu"]), np.asarray(fit["var"]), np.asarray(fit["pi"])
    order = np.argsort(mu)
    dark_idx, bright_idx = int(order[0]), int(order[1])
    mu_dark, mu_bright = float(mu[dark_idx]), float(mu[bright_idx])
    var_dark, var_bright = float(var[dark_idx]), float(var[bright_idx])
    pi_dark, pi_bright = float(pi[dark_idx]), float(pi[bright_idx])
    pdf_dark = pi_dark * gaussian_pdf(x_grid, mu_dark, var_dark)
    pdf_bright = pi_bright * gaussian_pdf(x_grid, mu_bright, var_bright)
    mix_pdf = pdf_dark + pdf_bright
    left, right = int(max(0, math.floor(mu_dark))), int(min(255, math.ceil(mu_bright)))
    valley_idx = int(round((mu_dark + mu_bright) / 2.0)) if right <= left else left + int(np.argmin(kde[left:right + 1]))
    diff = pdf_bright - pdf_dark
    sign_change_idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if sign_change_idx.size > 0:
        thr_idx = int(sign_change_idx[np.argmin(np.abs(sign_change_idx - valley_idx))])
        threshold_x, source = float(x_grid[thr_idx]), "posterior_intersection"
    else:
        threshold_x, source = float(x_grid[valley_idx]), "kde_valley"
    return {
        "x_grid": x_grid, "kde": kde, "pdf_dark": pdf_dark, "pdf_bright": pdf_bright, "mix_pdf": mix_pdf,
        "threshold_x": threshold_x, "threshold_source": source,
        "num_peaks": int(len(peaks)), "peak_positions": [float(x_grid[p]) for p in peaks[:8]],
        "mu_dark": mu_dark, "mu_bright": mu_bright,
        "pi_dark": pi_dark, "pi_bright": pi_bright,
        "var_dark": var_dark, "var_bright": var_bright,
        "em_iters": int(fit["iters"]), "em_ll": float(fit["ll"]), "valley_x": float(x_grid[valley_idx]),
    }


def otsu_threshold_from_values(values: np.ndarray, default: float = np.nan) -> float:
    values = np.asarray(values)
    if values.size < 2:
        return float(default)
    v = np.clip(values, 0, 255).astype(np.uint8)
    if int(v.min()) == int(v.max()):
        return float(v.min())
    thr, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)



def bin_otp_smooth_strict_loose_threshold(
    robust_contrast: float,
    strict_thr: float = FIXED_THRESHOLD_STRICT,
    loose_thr: float = FIXED_THRESHOLD_LOOSE,
) -> Tuple[float, float]:
    if not np.isfinite(robust_contrast):
        return float(strict_thr), 0.0
    z = (float(robust_contrast) - BIN_OTP_TRANSITION_CENTER) / max(BIN_OTP_TRANSITION_TAU, 1e-6)
    z = float(np.clip(z, -60.0, 60.0))
    loose_weight = 1.0 / (1.0 + math.exp(-z))
    thr = (1.0 - loose_weight) * float(strict_thr) + loose_weight * float(loose_thr)
    return float(np.clip(thr, 0, 255)), float(loose_weight)


def guarded_fallback_threshold(
    base_thr: float,
    use_shadow: bool,
    use_wk: bool,
    bg_reliable: bool,
    kde_num_peaks: int,
    robust_contrast: float,
    obj_p50: float,
    obj_p90: float,
    ring_p90: float,
) -> Dict:
    """Constrain fixed fallback threshold using local distribution guards.

    This is intentionally a single guard after KDE-GMM rejection. It covers two
    interpretable cases: shadow-compensated fallback over-strictness, and
    dark-ring/high-contrast single-peak objects where fixed thresholds are too
    conservative but the local background is clearly separated.
    """
    base_thr = float(base_thr)
    out = {
        "threshold": base_thr,
        "method": "bin_otp_smooth_strict_loose",
        "applied": False,
        "reason": "not_applicable",
        "ring_guard_x": np.nan,
        "object_guard_x": np.nan,
        "upper_guard_x": np.nan,
    }
    if not ENABLE_GUARDED_LOCAL_FALLBACK:
        out["reason"] = "disabled"
        return out
    if use_wk:
        out["reason"] = "kdegmm_selected"
        return out
    if not bg_reliable:
        out["reason"] = "unreliable_background"
        return out
    if not all(np.isfinite(v) for v in [base_thr, robust_contrast, obj_p50, obj_p90, ring_p90]):
        out["reason"] = "missing_percentile"
        return out

    object_supported_max = float(obj_p90) - float(FALLBACK_GUARD_OBJ_P90_EPS)
    shadow_overstrict = bool(use_shadow and base_thr > object_supported_max)
    dark_ring_high_contrast = bool(
        int(kde_num_peaks) <= int(FALLBACK_GUARD_MAX_KDE_PEAKS)
        and float(robust_contrast) >= float(FALLBACK_GUARD_HIGH_CONTRAST_MIN)
        and float(ring_p90) <= float(FALLBACK_GUARD_DARK_RING_P90_MAX)
    )
    if not (shadow_overstrict or dark_ring_high_contrast):
        out["reason"] = "guard_conditions_not_met"
        return out

    if dark_ring_high_contrast:
        ring_guard = float(ring_p90) + float(FALLBACK_GUARD_DARK_RING_MARGIN)
        reason = "dark_ring_high_contrast_single_peak"
    else:
        ring_guard = float(ring_p90) + float(FALLBACK_GUARD_RING_P90_MARGIN)
        reason = "shadow_base_exceeds_object_support"

    object_guard = float(obj_p50) - float(FALLBACK_GUARD_OBJ_P50_OFFSET)
    candidate = max(ring_guard, object_guard, float(FALLBACK_GUARD_MIN_THRESHOLD))
    upper_guard = min(base_thr, object_supported_max, float(FALLBACK_GUARD_MAX_THRESHOLD))

    out["ring_guard_x"] = float(ring_guard)
    out["object_guard_x"] = float(object_guard)
    out["upper_guard_x"] = float(upper_guard)

    if candidate > upper_guard:
        out["reason"] = "no_safe_guarded_fallback"
        return out
    if candidate >= base_thr:
        out["reason"] = "cap_not_lower_than_base"
        return out

    out["threshold"] = float(np.clip(candidate, 0, 255))
    out["method"] = "guarded_local_fallback"
    out["applied"] = True
    out["reason"] = reason
    return out


def classify_by_threshold(gray_crop: np.ndarray, objmask: np.ndarray, thr: float) -> np.ndarray:
    out = np.zeros_like(objmask, dtype=np.uint8)
    inside = objmask > 0
    out[inside] = (gray_crop[inside].astype(np.float32) >= float(thr)).astype(np.uint8)
    return out


def color_mask(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=(0, 0, 255)) -> np.ndarray:
    vis = rgb.copy().astype(np.float32)
    col = np.zeros_like(vis)
    col[..., 0], col[..., 1], col[..., 2] = color
    keep = mask.astype(bool)
    vis[keep] = vis[keep] * (1.0 - alpha) + col[keep] * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)


def overlay_ring_on_origin(rgb: np.ndarray, ring_mask: np.ndarray, object_mask: Optional[np.ndarray] = None) -> np.ndarray:
    vis = rgb.copy().astype(np.float32)
    if ring_mask is not None:
        keep = ring_mask.astype(bool)
        green = np.zeros_like(vis); green[..., 1] = 255
        vis[keep] = vis[keep] * 0.45 + green[keep] * 0.55
    if object_mask is not None:
        edge = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        keep = edge.astype(bool)
        red = np.zeros_like(vis); red[..., 0] = 255
        vis[keep] = vis[keep] * 0.25 + red[keep] * 0.75
    return np.clip(vis, 0, 255).astype(np.uint8)


def annotate_object_ids(rgb: np.ndarray, objects: List[Dict]) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for obj in objects:
        x = int(obj["bbox_x"] + obj["bbox_w"] / 2)
        y = int(obj["bbox_y"] + obj["bbox_h"] / 2)
        r = 10
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 0), outline=(0, 0, 0))
        draw.text((x - 4, y - 7), str(obj["object_id"]), fill=(0, 0, 0))
    return np.array(img)


def make_native_strip(parent_name: str, rgb: np.ndarray, gt: np.ndarray, ring_overlay: np.ndarray, pred_overlay: np.ndarray, final_mask: np.ndarray, final_overlay_ids: np.ndarray, out_path: Path) -> None:
    panels = [(rgb, "Original"), (color_mask(gt), "A1 GT-mask"), (ring_overlay, "Ring-on-Origin"), (pred_overlay, "A3 Pred-overlay"), (color_mask(final_mask), "A4 Final-mask"), (final_overlay_ids, "A4 Final-overlay + IDs")]
    h, w = rgb.shape[:2]
    canvas = Image.new("RGB", (w * len(panels), STRIP_TITLE_H + STRIP_LABEL_H + h), STRIP_BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 16), parent_name, fill=STRIP_TEXT_COLOR)
    for i, (im, label) in enumerate(panels):
        x0 = i * w
        draw.text((x0 + 12, STRIP_TITLE_H + 14), label, fill=STRIP_TEXT_COLOR)
        canvas.paste(Image.fromarray(im), (x0, STRIP_TITLE_H + STRIP_LABEL_H))
    canvas.save(out_path, format="PNG")


def plot_single_curve(ax, rec: Dict) -> None:
    x = rec["x_grid"]
    kde = rec["kde"]
    mix = rec["mix_pdf"]
    ax.fill_between(x, kde / max(kde.max(), 1e-12), alpha=0.25, label="KDE")
    ax.plot(x, rec["pdf_dark"] / max(mix.max(), 1e-12), linestyle="--", label="GMM-dark")
    ax.plot(x, rec["pdf_bright"] / max(mix.max(), 1e-12), linestyle="--", label="GMM-bright")
    ax.plot(x, mix / max(mix.max(), 1e-12), linewidth=2.0, label="GMM-mix")
    ax.axvline(float(rec["final_threshold_x"]), linestyle="-.", linewidth=2.0, label=f"final_thr={rec['final_threshold_x']:.1f}")
    ax.set_xlim(0, 255); ax.set_ylim(0, 1.05)
    ax.set_title(f"obj {rec['object_id']} | {rec['final_method']} | peaks={rec['num_peaks']} | final_thr={rec['final_threshold_x']:.1f}")
    ax.legend(fontsize=8, loc="upper right")


def save_parent_curves(parent: str, curve_records: List[Dict], out_dir: Path) -> List[Dict]:
    """Save KDE/GMM curves safely.

    Returns warning records. Curve saving is deliberately non-fatal, because the core
    output of this script is the R_Pred final mask and the CSV tables.
    """
    warnings = []
    if (not SAVE_KDE_CURVES) or (not curve_records):
        return warnings

    ensure_dir(out_dir)

    for rec in sorted(curve_records, key=lambda d: int(d["object_id"])):
        obj_id = int(rec["object_id"])
        out_path = out_dir / (f"obj_{obj_id:04d}.png" if CURVE_FILENAME_SHORT else f"{parent}__obj_{obj_id:04d}__kde_curve.png")

        fig = None
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3.8))
            plot_single_curve(ax, rec)
            fig.tight_layout()
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
        except Exception as e:
            warnings.append({
                "parent": parent,
                "object_id": obj_id,
                "output_path": str(out_path),
                "reason": repr(e),
            })
            print(f"[WARN] curve saving failed for {parent} obj {obj_id}: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    return warnings

def process_parent(parent: str, rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray, dirs: Dict[str, Path], all_rows: List[Dict], image_rows: List[Dict]) -> None:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    objects = extract_objects_with_eroded_seed(pred, SEED_ERODE_KSIZE, MIN_OBJECT_AREA)
    final = np.zeros_like(pred, dtype=np.uint8)
    processed_object_count = 0

    for obj in objects:
        x, y, w, h = int(obj["bbox_x"]), int(obj["bbox_y"]), int(obj["bbox_w"]), int(obj["bbox_h"])
        objmask = obj["full_mask_crop"].astype(np.uint8)
        gray_raw = gray_u8(rgb[y:y+h, x:x+w])
        ctx = build_local_object_context(rgb, x, y, w, h, objmask, RING_CONTEXT_PAD)
        bg_sel = select_background_region(ctx["local_mask"], ctx["local_gray"])
        ring_viz_local = bg_sel["ring_viz_local"]
        bg_reliable = int(bg_sel["bg_source"] != "unreliable")
        bg_mask_local = bg_sel["bg_mask_local"] if bg_reliable else None
        shadow = detect_shadow_crossing_and_compensate(ctx["local_gray"], ctx["local_mask"], ring_viz_local, bg_mask_local)
        use_shadow = bool(shadow["shadow_crossing_detected"])
        analysis_local_gray = shadow["compensated_local_gray"] if use_shadow else ctx["local_gray"]
        analysis_gray_source = "shadow_compensated" if use_shadow else "raw"
        analysis_gray_crop = analysis_local_gray[ctx["oy"]:ctx["oy"]+h, ctx["ox"]:ctx["ox"]+w]
        core = erode_mask(objmask, CORE_ERODE_KSIZE)
        core_pixels = int(core.sum())
        if core_pixels < MIN_CORE_PIXELS:
            continue
        obj_vals = analysis_gray_crop[objmask > 0].astype(np.float64)
        core_vals = analysis_gray_crop[core > 0].astype(np.uint8)
        if obj_vals.size == 0 or core_vals.size == 0:
            continue
        raw_obj_vals = gray_raw[objmask > 0].astype(np.float64)
        obj_p05, obj_p50, obj_p90, obj_p95 = [float(np.percentile(obj_vals, q)) for q in [5, 50, 90, 95]]
        obj_range = obj_p95 - obj_p05
        if bg_reliable:
            bg_vals = analysis_local_gray[bg_mask_local > 0].astype(np.float64)
            ring_mean = float(bg_vals.mean())
            ring_p50 = float(np.percentile(bg_vals, 50))
            ring_p90 = float(np.percentile(bg_vals, 90))
            mean_delta = float(obj_vals.mean() - ring_mean)
            robust_contrast = float(obj_p90 - ring_p50)
        else:
            ring_mean = ring_p50 = ring_p90 = mean_delta = robust_contrast = np.nan

        wk = weighted_kdegmm_threshold_from_core(core_vals)
        mu_gap = float(wk["mu_bright"] - wk["mu_dark"])
        pi_min = float(min(wk["pi_dark"], wk["pi_bright"]))
        cond_multipeak = int(wk["num_peaks"] >= 2)
        cond_nonuniform = int(obj_range > OBJECT_UNIFORM_P95_P05_MAX)
        cond_gmm_separation = int(mu_gap >= GMM_RESCUE_MU_GAP_MIN)
        cond_gmm_pi_ok = int(pi_min >= GMM_RESCUE_PI_MIN)
        rescue = int((wk["num_peaks"] < 2) and (obj_range >= GMM_RESCUE_OBJ_RANGE_MIN) and bool(cond_gmm_separation) and bool(cond_gmm_pi_ok))
        use_wk = bool((cond_multipeak and cond_nonuniform) or rescue)
        if cond_multipeak and cond_nonuniform:
            gate_reason = "double_peak_gate"
        elif rescue:
            gate_reason = "gmm_separation_rescue"
        elif not cond_nonuniform:
            gate_reason = "blocked_by_uniformity"
        else:
            gate_reason = "blocked_by_peak_gate"

        otsu_thr = otsu_threshold_from_values(obj_vals, default=np.nan)

        bin_otp_smooth_thr = np.nan
        bin_otp_loose_weight = np.nan
        bin_otp_hard_reference = "not_applicable"
        if np.isfinite(robust_contrast):
            bin_otp_smooth_thr, bin_otp_loose_weight = bin_otp_smooth_strict_loose_threshold(robust_contrast)
            bin_otp_hard_reference = "fixed_threshold_strict" if robust_contrast <= RING_CONTRAST_STRICT_MAX else "fixed_threshold_loose"

        if use_wk:
            final_method = "weighted_kdegmm"
            thr = float(wk["threshold_x"])
            fallback_guard = guarded_fallback_threshold(
                base_thr=thr,
                use_shadow=use_shadow,
                use_wk=True,
                bg_reliable=bool(bg_reliable),
                kde_num_peaks=int(wk["num_peaks"]),
                robust_contrast=robust_contrast,
                obj_p50=obj_p50,
                obj_p90=obj_p90,
                ring_p90=ring_p90,
            )
        else:
            if not bg_reliable:
                final_method, thr = "fixed_threshold_strict_unreliable_bg", float(FIXED_THRESHOLD_STRICT)
                fallback_guard = guarded_fallback_threshold(
                    base_thr=thr,
                    use_shadow=use_shadow,
                    use_wk=False,
                    bg_reliable=False,
                    kde_num_peaks=int(wk["num_peaks"]),
                    robust_contrast=robust_contrast,
                    obj_p50=obj_p50,
                    obj_p90=obj_p90,
                    ring_p90=ring_p90,
                )
            else:
                base_thr = float(bin_otp_smooth_thr)
                fallback_guard = guarded_fallback_threshold(
                    base_thr=base_thr,
                    use_shadow=use_shadow,
                    use_wk=False,
                    bg_reliable=bool(bg_reliable),
                    kde_num_peaks=int(wk["num_peaks"]),
                    robust_contrast=robust_contrast,
                    obj_p50=obj_p50,
                    obj_p90=obj_p90,
                    ring_p90=ring_p90,
                )
                final_method, thr = str(fallback_guard["method"]), float(fallback_guard["threshold"])
        final_crop = classify_by_threshold(analysis_gray_crop, objmask, thr)
        final[y:y+h, x:x+w] = np.maximum(final[y:y+h, x:x+w], final_crop.astype(np.uint8))

        area = int(objmask.sum())
        final_pixels = int((final_crop * objmask).sum())
        gt_pixels = int((gt[y:y+h, x:x+w] * objmask).sum())
        row = {
            "parent": parent, "object_id": int(obj["object_id"]),
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "object_area": area, "core_pixels": core_pixels,
            "ring_pixels": int(bg_sel["ring_pixels"]), "bg_pixels_used": int(bg_sel["bg_pixels_used"]),
            "bg_source": str(bg_sel["bg_source"]), "bg_reliable": bg_reliable,
            "analysis_gray_source": analysis_gray_source,
            "shadow_crossing_detected": int(use_shadow), "shadow_reason": str(shadow["shadow_reason"]),
            "shadow_by_grad": int(shadow["shadow_by_grad"]), "shadow_by_ring": int(shadow["shadow_by_ring"]),
            "obj_illum_range_p90_p10": float(shadow["obj_illum_range_p90_p10"]),
            "obj_illum_grad_p95": float(shadow["obj_illum_grad_p95"]),
            "ring_illum_iqr": float(shadow["ring_illum_iqr"]),
            "object_touches_image_border": int(ctx["object_touches_image_border"]),
            "context_truncated_by_image_border": int(ctx["context_truncated_by_image_border"]),
            "ring_touch_context_border": int(bg_sel["ring_touch_context_border"]),
            "selected_ring_outer_ksize": int(bg_sel["selected_outer_ksize"]),
            "raw_obj_mean_gray": float(raw_obj_vals.mean()), "raw_obj_p50": float(np.percentile(raw_obj_vals, 50)),
            "raw_obj_p90": float(np.percentile(raw_obj_vals, 90)), "raw_obj_p95": float(np.percentile(raw_obj_vals, 95)),
            "raw_obj_p95_p05": float(np.percentile(raw_obj_vals, 95) - np.percentile(raw_obj_vals, 5)),
            "obj_mean_gray": float(obj_vals.mean()), "obj_std_gray": float(obj_vals.std()),
            "obj_p05": obj_p05, "obj_p50": obj_p50, "obj_p90": obj_p90, "obj_p95": obj_p95, "obj_p95_p05": obj_range,
            "ring_mean_gray": ring_mean, "ring_p50": ring_p50, "ring_p90": ring_p90,
            "mean_delta": mean_delta, "robust_contrast_obj_p90_minus_ring_p50": robust_contrast,
            "kde_num_peaks": int(wk["num_peaks"]), "cond_multipeak": cond_multipeak, "cond_nonuniform": cond_nonuniform,
            "cond_gmm_separation": cond_gmm_separation, "cond_gmm_pi_ok": cond_gmm_pi_ok, "rescue_kdegmm": rescue,
            "kdegmm_gate_reason": gate_reason, "use_weighted_kdegmm": int(use_wk),
            "wk_mu_dark": float(wk["mu_dark"]), "wk_mu_bright": float(wk["mu_bright"]), "wk_mu_gap": mu_gap,
            "wk_pi_dark": float(wk["pi_dark"]), "wk_pi_bright": float(wk["pi_bright"]), "wk_pi_min": pi_min,
            "wk_threshold_x": float(wk["threshold_x"]), "wk_threshold_source": str(wk["threshold_source"]),
            "otsu_threshold_x": float(otsu_thr) if np.isfinite(otsu_thr) else np.nan,
            "bin_otp_smooth_threshold_x": float(bin_otp_smooth_thr) if np.isfinite(bin_otp_smooth_thr) else np.nan,
            "bin_otp_loose_weight": float(bin_otp_loose_weight) if np.isfinite(bin_otp_loose_weight) else np.nan,
            "bin_otp_hard_reference": bin_otp_hard_reference,
            "bin_otp_transition_center": float(BIN_OTP_TRANSITION_CENTER),
            "bin_otp_transition_tau": float(BIN_OTP_TRANSITION_TAU),
            "fallback_guard_enabled": int(ENABLE_GUARDED_LOCAL_FALLBACK),
            "fallback_guard_applied": int(bool(fallback_guard["applied"])),
            "fallback_guard_reason": str(fallback_guard["reason"]),
            "fallback_guard_threshold_x": float(fallback_guard["threshold"]),
            "fallback_guard_ring_guard_x": float(fallback_guard["ring_guard_x"]) if np.isfinite(fallback_guard["ring_guard_x"]) else np.nan,
            "fallback_guard_object_guard_x": float(fallback_guard["object_guard_x"]) if np.isfinite(fallback_guard["object_guard_x"]) else np.nan,
            "fallback_guard_upper_guard_x": float(fallback_guard["upper_guard_x"]) if np.isfinite(fallback_guard["upper_guard_x"]) else np.nan,
            "final_method": final_method, "final_method_id": int(METHOD_ID.get(final_method, 0)), "final_threshold_x": float(thr),
            "gt_pixels_in_object": gt_pixels, "pred_pixels_in_object": area, "final_pixels_in_object": final_pixels,
            "final_white_ratio": float(final_pixels / area) if area > 0 else np.nan,
            "final_blur_rate": float(1.0 - final_pixels / area) if area > 0 else np.nan,
        }
        all_rows.append(row)
        processed_object_count += 1

    # Save core R_Pred outputs first. These are required by downstream blur/validation scripts.
    save_mask(final, dirs["final"] / f"{parent}.png")
    save_mask(final, dirs["final"] / f"{parent}__final_mask.png")

    pred_area = int(pred.sum())
    final_white = int(final[pred > 0].sum())
    image_rows.append({
        "parent": parent, "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
        "object_count": len(objects), "valid_processed_object_count": processed_object_count,
        "pred_area_A3": pred_area, "final_white_pixels_A4_in_A3": final_white,
        "image_white_ratio": float(final_white / pred_area) if pred_area > 0 else np.nan,
        "image_blur_rate_D": float(1.0 - final_white / pred_area) if pred_area > 0 else np.nan,
        "gt_pixels_A1": int(gt.sum()), "final_pixels_A4_total": int(final.sum()),
    })



# ============================================================
# Parameter-sweep version from precomputed M_Pred Pred_Mask
# ============================================================
# This section replaces the original single-run main() with a one-at-a-time
# parameter sweep / ablation pipeline. It intentionally reuses the same
# process_parent() decision tree defined above, so the binarization logic is
# consistent with 02_binarize_residual_paint.py.

# Default paths for the current project. Change here or pass CLI arguments.
IMAGE_DIR = Path(r"")  # TODO: set path
PRED_MASK_DIR = Path(r"")  # TODO: set path
GT_MASK_DIR = Path(r"")      # M_GT, optional but recommended
GT_BIN_DIR = Path(r"")                       # R_GT, optional but recommended
OUTPUT_ROOT = Path(r"")  # TODO: set path
RUN_NAME = "MitB5_Bin-OTP_param_sweep_from_predmask"

# Baseline values for the Bin-OTP manuscript parameter sweep.
# MitB5 sweep 20260519_114448 showed tau=6 improved binary-mask consistency
# while retaining the original C0/Ts/Tl priors, so tau=6 is the updated baseline.
KDE_PEAK_PROMINENCE = 0.015
RING_CONTRAST_STRICT_MAX = 100.0
FIXED_THRESHOLD_STRICT = 205.0
FIXED_THRESHOLD_LOOSE = 165.0
BIN_OTP_TRANSITION_CENTER = RING_CONTRAST_STRICT_MAX
BIN_OTP_TRANSITION_TAU = 12.0
KDE_BANDWIDTH_SIGMA = 15.0
ENABLE_SHADOW_COMPENSATION = True
OBJECT_UNIFORM_P95_P05_MAX = 90.0

# Default sweep behavior.
SAVE_SIX_PANEL = False          # True = save 6-panel images for every config. Can be heavy.
SAVE_KDE_CURVES = False         # True = save KDE curves for every config. Can be heavy.
COPY_BASELINE_INTO_GROUP_PLOTS = True

# All parameters from the user's table.
TUNABLE_PARAMETER_NAMES = [
    "KDE_PEAK_PROMINENCE",
    "BIN_OTP_TRANSITION_CENTER",
    "BIN_OTP_TRANSITION_TAU",
    "FIXED_THRESHOLD_STRICT",
    "FIXED_THRESHOLD_LOOSE",
    "KDE_BANDWIDTH_SIGMA",
    "ENABLE_SHADOW_COMPENSATION",
    "OBJECT_UNIFORM_P95_P05_MAX",
]

PARAMETER_SWEEPS = [
    ("KDE_PEAK_PROMINENCE", [0.005, 0.01, 0.015, 0.02, 0.03]),
    ("BIN_OTP_TRANSITION_CENTER", [80.0, 90.0, 100.0, 110.0, 120.0]),
    ("BIN_OTP_TRANSITION_TAU", [6.0, 9.0, 12.0, 15.0, 18.0]),
    ("FIXED_THRESHOLD_STRICT", [195.0, 200.0, 205.0, 210.0, 215.0]),
    ("FIXED_THRESHOLD_LOOSE", [155.0, 160.0, 165.0, 170.0, 175.0]),
    ("KDE_BANDWIDTH_SIGMA", [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]),
    ("ENABLE_SHADOW_COMPENSATION", [True, False]),
    ("OBJECT_UNIFORM_P95_P05_MAX", [60.0, 70.0, 80.0, 90.0, 100.0]),
]

PARAMETER_DISPLAY_LABELS = {
    "KDE_PEAK_PROMINENCE": "KDE peak prominence",
    "BIN_OTP_TRANSITION_CENTER": "Transition center C0",
    "BIN_OTP_TRANSITION_TAU": "Transition softness tau",
    "FIXED_THRESHOLD_STRICT": "Strict endpoint T_s",
    "FIXED_THRESHOLD_LOOSE": "Loose endpoint T_l",
    "KDE_BANDWIDTH_SIGMA": "KDE bandwidth sigma",
    "ENABLE_SHADOW_COMPENSATION": "Shadow compensation",
    "OBJECT_UNIFORM_P95_P05_MAX": "Object uniformity threshold",
}


# Make the six-panel saving optional without changing process_parent().
_ORIGINAL_MAKE_NATIVE_STRIP = make_native_strip
def make_native_strip(parent_name: str, rgb: np.ndarray, gt: np.ndarray, ring_overlay: np.ndarray,
                      pred_overlay: np.ndarray, final_mask: np.ndarray, final_overlay_ids: np.ndarray,
                      out_path: Path) -> None:
    if SAVE_SIX_PANEL:
        _ORIGINAL_MAKE_NATIVE_STRIP(parent_name, rgb, gt, ring_overlay, pred_overlay, final_mask, final_overlay_ids, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-at-a-time parameter sweep for Weighted KDE-GMM from precomputed A3 Pred_Mask."
    )
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR), help="Full-size RGB image folder")
    parser.add_argument("--pred-mask-dir", type=str, default=str(PRED_MASK_DIR), help="A3 Pred_Mask folder")
    parser.add_argument("--gt-mask-dir", type=str, default=str(GT_MASK_DIR), help="A1 GT-original mask folder. Empty string disables A1.")
    parser.add_argument("--gt-bin-dir", type=str, default=str(GT_BIN_DIR), help="A2 GT-binarization folder. Empty string disables metric validation.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--num-random-parents", type=int, default=NUM_RANDOM_PARENTS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--resize-masks-to-image", action="store_true", default=RESIZE_MASKS_TO_IMAGE)
    parser.add_argument("--apply-pred-opening", action="store_true", default=APPLY_PRED_OPENING)
    parser.add_argument("--save-six-panel", action="store_true", help="Save six-panel visualizations for every sweep config.")
    parser.add_argument("--save-kde-curves", action="store_true", help="Save KDE/GMM curves for every sweep config.")
    parser.add_argument("--only-baseline", action="store_true", help="Run only the baseline configuration.")
    return parser.parse_args()


def short_param_name(name: str) -> str:
    mapping = {
        "BASE": "base",
        "KDE_PEAK_PROMINENCE": "pk",
        "BIN_OTP_TRANSITION_CENTER": "c0",
        "BIN_OTP_TRANSITION_TAU": "tau",
        "FIXED_THRESHOLD_STRICT": "ts",
        "FIXED_THRESHOLD_LOOSE": "tl",
        "KDE_BANDWIDTH_SIGMA": "bw",
        "ENABLE_SHADOW_COMPENSATION": "sg",
        "OBJECT_UNIFORM_P95_P05_MAX": "uni",
    }
    return mapping.get(str(name), str(name)[:8])


def safe_token(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        s = f"{value:g}"
    else:
        s = str(value)
    return s.replace(".", "p").replace("-", "m").replace(" ", "")


def group_folder_name(param_name: str) -> str:
    # Short folder names avoid Windows MAX_PATH failures during large sweeps.
    # Full parameter names and values are still saved in run_config.json and CSV files.
    mapping = {
        "BASE": "00_base",
        "KDE_PEAK_PROMINENCE": "01_pk",
        "BIN_OTP_TRANSITION_CENTER": "02_c0",
        "BIN_OTP_TRANSITION_TAU": "03_tau",
        "FIXED_THRESHOLD_STRICT": "04_ts",
        "FIXED_THRESHOLD_LOOSE": "05_tl",
        "KDE_BANDWIDTH_SIGMA": "06_bw",
        "ENABLE_SHADOW_COMPENSATION": "07_sg",
        "OBJECT_UNIFORM_P95_P05_MAX": "08_uni",
    }
    return mapping.get(str(param_name), str(param_name))


def capture_tunable_params() -> Dict[str, object]:
    return {name: globals()[name] for name in TUNABLE_PARAMETER_NAMES}


def apply_tunable_params(params: Dict[str, object]) -> None:
    """Apply active parameter values to the globals used by process_parent()."""
    global KDE_PEAK_PROMINENCE, RING_CONTRAST_STRICT_MAX, FIXED_THRESHOLD_STRICT, FIXED_THRESHOLD_LOOSE
    global BIN_OTP_TRANSITION_CENTER, BIN_OTP_TRANSITION_TAU
    global KDE_BANDWIDTH_SIGMA, ENABLE_SHADOW_COMPENSATION
    global OBJECT_UNIFORM_P95_P05_MAX

    KDE_PEAK_PROMINENCE = float(params["KDE_PEAK_PROMINENCE"])
    BIN_OTP_TRANSITION_CENTER = float(params["BIN_OTP_TRANSITION_CENTER"])
    BIN_OTP_TRANSITION_TAU = float(params["BIN_OTP_TRANSITION_TAU"])
    RING_CONTRAST_STRICT_MAX = float(BIN_OTP_TRANSITION_CENTER)
    FIXED_THRESHOLD_STRICT = float(params["FIXED_THRESHOLD_STRICT"])
    FIXED_THRESHOLD_LOOSE = float(params["FIXED_THRESHOLD_LOOSE"])
    KDE_BANDWIDTH_SIGMA = float(params["KDE_BANDWIDTH_SIGMA"])
    ENABLE_SHADOW_COMPENSATION = bool(params["ENABLE_SHADOW_COMPENSATION"])
    OBJECT_UNIFORM_P95_P05_MAX = float(params["OBJECT_UNIFORM_P95_P05_MAX"])


def same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-12
    except Exception:
        return a == b


def build_sweep_configs(base_params: Dict[str, object], only_baseline: bool = False) -> List[Dict]:
    configs = [{
        "sweep_param": "BASE",
        "sweep_value": "baseline",
        "params": dict(base_params),
        "group_name": group_folder_name("BASE"),
        "run_tag": "baseline",
        "is_baseline": True,
    }]
    if only_baseline:
        return configs

    for param, values in PARAMETER_SWEEPS:
        for v in values:
            if same_value(v, base_params[param]):
                continue
            p = dict(base_params)
            p[param] = v
            run_tag = f"{short_param_name(param)}_{safe_token(v)}"
            configs.append({
                "sweep_param": param,
                "sweep_value": v,
                "params": p,
                "group_name": group_folder_name(param),
                "run_tag": run_tag,
                "is_baseline": False,
            })
    return configs


def create_run_dirs(run_dir: Path) -> Dict[str, Path]:
    dirs = {
        "final": run_dir / "03_final_mask",
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def load_input_records(image_dir: Path, pred_dir: Path, gt_dir: Path, gt_bin_dir: Path,
                       num_random_parents: int, seed: int, resize: bool, apply_opening: bool) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    pred_paths = list_files(pred_dir)
    if not pred_paths:
        raise RuntimeError(f"No predicted masks found in: {pred_dir}")

    pred_paths = sorted(pred_paths)
    if num_random_parents and num_random_parents > 0:
        rng = random.Random(seed)
        pred_paths = rng.sample(pred_paths, k=min(num_random_parents, len(pred_paths)))
        pred_paths = sorted(pred_paths)

    records, missing_rows, skipped_rows, input_rows = [], [], [], []
    for pred_path in tqdm(pred_paths, desc="Loading A3 pred masks"):
        parent = normalize_stem(pred_path.stem)
        image_path = find_by_stem(image_dir, parent)
        gt_path = find_by_stem(gt_dir, parent) if str(gt_dir).strip() and gt_dir.exists() else None
        gt_bin_path = find_by_stem(gt_bin_dir, parent) if str(gt_bin_dir).strip() and gt_bin_dir.exists() else None

        if image_path is None:
            missing_rows.append({"parent": parent, "pred_mask": pred_path.name, "image_found": False, "A1_found": gt_path is not None, "A2_found": gt_bin_path is not None})
            continue

        try:
            rgb = read_rgb(image_path)
            pred = read_mask01(pred_path)
            if apply_opening:
                pred = smooth_binary(pred, PRED_OPEN_KSIZE)
            H, W = rgb.shape[:2]
            if pred.shape != (H, W):
                if resize:
                    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                else:
                    raise ValueError(f"A3/image shape mismatch: pred={pred.shape}, image={(H, W)}")

            if gt_path is not None:
                gt = read_mask01(gt_path)
                if gt.shape != (H, W):
                    if resize:
                        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    else:
                        raise ValueError(f"A1/image shape mismatch: A1={gt.shape}, image={(H, W)}")
            else:
                gt = np.zeros((H, W), dtype=np.uint8)

            if gt_bin_path is not None:
                gt_bin = read_mask01(gt_bin_path)
                if gt_bin.shape != (H, W):
                    if resize:
                        gt_bin = cv2.resize(gt_bin, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    else:
                        raise ValueError(f"A2/image shape mismatch: A2={gt_bin.shape}, image={(H, W)}")
            else:
                gt_bin = np.zeros((H, W), dtype=np.uint8)

            records.append({
                "parent": parent,
                "image_path": image_path,
                "pred_path": pred_path,
                "gt_path": gt_path,
                "gt_bin_path": gt_bin_path,
                "rgb": rgb,
                "pred": pred,
                "gt": gt,
                "gt_bin": gt_bin,
            })
            input_rows.append({
                "parent": parent,
                "image_name": image_path.name,
                "pred_mask_name": pred_path.name,
                "A1_gt_mask_name": gt_path.name if gt_path is not None else "",
                "A2_gt_bin_name": gt_bin_path.name if gt_bin_path is not None else "",
                "height": H,
                "width": W,
                "pred_pixels_A3": int(pred.sum()),
                "gt_pixels_A1": int(gt.sum()),
                "gt_bin_pixels_A2": int(gt_bin.sum()),
            })
        except Exception as e:
            skipped_rows.append({"parent": parent, "pred_mask": pred_path.name, "reason": repr(e)})
            print(f"[WARN] input skipped {parent}: {e}")

    return records, input_rows, missing_rows, skipped_rows


def safe_metric_ratio(num, den) -> float:
    return np.nan if den <= 0 else float(num) / float(den)


def D_from_Q(Q: float) -> float:
    if Q is None or np.isnan(Q):
        return np.nan
    return float(np.clip(1.0 - Q, 0.0, 1.0))


def pixel_counts(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> Dict[str, int]:
    p = (pred > 0) & (region > 0)
    g = (gt > 0) & (region > 0)
    r = region > 0
    return {
        "TP": int((p & g).sum()),
        "FP": int((p & (~g) & r).sum()),
        "FN": int(((~p) & g & r).sum()),
        "TN": int(((~p) & (~g) & r).sum()),
        "region_pixels": int(r.sum()),
        "pred_positive_pixels": int(p.sum()),
        "gt_positive_pixels": int(g.sum()),
    }


def metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pixel_acc": float(acc),
    }


def compute_image_validation_metrics(parent: str, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray) -> Dict:
    A1b, A2b, A3b, A4b = A1 > 0, A2 > 0, A3 > 0, A4 > 0
    A3_inter_A1 = A3b & A1b

    all_gt_Q = safe_metric_ratio(int((A2b & A1b).sum()), int(A1b.sum()))
    bi_gt_Q = safe_metric_ratio(int((A2b & A3_inter_A1).sum()), int(A3_inter_A1.sum()))
    orig_gt_Q = safe_metric_ratio(int((A4b & A1b).sum()), int(A1b.sum()))
    pred_ref_Q = safe_metric_ratio(int((A4b & A3_inter_A1).sum()), int(A3_inter_A1.sum()))
    proposed_Q = safe_metric_ratio(int((A4b & A3b).sum()), int(A3b.sum()))

    row = {
        "parent": parent,
        "A1_area": int(A1b.sum()),
        "A2_white_in_A1": int((A2b & A1b).sum()),
        "A3_area": int(A3b.sum()),
        "A3_inter_A1_area": int(A3_inter_A1.sum()),
        "A4_area": int(A4b.sum()),
        "A4_white_in_A3": int((A4b & A3b).sum()),

        "All_GT_Q": all_gt_Q,
        "All_GT_D": D_from_Q(all_gt_Q),
        "Bi_GT_Q": bi_gt_Q,
        "Bi_GT_D": D_from_Q(bi_gt_Q),
        "Orig_GT_Q": orig_gt_Q,
        "Orig_GT_D": D_from_Q(orig_gt_Q),
        "Pred_Ref_Q": pred_ref_Q,
        "Pred_Ref_D": D_from_Q(pred_ref_Q),
        "Proposed_A3_Q": proposed_Q,
        "Proposed_A3_D": D_from_Q(proposed_Q),
    }

    for name, region in [
        ("A4_vs_A2_within_A1", A1b.astype(np.uint8)),
        ("A4_vs_A2_within_A3_inter_A1", A3_inter_A1.astype(np.uint8)),
    ]:
        c = pixel_counts(A4, A2, region)
        m = metrics_from_counts(c)
        for k, v in c.items():
            row[f"{name}_{k}"] = v
        for k, v in m.items():
            row[f"{name}_{k}"] = v

    return row


def compute_error_summary(image_metric_df: pd.DataFrame) -> Dict:
    out = {}
    comparisons = [
        ("Pred_Ref_vs_All_GT", "All_GT_D", "Pred_Ref_D"),
        ("Orig_GT_vs_All_GT", "All_GT_D", "Orig_GT_D"),
        ("Bi_GT_vs_All_GT", "All_GT_D", "Bi_GT_D"),
        ("Proposed_A3_vs_All_GT", "All_GT_D", "Proposed_A3_D"),
    ]
    for name, x_col, y_col in comparisons:
        if x_col not in image_metric_df.columns or y_col not in image_metric_df.columns:
            continue
        sub = image_metric_df[[x_col, y_col, "A1_area"]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub["A1_area"] = pd.to_numeric(sub["A1_area"], errors="coerce")
        sub = sub.dropna(subset=[x_col, y_col])
        if sub.empty:
            out[f"{name}_n"] = 0
            out[f"{name}_MAE"] = np.nan
            out[f"{name}_RMSE"] = np.nan
            out[f"{name}_Bias"] = np.nan
            continue
        err = sub[y_col].to_numpy(float) - sub[x_col].to_numpy(float)
        abs_err = np.abs(err)
        out[f"{name}_n"] = int(len(sub))
        out[f"{name}_MAE"] = float(abs_err.mean())
        out[f"{name}_RMSE"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{name}_Bias"] = float(err.mean())
        out[f"{name}_Within_0p10"] = float((abs_err <= 0.10).mean())

        w = sub["A1_area"].fillna(0).to_numpy(float)
        if np.sum(w) > 0:
            out[f"{name}_AreaW_MAE"] = float(np.sum(w * abs_err) / np.sum(w))
            out[f"{name}_AreaW_RMSE"] = float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))
        else:
            out[f"{name}_AreaW_MAE"] = np.nan
            out[f"{name}_AreaW_RMSE"] = np.nan

    for eval_name in ["A4_vs_A2_within_A1", "A4_vs_A2_within_A3_inter_A1"]:
        counts = {}
        for k in ["TP", "FP", "FN", "TN", "region_pixels", "pred_positive_pixels", "gt_positive_pixels"]:
            col = f"{eval_name}_{k}"
            counts[k] = int(pd.to_numeric(image_metric_df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        mets = metrics_from_counts(counts)
        for k, v in counts.items():
            out[f"{eval_name}_{k}"] = v
        for k, v in mets.items():
            out[f"{eval_name}_{k}"] = v

    return out


def run_one_config(cfg: Dict, input_records: List[Dict], sweep_root: Path) -> Dict:
    apply_tunable_params(cfg["params"])

    group_dir = sweep_root / cfg["group_name"]
    run_dir = group_dir / cfg["run_tag"]
    dirs = create_run_dirs(run_dir)

    all_rows, image_rows, image_metric_rows, viz_warning_rows = [], [], [], []

    for rec in tqdm(input_records, desc=f"{cfg['sweep_param']}={cfg['sweep_value']}"):
        parent = rec["parent"]
        process_parent(
            parent=parent,
            rgb=rec["rgb"],
            pred=rec["pred"],
            gt=rec["gt"],
            dirs=dirs,
            all_rows=all_rows,
            image_rows=image_rows,
        )

        final_path = dirs["final"] / f"{parent}.png"
        if final_path.exists():
            A4 = read_mask01(final_path)
        else:
            # Fallback for compatibility.
            A4 = read_mask01(dirs["final"] / f"{parent}__final_mask.png")

        image_metric_rows.append(compute_image_validation_metrics(
            parent=parent,
            A1=rec["gt"],
            A2=rec["gt_bin"],
            A3=rec["pred"],
            A4=A4,
        ))

    object_df = pd.DataFrame(all_rows)
    image_df = pd.DataFrame(image_rows)
    img_metric_df = pd.DataFrame(image_metric_rows)

    object_df.to_csv(run_dir / "all_output_info.csv", index=False, encoding="utf-8-sig")
    image_df.to_csv(run_dir / "image_level_summary.csv", index=False, encoding="utf-8-sig")
    img_metric_df.to_csv(run_dir / "image_validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(viz_warning_rows).to_csv(run_dir / "visualization_warning_report.csv", index=False, encoding="utf-8-sig")

    method_counts = object_df["final_method"].value_counts().to_dict() if (not object_df.empty and "final_method" in object_df.columns) else {}
    summary = {
        "sweep_param": cfg["sweep_param"],
        "sweep_value": cfg["sweep_value"],
        "group_name": cfg["group_name"],
        "run_tag": cfg["run_tag"],
        "run_dir": str(run_dir),
        "is_baseline": bool(cfg.get("is_baseline", False)),
        "n_images": int(len(input_records)),
        "n_objects": int(len(object_df)),
        "mean_image_D_A3": float(pd.to_numeric(image_df.get("image_blur_rate_D", pd.Series(dtype=float)), errors="coerce").mean()) if not image_df.empty else np.nan,
        "method_counts_json": json.dumps(method_counts, ensure_ascii=False),
        "weighted_kdegmm_count": int(method_counts.get("weighted_kdegmm", 0)),
        "fixed_strict_count": int(method_counts.get("fixed_threshold_strict", 0) + method_counts.get("fixed_threshold_strict_unreliable_bg", 0)),
        "fixed_loose_count": int(method_counts.get("fixed_threshold_loose", 0)),
        "bin_otp_smooth_count": int(method_counts.get("bin_otp_smooth_strict_loose", 0)),
        "guarded_local_fallback_count": int(method_counts.get("guarded_local_fallback", 0)),
        "status": "success",
        "error": "",
    }
    summary.update({f"param_{k}": v for k, v in cfg["params"].items()})
    summary.update(compute_error_summary(img_metric_df))

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "one_at_a_time_parameter_sweep_from_A3_pred_mask",
            "RUN_DIR": str(run_dir),
            "SWEEP_PARAM": cfg["sweep_param"],
            "SWEEP_VALUE": cfg["sweep_value"],
            "ACTIVE_PARAMS": cfg["params"],
            "TUNABLE_PARAMETER_NAMES": TUNABLE_PARAMETER_NAMES,
            "PARAMETER_SWEEPS": PARAMETER_SWEEPS,
            "GUARDED_FALLBACK_BRANCH": "enabled_fixed_not_swept",
            "SAVE_SIX_PANEL": SAVE_SIX_PANEL,
            "SAVE_KDE_CURVES": SAVE_KDE_CURVES,
        }, f, ensure_ascii=False, indent=2)

    return summary


def add_baseline_rows_for_plot(summary_df: pd.DataFrame, base_summary: pd.Series, base_params: Dict[str, object]) -> pd.DataFrame:
    rows = []
    if base_summary is None or len(summary_df) == 0:
        return summary_df
    for param, _values in PARAMETER_SWEEPS:
        row = base_summary.to_dict()
        row["sweep_param"] = param
        row["sweep_value"] = base_params[param]
        row["group_name"] = group_folder_name(param)
        row["run_tag"] = "baseline_for_plot"
        row["is_baseline"] = True
        rows.append(row)
    non_base = summary_df[summary_df["sweep_param"] != "BASE"].copy()
    return pd.concat([non_base, pd.DataFrame(rows)], ignore_index=True)


def numeric_or_bool_sort_key(v):
    if isinstance(v, bool):
        return int(v)
    try:
        return float(v)
    except Exception:
        return str(v)


def plot_metric_by_parameter(plot_df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str) -> None:
    ensure_dir(out_dir)
    params = [p for p, _ in PARAMETER_SWEEPS]
    n = len(params)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), squeeze=False)

    for ax, param in zip(axes.ravel(), params):
        sub = plot_df[plot_df["sweep_param"] == param].copy()
        if sub.empty or metric not in sub.columns:
            ax.axis("off")
            continue
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            ax.axis("off")
            continue

        # Sort values in the natural numeric / bool order.
        sub = sub.sort_values("sweep_value", key=lambda s: s.map(numeric_or_bool_sort_key))
        xlabels = [str(v) for v in sub["sweep_value"].tolist()]
        y = sub[metric].to_numpy(float)
        x = np.arange(len(sub))

        ax.plot(x, y, marker="o", linewidth=2.2)
        baseline_mask = sub["is_baseline"].astype(bool).to_numpy()
        if baseline_mask.any():
            ax.scatter(x[baseline_mask], y[baseline_mask], s=110, marker="*", zorder=4, label="Baseline")

        ax.set_title(PARAMETER_DISPLAY_LABELS.get(param, param), fontsize=15, fontweight="bold")
        ax.set_xlabel("Tested value", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=10, loc="best")

    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.suptitle(metric, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_by_parameter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_summary_plots(summary_df: pd.DataFrame, base_params: Dict[str, object], out_dir: Path) -> None:
    ensure_dir(out_dir)
    if summary_df.empty:
        return
    base_rows = summary_df[summary_df["sweep_param"] == "BASE"]
    base_summary = base_rows.iloc[0] if len(base_rows) > 0 else None
    plot_df = add_baseline_rows_for_plot(summary_df, base_summary, base_params)

    metrics_to_plot = [
        ("Pred_Ref_vs_All_GT_MAE", "MAE of D"),
        ("Pred_Ref_vs_All_GT_RMSE", "RMSE of D"),
        ("Pred_Ref_vs_All_GT_AreaW_MAE", "Area-weighted MAE of D"),
        ("Orig_GT_vs_All_GT_MAE", "MAE of D"),
        ("A4_vs_A2_within_A1_f1", "Pixel F1"),
        ("A4_vs_A2_within_A1_iou", "Pixel IoU"),
        ("A4_vs_A2_within_A3_inter_A1_f1", "Pixel F1"),
        ("weighted_kdegmm_count", "Number of objects"),
    ]
    for metric, ylabel in metrics_to_plot:
        if metric in plot_df.columns:
            plot_metric_by_parameter(plot_df, out_dir, metric, ylabel)


def main() -> None:
    global IMAGE_DIR, PRED_MASK_DIR, GT_MASK_DIR, GT_BIN_DIR, OUTPUT_ROOT, RUN_NAME, RANDOM_SEED
    global SAVE_SIX_PANEL, SAVE_KDE_CURVES

    args = parse_args()
    IMAGE_DIR = Path(args.image_dir)
    PRED_MASK_DIR = Path(args.pred_mask_dir)
    GT_MASK_DIR = Path(args.gt_mask_dir) if str(args.gt_mask_dir).strip() else Path("__no_gt__")
    GT_BIN_DIR = Path(args.gt_bin_dir) if str(args.gt_bin_dir).strip() else Path("__no_gt_bin__")
    OUTPUT_ROOT = Path(args.output_root)
    RUN_NAME = args.run_name
    RANDOM_SEED = int(args.random_seed)
    SAVE_SIX_PANEL = bool(args.save_six_panel)
    SAVE_KDE_CURVES = bool(args.save_kde_curves)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if not PRED_MASK_DIR.exists():
        raise FileNotFoundError(f"PRED_MASK_DIR not found: {PRED_MASK_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Short sweep folder name avoids Windows MAX_PATH issues.
    # Full run metadata is still saved in run_config.json.
    sweep_root = OUTPUT_ROOT / f"PS_{safe_token(RUN_NAME)}_{timestamp}"
    ensure_dir(sweep_root)
    print(f"[INFO] Sweep root: {sweep_root}")

    input_records, input_rows, missing_rows, input_skipped_rows = load_input_records(
        image_dir=IMAGE_DIR,
        pred_dir=PRED_MASK_DIR,
        gt_dir=GT_MASK_DIR,
        gt_bin_dir=GT_BIN_DIR,
        num_random_parents=int(args.num_random_parents),
        seed=RANDOM_SEED,
        resize=bool(args.resize_masks_to_image),
        apply_opening=bool(args.apply_pred_opening),
    )
    if not input_records:
        raise RuntimeError("No valid input records were loaded. Check image/pred-mask matching and paths.")

    pd.DataFrame(input_rows).to_csv(sweep_root / "input_mask_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(sweep_root / "missing_input_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(input_skipped_rows).to_csv(sweep_root / "input_skipped_report.csv", index=False, encoding="utf-8-sig")

    base_params = capture_tunable_params()
    configs = build_sweep_configs(base_params, only_baseline=bool(args.only_baseline))

    plan_rows = []
    for idx, cfg in enumerate(configs, 1):
        plan = {
            "sweep_index": idx,
            "sweep_param": cfg["sweep_param"],
            "sweep_value": cfg["sweep_value"],
            "group_name": cfg["group_name"],
            "run_tag": cfg["run_tag"],
            "is_baseline": bool(cfg.get("is_baseline", False)),
        }
        plan.update({f"param_{k}": v for k, v in cfg["params"].items()})
        plan_rows.append(plan)
    pd.DataFrame(plan_rows).to_csv(sweep_root / "parameter_sweep_plan.csv", index=False, encoding="utf-8-sig")
    with open(sweep_root / "base_tunable_params.json", "w", encoding="utf-8") as f:
        json.dump(base_params, f, ensure_ascii=False, indent=2)

    summaries = []
    for idx, cfg in enumerate(configs, 1):
        print("\n" + "#" * 110)
        print(f"[{idx}/{len(configs)}] {cfg['sweep_param']} = {cfg['sweep_value']}")
        print("#" * 110)

        try:
            summary = run_one_config(cfg, input_records, sweep_root)
            summary["sweep_index"] = idx
        except Exception as e:
            summary = {
                "sweep_index": idx,
                "sweep_param": cfg["sweep_param"],
                "sweep_value": cfg["sweep_value"],
                "group_name": cfg["group_name"],
                "run_tag": cfg["run_tag"],
                "run_dir": "",
                "is_baseline": bool(cfg.get("is_baseline", False)),
                "n_images": int(len(input_records)),
                "n_objects": 0,
                "status": "failed",
                "error": repr(e),
            }
            summary.update({f"param_{k}": v for k, v in cfg["params"].items()})
            print(f"[FAILED] {cfg['sweep_param']}={cfg['sweep_value']}: {e}")

        summaries.append(summary)
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(sweep_root / "parameter_sweep_summary_by_config.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    make_summary_plots(summary_df, base_params, sweep_root / "Summary_Plots")

    with open(sweep_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "one_at_a_time_parameter_sweep_from_A3_pred_mask",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "IMAGE_DIR": str(IMAGE_DIR),
            "PRED_MASK_DIR_A3": str(PRED_MASK_DIR),
            "GT_MASK_DIR_A1": str(GT_MASK_DIR),
            "GT_BIN_DIR_A2": str(GT_BIN_DIR),
            "OUTPUT_ROOT": str(OUTPUT_ROOT),
            "SWEEP_ROOT": str(sweep_root),
            "RUN_NAME": RUN_NAME,
            "RANDOM_SEED": RANDOM_SEED,
            "NUM_RANDOM_PARENTS": int(args.num_random_parents),
            "RESIZE_MASKS_TO_IMAGE": bool(args.resize_masks_to_image),
            "APPLY_PRED_OPENING": bool(args.apply_pred_opening),
            "SAVE_SIX_PANEL": SAVE_SIX_PANEL,
            "SAVE_KDE_CURVES": SAVE_KDE_CURVES,
            "BASE_TUNABLE_PARAMS": base_params,
            "PARAMETER_SWEEPS": PARAMETER_SWEEPS,
            "GUARDED_FALLBACK_BRANCH": "enabled_fixed_not_swept",
        }, f, ensure_ascii=False, indent=2)

    apply_tunable_params(base_params)

    print("Done.")
    print("Sweep root:", sweep_root)
    print("Summary CSV:", sweep_root / "parameter_sweep_summary_by_config.csv")
    print("Summary plots:", sweep_root / "Summary_Plots")


if __name__ == "__main__":
    main()
