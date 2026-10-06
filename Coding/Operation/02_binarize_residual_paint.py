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
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import find_peaks
from tqdm import tqdm

# ============================================================
# 02. Test-only M_Pred Pred_Mask -> R_Pred Operation-binarization by Weighted KDE-GMM
#     with guarded local fallback
# ------------------------------------------------------------
# This script is model-agnostic. Any segmentation model can be used upstream
# as long as it outputs full-size binary Pred_Mask files.
#
# Required inputs:
#   IMAGE_DIR      full-size RGB image folder, e.g. 01 output/Image
#   PRED_MASK_DIR  M_Pred = inference-original/predicted ROI, e.g. Pred_Mask
# Main outputs:
#   03_final_mask/        R_Pred masks. Both parent.png and parent__final_mask.png are saved.
#   all_output_info.csv   object-level thresholds, method branch, gray statistics
#                         including fallback_guard_* audit columns
#   image_level_summary.csv
# ============================================================

IMAGE_DIR = Path(r"")  # TODO: set path
PRED_MASK_DIR = Path(r"")  # TODO: set path
OUTPUT_ROOT = Path(r"")  # TODO: set path
RUN_NAME = "Segformer_MitB5_wkdegmm_shadowcap_test_only"

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
OBJECT_UNIFORM_P95_P05_MAX = 90.0
EM_MAX_ITER = 80
EM_TOL = 1e-6
VAR_FLOOR = 9.0
GMM_RESCUE_MU_GAP_MIN = 45.0
GMM_RESCUE_OBJ_RANGE_MIN = 26.0
GMM_RESCUE_PI_MIN = 0.03

# Fallback thresholds – Bin-OTP sigmoid blend (C0=100, tau=12)
RING_CONTRAST_STRICT_MAX  = 100.0
FIXED_THRESHOLD_STRICT    = 205.0
FIXED_THRESHOLD_LOOSE     = 165.0
BIN_OTP_TRANSITION_CENTER = RING_CONTRAST_STRICT_MAX  # 100.0
BIN_OTP_TRANSITION_TAU    = 12.0
# Guarded local fallback. This only constrains the fixed sigmoid fallback
# when KDE-GMM is not selected and the local object/ring distribution supports
# a lower threshold.
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
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def save_mask(mask01: np.ndarray, path: Path) -> None:
    Image.fromarray((mask01.astype(np.uint8) * 255), mode="L").save(path)


def save_u16(arr: np.ndarray, path: Path) -> None:
    arr_u16 = np.clip(arr, 0, 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode="I;16").save(path)


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
    loose_thr:  float = FIXED_THRESHOLD_LOOSE,
) -> Tuple[float, float]:
    """Sigmoid-blended threshold between strict and loose endpoints.

    Returns (threshold, loose_weight) where loose_weight=sigmoid((contrast-C0)/tau).
    At contrast << C0 the result approaches strict_thr; at contrast >> C0 it approaches loose_thr.
    """
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


def process_parent(parent: str, rgb: np.ndarray, pred: np.ndarray, dirs: Dict[str, Path], all_rows: List[Dict], image_rows: List[Dict]) -> None:
    pred = pred.astype(np.uint8)
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

        bin_otp_smooth_thr   = np.nan
        bin_otp_loose_weight = np.nan
        bin_otp_hard_reference = "not_applicable"
        if np.isfinite(robust_contrast):
            bin_otp_smooth_thr, bin_otp_loose_weight = bin_otp_smooth_strict_loose_threshold(robust_contrast)
            bin_otp_hard_reference = ("fixed_threshold_strict" if robust_contrast <= RING_CONTRAST_STRICT_MAX
                                      else "fixed_threshold_loose")

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
        processed_object_count += 1

        area = int(objmask.sum())
        final_pixels = int((final_crop * objmask).sum())
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
            "bin_otp_transition_tau":    float(BIN_OTP_TRANSITION_TAU),
            "fallback_guard_enabled": int(ENABLE_GUARDED_LOCAL_FALLBACK),
            "fallback_guard_applied": int(bool(fallback_guard["applied"])),
            "fallback_guard_reason": str(fallback_guard["reason"]),
            "fallback_guard_threshold_x": float(fallback_guard["threshold"]),
            "fallback_guard_ring_guard_x": float(fallback_guard["ring_guard_x"]) if np.isfinite(fallback_guard["ring_guard_x"]) else np.nan,
            "fallback_guard_object_guard_x": float(fallback_guard["object_guard_x"]) if np.isfinite(fallback_guard["object_guard_x"]) else np.nan,
            "fallback_guard_upper_guard_x": float(fallback_guard["upper_guard_x"]) if np.isfinite(fallback_guard["upper_guard_x"]) else np.nan,
            "final_method": final_method, "final_method_id": int(METHOD_ID.get(final_method, 0)), "final_threshold_x": float(thr),
            "pred_pixels_in_object": area, "final_pixels_in_object": final_pixels,
            "final_white_ratio": float(final_pixels / area) if area > 0 else np.nan,
            "final_blur_rate": float(1.0 - final_pixels / area) if area > 0 else np.nan,
        }
        all_rows.append(row)

    save_mask(final, dirs["final"] / f"{parent}.png")
    save_mask(final, dirs["final"] / f"{parent}__final_mask.png")

    pred_area = int(pred.sum())
    final_white = int(final[pred > 0].sum())
    image_rows.append({
        "parent": parent, "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
        "object_count": len(objects), "valid_processed_object_count": int(processed_object_count),
        "pred_area_A3": pred_area, "final_white_pixels_A4_in_A3": final_white,
        "image_white_ratio": float(final_white / pred_area) if pred_area > 0 else np.nan,
        "image_blur_rate_D": float(1.0 - final_white / pred_area) if pred_area > 0 else np.nan,
        "final_pixels_A4_total": int(final.sum()),
    })


def build_run_dir_name(run_name: str, n: int, sample_n: int) -> str:
    """Short run folder name to avoid Windows MAX_PATH issues.

    Full parameter values are preserved in run_config.json, so shortening this
    folder name does not affect reproducibility.
    """
    tag = f"all{n}" if sample_n <= 0 else f"{n}"
    safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_name))
    safe_run = safe_run[:32].strip("_") or "run"
    return f"wk_{safe_run}_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    image_dir = Path(args.image_dir)
    pred_dir = Path(args.pred_mask_dir)
    output_root = Path(args.output_root)
    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {image_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"PRED_MASK_DIR not found: {pred_dir}")

    pred_paths = list_files(pred_dir)
    if args.num_random_parents and args.num_random_parents > 0:
        pred_paths = random.sample(pred_paths, k=min(args.num_random_parents, len(pred_paths)))
    pred_paths = sorted(pred_paths)
    run_dir = output_root / build_run_dir_name(args.run_name, len(pred_paths), args.num_random_parents)

    dirs = {
        "final": run_dir / "03_final_mask",
    }
    for d in dirs.values():
        ensure_dir(d)

    all_rows, image_rows, input_rows = [], [], []
    missing_image, skipped = [], []
    for pred_path in tqdm(pred_paths, desc="A3 pred-mask -> A4 binarization"):
        parent = normalize_stem(pred_path.stem)
        image_path = find_by_stem(image_dir, parent)
        if image_path is None:
            missing_image.append({"parent": parent, "pred_mask": pred_path.name})
            continue
        try:
            rgb = read_rgb(image_path)
            pred = read_mask01(pred_path)
            if args.apply_pred_opening:
                pred = smooth_binary(pred, PRED_OPEN_KSIZE)
            H, W = rgb.shape[:2]
            if pred.shape != (H, W):
                if args.resize_masks_to_image:
                    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                else:
                    raise ValueError(f"shape mismatch pred={pred.shape}, image={(H, W)}")
            input_rows.append({"parent": parent, "image_name": image_path.name, "pred_mask_name": pred_path.name, "height": H, "width": W, "pred_pixels_A3": int(pred.sum()), "apply_pred_opening": int(args.apply_pred_opening)})
            process_parent(parent, rgb, pred, dirs, all_rows, image_rows)
        except Exception as e:
            skipped.append({"parent": parent, "pred_mask": pred_path.name, "reason": repr(e)})
            print(f"[WARN] skipped {parent}: {e}")

    pd.DataFrame(all_rows).to_csv(run_dir / "all_output_info.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(image_rows).to_csv(run_dir / "image_level_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(input_rows).to_csv(run_dir / "input_mask_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_image).to_csv(run_dir / "missing_image_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped).to_csv(run_dir / "skipped_report.csv", index=False, encoding="utf-8-sig")
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "wkdegmm_from_A3_pred_mask", "RUN_NAME": args.run_name,
            "IMAGE_DIR": str(image_dir), "PRED_MASK_DIR_A3": str(pred_dir),
            "OUTPUT_ROOT": str(output_root), "RUN_DIR": str(run_dir),
            "FINAL_MASK_DIR_A4": str(dirs["final"]),
            "RANDOM_SEED": args.random_seed, "NUM_RANDOM_PARENTS": args.num_random_parents,
            "RESIZE_MASKS_TO_IMAGE": bool(args.resize_masks_to_image), "APPLY_PRED_OPENING": bool(args.apply_pred_opening),
            "METHOD_ID": METHOD_ID,
            "PARAMETERS": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, bool, list, dict))},
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Processed images: {len(image_rows)}")
    print(f"A4 final masks : {dirs['final']}")
    print(f"Object CSV     : {run_dir / 'all_output_info.csv'}")
    print(f"Run dir        : {run_dir}")


if __name__ == "__main__":
    main()
