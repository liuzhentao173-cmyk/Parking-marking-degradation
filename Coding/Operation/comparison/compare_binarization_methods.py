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
# P3. Binarization-method comparison from M_Pred Pred_Mask
# ------------------------------------------------------------
# Methods compared, under two grayscale variants:
#   1) proposed_baseline: Bin-OTP full decision tree
#   2) otsu: object-wise Otsu threshold
#   3) fixed_strict: high fixed threshold, same as baseline strict
#   4) fixed_loose: low fixed threshold, same as baseline loose
#   5) pure_wkdegmm: Weighted KDE-GMM threshold without decision-tree fallback
#
# Grayscale variants:
#   raw        : no illumination compensation
#   illum_comp : illumination compensation enabled; by default only used when
#                the script detects a shadow/illumination crossing.
#
# Required inputs:
#   IMAGE_DIR      full-size RGB images
#   PRED_MASK_DIR  M_Pred predicted ROI masks
#   GT_MASK_DIR    M_GT GT ROI masks, recommended for D evaluation
#   GT_BIN_DIR     R_GT GT binary residual-paint masks, required for pixel metrics
#
# Main outputs:
#   runs/<comp>/<method>/03_final_mask/*.png
#   runs/<comp>/<method>/all_output_info.csv
#   runs/<comp>/<method>/image_level_summary.csv
#   runs/<comp>/<method>/image_validation_metrics.csv
#   method_level_summary.csv
#   per_image_metrics_long.csv
#   object_level_info_long.csv
#   paper_rank_table.csv
#   plots/*.png
# ============================================================

# =============================
# User config: edit paths here
# =============================
IMAGE_DIR = Path(r"")  # TODO: set path
PRED_MASK_DIR = Path(r"")  # TODO: set path
GT_MASK_DIR = Path(r"")       # M_GT = GT ROI/original marking mask
GT_BIN_DIR = Path(r"")      # R_GT = GT binary residual paint mask
OUTPUT_ROOT = Path(r"")  # Main output folder for all results of this script. Subfolders will be created inside.
RUN_NAME = "MitB5_binarization_method_compare_BinOTP-1"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_THRESHOLD = 127
NUM_RANDOM_PARENTS = -1
RANDOM_SEED = 42
RESIZE_MASKS_TO_IMAGE = False
APPLY_PRED_OPENING = False
PRED_OPEN_KSIZE = 3

# Object extraction, copied from your baseline logic
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

# Weighted KDE-GMM
KDE_BANDWIDTH_SIGMA = 15.0
KDE_PEAK_PROMINENCE = 0.015
OBJECT_UNIFORM_P95_P05_MAX = 90.0
EM_MAX_ITER = 80
EM_TOL = 1e-6
VAR_FLOOR = 9.0
GMM_RESCUE_MU_GAP_MIN = 45.0
GMM_RESCUE_OBJ_RANGE_MIN = 26.0
GMM_RESCUE_PI_MIN = 0.03
PURE_WK_FALLBACK = "otsu"  # choices: otsu, zero, strict, loose

# Baseline fallback thresholds / Bin-OTP smooth transition
RING_CONTRAST_STRICT_MAX = 100.0
FIXED_THRESHOLD_STRICT = 205.0
FIXED_THRESHOLD_LOOSE = 165.0
BIN_OTP_TRANSITION_CENTER = RING_CONTRAST_STRICT_MAX
BIN_OTP_TRANSITION_TAU = 12.0
# Guarded local fallback for over-strict Bin-OTP thresholds.
# It is part of proposed_baseline, not an additional compared method.
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

# Illumination / shadow compensation
ILLUM_INPAINT_RADIUS = 5
ILLUM_BLUR_SIGMA = 21.0
SHADOW_OBJ_ILLUM_RANGE_MIN = 18.0
SHADOW_OBJ_GRAD_P95_MIN = 1.8
SHADOW_RING_IQR_MIN = 12.0
COMPENSATION_MODE = "detected"  # choices: detected, always

# Output controls
SAVE_VISUAL_PANELS = True
SAVE_KDE_CURVES = False
SAVE_EXCEL = True
N_HARDEST_CASES = 6
FONT_TITLE = 18
FONT_AXIS = 15
FONT_TICK = 12
FONT_LEGEND = 12
FONT_ANNOT = 11
FIG_DPI = 300

METHODS = [
    "proposed_baseline",
    "otsu",
    "fixed_strict",
    "fixed_loose",
    "pure_wkdegmm",
]
COMP_VARIANTS = ["raw", "illum_comp"]
METHOD_LABEL = {
    "proposed_baseline": "Proposed guarded Bin-OTP",
    "otsu": "Otsu",
    "fixed_strict": f"Fixed strict ({FIXED_THRESHOLD_STRICT:g})",
    "fixed_loose": f"Fixed loose ({FIXED_THRESHOLD_LOOSE:g})",
    "pure_wkdegmm": "Pure WKDE-GMM",
}
COMP_LABEL = {
    "raw": "No compensation",
    "illum_comp": "With compensation",
}

# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare binarization methods from A3 Pred_Mask.")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR), help="Full-size RGB image folder")
    parser.add_argument("--pred-mask-dir", type=str, default=str(PRED_MASK_DIR), help="A3 Pred_Mask folder")
    parser.add_argument("--gt-mask-dir", type=str, default=str(GT_MASK_DIR), help="A1 GT ROI mask folder. Empty string disables A1-dependent D metrics.")
    parser.add_argument("--gt-bin-dir", type=str, default=str(GT_BIN_DIR), help="A2 GT binary mask folder. Empty string disables pixel comparison metrics.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--num-random-parents", type=int, default=NUM_RANDOM_PARENTS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--resize-masks-to-image", action="store_true", default=RESIZE_MASKS_TO_IMAGE)
    parser.add_argument("--apply-pred-opening", action="store_true", default=APPLY_PRED_OPENING)
    parser.add_argument("--save-visual-panels", action="store_true", default=SAVE_VISUAL_PANELS)
    parser.add_argument("--no-visual-panels", action="store_true", help="Disable visual panel saving.")
    parser.add_argument("--save-kde-curves", action="store_true", default=SAVE_KDE_CURVES)
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--compensation-mode", type=str, default=COMPENSATION_MODE, choices=["detected", "always"], help="How illum_comp uses the compensated grayscale.")
    parser.add_argument("--pure-wk-fallback", type=str, default=PURE_WK_FALLBACK, choices=["otsu", "zero", "strict", "loose"], help="Fallback only when pure WKDE-GMM is invalid, e.g., too few pixels.")
    return parser.parse_args()

# ============================================================
# Basic IO and mask helpers
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files(folder: Path) -> List[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


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


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_stem(stem: str) -> str:
    suffixes = [
        "__final_mask", "_final_mask", "__binary", "_binary", "__final", "_final",
        "__pred_mask", "_pred_mask", "__Pred_Mask", "_Pred_Mask",
        "__mask", "_mask", "__pred", "_pred", "_overlay",
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
        for ext in IMAGE_EXTS:
            candidates.append(folder / f"{s}{ext}")
    for p in candidates:
        if p.exists():
            return p
    norm = normalize_stem(stem)
    for p in list_files(folder):
        if normalize_stem(p.stem) == norm:
            return p
    return None


def build_mask_map(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in list_files(folder):
        key = normalize_stem(p.stem)
        if key in out:
            old = out[key]
            if old.stem.endswith("__final_mask") and not p.stem.endswith("__final_mask"):
                out[key] = p
        else:
            out[key] = p
    return out


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

# ============================================================
# Object/context/illumination functions
# ============================================================
def extract_objects_with_eroded_seed(full_mask: np.ndarray, erode_ksize: int, min_area: int) -> List[Dict]:
    full_mask = full_mask.astype(np.uint8)
    seed_mask = erode_mask(full_mask, erode_ksize)
    n_full, labels_full, _, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
    n_seed, labels_seed, stats_seed, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
    objects, used_full_labels, obj_id = [], set(), 0
    for sid in range(1, n_seed):
        sarea = int(stats_seed[sid, cv2.CC_STAT_AREA])
        if sarea < min_area:
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
            "full_mask_crop": full_comp[y0:y1 + 1, x0:x1 + 1].astype(np.uint8),
            "seed_mask_crop": seed_comp[y0:y1 + 1, x0:x1 + 1].astype(np.uint8),
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
    local_mask[oy:oy + h, ox:ox + w] = objmask.astype(np.uint8)
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
    detected = bool(shadow_by_grad or shadow_by_ring)
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
        "shadow_crossing_detected": int(detected),
        "shadow_reason": "|".join(reasons) if reasons else "none",
        "shadow_by_grad": int(shadow_by_grad),
        "shadow_by_ring": int(shadow_by_ring),
        "obj_illum_range_p90_p10": float(obj_range),
        "obj_illum_grad_p95": float(obj_grad_p95),
        "ring_illum_iqr": float(ring_iqr),
        "illumination_ref_value": float(illum_ref),
    }

# ============================================================
# Thresholding functions
# ============================================================
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
    kernel /= max(kernel.sum(), 1e-12)
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
        pdfs = np.stack([gaussian_pdf(x, mu[k], var[k]) for k in range(2)], axis=1)
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


def empty_wk_record() -> Dict:
    return {
        "threshold_x": np.nan, "threshold_source": "invalid", "num_peaks": 0, "peak_positions": [],
        "mu_dark": np.nan, "mu_bright": np.nan, "pi_dark": np.nan, "pi_bright": np.nan,
        "var_dark": np.nan, "var_bright": np.nan, "em_iters": 0, "em_ll": np.nan, "valley_x": np.nan,
        "x_grid": np.arange(256, dtype=np.float64), "kde": np.zeros(256),
        "pdf_dark": np.zeros(256), "pdf_bright": np.zeros(256), "mix_pdf": np.zeros(256),
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
    if not np.isfinite(thr):
        return out
    out[inside] = (gray_crop[inside].astype(np.float32) >= float(thr)).astype(np.uint8)
    return out


def choose_baseline_threshold(
    wk: Dict,
    obj_range: float,
    obj_p50: float,
    obj_p90: float,
    ring_p90: float,
    robust_contrast: float,
    bg_reliable: int,
    use_shadow: bool,
) -> Dict:
    mu_gap = float(wk["mu_bright"] - wk["mu_dark"]) if np.isfinite(wk.get("mu_bright", np.nan)) else np.nan
    pi_min = float(min(wk["pi_dark"], wk["pi_bright"])) if np.isfinite(wk.get("pi_dark", np.nan)) else np.nan
    cond_multipeak = int(wk.get("num_peaks", 0) >= 2)
    cond_nonuniform = int(obj_range > OBJECT_UNIFORM_P95_P05_MAX)
    cond_gmm_separation = int(np.isfinite(mu_gap) and mu_gap >= GMM_RESCUE_MU_GAP_MIN)
    cond_gmm_pi_ok = int(np.isfinite(pi_min) and pi_min >= GMM_RESCUE_PI_MIN)
    rescue = int((wk.get("num_peaks", 0) < 2) and (obj_range >= GMM_RESCUE_OBJ_RANGE_MIN) and bool(cond_gmm_separation) and bool(cond_gmm_pi_ok))
    use_wk = bool((cond_multipeak and cond_nonuniform) or rescue)

    bin_otp_smooth_thr = np.nan
    bin_otp_loose_weight = np.nan
    bin_otp_hard_reference = "not_applicable"
    if np.isfinite(robust_contrast):
        bin_otp_smooth_thr, bin_otp_loose_weight = bin_otp_smooth_strict_loose_threshold(robust_contrast)
        bin_otp_hard_reference = "fixed_threshold_strict" if robust_contrast <= RING_CONTRAST_STRICT_MAX else "fixed_threshold_loose"

    if use_wk:
        final_branch = "weighted_kdegmm"
        thr = float(wk["threshold_x"])
        fallback_guard = guarded_fallback_threshold(
            base_thr=thr,
            use_shadow=use_shadow,
            use_wk=True,
            bg_reliable=bool(bg_reliable),
            kde_num_peaks=int(wk.get("num_peaks", 0)),
            robust_contrast=robust_contrast,
            obj_p50=obj_p50,
            obj_p90=obj_p90,
            ring_p90=ring_p90,
        )
    else:
        if not bg_reliable:
            final_branch, thr = "fixed_threshold_strict_unreliable_bg", float(FIXED_THRESHOLD_STRICT)
            fallback_guard = guarded_fallback_threshold(
                base_thr=thr,
                use_shadow=use_shadow,
                use_wk=False,
                bg_reliable=False,
                kde_num_peaks=int(wk.get("num_peaks", 0)),
                robust_contrast=robust_contrast,
                obj_p50=obj_p50,
                obj_p90=obj_p90,
                ring_p90=ring_p90,
            )
        else:
            base_thr_val = float(bin_otp_smooth_thr)
            fallback_guard = guarded_fallback_threshold(
                base_thr=base_thr_val,
                use_shadow=use_shadow,
                use_wk=False,
                bg_reliable=bool(bg_reliable),
                kde_num_peaks=int(wk.get("num_peaks", 0)),
                robust_contrast=robust_contrast,
                obj_p50=obj_p50,
                obj_p90=obj_p90,
                ring_p90=ring_p90,
            )
            final_branch, thr = str(fallback_guard["method"]), float(fallback_guard["threshold"])

    if cond_multipeak and cond_nonuniform:
        gate_reason = "double_peak_gate"
    elif rescue:
        gate_reason = "gmm_separation_rescue"
    elif not cond_nonuniform:
        gate_reason = "blocked_by_uniformity"
    else:
        gate_reason = "blocked_by_peak_gate"

    return {
        "final_branch": final_branch,
        "threshold": thr,
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
        "use_weighted_kdegmm": int(use_wk),
        "cond_multipeak": cond_multipeak,
        "cond_nonuniform": cond_nonuniform,
        "cond_gmm_separation": cond_gmm_separation,
        "cond_gmm_pi_ok": cond_gmm_pi_ok,
        "rescue_kdegmm": rescue,
        "kdegmm_gate_reason": gate_reason,
        "wk_mu_gap": mu_gap,
        "wk_pi_min": pi_min,
    }

# ============================================================
# Visualization
# ============================================================
def color_mask(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=(255, 0, 0)) -> np.ndarray:
    vis = rgb.copy().astype(np.float32)
    col = np.zeros_like(vis)
    col[..., 0], col[..., 1], col[..., 2] = color
    keep = mask.astype(bool)
    vis[keep] = vis[keep] * (1.0 - alpha) + col[keep] * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)


def overlay_diff(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # TP=white, FP=red, FN=blue, TN=black
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    vis[np.logical_and(gt, pred)] = (255, 255, 255)
    vis[np.logical_and(~gt, pred)] = (255, 0, 0)
    vis[np.logical_and(gt, ~pred)] = (0, 0, 255)
    return vis


def make_method_panel(parent: str, rgb: np.ndarray, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray, out_path: Path, title: str) -> None:
    panels = [
        (rgb, "RGB"),
        (color_mask(A1), "A1 ROI"),
        (color_mask(A2), "A2 GT-binary"),
        (overlay_mask(rgb, A3, color=(255, 0, 0)), "A3 pred overlay"),
        (color_mask(A4), "A4 final"),
        (overlay_diff(A2 & A1, A4 & A1), "A1 diff: TP/FP/FN"),
    ]
    h, w = rgb.shape[:2]
    thumb_w = min(480, w)
    scale = thumb_w / w
    thumb_h = int(h * scale)
    resized = []
    for im, label in panels:
        pil = Image.fromarray(im.astype(np.uint8)).resize((thumb_w, thumb_h), Image.Resampling.BILINEAR if im.ndim == 3 else Image.Resampling.NEAREST)
        resized.append((np.array(pil), label))
    pad_top, label_h = 48, 36
    canvas = Image.new("RGB", (thumb_w * len(panels), pad_top + label_h + thumb_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), f"{parent} | {title}", fill=(0, 0, 0))
    for i, (im, label) in enumerate(resized):
        x0 = i * thumb_w
        draw.text((x0 + 10, pad_top + 10), label, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(im), (x0, pad_top + label_h))
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def plot_kde_curve(rec: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    x = rec.get("x_grid", np.arange(256))
    kde = rec.get("kde", np.zeros_like(x))
    mix = rec.get("mix_pdf", np.zeros_like(x))
    if np.max(kde) > 0:
        ax.fill_between(x, kde / max(kde.max(), 1e-12), alpha=0.25, label="KDE")
    if np.max(mix) > 0:
        ax.plot(x, rec.get("pdf_dark", np.zeros_like(x)) / max(mix.max(), 1e-12), linestyle="--", label="GMM-dark")
        ax.plot(x, rec.get("pdf_bright", np.zeros_like(x)) / max(mix.max(), 1e-12), linestyle="--", label="GMM-bright")
        ax.plot(x, mix / max(mix.max(), 1e-12), linewidth=2.0, label="GMM-mix")
    if np.isfinite(rec.get("threshold", np.nan)):
        ax.axvline(float(rec["threshold"]), linestyle="-.", linewidth=2.0, label=f"thr={rec['threshold']:.1f}")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Brightness")
    ax.set_ylabel("Normalized density")
    ax.set_title(str(rec.get("title", "KDE-GMM curve")))
    ax.legend(fontsize=10)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Metrics
# ============================================================
def safe_metric_ratio(num: int, den: int) -> float:
    return np.nan if den <= 0 else float(num) / float(den)


def D_from_Q(Q: float) -> float:
    if Q is None or np.isnan(Q):
        return np.nan
    return float(np.clip(1.0 - Q, 0.0, 1.0))


def confusion_stats(gt: np.ndarray, pred: np.ndarray, region: Optional[np.ndarray] = None) -> Dict[str, float]:
    if region is None:
        region = np.ones_like(gt, dtype=bool)
    else:
        region = region.astype(bool)
    gt = gt.astype(bool) & region
    pred = pred.astype(bool) & region
    tp = int(np.logical_and(gt, pred).sum())
    tn = int(np.logical_and(~gt, ~pred & region).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall),
        "dice": float(dice), "iou": float(iou), "accuracy": float(accuracy),
        "specificity": float(specificity), "fpr": float(fpr), "fnr": float(fnr),
    }


def compute_image_metrics(parent: str, method: str, comp: str, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray) -> Dict:
    A1b, A2b, A3b, A4b = A1.astype(bool), A2.astype(bool), A3.astype(bool), A4.astype(bool)
    A3_inter_A1 = A3b & A1b

    # GT and method D under multiple reference regions.
    gt_Q_A1 = safe_metric_ratio(int((A2b & A1b).sum()), int(A1b.sum()))
    method_Q_A1 = safe_metric_ratio(int((A4b & A1b).sum()), int(A1b.sum()))
    gt_Q_A3_inter_A1 = safe_metric_ratio(int((A2b & A3_inter_A1).sum()), int(A3_inter_A1.sum()))
    method_Q_A3_inter_A1 = safe_metric_ratio(int((A4b & A3_inter_A1).sum()), int(A3_inter_A1.sum()))
    method_Q_A3 = safe_metric_ratio(int((A4b & A3b).sum()), int(A3b.sum()))

    row = {
        "parent": parent,
        "method": method,
        "method_label": METHOD_LABEL.get(method, method),
        "comp_variant": comp,
        "comp_label": COMP_LABEL.get(comp, comp),
        "A1_area": int(A1b.sum()),
        "A2_white_in_A1": int((A2b & A1b).sum()),
        "A3_area": int(A3b.sum()),
        "A3_inter_A1_area": int(A3_inter_A1.sum()),
        "A4_area": int(A4b.sum()),
        "A4_white_in_A1": int((A4b & A1b).sum()),
        "A4_white_in_A3_inter_A1": int((A4b & A3_inter_A1).sum()),
        "A4_white_in_A3": int((A4b & A3b).sum()),
        "GT_Q_A1": gt_Q_A1,
        "GT_D_A1": D_from_Q(gt_Q_A1),
        "Method_Q_A1": method_Q_A1,
        "Method_D_A1": D_from_Q(method_Q_A1),
        "AbsErr_D_A1": abs(D_from_Q(method_Q_A1) - D_from_Q(gt_Q_A1)) if np.isfinite(D_from_Q(method_Q_A1)) and np.isfinite(D_from_Q(gt_Q_A1)) else np.nan,
        "GT_Q_A3_inter_A1": gt_Q_A3_inter_A1,
        "GT_D_A3_inter_A1": D_from_Q(gt_Q_A3_inter_A1),
        "Method_Q_A3_inter_A1": method_Q_A3_inter_A1,
        "Method_D_A3_inter_A1": D_from_Q(method_Q_A3_inter_A1),
        "AbsErr_D_A3_inter_A1": abs(D_from_Q(method_Q_A3_inter_A1) - D_from_Q(gt_Q_A3_inter_A1)) if np.isfinite(D_from_Q(method_Q_A3_inter_A1)) and np.isfinite(D_from_Q(gt_Q_A3_inter_A1)) else np.nan,
        "Method_Q_A3": method_Q_A3,
        "Method_D_A3": D_from_Q(method_Q_A3),
    }

    for region_name, region in [
        ("within_A1", A1b),
        ("within_A3_inter_A1", A3_inter_A1),
        ("within_A3", A3b),
    ]:
        c = confusion_stats(A2b.astype(np.uint8), A4b.astype(np.uint8), region.astype(np.uint8))
        for k, v in c.items():
            row[f"{region_name}_{k}"] = v
    return row


def summarize_method_metrics(image_df: pd.DataFrame, object_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if image_df.empty:
        return pd.DataFrame()
    for (comp, method), sub in image_df.groupby(["comp_variant", "method"], sort=False):
        row = {
            "comp_variant": comp,
            "comp_label": COMP_LABEL.get(comp, comp),
            "method": method,
            "method_label": METHOD_LABEL.get(method, method),
            "n_images": int(len(sub)),
        }
        for col in [
            "within_A1_iou", "within_A1_dice", "within_A1_precision", "within_A1_recall", "within_A1_fpr", "within_A1_fnr",
            "within_A3_inter_A1_iou", "within_A3_inter_A1_dice", "within_A3_inter_A1_precision", "within_A3_inter_A1_recall", "within_A3_inter_A1_fpr", "within_A3_inter_A1_fnr",
            "AbsErr_D_A1", "AbsErr_D_A3_inter_A1", "Method_D_A1", "GT_D_A1",
        ]:
            if col in sub.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
                row[f"std_{col}"] = float(pd.to_numeric(sub[col], errors="coerce").std(ddof=0))

        # Micro pixel metrics by summing counts.
        for region_name in ["within_A1", "within_A3_inter_A1", "within_A3"]:
            counts = {}
            for k in ["tp", "tn", "fp", "fn"]:
                counts[k] = int(pd.to_numeric(sub.get(f"{region_name}_{k}", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            eps = 1e-12
            tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
            row[f"micro_{region_name}_precision"] = float(tp / (tp + fp + eps))
            row[f"micro_{region_name}_recall"] = float(tp / (tp + fn + eps))
            row[f"micro_{region_name}_dice"] = float(2 * tp / (2 * tp + fp + fn + eps))
            row[f"micro_{region_name}_iou"] = float(tp / (tp + fp + fn + eps))
            row[f"micro_{region_name}_fpr"] = float(fp / (fp + tn + eps))
            row[f"micro_{region_name}_fnr"] = float(fn / (fn + tp + eps))
            for k, v in counts.items():
                row[f"micro_{region_name}_{k}"] = v

        # Area-weighted D errors.
        for err_col, weight_col, out_name in [
            ("AbsErr_D_A1", "A1_area", "AreaW_MAE_D_A1"),
            ("AbsErr_D_A3_inter_A1", "A3_inter_A1_area", "AreaW_MAE_D_A3_inter_A1"),
        ]:
            tmp = sub[[err_col, weight_col]].copy()
            tmp[err_col] = pd.to_numeric(tmp[err_col], errors="coerce")
            tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0)
            tmp = tmp.dropna(subset=[err_col])
            if tmp.empty or tmp[weight_col].sum() <= 0:
                row[out_name] = np.nan
            else:
                row[out_name] = float(np.sum(tmp[err_col] * tmp[weight_col]) / np.sum(tmp[weight_col]))

        # Object/branch summaries.
        osub = object_df[(object_df["comp_variant"] == comp) & (object_df["method"] == method)] if not object_df.empty else pd.DataFrame()
        row["n_objects"] = int(len(osub))
        if not osub.empty:
            row["mean_object_white_ratio"] = float(pd.to_numeric(osub["final_white_ratio"], errors="coerce").mean())
            row["mean_threshold"] = float(pd.to_numeric(osub["threshold"], errors="coerce").mean())
            for branch, n in osub["final_branch"].astype(str).value_counts().items():
                safe_branch = branch.replace(" ", "_")
                row[f"branch_count_{safe_branch}"] = int(n)
                row[f"branch_ratio_{safe_branch}"] = float(n / max(len(osub), 1))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        # Recommended primary score: high mask IoU + low blur D error.
        if "mean_within_A1_iou" in out.columns and "AreaW_MAE_D_A1" in out.columns:
            out["rank_mean_IoU_A1"] = pd.to_numeric(out["mean_within_A1_iou"], errors="coerce").rank(ascending=False, method="min")
            out["rank_AreaW_MAE_D_A1"] = pd.to_numeric(out["AreaW_MAE_D_A1"], errors="coerce").rank(ascending=True, method="min")
            out["paper_score_mean_rank"] = out[["rank_mean_IoU_A1", "rank_AreaW_MAE_D_A1"]].mean(axis=1)
            out = out.sort_values("paper_score_mean_rank", ascending=True)
    return out

# ============================================================
# Processing one image/method
# ============================================================
def get_analysis_gray(ctx: Dict, shadow: Dict, comp_variant: str, compensation_mode: str) -> Tuple[np.ndarray, str, int]:
    if comp_variant == "raw":
        return ctx["local_gray"], "raw", 0
    if compensation_mode == "always":
        return shadow["compensated_local_gray"], "illum_compensated_always", 1
    use_comp = int(shadow["shadow_crossing_detected"])
    if use_comp:
        return shadow["compensated_local_gray"], "illum_compensated_detected", 1
    return ctx["local_gray"], "raw_no_shadow_detected", 0


def fallback_threshold_for_pure_wk(otsu_thr: float, fallback: str) -> Tuple[float, str]:
    if fallback == "otsu":
        return float(otsu_thr) if np.isfinite(otsu_thr) else float(FIXED_THRESHOLD_STRICT), "fallback_otsu"
    if fallback == "strict":
        return float(FIXED_THRESHOLD_STRICT), "fallback_strict"
    if fallback == "loose":
        return float(FIXED_THRESHOLD_LOOSE), "fallback_loose"
    return np.nan, "fallback_zero"


def process_parent_for_method(parent: str, rgb: np.ndarray, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray,
                              method: str, comp_variant: str, run_dir: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict, Dict]:
    objects = extract_objects_with_eroded_seed(A3, SEED_ERODE_KSIZE, MIN_OBJECT_AREA)
    A4 = np.zeros_like(A3, dtype=np.uint8)
    obj_rows = []
    curve_records = []

    for obj in objects:
        x, y, w, h = int(obj["bbox_x"]), int(obj["bbox_y"]), int(obj["bbox_w"]), int(obj["bbox_h"])
        objmask = obj["full_mask_crop"].astype(np.uint8)
        ctx = build_local_object_context(rgb, x, y, w, h, objmask, RING_CONTEXT_PAD)
        bg_sel = select_background_region(ctx["local_mask"], ctx["local_gray"])
        bg_reliable = int(bg_sel["bg_source"] != "unreliable")
        bg_mask_local = bg_sel["bg_mask_local"] if bg_reliable else None
        shadow = detect_shadow_crossing_and_compensate(ctx["local_gray"], ctx["local_mask"], bg_sel["ring_viz_local"], bg_mask_local)
        analysis_local_gray, analysis_gray_source, compensation_used = get_analysis_gray(ctx, shadow, comp_variant, str(args.compensation_mode))
        analysis_gray_crop = analysis_local_gray[ctx["oy"]:ctx["oy"] + h, ctx["ox"]:ctx["ox"] + w]

        core = erode_mask(objmask, CORE_ERODE_KSIZE)
        core_pixels = int(core.sum())
        obj_vals = analysis_gray_crop[objmask > 0].astype(np.float64)
        raw_gray_crop = ctx["local_gray"][ctx["oy"]:ctx["oy"] + h, ctx["ox"]:ctx["ox"] + w]
        raw_obj_vals = raw_gray_crop[objmask > 0].astype(np.float64)
        if obj_vals.size == 0:
            continue

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

        otsu_thr = otsu_threshold_from_values(obj_vals, default=np.nan)
        wk = empty_wk_record()
        wk_status = "not_used"
        if core_pixels >= MIN_CORE_PIXELS:
            core_vals = analysis_gray_crop[core > 0].astype(np.uint8)
            if core_vals.size >= MIN_CORE_PIXELS:
                wk = weighted_kdegmm_threshold_from_core(core_vals)
                wk_status = "valid"

        threshold = np.nan
        final_branch = method
        baseline_decision = {
            "bin_otp_smooth_threshold_x": np.nan,
            "bin_otp_loose_weight": np.nan,
            "bin_otp_hard_reference": "not_baseline",
            "bin_otp_transition_center": float(BIN_OTP_TRANSITION_CENTER),
            "bin_otp_transition_tau": float(BIN_OTP_TRANSITION_TAU),
            "fallback_guard_enabled": int(ENABLE_GUARDED_LOCAL_FALLBACK),
            "fallback_guard_applied": 0,
            "fallback_guard_reason": "not_baseline",
            "fallback_guard_threshold_x": np.nan,
            "fallback_guard_ring_guard_x": np.nan,
            "fallback_guard_object_guard_x": np.nan,
            "fallback_guard_upper_guard_x": np.nan,
            "use_weighted_kdegmm": 0,
            "cond_multipeak": 0,
            "cond_nonuniform": 0,
            "cond_gmm_separation": 0,
            "cond_gmm_pi_ok": 0,
            "rescue_kdegmm": 0,
            "kdegmm_gate_reason": "not_baseline",
            "wk_mu_gap": np.nan,
            "wk_pi_min": np.nan,
        }

        if method == "proposed_baseline":
            baseline_decision = choose_baseline_threshold(
                wk,
                obj_range=obj_range,
                obj_p50=obj_p50,
                obj_p90=obj_p90,
                ring_p90=ring_p90,
                robust_contrast=robust_contrast,
                bg_reliable=bg_reliable,
                use_shadow=bool(compensation_used),
            )
            threshold = baseline_decision["threshold"]
            final_branch = baseline_decision["final_branch"]
        elif method == "otsu":
            threshold = float(otsu_thr) if np.isfinite(otsu_thr) else float(FIXED_THRESHOLD_STRICT)
            final_branch = "otsu"
        elif method == "fixed_strict":
            threshold = float(FIXED_THRESHOLD_STRICT)
            final_branch = "fixed_threshold_strict"
        elif method == "fixed_loose":
            threshold = float(FIXED_THRESHOLD_LOOSE)
            final_branch = "fixed_threshold_loose"
        elif method == "pure_wkdegmm":
            if wk_status == "valid" and np.isfinite(wk.get("threshold_x", np.nan)):
                threshold = float(wk["threshold_x"])
                final_branch = "pure_weighted_kdegmm"
            else:
                threshold, final_branch = fallback_threshold_for_pure_wk(otsu_thr, str(args.pure_wk_fallback))
        else:
            raise ValueError(f"Unknown method: {method}")

        final_crop = classify_by_threshold(analysis_gray_crop, objmask, threshold)
        A4[y:y + h, x:x + w] = np.maximum(A4[y:y + h, x:x + w], final_crop.astype(np.uint8))

        area = int(objmask.sum())
        final_pixels = int((final_crop * objmask).sum())
        A2_crop = A2[y:y + h, x:x + w]
        gt_pixels_in_object = int((A2_crop * objmask).sum())

        row = {
            "parent": parent,
            "method": method,
            "method_label": METHOD_LABEL.get(method, method),
            "comp_variant": comp_variant,
            "comp_label": COMP_LABEL.get(comp_variant, comp_variant),
            "object_id": int(obj["object_id"]),
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "object_area": area,
            "core_pixels": core_pixels,
            "bg_source": str(bg_sel["bg_source"]),
            "bg_reliable": bg_reliable,
            "ring_pixels": int(bg_sel["ring_pixels"]),
            "bg_pixels_used": int(bg_sel["bg_pixels_used"]),
            "selected_ring_outer_ksize": int(bg_sel["selected_outer_ksize"]),
            "analysis_gray_source": analysis_gray_source,
            "compensation_used": int(compensation_used),
            "shadow_crossing_detected": int(shadow["shadow_crossing_detected"]),
            "shadow_reason": str(shadow["shadow_reason"]),
            "shadow_by_grad": int(shadow["shadow_by_grad"]),
            "shadow_by_ring": int(shadow["shadow_by_ring"]),
            "obj_illum_range_p90_p10": float(shadow["obj_illum_range_p90_p10"]),
            "obj_illum_grad_p95": float(shadow["obj_illum_grad_p95"]),
            "ring_illum_iqr": float(shadow["ring_illum_iqr"]),
            "object_touches_image_border": int(ctx["object_touches_image_border"]),
            "context_truncated_by_image_border": int(ctx["context_truncated_by_image_border"]),
            "ring_touch_context_border": int(bg_sel["ring_touch_context_border"]),
            "raw_obj_mean_gray": float(raw_obj_vals.mean()),
            "raw_obj_p50": float(np.percentile(raw_obj_vals, 50)),
            "raw_obj_p90": float(np.percentile(raw_obj_vals, 90)),
            "raw_obj_p95_p05": float(np.percentile(raw_obj_vals, 95) - np.percentile(raw_obj_vals, 5)),
            "obj_mean_gray": float(obj_vals.mean()),
            "obj_std_gray": float(obj_vals.std()),
            "obj_p05": obj_p05,
            "obj_p50": obj_p50,
            "obj_p90": obj_p90,
            "obj_p95": obj_p95,
            "obj_p95_p05": obj_range,
            "ring_mean_gray": ring_mean,
            "ring_p50": ring_p50,
            "ring_p90": ring_p90,
            "mean_delta": mean_delta,
            "robust_contrast_obj_p90_minus_ring_p50": robust_contrast,
            "otsu_threshold_x": float(otsu_thr) if np.isfinite(otsu_thr) else np.nan,
            "wk_status": wk_status,
            "wk_num_peaks": int(wk.get("num_peaks", 0)),
            "wk_mu_dark": float(wk.get("mu_dark", np.nan)),
            "wk_mu_bright": float(wk.get("mu_bright", np.nan)),
            "wk_pi_dark": float(wk.get("pi_dark", np.nan)),
            "wk_pi_bright": float(wk.get("pi_bright", np.nan)),
            "wk_threshold_x": float(wk.get("threshold_x", np.nan)) if np.isfinite(wk.get("threshold_x", np.nan)) else np.nan,
            "wk_threshold_source": str(wk.get("threshold_source", "invalid")),
            "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
            "final_branch": final_branch,
            "gt_pixels_in_object": gt_pixels_in_object,
            "final_pixels_in_object": final_pixels,
            "final_white_ratio": float(final_pixels / area) if area > 0 else np.nan,
            "final_blur_rate": float(1.0 - final_pixels / area) if area > 0 else np.nan,
        }
        row.update(baseline_decision)
        obj_rows.append(row)

        if SAVE_KDE_CURVES or bool(args.save_kde_curves):
            curve = dict(wk)
            curve["threshold"] = threshold
            curve["title"] = f"{parent} obj {obj['object_id']} | {method} | {comp_variant} | {final_branch}"
            curve_records.append(curve)

    final_dir = run_dir / "03_final_mask"
    save_mask(A4, final_dir / f"{parent}.png")
    save_mask(A4, final_dir / f"{parent}__final_mask.png")

    if bool(args.save_visual_panels) and not bool(args.no_visual_panels):
        panel_dir = run_dir / "02_method_panels"
        title = f"{METHOD_LABEL.get(method, method)} / {COMP_LABEL.get(comp_variant, comp_variant)}"
        make_method_panel(parent, rgb, A1, A2, A3, A4, panel_dir / f"{parent}__panel.png", title)

    if SAVE_KDE_CURVES or bool(args.save_kde_curves):
        for rec in curve_records:
            obj_id = str(rec.get("title", "obj")).split(" obj ")[-1].split(" ")[0]
            plot_kde_curve(rec, run_dir / "01_kde_curves" / parent / f"obj_{obj_id}.png")

    image_row = {
        "parent": parent,
        "method": method,
        "method_label": METHOD_LABEL.get(method, method),
        "comp_variant": comp_variant,
        "comp_label": COMP_LABEL.get(comp_variant, comp_variant),
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
        "object_count": int(len(objects)),
        "valid_processed_object_count": int(len(obj_rows)),
        "A3_area": int(A3.sum()),
        "A4_area": int(A4.sum()),
        "A4_white_in_A3": int((A4 & A3).sum()),
        "image_white_ratio_A4_in_A3": safe_metric_ratio(int((A4 & A3).sum()), int(A3.sum())),
        "image_blur_rate_D_A4_in_A3": D_from_Q(safe_metric_ratio(int((A4 & A3).sum()), int(A3.sum()))),
    }
    metric_row = compute_image_metrics(parent, method, comp_variant, A1, A2, A3, A4)
    return pd.DataFrame(obj_rows), image_row, metric_row

# ============================================================
# Input loading
# ============================================================
def load_input_records(image_dir: Path, pred_dir: Path, gt_dir: Path, gt_bin_dir: Path, args: argparse.Namespace) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    pred_paths = list_files(pred_dir)
    if not pred_paths:
        raise RuntimeError(f"No predicted masks found in: {pred_dir}")

    if int(args.num_random_parents) > 0:
        rng = random.Random(int(args.random_seed))
        pred_paths = rng.sample(pred_paths, k=min(int(args.num_random_parents), len(pred_paths)))
        pred_paths = sorted(pred_paths)

    records, input_rows, skipped_rows = [], [], []
    for pred_path in tqdm(pred_paths, desc="Loading inputs"):
        parent = normalize_stem(pred_path.stem)
        image_path = find_by_stem(image_dir, parent)
        gt_path = find_by_stem(gt_dir, parent) if str(gt_dir).strip() and gt_dir.exists() else None
        gt_bin_path = find_by_stem(gt_bin_dir, parent) if str(gt_bin_dir).strip() and gt_bin_dir.exists() else None
        if image_path is None:
            skipped_rows.append({"parent": parent, "pred_mask": pred_path.name, "reason": "missing RGB image"})
            continue
        try:
            rgb = read_rgb(image_path)
            H, W = rgb.shape[:2]
            pred = read_mask01(pred_path)
            if bool(args.apply_pred_opening):
                pred = smooth_binary(pred, PRED_OPEN_KSIZE)
            if pred.shape != (H, W):
                if bool(args.resize_masks_to_image):
                    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                else:
                    raise ValueError(f"A3/image shape mismatch: A3={pred.shape}, image={(H, W)}")

            if gt_path is not None:
                A1 = read_mask01(gt_path)
                if A1.shape != (H, W):
                    if bool(args.resize_masks_to_image):
                        A1 = cv2.resize(A1, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    else:
                        raise ValueError(f"A1/image shape mismatch: A1={A1.shape}, image={(H, W)}")
            else:
                A1 = np.zeros((H, W), dtype=np.uint8)

            if gt_bin_path is not None:
                A2 = read_mask01(gt_bin_path)
                if A2.shape != (H, W):
                    if bool(args.resize_masks_to_image):
                        A2 = cv2.resize(A2, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    else:
                        raise ValueError(f"A2/image shape mismatch: A2={A2.shape}, image={(H, W)}")
            else:
                A2 = np.zeros((H, W), dtype=np.uint8)

            records.append({
                "parent": parent,
                "image_path": image_path,
                "pred_path": pred_path,
                "gt_path": gt_path,
                "gt_bin_path": gt_bin_path,
                "rgb": rgb,
                "A3": pred,
                "A1": A1,
                "A2": A2,
            })
            input_rows.append({
                "parent": parent,
                "image_name": image_path.name,
                "pred_mask_name": pred_path.name,
                "A1_gt_mask_name": gt_path.name if gt_path is not None else "",
                "A2_gt_bin_name": gt_bin_path.name if gt_bin_path is not None else "",
                "height": H,
                "width": W,
                "A3_pixels": int(pred.sum()),
                "A1_pixels": int(A1.sum()),
                "A2_pixels": int(A2.sum()),
            })
        except Exception as e:
            skipped_rows.append({"parent": parent, "pred_mask": pred_path.name, "reason": repr(e)})
            print(f"[WARN] skipped {parent}: {e}")
    return records, pd.DataFrame(input_rows), pd.DataFrame(skipped_rows)

# ============================================================
# Plots and rank tables
# ============================================================
def setup_fonts() -> None:
    plt.rcParams.update({
        "font.size": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_TITLE,
    })


def method_sort_key(row: pd.Series) -> Tuple[int, int]:
    comp_order = {"raw": 0, "illum_comp": 1}
    method_order = {m: i for i, m in enumerate(METHODS)}
    return comp_order.get(str(row["comp_variant"]), 99), method_order.get(str(row["method"]), 99)


def add_plot_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["plot_label"] = df["method"].map(lambda m: METHOD_LABEL.get(m, m)) + "\n" + df["comp_variant"].map(lambda c: "SC" if c == "illum_comp" else "Raw")
    return df


def save_dual_axis_main_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    df = add_plot_label(summary_df.copy())
    df = df.sort_values(["comp_variant", "method"], key=lambda s: s.map({"raw": 0, "illum_comp": 1, **{m: i for i, m in enumerate(METHODS)}}))
    if "mean_within_A1_iou" not in df.columns or "AreaW_MAE_D_A1" not in df.columns:
        return
    x = np.arange(len(df))
    mean_iou = pd.to_numeric(df["mean_within_A1_iou"], errors="coerce").to_numpy(float)
    area_mae = pd.to_numeric(df["AreaW_MAE_D_A1"], errors="coerce").to_numpy(float)
    labels = df["plot_label"].tolist()

    fig, ax1 = plt.subplots(figsize=(max(12, len(df) * 1.0), 6.5))
    ax2 = ax1.twinx()
    ax2.bar(x, area_mae, alpha=0.35, label="Area-weighted MAE of D")
    ax1.plot(x, mean_iou, marker="o", linewidth=2.4, label="Mean IoU")
    ax1.set_ylabel("Mean IoU (A4 vs A2 within A1) ↑")
    ax2.set_ylabel("Area-weighted MAE of D ↓")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylim(max(0, np.nanmin(mean_iou) - 0.08), min(1.0, np.nanmax(mean_iou) + 0.08))
    if np.isfinite(area_mae).any():
        ax2.set_ylim(0, max(0.05, np.nanmax(area_mae) * 1.25))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax1.set_title("Binarization method comparison: mask quality and D-error", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(summary_df: pd.DataFrame, metric: str, out_path: Path, title: str, ascending: bool = False) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return
    df = add_plot_label(summary_df.copy())
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    if df.empty:
        return
    fig_h = max(5.5, 0.62 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    bars = ax.barh(df["plot_label"], df[metric])
    ax.set_xlabel(metric)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.45)
    vals = df[metric].to_numpy(float)
    pad = (np.nanmax(vals) - np.nanmin(vals)) * 0.015 if np.nanmax(vals) > np.nanmin(vals) else 0.005
    for bar, val in zip(bars, vals):
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=FONT_ANNOT)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(per_image_df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if per_image_df.empty or metric not in per_image_df.columns:
        return
    df = add_plot_label(per_image_df.copy())
    labels, data = [], []
    med = df.groupby("plot_label")[metric].median().sort_values(ascending=False)
    for label in med.index:
        vals = pd.to_numeric(df.loc[df["plot_label"] == label, metric], errors="coerce").dropna().values
        if vals.size:
            labels.append(label)
            data.append(vals)
    if not data:
        return
    fig_w = max(12, len(labels) * 0.9 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(summary_df: pd.DataFrame, metrics: List[str], out_path: Path) -> None:
    if summary_df.empty:
        return
    cols = [c for c in metrics if c in summary_df.columns]
    if not cols:
        return
    df = add_plot_label(summary_df.copy()).set_index("plot_label")
    arr_df = df[cols].apply(pd.to_numeric, errors="coerce")
    arr = arr_df.to_numpy(float)
    fig_w = max(12, len(cols) * 1.5 + 3)
    fig_h = max(5.5, len(arr_df) * 0.6 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(arr_df.index)))
    ax.set_yticklabels(arr_df.index)
    ax.set_title("Method-level metric heatmap", fontweight="bold")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, f"{val:.3f}" if np.isfinite(val) else "nan", ha="center", va="center", fontsize=FONT_ANNOT)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_scatter(summary_df: pd.DataFrame, out_path: Path) -> None:
    pcol, rcol = "mean_within_A1_precision", "mean_within_A1_recall"
    if summary_df.empty or pcol not in summary_df.columns or rcol not in summary_df.columns:
        return
    df = add_plot_label(summary_df.copy())
    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
    df = df.dropna(subset=[pcol, rcol])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(df[rcol], df[pcol], s=90, alpha=0.85, edgecolors="black", linewidths=0.7)
    for _, row in df.iterrows():
        ax.text(row[rcol] + 0.003, row[pcol] + 0.003, str(row["plot_label"]).replace("\n", " "), fontsize=FONT_ANNOT)
    ax.set_xlabel("Mean Recall within A1")
    ax.set_ylabel("Mean Precision within A1")
    ax.set_title("Precision-Recall tradeoff across methods", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_characteristic_cases(per_image_df: pd.DataFrame, records: List[Dict], run_root: Path, out_dir: Path, n_cases: int) -> None:
    if per_image_df.empty:
        return
    rec_map = {r["parent"]: r for r in records}
    ensure_dir(out_dir)
    mapping_rows = []
    # Hard cases for the recommended best method and baseline raw/comp.
    candidates = []
    best_keys = per_image_df.groupby(["comp_variant", "method"])["within_A1_iou"].mean().sort_values(ascending=False)
    if not best_keys.empty:
        candidates.append(best_keys.index[0])
    for key in [("raw", "proposed_baseline"), ("illum_comp", "proposed_baseline")]:
        if key not in candidates:
            candidates.append(key)

    for comp, method in candidates:
        sub = per_image_df[(per_image_df["comp_variant"] == comp) & (per_image_df["method"] == method)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("within_A1_iou", ascending=True).head(n_cases)
        for idx, (_, row) in enumerate(sub.iterrows(), 1):
            parent = row["parent"]
            rec = rec_map.get(parent)
            if rec is None:
                continue
            A4_path = run_root / "runs" / comp / method / "03_final_mask" / f"{parent}.png"
            if not A4_path.exists():
                continue
            A4 = read_mask01(A4_path)
            title = f"{METHOD_LABEL.get(method, method)} / {COMP_LABEL.get(comp, comp)}"
            out_name = f"{comp}__{method}__case_{idx:02d}__{parent[:36]}.png"
            make_method_panel(parent, rec["rgb"], rec["A1"], rec["A2"], rec["A3"], A4, out_dir / out_name, title)
            mapping_rows.append({
                "comp_variant": comp,
                "method": method,
                "case_index": idx,
                "parent": parent,
                "saved_file": out_name,
                "within_A1_iou": float(row.get("within_A1_iou", np.nan)),
                "AbsErr_D_A1": float(row.get("AbsErr_D_A1", np.nan)),
            })
    if mapping_rows:
        pd.DataFrame(mapping_rows).to_csv(out_dir / "characteristic_case_mapping.csv", index=False, encoding="utf-8-sig")


def save_plots(summary_df: pd.DataFrame, per_image_df: pd.DataFrame, out_dir: Path) -> None:
    setup_fonts()
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    save_dual_axis_main_plot(summary_df, plots_dir / "main_mean_iou_and_areaW_MAE_D.png")
    save_bar_plot(summary_df, "mean_within_A1_iou", plots_dir / "bar_mean_iou_within_A1.png", "Mean IoU within A1", ascending=False)
    save_bar_plot(summary_df, "AreaW_MAE_D_A1", plots_dir / "bar_area_weighted_MAE_D_A1.png", "Area-weighted MAE of D within A1", ascending=True)
    save_bar_plot(summary_df, "mean_within_A1_precision", plots_dir / "bar_mean_precision_within_A1.png", "Mean Precision within A1", ascending=False)
    save_bar_plot(summary_df, "mean_within_A1_recall", plots_dir / "bar_mean_recall_within_A1.png", "Mean Recall within A1", ascending=False)
    save_boxplot(per_image_df, "within_A1_iou", plots_dir / "box_per_image_iou_within_A1.png", "Per-image IoU within A1")
    save_boxplot(per_image_df, "AbsErr_D_A1", plots_dir / "box_per_image_abs_error_D_A1.png", "Per-image absolute error of D within A1")
    save_precision_recall_scatter(summary_df, plots_dir / "scatter_precision_recall_within_A1.png")
    save_heatmap(summary_df, [
        "mean_within_A1_iou", "micro_within_A1_iou", "mean_within_A1_dice",
        "mean_within_A1_precision", "mean_within_A1_recall", "mean_within_A1_fpr", "mean_within_A1_fnr",
        "AreaW_MAE_D_A1", "mean_AbsErr_D_A1",
        "mean_within_A3_inter_A1_iou", "AreaW_MAE_D_A3_inter_A1",
    ], plots_dir / "heatmap_method_metrics.png")

# ============================================================
# Main
# ============================================================
def main() -> None:
    global IMAGE_DIR, PRED_MASK_DIR, GT_MASK_DIR, GT_BIN_DIR, OUTPUT_ROOT, RUN_NAME
    global SAVE_KDE_CURVES, SAVE_EXCEL, COMPENSATION_MODE, PURE_WK_FALLBACK

    args = parse_args()
    IMAGE_DIR = Path(args.image_dir)
    PRED_MASK_DIR = Path(args.pred_mask_dir)
    GT_MASK_DIR = Path(args.gt_mask_dir) if str(args.gt_mask_dir).strip() else Path("__no_A1__")
    GT_BIN_DIR = Path(args.gt_bin_dir) if str(args.gt_bin_dir).strip() else Path("__no_A2__")
    OUTPUT_ROOT = Path(args.output_root)
    RUN_NAME = str(args.run_name)
    SAVE_KDE_CURVES = bool(args.save_kde_curves)
    SAVE_EXCEL = not bool(args.no_excel)
    COMPENSATION_MODE = str(args.compensation_mode)
    PURE_WK_FALLBACK = str(args.pure_wk_fallback)

    random.seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if not PRED_MASK_DIR.exists():
        raise FileNotFoundError(f"PRED_MASK_DIR not found: {PRED_MASK_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = OUTPUT_ROOT / f"MC_{RUN_NAME}_{timestamp}"
    ensure_dir(run_root)
    print(f"[INFO] Output root: {run_root}")

    records, input_df, skipped_df = load_input_records(IMAGE_DIR, PRED_MASK_DIR, GT_MASK_DIR, GT_BIN_DIR, args)
    if not records:
        raise RuntimeError("No valid input records were loaded.")
    input_df.to_csv(run_root / "input_mask_summary.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(run_root / "input_skipped_report.csv", index=False, encoding="utf-8-sig")

    all_object_frames = []
    all_image_rows = []
    all_metric_rows = []
    run_plan_rows = []

    for comp in COMP_VARIANTS:
        for method in METHODS:
            print("\n" + "#" * 100)
            print(f"[RUN] {COMP_LABEL.get(comp, comp)} / {METHOD_LABEL.get(method, method)}")
            print("#" * 100)
            method_run_dir = run_root / "runs" / comp / method
            ensure_dir(method_run_dir)
            run_plan_rows.append({"comp_variant": comp, "method": method, "run_dir": str(method_run_dir)})

            object_frames, image_rows, metric_rows = [], [], []
            for rec in tqdm(records, desc=f"{comp}/{method}"):
                obj_df, image_row, metric_row = process_parent_for_method(
                    parent=rec["parent"],
                    rgb=rec["rgb"],
                    A1=rec["A1"],
                    A2=rec["A2"],
                    A3=rec["A3"],
                    method=method,
                    comp_variant=comp,
                    run_dir=method_run_dir,
                    args=args,
                )
                object_frames.append(obj_df)
                image_rows.append(image_row)
                metric_rows.append(metric_row)

            object_df = pd.concat(object_frames, ignore_index=True) if object_frames else pd.DataFrame()
            image_df = pd.DataFrame(image_rows)
            metric_df = pd.DataFrame(metric_rows)

            object_df.to_csv(method_run_dir / "all_output_info.csv", index=False, encoding="utf-8-sig")
            image_df.to_csv(method_run_dir / "image_level_summary.csv", index=False, encoding="utf-8-sig")
            metric_df.to_csv(method_run_dir / "image_validation_metrics.csv", index=False, encoding="utf-8-sig")
            save_json({
                "MODE": "binarization_method_comparison_from_A3_pred_mask_Bin-OTP",
                "comp_variant": comp,
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "compensation_mode": str(args.compensation_mode),
                "pure_wk_fallback": str(args.pure_wk_fallback),
                "FIXED_THRESHOLD_STRICT": FIXED_THRESHOLD_STRICT,
                "FIXED_THRESHOLD_LOOSE": FIXED_THRESHOLD_LOOSE,
                "KDE_BANDWIDTH_SIGMA": KDE_BANDWIDTH_SIGMA,
                "KDE_PEAK_PROMINENCE": KDE_PEAK_PROMINENCE,
                "OBJECT_UNIFORM_P95_P05_MAX": OBJECT_UNIFORM_P95_P05_MAX,
            "BIN_OTP_TRANSITION_CENTER": BIN_OTP_TRANSITION_CENTER,
                "BIN_OTP_TRANSITION_TAU": BIN_OTP_TRANSITION_TAU,
                "RING_CONTRAST_STRICT_MAX": RING_CONTRAST_STRICT_MAX,
                "ENABLE_GUARDED_LOCAL_FALLBACK": ENABLE_GUARDED_LOCAL_FALLBACK,
            }, method_run_dir / "run_config.json")

            all_object_frames.append(object_df)
            all_image_rows.append(image_df)
            all_metric_rows.append(metric_df)

    object_long = pd.concat(all_object_frames, ignore_index=True) if all_object_frames else pd.DataFrame()
    image_long = pd.concat(all_image_rows, ignore_index=True) if all_image_rows else pd.DataFrame()
    metric_long = pd.concat(all_metric_rows, ignore_index=True) if all_metric_rows else pd.DataFrame()

    object_long.to_csv(run_root / "object_level_info_long.csv", index=False, encoding="utf-8-sig")
    image_long.to_csv(run_root / "image_level_summary_long.csv", index=False, encoding="utf-8-sig")
    metric_long.to_csv(run_root / "per_image_metrics_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(run_plan_rows).to_csv(run_root / "method_comparison_plan.csv", index=False, encoding="utf-8-sig")

    summary_df = summarize_method_metrics(metric_long, object_long)
    summary_df.to_csv(run_root / "method_level_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_root / "paper_rank_table.csv", index=False, encoding="utf-8-sig")

    if SAVE_EXCEL:
        try:
            with pd.ExcelWriter(run_root / "method_comparison_summary.xlsx", engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="method_level_summary", index=False)
                metric_long.to_excel(writer, sheet_name="per_image_metrics", index=False)
                image_long.to_excel(writer, sheet_name="image_level_summary", index=False)
                object_long.to_excel(writer, sheet_name="object_level_info", index=False)
                input_df.to_excel(writer, sheet_name="input_summary", index=False)
                skipped_df.to_excel(writer, sheet_name="skipped", index=False)
        except Exception as e:
            print(f"[WARN] Excel export failed: {e}")

    save_plots(summary_df, metric_long, run_root)
    save_characteristic_cases(metric_long, records, run_root, run_root / "characteristic_cases", int(N_HARDEST_CASES))

    best_text = ""
    if not summary_df.empty and "paper_score_mean_rank" in summary_df.columns:
        best = summary_df.sort_values("paper_score_mean_rank", ascending=True).iloc[0]
        best_text = (
            f"Best by mean-rank of Mean IoU and AreaW-MAE(D):\n"
            f"  method = {best['method_label']}\n"
            f"  compensation = {best['comp_label']}\n"
            f"  Mean IoU within A1 = {best.get('mean_within_A1_iou', np.nan):.6f}\n"
            f"  AreaW MAE D within A1 = {best.get('AreaW_MAE_D_A1', np.nan):.6f}\n"
        )
    summary_txt = [
        "Binarization method comparison summary (Bin-OTP baseline)",
        "=" * 60,
        f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"IMAGE_DIR: {IMAGE_DIR}",
        f"PRED_MASK_DIR_A3: {PRED_MASK_DIR}",
        f"GT_MASK_DIR_A1: {GT_MASK_DIR}",
        f"GT_BIN_DIR_A2: {GT_BIN_DIR}",
        "",
        "Recommended primary metrics:",
        "  1) mean_within_A1_iou: Mean IoU of A4 vs A2 inside A1; higher is better.",
        "  2) AreaW_MAE_D_A1: area-weighted absolute error of blur damage D; lower is better.",
        "",
        "Proposed baseline:",
        f"  Bin-OTP T(C)=(1-w)Ts+wTl, C0={BIN_OTP_TRANSITION_CENTER:g}, tau={BIN_OTP_TRANSITION_TAU:g}, Ts={FIXED_THRESHOLD_STRICT:g}, Tl={FIXED_THRESHOLD_LOOSE:g}.",
        "  Guarded fallback branch is enabled for over-strict single-peak local thresholds.",
        "",
        best_text,
        "Main files:",
        "  method_level_summary.csv",
        "  per_image_metrics_long.csv",
        "  object_level_info_long.csv",
        "  method_comparison_summary.xlsx",
        "  plots/main_mean_iou_and_areaW_MAE_D.png",
    ]
    (run_root / "analysis_summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    save_json({
        "MODE": "binarization_method_comparison_from_A3_pred_mask_Bin-OTP",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "IMAGE_DIR": str(IMAGE_DIR),
        "PRED_MASK_DIR_A3": str(PRED_MASK_DIR),
        "GT_MASK_DIR_A1": str(GT_MASK_DIR),
        "GT_BIN_DIR_A2": str(GT_BIN_DIR),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "RUN_ROOT": str(run_root),
        "RUN_NAME": RUN_NAME,
        "METHODS": METHODS,
        "COMP_VARIANTS": COMP_VARIANTS,
        "COMPENSATION_MODE": COMPENSATION_MODE,
        "PURE_WK_FALLBACK": PURE_WK_FALLBACK,
        "BASELINE_METHOD": "proposed_baseline_guarded_Bin-OTP_smooth_strict_loose",
        "KDE_PEAK_PROMINENCE": float(KDE_PEAK_PROMINENCE),
        "KDE_BANDWIDTH_SIGMA": float(KDE_BANDWIDTH_SIGMA),
        "OBJECT_UNIFORM_P95_P05_MAX": float(OBJECT_UNIFORM_P95_P05_MAX),
        "RING_CONTRAST_STRICT_MAX": float(RING_CONTRAST_STRICT_MAX),
        "FIXED_THRESHOLD_STRICT": float(FIXED_THRESHOLD_STRICT),
        "FIXED_THRESHOLD_LOOSE": float(FIXED_THRESHOLD_LOOSE),
        "BIN_OTP_TRANSITION_CENTER": float(BIN_OTP_TRANSITION_CENTER),
        "BIN_OTP_TRANSITION_TAU": float(BIN_OTP_TRANSITION_TAU),
        "ENABLE_GUARDED_LOCAL_FALLBACK": bool(ENABLE_GUARDED_LOCAL_FALLBACK),
        "NUM_RANDOM_PARENTS": int(args.num_random_parents),
        "RANDOM_SEED": int(args.random_seed),
        "RESIZE_MASKS_TO_IMAGE": bool(args.resize_masks_to_image),
        "APPLY_PRED_OPENING": bool(args.apply_pred_opening),
        "SAVE_VISUAL_PANELS": bool(args.save_visual_panels) and not bool(args.no_visual_panels),
        "SAVE_KDE_CURVES": bool(SAVE_KDE_CURVES),
    }, run_root / "run_config.json")

    print("\nDone.")
    print("Run root:", run_root)
    print("Method-level summary:", run_root / "method_level_summary.csv")
    print("Main figure:", run_root / "plots" / "main_mean_iou_and_areaW_MAE_D.png")


if __name__ == "__main__":
    main()
