# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

"""
04. GT-based local-window blur validation + comparison plots
================================================================================

Purpose
-------
This script is the window-level validation partner of Script 03.
It validates the proposed local-window blur damage against GT-based local blur scores.

Inputs
------
M_GT = GT_ORIG_DIR   : Manual marking ROI mask.
R_GT = GT_BIN_DIR    : Manual white-paint/binary mask.
M_Pred = PRED_ORIG_DIR : Model predicted ROI mask.
R_Pred = PRED_BIN_DIR  : Proposed final binary/white-paint mask.

Main local definitions
----------------------
For each overlapping local analysis window W_k:

GT local blur:
    U_k^GT = M_GT ∩ W_k
    Q_k^GT = |R_GT ∩ U_k^GT| / |U_k^GT|
    D_k^GT = 1 - Q_k^GT

Proposed local blur:
    U_k^Pred = M_Pred ∩ W_k
    Q_k^Pred = |R_Pred ∩ U_k^Pred| / |U_k^Pred|
    D_k^Pred = 1 - Q_k^Pred

A valid paired window requires both U_k^GT and U_k^Pred to have enough pixels.
This gives a local-window Y=X validation instead of only image-level validation.

Outputs
-------
1) window_level_validation.csv
   One row per valid local window candidate, including GT D, predicted D, error, and auxiliary modes.

2) image_level_window_validation_summary.csv
   One row per full-size image with MAE/RMSE/Bias/correlation for paired windows.

3) Global_Comparison_Plots
   Y=X scatter, error histogram, and area/error plots using all paired windows.

4) GT_Pred_Error_Heatmap_Overlay
   GT blur map, predicted blur map, and absolute-error map overlaid on original images.

5) Random_Window_Validation_Groups
   Only randomly selected 2 parent images are visualized as small-window crop groups.

6) Extra_Evaluation_Groups
   Additional BD-based diagnostic plots for:
   - Segmentation effect: R_GT/M_GT vs R_GT/M_Pred
   - Binarization effect : R_GT/M_GT vs R_Pred/M_GT
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# 0. Default paths
# ============================================================
IMAGE_DIR = Path(r"")  # TODO: set path
GT_ORIG_DIR = Path(r"")        # M_GT
GT_BIN_DIR = Path(r"")                         # R_GT
PRED_ORIG_DIR = Path(r"")  # M_Pred
PRED_BIN_DIR = Path(r"")  # R_Pred
OUTPUT_ROOT = Path(r"")  # TODO: set path

# ============================================================
# 1. Parameters
# ============================================================
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
MASK_THRESHOLD = 127
RESIZE_MASKS_TO_IMAGE = False

WINDOW_SIZE = 512
STRIDE = 256
MIN_GT_UNIT_AREA = 500
MIN_PRED_UNIT_AREA = 500
MIN_PAIR_AREA = 500

# Pixel-level tolerance for R_Pred-vs-R_GT validation. 5 roughly means ±2 px.
TOLERANCE_KSIZE = 5

RANDOM_SEED = 42
NUM_RANDOM_VIZ_PARENTS = 0
VIZ_WINDOWS_PER_PARENT = 12
SAVE_HEATMAP_FOR_ALL = True
SAVE_COMPARISON_PLOTS = True
ERROR_BAND = 0.10
FIG_DPI = 300
HEATMAP_SMOOTH_SIGMA = 55.0   # Visualization only; GT/Pred BD calculations remain window-based.
HEATMAP_COLORBAR_W = 95
HEATMAP_COLORBAR_PAD = 14

FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_ALPHA = 0.45
CONTOUR_THICKNESS = 2

# BGR colors.
COLOR_GREEN = (144, 238, 144)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GT = (255, 255, 255)
COLOR_PRED = (0, 255, 255)

# Large-font plot settings for PPT/paper use.
PLOT_FONT_BASE = 14
PLOT_FONT_TITLE = 18
PLOT_FONT_LABEL = 16
PLOT_FONT_TICK = 13
PLOT_FONT_LEGEND = 12
PLOT_FONT_METRIC_BOX = 12


# ============================================================
# 2. Argument parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT validation for local-window blur damage using A1/A2/A3/A4 masks.")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--gt-orig-dir", type=str, default=str(GT_ORIG_DIR), help="A1 GT original ROI mask folder.")
    parser.add_argument("--gt-bin-dir", type=str, default=str(GT_BIN_DIR), help="A2 GT binary/white-paint mask folder.")
    parser.add_argument("--pred-orig-dir", type=str, default=str(PRED_ORIG_DIR), help="A3 predicted ROI mask folder.")
    parser.add_argument("--pred-bin-dir", "--binary-mask-dir", dest="pred_bin_dir", type=str, default=str(PRED_BIN_DIR), help="A4 predicted binary/final white-paint mask folder.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--min-gt-unit-area", type=int, default=MIN_GT_UNIT_AREA)
    parser.add_argument("--min-pred-unit-area", type=int, default=MIN_PRED_UNIT_AREA)
    parser.add_argument("--min-pair-area", type=int, default=MIN_PAIR_AREA, help="Minimum area used when calculating paired window comparison.")
    parser.add_argument("--tolerance-ksize", type=int, default=TOLERANCE_KSIZE)
    parser.add_argument("--resize-masks-to-image", action="store_true", default=RESIZE_MASKS_TO_IMAGE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-random-viz-parents", type=int, default=NUM_RANDOM_VIZ_PARENTS)
    parser.add_argument("--viz-windows-per-parent", type=int, default=VIZ_WINDOWS_PER_PARENT)
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--no-comparison-plots", action="store_true")
    parser.add_argument("--heatmap-smooth-sigma", type=float, default=HEATMAP_SMOOTH_SIGMA, help="Gaussian smoothing sigma in pixels for direct pixel-damage heatmap only. Set <=0 to disable smoothing.")
    return parser.parse_args()


# ============================================================
# 3. Basic helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    if not folder.exists():
        return files
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def normalize_stem(stem: str) -> str:
    suffixes = [
        "__final_mask", "_final_mask", "__binary", "_binary", "__final", "_final",
        "__pred_mask", "_pred_mask", "__mask", "_mask", "__pred", "_pred", "_overlay",
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
    norm = normalize_stem(stem)
    candidate_stems = [stem, norm, f"{norm}__final_mask", f"{norm}_final_mask", f"{norm}__binary", f"{norm}_binary"]
    seen = set()
    for s in candidate_stems:
        if s in seen:
            continue
        seen.add(s)
        for ext in IMAGE_EXTS:
            p = folder / f"{s}{ext}"
            if p.exists():
                return p
    for p in list_files(folder):
        if normalize_stem(p.stem) == norm:
            return p
    return None


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def read_mask01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (img > MASK_THRESHOLD).astype(np.uint8)


def align_mask(mask: np.ndarray, H: int, W: int, resize: bool, name: str) -> np.ndarray:
    if mask.shape == (H, W):
        return mask.astype(np.uint8)
    if resize:
        return cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    raise ValueError(f"{name} shape mismatch: mask={mask.shape}, image={(H, W)}. Use --resize-masks-to-image if intended.")


# ============================================================
# 4. Sliding-window and metric helpers
# ============================================================

def _start_positions(length: int, window_size: int, stride: int) -> List[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, max(length - window_size + 1, 1), stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(int(s) for s in starts))


def generate_windows(H: int, W: int, window_size: int, stride: int) -> List[Tuple[int, int, int, int, int]]:
    xs = _start_positions(W, window_size, stride)
    ys = _start_positions(H, window_size, stride)
    out: List[Tuple[int, int, int, int, int]] = []
    k = 0
    for y in ys:
        for x in xs:
            k += 1
            w = min(window_size, W - x)
            h = min(window_size, H - y)
            out.append((k, x, y, w, h))
    return out


def safe_ratio(num: int, den: int) -> float:
    return np.nan if den <= 0 else float(num) / float(den)


def safe_D_from_Q(Q: float) -> float:
    if Q is None or np.isnan(Q):
        return np.nan
    return float(np.clip(1.0 - Q, 0.0, 1.0))


def blur_from_masks(denom_mask: np.ndarray, white_mask: np.ndarray) -> Dict[str, float]:
    """Compute Q and D from denominator and white-paint masks in one crop."""
    denom = denom_mask > 0
    area = int(denom.sum())
    white = int(((white_mask > 0) & denom).sum())
    Q = safe_ratio(white, area)
    D = safe_D_from_Q(Q)
    return {"area": area, "white": white, "Q": Q, "D": D}


def dilate_mask(mask01: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask01.astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1).astype(np.uint8)


def metric_counts(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> Dict[str, int]:
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
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou), "pixel_acc": float(acc)}


def paired_metrics(x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute basic and optional area-weighted metrics for y compared with x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if weight is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(weight, dtype=float)[valid]
        w = np.where(np.isfinite(w) & (w > 0), w, 0)
    if len(x) == 0:
        return {
            "n": 0, "MAE": np.nan, "RMSE": np.nan, "Bias_y_minus_x": np.nan,
            "Pearson_r": np.nan, "R2": np.nan, "Within_±0.10_ratio": np.nan,
            "AreaWeighted_MAE": np.nan, "AreaWeighted_RMSE": np.nan, "AreaWeighted_Bias": np.nan,
        }

    err = y - x
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    sse = float(np.sum((y - x) ** 2))
    sst = float(np.sum((x - np.mean(x)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    within = float(np.mean(abs_err <= ERROR_BAND))

    if np.sum(w) > 0:
        w_mae = float(np.sum(w * abs_err) / np.sum(w))
        w_rmse = float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))
        w_bias = float(np.sum(w * err) / np.sum(w))
    else:
        w_mae = w_rmse = w_bias = np.nan

    return {
        "n": int(len(x)),
        "MAE": mae,
        "RMSE": rmse,
        "Bias_y_minus_x": bias,
        "Pearson_r": pearson,
        "R2": r2,
        "Within_±0.10_ratio": within,
        "AreaWeighted_MAE": w_mae,
        "AreaWeighted_RMSE": w_rmse,
        "AreaWeighted_Bias": w_bias,
    }


# ============================================================
# 5. Visualization helpers
# ============================================================

def classify_blur(D: float) -> Tuple[str, str, Tuple[int, int, int]]:
    if D is None or np.isnan(D):
        return "Missing", "Missing", (160, 160, 160)
    if D <= 0.25:
        return "Green", "Low blur", COLOR_GREEN
    if D <= 0.45:
        return "Yellow", "Moderate blur", COLOR_YELLOW
    if D <= 0.65:
        return "Orange", "High blur", COLOR_ORANGE
    return "Red", "Severe blur", COLOR_RED


def draw_multiline_box(img: np.ndarray, lines: Sequence[str], x: int = 8, y: int = 8, font_scale: float = 0.55, thickness: int = 1) -> np.ndarray:
    if not lines:
        return img
    H, W = img.shape[:2]
    sizes = [cv2.getTextSize(str(line), FONT, font_scale, thickness)[0] for line in lines]
    max_w = max(s[0] for s in sizes)
    line_h = max(s[1] for s in sizes)
    line_gap = 7
    box_w = min(W - x - 2, max_w + 18)
    box_h = min(H - y - 2, len(lines) * (line_h + line_gap) + 14)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (245, 245, 245), thickness=-1)
    img = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (60, 60, 60), thickness=1)
    cur_y = y + line_h + 7
    for line in lines:
        cv2.putText(img, str(line), (x + 8, cur_y), FONT, font_scale, COLOR_BLACK, thickness, cv2.LINE_AA)
        cur_y += line_h + line_gap
    return img


def overlay_mask_color(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    idx = mask.astype(bool)
    if not np.any(idx):
        return img.copy()
    out = img.copy().astype(np.float32)
    out[idx] = out[idx] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contour(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], thickness: int = CONTOUR_THICKNESS) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, color, thickness=thickness)
    return img


def build_dense_heatmap_from_window_rows(window_rows: List[Dict], H: int, W: int, roi_mask: np.ndarray, value_key: str, sigma: float, require_pair_valid: bool = False) -> np.ndarray:
    """Build a smooth full-image heatmap from window-level values.

    GT/Pred BD values are still computed from overlapping windows. This function only
    turns those window-level values into a smooth large-image visualization by placing
    one sample at each window center and then applying normalized Gaussian smoothing.
    """
    out = np.full((H, W), np.nan, dtype=np.float32)
    if not window_rows:
        return out

    sample_sum = np.zeros((H, W), dtype=np.float32)
    sample_count = np.zeros((H, W), dtype=np.float32)

    for r in window_rows:
        if require_pair_valid and (not bool(r.get("pair_valid", False))):
            continue
        val = r.get(value_key, np.nan)
        if val is None or np.isnan(val):
            continue
        cx = int(round(float(r["x"]) + float(r["w"]) / 2.0))
        cy = int(round(float(r["y"]) + float(r["h"]) / 2.0))
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))
        sample_sum[cy, cx] += float(val)
        sample_count[cy, cx] += 1.0

    sample_valid = sample_count > 0
    if not np.any(sample_valid):
        return out

    sample_map = np.zeros((H, W), dtype=np.float32)
    sample_map[sample_valid] = sample_sum[sample_valid] / np.maximum(sample_count[sample_valid], 1e-6)

    if sigma <= 0:
        valid = sample_valid & (roi_mask > 0)
        out[valid] = sample_map[valid]
        return out

    valid_f = sample_valid.astype(np.float32)
    num = cv2.GaussianBlur(sample_map * valid_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    den = cv2.GaussianBlur(valid_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    valid = (den > 1e-6) & (roi_mask > 0)
    out[valid] = num[valid] / np.maximum(den[valid], 1e-6)
    return out


def append_vertical_colorbar(img: np.ndarray, title: str, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Append a heatmap legend/colorbar to the right side of an overlay image."""
    H, W = img.shape[:2]
    pad = HEATMAP_COLORBAR_PAD
    bar_w = HEATMAP_COLORBAR_W
    canvas = np.full((H, W + pad + bar_w, 3), 255, dtype=np.uint8)
    canvas[:, :W] = img

    x0 = W + pad
    top = 72
    bottom = max(top + 100, H - 60)
    inner_w = 24
    bar_h = bottom - top

    grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(-1, 1)
    grad_u8 = np.repeat(np.clip(grad * 255.0, 0, 255).astype(np.uint8), inner_w, axis=1)
    bar = cv2.applyColorMap(grad_u8, cv2.COLORMAP_JET)
    canvas[top:bottom, x0:x0 + inner_w] = bar
    cv2.rectangle(canvas, (x0, top), (x0 + inner_w, bottom), (50, 50, 50), 1)

    cv2.putText(canvas, title, (x0 - 2, 22), FONT, 0.54, COLOR_BLACK, 1, cv2.LINE_AA)
    cv2.putText(canvas, "High", (x0 + 34, top + 6), FONT, 0.50, COLOR_BLACK, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Low", (x0 + 34, bottom), FONT, 0.50, COLOR_BLACK, 1, cv2.LINE_AA)

    tick_vals = [vmax, 0.75, 0.50, 0.25, vmin]
    tick_ys = [top, top + bar_h // 4, top + bar_h // 2, top + 3 * bar_h // 4, bottom]
    for tv, ty in zip(tick_vals, tick_ys):
        cv2.line(canvas, (x0 + inner_w + 2, ty), (x0 + inner_w + 8, ty), (60, 60, 60), 1)
        cv2.putText(canvas, f"{tv:.2f}", (x0 + inner_w + 12, ty + 5), FONT, 0.46, COLOR_BLACK, 1, cv2.LINE_AA)
    return canvas


def heatmap_overlay_on_mask(img: np.ndarray, value_map: np.ndarray, valid_mask: np.ndarray, alpha: float = 0.55, invert: bool = False, title: str = "Value") -> np.ndarray:
    """Create a heatmap overlay where only valid mask pixels are colored, with a colorbar legend."""
    valid = valid_mask.astype(bool) & np.isfinite(value_map)
    if not np.any(valid):
        return img.copy()

    val = np.zeros_like(value_map, dtype=np.float32)
    val[valid] = np.clip(value_map[valid], 0, 1)
    if invert:
        val[valid] = 1.0 - val[valid]

    val_u8 = np.zeros_like(value_map, dtype=np.uint8)
    val_u8[valid] = np.clip(val[valid] * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(val_u8, cv2.COLORMAP_JET)

    out = img.copy().astype(np.float32)
    out[valid] = out[valid] * (1.0 - alpha) + color_map[valid].astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = append_vertical_colorbar(out, title=title, vmin=0.0, vmax=1.0)
    return out


def build_direct_pixel_damage_heatmap(roi_mask: np.ndarray, white_mask: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build a simple full-image heatmap directly from ROI and white-paint masks.

    This is only for visualization. Window-level BD calculation is unchanged.
    Raw pixel damage inside ROI is 1 where ROI=1 and white_mask=0, otherwise 0.
    Smooth heatmap = Gaussian(raw_damage) / Gaussian(ROI).
    """
    roi = (roi_mask > 0).astype(np.float32)
    white = ((white_mask > 0) & (roi_mask > 0)).astype(np.float32)
    damage_raw = np.clip(roi - white, 0, 1).astype(np.float32)

    heat = np.full(roi.shape, np.nan, dtype=np.float32)
    valid = roi > 0
    if not np.any(valid):
        return heat, valid

    if sigma <= 0:
        heat[valid] = damage_raw[valid]
        return heat, valid

    num = cv2.GaussianBlur(damage_raw, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    den = cv2.GaussianBlur(roi, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    good = (den > 1e-6) & valid
    heat[good] = np.clip(num[good] / den[good], 0, 1)
    return heat, valid


def absolute_error_heatmap_from_direct_maps(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Absolute error map between direct GT and Pred heatmaps where both are valid."""
    err = np.full(gt_map.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(gt_map) & np.isfinite(pred_map)
    if np.any(valid):
        err[valid] = np.abs(pred_map[valid] - gt_map[valid])
    return err, valid


def make_contact_sheet(images: List[np.ndarray], labels: List[str], out_path: Path, ncols: int = 3, pad: int = 16, label_h: int = 42) -> None:
    if not images:
        return
    target_w = 380
    resized = []
    for im in images:
        h, w = im.shape[:2]
        scale = target_w / max(w, 1)
        target_h = max(1, int(round(h * scale)))
        resized.append(cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA))
    cell_w = target_w
    cell_h = max(im.shape[0] for im in resized) + label_h
    ncols = max(1, min(ncols, len(resized)))
    nrows = int(np.ceil(len(resized) / ncols))
    canvas = np.full((nrows * cell_h + (nrows + 1) * pad, ncols * cell_w + (ncols + 1) * pad, 3), 255, dtype=np.uint8)
    for idx, im in enumerate(resized):
        r = idx // ncols
        c = idx % ncols
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)
        cv2.putText(canvas, labels[idx][:58], (x0, y0 + 26), FONT, 0.55, COLOR_BLACK, 1, cv2.LINE_AA)
        canvas[y0 + label_h:y0 + label_h + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.rectangle(canvas, (x0, y0 + label_h), (x0 + cell_w, y0 + label_h + im.shape[0]), (80, 80, 80), 1)
    cv2.imwrite(str(out_path), canvas)


def make_validation_window_crop(img: np.ndarray, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray, row: Dict) -> np.ndarray:
    """Create one crop visualizing GT and proposed blur for a selected local window."""
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    crop = img[y:y + h, x:x + w].copy()
    a1 = A1[y:y + h, x:x + w]
    a2 = A2[y:y + h, x:x + w]
    a3 = A3[y:y + h, x:x + w]
    a4 = A4[y:y + h, x:x + w]

    pred_D = float(row["Pred_D"])
    gt_D = float(row["GT_All_D"])
    abs_err = float(row["absolute_error"])
    _, _, pred_color = classify_blur(pred_D)

    # Predicted local analysis unit colored by proposed D.
    crop = overlay_mask_color(crop, a3.astype(np.uint8), pred_color, alpha=0.38)
    crop = draw_contour(crop, a3.astype(np.uint8), COLOR_PRED, thickness=2)

    # GT ROI and GT white-paint pixels as contours.
    crop = draw_contour(crop, a1.astype(np.uint8), COLOR_GT, thickness=1)
    crop = draw_contour(crop, a2.astype(np.uint8), COLOR_WHITE, thickness=1)

    # R_Pred white-paint pixels as blue contour to check binarization result.
    crop = draw_contour(crop, ((a4 > 0) & (a3 > 0)).astype(np.uint8), COLOR_BLUE, thickness=1)

    lines = [
        f"GT D = {gt_D:.3f}",
        f"Pred D = {pred_D:.3f}",
        f"|Error| = {abs_err:.3f}",
        f"A1/A3 area = {int(row['GT_All_area'])}/{int(row['Pred_area'])} px",
    ]
    crop = draw_multiline_box(crop, lines, x=8, y=8, font_scale=0.55, thickness=1)
    return crop


# ============================================================
# 6. Plot helpers
# ============================================================

def set_large_plot_style() -> None:
    plt.rcParams.update({
        "font.size": PLOT_FONT_BASE,
        "axes.titlesize": PLOT_FONT_TITLE,
        "axes.labelsize": PLOT_FONT_LABEL,
        "xtick.labelsize": PLOT_FONT_TICK,
        "ytick.labelsize": PLOT_FONT_TICK,
        "legend.fontsize": PLOT_FONT_LEGEND,
        "figure.titlesize": PLOT_FONT_TITLE,
    })


def plot_global_yx_scatter(df: pd.DataFrame, out_path: Path) -> Dict[str, float]:
    """Save the main local-window Y=X scatter plot and return metrics."""
    clean = df.dropna(subset=["GT_All_D", "Pred_D", "pair_weight_area"]).copy()
    clean = clean[(clean["GT_valid"]) & (clean["Pred_valid"]) & (clean["pair_weight_area"] > 0)].copy()
    metrics = paired_metrics(clean["GT_All_D"].values, clean["Pred_D"].values, clean["pair_weight_area"].values)
    if clean.empty:
        return metrics

    x = clean["GT_All_D"].to_numpy(dtype=float)
    y = clean["Pred_D"].to_numpy(dtype=float)
    area = clean["pair_weight_area"].to_numpy(dtype=float)
    sizes = np.sqrt(area)
    if np.max(sizes) > np.min(sizes):
        sizes = 40 + (sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes)) * 260
    else:
        sizes = np.full_like(sizes, 120.0)

    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    ax.scatter(x, y, s=sizes, alpha=0.65, edgecolors="black", linewidths=0.6, label="Local windows")
    line = np.linspace(0, 1, 200)
    ax.plot(line, line, linewidth=2.4, label="Y = X")
    ax.plot(line, np.clip(line + ERROR_BAND, 0, 1), linestyle="--", linewidth=1.8, label="+0.10 band")
    ax.plot(line, np.clip(line - ERROR_BAND, 0, 1), linestyle="--", linewidth=1.8, label="-0.10 band")
    metric_text = (
        f"n = {metrics['n']}\n"
        f"MAE = {metrics['MAE']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"Bias = {metrics['Bias_y_minus_x']:.3f}\n"
        f"r = {metrics['Pearson_r']:.3f}\n"
        f"R² = {metrics['R2']:.3f}\n"
        f"Within ±0.10 = {metrics['Within_±0.10_ratio'] * 100:.1f}%\n\n"
        f"Area-W MAE = {metrics['AreaWeighted_MAE']:.3f}\n"
        f"Area-W RMSE = {metrics['AreaWeighted_RMSE']:.3f}"
    )
    ax.text(0.04, 0.96, metric_text, transform=ax.transAxes, va="top", ha="left", fontsize=PLOT_FONT_METRIC_BOX, bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))
    ax.set_title("Local-window validation: Predicted blur vs GT blur", pad=12)
    ax.set_xlabel("GT local blur damage D")
    ax.set_ylabel("Predicted local blur damage D")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    ax.legend(loc="lower right", framealpha=0.90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_error_hist(df: pd.DataFrame, out_path: Path) -> None:
    clean = df.dropna(subset=["absolute_error"]).copy()
    clean = clean[(clean["GT_valid"]) & (clean["Pred_valid"])].copy()
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    vals = clean["absolute_error"].to_numpy(dtype=float)
    ax.hist(vals, bins=min(30, max(8, int(np.sqrt(len(vals))))), edgecolor="black", linewidth=0.8)
    ax.axvline(ERROR_BAND, linestyle="--", linewidth=2.0, label="±0.10 threshold")
    ax.set_title("Absolute error distribution at local-window level", pad=12)
    ax.set_xlabel("Absolute error |D_pred - D_GT|")
    ax.set_ylabel("Number of windows")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(framealpha=0.90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_abs_error_vs_area(df: pd.DataFrame, out_path: Path) -> None:
    clean = df.dropna(subset=["absolute_error", "pair_weight_area"]).copy()
    clean = clean[(clean["GT_valid"]) & (clean["Pred_valid"]) & (clean["pair_weight_area"] > 0)].copy()
    if clean.empty:
        return
    area = clean["pair_weight_area"].to_numpy(dtype=float)
    err = clean["absolute_error"].to_numpy(dtype=float)
    sizes = np.sqrt(area)
    if np.max(sizes) > np.min(sizes):
        sizes = 40 + (sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes)) * 260
    else:
        sizes = np.full_like(sizes, 120.0)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(area, err, s=sizes, alpha=0.65, edgecolors="black", linewidths=0.6)
    if np.nanmax(area) / max(np.nanmin(area), 1) > 20:
        ax.set_xscale("log")
    ax.set_title("Absolute error vs local reference area", pad=12)
    ax.set_xlabel("Pair weight area pixels")
    ax.set_ylabel("Absolute error |D_pred - D_GT|")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)




def _bubble_sizes_from_area(area: np.ndarray) -> np.ndarray:
    sizes = np.sqrt(np.asarray(area, dtype=float))
    if len(sizes) == 0:
        return sizes
    if np.nanmax(sizes) > np.nanmin(sizes):
        return 40 + (sizes - np.nanmin(sizes)) / (np.nanmax(sizes) - np.nanmin(sizes)) * 260
    return np.full_like(sizes, 120.0)


def prepare_extra_eval_df(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Prepare clean paired rows for an extra BD-based evaluation group."""
    needed = [cfg["x_col"], cfg["y_col"], cfg["weight_col"], cfg["pair_valid_col"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {cfg['name']}: {missing}")
    clean = df.dropna(subset=[cfg["x_col"], cfg["y_col"], cfg["weight_col"]]).copy()
    clean = clean[(clean[cfg["pair_valid_col"]] == True) & (clean[cfg["weight_col"]] > 0)].copy()
    clean["eval_x"] = pd.to_numeric(clean[cfg["x_col"]], errors="coerce")
    clean["eval_y"] = pd.to_numeric(clean[cfg["y_col"]], errors="coerce")
    clean["eval_weight_area"] = pd.to_numeric(clean[cfg["weight_col"]], errors="coerce")
    clean["eval_signed_error"] = clean["eval_y"] - clean["eval_x"]
    clean["eval_absolute_error"] = clean["eval_signed_error"].abs()
    clean = clean.dropna(subset=["eval_x", "eval_y", "eval_weight_area", "eval_absolute_error"]).copy()
    return clean


def plot_extra_yx_scatter(clean: pd.DataFrame, cfg: Dict, metrics: Dict[str, float], out_path: Path) -> None:
    if clean.empty:
        return
    x = clean["eval_x"].to_numpy(dtype=float)
    y = clean["eval_y"].to_numpy(dtype=float)
    area = clean["eval_weight_area"].to_numpy(dtype=float)
    sizes = _bubble_sizes_from_area(area)

    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    ax.scatter(x, y, s=sizes, alpha=0.65, edgecolors="black", linewidths=0.6, label="Local windows")
    line = np.linspace(0, 1, 200)
    ax.plot(line, line, linewidth=2.4, label="Y = X")
    ax.plot(line, np.clip(line + ERROR_BAND, 0, 1), linestyle="--", linewidth=1.8, label="+0.10 band")
    ax.plot(line, np.clip(line - ERROR_BAND, 0, 1), linestyle="--", linewidth=1.8, label="-0.10 band")
    metric_text = (
        f"n = {metrics['n']}\n"
        f"MAE = {metrics['MAE']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"Bias = {metrics['Bias_y_minus_x']:.3f}\n"
        f"r = {metrics['Pearson_r']:.3f}\n"
        f"R² = {metrics['R2']:.3f}\n"
        f"Within ±0.10 = {metrics['Within_±0.10_ratio'] * 100:.1f}%\n\n"
        f"Area-W MAE = {metrics['AreaWeighted_MAE']:.3f}\n"
        f"Area-W RMSE = {metrics['AreaWeighted_RMSE']:.3f}"
    )
    ax.text(0.04, 0.96, metric_text, transform=ax.transAxes, va="top", ha="left", fontsize=PLOT_FONT_METRIC_BOX, bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))
    ax.set_title(cfg["title"], pad=12)
    ax.set_xlabel(cfg["x_label"])
    ax.set_ylabel(cfg["y_label"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    ax.legend(loc="lower right", framealpha=0.90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_extra_error_hist(clean: pd.DataFrame, cfg: Dict, out_path: Path) -> None:
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    vals = clean["eval_absolute_error"].to_numpy(dtype=float)
    ax.hist(vals, bins=min(30, max(8, int(np.sqrt(len(vals))))), edgecolor="black", linewidth=0.8)
    ax.axvline(ERROR_BAND, linestyle="--", linewidth=2.0, label="±0.10 threshold")
    ax.set_title(cfg["title"] + " - absolute error", pad=12)
    ax.set_xlabel(cfg["err_label"])
    ax.set_ylabel("Number of windows")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(framealpha=0.90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_extra_abs_error_vs_area(clean: pd.DataFrame, cfg: Dict, out_path: Path) -> None:
    if clean.empty:
        return
    area = clean["eval_weight_area"].to_numpy(dtype=float)
    err = clean["eval_absolute_error"].to_numpy(dtype=float)
    sizes = _bubble_sizes_from_area(area)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(area, err, s=sizes, alpha=0.65, edgecolors="black", linewidths=0.6)
    finite_area = area[np.isfinite(area) & (area > 0)]
    if finite_area.size and np.nanmax(finite_area) / max(np.nanmin(finite_area), 1) > 20:
        ax.set_xscale("log")
    ax.set_title(cfg["title"] + " - error vs area", pad=12)
    ax.set_xlabel("Pair weight area pixels")
    ax.set_ylabel(cfg["err_label"])
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_extra_evaluation_group_plots(window_df: pd.DataFrame, output_root: Path) -> None:
    """Generate the two requested extra groups, each with 3 plots + 1 metrics table."""
    extra_root = output_root / "Extra_Evaluation_Groups"
    ensure_dir(extra_root)

    configs = [
        {
            "name": "Segmentation_A2A3_vs_A2A1",
            "title": "Segmentation-effect evaluation: A2/A3 vs A2/A1",
            "interpretation": "Reference A2/A1; compared A2/A3. GT binary A2 is fixed, denominator changes from GT ROI A1 to predicted ROI A3.",
            "x_col": "GT_All_D",
            "y_col": "Seg_A2A3_D",
            "weight_col": "Seg_A2A3_pair_weight_area",
            "pair_valid_col": "Seg_A2A3_pair_valid",
            "x_label": "Reference BD: A2/A1",
            "y_label": "Segmentation-effect BD: A2/A3",
            "err_label": "Absolute error |D(A2/A3) - D(A2/A1)|",
            "formula_reference": "D_ref = 1 - |A2 ∩ A1 ∩ W| / |A1 ∩ W|",
            "formula_compared": "D_seg = 1 - |A2 ∩ A3 ∩ W| / |A3 ∩ W|",
        },
        {
            "name": "Binarization_A4A1_vs_A2A1",
            "title": "Binarization-effect evaluation: A4/A1 vs A2/A1",
            "interpretation": "Reference A2/A1; compared A4/A1. GT ROI denominator A1 is fixed, binary white mask changes from GT A2 to proposed A4.",
            "x_col": "GT_All_D",
            "y_col": "Bin_A4A1_D",
            "weight_col": "Bin_A4A1_pair_weight_area",
            "pair_valid_col": "Bin_A4A1_pair_valid",
            "x_label": "Reference BD: A2/A1",
            "y_label": "Binarization-effect BD: A4/A1",
            "err_label": "Absolute error |D(A4/A1) - D(A2/A1)|",
            "formula_reference": "D_ref = 1 - |A2 ∩ A1 ∩ W| / |A1 ∩ W|",
            "formula_compared": "D_bin = 1 - |A4 ∩ A1 ∩ W| / |A1 ∩ W|",
        },
    ]

    overview_rows = []
    for cfg in configs:
        group_dir = extra_root / cfg["name"]
        ensure_dir(group_dir)
        clean = prepare_extra_eval_df(window_df, cfg)
        metrics = paired_metrics(clean["eval_x"].values, clean["eval_y"].values, clean["eval_weight_area"].values) if not clean.empty else paired_metrics(np.array([]), np.array([]))
        metrics_row = {
            "evaluation_group": cfg["name"],
            "title": cfg["title"],
            "interpretation": cfg["interpretation"],
            "reference_column": cfg["x_col"],
            "compared_column": cfg["y_col"],
            "area_weight_column": cfg["weight_col"],
            "formula_reference": cfg["formula_reference"],
            "formula_compared": cfg["formula_compared"],
            **metrics,
        }
        pd.DataFrame([metrics_row]).to_csv(group_dir / f"{cfg['name']}_metrics_summary.csv", index=False, encoding="utf-8-sig")
        clean.to_csv(group_dir / f"{cfg['name']}_paired_window_data.csv", index=False, encoding="utf-8-sig")

        plot_extra_yx_scatter(clean, cfg, metrics, group_dir / f"{cfg['name']}_YX_scatter.png")
        plot_extra_error_hist(clean, cfg, group_dir / f"{cfg['name']}_absolute_error_hist.png")
        plot_extra_abs_error_vs_area(clean, cfg, group_dir / f"{cfg['name']}_abs_error_vs_area.png")
        overview_rows.append(metrics_row)

    pd.DataFrame(overview_rows).to_csv(extra_root / "extra_evaluation_groups_metrics_overview.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 7. Main processing
# ============================================================

def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    set_large_plot_style()

    image_dir = Path(args.image_dir)
    gt_orig_dir = Path(args.gt_orig_dir)
    gt_bin_dir = Path(args.gt_bin_dir)
    pred_orig_dir = Path(args.pred_orig_dir)
    pred_bin_dir = Path(args.pred_bin_dir)
    output_root = Path(args.output_root)

    # Output folders: keep final tables, paper-ready overlays, and global comparison plots.
    heatmap_dir = output_root / "GT_Pred_Error_Heatmap_Overlay"
    plot_dir = output_root / "Global_Comparison_Plots"
    for d in [output_root, heatmap_dir, plot_dir]:
        ensure_dir(d)

    for p, name in [
        (image_dir, "IMAGE_DIR"), (gt_orig_dir, "A1 GT_ORIG_DIR"), (gt_bin_dir, "A2 GT_BIN_DIR"),
        (pred_orig_dir, "A3 PRED_ORIG_DIR"), (pred_bin_dir, "A4 PRED_BIN_DIR"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    image_files = list_files(image_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in IMAGE_DIR: {image_dir}")

    # Random small-window crop demonstrations are disabled in the simplified comparison pipeline.
    selected_viz_parents = set()

    window_rows: List[Dict] = []
    image_summary_rows: List[Dict] = []
    pixel_rows: List[Dict] = []
    missing_rows: List[Dict] = []
    skipped_rows: List[Dict] = []
    viz_index_rows: List[Dict] = []

    for image_path in tqdm(image_files, desc="GT local-window validation"):
        parent = normalize_stem(image_path.stem)
        paths = {
            "A1": find_by_stem(gt_orig_dir, parent),
            "A2": find_by_stem(gt_bin_dir, parent),
            "A3": find_by_stem(pred_orig_dir, parent),
            "A4": find_by_stem(pred_bin_dir, parent),
        }
        if any(v is None for v in paths.values()):
            missing_rows.append({
                "parent": parent,
                "image_file": image_path.name,
                **{f"{k}_found": v is not None for k, v in paths.items()},
                **{f"{k}_path": str(v) if v else "" for k, v in paths.items()},
            })
            continue

        try:
            img = read_image_bgr(image_path)
            H, W = img.shape[:2]
            A1 = align_mask(read_mask01(paths["A1"]), H, W, args.resize_masks_to_image, "A1 GT original")
            A2 = align_mask(read_mask01(paths["A2"]), H, W, args.resize_masks_to_image, "A2 GT binary")
            A3 = align_mask(read_mask01(paths["A3"]), H, W, args.resize_masks_to_image, "A3 Pred original")
            A4 = align_mask(read_mask01(paths["A4"]), H, W, args.resize_masks_to_image, "A4 Pred binary")

            windows = generate_windows(H, W, int(args.window_size), int(args.stride))

            rows_this_parent: List[Dict] = []

            for window_id, x, y, w, h in windows:
                a1 = A1[y:y + h, x:x + w]
                a2 = A2[y:y + h, x:x + w]
                a3 = A3[y:y + h, x:x + w]
                a4 = A4[y:y + h, x:x + w]

                # Main two values for local-window validation.
                gt_all = blur_from_masks(a1, a2)                 # D_GT = 1 - |R_GT∩M_GT∩W| / |M_GT∩W|
                pred = blur_from_masks(a3, a4)                   # D_Pred = 1 - |R_Pred∩M_Pred∩W| / |M_Pred∩W|

                # Auxiliary modes, kept for error-source interpretation.
                a3_inter_a1 = ((a3 > 0) & (a1 > 0)).astype(np.uint8)
                bi_gt = blur_from_masks(a3_inter_a1, a2)         # old auxiliary: GT white, predicted denominator constrained by GT
                orig_gt = blur_from_masks(a1, a4)                # old auxiliary: predicted white, GT denominator
                pred_ref = blur_from_masks(a3_inter_a1, a4)      # old auxiliary: predicted white, predicted denominator constrained by GT

                # Extra diagnostic evaluation 1: segmentation effect.
                # Reference = R_GT/M_GT; Compared = R_GT/M_Pred.
                # The binary/white mask is fixed to GT R_GT, while the denominator changes from M_GT to M_Pred.
                # This mainly reflects the impact of predicted ROI/segmentation on BD estimation.
                seg_a2a3 = blur_from_masks(a3, a2)

                # Extra diagnostic evaluation 2: binarization effect.
                # Reference = R_GT/M_GT; Compared = R_Pred/M_GT.
                # The denominator is fixed to GT ROI M_GT, while the binary white mask changes from R_GT to R_Pred.
                # This mainly reflects the impact of the proposed binarization result.
                bin_a4a1 = blur_from_masks(a1, a4)

                gt_valid = bool(gt_all["area"] >= int(args.min_gt_unit_area))
                pred_valid = bool(pred["area"] >= int(args.min_pred_unit_area))
                seg_valid = bool(seg_a2a3["area"] >= int(args.min_pred_unit_area))
                bin_valid = bool(bin_a4a1["area"] >= int(args.min_gt_unit_area))
                pair_valid = bool(gt_valid and pred_valid and min(gt_all["area"], pred["area"]) >= int(args.min_pair_area))
                seg_pair_valid = bool(gt_valid and seg_valid and min(gt_all["area"], seg_a2a3["area"]) >= int(args.min_pair_area))
                bin_pair_valid = bool(gt_valid and bin_valid and min(gt_all["area"], bin_a4a1["area"]) >= int(args.min_pair_area))

                if not (gt_valid or pred_valid):
                    # Skip pure-background local windows from the CSV to keep outputs compact.
                    continue

                signed_error = float(pred["D"] - gt_all["D"]) if pair_valid else np.nan
                abs_error = abs(signed_error) if pair_valid else np.nan
                pair_weight_area = int(min(gt_all["area"], pred["area"])) if pair_valid else 0

                seg_signed_error = float(seg_a2a3["D"] - gt_all["D"]) if seg_pair_valid else np.nan
                seg_abs_error = abs(seg_signed_error) if seg_pair_valid else np.nan
                seg_pair_weight_area = int(min(gt_all["area"], seg_a2a3["area"])) if seg_pair_valid else 0

                bin_signed_error = float(bin_a4a1["D"] - gt_all["D"]) if bin_pair_valid else np.nan
                bin_abs_error = abs(bin_signed_error) if bin_pair_valid else np.nan
                bin_pair_weight_area = int(min(gt_all["area"], bin_a4a1["area"])) if bin_pair_valid else 0

                gt_level, gt_desc, _ = classify_blur(float(gt_all["D"]) if gt_valid else np.nan)
                pred_level, pred_desc, _ = classify_blur(float(pred["D"]) if pred_valid else np.nan)

                row = {
                    "parent": parent,
                    "image_name": image_path.stem,
                    "window_id": int(window_id),
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "window_area": int(w * h),
                    "GT_valid": gt_valid,
                    "Pred_valid": pred_valid,
                    "pair_valid": pair_valid,
                    "pair_weight_area": int(pair_weight_area),
                    "GT_All_area": int(gt_all["area"]),
                    "GT_All_white": int(gt_all["white"]),
                    "GT_All_Q": float(gt_all["Q"]) if not np.isnan(gt_all["Q"]) else np.nan,
                    "GT_All_D": float(gt_all["D"]) if not np.isnan(gt_all["D"]) else np.nan,
                    "GT_blur_level": gt_level,
                    "GT_blur_description": gt_desc,
                    "Pred_area": int(pred["area"]),
                    "Pred_white": int(pred["white"]),
                    "Pred_Q": float(pred["Q"]) if not np.isnan(pred["Q"]) else np.nan,
                    "Pred_D": float(pred["D"]) if not np.isnan(pred["D"]) else np.nan,
                    "Pred_blur_level": pred_level,
                    "Pred_blur_description": pred_desc,
                    "signed_error_Pred_minus_GT": signed_error,
                    "absolute_error": abs_error,
                    "Bi_GT_area": int(bi_gt["area"]),
                    "Bi_GT_Q": float(bi_gt["Q"]) if not np.isnan(bi_gt["Q"]) else np.nan,
                    "Bi_GT_D": float(bi_gt["D"]) if not np.isnan(bi_gt["D"]) else np.nan,
                    "Orig_GT_area": int(orig_gt["area"]),
                    "Orig_GT_Q": float(orig_gt["Q"]) if not np.isnan(orig_gt["Q"]) else np.nan,
                    "Orig_GT_D": float(orig_gt["D"]) if not np.isnan(orig_gt["D"]) else np.nan,
                    "Pred_Ref_area": int(pred_ref["area"]),
                    "Pred_Ref_Q": float(pred_ref["Q"]) if not np.isnan(pred_ref["Q"]) else np.nan,
                    "Pred_Ref_D": float(pred_ref["D"]) if not np.isnan(pred_ref["D"]) else np.nan,

                    # Extra evaluation group: segmentation effect, R_GT/M_GT vs R_GT/M_Pred.
                    "Seg_A2A3_valid": seg_valid,
                    "Seg_A2A3_pair_valid": seg_pair_valid,
                    "Seg_A2A3_pair_weight_area": int(seg_pair_weight_area),
                    "Seg_A2A3_area": int(seg_a2a3["area"]),
                    "Seg_A2A3_white": int(seg_a2a3["white"]),
                    "Seg_A2A3_Q": float(seg_a2a3["Q"]) if not np.isnan(seg_a2a3["Q"]) else np.nan,
                    "Seg_A2A3_D": float(seg_a2a3["D"]) if not np.isnan(seg_a2a3["D"]) else np.nan,
                    "Seg_A2A3_signed_error_minus_GT": seg_signed_error,
                    "Seg_A2A3_absolute_error": seg_abs_error,

                    # Extra evaluation group: binarization effect, R_GT/M_GT vs R_Pred/M_GT.
                    "Bin_A4A1_valid": bin_valid,
                    "Bin_A4A1_pair_valid": bin_pair_valid,
                    "Bin_A4A1_pair_weight_area": int(bin_pair_weight_area),
                    "Bin_A4A1_area": int(bin_a4a1["area"]),
                    "Bin_A4A1_white": int(bin_a4a1["white"]),
                    "Bin_A4A1_Q": float(bin_a4a1["Q"]) if not np.isnan(bin_a4a1["Q"]) else np.nan,
                    "Bin_A4A1_D": float(bin_a4a1["D"]) if not np.isnan(bin_a4a1["D"]) else np.nan,
                    "Bin_A4A1_signed_error_minus_GT": bin_signed_error,
                    "Bin_A4A1_absolute_error": bin_abs_error,
                    "window_size": int(args.window_size),
                    "stride": int(args.stride),
                    "A1_file": paths["A1"].name,
                    "A2_file": paths["A2"].name,
                    "A3_file": paths["A3"].name,
                    "A4_file": paths["A4"].name,
                }
                window_rows.append(row)
                rows_this_parent.append(row)

            # Per-image window-level metrics.
            parent_df = pd.DataFrame(rows_this_parent)
            if not parent_df.empty:
                pair_df = parent_df[parent_df["pair_valid"] == True].copy()
                m = paired_metrics(pair_df["GT_All_D"].values, pair_df["Pred_D"].values, pair_df["pair_weight_area"].values) if not pair_df.empty else paired_metrics(np.array([]), np.array([]))
                image_summary_rows.append({
                    "parent": parent,
                    "image_name": image_path.stem,
                    "height": int(H), "width": int(W),
                    "total_windows_generated": int(len(windows)),
                    "kept_window_count": int(len(parent_df)),
                    "paired_valid_window_count": int(len(pair_df)),
                    "GT_valid_window_count": int(parent_df["GT_valid"].sum()),
                    "Pred_valid_window_count": int(parent_df["Pred_valid"].sum()),
                    "window_size": int(args.window_size),
                    "stride": int(args.stride),
                    "min_gt_unit_area": int(args.min_gt_unit_area),
                    "min_pred_unit_area": int(args.min_pred_unit_area),
                    "min_pair_area": int(args.min_pair_area),
                    "MAE_Pred_vs_GT": m["MAE"],
                    "RMSE_Pred_vs_GT": m["RMSE"],
                    "Bias_Pred_minus_GT": m["Bias_y_minus_x"],
                    "Pearson_r": m["Pearson_r"],
                    "R2": m["R2"],
                    "Within_±0.10_ratio": m["Within_±0.10_ratio"],
                    "AreaWeighted_MAE": m["AreaWeighted_MAE"],
                    "AreaWeighted_RMSE": m["AreaWeighted_RMSE"],
                    "AreaWeighted_Bias": m["AreaWeighted_Bias"],
                    "mean_GT_D_paired": float(pair_df["GT_All_D"].mean()) if not pair_df.empty else np.nan,
                    "mean_Pred_D_paired": float(pair_df["Pred_D"].mean()) if not pair_df.empty else np.nan,
                })

            # Pixel-level validation rows, similar to the old Script 04 but compact.
            A4_tol = dilate_mask(A4, int(args.tolerance_ksize))
            pixel_configs = [
                {"eval_name": "A4_vs_A2_within_A1", "pred": A4, "pred_tol": A4_tol, "gt": A2, "region": A1, "region_name": "A1"},
                {"eval_name": "A4_vs_A2_within_A3_and_A1", "pred": A4, "pred_tol": A4_tol, "gt": A2, "region": ((A3 > 0) & (A1 > 0)).astype(np.uint8), "region_name": "A3_inter_A1"},
                {"eval_name": "A3_vs_A1_segmentation", "pred": A3, "pred_tol": A3, "gt": A1, "region": np.ones_like(A1, dtype=np.uint8), "region_name": "whole_image"},
            ]
            for cfg in pixel_configs:
                for variant, pred_use in [("strict", cfg["pred"]), ("tolerant", cfg["pred_tol"] )]:
                    if cfg["eval_name"] == "A3_vs_A1_segmentation" and variant == "tolerant":
                        continue
                    counts = metric_counts(pred_use, cfg["gt"], cfg["region"])
                    pixel_rows.append({
                        "parent": parent,
                        "eval_name": cfg["eval_name"],
                        "variant": variant,
                        "region_name": cfg["region_name"],
                        "tolerance_ksize": int(args.tolerance_ksize) if variant == "tolerant" else 1,
                        **counts,
                        **metrics_from_counts(counts),
                    })

            # Save GT, Pred, and Error heatmaps for this image.
            # Window-level BD values are unchanged and remain the basis of CSV/evaluation.
            # The heatmaps below are direct full-image pixel-damage maps:
            #   GT heat   = smoothed(M_GT - R_GT) inside M_GT
            #   Pred heat = smoothed(M_Pred - R_Pred) inside M_Pred
            gt_map, gt_valid_mask = build_direct_pixel_damage_heatmap(
                roi_mask=A1,
                white_mask=A2,
                sigma=float(args.heatmap_smooth_sigma),
            )
            pred_map, pred_valid_mask = build_direct_pixel_damage_heatmap(
                roi_mask=A3,
                white_mask=A4,
                sigma=float(args.heatmap_smooth_sigma),
            )
            err_map, err_valid_mask = absolute_error_heatmap_from_direct_maps(gt_map, pred_map)

            if (not args.no_heatmap) and SAVE_HEATMAP_FOR_ALL:
                gt_mean_window_D = float(parent_df["GT_All_D"].dropna().mean()) if (not parent_df.empty and "GT_All_D" in parent_df.columns and parent_df["GT_All_D"].notna().any()) else np.nan
                pred_mean_window_D = float(parent_df["Pred_D"].dropna().mean()) if (not parent_df.empty and "Pred_D" in parent_df.columns and parent_df["Pred_D"].notna().any()) else np.nan
                mean_abs_error = float(parent_df.loc[parent_df["pair_valid"] == True, "absolute_error"].dropna().mean()) if (not parent_df.empty and "absolute_error" in parent_df.columns and (parent_df["pair_valid"] == True).any()) else np.nan

                gt_overlay = heatmap_overlay_on_mask(img, gt_map, gt_valid_mask, alpha=0.55, title="GT Blur D")
                pred_overlay = heatmap_overlay_on_mask(img, pred_map, pred_valid_mask, alpha=0.55, title="Pred Blur D")
                err_overlay = heatmap_overlay_on_mask(img, err_map, err_valid_mask, alpha=0.60, title="|Pred-GT| Error")
                gt_overlay = draw_multiline_box(gt_overlay, [
                    "GT blur heatmap",
                    "Heatmap = smoothed pixel damage (A1 - A2)",
                    "BD calculation = overlapping windows",
                    f"Mean GT BD(window) = {gt_mean_window_D:.3f}" if not np.isnan(gt_mean_window_D) else "Mean GT BD(window) = N/A",
                    f"Heatmap sigma = {float(args.heatmap_smooth_sigma):.1f}px" if float(args.heatmap_smooth_sigma) > 0 else "Heatmap smoothing: off",
                ], x=10, y=10, font_scale=0.70, thickness=2)
                pred_overlay = draw_multiline_box(pred_overlay, [
                    "Pred blur heatmap",
                    "Heatmap = smoothed pixel damage (A3 - A4)",
                    "BD calculation = overlapping windows",
                    f"Mean Pred BD(window) = {pred_mean_window_D:.3f}" if not np.isnan(pred_mean_window_D) else "Mean Pred BD(window) = N/A",
                    f"Heatmap sigma = {float(args.heatmap_smooth_sigma):.1f}px" if float(args.heatmap_smooth_sigma) > 0 else "Heatmap smoothing: off",
                ], x=10, y=10, font_scale=0.70, thickness=2)
                err_overlay = draw_multiline_box(err_overlay, [
                    "Absolute error heatmap",
                    "Error = |Pred heat - GT heat|",
                    "Window metrics are saved separately",
                    f"Mean window |error| = {mean_abs_error:.3f}" if not np.isnan(mean_abs_error) else "Mean window |error| = N/A",
                    f"Heatmap sigma = {float(args.heatmap_smooth_sigma):.1f}px" if float(args.heatmap_smooth_sigma) > 0 else "Heatmap smoothing: off",
                ], x=10, y=10, font_scale=0.70, thickness=2)
                cv2.imwrite(str(heatmap_dir / f"{parent}_GT_blur_heatmap_overlay.png"), gt_overlay)
                cv2.imwrite(str(heatmap_dir / f"{parent}_Pred_blur_heatmap_overlay.png"), pred_overlay)
                cv2.imwrite(str(heatmap_dir / f"{parent}_Abs_error_heatmap_overlay.png"), err_overlay)

        except Exception as e:
            skipped_rows.append({"parent": parent, "image_file": image_path.name, "reason": repr(e)})
            print(f"[WARN] skipped {parent}: {e}")

    # Save CSV outputs.
    window_df = pd.DataFrame(window_rows)
    image_df = pd.DataFrame(image_summary_rows)
    pixel_df = pd.DataFrame(pixel_rows)
    window_df.to_csv(output_root / "window_level_validation.csv", index=False, encoding="utf-8-sig")
    image_df.to_csv(output_root / "image_level_window_validation_summary.csv", index=False, encoding="utf-8-sig")
    pixel_df.to_csv(output_root / "pixel_metrics_by_image.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(output_root / "missing_pairs_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_rows).to_csv(output_root / "skipped_report.csv", index=False, encoding="utf-8-sig")

    # Global pixel metrics.
    if not pixel_df.empty:
        global_rows = []
        for (eval_name, variant), sub in pixel_df.groupby(["eval_name", "variant"]):
            counts = {k: int(sub[k].sum()) for k in ["TP", "FP", "FN", "TN", "region_pixels", "pred_positive_pixels", "gt_positive_pixels"]}
            global_rows.append({"eval_name": eval_name, "variant": variant, **counts, **metrics_from_counts(counts)})
        pd.DataFrame(global_rows).to_csv(output_root / "pixel_metrics_global_summary.csv", index=False, encoding="utf-8-sig")

    # Global comparison plots and metrics.
    if SAVE_COMPARISON_PLOTS and not args.no_comparison_plots and not window_df.empty:
        ensure_dir(plot_dir)
        global_metrics = plot_global_yx_scatter(window_df, plot_dir / "Pred_vs_GT_local_window_YX_scatter.png")
        plot_error_hist(window_df, plot_dir / "Pred_vs_GT_local_window_absolute_error_hist.png")
        plot_abs_error_vs_area(window_df, plot_dir / "Pred_vs_GT_local_window_abs_error_vs_area.png")
        pd.DataFrame([global_metrics]).to_csv(plot_dir / "Pred_vs_GT_local_window_metrics_summary.csv", index=False, encoding="utf-8-sig")

        # Additional requested diagnostic groups.
        # Each group outputs exactly 3 figures + 1 metrics summary table, plus a paired-data CSV for traceability.
        generate_extra_evaluation_group_plots(window_df, output_root)

    # Save run config.
    with open(output_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "gt_validation_local_overlapping_windows_A1_A2_A3_A4",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "IMAGE_DIR": str(image_dir),
            "A1_GT_ORIG_DIR": str(gt_orig_dir),
            "A2_GT_BIN_DIR": str(gt_bin_dir),
            "A3_PRED_ORIG_DIR": str(pred_orig_dir),
            "A4_PRED_BIN_DIR": str(pred_bin_dir),
            "OUTPUT_ROOT": str(output_root),
            "WINDOW_SIZE": int(args.window_size),
            "STRIDE": int(args.stride),
            "MIN_GT_UNIT_AREA": int(args.min_gt_unit_area),
            "MIN_PRED_UNIT_AREA": int(args.min_pred_unit_area),
            "MIN_PAIR_AREA": int(args.min_pair_area),
            "TOLERANCE_KSIZE": int(args.tolerance_ksize),
            "FORMULA_GT": "U_GT = A1 ∩ W_k ; D_GT = 1 - |A2 ∩ U_GT| / |U_GT|",
            "FORMULA_PRED": "U_Pred = A3 ∩ W_k ; D_Pred = 1 - |A4 ∩ U_Pred| / |U_Pred|",
            "FORMULA_SEGMENTATION_EFFECT": "D_seg = 1 - |A2 ∩ A3 ∩ W_k| / |A3 ∩ W_k| ; compared with D_GT = A2/A1",
            "FORMULA_BINARIZATION_EFFECT": "D_bin = 1 - |A4 ∩ A1 ∩ W_k| / |A1 ∩ W_k| ; compared with D_GT = A2/A1",
            "ERROR_BAND": float(ERROR_BAND),
            "RESIZE_MASKS_TO_IMAGE": bool(args.resize_masks_to_image),
            "RANDOM_SEED": int(args.random_seed),
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Window validation CSV:", output_root / "window_level_validation.csv")
    print("Image summary CSV     :", output_root / "image_level_window_validation_summary.csv")
    print("Global plots          :", plot_dir)
    print("Extra evaluation groups:", output_root / "Extra_Evaluation_Groups")


if __name__ == "__main__":
    main()


