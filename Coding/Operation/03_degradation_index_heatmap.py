# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

"""
03. Local-window blur calculation from M_Pred Pred_Mask and R_Pred Binary/Final mask
================================================================================

Purpose
-------
This script replaces the previous connected-component/object-level blur calculation
with a local-analysis-window workflow.

Why this version?
-----------------
Parking-marking wear is usually spatially heterogeneous: one part of a marking can be
clear while another part is heavily worn. Therefore, forcing each connected component
in M_Pred to become one "object" may over-average the local deterioration. This script
instead defines fixed-size overlapping windows on the stitched full-size image.

Inputs
------
IMAGE_DIR      : Full-size RGB images from Script 01.
PRED_MASK_DIR  : M_Pred = full-size predicted marking ROI mask from Script 01.
BINARY_MASK_DIR: R_Pred = full-size final white-paint/binary mask from Script 02.
                 In the old script this was called PRED_BIN_DIR.

Local analysis unit
-------------------
For each sliding window W_k:

    U_k = M_Pred ∩ W_k

Only valid windows with enough predicted marking pixels are used. The proposed local
blur damage is:

    Q_k^pred = |R_Pred ∩ M_Pred ∩ W_k| / |M_Pred ∩ W_k|
    D_k^pred = 1 - Q_k^pred

where:
    Q = remaining white-paint ratio
    D = blur damage / missing-white-paint ratio

Main outputs
------------
1) window_level_blur_scores.csv
   One row per valid overlapping local analysis window.

2) image_level_blur_summary.csv
   One row per full-size image, including area-weighted local blur and unique-image blur.

3) Pred_Blur_Map_NPY / Pred_Blur_Heatmap_Overlay
   Spatially resolved predicted blur map. The heatmap is drawn only on M_Pred ROI pixels.

4) Random_Window_Overlay_Groups and Random_Window_Overlay_Crops
   Only randomly selected 2 parent images are used for window demonstration by default.
   This avoids saving thousands of small-window visualizations.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# 0. Default paths
# ------------------------------------------------------------
# You can edit these directly, or override them by command line.
# Example:
#   python 03_blur_from_A3_A4_LOCAL_WINDOWS.py \
#       --image-dir "...\\Image" \
#       --pred-mask-dir "...\\Pred_Mask" \
#       --binary-mask-dir "...\\03_final_mask" \
#       --output-dir "...\\Blur_Local_Window"
# ============================================================

IMAGE_DIR = Path(r"")  # TODO: set path
PRED_MASK_DIR = Path(r"")  # M_Pred
BINARY_MASK_DIR = Path(r"")  # R_Pred
OUTPUT_DIR = Path(r"")  # TODO: set path

# ============================================================  
# 1. Local-window parameters
# ============================================================
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
MASK_THRESHOLD = 127
RESIZE_MASKS_TO_IMAGE = False

# Recommended starting point for your current full-size stitched masks.
# 512/256 means 50% overlap, smoother than non-overlapping 512 tiles.
WINDOW_SIZE = 512
STRIDE = 256

# A window is valid only when it contains enough M_Pred pixels.
# This avoids unstable D values from tiny fragments at window corners.
MIN_UNIT_AREA = 500
MIN_UNIT_AREA_RATIO = 0.0   # Optional: e.g. 0.001 means >=0.1% of window area.

# Visualization settings.
RANDOM_SEED = 42
NUM_RANDOM_VIZ_PARENTS = 0
VIZ_WINDOWS_PER_PARENT = 12
SAVE_HEATMAP_FOR_ALL = True
OVERLAY_ALPHA = 0.45
HEATMAP_SMOOTH_SIGMA = 55.0   # Visualization only; BD calculation remains window-based.
HEATMAP_COLORBAR_W = 95
HEATMAP_COLORBAR_PAD = 14
CONTOUR_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors for OpenCV.
COLOR_GREEN = (144, 238, 144)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# ============================================================
# 2. Argument parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate local-window proposed blur damage from full-size A3 Pred_Mask and A4 Binary/Final mask."
    )
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR), help="Folder containing full-size RGB images.")
    parser.add_argument("--pred-mask-dir", type=str, default=str(PRED_MASK_DIR), help="A3 full-size predicted ROI mask folder.")
    # Both option names point to the same destination, so old and new wording are both supported.
    parser.add_argument("--binary-mask-dir", "--pred-bin-dir", dest="binary_mask_dir", type=str, default=str(BINARY_MASK_DIR), help="A4 final binary/white-paint mask folder.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output folder.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Sliding window size in pixels, usually 512.")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Sliding window stride in pixels, e.g. 256 for 50% overlap.")
    parser.add_argument("--min-unit-area", type=int, default=MIN_UNIT_AREA, help="Minimum A3 pixels in a window to calculate local blur.")
    parser.add_argument("--min-unit-area-ratio", type=float, default=MIN_UNIT_AREA_RATIO, help="Optional minimum A3/window-area ratio.")
    parser.add_argument("--resize-masks-to-image", action="store_true", default=RESIZE_MASKS_TO_IMAGE, help="Resize masks to image size if shapes mismatch.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-random-viz-parents", type=int, default=NUM_RANDOM_VIZ_PARENTS, help="How many parent images are randomly selected for window crop demonstration.")
    parser.add_argument("--viz-windows-per-parent", type=int, default=VIZ_WINDOWS_PER_PARENT, help="How many valid windows are visualized for each selected parent image.")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap overlay saving for all images.")
    parser.add_argument("--heatmap-smooth-sigma", type=float, default=HEATMAP_SMOOTH_SIGMA, help="Gaussian smoothing sigma in pixels for direct A3-A4 pixel-damage heatmap only. Set <=0 to disable smoothing.")
    return parser.parse_args()


# ============================================================
# 3. Basic file and image helpers
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
    """Normalize file stems from different stages.

    Script 02 saves both parent.png and parent__final_mask.png. This function removes
    common suffixes so M_Pred/R_Pred/Image can still be paired by the same parent name.
    """
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
    """Find an image/mask file by normalized parent stem."""
    if folder is None or not folder.exists():
        return None
    norm = normalize_stem(stem)

    # Fast exact candidates first.
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

    # Slower fallback: normalize every file in the folder.
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
# 4. Sliding-window and blur calculation helpers
# ============================================================

def _start_positions(length: int, window_size: int, stride: int) -> List[int]:
    """Generate start positions and force coverage of the right/bottom border.

    If length is not exactly divisible by stride, the last window is shifted so the
    final image border is still covered.
    """
    if length <= window_size:
        return [0]
    starts = list(range(0, max(length - window_size + 1, 1), stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(int(s) for s in starts))


def generate_windows(H: int, W: int, window_size: int, stride: int) -> List[Tuple[int, int, int, int, int]]:
    """Return windows as (window_id, x, y, w, h)."""
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


def classify_blur(D: float) -> Tuple[str, str, Tuple[int, int, int]]:
    """Classify blur damage for visualization. D is 0~1, larger means more damaged."""
    if D <= 0.25:
        return "Green", "Low blur", COLOR_GREEN
    if D <= 0.45:
        return "Yellow", "Moderate blur", COLOR_YELLOW
    if D <= 0.65:
        return "Orange", "High blur", COLOR_ORANGE
    return "Red", "Severe blur", COLOR_RED


def safe_ratio(num: int, den: int) -> float:
    return np.nan if den <= 0 else float(num) / float(den)


def compute_window_blur(pred_crop: np.ndarray, binary_crop: np.ndarray) -> Dict[str, float]:
    """Compute proposed blur damage in one local analysis window.

    Denominator: M_Pred ∩ W_k
    Numerator  : R_Pred ∩ M_Pred ∩ W_k
    """
    unit = pred_crop > 0
    area = int(unit.sum())
    white = int(((binary_crop > 0) & unit).sum())
    Q = safe_ratio(white, area)
    D = np.nan if np.isnan(Q) else float(np.clip(1.0 - Q, 0.0, 1.0))
    return {"area": area, "white": white, "Q": Q, "D": D}


# ============================================================
# 5. Visualization helpers
# ============================================================

def draw_multiline_box(img: np.ndarray, lines: Sequence[str], x: int = 8, y: int = 8, font_scale: float = 0.55, thickness: int = 1) -> np.ndarray:
    """Draw a white semi-transparent text box at top-left of a crop or image."""
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
    """Overlay one binary mask using a fixed BGR color."""
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


def build_dense_heatmap_from_window_rows(window_rows: List[Dict], H: int, W: int, roi_mask: np.ndarray, value_key: str, sigma: float) -> np.ndarray:
    """Build a smooth full-image heatmap from window-level values.

    BD is still calculated by overlapping windows. For visualization only, each
    valid window contributes one sample at its center, then normalized Gaussian
    smoothing converts the window samples into a smooth large-image heatmap.
    """
    out = np.full((H, W), np.nan, dtype=np.float32)
    if not window_rows:
        return out

    sample_sum = np.zeros((H, W), dtype=np.float32)
    sample_count = np.zeros((H, W), dtype=np.float32)

    for r in window_rows:
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


def heatmap_overlay_on_roi(img: np.ndarray, blur_map: np.ndarray, valid_mask: np.ndarray, alpha: float = 0.55, title: str = "Blur Damage D") -> np.ndarray:
    """Create a heatmap overlay where only valid ROI pixels are colored, with a colorbar legend."""
    valid = valid_mask.astype(bool) & np.isfinite(blur_map)
    if not np.any(valid):
        return img.copy()

    value_u8 = np.zeros_like(blur_map, dtype=np.uint8)
    value_u8[valid] = np.clip(blur_map[valid] * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(value_u8, cv2.COLORMAP_JET)

    out = img.copy().astype(np.float32)
    out[valid] = out[valid] * (1.0 - alpha) + color_map[valid].astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = append_vertical_colorbar(out, title=title, vmin=0.0, vmax=1.0)
    return out


def build_direct_pixel_damage_heatmap(roi_mask: np.ndarray, white_mask: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build a simple full-image heatmap directly from M_Pred and R_Pred.

    This is only for visualization. Window-level BD calculation is unchanged.

    Raw pixel damage inside M_Pred:
      1 = M_Pred pixel without R_Pred white-paint pixel
      0 = M_Pred pixel with R_Pred white-paint pixel

    Smooth heatmap:
      heat = Gaussian(raw_damage) / Gaussian(M_Pred ROI)
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


def make_contact_sheet(images: List[np.ndarray], labels: List[str], out_path: Path, ncols: int = 3, pad: int = 16, label_h: int = 42) -> None:
    """Save a simple contact sheet for random window crops."""
    if not images:
        return
    target_w = 360
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
        label = labels[idx]
        cv2.putText(canvas, label[:54], (x0, y0 + 26), FONT, 0.55, COLOR_BLACK, 1, cv2.LINE_AA)
        canvas[y0 + label_h:y0 + label_h + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.rectangle(canvas, (x0, y0 + label_h), (x0 + cell_w, y0 + label_h + im.shape[0]), (80, 80, 80), 1)

    cv2.imwrite(str(out_path), canvas)


def make_window_crop_overlay(img: np.ndarray, pred: np.ndarray, binary: np.ndarray, win_row: Dict) -> np.ndarray:
    """Create one small crop overlay for a selected local analysis window."""
    x, y, w, h = int(win_row["x"]), int(win_row["y"]), int(win_row["w"]), int(win_row["h"])
    D = float(win_row["blur_damage_D"])
    level, desc, color = classify_blur(D)

    crop = img[y:y + h, x:x + w].copy()
    pred_crop = pred[y:y + h, x:x + w]
    binary_crop = binary[y:y + h, x:x + w]

    # Color the whole local analysis unit U_k = M_Pred ∩ W_k by its blur level.
    unit = pred_crop > 0
    crop = overlay_mask_color(crop, unit.astype(np.uint8), color, alpha=0.42)
    crop = draw_contour(crop, unit.astype(np.uint8), color, thickness=2)

    # Draw R_Pred white-paint pixels with a thin white contour. This helps visually check
    # why D is high or low inside the selected local window.
    white_in_unit = ((binary_crop > 0) & unit).astype(np.uint8)
    crop = draw_contour(crop, white_in_unit, COLOR_WHITE, thickness=1)

    lines = [
        f"Blur Damage D = {D:.3f}",
        f"Q white ratio = {float(win_row['white_ratio_Q']):.3f}",
        f"A3 area = {int(win_row['unit_area_A3'])} px",
        f"Level = {level}",
    ]
    crop = draw_multiline_box(crop, lines, x=8, y=8, font_scale=0.55, thickness=1)
    return crop


# ============================================================
# 6. Main processing
# ============================================================

def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    image_dir = Path(args.image_dir)
    pred_mask_dir = Path(args.pred_mask_dir)
    binary_mask_dir = Path(args.binary_mask_dir)
    output_dir = Path(args.output_dir)

    # Output folders: keep final tables and paper-ready heatmap overlays only.
    heatmap_overlay_dir = output_dir / "Pred_Blur_Heatmap_Overlay"
    for d in [output_dir, heatmap_overlay_dir]:
        ensure_dir(d)

    for p, name in [(image_dir, "IMAGE_DIR"), (pred_mask_dir, "PRED_MASK_DIR/A3"), (binary_mask_dir, "BINARY_MASK_DIR/A4")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    image_files = list_files(image_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in IMAGE_DIR: {image_dir}")

    # Random window crop demonstrations are disabled in the simplified comparison pipeline.
    selected_viz_parents = set()

    window_rows: List[Dict] = []
    image_rows: List[Dict] = []
    missing_rows: List[Dict] = []
    skipped_rows: List[Dict] = []
    viz_index_rows: List[Dict] = []

    for image_path in tqdm(image_files, desc="Local-window proposed blur from A3/A4"):
        parent = normalize_stem(image_path.stem)
        pred_path = find_by_stem(pred_mask_dir, parent)
        binary_path = find_by_stem(binary_mask_dir, parent)

        if pred_path is None or binary_path is None:
            missing_rows.append({
                "parent": parent,
                "image_file": image_path.name,
                "A3_pred_found": pred_path is not None,
                "A4_binary_found": binary_path is not None,
            })
            continue

        try:
            img = read_image_bgr(image_path)
            H, W = img.shape[:2]
            pred = align_mask(read_mask01(pred_path), H, W, args.resize_masks_to_image, "A3 pred mask")
            binary = align_mask(read_mask01(binary_path), H, W, args.resize_masks_to_image, "A4 binary/final mask")

            windows = generate_windows(H, W, int(args.window_size), int(args.stride))

            # For overlapping windows, a pixel may be covered by multiple windows.
            # We average the D values assigned to that pixel to obtain a smoother blur map.
            heat_sum = np.zeros((H, W), dtype=np.float32)
            heat_count = np.zeros((H, W), dtype=np.float32)

            valid_rows_this_parent: List[Dict] = []
            total_window_area = 0
            total_window_white = 0
            D_list = []
            class_count = {"Green": 0, "Yellow": 0, "Orange": 0, "Red": 0}

            for window_id, x, y, w, h in windows:
                pred_crop = pred[y:y + h, x:x + w]
                binary_crop = binary[y:y + h, x:x + w]
                res = compute_window_blur(pred_crop, binary_crop)
                area = int(res["area"])
                win_area = int(w * h)

                # Valid-unit filtering: skip windows with too few marking pixels.
                if area < int(args.min_unit_area):
                    continue
                if float(area) / max(win_area, 1) < float(args.min_unit_area_ratio):
                    continue

                Q = float(res["Q"])
                D = float(res["D"])
                level, desc, _ = classify_blur(D)
                class_count[level] += 1
                total_window_area += area
                total_window_white += int(res["white"])
                D_list.append(D)

                row = {
                    "parent": parent,
                    "image_name": image_path.stem,
                    "window_id": int(window_id),
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "window_area": int(win_area),
                    "unit_definition": "A3_intersect_window",
                    "unit_area_A3": int(area),
                    "white_pixels_A4_in_A3_window": int(res["white"]),
                    "white_ratio_Q": Q,
                    "blur_damage_D": D,
                    "blur_level": level,
                    "blur_description": desc,
                    "A3_pred_mask_file": pred_path.name,
                    "A4_binary_mask_file": binary_path.name,
                    "window_size": int(args.window_size),
                    "stride": int(args.stride),
                    "min_unit_area": int(args.min_unit_area),
                }
                window_rows.append(row)
                valid_rows_this_parent.append(row)

                # Fill heatmap only on predicted marking ROI pixels inside this window.
                unit = pred_crop > 0
                heat_sum[y:y + h, x:x + w][unit] += D
                heat_count[y:y + h, x:x + w][unit] += 1.0

            # Image-level summary.
            image_Q_local_weighted = safe_ratio(total_window_white, total_window_area)
            image_D_local_weighted = np.nan if np.isnan(image_Q_local_weighted) else float(1.0 - image_Q_local_weighted)
            unique_area = int((pred > 0).sum())
            unique_white = int(((binary > 0) & (pred > 0)).sum())
            unique_Q = safe_ratio(unique_white, unique_area)
            unique_D = np.nan if np.isnan(unique_Q) else float(1.0 - unique_Q)
            mean_window_D = float(np.mean(D_list)) if D_list else np.nan

            image_rows.append({
                "parent": parent,
                "image_name": image_path.stem,
                "height": int(H), "width": int(W),
                "total_windows_generated": int(len(windows)),
                "valid_window_count": int(len(valid_rows_this_parent)),
                "window_size": int(args.window_size),
                "stride": int(args.stride),
                "min_unit_area": int(args.min_unit_area),
                "sum_window_unit_area_A3_overlap_weighted": int(total_window_area),
                "sum_window_white_pixels_A4_overlap_weighted": int(total_window_white),
                "image_white_ratio_Q_local_area_weighted": float(image_Q_local_weighted) if not np.isnan(image_Q_local_weighted) else np.nan,
                "image_blur_damage_D_local_area_weighted": float(image_D_local_weighted) if not np.isnan(image_D_local_weighted) else np.nan,
                "mean_window_blur_damage_D": float(mean_window_D) if not np.isnan(mean_window_D) else np.nan,
                "unique_A3_area_image_level": int(unique_area),
                "unique_A4_white_in_A3_image_level": int(unique_white),
                "unique_image_white_ratio_Q": float(unique_Q) if not np.isnan(unique_Q) else np.nan,
                "unique_image_blur_damage_D": float(unique_D) if not np.isnan(unique_D) else np.nan,
                "green_count": class_count["Green"],
                "yellow_count": class_count["Yellow"],
                "orange_count": class_count["Orange"],
                "red_count": class_count["Red"],
            })

            # Save heatmap for this parent.
            # Quantitative BD remains window-based and is used in CSV/evaluation/small-window display.
            # The heatmap is computed directly on the full image from M_Pred and R_Pred:
            #   raw pixel damage = 1 where M_Pred=1 and R_Pred=0, otherwise 0 inside M_Pred.
            # Then normalized Gaussian filtering converts it to a smooth full-image heatmap.
            blur_map, valid_heat = build_direct_pixel_damage_heatmap(
                roi_mask=pred,
                white_mask=binary,
                sigma=float(args.heatmap_smooth_sigma),
            )
            if (not args.no_heatmap) and SAVE_HEATMAP_FOR_ALL:
                heat_overlay = heatmap_overlay_on_roi(img, blur_map, valid_heat, alpha=0.55, title="Blur Damage D")
                heat_overlay = draw_multiline_box(heat_overlay, [
                    "Predicted blur heatmap",
                    "Heatmap = smoothed pixel damage (A3 - A4)",
                    "BD calculation = overlapping windows",
                    f"Mean BD(window) = {image_D_local_weighted:.3f}" if not np.isnan(image_D_local_weighted) else "Mean BD(window) = N/A",
                    f"Unique-image BD = {unique_D:.3f}" if not np.isnan(unique_D) else "Unique-image BD = N/A",
                    f"Valid windows = {len(valid_rows_this_parent)}",
                    f"Window/stride = {args.window_size}/{args.stride}",
                    f"Heatmap sigma = {float(args.heatmap_smooth_sigma):.1f}px" if float(args.heatmap_smooth_sigma) > 0 else "Heatmap smoothing: off",
                ], x=10, y=10, font_scale=0.68, thickness=2)
                cv2.imwrite(str(heatmap_overlay_dir / f"{parent}_pred_blur_heatmap_overlay.png"), heat_overlay)

        except Exception as e:
            skipped_rows.append({"parent": parent, "image_file": image_path.name, "reason": repr(e)})
            print(f"[WARN] skipped {parent}: {e}")

    # Save all tables.
    pd.DataFrame(window_rows).to_csv(output_dir / "window_level_blur_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(image_rows).to_csv(output_dir / "image_level_blur_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(output_dir / "missing_pairs_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_rows).to_csv(output_dir / "skipped_report.csv", index=False, encoding="utf-8-sig")

    # Save reproducibility config.
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "proposed_blur_from_A3_A4_local_overlapping_windows",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "IMAGE_DIR": str(image_dir),
            "PRED_MASK_DIR_A3": str(pred_mask_dir),
            "BINARY_MASK_DIR_A4": str(binary_mask_dir),
            "OUTPUT_DIR": str(output_dir),
            "WINDOW_SIZE": int(args.window_size),
            "STRIDE": int(args.stride),
            "MIN_UNIT_AREA": int(args.min_unit_area),
            "MIN_UNIT_AREA_RATIO": float(args.min_unit_area_ratio),
            "FORMULA": "U_k = A3 ∩ W_k ; Q_pred = |A4 ∩ U_k| / |U_k| ; D_pred = 1 - Q_pred",
            "RESIZE_MASKS_TO_IMAGE": bool(args.resize_masks_to_image),
            "RANDOM_SEED": int(args.random_seed),
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Window CSV:", output_dir / "window_level_blur_scores.csv")
    print("Image CSV :", output_dir / "image_level_blur_summary.csv")
    print("Heatmaps  :", heatmap_overlay_dir)


if __name__ == "__main__":
    main()


