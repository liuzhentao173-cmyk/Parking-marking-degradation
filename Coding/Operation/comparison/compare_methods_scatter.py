# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

"""
P4. Window-level and object-level scatter validation from P3 method-comparison outputs
====================================================================================

Purpose
-------
This script is a post-processing validation script for compare_binarization_methods.py.
It does NOT rerun binarization. Instead, it scans the already-generated P3 outputs:

    METHOD_COMPARE_ROOT/runs/<comp_variant>/<method>/03_final_mask/*.png

and compares every method-specific R_Pred final/binary mask against M_GT/R_GT/M_Pred at two granularities:

1) Window-level validation
   For each overlapping window W_k:

   GT blur on GT ROI:
       D_GT_A1 = 1 - |R_GT ∩ M_GT ∩ W_k| / |M_GT ∩ W_k|

   Method blur using GT denominator, recommended for method comparison:
       D_Method_A1 = 1 - |R_Pred ∩ M_GT ∩ W_k| / |M_GT ∩ W_k|

   Method blur using predicted denominator, operational pipeline output:
       D_Method_A3 = 1 - |R_Pred ∩ M_Pred ∩ W_k| / |M_Pred ∩ W_k|

2) Object-level validation
   Two object definitions are supported:

   M_GT-object level:
       connected components from M_GT are used as units. This is strict and GT-anchored.

   M_Pred-object level:
       connected components from M_Pred are used as units. This isolates binarization inside the predicted ROI.

Main outputs
------------
- window_level_scatter_data.csv
- object_level_scatter_data.csv
- unit_level_summary.csv
- method_granularity_validation_summary.xlsx
- plots/window_A1den_YX_grid.png
- plots/window_A3den_YX_grid.png
- plots/object_A1_objects_YX_grid.png
- plots/object_A3_objects_YX_grid.png
- plots/summary_* figures

Recommended interpretation
--------------------------
For paper main text, prefer:
    window_A1den_YX_grid.png + unit_level_summary.csv rows where unit_level=window and eval_mode=A1den
because this directly evaluates the downstream D value on the same GT ROI denominator.

For discussion/error decomposition, use:
    A3den / object_A3 rows
because these show how the operational pipeline behaves when using the predicted ROI as denominator.
"""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================
# 0. User config: edit paths here
# ============================================================
# This should be the exact P3 output folder, for example:
#   <P3_OUTPUT_ROOT>/MC_<model>_binarization_method_compare_<timestamp>
# By default this points to the parent folder; main() will pick the latest MC_* run.
METHOD_COMPARE_ROOT = Path(r"")  # TODO: set path

# If METHOD_COMPARE_ROOT does not exist and --auto-latest is used, the script will select the latest MC_* folder here.
METHOD_COMPARE_PARENT = Path(r"")  # TODO: set path

# These paths are automatically read from METHOD_COMPARE_ROOT/run_config.json when possible.
# Keep them here as fallbacks or override them by CLI.
IMAGE_DIR = Path(r"")  # TODO: set path
GT_ORIG_DIR = Path(r"")         # M_GT
GT_BIN_DIR = Path(r"")        # R_GT
PRED_ORIG_DIR = Path(r"")  # M_Pred

OUTPUT_SUBDIR = "scatter_window_object_validation"

# ============================================================
# 1. Parameters
# ============================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_THRESHOLD = 127
RESIZE_MASKS_TO_IMAGE = False

# Window-level parameters, following your Script 03/04 default logic.
WINDOW_SIZE = 512
STRIDE = 256
MIN_GT_WINDOW_AREA = 500
MIN_PRED_WINDOW_AREA = 500
MIN_PAIR_WINDOW_AREA = 500

# Object-level parameters.
MIN_A1_OBJECT_AREA = 500
MIN_A3_OBJECT_AREA = 500
APPLY_OBJECT_OPENING = False
OBJECT_OPEN_KSIZE = 3

# Pixel-level metric region settings.
ERROR_BAND = 0.10
FIG_DPI = 300
SAVE_EXCEL = True

# Plot style.
FONT_TITLE = 18
FONT_AXIS = 15
FONT_TICK = 12
FONT_LEGEND = 11
FONT_ANNOT = 10

# Stable method order and display labels. Unknown methods are appended automatically.
METHOD_ORDER = ["proposed_baseline", "otsu", "fixed_strict", "fixed_loose", "pure_wkdegmm"]
COMP_ORDER = ["raw", "illum_comp"]
METHOD_LABEL = {
    "proposed_baseline": "Proposed guarded Bin-OTP",
    "otsu": "Otsu",
    "fixed_strict": "Fixed strict",
    "fixed_loose": "Fixed loose",
    "pure_wkdegmm": "Pure WKDE-GMM",
}
COMP_LABEL = {
    "raw": "No compensation",
    "illum_comp": "With compensation",
}

# Publication-friendly, colorblind-aware enough for method categories.
METHOD_COLOR = {
    "proposed_baseline": "#00796B",
    "otsu": "#E69F00",
    "fixed_strict": "#7A7A7A",
    "fixed_loose": "#BDBDBD",
    "pure_wkdegmm": "#6A51A3",
}

# ============================================================
# 2. CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create window-level and object-level scatter validation plots from P3 method-comparison final masks."
    )
    parser.add_argument("--method-root", type=str, default=str(METHOD_COMPARE_ROOT), help="P3 output folder containing runs/<comp>/<method>/03_final_mask.")
    parser.add_argument("--method-parent", type=str, default=str(METHOD_COMPARE_PARENT), help="Parent folder used with --auto-latest.")
    parser.add_argument("--auto-latest", action="store_true", help="Automatically choose the latest MC_* folder under --method-parent if --method-root is invalid or points to the parent folder.")
    parser.add_argument("--image-dir", type=str, default="", help="Full-size RGB image folder. Empty means read from run_config.json or fallback default.")
    parser.add_argument("--gt-orig-dir", type=str, default="", help="A1 GT ROI mask folder. Empty means read from run_config.json or fallback default.")
    parser.add_argument("--gt-bin-dir", type=str, default="", help="A2 GT binary white-paint mask folder. Empty means read from run_config.json or fallback default.")
    parser.add_argument("--pred-orig-dir", type=str, default="", help="A3 predicted ROI mask folder. Empty means read from run_config.json or fallback default.")
    parser.add_argument("--output-dir", type=str, default="", help="Output folder. Empty -> method_root/scatter_window_object_validation.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--min-gt-window-area", type=int, default=MIN_GT_WINDOW_AREA)
    parser.add_argument("--min-pred-window-area", type=int, default=MIN_PRED_WINDOW_AREA)
    parser.add_argument("--min-pair-window-area", type=int, default=MIN_PAIR_WINDOW_AREA)
    parser.add_argument("--min-a1-object-area", type=int, default=MIN_A1_OBJECT_AREA)
    parser.add_argument("--min-a3-object-area", type=int, default=MIN_A3_OBJECT_AREA)
    parser.add_argument("--resize-masks-to-image", action="store_true", default=RESIZE_MASKS_TO_IMAGE)
    parser.add_argument("--apply-object-opening", action="store_true", default=APPLY_OBJECT_OPENING)
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--skip-window", action="store_true", help="Skip window-level calculation.")
    parser.add_argument("--skip-object", action="store_true", help="Skip object-level calculation.")
    return parser.parse_args()

# ============================================================
# 3. Basic helpers
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def list_files(folder: Path) -> List[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


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


def build_file_map(folder: Path, strip_suffix: bool = True) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if folder is None or not folder.exists():
        return out
    for p in list_files(folder):
        key = normalize_stem(p.stem) if strip_suffix else p.stem
        # If both parent.png and parent__final_mask.png exist, prefer parent.png.
        if key in out:
            old = out[key]
            if old.stem.endswith("__final_mask") and not p.stem.endswith("__final_mask"):
                out[key] = p
        else:
            out[key] = p
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


def read_mask01(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > MASK_THRESHOLD).astype(np.uint8)


def read_rgb_if_exists(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def align_mask(mask: np.ndarray, H: int, W: int, resize: bool, name: str) -> np.ndarray:
    if mask.shape == (H, W):
        return mask.astype(np.uint8)
    if resize:
        return cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    raise ValueError(f"{name} shape mismatch: mask={mask.shape}, target={(H, W)}. Use --resize-masks-to-image if intended.")


def smooth_binary(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.uint8)


def choose_latest_method_root(parent: Path) -> Path:
    candidates = sorted([p for p in parent.glob("MC_*") if p.is_dir() and (p / "runs").exists()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No MC_* method-comparison folders found under: {parent}")
    return candidates[0]


def natural_sort_key(v):
    v = str(v)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", v)]

# ============================================================
# 4. Metric helpers
# ============================================================
def safe_ratio(num: int, den: int) -> float:
    return np.nan if den <= 0 else float(num) / float(den)


def D_from_Q(Q: float) -> float:
    if Q is None or np.isnan(Q):
        return np.nan
    return float(np.clip(1.0 - Q, 0.0, 1.0))


def blur_from_masks(denom_mask: np.ndarray, white_mask: np.ndarray) -> Dict[str, float]:
    denom = denom_mask > 0
    area = int(denom.sum())
    white = int(((white_mask > 0) & denom).sum())
    q = safe_ratio(white, area)
    return {"area": area, "white": white, "Q": q, "D": D_from_Q(q)}


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
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "pixel_acc": float(acc),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def paired_metrics(x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if weight is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(weight, dtype=float)[valid]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if len(x) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias_y_minus_x": np.nan,
            "Pearson_r": np.nan,
            "R2": np.nan,
            "Within_±0.10_ratio": np.nan,
            "AreaWeighted_MAE": np.nan,
            "AreaWeighted_RMSE": np.nan,
            "AreaWeighted_Bias": np.nan,
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
        aw_mae = float(np.sum(w * abs_err) / np.sum(w))
        aw_rmse = float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))
        aw_bias = float(np.sum(w * err) / np.sum(w))
    else:
        aw_mae = aw_rmse = aw_bias = np.nan
    return {
        "n": int(len(x)),
        "MAE": mae,
        "RMSE": rmse,
        "Bias_y_minus_x": bias,
        "Pearson_r": pearson,
        "R2": r2,
        "Within_±0.10_ratio": within,
        "AreaWeighted_MAE": aw_mae,
        "AreaWeighted_RMSE": aw_rmse,
        "AreaWeighted_Bias": aw_bias,
    }

# ============================================================
# 5. Window and object helpers
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


def connected_component_rows(mask: np.ndarray, min_area: int, prefix: str = "obj") -> List[Dict]:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows: List[Dict] = []
    obj_id = 0
    for lab in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[lab]]
        if area < int(min_area):
            continue
        obj_id += 1
        comp_crop = (labels[y:y + h, x:x + w] == lab).astype(np.uint8)
        rows.append({
            "unit_id": obj_id,
            "label_id": lab,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "mask_crop": comp_crop,
            "unit_name": f"{prefix}_{obj_id:04d}",
        })
    return rows

# ============================================================
# 6. Run discovery and record loading
# ============================================================
def resolve_paths_from_config(method_root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    cfg = read_json(method_root / "run_config.json")
    paths = {
        "image_dir": Path(args.image_dir) if str(args.image_dir).strip() else Path(cfg.get("IMAGE_DIR", str(IMAGE_DIR))),
        "gt_orig_dir": Path(args.gt_orig_dir) if str(args.gt_orig_dir).strip() else Path(cfg.get("GT_MASK_DIR_A1", str(GT_ORIG_DIR))),
        "gt_bin_dir": Path(args.gt_bin_dir) if str(args.gt_bin_dir).strip() else Path(cfg.get("GT_BIN_DIR_A2", str(GT_BIN_DIR))),
        "pred_orig_dir": Path(args.pred_orig_dir) if str(args.pred_orig_dir).strip() else Path(cfg.get("PRED_MASK_DIR_A3", str(PRED_ORIG_DIR))),
    }
    return paths


def discover_method_runs(method_root: Path) -> List[Dict]:
    runs_root = method_root / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Cannot find runs folder: {runs_root}")
    rows: List[Dict] = []
    for final_dir in sorted(runs_root.glob("*/*/03_final_mask")):
        if not final_dir.is_dir():
            continue
        method_dir = final_dir.parent
        comp_dir = method_dir.parent
        rows.append({
            "comp_variant": comp_dir.name,
            "method": method_dir.name,
            "final_mask_dir": final_dir,
            "run_dir": method_dir,
        })
    if not rows:
        raise RuntimeError(f"No method final-mask folders found under: {runs_root}")

    def sort_key(r):
        comp_idx = COMP_ORDER.index(r["comp_variant"]) if r["comp_variant"] in COMP_ORDER else 99
        meth_idx = METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99
        return comp_idx, meth_idx, r["comp_variant"], r["method"]

    return sorted(rows, key=sort_key)


def collect_common_parents(paths: Dict[str, Path], method_runs: List[Dict]) -> Tuple[List[str], Dict[str, Dict[str, Path]], pd.DataFrame]:
    maps = {
        "A1": build_file_map(paths["gt_orig_dir"], strip_suffix=True),
        "A2": build_file_map(paths["gt_bin_dir"], strip_suffix=True),
        "A3": build_file_map(paths["pred_orig_dir"], strip_suffix=True),
        "Image": build_file_map(paths["image_dir"], strip_suffix=True),
    }
    for mr in method_runs:
        key = f"A4::{mr['comp_variant']}::{mr['method']}"
        maps[key] = build_file_map(mr["final_mask_dir"], strip_suffix=True)

    base = set(maps["A1"].keys()) & set(maps["A2"].keys()) & set(maps["A3"].keys())
    # Require every method to have the parent; this keeps paired comparison fair.
    for mr in method_runs:
        key = f"A4::{mr['comp_variant']}::{mr['method']}"
        base &= set(maps[key].keys())
    parents = sorted(base, key=natural_sort_key)

    report_rows = []
    all_candidates = set(maps["A1"].keys()) | set(maps["A2"].keys()) | set(maps["A3"].keys())
    for mr in method_runs:
        all_candidates |= set(maps[f"A4::{mr['comp_variant']}::{mr['method']}"])
    for p in sorted(all_candidates, key=natural_sort_key):
        row = {
            "parent": p,
            "included_in_all_methods": p in set(parents),
            "A1_found": p in maps["A1"],
            "A2_found": p in maps["A2"],
            "A3_found": p in maps["A3"],
            "Image_found": p in maps["Image"],
        }
        for mr in method_runs:
            key = f"A4::{mr['comp_variant']}::{mr['method']}"
            row[f"A4_found__{mr['comp_variant']}__{mr['method']}"] = p in maps[key]
        report_rows.append(row)
    return parents, maps, pd.DataFrame(report_rows)

# ============================================================
# 7. Core calculation
# ============================================================
def process_window_rows_for_method(parent: str, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray,
                                   comp: str, method: str, args: argparse.Namespace) -> List[Dict]:
    H, W = A1.shape
    rows: List[Dict] = []
    for window_id, x, y, w, h in generate_windows(H, W, int(args.window_size), int(args.stride)):
        a1 = A1[y:y + h, x:x + w]
        a2 = A2[y:y + h, x:x + w]
        a3 = A3[y:y + h, x:x + w]
        a4 = A4[y:y + h, x:x + w]

        gt_a1 = blur_from_masks(a1, a2)
        method_a1 = blur_from_masks(a1, a4)
        method_a3 = blur_from_masks(a3, a4)
        gt_on_a3 = blur_from_masks(a3, a2)

        gt_valid = bool(gt_a1["area"] >= int(args.min_gt_window_area))
        pred_valid = bool(method_a3["area"] >= int(args.min_pred_window_area))
        pair_a1_valid = bool(gt_valid and method_a1["area"] >= int(args.min_gt_window_area))
        pair_a3_valid = bool(gt_valid and pred_valid and min(gt_a1["area"], method_a3["area"]) >= int(args.min_pair_window_area))
        pure_a3_valid = bool(pred_valid and gt_on_a3["area"] >= int(args.min_pred_window_area))

        if not (gt_valid or pred_valid):
            continue

        region_A1 = a1.astype(np.uint8)
        region_A3_inter_A1 = ((a3 > 0) & (a1 > 0)).astype(np.uint8)
        c_A1 = pixel_counts(a4, a2, region_A1)
        m_A1 = metrics_from_counts(c_A1)
        c_A3iA1 = pixel_counts(a4, a2, region_A3_inter_A1)
        m_A3iA1 = metrics_from_counts(c_A3iA1)

        row = {
            "unit_level": "window",
            "parent": parent,
            "comp_variant": comp,
            "method": method,
            "unit_id": int(window_id),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "window_area": int(w * h),
            "window_size": int(args.window_size),
            "stride": int(args.stride),

            "GT_A1_area": int(gt_a1["area"]),
            "GT_A1_white": int(gt_a1["white"]),
            "GT_A1_Q": float(gt_a1["Q"]) if np.isfinite(gt_a1["Q"]) else np.nan,
            "GT_A1_D": float(gt_a1["D"]) if np.isfinite(gt_a1["D"]) else np.nan,

            "Method_A1_area": int(method_a1["area"]),
            "Method_A1_white": int(method_a1["white"]),
            "Method_A1_Q": float(method_a1["Q"]) if np.isfinite(method_a1["Q"]) else np.nan,
            "Method_A1_D": float(method_a1["D"]) if np.isfinite(method_a1["D"]) else np.nan,

            "Method_A3_area": int(method_a3["area"]),
            "Method_A3_white": int(method_a3["white"]),
            "Method_A3_Q": float(method_a3["Q"]) if np.isfinite(method_a3["Q"]) else np.nan,
            "Method_A3_D": float(method_a3["D"]) if np.isfinite(method_a3["D"]) else np.nan,

            "GT_on_A3_area": int(gt_on_a3["area"]),
            "GT_on_A3_white": int(gt_on_a3["white"]),
            "GT_on_A3_Q": float(gt_on_a3["Q"]) if np.isfinite(gt_on_a3["Q"]) else np.nan,
            "GT_on_A3_D": float(gt_on_a3["D"]) if np.isfinite(gt_on_a3["D"]) else np.nan,

            "pair_A1den_valid": pair_a1_valid,
            "pair_A3den_valid": pair_a3_valid,
            "pair_A3pure_valid": pure_a3_valid,
            "pair_weight_A1den": int(gt_a1["area"]) if pair_a1_valid else 0,
            "pair_weight_A3den": int(min(gt_a1["area"], method_a3["area"])) if pair_a3_valid else 0,
            "pair_weight_A3pure": int(method_a3["area"]) if pure_a3_valid else 0,
            "error_A1den": float(method_a1["D"] - gt_a1["D"]) if pair_a1_valid else np.nan,
            "abs_error_A1den": abs(float(method_a1["D"] - gt_a1["D"])) if pair_a1_valid else np.nan,
            "error_A3den": float(method_a3["D"] - gt_a1["D"]) if pair_a3_valid else np.nan,
            "abs_error_A3den": abs(float(method_a3["D"] - gt_a1["D"])) if pair_a3_valid else np.nan,
            "error_A3pure": float(method_a3["D"] - gt_on_a3["D"]) if pure_a3_valid else np.nan,
            "abs_error_A3pure": abs(float(method_a3["D"] - gt_on_a3["D"])) if pure_a3_valid else np.nan,
        }
        for k, v in c_A1.items():
            row[f"A4_vs_A2_within_A1_{k}"] = v
        for k, v in m_A1.items():
            row[f"A4_vs_A2_within_A1_{k}"] = v
        for k, v in c_A3iA1.items():
            row[f"A4_vs_A2_within_A3_inter_A1_{k}"] = v
        for k, v in m_A3iA1.items():
            row[f"A4_vs_A2_within_A3_inter_A1_{k}"] = v
        rows.append(row)
    return rows


def process_object_rows_for_method(parent: str, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray, A4: np.ndarray,
                                   comp: str, method: str, args: argparse.Namespace) -> List[Dict]:
    rows: List[Dict] = []
    A1_src = smooth_binary(A1, OBJECT_OPEN_KSIZE) if bool(args.apply_object_opening) else A1
    A3_src = smooth_binary(A3, OBJECT_OPEN_KSIZE) if bool(args.apply_object_opening) else A3

    object_sets = [
        ("object_A1", "A1_connected_component", A1_src, int(args.min_a1_object_area), "A1"),
        ("object_A3", "A3_connected_component", A3_src, int(args.min_a3_object_area), "A3"),
    ]
    for unit_level, unit_def, src_mask, min_area, denom_name in object_sets:
        comps = connected_component_rows(src_mask, min_area=min_area, prefix=denom_name)
        for obj in comps:
            x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"])
            unit = obj["mask_crop"].astype(np.uint8)
            a2 = A2[y:y + h, x:x + w]
            a4 = A4[y:y + h, x:x + w]
            a1 = A1[y:y + h, x:x + w]
            a3 = A3[y:y + h, x:x + w]

            gt = blur_from_masks(unit, a2)
            method_res = blur_from_masks(unit, a4)
            c = pixel_counts(a4, a2, unit)
            m = metrics_from_counts(c)

            # Auxiliary, to help interpret whether the object is well aligned with the other ROI.
            a1_inter_unit = ((a1 > 0) & (unit > 0)).astype(np.uint8)
            a3_inter_unit = ((a3 > 0) & (unit > 0)).astype(np.uint8)
            overlap_a1_ratio = safe_ratio(int(a1_inter_unit.sum()), int(unit.sum()))
            overlap_a3_ratio = safe_ratio(int(a3_inter_unit.sum()), int(unit.sum()))

            valid = bool(gt["area"] >= min_area and method_res["area"] >= min_area)
            row = {
                "unit_level": unit_level,
                "unit_definition": unit_def,
                "parent": parent,
                "comp_variant": comp,
                "method": method,
                "unit_id": int(obj["unit_id"]),
                "unit_name": obj["unit_name"],
                "x": x, "y": y, "w": w, "h": h,
                "unit_area": int(gt["area"]),
                "GT_D": float(gt["D"]) if np.isfinite(gt["D"]) else np.nan,
                "GT_Q": float(gt["Q"]) if np.isfinite(gt["Q"]) else np.nan,
                "GT_white": int(gt["white"]),
                "Method_D": float(method_res["D"]) if np.isfinite(method_res["D"]) else np.nan,
                "Method_Q": float(method_res["Q"]) if np.isfinite(method_res["Q"]) else np.nan,
                "Method_white": int(method_res["white"]),
                "pair_valid": valid,
                "pair_weight_area": int(gt["area"]) if valid else 0,
                "error_Method_minus_GT": float(method_res["D"] - gt["D"]) if valid else np.nan,
                "absolute_error": abs(float(method_res["D"] - gt["D"])) if valid else np.nan,
                "overlap_ratio_with_A1": float(overlap_a1_ratio) if np.isfinite(overlap_a1_ratio) else np.nan,
                "overlap_ratio_with_A3": float(overlap_a3_ratio) if np.isfinite(overlap_a3_ratio) else np.nan,
            }
            for k, v in c.items():
                row[f"A4_vs_A2_within_unit_{k}"] = v
            for k, v in m.items():
                row[f"A4_vs_A2_within_unit_{k}"] = v
            rows.append(row)
    return rows

# ============================================================
# 8. Summary construction
# ============================================================
def summarize_eval(df: pd.DataFrame, unit_level: str, eval_mode: str, x_col: str, y_col: str, weight_col: str,
                   valid_col: str, iou_col: Optional[str], count_prefix: Optional[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for (comp, method), sub0 in df.groupby(["comp_variant", "method"], dropna=False):
        sub = sub0.copy()
        if valid_col in sub.columns:
            sub = sub[sub[valid_col] == True].copy()
        sub = sub.dropna(subset=[x_col, y_col])
        if sub.empty:
            met = paired_metrics(np.array([]), np.array([]))
        else:
            met = paired_metrics(sub[x_col].to_numpy(float), sub[y_col].to_numpy(float), sub[weight_col].to_numpy(float) if weight_col in sub.columns else None)
        row = {
            "unit_level": unit_level,
            "eval_mode": eval_mode,
            "comp_variant": comp,
            "method": method,
            "plot_label": f"{METHOD_LABEL.get(method, method)} | {COMP_LABEL.get(comp, comp)}",
            "x_col": x_col,
            "y_col": y_col,
            "weight_col": weight_col,
            **met,
        }
        if not sub.empty and iou_col and iou_col in sub.columns:
            row["mean_iou"] = float(pd.to_numeric(sub[iou_col], errors="coerce").mean())
            dice_col = iou_col.replace("_iou", "_dice")
            prec_col = iou_col.replace("_iou", "_precision")
            rec_col = iou_col.replace("_iou", "_recall")
            fpr_col = iou_col.replace("_iou", "_fpr")
            fnr_col = iou_col.replace("_iou", "_fnr")
            for src, dst in [(dice_col, "mean_dice"), (prec_col, "mean_precision"), (rec_col, "mean_recall"), (fpr_col, "mean_fpr"), (fnr_col, "mean_fnr")]:
                if src in sub.columns:
                    row[dst] = float(pd.to_numeric(sub[src], errors="coerce").mean())
        else:
            row["mean_iou"] = np.nan

        if count_prefix:
            counts = {}
            for k in ["TP", "FP", "FN", "TN", "region_pixels", "pred_positive_pixels", "gt_positive_pixels"]:
                col = f"{count_prefix}_{k}"
                counts[k] = int(pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            mets = metrics_from_counts(counts)
            for k, v in counts.items():
                row[f"micro_{k}"] = v
            for k, v in mets.items():
                row[f"micro_{k}"] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["method_order"] = out["method"].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    out["comp_order"] = out["comp_variant"].map(lambda c: COMP_ORDER.index(c) if c in COMP_ORDER else 99)
    return out.sort_values(["unit_level", "eval_mode", "comp_order", "method_order", "method"]).drop(columns=["method_order", "comp_order"])


def build_all_summaries(window_df: pd.DataFrame, object_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not window_df.empty:
        frames.append(summarize_eval(
            window_df,
            unit_level="window",
            eval_mode="A1den",
            x_col="GT_A1_D",
            y_col="Method_A1_D",
            weight_col="pair_weight_A1den",
            valid_col="pair_A1den_valid",
            iou_col="A4_vs_A2_within_A1_iou",
            count_prefix="A4_vs_A2_within_A1",
        ))
        frames.append(summarize_eval(
            window_df,
            unit_level="window",
            eval_mode="A3den_operational",
            x_col="GT_A1_D",
            y_col="Method_A3_D",
            weight_col="pair_weight_A3den",
            valid_col="pair_A3den_valid",
            iou_col="A4_vs_A2_within_A3_inter_A1_iou",
            count_prefix="A4_vs_A2_within_A3_inter_A1",
        ))
        frames.append(summarize_eval(
            window_df,
            unit_level="window",
            eval_mode="A3den_pure_binarization",
            x_col="GT_on_A3_D",
            y_col="Method_A3_D",
            weight_col="pair_weight_A3pure",
            valid_col="pair_A3pure_valid",
            iou_col="A4_vs_A2_within_A3_inter_A1_iou",
            count_prefix="A4_vs_A2_within_A3_inter_A1",
        ))
    if not object_df.empty:
        for ul in ["object_A1", "object_A3"]:
            sub = object_df[object_df["unit_level"] == ul].copy()
            if sub.empty:
                continue
            frames.append(summarize_eval(
                sub,
                unit_level=ul,
                eval_mode="unit_denominator",
                x_col="GT_D",
                y_col="Method_D",
                weight_col="pair_weight_area",
                valid_col="pair_valid",
                iou_col="A4_vs_A2_within_unit_iou",
                count_prefix="A4_vs_A2_within_unit",
            ))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ============================================================
# 9. Plotting
# ============================================================
def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_TITLE,
        "font.family": "Times New Roman",
    })


def method_comp_order(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    methods = list(df["method"].dropna().unique())
    comps = list(df["comp_variant"].dropna().unique())
    methods = sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    comps = sorted(comps, key=lambda c: COMP_ORDER.index(c) if c in COMP_ORDER else 99)
    return methods, comps


def bubble_sizes(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=float)
    area = np.where(np.isfinite(area) & (area > 0), area, 1)
    s = np.sqrt(area)
    if s.size == 0:
        return s
    if np.nanmax(s) > np.nanmin(s):
        return 18 + (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s)) * 95
    return np.full_like(s, 55.0)


def scatter_grid(df: pd.DataFrame, x_col: str, y_col: str, weight_col: str, valid_col: str,
                 title: str, x_label: str, y_label: str, out_path: Path) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    plot_df = df.copy()
    if valid_col in plot_df.columns:
        plot_df = plot_df[plot_df[valid_col] == True].copy()
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return
    methods, comps = method_comp_order(plot_df)
    nrows, ncols = len(comps), len(methods)
    fig_w = max(4.0 * ncols, 8.0)
    fig_h = max(4.1 * nrows, 5.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharex=True, sharey=True)
    line = np.linspace(0, 1, 200)
    for r, comp in enumerate(comps):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            sub = plot_df[(plot_df["comp_variant"] == comp) & (plot_df["method"] == method)].copy()
            ax.plot(line, line, color="black", linewidth=1.4, label="Y=X")
            ax.plot(line, np.clip(line + ERROR_BAND, 0, 1), color="#666666", linestyle="--", linewidth=1.0)
            ax.plot(line, np.clip(line - ERROR_BAND, 0, 1), color="#666666", linestyle="--", linewidth=1.0)
            if not sub.empty:
                area = sub[weight_col].to_numpy(float) if weight_col in sub.columns else np.ones(len(sub))
                ax.scatter(
                    sub[x_col].to_numpy(float),
                    sub[y_col].to_numpy(float),
                    s=bubble_sizes(area),
                    alpha=0.55,
                    color=METHOD_COLOR.get(method, "#4C72B0"),
                    edgecolors="black",
                    linewidths=0.35,
                )
                met = paired_metrics(sub[x_col].to_numpy(float), sub[y_col].to_numpy(float), area)
                txt = (
                    f"n={met['n']}\n"
                    f"MAE={met['MAE']:.3f}\n"
                    f"AW-MAE={met['AreaWeighted_MAE']:.3f}\n"
                    f"r={met['Pearson_r']:.2f}"
                )
                ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=FONT_ANNOT,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.6))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.42)
            if r == 0:
                ax.set_title(METHOD_LABEL.get(method, method), fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{COMP_LABEL.get(comp, comp)}\n{y_label}")
            if r == nrows - 1:
                ax.set_xlabel(x_label)
    fig.suptitle(title, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_summary_bar(summary_df: pd.DataFrame, unit_level: str, eval_mode: str, metric: str, out_path: Path,
                     ylabel: str, ascending: bool = True) -> None:
    sub = summary_df[(summary_df["unit_level"] == unit_level) & (summary_df["eval_mode"] == eval_mode)].copy()
    if sub.empty or metric not in sub.columns:
        return
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return
    sub["method_order"] = sub["method"].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    sub["comp_order"] = sub["comp_variant"].map(lambda c: COMP_ORDER.index(c) if c in COMP_ORDER else 99)
    sub = sub.sort_values(["comp_order", "method_order"])
    labels = [f"{METHOD_LABEL.get(m, m)}\n{COMP_LABEL.get(c, c)}" for m, c in zip(sub["method"], sub["comp_variant"])]
    vals = sub[metric].to_numpy(float)
    colors = [METHOD_COLOR.get(m, "#4C72B0") for m in sub["method"]]
    fig_h = max(5.8, 0.48 * len(sub) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    bars = ax.barh(np.arange(len(sub)), vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel)
    ax.set_title(f"{ylabel}: {unit_level} / {eval_mode}", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    if not ascending:
        ax.invert_yaxis()
    pad = max((np.nanmax(vals) - np.nanmin(vals)) * 0.015, 0.005) if vals.size else 0.005
    for bar, val in zip(bars, vals):
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=FONT_ANNOT)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_pareto(summary_df: pd.DataFrame, unit_level: str, eval_mode: str, out_path: Path) -> None:
    sub = summary_df[(summary_df["unit_level"] == unit_level) & (summary_df["eval_mode"] == eval_mode)].copy()
    if sub.empty or "mean_iou" not in sub.columns or "AreaWeighted_MAE" not in sub.columns:
        return
    sub["mean_iou"] = pd.to_numeric(sub["mean_iou"], errors="coerce")
    sub["AreaWeighted_MAE"] = pd.to_numeric(sub["AreaWeighted_MAE"], errors="coerce")
    sub = sub.dropna(subset=["mean_iou", "AreaWeighted_MAE"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.3, 7.0))
    for _, row in sub.iterrows():
        method = row["method"]
        comp = row["comp_variant"]
        marker = "o" if comp == "illum_comp" else "s"
        face = METHOD_COLOR.get(method, "#4C72B0")
        ax.scatter(row["AreaWeighted_MAE"], row["mean_iou"], s=120, marker=marker, color=face,
                   edgecolors="black", linewidths=0.8, alpha=0.88)
        ax.text(row["AreaWeighted_MAE"] + 0.002, row["mean_iou"] + 0.002,
                f"{METHOD_LABEL.get(method, method)}\n{COMP_LABEL.get(comp, comp)}", fontsize=FONT_ANNOT)
    ax.annotate("Better", xy=(0.04, 0.96), xytext=(0.20, 0.82), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", linewidth=1.5), fontsize=FONT_ANNOT + 1)
    ax.set_xlabel("Area-weighted MAE of D ↓")
    ax.set_ylabel("Mean IoU ↑")
    ax.set_title(f"Pareto view: {unit_level} / {eval_mode}", fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_error_violin_or_box(df: pd.DataFrame, unit_level: str, error_col: str, valid_col: str, title: str, out_path: Path) -> None:
    if df.empty or error_col not in df.columns:
        return
    sub = df.copy()
    if "unit_level" in sub.columns:
        sub = sub[sub["unit_level"] == unit_level].copy()
    if valid_col in sub.columns:
        sub = sub[sub[valid_col] == True].copy()
    sub[error_col] = pd.to_numeric(sub[error_col], errors="coerce")
    sub = sub.dropna(subset=[error_col])
    if sub.empty:
        return
    methods, comps = method_comp_order(sub)
    labels, data, colors = [], [], []
    for comp in comps:
        for method in methods:
            vals = sub[(sub["comp_variant"] == comp) & (sub["method"] == method)][error_col].dropna().to_numpy(float)
            if vals.size:
                labels.append(f"{METHOD_LABEL.get(method, method)}\n{COMP_LABEL.get(comp, comp)}")
                data.append(vals)
                colors.append(METHOD_COLOR.get(method, "#4C72B0"))
    if not data:
        return
    fig_w = max(10, 0.75 * len(data) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.78)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor("black")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(ERROR_BAND, color="#555555", linestyle="--", linewidth=1.4, label="0.10 error band")
    ax.set_ylabel("Absolute error of D")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(window_df: pd.DataFrame, object_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    plots = output_dir / "plots"
    ensure_dir(plots)

    if not window_df.empty:
        scatter_grid(
            window_df,
            x_col="GT_A1_D",
            y_col="Method_A1_D",
            weight_col="pair_weight_A1den",
            valid_col="pair_A1den_valid",
            title="Window-level validation using GT denominator: D(A4/A1) vs D(A2/A1)",
            x_label="GT local D: 1 - |A2∩A1∩W| / |A1∩W|",
            y_label="Method local D: 1 - |A4∩A1∩W| / |A1∩W|",
            out_path=plots / "window_A1den_YX_grid.png",
        )
        scatter_grid(
            window_df,
            x_col="GT_A1_D",
            y_col="Method_A3_D",
            weight_col="pair_weight_A3den",
            valid_col="pair_A3den_valid",
            title="Window-level operational validation: D(A4/A3) vs D(A2/A1)",
            x_label="GT local D: 1 - |A2∩A1∩W| / |A1∩W|",
            y_label="Operational local D: 1 - |A4∩A3∩W| / |A3∩W|",
            out_path=plots / "window_A3den_YX_grid.png",
        )
        scatter_grid(
            window_df,
            x_col="GT_on_A3_D",
            y_col="Method_A3_D",
            weight_col="pair_weight_A3pure",
            valid_col="pair_A3pure_valid",
            title="Window-level binarization-only validation inside A3: D(A4/A3) vs D(A2/A3)",
            x_label="GT-on-A3 local D: 1 - |A2∩A3∩W| / |A3∩W|",
            y_label="Method local D: 1 - |A4∩A3∩W| / |A3∩W|",
            out_path=plots / "window_A3pure_YX_grid.png",
        )
        save_summary_bar(summary_df, "window", "A1den", "AreaWeighted_MAE", plots / "summary_window_A1den_AreaWeighted_MAE.png", "Area-weighted MAE of D ↓", ascending=True)
        save_summary_bar(summary_df, "window", "A1den", "mean_iou", plots / "summary_window_A1den_mean_IoU.png", "Mean IoU ↑", ascending=False)
        save_pareto(summary_df, "window", "A1den", plots / "pareto_window_A1den_IoU_vs_MAE.png")
        save_error_violin_or_box(window_df, "window", "abs_error_A1den", "pair_A1den_valid", "Window-level absolute error distribution: A1 denominator", plots / "violin_window_A1den_abs_error.png")
        save_error_violin_or_box(window_df, "window", "abs_error_A3den", "pair_A3den_valid", "Window-level absolute error distribution: operational A3 denominator", plots / "violin_window_A3den_abs_error.png")

    if not object_df.empty:
        obj_a1 = object_df[object_df["unit_level"] == "object_A1"].copy()
        obj_a3 = object_df[object_df["unit_level"] == "object_A3"].copy()
        if not obj_a1.empty:
            scatter_grid(
                obj_a1,
                x_col="GT_D",
                y_col="Method_D",
                weight_col="pair_weight_area",
                valid_col="pair_valid",
                title="Object-level validation on A1 connected components: D(A4/object) vs D(A2/object)",
                x_label="GT object D",
                y_label="Method object D",
                out_path=plots / "object_A1_objects_YX_grid.png",
            )
            save_summary_bar(summary_df, "object_A1", "unit_denominator", "AreaWeighted_MAE", plots / "summary_object_A1_AreaWeighted_MAE.png", "Area-weighted MAE of D ↓", ascending=True)
            save_pareto(summary_df, "object_A1", "unit_denominator", plots / "pareto_object_A1_IoU_vs_MAE.png")
            save_error_violin_or_box(object_df, "object_A1", "absolute_error", "pair_valid", "Object-level absolute error distribution: A1 objects", plots / "violin_object_A1_abs_error.png")
        if not obj_a3.empty:
            scatter_grid(
                obj_a3,
                x_col="GT_D",
                y_col="Method_D",
                weight_col="pair_weight_area",
                valid_col="pair_valid",
                title="Object-level validation inside A3 connected components: D(A4/object) vs D(A2/object)",
                x_label="GT-on-A3-object D",
                y_label="Method A3-object D",
                out_path=plots / "object_A3_objects_YX_grid.png",
            )
            save_summary_bar(summary_df, "object_A3", "unit_denominator", "AreaWeighted_MAE", plots / "summary_object_A3_AreaWeighted_MAE.png", "Area-weighted MAE of D ↓", ascending=True)
            save_pareto(summary_df, "object_A3", "unit_denominator", plots / "pareto_object_A3_IoU_vs_MAE.png")
            save_error_violin_or_box(object_df, "object_A3", "absolute_error", "pair_valid", "Object-level absolute error distribution: A3 objects", plots / "violin_object_A3_abs_error.png")

# ============================================================
# 10. Main
# ============================================================
def main() -> None:
    args = parse_args()
    save_excel = not bool(args.no_excel)

    method_root = Path(args.method_root)
    method_parent = Path(args.method_parent)
    needs_latest = (not method_root.exists()) or (not (method_root / "runs").exists())
    if needs_latest and (bool(args.auto_latest) or method_root == method_parent):
        method_root = choose_latest_method_root(method_parent)
    if (not method_root.exists()) or (not (method_root / "runs").exists()):
        raise FileNotFoundError(
            f"METHOD_COMPARE_ROOT not found: {method_root}\n"
            f"Please edit METHOD_COMPARE_ROOT in the script, pass --method-root, or use --auto-latest."
        )

    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else method_root / OUTPUT_SUBDIR
    ensure_dir(output_dir)

    paths = resolve_paths_from_config(method_root, args)
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{key} not found: {p}")

    method_runs = discover_method_runs(method_root)
    parents, maps, availability_df = collect_common_parents(paths, method_runs)
    availability_df.to_csv(output_dir / "input_availability_report.csv", index=False, encoding="utf-8-sig")
    if not parents:
        raise RuntimeError("No parent image/mask names are common across A1/A2/A3 and all method A4 folders.")

    run_plan = pd.DataFrame(method_runs)
    run_plan.to_csv(output_dir / "discovered_method_runs.csv", index=False, encoding="utf-8-sig")

    print(f"[INFO] Method root: {method_root}")
    print(f"[INFO] Output dir : {output_dir}")
    print(f"[INFO] Common parent count: {len(parents)}")
    print(f"[INFO] Method runs: {len(method_runs)}")

    window_rows: List[Dict] = []
    object_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    # Load M_GT/R_GT/M_Pred once per parent, then iterate all method R_Pred masks.
    for parent in tqdm(parents, desc="Window/object scatter data"):
        try:
            A1 = read_mask01(maps["A1"][parent])
            A2 = read_mask01(maps["A2"][parent])
            A3 = read_mask01(maps["A3"][parent])
            H, W = A1.shape
            A2 = align_mask(A2, H, W, bool(args.resize_masks_to_image), "A2")
            A3 = align_mask(A3, H, W, bool(args.resize_masks_to_image), "A3")

            for mr in method_runs:
                comp = mr["comp_variant"]
                method = mr["method"]
                key = f"A4::{comp}::{method}"
                A4 = read_mask01(maps[key][parent])
                A4 = align_mask(A4, H, W, bool(args.resize_masks_to_image), f"A4 {comp}/{method}")

                if not bool(args.skip_window):
                    window_rows.extend(process_window_rows_for_method(parent, A1, A2, A3, A4, comp, method, args))
                if not bool(args.skip_object):
                    object_rows.extend(process_object_rows_for_method(parent, A1, A2, A3, A4, comp, method, args))

        except Exception as e:
            skipped_rows.append({"parent": parent, "reason": repr(e)})
            print(f"[WARN] skipped {parent}: {e}")

    window_df = pd.DataFrame(window_rows)
    object_df = pd.DataFrame(object_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    window_df.to_csv(output_dir / "window_level_scatter_data.csv", index=False, encoding="utf-8-sig")
    object_df.to_csv(output_dir / "object_level_scatter_data.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(output_dir / "skipped_report.csv", index=False, encoding="utf-8-sig")

    summary_df = build_all_summaries(window_df, object_df)
    summary_df.to_csv(output_dir / "unit_level_summary.csv", index=False, encoding="utf-8-sig")

    save_all_plots(window_df, object_df, summary_df, output_dir)

    if save_excel:
        try:
            with pd.ExcelWriter(output_dir / "method_granularity_validation_summary.xlsx", engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="unit_level_summary", index=False)
                if not window_df.empty:
                    # Excel has row limits; save full CSV anyway. Keep Excel manageable.
                    window_df.head(50000).to_excel(writer, sheet_name="window_data_head", index=False)
                if not object_df.empty:
                    object_df.head(50000).to_excel(writer, sheet_name="object_data_head", index=False)
                run_plan.to_excel(writer, sheet_name="method_runs", index=False)
                skipped_df.to_excel(writer, sheet_name="skipped", index=False)
        except Exception as e:
            print(f"[WARN] Excel export failed: {e}")

    p3_config = read_json(method_root / "run_config.json")
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "window_object_scatter_validation_from_P3_method_outputs",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "METHOD_COMPARE_ROOT": str(method_root),
            "P3_MODE": p3_config.get("MODE", ""),
            "P3_BASELINE_METHOD": p3_config.get("BASELINE_METHOD", ""),
            "P3_BIN_OTP_TRANSITION_CENTER": p3_config.get("BIN_OTP_TRANSITION_CENTER", None),
            "P3_BIN_OTP_TRANSITION_TAU": p3_config.get("BIN_OTP_TRANSITION_TAU", None),
            "P3_FIXED_THRESHOLD_STRICT": p3_config.get("FIXED_THRESHOLD_STRICT", None),
            "P3_FIXED_THRESHOLD_LOOSE": p3_config.get("FIXED_THRESHOLD_LOOSE", None),
            "P3_ENABLE_GUARDED_LOCAL_FALLBACK": p3_config.get("ENABLE_GUARDED_LOCAL_FALLBACK", None),
            "OUTPUT_DIR": str(output_dir),
            "IMAGE_DIR": str(paths["image_dir"]),
            "A1_GT_ORIG_DIR": str(paths["gt_orig_dir"]),
            "A2_GT_BIN_DIR": str(paths["gt_bin_dir"]),
            "A3_PRED_ORIG_DIR": str(paths["pred_orig_dir"]),
            "WINDOW_SIZE": int(args.window_size),
            "STRIDE": int(args.stride),
            "MIN_GT_WINDOW_AREA": int(args.min_gt_window_area),
            "MIN_PRED_WINDOW_AREA": int(args.min_pred_window_area),
            "MIN_PAIR_WINDOW_AREA": int(args.min_pair_window_area),
            "MIN_A1_OBJECT_AREA": int(args.min_a1_object_area),
            "MIN_A3_OBJECT_AREA": int(args.min_a3_object_area),
            "ERROR_BAND": float(ERROR_BAND),
            "FORMULA_WINDOW_A1DEN": "D_GT=1-|A2∩A1∩W|/|A1∩W|; D_Method=1-|A4∩A1∩W|/|A1∩W|",
            "FORMULA_WINDOW_A3DEN": "D_GT=1-|A2∩A1∩W|/|A1∩W|; D_Method=1-|A4∩A3∩W|/|A3∩W|",
            "FORMULA_WINDOW_A3PURE": "D_GT_on_A3=1-|A2∩A3∩W|/|A3∩W|; D_Method=1-|A4∩A3∩W|/|A3∩W|",
            "FORMULA_OBJECT_A1": "A1 connected components as units; D_GT=1-|A2∩obj|/|obj|; D_Method=1-|A4∩obj|/|obj|",
            "FORMULA_OBJECT_A3": "A3 connected components as units; D_GT=1-|A2∩obj|/|obj|; D_Method=1-|A4∩obj|/|obj|",
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Window scatter CSV:", output_dir / "window_level_scatter_data.csv")
    print("Object scatter CSV:", output_dir / "object_level_scatter_data.csv")
    print("Summary CSV       :", output_dir / "unit_level_summary.csv")
    print("Plots folder      :", output_dir / "plots")


if __name__ == "__main__":
    main()
