# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
#  Parameter-sweep comparison script for WKDE-GMM outputs
# ------------------------------------------------------------
# Compatible with: sensitivity_parameter_sweep.py
#
# It scans a Parameter_Sweep_* folder, finds every run folder containing
# 03_final_mask, compares R_Pred final masks against GT-binarization masks,
# and outputs the same core plot types as the previous comparison script:
#   - horizontal bar plots
#   - per-image boxplots
#   - metric heatmap
#   - precision-recall scatter
#   - characteristic hard cases
#
# Improvements:
#   - reads sweep_param / sweep_value from run_config.json
#   - supports nested parameter-group folders
#   - produces both global plots and per-parameter plots
#   - duplicates the single baseline into each parameter group for plotting
#   - documents the fixed guarded fallback branch inside the P1 baseline
#   - larger fonts for PPT / manuscript figures
# ============================================================


# ============================================================
# User config: edit these paths first
# ============================================================
SWEEP_ROOT = Path(r"")  # TODO: set path
GT_DIR = Path(r"")  # R_GT = GT-binarization mask folder

PRED_MASK_SUBDIR = "03_final_mask"
OUTPUT_DIR = SWEEP_ROOT / "cmp_binmask_bigfont1"

RECOMMENDED_BIN_OTP_BASELINE = {
    "BIN_OTP_TRANSITION_CENTER": 90.0,
    "BIN_OTP_TRANSITION_TAU": 6.0,
    "FIXED_THRESHOLD_STRICT": 205.0,
    "FIXED_THRESHOLD_LOOSE": 165.0,
}
GUARDED_BRANCH_NOTE = "guarded fallback branch is enabled inside the baseline decision tree and is not a swept parameter"

# Optional: compare only one parameter group, e.g. "01_KDE_PEAK_PROMINENCE".
# Use "all" to scan the entire sweep folder.
GROUP_FILTER = "all"

PRIMARY_SORT_METRIC = "macro_dice_mean"
N_HARDEST_CASES = 4
EXPORT_EXCEL = True
RESIZE_PRED_TO_GT = False
# Runs with too few matched masks are usually failed/partial sweep runs.
# They are still saved in run_level_summary_all.csv, but excluded from plots/ranking by default.
MIN_COMPLETENESS_RATIO = 0.95
INCLUDE_INCOMPLETE_RUNS_IN_PLOTS = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Bigger fonts for direct PPT / paper use
FONT_TITLE = 20
FONT_AXIS = 17
FONT_TICK = 14
FONT_LEGEND = 13
FONT_ANNOT = 12
FONT_SUPTITLE = 22
FIG_DPI = 300

PARAM_ORDER = [
    "KDE_PEAK_PROMINENCE",
    "BIN_OTP_TRANSITION_CENTER",
    "BIN_OTP_TRANSITION_TAU",
    "FIXED_THRESHOLD_STRICT",
    "FIXED_THRESHOLD_LOOSE",
    "KDE_BANDWIDTH_SIGMA",
    "ENABLE_SHADOW_COMPENSATION",
    "OBJECT_UNIFORM_P95_P05_MAX",
]

# Short folder names are used for new outputs to avoid Windows MAX_PATH issues.
# The reader still accepts both the old long group names and these short names.
PARAM_GROUP_NAME = {
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

LONG_GROUP_NAME_TO_PARAM = {
    "00_BASE": "BASE",
    "01_KDE_PEAK_PROMINENCE": "KDE_PEAK_PROMINENCE",
    "02_BIN_OTP_TRANSITION_CENTER": "BIN_OTP_TRANSITION_CENTER",
    "03_BIN_OTP_TRANSITION_TAU": "BIN_OTP_TRANSITION_TAU",
    "04_FIXED_THRESHOLD_STRICT": "FIXED_THRESHOLD_STRICT",
    "05_FIXED_THRESHOLD_LOOSE": "FIXED_THRESHOLD_LOOSE",
    "06_KDE_BANDWIDTH_SIGMA": "KDE_BANDWIDTH_SIGMA",
    "07_ENABLE_SHADOW_COMPENSATION": "ENABLE_SHADOW_COMPENSATION",
    "08_OBJECT_UNIFORM_P95_P05_MAX": "OBJECT_UNIFORM_P95_P05_MAX",
}
GROUP_NAME_TO_PARAM = {v: k for k, v in PARAM_GROUP_NAME.items()}
GROUP_NAME_TO_PARAM.update(LONG_GROUP_NAME_TO_PARAM)

SHORT_PARAM = {
    "BASE": "BASE",
    "KDE_PEAK_PROMINENCE": "P",
    "BIN_OTP_TRANSITION_CENTER": "C0",
    "BIN_OTP_TRANSITION_TAU": "Tau",
    "FIXED_THRESHOLD_STRICT": "Ts",
    "FIXED_THRESHOLD_LOOSE": "Tl",
    "ENABLE_HIGH_CONTRAST_SINGLE_PEAK_ADAPTIVE": "HCSP",
    "KDE_BANDWIDTH_SIGMA": "B",
    "ENABLE_SHADOW_COMPENSATION": "SC",
    "OBJECT_UNIFORM_P95_P05_MAX": "U",
}


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare WKDE-GMM parameter-sweep final masks against GT binary masks with big-font plots."
    )
    parser.add_argument("--sweep-root", type=str, default=str(SWEEP_ROOT), help="Parameter_Sweep_* root folder")
    parser.add_argument("--gt-dir", type=str, default=str(GT_DIR), help="GT-binarization mask folder, A2")
    parser.add_argument("--output-dir", type=str, default="", help="Output folder. Empty -> sweep_root/cmp_binmask_bigfont")
    parser.add_argument("--group-filter", type=str, default=GROUP_FILTER, help="all or a group folder name such as 01_KDE_PEAK_PROMINENCE")
    parser.add_argument("--primary-sort-metric", type=str, default=PRIMARY_SORT_METRIC)
    parser.add_argument("--n-hardest-cases", type=int, default=N_HARDEST_CASES)
    parser.add_argument("--resize-pred-to-gt", action="store_true", default=RESIZE_PRED_TO_GT)
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--no-per-parameter-plots", action="store_true")
    parser.add_argument("--min-completeness-ratio", type=float, default=MIN_COMPLETENESS_RATIO,
                        help="Exclude partial/failed runs from plots/ranking if n_images_compared / n_GT_images is below this ratio.")
    parser.add_argument("--include-incomplete-runs-in-plots", action="store_true", default=INCLUDE_INCOMPLETE_RUNS_IN_PLOTS,
                        help="Use even partial runs in plots/ranking. Usually not recommended.")
    return parser.parse_args()


# ============================================================
# Basic IO
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_case_filename(stem: str, idx: int) -> str:
    # Keep filenames short for Windows path safety. The full image name is stored in characteristic_case_mapping.csv.
    token = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stem))[:36]
    return f"case_{idx:02d}_{token}.png"


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_mask_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


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


def build_mask_map(folder: Path, strip_final_mask_suffix: bool = False) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in sorted(folder.glob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        key = normalize_stem(stem) if strip_final_mask_suffix else stem
        # If both parent.png and parent__final_mask.png exist, prefer parent.png.
        if key in out:
            old = out[key]
            if old.stem.endswith("__final_mask") and not p.stem.endswith("__final_mask"):
                out[key] = p
        else:
            out[key] = p
    return out


def safe_value_token(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    try:
        fv = float(v)
        if abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return f"{fv:g}"
    except Exception:
        return str(v)


def short_param_name(param: str) -> str:
    return SHORT_PARAM.get(str(param), str(param)[:8])


def global_plot_label(param: str, value) -> str:
    if str(param) == "BASE":
        return "Baseline"
    return f"{short_param_name(str(param))}={safe_value_token(value)}"


def group_plot_label(value) -> str:
    return safe_value_token(value)


def numeric_or_bool_sort_key(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        if v.lower() == "true":
            return 1
        if v.lower() == "false":
            return 0
    try:
        return float(v)
    except Exception:
        return str(v)


# ============================================================
# Run discovery and metadata
# ============================================================
def discover_run_dirs(sweep_root: Path, group_filter: str, output_dir: Path) -> List[Path]:
    if not sweep_root.exists():
        raise FileNotFoundError(f"SWEEP_ROOT not found: {sweep_root}")

    run_dirs = []
    for final_dir in sweep_root.rglob(PRED_MASK_SUBDIR):
        if not final_dir.is_dir():
            continue
        run_dir = final_dir.parent
        # Avoid re-scanning our own output folder if it is inside sweep_root.
        try:
            if output_dir.resolve() in [p.resolve() for p in run_dir.parents] or run_dir.resolve() == output_dir.resolve():
                continue
        except Exception:
            pass
        if group_filter != "all" and run_dir.parent.name != group_filter:
            continue
        run_dirs.append(run_dir)

    run_dirs = sorted(set(run_dirs), key=lambda p: (p.parent.name, p.name))
    if not run_dirs:
        raise RuntimeError(f"No run folders containing '{PRED_MASK_SUBDIR}' found under: {sweep_root}")
    return run_dirs


def load_base_params(sweep_root: Path) -> Dict:
    p = sweep_root / "base_tunable_params.json"
    return read_json(p)


def load_sweep_summary(sweep_root: Path) -> pd.DataFrame:
    p = sweep_root / "parameter_sweep_summary_by_config.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def infer_metadata_from_run_dir(run_dir: Path) -> Dict:
    group_name = run_dir.parent.name
    param = GROUP_NAME_TO_PARAM.get(group_name, "UNKNOWN")
    value = run_dir.name
    if param != "UNKNOWN":
        # Try to remove short prefix such as pk_0p005.
        parts = run_dir.name.split("_", 1)
        if len(parts) == 2:
            value = parts[1].replace("p", ".")
    if group_name == "00_BASE" or run_dir.name.lower() == "baseline":
        param = "BASE"
        value = "baseline"
    return {
        "sweep_param": param,
        "sweep_value": value,
        "group_name": group_name,
        "run_tag": run_dir.name,
        "is_baseline": param == "BASE",
        "active_params": {},
    }


def load_run_metadata(run_dir: Path) -> Dict:
    cfg = read_json(run_dir / "run_config.json")
    meta = infer_metadata_from_run_dir(run_dir)
    if cfg:
        param = cfg.get("SWEEP_PARAM", cfg.get("sweep_param", meta["sweep_param"]))
        value = cfg.get("SWEEP_VALUE", cfg.get("sweep_value", meta["sweep_value"]))
        active_params = cfg.get("ACTIVE_PARAMS", cfg.get("params", {}))
        meta.update({
            "sweep_param": param,
            "sweep_value": value,
            "group_name": run_dir.parent.name,
            "run_tag": run_dir.name,
            "is_baseline": str(param) == "BASE" or run_dir.name.lower() == "baseline",
            "active_params": active_params if isinstance(active_params, dict) else {},
        })
    meta["plot_label"] = global_plot_label(str(meta["sweep_param"]), meta["sweep_value"])
    meta["run_dir"] = str(run_dir)
    meta["run_name"] = run_dir.name
    return meta


# ============================================================
# Metrics
# ============================================================
def confusion_stats(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.logical_and(gt, pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def read_method_counts(run_dir: Path) -> Dict[str, float]:
    info_path = run_dir / "all_output_info.csv"
    default = {
        "n_objects": np.nan,
        "n_method_weighted_kdegmm": np.nan,
        "n_method_adaptive_hcsp": np.nan,
        "n_method_fixed_loose": np.nan,
        "n_method_fixed_strict": np.nan,
        "n_method_strict_unreliable_bg": np.nan,
        "n_method_bin_otp_smooth": np.nan,
        "adaptive_hcsp_object_ratio": np.nan,
        "weighted_kdegmm_object_ratio": np.nan,
        "bin_otp_smooth_object_ratio": np.nan,
    }
    if not info_path.exists():
        return default
    try:
        info_df = pd.read_csv(info_path)
    except Exception:
        return default
    if info_df.empty:
        out = dict(default)
        out["n_objects"] = 0
        return out
    if "final_method" not in info_df.columns:
        out = dict(default)
        out["n_objects"] = int(len(info_df))
        return out

    methods = info_df["final_method"].astype(str)
    n = int(len(methods))
    n_wk = int((methods == "weighted_kdegmm").sum())
    n_adaptive = int(methods.str.contains("adaptive", case=False, regex=False).sum())
    n_loose = int((methods == "fixed_threshold_loose").sum())
    n_strict = int(methods.str.startswith("fixed_threshold_strict").sum())
    n_unrel = int((methods == "fixed_threshold_strict_unreliable_bg").sum())
    n_bin_otp = int((methods == "bin_otp_smooth_strict_loose").sum())
    return {
        "n_objects": n,
        "n_method_weighted_kdegmm": n_wk,
        "n_method_adaptive_hcsp": n_adaptive,
        "n_method_fixed_loose": n_loose,
        "n_method_fixed_strict": n_strict,
        "n_method_strict_unreliable_bg": n_unrel,
        "n_method_bin_otp_smooth": n_bin_otp,
        "adaptive_hcsp_object_ratio": n_adaptive / max(n, 1),
        "weighted_kdegmm_object_ratio": n_wk / max(n, 1),
        "bin_otp_smooth_object_ratio": n_bin_otp / max(n, 1),
    }


def compare_one_run(run_dir: Path, gt_map: Dict[str, Path], resize_pred: bool = False) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    pred_dir = run_dir / PRED_MASK_SUBDIR
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction mask folder not found: {pred_dir}")

    meta = load_run_metadata(run_dir)
    pred_map = build_mask_map(pred_dir, strip_final_mask_suffix=True)
    common_names = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    missing_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))
    missing_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))

    rows = []
    for stem in common_names:
        gt = read_mask_binary(gt_map[stem])
        pred = read_mask_binary(pred_map[stem])

        if gt.shape != pred.shape:
            if resize_pred:
                pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST))
                pred = (pred > 127).astype(np.uint8)
            else:
                raise ValueError(f"Shape mismatch in run={run_dir.name}, image={stem}: GT={gt.shape}, Pred={pred.shape}")

        m = confusion_stats(gt, pred)
        rows.append({
            **{k: meta[k] for k in ["run_name", "run_dir", "group_name", "sweep_param", "sweep_value", "plot_label", "is_baseline"]},
            "image_name": stem,
            "height": int(gt.shape[0]),
            "width": int(gt.shape[1]),
            "n_pixels": int(gt.size),
            **m,
        })

    per_image_df = pd.DataFrame(rows)
    method_counts = read_method_counts(run_dir)

    summary_base = {k: meta[k] for k in ["run_name", "run_dir", "group_name", "sweep_param", "sweep_value", "plot_label", "is_baseline"]}
    summary_base.update({f"param_{k}": v for k, v in meta.get("active_params", {}).items()})

    if per_image_df.empty:
        summary = {
            **summary_base,
            "n_images_compared": 0,
            "n_missing_gt": len(missing_gt),
            "n_missing_pred": len(missing_pred),
            **method_counts,
        }
    else:
        tp = int(per_image_df["tp"].sum())
        tn = int(per_image_df["tn"].sum())
        fp = int(per_image_df["fp"].sum())
        fn = int(per_image_df["fn"].sum())
        eps = 1e-12
        summary = {
            **summary_base,
            "n_images_compared": int(len(per_image_df)),
            "n_missing_gt": len(missing_gt),
            "n_missing_pred": len(missing_pred),
            "macro_precision_mean": float(per_image_df["precision"].mean()),
            "macro_recall_mean": float(per_image_df["recall"].mean()),
            "macro_dice_mean": float(per_image_df["dice"].mean()),
            "macro_iou_mean": float(per_image_df["iou"].mean()),
            "macro_accuracy_mean": float(per_image_df["accuracy"].mean()),
            "macro_specificity_mean": float(per_image_df["specificity"].mean()),
            "macro_fpr_mean": float(per_image_df["fpr"].mean()),
            "macro_fnr_mean": float(per_image_df["fnr"].mean()),
            "macro_dice_std": float(per_image_df["dice"].std(ddof=0)),
            "macro_iou_std": float(per_image_df["iou"].std(ddof=0)),
            "micro_precision": float(tp / (tp + fp + eps)),
            "micro_recall": float(tp / (tp + fn + eps)),
            "micro_dice": float(2 * tp / (2 * tp + fp + fn + eps)),
            "micro_iou": float(tp / (tp + fp + fn + eps)),
            "micro_accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
            "micro_specificity": float(tn / (tn + fp + eps)),
            **method_counts,
        }

    if len(missing_gt) == 0 and len(missing_pred) == 0:
        missing_df = pd.DataFrame(columns=["run_name", "plot_label", "missing_gt_for_pred_name", "missing_pred_for_gt_name"])
    else:
        n = max(len(missing_gt), len(missing_pred))
        missing_df = pd.DataFrame({
            "run_name": [run_dir.name] * n,
            "run_dir": [str(run_dir)] * n,
            "plot_label": [meta["plot_label"]] * n,
            "missing_gt_for_pred_name": missing_gt + [""] * (n - len(missing_gt)),
            "missing_pred_for_gt_name": missing_pred + [""] * (n - len(missing_pred)),
        })

    return per_image_df, summary, missing_df


# ============================================================
# Plot helpers with big fonts
# ============================================================
def setup_matplotlib_fonts() -> None:
    plt.rcParams.update({
        "font.size": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_SUPTITLE,
    })


def save_bar_plot(df: pd.DataFrame, metric: str, out_path: Path, title: str, ascending: bool, xlabel: Optional[str] = None) -> None:
    if df.empty or metric not in df.columns:
        return
    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(metric, ascending=ascending)

    fig_h = max(5.2, 0.7 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    bars = ax.barh(plot_df["plot_label"], plot_df[metric])
    ax.set_xlabel(xlabel or metric)
    ax.set_ylabel("Run")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.45)

    # Add numeric labels at bar ends.
    vals = plot_df[metric].to_numpy(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    pad = (vmax - vmin) * 0.015 if vmax > vmin else 0.005
    for bar, val in zip(bars, vals):
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=FONT_ANNOT)

    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_metric_boxplot(per_image_df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if per_image_df.empty or metric not in per_image_df.columns:
        return

    labels, data = [], []
    # Preserve a stable label order by median performance.
    med = per_image_df.groupby("plot_label")[metric].median().sort_values(ascending=False)
    for label in med.index:
        vals = pd.to_numeric(per_image_df.loc[per_image_df["plot_label"] == label, metric], errors="coerce").dropna().values
        if vals.size > 0:
            labels.append(label)
            data.append(vals)
    if not data:
        return

    fig_w = max(10.5, 0.8 * len(labels) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_xlabel("Run")
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(summary_df: pd.DataFrame, metrics: List[str], out_path: Path, title: str) -> None:
    if summary_df.empty:
        return
    cols = [c for c in metrics if c in summary_df.columns]
    if not cols:
        return
    plot_df = summary_df[["plot_label"] + cols].copy().set_index("plot_label")
    for c in cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    arr = plot_df.values.astype(float)

    fig_w = max(12.0, 1.65 * len(cols) + 4)
    fig_h = max(5.5, 0.72 * len(plot_df) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title(title, fontweight="bold")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = f"{val:.3f}" if np.isfinite(val) else "nan"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FONT_ANNOT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_scatter(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if summary_df.empty or "macro_precision_mean" not in summary_df.columns or "macro_recall_mean" not in summary_df.columns:
        return
    df = summary_df.copy()
    df["macro_precision_mean"] = pd.to_numeric(df["macro_precision_mean"], errors="coerce")
    df["macro_recall_mean"] = pd.to_numeric(df["macro_recall_mean"], errors="coerce")
    df = df.dropna(subset=["macro_precision_mean", "macro_recall_mean"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    x = df["macro_recall_mean"].to_numpy(float)
    y = df["macro_precision_mean"].to_numpy(float)
    ax.scatter(x, y, alpha=0.85, s=90, edgecolors="black", linewidths=0.7)
    for _, row in df.iterrows():
        ax.text(row["macro_recall_mean"] + 0.003, row["macro_precision_mean"] + 0.003, str(row["plot_label"]), fontsize=FONT_ANNOT)
    ax.set_xlabel("Macro Recall")
    ax.set_ylabel("Macro Precision")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_xlim(max(0, np.nanmin(x) - 0.04), min(1, np.nanmax(x) + 0.04))
    ax.set_ylim(max(0, np.nanmin(y) - 0.04), min(1, np.nanmax(y) + 0.04))
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def overlay_diff(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # TP=white, FP=red, FN=blue, TN=black. Same visual convention as the pasted script.
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    vis[tp] = (255, 255, 255)
    vis[fp] = (255, 0, 0)
    vis[fn] = (0, 0, 255)
    return vis


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)


def save_characteristic_cases(best_run_dir: Path, best_label: str, gt_map: Dict[str, Path], per_image_df_best: pd.DataFrame, out_dir: Path, n_cases: int = 4, resize_pred: bool = False) -> None:
    ensure_dir(out_dir)
    if per_image_df_best.empty:
        return
    pred_dir = best_run_dir / PRED_MASK_SUBDIR
    pred_map = build_mask_map(pred_dir, strip_final_mask_suffix=True)
    hard_cases = per_image_df_best.sort_values("iou", ascending=True).head(n_cases)

    mapping_rows = []
    for case_idx, (_, row) in enumerate(hard_cases.iterrows(), start=1):
        stem = row["image_name"]
        if stem not in gt_map or stem not in pred_map:
            continue
        gt = read_mask_binary(gt_map[stem])
        pred = read_mask_binary(pred_map[stem])
        if gt.shape != pred.shape and resize_pred:
            pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST))
            pred = (pred > 127).astype(np.uint8)
        if gt.shape != pred.shape:
            continue

        gt_rgb = mask_to_rgb(gt)
        pred_rgb = mask_to_rgb(pred)
        diff_rgb = overlay_diff(gt, pred)
        h, w = gt.shape
        canvas = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
        canvas[:, 0:w] = gt_rgb
        canvas[:, w:2 * w] = pred_rgb
        canvas[:, 2 * w:3 * w] = diff_rgb

        fig = plt.figure(figsize=(14.5, 5.0))
        plt.imshow(canvas)
        plt.axis("off")
        plt.title(
            f"{best_label} | {stem} | IoU={row['iou']:.4f}, Dice={row['dice']:.4f}, "
            f"Prec={row['precision']:.4f}, Rec={row['recall']:.4f}\n"
            f"Left: GT | Middle: Pred | Right: TP/FP/FN diff",
            fontsize=FONT_TITLE,
            fontweight="bold",
        )

        out_name = safe_case_filename(stem, case_idx)
        out_path = out_dir / out_name
        mapping_rows.append({
            "case_index": case_idx,
            "image_name": stem,
            "saved_file": out_name,
            "best_label": best_label,
            "run_dir": str(best_run_dir),
            "iou": float(row["iou"]),
            "dice": float(row["dice"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        })

        try:
            ensure_dir(out_path.parent)
            # Avoid bbox_inches='tight' here because it can increase path/rendering failures on Windows.
            fig.tight_layout()
            fig.savefig(out_path, dpi=FIG_DPI)
        except Exception as e:
            print(f"[WARN] failed to save characteristic case {stem}: {e}")
        finally:
            plt.close(fig)

    if mapping_rows:
        pd.DataFrame(mapping_rows).to_csv(out_dir / "characteristic_case_mapping.csv", index=False, encoding="utf-8-sig")


# ============================================================
# Ranking and summaries
# ============================================================
def build_rank_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    df = summary_df.copy()
    rank_specs = {
        "rank_macro_dice": ("macro_dice_mean", False),
        "rank_macro_iou": ("macro_iou_mean", False),
        "rank_macro_precision": ("macro_precision_mean", False),
        "rank_macro_recall": ("macro_recall_mean", False),
        "rank_macro_fpr": ("macro_fpr_mean", True),
        "rank_macro_fnr": ("macro_fnr_mean", True),
    }
    for rank_col, (metric, ascending) in rank_specs.items():
        if metric in df.columns:
            df[rank_col] = pd.to_numeric(df[metric], errors="coerce").rank(method="min", ascending=ascending)
    used = [c for c in rank_specs if c in df.columns]
    if used:
        df["paper_score_mean_rank"] = df[used].mean(axis=1)
        df = df.sort_values("paper_score_mean_rank", ascending=True)
    elif PRIMARY_SORT_METRIC in df.columns:
        df = df.sort_values(PRIMARY_SORT_METRIC, ascending=False)
    return df


def write_summary_text(summary_df: pd.DataFrame, out_path: Path, primary_metric: str) -> None:
    lines = [
        "Bin-OTP parameter-sweep binary-mask consistency summary against GT-binarization mask",
        "=" * 76,
        f"Primary sort metric: {primary_metric}",
        "Recommended Bin-OTP baseline: C0=90, tau=6, Ts=205, Tl=165",
        f"Branch note: {GUARDED_BRANCH_NOTE}.",
        "",
    ]
    if summary_df.empty:
        lines.append("No valid runs were compared.")
    else:
        if primary_metric in summary_df.columns:
            best = summary_df.sort_values(primary_metric, ascending=False).iloc[0]
            lines.extend([
                f"Best run by {primary_metric}:",
                f"  plot_label : {best['plot_label']}",
                f"  group      : {best.get('group_name', '')}",
                f"  run_name   : {best['run_name']}",
                f"  run_dir    : {best['run_dir']}",
                f"  {primary_metric}: {best[primary_metric]:.6f}",
                "",
            ])
        lines.append("All runs:")
        sort_df = summary_df.sort_values(primary_metric, ascending=False) if primary_metric in summary_df.columns else summary_df
        for _, row in sort_df.iterrows():
            parts = [
                f"{row['plot_label']}",
                f"Group={row.get('group_name', '')}",
                f"Dice={row.get('macro_dice_mean', np.nan):.6f}",
                f"IoU={row.get('macro_iou_mean', np.nan):.6f}",
                f"Precision={row.get('macro_precision_mean', np.nan):.6f}",
                f"Recall={row.get('macro_recall_mean', np.nan):.6f}",
                f"FPR={row.get('macro_fpr_mean', np.nan):.6f}",
                f"FNR={row.get('macro_fnr_mean', np.nan):.6f}",
                f"WK_ratio={row.get('weighted_kdegmm_object_ratio', np.nan):.4f}",
                f"HCSP_ratio={row.get('adaptive_hcsp_object_ratio', np.nan):.4f}",
            ]
            lines.append("  " + " | ".join(parts))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def merge_external_sweep_summary(summary_df: pd.DataFrame, sweep_summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or sweep_summary_df.empty:
        return summary_df
    df = summary_df.copy()
    ext = sweep_summary_df.copy()
    if "run_dir" not in ext.columns:
        return df
    ext["_run_dir_norm"] = ext["run_dir"].astype(str).map(lambda s: str(Path(s)))
    df["_run_dir_norm"] = df["run_dir"].astype(str).map(lambda s: str(Path(s)))

    keep_cols = ["_run_dir_norm"]
    interesting_prefixes = (
        "Pred_Ref_vs_All_GT", "Orig_GT_vs_All_GT", "Bi_GT_vs_All_GT", "Proposed_A3_vs_All_GT",
        "A4_vs_A2_within_A1", "A4_vs_A2_within_A3_inter_A1",
    )
    for c in ext.columns:
        if c.startswith(interesting_prefixes) or c in ["status", "error"]:
            if c not in keep_cols:
                keep_cols.append(c)
    if len(keep_cols) <= 1:
        return df.drop(columns=["_run_dir_norm"])

    ext = ext[keep_cols].drop_duplicates("_run_dir_norm")
    ext = ext.rename(columns={c: f"sweep_{c}" for c in ext.columns if c != "_run_dir_norm"})
    df = df.merge(ext, on="_run_dir_norm", how="left")
    return df.drop(columns=["_run_dir_norm"])


# ============================================================
# Plot suites
# ============================================================
def save_core_plot_suite(summary_df: pd.DataFrame, per_image_df: pd.DataFrame, plots_dir: Path, primary_metric: str, title_suffix: str = "") -> None:
    ensure_dir(plots_dir)
    suffix = f" {title_suffix}" if title_suffix else ""
    save_bar_plot(summary_df, "macro_dice_mean", plots_dir / "bar_macro_dice.png", f"Macro Dice across Runs{suffix}", ascending=False, xlabel="Macro Dice")
    save_bar_plot(summary_df, "macro_iou_mean", plots_dir / "bar_macro_iou.png", f"Macro IoU across Runs{suffix}", ascending=False, xlabel="Macro IoU")
    save_bar_plot(summary_df, "macro_precision_mean", plots_dir / "bar_macro_precision.png", f"Macro Precision across Runs{suffix}", ascending=False, xlabel="Macro Precision")
    save_bar_plot(summary_df, "macro_recall_mean", plots_dir / "bar_macro_recall.png", f"Macro Recall across Runs{suffix}", ascending=False, xlabel="Macro Recall")
    save_bar_plot(summary_df, "macro_fpr_mean", plots_dir / "bar_macro_fpr.png", f"Macro False Positive Rate across Runs{suffix}", ascending=True, xlabel="Macro FPR")
    save_bar_plot(summary_df, "macro_fnr_mean", plots_dir / "bar_macro_fnr.png", f"Macro False Negative Rate across Runs{suffix}", ascending=True, xlabel="Macro FNR")
    save_bar_plot(summary_df, "adaptive_hcsp_object_ratio", plots_dir / "bar_adaptive_hcsp_object_ratio.png", f"Adaptive HCSP Object Ratio across Runs{suffix}", ascending=False, xlabel="Adaptive HCSP object ratio")
    save_bar_plot(summary_df, "weighted_kdegmm_object_ratio", plots_dir / "bar_weighted_kdegmm_object_ratio.png", f"Weighted KDE-GMM Object Ratio across Runs{suffix}", ascending=False, xlabel="Weighted KDE-GMM object ratio")

    save_metric_boxplot(per_image_df, "dice", plots_dir / "box_dice.png", f"Per-image Dice{suffix}")
    save_metric_boxplot(per_image_df, "iou", plots_dir / "box_iou.png", f"Per-image IoU{suffix}")
    save_metric_boxplot(per_image_df, "precision", plots_dir / "box_precision.png", f"Per-image Precision{suffix}")
    save_metric_boxplot(per_image_df, "recall", plots_dir / "box_recall.png", f"Per-image Recall{suffix}")

    save_heatmap(
        summary_df,
        ["macro_dice_mean", "macro_iou_mean", "macro_precision_mean", "macro_recall_mean", "macro_fpr_mean", "macro_fnr_mean", "adaptive_hcsp_object_ratio", "weighted_kdegmm_object_ratio"],
        plots_dir / "heatmap_seg_metrics.png",
        f"Binary Mask Consistency Metrics Heatmap{suffix}",
    )
    save_precision_recall_scatter(summary_df, plots_dir / "scatter_precision_recall_runs.png", f"Precision-Recall Tradeoff across Runs{suffix}")


def add_baseline_for_parameter_group(summary_df: pd.DataFrame, per_image_df: pd.DataFrame, param: str, base_params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub_summary = summary_df[summary_df["sweep_param"].astype(str) == param].copy()
    sub_per = per_image_df[per_image_df["sweep_param"].astype(str) == param].copy()

    base_summary_rows = summary_df[summary_df["is_baseline"].astype(bool)] if "is_baseline" in summary_df.columns else pd.DataFrame()
    base_per_rows = per_image_df[per_image_df["is_baseline"].astype(bool)] if "is_baseline" in per_image_df.columns else pd.DataFrame()

    if not base_summary_rows.empty:
        base_summary = base_summary_rows.iloc[0].copy()
        base_value = base_params.get(param, base_summary.get(f"param_{param}", "baseline"))
        base_summary["sweep_param"] = param
        base_summary["sweep_value"] = base_value
        base_summary["group_name"] = PARAM_GROUP_NAME.get(param, param)
        base_summary["plot_label"] = group_plot_label(base_value)
        base_summary["is_baseline"] = True
        sub_summary = pd.concat([sub_summary, pd.DataFrame([base_summary])], ignore_index=True)

    if not base_per_rows.empty:
        base_value = base_params.get(param, "baseline")
        base_per = base_per_rows.copy()
        base_per["sweep_param"] = param
        base_per["sweep_value"] = base_value
        base_per["group_name"] = PARAM_GROUP_NAME.get(param, param)
        base_per["plot_label"] = group_plot_label(base_value)
        base_per["is_baseline"] = True
        sub_per = pd.concat([sub_per, base_per], ignore_index=True)

    # For group-specific figures, labels should be simple tested values rather than Param=value.
    if not sub_summary.empty:
        sub_summary["plot_label"] = sub_summary["sweep_value"].map(group_plot_label)
        sub_summary = sub_summary.sort_values("sweep_value", key=lambda s: s.map(numeric_or_bool_sort_key))
    if not sub_per.empty:
        sub_per["plot_label"] = sub_per["sweep_value"].map(group_plot_label)
    return sub_summary, sub_per


def save_per_parameter_plot_suites(summary_df: pd.DataFrame, per_image_df: pd.DataFrame, base_params: Dict, output_dir: Path, primary_metric: str, n_hardest: int, gt_map: Dict[str, Path], resize_pred: bool) -> None:
    root = output_dir / "plots_by_parameter"
    ensure_dir(root)
    cases_root = output_dir / "characteristic_cases_by_parameter"
    ensure_dir(cases_root)

    for param in PARAM_ORDER:
        sub_summary, sub_per = add_baseline_for_parameter_group(summary_df, per_image_df, param, base_params)
        if sub_summary.empty:
            continue
        param_dir = root / PARAM_GROUP_NAME.get(param, param)
        save_core_plot_suite(sub_summary, sub_per, param_dir, primary_metric, title_suffix=f"({param})")
        sub_summary.to_csv(param_dir / "run_level_summary_this_parameter.csv", index=False, encoding="utf-8-sig")
        sub_per.to_csv(param_dir / "per_image_metrics_this_parameter.csv", index=False, encoding="utf-8-sig")

        if primary_metric in sub_summary.columns:
            best = sub_summary.sort_values(primary_metric, ascending=False).iloc[0]
            best_run_dir = Path(best["run_dir"])
            if best_run_dir.exists():
                best_per = sub_per[sub_per["run_dir"].astype(str) == str(best_run_dir)].copy()
                save_characteristic_cases(best_run_dir, str(best["plot_label"]), gt_map, best_per, cases_root / PARAM_GROUP_NAME.get(param, param), n_cases=n_hardest, resize_pred=resize_pred)


# ============================================================
# Main
# ============================================================
def main() -> None:
    global PRIMARY_SORT_METRIC
    args = parse_args()
    setup_matplotlib_fonts()

    sweep_root = Path(args.sweep_root)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else sweep_root / "cmp_binmask_bigfont"
    primary_metric = str(args.primary_sort_metric)
    PRIMARY_SORT_METRIC = primary_metric
    export_excel = not bool(args.no_excel)

    ensure_dir(output_dir)
    plots_dir = output_dir / "plots_all"
    cases_dir = output_dir / "characteristic_cases"
    ensure_dir(plots_dir)
    ensure_dir(cases_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT folder not found: {gt_dir}")
    gt_map = build_mask_map(gt_dir, strip_final_mask_suffix=False)
    if not gt_map:
        raise RuntimeError(f"No GT masks found in: {gt_dir}")

    run_dirs = discover_run_dirs(sweep_root, args.group_filter, output_dir)
    print(f"[INFO] Found run folders: {len(run_dirs)}")

    all_per_image = []
    all_summary = []
    all_missing = []
    failed_rows = []

    for run_dir in run_dirs:
        print(f"[RUN] Comparing: {run_dir.parent.name}/{run_dir.name}")
        try:
            per_image_df, summary, missing_df = compare_one_run(run_dir, gt_map, resize_pred=bool(args.resize_pred_to_gt))
            all_per_image.append(per_image_df)
            all_summary.append(summary)
            all_missing.append(missing_df)
        except Exception as e:
            meta = load_run_metadata(run_dir)
            failed_rows.append({**meta, "status": "failed", "error": repr(e)})
            print(f"[WARN] Failed: {run_dir} | {e}")

    per_image_df = pd.concat(all_per_image, ignore_index=True) if all_per_image else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    missing_df = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows)

    if summary_df.empty:
        raise RuntimeError("No valid runs were compared. Check GT matching, final-mask folders, and shape consistency.")

    sweep_summary_df = load_sweep_summary(sweep_root)
    summary_df = merge_external_sweep_summary(summary_df, sweep_summary_df)

    # Mark partial/failed runs. The previous parameter sweep could leave only a few masks before failing;
    # those runs should not be used for ranking or paper plots.
    total_gt_images = max(len(gt_map), 1)
    summary_df["expected_gt_image_count"] = int(len(gt_map))
    summary_df["completeness_ratio"] = pd.to_numeric(summary_df.get("n_images_compared", 0), errors="coerce").fillna(0) / float(total_gt_images)
    summary_df["is_complete_for_plots"] = summary_df["completeness_ratio"] >= float(args.min_completeness_ratio)

    summary_all_df = summary_df.copy()
    per_image_all_df = per_image_df.copy()

    if not bool(args.include_incomplete_runs_in_plots):
        valid_run_names = set(summary_df.loc[summary_df["is_complete_for_plots"], "run_name"].astype(str))
        dropped = len(summary_df) - len(valid_run_names)
        if dropped > 0:
            print(f"[INFO] Excluding {dropped} incomplete/partial runs from plots and ranking. "
                  f"Use --include-incomplete-runs-in-plots to override.")
        summary_df = summary_df[summary_df["run_name"].astype(str).isin(valid_run_names)].copy()
        per_image_df = per_image_df[per_image_df["run_name"].astype(str).isin(valid_run_names)].copy()

    if summary_df.empty:
        raise RuntimeError("All runs were incomplete after filtering. Rerun the parameter sweep V2, or use --include-incomplete-runs-in-plots.")

    if primary_metric in summary_df.columns:
        summary_df = summary_df.sort_values(primary_metric, ascending=False)

    rank_df = build_rank_table(summary_df)
    mapping_cols = ["plot_label", "group_name", "sweep_param", "sweep_value", "run_name", "run_dir"]
    mapping_df = summary_df[[c for c in mapping_cols if c in summary_df.columns]].drop_duplicates().copy()

    summary_all_df.to_csv(output_dir / "run_level_summary_all.csv", index=False, encoding="utf-8-sig")
    per_image_all_df.to_csv(output_dir / "per_image_metrics_long_all.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "run_level_summary.csv", index=False, encoding="utf-8-sig")
    per_image_df.to_csv(output_dir / "per_image_metrics_long.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(output_dir / "missing_match_report.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(output_dir / "run_name_mapping.csv", index=False, encoding="utf-8-sig")
    rank_df.to_csv(output_dir / "paper_rank_table.csv", index=False, encoding="utf-8-sig")
    failed_df.to_csv(output_dir / "failed_run_report.csv", index=False, encoding="utf-8-sig")

    if export_excel:
        try:
            with pd.ExcelWriter(output_dir / "comparison_summary.xlsx", engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="run_level_summary", index=False)
                per_image_df.to_excel(writer, sheet_name="per_image_metrics", index=False)
                mapping_df.to_excel(writer, sheet_name="run_name_mapping", index=False)
                rank_df.to_excel(writer, sheet_name="paper_rank_table", index=False)
                missing_df.to_excel(writer, sheet_name="missing_match_report", index=False)
                failed_df.to_excel(writer, sheet_name="failed_run_report", index=False)
        except Exception as e:
            print(f"[WARN] Failed to write Excel file: {e}")

    write_summary_text(summary_df, output_dir / "analysis_summary.txt", primary_metric)

    # Global plot suite, matching the original comparison script's outputs.
    save_core_plot_suite(summary_df, per_image_df, plots_dir, primary_metric)

    # Characteristic cases based on the global best run.
    if primary_metric in summary_df.columns:
        best_row = summary_df.sort_values(primary_metric, ascending=False).iloc[0]
        best_run_dir = Path(best_row["run_dir"])
        if best_run_dir.exists():
            per_image_best = per_image_df[per_image_df["run_dir"].astype(str) == str(best_run_dir)].copy()
            save_characteristic_cases(best_run_dir, str(best_row["plot_label"]), gt_map, per_image_best, cases_dir, n_cases=int(args.n_hardest_cases), resize_pred=bool(args.resize_pred_to_gt))

    # Per-parameter plot suites with baseline inserted into each group.
    base_params = load_base_params(sweep_root)
    if not bool(args.no_per_parameter_plots):
        save_per_parameter_plot_suites(
            summary_df=summary_df,
            per_image_df=per_image_df,
            base_params=base_params,
            output_dir=output_dir,
            primary_metric=primary_metric,
            n_hardest=int(args.n_hardest_cases),
            gt_map=gt_map,
            resize_pred=bool(args.resize_pred_to_gt),
        )

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MODE": "compare_parameter_sweep_binarymask_metrics_bigfont_Bin-OTP",
            "SWEEP_ROOT": str(sweep_root),
            "GT_DIR": str(gt_dir),
            "OUTPUT_DIR": str(output_dir),
            "RECOMMENDED_BIN_OTP_BASELINE": RECOMMENDED_BIN_OTP_BASELINE,
            "GUARDED_BRANCH_NOTE": GUARDED_BRANCH_NOTE,
            "PRED_MASK_SUBDIR": PRED_MASK_SUBDIR,
            "GROUP_FILTER": str(args.group_filter),
            "PRIMARY_SORT_METRIC": primary_metric,
            "N_HARDEST_CASES": int(args.n_hardest_cases),
            "RESIZE_PRED_TO_GT": bool(args.resize_pred_to_gt),
            "EXPORT_EXCEL": bool(export_excel),
            "FONT_TITLE": FONT_TITLE,
            "FONT_AXIS": FONT_AXIS,
            "FONT_TICK": FONT_TICK,
            "FONT_LEGEND": FONT_LEGEND,
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output folder: {output_dir}")
    print(f"Run-level summary: {output_dir / 'run_level_summary.csv'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
