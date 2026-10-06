# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math
import platform

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("")  # TODO: set path
MAIN_ROOT = PROJECT_ROOT / "Combin" / "Main"
OUTPUT_ROOT = PROJECT_ROOT / "Figure" / "Fig4_5_Downstream_Model_Validation"
SOURCE_DATA = OUTPUT_ROOT / "source_data"


MODEL_SPECS = [
    {
        "model_id": "UNetPlusPlus_R34",
        "family": "UNetPlusPlus",
        "backbone": "R34",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "UNetPlusPlus" / "R34" / "04_validation_output",
    },
    {
        "model_id": "UNetPlusPlus_R50",
        "family": "UNetPlusPlus",
        "backbone": "R50",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "UNetPlusPlus" / "R50" / "04_validation_output",
    },
    {
        "model_id": "UNetPlusPlus_R101",
        "family": "UNetPlusPlus",
        "backbone": "R101",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "UNetPlusPlus" / "R101" / "04_validation_output",
    },
    {
        "model_id": "FPN_R34",
        "family": "FPN",
        "backbone": "R34",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "FPN" / "R34" / "04_validation_output",
    },
    {
        "model_id": "FPN_R50",
        "family": "FPN",
        "backbone": "R50",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "FPN" / "R50" / "04_validation_output",
    },
    {
        "model_id": "FPN_R101",
        "family": "FPN",
        "backbone": "R101",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "FPN" / "R101" / "04_validation_output",
    },
    {
        "model_id": "DeepLabV3Plus_R34",
        "family": "DeepLabV3Plus",
        "backbone": "R34",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "DeepLabV3Plus" / "R34" / "04_validation_output",
    },
    {
        "model_id": "DeepLabV3Plus_R50",
        "family": "DeepLabV3Plus",
        "backbone": "R50",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "DeepLabV3Plus" / "R50" / "04_validation_output",
    },
    {
        "model_id": "DeepLabV3Plus_R101",
        "family": "DeepLabV3Plus",
        "backbone": "R101",
        "architecture_type": "CNN",
        "result_dir": MAIN_ROOT / "DeepLabV3Plus" / "R101" / "04_validation_output",
    },
    {
        "model_id": "SegFormer_MiT-B0",
        "family": "SegFormer",
        "backbone": "MiT-B0",
        "architecture_type": "Transformer",
        "result_dir": MAIN_ROOT / "Segformer" / "MitB0" / "04_validation_output",
    },
    {
        "model_id": "SegFormer_MiT-B1",
        "family": "SegFormer",
        "backbone": "MiT-B1",
        "architecture_type": "Transformer",
        "result_dir": MAIN_ROOT / "Segformer" / "MitB1" / "04_validation_output",
    },
    {
        "model_id": "SegFormer_MiT-B2",
        "family": "SegFormer",
        "backbone": "MiT-B2",
        "architecture_type": "Transformer",
        "result_dir": MAIN_ROOT / "Segformer" / "MitB2" / "04_validation_output",
    },
    {
        "model_id": "SegFormer_MiT-B3",
        "family": "SegFormer",
        "backbone": "MiT-B3",
        "architecture_type": "Transformer",
        "result_dir": MAIN_ROOT / "Segformer" / "MitB3" / "04_validation_output",
    },
    {
        "model_id": "SegFormer_MiT-B4",
        "family": "SegFormer",
        "backbone": "MiT-B4",
        "architecture_type": "Transformer",
        "result_dir": MAIN_ROOT / "Segformer" / "MitB4" / "04_validation_output",
    },
    {
        "model_id": "SegFormer_MiT-B5",
        "family": "SegFormer",
        "backbone": "MiT-B5",
        "architecture_type": "Transformer",
        "result_dir": MAIN_ROOT / "Segformer" / "MitB5" / "04_validation_output",
    },
]


RELATIVE_FILES = {
    "global_downstream": Path("Global_Comparison_Plots") / "Pred_vs_GT_local_window_metrics_summary.csv",
    "pixel_global": Path("pixel_metrics_global_summary.csv"),
    "extra_eval": Path("Extra_Evaluation_Groups") / "extra_evaluation_groups_metrics_overview.csv",
    "image_level": Path("image_level_window_validation_summary.csv"),
    "window_level": Path("window_level_validation.csv"),
    "run_config": Path("run_config.json"),
}


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def add_model_columns(df: pd.DataFrame, spec: dict, source_key: str, source_path: Path) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "model_id", spec["model_id"])
    out.insert(1, "family", spec["family"])
    out.insert(2, "backbone", spec["backbone"])
    out.insert(3, "architecture_type", spec["architecture_type"])
    out.insert(4, "source_key", source_key)
    out.insert(5, "source_csv", str(source_path))
    return out


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def finite_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = finite_float(values)
    w = finite_float(weights)
    ok = v.notna() & w.notna() & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def collect_model(spec: dict) -> dict:
    result_dir = Path(spec["result_dir"])
    files = {k: result_dir / rel for k, rel in RELATIVE_FILES.items()}
    missing = [k for k, p in files.items() if k != "run_config" and not p.exists()]
    if missing:
        raise FileNotFoundError(f"{spec['model_id']} missing required files: {missing}")

    global_downstream = add_model_columns(
        read_csv_strict(files["global_downstream"]),
        spec,
        "global_downstream",
        files["global_downstream"],
    )
    pixel_global = add_model_columns(
        read_csv_strict(files["pixel_global"]),
        spec,
        "pixel_global",
        files["pixel_global"],
    )
    extra_eval = add_model_columns(
        read_csv_strict(files["extra_eval"]),
        spec,
        "extra_eval",
        files["extra_eval"],
    )
    image_level = add_model_columns(
        read_csv_strict(files["image_level"]),
        spec,
        "image_level",
        files["image_level"],
    )
    window_level = add_model_columns(
        read_csv_strict(files["window_level"]),
        spec,
        "window_level",
        files["window_level"],
    )

    pair_mask = to_bool_series(window_level["pair_valid"])
    pair_df = window_level[pair_mask].copy()

    summary = {
        "model_id": spec["model_id"],
        "family": spec["family"],
        "backbone": spec["backbone"],
        "architecture_type": spec["architecture_type"],
        "result_dir": str(result_dir),
        "source_global_downstream_csv": str(files["global_downstream"]),
        "source_pixel_global_csv": str(files["pixel_global"]),
        "source_extra_eval_csv": str(files["extra_eval"]),
        "source_image_level_csv": str(files["image_level"]),
        "source_window_level_csv": str(files["window_level"]),
    }

    g = global_downstream.iloc[0]
    for col in [
        "n",
        "MAE",
        "RMSE",
        "Bias_y_minus_x",
        "Pearson_r",
        "R2",
        "Within_±0.10_ratio",
        "AreaWeighted_MAE",
        "AreaWeighted_RMSE",
        "AreaWeighted_Bias",
    ]:
        summary[f"downstream_{col}"] = g.get(col, np.nan)

    for eval_name, prefix in [
        ("A3_vs_A1_segmentation", "segmentation"),
        ("A4_vs_A2_within_A1", "binarization_A4_vs_A2_within_A1"),
    ]:
        sub = pixel_global[(pixel_global["eval_name"] == eval_name) & (pixel_global["variant"] == "strict")]
        if len(sub) == 1:
            r = sub.iloc[0]
            for col in ["precision", "recall", "f1", "iou", "pixel_acc"]:
                summary[f"{prefix}_{col}"] = r.get(col, np.nan)
            for col in ["TP", "FP", "FN", "TN", "region_pixels"]:
                summary[f"{prefix}_{col}"] = r.get(col, np.nan)

    for group_name, prefix in [
        ("Segmentation_A2A3_vs_A2A1", "segmentation_effect"),
        ("Binarization_A4A1_vs_A2A1", "binarization_effect"),
    ]:
        sub = extra_eval[extra_eval["evaluation_group"] == group_name]
        if len(sub) == 1:
            r = sub.iloc[0]
            for col in [
                "n",
                "MAE",
                "RMSE",
                "Bias_y_minus_x",
                "Pearson_r",
                "R2",
                "Within_±0.10_ratio",
                "AreaWeighted_MAE",
                "AreaWeighted_RMSE",
                "AreaWeighted_Bias",
            ]:
                summary[f"{prefix}_{col}"] = r.get(col, np.nan)

    if not pair_df.empty:
        summary["derived_pair_valid_window_n"] = int(len(pair_df))
        summary["derived_abs_error_p50"] = float(finite_float(pair_df["absolute_error"]).quantile(0.50))
        summary["derived_abs_error_p75"] = float(finite_float(pair_df["absolute_error"]).quantile(0.75))
        summary["derived_abs_error_p90"] = float(finite_float(pair_df["absolute_error"]).quantile(0.90))
        summary["derived_abs_error_p95"] = float(finite_float(pair_df["absolute_error"]).quantile(0.95))
        summary["derived_abs_error_p99"] = float(finite_float(pair_df["absolute_error"]).quantile(0.99))
        summary["derived_blur_level_agreement"] = float((pair_df["GT_blur_level"] == pair_df["Pred_blur_level"]).mean())
        summary["derived_signed_error_area_weighted_mean"] = weighted_mean(pair_df["signed_error_Pred_minus_GT"], pair_df["pair_weight_area"])
        summary["derived_abs_error_area_weighted_mean_check"] = weighted_mean(pair_df["absolute_error"], pair_df["pair_weight_area"])

    if not image_level.empty:
        summary["derived_image_n"] = int(len(image_level))
        summary["derived_image_median_MAE"] = float(finite_float(image_level["MAE_Pred_vs_GT"]).median())
        summary["derived_image_iqr_MAE"] = float(
            finite_float(image_level["MAE_Pred_vs_GT"]).quantile(0.75)
            - finite_float(image_level["MAE_Pred_vs_GT"]).quantile(0.25)
        )
        summary["derived_image_mean_within_0p10"] = float(finite_float(image_level["Within_±0.10_ratio"]).mean())

    return {
        "summary": summary,
        "global_downstream": global_downstream,
        "pixel_global": pixel_global,
        "extra_eval": extra_eval,
        "image_level": image_level,
        "window_level": window_level,
    }


def build_long_metric_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_cols = ["model_id", "family", "backbone", "architecture_type"]
    for _, row in summary_df.iterrows():
        for col in summary_df.columns:
            if col in id_cols or col.startswith("source_") or col == "result_dir":
                continue
            if col.startswith("downstream_"):
                group = "downstream_full_pipeline"
                metric = col.replace("downstream_", "")
                origin = "original_global_downstream_csv"
            elif col.startswith("segmentation_"):
                group = "pixel_segmentation"
                metric = col.replace("segmentation_", "")
                origin = "original_pixel_metrics_global_summary_csv"
            elif col.startswith("binarization_A4_vs_A2_within_A1_"):
                group = "pixel_binarization"
                metric = col.replace("binarization_A4_vs_A2_within_A1_", "")
                origin = "original_pixel_metrics_global_summary_csv"
            elif col.startswith("segmentation_effect_"):
                group = "downstream_error_decomposition_segmentation_effect"
                metric = col.replace("segmentation_effect_", "")
                origin = "original_extra_evaluation_groups_csv"
            elif col.startswith("binarization_effect_"):
                group = "downstream_error_decomposition_binarization_effect"
                metric = col.replace("binarization_effect_", "")
                origin = "original_extra_evaluation_groups_csv"
            elif col.startswith("derived_"):
                group = "derived_for_plotting"
                metric = col.replace("derived_", "")
                origin = "derived_by_extract_crossmodel_downstream_metrics.py"
            else:
                continue
            rows.append({
                **{k: row[k] for k in id_cols},
                "metric_group": group,
                "metric": metric,
                "value": row[col],
                "origin": origin,
            })
    return pd.DataFrame(rows)


def build_metric_dictionary() -> pd.DataFrame:
    rows = [
        ("downstream_MAE", "Mean absolute error between Pred_D and GT_All_D over paired windows.", "Original from Pred_vs_GT_local_window_metrics_summary.csv; computed by paired_metrics()."),
        ("downstream_RMSE", "Root mean squared error between Pred_D and GT_All_D over paired windows.", "Original from Pred_vs_GT_local_window_metrics_summary.csv; computed by paired_metrics()."),
        ("downstream_Bias_y_minus_x", "Mean signed error Pred_D minus GT_All_D.", "Original from paired_metrics(); y is prediction, x is reference."),
        ("downstream_Pearson_r", "Pearson correlation between GT_All_D and Pred_D.", "Original from paired_metrics()."),
        ("downstream_R2", "1 - SSE/SST using GT_All_D as reference.", "Original from paired_metrics()."),
        ("downstream_Within_±0.10_ratio", "Fraction of paired windows with absolute error <= 0.10.", "Original from paired_metrics(); ERROR_BAND is 0.10 in pipeline."),
        ("downstream_AreaWeighted_MAE", "Area-weighted MAE using pair_weight_area as weights.", "Original from paired_metrics()."),
        ("segmentation_iou", "Strict IoU for predicted ROI A3 against GT ROI A1 over whole image.", "Original row eval_name=A3_vs_A1_segmentation, variant=strict."),
        ("segmentation_f1", "Strict F1/Dice for predicted ROI A3 against GT ROI A1.", "Original row eval_name=A3_vs_A1_segmentation, variant=strict."),
        ("binarization_A4_vs_A2_within_A1_iou", "Strict IoU for predicted white mask A4 against GT white mask A2 inside A1.", "Original row eval_name=A4_vs_A2_within_A1, variant=strict."),
        ("segmentation_effect_AreaWeighted_MAE", "Task-level D error caused by using predicted ROI A3 while keeping GT binary A2 fixed.", "Original extra group Segmentation_A2A3_vs_A2A1."),
        ("binarization_effect_AreaWeighted_MAE", "Task-level D error caused by using predicted binary A4 while keeping GT ROI A1 fixed.", "Original extra group Binarization_A4A1_vs_A2A1."),
        ("derived_abs_error_p95", "95th percentile of window-level absolute_error among pair_valid windows.", "Derived from original window_level_validation.csv without changing source values."),
        ("derived_blur_level_agreement", "Fraction of pair_valid windows where GT_blur_level equals Pred_blur_level.", "Derived from original window_level_validation.csv categorical labels."),
    ]
    return pd.DataFrame(rows, columns=["metric", "definition", "source_or_derivation"])


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)

    collected = []
    for spec in MODEL_SPECS:
        collected.append(collect_model(spec))

    summary_df = pd.DataFrame([item["summary"] for item in collected])
    summary_df["rank_by_downstream_AreaWeighted_MAE"] = summary_df["downstream_AreaWeighted_MAE"].rank(method="min", ascending=True).astype(int)
    summary_df["rank_by_segmentation_iou"] = summary_df["segmentation_iou"].rank(method="min", ascending=False).astype(int)
    summary_df["rank_by_binarization_iou"] = summary_df["binarization_A4_vs_A2_within_A1_iou"].rank(method="min", ascending=False).astype(int)
    summary_df = summary_df.sort_values("rank_by_downstream_AreaWeighted_MAE")

    global_downstream_df = pd.concat([item["global_downstream"] for item in collected], ignore_index=True)
    pixel_global_df = pd.concat([item["pixel_global"] for item in collected], ignore_index=True)
    extra_eval_df = pd.concat([item["extra_eval"] for item in collected], ignore_index=True)
    image_level_df = pd.concat([item["image_level"] for item in collected], ignore_index=True)

    window_keep_cols = [
        "model_id", "family", "backbone", "architecture_type", "source_csv",
        "parent", "image_name", "window_id", "x", "y", "w", "h",
        "GT_valid", "Pred_valid", "pair_valid", "pair_weight_area",
        "GT_All_area", "GT_All_white", "GT_All_Q", "GT_All_D", "GT_blur_level",
        "Pred_area", "Pred_white", "Pred_Q", "Pred_D", "Pred_blur_level",
        "signed_error_Pred_minus_GT", "absolute_error",
        "Seg_A2A3_D", "Seg_A2A3_signed_error_minus_GT", "Seg_A2A3_absolute_error",
        "Bin_A4A1_D", "Bin_A4A1_signed_error_minus_GT", "Bin_A4A1_absolute_error",
        "window_size", "stride", "A1_file", "A2_file", "A3_file", "A4_file",
    ]
    window_level_df = pd.concat([item["window_level"] for item in collected], ignore_index=True)
    existing_window_cols = [c for c in window_keep_cols if c in window_level_df.columns]
    window_plot_df = window_level_df[existing_window_cols].copy()
    window_plot_df = window_plot_df[to_bool_series(window_plot_df["pair_valid"])].copy()

    metric_long_df = build_long_metric_table(summary_df)
    metric_dictionary_df = build_metric_dictionary()

    source_inventory_rows = []
    for item in collected:
        spec_summary = item["summary"]
        for key in ["global_downstream", "pixel_global", "extra_eval", "image_level", "window_level"]:
            df = item[key]
            source_inventory_rows.append({
                "model_id": spec_summary["model_id"],
                "family": spec_summary["family"],
                "backbone": spec_summary["backbone"],
                "source_key": key,
                "source_csv": df["source_csv"].iloc[0],
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            })
    source_inventory_df = pd.DataFrame(source_inventory_rows)

    save(summary_df, SOURCE_DATA / "model_level_summary_for_table.csv")
    save(metric_long_df, SOURCE_DATA / "model_metric_long_for_plots.csv")
    save(global_downstream_df, SOURCE_DATA / "global_downstream_metrics_original_aligned.csv")
    save(pixel_global_df, SOURCE_DATA / "pixel_metrics_global_original_aligned.csv")
    save(extra_eval_df, SOURCE_DATA / "extra_evaluation_metrics_original_aligned.csv")
    save(image_level_df, SOURCE_DATA / "image_level_downstream_metrics_original_aligned.csv")
    save(window_plot_df, SOURCE_DATA / "window_level_pair_valid_for_scatter_and_distributions.csv")
    save(metric_dictionary_df, SOURCE_DATA / "metric_dictionary.csv")
    save(source_inventory_df, SOURCE_DATA / "source_file_inventory.csv")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "model_count": len(MODEL_SPECS),
        "models": [
            {
                "model_id": s["model_id"],
                "family": s["family"],
                "backbone": s["backbone"],
                "architecture_type": s["architecture_type"],
                "result_dir": str(s["result_dir"]),
            }
            for s in MODEL_SPECS
        ],
        "outputs": {
            "model_level_summary_for_table.csv": "One row per model; compact table and ranking source.",
            "model_metric_long_for_plots.csv": "Long-format model metrics for bar plots, ranking plots, and consistency plots.",
            "global_downstream_metrics_original_aligned.csv": "Original full-pipeline downstream metrics from each model directory.",
            "pixel_metrics_global_original_aligned.csv": "Original pixel-level segmentation/binarization metrics from each model directory.",
            "extra_evaluation_metrics_original_aligned.csv": "Original segmentation-effect and binarization-effect downstream metrics.",
            "image_level_downstream_metrics_original_aligned.csv": "Original image-level downstream metrics; 30 rows per model when complete.",
            "window_level_pair_valid_for_scatter_and_distributions.csv": "Pair-valid window-level D values and errors for scatter/distribution plots.",
            "metric_dictionary.csv": "Definitions and source notes for manuscript methods/table legends.",
            "source_file_inventory.csv": "Original source CSV paths and row counts.",
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved source data to: {SOURCE_DATA}")
    print(f"Models: {len(summary_df)}")
    print(f"Pair-valid windows: {len(window_plot_df)}")
    print(summary_df[[
        "model_id",
        "rank_by_downstream_AreaWeighted_MAE",
        "downstream_AreaWeighted_MAE",
        "downstream_MAE",
        "downstream_R2",
        "segmentation_iou",
        "binarization_A4_vs_A2_within_A1_iou",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
