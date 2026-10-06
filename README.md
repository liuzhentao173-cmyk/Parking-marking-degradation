<div align="center">

# Recover-then-Measure

**Hybrid segmentation and adaptive binarization for quantitative assessment of parking lot pavement marking degradation from UAV imagery**

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md)

[![Live demo](https://img.shields.io/badge/demo-live-4cc2ff)](https://liuzhentao173-cmyk.github.io/Parking-marking-degradation/)
[![Weights](https://img.shields.io/badge/weights-10.5281%2Fzenodo.20825454-1682d4)](https://doi.org/10.5281/zenodo.20825454)
[![Journal](https://img.shields.io/badge/journal-Measurement-f0a33a)](https://doi.org/10.1016/j.measurement.2026.123332)

<br>
<a href="https://liuzhentao173-cmyk.github.io/Parking-marking-degradation/"><img src="docs/assets/parking_flyover.gif" width="720" alt="Blender synthetic parking lot"></a>

</div>

Zhentao Liu, Jiaming Liu, Zhengtao Xie, Kai Xue, Shan Gu, Ji Dang

From a single UAV orthophoto of a parking lot, this code produces a **degradation index D** for every region, a **spatial heat map**, and a **maintenance grade** with its **measurement uncertainty**. It works in two steps. First it *recovers* the original footprint of every marking, including segments whose paint has worn away. Then it *measures* the paint that still remains inside that footprint.

> **[▶ Try the interactive demo](https://liuzhentao173-cmyk.github.io/Parking-marking-degradation/)**: hover over any pavement marking to see its original footprint *M*, its residual paint *R*, the index *D* and the maintenance grade.

<img src="docs/assets/RGB_M_R_preview.jpg" alt="RGB, M and R of the Blender scene">

The demo scene is a parking lot modelled and rendered in Blender. Its masks and values come from the scene geometry and paint material, so they are a synthetic reference, not predictions of the model. Every marking in the scene counts (bay lines, numbers, arrows, text, hatching, wheelchair symbols, crossing, stop/lane lines and the blue accessible-bay surfacing: 132 in all), and each value is computed over one complete marking. R is taken at pixel centres, without anti-aliasing.

## Pipeline

```
UAV orthophoto ─▶ 01 segmentation ─▶ M_Pred ─▶ 02 binarization ─▶ R_Pred ─▶ 03 index + heat map
                                                                             │
                     dual-mask reference (M_GT, R_GT) ─▶ 04 validation ─▶ 05 uncertainty + grading
```

| Script | Stage | What it does |
|---|---|---|
| `01_segment_marking_roi.py` | Segmentation | Tiled SegFormer inference, stitched into the predicted marking footprint `M_Pred` |
| `02_binarize_residual_paint.py` | Binarization | Weighted KDE–GMM adaptive threshold with shadow compensation and guarded fallback, giving the residual-paint mask `R_Pred` |
| `03_degradation_index_heatmap.py` | Quantification | Window-level degradation index `D` and spatial heat map |
| `04_downstream_gt_validation.py` | Validation | Decomposition against the ground truth (`D_GT` vs `D_Pred`) with cross-substituted realizations for each stage |
| `05_measurement_evaluation.py` | Measurement evaluation | Error attribution, expanded uncertainty at window, marking and image support, handbook distribution, three-grade maintenance scale and repainting decision |

The degradation index is the delamination ratio used in Japanese maintenance practice:

$$D = 1 - \frac{|R \cap M|}{|M|}$$

| Grade | Range of D | Maintenance action |
|---|---|---|
| I | D < 0.23 | Routine inspection |
| II | 0.23 ≤ D < 0.40 | Schedule repainting |
| III | D ≥ 0.40 | Repaint with priority |

## Repository structure

```
.
├── Coding/
│   ├── Operation/                              # pipeline, stages 01–05
│   │   ├── 01_segment_marking_roi.py
│   │   ├── 02_binarize_residual_paint.py
│   │   ├── 03_degradation_index_heatmap.py
│   │   ├── 04_downstream_gt_validation.py
│   │   ├── 05_measurement_evaluation.py
│   │   └── comparison/                         # analyses behind the comparison figures
│   │       ├── compare_binarization_methods.py       # proposed vs Otsu / WKDE-GMM / Fixed-L / Fixed-S
│   │       ├── compare_methods_scatter.py            # window/object scatter metrics for the above
│   │       ├── sensitivity_parameter_sweep.py        # one-at-a-time sweep of 8 binarization parameters
│   │       ├── sensitivity_compute_metrics.py        # mIoU / area-weighted MAE for the sweep
│   │       ├── run_all_models.py                     # runs 02/03/04 for every segmentation model
│   │       └── extract_crossmodel_downstream_metrics.py
│   └── Train/                                  # segmentation training notebooks (Colab, A100)
│       ├── Segformer-Backbone-Comparision.ipynb      # SegFormer, MiT backbones
│       ├── FPN-Comparison-A100.ipynb                 # FPN, ResNet-34/50/101
│       ├── UNet++-Comparison-A100.ipynb              # UNet++, ResNet-34/50/101
│       └── deeplabv3+.ipynb                          # DeepLabV3+, ResNet-34/50/101
└── docs/                                       # interactive demo (GitHub Pages)
```

The notebooks keep their training curves. Sample tiles from the dataset have been removed from their outputs because the imagery is not public.

## Mask notation

| Paper | Meaning | Tag in code |
|---|---|---|
| `M_GT` | ground-truth marking footprint | A1 |
| `R_GT` | ground-truth residual paint | A2 |
| `M_Pred` | predicted marking footprint | A3 |
| `R_Pred` | predicted residual paint | A4 |

Variable names, CSV columns (e.g. `GT_A1_D`) and `--help` texts use the short tags.

## Requirements

- Python 3.9+
- Stages 02–05 and `comparison/`:
  ```bash
  pip install numpy pandas opencv-python scipy matplotlib pillow tqdm
  ```
- Stage 01 also needs `pip install torch transformers` and a trained checkpoint (see below).

## Pretrained weights

The SegFormer checkpoints `MiTB0.pt` … `MiTB5.pt` are archived on Zenodo, [doi:10.5281/zenodo.20825454](https://doi.org/10.5281/zenodo.20825454). The pipeline uses **`MiTB5.pt`** (SegFormer MiT-B5, the model selected in the paper). Point `WEIGHT_PATH` in `01_segment_marking_roi.py`, or `--weight-path`, to it.

## Usage

Each script has a short configuration block at the top (paths marked `# TODO: set path`), and most also take command-line arguments (`--help`). Run from `Coding/Operation/`:

```bash
cd Coding/Operation
python 01_segment_marking_roi.py            # -> M_Pred
python 02_binarize_residual_paint.py        # -> R_Pred
python 03_degradation_index_heatmap.py      # -> D per window + heat map
python 04_downstream_gt_validation.py       # -> window_level_validation.csv

python 05_measurement_evaluation.py \
    --m-gt   <M_GT dir>  --r-gt   <R_GT dir> \
    --m-pred <M_Pred dir> --r-pred <R_Pred dir> \
    --window-csv     <04 output>/window_level_validation.csv \
    --stage2-objects <02 output>/all_output_info.csv \
    --out <output dir>
```

`05` writes `uncertainty.csv`, `grading_accuracy.csv`, `error_decomposition.csv`, `error_by_condition.csv`, `handbook_distribution.csv`, `band_resolution.csv`, per-window, per-marking and per-image tables, and a `summary.json`. If `--window-csv` is omitted, the window table is recomputed from the masks, so the same command also evaluates a set that has masks only.

Optional analyses:

```bash
python comparison/compare_binarization_methods.py && python comparison/compare_methods_scatter.py
python comparison/sensitivity_parameter_sweep.py  && python comparison/sensitivity_compute_metrics.py
python comparison/run_all_models.py               && python comparison/extract_crossmodel_downstream_metrics.py
```

## Correspondence to the paper

Section numbers follow the final manuscript.

| Code | Paper |
|---|---|
| `01_segment_marking_roi.py`, `Train/*.ipynb` | §2.3 semantic segmentation · §4.2 model selection |
| `02_binarize_residual_paint.py` | §2.4 adaptive binarization (context, shadow compensation, weighted KDE–GMM, guarded fallback) |
| `03_degradation_index_heatmap.py` | §2.5 degradation index and heat map · §5.1 assessment outputs |
| `04_downstream_gt_validation.py` | §2.6 ground-truth decomposition · §5.2.1–5.2.2 stage effects |
| `05_measurement_evaluation.py` | §2.6.1 error propagation · §5.2.3–5.2.4 attribution (Table 6) · §5.3 uncertainty · §5.4 grading (Tables 7–8) |
| `comparison/compare_binarization_methods.py`, `compare_methods_scatter.py` | §4.3.1–4.3.2 binarization comparison |
| `comparison/sensitivity_parameter_sweep.py`, `sensitivity_compute_metrics.py` | §4.3.3 parameter sensitivity |
| `comparison/run_all_models.py`, `extract_crossmodel_downstream_metrics.py` | §5.5.1 cross-model downstream validation |

## Data availability

The code is released to support reproducibility. The image data are not publicly available because they contain commercially confidential site information, and may only be shared with the permission of the data provider.

## Citation

If you use this code or the pretrained weights, please cite the paper ([doi.org/10.1016/j.measurement.2026.123332](https://doi.org/10.1016/j.measurement.2026.123332), available online 2 October 2026; in press, journal pre-proof):

Zhentao Liu, Jiaming Liu, Zhengtao Xie, Kai Xue, Shan Gu, Ji Dang. Recover-then-measure: Hybrid segmentation and adaptive binarization for quantitative assessment of parking lot pavement marking degradation from UAV imagery. *Measurement*, 2026, 123332.

```bibtex
@article{Liu2026RecoverThenMeasure,
  title   = {Recover-then-measure: Hybrid segmentation and adaptive binarization for quantitative assessment of parking lot pavement marking degradation from {UAV} imagery},
  author  = {Liu, Zhentao and Liu, Jiaming and Xie, Zhengtao and Xue, Kai and Gu, Shan and Dang, Ji},
  journal = {Measurement},
  year    = {2026},
  pages   = {123332},
  doi     = {10.1016/j.measurement.2026.123332},
  note    = {In press, journal pre-proof}
}
```
