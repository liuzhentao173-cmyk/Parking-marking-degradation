<div align="center">

# Recover-then-Measure：復元してから測る

**UAV 画像による駐車場路面標示の劣化定量評価のためのハイブリッドセグメンテーションと適応的二値化**

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md)

[![デモ](https://img.shields.io/badge/demo-ライブデモ-4cc2ff)](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=ja)
[![モデル重み](https://img.shields.io/badge/weights-10.5281%2Fzenodo.20825454-1682d4)](https://doi.org/10.5281/zenodo.20825454)
[![論文誌](https://img.shields.io/badge/journal-Measurement-f0a33a)](#引用)

<br>
<a href="https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/"><img src="docs/assets/parking_flyover.gif" width="720" alt="Blender synthetic parking lot"></a>

</div>

Zhentao Liu, Jiaming Liu, Zhengtao Xie, Kai Xue, Shan Gu, Ji Dang

本コードは、駐車場を撮影した UAV オルソ画像 1 枚から、領域ごとの**劣化指標 D**、**空間的な劣化ヒートマップ**、および**測定不確かさ**付きの**維持管理グレード**を出力します。処理は 2 段階です。まず各区画線の元の範囲（塗料が摩耗で消えた部分を含む）を**復元**し、次にその範囲内に残っている塗料を**測定**します。

> **[▶ インタラクティブデモを開く](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=ja)**：駐車枠にカーソルを合わせると、元の範囲 *M*、残存塗料 *R*、劣化指標 *D*、維持管理グレードが表示されます。

<img src="docs/assets/RGB_M_R_preview.jpg" alt="RGB, M and R of the Blender scene">

デモのシーンは Blender でモデリング・レンダリングした駐車場です。マスクと値はシーンの形状と塗料マテリアルから得た合成上の参照値であり、モデルの予測ではありません。シーン内のすべての標示（枠線・番号・矢印・文字・ゼブラ・車いすマーク・横断歩道・停止線／区画線、青色の車いす用区画の着色舗装、計 132 個）を対象とし、値は標示 1 つ分の全体で計算します。R は画素中心で取得し、アンチエイリアスは使っていません。

## 処理の流れ

```
UAV オルソ画像 ─▶ 01 セグメンテーション ─▶ M_Pred ─▶ 02 二値化 ─▶ R_Pred ─▶ 03 指標 + ヒートマップ
                                                                              │
                   二重マスク正解 (M_GT, R_GT) ─▶ 04 検証 ─▶ 05 不確かさ + グレード判定
```

| スクリプト | 段階 | 内容 |
|---|---|---|
| `01_segment_marking_roi.py` | セグメンテーション | SegFormer のタイル推論と結合により、予測区画線範囲 `M_Pred` を得る |
| `02_binarize_residual_paint.py` | 二値化 | 影補正とガード付きフォールバックを備えた加重 KDE–GMM 適応しきい値により、残存塗料マスク `R_Pred` を得る |
| `03_degradation_index_heatmap.py` | 定量化 | ウィンドウ単位の劣化指標 `D` と空間ヒートマップ |
| `04_downstream_gt_validation.py` | 検証 | 正解との分解検証（`D_GT` 対 `D_Pred`）。段階ごとに入れ替えた実現値も作成 |
| `05_measurement_evaluation.py` | 測定評価 | 誤差の要因分解、ウィンドウ・区画線・画像の 3 単位での拡張不確かさ、ハンドブック評価の分布、3 段階の維持管理グレードと再塗装判定 |

劣化指標は、日本の維持管理で用いられる剥離率です。

$$D = 1 - \frac{|R \cap M|}{|M|}$$

| グレード | D の範囲 | 維持管理の対応 |
|---|---|---|
| I | D < 0.23 | 通常点検 |
| II | 0.23 ≤ D < 0.40 | 再塗装を計画 |
| III | D ≥ 0.40 | 優先して再塗装 |

## リポジトリ構成

```
.
├── Coding/
│   ├── Operation/                              # 処理パイプライン（段階 01–05）
│   │   ├── 01_segment_marking_roi.py
│   │   ├── 02_binarize_residual_paint.py
│   │   ├── 03_degradation_index_heatmap.py
│   │   ├── 04_downstream_gt_validation.py
│   │   ├── 05_measurement_evaluation.py
│   │   └── comparison/                         # 比較図表の元になる解析
│   │       ├── compare_binarization_methods.py       # 提案法 vs Otsu / WKDE-GMM / Fixed-L / Fixed-S
│   │       ├── compare_methods_scatter.py            # 上記のウィンドウ／物体単位の散布図指標
│   │       ├── sensitivity_parameter_sweep.py        # 二値化パラメータ 8 個を 1 つずつ変化
│   │       ├── sensitivity_compute_metrics.py        # 上記の mIoU／面積加重 MAE
│   │       ├── run_all_models.py                     # 全セグメンテーションモデルで 02/03/04 を実行
│   │       └── extract_crossmodel_downstream_metrics.py
│   └── Train/                                  # セグメンテーション学習ノートブック（Colab, A100）
│       ├── Segformer-Backbone-Comparision.ipynb      # SegFormer（MiT バックボーン）
│       ├── FPN-Comparison-A100.ipynb                 # FPN、ResNet-34/50/101
│       ├── UNet++-Comparison-A100.ipynb              # UNet++、ResNet-34/50/101
│       └── deeplabv3+.ipynb                          # DeepLabV3+、ResNet-34/50/101
└── docs/                                       # インタラクティブデモ（GitHub Pages）
```

ノートブックには学習曲線を残しています。データセットの画像は公開していないため、出力に含まれていたサンプルタイルは削除しました。

## マスクの表記

| 論文 | 意味 | コード内の記号 |
|---|---|---|
| `M_GT` | 正解の区画線範囲 | A1 |
| `R_GT` | 正解の残存塗料 | A2 |
| `M_Pred` | 予測された区画線範囲 | A3 |
| `R_Pred` | 予測された残存塗料 | A4 |

変数名、CSV の列名（例：`GT_A1_D`）、`--help` の説明には短い記号を使っています。

## 動作環境

- Python 3.9 以上
- 段階 02–05 と `comparison/`：
  ```bash
  pip install numpy pandas opencv-python scipy matplotlib pillow tqdm
  ```
- 段階 01 にはさらに `pip install torch transformers` と学習済み重み（下記）が必要です。

## 学習済み重み

SegFormer の重み `MiTB0.pt` … `MiTB5.pt` は Zenodo に保存しています：[doi:10.5281/zenodo.20825454](https://doi.org/10.5281/zenodo.20825454)。パイプラインでは **`MiTB5.pt`**（論文で選定した SegFormer MiT-B5）を使います。`01_segment_marking_roi.py` の `WEIGHT_PATH`（または `--weight-path`）にこのファイルを指定してください。

## 使い方

各スクリプトの冒頭に短い設定ブロックがあり、パスの箇所には `# TODO: set path` と記しています。多くのスクリプトはコマンドライン引数（`--help`）にも対応しています。`Coding/Operation/` で実行します。

```bash
cd Coding/Operation
python 01_segment_marking_roi.py            # -> M_Pred
python 02_binarize_residual_paint.py        # -> R_Pred
python 03_degradation_index_heatmap.py      # -> ウィンドウごとの D + ヒートマップ
python 04_downstream_gt_validation.py       # -> window_level_validation.csv

python 05_measurement_evaluation.py \
    --m-gt   <M_GT フォルダ>  --r-gt   <R_GT フォルダ> \
    --m-pred <M_Pred フォルダ> --r-pred <R_Pred フォルダ> \
    --window-csv     <04 の出力>/window_level_validation.csv \
    --stage2-objects <02 の出力>/all_output_info.csv \
    --out <出力フォルダ>
```

`05` は `uncertainty.csv`、`grading_accuracy.csv`、`error_decomposition.csv`、`error_by_condition.csv`、`handbook_distribution.csv`、`band_resolution.csv`、ウィンドウ・区画線・画像ごとの表、および `summary.json` を出力します。`--window-csv` を省略するとウィンドウ表をマスクから再計算するため、マスクしかないデータセットも同じコマンドで評価できます。

追加の解析：

```bash
python comparison/compare_binarization_methods.py && python comparison/compare_methods_scatter.py
python comparison/sensitivity_parameter_sweep.py  && python comparison/sensitivity_compute_metrics.py
python comparison/run_all_models.py               && python comparison/extract_crossmodel_downstream_metrics.py
```

## 論文との対応

節番号は最終稿に従います。

| コード | 論文 |
|---|---|
| `01_segment_marking_roi.py`、`Train/*.ipynb` | §2.3 セマンティックセグメンテーション · §4.2 モデル選定 |
| `02_binarize_residual_paint.py` | §2.4 適応的二値化（コンテキスト構築、影補正、加重 KDE–GMM、ガード付きフォールバック） |
| `03_degradation_index_heatmap.py` | §2.5 劣化指標とヒートマップ · §5.1 評価結果 |
| `04_downstream_gt_validation.py` | §2.6 正解分解 · §5.2.1–5.2.2 各段階の影響 |
| `05_measurement_evaluation.py` | §2.6.1 誤差伝播 · §5.2.3–5.2.4 要因分解（表 6）· §5.3 不確かさ · §5.4 グレード判定（表 7–8） |
| `comparison/compare_binarization_methods.py`、`compare_methods_scatter.py` | §4.3.1–4.3.2 二値化手法の比較 |
| `comparison/sensitivity_parameter_sweep.py`、`sensitivity_compute_metrics.py` | §4.3.3 パラメータ感度 |
| `comparison/run_all_models.py`、`extract_crossmodel_downstream_metrics.py` | §5.5.1 モデル間の下流検証 |

## データの利用可能性

コードは再現性のために公開しています。画像データには商業上秘密の現場情報が含まれるため公開しておらず、データ提供者の許可がある場合に限り共有できます。

## 引用

論文が *Measurement* に掲載され次第、引用情報を追加します。それまでは上記の題名と著者を引用してください。
