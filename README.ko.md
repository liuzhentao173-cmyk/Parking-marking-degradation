<div align="center">

# Recover-then-Measure: 복원한 뒤 측정한다

**UAV 영상 기반 주차장 노면 표시 열화의 정량 평가를 위한 하이브리드 분할과 적응형 이진화**

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어** · [Deutsch](README.de.md) · [Français](README.fr.md)

[![데모](https://img.shields.io/badge/demo-라이브_데모-4cc2ff)](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=ko)
[![모델 가중치](https://img.shields.io/badge/weights-10.5281%2Fzenodo.20825454-1682d4)](https://doi.org/10.5281/zenodo.20825454)
[![저널](https://img.shields.io/badge/journal-Measurement-f0a33a)](#인용)

<br>
<a href="https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/"><img src="docs/assets/parking_flyover.gif" width="720" alt="Blender synthetic parking lot"></a>

</div>

Zhentao Liu, Jiaming Liu, Zhengtao Xie, Kai Xue, Shan Gu, Ji Dang

이 코드는 주차장을 촬영한 UAV 정사영상 한 장으로부터 영역별 **열화 지수 D**, **공간 열화 히트맵**, 그리고 **측정 불확도**를 포함한 **유지관리 등급**을 산출합니다. 처리는 두 단계로 이루어집니다. 먼저 각 표시의 원래 영역(도료가 마모되어 사라진 부분 포함)을 **복원**하고, 그다음 그 영역 안에 남아 있는 도료를 **측정**합니다.

> **[▶ 인터랙티브 데모 열기](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=ko)**: 주차 구획 위에 마우스를 올리면 원래 영역 *M*, 잔존 도료 *R*, 열화 지수 *D*, 유지관리 등급이 나타납니다.

<img src="docs/assets/RGB_M_R_preview.jpg" alt="RGB, M and R of the Blender scene">

데모 장면은 Blender로 모델링하고 렌더링한 주차장입니다. 마스크와 값은 장면의 형상과 도료 재질에서 얻은 합성 참고값이며 모델 예측이 아닙니다. 장면의 모든 표시(구획선·번호·화살표·문자·빗금·휠체어 표시·횡단보도·정지선/차선, 파란색 장애인 주차구역 도색, 총 132개)를 대상으로 하며, 값은 표시 하나의 전체 영역으로 계산합니다. R은 픽셀 중심에서 취하며 안티에일리어싱은 쓰지 않았습니다.

## 처리 흐름

```
UAV 정사영상 ─▶ 01 분할 ─▶ M_Pred ─▶ 02 이진화 ─▶ R_Pred ─▶ 03 지수 + 히트맵
                                                             │
                이중 마스크 정답 (M_GT, R_GT) ─▶ 04 검증 ─▶ 05 불확도 + 등급 판정
```

| 스크립트 | 단계 | 내용 |
|---|---|---|
| `01_segment_marking_roi.py` | 분할 | SegFormer 타일 추론 후 이어 붙여 예측 표시 영역 `M_Pred` 생성 |
| `02_binarize_residual_paint.py` | 이진화 | 그림자 보정과 보호형 대체 경로를 갖춘 가중 KDE–GMM 적응 임계값으로 잔존 도료 마스크 `R_Pred` 생성 |
| `03_degradation_index_heatmap.py` | 정량화 | 윈도 단위 열화 지수 `D`와 공간 히트맵 |
| `04_downstream_gt_validation.py` | 검증 | 정답 대비 분해 검증(`D_GT` 대 `D_Pred`). 단계별로 교차 치환한 결과도 생성 |
| `05_measurement_evaluation.py` | 측정 평가 | 오차 요인 분해, 윈도·표시·영상 세 단위의 확장 불확도, 핸드북 등급 분포, 3단계 유지관리 등급과 재도장 판정 |

열화 지수는 일본의 유지관리 실무에서 쓰는 박리율과 같습니다.

$$D = 1 - \frac{|R \cap M|}{|M|}$$

| 등급 | D 범위 | 유지관리 조치 |
|---|---|---|
| I | D < 0.23 | 정기 점검 |
| II | 0.23 ≤ D < 0.40 | 재도장 계획 |
| III | D ≥ 0.40 | 우선 재도장 |

## 저장소 구조

```
.
├── Coding/
│   ├── Operation/                              # 처리 파이프라인, 단계 01–05
│   │   ├── 01_segment_marking_roi.py
│   │   ├── 02_binarize_residual_paint.py
│   │   ├── 03_degradation_index_heatmap.py
│   │   ├── 04_downstream_gt_validation.py
│   │   ├── 05_measurement_evaluation.py
│   │   └── comparison/                         # 비교 그림·표의 바탕이 되는 분석
│   │       ├── compare_binarization_methods.py       # 제안 기법 vs Otsu / WKDE-GMM / Fixed-L / Fixed-S
│   │       ├── compare_methods_scatter.py            # 위 기법들의 윈도/객체 단위 산점도 지표
│   │       ├── sensitivity_parameter_sweep.py        # 이진화 파라미터 8개를 하나씩 변화
│   │       ├── sensitivity_compute_metrics.py        # 위 결과의 mIoU / 면적 가중 MAE
│   │       ├── run_all_models.py                     # 모든 분할 모델에 대해 02/03/04 실행
│   │       └── extract_crossmodel_downstream_metrics.py
│   └── Train/                                  # 분할 모델 학습 노트북 (Colab, A100)
│       ├── Segformer-Backbone-Comparision.ipynb      # SegFormer, MiT 백본
│       ├── FPN-Comparison-A100.ipynb                 # FPN, ResNet-34/50/101
│       ├── UNet++-Comparison-A100.ipynb              # UNet++, ResNet-34/50/101
│       └── deeplabv3+.ipynb                          # DeepLabV3+, ResNet-34/50/101
└── docs/                                       # 인터랙티브 데모 (GitHub Pages)
```

노트북에는 학습 곡선을 남겨 두었습니다. 데이터셋 영상은 공개하지 않으므로 출력에 있던 샘플 타일은 삭제했습니다.

## 마스크 표기

| 논문 | 의미 | 코드 내 표기 |
|---|---|---|
| `M_GT` | 정답 표시 영역 | A1 |
| `R_GT` | 정답 잔존 도료 | A2 |
| `M_Pred` | 예측 표시 영역 | A3 |
| `R_Pred` | 예측 잔존 도료 | A4 |

변수명, CSV 열 이름(예: `GT_A1_D`), `--help` 설명에는 짧은 표기를 사용합니다.

## 요구 사항

- Python 3.9 이상
- 단계 02–05와 `comparison/`:
  ```bash
  pip install numpy pandas opencv-python scipy matplotlib pillow tqdm
  ```
- 단계 01에는 추가로 `pip install torch transformers`와 학습된 가중치(아래 참조)가 필요합니다.

## 사전 학습 가중치

SegFormer 가중치 `MiTB0.pt` … `MiTB5.pt`는 Zenodo에 보관되어 있습니다: [doi:10.5281/zenodo.20825454](https://doi.org/10.5281/zenodo.20825454). 파이프라인은 **`MiTB5.pt`**(논문에서 최종 선정한 SegFormer MiT-B5)를 사용합니다. `01_segment_marking_roi.py`의 `WEIGHT_PATH`(또는 `--weight-path`)를 이 파일로 지정하세요.

## 사용 방법

각 스크립트 맨 위에 짧은 설정 블록이 있으며, 경로 부분에는 `# TODO: set path`라고 표시되어 있습니다. 대부분의 스크립트는 명령행 인수(`--help`)도 지원합니다. `Coding/Operation/`에서 실행합니다.

```bash
cd Coding/Operation
python 01_segment_marking_roi.py            # -> M_Pred
python 02_binarize_residual_paint.py        # -> R_Pred
python 03_degradation_index_heatmap.py      # -> 윈도별 D + 히트맵
python 04_downstream_gt_validation.py       # -> window_level_validation.csv

python 05_measurement_evaluation.py \
    --m-gt   <M_GT 폴더>  --r-gt   <R_GT 폴더> \
    --m-pred <M_Pred 폴더> --r-pred <R_Pred 폴더> \
    --window-csv     <04 출력>/window_level_validation.csv \
    --stage2-objects <02 출력>/all_output_info.csv \
    --out <출력 폴더>
```

`05`는 `uncertainty.csv`, `grading_accuracy.csv`, `error_decomposition.csv`, `error_by_condition.csv`, `handbook_distribution.csv`, `band_resolution.csv`, 윈도·표시·영상별 결과 표, 그리고 `summary.json`을 출력합니다. `--window-csv`를 생략하면 윈도 표를 마스크로부터 다시 계산하므로, 마스크만 있는 데이터셋도 같은 명령으로 평가할 수 있습니다.

추가 분석:

```bash
python comparison/compare_binarization_methods.py && python comparison/compare_methods_scatter.py
python comparison/sensitivity_parameter_sweep.py  && python comparison/sensitivity_compute_metrics.py
python comparison/run_all_models.py               && python comparison/extract_crossmodel_downstream_metrics.py
```

## 논문과의 대응

절 번호는 최종 원고를 따릅니다.

| 코드 | 논문 |
|---|---|
| `01_segment_marking_roi.py`, `Train/*.ipynb` | §2.3 의미론적 분할 · §4.2 모델 선정 |
| `02_binarize_residual_paint.py` | §2.4 적응형 이진화(문맥 구성, 그림자 보정, 가중 KDE–GMM, 보호형 대체 경로) |
| `03_degradation_index_heatmap.py` | §2.5 열화 지수와 히트맵 · §5.1 평가 결과 |
| `04_downstream_gt_validation.py` | §2.6 정답 분해 · §5.2.1–5.2.2 단계별 영향 |
| `05_measurement_evaluation.py` | §2.6.1 오차 전파 · §5.2.3–5.2.4 요인 분해(표 6) · §5.3 불확도 · §5.4 등급 판정(표 7–8) |
| `comparison/compare_binarization_methods.py`, `compare_methods_scatter.py` | §4.3.1–4.3.2 이진화 기법 비교 |
| `comparison/sensitivity_parameter_sweep.py`, `sensitivity_compute_metrics.py` | §4.3.3 파라미터 민감도 |
| `comparison/run_all_models.py`, `extract_crossmodel_downstream_metrics.py` | §5.5.1 모델 간 하위 단계 검증 |

## 데이터 가용성

코드는 재현성을 위해 공개합니다. 영상 데이터에는 상업적으로 기밀인 현장 정보가 포함되어 있어 공개하지 않으며, 데이터 제공자의 허가가 있는 경우에만 공유할 수 있습니다.

## 인용

논문이 *Measurement*에 게재되면 인용 정보를 추가하겠습니다. 그 전까지는 위의 제목과 저자를 인용해 주세요.
