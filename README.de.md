<div align="center">

# Recover-then-Measure: erst rekonstruieren, dann messen

**Hybride Segmentierung und adaptive Binarisierung zur quantitativen Bewertung des Verschleißes von Parkplatzmarkierungen aus UAV-Bildern**

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Deutsch** · [Français](README.fr.md)

[![Demo](https://img.shields.io/badge/demo-live-4cc2ff)](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=de)
[![Modellgewichte](https://img.shields.io/badge/weights-10.5281%2Fzenodo.20825454-1682d4)](https://doi.org/10.5281/zenodo.20825454)
[![Zeitschrift](https://img.shields.io/badge/journal-Measurement-f0a33a)](#zitierung)

<br>
<a href="https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/"><img src="docs/assets/parking_flyover.gif" width="720" alt="Blender synthetic parking lot"></a>

</div>

Zhentao Liu, Jiaming Liu, Zhengtao Xie, Kai Xue, Shan Gu, Ji Dang

Aus einem einzelnen UAV-Orthofoto eines Parkplatzes erzeugt dieser Code für jeden Bereich einen **Degradationsindex D**, eine **räumliche Wärmekarte** und eine **Instandhaltungsklasse** mit ihrer **Messunsicherheit**. Das geschieht in zwei Schritten: Zuerst wird die ursprüngliche Fläche jeder Markierung *rekonstruiert*, einschließlich der Abschnitte, deren Farbe abgefahren ist. Danach wird die Farbe *gemessen*, die innerhalb dieser Fläche noch vorhanden ist.

> **[▶ Interaktive Demo öffnen](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=de)**: Fahren Sie mit dem Zeiger über einen Stellplatz, um seine ursprüngliche Fläche *M*, die verbliebene Farbe *R*, den Index *D* und die Instandhaltungsklasse zu sehen.

<img src="docs/assets/RGB_M_R_preview.jpg" alt="RGB, M and R of the Blender scene">

Die Demo-Szene ist ein in Blender modellierter und gerenderter Parkplatz. Masken und Werte stammen aus Geometrie und Farbmaterial der Szene; sie sind synthetische Referenzwerte, keine Modellvorhersagen. Jeder Wert bezieht sich auf die vollständige Umrandung eines Stellplatzes.

## Ablauf

```
UAV-Orthofoto ─▶ 01 Segmentierung ─▶ M_Pred ─▶ 02 Binarisierung ─▶ R_Pred ─▶ 03 Index + Wärmekarte
                                                                               │
                 Doppelmasken-Referenz (M_GT, R_GT) ─▶ 04 Validierung ─▶ 05 Unsicherheit + Klassen
```

| Skript | Stufe | Funktion |
|---|---|---|
| `01_segment_marking_roi.py` | Segmentierung | Kachelweise SegFormer-Inferenz, zusammengesetzt zur vorhergesagten Markierungsfläche `M_Pred` |
| `02_binarize_residual_paint.py` | Binarisierung | Gewichteter KDE–GMM-Schwellenwert mit Schattenkompensation und abgesichertem Rückfallpfad; liefert die Maske der Restfarbe `R_Pred` |
| `03_degradation_index_heatmap.py` | Quantifizierung | Degradationsindex `D` je Fenster und räumliche Wärmekarte |
| `04_downstream_gt_validation.py` | Validierung | Zerlegung gegen die Referenz (`D_GT` vs. `D_Pred`) mit stufenweise ausgetauschten Masken |
| `05_measurement_evaluation.py` | Messbewertung | Fehlerzuordnung, erweiterte Unsicherheit je Fenster, Markierung und Bild, Verteilung über die Handbuch-Ränge, dreistufige Instandhaltungsklassen und Entscheidung zur Neumarkierung |

Der Degradationsindex ist der Ablösungsgrad, der in der japanischen Instandhaltungspraxis verwendet wird:

$$D = 1 - \frac{|R \cap M|}{|M|}$$

| Klasse | Bereich von D | Maßnahme |
|---|---|---|
| I | D < 0,23 | Routinekontrolle |
| II | 0,23 ≤ D < 0,40 | Neumarkierung einplanen |
| III | D ≥ 0,40 | Vorrangig neu markieren |

## Aufbau des Repositorys

```
.
├── Coding/
│   ├── Operation/                              # Pipeline, Stufen 01–05
│   │   ├── 01_segment_marking_roi.py
│   │   ├── 02_binarize_residual_paint.py
│   │   ├── 03_degradation_index_heatmap.py
│   │   ├── 04_downstream_gt_validation.py
│   │   ├── 05_measurement_evaluation.py
│   │   └── comparison/                         # Analysen hinter den Vergleichsabbildungen
│   │       ├── compare_binarization_methods.py       # Verfahren vs. Otsu / WKDE-GMM / Fixed-L / Fixed-S
│   │       ├── compare_methods_scatter.py            # Streudiagramm-Kennzahlen je Fenster/Objekt
│   │       ├── sensitivity_parameter_sweep.py        # Einzelvariation von 8 Binarisierungsparametern
│   │       ├── sensitivity_compute_metrics.py        # mIoU / flächengewichteter MAE dazu
│   │       ├── run_all_models.py                     # führt 02/03/04 für jedes Segmentierungsmodell aus
│   │       └── extract_crossmodel_downstream_metrics.py
│   └── Train/                                  # Trainings-Notebooks (Colab, A100)
│       ├── Segformer-Backbone-Comparision.ipynb      # SegFormer, MiT-Backbones
│       ├── FPN-Comparison-A100.ipynb                 # FPN, ResNet-34/50/101
│       ├── UNet++-Comparison-A100.ipynb              # UNet++, ResNet-34/50/101
│       └── deeplabv3+.ipynb                          # DeepLabV3+, ResNet-34/50/101
└── docs/                                       # interaktive Demo (GitHub Pages)
```

Die Notebooks enthalten weiterhin die Trainingskurven. Beispielkacheln aus dem Datensatz wurden aus den Ausgaben entfernt, da die Bilddaten nicht öffentlich sind.

## Maskennotation

| Artikel | Bedeutung | Kürzel im Code |
|---|---|---|
| `M_GT` | Referenzfläche der Markierung | A1 |
| `R_GT` | Referenz der Restfarbe | A2 |
| `M_Pred` | vorhergesagte Markierungsfläche | A3 |
| `R_Pred` | vorhergesagte Restfarbe | A4 |

Variablennamen, CSV-Spalten (z. B. `GT_A1_D`) und `--help`-Texte verwenden die Kürzel.

## Voraussetzungen

- Python 3.9+
- Stufen 02–05 und `comparison/`:
  ```bash
  pip install numpy pandas opencv-python scipy matplotlib pillow tqdm
  ```
- Stufe 01 benötigt zusätzlich `pip install torch transformers` und ein trainiertes Checkpoint (siehe unten).

## Vortrainierte Gewichte

Die SegFormer-Checkpoints `MiTB0.pt` … `MiTB5.pt` sind auf Zenodo archiviert: [doi:10.5281/zenodo.20825454](https://doi.org/10.5281/zenodo.20825454). Die Pipeline verwendet **`MiTB5.pt`** (SegFormer MiT-B5, das im Artikel ausgewählte Modell). Setzen Sie `WEIGHT_PATH` in `01_segment_marking_roi.py` (oder `--weight-path`) auf diese Datei.

## Verwendung

Jedes Skript hat oben einen kurzen Konfigurationsblock (Pfade sind mit `# TODO: set path` markiert); die meisten nehmen auch Kommandozeilenargumente an (`--help`). Ausführung in `Coding/Operation/`:

```bash
cd Coding/Operation
python 01_segment_marking_roi.py            # -> M_Pred
python 02_binarize_residual_paint.py        # -> R_Pred
python 03_degradation_index_heatmap.py      # -> D je Fenster + Wärmekarte
python 04_downstream_gt_validation.py       # -> window_level_validation.csv

python 05_measurement_evaluation.py \
    --m-gt   <M_GT-Ordner>  --r-gt   <R_GT-Ordner> \
    --m-pred <M_Pred-Ordner> --r-pred <R_Pred-Ordner> \
    --window-csv     <Ausgabe 04>/window_level_validation.csv \
    --stage2-objects <Ausgabe 02>/all_output_info.csv \
    --out <Ausgabeordner>
```

`05` schreibt `uncertainty.csv`, `grading_accuracy.csv`, `error_decomposition.csv`, `error_by_condition.csv`, `handbook_distribution.csv`, `band_resolution.csv`, Tabellen je Fenster, Markierung und Bild sowie eine `summary.json`. Ohne `--window-csv` wird die Fenstertabelle aus den Masken neu berechnet; derselbe Befehl bewertet also auch einen Datensatz, der nur aus Masken besteht.

Optionale Analysen:

```bash
python comparison/compare_binarization_methods.py && python comparison/compare_methods_scatter.py
python comparison/sensitivity_parameter_sweep.py  && python comparison/sensitivity_compute_metrics.py
python comparison/run_all_models.py               && python comparison/extract_crossmodel_downstream_metrics.py
```

## Zuordnung zum Artikel

Die Abschnittsnummern folgen dem finalen Manuskript.

| Code | Artikel |
|---|---|
| `01_segment_marking_roi.py`, `Train/*.ipynb` | §2.3 semantische Segmentierung · §4.2 Modellauswahl |
| `02_binarize_residual_paint.py` | §2.4 adaptive Binarisierung (Kontext, Schattenkompensation, gewichtetes KDE–GMM, abgesicherter Rückfallpfad) |
| `03_degradation_index_heatmap.py` | §2.5 Degradationsindex und Wärmekarte · §5.1 Bewertungsergebnisse |
| `04_downstream_gt_validation.py` | §2.6 Referenzzerlegung · §5.2.1–5.2.2 Einfluss der Stufen |
| `05_measurement_evaluation.py` | §2.6.1 Fehlerfortpflanzung · §5.2.3–5.2.4 Fehlerzuordnung (Tab. 6) · §5.3 Unsicherheit · §5.4 Klassifizierung (Tab. 7–8) |
| `comparison/compare_binarization_methods.py`, `compare_methods_scatter.py` | §4.3.1–4.3.2 Vergleich der Binarisierungsverfahren |
| `comparison/sensitivity_parameter_sweep.py`, `sensitivity_compute_metrics.py` | §4.3.3 Parametersensitivität |
| `comparison/run_all_models.py`, `extract_crossmodel_downstream_metrics.py` | §5.5.1 modellübergreifende Validierung |

## Datenverfügbarkeit

Der Code wird zur Reproduzierbarkeit veröffentlicht. Die Bilddaten sind nicht öffentlich, da sie vertrauliche Standortinformationen enthalten; sie können nur mit Zustimmung des Datenbereitstellers weitergegeben werden.

## Zitierung

Die Zitierangabe folgt nach der Veröffentlichung in *Measurement*. Bis dahin zitieren Sie bitte Titel und Autoren wie oben angegeben.
