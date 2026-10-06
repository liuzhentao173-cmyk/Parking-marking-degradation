<div align="center">

# Recover-then-Measure : reconstituer, puis mesurer

**Segmentation hybride et binarisation adaptative pour l’évaluation quantitative de la dégradation des marquages de parking à partir d’images de drone**

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · **Français**

[![Démo](https://img.shields.io/badge/démo-en_ligne-4cc2ff)](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=fr)
[![Poids du modèle](https://img.shields.io/badge/weights-10.5281%2Fzenodo.20825454-1682d4)](https://doi.org/10.5281/zenodo.20825454)
[![Revue](https://img.shields.io/badge/journal-Measurement-f0a33a)](#citation)

<br>
<a href="https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/"><img src="docs/assets/parking_flyover.gif" width="720" alt="Blender synthetic parking lot"></a>

</div>

Zhentao Liu, Jiaming Liu, Zhengtao Xie, Kai Xue, Shan Gu, Ji Dang

À partir d’une seule orthophoto de parking prise par drone, ce code produit pour chaque zone un **indice de dégradation D**, une **carte thermique spatiale** et une **classe d’entretien** accompagnée de son **incertitude de mesure**. Il procède en deux étapes : il *reconstitue* d’abord l’emprise d’origine de chaque marquage, y compris les parties dont la peinture a disparu, puis il *mesure* la peinture qui subsiste à l’intérieur de cette emprise.

> **[▶ Ouvrir la démo interactive](https://liuzhentao173-cmyk.github.io/Quantitative-assessment-of-parking-lot-pavement-marking-degradation/?lang=fr)** : survolez un marquage pour afficher son emprise d’origine *M*, sa peinture résiduelle *R*, l’indice *D* et la classe d’entretien.

<img src="docs/assets/RGB_M_R_preview.jpg" alt="RGB, M and R of the Blender scene">

La scène de démonstration est un parking modélisé et rendu dans Blender. Les masques et les valeurs proviennent de la géométrie et du matériau de peinture de la scène : ce sont des références de synthèse, pas des prédictions du modèle. Tous les marquages de la scène comptent (lignes de place, numéros, flèches, inscriptions, hachures, symboles, passage piéton, lignes d’arrêt et de voie, et la surface bleue de la place PMR, soit 132 au total) ; chaque valeur porte sur un marquage complet. R est relevé au centre des pixels, sans anticrénelage.

## Chaîne de traitement

```
Orthophoto drone ─▶ 01 segmentation ─▶ M_Pred ─▶ 02 binarisation ─▶ R_Pred ─▶ 03 indice + carte thermique
                                                                               │
                    référence double masque (M_GT, R_GT) ─▶ 04 validation ─▶ 05 incertitude + classes
```

| Script | Étape | Rôle |
|---|---|---|
| `01_segment_marking_roi.py` | Segmentation | Inférence SegFormer par tuiles, assemblée en emprise prédite `M_Pred` |
| `02_binarize_residual_paint.py` | Binarisation | Seuil adaptatif KDE–GMM pondéré, avec compensation des ombres et repli sécurisé ; donne le masque de peinture résiduelle `R_Pred` |
| `03_degradation_index_heatmap.py` | Quantification | Indice `D` par fenêtre et carte thermique spatiale |
| `04_downstream_gt_validation.py` | Validation | Décomposition par rapport à la référence (`D_GT` vs `D_Pred`), avec des réalisations croisées pour chaque étape |
| `05_measurement_evaluation.py` | Évaluation métrologique | Attribution des erreurs, incertitude élargie par fenêtre, par marquage et par image, distribution selon les rangs du manuel, échelle d’entretien à trois classes et décision de remarquage |

L’indice de dégradation est le taux de délaminage utilisé dans la pratique d’entretien japonaise :

$$D = 1 - \frac{|R \cap M|}{|M|}$$

| Classe | Plage de D | Action d’entretien |
|---|---|---|
| I | D < 0,23 | Inspection courante |
| II | 0,23 ≤ D < 0,40 | Prévoir un remarquage |
| III | D ≥ 0,40 | Remarquer en priorité |

## Structure du dépôt

```
.
├── Coding/
│   ├── Operation/                              # chaîne de traitement, étapes 01–05
│   │   ├── 01_segment_marking_roi.py
│   │   ├── 02_binarize_residual_paint.py
│   │   ├── 03_degradation_index_heatmap.py
│   │   ├── 04_downstream_gt_validation.py
│   │   ├── 05_measurement_evaluation.py
│   │   └── comparison/                         # analyses derrière les figures comparatives
│   │       ├── compare_binarization_methods.py       # méthode proposée vs Otsu / WKDE-GMM / Fixed-L / Fixed-S
│   │       ├── compare_methods_scatter.py            # indicateurs de dispersion par fenêtre/objet
│   │       ├── sensitivity_parameter_sweep.py        # variation un à un de 8 paramètres de binarisation
│   │       ├── sensitivity_compute_metrics.py        # mIoU / MAE pondérée par la surface
│   │       ├── run_all_models.py                     # exécute 02/03/04 pour chaque modèle de segmentation
│   │       └── extract_crossmodel_downstream_metrics.py
│   └── Train/                                  # notebooks d’entraînement (Colab, A100)
│       ├── Segformer-Backbone-Comparision.ipynb      # SegFormer, encodeurs MiT
│       ├── FPN-Comparison-A100.ipynb                 # FPN, ResNet-34/50/101
│       ├── UNet++-Comparison-A100.ipynb              # UNet++, ResNet-34/50/101
│       └── deeplabv3+.ipynb                          # DeepLabV3+, ResNet-34/50/101
└── docs/                                       # démo interactive (GitHub Pages)
```

Les notebooks conservent leurs courbes d’entraînement. Les tuiles d’exemple issues du jeu de données ont été retirées des sorties, car les images ne sont pas publiques.

## Notation des masques

| Article | Signification | Repère dans le code |
|---|---|---|
| `M_GT` | emprise de référence du marquage | A1 |
| `R_GT` | peinture résiduelle de référence | A2 |
| `M_Pred` | emprise prédite | A3 |
| `R_Pred` | peinture résiduelle prédite | A4 |

Les noms de variables, les colonnes CSV (par ex. `GT_A1_D`) et les textes `--help` utilisent ces repères.

## Prérequis

- Python 3.9+
- Étapes 02–05 et `comparison/` :
  ```bash
  pip install numpy pandas opencv-python scipy matplotlib pillow tqdm
  ```
- L’étape 01 nécessite en plus `pip install torch transformers` et un modèle entraîné (voir ci-dessous).

## Poids pré-entraînés

Les modèles SegFormer `MiTB0.pt` … `MiTB5.pt` sont archivés sur Zenodo : [doi:10.5281/zenodo.20825454](https://doi.org/10.5281/zenodo.20825454). La chaîne utilise **`MiTB5.pt`** (SegFormer MiT-B5, le modèle retenu dans l’article). Indiquez ce fichier dans `WEIGHT_PATH` de `01_segment_marking_roi.py` (ou via `--weight-path`).

## Utilisation

Chaque script comporte en tête un court bloc de configuration (chemins signalés par `# TODO: set path`) ; la plupart acceptent aussi des arguments en ligne de commande (`--help`). À lancer depuis `Coding/Operation/` :

```bash
cd Coding/Operation
python 01_segment_marking_roi.py            # -> M_Pred
python 02_binarize_residual_paint.py        # -> R_Pred
python 03_degradation_index_heatmap.py      # -> D par fenêtre + carte thermique
python 04_downstream_gt_validation.py       # -> window_level_validation.csv

python 05_measurement_evaluation.py \
    --m-gt   <dossier M_GT>  --r-gt   <dossier R_GT> \
    --m-pred <dossier M_Pred> --r-pred <dossier R_Pred> \
    --window-csv     <sortie 04>/window_level_validation.csv \
    --stage2-objects <sortie 02>/all_output_info.csv \
    --out <dossier de sortie>
```

`05` produit `uncertainty.csv`, `grading_accuracy.csv`, `error_decomposition.csv`, `error_by_condition.csv`, `handbook_distribution.csv`, `band_resolution.csv`, des tables par fenêtre, par marquage et par image, ainsi qu’un `summary.json`. Sans `--window-csv`, la table des fenêtres est recalculée à partir des masques : la même commande évalue donc aussi un jeu qui ne comporte que des masques.

Analyses complémentaires :

```bash
python comparison/compare_binarization_methods.py && python comparison/compare_methods_scatter.py
python comparison/sensitivity_parameter_sweep.py  && python comparison/sensitivity_compute_metrics.py
python comparison/run_all_models.py               && python comparison/extract_crossmodel_downstream_metrics.py
```

## Correspondance avec l’article

Les numéros de section suivent le manuscrit final.

| Code | Article |
|---|---|
| `01_segment_marking_roi.py`, `Train/*.ipynb` | §2.3 segmentation sémantique · §4.2 choix du modèle |
| `02_binarize_residual_paint.py` | §2.4 binarisation adaptative (contexte, compensation des ombres, KDE–GMM pondéré, repli sécurisé) |
| `03_degradation_index_heatmap.py` | §2.5 indice de dégradation et carte thermique · §5.1 résultats d’évaluation |
| `04_downstream_gt_validation.py` | §2.6 décomposition par rapport à la référence · §5.2.1–5.2.2 effets de chaque étape |
| `05_measurement_evaluation.py` | §2.6.1 propagation des erreurs · §5.2.3–5.2.4 attribution (tableau 6) · §5.3 incertitude · §5.4 classement (tableaux 7–8) |
| `comparison/compare_binarization_methods.py`, `compare_methods_scatter.py` | §4.3.1–4.3.2 comparaison des méthodes de binarisation |
| `comparison/sensitivity_parameter_sweep.py`, `sensitivity_compute_metrics.py` | §4.3.3 sensibilité aux paramètres |
| `comparison/run_all_models.py`, `extract_crossmodel_downstream_metrics.py` | §5.5.1 validation inter-modèles |

## Disponibilité des données

Le code est publié pour permettre la reproductibilité. Les images ne sont pas publiques, car elles contiennent des informations confidentielles sur les sites ; elles ne peuvent être partagées qu’avec l’autorisation du fournisseur des données.

## Citation

La référence sera ajoutée après la publication de l’article dans *Measurement*. D’ici là, merci de citer le titre et les auteurs ci-dessus.
