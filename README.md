[README.md](https://github.com/user-attachments/files/35079854/README.md)
# Left Ventricular Localisation and EF Estimation (EchoNet-Dynamic)

Segmentation-guided estimation of **left-ventricular ejection fraction (LVEF)** from apical four-chamber echocardiogram videos. The project builds the full chain from raw videos to an EF value:
LV segmentation → ED/ES phase selection → contour and mask-feature extraction → EF models (geometric, classical ML, video deep learning, multimodal fusion).

## Problem

LVEF is the key measure of cardiac function, but manual tracing is slow and operator dependent. This project explores whether combining **video**, **predicted LV contours** and **mask-derived morphology features**, with **adaptive end-diastole / end-systole (ED/ES) selection**, gives a more reliable automatic EF estimate than video alone. It also studies the strong label imbalance of EF (few severely reduced or hyperdynamic hearts) using LDS/FDS-style imbalanced-regression techniques.

## Dataset

[EchoNet-Dynamic](https://echonet.github.io/dynamic/) (Ouyang et al., *Nature* 2020): apical four-chamber echo videos (112×112) with `FileList.csv` (EF, split) and `VolumeTracings.csv` (LV traces at ED/ES).
**The data is not included** and requires accepting the dataset's own terms. Place it under one root folder:

```
EchoNet_LV/                 # any name; point ECHONET_ROOT to it
├── data/
│   ├── FileList.csv
│   ├── VolumeTracings.csv
│   └── videos_split/{train,val,test}/*.avi
└── outputs/                # created by the scripts
```

```bash
export ECHONET_ROOT=/path/to/EchoNet_LV      # Windows PowerShell: $env:ECHONET_ROOT="D:\EchoNet_LV"
```
Scripts in `01`–`05` default to `./EchoNet_LV`; scripts in `06` default to the Colab path `/content/drive/MyDrive/EchoNet-Dynamic`.

## Methodology and workflow

| Stage | Folder | What it does |
|---|---|---|
| 1 | `01_data_preparation/` | Frame extraction (112×112, ED/ES labelled frames), GT mask rasterisation, GT test masks, metadata tables |
| 2 | `02_segmentation/` | U-Net → Attention U-Net → edge-aware boundary fine-tuning (TensorFlow/Keras); evaluation with Dice, IoU and true HD95; mask inference for test/val |
| 3 | `03_contours_and_mask_features/` | Contour extraction from predicted masks (baseline, apex-based, anatomical open-wall), resampling to 100 points, GT-vs-prediction contour metrics (24-dimensional fusion input) |
| 4 | `04_adaptive_phase_selection/` | Framewise LV area (Savitzky–Golay smoothed), its derivative, Farneback optical-flow motion → stability / ED-ES scores → predicted ED/ES frames; coverage analysis of uniform 32-frame sampling; `phase_eval.py` (average absolute frame difference) |
| 5 | `05_ef_estimation/` | Geometric hybrid EF (mask area + contour length), XGBoost baseline, DenseNet121-BiLSTM video baseline, **R(2+1)D multimodal model** (video + contours + mask features), forward-pass demo |
| 6 | `06_anchor_fusion_imbalanced_regression/` | ED/ES-anchored DenseNet-BiLSTM with CBAM, LDS-weighted loss, FDS, cached-feature temporal-head experiments (TCN, Transformer, TAM, ConvLSTM, R3D-18) and an end-to-end DenseNet+TCN run |
| 7 | `07_analysis_and_evaluation/` | EF scatter/residual/error-by-range plots, feature-distribution checks, bootstrap CIs / Bland–Altman / AUC / Wilcoxon (`stats_eval.py`) |

**Models and technologies:** Python, TensorFlow/Keras (U-Net, Attention U-Net), PyTorch (DenseNet121, R(2+1)D-18, R3D-18, BiLSTM, TCN, Transformer, ConvLSTM, CBAM), OpenCV, NumPy/pandas/SciPy, scikit-learn, XGBoost.

## Repository structure

```
01_data_preparation/                 data and mask preparation
02_segmentation/{training,evaluation,inference,experiments}/
03_contours_and_mask_features/{contour_extraction,contour_resampling,contour_evaluation,mask_features}/
04_adaptive_phase_selection/
05_ef_estimation/{geometric_ef,classical_ml,video_models}/
06_anchor_fusion_imbalanced_regression/   (see its README; experiments in cached_feature_experiments/)
07_analysis_and_evaluation/
docs/figures/                        training curves of the two segmentation models
requirements.txt
```

## Installation

Developed locally with Python 3.11 and on Google Colab (GPU) for the PyTorch experiments.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Versions are not pinned because the original environments were not recorded.

## Running the main pipeline

Run scripts from inside their own folder (several import siblings by module name). Each script's header lists its inputs/outputs.

```bash
# 1  data
python 01_data_preparation/frame_extraction.py
python 01_data_preparation/mask_generation.py
# 2  segmentation (weights are written to the current directory)
python 02_segmentation/training/train_unet.py
python 02_segmentation/training/train_attention_unet.py
python 02_segmentation/training/finetune_attention_unet_edge_aware.py
python 02_segmentation/inference/generate_predicted_masks.py test      # or: val | val_full
# 3-4 phase selection, contours, features
python 04_adaptive_phase_selection/smoothed_framewise_lv_area.py
python 04_adaptive_phase_selection/derivative_framewise_analysis.py
python 04_adaptive_phase_selection/opticalflow_framewise_analysis.py
python 04_adaptive_phase_selection/multi_feature_cardiac_analysis.py
python 04_adaptive_phase_selection/export_predicted_frames.py
# 5  multimodal EF model
python 05_ef_estimation/video_models/train_r21d_multimodal.py
python 05_ef_estimation/video_models/test_r21d_multimodal.py
# quick check without data or weights (synthetic tensors)
python 05_ef_estimation/video_models/demo_forward_pass.py
```
Contour extraction/resampling, mask features and metadata tables (`03`, `01`) must run between steps 4 and 5; the exact order is given by each script's input files.

## Evaluation and results

Metrics implemented: Dice, IoU, true HD95, contour distance metrics (segmentation/contours); MAE, RMSE, R², Pearson r, bias, MAE by EF range and clinical EF buckets, bootstrap CIs, Bland–Altman, AUC, Wilcoxon (EF).



<!-- TODO (author): add the final test-set EF results from your own runs, e.g.
| Model | MAE | RMSE | R² | r |
No result tables are included because no prediction files were part of the material this repository was built from. -->

## Known limitations

- **ED/ES hand-off:** several scripts (`extract_predicted_contours_*`, `compute_hybrid_ef.py`, `generate_test_mask_features.py`, `evaluate_ed_es_masks_gt_vs_pred.py`) read `outputs/adaptive_phase_selection/<video>/{ED,ES}/`. The script that creates this folder is not in the repository. `export_predicted_frames.py` writes `predicted_frames.csv`, and `extract_predicted_contours_from_frame_csv.py` consumes it instead (its outputs carry a `_70_30` suffix that later scripts do not expect).
- **Edge-aware fine-tuning:** `finetune_attention_unet_edge_aware.py` continues from `attention_unet_edge_finetuned.keras`; the first fine-tuning round that creates that file is not included.
- **R(2+1)D paths:** training writes to `outputs/r21d_multimodal_adaptive`, testing and the analysis plots read `outputs/r21d_multimodal`; adjust one side.
- **R(2+1)D sampling:** `train_r21d_multimodal.py` builds `FusionDataset` without a framewise CSV, so frames are sampled uniformly; the adaptive window path only runs when a framewise CSV is supplied.
- Phase-selection weights (area vs. stability) are constants in `multi_feature_cardiac_analysis.py`.
- Trained weights, data and generated outputs are intentionally not versioned.
