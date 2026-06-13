---
name: expo_escom_overview
description: ExpoEscom — multi-label cartoon/anime image classifier project; local + Colab training
metadata:
  type: project
---

**ExpoEscom** — cartoon/anime multi-label image classifier for expo/science fair.

**Model:** PyTorch MobileNetV2 (224×224, BCEWithLogitsLoss for multi-label). Two-phase training: Phase 1 frozen backbone, Phase 2 fine-tune last N blocks.

**Key files (local, C:\Users\Samsung\Desktop\proyectos\expo-escom):**
- `config.py` — hyperparameters, detect_classes(), paths
- `model.py` — CartoonClassifier, train() override keeps frozen BN in eval mode
- `train.py` — two-phase run, early stopping, label smoothing, threshold tuning
- `dataset_utils.py` — stratified split, transforms (RandomResizedCrop, GaussianBlur, RandomErasing for paper/expo)
- `predict.py` — inference with per-class thresholds
- `verificar_gpu.py` — validate normalization + GPU
- `00_conteo_dataset.ipynb` (Colab) — v4.0.0: dataset utilities menu

**Colab (ToonVerse_Training.ipynb):** A100, 2.6M images, all local improvements ported. Also: `01_estandarizar_pokemon_fusion_1.ipynb` (v3.1.0) and `09_augmentacion_100k_3.ipynb` for fill-to-100k.

**Dataset:** Currently pokemon + yugioh in shared Drive folder (FOLDER_ID). All `.jpg` only.
