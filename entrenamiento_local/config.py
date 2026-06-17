# ============================================================
# ExpoEscom - Entrenamiento LOCAL v3.0 (feature caching)
# Traducción del notebook 03_entrenamiento_v3.ipynb a scripts .py
# Pensado para RTX 2070 Super (8 GB VRAM) en Windows.
# ============================================================
import os
import torch

VERSION = "3.1-local"

# ─── RUTAS ───────────────────────────────────────────────────
# >>> EDITA DATASET_ROOT si tus imágenes están en otra carpeta <<<
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT  = r"D:\Descargas\ExpoEscom\dataset"  # una subcarpeta por clase
SAVE_DIR      = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME    = "cartoon_v3_local.pt"
SAVE_PATH     = os.path.join(SAVE_DIR, MODEL_NAME)
FEATURE_CACHE = os.path.join(SAVE_DIR, "features_cache.pt")

# ─── CLASE DE RUIDO ──────────────────────────────────────────
# La carpeta de "ruido/otra" puede llamarse de varias formas; la que
# coincida se trata como la clase de ruido (con su propio límite, MAX_OTRA).
NOISE_CLASS_NAMES = ("otra", "noise", "ruido")

# ─── LÍMITES DE DATOS ────────────────────────────────────────
MAX_PER_CLASS = 20_000      # imgs por clase de cartoon
MAX_OTRA      = 30_000      # imgs por subcarpeta de ruido (3 × 30k = 90k ≈ 35% del total)
VAL_PER_CLASS = 1_500       # validación capada por clase (F1 estable, sin overhead)

# ─── FASE 1 — cabeza sobre features cacheadas (baratísima) ───
NUM_EPOCHS_P1 = 20          # son vectores, no imágenes → vuela
LR_PHASE1     = 1e-3
HEAD_BATCH    = 4096        # batch grande: la cabeza es solo un MLP

# ─── FASE 2 — fine-tuning sobre subconjunto real ─────────────
FT_PER_CLASS  = 5_000       # imgs por clase para afinar (subconjunto balanceado)
NUM_EPOCHS_P2 = 2
LR_PHASE2     = 1e-5
UNFREEZE_N    = 3           # bloques del backbone a descongelar

# ─── HARDWARE (ajustado a 8 GB de la 2070 Super) ─────────────
BATCH_SIZE    = 64          # Fase 2 (con gradientes). Si da OOM, baja a 32.
BATCH_INFER   = 256         # extracción de features (sin gradientes → cabe más)
NUM_WORKERS   = 2           # 6 workers causa WinError 1455 (pagefile agotado con CUDA DLLs)
PREFETCH      = 4
SEED          = 42

# ─── NORMALIZACIÓN / IMAGEN ──────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224
IMG_EXTS      = (".jpg", ".jpeg", ".png", ".webp")

# ─── DISPOSITIVO ─────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


def is_noise(class_name):
    """True si la clase es de ruido. Soporta 'otra/ruido', 'otra/vida_real', etc."""
    return class_name.lower().split("/")[0] in NOISE_CLASS_NAMES
