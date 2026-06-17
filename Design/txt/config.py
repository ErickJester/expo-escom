# ============================================================
# ExpoEscom - Configuración central (versión LOCAL)
# Antes corría en Colab T4. Ahora corre local en RTX 2070 Super.
# ============================================================
import os
import torch

# ─── Rutas (relativas a la carpeta del proyecto) ─────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
SAVE_DIR     = os.path.join(PROJECT_ROOT, 'models')
MODEL_NAME   = 'mini_poc.pt'
SAVE_PATH    = os.path.join(SAVE_DIR, MODEL_NAME)

# ─── Dispositivo ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Hiperparámetros ─────────────────────────────────────────
# "otra" tiene subcarpetas (cartoons_anime, noise, real_life) →
# se recolectan todas hasta MAX_OTRA.
MAX_PER_CLASS = 500     # imgs por clase de cartoon
MAX_OTRA      = 3500    # imgs totales de la clase "otra"

BATCH_SIZE    = 32
NUM_EPOCHS_P1 = 5       # Fase 1: solo cabeza
NUM_EPOCHS_P2 = 15      # Fase 2: fine-tuning (con early stopping)
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-4    # con cosine annealing hacia 0
THRESHOLD     = 0.5     # fallback; el entrenamiento ajusta umbral por clase

UNFREEZE_BLOCKS     = 6     # bloques del backbone a descongelar en Fase 2
EARLY_STOP_PATIENCE = 5     # épocas sin mejora de F1 antes de cortar Fase 2
LABEL_SMOOTH        = 0.05  # suavizado de etiquetas para BCE
POS_WEIGHT_MAX      = 20.0  # tope del pos_weight por clase

VAL_SPLIT     = 0.2     # 20% validación
SEED          = 42

# DataLoader. En local con CPU decente puedes subir NUM_WORKERS.
# En WSL2 si da problemas de memoria compartida, baja a 0.
NUM_WORKERS   = 4
PIN_MEMORY    = torch.cuda.is_available()

# Normalización ImageNet (MobileNetV2 fue pre-entrenado con esto)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

# ─── Clases candidatas (el código detecta cuáles existen) ────
ALL_CLASSES = [
    'pokemon', 'one_piece', 'dragon_ball', 'bleach', 'ben_10',
    'hora_de_aventura', 'un_show_mas', 'naruto', 'doraemon', 'yu_gi_oh',
    'barbie', 'attack_on_titan', 'simpson', 'star_wars', 'otra'
]

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def detect_classes(dataset_root=DATASET_ROOT, all_classes=ALL_CLASSES):
    """Devuelve solo las clases que existen y tienen imágenes."""
    classes = []
    for cls in all_classes:
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = 0
        for _, _, files in os.walk(cls_dir):
            n += sum(1 for f in files if f.lower().endswith(IMG_EXTS))
        if n > 0:
            classes.append(cls)
    return classes
