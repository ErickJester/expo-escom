# ============================================================
# ToonVerse - Backend de inferencia (Flask + PyTorch)
#
# Sirve la página (ToonVerse/index.html) Y el endpoint /predict
# en el mismo puerto → sin CORS. La cámara del navegador manda
# un frame y recibe las probabilidades reales del modelo
# cartoon_v3.pt (MobileNetV2, 17 clases single-label).
#
# Uso:
#   py server.py
#   → abre http://localhost:8000 en Chrome
# ============================================================
import os
import sys
import io
import base64

# La consola de Windows usa cp1252 por defecto y revienta con emojis → fuerza UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import threading
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_from_directory

try:
    import timm as _timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False

# ─── Rutas ───────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))          # ...\Design
PROJECT_ROOT = os.path.dirname(HERE)                               # ...\expo-escom
STATIC_DIR   = os.path.join(HERE, "ToonVerse")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "entrenamiento_local")
MODELS_DIR   = os.path.join(TRAIN_DIR, "models")                   # ...\entrenamiento_local\models
MODEL_PATH   = os.path.join(MODELS_DIR, "toonverse_full.pt")

# ─── Arquitecturas ───────────────────────────────────────────
import torch.nn as nn
import torchvision.models as tv_models

# MobileNetV2 (Plan A / modelos anteriores)
sys.path.insert(0, TRAIN_DIR)
from model import CartoonClassifier  # noqa: E402

# ConvNeXt-Small (Plan B)
class CartoonConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone      = tv_models.convnext_small(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.norm     = backbone.classifier[0]
        self.head = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return self.head(torch.flatten(x, 1))

# ─── Preprocesado (idéntico a transform_plain de train.py) ───
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224
DEVICE        = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # squash a 224x224 (como en validación)
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── Mapa: 17 clases del modelo → 15 filas de la UI ──────────
# Las 3 subclases de ruido (otra/*) se suman en una sola "otra".
MODEL_TO_UI = {
    "aot":                 "attack_on_titan",
    "barbie":              "barbie",
    "ben_10":              "ben_10",
    "bleach":              "bleach",
    "doraemon":            "doraemon",
    "dragon_ball":         "dragon_ball",
    "hora_de_aventura":    "hora_de_aventura",
    "naruto":              "naruto",
    "one_piece":           "one_piece",
    "pokemon":             "pokemon",
    "simpson":             "simpson",
    "star_wars":           "star_wars",
    "unshowmas":           "un_show_mas",
    "yugioh":              "yu_gi_oh",
    "otra/cartoons_anime": "otra",
    "otra/noise":          "otra",
    "otra/real_life":      "otra",
}

# ─── Estado del modelo activo (mutable para hot-swap) ─────────
_model_lock = threading.Lock()
_state = {"model": None, "classes": [], "version": "?", "path": MODEL_PATH, "multilabel": False}

_EXCLUDED = {"features_cache.pt"}  # archivos .pt que no son modelos

def _load_model(path):
    """Carga un checkpoint y devuelve (model, classes, version, multilabel)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # ── Modelos timm (tienen clave "model_name") ─────────────
    if "model_name" in ckpt:
        if not _TIMM_OK:
            raise ValueError("timm no instalado. Ejecuta: pip install timm")
        mn  = ckpt["model_name"]
        cls = ckpt["classes"]
        m   = _timm.create_model(mn, pretrained=False, num_classes=len(cls))
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE).eval()
        ver = ckpt.get("version", f"timm/{len(cls)}cls")
        return m, cls, ver, True   # multilabel=True → sigmoid en infer()

    # ── Modelos propios (MobileNetV2 / ConvNeXt-Small) ───────
    cls  = ckpt["classes"]
    arch = ckpt.get("arch", "mobilenet_v2")
    if arch == "convnext_small":
        m = CartoonConvNeXt(num_classes=len(cls))
    else:
        m = CartoonClassifier(num_classes=len(cls), pretrained=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    return m, cls, ckpt.get("version", "?"), False  # multilabel=False → softmax

def _list_models():
    return sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pt") and f not in _EXCLUDED
    )

print(f"⏳ Cargando modelo desde {MODEL_PATH} ...")
_state["model"], _state["classes"], _state["version"], _state["multilabel"] = _load_model(MODEL_PATH)
print(f"✅ Modelo listo | {len(_state['classes'])} clases | versión {_state['version']} | device {DEVICE}")


@torch.no_grad()
def infer(img):
    """Devuelve {ui_id: prob} con las salidas mapeadas a filas de la UI."""
    with _model_lock:
        m, cls, multilabel = _state["model"], _state["classes"], _state["multilabel"]
    x   = transform(img).unsqueeze(0).to(DEVICE)
    out = m(x)                                  # [1, n_classes]
    if multilabel:
        probs = torch.sigmoid(out[0]).cpu().numpy()   # independiente por clase
    else:
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
    ui = {}
    for c, p in zip(cls, probs):
        uid = MODEL_TO_UI.get(c, c)
        ui[uid] = ui.get(uid, 0.0) + float(p)
    return ui


# ─── App ─────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "classes": len(_state["classes"]),
        "version": _state["version"],
        "model":   os.path.basename(_state["path"]),
    })


@app.route("/models")
def list_models():
    return jsonify({"models": _list_models(), "active": os.path.basename(_state["path"])})


@app.route("/switch", methods=["POST"])
def switch_model():
    name = (request.get_json(force=True, silent=True) or {}).get("model", "")
    if not name or "/" in name or not name.endswith(".pt"):
        return jsonify({"error": "nombre inválido"}), 400
    path = os.path.join(MODELS_DIR, name)
    if not os.path.isfile(path):
        return jsonify({"error": "archivo no encontrado"}), 404
    try:
        m, cls, ver, multilabel = _load_model(path)
        with _model_lock:
            _state["model"]      = m
            _state["classes"]    = cls
            _state["version"]    = ver
            _state["multilabel"] = multilabel
            _state["path"]       = path
        kind = "multilabel" if multilabel else "single-label"
        print(f"🔄 Modelo cambiado → {name} | {len(cls)} clases | v{ver} | {kind}")
        return jsonify({"ok": True, "model": name, "classes": len(cls), "version": ver})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    img_data = data.get("image", "")
    if not img_data:
        return jsonify({"error": "no image"}), 400
    try:
        b64 = img_data.split(",", 1)[-1]            # quita el prefijo data:image/...
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"imagen inválida: {e}"}), 400
    return jsonify({"probs": infer(img)})


@app.route("/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    url  = f"http://localhost:{port}"
    # El launcher (iniciar.bat) pone OPEN_BROWSER=1 para abrir Chrome solo.
    if os.environ.get("OPEN_BROWSER") == "1":
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"🚀 {url}  (Ctrl+C para detener)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
