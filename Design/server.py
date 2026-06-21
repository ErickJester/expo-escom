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

# ─── Rutas ───────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))          # ...\Design
PROJECT_ROOT = os.path.dirname(HERE)                               # ...\expo-escom
STATIC_DIR   = os.path.join(HERE, "ToonVerse")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "entrenamiento_local")
MODEL_PATH   = os.path.join(TRAIN_DIR, "models", "toonverse_full.pt")

# Reutiliza la arquitectura REAL del entrenamiento (única fuente de verdad)
sys.path.insert(0, TRAIN_DIR)
from model import CartoonClassifier  # noqa: E402

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
_state = {"model": None, "classes": [], "version": "?", "path": MODEL_PATH}

def _load_model(path):
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    cls   = ckpt["classes"]
    m     = CartoonClassifier(num_classes=len(cls), pretrained=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    return m, cls, ckpt.get("version", "?")

def _list_models():
    models_dir = os.path.join(TRAIN_DIR, "models")
    return sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))

print(f"⏳ Cargando modelo desde {MODEL_PATH} ...")
_state["model"], _state["classes"], _state["version"] = _load_model(MODEL_PATH)
print(f"✅ Modelo listo | {len(_state['classes'])} clases | versión {_state['version']} | device {DEVICE}")


@torch.no_grad()
def infer(img):
    """Devuelve {ui_id: prob} con las salidas mapeadas a filas de la UI."""
    with _model_lock:
        m, cls = _state["model"], _state["classes"]
    x     = transform(img).unsqueeze(0).to(DEVICE)
    probs = F.softmax(m(x), dim=1)[0].cpu().numpy()
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
    path = os.path.join(TRAIN_DIR, "models", name)
    if not os.path.isfile(path):
        return jsonify({"error": "archivo no encontrado"}), 404
    try:
        m, cls, ver = _load_model(path)
        with _model_lock:
            _state["model"]   = m
            _state["classes"] = cls
            _state["version"] = ver
            _state["path"]    = path
        print(f"🔄 Modelo cambiado → {name} | {len(cls)} clases | v{ver}")
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
    print(f"🚀 http://localhost:{port}  (Ctrl+C para detener)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
