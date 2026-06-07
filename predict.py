# ============================================================
# ExpoEscom - Inferencia con el modelo entrenado
#
# Uso:
#   python predict.py ruta/a/imagen.jpg
#   python predict.py img1.jpg img2.png ...
# ============================================================
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

import config as C
from model import CartoonClassifier


def load_model(ckpt_path=C.SAVE_PATH):
    # weights_only=False: el checkpoint incluye metadatos (clases, history)
    # además de los pesos. El archivo es propio, no de una fuente externa.
    ckpt    = torch.load(ckpt_path, map_location=C.DEVICE, weights_only=False)
    classes = ckpt['classes']
    model   = CartoonClassifier(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(C.DEVICE).eval()
    threshold = ckpt.get('threshold', C.THRESHOLD)
    return model, classes, threshold


def build_transform():
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


@torch.no_grad()
def predict(model, classes, threshold, transform, img_path):
    img = Image.open(img_path).convert('RGB')
    x   = transform(img).unsqueeze(0).to(C.DEVICE)
    probs = torch.sigmoid(model(x))[0].cpu().numpy()

    ranking = sorted(zip(classes, probs), key=lambda t: t[1], reverse=True)
    print(f"\n🖼️  {img_path}")
    for cls, p in ranking:
        mark = "✅" if p >= threshold else "  "
        bar  = "█" * int(p * 20)
        print(f"  {mark} {cls:18s} {p*100:5.1f}%  {bar}")
    top_cls, top_p = ranking[0]
    print(f"  → Predicción: {top_cls} ({top_p*100:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Uso: python predict.py <imagen> [imagen2 ...]")
        sys.exit(1)

    model, classes, threshold = load_model()
    transform = build_transform()
    print(f"✅ Modelo cargado | {len(classes)} clases | umbral {threshold}")

    for img_path in sys.argv[1:]:
        try:
            predict(model, classes, threshold, transform, img_path)
        except Exception as e:
            print(f"  ❌ Error con {img_path}: {e}")


if __name__ == '__main__':
    main()
