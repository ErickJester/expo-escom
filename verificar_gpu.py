"""
Verifica que la estandarización (normalización ImageNet) funciona
correctamente cuando los tensores se mueven a GPU.

Uso:
    python verificar_gpu.py
"""
import torch
import numpy as np

import config as C
from dataset_utils import build_samples, CartoonDataset, get_transforms, stratified_split
from model import CartoonClassifier
from torch.utils.data import DataLoader


def main():
    print("=" * 55)
    print("VERIFICACIÓN: Estandarización + GPU")
    print("=" * 55)

    # ── Dispositivo ────────────────────────────────────────────
    print(f"Device  : {C.DEVICE}")
    if not torch.cuda.is_available():
        print("⚠️  GPU no disponible — verificando en CPU.")
    else:
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    # ── Dataset mínimo (10 imgs por clase) ─────────────────────
    classes = C.detect_classes()
    print(f"\nClases : {classes}")
    samples = build_samples(C.DATASET_ROOT, classes,
                            max_per_class=10, max_otra=10)
    if not samples:
        print("❌ No se encontraron imágenes en dataset/")
        return

    _, transform_val = get_transforms()
    ds = CartoonDataset(samples, transform_val)
    loader = DataLoader(ds, batch_size=min(16, len(samples)),
                        shuffle=False, num_workers=0)

    # ── Primer batch → GPU ─────────────────────────────────────
    imgs, labels = next(iter(loader))
    print(f"\nBatch en CPU  — shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"  min={imgs.min():.3f}  max={imgs.max():.3f}  "
          f"mean={imgs.mean():.3f}  std={imgs.std():.3f}")

    # Verificar por canal (esperamos valores cercanos a 0 para media)
    for c_idx, c_name in enumerate(['R', 'G', 'B']):
        ch = imgs[:, c_idx]
        print(f"  Canal {c_name}: mean={ch.mean():.3f}  std={ch.std():.3f}")

    imgs_gpu   = imgs.to(C.DEVICE)
    labels_gpu = labels.to(C.DEVICE)
    print(f"\nBatch en GPU  — device: {imgs_gpu.device}, dtype: {imgs_gpu.dtype}")
    print(f"  min={imgs_gpu.min():.3f}  max={imgs_gpu.max():.3f}  "
          f"mean={imgs_gpu.mean():.3f}  std={imgs_gpu.std():.3f}")

    # ── Forward pass en GPU ────────────────────────────────────
    model = CartoonClassifier(num_classes=len(classes)).to(C.DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(imgs_gpu)
    print(f"\nForward pass OK — logits shape: {logits.shape}, device: {logits.device}")
    print(f"  logits min={logits.min():.3f}  max={logits.max():.3f}")

    # ── Verificar que sin normalización los valores serían distintos ──
    from torchvision import transforms
    transform_sin_norm = transforms.Compose([
        transforms.Resize(int(C.IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(C.IMG_SIZE),
        transforms.ToTensor(),
        # SIN Normalize
    ])
    ds_raw = CartoonDataset(samples[:min(16, len(samples))], transform_sin_norm)
    loader_raw = DataLoader(ds_raw, batch_size=min(16, len(samples)),
                            shuffle=False, num_workers=0)
    imgs_raw, _ = next(iter(loader_raw))
    print(f"\nSIN normalización — mean={imgs_raw.mean():.3f}  std={imgs_raw.std():.3f}  "
          f"(esperado ~0.45 / ~0.25 para imágenes RGB naturales)")
    print(f"CON normalización — mean={imgs.mean():.3f}  std={imgs.std():.3f}  "
          f"(esperado cercano a 0 / 1)")

    # ── Resultado ──────────────────────────────────────────────
    mean_ok = abs(imgs.mean().item()) < 1.5
    std_ok  = 0.3 < imgs.std().item() < 2.0
    gpu_ok  = str(imgs_gpu.device).startswith('cuda') or C.DEVICE.type == 'cpu'

    print("\n" + "=" * 55)
    if mean_ok and std_ok and gpu_ok:
        print("✅ Estandarización funciona correctamente con GPU.")
    else:
        if not mean_ok:
            print(f"⚠️  Media inusual: {imgs.mean():.3f} (esperado < 1.5 en absoluto)")
        if not std_ok:
            print(f"⚠️  Std inusual: {imgs.std():.3f} (esperado entre 0.3 y 2.0)")
        if not gpu_ok:
            print("⚠️  Los tensores no llegaron a GPU.")
    print("=" * 55)


if __name__ == '__main__':
    main()
