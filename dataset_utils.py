# ============================================================
# ExpoEscom - Carga de datos
# ============================================================
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import config as C


class CartoonDataset(Dataset):
    """Dataset multi-etiqueta. Cada muestra es (ruta, vector_one_hot)."""

    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Imagen corrupta → gris de reemplazo (no rompe el batch)
            img = Image.new('RGB', (C.IMG_SIZE, C.IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_samples(root_dir, classes, max_per_class, max_otra, seed=C.SEED):
    """
    Construye lista de (ruta_imagen, vector_one_hot).
    os.walk recorre tanto carpetas simples (pokemon/) como
    'otra/' que tiene subcarpetas.
    """
    rng = np.random.default_rng(seed)
    all_samples = []

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        limite = max_otra if cls_name == 'otra' else max_per_class

        imgs = []
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith(C.IMG_EXTS):
                    imgs.append(os.path.join(root, f))

        rng.shuffle(imgs)              # mezclar antes de cortar
        imgs = imgs[:limite]
        if not imgs:
            print(f"  ⚠️  {cls_name}: sin imágenes")
            continue

        label = np.zeros(len(classes), dtype=np.float32)
        label[cls_idx] = 1.0
        for path in imgs:
            all_samples.append((path, label.copy()))

        print(f"  ✅ {cls_name:22s}: {len(imgs):,} imágenes")

    rng.shuffle(all_samples)
    return all_samples


def get_transforms():
    """Devuelve (transform_train, transform_val)."""
    transform_train = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    return transform_train, transform_val
