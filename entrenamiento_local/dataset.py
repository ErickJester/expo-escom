# ============================================================
# ExpoEscom - Dataset y utilidades de datos (LOCAL v3.0)
# ============================================================
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import config


class CartoonDataset(Dataset):
    """Lee (ruta, etiqueta_entera). Etiqueta como long → CrossEntropyLoss."""

    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # imagen corrupta → gris de reemplazo, no rompe el batch
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def detect_classes(root):
    """
    Toda subcarpeta con imágenes es una clase.
    Si una carpeta de ruido (config.NOISE_CLASS_NAMES) contiene subcarpetas,
    se expanden como clases separadas (p.ej. 'otra/ruido', 'otra/vida_real').
    Las clases de ruido se colocan al final.
    """
    if not os.path.isdir(root):
        return []
    cartoons, noise = [], []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if config.is_noise(name):
            # buscar subcarpetas con imágenes dentro de "otra"
            subs = []
            for subname in sorted(os.listdir(d)):
                subd = os.path.join(d, subname)
                if not os.path.isdir(subd):
                    continue
                has_img = any(
                    f.lower().endswith(config.IMG_EXTS)
                    for _, _, fs in os.walk(subd) for f in fs
                )
                if has_img:
                    subs.append(f"{name}/{subname}")
            if subs:
                noise.extend(subs)
            else:
                # sin subcarpetas → clase de ruido directa
                has_img = any(
                    f.lower().endswith(config.IMG_EXTS)
                    for _, _, fs in os.walk(d) for f in fs
                )
                if has_img:
                    noise.append(name)
        else:
            has_img = False
            for _, _, files in os.walk(d):
                if any(f.lower().endswith(config.IMG_EXTS) for f in files):
                    has_img = True
                    break
            if has_img:
                cartoons.append(name)
    return cartoons + noise


def build_samples(root, classes):
    """Lista de (ruta, idx_clase), capada por clase y mezclada."""
    rng = np.random.default_rng(config.SEED)
    all_samples = []
    for idx, name in enumerate(classes):
        cls_dir = os.path.join(root, *name.split("/"))   # soporta "otra/ruido"
        limite  = config.MAX_OTRA if config.is_noise(name) else config.MAX_PER_CLASS
        imgs = []
        for r, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith(config.IMG_EXTS):
                    imgs.append(os.path.join(r, f))
        rng.shuffle(imgs)
        imgs = imgs[:limite]
        for p in imgs:
            all_samples.append((p, idx))
        print(f"  ✅ {name:22s}: {len(imgs):,}")
    rng.shuffle(all_samples)
    return all_samples


def split_samples(all_samples, num_classes):
    """Split con validación CAPADA por clase (VAL_PER_CLASS)."""
    by_class = defaultdict(list)
    for s in all_samples:
        by_class[s[1]].append(s)

    rng = np.random.default_rng(config.SEED)
    train, val = [], []
    for _, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        val   += items[:config.VAL_PER_CLASS]
        train += items[config.VAL_PER_CLASS:]
    rng.shuffle(train)

    counts = np.bincount([s[1] for s in train], minlength=num_classes).astype(float)
    return train, val, counts
