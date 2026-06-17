# ============================================================
# ExpoEscom - Entrenamiento LOCAL v3.0 (feature caching)
#
#   python train.py
#
# Estrategia (igual que el notebook de Colab v3.0):
#   1) Backbone congelado → extrae features de TODAS las imágenes 1 sola vez
#      y las cachea en disco. La cabeza se entrena sobre esos vectores (rápido).
#   2) Fine-tuning de los últimos bloques sobre un subconjunto balanceado.
#
# El guard if __name__ == "__main__" es OBLIGATORIO en Windows para que
# num_workers > 0 funcione (multiprocessing).
# ============================================================
import gc
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use("Agg")          # sin ventana → guarda la gráfica a archivo
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from dataset import CartoonDataset, detect_classes, build_samples, split_samples
from model import CartoonClassifier


# ─── Transforms ──────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])
transform_plain = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

criterion = nn.CrossEntropyLoss()


def make_loader(samples, transform, batch_size, shuffle):
    kwargs = dict(num_workers=config.NUM_WORKERS, pin_memory=config.USE_AMP)
    if config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = config.PREFETCH
    return DataLoader(CartoonDataset(samples, transform),
                      batch_size=batch_size, shuffle=shuffle, **kwargs)


# ─── Funciones núcleo ────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, desc=""):
    """Una pasada por el backbone congelado → vectores [N, 1280] en fp16."""
    model.eval()
    feats, labs = [], []
    total_imgs = len(loader.dataset)
    bar = tqdm(loader, desc=f"  {desc}", unit="batch",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                          "[{elapsed}<{remaining}, {rate_fmt}]")
    t0 = time.time()
    for imgs, labels in bar:
        imgs = imgs.to(config.DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.USE_AMP):
            f = model.extract(imgs)
        feats.append(f.half().cpu())
        labs.append(labels)
        done = len(labs) * loader.batch_size
        img_s = done / (time.time() - t0)
        bar.set_postfix(imgs=f"{min(done,total_imgs):,}", img_s=f"{img_s:.0f}")
    return torch.cat(feats), torch.cat(labs)


def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total = 0.0
    bar = tqdm(loader, desc="  train", leave=False, unit="batch",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    for imgs, labels in bar:
        imgs   = imgs.to(config.DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(config.DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.USE_AMP):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, gts, preds = 0.0, [], []
    bar = tqdm(loader, desc="  val  ", leave=False, unit="batch",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for imgs, labels in bar:
        imgs   = imgs.to(config.DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(config.DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.USE_AMP):
            logits = model(imgs)
            total += criterion(logits, labels).item()
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(labels.cpu().numpy())
    y_true, y_pred = np.concatenate(gts), np.concatenate(preds)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1c = f1_score(y_true, y_pred, average=None,    zero_division=0)
    return total / len(loader), f1m, f1c


def save_best(state_dict, classes, best_f1, epoch, phase, history):
    torch.save({
        "model_state_dict": state_dict,
        "classes"         : classes,
        "num_classes"     : len(classes),
        "best_f1"         : best_f1,
        "epoch"           : epoch,
        "phase"           : phase,
        "history"         : history,
        "version"         : config.VERSION,
    }, config.SAVE_PATH)


def plot_history(history, num_classes, n_imgs, best_f1):
    ep = range(1, len(history["val_f1"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"ExpoEscom v3 LOCAL | {num_classes} clases | "
                 f"{n_imgs:,} imgs | Mejor F1: {best_f1:.4f}")
    axes[0].plot(ep, history["train_loss"], "b-o", ms=3, label="Train")
    axes[0].plot(ep, history["val_loss"],   "r-o", ms=3, label="Val")
    axes[0].axvline(config.NUM_EPOCHS_P1 + 0.5, color="gray", ls="--", lw=1.5, label="→ Fase 2")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["val_f1"], "g-o", ms=3)
    axes[1].axhline(0.7, color="green", ls="--", lw=1.5, label="Meta buena")
    axes[1].set_title("Macro F1"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    for a in axes:
        a.set_xlabel("Época")
    plt.tight_layout()
    out = config.SAVE_PATH.replace(".pt", "_grafica.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"\n📈 Gráfica: {out}")


# ─── Orquestación ────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"📦 ExpoEscom Entrenamiento LOCAL v{config.VERSION}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True   # solo ayuda en Ampere+ (no en Turing)
        torch.backends.cudnn.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name} | {props.total_memory/1e9:.1f} GB")
    else:
        print("⚠️  Sin GPU CUDA detectada — irá MUY lento en CPU.")
    print(f"Device : {config.DEVICE} | AMP: {config.USE_AMP}\n")

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 1) Clases + muestras
    classes = detect_classes(config.DATASET_ROOT)
    num_classes = len(classes)
    if num_classes < 2:
        raise SystemExit(
            f"Se detectaron {num_classes} clases en {config.DATASET_ROOT!r}.\n"
            "Revisa la ruta DATASET_ROOT en config.py (debe tener una subcarpeta por clase)."
        )
    print(f"Clases detectadas ({num_classes}): {classes}\n")

    print("Recolectando imágenes...")
    all_samples = build_samples(config.DATASET_ROOT, classes)
    train_samples, val_samples, class_counts = split_samples(all_samples, num_classes)
    print(f"\nTrain: {len(train_samples):,} | Val: {len(val_samples):,}\n")

    # 2) Modelo
    model = CartoonClassifier(num_classes).to(config.DEVICE).to(memory_format=torch.channels_last)

    # 3) Features (1 sola pasada) + cache en disco
    if os.path.exists(config.FEATURE_CACHE):
        print("✅ Cargando features del cache...")
        cache = torch.load(config.FEATURE_CACHE)
        train_feats, train_lab = cache["train_feats"], cache["train_labels"]
        val_feats,   val_lab   = cache["val_feats"],   cache["val_labels"]
    else:
        print("Extrayendo features (backbone congelado, 1 pasada)...")
        t0 = time.time()
        feat_loader      = make_loader(train_samples, transform_plain, config.BATCH_INFER, shuffle=False)
        val_loader_plain = make_loader(val_samples,   transform_plain, config.BATCH_INFER, shuffle=False)
        train_feats, train_lab = extract_features(model, feat_loader, "train")
        val_feats,   val_lab   = extract_features(model, val_loader_plain, "val")
        torch.save({"train_feats": train_feats, "train_labels": train_lab,
                    "val_feats": val_feats, "val_labels": val_lab}, config.FEATURE_CACHE)
        print(f"✅ Cache guardado en {config.FEATURE_CACHE} ({time.time()-t0:.0f}s)\n")

    print(f"   Train feats: {tuple(train_feats.shape)} | Val feats: {tuple(val_feats.shape)}\n")

    # 4) FASE 1 — cabeza sobre features cacheadas
    print("=" * 60)
    print("🚀 FASE 1 — cabeza sobre features cacheadas (sin tocar imágenes)")
    print("=" * 60)
    ft = DataLoader(TensorDataset(train_feats, train_lab), batch_size=config.HEAD_BATCH, shuffle=True)
    fv = DataLoader(TensorDataset(val_feats,   val_lab),   batch_size=config.HEAD_BATCH, shuffle=False)

    # Loss ponderada por clase → compensa el desbalance del ruido
    cls_w = class_counts.sum() / (num_classes * np.clip(class_counts, 1, None))
    criterion_w = nn.CrossEntropyLoss(
        weight=torch.tensor(cls_w, dtype=torch.float32, device=config.DEVICE))

    opt1 = torch.optim.Adam(model.head.parameters(), lr=config.LR_PHASE1, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state = 0.0, None

    for epoch in range(1, config.NUM_EPOCHS_P1 + 1):
        t0 = time.time()
        model.head.train()
        tl = 0.0
        bar = tqdm(ft, desc=f"  [P1 {epoch:02d}/{config.NUM_EPOCHS_P1}] train",
                   leave=False, unit="batch",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
        for fb, lb in bar:
            fb, lb = fb.float().to(config.DEVICE), lb.to(config.DEVICE)
            opt1.zero_grad()
            loss = criterion_w(model.head(fb), lb)
            loss.backward()
            opt1.step()
            tl += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        tl /= len(ft)

        model.head.eval()
        vl, gts, preds = 0.0, [], []
        with torch.no_grad():
            for fb, lb in tqdm(fv, desc=f"  [P1 {epoch:02d}/{config.NUM_EPOCHS_P1}] val  ",
                               leave=False, unit="batch",
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
                out = model.head(fb.float().to(config.DEVICE))
                vl += criterion_w(out, lb.to(config.DEVICE)).item()
                preds.append(out.argmax(1).cpu().numpy())
                gts.append(lb.numpy())
        vl /= len(fv)
        val_f1 = f1_score(np.concatenate(gts), np.concatenate(preds),
                          average="macro", zero_division=0)
        sch1.step(val_f1)

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["val_f1"].append(val_f1)
        star = " ⭐" if val_f1 > best_f1 else ""
        print(f"[P1 {epoch:02d}/{config.NUM_EPOCHS_P1}] {time.time()-t0:.0f}s | "
              f"loss {tl:.4f}/{vl:.4f} | F1 {val_f1:.4f}{star}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_best(best_state, classes, best_f1, epoch, "P1", history)

    print(f"\n🏁 Mejor F1 Fase 1: {best_f1:.4f}\n")

    # liberar features de RAM antes de la Fase 2 (que usa imágenes reales)
    del train_feats, val_feats, ft, fv
    gc.collect()
    torch.cuda.empty_cache()

    # 5) FASE 2 — fine-tuning sobre subconjunto balanceado
    print("=" * 60)
    print(f"🔥 FASE 2 — fine-tuning {config.UNFREEZE_N} bloques | {config.FT_PER_CLASS:,} imgs/clase")
    print("=" * 60)
    by_c = defaultdict(list)
    for s in train_samples:
        by_c[s[1]].append(s)
    rng = np.random.default_rng(0)
    ft_samples = []
    for _, items in by_c.items():
        items = list(items)
        rng.shuffle(items)
        ft_samples += items[:config.FT_PER_CLASS]
    rng.shuffle(ft_samples)
    print(f"Subconjunto fine-tuning: {len(ft_samples):,} imágenes")

    ft_loader  = make_loader(ft_samples,  transform_train, config.BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_samples, transform_plain, config.BATCH_SIZE, shuffle=False)

    model.load_state_dict(best_state)
    model.unfreeze_last_n(config.UNFREEZE_N)
    model = model.to(memory_format=torch.channels_last)

    opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config.LR_PHASE2, weight_decay=1e-5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=config.NUM_EPOCHS_P2)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    for epoch in range(1, config.NUM_EPOCHS_P2 + 1):
        t0 = time.time()
        train_loss = train_epoch(model, ft_loader, opt2, scaler)
        val_loss, val_f1, _ = eval_epoch(model, val_loader)
        sch2.step()

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        star = " ⭐" if val_f1 > best_f1 else ""
        print(f"[P2 {epoch:02d}/{config.NUM_EPOCHS_P2}] {time.time()-t0:.0f}s | "
              f"loss {train_loss:.4f}/{val_loss:.4f} | F1 {val_f1:.4f}{star}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_best(best_state, classes, best_f1, epoch, "P2", history)

    print(f"\n🏆 Mejor F1 Final: {best_f1:.4f}")
    print(f"   Modelo: {config.SAVE_PATH}")

    # 6) Reporte final + gráfica
    model.load_state_dict(best_state)
    _, final_f1, f1c = eval_epoch(model, val_loader)
    plot_history(history, num_classes, len(all_samples), best_f1)

    print("\n" + "=" * 60)
    print("📊 F1 POR CLASE (validation)")
    print("=" * 60)
    for cls, f1 in zip(classes, f1c):
        alert = " ⚠️" if f1 < 0.5 else ""
        print(f"  {cls:22s}: {f1:.4f}  {'█'*int(f1*20)}{alert}")
    print(f"\n  {'MACRO F1':22s}: {final_f1:.4f}")


if __name__ == "__main__":
    main()
