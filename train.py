# ============================================================
# ExpoEscom - Mini PoC | Entrenamiento LOCAL (RTX 2070 Super)
# Dataset: clases de caricaturas/anime + "otra"
#
# Uso:
#   python train.py
# ============================================================
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score
import matplotlib
matplotlib.use('Agg')              # backend sin ventana (guarda PNG)
import matplotlib.pyplot as plt

import config as C
from dataset_utils import CartoonDataset, build_samples, get_transforms
from model import CartoonClassifier

warnings.filterwarnings('ignore')


# ─── Métricas ────────────────────────────────────────────────
def compute_metrics(y_true, y_logits, threshold=C.THRESHOLD):
    """Macro F1 y mAP dado arrays numpy [N, C]."""
    probs  = 1 / (1 + np.exp(-y_logits))
    y_pred = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    aps = [
        average_precision_score(y_true[:, c], probs[:, c])
        for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0
    ]
    return f1, (np.mean(aps) if aps else 0.0)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(C.DEVICE), labels.to(C.DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"  batch {i+1}/{len(loader)}", end='\r')
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, yt_all, yl_all = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(C.DEVICE), labels.to(C.DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        yt_all.append(labels.cpu().numpy())
        yl_all.append(logits.cpu().numpy())
    y_true, y_logits = np.concatenate(yt_all), np.concatenate(yl_all)
    f1, mAP = compute_metrics(y_true, y_logits)
    return total_loss / len(loader), f1, mAP


def save_checkpoint(state_dict, classes, epoch, best_f1, phase, history):
    torch.save({
        'model_state_dict': state_dict,
        'classes'         : classes,
        'num_classes'     : len(classes),
        'best_f1'         : best_f1,
        'epoch'           : epoch,
        'phase'           : phase,
        'history'         : history,
        'threshold'       : C.THRESHOLD,
    }, C.SAVE_PATH)


def run_phase(name, model, train_loader, val_loader, criterion, optimizer,
              scheduler, n_epochs, classes, history, best_f1, best_state,
              step_with_metric):
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1, val_mAP = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_f1) if step_with_metric else scheduler.step()
        elapsed = time.time() - t0

        for k, v in zip(['train_loss', 'val_loss', 'val_f1', 'val_mAP'],
                        [train_loss, val_loss, val_f1, val_mAP]):
            history[k].append(v)

        star = " ⭐ guardado" if val_f1 > best_f1 else ""
        print(f"[{name} {epoch:02d}/{n_epochs}] {elapsed:.0f}s | "
              f"Loss {train_loss:.4f}/{val_loss:.4f} | "
              f"F1 {val_f1:.4f} | mAP {val_mAP:.4f}{star}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_checkpoint(best_state, classes, epoch, best_f1, name, history)
    return best_f1, best_state


def plot_history(history, classes, n_samples, best_f1):
    total_ep = C.NUM_EPOCHS_P1 + C.NUM_EPOCHS_P2
    ep_r = range(1, total_ep + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'ExpoEscom Mini PoC | {len(classes)} clases | '
        f'{n_samples:,} imgs | Mejor F1: {best_f1:.4f}', fontsize=12)

    axes[0].plot(ep_r, history['train_loss'], 'b-o', ms=4, label='Train')
    axes[0].plot(ep_r, history['val_loss'],   'r-o', ms=4, label='Val')
    axes[0].axvline(C.NUM_EPOCHS_P1, color='gray', ls='--', lw=1.5, label='→ Fase 2')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep_r, history['val_f1'], 'g-o', ms=4)
    axes[1].axhline(0.5, color='orange', ls='--', lw=1.5, label='Meta mínima')
    axes[1].axhline(0.7, color='green',  ls='--', lw=1.5, label='Meta buena')
    axes[1].set_title('Macro F1'); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep_r, history['val_mAP'], 'm-o', ms=4)
    axes[2].set_title('mAP'); axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Época')
    plt.tight_layout()
    graph_path = C.SAVE_PATH.replace('.pt', '_grafica.png')
    plt.savefig(graph_path, dpi=100, bbox_inches='tight')
    print(f"📈 Gráfica guardada: {graph_path}")


def main():
    print("═" * 60)
    print("🚀 ExpoEscom Mini PoC — Entrenamiento LOCAL")
    print("═" * 60)
    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {C.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("⚠️  No se detectó GPU — entrenará en CPU (lento).")
    os.makedirs(C.SAVE_DIR, exist_ok=True)

    # ── Datos ───────────────────────────────────────────────
    classes = C.detect_classes()
    print(f"\nClases detectadas ({len(classes)}): {classes}")
    print("\nConstruyendo dataset...")
    all_samples = build_samples(C.DATASET_ROOT, classes,
                                C.MAX_PER_CLASS, C.MAX_OTRA)

    split_idx     = int((1 - C.VAL_SPLIT) * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]
    print(f"\nTotal: {len(all_samples):,} | "
          f"Train: {len(train_samples):,} | Val: {len(val_samples):,}")

    transform_train, transform_val = get_transforms()
    train_ds = CartoonDataset(train_samples, transform_train)
    val_ds   = CartoonDataset(val_samples,   transform_val)

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)

    # ── Modelo ──────────────────────────────────────────────
    model = CartoonClassifier(num_classes=len(classes)).to(C.DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    total = sum(p.numel() for p in model.parameters())
    trn   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Modelo listo — params totales {total:,} | "
          f"entrenables P1 {trn:,} ({100*trn/total:.1f}%)")

    history    = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_mAP': []}
    best_f1    = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Fase 1: solo cabeza ─────────────────────────────────
    print("\n" + "═" * 60)
    print(f"🚀 FASE 1 — Backbone congelado | LR={C.LR_PHASE1} | "
          f"Épocas={C.NUM_EPOCHS_P1}")
    print("═" * 60)
    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.LR_PHASE1, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode='max', factor=0.5, patience=2)
    best_f1, best_state = run_phase(
        'P1', model, train_loader, val_loader, criterion, opt1, sch1,
        C.NUM_EPOCHS_P1, classes, history, best_f1, best_state,
        step_with_metric=True)
    print(f"🏁 Mejor F1 Fase 1: {best_f1:.4f}")

    # ── Fase 2: fine-tuning ─────────────────────────────────
    print("\n" + "═" * 60)
    print(f"🔥 FASE 2 — Fine-tuning últimas 3 capas | LR={C.LR_PHASE2} | "
          f"Épocas={C.NUM_EPOCHS_P2}")
    print("═" * 60)
    model.load_state_dict(best_state)
    model.unfreeze_last_n(n=3)
    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.LR_PHASE2, weight_decay=1e-5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=C.NUM_EPOCHS_P2)
    best_f1, best_state = run_phase(
        'P2', model, train_loader, val_loader, criterion, opt2, sch2,
        C.NUM_EPOCHS_P2, classes, history, best_f1, best_state,
        step_with_metric=False)

    # ── Resultados ──────────────────────────────────────────
    model.load_state_dict(best_state)
    plot_history(history, classes, len(all_samples), best_f1)

    print("\n" + "═" * 60)
    print("📊 RESUMEN")
    print("═" * 60)
    print(f"Clases          : {len(classes)} → {classes}")
    print(f"Imágenes        : {len(all_samples):,}")
    print(f"Mejor F1 macro  : {best_f1:.4f}")
    print(f"Modelo guardado : {C.SAVE_PATH}")
    if best_f1 >= 0.75:
        print("✅✅✅  EXCELENTE — Arquitectura validada.")
    elif best_f1 >= 0.55:
        print("✅✅    BIEN.")
    elif best_f1 >= 0.40:
        print("✅      ACEPTABLE.")
    else:
        print("⚠️     REVISAR — verifica que las carpetas no estén mezcladas.")


if __name__ == '__main__':
    main()
