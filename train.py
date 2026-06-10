# ============================================================
# ExpoEscom - Mini PoC | Entrenamiento LOCAL (RTX 2070 Super)
# Dataset: clases de caricaturas/anime + "otra"
#
# Uso:
#   python train.py                    # guarda con timestamp automático
#   python train.py --name mi_prueba   # guarda como models/mi_prueba.pt
# ============================================================
import argparse
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
from dataset_utils import (CartoonDataset, build_samples, get_transforms,
                           stratified_split)
from model import CartoonClassifier

warnings.filterwarnings('ignore')

USE_AMP = torch.cuda.is_available()


# ─── Métricas ────────────────────────────────────────────────
def compute_metrics(y_true, y_logits, thresholds=None):
    """Macro F1 y mAP dado arrays numpy [N, C]. thresholds: escalar o [C]."""
    if thresholds is None:
        thresholds = C.THRESHOLD
    probs  = 1 / (1 + np.exp(-y_logits))
    y_pred = (probs >= np.asarray(thresholds)).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    aps = [
        average_precision_score(y_true[:, c], probs[:, c])
        for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0
    ]
    return f1, (np.mean(aps) if aps else 0.0)


def tune_thresholds(y_true, y_logits):
    """
    Busca por clase el umbral que maximiza su F1 en validación.
    Con desbalance (otra = 7x) el 0.5 global castiga a las clases chicas.
    """
    probs = 1 / (1 + np.exp(-y_logits))
    thresholds = []
    for c in range(y_true.shape[1]):
        best_t, best_f = C.THRESHOLD, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            f = f1_score(y_true[:, c], (probs[:, c] >= t).astype(int),
                         zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thresholds.append(round(float(best_t), 2))
    return thresholds


def compute_pos_weight(samples, n_classes):
    """pos_weight = negativos/positivos por clase, acotado a POS_WEIGHT_MAX."""
    labels = np.stack([lbl for _, lbl in samples])
    pos = labels.sum(axis=0)
    weight = (len(labels) - pos) / np.maximum(pos, 1)
    return torch.tensor(np.clip(weight, 1.0, C.POS_WEIGHT_MAX),
                        dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    eps = C.LABEL_SMOOTH
    for i, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(C.DEVICE, non_blocking=True)
        labels = labels.to(C.DEVICE, non_blocking=True)
        labels = labels * (1 - eps) + 0.5 * eps     # suavizado
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=USE_AMP):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"  batch {i+1}/{len(loader)}", end='\r')
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, yt_all, yl_all = 0.0, [], []
    for imgs, labels in loader:
        imgs   = imgs.to(C.DEVICE, non_blocking=True)
        labels = labels.to(C.DEVICE, non_blocking=True)
        with torch.autocast('cuda', enabled=USE_AMP):
            logits = model(imgs)
            total_loss += criterion(logits, labels).item()
        yt_all.append(labels.cpu().numpy())
        yl_all.append(logits.float().cpu().numpy())
    y_true, y_logits = np.concatenate(yt_all), np.concatenate(yl_all)
    f1, mAP = compute_metrics(y_true, y_logits)
    return total_loss / len(loader), f1, mAP, y_true, y_logits


def save_checkpoint(state_dict, classes, epoch, best_f1, phase, history,
                    thresholds=None):
    torch.save({
        'model_state_dict': state_dict,
        'classes'         : classes,
        'num_classes'     : len(classes),
        'best_f1'         : best_f1,
        'epoch'           : epoch,
        'phase'           : phase,
        'history'         : history,
        'threshold'       : C.THRESHOLD,
        'thresholds'      : thresholds,
    }, C.SAVE_PATH)


def run_phase(name, model, train_loader, val_loader, criterion, optimizer,
              scheduler, n_epochs, classes, history, best_f1, best_state,
              step_with_metric, scaler, patience=None):
    epochs_run = 0
    since_best = 0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 scaler)
        val_loss, val_f1, val_mAP, _, _ = eval_epoch(model, val_loader,
                                                     criterion)
        scheduler.step(val_f1) if step_with_metric else scheduler.step()
        elapsed = time.time() - t0
        epochs_run += 1

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
            since_best = 0
        else:
            since_best += 1
            if patience and since_best >= patience:
                print(f"   ⏹️  Early stop: {patience} épocas sin mejora.")
                break
    return best_f1, best_state, epochs_run


def plot_history(history, classes, n_samples, best_f1, p1_epochs):
    total_ep = len(history['train_loss'])
    ep_r = range(1, total_ep + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'ExpoEscom Mini PoC | {len(classes)} clases | '
        f'{n_samples:,} imgs | Mejor F1: {best_f1:.4f}', fontsize=12)

    axes[0].plot(ep_r, history['train_loss'], 'b-o', ms=4, label='Train')
    axes[0].plot(ep_r, history['val_loss'],   'r-o', ms=4, label='Val')
    axes[0].axvline(p1_epochs, color='gray', ls='--', lw=1.5, label='→ Fase 2')
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='',
                        help='Nombre base del modelo (sin .pt). '
                             'Por defecto usa timestamp para no pisar runs anteriores.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_name  = args.name if args.name else time.strftime('run_%Y%m%d_%H%M%S')
    save_path = os.path.join(C.SAVE_DIR, f'{run_name}.pt')
    # Sobreescribe save_path para todo el run
    C.SAVE_PATH = save_path
    print("═" * 60)
    print("🚀 ExpoEscom Mini PoC — Entrenamiento LOCAL")
    print("═" * 60)
    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {C.DEVICE}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"AMP     : activado (mixed precision)")
    else:
        print("⚠️  No se detectó GPU — entrenará en CPU (lento).")
    os.makedirs(C.SAVE_DIR, exist_ok=True)

    # ── Datos ───────────────────────────────────────────────
    classes = C.detect_classes()
    print(f"\nClases detectadas ({len(classes)}): {classes}")
    print("\nConstruyendo dataset...")
    all_samples = build_samples(C.DATASET_ROOT, classes,
                                C.MAX_PER_CLASS, C.MAX_OTRA)

    train_samples, val_samples = stratified_split(all_samples)
    print(f"\nTotal: {len(all_samples):,} | "
          f"Train: {len(train_samples):,} | Val: {len(val_samples):,} "
          f"(split estratificado)")

    transform_train, transform_val = get_transforms()
    train_ds = CartoonDataset(train_samples, transform_train)
    val_ds   = CartoonDataset(val_samples,   transform_val)

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)

    # ── Modelo ──────────────────────────────────────────────
    model = CartoonClassifier(num_classes=len(classes)).to(C.DEVICE)

    # pos_weight compensa el desbalance (otra tiene 7x más imágenes):
    # sin esto el modelo aprende a irse a lo seguro con "otra".
    pos_weight = compute_pos_weight(train_samples, len(classes)).to(C.DEVICE)
    print(f"\npos_weight por clase: "
          f"{[f'{c}={w:.1f}' for c, w in zip(classes, pos_weight.tolist())]}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    total = sum(p.numel() for p in model.parameters())
    trn   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Modelo listo — params totales {total:,} | "
          f"entrenables P1 {trn:,} ({100*trn/total:.1f}%)")

    history    = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_mAP': []}
    best_f1    = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    t_total = time.time()

    # ── Fase 1: solo cabeza ─────────────────────────────────
    print("\n" + "═" * 60)
    print(f"🚀 FASE 1 — Backbone congelado | LR={C.LR_PHASE1} | "
          f"Épocas={C.NUM_EPOCHS_P1}")
    print("═" * 60)
    t_p1 = time.time()
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.LR_PHASE1, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode='max', factor=0.5, patience=2)
    best_f1, best_state, p1_run = run_phase(
        'P1', model, train_loader, val_loader, criterion, opt1, sch1,
        C.NUM_EPOCHS_P1, classes, history, best_f1, best_state,
        step_with_metric=True, scaler=scaler)
    t_p1 = time.time() - t_p1
    print(f"🏁 Mejor F1 Fase 1: {best_f1:.4f}  |  Tiempo Fase 1: {t_p1/60:.1f} min")

    # ── Fase 2: fine-tuning ─────────────────────────────────
    print("\n" + "═" * 60)
    print(f"🔥 FASE 2 — Fine-tuning últimos {C.UNFREEZE_BLOCKS} bloques | "
          f"LR={C.LR_PHASE2} | Épocas={C.NUM_EPOCHS_P2} "
          f"(early stop: {C.EARLY_STOP_PATIENCE})")
    print("═" * 60)
    t_p2 = time.time()
    model.load_state_dict(best_state)
    model.unfreeze_last_n(n=C.UNFREEZE_BLOCKS)
    opt2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.LR_PHASE2, weight_decay=1e-5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=C.NUM_EPOCHS_P2)
    best_f1, best_state, _ = run_phase(
        'P2', model, train_loader, val_loader, criterion, opt2, sch2,
        C.NUM_EPOCHS_P2, classes, history, best_f1, best_state,
        step_with_metric=False, scaler=scaler,
        patience=C.EARLY_STOP_PATIENCE)
    t_p2 = time.time() - t_p2
    t_total = time.time() - t_total
    print(f"⏱️  Tiempo Fase 2: {t_p2/60:.1f} min")

    # ── Umbrales por clase ──────────────────────────────────
    model.load_state_dict(best_state)
    _, _, _, y_true, y_logits = eval_epoch(model, val_loader, criterion)
    thresholds = tune_thresholds(y_true, y_logits)
    f1_tuned, mAP_final = compute_metrics(y_true, y_logits, thresholds)
    print(f"\n🎯 Umbrales por clase: "
          f"{[f'{c}={t}' for c, t in zip(classes, thresholds)]}")
    print(f"   F1 con umbral 0.5: {best_f1:.4f} → "
          f"F1 con umbrales ajustados: {f1_tuned:.4f}")
    final_f1 = max(best_f1, f1_tuned)
    save_checkpoint(best_state, classes, len(history['val_f1']), final_f1,
                    'final', history, thresholds=thresholds)

    # ── Resultados ──────────────────────────────────────────
    plot_history(history, classes, len(all_samples), final_f1, p1_run)

    print("\n" + "═" * 60)
    print("📊 RESUMEN")
    print("═" * 60)
    print(f"Clases          : {len(classes)} → {classes}")
    print(f"Imágenes        : {len(all_samples):,}")
    print(f"Mejor F1 macro  : {final_f1:.4f}  (mAP {mAP_final:.4f})")
    print(f"Modelo guardado : {C.SAVE_PATH}")
    print(f"⏱️  Tiempo total  : {t_total/60:.1f} min  "
          f"(F1: {t_p1/60:.1f} min  |  F2: {t_p2/60:.1f} min)")
    if final_f1 >= 0.85:
        print("✅✅✅✅ META ALCANZADA — F1 ≥ 0.85.")
    elif final_f1 >= 0.75:
        print("✅✅✅  EXCELENTE — Arquitectura validada.")
    elif final_f1 >= 0.55:
        print("✅✅    BIEN.")
    elif final_f1 >= 0.40:
        print("✅      ACEPTABLE.")
    else:
        print("⚠️     REVISAR — verifica que las carpetas no estén mezcladas.")


if __name__ == '__main__':
    main()
