# ExpoEscom — Entrenamiento LOCAL v3.0

Versión **local** (scripts `.py`) del notebook `03_entrenamiento_v3.ipynb`.
Misma estrategia de **feature caching**, pensada para correr en tu PC con
**RTX 2070 Super (8 GB)** en Windows — sin Colab, sin Drive, sin ZIP.

## Por qué scripts y no notebook

En Windows, un notebook con `num_workers > 0` suele romper por el
`multiprocessing` de PyTorch. Como script con `if __name__ == "__main__"`,
los workers **sí** funcionan → la carga de imágenes va en paralelo y no
deja a la GPU esperando. Por eso esto es `.py` y no `.ipynb`.

## Estructura esperada del dataset

Una subcarpeta por clase (la de ruido puede llamarse `otra`, `noise` o `ruido`):

```
DATASET_ROOT/
├── pokemon/        *.jpg
├── naruto/         *.jpg
├── simpson/        *.jpg
├── ...
└── noise/          *.jpg        ← clase de ruido (se detecta sola)
```

El código **auto-detecta** toda subcarpeta con imágenes; no necesitas
renombrar nada. Soporta subcarpetas anidadas dentro de cada clase.

## Instalación

```powershell
# 1) PyTorch con CUDA (ajusta cuXXX a tu versión; cu121 va bien con la 2070S)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Resto
pip install -r requirements.txt

# 3) Verifica que la GPU se ve
python check_gpu.py
```

## Uso

1. Edita `config.py` → `DATASET_ROOT` (ruta a tus carpetas de imágenes).
2. Corre:

```powershell
python train.py
```

Eso hace, en orden:
1. **Extrae features** del backbone congelado (1 sola pasada) → cache en
   `models/features_cache.pt`.
2. **Fase 1:** entrena la cabeza sobre esos vectores (rápido).
3. **Fase 2:** fine-tuning de los últimos 3 bloques sobre un subconjunto
   balanceado (20k/clase).
4. Guarda el mejor modelo en `models/cartoon_v3_local.pt` + gráfica + F1 por clase.

## Reanudar

Si la corrida se corta **después** de extraer features, la próxima vez
detecta `features_cache.pt` y se salta la parte más lenta. Para re-extraer
desde cero, borra ese archivo.

## Tiempos estimados (1.27M imgs, RTX 2070 Super)

| Almacenamiento de las imágenes | Total aprox. |
|---|---|
| NVMe SSD   | ~35-55 min |
| SATA SSD   | ~55-90 min |
| HDD        | ~3-5 h (I/O aleatorio mata el rendimiento) |

La extracción de features es lo más pesado (es donde pega el disco).
Fase 1 son minutos; Fase 2 depende otra vez del disco.

## Ajustes (en `config.py`)

| Si...                          | Cambia |
|--------------------------------|--------|
| Da **OOM** de VRAM             | `BATCH_SIZE` 64 → 32 (y `BATCH_INFER` 256 → 128) |
| Workers dan error en Windows   | `NUM_WORKERS` → 0 (más lento, pero estable) |
| Quieres aún menos tiempo       | `FT_PER_CLASS` 20k → 10k, o `NUM_EPOCHS_P2` → 2 |
| Solo probar rápido (linear probe) | `NUM_EPOCHS_P2 = 0` (omite Fase 2) |

> Nota: TF32 (`allow_tf32`) solo acelera en GPUs Ampere+ (RTX 30xx en
> adelante). En la 2070 Super (Turing) no hace nada, pero AMP fp16 **sí**
> acelera y está activo.

## Archivos

- `config.py` — rutas, hiperparámetros, hardware.
- `dataset.py` — `CartoonDataset`, detección de clases, split capado.
- `model.py` — MobileNetV2 + cabeza (con `extract()` para el caching).
- `train.py` — pipeline completo (features → Fase 1 → Fase 2 → reporte).
- `check_gpu.py` — verificación rápida de CUDA/GPU.

## Changelog

- **v3.0-local** — traducción del notebook Colab v3.0 a scripts locales:
  feature caching, fine-tuning sobre subconjunto, validación capada,
  AMP/channels_last, detección automática de clases, workers en Windows.
