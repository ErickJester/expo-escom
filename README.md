# ExpoEscom — Clasificador de Caricaturas/Anime (versión LOCAL)

Clasificador de imágenes que reconoce a qué caricatura/anime pertenece una
imagen (Pokemon, Naruto, Dragon Ball, etc.) más una clase `otra` para
imágenes que no son caricaturas.

Originalmente corría en **Google Colab (GPU T4)**. Esta versión está adaptada
para correr **local** en una **RTX 2070 Super (8 GB VRAM)**.

## Arquitectura
- **Backbone:** MobileNetV2 pre-entrenado en ImageNet
- **Cabeza:** Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→N_clases)
- **Pérdida:** BCEWithLogitsLoss (multi-etiqueta)
- **Entrenamiento en 2 fases:** (1) solo cabeza, (2) fine-tuning de las
  últimas 3 capas del backbone.

## Estructura
```
ExpoEscom/
├── config.py          # rutas e hiperparámetros (edita aquí)
├── dataset_utils.py   # Dataset y construcción de muestras
├── model.py           # arquitectura CartoonClassifier
├── train.py           # entrenamiento (genera models/mini_poc.pt)
├── predict.py         # inferencia sobre imágenes nuevas
├── requirements.txt
├── dataset/           # una subcarpeta por clase
│   ├── pokemon/
│   ├── naruto/
│   ├── ...
│   └── otra/          # con subcarpetas (cartoons_anime, noise, real_life)
└── models/            # checkpoints y gráficas
```

## Instalación
Requiere drivers NVIDIA con CUDA. En WSL2 / Linux:

```bash
# 1) (recomendado) entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# 2) PyTorch con CUDA (ajusta cuXXX a tu driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) el resto
pip install -r requirements.txt
```

Verifica que la GPU se detecte:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Entrenar
```bash
python train.py
```
Guarda el mejor modelo en `models/mini_poc.pt` y la gráfica en
`models/mini_poc_grafica.png`. En la 2070 Super son ~12 épocas en pocos
minutos.

## Predecir
```bash
python predict.py ruta/a/imagen.jpg
python predict.py img1.jpg img2.png img3.webp
```

## Configuración
Todo lo ajustable está en `config.py`: rutas, `BATCH_SIZE`, número de épocas,
learning rates, `MAX_PER_CLASS`, `NUM_WORKERS`, etc.

> **Nota WSL2:** si `NUM_WORKERS > 0` da errores de memoria compartida,
> ponlo en `0` en `config.py`.
