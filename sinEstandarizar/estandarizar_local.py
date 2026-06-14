"""
Estandarización local — GENÉRICA (cualquier clase) · versión FUSIÓN
ExpoEscom — Clasificador Multietiqueta
Versión 3.0.0

Equivalente LOCAL del notebook 01_estandarizar_pokemon_fusion_1.ipynb.
Toda la lógica del notebook, pero leyendo los archivos desde la carpeta
sinEstandarizar/ en vez de descargarlos de Google Drive.

En cada ejecución el programa pregunta:
  PASO 1 — La carpeta DESTINO dentro de dataset/ (la clase: poke, naruto, ...).
  PASO 2 — Lista TODOS los .zip / .rar de sinEstandarizar/ NUMERADOS y tú
           escribes los números separados por coma (ej.  1,3,4) para elegir
           cuáles estandarizar a la vez. También aceptas "todos".

Los archivos elegidos se tratan como UN solo trabajo con cuotas inteligentes
(las fuentes chicas entran completas, el resto se reparte entre las grandes),
y las imágenes se guardan en ../dataset/<clase>/ como <clase>_NNNNNN.jpg
(224x224 RGB JPG).

Características portadas del notebook fusión (v4.0.0):
  • Procesamiento PARALELO por fuente (ThreadPoolExecutor, MAX_WORKERS).
  • Filtro de variedad dHash THREAD-SAFE (descarta imágenes casi idénticas).
  • Contador global atómico → corte EXACTO en la meta (nunca de más).
  • Cuotas inteligentes con redistribución de déficit por rondas.
  • RARs anidados dentro de ZIPs (requiere unrar).
  • Checkpoint en la carpeta destino → resume automático tras un corte.
  • Garantía de meta EXACTA (FASE B):
       1) pase normal con umbral base,
       2) relaja el umbral dHash en pasos hasta UMBRAL_MAX y reintenta,
       3) último recurso: rellena con augmentación PARALELA (flip, rotación,
          brillo, contraste, zoom) hasta completar la meta.

Uso:
    python estandarizar_local.py --max 200000     # meta total (pregunta clase + archivos)
    python estandarizar_local.py --inventario     # solo lista fuentes y cuenta
    python estandarizar_local.py --max 200000 --filtro-existentes
    python estandarizar_local.py --max 200000 --workers 8 --umbral 8

Para los .rar (directos o anidados) necesitas unrar en el PATH:
    Windows : instala WinRAR (UnRAR.exe) o  `choco install unrar` / `winget install`
    Linux   : sudo apt-get install unrar
"""

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

# cv2 (OpenCV) decodifica/redimensiona JPEG 3-5x más rápido que PIL. Es
# opcional: si no está instalado, todo cae a PIL automáticamente (más lento
# pero idéntico resultado). Instala con: pip install opencv-python-headless
try:
    import cv2
    cv2.setNumThreads(1)   # 1 hilo interno: el paralelismo lo damos nosotros
    USAR_CV2 = True
except ImportError:
    USAR_CV2 = False

try:
    from tqdm import tqdm
except ImportError:  # fallback mínimo si tqdm no está instalado
    def tqdm(it=None, total=None, **kw):
        return it if it is not None else range(0)

# Silenciar el warning ruidoso de PIL (palette+transparency); convert('RGB') ya lo maneja
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# La consola de Windows suele venir en cp1252 y no puede imprimir los caracteres
# de caja/emoji que usa este programa. Forzamos UTF-8 en la salida.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Versión ──────────────────────────────────────────────────
# 1.x   → específico de pokemon.
# 2.x   → genérico (clase + un zip por entrada).
# 3.0.0 → FUSIÓN: paralelo + contador atómico + checkpoint + umbral
#         progresivo + augmentación, y selección MÚLTIPLE por números
#         separados por coma. Equivalente local del notebook fusión.
# 3.1.0 → FuenteCarpeta: soporte para carpetas ya extraídas (3-10x más
#         rápido que ZIP). Listado numerado muestra ZIPs y carpetas juntos.
# 3.2.0 → Augmentación condicional: solo se activa si las fuentes se
#         agotaron Y ya hay ≥ --min-augmentar imágenes reales (default 170k).
# 3.2.1 → FIX: FuenteCarpeta no se contaba (disponibles=0) → no extraía
#         nada. Ahora se escanean RAR y carpetas antes de asignar cuotas.
# 3.3.0 → FuenteCarpeta se divide en MAX_WORKERS shards al descubrirse,
#         así los 14 hilos trabajan en paralelo real (antes: 1 hilo solo).
# 3.4.0 → Optimización: backend cv2 (decode/resize/encode 3-5x más rápido
#         que PIL, con fallback automático a PIL), dHash vectorizado con
#         numpy (packbits) y downscale INTER_AREA/BILINEAR. Lectura de
#         archivos Unicode-safe en Windows (np.fromfile / buf.tofile).
# 3.5.0 → Por defecto YA NO indexa las imágenes existentes (arranque
#         directo). Usa --filtro-existentes para reactivar el indexado.
VERSION = '3.5.0'

# ── Configuración fija ───────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent / 'dataset'
IMG_SIZE     = (224, 224)
SEED         = 42
TMP_DIR      = SCRIPT_DIR / 'tmp_extract'
TMP_EXTRACT  = TMP_DIR / 'extract'

EXTENSIONES_IMAGEN = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
EXT_SOPORTADAS     = {'.zip', '.rar'}

HASH_SIZE        = 8    # dHash 8x8 → 64 bits
UMBRAL_SIMILITUD = 8    # distancia Hamming máxima para considerar "similar"

# ── Paralelismo y rondas ─────────────────────────────────────
# Ryzen 7 5700X: 8 cores / 16 threads.
# PIL libera el GIL en resize/convert (código C), así que los hilos
# corren en paralelo real. 14 workers deja 2 threads libres para el OS
# y el hilo principal sin saturar el scheduler.
MAX_WORKERS = min(14, (os.cpu_count() or 4))   # hilos de extracción simultáneos
LOTE_MULT   = 5    # 32 GB RAM → batches más grandes, menos llamadas a unrar
MAX_RONDAS  = 8    # rondas de redistribución de déficit por umbral

# ── Garantía de la meta EXACTA ───────────────────────────────
GARANTIZAR_EXACTO = True   # se puede desactivar con --no-exacto
UMBRAL_MAX        = 20     # tope al relajar el umbral (0 = idénticas permitidas)
UMBRAL_PASO       = 4      # cuánto sube el umbral en cada reintento
PERMITIR_AUGMENT  = True   # último recurso: rellenar con augmentación

# ── Estado global (se fija en main) ──────────────────────────
CLASE           = None
DESTINO_CLASE   = None
CHECKPOINT_PATH = None
MAX_IMAGENES    = 0


# =============================================================================
# UTILIDADES
# =============================================================================

def es_imagen(nombre):
    return Path(nombre).suffix.lower() in EXTENSIONES_IMAGEN


def hay_unrar():
    return shutil.which('unrar') is not None


# =============================================================================
# FILTRO DE VARIEDAD (dHash thread-safe)
# =============================================================================

# ── Backend de imagen: cv2 (rápido) o PIL (fallback) ─────────
# En modo cv2 una "imagen" es un ndarray BGR (HxWx3). En modo PIL es un
# Image. Estos 3 helpers ocultan la diferencia al resto del programa.

def leer_bytes(data):
    """Decodifica bytes de imagen → ndarray BGR (cv2) o Image (PIL). None si falla."""
    if USAR_CV2:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    try:
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return None


def leer_archivo(ruta):
    """Lee un archivo de disco → ndarray BGR (cv2) o Image (PIL). None si falla.
    np.fromfile/Image manejan rutas con caracteres Unicode en Windows."""
    if USAR_CV2:
        try:
            return cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
    try:
        return Image.open(ruta).convert('RGB')
    except Exception:
        return None


def guardar_estandarizada(img, dest):
    """Redimensiona a 224x224 y guarda JPEG q90. INTER_AREA al reducir
    (mejor para downscale), INTER_CUBIC al ampliar."""
    if USAR_CV2:
        h, w = img.shape[:2]
        interp = (cv2.INTER_AREA if (w >= IMG_SIZE[0] and h >= IMG_SIZE[1])
                  else cv2.INTER_CUBIC)
        out = cv2.resize(img, IMG_SIZE, interpolation=interp)
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise IOError('cv2.imencode falló')
        buf.tofile(str(dest))   # tofile maneja rutas Unicode en Windows
    else:
        img.resize(IMG_SIZE, Image.LANCZOS).save(
            dest, 'JPEG', quality=90, optimize=True)


def dhash(img):
    """Hash perceptual de 64 bits (gradiente horizontal en rejilla 8x8),
    vectorizado con numpy. Acepta ndarray BGR (cv2) o Image (PIL)."""
    if USAR_CV2:
        gris  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        chico = cv2.resize(gris, (HASH_SIZE + 1, HASH_SIZE),
                           interpolation=cv2.INTER_AREA)
    else:
        # BILINEAR basta para 9x8 px (la calidad no afecta el hash) y es rápido
        g = img.convert('L').resize((HASH_SIZE + 1, HASH_SIZE), Image.BILINEAR)
        chico = np.asarray(g, dtype=np.uint8)
    diff = chico[:, 1:] > chico[:, :-1]                  # (8,8) booleano
    return int.from_bytes(np.packbits(diff.ravel()).tobytes(), 'big')


class IndiceSimilitud:
    """
    Índice de hashes perceptuales THREAD-SAFE para descartar imágenes muy
    parecidas, aun cuando varios hilos procesan fuentes en paralelo.

    Cada hash se indexa por sus 4 bloques de 16 bits: dos imágenes similares
    comparten casi siempre al menos un bloque exacto, así cada consulta
    compara contra un puñado de candidatos en vez de contra todo el dataset.
    """

    def __init__(self, umbral=UMBRAL_SIMILITUD):
        self.umbral   = umbral
        self._buckets = [{}, {}, {}, {}]
        self._lock    = threading.Lock()

    @staticmethod
    def _bloques(h):
        return ((h >> 48) & 0xFFFF, (h >> 32) & 0xFFFF,
                (h >> 16) & 0xFFFF, h & 0xFFFF)

    def _es_similar_unsafe(self, h):
        vistos = set()
        for i, b in enumerate(self._bloques(h)):
            for otro in self._buckets[i].get(b, ()):
                if otro in vistos:
                    continue
                vistos.add(otro)
                if bin(h ^ otro).count('1') <= self.umbral:
                    return True
        return False

    def _agregar_unsafe(self, h):
        for i, b in enumerate(self._bloques(h)):
            self._buckets[i].setdefault(b, []).append(h)

    def verificar_y_agregar(self, h):
        """ATÓMICO: si h no es similar a ninguno, lo agrega y devuelve True
        (la imagen pasa). Si es similar, devuelve False (descartar)."""
        with self._lock:
            if self._es_similar_unsafe(h):
                return False
            self._agregar_unsafe(h)
            return True

    def agregar(self, h):
        with self._lock:
            self._agregar_unsafe(h)

    def set_umbral(self, nuevo):
        """Relaja (o endurece) el umbral de similitud en caliente."""
        with self._lock:
            self.umbral = nuevo


def indexar_existentes(indice):
    """Hashea lo que ya está en destino para que las imágenes nuevas tampoco
    repitan lo ya guardado (resume coherente)."""
    rutas = list(DESTINO_CLASE.glob('*.jpg'))
    if not rutas:
        print('No hay imágenes previas que indexar.')
        return
    print(f'Indexando {len(rutas):,} imágenes existentes…')
    t0 = time.time()
    for ruta in tqdm(rutas, unit='img'):
        img = leer_archivo(ruta)
        if img is not None:
            indice.agregar(dhash(img))
    print(f'Listo en {(time.time()-t0)/60:.1f} min')


# =============================================================================
# CONTADORES GLOBALES THREAD-SAFE (nombres sin colisión + corte EXACTO)
# =============================================================================

_lock_idx   = threading.Lock()
_idx_global = [0]

def _siguiente_indice():
    with _lock_idx:
        i = _idx_global[0]
        _idx_global[0] += 1
    return i


_lock_total      = threading.Lock()
_total_guardadas = [0]   # imágenes reales en disco (incluye las previas)

def _reservar_cupo():
    """Reserva atómicamente un espacio. True si aún hay cupo hacia la meta."""
    with _lock_total:
        if _total_guardadas[0] >= MAX_IMAGENES:
            return False
        _total_guardadas[0] += 1
        return True

def _liberar_cupo():
    """Devuelve un cupo si la imagen reservada no se pudo guardar (error)."""
    with _lock_total:
        _total_guardadas[0] -= 1

def _cupo_restante():
    with _lock_total:
        return max(0, MAX_IMAGENES - _total_guardadas[0])

def _inicializar_indice():
    """Detecta el mayor índice ya usado y cuántas hay para no sobreescribir."""
    maxidx = -1
    n_existentes = 0
    for f in DESTINO_CLASE.glob(f'{CLASE}_*.jpg'):
        n_existentes += 1
        try:
            n = int(f.stem[len(CLASE) + 1:])
            maxidx = max(maxidx, n)
        except ValueError:
            pass
    _idx_global[0]      = maxidx + 1
    _total_guardadas[0] = n_existentes


def procesar_imagen(img, indice, pbar):
    """Filtro de variedad + corte exacto + estandarización + guardado.
    Devuelve True si se guardó, False si se descartó (similar o sin cupo)."""
    # 1) Filtro de variedad
    if not indice.verificar_y_agregar(dhash(img)):
        return False
    # 2) Reservar cupo: corta EXACTAMENTE en MAX_IMAGENES
    if not _reservar_cupo():
        return False
    # 3) Estandarizar y guardar
    try:
        idx  = _siguiente_indice()
        dest = DESTINO_CLASE / f'{CLASE}_{idx:06d}.jpg'
        guardar_estandarizada(img, dest)
        if pbar is not None:
            pbar.update(1)
        return True
    except Exception:
        _liberar_cupo()
        return False


# =============================================================================
# FUENTES (ZIP, RAR y RARs anidados)
# =============================================================================

class FuenteZip:
    """Imágenes directas dentro de un ZIP (se leen en memoria, sin extraer)."""

    def __init__(self, nombre, zip_path, nombres_dentro):
        self.nombre      = nombre
        self.zip_path    = Path(zip_path)
        self._nombres    = nombres_dentro
        self.disponibles = len(nombres_dentro)
        self.cuota       = 0
        self.guardadas   = 0
        self.agotada     = False

    def reabrir(self):
        """Reactiva la fuente para un nuevo pase (umbral relajado)."""
        self.agotada = False

    def extraer(self, indice, pbar):
        nombres = self._nombres[:]
        random.shuffle(nombres)
        nuevas = descartadas = 0
        try:
            with zipfile.ZipFile(self.zip_path) as zf:
                for nombre in nombres:
                    if self.guardadas >= self.cuota:
                        break
                    try:
                        with zf.open(nombre) as f:
                            img = leer_bytes(f.read())
                        if img is None:
                            continue
                        if procesar_imagen(img, indice, pbar):
                            nuevas += 1
                            self.guardadas += 1
                        else:
                            descartadas += 1
                    except Exception:
                        continue
                else:
                    self.agotada = True   # recorrió todo sin llenar la cuota
        except Exception as e:
            print(f'  ⚠️ {self.nombre}: {e}', flush=True)
        return nuevas, descartadas


class FuenteRar:
    """RAR directo en disco, o RAR anidado dentro de un ZIP."""

    def __init__(self, nombre, rar_path=None, zip_origen=None, rar_interno=None):
        self.nombre      = nombre
        self.rar_path    = Path(rar_path) if rar_path else None
        self.zip_origen  = Path(zip_origen) if zip_origen else None
        self.rar_interno = rar_interno
        self.disponibles = -1
        self.cuota       = 0
        self.guardadas   = 0
        self.agotada     = False
        self._nombres    = None
        self._pendientes = None   # nombres aún no intentados (persiste entre rondas)

    def reabrir(self):
        """Reactiva la fuente: repuebla pendientes con TODA la lista, así
        reintenta las que antes se descartaron (ahora con umbral mayor)."""
        self.agotada     = False
        self._pendientes = None

    def _asegurar_en_disco(self):
        if self.rar_path and self.rar_path.exists():
            return True
        if not self.zip_origen:
            return False
        TMP_EXTRACT.mkdir(parents=True, exist_ok=True)
        dest = TMP_EXTRACT / Path(self.rar_interno).name
        if not dest.exists():
            print(f'  → Extrayendo {dest.name} del ZIP…', flush=True)
            with zipfile.ZipFile(self.zip_origen) as zf:
                with zf.open(self.rar_interno) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
        self.rar_path = dest
        return True

    def listar(self):
        if self._nombres is not None:
            return self._nombres
        if not self._asegurar_en_disco():
            return []
        r = subprocess.run(['unrar', 'lb', str(self.rar_path)],
                           capture_output=True, text=True, timeout=600)
        self._nombres = [l.strip() for l in r.stdout.splitlines()
                         if l.strip() and es_imagen(l.strip())]
        return self._nombres

    def contar(self):
        self.disponibles = len(self.listar())
        return self.disponibles

    def extraer(self, indice, pbar):
        if self._pendientes is None:
            self._pendientes = self.listar()[:]
            random.shuffle(self._pendientes)

        nuevas = descartadas = 0
        tmp = TMP_EXTRACT / f'batch_{self.nombre}_{threading.get_ident()}'

        # Lotes: si el filtro descarta muchas, pide otro lote
        while self.guardadas < self.cuota and self._pendientes:
            faltan = self.cuota - self.guardadas
            lote, self._pendientes = (self._pendientes[:faltan * LOTE_MULT],
                                      self._pendientes[faltan * LOTE_MULT:])
            tmp.mkdir(parents=True, exist_ok=True)
            listfile = TMP_EXTRACT / f'_list_{self.nombre}_{threading.get_ident()}.txt'
            listfile.write_text('\n'.join(lote))
            try:
                subprocess.run(
                    ['unrar', 'x', '-o+', str(self.rar_path),
                     f'@{listfile}', str(tmp) + '/'],
                    capture_output=True, timeout=14400)
            finally:
                listfile.unlink(missing_ok=True)

            rutas = [r for r in tmp.rglob('*') if r.is_file() and es_imagen(r.name)]
            random.shuffle(rutas)
            for ruta in rutas:
                if self.guardadas >= self.cuota:
                    break
                img = leer_archivo(ruta)
                if img is None:
                    continue
                if procesar_imagen(img, indice, pbar):
                    nuevas += 1
                    self.guardadas += 1
                else:
                    descartadas += 1
            shutil.rmtree(tmp, ignore_errors=True)

        if not self._pendientes:
            self.agotada = True
        return nuevas, descartadas


class FuenteCarpeta:
    """
    Shard de imágenes directas en disco (ext4 / NTFS).

    descubrir_fuentes() divide automáticamente la carpeta en MAX_WORKERS
    shards, uno por hilo, para que todos los workers corran en paralelo.
    Sin sharding solo usaría 1 hilo de 14.
    """

    def __init__(self, nombre, rutas_shard):
        self.nombre      = nombre
        self._rutas      = list(rutas_shard)
        self.disponibles = len(self._rutas)
        self.cuota       = 0
        self.guardadas   = 0
        self.agotada     = False

    def reabrir(self):
        self.agotada = False

    def listar(self):
        return self._rutas

    def contar(self):
        return self.disponibles

    def extraer(self, indice, pbar):
        rutas = self._rutas[:]
        random.shuffle(rutas)
        nuevas = descartadas = 0
        for ruta in rutas:
            if self.guardadas >= self.cuota:
                break
            img = leer_archivo(ruta)
            if img is None:
                continue
            if procesar_imagen(img, indice, pbar):
                nuevas += 1
                self.guardadas += 1
            else:
                descartadas += 1
        else:
            self.agotada = True
        return nuevas, descartadas


def descubrir_fuentes(ruta_local):
    """
    Convierte un archivo o carpeta en una o más fuentes:
      carpeta           → N FuenteCarpeta (1 shard por worker, paralelo real)
      .rar              → 1 FuenteRar
      .zip con imágenes → 1 FuenteZip
      .zip con RARs     → 1 FuenteRar por cada RAR anidado
      (un ZIP puede aportar ambas)
    """
    if ruta_local.is_dir():
        print(f'  Escaneando {ruta_local.name}…', end='', flush=True)
        todas = sorted(p for p in ruta_local.rglob('*')
                       if p.is_file() and es_imagen(p.name))
        print(f' {len(todas):,} imágenes')
        if not todas:
            return []
        n      = min(MAX_WORKERS, len(todas))
        size   = -(-len(todas) // n)   # ceil
        shards = [todas[i * size:(i + 1) * size] for i in range(n)
                  if todas[i * size:(i + 1) * size]]
        return [FuenteCarpeta(f'{ruta_local.name} [{i+1}/{len(shards)}]', s)
                for i, s in enumerate(shards)]

    ext = ruta_local.suffix.lower()
    if ext == '.rar':
        return [FuenteRar(ruta_local.stem, rar_path=ruta_local)]

    fuentes = []
    with zipfile.ZipFile(ruta_local) as zf:
        entradas = [n for n in zf.namelist() if '__MACOSX' not in n]
    imgs = [n for n in entradas if es_imagen(n)]
    rars = [n for n in entradas if Path(n).suffix.lower() == '.rar']
    if imgs:
        fuentes.append(FuenteZip(ruta_local.stem, ruta_local, imgs))
    for rar in rars:
        fuentes.append(FuenteRar(Path(rar).stem,
                                 zip_origen=ruta_local, rar_interno=rar))
    return fuentes


# =============================================================================
# CHECKPOINT Y CUOTAS
# =============================================================================

def cargar_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def guardar_checkpoint(ckpt):
    tmp = str(CHECKPOINT_PATH) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, str(CHECKPOINT_PATH))


def asignar_cuotas(fuentes, total):
    """
    De menor a mayor fuente: si una cabe completa en su parte proporcional,
    se toma TODA y su sobrante se reparte entre las más grandes. Maximiza la
    variedad del dataset.
    """
    activas = sorted([f for f in fuentes if f.disponibles > 0 and not f.agotada],
                     key=lambda f: f.disponibles)
    restante = total
    for i, f in enumerate(activas):
        n_restantes = len(activas) - i
        cuota_ideal = -(-restante // n_restantes)   # ceil
        f.cuota   = f.guardadas + min(cuota_ideal, f.disponibles - f.guardadas)
        restante -= (f.cuota - f.guardadas)


# =============================================================================
# ENTRADA INTERACTIVA
# =============================================================================

def pedir_clase():
    """Pregunta la carpeta destino dentro de dataset/ (se crea si no existe)."""
    existentes = sorted(p.name for p in DATASET_ROOT.iterdir()
                        if p.is_dir()) if DATASET_ROOT.is_dir() else []
    print()
    print('PASO 1 — Carpeta destino')
    print('  Es la CLASE donde se guardarán las imágenes, dentro de dataset/.')
    print('  Escribe solo el nombre (ej. "poke"), sin barras ni rutas.')
    print('  Si la carpeta no existe, se crea automáticamente.')
    if existentes:
        print(f'  Carpetas que ya existen: {", ".join(existentes)}')
    while True:
        clase = (input('➡️  Carpeta destino: ')
                 .replace('﻿', '').strip().strip('/').strip('\\'))
        if not clase:
            print('   ⚠️  No escribiste nada. Intenta de nuevo (ej. poke).')
            continue
        if '/' in clase or '\\' in clase:
            print('   ⚠️  Escribe solo el nombre de la carpeta, sin rutas ni barras.')
            continue
        return clase


def _fmt_tam(mb):
    return f'{mb/1024:.1f} GB' if mb >= 1024 else f'{mb:.1f} MB'


_DIRS_IGNORADAS = {'tmp_extract', '__pycache__', '.git'}


def seleccionar_archivos():
    """Lista NUMERADOS los .zip/.rar Y carpetas de sinEstandarizar/ y deja
    elegir varios escribiendo los números separados por coma o "todos".

    Las carpetas con imágenes ya extraídas son más rápidas que los ZIPs
    porque cada hilo lee sus propios archivos sin competir por un único
    archivo de 13 GB.
    """
    archivos = sorted(
        [p for p in SCRIPT_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in EXT_SOPORTADAS],
        key=lambda p: p.name.lower())

    carpetas = sorted(
        [p for p in SCRIPT_DIR.iterdir()
         if p.is_dir() and p.name not in _DIRS_IGNORADAS
         and not p.name.startswith('.')],
        key=lambda p: p.name.lower())

    opciones = archivos + carpetas   # archivos primero, carpetas al final

    print()
    print('PASO 2 — Fuentes a estandarizar')
    if not opciones:
        print('  ⚠️  No hay ningún .zip, .rar ni carpeta en sinEstandarizar/.')
        print('     Copia primero tus archivos ahí y vuelve a ejecutar.')
        sys.exit(1)

    print('  Fuentes disponibles en sinEstandarizar/:')
    print()
    print(f"  {'#':>3}  {'Nombre':<42} {'Tamaño':>10}  {'Tipo'}")
    print('  ' + '─' * 72)
    for i, p in enumerate(opciones, 1):
        if p.is_dir():
            tam  = '—'
            tipo = '📁 carpeta  ← más rápido'
        else:
            mb   = p.stat().st_size / (1024 * 1024)
            tam  = _fmt_tam(mb)
            aviso = '  (necesita unrar)' if (p.suffix.lower() == '.rar' and not hay_unrar()) else ''
            tipo = f'📦 {p.suffix.lower()}{aviso}'
        print(f"  {i:>3}. {p.name:<42} {tam:>10}  {tipo}")
    print()
    print('  Escribe los NÚMEROS separados por coma (ej.  1,3,4).')
    print('  O escribe  "todos"  para seleccionar todos.')
    if carpetas:
        print('  TIP: las carpetas son 3-10x más rápidas que los ZIPs.')

    while True:
        sel = input('➡️  Fuentes a estandarizar: ').replace('﻿', '').strip().lower()
        if not sel:
            print('   ⚠️  No escribiste nada. Intenta de nuevo (ej. 1,3,4).')
            continue
        if sel in ('todos', 'todo', 'all', '*'):
            return opciones
        try:
            idxs = [int(x) for x in sel.replace(' ', '').split(',') if x]
        except ValueError:
            print('   ⚠️  Usa solo números separados por coma (ej. 1,3,4).')
            continue
        fuera = sorted({n for n in idxs if n < 1 or n > len(opciones)})
        if fuera:
            print(f'   ⚠️  Fuera de rango: {fuera}. Válidos: 1..{len(opciones)}.')
            continue
        vistos, elegidos = set(), []
        for n in idxs:
            if n not in vistos:
                vistos.add(n)
                elegidos.append(opciones[n - 1])
        if not elegidos:
            print('   ⚠️  No seleccionaste ningún número válido.')
            continue
        return elegidos


# =============================================================================
# PIPELINE (rondas paralelas con redistribución de déficit)
# =============================================================================

def correr_rondas(fuentes, indice, pbar, checkpoint, estado_global):
    """Ejecuta hasta MAX_RONDAS rondas paralelas con el umbral actual.
    Para en cuanto se llena el cupo global o no quedan fuentes."""
    ronda = 0
    while ronda < MAX_RONDAS:
        ronda += 1
        if _cupo_restante() <= 0:
            break
        disponibles = [f for f in fuentes
                       if not f.agotada and f.guardadas < f.disponibles]
        if not disponibles:
            break

        asignar_cuotas(fuentes, MAX_IMAGENES)
        activas = [f for f in disponibles if f.cuota > f.guardadas]
        if not activas:
            break

        print(f'\n── RONDA {ronda} ──  faltan {_cupo_restante():,} imgs, '
              f'{len(activas)} fuentes activas')

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(f.extraer, indice, pbar): f for f in activas}
            for fut in as_completed(futs):
                f = futs[fut]
                try:
                    nuevas, desc = fut.result()
                    estado_global['descartadas'] += desc
                    checkpoint[f.nombre] = f.guardadas
                    guardar_checkpoint(checkpoint)
                    print(f'   ✅ {f.nombre}: +{nuevas:,} '
                          f'({desc:,} descartadas por similitud)', flush=True)
                except Exception as e:
                    print(f'   ❌ {f.nombre}: {e}', flush=True)


def rellenar_augmentando(pbar, checkpoint):
    """Último recurso: genera imágenes augmentadas EN PARALELO a partir de las
    ya guardadas hasta completar la meta. Devuelve cuántas generó."""
    objetivo_aug = _cupo_restante()
    print('\n' + '·' * 55)
    print(f'🎨 Las fuentes se agotaron — generando {objetivo_aug:,} imágenes '
          f'AUGMENTADAS ({MAX_WORKERS} hilos) para completar {MAX_IMAGENES:,}')
    print('·' * 55)

    base = sorted(DESTINO_CLASE.glob(f'{CLASE}_*.jpg'))
    if not base:
        print('   ⚠️ No hay imágenes base para augmentar; meta no alcanzable.')
        return 0

    def augmentar(img):
        op = random.choice(['flip', 'rot', 'bright', 'contrast', 'zoom', 'combo'])
        if op in ('flip', 'combo'):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if op in ('rot', 'combo'):
            img = img.rotate(random.uniform(-18, 18),
                             resample=Image.BILINEAR, expand=False)
        if op in ('bright', 'combo'):
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
        if op in ('contrast', 'combo'):
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
        if op in ('zoom', 'combo'):
            w, h = img.size
            fz = random.uniform(0.8, 0.95)
            cw, ch = int(w * fz), int(h * fz)
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.LANCZOS)
        return img

    def _aug_one(_):
        if not _reservar_cupo():
            return False
        try:
            origen = random.choice(base)
            with Image.open(origen) as im:
                aug = augmentar(im.convert('RGB'))
            idx  = _siguiente_indice()
            dest = DESTINO_CLASE / f'{CLASE}_{idx:06d}.jpg'
            aug.resize(IMG_SIZE, Image.LANCZOS).save(
                dest, 'JPEG', quality=90, optimize=True)
            return True
        except Exception:
            _liberar_cupo()
            return False

    n_aug = 0
    with tqdm(total=objetivo_aug, desc='Augmentando', unit='img') as pbar_aug:
        while _cupo_restante() > 0:
            antes = n_aug
            faltan_aug = _cupo_restante()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for ok in ex.map(_aug_one, range(faltan_aug)):
                    if ok:
                        n_aug += 1
                        if pbar is not None:
                            pbar.update(1)
                        pbar_aug.update(1)
            if n_aug == antes:
                print('   ⚠️ No se pudo augmentar más (imágenes base con errores).')
                break

    checkpoint['_augmentadas'] = checkpoint.get('_augmentadas', 0) + n_aug
    guardar_checkpoint(checkpoint)
    return n_aug


# =============================================================================
# MAIN
# =============================================================================

def main():
    global CLASE, DESTINO_CLASE, CHECKPOINT_PATH, MAX_IMAGENES
    global MAX_WORKERS, GARANTIZAR_EXACTO, PERMITIR_AUGMENT

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max', type=int, default=500,
                        help='Total de imágenes objetivo en dataset/<clase> '
                             '(incluye las que ya existen). Default: 500')
    parser.add_argument('--inventario', action='store_true',
                        help='Solo lista fuentes y cuenta imágenes, sin extraer')
    parser.add_argument('--umbral', type=int, default=UMBRAL_SIMILITUD,
                        help='Distancia Hamming máxima del dHash para descartar '
                             f'por similitud. Default: {UMBRAL_SIMILITUD}. '
                             'Más alto = acepta más parecidas')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Hilos de extracción en paralelo. Default: {MAX_WORKERS}')
    parser.add_argument('--filtro-existentes', action='store_true',
                        help='Indexa las imágenes ya guardadas en destino para '
                             'que las nuevas no las repitan. Por defecto NO se '
                             'indexa (arranque directo, las nuevas solo se '
                             'comparan entre sí)')
    parser.add_argument('--no-exacto', action='store_true',
                        help='No fuerza la meta exacta (sin umbral progresivo '
                             'ni augmentación de relleno)')
    parser.add_argument('--sin-augmentar', action='store_true',
                        help='Permite relajar el umbral pero NO rellena con '
                             'imágenes augmentadas')
    parser.add_argument('--min-augmentar', type=int, default=170_000,
                        help='Mínimo de imágenes guardadas para que la '
                             'augmentación de relleno se active. '
                             'Default: 170000')
    args = parser.parse_args()

    random.seed(SEED)
    MAX_WORKERS       = max(1, args.workers)
    MAX_IMAGENES      = args.max
    GARANTIZAR_EXACTO = not args.no_exacto
    PERMITIR_AUGMENT  = not args.sin_augmentar

    # ── Encabezado ───────────────────────────────────────────
    print('═' * 62)
    print(f'  ESTANDARIZACIÓN DE IMÁGENES — ExpoEscom')
    print(f'  versión {VERSION}  |  fusión local  |  {Path(__file__).resolve()}')
    print('═' * 62)
    print('Instrucciones:')
    print('  1. Coloca tus .zip / .rar dentro de la carpeta sinEstandarizar/')
    print('     (pueden contener imágenes directas o RARs anidados).')
    print('  2. Indica la carpeta DESTINO dentro de dataset/ (la clase).')
    print('  3. Elige los archivos a procesar por NÚMERO, separados por coma.')
    print('  4. Las imágenes se guardan como <clase>_NNNNNN.jpg (224x224 RGB).')
    backend = 'cv2 (rápido)' if USAR_CV2 else 'PIL (instala opencv-python-headless para 3-5x)'
    print(f'  • Meta (--max): {MAX_IMAGENES:,}   ·   Filtro (--umbral): {args.umbral}'
          f'   ·   Workers: {MAX_WORKERS}')
    print(f'  • Backend de imagen: {backend}')
    print('─' * 62)

    # ── PASO 1: clase destino ────────────────────────────────
    CLASE           = pedir_clase()
    DESTINO_CLASE   = DATASET_ROOT / CLASE
    CHECKPOINT_PATH = DESTINO_CLASE / '_checkpoint.json'
    DESTINO_CLASE.mkdir(parents=True, exist_ok=True)
    _inicializar_indice()

    # ── PASO 2: elegir archivos (números separados por coma) ─
    elegidos = seleccionar_archivos()

    print(f'\nClase / destino : {DESTINO_CLASE}')
    print('Archivos        :')
    for p in elegidos:
        print(f'   • {p.name}')

    ya_tenemos = len(list(DESTINO_CLASE.glob(f'{CLASE}_*.jpg')))
    print(f'Ya tenemos      : {ya_tenemos:,}')

    # ── Descubrir fuentes de TODOS los archivos elegidos ─────
    fuentes = []
    for p in elegidos:
        try:
            fuentes.extend(descubrir_fuentes(p))
        except Exception as e:
            print(f'   ⚠️ No se pudo leer {p.name}: {e}')
    if not fuentes:
        sys.exit('❌ Los archivos elegidos no contienen imágenes ni RARs.')

    # ── Filtrar RARs si no hay unrar ─────────────────────────
    if not hay_unrar():
        rar = [f for f in fuentes if isinstance(f, FuenteRar)]
        if rar:
            print(f'\n⚠️  {len(rar)} fuente(s) RAR ignoradas (unrar no instalado):')
            for f in rar:
                print(f'   - {f.nombre}')
            print('   En Windows instala WinRAR (UnRAR.exe) o usa choco/winget.')
            fuentes = [f for f in fuentes if not isinstance(f, FuenteRar)]
            if not fuentes:
                sys.exit('❌ Solo había fuentes RAR y no hay unrar disponible.')

    # ── Contar fuentes que requieren escaneo (RAR y carpetas) ──
    # FuenteZip ya sabe su conteo al construirse; FuenteRar y FuenteCarpeta
    # arrancan con disponibles<=0 y deben escanearse ANTES de asignar cuotas.
    por_contar = [f for f in fuentes if isinstance(f, (FuenteRar, FuenteCarpeta))]
    if por_contar:
        print(f'\nContando imágenes en {len(por_contar)} fuente(s)…')
        for f in por_contar:
            print(f'  {f.nombre}…', end='', flush=True)
            print(f' {f.contar():,}')

    fuentes.sort(key=lambda f: f.disponibles)
    print('\n' + '─' * 55)
    print('FUENTES DETECTADAS (menor → mayor)')
    print('─' * 55)
    for f in fuentes:
        print(f'  {f.nombre:<40s} {f.disponibles:>9,} imgs')

    if args.inventario:
        total_disp = sum(f.disponibles for f in fuentes if f.disponibles > 0)
        print(f'\nTotal disponible en fuentes elegidas: {total_disp:,}')
        return

    # ── Resume desde checkpoint ──────────────────────────────
    checkpoint = cargar_checkpoint()
    for f in fuentes:
        f.guardadas = min(checkpoint.get(f.nombre, 0), max(0, f.disponibles))

    faltan = _cupo_restante()
    print(f'\nMeta    : {MAX_IMAGENES:,}')
    print(f'Faltan  : {faltan:,}')
    if faltan == 0:
        print('✅ Ya hay suficientes imágenes en destino. Nada que hacer.')
        return

    # ── Filtro de similitud ──────────────────────────────────
    indice = IndiceSimilitud(umbral=args.umbral)
    print(f'\nFiltro de variedad: dHash, umbral Hamming ≤ {args.umbral}')
    if args.filtro_existentes:
        indexar_existentes(indice)
    else:
        print('Sin indexar existentes (directo). Las nuevas solo se comparan entre sí.')

    # ── FASE B: extracción paralela por rondas ───────────────
    print('\n' + '═' * 55)
    print('FASE B — EXTRACCIÓN PARALELA POR RONDAS')
    print('═' * 55)
    t0_total = time.time()
    estado_global = {'descartadas': 0}
    pbar = tqdm(total=MAX_IMAGENES, initial=_total_guardadas[0],
                desc='Imágenes guardadas', unit='img', dynamic_ncols=True)

    # PASO 1: pase normal con umbral base
    correr_rondas(fuentes, indice, pbar, checkpoint, estado_global)

    # PASO 2: relajar umbral progresivamente si falta
    if GARANTIZAR_EXACTO and _cupo_restante() > 0:
        umbral_actual = args.umbral
        while _cupo_restante() > 0 and umbral_actual < UMBRAL_MAX:
            umbral_actual = min(umbral_actual + UMBRAL_PASO, UMBRAL_MAX)
            print('\n' + '·' * 55)
            print(f'⚙️  Faltan {_cupo_restante():,} imgs — relajando umbral '
                  f'dHash a {umbral_actual} y reintentando')
            print('·' * 55)
            indice.set_umbral(umbral_actual)
            for f in fuentes:
                if f.guardadas < f.disponibles:
                    f.reabrir()
            correr_rondas(fuentes, indice, pbar, checkpoint, estado_global)

    # PASO 3: augmentación de relleno como último recurso
    # Solo si: fuentes agotadas Y ya se superó el mínimo de imágenes reales
    n_augmentadas = 0
    ya_guardadas  = _total_guardadas[0]
    if GARANTIZAR_EXACTO and PERMITIR_AUGMENT and _cupo_restante() > 0:
        if ya_guardadas >= args.min_augmentar:
            n_augmentadas = rellenar_augmentando(pbar, checkpoint)
        else:
            print(f'\n⚠️  Augmentación omitida: solo hay {ya_guardadas:,} imágenes '
                  f'reales (mínimo: {args.min_augmentar:,}).')
            print('   Las fuentes se agotaron antes de llegar al mínimo.')
            print('   Agrega más fuentes o baja --min-augmentar.')

    pbar.close()

    # ── Limpieza de temporales ───────────────────────────────
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # ── Verificación + resumen ───────────────────────────────
    total_final = len(list(DESTINO_CLASE.glob(f'{CLASE}_*.jpg')))
    reales      = total_final - n_augmentadas
    muestra     = list(DESTINO_CLASE.glob(f'{CLASE}_*.jpg'))
    muestra_n   = random.sample(muestra, min(100, len(muestra))) if muestra else []
    malas = 0
    for ruta in muestra_n:
        try:
            with Image.open(ruta) as img:
                assert img.size == IMG_SIZE and img.mode == 'RGB'
        except Exception:
            malas += 1

    print('\n' + '═' * 55)
    print('RESUMEN FINAL')
    print('═' * 55)
    print(f'  Imágenes en disco       : {total_final:,}')
    print(f'    · reales (fuentes)    : {reales:,}')
    print(f'    · augmentadas         : {n_augmentadas:,}')
    print(f'  Descartadas (similitud) : {estado_global["descartadas"]:,}')
    print(f'  Verificación            : {len(muestra_n)-malas}/{len(muestra_n)} '
          f'válidas (224x224 RGB)')
    print(f'  Meta                    : {MAX_IMAGENES:,}')
    print(f'  Tiempo total            : {(time.time()-t0_total)/60:.1f} min')
    if total_final == MAX_IMAGENES:
        estado = '✅ EXACTO — meta lograda'
    elif total_final > MAX_IMAGENES:
        estado = f'⚠️ {total_final - MAX_IMAGENES} de más (revisa cortes)'
    else:
        estado = '⚠️ INCOMPLETO — fuentes agotadas (¿sin imágenes base?)'
    print(f'  Estado                  : {estado}')
    print(f'  Destino                 : {DESTINO_CLASE}')


if __name__ == '__main__':
    main()
