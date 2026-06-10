"""
Estandarización local: clase `pokemon`
ExpoEscom — Clasificador Multietiqueta

Extrae imágenes de los ZIP/RAR en esta carpeta y las deja en
../dataset/pokemon/ como pokemon_NNNNNN.jpg (224x224 RGB JPG).

ESTRATEGIA DE CUOTAS (de menor a mayor fuente):
  Las fuentes chicas (official_art, fanart, ...) se estandarizan COMPLETAS.
  El resto de la meta se reparte en partes iguales entre las fuentes
  grandes (primera_temporada, juego, ...) para maximizar variedad.

FILTRO DE VARIEDAD:
  Cada imagen candidata se compara con las ya guardadas mediante un hash
  perceptual (dHash). Si es muy parecida a una existente se descarta y se
  toma otra en su lugar — así el dataset no se llena de frames casi
  idénticos (crítico para el juego: 1.3M capturas muy repetitivas).

Uso:
    python estandarizar_local.py --max 200000   # objetivo total
    python estandarizar_local.py --inventario   # solo muestra fuentes y cuenta
    python estandarizar_local.py --max 200000 --sin-filtro-existentes
        # no indexa las imágenes ya guardadas (más rápido, pero las nuevas
        # solo se comparan entre sí)

Para los .rar necesitas unrar:
    sudo apt-get install unrar
"""

import argparse
import io
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image

# ── Configuración ────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
DESTINO_CLASE = SCRIPT_DIR.parent / 'dataset' / 'pokemon'
CLASE         = 'pokemon'
IMG_SIZE      = (224, 224)
SEED          = 42
TMP_DIR       = SCRIPT_DIR / 'tmp_extract'
EXTENSIONES   = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

HASH_SIZE        = 8    # dHash 8x8 → 64 bits
UMBRAL_SIMILITUD = 8    # distancia Hamming máxima para considerar "similar"


def es_imagen(nombre):
    return Path(nombre).suffix.lower() in EXTENSIONES


def hay_unrar():
    return shutil.which('unrar') is not None


def guardar(img, idx):
    dest = DESTINO_CLASE / f'{CLASE}_{idx:06d}.jpg'
    img.convert('RGB').resize(IMG_SIZE, Image.LANCZOS).save(
        dest, 'JPEG', quality=90, optimize=True)


# ── Filtro de similitud (variedad del dataset) ───────────────

def dhash(img):
    """Hash perceptual de 64 bits: gradiente horizontal en rejilla 8x8."""
    g = img.convert('L').resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    px = list(g.get_flattened_data())
    h = 0
    for fila in range(HASH_SIZE):
        base = fila * (HASH_SIZE + 1)
        for col in range(HASH_SIZE):
            h = (h << 1) | int(px[base + col] > px[base + col + 1])
    return h


class IndiceSimilitud:
    """
    Índice de hashes perceptuales para descartar imágenes muy parecidas.

    Cada hash se indexa por sus 4 bloques de 16 bits: dos imágenes
    similares comparten casi siempre al menos un bloque exacto, así cada
    consulta compara contra un puñado de candidatos en vez de contra todo
    el dataset. La detección es exacta hasta distancia 3 y aproximada
    hasta el umbral — suficiente para variedad, no es dedup forense.
    """

    def __init__(self, umbral=UMBRAL_SIMILITUD):
        self.umbral = umbral
        self._buckets = [{}, {}, {}, {}]

    @staticmethod
    def _bloques(h):
        return ((h >> 48) & 0xFFFF, (h >> 32) & 0xFFFF,
                (h >> 16) & 0xFFFF, h & 0xFFFF)

    def es_similar(self, h):
        vistos = set()
        for i, b in enumerate(self._bloques(h)):
            for otro in self._buckets[i].get(b, ()):
                if otro in vistos:
                    continue
                vistos.add(otro)
                if bin(h ^ otro).count('1') <= self.umbral:
                    return True
        return False

    def agregar(self, h):
        for i, b in enumerate(self._bloques(h)):
            self._buckets[i].setdefault(b, []).append(h)


def procesar_imagen(img, contador, indice):
    """Descarta similares; si pasa el filtro, estandariza y guarda."""
    h = dhash(img)
    if indice.es_similar(h):
        return False
    guardar(img, contador[0])
    indice.agregar(h)
    contador[0] += 1
    return True


def indexar_existentes(indice):
    """Hashea lo que ya está en dataset/pokemon para que las imágenes
    nuevas tampoco repitan lo ya guardado."""
    rutas = list(DESTINO_CLASE.glob('*.jpg'))
    if not rutas:
        return
    print(f'\nIndexando {len(rutas):,} imágenes existentes '
          f'(filtro de similitud)...')
    t0 = time.time()
    for i, ruta in enumerate(rutas, 1):
        try:
            with Image.open(ruta) as img:
                indice.agregar(dhash(img))
        except Exception:
            continue
        if i % 10000 == 0:
            print(f'  {i:,}/{len(rutas):,}', flush=True)
    print(f'  Listo en {(time.time()-t0)/60:.1f} min')


# ── Tipos de fuente ──────────────────────────────────────────

class FuenteZip:
    """ZIP con imágenes directas (sin RARs anidados)."""
    def __init__(self, nombre, zip_path, nombres_dentro):
        self.nombre    = nombre
        self.zip_path  = zip_path
        self._nombres  = nombres_dentro
        self.disponibles = len(nombres_dentro)
        self.cuota     = 0

    def extraer(self, contador, indice):
        nombres = self._nombres[:]
        random.shuffle(nombres)
        procesadas = descartadas = 0
        with zipfile.ZipFile(self.zip_path) as zf:
            # Recorre TODOS los nombres hasta cubrir la cuota: si el filtro
            # de similitud rechaza una imagen, se intenta con la siguiente.
            for nombre in nombres:
                if procesadas >= self.cuota:
                    break
                try:
                    with zf.open(nombre) as f:
                        img = Image.open(io.BytesIO(f.read()))
                    if procesar_imagen(img, contador, indice):
                        procesadas += 1
                        if procesadas % 2000 == 0:
                            print(f'    {procesadas:,}/{self.cuota:,}',
                                  flush=True)
                    else:
                        descartadas += 1
                except Exception:
                    continue
        return procesadas, descartadas


class FuenteRar:
    """RAR directo o RAR extraído de un ZIP."""
    def __init__(self, nombre, rar_path, zip_origen=None, rar_interno=None):
        self.nombre       = nombre
        self.rar_path     = rar_path       # ruta real en disco (None si aún no extraído)
        self.zip_origen   = zip_origen     # ZIP que contiene este RAR
        self.rar_interno  = rar_interno    # nombre del RAR dentro del ZIP
        self.disponibles  = -1             # se llena en contar()
        self.cuota        = 0
        self._nombres     = None           # caché de filenames

    def _asegurar_rar_en_disco(self):
        """Si el RAR está dentro de un ZIP, lo extrae a TMP_DIR."""
        if self.rar_path and self.rar_path.exists():
            return True
        if not self.zip_origen:
            return False
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        dest = TMP_DIR / Path(self.rar_interno).name
        print(f'  → Extrayendo {Path(self.rar_interno).name} del ZIP...',
              flush=True)
        with zipfile.ZipFile(self.zip_origen) as zf:
            with zf.open(self.rar_interno) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        self.rar_path = dest
        return True

    def listar(self):
        if self._nombres is not None:
            return self._nombres
        if not self._asegurar_rar_en_disco():
            return []
        result = subprocess.run(['unrar', 'lb', str(self.rar_path)],
                                capture_output=True, text=True, timeout=600)
        self._nombres = [l.strip() for l in result.stdout.splitlines()
                         if l.strip() and es_imagen(l.strip())]
        return self._nombres

    def contar(self):
        nombres = self.listar()
        self.disponibles = len(nombres)
        return self.disponibles

    def extraer(self, contador, indice):
        nombres = self.listar()
        if not nombres:
            return 0, 0

        cuota_real = min(self.cuota, len(nombres))
        pendientes = nombres[:]
        random.shuffle(pendientes)

        tmp = TMP_DIR / f'batch_{self.nombre}'
        procesadas = descartadas = 0

        # Extrae por lotes: si el filtro de similitud descarta muchas,
        # pide otro lote hasta cubrir la cuota o agotar la fuente.
        while procesadas < cuota_real and pendientes:
            faltan = cuota_real - procesadas
            lote, pendientes = pendientes[:faltan * 2], pendientes[faltan * 2:]

            tmp.mkdir(parents=True, exist_ok=True)
            listfile = TMP_DIR / f'_list_{self.nombre}.txt'
            listfile.write_text('\n'.join(lote))
            subprocess.run(
                ['unrar', 'x', '-o+', str(self.rar_path),
                 f'@{listfile}', str(tmp) + '/'],
                capture_output=True, timeout=14400)
            listfile.unlink(missing_ok=True)

            rutas = [r for r in tmp.rglob('*')
                     if r.is_file() and es_imagen(r.name)]
            random.shuffle(rutas)
            for ruta in rutas:
                if procesadas >= cuota_real:
                    break
                try:
                    with Image.open(ruta) as img:
                        if procesar_imagen(img, contador, indice):
                            procesadas += 1
                            if procesadas % 2000 == 0:
                                print(f'    {procesadas:,}/{cuota_real:,}',
                                      flush=True)
                        else:
                            descartadas += 1
                except Exception:
                    continue

            shutil.rmtree(tmp, ignore_errors=True)

        return procesadas, descartadas


# ── Inventario ───────────────────────────────────────────────

def descubrir_fuentes():
    """Devuelve lista de FuenteZip / FuenteRar encontradas en SCRIPT_DIR."""
    fuentes = []
    zips = sorted(SCRIPT_DIR.glob('*.zip'))
    rars = sorted(SCRIPT_DIR.glob('*.rar'))

    for z in zips:
        with zipfile.ZipFile(z) as zf:
            entradas = [n for n in zf.namelist()
                        if not n.startswith('__MACOSX')]
        imgs   = [n for n in entradas if es_imagen(n)]
        anid   = [n for n in entradas if Path(n).suffix.lower() == '.rar']

        if imgs:
            fuentes.append(FuenteZip(z.stem, z, imgs))
        for rar in anid:
            fuentes.append(FuenteRar(Path(rar).stem, None,
                                     zip_origen=z, rar_interno=rar))

    for r in rars:
        fuentes.append(FuenteRar(r.stem, r))

    return fuentes


def asignar_cuotas(fuentes, total):
    """
    De menor a mayor fuente: si una fuente cabe completa en su parte
    proporcional, se toma TODA (official_art, fanart, ... entran enteras)
    y su sobrante se reparte entre las fuentes más grandes. Las últimas
    (primera_temporada, juego) absorben el resto en partes iguales.
    """
    activas = sorted([f for f in fuentes if f.disponibles > 0],
                     key=lambda f: f.disponibles)
    if not activas:
        return

    restante = total
    for i, f in enumerate(activas):
        n_restantes = len(activas) - i
        cuota_ideal = -(-restante // n_restantes)  # ceil
        f.cuota  = min(cuota_ideal, f.disponibles)
        restante -= f.cuota


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max', type=int, default=500,
                        help='Total de imágenes objetivo en dataset/pokemon '
                             '(incluye las que ya existen). Default: 500')
    parser.add_argument('--inventario', action='store_true',
                        help='Solo muestra fuentes y cuenta imágenes '
                             'disponibles, sin extraer nada')
    parser.add_argument('--umbral', type=int, default=UMBRAL_SIMILITUD,
                        help='Distancia Hamming máxima del dHash para '
                             f'descartar por similitud. Default: {UMBRAL_SIMILITUD}. '
                             'Más alto = más estricto = más variedad')
    parser.add_argument('--sin-filtro-existentes', action='store_true',
                        help='No indexa las imágenes ya guardadas en destino '
                             '(arranque más rápido; las nuevas solo se '
                             'comparan entre sí)')
    args = parser.parse_args()

    random.seed(SEED)
    DESTINO_CLASE.mkdir(parents=True, exist_ok=True)

    ya_tenemos = len(list(DESTINO_CLASE.glob('*.jpg')))
    print(f'Destino    : {DESTINO_CLASE}')
    print(f'Ya tenemos : {ya_tenemos:,}')

    # ── Descubrir fuentes ────────────────────────────────────
    fuentes = descubrir_fuentes()
    if not fuentes:
        sys.exit('❌ No hay .zip ni .rar en esta carpeta.')

    sin_unrar = [f for f in fuentes
                 if isinstance(f, FuenteRar) and not hay_unrar()]
    if sin_unrar:
        print(f'\n⚠️  {len(sin_unrar)} fuente(s) RAR ignoradas (unrar no instalado):')
        for f in sin_unrar:
            print(f'   - {f.nombre}')
        print('   Instala con: sudo apt-get install unrar')
        fuentes = [f for f in fuentes if f not in sin_unrar]
        if not fuentes:
            sys.exit('❌ Solo había RARs y no hay unrar.')

    # ── Contar todas las fuentes (siempre, para cuotas correctas) ──
    rar_fuentes = [f for f in fuentes if isinstance(f, FuenteRar)]
    if rar_fuentes:
        print(f'\nContando imágenes en {len(rar_fuentes)} fuente(s) RAR...')
        for f in rar_fuentes:
            print(f'  {f.nombre}...', end='', flush=True)
            n = f.contar()
            print(f' {n:,}')

    # Orden de menor a mayor: las chicas se procesan primero y cualquier
    # déficit fluye hacia las fuentes grandes que sí tienen de sobra.
    fuentes.sort(key=lambda f: f.disponibles)

    print('\n' + '─' * 55)
    print('FUENTES DETECTADAS (menor → mayor)')
    print('─' * 55)
    for f in fuentes:
        print(f'  {f.nombre:<40s} {f.disponibles:,} imgs')

    if args.inventario:
        return

    # ── Cuotas ───────────────────────────────────────────────
    faltan = max(0, args.max - ya_tenemos)
    print(f'\nMeta    : {args.max:,}')
    print(f'Faltan  : {faltan:,}')
    if faltan == 0:
        print('✅ Ya hay suficientes imágenes.')
        return

    asignar_cuotas(fuentes, faltan)

    print('\n' + '─' * 55)
    print('CUOTAS POR FUENTE')
    print('─' * 55)
    for f in fuentes:
        completa = '  (COMPLETA)' if f.cuota >= f.disponibles > 0 else ''
        print(f'  {f.nombre:<40s} → {f.cuota:,}{completa}')

    # ── Filtro de similitud ──────────────────────────────────
    indice = IndiceSimilitud(umbral=args.umbral)
    print(f'\nFiltro de variedad: dHash, umbral Hamming ≤ {args.umbral}')
    if not args.sin_filtro_existentes:
        indexar_existentes(indice)

    # ── Extracción ───────────────────────────────────────────
    print('\n' + '═' * 55)
    print('EXTRAYENDO')
    print('═' * 55)
    contador = [ya_tenemos]
    total_descartadas = 0
    t0_total = time.time()

    for i, f in enumerate(fuentes):
        if f.cuota <= 0:
            continue
        print(f'\n📦 {f.nombre}  (cuota: {f.cuota:,})')
        t0 = time.time()
        try:
            n, descartadas = f.extraer(contador, indice)
        except Exception as e:
            print(f'  ❌ Error: {e}')
            n, descartadas = 0, 0
        total_descartadas += descartadas
        elapsed = time.time() - t0
        vel = n / elapsed if elapsed > 0 else 0
        print(f'  ✅ +{n:,} imágenes  '
              f'({descartadas:,} descartadas por similitud, '
              f'{elapsed/60:.1f} min, ~{vel:.0f} img/s)')

        # Redistribuir déficit a las fuentes siguientes (las grandes)
        deficit = f.cuota - n
        if deficit > 0:
            siguientes = [s for s in fuentes[i+1:] if s.cuota > 0]
            if siguientes:
                extra = -(-deficit // len(siguientes))
                for s in siguientes:
                    s.cuota += extra

    # ── Limpieza de temporales ───────────────────────────────
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # ── Resumen ──────────────────────────────────────────────
    total = len(list(DESTINO_CLASE.glob('*.jpg')))
    muestra = list(DESTINO_CLASE.glob('*.jpg'))
    muestra_n = random.sample(muestra, min(100, len(muestra)))
    malas = 0
    for ruta in muestra_n:
        try:
            with Image.open(ruta) as img:
                assert img.size == IMG_SIZE and img.mode == 'RGB'
        except Exception:
            malas += 1

    print('\n' + '═' * 55)
    print('RESUMEN')
    print('═' * 55)
    print(f'Imágenes en destino : {total:,}')
    print(f'Descartadas (similitud): {total_descartadas:,}')
    print(f'Verificación        : {len(muestra_n)-malas}/{len(muestra_n)} '
          f'válidas (224x224 RGB)')
    print(f'Tiempo total        : {(time.time()-t0_total)/60:.1f} min')
    estado = '✅ COMPLETO' if total >= args.max else '⚠️ INCOMPLETO'
    print(f'Estado              : {estado}')


if __name__ == '__main__':
    main()
