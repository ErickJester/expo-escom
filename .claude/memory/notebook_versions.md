---
name: notebook_versions
description: 00_conteo_dataset.ipynb evolution — v3.0.0 menu, v3.1.x .jpg-only + tqdm, v4.0.0 augment option
metadata:
  type: project
---

**File:** `00_conteo_dataset.ipynb` (Colab, for shared Drive folder)

**Versions:**
- **v3.0.0:** Main menu [1] count, [2] aplanar (flatten nested subfolder)
- **v3.1.0:** Only `.jpg` (not png/webp), TARGET_IMAGENES=100k constant, opción_contar shows % + faltantes
- **v3.1.1:** tqdm.notebook progress bars (indeterminate for count, determinate for move)
- **v3.1.2:** Fix cannotAddParent 403 in mover_contenido (catch + removeParents only if already in dest)
- **v4.0.0:** Option [3] Completar a 100k — augment ONLY the deficit (e.g. pokemon 92k → gen 8k)
- **v4.2.0:** Option [2] copy parallelized — `ThreadPoolExecutor(NUM_COPY_WORKERS=8)`, thread-local `service` (httplib2 not thread-safe), 429 backoff retry. ~4-8x faster (10h → ~1h on 24k files).
- **v4.3.0:** Added option [4] Completar a 100k VÍA API (`augmentar_clase_api`, `opcion_augmentar_api`). Option [3] (FUSE) fails with `Input/output error` on huge folders (e.g. pokemon, tens of thousands of .jpg) — `find` over the Drive mount chokes. [4] uses Drive API instead (no FUSE): lists via pagination, downloads source with `get_media().execute()`, augments in-memory, uploads with `MediaIoBaseUpload`. Parallel (NUM_COPY_WORKERS), thread-local service+transform, 429 backoff. Uses FOLDER_ID directly — no MyDrive shortcut needed. [3] kept for small folders.
- **v4.1.0:** Option [2] changed from MOVE to COPY. Drive removed multi-parent (2020), so the `cannotAddParent` error from the old move can't be worked around by keeping both parents — to leave a copy in the subfolder you must `files().copy()` (new ID, doubles storage). `mover_contenido` → `copiar_contenido`: copies file-by-file, skips names already in destino (re-run safe), skips folders (copy() can't copy folders). Subfolder is NOT emptied, so the trash-empty-subfolder step was removed.

**Option [3] details (v4.0.0):**
- Requires acceso directo (shortcut) in MyDrive → mounts as ruta montada
- Counts .jpg with shell `find` (resists FUSE better than os.scandir)
- Augments: flip, rotation, RandomResizedCrop, ColorJitter (transforms from `09_augmentacion_100k_3.ipynb`)
- ThreadPoolExecutor with 4 workers, thread-local transforms
- Guard MAX_AUG=30k (won't augment >30k per class — dataset quality check)
- Uploads new .jpgs to Drive with tqdm progress

**Setup:** Run cells 1→4 once (auth, config, helpers, options). Then cell-6 (menu) repeatedly.

**Config (cell-2):**
- FOLDER_ID = shared folder (read-only via API for [1] [2])
- NOMBRE_ACCESO_DIRECTO = shortcut name in MyDrive (for [3])
- TARGET_IMAGENES = 100_000
- MAX_AUG = 30_000
