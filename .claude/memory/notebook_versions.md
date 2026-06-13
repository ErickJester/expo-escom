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
