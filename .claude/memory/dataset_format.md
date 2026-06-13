---
name: dataset_format
description: Dataset is 100% .jpg only; all counting and augmentation tools restricted to .jpg
metadata:
  type: project
---

**Critical constraint:** All tools count/process **only `.jpg` files**. No `.jpeg`, `.png`, `.webp`, etc.

- `EXTENSIONES_IMAGEN = {'.jpg'}` in `00_conteo_dataset.ipynb` (cell-2)
- Option [1] (count): uses Drive API, counts only `.jpg`
- Option [3] (augment): uses `find ... -iname "*.jpg"` (not "*.jpeg" or "*.png")
- All augmented images saved as `.jpg` (JPEG_QUALITY=95)

**Why:** Consistency. User has 100% `.jpg` dataset; treating mixed formats would create count divergence between [1] and [3].

**Related:** `01_estandarizar_pokemon_fusion_1.ipynb` (v3.1.0) also filters `.jpg` only via extension check in helper functions.
