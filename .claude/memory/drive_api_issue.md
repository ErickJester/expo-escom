---
name: drive_api_cannotAddParent
description: Google Drive API 403 cannotAddParent when partial move runs leave files with dual parents
metadata:
  type: project
---

## Problem

In `mover_contenido()` (option [2] — flatten subcontent to parent), runs can be interrupted before completion. When retrying:

1. Files that were moved in the partial run still show up in the query `'{origen_id}' in parents` because Drive allows files to have **multiple parents simultaneously**.
2. Attempting `addParents=destino_id` fails with **403 Forbidden: "Increasing the number of parents is not allowed"** because the file already has `destino_id` as a parent.
3. **Pre-v3.1.2:** Code would break on this error and never reach the summary. User would see the error but not know if files moved.

## Solution (v3.1.2)

In `mover_contenido()`, catch `cannotAddParent` or "Increasing the number" in the error string and retry with **only** `removeParents=origen_id`:

```python
except Exception as e:
    err = str(e)
    if 'cannotAddParent' in err or 'Increasing the number of parents' in err:
        # File already at dest; just clean up the old parent
        service.files().update(
            fileId=f['id'], removeParents=origen_id,
            fields='id', supportsAllDrives=True).execute()
        _contar(f)
    else:
        print(f'⚠️ No se pudo mover {f["name"]}: {e}')
```

This completes the move without failing, and subsequent retries won't re-list that file.

## Real Incident

User ran option [2] on a folder with ~24k files. After 1179 moved, a cannotAddParent error occurred. Pre-v3.1.2 code threw an exception → KeyboardInterrupt. User saw "moved nothing" in summary, but actually 1179 files HAD moved.

With v3.1.2+, a re-run would find ~22.8k files remaining and complete without error.
