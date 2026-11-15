"""T3.5: deterministic storage.

``numpy.savez_compressed`` stamps each zip entry with the current time, so identical arrays give different file hashes.
``write_npz`` writes the same .npz layout (one ``<name>.npy`` member per array, loadable with ``numpy.load``) with a fixed
member timestamp and order, so identical inputs give byte-identical files. Pickle is never used. Files are written to a
temporary name and renamed into place (text files use the existing Task 2 helper).
"""

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from ..image_kinematics_run import _atomic_write_text, sha256_file

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)
    return sha256_file(path)


def write_json(path: Path, obj: Any) -> str:
    return _atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False, sort_keys=False) + "\n")


def write_text(path: Path, text: str) -> str:
    return _atomic_write_text(path, text)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


__all__ = ["write_npz", "write_json", "write_text", "load_npz"]
