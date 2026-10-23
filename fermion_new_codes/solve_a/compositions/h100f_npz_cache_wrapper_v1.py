#!/usr/bin/env python
"""h100f_npz_cache_wrapper_v1.py (solve 8a7b5274, COMP1 h100-f checks): run an engine file unchanged, with one I/O-only patch.

The engines' restart (engine_collapse_v36.resume_slice_v34 and engine_twofield_v*.light_resume_v36) does, for every stored fields key and every
row i <= ks, np.asarray(Zf[key][i]) on an np.load()-ed *_fields.npz. NpzFile.__getitem__ re-reads and decompresses the whole member on every access,
and the row view keeps that fresh full copy alive: memory ~ keys x (ks+1) x member size (here ~20 x 961 x 21 MB = ~400 GB; attempt 1 on h100-f
grew ~100 MB/s and was stopped). This wrapper makes NpzFile.__getitem__ decompress each member of a file whose name ends in "_fields.npz" once,
cache it per NpzFile object and mark it read-only (an in-place write would raise rather than silently differ). Values are bitwise those
np.load returns; nothing else is touched. Usage: python -I -B h100f_npz_cache_wrapper_v1.py ENGINE.py ARGS..."""
import runpy
import sys

import numpy as np
import numpy.lib.npyio as npio

_orig = npio.NpzFile.__getitem__


def _getitem(self, key):
    name = str(getattr(getattr(self, "fid", None), "name", "") or "")
    if not name.endswith("_fields.npz"):
        return _orig(self, key)
    cache = self.__dict__.setdefault("_h100f_cache", {})
    if key not in cache:
        arr = _orig(self, key)
        if isinstance(arr, np.ndarray):
            arr.flags.writeable = False
        cache[key] = arr
    return cache[key]


npio.NpzFile.__getitem__ = _getitem
engine = sys.argv[1]
sys.argv = sys.argv[1:]
print("h100f_npz_cache_wrapper_v1: numpy %s; NpzFile.__getitem__ cached read-only for *_fields.npz; running %s" % (np.__version__, engine), flush=True)
runpy.run_path(engine, run_name="__main__")
