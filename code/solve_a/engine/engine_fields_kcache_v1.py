"""engine_fields_kcache_v1.py (the lane's analyst builder; engine spec 42.1 (3) (ii)): the field solve's averaging kernels cached.
The profile of one field solve at 2048 sites (the r_s/a 32 cost run's sources, pod 1's CPU; results/engine/V36/field_profile_v1.md): 4.23 s,
of which 3.01 s in engine_fields_v4.kernels (37 calls of the Gaussian shell kernels K_0, K_1 on the fine grid against the source shells),
the rest the outward integration (6467 right-hand-side calls). The kernels depend on the grid points, the shells and ell only, never on the
sources, so the same arrays are returned for the same arguments: install() replaces the name `kernels` in the field modules by a cached
function keyed by the exact bytes of its arguments (a hit returns the stored arrays, identical to a recomputation bit for bit; read-only
copies are never mutated by the callers, which only multiply them), least recently used entries dropped above max_bytes (4 GiB). Counts of
hits, misses and the seconds saved are kept in STATS."""
import collections
import hashlib
import time

import numpy as np

STATS = {"hits": 0, "misses": 0, "compute_s": 0.0, "entries": 0, "bytes": 0}
_CACHE = collections.OrderedDict()
_ORIG = {}


def _key(r, R, ell):
    r = np.ascontiguousarray(np.asarray(r, float)); R = np.ascontiguousarray(np.asarray(R, float))
    h = hashlib.blake2b(digest_size=20)
    h.update(r.tobytes()); h.update(b"|"); h.update(R.tobytes()); h.update(b"|"); h.update(np.float64(ell).tobytes())
    return (r.shape, R.shape, h.digest())


def make_cached(orig, max_bytes):
    def kernels(r, R, ell):
        k = _key(r, R, ell)
        v = _CACHE.get(k)
        if v is not None:
            _CACHE.move_to_end(k); STATS["hits"] += 1
            return v
        t0 = time.time()
        v = orig(r, R, ell)
        for x in v:
            x.setflags(write=False)
        STATS["compute_s"] += time.time() - t0; STATS["misses"] += 1
        _CACHE[k] = v; STATS["bytes"] += sum(x.nbytes for x in v)
        while STATS["bytes"] > max_bytes and len(_CACHE) > 1:
            _, old = _CACHE.popitem(last=False); STATS["bytes"] -= sum(x.nbytes for x in old)
        STATS["entries"] = len(_CACHE)
        return v
    return kernels


def install(modules, max_bytes=4 * 2 ** 30):
    """replace `kernels` in each module (engine_fields_v4, v12, v14, v16, v17 as imported) by one cached function."""
    import engine_fields_v4 as F4
    orig = _ORIG.setdefault("kernels", F4.kernels)
    cached = make_cached(orig, max_bytes)
    for m in modules:
        if hasattr(m, "kernels"):
            m.kernels = cached
    return cached
