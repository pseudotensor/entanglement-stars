import os
import pickle
import struct
import sys
import time
import types

import numpy as np


def compare_files(a, b):
    with open(a, "rb") as f:
        aa = pickle.load(f)
    with open(b, "rb") as f:
        bb = pickle.load(f)
    if aa.keys() != bb.keys():
        raise AssertionError("different replay record sets")
    for rec in aa:
        x, y = aa[rec], bb[rec]
        if x.keys() != y.keys():
            missing = sorted(set(x) ^ set(y))
            raise AssertionError((rec, "different leaf paths", missing[:10]))
        for path in x:
            if x[path] != y[path]:
                raise AssertionError((rec, "bytewise mismatch", path))
    print("BITWISE PASS:", a, b)


def probes_for(info, R, ell):
    rc = float(info.get("r_collar", R[-1]))
    ivp = info.get("ivp") or {}
    r0 = float(ivp.get("r0", ell / 64.0))
    special = [0.0, r0, rc, float(R[-1] + 8.0 * ell)]
    special += [
        float(z["r"])
        for z in (info.get("gauge") or {}).get("zeros", [])
    ]
    special = np.asarray(special, dtype=float)
    edges = np.concatenate([
        special,
        np.nextafter(special, -np.inf),
        np.nextafter(special, np.inf),
    ])
    rf = np.asarray(info.get("rfine", []), dtype=float)
    parts = [
        np.asarray(R, dtype=float),
        np.asarray(info.get("r_nodes", []), dtype=float),
        rf,
        edges,
    ]
    for d in (32.0, 64.0, 128.0):
        parts.extend([rf + ell / d, rf - ell / d])
    q = np.concatenate(parts)
    q = np.unique(np.concatenate([q, -q]))
    return q, edges


def snapshot(value, q, edges):
    out = {}
    active = set()

    def walk(x, path):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise TypeError(("object array in gate", path))
            out[path] = ("array", x.dtype.str, x.shape, x.tobytes(order="C"))
            return
        if isinstance(x, np.generic):
            out[path] = ("numpy-scalar", x.dtype.str, x.tobytes())
            return
        if type(x) is float:
            out[path] = ("python-float", struct.pack("=d", x))
            return
        if x is None or type(x) in (bool, int, str, bytes):
            out[path] = (type(x).__name__, x)
            return

        ident = id(x)
        if ident in active:
            raise TypeError(("cycle in gate object", path))
        active.add(ident)
        try:
            if isinstance(x, dict):
                keys = sorted(x, key=repr)
                out[path] = ("dict", tuple(keys))
                for k in keys:
                    walk(x[k], path + "[" + repr(k) + "]")
            elif isinstance(x, (tuple, list)):
                out[path] = (type(x).__name__, len(x))
                for j, v in enumerate(x):
                    walk(v, path + "[" + str(j) + "]")
            elif isinstance(x, types.FunctionType):
                out[path] = ("function-values",)
                walk(x(q), path + ".vector")
                for j, r in enumerate(edges):
                    walk(x(float(r)), path + ".scalar[" + str(j) + "]")
                    walk(x(np.array([r], dtype=float)),
                         path + ".one[" + str(j) + "]")
            elif hasattr(x, "__dict__"):
                out[path] = (
                    "object", type(x).__module__, type(x).__qualname__)
                walk(vars(x), path + ".__dict__")
                if callable(x):
                    walk(x(q), path + ".values")
            else:
                # Never silently skip an unknown info object.
                raise TypeError(("unsupported gate object", path, type(x)))
        finally:
            active.remove(ident)

    walk(value, "result")
    return out


if sys.argv[1] == "--compare":
    compare_files(sys.argv[2], sys.argv[3])
    raise SystemExit(0)

# Same first five arguments as Input 05, plus the gate artifact path.
src_dir, fields_file, ell_text, nrec_text, reps_text, output = sys.argv[1:]
sys.path.insert(0, src_dir)

import engine_fields_v16 as FLD
import engine_fields_v4 as F4
import engine_fields_v12 as F12
import engine_fields_v14 as F14
import engine_fields_kcache_v1 as KC

KC.install([F4, F12, F14, FLD])

z = np.load(fields_file, allow_pickle=False)
R = z["R"]
g = float(z["g"])
ell = float(ell_text)
nrec = int(nrec_text)
reps = int(reps_text)
idx = list(range(len(z["T"]) - nrec, len(z["T"])))

artifacts = {}
tt = []
for rep in range(reps):
    for i in idx:
        t0 = time.perf_counter()
        Nn, vn, Wn, info = FLD.solve_constraints_ivp(
            z["eps_tot"][i], z["J_tot"][i], g, R, ell,
            N0=None, rtol=1e-12, atol=1e-16,
            collar_policy="flag", support_rel=1e-14,
        )
        averaged = None
        if not info.get("failed"):
            averaged = FLD.site_fields_kernel_average(info, R, ell)
        elapsed = time.perf_counter() - t0
        tt.append(elapsed)

        # Gate work is deliberately outside the solve timing.
        q, edges = probes_for(info, R, ell)
        artifacts[(rep, i)] = snapshot(
            (Nn, vn, Wn, info, averaged), q, edges)

        print(
            "rep %d rec %d T %.4f solve %.6f nfev %s failed %s"
            % (rep, i, z["T"][i], elapsed,
               (info.get("ivp") or {}).get("nfev"),
               info.get("failed")),
            flush=True,
        )

with open(output, "wb") as f:
    pickle.dump(artifacts, f, protocol=5)

print("solves %d mean %.6f total %.6f; kcache %s"
      % (len(tt), np.mean(tt), sum(tt), KC.STATS))
