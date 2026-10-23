#!/usr/bin/env python
"""W7_common_v1.py (the solve lane; reorientation 6 composition 2, W7's five items). Shared helpers for W7_ancestry_v1.py, W7_slices_v1.py
and W7_compose_v1.py. Every definition is an existing one, imported or restated with its source:
- the engine: engine_single_object_B_v2 (CombinedHistory, roots_at, outer_inner_of) and the ops module behind it (chain_sparse, clocked_csr,
  number_current_csr), loaded exactly as dust_root_history_v2b.load_engine does;
- the dust columns at a launch: engine_single_object_B5_prep_v2's construction (the slice's occupied set [sea, dust] re-orthonormalized
  together by Loewdin in float64, the dust columns taken, advanced record by record by B's Chebyshev action when the target record follows
  the slice);
- the roots: dust_root_history_v2b's applied_outer_inner (the zeros of N + v on the applied site fields) and record_roots (the record's
  trapping zeros, the dense roots);
- the Killing bins: horizon3_bins_v1 (killing_bins_5, threshold_columns, band_top, the (lo, hi] rule);
- the held exterior block: export_horizon_refcol_v2's block (the held operator's restriction to the sites beyond the cut, both spinor
  components, site-major rows) and dust_ledger_postpass_v1b's alternative (the exterior part 1_(R > cut) psi on the full held operator's
  eigenvectors, extended = weight inside R_cut below a threshold, a finite-chain proxy for P_ac).
No mutual information. No plotting. No dates.
"""
import hashlib
import json
import os
import sys

import numpy as np

ENGINE_DIR = "<internal>/code"
PAGE_V6 = "<internal>/v6"
REC1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json"
FLD1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz"
REC2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json"
FLD2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz"
SLICE_DIR = "<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11"
SLICE_FMT = SLICE_DIR + "/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T%07.2f.npz"
EXT_FMT = "orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T%07.2f.npz"
EXT_MANIFEST = "<internal>/MANIFEST_B_ext100_slices.md5"
SLICE_T_R11 = (12.0, 15.0, 16.0, 20.0, 24.0, 25.0, 28.0, 30.0, 35.0, 40.0)
SLICE_T_EXT = (48.0, 50.0, 52.0, 54.0, 56.0)
V2_JSON = "<internal>/dust_root_history_v2.json"
V2_NPZ = "<internal>/dust_root_history_v2.npz"
B5_NPZ = "<internal>/B5v2_cols_prod.npz"
G_NEWTON = 0.25          # the production's g (record json args.g)
M_MASS = 0.2
KAPPAS = (1, 2, 3, 4, 5)


def set_threads(argv, default="8"):
    t = default
    for i, s in enumerate(argv):
        if s == "--threads" and i + 1 < len(argv):
            t = argv[i + 1]
        elif s.startswith("--threads="):
            t = s.split("=", 1)[1]
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = t
    return int(t)


def md5(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            c = fh.read(block)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def dump_json(obj, path):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, default=jdefault, indent=1)
    os.replace(path + ".tmp", path)


def save_npz(arrays, path):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_engine():
    os.environ.setdefault("PAGE_V6_PATH", PAGE_V6)
    for p in (PAGE_V6, ENGINE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import engine_single_object_B_v2 as BV2
    import horizon3_bins_v1 as HB
    import dust_root_history_v2b as D2
    return torch, BV2, HB, D2


def site_w(X, n):
    """site weights (sites x columns) in the site-major layout (rows 2j, 2j+1 = site j); dust_root_history_v2b.site_w."""
    return (np.abs(X) ** 2).reshape(n, 2, -1).sum(axis=1)


def loewdin_dust(Z, kap):
    """engine_single_object_B5_prep_v2: Phi = [sea, dust] in complex128, Phi G^(-1/2); returns the dust columns, f, the stored purity defect,
    the Loewdin orthonormality defect (on the dust columns against the whole set), and the dust columns as stored (complex128)."""
    sea = np.asarray(Z["sea_%d" % kap], np.complex128)
    dust = np.asarray(Z["dust_%d" % kap], np.complex128)
    fd = np.asarray(Z["f_%d" % kap], float)
    ns, nd = sea.shape[1], dust.shape[1]
    Phi = np.concatenate([sea, dust], axis=1)
    del sea
    G = Phi.conj().T @ Phi
    f = np.concatenate([np.ones(ns), fd])
    fh = np.sqrt(np.clip(f, 0.0, None))
    mu = np.linalg.eigvalsh((fh[:, None] * G) * fh[None, :])
    stored_idem = float(np.max(np.abs(mu * mu - mu)))
    gw, gv = np.linalg.eigh(G)
    Mi = (gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T[:, ns:]
    D = Phi @ Mi
    orth = float(np.max(np.abs(D.conj().T @ D - np.eye(nd)))) if nd else 0.0
    return D, fd, stored_idem, orth, dust


def zeros_of(R, f):
    f = np.asarray(f, float); R = np.asarray(R, float); out = []
    s0, s1 = f[:-1], f[1:]
    idx = np.nonzero(((s0 < 0) & (s1 >= 0)) | ((s0 >= 0) & (s1 < 0)))[0]
    for i in idx:
        if s1[i] == s0[i]:
            continue
        out.append((float(R[i] - s0[i] * (R[i + 1] - R[i]) / (s1[i] - s0[i])), 1 if s1[i] > s0[i] else -1))
    return out


def applied_outer_inner(R, f):
    """dust_root_history_v2b.applied_outer_inner, restated (the largest rising zero of N + v; the largest falling zero below it)."""
    z = zeros_of(R, f)
    rising = [r for r, s in z if s > 0]
    if not rising:
        return None, None, z
    Ro = max(rising)
    falling = [r for r, s in z if s < 0 and r < Ro]
    return Ro, (max(falling) if falling else None), z


def record_roots(rec):
    """dust_root_history_v2b.record_roots: the record's trapping zeros (dense), largest and smallest."""
    z = rec.get("trapping_zeros") or []
    if z:
        return max(z), (min(z) if len(z) > 1 else None)
    return rec.get("R_t"), rec.get("r_in")


def block_rows(R, cut):
    """export_horizon_refcol_v2 (site-major, strict 'R > c'): the 2n-vector rows of the sites beyond the cut."""
    return np.nonzero(np.repeat(np.asarray(R, float) > cut, 2))[0]


def weights_in_basis(U, X):
    return np.abs((X.conj().T @ U).conj().T) ** 2


def mean_positive(e, w, m):
    """the mean Killing energy per particle of the part above m (levels e > m): sum e w / sum w over e > m, per column; and weighted."""
    sel = np.asarray(e) > m
    num = (np.asarray(e)[sel][:, None] * w[sel]).sum(axis=0)
    den = w[sel].sum(axis=0)
    return num, den
