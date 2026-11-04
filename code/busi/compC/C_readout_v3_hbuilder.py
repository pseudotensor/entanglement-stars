#!/usr/bin/env python
"""C_readout_v3_hbuilder.py (the busi lane, Computation C): the slice adapter of C_readout_v3b (the ruling C_readout_v3b_ruling.md, sections 1 and 2).

The engine's slices (engine_twofield_v3.py, run(): the "orbitals_<tag>_T....npz" and "light_<tag>_T....npz" files) store, per channel, the state's
orbital columns (sea_<k> with occupation 1, dust_<k> with the occupations f_<k>; sea_L<k> for the light field), the continued reference's columns (R2_<k>,
R2_L<k>), the eigenvalues e_<k>, e_L<k> of the channel's one-body operator at the slice's applied fields and the occupation mask over them, and the applied
fields N_applied, v_applied themselves; no one-body operator is stored. This module gives the readout that operator by the engine's own code (nothing
re-derived):
    h_of(npz, field, kappa) -> engine_collapse_v36.clocked_host_sparse(ch, N_applied, v_applied)   (a host CSR, 2 n_sites square, complex128, Hermitian)
with ch = engine_collapse_v5.Chan(kappa, m, nsites, half_offset) for the heavy field (engine_vacflow_v1.chain_sparse(R, m, kappa): on-site m sigma_z +
(kappa/R) sigma_y, the sine bond -(i/2) sigma_x, the two-step Q = -(i/4) 1) and ch = engine_twofield_v3.LightChan(kappa, nsites, half_offset, sourcing) for
the light field (page_ops_v5.chain_sparse(R, 0, kappa), checked against engine_vacflow_v1's at build); the clocked operator is (1/2){N, h0} + (1/2){v, Q}
with N, v repeated over the two components (flattened index 2 n + a). m, nsites, half_offset, kappa_sea and the light channel lists come from the run json's
args through configure(args[, args_control]); the readout's load_builder calls it with the two continued runs' args and stops when they differ.
The engine's modules are imported read-only (engine_twofield_v3 sets sys.dont_write_bytecode; nothing is written into the engine's directories): the busi
copies directory first (--engine-dir, BUSI_COMPC_DIR, default <internal>/compC), the solve lane's directories appended by
engine_twofield_v3 itself (BUSI_ENGINE_DIR, BUSI_PAGE_DIR; defaults <internal>/engine and .../page). The torch backend is bound to the CPU
(engine_collapse_v5.backend(None)) because Chan and LightChan build device tensors; torch's threads follow OMP_NUM_THREADS.
Also here:
  eigenpair_check(h, R2, e, mask): the adapter's h against a slice's stored eigenpairs (the reference's columns are eigenvectors of h at the stored
      eigenvalues): the residual ||h r - rq r||_2 per column (rq the Rayleigh quotient) against n eps ||h||_G + the width of the degenerate block the
      quotient falls in (engine_ops_v3.blocks_of_v3 at the engine's tol_block 1e-8; a rotated block's columns mix eigenvectors within its width), the sorted
      quotients against the sorted masked eigenvalues under the same threshold, the orthonormality defect max |R2^dagger R2 - 1| against n eps, the mask
      count against the column count; the threshold is the backward-error bound of a stable eigensolver (eigenpairs exact for h + E, ||E||_2 <= n eps
      ||h||_2 <= n eps ||h||_G; ||h||_G the Gershgorin bound, engine_ops_v2.gershgorin).
  check_slice(...): the real-data check of RUN_C_readout_v3b.md section 4 on one slice against its record: the eigenpairs per channel; the record's E_sea,
      E_ref, E_inv per light channel and E_inv_heavy, H_in (the chain-end cut, the flattened index range [0, 2 cut): inside_energy_sp's Hr = H[:2 cut, :]),
      H_in_light and H_in_weighted, M1_I_E, I_E_light, I_E_light_sourcing and I_E_cut_weighted by the engine's own energy_cut_current_sp (verbatim, the torch
      backend on the CPU); the stored mask against light_select_v36(e, sigma, mu) (light) and the negative set (heavy); the heavy weights against 16 kappa;
      N_applied, v_applied against the fields npz row; and, information only, the readout's own field_ops/point_terms on the same columns (its I_X must equal
      the engine's current, its H_in the engine's). The reproduction's resolution of a difference of two traces is the traces' roundoff: E_inv and H_in are
      compared against max(1e-6 relative, 64 eps (|Tr h C| + |Tr h P_ref|)); the currents, order-1 window sums cancelling to 1e-9, against 1e-3 relative.
  selftest_manufactured(...): the engine's h on a manufactured slice at 48 sites against engine_ops_v1.chain_parts + clocked (the lead's dense construction),
      Hermiticity, the eigenpair check on its own eigenpairs, a wrong R and a differing control stopping.
Usage:
    python C_readout_v3_hbuilder.py --selftest [--engine-dir DIR]
    python C_readout_v3_hbuilder.py --check --slice S.npz --run-json J1[,J2] [--fields-npz F.npz] [--channels all|heavy|light|1,2,L1] [--out out.json] [--engine-dir DIR]
Exit codes: 0 every check passed; 1 a check failed (the json and the printed table carry every number)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import math
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

VERSION = "C_readout_v3_hbuilder"
COMPC_DIR_DEFAULT = "<internal>/compC"
ENGINE_DIR_DEFAULT = "<internal>/engine"
PAGE_DIR_DEFAULT = "<internal>/page"
EPS = float(np.finfo(float).eps)
TOL_BLOCK = 1e-8        # engine_ops_v3.assign_max_overlap_v3's tol_block and engine_twofield_v3's --light-tol-block: the degenerate-block tolerance (relative to the band scale)
TOL_EXACT = 1e-6        # C_readout_v2.TOL_EXACT: a stored total reproduced from the stored operators
TOL_CURRENT = 1e-3      # C_readout_v3.TOL_CURRENT: a cut current reproduced (a difference of order-1 window sums cancelling to 1e-9)
CONFIG_KEYS = ("m", "nsites", "half_offset", "kappa_sea", "light_kappas", "light_guard_kappas")
CONFIG = {k: None for k in CONFIG_KEYS}
_ENG = {}
_CHAN = {}


# ------------------------------------------------------------------------------------------------------------------- the engine
def engine(engine_dir=None):
    """The engine's modules, imported once, read-only: engine_twofield_v3 (the busi copy; it appends BUSI_ENGINE_DIR and BUSI_PAGE_DIR to sys.path at import),
    engine_collapse_v36 (clocked_host_sparse, energy_cut_current_sp), engine_collapse_v5 (Chan, backend, BK), engine_ops_v3 (blocks_of_v3, gershgorin)."""
    if _ENG:
        return _ENG
    for d in (engine_dir, os.environ.get("BUSI_COMPC_DIR"), COMPC_DIR_DEFAULT):
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    os.environ.setdefault("BUSI_ENGINE_DIR", ENGINE_DIR_DEFAULT)
    os.environ.setdefault("BUSI_PAGE_DIR", PAGE_DIR_DEFAULT)
    import engine_twofield_v3 as E3
    import engine_collapse_v36 as V35
    import engine_collapse_v5 as C5
    import engine_ops_v3 as OPS3
    if C5.BK["torch"] is None:
        C5.backend(None)                     # the CPU: Chan and LightChan build device tensors on BK["dev"]
    torch = C5.BK["torch"]
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    except Exception:  # noqa: BLE001
        pass
    _ENG.update({"E3": E3, "V35": V35, "C5": C5, "OPS3": OPS3, "torch": torch})
    return _ENG


def configure(args, args_control=None):
    """The engine's arguments the operator needs, from the run json's args (input 09: m 0.2, nsites 2048, half_offset 0, kappa_sea 10, light_kappas '1,2',
    light_guard_kappas '3'); the control's must agree on every one of them (the adapter builds one operator family)."""
    missing = [k for k in CONFIG_KEYS if k not in args]
    if missing:
        raise KeyError("the run json's args carry no %s; the adapter's operator cannot be built without them" % missing)
    if args_control is not None:
        bad = [k for k in CONFIG_KEYS if str(args_control.get(k)) != str(args.get(k))]
        if bad:
            raise ValueError("the member's and the control's args differ in %s (%s against %s); the adapter builds one operator family"
                             % (bad, [args.get(k) for k in bad], [args_control.get(k) for k in bad]))
    new = {"m": float(args["m"]), "nsites": int(args["nsites"]), "half_offset": int(args.get("half_offset") or 0), "kappa_sea": int(args["kappa_sea"]),
           "light_kappas": str(args["light_kappas"]), "light_guard_kappas": str(args["light_guard_kappas"])}
    if any(CONFIG[k] != new[k] for k in CONFIG_KEYS):
        _CHAN.clear()
    CONFIG.update(new)
    return dict(CONFIG)


def channel(field, kappa):
    """The engine's channel object (cached): engine_collapse_v5.Chan(kappa, m, nsites, half) for the heavy field (self.hs, self.Qs = VF.chain_sparse(self.R, m,
    kappa)), engine_twofield_v3.LightChan(kappa, nsites, half, sourcing) for the light field (PO5.chain_sparse(self.R, 0.0, kappa), checked against VF at build)."""
    if CONFIG["m"] is None:
        raise RuntimeError("%s: configure(args) first (m, nsites, half_offset, kappa_sea from the run json's args)" % VERSION)
    key = (field, int(kappa))
    if key not in _CHAN:
        E = engine()
        if field == "heavy":
            _CHAN[key] = E["C5"].Chan(int(kappa), CONFIG["m"], CONFIG["nsites"], bool(CONFIG["half_offset"]))
        elif field == "light":
            src = [int(x) for x in CONFIG["light_kappas"].split(",") if x.strip()]
            _CHAN[key] = E["E3"].LightChan(int(kappa), CONFIG["nsites"], bool(CONFIG["half_offset"]), sourcing=(int(kappa) in src))
        else:
            raise ValueError("field %r: 'heavy' or 'light'" % (field,))
    return _CHAN[key]


def _keys(z):
    return list(z.files) if hasattr(z, "files") else list(z.keys())


def h_of(z, field, kappa):
    """The readout's builder: the channel's one-body operator at the slice's applied fields, the engine's host CSR
    engine_collapse_v36.clocked_host_sparse(ch, N_applied, v_applied) = (1/2){N, h0} + (1/2){v, Q} (complex128, 2 n_sites square, Hermitian), the object the
    engine's record readouts use (H_new in run()); the slice's R must be the channel's (a wrong half offset or chain stops)."""
    E = engine()
    ch = channel(field, kappa)
    N = np.asarray(z["N_applied"], dtype=float).ravel()
    v = np.asarray(z["v_applied"], dtype=float).ravel()
    if N.size != CONFIG["nsites"] or v.size != CONFIG["nsites"]:
        raise ValueError("%s: N_applied (%d) and v_applied (%d) against nsites %d" % (VERSION, N.size, v.size, CONFIG["nsites"]))
    if "R" in _keys(z):
        R = np.asarray(z["R"], dtype=float).ravel()
        if R.size != ch.R.size or not np.array_equal(R, ch.R):
            raise ValueError("%s: the slice's R is not the channel's (half_offset %d, nsites %d): first %s against %s" % (VERSION, CONFIG["half_offset"], CONFIG["nsites"], R[:3], ch.R[:3]))
    return E["V35"].clocked_host_sparse(ch, N, v)


def h_dense(z, field, kappa):
    """h_of as a dense array (the readout keeps the CSR: with the state and the reference stored as columns no dense product is needed)."""
    return h_of(z, field, kappa).toarray()


# ----------------------------------------------------------------------------------------------------------- the checks
def gershgorin(A):
    """The largest absolute row sum (engine_ops_v2.gershgorin's bound), for a sparse or a dense Hermitian matrix: an upper bound on ||A||_2."""
    return float(np.max(np.asarray(abs(A).sum(axis=1)).ravel()))


def block_widths(e, tol_block=TOL_BLOCK):
    """The width of the numerically degenerate block each level sits in (engine_ops_v3.blocks_of_v3: consecutive levels whose whole width is below tol_block times
    the band scale), e ascending."""
    e = np.asarray(e, dtype=float)
    blocks = engine()["OPS3"].blocks_of_v3(e, tol_block)
    w = np.zeros(len(e))
    for (i, j) in blocks:
        w[i:j] = e[j - 1] - e[i]
    return w


def eigenpair_check(h, R2, e, mask, tol_block=TOL_BLOCK):
    """The docstring's check; returns a dict with 'pass' and every number."""
    n = int(h.shape[0])
    G = gershgorin(h)
    thr0 = n * EPS * G
    R2 = np.asarray(R2)
    e = np.asarray(e, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()
    order = np.argsort(e, kind="stable")
    e_s, mask_s = e[order], mask[order]
    wid = block_widths(e_s, tol_block)
    out = {"n": n, "columns": int(R2.shape[1]), "gershgorin": G, "threshold_backward_error": thr0, "mask_count": int(mask.sum()), "eigenvalue_order_stored_ascending": bool(np.all(np.diff(e) >= 0.0)),
           "mask_count_equals_columns": bool(int(mask.sum()) == int(R2.shape[1])), "threshold_orthonormality": n * EPS}
    if R2.shape[1] == 0:
        out.update({"residual_max": 0.0, "residual_over_threshold_max": 0.0, "eigenvalue_mismatch_max": 0.0, "eigenvalue_mismatch_over_threshold_max": 0.0, "orthonormality_defect": 0.0,
                    "block_width_max_used": 0.0, "pass": out["mask_count_equals_columns"]})
        return out
    HR = np.asarray(h @ R2)
    rq = np.einsum("ij,ij->j", R2.conj(), HR).real
    res = np.linalg.norm(HR - R2 * rq[None, :], axis=0)
    pos = np.clip(np.searchsorted(e_s, rq), 0, n - 1)
    left = np.maximum(pos - 1, 0)
    near = np.where(np.abs(e_s[left] - rq) < np.abs(e_s[pos] - rq), left, pos)
    thr = thr0 + wid[near]
    out["residual_max"] = float(res.max())
    out["residual_over_threshold_max"] = float((res / thr).max())
    out["block_width_max_used"] = float(wid[near].max())
    G2 = R2.conj().T @ R2
    out["orthonormality_defect"] = float(np.max(np.abs(G2 - np.eye(G2.shape[0]))))
    if out["mask_count_equals_columns"]:
        em = e_s[mask_s]
        d = np.abs(np.sort(rq) - em)
        thrm = thr0 + wid[mask_s]
        out["eigenvalue_mismatch_max"] = float(d.max())
        out["eigenvalue_mismatch_over_threshold_max"] = float((d / thrm).max())
    else:
        out["eigenvalue_mismatch_max"] = None
        out["eigenvalue_mismatch_over_threshold_max"] = None
    out["pass"] = bool(out["mask_count_equals_columns"] and out["residual_over_threshold_max"] <= 1.0 and out["eigenvalue_mismatch_over_threshold_max"] <= 1.0
                       and out["orthonormality_defect"] <= n * EPS)
    return out


def _as_cols(a, n, name):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != n:
        raise ValueError("%s: a column array of shape %s against the dimension %d" % (name, a.shape, n))
    return a if a.dtype == np.complex128 else a.astype(np.complex128)


def _unpack_mask(bits, n):
    return np.unpackbits(np.asarray(bits, dtype=np.uint8))[:n].astype(bool)


def channel_numbers(E, h, sea, dust, f, R2, cut):
    """The engine's record quantities of one channel from the slice's columns and h (unweighted): the chain traces Tr[h C] and Tr[h P_ref] (the per-index
    diagonals summed, C = sea sea^dagger + dust diag(f) dust^dagger, P_ref = R2 R2^dagger), the inside and outside sums at the cut (the flattened index range
    [0, 2 cut) inside: inside_energy_sp's Hr = H[:2 cut, :]), and the cut current by the engine's energy_cut_current_sp (verbatim; the torch backend on the CPU)
    for D = C - P_ref (I_E, the record's) and for C alone (I_C)."""
    torch = E["torch"]
    HS = np.asarray(h @ sea)
    dC = np.einsum("ij,ij->i", sea.conj(), HS).real
    HD = None
    if dust is not None and dust.shape[1]:
        HD = np.asarray(h @ dust)
        dC = dC + np.einsum("ij,ij,j->i", dust.conj(), HD, np.asarray(f, dtype=float)).real
    HR = np.asarray(h @ R2)
    dP = np.einsum("ij,ij->i", R2.conj(), HR).real
    n2 = 2 * cut
    out = {"trace_C": float(dC.sum()), "trace_P": float(dP.sum()), "H_in": float((dC - dP)[:n2].sum()), "H_out": float((dC - dP)[n2:].sum()),
           "bare_in_C": float(dC[:n2].sum()), "bare_in_P": float(dP[:n2].sum())}
    cols = [sea] + ([dust] if HD is not None else []) + [R2]
    fs = np.concatenate([np.ones(sea.shape[1])] + ([np.asarray(f, dtype=float)] if HD is not None else []) + [-np.ones(R2.shape[1])])
    nC = int(sum(c.shape[1] for c in cols[:-1]))
    cat = np.ascontiguousarray(np.concatenate(cols, axis=1))
    try:
        out["I_E"] = float(E["V35"].energy_cut_current_sp(h, torch.as_tensor(cat), fs, cut))
        out["I_C"] = float(E["V35"].energy_cut_current_sp(h, torch.as_tensor(np.ascontiguousarray(cat[:, :nC])), fs[:nC], cut))
        out["current_error"] = None
    except Exception as ex:  # noqa: BLE001  reported, the other numbers stand
        out["I_E"] = out["I_C"] = None
        out["current_error"] = "%s: %s" % (type(ex).__name__, ex)
    del cat
    return out, dC, dP


def _res_trace(a, b):
    return 64.0 * EPS * (abs(a) + abs(b))


def check_slice(slice_path, run_jsons, fields_npz=None, channels="all", out_json=None, engine_dir=None, cross_readout=True, log=print):
    """The real-data check (the docstring); returns the result dict (with 'pass') and writes it to out_json when given."""
    t0 = time.time()
    E = engine(engine_dir)
    z = np.load(slice_path, allow_pickle=False)
    keys = list(z.files)
    T = float(np.asarray(z["T"]).ravel()[0])
    rs = float(np.asarray(z["rs"]).ravel()[0])
    clock = T / rs
    args0, rec, rec_from = None, None, None
    for p in run_jsons:
        with open(p) as fh:
            J = json.load(fh)
        if args0 is None:
            args0 = J["args"]
            configure(args0)
        else:
            configure(args0, J["args"])
        if rec is None:
            for r in J["records"]:
                if r.get("E") is not None and abs(float(r.get("T_over_rs", -1.0)) - clock) < 1e-9:
                    rec, rec_from = r, p
                    break
        del J
    if rec is None:
        raise RuntimeError("no record at T/r_s %g in %s" % (clock, run_jsons))
    nsites = CONFIG["nsites"]
    n = 2 * nsites
    cut = int(rec["M1_parts"]["cut"])
    cut_rule = int(min(max(round(float(rec["r_collar"])) - 1, 2), nsites - 10))
    R = np.asarray(z["R"], dtype=float).ravel()
    i24 = int(np.searchsorted(R, 24.0 * rs, side="right"))
    res = {"version": VERSION, "slice": os.path.abspath(slice_path), "clock": clock, "record_json": rec_from, "cut": cut, "cut_from_r_collar": cut_rule, "config": dict(CONFIG),
           "channels": {}, "totals": [], "checks": [], "readout_cross_check": {}}

    def chk(name, ok, detail=""):
        res["checks"].append({"name": name, "pass": (None if ok is None else bool(ok)), "detail": detail})
        log("%s  %s%s" % ("PASS" if ok else ("INFO" if ok is None else "FAIL"), name, (": " + detail) if detail else ""))

    def cmp(name, rec_v, sl_v, resolution):
        ok = None if (rec_v is None or sl_v is None) else bool(abs(sl_v - rec_v) <= resolution)
        res["totals"].append({"name": name, "record": rec_v, "slice": sl_v, "abs_difference": (None if ok is None else abs(sl_v - rec_v)), "resolution": resolution, "pass": ok})
        log("%s  %s: record %s, slice %s, |d| %s, resolution %s" % ("PASS" if ok else ("INFO" if ok is None else "FAIL"), name, rec_v, sl_v,
                                                                      (None if ok is None else "%.3e" % abs(sl_v - rec_v)), "%.3e" % resolution))
        return ok
    chk("the chain-end cut index: the record's M1_parts.cut = min(max(round(r_collar) - 1, 2), nsites - 10)", cut == cut_rule, "%d, %d (r_collar %s)" % (cut, cut_rule, rec["r_collar"]))
    if fields_npz:
        Zf = np.load(fields_npz, allow_pickle=False)
        Tz = np.asarray(Zf["T"], dtype=float).ravel()
        j = int(np.argmin(np.abs(Tz - T)))
        dN = float(np.max(np.abs(np.asarray(z["N_applied"], float) - np.asarray(Zf["N_applied"][j], float))))
        dv = float(np.max(np.abs(np.asarray(z["v_applied"], float) - np.asarray(Zf["v_applied"][j], float))))
        chk("N_applied, v_applied of the slice equal the fields npz row at the clock (the engine's restart check: slice_fields_vs_stored_max 0.0)", abs(Tz[j] - T) < 1e-9 and dN == 0.0 and dv == 0.0,
            "row %d (T %g), max |dN| %.3e, max |dv| %.3e" % (j, Tz[j], dN, dv))
        Zf.close()
    RD = None
    if cross_readout:
        try:
            import C_readout_v3b as RD  # noqa: F401  information only
        except Exception as ex:  # noqa: BLE001
            res["readout_cross_check"]["import_error"] = "%s: %s" % (type(ex).__name__, ex)
            RD = None

    def cross(field, kappa, zz, cn):
        if RD is None:
            return None
        try:
            ops = RD.field_ops(zz, list(zz), RD.role_keys({}, field, kappa), h_of, field, kappa, "adapter check")
            mask_end = RD.inside_mask(RD.site_index_map(n, nsites, "site_major"), cut, "<")
            pt = RD.point_terms(ops, {"end": mask_end})
            c = pt["cuts"]["end"]
            out = {"H_in_readout": c["H_in"], "H_in_engine": cn["H_in"], "I_X_readout": c["I_X"], "I_C_readout": c["I_C"], "I_E_engine": cn["I_E"], "I_C_engine": cn["I_C"],
                   "trace_X_readout": pt["trace_X"], "trace_X_engine": cn["trace_C"] - cn["trace_P"], "route": ops.get("route"), "orth_P": ops.get("orth_P")}
            out["rel_I_X_vs_engine"] = (abs(c["I_X"] - cn["I_E"]) / max(abs(cn["I_E"]), 1e-300)) if cn["I_E"] is not None else None
            out["abs_H_in_diff"] = abs(c["H_in"] - cn["H_in"])
            return out
        except Exception as ex:  # noqa: BLE001
            return {"error": "%s: %s" % (type(ex).__name__, ex)}
    sel = str(channels)
    want_heavy = sel in ("all", "heavy") or any(x.strip().isdigit() for x in sel.split(","))
    want_light = sel in ("all", "light") or any(x.strip().startswith("L") for x in sel.split(","))
    heavy_sel = None if sel in ("all", "heavy", "light") else [int(x) for x in sel.split(",") if x.strip().isdigit()]
    light_sel = None if sel in ("all", "heavy", "light") else [int(x.strip()[1:]) for x in sel.split(",") if x.strip().startswith("L")]
    # ---- the heavy channels
    heavy_kappas = [int(x) for x in np.asarray(z["kappas"]).ravel()] if "kappas" in keys else []
    if "weights" in keys and heavy_kappas:
        ww = [float(x) for x in np.asarray(z["weights"]).ravel()]
        chk("the slice's heavy weights are 16 kappa (engine_collapse_v5.Chan.w) for the channels %s" % heavy_kappas, all(abs(w - 16.0 * k) == 0.0 for k, w in zip(heavy_kappas, ww)), str(ww))
    totH = {"H_in": 0.0, "E_inv": 0.0, "I_E": 0.0, "I_C": 0.0, "bare_C": 0.0, "bare_P": 0.0, "trace_C": 0.0, "trace_P": 0.0, "n": 0, "I_missing": False}
    for k in heavy_kappas:
        if not want_heavy or (heavy_sel is not None and k not in heavy_sel):
            continue
        t1 = time.time()
        h = h_of(z, "heavy", k)
        sea = _as_cols(z["sea_%d" % k], n, "sea_%d" % k)
        dust = _as_cols(z["dust_%d" % k], n, "dust_%d" % k)
        f = np.asarray(z["f_%d" % k], dtype=float).ravel()
        R2 = _as_cols(z["R2_%d" % k], n, "R2_%d" % k)
        e = np.asarray(z["e_%d" % k], dtype=float).ravel()
        mask = _unpack_mask(z["mask_%d" % k], n)
        ec = eigenpair_check(h, R2, e, mask)
        cn, dC, dP = channel_numbers(E, h, sea, dust, f, R2, cut)
        cn["E_inv"] = cn["trace_C"] - cn["trace_P"]
        cn["E_ref_from_e"] = float(e[mask].sum())
        cn["H_out_24"] = float((dC - dP)[2 * i24:].sum())
        w = 16.0 * k
        for q, s_ in (("H_in", "H_in"), ("E_inv", "E_inv"), ("bare_C", "bare_in_C"), ("bare_P", "bare_in_P"), ("trace_C", "trace_C"), ("trace_P", "trace_P")):
            totH[q] += w * cn[s_]
        if cn["I_E"] is None:
            totH["I_missing"] = True
        else:
            totH["I_E"] += w * cn["I_E"]
            totH["I_C"] += w * cn["I_C"]
        totH["n"] += 1
        chk("heavy %d: R2's columns are eigenvectors of h at the stored eigenvalues (residual/threshold %.3g, eigenvalue mismatch/threshold %s, orthonormality %.1e against %.1e, %d columns = mask count %s; ||h||_G %.3f, threshold %.2e + block width <= %.1e)"
            % (k, ec["residual_over_threshold_max"], ("%.3g" % ec["eigenvalue_mismatch_over_threshold_max"]) if ec["eigenvalue_mismatch_over_threshold_max"] is not None else "-", ec["orthonormality_defect"],
               ec["threshold_orthonormality"], ec["columns"], ec["mask_count_equals_columns"], ec["gershgorin"], ec["threshold_backward_error"], ec["block_width_max_used"]), ec["pass"])
        chk("heavy %d: Tr[h P_ref] from the columns = the sum of the masked eigenvalues (|d| %.2e against 64 eps |Tr| %.2e)" % (k, abs(cn["trace_P"] - cn["E_ref_from_e"]), 64.0 * EPS * abs(cn["trace_P"])),
            abs(cn["trace_P"] - cn["E_ref_from_e"]) <= 64.0 * EPS * abs(cn["trace_P"]))
        chk("heavy %d: the stored mask equals the negative set (information; the record's mask_equals_negative_set %s)" % (k, (rec.get("continuation") or {}).get(str(k), {}).get("mask_equals_negative_set")),
            None, str(bool(np.array_equal(mask, e < 0.0))))
        if cn["current_error"]:
            chk("heavy %d: the engine's energy_cut_current_sp on the CPU" % k, False, cn["current_error"])
        zz = {"N_applied": z["N_applied"], "v_applied": z["v_applied"], "R": z["R"], "sea_%d" % k: sea, "dust_%d" % k: dust, "f_%d" % k: f, "R2_%d" % k: R2}
        xc = cross("heavy", k, zz, cn)
        res["channels"]["H%d" % k] = {"weight": w, "eigenpairs": ec, "numbers": cn, "mask_equals_negative_set": bool(np.array_equal(mask, e < 0.0)), "readout": xc, "seconds": time.time() - t1}
        if xc and "error" not in xc:
            res["readout_cross_check"]["H%d" % k] = xc
        log("heavy %d: %.1f s; E_inv %.6e (w %g), H_in %.9e, I_E %s, I_C %s%s" % (k, time.time() - t1, cn["E_inv"], w, cn["H_in"], cn["I_E"], cn["I_C"],
                                                                                   ("; readout I_X/engine I_E rel %.2e, |dH_in| %.2e" % (xc["rel_I_X_vs_engine"], xc["abs_H_in_diff"])) if (xc and "rel_I_X_vs_engine" in xc and xc["rel_I_X_vs_engine"] is not None) else ""))
        del h, sea, dust, R2, dC, dP, zz
    if totH["n"] and heavy_sel is None and want_heavy:
        cmp("H_in (the record's; the heavy channels' Tr_in(h (C - P_ref)) weighted 16 kappa, inside = flattened index < 2 cut)", float(rec["H_in"]), totH["H_in"],
            max(TOL_EXACT * abs(float(rec["H_in"])), _res_trace(totH["bare_C"], totH["bare_P"])))
        cmp("light_v36.E_inv_heavy (the heavy channels' Tr[h (C - P_ref)] weighted)", (rec.get("light_v36") or {}).get("E_inv_heavy"), totH["E_inv"],
            max(TOL_EXACT * abs(float((rec.get("light_v36") or {}).get("E_inv_heavy") or 0.0)), _res_trace(totH["trace_C"], totH["trace_P"])))
        if not totH["I_missing"]:
            cmp("M1_I_E (the heavy channels' energy_cut_current_sp weighted; inward-positive)", float(rec["M1_I_E"]), totH["I_E"], TOL_CURRENT * abs(float(rec["M1_I_E"])))
    # ---- the light channels
    lv = rec.get("light_v36") or {}
    chans = lv.get("channels") or {}
    kl = [int(x) for x in np.asarray(z["kappas_light"]).ravel()] if "kappas_light" in keys else []
    wl = {k_: float(w_) for k_, w_ in zip(kl, np.asarray(z["weights_light"]).ravel())} if "weights_light" in keys else {}
    srcl = {k_: int(s_) for k_, s_ in zip(kl, np.asarray(z["light_sourcing"]).ravel())} if "light_sourcing" in keys else {}
    if kl:
        chk("the slice's light weights are 16 kappa and the sourcing flags match the record's channels (%s)" % kl,
            all(abs(wl[k_] - 16.0 * k_) == 0.0 for k_ in kl) and all(bool(srcl.get(k_)) == bool((chans.get(str(k_)) or {}).get("sourcing")) for k_ in kl), "%s %s" % (wl, srcl))
    totL = {"H_in": 0.0, "E_inv": 0.0, "E_inv_src": 0.0, "I_E": 0.0, "I_E_src": 0.0, "bare_C": 0.0, "bare_P": 0.0, "trace_C": 0.0, "trace_P": 0.0, "n": 0, "I_missing": False}
    for k in kl:
        if not want_light or (light_sel is not None and k not in light_sel):
            continue
        t1 = time.time()
        h = h_of(z, "light", k)
        sea = _as_cols(z["sea_L%d" % k], n, "sea_L%d" % k)
        R2 = _as_cols(z["R2_L%d" % k], n, "R2_L%d" % k)
        e = np.asarray(z["e_L%d" % k], dtype=float).ravel()
        mask = _unpack_mask(z["mask_L%d" % k], n)
        sigma = np.asarray(z["sigma_L%d" % k], dtype=float).ravel()
        mu = float(np.asarray(z["mu_L%d" % k]).ravel()[0])
        nplus = int(np.asarray(z["nplus_L%d" % k]).ravel()[0])
        ec = eigenpair_check(h, R2, e, mask)
        cn, dC, dP = channel_numbers(E, h, sea, None, None, R2, cut)
        cn["E_inv"] = cn["trace_C"] - cn["trace_P"]
        cn["E_ref_from_e"] = float(e[mask].sum())
        cn["H_out_24"] = float((dC - dP)[2 * i24:].sum())
        w = wl.get(k, 16.0 * k)
        c_rec = chans.get(str(k)) or {}
        for q, s_ in (("H_in", "H_in"), ("E_inv", "E_inv"), ("bare_C", "bare_in_C"), ("bare_P", "bare_in_P"), ("trace_C", "trace_C"), ("trace_P", "trace_P")):
            totL[q] += w * cn[s_]
        if srcl.get(k):
            totL["E_inv_src"] += w * cn["E_inv"]
        if cn["I_E"] is None:
            totL["I_missing"] = True
        else:
            totL["I_E"] += w * cn["I_E"]
            if srcl.get(k):
                totL["I_E_src"] += w * cn["I_E"]
        totL["n"] += 1
        chk("light %d: R2's columns are eigenvectors of h at the stored eigenvalues (residual/threshold %.3g, eigenvalue mismatch/threshold %s, orthonormality %.1e against %.1e, %d columns = mask count %s; ||h||_G %.3f, threshold %.2e + block width <= %.1e)"
            % (k, ec["residual_over_threshold_max"], ("%.3g" % ec["eigenvalue_mismatch_over_threshold_max"]) if ec["eigenvalue_mismatch_over_threshold_max"] is not None else "-", ec["orthonormality_defect"],
               ec["threshold_orthonormality"], ec["columns"], ec["mask_count_equals_columns"], ec["gershgorin"], ec["threshold_backward_error"], ec["block_width_max_used"]), ec["pass"])
        occ_mu, gap_mu = E["E3"].light_select_v36(e, sigma, mu)
        chk("light %d: the stored mask is the mu S selection of the stored e, sigma, mu (engine_twofield_v3.light_select_v36; gap %.3e against the record's %s) and nplus = the + sector's count"
            % (k, gap_mu, c_rec.get("gap")), bool(np.array_equal(occ_mu, mask)) and int(np.sum(mask & (sigma > 0.0))) == nplus, "nplus %d" % nplus)
        if c_rec:
            cmp("light %d: E_sea = Tr[h C] (record channels.%d.E_sea)" % (k, k), c_rec.get("E_sea"), cn["trace_C"], 64.0 * EPS * abs(float(c_rec.get("E_sea") or 1.0)))
            cmp("light %d: E_ref = Tr[h P_ref] = the sum of the masked eigenvalues (record channels.%d.E_ref)" % (k, k), c_rec.get("E_ref"), cn["trace_P"], 64.0 * EPS * abs(float(c_rec.get("E_ref") or 1.0)))
            cmp("light %d: E_inv = Tr[h (C - P_ref)] (record channels.%d.E_inv; the resolution 64 eps (|E_sea| + |E_ref|))" % (k, k), c_rec.get("E_inv"), cn["E_inv"],
                max(TOL_EXACT * abs(float(c_rec.get("E_inv") or 0.0)), _res_trace(cn["trace_C"], cn["trace_P"])))
            cmp("light %d: H_in_weighted = w Tr_in(h (C - P_ref)) (record channels.%d.H_in_weighted)" % (k, k), c_rec.get("H_in_weighted"), w * cn["H_in"],
                max(TOL_EXACT * abs(float(c_rec.get("H_in_weighted") or 0.0)), w * _res_trace(cn["bare_in_C"], cn["bare_in_P"])))
            if cn["I_E"] is not None:
                cmp("light %d: I_E_cut_weighted = w energy_cut_current_sp (record channels.%d.I_E_cut_weighted)" % (k, k), c_rec.get("I_E_cut_weighted"), w * cn["I_E"],
                    TOL_CURRENT * abs(float(c_rec.get("I_E_cut_weighted") or 0.0)))
            b24 = ((c_rec.get("beyond") or {}).get("24") or {}).get("energy_density_route")
            if b24 is not None:
                res["channels"].setdefault("L%d" % k, {})["beyond_24"] = {"record_energy_density_route": b24, "slice_operator_trace_H_out": cn["H_out_24"],
                                                                        "rel_difference": abs(cn["H_out_24"] - b24) / max(abs(b24), 1e-300),
                                                                        "note": "the two routes differ by the cross-cut term (1/2) Re sum_{i out, j in} [h0_ij (N_j - N_i) + Q_ij (v_j - v_i)] X_ji; information"}
                log("INFO  light %d: 5d beyond 24 r_s energy_density_route %s against the operator trace H_out %s (rel %.2e; the cross-cut term is the expected difference)"
                    % (k, b24, cn["H_out_24"], abs(cn["H_out_24"] - b24) / max(abs(b24), 1e-300)))
        if cn["current_error"]:
            chk("light %d: the engine's energy_cut_current_sp on the CPU" % k, False, cn["current_error"])
        zz = {"N_applied": z["N_applied"], "v_applied": z["v_applied"], "R": z["R"], "sea_L%d" % k: sea, "R2_L%d" % k: R2}
        xc = cross("light", str(k), zz, cn)
        res["channels"].setdefault("L%d" % k, {}).update({"weight": w, "sourcing": bool(srcl.get(k)), "eigenpairs": ec, "numbers": cn, "mask_is_mu_S_selection": bool(np.array_equal(occ_mu, mask)),
                                                           "readout": xc, "seconds": time.time() - t1})
        if xc and "error" not in xc:
            res["readout_cross_check"]["L%d" % k] = xc
        log("light %d: %.1f s; E_inv %.6e (w %g), H_in %.9e, I_E %s%s" % (k, time.time() - t1, cn["E_inv"], w, cn["H_in"], cn["I_E"],
                                                                           ("; readout I_X/engine I_E rel %.2e, |dH_in| %.2e" % (xc["rel_I_X_vs_engine"], xc["abs_H_in_diff"])) if (xc and "rel_I_X_vs_engine" in xc and xc["rel_I_X_vs_engine"] is not None) else ""))
        del h, sea, R2, dC, dP, zz
    if totL["n"] and light_sel is None and want_light:
        cmp("light_v36.H_in_light (every light channel's w Tr_in(h (C - P_ref)))", lv.get("H_in_light"), totL["H_in"], max(TOL_EXACT * abs(float(lv.get("H_in_light") or 0.0)), _res_trace(totL["bare_C"], totL["bare_P"])))
        cmp("light_v36.E_inv_light (every channel, weighted)", lv.get("E_inv_light"), totL["E_inv"], max(TOL_EXACT * abs(float(lv.get("E_inv_light") or 0.0)), _res_trace(totL["trace_C"], totL["trace_P"])))
        cmp("light_v36.E_inv_light_sourcing (the sourcing channels, weighted)", lv.get("E_inv_light_sourcing"), totL["E_inv_src"],
            max(TOL_EXACT * abs(float(lv.get("E_inv_light_sourcing") or 0.0)), _res_trace(totL["trace_C"], totL["trace_P"])))
        if not totL["I_missing"]:
            cmp("light_v36.I_E_light (every channel's weighted current; inward-positive)", lv.get("I_E_light"), totL["I_E"], TOL_CURRENT * abs(float(lv.get("I_E_light") or 0.0)))
            cmp("light_v36.I_E_light_sourcing", lv.get("I_E_light_sourcing"), totL["I_E_src"], TOL_CURRENT * abs(float(lv.get("I_E_light_sourcing") or 0.0)))
    z.close()
    xs = [x for x in res["readout_cross_check"].values() if isinstance(x, dict) and x.get("rel_I_X_vs_engine") is not None]
    if xs:
        worst = max(x["rel_I_X_vs_engine"] for x in xs)
        worstH = max(x["abs_H_in_diff"] for x in xs)
        chk("the readout's own column route (C_readout_v3b.field_ops + point_terms) on the same columns: its I_X equals the engine's current within %s relative (worst %.2e) and its H_in the engine's (worst |d| %.2e)"
            % ("%.0e" % TOL_CURRENT, worst, worstH), worst <= TOL_CURRENT and worstH <= 1e-6 * max(1.0, abs(float(rec["H_in"]))))
    res["pass"] = bool(all(c["pass"] for c in res["checks"] if c["pass"] is not None) and all(t["pass"] for t in res["totals"] if t["pass"] is not None))
    res["seconds"] = time.time() - t0
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=1, default=(lambda o_: o_.tolist() if isinstance(o_, np.ndarray) else (float(o_) if isinstance(o_, np.floating) else (int(o_) if isinstance(o_, np.integer) else str(o_)))))
    log("ADAPTER CHECK %s: slice %s (T/r_s %g) against the record of %s; %d checks, %d totals; %.0f s%s"
        % ("PASS" if res["pass"] else "FAIL", slice_path, clock, rec_from, len(res["checks"]), len(res["totals"]), res["seconds"], ("; written %s" % out_json) if out_json else ""))
    return res


# ------------------------------------------------------------------------------------------------- the manufactured self-test
def selftest_manufactured(nsites=48, engine_dir=None, log=print):
    """The engine's h on a manufactured slice: Chan / LightChan with clocked_host_sparse against engine_ops_v1's chain_parts + clocked (the lead's dense
    construction, which engine_vacflow_v1.chain_sparse was checked against), Hermitian, its own eigenpairs passing eigenpair_check; a wrong R and a control
    with a different m stop. Returns {"ok", "summary", "details"}."""
    E = engine(engine_dir)
    import engine_ops_v1 as OPS1
    base = {"m": 0.2, "nsites": nsites, "half_offset": 0, "kappa_sea": 2, "light_kappas": "1,2", "light_guard_kappas": "3"}
    configure(base)
    R = np.arange(nsites, dtype=float) + 1.0
    N = 1.0 - 0.3 * np.exp(-R ** 2 / 50.0)
    v = -0.4 * (R / 5.0) * np.exp(-R ** 2 / 50.0)
    details = []
    ok_all = True
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "orbitals_test_T0040.00.npz")
        np.savez(p, T=1280.0, rs=32.0, N_applied=N, v_applied=v, R=R, dtype="complex128")
        z = np.load(p, allow_pickle=False)
        for field, kappa, m in (("heavy", 1, 0.2), ("heavy", 2, 0.2), ("light", 1, 0.0), ("light", 3, 0.0)):
            hs = h_of(z, field, kappa)
            hd = hs.toarray()
            h0, Q, R1 = OPS1.chain_parts(nsites, m, float(kappa))
            href = OPS1.clocked(h0, Q, N, v)
            sc = float(np.max(np.abs(href)))
            d_ref = float(np.max(np.abs(hd - href)))
            herm = float(np.max(np.abs(hd - hd.conj().T)))
            w, U = np.linalg.eigh(hd)
            neg = w < 0.0
            ec = eigenpair_check(hs, U[:, neg], w, neg)
            ok = bool(sp.issparse(hs) and hs.shape == (2 * nsites, 2 * nsites) and hs.dtype == np.complex128 and d_ref <= 8.0 * EPS * sc and herm <= 4.0 * EPS * sc and ec["pass"] and np.array_equal(R1, R))
            ok_all = ok_all and ok
            details.append({"field": field, "kappa": kappa, "max_abs_diff_vs_engine_ops_v1": d_ref, "scale": sc, "hermiticity": herm, "eigenpairs": ec, "ok": ok})
            log("%s  adapter manufactured %s kappa %d: |h - engine_ops_v1| %.2e (8 eps scale %.2e), hermiticity %.1e, eigenpair residual/threshold %.3g, orthonormality %.1e"
                % ("PASS" if ok else "FAIL", field, kappa, d_ref, 8.0 * EPS * sc, herm, ec["residual_over_threshold_max"], ec["orthonormality_defect"]))
        z.close()
    try:
        h_of({"N_applied": N, "v_applied": v, "R": R + 0.5}, "heavy", 1)
        bad_R = False
    except ValueError:
        bad_R = True
    try:
        configure(base, dict(base, m=0.3))
        bad_m = False
    except ValueError:
        bad_m = True
    configure(base)
    ok_all = ok_all and bad_R and bad_m
    summary = "%d channels %s; wrong R stops %s; differing control stops %s" % (len(details), "ok" if all(d["ok"] for d in details) else "FAILED", bad_R, bad_m)
    log("%s  adapter manufactured: %s" % ("PASS" if ok_all else "FAIL", summary))
    return {"ok": bool(ok_all), "summary": summary, "details": details}


def main(argv=None):
    ap = argparse.ArgumentParser(description="C_readout_v3b's slice adapter: the engine's one-body operator for the stored slices, its checks.")
    ap.add_argument("--selftest", action="store_true", help="the engine's h on a manufactured slice against engine_ops_v1's dense chain")
    ap.add_argument("--check", action="store_true", help="the real-data check of one slice against its record")
    ap.add_argument("--slice", default=None)
    ap.add_argument("--run-json", default=None, help="comma list; the first carries the args; the record at the slice's clock is taken from the first json carrying it")
    ap.add_argument("--fields-npz", default=None, help="the run's fields npz (N_applied, v_applied of the slice against its row)")
    ap.add_argument("--channels", default="all", help="all | heavy | light | a comma list such as 1,2,L1")
    ap.add_argument("--out", default=None, help="the check's json")
    ap.add_argument("--engine-dir", default=None, help="the busi copies of the engine (default BUSI_COMPC_DIR or %s)" % COMPC_DIR_DEFAULT)
    ap.add_argument("--no-cross-readout", action="store_true", help="skip the information-only cross-check with C_readout_v3b's own functions")
    a = ap.parse_args(argv)
    if a.selftest:
        rep = selftest_manufactured(engine_dir=a.engine_dir)
        print("ADAPTER SELFTEST %s: %s" % ("PASS" if rep["ok"] else "FAIL", rep["summary"]))
        return 0 if rep["ok"] else 1
    if a.check:
        if not (a.slice and a.run_json):
            ap.error("--check needs --slice and --run-json")
        res = check_slice(a.slice, [s for s in a.run_json.split(",") if s], a.fields_npz, a.channels, a.out, a.engine_dir, not a.no_cross_readout)
        return 0 if res["pass"] else 1
    ap.error("one of --selftest, --check")
    return 2


if __name__ == "__main__":
    sys.exit(main())
