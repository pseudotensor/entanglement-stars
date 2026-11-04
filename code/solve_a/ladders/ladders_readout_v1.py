#!/usr/bin/env python
"""ladders_readout_v1.py (composition builder, the solve lane): the producer of MASS_ladder_v1 and SCALING_ladder_v1.

Reads only stored run files (the members' collapse_<tag>.json records; the catalog json of ladders_catalog_v1.py) and saves every array
and number the tables, fits and figures use:
  <out>/ladders_readout_v1.npz   per member per record: T/r_s, E/E0, R_t/a, R_t/r_s, R_t/(2 g E), min(N - |v|), R_c/a, the kept residual,
                                 the confirming residual (Newton members), the converged flag, the counting flag, the number of trapping
                                 zeros, the tangencies, the pockets, the inversions, the slope |d(N + v)/dr| at R_t
  <out>/ladders_readout_v1.json  per member: inputs (path, bytes, md5, script, engine entries, arguments), counts (N_B, E0, f_kin,
                                 E/E_max, p_F/m), the catalog and onsets, the convergence summary (certified prefixes, excluded records),
                                 the formation (clock, radius, ratios, gate, residual, slope, tangencies, certification), the first episode,
                                 R_c's minimum, the matched-clock rows; per ladder: the fits (model, members, dof, coefficients,
                                 residuals) and the control spreads.
Definitions (the house rules of COLLAPSE_ladder v2.1 and engine_ladder_compare_v5, unchanged):
  a record holds a trapping zero when its trapping_zeros list is non-empty; R_t is the outermost zero of the dense N + v (the record's R_t);
  formation is the first record holding a zero; the first episode closes at the first later record holding none;
  the residual is engine_ladder_compare_v5.rec_residual (a v32-style record: its kept residual; a plain record: its last iterate's
  max(dN, dv) when converged, the stored unconverged residual otherwise); the confirming residual its confirm_residual;
  a record COUNTS when its converged entry is True and its residual is at most 1e-3; a capped record (converged False: the iteration cap
  reached) or a record above 1e-3 does not count and is named;
  the last converged record: the last record before the member's first non-counting record (the strict prefix);
  the reading rule 30.3: a record kept above 1e-3 uncertifies its member from it (the 30.3 prefix ends before it);
  a formation is certified when its record lies in the strict prefix; otherwise it is printed with its label and excluded from every fit;
  the root gate: sigma/R_t with sigma = ell (the Gaussian's per-axis standard deviation in lattice units);
  E/E_max = N_B (m a)^3/1.087; p_F/m = kappa_max/((R_0/a)(m a)), kappa_max = (9 pi N_B/32)^(1/3) (the isotropic release form of
  ladder_manifest_v1; at R_0 = 4 r_s it is 0.24 (2 g m a)^(-1/3) (a/r_s)^(2/3)/(m a)); the members' own (p_F/m)^2 = f_kin/0.3 from the
  stored head (the cold isotropic conversion, ladder_manifest_v1); the isotropic onset 1.222 N_B^(1/3) a = 4 a kappa_max/pi; the exclusion
  floor (3 N_B/(8 pi))^(1/3) a.
Fits (the requirement's rule): a law only where the members over-determine the fit; the model, the members, the dof and the residuals saved;
  otherwise the dependence (the per-member numbers) only.
Usage (box CPU, conda env qi2, nice -n 10, OMP_NUM_THREADS <= 16):
  python ladders_readout_v1.py --catalog CAT.json [--catalog CAT2.json ...] --out-dir DIR [--snap-dir DIR]"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_ladder_compare_v5 as LC  # noqa: E402

G = 0.25
C_ISO = (9.0 * math.pi / 32.0) ** (1.0 / 3.0)
C_EXCL = (3.0 / (8.0 * math.pi)) ** (1.0 / 3.0)
NMAX_COEF = 1.087
L = "<internal>/ladders"
PS = "<internal>/pod_sync"


def lad(card, prefix):
    d = os.path.join(L, card, "runs")
    hits = [x for x in sorted(os.listdir(d)) if x.startswith("lad_" + prefix + "_m0p")]
    if len(hits) != 1:
        raise SystemExit("member %s on %s: %d run directories %s" % (prefix, card, len(hits), hits))
    return os.path.join(d, hits[0], "collapse_%s.json" % hits[0])


# name: (path, label, role); the roles: member, control (sigma / step / solver), comparator, preparation, running
MEMBERS = [
    ("M3b_s4", lambda: lad("p6", "M3b_s4"), "(0.4, 64) sigma/a 4 tau 8", "member"),
    ("P64_s4", lambda: "<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json",
     "(0.2, 64) sigma/a 4 tau 8 (rung 4 on v36d from T = 0; the production's history)", "member"),
    ("M3a_s4", lambda: lad("p6", "M3a_s4"), "(0.1, 64) sigma/a 4 tau 8", "member"),
    ("M3c_s4_nk", lambda: lad("p8", "M3c_s4_nk"), "(0.1, 96) sigma/a 4 tau 8 (Newton, v36j)", "member"),
    ("M3d_s4", lambda: lad("r1", "M3d_s4"), "(0.05, 326) sigma/a 4 tau 8", "member"),
    ("S4_16_s4", lambda: lad("p1", "S4_16_s4"), "(0.2, 16) sigma/a 4 tau 2", "member"),
    ("RS32_s4", lambda: "<internal>/collapse_c31_rs32_A_R2_tau2_ell4_prep4_avg.json",
     "(0.2, 32) sigma/a 4 tau 2, the 1-field production c31_rs32_A_R2_tau2_ell4_prep4_avg", "member"),
    ("S4_128_s4", lambda: lad("r2", "S4_128_s4"), "(0.2, 128) sigma/a 4 tau 8 (plain)", "member"),
    # controls and comparators
    ("M3b_s8", lambda: lad("p8", "M3b_s8"), "(0.4, 64) sigma/a 8 tau 8", "control_sigma:M3b_s4"),
    ("M3a_s2", lambda: lad("p6", "M3a_s2"), "(0.1, 64) sigma/a 2 tau 8", "control_sigma:M3a_s4"),
    ("M3a_s4_tau4", lambda: lad("p6", "M3a_s4_tau4"), "(0.1, 64) sigma/a 4 tau 4 (to 16)", "control_step:M3a_s4"),
    ("M3c_s4_plain", lambda: lad("r3", "M3c_s4"), "(0.1, 96) sigma/a 4 tau 8 (plain, v36d)", "control_solver:M3c_s4_nk"),
    ("P64_s2", lambda: "<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json",
     "(0.2, 64) sigma/a 2 tau 8, the composite history (Newton, across its seams)", "control_sigma:P64_s4"),
    ("P64_s8", lambda: PS + "/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json",
     "(0.2, 64) sigma/a 8 tau 8 (rung 8 on v36d from T = 0)", "control_sigma:P64_s4"),
    ("P64_s4_tau4", lambda: PS + "/pod1_engine/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg.json",
     "(0.2, 64) sigma/a 4 tau 4 (engine v19)", "control_step:P64_s4"),
    ("P64_s4_tau2", lambda: PS + "/pod1_engine/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg.json",
     "(0.2, 64) sigma/a 4 tau 2 (engine v25, to 21.5)", "control_step:P64_s4"),
    ("P64_s4_c19", lambda: "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json",
     "(0.2, 64) sigma/a 4 tau 8 production record (engine v19)", "comparator:P64_s4"),
    ("S4_16_s2", lambda: lad("p1", "S4_16_s2"), "(0.2, 16) sigma/a 2 tau 2", "control_sigma:S4_16_s4"),
    ("S4_16_s4_tau1", lambda: lad("p1", "S4_16_s4_tau1"), "(0.2, 16) sigma/a 4 tau 1 (to 16)", "control_step:S4_16_s4"),
    ("RS32_s2", lambda: PS + "/pod1_engine/C3lad32r0b/collapse_c32_rs32_A_R2_tau2_ell2_prep4_avg_nk40r0b.json",
     "(0.2, 32) sigma/a 2 tau 2 (Newton)", "control_sigma:RS32_s4"),
    ("RS32_s4_tau1", lambda: PS + "/pod1_engine/C3set25/collapse_c25_rs32_A_R2_tau1_ell4_avg.json",
     "(0.2, 32) sigma/a 4 tau 1 (engine v25)", "control_step:RS32_s4"),
    ("S4_128_s4_nk_v36m", lambda: lad("p8", "S4_128_s4_nk"), "(0.2, 128) sigma/a 4 tau 8 Newton (v36m)", "control_solver:S4_128_s4"),
    ("S4_128_s4_nk_v36q", lambda: os.path.join(L, "r3/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q/"
                                                 "collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q.json"),
     "(0.2, 128) sigma/a 4 tau 8 Newton (v36q)", "control_solver:S4_128_s4"),
    ("S4_128_s8", lambda: lad("r1", "S4_128_s8"), "(0.2, 128) sigma/a 8 tau 8 (running; the mirrored records)", "control_sigma:S4_128_s4"),
    ("S4_128_s16", lambda: lad("p8", "S4_128_s16"), "(0.2, 128) sigma/a 16 tau 8", "control_sigma:S4_128_s4"),
    ("S4_R2_m0p2", lambda: lad("p8", "S4_R2_m0p2"), "(0.2, 64) sigma/a 4 tau 8, release at 2 r_s", "preparation:P64_s4"),
    ("S4_R2_m0p1", lambda: lad("p8", "S4_R2_m0p1"), "(0.1, 64) sigma/a 4 tau 8, release at 2 r_s", "preparation:M3a_s4"),
]
RUNNING = {"S4_128_s8"}
# the catalog entry each member reads (the same preparation: same m, r_s, sites, kappa_sea, R_0, prep_ell)
CATALOG_OF = {"M3a_s4": "M3a_s4_tau4", "M3a_s2": "M3a_s4_tau4", "M3a_s4_tau4": "M3a_s4_tau4", "M3b_s4": "M3b_s4", "M3b_s8": "M3b_s8",
              "M3c_s4_nk": "M3c_s4_nk", "M3c_s4_plain": "M3c_s4", "S4_128_s4": "S4_128_s4_nk", "S4_128_s8": "S4_128_s8",
              "S4_128_s16": "S4_128_s4_nk", "S4_128_s4_nk_v36m": "S4_128_s4_nk", "S4_128_s4_nk_v36q": "S4_128_s4_nk",
              "S4_16_s4": "S4_16_s4", "S4_16_s2": "S4_16_s4", "S4_16_s4_tau1": "S4_16_s4", "RS32_s4": "RS32_s4", "RS32_s2": "RS32_s4",
              "RS32_s4_tau1": "RS32_s4", "P64_s4": "P64_s4", "P64_s2": "P64_s4", "P64_s8": "P64_s4", "P64_s4_tau4": "P64_s4",
              "P64_s4_tau2": "P64_s4", "P64_s4_c19": "P64_s4", "S4_R2_m0p2": "S4_R2_m0p2", "S4_R2_m0p1": "S4_R2_m0p1"}


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fnum(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def rle(vals):
    out = []
    for t, v in vals:
        if out and out[-1]["value"] == v:
            out[-1]["last"] = t
            out[-1]["records"] += 1
        else:
            out.append({"value": v, "first": t, "last": t, "records": 1})
    return out


def load_member(name, path, label, role, snap_dir):
    src = path
    if name in RUNNING and snap_dir:
        os.makedirs(snap_dir, exist_ok=True)
        dst = os.path.join(snap_dir, os.path.basename(path))
        shutil.copy2(path, dst)
        src = dst
    t0 = time.time()
    J = json.load(open(src))
    A = J.get("args") or {}
    H = J.get("head") or {}
    newton = A.get("mix") == "newton"
    rs = float(A["rs"])
    m = float(A["m"])
    ell = float(A["ell"])
    recs = [r for r in (J.get("records") or []) if isinstance(r, dict) and "T_over_rs" in r and r.get("E") is not None]
    n = len(recs)
    cols = {k: np.full(n, np.nan) for k in ("T", "EE0", "E", "Rt", "Rt_rs", "Rt_2gE", "minNv", "r_minNv", "Rc", "res", "conf", "slope",
                                             "four_gE_kappa_T")}
    ints = {k: np.zeros(n, dtype=np.int64) for k in ("conv", "counts", "nzeros", "ntang", "pockets", "inversions", "capped")}
    zeros_all = []
    for i, r in enumerate(recs):
        cols["T"][i] = float(r["T_over_rs"])
        cols["EE0"][i] = fnum(r.get("E_over_E0")) if fnum(r.get("E_over_E0")) is not None else np.nan
        cols["E"][i] = fnum(r.get("E")) or np.nan
        tz = r.get("trapping_zeros") or []
        tz = [float(z) for z in tz] if isinstance(tz, (list, tuple)) else []
        zeros_all.append(tz)
        ints["nzeros"][i] = len(tz)
        tg = r.get("tangencies") or []
        ints["ntang"][i] = len(tg) if isinstance(tg, (list, tuple)) else 0
        if tz:
            rt = fnum(r.get("R_t"))
            cols["Rt"][i] = rt if rt is not None else max(tz)
            cols["Rt_rs"][i] = cols["Rt"][i] / rs
            v = fnum(r.get("R_t_over_2gE"))
            cols["Rt_2gE"][i] = v if v is not None else np.nan
            s = LC.slope_at_Rt(r)
            cols["slope"][i] = s if s is not None else np.nan
            v = fnum(r.get("four_gE_kappa_T"))
            cols["four_gE_kappa_T"][i] = v if v is not None else np.nan
        for k, key in (("minNv", "min_N_minus_absv"), ("r_minNv", "r_min_N_minus_absv"), ("Rc", "R_c")):
            v = fnum(r.get(key))
            cols[k][i] = v if v is not None else np.nan
        res = LC.rec_residual(r, newton)
        cols["res"][i] = res if res is not None else np.nan
        if newton:
            c = LC.confirm_residual(r)
            cols["conf"][i] = c if c is not None else np.nan
        conv = r.get("converged") is True or (newton and r.get("converged") is None and res is not None)
        ints["conv"][i] = 1 if conv else 0
        ints["capped"][i] = 1 if r.get("converged") is False else 0
        ok = conv and (res is None or res <= 1e-3)
        ints["counts"][i] = 1 if ok else 0
        ints["pockets"][i] = LC.pockets(r)
        ints["inversions"][i] = int(r.get("inversions_total") or 0)
    engines = rle([(float(r["T_over_rs"]), str(r.get("engine"))) for r in recs])
    meta = {"name": name, "label": label, "role": role, "path": path, "read_from": src, "bytes": os.path.getsize(src), "md5": md5sum(src),
            "script": J.get("script"), "engine_entries": engines, "newton": newton, "running": name in RUNNING,
            "args": {k: A.get(k) for k in ("m", "g", "rs", "R0_over_rs", "nsites", "ell", "prep_ell", "kappa_sea", "tau", "T_over_rs", "mix",
                                           "maxit", "nk_maxiter", "nk_krylov", "nk_set", "end_above", "reference", "site_fields", "c_LZ",
                                           "reading", "half_offset")},
            "head": {k: H.get(k) for k in ("N_B", "E0", "f_kin", "f_kin_bulk_frame", "passes", "last_change")},
            "records": n, "load_seconds": round(time.time() - t0, 1)}
    meta["m"], meta["rs"], meta["sigma"], meta["tau"] = m, rs, ell, float(A["tau"])
    meta["R0_over_rs"] = float(A.get("R0_over_rs") or 4.0)
    meta["nsites"] = int(A["nsites"])
    return meta, cols, ints, zeros_all


def summarize(meta, cols, ints, zeros_all):
    T = cols["T"]
    n = len(T)
    nb = meta["head"].get("N_B")
    nb = float(nb) if nb is not None else meta["rs"] / (2 * G * meta["m"])
    m, rs, sig = meta["m"], meta["rs"], meta["sigma"]
    kmax = C_ISO * nb ** (1.0 / 3.0)
    counts = {"N_B": nb, "N_B_rule": rs / (2 * G * m), "E0": fnum(meta["head"].get("E0")), "f_kin": fnum(meta["head"].get("f_kin")),
              "E_over_Emax": nb * m ** 3 / NMAX_COEF, "kappa_max_iso": kmax,
              "p_F_over_m_iso": kmax / (meta["R0_over_rs"] * rs * m), "onset_iso_a": 4.0 * kmax / math.pi, "onset_iso_rs": 4.0 * kmax / math.pi / rs,
              "excl_floor_a": C_EXCL * nb ** (1.0 / 3.0), "tau_over_rs": meta["tau"] / rs, "sigma_over_rs": sig / rs}
    counts["pressure_corr_iso"] = counts["p_F_over_m_iso"] ** 2 / 5.0
    counts["x_iso"] = counts["p_F_over_m_iso"] ** 2
    counts["x_own"] = (counts["f_kin"] / 0.3) if counts["f_kin"] else None
    counts["p_F_over_m_own"] = math.sqrt(counts["x_own"]) if counts["x_own"] else None
    # convergence
    nc = np.where(ints["counts"] == 0)[0]
    above = np.where(np.nan_to_num(cols["res"], nan=0.0) > 1e-3)[0]
    r303_end = (n - 1) if len(above) == 0 else (above[0] - 1)
    # the last converged record: the last counting record inside the 30.3 prefix (non-counting records inside it are named)
    inside = np.where(ints["counts"][:r303_end + 1] == 1)[0] if r303_end >= 0 else np.array([], dtype=int)
    strict_end = int(inside[-1]) if len(inside) else -1
    nc_inside = [int(i) for i in nc if i <= r303_end]
    excluded = [{"T": float(T[i]), "residual": fnum(cols["res"][i]), "converged": bool(ints["conv"][i]),
                 "confirming": fnum(cols["conf"][i])} for i in nc]
    resmax_i = int(np.nanargmax(cols["res"])) if np.any(np.isfinite(cols["res"])) else None
    conv = {"records": n, "first": float(T[0]) if n else None, "last": float(T[-1]) if n else None,
            "converged_true": int(ints["conv"].sum()), "capped": int(ints["capped"].sum()), "counting": int(ints["counts"].sum()),
            "strict_prefix_end": float(T[strict_end]) if strict_end >= 0 else None,
            "last_converged": float(T[strict_end]) if strict_end >= 0 else None,
            "first_noncounting": float(T[nc[0]]) if len(nc) else None,
            "noncounting_inside_r303": [float(T[i]) for i in nc_inside],
            "r303_prefix_end": float(T[r303_end]) if r303_end >= 0 else None, "first_above_1e-3": float(T[above[0]]) if len(above) else None,
            "above_1e-3_count": int(len(above)), "excluded": excluded,
            "residual_max": fnum(cols["res"][resmax_i]) if resmax_i is not None else None,
            "residual_max_at": float(T[resmax_i]) if resmax_i is not None else None,
            "residual_max_in_strict_prefix": fnum(np.nanmax(cols["res"][:strict_end + 1])) if strict_end >= 0 else None,
            "below_1e-4": int(np.sum(np.nan_to_num(cols["res"], nan=1.0) < 1e-4)),
            "band_1e-4_1e-3": int(np.sum((np.nan_to_num(cols["res"], nan=0) >= 1e-4) & (np.nan_to_num(cols["res"], nan=0) <= 1e-3)))}
    # formation and episodes
    zi = np.where(ints["nzeros"] > 0)[0]
    form = None
    episodes = []
    if len(zi):
        i = int(zi[0])
        cert = bool(i <= r303_end and ints["counts"][i] == 1)
        before = [float(T[j]) for j in nc_inside if j < i]
        if cert:
            lab = "certified (the record counts and lies inside the 30.3 prefix)" + (
                "; %d non-counting (capped) records before it, %.4f to %.4f, named and excluded" % (len(before), before[0], before[-1]) if before else "")
        elif i > r303_end:
            lab = "uncertified: after the member's first record above 1e-3 (%.4f)" % conv["first_above_1e-3"]
        else:
            lab = "uncertified: the formation record itself does not count (converged %s, residual %s)" % (bool(ints["conv"][i]), fnum(cols["res"][i]))
        form = {"T": float(T[i]), "index": i, "R_t": fnum(cols["Rt"][i]), "R_t_over_rs": fnum(cols["Rt_rs"][i]), "R_t_over_2gE": fnum(cols["Rt_2gE"][i]),
                "E_over_E0": fnum(cols["EE0"][i]), "minNv": fnum(cols["minNv"][i]), "R_c": fnum(cols["Rc"][i]), "zeros": zeros_all[i],
                "n_zeros": int(ints["nzeros"][i]), "tangencies": int(ints["ntang"][i]), "slope": fnum(cols["slope"][i]),
                "residual": fnum(cols["res"][i]), "confirming": fnum(cols["conf"][i]), "converged": bool(ints["conv"][i]),
                "counts": bool(ints["counts"][i]), "gate_sigma_over_Rt": sig / cols["Rt"][i], "certified": bool(cert), "label": lab,
                "simple": bool(ints["ntang"][i] == 0 and np.isfinite(cols["slope"][i]) and cols["slope"][i] > 0),
                "four_gE_kappa_T": fnum(cols["four_gE_kappa_T"][i]),
                "prev_record_T": float(T[i - 1]) if i > 0 else None, "prev_minNv": fnum(cols["minNv"][i - 1]) if i > 0 else None}
        cur = None
        for j in range(n):
            if ints["nzeros"][j] > 0:
                if cur is None:
                    cur = {"first": float(T[j]), "last": float(T[j]), "records": 0, "min_minNv": None, "min_at": None,
                           "R_t_first": fnum(cols["Rt"][j]), "R_t_last": None, "closes_at": None, "noncounting_inside": 0}
                cur["last"] = float(T[j]); cur["records"] += 1; cur["R_t_last"] = fnum(cols["Rt"][j])
                if ints["counts"][j] == 0:
                    cur["noncounting_inside"] += 1
                mv = cols["minNv"][j]
                if np.isfinite(mv) and (cur["min_minNv"] is None or mv < cur["min_minNv"]):
                    cur["min_minNv"], cur["min_at"] = float(mv), float(T[j])
            elif cur is not None:
                cur["closes_at"] = float(T[j]); episodes.append(cur); cur = None
        if cur is not None:
            episodes.append(cur)
    # R_c minimum over the strict prefix
    rc = {}
    if strict_end >= 0:
        seg = np.where(ints["counts"][:strict_end + 1] == 1, cols["Rc"][:strict_end + 1], np.nan)
        if np.any(np.isfinite(seg)):
            k = int(np.nanargmin(seg))
            rc = {"min_Rc": float(seg[k]), "at": float(T[k]), "min_Rc_over_rs": float(seg[k]) / rs,
                  "over_end": (k == strict_end)}
    # matched clocks: integer clocks and formation offsets
    def row_at(t):
        j = np.where(np.abs(T - t) < 1e-6)[0]
        if not len(j):
            return None
        j = int(j[0])
        return {"T": float(T[j]), "EE0": fnum(cols["EE0"][j]), "R_t": fnum(cols["Rt"][j]), "R_t_over_rs": fnum(cols["Rt_rs"][j]),
                "R_t_over_2gE": fnum(cols["Rt_2gE"][j]), "minNv": fnum(cols["minNv"][j]), "R_c": fnum(cols["Rc"][j]),
                "R_c_over_rs": fnum(cols["Rc"][j] / rs), "residual": fnum(cols["res"][j]), "counts": bool(ints["counts"][j]),
                "gate": fnum(sig / cols["Rt"][j]) if np.isfinite(cols["Rt"][j]) else None, "pockets": int(ints["pockets"][j])}
    matched = {"%g" % t: row_at(float(t)) for t in range(0, 41)}
    rel = {}
    if form:
        for d in (0.0, 0.5, 1.0, 2.0, 4.0):
            rel["%g" % d] = row_at(form["T"] + d)
    return {"counts": counts, "convergence": conv, "formation": form, "episodes": episodes, "Rc_min": rc, "matched": matched,
            "matched_from_formation": rel}


def lstsq(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    return c, pred, y - pred


def fit_block(model, names, x_cols, ys, extra=None):
    """one linear-in-coefficients fit; dof = members - coefficients; residuals per member."""
    X = np.column_stack(x_cols)
    out = {"model": model, "members": names, "n": len(names), "coefficients": X.shape[1], "dof": len(names) - X.shape[1]}
    for key, y in ys.items():
        c, pred, res = lstsq(X, y)
        out[key] = {"coef": [float(v) for v in c], "pred": [float(v) for v in pred], "residual": [float(v) for v in res],
                    "rms": float(math.sqrt(np.mean(res ** 2))), "max_abs": float(np.max(np.abs(res))), "y": [float(v) for v in y]}
    if extra:
        out.update(extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snap-dir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cat = {}
    for c in a.catalog:
        cat.update(json.load(open(c)))
    S, arrays = {}, {}
    for name, pf, label, role in MEMBERS:
        path = pf()
        meta, cols, ints, zeros_all = load_member(name, path, label, role, a.snap_dir)
        summ = summarize(meta, cols, ints, zeros_all)
        ck = CATALOG_OF.get(name)
        c = cat.get(ck) if ck else None
        if c and c.get("n_rad_max"):
            summ["catalog"] = {"from": ck, "source": c.get("source"), "path": c.get("path"), "columns": c.get("columns"), "top": c.get("top"),
                               "n_rad_max": c.get("n_rad_max"), "onset_own_a": c.get("onset_4a_nradmax"),
                               "onset_own_rs": c.get("onset_4a_nradmax") / meta["rs"], "head_comparison": c.get("head_comparison"),
                               "same_preparation_as": (ck if ck != name else None)}
        else:
            summ["catalog"] = {"from": ck, "source": "not available", "onset_own_a": None}
        S[name] = {"meta": meta, **summ}
        for k, v in cols.items():
            arrays["%s__%s" % (name, k)] = v
        for k, v in ints.items():
            arrays["%s__%s" % (name, k)] = v
        print("%-18s %4d recs %7.3f->%7.3f strict %s form %s" % (name, meta["records"], summ["convergence"]["first"], summ["convergence"]["last"],
              summ["convergence"]["strict_prefix_end"], (summ["formation"] or {}).get("T")), flush=True)

    def F(nm):
        f = S[nm]["formation"]
        return f if (f and f["certified"]) else None

    fits = {}
    # ---------------- MASS: the fixed-exponent test (x = (p_F/m)^2, the isotropic release form) on the formed members
    mass_members = ["M3b_s4", "P64_s4", "M3a_s4", "M3c_s4_nk", "M3d_s4"]
    formed = [nm for nm in mass_members if F(nm)]
    if len(formed) >= 2:
        ys = {"T_f": [F(nm)["T"] for nm in formed], "R_t_over_rs": [F(nm)["R_t_over_rs"] for nm in formed],
              "R_t_over_2gE": [F(nm)["R_t_over_2gE"] for nm in formed]}
        xi = [S[nm]["counts"]["x_iso"] for nm in formed]
        fits["mass_fixed_iso"] = fit_block("Y = c0 + c1 x, x = (p_F/m)^2 of the isotropic release form (exponent fixed: -8/3 in m a at fixed r_s/a, "
                                           "-4/3 in r_s/a at fixed m a)", formed, [np.ones(len(formed)), xi], ys, {"x": xi})
        xo = [S[nm]["counts"]["x_own"] for nm in formed]
        if all(v is not None for v in xo):
            fits["mass_fixed_own"] = fit_block("Y = c0 + c1 x, x = f_kin/0.3 (the members' own stored preparation)", formed,
                                               [np.ones(len(formed)), xo], ys, {"x": xo})
        same64 = [nm for nm in formed if abs(S[nm]["meta"]["rs"] - 64.0) < 1e-9]
        if len(same64) >= 2:
            xm = [S[nm]["meta"]["m"] ** (-8.0 / 3.0) for nm in same64]
            fits["mass_fixed_r64"] = fit_block("Y = c0 + c1 (m a)^(-8/3) at r_s/a 64 only", same64, [np.ones(len(same64)), xm],
                                               {k: [F(nm)[kk] for nm in same64] for k, kk in (("T_f", "T"), ("R_t_over_rs", "R_t_over_rs"),
                                                                                              ("R_t_over_2gE", "R_t_over_2gE"))}, {"x": xm})
    # ---------------- SIZE: at sigma/a 4, the free exponent needs 4 formed members; fixed exponent 1/3 in a/r_s otherwise
    size_members = ["S4_16_s4", "RS32_s4", "P64_s4", "S4_128_s4"]
    sformed = [nm for nm in size_members if F(nm)]
    fits["size_formed_certified"] = sformed
    if len(sformed) >= 4:
        fits["size_free"] = "to be fitted (4 members): not reached"
    if len(sformed) >= 2:
        chi3 = [(1.0 / S[nm]["meta"]["rs"]) ** (1.0 / 3.0) for nm in sformed]
        fits["size_fixed_third"] = fit_block("Y = c0 + c1 (a/r_s)^(1/3) on the certified size members", sformed, [np.ones(len(sformed)), chi3],
                                             {"T_f": [F(nm)["T"] for nm in sformed], "R_t_over_rs": [F(nm)["R_t_over_rs"] for nm in sformed],
                                              "R_t_over_2gE": [F(nm)["R_t_over_2gE"] for nm in sformed]}, {"x": chi3})
    # the same with the uncertified formation records of size members beside (labelled; not a law)
    sall = [nm for nm in size_members if S[nm]["formation"]]
    if len(sall) > len(sformed) and len(sall) >= 3:
        chi3 = [(1.0 / S[nm]["meta"]["rs"]) ** (1.0 / 3.0) for nm in sall]
        fits["size_fixed_third_with_uncertified"] = fit_block(
            "Y = c0 + c1 (a/r_s)^(1/3) including uncertified formation records (labelled; excluded from every law)", sall,
            [np.ones(len(sall)), chi3], {"T_f": [S[nm]["formation"]["T"] for nm in sall], "R_t_over_rs": [S[nm]["formation"]["R_t_over_rs"] for nm in sall],
                                         "R_t_over_2gE": [S[nm]["formation"]["R_t_over_2gE"] for nm in sall]},
            {"x": chi3, "uncertified": [nm for nm in sall if nm not in sformed]})
    # ---------------- WIDTH cross: model 1, z, h, zh with z = ln(a/r_s), h = ln(sigma/a)
    wpts = ["P64_s2", "P64_s4", "P64_s8", "S4_128_s4", "S4_128_s8", "S4_128_s16"]

    def zh(nm):
        return math.log(1.0 / S[nm]["meta"]["rs"]), math.log(S[nm]["meta"]["sigma"])
    wc = [nm for nm in wpts if F(nm)]
    if len(wc) >= 4:
        Z = [zh(nm)[0] for nm in wc]; Hh = [zh(nm)[1] for nm in wc]
        fits["width_cross_certified"] = fit_block("Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a)", wc,
                                                  [np.ones(len(wc)), Z, Hh, np.multiply(Z, Hh)],
                                                  {"T_f": [F(nm)["T"] for nm in wc], "R_t_over_rs": [F(nm)["R_t_over_rs"] for nm in wc],
                                                   "R_t_over_2gE": [F(nm)["R_t_over_2gE"] for nm in wc]}, {"z": Z, "h": Hh})
    wa = [nm for nm in wpts if S[nm]["formation"]]
    if len(wa) > len(wc) and len(wa) >= 4:
        Z = [zh(nm)[0] for nm in wa]; Hh = [zh(nm)[1] for nm in wa]
        fits["width_cross_with_uncertified"] = fit_block("Y = c0 + c1 z + c2 h + c3 z h including uncertified formation records (labelled)", wa,
                                                         [np.ones(len(wa)), Z, Hh, np.multiply(Z, Hh)],
                                                         {"T_f": [S[nm]["formation"]["T"] for nm in wa],
                                                          "R_t_over_rs": [S[nm]["formation"]["R_t_over_rs"] for nm in wa],
                                                          "R_t_over_2gE": [S[nm]["formation"]["R_t_over_2gE"] for nm in wa]},
                                                         {"z": Z, "h": Hh, "uncertified": [nm for nm in wa if nm not in wc]})
    # ---------------- each fit's value at the members it does not contain (the non-formed or uncertified members; labelled in the tables)
    def basis(key, nm):
        mm = S[nm]["meta"]
        if key.startswith("mass_fixed_iso"):
            return [1.0, S[nm]["counts"]["x_iso"]]
        if key.startswith("mass_fixed_own"):
            return [1.0, S[nm]["counts"]["x_own"]] if S[nm]["counts"]["x_own"] is not None else None
        if key.startswith("size_fixed_third"):
            return [1.0, (1.0 / mm["rs"]) ** (1.0 / 3.0)]
        if key.startswith("width_cross"):
            z, h = math.log(1.0 / mm["rs"]), math.log(mm["sigma"])
            return [1.0, z, h, z * h]
        return None
    targets = {"mass_fixed_iso": ["M3a_s4", "M3d_s4"], "mass_fixed_own": ["M3a_s4", "M3d_s4"],
               "size_fixed_third": ["S4_16_s4", "S4_128_s4"], "size_fixed_third_with_uncertified": ["S4_16_s4"],
               "width_cross_certified": ["S4_128_s4"]}
    for key, tl in targets.items():
        Fk = fits.get(key)
        if not isinstance(Fk, dict) or Fk["dof"] < 1:
            continue
        pa = {}
        for nm in tl:
            b = basis(key, nm)
            if b is None:
                continue
            fo = S[nm]["formation"]
            pa[nm] = {"basis": b, "record": ({"T_f": fo["T"], "R_t_over_rs": fo["R_t_over_rs"], "R_t_over_2gE": fo["R_t_over_2gE"],
                                              "certified": fo["certified"]} if fo else {"none_to": S[nm]["convergence"]["last"]})}
            for y in ("T_f", "R_t_over_rs", "R_t_over_2gE"):
                if y in Fk:
                    pa[nm][y] = float(np.dot(Fk[y]["coef"], b))
        Fk["value_at_other_members"] = pa
    # ---------------- control spreads (member minus control) at formation and at matched clocks
    spreads = []
    for name, _, _, role in MEMBERS:
        if ":" not in role:
            continue
        kind, of = role.split(":")
        A_, B_ = S[of], S[name]
        fa, fb = A_["formation"], B_["formation"]
        row = {"member": of, "control": name, "kind": kind}
        if fa and fb:
            row.update({"dT_f": fb["T"] - fa["T"], "dR_t": fb["R_t"] - fa["R_t"], "dR_t_over_rs": fb["R_t_over_rs"] - fa["R_t_over_rs"],
                        "dR_t_over_2gE": (fb["R_t_over_2gE"] - fa["R_t_over_2gE"]) if (fb["R_t_over_2gE"] is not None and fa["R_t_over_2gE"] is not None) else None,
                        "both_certified": bool(fa["certified"] and fb["certified"])})
        # matched clocks common to both strict prefixes: max |d E/E0|, |d min(N - |v|)|, |d R_c|/r_s, R_t differences where both hold a zero
        ta = arrays["%s__T" % of]; tb = arrays["%s__T" % name]
        ea = A_["convergence"]["strict_prefix_end"]; eb = B_["convergence"]["strict_prefix_end"]
        if ea is not None and eb is not None:
            hi = min(ea, eb)
            common = sorted(set(np.round(ta[ta <= hi + 1e-9], 6)) & set(np.round(tb[tb <= hi + 1e-9], 6)))
            ia = {round(float(t), 6): i for i, t in enumerate(ta)}; ib = {round(float(t), 6): i for i, t in enumerate(tb)}
            dE, dM, dR, dRt = [], [], [], []
            for t in common:
                i, j = ia[t], ib[t]
                dE.append(arrays["%s__EE0" % name][j] - arrays["%s__EE0" % of][i])
                dM.append(arrays["%s__minNv" % name][j] - arrays["%s__minNv" % of][i])
                dR.append((arrays["%s__Rc" % name][j] - arrays["%s__Rc" % of][i]) / S[of]["meta"]["rs"])
                if arrays["%s__nzeros" % name][j] > 0 and arrays["%s__nzeros" % of][i] > 0:
                    dRt.append((t, arrays["%s__Rt" % name][j] - arrays["%s__Rt" % of][i]))
            def mx(v):
                v = np.asarray(v, float)
                return (float(np.nanmax(np.abs(v))) if v.size and np.any(np.isfinite(v)) else None)
            row.update({"common_clocks": len(common), "common_to": (common[-1] if common else None), "max_abs_dEE0": mx(dE),
                        "max_abs_dminNv": mx(dM), "max_abs_dRc_over_rs": mx(dR), "Rt_common_count": len(dRt),
                        "max_abs_dRt": (max(abs(x[1]) for x in dRt) if dRt else None)})
        spreads.append(row)
    out = {"members": S, "fits": fits, "spreads": spreads, "order": [m[0] for m in MEMBERS],
           "definitions": __doc__.split("Definitions")[1].split("Usage")[0]}
    jp = os.path.join(a.out_dir, "ladders_readout_v1.json")
    json.dump(out, open(jp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    npz = os.path.join(a.out_dir, "ladders_readout_v1.npz")
    np.savez_compressed(npz, **arrays)
    with open(os.path.join(a.out_dir, "MANIFEST_ladders_readout_v1.md5"), "w") as f:
        for p in (jp, npz):
            f.write("%s  %s\n" % (md5sum(p), os.path.basename(p)))
    print("wrote", jp, npz)


if __name__ == "__main__":
    main()
