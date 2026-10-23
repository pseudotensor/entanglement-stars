#!/usr/bin/env python
"""ladders_readout_v1_2.py (composition builder, the solve lane): the producer of MASS_ladder_v1.2 and SCALING_ladder_v1.2, version 1's
producer with the corrections of deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md (the certified prefix with propagation, the
confirming-residual audit, the certified flag per record, the bracket of each formation, the fits relabelled as diagnostics, the 3-point
exact power forms, the design rank, the charge ratios); see summarize's docstring for the rules, which supersede the definitions below.

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
  a record COUNTS when its converged entry is True (a Newton record without the entry: its kept residual), its residual is at most 1e-3
  and, for a Newton record, its confirming residual is at most the engine's stop_fields; a capped record, a record above 1e-3 or a
  Newton record failing the confirming test does not count and is named. Two extents print for every member: the 30.3 prefix, ending
  before the first record above 1e-3, and the certified prefix, ending before the first non-counting record of any kind, since a record
  kept unconverged propagates into every later record until a replay from the last certified record. The last converged record is the
  last record of the certified prefix. A formation is certified when its record lies inside the certified prefix, and only such
  formations enter a fit; a formation inside the 30.3 prefix but after a non-counting record prints as 'certified under the 30.3 rule
  only' and enters no fit;
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
    ("S4_128_s8", lambda: lad("r1", "S4_128_s8"), "(0.2, 128) sigma/a 8 tau 8 (finished to 40; its final run json mirrored from R1)", "control_sigma:S4_128_s4"),
    ("S4_128_s16", lambda: lad("p8", "S4_128_s16"), "(0.2, 128) sigma/a 16 tau 8", "control_sigma:S4_128_s4"),
    ("S4_R2_m0p2", lambda: lad("p8", "S4_R2_m0p2"), "(0.2, 64) sigma/a 4 tau 8, release at 2 r_s", "preparation:P64_s4"),
    ("S4_R2_m0p1", lambda: lad("p8", "S4_R2_m0p1"), "(0.1, 64) sigma/a 4 tau 8, release at 2 r_s", "preparation:M3a_s4"),
]
RUNNING = set()   # v1.2: S4_128_s8 finished to T/r_s 40 on R1 (queue one_S4_128_s8, 2026-10-19T23:14:48Z); read from its final run json
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
    raw = J.get("records") or []
    recs = [r for r in raw if isinstance(r, dict) and "T_over_rs" in r and r.get("E") is not None]
    n_dropped = len(raw) - len(recs)
    stop = float(A.get("stop_fields") or 1e-6)
    n = len(recs)
    cols = {k: np.full(n, np.nan) for k in ("T", "EE0", "E", "Rt", "Rt_rs", "Rt_2gE", "minNv", "r_minNv", "Rc", "res", "conf", "slope",
                                             "four_gE_kappa_T")}
    ints = {k: np.zeros(n, dtype=np.int64) for k in ("conv", "counts", "nzeros", "ntang", "pockets", "inversions", "capped", "conv_stored",
                                                      "conf_fail", "above")}
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
        ints["conv_stored"][i] = 1 if r.get("converged") is True else 0
        ints["capped"][i] = 1 if r.get("converged") is False else 0
        # version 1.1 (reading R2): a Newton record's confirming residual is tested against the engine's stop rule (stop_fields)
        cf = cols["conf"][i]
        ints["conf_fail"][i] = 1 if (newton and np.isfinite(cf) and cf > stop) else 0
        ints["above"][i] = 1 if (res is not None and res > 1e-3) else 0
        ok = conv and (res is not None and res <= 1e-3) and not ints["conf_fail"][i]
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
            "records": n, "records_dropped_without_E": n_dropped, "stop_fields": stop, "stop_E": A.get("stop_E"),
            "load_seconds": round(time.time() - t0, 1)}
    meta["m"], meta["rs"], meta["sigma"], meta["tau"] = m, rs, ell, float(A["tau"])
    meta["R0_over_rs"] = float(A.get("R0_over_rs") or 4.0)
    meta["nsites"] = int(A["nsites"])
    return meta, cols, ints, zeros_all


C_E_STATIC = 0.135858   # Sec. V's c_E (the comparator's maximum mass E_max = c_E/(g^(3/2) m^2), lattice units)


def gate_of(ints, j):
    g = []
    if ints["capped"][j]:
        g.append("capped (converged False)")
    if ints["above"][j]:
        g.append("residual above 1e-3")
    if ints["conf_fail"][j]:
        g.append("confirming residual above the stop rule")
    if not g and not ints["counts"][j]:
        g.append("not converged")
    return "; ".join(g)


def summarize(meta, cols, ints, zeros_all):
    """version 1.1 (the correctness reading R1 to R3, R6, M1, M10, M12, M16 and reviewer B's R1 to R4, R7, M3): two extents per member, the
    30.3 prefix (before the first record above 1e-3) and the certified prefix (before the first non-counting record of any kind: capped,
    above 1e-3, or a Newton record whose confirming residual exceeds the stop rule), a record kept unconverged propagating into every
    later record until a replay; the last converged record is the certified prefix's last record; a formation is certified when its
    record lies inside the certified prefix; every readout carries the record's own counting flag and its certified flag."""
    T = cols["T"]
    n = len(T)
    nb = meta["head"].get("N_B")
    nb = float(nb) if nb is not None else meta["rs"] / (2 * G * meta["m"])
    m, rs, sig = meta["m"], meta["rs"], meta["sigma"]
    kmax = C_ISO * nb ** (1.0 / 3.0)
    E0 = fnum(meta["head"].get("E0"))
    Emax = C_E_STATIC / (G ** 1.5 * m * m)
    counts = {"N_B": nb, "N_B_rule": rs / (2 * G * m), "E0": E0, "f_kin": fnum(meta["head"].get("f_kin")),
              "E_over_Emax": nb * m ** 3 / NMAX_COEF, "E_max": Emax, "E0_over_Emax": (E0 / Emax) if E0 else None,
              "two_gE0": (2 * G * E0) if E0 else None, "E0_over_NBm": (E0 / (nb * m)) if E0 else None, "kappa_max_iso": kmax,
              "p_F_over_m_iso": kmax / (meta["R0_over_rs"] * rs * m), "onset_iso_a": 4.0 * kmax / math.pi, "onset_iso_rs": 4.0 * kmax / math.pi / rs,
              "excl_floor_a": C_EXCL * nb ** (1.0 / 3.0), "tau_over_rs": meta["tau"] / rs, "sigma_over_rs": sig / rs}
    counts["pressure_corr_iso"] = counts["p_F_over_m_iso"] ** 2 / 5.0
    counts["x_iso"] = counts["p_F_over_m_iso"] ** 2
    counts["x_own"] = (counts["f_kin"] / 0.3) if counts["f_kin"] else None
    counts["p_F_over_m_own"] = math.sqrt(counts["x_own"]) if counts["x_own"] else None
    nc = np.where(ints["counts"] == 0)[0]
    above = np.where(ints["above"] == 1)[0]
    r303_end = (n - 1) if len(above) == 0 else (above[0] - 1)
    cert_end = (n - 1) if len(nc) == 0 else (nc[0] - 1)
    inside = np.where(ints["counts"][:r303_end + 1] == 1)[0] if r303_end >= 0 else np.array([], dtype=int)
    r303_last_counting = int(inside[-1]) if len(inside) else -1
    certified = np.arange(n) <= cert_end
    excluded = [{"T": float(T[i]), "residual": fnum(cols["res"][i]), "converged": bool(ints["conv"][i]), "confirming": fnum(cols["conf"][i]),
                 "gate": gate_of(ints, i)} for i in nc]
    resmax_i = int(np.nanargmax(cols["res"])) if np.any(np.isfinite(cols["res"])) else None
    newton = meta["newton"]
    cf = cols["conf"]
    conv = {"records": n, "records_dropped_without_E": meta.get("records_dropped_without_E"), "first": float(T[0]) if n else None,
            "last": float(T[-1]) if n else None, "converged_true_stored": int(ints["conv_stored"].sum()), "capped": int(ints["capped"].sum()),
            "counting": int(ints["counts"].sum()), "conf_fail_count": int(ints["conf_fail"].sum()),
            "conf_fail_clocks": [float(T[i]) for i in np.where(ints["conf_fail"] == 1)[0]],
            "conf_max": (fnum(np.nanmax(cf)) if newton and np.any(np.isfinite(cf)) else None),
            "conf_max_at": (float(T[int(np.nanargmax(cf))]) if newton and np.any(np.isfinite(cf)) else None),
            "stop_fields": meta.get("stop_fields"),
            "last_converged": float(T[cert_end]) if cert_end >= 0 else None, "certified_prefix_end": float(T[cert_end]) if cert_end >= 0 else None,
            "first_noncounting": float(T[nc[0]]) if len(nc) else None, "first_noncounting_gate": gate_of(ints, nc[0]) if len(nc) else None,
            "r303_prefix_end": float(T[r303_end]) if r303_end >= 0 else None,
            "r303_last_counting": float(T[r303_last_counting]) if r303_last_counting >= 0 else None,
            "first_above_1e-3": float(T[above[0]]) if len(above) else None, "above_1e-3_count": int(len(above)), "excluded": excluded,
            "residual_max": fnum(cols["res"][resmax_i]) if resmax_i is not None else None,
            "residual_max_at": float(T[resmax_i]) if resmax_i is not None else None,
            "residual_max_certified": fnum(np.nanmax(cols["res"][:cert_end + 1])) if cert_end >= 0 else None,
            "below_1e-4": int(np.sum(np.nan_to_num(cols["res"], nan=1.0) < 1e-4)),
            "band_1e-4_1e-3": int(np.sum((np.nan_to_num(cols["res"], nan=0) >= 1e-4) & (np.nan_to_num(cols["res"], nan=0) <= 1e-3)))}
    zi = np.where(ints["nzeros"] > 0)[0]
    form = None
    episodes = []
    if len(zi):
        i = int(zi[0])
        if i <= cert_end:
            lab, cl = "certified (inside the certified prefix)", "certified"
        elif i <= r303_end and ints["counts"][i]:
            lab, cl = ("certified under the 30.3 rule only; after a non-counting record at %.4f (%s)" % (T[nc[0]], gate_of(ints, nc[0])), "30.3 only")
        elif i > r303_end:
            lab, cl = "uncertified: after the member's first record above 1e-3 (%.4f)" % T[above[0]], "uncertified"
        else:
            lab, cl = "uncertified: the formation record itself does not count (%s)" % gate_of(ints, i), "uncertified"
        # the bracket and the zero of the stored min(N - |v|) series between the records (a diagnostic, not an event location)
        br = {"prev_T": float(T[i - 1]) if i > 0 else None, "prev_minNv": fnum(cols["minNv"][i - 1]) if i > 0 else None,
              "width": float(T[i] - T[i - 1]) if i > 0 else None, "T_lin": None, "T_quad": None}
        if i > 0 and np.isfinite(cols["minNv"][i - 1]) and np.isfinite(cols["minNv"][i]) and cols["minNv"][i - 1] > 0 > cols["minNv"][i]:
            a0, a1 = cols["minNv"][i - 1], cols["minNv"][i]
            br["T_lin"] = float(T[i - 1] + (T[i] - T[i - 1]) * a0 / (a0 - a1))
            if i > 1 and np.isfinite(cols["minNv"][i - 2]):
                cq = np.polyfit(T[i - 2:i + 1], cols["minNv"][i - 2:i + 1], 2)
                rts = [float(np.real(x)) for x in np.roots(cq) if abs(np.imag(x)) < 1e-12 and T[i - 1] - 1e-12 <= np.real(x) <= T[i] + 1e-12]
                br["T_quad"] = rts[0] if len(rts) == 1 else None
        form = {"T": float(T[i]), "index": i, "R_t": fnum(cols["Rt"][i]), "R_t_over_rs": fnum(cols["Rt_rs"][i]), "R_t_over_2gE": fnum(cols["Rt_2gE"][i]),
                "E_over_E0": fnum(cols["EE0"][i]), "minNv": fnum(cols["minNv"][i]), "R_c": fnum(cols["Rc"][i]), "zeros": zeros_all[i],
                "n_zeros": int(ints["nzeros"][i]), "tangencies": int(ints["ntang"][i]), "slope": fnum(cols["slope"][i]),
                "residual": fnum(cols["res"][i]), "confirming": fnum(cols["conf"][i]), "converged": bool(ints["conv"][i]),
                "counts": bool(ints["counts"][i]), "gate_sigma_over_Rt": sig / cols["Rt"][i], "certified": bool(cl == "certified"), "class": cl,
                "label": lab, "simple": bool(ints["ntang"][i] == 0 and np.isfinite(cols["slope"][i]) and cols["slope"][i] > 0),
                "four_gE_kappa_T": fnum(cols["four_gE_kappa_T"][i]), "bracket": br,
                "prev_record_T": br["prev_T"], "prev_minNv": br["prev_minNv"]}
        cur = None
        for j in range(n):
            if ints["nzeros"][j] > 0:
                if cur is None:
                    cur = {"first": float(T[j]), "last": float(T[j]), "records": 0, "min_minNv": None, "min_at": None, "min_cert": None, "min_cert_at": None,
                           "R_t_first": fnum(cols["Rt"][j]), "R_t_last": None, "closes_at": None, "noncounting_inside": 0, "uncertified_inside": 0}
                cur["last"] = float(T[j]); cur["records"] += 1; cur["R_t_last"] = fnum(cols["Rt"][j])
                if ints["counts"][j] == 0:
                    cur["noncounting_inside"] += 1
                if not certified[j]:
                    cur["uncertified_inside"] += 1
                mv = cols["minNv"][j]
                if np.isfinite(mv) and (cur["min_minNv"] is None or mv < cur["min_minNv"]):
                    cur["min_minNv"], cur["min_at"] = float(mv), float(T[j])
                if np.isfinite(mv) and certified[j] and (cur["min_cert"] is None or mv < cur["min_cert"]):
                    cur["min_cert"], cur["min_cert_at"] = float(mv), float(T[j])
            elif cur is not None:
                cur["closes_at"] = float(T[j]); cur["closes_certified"] = bool(certified[j]); episodes.append(cur); cur = None
        if cur is not None:
            episodes.append(cur)
    rc = {}
    if cert_end >= 0:
        seg = cols["Rc"][:cert_end + 1]
        if np.any(np.isfinite(seg)):
            k = int(np.nanargmin(seg))
            rc = {"min_Rc": float(seg[k]), "at": float(T[k]), "min_Rc_over_rs": float(seg[k]) / rs, "over_end": (k == cert_end)}

    def row_at(t):
        j = np.where(np.abs(T - t) < 1e-6)[0]
        if not len(j):
            return None
        j = int(j[0])
        return {"T": float(T[j]), "EE0": fnum(cols["EE0"][j]), "R_t": fnum(cols["Rt"][j]), "R_t_over_rs": fnum(cols["Rt_rs"][j]),
                "R_t_over_2gE": fnum(cols["Rt_2gE"][j]), "minNv": fnum(cols["minNv"][j]), "R_c": fnum(cols["Rc"][j]),
                "R_c_over_rs": fnum(cols["Rc"][j] / rs), "residual": fnum(cols["res"][j]), "counts": bool(ints["counts"][j]),
                "certified": bool(certified[j]), "gate": fnum(sig / cols["Rt"][j]) if np.isfinite(cols["Rt"][j]) else None,
                "pockets": int(ints["pockets"][j])}
    matched = {"%g" % t: row_at(float(t)) for t in range(0, 41)}
    rel = {}
    if form:
        for d in (0.0, 0.5, 1.0, 2.0, 4.0):
            rel["%g" % d] = row_at(form["T"] + d)
    return {"counts": counts, "convergence": conv, "formation": form, "episodes": episodes, "Rc_min": rc, "matched": matched,
            "matched_from_formation": rel, "cert_end_index": cert_end}


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


def power3(xs, ys):
    """the exact 3-point form Y = Y0 + A x^p through 3 points (sorted by x): every real root p in [-12, 12] with its Y0 and A, or none."""
    o = np.argsort(xs)
    x = np.asarray(xs, float)[o]; y = np.asarray(ys, float)[o]
    if abs(y[1] - y[0]) < 1e-300:
        return []
    r = (y[2] - y[1]) / (y[1] - y[0])

    def g(p):
        if abs(p) < 1e-9:
            return (math.log(x[2]) - math.log(x[1])) / (math.log(x[1]) - math.log(x[0])) - r
        return (x[2] ** p - x[1] ** p) / (x[1] ** p - x[0] ** p) - r
    grid = np.linspace(-12, 12, 4801)
    vals = [g(p) for p in grid]
    roots = []
    for k in range(len(grid) - 1):
        if np.isfinite(vals[k]) and np.isfinite(vals[k + 1]) and vals[k] * vals[k + 1] < 0:
            lo, hi = grid[k], grid[k + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if g(lo) * g(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            p = 0.5 * (lo + hi)
            if abs(p) < 1e-6:
                continue
            A = (y[1] - y[0]) / (x[1] ** p - x[0] ** p)
            roots.append({"p": float(p), "A": float(A), "Y0": float(y[0] - A * x[0] ** p)})
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snapshot", action="append", default=[], help="NAME=PATH: read a running member from this stored snapshot (version 1's)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    snaps = dict(x.split("=", 1) for x in a.snapshot)
    cat = {}
    for c in a.catalog:
        cat.update(json.load(open(c)))
    S, arrays = {}, {}
    for name, pf, label, role in MEMBERS:
        path = snaps.get(name) or pf()
        meta, cols, ints, zeros_all = load_member(name, path, label, role, None)
        if name in snaps:
            meta["snapshot_of"] = pf()
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
        S[name] = {"meta": meta, **{k: v for k, v in summ.items() if k != "cert_end_index"}}
        S[name]["cert_end_index"] = summ["cert_end_index"]
        for k, v in cols.items():
            arrays["%s__%s" % (name, k)] = v
        for k, v in ints.items():
            arrays["%s__%s" % (name, k)] = v
        arrays["%s__certified" % name] = (np.arange(len(cols["T"])) <= summ["cert_end_index"]).astype(np.int64)
        print("%-18s %4d recs %7.3f->%7.3f certified to %s form %s %s" % (name, meta["records"], summ["convergence"]["first"], summ["convergence"]["last"],
              summ["convergence"]["last_converged"], (summ["formation"] or {}).get("T"), (summ["formation"] or {}).get("class")), flush=True)

    def F(nm):
        f = S[nm]["formation"]
        return f if (f and f["certified"]) else None

    def Fany(nm):
        return S[nm]["formation"]

    def ys_of(names, getf):
        return {"T_f": [getf(nm)["T"] for nm in names], "R_t_over_rs": [getf(nm)["R_t_over_rs"] for nm in names],
                "R_t_over_2gE": [getf(nm)["R_t_over_2gE"] for nm in names]}
    fits = {}
    mass_members = ["M3b_s4", "P64_s4", "M3a_s4", "M3c_s4_nk", "M3d_s4"]
    formed = [nm for nm in mass_members if F(nm)]
    fits["mass_formed_certified"] = formed
    if len(formed) >= 2:
        xi = [S[nm]["counts"]["x_iso"] for nm in formed]
        fits["mass_diag_iso"] = fit_block("2-coefficient diagnostic Y = c0 + c1 x, x = (p_F/m)^2 of the isotropic release form", formed,
                                          [np.ones(len(formed)), xi], ys_of(formed, F), {"x": xi})
        xo = [S[nm]["counts"]["x_own"] for nm in formed]
        if all(v is not None for v in xo):
            fits["mass_diag_own"] = fit_block("2-coefficient diagnostic Y = c0 + c1 x, x = f_kin/0.3 (an equivalent cold-isotropic kinetic parameter "
                                              "of each stored preparation, the composer's cold-gas conversion)", formed, [np.ones(len(formed)), xo],
                                              ys_of(formed, F), {"x": xo})
        if len(formed) == 3:
            fits["mass_power3_iso"] = {k: power3(xi, v) for k, v in ys_of(formed, F).items()}
            fits["mass_power3_iso"]["members"] = formed
    size_members = ["S4_16_s4", "RS32_s4", "P64_s4", "S4_128_s4"]
    sformed = [nm for nm in size_members if F(nm)]
    fits["size_formed_certified"] = sformed
    if len(sformed) >= 2:
        chi3 = [(1.0 / S[nm]["meta"]["rs"]) ** (1.0 / 3.0) for nm in sformed]
        fits["size_diag_third"] = fit_block("2-coefficient diagnostic Y = c0 + c1 (a/r_s)^(1/3) on the certified size members", sformed,
                                            [np.ones(len(sformed)), chi3], ys_of(sformed, F), {"x": chi3})
    sall = [nm for nm in size_members if Fany(nm)]
    if len(sall) > len(sformed) and len(sall) >= 3:
        chi3 = [(1.0 / S[nm]["meta"]["rs"]) ** (1.0 / 3.0) for nm in sall]
        fits["size_diag_third_with_uncertified"] = fit_block(
            "2-coefficient diagnostic Y = c0 + c1 (a/r_s)^(1/3) including uncertified formation records (labelled; no law)", sall,
            [np.ones(len(sall)), chi3], ys_of(sall, Fany), {"x": chi3, "uncertified": [nm for nm in sall if nm not in sformed]})
        if len(sall) == 3:
            xs = [1.0 / S[nm]["meta"]["rs"] for nm in sall]
            fits["size_power3_with_uncertified"] = {k: power3(xs, v) for k, v in ys_of(sall, Fany).items()}
            fits["size_power3_with_uncertified"]["members"] = sall
    wpts = ["P64_s2", "P64_s4", "P64_s8", "S4_128_s4", "S4_128_s8", "S4_128_s16"]

    def zh(nm):
        return math.log(1.0 / S[nm]["meta"]["rs"]), math.log(S[nm]["meta"]["sigma"])
    for key, sel, lab in (("width_cross_certified", [nm for nm in wpts if F(nm)], "the certified points"),
                          ("width_cross_with_s2", [nm for nm in wpts if F(nm) or nm == "P64_s2"], "the certified points and the sigma/a 2 composite (labelled)"),
                          ("width_cross_with_uncertified", [nm for nm in wpts if Fany(nm)], "every formation record (labelled)")):
        if len(sel) >= 4 and (key == "width_cross_certified" or set(sel) != set(fits.get("width_cross_certified", {}).get("members", []))):
            Z = [zh(nm)[0] for nm in sel]; Hh = [zh(nm)[1] for nm in sel]
            fits[key] = fit_block("Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a), on %s" % lab, sel,
                                  [np.ones(len(sel)), Z, Hh, np.multiply(Z, Hh)], ys_of(sel, Fany), {"z": Z, "h": Hh,
                                  "uncertified": [nm for nm in sel if not F(nm)]})
    # the width dependence per doubling of sigma/a at each size (the records' numbers)
    wd = []
    for rs_, pairs in ((64, (("P64_s2", "P64_s4"), ("P64_s4", "P64_s8"))), (128, (("S4_128_s4", "S4_128_s8"), ("S4_128_s8", "S4_128_s16")))):
        for p_, q_ in pairs:
            fp, fq = Fany(p_), Fany(q_)
            if fp and fq:
                wd.append({"r_s/a": rs_, "from": p_, "to": q_, "dT_f": fq["T"] - fp["T"], "dR_t_over_rs": fq["R_t_over_rs"] - fp["R_t_over_rs"],
                           "dR_t_over_2gE": fq["R_t_over_2gE"] - fp["R_t_over_2gE"], "both_certified": bool(fp["certified"] and fq["certified"])})
    fits["width_per_doubling"] = wd
    # the design rank on the formed rows at m a 0.2, release 4 r_s: basis [1, z, x^2, h, zh, j, j^2, zj], x = 64 a/r_s, z = x^(4/3),
    # h = (16 sigma/r_s)^2 (1 at (64, 4)), j = 8 tau/r_s (1 at tau 8, r_s/a 64); the rank does not depend on the columns' normalization
    rank_rows = ["RS32_s4", "RS32_s4_tau1", "RS32_s2", "P64_s4", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2", "S4_128_s4", "S4_128_s8", "S4_128_s16"]

    def brow(nm):
        mm = S[nm]["meta"]; x = 64.0 / mm["rs"]; z = x ** (4.0 / 3.0); h = (16.0 * mm["sigma"] / mm["rs"]) ** 2; j = 8.0 * mm["tau"] / mm["rs"]
        return [1.0, z, x * x, h, z * h, j, j * j, z * j]
    ranks = {}
    for key, sel in (("certified", [nm for nm in rank_rows if F(nm)]), ("every_formation_record", [nm for nm in rank_rows if Fany(nm)])):
        M = np.array([brow(nm) for nm in sel])
        sv = np.linalg.svd(M, compute_uv=False) if len(sel) else np.array([])
        rk = int(np.sum(sv > sv[0] * 1e-10)) if len(sv) else 0
        ranks[key] = {"rows": sel, "rows_rs_sigma_tau": [[S[nm]["meta"]["rs"], S[nm]["meta"]["sigma"], S[nm]["meta"]["tau"]] for nm in sel],
                      "rank": rk, "columns": 8, "singular_values": [float(v) for v in sv], "residual_dof": len(sel) - rk}
    fits["design_rank"] = ranks

    def basis(key, nm):
        mm = S[nm]["meta"]
        if key.startswith("mass_diag_iso"):
            return [1.0, S[nm]["counts"]["x_iso"]]
        if key.startswith("mass_diag_own"):
            return [1.0, S[nm]["counts"]["x_own"]] if S[nm]["counts"]["x_own"] is not None else None
        if key.startswith("size_diag_third"):
            return [1.0, (1.0 / mm["rs"]) ** (1.0 / 3.0)]
        if key.startswith("width_cross"):
            z, h = math.log(1.0 / mm["rs"]), math.log(mm["sigma"])
            return [1.0, z, h, z * h]
        return None
    targets = {"mass_diag_iso": ["M3a_s4", "M3d_s4"], "mass_diag_own": ["M3a_s4", "M3d_s4"], "size_diag_third_with_uncertified": ["S4_16_s4"],
               "width_cross_with_s2": ["S4_128_s4"], "width_cross_certified": ["S4_128_s4", "P64_s2"]}
    for key, tl in targets.items():
        Fk = fits.get(key)
        if not isinstance(Fk, dict):
            continue
        pa = {}
        for nm in tl:
            if nm in Fk["members"]:
                continue
            b = basis(key, nm)
            if b is None:
                continue
            fo = S[nm]["formation"]
            pa[nm] = {"basis": b, "record": ({"T_f": fo["T"], "R_t_over_rs": fo["R_t_over_rs"], "R_t_over_2gE": fo["R_t_over_2gE"],
                                              "class": fo["class"]} if fo else {"none_to": S[nm]["convergence"]["last"]})}
            for y in ("T_f", "R_t_over_rs", "R_t_over_2gE"):
                if y in Fk:
                    pa[nm][y] = float(np.dot(Fk[y]["coef"], b))
        Fk["value_at_other_members"] = pa
    # control spreads (control minus member) at formation and at the clocks counting and certified in both histories
    pairs = []
    for name, _, _, role in MEMBERS:
        if ":" in role:
            kind, of = role.split(":")
            pairs.append((of, name, kind))
    pairs.append(("S4_128_s8", "S4_128_s16", "control_sigma (the certified factor-2 pair at 128)"))
    spreads = []
    for of, name, kind in pairs:
        A_, B_ = S[of], S[name]
        fa, fb = A_["formation"], B_["formation"]
        row = {"member": of, "control": name, "kind": kind}
        if fa and fb:
            row.update({"dT_f": fb["T"] - fa["T"], "dR_t": fb["R_t"] - fa["R_t"], "dR_t_rel": (fb["R_t"] - fa["R_t"]) / fa["R_t"],
                        "dR_t_over_rs": fb["R_t_over_rs"] - fa["R_t_over_rs"],
                        "dR_t_over_2gE": (fb["R_t_over_2gE"] - fa["R_t_over_2gE"]) if (fb["R_t_over_2gE"] is not None and fa["R_t_over_2gE"] is not None) else None,
                        "classes": "%s / %s" % (fa["class"], fb["class"]), "both_certified": bool(fa["certified"] and fb["certified"])})
        ta = arrays["%s__T" % of]; tb = arrays["%s__T" % name]
        okA = (arrays["%s__counts" % of] == 1) & (arrays["%s__certified" % of] == 1)
        okB = (arrays["%s__counts" % name] == 1) & (arrays["%s__certified" % name] == 1)
        for tag, oa, ob in (("raw_", arrays["%s__counts" % of] == 1, arrays["%s__counts" % name] == 1),):
            # the counting records of both histories regardless of certification (a labelled diagnostic beside the certified set)
            ra = {round(float(t), 6): i for i, t in enumerate(ta) if oa[i]}
            rb = {round(float(t), 6): i for i, t in enumerate(tb) if ob[i]}
            rc_ = sorted(set(ra) & set(rb))
            rr = [(t, (arrays["%s__Rt" % name][rb[t]] - arrays["%s__Rt" % of][ra[t]]) / arrays["%s__Rt" % of][ra[t]]) for t in rc_
                  if arrays["%s__nzeros" % name][rb[t]] > 0 and arrays["%s__nzeros" % of][ra[t]] > 0]
            ra_ = [abs(arrays["%s__Rt" % name][rb[t]] - arrays["%s__Rt" % of][ra[t]]) for t in rc_
                   if arrays["%s__nzeros" % name][rb[t]] > 0 and arrays["%s__nzeros" % of][ra[t]] > 0]
            rm = max(rr, key=lambda x: abs(x[1])) if rr else None
            row.update({tag + "common_clocks": len(rc_), tag + "Rt_common_count": len(rr), tag + "max_abs_dRt": (max(ra_) if ra_ else None),
                        tag + "max_rel_dRt": (abs(rm[1]) if rm else None), tag + "max_rel_dRt_at": (rm[0] if rm else None),
                        tag + "rel_dRt_at_16": next((abs(x[1]) for x in rr if abs(x[0] - 16.0) < 1e-6), None)})
        sa = {round(float(t), 6): i for i, t in enumerate(ta) if okA[i]}
        sb = {round(float(t), 6): i for i, t in enumerate(tb) if okB[i]}
        common = sorted(set(sa) & set(sb))
        dE, dM, dR, dRt, rel = [], [], [], [], []
        for t in common:
            i, j = sa[t], sb[t]
            dE.append(arrays["%s__EE0" % name][j] - arrays["%s__EE0" % of][i])
            dM.append(arrays["%s__minNv" % name][j] - arrays["%s__minNv" % of][i])
            dR.append((arrays["%s__Rc" % name][j] - arrays["%s__Rc" % of][i]) / S[of]["meta"]["rs"])
            if arrays["%s__nzeros" % name][j] > 0 and arrays["%s__nzeros" % of][i] > 0:
                d = arrays["%s__Rt" % name][j] - arrays["%s__Rt" % of][i]
                dRt.append((t, d)); rel.append((t, d / arrays["%s__Rt" % of][i]))

        def mx(v):
            v = np.asarray(v, float)
            return (float(np.nanmax(np.abs(v))) if v.size and np.any(np.isfinite(v)) else None)
        rmax = max(rel, key=lambda x: abs(x[1])) if rel else None
        row.update({"common_clocks": len(common), "common_from": (common[0] if common else None), "common_to": (common[-1] if common else None),
                    "max_abs_dEE0": mx(dE), "max_abs_dminNv": mx(dM), "max_abs_dRc_over_rs": mx(dR), "Rt_common_count": len(dRt),
                    "max_abs_dRt": (max(abs(x[1]) for x in dRt) if dRt else None),
                    "max_rel_dRt": (abs(rmax[1]) if rmax else None), "max_rel_dRt_at": (rmax[0] if rmax else None),
                    "Rt_common_from": (dRt[0][0] if dRt else None), "Rt_common_to": (dRt[-1][0] if dRt else None)})
        spreads.append(row)
    out = {"members": S, "fits": fits, "spreads": spreads, "order": [m[0] for m in MEMBERS], "definitions": summarize.__doc__}
    jp = os.path.join(a.out_dir, "ladders_readout_v1_2.json")
    json.dump(out, open(jp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    npz = os.path.join(a.out_dir, "ladders_readout_v1_2.npz")
    np.savez_compressed(npz, **arrays)
    with open(os.path.join(a.out_dir, "MANIFEST_ladders_readout_v1_2.md5"), "w") as f:
        for p in (jp, npz):
            f.write("%s  %s\n" % (md5sum(p), os.path.basename(p)))
    print("wrote", jp, npz)


if __name__ == "__main__":
    main()
