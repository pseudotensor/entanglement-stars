#!/usr/bin/env python
"""ladder_manifest_v1.py (reviewer A, the busi lane; W11 of WEB40I_TASKS/busi.md: the ladders' manifest, box CPU, the gate of every
MASS_ladder and SCALING_ladder launch).

What it reads, and nothing else: (1) the launcher's member table, parsed from launch_ladder_v36d_v1c.sh (--launcher PATH: the member()
function's defaults line and its case entries); without the file, the built-in copy of that table (input 12 of the request), which is
checked against the file whenever the file is given (the file governs where they differ, the difference printed); (2) the requirement's
ladder rows the launcher does not carry (the (0.2, 64) production's 3 averaging scales and its tau 4 and tau 2 members, stored histories;
the size ladder's r_s/a 32 member, absent from the launcher); (3) as options, the launcher's status files (--status-dir DIR holding
status_<member>.txt as copied from the pods) for the t0 check's "dust per channel at T = 0" and "preparation:" lines, which give a running
member's actual dust catalog, and the (0.2, 64) sigma/a 4 tau 8 history's json (--formation-json PATH) for the transferred formation
radius; (4) with --prepare MEMBER --code-dir DIR, the engine's own preparation, engine_collapse_v36d.prepare_dust on the CPU backend with
the member's arguments (the levitating well, the zero-temperature filling, the projection; no record and no propagation), to read the
actual dust catalog of a member whose status file is not at hand. No engine run beyond that preparation happens here.

What it prints per member (markdown to stdout; ladder_manifest_v1.md and ladder_manifest_v1.json in --out-dir):
  N_B = r_s/(2 g m a) (13040 at (0.05, 326)); E/E_max = N_B (m a)^3/1.087;
  the isotropic Fermi ball's channel set: kappa_max = (9 pi N_B/32)^(1/3) = 0.96 N_B^(1/3) (16 states per momentum in 3 dimensions;
    k_F R_0 = kappa_max), K_dust = round(kappa_max); the semiclassical radial mode count per channel n_rad(kappa) = (kappa_max/pi)
    [sqrt(1 - x^2) - x arccos x], x = kappa/kappa_max (the WKB radial phase of the ball's channel kappa over pi); n_rad,max = kappa_max/pi,
    its kappa -> 0 value 0.3055 N_B^(1/3), the onset radius 4 a n_rad,max = 1.222 N_B^(1/3) a with 4 a n_rad(1) beside it, the exclusion
    floor 0.49 N_B^(1/3) a; the capacity check sum_kappa 16 kappa n_rad(kappa) over N_B with the chains' weights 16 kappa (it is 1/2:
    finding 1); the same ball counted with 16 kappa per channel, kappa_max' = (9 pi N_B/16)^(1/3) = 1.21 N_B^(1/3), beside it;
  the release Fermi momentum p_F/m = kappa_max/((R_0/a)(m a)), which is 0.24 (2 g m a)^(-1/3) (a/r_s)^(2/3)/(m a) at R_0 = 4 r_s, and
    the dust comparator's pressure correction (p_F/m)^2/5;
  tau/r_s and the equal-step partners; sigma/r_s;
  the root gate sigma/R_t at the transferred formation radius. The transfer: R_t^tr = f_form r_s with f_form = R_t/r_s of the (0.2, 64)
    sigma/a 4 tau 8 history at its formation record (the first record holding a zero of N - |v|: 0.66 of that record's 2GE at the
    record's own collar charge, about 0.61 r_s), the formation root's radius kept in units of r_s across the members, so
    sigma/R_t^tr = (sigma/a)/(f_form r_s/a); f_form comes from --formation-json, else --f-form, else the placeholder 0.61 and is labelled;
    inside the hydrodynamic gate's working value at <= 0.1, at its edge to 0.21, outside above;
  the launcher's own derived estimates (ladder_helper_v1.cmd_estimate's formulas): device memory n^2 (256 K + 150) bytes at the member's
    kappa_sea and at the kappa_sea the isotropic set needs, the record count and the H100-hours at the base pace 24 s per record at
    2048 sites and 10 channels scaled by K/10 and by (n/2048)^2 to (n/2048)^3;
  the flags: whether the sea set can hold the isotropic set; whether the well's filling may reach the sea set's top (the expected reach
    5 (N_B/640)^(1/3): the production's 5 at 640 scaled as any 3-dimensional filling scales; the t0 line decides); the preparation rule in
    force (the well's filling within kappa_sea: the engine's only rule);
  the actual catalog where a status file or a preparation supplies it: columns per channel (the radial modes), the top channel, the
    truncation by kappa_sea, 4 a n_rad,max of the member's own set, the fractional multiplet, the capacity fraction F kept, f_kin and p_F/m
    from it (the cold isotropic conversion f_kin = 0.3 (p_F/m)^2);
and the width cross at r_s/a 64 and 128 as 6 points for the model 1, z, h, zh; the equal-step pairs; the findings.
Usage (box CPU, conda env qi2):
  python ladder_manifest_v1.py [--launcher PATH] [--status-dir DIR] [--formation-json PATH | --f-form 0.61] [--running a,b,c | --running none]
      [--out-dir DIR] [--card-mib 81559] [--prepare MEMBER --code-dir DIR [--prep-passes 200] [--prep-tol 1e-10] [--threads 16]]"""
import argparse
import json
import math
import os
import re
import sys

PI = math.pi
G = 0.25
C_ISO = (9.0 * PI / 32.0) ** (1.0 / 3.0)      # 0.9596: kappa_max = C_ISO N_B^(1/3), 16 states per momentum in 3 dimensions
C_CHAIN = (9.0 * PI / 16.0) ** (1.0 / 3.0)    # 1.2090: the same ball counted with the chains' 16 kappa per channel
C_EXCL = (3.0 / (8.0 * PI)) ** (1.0 / 3.0)    # 0.4924: the exclusion floor's coefficient
N_MAX_COEF = 1.087                            # the continuum maximum count 1.087/(m a)^3 at g = 1/4
WELL_REACH_640 = 5.0                          # the production's highest dust channel at N_B 640 (kappa <= 5)
F_FORM_DEFAULT = 0.61
CARD_MIB_DEFAULT = 81559.0                    # an H100 80 GB card's reported device memory; the launcher wants free >= 1.3 x its estimate
NUMERIC = {"M": float, "RS": float, "N": int, "R0": float, "K": int, "TAU": float, "ELL": float, "PREP": float, "TEND": float, "COLL": float, "ORDER": int}
BUILTIN_DEFAULTS = {"M": 0.2, "RS": 64.0, "N": 2048, "R0": 4.0, "K": 10, "TAU": 8.0, "ELL": 4.0, "PREP": 4.0, "TEND": 40.0, "COLL": 16.0,
                    "ENGINE": "v36d", "EXTRA": "", "SLX": ""}
SLX_R2 = "3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10"
BUILTIN_TABLE = [
    ("M3a_s4", {"M": 0.1, "CARD": "R3", "ORDER": 1}),
    ("M3a_s2", {"M": 0.1, "ELL": 2.0, "CARD": "R3", "ORDER": 2}),
    ("M3a_s4_tau4", {"M": 0.1, "TAU": 4.0, "TEND": 16.0, "CARD": "R3", "ORDER": 3}),
    ("M3b_s4", {"M": 0.4, "CARD": "R3", "ORDER": 4}),
    ("M3b_s8", {"M": 0.4, "ELL": 8.0, "CARD": "R3", "ORDER": 5}),
    ("S4_16_s4", {"RS": 16.0, "N": 512, "TAU": 2.0, "CARD": "R3", "ORDER": 6}),
    ("S4_16_s2", {"RS": 16.0, "N": 512, "TAU": 2.0, "ELL": 2.0, "CARD": "R3", "ORDER": 7}),
    ("S4_16_s4_tau1", {"RS": 16.0, "N": 512, "TAU": 1.0, "TEND": 16.0, "CARD": "R3", "ORDER": 8}),
    ("M3c_s4", {"M": 0.1, "RS": 96.0, "N": 3072, "CARD": "R3", "ORDER": 9}),
    ("M3d_s4", {"M": 0.05, "RS": 326.0, "N": 3912, "K": 12, "TEND": 30.0, "COLL": 8.0, "CARD": "R1", "ORDER": 1, "ENGINE": "v36e",
                "EXTRA": "--first-return-collar-over-rs 8 --first-return-end-rule end", "SLX": "28"}),
    ("S4_128_s4", {"RS": 128.0, "N": 4096, "CARD": "R2", "ORDER": 1}),
    ("S4_128_s8", {"RS": 128.0, "N": 4096, "ELL": 8.0, "CARD": "R2", "ORDER": 2}),
    ("S4_128_s16", {"RS": 128.0, "N": 4096, "ELL": 16.0, "CARD": "R2", "ORDER": 3}),
    ("S4_R2_m0p2", {"R0": 2.0, "SLX": SLX_R2, "CARD": "R2", "ORDER": 4}),
    ("S4_R2_m0p1", {"M": 0.1, "R0": 2.0, "SLX": SLX_R2, "CARD": "R2", "ORDER": 5}),
]
REQ_ROWS = [   # the requirement's ladder rows the launcher does not carry: (name, overrides, status)
    ("P64_s2", {"ELL": 2.0}, "stored history: the (0.2, 64) sigma/a 2 tau 8 member of the production's averaging ladder (the solve lane's)"),
    ("P64_s4", {}, "stored history: the (0.2, 64) sigma/a 4 tau 8 production (the solve lane's)"),
    ("P64_s8", {"ELL": 8.0}, "stored history: the (0.2, 64) sigma/a 8 tau 8 member (the solve lane's)"),
    ("P64_s4_tau4", {"TAU": 4.0}, "stored history: the (0.2, 64) sigma/a 4 tau 4 step control (the solve lane's)"),
    ("P64_s4_tau2", {"TAU": 2.0}, "stored history: the (0.2, 64) sigma/a 4 tau 2 step control (the solve lane's)"),
    ("S4_32_s4", {"RS": 32.0, "N": 1024, "TAU": 2.0}, "ABSENT from the launcher: the requirement's size ladder member r_s/a 32 (N_B 320), the equal-step partner of r_s/a 128 at tau 8"),
]
LEAD_RUNNING = ["M3a_s2", "M3a_s4", "M3a_s4_tau4", "M3b_s4", "M3b_s8", "M3c_s4", "M3d_s4", "S4_128_s4", "S4_128_s16",
                "S4_16_s4", "S4_16_s2", "S4_16_s4_tau1", "S4_R2_m0p2", "S4_R2_m0p1"]
WIDTH_CROSS = [("P64_s2", 64.0, 2.0), ("P64_s4", 64.0, 4.0), ("P64_s8", 64.0, 8.0), ("S4_128_s4", 128.0, 4.0), ("S4_128_s8", 128.0, 8.0), ("S4_128_s16", 128.0, 16.0)]
ASSIGN = re.compile(r'([A-Z][A-Z0-9_]*)=("([^"]*)"|[^;\s]+)')
CASE = re.compile(r'^([A-Za-z0-9_]+)\)\s+(.*?);;')
T0_ROW = re.compile(r"kappa (\d+) \(w ([0-9.eE+-]+)\): (\d+) columns, ([0-9.]+) particles, lowest occupation ([0-9.]+|-)")
T0_HIGH = re.compile(r"highest channel holding dust (\d+) against the plan's \(N_B/8\)\^\(1/3\) rule ([0-9.]+) \((\d+)\) and the set's top (\d+); the largest per-channel level count (\d+)")
PREP_LINE = re.compile(r"preparation: N_B (\S+) against r_s/\(2 g m\) = (\S+) \(difference (\S+)\); E0 (\S+); E0/\(N_B m\) (\S+); E0/m (\S+); passes (\S+); last change (\S+); f_kin (\S+);")


def rnd(x):
    return int(math.floor(x + 0.5))


def n_rad(kappa, kmax):
    """the isotropic ball's radial mode count in channel kappa: the WKB radial phase (1/pi) int sqrt(k_F^2 - kappa^2/r^2) dr over the ball."""
    x = kappa / kmax
    if x >= 1.0:
        return 0.0
    return (kmax / PI) * (math.sqrt(1.0 - x * x) - x * math.acos(x))


def f_capacity(u):
    """the fraction of the isotropic ball's angular capacity in channels kappa <= u kappa_max (the weight kappa n_rad(kappa) integrated)."""
    u = min(max(u, 0.0), 1.0)
    s = math.sqrt(max(1.0 - u * u, 0.0))
    return 1.0 + 3.0 * s - 4.0 * s ** 3 - 3.0 * u ** 3 * math.acos(u)


def parse_assignments(s):
    out = {}
    for mm in ASSIGN.finditer(s):
        out[mm.group(1)] = mm.group(3) if mm.group(3) is not None else mm.group(2)
    return out


def typed(d):
    out = {}
    for k, v in d.items():
        if k in NUMERIC:
            try:
                out[k] = int(float(v)) if NUMERIC[k] is int else float(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


def parse_launcher(path):
    """the launcher's member(): its defaults line and its case entries (name, overrides)."""
    txt = open(path, errors="replace").read()
    i = txt.find("member() {")
    if i < 0:
        raise SystemExit("ladder_manifest_v1: no member() function in %s" % path)
    j = txt.find("\n}", i)
    body = txt[i:j if j > 0 else len(txt)].split("\n")
    defaults, table = None, []
    for ln in body[1:]:
        s = ln.strip()
        if defaults is None and s.startswith("M="):
            defaults = parse_assignments(s.split(" #")[0])
            continue
        mm = CASE.match(s)
        if mm:
            table.append((mm.group(1), parse_assignments(mm.group(2))))
    if defaults is None:
        raise SystemExit("ladder_manifest_v1: no defaults line (M=...) in member() of %s" % path)
    return typed(defaults), [(n_, typed(o_)) for n_, o_ in table]


def _num_eq(x, y):
    try:
        return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(x)))
    except (TypeError, ValueError):
        return str(x) == str(y)


def compare_tables(bdef, btab, pdef, ptab):
    lines = []
    pb, pp = dict(btab), dict(ptab)
    for k in sorted(set(pb) | set(pp)):
        if k not in pb:
            lines.append("%s: in the launcher file only (%s)" % (k, pp[k]))
            continue
        if k not in pp:
            lines.append("%s: in the built-in table only (not in the launcher file)" % k)
            continue
        a_ = dict(bdef); a_.update(pb[k]); b_ = dict(pdef); b_.update(pp[k])
        diffs = [kk for kk in sorted(set(a_) | set(b_)) if not _num_eq(a_.get(kk), b_.get(kk))]
        if diffs:
            lines.append("%s: differs in %s (file %s; built-in %s)" % (k, ",".join(diffs), {kk: b_.get(kk) for kk in diffs}, {kk: a_.get(kk) for kk in diffs}))
    return lines


def read_formation(path):
    """the (0.2, 64) sigma/a 4 tau 8 history's formation record: the first accepted record whose trapping_zeros is non-empty."""
    J = json.load(open(path))
    rs = float(J["args"]["rs"])
    for r in J["records"]:
        if r.get("E") is not None and r.get("trapping_zeros"):
            return {"T_over_rs": r["T"] / rs, "R_t": r["R_t"], "f_form": r["R_t"] / rs, "R_t_over_2gE": r.get("R_t_over_2gE"), "E_over_E0": r.get("E_over_E0"), "source": path}
    raise SystemExit("ladder_manifest_v1: no record with a zero of N - |v| in %s" % path)


def read_status(path):
    """the launcher's t0 lines of one member: the per-channel dust catalog and the preparation line (the last occurrence of each)."""
    rows, info = {}, {}
    for ln in open(path, errors="replace"):
        if "dust per channel at T = 0" in ln:
            rows = {}
            for mm in T0_ROW.finditer(ln):
                rows[int(mm.group(1))] = {"w": float(mm.group(2)), "columns": int(mm.group(3)), "particles": float(mm.group(4)),
                                          "lowest_occupation": (float(mm.group(5)) if mm.group(5) != "-" else None)}
            mh = T0_HIGH.search(ln)
            if mh:
                info.update({"top": int(mh.group(1)), "rule8_launcher": float(mh.group(2)), "K": int(mh.group(4)), "level_count_max": int(mh.group(5))})
        mp = PREP_LINE.search(ln)
        if mp:
            info["N_B"] = float(mp.group(1)); info["E0"] = float(mp.group(4)); info["passes"] = mp.group(7)
            try:
                info["f_kin"] = float(mp.group(9))
            except ValueError:
                info["f_kin"] = None
        if "the set's top channel" in ln and "holds dust" in ln:
            info["top_note"] = True
        if "t0 PASS" in ln:
            info["t0"] = "PASS"
        if "t0 FAIL" in ln:
            info["t0"] = "FAIL"
    if not rows and not info:
        return None
    info["channels"] = rows
    info["source"] = path
    return info


def prepare_member(name, d, code_dir, passes, tol, threads):
    """the engine's own preparation on the CPU backend (engine_collapse_v36d.prepare_dust with the member's arguments; no propagation)."""
    import numpy as np
    sys.path.insert(0, code_dir)
    import torch
    torch.set_num_threads(int(threads))
    import engine_collapse_v5 as C5
    import engine_collapse_v36d as E36
    C5.backend(None)
    E36.OPT["site_fields"] = "averaged"; E36.OPT["collar_policy"] = "flag"; E36.OPT["support_rel"] = 1e-14
    m = float(d["M"]); rs = float(d["RS"]); n = int(d["N"]); K = int(d["K"]); prep = int(float(d.get("PREP", 4)))
    nb = rs / (2.0 * G * m)
    chs = [C5.Chan(k, m, n, False) for k in range(1, K + 1)]
    R = chs[0].R
    N, v, dust, info, head = E36.prepare_dust(chs, G, R, prep, nb, float(d["R0"]) * rs, m, "A", None, passes=passes, tol=tol)
    rows = {}
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]
            rows[ch.kappa] = {"w": ch.w, "columns": int(Phi.shape[1]), "particles": float(ch.w * float(np.sum(fk))), "lowest_occupation": float(np.min(fk)), "f": [float(x) for x in fk]}
    top = max(rows) if rows else 0
    return {"channels": rows, "top": top, "K": K, "N_B": float(head["N_B"]), "E0": float(head["E0"]), "passes": head.get("passes"), "f_kin": float(head["f_kin"]),
            "level_count_max": max((r["columns"] for r in rows.values()), default=0), "top_note": bool(top == K), "source": "engine_collapse_v36d.prepare_dust on the CPU backend (%s)" % name}


def member_row(name, d, source, status, f_form, card_mib, t0):
    m = float(d["M"]); rs = float(d["RS"]); n = int(d["N"]); R0 = float(d["R0"]) * rs
    K = int(d["K"]); tau = float(d["TAU"]); ell = float(d["ELL"]); prep = float(d.get("PREP", 4.0)); tend = float(d["TEND"])
    nb = rs / (2.0 * G * m); nb13 = nb ** (1.0 / 3.0)
    kmax = C_ISO * nb13; Kd = rnd(kmax); kmax_c = C_CHAIN * nb13; Kd_c = rnd(kmax_c)
    rule8 = (nb / 8.0) ** (1.0 / 3.0)
    nrad = [(k, n_rad(k, kmax)) for k in range(1, Kd + 1)]
    cap16 = sum(16.0 * k * x for k, x in nrad)
    nrad_max = kmax / PI
    Rc = 4.0 * nrad_max; Rc1 = 4.0 * n_rad(1, kmax)
    floor_ex = C_EXCL * nb13
    pF = kmax / (R0 * m); pF_c = kmax_c / (R0 * m)
    tau_rs = tau / rs; sigma_rs = ell / rs
    Rt_tr = f_form * rs; gate = ell / Rt_tr
    gate_class = "inside (<= 0.1)" if gate <= 0.1 + 1e-12 else ("edge (0.1 to 0.21)" if gate <= 0.21 + 1e-12 else "outside (> 0.21)")
    mem = n * n * (256 * K + 150) / 2.0 ** 20
    K_iso = max(K, Kd); mem_iso = n * n * (256 * K_iso + 150) / 2.0 ** 20
    recs = int(round(tend * rs / tau)) + 1
    base = 24.0 * K / 10.0
    t_lo = max(5.0, base * (n / 2048.0) ** 2); t_hi = max(5.0, base * (n / 2048.0) ** 3)
    reach = WELL_REACH_640 * (nb / 640.0) ** (1.0 / 3.0)
    flags = []
    if K < Kd:
        flags.append("cannot hold the isotropic set in its sea set: kappa_sea %d below K_dust %d" % (K, Kd))
    elif K == Kd:
        flags.append("holds the isotropic set only with the sea set's top channel occupied: kappa_sea = K_dust = %d" % K)
    if 1.3 * mem_iso > card_mib:
        flags.append("the isotropic set (%d channels at %d sites) needs %.0f MiB by the launcher's estimate, and 1.3 x that exceeds the card's %.0f MiB" % (K_iso, n, mem_iso, card_mib))
    if reach > K - 1.0:
        flags.append("the well's filling may reach the sea set's top channel: expected reach %.1f against kappa_sea %d (the t0 line's highest dust channel decides; equal to kappa_sea means a catalog truncated by the sea set)" % (reach, K))
    flags.append("preparation in force: the well's filling within kappa_sea %d (prepare_dust and fill_lowest; no engine option selects a catalog), not the isotropic set" % K)
    row = {"name": name, "source": source, "status": status, "m": m, "rs": rs, "nsites": n, "R0_over_rs": R0 / rs, "kappa_sea": K, "tau": tau, "sigma_a": ell, "prep_ell": prep,
           "T_end": tend, "card": d.get("CARD"), "engine": d.get("ENGINE"), "N_B": nb, "E_over_Emax": nb * m ** 3 / N_MAX_COEF, "kappa_max": kmax, "K_dust": Kd,
           "kappa_max_chain": kmax_c, "K_dust_chain": Kd_c, "rule8": rule8, "n_rad": nrad, "capacity16_over_NB": cap16 / nb, "n_rad_max": nrad_max, "R_c": Rc,
           "R_c_from_n_rad_1": Rc1, "R_c_over_rs": Rc / rs, "exclusion_floor": floor_ex, "p_F_over_m": pF, "p_F_over_m_chain": pF_c, "pressure_corr": pF ** 2 / 5.0,
           "tau_over_rs": tau_rs, "sigma_over_rs": sigma_rs, "R_t_transferred": Rt_tr, "gate": gate, "gate_class": gate_class, "mem_MiB": mem, "K_iso": K_iso,
           "mem_MiB_iso": mem_iso, "records": recs, "hours_lo": recs * t_lo / 3600.0, "hours_hi": recs * t_hi / 3600.0, "well_reach_expected": reach, "flags": flags, "t0": None}
    if t0:
        ch = t0.get("channels") or {}
        cols = {k: v["columns"] for k, v in ch.items()}
        top = t0.get("top", max(cols) if cols else None)
        nmax = max(cols.values()) if cols else None
        frac = {k: v["lowest_occupation"] for k, v in ch.items() if v.get("lowest_occupation") is not None and v["lowest_occupation"] < 1.0 - 1e-9}
        fk = t0.get("f_kin")
        row["t0"] = {"columns": cols, "top": top, "truncated_by_kappa_sea": (top is not None and top >= K), "n_rad_max": nmax, "R_c_own": (4.0 * nmax if nmax else None),
                     "R_c_own_over_rs": (4.0 * nmax / rs if nmax else None), "fractional_multiplets": frac, "F_kept": (f_capacity(top / kmax) if top else None),
                     "f_kin": fk, "p_F_over_m_from_f_kin": (math.sqrt(fk / 0.3) if (fk is not None and fk > 0) else None), "N_B": t0.get("N_B"), "E0": t0.get("E0"),
                     "t0_verdict": t0.get("t0"), "source": t0.get("source")}
        if row["t0"]["truncated_by_kappa_sea"]:
            flags.insert(0, "CATALOG TRUNCATED BY THE SEA SET: the highest dust channel %d equals kappa_sea %d (a third preparation, neither the well's full filling nor the isotropic ball)" % (top, K))
        else:
            flags.insert(0, "the catalog is the well's full filling (highest dust channel %d below kappa_sea %d): %d radial modes at most, 4 a n_rad,max = %.0f a = %.2f r_s, F kept %.2f" % (
                top, K, nmax, 4.0 * nmax, 4.0 * nmax / rs, f_capacity(top / kmax)))
    return row


def f3(x, f="%.3f"):
    return "-" if x is None else (f % x)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render(rows, running, f_form, f_src, check_lines, table_src, card_mib):
    L = []
    L.append("# ladder_manifest_v1 (busi W11): the ladders' manifest")
    L.append("")
    L.append("Gates: no MASS_ladder or SCALING_ladder launch before this manifest; the (0.05, 326) member's channel count before its launch (W5's gate); "
             "the isotropic catalog (W12) held until the owner's scope word. The member table: %s. The transferred formation radius: f_form = %.4f (%s). "
             "Device memory and hours by the launcher's own estimates; the card %.0f MiB." % (table_src, f_form, f_src, card_mib))
    L.append("")
    if check_lines:
        L.append("Launcher table against the built-in copy of input 12: " + "; ".join(check_lines))
        L.append("")
    L.append("## Table 1: arguments and counts")
    L.append("")
    L.append(md_table(["member", "status", "m a", "r_s/a", "sites", "R_0/r_s", "kappa_sea", "tau/a", "sigma/a", "T_end", "N_B", "E/E_max", "kappa_max = 0.96 N_B^(1/3)", "K_dust", "kappa_max' (chains)", "K_dust'", "(N_B/8)^(1/3)", "expected well reach"],
                      [[r["name"], r["status"], "%g" % r["m"], "%g" % r["rs"], r["nsites"], "%g" % r["R0_over_rs"], r["kappa_sea"], "%g" % r["tau"], "%g" % r["sigma_a"], "%g" % r["T_end"], "%.0f" % r["N_B"],
                        "%.2f" % r["E_over_Emax"], "%.2f" % r["kappa_max"], r["K_dust"], "%.2f" % r["kappa_max_chain"], r["K_dust_chain"], "%.2f" % r["rule8"], "%.1f" % r["well_reach_expected"]] for r in rows]))
    L.append("")
    L.append("## Table 2: the isotropic Fermi ball's set (the requirement's rule) per member")
    L.append("")
    L.append(md_table(["member", "n_rad(kappa), kappa = 1..K_dust (WKB)", "n_rad,max = kappa_max/pi", "R_c = 4 a n_rad,max (1.222 N_B^(1/3) a)", "4 a n_rad(1)", "R_c/r_s", "exclusion floor 0.49 N_B^(1/3) a", "sum 16 kappa n_rad / N_B"],
                      [[r["name"], " ".join("%d:%.2f" % (k, x) for k, x in r["n_rad"]), "%.2f" % r["n_rad_max"], "%.1f a" % r["R_c"], "%.1f a" % r["R_c_from_n_rad_1"], "%.3f" % r["R_c_over_rs"],
                        "%.1f a" % r["exclusion_floor"], "%.2f" % r["capacity16_over_NB"]] for r in rows]))
    L.append("")
    L.append("## Table 3: coldness, step and the root gate")
    L.append("")
    L.append(md_table(["member", "p_F/m = kappa_max/((R_0/a)(m a))", "p_F/m' (chains)", "(p_F/m)^2/5", "tau/r_s", "sigma/r_s", "R_t^tr = f_form r_s", "sigma/R_t^tr", "gate"],
                      [[r["name"], "%.3f" % r["p_F_over_m"], "%.3f" % r["p_F_over_m_chain"], "%.4f" % r["pressure_corr"], "%.5f" % r["tau_over_rs"], "%.4f" % r["sigma_over_rs"], "%.1f a" % r["R_t_transferred"],
                        "%.3f" % r["gate"], r["gate_class"]] for r in rows]))
    L.append("")
    L.append("## Table 4: the launcher's derived cost estimates")
    L.append("")
    L.append(md_table(["member", "records", "MiB at kappa_sea", "MiB at the isotropic set (K_iso)", "1.3 x MiB_iso against the card", "H100-hours lo to hi (launcher pace)"],
                      [[r["name"], r["records"], "%.0f" % r["mem_MiB"], "%.0f (K %d)" % (r["mem_MiB_iso"], r["K_iso"]), "%.0f (%s)" % (1.3 * r["mem_MiB_iso"], "fits" if 1.3 * r["mem_MiB_iso"] <= card_mib else "DOES NOT FIT"),
                        "%.1f to %.1f" % (r["hours_lo"], r["hours_hi"])] for r in rows]))
    L.append("")
    t0rows = [r for r in rows if r.get("t0")]
    L.append("## Table 5: the actual dust catalog (the launcher's t0 line or the engine's preparation)")
    L.append("")
    if t0rows:
        L.append(md_table(["member", "columns per channel (radial modes)", "top channel", "truncated by kappa_sea", "n_rad,max own", "4 a n_rad,max own", "fractional multiplets (channel: lowest f)", "F kept", "f_kin", "p_F/m from f_kin", "t0"],
                          [[r["name"], " ".join("%d:%d" % (k, c) for k, c in sorted(r["t0"]["columns"].items())), r["t0"]["top"], "YES" if r["t0"]["truncated_by_kappa_sea"] else "no", r["t0"]["n_rad_max"],
                            ("%.0f a = %.2f r_s" % (r["t0"]["R_c_own"], r["t0"]["R_c_own_over_rs"])) if r["t0"]["R_c_own"] else "-",
                            (" ".join("%d:%.4f" % (k, f) for k, f in sorted(r["t0"]["fractional_multiplets"].items())) or "none (whole multiplets)"), f3(r["t0"]["F_kept"], "%.2f"),
                            f3(r["t0"]["f_kin"], "%.5f"), f3(r["t0"]["p_F_over_m_from_f_kin"]), r["t0"].get("t0_verdict") or "-"] for r in t0rows]))
    else:
        L.append("No status file or preparation given: the actual catalogs (columns per channel, the top channel, the fractional multiplet, f_kin) are read from the pods' status_<member>.txt t0 lines (--status-dir) or by --prepare MEMBER.")
    L.append("")
    L.append("## Flags per member")
    L.append("")
    for r in rows:
        L.append("- %s (%s): %s" % (r["name"], r["status"], "; ".join(r["flags"])))
    L.append("")
    L.append("## The running or done members against the rule")
    L.append("")
    L.append("Under the isotropic rule (OWNER 3's option (a)): every member below violates it, because the engine has no catalog option (the dust set is the well's filling within kappa_sea) and no (0.2, 64) member on the isotropic set exists (W12 held); in addition "
             + ", ".join("%s (kappa_sea %d below K_dust %d)" % (r["name"], r["kappa_sea"], r["K_dust"]) for r in rows if r["name"] in running and r["kappa_sea"] < r["K_dust"])
             + " cannot hold the set in their sea sets and "
             + ", ".join("%s" % r["name"] for r in rows if r["name"] in running and r["kappa_sea"] == r["K_dust"])
             + " hold it only with the top channel occupied. Under the (N_B/8)^(1/3) rule (option (b)): none violates by its arguments; the only open case is a catalog truncated by the sea set, which Table 5 decides where a t0 line is given ("
             + ", ".join("%s (expected reach %.1f against kappa_sea %d)" % (r["name"], r["well_reach_expected"], r["kappa_sea"]) for r in rows if r["name"] in running and r["well_reach_expected"] > r["kappa_sea"] - 1.0)
             + " at risk). Comparability (W12's question): the running members and the production are one ladder at one preparation rule, the well's filling from coordinate rest, so they compare with each other and with the production at matched clocks; the asymptotic onset "
             "form 1.222 N_B^(1/3) a and the isotropic p_F/m are not theirs (their own 4 a n_rad,max and f_kin print from Table 5), and the comparison with the isotropic forms waits for the bridge member; a member whose catalog is truncated by its sea set is on neither rule.")
    L.append("")
    L.append("## The width cross: 6 points for the model 1, z, h, zh")
    L.append("")
    byname = {r["name"]: r for r in rows}
    wrows = []
    for nm, rs_, sg in WIDTH_CROSS:
        r = byname.get(nm)
        wrows.append([nm, "%g" % rs_, "%g" % sg, "%.4f" % math.log(1.0 / rs_), "%.4f" % math.log(sg), "%.4f" % (sg / rs_), "%.3f" % (sg / (f_form * rs_)), (r["status"] if r else "-")])
    L.append(md_table(["point", "r_s/a", "sigma/a", "z = ln(a/r_s)", "h = ln(sigma/a)", "sigma/r_s", "sigma/R_t^tr", "status"], wrows))
    L.append("")
    L.append("The 6 points form a 2 by 3 grid: 2 values of z and 3 values of sigma/r_s (1/32, 1/16, 1/8), each sigma/r_s met at both sizes ((2, 64) with (4, 128), (4, 64) with (8, 128), (8, 64) with (16, 128)) and each sigma/a (4, 8) at both sizes; "
             "the tensor basis {1, z} x {1, h} = {1, z, h, zh} is independent on such a grid, 4 parameters on 6 points with 2 residual degrees of freedom, so the fixed-sigma/a direction ((a/r_s)^p) and the fixed-sigma/r_s direction are separated; "
             "the sigma/a 16 point at r_s/a 128 sits at the gate's edge (sigma/R_t^tr %.2f) and prints labelled so." % (16.0 / (f_form * 128.0)))
    L.append("")
    L.append("## The equal-step pairs (tau/r_s)")
    L.append("")
    groups = {}
    for r in rows:
        groups.setdefault(round(r["tau_over_rs"], 6), []).append(r["name"])
    for k in sorted(groups, reverse=True):
        L.append("- tau/r_s %.5f: %s" % (k, ", ".join(groups[k])))
    L.append("")
    L.append("## Findings")
    L.append("")
    caps = [r["capacity16_over_NB"] for r in rows]
    L.append("1. The multiplicity behind the isotropic rule. With the chains' weights 16 kappa per channel (Chan.w = 16.0 * kappa), the capacity sum of the isotropic set under the rule kappa_max = 0.96 N_B^(1/3) is sum_kappa 16 kappa n_rad(kappa) = %.2f to %.2f of N_B (Table 2): "
             "the rule counts 16 states per momentum in 3 dimensions, which in the angular decomposition is 32 kappa states per |kappa| (both parities of the 4-component field's channel kappa for the 8 species), while one chain per kappa with 16 kappa carries one parity's count. "
             "On the chains as built the isotropic filling of N_B reaches kappa_max' = (9 pi N_B/16)^(1/3) = 1.21 N_B^(1/3) (Table 1's K_dust'), with n_rad,max' = 0.385 N_B^(1/3), R_c' = 1.54 N_B^(1/3) a and p_F/m' = 1.26 x the requirement's value (Table 3). "
             "Which multiplicity the chain carries is a model question for the lead and icloud X11, to be ruled before any isotropic catalog (W12) is built; every number here follows the requirement's rule, the chain alternative beside it." % (min(caps), max(caps)))
    m3d = byname.get("M3d_s4"); m3c = byname.get("M3c_s4")
    if m3d:
        L.append("2. The (0.05, 326) member's channel count. N_B 13040 (r_s/a = 2 g m a N_B exactly); K_dust 23 by the rule (28 on the chains' multiplicity); the launcher's kappa_sea 12 is the (N_B/8)^(1/3) rule's 11.77 rounded, so the sea set cannot hold the isotropic set; "
                 "at 3912 sites the isotropic set needs %.0f MiB by the launcher's estimate against the card's %.0f, so no isotropic rerun fits the card without an engine memory change (14 channels fit: %.0f MiB, 1.3 x that %.0f); its expected well reach %.1f exceeds kappa_sea 12, so its catalog may be truncated by the sea set: Table 5's row (its t0 line) decides, and RUN_W11_W5_v1.md states the consequence." % (
                     m3d["mem_MiB_iso"], card_mib, 3912 * 3912 * (256 * 14 + 150) / 2.0 ** 20, 1.3 * 3912 * 3912 * (256 * 14 + 150) / 2.0 ** 20, m3d["well_reach_expected"]))
    if m3c:
        L.append("3. M3c_s4 (0.1, 96): N_B 1920, K_dust 12 against kappa_sea 10: cannot hold the isotropic set; its well's expected reach %.1f stays below 10, so its catalog is the well's full filling." % m3c["well_reach_expected"])
    L.append("4. No member of the launcher runs the isotropic set: the dust catalog is prepare_dust's filling of the levitating well's levels across all channels of the sea set by fill_lowest, so every ladder member and the production share one preparation rule; the ladders' onset radii 4 a n_rad,max are each member's own (Table 5) and not 1.222 N_B^(1/3) a; the bridge is W12 (held).")
    L.append("5. The size ladder's r_s/a 32 member (N_B 320, tau 2 for the equal-step pair with r_s/a 128 at tau 8) is absent from the launcher; MASS_ladder and SCALING_ladder need its source named (a stored r_s/a 32 history on the same preparation or a launcher line).")
    L.append("6. The transferred gate values at r_s/a 128: sigma/a 4, 8, 16 give sigma/R_t^tr %.2f, %.2f, %.2f with f_form %.3f (the requirement's 0.05, 0.10, 0.21 at f_form about 0.60); at r_s/a 64 the late root at 36.3 a gives 0.11 for sigma/a 4 and the inner roots at 7 and 13 a lie outside the gate for every sigma." % (
        4.0 / (f_form * 128.0), 8.0 / (f_form * 128.0), 16.0 / (f_form * 128.0), f_form))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="the ladders' manifest (busi W11)")
    ap.add_argument("--launcher", default=None, help="launch_ladder_v36d_v1c.sh (its member table is parsed; the file governs)")
    ap.add_argument("--status-dir", default=None, help="a directory holding the pods' status_<member>.txt files (the t0 lines)")
    ap.add_argument("--formation-json", default=None, help="the (0.2, 64) sigma/a 4 tau 8 history's collapse_<tag>.json (f_form from its formation record)")
    ap.add_argument("--f-form", type=float, default=None, help="the transferred formation radius over r_s when no json is given (default 0.61, labelled a placeholder)")
    ap.add_argument("--running", default=",".join(LEAD_RUNNING), help="the members running or done (the lead's list), or none")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--card-mib", type=float, default=CARD_MIB_DEFAULT)
    ap.add_argument("--prepare", default=None, help="run the engine's own preparation (CPU) for this member and read its catalog")
    ap.add_argument("--code-dir", default=None, help="the engine code tree (engine_collapse_v36d.py and its modules) for --prepare")
    ap.add_argument("--prep-passes", type=int, default=200)
    ap.add_argument("--prep-tol", type=float, default=1e-10)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    defaults, table, table_src, check_lines = BUILTIN_DEFAULTS, BUILTIN_TABLE, "the built-in copy of input 12's member table", []
    if a.launcher:
        pdef, ptab = parse_launcher(a.launcher)
        check_lines = compare_tables(BUILTIN_DEFAULTS, BUILTIN_TABLE, pdef, ptab) or ["identical in every member and argument"]
        defaults, table, table_src = pdef, ptab, a.launcher
    running = set() if a.running.strip().lower() in ("", "none") else set(x.strip() for x in a.running.split(",") if x.strip())
    f_form, f_src = F_FORM_DEFAULT, "the placeholder 0.61 (0.66 of the formation record's 2GE at its collar charge on the (0.2, 64) sigma/a 4 tau 8 history; give --formation-json)"
    if a.f_form is not None:
        f_form, f_src = a.f_form, "--f-form %g" % a.f_form
    if a.formation_json:
        fi = read_formation(a.formation_json)
        f_form = fi["f_form"]
        f_src = "%s: R_t %.3f a at T/r_s %.4f, %.3f of the record's 2GE at E/E0 %.4f" % (a.formation_json, fi["R_t"], fi["T_over_rs"], fi["R_t_over_2gE"] or float("nan"), fi["E_over_E0"] or float("nan"))
    members = []
    for name, over in table:
        d = dict(defaults); d.update(over)
        members.append((name, d, "launcher", "running or done (the lead's list)" if name in running else "not launched (not in the lead's list)"))
    for name, over, status in REQ_ROWS:
        d = dict(BUILTIN_DEFAULTS); d.update(over)
        members.append((name, d, "requirements_new.txt", status))
    t0s = {}
    if a.status_dir:
        for name, _d, _s, _st in members:
            p = os.path.join(a.status_dir, "status_%s.txt" % name)
            if os.path.exists(p):
                t0 = read_status(p)
                if t0:
                    t0s[name] = t0
    if a.prepare:
        if not a.code_dir:
            raise SystemExit("ladder_manifest_v1: --prepare needs --code-dir")
        d = dict((m_[1] for m_ in members if m_[0] == a.prepare), **{})
        if not d:
            raise SystemExit("ladder_manifest_v1: unknown member %s" % a.prepare)
        t0s[a.prepare] = prepare_member(a.prepare, d, a.code_dir, a.prep_passes, a.prep_tol, a.threads)
    rows = [member_row(name, d, src, st, f_form, a.card_mib, t0s.get(name)) for name, d, src, st in members]
    text = render(rows, running, f_form, f_src, check_lines, table_src, a.card_mib)
    print(text)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "ladder_manifest_v1.md"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(a.out_dir, "ladder_manifest_v1.json"), "w") as fh:
        json.dump({"f_form": f_form, "f_form_source": f_src, "table_source": table_src, "table_check": check_lines, "card_mib": a.card_mib, "running": sorted(running),
                   "constants": {"C_ISO": C_ISO, "C_CHAIN": C_CHAIN, "C_EXCL": C_EXCL}, "members": rows}, fh, indent=1, default=str)
    print("\nwritten: %s and %s" % (os.path.join(a.out_dir, "ladder_manifest_v1.md"), os.path.join(a.out_dir, "ladder_manifest_v1.json")))


if __name__ == "__main__":
    main()
