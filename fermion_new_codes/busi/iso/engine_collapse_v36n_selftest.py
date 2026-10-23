#!/usr/bin/env python
"""engine_collapse_v36n_selftest.py (busi, reviewer A; ISO_ruling_v1, ISO_fix_v1b): the self-test of engine_collapse_v36n.py (built by build_engine_collapse_v36n.py
over engine_collapse_v36j.py) on a small fixture, CPU or GPU; the T = 0 gate of a member's isotropic run; the catalog preview of any member.

The fixture: m a 0.2, r_s/a 16 (N_B 160), 512 sites, R_0 4 r_s (64 cavity sites), kappa_sea 10, sigma/a 4 (prepared at 4), tau 2, 3 records
(T/r_s 0, 0.125, 0.25), the ladders' flags (--reading A --reference R2 --site-fields averaged --mix plain --v36-fixes --field-kernel-cache), the
T = 0 complex128 slice written. Each case is one engine process in its own directory under --out; "bitwise" means the json's head and records
with the wall-clock keys removed, the fields, spectra and handoff npz and the T = 0 slice, byte for byte.
  J   engine_collapse_v36j.py, the host's defaults
  N   engine_collapse_v36n.py, defaults                                        G-A   bitwise with J (the default path)
  D   v36n --iter-cache drop                                                    G-A'  bitwise with J
  L   v36n --iter-cache drop --prep-memory lean                                 G-A'' bitwise with J
  U   v36n --mix damped --mix-alpha 3 --maxit 3 --stop-fields 1e-14             the overshooting iteration: unconverged records whose best iterate is not the last
  UD  U with --iter-cache drop                                                  G-A*  bitwise with U; the recompute path runs (counted in the records' timing)
  I   v36n --dust-set isotropic                                                 G-B   the ruling's catalog (below)
  I2  v36n --dust-set isotropic --dust-multiplicity 2                           G-C   the requirement's degeneracy-16 count (f <= 2)
  IC  v36n --dust-set isotropic --dust-kappa-max 2                              G-D   the cap binds (labelled; the fixture's uncapped top is 3), N_B exact
G-B (also G-C, G-D and the T = 0 gate of a member, --t0): N_B = r_s/(2 g m) to 1e-8 from the slice's weights and occupations and from the head; the
catalog equals the lowest-N_B filling of the lattice cavity recomputed here (the free chain of engine_collapse_v5.Chan on the sites R <= R_0,
C5.fill_lowest with the weights 16 kappa x multiplicity: the columns and occupations per channel); the multiplicity rule (the weights 16 kappa;
f whole at the multiplicity except one fractional value, uniform over its group; the head's coefficient 1.209 or 0.960); not truncated by the sea
set, capped only when asked; the frame checks (ISO_fix_v1b: the well's orbitals are stationary states of h_m[N, v] + U_w, at rest in the static
frame, which moves through the chain's Painleve-Gullstrand frame at the Eulerian speed |v|/N, while the chain's h and Q measure the river frame):
f_kin of the orbitals in the static frame, (sum_s sqrt(eps_s^2 - J_s^2) - N_B m)/(N_B m) from the slice's densities (C5.dens), within 25 percent of
the cavity key's (the well's wall is a smoothed step of height m/2, not the lattice's hard wall; the ratio is the wall's factor), and the dust at
rest in the static frame, |sum J| = sum_s eps_s |v_s|/N_s within 10 percent with the T = 0 fields N_cinit, v_cinit of the fields npz (reading B:
|sum J| below 1e-6 N_B); the slice's chain-frame f_kin and sum J equal to the head's, and the head's frame figures to the slice's, at 1e-6; the dust
columns orthonormal and orthogonal to the T = 0 sea (1e-7); the label. READ lines, printed and never judged: the continuum Dirichlet cavity in
both l conventions (the zeros of j_l, l = kappa and kappa - 1: columns, top, k_F R_0 and f_kin on the key with the ratio lattice/continuum) and the
semiclassical ball (Weyl and surface-corrected).
Usage:
  python engine_collapse_v36n_selftest.py --code-dir DIR [--gpu N] [--out DIR] [--cases J,N,D,L,U,UD,I,I2,IC] [--threads 16] [--no-kcache]
  python engine_collapse_v36n_selftest.py --code-dir DIR --t0 RUN/collapse_<tag>.json --t0-slice STATES/orbitals_<tag>_T0000.00.npz [--t0-fields RUN/collapse_<tag>_fields.npz] [--no-ortho]
  python engine_collapse_v36n_selftest.py --code-dir DIR --preview --m 0.05 --rs 326 --nsites 3912 --kappa-sea 27 [--mult 2] [--cap K]
Exit status 0 when every check passes (the cases mode and the --t0 mode)."""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

SKIP = ("seconds", "timing", "handoff_s", "oracle34_s", "propagation_prev_s", "slice_saved", "compute_s")   # ISO gate fix v1b: compute_s (v36/kernel_cache) is a wall
FIX = {"m": 0.2, "g": 0.25, "rs": 16.0, "nsites": 512, "R0": 4.0, "K": 10, "ell": 4, "prep_ell": 4, "tau": 2.0, "T_over_rs": 0.25}
C_CHAIN = (9.0 * math.pi / 16.0) ** (1.0 / 3.0)
C_G16 = (9.0 * math.pi / 32.0) ** (1.0 / 3.0)
CASES = [
    ("J", "engine_collapse_v36j.py", [], "the host's defaults"),
    ("N", "engine_collapse_v36n.py", [], "v36n's defaults"),
    ("D", "engine_collapse_v36n.py", ["--iter-cache", "drop"], "the drop policy"),
    ("L", "engine_collapse_v36n.py", ["--iter-cache", "drop", "--prep-memory", "lean"], "the drop policy and the lean preparation"),
    ("U", "engine_collapse_v36n.py", ["--mix", "damped", "--mix-alpha", "3", "--maxit", "3", "--stop-fields", "1e-14"], "the overshooting iteration"),
    ("UD", "engine_collapse_v36n.py", ["--mix", "damped", "--mix-alpha", "3", "--maxit", "3", "--stop-fields", "1e-14", "--iter-cache", "drop"], "the recompute path"),
    ("I", "engine_collapse_v36n.py", ["--dust-set", "isotropic"], "the ruling's catalog"),
    ("I2", "engine_collapse_v36n.py", ["--dust-set", "isotropic", "--dust-multiplicity", "2"], "the requirement's count, doubled"),
    ("IC", "engine_collapse_v36n.py", ["--dust-set", "isotropic", "--dust-kappa-max", "2"], "the cap (2: below the fixture's uncapped top 3, so it binds)"),
]


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in SKIP}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


# ------------------------------------------------------------------------------------------------------------------- the engine runs
def base_cmd(a, engine, out, tag, slice_dir):
    c = [sys.executable, engine, "--m", "%g" % FIX["m"], "--g", "%g" % FIX["g"], "--rs", "%g" % FIX["rs"], "--R0-over-rs", "%g" % FIX["R0"], "--reading", "A",
         "--nsites", str(FIX["nsites"]), "--kappa-sea", str(FIX["K"]), "--reference", "R2", "--site-fields", "averaged", "--prep-ell", str(FIX["prep_ell"]),
         "--ell", str(FIX["ell"]), "--tau", "%g" % FIX["tau"], "--T-over-rs", "%g" % FIX["T_over_rs"], "--mix", "plain", "--v36-fixes",
         "--slices-c128", "0", "--slice-dir", slice_dir, "--out", out, "--tag", tag]
    if not a.no_kcache:
        c.append("--field-kernel-cache")
    if a.gpu is not None:
        c += ["--gpu", str(a.gpu)]
    return c


def run_case(a, name, engine, extra, root):
    out = os.path.join(root, name); os.makedirs(out, exist_ok=True)
    tag = "fx_" + name; sdir = os.path.join(out, "states")
    cmd = base_cmd(a, os.path.join(a.code_dir, engine), out, tag, sdir) + extra
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(a.threads)
    log = os.path.join(out, "log.txt")
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n"); fh.flush()
        rc = subprocess.call(cmd, cwd=a.code_dir, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return {"name": name, "rc": rc, "seconds": time.time() - t0, "log": log, "json": os.path.join(out, "collapse_%s.json" % tag),
            "fields": os.path.join(out, "collapse_%s_fields.npz" % tag), "spectra": os.path.join(out, "collapse_%s_spectra.npz" % tag),
            "handoff": os.path.join(out, "collapse_%s_handoff.npz" % tag), "slice": os.path.join(sdir, "orbitals_%s_T0000.00.npz" % tag)}


def npz_equal(pa, pb, label, diffs):
    ZA = np.load(pa, allow_pickle=True); ZB = np.load(pb, allow_pickle=True)
    if sorted(ZA.files) != sorted(ZB.files):
        diffs.append("%s: the keys differ (%s)" % (label, sorted(set(ZA.files) ^ set(ZB.files))))
        return
    for k in ZA.files:
        xa, xb = ZA[k], ZB[k]
        if xa.dtype == object or xb.dtype == object:
            same = xa.shape == xb.shape and all(np.array_equal(np.asarray(p), np.asarray(q)) for p, q in zip(xa.ravel(), xb.ravel()))
        else:
            same = xa.dtype == xb.dtype and xa.shape == xb.shape and np.array_equal(xa, xb)
        if not same:
            diffs.append("%s[%s]" % (label, k))


def compare_runs(A, B):
    """the records (the walls removed), the head, the fields, spectra and handoff npz and the T = 0 slice: byte for byte."""
    diffs = []
    JA, JB = json.load(open(A["json"])), json.load(open(B["json"]))
    ra = [json.dumps(strip(q), sort_keys=True, default=float) for q in JA["records"]]
    rb = [json.dumps(strip(q), sort_keys=True, default=float) for q in JB["records"]]
    if len(ra) != len(rb):
        diffs.append("record count %d against %d" % (len(ra), len(rb)))
    bad = [JA["records"][i].get("T_over_rs") for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]]
    if bad:
        diffs.append("records differ at T/rs %s" % bad)
    if json.dumps(strip(JA["head"]), sort_keys=True, default=float) != json.dumps(strip(JB["head"]), sort_keys=True, default=float):
        diffs.append("heads differ")
    for key in ("fields", "spectra", "handoff", "slice"):
        if os.path.exists(A[key]) and os.path.exists(B[key]):
            npz_equal(A[key], B[key], key, diffs)
        else:
            diffs.append("%s missing" % key)
    return diffs


# ------------------------------------------------------------------------------------------------------------------- the analytic ball
def weyl_kR(g, NB):
    return (9.0 * math.pi * NB / (2.0 * g)) ** (1.0 / 3.0)


def surface_kR(g, NB):
    """k_F R_0 with the Dirichlet surface term: 2 g x^3/(9 pi) - g x^2/4 = N_B."""
    x = weyl_kR(g, NB)
    for _ in range(100):
        fx = 2.0 * g * x ** 3 / (9.0 * math.pi) - g * x * x / 4.0 - NB
        dx = 6.0 * g * x * x / (9.0 * math.pi) - g * x / 2.0
        if dx <= 0:
            break
        step = fx / dx; x -= step
        if abs(step) < 1e-12:
            break
    return x


def ball_fkin(g, NB, R0, m, kR, surface):
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    pF = kR / R0
    p = np.linspace(0.0, pF, 4001)
    dos = 2.0 * g * R0 ** 3 / (3.0 * math.pi) * p * p - (g * R0 ** 2 / 2.0 if surface else 0.0) * p
    return float(tz((np.sqrt(p * p + m * m) - m) * dos, p)) / (NB * m)


def load_c5(code_dir, threads):
    sys.path.insert(0, code_dir)
    import torch
    torch.set_num_threads(int(threads))
    import engine_collapse_v5 as C5
    C5.backend(None)
    return C5, torch


def lattice_cavity(C5, torch, m, nsites, K, R0):
    """the free chain's positive levels (at or above m) on the sites R <= R_0, per channel: the engine's ordering key recomputed independently."""
    chs = [C5.Chan(k, m, nsites, False) for k in range(1, K + 1)]
    R = np.asarray(chs[0].R, float)
    n0 = int(np.sum(R <= R0 * (1.0 + 1e-12)))
    ones = np.ones(nsites); zeros = np.zeros(nsites)
    keys, edge = {}, {}
    for ch in chs:
        H = ch.clocked_dense(ones, zeros)
        ev = torch.linalg.eigvalsh(H[:2 * n0, :2 * n0]).cpu().numpy().astype(float)
        del H
        pos = np.sort(ev[ev > 0])
        inband = pos >= m * (1.0 - 1e-9)
        keys[ch.kappa] = pos[inband]; edge[ch.kappa] = int(np.sum(~inband))
    return n0, keys, edge


def fill_catalog(C5, keys, K, mult, NB, m, R0, cap=None):
    """the lowest-N_B filling of a level set with the weights 16 kappa x mult (C5.fill_lowest; f per copy in (0, 1], scaled by mult)."""
    levels = []
    for ic, kappa in enumerate(range(1, K + 1)):
        if cap is not None and kappa > cap:
            continue
        for j, e in enumerate(keys[kappa]):
            levels.append((float(e), 16.0 * kappa * mult, ic, j))
    occ = C5.fill_lowest(levels, NB)
    cols, fs, ekin, EF = {}, {}, 0.0, m
    for (ic, j), f in sorted(occ.items()):
        if f <= 0:
            continue
        kappa = ic + 1
        cols[kappa] = cols.get(kappa, 0) + 1
        fs.setdefault(kappa, []).append(mult * f)
        ekin += 16.0 * kappa * mult * f * (float(keys[kappa][j]) - m)
        EF = max(EF, float(keys[kappa][j]))
    return {"columns": cols, "f": fs, "f_kin": ekin / (NB * m), "E_F": EF, "k_F_R0": math.sqrt(max(EF * EF - m * m, 0.0)) * R0, "top": (max(cols) if cols else 0)}


def bessel_zeros(l, zmax):
    from scipy.special import spherical_jn
    from scipy.optimize import brentq
    grid = np.arange(max(float(l), 0.0) + 1e-6, zmax, 0.02)
    f = spherical_jn(l, grid)
    zs = []
    for i in range(len(grid) - 1):
        if f[i] == 0.0:
            zs.append(float(grid[i]))
        elif f[i] * f[i + 1] < 0.0:
            zs.append(float(brentq(lambda z: spherical_jn(l, z), grid[i], grid[i + 1], xtol=1e-12)))
    return zs


def continuum_catalog(C5, K, mult, NB, R0, m, shift):
    """the continuum Dirichlet cavity: the zeros of j_l with l = kappa - shift, E = sqrt(m^2 + (z/R_0)^2), the same weights and filling."""
    zmax = 1.6 * surface_kR(8 * mult, NB) + 10.0
    keys = {}
    for kappa in range(1, K + 1):
        l = kappa - shift
        zs = bessel_zeros(l, zmax) if l >= 0 else []
        keys[kappa] = np.array([math.sqrt(m * m + (z / R0) ** 2) for z in zs])
    return fill_catalog(C5, keys, K, mult, NB, m, R0)


def memory_mib(n, K):
    return {"plain": n * n * (256 * K + 150) / 2.0 ** 20, "drop": n * n * (160 * K + 130) / 2.0 ** 20,
            "prep_full": n * n * (128 * K + 390) / 2.0 ** 20, "prep_lean": n * n * (64 * K + 390) / 2.0 ** 20}


def fmtcols(c):
    return " ".join("%d:%d" % (k, c[k]) for k in sorted(c))


# ------------------------------------------------------------------------------------------------------------------- the catalog checks
def slice_densities(C5, torch, Z, m, n, half_offset=False):
    """ISO_fix_v1b: the dust's energy and momentum densities per site in the chain's frame from the T = 0 slice, the engine's own sums
    (prepare_dust's eps += w dens(ch.h, Phi, f) and J += w dens(ch.Q, Phi, f))."""
    eps = np.zeros(n); J = np.zeros(n)
    for k_, w_ in zip([int(k) for k in Z["kappas"]], np.asarray(Z["weights"], float)):
        f = np.asarray(Z["f_%d" % k_], float)
        if not f.size:
            continue
        ch = C5.Chan(k_, m, n, bool(half_offset))
        Phi = torch.as_tensor(np.asarray(Z["dust_%d" % k_]), dtype=torch.complex128)
        eps += w_ * np.asarray(C5.dens(ch.h, Phi, f), float); J += w_ * np.asarray(C5.dens(ch.Q, Phi, f), float)
    return eps, J


def catalog_checks(a, C5, torch, J, slice_path, mult, cap, fields_path=None):
    """G-B on a run's json J and its T = 0 slice (and its fields npz, for the rest check): a list of (name, ok, text); ok None is a READ line."""
    args = J["args"]; head = J["head"]; out = []
    m = float(args["m"]); g = float(args["g"]); rs = float(args["rs"]); R0 = float(args["R0_over_rs"]) * rs; n = int(args["nsites"]); K = int(args["kappa_sea"])
    NB_t = rs / (2.0 * g * m)
    cat = head.get("catalog")
    out.append(("head: dust_set isotropic and a catalog present", bool(head.get("dust_set") == "isotropic" and cat), str(head.get("dust_set"))))
    if not cat:
        return out
    out.append(("the run continued past T = 0 (records written)", len([q for q in J["records"] if q.get("E") is not None]) >= 2, "%d records" % len(J["records"])))
    Z = np.load(slice_path)
    kap = [int(k) for k in Z["kappas"]]; w = np.asarray(Z["weights"], float)
    cols_s, f_s, NB_s = {}, {}, 0.0
    for k_, w_ in zip(kap, w):
        f = np.asarray(Z["f_%d" % k_], float)
        if f.size:
            cols_s[k_] = int(f.size); f_s[k_] = f; NB_s += w_ * float(f.sum())
    out.append(("weights 16 kappa", bool(np.allclose(w, 16.0 * np.asarray(kap, float))), "%s" % [float(x) for x in w]))
    out.append(("N_B from the slice = r_s/(2 g m) (1e-8) and = the head's", abs(NB_s - NB_t) <= 1e-8 * NB_t and abs(float(head["N_B"]) - NB_t) <= 1e-8 * NB_t,
                "%.9f against %.9f (head %.9f)" % (NB_s, NB_t, head["N_B"])))
    n0, keys, edge = lattice_cavity(C5, torch, m, n, K, R0)
    ref = fill_catalog(C5, keys, K, mult, NB_t, m, R0, cap)
    same_cols = (ref["columns"] == cols_s)
    same_f = same_cols and all(np.allclose(np.asarray(ref["f"][k]), f_s[k], atol=1e-9, rtol=0.0) for k in cols_s)
    out.append(("catalog = the lowest-N_B filling of the lattice cavity recomputed here (columns and occupations)", same_cols and same_f,
                "slice %s; recomputed %s" % (fmtcols(cols_s), fmtcols(ref["columns"]))))
    out.append(("cavity sites n_0 as the head's; edge modes dropped as the head's", n0 == cat.get("cavity_sites") and sum(edge.values()) == sum(v["edge_modes_dropped"] for v in cat["key_info"].values()),
                "%d against %s; edge modes %d" % (n0, cat.get("cavity_sites"), sum(edge.values()))))
    allf = np.concatenate([f_s[k] for k in sorted(f_s)]) if f_s else np.zeros(0)
    whole = np.abs(allf - mult) <= 1e-10
    frac = allf[~whole]
    out.append(("f <= multiplicity %d, whole except one fractional value uniform over its group" % mult,
                bool(allf.size and allf.max() <= mult + 1e-10 and (frac.size == 0 or np.ptp(frac) <= 1e-10)), "fractional values %s" % np.unique(np.round(frac, 12)).tolist()))
    top = max(cols_s) if cols_s else 0
    out.append(("head top channel = the slice's", cat.get("top_channel") == top, "%s against %d" % (cat.get("top_channel"), top)))
    nb13 = NB_t ** (1.0 / 3.0)
    out.append(("the head's kappa_max coefficient (chain rule %.4f, g16 rule %.4f; semiclassical kappa_max' %.2f, the g16 rule's %.2f)" % (C_CHAIN, C_G16, C_CHAIN * nb13, C_G16 * nb13),
                abs(cat["kappa_max_coefficient"] - (C_CHAIN if mult == 1 else C_G16)) < 1e-6, "%.4f" % cat["kappa_max_coefficient"]))
    if cap is not None:
        out.append(("the cap binds: labelled capped, top = cap, the uncapped top above it", bool(cat.get("capped") and top == cap and (cat.get("top_uncapped") or 0) > cap),
                    "%s" % {k: cat.get(k) for k in ("capped", "kappa_cap", "top_uncapped")}))
    else:
        out.append(("not capped, not truncated by the sea set", not cat.get("capped") and not cat.get("truncated_by_sea_set"), "top %d, sea top %d" % (top, K)))
    g8 = 8 * mult                                                                              # ISO gate fix v1b: the continuum and the ball are READ lines (ok None), never FAIL
    for shift in (0, 1):
        cont = continuum_catalog(C5, K, mult, NB_t, R0, m, shift)
        maxd = max(abs(cont["columns"].get(k, 0) - cols_s.get(k, 0)) for k in range(1, K + 1))
        out.append(("READ continuum Dirichlet cavity (l = kappa - %d), lattice / continuum: columns (max |d|); top; k_F R_0 (ratio); f_kin on the key (ratio)" % shift, None,
                    "%s / %s (%d); %d / %d; %.3f / %.3f (%.3f); %.5f / %.5f (%.3f)" % (
                        fmtcols(cols_s), fmtcols(cont["columns"]), maxd, top, cont["top"], cat["k_F_R0_catalog"], cont["k_F_R0"], cat["k_F_R0_catalog"] / cont["k_F_R0"],
                        cat["f_kin_on_cavity_key"], cont["f_kin"], cat["f_kin_on_cavity_key"] / cont["f_kin"])))
    out.append(("READ the semiclassical ball (g = %d), Weyl / surface-corrected: k_F R_0; f_kin; p_F/m" % g8, None, "%.3f / %.3f; %.5f / %.5f; %.4f / %.4f" % (
        weyl_kR(g8, NB_t), surface_kR(g8, NB_t), cat["ball"]["weyl"]["f_kin"], cat["ball"]["surface"]["f_kin"], cat["ball"]["weyl"]["p_F_over_m"], cat["ball"]["surface"]["p_F_over_m"])))
    # ISO_fix_v1b: the lattice checks in the frame the dust is at rest in. The well's orbitals are stationary states of h_m[N, v] + U_w: at rest in the
    # static frame, which moves through the chain's Painleve-Gullstrand frame at the Eulerian speed |v|/N; the chain's h and Q measure the river frame,
    # so the orbitals' f_kin carries the bulk flow and sum J = sum eps |v|/N (not 0: zero J is dust comoving with the river, free fall).
    eps_s, J_s = slice_densities(C5, torch, Z, m, n, bool(args.get("half_offset") or 0))
    fk_r = (float(eps_s.sum()) - NB_t * m) / (NB_t * m); Jt = float(J_s.sum()); fr = cat.get("frame") or {}
    out.append(("the slice's chain-frame f_kin and sum J = the head's f_kin and dust_momentum_total (1e-6: the slice is the prepared dust, C5.dens its sums)",
                abs(fk_r - cat["f_kin_orbitals"]) <= 1e-6 and abs(Jt - cat["dust_momentum_total"]) <= 1e-6 * NB_t * m,
                "%.7f against %.7f; %.6f against %.6f" % (fk_r, cat["f_kin_orbitals"], Jt, cat["dust_momentum_total"])))
    rest = float(np.sum(np.sqrt(np.maximum(eps_s * eps_s - J_s * J_s, 0.0))))
    fk_s = (rest - NB_t * m) / (NB_t * m); key = cat["f_kin_on_cavity_key"]
    out.append(("READ f_kin of the orbitals in the static frame, (sum_s sqrt(eps_s^2 - J_s^2) - N_B m)/(N_B m), against the cavity key's (ISO_fix_v1c: the well's own wall against the hard wall; the ratio and the excess f_kin_s - key are read into the launch record, never judged: 1.10 to 1.55 on the fixture, the lowest levels most; geometry alone gives 0.82 to 0.89)",
                None,
                "%.5f against %.5f (ratio %.3f); the chain's frame %.5f, the bulk flow's share %.5f, the host's bulk-frame figure %s" % (
                    fk_s, key, fk_s / key, fk_r, fk_r - fk_s, ("%.5f" % head["f_kin_bulk_frame"]) if head.get("f_kin_bulk_frame") is not None else "-")))
    out.append(("the head's frame figures = the slice's (1e-6; a head written before ISO_fix_v1b carries none and passes)",
                ("f_kin_static_frame" not in fr) or (abs(fr["f_kin_static_frame"] - fk_s) <= 1e-6 and abs(fr["dust_momentum_total"] - Jt) <= 1e-6 * NB_t * m),
                ("%.7f against %.7f" % (fr["f_kin_static_frame"], fk_s)) if "f_kin_static_frame" in fr else "no frame entry in the head"))
    fp = fields_path if (fields_path and os.path.exists(fields_path)) else None
    if str(args.get("reading") or "A") != "A":
        out.append(("dust momentum below 1e-6 N_B (reading B: the well's orbitals are standing waves in the chain's frame)", abs(Jt) <= 1e-6 * NB_t, "%.3e" % Jt))
    elif fp is None:
        out.append(("the dust at rest in the static frame: |sum J| against sum_s eps_s |v_s|/N_s (the fields npz is needed: --t0-fields)", False, "not found: %s" % fields_path))
    else:
        F = np.load(fp, allow_pickle=True)
        N0 = np.asarray(F["N_cinit"] if "N_cinit" in F.files else F["N"][0], float); v0 = np.asarray(F["v_cinit"] if "v_cinit" in F.files else F["v"][0], float)
        uE = np.abs(v0) / N0; exp_J = float(np.sum(eps_s * uE))
        out.append(("the dust at rest in the static frame: |sum J| = sum_s eps_s |v_s|/N_s within 10 percent (the Eulerian relation J = eps v/N up to the pressure's few percent; the T = 0 fields N_cinit, v_cinit)",
                    exp_J > 0 and abs(abs(Jt) - exp_J) <= 0.10 * exp_J,
                    "%.4f against %.4f (ratio %.4f); sum J / sum eps %.4f, the static frame's eps-weighted |v|/N %.4f, largest |v|/N on the chain %.4f%s" % (
                        Jt, exp_J, (abs(Jt) / exp_J) if exp_J > 0 else float("nan"), Jt / float(eps_s.sum()), exp_J / float(eps_s.sum()), float(uE.max()),
                        ("; the head's %.4f" % fr["static_frame_momentum"]) if "static_frame_momentum" in fr else "")))
    if not a.no_ortho:
        worst_on = 0.0; worst_sea = 0.0
        for k_ in cols_s:
            D = np.asarray(Z["dust_%d" % k_]); S = np.asarray(Z["sea_%d" % k_])
            G = D.conj().T @ D
            worst_on = max(worst_on, float(np.max(np.abs(G - np.eye(G.shape[0])))))
            worst_sea = max(worst_sea, float(np.max(np.abs(S.conj().T @ D))))
        out.append(("dust columns orthonormal and orthogonal to the T = 0 sea (1e-7)", worst_on <= 1e-7 and worst_sea <= 1e-7, "%.1e, %.1e" % (worst_on, worst_sea)))
    lab = head.get("label") or ""
    out.append(("label names the isotropic cavity catalog%s" % (" and doubled" if mult != 1 else ""), ("dust set: the isotropic cavity catalog" in lab) and (("doubled" in lab) == (mult != 1)), lab[-200:]))
    return out


def print_checks(checks):
    ok_all = True
    for grp, name, ok, text in checks:
        if ok is not None:                                                                     # ISO gate fix v1b: a READ line (ok None) is printed, not judged
            ok_all = ok_all and ok
        print("%-5s %-6s %s%s" % (grp, "READ" if ok is None else ("PASS" if ok else "FAIL"), name, (" | " + text) if text else ""))
    return ok_all


# ------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="the self-test of engine_collapse_v36n (busi, ISO_ruling_v1)")
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default=None, help="the scratch directory (default <code-dir>/selftest_v36n)")
    ap.add_argument("--cases", default="J,N,D,L,U,UD,I,I2,IC")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-kcache", dest="no_kcache", action="store_true", help="omit --field-kernel-cache (where engine_fields_kcache_v1 is absent)")
    ap.add_argument("--t0", default=None, help="a member run's collapse_<tag>.json: the T = 0 catalog gate")
    ap.add_argument("--t0-slice", dest="t0_slice", default=None, help="that run's orbitals_<tag>_T0000.00.npz")
    ap.add_argument("--t0-fields", dest="t0_fields", default=None, help="that run's collapse_<tag>_fields.npz (default: beside the json); its N_cinit, v_cinit are the T = 0 fields of the rest check (ISO_fix_v1b)")
    ap.add_argument("--levels", dest="levels", type=int, default=0, help="--preview: print the lowest n lattice cavity levels of kappa 1..4 as k R_0 beside the zeros of j_{kappa-1} and j_kappa (ISO_gate_fix_v1 G-spec; ISO_fix_v1c)")
    ap.add_argument("--no-ortho", dest="no_ortho", action="store_true", help="skip the orthogonality check (large slices)")
    ap.add_argument("--preview", action="store_true", help="the catalog of a member from its arguments (no preparation)")
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--g", type=float, default=0.25); ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--nsites", type=int, default=2048); ap.add_argument("--kappa-sea", dest="kappa_sea", type=int, default=10)
    ap.add_argument("--R0-over-rs", dest="R0_over_rs", type=float, default=4.0); ap.add_argument("--mult", type=int, default=1, choices=[1, 2]); ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args()
    a.code_dir = os.path.abspath(a.code_dir)
    C5, torch = load_c5(a.code_dir, a.threads)
    if a.preview:
        NB = a.rs / (2.0 * a.g * a.m); R0 = a.R0_over_rs * a.rs; g = 8 * a.mult
        n0, keys, edge = lattice_cavity(C5, torch, a.m, a.nsites, a.kappa_sea, R0)
        lat = fill_catalog(C5, keys, a.kappa_sea, a.mult, NB, a.m, R0, a.cap)
        kw, ks = weyl_kR(g, NB), surface_kR(g, NB)
        print("preview: m a %g, r_s/a %g, N_B %.2f, R_0 %g a (%d cavity sites of %d), kappa_sea %d, multiplicity %d (g = %d)%s" % (a.m, a.rs, NB, R0, n0, a.nsites, a.kappa_sea, a.mult, g, (", cap %d" % a.cap) if a.cap else ""))
        print("semiclassical: kappa_max %.2f (coefficient %.4f N_B^(1/3); the chain rule's %.2f, the g = 16 rule's %.2f); k_F R_0 Weyl %.3f, surface-corrected %.3f; p_F/m %.4f / %.4f; f_kin %.5f / %.5f (cold 0.3 (p_F/m)^2 %.5f)" % (
            kw, kw / NB ** (1.0 / 3.0), C_CHAIN * NB ** (1.0 / 3.0), C_G16 * NB ** (1.0 / 3.0), kw, ks, kw / (R0 * a.m), ks / (R0 * a.m), ball_fkin(g, NB, R0, a.m, kw, False), ball_fkin(g, NB, R0, a.m, ks, True), 0.3 * (kw / (R0 * a.m)) ** 2))
        print("the lattice cavity's catalog (what the engine fills): top channel %d; E_F %.6f; k_F R_0 %.3f; f_kin on the key %.5f; edge modes dropped %d; columns %s; lowest f per channel %s" % (
            lat["top"], lat["E_F"], lat["k_F_R0"], lat["f_kin"], sum(edge.values()), fmtcols(lat["columns"]), " ".join("%d:%.4f" % (k, min(v)) for k, v in sorted(lat["f"].items()))))
        if a.levels:                                                                           # ISO_fix_v1c (G-spec): the lattice cavity's lowest levels per channel beside the continuum's two series
            JZ = {0: (3.14, 6.28, 9.42, 12.57, 15.71, 18.85), 1: (4.49, 7.73, 10.90, 14.07, 17.22, 20.37), 2: (5.76, 9.10, 12.32, 15.51, 18.69, 21.85), 3: (6.99, 10.42, 13.70, 16.92, 20.12, 23.30), 4: (8.18, 11.70, 15.04, 18.30, 21.53, 24.73)}
            nc = int(round(R0))
            for k_ in range(1, min(4, a.kappa_sea) + 1):
                ch = C5.Chan(k_, a.m, nc, bool(getattr(a, "half_offset", 0) or 0)); H = (ch.h.to_dense() if ch.h.layout != torch.strided else ch.h) if torch.is_tensor(ch.h) else torch.as_tensor(ch.hs.toarray())  # busi: ch.h is a sparse CSR tensor; eigvalsh needs it dense (C3 as posted failed: aten::empty_strided on SparseCsrCPU)
                E = np.sort(torch.linalg.eigvalsh(H).cpu().numpy().astype(float)); E = E[E > 0][:a.levels]
                print("READ levels kappa %d (%d sites): lattice k R_0 %s | zeros of j_%d %s | j_%d %s   (an edge mode prints as 0.00 or far below 3.14; (ii) = both series merged, (i) = pairs or an alien series, (iii) = one series at a shifted spacing)" % (k_, nc, " ".join("%.2f" % x for x in np.sqrt(np.maximum(E * E - a.m * a.m, 0.0)) * R0), k_ - 1, " ".join("%.2f" % x for x in JZ[k_ - 1][:a.levels]), k_, " ".join("%.2f" % x for x in JZ[k_][:a.levels])))
        for shift in (0, 1):                                                                   # ISO gate fix v1b: a reading beside the lattice catalog, not a reference
            cont = continuum_catalog(C5, a.kappa_sea, a.mult, NB, R0, a.m, shift)
            print("READ the continuum Dirichlet cavity (l = kappa - %d): top %d (lattice %d); k_F R_0 %.3f (lattice/continuum %.3f); f_kin %.5f (ratio %.3f); columns %s" % (
                shift, cont["top"], lat["top"], cont["k_F_R0"], lat["k_F_R0"] / cont["k_F_R0"], cont["f_kin"], lat["f_kin"] / cont["f_kin"], fmtcols(cont["columns"])))
        mem = memory_mib(a.nsites, a.kappa_sea)
        print("kappa_sea rule (the top plus 2 <= kappa_sea): %d <= %d: %s%s" % (lat["top"] + 2, a.kappa_sea, "OK" if lat["top"] + 2 <= a.kappa_sea else "NOT MET",
                                                                             " (TRUNCATED: the top is the sea set's top)" if lat["top"] >= a.kappa_sea else ""))
        print("memory at %d sites, K %d: plain %.0f MiB (1.3 x %.0f); --iter-cache drop %.0f; preparation full %.0f, lean %.0f (the card 81,559)" % (a.nsites, a.kappa_sea, mem["plain"], 1.3 * mem["plain"], mem["drop"], mem["prep_full"], mem["prep_lean"]))
        return
    if a.t0:
        if not a.t0_slice:
            raise SystemExit("--t0 needs --t0-slice (the run's orbitals_<tag>_T0000.00.npz)")
        J = json.load(open(a.t0))
        mult = int(J["args"].get("dust_multiplicity") or 1); cap = J["args"].get("dust_kappa_max")
        fp0 = a.t0_fields or (a.t0[:-5] + "_fields.npz" if a.t0.endswith(".json") else None)   # ISO gate fix v1b: the T = 0 fields for the rest check
        checks = [("G-t0", nm, ok, tx) for nm, ok, tx in catalog_checks(a, C5, torch, J, a.t0_slice, mult, cap, fp0)]
        ok = print_checks(checks)
        print("G-t0 %s: %s" % ("PASS" if ok else "FAIL", a.t0))
        sys.exit(0 if ok else 1)
    for eng in ("engine_collapse_v36j.py", "engine_collapse_v36n.py", "engine_collapse_v5.py"):
        if not os.path.exists(os.path.join(a.code_dir, eng)):
            raise SystemExit("%s is not in %s (build engine_collapse_v36n.py with build_engine_collapse_v36n.py)" % (eng, a.code_dir))
    root = a.out or os.path.join(a.code_dir, "selftest_v36n")
    os.makedirs(root, exist_ok=True)
    want = [c.strip() for c in a.cases.split(",") if c.strip()]
    runs = {}
    for name, engine, extra, purpose in CASES:
        if name not in want:
            continue
        print("running %s (%s: %s) ..." % (name, engine, purpose), flush=True)
        runs[name] = run_case(a, name, engine, extra, root)
        print("  exit %d in %.0f s (%s)" % (runs[name]["rc"], runs[name]["seconds"], runs[name]["log"]), flush=True)
    checks = []

    def add(grp, name, ok, text=""):
        checks.append((grp, name, (None if ok is None else bool(ok)), text))             # ISO gate fix v1b: None = a READ line
    for r in runs.values():
        add("run", "%s exit status 0 (%.0f s)" % (r["name"], r["seconds"]), r["rc"] == 0, r["log"] if r["rc"] != 0 else "")
    for x, y, grp in (("N", "J", "G-A"), ("D", "J", "G-A'"), ("L", "J", "G-A''"), ("UD", "U", "G-A*")):
        if x in runs and y in runs and runs[x]["rc"] == 0 and runs[y]["rc"] == 0:
            d = compare_runs(runs[x], runs[y])
            add(grp, "%s bitwise with %s (records but the walls, head, fields, spectra, handoff, T = 0 slice)" % (x, y), not d, "; ".join(d) or "identical")
    if "U" in runs and runs["U"]["rc"] == 0:
        JU = json.load(open(runs["U"]["json"]))
        nb = sum(1 for q in JU["records"] if q.get("E") is not None and q.get("unconverged") and int(q["unconverged"].get("iteration") or 0) < int(q.get("iterations") or 0) - 1)
        add("G-A*", "U: unconverged records whose best iterate is not the last (the recompute path's precondition; a miss is a WARN: raise --mix-alpha)", nb > 0, "%d of %d records" % (nb, len(JU["records"])))
        if "UD" in runs and runs["UD"]["rc"] == 0:
            JD = json.load(open(runs["UD"]["json"]))
            nr = sum(1 for q in JD["records"] if "iter_cache_recompute_s" in (q.get("timing") or {}))
            add("G-A*", "UD: the recompute path ran (timing.iter_cache_recompute_s in the records)", nr > 0 or nb == 0, "%d records recomputed" % nr)
    for nm, mult, cap, grp in (("I", 1, None, "G-B"), ("I2", 2, None, "G-C"), ("IC", 1, 2, "G-D")):
        if nm in runs and runs[nm]["rc"] == 0:
            J = json.load(open(runs[nm]["json"]))
            for name, ok, text in catalog_checks(a, C5, torch, J, runs[nm]["slice"], mult, cap, runs[nm]["fields"]):
                add(grp, "%s: %s" % (nm, name), ok, text)
            try:
                for ln in [q for q in open(runs[nm]["log"], errors="replace") if q.startswith("dust set (v36n)")]:   # ISO gate fix v1b: the catalog line and the frames line
                    print("%s's printed line: %s" % (nm, ln.rstrip()))
            except OSError:
                pass
    ok = print_checks(checks)
    print("engine_collapse_v36n_selftest: %s (%d checks, %d cases, %s)" % ("PASS" if ok else "FAIL", len(checks), len(runs), "GPU %d" % a.gpu if a.gpu is not None else "CPU"))
    with open(os.path.join(root, "selftest_v36n.json"), "w") as fh:
        json.dump({"pass": ok, "gpu": a.gpu, "fixture": FIX, "checks": [{"group": g_, "name": n_, "ok": o_, "text": t_} for g_, n_, o_, t_ in checks],
                   "runs": {k: {q: v for q, v in r.items()} for k, r in runs.items()}}, fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
