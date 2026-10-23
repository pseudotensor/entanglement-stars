#!/usr/bin/env python
"""s12_postprocess.py, version 2 (busi lane task 5 = review task S12, J-VF post-processing; first draft and this revision by reviewer A at
maximum effort; the builder analyst runs it on the box CPU with 1 thread). Reads the solve lane's vacuum-flow records in place and writes
a small JSON and a markdown table set; it reruns nothing. Version 1 is kept beside this file as s12_postprocess_v1.py; the judgment of the
version-1 run is busi/notes/S12_judgment_v1.md and the changes below follow it.

Changes from version 1:
  T1 sharp cut: the rule is absolute, max |I_C - I_R2| < 1e-13 over channels, readouts and cuts (the residual is the rounding of O(1)
    summands, about 1e-16 per readout; version 1 divided it by the record's largest |I_C|, a scale 1000 times smaller in vf9 than in vf10
    because vf9's pocket channels 1 to 4 stopped before their currents grew, so the same 3.6e-16 passed in vf10 and failed in vf9); the
    relative figure is printed as information.
  T1 smoothed cut: the rule is in units of the erf step's tail at the window's edge (below 100 tails); between two records the residual
    scales with the tail (vf10 against vf9: 168 to 192 against 152), which is what shows it to be the window truncation.
  T4 exterior annuli: reported, not pass/fail (the spec's second branch): the ratios with the own share 2 own2/B, the local share
    (B/2)/clock2 and B_ann; the virial ratio is a statement for regions whose boundary term is small, which the exterior annuli are not.
  T5: the literal 1e-5 threshold is a reported size (met / not met) for the NO annuli and for the R2 annuli where stored; the control
    proper is the subtraction's arithmetic residual per e-fold over the slope (the ramp-to-ramp deviation of the removed response, raw - NO
    and raw - R2 where stored) beside T1's integrated reference current.
  T6c: the engine's probe convention is detected from the data (the offset d in 'the sum over the sites n < round(p r_s) - 1 + d', d = 0
    being 'R = n + 1 and R < p r_s', the convention of vf10's cuts), the rule 1e-9 applies under the detected convention, the fine bins use
    it, and alpha is printed under both conventions (version 1 compared bins of one convention with running sums of another: a one-site
    edge mismatch, 1.3e-2 at r_s 32 and 3.0e-2 at r_s 16, not a channel-set difference).
  T3a-R2 (new): the removed response under the continued reference, raw - R2, and its ramp spread where R2 is stored.
  T3f (new): the linearity of the removed response in r_s at a fixed lattice annulus (r_s 32 [2, 4] and [4, 8] r_s against r_s 16 [4, 8]
    and [8, 16] r_s, the same sites), the statement that the response is quadratic in the flow amplitude; rule within 1 percent of 2.
  The verdict table carries a kind: control (pass/fail; a failure is a defect to be judged), size (met / not met; a number against a
    literal threshold), reported (no rule). --engine-dir defaults to <internal>/code, where engine_vacflow_v7.py is.

The records and what is formed from them (busi/specs/S12_spec_v1.md, sections 1 and 2):
  vf10, vf9 (engine_vacflow_v10, v9): per channel kappa (weight 16 kappa) and readout (every 0.5 r_s over the ramp of 32 r_s and the hold
    of 2 r_s): s, sdot; E_in/E_out of the clock density hcal_n = (1/2){P_n, h_0} under NO (against R0(s), the negative projector of h(s)),
    R2 (against the continued reference) and raw (against the flat sea); at the cuts R = r_s, 1.5 r_s, 2 r_s: E_K_C = Tr[C (1/2){P_c, h(s)}],
    E_K_R0 the same for R0(s); W_C = sdot Tr[C V_c], W_R0 = sdot Tr[R0 V_c], W = W_C - W_R0 with V_c = (1/2){P_c, V}; the sharp-cut Killing
    currents I_C, I_NO, I_R2 = (i/2) Tr[D [h^2, P_c]] for D = C, C - R0, C - R2; the smoothed-cut currents IS_*; N_p, N_h.
    Formed: inside each cut and in the annuli between the cuts, the generator, flow and clock energies of R0 relative to the flat sea
    (gen_ref = E_K_R0(t) - E_K_R0(0), flow_ref = s W_R0/sdot, clock_ref = gen_ref - flow_ref; a static quantity), of C relative to the flat
    sea (gen_C, flow_C, clock_C; raw) and of C relative to R0 (gen_NO, flow_NO, clock_NO); their s^2 coefficients at small s (the
    1 : -2 : -1 test) and their values at the ramp's end; the continued-reference residual I_C - I_R2 (zero for a projector commuting with
    h(s)) and its smoothed version (bounded by the erf step's tail at the window's edge); the field identities clock_NO = E_in_NO and
    clock_C = E_in_raw at the cut r_s.
  vf3 (r_s 16, 32), vf7 (r_s 32) (engine_vacflow_v3, v7): running channel sums of the cumulative clock energy below the probes
    1.5, 2, 4, 8, 16 r_s after the hold (E_raw_at, E_NO_at, possibly E_R2_at) per ramp t_r = 2, 8, 32 r_s; per-site totals at the hold.
    Formed: the annuli raw, NO, R2 (where stored) and the removed response raw - NO = Tr[(R0 - flat) hcal] (and raw - R2) identical across
    ramps; its falloff fit (the channel-limited prediction: annulus energy ~ R^-2, 4 per doubling, the 3D density r^-5; the logarithm is
    alpha 0); the two-size ratio against the scale-covariant 1/2 and the linearity in r_s at fixed lattice annulus; the NO and R2 annuli
    per e-fold against the full-zone flat slope (9 pi/2) B_L r_s (the literal threshold 1e-5, a size); the per-site totals binned out to
    60 r_s under the engine's probe convention.
Modes:
  --selftest                       checks the fits on synthetic power laws and s-expansions and the record logic on synthetic records
  --dump-keys FILE [--depth D]     prints a record's structure
  --vf10 F --vf9 F --vf3-32 F --vf3-16 F --vf7 F --out DIR [--kmax 16 --BL 0.0194139 --smax 0.04]   the post-processing
  --static-run --lead-approved ... the smallest run supplying the missing far-zone flow and generator annuli (spec section 5); it does not
                                   start without --lead-approved, which the builder passes only after the lead's approval
Outputs: DIR/s12_results.json, DIR/s12_tables.md (and DIR/s12_static_response.json, DIR/s12_static_tables.md for the static run)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np

try:
    from scipy.special import erfc as _erfc
except Exception:  # scipy is in the qi2 env; math.erfc is the fallback
    _erfc = math.erfc

VERSION = 2
NAN = float("nan")
PROBES = [1.5, 2.0, 4.0, 8.0, 16.0]
BIN_EDGES = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 60.0]
FIT_RANGES = (("far [4, 32] r_s", 4.0, 32.0), ("mid [2, 32] r_s", 2.0, 32.0), ("all [2, 60] r_s", 2.0, 60.0), ("near [1.5, 4] r_s", 1.5, 4.0))
B_L_DEFAULT = 0.0194139
QUANT = ("clock_ref", "flow_ref", "gen_ref", "clock_C", "flow_C", "gen_C", "clock_NO", "flow_NO", "gen_NO")
SUBKEY = {"raw": "E_raw_at", "NO": "E_NO_at", "R2": "E_R2_at"}
OFFSETS = (-2, -1, 0, 1, 2)
RULES = {"residual_sharp_abs": 1e-13, "residual_smooth_over_tail": 100.0, "residual_integrated_over_slope": 1e-5, "identity_rel": 1e-9,
         "static_spread_rel": 1e-6, "arith_residual_over_slope": 1e-5, "alpha_window": (-2.5, -1.5), "alpha_log_sigmas": 3.0, "ratio_tol": 0.10,
         "two_size_tol": 0.05, "rs_linearity_tol": 0.01, "annulus_threshold_over_slope": 1e-5, "record_consistency_rel": 1e-8, "running_vs_totals_abs": 1e-9}


# ----------------------------------------------------------------------------------------------------------------------------- utilities
def load(path):
    with open(path) as f:
        return json.load(f)


def nmax(x):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else NAN


def trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if len(x) < 2:
        return NAN
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def fmt(x, f="%.4e"):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return "nan"
    return f % xf


def clean(o):
    """json-ready: numpy scalars and arrays to python, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def table(md, headers, rows):
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "---|" * len(headers))
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")


def verdict(RES, test, value, rule, ok, note="", kind="control"):
    """kind 'control': pass/fail, a failure being a defect (of the test or of the records) to be judged; 'size': met / not met, a reported
    number held against a literal threshold, not a defect when not met; 'reported': no rule, the numbers stated."""
    if kind == "reported":
        word = "reported"
    elif ok is None:
        word = "not available"
    elif kind == "size":
        word = "met" if ok else "not met"
    else:
        word = "pass" if ok else "fail"
    RES.setdefault("verdicts", []).append({"test": test, "kind": kind, "value": value, "rule": rule, "verdict": word, "note": note})


def slope_full_zone(BL, rs):
    """the full-zone flat comparator's clock charge per e-fold, (9 pi/2) B_L r_s (review 1.25: 8.78/a at r_s/a 32 and m a 0)."""
    return 4.5 * math.pi * BL * rs


def describe(obj, depth, prefix="", lines=None, maxkeys=80):
    if lines is None:
        lines = []
    if isinstance(obj, dict):
        keys = list(obj.keys())
        lines.append("%s dict %d keys: %s%s" % (prefix, len(keys), ", ".join(str(k) for k in keys[:maxkeys]), " ..." if len(keys) > maxkeys else ""))
        if depth > 0:
            for k in keys[:maxkeys]:
                describe(obj[k], depth - 1, prefix + "/" + str(k), lines, maxkeys)
    elif isinstance(obj, list):
        lines.append("%s list %d" % (prefix, len(obj)))
        if depth > 0 and obj:
            describe(obj[0], depth - 1, prefix + "[0]", lines, maxkeys)
    else:
        lines.append("%s %s %s" % (prefix, type(obj).__name__, repr(obj)[:70]))
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------- fits
def sinhc(x):
    return math.sinh(x) / x if abs(x) > 1e-12 else 1.0


def fit_powerlaw(lo, hi, E, iters=8):
    """the annulus energies E_j over [lo_j, hi_j] of a line density A R^(alpha - 1): E_j = A (hi^alpha - lo^alpha)/alpha
    = A ln(q) R_geo^alpha sinhc(alpha ln q / 2) with q = hi/lo and R_geo = sqrt(lo hi) (alpha = 0: the logarithm, A ln q). Least squares in
    ln(E/ln q) against ln R_geo with the width factor removed iteratively (exact for a pure power law with bins of unequal width; for
    doubling bins the factor is common and drops out). Points with E <= 0 are dropped and counted. Errors: the standard error from the
    scatter (n > 2) and the jackknife half-range over single deletions (n > 3); err is the larger. The ratio per doubling is 2^-alpha
    (an annulus exponent -2 means the line density R^-3, the 3-dimensional density r^-5, 4 per doubling; alpha 0 is the logarithm)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    E = np.asarray(E, float)
    m = np.isfinite(E) & (E > 0) & (lo > 0) & (hi > lo)
    res = {"n": int(m.sum()), "n_dropped": int(len(E) - m.sum()), "alpha": NAN, "se": NAN, "jack_half_range": NAN, "err": NAN,
           "ratio_per_doubling": NAN, "lnA": NAN}
    if res["n"] < 2:
        return res
    lo, hi, E = lo[m], hi[m], E[m]
    lq = np.log(hi / lo)
    x = 0.5 * np.log(lo * hi)
    y0 = np.log(E / lq)

    def solve(xs, ys, lqs):
        alpha = 0.0
        coef, A, ysc = None, None, None
        for _ in range(iters):
            ysc = ys - np.log(np.array([sinhc(alpha * l / 2.0) for l in lqs]))
            A = np.vstack([np.ones(len(xs)), xs]).T
            coef = np.linalg.lstsq(A, ysc, rcond=None)[0]
            alpha = float(coef[1])
        return coef, A, ysc

    coef, A, yc = solve(x, y0, lq)
    res["lnA"], res["alpha"] = float(coef[0]), float(coef[1])
    n = res["n"]
    if n > 2:
        r = yc - A @ coef
        cov = (float(r @ r) / (n - 2)) * np.linalg.inv(A.T @ A)
        res["se"] = float(math.sqrt(max(cov[1, 1], 0.0)))
    if n > 3:
        al = [solve(np.delete(x, i), np.delete(y0, i), np.delete(lq, i))[0][1] for i in range(n)]
        res["jack_half_range"] = float((max(al) - min(al)) / 2.0)
    vals = [v for v in (res["se"], res["jack_half_range"]) if np.isfinite(v)]
    res["err"] = float(max(vals)) if vals else NAN
    res["ratio_per_doubling"] = float(2.0 ** (-res["alpha"]))
    return res


def fit_even(s, y, orders=(2, 4)):
    """y = sum_k c_k s^k over the even orders, least squares in the scaled variable s/max|s|; the errors from the residual variance."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    sc = float(np.max(np.abs(s))) or 1.0
    u = s / sc
    A = np.vstack([u ** k for k in orders]).T
    coef_u = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ coef_u
    dof = len(y) - len(orders)
    if dof > 0:
        cov = (float(r @ r) / dof) * np.linalg.inv(A.T @ A)
        err_u = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        err_u = np.full(len(orders), NAN)
    scale = np.array([sc ** k for k in orders], float)
    return coef_u / scale, err_u / scale, r


def small_s(s, y, smax):
    """the s^2 coefficient of y(s) from y = c2 s^2 + c4 s^4 on 0 < s <= smax; the error is the larger of the fit's standard error and the
    change of c2 when smax is halved (the s^6 term's effect)."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    out = {"c2": NAN, "c4": NAN, "se": NAN, "half_shift": NAN, "err": NAN, "n": 0, "smax": smax}
    m = np.isfinite(y) & (s > 0) & (s <= smax)
    if m.sum() < 3:
        return out
    coef, err, _ = fit_even(s[m], y[m])
    out.update(c2=float(coef[0]), c4=float(coef[1]), se=float(err[0]), n=int(m.sum()))
    m2 = np.isfinite(y) & (s > 0) & (s <= smax / 2.0)
    if m2.sum() >= 3:
        out["half_shift"] = float(abs(fit_even(s[m2], y[m2])[0][0] - coef[0]))
    elif m2.sum() >= 2:
        out["half_shift"] = float(abs(fit_even(s[m2], y[m2], (2,))[0][0] - coef[0]))
    vals = [v for v in (out["se"], out["half_shift"]) if np.isfinite(v)]
    out["err"] = float(max(vals)) if vals else NAN
    return out


def intercept(s, y, npts=3):
    """a + b s on the first npts finite points with s > 0 (the s to 0 extrapolation of a quantity linear in s)."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & (s > 0)
    s_, y_ = s[m][:npts], y[m][:npts]
    if len(s_) < 2:
        return NAN, NAN
    A = np.vstack([np.ones(len(s_)), s_]).T
    a, b = np.linalg.lstsq(A, y_, rcond=None)[0]
    return float(a), float(b)


def ratio_err(num, den, enum, eden):
    """num/den with the propagated error sqrt((enum/den)^2 + (num eden/den^2)^2)."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return NAN, NAN
    r = num / den
    e1 = (enum / den) if np.isfinite(enum) else 0.0
    e2 = (num * eden / den ** 2) if np.isfinite(eden) else 0.0
    return float(r), float(math.sqrt(e1 ** 2 + e2 ** 2))


# ----------------------------------------------------------------------------------------------------------------------- vf9/vf10 records
class VF:
    """an engine_vacflow_v9/v10 record."""

    def __init__(self, D, name, path=""):
        self.D, self.name, self.path = D, name, path
        self.rs = float(D["args"]["rs"])
        self.ch = D["channels"]
        self.nfull = max(len(c["reads"]) for c in self.ch)
        first = next(c for c in self.ch if c["reads"])
        self.cuts = sorted(first["reads"][0]["cuts"].keys(), key=float)
        self.t_r = float(D["args"].get("tr", 32.0)) * self.rs
        self.current = str(D["args"].get("chain_current", "?"))

    def w(self, c):
        return float(c.get("weight", 16.0 * c["kappa"]))

    def complete(self, c):
        return (not c.get("rejected")) and len(c["reads"]) == self.nfull

    def arr(self, c, key):
        return np.array([q.get(key, NAN) for q in c["reads"]], float)

    def cut_arr(self, c, ck, key):
        return np.array([q.get("cuts", {}).get(ck, {}).get(key, NAN) for q in c["reads"]], float)

    def cut_rs(self):
        return min(self.cuts, key=lambda k: abs(float(k) - self.rs))

    def regions(self):
        reg = [("inside %g r_s" % (float(k) / self.rs), "cut", (k,)) for k in self.cuts]
        for lo, hi in zip(self.cuts[:-1], self.cuts[1:]):
            reg.append(("annulus [%g, %g] r_s" % (float(lo) / self.rs, float(hi) / self.rs), "annulus", (lo, hi)))
        return reg

    def smooth_tail(self):
        if "smooth_tail" in self.D:
            return float(self.D["smooth_tail"])
        sw = float(self.D["args"].get("smooth_width", 7.0))
        swin = int(self.D["args"].get("smooth_window", 64))
        return 0.5 * float(_erfc((swin // 2) / sw))


def decompose(V, c):
    """per readout, inside each cut: the generator, flow and clock energies of R0 relative to the flat sea (ref; static), of C relative
    to the flat sea (C; raw) and of C relative to R0 (NO). gen = E_K difference, flow = s times the V-expectation inside the cut read as
    W/sdot where sdot != 0 (nan where sdot = 0, 0 where s = 0), clock = gen - flow."""
    s = V.arr(c, "s")
    sdot = V.arr(c, "sdot")
    ok = sdot != 0
    d = {"s": s, "sdot": sdot, "t": V.arr(c, "t_over_rs") * V.rs, "phase": [q.get("phase", "?") for q in c["reads"]], "ok": ok, "cuts": {}}
    for ck in V.cuts:
        EKC = V.cut_arr(c, ck, "E_K_C")
        EKR0 = V.cut_arr(c, ck, "E_K_R0")
        WC = V.cut_arr(c, ck, "W_C")
        WR0 = V.cut_arr(c, ck, "W_R0")
        W = V.cut_arr(c, ck, "W")
        with np.errstate(divide="ignore", invalid="ignore"):
            TrCV = np.where(ok, WC / sdot, NAN)
            TrR0V = np.where(ok, WR0 / sdot, NAN)
            TrDV = np.where(ok, W / sdot, NAN)
        q = {"TrCV": TrCV, "TrR0V": TrR0V}
        for tag, EA, E0, T in (("ref", EKR0, EKR0[0], TrR0V), ("C", EKC, EKC[0], TrCV)):
            gen = EA - E0
            flow = np.where(s == 0, 0.0, s * T)
            q["gen_" + tag], q["flow_" + tag], q["clock_" + tag] = gen, flow, gen - flow
        gen = EKC - EKR0
        flow = np.where(s == 0, 0.0, s * TrDV)
        q["gen_NO"], q["flow_NO"], q["clock_NO"] = gen, flow, gen - flow
        d["cuts"][ck] = q
    return d


def weighted_sum(V, decs, key, ck, which="all"):
    """the weighted channel sum per readout index of a decomposed quantity; 'all' sums the channels present at the index, 'complete'
    only the complete channels; nan where no channel contributes."""
    tot = np.zeros(V.nfull)
    cnt = np.zeros(V.nfull, int)
    for c, d in zip(V.ch, decs):
        if which == "complete" and not V.complete(c):
            continue
        y = np.asarray(d["cuts"][ck][key], float)
        k = len(y)
        m = np.isfinite(y)
        tot[:k][m] += V.w(c) * y[m]
        cnt[:k][m] += 1
    return np.where(cnt > 0, tot, NAN), cnt


def weighted_region(V, decs, key, keys, which):
    if len(keys) == 1:
        return weighted_sum(V, decs, key, keys[0], which)
    y_hi, cnt = weighted_sum(V, decs, key, keys[1], which)
    y_lo, _ = weighted_sum(V, decs, key, keys[0], which)
    return y_hi - y_lo, cnt


def weighted_top(V, key, which="all"):
    tot = np.zeros(V.nfull)
    cnt = np.zeros(V.nfull, int)
    for c in V.ch:
        if which == "complete" and not V.complete(c):
            continue
        y = V.arr(c, key)
        k = len(y)
        m = np.isfinite(y)
        tot[:k][m] += V.w(c) * y[m]
        cnt[:k][m] += 1
    return np.where(cnt > 0, tot, NAN), cnt


def analyze_vf(V, a, RES, md):
    slope = slope_full_zone(a.BL, V.rs)
    tail = V.smooth_tail()
    out = {"file": V.path, "rs": V.rs, "chain_current": V.current, "n_channels": len(V.ch), "n_readouts_full": V.nfull, "cuts": V.cuts,
           "slope_full_zone_per_efold": slope, "smooth_tail": tail}
    comp = [c["kappa"] for c in V.ch if V.complete(c)]
    out["complete_channels"] = comp
    out["channels_apart"] = [{"kappa": c["kappa"], "readouts": len(c["reads"]), "why": str(c.get("rejected") or "readout count")} for c in V.ch if not V.complete(c)]
    decs = [decompose(V, c) for c in V.ch]
    iref = next((i for i, c in enumerate(V.ch) if V.complete(c)), int(np.argmax([len(c["reads"]) for c in V.ch])))
    s_grid, sdot_grid, phase = decs[iref]["s"], decs[iref]["sdot"], decs[iref]["phase"]
    ok = sdot_grid != 0
    ramp_ok = [i for i in range(V.nfull) if phase[i] == "ramp" and ok[i]]
    i_end = max(ramp_ok) if ramp_ok else V.nfull - 1
    i_hold = V.nfull - 1
    out["readouts"] = {"s_first_12": s_grid[:12].tolist(), "i_end_of_ramp": int(i_end), "s_end": float(s_grid[i_end]), "sdot_end": float(sdot_grid[i_end]),
                       "t_end_over_rs": float(decs[iref]["t"][i_end] / V.rs), "t_hold_end_over_rs": float(decs[iref]["t"][i_hold] / V.rs)}
    md += ["## %s: %s" % (V.name, V.path), "",
           "- r_s %g, chain current %s, %d channels, %d readouts on the full grid; complete channels %s; apart: %s" % (
               V.rs, V.current, len(V.ch), V.nfull, comp, [(x["kappa"], x["readouts"]) for x in out["channels_apart"]] or "none"),
           "- the full-zone flat slope (9 pi/2) B_L r_s = %.4f per e-fold at a^2 B_L = %g; the smoothed cut's tail at the window's edge %.2e" % (slope, a.BL, tail),
           "- the ramp's last readout with sdot != 0: index %d, t/r_s %.2f, s %.8f, sdot %.3e (the flow part at s = 1 is read there by W/sdot)" % (
               i_end, out["readouts"]["t_end_over_rs"], out["readouts"]["s_end"], out["readouts"]["sdot_end"]), ""]

    # (1) the continued-reference residual
    res = {}
    rows = []
    for ck in V.cuts:
        acc = {k: [] for k in ("I_C", "dI_R2", "dI_R0", "IS_C", "dIS_R2", "dIS_R0")}
        integ_R2, integ_R0, have_R2 = 0.0, 0.0, False
        for c in V.ch:
            t = V.arr(c, "t_over_rs") * V.rs
            IC = V.cut_arr(c, ck, "I_C")
            INO = V.cut_arr(c, ck, "I_NO")
            IR2 = V.cut_arr(c, ck, "I_R2")
            ISC = V.cut_arr(c, ck, "IS_C")
            ISNO = V.cut_arr(c, ck, "IS_NO")
            ISR2 = V.cut_arr(c, ck, "IS_R2")
            acc["I_C"].append(IC)
            acc["dI_R0"].append(IC - INO)
            acc["dI_R2"].append(IC - IR2)
            acc["IS_C"].append(ISC)
            acc["dIS_R0"].append(ISC - ISNO)
            acc["dIS_R2"].append(ISC - ISR2)
            if np.isfinite(IR2).any():
                have_R2 = True
                integ_R2 += V.w(c) * trapz(np.abs(IC - IR2), t)
            integ_R0 += V.w(c) * trapz(np.abs(IC - INO), t)
        scale = nmax(np.abs(np.concatenate(acc["I_C"])))

        def stat(key, acc=acc):
            v = np.concatenate(acc[key])
            v = v[np.isfinite(v)]
            return {"max_abs": float(np.max(np.abs(v))) if len(v) else NAN, "rms": float(np.sqrt(np.mean(v ** 2))) if len(v) else NAN, "n": int(len(v))}

        r = {"R_over_rs": float(ck) / V.rs, "max_abs_I_C": scale, "sharp_R2": stat("dI_R2"), "sharp_R0": stat("dI_R0"), "smooth_R2": stat("dIS_R2"),
             "smooth_R0": stat("dIS_R0"), "integrated_abs_R2_weighted": integ_R2 if have_R2 else NAN, "integrated_abs_R0_weighted": integ_R0}
        r["integrated_R2_over_slope_ln2"] = r["integrated_abs_R2_weighted"] / (slope * math.log(2.0))
        r["sharp_R2_rel_info"] = r["sharp_R2"]["max_abs"] / scale if scale else NAN
        r["smooth_R2_rel_info"] = r["smooth_R2"]["max_abs"] / scale if scale else NAN
        r["smooth_R2_over_tail"] = r["smooth_R2"]["max_abs"] / tail if tail else NAN
        r["smooth_R0_over_tail"] = r["smooth_R0"]["max_abs"] / tail if tail else NAN
        res[ck] = r
        rows.append([fmt(r["R_over_rs"], "%g"), fmt(scale), fmt(r["sharp_R2"]["max_abs"]), fmt(r["sharp_R2"]["rms"]), fmt(r["sharp_R0"]["max_abs"]),
                     fmt(r["smooth_R2"]["max_abs"]), fmt(r["smooth_R0"]["max_abs"]), fmt(r["sharp_R2_rel_info"], "%.2e"), fmt(r["smooth_R2_over_tail"], "%.2f"),
                     fmt(r["integrated_R2_over_slope_ln2"], "%.2e")])
    out["residual"] = res
    md += ["### (1) The continued-reference residual: the reference's Killing-energy current through each cut, I_C - I_R2 = (i/2) Tr[R2 [h^2, P_c]] (zero for a projector commuting with h(s); the measured value is the rounding of O(1) summands); R0 likewise; the smoothed cut IS_C - IS_R2 carries the window truncation of the erf step's tail and is read in units of the tail", ""]
    table(md, ["cut R/r_s", "max |I_C| (information)", "sharp R2 max", "sharp R2 rms", "sharp R0 max", "smooth R2 max", "smooth R0 max", "sharp R2 / max|I_C| (information)",
               "smooth R2 / tail", "sum_k w int |I_C - I_R2| dt / (slope ln 2)"], rows)
    sharp_abs = nmax([r["sharp_R2"]["max_abs"] for r in res.values()])
    sharp_rms = nmax([r["sharp_R2"]["rms"] for r in res.values()])
    sharp_rel = nmax([r["sharp_R2_rel_info"] for r in res.values()])
    smooth_tails = nmax([r["smooth_R2_over_tail"] for r in res.values()])
    integ_rel = nmax([r["integrated_R2_over_slope_ln2"] for r in res.values()])
    verdict(RES, "%s T1 continued-reference residual, sharp cut: max |I_C - I_R2| over channels, readouts and cuts (absolute)" % V.name, sharp_abs,
            "< %g (the rounding of O(1) summands)" % RULES["residual_sharp_abs"], (sharp_abs < RULES["residual_sharp_abs"]) if np.isfinite(sharp_abs) else None,
            "rms %s; relative to the record's largest |I_C| %s (information: that scale depends on how far the pocket channels ran)" % (fmt(sharp_rms, "%.1e"), fmt(sharp_rel, "%.1e")))
    verdict(RES, "%s T1 continued-reference residual, smoothed cut: max |IS_C - IS_R2| in units of the step's tail at the window's edge" % V.name, smooth_tails,
            "< %g tails (tail %.2e)" % (RULES["residual_smooth_over_tail"], tail), (smooth_tails < RULES["residual_smooth_over_tail"]) if np.isfinite(smooth_tails) else None,
            "the window truncation of the erf step, not a property of the reference; the sharp cut is the exact statement")
    verdict(RES, "%s T1 integrated |I_C - I_R2| (weighted) over slope ln 2" % V.name, integ_rel, "< %g" % RULES["residual_integrated_over_slope"],
            (integ_rel < RULES["residual_integrated_over_slope"]) if np.isfinite(integ_rel) else None, "an upper bound on any energy the reference could have moved across a cut over the run")

    # (2) the field identities at the cut r_s
    ck0 = V.cut_rs()
    dNO, draw, sc = [], [], []
    for c, d in zip(V.ch, decs):
        q = d["cuts"][ck0]
        m = d["ok"]
        EinNO = V.arr(c, "E_in_NO")
        Einraw = V.arr(c, "E_in_raw")
        dNO.append(nmax(np.abs(q["clock_NO"][m] - EinNO[m])))
        draw.append(nmax(np.abs(q["clock_C"][m] - Einraw[m])))
        sc.append(nmax(np.abs(np.concatenate([EinNO, Einraw, V.cut_arr(c, ck0, "E_K_C")]))))
    ident = {"cut": ck0, "max_dev_E_in_NO": nmax(dNO), "max_dev_E_in_raw": nmax(draw), "scale": nmax(sc)}
    ident["rel"] = max(ident["max_dev_E_in_NO"], ident["max_dev_E_in_raw"]) / ident["scale"] if ident["scale"] else NAN
    out["identity"] = ident
    md += ["### (2) The field identities at the cut R = r_s (the reading of E_K, W and E_in): E_in_NO = E_K_C - E_K_R0 - s W/sdot and E_in_raw = E_K_C(t) - E_K_C(0) - s W_C/sdot over the readouts with sdot != 0", "",
           "- max deviation %.3e (NO) and %.3e (raw) against the scale %.3e: relative %.2e" % (ident["max_dev_E_in_NO"], ident["max_dev_E_in_raw"], ident["scale"], ident["rel"]), ""]
    verdict(RES, "%s T2 field identities at r_s (relative deviation)" % V.name, ident["rel"], "< %g" % RULES["identity_rel"], (ident["rel"] < RULES["identity_rel"]) if np.isfinite(ident["rel"]) else None)

    # (3) the decomposition: the small-s coefficients, the ramp's end, the hold's end
    dec = {}
    rows_small, rows_end, rows_hold = [], [], []
    idx_s = np.where((s_grid > 0) & (s_grid <= a.smax))[0]
    for label, kind, keys in V.regions():
        lnr = math.log(float(keys[1]) / float(keys[0])) if kind == "annulus" else NAN
        reg = {"kind": kind, "keys": list(keys), "ln_R_ratio": lnr, "small_s": {}, "end_of_ramp": {}, "hold_end": {}}
        for key in QUANT:
            y_all, cnt = weighted_region(V, decs, key, keys, "all")
            fit = small_s(s_grid, y_all, a.smax)
            fit["channels_summed_at_smax"] = int(cnt[idx_s[-1]]) if len(idx_s) else 0
            reg["small_s"][key] = fit
            y_c, _ = weighted_region(V, decs, key, keys, "complete")
            reg["end_of_ramp"][key] = float(y_c[i_end])
            reg["hold_end"][key] = float(y_c[i_hold])
        c2 = {k: reg["small_s"][k]["c2"] for k in QUANT}
        e2 = {k: reg["small_s"][k]["err"] for k in QUANT}
        B = -c2["flow_ref"]
        own2 = c2["gen_ref"] - c2["flow_ref"] / 2.0
        rf, rfe = ratio_err(c2["flow_ref"], c2["clock_ref"], e2["flow_ref"], e2["clock_ref"])
        rg, rge = ratio_err(c2["gen_ref"], c2["clock_ref"], e2["gen_ref"], e2["clock_ref"])
        der = {"B": B, "B_err": e2["flow_ref"], "own2": own2, "own2_over_B": (own2 / B) if B else NAN, "own_share_2own_over_B": (2 * own2 / B) if B else NAN,
               "local_share_halfB_over_clock2": (0.5 * B / c2["clock_ref"]) if c2["clock_ref"] else NAN,
               "flow_over_clock": rf, "flow_over_clock_err": rfe, "gen_over_clock": rg, "gen_over_clock_err": rge,
               "ratio_predicted_from_own_share": [(-2.0 / (1.0 + 2 * own2 / B)) if (B and (1.0 + 2 * own2 / B) != 0) else NAN,
                                                  ((-1.0 + 2 * own2 / B) / (1.0 + 2 * own2 / B)) if (B and (1.0 + 2 * own2 / B) != 0) else NAN],
               "signs_ok": bool(np.isfinite(c2["clock_ref"]) and c2["clock_ref"] > 0 and c2["flow_ref"] < 0 and c2["gen_ref"] < 0),
               "ratio_within_tol": bool(np.isfinite(rf) and np.isfinite(rg) and abs(rf + 2.0) <= 2.0 * RULES["ratio_tol"] and abs(rg + 1.0) <= RULES["ratio_tol"]),
               "excitation_c2_gen_NO": c2["gen_NO"], "excitation_c4_gen_NO": reg["small_s"]["gen_NO"]["c4"]}
        if kind == "annulus":
            der["slope_per_efold_s2"] = {k: c2[k] / lnr for k in ("clock_ref", "flow_ref", "gen_ref")}
        reg["derived"] = der
        dec[label] = reg
        rows_small.append([label, fmt(c2["clock_ref"]), fmt(c2["flow_ref"]), fmt(c2["gen_ref"]), fmt(e2["clock_ref"], "%.1e"), fmt(B), fmt(own2),
                           fmt(der["own_share_2own_over_B"], "%.3e"), fmt(der["local_share_halfB_over_clock2"], "%.3e"),
                           "%s +- %s" % (fmt(rf, "%.4f"), fmt(rfe, "%.1e")), "%s +- %s" % (fmt(rg, "%.4f"), fmt(rge, "%.1e")), fmt(der["signs_ok"]), fmt(der["ratio_within_tol"])])
        e = reg["end_of_ramp"]
        rows_end.append([label] + [fmt(e[k]) for k in QUANT])
        h = reg["hold_end"]
        rows_hold.append([label, fmt(h["gen_ref"]), fmt(h["gen_C"]), fmt(h["gen_NO"])])
    out["decomposition"] = dec
    md += ["### (3) The clock, flow and generator parts inside the cuts and in the annuli between them: the s^2 coefficients at 0 < s <= %g of the static response of R0 relative to the flat sea (weighted channel sums; the virial prediction clock : flow : generator = 1 : -2 : -1 up to the cut's own term: clock2 = B/2 + own2, flow2 = -B, gen2 = -B/2 + own2, own share = 2 own2/B, local share = (B/2)/clock2; the ratios predicted from the own share are -2/(1 + own share) and (-1 + own share)/(1 + own share))" % a.smax, ""]
    table(md, ["region", "clock_ref c2", "flow_ref c2", "gen_ref c2", "err(clock c2)", "B = -flow c2", "own2 = gen2 - flow2/2", "own share 2 own2/B", "local share (B/2)/clock2",
               "flow/clock (-2)", "gen/clock (-1)", "signs", "within tol"], rows_small)
    md += ["The same parts at the ramp's last readout (s = %.6f; complete channels; the exterior annuli have v^2 = 0.5 to 0.67 there and no quadratic ratio is expected):" % out["readouts"]["s_end"], ""]
    table(md, ["region"] + list(QUANT), rows_end)
    md += ["The generators at the hold's end (t/r_s %.1f; the flow parts are not readable at sdot = 0):" % out["readouts"]["t_hold_end_over_rs"], ""]
    table(md, ["region", "gen_ref (R0 - flat)", "gen_C (C - flat)", "gen_NO (C - R0)"], rows_hold)
    big = "inside %g r_s" % (float(V.cuts[-1]) / V.rs)
    dbig = dec[big]["derived"]
    verdict(RES, "%s T4 signs of the s^2 response (clock > 0, flow < 0, generator < 0) in %s" % (V.name, big), dbig["signs_ok"], "all three", dbig["signs_ok"])
    verdict(RES, "%s T4 ratios flow/clock, gen/clock in %s" % (V.name, big), [dbig["flow_over_clock"], dbig["gen_over_clock"]],
            "within %g of -2 and -1" % RULES["ratio_tol"], dbig["ratio_within_tol"],
            "own share 2 own2/B = %s; the ratios predicted from it %s, %s" % (fmt(dbig["own_share_2own_over_B"], "%.3e"), fmt(dbig["ratio_predicted_from_own_share"][0], "%.4f"), fmt(dbig["ratio_predicted_from_own_share"][1], "%.4f")))
    for label, reg in dec.items():
        dr = reg["derived"]
        if reg["kind"] == "cut" and label != big:
            verdict(RES, "%s T4 ratios flow/clock, gen/clock %s" % (V.name, label), [dr["flow_over_clock"], dr["gen_over_clock"]],
                    "the rule (within %g of -2 and -1) is set for the largest cut; the smaller cuts are reported" % RULES["ratio_tol"], None,
                    "own share 2 own2/B = %s; predicted from it %s, %s; signs as predicted: %s" % (fmt(dr["own_share_2own_over_B"], "%.3e"), fmt(dr["ratio_predicted_from_own_share"][0], "%.4f"),
                                                                                                   fmt(dr["ratio_predicted_from_own_share"][1], "%.4f"), fmt(dr["signs_ok"])), kind="reported")
        elif reg["kind"] == "annulus":
            verdict(RES, "%s T4 the %s: flow/clock, gen/clock" % (V.name, label), [dr["flow_over_clock"], dr["gen_over_clock"]],
                    "no rule: the virial ratio is a statement for a region whose boundary term is small; the own share and the local share say how far this region is from that", None,
                    "own2 = %s, own share 2 own2/B = %s, local share (B/2)/clock2 = %s, B_ann = %s; signs as predicted: %s" % (
                        fmt(dr["own2"]), fmt(dr["own_share_2own_over_B"], "%.3e"), fmt(dr["local_share_halfB_over_clock2"], "%.3e"), fmt(dr["B"]), fmt(dr["signs_ok"])), kind="reported")

    # per channel at the largest cut
    rows = []
    perch = []
    for c, d in zip(V.ch, decs):
        q = d["cuts"][V.cuts[-1]]
        f = {k: small_s(d["s"], q[k], a.smax)["c2"] for k in ("clock_ref", "flow_ref", "gen_ref")}
        Bk = -f["flow_ref"]
        ownk = f["gen_ref"] - f["flow_ref"] / 2.0
        row = {"kappa": c["kappa"], "B": Bk, "own_share": (2 * ownk / Bk) if Bk else NAN, "flow_over_clock": (f["flow_ref"] / f["clock_ref"]) if f["clock_ref"] else NAN,
               "gen_over_clock": (f["gen_ref"] / f["clock_ref"]) if f["clock_ref"] else NAN}
        perch.append(row)
        rows.append([c["kappa"], fmt(Bk), fmt(row["own_share"], "%.3e"), fmt(row["flow_over_clock"], "%.4f"), fmt(row["gen_over_clock"], "%.4f")])
    out["per_channel_largest_cut"] = perch
    md += ["Per channel in %s (unweighted; the same fit):" % big, ""]
    table(md, ["kappa", "B", "own share", "flow/clock", "gen/clock"], rows)

    # the intercepts: Tr[flat V_c] (expected 0 by time reversal)
    inter = {}
    rows = []
    for ck in V.cuts:
        yC, _ = weighted_sum(V, decs, "TrCV", ck, "all")
        yR, _ = weighted_sum(V, decs, "TrR0V", ck, "all")
        aC, bC = intercept(s_grid, yC)
        aR, bR = intercept(s_grid, yR)
        inter[ck] = {"TrCV_intercept": aC, "TrCV_slope": bC, "TrR0V_intercept": aR, "TrR0V_slope": bR, "intercept_over_B": (aR / abs(bR)) if bR else NAN,
                     "C_intercept_over_B": (aC / abs(bC)) if bC else NAN}
        rows.append([fmt(float(ck) / V.rs, "%g"), fmt(aR), fmt(bR), fmt(aC), fmt(bC), fmt(inter[ck]["intercept_over_B"], "%.2e"), fmt(inter[ck]["C_intercept_over_B"], "%.2e")])
    out["flat_flow_intercepts"] = inter
    md += ["The s to 0 intercepts of Tr[R0(s) V_c] = W_R0/sdot and Tr[C V_c] = W_C/sdot (the first 3 readouts; the intercept is Tr[flat V_c], expected 0 by time reversal; its size is the systematic error of the flow parts relative to the flat sea; the slope is -B_c):", ""]
    table(md, ["cut R/r_s", "R0 intercept", "R0 slope", "C intercept", "C slope", "R0 intercept / |slope|", "C intercept / |slope|"], rows)

    # the adiabaticity at smax and the clock energies in/out of r_s
    EinNO, _ = weighted_top(V, "E_in_NO", "all")
    Einraw, _ = weighted_top(V, "E_in_raw", "all")
    EoutNO, _ = weighted_top(V, "E_out_NO", "all")
    Eoutraw, _ = weighted_top(V, "E_out_raw", "all")
    i_s = int(idx_s[-1]) if len(idx_s) else 0
    adiab = {"i": i_s, "s": float(s_grid[i_s]), "E_in_NO": float(EinNO[i_s]), "E_in_raw": float(Einraw[i_s]), "E_out_NO": float(EoutNO[i_s]), "E_out_raw": float(Eoutraw[i_s])}
    adiab["in_NO_over_raw"] = adiab["E_in_NO"] / adiab["E_in_raw"] if adiab["E_in_raw"] else NAN
    adiab["out_NO_over_raw"] = adiab["E_out_NO"] / adiab["E_out_raw"] if adiab["E_out_raw"] else NAN
    out["adiabaticity_at_smax"] = adiab
    md += ["- adiabaticity at s = %.4f (readout %d): E_in_NO/E_in_raw = %s, E_out_NO/E_out_raw = %s (the excitation's share of the clock response; E_out_NO %s, E_out_raw %s)" % (
        adiab["s"], i_s, fmt(adiab["in_NO_over_raw"], "%.3e"), fmt(adiab["out_NO_over_raw"], "%.3e"), fmt(adiab["E_out_NO"]), fmt(adiab["E_out_raw"])), ""]

    # the exterior at the hold's end
    rows = []
    ext = []
    sums = {"E_out_raw_minus_NO": 0.0, "E_out_R2_minus_NO": 0.0, "E_in_raw_minus_NO": 0.0, "E_in_R2_minus_NO": 0.0, "E_out_NO": 0.0, "E_out_R2": 0.0, "N_p": 0.0, "N_h": 0.0}
    for c in V.ch:
        q = c["reads"][-1]
        pk = c.get("continuation", {}).get("pockets_max", None)
        if pk is None:
            pk = max([(r.get("continuation") or {}).get("n_pocket", 0) or 0 for r in c["reads"]] + [0])
        row = {"kappa": c["kappa"], "t_over_rs": q["t_over_rs"], "complete": V.complete(c), "E_in_NO": q.get("E_in_NO"), "E_in_R2": q.get("E_in_R2"), "E_out_NO": q.get("E_out_NO"),
               "E_out_R2": q.get("E_out_R2"), "E_out_raw": q.get("E_out_raw"), "N_p": q.get("N_p"), "N_h": q.get("N_h"), "pockets_max": pk}
        row["E_out_raw_minus_NO"] = (q.get("E_out_raw", NAN) - q.get("E_out_NO", NAN))
        row["E_out_R2_minus_NO"] = (q.get("E_out_R2", NAN) - q.get("E_out_NO", NAN))
        ext.append(row)
        if V.complete(c):
            w = V.w(c)
            sums["E_out_raw_minus_NO"] += w * row["E_out_raw_minus_NO"]
            sums["E_out_R2_minus_NO"] += w * row["E_out_R2_minus_NO"]
            sums["E_in_raw_minus_NO"] += w * (q.get("E_in_raw", NAN) - q.get("E_in_NO", NAN))
            sums["E_in_R2_minus_NO"] += w * (q.get("E_in_R2", NAN) - q.get("E_in_NO", NAN))
            sums["E_out_NO"] += w * q.get("E_out_NO", NAN)
            sums["E_out_R2"] += w * q.get("E_out_R2", NAN)
            sums["N_p"] += w * q.get("N_p", NAN)
            sums["N_h"] += w * q.get("N_h", NAN)
        rows.append([c["kappa"], fmt(row["t_over_rs"], "%.1f"), fmt(row["complete"]), fmt(row["E_in_NO"]), fmt(row["E_in_R2"]), fmt(row["E_out_NO"]), fmt(row["E_out_R2"]),
                     fmt(row["E_out_raw_minus_NO"]), fmt(row["E_out_R2_minus_NO"]), fmt(row["N_p"], "%.2e"), fmt(row["N_h"], "%.2e"), pk])
    out["exterior_last_readout"] = {"per_channel": ext, "weighted_sums_complete": sums}
    md += ["### (4) The last readout per channel: the clock energies in and out of r_s under NO and R2, the exterior static clock response E_out_raw - E_out_NO (R0 against flat at s = 1), the exterior difference of the two references E_out_R2 - E_out_NO, the pair counts", ""]
    table(md, ["kappa", "t/r_s", "complete", "E_in_NO", "E_in_R2", "E_out_NO", "E_out_R2", "E_out raw - NO", "E_out R2 - NO", "N_p", "N_h", "pockets max"], rows)
    md += ["- weighted sums over the complete channels: E_out raw - NO (exterior static clock response at s = 1, %s current) %s; E_out R2 - NO %s; E_in raw - NO %s; E_in R2 - NO %s; E_out_NO %s; E_out_R2 %s; N_p %s; N_h %s" % (
        V.current, fmt(sums["E_out_raw_minus_NO"]), fmt(sums["E_out_R2_minus_NO"]), fmt(sums["E_in_raw_minus_NO"]), fmt(sums["E_in_R2_minus_NO"]), fmt(sums["E_out_NO"]), fmt(sums["E_out_R2"]),
        fmt(sums["N_p"], "%.3e"), fmt(sums["N_h"], "%.3e")), ""]
    RES[V.name] = out
    return out


def compare_vf(Va, Vb, a, RES, md):
    """the reference quantities (E_K_R0, W_R0) and the propagated state's (E_K_C, W_C, E_in_raw, E_in_NO), both independent of the
    continuation convention, over the common readouts of the channels present in both records."""
    kb = {c["kappa"]: c for c in Vb.ch}
    rows = []
    worst = NAN
    out = {"a": Va.name, "b": Vb.name, "per_channel": []}
    for c in Va.ch:
        if c["kappa"] not in kb:
            continue
        d = kb[c["kappa"]]
        n = min(len(c["reads"]), len(d["reads"]))
        ta = np.array([q["t_over_rs"] for q in c["reads"][:n]], float)
        tb = np.array([q["t_over_rs"] for q in d["reads"][:n]], float)
        row = {"kappa": c["kappa"], "n_common": n}
        if n == 0 or np.max(np.abs(ta - tb)) > 1e-9:
            row["note"] = "the readout grids differ"
            out["per_channel"].append(row)
            rows.append([c["kappa"], n, "grids differ", "", "", "", "", ""])
            continue
        vals = []
        for key in ("E_K_R0", "W_R0", "E_K_C", "W_C"):
            dev = []
            for ck in Va.cuts:
                if ck not in Vb.cuts:
                    continue
                x = np.array([q["cuts"][ck].get(key, NAN) for q in c["reads"][:n]], float)
                y = np.array([q["cuts"][ck].get(key, NAN) for q in d["reads"][:n]], float)
                scl = nmax(np.abs(x))
                dev.append(nmax(np.abs(x - y)) / scl if scl else NAN)
            row[key] = nmax(dev)
            vals.append(row[key])
        for key in ("E_in_raw", "E_in_NO"):
            x = np.array([q.get(key, NAN) for q in c["reads"][:n]], float)
            y = np.array([q.get(key, NAN) for q in d["reads"][:n]], float)
            scl = nmax(np.abs(x))
            row[key] = nmax(np.abs(x - y)) / scl if scl else NAN
            vals.append(row[key])
        w_ = nmax(vals)
        worst = w_ if not np.isfinite(worst) else max(worst, w_)
        out["per_channel"].append(row)
        rows.append([c["kappa"], n] + [fmt(row.get(k), "%.2e") for k in ("E_K_R0", "W_R0", "E_K_C", "W_C", "E_in_raw", "E_in_NO")])
    out["worst_rel"] = worst
    md += ["## %s against %s over the common readouts (relative maxima; the continuation convention is the only difference between the records, so all of these should be rounding)" % (Va.name, Vb.name), ""]
    table(md, ["kappa", "n common", "E_K_R0", "W_R0", "E_K_C", "W_C", "E_in_raw", "E_in_NO"], rows)
    verdict(RES, "T6a %s vs %s: worst relative difference of E_K_R0, W_R0, E_K_C, W_C, E_in_raw, E_in_NO" % (Va.name, Vb.name), worst, "< %g" % RULES["record_consistency_rel"],
            (worst < RULES["record_consistency_rel"]) if np.isfinite(worst) else None)
    RES["%s_vs_%s" % (Va.name, Vb.name)] = out
    return out


# ------------------------------------------------------------------------------------------------------------------------ vf3/vf7 records
def find_total_key(tot, sub):
    sub_l = sub.lower()
    exact = [k for k in tot if "hold" in k.lower() and sub_l in [tok.lower() for tok in k.replace("-", "_").split("_")]]
    if exact:
        return exact[0]
    loose = [k for k in tot if "hold" in k.lower() and sub_l in k.lower()]
    return loose[0] if loose else None


class OLD:
    """an engine_vacflow_v3/v7 record: running channel sums at the probes, per-site totals at the hold."""

    def __init__(self, D, name, path=""):
        self.D, self.name, self.path = D, name, path
        self.rs = float(D["args"]["rs"])
        self.kdone = int(D.get("kappa_done", 0) or 0)
        self.trs = sorted(D.get("running", {}).keys(), key=lambda k: float(k))
        self.current = str(D["args"].get("chain_current", "not in args (engine v3/v7: the two-step current)"))
        self.probes = list(PROBES)
        self.keys_seen = sorted({k for t in self.trs for x in D["running"][t] for k in x.keys()})
        if self.trs and D["running"][self.trs[0]]:
            n_probe = len(D["running"][self.trs[0]][0].get("E_raw_at", []))
            if n_probe != len(PROBES):
                self.probes = list(D["args"].get("probes", [NAN] * n_probe))[:n_probe]

    def run_last(self, t, kmax):
        xs = [x for x in self.D["running"][t] if x["kappa"] <= kmax]
        return xs[-1] if xs else None

    def cumul(self, t, kmax, sub):
        r = self.run_last(t, kmax)
        key = SUBKEY[sub]
        if r is None or key not in r:
            return None, None
        return np.array(r[key], float), int(r["kappa"])

    def annuli(self, t, kmax, sub):
        E, k = self.cumul(t, kmax, sub)
        return (np.diff(E), k) if E is not None else (None, None)

    def per_channel(self, t, sub):
        key = SUBKEY[sub]
        out = {}
        prev = None
        for x in self.D["running"][t]:
            if key not in x:
                return None
            E = np.array(x[key], float)
            d = E - prev if prev is not None else E.copy()
            out[int(x["kappa"])] = np.diff(d)
            prev = E
        return out

    def total_hold(self, t, sub):
        tot = self.D.get("totals", {}).get(t, {})
        key = find_total_key(tot, sub) if isinstance(tot, dict) else None
        if key is None:
            return None, sorted(tot.keys()) if isinstance(tot, dict) else []
        return np.array(tot[key], float), key


def cum_below(arr, N):
    """the sum of a per-site array over the indices n < N (N clipped to the array)."""
    N = int(max(0, min(len(arr), N)))
    return float(np.nansum(arr[:N])) if N > 0 else 0.0


def detect_probe_offset(arr, rs, probes, running_annuli, offsets=OFFSETS):
    """the engine's probe convention: the cumulative energy below the probe p is read as the sum over the indices n < round(p r_s) - 1 + d;
    d = 0 is 'R = n + 1 and R < p r_s' (the convention of the vf10 cuts, where 'site' is the count R < cut), d = +1 is 'R <= p r_s'.
    Returns the offset whose probe annuli reproduce the running annuli best and the max abs discrepancy per offset."""
    table_ = {}
    for d in offsets:
        cums = [cum_below(arr, int(round(p * rs)) - 1 + d) for p in probes]
        table_[int(d)] = float(np.max(np.abs(np.diff(cums) - np.asarray(running_annuli, float))))
    best = min(table_, key=lambda d: table_[d])
    return best, table_


def binned(arr, rs, edges, offset=0):
    """the sums of a per-site array over the bins with indices round(lo r_s) - 1 + offset <= n < round(hi r_s) - 1 + offset (offset 0:
    R = n + 1 in [lo r_s, hi r_s); offset +1: R in (lo r_s, hi r_s], the engine's probe convention if detected so); bins reaching past
    the chain are left out."""
    idx = np.arange(len(arr))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i_lo = max(0, int(round(lo * rs)) - 1 + offset)
        i_hi = int(round(hi * rs)) - 1 + offset
        if i_hi > len(arr):
            break
        m = (idx >= i_lo) & (idx < i_hi)
        rows.append({"lo": lo, "hi": hi, "R_geo": math.sqrt(lo * hi) * rs, "E": float(np.nansum(arr[m])), "n_sites": int(m.sum()), "ln_ratio": math.log(hi / lo)})
    return rows


def analyze_old(O, a, RES, md):
    kmax = min(a.kmax, O.kdone) if O.kdone else a.kmax
    slope = slope_full_zone(a.BL, O.rs)
    out = {"file": O.path, "rs": O.rs, "kappa_done": O.kdone, "kmax_used": kmax, "ramps_t_r_over_rs": O.trs, "chain_current": O.current, "probes_over_rs": O.probes,
           "running_keys": O.keys_seen, "slope_full_zone_per_efold": slope}
    md += ["## %s: %s" % (O.name, O.path), "", "- r_s %g, channels done %d (used kappa <= %d), ramps t_r/r_s %s, chain current %s; the running entries carry the keys %s" % (
        O.rs, O.kdone, kmax, O.trs, O.current, O.keys_seen), "- the full-zone flat slope (9 pi/2) B_L r_s = %.4f per e-fold at this r_s" % slope, ""]
    if not O.trs:
        md += ["- no running sums in the record", ""]
        RES[O.name] = out
        return out
    lo_p = np.array(O.probes[:-1], float) * O.rs
    hi_p = np.array(O.probes[1:], float) * O.rs
    ann_lab = ["[%g, %g] r_s" % (lo, hi) for lo, hi in zip(O.probes[:-1], O.probes[1:])]
    lnr = np.log(hi_p / lo_p)
    Rgeo = np.sqrt(lo_p * hi_p)
    ann = {}
    rows = []
    for t in O.trs:
        raw, k = O.annuli(t, kmax, "raw")
        no, _ = O.annuli(t, kmax, "NO")
        r2_raw, _ = O.annuli(t, kmax, "R2")
        r2_state = "absent" if r2_raw is None else ("nan" if not np.isfinite(r2_raw).any() else "stored")
        r2 = r2_raw if r2_state == "stored" else None
        st = (raw - no) if (raw is not None and no is not None) else None
        st2 = (raw - r2) if (raw is not None and r2 is not None) else None
        ann[t] = {"raw": raw, "NO": no, "R2": r2, "R2_state": r2_state, "static": st, "static_R2": st2, "kappa_summed": k}
        for j, lab in enumerate(ann_lab):
            rows.append([t, lab, fmt(raw[j]) if raw is not None else "-", fmt(no[j]) if no is not None else "-",
                         fmt(r2[j]) if r2 is not None else ("not in record" if r2_state == "absent" else "nan in record"),
                         fmt(st[j]) if st is not None else "-", fmt(st2[j]) if st2 is not None else "-",
                         fmt(st[j] / lnr[j]) if st is not None else "-", fmt(st[j] / lnr[j] / slope, "%.3e") if st is not None else "-",
                         fmt((kmax / Rgeo[j]) ** 2, "%.3e"), fmt(no[j] / lnr[j] / slope, "%.3e") if no is not None else "-",
                         fmt(r2[j] / lnr[j] / slope, "%.3e") if r2 is not None else "-"])
    out["annuli"] = {t: {k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in d.items()} for t, d in ann.items()}
    md += ["### (a) The annuli from the running channel sums (kappa <= %d) after the hold, in the engine's own probe convention: raw (against the flat sea), NO (against R0), R2 (if stored and finite), the removed response raw - NO = Tr[(R0 - flat) hcal] and raw - R2 (the sea's clock response at s = 1, no dynamics in it); per e-fold; against the full-zone flat slope; the resolved-channel fraction (kmax/R_geo)^2 for orientation; the NO and R2 energies per e-fold against the slope (the literal threshold 1e-5, a size)" % kmax, ""]
    table(md, ["t_r/r_s", "annulus", "raw", "NO", "R2", "raw - NO", "raw - R2", "(raw - NO) per e-fold", "(raw - NO) per e-fold / slope", "(kmax/R_geo)^2", "NO per e-fold / slope", "R2 per e-fold / slope"], rows)
    # the ramp spreads of the removed response (T3a) and the subtraction's arithmetic residual per e-fold (T5 control)
    for key, label in (("static", "raw - NO"), ("static_R2", "raw - R2")):
        S = np.array([ann[t][key] for t in O.trs if ann[t][key] is not None])
        if len(S) >= 2:
            mean = S.mean(axis=0)
            dev = np.max(np.abs(S - mean), axis=0)
            spread = dev / np.maximum(np.abs(mean), 1e-300)
            arith = dev / lnr / slope
            out["%s_ramp_spread_rel" % key] = spread.tolist()
            out["%s_arith_residual_per_efold_over_slope" % key] = arith.tolist()
            md += ["- the removed response %s: spread across the %d ramps (max |deviation| / |mean|): %s; the same deviation per e-fold over the slope (the subtraction's arithmetic residual): %s" % (
                label, len(S), ", ".join("%s %.2e" % (lab, sp) for lab, sp in zip(ann_lab, spread)), ", ".join("%s %.1e" % (lab, x) for lab, x in zip(ann_lab, arith))), ""]
            verdict(RES, "%s T3a%s the removed response %s identical across the %d ramps (max relative spread over the annuli)" % (O.name, "" if key == "static" else "-R2", label, len(S)),
                    float(np.max(spread)), "< %g" % RULES["static_spread_rel"], bool(np.max(spread) < RULES["static_spread_rel"]))
            verdict(RES, "%s T5 the subtraction's arithmetic residual per e-fold over the full-zone slope (the ramp-to-ramp deviation of the removed response %s)" % (O.name, label),
                    float(np.max(arith)), "< %g" % RULES["arith_residual_over_slope"], bool(np.max(arith) < RULES["arith_residual_over_slope"]),
                    "this, with T1's integrated reference current, is the annulus residual of the subtraction; the energy left under NO or R2 is the excitation, a size (T5a, T5b)")
    # the falloff of the static annuli (probes)
    fits = {}
    rows = []
    for t in O.trs:
        st = ann[t]["static"]
        if st is None:
            continue
        f_far = fit_powerlaw(lo_p[1:], hi_p[1:], st[1:])
        f_all = fit_powerlaw(lo_p, hi_p, st)
        fits[t] = {"far_probe_annuli": f_far, "all_probe_annuli": f_all, "ratios_per_doubling": [float(st[j] / st[j + 1]) if st[j + 1] else NAN for j in range(len(st) - 1)]}
        rows.append([t, fmt(f_far["alpha"], "%.3f"), fmt(f_far["err"], "%.3f"), fmt(f_far["ratio_per_doubling"], "%.2f"), fmt(f_all["alpha"], "%.3f"), fmt(f_all["err"], "%.3f"),
                     ", ".join(fmt(x, "%.3f") for x in fits[t]["ratios_per_doubling"])])
    out["static_fits_probes"] = fits
    md += ["### (b) The falloff of the removed response raw - NO: alpha in E_annulus ~ R^alpha (the channel-limited prediction -2, 4 per doubling, the density r^-5; the logarithm 0); the far fit uses %s" % ", ".join(ann_lab[1:]), ""]
    table(md, ["t_r/r_s", "alpha far", "err", "ratio per doubling", "alpha all 4", "err", "successive ratios"], rows)
    if fits:
        tl = O.trs[-1]
        f = fits[tl]["far_probe_annuli"]
        lo_w, hi_w = RULES["alpha_window"]
        verdict(RES, "%s T3b alpha of the static probe annuli (far, t_r %s r_s)" % (O.name, tl), [f["alpha"], f["err"]], "within [%g, %g]" % (lo_w, hi_w), bool(np.isfinite(f["alpha"]) and lo_w <= f["alpha"] <= hi_w),
                "ratio per doubling %s" % fmt(f["ratio_per_doubling"], "%.2f"))
    # the NO and R2 annuli against the literal threshold (sizes)
    for sub in ("NO", "R2"):
        rows = []
        for t in O.trs:
            arr = ann[t][sub]
            if arr is None:
                continue
            rel = arr / lnr / slope
            for j, lab in enumerate(ann_lab):
                if j >= 1:
                    st = ann[t]["static"]
                    verdict(RES, "%s T5a %s energy per e-fold over the full-zone slope, t_r %s r_s, %s" % (O.name, sub, t, lab), float(rel[j]), "< %g (literal threshold)" % RULES["annulus_threshold_over_slope"],
                            bool(abs(rel[j]) < RULES["annulus_threshold_over_slope"]),
                            "%s / (raw - NO) removed in the same annulus = %s" % (sub, fmt(arr[j] / st[j] if (st is not None and st[j]) else NAN, "%.3e")), kind="size")
            rows.append([t] + [fmt(x, "%.3e") for x in rel])
        if rows:
            md += ["The %s energy per e-fold over the full-zone slope per annulus (the physical state's excess over the %s reference, the switch-on's excitation; for the pocket-free channels NO equals R2):" % (
                sub, "instantaneous" if sub == "NO" else "continued"), ""]
            table(md, ["t_r/r_s"] + ann_lab, rows)
    # per channel static annuli at the longest ramp
    tl = O.trs[-1]
    pr = O.per_channel(tl, "raw")
    pn = O.per_channel(tl, "NO")
    if pr is not None and pn is not None:
        rows = []
        perch = {}
        for kappa in sorted(pr.keys()):
            if kappa > kmax:
                continue
            st = pr[kappa] - pn[kappa]
            perch[kappa] = st.tolist()
            rows.append([kappa] + [fmt(x) for x in st] + [", ".join(fmt(st[j] / st[j + 1] if st[j + 1] else NAN, "%.2f") for j in range(len(st) - 1))] +
                        [", ".join(fmt(x / (16.0 * kappa), "%.3e") for x in st)])
        out["per_channel_static_longest_ramp"] = perch
        md += ["### (c) Per channel (weights included): the static annuli at t_r = %s r_s, their successive ratios, and the values per unit weight (the channel-limited reading needs them nearly equal at large R)" % tl, ""]
        table(md, ["kappa"] + ann_lab + ["ratios", "per unit weight"], rows)
    # the per-site totals
    tot_out = {}
    for t in O.trs:
        raw_arr, raw_key = O.total_hold(t, "raw")
        no_arr, no_key = O.total_hold(t, "NO")
        r2_arr, r2_key = O.total_hold(t, "R2")
        r2_ok = r2_arr is not None and bool(np.isfinite(r2_arr).any())
        info = {"keys": {"raw": raw_key if raw_arr is not None else None, "NO": no_key if no_arr is not None else None, "R2": r2_key if r2_arr is not None else None},
                "R2_totals_state": "absent" if r2_arr is None else ("stored" if r2_ok else "nan")}
        if raw_arr is None or no_arr is None:
            info["note"] = "per-site totals not found; the totals' keys at this ramp: %s" % (raw_key if raw_arr is None else no_key)
            tot_out[t] = info
            md += ["### (d) The per-site totals at t_r = %s r_s: %s" % (t, info["note"]), ""]
            continue
        n = min(len(raw_arr), len(no_arr))
        st = raw_arr[:n] - no_arr[:n]
        info["n_sites"] = n
        d_best, d_table = 0, {}
        if ann[t]["static"] is not None:
            d_best, d_table = detect_probe_offset(st, O.rs, O.probes, ann[t]["static"])
        info["probe_offset_detected"] = int(d_best)
        info["running_vs_totals_by_offset"] = {str(d): v for d, v in d_table.items()}
        info["running_vs_totals_static_max_abs"] = float(d_table.get(d_best, NAN))
        info["running_vs_totals_static_max_abs_offset0"] = float(d_table.get(0, NAN))
        b_st = binned(st, O.rs, BIN_EDGES, d_best)
        b_no = binned(no_arr[:n], O.rs, BIN_EDGES, d_best)
        b_raw = binned(raw_arr[:n], O.rs, BIN_EDGES, d_best)
        b_r2 = binned(r2_arr[:n], O.rs, BIN_EDGES, d_best) if r2_ok else None
        b_st0 = binned(st, O.rs, BIN_EDGES, 0) if d_best != 0 else b_st
        info["fits"], info["fits_offset0"] = {}, {}
        for lab, rlo, rhi in FIT_RANGES:
            for key, bins_ in (("fits", b_st), ("fits_offset0", b_st0)):
                sel = [b for b in bins_ if b["lo"] >= rlo - 1e-9 and b["hi"] <= rhi + 1e-9]
                info[key][lab] = fit_powerlaw([b["lo"] * O.rs for b in sel], [b["hi"] * O.rs for b in sel], [b["E"] for b in sel]) if len(sel) >= 2 else \
                    {"alpha": NAN, "err": NAN, "n": len(sel), "ratio_per_doubling": NAN, "n_dropped": 0}

        def below_from(bins_):
            rel = [b["E"] / b["ln_ratio"] / slope for b in bins_]
            r_from = None
            for j in range(len(rel) - 1, -1, -1):
                if abs(rel[j]) < RULES["annulus_threshold_over_slope"]:
                    r_from = bins_[j]["lo"]
                else:
                    break
            return rel, r_from

        no_rel, no_from = below_from(b_no)
        if b_r2:
            r2_rel, r2_from = below_from(b_r2)
        else:
            r2_rel, r2_from = [None] * len(b_no), None
        info["NO_per_efold_over_slope_by_bin"] = no_rel
        info["NO_below_threshold_from_R_over_rs"] = no_from
        info["R2_per_efold_over_slope_by_bin"] = r2_rel if b_r2 else None
        info["R2_below_threshold_from_R_over_rs"] = r2_from
        ext = {}
        for d in sorted({0, int(d_best)}):
            i_ext = max(0, int(round(O.rs)) - 1 + d)
            ext[str(d)] = {"static": float(np.nansum(st[i_ext:])), "NO": float(np.nansum(no_arr[:n][i_ext:])), "raw": float(np.nansum(raw_arr[:n][i_ext:])),
                           "R2": float(np.nansum(r2_arr[:n][i_ext:])) if r2_ok else None}
        info["exterior_R_ge_rs_by_offset"] = ext
        info["bins"] = [{"lo": b["lo"], "hi": b["hi"], "n_sites": b["n_sites"], "static": b["E"], "NO": bn["E"], "raw": br["E"], "R2": (b_r2[j]["E"] if b_r2 else None),
                         "static_per_efold_over_slope": b["E"] / b["ln_ratio"] / slope, "NO_per_efold_over_slope": no_rel[j], "R2_per_efold_over_slope": (r2_rel[j] if b_r2 else None),
                         "NO_minus_R2": ((bn["E"] - b_r2[j]["E"]) if b_r2 else None)} for j, (b, bn, br) in enumerate(zip(b_st, b_no, b_raw))]
        tot_out[t] = info
        md += ["### (d) The per-site totals at t_r = %s r_s (keys %s; %d sites): the engine's probe convention detected as offset %+d site(s) relative to 'R = n + 1, R < p r_s' (the running annuli agree with the totals' sums to %s under it; at offset 0 to %s; by offset %s); the bins below use the detected convention" % (
            t, info["keys"], n, d_best, fmt(info["running_vs_totals_static_max_abs"], "%.2e"), fmt(info["running_vs_totals_static_max_abs_offset0"], "%.2e"),
            ", ".join("%+d: %s" % (d, fmt(v, "%.1e")) for d, v in sorted(d_table.items()))), ""]
        rows = [[fmt(b["lo"], "%g"), fmt(b["hi"], "%g"), b["n_sites"], fmt(b["static"]), fmt(b["static_per_efold_over_slope"], "%.3e"), fmt(b["NO"]), fmt(b["NO_per_efold_over_slope"], "%.3e"), fmt(b["raw"]),
                 fmt(b["R2"]) if b["R2"] is not None else info["R2_totals_state"], fmt(b["R2_per_efold_over_slope"], "%.3e") if b["R2"] is not None else "-",
                 fmt(b["NO_minus_R2"]) if b["R2"] is not None else "-"] for b in info["bins"]]
        table(md, ["R_lo/r_s", "R_hi/r_s", "sites", "raw - NO", "(raw - NO) per e-fold / slope", "NO", "NO per e-fold / slope", "raw", "R2", "R2 per e-fold / slope", "NO - R2"], rows)
        rows = [[lab, f["n"], fmt(f["alpha"], "%.3f"), fmt(f["err"], "%.3f"), fmt(f["ratio_per_doubling"], "%.2f"), f.get("n_dropped", 0),
                 fmt(info["fits_offset0"][lab]["alpha"], "%.3f"), fmt(abs(info["fits_offset0"][lab]["alpha"] - f["alpha"]), "%.3f")] for lab, f in info["fits"].items()]
        table(md, ["fit range", "bins", "alpha (detected convention)", "err", "ratio per doubling", "bins dropped (E <= 0)", "alpha (offset 0)", "|difference|"], rows)
        e_d = ext[str(int(d_best))]
        md += ["- exterior (R >= r_s, detected convention) sums of the totals: raw - NO %s, NO %s, raw %s, R2 %s; the NO energy per e-fold stays below %g of the slope from R/r_s = %s outward and the R2 energy from %s (none if '-')" % (
            fmt(e_d["static"]), fmt(e_d["NO"]), fmt(e_d["raw"]), fmt(e_d["R2"]) if e_d["R2"] is not None else info["R2_totals_state"], RULES["annulus_threshold_over_slope"], fmt(no_from, "%g"), fmt(r2_from, "%g") if b_r2 else "not stored"), ""]
        verdict(RES, "%s T6c running annuli against the totals' sums under the detected probe convention (t_r %s r_s, max abs)" % (O.name, t), info["running_vs_totals_static_max_abs"],
                "< %g" % RULES["running_vs_totals_abs"], (info["running_vs_totals_static_max_abs"] < RULES["running_vs_totals_abs"]) if np.isfinite(info["running_vs_totals_static_max_abs"]) else None,
                "detected offset %+d site(s) relative to 'R = n + 1, R < p r_s'; at offset 0 the discrepancy is %s; a pass means the totals are the running sums' channel set" % (d_best, fmt(info["running_vs_totals_static_max_abs_offset0"], "%.2e")))
        if t == O.trs[-1]:
            f = info["fits"]["far [4, 32] r_s"]
            f0 = info["fits_offset0"]["far [4, 32] r_s"]
            lo_w, hi_w = RULES["alpha_window"]
            syst_range = nmax([abs(info["fits"][l]["alpha"] - f["alpha"]) for l in ("mid [2, 32] r_s", "all [2, 60] r_s") if np.isfinite(info["fits"][l]["alpha"])])
            syst_conv = abs(f0["alpha"] - f["alpha"]) if (np.isfinite(f0["alpha"]) and np.isfinite(f["alpha"])) else 0.0
            syst = nmax([syst_range, syst_conv])
            comb = math.sqrt((f["err"] if np.isfinite(f["err"]) else 0.0) ** 2 + (syst if np.isfinite(syst) else 0.0) ** 2)
            verdict(RES, "%s T3b alpha of the static response, binned totals far [4, 32] r_s, t_r %s r_s" % (O.name, t), [f["alpha"], f["err"], syst], "within [%g, %g]" % (lo_w, hi_w),
                    bool(np.isfinite(f["alpha"]) and lo_w <= f["alpha"] <= hi_w), "the systematic is the larger of the change over the ranges [2, 32] and [2, 60] r_s (%s) and the change of the edge convention (%s)" % (fmt(syst_range, "%.3f"), fmt(syst_conv, "%.3f")))
            verdict(RES, "%s T3b the logarithm (alpha 0) excluded: alpha + %g combined errors < 0" % (O.name, RULES["alpha_log_sigmas"]), f["alpha"] + RULES["alpha_log_sigmas"] * comb, "< 0",
                    bool(np.isfinite(f["alpha"]) and f["alpha"] + RULES["alpha_log_sigmas"] * comb < 0))
            verdict(RES, "%s T5b NO energy per e-fold below %g of the slope in every bin from R/r_s" % (O.name, RULES["annulus_threshold_over_slope"]), no_from, "a radius exists within the chain (literal threshold)",
                    no_from is not None, "bins to 60 r_s at t_r %s r_s" % t, kind="size")
            if b_r2:
                verdict(RES, "%s T5b R2 energy per e-fold below %g of the slope in every bin from R/r_s" % (O.name, RULES["annulus_threshold_over_slope"]), r2_from, "a radius exists within the chain (literal threshold)",
                        r2_from is not None, "bins to 60 r_s at t_r %s r_s" % t, kind="size")
    out["totals"] = tot_out
    RES[O.name] = out
    return out


def two_size(Oa, Ob, a, RES, md):
    """the static clock response at the smaller and the larger r_s at equal R/r_s and t_r/r_s: the ratio larger/smaller against the
    scale-covariant small/large (an energy at fixed R/r_s scales as 1/r_s); and at equal lattice sites: the ratio against r_s_big/r_s_small
    (the response quadratic in the flow amplitude, v^2 = r_s/R at fixed R)."""
    small, big = (Oa, Ob) if Oa.rs < Ob.rs else (Ob, Oa)
    kmax = min(a.kmax, small.kdone or a.kmax, big.kdone or a.kmax)
    common = [t for t in small.trs if t in big.trs]
    labs = ["[%g, %g] r_s" % (lo, hi) for lo, hi in zip(small.probes[:-1], small.probes[1:])]
    labs_big = ["[%g, %g] r_s" % (lo, hi) for lo, hi in zip(big.probes[:-1], big.probes[1:])]
    edges_small = [(int(round(lo * small.rs)), int(round(hi * small.rs))) for lo, hi in zip(small.probes[:-1], small.probes[1:])]
    edges_big = [(int(round(lo * big.rs)), int(round(hi * big.rs))) for lo, hi in zip(big.probes[:-1], big.probes[1:])]
    pairs = [(i, j) for i, e in enumerate(edges_small) for j, f in enumerate(edges_big) if e == f]
    out = {"small_rs": small.rs, "big_rs": big.rs, "kmax": kmax, "scale_covariant": small.rs / big.rs, "rs_ratio": big.rs / small.rs, "ratios": {}, "rs_linearity": {},
           "lattice_annulus_pairs": [{"sites": list(edges_small[i]), "small": labs[i], "big": labs_big[j]} for i, j in pairs]}
    rows, rows_lin = [], []
    for t in common:
        s1, _ = small.annuli(t, kmax, "raw")
        n1, _ = small.annuli(t, kmax, "NO")
        s2, _ = big.annuli(t, kmax, "raw")
        n2, _ = big.annuli(t, kmax, "NO")
        if any(x is None for x in (s1, n1, s2, n2)):
            continue
        st1, st2 = s1 - n1, s2 - n2
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = st2 / st1
        out["ratios"][t] = ratio.tolist()
        rows.append([t] + [fmt(x, "%.4f") for x in ratio] + [fmt(out["scale_covariant"], "%.3f")])
        lin = [float(st2[j] / st1[i]) if st1[i] else NAN for i, j in pairs]
        out["rs_linearity"][t] = lin
        rows_lin.append([t] + [fmt(x, "%.5f") for x in lin] + [fmt(out["rs_ratio"], "%.3f")])
    md += ["## Two sizes: the static clock response at r_s %g over r_s %g at equal R/r_s and t_r/r_s (kappa <= %d) against the scale-covariant %.3f" % (big.rs, small.rs, kmax, out["scale_covariant"]), ""]
    table(md, ["t_r/r_s"] + labs + ["scale-covariant"], rows)
    if pairs:
        md += ["The same at equal lattice sites (the response quadratic in the flow amplitude gives the ratio r_s %g / r_s %g = %.3f exactly): %s" % (
            big.rs, small.rs, out["rs_ratio"], "; ".join("sites [%d, %d] = r_s %g's %s = r_s %g's %s" % (p["sites"][0], p["sites"][1], small.rs, p["small"], big.rs, p["big"]) for p in out["lattice_annulus_pairs"])), ""]
        table(md, ["t_r/r_s"] + ["sites [%d, %d]" % tuple(p["sites"]) for p in out["lattice_annulus_pairs"]] + ["r_s ratio"], rows_lin)
    if out["ratios"]:
        tl = common[-1]
        r = out["ratios"][tl][-1]
        verdict(RES, "T3c two-size ratio of the static response at %s, t_r %s r_s" % (labs[-1], tl), r, "within %g of %.3f" % (RULES["two_size_tol"], out["scale_covariant"]),
                bool(np.isfinite(r) and abs(r - out["scale_covariant"]) < RULES["two_size_tol"]))
        for q, (i, j) in enumerate(pairs):
            x = out["rs_linearity"][tl][q]
            verdict(RES, "T3f linearity in r_s at the lattice annulus [%d, %d] sites (r_s %g %s over r_s %g %s), t_r %s r_s" % (edges_small[i][0], edges_small[i][1], big.rs, labs_big[j], small.rs, labs[i], tl), x,
                    "within %g percent of %.3f" % (100 * RULES["rs_linearity_tol"], out["rs_ratio"]), bool(np.isfinite(x) and abs(x / out["rs_ratio"] - 1.0) < RULES["rs_linearity_tol"]),
                    "the deviation from the ratio is the quartic term's share in the flow amplitude at the larger size: %s" % fmt(x / out["rs_ratio"] - 1.0, "%.1e"))
    RES["two_size"] = out
    return out


# ------------------------------------------------------------------------------------------------------------------- the static run (gated)
def static_run(a):
    """the smallest run supplying the far-zone clock, flow and generator annuli (spec section 5): the instantaneous ground state of
    h(s) = h_0 + s V per channel at a few s, no time evolution; the clock density Tr[(R0(s) - flat) hcal_n], the flow density
    s Tr[R0(s) (1/2){P_n, V}] and the generator density (their sum, equal to Tr[R0(s) (1/2){P_n, h(s)}] - Tr[flat (1/2){P_n, h_0}]), summed
    over the channels with the weights 16 kappa, binned in the annuli, fitted, and the ratios flow/clock and gen/clock per bin. The chain
    is built exactly as engine_vacflow_v10 builds it (engine_vacflow_v7.chain_sparse, flow_term, profile; the interface inferred from
    v10's calls and not verified). Starts only with --lead-approved."""
    if not a.lead_approved:
        print("the static run is not approved: pass --lead-approved only after the lead's approval (spec section 5)")
        return 2
    if not a.out:
        print("--out is required")
        return 2
    sys.path.insert(0, a.engine_dir)
    import engine_vacflow_v7 as V7  # noqa: E402
    V7.CC["full"] = (a.chain_current == "full")
    rs = a.rs
    Rfull = np.arange(int(64 * rs), dtype=float) + 1.0
    n = len(Rfull)
    svals = [float(x) for x in a.svals.split(",")]
    acc = {s: {"clock": np.zeros(n), "flow": np.zeros(n), "gen": np.zeros(n)} for s in svals}
    flat_flow = {}
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    log = []
    for kappa in range(1, a.kmax + 1):
        n0 = int(np.searchsorted(Rfull, kappa / a.Ecut))
        Rk = Rfull[n0:]
        h0, Q = V7.chain_sparse(Rk, a.m, kappa)
        V = V7.flow_term(Q, V7.profile(Rk, rs, a.profile))
        h0 = h0.tocsr()
        V = V.tocsr()

        def sea(s):
            e, U = np.linalg.eigh((h0 + s * V).toarray())
            return U[:, e < 0]

        def dens(A, U):
            AU = A @ U
            return np.real(np.einsum("ik,ik->i", U.conj(), AU)).reshape(-1, 2).sum(axis=1)

        Uf = sea(0.0)
        d_h0_flat = dens(h0, Uf)
        flat_flow[kappa] = float(dens(V, Uf).sum())
        w = 16.0 * kappa
        for s in svals:
            U = sea(s)
            clock = dens(h0, U) - d_h0_flat
            flow = s * dens(V, U)
            acc[s]["clock"][n0:] += w * clock
            acc[s]["flow"][n0:] += w * flow
            acc[s]["gen"][n0:] += w * (clock + flow)
        msg = "kappa %d done (%.0f s; n0 %d; Tr[flat V] %.3e)" % (kappa, time.time() - t0, n0, flat_flow[kappa])
        print(msg, flush=True)
        log.append(msg)
    out = {"args": vars(a), "svals": svals, "flat_flow_expectation_per_channel": flat_flow, "seconds": time.time() - t0, "per_s": {}}
    md = ["# S12 static response (the run of spec section 5): r_s %g, m %g, profile %s, current %s, kappa <= %d, E_cut %g" % (rs, a.m, a.profile, a.chain_current, a.kmax, a.Ecut), ""]
    slope = slope_full_zone(a.BL, rs)
    for s in svals:
        bins = {k: binned(acc[s][k], rs, BIN_EDGES) for k in ("clock", "flow", "gen")}
        rows = []
        per = []
        for j in range(len(bins["clock"])):
            bc, bf, bg = bins["clock"][j], bins["flow"][j], bins["gen"][j]
            row = {"lo": bc["lo"], "hi": bc["hi"], "clock": bc["E"], "flow": bf["E"], "gen": bg["E"], "flow_over_clock": bf["E"] / bc["E"] if bc["E"] else NAN,
                   "gen_over_clock": bg["E"] / bc["E"] if bc["E"] else NAN, "boundary_term_gen_minus_half_flow_over_abs_flow": (bg["E"] - bf["E"] / 2) / abs(bf["E"]) if bf["E"] else NAN,
                   "clock_per_efold_over_slope": bc["E"] / bc["ln_ratio"] / slope}
            per.append(row)
            rows.append([fmt(row["lo"], "%g"), fmt(row["hi"], "%g"), fmt(row["clock"]), fmt(row["flow"]), fmt(row["gen"]), fmt(row["flow_over_clock"], "%.4f"), fmt(row["gen_over_clock"], "%.4f"),
                         fmt(row["boundary_term_gen_minus_half_flow_over_abs_flow"], "%.3e"), fmt(row["clock_per_efold_over_slope"], "%.3e")])
        fits = {}
        for lab, rlo, rhi in FIT_RANGES:
            fits[lab] = {}
            for k in ("clock", "flow", "gen"):
                sel = [b for b in bins[k] if b["lo"] >= rlo - 1e-9 and b["hi"] <= rhi + 1e-9]
                fits[lab][k] = fit_powerlaw([b["lo"] * rs for b in sel], [b["hi"] * rs for b in sel], [abs(b["E"]) for b in sel]) if len(sel) >= 2 else {"alpha": NAN, "err": NAN}
        tot = {k: float(acc[s][k].sum()) for k in ("clock", "flow", "gen")}
        out["per_s"][str(s)] = {"bins": per, "fits": fits, "whole_chain": tot, "whole_chain_gen_over_flow": tot["gen"] / tot["flow"] if tot["flow"] else NAN,
                                "density": {k: acc[s][k].tolist() for k in ("clock", "flow", "gen")}}
        md += ["## s = %g: whole chain clock %s, flow %s, generator %s (gen/flow %s; the virial 1/2 in the quadratic regime)" % (s, fmt(tot["clock"]), fmt(tot["flow"]), fmt(tot["gen"]), fmt(out["per_s"][str(s)]["whole_chain_gen_over_flow"], "%.4f")), ""]
        table(md, ["R_lo/r_s", "R_hi/r_s", "clock", "flow", "gen", "flow/clock (-2)", "gen/clock (-1)", "(gen - flow/2)/|flow|", "clock per e-fold / slope"], rows)
        rows = [[lab, k, fmt(fits[lab][k]["alpha"], "%.3f"), fmt(fits[lab][k].get("err", NAN), "%.3f")] for lab in fits for k in ("clock", "flow", "gen")]
        table(md, ["fit range", "part", "alpha (|E| ~ R^alpha)", "err"], rows)
    if len(svals) >= 2:
        s1, s2 = sorted(svals)[:2]
        g1, g2 = out["per_s"][str(s1)]["whole_chain"]["gen"], out["per_s"][str(s2)]["whole_chain"]["gen"]
        f1, f2 = out["per_s"][str(s1)]["whole_chain"]["flow"], out["per_s"][str(s2)]["whole_chain"]["flow"]
        hf = {"gen_difference": g2 - g1, "trapezoid_of_Tr_R0_V": 0.5 * (f1 / s1 + f2 / s2) * (s2 - s1)}
        hf["relative_closure"] = (hf["gen_difference"] - hf["trapezoid_of_Tr_R0_V"]) / abs(hf["gen_difference"]) if hf["gen_difference"] else NAN
        out["hellmann_feynman_s%g_s%g" % (s1, s2)] = hf
        md += ["- whole-chain Hellmann-Feynman between s = %g and %g: gen difference %s against the trapezoid of Tr[R0 V] %s (relative closure %s; exact for a linear Tr[R0 V])" % (
            s1, s2, fmt(hf["gen_difference"]), fmt(hf["trapezoid_of_Tr_R0_V"]), fmt(hf["relative_closure"], "%.2e")), ""]
    out["log"] = log
    json.dump(clean(out), open(os.path.join(a.out, "s12_static_response.json"), "w"), indent=1)
    open(os.path.join(a.out, "s12_static_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ---------------------------------------------------------------------------------------------------------------------------- selftest
def synthetic_vf(rs=32.0, tr=32.0, nramp=64, nhold=4):
    """a vf10-like record with planted quadratic responses: Tr[R0 V_c] = -B s - B4 s^3, an own term, a physical excitation ~ s^4."""
    t_r = tr * rs
    every = t_r / nramp
    keys = ["%g" % rs, "%g" % (1.5 * rs), "%g" % (2 * rs)]
    sites = {keys[0]: int(rs) - 1, keys[1]: int(1.5 * rs) - 1, keys[2]: int(2 * rs) - 1}
    plant = {keys[0]: dict(B=50.0, B4=20.0, own=1.5, E0=-500.0, exc=0.02, excf=-0.01),
             keys[1]: dict(B=56.0, B4=22.0, own=0.8, E0=-700.0, exc=0.025, excf=-0.012),
             keys[2]: dict(B=60.0, B4=24.0, own=0.3, E0=-900.0, exc=0.03, excf=-0.015)}
    chans = []
    for kappa in (1, 2):
        f = 1.0 + 0.1 * kappa
        reads = []
        times = [k * every for k in range(nramp + 1)] + [t_r + (j + 1) * 2 * rs / nhold for j in range(nhold)]
        for t in times:
            ramp = t <= t_r + 1e-9
            s = math.sin(math.pi * t / (2 * t_r)) ** 2 if ramp else 1.0
            sdot = (math.pi / (2 * t_r)) * math.sin(math.pi * t / t_r) if ramp else 0.0
            cuts = {}
            for ck, p in plant.items():
                B, B4, own, E0, exc, excf = (f * p[k] for k in ("B", "B4", "own", "E0", "exc", "excf"))
                TrR0V = -B * s - B4 * s ** 3
                gen_ref = -B * s ** 2 / 2 - B4 * s ** 4 / 4 + own * s ** 2
                gen_NO = exc * s ** 4
                TrDV = excf * s ** 3
                EKR0 = E0 + gen_ref
                EKC = EKR0 + gen_NO
                WR0 = sdot * TrR0V
                W = sdot * TrDV
                IC = -1e-3 * s ** 2 * f
                cuts[ck] = {"site": sites[ck], "E_K_C": EKC, "E_K_R0": EKR0, "W": W, "W_C": WR0 + W, "W_R0": WR0, "I_C": IC, "I_NO": IC + 1e-17, "I_R2": IC - 2e-17,
                            "IS_C": 0.99 * IC, "IS_NO": 0.99 * IC + 3e-8, "IS_R2": 0.99 * IC + 3e-8}
            p = plant[keys[0]]
            B, B4, own, exc, excf = (f * p[k] for k in ("B", "B4", "own", "exc", "excf"))
            clock_ref0 = (-B * s ** 2 / 2 - B4 * s ** 4 / 4 + own * s ** 2) - s * (-B * s - B4 * s ** 3)
            E_in_NO = exc * s ** 4 - excf * s ** 4
            reads.append({"s": s, "sdot": sdot, "E_in_NO": E_in_NO, "E_out_NO": 0.1 * s ** 4, "E_in_raw": E_in_NO + clock_ref0, "E_out_raw": 0.1 * s ** 4 + 0.3 * s ** 2,
                          "E_in_R2": E_in_NO, "E_out_R2": 0.1 * s ** 4, "cuts": cuts, "t_over_rs": t / rs, "N_p": 0.0, "N_h": 0.0, "nneg": 100, "nsea": 100,
                          "phase": "ramp" if ramp else "hold", "continuation": {"n_pocket": 0}})
        chans.append({"kappa": kappa, "n0": 0, "nsea": 100, "halvings": 0, "reads": reads, "rejected": None, "seconds": 0.0, "bound": 1.0,
                      "continuation": {"pockets_max": 0}, "weight": 16.0 * kappa, "fluences": {}})
    return {"script": "synthetic", "args": {"rs": rs, "tr": tr, "hold": 2.0, "chain_current": "synthetic", "smooth_width": 6.0, "smooth_window": 48}, "bands": [],
            "kappa_done": 2, "channels": chans, "smooth_tail": 0.5 * float(_erfc(24 / 6.0)), "totals": {}}


def synthetic_old(rs=32.0, kmax=3, inclusive=False):
    """a vf3-like record: a static clock response ~ 0.3 r_s kappa / R^3 with a staggered part, an NO pulse at 5 r_s plus a far tail; the
    probe sums below p r_s in the convention R < p r_s (inclusive False) or R <= p r_s (inclusive True; the engine's)."""
    n = int(64 * rs)
    R = np.arange(n) + 1.0
    alt = (-1.0) ** np.arange(n)
    running, totals = {}, {}
    for t in ("2", "8", "32"):
        tf = float(t)
        cum_raw = np.zeros(len(PROBES))
        cum_no = np.zeros(len(PROBES))
        run = []
        tot_raw = np.zeros(n)
        tot_no = np.zeros(n)
        for kappa in range(1, kmax + 1):
            w = 16.0 * kappa
            stat = np.where(R >= 1.5 * rs, 0.3 * rs * kappa / R ** 3 * (1 + 0.3 * alt), -0.5 / kappa)
            no = (1e-2 / tf) * np.exp(-((R - 5 * rs) / (0.5 * rs)) ** 2) + 1e-7 * rs / R
            tot_raw += w * (stat + no)
            tot_no += w * no
            if inclusive:
                cum_raw += np.array([np.sum((w * (stat + no))[R <= p * rs]) for p in PROBES])
                cum_no += np.array([np.sum((w * no)[R <= p * rs]) for p in PROBES])
            else:
                cum_raw += np.array([np.sum((w * (stat + no))[R < p * rs]) for p in PROBES])
                cum_no += np.array([np.sum((w * no)[R < p * rs]) for p in PROBES])
            run.append({"kappa": kappa, "E_raw_at": cum_raw.tolist(), "E_NO_at": cum_no.tolist()})
        running[t] = run
        totals[t] = {"raw_hold": tot_raw.tolist(), "NO_hold": tot_no.tolist(), "R2_hold": [NAN] * n}
    return {"script": "synthetic", "args": {"rs": rs}, "kappa_done": kmax, "klist": list(range(1, kmax + 1)), "pockets": {}, "running": running, "per_channel": {}, "summary": {},
            "totals": totals, "seconds": 0.0}


def selftest():
    fails = []
    count = [0]

    def check(name, ok, detail=""):
        count[0] += 1
        print("%s %s %s" % ("ok  " if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    # 1 power laws on doubling bins and on the unequal bins of BIN_EDGES (the width factor)
    for alpha in (-2.0, 0.0, -1.0, -3.0):
        for edges in (np.array([2.0, 4.0, 8.0, 16.0, 32.0]) * 32, np.array(BIN_EDGES[4:]) * 32):
            lo, hi = edges[:-1], edges[1:]
            E = np.log(hi / lo) if alpha == 0.0 else (hi ** alpha - lo ** alpha) / alpha
            f = fit_powerlaw(lo, hi, np.abs(E))
            check("power law alpha %g on %d bins" % (alpha, len(lo)), abs(f["alpha"] - alpha) < 1e-6 and abs(f["ratio_per_doubling"] - 2 ** (-alpha)) < 1e-5,
                  "fit %.8f ratio %.5f" % (f["alpha"], f["ratio_per_doubling"]))
    # 2 the s^2 coefficient
    s_grid = np.array([math.sin(math.pi * k / 128) ** 2 for k in range(0, 65)])
    y = 3.0 * s_grid ** 2 - 7.0 * s_grid ** 4 + 2.0 * s_grid ** 6
    f = small_s(s_grid, y, 0.04)
    check("small-s c2", abs(f["c2"] - 3.0) < 1e-3, "c2 %.6f c4 %.4f n %d err %.1e" % (f["c2"], f["c4"], f["n"], f["err"]))
    # 3 the intercept
    a_, b_ = intercept(s_grid, 0.5 + 2.0 * s_grid)
    check("intercept", abs(a_ - 0.5) < 1e-9 and abs(b_ - 2.0) < 1e-6, "a %.9f b %.6f" % (a_, b_))
    # 4 the synthetic vf record through the analysis
    args = argparse.Namespace(kmax=16, BL=B_L_DEFAULT, smax=0.04)
    V = VF(synthetic_vf(), "synthetic_vf")
    RES, md = {"verdicts": []}, []
    out = analyze_vf(V, args, RES, md)
    wsum = 16.0 * 1.1 + 32.0 * 1.2
    reg = out["decomposition"]["inside 1 r_s"]
    for k, v in (("flow_ref", -50.0 * wsum), ("gen_ref", (-25.0 + 1.5) * wsum), ("clock_ref", (25.0 + 1.5) * wsum)):
        got = reg["small_s"][k]["c2"]
        check("synthetic vf c2 %s" % k, abs(got - v) < 1e-3 * abs(v), "got %.6f expected %.6f" % (got, v))
    check("synthetic vf own2", abs(reg["derived"]["own2"] - 1.5 * wsum) < 1e-3 * 1.5 * wsum, "%.6f" % reg["derived"]["own2"])
    pred = reg["derived"]["ratio_predicted_from_own_share"]
    check("synthetic vf ratios predicted from the own share", abs(pred[0] - reg["derived"]["flow_over_clock"]) < 1e-6 and abs(pred[1] - reg["derived"]["gen_over_clock"]) < 1e-6,
          "%.6f %.6f against %.6f %.6f" % (pred[0], pred[1], reg["derived"]["flow_over_clock"], reg["derived"]["gen_over_clock"]))
    ann = out["decomposition"]["annulus [1, 1.5] r_s"]
    check("synthetic vf annulus B", abs(ann["derived"]["B"] - 6.0 * wsum) < 1e-3 * 6.0 * wsum, "%.6f" % ann["derived"]["B"])
    check("synthetic vf identities", out["identity"]["rel"] < 1e-12, "%.2e" % out["identity"]["rel"])
    r0 = out["residual"][V.cuts[0]]
    check("synthetic vf residual sharp (absolute)", r0["sharp_R2"]["max_abs"] < 1e-13, "%.2e" % r0["sharp_R2"]["max_abs"])
    check("synthetic vf residual smooth (tails)", 1.0 < r0["smooth_R2_over_tail"] < 100.0, "%.2f tails" % r0["smooth_R2_over_tail"])
    exp_flow1 = -(50.0 + 20.0) * wsum
    check("synthetic vf flow at s = 1 from W/sdot at the ramp's last readout", abs(reg["end_of_ramp"]["flow_ref"] - exp_flow1) < 1e-6 * abs(exp_flow1),
          "got %.6f expected %.6f (sdot %.2e)" % (reg["end_of_ramp"]["flow_ref"], exp_flow1, out["readouts"]["sdot_end"]))
    check("synthetic vf verdicts T1/T2 pass", all(v["verdict"] == "pass" for v in RES["verdicts"] if " T1 " in v["test"] or " T2 " in v["test"]))
    check("synthetic vf annuli reported, not judged", all(v["verdict"] == "reported" for v in RES["verdicts"] if " T4 the annulus" in v["test"]))
    # 5 the synthetic old record (the probe convention 'R < p r_s' planted: offset 0 must be detected)
    O = OLD(synthetic_old(32.0), "synthetic_old")
    RES2, md2 = {"verdicts": []}, []
    o = analyze_old(O, args, RES2, md2)
    t32 = o["totals"]["32"]
    far = t32["fits"]["far [4, 32] r_s"]
    check("synthetic old far alpha", abs(far["alpha"] + 2.0) < 0.03, "%.4f err %.4f" % (far["alpha"], far["err"]))
    check("synthetic old ramp spread", max(o["static_ramp_spread_rel"]) < 1e-10, "%.2e" % max(o["static_ramp_spread_rel"]))
    check("synthetic old arithmetic residual", max(o["static_arith_residual_per_efold_over_slope"]) < 1e-10, "%.2e" % max(o["static_arith_residual_per_efold_over_slope"]))
    check("synthetic old probe convention detected (offset 0) and running vs totals", t32["probe_offset_detected"] == 0 and t32["running_vs_totals_static_max_abs"] < 1e-9,
          "offset %d, %.2e" % (t32["probe_offset_detected"], t32["running_vs_totals_static_max_abs"]))
    check("synthetic old probe alpha", abs(o["static_fits_probes"]["32"]["far_probe_annuli"]["alpha"] + 2.0) < 0.03, "%.4f" % o["static_fits_probes"]["32"]["far_probe_annuli"]["alpha"])
    check("synthetic old T5a are sizes", all(v["kind"] == "size" for v in RES2["verdicts"] if " T5a " in v["test"]) and any(" T5a " in v["test"] for v in RES2["verdicts"]))
    # 5b the inclusive convention 'R <= p r_s' planted: offset +1 must be detected, and the version-1 reading (offset 0) must show the one-site mismatch
    O1 = OLD(synthetic_old(32.0, inclusive=True), "synthetic_old_inclusive")
    RES2b, md2b = {"verdicts": []}, []
    o1 = analyze_old(O1, args, RES2b, md2b)
    t1 = o1["totals"]["32"]
    check("synthetic old inclusive convention detected (offset +1); offset 0 shows the mismatch",
          t1["probe_offset_detected"] == 1 and t1["running_vs_totals_static_max_abs"] < 1e-9 and t1["running_vs_totals_static_max_abs_offset0"] > 1e-6,
          "offset %d, %.2e under it, %.2e at offset 0" % (t1["probe_offset_detected"], t1["running_vs_totals_static_max_abs"], t1["running_vs_totals_static_max_abs_offset0"]))
    check("synthetic old inclusive far alpha (bins in the detected convention)", abs(t1["fits"]["far [4, 32] r_s"]["alpha"] + 2.0) < 0.03, "%.4f" % t1["fits"]["far [4, 32] r_s"]["alpha"])
    # 6 two sizes
    O16 = OLD(synthetic_old(16.0), "synthetic_old16")
    RES3, md3 = {"verdicts": []}, []
    ts = two_size(O16, O, args, RES3, md3)
    r = ts["ratios"]["32"]
    check("synthetic two-size ratio at the far annulus", abs(r[-1] - 0.5) < 0.02, "%s" % ["%.4f" % x for x in r])
    lin = ts["rs_linearity"]["32"]
    check("synthetic linearity in r_s at fixed lattice annuli (2 pairs, ratio 2)", len(lin) == 2 and all(abs(x - 2.0) < 1e-8 for x in lin), "%s" % ["%.10f" % x for x in lin])
    # 7 the compare path
    RES4, md4 = {"verdicts": []}, []
    cv = compare_vf(V, VF(synthetic_vf(), "synthetic_vf_b"), args, RES4, md4)
    check("synthetic compare", cv["worst_rel"] == 0.0 or cv["worst_rel"] < 1e-15, "%s" % cv["worst_rel"])
    print("SELFTEST %s (%d checks%s)" % ("PASS" if not fails else "FAIL", count[0], "" if not fails else "; failed: " + ", ".join(fails)))
    return 0 if not fails else 1


# -------------------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vf10")
    ap.add_argument("--vf9")
    ap.add_argument("--vf3-32", dest="vf3_32")
    ap.add_argument("--vf3-16", dest="vf3_16")
    ap.add_argument("--vf7")
    ap.add_argument("--out")
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--BL", type=float, default=B_L_DEFAULT, help="a^2 B_L at m a 0 (review 1.25); the slope (9 pi/2) B_L r_s is formed per record")
    ap.add_argument("--smax", type=float, default=0.04, help="the s range of the quadratic fits (0 < s <= smax)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-keys", dest="dump_keys")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--static-run", dest="static_run", action="store_true")
    ap.add_argument("--lead-approved", dest="lead_approved", action="store_true")
    ap.add_argument("--engine-dir", dest="engine_dir", default="<internal>/code")
    ap.add_argument("--rs", type=float, default=32.0)
    ap.add_argument("--Ecut", type=float, default=6.0)
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--profile", default="exact")
    ap.add_argument("--chain-current", dest="chain_current", default="full")
    ap.add_argument("--svals", default="0.02,0.04,0.5,1.0")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.dump_keys:
        print("\n".join(describe(load(a.dump_keys), a.depth, os.path.basename(a.dump_keys))))
        return
    if a.static_run:
        sys.exit(static_run(a))
    if not a.out:
        ap.error("--out is required")
    os.makedirs(a.out, exist_ok=True)
    RES = {"script": "s12_postprocess.py", "version": VERSION, "inputs": {k: getattr(a, k) for k in ("vf10", "vf9", "vf3_32", "vf3_16", "vf7")}, "args": vars(a), "rules": RULES, "verdicts": []}
    md = ["# S12 J-VF post-processing tables (s12_postprocess.py version %d; the records read in place; a^2 B_L = %g, the full-zone flat slope (9 pi/2) B_L r_s per record)" % (VERSION, a.BL), ""]
    vfs = {}
    for name, path in (("vf10", a.vf10), ("vf9", a.vf9)):
        if path:
            vfs[name] = VF(load(path), name, path)
            analyze_vf(vfs[name], a, RES, md)
    if "vf10" in vfs and "vf9" in vfs:
        compare_vf(vfs["vf9"], vfs["vf10"], a, RES, md)
        r10, r9 = RES["vf10"]["residual"], RES["vf9"]["residual"]
        rows = []
        for ck in vfs["vf10"].cuts:
            if ck in r9:
                num = r10[ck]["smooth_R2"]["max_abs"] / r9[ck]["smooth_R2"]["max_abs"] if r9[ck]["smooth_R2"]["max_abs"] else NAN
                rows.append([fmt(float(ck) / vfs["vf10"].rs, "%g"), fmt(r10[ck]["smooth_R2"]["max_abs"]), fmt(r9[ck]["smooth_R2"]["max_abs"]), fmt(num, "%.1f"),
                             fmt(RES["vf10"]["smooth_tail"] / RES["vf9"]["smooth_tail"], "%.1f"), fmt(r10[ck]["sharp_R2"]["max_abs"]), fmt(r9[ck]["sharp_R2"]["max_abs"])])
        RES["smoothed_residual_scaling"] = rows
        md += ["## The smoothed residual scales with the window tail between the records (the demonstration that it is the window truncation): vf10 against vf9 per cut, the tails' ratio beside it; the sharp residuals are the same rounding in both", ""]
        table(md, ["cut R/r_s", "vf10 smooth R2 max", "vf9 smooth R2 max", "ratio", "tails' ratio", "vf10 sharp R2 max", "vf9 sharp R2 max"], rows)
    olds = {}
    for name, path in (("vf3_rs32", a.vf3_32), ("vf3_rs16", a.vf3_16), ("vf7_rs32", a.vf7)):
        if path:
            olds[name] = OLD(load(path), name, path)
            analyze_old(olds[name], a, RES, md)
    if "vf3_rs16" in olds and "vf3_rs32" in olds:
        two_size(olds["vf3_rs16"], olds["vf3_rs32"], a, RES, md)
    md += ["## Verdicts (the rules of busi/specs/S12_spec_v1.md section 3 as corrected in busi/notes/S12_judgment_v1.md; kind: control = pass/fail, a failure a defect to be judged; size = met / not met against a literal threshold; reported = no rule)", ""]
    table(md, ["test", "kind", "value", "rule", "verdict", "note"], [[v["test"], v["kind"], fmt(v["value"]) if not isinstance(v["value"], list) else ", ".join(fmt(x) for x in v["value"]), v["rule"], v["verdict"], v["note"]] for v in RES["verdicts"]])
    kinds = {}
    for v in RES["verdicts"]:
        kinds.setdefault(v["kind"], {}).setdefault(v["verdict"], 0)
        kinds[v["kind"]][v["verdict"]] += 1
    md += ["Summary: " + "; ".join("%s: %s" % (k, ", ".join("%d %s" % (n_, w) for w, n_ in sorted(d.items()))) for k, d in sorted(kinds.items())), ""]
    md += ["Not in any record (spec section 2.4): the flow and generator energies beyond 2 r_s, the clock/flow/generator split against the continued reference R2, the continued reference's annuli at r_s 32 (vf7's are nan), the massive case; the far-zone slope ratio and falloffs of the flow and generator parts are supplied only by the static run of spec section 5 (not run without the lead's approval).", ""]
    json.dump(clean(RES), open(os.path.join(a.out, "s12_results.json"), "w"), indent=1)
    open(os.path.join(a.out, "s12_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
