#!/usr/bin/env python3
"""obs_plateau_v5.py (builder analyst; B5 of SPEC_final4b_observables.md sec. 11.6, PL-ext at r_s = 64 per sec. 11.2: version 5 of
obs_plateau_v4.py; v4's name was taken by the smooth/nodes controls, so the spec's "obs_plateau_v4 = v3 with --fields-source" is this file).

Version 5: the solved slice from the engine's dense site fields (engine sec. 15.1; engine_collapse_v5 writes them as the fields npz's site
arrays N, v, the integrators' dense solution at the sites R = 1 .. n): --fields-source dense (the default) evaluates the slice at the blend's
radii rho(R) by the degree-7 parity spline through the SITE values (spacing 1; N even, v odd through the centre), and --fields-source spline
is version 4's quintic parity spline through the NODE values (spacing ell), the comparator, never used for a coefficient. R_cut is given in
lattice units (--Rcut-sites, default 12 16 20) with w = 4; the label on every row is "solved exterior (quasi-static, tau, ell), control's
interior below R_cut; site fields: dense (engine 15.1)" or "... spline comparator". Per slice the script prints, before the rows: the slice
(T, T/r_s, E, r_s = 2 g E, R_t and R_sh of the record, the record's late-surface flag, the engine's own interpolation floors); the dense
against the spline at the sites inside the collar (max |dN|, max |dv| over [min R_cut, r_collar]); the off-site floor of the dense route
(max |s7 - s5| at the half-integer points of [min R_cut, r_collar], s5 and s7 the quintic and septic splines through the same site values);
the surface's Taylor coefficients d1, d2, d3 of N + v at the slice's own zero (central differences, h = 1/4 site, on the evaluated source)
against the exact profile's 1/(4 g E), -3/(16 g^2 E^2), 15/(64 g^3 E^3) at the solved E; the spectral fractions of v - (its linear trend)
over [min R_cut, R_t + 5 ell] above pi/ell and above pi/2 (Hann window). For the solved slices kappa a is the blended chain's own surface
gravity (one size: c = (1 - theta)/kappa^2 is printed per row, no Richardson value). Every other profile is version 4's unchanged (the
reproduction controls run through this file's driver). Version 4's docstring follows (in obs_plateau_v4.py)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_v4 as P4  # noqa: E402


def site_spline(R, y, parity, k):
    """the degree-k spline through the site values with the parity extension through the centre (v odd: the point (0, 0) added)."""
    R = np.asarray(R, float); y = np.asarray(y, float)
    if parity < 0:
        xx = np.concatenate([-R[::-1], [0.0], R]); yy = np.concatenate([-y[::-1], [0.0], y])
    else:
        xx = np.concatenate([-R[::-1], R]); yy = np.concatenate([y[::-1], y])
    return make_interp_spline(xx, yy, k=k)


def slice_fields_v5(npz_path, rec_path, slice_index, g, source="dense", order=7):
    """the slice's (N, v) as functions of radius. dense: the degree-`order` parity spline through the engine's site values (the dense solution
    at the sites) inside min(collar, the engine's last site - 8), the analytic exterior N = 1, v = -sqrt(2 g E/rho) beyond; spline: version 4's
    node splines (the comparator). Returns fn, info (with the comparison helpers)."""
    Z = np.load(npz_path, allow_pickle=True)
    recs = [r for r in json.load(open(rec_path))["records"] if "E" in r]
    idx = slice_index if slice_index >= 0 else len(Z["T"]) + slice_index
    T = float(Z["T"][idx])
    rec = min(recs, key=lambda r: abs(r["T"] - T))
    E, rcol = float(rec["E"]), float(rec["r_collar"])
    Rs = np.asarray(Z["R"], float); Ns = np.asarray(Z["N"][idx], float); vs = np.asarray(Z["v"][idx], float)
    rn = np.asarray(Z["r_nodes"][idx], float); Nn = np.asarray(Z["N_nodes"][idx], float); vn = np.asarray(Z["v_nodes"][idx], float)
    r_in_max = min(rcol, Rs[-1] - 8.0)
    if source == "dense":
        Nf = site_spline(Rs, Ns, +1, order); vf = site_spline(Rs, vs, -1, order)
    else:
        Nf = P4._parity_spline(rn, Nn, +1); vf = P4._parity_spline(rn, vn, -1)

    def fn(rho):
        rho = np.asarray(rho, float)
        inside = rho < r_in_max
        N_ = np.where(inside, Nf(np.minimum(rho, r_in_max)), 1.0)
        v_ = np.where(inside, vf(np.minimum(rho, r_in_max)), -np.sqrt(2.0 * g * E / np.maximum(rho, 1e-12)))
        return N_, v_
    rs = 2.0 * g * E
    info = {"T": T, "T_over_rs": T / rs, "E": E, "r_s": rs, "r_collar": rcol, "r_switch": r_in_max, "ell": float(Z["ell"]), "tau": float(Z["tau"]),
            "label_engine": str(Z["label"]), "R_t_record": rec.get("R_t"), "R_sh_record": rec.get("R_sh"), "late_surface_record": rec.get("late_surface"),
            "engine_interp_floor_N": rec.get("interp_floor_N"), "engine_interp_floor_v": rec.get("interp_floor_v"), "site_fields_record": rec.get("site_fields"),
            "source": source, "slice_index": idx}
    aux = {"Rs": Rs, "Ns": Ns, "vs": vs, "rn": rn, "Nn": Nn, "vn": vn}
    return fn, info, aux


def slice_diagnostics(fn, info, aux, Rcut_min, g):
    """the per-slice lines of the docstring."""
    Rs, Ns, vs = aux["Rs"], aux["Ns"], aux["vs"]
    E = info["E"]; ell = info["ell"]
    hi = info["r_switch"]
    sel = (Rs >= Rcut_min) & (Rs <= hi)
    Nsp = P4._parity_spline(aux["rn"], aux["Nn"], +1); vsp = P4._parity_spline(aux["rn"], aux["vn"], -1)
    d = {"dense_minus_spline_at_sites_N": float(np.max(np.abs(Ns[sel] - Nsp(Rs[sel])))) if sel.any() else None,
         "dense_minus_spline_at_sites_v": float(np.max(np.abs(vs[sel] - vsp(Rs[sel])))) if sel.any() else None}
    s5N, s7N = site_spline(Rs, Ns, +1, 5), site_spline(Rs, Ns, +1, 7)
    s5v, s7v = site_spline(Rs, vs, -1, 5), site_spline(Rs, vs, -1, 7)
    xm = np.arange(math.floor(Rcut_min), math.floor(hi)) + 0.5
    d["offsite_floor_N"] = float(np.max(np.abs(s7N(xm) - s5N(xm)))) if len(xm) else None
    d["offsite_floor_v"] = float(np.max(np.abs(s7v(xm) - s5v(xm)))) if len(xm) else None
    f = lambda x: float(np.sum(fn(np.array([x]))[0] + fn(np.array([x]))[1]))
    xs = np.arange(max(Rcut_min, 2.0), hi - 1.0, 0.25)
    fs = np.array([f(x) for x in xs])
    zz = np.nonzero(fs[:-1] * fs[1:] <= 0)[0]
    if len(zz):
        i = zz[-1]
        Rt = brentq(f, xs[i], xs[i + 1], xtol=1e-13)
        h = 0.25
        fm2, fm1, f0, fp1, fp2 = (f(Rt + q * h) for q in (-2, -1, 0, 1, 2))
        d1 = (fp1 - fm1) / (2 * h); d2 = (fp1 - 2 * f0 + fm1) / h ** 2; d3 = (fp2 - 2 * fp1 + 2 * fm1 - fm2) / (2 * h ** 3)
        ex = (1.0 / (4 * g * E), -3.0 / (16 * (g * E) ** 2), 15.0 / (64 * (g * E) ** 3))
        d.update({"R_t": Rt, "R_t_over_2gE": Rt / (2 * g * E), "taylor": [d1, d2, d3], "taylor_over_exact": [d1 / ex[0], d2 / ex[1], d3 / ex[2]]})
        top = Rt + 5 * ell
    else:
        d.update({"R_t": None, "taylor": None, "taylor_over_exact": None})
        top = Rcut_min + 64
    w_sel = (Rs >= Rcut_min) & (Rs <= min(top, hi))
    vv = vs[w_sel]
    if len(vv) > 16:
        nn = np.arange(len(vv)); tr_ = np.polyval(np.polyfit(nn, vv, 1), nn); win = np.hanning(len(vv))
        sp = np.abs(np.fft.rfft((vv - tr_) * win, n=8 * len(vv))) ** 2
        kk = 2 * np.pi * np.fft.rfftfreq(8 * len(vv))
        tot = float(sp.sum())
        d["spectral_fraction_above_pi_over_ell"] = float(sp[kk > math.pi / ell].sum() / tot) if tot > 0 else None
        d["spectral_fraction_above_pi_over_2"] = float(sp[kk > math.pi / 2].sum() / tot) if tot > 0 else None
        d["spectral_window"] = [float(Rcut_min), float(min(top, hi))]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["tanh", "pg", "pgs", "exactc", "smooth", "nodes", "solved"], default="solved")
    ap.add_argument("--fields", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--slice", type=int, nargs="+", default=[-1], help="solved: slice indices (default the last)")
    ap.add_argument("--fields-source", choices=["dense", "spline"], nargs="+", default=["dense", "spline"])
    ap.add_argument("--site-order", type=int, default=7)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--Rcut-sites", type=float, nargs="+", default=[12.0, 16.0, 20.0])
    ap.add_argument("--wcut", type=float, nargs="+", default=[4.0])
    ap.add_argument("--R-lead", type=float, nargs="+", default=[32.0, 64.0, 128.0, 256.0], help="units of r_s")
    ap.add_argument("--outer-w", type=float, default=None)
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--extra", default="")
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    pts = [(r, mu) for mu in a.mu for r in a.r if r > mu]
    if a.extra:
        pts += [tuple(float(q) for q in p.split(":")) for p in a.extra.split(",")]
    res, slices = [], []
    for si in a.slice:
        for src in a.fields_source:
            fn, sinfo, aux = slice_fields_v5(a.fields, a.records, si, a.g, src, a.site_order)
            sd = slice_diagnostics(fn, sinfo, aux, min(a.Rcut_sites), a.g)
            sinfo["label"] = "solved exterior (quasi-static, tau = %g, ell = %g), control's interior below R_cut; site fields: %s" % (
                sinfo["tau"], sinfo["ell"], "dense (engine 15.1)" if src == "dense" else "spline comparator (node values)")
            slices.append({"info": sinfo, "diagnostics": sd})
            print("slice %d (%s): T/r_s %.3f E %.6f R_t %s taylor/exact %s dense-spline at sites N %.2e v %.2e off-site floor N %.2e v %.2e spectral >pi/ell %s >pi/2 %s" % (
                sinfo["slice_index"], src, sinfo["T_over_rs"], sinfo["E"], sd.get("R_t"), sd.get("taylor_over_exact"), sd["dense_minus_spline_at_sites_N"] or 0,
                sd["dense_minus_spline_at_sites_v"] or 0, sd["offsite_floor_N"] or 0, sd["offsite_floor_v"] or 0, sd.get("spectral_fraction_above_pi_over_ell"),
                sd.get("spectral_fraction_above_pi_over_2")), flush=True)
            if a.diagnostics_only:
                continue
            E = sinfo["E"]; rs = 2 * a.g * E
            for RL in a.R_lead:
                for rc in a.Rcut_sites:
                    for wv in a.wcut:
                        R_, v_, N_, gi = P4.blended_chain(fn, rs, rc, wv, RL * rs, a.outer_w, 1.0 / (4 * a.g * E))
                        gi.update({"source": "solved-%s" % src, "slice": sinfo, "Rcut_over_rs": rc / rs, "Rcut_sites": rc,
                                   "exterior_check": P4.exterior_check(fn, rc, sinfo["r_collar"], a.g, E)})
                        for (r, mu) in pts:
                            try:
                                rd = P4.run_point(v_, N_, gi["kappa"], r, mu)
                            except RuntimeError as e:
                                rd = {"r": r, "mu": mu, "error": str(e)}
                            rd.update({"kappa_a": gi["kappa"], "geometry": gi})
                            if rd.get("theta") is not None:
                                rd["c_one_size"] = (1.0 - rd["theta"]) / gi["kappa"] ** 2
                            res.append(rd)
                            print(json.dumps({k: rd.get(k) for k in ("r", "mu", "theta", "c_one_size", "Gamma", "unitarity", "error")}),
                                  "slice %d %s R_cut %g w %g R_lead/r_s %g kappa 4gE %.9f" % (sinfo["slice_index"], src, rc, wv, RL, gi["kappa"] * 4 * a.g * E), flush=True)
                            json.dump({"script": "obs_plateau_v5.py", "args": vars(a), "slices": slices, "rows": res}, open(a.out_json, "w"), indent=1, default=float)
    json.dump({"script": "obs_plateau_v5.py", "args": vars(a), "slices": slices, "rows": res}, open(a.out_json, "w"), indent=1, default=float)
    write_md(res, slices, a)


def write_md(res, slices, a):
    L = ["## PL-ext (obs_plateau_v5.py): the surface's coefficient on the solved exterior, the engine's dense site fields against the spline comparator", "",
         "### The slices", "",
         "| slice | source | T/r_s | E | R_t (record; this evaluation) | late surface (record) | engine floors N, v | dense - spline at the sites N, v | off-site floor N, v | d1 d2 d3 / exact | spectral fraction > pi/ell, > pi/2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in slices:
        i, d = s["info"], s["diagnostics"]
        f = lambda x, fmt="%.2e": (fmt % x) if isinstance(x, (int, float)) else "-"
        L.append("| %d | %s | %.3f | %.6f | %s; %s | %s | %s, %s | %s, %s | %s, %s | %s | %s, %s |" % (
            i["slice_index"], i["source"], i["T_over_rs"], i["E"], f(i["R_t_record"], "%.4f"), f(d.get("R_t"), "%.4f"), i["late_surface_record"],
            f(i["engine_interp_floor_N"]), f(i["engine_interp_floor_v"]), f(d["dense_minus_spline_at_sites_N"]), f(d["dense_minus_spline_at_sites_v"]),
            f(d["offsite_floor_N"]), f(d["offsite_floor_v"]), (", ".join("%.6f" % x for x in d["taylor_over_exact"]) if d.get("taylor_over_exact") else "-"),
            f(d.get("spectral_fraction_above_pi_over_ell")), f(d.get("spectral_fraction_above_pi_over_2"))))
    if res:
        L += ["", "### The rows (label per slice above; c = (1 - theta)/kappa^2 with kappa the blended chain's surface gravity)", "",
              "| slice | source | R_cut | w | R_lead/r_s | kappa 4 g E | r | mu | theta | c (one size) | Gamma | |beta|^2 | unitarity | exterior check (max |N - 1|, max |rho v^2 - 2gE|/2gE) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rd in res:
            g = rd["geometry"]
            ec = g.get("exterior_check", {})
            if "error" in rd:
                L.append("| %d | %s | %g | %g | %g | | %g | %g | %s | | | | | |" % (g["slice"]["slice_index"], g["source"], g["Rcut_sites"], g["w"], g["R_lead_nominal"],
                         rd["r"], rd["mu"], rd["error"]))
                continue
            E = g["slice"]["E"]
            L.append("| %d | %s | %g | %g | %g | %.9f | %g | %g | %.9f | %s | %.9f | %.6e | %.1e | %.2e, %.2e |" % (
                g["slice"]["slice_index"], g["source"], g["Rcut_sites"], g["w"], g["R_lead_nominal"], g["kappa"] * 4 * a.g * E, rd["r"], rd["mu"], rd["theta"],
                ("%.5f" % rd["c_one_size"]) if rd.get("c_one_size") is not None else "-", rd["Gamma"], rd["beta2"], rd["unitarity"],
                ec.get("max_abs_N_minus_1", float("nan")), ec.get("max_abs_rv2_minus_2gE_over_2gE", float("nan"))))
        L += ["", "### The spread over R_cut (the error of 11.2 (ii)) per (slice, source, R_lead, r, mu)", "", "| slice | source | R_lead/r_s | r | mu | c at R_cut 12, 16, 20 | spread/mean |", "|---|---|---|---|---|---|---|"]
        keys = sorted({(rd["geometry"]["slice"]["slice_index"], rd["geometry"]["source"], rd["geometry"]["R_lead_nominal"], rd["r"], rd["mu"]) for rd in res if rd.get("c_one_size") is not None})
        for kk in keys:
            cs = [(rd["geometry"]["Rcut_sites"], rd["c_one_size"]) for rd in res if rd.get("c_one_size") is not None and (rd["geometry"]["slice"]["slice_index"], rd["geometry"]["source"], rd["geometry"]["R_lead_nominal"], rd["r"], rd["mu"]) == kk]
            cs.sort()
            vals = np.array([c for _, c in cs])
            sp = float((vals.max() - vals.min()) / abs(vals.mean())) if len(vals) > 1 and vals.mean() != 0 else None
            L.append("| %d | %s | %g | %g | %g | %s | %s |" % (kk[0], kk[1], kk[2], kk[3], kk[4], ", ".join("%g: %.5f" % (r_, c) for r_, c in cs), ("%.4f" % sp) if sp is not None else "-"))
    L += ["", "- every number above is printed by the script; P25's Painleve-Gullstrand law for comparison: c = -(5/8 + r^2 - 3 mu^2/4) (the exact ladder's (kappa a)^2 law);",
          "  the dense route is the measurement, the spline route the comparator (observables 11.2 (i)); B3 (obs_plateau_mix_v3) must pass on the dense slice first (11.2 (iii))."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
