#!/usr/bin/env python
"""flat_profile_v1.py (busi, GATECOMP flat build; ReviewA_gatecomp_v1 sec. 2 (ii), ReviewA_flatq_v1 sec. 1). DATA script, box CPU.

The flat preparation's T = 0 profile at a size where the engine's preparation is hours on a CPU (r_s/a 256, 4096 sites), by the child
engine's own functions (engine_collapse_v36nq_rd13mfb.py, imported, not run): flat_configure, flat_begin (the box's levels and orbitals per
channel), fill_lowest to N_B = r_s/(2 g m), the dust's eps and J by the engine's dens, the fields by the engine's solve (site fields as
--site-fields), E_0 = FLD.charge, the head's flatbox entry by flat_head. ONE difference from the engine's preparation, stated: the
kept columns are NOT projected on the positive band of h_m[N, v] (that projection needs one 2n = 8192 eigendecomposition per dust channel
per pass); without it eps and J do not depend on the fields and one field solve is the fixed point. The projection's size is measured at
r_s/a 32 by the engine itself (the fixture's "discarded" and E_0 against this script's at the same size: flat_fixture_read).

Outputs: OUT.json (the catalog over --K-keys channels: the top channel, K = top + 5, N_B, E_0, the flatbox head, per sigma the T = 0 field
gate terms on the dense ivp fields re-solved at ell = sigma: sigma |H|, sigma ||K||_op and the five static terms of the six-term
definition, max over 0 <= r <= 2 R_b and where; the sixth term's D k needs a second record and is not read here) and OUT_T0_dust.npz in
the engine's slice format for the keys the dust comparator reads (T, rs, R, kappas, weights, N_applied, v_applied, dtype, dust_K, f_K).
    flat_profile_v1.py --code DIR --rs 256 --nsites 4096 --K-keys 24 --edge hard [--W 1.0] --prep-ell 4 --sigmas 4,8,16 --out PREFIX
"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--engine", default="engine_collapse_v36nq_rd13mfb")
    ap.add_argument("--rs", type=float, required=True); ap.add_argument("--nsites", type=int, required=True)
    ap.add_argument("--K-keys", dest="K_keys", type=int, default=24, help="channels whose box levels enter fill_lowest (above the catalog's top)")
    ap.add_argument("--K-sea", dest="K_sea", type=int, default=None, help="the sea set's top kept in the npz (default: top + 5)")
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--g", type=float, default=0.25); ap.add_argument("--R0-over-rs", dest="R0_over_rs", type=float, default=4.0)
    ap.add_argument("--edge", default="hard", choices=["hard", "taper"]); ap.add_argument("--W", type=float, default=None)
    ap.add_argument("--prep-ell", dest="prep_ell", type=float, default=4.0); ap.add_argument("--site-fields", dest="site_fields", default="averaged")
    ap.add_argument("--sigmas", default="4,8,16"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in (a.out + ".json", a.out + "_T0_dust.npz"):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    t0 = time.time()
    sys.path.insert(0, a.code)
    E = importlib.import_module(a.engine)
    E.C5.backend(None)
    E.OPT.update({"collar_policy": "flag", "support_rel": 1e-14, "site_fields": a.site_fields})
    E.SOLVE_TOL["rtol"] = 1e-12; E.SOLVE_TOL["atol"] = 1e-16
    ns_ = argparse.Namespace(prep="flatbox", prep_R=None, prep_edge=a.edge, prep_taper=a.W, dust_set="well", massless_ball=False, shell=None,
                             m=a.m, rs=a.rs, R0_over_rs=a.R0_over_rs)
    E.flat_configure(ns_, argparse.ArgumentParser())
    NB = a.rs / (2.0 * a.g * a.m)
    chs = [E.Chan(k, a.m, a.nsites, False) for k in range(1, a.K_keys + 1)]
    R = chs[0].R
    E.flat_begin(chs, R, a.m, NB)
    levels = []
    for ic, ch in enumerate(chs):
        levels += [(float(x), ch.w, ic, j) for j, x in enumerate(E.FLAT["keys"][ch.kappa][:400])]
    occ = E.fill_lowest(levels, NB)
    top = max(chs[ic].kappa for (ic, j), f in occ.items() if f > 0)
    Ksea = a.K_sea or (top + 5)
    print("box catalog at r_s/a %g, %d sites, N_B %.3f: top channel %d (of %d keyed), K = top + 5 = %d; box %d sites (%.1f s)" % (
        a.rs, a.nsites, NB, top, a.K_keys, top + 5, E.FLAT["nc"], time.time() - t0), flush=True)
    if top >= a.K_keys:
        sys.exit("the catalog reaches the keyed set's top: raise --K-keys")
    nsites = len(R); eps = np.zeros(nsites); J = np.zeros(nsites); dust = {}; NBs = 0.0; kin = 0.0
    for ic, ch in enumerate(chs[:Ksea]):
        keep = [j for j in range(len(E.FLAT["keys"][ch.kappa])) if occ.get((ic, j), 0.0) > 0]
        if not keep:
            continue
        fk = np.array([occ[(ic, j)] for j in keep])
        Phi = E.FLAT["vecs"][ch.kappa][:, keep]
        eps += ch.w * E.dens(ch.h, Phi, fk); J += ch.w * E.dens(ch.Q, Phi, fk); NBs += ch.w * fk.sum()
        hD = (Phi.conj() * (ch.h @ Phi)).sum(dim=0).real.cpu().numpy(); kin += ch.w * float(fk @ (hD - a.m))
        dust[ch.kappa] = (Phi, fk)
    N, v, W, info = E.solve(eps, J, a.g, R, a.prep_ell, np.ones(nsites))
    if info.get("failed"):
        sys.exit("field solve failed: %s" % info["failed"])
    E0 = float(E.FLD.charge(info, a.g))
    head = {"N_B": NBs, "f_kin": kin / (NBs * a.m), "dust_momentum_total": float(J.sum())}
    E.flat_head(head, chs[:Ksea], dust, a.m, R, N, v, eps, J)
    E.flat_print(head)
    print("E0 %.6f (2gE0 %.3f a = %.4f r_s); N(centre) %.5f; min(N - |v|) %.5f; dust momentum %.3e (%.1f s)" % (
        E0, 2 * a.g * E0, 2 * a.g * E0 / a.rs, float(N[0]), float(np.min(N - np.abs(v))), float(J.sum()), time.time() - t0), flush=True)
    # T = 0 field gate terms at each sigma (ell = sigma), on the dense ivp fields
    here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.dirname(here)]   # gate_profiles_v1.py (busi/code/gatecomp, or beside)
    GP = importlib.import_module("gate_profiles_v1")
    gates = {}
    Rb = E.FLAT["R_b"]
    for s in [float(x) for x in a.sigmas.split(",") if x]:
        _, _, _, inf_s = E.FLD.solve_constraints_ivp(eps, J, a.g, R, s, collar_policy="flag", support_rel=1e-14)
        Nf, vf = GP.sym(inf_s["N_dense"], +1), GP.sym(inf_s["v_dense"], -1)
        r = np.arange(0.0, 2.0 * Rb + 1e-9, s / 8.0)
        k0 = GP.kfields(Nf, vf, r, s / 32.0)
        T5 = {"abs(v'/N)": np.abs(k0["kr"]), "abs(v/(N r))": np.abs(k0["kt"]), "abs(N'/N)": np.abs(k0["N1"] / k0["N"]), "sqrt(abs(N''/N))": np.sqrt(np.abs(k0["N2"] / k0["N"]))}
        H = np.abs(k0["v1"]); Kop = np.maximum(np.abs(k0["kr"]), np.abs(k0["kt"])); L5 = np.max(np.array(list(T5.values())), axis=0)
        mx = lambda q: [float(s * np.max(q)), float(r[int(np.argmax(q))])]  # noqa: E731
        nodes = r >= s - 1e-12
        gates[str(int(s))] = {"sigma_H": mx(H), "sigma_Kop": mx(Kop), "sigma_static5_all": mx(L5), "sigma_static5_nodes": [float(s * np.max(L5[nodes])), float(r[nodes][int(np.argmax(L5[nodes]))])],
                              "terms": {t: mx(q) for t, q in T5.items()}, "E0_at_sigma": float(E.FLD.charge(inf_s, a.g)),
                              "sqrt_three_halves_H_estimate_of_Dk": float(s * math.sqrt(1.5) * np.max(H))}
        print("T=0 gate at sigma %g: sigma|H| %.4f @ %.0f; sigma||K|| %.4f @ %.0f; static five-term %.4f @ %.0f (nodes r >= sigma %.4f @ %.0f); E0 at this ell %.6f" % (
            s, gates[str(int(s))]["sigma_H"][0], gates[str(int(s))]["sigma_H"][1], gates[str(int(s))]["sigma_Kop"][0], gates[str(int(s))]["sigma_Kop"][1],
            gates[str(int(s))]["sigma_static5_all"][0], gates[str(int(s))]["sigma_static5_all"][1], gates[str(int(s))]["sigma_static5_nodes"][0],
            gates[str(int(s))]["sigma_static5_nodes"][1], gates[str(int(s))]["E0_at_sigma"]), flush=True)
    kap = np.array([ch.kappa for ch in chs[:Ksea]]); wts = np.array([ch.w for ch in chs[:Ksea]])
    arrs = {"T": np.array(0.0), "rs": np.array(a.rs), "R": np.asarray(R, float), "kappas": kap, "weights": wts, "N_applied": np.asarray(N, float),
            "v_applied": np.asarray(v, float), "dtype": np.array("complex128")}
    for k, (Phi, fk) in dust.items():
        arrs["dust_%d" % k] = Phi.cpu().numpy(); arrs["f_%d" % k] = np.asarray(fk, float)
    np.savez(a.out + "_T0_dust.npz", **arrs)
    out = {"script": "flat_profile_v1.py", "args": vars(a), "engine": E.__file__, "N_B": NB, "N_B_filled": NBs, "top_channel": top, "K_top_plus_5": top + 5,
           "K_sea_written": Ksea, "E0": E0, "two_gE0": 2 * a.g * E0, "two_gE0_over_rs": 2 * a.g * E0 / a.rs, "N_centre": float(N[0]),
           "min_N_minus_absv": float(np.min(N - np.abs(v))), "flatbox": head["flatbox"], "gate_T0": gates, "projection": "not applied (see the docstring)",
           "seconds": time.time() - t0}
    json.dump(out, open(a.out + ".json", "x"), indent=1, default=float)
    print("wrote %s.json and %s_T0_dust.npz (%.0f s)" % (a.out, a.out, time.time() - t0))


if __name__ == "__main__":
    main()
