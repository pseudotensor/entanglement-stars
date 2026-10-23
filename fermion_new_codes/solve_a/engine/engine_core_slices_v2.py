#!/usr/bin/env python
"""engine_core_slices_v2.py (builder analyst; engine spec 29.5, S8: "the band-kinematic defect uses the builder's estimator, which is an
identity: h_0^2 = m^2 + sum_i sin^2(a k_i) for the sine operator ... so sum_i (1 - cos 2 a k_i) = 2 a^2 (h_0^2 - m^2) as an operator ...; per
dust orbital its expectation is <2 a^2 (h_0^2 - m^2)> with h_0 the record's free channel operator (N = 1, v = 0), eps_n the record's
eigenvalue, the defect -2 H sum_n eps_n <...>_n over the dust orbitals in the core shell. N_core is the particle count inside R_c, Tr[(1 -
P) C (1 - P) Pi_core] summed over channels with the weights 16 kappa, the dust's occupied count and the normal-ordered number Tr[(C - P)
Pi_core] printed beside it ...; R_pack = a (3 N_core/(8 pi))^(1/3) as the goal writes it. engine_core_slices_v2 on the stored slices'
orbitals (CPU)"). Per stored slice (the orbitals file: per channel the propagated sea, the dust columns with their occupations f, the
continued reference R2): C = sea sea^dagger + sum_n f_n dust_n dust_n^dagger and P = R2 R2^dagger per channel; Pi_core the sites with R_n at
most the record's R_c (its json); h_0 the free channel operator (engine_collapse_v19.clocked_host_sparse at N = 1, v = 0) and h_m the record's
(at the slice's applied fields); per channel and dust orbital: b_n = <Pi_core phi_n| 2 (h_0^2 - m^2) |Pi_core phi_n> (a = 1) and eps_n =
<phi_n|h_m|phi_n> (the orbital's energy at the record's applied fields: the builder's reading of "the record's eigenvalue", the dust
orbitals being propagated states, not eigenvectors; labelled); the band sum B = sum_kappa 16 kappa sum_n f_n eps_n b_n; the core's H from the
dense fields re-solved from the record's stored shell totals (engine_core_slices_v1's route): H = -d_r v at the coarse nodes up to R_c, its
core-volume mean H_core (the builder's choice; labelled) and the value at the median node; the defect D = -2 H_core B, beside it D/V_core
(V_core = 4 pi R_c^3/3) and core_slices_v1's rho_dot - 3 H (rho + p) at the median node of the same record (from its json); N_core, the dust's
count inside R_c (sum f_n <phi_n|Pi_core|phi_n>), the normal-ordered number Tr[(C - P) Pi_core], R_pack and the packing ratio ell/R_pack.
Every number printed by the script. CPU. Args: --slices (glob) --json (the run's record jsons, comma list) --data (a COLLAPSE_records data
npz with the record's shell totals) --fm-segments --fm-default --core-v1-json --engine --md --out-json."""
import argparse
import glob
import importlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_v2 as RW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True); ap.add_argument("--json", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14"); ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--core-v1-json", default=None); ap.add_argument("--engine", default="engine_collapse_v19")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True); ap.add_argument("--hfrac", type=float, default=32.0)
    a = ap.parse_args()
    E = importlib.import_module(a.engine)
    E.C5.backend(None)
    args = None; recs = {}
    for f in a.json.split(","):
        J = json.load(open(f))
        args = args or J["args"]
        for r in J["records"]:
            if r.get("E") is not None:
                recs[round(float(r["T"]), 6)] = r
    m = float(args["m"]); nsites = int(args["nsites"]); ell = float(args["ell"]); rs = float(args["rs"])
    Z = np.load(a.data, allow_pickle=True)
    Rd = np.asarray(Z["R"], float); Td = np.asarray(Z["T"], float); g = float(Z["g"])
    seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(Td), int)
    fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";"))
    v1 = {}
    if a.core_v1_json and os.path.exists(a.core_v1_json):
        for r in json.load(open(a.core_v1_json))["rows"]:
            if not r.get("failed"):
                v1[round(float(r["T_over_rs"]), 4)] = r
    rows = []; t0 = time.time()
    for path in sorted(glob.glob(a.slices)):
        S = np.load(path, allow_pickle=True)
        T = float(S["T"]); key = round(T, 6)
        rec = recs.get(key)
        if rec is None or rec.get("R_c") is None:
            rows.append({"slice": os.path.basename(path), "T_over_rs": T / rs, "skipped": "no record with R_c at this time"}); continue
        Rc = float(rec["R_c"])
        R = np.asarray(S["R"], float); N_app = np.asarray(S["N_applied"], float); v_app = np.asarray(S["v_applied"], float)
        core = np.repeat(R <= Rc, 2)
        kap = [int(k) for k in S["kappas"]]; wts = [float(w) for w in S["weights"]]
        B = 0.0; Ncore = 0.0; Ndust = 0.0; Nno = 0.0; per = []
        for k, w in zip(kap, wts):
            ch = E.Chan(k, m, nsites, bool(args.get("half_offset", 0)))
            h0 = E.clocked_host_sparse(ch, np.ones(nsites), np.zeros(nsites)).tocsr()
            hm = E.clocked_host_sparse(ch, N_app, v_app).tocsr()
            sea = np.asarray(S["sea_%d" % k]); R2 = np.asarray(S["R2_%d" % k]); dust = np.asarray(S["dust_%d" % k]); f = np.asarray(S["f_%d" % k], float)
            if dust.shape[1]:
                Pd = dust * core[:, None]
                Xd = 2.0 * (h0 @ (h0 @ Pd) - m * m * Pd)
                b = np.real(np.sum(np.conj(Pd) * Xd, axis=0))
                ep = np.real(np.sum(np.conj(dust) * (hm @ dust), axis=0))
                B += w * float(np.sum(f * ep * b))
                Ndust += w * float(np.sum(f * np.sum(np.abs(dust[core]) ** 2, axis=0)))
            Ys = sea - R2 @ (R2.conj().T @ sea)
            nc = float(np.sum(np.abs(Ys[core]) ** 2))
            if dust.shape[1]:
                Yd = dust - R2 @ (R2.conj().T @ dust)
                nc += float(np.sum(f * np.sum(np.abs(Yd[core]) ** 2, axis=0)))
            Ncore += w * nc
            nno = float(np.sum(np.abs(sea[core]) ** 2)) - float(np.sum(np.abs(R2[core]) ** 2))
            if dust.shape[1]:
                nno += float(np.sum(f * np.sum(np.abs(dust[core]) ** 2, axis=0)))
            Nno += w * nno
            per.append({"kappa": k, "N_core_part": w * nc})
            del sea, R2, dust, Ys
        kd = int(np.argmin(np.abs(Td - T)))
        name = fmap.get(seg_names[seg[kd]], a.fm_default) if seg_names else a.fm_default
        FM = importlib.import_module(name)
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][kd], float), np.asarray(Z["J_tot"][kd], float), g, Rd, ell, collar_policy="flag", support_rel=1e-14)
        Hc = Hmed = None
        if not info.get("failed"):
            nodes = np.arange(ell, Rc + 1e-9, ell)
            if len(nodes):
                _, v1d, _ = RW.deriv(info["v_dense"], nodes, ell / a.hfrac)
                H = -v1d
                wv = nodes ** 2
                Hc = float(np.sum(H * wv) / np.sum(wv)); Hmed = float(H[len(nodes) // 2])
        Vc = 4.0 * math.pi * Rc ** 3 / 3.0
        D = (-2.0 * Hc * B) if Hc is not None else None
        Rpack = (3.0 * Ncore / (8.0 * math.pi)) ** (1.0 / 3.0) if Ncore > 0 else None
        vr = v1.get(round(T / rs, 4))
        med = vr.get("median") if vr else None
        rows.append({"slice": os.path.basename(path), "T_over_rs": T / rs, "R_c": Rc, "N_core": Ncore, "N_dust_core": Ndust, "N_normal_ordered_core": Nno, "R_pack": Rpack,
                     "packing_ratio": (ell / Rpack) if Rpack else None, "B": B, "H_core_mean": Hc, "H_median": Hmed, "defect": D, "defect_per_volume": (D / Vc) if D is not None else None,
                     "v1_median_rho_dot_minus_rhs": (med[1] - med[2]) if med else None, "v1_median_r": med[0] if med else None, "field_module": name, "per_channel": per})
        print("%s: T/rs %.3f, N_core %.4f, R_pack %s, defect %s (%.0f s)" % (os.path.basename(path), T / rs, Ncore, ("%.4f" % Rpack) if Rpack else "-", ("%+.4e" % D) if D is not None else "-", time.time() - t0), flush=True)
    L = ["## The core slices, version 2 (engine_core_slices_v2.py; engine spec 29.5, S8): the band-kinematic defect and the packing ratio on the stored orbitals", "",
         "- N_core = sum_kappa 16 kappa Tr[(1 - P) C (1 - P) Pi_core] (C the slice's sea and dust, P its continued reference R2, Pi_core the sites up to the record's R_c); beside it the dust's occupied count inside R_c and the normal-ordered number Tr[(C - P) Pi_core]; R_pack = (3 N_core/(8 pi))^(1/3) (a = 1).",
         "- the band sum B = sum_kappa 16 kappa sum_n f_n eps_n <Pi_core phi_n| 2 (h_0^2 - m^2) |Pi_core phi_n> over the dust orbitals, h_0 the free channel operator; eps_n = <phi_n|h_m|phi_n> at the record's applied fields (the builder's reading of \"the record's eigenvalue\"; labelled); the defect D = -2 H_core B with H_core the core-volume mean of H = -d_r v (the builder's choice; labelled), D/V_core beside core_slices_v1's rho_dot - 3 H (rho + p) at the median node of the same record.", "",
         "| T/rs | R_c | N_core | dust count inside R_c | normal-ordered number inside R_c | R_pack | ell/R_pack | B | H_core (mean) | H (median node) | D = -2 H_core B | D/V_core | v1: rho_dot - 3H(rho+p) at the median node (r) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            L.append("| %.3f | %s | - | - | - | - | - | - | - | - | - | - | - |" % (r["T_over_rs"], r["skipped"])); continue
        fx = lambda x, f_: (f_ % x) if x is not None else "-"
        L.append("| %.3f | %.2f | %.4f | %.4f | %+.4f | %s | %s | %+.4e | %s | %s | %s | %s | %s |" % (
            r["T_over_rs"], r["R_c"], r["N_core"], r["N_dust_core"], r["N_normal_ordered_core"], fx(r["R_pack"], "%.4f"), fx(r["packing_ratio"], "%.4f"), r["B"],
            fx(r["H_core_mean"], "%+.4e"), fx(r["H_median"], "%+.4e"), fx(r["defect"], "%+.4e"), fx(r["defect_per_volume"], "%+.4e"),
            ("%+.4e (%.0f)" % (r["v1_median_rho_dot_minus_rhs"], r["v1_median_r"])) if r["v1_median_rho_dot_minus_rhs"] is not None else "-"))
    L += ["", "- every number above is printed by the script from the stored orbitals, the run's records and core_slices_v1's json; the reading is the lead's."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_core_slices_v2.py", "slices": a.slices, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[-len(rows) - 2:]))


if __name__ == "__main__":
    main()
