#!/usr/bin/env python
"""engine_core_slices_v3.py (builder analyst; engine spec 37.7: "engine_core_slices_v3 (new; CPU) on the production's ten slices and the
ladder members' slices adds S7's columns: R_50, R_90 and the edge width of the tagged dust; ell Lambda_geom with the six terms; both trapping
zeros (dense N + v, applied); the local bound 4 sqrt(m^2 + 3/(2a^2))/a^3 against the slice's largest |rho_x|; the predicted band part
-(2/3)(a k_F)^2 per slice beside the third-moment term; the exact cut fluxes F = -2 Im Tr(h_AB D_BA) at R_c and R_t; the header's inference
'ell/R_pack above 1 means the core is unresolved' is struck (R_pack is a minimum radius; REVIEWA_REVIEW 1.9)"; S7 (WEB40F_TASKS/solve.md):
"Lambda_geom = max(|v'/N|, |v/(Nr)|, |N'/N|, sqrt|D k_r|, sqrt|D k_t|, sqrt|N''/N|)"): version 2's computation unchanged (every version 2
column printed again, so a version 3 run on a member reproduces its version 2 table), and per slice:
(1) the tagged dust D = sum_n f_n phi_n phi_n^dagger per channel, weighted 16 kappa (the slice's weights): its number per site summed over
the spin components, R_50 and R_90 the radii holding 50 and 90 percent of it (linear between sites), the edge width the distance between
the outermost radii where the dust's number density n(r) = (number per site)/(4 pi r^2) equals 0.9 and 0.1 of its largest value (the
builder's definition; labelled), ell/R_90 and ell/(edge width) beside;
(2) on the dense fields re-solved from the record's stored shell totals (version 2's route) at the coarse nodes r = ell, 2 ell, ...: k_r =
v'/N, k_t = v/(N r), N'/N, N''/N (engine_rows_v2.deriv, Richardson at ell/32) and D k = (d_T k + v d_r k)/N with d_r k_r = (v'' N - v' N')/N^2,
d_r k_t = v'/(r N) - v/(r^2 N) - v N'/(r N^2) (engine_rows_v2.brackets' forms) and d_T k by the central difference of the re-solved k between
the records before and after the slice's (one-sided at the run's last record; labelled); the six terms' largest values over the nodes up to
R_c and over the nodes up to the outer dense root R_t, Lambda_geom their maximum, ell Lambda_geom against 0.1, the dominant term named;
(3) both trapping zeros: the dense N + v (the field module's trapping_radius on the re-solved fields) and the applied N_bar + v_bar (sign
changes of the slice's N_applied + v_applied between sites, linear), beside the record's stored R_t and r_in;
(4) rho_x = Re Tr(Pi_x h_0 S)/(4 pi r_x^2 a) per site, S = C - P (the slice's sea and dust against its continued reference R2), h_0 the free
channel operator, summed over the channels with their weights: its largest |rho_x| and where, against the bound 4 sqrt(m^2 + 3/(2a^2))/a^3
(a = 1), the ratio; a radial-channel reading (the channel operator's centrifugal term kappa/r makes its site vertex differ from the cubic
lattice's, whose trace norm the bound is), not a certificate;
(5) the third-moment term of the tagged dust inside R_c, -(T_3/T_1 - m^2) with T_k = sum_kappa 16 kappa sum_n f_n Re <phi_n|Pi_core h_0^k|phi_n>
(the radial channel's h_0, its centrifugal term included: not the cubic band moment), beside the predicted band parts -(2/3)(a k_F)^2 and
-(3/5)(a k_F)^2 with k_F = (3 pi^2 n_core)^(1/3), n_core the dust count inside R_c over 4 pi R_c^3/3 (two states per momentum), k_F/m printed;
(6) the exact cut fluxes of the tagged dust F = -2 Im Tr(h_AB D_BA) = -2 sum_n f_n Im(phi_A^dagger h_AB phi_B) (A the sites at R <= the cut,
h = h_m the record's operator at the slice's applied fields; outward positive), at R_c and at the dense outer root R_t, summed over channels.
R_pack = (3 N_core/(8 pi))^(1/3) is printed as the minimum radius a core of that count needs (a capacity radius, not the core's radius):
ell/R_pack above 1 does not mean the core is unresolved; the interior resolution reads ell/R_90 and ell/(edge width). Every number printed
by the script. CPU. Args: version 2's (--slices --json --data --fm-segments --fm-default --core-v1-json --engine --md --out-json --hfrac)."""
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


def frac_radius(R, cum, q):
    """the radius at which the cumulative fraction cum (non-decreasing, ending at 1) first reaches q, linear between sites."""
    i = int(np.searchsorted(cum, q))
    if i == 0:
        return float(R[0])
    if i >= len(R):
        return float(R[-1])
    c0, c1 = cum[i - 1], cum[i]
    return float(R[i - 1] + (q - c0) / (c1 - c0) * (R[i] - R[i - 1])) if c1 > c0 else float(R[i])


def outer_crossing(R, n, level):
    """the outermost radius where n falls through level (n >= level inside, < level outside), linear between sites."""
    idx = np.nonzero(n >= level)[0]
    if not len(idx):
        return None
    i = int(idx[-1])
    if i + 1 >= len(R):
        return float(R[i])
    return float(R[i] + (n[i] - level) / (n[i] - n[i + 1]) * (R[i + 1] - R[i])) if n[i] != n[i + 1] else float(R[i])


def kfields(FM, Z, kd, g, Rd, ell, nodes, hh):
    _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][kd], float), np.asarray(Z["J_tot"][kd], float), g, Rd, ell, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        return None, info
    N, N1, N2 = RW.deriv(info["N_dense"], nodes, hh)
    v, v1, v2 = RW.deriv(info["v_dense"], nodes, hh)
    kr = v1 / N; kt = v / (nodes * N)
    kr1 = (v2 * N - v1 * N1) / N ** 2
    kt1 = v1 / (nodes * N) - v / (nodes * nodes * N) - v * N1 / (nodes * N * N)
    return {"N": N, "N1": N1, "N2": N2, "v": v, "v1": v1, "kr": kr, "kt": kt, "kr1": kr1, "kt1": kt1}, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True); ap.add_argument("--json", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14"); ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--core-v1-json", default=None); ap.add_argument("--engine", default="engine_collapse_v19")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True); ap.add_argument("--hfrac", type=float, default=32.0)
    a = ap.parse_args()
    for p in (a.md, a.out_json):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
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
    bound = 4.0 * math.sqrt(m * m + 1.5)
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
        kd = int(np.argmin(np.abs(Td - T)))
        name = fmap.get(seg_names[seg[kd]], a.fm_default) if seg_names else a.fm_default
        FM = importlib.import_module(name)
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][kd], float), np.asarray(Z["J_tot"][kd], float), g, Rd, ell, collar_policy="flag", support_rel=1e-14)
        tz = FM.trapping_radius(None, None, Rd, info) if not info.get("failed") else {"zeros": [], "min": None, "r_min": None}
        Rt_dense = tz["zeros"][-1] if tz["zeros"] else None
        kap = [int(k) for k in S["kappas"]]; wts = [float(w) for w in S["weights"]]
        B = 0.0; Ncore = 0.0; Ndust = 0.0; Nno = 0.0; per = []
        nd_site = np.zeros(len(R)); e_site = np.zeros(len(R)); T1 = 0.0; T3 = 0.0; F_Rc = 0.0; F_Rt = 0.0 if Rt_dense is not None else None
        A_c = np.repeat(R <= Rc, 2); A_t = np.repeat(R <= Rt_dense, 2) if Rt_dense is not None else None
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
            del Ys
            # version 3: (1) the tagged dust per site; (4) rho_x of S = C - P with h_0; (5) the third moment; (6) the cut fluxes
            if dust.shape[1]:
                nd_site += w * (np.abs(dust[0::2]) ** 2 + np.abs(dust[1::2]) ** 2) @ f
            es = np.real(np.conj(sea) * (h0 @ sea)).sum(axis=1)
            er = np.real(np.conj(R2) * (h0 @ R2)).sum(axis=1)
            ed = (np.real(np.conj(dust) * (h0 @ dust)) @ f) if dust.shape[1] else np.zeros(2 * len(R))
            ex = es + ed - er
            e_site += w * (ex[0::2] + ex[1::2])
            if dust.shape[1]:
                h1 = h0 @ dust; h3 = h0 @ (h0 @ h1)
                T1 += w * float(np.sum(f * np.real(np.sum(np.conj(dust[core]) * h1[core], axis=0))))
                T3 += w * float(np.sum(f * np.real(np.sum(np.conj(dust[core]) * h3[core], axis=0))))
                for Acut, tag in ((A_c, "c"), (A_t, "t")):
                    if Acut is None:
                        continue
                    yB = hm @ (dust * (~Acut)[:, None])
                    fl = -2.0 * float(np.sum(f * np.imag(np.sum(np.conj(dust[Acut]) * yB[Acut], axis=0))))
                    if tag == "c":
                        F_Rc += w * fl
                    else:
                        F_Rt += w * fl
            del sea, R2, dust
        Hc = Hmed = None
        lam = None
        if not info.get("failed"):
            nodes = np.arange(ell, Rc + 1e-9, ell)
            if len(nodes):
                _, v1d, _ = RW.deriv(info["v_dense"], nodes, ell / a.hfrac)
                H = -v1d
                wv = nodes ** 2
                Hc = float(np.sum(H * wv) / np.sum(wv)); Hmed = float(H[len(nodes) // 2])
            # (2) the six terms on the nodes up to R_t (or R_c)
            rmax_ = max(Rc, Rt_dense if Rt_dense is not None else 0.0)
            nod = np.arange(ell, rmax_ + 1e-9, ell)
            hh = ell / a.hfrac
            k0, _ = kfields(FM, Z, kd, g, Rd, ell, nod, hh)
            kdn = kd + 1 if kd + 1 < len(Td) else None; kdp = kd - 1 if kd - 1 >= 0 else None
            def fmod(kx):
                return importlib.import_module(fmap.get(seg_names[seg[kx]], a.fm_default) if seg_names else a.fm_default)
            kp = kfields(fmod(kdp), Z, kdp, g, Rd, ell, nod, hh)[0] if kdp is not None else None
            kn = kfields(fmod(kdn), Z, kdn, g, Rd, ell, nod, hh)[0] if kdn is not None else None
            if kp is not None and kn is not None:
                dT_kr = (kn["kr"] - kp["kr"]) / (Td[kdn] - Td[kdp]); dT_kt = (kn["kt"] - kp["kt"]) / (Td[kdn] - Td[kdp]); dmode = "central"
            elif kp is not None:
                dT_kr = (k0["kr"] - kp["kr"]) / (Td[kd] - Td[kdp]); dT_kt = (k0["kt"] - kp["kt"]) / (Td[kd] - Td[kdp]); dmode = "one-sided (backward)"
            else:
                dT_kr = (kn["kr"] - k0["kr"]) / (Td[kdn] - Td[kd]); dT_kt = (kn["kt"] - k0["kt"]) / (Td[kdn] - Td[kd]); dmode = "one-sided (forward)"
            Dkr = (dT_kr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dT_kt + k0["v"] * k0["kt1"]) / k0["N"]
            terms = {"abs(v'/N)": np.abs(k0["kr"]), "abs(v/(N r))": np.abs(k0["kt"]), "abs(N'/N)": np.abs(k0["N1"] / k0["N"]),
                     "sqrt(abs(D k_r))": np.sqrt(np.abs(Dkr)), "sqrt(abs(D k_t))": np.sqrt(np.abs(Dkt)), "sqrt(abs(N''/N))": np.sqrt(np.abs(k0["N2"] / k0["N"]))}
            lam = {"dT_mode": dmode, "records_used": [float(Td[x]) / rs for x in (kdp, kd, kdn) if x is not None]}
            for zone, rz in (("core", Rc), ("root", Rt_dense)):
                if rz is None:
                    lam[zone] = None; continue
                sel = nod <= rz + 1e-9
                if not np.any(sel):
                    lam[zone] = None; continue
                mx = {t_: float(np.max(v_[sel])) for t_, v_ in terms.items()}
                dom = max(mx, key=mx.get)
                lam[zone] = {"terms": mx, "Lambda_geom": mx[dom], "ell_Lambda_geom": ell * mx[dom], "dominant": dom,
                             "at_r": float(nod[sel][int(np.argmax(terms[dom][sel]))]), "nodes": int(np.sum(sel))}
        Vc = 4.0 * math.pi * Rc ** 3 / 3.0
        D = (-2.0 * Hc * B) if Hc is not None else None
        Rpack = (3.0 * Ncore / (8.0 * math.pi)) ** (1.0 / 3.0) if Ncore > 0 else None
        vr = v1.get(round(T / rs, 4))
        med = vr.get("median") if vr else None
        # (1) R_50, R_90, the edge width
        tot = float(np.sum(nd_site)); cum = np.cumsum(nd_site) / tot if tot > 0 else None
        R50 = frac_radius(R, cum, 0.5) if cum is not None else None; R90 = frac_radius(R, cum, 0.9) if cum is not None else None
        n3 = nd_site / (4.0 * math.pi * R ** 2)
        r09 = outer_crossing(R, n3, 0.9 * float(np.max(n3))) if tot > 0 else None; r01 = outer_crossing(R, n3, 0.1 * float(np.max(n3))) if tot > 0 else None
        edge = (r01 - r09) if (r09 is not None and r01 is not None) else None
        # (3) the applied zeros
        fa = N_app + v_app
        az = [float(R[i] + fa[i] / (fa[i] - fa[i + 1]) * (R[i + 1] - R[i])) for i in range(len(R) - 1) if fa[i] * fa[i + 1] < 0]
        # (4) rho_x
        rho = e_site / (4.0 * math.pi * R ** 2)
        ir = int(np.argmax(np.abs(rho)))
        # (5) the band parts
        ncore = Ndust / Vc
        kF = (3.0 * math.pi ** 2 * ncore) ** (1.0 / 3.0) if ncore > 0 else None
        third = -(T3 / T1 - m * m) if T1 != 0 else None
        rows.append({"slice": os.path.basename(path), "T_over_rs": T / rs, "R_c": Rc, "N_core": Ncore, "N_dust_core": Ndust, "N_normal_ordered_core": Nno, "R_pack": Rpack,
                     "packing_ratio": (ell / Rpack) if Rpack else None, "B": B, "H_core_mean": Hc, "H_median": Hmed, "defect": D, "defect_per_volume": (D / Vc) if D is not None else None,
                     "v1_median_rho_dot_minus_rhs": (med[1] - med[2]) if med else None, "v1_median_r": med[0] if med else None, "field_module": name, "per_channel": per,
                     "v3": {"dust_total": tot, "R_50": R50, "R_90": R90, "edge_r09": r09, "edge_r01": r01, "edge_width": edge,
                            "ell_over_R90": (ell / R90) if R90 else None, "ell_over_edge": (ell / edge) if edge else None,
                            "Lambda_geom": lam, "dense_zeros": tz["zeros"], "dense_min_N_minus_absv": tz.get("min"), "applied_zeros": az,
                            "record_R_t": rec.get("R_t"), "record_r_in": rec.get("r_in"),
                            "rho_x_max_abs": float(abs(rho[ir])), "rho_x_at_R": float(R[ir]), "rho_x_signed": float(rho[ir]), "bound": bound, "rho_over_bound": float(abs(rho[ir])) / bound,
                            "T1": T1, "T3": T3, "third_moment_term": third, "n_core": ncore, "k_F": kF, "kF_over_m": (kF / m) if kF else None,
                            "band_23": (-(2.0 / 3.0) * kF ** 2) if kF else None, "band_35": (-(3.0 / 5.0) * kF ** 2) if kF else None,
                            "F_at_Rc": F_Rc, "F_at_Rt": F_Rt, "R_t_cut": Rt_dense}})
        print("%s: T/rs %.3f, N_core %.4f, R_50 %s, R_90 %s, edge %s, ell Lambda_geom (core) %s, max|rho_x| %.4e (bound %.4f), third moment %s, F(R_c) %+.4e (%.0f s)" % (
            os.path.basename(path), T / rs, Ncore, ("%.3f" % R50) if R50 else "-", ("%.3f" % R90) if R90 else "-", ("%.3f" % edge) if edge else "-",
            ("%.4f" % lam["core"]["ell_Lambda_geom"]) if lam and lam.get("core") else "-", abs(rho[ir]), bound, ("%+.4e" % third) if third is not None else "-", F_Rc, time.time() - t0), flush=True)
    fx = lambda x, f_: (f_ % x) if x is not None else "-"  # noqa: E731
    L = ["## The core slices, version 3 (engine_core_slices_v3.py; engine spec 37.7, S7): version 2's columns and S7's on the stored orbitals", "",
         "- version 2's columns: N_core = sum_kappa 16 kappa Tr[(1 - P) C (1 - P) Pi_core] (Pi_core the sites up to the record's R_c), the dust's count inside R_c, the normal-ordered number Tr[(C - P) Pi_core]; R_pack = (3 N_core/(8 pi))^(1/3) (a = 1) is the minimum radius a core of that count needs, a capacity radius and not the core's radius: ell/R_pack above 1 does not mean the core is unresolved (version 2's packing reading struck; REVIEWA_REVIEW 1.9); the band sum B and D = -2 H_core B as version 2 defines them.",
         "- S7's columns: R_50, R_90 of the tagged dust (sum_n f_n phi_n phi_n^dagger, weighted 16 kappa) and its edge width, the distance between the outermost radii where its number density n(r) = number per site/(4 pi r^2) equals 0.9 and 0.1 of its largest value (the builder's definition); ell/R_90 and ell/(edge width) are the interior-resolution readings.",
         "- ell Lambda_geom: Lambda_geom = max(abs(v'/N), abs(v/(N r)), abs(N'/N), sqrt(abs(D k_r)), sqrt(abs(D k_t)), sqrt(abs(N''/N))) on the dense fields re-solved from the record's stored shell totals at the nodes r = ell, 2 ell, ..., D = (d_T + v d_r)/N with d_T by the central difference of the re-solved k between the adjacent records (one-sided at a run's end, marked); the largest over the nodes up to R_c (core) and up to the dense outer root (root); against 0.1.",
         "- the trapping zeros: the dense N + v (the field module's trapping_radius) and the applied N_bar + v_bar (the slice's site fields), the record's stored R_t and r_in beside.",
         "- rho_x = Re Tr(Pi_x h_0 (C - P))/(4 pi r_x^2) per site (h_0 the free channel operator; summed over the channels with their weights): the largest |rho_x| against 4 sqrt(m^2 + 3/(2 a^2))/a^3 (a = 1; %.6f at m a %g); the channel operator's centrifugal term makes its site vertex differ from the cubic lattice's, so this is a radial-channel reading, not a certificate." % (bound, m),
         "- the third-moment term of the tagged dust inside R_c, -(T_3/T_1 - m^2), T_k = sum_kappa 16 kappa sum_n f_n Re <phi_n|Pi_core h_0^k|phi_n> (the radial channel's h_0, its centrifugal term included; not the cubic band moment), beside the predicted band parts -(2/3)(a k_F)^2 (k_F >> m) and -(3/5)(a k_F)^2 (k_F << m), k_F = (3 pi^2 n_core)^(1/3), n_core the dust count inside R_c over 4 pi R_c^3/3.",
         "- the exact cut fluxes of the tagged dust, F = -2 Im Tr(h_AB D_BA) (A the sites at R up to the cut, h = h_m at the slice's applied fields; outward positive; summed over the channels with their weights), at R_c and at the dense outer root R_t.", "",
         "### Version 2's columns (unchanged)", "",
         "| T/rs | R_c | N_core | dust count inside R_c | normal-ordered number inside R_c | R_pack (a minimum radius) | ell/R_pack | B | H_core (mean) | H (median node) | D = -2 H_core B | D/V_core | v1: rho_dot - 3H(rho+p) at the median node (r) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            L.append("| %.3f | %s | - | - | - | - | - | - | - | - | - | - | - |" % (r["T_over_rs"], r["skipped"])); continue
        L.append("| %.3f | %.2f | %.4f | %.4f | %+.4f | %s | %s | %+.4e | %s | %s | %s | %s | %s |" % (
            r["T_over_rs"], r["R_c"], r["N_core"], r["N_dust_core"], r["N_normal_ordered_core"], fx(r["R_pack"], "%.4f"), fx(r["packing_ratio"], "%.4f"), r["B"],
            fx(r["H_core_mean"], "%+.4e"), fx(r["H_median"], "%+.4e"), fx(r["defect"], "%+.4e"), fx(r["defect_per_volume"], "%+.4e"),
            ("%+.4e (%.0f)" % (r["v1_median_rho_dot_minus_rhs"], r["v1_median_r"])) if r["v1_median_rho_dot_minus_rhs"] is not None else "-"))
    L += ["", "### S7: the tagged dust's radii, the trapping zeros, the local bound", "",
          "| T/rs | R_c | dust total | R_50 | R_90 | edge width (r at 0.9 to r at 0.1) | ell/R_90 | ell/edge | dense zeros of N + v (min N - abs v) | applied zeros of N_bar + v_bar | record R_t, r_in | max abs rho_x (at R) | its sign | the bound | ratio |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            continue
        q = r["v3"]
        L.append("| %.3f | %.2f | %.4f | %s | %s | %s (%s to %s) | %s | %s | %s (%s) | %s | %s, %s | %.4e (%.0f) | %s | %.4f | %.4e |" % (
            r["T_over_rs"], r["R_c"], q["dust_total"], fx(q["R_50"], "%.3f"), fx(q["R_90"], "%.3f"), fx(q["edge_width"], "%.3f"), fx(q["edge_r09"], "%.2f"), fx(q["edge_r01"], "%.2f"),
            fx(q["ell_over_R90"], "%.4f"), fx(q["ell_over_edge"], "%.4f"), ", ".join("%.4f" % z for z in q["dense_zeros"]) or "none", fx(q["dense_min_N_minus_absv"], "%+.4f"),
            ", ".join("%.3f" % z for z in q["applied_zeros"]) or "none", fx(q["record_R_t"], "%.4f"), fx(q["record_r_in"], "%.4f"),
            q["rho_x_max_abs"], q["rho_x_at_R"], "+" if q["rho_x_signed"] >= 0 else "-", q["bound"], q["rho_over_bound"]))
    L += ["", "### S7: ell Lambda_geom (the six terms' largest values; core: nodes up to R_c, root: nodes up to the dense outer root)", "",
          "| T/rs | zone (nodes) | abs(v'/N) | abs(v/(N r)) | abs(N'/N) | sqrt(abs(D k_r)) | sqrt(abs(D k_t)) | sqrt(abs(N''/N)) | Lambda_geom | ell Lambda_geom (against 0.1) | dominant (at r) | d_T |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tnames = ["abs(v'/N)", "abs(v/(N r))", "abs(N'/N)", "sqrt(abs(D k_r))", "sqrt(abs(D k_t))", "sqrt(abs(N''/N))"]
    for r in rows:
        if r.get("skipped") or not r["v3"]["Lambda_geom"]:
            continue
        lg = r["v3"]["Lambda_geom"]
        for zone in ("core", "root"):
            z = lg.get(zone)
            if not z:
                L.append("| %.3f | %s | - | - | - | - | - | - | - | - | - | %s |" % (r["T_over_rs"], zone, lg["dT_mode"])); continue
            L.append("| %.3f | %s (%d) | %s | %.4e | %.4e | %.4f %s | %s (%.0f) | %s |" % (
                r["T_over_rs"], zone, z["nodes"], " | ".join("%.4e" % z["terms"][t_] for t_ in tnames[:5]), z["terms"][tnames[5]], z["Lambda_geom"],
                z["ell_Lambda_geom"], "(passes)" if z["ell_Lambda_geom"] <= 0.1 else "(above)", z["dominant"], z["at_r"], lg["dT_mode"]))
    L += ["", "### S7: the third-moment term against the predicted band parts; the exact cut fluxes of the tagged dust", "",
          "| T/rs | n_core | a k_F | k_F/m | -(2/3)(a k_F)^2 | -(3/5)(a k_F)^2 | third-moment term -(T_3/T_1 - m^2) | T_1 | T_3 | F at R_c | F at the dense R_t (R) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            continue
        q = r["v3"]
        L.append("| %.3f | %.4e | %s | %s | %s | %s | %s | %+.4e | %+.4e | %+.4e | %s |" % (
            r["T_over_rs"], q["n_core"], fx(q["k_F"], "%.4f"), fx(q["kF_over_m"], "%.3f"), fx(q["band_23"], "%+.4e"), fx(q["band_35"], "%+.4e"),
            fx(q["third_moment_term"], "%+.4e"), q["T1"], q["T3"], q["F_at_Rc"], ("%+.4e (%.3f)" % (q["F_at_Rt"], q["R_t_cut"])) if q["F_at_Rt"] is not None else "-"))
    L += ["", "- every number above is printed by the script from the stored orbitals, the run's records, its records data file and core_slices_v1's json; the reading is the lead's."]
    open(a.md, "x").write("\n".join(L) + "\n")
    json.dump({"script": "engine_core_slices_v3.py", "slices": a.slices, "args": vars(a), "rows": rows}, open(a.out_json, "x"), indent=1, default=float)
    print("wrote %s and %s" % (a.md, a.out_json))


if __name__ == "__main__":
    main()
