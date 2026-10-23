#!/usr/bin/env python
"""run_bprime_b11_v1.py (busi; B' v2 test B11, ReviewA_bpchk_v1 sec. 2: is the port at 2 L in the near zone of the window's quanta?).
On the paper's chains (pt3_core.ChainOps, m = 0) at kappa = --kappas, size s = L/a, family F2 or F1, the fields the production back-average
of the profile at the chain sites (pt3b_v2lib.engine_avg_radial, std --sigma sites), the same retained block (sites inside the averaged
outer root) and the same window inputs (its eigenvectors in (0, --win T0) and (-T0, 0), deterministic), four exteriors:
  (a) cap2:   the launched port, sin^2 CAP from 2 L to eta_max over 0.5 L, box 2.5 L;
  (b) lead:   a unitary radial lead with the profile's (vacuum) fields to --lead L (default 200 L), no absorber; collected = the weight beyond
              r_c = --rc L (default 50 L) at time t (nothing returns within T = 1e2 L from beyond r_c through the lead's far end);
  (b2) lead2: the same lead doubled (the lead-length control);
  (c) cap50:  the same sin^2 CAP placed at r_c on that lead (box r_c + 0.5 L).
Per exterior and input: the retained weight ||P_I psi(t)||^2 (sampled every --dt L), the collected Gram G(t) (cap: 1 - Psi^dag Psi, the
dilation; lead: Psi^dag P_(r > r_c) Psi), its spectrum p at the bins t = 10 L .. T, N_acc(0.1, 0.01) by the time route, the pole of each
input's return signal (pt3b_v2lib.select_pole), and the nested output entropies of the supply-filled declared covariance on bins of --bin L
(pt3b_v2lib.bins_reader). Comparisons printed: (b) vs (c) (the dilation as the unitary realization; pass 2 percent), (b2) vs (b) (lead
length), (a) vs (c) (the near-zone factor: widths, retained weights, p). Box CPU. Propagation: scipy expm_multiply on the sparse operator."""
import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_v2lib as V2  # noqa: E402


def build(kappa, prof, L, sigma, n_sites, cap_onset_L, cap_width_L=0.5, eta_max=1.0):
    ops = K3.ChainOps(n_sites, kappa)
    Nf = lambda r: prof["N"](np.asarray(r) / L); vf = lambda r: prof["v"](np.asarray(r) / L)
    N, v, _ = V2.engine_avg_radial(Nf, vf, ops.R, sigma)
    h = ops.h(N, v)
    eta = np.zeros(ops.n) if cap_onset_L is None else CU.cap_reviewa(ops.R, L, cap_onset_L, cap_width_L, eta_max)
    return ops, N, v, h, np.repeat(eta, 2)


def run_ext(name, h, eta, ops, F, imask, L, ts, rc_mask=None):
    A = (-1j * (h - 1j * sp.diags(eta))).tocsc()
    dt = ts[1] - ts[0]
    Ps = expm_multiply(A, F, start=0.0, stop=ts[-1], num=len(ts), endpoint=True)       # (n_t, d, K)
    K = F.shape[1]
    ret = np.real(np.einsum("tik,tik->tk", Ps[:, imask].conj(), Ps[:, imask]))
    S = np.einsum("ik,til->tkl", F.conj(), Ps)
    if rc_mask is None:
        G = np.array([np.eye(K) - P.conj().T @ P for P in Ps])
    else:
        G = np.array([P[rc_mask].conj().T @ P[rc_mask] for P in Ps])
    G = 0.5 * (G + np.conj(np.transpose(G, (0, 2, 1))))
    return {"ret": ret, "S": S, "G": G, "dt": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["F1", "F2"], default="F2")
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--sigma", type=float, default=2.0, help="back-average std in sites (s = 16: sigma/a 2 in both specs)")
    ap.add_argument("--kappas", default="1,2,3")
    ap.add_argument("--win", type=float, default=3.0, help="supply window (0, win T0)")
    ap.add_argument("--nearest", type=int, default=3)
    ap.add_argument("--T-over-L", type=float, default=100.0)
    ap.add_argument("--dt-over-L", type=float, default=0.25)
    ap.add_argument("--bin-over-L", type=float, default=10.0)
    ap.add_argument("--lead", type=float, default=200.0)
    ap.add_argument("--rc", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    L = a.L; T0 = 1.0 / (4 * math.pi * L)
    prof = (lambda: CU.dense_record_profile(*CU.REC_F1)) if a.family == "F1" else CU.f2_profile
    prof = prof()
    ts = np.arange(0.0, a.T_over_L * L + 1e-9, a.dt_over_L * L)
    res = {"script": "run_bprime_b11_v1.py", "args": vars(a), "profile": prof["label"], "L": L, "T0": T0, "own_md5": K3.own_md5s(), "kappa": {}}
    npz = {}
    for kappa in [int(x) for x in a.kappas.split(",")]:
        ex = {"cap2": (int(round(2.5 * L)), 2.0, None), "lead": (int(round(a.lead * L)), None, a.rc), "lead2": (int(round(2 * a.lead * L)), None, a.rc),
              "cap50": (int(round((a.rc + 0.5) * L)), a.rc, None)}
        ops0, N0, v0, h0, _ = build(kappa, prof, L, a.sigma, ex["cap2"][0], 2.0)
        g = N0 - np.abs(v0)
        sc = np.where(np.sign(g[1:]) != np.sign(g[:-1]))[0]
        R_I = float(ops0.R[sc[-1]] - g[sc[-1]] * (ops0.R[sc[-1] + 1] - ops0.R[sc[-1]]) / (g[sc[-1] + 1] - g[sc[-1]])) if len(sc) else max(prof["roots_over_L"]) * L
        im0 = ops0.site_mask(R_I)
        zi = {}
        es, Us, dI = V2.interior_window_det(h0, im0, 0.0, a.win * T0, info=zi)
        ee, Ue, _ = V2.interior_window_det(h0, im0, -T0, 0.0, info=zi)
        if a.nearest and len(es) + len(ee) < 2 * a.nearest:
            # the chains have (almost) no level in the window at any size (count ~ R_I T0/pi per valley): B11 then uses the --nearest levels
            # of each sign closest to e = 0 (labelled; their e/T0 printed), which are the lowest-energy quanta the port can carry
            ea, Ua, dI = V2.interior_window_det(h0, im0, 0.0, 1e3, info=zi)
            eb, Ub, _ = V2.interior_window_det(h0, im0, -1e3, 0.0, info=zi)
            es, Us = ea[: a.nearest], Ua[:, : a.nearest]
            ee, Ue = eb[-a.nearest:], Ub[:, -a.nearest:]
            zi["inputs"] = "nearest %d per sign (window empty or short)" % a.nearest
        e_in = np.concatenate([es, ee]); ns = len(es); Kc = len(e_in)
        kres = {"R_I": R_I, "retained_dim": dI, "n_supply": ns, "n_exit": len(ee), "e_over_T0": [float(x) for x in e_in / T0], "zero": zi}
        K3.log("kappa %d: R_I %.3f, inputs %d + %d" % (kappa, R_I, ns, len(ee)))
        if Kc == 0:
            res["kappa"][str(kappa)] = kres; continue
        Fsmall = np.concatenate([Us, Ue], axis=1)                         # on the cap2 chain's sites (2 n0)
        out = {}
        for name, (n_s, onset, rc) in ex.items():
            ops, N, v, h, eta = build(kappa, prof, L, a.sigma, n_s, onset)
            dmax = float(np.max(np.abs(N[: ops0.n] - N0[: ops.n])) if ops.n >= ops0.n else 0.0)
            F = np.zeros((2 * ops.n, Kc), complex); F[: 2 * ops0.n] = Fsmall
            im = ops.site_mask(R_I)
            rcm = None if rc is None else np.repeat(ops.R > rc * L, 2)
            r_ = run_ext(name, h, eta, ops, F, im, L, ts, rcm)
            bins = [k for k in range(len(ts)) if k > 0 and abs(ts[k] / L / a.bin_over_L - round(ts[k] / L / a.bin_over_L)) < 1e-9]
            Gl = [(float(ts[k]), r_["G"][k]) for k in bins]
            C0 = np.zeros((Kc, Kc), complex); C0[np.arange(ns), np.arange(ns)] = 1.0
            rd = V2.bins_reader_json(V2.bins_reader(Gl, C0, ns=ns)) if ns else None
            poles = V2.poles_for_inputs(r_["S"][:, np.arange(Kc), np.arange(Kc)], r_["dt"], e_in, L, ts[-1])
            spec = {"%g" % (ts[k] / L): [float(x) for x in np.sort(np.linalg.eigvalsh(r_["G"][k]))[::-1]] for k in bins}
            Gby = {float(ts[k]): r_["G"][k] for k in range(1, len(ts))}
            Ttheta = {th: min(range(1, len(ts)), key=lambda k: abs(ts[k] - math.log(2) * L / th)) for th in (0.1, 0.01)}
            nacc = {"%g" % th: int(np.sum(np.linalg.eigvalsh(r_["G"][k]) > 0.5)) for th, k in Ttheta.items()}
            out[name] = {"n_sites": n_s, "fields_inside_same_max_dN": dmax, "p_spectrum_by_bin": spec, "N_acc_time": nacc, "poles": poles,
                         "retained_at_bins": {"%g" % (ts[k] / L): [float(x) for x in r_["ret"][k]] for k in bins}, "bins_reader": rd}
            npz["k%d_%s_ret" % (kappa, name)] = r_["ret"]; npz["k%d_%s_Sdiag" % (kappa, name)] = r_["S"][:, np.arange(Kc), np.arange(Kc)]
            npz["k%d_%s_Gbins" % (kappa, name)] = np.array([g_ for _, g_ in Gl])
            K3.log("  kappa %d %s: n %d, p(T) %s, ret(T) %s (%.0f s)" % (kappa, name, n_s, np.round(spec["%g" % (ts[-1] / L)], 4),
                                                                     np.round(r_["ret"][-1], 4), time.time() - t0))
        # comparisons
        def rel(pa, pb, floor=1e-2):
            pa = np.asarray(pa); pb = np.asarray(pb); m = (pa > floor) | (pb > floor)
            return float(np.max(np.abs(pa[m] - pb[m]) / np.maximum(np.abs(pb[m]), 1e-300))) if m.any() else 0.0
        cmp = {}
        for A_, B_ in (("lead", "cap50"), ("lead2", "lead"), ("cap2", "cap50")):
            sa, sb = out[A_]["p_spectrum_by_bin"], out[B_]["p_spectrum_by_bin"]
            pr = {t: rel(sa[t], sb[t]) for t in sa}
            ra = np.array(out[A_]["retained_at_bins"][list(sa)[-1]]); rb = np.array(out[B_]["retained_at_bins"][list(sb)[-1]])
            wa = [p_.get("Gamma") for p_ in out[A_]["poles"]]; wb = [p_.get("Gamma") for p_ in out[B_]["poles"]]
            wr = [(x / y if (x is not None and y is not None and y > 0) else None) for x, y in zip(wa, wb)]
            rbins = list(sa)
            RA = np.array([out[A_]["retained_at_bins"][t] for t in rbins]); RB = np.array([out[B_]["retained_at_bins"][t] for t in rbins])
            mm = (RA > 1e-2) | (RB > 1e-2)
            ret_rel = float(np.max(np.abs(RA[mm] - RB[mm]) / np.maximum(RB[mm], 1e-300))) if mm.any() else 0.0
            ent = None
            if out[A_]["bins_reader"] and out[B_]["bins_reader"]:
                ent = float(max(abs(x["S_R"] - y["S_R"]) for x, y in zip(out[A_]["bins_reader"]["bins"], out[B_]["bins_reader"]["bins"])))
            cmp["%s_vs_%s" % (A_, B_)] = {"max_rel_dp_above_1e-2_by_bin": pr, "max_rel_dp_overall": max(pr.values()) if pr else None,
                                          "retained_T_abs_diff_max": float(np.max(np.abs(ra - rb))),
                                          "retained_rel_max_over_bins_above_1e-2": ret_rel, "width_ratio_per_input": wr,
                                          "S_R_abs_diff_max": ent}
        cmp["dilation_pass_2pct"] = bool(cmp["lead_vs_cap50"]["max_rel_dp_overall"] is not None and cmp["lead_vs_cap50"]["max_rel_dp_overall"] < 0.02)
        cmp["lead_length_pass_2pct"] = bool(cmp["lead2_vs_lead"]["max_rel_dp_overall"] is not None and cmp["lead2_vs_lead"]["max_rel_dp_overall"] < 0.02)
        cmp["port2L_clean_2pct"] = bool(cmp["cap2_vs_cap50"]["max_rel_dp_overall"] is not None and cmp["cap2_vs_cap50"]["max_rel_dp_overall"] < 0.02)
        cmp["retained_lead_vs_cap50_2pct"] = bool(cmp["lead_vs_cap50"]["retained_rel_max_over_bins_above_1e-2"] < 0.02)
        cmp["retained_cap2_vs_cap50_2pct"] = bool(cmp["cap2_vs_cap50"]["retained_rel_max_over_bins_above_1e-2"] < 0.02)
        kres["exteriors"] = out; kres["comparisons"] = cmp
        K3.log("kappa %d comparisons: retained rel lead/cap50 %.4g, cap2/cap50 %.4g; p rel lead/cap50 %.4g, lead2/lead %.4g, cap2/cap50 %.4g; width ratios cap2/cap50 %s" % (
            kappa, cmp["lead_vs_cap50"]["retained_rel_max_over_bins_above_1e-2"], cmp["cap2_vs_cap50"]["retained_rel_max_over_bins_above_1e-2"], cmp["lead_vs_cap50"]["max_rel_dp_overall"] or 0, cmp["lead2_vs_lead"]["max_rel_dp_overall"] or 0,
            cmp["cap2_vs_cap50"]["max_rel_dp_overall"] or 0, cmp["cap2_vs_cap50"]["width_ratio_per_input"]))
        res["kappa"][str(kappa)] = kres
        K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    np.savez_compressed(os.path.join(a.out, a.tag + ".npz"), t=ts, **npz)
    res["seconds"] = time.time() - t0
    K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    K3.log("wrote %s (%.0f s)" % (a.tag, res["seconds"]))


if __name__ == "__main__":
    main()
