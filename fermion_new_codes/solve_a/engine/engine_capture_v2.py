#!/usr/bin/env python
"""engine_capture_v2.py (builder analyst; C-capture as restated in SPEC_final4b_engine.md sec. 13.3: the non-return fraction; version 2 of the
lead's engine_capture_v1.py, whose first-passage fluence is kept only as a crossing monitor).

On the frozen late background (a collapse slice's dense functions re-solved from its stored shell totals; and, as the control, the same interior
with the exterior replaced by N = 1, v = -sqrt(2 g E/R) through a tanh junction at --Rx r_s of width 4), per channel kappa (both signs: the
chain's two valleys) and Killing frequency omega: an incident packet of the exterior's ingoing branch (flow_chart_v4.branch_momenta with the
channel's transverse mass kappa/R_0, the frequency width --sigma-omega-rel omega) built at R_0 = --R0-over-rs r_s (build_packet with the local
spinor) and projected on positive Killing frequency (the erf step of h_m at width --ws, Chebyshev); the incident weight I; the state propagated
on the frozen h_m (Chebyshev); at the report times T = 2 r_s ln(1/(a omega)) + {2, 4, 8} r_s the returned weight R(T): the norm at R > R_t + 2
on the outward branch (a block-wise Gabor projection: blocks of --block sites, each kept on the momenta of positive local group velocity of the
channel symbol at the block's N, v, kappa/R), beside the plain outside norm; A(omega, kappa, T) = 1 - R(T)/I; the conversion monitor: the
weight inside R_t on positive local group velocity (the escape branch) against time; the cross section sigma(omega, T) = (pi/p^2) sum_{kappa>0}
kappa [A_{+kappa} + A_{-kappa}], the geometric check b_c^2 = min_{R > R_t} R^2/(1 - v^2) (and with the lapse, R^2 N^2/(N^2 - v^2), labelled),
pi b_c^2 against 27 pi (g E)^2, sigma/(pi b_c^2); the absorbing-port variant (--absorber: -i Gamma(R) inside R_t - 2, a smooth ramp to --gamma,
propagated by scipy's expm_multiply) as the ideal-absorber comparator. The probes: the massless fermion at omega r_s in --omega-rs and the
massive at a omega in --fermion-omega-a with m a = --m. Channels from |kappa| = 1 until A < 1e-3 at both signs and at every T for two
consecutive |kappa| beyond omega b_c. CPU (sparse). Args: --fields --records --slice --g --Rx --omega-rs --fermion-omega-a --m --R0-over-rs
--sigma-omega-rel --ws --block --absorber --gamma --kappa-cap --control --out-json --md --tag --gpu (accepted, unused)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.fft import dct
from scipy.special import erfc, jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "flow", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import engine_vacflow_v6 as VF  # noqa: E402  (chain_sparse: m sz + (kappa/R) sy on site, -(i/2) sx bond, the two-step Q)
import flow_chart_v4 as FC  # noqa: E402


class SB:
    def __init__(self, H):
        self.H = H.tocsr()
        self.bound = float(abs(self.H).sum(axis=1).max())

    def apply(self, X):
        return self.H @ X


def cheb_expm(B, X, tau, tol=1e-15):
    if tau == 0:
        return X
    r = B.bound; a = r * tau
    K = int(abs(a) + 12)
    while abs(jv(K, abs(a))) > tol or K < abs(a) + 4:
        K += 4
    T0, T1 = X, B.apply(X) / r
    Y = jv(0, a) * T0 + 2 * (-1j) * jv(1, a) * T1
    for k in range(2, K + 1):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + 2 * ((-1j) ** k) * jv(k, a) * T2
        T0, T1 = T1, T2
    return Y


def cheb_neg_step(B, X, ws):
    """f(h) X with f(E) = erfc(E/(sqrt2 ws))/2 (the negative-energy step)."""
    r = B.bound
    M = 1
    while M < 8 * r / ws:
        M *= 2
    th = (np.arange(M) + 0.5) * math.pi / M
    c = dct(0.5 * erfc(r * np.cos(th) / (math.sqrt(2.0) * ws)), type=2) / M
    c[0] /= 2.0
    K = int(np.nonzero(np.abs(c) > 1e-13)[0][-1]) + 1
    T0, T1 = X, B.apply(X) / r
    Y = c[0] * T0 + c[1] * T1
    for k in range(2, K):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
    return Y


def clocked_sparse(hs, Qs, N, v):
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ hs + hs @ Nd) + 0.5 * (Vd @ Qs + Qs @ Vd)).tocsr()


def outward_weight(psi, R, N, v, m, kappa, lo, hi, block):
    """the norm on [lo, hi) kept on the momenta of positive local group velocity, block by block (a Gabor projection with a cosine-squared
    window per block, the blocks overlapping by half, the windows summing to one)."""
    a = psi.reshape(-1, 2)
    tot = 0.0
    starts = np.arange(lo, hi, block // 2)
    for s0 in starts:
        sel = np.nonzero((R >= s0) & (R < s0 + block) & (R >= lo) & (R < hi))[0]
        if len(sel) < 4:
            continue
        xw = (R[sel] - s0) / block
        win = np.sin(math.pi * xw) ** 2                                  # half-overlapping sin^2 windows sum to one
        seg = a[sel] * np.sqrt(win)[:, None]
        n = 4 * len(sel)
        k = 2 * np.pi * np.fft.fftfreq(n)
        ak = np.fft.fft(seg, n=n, axis=0)
        mid = int(sel[len(sel) // 2])
        vg = N[mid] * np.sin(k) * np.cos(k) / np.sqrt(m * m + (kappa / R[mid]) ** 2 + np.sin(k) ** 2) + v[mid] * np.cos(2 * k)
        keep = vg > 0
        tot += float(np.sum(np.abs(ak[keep]) ** 2)) / n
    return tot


def run_channel(R, N, v, m, kappa, omega, Rt, R0, sig_om, ws, T_list, mon_dt, block, absorber=None):
    hs, Qs = VF.chain_sparse(R, m, float(kappa))
    hm = clocked_sparse(hs, Qs, N, v)
    B = SB(hm)
    mc = math.sqrt(m * m + (kappa / R0) ** 2)
    vext = float(np.interp(R0, R, v))
    try:
        k, om, a2 = FC.branch_momenta(omega, sig_om, mc, vext, outgoing=False)
    except ValueError as err:
        return {"error": str(err)}
    psi0 = FC.build_packet(len(R), int(round(R0 - R[0])), k, a2, mc, m=m, a=kappa / R0)
    psi0 = psi0 - cheb_neg_step(B, psi0, ws)                                # positive Killing frequency
    I = float(np.vdot(psi0, psi0).real)
    out = {"kappa": kappa, "I": I, "A": {}, "R_out": {}, "outside_plain": {}, "monitor": []}
    tgrid = sorted(set([round(t, 6) for t in T_list] + list(np.round(np.arange(mon_dt, max(T_list) + 1e-9, mon_dt), 6))))
    psi, tprev = psi0, 0.0
    Rmax = float(R[-1]) - 16.0
    for t in tgrid:
        psi = cheb_expm(B, psi, t - tprev); tprev = t
        inside_out = outward_weight(psi, R, N, v, m, kappa, 1.0, Rt, block)
        out["monitor"].append({"t": t, "inside_outward": inside_out / I, "inside_total": float(np.sum(np.abs(psi.reshape(-1, 2)[R < Rt]) ** 2)) / I})
        if any(abs(t - T) < 1e-6 for T in T_list):
            Rw = outward_weight(psi, R, N, v, m, kappa, Rt + 2.0, Rmax, block)
            plain = float(np.sum(np.abs(psi.reshape(-1, 2)[(R > Rt + 2.0)]) ** 2))
            key = "%.4f" % t
            out["R_out"][key] = Rw / I; out["outside_plain"][key] = plain / I; out["A"][key] = 1.0 - Rw / I
    if absorber is not None:
        from scipy.sparse.linalg import expm_multiply
        gam = absorber["gamma"] * np.clip((Rt - 2.0 - R) / 8.0, 0.0, 1.0)
        Hc = (hm - 1j * sp.diags(np.repeat(gam, 2))).tocsc()
        out["absorbed"] = {}
        for T in T_list:
            pT = expm_multiply(-1j * Hc * T, psi0)
            out["absorbed"]["%.4f" % T] = 1.0 - float(np.vdot(pT, pT).real) / I
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--Rx", type=float, default=0.5)
    ap.add_argument("--omega-rs", type=float, nargs="*", default=[2.0, 4.0, 8.0])
    ap.add_argument("--fermion-omega-a", type=float, nargs="*", default=[0.25, 0.3, 0.4])
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--R0-over-rs", type=float, default=8.0)
    ap.add_argument("--sigma-omega-rel", type=float, default=0.1)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--absorber", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--kappa-cap", type=int, default=200)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--collar-policy", default="flag")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--tag", default="capture")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    out_json = a.out_json or "CAP/%s.json" % a.tag
    md = a.md or "CAP/%s.md" % a.tag
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    import engine_fields_v8 as F8
    Z = np.load(a.fields, allow_pickle=True)
    idx = a.slice if a.slice >= 0 else len(Z["T"]) + a.slice
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints(np.asarray(Z["eps_tot"][idx], float), np.asarray(Z["J_tot"][idx], float), a.g, R, ell,
                                            N0=np.asarray(Z["N"][idx], float), collar_policy=a.collar_policy)
    if info.get("failed"):
        raise SystemExit("the re-solve failed: %s" % info["failed"])
    E = float(info["E_collar"]); rs = 2 * a.g * E
    tz = F8.trapping_radius(Ns, vs, R, info)
    Rt = tz["zeros"][-1] if tz["zeros"] else None
    if Rt is None:
        raise SystemExit("no trapping surface on this slice")
    bgs = [("solved slice", Ns, vs)]
    if a.control:
        s = 0.5 * (1 + np.tanh((R - a.Rx * rs) / 4.0))
        Nc = (1 - s) * Ns + s * 1.0
        vc = (1 - s) * vs + s * (-np.sqrt(rs / R))
        bgs.append(("control: the exact exterior from %g r_s, the same interior" % a.Rx, Nc, vc))
    probes = [("massless", 0.0, w / rs) for w in a.omega_rs] + [("fermion m a %g" % a.m, a.m, w) for w in a.fermion_omega_a]
    rec = {"script": "engine_capture_v2.py", "args": vars(a), "slice": idx, "T": float(Z["T"][idx]), "E": E, "r_s": rs, "R_t": Rt, "results": []}
    t0 = time.time()
    for bl, N, v in bgs:
        ext = R > Rt
        bc2 = float(np.min(R[ext] ** 2 / (1 - v[ext] ** 2)))
        bc2N = float(np.min(R[ext] ** 2 * N[ext] ** 2 / (N[ext] ** 2 - v[ext] ** 2)))
        Rt_b = Rt
        if bl.startswith("control"):
            zz = np.nonzero((N[:-1] + v[:-1]) * (N[1:] + v[1:]) <= 0)[0]
            Rt_b = float(R[zz[-1]]) if len(zz) else Rt
        for pl, m, om in probes:
            if om <= m:
                continue
            p = math.sqrt(om * om - m * m)
            T_list = [2 * rs * math.log(1.0 / om) + x * rs for x in (2.0, 4.0, 8.0)]
            chans = []
            quiet = 0
            for ka in range(1, a.kappa_cap + 1):
                pair = []
                for sgn in (1, -1):
                    r_ = run_channel(R, N, v, m, sgn * ka, om, Rt_b, a.R0_over_rs * rs, a.sigma_omega_rel * om, a.ws, T_list, rs, a.block,
                                     {"gamma": a.gamma} if a.absorber else None)
                    pair.append(r_)
                chans.append({"kappa": ka, "plus": pair[0], "minus": pair[1]})
                Amax = max([x for r_ in pair if "A" in r_ for x in r_["A"].values()] + [0.0])
                quiet = quiet + 1 if (Amax < 1e-3 and ka > om * math.sqrt(bc2)) else 0
                print("%s | %s omega %.5f: |kappa| %d: A(+) %s A(-) %s (%.0f s)" % (bl[:12], pl, om, ka, {k: round(x, 5) for k, x in pair[0].get("A", {}).items()},
                      {k: round(x, 5) for k, x in pair[1].get("A", {}).items()}, time.time() - t0), flush=True)
                if quiet >= 2:
                    break
            sig = {}
            sig_abs = {}
            for T in T_list:
                key = "%.4f" % T
                sig[key] = math.pi / p ** 2 * sum(c["kappa"] * (c["plus"].get("A", {}).get(key, 0.0) + c["minus"].get("A", {}).get(key, 0.0)) for c in chans)
                if a.absorber:
                    sig_abs[key] = math.pi / p ** 2 * sum(c["kappa"] * (c["plus"].get("absorbed", {}).get(key, 0.0) + c["minus"].get("absorbed", {}).get(key, 0.0)) for c in chans)
            res = {"background": bl, "probe": pl, "m": m, "omega": om, "p": p, "T_list": T_list, "sigma": sig, "sigma_absorber": sig_abs,
                   "pi_bc2": math.pi * bc2, "pi_bc2_lapse": math.pi * bc2N, "27_pi_gE2": 27 * math.pi * (a.g * E) ** 2,
                   "sigma_over_pi_bc2": {k: s_ / (math.pi * bc2) for k, s_ in sig.items()}, "channels": chans, "R_t_background": Rt_b}
            rec["results"].append(res)
            json.dump(rec, open(out_json, "w"), indent=1, default=float)
    L = ["## C-capture (engine_capture_v2.py, engine 13.3): slice %d (T %.1f), E %.6f, r_s %.4f, R_t %.4f" % (idx, rec["T"], E, rs, Rt), "",
         "| background | probe | omega | T (r_s) | sigma | sigma/(pi b_c^2) | pi b_c^2 (with the lapse) | 27 pi (g E)^2 | absorber sigma | channels |", "|---|---|---|---|---|---|---|---|---|---|"]
    for res in rec["results"]:
        for T in res["T_list"]:
            key = "%.4f" % T
            L.append("| %s | %s | %.5f | %.2f | %.4e | %.4f | %.4e (%.4e) | %.4e | %s | %d |" % (res["background"], res["probe"], res["omega"], T / rs, res["sigma"][key],
                     res["sigma_over_pi_bc2"][key], res["pi_bc2"], res["pi_bc2_lapse"], res["27_pi_gE2"], ("%.4e" % res["sigma_absorber"][key]) if res["sigma_absorber"] else "-",
                     len(res["channels"])))
    L += ["", "- every number above is printed by the script; A = 1 - R(T)/I with R(T) the outward-branch weight outside R_t + 2 (block-wise Gabor projection);",
          "  the per-channel A, the plain outside norms and the conversion monitor are in the JSON."]
    open(md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
