#!/usr/bin/env python
"""engine_rows_floor_v1.py (builder analyst; engine spec 26.1, BUILDER: "the candidates (R28b's), tested in this order on the regime member at
ell 4 and 8 ...: (i) density consistency, rho_ell from the constraint solve (info["rho_ell"]) against K_0[V rho_shell] of the same state at the
nodes, the relative difference printed (1e-2 is the floor; 1e-12 excludes the candidate); (ii) the differentiation, the brackets at h = ell/16,
ell/32, ell/64 with the radius of each maximum and the nodes from ell (the interval ell to 2 ell is uncovered by the clocked comparison);
(iii) the stress-to-source map, the brackets with shell n's S_r taken as its outer face's value and, apart, with the K_2 term dropped, each
labelled", and "at T/rs 0, 5 and 10 the residual with d_t k set to zero beside the full one and the radius of the maximum"). --mode static:
a converged static member (engine_star_v7's family json at its own ell) rebuilt as in engine_rows_static_v2 (the dust's Noether stresses of
engine_stress_v2; the dense re-solve on engine_fields_v14), the three tests on the nodes from ell to 0.9 R_surf (and from 2 ell beside); the
brackets of engine_rows_v2 (d_t k = 0 for a static member) relative to their largest term. --mode records: a collapse run with the handoff
npz of its shell quantities, at --times (T/rs) the residual d_t k - bracket (d_t k by the centred difference of the neighbouring records)
and the bracket alone (d_t k set to zero), each's largest |value|/max(term, |d_t k|) over the matter's support with its radius. Every
number is printed by the script. CPU. Args: --mode static --family --NB | --mode records --run --handoff --times; --md."""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_v2 as RW  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402


def smooth_noK2(Sr, St, R, ell, r_eval):
    """the tensor smoothing with the K_2 projection dropped: S_r,ell = S_t,ell = [S_t]_0 + (1/3)[S_r - S_t]_0."""
    V = 4.0 * math.pi * np.asarray(R, float) ** 2
    K0 = STR2.kernels_l(r_eval, R, ell, 0)
    St0 = K0 @ (V * St); D0 = K0 @ (V * (Sr - St))
    x = St0 + D0 / 3.0
    return x, x.copy()


def floor(info, rho_f, Sr_l, St_l, g, rr, h):
    b = RW.brackets(info["N_dense"], info["v_dense"], rho_f, Sr_l, St_l, g, rr, h)
    qr = np.abs(b["rhs_rr"]) / b["terms_rr"]; qt = np.abs(b["rhs_tt"]) / b["terms_tt"]
    ir = int(np.argmax(qr)); it = int(np.argmax(qt))
    return float(qr[ir]), float(rr[ir]), float(qt[it]), float(rr[it])


def static_mode(a):
    import torch
    import engine_rows_static_v2 as RS2
    import engine_fields_v14 as F14
    from engine_ops_v1 import sources_static
    Jf = json.load(open(a.family))
    mem = [q for q in Jf["members"] if not q.get("failed") and abs(q["N_B"] - a.NB) < 1e-3][0]
    m = float(mem["m"]); g = float(mem["g"]); ell = float(mem["ell"]); nsites = int(mem["nsites"])
    R, chans = RS2.orbitals(np.asarray(mem["N"], float), np.asarray(mem["v"], float), m, nsites, mem["N_B"], bool(Jf["args"].get("half_offset", 0)))
    eps = np.zeros(nsites); J = np.zeros(nsites); P = None
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, J1 = sources_static(h, Q, U, f, w); eps += e1; J += J1
        ch = SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R)
        cs = STR2.ChanStress(ch, torch, torch.device("cpu"))
        p = STR2.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
    SV = STR2.shell_values(P, R, m)
    _, _, _, info = F14.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag")
    Rs = float(mem["R_surf"])
    L = ["## The rows' static floor: R28b's candidates (engine_rows_floor_v1.py; %s, N_B %.3f, ell %g, m a %g, R_surf %.2f)" % (
        os.path.basename(a.family), mem["N_B"], ell, m, Rs), ""]
    out = {}
    for lo_lab, lo in (("from ell", ell), ("from 2 ell", 2 * ell)):
        rr = np.arange(lo, 0.9 * Rs, ell)
        rho_s = np.asarray(info["rho_ell"](rr), float)
        rho_k = STR2.kernels_l(rr, R, ell, 0) @ eps
        d_rho = float(np.max(np.abs(rho_s - rho_k))) / max(float(np.max(np.abs(rho_k))), 1e-300)
        L += ["### Nodes %s to 0.9 R_surf (%d nodes)" % (lo_lab, len(rr)), "",
              "- (i) density consistency: max |rho_ell (the solve's) - K_0[V rho_shell]| over the nodes, relative to the largest: %.3e." % d_rho, "",
              "| test | rr max (at r) | tt max (at r) |", "|---|---|---|"]
        Sr_l, St_l, _ = STR2.smooth_stress(SV["S_r"], SV["S_t"], R, ell, rr)
        res = {}
        for lab, hh in (("(ii) h = ell/16", ell / 16.0), ("(ii) h = ell/32 (the rows' choice)", ell / 32.0), ("(ii) h = ell/64", ell / 64.0)):
            res[lab] = floor(info, rho_s, Sr_l, St_l, g, rr, hh)
        Srf = SV["S_r_face"]
        Sr_out = Srf.copy()
        S1, T1, _ = STR2.smooth_stress(Sr_out, SV["S_t"], R, ell, rr)
        res["(iii) shell S_r := its outer face's value"] = floor(info, rho_s, S1, T1, g, rr, ell / 32.0)
        S2, T2 = smooth_noK2(SV["S_r"], SV["S_t"], R, ell, rr)
        res["(iii) the K_2 term dropped"] = floor(info, rho_s, S2, T2, g, rr, ell / 32.0)
        res["(i') the rows with rho := K_0[V rho_shell]"] = floor(info, rho_k, Sr_l, St_l, g, rr, ell / 32.0)
        for lab, (qr, rr_, qt, rt_) in res.items():
            L.append("| %s | %.3e (%.1f) | %.3e (%.1f) |" % (lab, qr, rr_, qt, rt_))
        L.append("")
        out[lo_lab] = {"density_rel": d_rho, **{k: list(v) for k, v in res.items()}}
    L += ["- the brackets relative to their largest term (d_t k = 0); every number above is printed by the script."]
    return L, out


def records_mode(a):
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    H = np.load(a.handoff, allow_pickle=True)
    script = J.get("script", "")
    if "v17" in script:
        import engine_fields_v12 as FM
    elif "v19" in script:
        import engine_fields_v14 as FM
    else:
        import engine_fields_v16 as FM
    ra = J["args"]; rs = float(ra["rs"]); g = float(ra["g"]); ell = float(ra["ell"]); tau = float(ra["tau"])
    R = np.asarray(Z["R"], float); Tz = np.asarray(Z["T"], float); TH = np.asarray(H["T"], float)

    def at(T):
        k = int(np.argmin(np.abs(TH - T))); kz = int(np.argmin(np.abs(Tz - T)))
        if abs(TH[k] - T) > 1e-9 or abs(Tz[kz] - T) > 1e-9:
            return None
        e = np.asarray(Z["eps_tot"][kz], float); Jt = np.asarray(Z["J_tot"][kz], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, g, R, ell, collar_policy=ra.get("collar_policy", "flag"), support_rel=ra.get("support_rel", 1e-14))
        if info.get("failed"):
            return None
        Sr = np.zeros(len(R)); St = np.zeros(len(R))
        for part in ("dust", "pairs"):
            s1, s2 = RW.shell_stresses(H, k, part, R); Sr += s1; St += s2
        rc = float(info["r_collar"]); nodes = np.arange(ell, rc - 2 * ell + 1e-9, ell)
        return {"info": info, "nodes": nodes, "Sr": Sr, "St": St, "rho_ell": np.asarray(info["rho_ell"](nodes), float)}

    L = ["## The rows' residual with and without d_t k on the records of %s (engine_rows_floor_v1.py; fields %s; tau %g)" % (tag, FM.__name__, tau), "",
         "| T/rs | d_t k | res_rr support (at r) | res_tt support (at r) |", "|---|---|---|---|"]
    out = {}
    for t in [float(x) for x in a.times.split(",")]:
        T = t * rs
        cur = at(T)
        prv = at(T - tau) if T - tau >= -1e-9 else None
        nxt = at(T + tau)
        if cur is None:
            L.append("| %.3f | the record is missing or its solve failed | - | - |" % t); continue
        nodes = cur["nodes"]; n = len(nodes)
        Sr_l, St_l, _ = STR2.smooth_stress(cur["Sr"], cur["St"], R, ell, nodes)
        bb = RW.brackets(cur["info"]["N_dense"], cur["info"]["v_dense"], cur["rho_ell"], Sr_l, St_l, g, nodes, ell / 32.0)
        sup = np.abs(cur["rho_ell"]) > 1e-6 * max(float(np.max(np.abs(cur["rho_ell"]))), 1e-300)
        kk = {}
        for lab_, s_ in (("prv", prv), ("nxt", nxt)):
            if s_ is not None:
                b0 = RW.brackets(s_["info"]["N_dense"], s_["info"]["v_dense"], np.asarray(s_["info"]["rho_ell"](nodes), float), np.zeros(n), np.zeros(n), g, nodes, ell / 32.0)
                kk[lab_] = (b0["k_r"], b0["k_t"])
        if "prv" in kk and "nxt" in kk:
            dkr = (kk["nxt"][0] - kk["prv"][0]) / (2 * tau); dkt = (kk["nxt"][1] - kk["prv"][1]) / (2 * tau); how = "centred"
        elif "nxt" in kk:
            dkr = (kk["nxt"][0] - bb["k_r"]) / tau; dkt = (kk["nxt"][1] - bb["k_t"]) / tau; how = "one-sided"
        else:
            dkr = np.zeros(n); dkt = np.zeros(n); how = "none"
        out[t] = {}
        for lab, (xr, xt) in (("full (%s)" % how, (dkr, dkt)), ("set to zero", (np.zeros(n), np.zeros(n)))):
            resr = xr - bb["rhs_rr"]; rest = xt - bb["rhs_tt"]
            tr = np.maximum(bb["terms_rr"], np.abs(xr)); tt = np.maximum(bb["terms_tt"], np.abs(xt))
            qr = np.where(sup, np.abs(resr) / tr, 0.0); qt = np.where(sup, np.abs(rest) / tt, 0.0)
            ir = int(np.argmax(qr)); it = int(np.argmax(qt))
            out[t][lab] = (float(qr[ir]), float(nodes[ir]), float(qt[it]), float(nodes[it]))
            L.append("| %.3f | %s | %.3e (%.1f) | %.3e (%.1f) |" % (t, lab, qr[ir], nodes[ir], qt[it], nodes[it]))
    L += ["", "- the stresses of the handoff (dust and pairs summed) smoothed by engine_stress_v2; every number above is printed by the script."]
    return L, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "records"], required=True)
    ap.add_argument("--family"); ap.add_argument("--NB", type=float)
    ap.add_argument("--run"); ap.add_argument("--handoff"); ap.add_argument("--times", default="0,5,10")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    L, out = static_mode(a) if a.mode == "static" else records_mode(a)
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
