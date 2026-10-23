#!/usr/bin/env python
"""engine_rows_variants_v1.py (builder analyst; engine spec 25.3, BUILDER: "then the rows on the regime member (ell 4, 8) and on the
production's records at T/rs 5, 10 and 20.6 with two labelled stress variants, (A) S_t replaced by its continuum form, (B) the trace fixed
to rho - m b with the Noether anisotropy kept: a variant whose floor falls to the (ell k_F)^2 trend is a candidate dictionary, declared only
after its red team (R28b)"): the evolution rows' brackets (engine_rows_v2.brackets on the dense re-solve of the stored shell totals) with
three per-shell stresses smoothed by engine_stress_v2.smooth_stress: (N) the Noether stress S_r (the faces' flux, shells the mean of
their faces), S_t = f/(8 pi R); (A) S_r as N, S_t := (kappa/R) <sigma_y>/(8 pi R^2) summed over the channels (the continuum form); (B) the
anisotropy D = S_r - S_t of N kept and the trace fixed: S_t := (rho - m b - D)/3, S_r := S_t + D, rho and b the shell densities of the
same state. --mode static (a converged static member's family json at its own ell: the dust's stresses, the reference the ground state;
d_t k = 0): max |bracket|/max term over the nodes from ell to 0.9 R_surf, at 0.5 R_surf and outside 1.1 R_surf; --mode records (a
collapse run with its handoff npz, the times --times in T/rs): per time the residual d_t k - bracket with d_t k by the centred difference
of the neighbouring records' k (the fields' own, the same for the three variants), max |res|/max(term, |d_t k|) over the matter's support
(rho_ell above 1e-6 of its largest value). Every number is printed by the script. CPU. Args: --mode static --family --NB | --mode records
--run --handoff --times; --md."""
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


def variants(Sr, St, St_cont, rho, b, m):
    D = Sr - St
    StB = (rho - m * b - D) / 3.0
    return {"N": (Sr, St), "A": (Sr, St_cont), "B": (StB + D, StB)}


def static_mode(a):
    import torch
    import engine_rows_static_v2 as RS2
    import engine_fields_v14 as F14
    from engine_ops_v1 import sources_static
    Jf = json.load(open(a.family))
    mem = [q for q in Jf["members"] if not q.get("failed") and abs(q["N_B"] - a.NB) < 1e-3][0]
    m = float(mem["m"]); g = float(mem["g"]); ell = float(mem["ell"]); nsites = int(mem["nsites"])
    R, chans = RS2.orbitals(np.asarray(mem["N"], float), np.asarray(mem["v"], float), m, nsites, mem["N_B"], bool(Jf["args"].get("half_offset", 0)))
    eps = np.zeros(nsites); J = np.zeros(nsites); P = None; syk = np.zeros(nsites)
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, J1 = sources_static(h, Q, U, f, w); eps += e1; J += J1
        ch = SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R)
        cs = STR2.ChanStress(ch, torch, torch.device("cpu"))
        p = STR2.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
        syk += w * kappa * np.asarray(p["sy"], float)
    P["sy_k"] = syk
    SV = STR2.shell_values(P, R, m)
    _, _, _, info = F14.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag")
    Rs = float(mem["R_surf"])
    V = variants(SV["S_r"], SV["S_t"], SV["S_t_continuum"], SV["rho"], SV["b"], m)
    L = ["## The rows with the stress variants on a static member (engine_rows_variants_v1.py; %s, N_B %.3f, ell %g, m a %g, R_surf %.2f)" % (
        os.path.basename(a.family), mem["N_B"], ell, m, Rs), "",
         "| variant | rr inside (ell to 0.9 R_surf) | tt inside | at 0.5 R_surf (rr, tt) | outside 1.1 R_surf (rr, tt) |", "|---|---|---|---|---|"]
    out = {}
    for lab in ("N", "A", "B"):
        Sr, St = V[lab]
        res = []
        for rset in (np.arange(ell, 0.9 * Rs, ell), np.array([0.5 * Rs]), np.arange(math.ceil(1.1 * Rs / ell) * ell, float(info["r_collar"]) - 2 * ell, ell)):
            Sr_l, St_l, _ = STR2.smooth_stress(Sr, St, R, ell, rset)
            bb = RW.brackets(info["N_dense"], info["v_dense"], np.asarray(info["rho_ell"](rset), float), Sr_l, St_l, g, rset, ell / 32.0)
            res.append((float(np.max(np.abs(bb["rhs_rr"]) / bb["terms_rr"])), float(np.max(np.abs(bb["rhs_tt"]) / bb["terms_tt"]))))
        out[lab] = res
        L.append("| %s | %.3e | %.3e | %.3e, %.3e | %.3e, %.3e |" % ({"N": "(N) Noether", "A": "(A) S_t continuum form", "B": "(B) trace fixed to rho - m b"}[lab],
                                                                 res[0][0], res[0][1], res[1][0], res[1][1], res[2][0], res[2][1]))
    L += ["", "- d_t k = 0 for a static member; the brackets relative to their largest term; every number above is printed by the script."]
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
    ra = J["args"]; rs = float(ra["rs"]); g = float(ra["g"]); ell = float(ra["ell"]); tau = float(ra["tau"]); m = float(ra["m"])
    R = np.asarray(Z["R"], float); Tz = np.asarray(Z["T"], float); TH = np.asarray(H["T"], float)
    vol = 4.0 * math.pi * R ** 2

    def at(T):
        k = int(np.argmin(np.abs(TH - T))); kz = int(np.argmin(np.abs(Tz - T)))
        if abs(TH[k] - T) > 1e-9 or abs(Tz[kz] - T) > 1e-9:
            return None
        e = np.asarray(Z["eps_tot"][kz], float); Jt = np.asarray(Z["J_tot"][kz], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, g, R, ell, collar_policy=ra.get("collar_policy", "flag"), support_rel=ra.get("support_rel", 1e-14))
        if info.get("failed"):
            return None
        Sr = np.zeros(len(R)); St = np.zeros(len(R)); b = np.zeros(len(R)); syk = np.zeros(len(R))
        for part in ("dust", "pairs"):
            s1, s2 = RW.shell_stresses(H, k, part, R)
            Sr += s1; St += s2
            b += np.asarray(H["b_%s" % part][k], float); syk += np.asarray(H["sy_k_%s" % part][k], float)
        rho = e / vol; bden = b / vol
        Stc = syk / R / (8.0 * math.pi * R ** 2)
        rc = float(info["r_collar"]); nodes = np.arange(ell, rc - 2 * ell + 1e-9, ell)
        return {"info": info, "nodes": nodes, "V": variants(Sr, St, Stc, rho, bden, m), "rho_ell": np.asarray(info["rho_ell"](nodes), float)}

    L = ["## The rows with the stress variants on the records of %s (engine_rows_variants_v1.py; fields %s; tau %g)" % (tag, FM.__name__, tau), "",
         "| T/rs | variant | res_rr support | res_tt support |", "|---|---|---|---|"]
    out = {}
    for t in [float(x) for x in a.times.split(",")]:
        T = t * rs
        cur, prv, nxt = at(T), at(T - tau), at(T + tau)
        if cur is None or prv is None or nxt is None:
            L.append("| %.3f | the record or a neighbour is missing or its solve failed | - | - |" % t); continue
        n = len(cur["nodes"])
        kk = {}
        for lab_, s_ in (("prv", prv), ("nxt", nxt)):
            b0 = RW.brackets(s_["info"]["N_dense"], s_["info"]["v_dense"], np.asarray(s_["info"]["rho_ell"](cur["nodes"]), float), np.zeros(n), np.zeros(n), g, cur["nodes"], ell / 32.0)
            kk[lab_] = (b0["k_r"], b0["k_t"])
        dkr = (kk["nxt"][0] - kk["prv"][0]) / (2 * tau); dkt = (kk["nxt"][1] - kk["prv"][1]) / (2 * tau)
        sup = np.abs(cur["rho_ell"]) > 1e-6 * max(float(np.max(np.abs(cur["rho_ell"]))), 1e-300)
        out[t] = {}
        for lab in ("N", "A", "B"):
            Sr, St = cur["V"][lab]
            Sr_l, St_l, _ = STR2.smooth_stress(Sr, St, R, ell, cur["nodes"])
            bb = RW.brackets(cur["info"]["N_dense"], cur["info"]["v_dense"], cur["rho_ell"], Sr_l, St_l, g, cur["nodes"], ell / 32.0)
            resr = dkr - bb["rhs_rr"]; rest = dkt - bb["rhs_tt"]
            tr = np.maximum(bb["terms_rr"], np.abs(dkr)); tt = np.maximum(bb["terms_tt"], np.abs(dkt))
            rr_ = float(np.max(np.abs(resr[sup]) / tr[sup])); rt_ = float(np.max(np.abs(rest[sup]) / tt[sup]))
            out[t][lab] = (rr_, rt_)
            L.append("| %.3f | %s | %.3e | %.3e |" % (t, {"N": "(N) Noether", "A": "(A) S_t continuum form", "B": "(B) trace fixed to rho - m b"}[lab], rr_, rt_))
    L += ["", "- d_t k by the centred difference of the neighbouring records' k (the dense fields' own, the same for the three variants); the stresses of the handoff (dust and pairs summed) smoothed by engine_stress_v2; every number above is printed by the script."]
    return L, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "records"], required=True)
    ap.add_argument("--family"); ap.add_argument("--NB", type=float)
    ap.add_argument("--run"); ap.add_argument("--handoff"); ap.add_argument("--times", default="5,10,20.625")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    L, out = static_mode(a) if a.mode == "static" else records_mode(a)
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
