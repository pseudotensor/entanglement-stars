#!/usr/bin/env python
"""Strand 5 production P3's operator diagnostic, version 3 (builder analyst; SPEC_strand5_covariant_row.md sec. 9(a), after version 2's
finding).  Version 2 found route S's V(q) at the ring's smallest q (2 pi/2^18) far from V(q -> 0) taken from the stencil's moments, and
K^V nearly constant (|K_n| about 1e-2 down to n of order 1e4).  The builder's diagnosis (numerical, stated for the lead): strand5_P3_p06_v1's
V_stencil evaluates C(q) = sum_n K_n cos(q n) with the stencil's float64 coefficients, whose sum is zero only to rounding (setup_kernel sets
K_0 = -sum_{n != 0} K_n in float64); in 40-digit arithmetic the residue eps_0 = sum_n K_n adds -eps_0/q_lat^2 to k_a and about
-eps_0/(q^4 k_0^2) to V at the smallest q, a spike whose line kernel is an almost constant tail; in the radial action (V g)_j =
(1/r_j) sum_i (K_|j-i| - K_(j+i)) u_i a constant tail cancels, so Pi is unaffected, while P3b's checks compared against the spiked V(0).
Version 3 removes the residue at its source: C(q) = -2 sum_{n != 0} K_n sin^2(q n/2) (equal to sum_n K_n cos(q n) when sum_n K_n = 0, and
exactly zero at q = 0 whatever K_0's rounding), C'(q) = -sum_n n K_n sin(q n), q_lat^2 = 4 sin^2(q/2), in float64 for q >= 0.2 and in 40-digit
arithmetic below (as version 1); then, per kernel family: (i) V(0) from the ring's smallest q against the moments' intercept (M_4 - M_2)/
(6 M_2^2) and the five-point fit; (ii) K^V's decay, the partial sums and N_eff (relative 1e-10); (iii) P3b's two checks beyond max(2R, N_eff)
on a domain of four times that plus 400 nodes; (iv) Pi at the five members with this K^V against version 1's Pi (P3run1/P3.json), and the
prediction/measured ratios.  Version 2's (iv) (a K_0 shift by the DC difference) is not repeated: the difference was a long tail, not a
local term.  Writes the markdown (--md) and the JSON (--out-json)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import types

import mpmath as mp
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_P3_p06_v1 as P3  # noqa: E402
import strand5_lattice_star_v3 as S  # noqa: E402

mp.mp.dps = 40


def mp_V3(ns, ks, x):
    qq = mp.mpf(x)
    Cm = -2 * mp.fsum(mp.mpf(k) * mp.sin(qq * int(n) / 2) ** 2 for n, k in zip(ns, ks) if int(n) != 0)
    Cpm = -mp.fsum(mp.mpf(k) * int(n) * mp.sin(qq * int(n)) for n, k in zip(ns, ks))
    l2 = 4 * mp.sin(qq / 2) ** 2
    kam = -Cm / l2
    dkam = -(Cpm * l2 - Cm * 2 * mp.sin(qq)) / l2 ** 2
    return float(-dkam / (2 * qq * kam * kam)), float(kam)


def V_stencil3(stencil, q, small=0.2):
    ns = np.array(sorted(stencil), dtype=float)
    ks = np.array([stencil[int(n)] for n in ns])
    nz = ns != 0
    q = np.asarray(q, float)
    out, ka_out = np.empty_like(q), np.empty_like(q)
    ib = np.nonzero(q >= small)[0]
    for s in range(0, len(ib), 2048):
        sel = ib[s:s + 2048]
        qb = q[sel]
        C = -2 * (np.sin(np.outer(qb, ns[nz]) / 2) ** 2) @ ks[nz]
        Cp = -(np.sin(np.outer(qb, ns)) @ (ks * ns))
        ql2 = 4 * np.sin(qb / 2) ** 2
        ka = -C / ql2
        dka = -(Cp * ql2 - C * 2 * np.sin(qb)) / ql2 ** 2
        out[sel], ka_out[sel] = -dka / (2 * qb * ka * ka), ka
    isml = np.nonzero(q < small)[0]
    if len(isml) and len(isml) <= 64:
        for i in isml:
            out[i], ka_out[i] = mp_V3(ns, ks, q[i])
    elif len(isml):
        nodes = np.geomspace(0.999 * q[isml].min(), 1.25 * small, 160)
        vk = np.array([mp_V3(ns, ks, x) for x in nodes])
        out[isml] = CubicSpline(np.log(nodes), vk[:, 0])(np.log(q[isml]))
        ka_out[isml] = CubicSpline(np.log(nodes), vk[:, 1])(np.log(q[isml]))
    return out, ka_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--p3-json", required=True, help="version 1's record (P3run1/P3.json) for the Pi comparison")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    f, table = P3.f, P3.table
    old = json.load(open(a.p3_json))
    rec, r1, r2, r3, r4, summ = {}, [], [], [], [], []
    for item in a.families.split(";"):
        lab, fam_p, s2_p, karg, tab_p, twin_p = item.split("|")
        F = json.load(open(fam_p))
        S2 = json.load(open(s2_p))
        prof = np.load(fam_p.replace(".json", "_profiles.npz"))
        M = F["meta"]
        kk, kv = karg.split(" ", 1)
        args = types.SimpleNamespace(kernel=kv if kk == "--kernel" else None, packets=kv if kk == "--packets" else None, mlight=M["mlight"], aH=M["aH"],
                                     operator=M["operator"])
        stencil, kop, info = S.setup_kernel(args)
        gM, gB = M["couplings"]
        e1t = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
        R = max(stencil)
        ns = np.array(sorted(stencil), dtype=float)
        ks = np.array([stencil[int(n)] for n in ns])
        eps0 = float(mp.fsum(mp.mpf(k) for k in ks))
        M2, M4 = float(np.sum(ns ** 2 * ks)), float(np.sum(ns ** 4 * ks))
        V0_mom = (M4 - M2) / (6 * M2 * M2)
        q5 = 2 * np.pi * np.arange(1, 6) / 256
        _, ka5 = V_stencil3(stencil, q5)
        c = np.linalg.lstsq(np.stack([np.ones(5), q5 ** 2, q5 ** 4], 1), ka5, rcond=None)[0]
        V0_fit = -c[1] / c[0] ** 2
        nmax = P3.LR // 2
        K, V0S, qg, Vg = P3.line_kernel_V(lambda q: V_stencil3(stencil, q)[0], nmax)
        Ptot = float(K[0] + 2 * np.sum(K[1:nmax]) + K[nmax])
        Pn = K[0] + 2 * np.cumsum(np.concatenate([[0.0], K[1:nmax]]))
        below = np.nonzero(np.abs(Pn - Ptot) < 1e-10 * abs(Ptot))[0]
        Neff = int(below[0]) if len(below) else None
        r1.append([lab, "%.3e" % eps0, "%.10e" % V0S, "%.10e" % V0_mom, "%.10e" % V0_fit, f(V0S / V0_mom - 1, 3), f(V0_fit / V0_mom - 1, 3)])
        Ns = [x for x in (R, 2 * R, 4 * R, 1000, 4000, 16000) if x < nmax]
        r2.append([lab, str(R), "; ".join("N %d: |K_N| %.3e, P(N)/P_tot - 1 %.3e" % (N_, abs(K[N_]), Pn[N_] / Ptot - 1) for N_ in Ns), str(Neff)])
        L0 = max(2 * R, Neff or 0)
        if L0 <= 20000:
            Jc = 4 * L0 + 400
            Kc = K[:2 * Jc + 2]
            rc = np.arange(1, Jc + 1, dtype=float)
            flat = P3.apply_V(Kc, np.ones(Jc), rc) - V0S
            coul = rc * (P3.apply_V(Kc, 1.0 / rc, rc) - V0S / rc)
            sel = (rc > L0) & (rc < Jc - L0)
            r3.append([lab, str(L0), str(Jc), f(np.max(np.abs(flat[sel])) / abs(V0S), 2), f(np.max(np.abs(coul[sel])) / abs(V0S), 2)])
        else:
            r3.append([lab, str(L0), "skipped (above 20000)", "-", "-"])
        mems = S2["members"]
        five = S2["meta"].get("five_members") or list(range(len(mems)))
        Kfull = K[:2 * max(int(mems[i]["J"]) for i in five) + 4]
        fam = []
        oldf = {round(x["z_c"], 9): x for x in old.get(lab, [])}
        for i in five:
            d = mems[i]
            key = "z%.6f" % d["z_c"]
            r, psi, N, n = prof[key + "_r"], prof[key + "_psi"], prof[key + "_N"], prof[key + "_n"]
            star = S.Star(len(r), stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1t)
            eb, comb, nn, kap = star.matter(psi, N, d["mu_inf"])
            Pi = float(np.sum(4 * np.pi * r * r * psi ** 5 * eb * P3.apply_V(Kfull, N * psi ** 5 * comb, r)))
            meas = d["lock"].get("vac_part_of_AmB_over_A")
            pred = -Pi / (math.pi * kop * d["A"])
            o = oldf.get(round(d["z_c"], 9))
            fam.append({"z_c": d["z_c"], "Pi": Pi, "Pi_v1": o["Pi"] if o else None, "prediction": pred, "measured": meas})
            r4.append([lab, "%.6g" % d["z_c"], "%.6e" % Pi, "%.6e" % o["Pi"] if o else "-", f(Pi / o["Pi"] - 1, 3) if o else "-", f(pred / meas, 6) if meas else "-"])
        rec[lab] = {"eps0": eps0, "V0_ring": V0S, "V0_moment": V0_mom, "V0_fit": V0_fit, "N_eff": Neff, "R": R, "members": fam}
        summ.append("%s: the stencil's float64 sum eps_0 = %s; with it removed, V(0) at the ring's smallest q %s against the moments' %s (relative %s); "
                    "N_eff %s; Pi changes from version 1 by at most %s." % (
                        lab, "%.3e" % eps0, "%.6e" % V0S, "%.6e" % V0_mom, f(V0S / V0_mom - 1, 3), Neff,
                        f(max(abs(x["Pi"] / x["Pi_v1"] - 1) for x in fam if x["Pi_v1"]), 2)))
    out = ["# Strand 5 production P3's operator diagnostic, version 3 (strand5_P3_p06_v3.py)", "", "## S Summary (emitted)", ""] + ["- " + x for x in summ] + [
        "", "## E1 (i) The stencil's float64 sum and V(0): the ring's smallest q (zero-sum form), the moments' intercept, the five-point fit", "",
        table(["family", "eps_0 = sum K_n", "V(0) ring", "V(0) moments", "V(0) fit", "ring/moments - 1", "fit/moments - 1"], r1), "",
        "## E2 (ii) K^V's decay and DC partial sums (zero-sum form)", "", table(["family", "R", "N: |K_N|, P(N)/P_total - 1", "N_eff (relative 1e-10)"], r2), "",
        "## E3 (iii) P3b's checks beyond max(2R, N_eff) (relative to V(0))", "", table(["family", "max(2R, N_eff)", "J_c", "flat check", "Coulomb check"], r3), "",
        "## E4 (iv) Pi at the five members (zero-sum form) against version 1's, and the prediction/measured ratio", "",
        table(["family", "z_c", "Pi", "Pi (version 1)", "Pi/Pi_v1 - 1", "prediction/measured"], r4), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))


if __name__ == "__main__":
    main()
