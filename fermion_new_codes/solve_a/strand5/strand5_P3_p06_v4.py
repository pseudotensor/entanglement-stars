#!/usr/bin/env python
"""Strand 5 production P3, the complete-symbol variant (builder analyst; for the lead's ruling on the reviewer B red team of P1 with P3, which
found that P06's theorem needs the dilation multiplier of the complete stiffness T(q) = q^2 kappa(q^2), not of k_a(q) with q_lat^2 read as q^2).
Per kernel family and per its local twin, two multipliers on the same stencils and profiles:
  V_code(q) = -(dk_a/dq)/(2 q k_a^2), k_a = -C/q_lat^2 (P3 version 1's, here in the zero-sum form of version 3);
  V_eff(q) = (2 phi - q phi')/(2 phi^2), phi(q) = -C(q) = 2 sum_{n != 0} K_n sin^2(q n/2) (the complete symbol; kappa_eff = phi/q^2; the red
  team's form, equal to (q^2/d) V_code + (2d - q d')/(2 k_a d^2) with d = q_lat^2, so V_eff(0) = V_code(0) + 1/(12 k_a(0)));
both in float64 for q >= 0.2 and in 40-digit arithmetic below (cubic spline in ln q on 160 nodes, as version 1), the line kernel on the ring
of 2^18 points and the radial action of version 1.  Per member of the five: Pi with each multiplier, the prediction -Pi/(pi k_op A) and its
ratio to stage 2's measured kernel part of (A - B)/A; for the local twin (the stencil {-1: k_op, 0: -2 k_op, 1: k_op} on the twin's own
profiles; V_code = 0 identically) the same with V_eff against the twin's measured part; and V_code(0), V_eff(0), 1/(12 k_a(0)).
Writes the markdown (--md) and the JSON (--out-json).  Labelled: the red team's form, run for the lead's ruling; not a ruling."""
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
import strand5_P3_p06_v3 as P3v3  # noqa: E402
import strand5_lattice_star_v3 as S  # noqa: E402

mp.mp.dps = 40


def mp_Veff(ns, ks, x):
    qq = mp.mpf(x)
    phi = 2 * mp.fsum(mp.mpf(k) * mp.sin(qq * int(n) / 2) ** 2 for n, k in zip(ns, ks) if int(n) != 0)
    dphi = mp.fsum(mp.mpf(k) * int(n) * mp.sin(qq * int(n)) for n, k in zip(ns, ks))
    return float((2 * phi - qq * dphi) / (2 * phi * phi))


def V_eff(stencil, q, small=0.2):
    ns = np.array(sorted(stencil), dtype=float)
    ks = np.array([stencil[int(n)] for n in ns])
    nz = ns != 0
    q = np.asarray(q, float)
    out = np.empty_like(q)
    ib = np.nonzero(q >= small)[0]
    for s in range(0, len(ib), 2048):
        sel = ib[s:s + 2048]
        qb = q[sel]
        phi = 2 * (np.sin(np.outer(qb, ns[nz]) / 2) ** 2) @ ks[nz]
        dphi = np.sin(np.outer(qb, ns)) @ (ks * ns)
        out[sel] = (2 * phi - qb * dphi) / (2 * phi * phi)
    isml = np.nonzero(q < small)[0]
    if len(isml) and len(isml) <= 64:
        for i in isml:
            out[i] = mp_Veff(ns, ks, q[i])
    elif len(isml):
        nodes = np.geomspace(0.999 * q[isml].min(), 1.25 * small, 160)
        vk = np.array([mp_Veff(ns, ks, x) for x in nodes])
        out[isml] = CubicSpline(np.log(nodes), vk)(np.log(q[isml]))
    return out


def pis(stencil, kop, M, mem_list, prof, K, e1t):
    gM, gB = M["couplings"]
    out = []
    for d in mem_list:
        key = "z%.6f" % d["z_c"]
        r, psi, N = prof[key + "_r"], prof[key + "_psi"], prof[key + "_N"]
        star = S.Star(len(r), stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1t)
        eb, comb, nn, kap = star.matter(psi, N, d["mu_inf"])
        Pi = float(np.sum(4 * np.pi * r * r * psi ** 5 * eb * P3.apply_V(K, N * psi ** 5 * comb, r)))
        meas = d["lock"].get("vac_part_of_AmB_over_A")
        pred = -Pi / (math.pi * kop * d["A"])
        out.append((d["z_c"], Pi, pred, meas))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    f, table = P3.f, P3.table
    rec, rows, rows0, summ = {}, [], [], []
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
        mems = S2["members"]
        five = S2["meta"].get("five_members") or list(range(len(mems)))
        ml = [mems[i] for i in five]
        nK = 2 * max(int(d["J"]) for d in ml) + 4
        Kc, V0c, _, _ = P3.line_kernel_V(lambda q: P3v3.V_stencil3(stencil, q)[0], nK)
        Ke, V0e, _, _ = P3.line_kernel_V(lambda q: V_eff(stencil, q), nK)
        ns = np.array(sorted(stencil), dtype=float)
        ks = np.array([stencil[int(n)] for n in ns])
        ka0 = 0.5 * float(np.sum(ns ** 2 * ks))
        rows0.append([lab, "kernel", "%.8e" % V0c, "%.8e" % V0e, "%.8e" % (V0e - V0c), "%.8e" % (1 / (12 * ka0))])
        pc = pis(stencil, kop, M, ml, prof, Kc, e1t)
        pe = pis(stencil, kop, M, ml, prof, Ke, e1t)
        fam = []
        for (z, Pc, prc, ms), (_, Pe, pre, _) in zip(pc, pe):
            rows.append([lab, "kernel", "%.6g" % z, "%.6e" % Pc, "%.6e" % Pe, f(ms, 5), f(prc / ms, 6) if ms else "-", f(pre / ms, 6) if ms else "-"])
            fam.append({"z_c": z, "Pi_code": Pc, "Pi_eff": Pe, "pred_code": prc, "pred_eff": pre, "measured": ms})
        rec[lab] = {"kernel": fam, "V0_code": V0c, "V0_eff": V0e, "ka0": ka0}
        if twin_p != "-":
            T2 = json.load(open(twin_p))
            tfam_p = fam_p.replace("_kernel_", "_local_ka0_")
            if os.path.exists(tfam_p):
                TF = json.load(open(tfam_p))
                tprof = np.load(tfam_p.replace(".json", "_profiles.npz"))
                tst, tkop, _ = S.setup_kernel(types.SimpleNamespace(kernel=kv if kk == "--kernel" else None, packets=kv if kk == "--packets" else None,
                                                                    mlight=M["mlight"], aH=M["aH"], operator="local_ka0"))
                tm = T2["members"]
                tfive = T2["meta"].get("five_members") or list(range(len(tm)))
                tml = [tm[i] for i in tfive]
                nKt = 2 * max(int(d["J"]) for d in tml) + 4
                Kt, V0t, _, _ = P3.line_kernel_V(lambda q: V_eff(tst, q), nKt)
                rows0.append([lab, "local twin", "0", "%.8e" % V0t, "%.8e" % V0t, "%.8e" % (1 / (12 * tkop))])
                pt = pis(tst, tkop, TF["meta"], tml, tprof, Kt, e1t)
                tw = []
                for z, Pe, pre, ms in pt:
                    rows.append([lab, "local twin", "%.6g" % z, "0", "%.6e" % Pe, f(ms, 5), "-", f(pre / ms, 6) if ms else "-"])
                    tw.append({"z_c": z, "Pi_eff": Pe, "pred_eff": pre, "measured": ms})
                rec[lab]["local_twin"] = tw
        r_c = [x["pred_code"] / x["measured"] for x in fam if x["measured"]]
        r_e = [x["pred_eff"] / x["measured"] for x in fam if x["measured"]]
        summ.append("%s: prediction/measured with V_code %s; with V_eff %s." % (lab, ", ".join("%.4f" % x for x in r_c), ", ".join("%.4f" % x for x in r_e)))
    out = ["# Strand 5 production P3, the complete-symbol variant (strand5_P3_p06_v4.py; the red team's form, for the lead's ruling)", "",
           "## S Summary (emitted)", ""] + ["- " + x for x in summ] + [
        "", "## F1 V(0) per family: V_code, V_eff, their difference and 1/(12 k_a(0))", "",
        table(["family", "operator", "V_code(0)", "V_eff(0)", "difference", "1/(12 k_a(0))"], rows0), "",
        "## F2 Per member: Pi with each multiplier, stage 2's measured kernel part of (A - B)/A (for the twin: its own measured part), the ratios", "",
        table(["family", "operator", "z_c", "Pi (V_code)", "Pi (V_eff)", "measured", "prediction/measured (V_code)", "prediction/measured (V_eff)"], rows), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))


if __name__ == "__main__":
    main()
