#!/usr/bin/env python
"""Strand 5 production P3 (SPEC_strand5_covariant_row.md sec. 3, P3; builder analyst): the frozen kernel's lock defect against P06's theorem.

P06 (FRESH1216 sec. 7), in the lane's conventions (LEAD.tex "The lane under final3"): for the stationary branch of the frozen quadratic kernel,
pi k (B - A) = <f, V(-Lap) g>, V(t) = -kappa'(t)/kappa(t)^2 with kappa(q^2) := k_a(q), i.e. V(q) = -(dk_a/dq)/(2 q k_a(q)^2), with the clock-row
source f = psi^5 e_b and the combined (ruler) row source g = N psi^5 (e_0 - 2 T_0 + e_1 + 6 P_1) exactly as strand5_lattice_star_v3.py's rows use
them (Star.matter: eb, comb); the prediction for the kernel row's own part of (A - B)/A is -Pi/(pi k_op A), Pi = sum_j 4 pi r_j^2 f_j (V g)_j.
The measurement: stage 2's lock pieces (strand5_J4_stage2_v9/v10's record, "vac_part_of_AmB_over_A" = -(D_vac + S_inf)/(2 pi k A)) at the
family's five S3b members.

The operator V(-Lap) on the radial grid r_j = j (lattice units): (V g)_j = (1/r_j) sum_i (K^V_{j-i} - K^V_{j+i}) u_i, u_i = r_i g_i (the 3D radial
action of an isotropic symbol from its line kernel, as strand5_kernel_v3.py's radial convolution operator, item (4)), with the line kernel
K^V_n = (1/pi) int_0^pi V(q) cos(q n) dq evaluated by the discrete transform on a ring of Lr points (Lr = 2^18; the sharp cut at q = pi as the
Green function of J2, item (3)).  Two routes to V(q):
  S (primary): the family's own stencil K_n (the line kernel the rows use; strand5_lattice_star_v3.setup_kernel with the family's kernel source):
     C(q) = sum_n K_n cos(q n), k_a(q) = -C(q)/q_lat^2, q_lat^2 = 2 (1 - cos q), dk_a/dq analytically from the cosine series; below q = 0.2 in
     40-digit arithmetic (mpmath; the cancellation of C(q) against q_lat^2);
  T (the spec's): the [100] table where the family has one (the 2c table at (0.1, 1) and (0.1, 0.5), the zone-edge entry halved; J2b's
     kernel_J4_L256.json at (0.03, 0.3)), k_a by a cubic spline of the table and its derivative, below the first tabulated q the fit
     k_0 + k_2 q^2 + k_4 q^4 to the first five points (T1) or with the fit taking over below the third point (T2).
Reported per member: Pi (route S; routes T1, T2 beside where a table exists), the prediction -Pi/(pi k_op A), the measured kernel part of
(A - B)/A, their ratio; Pi with the continuum-form g = N psi^5 (e_b + 6 P_b), P_b = n mu_inf/N - e_b (the TF pressure) beside.
Per family: the sign of V(q) on q = 2 pi n/256 (n = 1..128) and the first q where k_a(q) rises, if any; the radial kernel
g_V(r) = (1/(2 pi^2 r)) int_0^pi q sin(q r) V(q) dq at r = 1..40 (its sign changes); the checks ((V - V(0)) 1)_j and r_j ((V - V(0)) (1/r))_j
beyond the stencil range (expected 0) and the local twin's V = 0 with stage 2's local part of (A - B)/A (consistency).
Numbers emitted, not retyped."""
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
import strand5_lattice_star_v3 as S  # noqa: E402

LR = 2 ** 18
mp.mp.dps = 40


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def mp_V(ns, ks, x):
    qq = mp.mpf(x)
    Cm = mp.fsum(mp.mpf(k) * mp.cos(qq * int(n)) for n, k in zip(ns, ks))
    Cpm = -mp.fsum(mp.mpf(k) * int(n) * mp.sin(qq * int(n)) for n, k in zip(ns, ks))
    l2 = 2 * (1 - mp.cos(qq))
    kam = -Cm / l2
    dkam = -(Cpm * l2 - Cm * 2 * mp.sin(qq)) / l2 ** 2
    return float(-dkam / (2 * qq * kam * kam)), float(kam)


def V_stencil(stencil, q, small=0.2):
    """V(q) = -(dk_a/dq)/(2 q k_a^2) from the cosine series of the stencil, q > 0 (array): float64 in chunks for q >= small; below it 40-digit
    values (directly for up to 64 points, else on 160 log-spaced nodes with a cubic spline in ln q)."""
    ns = np.array(sorted(stencil), dtype=float)
    ks = np.array([stencil[int(n)] for n in ns])
    q = np.asarray(q, float)
    out = np.empty_like(q)
    ka_out = np.empty_like(q)
    ib = np.nonzero(q >= small)[0]
    for s in range(0, len(ib), 2048):
        sel = ib[s:s + 2048]
        qb = q[sel]
        C = np.cos(np.outer(qb, ns)) @ ks
        Cp = -(np.sin(np.outer(qb, ns)) @ (ks * ns))
        ql2 = 2 * (1 - np.cos(qb))
        ka = -C / ql2
        dka = -(Cp * ql2 - C * 2 * np.sin(qb)) / ql2 ** 2
        out[sel] = -dka / (2 * qb * ka * ka)
        ka_out[sel] = ka
    isml = np.nonzero(q < small)[0]
    if len(isml) and len(isml) <= 64:
        for i in isml:
            out[i], ka_out[i] = mp_V(ns, ks, q[i])
    elif len(isml):
        nodes = np.geomspace(0.999 * q[isml].min(), 1.25 * small, 160)
        vk = np.array([mp_V(ns, ks, x) for x in nodes])
        out[isml] = CubicSpline(np.log(nodes), vk[:, 0])(np.log(q[isml]))
        ka_out[isml] = CubicSpline(np.log(nodes), vk[:, 1])(np.log(q[isml]))
    return out, ka_out


def V_table(qt, kt, q, nfit_below=1):
    """V(q) from a cubic spline of the [100] table (qt ascending, kt), the fit k0 + k2 q^2 + k4 q^4 to the first five points below qt[nfit_below - 1]."""
    sp = CubicSpline(qt, kt)
    X = np.stack([np.ones(5), qt[:5] ** 2, qt[:5] ** 4], 1)
    c = np.linalg.lstsq(X, kt[:5], rcond=None)[0]
    q = np.asarray(q, float)
    lo = q < qt[nfit_below - 1]
    ka = np.where(lo, c[0] + c[1] * q * q + c[2] * q ** 4, sp(np.clip(q, qt[0], qt[-1])))
    dka = np.where(lo, 2 * c[1] * q + 4 * c[2] * q ** 3, sp(np.clip(q, qt[0], qt[-1]), 1))
    return -dka / (2 * q * ka * ka), c


def line_kernel_V(Vfun, nmax):
    """K^V_n = (1/pi) int_0^pi V(q) cos(q n) dq by the ring transform of LR points (V even; V(0) by the limit from the smallest q)."""
    m = np.arange(LR // 2 + 1)
    q = 2 * np.pi * m / LR
    V = np.empty(len(q))
    V[1:] = Vfun(q[1:])
    V[0] = V[1]
    full = np.concatenate([V, V[-2:0:-1]])
    K = np.real(np.fft.ifft(full))
    return K[:nmax + 1], float(V[1]), q, V


def apply_V(K, g, r):
    """(V g)_j = (1/r_j) sum_i (K_{|j-i|} - K_{j+i}) u_i on the nodes r = 1..J."""
    J = len(r)
    u = r * g
    idx = np.arange(1, J + 1)
    out = np.zeros(J)
    nz = np.nonzero(u)[0]
    for s in range(0, len(nz), 512):
        ii = nz[s:s + 512] + 1
        D = K[np.abs(idx[:, None] - ii[None, :])] - K[idx[:, None] + ii[None, :]]
        out += D @ u[ii - 1]
    return out / r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True, help="semicolon-separated: label|stage1 family json|stage-2 json|kernel arg (--kernel P or --packets P)|table json or -|local twin stage-2 json or -")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    out = ["# Strand 5 production P3: the frozen kernel's lock defect against P06's theorem (generated by solve/code/strand5/strand5_P3_p06_v1.py)", ""]
    summ, rows, frows, crows, rec = [], [], [], [], {}
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
        if abs(kop / M["k_op"] - 1) > 1e-12:
            raise SystemExit("k_op mismatch for %s" % lab)
        gM, gB = M["couplings"]
        e1t = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
        Rst = max(stencil)
        # V by route S and its line kernel
        mems = S2["members"]
        five = S2["meta"].get("five_members") or list(range(len(mems)))
        Jmax = max(int(mems[i]["J"]) for i in five)
        Jc = 4 * Rst + 400
        nK = 2 * max(Jmax, Jc) + 2
        KS, V0S, qg, VgS = line_kernel_V(lambda q: V_stencil(stencil, q)[0], nK)
        q256 = 2 * np.pi * np.arange(1, 129) / 256
        V256, ka256 = V_stencil(stencil, q256)
        rises = [float(q256[i + 1]) for i in range(len(q256) - 1) if ka256[i + 1] > ka256[i]]
        rgv = np.arange(1, 41, dtype=float)
        qq = np.linspace(1e-6, np.pi, 20001)
        Vq = V_stencil(stencil, qq)[0]
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        gV = np.array([trap(qq * np.sin(qq * rr_) * Vq, qq) / (2 * np.pi ** 2 * rr_) for rr_ in rgv])
        sgn_changes = int(np.sum(np.diff(np.sign(gV)) != 0))
        # route T
        KT = {}
        if tab_p != "-":
            D = json.load(open(tab_p))
            rws = sorted([r_ for r_ in D["equal"] if abs(r_["aH"] - M["aH"]) < 1e-12 and abs(r_.get("m_light", 0.1) - M["mlight"]) < 1e-12], key=lambda r_: r_["n"])
            qt = np.array([r_["q"] for r_ in rws])
            kt = np.array([r_["k_a"] for r_ in rws])
            if abs(qt[-1] - np.pi) < 1e-12:
                kt = kt.copy()
                kt[-1] *= 0.5
            for nm, nb in (("T1", 1), ("T2", 3)):
                KT[nm] = line_kernel_V(lambda q, nb=nb: V_table(qt, kt, q, nb)[0], nK)[0]
        # checks on the operator: ((V - V0) 1) and r ((V - V0)(1/r)) beyond the range
        rc = np.arange(1, Jc + 1, dtype=float)
        flat = apply_V(KS, np.ones(Jc), rc) - V0S
        coul = rc * (apply_V(KS, 1.0 / rc, rc) - V0S / rc)
        sel = (rc > 2 * Rst) & (rc < Jc - 2 * Rst)
        crows.append([lab, "%.6e" % V0S, f(np.max(np.abs(flat[sel])) / abs(V0S), 2), f(np.max(np.abs(coul[sel])) / abs(V0S), 2), Rst,
                      "+" if np.all(V256 > 0) else ("-" if np.all(V256 < 0) else "mixed"), ("%.4g" % rises[0]) if rises else "none", sgn_changes,
                      " ".join("%.2e" % x for x in gV[:5])])
        # per member
        for i in five:
            d = mems[i]
            key = "z%.6f" % d["z_c"]
            r, psi, N, n = prof[key + "_r"], prof[key + "_psi"], prof[key + "_N"], prof[key + "_n"]
            J = len(r)
            star = S.Star(J, stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1t)
            eb, comb, nn, kap = star.matter(psi, N, d["mu_inf"])
            fsrc = psi ** 5 * eb
            gsrc = N * psi ** 5 * comb
            Pb = np.where(nn > 0, nn * d["mu_inf"] / N - eb, 0.0)
            gcont = N * psi ** 5 * (eb + 6 * Pb)
            Vg = apply_V(KS, gsrc, r)
            Pi = float(np.sum(4 * np.pi * r * r * fsrc * Vg))
            Pic = float(np.sum(4 * np.pi * r * r * fsrc * apply_V(KS, gcont, r)))
            pred = -Pi / (math.pi * kop * d["A"])
            meas = d["lock"].get("vac_part_of_AmB_over_A")
            PiT = {nm: float(np.sum(4 * np.pi * r * r * fsrc * apply_V(K_, gsrc, r))) for nm, K_ in KT.items()}
            rows.append([lab, "%.6g" % d["z_c"], "%.6e" % Pi, "%.6e" % pred, f(meas, 5), f(pred / meas, 6) if meas else "-",
                         f(PiT["T1"] / Pi - 1, 2) if "T1" in PiT else "-", f(PiT["T2"] / Pi - 1, 2) if "T2" in PiT else "-", f(Pic / Pi - 1, 3)])
            rec.setdefault(lab, []).append({"z_c": d["z_c"], "Pi": Pi, "prediction": pred, "measured": meas, "Pi_T": PiT, "Pi_continuum_g": Pic})
        # the local twin's consistency
        if twin_p != "-":
            T2 = json.load(open(twin_p))
            vals = [T2["members"][i]["lock"].get("vac_part_of_AmB_over_A") for i in five if i < len(T2["members"])]
            frows.append([lab, os.path.basename(twin_p), " ".join(f(v_, 2) for v_ in vals)])
        ratios = [float(r_[5]) for r_ in rows if r_[0] == lab and r_[5] != "-"]
        summ.append("%s: prediction/measured at the five members %s; V(0) = %s, V(q) on the 256-zone %s, k_a first rises at q = %s; g_V(r) sign changes over r = 1..40: %d." % (
            lab, ", ".join("%.5f" % x for x in ratios), "%.5e" % V0S, crows[-1][5], crows[-1][6], sgn_changes))
    out += ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""]
    out += ["## P3a Per member: Pi (route S), the prediction -Pi/(pi k_op A), stage 2's measured kernel part of (A - B)/A, their ratio, routes T1 and T2's "
            "Pi relative to S, and Pi with the continuum-form g relative to S", "",
            table(["family", "z_c", "Pi", "prediction", "measured", "prediction/measured", "Pi_T1/Pi_S - 1", "Pi_T2/Pi_S - 1", "Pi(continuum g)/Pi - 1"], rows), "",
            "## P3b The operator per family: V(0), the checks ((V - V(0)) 1) and r ((V - V(0))(1/r)) beyond twice the stencil range (relative to V(0)), the "
            "stencil range, V's sign on the 256-zone, the first q where k_a rises, g_V's sign changes over r = 1..40 and g_V(1..5)", "",
            table(["family", "V(0)", "flat check", "Coulomb check", "stencil range", "sign of V", "k_a rises at", "g_V sign changes", "g_V(1..5)"], crows), "",
            "## P3c The local twins: stage 2's kernel part of (A - B)/A at the same members (V = 0 identically for a local stencil)", "",
            table(["family", "local twin record", "kernel part of (A - B)/A"], frows), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
