#!/usr/bin/env python
"""Strand 6 (h) (SPEC_strand6_validation.md, final3 (h); builder analyst): the band-projected first-order vertex of the light sector on the
ring against the 2a addendum object, by GAP2_spinapparatus's map.

Ring (strand2a_rtchecks_CDE_v1.py's machinery): L sites, the one-component Weyl D (D|p> = d(p)|p>, d(p) = sin p), A = m psi^-1 + i psi^-3 D,
E = (A^dag A)^(1/2), U = A E^-1, B = E^(1/2) L E^(1/2) with L = N psi; the polar law's chiral Hamiltonian h_* = [[0, T^dag], [T, 0]] with the native
block T = U B (= A at the flat point); the comparator h_nat = [[0, A_nat^dag], [A_nat, 0]], A_nat = m + i psi^-1 D psi^-1 (the Hermitian minimal
placement, the mass unscaled).  The flat positive states of the polar law, v_p^+ = (|p>, U_p |p>)/sqrt2, U_p = (m + i d(p))/E_p, serve BOTH laws
(the common fixed flat polar frame).  Source psi = 1 + u, u_x = eps cos(q x), q = 2 pi n/L, so <p + q|u|p> = <p - q|u|p> = eps/2 =: u_q (every
element below is divided by u_q).  Valleys: nu = +1 at p = 0 and nu = -1 at p = pi (the valley's p' = p_0 + q, d(p') = nu sin q).
Convention: M(p', p) := <v_p'^+|delta h|v_p^+>/u_q (OUTGOING p', INCOMING p); the transposed element M^T := <v_p^+|delta h|v_p'^+>/u_q (2a's
part E tabulated this one) is given beside; b := Im M/(nu d(q)), c := Re M/d(q)^2 for every object.
Objects at (p, p') = (p_0, p_0 + q), N = 1:
  (1) M_law (P23's object; the law's h_* with T = U B, L = psi);  (2) M_nat (the comparator), against the closed form
      (i/2) nu d (1 - m/E_q) - d^2/(2 E_q);  (3) M_law - M_nat against GAP2's (31) [sqrt(m E) + E/2 - m/2 - m^2/E] + i nu d (1 + m/E);
  (4) delta T(q, 0) = <p'|delta(U B)|p>/u_q against m (f - 1) + i nu f d, f = (m/E_q)^(1/2), and beside it the raw lower-left block of the
      A-built chiral Hamiltonian <p'|delta A|p>/u_q;
  (5) Map B: delta B_inst = delta B_* - delta B_nat on (p', p), with B_* = E^(1/2) L E^(1/2) (L = psi) and B_nat = |A_nat|, against GAP2's (33)
      [sqrt(m E) + (E^2 - 3 m^2)/(E + m)] - i nu 4 m d/(E + m); beside it the reading B_* = |A| (no L), delta E_* - delta E_nat; and the frame
      relation (34) delta B_inst = (M_law - M_nat) + [E, Gamma_* - Gamma_nat] on the (p', p) element, Gamma = (1/2) U^dag delta U per law;
  (6) the backscattering ratio: with N = 1 - 2u, psi = 1 + u (n = -2s), the law's elastic element M_*(-p, p) at p = p_0 + q/2 (so p' - p = -q),
      |M_*|^2/|M_min|^2 with the comparator's element -2 m s (|M_min| = 2 m per unit u_q, the spec's [S] form), against 1 + d(p)^2/m^2.
Routes: (a) exact first order in the plane-wave basis (the flat operators are diagonal in p; delta A, delta A_nat, delta L couple p to p +- q; the
first-order E, E^(1/2), U by the Sylvester equations elementwise), for every (L, m, valley, q); (b) the dense ring with central differences in eps
(1e-5 and 1e-6) at L = 1024 (law_parts as strand 2a's), whose agreement with (a) is the check of (a).  Small-q extrapolations (H2): b and c
fitted as b_0 + b_2 d^2 over the three smallest q at L = 4096, against 2, 1/(2m) (law); 0, -1/(2m) (comparator); 2, 1/m (difference);
1, -1/(4m) (native block).  Metadata per row: a = 1, m, p, p', d(p), d(p'), u_q, the raw lower-left derivative and the reverse-block adjoint;
no source-dependent band transformation is differentiated (the frame is the flat one).  Numbers emitted, not retyped."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math

import numpy as np


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def analytic(m, p, pp, kind="psi"):
    """Exact first-order elements between the plane waves p (in) and pp (out), per unit u_q.
    kind 'psi': psi = 1 + u, N = 1 (delta L = u); kind 'weak': psi = 1 + u, N = 1 - 2u (delta L = -u)."""
    s, sp_ = math.sin(p), math.sin(pp)
    A = lambda k: m + 1j * math.sin(k)
    E = lambda k: math.sqrt(m * m + math.sin(k) ** 2)
    U = lambda k: A(k) / E(k)
    dL = 1.0 if kind == "psi" else -1.0
    # delta A_{k' k} = (-m - 3 i sin k) u ; delta A_nat_{k' k} = -i (sin k + sin k') u ; delta L_{k' k} = dL u
    dA = lambda kp, k: -m - 3j * math.sin(k)
    dAn = lambda kp, k: -1j * (math.sin(k) + math.sin(kp))
    # law: E^2 = A^dag A; delta(E^2)_{kp k} = conj(dA_{k kp}) A_k + conj(A_kp) dA_{kp k}
    def dE2(kp, k, dAf):
        return np.conj(dAf(k, kp)) * A(k) + np.conj(A(kp)) * dAf(kp, k)

    def dE(kp, k, dAf):
        return dE2(kp, k, dAf) / (E(kp) + E(k))

    def dEh(kp, k, dAf):
        return dE(kp, k, dAf) / (math.sqrt(E(kp)) + math.sqrt(E(k)))

    def dU(kp, k, dAf):
        # U = A E^-1: dU = dA E^-1 - A E^-1 dE E^-1
        return dAf(kp, k) / E(k) - U(kp) * dE(kp, k, dAf) / E(k)

    def dT(kp, k):
        # T = U B = A E^(-1/2) L E^(1/2): dT = dA + A_kp E_kp^-1/2 dL E_k^1/2 (the E^(+-1/2) variations cancel at L = 1)
        return dA(kp, k) + A(kp) * math.sqrt(E(k) / E(kp)) * dL

    def vertex(dX):
        # <v_pp|delta h|v_p> = (1/2) [ (dX^dag)_{pp p} U_p + conj(U_pp) dX_{pp p} ], (dX^dag)_{pp p} = conj(dX_{p pp})
        return 0.5 * (np.conj(dX(p, pp)) * U(p) + np.conj(U(pp)) * dX(pp, p))

    def vertexT(dX):
        return 0.5 * (np.conj(dX(pp, p)) * U(pp) + np.conj(U(p)) * dX(p, pp))
    Ml = vertex(dT)
    Mn = vertex(dAn)
    out = {"M_law": Ml, "M_nat": Mn, "MT_law": vertexT(dT), "MT_nat": vertexT(dAn), "dT": dT(pp, p), "dT_adj_rev": np.conj(dT(p, pp)), "dA_raw": dA(pp, p),
           "E_p": E(p), "E_pp": E(pp)}
    # Map B: B_* = E^(1/2) L E^(1/2): dB = dEh E^(1/2) + E^(1/2) dEh + E^(1/2) dL E^(1/2)
    dBl = dEh(pp, p, dA) * (math.sqrt(E(p)) + math.sqrt(E(pp))) + math.sqrt(E(pp) * E(p)) * dL
    dBn = dE(pp, p, dAn)
    out["dB_inst"] = dBl - dBn
    out["dE_diff"] = dE(pp, p, dA) - dBn
    G_l = 0.5 * np.conj(U(pp)) * dU(pp, p, dA)
    G_n = 0.5 * np.conj(U(pp)) * dU(pp, p, dAn)
    out["frame34_residual"] = out["dB_inst"] - ((Ml - Mn) + (E(pp) - E(p)) * (G_l - G_n))
    return out


def ring_dense(L, m, eps, n, kind="psi"):
    """the dense ring's first-order elements by central differences (strand 2a's law_parts): returns a function giving M(pp, p) per unit u_q."""
    x = np.arange(L)
    D = np.zeros((L, L), dtype=complex)
    for i in range(L):
        D[i, (i + 1) % L] += -0.5j
        D[i, (i - 1) % L] += 0.5j
    u = np.cos(2 * np.pi * n * x / L)

    def parts(e):
        psi = 1 + e * u
        N = np.ones(L) if kind == "psi" else 1 - 2 * e * u
        A = m * np.diag(1 / psi) + 1j * (psi ** -3)[:, None] * D
        w, V = np.linalg.eigh(0.5 * (A.conj().T @ A + (A.conj().T @ A).conj().T))
        Eh = (V * w ** 0.25) @ V.conj().T
        Eihalf = (V * w ** -0.25) @ V.conj().T
        T = A @ Eihalf @ ((N * psi)[:, None] * Eh)
        An = m * np.eye(L) + 1j * np.diag(1 / psi) @ D @ np.diag(1 / psi)
        return T, An
    Tp, Anp = parts(eps)
    Tm, Anm = parts(-eps)
    dT = (Tp - Tm) / (2 * eps)
    dAn = (Anp - Anm) / (2 * eps)
    uq = 0.5

    def pw(k):
        return np.exp(1j * k * x) / math.sqrt(L)

    def elem(dX, kp, k):
        return pw(kp).conj() @ dX @ pw(k)

    def Uk(k):
        return (m + 1j * math.sin(k)) / math.sqrt(m * m + math.sin(k) ** 2)

    def vertex(dX, pp, p):
        return 0.5 * (np.conj(elem(dX, p, pp)) * Uk(p) + np.conj(Uk(pp)) * elem(dX, pp, p)) / uq
    return lambda pp, p: {"M_law": vertex(dT, pp, p), "M_nat": vertex(dAn, pp, p), "dT": elem(dT, pp, p) / uq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--dense-L", type=int, default=1024)
    a = ap.parse_args()
    Ls, ms, ns = (1024, 4096), (0.1, 0.2, 0.125), (1, 2, 4, 8, 16)
    rows, meta_rows, dense_rows, rec, summ = [], [], [], [], []
    for L in Ls:
        for m in ms:
            for p0, nu in ((0.0, 1), (math.pi, -1)):
                for n in ns:
                    q = 2 * math.pi * n / L
                    p, pp = p0, p0 + q
                    d = nu * math.sin(pp)
                    o = analytic(m, p, pp)
                    E = o["E_pp"]
                    cl_nat = 0.5j * nu * d * (1 - m / E) - d * d / (2 * E)
                    cl_31 = (math.sqrt(m * E) + E / 2 - m / 2 - m * m / E) + 1j * nu * d * (1 + m / E)
                    fq = math.sqrt(m / E)
                    cl_T = m * (fq - 1) + 1j * nu * fq * d
                    cl_33 = (math.sqrt(m * E) + (E * E - 3 * m * m) / (E + m)) - 1j * nu * 4 * m * d / (E + m)
                    pb = p0 + q / 2
                    ow = analytic(m, pb, p0 - q / 2, kind="weak")
                    db = math.sin(pb) if nu > 0 else -math.sin(pb - math.pi)
                    ratio = abs(ow["M_law"]) ** 2 / (2 * m) ** 2
                    bc = lambda z: (z.imag / (nu * d), z.real / d ** 2)
                    bl, cl_ = bc(o["M_law"])
                    bn, cn = bc(o["M_nat"])
                    bdf, cdf = bc(o["M_law"] - o["M_nat"])
                    bt, ct = bc(o["dT"])
                    bB, cB = bc(o["dB_inst"])
                    r = {"L": L, "m": m, "nu": nu, "n": n, "q": q, "q_over_m": q / m, "b_law": bl, "c_law": cl_, "b_nat": bn, "c_nat": cn, "b_diff": bdf, "c_diff": cdf,
                         "b_T": bt, "c_T": ct, "b_B": bB, "c_B": cB, "nat_vs_closed": abs(o["M_nat"] - cl_nat) / abs(cl_nat), "diff_vs_31": abs(o["M_law"] - o["M_nat"] - cl_31) / abs(cl_31),
                         "T_vs_closed": abs(o["dT"] - cl_T) / abs(cl_T), "B_vs_33": abs(o["dB_inst"] - cl_33) / abs(cl_33),
                         "B_vs_33_conj": abs(o["dB_inst"] - np.conj(cl_33 - cl_33.real) - cl_33.real) / abs(cl_33),
                         "dE_vs_33": abs(o["dE_diff"] - cl_33) / abs(cl_33), "frame34": abs(o["frame34_residual"]) / max(abs(o["dB_inst"]), 1e-300),
                         "back_ratio": ratio, "back_pred": 1 + db * db / (m * m), "MT_b": (o["MT_law"].imag / (nu * d)), "MT_c": o["MT_law"].real / d ** 2}
                    rec.append(r)
                    rows.append([L, m, nu, n, "%.4g" % (q / m), "%.6f" % bl, "%.6f" % (cl_ * m), "%.6f" % bn, "%.6f" % (cn * m), "%.6f" % bdf, "%.6f" % (cdf * m),
                                 "%.6f" % bt, "%.6f" % (ct * m), "%.6f" % bB, "%.6f" % (cB * m), f(r["frame34"], 2), "%.8f" % ratio, "%.8f" % r["back_pred"],
                                 "%.6f" % r["MT_b"], "%.6f" % (r["MT_c"] * m)])
                    if L == Ls[0] and m == ms[0] and n in (1, 16):
                        meta_rows.append([L, m, nu, "%.6f" % p, "%.6f" % pp, "%.6e" % math.sin(p), "%.6e" % math.sin(pp), "eps/2",
                                          "%.6e%+.6ei" % (o["dT"].real, o["dT"].imag), "%.6e%+.6ei" % (o["dT_adj_rev"].real, o["dT_adj_rev"].imag), "no"])
    # the closed forms' agreement (exact first order: the route checks itself against the closed forms)
    mx = {k: max(r_[k] for r_ in rec) for k in ("nat_vs_closed", "diff_vs_31", "T_vs_closed", "dE_vs_33", "frame34")}
    mx33 = min(max(r_["B_vs_33"] for r_ in rec), max(r_["B_vs_33_conj"] for r_ in rec))
    which33 = "as printed" if max(r_["B_vs_33"] for r_ in rec) <= max(r_["B_vs_33_conj"] for r_ in rec) else "with the odd term's sign reversed"
    summ.append("H1 (route (a), exact first order, %d rows): M_nat against its closed form to %s; M_law - M_nat against GAP2's (31) to %s; delta T against "
                "m (f - 1) + i nu f d to %s; delta B_inst (B_* = E^(1/2) psi E^(1/2), B_nat = |A_nat|) against (33) to %s (%s), while the reading B_* = |A| "
                "(delta E_* - delta E_nat) departs from (33) by up to %s; the frame relation (34) holds to %s." % (
                    len(rec), f(mx["nat_vs_closed"], 2), f(mx["diff_vs_31"], 2), f(mx["T_vs_closed"], 2), f(mx33, 2), which33, f(mx["dE_vs_33"], 2), f(mx["frame34"], 2)))
    # (b) the dense ring at L = 1024
    for m in ms:
        for n in (1, 4, 16):
            res = {}
            for eps in (1e-5, 1e-6):
                g = ring_dense(a.dense_L, m, eps, n)
                for p0, nu in ((0.0, 1), (math.pi, -1)):
                    q = 2 * math.pi * n / a.dense_L
                    res[(eps, nu)] = (g(p0 + q, p0), analytic(m, p0, p0 + q))
            for nu in (1, -1):
                dn, an = res[(1e-6, nu)]
                d5, _ = res[(1e-5, nu)]
                dense_rows.append([a.dense_L, m, nu, n, f(abs(dn["M_law"] - an["M_law"]) / abs(an["M_law"]), 2), f(abs(dn["M_nat"] - an["M_nat"]) / abs(an["M_nat"]), 2),
                                   f(abs(dn["dT"] - an["dT"]) / abs(an["dT"]), 2), f(abs(d5["M_law"] - dn["M_law"]) / abs(dn["M_law"]), 2)])
    worst = max(max(float(r_[k]) for r_ in dense_rows) for k in (4, 5, 6))
    summ.append("Route (b), the dense ring at L = %d (central differences, eps = 1e-6, the 1e-5 step beside): the largest relative difference from route (a) "
                "over M_law, M_nat and delta T is %s; the two steps agree to %s." % (a.dense_L, f(worst, 2), f(max(float(r_[7]) for r_ in dense_rows), 2)))
    # H2 small-q extrapolations at L = 4096 over the three smallest q
    h2 = []
    for m in ms:
        for nu in (1, -1):
            sel = sorted([r_ for r_ in rec if r_["L"] == 4096 and r_["m"] == m and r_["nu"] == nu], key=lambda r_: r_["n"])[:3]
            d2 = np.array([math.sin(r_["q"]) ** 2 for r_ in sel])
            row = [m, nu]
            for kb, kc, tb, tc in (("b_law", "c_law", 2.0, 1 / (2 * m)), ("b_nat", "c_nat", 0.0, -1 / (2 * m)), ("b_diff", "c_diff", 2.0, 1 / m), ("b_T", "c_T", 1.0, -1 / (4 * m))):
                b0 = np.polyfit(d2, [r_[kb] for r_ in sel], 1)[1]
                c0 = np.polyfit(d2, [r_[kc] for r_ in sel], 1)[1]
                row += ["%.6f (%s)" % (b0, f(b0 - tb, 2)), "%.6f (%s)" % (c0 * m, f(c0 / tc - 1, 2))]
            h2.append(row)
    summ.append("H2 at L = 4096 (three smallest q; q/m from %s): the small-q extrapolations against (2, 1/(2m)), (0, -1/(2m)), (2, 1/m), (1, -1/(4m)) in table H2 "
                "(b with its difference, c times m with its relative difference)." % ", ".join(sorted({"%.3g" % r_["q_over_m"] for r_ in rec if r_["L"] == 4096 and r_["n"] == 1})))
    back = max(abs(r_["back_ratio"] / r_["back_pred"] - 1) for r_ in rec)
    summ.append("(6): the backscattering ratio |M_*(-p, p)|^2/(2 m)^2 against 1 + d(p)^2/m^2 over every row: largest relative difference %s." % f(back, 2))
    out = ["# Strand 6 (h): the band-projected first-order vertex (generated by solve/code/strand6/strand6h_vertex_v1.py)", "", "## S Summary (emitted)", ""]
    out += ["- " + x for x in summ] + [""]
    out += ["## H1 Per L, m, valley nu and n (q = 2 pi n/L): b and c (c times m) of the law (M_law), the comparator (M_nat), their difference, the native block "
            "delta T and Map B's delta B_inst; the frame relation's residual; the backscattering ratio and its prediction; the transposed element's b and c m", "",
            table(["L", "m", "nu", "n", "q/m", "b_law", "c_law m", "b_nat", "c_nat m", "b_diff", "c_diff m", "b_T", "c_T m", "b_B", "c_B m", "(34) residual",
                   "back ratio", "1 + d^2/m^2", "b (transposed)", "c m (transposed)"], rows), "",
            "## H2 The small-q extrapolations at L = 4096 (fits b_0 + b_2 d^2 over the three smallest q): value (difference from the closed form) for b; value "
            "times m (relative difference) for c", "",
            table(["m", "nu", "b_law", "c_law m", "b_nat", "c_nat m", "b_diff", "c_diff m", "b_T", "c_T m"], h2), "",
            "## H3 Route (b): the dense ring against route (a) (relative differences) and the two steps", "",
            table(["L", "m", "nu", "n", "M_law", "M_nat", "delta T", "step 1e-5 vs 1e-6"], dense_rows), "",
            "## H4 Metadata (a = 1; L = 1024, m = 0.1, n = 1 and 16): p, p', d(p), d(p'), the source amplitude, the raw lower-left derivative delta T(p', p) and "
            "the reverse-block adjoint conj(delta T(p, p')), whether a source-dependent band transformation was differentiated", "",
            table(["L", "m", "nu", "p", "p'", "d(p)", "d(p')", "u_q", "delta T(p', p)", "conj(delta T(p, p'))", "band transformation differentiated"], meta_rows), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
