#!/usr/bin/env python
"""n4_b_zero_v1.py (busi, foundations item N4; ReviewA_foundations_v1.md sec. 3.2 N4, Chain 4, P14 'The rest orientation';
ReviewB_foundations_v1.md sec. 2.2 'Important countercheck').

DATA script (box CPU: sympy for the exact parts, numpy on a grid for the explicit configuration; no eigensolver, no card).
Density of the paper, Eq. (dm-density), with a_3 = 0 (excluded on potential flows and irrelevant to b):
   A = N/(16 pi G) [a_1 (tr K)^2 + a_2 tr K^2 + b tr(Omega^T Omega) + c |grad N/N|^2],
   K = (grad v + grad v^T)/(2N), Omega = (grad v - grad v^T)/(2N), (grad v)_ij = d_i v_j;
laws rho = dA/dN (variational), J = dA/dv (variational).

Part S (sympy, exact):
 S1  J = dA/dv equals App. app:rows' printed form -(1/8 pi G) div[a_1 trK I + a_2 K + b Omega]  (difference simplified to 0);
 S2  Chain 4's lemma: the Euler-Lagrange expression of (d_1 v_2 - d_2 v_1)/N in v_2 is d_1 N/N^2 (not identically 0: no null Lagrangian);
 S3  the P14 configuration v = omega x x, N = N(x) arbitrary: K = 0 and J = (b/8 pi G) omega x grad N / N^2 exactly;
 S4  second route (D_s(omega x x) = 0 tested on a second configuration): for a generic N and a generic flow v (potential part plus a
     differential rotation), the change of the sources under v -> v + omega x x is
       delta rho = -(b/16 pi G)[2 tr(Omega^T Omega_w) + tr(Omega_w^T Omega_w)],  delta J = -(b/8 pi G) div Omega_w,  Omega_w = A_w/N,
     with the strain part unchanged; both vanish identically only for b = 0 (checked: the b = 0 difference simplifies to 0; the b = 1
     difference evaluated at sample points is nonzero);
 S5  reviewer B's countercheck: at N = 1, a_1 = -1, a_2 = 1, c = 0: 16 pi G A + div F = (1 + b) ||A_v||^2, F = v div v - (v.grad) v; the
     Euler-Lagrange expression of ||A_v||^2 in v is not identically zero, so the energy-flux clause on ALL flows would select b = -1.
Part N (numpy grid): P14 evaluated numerically. N = 1 - eta chi(|x - x0|/a_b) (chi the smooth compact bump), v = omega x x, a_1 = -1,
 a_2 = 1, c = 0; J from the printed momentum law by 4th-order central differences; L = int x x J d^3y over the grid against the closed
 form (b/4 pi G) omega int (1/N - 1) d^3y; at two spacings; b = 1 and b = 0; also int J d^3y (a divergence: 0).
Output: <outdir>/n4_b_zero_<tag>.json."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import sympy as S  # noqa: E402

VERSION = "n4_b_zero_v1"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ============================================================================ Part S: exact
def part_symbolic():
    x = S.symbols("x1:4", real=True)
    G = S.symbols("G", positive=True)
    a1, a2, b, c = S.symbols("a1 a2 b c", real=True)
    Nf = S.Function("N")(*x)
    vf = [S.Function("v%d" % (i + 1))(*x) for i in range(3)]
    gv = S.Matrix(3, 3, lambda i, j: S.diff(vf[j], x[i]))           # (grad v)_ij = d_i v_j
    K = (gv + gv.T) / (2 * Nf); Om = (gv - gv.T) / (2 * Nf)
    gN = S.Matrix([S.diff(Nf, xi) for xi in x])
    Adens = Nf / (16 * S.pi * G) * (a1 * K.trace() ** 2 + a2 * (K * K).trace() + b * (Om.T * Om).trace() + c * (gN.T * gN)[0] / Nf ** 2)
    out = {}
    # S1: J = dA/dv (A independent of v itself): J_j = -sum_i d_i (dA/d(d_i v_j))
    dsym = S.symbols("d0:9")
    sub = {S.diff(vf[j], x[i]): dsym[3 * i + j] for i in range(3) for j in range(3)}
    Ad = Adens.subs(sub)
    back = {dsym[3 * i + j]: S.diff(vf[j], x[i]) for i in range(3) for j in range(3)}
    J = []
    for j in range(3):
        Jj = 0
        for i in range(3):
            Jj += -S.diff(S.diff(Ad, dsym[3 * i + j]).subs(back), x[i])
        J.append(Jj)
    M = a1 * K.trace() * S.eye(3) + a2 * K + b * Om
    Jprint = [-sum(S.diff(M[i, j], x[i]) for i in range(3)) / (8 * S.pi * G) for j in range(3)]
    out["S1_J_minus_printed_simplifies_to_zero"] = [bool(S.simplify(S.expand(J[j] - Jprint[j])) == 0) for j in range(3)]
    # S2: EL of (d1 v2 - d2 v1)/N in v2
    Lag = (S.diff(vf[1], x[0]) - S.diff(vf[0], x[1])) / Nf
    el = S.euler_equations(Lag, [vf[1]], list(x))[0].lhs
    out["S2_EL_in_v2"] = str(S.simplify(el))
    out["S2_EL_minus_d1N_over_N2_is_zero"] = bool(S.simplify(el - S.diff(Nf, x[0]) / Nf ** 2) == 0)
    # S3: the P14 configuration
    w = S.symbols("w1:4", real=True)
    vrot = [w[1] * x[2] - w[2] * x[1], w[2] * x[0] - w[0] * x[2], w[0] * x[1] - w[1] * x[0]]   # omega x x
    subs_rot = {}
    Jrot = [S.simplify(Jprint[j].subs({vf[k]: vrot[k] for k in range(3)}).doit()) for j in range(3)]
    gNv = [S.diff(Nf, xi) for xi in x]
    wxgN = [w[1] * gNv[2] - w[2] * gNv[1], w[2] * gNv[0] - w[0] * gNv[2], w[0] * gNv[1] - w[1] * gNv[0]]
    target = [b / (8 * S.pi * G) * wxgN[j] / Nf ** 2 for j in range(3)]
    Krot = K.subs({vf[k]: vrot[k] for k in range(3)}).doit()
    out["S3_K_zero_for_rigid_rotation"] = bool(S.simplify(Krot) == S.zeros(3, 3))
    out["S3_J_equals_b_omega_x_gradN_over_8piG_N2"] = [bool(S.simplify(Jrot[j] - target[j]) == 0) for j in range(3)]
    # S4: shift v -> v + omega x x on a generic configuration; the source changes
    rho = S.diff(Ad.subs(back), Nf) if False else None
    # rho = dA/dN variational: A depends on N and grad N (c term); use the Euler operator
    Nsym, gsym = S.symbols("Nsym"), S.symbols("g1:4")
    subN = {S.diff(Nf, x[i]): gsym[i] for i in range(3)}
    A2 = Adens.subs(sub).subs(subN).subs(Nf, Nsym)
    backN = {Nsym: Nf}
    backN.update({gsym[i]: S.diff(Nf, x[i]) for i in range(3)})
    backN.update(back)
    rho_expr = S.diff(A2, Nsym).subs(backN) - sum(S.diff(S.diff(A2, gsym[i]).subs(backN), x[i]) for i in range(3))
    shift = {vf[k]: vf[k] + vrot[k] for k in range(3)}
    drho = S.expand((rho_expr.subs(shift).doit() - rho_expr))
    dJ = [S.expand(Jprint[j].subs(shift).doit() - Jprint[j]) for j in range(3)]
    out["S4_b0_source_change_zero"] = {"rho": bool(S.simplify(drho.subs(b, 0)) == 0), "J": [bool(S.simplify(dJ[j].subs(b, 0)) == 0) for j in range(3)]}
    # evaluate the b = 1 change on a concrete second configuration: N = 1 + 0.3 exp(-|x - x1|^2), v = grad phi + Om(r) z x x
    x1 = (0.4, -0.3, 0.2)
    Nc = 1 + S.Rational(3, 10) * S.exp(-sum((x[i] - x1[i]) ** 2 for i in range(3)))
    phi = S.Rational(1, 5) * S.exp(-(x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2) / 4)
    Omr = S.Rational(1, 2) * S.exp(-(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / 2)
    vc = [S.diff(phi, x[0]) - Omr * x[1], S.diff(phi, x[1]) + Omr * x[0], S.diff(phi, x[2])]
    conc = {Nf: Nc}
    conc.update({vf[k]: vc[k] for k in range(3)})
    vals = {w[0]: S.Rational(1, 10), w[1]: -S.Rational(1, 5), w[2]: S.Rational(3, 10), G: 1, a1: -1, a2: 1, c: 0}
    pts = [(0.1, 0.2, -0.3), (0.5, -0.4, 0.6), (-0.7, 0.3, 0.1)]
    ev = []
    for p in pts:
        sp_ = {x[i]: p[i] for i in range(3)}
        r_b1 = float(drho.subs(conc).doit().subs(vals).subs(b, 1).subs(sp_).evalf())
        J_b1 = [float(dJ[j].subs(conc).doit().subs(vals).subs(b, 1).subs(sp_).evalf()) for j in range(3)]
        r_b0 = float(drho.subs(conc).doit().subs(vals).subs(b, 0).subs(sp_).evalf())
        ev.append({"x": p, "delta_rho_b1": r_b1, "delta_J_b1": J_b1, "delta_rho_b0": r_b0})
    out["S4_second_configuration_b1"] = ev
    # the same, in closed form at b: delta rho for the uniform rotation alone (v = 0 before the shift, N = 1): -(b/8 pi G) |omega|^2
    rho_unif = S.simplify(rho_expr.subs({vf[k]: vrot[k] for k in range(3)}).doit().subs(Nf, 1).doit())
    out["S4_rigid_rotation_alone_rho"] = str(rho_unif)
    # S5: reviewer B's countercheck at N = 1
    sub1 = {Nf: 1}
    gv1 = S.Matrix(3, 3, lambda i, j: S.diff(vf[j], x[i]))
    Sm = (gv1 + gv1.T) / 2; Am = (gv1 - gv1.T) / 2
    A_N1 = (1 / (16 * S.pi * G)) * (-(Sm.trace()) ** 2 + (Sm * Sm).trace() + b * (Am.T * Am).trace())
    divv = sum(S.diff(vf[i], x[i]) for i in range(3))
    F = [vf[i] * divv - sum(vf[j] * S.diff(vf[i], x[j]) for j in range(3)) for i in range(3)]
    divF = sum(S.diff(F[i], x[i]) for i in range(3))
    A2n = (Am.T * Am).trace()
    out["S5_16piG_A_plus_divF_minus_(1+b)|A|^2_is_zero"] = bool(S.simplify(S.expand(16 * S.pi * G * A_N1 + divF - (1 + b) * A2n)) == 0)
    elA = [S.simplify(e_.lhs) for e_ in S.euler_equations(A2n, vf, list(x))]
    out["S5_EL_of_|A|^2_in_v"] = [str(e_) for e_ in elA]
    out["S5_EL_identically_zero"] = bool(all(S.simplify(e_) == 0 for e_ in elA))
    return out


# ============================================================================ Part N: the P14 configuration on a grid
def bump(s):
    out = np.zeros_like(s)
    m = s < 1.0
    out[m] = np.exp(1.0 - 1.0 / (1.0 - s[m] ** 2))
    return out


def d1(F, h, ax):
    """4th-order central difference along axis ax (periodic roll; the fields used are compactly supported or linear, and the
    boundary layer of 2 points is excluded from every integral)."""
    return (-np.roll(F, -2, ax) + 8 * np.roll(F, -1, ax) - 8 * np.roll(F, 1, ax) + np.roll(F, 2, ax)) / (12.0 * h)


def part_numeric(h, Lb, omega, x0, ab, eta, bval, G=1.0):
    g = np.arange(-Lb, Lb + 1e-9, h)
    X = np.meshgrid(g, g, g, indexing="ij")
    s = np.sqrt(sum((X[i] - x0[i]) ** 2 for i in range(3))) / ab
    N = 1.0 - eta * bump(s)
    v = [omega[1] * X[2] - omega[2] * X[1], omega[2] * X[0] - omega[0] * X[2], omega[0] * X[1] - omega[1] * X[0]]
    # grad v analytically constant for the rigid rotation (finite differences of a linear function are exact away from the roll seam)
    gv = [[d1(v[j], h, i) for j in range(3)] for i in range(3)]
    K = [[(gv[i][j] + gv[j][i]) / (2 * N) for j in range(3)] for i in range(3)]
    Om = [[(gv[i][j] - gv[j][i]) / (2 * N) for j in range(3)] for i in range(3)]
    trK = K[0][0] + K[1][1] + K[2][2]
    Mm = [[-(trK if i == j else 0.0) + K[i][j] + bval * Om[i][j] for j in range(3)] for i in range(3)]
    J = [-sum(d1(Mm[i][j], h, i) for i in range(3)) / (8 * np.pi * G) for j in range(3)]
    inner = np.ones_like(N, bool)
    for i in range(3):
        inner &= (np.abs(X[i]) < Lb - 4 * h)
    dV = h ** 3
    xJ = [X[1] * J[2] - X[2] * J[1], X[2] * J[0] - X[0] * J[2], X[0] * J[1] - X[1] * J[0]]
    L = np.array([float(np.sum(xJ[k][inner]) * dV) for k in range(3)])
    intf = float(np.sum((1.0 / N - 1.0)[inner]) * dV)
    Lpred = bval / (4 * np.pi * G) * np.array(omega) * intf
    Jtot = np.array([float(np.sum(J[k][inner]) * dV) for k in range(3)])
    # pointwise check of J against the closed form (b/8 pi G) omega x grad N / N^2 (S3)
    gN = [d1(N, h, i) for i in range(3)]
    wxg = [omega[1] * gN[2] - omega[2] * gN[1], omega[2] * gN[0] - omega[0] * gN[2], omega[0] * gN[1] - omega[1] * gN[0]]
    Jcf = [bval / (8 * np.pi * G) * wxg[k] / N ** 2 for k in range(3)]
    jerr = max(float(np.max(np.abs(J[k] - Jcf[k])[inner])) for k in range(3))
    jmax = max(float(np.max(np.abs(Jcf[k])[inner])) for k in range(3))
    # the energy density of the rotation itself, outside the bump (N = 1): rho = -(b/8 pi G) |omega|^2 from the b-term
    trOO = sum(Om[i][j] ** 2 for i in range(3) for j in range(3))
    rho_b = -(bval / (16 * np.pi * G)) * trOO           # rho's b part, -(b/16 pi G) tr(Omega^T Omega)
    far = inner & (s > 1.2)
    return {"h": h, "Lb": Lb, "b": bval, "L_numeric": L.tolist(), "L_closed_form": Lpred.tolist(),
            "L_rel_err": float(np.max(np.abs(L - Lpred)) / max(float(np.max(np.abs(Lpred))), 1e-300)) if bval != 0 else None,
            "L_abs_max": float(np.max(np.abs(L))), "int_1_over_N_minus_1": intf, "int_J": Jtot.tolist(),
            "J_minus_closed_form_max_over_max": jerr / max(jmax, 1e-300) if bval != 0 else jerr,
            "rho_b_part_outside_bump": float(np.mean(rho_b[far])), "rho_b_predicted": float(-(bval / (8 * np.pi * G)) * float(np.dot(omega, omega)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1"); ap.add_argument("--outdir", default="<internal>/n345")
    ap.add_argument("--skip-symbolic", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.outdir, "n4_b_zero_%s.json" % a.tag)
    if os.path.exists(out):
        sys.exit("refusing to overwrite " + out)
    t0 = time.time()
    res = {"version": VERSION, "script_md5": md5(os.path.abspath(__file__)), "sympy": S.__version__, "numpy": np.__version__}
    omega = (0.3, -0.2, 0.5); x0 = (0.7, -0.4, 0.25); ab = 1.5; eta = 0.3
    res["numeric_config"] = {"omega": omega, "x0": x0, "bump_radius": ab, "eta": eta, "N": "1 - eta exp(1 - 1/(1 - s^2)), s = |x - x0|/a_b < 1",
                             "v": "omega x x", "a1": -1, "a2": 1, "c": 0, "G": 1}
    res["numeric"] = []
    for h in (0.08, 0.04):
        for bval in (1.0, 0.0):
            r = part_numeric(h, 4.0, omega, x0, ab, eta, bval)
            print(json.dumps(r), flush=True)
            res["numeric"].append(r)
    if not a.skip_symbolic:
        res["symbolic"] = part_symbolic()
        print(json.dumps(res["symbolic"], indent=1), flush=True)
    res["elapsed_s"] = time.time() - t0
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
