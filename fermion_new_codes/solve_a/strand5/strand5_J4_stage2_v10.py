#!/usr/bin/env python
"""Strand 5 J4 stage 2, version 9 (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(m)(5'), (6); builder analyst): the fixed-N_B
radial Hessian of M[n_c] = 4 pi k_op A[n_c] by the exact second variation, with version 8's lock pieces, kappa_max and EOS-series
record per member (version 8's functions imported unchanged).

The map: psi[n_c] and A from stage 2's clock row G(x; n_c) = 0, x = (psi_1..J, A): G_j = [C r(psi - 1)]_j + tail_j A + loc_j(psi_j, n_c,j)
(j = 1..J), loc = psi^5 e_b(n_c psi^-6, psi^2), and G_J+1 = r_J (psi_J - 1) - A (ClockSolver of version 7).  The operator part and the
last row are linear in x, so the second variation comes from the local term alone, per node.  With G_x the Jacobian (the operator
plus diag(loc_psi), the tail column and the last row; one sparse LU), lambda = G_x^-T c, c = 4 pi k_op e_A, the gradient
dM/dn_c,j = -lambda_j loc_n,j and, for modes v (columns of Q), x' = -G_x^-1 [loc_n v; 0], psi' its first J rows,
H = -[psi'^T diag(lambda loc_pp) psi' + psi'^T diag(lambda loc_pn) Q + Q^T diag(lambda loc_pn) psi' + Q^T diag(lambda loc_nn) Q].
The per-node derivatives loc_p, loc_n, loc_pp, loc_pn, loc_nn are central differences in (psi_j, n_c,j) with relative steps 1e-3 and
5e-4, Richardson (4 D(s/2) - D(s))/3 (n derivatives only where n_c > 0).  The state is the clock row re-solved at the member's n_c.
(i) Basis (c): w = T_i(2x - 1), i = 0..12, x = (r/r_s)^2, on the nodes r < r_s with n_c > 0; v = n_c w projected onto sum 4 pi r^2 v = 0
(12 raw modes); the generalized eigenproblem H v = omega G v with G the volume Gram matrix (sum 4 pi r^2 v_k v_l): G's
eigendecomposition, directions below 1e-10 of its largest eigenvalue dropped (the count kept reported), the reduced H diagonalized; at
every member.  Beside it the exact Hessian in basis (a) (version 7's modes on r < r_cut, n_c >= 1e-3 max n_c, volume-orthonormal) at every
member, for the comparison with the finite-difference (a) values of batch 6.
(ii) The full matter-node basis (every node with n_c > 0), the fixed-N_B projection in the volume metric (y = D v, D = diag((4 pi r^2)^1/2);
the complement of D 1 by a Householder reflection), at five members (the dilute, the mid, the two bracketing the fitted maximum, the
most compact): the inertia (negative eigenvalues) and the three lowest eigenvalues.
Checks: the gradient against mu_inf 4 pi r_j^2 on the matter nodes (the TF condition); at the five members the Hessian in basis (a) by
central differences of the exact gradient (steps eps max(n_c)/max|v_k| with eps = 1e-3 and 5e-4, Richardson), its asymmetry and its
largest difference from the exact (a) Hessian; the re-solved state's residual and its distance from the stage-1 profile.
Writes <out>/stage2_<tag>.json.
(version 10: --full-node all computes the full matter-node inertia and the gradient checks at every member (for spec sec. 10(r)(b)'s refine
families of a few members); version 9's default "five" is unchanged.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
import types

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev as Ch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J4_stage2_v8 as V8  # noqa: E402

S = V8.S
STEPS = (1e-3, 5e-4)


def loc_eval(cs, psi, nc, kap):
    return cs.local(psi, nc, kap)[0]


def local_derivs(cs, psi, nc, kap):
    """Per node: loc_p, loc_n, loc_pp, loc_pn, loc_nn by central differences (relative steps), Richardson over STEPS."""
    mk = nc > 0
    out = {}
    D = {}
    for s in STEPS:
        dp = s * psi
        dn = np.where(mk, s * nc, 0.0)
        L = {}
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                L[(a, b)] = loc_eval(cs, psi + a * dp, nc + b * dn, kap)
        d = {"p": (L[(1, 0)] - L[(-1, 0)]) / (2 * dp), "pp": (L[(1, 0)] - 2 * L[(0, 0)] + L[(-1, 0)]) / dp ** 2}
        with np.errstate(divide="ignore", invalid="ignore"):
            d["n"] = np.where(mk, (L[(0, 1)] - L[(0, -1)]) / (2 * dn), 0.0)
            d["nn"] = np.where(mk, (L[(0, 1)] - 2 * L[(0, 0)] + L[(0, -1)]) / dn ** 2, 0.0)
            d["pn"] = np.where(mk, (L[(1, 1)] - L[(1, -1)] - L[(-1, 1)] + L[(-1, -1)]) / (4 * dp * dn), 0.0)
        D[s] = d
    s1, s2 = STEPS
    for k in D[s1]:
        out[k] = (4 * D[s2][k] - D[s1][k]) / 3
        out[k + "_richardson_change"] = float(np.max(np.abs(out[k] - D[s2][k])))
    return out


class Exact:
    def __init__(self, F, star, x, n, kop, kap_nodes):
        J = star.J
        self.J, self.star, self.kop = J, star, kop
        r = star.r
        self.r = r
        psi, A = x[:J], x[2 * J]
        self.nc = n * psi ** 6
        cs = V8.ClockSolver(F, star)
        self.cs = cs
        psi_c, A_c, kap_c, res_c, it_c = cs.solve(self.nc, psi, A, kap_nodes)
        self.psi, self.A, self.kap = psi_c, A_c, kap_c
        self.resolve = {"residual": res_c, "iterations": it_c, "psi_maxdiff": float(np.max(np.abs(psi_c - psi))), "A_diff": float(A_c - A)}
        self.d = local_derivs(cs, psi_c, self.nc, kap_c)
        top = sp.hstack([cs.Mr + sp.diags(self.d["p"]), sp.csr_matrix(star.op.tail[:, None])])
        last = np.zeros(J + 1)
        last[J - 1] = r[-1]
        last[J] = -1
        self.lu = spla.splu(sp.vstack([top, sp.csr_matrix(last[None, :])]).tocsc())
        c = np.zeros(J + 1)
        c[J] = 4 * np.pi * kop
        self.lam = self.lu.solve(c, trans="T")[:J]
        self.grad = -self.lam * self.d["n"]
        self.M = 4 * np.pi * kop * A_c

    def hessian(self, Q):
        J = self.J
        rhs = np.vstack([self.d["n"][:, None] * Q, np.zeros((1, Q.shape[1]))])
        Xp = -self.lu.solve(rhs)
        P = Xp[:J]
        lp, ln, lnn = self.lam * self.d["pp"], self.lam * self.d["pn"], self.lam * self.d["nn"]
        H = -(P.T @ (lp[:, None] * P) + P.T @ (ln[:, None] * Q) + Q.T @ (ln[:, None] * P) + Q.T @ (lnn[:, None] * Q))
        return 0.5 * (H + H.T), float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))


def basis_c_raw(r, rs, nc0, K=12):
    x = (r / rs) ** 2
    inside = (r < rs) & (nc0 > 0)
    T = np.zeros((len(r), K + 1))
    for i in range(K + 1):
        c = np.zeros(i + 1)
        c[i] = 1
        T[:, i] = np.where(inside, Ch.chebval(2 * x - 1, c), 0.0)
    V = nc0[:, None] * T
    w = 4 * np.pi * r * r
    _, _, Vt = np.linalg.svd((w @ V)[None, :])
    Z = V @ Vt[1:].T
    return Z, int(np.sum(inside)), float(np.max(np.abs(w @ Z)) / max(np.max(np.abs(w[:, None] * Z).sum(0)), 1e-300))


def gen_eig(H, G, rel=1e-10):
    g, U = np.linalg.eigh(G)
    keep = g > rel * g.max()
    B = U[:, keep] / np.sqrt(g[keep])
    ev = np.linalg.eigvalsh(B.T @ H @ B)
    return ev, int(np.sum(keep)), float(g.max() / g[keep].min())


def full_node(ex):
    r, nc = ex.r, ex.nc
    idx = np.nonzero(nc > 0)[0]
    Jm = len(idx)
    Q = np.zeros((ex.J, Jm))
    Q[idx, np.arange(Jm)] = 1.0
    H, asym = ex.hessian(Q)
    del Q
    dvec = np.sqrt(4 * np.pi * r[idx] ** 2)
    Hy = H / dvec[:, None] / dvec[None, :]
    del H
    u = dvec / np.linalg.norm(dvec)
    e1 = np.zeros(Jm)
    e1[0] = 1.0
    v = u - e1
    v /= np.linalg.norm(v)
    # P = I - 2 v v^T maps u to e1; the columns 1.. of P span u's complement; P^T Hy P in place, then drop row/col 0
    Hv = Hy @ v
    Hy -= 2 * np.outer(Hv, v) + 2 * np.outer(v, Hv) - 4 * (v @ Hv) * np.outer(v, v)
    Hp = Hy[1:, 1:]
    neg = sla.eigh(Hp, eigvals_only=True, subset_by_value=[-np.inf, 0.0], check_finite=False)
    low = sla.eigh(Hp, eigvals_only=True, subset_by_index=[0, 2], check_finite=False)
    return {"matter_nodes": Jm, "inertia": int(len(neg)), "lowest3": [float(x) for x in low], "asymmetry_before_symmetrization": asym}


def fd_gradient_check(F, star, ex, Qa, kap_nodes):
    """Basis (a) Hessian by central differences of the exact gradient, steps eps max(n_c)/max|v_k| (eps 1e-3, 5e-4), Richardson."""
    J = star.J
    Hs = {}
    for eps in (1e-3, 5e-4):
        h = eps * np.max(ex.nc) / np.max(np.abs(Qa), axis=0)
        cap = np.array([0.5 * float(np.min(ex.nc[np.abs(Qa[:, k]) > 0] / np.abs(Qa[np.abs(Qa[:, k]) > 0, k]))) for k in range(Qa.shape[1])])
        h = np.minimum(h, cap)
        H = np.zeros((Qa.shape[1], Qa.shape[1]))
        for k in range(Qa.shape[1]):
            g = []
            for sgn in (+1, -1):
                ncp = ex.nc + sgn * h[k] * Qa[:, k]
                x = np.concatenate([ex.psi, ex.psi, [ex.A, ex.A, 0.0]])
                # the exact gradient at the perturbed density: the clock row re-solved there, lambda and loc_n recomputed
                e2 = Exact(F, star, x, ncp / ex.psi ** 6, ex.kop, ex.kap)
                g.append(Qa.T @ e2.grad)
            H[:, k] = (g[0] - g[1]) / (2 * h[k])
        Hs[eps] = H
    Hr = (4 * Hs[5e-4] - Hs[1e-3]) / 3
    return Hr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="lstar_<tag>.json")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-node", default="five", choices=["five", "none", "all"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = json.load(open(a.family))
    M = R["meta"]
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = types.SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, info = S.setup_kernel(args)
    if abs(kop / M["k_op"] - 1) > 1e-12:
        raise SystemExit("k_op mismatch: %r against %r" % (kop, M["k_op"]))
    gM, gB = M["couplings"]
    coup = (gM, gB, 1.0, 1.0)
    e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    mem = [d for d in R["members"] if d.get("newton_ok")]
    Ms = np.array([d["M"] for d in mem])
    imax = int(np.argmax(Ms))
    # the fitted maximum (parabola in log z_c) and the two members bracketing it
    br = (imax, imax + 1)
    if 0 < imax < len(mem) - 1:
        xx = np.log([mem[j]["z_c"] for j in (imax - 1, imax, imax + 1)])
        cc = np.polyfit(xx, Ms[imax - 1:imax + 2], 2)
        zfit = float(np.exp(-cc[1] / (2 * cc[0])))
        br = (imax, imax + 1) if zfit >= mem[imax]["z_c"] else (imax - 1, imax)
    else:
        zfit = mem[imax]["z_c"]
    five = sorted({0, max(0, imax // 2), br[0], min(br[1], len(mem) - 1), len(mem) - 1})
    res = {"meta": {"family": a.family, "k_op": kop, "operator": M["operator"], "mlight": M["mlight"], "aH": M["aH"], "couplings": [gM, gB],
                    "version": 10, "full_node_option": a.full_node, "eos_fast_check_max_rel": V8.FAST_CHECK, "series_ext_check_max_rel": V8.SERC_CHECK, "index_of_max": imax,
                    "z_c_fitted_max": zfit, "bracketing": list(br), "five_members": five, "steps": list(STEPS)}, "members": []}
    out = os.path.join(a.out, "stage2_" + os.path.basename(a.family))
    for i, d in enumerate(mem):
        zc = d["z_c"]
        key = "z%.6f" % zc
        r = prof[key + "_r"]
        psi = prof[key + "_psi"]
        N = prof[key + "_N"]
        n = prof[key + "_n"]
        qF = prof[key + "_qF"]
        J = len(r)
        star = S.Star(J, stencil, kop, M["mlight"], coup, e1)
        chi = N * psi
        A, B, mu_inf = d["A"], d["B"], d["mu_inf"]
        x = np.concatenate([psi, chi, [A, B, mu_inf]])
        F = V8.Fam(star, M["mlight"], e1)
        kap_nodes = psi * psi * qF
        sps, sch, _ = V8.splines(r, psi, chi, n, A, B, star.op.R)
        nf = V8.n_tf(star, sps, sch, mu_inf, J)
        rec = {"z_c": zc, "A": A, "B": B, "M": d["M"], "J": J, "r_s": d["r_s"], "N_B": d["N_B"], "mu_inf": mu_inf}
        rec["lock"] = V8.lock_pieces(F, star, x, r, sps, sch, nf, A, B, mu_inf, kop, kap_nodes)
        rec["eos_series"] = V8.series_truncation(float(np.max(kap_nodes)))
        t1 = time.time()
        ex = Exact(F, star, x, n, kop, kap_nodes)
        mk = ex.nc > 0
        ratio = ex.grad[mk] / (mu_inf * 4 * np.pi * r[mk] ** 2)
        rec["exact"] = {"resolve": ex.resolve, "M_resolved": ex.M, "matter_nodes": int(np.sum(mk)),
                        "gradient_over_mu_volume": {"max_abs_minus_1": float(np.max(np.abs(ratio - 1))), "median_abs_minus_1": float(np.median(np.abs(ratio - 1))),
                                                    "max_abs_minus_1_excluding_last_node": float(np.max(np.abs(ratio[:-1] - 1))) if ratio.size > 1 else None,
                                                    "last_node": float(ratio[-1] - 1)},
                        "local_derivative_richardson_change": {k: v for k, v in ex.d.items() if k.endswith("_richardson_change")}}
        Z, nsup, cons = basis_c_raw(r, d["r_s"], ex.nc)
        Hz, asym = ex.hessian(Z)
        Gz = Z.T @ ((4 * np.pi * r * r)[:, None] * Z)
        evc, kept, gcond = gen_eig(Hz, Gz)
        rec["hessian_c_exact"] = {"eigenvalues": evc.tolist(), "n_negative": int(np.sum(evc < 0)), "kept": kept, "gram_condition_kept": gcond,
                                  "support_nodes": nsup, "constraint_err_rel": cons, "asymmetry_before_symmetrization": asym}
        sig = np.nonzero(ex.nc >= 1e-3 * np.max(ex.nc))[0]
        r_cut = float(r[sig.max()]) if len(sig) else d["r_s"]
        Qa, cons_a, orth_a = V8.modes(r, r_cut)
        Ha, asym_a = ex.hessian(Qa)
        eva = np.linalg.eigvalsh(Ha)
        rec["hessian_a_exact"] = {"eigenvalues": eva.tolist(), "n_negative": int(np.sum(eva < 0)), "r_cut": r_cut,
                                  "gradient_along_modes_max": float(np.max(np.abs(Qa.T @ ex.grad))), "asymmetry_before_symmetrization": asym_a}
        if a.full_node == "all" or (a.full_node == "five" and i in five):
            fn = full_node(ex)
            rec["full_node"] = fn
            Hfd = fd_gradient_check(F, star, ex, Qa, kap_nodes)
            rec["fd_gradient_check_a"] = {"eigenvalues": np.linalg.eigvalsh(0.5 * (Hfd + Hfd.T)).tolist(),
                                          "asymmetry": float(np.linalg.norm(Hfd - Hfd.T) / max(np.linalg.norm(Hfd), 1e-300)),
                                          "max_rel_diff_from_exact": float(np.max(np.abs(Hfd - Ha)) / max(np.max(np.abs(Ha)), 1e-300))}
        rec["exact_seconds"] = time.time() - t1
        res["members"].append(rec)
        print("z_c %.4f J %d: (c) %s neg %d kept %d | (a) %s | grad/mu-1 med %.1e%s (%.0f s)" % (
            zc, J, np.array2string(evc[:3], precision=4), int(np.sum(evc < 0)), kept, np.array2string(eva[:3], precision=4),
            rec["exact"]["gradient_over_mu_volume"]["median_abs_minus_1"],
            ("; full-node inertia %d lowest %s; FD-gradient check %.1e" % (rec["full_node"]["inertia"], np.array2string(np.array(rec["full_node"]["lowest3"]), precision=4),
                                                                            rec["fd_gradient_check_a"]["max_rel_diff_from_exact"])) if "full_node" in rec else "",
            time.time() - t0), flush=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    res["meta"]["seconds"] = time.time() - t0
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
