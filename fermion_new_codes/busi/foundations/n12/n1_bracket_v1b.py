#!/usr/bin/env python
"""n1_bracket_v1.py (reviewer A, foundations item N1: ReviewA_foundations_v1.md sec. 3.2 N1 and Chain 1; paper Sec. II D,
App. app:rows, Eqs. dr-dirac-schwinger and dr-mixed-bracket).

DATA script (box CPU only, torch for the dense products, at most 8 threads). Exact periodic lattice operators of the paper
(sine hopping, Dirac matrices, the two-step current Q, site allocations e_x = {Pi_x, h_0}/2, p_x = {Pi_x, Q}/2), a = 1.

Checks:
(1) the leading-order decomposition of the commutator of two coupled generators,
      i[h[g], h[u]] = -( E[phi] + P[w, psi] + S[M] ) + O(second order in the field gradients),
    E[phi]   = sum_i {phi_i, Chat_i h_0}/2,          phi_i   = v_i d_i g_N - g_{v,i} d_i N,
    P[w,psi] = sum_i {w_i, Q_i}/2 + sum_ij {psi_ij, Chat_i Q_j}/2,
               w_i = N d_i g_N - g_N d_i N,  psi_ij = v_i d_i g_{v,j} - g_{v,i} d_i v_j,
    S[M]     = sum_ij {M_ij, alpha_i chat_i Q_j}/2,  M_ij = N d_i g_{v,j} - g_N d_i v_j   (the stress coefficient),
    chat_i = (T_i + T_i^+)/2, Chat_i = (T_i^2 + T_i^+2)/2 (the cos(a p_i) and cos(2 a p_i) weights of the packet velocity),
    read as the Frobenius residual |X - X_pred| / |X| at several L with the fields' wavelength L (a k_f = 2 pi / L);
(2) in the matter's low-momentum sector (plane waves within n_c modes of each of the eight valleys, both bands), the fraction of
    X outside the span of the retained energy and momentum vertices e[e^{iKx}], p_j[e^{iKx}] over the field transfers K present,
    for weights with M = 0 (g = u + (0, c)) and with M != 0 (clock reweighted, flow reweighted, position-dependent lambda);
    then the same with the nine stress vertices S_ij[e^{iKx}] added, and the fitted stress coefficients against -M_ij(K).

Outputs: <out>/n1_bracket_v1_L<L>_m<m>.json (every number). Never edits its inputs."""
import os
import sys


def _threads():
    t = "8"
    if "--threads" in sys.argv:
        t = sys.argv[sys.argv.index("--threads") + 1]
    t = str(max(1, min(int(t), 8)))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = t
    return int(t)


NTH = _threads()
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NTH)
VERSION = "n1_bracket_v1b"

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)
ALPHA = [np.kron(SX, s) for s in (SX, SY, SZ)]   # alpha_i = [[0, sigma_i], [sigma_i, 0]]
BETA = np.kron(SZ, I2)                            # beta = diag(1, 1, -1, -1)


def site_shifts(L):
    """T_d (d = 0, 1, 2): (T_d u)_x = u_{x + e_d}, periodic; site index x = (i L + j) L + k."""
    n = L ** 3
    idx = np.arange(n).reshape(L, L, L)
    T = []
    for d in range(3):
        tgt = np.roll(idx, -1, axis=d).ravel()          # x -> x + e_d
        T.append(sp.csr_matrix((np.ones(n), (np.arange(n), tgt)), shape=(n, n)))
    return T


def lattice_ops(L, m):
    """h_0, Q_d, phat_d, chat_d, Chat_d, and the spin matrices lifted to the 4 L^3 space (site-major, component 4x + a)."""
    T = site_shifts(L)
    n = L ** 3
    I4 = sp.identity(4, format="csr", dtype=complex)
    In = sp.identity(n, format="csr", dtype=complex)
    phat = [(Td - Td.T) / (2j) for Td in T]
    Q = [((Td @ Td) - (Td.T @ Td.T)) / (4j) for Td in T]
    chat = [(Td + Td.T) / 2 for Td in T]
    Chat = [((Td @ Td) + (Td.T @ Td.T)) / 2 for Td in T]
    h0 = sp.kron(In, m * sp.csr_matrix(BETA), format="csr")
    for d in range(3):
        h0 = h0 + sp.kron(phat[d], sp.csr_matrix(ALPHA[d]), format="csr")
    lift = lambda A: sp.kron(A, I4, format="csr")
    alpha = [sp.kron(In, sp.csr_matrix(ALPHA[d]), format="csr") for d in range(3)]
    return dict(h0=h0.tocsr(), Q=[lift(q) for q in Q], chat=[lift(c) for c in chat], Chat=[lift(c) for c in Chat],
                alpha=alpha, n=n)


def smear(f, A):
    """{F, A}/2 with F = diag(f) repeated over the 4 components (the paper's site-symmetric allocation)."""
    F = sp.diags(np.repeat(np.asarray(f, float), 4))
    return 0.5 * (F @ A + A @ F)


def coupled(ops, N, v):
    """h[N, v] = {N, h_0}/2 + sum_d {v_d, Q_d}/2."""
    H = smear(N, ops["h0"])
    for d in range(3):
        H = H + smear(v[:, d], ops["Q"][d])
    return H.tocsr()


class Field:
    """a real periodic field as a sum of cosine modes: value and exact gradient on the sites."""

    def __init__(self, L, const, modes):
        self.L, self.const, self.modes = L, const, modes   # modes: list of (amplitude, (n1, n2, n3), phase)

    def grid(self):
        g = np.arange(self.L)
        return np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)

    def value(self):
        X = self.grid()
        out = np.full(self.L ** 3, float(self.const))
        for A, nvec, ph in self.modes:
            out += A * np.cos(2 * np.pi * (X @ np.asarray(nvec, float)) / self.L + ph)
        return out

    def grad(self):
        X = self.grid()
        out = np.zeros((self.L ** 3, 3))
        for A, nvec, ph in self.modes:
            s = np.sin(2 * np.pi * (X @ np.asarray(nvec, float)) / self.L + ph)
            for d in range(3):
                out[:, d] += -A * (2 * np.pi * nvec[d] / self.L) * s
        return out


def background_fields(L):
    """generic smooth clock and flow with one wavelength L per mode; grad v has rank 3 almost everywhere."""
    N = Field(L, 1.0, [(0.15, (1, 0, 0), 0.0), (0.10, (0, 1, 0), 1.0), (0.08, (0, 0, 1), 2.0), (0.05, (1, 1, 0), 0.4)])
    v = [Field(L, 0.0, [(0.20, (0, 1, 0), 0.3), (0.08, (1, 0, 0), 1.3)]),
         Field(L, 0.0, [(0.20, (0, 0, 1), 0.6), (0.08, (0, 1, 0), 2.3)]),
         Field(L, 0.0, [(0.20, (1, 0, 0), 0.9), (0.08, (0, 0, 1), 0.2)])]
    return N, v


def weight_cases(L, N, v):
    """the weights g = (g_N, g_v) tested, each as (clock Field, [flow Fields], label)."""
    c = (0.3, -0.2, 0.1)
    cases = {}
    cases["A_uniform_flow"] = (N, [Field(L, v[d].const + c[d], v[d].modes) for d in range(3)])      # M = 0
    dN = Field(L, 0.0, [(0.3, (1, 0, 0), 0.5)])
    cases["B_clock_reweighted"] = ("N(1+delta)", dN)                                                  # M = -N delta grad v
    dv = [Field(L, 0.0, [(0.2, (0, 1, 0), 1.1)]), Field(L, 0.0, [(0.2, (0, 0, 1), 2.2)]), Field(L, 0.0, [(0.2, (1, 0, 0), 3.3)])]
    cases["C_flow_reweighted"] = ("v+dv", dv)                                                        # M = N grad dv
    lam = Field(L, 1.0, [(0.3, (0, 1, 0), 0.7)])
    cases["D_position_dependent_lambda"] = ("lambda u", lam)                                        # M = N grad lam (x) v
    cases["E_identity"] = ("u", None)                                                                # X = 0
    return cases


def weights_from_case(case, N, v):
    """returns g_N (values), grad g_N, g_v (values (n,3)), grad g_v ((n,3,3): d_i g_{v,j})."""
    Nv, gN = N.value(), N.grad()
    vv = np.stack([f.value() for f in v], axis=1)
    gv = np.stack([f.grad() for f in v], axis=2)        # [:, i, j] = d_i v_j
    tag = case[0]
    if isinstance(tag, Field):                           # case A: (N, v + c)
        gNw, ggNw = Nv, gN
        gvw = np.stack([f.value() for f in case[1]], axis=1)
        ggvw = np.stack([f.grad() for f in case[1]], axis=2)
    elif tag == "N(1+delta)":
        dN = case[1]
        dv_, gd = dN.value(), dN.grad()
        gNw = Nv * (1 + dv_)
        ggNw = gN * (1 + dv_)[:, None] + Nv[:, None] * gd
        gvw, ggvw = vv, gv
    elif tag == "v+dv":
        gNw, ggNw = Nv, gN
        gvw = vv + np.stack([f.value() for f in case[1]], axis=1)
        ggvw = gv + np.stack([f.grad() for f in case[1]], axis=2)
    elif tag == "lambda u":
        lam = case[1]
        lv, gl = lam.value(), lam.grad()
        gNw = lv * Nv
        ggNw = gl * Nv[:, None] + lv[:, None] * gN
        gvw = lv[:, None] * vv
        ggvw = gl[:, :, None] * vv[:, None, :] + lv[:, None, None] * gv
    else:
        gNw, ggNw, gvw, ggvw = Nv, gN, vv, gv
    return dict(N=Nv, gN=gN, v=vv, gv=gv, g_N=gNw, gg_N=ggNw, g_v=gvw, gg_v=ggvw)


def predicted(ops, W):
    """-(E[phi] + P[w, psi] + S[M]) with the coefficient fields of the leading-order symbol expansion."""
    N, gN, v, gv = W["N"], W["gN"], W["v"], W["gv"]
    g_N, gg_N, g_v, gg_v = W["g_N"], W["gg_N"], W["g_v"], W["gg_v"]
    n = len(N)
    phi = np.stack([v[:, i] * gg_N[:, i] - g_v[:, i] * gN[:, i] for i in range(3)], axis=1)
    w = np.stack([N * gg_N[:, i] - g_N * gN[:, i] for i in range(3)], axis=1)
    psi = np.zeros((n, 3, 3))
    M = np.zeros((n, 3, 3))
    for i in range(3):
        for j in range(3):
            psi[:, i, j] = v[:, i] * gg_v[:, i, j] - g_v[:, i] * gv[:, i, j]
            M[:, i, j] = N * gg_v[:, i, j] - g_N * gv[:, i, j]
    E = sp.csr_matrix(ops["h0"].shape, dtype=complex)
    P = sp.csr_matrix(ops["h0"].shape, dtype=complex)
    S = sp.csr_matrix(ops["h0"].shape, dtype=complex)
    for i in range(3):
        E = E + smear(phi[:, i], ops["Chat"][i] @ ops["h0"])
        P = P + smear(w[:, i], ops["Q"][i])
        for j in range(3):
            P = P + smear(psi[:, i, j], ops["Chat"][i] @ ops["Q"][j])
            S = S + smear(M[:, i, j], ops["alpha"][i] @ ops["chat"][i] @ ops["Q"][j])
    return dict(E=-E.tocsr(), P=-P.tocsr(), S=-S.tocsr(), M=M, phi=phi, w=w, psi=psi)


def fro(A):
    return float(np.sqrt(np.sum(np.abs(A.data) ** 2))) if sp.issparse(A) else float(np.linalg.norm(A))


def low_momentum_basis(L, n_c):
    """columns: plane waves within n_c modes (per axis) of each valley, times each of the 4 components; returns U (dense),
    the integer momenta nvec (mod L) per column and the component per column."""
    g = np.arange(L)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    half = L // 2
    cols, nvecs, comps = [], [], []
    for valley in itertools.product((0, half), repeat=3):
        for dn in itertools.product(range(-n_c, n_c + 1), repeat=3):
            nvec = tuple((np.array(valley) + np.array(dn)) % L)
            p = 2 * np.pi * np.array(nvec, float) / L
            wave = np.exp(1j * (X @ p)) / L ** 1.5
            for a in range(4):
                col = np.zeros(4 * L ** 3, complex)
                col[a::4] = wave
                cols.append(col)
                nvecs.append(nvec)
                comps.append(a)
    U = np.stack(cols, axis=1)
    return U, nvecs, np.array(comps)


def compress(A, U):
    """U^+ (A U) for sparse A and dense U, through torch."""
    AU = A @ U
    Ut = torch.from_numpy(np.ascontiguousarray(U))
    AUt = torch.from_numpy(np.ascontiguousarray(AU))
    return (Ut.conj().T @ AUt).numpy()


def site0_vertices(ops, L):
    """the allocations at site 0: e_0 = {Pi_0, h_0}/2, p_{0,d} = {Pi_0, Q_d}/2, S_{0,ij} = {Pi_0, alpha_i chat_i Q_j}/2."""
    n = L ** 3
    pi0 = np.zeros(n)
    pi0[0] = 1.0
    verts = {"e": smear(pi0, ops["h0"])}
    for d in range(3):
        verts["p%d" % d] = smear(pi0, ops["Q"][d])
    for i in range(3):
        for j in range(3):
            verts["S%d%d" % (i, j)] = smear(pi0, ops["alpha"][i] @ ops["chat"][i] @ ops["Q"][j])
    for l in range(3):
        m_, n_ = (l + 1) % 3, (l + 2) % 3
        verts["sp%d" % l] = smear(pi0, -0.5j * (ops["alpha"][m_] @ ops["alpha"][n_] - ops["alpha"][n_] @ ops["alpha"][m_]))
    return verts


def span_residual(Xt, Vt, nvecs, comps, L, n_K, types):
    """fraction of the compressed X outside the span of the plane-wave vertices of the given types over |n_K| <= n_K per axis,
    and the fitted coefficients per K. Vertices at different K are Hilbert-Schmidt orthogonal (translation invariance), so the
    projection is assembled per K from the small Gram matrices. Returns (residual fraction, outside-band fraction, fits)."""
    n = len(nvecs)
    key = {}
    for c, (nv, a) in enumerate(zip(nvecs, comps)):
        key.setdefault(tuple(nv), [None] * 4)[a] = c
    total = float(np.sum(np.abs(Xt) ** 2))
    captured = 0.0
    inband = 0.0
    fits = {}
    for nK in itertools.product(range(-n_K, n_K + 1), repeat=3):
        rows, cols = [], []
        for nv, cs in key.items():
            tgt = tuple((np.array(nv) + np.array(nK)) % L)
            if tgt in key:
                for a in range(4):
                    for b in range(4):
                        rows.append(cs[a])
                        cols.append(key[tgt][b])
        if not rows:
            continue
        rows, cols = np.array(rows), np.array(cols)
        x = Xt[rows, cols]
        inband += float(np.sum(np.abs(x) ** 2))
        V = np.stack([Vt[t][rows, cols] * L ** 3 for t in types], axis=1)     # columns: the vertices A_{t,K} on their support
        G = V.conj().T @ V
        mvec = V.conj().T @ x
        coef = np.linalg.lstsq(G, mvec, rcond=1e-13)[0]
        captured += float(np.real(np.vdot(mvec, coef)))
        fits[nK] = coef
    res = max(total - captured, 0.0)
    return float(np.sqrt(res / total)) if total > 0 else 0.0, float(np.sqrt(max(total - inband, 0) / total)) if total > 0 else 0.0, fits


def fourier_coeff(field_vals, L, nK):
    """Mhat(K) = (1/L^3) sum_x M_x e^{-i K x} for the site field (so M_x = sum_K Mhat(K) e^{i K x})."""
    g = np.arange(L)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    ph = np.exp(-1j * 2 * np.pi * (X @ np.asarray(nK, float)) / L)
    return np.sum(field_vals * ph) / L ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--n_c", type=int, default=1)
    ap.add_argument("--n_K", type=int, default=2)
    ap.add_argument("--out", default=".")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    L, m = args.L, args.m
    t0 = time.time()
    ops = lattice_ops(L, m)
    N, v = background_fields(L)
    H = coupled(ops, N.value(), np.stack([f.value() for f in v], axis=1))
    herm = fro(H - H.conj().T) / fro(H)
    cases = weight_cases(L, N, v)
    out = dict(version=VERSION, L=L, m=m, n_c=args.n_c, n_K=args.n_K, a_k_field=2 * np.pi / L,
               a_k_c=2 * np.pi * args.n_c / L, herm_defect=herm, threads=NTH, cases={})
    # low-momentum compression
    U, nvecs, comps = low_momentum_basis(L, args.n_c)
    verts = site0_vertices(ops, L)
    Vt = {k: compress(A, U) for k, A in verts.items()}
    types1 = ["e", "p0", "p1", "p2"]
    typesS = ["S%d%d" % (i, j) for i in range(3) for j in range(3)]
    types2 = types1 + typesS
    types3 = types2 + ["sp0", "sp1", "sp2"]
    for label, case in cases.items():
        W = weights_from_case(case, N, v)
        G = coupled(ops, W["g_N"], W["g_v"])
        X = (1j * (G @ H - H @ G)).tocsr()
        nX = fro(X)
        rec = dict(norm_X=nX)
        if label == "E_identity":
            rec["note"] = "g = u: X must vanish exactly"
            out["cases"][label] = rec
            continue
        pred = predicted(ops, W)
        Xp = pred["E"] + pred["P"] + pred["S"]
        rec["decomposition_residual"] = fro(X - Xp) / nX
        rec["norm_E_pred"] = fro(pred["E"]) / nX
        rec["norm_P_pred"] = fro(pred["P"]) / nX
        rec["norm_S_pred"] = fro(pred["S"]) / nX
        rec["max_abs_M"] = float(np.max(np.abs(pred["M"])))
        # compressed
        Xt = compress(X, U)
        r1, outb, fits1 = span_residual(Xt, Vt, nvecs, comps, L, args.n_K, types1)
        r2, _, fits2 = span_residual(Xt, Vt, nvecs, comps, L, args.n_K, types2)
        rec["frac_outside_e_p_span_lowk"] = r1
        rec["frac_outside_band"] = outb
        rec["frac_outside_e_p_S_span_lowk"] = r2
        rec["frac_outside_e_p_S_spin_span_lowk"] = span_residual(Xt, Vt, nvecs, comps, L, args.n_K, types3)[0]
        # fitted stress coefficients against -Mhat(K)
        num, den = 0.0, 0.0
        for nK, coef in fits2.items():
            cS = coef[4:]
            Mh = np.array([-np.conj(fourier_coeff(pred["M"][:, i, j], L, nK)) for i in range(3) for j in range(3)])
            num += float(np.sum(np.abs(cS - Mh) ** 2))
            den += float(np.sum(np.abs(Mh) ** 2))
        rec["stress_fit_rel_error"] = float(np.sqrt(num / den)) if den > 0 else None
        rec["stress_fit_norm_Mhat"] = float(np.sqrt(den))
        # second route: fit the nine stress vertices alone to X - E_pred - P_pred (the e, p parts removed by the formula)
        Rt = Xt - compress(pred["E"], U) - compress(pred["P"], U)
        rS, _, fitsS = span_residual(Rt, Vt, nvecs, comps, L, args.n_K, typesS)
        num, den = 0.0, 0.0
        for nK, coef in fitsS.items():
            Mh = np.array([-np.conj(fourier_coeff(pred["M"][:, i, j], L, nK)) for i in range(3) for j in range(3)])
            num += float(np.sum(np.abs(coef - Mh) ** 2))
            den += float(np.sum(np.abs(Mh) ** 2))
        rec["stress_fit_rel_error_residual_route"] = float(np.sqrt(num / den)) if den > 0 else None
        rec["frac_of_X_minus_EP_pred_outside_S_span_lowk"] = rS
        rec["norm_X_minus_EP_pred_over_X"] = fro(Rt) / fro(Xt) if fro(Xt) > 0 else None
        # the pure vertices' own fractions (sanity: E, P in span up to the C weights; S outside)
        for nm in ("E", "P", "S"):
            Pt = compress(pred[nm], U)
            if fro(Pt) > 0:
                rec["frac_outside_e_p_span_lowk_%s_pred" % nm] = span_residual(Pt, Vt, nvecs, comps, L, args.n_K, types1)[0]
        out["cases"][label] = rec
        print(label, {k: (round(x, 6) if isinstance(x, float) else x) for k, x in rec.items()}, flush=True)
    out["elapsed_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s_L%d_m%g.json" % (VERSION, L, m))
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", path, "elapsed", round(out["elapsed_s"], 1), flush=True)


if __name__ == "__main__":
    main()
