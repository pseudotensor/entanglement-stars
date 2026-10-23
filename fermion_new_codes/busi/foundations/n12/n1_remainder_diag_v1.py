#!/usr/bin/env python
"""n1_remainder_diag_v1.py (reviewer A, foundations N1 diagnostic). For the exact lattice commutator X = i[h[g], h[u]] on the
low-momentum sector, remove the projection onto the retained energy, momentum and stress vertices and decompose what is
left on the 16 Dirac matrices (per transfer K, summed over the sector's momenta), to identify the second-order remainder.
Also: the Dirac decomposition of the pure energy-energy bracket i[e[f], e[g]] minus p[f grad g - g grad f] (is the
Dirac-Schwinger spin term there on the lattice?). Output <out>/n1_remainder_diag_v1_L<L>.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_bracket_v1b as nb  # noqa: E402

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

# 16 orthonormal Dirac matrices: 1, beta, alpha_i, Sigma_i, beta alpha_i, gamma5 (= alpha_1 alpha_2 alpha_3 i), beta Sigma_i, beta gamma5
I4 = np.eye(4, dtype=complex)
AL = nb.ALPHA
BE = nb.BETA
SIG = [-0.5j * (AL[(l + 1) % 3] @ AL[(l + 2) % 3] - AL[(l + 2) % 3] @ AL[(l + 1) % 3]) for l in range(3)]
G5 = -1j * AL[0] @ AL[1] @ AL[2]
DIRAC = {"1": I4, "beta": BE, "g5": G5, "i_beta_g5": 1j * BE @ G5}
for i in range(3):
    DIRAC["alpha%d" % i] = AL[i]
    DIRAC["Sigma%d" % i] = SIG[i]
    DIRAC["i_beta_alpha%d" % i] = 1j * BE @ AL[i]
    DIRAC["beta_Sigma%d" % i] = BE @ SIG[i]
for k, M in DIRAC.items():
    assert np.allclose(M, M.conj().T), k
    assert abs(np.trace(M @ M) - 4) < 1e-12, k


def dirac_weights(Xt, nvecs, comps, L, n_K):
    """per Dirac matrix, the Hilbert-Schmidt weight of Xt's blocks (p, p+K), summed over p and K in the band; returns fractions."""
    key = {}
    for c, (nv, a) in enumerate(zip(nvecs, comps)):
        key.setdefault(tuple(nv), [None] * 4)[a] = c
    w = {k: 0.0 for k in DIRAC}
    tot = 0.0
    for nK in itertools.product(range(-n_K, n_K + 1), repeat=3):
        for nv, cs in key.items():
            tgt = tuple((np.array(nv) + np.array(nK)) % L)
            if tgt not in key:
                continue
            blk = Xt[np.ix_(cs, key[tgt])]
            tot += float(np.sum(np.abs(blk) ** 2))
            for k, M in DIRAC.items():
                w[k] += abs(np.trace(M.conj().T @ blk)) ** 2 / 4.0
    return {k: v / tot for k, v in w.items()}, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    L, m = args.L, args.m
    ops = nb.lattice_ops(L, m)
    N, v = nb.background_fields(L)
    H = nb.coupled(ops, N.value(), np.stack([f.value() for f in v], axis=1))
    cases = nb.weight_cases(L, N, v)
    U, nvecs, comps = nb.low_momentum_basis(L, 1)
    verts = nb.site0_vertices(ops, L)
    Vt = {k: nb.compress(A, U) for k, A in verts.items()}
    types1 = ["e", "p0", "p1", "p2"]
    typesS = ["S%d%d" % (i, j) for i in range(3) for j in range(3)]
    out = dict(L=L, m=m, cases={})
    for label in ("B_clock_reweighted", "C_flow_reweighted"):
        W = nb.weights_from_case(cases[label], N, v)
        G = nb.coupled(ops, W["g_N"], W["g_v"])
        X = (1j * (G @ H - H @ G)).tocsr()
        Xt = nb.compress(X, U)
        pred = nb.predicted(ops, W)
        Xp = nb.compress(pred["E"] + pred["P"] + pred["S"], U)
        R = Xt - Xp
        wX, tX = dirac_weights(Xt, nvecs, comps, L, 2)
        wR, tR = dirac_weights(R, nvecs, comps, L, 2)
        out["cases"][label] = dict(dirac_fractions_X=wX, dirac_fractions_remainder=wR, remainder_over_X=float(np.sqrt(tR / tX)))
        print(label, "remainder/X", round(float(np.sqrt(tR / tX)), 5))
        print("  X   :", {k: round(x, 4) for k, x in wX.items() if x > 1e-3})
        print("  rem :", {k: round(x, 4) for k, x in wR.items() if x > 1e-3})
    # the energy-energy bracket alone: f = N, g = 1 + 0.3 cos(x1): is the spin term there?
    f = N.value()
    gfield = nb.Field(L, 1.0, [(0.3, (0, 1, 0), 0.5)])
    g, gg = gfield.value(), gfield.grad()
    ef, eg = nb.smear(f, ops["h0"]), nb.smear(g, ops["h0"])
    Xee = (1j * (ef @ eg - eg @ ef)).tocsr()
    u = f[:, None] * gg - g[:, None] * N.grad()
    Pu = sp.csr_matrix(Xee.shape, dtype=complex)
    for i in range(3):
        Pu = Pu + nb.smear(u[:, i], ops["Q"][i])
    Xee_t, Pu_t = nb.compress(Xee, U), nb.compress(Pu, U)
    wE, tE = dirac_weights(Xee_t, nvecs, comps, L, 2)
    wD, tD = dirac_weights(Xee_t - Pu_t, nvecs, comps, L, 2)
    out["ee_bracket"] = dict(dirac_fractions=wE, dirac_fractions_minus_Pu=wD, minus_Pu_over_X=float(np.sqrt(tD / tE)))
    print("ee bracket: (X - P[u])/X", round(float(np.sqrt(tD / tE)), 5))
    print("  X - P[u] :", {k: round(x, 4) for k, x in wD.items() if x > 1e-3})
    # spin-term prediction: (1/2) Sigma . (grad f x grad g), as a site-diagonal vertex
    cf = np.cross(N.grad(), gg)
    Sp = sp.csr_matrix(Xee.shape, dtype=complex)
    for l in range(3):
        Sp = Sp + nb.smear(0.5 * cf[:, l], sp.kron(sp.identity(L ** 3), sp.csr_matrix(SIG[l]), format="csr"))
    Sp_t = nb.compress(Sp, U)
    for sign in (+1, -1):
        wS, tS = dirac_weights(Xee_t - Pu_t - sign * Sp_t, nvecs, comps, L, 2)
        out["ee_bracket"]["minus_Pu_minus_%s_spin_over_X" % ("plus" if sign > 0 else "minus")] = float(np.sqrt(tS / tE))
        print("  (X - P[u] - (%+d) spin)/X" % sign, round(float(np.sqrt(tS / tE)), 5))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "n1_remainder_diag_v1_L%d.json" % L), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
