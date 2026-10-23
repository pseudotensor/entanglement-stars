#!/usr/bin/env python
"""test_w10_defect8_cut_locality.py (busi W10): defect 8 of REVIEWA_REVIEW_web40i section 5. SPEC_final4b_engine.md line 523's cut
locality is false on the unilateral chain (origin and outer-end traction); the fix is the sentence (W10_fix_blocks_v1.md, defect 8) and a
B_in = 0 check for localized held fillings. Gate: the B_in = 0 check.
(1) the support of i[h_bond, Q] at unit clock is exactly the first and the last bond (every other entry zero): the bond force is local at
    the ends of the chain, not at the cut; (2) a held filling localized away from the ends (the bound orbitals of a confining well on
    the chain, stationary under h_m + U_well) has B_in = (1/2) Tr D {P, i[h_bond, Q]} below 1e-12 for every interior cut, while the
    ground state of the free chain, which touches the origin, has B_in = the origin traction, not zero; (3) --slice FILE [--m 0.2]: a
    stored complex slice of the production (orbitals_<tag>_T*.npz) read for the dust's and the pairs' end tractions per channel,
    weights 16 kappa (the dust's origin traction is the B_in = 0 check on the records: zero while the dust is away from the origin).
Run: qi2 python test_w10_defect8_cut_locality.py [--slice FILE --m 0.2]"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def held_localized_filling(hs, Qs, N, v, R, center, width, nmodes):
    """the lowest nmodes positive-band bound orbitals of h_m + U_well, U_well = 2 (1 - exp(-((R - center)/width)^2)) on both components:
    stationary states of the confining operator, localized about the centre with tails below 1e-20 at the ends."""
    import scipy.sparse as sp
    U = 2.0 * (1.0 - np.exp(-((R - center) / width) ** 2))
    H = (W10.clocked_full(hs, Qs, N, v) + sp.diags(np.repeat(U, 2))).toarray()
    e, V = np.linalg.eigh(H)
    pos = np.where(e > 0)[0][:nmodes]
    return V[:, pos], np.ones(len(pos)), e[pos]


def slice_part(a):
    Z = np.load(a.slice, allow_pickle=True)
    R = np.asarray(Z["R"], float); N = np.asarray(Z["N_applied"], float)
    kappas = [int(k) for k in np.asarray(Z["kappas"])]; weights = [float(w) for w in np.asarray(Z["weights"])]
    tot = {"origin": {"dust": 0.0, "pairs": 0.0}, "outer": {"dust": 0.0, "pairs": 0.0}}
    for kap, w in zip(kappas, weights):
        hs, Qs = W10.chain_sparse_ref(R, a.m, float(kap))
        op = W10.bond_traction_operator(hs, Qs, N, W10.END_WINDOW)
        nr = op["n_rows"]
        dust = np.asarray(Z["dust_%d" % kap]); f = np.asarray(Z["f_%d" % kap], float)
        sea = np.asarray(Z["sea_%d" % kap]); ref = np.asarray(Z["R2_%d" % kap])
        o_d, u_d = W10.end_tractions(op, dust[:nr], dust[-nr:], f) if dust.shape[1] else (0.0, 0.0)
        cols = np.concatenate([sea, ref], axis=1); fp = np.concatenate([np.ones(sea.shape[1]), -np.ones(ref.shape[1])])
        o_p, u_p = W10.end_tractions(op, cols[:nr], cols[-nr:], fp)
        tot["origin"]["dust"] += w * o_d; tot["outer"]["dust"] += w * u_d; tot["origin"]["pairs"] += w * o_p; tot["outer"]["pairs"] += w * u_p
        print("  kappa %d: dust origin %.3e outer %.3e; pairs origin %.3e outer %.3e; clock-gradient entries outside the end windows %.2e" % (kap, o_d, u_d, o_p, u_p, op["max_outside"]))
    print("INFO slice %s (T/rs %.3f): the end tractions, weights 16 kappa: dust origin %.3e outer %.3e; pairs origin %.3e outer %.3e" % (
        a.slice, float(Z["T"]) / float(Z["rs"]), tot["origin"]["dust"], tot["outer"]["dust"], tot["origin"]["pairs"], tot["outer"]["pairs"]))
    check(abs(tot["origin"]["dust"]) <= 1e-6 * max(abs(tot["origin"]["pairs"]), 1e-300) or abs(tot["origin"]["dust"]) < 1e-10,
          "the stored slice's dust has B_in = origin traction %.3e (the B_in = 0 check for the localized dust; the pairs' %.3e is the polarized sea's)" % (tot["origin"]["dust"], tot["origin"]["pairs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default=None, help="a stored orbitals_<tag>_T*.npz slice of the production")
    ap.add_argument("--m", type=float, default=0.2)
    a = ap.parse_args()
    n, m, kappa = 64, 0.2, 1.0
    R = np.arange(n, dtype=float) + 1.0
    hs, Qs = W10.chain_sparse_ref(R, m, kappa)
    N = np.ones(n); v = -0.3 * np.exp(-((R - 32.0) / 8.0) ** 2)
    op = W10.bond_traction_operator(hs, Qs, N, W10.END_WINDOW)
    C = op["C"].tocoo()
    bonds = {(int(r_ // 2), int(c_ // 2)) for r_, c_, d_ in zip(C.row, C.col, C.data) if abs(d_) > 1e-15}
    check(bonds == {(0, 1), (1, 0), (n - 2, n - 1), (n - 1, n - 2)}, "the support of i[h_bond, Q] at unit clock is the first and the last bond only: site pairs %s (the force is local at the ends, not at the cut: the spec's cut-locality sentence is false on the unilateral chain)" % sorted(bonds))
    cols, f, en = held_localized_filling(hs, Qs, N, v, R, 32.0, 6.0, 6)
    hfull = W10.clocked_full(hs, Qs, N, v)
    origin, outer = W10.end_tractions(op, cols[:op["n_rows"]], cols[-op["n_rows"]:], f)
    Bin = [W10.momentum_identity_terms(hs, Qs, N, v, cols, f, cut)["force"]["bond"] for cut in range(2, n - 1)]
    check(max(abs(b) for b in Bin) < 1e-12 and abs(origin) < 1e-12 and abs(outer) < 1e-12,
          "gate (B_in = 0 for localized held fillings): 6 bound orbitals of a confining well at R 32 (energies %.3f to %.3f): B_in below %.1e at every interior cut; the end tractions %.1e, %.1e" % (en[0], en[-1], max(abs(b) for b in Bin), origin, outer))
    e, V = np.linalg.eigh(hfull.toarray())
    g = V[:, e < 0]; fg = np.ones(g.shape[1])
    og, ug = W10.end_tractions(op, g[:op["n_rows"]], g[-op["n_rows"]:], fg)
    check(abs(og) > 1e-6, "the free chain's ground state touches the origin: B_in = origin traction %.6e, not zero (the outer-end traction %.3e)" % (og, ug))
    if a.slice:
        slice_part(a)
    print("test_w10_defect8_cut_locality: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
