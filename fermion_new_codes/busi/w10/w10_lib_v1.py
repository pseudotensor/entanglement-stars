#!/usr/bin/env python
"""w10_lib_v1.py (busi W10, reviewer A; REVIEWA_REVIEW_web40i section 5, code defects 3 to 9 and 14). The helpers of the W10 fixes, each
tested in test_w10_defect<n>_*.py beside this file; imported by engine_collapse_v36e.py (defects 3, 5, 6, 7/8, 9, 14) and by the archived
readers the fix blocks name (defects 7, 9). numpy and scipy only; the subspace helpers accept torch tensors (duck-typed on .conj().T,
@ and .abs().pow(2).sum()).

  run_threads_reraise(jobs, out, where)      defect 3: threads joined, the first exception re-raised with the channels named, the keys checked
  half_bin_mask(x, thr, side, tol)           defect 5: weight 1 strictly past the threshold, 1/2 within tol of it, 0 otherwise
  ingoing_weights(k), outgoing_weights(k), literal_k_positive(k)   defect 9: the group-velocity selection s cos k over both valleys and
                                             both spinors ((1, 1)/sqrt 2: omega = sin k, v = cos k; (1, -1)/sqrt 2: omega = -sin k,
                                             v = -cos k; the chain's bond symbol is sin k sigma_x) and the literal k > 0 selector for comparison
  sorted_slot_flips(m, prev)                 defect 6: the shape-checked bit count of two sorted-slot masks (a diagnostic; None on a mismatch)
  subspace_flips(refs, prevs)                defect 6: D = sum_k (n_k - ||R_prev,k^dag R_k||_F^2), the rule
  distinct_meeting_sets(meet)                defect 6: the distinct signatures of the rounds that meet the stop rule
  chain_sparse_ref(R, m, kappa)              the chain operator and Q (page_ops_v6.chain_sparse verbatim) for the fixtures
  strip_onsite, clocked, clocked_full, bond_traction_operator, end_tractions, momentum_identity_terms   defects 7 and 8
  e_prev_rule(fixes, held_ok, improved)      defect 14 (b)
  cheb_debye_phi, cheb_tail_bound, cheb_tail, cheb_order   defect 4 (the same code as page_ops_v7's block; the test checks them equal)
"""
import math
import threading

import numpy as np
import scipy.sparse as sp
from scipy.special import jv

END_WINDOW = 4   # sites per end window of the bond traction (at unit clock the commutator i[h_bond, Q] lives on the first and the last bond)


# ------------------------------------------------------------------------------------------------------------------ defect 3
def run_threads_reraise(jobs, out, where="threads"):
    """jobs: {key: callable()} (each callable writes out[key] itself); every job runs in its own thread and all are joined; then the
    first exception is re-raised (threading.Thread swallows the target's exception: engine_collapse_v36d.eigh_all returned a partial
    dictionary and counted every channel) with the failed and the missing keys named, and the output is checked to hold every key.
    Returns the number of completed keys."""
    errors = {}

    def wrap(k, fn):
        try:
            fn()
        except BaseException as ex:  # noqa: BLE001 (re-raised after the join)
            errors[k] = ex
    th = [threading.Thread(target=wrap, args=(k, fn)) for k, fn in jobs.items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    missing = [k for k in jobs if k not in out]
    if errors:
        k0 = next(iter(errors))
        raise RuntimeError("%s: %d of %d threads failed (channels %s; the output lacks %s); the first: %s: %s" % (
            where, len(errors), len(jobs), sorted(errors, key=str), sorted(missing, key=str), type(errors[k0]).__name__, errors[k0])) from errors[k0]
    if missing:
        raise RuntimeError("%s: the output lacks the channels %s although no thread raised" % (where, sorted(missing, key=str)))
    return len(out)


# ------------------------------------------------------------------------------------------------------------- defects 5, 9
def half_bin_mask(x, thr, side, tol=1e-12):
    """weights over the bins: 1 where x is strictly past the threshold (side 'below': x < thr - tol; 'above': x > thr + tol), 1/2 where
    |x - thr| <= tol (the bin sits on the threshold: the half-bin convention), 0 otherwise. A strict mask assigns the boundary bins 0 or 1
    by the rounding of the test quantity (np.sin(-np.pi) = -1.2e-16), so a k-symmetric spectrum does not read 1/2."""
    x = np.asarray(x, float)
    on = np.abs(x - thr) <= tol
    if side == "below":
        strict = x < thr - tol
    elif side == "above":
        strict = x > thr + tol
    else:
        raise ValueError("side must be 'below' or 'above', not %r" % (side,))
    return np.where(strict, 1.0, np.where(on, 0.5, 0.0))


def ingoing_weights(k, tol=1e-12):
    """(w_plus, w_minus): the ingoing (group velocity < 0) weights per momentum bin on the (1, 1)/sqrt 2 spinor (v = cos k: ingoing where
    cos k < 0, |k| > pi/2) and on the (1, -1)/sqrt 2 spinor (v = -cos k: ingoing where cos k > 0, |k| < pi/2); the bins at cos k = 0 weigh 1/2."""
    c = np.cos(np.asarray(k, float))
    return half_bin_mask(c, 0.0, "below", tol), half_bin_mask(c, 0.0, "above", tol)


def outgoing_weights(k, tol=1e-12):
    """(w_plus, w_minus): the outgoing (group velocity > 0) weights: the (1, 1) spinor at cos k > 0 (the first valley) and the (1, -1)
    spinor at cos k < 0 (the second valley, the lattice branch); the selector of both outgoing valleys, s cos k > 0, never k > 0."""
    c = np.cos(np.asarray(k, float))
    return half_bin_mask(c, 0.0, "above", tol), half_bin_mask(c, 0.0, "below", tol)


def literal_k_positive(k):
    """the literal k > 0 selector of the quoted spec line (for the comparison in the tests)."""
    return (np.asarray(k, float) > 0.0).astype(float)


# ------------------------------------------------------------------------------------------------------------------ defect 6
def sorted_slot_flips(m, prev):
    """the number of sorted-slot bits that differ between two masks of the same shape; None when either is None or the shapes differ
    (numpy would broadcast (2, 1) against (2,) and count a (2, 2) comparison: v36d's .size test admitted that). A diagnostic only: the
    sorted slot of an eigenvector changes at every level crossing, so this count is not a distance between occupied sets."""
    if m is None or prev is None:
        return None
    m = np.asarray(m, bool); prev = np.asarray(prev, bool)
    if m.shape != prev.shape:
        return None
    return int(np.sum(m != prev))


def _abs2sum(X):
    if hasattr(X, "abs") and hasattr(X, "pow"):
        return float(X.abs().pow(2).sum())
    return float(np.sum(np.abs(np.asarray(X)) ** 2))


def subspace_flips(refs, prevs):
    """D = sum_k (n_k - ||R_prev,k^dag R_k||_F^2) over the channels k: R_k the occupied columns (orthonormal) of an evaluation's continued
    reference, R_prev,k the previous record's; per channel this is ||P_k - P_prev,k||_F^2 / 2, 0 for the same occupied subspace, the
    number of exchanged orbitals when the set is the old one with orbitals exchanged, continuous through level crossings. The rule of
    defect 6: the set with the smaller D is the previous record's set continued. refs and prevs: {k: array or tensor (n2 x n_k)}."""
    D = 0.0
    for k in refs:
        R = refs[k]; P = prevs[k]
        n = int(R.shape[1])
        if n == 0:
            continue
        D += n - _abs2sum(P.conj().T @ R)
    return float(D)


def distinct_meeting_sets(meet, key="held_sig"):
    """the sorted distinct signatures of the rounds in meet (v36d's len(meet) > 1 counted rounds, not sets)."""
    return sorted({str(r[key]) for r in meet})


# ------------------------------------------------------------------------------------------------------------ defects 7, 8
def chain_sparse_ref(R, m, kappa):
    """page_ops_v6.chain_sparse verbatim: h_0 (unit clock, no flow) and the two-step Q as host CSR matrices (2n x 2n): on-site m sigma_z
    + (kappa/R_n) sigma_y, the sine bond -(i/2) sigma_x from site n to n + 1 (symbol sin k sigma_x), Q = -(i/4) 1 from site n to n + 2
    (symbol sin(2k)/2). Flattened index 2 n + a."""
    R = np.asarray(R, float)
    n = len(R)
    even = 2 * np.arange(n)
    rows = np.concatenate([even, even + 1, even, even + 1])
    cols = np.concatenate([even, even + 1, even + 1, even])
    vals = np.concatenate([np.full(n, m, complex), np.full(n, -m, complex), -1j * kappa / R, 1j * kappa / R])
    b = 2 * np.arange(n - 1)
    rows = np.concatenate([rows, b, b + 3, b + 1, b + 2])
    cols = np.concatenate([cols, b + 3, b, b + 2, b + 1])
    vals = np.concatenate([vals, np.full(n - 1, -0.5j), np.full(n - 1, 0.5j), np.full(n - 1, -0.5j), np.full(n - 1, 0.5j)])
    h0 = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    t = np.arange(2 * n - 4)
    Q = sp.csr_matrix((np.concatenate([np.full(2 * n - 4, -0.25j), np.full(2 * n - 4, 0.25j)]),
                       (np.concatenate([t, t + 4]), np.concatenate([t + 4, t]))), shape=(2 * n, 2 * n))
    return h0, Q


def strip_onsite(hs):
    """the sine-bond part of the channel's host CSR chain operator: the entries within a site's own 2 x 2 block (the mass and the
    kappa/R terms) removed."""
    C = sp.coo_matrix(hs)
    keep = (C.row // 2) != (C.col // 2)
    return sp.csr_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=C.shape)


def clocked(A, N):
    """(1/2){N, A} as a host CSR for a site field N (repeated on the 2 components)."""
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2))
    return (0.5 * (Nd @ A + A @ Nd)).tocsr()


def clocked_full(hs, Qs, N, v):
    """h_m = (1/2){N, h} + (1/2){v, Q} (engine_collapse_v36d.clocked_host_sparse's operator)."""
    return (clocked(hs, N) + clocked(Qs, v)).tocsr()


def bond_traction_operator(hs, Qs, N, w=END_WINDOW):
    """i[h_bond, Q] with h_bond = (1/2){N, h_sine} (the chain's sine bonds, the on-site terms stripped) as a host CSR, its two end windows
    (the first and the last w sites: 2w rows each, dense) and the largest entry outside them. At unit clock the commutator is exactly the
    two end bonds, (i/8) sigma_x (|N-2><N-1| - |N-1><N-2| - |0><1| + |1><0|) (the range-3 terms of h Q and Q h cancel pairwise on the whole
    chain and the range-1 terms cancel except at the first and the last bond: test_w10_defect7 checks the formula numerically); with a
    nonuniform clock the bulk entries are the clock-gradient force of the rows, reported as max_outside."""
    hb = clocked(strip_onsite(hs), N)
    C = (1j * (hb @ Qs - Qs @ hb)).tocsr()
    n2 = C.shape[0]; nr = 2 * int(w)
    C0 = C[:nr, :nr].toarray(); C1 = C[n2 - nr:, n2 - nr:].toarray()
    Cc = C.tocoo()
    inside = ((Cc.row < nr) & (Cc.col < nr)) | ((Cc.row >= n2 - nr) & (Cc.col >= n2 - nr))
    mo = float(np.max(np.abs(Cc.data[~inside]))) if np.any(~inside) else 0.0
    return {"C": C, "origin": C0, "outer": C1, "n_rows": nr, "max_outside": mo, "window_sites": int(w)}


def end_tractions(op, cols_origin, cols_outer, f):
    """(origin, outer): Re sum_j f_j c_j[W]^dag C_W c_j[W] on each end window W of op (bond_traction_operator); for an interior cut the
    cut's projector is 1 on the origin window and 0 on the outer one, so (1/2) Tr D {P, C} is the origin term and the outer term is the
    traction of the complementary region. cols_*: the columns' rows on the window (n_rows x k), f the occupations (k,)."""
    f = np.asarray(f, float)

    def tr(Cw, X):
        X = np.asarray(X)
        if X.shape[1] == 0:
            return 0.0
        return float(np.real(np.einsum("ij,ik,kj,j->", X.conj(), Cw, X, f)))
    return tr(op["origin"], cols_origin), tr(op["outer"], cols_outer)


def momentum_identity_terms(hs, Qs, N, v, cols, f, cut):
    """every term of the enclosed-momentum identity on a fixture (dense numpy; the cut's projector P is 1 on the sites with index below
    cut): M_in = Tr D (1/2){P, Q}; rate_exact = Tr D i[h_m, (1/2){P, Q}]; per part A of h_m = (1/2){N, h_onsite} + (1/2){N, h_bond} +
    (1/2){v, Q}: face[A] = (1/2) Tr D {i[A, P], Q} and force[A] = (1/2) Tr D {P, i[A, Q]}; sum_terms = sum of all faces and forces (equal
    to rate_exact exactly); archived = sum_terms - force["bond"] (the face flux and the on-site and flow forces: what a reader assembles
    from the archive without the end traction); defect_without_bond_force = rate_exact - archived = force["bond"], the origin traction
    for an interior cut."""
    n2 = hs.shape[0]
    P = np.zeros(n2); P[: 2 * int(cut)] = 1.0
    Pd = sp.diags(P)
    hb0 = strip_onsite(hs); ho0 = (hs - hb0).tocsr()
    parts = {"onsite": clocked(ho0, N), "bond": clocked(hb0, N), "flow": clocked(Qs, v)}
    X = np.asarray(cols); f = np.asarray(f, float)

    def expect(A):
        AX = A @ X
        return float(np.real(np.einsum("ij,ij,j->", X.conj(), AX, f)))
    M = 0.5 * (Pd @ Qs + Qs @ Pd)
    htot = parts["onsite"] + parts["bond"] + parts["flow"]
    out = {"cut": int(cut), "M_in": expect(M), "rate_exact": expect(1j * (htot @ M - M @ htot)), "face": {}, "force": {}}
    for nm, A in parts.items():
        iAP = 1j * (A @ Pd - Pd @ A); iAQ = 1j * (A @ Qs - Qs @ A)
        out["face"][nm] = expect(0.5 * (iAP @ Qs + Qs @ iAP))
        out["force"][nm] = expect(0.5 * (Pd @ iAQ + iAQ @ Pd))
    out["sum_terms"] = sum(out["face"].values()) + sum(out["force"].values())
    out["archived"] = out["sum_terms"] - out["force"]["bond"]
    out["defect_without_bond_force"] = out["rate_exact"] - out["archived"]
    return out


# ----------------------------------------------------------------------------------------------------------------- defect 14
def e_prev_rule(fixes, held_ok, improved):
    """defect 14 (b): whether a Newton evaluation moves the solve's E_prev (the reference of the history's |dE/E|): without --v36-fixes at
    every evaluation (version 32: line-search trials and finite-difference evaluations included), with it only at an evaluation on the
    held set that becomes the best iterate (version 36 (b))."""
    if not fixes:
        return True
    return bool(held_ok and improved)


# ------------------------------------------------------------------------------------------------------------------ defect 4
def cheb_debye_phi(j, z):
    """the exponent of the Bessel contour bound: for j > z >= 0, |J_j(z)| <= exp(-phi) with phi = j arccosh(j/z) - sqrt(j^2 - z^2) (the
    Cauchy integral of the generating function on |t| = rho at the saddle rho = (j + sqrt(j^2 - z^2))/z); 0 for j <= z; increasing and
    convex in j (d phi/dj = arccosh(j/z))."""
    j = np.asarray(j, float)
    z = float(z)
    if z <= 0.0:
        return np.where(j > 0.0, np.inf, 0.0)
    r = np.maximum(j / z, 1.0)
    return j * np.arccosh(r) - np.sqrt(np.maximum(j * j - z * z, 0.0))


def cheb_tail_bound(z, K):
    """a rigorous upper bound on 2 sum_{j > K} |J_j(|z|)|: by the convexity of phi every later increment is at least d = phi(K + 2) -
    phi(K + 1), so the sum is at most 2 e^(-phi(K + 1)) / (1 - e^(-d)); inf where K + 1 <= |z|; 0 at z = 0."""
    az = abs(float(z))
    if az == 0.0:
        return 0.0
    if K + 1 <= az:
        return math.inf
    p1 = float(cheb_debye_phi(K + 1, az)); p2 = float(cheb_debye_phi(K + 2, az))
    d = p2 - p1
    if not (d > 0.0):
        return math.inf
    return 2.0 * math.exp(-p1) / (-math.expm1(-d))


def cheb_tail(z, K, rel=1e-6):
    """the tail 2 sum_{j > K} |J_j(|z|)| as the computed sum up to j_max plus the contour bound of the remainder beyond j_max, j_max
    extended from |z| + 120 until that remainder is below rel times the sum (or below 1e-300). Returns (tail, remainder bound, j_max)."""
    az = abs(float(z))
    K = int(K)
    jmax = max(int(az + 120), K)
    while True:
        js = np.arange(K + 1, jmax + 1)
        partial = 2.0 * float(np.sum(np.abs(jv(js, az))[::-1])) if js.size else 0.0
        rem = cheb_tail_bound(az, jmax)
        if rem <= max(rel * partial, 1e-300) or jmax > az + 200000:
            return partial + rem, rem, jmax
        jmax = int(1.5 * jmax) + 120


def cheb_order(z, tol):
    """page_ops_v7.cheb_order (the same code): the smallest K >= |z| + 4 with 2 sum_{j > K} |J_j(|z|)| <= tol, the tail the computed sum
    plus the bounded remainder beyond the summation limit, which is extended until the remainder is below 0.1 tol."""
    az = abs(float(z))
    K = int(math.ceil(az + 4))
    jmax = int(az + 120)
    while True:
        J = np.abs(jv(np.arange(jmax + 1), az))
        tail = 2.0 * np.concatenate([np.cumsum(J[::-1])[::-1][1:], [0.0]])
        rem = cheb_tail_bound(az, jmax)
        while K < jmax and tail[K] + rem > tol:
            K += 1
        if K < jmax and rem <= 0.1 * tol:
            return K, float(tail[K] + rem)
        jmax = int(1.5 * jmax) + 120
