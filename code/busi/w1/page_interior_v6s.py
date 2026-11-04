#!/usr/bin/env python
"""page_interior_v6s.py [v6s verify] (busi lane, inbox #309, the Page-curve verification plan final4/PAGE_VERIFY_PLAN.md; reviewer A's
LEAD_reviewa_page_verify_routes_v1.md A2, A3, B2, B6). A new module imported only by page_history_v6s.py (= page_history_v6q.py, md5 b4f4ea69,
plus build_page_history_v6s.py's marked options); nothing shared is changed by it. Lattice units a = 1, torch complex128, page_ops_v6's chain
conventions (sites j = 0 ... n - 1, R_j = j + 1, the + sector coordinates of page_w1_sector_v1d: site j <-> W_+ column j).

1. The interior-side readout (A3 purity, B6). For the pure + sector state C_+ = Phi_+ Phi_+^dagger and the held vacuum P_{-,+} = 1(h_+ < 0), every
   exterior cut X = {R > x} has S_X(C_+) = S_I(C_+) and S_X(P_{-,+}) = S_I(P_{-,+}) with I = {R <= x} the interior block (the complement of a pure
   Gaussian state's block has the same spectrum up to 0 <-> 1). The excess S_X(C) - S_X(P_-) = 2 [S_I(C_+) - S_I(P_{-,+})] is read on the N_I-site
   interior block alone. The state's block is Phi_+[I] Phi_+[I]^dagger (N_I x N_I). The vacuum's block E_I^dagger P_{-,+} E_I = (1 - E_I^dagger
   sign(h_+) E_I)/2 is computed without any eigendecomposition by the contour (resolvent) integral along the imaginary axis,
       sign(h) = (2/pi) int_0^inf h (h^2 + t^2)^(-1) dt = (2/pi) int ds t Herm[(h - i t)^(-1)],   t = e^s,
   on a trapezoid grid in s (step hs; the integrand (1/2) sech(s - ln|e|) is analytic in the strip |Im s| < pi/2, so the error is about
   2 exp(-pi^2/hs), 5e-15 at hs = 0.3; s from ln(delta) - U to ln(w) + U with delta the smallest |e| (shift-invert Lanczos) and w the
   Gershgorin bound, the truncated tails below e^(-U)). Each node needs only the interior block of the resolvent, by the Schur complement on the
   banded h_+ (half-bandwidth 2): G_II(z) = (A - z - Sigma(z))^(-1), Sigma(z) = B (D - z)^(-1) B^dagger the exterior's self-energy, a 2 x 2 corner
   from a banded solve with 2 right-hand sides on the exterior; then one banded solve of size N_I with N_I right-hand sides. The nodes run on host
   threads (LAPACK zgbsv releases the GIL); a check sum from the even nodes at step 2 hs prints beside (the quadrature's own error estimate).
2. The lean interior mode (B6): the + sector state propagated by the production's linear sub-steps (no per-sub-step traces, flows or jumps; the
   collar current Tr[J C] once per record), held records (--hold-fields-from) propagated in large Chebyshev steps of --hold-step-rs (the same
   polynomial, the order by the Bessel tail); every step by propagate_lean on BandedLean (page_ops_v6.Banded's diagonals applied in place, four
   reusable buffers per column chunk: 1.7 times page_history's propagate on the A6000, equal to it to 7e-16 in the self-test's S10); the readouts by the interior side only (no assembled full columns, no dense vacuum, no D13 store),
   the initial vacuum Phi_+ = V_+[:, e_+ < 0] by one dense eigendecomposition of h_+(T_0) on the host when the device cannot hold it (cached).
3. The hold (B2): the history truncated at the record of --hold-fields-from and continued by copies of that record (N, v, E) to --T-end-rs.
4. kappa = 0 (A2's conformal control): _mass_view for page_w1_sector_v1d.fast_banded_matches (the field-built banded operator built with the
   channel's mass coefficient, 0 under --kappa-zero; the channel object itself when the coefficient is kappa's, so the defaults are v6q's).
"""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import torch

import page_ops_v6 as OPS
import page_w1_sector_v1d as W1S

CD = OPS.CD
RD = OPS.RD
CODE = "page_history_v6s.py"


def _log_default(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------------------------ kappa = 0
class _MassView:
    """the attributes page_w1_sector_v1d.fast_banded_matches reads (kappa, R, h_csr), with kappa the channel's mass coefficient."""

    def __init__(self, ops):
        self.kappa = ops.kappa_mass; self.R = ops.R; self.h_csr = ops.h_csr


def mass_view(ops):
    km = getattr(ops, "kappa_mass", None)
    if km is None or km == float(ops.kappa):
        return ops
    return _MassView(ops)


# ------------------------------------------------------------------------------------------------------------------- the hold
def hold_extend(H, T_hold_abs, T_end_abs, tau, log=_log_default):
    """the history H (page_history's History) truncated at the record of the clock T_hold_abs and continued by copies of that record (its fields
    N_e, v_e, its energy E and its record dict) at the records' spacing tau up to T_end_abs. Returns the hold's description."""
    if H.ext is not None or H.follow:
        raise ValueError("--hold-fields-from is not combined with a continuation (--record-ext/--follow)")
    k_h = int(np.argmin(np.abs(H.T - T_hold_abs)))
    if abs(H.T[k_h] - T_hold_abs) > 1e-6 * max(1.0, tau):
        raise ValueError("--hold-fields-from: no record at the clock %g (the nearest is %g)" % (T_hold_abs, H.T[k_h]))
    n_old = len(H.T)
    n_new = max(0, int(math.ceil((T_end_abs - H.T[k_h]) / tau - 1e-9)))
    T_new = H.T[k_h] + tau * np.arange(1, n_new + 1, dtype=float)
    H.T = np.concatenate([H.T[: k_h + 1], T_new])
    H.N_e = np.concatenate([H.N_e[: k_h + 1], np.repeat(H.N_e[k_h: k_h + 1], n_new, axis=0)], axis=0)
    H.v_e = np.concatenate([H.v_e[: k_h + 1], np.repeat(H.v_e[k_h: k_h + 1], n_new, axis=0)], axis=0)
    H.E = np.concatenate([H.E[: k_h + 1], np.full(n_new, H.E[k_h])])
    H.rec = list(H.rec[: k_h + 1]) + [H.rec[k_h]] * n_new
    H.n_primary = len(H.T)
    info = {"k_hold": k_h, "T_hold": float(H.T[k_h]), "records_before": n_old, "records_dropped_after_hold": int(n_old - k_h - 1), "records_added": int(n_new),
            "T_last": float(H.T[-1]), "E_hold": float(H.E[k_h]),
            "label": "the fields (N, v) and E of the record at T_hold held for every later record (copies at the spacing tau); the propagation across each held record is exact in the held h"}
    H.hold = info
    log("v6s verify: the fields held from record %d (T %.4f) to T %.4f: %d copies appended, %d stored records after the hold dropped; E held %.10f" % (
        k_h, H.T[k_h], H.T[-1], n_new, n_old - k_h - 1, H.E[k_h]))
    return info


# ----------------------------------------------------------------------------------------------------- the contour block of P_-
def band_general(A, l, u):
    """LAPACK general band storage of the CSR A for scipy.linalg.solve_banded((l, u), ab, b): ab[u + i - j, j] = A[i, j]."""
    co = A.tocoo(); n = A.shape[0]
    if co.nnz and (int((co.row - co.col).max()) > l or int((co.col - co.row).max()) > u):
        raise ValueError("the operator's bandwidth exceeds (%d, %d)" % (l, u))
    ab = np.zeros((l + u + 1, n), dtype=complex)
    np.add.at(ab, (u + co.row - co.col, co.col), co.data)
    return ab


def smallest_abs_eig(hp, fallback):
    """the smallest |e| of the sparse Hermitian h_+ by shift-invert Lanczos at 0 (two eigenvalues); fallback when the factorization fails
    (a singular h_+: an exact zero level)."""
    try:
        e = spla.eigsh(hp.tocsc(), k=2, sigma=0.0, which="LM", return_eigenvectors=False)
        return float(np.min(np.abs(e))), "shift-invert Lanczos at 0 (scipy eigsh, 2 levels)"
    except Exception as ex:                                                     # noqa: BLE001
        return float(fallback), "fallback %g (%s: %s)" % (fallback, type(ex).__name__, str(ex)[:120])


def sign_block_contour(hp, nI, hs=0.3, U=37.0, threads=8, delta=None):
    """the interior block E_I^dagger sign(h_+) E_I (sites 0 ... nI - 1) of the banded Hermitian h_+ (half-bandwidth 2) by the imaginary-axis
    resolvent integral with the Schur complement on the exterior; returns (P_II = (1 - sign_II)/2 as a host complex array, info)."""
    t0 = time.time()
    hp = hp.tocsr(); n = hp.shape[0]; l = u = 2
    if not (3 <= nI <= n - 3):
        raise ValueError("the interior block needs 3 <= N_I <= n - 3 (N_I %d, n %d)" % (nI, n))
    w = OPS.gershgorin(hp)
    if delta is None:
        delta, droute = smallest_abs_eig(hp, 1e-13 * w)
    else:
        droute = "given"
    delta_eff = max(float(delta), 1e-15 * w)
    A = hp[:nI, :nI]; D = hp[nI:, nI:]; B = hp[:nI, nI:].tocoo()
    if B.nnz and (int(B.row.min()) < nI - 2 or int(B.col.max()) > 1):
        raise ValueError("the interior-exterior coupling is not confined to the 2 x 2 corner (bandwidth > 2)")
    Bc = hp[nI - 2: nI, nI: nI + 2].toarray()                                 # the coupling's corner (rows nI-2, nI-1; exterior sites 0, 1)
    R = np.zeros((n - nI, 2), dtype=complex); R[:2, :] = Bc.conj().T          # B^dagger's two nonzero columns
    abA0 = band_general(A, l, u); abD0 = band_general(D, l, u)
    s_lo = math.log(delta_eff) - U; s_hi = math.log(w) + U
    m = int(math.ceil((s_hi - s_lo) / hs)) + 1
    s_nodes = s_lo + hs * np.arange(m)
    I_rhs = np.eye(nI, dtype=complex)

    def node_terms(js):
        acc = np.zeros((nI, nI), dtype=complex); acc_e = np.zeros((nI, nI), dtype=complex)
        for j in js:
            tj = math.exp(s_nodes[j]); z = 1j * tj
            abD = abD0.copy(); abD[u, :] -= z
            Y = sla.solve_banded((l, u), abD, R, check_finite=False)
            Sig = Bc @ Y[:2, :]                                                  # Sigma(z) = B (D - z)^(-1) B^dagger on the corner
            abA = abA0.copy(); abA[u, :] -= z
            for a_ in range(2):
                for b_ in range(2):
                    i_ = nI - 2 + a_; j_ = nI - 2 + b_
                    abA[u + i_ - j_, j_] -= Sig[a_, b_]
            G = sla.solve_banded((l, u), abA, I_rhs, check_finite=False)
            Gh = 0.5 * (G + G.conj().T) * tj
            acc += Gh
            if j % 2 == 0:
                acc_e += Gh
        return acc, acc_e

    nth = max(1, int(threads))
    chunks = [list(range(i, m, nth)) for i in range(nth)] if nth > 1 else [list(range(m))]
    chunks = [c_ for c_ in chunks if c_]
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            parts = list(ex.map(node_terms, chunks))
    else:
        parts = [node_terms(chunks[0])]
    S = np.zeros((nI, nI), dtype=complex); S_e = np.zeros((nI, nI), dtype=complex)
    for acc, acc_e in parts:                                                     # a fixed order: the chunks are fixed by (m, threads)
        S += acc; S_e += acc_e
    sgn = (2.0 / math.pi) * hs * S
    sgn_e = (2.0 / math.pi) * (2.0 * hs) * S_e
    sgn = 0.5 * (sgn + sgn.conj().T)
    P = 0.5 * (np.eye(nI) - sgn)
    info = {"route": "contour: sign(h_+) = (2/pi) int ds t Herm[(h_+ - i t)^(-1)], t = e^s, trapezoid step %g over s in [ln delta - %g, ln w + %g] (%d nodes); "
                     "the interior block of each resolvent by the Schur complement on the exterior (banded solves, half-bandwidth 2)" % (hs, U, U, m),
            "n": int(n), "N_I": int(nI), "nodes": int(m), "hs": float(hs), "U": float(U), "w_gershgorin": float(w), "delta": float(delta), "delta_route": droute,
            "quad_check_2hs_max": float(np.max(np.abs(sgn - 0.5 * (sgn_e + sgn_e.conj().T)))),
            "label_quad_check": "max |sign_II(hs) - sign_II(2 hs)|: the error of the coarser grid, a conservative bound on the used grid's (whose error is about its square)",
            "threads": int(nth), "seconds": time.time() - t0}
    return P, info


_CONTOUR_CACHE = {}


def sign_block_contour_cached(hp, nI, hs=0.3, U=37.0, threads=8):
    """sign_block_contour memoized on (the md5 of h_+'s CSR arrays, N_I, hs, U, threads): a held record's readouts reuse the block (the held h_+
    does not change); one entry kept. The returned info carries 'cached'."""
    key = (_hp_md5(hp), int(nI), float(hs), float(U), int(threads))
    if key in _CONTOUR_CACHE:
        P, info = _CONTOUR_CACHE[key]
        return P, dict(info, cached=True, seconds_this_call=0.0)
    P, info = sign_block_contour(hp, nI, hs=hs, U=U, threads=threads)
    _CONTOUR_CACHE.clear(); _CONTOUR_CACHE[key] = (P, info)
    return P, dict(info, cached=False, md5_h_plus=key[0])


# ----------------------------------------------------------------------------------------------------------- the readout
def _ent(ev):
    return float(OPS.binary_entropy_t(ev).sum()) if ev.numel() else 0.0


def _exc(ev):
    if ev.numel() == 0:
        return 0.0
    return max(0.0, float(-ev.min()), float(ev.max()) - 1.0)


def interior_readout(Pp, hp, cuts, R_sites, device, threads=8, hs=0.3, U=37.0, floor=None, vac=None, sc_cuts=None, check=False, log=_log_default,
                     kappa=None, trs=None):
    """the excess at every cut (label, R_abs) on the interior side: S_X(C) = 2 S_I(C_+) from Phi_+'s interior rows, S_X(P_-) = 2 S_I(P_{-,+}) from
    the contour block, the excess and (floor given: page_w1_sector_v1d.propagation_floor's dict with 'q') its propagation floors. With check and a
    dense vacuum (vac with V_plus, e_plus) the purity identity's sides beside: the state's exterior block S_E(C), the dense vacuum's interior and
    exterior blocks S_I^dense(P), S_E^dense(P), the contour block's largest entry difference from the dense one, and (sc_cuts, the D13 readout's
    per-cut dict) the production readout's S_X, S_X_vac and excess."""
    t0 = time.time()
    Rs = np.asarray(R_sites, float)
    nC = {lab: int(np.sum(Rs <= x)) for lab, x in cuts}
    nI = max(nC.values())
    P_np, cinfo = sign_block_contour_cached(hp, nI, hs=hs, U=U, threads=threads)
    PT = torch.as_tensor(P_np, dtype=CD, device=device); del P_np
    Pi = Pp[:nI]
    Gc = Pi @ Pi.mH
    out = {}
    for lab, x in cuts:
        m = nC[lab]
        GC = Gc[:m, :m]; PM = PT[:m, :m]
        evC = torch.linalg.eigvalsh(0.5 * (GC + GC.mH)); evP = torch.linalg.eigvalsh(0.5 * (PM + PM.mH))
        SC = 2.0 * _ent(evC); SP = 2.0 * _ent(evP)
        out[lab] = {"R_abs": float(x), "n_sites_I": int(m), "S_I_C": SC, "S_I_P": SP, "excess": SC - SP, "excursion_C": _exc(evC), "excursion_P": _exc(evP),
                    "label": "S_X(C) = 2 S_I(C_+) and S_X(P_-) = 2 S_I(P_{-,+}) on the interior block R <= x (purity); P_{-,+}'s block by the contour"}
    if floor:
        W1S.excess_floors(out, Rs, cuts, floor, sector=True)
    chk = None
    if check and vac is not None and vac.get("V_plus") is not None:
        ep = torch.as_tensor(np.asarray(vac["e_plus"], float), dtype=RD, device=device)
        Vp = vac["V_plus"][:, ep < 0]
        Pd = Vp[:nI] @ Vp[:nI].mH
        chk = {"contour_minus_dense_block_max": float((Pd - PT).abs().max()), "cuts": {}}
        for lab, x in cuts:
            m = nC[lab]
            evId, _ = W1S.gram_eigs(Vp[:m]); evEd, _ = W1S.gram_eigs(Vp[m:]); evEC, _ = W1S.gram_eigs(Pp[m:])
            SId = 2.0 * _ent(evId); SEd = 2.0 * _ent(evEd); SEC = 2.0 * _ent(evEC)
            d = out[lab]
            cc = {"S_E_C": SEC, "S_I_P_dense": SId, "S_E_P_dense": SEd, "excess_exterior_dense": SEC - SEd,
                  "state_I_minus_E": d["S_I_C"] - SEC, "vacuum_dense_I_minus_E": SId - SEd, "vacuum_contour_minus_dense_I": d["S_I_P"] - SId,
                  "excess_interior_minus_exterior": d["excess"] - (SEC - SEd)}
            if sc_cuts and isinstance(sc_cuts.get(lab), dict):
                q = sc_cuts[lab]
                for key in ("S_X", "S_X_vac", "excess", "S_X_both_Xblocks"):
                    if q.get(key) is not None:
                        cc["production_" + key] = float(q[key])
                if q.get("excess") is not None:
                    cc["excess_interior_minus_production"] = d["excess"] - float(q["excess"])
            chk["cuts"][lab] = cc
        del Pd, Vp
    del Gc, PT
    res = {"cuts": out, "contour": cinfo, "check": chk, "seconds": time.time() - t0, "code": CODE,
           "label": "v6s verify item 2: the interior-side readout (A3 purity, B6): the excess 2 [S_I(C_+) - S_I(P_{-,+})] on the N_I-site interior block"}
    line = "; ".join("%s r_s: excess %.6e (S_I(C) %.6f, S_I(P) %.6f%s)" % (lab, out[lab]["excess"], out[lab]["S_I_C"], out[lab]["S_I_P"],
                     (", I-E state %.1e, contour-dense %.1e, interior-production %.1e" % (chk["cuts"][lab]["state_I_minus_E"], chk["cuts"][lab]["vacuum_contour_minus_dense_I"],
                      chk["cuts"][lab].get("excess_interior_minus_production", float("nan")))) if chk else "") for lab, _ in cuts)
    log("v6s verify kappa %s T/r_s %s: interior readout (%d contour nodes, delta %.2e, quad check %.1e, %.1f s): %s" % (
        kappa, ("%.3f" % trs) if trs is not None else "-", cinfo["nodes"], cinfo["delta"], cinfo["quad_check_2hs_max"], res["seconds"], line))
    return res


# ----------------------------------------------------------------------------------------------------------- the lean mode
class BandedLean:
    """page_ops_v6.Banded's operator (the same diagonals from the same CSR, the same Gershgorin bound) with an in-place apply into a given buffer
    (torch.mul and addcmul_, no temporaries): the lean mode's propagator. The sums run in another order than Banded.apply's (the main diagonal
    first), so the result equals Banded's to rounding, not bit for bit (the self-test's S10)."""

    def __init__(self, A, device):
        B = OPS.Banded(A, device)
        self.n2 = B.n2; self.bound = B.bound
        self.d0 = None; self.off = []
        for d, D in B.diags:
            if d == 0:
                self.d0 = D[:, None]
            else:
                self.off.append((d, D[:, None]))

    def apply_into(self, X, Y):
        n = self.n2
        if self.d0 is not None:
            torch.mul(X, self.d0, out=Y)
        else:
            Y.zero_()
        for d, D in self.off:
            if d > 0:
                Y[: n - d].addcmul_(D, X[d:])
            else:
                Y[-d:].addcmul_(D, X[: n + d])
        return Y


def propagate_lean(B, X, tau, tol, chunk):
    """exp(-i tau h) X in place over column chunks by the Chebyshev series on [-w, w] (page_ops_v6.cheb_order's degree and tail, the coefficients
    2 (-i)^k J_k(w tau)) with four reusable buffers per chunk width and in-place recursions; returns (X, K, tail) as page_history's propagate."""
    from scipy.special import jv
    if X.shape[1] == 0:
        return X, 0, 0.0
    w = B.bound; z = w * tau
    K, tail = OPS.cheb_order(z, tol)
    k = np.arange(K + 1)
    ck = jv(k, z) * ((-1j) ** k) * np.where(k == 0, 1.0, 2.0)
    m = min(int(chunk), X.shape[1])
    bufs = [torch.empty((X.shape[0], m), dtype=X.dtype, device=X.device) for _ in range(4)]
    for j0 in range(0, X.shape[1], m):
        Xc = X[:, j0: j0 + m]; mm = Xc.shape[1]
        T1, Y, b2, b3 = (b_[:, :mm] for b_ in bufs)
        B.apply_into(Xc, T1); T1.mul_(1.0 / w)
        torch.mul(Xc, complex(ck[0]), out=Y); Y.add_(T1, alpha=complex(ck[1]))
        prev, cur, nxt, spare = Xc, T1, b2, b3
        for kk in range(2, K + 1):
            B.apply_into(cur, nxt); nxt.mul_(2.0 / w).sub_(prev)
            Y.add_(nxt, alpha=complex(ck[kk]))
            if prev is Xc:
                prev, cur, nxt = cur, nxt, spare
            else:
                prev, cur, nxt = cur, nxt, prev
        Xc.copy_(Y)
    del bufs
    return X, K, tail


def _dense_init_plus(hp, device, eig_device, threads, log):
    """Phi_+ = V_+[:, e_+ < 0] of the dense h_+ (eigh on eig_device), moved to device; (Phi_+, info)."""
    t0 = time.time()
    if eig_device.type == "cpu":
        # the host route: LAPACK zheevr for the negative levels only (scipy, subset_by_value (-inf, 0]) on the F-ordered view of A^T = conj(A)
        # (no copy; its eigenvectors are the conjugates of A's), overwritten in place: about 1.5 n^2 complex of memory (26 GB at n 32768) in
        # place of the full zheevd's about 4 n^2 (a first host attempt at 32768 with torch's zheevd ended without a traceback on the box).
        A = hp.toarray()
        e_neg, Z = sla.eigh(A.T, subset_by_value=[-np.inf, 0.0], driver="evr", overwrite_a=True, check_finite=False)
        del A
        Phi = torch.as_tensor(np.ascontiguousarray(Z.conj()), dtype=CD); del Z
        en = np.asarray(e_neg, float)
        ab, _u = _herm_band_upper(hp)
        e_all = np.asarray(sla.eigvals_banded(ab, lower=False, check_finite=False), float)
        info = {"n": int(hp.shape[0]), "n_neg_plus": int(Phi.shape[1]), "n_neg_banded_check": int((e_all < 0).sum()), "n_zero_1e-12": int((np.abs(e_all) < 1e-12).sum()),
                "min_abs_e": float(np.min(np.abs(e_all))), "E_ref_plus": float(e_all[e_all < 0].sum() - e_all[e_all > 0].sum()),
                "neg_levels_max_dev_from_banded": float(np.max(np.abs(np.sort(en) - np.sort(e_all[e_all < 0])))) if len(en) == int((e_all < 0).sum()) else None,
                "eig_device": "cpu (scipy zheevr, negative subset)", "seconds": time.time() - t0}
        if info["n_neg_plus"] != info["n_neg_banded_check"]:
            raise RuntimeError("the host subset eigensolve returned %d negative levels, the banded spectrum has %d" % (info["n_neg_plus"], info["n_neg_banded_check"]))
        Phi = Phi.to(device)
        log("v6s verify: the initial + sector vacuum by zheevr (negative subset) of h_+ (%d x %d) on the host: %d occupied (banded check %d), %d levels within 1e-12 of zero, min|e| %.3e, the levels against the banded spectrum %.1e (%.1f s)" % (
            info["n"], info["n"], info["n_neg_plus"], info["n_neg_banded_check"], info["n_zero_1e-12"], info["min_abs_e"], info["neg_levels_max_dev_from_banded"] or -1.0, info["seconds"]))
        return Phi, info
    A = torch.as_tensor(hp.toarray(), dtype=CD, device=eig_device)
    e, V = torch.linalg.eigh(A); del A
    neg = e < 0
    Phi = V[:, neg].clone(); del V
    en = e.cpu().numpy().astype(float)
    info = {"n": int(hp.shape[0]), "n_neg_plus": int(neg.sum()), "n_zero_1e-12": int((np.abs(en) < 1e-12).sum()), "min_abs_e": float(np.min(np.abs(en))),
            "E_ref_plus": float(en[en < 0].sum() - en[en > 0].sum()), "eig_device": str(eig_device), "seconds": time.time() - t0}
    Phi = Phi.to(device)
    log("v6s verify: the initial + sector vacuum by eigh of h_+ (%d x %d) on %s: %d occupied, %d levels within 1e-12 of zero, min|e| %.3e (%.1f s)" % (
        info["n"], info["n"], eig_device, info["n_neg_plus"], info["n_zero_1e-12"], info["min_abs_e"], info["seconds"]))
    return Phi, info


def _hp_md5(hp):
    import hashlib
    hp = hp.tocsr(copy=True); hp.sort_indices()
    m = hashlib.md5()
    for arr in (hp.indptr, hp.indices, hp.data):
        m.update(np.ascontiguousarray(arr).tobytes())
    return m.hexdigest()


def _ick(run, a):
    """the interior readout clocks: every --interior-every-rs from T_0, the end, the hold's record."""
    rs = run.rs_nom; H = run.H
    T0r = float(H.T[run.k0]) / rs; T1r = run._T_of_k(run.kend) / rs
    x = float(a.interior_every_rs)
    ks = {run.k0, run.kend}
    if x > 0:
        y = math.ceil(T0r / x - 1e-9) * x
        while y <= T1r + 1e-9:
            ks.add(run._k_of(y * rs)); y += x
    hold = getattr(H, "hold", None)
    if hold and run.k0 <= hold["k_hold"] <= run.kend:
        ks.add(int(hold["k_hold"]))
    return set(k for k in ks if run.k0 <= k <= run.kend)


def _start(run, api):
    a = run.a; H = run.H; log = api["log"]
    N, v = H.fields(run.k0)
    for c in run.chs:
        h = c.ops.h_csr(N, v)
        aud = W1S.operator_audit(h)
        if not aud["ok"]:
            raise RuntimeError("--lean-interior needs the + sector route: the operator at T_0 does not respect S_stag or the pairing: %s" % aud)
        hp = W1S.sector_plus_csr(h)
        n = hp.shape[0]
        ed = getattr(a, "init_eig_device", "auto")
        if ed == "auto":
            ed = "cpu" if (n > 12000 or run.dev.type == "cpu") else "device"
        eig_dev = torch.device("cpu") if ed == "cpu" else run.dev
        md5_h = _hp_md5(hp)
        cache = getattr(a, "init_cache", None)
        info = None
        if cache and os.path.exists(cache):
            st = torch.load(cache, map_location="cpu", weights_only=False)
            if st.get("md5_h_plus") == md5_h and int(st.get("n", -1)) == n:
                c.Phi = st["Phi_plus"].to(run.dev); info = dict(st.get("info") or {}, cache=cache, cache_loaded=True)
                log("v6s verify kappa %d: the initial + sector vacuum loaded from %s (h_+ md5 %s, %d x %d)" % (c.ops.kappa, cache, md5_h, c.Phi.shape[0], c.Phi.shape[1]))
            else:
                raise RuntimeError("--init-cache %s holds another operator (md5 %s, n %s) than h_+(T_0) (md5 %s, n %d)" % (cache, st.get("md5_h_plus"), st.get("n"), md5_h, n))
        if info is None:
            c.Phi, info = _dense_init_plus(hp, run.dev, eig_dev, a.threads, log)
            if cache:
                tmp = cache + ".tmp"
                torch.save({"Phi_plus": c.Phi.cpu(), "md5_h_plus": md5_h, "n": n, "info": info, "kappa": c.ops.kappa, "kappa_mass": c.ops.kappa_mass,
                            "record": a.record, "fields": a.fields, "T0": float(H.T[run.k0]), "nsites": int(H.nsites), "code": CODE}, tmp)
                os.replace(tmp, cache)
                log("v6s verify kappa %d: the initial + sector vacuum cached to %s" % (c.ops.kappa, cache))
        c.F = torch.zeros((c.ops.n2, 0), dtype=CD, device=run.dev)
        c.sector_prop = True
        c.init_info = dict(info, md5_h_plus=md5_h, operator_audit=aud)
    run.sector_on = True
    run.sector_report = {"used": True, "reason": "--lean-interior: the state built in the + sector (Phi_+ = V_+[:, e_+ < 0]) by construction; the operator audited at T_0",
                         "operator": aud}
    run.min_N_minus_absv_T0 = float(np.min(N - np.abs(v)))


def _L_collar(run, api, c, hp):
    Jp = OPS.Banded(W1S.current_out_plus_csr(hp, np.asarray(c.ops.R, float), run.R_collar_abs), run.dev)
    L = 2.0 * api["tr_blocked"](Jp, c.Phi)
    del Jp
    return L


def _step_record(run, api, c, k):
    """the production's sub-steps of record k on Phi_+ (record_substeps), no traces but the collar current at the record's start."""
    a = run.a; H = run.H
    t0 = H.T[k]; t1 = H.T[k + 1] if k + 1 < len(H) else t0 + run.tau
    sub = W1S.record_substeps(H, k, t0, t1, False, run.cw, run.interp_mode, run.interp_sub)
    if getattr(run, "audit_on", False):
        run.w1_audit_record(c, c.ops.h_csr(*H.fields(k)), k)
    L0 = None; Kmax = 0; tail = 0.0; Ksum = 0
    for (sa, sb, mode, (N, v)) in sub:
        hp = W1S.sector_plus_csr(c.ops.h_csr(N, v)); Bp = BandedLean(hp, run.dev)
        if L0 is None and getattr(a, "lean_flux", "on") == "on":
            L0 = _L_collar(run, api, c, hp)
        c.Phi, K_, tl = propagate_lean(Bp, c.Phi, sb - sa, a.cheb_tol, a.chunk)
        Kmax = max(Kmax, K_); tail += tl; Ksum += int(K_)
        del Bp
    c.n_substeps += len(sub); c.cheb_K_max = max(c.cheb_K_max, Kmax); c.cheb_tail_acc += tail
    return {"k": int(k), "t": float(t0), "L_at_t": L0, "cheb_K": Kmax, "cheb_K_sum": Ksum, "cheb_tail": tail, "sub_steps": len(sub), "held_step": False}


def _step_held(run, api, c, k, k_next):
    """records k ... k_next - 1 held (equal fields): one Chebyshev series over T(k_next) - T(k) on h_+ of record k."""
    a = run.a; H = run.H
    N, v = H.fields(k); N1, v1 = H.fields(k_next)
    if not (np.array_equal(N, N1) and np.array_equal(v, v1)):
        raise RuntimeError("the held step %d -> %d spans unequal fields" % (k, k_next))
    hp = W1S.sector_plus_csr(c.ops.h_csr(N, v)); Bp = BandedLean(hp, run.dev)
    L0 = _L_collar(run, api, c, hp) if getattr(a, "lean_flux", "on") == "on" else None
    dt = run._T_of_k(k_next) - run._T_of_k(k)
    c.Phi, K_, tl = propagate_lean(Bp, c.Phi, dt, a.cheb_tol, a.chunk)
    c.n_substeps += 1; c.cheb_K_max = max(c.cheb_K_max, int(K_)); c.cheb_tail_acc += tl
    del Bp
    return {"k": int(k), "k_next": int(k_next), "t": float(H.T[k]), "dt": float(dt), "L_at_t": L0, "cheb_K": int(K_), "cheb_K_sum": int(K_), "cheb_tail": tl, "sub_steps": 1, "held_step": True}


def _checkpoint(run, api, k):
    a = run.a; H = run.H; log = api["log"]
    t = float(H.T[k]); trs = t / run.rs_nom
    E_k = float(H.E[k]); rs_k = 2.0 * run.g * E_k; TH = 1.0 / (4.0 * math.pi * rs_k)
    N, v = H.fields(k)
    hold = getattr(H, "hold", None)
    row = {"k": int(k), "t": t, "t_over_rs": trs, "E": E_k, "two_gE": rs_k, "T_H": TH, "held": bool(hold and k > hold["k_hold"]),
           "min_N_minus_absv": float(np.min(N - np.abs(v))), "per_channel": {}}
    if run.dev.type == "cuda":
        torch.cuda.synchronize(run.dev)
        row["device_peak_since_last_GiB"] = torch.cuda.max_memory_allocated(run.dev) / 2.0 ** 30
        torch.cuda.reset_peak_memory_stats(run.dev)
    tck = time.time()
    for c in run.chs:
        h = c.ops.h_csr(N, v); hp = W1S.sector_plus_csr(h); Bp = OPS.Banded(hp, run.dev)
        Phi = c.Phi; n0 = int(Phi.shape[1])
        ti = time.time()
        E_hC = 2.0 * api["tr_blocked"](Bp, Phi) - W1S.trace_diag(hp)
        ab, _u = _herm_band_upper(hp)
        e = np.asarray(sla.eigvals_banded(ab, lower=False, check_finite=False), float)
        E_ref = float(e[e < 0].sum() - e[e > 0].sum())                          # Tr min(h, 0) = -Tr|h_+| (spec h = spec h_+ U -spec h_+)
        L = _L_collar(run, api, c, hp)
        t_energy = time.time() - ti
        ti = time.time()
        due_imp = (float(getattr(a, "impurity_every_rs", 0.0) or 0.0) <= 0) or (getattr(c, "imp_last_t", None) is None) or \
                  (trs - c.imp_last_t >= float(a.impurity_every_rs) - 1e-9) or k == run.kend
        coln = torch.linalg.vector_norm(Phi, dim=0) ** 2 - 1.0
        col_drift = float(coln.abs().sum()); del coln
        if due_imp:
            G = Phi.mH @ Phi
            G.diagonal().sub_(1.0)
            unit_F = float(torch.linalg.norm(G))
            unit_tr = float(torch.linalg.eigvalsh(0.5 * (G + G.mH)).abs().sum()); del G
            c.imp_last = unit_tr; c.imp_last_t = trs; c.unit_defect = max(c.unit_defect, unit_F)
        else:
            unit_F = None; unit_tr = getattr(c, "imp_last", None)
        if getattr(c, "T_init", None) is None:
            c.T_init = unit_tr
        t_imp = time.time() - ti
        fl = W1S.propagation_floor(n0, c.cheb_tail_acc, int(c.n_substeps), int(c.cheb_K_max), unit_tr, T_init=getattr(c, "T_init", None))
        fl = dict(fl, q=int((e < 0).sum()), sector=True, h_bound=Bp.bound,
                  T_meas_clock_over_rs=getattr(c, "imp_last_t", None), T_meas_measured_here=bool(due_imp))
        cuts = [("%g" % x, x * run.rs_nom) for x in run.d13_cfg["cuts_rs"]] if run.d13_cfg else [("%g" % x, x * run.rs_nom) for x in (2, 16)]
        ir = interior_readout(Phi, hp, cuts, np.asarray(c.ops.R, float), run.dev, threads=(a.interior_threads or a.threads), hs=a.interior_hs, U=a.interior_U,
                              floor=fl, log=log, kappa=c.ops.kappa, trs=trs)
        pc = {"n0_plus": n0, "E_hC": E_hC, "E_ref": E_ref, "E_inv": E_hC - E_ref, "n_neg_plus": int((e < 0).sum()), "n_zero_plus_1e-12": int((np.abs(e) < 1e-12).sum()),
              "min_abs_e_plus": float(np.min(np.abs(e))), "L_at_t": L, "impurity_trace": unit_tr, "impurity_F": unit_F, "column_norm_drift_sum": col_drift,
              "impurity_measured_here": bool(due_imp), "T_init": getattr(c, "T_init", None), "propagation_floor": fl, "n_substeps": int(c.n_substeps), "cheb_K_max": int(c.cheb_K_max),
              "cheb_tail_acc": float(c.cheb_tail_acc), "interior": ir, "seconds_energy": t_energy, "seconds_impurity": t_imp, "kappa_mass": float(c.ops.kappa_mass)}
        row["per_channel"][str(c.ops.kappa)] = pc
        del Bp
        log("v6s verify lean kappa %d T/r_s %.3f%s: E_inv %.6e, L %.4e, impurity %s (column drift %.2e), K_max %d, sub-steps %d; %s" % (
            c.ops.kappa, trs, " (held)" if row["held"] else "", E_hC - E_ref, L, ("%.3e" % unit_tr) if unit_tr is not None else "-", col_drift, c.cheb_K_max, c.n_substeps,
            "; ".join("%s: excess %.6e (floor low %s, high %s, %s)" % (lab, d["excess"], ("%.1e" % d["excess_floor_low"]) if d.get("excess_floor_low") is not None else "-",
                      ("%.1e" % d["excess_floor_high"]) if d.get("excess_floor_high") is not None else "-", d.get("excess_resolution")) for lab, d in ir["cuts"].items())))
    row["seconds"] = time.time() - tck
    if run.dev.type == "cuda":
        torch.cuda.synchronize(run.dev)
        row["device_peak_checkpoint_GiB"] = torch.cuda.max_memory_allocated(run.dev) / 2.0 ** 30
        torch.cuda.reset_peak_memory_stats(run.dev)
        torch.cuda.empty_cache()
    row["wall_s"] = time.time() - api["T_START"]
    run.rows.append(row); run.n_ck += 1


def _herm_band_upper(h):
    co = h.tocoo(); n = co.shape[0]
    up = co.col >= co.row
    u = int((co.col[up] - co.row[up]).max()) if up.any() else 0
    ab = np.zeros((u + 1, n), dtype=complex)
    np.add.at(ab, (u + co.row[up] - co.col[up], co.col[up]), co.data[up])
    return ab, u


def _state_path(run):
    return run.prefix + "_istate.pt"


def _save(run, api, k):
    st = {"k": int(k), "rows": run.rows, "records": run.records, "n_ck": run.n_ck, "code": CODE, "channels": {}}
    for c in run.chs:
        st["channels"][str(c.ops.kappa)] = {"Phi": c.Phi.cpu(), "cheb_K_max": c.cheb_K_max, "cheb_tail_acc": c.cheb_tail_acc, "n_substeps": int(c.n_substeps),
                                             "T_init": getattr(c, "T_init", None), "imp_last": getattr(c, "imp_last", None), "imp_last_t": getattr(c, "imp_last_t", None),
                                             "unit_defect": c.unit_defect, "init_info": getattr(c, "init_info", None)}
    dest = _state_path(run); tmp = dest + ".tmp"
    torch.save(st, tmp); os.replace(tmp, dest)
    _write(run, api, final=False)
    api["log"]("v6s verify: the lean state saved at record %d (T/r_s %.3f) to %s" % (k, run._T_of_k(k) / run.rs_nom, dest))


def _load(run, api):
    st = torch.load(_state_path(run), map_location="cpu", weights_only=False)
    run.k = int(st["k"]); run.rows = st["rows"]; run.records = st["records"]; run.n_ck = st["n_ck"]
    for c in run.chs:
        s = st["channels"][str(c.ops.kappa)]
        c.Phi = s["Phi"].to(run.dev); c.cheb_K_max = s["cheb_K_max"]; c.cheb_tail_acc = s["cheb_tail_acc"]; c.n_substeps = int(s["n_substeps"])
        c.T_init = s.get("T_init"); c.imp_last = s.get("imp_last"); c.imp_last_t = s.get("imp_last_t"); c.unit_defect = s.get("unit_defect", 0.0)
        c.init_info = s.get("init_info"); c.F = torch.zeros((c.ops.n2, 0), dtype=CD, device=run.dev); c.sector_prop = True
    run.sector_on = True
    api["log"]("v6s verify: the lean run resumed at record %d (T/r_s %.3f), %d readouts done" % (run.k, run._T_of_k(run.k) / run.rs_nom, len(run.rows)))


def _write(run, api, final):
    a = run.a
    summ = {"code": CODE, "mode": "lean interior (v6s verify item 3)", "final": bool(final), "settings": vars(a), "hold": getattr(run.H, "hold", None),
            "history": {"label": run.H.label, "records": len(run.H), "tau": run.tau, "g": run.g, "rs_nominal": run.rs_nom, "engine_sites": run.H.ne, "nsites": int(run.H.nsites),
                        "T_start": float(run.H.T[run.k0]), "T_end": run._T_of_k(run.kend)},
            "propagation": run.w1_propagation_label() + "; held records: one Chebyshev series per --hold-step-rs", "audit": run.w1_audit,
            "init": {str(c.ops.kappa): getattr(c, "init_info", None) for c in run.chs},
            "rows": run.rows, "records": run.records, "wall_s": time.time() - api["T_START"]}
    with open(run.prefix + "_summary.json", "w") as f:
        json.dump(summ, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o), indent=1)
    L = ["# %s: the lean interior run (%s)" % (a.tag, "final" if final else "running"), "",
         "page_history_v6s.py --lean-interior; chain %d sites; channels %s%s; hold %s; cuts %s r_s; readouts by the interior side (purity), the vacuum block by the contour." % (
             int(run.H.nsites), run.channels, " (kappa = 0: the mass term off)" if getattr(a, "kappa_zero", False) else "",
             ("from T/r_s %.3f" % (run.H.hold["T_hold"] / run.rs_nom)) if getattr(run.H, "hold", None) else "none", run.d13_cfg["cuts_rs"] if run.d13_cfg else "-"), "",
         "| T/r_s | held | kappa | E_inv | L (collar) | impurity | " + " | ".join("excess %g" % x for x in (run.d13_cfg["cuts_rs"] if run.d13_cfg else [])) + " | readout s | wall s | device peak GiB |",
         "|---|---|---|---|---|---|" + "---|" * len(run.d13_cfg["cuts_rs"] if run.d13_cfg else []) + "---|---|---|"]
    for r in run.rows:
        for kap, pc in r["per_channel"].items():
            ex = [pc["interior"]["cuts"][lab]["excess"] for lab in pc["interior"]["cuts"]]
            L.append("| %.3f | %s | %s | %.6e | %s | %s | %s | %.1f | %.0f | %s |" % (
                r["t_over_rs"], "yes" if r["held"] else "no", kap, pc["E_inv"], ("%.4e" % pc["L_at_t"]) if pc["L_at_t"] is not None else "-",
                ("%.2e" % pc["impurity_trace"]) if pc["impurity_trace"] is not None else "-", " | ".join("%.6f" % x for x in ex), r["seconds"], r["wall_s"],
                ("%.2f" % r["device_peak_since_last_GiB"]) if r.get("device_peak_since_last_GiB") is not None else "-"))
    with open(run.prefix + "_interior.md", "w") as f:
        f.write("\n".join(L) + "\n")


def run_lean_interior(run, api, resume=False):
    """the lean interior run (B2's 32k chain): start (or resume), the records sub-stepped before the hold and in --hold-step-rs Chebyshev steps
    after it, the interior readouts at the readout clocks, the state saved every --save-every readouts and at the wall cap."""
    a = run.a; log = api["log"]; H = run.H
    if resume and os.path.exists(_state_path(run)):
        _load(run, api)
    else:
        run.rows = []; run.records = []; run.n_ck = 0; run.k = run.k0
        for c in run.chs:
            c.n_substeps = 0
        _start(run, api)
    ick = _ick(run, a)
    hold = getattr(H, "hold", None)
    k_hold = int(hold["k_hold"]) if hold else None
    n_hold = max(1, int(round(float(a.hold_step_rs) * run.rs_nom / run.tau)))
    done = {int(r_["k"]) for r_ in run.rows}
    log("v6s verify: the lean interior run: %d readout clocks (every %g r_s; T/r_s %.3f to %.3f), the hold %s, hold step %d records; chain %d sites, + sector %d x %d per channel" % (
        len(ick), a.interior_every_rs, run._T_of_k(run.k0) / run.rs_nom, run._T_of_k(run.kend) / run.rs_nom,
        ("from record %d (T/r_s %.3f)" % (k_hold, H.T[k_hold] / run.rs_nom)) if hold else "none", n_hold, int(H.nsites), run.chs[0].Phi.shape[0], run.chs[0].Phi.shape[1]))
    k = run.k
    t_prop = 0.0; n_prop_records = 0; t_last = time.time()
    while True:
        if k in ick and k not in done:
            _checkpoint(run, api, k); done.add(k)
            if run.n_ck % max(1, a.save_every) == 0 or k == run.kend:
                _save(run, api, k)
            else:
                _write(run, api, final=False)
        if k >= run.kend:
            break
        ts = time.time()
        if k_hold is not None and k >= k_hold:
            nxt = [q for q in ick if q > k]
            k_next = min([k + n_hold, run.kend] + nxt)
            for c in run.chs:
                rec = _step_held(run, api, c, k, k_next)
            n_adv = k_next - k
        else:
            for c in run.chs:
                rec = _step_record(run, api, c, k)
            k_next = k + 1; n_adv = 1
        if run.dev.type == "cuda":
            torch.cuda.synchronize(run.dev)
        dtw = time.time() - ts; t_prop += dtw; n_prop_records += n_adv
        rec["wall_step_s"] = dtw
        run.records.append(rec)
        k = k_next; run.k = k
        if time.time() - t_last > 600.0:
            t_last = time.time()
            log("v6s verify: at record %d (T/r_s %.3f): %.1f s of propagation for %d records (%.1f s per r_s)" % (
                k, run._T_of_k(k) / run.rs_nom, t_prop, n_prop_records, t_prop / max(1e-9, n_prop_records * run.tau / run.rs_nom)))
        if (time.time() - api["T_START"]) / 3600.0 > a.max_wall_hours:
            log("v6s verify: the wall cap: the lean state saved at record %d" % k)
            _save(run, api, k)
            return False
    _write(run, api, final=True)
    log("v6s verify: the lean interior run done at T/r_s %.3f: %.1f s of propagation for %d records (%.1f s per r_s), %d readouts" % (
        run._T_of_k(k) / run.rs_nom, t_prop, n_prop_records, t_prop / max(1e-9, n_prop_records * run.tau / run.rs_nom), len(run.rows)))
    return True
