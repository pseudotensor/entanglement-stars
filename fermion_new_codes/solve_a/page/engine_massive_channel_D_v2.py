#!/usr/bin/env python
"""engine_massive_channel_D_v2.py (builder analyst). Version 2 = version 1 with the entropy's sector corrected: version 1 took P01 sec. 2's
split hull of the deviation's kept eigenvectors beyond the collar in the eigenbasis of the whole chain's h_m, so the hull (the kept vectors'
P_- and 1 - P_- halves) reached into the interior, and at T/r_s 12 kappa 1 it read 2.379 nats from the interior's own deviation (the dust)
against 5.6e-10 at eps 1e-6 and a region excess of 1.4e-8 (P01 sec. 3: collecting both halves collects their hull, which can include
interior partners); its run also stopped on a LAPACK non-convergence in that hull's SVD. Version 2 builds the escaped sector first, as the
Page collector does: in each spectral half of h_m the eigenvectors of the half's outside-weight operator with weight >= 1 - eps_loc (1e-2;
1e-3 the control), so the sector reduces P_-; it prints the sector's entropy S_F = tr s(F^dag C F) and takes P01 sec. 2's hull of the kept
eigenvectors of F^dag (C - P_-) F in the sector's own coordinates (hull_in_coords with c the identity there, mode "exact"; scipy's gesvd if
torch's SVD does not converge, flagged): S_esc = S_hull, with the dropped eigenvalues' bound and S_F beside it. The sector's own energy E_F
(the mode route) is printed beside the collar's E_out. Nothing else changes. Version 1's docstring follows.
engine_massive_channel_D_v1.py (builder analyst; the quasi-frozen ruling's S1 D, owner-approved (inbox #36); brief
lead/BRIEF_D_S11_builder.md). The ruling, verbatim (REVIEWA_RULING_quasifrozen.md line 78):
"D. The massive channel's own emission, with back-reaction, from the stored records (the one-species statement): on the stored full states
(the 10 slices T/r_s 12 to 40 and the continuation's slices every 2 r_s), the massive deviation C - P beyond the collar 16 to 24 r_s: its
Killing spectrum in bins of T_H/2 and in units of m (m/T_H = 134 at the solved charge; print the factor e^{-m/T_H}), its energy against the
collar ledger of S5 (d) with -Delta W, its entropy by the spectral hull of P01 sec. 2 (keeping every nonzero singular vector); a coarse
S_esc(t) and E_rad(t) of the massive channel at the slice clocks; labelled 'the model's own channel; thermal part suppressed by e^{-134}; the
hard part is the formation's creation'."
S5 (d), verbatim (WEB40F_TASKS/solve.md): "(d) the charge ledger with -Delta W, W = (1/2) int d mu v J, beside a + b + c (the stored
M1_balance is a ledger for the inside Killing expectation K_chi = E_G + W)".

Readers reused, not rewritten: engine_single_object_A_v2 (hm_host: the record's h_m = (1/2){N, h_0} + (1/2){v, Q} on the stored chain;
eigh_d and mm_d: the dense linear algebra by torch, here on the CPU; s_ent; fmt), engine_replay_v9.ledger_terms (S5 (d)'s per-record terms:
E_G, K by the density route, K_src and W = (1/2) int dmu v J_ell on the solve's fine grid, its site form, K_chi = X1 - X2, K_out, eps_adj,
eps_G, and the record's M1_balance a, b, c), page_ops_v6.hull_in_coords (mode "exact": every left singular vector of each nonzero half
block, P01 sec. 2's repair as built for the Page code) and page_ops_v6.rank_sensitive_bound, the field module's solve_constraints_ivp and
mass_function (W(r)/(2g)).

Conventions (printed in the outputs):
- The state per channel kappa (weight w = 16 kappa, the engine's Chan.w): Phi = [sea, dust] with occupations f (1 on the sea, the stored f on
  the dust), re-orthonormalized in float64 by the Loewdin step Phi G^(-1/2) (engine_single_object_A_v2's operation), C = Phi f Phi^dagger;
  the stored idempotency max|mu^2 - mu| of f^(1/2) G f^(1/2) printed (single precision on the complex64 slices).
- P: the Killing vacuum P_- = 1_{h_m < 0} of h_m at the slice's applied fields on the stored chain (one dense eigendecomposition per slice
  and channel; its eigenvalues printed against the slice's stored e_kappa), the primary; the continued reference P_ref = R2 R2^dagger (the
  stored R2 columns, Loewdin-orthonormalized) as the second reference, printed beside.
- The collar: the Page code's (page_history_v6b): a sharp step beyond R_collar = 16 r_s (r_s the run's nominal rs), the energy beyond it
  E_out = Tr[{chi, h_m}/2 (C - P)] = Re Tr[chi h_m (C - P)]; the collector cut R_cut = 24 r_s; both regions Y (R > 16 r_s, R > 24 r_s) are
  read, Y16 primary.
- The Killing spectrum beyond the collar: the eigenbasis (omega_j, phi_j) of the exterior block h_YY of h_m on Y (S1 A.6's "Killing-energy
  shells of the exterior block's h"), n_j = phi_j^dagger (C - P)_YY phi_j; bins of T_H/2 in omega with T_H = 1/(8 pi g E) at the record's E
  (bin j covers [j T_H/2, (j + 1) T_H/2)); also in units of m; the block energy sum_j omega_j n_j = Tr[h_YY (C - P)_YY] checked against the
  direct trace; e^{-m/T_H} printed per slice.
- The entropy by the spectral hull: the eigenvectors of D_Y = (C - P_-)_YY with |lambda| > eps_rank (1e-9 primary; the counts at other
  thresholds and the hull entropy at --eps-control printed beside), embedded in the chain, written in the eigenbasis of h_m and split by
  the sign of the Killing energy; the hull L of the kept vectors (hull_in_coords, mode "exact"); S = tr s(L^dagger C L); the check that L
  reduces P_- (max |mu^2 - mu| of L^dagger P_- L) printed; the dropped eigenvalues' rank-sensitive bound q h(min(delta/q, 1/2)) printed.
  Beside it, labelled a comparator, the region's own entropies S(C_YY), S((P_-)_YY) and their difference.
- The ledger (S5 (d)): at T = 0 and at each slice clock the terms of engine_replay_v9.ledger_terms (one re-solve of the stored shell totals
  per such record by the run's field module, as the replay does); per slice interval Delta E_G, the sum of a + b + c over the record
  intervals in it (the stored M1_balance per record), -Delta W, Delta K_out, Delta eps_adj, Delta eps_G and the remainder (minus the summed
  M1 closures by construction); beside them the energy beyond the collar: E_out (P_-), E_out (R2), the density route
  sum_{R > R_c} (N_x eps_x + v_x J_x) of the stored shell totals (dust and pairs apart; the sources are relative to R2), its W part
  (1/2) sum_{R > R_c} v_x J_x, and the charge beyond the collar E_G - M(R_c) with M(r) = W(r)/(2g) of the re-solve.
- E_rad(t): the density route beyond the collar at every stored record (the stored shell totals; no state needed), so the coarse slice
  values have a fine series beside them; S_esc(t): the hull entropy at the slice clocks.
Every number is printed by the script. No mutual information is computed (D15).
Args: --run dir/tag --slices a.npz,b.npz --fields-module --page-code --R-collar-rs --R-cut-rs --eps-rank --eps-control --kappas
--round-c64 (control: the columns rounded to complex64 before the readout) --records-only --threads --out-json --md; --compose a.json,...
--md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import importlib
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_A_v2 as SA  # noqa: E402
import engine_replay_v9 as RP9  # noqa: E402

LABEL = "the model's own channel; thermal part suppressed by e^{-134}; the hard part is the formation's creation"
D15 = "no mutual information is computed or printed (D15)"
THRESH = (1e-12, 1e-9, 1e-7, 1e-6, 1e-4, 1e-2)
M_BINS = (-1e9, -3.0, -2.0, -1.5, -1.1, -1.0, -0.9, 0.0, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 1e9)   # omega/m edges
PO = None
fmt = SA.fmt


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def loewdin(X):
    """X G^(-1/2) with G = X^dagger X (engine_single_object_A_v2's Loewdin step); returns (X_o, G, max|G - I|, max|X_o^dag X_o - I|)."""
    G = SA.mm_d(X.conj().T, X)
    gw, gv = SA.eigh_d(G)
    Xo = SA.mm_d(X, (gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T)
    e0 = float(np.max(np.abs(G - np.eye(G.shape[0])))) if G.shape[0] else 0.0
    e1 = float(np.max(np.abs(SA.mm_d(Xo.conj().T, Xo) - np.eye(G.shape[0])))) if G.shape[0] else 0.0
    return Xo, G, e0, e1


def herm(A):
    return 0.5 * (A + A.conj().T)


def ent(mu):
    return float(SA.s_ent(np.asarray(mu, float)).sum())


def region_rows(R, Rc):
    """the rows (2 components per site) of the sites with R > Rc (the Page code's sharp collar)."""
    i0 = int(np.searchsorted(np.asarray(R, float), Rc, side="right"))
    return np.arange(2 * i0, 2 * len(R)), i0


def bin_spectrum(om, n_list, T_H, m):
    """bins of T_H/2 in omega (bin j covers [j T_H/2, (j + 1) T_H/2)) and the m-unit bins: per bin the summed n and omega n for each
    occupation list in n_list, and the level count."""
    j = np.floor(om / (0.5 * T_H)).astype(np.int64)
    out = {}
    for jj in np.unique(j):
        s = j == jj
        out[int(jj)] = {"count": int(s.sum()), "n": [float(np.sum(n[s])) for n in n_list], "E": [float(np.sum(om[s] * n[s])) for n in n_list]}
    mb = []
    x = om / m
    for lo, hi in zip(M_BINS[:-1], M_BINS[1:]):
        s = (x >= lo) & (x < hi)
        mb.append({"lo": lo, "hi": hi, "count": int(s.sum()), "n": [float(np.sum(n[s])) for n in n_list], "E": [float(np.sum(om[s] * n[s])) for n in n_list]})
    return out, mb


def outside_weight_halves(V, e, ry):
    """v2: per spectral half of h_m (e >= 0: +1, e < 0: -1) the eigendecomposition of the half's outside-weight operator V_h^dagger Pi_Y V_h."""
    out = {}
    for hsign, sel in ((1, e >= 0), (-1, e < 0)):
        Vh = V[:, sel]
        w, Q = SA.eigh_d(herm(SA.mm_d(Vh[ry].conj().T, Vh[ry])))
        out[hsign] = (Vh, w, Q)
    return out


def escaped_sector(Wq, eps_loc):
    """v2: the escaped sector beyond the collar, built as the Page collector builds its collected modes: in each spectral half of h_m the
    eigenvectors of the half's outside-weight operator with eigenvalue >= 1 - eps_loc; F = [F_+, F_-] with its half labels (every column
    inside one half, so F^dagger P_- F is diagonal 0/1)."""
    Fs, hs, info = [], [], {"eps_loc": eps_loc}
    for hsign in (1, -1):
        Vh, w, Q = Wq[hsign]
        keep = w >= 1.0 - eps_loc
        Fs.append(SA.mm_d(Vh, Q[:, keep])); hs.append(np.full(int(keep.sum()), hsign, dtype=np.int64))
        info["dim_%s" % ("plus" if hsign > 0 else "minus")] = int(keep.sum())
        info["outside_weight_min_%s" % ("plus" if hsign > 0 else "minus")] = float(w[keep].min()) if keep.any() else None
    return np.concatenate(Fs, axis=1), np.concatenate(hs), info


def hull_coords(half, Ukeep):
    """page_ops_v6.hull_in_coords (mode "exact") in the escaped sector's own coordinates (c = the identity there; each coordinate inside one
    spectral half), so the hull lies in the sector exactly; on a LAPACK convergence failure of torch's SVD the same per-half left singular
    vectors by scipy's gesvd (flagged in the output)."""
    import torch
    k = Ukeep.shape[0]
    st = {}
    try:
        L, hh = PO.hull_in_coords(torch.eye(k, dtype=torch.complex128), torch.as_tensor(half), torch.as_tensor(np.ascontiguousarray(Ukeep)), mode="exact", stats=st)
        return L.numpy(), hh.numpy(), st, False
    except RuntimeError:
        import scipy.linalg as sl
        outs, hs = [], []
        for hsign in (1, -1):
            cols = np.nonzero(half == hsign)[0]
            if cols.size == 0:
                continue
            Uo, so, _ = sl.svd(Ukeep[cols, :], full_matrices=False, lapack_driver="gesvd")
            if so.size == 0 or not bool((so > 0).any()):
                continue
            B = np.zeros((k, Uo.shape[1]), complex); B[cols, :] = Uo
            outs.append(B); hs.append(np.full(Uo.shape[1], hsign, dtype=np.int64))
            st["kept"] = st.get("kept", 0) + int(Uo.shape[1]); st["s_min_kept"] = min(st.get("s_min_kept", math.inf), float(so.min()))
        L = np.concatenate(outs, axis=1) if outs else np.zeros((k, 0), complex)
        return L, (np.concatenate(hs) if hs else np.zeros(0, np.int64)), st, True


def sector_entropy(F, half, Phi_o, Pf, H, eps_list):
    """v2: on the escaped sector F: M = F^dagger C F, Pi = F^dagger P_- F = diag(half == -1), M_Delta = M - Pi; S_F = tr s(M) (the sector's
    entropy, no truncation); per eps the P01 sec. 2 hull of the eigenvectors of M_Delta with |lambda| > eps (hull_coords, every left
    singular vector of each nonzero half block) and S_hull = tr s(L^dagger M L), with the check that L reduces Pi (max|mu^2 - mu| of
    L^dagger Pi L) and the dropped eigenvalues' rank-sensitive bound q h(min(delta/q, 1/2)); the sector's energy E_F = Re tr[(F^dag h F) M_Delta]
    (the mode route) and its particle and hole numbers (tr M_Delta on the + and - coordinates)."""
    k = F.shape[1]
    A = SA.mm_d(F.conj().T, Phi_o)
    M = herm(SA.mm_d(A * Pf[None, :], A.conj().T))
    Pi = np.diag((half < 0).astype(float))
    MD = M - Pi
    hF = herm(SA.mm_d(F.conj().T, np.asarray(H @ F)))
    out = {"dim": int(k), "S_F": ent(np.linalg.eigvalsh(M)), "E_F": float(np.real(np.sum(hF * MD.T))),
           "particles": float(np.real(np.trace(MD[np.ix_(half > 0, half > 0)]))), "holes": float(-np.real(np.trace(MD[np.ix_(half < 0, half < 0)]))),
           "F_orthonormality": float(np.max(np.abs(SA.mm_d(F.conj().T, F) - np.eye(k)))) if k else 0.0, "hulls": {}}
    lam, U = SA.eigh_d(MD)
    out["lambda_counts"] = {"%g" % t: int(np.sum(np.abs(lam) > t)) for t in THRESH}
    out["lambda_max_abs"] = float(np.max(np.abs(lam))) if k else 0.0
    for eps in eps_list:
        keep = np.abs(lam) > eps
        kk = int(keep.sum()); drop = lam[~keep]
        q = int(drop.size); delta = float(np.sum(np.abs(drop)))
        res = {"eps": eps, "kept": kk, "dropped": q, "dropped_trace_norm": delta, "dropped_bound": PO.rank_sensitive_bound(q, delta)}
        if kk == 0:
            res.update({"hull_dim": 0, "S_hull": 0.0, "vacuum_check": 0.0, "s_min_kept": None, "svd_fallback": False})
        else:
            L, hh, st, fb = hull_coords(half, U[:, keep])
            mu = np.linalg.eigvalsh(herm(SA.mm_d(L.conj().T, SA.mm_d(M, L))))
            mv = np.linalg.eigvalsh(herm(SA.mm_d(L.conj().T, Pi @ L)))
            res.update({"hull_dim": int(L.shape[1]), "S_hull": ent(mu), "vacuum_check": float(np.max(np.abs(mv * mv - mv))),
                        "hull_orthonormality": float(np.max(np.abs(SA.mm_d(L.conj().T, L) - np.eye(L.shape[1])))), "s_min_kept": st.get("s_min_kept"),
                        "svd_fallback": bool(fb)})
        out["hulls"]["%g" % eps] = res
    return out


def channel_D(R, m, kappa, w, N_app, v_app, sea, dust, fdust, R2, e_stored, Rcs, T_H, eps_rank, eps_control, round_c64, log, eps_locs=(1e-2, 1e-3)):
    t0 = time.time()
    cast = (lambda X: np.asarray(X).astype(np.complex64).astype(np.complex128)) if round_c64 else (lambda X: np.asarray(X, np.complex128))
    Phi = np.concatenate([cast(sea), cast(dust)], axis=1)
    nd = int(np.asarray(dust).shape[1])
    f = np.concatenate([np.ones(sea.shape[1]), np.asarray(fdust, float)])
    Phi_o, G, orth0, orth1 = loewdin(Phi)
    fh = np.sqrt(np.clip(f, 0.0, None))
    mu_st = np.linalg.eigvalsh(herm((fh[:, None] * G) * fh[None, :]))
    R2o, _, orth2, orth3 = loewdin(cast(R2))
    Pf = f.astype(float)
    H = SA.hm_host(R, m, kappa, N_app, v_app)
    e, V = SA.eigh_d(H.toarray())
    neg = e < 0
    Vn = V[:, neg]
    out = {"kappa": int(kappa), "w": float(w), "n_sea": int(sea.shape[1]), "n_dust": nd, "dust_f": [float(x) for x in fdust],
           "stored_idempotency": float(np.max(np.abs(mu_st * mu_st - mu_st))), "stored_orthonormality": orth0, "loewdin_orthonormality": orth1,
           "R2_stored_orthonormality": orth2, "R2_loewdin_orthonormality": orth3, "n_negative": int(neg.sum()),
           "eig_vs_stored": float(np.max(np.abs(np.sort(e) - np.sort(np.asarray(e_stored, float))))) if e_stored is not None else None,
           "gap_at_zero": float(e[~neg].min() - e[neg].max()) if (neg.any() and (~neg).any()) else None, "regions": {}}
    hPhi = H @ Phi_o; hR2 = H @ R2o; hVn = H @ Vn
    for Rc_rs, Rc in Rcs:
        ry, i0 = region_rows(R, Rc)
        reg = {"R_c_over_rs": Rc_rs, "R_c": Rc, "first_site": int(i0), "sites": int(len(R) - i0)}
        eCcols = Pf * np.sum((Phi_o[ry].conj() * hPhi[ry]).real, axis=0)
        eC = float(np.sum(eCcols)); eCd = float(np.sum(eCcols[sea.shape[1]:]))
        eM = float(np.sum((Vn[ry].conj() * hVn[ry]).real))
        e2 = float(np.sum((R2o[ry].conj() * hR2[ry]).real))
        reg["E_out_Pm"] = eC - eM; reg["E_out_R2"] = eC - e2
        reg["E_out_dust"] = eCd; reg["E_out_pairs_Pm"] = eC - eCd - eM; reg["E_out_pairs_R2"] = eC - eCd - e2
        CY = herm(SA.mm_d(Phi_o[ry] * Pf[None, :], Phi_o[ry].conj().T))
        PmY = herm(SA.mm_d(Vn[ry], Vn[ry].conj().T))
        P2Y = herm(SA.mm_d(R2o[ry], R2o[ry].conj().T))
        DmY = CY - PmY; D2Y = CY - P2Y
        reg["refs_diff_op"] = float(np.max(np.abs(np.linalg.eigvalsh(P2Y - PmY))))
        reg["dust_weight"] = float(np.sum(Pf[sea.shape[1]:] * np.sum(np.abs(Phi_o[ry][:, sea.shape[1]:]) ** 2, axis=0))) if nd else 0.0
        # the Killing spectrum in the exterior block's eigenbasis
        hY = H[ry][:, ry].toarray()
        om, ph = SA.eigh_d(hY)
        nm = np.real(np.einsum("ij,ij->j", ph.conj(), SA.mm_d(DmY, ph)))
        n2 = np.real(np.einsum("ij,ij->j", ph.conj(), SA.mm_d(D2Y, ph)))
        if nd:
            Ad = SA.mm_d(ph.conj().T, Phi_o[ry][:, sea.shape[1]:])               # the dust columns in the block's Killing basis
            ndust = np.real(np.sum((np.abs(Ad) ** 2) * Pf[None, sea.shape[1]:], axis=1))
        else:
            ndust = np.zeros_like(nm)
        reg["block_energy_Pm"] = float(np.sum(om * nm)); reg["block_energy_Pm_trace"] = float(np.real(np.sum(hY * DmY.T)))
        reg["block_energy_R2"] = float(np.sum(om * n2)); reg["block_energy_R2_trace"] = float(np.real(np.sum(hY * D2Y.T)))
        reg["n_total_Pm"] = float(np.sum(nm)); reg["n_total_R2"] = float(np.sum(n2))
        reg["omega_range"] = [float(om.min()), float(om.max())]
        pos = om > 0
        reg["omega_min_positive"] = float(om[pos].min()) if pos.any() else None
        reg["omega_max_negative"] = float(om[~pos].max()) if (~pos).any() else None
        reg["levels_in_gap_0.9m"] = int(np.sum(np.abs(om) < 0.9 * m))
        # the Killing-basis particle and hole numbers and energies (positive branch: sum n; negative branch: -sum n)
        for lab_, nn in (("Pm", nm), ("R2", n2), ("dust", ndust)):
            reg["particles_%s" % lab_] = float(np.sum(nn[pos])); reg["holes_%s" % lab_] = float(-np.sum(nn[~pos]))
            reg["E_particles_%s" % lab_] = float(np.sum(om[pos] * nn[pos])); reg["E_holes_%s" % lab_] = float(np.sum(om[~pos] * nn[~pos]))
        bins, mbins = bin_spectrum(om, [nm, n2, ndust], T_H, m)
        reg["bins_TH2"] = bins; reg["bins_m"] = mbins
        # the deviation's eigenvalues beyond the collar, the hull entropy
        lam = np.linalg.eigvalsh(DmY)
        reg["lambda_counts"] = {"%g" % t: int(np.sum(np.abs(lam) > t)) for t in THRESH}
        reg["lambda_max_abs"] = float(np.max(np.abs(lam))); reg["N_plus"] = float(np.sum(lam[lam > 0])); reg["N_minus"] = float(-np.sum(lam[lam < 0]))
        reg["trace_norm"] = float(np.sum(np.abs(lam)))
        dec = np.floor(np.log10(np.clip(np.abs(lam), 1e-300, None))).astype(int)
        reg["lambda_decades"] = {str(d): int(np.sum(dec == d)) for d in range(-16, 1)}
        lam2 = np.linalg.eigvalsh(D2Y)
        reg["R2_lambda_counts"] = {"%g" % t: int(np.sum(np.abs(lam2) > t)) for t in THRESH}
        reg["R2_N_plus"] = float(np.sum(lam2[lam2 > 0])); reg["R2_N_minus"] = float(-np.sum(lam2[lam2 < 0]))
        # v2: the escaped sector (half-split modes with outside weight >= 1 - eps_loc) and its entropy by P01 sec. 2's hull in its own coordinates
        reg["sector"] = {}
        Wq = outside_weight_halves(V, e, ry)
        for eloc in eps_locs:
            F, half, sinfo = escaped_sector(Wq, eloc)
            sec = sector_entropy(F, half, Phi_o, Pf, H, [eps_rank, eps_control])
            sec.update(sinfo)
            reg["sector"]["%g" % eloc] = sec
            del F
        del Wq
        prim = reg["sector"]["%g" % eps_locs[0]]
        reg["hull"] = prim["hulls"]["%g" % eps_rank]; reg["hull_control"] = prim["hulls"]["%g" % eps_control]
        reg["S_F"] = prim["S_F"]; reg["E_F"] = prim["E_F"]; reg["sector_particles"] = prim["particles"]; reg["sector_holes"] = prim["holes"]
        reg["S_F_loc_control"] = reg["sector"]["%g" % eps_locs[-1]]["S_F"]
        reg["S_hull_loc_control"] = reg["sector"]["%g" % eps_locs[-1]]["hulls"]["%g" % eps_rank]["S_hull"]
        sC = np.linalg.eigvalsh(CY); sP = np.linalg.eigvalsh(PmY)
        reg["region_S_C"] = ent(sC); reg["region_S_Pm"] = ent(sP); reg["region_S_excess"] = reg["region_S_C"] - reg["region_S_Pm"]
        out["regions"]["%g" % Rc_rs] = reg
        del CY, PmY, P2Y, DmY, D2Y, hY, ph
    out["seconds"] = time.time() - t0
    r16 = out["regions"]["%g" % Rcs[0][0]]
    log("  kappa %d: eig vs stored %s, n_neg %d, Loewdin %.1e (stored %.1e), R2 Loewdin %.1e (stored %.1e); beyond %g r_s: E_out P_- %+.6e (dust %+.3e), R2 %+.6e; Killing-basis particles %.3e, holes %.3e; eigenvalue sums %.3e, %.3e; "
        "|lambda| > 1e-9: %d, max %.3e; escaped sector %d (S_F %.6e, E_F %+.4e); hull %d -> S %.6e (vacuum check %.1e; control eps %g: kept %d, S %.6e); region S excess %+.3e; %.1f s" % (
            kappa, fmt(out["eig_vs_stored"], "%.1e"), out["n_negative"], orth1, orth0, orth3, orth2, Rcs[0][0], r16["E_out_Pm"], r16["E_out_dust"], r16["E_out_R2"],
            r16["particles_Pm"], r16["holes_Pm"], r16["N_plus"], r16["N_minus"], r16["lambda_counts"]["1e-09"], r16["lambda_max_abs"], r16["sector"]["%g" % eps_locs[0]]["dim"], r16["S_F"], r16["E_F"], r16["hull"]["hull_dim"], r16["hull"]["S_hull"],
            r16["hull"]["vacuum_check"], eps_control, r16["hull_control"]["kept"], r16["hull_control"]["S_hull"], r16["region_S_excess"], out["seconds"]))
    return out


# ------------------------------------------------------------------------------------------------------------------ the records' routes
def load_run(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Zf = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    return J, Zf


def density_route(J, Zf, Rcs):
    """per stored record: the density route beyond each collar (the stored shell totals at the record's applied fields; sources relative
    to R2; dust = total - pairs), its W part, E_G, T_H, m/T_H, and the record's M1_balance a + b + c."""
    a = J["args"]; g = float(a["g"]); m = float(a["m"]); rs = float(a["rs"])
    R = np.asarray(Zf["R"], float)
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    Tz = np.asarray(Zf["T"], float)
    rows = []
    for k in range(len(Tz)):
        T = float(Tz[k]); rec = recs.get(round(T, 6))
        if rec is None:
            continue
        N = np.asarray(Zf["N_applied"][k], float); v = np.asarray(Zf["v_applied"][k], float)
        et = np.asarray(Zf["eps_tot"][k], float); jt = np.asarray(Zf["J_tot"][k], float)
        ep = np.asarray(Zf["eps_pairs"][k], float); jp = np.asarray(Zf["J_pairs"][k], float)
        E = float(rec["E"]); TH = 1.0 / (8.0 * math.pi * g * E)
        row = {"k": k, "T": T, "T_over_rs": T / rs, "E_G": E, "two_gE": 2.0 * g * E, "T_H": TH, "m_over_TH": m / TH, "K_total_density": float(np.sum(N * et + v * jt))}
        mb = rec.get("M1_balance") or {}
        row["abc"] = (float(mb["evolution_exact"]) + float(mb["field_work"]) + float(mb["reference_reset"])) if mb.get("evolution_exact") is not None else None
        row["M1_closure"] = mb.get("closure"); row["cut_changed"] = mb.get("cut_changed")
        for Rc_rs, Rc in Rcs:
            s = R > Rc
            row["beyond_%g" % Rc_rs] = {"K_out": float(np.sum(N[s] * et[s] + v[s] * jt[s])), "K_out_pairs": float(np.sum(N[s] * ep[s] + v[s] * jp[s])),
                                         "K_out_dust": float(np.sum(N[s] * (et[s] - ep[s]) + v[s] * (jt[s] - jp[s]))), "W_out": 0.5 * float(np.sum(v[s] * jt[s])),
                                         "eps_out": float(np.sum(et[s])), "eps_out_dust": float(np.sum(et[s] - ep[s]))}
        rows.append(row)
    return rows


def ledger_at(J, Zf, k, FM, Rcs):
    """S5 (d)'s terms at the fields record k (engine_replay_v9.ledger_terms on one re-solve of the stored shell totals) and the mass
    function W(r)/(2g) at the collars."""
    a = J["args"]; g = float(a["g"]); ell = float(a["ell"])
    R = np.asarray(Zf["R"], float); T = float(Zf["T"][k])
    rec = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}[round(T, 6)]
    eps = np.asarray(Zf["eps_tot"][k], float); Jt = np.asarray(Zf["J_tot"][k], float)
    t0 = time.time()
    _, _, _, info = FM.solve_constraints_ivp(eps, Jt, g, R, ell, collar_policy=a.get("collar_policy", "flag"), support_rel=a.get("support_rel", 1e-14))
    L = RP9.ledger_terms(info, R, g, eps, Jt, np.asarray(Zf["N_applied"][k], float), np.asarray(Zf["v_applied"][k], float), rec)
    L["T"] = T; L["k"] = int(k); L["solve_failed"] = info.get("failed"); L["seconds"] = time.time() - t0
    for Rc_rs, Rc in Rcs:
        try:
            L["M_at_%g" % Rc_rs] = FM.mass_function(info, g, Rc)
        except Exception as ex:  # noqa: BLE001
            L["M_at_%g" % Rc_rs] = None; L["M_error_%g" % Rc_rs] = repr(ex)
    return {q: (float(v_) if isinstance(v_, np.floating) else v_) for q, v_ in L.items()}


# ------------------------------------------------------------------------------------------------------------------ the run
def run(a):
    global PO
    t_all = time.time()
    sys.path.append(a.page_code)
    import page_ops_v6 as _PO
    PO = _PO
    import torch
    torch.set_num_threads(a.threads)
    SA.set_device(-1)                                                          # torch on the CPU for the dense eigendecompositions
    FM = importlib.import_module(a.fields_module)
    J, Zf = load_run(a.run)
    ra = J["args"]; rs = float(ra["rs"]); g = float(ra["g"]); m = float(ra["m"])
    Rcs = [(a.R_collar_rs, a.R_collar_rs * rs), (a.R_cut_rs, a.R_cut_rs * rs)]
    out = {"script": "engine_massive_channel_D_v2.py", "run": a.run, "label": LABEL, "d15": D15, "args": vars(a), "utc_start": utc(),
           "rs": rs, "g": g, "m": m, "ell": float(ra["ell"]), "tau": float(ra["tau"]), "nsites": int(ra["nsites"]), "collars": Rcs, "slices": []}
    # the records' routes
    t_ = time.time()
    dr = density_route(J, Zf, Rcs)
    out["density_route"] = dr
    print("density route: %d records (T/rs %.3f to %.3f) in %.1f s; E_G %.6f -> %.6f; beyond %g r_s K_out %+.6e -> %+.6e (dust %+.3e, pairs %+.3e at the end)" % (
        len(dr), dr[0]["T_over_rs"], dr[-1]["T_over_rs"], time.time() - t_, dr[0]["E_G"], dr[-1]["E_G"], Rcs[0][0], dr[0]["beyond_%g" % Rcs[0][0]]["K_out"],
        dr[-1]["beyond_%g" % Rcs[0][0]]["K_out"], dr[-1]["beyond_%g" % Rcs[0][0]]["K_out_dust"], dr[-1]["beyond_%g" % Rcs[0][0]]["K_out_pairs"]), flush=True)
    Tz = np.asarray(Zf["T"], float)
    slices = [p for p in (a.slices.split(",") if a.slices else []) if p]
    want = [0]
    sl_T = []
    for p in slices:
        with np.load(p) as Z:
            sl_T.append(float(Z["T"]))
    for T in sl_T:
        want.append(int(np.argmin(np.abs(Tz - T))))
    if a.ledger_T:
        for x in a.ledger_T.split(","):
            want.append(int(np.argmin(np.abs(Tz / rs - float(x)))))
    out["ledger"] = []
    for k in sorted(set(want)):
        L = ledger_at(J, Zf, k, FM, Rcs)
        out["ledger"].append(L)
        print("ledger at T/rs %.3f: E_G %.6f, W %s (site %s), K %s, K_src %s, K_chi %s, K_out %s, eps_adj %s, eps_G %s; M(%g r_s) %s, M(%g r_s) %s (%.1f s)" % (
            L["T"] / rs, L["E_G"], fmt(L.get("W"), "%+.6e"), fmt(L.get("W_site"), "%+.6e"), fmt(L.get("K"), "%.6f"), fmt(L.get("K_src"), "%.6f"), fmt(L.get("K_chi"), "%.6f"),
            fmt(L.get("K_out"), "%+.4e"), fmt(L.get("eps_adj"), "%+.4e"), fmt(L.get("eps_G"), "%+.4e"), Rcs[0][0], fmt(L.get("M_at_%g" % Rcs[0][0]), "%.6f"),
            Rcs[1][0], fmt(L.get("M_at_%g" % Rcs[1][0]), "%.6f"), L["seconds"]), flush=True)
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    if a.records_only:
        out["utc_end"] = utc(); out["seconds"] = time.time() - t_all
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
        return
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    for p in slices:
        t_s = time.time()
        Z = np.load(p)
        T = float(Z["T"]); rec = recs.get(round(T, 6)) or {}
        E = float(rec["E"]); TH = 1.0 / (8.0 * math.pi * g * E)
        R = np.asarray(Z["R"], float)
        ks = [int(k) for k in Z["kappas"]] if a.kappas == "all" else [int(k) for k in a.kappas.split(",")]
        wts = {int(k): float(w) for k, w in zip(Z["kappas"], Z["weights"])}
        row = {"slice": os.path.basename(p), "T": T, "T_over_rs": T / rs, "dtype": str(Z["dtype"]), "round_c64": bool(a.round_c64), "E_G": E, "two_gE": 2 * g * E,
               "T_H": TH, "m_over_TH": m / TH, "exp_minus_m_over_TH": math.exp(-m / TH), "R_t": rec.get("R_t"), "trapping_zeros": rec.get("trapping_zeros"),
               "min_N_minus_absv": rec.get("min_N_minus_absv"), "kappas": ks, "channels": []}
        print("%s: T/rs %.3f (%s%s), E_G %.6f, 2gE %.4f, T_H %.6e, m/T_H %.4f, e^(-m/T_H) %.4e; the record's roots %s" % (
            row["slice"], T / rs, row["dtype"], ", rounded to complex64 (control)" if a.round_c64 else "", E, 2 * g * E, TH, m / TH, row["exp_minus_m_over_TH"],
            rec.get("trapping_zeros")), flush=True)
        Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
        k_f = int(np.argmin(np.abs(Tz - T)))
        row["applied_vs_fields_npz"] = [float(np.max(np.abs(Napp - np.asarray(Zf["N_applied"][k_f], float)))), float(np.max(np.abs(vapp - np.asarray(Zf["v_applied"][k_f], float))))]
        for kap in ks:
            ch = channel_D(R, m, kap, wts[kap], Napp, vapp, Z["sea_%d" % kap], Z["dust_%d" % kap], Z["f_%d" % kap], Z["R2_%d" % kap],
                           Z["e_%d" % kap] if ("e_%d" % kap) in Z.files else None, Rcs, TH, a.eps_rank, a.eps_control, a.round_c64, lambda s: print(s, flush=True),
                           eps_locs=(a.eps_loc, a.eps_loc_control))
            row["channels"].append(ch)
        Z.close()
        # weighted sums per region
        row["weighted"] = {}
        for Rc_rs, Rc in Rcs:
            key = "%g" % Rc_rs
            regs = [(c["w"], c["regions"][key]) for c in row["channels"]]
            W_ = lambda f_: float(sum(w * f_(r) for w, r in regs))
            wsum = {"E_out_Pm": W_(lambda r: r["E_out_Pm"]), "E_out_R2": W_(lambda r: r["E_out_R2"]), "E_out_dust": W_(lambda r: r["E_out_dust"]),
                    "E_out_pairs_Pm": W_(lambda r: r["E_out_pairs_Pm"]), "E_out_pairs_R2": W_(lambda r: r["E_out_pairs_R2"]),
                    "particles_Pm": W_(lambda r: r["particles_Pm"]), "holes_Pm": W_(lambda r: r["holes_Pm"]), "particles_R2": W_(lambda r: r["particles_R2"]),
                    "holes_R2": W_(lambda r: r["holes_R2"]), "particles_dust": W_(lambda r: r["particles_dust"]), "holes_dust": W_(lambda r: r["holes_dust"]),
                    "E_particles_Pm": W_(lambda r: r["E_particles_Pm"]), "E_holes_Pm": W_(lambda r: r["E_holes_Pm"]),
                    "E_particles_dust": W_(lambda r: r["E_particles_dust"]), "E_holes_dust": W_(lambda r: r["E_holes_dust"]),
                    "block_energy_Pm": W_(lambda r: r["block_energy_Pm"]),
                    "block_energy_R2": W_(lambda r: r["block_energy_R2"]), "N_plus": W_(lambda r: r["N_plus"]), "N_minus": W_(lambda r: r["N_minus"]),
                    "R2_N_plus": W_(lambda r: r["R2_N_plus"]), "R2_N_minus": W_(lambda r: r["R2_N_minus"]), "S_hull": W_(lambda r: r["hull"]["S_hull"]),
                    "S_hull_control": W_(lambda r: r["hull_control"]["S_hull"]), "dropped_bound": W_(lambda r: r["hull"]["dropped_bound"]),
                    "S_F": W_(lambda r: r["S_F"]), "S_F_loc_control": W_(lambda r: r["S_F_loc_control"]), "S_hull_loc_control": W_(lambda r: r["S_hull_loc_control"]),
                    "E_F": W_(lambda r: r["E_F"]), "sector_particles": W_(lambda r: r["sector_particles"]), "sector_holes": W_(lambda r: r["sector_holes"]),
                    "svd_fallbacks": int(sum(1 for _, r in regs for sq in r["sector"].values() for hq in sq["hulls"].values() if hq.get("svd_fallback"))),
                    "dropped_bound_control": W_(lambda r: r["hull_control"]["dropped_bound"]), "region_S_excess": W_(lambda r: r["region_S_excess"]),
                    "dust_weight": W_(lambda r: r["dust_weight"]), "n_total_Pm": W_(lambda r: r["n_total_Pm"]),
                    "max_vacuum_check": max(r["hull"]["vacuum_check"] for _, r in regs), "max_block_identity": max(abs(r["block_energy_Pm"] - r["block_energy_Pm_trace"]) for _, r in regs),
                    "max_refs_diff_op": max(r["refs_diff_op"] for _, r in regs)}
            # the weighted spectrum
            bins = {}
            for w, r in regs:
                for jj, b in r["bins_TH2"].items():
                    q = bins.setdefault(int(jj), {"count": 0, "n_Pm": 0.0, "n_R2": 0.0, "n_dust": 0.0, "E_Pm": 0.0, "E_R2": 0.0, "E_dust": 0.0})
                    q["count"] += b["count"]; q["n_Pm"] += w * b["n"][0]; q["n_R2"] += w * b["n"][1]; q["n_dust"] += w * b["n"][2]
                    q["E_Pm"] += w * b["E"][0]; q["E_R2"] += w * b["E"][1]; q["E_dust"] += w * b["E"][2]
            wsum["bins_TH2"] = {str(j): bins[j] for j in sorted(bins)}
            mb = []
            for i_, (lo, hi) in enumerate(zip(M_BINS[:-1], M_BINS[1:])):
                mb.append({"lo": lo, "hi": hi, "count": int(sum(r["bins_m"][i_]["count"] for _, r in regs)), "n_Pm": W_(lambda r: r["bins_m"][i_]["n"][0]),
                           "n_R2": W_(lambda r: r["bins_m"][i_]["n"][1]), "n_dust": W_(lambda r: r["bins_m"][i_]["n"][2]), "E_Pm": W_(lambda r: r["bins_m"][i_]["E"][0]),
                           "E_R2": W_(lambda r: r["bins_m"][i_]["E"][1]), "E_dust": W_(lambda r: r["bins_m"][i_]["E"][2])})
            wsum["bins_m"] = mb
            row["weighted"][key] = wsum
        w16 = row["weighted"]["%g" % Rcs[0][0]]
        print("  weighted beyond %g r_s: E_out P_- %+.6e (dust %+.4e, pairs %+.6e), R2 %+.6e; Killing-basis particles %.4e (E %+.4e), holes %.4e (E %+.4e) [dust: %.3e, %.3e]; eigenvalue sums %.4e, %.4e; escaped sector S_F %.6e (eps_loc control %.6e), E_F %+.4e; S_hull %.6e (control %.6e; dropped bound %.2e; SVD fallbacks %d); region S excess %+.4e; dust weight %.3e; checks: vacuum %.1e, block %.1e, refs %.1e (%.0f s)" % (
            Rcs[0][0], w16["E_out_Pm"], w16["E_out_dust"], w16["E_out_pairs_Pm"], w16["E_out_R2"], w16["particles_Pm"], w16["E_particles_Pm"], w16["holes_Pm"], w16["E_holes_Pm"],
            w16["particles_dust"], w16["holes_dust"], w16["N_plus"], w16["N_minus"], w16["S_F"], w16["S_F_loc_control"], w16["E_F"], w16["S_hull"], w16["S_hull_control"], w16["dropped_bound"],
            w16["svd_fallbacks"], w16["region_S_excess"],
            w16["dust_weight"], w16["max_vacuum_check"], w16["max_block_identity"], w16["max_refs_diff_op"], time.time() - t_s), flush=True)
        row["seconds"] = time.time() - t_s
        out["slices"].append(row)
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    out["utc_end"] = utc(); out["seconds"] = time.time() - t_all
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("done: %d slices in %.0f s" % (len(out["slices"]), out["seconds"]), flush=True)


# ------------------------------------------------------------------------------------------------------------------ the tables
def compose(a):
    runs = [json.load(open(p)) for p in a.compose.split(",")]
    L = ["## S1 D: the massive channel's own emission beyond the collar, on the stored full states (engine_massive_channel_D_v2.py)", "",
         "Label (every table): %s." % LABEL, "",
         "- %s; S_esc is the escaped sector's entropy by P01 sec. 2's hull: the sector F = the modes inside one spectral half of h_m with outside weight >= 1 - eps_loc (the Page collector's rule), the hull of the kept eigenvectors of F^dag (C - P_-) F taken in F's own coordinates; S_F = tr s(F^dag C F) beside it; the region's own entropy excess S(C_YY) - S((P_-)_YY) only as a comparator." % D15,
         "- per channel kappa = 1 .. 10 of the engine (weight 16 kappa, Chan.w); 'weighted' = summed with those weights.",
         "- C: the stored sea and dust columns, Loewdin-orthonormalized in float64; P_-: the Killing vacuum 1_{h_m < 0} at the slice's applied fields on the stored chain (to 32 r_s); R2: the stored continued reference.",
         "- the collar (the Page code's): a sharp step beyond R_c; E_out = Re Tr[chi h_m (C - P)]; the Killing spectrum in the eigenbasis of the block h_YY beyond R_c; T_H = 1/(8 pi g E_G) at the record's E_G; bins of T_H/2 in omega.",
         "- the density route: sum over the sites beyond R_c of (N eps + v J) of the stored shell totals (sources relative to R2; dust = total - pairs), every stored record.", ""]
    for F in runs:
        rs = F["rs"]; m = F["m"]
        L += ["### Run %s (%s to %s; %s)" % (F["run"], F.get("utc_start"), F.get("utc_end"), "round_c64 control" if F["args"].get("round_c64") else "as stored"), ""]
        for Rc_rs, _Rc in F["collars"]:
            key = "%g" % Rc_rs
            L += ["#### The coarse history at the slice clocks, beyond R_c = %g r_s (weighted): E_rad(t) and S_esc(t) of the massive channel. Label: %s" % (Rc_rs, LABEL), "",
                  "| T/r_s | dtype | E_G | 2gE | T_H a | m/T_H | e^(-m/T_H) | E_out (P_-): total; dust; pairs | E_out (R2) | density route (R2): total; dust; pairs | E_G - M(R_c) | Killing-basis particles; holes (P_-) | their energies | of which the dust: particles; holes | escaped sector (eps_loc %g): S_F; E_F | S_esc = S_hull, eps %g (dropped bound) | S_hull, eps %g | S_F; S_hull at eps_loc %g | region S excess (comparator) | dust weight |" % (
                      F["args"]["eps_loc"], F["args"]["eps_rank"], F["args"]["eps_control"], F["args"]["eps_loc_control"]),
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            dr = {round(r["T"], 6): r for r in F["density_route"]}
            lg = {round(x["T"], 6): x for x in F["ledger"]}
            for s in F["slices"]:
                w_ = s["weighted"][key]; d_ = (dr.get(round(s["T"], 6)) or {}).get("beyond_%s" % key) or {}; l_ = lg.get(round(s["T"], 6)) or {}
                Mc = l_.get("M_at_%s" % key)
                L.append("| %.3f | %s | %.6f | %.4f | %.6e | %.4f | %.4e | %+.6e; %+.3e; %+.6e | %+.6e | %s; %s; %s | %s | %.4e; %.4e | %+.4e; %+.4e | %.3e; %.3e | %.6e; %+.4e | %.6e (%.1e) | %.6e | %.6e; %.6e | %+.4e | %.4e |" % (
                    s["T_over_rs"], s["dtype"].replace("complex", "c"), s["E_G"], s["two_gE"], s["T_H"], s["m_over_TH"], s["exp_minus_m_over_TH"], w_["E_out_Pm"], w_["E_out_dust"],
                    w_["E_out_pairs_Pm"], w_["E_out_R2"], fmt(d_.get("K_out"), "%+.6e"), fmt(d_.get("K_out_dust"), "%+.3e"), fmt(d_.get("K_out_pairs"), "%+.6e"),
                    fmt((s["E_G"] - Mc) if Mc is not None else None, "%+.6e"), w_["particles_Pm"], w_["holes_Pm"], w_["E_particles_Pm"], w_["E_holes_Pm"],
                    w_["particles_dust"], w_["holes_dust"], w_["S_F"], w_["E_F"], w_["S_hull"], w_["dropped_bound"], w_["S_hull_control"], w_["S_F_loc_control"], w_["S_hull_loc_control"],
                    w_["region_S_excess"], w_["dust_weight"]))
            L.append("")
        # the ledger
        key = "%g" % F["collars"][0][0]
        cut_lab = "M1's cut unchanged over the records" if not any(x.get("cut_changed") for x in F["density_route"]) else "M1's cut CHANGED at %d records" % sum(1 for x in F["density_route"] if x.get("cut_changed"))
        L += ["#### The energy beyond the collar against the charge ledger of S5 (d) with -Delta W (engine_replay_v9.ledger_terms; E_G the record's charge at the engine's collar, the chain end; %s). Label: %s" % (cut_lab, LABEL), "",
              "| interval T/r_s | Delta E_G | sum a + b + c (M1) | -Delta W | Delta K_out | Delta eps_adj | Delta eps_G | remainder | sum of M1 closures | Delta E_out (P_-) beyond %s r_s | Delta E_out (R2) | Delta density route beyond %s (dust; pairs) | Delta W beyond %s (site form) | Delta (E_G - M(%s)) |" % (key, key, key, key),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        dr_all = F["density_route"]
        dr = {round(r["T"], 6): r for r in dr_all}
        lgs = sorted(F["ledger"], key=lambda x: x["T"])
        sl = {round(s["T"], 6): s for s in F["slices"]}
        pts = [x for x in lgs]

        def seg(x0, x1):
            t0_, t1_ = x0["T"], x1["T"]
            abc = [r["abc"] for r in dr_all if t0_ < r["T"] <= t1_ + 1e-9]
            clo = [r["M1_closure"] for r in dr_all if t0_ < r["T"] <= t1_ + 1e-9]
            if any(q is None for q in abc) or any(x0.get(q_) is None or x1.get(q_) is None for q_ in ("W", "K_out", "eps_adj", "eps_G")):
                return None
            dE = x1["E_G"] - x0["E_G"]; sabc = float(sum(abc)); dW = x1["W"] - x0["W"]
            dKo = x1["K_out"] - x0["K_out"]; dEa = x1["eps_adj"] - x0["eps_adj"]; dEg = x1["eps_G"] - x0["eps_G"]
            rem = dE - (sabc - dW + dKo + dEa + dEg)
            s0 = sl.get(round(t0_, 6)); s1 = sl.get(round(t1_, 6))
            dEoP = (s1["weighted"][key]["E_out_Pm"] - (s0["weighted"][key]["E_out_Pm"] if s0 else 0.0)) if s1 else None
            dEoR = (s1["weighted"][key]["E_out_R2"] - (s0["weighted"][key]["E_out_R2"] if s0 else 0.0)) if s1 else None
            d0 = dr[round(t0_, 6)]["beyond_%s" % key]; d1 = dr[round(t1_, 6)]["beyond_%s" % key]
            M0 = x0.get("M_at_%s" % key); M1 = x1.get("M_at_%s" % key)
            return {"T0": t0_ / rs, "T1": t1_ / rs, "dE_G": dE, "abc": sabc, "minus_dW": -dW, "dK_out": dKo, "deps_adj": dEa, "deps_G": dEg, "remainder": rem,
                    "closures": float(sum(q for q in clo if q is not None)), "dE_out_Pm": dEoP, "dE_out_R2": dEoR, "d_density": d1["K_out"] - d0["K_out"],
                    "d_density_dust": d1["K_out_dust"] - d0["K_out_dust"], "d_density_pairs": d1["K_out_pairs"] - d0["K_out_pairs"], "dW_out": d1["W_out"] - d0["W_out"],
                    "d_charge_beyond": ((x1["E_G"] - M1) - (x0["E_G"] - M0)) if (M0 is not None and M1 is not None) else None}
        segs = []
        for i_ in range(len(pts) - 1):
            q = seg(pts[i_], pts[i_ + 1])
            if q:
                q["kind"] = "interval"; segs.append(q)
        if len(pts) >= 2:
            for x1 in pts[1:]:
                q = seg(pts[0], x1)
                if q:
                    q["kind"] = "cumulative"; segs.append(q)
        for q in segs:
            L.append("| %s %.3f to %.3f | %+.6e | %+.6e | %+.6e | %+.4e | %+.4e | %+.4e | %+.2e | %+.2e | %s | %s | %+.6e (%+.3e; %+.6e) | %+.4e | %s |" % (
                "cumulative" if q["kind"] == "cumulative" else "", q["T0"], q["T1"], q["dE_G"], q["abc"], q["minus_dW"], q["dK_out"], q["deps_adj"], q["deps_G"], q["remainder"], q["closures"],
                fmt(q["dE_out_Pm"], "%+.6e") + (" (from 0 at T/r_s 0: no slice there)" if (q["T0"] == 0 and q["dE_out_Pm"] is not None) else ""), fmt(q["dE_out_R2"], "%+.6e"),
                q["d_density"], q["d_density_dust"], q["d_density_pairs"], q["dW_out"], fmt(q["d_charge_beyond"], "%+.6e")))
        F["_segments"] = segs
        L.append("")
        # the spectrum in units of m
        for Rc_rs, _Rc in F["collars"]:
            key = "%g" % Rc_rs
            L += ["#### The Killing spectrum beyond R_c = %g r_s in units of m (weighted; n = the summed occupation deviation, E = sum omega n; P_- reference; R2 and the dust's own share beside). Label: %s" % (Rc_rs, LABEL), "",
                  "| omega/m bin | " + " | ".join("T/r_s %.3f: levels; n (P_-); E (P_-); n (R2); n dust" % s["T_over_rs"] for s in F["slices"]) + " |",
                  "|---|" + "---|" * len(F["slices"])]
            for i_, (lo, hi) in enumerate(zip(M_BINS[:-1], M_BINS[1:])):
                cells = []
                for s in F["slices"]:
                    b = s["weighted"][key]["bins_m"][i_]
                    cells.append("%d; %+.3e; %+.3e; %+.3e; %+.3e" % (b["count"], b["n_Pm"], b["E_Pm"], b["n_R2"], b["n_dust"]))
                L.append("| [%s, %s) | %s |" % ("-inf" if lo < -1e8 else "%g" % lo, "inf" if hi > 1e8 else "%g" % hi, " | ".join(cells)))
            L.append("")
        # the spectrum in bins of T_H/2 near the thresholds
        for Rc_rs, _Rc in F["collars"][:1]:
            key = "%g" % Rc_rs
            L += ["#### The Killing spectrum beyond R_c = %g r_s in bins of T_H/2 (weighted, P_- reference): the bins within 40 T_H of each branch's threshold (the levels nearest omega = 0 on each side); every bin is in the json. Label: %s" % (Rc_rs, LABEL), ""]
            for s in F["slices"]:
                TH = s["T_H"]
                bins = {int(j): b for j, b in s["weighted"][key]["bins_TH2"].items()}
                pos = sorted(j for j in bins if j >= 0); neg = sorted((j for j in bins if j < 0), reverse=True)
                L += ["T/r_s %.3f (T_H a %.6e, m/T_H %.4f, e^(-m/T_H) %.4e): bin j covers omega in [j T_H/2, (j+1) T_H/2); omega/m at the bin's lower edge printed." % (s["T_over_rs"], TH, s["m_over_TH"], s["exp_minus_m_over_TH"]), "",
                      "| branch | j | omega/T_H | omega/m | levels | n (P_-) | E (P_-) | n (R2) | n dust |", "|---|---|---|---|---|---|---|---|---|"]
                for lab_, js in (("positive", [j for j in pos if j < (pos[0] + 80 if pos else 0)]), ("negative", [j for j in neg if j > (neg[0] - 80 if neg else 0)])):
                    for j in js:
                        b = bins[j]
                        L.append("| %s | %d | %.1f | %.5f | %d | %+.4e | %+.4e | %+.4e | %+.4e |" % (lab_, j, 0.5 * j, 0.5 * j * TH / m, b["count"], b["n_Pm"], b["E_Pm"], b["n_R2"], b["n_dust"]))
                L.append("")
        # per channel
        L += ["#### Per channel beyond R_c = %g r_s (P_-): the checks and the hull. Label: %s" % (F["collars"][0][0], LABEL), "",
              "| T/r_s | kappa | eig vs stored | n_neg | Loewdin (stored) | R2 Loewdin (stored) | E_out (P_-) | E_out (R2) | block energy: spectrum; trace | Killing-basis particles; holes | the eigenvalue sums of D_Y, positive; negative (its size, not occupations) | count of abs(lambda) above 1e-12, 1e-9, 1e-7, 1e-6, 1e-4, 1e-2 | max abs(lambda) | escaped sector: dim (+, -); S_F; E_F | hull dim (kept) | S_hull | L reduces Pi | s_min kept | S_hull (control eps) | region S: C; P_-; excess | levels abs(omega) < 0.9 m |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        key = "%g" % F["collars"][0][0]
        for s in F["slices"]:
            for c in s["channels"]:
                r = c["regions"][key]; h_ = r["hull"]; hc = r["hull_control"]
                sq = r["sector"]["%g" % F["args"]["eps_loc"]]
                L.append("| %.3f | %d | %s | %d | %.1e (%.1e) | %.1e (%.1e) | %+.6e | %+.6e | %+.6e; %+.6e | %.3e; %.3e | %.3e ; %.3e | %s | %.3e | %d (%d, %d); %.6e; %+.4e | %d (%d) | %.6e | %.1e | %s | %.6e (%d kept) | %.4e; %.4e; %+.3e | %d |" % (
                    s["T_over_rs"], c["kappa"], fmt(c["eig_vs_stored"], "%.1e"), c["n_negative"], c["loewdin_orthonormality"], c["stored_orthonormality"], c["R2_loewdin_orthonormality"],
                    c["R2_stored_orthonormality"], r["E_out_Pm"], r["E_out_R2"], r["block_energy_Pm"], r["block_energy_Pm_trace"], r["particles_Pm"], r["holes_Pm"], r["N_plus"], r["N_minus"],
                    ", ".join(str(r["lambda_counts"]["%g" % t]) for t in THRESH), r["lambda_max_abs"], sq["dim"], sq["dim_plus"], sq["dim_minus"], sq["S_F"], sq["E_F"],
                    h_["hull_dim"], h_["kept"], h_["S_hull"], h_["vacuum_check"],
                    fmt(h_.get("s_min_kept"), "%.1e"), hc["S_hull"], hc["kept"], r["region_S_C"], r["region_S_Pm"], r["region_S_excess"], r["levels_in_gap_0.9m"]))
        L.append("")
        # E_rad(t) by the density route
        key = "%g" % F["collars"][0][0]; key2 = "%g" % F["collars"][1][0]
        L += ["#### E_rad(t) by the density route at every stored record (printed every 1 r_s and at the slice clocks; every record in the json): the energy beyond the collar of the stored shell totals (sources relative to R2). Label: %s" % LABEL, "",
              "| T/r_s | E_G | m/T_H | beyond %s r_s: total; dust; pairs | W beyond %s (site) | beyond %s r_s: total; dust; pairs |" % (key, key, key2), "|---|---|---|---|---|---|"]
        slT = {round(s["T_over_rs"], 6) for s in F["slices"]}
        for r in F["density_route"]:
            t = r["T_over_rs"]
            if abs(t - round(t)) < 1e-9 or round(t, 6) in slT:
                b = r["beyond_%s" % key]; b2 = r["beyond_%s" % key2]
                L.append("| %.3f | %.6f | %.4f | %+.6e; %+.3e; %+.6e | %+.4e | %+.6e; %+.3e; %+.6e |" % (t, r["E_G"], r["m_over_TH"], b["K_out"], b["K_out_dust"], b["K_out_pairs"], b["W_out"],
                                                                                    b2["K_out"], b2["K_out_dust"], b2["K_out_pairs"]))
        L.append("")
        # checks
        L += ["#### Checks. Label: %s" % LABEL, ""]
        for s in F["slices"]:
            ch = s["channels"]
            L.append("- T/r_s %.3f: the slice's applied fields against the fields npz: N %.1e, v %.1e; eigenvalues against the stored e_kappa: largest %s; L reduces P_- (largest max|mu^2 - mu| of L^dag P_- L): %.1e; the block energy's two forms: largest difference %.1e; Loewdin orthonormality: largest %.1e (stored columns %.1e); the two references beyond R_c differ by at most %.3e in operator norm (largest over channels); %.0f s." % (
                s["T_over_rs"], s["applied_vs_fields_npz"][0], s["applied_vs_fields_npz"][1], fmt(max((c["eig_vs_stored"] for c in ch if c["eig_vs_stored"] is not None), default=None), "%.1e"),
                max(s["weighted"][k_]["max_vacuum_check"] for k_ in s["weighted"]), max(s["weighted"][k_]["max_block_identity"] for k_ in s["weighted"]),
                max(c["loewdin_orthonormality"] for c in ch), max(c["stored_orthonormality"] for c in ch), max(s["weighted"][k_]["max_refs_diff_op"] for k_ in s["weighted"]), s["seconds"]))
        L.append("- every number above is printed by the script.")
        L.append("")
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump({"script": "engine_massive_channel_D_v2.py --compose", "label": LABEL, "d15": D15, "runs": [F["run"] for F in runs], "inputs": a.compose.split(","),
                   "segments": [F.get("_segments") for F in runs],
                   "coarse_history": [[{"T_over_rs": s["T_over_rs"], "dtype": s["dtype"], "E_G": s["E_G"], "m_over_TH": s["m_over_TH"], "exp_minus_m_over_TH": s["exp_minus_m_over_TH"],
                                        "beyond": {k_: {q: v_ for q, v_ in s["weighted"][k_].items() if q not in ("bins_TH2", "bins_m")} for k_ in s["weighted"]}} for s in F["slices"]] for F in runs]},
                  open(a.out_json, "w"), indent=1, default=float)
    print("composed %s" % a.md, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="the run's dir/tag (its record json and fields npz)")
    ap.add_argument("--slices", default=None, help="comma-separated slice npz paths")
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--page-code", default="<internal>/v6", help="the directory holding page_ops_v6.py")
    ap.add_argument("--R-collar-rs", dest="R_collar_rs", type=float, default=16.0)
    ap.add_argument("--R-cut-rs", dest="R_cut_rs", type=float, default=24.0)
    ap.add_argument("--eps-rank", dest="eps_rank", type=float, default=1e-9)
    ap.add_argument("--eps-control", dest="eps_control", type=float, default=1e-6)
    ap.add_argument("--eps-loc", dest="eps_loc", type=float, default=1e-2, help="v2: the escaped sector's outside weight >= 1 - eps_loc (the Page collector's)")
    ap.add_argument("--eps-loc-control", dest="eps_loc_control", type=float, default=1e-3)
    ap.add_argument("--kappas", default="all")
    ap.add_argument("--ledger-T", dest="ledger_T", default=None, help="extra T/r_s values for the ledger's points")
    ap.add_argument("--round-c64", dest="round_c64", action="store_true", help="control: the columns rounded to complex64 before the readout")
    ap.add_argument("--records-only", dest="records_only", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--compose", default=None)
    a = ap.parse_args()
    if a.compose:
        return compose(a)
    run(a)


if __name__ == "__main__":
    main()
