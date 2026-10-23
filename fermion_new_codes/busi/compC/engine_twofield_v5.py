#!/usr/bin/env python
"""engine_twofield_v5.py (the busi lane, Computation C): version 4 with the second sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v2.md);
every change against version 4 is marked "v5:". One option, --light-coupling-ramp S1,...,1 with --light-ramp-bisect B (default off; off, every code path and
every record is version 4's, bitwise): at a record whose direct coupled Newton solve misses the stop rule, the record's fixed point is sought along the
homotopy G_s = Solve(heavy sources + s x light sources): the base stage s = 0 is the uncoupled map (the light channels not evaluated; the heavy sum bitwise
version 35's) from version 31's start; each listed s is version 32's Newton stage on the coupled map with the light sources and their exact J.v weighted by
s, from the previous stage's kept applied fields; a stage that misses the stop rule is retried over the half interval from the last converged s, at most B
times per target; a stall is recorded with its interval (the fold readout) and a closing stage at s = 1 runs from the last converged iterate; a base that
misses the stop rule is not walked, only closed. Only an s = 1 stage can be kept, by version 32's kept rule; G, the stop rule, the end rule, the records'
content and the propagation are unchanged, so the fixed point is unchanged. Per stage the record carries the status, the kept and confirming residuals, the
largest trial displacement, the kept applied fields' change from the stage's start with its field and site, and the central cell at the kept iterate
(N_dense(0), W/2g at r = 1, 2 and ell, rho_ell(0), J_ell/v_dense at r = 1). The first record is written when its kept residual is below --end-above, the
production's practice on its sigma/a 2 member (labelled).
Version 4's docstring follows.
engine_twofield_v4.py (the busi lane, Computation C): version 3 with the sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v1.md);
every change against version 3 is marked "v4:". One option, --light-start-continuation {off,coupling} (default off; with it off every code path
and every record is version 3's, bitwise): at the first record of a run (no previous record, no resume), when the direct coupled Newton solve
does not meet the stop rule, the record is re-solved in stages: A, the uncoupled map (the light sources removed from eps, J and the exact J.v,
the light channels not evaluated; the heavy sum bitwise version 35's) from version 31's start, the solve the production is known to finish;
B, the coupled map from stage A's kept iterate's applied fields (version 32's stage 2 rule); C, when B misses the stop rule, plain fixed-point
steps from B's kept iterate while they lower the residual (at most --light-start-plain-steps). The kept rule is version 32's (the smaller kept
residual of the direct solve and stage B/C; stage A is never kept, it is not the coupled fixed point); the map G, the stop rule, the end rule,
the records' content and the propagation are unchanged, so the fixed point is unchanged. With the option on and the coupling on every Newton
stage prints the largest displacement of any trial from its start (the ill-conditioning readout). With the coupling off the option does
nothing (said at launch).
Version 3's docstring follows.
engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every
change against version 2 is marked "v3:". (i) Criterion (2): the walk's banded Cholesky of K(s)^2 - d^2 certifies |spec K(s)|^2 >= d^2 - F with
F = 64 eps ||K||_G^2 its rounding floor, so a Cholesky-certified piece has the Weyl length sqrt(d^2 - F)/||dK||_G, no d below 2 sqrt(F) is tested
(the effective floor max(--light-d-min, 2 sqrt(F)), recorded), the start d carries the eigensolver's slack 8 eps ||K||_G, the mu S selection reads
the sector by the sign of <u|S|u>, and the full piece list is kept for the oracle (the record json drops it). (ii) Criterion (3): a certified walk
through a within-sector avoided crossing at the cut narrower than the convention's resolution is the spectral continuation, which refills the
sector; the lane's never-refilled LZ convention and the propagated sea pass diabatically. So after a certified walk the mu S selection is tested
against the one-step maximum-overlap weights of the trial eigenvectors within reach of the cut, and where a selected level carries weight below
1/2 (or an unselected one above) the evaluation falls back, marked "cut_crossing", to the declared convention; the fallback is
engine_ops_v3.continue_sea_v3 (the LZ-inspired maximum-overlap continuation at c_LZ, one step at the pathfinder's rates) in place of the one-step
assignment; the light label names the LZ convention from the first fallback. The heavy arithmetic is untouched (gate C's Part 1 stands).
Version 2's docstring follows.
engine_twofield_v2.py (the busi lane, Computation C): version 1 rebased on engine_collapse_v36 (the solve lead's turn 51, spec 43.1 (2):
"engine_collapse_v36 --field-kernel-cache --v36-fixes --nk-jvp exact ... the base the busi lane's C build takes, both options on in its runs, its
own regression and oracle its gate"). Every change against version 1 is marked "v2:": the base module is engine_collapse_v36 (still bound to
the name V35 so that version 1's code reads unchanged); v36's 2 options are added to main (--field-kernel-cache installs the cached kernels
on the same field modules as v36's main, --v36-fixes sets the base module's V36["fixes"]); v36's fixes (b) and (c), which sit in
nk_stage_v32's F that version 1 copies, are carried into that copy as v36 has them; v36's fix (a) lives in the base module's continuations,
which this engine calls there; the record carries v36's "v36" entry where v36 writes it. With both options off this engine is version 1
on a base that is version 35 bitwise (v36's gate G1v36).
Version 1's docstring follows.
engine_twofield_v1.py (the busi lane, Computation C; owner decision D16; the solve inbox's entry 62 item 5; the busi inbox's entry 55).
Version 36 = version 35 (imported; its helpers, field solve, Newton loop, heavy-field continuation and records reused unchanged) with the
light field added. The functions copied from version 35 (run, nk_stage_v32, newton_record_v32, newton_record, main) are copied verbatim and
every change inside them is marked "v36:"; everything else here is new.
(a) The light field: a second species of channels at m = 0 on the same chain, the channel operator page_ops_v5.chain_sparse(R, 0, kappa)
    (checked against engine_vacflow_v1.chain_sparse at build, the largest difference printed), multiplicity 16 kappa (the angular multiplicity
    of the spherical reduction; no other weight, the g_n guard), --light-kappas 1,2 sourcing and --light-guard-kappas 3 the truncation guard
    (propagated and read out, never sourcing); each light channel has its own sea (the ground state of its h(0) at the C-init fields: the
    negative levels, as the heavy sea is prepared), its own continued reference R2 and its own window, keyed ("L", kappa) in the engine's
    state; its sources dens(h, sea) - dens(h, R2) and the same with Q enter eps and J after the heavy channels' (and the exact J.v) when
    --light-coupling on; with --light-coupling off the light channels propagate and are read out but source nothing and the heavy field's
    arithmetic is version 35's bitwise (gate_regression_v1.py).
(b) The certified massless transport (--light-transport banded, the default): the reference 1(h_m - mu S < 0) with S = sum_n (-1)^n sigma_y,
    which commutes with the massless h_m at every N, v (h_m purely imaginary there; the 2 sectors conjugate, the spectrum symmetric); mu in the
    gap of the S = +1 sector at its occupation cut (midgap at every accepted record, n_plus levels below it); per evaluation the panel from the
    previous accepted fields to the trial fields, K(s) = (1 - s)(h_prev - mu_prev S) + s(h_trial - mu_trial S), walked in pieces
    s -> s + d/||dK||_G with d a certified lower bound on the gap of K(s) at the piece's start (the previous record's eigenvalues at s = 0;
    the banded Cholesky of K(s)^2 - d^2 elsewhere, a ladder of d down to --light-d-min), Weyl's inequality keeping every eigenvalue away from 0
    inside the piece; a walk reaching s = 1 certifies the spectral projector's continuity on the panel, so the continued reference at the trial
    fields is the trial eigenvectors with e_i < mu_trial sigma_i (sigma_i = <u_i|S|u_i>; degenerate blocks rotated to S's eigenbasis); a walk
    that fails falls back to version 3's maximum-overlap assignment, marked; the controls --light-transport v3 and v34-eigvals.
(c) Per record (COMMS rule 10): json "light_v36" (per channel the certificate, mu, the gap, pockets, E_inv = Tr[h_m (C - R2)], the collar cut
    current and inside energy, the Killing-bin deviations, the weight and energy beyond --s11-collars, the sources' totals; the fed totals;
    the guard's ratios; M1_parts_light and M1_balance_light; E_inv_heavy and E_inv_light); json "s11" (the mass function's sign outside the
    central cell with the 3 cases of P28 sec. 4 on the heavy dust, the light fed source beside; the P2 stop under --s11-stop when the
    coupling is on); fields npz eps_L<k>, J_L<k>, eps_light_fed, J_light_fed; spectra npz mask_L<k>, e_L<k>, nocc_L<k>; handoff npz the light
    stress parts; slices sea_L<k>, R2_L<k>, e_L<k>, mask_L<k>, sigma_L<k>, mu_L<k>, nplus_L<k>; --slices-light-c128 light-only slices;
    --resume-slice restarts the light state from a version 36 slice.
(d) The head: min(N - |v|) at T = 0 (the subcritical check), the light seas' counts and gaps, the operator check, [h_m, S], the first-return
    estimate on the T = 0 fields.
(e) --light-follow DIR/TAG: the light channels alone propagated on a stored run's applied fields (the first-order control on the production's
    own records, E_rad^(0)/E per record); no heavy state, no field solve.
Usage (the pathfinder): see RUN_ORDER_C_v1.md. The solve lane's modules are found through BUSI_ENGINE_DIR (default <internal>/engine)
and BUSI_PAGE_DIR (default <internal>/page), appended to sys.path after this file's own directory (a copy beside this file wins)."""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True                        # v36: the solve lane's modules are imported read-only (no __pycache__ written there)
for _d in (os.environ.get("BUSI_ENGINE_DIR", "<internal>/engine"), os.environ.get("BUSI_PAGE_DIR", "<internal>/page")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

import engine_collapse_v36 as V35                       # v2: the base is version 36 (turn 51, 43.1 (2)); the name V35 kept
import engine_collapse_v5 as C5
import engine_mixing_v1 as MIX
import engine_ops_v3 as OPS3
import engine_ops_v34_cert as CERT
import engine_stress_v2 as STR
import engine_vacflow_v1 as VF
import page_ops_v5 as PO5
from engine_collapse_v36 import (BK, TM, JV, OPT, SOLVE_TOL, DICT24, T_, eigT, dens, Chan, threshold_radii, solve, kernel_identities, indicator,
                                 prepare_dust, prepare_massless_ball_v33, surface_jets, inside_energy_sp, energy_cut_current_sp, clocked_gpu,
                                 clocked_host_sparse, dev_csr, commutator_norm, eig_at, continue_ref_v34, v34_on, v34_desc, resume_slice_v34,
                                 resume_state_v25, fmt, s11_record, s11_audit_save, s11_smoothed, v33_on, v33_desc, sync_dev, tm_reset, eigh_all,
                                 nk_loop_v32, confirm_v32, stage_entry_v32, print_stage_v32, drop_arrays_v32, v32_on, v32_options, v32_desc,
                                 v32_own_loop, v32_record_segment)

FLD = V35.FLD                                          # rebound with V35.FLD under --fields-v17 in main

TML = {}                                               # v36: the light field's per-record accounting
LS = {"s": 1.0}                                        # v5: the light sources' weight inside a ramp stage (1.0 everywhere else, so every other path is bitwise version 4's)


def tml_reset_v36():
    TML.clear()
    TML.update({"eigh_s": 0.0, "eigh_n": 0, "continuation_s": 0.0, "walk_pieces": 0, "walk_cholesky": 0, "walk_fallbacks": 0, "cut_crossings": 0, "readout_s": 0.0,
                "M1_s": 0.0, "handoff_s": 0.0, "propagation_s": 0.0, "jvp_products": 0, "jvp_dropped_pairs": 0, "jvp_cross_sector_pairs": 0})   # v3: cut_crossings


# ------------------------------------------------------------------------------------------------------------- v36: the light channel
class LightChan:
    """the light field's channel: page_ops_v5.chain_sparse(R, 0, kappa) (h_0 and the 2-step Q), multiplicity 16 kappa, m = 0; key ("L", kappa);
    the same attributes the engine's helpers read on a heavy channel (kappa, w, R, hs, Qs, h, Q, _coo)."""

    def __init__(self, kappa, nsites, half=False, sourcing=True):
        torch = BK["torch"]
        self.kappa = int(kappa); self.w = 16.0 * kappa; self.m = 0.0; self.key = ("L", int(kappa)); self.sourcing = bool(sourcing); self.species = "light"
        self.R = np.arange(nsites, dtype=float) + (0.5 if half else 1.0)
        h0, Q = PO5.chain_sparse(self.R, 0.0, float(kappa))
        self.hs = h0.tocsr(); self.hs.sort_indices(); self.Qs = Q.tocsr(); self.Qs.sort_indices()
        hv, qv = VF.chain_sparse(self.R, 0.0, float(kappa))
        dh = (self.hs - hv.tocsr()).tocsr(); dq = (self.Qs - qv.tocsr()).tocsr()
        self.operator_check = {"h_max_abs_diff_vs_vacflow": (float(abs(dh).max()) if dh.nnz else 0.0), "Q_max_abs_diff_vs_vacflow": (float(abs(dq).max()) if dq.nnz else 0.0),
                               "h_real_part_max": (float(abs(self.hs.real).max()) if self.hs.real.nnz else 0.0), "h_nnz": int(self.hs.nnz), "Q_nnz": int(self.Qs.nnz)}
        csr = lambda A: torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                                torch.as_tensor(A.data, dtype=torch.complex128), size=A.shape, device=BK["dev"])
        self.h = csr(self.hs); self.Q = csr(self.Qs)
        self._coo = None


def light_channels_v36(a):
    ks = [int(x) for x in str(a.light_kappas or "").split(",") if x.strip()]
    kg = [int(x) for x in str(a.light_guard_kappas or "").split(",") if x.strip()]
    out = [LightChan(k, a.nsites, bool(a.half_offset), sourcing=True) for k in ks]
    out += [LightChan(k, a.nsites, bool(a.half_offset), sourcing=False) for k in kg if k not in ks]
    return out


def light_sign_v36(nsites):
    """(-1)^n per site, n = 0 .. nsites - 1."""
    return (1.0 - 2.0 * (np.arange(nsites) % 2)).astype(float)


def light_S_v36(nsites):
    """S = sum_n (-1)^n sigma_y: the sign vector on the device and the host CSR (sigma_y = [[0, -i], [i, 0]])."""
    torch = BK["torch"]
    sgn = light_sign_v36(nsites)
    ev = 2 * np.arange(nsites)
    S = sp.csr_matrix((np.concatenate([-1j * sgn, 1j * sgn]), (np.concatenate([ev, ev + 1]), np.concatenate([ev + 1, ev]))), shape=(2 * nsites, 2 * nsites))
    return torch.as_tensor(sgn, dtype=torch.float64, device=BK["dev"]), S


def light_S_apply_v36(X, sgn_dev):
    torch = BK["torch"]
    out = torch.empty_like(X)
    s = sgn_dev.to(X.dtype)[:, None]
    out[0::2] = (-1j) * s * X[1::2]
    out[1::2] = (1j) * s * X[0::2]
    return out


def light_sigma_v36(U, sgn_dev):
    """<u|S|u> per column: sum_n (-1)^n 2 Im(conj(u_{n,0}) u_{n,1})."""
    sy = 2.0 * (U[0::2, :].conj() * U[1::2, :]).imag
    return (sgn_dev[:, None] * sy).sum(dim=0)


def light_sectorize_v36(e, U, sgn_dev, tol_block=1e-8):
    """numerically degenerate blocks of e (engine_ops_v3.blocks_of_v3) rotated in place to S's eigenbasis; returns (sigma per column (device
    float64), rotated blocks, the sector impurity max | |sigma| - 1 |)."""
    torch = BK["torch"]
    e_np = e.cpu().numpy().astype(float)
    n_rot = 0
    for (i, j) in OPS3.blocks_of_v3(e_np, tol_block):
        if j - i > 1:
            Ub = U[:, i:j]
            Ms = Ub.conj().T @ light_S_apply_v36(Ub, sgn_dev)
            lam, V = torch.linalg.eigh(0.5 * (Ms + Ms.conj().T))
            U[:, i:j] = Ub @ V
            n_rot += 1
    sigma = light_sigma_v36(U, sgn_dev)
    imp = float((sigma.abs() - 1.0).abs().max()) if sigma.numel() else 0.0
    return sigma, n_rot, imp


def light_select_v36(e_np, sig, mu):
    """the mu S selection: occupied where e - mu sigma < 0; the gap min |e - mu sigma|. v3: sigma enters by its sign (the sector label); its
    computed modulus differs from 1 by the impurity, which would otherwise enter the gap, the walk's start d, at |mu| times the impurity."""
    sgn = np.where(np.asarray(sig, float) > 0.0, 1.0, -1.0)                                      # v3
    kv = np.asarray(e_np, float) - mu * sgn
    return kv < 0.0, float(np.min(np.abs(kv)))


def light_mu_midgap_v36(e_np, sig, n_plus):
    """mu at the midpoint of the S = +1 sector's gap between its n_plus-th and (n_plus + 1)-th sorted levels; (None, None) without one."""
    ep = np.sort(np.asarray(e_np, float)[np.asarray(sig, float) > 0.0])
    if n_plus is None or not (0 < n_plus < len(ep)):
        return None, None
    return 0.5 * (ep[n_plus - 1] + ep[n_plus]), 0.5 * (ep[n_plus] - ep[n_plus - 1])


def light_sorted_consistent_v36(e_np, sig, occ):
    """whether the occupied set is the lowest levels of each sector (a prefix in energy order per sector); returns (ok, n_plus)."""
    e_np = np.asarray(e_np, float); sig = np.asarray(sig, float); occ = np.asarray(occ, bool)
    ok = True; n_plus = 0
    for sgn_ in (+1.0, -1.0):
        idx = np.where((sig > 0.0) if sgn_ > 0 else (sig <= 0.0))[0]
        o = occ[idx[np.argsort(e_np[idx], kind="stable")]]
        k = int(o.sum())
        if not bool(np.all(o[:k])) or bool(np.any(o[k:])):
            ok = False
        if sgn_ > 0:
            n_plus = k
    return ok, n_plus


def light_sector_inversions_v36(e_np, sig, occ):
    """occupied levels above the lowest empty level of the same sector, summed over the 2 sectors."""
    e_np = np.asarray(e_np, float); sig = np.asarray(sig, float); occ = np.asarray(occ, bool)
    inv = 0
    for sel in ((sig > 0.0), (sig <= 0.0)):
        if (sel & ~occ).any() and (sel & occ).any():
            e_low = float(np.min(e_np[sel & ~occ]))
            inv += int(np.sum(sel & occ & (e_np > e_low)))
    return inv


def band_upper_v36(A):
    """the upper band storage of a Hermitian host sparse matrix for scipy.linalg.cholesky_banded(lower=False): ab[u + i - j, j] = A[i, j]."""
    co = A.tocoo()
    off = co.col - co.row
    u = int(off.max()) if off.size else 0
    n = A.shape[0]
    ab = np.zeros((u + 1, n), dtype=complex)
    sel = off >= 0
    np.add.at(ab, (u - off[sel], co.col[sel]), co.data[sel])
    ab[u, :] = ab[u, :].real
    return ab, u


def pd_banded_v36(ab, d):
    """True when ab - d^2 (the main diagonal shifted) is positive definite by the banded Cholesky factorization."""
    ab2 = ab.copy(); ab2[-1, :] -= d * d
    try:
        sla.cholesky_banded(ab2, lower=False, check_finite=False)
        return True
    except (np.linalg.LinAlgError, ValueError):
        return False


def light_walk_v36(ch, f_prev, f_cur, mu0, mu1, g0, S, d_min=1e-7, max_pieces=4096):
    """the certified walk of the docstring's (b) over K(s) = (1 - s) K0 + s K1. Returns the certificate dict (no arrays).
    v3 (gate D's criterion (2)): the banded Cholesky of K(s)^2 - d^2 completing certifies |spec K(s)|^2 >= d^2 - F with F the factorization's
    backward error, taken as 64 eps G_K^2 (G_K the larger Gershgorin bound of K0 and K1, eps the double precision), so (i) a Cholesky-certified piece
    has the Weyl length sqrt(d^2 - F)/||dK||_G, (ii) no d with d^2 below 4 F is tested (d_floor = 2 sqrt(F); the effective floor max(d_min, d_floor);
    both recorded), (iii) the start d, from the previous record's eigenvalues, carries the eigensolver's slack 8 eps G_K, and (iv) the full piece
    list is kept whatever its length (the oracle reads it; the record json drops it)."""
    t0 = time.time()
    K0 = (clocked_host_sparse(ch, f_prev[0], f_prev[1]) - mu0 * S).tocsr()
    K1 = (clocked_host_sparse(ch, f_cur[0], f_cur[1]) - mu1 * S).tocsr()
    dK = (K1 - K0).tocsr()
    nG = OPS3.gershgorin(dK)
    eps = float(np.finfo(float).eps)                                                                   # v3
    GK = max(OPS3.gershgorin(K0), OPS3.gershgorin(K1))                                                 # v3
    F = 64.0 * eps * GK * GK                                                                            # v3: the Cholesky's rounding floor in d^2
    d_floor = 2.0 * math.sqrt(F)                                                                        # v3
    d_eff = max(float(d_min), d_floor)                                                                  # v3
    out = {"norm_dK_G": nG, "mu0": mu0, "mu1": mu1, "gap_start": g0, "pieces": 0, "cholesky_calls": 0, "d_min_certified": None, "certified": False,
           "why": None, "smallest_piece": None, "piece_list": None, "seconds": 0.0,
           "norm_K_G": GK, "rounding_F": F, "d_floor": d_floor, "d_min_effective": d_eff}               # v3
    if g0 is None or not (g0 > 0.0):
        out["why"] = "no start gap (%s)" % g0; out["seconds"] = time.time() - t0
        return out
    if nG == 0.0:
        out.update({"certified": True, "pieces": 0, "d_min_certified": g0, "seconds": time.time() - t0, "why": None})
        return out
    d = g0 - 8.0 * eps * GK                                                                             # v3: the eigensolver's slack in place of g0 (1 - 1e-9)
    if not (d >= d_eff):                                                                                # v3
        out.update({"why": "the start gap %.3e is below the floor %.1e" % (g0, d_eff), "seconds": time.time() - t0})
        return out
    s = 0.0; dmin = d; pieces = []; nchol = 0
    while True:
        d_rig = d if s == 0.0 else math.sqrt(max(d * d - F, 0.0))                                      # v3: the rigorous bound behind a Cholesky-certified d
        s1 = min(1.0, s + d_rig / nG)
        pieces.append((s, s1, d)); dmin = min(dmin, d)
        if s1 >= 1.0:
            break
        if len(pieces) >= max_pieces:
            out.update({"why": "more than %d pieces (||dK||_G %.3e, d %.3e)" % (max_pieces, nG, d), "pieces": len(pieces), "cholesky_calls": nchol,
                        "d_min_certified": None, "seconds": time.time() - t0})
            return out
        s = s1
        Ks = (K0 + s * dK).tocsr()
        ab, _u = band_upper_v36(Ks @ Ks)
        d_new = None; d_try = d
        while d_try >= d_eff:                                                                           # v3: nothing below the effective floor is tested
            nchol += 1
            if pd_banded_v36(ab, d_try):
                d_new = d_try
                break
            d_try *= 0.5
        if d_new is None:
            out.update({"why": "the gap of K(s) at s = %.6f fell below the floor %.1e (d_min %.1e, the Cholesky's rounding floor %.1e)" % (s, d_eff, d_min, d_floor),
                        "pieces": len(pieces), "cholesky_calls": nchol, "d_min_certified": None, "seconds": time.time() - t0})
            return out
        if d_new == d and d_new * 2.0 < 1.0:
            nchol += 1
            if pd_banded_v36(ab, 2.0 * d_new):
                d_new = 2.0 * d_new
        d = d_new
    out.update({"certified": True, "pieces": len(pieces), "cholesky_calls": nchol, "d_min_certified": dmin, "smallest_piece": min(p[1] - p[0] for p in pieces),
                "piece_list": [[float(p[0]), float(p[1]), float(p[2])] for p in pieces], "seconds": time.time() - t0, "why": None})   # v3: the full list
    return out


def light_cut_check_v3(ref_prev, e_np, sig, occ, U, mu1, nG, gap_end):
    """v3 (gate D's criterion (3)): the declared convention's verdict at the cut. The weights ||P_prev u_j||^2 of the trial eigenvectors within
    reach of the cut (|e - mu sigma| <= 4 ||dK||_G + 2 gap_end: every level that can have crossed the cut on the panel, and the cut's neighbours);
    the mu S selection agrees with the one-step maximum-overlap assignment there when every selected level has weight above 1/2 and every
    unselected one below 1/2. A selected level of weight below 1/2 is a within-sector crossing at the cut that the spectral continuation took
    adiabatically (the sector refilled) and the convention takes diabatically. Returns the counts and extreme weights (no arrays)."""
    torch = BK["torch"]
    sgn = np.where(np.asarray(sig, float) > 0.0, 1.0, -1.0)
    kv = np.asarray(e_np, float) - mu1 * sgn
    half = 4.0 * float(nG) + 2.0 * float(gap_end)
    idx = np.where(np.abs(kv) <= half)[0]
    out = {"window_levels": int(len(idx)), "window_half_width": half, "min_weight_selected": None, "max_weight_unselected": None, "crossing_levels": 0}
    if len(idx) == 0:
        return out
    Uw = U[:, torch.as_tensor(idx, dtype=torch.int64, device=U.device)]
    w = (ref_prev.conj().T @ Uw).abs().pow(2).sum(dim=0).cpu().numpy().astype(float)
    o = np.asarray(occ, bool)[idx]
    if o.any():
        out["min_weight_selected"] = float(w[o].min())
    if (~o).any():
        out["max_weight_unselected"] = float(w[~o].max())
    out["crossing_levels"] = int(np.sum(o & (w < 0.5)) + np.sum((~o) & (w > 0.5)))
    return out


def light_continue_v36(a, ch, window, f_cur, e, U, n_sea, sgn_dev, S_host):
    """the light reference continued from the window's last entry to the trial fields (the docstring's (b)); returns (ref, info, mask) with
    info["_sigma"] the sectors of the (rotated in place) trial eigenbasis."""
    torch = BK["torch"]
    t0 = time.time()
    prev = window[-1]
    sigma, n_rot, imp = light_sectorize_v36(e, U, sgn_dev, a.light_tol_block)
    e_np = e.cpu().numpy().astype(float); sig = sigma.cpu().numpy().astype(float)
    route = a.light_transport
    info = {"rotated_blocks": n_rot, "sigma_impurity_max": imp, "nneg": int((e_np < 0).sum()), "route": route, "reanchored": False}
    if route == "v3":
        ref, inf3, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)
        info.update(inf3); info.update({"mode": "v3: %s" % inf3.get("mode"), "certified": None, "fallback": False, "why": None})
    elif route == "v34-eigvals":
        ref, inf4, mask = CERT.continue_sea_v34(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse, "eigvals", a.weyl_refine_max)
        info.update(inf4); info.update({"mode": "v34 eigvals: %s" % inf4.get("mode"), "fallback": False, "why": None})
    else:
        n_plus = prev.get("n_plus"); mu0 = prev.get("mu")
        mu1, gap_plus = light_mu_midgap_v36(e_np, sig, n_plus) if n_plus is not None else (None, None)
        walk = None; why = None; occ = None; gap_end = None; cut = None                                  # v3: cut
        if imp > a.light_sigma_tol:
            why = "sector impurity %.2e" % imp
        elif mu0 is None:
            why = "no anchored mu (the previous assignment left a sector inversion)"
        elif mu1 is None:
            why = "no midgap of the S = +1 sector at the trial fields"
        else:
            walk = light_walk_v36(ch, prev["f"], f_cur, mu0, mu1, prev.get("gap"), S_host, a.light_d_min, a.light_max_pieces)
            if not walk["certified"]:
                why = walk["why"]
        if why is None:
            occ, gap_end = light_select_v36(e_np, sig, mu1)
            if int(occ.sum()) != n_sea:
                why = "the mu S selection counts %d against n_sea %d" % (int(occ.sum()), n_sea)
        if why is None and walk["norm_dK_G"] > 0.0:                                                     # v3: the convention's verdict at the cut
            cut = light_cut_check_v3(prev["ref"], e_np, sig, occ, U, mu1, walk["norm_dK_G"], gap_end)
            if cut["crossing_levels"] > 0:
                why = ("within-sector crossing at the cut below the convention's resolution: %d of the %d level(s) within reach of the cut are selected with weight "
                       "below 1/2 on the previous reference or unselected with weight above 1/2 (certified d_min %.2e, ||dK||_G %.2e); the spectral continuation "
                       "refills the sector; the LZ convention is taken" % (cut["crossing_levels"], cut["window_levels"], walk["d_min_certified"], walk["norm_dK_G"]))
        if why is None:
            ref = U[:, torch.as_tensor(occ, device=U.device)]; mask = occ
            info.update({"mode": "mu S selector, certified walk (%d pieces)" % walk["pieces"], "verified": True, "isolation_verified": True, "certified": walk,
                         "nsub": walk["pieces"], "fallback": False, "why": None, "mu": mu1, "gap": gap_end, "weyl_gap": gap_end, "gap_plus_half": gap_plus,
                         "n_plus": n_plus, "margin_min": walk["d_min_certified"], "norm_dH_G": walk["norm_dK_G"], "cut_check": cut, "cut_crossing": False})   # v3
        else:
            ref, inff, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)             # v3: the declared LZ convention
            info.update(inff)
            info.update({"mode": "fallback: the LZ convention, version 3's maximum-overlap continuation (%s) (%s)" % (inff.get("mode"), why), "verified": False,
                         "isolation_verified": False, "certified": walk, "nsub": int(inff.get("nsub") or 1), "fallback": True, "why": why, "margin_min": None,
                         "cut_check": cut, "cut_crossing": bool(cut is not None and cut["crossing_levels"] > 0)})                          # v3
    mask = np.asarray(mask, bool)
    if route != "banded" or info.get("fallback"):
        ok, n_plus_new = light_sorted_consistent_v36(e_np, sig, mask)
        mu_new, gap_plus = light_mu_midgap_v36(e_np, sig, n_plus_new) if ok else (None, None)
        gap_new = light_select_v36(e_np, sig, mu_new)[1] if mu_new is not None else None
        info.update({"mu": mu_new, "gap": gap_new, "weyl_gap": (gap_new if gap_new is not None else info.get("weyl_gap")), "gap_plus_half": gap_plus,
                     "n_plus": (n_plus_new if ok else None), "reanchored": bool(ok and info.get("fallback"))})
    info.update({"n_occ": int(mask.sum()), "n_sea": n_sea, "pockets_occupied_positive": int(np.sum(mask & (e_np > 0.0))),
                 "holes_empty_negative": int(np.sum(~mask & (e_np < 0.0))), "inversions_sector": light_sector_inversions_v36(e_np, sig, mask)})
    info["n_pocket"] = info["pockets_occupied_positive"] + info["holes_empty_negative"]
    info["_sigma"] = sigma
    info["seconds"] = time.time() - t0
    return ref, info, mask


def light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host):
    """the light channels' eigendecompositions at the fields (N_a, v_a) and their continuation, written into cache_ under the channel keys."""
    for chL in chl:
        t_ = time.time()
        e, U = eigT(clocked_gpu(chL, N_a, v_a))
        sync_dev()
        TML["eigh_s"] += time.time() - t_; TML["eigh_n"] += 1
        t_c = time.time()
        s_ = st[chL.key]
        ref, cinfo, mask = light_continue_v36(a, chL, s_["window"], (N_a, v_a), e, U, s_["n_sea"], sgn_dev, S_host)
        cinfo["_mask"] = mask
        TML["continuation_s"] += time.time() - t_c
        c_ = cinfo.get("certified")
        if isinstance(c_, dict):
            TML["walk_pieces"] += int(c_.get("pieces") or 0); TML["walk_cholesky"] += int(c_.get("cholesky_calls") or 0)
        if cinfo.get("fallback"):
            TML["walk_fallbacks"] += 1
        if cinfo.get("cut_crossing"):                                                                  # v3
            TML["cut_crossings"] += 1
        cache_[chL.key] = (None, e, U, ref, cinfo)


def light_init_v36(a, chl, N, v, sgn_dev, S_host):
    """the light seas at the fields (N, v): the ground state of h_m there (its negative levels), the mu S selector anchored at the midgap of
    the S = +1 sector at zero; returns (state entries keyed by channel, the head's light entry)."""
    torch = BK["torch"]
    stL, headL = {}, {}
    for chL in chl:
        e, U = eigT(clocked_gpu(chL, N, v))
        sigma, n_rot, imp = light_sectorize_v36(e, U, sgn_dev, a.light_tol_block)
        e_np = e.cpu().numpy().astype(float); sig = sigma.cpu().numpy().astype(float)
        neg = e_np < 0.0
        n_plus0 = int(np.sum((sig > 0.0) & neg))
        mu0, gap_plus = light_mu_midgap_v36(e_np, sig, n_plus0)
        if mu0 is None:
            raise SystemExit("v36: the light channel kappa %d has no midgap of its S = +1 sector at T = 0 (n_plus %d of %d)" % (chL.kappa, n_plus0, int(np.sum(sig > 0))))
        occ, gap0 = light_select_v36(e_np, sig, mu0)
        n_sea = int(occ.sum())
        ref0 = U[:, torch.as_tensor(occ, device=U.device)].clone()
        Hm = clocked_host_sparse(chL, N, v)
        comm = (Hm @ S_host - S_host @ Hm).tocsr()
        stL[chL.key] = {"sea": ref0.clone(), "dust": torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), "f": np.zeros(0), "ref": ref0,
                        "n_sea": n_sea, "window": [{"ref": ref0, "f": (np.asarray(N, float).copy(), np.asarray(v, float).copy()), "gap": gap0, "weyl_gap": gap0,
                                                    "mask": occ, "mu": mu0, "n_plus": n_plus0}]}
        headL[str(chL.kappa)] = {"kappa": chL.kappa, "w": chL.w, "sourcing": chL.sourcing, "n_sea": n_sea, "n_sea_equals_nsites": bool(n_sea == a.nsites),
                                 "n_negative_levels": int(neg.sum()), "n_plus": n_plus0, "n_sector_plus": int(np.sum(sig > 0)), "mu0": mu0, "gap_mu0": gap0,
                                 "gap_plus_half": gap_plus, "min_abs_e": float(np.min(np.abs(e_np))), "selection_equals_negative_set": bool(np.array_equal(occ, neg)),
                                 "sigma_impurity_max": imp, "rotated_blocks": n_rot, "commutator_h_S_max_abs": (float(abs(comm).max()) if comm.nnz else 0.0),
                                 "h_m_real_part_max": (float(abs(Hm.real).max()) if Hm.real.nnz else 0.0), "operator_check": chL.operator_check}
        del e, U
    return stL, headL


def light_fields_new_v36(chl):
    out = {"eps_light_fed": [], "J_light_fed": []}
    for chL in chl:
        out["eps_L%d" % chL.kappa] = []; out["J_L%d" % chL.kappa] = []
    return out


def light_resume_v36(a, chl, sgn_dev, S_host):
    """version 34 (a) for the light field: the light state from the same complex128 slice, the window entry the slice's, the sea propagated over
    the record's step by the eigendecomposition recomputed at the slice's applied fields; the light rows of the fields npz to the slice."""
    torch = BK["torch"]
    rdir, rtag = os.path.split(a.resume_run)
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag)))
    for k_ in ("light_kappas", "light_guard_kappas", "light_transport", "light_coupling"):
        if str(J["args"].get(k_)) != str(getattr(a, k_)):
            raise SystemExit("v36: the run's argument %s is %r, this call's %r" % (k_, J["args"].get(k_), getattr(a, k_)))
    Z = np.load(a.resume_slice)
    Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
    Ts = float(Z["T"]); Tz = np.asarray(Zf["T"], float); ks = int(np.argmin(np.abs(Tz - Ts)))
    N_s = np.asarray(Z["N_applied"], float); v_s = np.asarray(Z["v_applied"], float)
    stL = {}; baseL = {}; chk = {}
    for chL in chl:
        k = chL.kappa
        if "sea_L%d" % k not in Z.files:
            raise SystemExit("v36: the slice %s carries no light state for kappa %d" % (a.resume_slice, k))
        e, U = eigT(clocked_gpu(chL, N_s, v_s))
        sigma, n_rot, imp = light_sectorize_v36(e, U, sgn_dev, a.light_tol_block)
        de = float(np.max(np.abs(e.cpu().numpy() - np.asarray(Z["e_L%d" % k], float))))
        sea = torch.as_tensor(np.asarray(Z["sea_L%d" % k]), dtype=torch.complex128, device=BK["dev"])
        ref = torch.as_tensor(np.asarray(Z["R2_L%d" % k]), dtype=torch.complex128, device=BK["dev"])
        mask = np.unpackbits(np.asarray(Z["mask_L%d" % k], np.uint8))[: 2 * a.nsites].astype(bool)
        mu = float(Z["mu_L%d" % k]) if np.isfinite(float(Z["mu_L%d" % k])) else None
        n_plus = int(Z["nplus_L%d" % k]) if int(Z["nplus_L%d" % k]) >= 0 else None
        sig = sigma.cpu().numpy().astype(float); e_np = e.cpu().numpy().astype(float)
        gap = light_select_v36(e_np, sig, mu)[1] if mu is not None else None
        baseL[chL.key] = (dens(chL.h, sea), dens(chL.Q, sea))
        ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
        sea_n = U @ (ph[:, None] * (U.conj().T @ sea))
        stL[chL.key] = {"sea": sea_n, "dust": torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), "f": np.zeros(0), "ref": ref, "n_sea": int(ref.shape[1]),
                        "window": [{"ref": ref, "f": (N_s.copy(), v_s.copy()), "gap": gap, "weyl_gap": gap, "mask": mask, "mu": mu, "n_plus": n_plus}]}
        chk[str(k)] = {"eig_vs_stored_max": de, "mask_bits": int(mask.sum()), "mu": mu, "gap": gap, "n_plus": n_plus, "sigma_impurity_max": imp}
        del e, U
    fieldsL = light_fields_new_v36(chl)
    for k_ in list(fieldsL):
        if k_ in Zf.files:
            fieldsL[k_] = [np.asarray(Zf[k_][i]) for i in range(ks + 1)]
    sync_dev()
    print("v36: the light state restarted from the slice (T/rs %.4f): per channel the recomputed eigenvalues against the stored ones %s" % (
        Ts / a.rs, ", ".join("%s: %.1e" % (k_, q["eig_vs_stored_max"]) for k_, q in chk.items())), flush=True)
    return stL, fieldsL, baseL, {"slice": a.resume_slice, "channels": chk}


def first_return_estimate_v36(R, N, v, rs, collars_rs, launch_rs=(0.0, 2.0)):
    """a light signal on the T = 0 fields: out at the coordinate speed N - |v| to the chain's end, back at N + |v| to each collar."""
    R = np.asarray(R, float); N = np.asarray(N, float); v = np.asarray(v, float)
    c_out = np.maximum(N - np.abs(v), 1e-9); c_in = N + np.abs(v)
    dR = np.gradient(R)
    out = {}
    for r0 in launch_rs:
        sel_o = R >= r0 * rs
        t_out = float(np.sum(dR[sel_o] / c_out[sel_o]))
        out["from_%g_rs_to_end" % r0] = t_out / rs
        for Rc in collars_rs:
            sel_i = R >= Rc * rs
            out["from_%g_rs_back_to_%g_rs" % (r0, Rc)] = (t_out + float(np.sum(dR[sel_i] / c_in[sel_i]))) / rs
    out["note"] = ("a light signal at the coordinate speeds N - |v| out and N + |v| in on the T = 0 fields, in units of r_s; the heavy pairs at 1.5 to 3 m "
                   "move at 0.745 to 0.943 of it; nothing returns into 24 r_s before the run's end at T/rs %g" % 40)
    return out


def light_slice_arrays_v36(chl, st, cache, masksL, sigL, muL, nplusL, dt_):
    arrs = {}
    for chL in chl:
        k = chL.kappa
        _hm, e, U, ref, _ci = cache[chL.key]; s_ = st[chL.key]
        arrs["sea_L%d" % k] = s_["sea"].cpu().numpy().astype(dt_); arrs["R2_L%d" % k] = ref.cpu().numpy().astype(dt_)
        arrs["e_L%d" % k] = e.cpu().numpy(); arrs["mask_L%d" % k] = np.packbits(masksL[chL.key]); arrs["sigma_L%d" % k] = np.asarray(sigL[chL.key], float)
        arrs["mu_L%d" % k] = float(muL[chL.key]) if muL[chL.key] is not None else float("nan")
        arrs["nplus_L%d" % k] = int(nplusL[chL.key]) if nplusL[chL.key] is not None else -1
    arrs["kappas_light"] = np.array([chL.kappa for chL in chl]); arrs["weights_light"] = np.array([chL.w for chL in chl]); arrs["light_sourcing"] = np.array([int(chL.sourcing) for chL in chl])
    return arrs


def c_audit_save_v36(a, chl, st, cache, masksL, sigL, muL, nplusL, N_app, v_app, R, T, rs, eps_light_fed, J_light_fed, dsL, JsL, light_label):
    """the light field's companion to version 33's audit at a stop record: the light columns, references, masks, sources."""
    path = os.path.join(a.out, "c_audit_light_%s_T%07.3f.npz" % (a.tag, T / rs))
    arrs = {"T": T, "rs": rs, "R": R, "N_applied": N_app, "v_applied": v_app, "eps_light_fed": eps_light_fed, "J_light_fed": J_light_fed, "light_label": light_label}
    arrs.update(light_slice_arrays_v36(chl, st, cache, masksL, sigL, muL, nplusL, np.complex128))
    for chL in chl:
        arrs["eps_L%d" % chL.kappa] = dsL[chL.key]; arrs["J_L%d" % chL.kappa] = JsL[chL.key]
    np.savez(path, **arrs)
    return path


def v36_desc(a):
    if a.light_follow:
        return "light-follow on %s: the light channels alone on the stored applied fields (kappas %s, guard %s, transport %s)" % (a.light_follow, a.light_kappas, a.light_guard_kappas, a.light_transport)
    return ("light field (D16, candidate 1a): page_ops_v5's massless channel operator, kappas %s sourcing (multiplicity 16 kappa), guard %s; coupling %s; transport %s "
            "(d_min %g, raised to the Cholesky's rounding floor 2 sqrt(64 eps) ||K||_G where that is larger, at most %d pieces; the fallback the LZ convention, "
            "version 3's maximum-overlap continuation, marked; a within-sector crossing at the cut hands over to it); collars %s r_s; the P2 stop %s") % (   # v3
        a.light_kappas, a.light_guard_kappas, a.light_coupling, a.light_transport, a.light_d_min, a.light_max_pieces, a.s11_collars, "on" if a.s11_stop else "off")


# --------------------------------------------------------------------------------------------------------- v36: the light-follow mode
def light_follow_v36(a, chl, R, sgn_dev, S_host, label):
    """the docstring's (e): the light channels on a stored run's applied fields; writes collapse_<tag>_lightfollow.json and _lightfollow.npz."""
    torch = BK["torch"]
    rdir, rtag = os.path.split(a.light_follow)
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag)))
    Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
    ra = J["args"]
    for k_ in ("nsites", "rs", "tau", "g", "half_offset"):
        if ra.get(k_) is not None and float(ra[k_]) != float(getattr(a, k_)):
            raise SystemExit("v36: --light-follow needs this call's --%s to equal the stored run's %r (got %r)" % (k_.replace("_", "-"), ra[k_], getattr(a, k_)))
    rs = a.rs; tau = float(Zf["tau"]) if "tau" in Zf.files else a.tau
    Tz = np.asarray(Zf["T"], float)
    recs_by_T = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    N0 = np.asarray(Zf["N_cinit"], float); v0 = np.asarray(Zf["v_cinit"], float)
    st, headL = light_init_v36(a, chl, N0, v0, sgn_dev, S_host)
    tz0 = {"min_N_minus_absv_sites": float(np.min(N0 - np.abs(v0)))}
    print("v36 light-follow: %s; %d stored rows; the light seas prepared at the C-init fields: %s" % (a.light_follow, len(Tz), json.dumps(headL, default=float)[:1200]), flush=True)
    collars = [float(x_) for x_ in a.s11_collars.split(",") if x_.strip()]
    rows = []; spec = {"T": []}
    for chL in chl:
        for nm in ("nocc_L%d", "e_L%d", "mask_L%d", "eps_L%d", "J_L%d"):
            spec[nm % chL.kappa] = []
    sl = [float(x) for x in a.slices_light_c128.split(",") if x.strip()]
    if sl and a.slice_dir:
        os.makedirs(a.slice_dir, exist_ok=True)
    path = os.path.join(a.out, "collapse_%s_lightfollow.json" % a.tag)
    t_run = time.time()
    for k in range(len(Tz)):
        tr = time.time(); tml_reset_v36()
        T = float(Tz[k]); rec = recs_by_T.get(round(T, 6), {})
        N_app = np.asarray(Zf["N_applied"][k], float); v_app = np.asarray(Zf["v_applied"][k], float)
        cache = {}
        light_eval_v36(a, chl, st, cache, N_app, v_app, sgn_dev, S_host)
        cut = int((rec.get("M1_parts") or {}).get("cut") or min(max(round(float(rec.get("r_collar") or (a.nsites - 9))) - 1, 2), a.nsites - 10))
        row = {"T": T, "T_over_rs": T / rs, "E": rec.get("E"), "R_t": rec.get("R_t"), "min_N_minus_absv": rec.get("min_N_minus_absv"), "channels": {}}
        Einv_src = 0.0; Einv_all = 0.0; IE_src = 0.0; IE_all = 0.0; eps_fed = np.zeros(a.nsites); J_fed = np.zeros(a.nsites)
        for chL in chl:
            _h, e, U, ref, ci = cache[chL.key]; s_ = st[chL.key]
            mask = np.asarray(ci.pop("_mask"), bool); sigma = ci.pop("_sigma"); sig = sigma.cpu().numpy().astype(float)
            A = U.conj().T @ s_["sea"]
            nocc = (A.abs() ** 2).sum(dim=1)
            e64 = e.to(torch.float64); e_np = e64.cpu().numpy()
            E_sea = float((e64 * nocc).sum()); E_ref = float(e_np[mask].sum()); E_inv = E_sea - E_ref
            ds = dens(chL.h, s_["sea"]) - dens(chL.h, ref); Js = dens(chL.Q, s_["sea"]) - dens(chL.Q, ref)
            H_new = clocked_host_sparse(chL, N_app, v_app); Hr = dev_csr(H_new[:2 * cut, :])
            cols = torch.cat([s_["sea"], ref], dim=1); fs = np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])])
            IE = float(energy_cut_current_sp(H_new, cols, fs, cut)); Hin = float(inside_energy_sp(Hr, cols, fs, 2 * cut))
            dev = nocc.cpu().numpy() - mask.astype(float)
            bins = {"(-inf,-1)": float(dev[e_np < -1.0].sum()), "(-1,0)": float(dev[(e_np >= -1.0) & (e_np < 0.0)].sum()), "(0,1)": float(dev[(e_np >= 0.0) & (e_np < 1.0)].sum()), "(1,inf)": float(dev[e_np >= 1.0].sum())}
            beyond = {}
            for Rc_rs in collars:
                i0 = int(np.searchsorted(np.asarray(R, float), Rc_rs * rs, side="right")); sel = np.asarray(R, float) > Rc_rs * rs
                beyond["%g" % Rc_rs] = {"number": float((s_["sea"][2 * i0:, :].abs() ** 2).sum() - (ref[2 * i0:, :].abs() ** 2).sum()),
                                        "energy_density_route": float(np.sum(N_app[sel] * ds[sel] + v_app[sel] * Js[sel]))}
            w_ = (ref.conj().T @ U).abs().pow(2).sum(dim=0).cpu().numpy() if ci.get("fallback") else None
            cert = {k_: v_ for k_, v_ in (ci.get("certified") or {}).items() if k_ != "piece_list"} if isinstance(ci.get("certified"), dict) else ci.get("certified")
            row["channels"][str(chL.kappa)] = {"sourcing": chL.sourcing, "E_inv": E_inv, "E_sea": E_sea, "E_ref": E_ref, "I_E_cut": IE, "H_in": Hin, "cut": cut, "eps_sum": float(ds.sum()),
                                               "J_sum": float(Js.sum()), "nocc_deviation_bins": bins, "beyond": beyond, "mode": ci.get("mode"), "verified": ci.get("verified"),
                                               "fallback": ci.get("fallback"), "why": ci.get("why"), "certified": cert, "mu": ci.get("mu"), "gap": ci.get("gap"), "n_plus": ci.get("n_plus"),
                                               "n_pocket": ci.get("n_pocket"), "pockets_occupied_positive": ci.get("pockets_occupied_positive"), "holes_empty_negative": ci.get("holes_empty_negative"),
                                               "inversions_sector": ci.get("inversions_sector"), "sigma_impurity_max": ci.get("sigma_impurity_max"), "rotated_blocks": ci.get("rotated_blocks"),
                                               "cut_check": ci.get("cut_check"), "cut_crossing": ci.get("cut_crossing"),                     # v3
                                               "min_abs_e": float(np.min(np.abs(e_np)))}
            Einv_all += chL.w * E_inv; IE_all += chL.w * IE
            if chL.sourcing:
                Einv_src += chL.w * E_inv; IE_src += chL.w * IE; eps_fed += chL.w * ds; J_fed += chL.w * Js
            spec["nocc_L%d" % chL.kappa].append(nocc.cpu().numpy()); spec["e_L%d" % chL.kappa].append(e_np.astype(np.float64)); spec["mask_L%d" % chL.kappa].append(np.packbits(mask))
            spec["eps_L%d" % chL.kappa].append(ds); spec["J_L%d" % chL.kappa].append(Js)
            # the window and the propagation
            s_["ref"] = ref
            s_["window"] = [{"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap"), "weyl_gap": ci.get("weyl_gap", ci.get("gap")), "mask": mask, "mu": ci.get("mu"), "n_plus": ci.get("n_plus")}]
            if abs(T / rs - round(T / rs, 6)) < 1e-9 and any(abs(T / rs - x) < 1e-6 for x in sl) and a.slice_dir:
                arrs = {"T": T, "rs": rs, "R": R, "N_applied": N_app, "v_applied": v_app, "dtype": "complex128", "sea_L%d" % chL.kappa: s_["sea"].cpu().numpy(),
                        "R2_L%d" % chL.kappa: ref.cpu().numpy(), "e_L%d" % chL.kappa: e_np, "mask_L%d" % chL.kappa: np.packbits(mask), "sigma_L%d" % chL.kappa: sig,
                        "mu_L%d" % chL.kappa: (ci.get("mu") if ci.get("mu") is not None else float("nan")), "nplus_L%d" % chL.kappa: (ci.get("n_plus") if ci.get("n_plus") is not None else -1)}
                np.savez(os.path.join(a.slice_dir, "lightfollow_%s_k%d_T%07.2f.npz" % (a.tag, chL.kappa, T / rs)), **arrs)
            ph = torch.exp(-1j * e.to(torch.complex128) * tau)
            s_["sea"] = U @ (ph[:, None] * A)
            del A, U, e, cols
        sync_dev()
        E_ = row["E"]
        row.update({"E_inv_light_sourcing": Einv_src, "E_inv_light_all": Einv_all, "E_rad0_over_E": (Einv_src / E_) if E_ else None, "I_E_light_sourcing": IE_src, "I_E_light_all": IE_all,
                    "eps_fed_sum": float(eps_fed.sum()), "J_fed_sum": float(J_fed.sum()),
                    "truncation_guard": {"E_inv_guard_over_sourcing": ((Einv_all - Einv_src) / Einv_src) if Einv_src != 0.0 else None, "I_E_guard_over_sourcing": ((IE_all - IE_src) / IE_src) if IE_src != 0.0 else None},
                    "timing": dict(TML), "seconds": time.time() - tr})
        rows.append(row); spec["T"].append(T)
        print("light-follow T/rs %.3f: E %s E_rad0/E %s E_inv (sourcing, all) %.4e %.4e I_E (sourcing, all) %.3e %.3e; per channel [%s] (%.1f s; eigh %d in %.1f s, walks %d pieces %d Cholesky %d fallbacks)" % (
            T / rs, fmt(E_, "%.6f"), fmt(row["E_rad0_over_E"], "%.3e"), Einv_src, Einv_all, IE_src, IE_all,
            "; ".join("k %s: %s mu %s gap %s pockets %s E_inv %.3e" % (k_, "cert" if q["verified"] else "FALLBACK(%s)" % q["why"], fmt(q["mu"], "%.3e"), fmt(q["gap"], "%.2e"), q["n_pocket"], q["E_inv"])
                      for k_, q in row["channels"].items()), time.time() - tr, TML["eigh_n"], TML["eigh_s"], TML["walk_pieces"], TML["walk_cholesky"], TML["walk_fallbacks"]), flush=True)
        if k % 10 == 0 or k == len(Tz) - 1:
            with open(path, "w") as fh:
                json.dump({"script": "engine_twofield_v5.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v4
                           "records": rows}, fh, default=float)
            np.savez(os.path.join(a.out, "collapse_%s_lightfollow.npz" % a.tag), R=R, kappas_light=np.array([c.kappa for c in chl]), weights_light=np.array([c.w for c in chl]),
                     light_sourcing=np.array([int(c.sourcing) for c in chl]), **{k_: np.array(v_) for k_, v_ in spec.items()})
    print("light-follow done: %d records in %.0f s; %s" % (len(rows), time.time() - t_run, path), flush=True)


# --------------------------------------------------------------------------------------------------- version 35's run, with v36 marks
def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)
    label = ("quasi-static (tau = %g); ell = %d; reference: %s (%s); site fields: %s; engine_fields_v16 (one outward initial-value problem, the momentum "
             "equation kept everywhere, the flat-slice gauge continued and labelled); economy: lagged first iterate, stop |dE/E| < %g and fields < %g, "
             "%s iteration, at most %d iterations (the smallest-residual iterate returned, flagged); tau = %g; collar policy %s; %s") % (
        a.tau, a.ell, a.reference, ("the continued sea, the Landau-Zener-inspired maximum-overlap convention (c_LZ = %g; ops v3: one step, no window)" % a.c_LZ) if a.reference == "R2" else "the propagated sea",
        a.site_fields, a.stop_E, a.stop_fields, a.mix, a.maxit, a.tau, OPT["collar_policy"], DICT24)
    v32 = v32_on(a)
    if v32:
        label += "; v32: " + v32_desc(a)
    if v33_on(a):
        label += "; v33: " + v33_desc(a)
    if v34_on(a):
        label += "; v34: " + v34_desc(a)
    label += "; v36: " + v36_desc(a)                                                                   # v36
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m) if a.m > 0 else None
    chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    chl = light_channels_v36(a)                                                                        # v36: the light channels (possibly none)
    light_on = bool(chl) and a.light_coupling == "on"                                                  # v36: the source coupling's switch
    if getattr(a, "light_start_continuation", "off") != "off":                                           # v4: the first record's continuation (the note's section 2)
        if light_on:
            label += ("; v4: the first record's continuation in the light coupling when the direct solve misses the stop rule (stage A the uncoupled map from version 31's "
                      "start, stage B the coupled map from A's kept applied fields, stage C at most %d plain steps while they lower the residual; version 32's kept rule)" % a.light_start_plain_steps)
        else:
            print("v4: --light-start-continuation %s acts on the coupled solve only; with the coupling %s and %d light channel(s) it changes nothing" % (
                a.light_start_continuation, a.light_coupling, len(chl)), flush=True)
    if getattr(a, "light_ramp", None):                                                                  # v5: the light-coupling ramp (C_s2_start_judgment_v2.md, section 2)
        if light_on:
            label += ("; v5: the light-coupling ramp at a record whose direct solve misses the stop rule (base s = 0 the uncoupled map from version 31's start, then s = %s by "
                      "version 32's Newton stage from the previous stage's kept applied fields with the light sources and their exact J.v weighted by s; a missed stage retried "
                      "over the half interval, at most %d bisections per target; a stall recorded with its interval and closed by a stage at s = 1; only an s = 1 stage kept, by "
                      "version 32's rule); the first record written when its kept residual is below the end rule %s, the production's practice on its sigma/a 2 member" % (
                          a.light_coupling_ramp, a.light_ramp_bisect, a.end_above))
        else:
            print("v5: --light-coupling-ramp %s acts on the coupled solve only; with the coupling %s and %d light channel(s) it changes nothing" % (
                a.light_coupling_ramp, a.light_coupling, len(chl)), flush=True)
    sgn_dev, S_host = light_S_v36(a.nsites)                                                            # v36
    if a.light_follow:                                                                                 # v36 (e)
        if not chl:
            raise SystemExit("v36: --light-follow needs light channels")
        light_follow_v36(a, chl, R, sgn_dev, S_host, label)
        return
    shell = [float(x) * rs for x in a.shell.split(",")] if a.shell else None
    kid = kernel_identities(R, a.ell) if a.site_fields == "averaged" else None
    if kid:
        print("kernel identities (site fields averaged): max |K_0 average of 1 - 1| %.2e; max |K_1 average of c r - c R_n| %.2e (relative %.2e); K_0 weight sums %.15f to %.15f" % (
            kid["K0_constant_max_abs"], kid["K1_linear_max_abs"], kid["K1_linear_max_rel"], kid["K0_weight_sums"][0], kid["K0_weight_sums"][1]), flush=True)
    t0 = time.time()
    resume = None
    fieldsL = light_fields_new_v36(chl)                                                                # v36
    if a.resume_from:
        if chl:
            raise SystemExit("v36: --resume-from (the v17/v19/v24 replay) is not supported with light channels; use --resume-slice")
        st, f_ref, stored, head, resume = resume_state_v25(a, chs, R, shell)
        head = dict(head); head["resume"] = resume; head["label_v19"] = label; head["kernel_identities_v19"] = kid
        E0 = head["E0"]
        N_cinit, v_cinit = stored["N_cinit"], stored["v_cinit"]
        recs = stored["records"]; fields = stored["fields"]
        step0 = stored["k_res"] + 1; T = stored["T_next"]
    elif a.resume_slice:
        st, f_ref, recs, fields, head, E0, N_cinit, v_cinit, step0, T, base_resume34, resume, pre34 = resume_slice_v34(a, chs, R)
        head["label_v34"] = label
        if chl:                                                                                        # v36: the light state from the same slice
            stL, fieldsL, baseL_resume, headLr = light_resume_v36(a, chl, sgn_dev, S_host)
            st.update(stL); base_resume34.update(baseL_resume); head["light_v36_resume"] = headLr
    else:
        prep_ell = a.prep_ell if a.prep_ell is not None else a.ell
        if a.massless_ball:
            N, v, dust, info, head = prepare_massless_ball_v33(chs, a.g, R, prep_ell, rs, a)
            for k_ in ("f_kin", "f_kin_bulk_frame"):
                head[k_] = float("nan") if head.get(k_) is None else head[k_]
            for k_ in ("discarded_norm_max", "gram_cond_max", "stationarity_max"):
                head["projection"][k_] = float("nan") if head["projection"].get(k_) is None else head["projection"][k_]
        else:
            N, v, dust, info, head = prepare_dust(chs, a.g, R, prep_ell, NB, a.R0_over_rs * rs, a.m, a.reading, shell)
        head["prep_ell"] = prep_ell; head["evolution_ell"] = a.ell
        head["kernel"] = "the Gaussian of standard deviation ell (per-axis variance sigma^2 = ell^2, plus a^2/12 where the cell average enters); prepared at ell %d, evolved at ell %d" % (prep_ell, a.ell)
        N_cinit, v_cinit = N.copy(), v.copy()
        head.update({"label": label, "rs_rest_label": rs, "R0": a.R0_over_rs * rs, "seconds": time.time() - t0, "kernel_identities": kid})
        print("initial data (%s): N_B %.2f E0 %.6f passes %d f_kin %.4f (bulk frame %.4f) discarded %.2e Gram cond %.2e stationarity %.2e; the spline's distance from the dense fields at the sites N %.3e v %.3e; residuals W %.1e 1/N %.1e; source splines %s; centre series %.6f; collar %s (%.0f s)" % (
            label, head["N_B"], head["E0"], head["passes"], head["f_kin"], head["f_kin_bulk_frame"], head["projection"]["discarded_norm_max"],
            head["projection"]["gram_cond_max"], head["projection"]["stationarity_max"], head["interp_floor_N"] or float("nan"), head["interp_floor_v"] or float("nan"),
            head["residual_W"] or float("nan"), head["residual_invN"] or float("nan"), head["source_spline_error"], head["centre_series_ratio"] or float("nan"),
            head["collar_margin"], head["seconds"]), flush=True)
        if a.massless_ball:
            b_ = head["ball"]
            print("massless ball (S11): R0 %.2f, x0 %.2f, sigma %.2f, orbitals %s, k0 %.10f; E0 %.10f (target %.6f); the bare energy (weighted) %.6f = %.6f E" % (
                b_["R0"], b_["x0"], b_["sigma"], b_["orbitals"], head["k0"], head["E0"], head["E_target"], b_["E_bare_weighted"], b_["E_bare_over_E"]), flush=True)
        E0 = head["E0"]
        st = {}
        for ch in chs:
            e, U = eigT(clocked_gpu(ch, N, v))
            neg = e < 0
            Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), np.zeros(0)))
            _en = e.real.cpu().numpy(); _k = int(neg.sum())
            gap = float(_en[_k] - _en[_k - 1]) if 0 < _k < len(_en) else None
            ref0 = U[:, neg].clone()
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": _k, "window": [{"ref": ref0, "f": (N.copy(), v.copy()), "gap": gap}]}
            del e, U
        if chl:                                                                                        # v36: the light seas (the ground state of h(0)) and the subcritical check
            stL, headL = light_init_v36(a, chl, N, v, sgn_dev, S_host)
            st.update(stL)
            tz0 = FLD.trapping_radius(N, v, R, info)
            head["light_v36"] = {"channels": headL, "coupling": a.light_coupling, "transport": a.light_transport,
                                 "subcritical_T0": {"min_N_minus_absv_sites": float(np.min(N - np.abs(v))), "min_N_minus_absv_dense": tz0.get("min"), "r_min": tz0.get("r_min"),
                                                    "zeros": tz0.get("zeros")},
                                 "first_return": first_return_estimate_v36(R, N, v, rs, [float(x_) for x_ in a.s11_collars.split(",") if x_.strip()]),
                                 "chain_end_over_rs": float(R[-1]) / rs}
            print("light field (v36): subcritical check at T = 0: min(N - |v|) %.4f on the sites, %s on the dense fields; chain end %.1f r_s; first return %s; per channel %s" % (
                head["light_v36"]["subcritical_T0"]["min_N_minus_absv_sites"], fmt(tz0.get("min"), "%.4f"), float(R[-1]) / rs, json.dumps(head["light_v36"]["first_return"], default=float),
                "; ".join("kappa %s: n_sea %d (= nsites %s) n_plus %d mu0 %.3e gap %.3e min|e| %.2e selection = negative set %s impurity %.1e [h,S] %.1e Re h %.1e operator vs vacflow %.1e" % (
                    k_, q["n_sea"], q["n_sea_equals_nsites"], q["n_plus"], q["mu0"], q["gap_mu0"], q["min_abs_e"], q["selection_equals_negative_set"], q["sigma_impurity_max"],
                    q["commutator_h_S_max_abs"], q["h_m_real_part_max"], max(q["operator_check"]["h_max_abs_diff_vs_vacflow"], q["operator_check"]["Q_max_abs_diff_vs_vacflow"])) for k_, q in headL.items())), flush=True)
        f_ref = (N.copy(), v.copy())
        recs = []
        fields = {"T": [], "N": [], "v": [], "eps_tot": [], "J_tot": [], "eps_pairs": [], "J_pairs": [], "N_applied": [], "v_applied": [], "r_nodes": [], "N_nodes": [], "v_nodes": []}
        step0 = 0; T = 0.0
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    cstrL = {chL.key: STR.ChanStress(chL, torch, BK["dev"]) for chL in chl}                               # v36
    r_nodes_fixed = FLD.coarse_grid(R, a.ell)
    drtol_last = {"value": None, "T_over_rs": None}
    for r_ in reversed(recs):
        ind_ = r_.get("indicator") or {}
        if ind_.get("delta_rtol_source") == "this record" and ind_.get("delta_rtol") is not None:
            drtol_last = {"value": ind_["delta_rtol"], "T_over_rs": r_["T"] / rs}
            break
    nsteps = int(round(a.T_over_rs * rs / a.tau))
    if a.m1_simpson_steps == "auto":
        q8 = max(1, nsteps // 8)
        simpson_steps = sorted({min(q8, nsteps - 1), nsteps // 2, max(nsteps - q8, 1)}) if nsteps >= 2 else []
    else:
        simpson_steps = [int(x) for x in a.m1_simpson_steps.split(",") if x.strip()]
    sl128 = [float(x) for x in a.slices_c128.split(",") if x.strip()]
    sl64 = [float(x) for x in a.slices_c64.split(",") if x.strip()]
    slL = [float(x) for x in a.slices_light_c128.split(",") if x.strip()]                                 # v36: light-only slices
    if (sl128 or sl64 or slL) and a.slice_dir:
        os.makedirs(a.slice_dir, exist_ok=True)
    path = os.path.join(a.out, "collapse_%s.json" % a.tag)
    prevW = None
    ind_state = {"first_failure": None}
    for r_ in recs:
        if (r_.get("indicator") or {}).get("indicator_pass") is False:
            ind_state["first_failure"] = r_["T"] / rs
            break
    lz_state = {}
    deg_state = {}
    unconv_state = {"first": None}
    light_lz_state = {}                                                                                 # v36: the first fallback per light channel
    for r_ in recs:                                                                                     # v36: from the stored records on a restart
        for kk, c in ((r_.get("light_v36") or {}).get("channels") or {}).items():
            if c.get("fallback") and int(kk) not in light_lz_state:
                light_lz_state[int(kk)] = r_["T"] / rs
    t_run0 = time.time()
    solver_meta = {"rtol": 1e-12, "atol": 1e-16, "collar_policy": OPT["collar_policy"], "support_rel": OPT["support_rel"], "module": FLD.__name__,
                   "solver": "solve_constraints_ivp", "site_fields": a.site_fields, "engine": "engine_collapse_v25", "stop_E": a.stop_E, "stop_fields": a.stop_fields,
                   "mix": a.mix, "mix_alpha": (a.mix_alpha if a.mix == "damped" else None), "maxit": a.maxit, "c_LZ": a.c_LZ, "window_max": a.window_max, "delta_rtol_every": a.delta_rtol_every, "streams": int(a.streams),
                   "resumed": resume is not None}
    if v32:
        solver_meta["v32"] = v32_options(a)
    light_meta = {"kappas": [c.kappa for c in chl], "weights": [c.w for c in chl], "sourcing": [c.sourcing for c in chl], "coupling": a.light_coupling, "transport": a.light_transport,
                  "d_min": a.light_d_min, "max_pieces": a.light_max_pieces, "operator": "page_ops_v5.chain_sparse(R, 0, kappa)", "reference": "1(h_m - mu S < 0), S = sum (-1)^n sigma_y",
                  "eps_tot_includes_fed_light": True, "note": "eps_heavy = eps_tot - eps_light_fed; eps_L<k>, J_L<k> are the channel's unweighted sources (times 16 kappa when fed)"}
    end32 = None
    base_last = None
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc", "b", "sy_k"):
            hand["%s_%s" % (q, part)] = []
    if chl:                                                                                             # v36: the light field's handoff
        for q in ("F_face", "fc", "b", "sy_k"):
            hand["%s_light" % q] = []
        hand["eps_light_sea"] = []; hand["eps_light_R2"] = []
    spectra = {"T": []}
    for ch in chs:
        spectra["mask_%d" % ch.kappa] = []; spectra["e_%d" % ch.kappa] = []
    for chL in chl:                                                                                     # v36
        spectra["mask_L%d" % chL.kappa] = []; spectra["e_L%d" % chL.kappa] = []; spectra["nocc_L%d" % chL.kappa] = []

    s11_store = {}
    end33 = None
    end34 = None
    if a.resume_slice:
        base_last = base_resume34
        for k_, v_ in pre34["hand"].items():
            hand[k_] = list(v_)
        for k_, v_ in pre34["spectra"].items():
            spectra[k_] = list(v_)
        for k_, v_ in pre34["s11"].items():
            s11_store[k_] = list(v_)
        lz_state.update(pre34["lz_state"]); deg_state.update(pre34["deg_state"]); unconv_state["first"] = pre34["unconv_first"]

    def save_fields():
        if s11_store:
            np.savez_compressed(os.path.join(a.out, "collapse_%s_s11.npz" % a.tag), **{k_: np.array(v_) for k_, v_ in s11_store.items()})
        extra = {}                                                                                      # v36: the light arrays of the fields npz
        if chl:
            extra = {k_: np.array(v_) for k_, v_ in fieldsL.items()}
            extra.update({"kappas_light": np.array([c.kappa for c in chl]), "weights_light": np.array([c.w for c in chl]), "light_sourcing": np.array([int(c.sourcing) for c in chl]),
                          "light_meta": json.dumps(light_meta)})
        np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),
                            eps_tot=np.array(fields["eps_tot"]), J_tot=np.array(fields["J_tot"]), eps_pairs=np.array(fields["eps_pairs"]), J_pairs=np.array(fields["J_pairs"]),
                            N_applied=np.array(fields["N_applied"]), v_applied=np.array(fields["v_applied"]), N_cinit=N_cinit, v_cinit=v_cinit,
                            g=a.g, ell=a.ell, tau=a.tau, solver=json.dumps(solver_meta),
                            r_nodes=np.array(fields["r_nodes"], dtype=object), N_nodes=np.array(fields["N_nodes"], dtype=object),
                            v_nodes=np.array(fields["v_nodes"], dtype=object), label=label, **extra)
        if hand["T"]:
            np.savez_compressed(os.path.join(a.out, "collapse_%s_handoff.npz" % a.tag), R=R, r_nodes=r_nodes_fixed, g=a.g, ell=a.ell, m=a.m, tau=a.tau,
                                kappas=np.array([ch.kappa for ch in chs]), weights=np.array([ch.w for ch in chs]), **{k: np.array(v_) for k, v_ in hand.items()})
        if spectra["T"]:
            np.savez(os.path.join(a.out, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})
    last_prop_s = 0.0
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    collars_rs = [float(x_) for x_ in a.s11_collars.split(",") if x_.strip()]                             # v36
    for step in range(step0, nsteps + 1):
        tr = time.time()
        tm_reset()
        tml_reset_v36()                                                                                 # v36
        TM["propagation_prev_s"] = last_prop_s
        mixer = MIX.Anderson(beta=1.0) if a.mix == "anderson" else None
        converged = False
        it_hist = []
        t_d = time.time()
        base = {}
        for ch in chs:
            s_ = st[ch.kappa]
            base[ch.kappa] = (dens(ch.h, s_["sea"]), dens(ch.Q, s_["sea"]), dens(ch.h, s_["dust"], s_["f"]), dens(ch.Q, s_["dust"], s_["f"]))
        for chL in chl:                                                                                 # v36: the light seas' base densities
            base[chL.key] = (dens(chL.h, st[chL.key]["sea"]), dens(chL.Q, st[chL.key]["sea"]))
        TM["densities_s"] += time.time() - t_d

        def sources(cache_):
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            if light_on:                                                                                # v36: the light field's sources, after the heavy sum
                for chL in chl:
                    if chL.sourcing:
                        ref = cache_[chL.key][3]; hs_, Qs_ = base[chL.key]
                        eps += chL.w * (hs_ - dens(chL.h, ref)); J += chL.w * (Qs_ - dens(chL.Q, ref))
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def sources_with(base_, cache_):
            """version 32: sources(cache_) with the base densities base_ (stage 1 of the continuation: the midpoint covariance's)."""
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base_[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            if light_on:                                                                                # v36
                for chL in chl:
                    if chL.sourcing:
                        ref = cache_[chL.key][3]; hs_, Qs_ = base_[chL.key]
                        wL = chL.w if LS["s"] == 1.0 else chL.w * LS["s"]                                       # v5: the ramp's weight (1.0 outside a ramp stage: bitwise)
                        eps += wL * (hs_ - dens(chL.h, ref)); J += wL * (Qs_ - dens(chL.Q, ref))
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def sources_with_heavy(base_, cache_):
            """v4: sources_with(base_, cache_) without the light sum (the uncoupled map of stage A); the heavy loop is sources_with's first loop verbatim,
            so its sum is bitwise version 35's."""
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base_[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def do_solve(srcs_):
            t_s = time.time()
            out = solve(srcs_[0], srcs_[1], a.g, R, a.ell, None)
            TM["solve_s"] += time.time() - t_s; TM["solve_n"] += 1
            if out[3].get("ivp"):
                TM["solve_nfev"] += int(out[3]["ivp"]["nfev"])
            return out
        N_app, v_app = f_ref[0].copy(), f_ref[1].copy()
        cache = {ch.kappa: (None, None, None, (st[ch.kappa]["ref"] if a.reference == "R2" else st[ch.kappa]["sea"]), {"mode": "lagged"}) for ch in chs}
        for chL in chl:                                                                                 # v36: the lagged first iterate of the light references
            cache[chL.key] = (None, None, None, st[chL.key]["ref"], {"mode": "lagged"})
        TM["lagged"] += len(chs)
        srcs = sources(cache)
        Nn, vn, Wn, info = do_solve(srcs)
        E_prev = FLD.charge(info, a.g) if not info.get("failed") else None
        if not info.get("failed"):
            it_hist.append([None, float(np.max(np.abs(Nn - N_app))), float(np.max(np.abs(vn - v_app)))])
        k_it = 0
        best = None
        solve32 = None
        if a.mix == "newton" and not info.get("failed") and v32:
            k_it, converged, best, cache, srcs, Nn, vn, Wn, info, N_app, v_app, solve32 = newton_record_v32(
                a, chs, st, cache, sources_with, base, (base_last if (step > step0 or a.resume_slice) else None), ((f_ref[0].copy(), f_ref[1].copy()) if (step > step0 or a.resume_slice) else None),
                do_solve, Nn, vn, E_prev, it_hist, T, chl=chl, light_on=light_on, sgn_dev=sgn_dev, S_host=S_host,                       # v36
                ramp=getattr(a, "light_ramp", None), ramp_bisect=getattr(a, "light_ramp_bisect", 2),                                   # v5: the light-coupling ramp
                sources_with_heavy=sources_with_heavy, start_cont=a.light_start_continuation, plain_steps=a.light_start_plain_steps)      # v4
        elif a.mix == "newton" and not info.get("failed"):
            k_it, converged, best, cache, srcs, Nn, vn, Wn, info, N_app, v_app = newton_record(a, chs, st, cache, sources, do_solve, Nn, vn, E_prev, it_hist, T,
                                                                                               chl=chl, sgn_dev=sgn_dev, S_host=S_host)        # v36
        for k_it in (range(1, a.maxit + 1) if (not info.get("failed") and a.mix != "newton") else ()):
            if mixer is not None:
                x = mixer.step(np.concatenate([N_app, v_app]), np.concatenate([Nn, vn]))
                N_app, v_app = x[:a.nsites].copy(), x[a.nsites:].copy()
            elif a.mix == "damped":
                N_app, v_app = N_app + a.mix_alpha * (Nn - N_app), v_app + a.mix_alpha * (vn - v_app)
            else:
                N_app, v_app = Nn.copy(), vn.copy()
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)
            for ch in chs:
                e, U = eig.pop(ch.kappa)
                if a.reference == "R2":
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])
                    cinfo["_mask"] = mask
                    TM["continuation_s"] += time.time() - t_c
                    TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
                else:
                    ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
                cache[ch.kappa] = (None, e, U, ref, cinfo)
            del eig
            light_eval_v36(a, chl, st, cache, N_app, v_app, sgn_dev, S_host)                            # v36
            srcs = sources(cache)
            Nn, vn, Wn, info = do_solve(srcs)
            if info.get("failed"):
                break
            E = FLD.charge(info, a.g)
            dE = abs(E - E_prev) / abs(E)
            E_prev = E
            dN = float(np.max(np.abs(Nn - N_app))); dv = float(np.max(np.abs(vn - v_app)))
            it_hist.append([float(dE), dN, dv])
            resid = max(dN, dv)
            if best is None or resid < best["resid"]:
                best = {"resid": resid, "k_it": k_it, "cache": cache, "srcs": srcs, "sol": (Nn, vn, Wn, info), "app": (N_app.copy(), v_app.copy()), "dE": dE}
            if dE < a.stop_E and dN < a.stop_fields and dv < a.stop_fields:
                converged = True
                break
        if info.get("failed"):
            recs.append({"T": T, "failed": info["failed"], "label": label, "iteration_history": it_hist})
            print("T/rs %.2f: the re-solve is REJECTED (%s); the run stops with the record written" % (T / rs, info["failed"]), flush=True)
            break
        if k_it == 0 or cache[chs[0].kappa][1] is None:
            recs.append({"T": T, "failed": "no eigendecomposition at the accepted fields", "label": label})
            print("T/rs %.2f: no eigendecomposition at the accepted fields; the run stops" % (T / rs), flush=True)
            break
        unconv = None
        if not converged and best is not None:
            cache, srcs = best["cache"], best["srcs"]
            Nn, vn, Wn, info = best["sol"]
            N_app, v_app = best["app"]
            unconv = {"residual": best["resid"], "iteration": best["k_it"], "dE": best["dE"]}
            if unconv_state["first"] is None:
                unconv_state["first"] = T / rs
        best = None
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)
        # the reference's diagnostics on the accepted state; ||[h_m, R2]||_F per channel; the continuation's record
        wmin, npocket, nrefill, dR2R0, nsub_max = 1.0, {}, {}, 0.0, 1
        comm = {}
        cont = {}
        n_inter = n_inv = 0
        mask_neg_equal = True
        masks = {}
        if a.reference == "R2":
            for ch in chs:
                hm_, e, U, ref, cinfo = cache[ch.kappa]
                s_ = st[ch.kappa]
                masks[ch.kappa] = np.asarray(cinfo.pop("_mask"), bool)
                eneg = (e < 0).cpu().numpy()
                same = bool(np.array_equal(masks[ch.kappa], eneg))
                mask_neg_equal = mask_neg_equal and same
                if cinfo.get("min_occupied_weight") is not None:
                    wmin = min(wmin, cinfo["min_occupied_weight"])
                npocket[ch.kappa] = cinfo["n_pocket"]
                nrefill[ch.kappa] = cinfo["nneg"] - s_["n_sea"]
                nsub_max = max(nsub_max, int(cinfo.get("nsub") or 1))
                n_inter += int(cinfo.get("n_intermediate") or 0); n_inv += int(cinfo.get("inversions") or 0)
                ov = float(((U[:, e < 0].conj().T @ ref).abs() ** 2).sum())
                dR2R0 += math.sqrt(max(cinfo["nneg"] + s_["n_sea"] - 2 * ov, 0.0))
                cont[str(ch.kappa)] = {k_: cinfo.get(k_) for k_ in ("mode", "nsub", "m", "m_eff", "verified", "margin_min", "s_T", "tau_LZ", "norm_dH_G", "window_consistency",
                                                                     "n_intermediate", "min_occupied_weight", "max_empty_weight", "inversions", "n_pocket",
                                                                     "pockets_occupied_positive", "holes_empty_negative", "tol_block", "rotated_blocks",
                                                                     "gap_between_blocks", "partly_occupied_blocks")}
                cont[str(ch.kappa)]["mask_equals_negative_set"] = same
                if cinfo.get("certified") is not None:
                    cont[str(ch.kappa)]["certified"] = cinfo["certified"]
                if not cinfo.get("verified") and ch.kappa not in lz_state:
                    lz_state[ch.kappa] = T / rs
                if cinfo.get("min_occupied_weight") is not None and cinfo["min_occupied_weight"] < 0.5 and ch.kappa not in deg_state:
                    deg_state[ch.kappa] = T / rs
            if a.commutator_every > 0 and step % a.commutator_every == 0:
                t_q = time.time()
                for ch in chs:
                    comm[str(ch.kappa)] = commutator_norm(ch, N_app, v_app, cache[ch.kappa][3])
                TM["commutator_s"] += time.time() - t_q
        else:
            for ch in chs:
                masks[ch.kappa] = np.asarray(cache[ch.kappa][4].pop("_mask"), bool)
        # v36: the light references' diagnostics on the accepted state (the masks, the sectors, the overlap with the previous reference)
        masksL, sigL, muL, nplusL, contL = {}, {}, {}, {}, {}
        t_lr = time.time()
        for chL in chl:
            _hm, e, U, ref, ci = cache[chL.key]
            masksL[chL.key] = np.asarray(ci.pop("_mask"), bool)
            sigL[chL.key] = ci.pop("_sigma").cpu().numpy().astype(float)
            muL[chL.key] = ci.get("mu"); nplusL[chL.key] = ci.get("n_plus")
            wts = (st[chL.key]["ref"].conj().T @ U).abs().pow(2).sum(dim=0).cpu().numpy()
            mk = masksL[chL.key]
            ci["min_occupied_weight"] = float(wts[mk].min()) if mk.any() else None
            ci["max_empty_weight"] = float(wts[~mk].max()) if (~mk).any() else None
            c_ = {k_: ci.get(k_) for k_ in ("mode", "route", "nsub", "verified", "fallback", "why", "reanchored", "mu", "gap", "gap_plus_half", "n_plus", "n_occ", "n_sea", "nneg",
                                           "n_pocket", "pockets_occupied_positive", "holes_empty_negative", "inversions_sector", "sigma_impurity_max", "rotated_blocks",
                                           "min_occupied_weight", "max_empty_weight", "margin_min", "norm_dH_G", "cut_check", "cut_crossing")}     # v3: the cut check
            cert = ci.get("certified")
            c_["certified"] = ({k_: v_ for k_, v_ in cert.items() if k_ != "piece_list"} if isinstance(cert, dict) else cert)
            c_["mask_equals_negative_set"] = bool(np.array_equal(mk, (e < 0).cpu().numpy()))
            contL[str(chL.kappa)] = c_
            if ci.get("fallback") and chL.kappa not in light_lz_state:
                light_lz_state[chL.kappa] = T / rs
        TML["readout_s"] += time.time() - t_lr
        if lz_state:
            first = min(lz_state.values())
            cont_label = "LZ convention (c_LZ = %g) from T/rs %.3f (channels %s); inversions %d; n_pocket %d; intermediate weights %d" % (
                a.c_LZ, first, ",".join(str(k_) for k_ in sorted(lz_state)), n_inv, sum(npocket.values()) if npocket else 0, n_inter)
        else:
            cont_label = "verified (Weyl)"
        if deg_state:
            cont_label += "; reference degraded from T/rs %.3f (channels %s)" % (min(deg_state.values()), ",".join(str(k_) for k_ in sorted(deg_state)))
        light_label = ("light reference: certified (mu S walk) on every record" if not light_lz_state else
                       "light reference: the LZ convention (version 3's maximum-overlap continuation, c_LZ = %g) from T/rs %.3f (channels %s)" % (
                           a.c_LZ, min(light_lz_state.values()), ",".join(str(k_) for k_ in sorted(light_lz_state))))  # v3: the fallback named as the lane's convention
        light_certified_now = bool(chl) and all(c_.get("verified") for c_ in contL.values())                                                         # v36
        unconv_label = ("unconverged (residual %.2e at iteration %d) from T/rs %.3f" % (unconv["residual"], unconv["iteration"], unconv_state["first"])) if unconv else (
            ("converged (unconverged records from T/rs %.3f)" % unconv_state["first"]) if unconv_state["first"] is not None else "converged")
        # the indicator, the r_0 check and the collar (engine 18.1, 20.2)
        t_ce = time.time()
        fresh = bool(step % max(1, a.delta_rtol_every) == 0 or drtol_last["value"] is None)
        ind = indicator(eps, J, a.g, R, a.ell, info, N, v, fresh, drtol_last)
        if fresh and ind["delta_rtol"] is not None:
            drtol_last.update({"value": ind["delta_rtol"], "T_over_rs": T / rs})
        r0c = FLD.r0_check(eps, J, a.g, R, a.ell, info, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"]) if fresh else None
        TM["indicator_s"] += time.time() - t_ce
        if not ind["indicator_pass"] and ind_state["first_failure"] is None:
            ind_state["first_failure"] = T / rs
        ind_label = "indicator pass" if ind_state["first_failure"] is None else ("indicator fail from T/rs %.2f" % ind_state["first_failure"])
        gauge = info.get("gauge") or {}
        gauge_label = "intact" if not gauge.get("intervals") else ("continued on %s" % ", ".join("[%.2f, %.2f]" % (x_, y_) for x_, y_ in gauge["intervals"]))
        oracle34 = None
        if getattr(a, "weyl_oracle", False) and a.reference == "R2":
            oracle34 = {}
            t_o = time.time()
            for ch in chs:
                _hm, e_o, U_o, ref_o, _ci = cache[ch.kappa]
                oracle34[str(ch.kappa)] = CERT.oracle_v34(ch, st[ch.kappa]["window"], (N_app, v_app), e_o, U_o, st[ch.kappa]["n_sea"], a.tau, a.c_LZ, eig_at,
                                                          clocked_host_sparse, ref_o, masks[ch.kappa], a.weyl_refine_max)
            TM["oracle34_s"] = time.time() - t_o
        # M1's balance (17.3 (4)): the old configuration's terms before the reference and h are replaced
        cut = int(min(max(round(info["r_collar"]) - 1, 2), a.nsites - 10))
        Y1 = Y2 = IEend = 0.0
        t_m1 = time.time()
        if recs and recs[-1].get("E") is not None:
            for ch in chs:
                s_ = st[ch.kappa]; H_old = clocked_host_sparse(ch, f_ref[0], f_ref[1]); Hr_old = dev_csr(H_old[:2 * cut, :]); ref_old = s_["ref"]; ref_new = cache[ch.kappa][3]
                Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
                Y1 += ch.w * inside_energy_sp(Hr_old, Ccols, fC, 2 * cut)
                Y2 += ch.w * inside_energy_sp(Hr_old, ref_new, np.ones(ref_new.shape[1]), 2 * cut)
                colsE = torch.cat([s_["sea"], s_["dust"], ref_old], dim=1)
                fE = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref_old.shape[1])])
                IEend += ch.w * energy_cut_current_sp(H_old, colsE, fE, cut)
                del Hr_old, colsE
        TM["M1_s"] += time.time() - t_m1
        Y1L = Y2L = IEendL = 0.0                                                                       # v36: the light field's old-configuration terms
        t_m1 = time.time()
        if chl and recs and recs[-1].get("E") is not None:
            for chL in chl:
                s_ = st[chL.key]; H_old = clocked_host_sparse(chL, f_ref[0], f_ref[1]); Hr_old = dev_csr(H_old[:2 * cut, :]); ref_old = s_["ref"]; ref_new = cache[chL.key][3]
                Y1L += chL.w * inside_energy_sp(Hr_old, s_["sea"], np.ones(s_["sea"].shape[1]), 2 * cut)
                Y2L += chL.w * inside_energy_sp(Hr_old, ref_new, np.ones(ref_new.shape[1]), 2 * cut)
                colsE = torch.cat([s_["sea"], ref_old], dim=1); fE = np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref_old.shape[1])])
                IEendL += chL.w * energy_cut_current_sp(H_old, colsE, fE, cut)
                del Hr_old, colsE
        TML["M1_s"] += time.time() - t_m1
        for ch in chs:
            ci_ = cache[ch.kappa][4]
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks"),
                                           "weyl_gap": ci_.get("weyl_gap", ci_.get("gap_between_blocks")), "mask": masks.get(ch.kappa)})
            st[ch.kappa]["window"] = st[ch.kappa]["window"][-1:]
        for chL in chl:                                                                                 # v36: the light references and windows
            ci_ = cache[chL.key][4]
            st[chL.key]["ref"] = cache[chL.key][3]
            st[chL.key]["window"] = [{"ref": cache[chL.key][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap"), "weyl_gap": ci_.get("weyl_gap", ci_.get("gap")),
                                      "mask": masksL.get(chL.key), "mu": ci_.get("mu"), "n_plus": ci_.get("n_plus")}]
        f_ref = (N_app.copy(), v_app.copy())
        tz = FLD.trapping_radius(N, v, R, info)
        Rt = tz["zeros"][-1] if tz["zeros"] else None
        r_in = tz["zeros"][0] if len(tz["zeros"]) >= 2 else None
        jets = surface_jets(info, Rt, a.g, E)
        kT = jets["kappa_T"] if jets else None
        M_wg = FLD.mass_function(info, a.g, Rt) if Rt else None
        M_ms = FLD.misner_sharp(info, a.g, Rt) if Rt else None
        Rc = Rsh = None
        tot = eps_dust.sum()
        if tot > 0:
            cum = np.cumsum(eps_dust); Rsh = float(R[min(int(np.searchsorted(cum, 0.99 * tot)), a.nsites - 1)]); Rc = float(eps_dust @ R / tot)
        late = None
        if Rt:
            rr = np.linspace(max(Rt - 5 * a.ell, 0.0), Rt + 5 * a.ell, 101)
            rho_l = np.abs(info["rho_ell"](rr)); J_l = np.abs(info["J_ell"](rr))
            rmax = max(float(np.max(np.abs(info["rho_ell"](np.linspace(0, info["r_collar"], 2001))))), 1e-300)
            jmax = max(float(np.max(np.abs(info["J_ell"](np.linspace(0, info["r_collar"], 2001))))), 1e-300)
            late = {"rho_ratio": float(rho_l.max() / rmax), "J_ratio": float(J_l.max() / jmax)}
            late["flag"] = bool(late["rho_ratio"] < 1e-6 and late["J_ratio"] < 1e-6)
        IE = Hin = 0.0
        X1 = X2 = 0.0
        t_m1 = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]
            H_new = clocked_host_sparse(ch, N_app, v_app); Hr = dev_csr(H_new[:2 * cut, :])
            s_ = st[ch.kappa]
            cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
            fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
            IE += ch.w * energy_cut_current_sp(H_new, cols, fs, cut)
            Hin += ch.w * inside_energy_sp(Hr, cols, fs, 2 * cut)
            Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
            X1 += ch.w * inside_energy_sp(Hr, Ccols, fC, 2 * cut)
            X2 += ch.w * inside_energy_sp(Hr, ref, np.ones(ref.shape[1]), 2 * cut)
            del Hr, cols
        TM["M1_s"] += time.time() - t_m1
        # v36: the light field at the accepted fields: E_inv and the occupation per level (the propagation's coefficients), the collar cut current,
        # the inside energies, the sources (fed and guard), the weight and energy beyond the collars, the Killing-bin deviations
        t_lr = time.time()
        coefL, dsL, JsL, EinvL, chanL = {}, {}, {}, {}, {}
        IE_L_src = IE_L_all = Hin_L = X1L = X2L = 0.0
        Einv_src = Einv_all = 0.0
        eps_light_fed = np.zeros(a.nsites); J_light_fed = np.zeros(a.nsites); eps_light_all = np.zeros(a.nsites)
        EinvH = None
        for chL in chl:
            _hm, e, U, ref, ci = cache[chL.key]; s_ = st[chL.key]
            mk = masksL[chL.key]
            A = U.conj().T @ s_["sea"]
            nocc = (A.abs() ** 2).sum(dim=1)
            e64 = e.to(torch.float64); e_np = e64.cpu().numpy()
            E_sea = float((e64 * nocc).sum()); E_ref = float(e_np[mk].sum()); E_inv = E_sea - E_ref
            coefL[chL.key] = (A, e, U)
            ds = base[chL.key][0] - dens(chL.h, ref); Js = base[chL.key][1] - dens(chL.Q, ref)
            dsL[chL.key] = ds; JsL[chL.key] = Js
            H_new = clocked_host_sparse(chL, N_app, v_app); Hr = dev_csr(H_new[:2 * cut, :])
            cols = torch.cat([s_["sea"], ref], dim=1); fs = np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])])
            IEk = chL.w * energy_cut_current_sp(H_new, cols, fs, cut)
            x1 = chL.w * inside_energy_sp(Hr, s_["sea"], np.ones(s_["sea"].shape[1]), 2 * cut); x2 = chL.w * inside_energy_sp(Hr, ref, np.ones(ref.shape[1]), 2 * cut)
            Hin_L += x1 - x2; X1L += x1; X2L += x2; IE_L_all += IEk
            nocc_np = nocc.cpu().numpy(); devn = nocc_np - mk.astype(float)
            bins = {"(-inf,-1)": float(devn[e_np < -1.0].sum()), "(-1,0)": float(devn[(e_np >= -1.0) & (e_np < 0.0)].sum()), "(0,1)": float(devn[(e_np >= 0.0) & (e_np < 1.0)].sum()),
                    "(1,inf)": float(devn[e_np >= 1.0].sum())}
            beyond = {}
            for Rc_rs in collars_rs:
                i0 = int(np.searchsorted(np.asarray(R, float), Rc_rs * rs, side="right")); sel = np.asarray(R, float) > Rc_rs * rs
                beyond["%g" % Rc_rs] = {"number": float((s_["sea"][2 * i0:, :].abs() ** 2).sum() - (ref[2 * i0:, :].abs() ** 2).sum()),
                                        "energy_density_route": float(np.sum(N_app[sel] * ds[sel] + v_app[sel] * Js[sel])),
                                        "weighted_energy_density_route": chL.w * float(np.sum(N_app[sel] * ds[sel] + v_app[sel] * Js[sel]))}
            c_ = contL[str(chL.kappa)]
            c_.update({"sourcing": chL.sourcing, "w": chL.w, "E_inv": E_inv, "E_sea": E_sea, "E_ref": E_ref, "E_inv_weighted": chL.w * E_inv, "I_E_cut_weighted": IEk,
                       "H_in_weighted": x1 - x2, "eps_sum": float(ds.sum()), "J_sum": float(Js.sum()), "eps_sum_weighted": chL.w * float(ds.sum()),
                       "nocc_deviation_bins": bins, "beyond": beyond, "min_abs_e": float(np.min(np.abs(e_np))), "threshold_radius": threshold_radii(N, v, R, 0.0, [chL.kappa])[str(chL.kappa)]})
            chanL[chL.key] = c_
            Einv_all += chL.w * E_inv; eps_light_all += chL.w * ds
            spectra["nocc_L%d" % chL.kappa].append(nocc_np.astype(np.float64))
            if chL.sourcing:
                Einv_src += chL.w * E_inv; IE_L_src += IEk; eps_light_fed += chL.w * ds; J_light_fed += chL.w * Js
            del Hr, cols
        if chl and a.c_heavy_einv:                                                                      # v36: the heavy field's E_inv (the per-field energy identity)
            EinvH = 0.0
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]; s_ = st[ch.kappa]
                e64 = e.to(torch.float64)
                nocc_s = ((U.conj().T @ s_["sea"]).abs() ** 2).sum(dim=1)
                E_sea = float((e64 * nocc_s).sum())
                if s_["dust"].shape[1]:
                    nocc_d = ((U.conj().T @ s_["dust"]).abs() ** 2 * T_(s_["f"])[None, :].real.to(torch.float64)).sum(dim=1)
                    E_sea += float((e64 * nocc_d).sum())
                E_ref = float(e64[torch.as_tensor(masks[ch.kappa], device=U.device)].sum())
                EinvH += ch.w * (E_sea - E_ref)
        TML["readout_s"] += time.time() - t_lr
        IE_mid = None
        if step in simpson_steps and step < nsteps and not a.benchmark:
            t_sm = time.time()
            IE_mid = 0.0
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]
                ph2 = torch.exp(-1j * e.to(torch.complex128) * (0.5 * a.tau))
                s_ = st[ch.kappa]
                halfp = lambda X: U @ (ph2[:, None] * (U.conj().T @ X))
                cols = torch.cat([halfp(s_["sea"]), halfp(s_["dust"]) if s_["dust"].shape[1] else s_["dust"], ref], dim=1)
                fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                IE_mid += ch.w * energy_cut_current_sp(clocked_host_sparse(ch, N_app, v_app), cols, fs, cut)
                del cols
            TM["simpson_s"] += time.time() - t_sm
        # the handoff quantities (20.1): the stresses (dust and pairs apart), <N^2>_n, the sea's and R2's shell energies
        t_h = time.time()
        Pd = None; Pp = None
        eps_sea = np.zeros(a.nsites); eps_R2 = np.zeros(a.nsites)
        for ch in chs:
            s_ = st[ch.kappa]; ref = cache[ch.kappa][3]; cs = cstr[ch.kappa]
            pd_ = STR.stress_parts(cs, s_["dust"], s_["f"])
            pp_ = STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])]))
            for P_, p_ in ((0, pd_), (1, pp_)):
                p_ = dict(p_); p_["sy_k"] = ch.kappa * p_["sy"]
                if P_ == 0:
                    Pd = {k_: ch.w * np.asarray(v_, float) for k_, v_ in p_.items()} if Pd is None else {k_: Pd[k_] + ch.w * np.asarray(p_[k_], float) for k_ in Pd}
                else:
                    Pp = {k_: ch.w * np.asarray(v_, float) for k_, v_ in p_.items()} if Pp is None else {k_: Pp[k_] + ch.w * np.asarray(p_[k_], float) for k_ in Pp}
            eps_sea += ch.w * base[ch.kappa][0]; eps_R2 += ch.w * dens(ch.h, ref)
        SVd = STR.shell_values(Pd, R, a.m); SVp = STR.shell_values(Pp, R, a.m)
        Sr_tot = SVd["S_r"] + SVp["S_r"]; St_tot = SVd["S_t"] + SVp["S_t"]
        Sr_l, St_l, tr_err = STR.smooth_stress(Sr_tot, St_tot, R, a.ell, r_nodes_fixed)
        N2avg, _vz, _nn = FLD.site_fields_kernel_average({"N_dense": (lambda r_: info["N_dense"](r_) ** 2), "v_dense": zero}, R, a.ell)
        vir = {}
        for lab_, SV in (("dust", SVd), ("pairs", SVp)):
            sc = max(float(np.max(np.abs(SV["virial_rhs"]))), 1e-300)
            vir[lab_] = float(np.max(np.abs(SV["virial_lhs"] - SV["virial_rhs"]))) / sc
        stress_rec = {"identity_max_abs": {"dust": SVd["identity_max_abs"], "pairs": SVp["identity_max_abs"]}, "virial_rel": vir, "trace_identity_rel": tr_err,
                      "F_total_at_last_face": float(SVd["F_face"][-1] + SVp["F_face"][-1]), "seconds": time.time() - t_h}
        hand["T"].append(T); hand["N2avg"].append(np.asarray(N2avg, float)); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(np.asarray(eps_dust, float))
        hand["S_r_ell"].append(Sr_l); hand["S_t_ell"].append(St_l)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
            hand["b_%s" % part].append(np.asarray(P_["b"], float)); hand["sy_k_%s" % part].append(np.asarray(P_["sy_k"], float))
        stressL = None
        if chl:                                                                                         # v36: the light field's stress parts (sea minus its reference)
            t_hl = time.time()
            PL = None; eps_light_sea = np.zeros(a.nsites); eps_light_R2 = np.zeros(a.nsites)
            for chL in chl:
                s_ = st[chL.key]; ref = cache[chL.key][3]
                pl_ = dict(STR.stress_parts(cstrL[chL.key], torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])])))
                pl_["sy_k"] = chL.kappa * pl_["sy"]
                PL = {k_: chL.w * np.asarray(v_, float) for k_, v_ in pl_.items()} if PL is None else {k_: PL[k_] + chL.w * np.asarray(pl_[k_], float) for k_ in PL}
                eps_light_sea += chL.w * base[chL.key][0]; eps_light_R2 += chL.w * dens(chL.h, ref)
            SVl = STR.shell_values(PL, R, 0.0)
            hand["F_face_light"].append(SVl["F_face"]); hand["fc_light"].append(np.asarray(PL["fc"], float)); hand["b_light"].append(np.asarray(PL["b"], float))
            hand["sy_k_light"].append(np.asarray(PL["sy_k"], float)); hand["eps_light_sea"].append(eps_light_sea); hand["eps_light_R2"].append(eps_light_R2)
            scl = max(float(np.max(np.abs(SVl["virial_rhs"]))), 1e-300)
            stressL = {"identity_max_abs": SVl["identity_max_abs"], "virial_rel": float(np.max(np.abs(SVl["virial_lhs"] - SVl["virial_rhs"]))) / scl, "F_at_last_face": float(SVl["F_face"][-1])}
            TML["handoff_s"] += time.time() - t_hl
        spectra["T"].append(T)
        for ch in chs:
            spectra["mask_%d" % ch.kappa].append(np.packbits(masks[ch.kappa]))
            spectra["e_%d" % ch.kappa].append(cache[ch.kappa][1].cpu().numpy().astype(np.float64))
        for chL in chl:                                                                                 # v36
            spectra["mask_L%d" % chL.kappa].append(np.packbits(masksL[chL.key]))
            spectra["e_L%d" % chL.kappa].append(cache[chL.key][1].cpu().numpy().astype(np.float64))
        TM["handoff_s"] = time.time() - t_h
        saved = None
        tslice = [x for x in sl128 + sl64 if abs(T / rs - x) < 1e-6]
        if tslice and a.slice_dir:
            dt_ = np.complex128 if any(abs(T / rs - x) < 1e-6 for x in sl128) else np.complex64
            arrs = {"T": T, "rs": rs, "N_applied": N_app, "v_applied": v_app, "R": R, "kappas": np.array([ch.kappa for ch in chs]), "weights": np.array([ch.w for ch in chs]),
                    "dtype": str(np.dtype(dt_))}
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]; s_ = st[ch.kappa]
                arrs["sea_%d" % ch.kappa] = s_["sea"].cpu().numpy().astype(dt_)
                arrs["dust_%d" % ch.kappa] = s_["dust"].cpu().numpy().astype(dt_)
                arrs["f_%d" % ch.kappa] = np.asarray(s_["f"], float)
                arrs["R2_%d" % ch.kappa] = ref.cpu().numpy().astype(dt_)
                arrs["e_%d" % ch.kappa] = e.cpu().numpy()
                arrs["mask_%d" % ch.kappa] = np.packbits(masks[ch.kappa])
            if chl:                                                                                     # v36: the light state in the same slice
                arrs.update(light_slice_arrays_v36(chl, st, cache, masksL, sigL, muL, nplusL, dt_))
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(saved, **arrs)
            del arrs
        savedL = None
        if chl and a.slice_dir and any(abs(T / rs - x) < 1e-6 for x in slL) and not tslice:                # v36: a light-only complex128 slice
            arrs = {"T": T, "rs": rs, "N_applied": N_app, "v_applied": v_app, "R": R, "dtype": "complex128", "light_only": True}
            arrs.update(light_slice_arrays_v36(chl, st, cache, masksL, sigL, muL, nplusL, np.complex128))
            savedL = os.path.join(a.slice_dir, "light_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(savedL, **arrs)
            del arrs
        s11 = None
        if a.massless_ball:
            t_11 = time.time()
            s11, s11_arr = s11_record(a, info, R, rs, eps, J, eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T)
            s11["seconds"] = time.time() - t_11
            s11_store.setdefault("T", []).append(T)
            for k_, v_ in s11_arr.items():
                s11_store.setdefault(k_, []).append(v_)
        elif chl and a.c_readouts:                                                                      # v36: the mass function's sign per record, the 3 cases (P2's stop rule)
            t_11 = time.time()
            s11, s11_arr = s11_record(a, info, R, rs, eps, J - J_light_fed, eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T)
            rc_ = float(info["r_collar"]); rf_ = np.asarray(info["rfine"], float); rf_ = rf_[(rf_ > 0) & (rf_ <= rc_)]
            nodes_ = np.asarray(info["r_nodes"], float)
            smL, errL = s11_smoothed(FLD, eps_light_fed, J_light_fed, R, a.ell, info, [rf_, nodes_])
            (rho_L, J_L), (rho_Ln, J_Ln) = smL
            Nf_ = np.asarray(info["N_dense"](rf_), float); vf_ = np.asarray(info["v_dense"](rf_), float)
            w_ = np.where(np.abs(Nf_) > 0, vf_ / Nf_, 0.0)
            out_cell = rf_ > a.ell
            s11["light_source_min_outside_cell"] = (float((rho_L + w_ * J_L)[out_cell].min()) if (errL is None and out_cell.any()) else None)
            s11["light_source_integral"] = float(np.sum(eps_light_fed)); s11["light_smoothing_error"] = errL
            s11["light_fed"] = bool(light_on); s11["light_certified_on_record"] = light_certified_now
            if s11.get("stop"):
                r_neg = s11["stop"]["r_first_negative_M"]; inner = (rf_ > a.ell) & (rf_ <= r_neg)
                s11["stop"]["light_source_min_inside_r_neg"] = (float((rho_L + w_ * J_L)[inner].min()) if (errL is None and inner.any()) else None)
                if s11["stop"]["case"].startswith("(ii)") and not light_certified_now:
                    s11["stop"]["case"] = ("(iii) or (ii): the averaged source is negative inside r_neg and the light field's reference is not certified on this record (%s): "
                                           "a reference error cannot be excluded" % light_label)
                    s11["stop"]["reference_certified"] = False
                s11["stop"]["heavy_reference_certified"] = (cont_label == "verified (Weyl)"); s11["stop"]["light_reference_certified"] = light_certified_now
            s11["v36"] = "Computation C: the heavy dust in the role of the ball; the light fed sources smoothed beside; the stop under --s11-stop when the coupling is on"
            s11["seconds"] = time.time() - t_11
            s11_arr.update({"rho_ell_light_fed": rho_Ln, "J_ell_light_fed": J_Ln})
            s11_store.setdefault("T", []).append(T)
            for k_, v_ in s11_arr.items():
                s11_store.setdefault(k_, []).append(v_)
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "E_dust_shells": float(eps_dust.sum()), "E_pairs_shells": float(eps_pairs.sum()),
               "pairs_inside_Rt": (float(eps_pairs[R < Rt].sum()) if Rt else None), "pairs_outside_Rt": (float(eps_pairs[R >= Rt].sum()) if Rt else float(eps_pairs.sum())),
               "R_t": Rt, "R_t_over_2gE": (Rt / (2 * a.g * E) if Rt else None), "W_over_2g_at_Rt": M_wg, "misner_sharp_at_Rt": M_ms,
               "R_t_over_2g_MS": (Rt / (2 * a.g * M_ms) if (Rt and M_ms) else None), "kappa_T": kT, "four_gE_kappa_T": (4.0 * a.g * E * kT if kT is not None else None),
               "surface_jets": jets, "r_in": r_in, "trapping_zeros": tz["zeros"], "tangencies": tz["tangencies"], "min_N_minus_absv": tz["min"], "r_min_N_minus_absv": tz["r_min"],
               "late_surface": late, "N_min": float(N.min()), "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(info["v_dense"](Rc)[0]) if Rc else None),
               "threshold_radius": threshold_radii(N, v, R, a.m, [1, 2, a.kappa_sea]), "n_pocket": npocket, "n_refill_R0": nrefill, "R2_minus_R0_F": dR2R0,
               "min_occupied_weight": wmin, "nsub_max": nsub_max, "M1_I_E": IE, "H_in": Hin,
               "iterations": k_it + 1, "eigendecompositions": TM["eigh_n"], "converged": converged, "unconverged": unconv, "iteration_history": it_hist,
               "continuation": cont, "continuation_label": cont_label, "mask_equals_negative_set": mask_neg_equal, "inversions_total": n_inv, "intermediate_weights_total": n_inter,
               "commutator_h_R2_F": comm, "gauge": {k_: gauge.get(k_) for k_ in ("label", "intervals", "min_U_over_U_ref", "delta_L_bound", "delta_L_bound_two_resolutions", "lapse_unresolved", "segments", "zeros", "stretches", "near_tangencies")},
               "regime": info.get("regime"), "r0_check": r0c, "stress": stress_rec, "slice_saved": saved,
               "solver": {"module": FLD.__name__ + ".solve_constraints_ivp", "nfev": (info.get("ivp") or {}).get("nfev"), "N_centre": (info.get("ivp") or {}).get("N_centre"),
                          "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "residual_matching_region": info.get("residual_matching_region"),
                          "residual_ladder": info.get("residual_ladder"), "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"),
                          "centre_series_check": info.get("centre_series_check"), "collar_margin": info.get("collar_margin"), "W_tail": info.get("W_tail"),
                          "W_roundoff_negative_points": info.get("W_roundoff_negative_points"), "W_negative": info.get("W_negative"), "site_fields": a.site_fields, "kernel_norms": info.get("kernel_norms")},
               "spectrum": info["spectrum"], "r_collar": info["r_collar"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
               "label": label, "seconds": time.time() - tr, "indicator": ind,
               "record_label": (FLD.__name__ + " (rtol %g, atol %g); v31 (v30 with the Newton-Krylov solve as an option; mixer %s%s; ops v3);" % (SOLVE_TOL["rtol"], SOLVE_TOL["atol"], a.mix, (" a = %g" % a.mix_alpha) if a.mix == "damped" else "") if not v32 else
                                FLD.__name__ + " (rtol %g, atol %g); v32 (v31 with %s; mixer %s; ops v3);" % (SOLVE_TOL["rtol"], SOLVE_TOL["atol"], v32_desc(a), a.mix)) + " kernel: the Gaussian of standard deviation ell, sigma^2 = ell^2 (plus a^2/12 where the cell average enters), evolved at ell %d, prepared at ell %d; tau = %g; continuation: %s; stresses: the lane's declared stress (the Noether flux of Q at unit clock and zero flow); rows: flat-slice ADM, K^ADM = -K, raw, floor in the stress's trace (25.3); J_ell = <P_r> = -J_r of the goal's display; gauge: %s; %s; %s; collar: %s" % (
                   a.ell, (a.prep_ell if a.prep_ell is not None else a.ell), a.tau, cont_label.replace("LZ convention (c_LZ = %g)" % a.c_LZ, "the LZ-inspired maximum-overlap convention (c_LZ = %g, the lane's declaration)" % a.c_LZ), gauge_label, unconv_label, ind_label, ind["collar_label"]),
               "timing": dict(TM),
               "M1_parts": {"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2, "I_E_end_prev_interval": IEend, "I_E_start": IE, "cut": cut, "I_E_mid": IE_mid}}
        if resume is not None and not any(q.get("engine") in ("engine_collapse_v19", "engine_collapse_v24", "engine_collapse_v25") for q in recs):
            rec["resumed_from"] = {"run": resume["from"], "T_over_rs": resume["T_over_rs"]}
        rec["engine"] = "engine_collapse_v25"
        if solve32 is not None:
            rec["solve_v32"] = solve32
        if s11 is not None:
            rec["s11"] = s11
        if v33_on(a):
            rec["engine_v33"] = v33_desc(a)
        if v34_on(a):
            rec["engine_v34"] = v34_desc(a)
        if getattr(a, "nk_jvp", None) == "exact":
            rec["jvp_v35"] = {k_: v_ for k_, v_ in JV.items() if not k_.startswith("_")}
            rec["engine_v35"] = "the Newton direction by the exact J.v (--nk-jvp exact; engine spec 40.4 (2) (b))"
            for k_ in JV:
                JV[k_] = None if k_ == "ref_mismatch" else (0.0 if k_ == "seconds" else 0)
        if V35.V36["fixes"] or V35.V36["kcache"]:                                                       # v2: version 36's record entry
            rec["v36"] = {"fixes": V35.V36["fixes"], "gap_none_substeps": V35.V36["gap_none_substeps"], "kernel_cache": dict(V35.V36["kcache"]) if V35.V36["kcache"] else None}
        if chl:                                                                                         # v36: the light field's record entry
            recL = {"coupling": a.light_coupling, "fed": light_on, "transport": a.light_transport, "label": light_label, "certified_on_record": light_certified_now,
                    "channels": {str(chL.kappa): chanL[chL.key] for chL in chl},
                    "E_inv_light": Einv_all, "E_inv_light_sourcing": Einv_src, "E_inv_light_over_E": Einv_all / E, "E_inv_light_sourcing_over_E": Einv_src / E, "E_inv_heavy": EinvH,
                    "I_E_light": IE_L_all, "I_E_light_sourcing": IE_L_src, "H_in_light": Hin_L, "eps_fed_sum": float(eps_light_fed.sum()), "J_fed_sum": float(J_light_fed.sum()),
                    "eps_light_all_sum": float(eps_light_all.sum()),
                    "truncation_guard": {"E_inv_guard_over_sourcing": ((Einv_all - Einv_src) / Einv_src) if Einv_src != 0.0 else None,
                                         "I_E_guard_over_sourcing": ((IE_L_all - IE_L_src) / IE_L_src) if IE_L_src != 0.0 else None,
                                         "eps_guard_over_sourcing": ((float(eps_light_all.sum()) - float(eps_light_fed.sum())) / float(eps_light_fed.sum())) if float(eps_light_fed.sum()) != 0.0 else None},
                    "threshold_radius_massless": threshold_radii(N, v, R, 0.0, [chL.kappa for chL in chl]), "stress": stressL, "light_slice_saved": savedL,
                    "M1_parts": {"X1": X1L, "X2": X2L, "Y1": Y1L, "Y2": Y2L, "I_E_end_prev_interval": IEendL, "I_E_start": IE_L_all, "cut": cut},
                    "timing": dict(TML), "engine_v36": v36_desc(a)}
            prevL = (recs[-1].get("light_v36") or {}).get("M1_parts") if (recs and recs[-1].get("E") is not None) else None
            if prevL is not None:
                dT = T - recs[-1]["T"]
                a_ex = Y1L - prevL["X1"]; b_w = (X1L - X2L) - (Y1L - Y2L); c_r = -(Y2L - prevL["X2"]); dH_ = (X1L - X2L) - (prevL["X1"] - prevL["X2"])
                recL["M1_balance"] = {"dHin": dH_, "evolution_exact": a_ex, "evolution_trapezoid": 0.5 * dT * (prevL["I_E_start"] + IEendL), "field_work": b_w, "reference_reset": c_r,
                                      "closure": a_ex + b_w + c_r - dH_, "per_unit_time": {"dHin": dH_ / dT, "evolution": a_ex / dT, "field_work": b_w / dT, "reference_reset": c_r / dT}}
                prevE = (recs[-1].get("light_v36") or {})
                if prevE.get("E_inv_light") is not None:
                    recL["dE_inv_light_dT"] = (Einv_all - prevE["E_inv_light"]) / dT
            rec["light_v36"] = recL
        if oracle34 is not None:
            rec["certificate_oracle"] = oracle34
        prevM = recs[-1].get("M1_parts") if (recs and recs[-1].get("E") is not None) else None
        if prevM is not None:
            dT = T - recs[-1]["T"]
            a_exact = Y1 - prevM["X1"]; b_work = (X1 - X2) - (Y1 - Y2); c_reset = -(Y2 - prevM["X2"]); dH = (X1 - X2) - (prevM["X1"] - prevM["X2"])
            rec["M1_balance"] = {"dHin": dH, "evolution_exact": a_exact, "evolution_trapezoid": 0.5 * dT * (prevM["I_E_start"] + IEend), "field_work": b_work,
                                 "reference_reset": c_reset, "closure": a_exact + b_work + c_reset - dH, "cut_changed": bool(prevM.get("cut") != cut),
                                 "per_unit_time": {"dHin": dH / dT, "evolution": a_exact / dT, "field_work": b_work / dT, "reference_reset": c_reset / dT}}
            if prevM.get("I_E_mid") is not None:
                rec["M1_balance"]["evolution_simpson"] = dT / 6.0 * (prevM["I_E_start"] + 4.0 * prevM["I_E_mid"] + IEend)
                rec["M1_balance"]["trapezoid_minus_exact"] = rec["M1_balance"]["evolution_trapezoid"] - a_exact
                rec["M1_balance"]["simpson_minus_exact"] = rec["M1_balance"]["evolution_simpson"] - a_exact
        if recs and recs[-1].get("E") is not None:
            dT = T - recs[-1]["T"]
            rec["dE_dT"] = (E - recs[-1]["E"]) / dT
            rec["dHin_dT"] = (Hin - recs[-1]["H_in"]) / dT
            rec["balance_d_E_minus_Hin_dT"] = rec["dE_dT"] - rec["dHin_dT"]
            if Rc is not None and recs[-1].get("R_c") is not None:
                rec["dRc_dT"] = (Rc - recs[-1]["R_c"]) / dT
        if step % 10 == 0 and prevW is not None and T > prevW[1]:
            rec["M9"] = {}
            for lab, rr_ in (("0.5 R_sh", 0.5 * Rsh if Rsh else None), ("R_t", Rt), ("collar", float(info["r_collar"]) - 1.0)):
                if rr_:
                    ci = int(min(max(round(rr_), 2), a.nsites - 10))
                    I_c = 0.0
                    for ch in chs:
                        _hm, e, U, ref, _ci = cache[ch.kappa]
                        H_new = clocked_host_sparse(ch, N_app, v_app)
                        s_ = st[ch.kappa]
                        cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
                        fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                        I_c += ch.w * energy_cut_current_sp(H_new, cols, fs, ci)
                        del cols
                    rec["M9"][lab] = {"r": rr_, "cut_site": ci, "dW_over_2g_dT": float((info["W_field"](rr_)[0] - prevW[0](rr_)[0]) / (2 * a.g * (T - prevW[1]))),
                                      "I_E_cut": I_c}
        if step % 10 == 0:
            prevW = (info["W_field"], T)
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy()); fields["eps_tot"].append(eps.copy()); fields["J_tot"].append(J.copy())
        fields["eps_pairs"].append(eps_pairs.copy()); fields["J_pairs"].append(J_pairs.copy())
        fields["N_applied"].append(N_app.copy()); fields["v_applied"].append(v_app.copy())
        fields["r_nodes"].append(info["r_nodes"]); fields["N_nodes"].append(info["N_nodes"]); fields["v_nodes"].append(info["v_nodes"])
        if chl:                                                                                         # v36: the light arrays of the fields npz
            fieldsL["eps_light_fed"].append(eps_light_fed.copy()); fieldsL["J_light_fed"].append(J_light_fed.copy())
            for chL in chl:
                fieldsL["eps_L%d" % chL.kappa].append(np.asarray(dsL[chL.key], float).copy()); fieldsL["J_L%d" % chL.kappa].append(np.asarray(JsL[chL.key], float).copy())
        oe = jets["over_exact"] if jets else None
        mb = rec.get("M1_balance")
        pc = ind.get("collar_padded") or {}
        last_h = it_hist[-1] if it_hist else [None, float("nan"), float("nan")]
        cmin = [c_.get("min_occupied_weight") for c_ in cont.values() if c_.get("min_occupied_weight") is not None]
        cmax = [c_.get("max_empty_weight") for c_ in cont.values() if c_.get("max_empty_weight") is not None]
        tlz = [c_.get("tau_LZ") for c_ in cont.values() if c_.get("tau_LZ") is not None]
        _line = ("T/rs %.3f E/E0 %.7f dust %.4f pairs %.3e R_t %s (2gE %.3f; d1..d4 of N+v over exact %s; r_in %s) min(N-|v|) %.4f R_sh %s R_c %s pockets %s inversions %d intermediate %d mask=neg %s nsub max %d it %d (eigh %d)%s M1 %.3e dE/dT %s res %.1e %.1e; B_estimate %s; r0 check dE %s; gauge %s; regime max %.2e; continuation [%s]; weights min occ %s max empty %s; tau_LZ min %s; [h,R2] max %s; stress identity %.1e %.1e virial %.1e %.1e trace %.1e; M1 balance: evolution %s work %s reset %s closure %s [%s] last iteration dE %s dN %.1e dv %.1e (%.0f s; eigh %d in %.1f s, continuation %.1f s, densities %.1f s, solves %d in %.1f s, indicator %.1f s, handoff %.1f s, M1 %.1f s, propagation %.1f s)" % (
            T / rs, E / E0, eps_dust.sum(), eps_pairs.sum(), ("%.3f" % Rt) if Rt else None, 2 * a.g * E, (", ".join("%.4f" % q for q in oe)) if oe else None,
            ("%.3f" % r_in) if r_in else None, tz["min"], Rsh, ("%.2f" % Rc) if Rc else None, sum(npocket.values()) if npocket else 0, n_inv, n_inter, mask_neg_equal, nsub_max,
            k_it + 1, TM["eigh_n"], ("" if converged else " UNCONVERGED (residual %.1e)" % (unconv["residual"] if unconv else float("nan"))), IE,
            ("%.3e" % rec["dE_dT"]) if "dE_dT" in rec else "-", info.get("residual_W") or 0.0, info.get("residual_invN") or 0.0,
            ("%.1e" % ind["B_estimate"]) if ind["B_estimate"] is not None else "none", ("%.1e" % r0c["dE_rel"]) if (r0c and "dE_rel" in r0c) else "-",
            gauge_label, (info.get("regime") or {}).get("max_ratio", float("nan")), cont_label, ("%.3e" % min(cmin)) if cmin else "-", ("%.3e" % max(cmax)) if cmax else "-",
            ("%.2f" % min(tlz)) if tlz else "-", ("%.1e" % max(comm.values())) if comm else "-",
            SVd["identity_max_abs"], SVp["identity_max_abs"], vir["dust"], vir["pairs"], tr_err,
            ("%.3e" % mb["evolution_exact"]) if mb else "-", ("%.3e" % mb["field_work"]) if mb else "-", ("%.3e" % mb["reference_reset"]) if mb else "-",
            ("%.1e" % mb["closure"]) if mb else "-", rec["record_label"],
            ("%.1e" % last_h[0]) if last_h[0] is not None else "-", last_h[1], last_h[2], time.time() - tr, TM["eigh_n"], TM["eigh_s"], TM["continuation_s"],
            TM["densities_s"], TM["solve_n"], TM["solve_s"], TM["indicator_s"], TM.get("handoff_s", 0.0), TM["M1_s"], TM["propagation_prev_s"]))
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if chl:                                                                                         # v36: the light line
            recL = rec["light_v36"]; mbl = recL.get("M1_balance")
            print("  light v36 T/rs %.3f [%s; coupling %s]: E_inv/E all %.4e sourcing %.4e (heavy E_inv %s); I_E light %.3e (sourcing %.3e) heavy %.3e; fed eps %.3e J %.3e; guard ratios E_inv %s I_E %s eps %s; M1 light closure %s; per channel %s (light eigh %d in %.1f s, continuation %.1f s, walks %d pieces %d Cholesky %d fallbacks, readouts %.1f s)" % (
                T / rs, light_label, a.light_coupling, recL["E_inv_light_over_E"], recL["E_inv_light_sourcing_over_E"], fmt(recL["E_inv_heavy"], "%.4e"), recL["I_E_light"], recL["I_E_light_sourcing"], IE,
                recL["eps_fed_sum"], recL["J_fed_sum"], fmt(recL["truncation_guard"]["E_inv_guard_over_sourcing"], "%.3e"), fmt(recL["truncation_guard"]["I_E_guard_over_sourcing"], "%.3e"),
                fmt(recL["truncation_guard"]["eps_guard_over_sourcing"], "%.3e"), fmt(mbl["closure"], "%.1e") if mbl else "-",
                "; ".join("kappa %s: %s mu %s gap %s pieces %s d_min %s pockets %d inv %d wmin %s E_inv %.3e" % (
                    k_, ("certified" if q.get("verified") else "FALLBACK (%s)" % q.get("why")), fmt(q.get("mu"), "%.3e"), fmt(q.get("gap"), "%.2e"), q.get("nsub"),
                    fmt((q.get("certified") or {}).get("d_min_certified") if isinstance(q.get("certified"), dict) else None, "%.2e"), q.get("n_pocket") or 0, q.get("inversions_sector") or 0,
                    fmt(q.get("min_occupied_weight"), "%.4f"), q.get("E_inv")) for k_, q in recL["channels"].items()),
                TML["eigh_n"], TML["eigh_s"], TML["continuation_s"], TML["walk_pieces"], TML["walk_cholesky"], TML["walk_fallbacks"], TML["readout_s"]), flush=True)
        if s11 is not None:
            print("%s T/rs %.4f: M min outside the cell %s at r %s (negative points %d); U_phys min %s; positive-branch v residual %s; rho + wJ min outside the cell %s at r %s (heavy dust %s, heavy pairs %s, light fed %s); residuals W %s 1/N %s; continued %s; %s; %s (%.1f s)" % (
                "S11" if a.massless_ball else "C-S11", T / rs, fmt(s11["M_min_outside_cell"], "%+.6e"), fmt(s11["r_at_M_min"], "%.3f"), s11["negative_points_outside_cell"], fmt(s11["U_phys_min_outside_cell"], "%+.4e"),
                fmt(s11["positive_branch_v_residual_max"], "%.2e"), fmt(s11["source_min_outside_cell"], "%+.4e"), fmt(s11["r_at_source_min"], "%.3f"),
                fmt(s11.get("ball_source_min_outside_cell"), "%+.4e"), fmt(s11.get("pairs_source_min_outside_cell"), "%+.4e"), fmt(s11.get("light_source_min_outside_cell"), "%+.4e"),
                fmt(s11["residual_W"], "%.1e"), fmt(s11["residual_invN"], "%.1e"), s11["continued_intervals"],
                "; ".join("dust beyond %s r_s: weight %s (fraction %s), energy %s, pairs energy %s" % (
                    k_.replace("ball_beyond_", ""), fmt(v_["weight"], "%.4e"), fmt(v_["weight_fraction"], "%.4e"), fmt(v_["energy_density_route"], "%+.4e"), fmt(v_["pairs_energy_density_route"], "%+.4e"))
                    for k_, v_ in s11.items() if k_.startswith("ball_beyond_")), ("STOP: " + s11["stop"]["case"]) if s11.get("stop") else "no stop", s11["seconds"]), flush=True)
        if step > step0 and (step - step0) % 10 == 0:
            last = [q["seconds"] for q in recs[-10:] if "seconds" in q]
            mean = float(np.mean(last)) if last else 0.0
            remain = (nsteps - step) * mean
            print("projection at T/rs %.2f: %.0f s per re-solve over the last 10; %d re-solves left; about %.1f hours" % (T / rs, mean, nsteps - step, remain / 3600.0), flush=True)
        if step % 5 == 0 or step == nsteps or step == step0:
            with open(path, "w") as fh:
                json.dump({"script": "engine_twofield_v5.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v4
            save_fields()
        if a.segment_end_T_over_rs is not None and T / rs >= a.segment_end_T_over_rs - 1e-9:
            end34 = {"T": T, "T_over_rs": T / rs, "segment_end_T_over_rs": a.segment_end_T_over_rs, "rule": "version 34 (b): the segment ends after this record"}
            print("T/rs %.4f: version 34 (b): the segment ends after this record" % (T / rs), flush=True)
            break
        if solve32 is not None and a.end_above is not None and solve32["kept_residual"] > a.end_above:
            solve32["ends_member"] = True
            end32 = {"T": T, "T_over_rs": T / rs, "kept_residual": solve32["kept_residual"], "end_above": a.end_above, "kept_from": solve32["kept_from"],
                     "continuation": solve32["continuation"],
                     "rule": "engine spec 32.1 (3): a record whose kept residual exceeds the bound after the continuation ends the member at that record; no record is propagated from an uncertified state"}
            print("T/rs %.3f: the member ends at this record (engine spec 32.1 (3)): the kept residual %.2e (%s) exceeds %g after the source continuation (%s); the record is written and no record is propagated from an uncertified state" % (
                T / rs, solve32["kept_residual"], solve32["kept_from"], a.end_above, solve32["continuation"]), flush=True)
            break
        if s11 is not None and s11.get("stop") and a.s11_stop and (a.massless_ball or light_on):          # v36: the P2 stop applies to the 2-field run with the coupling on
            audit = s11_audit_save(a, st, chs, cache, N_app, v_app, R, T, rs, eps, J, eps_pairs, J_pairs, info, s11)
            auditL = c_audit_save_v36(a, chl, st, cache, masksL, sigL, muL, nplusL, N_app, v_app, R, T, rs, eps_light_fed, J_light_fed, dsL, JsL, light_label) if chl else None
            end33 = {"T": T, "T_over_rs": T / rs, "stop": s11["stop"], "audit": audit, "audit_light": auditL,
                     "rule": "P2 (S11): the record stopped where M(r) < 0 outside the central averaging cell (r > ell); the three cases of P28 sec. 4 told apart; nothing propagated from it"}
            print("T/rs %.4f: STOP (P2): M(r) < 0 outside the central averaging cell from r %.4f (M min %.6e): %s; the audit state %s (light %s); the record is written and nothing is propagated from it" % (
                T / rs, s11["stop"]["r_first_negative_M"], s11["stop"]["M_min"], s11["stop"]["case"], audit, auditL), flush=True)
            with open(path, "w") as fh:
                json.dump({"script": "engine_twofield_v5.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v4
            save_fields()
            break
        if a.benchmark or step >= nsteps:
            break
        if v32:
            base_last = base
        t_p = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[ch.kappa]
            prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
            cache[ch.kappa] = (None, None, None, ref, _ci)
            del e, U
        sync_dev()
        last_prop_s = time.time() - t_p
        t_pl = time.time()
        for chL in chl:                                                                                 # v36: the light seas propagated by the record's eigendecomposition
            A, e, U = coefL[chL.key]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            st[chL.key]["sea"] = U @ (ph[:, None] * A)
            cache[chL.key] = (None, None, None, cache[chL.key][3], cache[chL.key][4])
            del A, e, U
        coefL.clear()
        sync_dev()
        TML["propagation_s"] = time.time() - t_pl
        T += a.tau
    with open(path, "w") as fh:
        json.dump(dict({"script": "engine_twofield_v5.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v4
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
    save_fields()


# ------------------------------------------------------------------------------------- version 31's newton_record, with v36 marks
def newton_record(a, chs, st, cache0, sources, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), sgn_dev=None, S_host=None):
    """version 31: the record's fixed point by Newton-Krylov (docstring); returns the loop's variables."""
    from scipy.optimize import newton_krylov, NoConvergence
    n = a.nsites
    nk = {"best": None, "n": 0, "E_prev": E0, "failed": None}

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if a.reference == "R2":
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36
        srcs_ = sources(cache_)
        return cache_, srcs_, do_solve(srcs_)

    def F(x):
        N_a = np.array(x[:n], dtype=float); v_a = np.array(x[n:], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        nk["n"] += 1
        if info_.get("failed"):
            nk["failed"] = info_["failed"]
            return np.full(2 * n, 1.0)
        E_ = FLD.charge(info_, a.g)
        dN_ = float(np.max(np.abs(Nn_ - N_a))); dv_ = float(np.max(np.abs(vn_ - v_a)))
        dE_ = abs(E_ - nk["E_prev"]) / abs(E_); nk["E_prev"] = E_
        it_hist.append([float(dE_), dN_, dv_])
        r_ = max(dN_, dv_)
        if nk["best"] is None or r_ < nk["best"]["resid"]:
            nk["best"] = {"resid": r_, "k_it": nk["n"], "cache": cache_, "srcs": srcs_, "sol": sol_, "app": (N_a.copy(), v_a.copy()), "dE": dE_, "E": E_}
        return np.concatenate([Nn_ - N_a, vn_ - v_a])

    t0 = time.time()
    x0 = np.concatenate([np.asarray(Nn0, float), np.asarray(vn0, float)])
    status = "converged"
    try:
        newton_krylov(F, x0, f_tol=a.stop_fields, maxiter=a.nk_maxiter, method="lgmres", inner_maxiter=a.nk_inner, verbose=False)
    except NoConvergence:
        status = "no convergence within %d Newton steps" % a.nk_maxiter
    except Exception as ex:  # a failed solve inside a line search, a non-finite residual
        status = "stopped: %s" % str(ex)[:120]
    b = nk["best"]
    if b is None:
        return nk["n"], False, None, cache0, None, Nn0, vn0, None, {"failed": nk["failed"] or "no evaluation"}, Nn0, vn0
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
    cache_c, srcs_c, sol_c = evalG(np.array(Nn_b, float), np.array(vn_b, float))
    nk["n"] += 1
    dE_c, r_c = None, None
    if not sol_c[3].get("failed"):
        E_c = FLD.charge(sol_c[3], a.g)
        dE_c = abs(E_c - b["E"]) / abs(E_c)
        r_c = max(float(np.max(np.abs(sol_c[0] - Nn_b))), float(np.max(np.abs(sol_c[1] - vn_b))))
        it_hist.append([float(dE_c), float(np.max(np.abs(sol_c[0] - Nn_b))), float(np.max(np.abs(sol_c[1] - vn_b)))])
    conv = b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E
    b["dE"] = dE_c if dE_c is not None else b["dE"]
    print("T/rs %.3f newton: %s; %d evaluations in %.0f s; kept residual %.2e (evaluation %d), the confirming plain step's residual %s and abs dE/E %s; %s" % (
        T / (a.rs), status, nk["n"], time.time() - t0, b["resid"], b["k_it"], "%.2e" % r_c if r_c is not None else "none", "%.1e" % dE_c if dE_c is not None else "none",
        "converged" if conv else "not converged"), flush=True)
    N_app_b, v_app_b = b["app"]
    return nk["n"], conv, b, b["cache"], b["srcs"], Nn_b, vn_b, Wn_b, info_b, N_app_b, v_app_b


# ------------------------------------------------------------------------------------ version 32's Newton stage, with v36 marks
def nk_stage_v32(a, chs, st, cache0, sources_fn, do_solve, x0, it_hist, T, name, start, ctr, chl=(), light_on=False, sgn_dev=None, S_host=None, v4_report=False):   # v4
    """version 32's nk_stage_v32 (its docstring) with the light channels in every evaluation and the light sources in the exact J.v (v36).
    v4: with v4_report the stage records and prints the largest displacement of any evaluation from its start (displacement_report_v4)."""
    n = a.nsites
    nk = {"best": None, "failed": None, "disp_max": 0.0, "disp_index": None}                             # v4: the displacement readout's state
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if a.reference == "R2":
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36
        srcs_ = sources_fn(cache_)
        return cache_, srcs_, do_solve(srcs_)

    def F(x):
        N_a = np.array(x[:n], dtype=float); v_a = np.array(x[n:], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        ctr["n"] += 1
        if v4_report:                                                                                   # v4: the largest displacement from the stage's start
            dxa = np.abs(np.asarray(x, dtype=float) - x0v); i_ = int(np.argmax(dxa))
            if dxa[i_] > nk["disp_max"]:
                nk["disp_max"] = float(dxa[i_]); nk["disp_index"] = i_
        if jv_on:
            memo["x"] = np.asarray(x, dtype=float).copy(); memo["cache"] = cache_; memo["srcs"] = srcs_
        if info_.get("failed"):
            nk["failed"] = info_["failed"]
            return np.full(2 * n, np.inf if V35.V36["fixes"] else 1.0)     # v2: version 36 (c), a failed solve is rejected
        E_ = FLD.charge(info_, a.g)
        dN_ = float(np.max(np.abs(Nn_ - N_a))); dv_ = float(np.max(np.abs(vn_ - v_a)))
        dE_ = abs(E_ - ctr["E_prev"]) / abs(E_)
        if not V35.V36["fixes"]:
            ctr["E_prev"] = E_
        elif nk["best"] is None or max(dN_, dv_) < nk["best"]["resid"]:
            ctr["E_prev"] = E_                                          # v2: version 36 (b), moved at the accepted (best) iterate only
        it_hist.append([float(dE_), dN_, dv_])
        r_ = max(dN_, dv_)
        if nk["best"] is None or r_ < nk["best"]["resid"]:
            nk["best"] = {"resid": r_, "k_it": ctr["n"], "cache": cache_, "srcs": srcs_, "sol": sol_, "app": (N_a.copy(), v_a.copy()), "dE": dE_, "E": E_}
        return np.concatenate([Nn_ - N_a, vn_ - v_a])

    def jvp(x, dx):
        """version 35: the exact J.v at the iterate x (the docstring's --nk-jvp exact); v36: the light sourcing channels added."""
        torch = BK["torch"]
        t_j = time.time()
        if memo["x"] is None or not np.array_equal(memo["x"], np.asarray(x, dtype=float)):
            F(np.asarray(x, dtype=float))             # the iterate's own evaluation (counted as one), when the last one was elsewhere
            JV["extra_evaluations"] += 1
        cache_, srcs_ = memo["cache"], memo["srcs"]
        dN = np.asarray(dx[:n], dtype=float); dv = np.asarray(dx[n:], dtype=float)
        deps = np.zeros(n); dJ = np.zeros(n)
        for ch in chs:
            _, e, U, ref, cinfo = cache_[ch.kappa]
            mask = torch.as_tensor(np.asarray(cinfo["_mask"], bool), device=U.device)
            Uo = U[:, mask]; Uu = U[:, ~mask]
            eo = e[mask].to(torch.float64); eu = e[~mask].to(torch.float64)
            dh = dev_csr(clocked_host_sparse(ch, dN, dv))
            Hm = Uu.conj().T @ (dh @ Uo)
            den = eo[None, :] - eu[:, None]
            small = den.abs() < 1e-12
            JV["dropped_pairs"] += int(small.sum())
            M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
            X = Uu @ M
            if JV["ref_mismatch"] is None or JV["_chk"] < len(chs):
                dg = (ref.conj() * ref).real.sum(dim=1) - (Uo.conj() * Uo).real.sum(dim=1)
                JV["ref_mismatch"] = max(JV["ref_mismatch"] or 0.0, float(dg.abs().max())); JV["_chk"] += 1
            for A_, acc in ((ch.h, deps), (ch.Q, dJ)):
                d_ = (Uo.conj() * (A_ @ X)).real.sum(dim=1) + (X.conj() * (A_ @ Uo)).real.sum(dim=1)
                acc -= ch.w * d_.reshape(-1, 2).sum(dim=1).cpu().numpy()
        if light_on:                                                                                    # v36: the light sourcing channels' dP (cross-sector pairs dropped: [dh, S] = 0)
            for chL in chl:
                if not chL.sourcing:
                    continue
                _, e, U, ref, cinfo = cache_[chL.key]
                mask = torch.as_tensor(np.asarray(cinfo["_mask"], bool), device=U.device)
                sig = cinfo["_sigma"]
                Uo = U[:, mask]; Uu = U[:, ~mask]
                eo = e[mask].to(torch.float64); eu = e[~mask].to(torch.float64)
                so = sig[mask]; su = sig[~mask]
                dh = dev_csr(clocked_host_sparse(chL, dN, dv))
                Hm = Uu.conj().T @ (dh @ Uo)
                den = eo[None, :] - eu[:, None]
                small = den.abs() < 1e-12
                cross = (su[:, None] * so[None, :]) < 0.0
                TML["jvp_dropped_pairs"] += int(small.sum()); TML["jvp_cross_sector_pairs"] += int(cross.sum())
                drop = small | cross
                M = torch.where(drop, torch.zeros_like(Hm), Hm / torch.where(drop, torch.ones_like(den), den).to(Hm.dtype))
                X = Uu @ M
                for A_, acc in ((chL.h, deps), (chL.Q, dJ)):
                    d_ = (Uo.conj() * (A_ @ X)).real.sum(dim=1) + (X.conj() * (A_ @ Uo)).real.sum(dim=1)
                    acc -= (chL.w if LS["s"] == 1.0 else chL.w * LS["s"]) * d_.reshape(-1, 2).sum(dim=1).cpu().numpy()   # v5: the ramp's weight
                TML["jvp_products"] += 1
        eps0 = np.asarray(srcs_[0], float); J0 = np.asarray(srcs_[1], float)
        sc = max(float(np.max(np.abs(deps))), float(np.max(np.abs(dJ))))
        if sc == 0.0:
            out = -np.asarray(dx, dtype=float)
        else:
            t = 1e-4 * max(1.0, float(np.max(np.abs(eps0))), float(np.max(np.abs(J0)))) / sc
            sp_ = do_solve((eps0 + t * deps, J0 + t * dJ) + tuple(srcs_[2:]))
            sm_ = do_solve((eps0 - t * deps, J0 - t * dJ) + tuple(srcs_[2:]))
            if sp_[3].get("failed") or sm_[3].get("failed"):
                raise RuntimeError("version 35: a field solve failed inside the exact J.v")
            dNo = (np.asarray(sp_[0]) - np.asarray(sm_[0])) / (2.0 * t); dvo = (np.asarray(sp_[1]) - np.asarray(sm_[1])) / (2.0 * t)
            out = np.concatenate([dNo - dN, dvo - dv])
        JV["products"] += 1; JV["seconds"] += time.time() - t_j
        return out

    jv_on = getattr(a, "nk_jvp", None) == "exact"
    memo = {"x": None, "cache": None, "srcs": None}
    if jv_on:
        F.jvp = jvp
    t0 = time.time()
    h0 = len(it_hist)
    x0v = np.concatenate([np.asarray(x0[0], float), np.asarray(x0[1], float)])
    steps = []
    try:
        if v32_own_loop(a):
            status, steps = nk_loop_v32(a, F, x0v)
        else:
            from scipy.optimize import newton_krylov, NoConvergence
            try:
                newton_krylov(F, x0v, f_tol=a.stop_fields, maxiter=a.nk_maxiter, method="lgmres", inner_maxiter=a.nk_inner, verbose=False)
                status = "converged"
            except NoConvergence:
                status = "no convergence within %d Newton steps" % a.nk_maxiter
    except Exception as ex:  # a failed solve inside a line search, a non-finite residual
        status = "stopped: %s" % str(ex)[:120]
    out = {"name": name, "start": start, "status": status, "line_search": steps, "best": nk["best"], "failed": nk["failed"], "evalG": evalG,
           "r_c": None, "dE_c": None, "conv": False}
    if nk["best"] is not None:
        out.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))
    out["evaluations"] = ctr["n"] - n_start
    out["seconds"] = time.time() - t0
    out["history"] = [h0, len(it_hist)]
    if v4_report:                                                                                       # v4
        out["v4_displacement"] = displacement_report_v4(a, T, name, nk, x0v)
    return out


def displacement_report_v4(a, T, name, nk, x0v):
    """v4: the largest displacement of any evaluation of a Newton stage from the stage's start, the field and the site that carry it (the note's
    ill-conditioning readout: a direction orders above the residual marks a near-null direction of I - G' that the inner solve left unresolved);
    printed, and returned for the stage's record entry (the option on and the coupling on only)."""
    i_ = nk.get("disp_index")
    if i_ is None:
        return None
    n = a.nsites
    field = "N" if i_ < n else "v"; site = int(i_ if i_ < n else i_ - n)
    out = {"max_abs_displacement": float(nk["disp_max"]), "field": field, "site": site, "R_over_rs": (site + (0.5 if a.half_offset else 1.0)) / a.rs,
           "start_norm_inf": float(np.max(np.abs(x0v)))}
    print("T/rs %.3f newton (%s) v4: the trial fields ranged up to %.3e from the stage's start, in %s at site %d (R/rs %.3f)" % (
        T / a.rs, name, out["max_abs_displacement"], field, site, out["R_over_rs"]), flush=True)
    return out


def plain_closure_v4(a, stage, it_hist, T, ctr, max_steps):
    """v4 stage C: plain fixed-point steps x <- G(x) from the stage's kept iterate's output while each lowers the residual max(dN, dv) (the coupled
    map contracted by 2.7 at the sigma/a 2 stall), at most max_steps; the bookkeeping is nk_stage_v32's F (ctr, it_hist, E_prev under version 36's
    fix (b)); the kept rule unchanged (the smallest residual); the confirming step is redone on the new kept iterate."""
    evalG = stage["evalG"]; b = stage["best"]
    path = [b["resid"]]; taken = 0
    while taken < max_steps:
        N_a = np.array(b["sol"][0], dtype=float); v_a = np.array(b["sol"][1], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        ctr["n"] += 1; taken += 1
        Nn_, vn_, Wn_, info_ = sol_
        if info_.get("failed"):
            path.append(None)
            break
        E_ = FLD.charge(info_, a.g)
        dN_ = float(np.max(np.abs(Nn_ - N_a))); dv_ = float(np.max(np.abs(vn_ - v_a)))
        dE_ = abs(E_ - ctr["E_prev"]) / abs(E_)
        r_ = max(dN_, dv_)
        it_hist.append([float(dE_), dN_, dv_])
        path.append(r_)
        better = r_ < b["resid"]
        if better or not V35.V36["fixes"]:
            ctr["E_prev"] = E_
        if not better:
            break
        b = {"resid": r_, "k_it": ctr["n"], "cache": cache_, "srcs": srcs_, "sol": sol_, "app": (N_a.copy(), v_a.copy()), "dE": dE_, "E": E_}
        if r_ < a.stop_fields:
            break
    stage["best"] = b
    stage.update(confirm_v32(a, b, evalG, it_hist, ctr))
    stage["v4_plain_closure"] = {"steps": taken, "residual_path": path, "kept_evaluation": b["k_it"], "kept_residual": b["resid"]}
    print("T/rs %.3f newton (v4 stage C, plain steps from stage B's kept iterate): %d step(s); residual path %s; kept %.2e (evaluation %d), the confirming plain step's residual %s and abs dE/E %s; %s" % (
        T / a.rs, taken, ", ".join((("%.2e" % q) if q is not None else "failed") for q in path), b["resid"], b["k_it"],
        fmt(stage.get("r_c"), "%.2e"), fmt(stage.get("dE_c"), "%.1e"), "converged" if stage.get("conv") else "not converged"), flush=True)
    return stage


def start_continuation_v4(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, x0, it_hist, T, ctr, d, solves, kw, plain_steps):
    """v4: the first record's continuation in the light coupling (the docstring): stage A, the uncoupled map from version 31's start (the light
    channels not evaluated, the light sources absent from eps, J and the J.v; the heavy sum bitwise version 35's); stage B, the coupled map from
    stage A's kept iterate's applied fields; stage C (plain_closure_v4) when B misses the stop rule. The kept rule is version 32's: the smaller
    kept residual of the direct solve and stage B/C; stage A is never kept. Returns (the continuation's label, the final stage, the stage list)."""
    kwA = dict(kw); kwA.update(chl=(), light_on=False)
    sA = nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with_heavy(base_new, c_)), do_solve, x0, it_hist, T,
                      "v4 stage A, the light coupling off", "the first solve's output (version 31's start)", ctr, **kwA)
    solves.append(sA)
    print_stage_v32(a, T, sA)
    if sA["best"] is None:
        d["confirm_repeated"] = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)
        return "v4: ran at the first record; stage A (the light coupling off) has no evaluation (%s); kept from the direct solve" % sA["failed"], d, [sA]
    xA = (np.array(sA["best"]["app"][0], dtype=float), np.array(sA["best"]["app"][1], dtype=float))
    tagA = "stage A %s (kept residual %.2e, confirming %s)" % ("met the stop rule" if sA["conv"] else "did not meet the stop rule", sA["best"]["resid"], fmt(sA.get("r_c"), "%.2e"))
    drop_arrays_v32(sA)
    sB = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, xA, it_hist, T, "v4 stage B, the light coupling on", "stage A's kept iterate's applied fields", ctr, **kw)
    solves.append(sB)
    print_stage_v32(a, T, sB)
    if sB["best"] is not None and not sB["conv"] and plain_steps > 0:
        plain_closure_v4(a, sB, it_hist, T, ctr, plain_steps)
    if sB["best"] is not None and sB["best"]["resid"] < d["best"]["resid"]:
        drop_arrays_v32(d)
        return ("v4: ran at the first record (the light coupling continuation; %s); kept from stage B%s" % (tagA, " after stage C's plain steps" if sB.get("v4_plain_closure") else ""), sB, [sA, sB])
    if sB["best"] is not None:
        drop_arrays_v32(sB)
    d["confirm_repeated"] = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)
    return "v4: ran at the first record (the light coupling continuation; %s); kept from the direct solve (stage B's kept residual is not smaller)" % tagA, d, [sA, sB]


def central_cell_v5(a, info):
    """v5: the central cell at a kept iterate (C_s2_start_judgment_v2.md, section 1): the dense lapse at the centre, W/(2g) at r = 1, 2 and ell (the form the
    record's W_over_2g readouts use), the smoothed energy density at the centre and J_ell/v_dense at r = 1 (the momentum equation fixes the lapse through
    it, N'/N proportional to r N J/v in the flat-slice chart, and v in the cell is set by the cell's mass). An error entry where the solve carries no dense fields."""
    out = {}
    try:
        r0 = np.array([0.0]); r1 = np.array([1.0]); r2 = np.array([2.0]); rl = np.array([float(a.ell)])
        W = info["W_field"]
        first = lambda f_, r_: float(np.asarray(f_(r_), dtype=float).ravel()[0])
        out["N_dense_0"] = first(info["N_dense"], r0)
        out["W_over_2g_at_1"] = first(W, r1) / (2.0 * a.g); out["W_over_2g_at_2"] = first(W, r2) / (2.0 * a.g); out["W_over_2g_at_ell"] = first(W, rl) / (2.0 * a.g)
        out["rho_ell_0"] = first(info["rho_ell"], r0)
        J1 = first(info["J_ell"], r1); v1 = first(info["v_dense"], r1)
        out.update({"J_ell_1": J1, "v_dense_1": v1, "J_over_v_at_1": (J1 / v1) if v1 != 0.0 else None})
    except Exception as ex:                                                                             # a solve without these callables
        out["error"] = str(ex)[:120]
    return out


def coupling_ramp_v5(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, x0, it_hist, T, ctr, solves, kw, ramp, bisect_max):
    """v5: the light-coupling ramp (C_s2_start_judgment_v2.md, section 2): the record's fixed point sought along the homotopy G_s = Solve(heavy sources +
    s x light sources). The base stage s = 0 is the uncoupled map (the light channels not evaluated, the light sources absent from eps, J and the J.v; the
    heavy sum bitwise version 35's) from version 31's start; each target s of the list is version 32's Newton stage (nk_stage_v32) on the coupled map with
    the light sources and their exact J.v weighted by s (LS["s"]), started from the previous stage's kept iterate's applied fields; a stage that misses the
    stop rule is retried over the half interval from the last converged s, at most bisect_max times per target, and the ramp stops at a stall (recorded
    with its interval: the fold readout); a closing stage at s = 1 then runs from the last converged iterate unless one already ran from it; a base that
    misses the stop rule is not walked (its stages would start off the branch), only closed. Only an s = 1 stage can be kept (the caller applies version
    32's kept rule); every other stage's arrays are dropped once its applied fields are taken. Per stage the entry carries s, the status, the kept and
    confirming residuals, the largest trial displacement (v4's readout), the kept applied fields' change from the stage's start with its field and site,
    and the central cell at the kept iterate (central_cell_v5)."""
    n = a.nsites
    t_r = time.time()

    def stage_at(s, start, name, start_desc):
        LS["s"] = float(s)
        try:
            if s == 0.0:
                kwA = dict(kw); kwA.update(chl=(), light_on=False)
                return nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with_heavy(base_new, c_)), do_solve, start, it_hist, T, name, start_desc, ctr, **kwA)
            return nk_stage_v32(a, chs, st, cache0, src_new, do_solve, start, it_hist, T, name, start_desc, ctr, **kw)
        finally:
            LS["s"] = 1.0

    def app_of(stage):
        return (np.array(stage["best"]["app"][0], dtype=float), np.array(stage["best"]["app"][1], dtype=float))

    def entry(stage, s, s_from, x_from):
        b = stage.get("best")
        e = {"s": s, "s_from": s_from, "name": stage["name"], "status": stage["status"], "converged": bool(stage.get("conv")), "evaluations": stage.get("evaluations"),
             "seconds": stage.get("seconds"), "kept_residual": (b["resid"] if b is not None else None), "confirming_residual": stage.get("r_c"), "confirming_dE": stage.get("dE_c"),
             "displacement": stage.get("v4_displacement"), "change_from_start": None, "central_cell": None}
        if b is not None:
            xa = np.concatenate([np.asarray(b["app"][0], dtype=float), np.asarray(b["app"][1], dtype=float)]); xf = np.concatenate([x_from[0], x_from[1]])
            d_ = np.abs(xa - xf); i_ = int(np.argmax(d_)); site = int(i_ if i_ < n else i_ - n)
            e["change_from_start"] = {"max_abs": float(d_[i_]), "field": ("N" if i_ < n else "v"), "site": site, "R_over_rs": (site + (0.5 if a.half_offset else 1.0)) / a.rs,
                                      "N_site0": float(xa[0]), "v_site0": float(xa[n])}
            e["central_cell"] = central_cell_v5(a, b["sol"][3])
        cc = e["central_cell"] or {}; ch_ = e["change_from_start"] or {}
        print("T/rs %.3f v5 ramp s = %.6g (from s = %.6g): %s; kept %s, confirming %s; the kept applied fields moved %s from the stage's start (in %s at site %s, R/rs %s; N at site 0 %s); "
              "central cell at the kept iterate: N_dense(0) %s, W/2g at r = 1, 2, ell %s %s %s, rho_ell(0) %s, J_ell/v at r = 1 %s" % (
                  T / a.rs, s, s_from, ("converged" if e["converged"] else "NOT converged (%s)" % stage["status"]), fmt(e["kept_residual"], "%.2e"), fmt(e["confirming_residual"], "%.2e"),
                  fmt(ch_.get("max_abs"), "%.3e"), ch_.get("field"), ch_.get("site"), fmt(ch_.get("R_over_rs"), "%.3f"), fmt(ch_.get("N_site0"), "%.7f"),
                  fmt(cc.get("N_dense_0"), "%.7f"), fmt(cc.get("W_over_2g_at_1"), "%.4e"), fmt(cc.get("W_over_2g_at_2"), "%.4e"), fmt(cc.get("W_over_2g_at_ell"), "%.4e"),
                  fmt(cc.get("rho_ell_0"), "%.3e"), fmt(cc.get("J_over_v_at_1"), "%.4e")), flush=True)
        return e

    targets = [float(s_) for s_ in ramp]
    out = {"targets": targets, "bisect_max": int(bisect_max), "base": None, "walked": False, "stages": [], "bisections": 0, "reached_1": False, "stalled": None,
           "closing_stage": None, "final_kept_residual": None, "final_converged": None, "label": None, "ctr_end": None, "seconds": 0.0, "final": None}
    base = stage_at(0.0, x0, "v5 ramp base, s = 0 (the light coupling off)", "the first solve's output (version 31's start)")
    solves.append(base)
    print_stage_v32(a, T, base)
    if base["best"] is None:
        out.update({"label": "v5: the light-coupling ramp has no base stage evaluation (%s); the kept rule runs over the other stages" % base["failed"], "ctr_end": ctr["n"],
                    "seconds": time.time() - t_r})
        return out
    out["base"] = entry(base, 0.0, 0.0, (np.asarray(x0[0], dtype=float), np.asarray(x0[1], dtype=float)))
    base_conv = bool(base["conv"])
    x_cur = app_of(base); s_cur = 0.0
    drop_arrays_v32(base)
    final = None; stalled = None; s1_from = None
    cap = 3 * len(targets) + 2; n_st = 0; i = 0
    if base_conv:
        out["walked"] = True
        while i < len(targets) and stalled is None and n_st < cap:
            s_t = targets[i]; s_try = s_t; nb = 0
            while True:
                n_st += 1
                sg = stage_at(s_try, x_cur, "v5 ramp s = %.6g" % s_try, "the ramp's kept iterate at s = %.6g" % s_cur)
                solves.append(sg)
                print_stage_v32(a, T, sg)
                out["stages"].append(entry(sg, s_try, s_cur, x_cur))
                if s_try >= 1.0:
                    s1_from = s_cur
                if sg["best"] is not None and bool(sg["conv"]):
                    x_cur = app_of(sg); s_cur = s_try
                    if s_try >= 1.0:
                        if final is not None:
                            drop_arrays_v32(final)
                        final = sg
                    else:
                        drop_arrays_v32(sg)
                    break
                if s_try >= 1.0 and sg["best"] is not None and (final is None or sg["best"]["resid"] < final["best"]["resid"]):
                    if final is not None:
                        drop_arrays_v32(final)
                    final = sg
                elif sg["best"] is not None:
                    drop_arrays_v32(sg)
                if nb >= bisect_max or n_st >= cap:
                    stalled = {"s_from": s_cur, "s_to": s_try, "bisections": nb, "stages": n_st, "cap_reached": bool(n_st >= cap)}
                    break
                nb += 1; out["bisections"] += 1; s_try = 0.5 * (s_cur + s_try)
            if stalled is None and s_cur >= s_t:
                i += 1
        if stalled is None and s_cur < 1.0:
            stalled = {"s_from": s_cur, "s_to": (targets[i] if i < len(targets) else 1.0), "bisections": out["bisections"], "stages": n_st, "cap_reached": True}
    out["reached_1"] = bool(s_cur >= 1.0); out["stalled"] = stalled
    if (not base_conv or stalled is not None) and s_cur < 1.0 and s1_from != s_cur:
        sg = stage_at(1.0, x_cur, "v5 ramp closing stage, s = 1", "the ramp's kept iterate at s = %.6g" % s_cur)
        solves.append(sg)
        print_stage_v32(a, T, sg)
        e_ = entry(sg, 1.0, s_cur, x_cur); e_["closing"] = True; out["stages"].append(e_)
        out["closing_stage"] = {k_: e_.get(k_) for k_ in ("converged", "kept_residual", "confirming_residual", "status")}
        if sg["best"] is not None and (final is None or bool(sg["conv"]) or sg["best"]["resid"] < final["best"]["resid"]):
            if final is not None:
                drop_arrays_v32(final)
            final = sg
        elif sg["best"] is not None:
            drop_arrays_v32(sg)
    out["final"] = final
    out["final_kept_residual"] = final["best"]["resid"] if (final is not None and final.get("best") is not None) else None
    out["final_converged"] = bool(final["conv"]) if final is not None else None
    tgt = ",".join(("%g" % s_) for s_ in targets)
    if not base_conv:
        how = "the base did not meet the stop rule, so the ramp was not walked (its stages would start off the branch) and only the closing stage at s = 1 ran from the base's kept iterate"
    elif out["reached_1"]:
        how = "reached s = 1"
    else:
        how = ("stalled between s = %g and s = %g after %d bisection(s)%s: the fixed points' branch has no continuation there within the stop rule (a fold) or needs finer steps" % (
            stalled["s_from"], stalled["s_to"], stalled["bisections"], (" (the stage cap %d reached)" % cap) if stalled.get("cap_reached") else ""))
    fin = (("the s = 1 stage met the stop rule (kept %.2e)" % out["final_kept_residual"]) if out["final_converged"] else
           (("the s = 1 stage missed the stop rule (kept %.2e)" % out["final_kept_residual"]) if out["final_kept_residual"] is not None else "no s = 1 stage"))
    out["label"] = "v5: the light-coupling ramp ran (base s = 0 %s, kept %s; targets %s; %d stage(s), %d bisection(s); %s; %s)" % (
        "met the stop rule" if base_conv else "did not meet the stop rule", fmt(out["base"]["kept_residual"], "%.2e"), tgt, len(out["stages"]), out["bisections"], how, fin)
    out["ctr_end"] = ctr["n"]; out["seconds"] = time.time() - t_r
    print("T/rs %.3f %s (%.0f s)" % (T / a.rs, out["label"], out["seconds"]), flush=True)
    return out


def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None, ramp=None, ramp_bisect=2,   # v5: the light-coupling ramp
                      sources_with_heavy=None, start_cont="off", plain_steps=12):                                                      # v4
    """version 32's record solve (its docstring), calling this file's nk_stage_v32 (v36); the midpoint base of stage 1 covers every key of base_new,
    the light channels' included. v4: at the first record (no previous record), with start_cont "coupling" and the coupling on, a direct solve that
    misses the stop rule is followed by start_continuation_v4 (stages A, B, C); with start_cont "off" nothing here differs from version 3."""
    ctr = {"n": 0, "E_prev": E0}
    src_new = lambda c_: sources_with(base_new, c_)
    solves = []
    kw = dict(chl=chl, light_on=light_on, sgn_dev=sgn_dev, S_host=S_host)                               # v36
    kw["v4_report"] = bool((start_cont != "off" or bool(ramp)) and light_on)   # v5: the ramp too;                                            # v4: the displacement readout, the option on and the coupling on only
    d = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, (Nn0, vn0), it_hist, T, "direct", "the first solve's output (version 31's start)", ctr, **kw)
    solves.append(d)
    print_stage_v32(a, T, d)
    if d["best"] is None:
        e32 = {"options": v32_options(a), "solves": [stage_entry_v32(q) for q in solves], "continuation": "not run (the direct solve has no evaluation)",
               "kept_from": None, "kept_residual": None, "confirming_residual": None, "ends_member": False}
        return ctr["n"], False, None, cache0, None, Nn0, vn0, None, {"failed": d["failed"] or "no evaluation"}, Nn0, vn0, e32
    ramp_v5 = None                                                                                      # v5: the light-coupling ramp (C_s2_start_judgment_v2.md, section 2)
    if (not d["conv"]) and ramp and light_on and sources_with_heavy is not None:
        ramp_v5 = coupling_ramp_v5(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, (Nn0, vn0), it_hist, T, ctr, solves, kw, ramp, ramp_bisect)
    final = d
    v4_stages = None                                                                                    # v4
    if d["conv"]:
        cont = "not needed (the direct solve meets the stop rule)"
    elif ramp_v5 is not None and ramp_v5.get("final") is not None and ramp_v5["final"].get("conv"):          # v5: the ramp's s = 1 stage meets the stop rule
        final = ramp_v5["final"]; drop_arrays_v32(d); cont = ramp_v5["label"]
    elif (base_prev is None or f_prev is None) and start_cont == "coupling" and light_on and sources_with_heavy is not None:   # v4: the first record's continuation
        cont, final, v4_stages = start_continuation_v4(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, (Nn0, vn0), it_hist, T, ctr, d, solves, kw, plain_steps)
    elif a.source_continuation != "two-stage":
        cont = "off"
    elif base_prev is None or f_prev is None:
        cont = "not run: no previous record in this run"
    else:
        base_mid = {k: tuple(0.5 * (p_ + q_) for p_, q_ in zip(base_prev[k], base_new[k])) for k in base_new}
        s1 = nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with(base_mid, c_)), do_solve, f_prev, it_hist, T,
                          "stage 1, the midpoint covariance", "the previous record's accepted applied fields", ctr, **kw)
        solves.append(s1)
        print_stage_v32(a, T, s1)
        if s1["best"] is None:
            cont = "ran; stage 1 has no evaluation (%s); kept from the direct solve" % s1["failed"]
            s2 = None
        else:
            x1 = s1["best"]["app"]
            drop_arrays_v32(s1)
            s2 = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, x1, it_hist, T, "stage 2, this record's covariance", "stage 1's kept iterate's applied fields", ctr, **kw)
            solves.append(s2)
            print_stage_v32(a, T, s2)
        if s2 is not None and s2["best"] is not None and s2["best"]["resid"] < d["best"]["resid"]:
            final = s2
            drop_arrays_v32(d)
            cont = "ran; kept from stage 2"
        else:
            if s2 is not None:
                drop_arrays_v32(s2)
                cont = "ran; kept from the direct solve (stage 2's kept residual is not smaller)"
            rc = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)   # the kept iterate's confirming step again, the history's last entry
            d["confirm_repeated"] = rc
    if ramp_v5 is not None and ramp_v5.get("final") is not None and ramp_v5["final"] is not final:          # v5: version 32's kept rule over the ramp's s = 1 stage
        rf = ramp_v5["final"]
        if rf.get("best") is not None and (final.get("best") is None or rf["best"]["resid"] < final["best"]["resid"]):
            drop_arrays_v32(final); d.pop("confirm_repeated", None)
            if ctr["n"] > ramp_v5["ctr_end"]:
                rf["confirm_repeated"] = confirm_v32(a, rf["best"], rf["evalG"], it_hist, ctr)                  # the kept iterate's confirming step again, the history's last entry
            final = rf; cont = "%s; kept from the ramp's s = 1 stage: %s (v5)" % (cont, ramp_v5["label"])
        else:
            if rf.get("best") is not None:
                drop_arrays_v32(rf)
            cont = "%s; the ramp's s = 1 stage not kept: %s (v5)" % (cont, ramp_v5["label"])
    elif ramp_v5 is not None and ramp_v5.get("final") is None:
        cont = "%s; %s (v5)" % (cont, ramp_v5["label"])
    b = final["best"]
    b["dE"] = final["dE_c"] if final["dE_c"] is not None else b["dE"]
    for q in solves:
        q.pop("evalG", None)
    e32 = {"options": v32_options(a), "solves": [stage_entry_v32(q) for q in solves], "continuation": cont, "kept_from": final["name"],
           "kept_residual": b["resid"], "kept_evaluation": b["k_it"], "confirming_residual": final["r_c"], "confirming_dE": final["dE_c"],
           "meets_stop_rule": final["conv"], "evaluations": ctr["n"], "ends_member": False}
    if "confirm_repeated" in d:
        e32["confirming_step_repeated"] = {"residual": d["confirm_repeated"]["r_c"], "dE": d["confirm_repeated"]["dE_c"]}
    if v4_stages is not None:                                                                            # v4: the record's entry (the option on and run only)
        sA_, sB_ = v4_stages[0], (v4_stages[1] if len(v4_stages) > 1 else None)
        e32["light_start_continuation_v4"] = {"option": start_cont, "plain_steps_max": plain_steps, "stages": [q["name"] for q in v4_stages],
                                              "stage_A": {"converged": sA_.get("conv"), "kept_residual": (sA_.get("best") or {}).get("resid"), "confirming_residual": sA_.get("r_c"),
                                                          "status": sA_.get("status")},
                                              "stage_B": ({"converged": sB_.get("conv"), "kept_residual": (sB_.get("best") or {}).get("resid"), "confirming_residual": sB_.get("r_c"),
                                                           "status": sB_.get("status"), "plain_closure": sB_.get("v4_plain_closure")} if sB_ is not None else None),
                                              "displacements": {q["name"]: q.get("v4_displacement") for q in solves}}
    if ramp_v5 is not None:                                                                              # v5: the ramp's record entry
        e32["light_coupling_ramp_v5"] = {k_: v_ for k_, v_ in ramp_v5.items() if k_ != "final"}
        if final is not d and "confirm_repeated" in final:
            e32["confirming_step_repeated"] = {"residual": final["confirm_repeated"]["r_c"], "dE": final["confirm_repeated"]["dE_c"]}
    if ramp and light_on and (base_prev is None or f_prev is None):                                        # v5: the first record's rule, the production's practice made explicit
        e32["first_record_rule_v5"] = ("the first record is written when its kept residual is below --end-above (%s), the production's practice on its sigma/a 2 member "
                                       "(member2 record 0: unconverged, 1.857e-04 at iteration 15; its 0.0625 converged); nothing propagated differs from version 4" % (a.end_above,))
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
    N_app_b, v_app_b = b["app"]
    return ctr["n"], final["conv"], b, b["cache"], b["srcs"], Nn_b, vn_b, Wn_b, info_b, N_app_b, v_app_b, e32


# ------------------------------------------------------------------------------------------------------- version 35's main, with v36 args
def main():
    global FLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--shell", default=None, help="R_in,R_out in units of r_s (S-shell)")
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--prep-ell", dest="prep_ell", type=int, default=None, help="the averaging scale of the initial data (default --ell)")
    ap.add_argument("--field-rtol", type=float, default=1e-12); ap.add_argument("--field-atol", type=float, default=1e-16)
    ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--reference", default="R2", choices=["R2", "R1"])
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--collar-policy", default="flag", choices=["flag", "fail"])
    ap.add_argument("--support-rel", type=float, default=1e-14)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--site-fields", default="averaged", choices=["averaged", "dense"])
    ap.add_argument("--stop-E", type=float, default=1e-7)
    ap.add_argument("--stop-fields", type=float, default=1e-6)
    ap.add_argument("--maxit", type=int, default=20)
    ap.add_argument("--mix", default="plain", choices=["plain", "anderson", "damped", "newton"]); ap.add_argument("--nk-maxiter", type=int, default=8); ap.add_argument("--nk-inner", type=int, default=15); ap.add_argument("--mix-alpha", type=float, default=0.5, help="the damped mixer's step (version 30)")
    ap.add_argument("--delta-rtol-every", type=int, default=10)
    ap.add_argument("--commutator-every", type=int, default=1)
    ap.add_argument("--streams", type=int, default=0)
    ap.add_argument("--m1-simpson-steps", default="auto")
    ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0)
    ap.add_argument("--window-max", type=int, default=4)
    ap.add_argument("--resume-from", default=None, help="dir/tag of a stored engine_collapse_v17 run")
    ap.add_argument("--resume-T-over-rs", type=float, default=None)
    ap.add_argument("--resume-grandparent", default=None, help="the v17 run under a v19/v24 parent (overrides the parent's stored path)")
    ap.add_argument("--slices-c128", default="")
    ap.add_argument("--slices-c64", default="")
    ap.add_argument("--slice-dir", default=None)
    ap.add_argument("--nk-line-search", default=None, choices=["halving"], help="(a) the backtracking line search on the fixed-point residual (version 32)")
    ap.add_argument("--nk-halvings", type=int, default=None, help="(a) at most this many halvings (6 when absent)")
    ap.add_argument("--nk-krylov", type=int, default=None, help="(b) LGMRES Krylov vectors per cycle (scipy's default 30 when absent)")
    ap.add_argument("--nk-restarts", type=int, default=None, help="(b) LGMRES restarts (none when absent)")
    ap.add_argument("--source-continuation", default=None, choices=["two-stage"], help="(c) the two-stage continuation in the propagated covariance")
    ap.add_argument("--end-above", type=float, default=None, help="(d) the member ends at a record whose kept residual exceeds this after the continuation")
    ap.add_argument("--massless-ball", dest="massless_ball", action="store_true", help="(b) the massless Fermi ball's preparation (requires --m 0)")
    ap.add_argument("--ball-R0-over-rs", dest="ball_R0_over_rs", type=float, default=4.0)
    ap.add_argument("--ball-x0-over-R0", dest="ball_x0_over_R0", type=float, default=0.5)
    ap.add_argument("--ball-sigma-over-R0", dest="ball_sigma_over_R0", type=float, default=0.25)
    ap.add_argument("--ball-orbitals", dest="ball_orbitals", default="1:6", help="kappa:b,... the chiral seeds per channel")
    ap.add_argument("--ball-E-target", dest="ball_E_target", type=float, default=None, help="the solved charge's target (default r_s/(2g))")
    ap.add_argument("--ball-k0-max", dest="ball_k0_max", type=float, default=0.5)
    ap.add_argument("--ball-E-tol", dest="ball_E_tol", type=float, default=1e-10)
    ap.add_argument("--ball-passes", dest="ball_passes", type=int, default=200)
    ap.add_argument("--ball-tol", dest="ball_tol", type=float, default=1e-10)
    ap.add_argument("--fields-v17", dest="fields_v17", action="store_true", help="(e) engine_fields_v17 for every solve")
    ap.add_argument("--s11-collars", dest="s11_collars", default="8,16")
    ap.add_argument("--s11-stop", dest="s11_stop", type=int, default=1)
    ap.add_argument("--resume-slice", dest="resume_slice", default=None, help="(v34 a) a stored complex128 slice of a v33, v34, v35 or v36 run to restart from")
    ap.add_argument("--resume-run", dest="resume_run", default=None, help="(v34 a) that run's dir/tag")
    ap.add_argument("--segment-end-T-over-rs", dest="segment_end_T_over_rs", type=float, default=None, help="(v34 b) end after the record at this clock")
    ap.add_argument("--weyl-certified", dest="weyl_certified", default=None, choices=["full", "eigvals"], help="(v34 c) the certified continuation's route (heavy field)")
    ap.add_argument("--weyl-refine-max", dest="weyl_refine_max", type=int, default=10, help="(v34 c) at most this many halvings of a sub-step")
    ap.add_argument("--weyl-oracle", dest="weyl_oracle", action="store_true", help="(v34 d) the full-route oracle at every record's accepted fields")
    ap.add_argument("--field-kernel-cache", dest="field_kernel_cache", action="store_true", help="(v36) the field solve's kernels cached, bitwise")   # v2
    ap.add_argument("--v36-fixes", dest="v36_fixes", action="store_true", help="(v36) #65 item 7's three corrections")                       # v2
    ap.add_argument("--nk-jvp", dest="nk_jvp", default=None, choices=["exact"], help="(v35) the Krylov Jacobian's product by the exact linearization")
    # v36: the light field
    ap.add_argument("--light-kappas", dest="light_kappas", default="1,2", help="(v36) the light field's sourcing channels (empty: none)")
    ap.add_argument("--light-guard-kappas", dest="light_guard_kappas", default="3", help="(v36) the truncation guard's channels (propagated, never sourcing)")
    ap.add_argument("--light-coupling", dest="light_coupling", default="on", choices=["on", "off"], help="(v36) the light sources in eps, J and the exact J.v")
    ap.add_argument("--light-transport", dest="light_transport", default="banded", choices=["banded", "v34-eigvals", "v3"], help="(v36) the light reference's continuation")
    ap.add_argument("--light-d-min", dest="light_d_min", type=float, default=1e-7, help="(v36) the smallest certifiable gap of the walk; v3: raised to the Cholesky's rounding floor 2 sqrt(64 eps) ||K||_G where that is larger")
    ap.add_argument("--light-max-pieces", dest="light_max_pieces", type=int, default=4096, help="(v36) at most this many pieces per walk")
    ap.add_argument("--light-tol-block", dest="light_tol_block", type=float, default=1e-8, help="(v36) the degenerate-block tolerance of the sector rotation")
    ap.add_argument("--light-sigma-tol", dest="light_sigma_tol", type=float, default=1e-6, help="(v36) the sector impurity above which an evaluation falls back")
    ap.add_argument("--slices-light-c128", dest="slices_light_c128", default="", help="(v36) T/rs clocks of light-only complex128 slices")
    ap.add_argument("--c-readouts", dest="c_readouts", type=int, default=1, help="(v36) the mass function's sign per record and the C readouts")
    ap.add_argument("--c-heavy-einv", dest="c_heavy_einv", type=int, default=1, help="(v36) the heavy field's E_inv per record")
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    ap.add_argument("--light-start-continuation", dest="light_start_continuation", default="off", choices=["off", "coupling"],
                    help="(v4) at the first record, when the direct coupled Newton solve misses the stop rule: stage A the uncoupled map, stage B the coupled map from A's kept fields, stage C plain steps; off: version 3 bitwise")   # v4
    ap.add_argument("--light-coupling-ramp", dest="light_coupling_ramp", default="",
                    help="(v5) at a record whose direct coupled Newton solve misses the stop rule: the light sources' weights s of the homotopy from the uncoupled map (s = 0) to the coupled map (s = 1), increasing, ending at 1, for example 0.25,0.5,0.75,1; empty: off (version 4 bitwise)")   # v5
    ap.add_argument("--light-ramp-bisect", dest="light_ramp_bisect", type=int, default=2, help="(v5) a ramp stage that misses the stop rule is retried over the half interval from the last converged weight, at most this many times per target")   # v5
    ap.add_argument("--light-start-plain-steps", dest="light_start_plain_steps", type=int, default=12, help="(v4) stage C: at most this many plain fixed-point steps (0: none)")   # v4
    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:
        ap.error("version 33 (a): m = 0 runs only with --massless-ball")
    if a.massless_ball and a.m != 0.0:
        ap.error("version 33 (b): the massless ball needs --m 0 exactly")
    if a.massless_ball and a.ball_E_target is None:
        a.ball_E_target = a.rs / (2.0 * a.g)
    if a.fields_v17:
        import engine_fields_v17 as F17
        FLD = F17; V35.FLD = F17                                                                        # v36: both modules' field module
    if a.field_kernel_cache:                                                                            # v2: version 36's main, on the same modules
        import engine_fields_kcache_v1 as KC
        import engine_fields_v4 as _F4, engine_fields_v12 as _F12, engine_fields_v14 as _F14, engine_fields_v16 as _F16
        KC.install([_F4, _F12, _F14, _F16, FLD]); V35.V36["kcache"] = KC.STATS
    V35.V36["fixes"] = bool(a.v36_fixes)                                                                # v2
    if a.resume_slice and (a.resume_from or not a.resume_run):
        ap.error("version 34 (a): --resume-slice needs --resume-run and excludes --resume-from")
    if a.nk_jvp and (a.mix != "newton" or not v32_own_loop(a)):
        ap.error("version 35: --nk-jvp acts inside version 32's own Newton loop (--mix newton with --nk-line-search, --nk-krylov or --nk-restarts)")
    if a.weyl_oracle and a.reference != "R2":
        ap.error("version 34 (d): the oracle compares R2's continuation")
    if a.light_start_continuation != "off" and (a.mix != "newton" or not v32_own_loop(a) or a.light_follow):                              # v4
        ap.error("v4: --light-start-continuation acts inside version 32's own Newton loop (--mix newton with --nk-line-search, --nk-krylov or --nk-restarts) and not under --light-follow")
    a.light_ramp = []                                                                                                                    # v5: the parsed ramp (empty: off)
    if a.light_coupling_ramp:
        try:
            a.light_ramp = [float(x_) for x_ in a.light_coupling_ramp.split(",") if x_.strip()]
        except ValueError:
            ap.error("--light-coupling-ramp: a comma-separated list of weights")
        if not a.light_ramp or any(not (0.0 < s_ <= 1.0) for s_ in a.light_ramp) or any(q_ >= p_ for q_, p_ in zip(a.light_ramp, a.light_ramp[1:])) or a.light_ramp[-1] != 1.0:
            ap.error("--light-coupling-ramp: increasing weights in (0, 1] ending at 1 (for example 0.25,0.5,0.75,1)")
        if a.mix != "newton" or not v32_own_loop(a) or a.light_follow:
            ap.error("v5: --light-coupling-ramp acts inside version 32's own Newton loop (--mix newton with --nk-line-search, --nk-krylov or --nk-restarts) and not under --light-follow")
    if a.light_ramp_bisect < 0:                                                                                                          # v5
        ap.error("--light-ramp-bisect must be nonnegative")
    if a.light_start_plain_steps < 0:                                                                                                    # v4
        ap.error("--light-start-plain-steps must be nonnegative")
    if v32_on(a) and a.mix != "newton":
        ap.error("version 32's options act on the Newton solve: --mix newton is required")
    if a.nk_halvings is not None and a.nk_line_search != "halving":
        ap.error("--nk-halvings needs --nk-line-search halving")
    for k_, v_ in (("--nk-halvings", a.nk_halvings), ("--nk-krylov", a.nk_krylov), ("--nk-restarts", a.nk_restarts)):
        if v_ is not None and v_ < (0 if k_ != "--nk-krylov" else 1):
            ap.error("%s must be %s" % (k_, "at least 1" if k_ == "--nk-krylov" else "nonnegative"))
    if a.reference != "R2" and (a.light_kappas or a.light_guard_kappas):
        ap.error("v36: the light field's continued reference needs --reference R2")
    OPT["collar_policy"] = a.collar_policy; OPT["support_rel"] = a.support_rel; OPT["site_fields"] = a.site_fields
    SOLVE_TOL["rtol"] = a.field_rtol; SOLVE_TOL["atol"] = a.field_atol
    C5.backend(a.gpu)
    run(a)


if __name__ == "__main__":
    main()
