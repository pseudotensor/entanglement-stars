#!/usr/bin/env python
"""engine_collapse_v11.py (builder analyst; version 11 = version 10 on engine_fields_v9 with on_maxit = "return": a re-solve whose
iteration reaches the limit (200) with the larger of its last changes below 1e-8 no longer stops the run; the last iterate is the pair,
the record carries solver.not_converged = {dN, dW, it, stall_cap}, the line prints "NOT CONVERGED (dN, dW)", and engine 17.3 (1)'s
certificate decides the label as before (B with q_c over the last three changes; B_prefloor printed beside it, not used for the label);
a change at the limit above 1e-8 still rejects and stops. Cause: the v9 primary (r_s 64, ell 4, tau 8, R2) stopped at T/rs 5.88 with
dN 1.77e-11, dW 1.31e-10 after 200 iterations (version 7's floor rule needs three changes below 1e-11). Nothing else changes.
Version 10's docstring follows.
engine_collapse_v10.py (builder analyst; engine 17.3 (3) to (5) and the certificates of (1) and (2) inline; version 10 = version 9 on
engine_fields_v8 plus: (a) per record the certificate of engine_fields_v8.certify (q_c and its pre-floor value, eta, eps_map at rtol 1e-10 on
every record, B or "uncertified", the U = W/N^2 and ln N routes' defects, the residual ladder and the matching region's residuals, the collar's
tail sums against 1e-9) and the label "engine_fields_v8; tau = ...; certified | uncertified", sticky from the first failing re-solve;
(b) R2 at avoided crossings: no early return; every continuation passes the midpoint configuration (the fields interpolated linearly), and
deeper halvings (to 64 sub-steps) when needed; a block is accepted only when its weight on the previous reference is within 0.1 of an integer
and every eigenvalue of its weight matrix lies within 0.1 of 0 or 1 (the resolved split); otherwise the re-solve is rejected and the run stops
with the record; the smallest occupied-to-empty gap per channel at the start, the midpoint and the end is recorded (a partly occupied
degenerate block counts as gap 0, flagged); (c) M1's balance per interval: with X = Tr[C A_k], Tr[R2 A_k] at the record and Y = Tr[C A_(k-1)],
Tr[R2_k A_(k-1)] at the old configuration, the exact split d Tr(D A) = evolution under the frozen h + field work + reference reset (their sum
closing to rounding), the evolution term against the interval's trapezoid of I_E; (d) per-jet errors: two Richardson estimates (h = 1/4, 1/8 and
1/8, 1/16) of kappa_T and of A, B, C of v and v/N, their differences recorded; v/N's jets in z = kappa' (R - R_t) with kappa' = (v/N)' at R_t
(observables 13.1; kappa_T/N_t printed against it; the kappa_T-normalized v/N jets kept beside, labelled); (e) the fields npz written every 5 records (certification can
run during the run), with N_applied and v_applied: the fields of the last self-consistency iteration's h_m, the Hamiltonian that continued the
reference and propagated the sea (the stored N, v are that iteration's solve; they differ at the convergence level), so C5.3' can replay exactly; (f) every 10 re-solves the projection of the end and engine 12.6's decision printed. Version 9's docstring follows.
engine_collapse_v9.py (builder analyst; version 9: engine_collapse_v8 plus the pair block's own source exported per step: the fields npz
carries eps_pairs and J_pairs (the shell totals of the continued sea against its reference, without the dust) beside eps_tot and J_tot. The
spec's no-dust control (sec. 5.3, C5.3: "the vacuum under the primary run's stored fields: the created-pair source alone") evolves the same
initial sea under the same stored N, v; by linearity it reproduces this block exactly (the engine red team's remark: bookkeeping, not an
independent evolution), so the block is stored by the run itself instead of a second run. Nothing else changes. Version 8's docstring follows.
engine_collapse_v8.py (builder analyst; version 8: engine_collapse_v7 on engine_fields_v7, whose iteration accepts a stall at the
integrator's floor (both changes below 1e-11 for three iterations; flagged): version 7's r_s 64 ell 4 primary was rejected at T/rs 0.25 with
dN 7.6e-13 and dW 1.46e-12 after 200 iterations. Each record now carries the solver's converged_at_floor entry (None when converged below tol).
Nothing else changes. Version 7's docstring follows.
engine_collapse_v7.py (builder analyst; version 7: engine_collapse_v6 on engine_fields_v6, whose trapping-radius scan reaches the chain's end
(version 5 of the fields stopped at the collar). Nothing else changes. Version 6's docstring follows.
engine_collapse_v6.py (builder analyst; SPEC_final4b_engine.md sec. 16.2 items (1) to (9) and the restart at tau = 8; version 6 of
engine_collapse_v5.py, whose docstring and helpers (the channel chains, the dust's preparation, the energy cut current) it imports).

Version 6: (a) the fields from engine_fields_v5 (the lead's "engine_fields_v4" of 16.2: dense-function convergence and rejection, parity-extended
sources, the support of both sources for the collar, refined trapping zeros, the exterior W as the field, the amended dictionary line); the
collar policy --collar-policy flag (BUILDER, labelled: the continued sea polarizes the whole chain, so the sources' support reaches the chain's
end and "fail" would stop every re-solve; the margin's deficit and the sources' share in the last 8 ell are recorded per re-solve) or fail;
(b) B1 to the fourth derivative of N + v at R_t by Richardson in h (1/4 and 1/8, central differences of the dense functions), with P25's jets
A, B, C of v and of v/N (z = kappa (R - R_t), kappa = d(N + v)/dR) and the exact profile's at the solved E; (c) M1 corrected: I_E =
Tr[(C - R2) i[h_m, H_in]] is +d<H_in>/dt (the rate of the inside Killing energy at the collar cut), printed beside dE/dT and d<H_in>/dT (from
successive re-solves) and the balance d(E - <H_in>)/dT; M9 stores the matched cut currents at 0.5 R_sh, R_t and the collar beside the rates of
W/(2g) there; (d) R2's continuation degeneracy aware: the current eigenvalues within 1e-8 of the band scale form blocks; a block's summed weight
on the previous reference fixes its occupied dimension (an integer within 0.1, else ambiguous), a partly occupied block takes the top
eigenvectors of its weight matrix; the total must equal n_sea; an ambiguous step is retried through intermediate configurations
h(s) = h_prev + s (h_cur - h_prev) with up to 6 halvings (64 sub-steps), and an ambiguity that survives is a rejected re-solve (the run stops
with the record written). R2 is this lane's declared reference: the adiabatically continued, never refilled sea (the literal propagated sea R1
sources nothing); (e) the export: the fields npz stores per slice the shell totals eps_tot and J_tot (dust plus pairs), with g, ell and the
solver's parameters, so a consumer re-solves the dense functions (engine_fields_v5.resolve_dense); the unit-site arrays stay; (f) the record's
labels: the collar charge W/(2g) and the Misner-Sharp W/(2 g N^2) at R_t, late_surface by the total source's vacuum condition (|rho_ell| and
|J_ell| over [R_t - 5 ell, R_t + 5 ell] below 1e-6 of their maxima; the ratios printed), v at R_c from the dense flow; the residuals, the source
splines' error, the centre series and the tail diagnostic per re-solve; "engine_fields_v6; tau = %g" in the label.
Usage: python engine_collapse_v6.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40
--kappa-sea 10 --reference R2 --out <dir> --tag <tag> [--gpu 0] [--collar-policy flag] [--benchmark]"""
import argparse
import json
import math
import os
import time

import numpy as np

import engine_collapse_v5 as C5
import engine_fields_v9 as FLD
import engine_mixing_v1 as MIX

BK = C5.BK
T_ = C5.T_
eigT = C5.eigT
dens = C5.dens
Chan = C5.Chan
logistic = C5.logistic
fill_lowest = C5.fill_lowest
threshold_radii = C5.threshold_radii
energy_cut_current = C5.energy_cut_current
OPT = {"collar_policy": "flag", "support_rel": 1e-14}


def solve(eps, J, g, R, ell, N0):
    return FLD.solve_constraints(eps, J, g, R, ell, N0=N0, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"], on_maxit="return", stall_cap=1e-8)


def prepare_dust(chs, g, R, ell, NB, R0, m, reading, shell, passes=200, tol=1e-10):
    """version 5's preparation (the levitating well, the zero-temperature filling, the projection) on engine_fields_v5."""
    torch = BK["torch"]
    nsites = len(R)
    N = np.ones(nsites); v = np.zeros(nsites)
    mixer = MIX.Anderson()
    U0 = 0.5 * m
    for ip in range(passes):
        Ns = np.sqrt(np.maximum(N * N - v * v, 1e-12)) if reading == "A" else N.copy()
        if shell:
            s_in, s_out = logistic(R, shell[0]), logistic(R, shell[1])
            Uw = U0 * (1 - s_in) + U0 * s_out + m * (1 - Ns) * s_in * (1 - s_out)
        else:
            s = logistic(R, R0)
            Uw = U0 * s + m * (1 - Ns) * (1 - s)
        Ud = T_(np.diag(np.repeat(Uw, 2)))
        spec, levels = {}, []
        for ic, ch in enumerate(chs):
            hw = ch.clocked_dense(N, v if reading == "A" else 0 * v) + Ud
            ew, Uwv = eigT(hw)
            em, Um = eigT(ch.clocked_dense(N, v))
            pos = ew > 0
            ewp = ew[pos].cpu().numpy()
            spec[ch.kappa] = (ewp, Uwv[:, pos], Um[:, em > 0], hw)
            levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ewp[:400])]
        occ = fill_lowest(levels, NB)
        eps = np.zeros(nsites); J = np.zeros(nsites); dust = {}; NBs = 0.0
        diag = {"discarded_norm_max": 0.0, "gram_cond_max": 1.0, "stationarity_max": 0.0}
        for ic, ch in enumerate(chs):
            ewp, Uwv, Pplus, hw = spec[ch.kappa]
            keep = [j for j in range(min(400, len(ewp))) if occ.get((ic, j), 0.0) > 0]
            if not keep:
                continue
            fk = np.array([occ[(ic, j)] for j in keep])
            raw = Pplus @ (Pplus.conj().T @ Uwv[:, keep])
            diag["discarded_norm_max"] = max(diag["discarded_norm_max"], float((1 - (raw.abs() ** 2).sum(dim=0)).max()))
            S = raw.conj().T @ raw
            sv, sU = torch.linalg.eigh(S)
            diag["gram_cond_max"] = max(diag["gram_cond_max"], float(sv.max() / sv.min().clamp_min(1e-300)))
            Phi = raw @ (sU * sv.clamp_min(1e-300).rsqrt()[None, :].to(sU.dtype)) @ sU.conj().T
            Ew = (Phi.conj() * (hw @ Phi)).sum(dim=0).real
            res = (hw @ Phi - Phi * Ew[None, :].to(Phi.dtype)).norm(dim=0)
            diag["stationarity_max"] = max(diag["stationarity_max"], float(res.max()))
            eps += ch.w * dens(ch.h, Phi, fk); J += ch.w * dens(ch.Q, Phi, fk); NBs += ch.w * fk.sum()
            dust[ch.kappa] = (Phi, fk)
        Nn, vn, Wn, info = solve(eps, J, g, R, ell, N)
        if info.get("failed"):
            raise RuntimeError("initial data: the solver failed: %s" % info["failed"])
        d = max(float(np.max(np.abs(Nn - N))), float(np.max(np.abs(vn - v))))
        x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
        N, v = x[:nsites], x[nsites:]
        if d < tol:
            break
    N, v, Wn, info = solve(eps, J, g, R, ell, N)
    if info.get("failed"):
        raise RuntimeError("initial data: the final solve failed: %s" % info["failed"])
    kin = 0.0
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]
            hD = (Phi.conj() * (ch.h @ Phi)).sum(dim=0).real.cpu().numpy()
            kin += ch.w * float(fk @ (hD - m))
    u = np.where(np.abs(eps) > 1e-14 * np.abs(eps).max(), J / np.where(np.abs(eps) > 0, eps, 1.0), 0.0)
    bulk = 0.5 * float(np.sum(eps * u * u))
    head = {"N_B": NBs, "passes": ip + 1, "last_change": d, "E0": FLD.charge(info, g), "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
            "site_fields": "dense (engine_fields_v9)", "f_kin": kin / (NBs * m), "f_kin_bulk_frame": (kin - bulk) / (NBs * m), "dust_momentum_total": float(J.sum()),
            "projection": diag, "reading": reading, "shell": shell, "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"),
            "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"), "collar_margin": info.get("collar_margin"),
            "solver_iterations": info.get("it"), "dictionary": FLD.DICTIONARY}
    return N, v, dust, info, head


def surface_jets(info, Rt, g, E):
    """B1 to the fourth derivative of f = N + v at R_t by Richardson in h (1/4, 1/8); P25's jets A, B, C of v and of v/N with z = kappa (R - R_t)."""
    if Rt is None:
        return None
    Nf = lambda x: float(info["N_dense"](x)[0]); vf = lambda x: float(info["v_dense"](x)[0])

    def d14(F, x, h):
        f = {q: F(x + q * h) for q in (-2, -1, 0, 1, 2)}
        d1 = (f[1] - f[-1]) / (2 * h)
        d2 = (f[1] - 2 * f[0] + f[-1]) / h ** 2
        d3 = (f[2] - 2 * f[1] + 2 * f[-1] - f[-2]) / (2 * h ** 3)
        d4 = (f[2] - 4 * f[1] + 6 * f[0] - 4 * f[-1] + f[-2]) / h ** 4
        return np.array([d1, d2, d3, d4])

    def rich(F):
        a, b = d14(F, Rt, 0.25), d14(F, Rt, 0.125)
        return (4.0 * b - a) / 3.0, float(np.max(np.abs(b - a)))

    def rich2(F):
        b, c = d14(F, Rt, 0.125), d14(F, Rt, 0.0625)
        return (4.0 * c - b) / 3.0
    fnv = lambda x: Nf(x) + vf(x)
    qv = lambda x: vf(x) / Nf(x)
    dfn, spread = rich(fnv)
    dv, _ = rich(vf)
    dq, _ = rich(qv)
    kap = float(dfn[0])
    dfn2 = rich2(fnv); dv2 = rich2(vf); dq2 = rich2(qv)
    kap2 = float(dfn2[0])
    jets = lambda d, k: [float(d[1] / (2 * k ** 2)), float(d[2] / (6 * k ** 3)), float(d[3] / (24 * k ** 4))]
    ex = [1.0 / (4 * g * E), -3.0 / (16 * (g * E) ** 2), 15.0 / (64 * (g * E) ** 3), -105.0 / (256 * (g * E) ** 4)]
    # v's jets in z = kappa_T (R - R_t); v/N's in z = kappa' (R - R_t), kappa' = (v/N)' = kappa_T/N_t (observables 13.1's lapse mapping; the
    # manufactured variable-lapse fixture showed the kappa_T normalization of v/N's jets off by powers of N_t)
    kq, kq2 = float(dq[0]), float(dq2[0])
    jv1, jv2 = jets(dv, kap), jets(dv2, kap2); jq1, jq2 = jets(dq, kq), jets(dq2, kq2)
    Nt = Nf(Rt)
    return {"d_N_plus_v": [float(x) for x in dfn], "over_exact": [float(dfn[i] / ex[i]) for i in range(4)], "richardson_spread": spread,
            "kappa_T": kap, "kappa_prime": kq, "N_t": Nt, "kappa_prime_minus_kappa_T_over_N_t": kq - kap / Nt,
            "jets_v": jv1, "jets_v_over_N": jq1, "jets_v_over_N_in_kappa_T": jets(dq, kap), "jets_exact": [-1.5, 2.5, -4.375],
            "kappa_T_second_estimate": kap2, "kappa_T_err": abs(kap - kap2), "jets_v_err": [abs(x - y) for x, y in zip(jv1, jv2)],
            "jets_v_over_N_err": [abs(x - y) for x, y in zip(jq1, jq2)]}


def inside_energy(hmT, cols, f, cut):
    """<H_in> = Tr[D H_in], H_in = (1/2){P_in, h_m}: Re sum_j f_j (P_in c_j)^dagger (h_m c_j)."""
    hc = hmT @ cols
    P = slice(0, 2 * cut)
    w = (cols[P, :].conj() * hc[P, :]).sum(dim=0).real.cpu().numpy()
    return float(np.dot(np.asarray(f, float), w))


def blocks_of(e, tol_rel=1e-8):
    e = np.asarray(e, float)
    scale = max(float(np.max(np.abs(e))), 1e-300)
    out, i = [], 0
    while i < len(e):
        j = i + 1
        while j < len(e) and abs(e[j] - e[j - 1]) < tol_rel * scale:
            j += 1
        out.append((i, j))
        i = j
    return out


def continue_reference_aware(ref_prev, e, U, n_sea):
    """one continuation step: (ref_new columns, info) or (None, info) when ambiguous."""
    torch = BK["torch"]
    A = ref_prev.conj().T @ U                                       # n_prev x n
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy()
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    cols, n_occ, amb, npocket, wmin, wmax_empty = [], 0, 0, 0, 1.0, 0.0
    for (i, j) in blocks_of(e_np):
        WB = float(w[i:j].sum()); nB = int(round(WB))
        if abs(WB - nB) >= 0.1:
            amb += 1
            continue
        size = j - i
        if nB == size:
            cols.append(U[:, i:j]); wmin = min(wmin, WB / size)
        elif nB > 0:
            M = A[:, i:j].conj().T @ A[:, i:j]
            ev, V = torch.linalg.eigh(M)
            top = V[:, -nB:]
            cols.append(U[:, i:j] @ top); wmin = min(wmin, float(ev[-nB:].real.min()))
            wmax_empty = max(wmax_empty, float(ev[:-nB].real.max()) if size > nB else 0.0)
        else:
            wmax_empty = max(wmax_empty, WB / size)
        n_occ += nB
        eB = float(np.mean(e_np[i:j]))
        npocket += nB if eB > 0 else (size - nB if eB < 0 else 0)
    info = {"ambiguous_blocks": amb, "n_occ": n_occ, "n_sea": n_sea, "n_pocket": npocket, "min_occupied_weight": wmin, "max_empty_weight": wmax_empty,
            "nneg": int((e_np < 0).sum())}
    if amb or n_occ != n_sea or not cols:
        return None, info
    return torch.cat(cols, dim=1), info


def continue_reference_resolved(ref_prev, e, U, n_sea, tol_split=0.1):
    """version 10's step: every block's weight-matrix eigenvalues within tol_split of 0 or 1 and its trace within tol_split of an integer (the
    resolved split); the occupied and empty eigenvalues collected for the gap (a partly occupied degenerate block gives gap 0)."""
    torch = BK["torch"]
    A = ref_prev.conj().T @ U
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy()
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    cols, n_occ, amb, npocket, wmin, wmax_empty, partial = [], 0, 0, 0, 1.0, 0.0, 0
    occ_e, emp_e = [], []
    for (i, j) in blocks_of(e_np):
        size = j - i
        if size == 1:
            lam = np.array([w[i]]); V = None
        else:
            M = A[:, i:j].conj().T @ A[:, i:j]
            lam_t, V = torch.linalg.eigh(M)
            lam = lam_t.real.cpu().numpy()
        WB = float(lam.sum()); nB = int(round(WB))
        if abs(WB - nB) >= tol_split or not bool(np.all((np.abs(lam) < tol_split) | (np.abs(lam - 1.0) < tol_split))):
            amb += 1
            continue
        if nB == size:
            cols.append(U[:, i:j]); wmin = min(wmin, float(lam.min())); occ_e += list(e_np[i:j])
        elif nB > 0:
            cols.append(U[:, i:j] @ V[:, -nB:]); wmin = min(wmin, float(lam[-nB:].min()))
            wmax_empty = max(wmax_empty, float(lam[:-nB].max())); partial += 1
            occ_e += list(e_np[i:j][:nB]); emp_e += list(e_np[i:j][nB:])
        else:
            wmax_empty = max(wmax_empty, float(lam.max())); emp_e += list(e_np[i:j])
        n_occ += nB
        eB = float(np.mean(e_np[i:j]))
        npocket += nB if eB > 0 else (size - nB if eB < 0 else 0)
    gap = None
    if occ_e and emp_e:
        lab = np.concatenate([np.zeros(len(occ_e)), np.ones(len(emp_e))]); vals = np.concatenate([occ_e, emp_e])
        o = np.argsort(vals); vs, ls = vals[o], lab[o]
        dif = np.abs(np.diff(vs))[ls[1:] != ls[:-1]]
        gap = float(dif.min()) if len(dif) else None
    info = {"ambiguous_blocks": amb, "n_occ": n_occ, "n_sea": n_sea, "n_pocket": npocket, "min_occupied_weight": wmin, "max_empty_weight": wmax_empty,
            "nneg": int((e_np < 0).sum()), "gap": gap, "partly_occupied_blocks": partial}
    if amb or n_occ != n_sea or not cols:
        return None, info
    return torch.cat(cols, dim=1), info


def continue_resolved(ref_prev, hm_prev, hm_cur, e_cur, U_cur, n_sea, max_halvings=6):
    """version 10: always through the midpoint (2 sub-steps), then 4, 8, ... 64 when a sub-step is unresolved; the gaps at the midpoint and the end."""
    last = None
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        r_ = ref_prev; ok = True; gap_mid = None
        for k in range(1, nsub + 1):
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                e_s, U_s = eigT(hm_prev + (k / nsub) * (hm_cur - hm_prev))
            r_, inf_ = continue_reference_resolved(r_, e_s, U_s, n_sea)
            last = inf_
            if r_ is None:
                ok = False
                break
            if k == nsub // 2:
                gap_mid = inf_["gap"]
        if ok:
            inf_["halvings"] = hv - 1
            inf_["gap_mid"] = gap_mid; inf_["gap_end"] = inf_["gap"]
            return r_, inf_
    info = dict(last or {}); info["halvings"] = max_halvings; info["rejected"] = True
    return None, info


def continue_with_halving(ref_prev, hm_prev, hm_cur, e_cur, U_cur, n_sea, max_halvings=6):
    """the step from the previous reference to the current eigenbasis; on ambiguity, through intermediate configurations, up to 6 halvings."""
    ref, info = continue_reference_aware(ref_prev, e_cur, U_cur, n_sea)
    info["halvings"] = 0
    if ref is not None or hm_prev is None:
        return ref, info
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        r_ = ref_prev
        ok = True
        for k in range(1, nsub + 1):
            s = k / nsub
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                e_s, U_s = eigT(hm_prev + s * (hm_cur - hm_prev))
            r_, inf_ = continue_reference_aware(r_, e_s, U_s, n_sea)
            if r_ is None:
                ok = False
                break
        if ok:
            inf_["halvings"] = hv
            return r_, inf_
    info["halvings"] = max_halvings
    info["rejected"] = True
    return None, info


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)
    label = "quasi-static (tau = %g); ell = %d; reference: %s (%s); site fields: dense; engine_fields_v9; tau = %g; collar policy %s; %s" % (
        a.tau, a.ell, a.reference, "the continued sea, degeneracy aware" if a.reference == "R2" else "the propagated sea", a.tau, OPT["collar_policy"], FLD.DICTIONARY)
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m)
    chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    shell = [float(x) * rs for x in a.shell.split(",")] if a.shell else None
    t0 = time.time()
    N, v, dust, info, head = prepare_dust(chs, a.g, R, a.ell, NB, a.R0_over_rs * rs, a.m, a.reading, shell)
    N_cinit, v_cinit = N.copy(), v.copy()                # the fields of the initial sea (C5.3' replays from them)
    head.update({"label": label, "rs_rest_label": rs, "R0": a.R0_over_rs * rs, "seconds": time.time() - t0})
    print("initial data (%s): N_B %.2f E0 %.6f passes %d f_kin %.4f (bulk frame %.4f) discarded %.2e Gram cond %.2e stationarity %.2e; the spline's distance from the dense fields at the sites N %.3e v %.3e; residuals W %.1e 1/N %.1e; source splines %s; centre series %.6f; collar %s (%.0f s)" % (
        label, head["N_B"], head["E0"], head["passes"], head["f_kin"], head["f_kin_bulk_frame"], head["projection"]["discarded_norm_max"],
        head["projection"]["gram_cond_max"], head["projection"]["stationarity_max"], head["interp_floor_N"] or float("nan"), head["interp_floor_v"] or float("nan"),
        head["residual_W"] or float("nan"), head["residual_invN"] or float("nan"), head["source_spline_error"], head["centre_series_ratio"] or float("nan"),
        head["collar_margin"], head["seconds"]), flush=True)
    E0 = head["E0"]
    st = {}
    for ch in chs:
        hm = ch.clocked_dense(N, v)
        e, U = eigT(hm)
        neg = e < 0
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), np.zeros(0)))
        st[ch.kappa] = {"sea": U[:, neg], "dust": Phi_d, "f": f_d, "ref": U[:, neg], "n_sea": int(neg.sum()), "hm_ref": hm}
    nsteps = int(round(a.T_over_rs * rs / a.tau))
    recs = []
    fields = {"T": [], "N": [], "v": [], "eps_tot": [], "J_tot": [], "eps_pairs": [], "J_pairs": [], "N_applied": [], "v_applied": [], "r_nodes": [], "N_nodes": [], "v_nodes": []}
    path = os.path.join(a.out, "collapse_%s.json" % a.tag)
    T = 0.0
    prevW = None
    prev_Hin = None
    cert_state = {"first_failure": None}
    t_run0 = time.time()

    def save_fields():
        np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),
                            eps_tot=np.array(fields["eps_tot"]), J_tot=np.array(fields["J_tot"]), eps_pairs=np.array(fields["eps_pairs"]), J_pairs=np.array(fields["J_pairs"]),
                            N_applied=np.array(fields["N_applied"]), v_applied=np.array(fields["v_applied"]), N_cinit=N_cinit, v_cinit=v_cinit,
                            g=a.g, ell=a.ell, tau=a.tau,
                            solver=json.dumps({"tol": 1e-12, "maxit": 200, "rtol": 1e-12, "atol": 1e-16, "collar_policy": OPT["collar_policy"], "support_rel": OPT["support_rel"],
                                               "module": "engine_fields_v9", "on_maxit": "return", "stall_cap": 1e-8}),
                            r_nodes=np.array(fields["r_nodes"], dtype=object), N_nodes=np.array(fields["N_nodes"], dtype=object),
                            v_nodes=np.array(fields["v_nodes"], dtype=object), label=label)
    for step in range(nsteps + 1):
        tr = time.time()
        mixer = MIX.Anderson()
        E_prev = None
        converged = False
        rejected = None
        for it in range(30):
            N_app, v_app = N.copy(), v.copy()            # the fields of this iteration's h_m: the applied Hamiltonian if this iteration is the last
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            cache = {}
            amb_tot, wmin, npocket, nrefill, dR2R0, halv = 0, 1.0, {}, {}, 0.0, 0
            for ch in chs:
                s_ = st[ch.kappa]
                hm = ch.clocked_dense(N, v)
                e, U = eigT(hm)
                if a.reference == "R2":
                    ref, cinfo = continue_resolved(s_["ref"], s_["hm_ref"], hm, e, U, s_["n_sea"])
                    if ref is None:
                        rejected = "kappa %d: the continuation stayed ambiguous after %d halvings (%s)" % (ch.kappa, cinfo.get("halvings", 0), {k: cinfo[k] for k in cinfo if k != "rejected"})
                        break
                    amb_tot += cinfo["ambiguous_blocks"]; wmin = min(wmin, cinfo["min_occupied_weight"]); halv += cinfo["halvings"]
                    npocket[ch.kappa] = cinfo["n_pocket"]
                    nrefill[ch.kappa] = cinfo["nneg"] - s_["n_sea"]
                    ov = float(((U[:, e < 0].conj().T @ ref).abs() ** 2).sum())
                    dR2R0 += math.sqrt(max(cinfo["nneg"] + s_["n_sea"] - 2 * ov, 0.0))
                else:
                    ref = s_["sea"]; cinfo = {}
                cache[ch.kappa] = (hm, e, U, ref, cinfo)
                ds = dens(ch.h, s_["sea"]) - dens(ch.h, ref); Js = dens(ch.Q, s_["sea"]) - dens(ch.Q, ref)
                dd = dens(ch.h, s_["dust"], s_["f"]); Jd = dens(ch.Q, s_["dust"], s_["f"])
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            if rejected:
                break
            Nn, vn, Wn, info = solve(eps, J, a.g, R, a.ell, N)
            if info.get("failed"):
                break
            E = FLD.charge(info, a.g)
            dE = abs(E - E_prev) / abs(E) if E_prev is not None else np.inf
            E_prev = E
            if dE < 1e-9:
                N, v = Nn, vn
                converged = True
                break
            x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
            N, v = x[:a.nsites], x[a.nsites:]
        if rejected or info.get("failed"):
            why = rejected or info["failed"]
            recs.append({"T": T, "failed": why, "label": label})
            print("T/rs %.2f: the re-solve is REJECTED (%s); the run stops with the record written" % (T / rs, why), flush=True)
            break
        if not converged:
            N, v = Nn, vn
        # the certificate (engine 17.3 (1), (2)) on this re-solve
        cert = FLD.certify(eps, J, a.g, R, a.ell, Nn, vn, info, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"], with_eps_map=True)
        if not (cert["certified"] and cert["collar"]["certified"]) and cert_state["first_failure"] is None:
            cert_state["first_failure"] = T / rs
            cert_state["which"] = ", ".join(w for w, ok in (("iteration", cert["certified"]), ("collar", cert["collar"]["certified"])) if not ok)
        cert_label = "certified" if cert_state["first_failure"] is None else ("uncertified from T/rs %.2f (%s)" % (cert_state["first_failure"], cert_state["which"]))
        # M1's balance (17.3 (4)): the old configuration's terms before the reference and h are replaced
        cut = int(min(max(round(info["r_collar"]) - 1, 2), a.nsites - 10))
        Y1 = Y2 = IEend = 0.0
        if recs and recs[-1].get("E") is not None:
            for ch in chs:
                s_ = st[ch.kappa]; hm_old = s_["hm_ref"]; ref_old = s_["ref"]; ref_new = cache[ch.kappa][3]
                Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
                Y1 += ch.w * inside_energy(hm_old, Ccols, fC, cut)
                Y2 += ch.w * inside_energy(hm_old, ref_new, np.ones(ref_new.shape[1]), cut)
                colsE = torch.cat([s_["sea"], s_["dust"], ref_old], dim=1)
                fE = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref_old.shape[1])])
                IEend += ch.w * energy_cut_current(hm_old, colsE, fE, cut)
        gaps = {}
        for ch in chs:
            ci = cache[ch.kappa][4]
            gaps[ch.kappa] = [st[ch.kappa].get("gap_end_prev"), ci.get("gap_mid"), ci.get("gap_end"), ci.get("partly_occupied_blocks")]
            st[ch.kappa]["gap_end_prev"] = ci.get("gap_end")
        for ch in chs:
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
            st[ch.kappa]["hm_ref"] = cache[ch.kappa][0]
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
        for ch in chs:
            hm, e, U, ref, _ci = cache[ch.kappa]
            s_ = st[ch.kappa]
            cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
            fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
            IE += ch.w * energy_cut_current(hm, cols, fs, cut)
            Hin += ch.w * inside_energy(hm, cols, fs, cut)
            Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
            X1 += ch.w * inside_energy(hm, Ccols, fC, cut)
            X2 += ch.w * inside_energy(hm, ref, np.ones(ref.shape[1]), cut)
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "E_dust_shells": float(eps_dust.sum()), "E_pairs_shells": float(eps_pairs.sum()),
               "pairs_inside_Rt": (float(eps_pairs[R < Rt].sum()) if Rt else None), "pairs_outside_Rt": (float(eps_pairs[R >= Rt].sum()) if Rt else float(eps_pairs.sum())),
               "R_t": Rt, "R_t_over_2gE": (Rt / (2 * a.g * E) if Rt else None), "W_over_2g_at_Rt": M_wg, "misner_sharp_at_Rt": M_ms,
               "R_t_over_2g_MS": (Rt / (2 * a.g * M_ms) if (Rt and M_ms) else None), "kappa_T": kT, "four_gE_kappa_T": (4.0 * a.g * E * kT if kT is not None else None),
               "surface_jets": jets, "r_in": r_in, "trapping_zeros": tz["zeros"], "tangencies": tz["tangencies"], "min_N_minus_absv": tz["min"], "r_min_N_minus_absv": tz["r_min"],
               "late_surface": late, "N_min": float(N.min()), "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(info["v_dense"](Rc)[0]) if Rc else None),
               "threshold_radius": threshold_radii(N, v, R, a.m, [1, 2, a.kappa_sea]), "n_pocket": npocket, "n_refill_R0": nrefill, "R2_minus_R0_F": dR2R0,
               "ambiguous_blocks": amb_tot, "min_occupied_weight": wmin, "halvings": halv, "M1_I_E": IE, "H_in": Hin, "iterations": it + 1, "converged": converged,
               "solver": {"it": info.get("it"), "dN": info.get("last_dN"), "dW": info.get("last_dW"), "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"),
                          "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"), "collar_margin": info.get("collar_margin"),
                          "W_tail": info.get("W_tail"), "W_roundoff_negative_points": info.get("W_roundoff_negative_points"),
                          "converged_at_floor": info.get("converged_at_floor"), "not_converged": info.get("not_converged")},
               "spectrum": info["spectrum"], "r_collar": info["r_collar"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
               "label": label, "seconds": time.time() - tr, "certificate": cert, "certified_label": "engine_fields_v9; tau = %g; %s" % (a.tau, cert_label),
               "r2_gaps": {str(k): v_ for k, v_ in gaps.items()}, "M1_parts": {"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2, "I_E_end_prev_interval": IEend, "I_E_start": IE, "cut": cut}}
        prevM = recs[-1].get("M1_parts") if (recs and recs[-1].get("E") is not None) else None
        if prevM is not None:
            dT = T - recs[-1]["T"]
            a_exact = Y1 - prevM["X1"]; b_work = (X1 - X2) - (Y1 - Y2); c_reset = -(Y2 - prevM["X2"]); dH = (X1 - X2) - (prevM["X1"] - prevM["X2"])
            rec["M1_balance"] = {"dHin": dH, "evolution_exact": a_exact, "evolution_trapezoid": 0.5 * dT * (prevM["I_E_start"] + IEend), "field_work": b_work,
                                 "reference_reset": c_reset, "closure": a_exact + b_work + c_reset - dH, "cut_changed": bool(prevM.get("cut") != cut),
                                 "per_unit_time": {"dHin": dH / dT, "evolution": a_exact / dT, "field_work": b_work / dT, "reference_reset": c_reset / dT}}
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
                        hm, e, U, ref, _ci = cache[ch.kappa]
                        s_ = st[ch.kappa]
                        cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
                        fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                        I_c += ch.w * energy_cut_current(hm, cols, fs, ci)
                    rec["M9"][lab] = {"r": rr_, "cut_site": ci, "dW_over_2g_dT": float((info["W_field"](rr_)[0] - prevW[0](rr_)[0]) / (2 * a.g * (T - prevW[1]))),
                                      "I_E_cut": I_c}
        if step % 10 == 0:
            prevW = (info["W_field"], T)
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy()); fields["eps_tot"].append(eps.copy()); fields["J_tot"].append(J.copy())
        fields["eps_pairs"].append(eps_pairs.copy()); fields["J_pairs"].append(J_pairs.copy())
        fields["N_applied"].append(N_app.copy()); fields["v_applied"].append(v_app.copy())
        fields["r_nodes"].append(info["r_nodes"]); fields["N_nodes"].append(info["N_nodes"]); fields["v_nodes"].append(info["v_nodes"])
        oe = jets["over_exact"] if jets else None
        gmin = [x for gl in gaps.values() for x in gl[:3] if x is not None]
        mb = rec.get("M1_balance")
        print("T/rs %.2f E/E0 %.7f dust %.4f pairs %.3e R_t %s (2gE %.3f; d1..d4 of N+v over exact %s; r_in %s) min(N-|v|) %.4f R_sh %s R_c %s pockets %s amb %d halv %d it %d M1 %.3e dE/dT %s dHin/dT %s solver it %s%s res %.1e %.1e; B %s (pre-floor %s) q_c %s eps_map %s collar dE/E %.1e; R2 gap min %s; M1 balance: evolution %s work %s reset %s closure %s [engine_fields_v9; tau = %g; %s] (%.0f s)" % (
            T / rs, E / E0, eps_dust.sum(), eps_pairs.sum(), ("%.3f" % Rt) if Rt else None, 2 * a.g * E, (", ".join("%.4f" % q for q in oe)) if oe else None,
            ("%.3f" % r_in) if r_in else None, tz["min"], Rsh, ("%.2f" % Rc) if Rc else None, sum(npocket.values()) if npocket else 0, amb_tot, halv, it + 1, IE,
            ("%.3e" % rec["dE_dT"]) if "dE_dT" in rec else "-", ("%.3e" % rec["dHin_dT"]) if "dHin_dT" in rec else "-", info.get("it"),
            (" NOT CONVERGED (dN %.1e, dW %.1e)" % (info["not_converged"]["dN"], info["not_converged"]["dW"])) if info.get("not_converged") else "",
            info.get("residual_W") or 0.0, info.get("residual_invN") or 0.0, ("%.1e" % cert["B"]) if cert["B"] is not None else "uncertified",
            ("%.1e" % cert["B_prefloor"]) if cert.get("B_prefloor") is not None else "-",
            ("%.3f" % cert["q_c"]) if cert["q_c"] is not None else "-", ("%.1e" % cert["eps_map"]) if cert["eps_map"] is not None else "-", cert["collar"]["delta_E_rel_abs"],
            ("%.2e" % min(gmin)) if gmin else "-", ("%.3e" % mb["evolution_exact"]) if mb else "-", ("%.3e" % mb["field_work"]) if mb else "-",
            ("%.3e" % mb["reference_reset"]) if mb else "-", ("%.1e" % mb["closure"]) if mb else "-", a.tau, cert_label, time.time() - tr), flush=True)
        if step > 0 and step % 10 == 0:
            last = [q["seconds"] for q in recs[-10:] if "seconds" in q]
            mean = float(np.mean(last)) if last else 0.0
            remain = (nsteps - step) * mean
            total_h = ((time.time() - t_run0) + remain) / 3600.0
            end = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + remain))
            print("projection at T/rs %.2f: %.0f s per re-solve over the last 10; %d re-solves left; total %.1f hours; the end about %s UTC; engine 12.6: %s" % (
                T / rs, mean, nsteps - step, total_h, end, "within the 20-hour rule" if total_h <= 20.0 else "exceeds 20 hours: the r_s = 32 tau ladder decides tau = 16 before any switch"),
                flush=True)
        if step % 5 == 0 or step == nsteps:
            with open(path, "w") as fh:
                json.dump({"script": "engine_collapse_v11.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
            save_fields()
        if a.benchmark or step >= nsteps:
            break
        for ch in chs:
            hm, e, U, ref, _ci = cache[ch.kappa]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[ch.kappa]
            prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        T += a.tau
    with open(path, "w") as fh:
        json.dump({"script": "engine_collapse_v11.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
    save_fields()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--shell", default=None, help="R_in,R_out in units of r_s (S-shell)")
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=4)
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
    a = ap.parse_args()
    OPT["collar_policy"] = a.collar_policy; OPT["support_rel"] = a.support_rel
    C5.backend(a.gpu)
    run(a)


if __name__ == "__main__":
    main()
