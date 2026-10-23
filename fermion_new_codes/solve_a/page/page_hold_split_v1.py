#!/usr/bin/env python
"""page_hold_split_v1.py (lead reviewer A for the solve lane's builder; reorientation 4 item C4 and the solve entry's item 4): the held
excitation's spectral split, the hold's principled reading in place of the emptying-clock extrapolation and the long chains.

Per hold (one --init-state, the hold's state at the freeze clock --freeze-at-rs) and channel kappa (--channels, those present in the state):
(1) the held one-particle operator h = h[N, v] of the frozen record on the stored chain (n2 = 2 nsites = 16384 rows), built as the hold and
    page_graft_pullback_v2 build it (PK.History(record, fields, nsites, None, key).fields(kf) at the record kf nearest the freeze clock and
    PK.ChannelOps(kappa, R, device).h_csr(N, v), imported read-only through page_graft_pullback_v2); its dense eigendecomposition
    h = V diag(e) V^dagger (page_ops_v6.eigh_csr, torch complex128, on --eig-device, V then on --device; the memory figure printed first, the
    measured device peak after); the Killing vacuum P_- = 1(h < 0) (zero modes empty, as page_core and the graft code; their count printed);
    the stored occupied columns Phi of the hold's state, C = Phi Phi^dagger.
(2) D = C - P_- in the eigenbasis through the blocks the readouts need only: the diagonal occupations n_alpha = ||V_alpha^dagger Phi||^2 (chunked
    over Phi's columns), d_alpha = n_alpha - 1(e_alpha < 0), E_inv = Tr[h D] = sum_alpha e_alpha d_alpha (page_core.initial_memory's inversion
    energy: the particle part sum_{e>0} e n and the hole part sum_{e<0} |e| (1 - n)), N_+, N_-, and ||D||_F^2 = Tr[C^2] - 2 Tr[C P_-] + n_neg from
    Phi's Gram matrix (its defect from the identity printed). No off-diagonal block of D is formed.
(3) every eigenvector's weight inside R_cut (24 r_s) and inside the collar (16 r_s); the split of E_inv into the bound part (weight inside
    R_cut at least --theta, 0.99) and the extended part, the thresholds of --theta-beside (0.9, 0.999) beside, by sign (particles, holes), and
    the same split by the collar weight; the parts sum to E_inv exactly (the self-test's item).
(4) per energy bin of |e|/T_H (--bin-TH wide to --bin-max-TH, then --bin-coarse-TH wide up to the band's top): the counts, the occupation sums
    and the Killing-energy sums of the bound and the extended parts on both halves (the extended part's Killing-energy density), and the
    extended part's exit scale per bin from the record's exterior symbol omega_+-(k) = +-N sqrt(sin^2 k + (kappa/R)^2 + m^2) + (v/2) sin 2k
    (page_graft_pullback_v2.split_weight's symbol): the slowest and the fastest outgoing root (d omega/dk > 0) at the chain's end, the box time
    (R_end - R_cut)/v_end, the path time sum dR / v_min(E; R) from R_cut to the end along the sampled exterior (every --exit-stride sites, the
    last site always; infinite where the energy stops propagating before the end, the turning radius recorded), on the stored chain's fields
    and on the long chain's; the classical turning radius of every eigenvector from the per-site band edges (the extended content split into
    what exits each chain's box and what turns before its end); the remaining-extended-energy envelope E_ext(t) = the extended content whose
    slowest carrier has not reached the chain's end by t (one pass, no reflection).
(5) the eigenvalue extremes, the Killing gap about zero, the Gershgorin bound, the band edges from the symbol (per site, the global extremes,
    at the chain's end).
(6) the chain-length dependence on the long grafted operator (--nsites-long sites, as page_graft_pullback_v2 builds it with append =
    nsites-long - nsites-short; default 69992): the eigenpairs nearest the band top and bottom (the shift at the symbol's global edge plus
    --si-offset, re-shifted to the found extreme while eigenvalues lie beyond the shift or a residual exceeds --si-res-tol) and nearest the
    exterior band edges at the long chain's end (--si-targets; extra shifts by --si-shifts), by shift-invert Arnoldi (scipy eigsh on the complex
    Hermitian CSC with a banded SuperLU factor in the natural ordering; --si-nev per target), their residuals ||h psi - e psi||, their weights
    inside R_cut, inside the collar and on the short chain's rows, their occupations in the graft state
    d^g_alpha = psi_alpha^dagger E D_S E^dagger psi_alpha = ||Phi^dagger psi_s||^2 - ||V_-^dagger psi_s||^2 (psi_s the window vector's
    short-chain rows, V_- the short chain's negative eigenvectors; the graft's deviation of page_graft_posbound_v1), the bound/extended split
    of sum e d^g over the window against the 16384-row operator's eigenpairs in the same energy range, and the window's bound modes matched one
    to one with the stored chain's (the nearest eigenvalue, its occupation, its weight).
Outputs (rule 10): <out>/<tag>_split.json (every number), <out>/<tag>_k<kappa>_split.npz per channel (the resume unit: every raw array:
eigenvalues, occupations, d, the weights, the per-bin arrays, the exit arrays, the envelopes, the per-site band edges, the long windows'
eigenvalues, residuals, weights, d^g and matches; the window vectors with --save-vectors 1) and <out>/<tag>_split.npz (every channel's arrays
with the prefix k<kappa>_). --resume 1 (default) keeps the channels finished before. EXPORT lines (one json object per channel: eigenvalue
extremes, band edges, the split; EXPORT-LONG for the long chain) are printed for the second_route lane. Every threshold is an argument.
Usage: --record --fields --init-state --freeze-at-rs --out --tag [--channels 1,2,3] [--device cuda:0] [--eig-device cpu]
[--nsites-short 8192] [--nsites-long 69992] [--theta 0.99] [--long 1] [--si-nev 128] ..."""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v6 as OPS   # noqa: E402  (read-only: eigh_csr, gershgorin, device_memory, reset_device_peak)

CD = OPS.CD
RD = OPS.RD
T0 = time.time()


def log(m):
    print("[%7.1f s] %s" % (time.time() - T0, m), flush=True)


def _np(x):
    """a numpy float64 array from a numpy array, a list or a (CPU or device) torch tensor."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, dtype=float)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if torch.is_tensor(o):
        return jsonable(o.detach().cpu().numpy())
    return o


def key_of(tgt):
    """an npz key fragment of a target name ("shift:0.98" -> "shift_0p98")."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", tgt.replace("-", "m").replace(".", "p"))


# ------------------------------------------------------------------------------------------------- the exterior symbol
def symbol_branches(N, v, kr, m, k):
    """omega_+-(k) = +-N sqrt(sin^2 k + kr^2 + m^2) + (v/2) sin 2k and d omega_+-/dk = +-N sin k cos k / sqrt(...) + v cos 2k at a site with
    lapse N, flow v, kr = kappa/R and mass m (page_graft_pullback_v2.split_weight's symbol with the mass added); every argument broadcasts.
    Returns (omega_+, omega_-, v_g+, v_g-); the outgoing direction is d omega/dk > 0 (page_ops_v6.velocity_csr's symbol)."""
    s = np.sin(k); c = np.cos(k)
    Ek = np.sqrt(s * s + kr * kr + m * m)
    pos = Ek > 0.0
    sc = np.where(pos, s * c / np.where(pos, Ek, 1.0), 0.0)
    s2 = np.sin(2.0 * k); c2 = np.cos(2.0 * k)
    om_p = N * Ek + 0.5 * v * s2
    om_m = -N * Ek + 0.5 * v * s2
    vg_p = N * sc + v * c2
    vg_m = -N * sc + v * c2
    return om_p, om_m, vg_p, vg_m


def band_edges_sites(N, v, R, kappa, m, kgrid, site_chunk=512):
    """the symbol's band edges per site on a k grid of kgrid points over [-pi, pi): the upper branch's maximum (the local band top) and minimum,
    the lower branch's minimum (the local band bottom) and maximum; the global extremes with their radii; the edges at the chain's end."""
    N = _np(N); v = _np(v); R = _np(R)
    n = len(R)
    k = np.linspace(-math.pi, math.pi, int(kgrid), endpoint=False)[None, :]
    top = np.empty(n); bot = np.empty(n); upmin = np.empty(n); lomax = np.empty(n)
    for i0 in range(0, n, site_chunk):
        i1 = min(n, i0 + site_chunk)
        om_p, om_m, _, _ = symbol_branches(N[i0:i1, None], v[i0:i1, None], (float(kappa) / R[i0:i1])[:, None], m, k)
        top[i0:i1] = om_p.max(axis=1); upmin[i0:i1] = om_p.min(axis=1)
        bot[i0:i1] = om_m.min(axis=1); lomax[i0:i1] = om_m.max(axis=1)
    it = int(np.argmax(top)); ib = int(np.argmin(bot))
    return {"top_site": top, "bottom_site": bot, "upper_branch_min_site": upmin, "lower_branch_max_site": lomax,
            "global_top": float(top[it]), "global_top_R": float(R[it]), "global_bottom": float(bot[ib]), "global_bottom_R": float(R[ib]),
            "end_top": float(top[-1]), "end_bottom": float(bot[-1]), "end_R": float(R[-1]), "kgrid": int(kgrid)}


def outgoing_velocities(N, v, kr, m, E, k):
    """at one site: for every signed energy E the outgoing roots (d omega/dk > 0) of omega_+-(k) = E on the k grid (linear interpolation between
    grid points, the velocity from the analytic derivative at the interpolated k); returns (the slowest, the fastest, the count); the slowest is
    +inf and the count 0 where no outgoing root exists (the energy does not propagate outward at that site)."""
    k = np.asarray(k, float)
    kk = np.concatenate([k, [k[0] + 2.0 * math.pi]])
    om_p, om_m, _, _ = symbol_branches(N, v, kr, m, kk)
    E = np.asarray(E, float)
    nE = E.shape[0]
    vmin = np.full(nE, np.inf); vmax = np.full(nE, -np.inf); nroots = np.zeros(nE, dtype=int)
    for branch, om in ((1, om_p), (-1, om_m)):
        f = om[None, :] - E[:, None]
        f0 = f[:, :-1]; f1 = f[:, 1:]
        sc = (f0 * f1 < 0.0) | ((f0 == 0.0) & (f1 != 0.0))
        iE, j = np.nonzero(sc)
        if iE.size == 0:
            continue
        a0 = f0[iE, j]; a1 = f1[iE, j]
        t = a0 / (a0 - a1)
        ks = kk[j] + t * (kk[j + 1] - kk[j])
        _, _, vgp, vgm = symbol_branches(N, v, kr, m, ks)
        vg = vgp if branch > 0 else vgm
        outg = vg > 0.0
        if outg.any():
            np.minimum.at(vmin, iE[outg], vg[outg]); np.maximum.at(vmax, iE[outg], vg[outg]); np.add.at(nroots, iE[outg], 1)
    return vmin, vmax, nroots


def exit_scales(N, v, R, kappa, m, E, R_from, stride, kgrid, vg_min=1e-12):
    """the exit of an outgoing carrier at every signed energy E from R_from to the chain's end along the sampled exterior (the sites with
    R > R_from, every `stride`, the last site always): the path time t_exit = sum dR / v_min(E; R) (v_min the slowest outgoing root at the
    sampled site; +inf where the energy stops propagating before the end, the turning radius R_turn the last sampled radius where it did), the
    slowest and the fastest outgoing root at the chain's end (v_end_min, v_end_max; inf / -inf without a root) and at the first sampled site,
    the slowest root along the path, the box time t_box = (R_end - R_from)/v_end_min for the energies that exit."""
    N = _np(N); v = _np(v); R = _np(R); E = np.asarray(E, float)
    nE = E.shape[0]
    out = {"E": E, "R_from": float(R_from), "R_end": float(R[-1]), "stride": int(stride), "kgrid": int(kgrid)}
    idx = np.nonzero(R > R_from)[0]
    if idx.size == 0 or nE == 0:
        inf = np.full(nE, np.inf)
        out.update({"t_exit": inf.copy(), "t_box": inf.copy(), "R_turn": np.full(nE, float(R_from)), "v_end_min": inf.copy(),
                    "v_end_max": -inf.copy(), "v_first_min": inf.copy(), "v_first_max": -inf.copy(), "v_path_min": inf.copy(),
                    "n_roots_end": np.zeros(nE, dtype=int), "R_sampled": np.zeros(0), "exits": np.zeros(nE, dtype=bool)})
        return out
    sel = idx[:: max(1, int(stride))]
    if sel[-1] != idx[-1]:
        sel = np.append(sel, idx[-1])
    k = np.linspace(-math.pi, math.pi, int(kgrid), endpoint=False)
    t_path = np.zeros(nE); alive = np.ones(nE, dtype=bool); R_turn = np.full(nE, float(R_from)); v_path_min = np.full(nE, np.inf)
    v_first_min = v_first_max = None
    vmin = vmax = nr = None
    for p, i in enumerate(sel):
        vmin, vmax, nr = outgoing_velocities(N[i], v[i], float(kappa) / R[i], m, E, k)
        if p == 0:
            v_first_min, v_first_max = vmin.copy(), vmax.copy()
        prop = alive & (nr > 0) & (vmin > vg_min)
        alive = prop
        R_turn[prop] = R[i]
        if p + 1 < len(sel):
            dR = R[sel[p + 1]] - R[i]
            t_path[prop] += dR / vmin[prop]
        v_path_min[prop] = np.minimum(v_path_min[prop], vmin[prop])
    exits = alive.copy()
    t_exit = np.where(exits, t_path, np.inf)
    t_box = np.where(exits, (R[-1] - R_from) / np.where(exits, vmin, 1.0), np.inf)
    out.update({"t_exit": t_exit, "t_box": t_box, "R_turn": R_turn, "v_end_min": vmin, "v_end_max": vmax, "v_first_min": v_first_min,
                "v_first_max": v_first_max, "v_path_min": v_path_min, "n_roots_end": nr, "R_sampled": R[sel], "exits": exits})
    return out


def turning_radii(e, top_site, bottom_site, R, R_from, chunk=1024):
    """for every eigenvalue e: the largest exterior radius (R > R_from) where the local band still contains e (top_site >= e for e >= 0,
    bottom_site <= e for e < 0); R_from where none (the energy does not propagate on the exterior); the chain's end where it propagates there."""
    e = np.asarray(e, float); R = np.asarray(R, float)
    out = np.full(e.shape[0], float(R_from))
    ext = np.nonzero(R > R_from)[0]
    if ext.size == 0 or e.size == 0:
        return out
    Rext = R[ext]; top = np.asarray(top_site, float)[ext]; bot = np.asarray(bottom_site, float)[ext]
    for i0 in range(0, e.shape[0], chunk):
        ee = e[i0:i0 + chunk][:, None]
        ok = np.where(ee >= 0.0, top[None, :] >= ee, bot[None, :] <= ee)
        has = ok.any(axis=1)
        last = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
        out[i0:i0 + chunk] = np.where(has, Rext[last], float(R_from))
    return out


def envelope_remaining(E_pos, E_neg, t_pos, t_neg, t_grid):
    """E_ext(t) = the extended content (per-bin sums, both halves) whose exit time exceeds t, on the grid t_grid; and the trapped part (infinite
    exit time, which never leaves)."""
    E_pos = np.asarray(E_pos, float); E_neg = np.asarray(E_neg, float); t_pos = np.asarray(t_pos, float); t_neg = np.asarray(t_neg, float)
    rem = np.array([float(E_pos[t_pos > t].sum() + E_neg[t_neg > t].sum()) for t in np.asarray(t_grid, float)])
    trapped = float(E_pos[~np.isfinite(t_pos)].sum() + E_neg[~np.isfinite(t_neg)].sum())
    return rem, trapped


def bin_edges(d_fine, x_fine_max, d_coarse, x_max):
    """the |e|/T_H bin edges: d_fine wide from 0 to x_fine_max, then d_coarse wide until the last edge exceeds x_max."""
    fine = np.arange(0.0, float(x_fine_max) + 1e-9, float(d_fine))
    if fine[-1] < x_fine_max - 1e-9:
        fine = np.append(fine, float(x_fine_max))
    last = float(fine[-1])
    n_c = int(math.ceil(max(float(x_max) - last, 0.0) / float(d_coarse))) + 1
    coarse = last + float(d_coarse) * np.arange(1, n_c + 1)
    return np.concatenate([fine, coarse])


# ----------------------------------------------------------------------------------------------------- the dense part
def dense_spectrum(h_csr, device, eig_device=None):
    """h's dense eigendecomposition (page_ops_v6.eigh_csr) on eig_device (default device), V moved to device; returns (e ascending, V, seconds,
    the measured device peak in GB on eig_device or None on a CPU)."""
    eig_device = device if eig_device is None else eig_device
    OPS.reset_device_peak(eig_device)
    t0 = time.time()
    e, V = OPS.eigh_csr(h_csr, eig_device)
    sec = time.time() - t0
    mem = OPS.device_memory(eig_device)
    peak = mem["peak_GB"] if mem else None
    if eig_device != device:
        V = V.to(device); e = e.to(device)
        if eig_device.type == "cuda":
            torch.cuda.empty_cache()
    return e, V, sec, peak


def weights_inside(V, nrows, chunk):
    """sum over the first nrows rows of |V_{i alpha}|^2 per column alpha (the weight inside a radius: rows 2 n + a of the sites inside)."""
    n2 = int(V.shape[1])
    w = torch.zeros(n2, dtype=RD, device=V.device)
    if nrows <= 0:
        return w
    nrows = min(int(nrows), int(V.shape[0]))
    for j0 in range(0, n2, chunk):
        blk = V[:nrows, j0:j0 + chunk]
        w[j0:j0 + chunk] = (blk.real * blk.real + blk.imag * blk.imag).sum(dim=0)
    return w


def occupations(V, Phi, chunk):
    """n_alpha = sum_j |(V^dagger Phi)_{alpha j}|^2, the eigenmodes' occupations in C = Phi Phi^dagger (chunked over Phi's columns)."""
    n2 = int(V.shape[0])
    n = torch.zeros(n2, dtype=RD, device=V.device)
    VH = V.mH
    for j0 in range(0, int(Phi.shape[1]), chunk):
        X = VH @ Phi[:, j0:j0 + chunk]
        n += (X.real * X.real + X.imag * X.imag).sum(dim=1)
        del X
    return n


def gram_stats(Phi, chunk):
    """Tr C = ||Phi||_F^2, Tr C^2 = ||Phi^dagger Phi||_F^2 and the Gram defect ||Phi^dagger Phi - 1||_F, chunked over Phi's columns."""
    nocc = int(Phi.shape[1])
    trC = 0.0; trC2 = 0.0; def2 = 0.0
    PhiH = Phi.mH
    for j0 in range(0, nocc, chunk):
        j1 = min(nocc, j0 + chunk)
        Pc = Phi[:, j0:j1]
        trC += float((Pc.real * Pc.real + Pc.imag * Pc.imag).sum())
        G = PhiH @ Pc
        trC2 += float((G.real * G.real + G.imag * G.imag).sum())
        G[j0:j1, :].diagonal().sub_(1.0)
        def2 += float((G.real * G.real + G.imag * G.imag).sum())
        del G
    return {"n_occ": nocc, "trace_C": trC, "trace_C2": trC2, "gram_defect_F": math.sqrt(max(def2, 0.0))}


def split_energy(e, d, w, theta):
    """the split of sum e d by the weight w: bound (w >= theta) against extended, by sign; the particle-number content of each part."""
    e = np.asarray(e, float); d = np.asarray(d, float); w = np.asarray(w, float)
    bound = w >= theta
    pos = e > 0.0; neg = e < 0.0
    ed = e * d
    E_tot = float(ed.sum())
    out = {"theta": float(theta), "n_bound": int(bound.sum()), "n_extended": int((~bound).sum()),
           "E_bound": float(ed[bound].sum()), "E_extended": float(ed[~bound].sum()),
           "E_bound_particles": float(ed[bound & pos].sum()), "E_bound_holes": float(ed[bound & neg].sum()),
           "E_extended_particles": float(ed[~bound & pos].sum()), "E_extended_holes": float(ed[~bound & neg].sum()),
           "N_bound_particles": float(d[bound & pos].sum()), "N_bound_holes": float(-d[bound & neg].sum()),
           "N_extended_particles": float(d[~bound & pos].sum()), "N_extended_holes": float(-d[~bound & neg].sum()),
           "E_total": E_tot, "bound_share": (float(ed[bound].sum()) / E_tot) if E_tot != 0.0 else None}
    return out


def bin_arrays(e, d, w, theta, edges_TH, T_H):
    """per bin of |e|/T_H and per class (bound: w >= theta; extended) and half (pos: e > 0, neg: e < 0): the counts, the occupation sums
    sum d, the Killing-energy sums sum e d and their density per T_H."""
    e = np.asarray(e, float); d = np.asarray(d, float); w = np.asarray(w, float)
    x = np.abs(e) / T_H
    ed = e * d
    width = np.diff(edges_TH)
    out = {"edges_TH": np.asarray(edges_TH, float), "centres_TH": 0.5 * (edges_TH[:-1] + edges_TH[1:])}
    for cn, cm in (("bound", w >= theta), ("extended", w < theta)):
        for sn, sm in (("pos", e > 0.0), ("neg", e < 0.0)):
            msk = cm & sm
            cnt, _ = np.histogram(x[msk], edges_TH)
            dsum, _ = np.histogram(x[msk], edges_TH, weights=d[msk])
            Esum, _ = np.histogram(x[msk], edges_TH, weights=ed[msk])
            out["count_%s_%s" % (cn, sn)] = cnt.astype(float)
            out["d_%s_%s" % (cn, sn)] = dsum
            out["E_%s_%s" % (cn, sn)] = Esum
            out["Edens_%s_%s" % (cn, sn)] = Esum / width
    return out


# ------------------------------------------------------------------------------------------------------ the long chain
def shift_invert_window(A_csc, sigma, nev, tol, ncv, maxiter):
    """the nev eigenpairs of the complex Hermitian CSC A nearest sigma by scipy's shift-invert Arnoldi (eigsh, which 'LM' on (A - sigma)^-1, the
    factor a SuperLU LU in the natural ordering: the banded chain fills within its band); returns (eigenvalues ascending, vectors, the residuals
    ||A x - e x||, seconds, converged); a non-convergence keeps ARPACK's converged pairs and sets converged False."""
    n = int(A_csc.shape[0]); k = int(min(int(nev), n - 2))
    t0 = time.time()
    M = (A_csc - float(sigma) * sp.identity(n, dtype=np.complex128, format="csc")).tocsc()
    lu = spla.splu(M, permc_spec="NATURAL")
    OPinv = spla.LinearOperator(M.shape, matvec=lu.solve, dtype=np.complex128)
    kw = dict(k=k, sigma=float(sigma), which="LM", OPinv=OPinv, tol=float(tol))
    if ncv:
        kw["ncv"] = int(ncv)
    if maxiter:
        kw["maxiter"] = int(maxiter)
    converged = True
    try:
        vals, vecs = spla.eigsh(A_csc, **kw)
    except spla.ArpackNoConvergence as err:
        vals, vecs = err.eigenvalues, err.eigenvectors
        converged = False
    vals = np.asarray(vals).real.astype(float)
    vecs = np.asarray(vecs)
    if vals.size:
        order = np.argsort(vals); vals = vals[order]; vecs = np.ascontiguousarray(vecs[:, order])
        res = np.linalg.norm(A_csc @ vecs - vecs * vals[None, :], axis=0)
    else:
        vecs = np.zeros((n, 0), dtype=np.complex128); res = np.zeros(0)
    del lu, OPinv, M
    return vals, vecs, res, time.time() - t0, converged


def window_for_target(A_csc, tgt, bound, be, nev, offset, k0, tol, tol_coarse, ncv, maxiter, passes_max, res_tol, first="symbol"):
    """the window of a target: "top"/"bottom": the band's extreme, estimated from the symbol's global edge (first "symbol") or from a coarse
    shift-invert pass of k0 pairs at the Gershgorin bound (first "gershgorin"), then the nev pairs nearest the estimate +- offset, re-shifted
    to the found extreme while eigenvalues lie beyond the shift or a residual exceeds res_tol (at most passes_max passes); "ext_top"/
    "ext_bottom": the nev pairs nearest the exterior band edge at the chain's end; "shift:<x>": nearest x. Returns (vals, vecs, res, info)."""
    info = {"target": tgt, "passes": []}

    def run(sigma, k, tl):
        vals, vecs, res, sec, conv = shift_invert_window(A_csc, sigma, k, tl, ncv, maxiter)
        info["passes"].append({"sigma": float(sigma), "k": int(k), "tol": float(tl), "n_returned": int(vals.size), "converged": bool(conv),
                               "seconds": sec, "e_min": float(vals.min()) if vals.size else None, "e_max": float(vals.max()) if vals.size else None,
                               "residual_max": float(res.max()) if res.size else None})
        return vals, vecs, res

    if tgt in ("top", "bottom"):
        sgn = 1.0 if tgt == "top" else -1.0
        if first == "gershgorin":
            v0, _, _ = run(sgn * float(bound) * (1.0 + 1e-6), k0, tol_coarse)
            est = float(v0.max() if sgn > 0 else v0.min()) if v0.size else sgn * float(bound)
        else:
            est = float(be["global_top"] if sgn > 0 else be["global_bottom"])
        vals = np.zeros(0); vecs = np.zeros((A_csc.shape[0], 0), dtype=np.complex128); res = np.zeros(0); sigma = est + sgn * offset
        for _ in range(max(1, int(passes_max))):
            sigma = est + sgn * offset
            vals, vecs, res = run(sigma, nev, tol)
            if vals.size == 0:
                break
            ext = float(vals.max() if sgn > 0 else vals.min())
            beyond = bool((vals > sigma).any()) if sgn > 0 else bool((vals < sigma).any())
            bad = bool(res.size and res.max() > res_tol)
            est = ext
            if not (beyond or bad):
                break
        info.update({"sigma": float(sigma), "extreme": (float(vals.max() if sgn > 0 else vals.min()) if vals.size else None),
                     "estimate_rule": first, "n_passes": len(info["passes"])})
        return vals, vecs, res, info
    if tgt == "ext_top":
        sigma = float(be["end_top"])
    elif tgt == "ext_bottom":
        sigma = float(be["end_bottom"])
    elif tgt.startswith("shift:"):
        sigma = float(tgt.split(":", 1)[1])
    else:
        raise ValueError("unknown shift-invert target %r" % tgt)
    vals, vecs, res = run(sigma, nev, tol)
    info.update({"sigma": float(sigma), "n_passes": 1})
    return vals, vecs, res, info


def graft_occupations(vecs, n2s, Phi, Vneg, chunk):
    """d^g_alpha = ||Phi^dagger psi_s||^2 - ||V_-^dagger psi_s||^2 for the window vectors psi (columns of vecs, numpy, the long chain's rows),
    psi_s their first n2s rows (the short chain): psi^dagger E D_S E^dagger psi with D_S = Phi Phi^dagger - V_- V_-^dagger (the graft's
    deviation). Returns (d^g, the Phi part, the vacuum part) as numpy arrays."""
    dev = Phi.device
    Ws = torch.as_tensor(np.ascontiguousarray(np.asarray(vecs)[:n2s, :]), dtype=CD, device=dev)
    nev = int(Ws.shape[1])
    X = Phi.mH @ Ws
    a_ = (X.real * X.real + X.imag * X.imag).sum(dim=0)
    del X
    b_ = torch.zeros(nev, dtype=RD, device=dev)
    for j0 in range(0, int(Vneg.shape[1]), chunk):
        Y = Vneg[:, j0:j0 + chunk].mH @ Ws
        b_ += (Y.real * Y.real + Y.imag * Y.imag).sum(dim=0)
        del Y
    del Ws
    return (a_ - b_).cpu().numpy(), a_.cpu().numpy(), b_.cpu().numpy()


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="page_hold_split_v1: the held excitation's spectral split (the module docstring has the definitions)")
    ap.add_argument("--record", required=True, help="the frozen record's json (collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json)")
    ap.add_argument("--fields", required=True, help="the record's fields npz")
    ap.add_argument("--fields-key", default="applied")
    ap.add_argument("--init-state", required=True, help="the hold's stored state (v6bA_s4_state_at_015.000.pt or _030.000.pt)")
    ap.add_argument("--freeze-at-rs", type=float, required=True, help="the hold's freeze clock in r_s (15 or 30); the record nearest it builds h")
    ap.add_argument("--nsites-short", type=int, default=8192, help="the stored chain's sites (the hold's --nsites)")
    ap.add_argument("--nsites-long", type=int, default=69992, help="the long grafted operator's sites (page_graft_pullback_v2's short + append)")
    ap.add_argument("--channels", default="1,2,3", help="channels kappa; one absent from the stored state is reported and skipped")
    ap.add_argument("--R-cut-rs", type=float, default=24.0)
    ap.add_argument("--R-collar-rs", type=float, default=16.0)
    ap.add_argument("--theta", type=float, default=0.99, help="bound: weight inside R_cut at least this")
    ap.add_argument("--theta-beside", default="0.9,0.999", help="thresholds printed beside the main one (comma separated)")
    ap.add_argument("--mass", type=float, default=0.0, help="the field mass in the exterior symbol (0: the massless records)")
    ap.add_argument("--zero-tol", type=float, default=1e-12, help="|e| below this counts as a zero mode (printed; P_- takes e < 0)")
    ap.add_argument("--bin-TH", type=float, default=0.5, help="the fine bin width in T_H")
    ap.add_argument("--bin-max-TH", type=float, default=40.0, help="the fine bins' upper end in T_H")
    ap.add_argument("--bin-coarse-TH", type=float, default=10.0, help="the coarse bin width in T_H beyond --bin-max-TH up to the band's top")
    ap.add_argument("--exit-kgrid", type=int, default=4096, help="k points of the symbol over [-pi, pi) (band edges and roots)")
    ap.add_argument("--exit-stride", type=int, default=64, help="every this many exterior sites along the exit path (the last site always)")
    ap.add_argument("--exit-vg-min", type=float, default=1e-12, help="an outgoing root slower than this counts as not propagating")
    ap.add_argument("--envelope-max-rs", type=float, default=3000.0)
    ap.add_argument("--envelope-step-rs", type=float, default=5.0)
    ap.add_argument("--long", type=int, default=1, help="1: the long chain's part; 0: the stored chain only")
    ap.add_argument("--si-nev", type=int, default=128, help="eigenpairs per shift-invert target")
    ap.add_argument("--si-targets", default="top,bottom,ext_top,ext_bottom",
                    help="top, bottom: the band's extremes; ext_top, ext_bottom: the exterior band edges at the long chain's end")
    ap.add_argument("--si-shifts", default="", help="extra shifts in lattice energy units (comma separated), a window of --si-nev each")
    ap.add_argument("--si-first-estimate", default="symbol", choices=["symbol", "gershgorin"],
                    help="the band extreme's first estimate: the symbol's global edge, or a coarse shift-invert pass at the Gershgorin bound")
    ap.add_argument("--si-offset", type=float, default=1e-5, help="the shift's distance beyond the extreme's estimate (lattice energy units)")
    ap.add_argument("--si-k0", type=int, default=8, help="eigenpairs of the coarse Gershgorin pass")
    ap.add_argument("--si-tol", type=float, default=1e-10)
    ap.add_argument("--si-tol-coarse", type=float, default=1e-6)
    ap.add_argument("--si-res-tol", type=float, default=1e-8, help="a window with a residual above this is re-shifted (passes capped)")
    ap.add_argument("--si-ncv", type=int, default=0, help="ARPACK ncv (0: scipy's default max(2 nev + 1, 20))")
    ap.add_argument("--si-maxiter", type=int, default=400, help="ARPACK restart cycles; a non-convergence keeps the converged pairs and flags")
    ap.add_argument("--si-passes-max", type=int, default=3)
    ap.add_argument("--match-tol", type=float, default=1e-8, help="the bound modes' eigenvalue match tolerance between the chains (printed)")
    ap.add_argument("--save-vectors", type=int, default=0, help="1: the long windows' eigenvectors in the npz (2.2 MB per vector at 69992 sites)")
    ap.add_argument("--device", default="cuda:0", help="the contractions' device (V, Phi)")
    ap.add_argument("--eig-device", default="", help="the dense eigendecomposition's device (default --device; cpu keeps the GPU peak at V + Phi)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1024, help="columns per contraction chunk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", type=int, default=1, help="1: channels finished in <tag>_split.json with their npz are kept")
    a = ap.parse_args()

    import page_graft_pullback_v2 as G2   # noqa: E402  (read-only: the hold's operator construction, PK.History and PK.ChannelOps)
    PK = G2.PK
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    eig_dev = torch.device(a.eig_device) if a.eig_device else dev
    for d_ in (dev, eig_dev):
        if d_.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("device %s requested but cuda is not available; use --device cpu --eig-device cpu" % d_)
    os.makedirs(a.out, exist_ok=True)
    pre = os.path.join(a.out, a.tag); jp = pre + "_split.json"
    ns = int(a.nsites_short); n2s = 2 * ns; nl = int(a.nsites_long)
    gb = n2s * n2s * 16.0 / 1e9
    log("memory figure: the dense h and its eigenvectors %.2f GB each (%d x %d complex128) on %s; the eigendecomposition's own workspace is the "
        "library's (LAPACK zheevd's formula about %.1f GB more; cuSOLVER's not estimated here) and the measured device peak is printed after it; "
        "then V (%.2f GB) on %s beside the stored columns (%.2f GB per 8192 columns) and a %d-column contraction chunk (%.2f GB)" % (
            gb, n2s, n2s, eig_dev, 2.0 * gb, gb, dev, n2s * 8192 * 16.0 / 1e9, a.chunk, n2s * a.chunk * 16.0 / 1e9))
    # the record and its frozen fields (as the hold and the graft code read them)
    Hs = PK.History(a.record, a.fields, ns, None, a.fields_key)
    rs = float(Hs.rs_nom or 2.0 * Hs.g * Hs.E[0])
    kf = int(np.argmin(np.abs(Hs.T - a.freeze_at_rs * rs))); tf = float(Hs.T[kf])
    T_H = 1.0 / (4.0 * math.pi * rs)
    Rs = _np(Hs.R)
    Ns_raw, vs_raw = Hs.fields(kf)
    Ns = _np(Ns_raw); vs = _np(vs_raw)
    if len(Rs) != ns or not np.all(np.diff(Rs) > 0):
        raise SystemExit("the record's chain has %d sites (expected %d) or non-increasing radii" % (len(Rs), ns))
    Rc = a.R_cut_rs * rs; Rcol = a.R_collar_rs * rs
    nc = int(np.count_nonzero(Rs <= Rc)); ncol = int(np.count_nonzero(Rs <= Rcol))
    try:
        E_kf = float(Hs.E[kf])
    except Exception:
        E_kf = None
    Hl = None
    if a.long:
        if nl <= ns:
            raise SystemExit("--nsites-long %d must exceed --nsites-short %d" % (nl, ns))
        Hl = PK.History(a.record, a.fields, nl, None, a.fields_key)
        Rl = _np(Hl.R)
        Nl_raw, vl_raw = Hl.fields(kf)
        Nl = _np(Nl_raw); vl = _np(vl_raw)
        if len(Rl) != nl or not np.allclose(Rl[:ns], Rs):
            raise SystemExit("the long chain's first %d radii differ from the stored chain's (or %d sites were not built)" % (ns, nl))
    log("record kf %d at T/r_s %.6f (requested %.3f; |dT|/r_s %.2e); r_s %.6f, T_H %.6e; R_cut %.1f (%d sites inside), collar %.1f (%d sites); "
        "stored chain %d sites, long chain %s sites" % (kf, tf / rs, a.freeze_at_rs, abs(tf - a.freeze_at_rs * rs) / rs, rs, T_H, Rc, nc, Rcol, ncol,
                                                        ns, nl if a.long else "-"))
    # the hold's state
    st = torch.load(a.init_state, map_location="cpu", weights_only=False)
    if not isinstance(st, dict) or "channels" not in st:
        raise SystemExit("the state %s has no 'channels' entry (keys %s)" % (a.init_state, list(st.keys()) if isinstance(st, dict) else type(st)))
    chan_store = st["channels"]
    state_meta = {k_: (v_ if isinstance(v_, (int, float, str, bool)) else str(type(v_))) for k_, v_ in st.items() if k_ != "channels"}
    log("state %s: top-level keys %s; channels present %s" % (a.init_state, list(st.keys()), sorted(chan_store.keys(), key=lambda s: int(s) if str(s).isdigit() else 0)))
    thetas = []
    for th in [a.theta] + [float(x) for x in a.theta_beside.split(",") if x.strip()]:
        if th not in thetas:
            thetas.append(th)
    chans = [int(x) for x in a.channels.split(",") if x.strip()]
    targets = [t.strip() for t in a.si_targets.split(",") if t.strip()] + ["shift:%s" % s.strip() for s in a.si_shifts.split(",") if s.strip()]
    # the results (resumed or fresh)
    res = None
    if a.resume and os.path.exists(jp):
        try:
            res = json.load(open(jp))
            log("resume: %s read; channels done %s" % (jp, [k_ for k_, c_ in res.get("channels", {}).items() if c_.get("done")]))
        except Exception as ex:
            log("resume: %s unreadable (%s); starting afresh" % (jp, ex)); res = None
    if res is None:
        res = {"settings": vars(a), "channels": {}}
    res["settings_last_run"] = vars(a)
    res["label"] = ("the held excitation's spectral split: D = C - P_-(h) in the held operator's eigenbasis, E_inv = Tr[h D] split by the "
                    "eigenvectors' weight inside R_cut (bound at weight >= theta, else extended), the extended part's Killing-energy density per "
                    "bin and its exit scale from the exterior symbol, the chain-length dependence by shift-invert windows on the long grafted operator")
    res["record"] = {"record": a.record, "fields": a.fields, "fields_key": a.fields_key, "kf": kf, "T_freeze": tf, "T_freeze_over_rs": tf / rs,
                     "rs": rs, "T_H": T_H, "E_record_at_kf": E_kf, "R_cut": Rc, "R_collar": Rcol, "sites_inside_cut": nc, "sites_inside_collar": ncol,
                     "nsites_short": ns, "n2_short": n2s, "nsites_long": nl if a.long else None, "R_end_short": float(Rs[-1]),
                     "R_end_long": float(Rl[-1]) if a.long else None}
    res["hold"] = {"init_state": a.init_state, "state_keys": list(st.keys()), "state_meta": state_meta, "channels_in_state": sorted(chan_store.keys()),
                   "vacuum_rule": "P_- = 1(h < 0): zero modes empty (|e| < zero_tol counted)", "thetas": thetas}
    kgrid_arr = np.linspace(-math.pi, math.pi, int(a.exit_kgrid), endpoint=False)
    t_grid = np.arange(0.0, a.envelope_max_rs + 1e-9, a.envelope_step_rs) * rs
    for kap in chans:
        key = str(kap)
        npz_k = pre + "_k%d_split.npz" % kap
        prev = res["channels"].get(key)
        if a.resume and prev and prev.get("done") and os.path.exists(npz_k):
            log("kappa %d: finished in a previous run (%s); kept" % (kap, npz_k))
            continue
        if key not in chan_store or not isinstance(chan_store[key], dict) or "Phi" not in chan_store[key]:
            res["channels"][key] = {"done": False, "missing": True,
                                    "note": "no stored occupied columns for kappa %d in %s (channels present: %s)" % (kap, a.init_state, sorted(chan_store.keys()))}
            log("kappa %d: NOT in the stored state (channels present %s); skipped" % (kap, sorted(chan_store.keys())))
            json.dump(jsonable(res), open(jp, "w"), indent=1)
            continue
        t_c = time.time()
        arrays = {}
        # (1) the held operator and its dense eigendecomposition
        ops_s = PK.ChannelOps(kap, Hs.R, dev)
        hs = ops_s.h_csr(Ns_raw, vs_raw).tocsr()
        if hs.shape[0] != n2s:
            raise SystemExit("kappa %d: h has %d rows, expected %d" % (kap, hs.shape[0], n2s))
        wk = float(getattr(ops_s, "w", 16.0 * kap))
        bound_s = OPS.gershgorin(hs)
        log("kappa %d (weight %g): h built (%d rows, %d nonzeros, Gershgorin bound %.6f); dense eigendecomposition on %s ..." % (
            kap, wk, n2s, hs.nnz, bound_s, eig_dev))
        e_t, V, eig_s, peak = dense_spectrum(hs, dev, eig_dev)
        e = e_t.cpu().numpy().astype(float)
        del e_t
        neg = e < 0.0; pos = e > 0.0
        nneg = int(neg.sum()); nzero = int((np.abs(e) < a.zero_tol).sum())
        e_max_neg = float(e[neg].max()) if nneg else None
        e_min_pos = float(e[pos].min()) if pos.any() else None
        log("kappa %d: eigh %.1f s (measured device peak %s GB); e_min %.12g, e_max %.12g; %d negative, %d zero modes (|e| < %g); Killing gap (%s, %s)" % (
            kap, eig_s, ("%.2f" % peak) if peak is not None else "-", e[0], e[-1], nneg, nzero, a.zero_tol,
            ("%.6e" % e_max_neg) if e_max_neg is not None else "-", ("%.6e" % e_min_pos) if e_min_pos is not None else "-"))
        # (3) the weights
        w_cut = weights_inside(V, 2 * nc, a.chunk).cpu().numpy()
        w_col = weights_inside(V, 2 * ncol, a.chunk).cpu().numpy()
        # (2) the stored columns and D's diagonal in the eigenbasis
        Phi = chan_store[key]["Phi"].to(device=dev, dtype=CD)
        if Phi.shape[0] != n2s:
            raise SystemExit("kappa %d: the stored columns have %d rows, expected %d" % (kap, Phi.shape[0], n2s))
        gram = gram_stats(Phi, a.chunk)
        n_alpha = occupations(V, Phi, a.chunk).cpu().numpy()
        d = n_alpha - neg.astype(float)
        ed = e * d
        E_inv = float(ed.sum()); E_p = float(ed[pos].sum()); E_h = float(ed[neg].sum())
        N_plus = float(n_alpha[pos].sum()); N_minus = float((1.0 - n_alpha[neg]).sum())
        trCP = float(n_alpha[neg].sum())
        D_F = math.sqrt(max(gram["trace_C2"] - 2.0 * trCP + nneg, 0.0))
        log("kappa %d: %d stored columns, Tr C %.9f, Gram defect %.3e; Tr D %.9f (N_+ %.6f, N_- %.6f), ||D||_F %.6f; E_inv %.9e (particles %.6e, holes %.6e) = %.4f T_H%s" % (
            kap, gram["n_occ"], gram["trace_C"], gram["gram_defect_F"], float(d.sum()), N_plus, N_minus, D_F, E_inv, E_p, E_h, E_inv / T_H,
            ("; E_inv/E %.6e" % (E_inv / E_kf)) if E_kf else ""))
        splits = {str(th): split_energy(e, d, w_cut, th) for th in thetas}
        splits_col = {str(th): split_energy(e, d, w_col, th) for th in thetas}
        for th in thetas:
            s_ = splits[str(th)]; c_ = splits_col[str(th)]
            log("kappa %d theta %g: %d bound eigenvectors (R_cut weight >= theta), E_bound %.6e (particles %.4e, holes %.4e), E_extended %.6e (%.4e, %.4e), "
                "bound share %s; by the collar weight: %d bound, E_bound %.6e, E_extended %.6e" % (
                    kap, th, s_["n_bound"], s_["E_bound"], s_["E_bound_particles"], s_["E_bound_holes"], s_["E_extended"], s_["E_extended_particles"],
                    s_["E_extended_holes"], ("%.6f" % s_["bound_share"]) if s_["bound_share"] is not None else "-", c_["n_bound"], c_["E_bound"], c_["E_extended"]))
        # (4) the bins, the band edges, the turning radii, the exit scales, the envelope
        xmax = max(bound_s, float(np.abs(e).max())) / T_H
        edges = bin_edges(a.bin_TH, a.bin_max_TH, a.bin_coarse_TH, xmax)
        nb = len(edges) - 1
        bins = bin_arrays(e, d, w_cut, a.theta, edges, T_H)
        centres_TH = bins["centres_TH"]
        E_signed = np.concatenate([centres_TH * T_H, -centres_TH * T_H])
        be_s = band_edges_sites(Ns, vs, Rs, kap, a.mass, a.exit_kgrid)
        Rt_s = turning_radii(e, be_s["top_site"], be_s["bottom_site"], Rs, Rc)
        ex_s = exit_scales(Ns, vs, Rs, kap, a.mass, E_signed, Rc, a.exit_stride, a.exit_kgrid, a.exit_vg_min)
        env_s, trapped_s = envelope_remaining(bins["E_extended_pos"], bins["E_extended_neg"], ex_s["t_exit"][:nb], ex_s["t_exit"][nb:], t_grid)
        bound_mask = w_cut >= a.theta
        exits_s = Rt_s >= Rs[-1]
        E_ext_exits_s = float(ed[~bound_mask & exits_s].sum()); E_ext_turn_s = float(ed[~bound_mask & ~exits_s].sum())
        log("kappa %d: band edges from the symbol: global top %.9f at R %.0f, global bottom %.9f at R %.0f, at the chain's end (R %.0f) top %.9f bottom %.9f; "
            "dense extremes e_max %.9f (%.2e below the symbol's top), e_min %.9f (%.2e above its bottom); Gershgorin %.6f" % (
                kap, be_s["global_top"], be_s["global_top_R"], be_s["global_bottom"], be_s["global_bottom_R"], be_s["end_R"], be_s["end_top"], be_s["end_bottom"],
                e[-1], be_s["global_top"] - e[-1], e[0], e[0] - be_s["global_bottom"], bound_s))
        log("kappa %d: extended part (theta %g) %.6e: propagates to the stored chain's end %.6e, turns before it (above the exterior band at its end) %.6e; "
            "envelope: trapped %.6e, remaining at 100/300/1000 r_s %s" % (
                kap, a.theta, splits[str(a.theta)]["E_extended"], E_ext_exits_s, E_ext_turn_s, trapped_s,
                " / ".join("%.4e" % env_s[int(np.argmin(np.abs(t_grid / rs - tt)))] for tt in (100.0, 300.0, 1000.0) if tt <= a.envelope_max_rs)))
        groups = [(0.0, 5.0), (5.0, 25.0), (25.0, 40.0), (40.0, np.inf)]
        group_rows = []
        for lo_, hi_ in groups:
            sb = (centres_TH >= lo_) & (centres_TH < hi_)
            Eext_g = float(bins["E_extended_pos"][sb].sum() + bins["E_extended_neg"][sb].sum())
            Ebnd_g = float(bins["E_bound_pos"][sb].sum() + bins["E_bound_neg"][sb].sum())
            te = np.concatenate([ex_s["t_exit"][:nb][sb], ex_s["t_exit"][nb:][sb]]); fin = te[np.isfinite(te)]
            row = {"lo_TH": lo_, "hi_TH": hi_, "E_extended": Eext_g, "E_bound": Ebnd_g, "n_bins": int(sb.sum()),
                   "t_exit_short_min_rs": float(fin.min() / rs) if fin.size else None, "t_exit_short_max_rs": float(fin.max() / rs) if fin.size else None,
                   "bins_trapped_short": int((~np.isfinite(te)).sum())}
            group_rows.append(row)
            log("kappa %d bins [%g, %s) T_H: E_extended %.4e, E_bound %.4e; exit on the stored chain %s to %s r_s (%d of %d signed bins trapped)" % (
                kap, lo_, ("%g" % hi_) if np.isfinite(hi_) else "top", Eext_g, Ebnd_g,
                ("%.1f" % row["t_exit_short_min_rs"]) if fin.size else "-", ("%.1f" % row["t_exit_short_max_rs"]) if fin.size else "-", row["bins_trapped_short"], len(te)))
        arrays.update({"e": e, "n_alpha": n_alpha, "d": d, "w_cut": w_cut, "w_col": w_col, "R_turn_short_fields": Rt_s, "bin_edges_TH": edges,
                       "bin_centres_TH": centres_TH, "envelope_t_grid": t_grid, "envelope_short": env_s,
                       "band_top_site_short": be_s["top_site"], "band_bottom_site_short": be_s["bottom_site"],
                       "band_upper_branch_min_site_short": be_s["upper_branch_min_site"], "band_lower_branch_max_site_short": be_s["lower_branch_max_site"]})
        for k_, v_ in bins.items():
            if k_ not in ("edges_TH", "centres_TH"):
                arrays["bin_" + k_] = v_
        for k_, v_ in ex_s.items():
            if isinstance(v_, np.ndarray):
                arrays["exit_short_" + k_] = v_
        ch = {"done": False, "w": wk, "n2": n2s, "n_occ": gram["n_occ"], "trace_C": gram["trace_C"], "trace_C2": gram["trace_C2"],
              "gram_defect_F": gram["gram_defect_F"], "trace_D": float(d.sum()), "D_F": D_F, "eigh_seconds": eig_s, "eigh_device": str(eig_dev),
              "device_peak_GB": peak, "gershgorin": bound_s, "e_min": float(e[0]), "e_max": float(e[-1]), "n_neg": nneg, "n_zero": nzero,
              "e_max_neg": e_max_neg, "e_min_pos": e_min_pos, "E_inv": E_inv, "E_inv_particles": E_p, "E_inv_holes": E_h, "E_inv_over_TH": E_inv / T_H,
              "E_inv_over_E_record": (E_inv / E_kf) if E_kf else None, "N_plus": N_plus, "N_minus": N_minus,
              "weights": {"cut_min": float(w_cut.min()), "cut_max": float(w_cut.max()), "cut_mean": float(w_cut.mean()), "collar_min": float(w_col.min()),
                          "collar_max": float(w_col.max()), "n_cut_ge_theta": int(bound_mask.sum()), "n_collar_ge_theta": int((w_col >= a.theta).sum())},
              "splits_Rcut": splits, "splits_collar": splits_col,
              "band_edges_short": {k_: v_ for k_, v_ in be_s.items() if not isinstance(v_, np.ndarray)},
              "extended_by_exit_short": {"propagates_to_end": E_ext_exits_s, "turns_before_end": E_ext_turn_s},
              "envelope_short": {"trapped": trapped_s, "t_grid_rs": [0.0, a.envelope_step_rs, a.envelope_max_rs],
                                 "remaining_at_rs": {str(tt): float(env_s[int(np.argmin(np.abs(t_grid / rs - tt)))]) for tt in (0.0, 100.0, 300.0, 1000.0, 3000.0) if tt <= a.envelope_max_rs}},
              "bins": {"n_bins": nb, "edges_TH_first_last": [float(edges[0]), float(edges[-1])], "groups": group_rows}}
        # (6) the long chain
        if a.long:
            t_l = time.time()
            ops_l = PK.ChannelOps(kap, Hl.R, torch.device("cpu"))
            hl = ops_l.h_csr(Nl_raw, vl_raw).tocsr()
            if hl.shape[0] != 2 * nl:
                raise SystemExit("kappa %d: the long h has %d rows, expected %d" % (kap, hl.shape[0], 2 * nl))
            bound_l = OPS.gershgorin(hl)
            dblk = float(abs(hl[:n2s, :n2s] - hs).max())
            A = hl.tocsc()
            be_l = band_edges_sites(Nl, vl, Rl, kap, a.mass, a.exit_kgrid)
            Rt_l = turning_radii(e, be_l["top_site"], be_l["bottom_site"], Rl, Rc)
            ex_l = exit_scales(Nl, vl, Rl, kap, a.mass, E_signed, Rc, a.exit_stride, a.exit_kgrid, a.exit_vg_min)
            env_l, trapped_l = envelope_remaining(bins["E_extended_pos"], bins["E_extended_neg"], ex_l["t_exit"][:nb], ex_l["t_exit"][nb:], t_grid)
            exits_l = Rt_l >= Rl[-1]
            E_ext_exits_l = float(ed[~bound_mask & exits_l].sum()); E_ext_turn_l = float(ed[~bound_mask & ~exits_l].sum())
            log("kappa %d long chain (%d sites): h built (Gershgorin %.6f; its short block against the stored h: max |difference| %.3e, the join's two sites); "
                "symbol edges: global top %.9f, bottom %.9f, at its end (R %.0f) top %.9f bottom %.9f; extended part: propagates to the long end %.6e, turns before it %.6e; "
                "envelope trapped %.6e, remaining at 300/1000/3000 r_s %s (%.0f s)" % (
                    kap, nl, bound_l, dblk, be_l["global_top"], be_l["global_bottom"], be_l["end_R"], be_l["end_top"], be_l["end_bottom"], E_ext_exits_l, E_ext_turn_l, trapped_l,
                    " / ".join("%.4e" % env_l[int(np.argmin(np.abs(t_grid / rs - tt)))] for tt in (300.0, 1000.0, 3000.0) if tt <= a.envelope_max_rs), time.time() - t_l))
            ncv = a.si_ncv if a.si_ncv > 0 else None
            Vneg = V[:, :nneg]
            windows = {}
            for tgt in targets:
                t_w = time.time()
                vals, vecs, resid, info = window_for_target(A, tgt, bound_l, be_l, nev=a.si_nev, offset=a.si_offset, k0=a.si_k0, tol=a.si_tol,
                                                            tol_coarse=a.si_tol_coarse, ncv=ncv, maxiter=a.si_maxiter, passes_max=a.si_passes_max,
                                                            res_tol=a.si_res_tol, first=a.si_first_estimate)
                kk_ = key_of(tgt)
                if vals.size == 0:
                    windows[tgt] = {"info": info, "n": 0, "note": "no eigenpair returned (ARPACK did not converge); raise --si-maxiter or lower --si-nev"}
                    log("kappa %d window %s: NO eigenpair returned (%.0f s)" % (kap, tgt, time.time() - t_w))
                    continue
                wcut_w = (np.abs(vecs[:2 * nc]) ** 2).sum(axis=0)
                wcol_w = (np.abs(vecs[:2 * ncol]) ** 2).sum(axis=0)
                wshort_w = (np.abs(vecs[:n2s]) ** 2).sum(axis=0)
                dg, dg_phi, dg_vac = graft_occupations(vecs, n2s, Phi, Vneg, a.chunk)
                Rt_w = turning_radii(vals, be_l["top_site"], be_l["bottom_site"], Rl, Rc)
                vend_min, vend_max, nr_end = outgoing_velocities(Nl[-1], vl[-1], float(kap) / Rl[-1], a.mass, vals, kgrid_arr)
                spl_w = {str(th): split_energy(vals, dg, wcut_w, th) for th in thetas}
                lo, hi = float(vals.min()), float(vals.max())
                ins = (e >= lo - a.match_tol) & (e <= hi + a.match_tol)
                spl_s = {str(th): split_energy(e[ins], d[ins], w_cut[ins], th) for th in thetas}
                bnd = wcut_w >= a.theta
                match = {"n": int(bnd.sum())}
                if bnd.any():
                    jn = np.argmin(np.abs(e[None, :] - vals[bnd][:, None]), axis=1)
                    de = e[jn] - vals[bnd]; dd = d[jn] - dg[bnd]; dw = w_cut[jn] - wcut_w[bnd]
                    match.update({"max_abs_de": float(np.abs(de).max()), "max_abs_dd": float(np.abs(dd).max()), "max_abs_dw": float(np.abs(dw).max()),
                                  "within_match_tol": int((np.abs(de) <= a.match_tol).sum()),
                                  "E_bound_window_long": float((vals[bnd] * dg[bnd]).sum()), "E_bound_matched_short": float((e[jn] * d[jn]).sum())})
                    arrays["long_%s_match_j" % kk_] = jn; arrays["long_%s_match_de" % kk_] = de; arrays["long_%s_match_dd" % kk_] = dd; arrays["long_%s_match_dw" % kk_] = dw
                windows[tgt] = {"info": info, "n": int(vals.size), "e_min": lo, "e_max": hi, "residual_max": float(resid.max()), "residual_median": float(np.median(resid)),
                                "weights_cut_min": float(wcut_w.min()), "weights_cut_max": float(wcut_w.max()), "weights_short_min": float(wshort_w.min()),
                                "weights_short_max": float(wshort_w.max()), "n_bound_theta": int(bnd.sum()), "n_turning_before_end": int((Rt_w < Rl[-1]).sum()),
                                "sum_e_dg": float((vals * dg).sum()), "sum_dg": float(dg.sum()), "splits_long_window": spl_w,
                                "short_in_range": {"n": int(ins.sum()), "sum_e_d": float(ed[ins].sum()), "splits": spl_s}, "bound_mode_match": match,
                                "seconds": time.time() - t_w}
                arrays["long_%s_e" % kk_] = vals; arrays["long_%s_residual" % kk_] = resid; arrays["long_%s_wcut" % kk_] = wcut_w; arrays["long_%s_wcol" % kk_] = wcol_w
                arrays["long_%s_wshort" % kk_] = wshort_w; arrays["long_%s_dg" % kk_] = dg; arrays["long_%s_dg_phi_part" % kk_] = dg_phi
                arrays["long_%s_dg_vac_part" % kk_] = dg_vac; arrays["long_%s_R_turn" % kk_] = Rt_w; arrays["long_%s_v_end_min" % kk_] = vend_min
                arrays["long_%s_v_end_max" % kk_] = vend_max; arrays["long_%s_short_in_range_index" % kk_] = np.nonzero(ins)[0]
                if a.save_vectors:
                    arrays["long_%s_vecs" % kk_] = vecs
                sw = spl_w[str(a.theta)]; ss = spl_s[str(a.theta)]
                log("kappa %d window %s: %d eigenpairs in [%.10f, %.10f] (sigma %.10f, %d passes, residual max %.2e, converged %s); weights inside R_cut %.3e..%.3f, "
                    "on the short rows %.3e..%.3f; %d bound at theta %g, %d turn before the long end; sum e d^g %.6e = bound %.6e + extended %.6e; the stored chain in "
                    "that range: %d eigenvalues, sum e d %.6e = bound %.6e + extended %.6e; bound modes matched %d (max |de| %s, max |dd| %s) (%.0f s)" % (
                        kap, tgt, vals.size, lo, hi, info["sigma"], info.get("n_passes", 1), float(resid.max()), all(p_["converged"] for p_ in info["passes"]),
                        float(wcut_w.min()), float(wcut_w.max()), float(wshort_w.min()), float(wshort_w.max()), int(bnd.sum()), a.theta, int((Rt_w < Rl[-1]).sum()),
                        float((vals * dg).sum()), sw["E_bound"], sw["E_extended"], int(ins.sum()), float(ed[ins].sum()), ss["E_bound"], ss["E_extended"], match["n"],
                        ("%.2e" % match["max_abs_de"]) if "max_abs_de" in match else "-", ("%.2e" % match["max_abs_dd"]) if "max_abs_dd" in match else "-", time.time() - t_w))
                del vecs
            e_max_long = windows.get("top", {}).get("info", {}).get("extreme") if "top" in windows else None
            e_min_long = windows.get("bottom", {}).get("info", {}).get("extreme") if "bottom" in windows else None
            ch["long"] = {"nsites": nl, "n2": 2 * nl, "gershgorin": bound_l, "short_block_vs_short_h_max_abs": dblk,
                          "band_edges_long": {k_: v_ for k_, v_ in be_l.items() if not isinstance(v_, np.ndarray)},
                          "e_max_long_from_window": e_max_long, "e_min_long_from_window": e_min_long,
                          "e_max_long_minus_short": (e_max_long - float(e[-1])) if e_max_long is not None else None,
                          "e_min_long_minus_short": (e_min_long - float(e[0])) if e_min_long is not None else None,
                          "extended_by_exit_long": {"propagates_to_end": E_ext_exits_l, "turns_before_end": E_ext_turn_l},
                          "envelope_long": {"trapped": trapped_l, "remaining_at_rs": {str(tt): float(env_l[int(np.argmin(np.abs(t_grid / rs - tt)))]) for tt in (0.0, 300.0, 1000.0, 3000.0) if tt <= a.envelope_max_rs}},
                          "windows": windows, "seconds": time.time() - t_l}
            arrays.update({"R_turn_long_fields": Rt_l, "envelope_long": env_l, "band_top_site_long": be_l["top_site"], "band_bottom_site_long": be_l["bottom_site"],
                           "band_upper_branch_min_site_long": be_l["upper_branch_min_site"], "band_lower_branch_max_site_long": be_l["lower_branch_max_site"]})
            for k_, v_ in ex_l.items():
                if isinstance(v_, np.ndarray):
                    arrays["exit_long_" + k_] = v_
            log("kappa %d long chain: e_max %s (%s against the stored chain's), e_min %s (%s)" % (
                kap, ("%.12f" % e_max_long) if e_max_long is not None else "-", ("%.3e" % (e_max_long - e[-1])) if e_max_long is not None else "-",
                ("%.12f" % e_min_long) if e_min_long is not None else "-", ("%.3e" % (e_min_long - e[0])) if e_min_long is not None else "-"))
            del A, hl, Vneg
        # the export lines for the second_route lane
        exp = {"hold": a.tag, "init_state": os.path.basename(a.init_state), "freeze_T_over_rs": tf / rs, "record_kf": kf, "rs": rs, "T_H": T_H, "kappa": kap,
               "n2": n2s, "e_min": float(e[0]), "e_max": float(e[-1]), "n_neg": nneg, "n_zero": nzero, "e_max_neg": e_max_neg, "e_min_pos": e_min_pos,
               "gershgorin": bound_s, "n_occ": gram["n_occ"], "trace_C": gram["trace_C"], "gram_defect_F": gram["gram_defect_F"], "D_F": D_F,
               "E_inv": E_inv, "E_inv_particles": E_p, "E_inv_holes": E_h, "N_plus": N_plus, "N_minus": N_minus, "R_cut": Rc,
               "splits": {str(th): {"n_bound": splits[str(th)]["n_bound"], "E_bound": splits[str(th)]["E_bound"], "E_extended": splits[str(th)]["E_extended"]} for th in thetas},
               "band_top_global": be_s["global_top"], "band_bottom_global": be_s["global_bottom"], "band_top_end": be_s["end_top"], "band_bottom_end": be_s["end_bottom"],
               "R_end": float(Rs[-1])}
        log("EXPORT " + json.dumps(jsonable(exp), sort_keys=True))
        if a.long:
            expl = {"hold": a.tag, "kappa": kap, "nsites_long": nl, "e_max_long": e_max_long, "e_min_long": e_min_long, "gershgorin_long": bound_l,
                    "band_top_end_long": be_l["end_top"], "band_bottom_end_long": be_l["end_bottom"],
                    "windows": {t_: {"n": w_.get("n"), "e_min": w_.get("e_min"), "e_max": w_.get("e_max"), "residual_max": w_.get("residual_max"),
                                     "E_bound": (w_.get("splits_long_window") or {}).get(str(a.theta), {}).get("E_bound"),
                                     "E_extended": (w_.get("splits_long_window") or {}).get(str(a.theta), {}).get("E_extended"),
                                     "short_E_bound": ((w_.get("short_in_range") or {}).get("splits") or {}).get(str(a.theta), {}).get("E_bound"),
                                     "short_E_extended": ((w_.get("short_in_range") or {}).get("splits") or {}).get(str(a.theta), {}).get("E_extended")}
                                for t_, w_ in windows.items()}}
            log("EXPORT-LONG " + json.dumps(jsonable(expl), sort_keys=True))
        ch["export"] = exp
        ch["seconds"] = time.time() - t_c
        ch["done"] = True
        np.savez_compressed(npz_k, **arrays)
        res["channels"][key] = ch
        json.dump(jsonable(res), open(jp, "w"), indent=1)
        log("kappa %d: written %s and %s (%.0f s)" % (kap, npz_k, jp, time.time() - t_c))
        del V, Phi, hs, arrays
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    # the weighted summary over the channels computed and the combined npz
    done = {k_: c_ for k_, c_ in res["channels"].items() if c_.get("done")}
    if done:
        summ = {"channels": sorted(done.keys(), key=int), "note": "weighted (w = 16 kappa) sums over the channels computed here only"}
        for th in thetas:
            s_ = str(th)
            summ["theta_%s" % s_] = {"E_inv_weighted": sum(c_["w"] * c_["E_inv"] for c_ in done.values()),
                                     "E_bound_weighted": sum(c_["w"] * c_["splits_Rcut"][s_]["E_bound"] for c_ in done.values()),
                                     "E_extended_weighted": sum(c_["w"] * c_["splits_Rcut"][s_]["E_extended"] for c_ in done.values())}
        res["summary"] = summ
        s0 = summ["theta_%s" % str(a.theta)]
        log("summary over channels %s (weights 16 kappa): E_inv %.6e = bound %.6e + extended %.6e at theta %g" % (
            summ["channels"], s0["E_inv_weighted"], s0["E_bound_weighted"], s0["E_extended_weighted"], a.theta))
    json.dump(jsonable(res), open(jp, "w"), indent=1)
    comb = {}
    for key, c_ in res["channels"].items():
        p = pre + "_k%s_split.npz" % key
        if c_.get("done") and os.path.exists(p):
            with np.load(p) as z:
                for k_ in z.files:
                    comb["k%s_%s" % (key, k_)] = z[k_]
    np.savez_compressed(pre + "_split.npz", **comb)
    log("written %s_split.json and %s_split.npz (%d arrays)" % (pre, pre, len(comb)))


if __name__ == "__main__":
    main()
