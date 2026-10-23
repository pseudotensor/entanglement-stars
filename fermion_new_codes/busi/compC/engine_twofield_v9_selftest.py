#!/usr/bin/env python
"""engine_twofield_v9_selftest.py (the busi lane, Computation C; the transport ruling's G0', busi/notes/C64_transport_ruling_v1.md 4.4): the
checks of engine_twofield_v9.py against engine_twofield_v8.py on manufactured small systems on the CPU (no --gpu) or on the device C5.backend
(--gpu) gives, plus the gate's tools. Version 8's self-test (engine_twofield_v8_selftest.py, beside this file) is imported for its fixture
(make_fixture, record_solve, fixture_args), its comparisons (deep_diff, report, compare_runs, load_run) and its endtoend arguments (e2e_args).
Subcommands:
  all | fixture | marks | endtoend | compare-runs       (all = fixture, and marks when --v8 and --v9 are given)
  fixture (CPU, a few minutes): on version 8's manufactured record (128 sites; 3 heavy channels at m 0.2; the light channels 1,2 sourcing and 3 the
          guard; the manufactured field solve; the production's Newton loop):
          (a) record bitwise: (a-i) version 9 --cut-transport record implicit against version 8 (newton_record_v32: the fields, sources, histories,
              every cache entry, the counters); (a-ii) lagged: the record solve and lagged_close_v8 (the transported references, the lag block) against
              version 8's; (a-iii) sector against record on a path with one pocket and no sub-floor crossing: light_continue_v36 in both modes on the
              same eigendecomposition (the reference columns, the mask, mu, gap, n_plus, the certificate's pieces, Cholesky calls, d_min_certified and
              piece list, the cut check) and continue_ref_v9 on the heavy channels (the Weyl route's result and entry), bitwise;
          (c) one exact crossing (a manufactured path h(s) = h0 + s D on the light channel, D the pair at the cut decoupled and crossing exactly at
              s = 1/2, with its conjugate mirror): the walk detects the sub-floor gap, assigns the pair with overlap 1 to rounding, the inversion
              count changes by 1, the transported reference equals the diabatic continuation (the eigenvectors are constant along this path: the
              previous reference's subspace, residual at most 1e-10); the record-scale convention's result on the same path printed beside;
          (e) the dived mode rising back on a second record (the path reversed from the end of (c)): the inversion count returns to 0, the anchor set to one;
          (b) two pockets, no crossing: the inverted state of (c) continued along a path on which no boundary closes (a small field change): every piece
              certified, the inversion count unchanged, the anchor mask equal to the index mask (the direct spectral projector onto the occupied
              intervals; residual 0) ;
          (d) an avoided crossing of gap g at the boundary within one record: g = 1e-3 certified and adiabatic (the branch swap: the selected level's
              weight on the previous reference below 1/2, the cut check's crossing_levels 2, recorded not acted on), g = 1e-8 the diabatic one of (c)
              (the overlap near 1, the residual to the previous reference the admixture g/(2 Delta)); the piece-refinement convergence (step_scale 1/2:
              twice the pieces) in both cases;
          the heavy: (c) and (b) on the heavy channel's manufactured paths (heavy_sector_walk_v9: the crossing detected by the boundary gap, the pair
              assigned with overlap 1, the inversion count; the no-crossing path certified with the count unchanged);
  marks --v8 V8 --v9 V9: every line difflib finds added or changed in v9 against v8, outside the module docstring, carries the mark "v9 sector"
          (blank lines skipped); nothing deleted without a replacement (4.4(f));
  endtoend --v8 V8 --v9 V9 [--gpu G] [--nsites 1024] [--rs 32] [--records 4] [--work DIR]: 5 engine runs from T = 0 with the r_s/a 32 sigma/a 4 member's
          argument set (version 8 implicit; version 9 record implicit; version 8 lagged; version 9 record lagged; version 9 sector lagged): version 9
          record against version 8 bitwise in both modes (every record number but wall clocks, every npz row); version 9 sector against version 9
          record on this certified path bitwise after the sector-only keys are stripped (sector, cut_transport, the labels, the anchors/crossings
          counters), the sector routes per record printed;
  compare-runs A/TAG B/TAG [--after T] [--strip-sector]: version 8's compare, optionally with the sector-only keys stripped.
Each check prints PASS or FAIL; the exit code is the number of failures."""
import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
for _d in (os.environ.get("BUSI_ENGINE_DIR", "<internal>/engine"), os.environ.get("BUSI_PAGE_DIR", "<internal>/page")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)

import numpy as np
import scipy.sparse as sp

MARK = "v9 sector"
STRIP_SECTOR = {"sector", "cut_transport", "cut_transport_v9", "anchors", "crossings", "record_label", "label", "seconds_sector"}
STRIP_INFO = {"sector", "anchors", "seconds", "seconds_sector", "_sigma", "_mask"}


def strip_keys(obj, keys):
    """a nested structure without the keys (dicts) at any depth."""
    if isinstance(obj, dict):
        return {k: strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [strip_keys(v, keys) for v in obj]
    return obj


def resid(P, W):
    """the subspace residual max |W - P (P^dagger W)| of the columns W against the span of the orthonormal columns P."""
    return float((W - P @ (P.conj().T @ W)).abs().max())


def args9(S8, n, kappa_sea, reference_solve="implicit", cut="record"):
    a = S8.fixture_args(n, kappa_sea)
    a.reference_solve = reference_solve; a.cut_transport = cut
    return a


# ------------------------------------------------------------------------------------------------------- the manufactured paths
class ManufacturedLightPath:
    """h(s) = h0 + s D on the light channel (dense on the device): h_host(s) the host CSR, eig(s) the sectorized eigendecomposition (the trial's at s = 1 once set)."""

    def __init__(self, V, h0, D, sgn_dev, tol_block):
        self.V = V; self.h0 = h0; self.D = D; self.sgn_dev = sgn_dev; self.tol_block = tol_block; self.eigh_n = 0; self.trial = None

    def h_host(self, s):
        return sp.csr_matrix((self.h0 + s * self.D).cpu().numpy())

    def eig(self, s):
        if s >= 1.0 and self.trial is not None:
            return self.trial
        torch = self.V.BK["torch"]
        e, U = torch.linalg.eigh(self.h0 + s * self.D); self.eigh_n += 1
        sigma, _n, _imp = self.V.light_sectorize_v36(e, U, self.sgn_dev, self.tol_block)
        return e.cpu().numpy().astype(float), sigma.cpu().numpy().astype(float), U


class ManufacturedHeavyPath:
    """h(s) = h0 + s D on a heavy channel (dense on the device): h_host(s), eigvals(s), eig(s)."""

    def __init__(self, V, h0, D):
        self.V = V; self.h0 = h0; self.D = D; self.eigh_n = 0; self.eigvals_n = 0; self.trial = None

    def h_host(self, s):
        return sp.csr_matrix((self.h0 + s * self.D).cpu().numpy())

    def eigvals(self, s):
        torch = self.V.BK["torch"]; self.eigvals_n += 1
        return np.sort(torch.linalg.eigvalsh(self.h0 + s * self.D).cpu().numpy().astype(float))

    def eig(self, s):
        if s >= 1.0 and self.trial is not None:
            return self.trial
        torch = self.V.BK["torch"]
        e, U = torch.linalg.eigh(self.h0 + s * self.D); self.eigh_n += 1
        return e.cpu().numpy().astype(float), U


def light_pair_path(V, chL, N0, v0, sgn_dev, tol_block, g=0.0):
    """the light channel's h0 at (N0, v0) and D = X - X*, X = c (|u_a><u_a| - |u_b><u_b|) + (g/2)(|u_a><u_b| + |u_b><u_a|) on the S = +1 sector's pair at the
    cut (u_a the highest occupied, u_b the lowest empty + level, c = e_b - e_a: an exact crossing at s = 1/2 when g = 0, an avoided one of gap g
    otherwise), X* the conjugate on the S = -1 mirror pair (D purely imaginary and Hermitian, commuting with S to the sector impurity).
    Returns (path, e0, sig0, occ0 (the mu S selection), ref0, n_sea, mu0, c)."""
    torch = V.BK["torch"]
    h0 = V.clocked_gpu(chL, N0, v0)
    path = ManufacturedLightPath(V, h0, torch.zeros_like(h0), sgn_dev, tol_block)
    e0, sig0, U0 = path.eig(0.0)
    n_plus0 = int(np.sum((sig0 > 0.0) & (e0 < 0.0)))
    mu0, _gp = V.light_mu_midgap_v36(e0, sig0, n_plus0)
    occ0, _gap0 = V.light_select_v36(e0, sig0, mu0)
    order0, ep0 = V.light_plus_order_v9(e0, sig0)
    ia, ib = int(order0[n_plus0 - 1]), int(order0[n_plus0])
    ua = U0[:, ia].clone(); ub = U0[:, ib].clone(); c = float(ep0[n_plus0] - ep0[n_plus0 - 1])
    X = c * (torch.outer(ua, ua.conj()) - torch.outer(ub, ub.conj()))
    if g > 0.0:
        X = X + (0.5 * g) * (torch.outer(ua, ub.conj()) + torch.outer(ub, ua.conj()))
    path.D = X - X.conj()
    ref0 = U0[:, torch.as_tensor(occ0, device=U0.device)].clone()
    return path, e0, sig0, occ0, ref0, int(occ0.sum()), mu0, c


def light_index_mask(V, e_np, sig, occp):
    """the index mask: the + sector's occupation occp by sorted index and the S = -1 sector its conjugate mirror (the - level at -e_+ occupied where the + level is empty)."""
    order, _ep = V.light_plus_order_v9(e_np, sig)
    mask = np.zeros(len(e_np), bool); mask[order] = occp
    selm = np.where(np.asarray(sig, float) <= 0.0)[0]; orderm = selm[np.argsort(np.asarray(e_np, float)[selm], kind="stable")]
    n = len(orderm)
    for k in range(n):
        mask[orderm[k]] = not occp[n - 1 - k]
    return mask


def report_ok(name, ok, extra=""):
    print("%s %s%s" % ("PASS" if ok else "FAIL", name, (": " + extra) if extra else ""), flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------- the fixture
def test_fixture(V8, V9, S8, gpu, n=128, kappa_sea=3):
    V9.C5.backend(gpu)
    V35 = V9.V35
    V35.V36["fixes"] = True
    fails = 0
    a_i8 = args9(S8, n, kappa_sea); a_i9 = args9(S8, n, kappa_sea, "implicit", "record")
    FX = S8.make_fixture(V9, a_i9)
    S8.sparse_probe_and_patch([V8, V9], FX.chs[0])
    stub = SimpleNamespace(charge=lambda info, g: float(info["E"]), __name__="fixture_fields")
    saved = (V8.FLD, V9.FLD, V35.FLD)
    V8.FLD = V9.FLD = V35.FLD = stub
    torch = V9.BK["torch"]
    try:
        print("fixture: %d sites, heavy kappa 1..%d at m %.1f, light %s sourcing and %s the guard, device %s" % (n, kappa_sea, a_i9.m, a_i9.light_kappas, a_i9.light_guard_kappas, V9.BK["dev"]), flush=True)
        # (a-i) record implicit bitwise
        r8 = S8.record_solve(V8, a_i8, FX); r9 = S8.record_solve(V9, a_i9, FX)
        diffs = []
        for k_ in ("Nn", "vn", "N_app", "v_app", "E", "k_it", "converged", "it_hist", "solve32"):
            S8.deep_diff(r8[k_], r9[k_], k_, diffs)
        S8.deep_diff(list(r8["srcs"]), list(r9["srcs"]), "sources", diffs)
        for key in r8["cache"]:
            S8.deep_diff(r8["cache"][key], r9["cache"][key], "cache[%s]" % (key,), diffs)
        S8.deep_diff(r8["TM"], r9["TM"], "TM", diffs); S8.deep_diff(r8["TML"], r9["TML"], "TML", diffs); S8.deep_diff(r8["JV"], r9["JV"], "JV", diffs)
        fails += S8.report("(a-i) version 9 --cut-transport record implicit against version 8 (newton_record_v32 on the manufactured record)", diffs,
                           " (%d evaluations, kept %.2e; %s)" % (r9["solve32"]["evaluations"], r9["solve32"]["kept_residual"], S8.routes(r9["cache"], FX.chs, FX.chl)))
        # (a-ii) record lagged bitwise: the solve, the transport, the lag block
        a_L8 = args9(S8, n, kappa_sea, "lagged"); a_L9 = args9(S8, n, kappa_sea, "lagged", "record")
        rL8 = S8.record_solve(V8, a_L8, FX)
        lag8 = V8.lagged_close_v8(a_L8, FX.chs, FX.chl, FX.st, rL8["cache"], rL8["srcs"], rL8["N_app"], rL8["v_app"], rL8["sources"], FX.do_solve, FX.sgn_dev, FX.S_host, FX.R)
        rL9 = S8.record_solve(V9, a_L9, FX)
        lag9 = V9.lagged_close_v8(a_L9, FX.chs, FX.chl, FX.st, rL9["cache"], rL9["srcs"], rL9["N_app"], rL9["v_app"], rL9["sources"], FX.do_solve, FX.sgn_dev, FX.S_host, FX.R)
        diffs = []
        for k_ in ("Nn", "vn", "N_app", "v_app", "E", "k_it", "converged", "it_hist", "solve32"):
            S8.deep_diff(rL8[k_], rL9[k_], k_, diffs)
        for key in rL8["cache"]:
            S8.deep_diff(rL8["cache"][key], rL9["cache"][key], "cache[%s]" % (key,), diffs)
        S8.deep_diff(lag8, lag9, "lag", diffs)
        fails += S8.report("(a-ii) version 9 --cut-transport record lagged against version 8 (the solve, the transported references, the lag block)", diffs,
                           " (lag residual %.3e, E_cons - E %+.3e)" % (lag9["residual"], lag9["E_cons"] - rL9["E"]))
        # (a-iii) sector against record on one pocket, no sub-floor crossing: the light continuation and the heavy continuation
        a_S9 = args9(S8, n, kappa_sea, "lagged", "sector")
        N_app, v_app = rL9["N_app"], rL9["v_app"]
        diffs = []; notes = []
        for chL in FX.chl:
            s_ = FX.st[chL.key]
            e_r, U_r = V9.eigT(V9.clocked_gpu(chL, N_app, v_app)); e_s, U_s = V9.eigT(V9.clocked_gpu(chL, N_app, v_app))
            ref_r, ci_r, m_r = V9.light_continue_v36(a_L9, chL, s_["window"], (N_app, v_app), e_r, U_r, s_["n_sea"], FX.sgn_dev, FX.S_host)
            ref_s, ci_s, m_s = V9.light_continue_v36(a_S9, chL, s_["window"], (N_app, v_app), e_s, U_s, s_["n_sea"], FX.sgn_dev, FX.S_host)
            S8.deep_diff(ref_r, ref_s, "light %d ref" % chL.kappa, diffs); S8.deep_diff(np.asarray(m_r, bool), np.asarray(m_s, bool), "light %d mask" % chL.kappa, diffs)
            S8.deep_diff(U_r, U_s, "light %d U (rotated)" % chL.kappa, diffs)
            S8.deep_diff(strip_keys(ci_r, STRIP_INFO), strip_keys(ci_s, STRIP_INFO), "light %d cinfo" % chL.kappa, diffs)
            sec = ci_s.get("sector") or {}
            notes.append("light %d: route %s, anchors %s, pieces %s, Cholesky %s, crossings %d" % (chL.kappa, sec.get("route"), sec.get("anchors_end"), sec.get("pieces"), sec.get("cholesky"), len(sec.get("crossings") or [])))
        for ch in FX.chs:
            s_ = FX.st[ch.kappa]
            e_r, U_r = V9.eigT(V9.clocked_gpu(ch, N_app, v_app))
            ref_r, ci_r, m_r = V9.continue_ref_v9(a_L9, ch, s_["window"], (N_app, v_app), e_r, U_r, s_["n_sea"])
            ref_s, ci_s, m_s = V9.continue_ref_v9(a_S9, ch, s_["window"], (N_app, v_app), e_r, U_r, s_["n_sea"])
            S8.deep_diff(ref_r, ref_s, "heavy %d ref" % ch.kappa, diffs); S8.deep_diff(np.asarray(m_r, bool), np.asarray(m_s, bool), "heavy %d mask" % ch.kappa, diffs)
            S8.deep_diff(strip_keys(ci_r, STRIP_INFO), strip_keys(ci_s, STRIP_INFO), "heavy %d cinfo" % ch.kappa, diffs)
            notes.append("heavy %d: route %s" % (ch.kappa, (ci_s.get("sector") or {}).get("route")))
        fails += S8.report("(a-iii) sector against record on a path with one pocket and no sub-floor crossing (the mu S walk and the Weyl route reproduced)", diffs, " (%s)" % "; ".join(notes))
    finally:
        V8.FLD, V9.FLD, V35.FLD = saved
    # the manufactured paths on the light channel kappa 1 and the heavy channel kappa 1
    chL = FX.chl[0]; chH = FX.chs[0]; N0, v0 = FX.f_ref; tol_block = a_S9.light_tol_block; d_min = a_S9.light_d_min
    OPS3 = V9.OPS3; CERT = V9.CERT
    try:
        # (c) one exact crossing
        path, e0, sig0, occ0, ref0, n_sea, mu0, c = light_pair_path(V9, chL, N0, v0, FX.sgn_dev, tol_block, g=0.0)
        trial = path.eig(1.0); path.trial = trial; e1, sig1, U1 = trial
        cert, sec, anchors_end, occp_end = V9.light_sector_walk_v9(path, FX.S_host, e0, sig0, occ0, trial, n_sea, d_min, 4096, tol_block, mu_prev=mu0)
        mask_c, gap_c = V9.light_select_sector_v9(e1, sig1, anchors_end) if cert["certified"] else (None, None)
        cr = sec.get("crossings") or []
        ok = (cert["certified"] and len(cr) == 1 and cr[0]["overlap"] is not None and cr[0]["overlap"] >= 1.0 - 1e-6 and cr[0]["inversion_delta"] == 1 and cr[0]["block"] == 2 and not cr[0]["flagged"]
              and sec["anchors_end"] == 3 and sec["inversions_plus_end"] == 1 and int(mask_c.sum()) == n_sea)
        r_c = resid(U1[:, torch.as_tensor(mask_c, device=U1.device)], ref0) if ok else float("nan")
        ok = ok and r_c <= 1e-10 and V9.light_sector_inversions_v36(e1, sig1, mask_c) == 2
        ref_rs, info_rs, mask_rs = OPS3.assign_max_overlap_v3(ref0, e1, U1, n_sea, tol_block)
        fails += report_ok("(c) one exact crossing (the light channel, the pair at the cut decoupled, c = %.3e, crossing at s = 1/2)" % c, ok,
                           "certified %s, crossings %d%s, anchors %s, inversions (+ sector) %s, the residual of the transported reference against the previous (the diabatic continuation) %.1e, pieces %s, Cholesky %s, eigh %s; the record-scale convention beside: wmin %s inversions %s, the same mask %s" % (
                               cert["certified"], len(cr), (" (s* %.6f, overlap %.8f, delta %d)" % (cr[0]["s"], cr[0]["overlap"], cr[0]["inversion_delta"])) if cr else "", sec.get("anchors_end"), sec.get("inversions_plus_end"),
                               r_c, cert.get("pieces"), cert.get("cholesky_calls"), sec.get("eigh"), S8.deep_diff and ("%.4f" % info_rs["min_occupied_weight"]), info_rs.get("inversions"), bool(np.array_equal(np.asarray(mask_rs, bool), mask_c)) if mask_c is not None else None))
        ref_c = U1[:, torch.as_tensor(mask_c, device=U1.device)].clone() if mask_c is not None else None
        # (e) the dived mode rising back on a second record (the path reversed)
        if ref_c is not None:
            path2 = ManufacturedLightPath(V9, path.h0 + path.D, -path.D, FX.sgn_dev, tol_block); trial2 = path2.eig(1.0); path2.trial = trial2
            cert2, sec2, anch2, occp2 = V9.light_sector_walk_v9(path2, FX.S_host, e1, sig1, mask_c, trial2, n_sea, d_min, 4096, tol_block, mu_prev=None)
            mask_e = V9.light_select_sector_v9(trial2[0], trial2[1], anch2)[0] if cert2["certified"] else None
            cr2 = sec2.get("crossings") or []
            ok = (cert2["certified"] and len(cr2) == 1 and cr2[0]["inversion_delta"] == -1 and sec2["anchors_end"] == 1 and sec2["inversions_plus_end"] == 0 and mask_e is not None
                  and bool(np.array_equal(mask_e, occ0)))
            r_e = resid(trial2[2][:, torch.as_tensor(mask_e, device=trial2[2].device)], ref0) if mask_e is not None else float("nan")
            fails += report_ok("(e) the inversion removed: the dived mode rising back on a second record", ok and r_e <= 1e-9,
                               "certified %s, crossings %d%s, anchors %s (expected 1), inversions %s (expected 0), the mask the mu S selection again %s, residual against the first reference %.1e" % (
                                   cert2["certified"], len(cr2), (" (overlap %.8f, delta %d)" % (cr2[0]["overlap"], cr2[0]["inversion_delta"])) if cr2 else "", sec2.get("anchors_end"), sec2.get("inversions_plus_end"), bool(np.array_equal(mask_e, occ0)) if mask_e is not None else None, r_e))
            # (b) two pockets, no crossing: the inverted state along a small field change
            x = chL.R / float(len(chL.R))
            dN = 2e-3 * np.exp(-((x - 0.45) / 0.15) ** 2); dv = -2e-3 * x * np.exp(-((x - 0.5) / 0.3) ** 2)
            Db = V9.clocked_gpu(chL, N0 + dN, v0 + dv) - path.h0
            path3 = ManufacturedLightPath(V9, path.h0 + path.D, Db, FX.sgn_dev, tol_block); trial3 = path3.eig(1.0); path3.trial = trial3
            cert3, sec3, anch3, occp3 = V9.light_sector_walk_v9(path3, FX.S_host, e1, sig1, mask_c, trial3, n_sea, d_min, 4096, tol_block, mu_prev=None)
            mask_b = V9.light_select_sector_v9(trial3[0], trial3[1], anch3)[0] if cert3["certified"] else None
            mask_idx = light_index_mask(V9, trial3[0], trial3[1], occp3) if occp3 is not None else None
            ok = (cert3["certified"] and not (sec3.get("crossings") or []) and sec3["anchors_end"] == 3 and sec3["inversions_plus_end"] == 1 and mask_b is not None and bool(np.array_equal(mask_b, mask_idx)))
            r_b = resid(trial3[2][:, torch.as_tensor(mask_b, device=trial3[2].device)], trial3[2][:, torch.as_tensor(mask_idx, device=trial3[2].device)]) if ok else float("nan")
            fails += report_ok("(b) two pockets, no crossing: the inverted state continued along a path on which no boundary closes", ok and r_b <= 1e-12,
                               "certified %s, pieces %s (Cholesky %s), crossings %d, anchors %s (expected 3), inversions %s (unchanged 1), the anchor mask equal to the index mask (the direct spectral projector onto the occupied intervals) %s, residual %.1e" % (
                                   cert3["certified"], cert3.get("pieces"), cert3.get("cholesky_calls"), len(sec3.get("crossings") or []), sec3.get("anchors_end"), sec3.get("inversions_plus_end"), bool(np.array_equal(mask_b, mask_idx)) if mask_b is not None else None, r_b))
        # (d) an avoided crossing at the boundary: g = 1e-3 (adiabatic) and g = 1e-8 (diabatic), the piece-refinement convergence in both cases
        for g, expect in ((1e-3, "adiabatic"), (1e-8, "diabatic")):
            pathd, e0d, sig0d, occ0d, ref0d, n_sead, mu0d, cd = light_pair_path(V9, chL, N0, v0, FX.sgn_dev, tol_block, g=g)
            triald = pathd.eig(1.0); pathd.trial = triald; e1d, sig1d, U1d = triald
            out = []
            for scale in (1.0, 0.5):
                certd, secd, anchd, occpd = V9.light_sector_walk_v9(pathd, FX.S_host, e0d, sig0d, occ0d, triald, n_sead, d_min, 4096, tol_block, mu_prev=mu0d, step_scale=scale)
                maskd, gapd = V9.light_select_sector_v9(e1d, sig1d, anchd) if certd["certified"] else (None, None)
                out.append((certd, secd, anchd, occpd, maskd, gapd))
            (certd, secd, anchd, occpd, maskd, gapd), (certd2, secd2, anchd2, occpd2, maskd2, gapd2) = out
            crd = secd.get("crossings") or []
            if maskd is not None and maskd2 is not None:
                r_conv = resid(U1d[:, torch.as_tensor(maskd, device=U1d.device)], U1d[:, torch.as_tensor(maskd2, device=U1d.device)])
            else:
                r_conv = float("nan")
            if expect == "adiabatic":
                cut = V9.light_cut_check_sector_v9(ref0d, e1d, sig1d, maskd, U1d, anchd, certd["norm_dK_G"], gapd) if maskd is not None else {"crossing_levels": None, "min_weight_selected": None}
                ok = (certd["certified"] and not crd and secd["anchors_end"] == 1 and secd["inversions_plus_end"] == 0 and maskd is not None and bool(np.array_equal(maskd, occ0d))
                      and cut["crossing_levels"] == 2 and r_conv <= 1e-10 and certd2["certified"] and certd2["pieces"] > certd["pieces"])
                fails += report_ok("(d) an avoided crossing of gap %.0e at the boundary: certified and adiabatic (the branch swap)" % g, ok,
                                   "certified %s, crossings %d, anchors %s, inversions %s, the mask the mu S prefix %s, the selected level's weight on the previous reference %s (below 1/2: the swap; the cut check records crossing_levels %s, not acted on), pieces %s against %s at half the step, the two references' residual %.1e (to the floor)" % (
                                       certd["certified"], len(crd), secd.get("anchors_end"), secd.get("inversions_plus_end"), bool(np.array_equal(maskd, occ0d)) if maskd is not None else None, cut.get("min_weight_selected"), cut.get("crossing_levels"), certd.get("pieces"), certd2.get("pieces"), r_conv))
            else:
                r_d = resid(U1d[:, torch.as_tensor(maskd, device=U1d.device)], ref0d) if maskd is not None else float("nan")
                ok = (certd["certified"] and len(crd) == 1 and crd[0]["overlap"] is not None and crd[0]["overlap"] >= 0.9 and crd[0]["inversion_delta"] == 1 and secd["anchors_end"] == 3 and secd["inversions_plus_end"] == 1
                      and maskd is not None and mask_c is not None and bool(np.array_equal(maskd, mask_c)) and r_d <= 1e-5 and certd2["certified"] and r_conv <= 1e-10 and len(secd2.get("crossings") or []) == 1)
                fails += report_ok("(d) an avoided crossing of gap %.0e at the boundary: the diabatic one of (c)" % g, ok,
                                   "certified %s, crossings %d%s, anchors %s, inversions %s, the mask equal to (c)'s %s, the residual against the previous reference %.1e (the admixture g/(2 Delta) = %.1e), at half the step: crossings %d%s, the two references' residual %.1e" % (
                                       certd["certified"], len(crd), (" (s* %.6f, overlap %.6f)" % (crd[0]["s"], crd[0]["overlap"])) if crd else "", secd.get("anchors_end"), secd.get("inversions_plus_end"), bool(np.array_equal(maskd, mask_c)) if (maskd is not None and mask_c is not None) else None, r_d, g / (2.0 * cd),
                                       len(secd2.get("crossings") or []), (" (s* %.6f, overlap %.6f)" % (secd2["crossings"][0]["s"], secd2["crossings"][0]["overlap"])) if secd2.get("crossings") else "", r_conv))
        # the heavy: (c) an exact crossing and (b) a no-crossing path from the inverted state
        h0h = V9.clocked_gpu(chH, N0, v0)
        pathh = ManufacturedHeavyPath(V9, h0h, torch.zeros_like(h0h)); e0h, U0h = pathh.eig(0.0)
        n_seah = int(np.sum(e0h < 0.0)); occ0h = e0h < 0.0
        ua = U0h[:, n_seah - 1].clone(); ub = U0h[:, n_seah].clone(); ch_ = float(e0h[n_seah] - e0h[n_seah - 1])
        pathh.D = ch_ * (torch.outer(ua, ua.conj()) - torch.outer(ub, ub.conj()))
        ref0h = U0h[:, torch.as_tensor(occ0h, device=U0h.device)].clone()
        trialh = pathh.eig(1.0); pathh.trial = trialh; e1h, U1h = trialh
        walkh, occh = V9.heavy_sector_walk_v9(pathh, ref0h, e0h, occ0h, trialh, n_seah, d_min, 4096, 1e-8)
        crh = walkh.get("crossings") or []
        if walkh["certified"]:
            refh, infoh, maskh = CERT.assign_occupation_v34(ref0h, torch.as_tensor(e1h, device=U1h.device), U1h, n_seah, occh)
            r_h = resid(refh, ref0h)
        else:
            infoh = {}; r_h = float("nan")
        ok = walkh["certified"] and len(crh) == 1 and crh[0]["overlap"] is not None and crh[0]["overlap"] >= 1.0 - 1e-6 and crh[0]["inversion_delta"] == 1 and walkh["anchors_end"] == 3 and walkh["inversions_end"] == 1 and r_h <= 1e-10
        ref_rsh, info_rsh, mask_rsh = OPS3.assign_max_overlap_v3(ref0h, e1h, U1h, n_seah, 1e-8)
        fails += report_ok("heavy (c) one exact crossing (the heavy channel, c = %.3e)" % ch_, ok,
                           "certified %s, pieces %s, eigvals %s, eigh %s, crossings %d%s, anchors %s, inversions %s, residual against the previous reference %.1e, wmin %s; the record-scale convention beside: wmin %s inversions %s" % (
                               walkh["certified"], walkh.get("pieces"), walkh.get("eigvals"), walkh.get("eigh"), len(crh), (" (s* %.6f, overlap %.8f)" % (crh[0]["s"], crh[0]["overlap"])) if crh else "", walkh.get("anchors_end"), walkh.get("inversions_end"), r_h, infoh.get("min_occupied_weight"), info_rsh.get("min_occupied_weight"), info_rsh.get("inversions")))
        if walkh["certified"]:
            Dbh = V9.clocked_gpu(chH, N0 + dN, v0 + dv) - h0h
            pathb = ManufacturedHeavyPath(V9, h0h + pathh.D, Dbh); trialb = pathb.eig(1.0); pathb.trial = trialb
            walkb, occb = V9.heavy_sector_walk_v9(pathb, refh, e1h, occh, trialb, n_seah, d_min, 4096, 1e-8)
            ok = walkb["certified"] and not (walkb.get("crossings") or []) and walkb["anchors_end"] == 3 and walkb["inversions_end"] == 1 and bool(np.array_equal(occb, occh))
            fails += report_ok("heavy (b) two pockets, no crossing: the inverted state along a small field change", ok,
                               "certified %s, pieces %s, eigvals %s, crossings %d, anchors %s (expected 3), inversions %s (unchanged 1), the occupation by index unchanged %s" % (
                                   walkb["certified"], walkb.get("pieces"), walkb.get("eigvals"), len(walkb.get("crossings") or []), walkb.get("anchors_end"), walkb.get("inversions_end"), bool(np.array_equal(occb, occh))))
    except Exception:
        import traceback
        fails += 1
        print("FAIL manufactured fixtures: exception\n%s" % traceback.format_exc(), flush=True)
    return fails


# ------------------------------------------------------------------------------------------------------- marks
def test_marks(v8_path, v9_path):
    a_ = open(v8_path, encoding="utf-8").read().splitlines(); b_ = open(v9_path, encoding="utf-8").read().splitlines()
    doc_end = next(i for i, l_ in enumerate(b_) if i > 0 and l_.rstrip().endswith('"""'))
    sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
    bad = []; hunks = 0; added = 0; changed_lines = 0; blank = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hunks += 1; added += j2 - j1
        if j2 <= doc_end + 1:
            continue
        if tag == "delete":
            bad.append(("deleted without replacement", a_[i1:i2][:2]))
            continue
        for l_ in b_[j1:j2]:
            if not l_.strip():
                blank += 1
                continue
            changed_lines += 1
            if MARK not in l_:
                bad.append(("changed line without the mark", l_[:100]))
    if bad:
        print("FAIL marks: %d hunks, %d lines added or changed (%d blank skipped), %d without the mark or deleted: %s" % (hunks, added, blank, len(bad), bad[:4]), flush=True)
        return 1
    print("PASS marks: %d changed hunks, %d lines added or changed outside the docstring (%d blank lines skipped), every one carries %r, nothing deleted without a replacement" % (hunks, changed_lines, blank, MARK), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------- compare with the sector keys stripped
def compare_runs_stripped(S8, A, B, keys, after=None):
    """version 8's compare_runs with the keys stripped from both runs' records (the sector mode's additions)."""
    JA, ZA = S8.load_run(A); JB, ZB = S8.load_run(B)
    rs = float(JA["args"]["rs"])

    def recs(J):
        return {round(r["T"] / rs, 9): strip_keys(r, keys) for r in J["records"] if r.get("E") is not None and (after is None or r["T"] / rs > after + 1e-9)}
    RA, RB = recs(JA), recs(JB)
    common = sorted(set(RA) & set(RB))
    diffs = []
    if set(RA) != set(RB):
        diffs.append(("records", "clocks", (sorted(set(RA) - set(RB))[:3], sorted(set(RB) - set(RA))[:3])))
    for t in common:
        S8.deep_diff(RA[t], RB[t], "record@%.5f" % t, diffs)
    for nm in ZA:
        if nm not in ZB:
            diffs.append((nm, "missing npz", 1.0))
            continue
        za = np.load(ZA[nm], allow_pickle=True); zb = np.load(ZB[nm], allow_pickle=True)
        TA = np.asarray(za["T"], float) / rs; TB = np.asarray(zb["T"], float) / rs
        for k in za.files:
            if k == "T" or k not in zb.files:
                continue
            x = za[k]; y = zb[k]
            if x.dtype.kind in "SU" or (x.dtype.kind == "O" and x.ndim == 0):
                continue
            if x.ndim >= 1 and x.shape[0] == len(TA) and y.ndim >= 1 and y.shape[0] == len(TB):
                for t in common:
                    ia = int(np.argmin(np.abs(TA - t))); ib = int(np.argmin(np.abs(TB - t)))
                    S8.deep_diff(x[ia], y[ib], "%s/%s@%.5f" % (nm, k, t), diffs)
            else:
                S8.deep_diff(x, y, "%s/%s" % (nm, k), diffs)
    return common, diffs


# ------------------------------------------------------------------------------------------------------- endtoend
def test_endtoend(S8, v8, v9, gpu, nsites, rs, records, work):
    work = work or tempfile.mkdtemp(prefix="v9_e2e_")
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k_, "16")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    runs = [("e2e_v8i", v8, []), ("e2e_v9ri", v9, ["--cut-transport", "record"]), ("e2e_v8L", v8, ["--reference-solve", "lagged"]),
            ("e2e_v9rL", v9, ["--reference-solve", "lagged", "--cut-transport", "record"]), ("e2e_v9sL", v9, ["--reference-solve", "lagged", "--cut-transport", "sector"])]
    for tag, script, extra in runs:
        cmd = [sys.executable, os.path.abspath(script)] + S8.e2e_args(nsites, rs, records, 2.0, work, tag, gpu, extra)
        log = os.path.join(work, tag + ".log"); t0 = time.time()
        print("endtoend: running %s (%s)" % (tag, log), flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(script)) or None)
        print("endtoend: %s exit %d in %.0f s" % (tag, rc, time.time() - t0), flush=True)
        if rc != 0:
            if tag.startswith("e2e_v8"):
                print("SKIP endtoend: version 8 itself exits %d at %d sites, rs %g (see %s): the stand-in fixture is missing at this size, not a failure of the build; report it" % (rc, nsites, rs, log), flush=True)
                return 0
            print("FAIL endtoend: %s exits %d where version 8 ran (see %s)" % (tag, rc, log), flush=True)
            return 1
    fails = 0
    common, diffs = S8.compare_runs(os.path.join(work, "e2e_v8i"), os.path.join(work, "e2e_v9ri"))
    fails += S8.report("endtoend (a): version 9 --cut-transport record implicit against version 8 implicit over %d records at %d sites" % (len(common), nsites), diffs)
    common, diffs = S8.compare_runs(os.path.join(work, "e2e_v8L"), os.path.join(work, "e2e_v9rL"))
    fails += S8.report("endtoend (a): version 9 --cut-transport record lagged against version 8 lagged over %d records at %d sites" % (len(common), nsites), diffs)
    common, diffs = compare_runs_stripped(S8, os.path.join(work, "e2e_v9rL"), os.path.join(work, "e2e_v9sL"), STRIP_SECTOR)
    fails += S8.report("endtoend (a): version 9 sector lagged against version 9 record lagged on this certified path over %d records (the sector-only keys stripped)" % len(common), diffs)
    JS = json.load(open(os.path.join(work, "collapse_e2e_v9sL.json")))
    ok_route = True
    for r in [q for q in JS["records"] if q.get("E") is not None]:
        chs_ = ((r.get("light_v36") or {}).get("channels") or {})
        rl = {k: (c.get("sector") or {}).get("route") for k, c in chs_.items()}; rh = {k: (c.get("sector") or {}).get("route") for k, c in (r.get("continuation") or {}).items()}
        ok_route = ok_route and r.get("cut_transport") == "sector" and all(v in ("mu S walk", "sector walk") for v in rl.values()) and all(v in ("Weyl", "sector walk") for v in rh.values())
        print("  T/rs %.4f: cut_transport %s; light routes %s; heavy routes %s; record label carries the sector sentence %s" % (
            r["T_over_rs"], r.get("cut_transport"), rl, sorted(set(rh.values())), "sector walk" in str(r.get("record_label"))), flush=True)
    fails += report_ok("endtoend: the sector run's records carry cut_transport sector and the routes (mu S walk | sector walk; Weyl | sector walk) with no fallback", ok_route)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "fixture", "marks", "endtoend", "compare-runs"])
    ap.add_argument("runs", nargs="*", help="compare-runs: A/TAG B/TAG")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--v8", default=None); ap.add_argument("--v9", default=None)
    ap.add_argument("--code-dir", default=_here, help="where engine_twofield_v8.py, engine_twofield_v9.py and engine_twofield_v8_selftest.py are imported from")
    ap.add_argument("--nsites", type=int, default=1024); ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--records", type=int, default=4)
    ap.add_argument("--work", default=None)
    ap.add_argument("--after", type=float, default=None, help="compare-runs: compare the records after this T/rs")
    ap.add_argument("--strip-sector", action="store_true", help="compare-runs: strip the sector mode's keys before comparing")
    a = ap.parse_args()
    if a.code_dir and a.code_dir not in sys.path:
        sys.path.insert(0, a.code_dir)
    import engine_twofield_v8_selftest as S8
    if a.what == "compare-runs":
        if len(a.runs) != 2:
            ap.error("compare-runs needs A/TAG B/TAG")
        if a.strip_sector:
            common, diffs = compare_runs_stripped(S8, a.runs[0], a.runs[1], STRIP_SECTOR, a.after)
        else:
            common, diffs = S8.compare_runs(a.runs[0], a.runs[1], a.after)
        rc = S8.report("compare-runs %s against %s over %d common records%s%s" % (a.runs[0], a.runs[1], len(common), (" after T/rs %g" % a.after) if a.after is not None else "", " (sector keys stripped)" if a.strip_sector else ""), diffs)
        for d_ in diffs[:40]:
            print("  ", d_)
        sys.exit(rc)
    if a.what == "endtoend":
        if not (a.v8 and a.v9):
            ap.error("endtoend needs --v8 and --v9 (the 2 engine files)")
        sys.exit(test_endtoend(S8, a.v8, a.v9, a.gpu, a.nsites, a.rs, a.records, a.work))
    fails = 0
    if a.what in ("all", "fixture"):
        import engine_twofield_v8 as V8
        import engine_twofield_v9 as V9
        fails += test_fixture(V8, V9, S8, a.gpu)
    if a.what == "marks" or (a.what == "all" and a.v8 and a.v9):
        if not (a.v8 and a.v9):
            ap.error("marks needs --v8 and --v9")
        fails += test_marks(a.v8, a.v9)
    print("selftest: %d failure(s)" % fails)
    sys.exit(fails)


if __name__ == "__main__":
    main()
