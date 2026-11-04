#!/usr/bin/env python
"""evolvedcut_slices_v1.py (reviewer A for the solve builder, the busi lane). C18 of reorientation 4 (input 01), the engine spec 55.4 item 9
(input 02), the ruling REVIEWA_fix_required_v1 II.4 ("start now on the box CPU, hreduce G as its second route"): the evolved trapping cut as a
slices144 readout module (READOUT "evolvedcut"), loaded by slices144_io_v1 (the driver, input 05) through run_slices144_v1 (the launcher,
input 06); the module registers itself in IO.READOUT_MODULES, the driver block (EVOLVEDCUT_driver_blocks_v1.md) adds it to the driver's own
table and adds the rung 2 and relaunch histories to the inventory.

C18's 2 pieces, on every stored slice (the production's 10, the sigma/a 8 member's 8, rung 2's 40, the relaunch's 23):

The readout. Per evolved channel kappa (every channel the slice stores, weight 16 kappa) at the cut X = {R > R_root}, R_root the outer
trapping root at the slice's clock (the record's outermost trapping zero by default, the applied fields' outer root beside it and used where
the record has none; --ec-root-source; a clock without any root reads the cut at 2gE of the record, labelled, or is skipped):
  S_X(C) - S_X(P_-), C = Phi Phi^dagger the pure Gaussian covariance of the channel's Loewdin-orthonormalized occupied columns (sea and dust,
  f all 1) and P_- = 1(h_m < 0) the held Killing vacuum at the slice's applied fields (page_d13_readout_v1b.exterior_entropies and
  killing_vacuum: page_own_object_v1's machinery moved from the 16 r_s cut to the root), summed with the multiplicity 16 kappa, against
  eta A/a^2 = 47,312 nats (input 01) as a ratio, and per lattice area A/a^2 = 4 pi (R_root/a)^2 at the actual applied root (G5: not the run's
  nominal radius; 16384 pi at R = 64 a beside).
The theorem (hreduce's EVOLVEDCUT_theorem_v1 and its red team, inputs 03 and 04): its terms computed exactly on the chain's channels, its
premises checked on the records.
  - EC1/G2: |S_X(C) - S_X(P_-)| <= t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1 <= nu = ||C - P_-||_1 (page_d13_readout_v1b.cross_cut_norms and
    full_trace_norm: the cross block's singular values and the principal angles between ran C and ran P_-); the check per channel and
    weighted; the pure-cut identity S_X(C) = sum_j F(sigma_j(Pi_X C Pi_Xc)) as the arithmetic's check. EC2/G3: rank(C - P_-) log 2 and
    d_X log 2 (d_X the complex modes beyond the cut), min{nu, r log 2, d_X log 2}; n_pair = Tr[(1 - P_-) C] beside as EC2's comparator, not a
    bound; the premises' defects ||Phi^dagger Phi - 1||_F and ||V^dagger V - 1||_F.
  - EC6/G6: P_- is the auxiliary instantaneous Killing ground Q, not the binding continued reference P. The reference the records carry is
    the stored occupation mask over the channel's Killing eigenlevels (mask_k; interior_continuation_v1.pocket_stats counts its k inverted
    levels against 1(e_k < 0)): where it differs, P_mask = the eigenvectors it marks, ||P_mask - P_-||_1 = k exactly (commuting projectors),
    G6.1's bridge |S_X(C) - S_X(P_-)| <= 2||Pi_X (C - P_mask) Pi_Xc||_1 + 2||Pi_X (P_mask - P_-) Pi_Xc||_1 with its second term <= k log 2,
    and S_X(C) - S_X(P_mask). Whether P_mask is the theorem's compatible continued P is the lead's and hreduce's reading; the module
    computes the reference the records carry and says so in every label.
  - the flat sea P_0 = 1(h_0 < 0) at N = 1, v = 0 on the same chain and cut (one dense eigendecomposition per channel per chain, cached):
    S_X(P_0), S_X(P_-) - S_X(P_0) and 2||Pi_X (P_- - P_0) Pi_Xc||_1 (the reference term of EC5's triangle on the chain's channels). The
    weighted S_X(P_0), S_X(P_-), S_X(C) are the stored channels' partial sums in the chain's reduction, not the 3D cut's values (EC1, G6;
    input 01: the genuine 3D evolved cut is about 3e7 modes at r_s/a 64 and is not computed).
  - the premises on the records: the gap at the root (EC3/G1: 2 N(root) m and N(root) m^2/(sqrt(m^2 + 3) + sqrt 3) from the applied fields
    at the root; the channels' own gaps at zero from the dense eigenvalues and the stored e_k; not the global pathwise delta); N bounded
    below (n_* = min N over the chain, inside and beyond the root; the record's N_min and min(N - |v|); EC4/G5a's profile bound exp(-B) with
    B = sum G |J/(r u)| in the chain's shell-total convention where the fields npz carries J_tot, its convention check beside); the slowly
    varying fields' eps = a/L (max |N'|, |v'|, |N''|, |v''| of the applied fields in lattice units over the chain and within
    --ec-eps-window-rs of the root; L_1, L_2; f_loc(eps) = eps^2 (1 + |log eps|)^4 at the measured eps and at a/r_s, which multiplies the
    theorem's unsupplied C_loc; f_exc = 1/A_lat); the pair content's trace norm nu (above).
Input 01's expectation of O(10 to 100) nats against 47,312 is an expectation to be measured: nothing here asserts it; the tables print the
measurement and the reading is the lead's.
Outputs per slice: evolvedcut_<history>_T<clock>.json and .npz (the exterior spectra of C, P_-, P_0, P_mask at the cut; the singular values of
every cross-cut block and of (1 - C) V); compose writes EVOLVEDCUT_slices_v1_<history>.md, EVOLVEDCUT_slices_v1_ladder.md (the histories at
matched clocks) and EVOLVEDCUT_slices_v1_all.json. --ec-self-test runs manufactured slices through the same functions: the excess exactly
zero (C = P_-, also through a unitary mixing of the columns), a one-mode excess with the closed form h_2(p), nu = 2 sqrt p,
t_X = 2 sqrt(p (1 - p)), rank 2, n_pair = p; the chain's vacuum as the sea (excess zero to rounding; P_- = P_0 there), the chain's one-mode
rotation (nu = 2 sqrt p exactly, the excess bounded by t_X and by 2 log 2), a two-level mask inversion (||P_mask - P_-||_1 = 2, the bridge).
CPU only; no plotting; no mutual information; no dates (elapsed seconds only). Complex64 slices carry the driver's floor in every row."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slices144_io_v1 as IO  # noqa: E402  (sets the thread environment before numpy is imported)
import interior_continuation_v1 as IC  # noqa: E402  (pocket_stats: the stored mask against 1(e_k < 0))
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

READOUT = "evolvedcut"
TITLE = "EVOLVEDCUT_slices_v1"
CODE = "evolvedcut_slices_v1.py"
ETA_A_NATS = 47312.0                      # input 01: eta A/a^2 = 47,312 nats at the root cut
LOG2 = math.log(2.0)
A_LAT_NOMINAL_64 = 16384.0 * math.pi      # input 03 EC5: A_lat = 4 pi (R/a)^2 = 16384 pi at R = 64 a
IO.READOUT_MODULES.setdefault(READOUT, "evolvedcut_slices_v1")
FLAT_CACHE = {"bytes": 0, "items": {}}
fx = IO.fx
C18 = {"item": "C18, the evolved trapping cut (reorientation 4, input 01); the engine spec 55.4 item 9 (input 02); the ruling REVIEWA_fix_required_v1 II.4",
       "readout": "per evolved channel at the outer root: S_X(C) - S_X(P_-) with the multiplicity 16 kappa, summed, against eta A/a^2 = 47,312 nats (input 01)",
       "theorem": ("hreduce's EVOLVEDCUT_theorem_v1 (G) and its red team (inputs 03, 04): EC1/G2 |S_X(C) - S_X(P)| <= 2||Pi_X (C - P) Pi_Xc||_1 <= ||C - P||_1; EC2/G3 the rank "
                   "and d_X log 2 terms; EC5/G5 the local term C_loc eps^2 (1 + |log eps|)^4 with C_loc unsupplied; EC6/G6 the auxiliary Killing ground against the continued reference"),
       "scope": ("the stored channels' sum in the chain's reduction is a truncated-channel sum, not the genuine 3D cut (input 01: about 3e7 modes at r_s/a 64; EC1, G6); "
                 "input 01's expectation of O(10 to 100) nats is to be measured, not asserted here; the reading is the lead's")}


def add_arguments(ap):
    g = ap.add_argument_group("evolvedcut_slices_v1 (C18: the evolved trapping cut)")
    g.add_argument("--ec-eta-A-nats", dest="ec_eta_A_nats", type=float, default=ETA_A_NATS, help="eta A/a^2 in nats (input 01: 47,312)")
    g.add_argument("--ec-root-source", dest="ec_root_source", default="record", choices=["record", "applied"],
                   help="the cut: the record's outermost trapping zero or the applied fields' outer root (both printed; the other where one is absent)")
    g.add_argument("--ec-no-root-cut", dest="ec_no_root_cut", default="2gE", help="a clock without any root: 2gE (the cut at R = 2 g E of the record), skip, or a radius in r_s")
    g.add_argument("--ec-extra-cuts-rs", dest="ec_extra_cuts_rs", default="", help="further cuts in r_s beside the root (16,24 for continuity with page_own_object_v1; none by default)")
    g.add_argument("--ec-channels", dest="ec_channels", default="", help="a comma list of kappa to restrict the channels (default: every channel the slice stores)")
    g.add_argument("--ec-flat-reference", dest="ec_flat_reference", default="on", choices=["on", "off"], help="the flat sea P_0 = 1(h_0 < 0) at N = 1, v = 0 on the same chain and cut")
    g.add_argument("--ec-flat-cache-gb", dest="ec_flat_cache_gb", type=float, default=4.0, help="host memory for the cached flat vacua (one 2n x n complex128 block per channel per chain)")
    g.add_argument("--ec-mask-reference", dest="ec_mask_reference", default="on", choices=["on", "off"], help="P_mask: the stored occupation mask over the Killing eigenlevels where it differs from 1(e_k < 0)")
    g.add_argument("--ec-full-norm", dest="ec_full_norm", default="gram", choices=["gram", "svd", "bound", "off"], help="the full ||C - P_-||_1 route (page_d13_readout_v1b.full_trace_norm)")
    g.add_argument("--ec-full-norm-gb", dest="ec_full_norm_gb", type=float, default=8.0, help="the memory allowed to the full trace norm (GiB)")
    g.add_argument("--ec-block-route", dest="ec_block_route", default="svd", choices=["svd", "gram"], help="the cross-cut block's singular values: svdvals (exact to rounding) or the smaller Gram")
    g.add_argument("--ec-bound-tol", dest="ec_bound_tol", type=float, default=1e-6, help="the rounding allowance of the EC1 check in nats")
    g.add_argument("--ec-sigma-tol", dest="ec_sigma_tol", type=float, default=1e-10, help="singular values of (1 - C) V above this count in rank(C - P_-)")
    g.add_argument("--ec-eps-window-rs", dest="ec_eps_window_rs", type=float, default=2.0, help="the window |R - R_root| <= this (r_s) for the near-root derivative figures")
    g.add_argument("--ec-degenerate-tol", dest="ec_degenerate_tol", type=float, default=1e-10, help="relative clustering tolerance of the Killing eigenvalues for the mask reference")
    g.add_argument("--ec-trace-chunk", dest="ec_trace_chunk", type=int, default=1024)
    g.add_argument("--ec-self-test", dest="ec_self_test", action="store_true", help="run the manufactured slices (python evolvedcut_slices_v1.py --ec-self-test ...) and exit")
    g.add_argument("--ec-self-test-sites", dest="ec_self_test_sites", type=int, default=48, help="the manufactured chain's sites")
    g.add_argument("--ec-self-test-p", dest="ec_self_test_p", type=float, default=0.3, help="the one-mode mixing weight p of the self-test")
    g.add_argument("--ec-self-test-m", dest="ec_self_test_m", type=float, default=0.2, help="the manufactured chain's m a (input 03: mu = .2)")
    g.add_argument("--ec-self-test-g", dest="ec_self_test_g", type=float, default=0.25, help="the manufactured records' g (input 03: G/a^2 = 1/4)")


def is_done(entry, a):
    return os.path.exists(IO.out_paths(a, READOUT, entry)[0])


# ---------------------------------------------------------------------------------------------------------- the arithmetic
def f_loc(eps):
    """the local-area theorem's factor eps^2 (1 + |log eps|)^4 (input 03 EC5: f_loc; about .173 at eps = 1/64); None where eps is not positive."""
    if eps is None:
        return None
    eps = float(eps)
    if not (eps > 0.0):
        return None
    return eps * eps * (1.0 + abs(math.log(eps))) ** 4


def h2(p):
    """the binary entropy in nats; 0 at 0 and at 1."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log1p(-p))


def interp_at(R, f, x):
    return float(np.interp(float(x), np.asarray(R, float), np.asarray(f, float)))


def _argmax_where(vals, sel):
    v = np.where(sel, np.asarray(vals, float), -np.inf)
    if not np.isfinite(v).any():
        return None
    return int(np.argmax(v))


def _pop_arrays(d, arrays, prefix):
    """every numpy array of the dict moved to arrays under prefix_key (the json keeps scalars only)."""
    for key in list(d.keys()):
        if isinstance(d[key], np.ndarray):
            arrays["%s_%s" % (prefix, key)] = d.pop(key)


def roots_of(rec, N, v, R):
    """(the record's outermost trapping zero, the applied fields' outer root by page_d13_readout_v1b.outer_root, the record's zeros)."""
    zs = [float(z) for z in (rec.get("trapping_zeros") or [])]
    r_rec = max(zs) if zs else None
    r_app = IO.D13.outer_root(np.asarray(N, float), np.asarray(v, float), np.asarray(R, float)) if (N is not None and v is not None) else None
    return r_rec, r_app, zs


def choose_cut(a, rec, r_rec, r_app, g, rs):
    """(R_cut, status, source): the record's outermost zero or the applied outer root (--ec-root-source first, the other where it is absent);
    a clock without either reads --ec-no-root-cut."""
    if a.ec_root_source == "record":
        first, second, names = r_rec, r_app, ("the record's outermost trapping zero", "the applied fields' outer root")
    else:
        first, second, names = r_app, r_rec, ("the applied fields' outer root", "the record's outermost trapping zero")
    if first is not None:
        return float(first), "root", names[0]
    if second is not None:
        return float(second), "root (%s absent: %s used)" % (names[0], names[1]), names[1]
    rule = str(a.ec_no_root_cut).strip()
    if rule.lower() == "skip":
        return None, "no root at this clock: skipped (--ec-no-root-cut skip)", None
    if rule.lower() == "2ge":
        E = rec.get("E")
        if E is None or g is None:
            return None, "no root at this clock and no record E (or no g) for the 2gE cut: skipped", None
        x = 2.0 * float(g) * float(E)
        return x, "no root at this clock: the cut at 2gE = %.4f of the record" % x, "2gE of the record"
    return float(rule) * float(rs), "no root at this clock: the cut at %s r_s (--ec-no-root-cut)" % rule, "fixed radius"


def fields_arrays(ctx, entry, T, keys):
    """the fields npz's arrays at the slice's record (keys present only), or (None, why)."""
    p = entry.get("fields_npz") or ctx.H.get("fields_npz")
    if not p or not os.path.isfile(p):
        return None, "no fields npz for this history (%s): the EC4 profile bound and the site-field fallback are skipped" % p
    try:
        k = ctx.fields_index(entry, T)
        out = {}
        for key in keys:
            arr = ctx.fields_key(entry, key)
            if arr is not None and arr.ndim >= 1 and k < arr.shape[0]:
                out[key] = np.asarray(arr[k], float)
        return out, "%s (record index %d; keys %s)" % (p, k, sorted(out))
    except Exception as ex:  # noqa: BLE001
        return None, "the fields npz %s could not be read (%s: %s)" % (p, type(ex).__name__, ex)


# ------------------------------------------------------------------------------------------------------------- the premises
def gap_block(R, N, v, x, m):
    """EC3/G1 at the applied root: the local band separation's lower bound 2 N(root) m and the conservative zero distance; the channels' gaps
    at zero are filled by process_slice."""
    Nx = interp_at(R, N, x)
    ax = interp_at(R, np.abs(np.asarray(v, float)), x)
    mu = float(m)
    return {"R_cut": float(x), "N_root": Nx, "abs_v_root": ax, "N_minus_absv_root": Nx - ax, "mu": mu, "band_separation_lower_bound_2Nm": 2.0 * Nx * mu,
            "zero_distance_conservative": Nx * mu * mu / (math.sqrt(mu * mu + 3.0) + math.sqrt(3.0)),
            "channel_gap_at_zero_dense": {}, "channel_gap_at_zero_stored": {}, "channel_gap_min": None, "channel_gap_min_kappa": None,
            "label": ("the local massive symbol at the applied root (EC3, G1; lattice units a = 1): the band separation 2 n e(k) >= 2 n mu with n = N(root), mu = m a, and the "
                      "conservative distance to zero n mu^2/(sqrt(mu^2 + 3) + sqrt 3) at |w| = n (N - |v| at the root is printed as the root's defect); the channels' gaps "
                      "at zero are the finite chain operators' own (the dense eigenvalues; the stored e_k beside): a checkable proxy on the records, not the theorem's "
                      "global pathwise delta (G1, EC3)")}


def lapse_block(R, N, v, x, rec, g, F, Finfo):
    """N bounded below: n_* on the applied fields (the chain, inside and beyond the root), the record's keys, and EC4/G5a's profile bound where
    the fields npz carries J_tot (the chain's shell-total convention, derived; its convention check beside)."""
    R = np.asarray(R, float)
    N = np.asarray(N, float)
    i = int(np.argmin(N))
    inside = R <= x
    beyond = R > x
    d = {"n_star_chain": float(N[i]), "at_R": float(R[i]), "n_star_inside_root": float(N[inside].min()) if inside.any() else None,
         "n_star_beyond_root": float(N[beyond].min()) if beyond.any() else None, "N_chain_end": float(N[-1]), "record_N_min": rec.get("N_min"),
         "record_min_N_minus_absv": rec.get("min_N_minus_absv"), "record_r_min_N_minus_absv": rec.get("r_min_N_minus_absv"), "ec4": None, "fields_source": Finfo,
         "label": ("n_* = min N of the applied fields over the chain, inside (R <= root) and beyond the root; the record's own N_min and min(N - |v|) beside; the theorem "
                   "needs n >= n_* > 0 along the whole path (G1/EC4): the stored slices' clocks are tabulated, the path between them is not read")}
    if F is not None and all(k in F for k in ("J_tot", "N", "v")):
        Nf = np.asarray(F["N"], float)
        vf = np.asarray(F["v"], float)
        J = np.asarray(F["J_tot"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = vf / Nf
            integrand = float(g) * np.abs(J / (R * u))
            rhs = -float(g) * J / (R * u)
            lhs = np.gradient(np.log(np.clip(Nf, 1e-300, None)), R)
        ok = np.isfinite(integrand) & (R > x)
        dR = np.gradient(R)
        B = float(np.sum((integrand * dR)[ok])) if ok.any() else None
        okr = ok & np.isfinite(rhs) & (np.abs(rhs) > 1e-14)
        ratio = float(np.median(lhs[okr] / rhs[okr])) if okr.any() else None
        d["ec4"] = {"B_beyond_root": B, "exp_minus_B_times_N_end": (math.exp(-B) * float(Nf[-1])) if B is not None else None,
                    "min_N_fields_beyond_root": float(Nf[R > x].min()) if (R > x).any() else None, "N_end_fields": float(Nf[-1]), "sites_used": int(ok.sum()),
                    "sites_u_zero_or_nonfinite": int((~np.isfinite(integrand)).sum()), "convention_check_median_ratio": ratio, "convention_check_sites": int(okr.sum()),
                    "convention": ("EC4/G5a: (log N)' = -4 pi G r J/u with 4 pi r^2 J read as the stored shell total J_tot per site (lattice units, a = 1), so (log N)' = -G J_tot/(r u), "
                                   "u = v/N on the fields npz's site fields; B = sum over the sites beyond the root of G |J_tot/(r u)| dR; exp(-B) N(end) is EC4's lower bound on N beyond "
                                   "the root (derived; the median ratio of the finite-difference (log N)' to -G J_tot/(r u) over those sites is the convention's check: far from 1 marks the "
                                   "convention or the averaging, not a result)")}
    return d


def eps_block(R, N, v, x, rs, window_rs, A_lat):
    """the slowly varying fields' eps = a/L on the applied (averaged) site fields: |f'|, |f''| by numpy.gradient in lattice units for f = N, v
    over the chain and within |R - root| <= window; L_1 = 1/max|f'|, L_2 = 1/sqrt(max|f''|); the theorem's factors at the measured eps and at a/r_s."""
    R = np.asarray(R, float)
    win = np.abs(R - x) <= float(window_rs) * float(rs)
    out = {"window_rs": float(window_rs), "window_sites": int(win.sum()), "a_over_rs": 1.0 / float(rs), "fields": {}}
    eps1_w = eps1_c = eps2_w = 0.0
    for name, f in (("N", N), ("v", v)):
        f = np.asarray(f, float)
        d1 = np.gradient(f, R)
        d2 = np.gradient(d1, R)
        a1 = np.abs(d1)
        a2 = np.abs(d2)
        i1 = int(np.argmax(a1))
        i2 = int(np.argmax(a2))
        j1 = _argmax_where(a1, win)
        j2 = _argmax_where(a2, win)
        rel = a1 / np.maximum(np.abs(f), 1e-300)
        fig = {"d1_max_chain": float(a1[i1]), "d1_max_chain_at_R": float(R[i1]), "d2_max_chain": float(a2[i2]), "d2_max_chain_at_R": float(R[i2]),
               "d1_max_window": float(a1[j1]) if j1 is not None else None, "d1_max_window_at_R": float(R[j1]) if j1 is not None else None,
               "d2_max_window": float(a2[j2]) if j2 is not None else None, "rel_d1_max_window": float(np.max(rel[win])) if win.any() else None,
               "L1_window": (1.0 / float(a1[j1])) if (j1 is not None and a1[j1] > 0) else None,
               "L2_window": (1.0 / math.sqrt(float(a2[j2]))) if (j2 is not None and a2[j2] > 0) else None}
        out["fields"][name] = fig
        eps1_c = max(eps1_c, fig["d1_max_chain"])
        eps1_w = max(eps1_w, fig["d1_max_window"] or 0.0)
        eps2_w = max(eps2_w, math.sqrt(fig["d2_max_window"]) if fig["d2_max_window"] else 0.0)
    out.update({"eps_window": eps1_w, "eps_chain": eps1_c, "eps2_window": eps2_w, "L_window": (1.0 / eps1_w) if eps1_w > 0 else None,
                "L_over_rs_window": (1.0 / eps1_w / float(rs)) if eps1_w > 0 else None, "f_loc_eps_window": f_loc(eps1_w), "f_loc_eps2_window": f_loc(eps2_w),
                "f_loc_eps_chain": f_loc(eps1_c), "f_loc_a_over_rs": f_loc(1.0 / float(rs)), "f_exc_A_lat_cut": 1.0 / float(A_lat), "f_exc_nominal_64": 1.0 / A_LAT_NOMINAL_64,
                "label": ("eps = a/L read as max|f'| of the applied site fields (averaged over sigma = ell by the engine) in lattice units, F_1 = 1 (G1's |partial^alpha (n, w)| <= "
                          "F_alpha (a/L)^|alpha|), L_2 from the second derivative; f_loc(eps) = eps^2 (1 + |log eps|)^4 multiplies the theorem's C_loc, which no input supplies "
                          "(EC5, G5); f_exc = 1/A_lat multiplies the excitation norm; L is the actual variation length, not automatically r_s (G5, EC4)")})
    return out


# ------------------------------------------------------------------------------------------------------- the flat reference
def flat_reference(rd, m, kap, h0, dev, a, log):
    """P_0 = 1(h_0 < 0) at N = 1, v = 0 on the slice's chain (page_d13_readout_v1b.killing_vacuum on h_0), cached per (nsites, m, kappa, R)."""
    key = (int(rd.nsites), round(float(m), 12), int(kap), zlib.crc32(np.ascontiguousarray(np.asarray(rd.R, float)).tobytes()))
    it = FLAT_CACHE["items"].get(key)
    if it is not None:
        return dict(it, cached=True)
    vac0 = IO.D13.killing_vacuum(h0, dev, "dense")
    it = {"V0": vac0["Vneg"], "e0": vac0["e"], "n_neg": vac0["n_neg"], "E_ref": vac0["E_ref"], "eigh_seconds": vac0["seconds"], "n_zero_1e-12": vac0.get("n_zero_1e-12")}
    nb = int(it["V0"].numel()) * 16
    if FLAT_CACHE["bytes"] + nb <= float(a.ec_flat_cache_gb) * 1e9:
        FLAT_CACHE["items"][key] = it
        FLAT_CACHE["bytes"] += nb
        log("  flat sea kappa %d (N = 1, v = 0): %d occupied, E_ref %.6e, %.0f s; cached (%.2f GB of %.1f)" % (kap, it["n_neg"], it["E_ref"], it["eigh_seconds"], FLAT_CACHE["bytes"] / 1e9, a.ec_flat_cache_gb))
    else:
        log("  flat sea kappa %d (N = 1, v = 0): %d occupied, E_ref %.6e, %.0f s; not cached (the budget %.1f GB is full: recomputed per slice)" % (kap, it["n_neg"], it["E_ref"], it["eigh_seconds"], a.ec_flat_cache_gb))
    return dict(it, cached=False)


# ---------------------------------------------------------------------------------------------------- the EC1 pair readout
def pair_readout(A, B, R2_t, cuts, a, full_norm=None, identity=True):
    """EC1 on the pair of projectors (A A^dagger, B B^dagger) at the cuts with page_d13_readout_v1b's arithmetic: S_X of both and the excess
    S_X(A) - S_X(B) (exterior_entropies), t_X = 2||Pi_X (A A^dagger - B B^dagger) Pi_Xc||_1 from the cross block's singular values
    (cross_cut_norms) and the check |excess| <= t_X (cross_cut_check), the full ||A A^dagger - B B^dagger||_1 by the principal angles with its
    rank, sigma_max, ||.||_2, n_pair and n_hole (full_trace_norm), the premises' defects ||A^dagger A - 1||_F, ||B^dagger B - 1||_F
    (premise_defects); where identity is set, the pure-cut identity S_X(A) = sum_j F(sigma_j(Pi_X A A^dagger Pi_Xc)) from A's own cross block."""
    D13 = IO.D13
    torch = IO.torch
    ent = D13.exterior_entropies(A, R2_t, cuts, B)
    cc = D13.cross_cut_norms(A, B, R2_t, cuts, a.ec_block_route)
    fails = D13.cross_cut_check(ent, cc, float(a.ec_bound_tol))
    ccA = D13.cross_cut_norms(A, None, R2_t, cuts, a.ec_block_route) if identity else None
    out = {"cuts": {}, "fails": fails, "sv": {}}
    for lab, _ in cuts:
        d = ent[lab]
        out["sv"][lab] = {"sv_block": cc[lab]["sv_block"]}
        if ccA is not None:
            S_sv = float(D13.pure_cut_F(torch.as_tensor(np.asarray(ccA[lab]["sv_block"], float))).sum())
            d["pure_cut_identity"] = {"S_X_from_singular_values": S_sv, "defect": float(d["S_X"]) - S_sv, "n_sv": int(ccA[lab]["n_sv"]), "sigma_max": float(ccA[lab]["sigma_max_block"]),
                                      "label": "S_X(Z) = sum_j F(sigma_j(K_Z)), K_Z = Pi_X Z Pi_Xc, F(s) = h_2((1 - sqrt(1 - 4 s^2))/2), for a projector Z (EC1/G2's identity behind the bound): the arithmetic's check"}
            out["sv"][lab]["sv_block_A"] = ccA[lab]["sv_block"]
        out["cuts"][lab] = d
    tn = D13.full_trace_norm(A, B, full_norm or a.ec_full_norm, float(a.ec_full_norm_gb), float(a.ec_sigma_tol), int(a.ec_trace_chunk))
    out["sv_full"] = tn.pop("sv", None)
    out["trace_norm"] = tn
    out["premise_defects"] = D13.premise_defects(A, B, int(a.ec_trace_chunk))
    return out


# --------------------------------------------------------------------------------------------------------------- one channel
def channel_readout(rd, m, kap, N_app, v_app, cuts, R2_t, dev, a, log, arrays):
    t1 = time.time()
    torch = IO.torch
    D13 = IO.D13
    ch = rd.channel(kap, orthonormalize=(a.loewdin == "on"))
    Phi = ch["Phi"]
    out = {"kappa": int(kap), "weight": float(ch["weight"]), "n_sea": int(ch["n_sea"]), "n_dust": int(ch["n_dust"]), "N_occ": int(Phi.shape[1]),
           "f_dust_min_max": [float(np.min(ch["f_dust"])), float(np.max(ch["f_dust"]))] if ch["n_dust"] else None,
           "stored_orthonormality": ch["stored_orthonormality"], "loewdin_orthonormality": ch["loewdin_orthonormality"], "orthonormalization": ch["orthonormalization"],
           "dust_sea_overlap_max": ch["dust_sea_overlap_max"]}
    h0, Q, H = IO.operators(rd.R, m, kap, N_app, v_app)
    e_st, mask_st = ch["e"], ch["mask"]
    pk = IC.pocket_stats(e_st, mask_st, ch["n_sea"], ch["n_dust"]) if (e_st is not None and mask_st is not None) else None
    out["pockets_slice"] = pk
    need_mask = bool(a.ec_mask_reference == "on" and pk is not None and not pk["mask_equals_negative_set"])
    Vmask = None
    mask_note = {}
    if need_mask:
        t2 = time.time()
        e_t, V = IO.OPS.eigh_csr(H, dev)
        e_np = np.asarray(e_t.cpu().numpy(), float)
        neg = e_np < 0.0
        Vneg = V[:, torch.as_tensor(neg)].clone()
        order = np.argsort(np.asarray(e_st, float), kind="stable")
        mask_sorted = np.asarray(mask_st, bool)[: len(e_st)][order]
        scale = max(float(np.max(np.abs(e_np))), 1e-300)
        brk = np.nonzero(np.diff(e_np) > float(a.ec_degenerate_tol) * scale)[0] + 1
        bounds = np.concatenate([[0], brk, [len(e_np)]]).astype(int)
        split = [[int(b0), int(b1)] for b0, b1 in zip(bounds[:-1], bounds[1:]) if (b1 - b0 > 1 and mask_sorted[b0:b1].any() and not mask_sorted[b0:b1].all())]
        Vmask = V[:, torch.as_tensor(mask_sorted)].clone()
        del V
        vac = {"e": e_np, "Vneg": Vneg, "n_neg": int(neg.sum()), "n_zero_1e-12": int((np.abs(e_np) < 1e-12).sum()), "E_ref": float(e_np[neg].sum()),
               "route": "dense (page_ops_v6.eigh_csr: the eigendecomposition page_d13_readout_v1b.killing_vacuum takes, kept whole for the mask reference)", "seconds": time.time() - t2}
        mask_note = {"n_mask": int(mask_sorted.sum()), "degenerate_clusters_split_by_mask": split, "stored_e_sorted_vs_dense_max": float(np.max(np.abs(np.asarray(e_st, float)[order] - e_np)))}
    else:
        vac = D13.killing_vacuum(H, dev, "dense")
        Vneg = vac["Vneg"]
    e_np = np.asarray(vac["e"], float)
    pos_e = e_np[e_np > 0.0]
    neg_e = e_np[e_np < 0.0]
    out["vacuum"] = {"n_neg": vac["n_neg"], "E_ref": vac["E_ref"], "route": vac["route"], "seconds": vac["seconds"], "n_zero_1e-12": vac.get("n_zero_1e-12"),
                     "eig_vs_stored": float(np.max(np.abs(np.sort(e_np) - np.sort(np.asarray(e_st, float))))) if e_st is not None else None,
                     "gap_at_zero_dense": float(pos_e.min() - neg_e.max()) if (pos_e.size and neg_e.size) else None,
                     "gap_at_zero_stored": pk["gap_at_zero"] if pk else None, "N_occ_minus_n_neg": int(Phi.shape[1]) - int(vac["n_neg"])}
    Phi_t = torch.as_tensor(Phi)
    main = pair_readout(Phi_t, Vneg, R2_t, cuts, a, identity=True)
    tn = main["trace_norm"]
    out["trace_norm"] = tn
    out["premise_defects"] = main["premise_defects"]
    if main.get("sv_full") is not None:
        arrays["k%d_sv_full_C_minus_P" % kap] = main["sv_full"]
    fl = None
    if a.ec_flat_reference == "on":
        it = flat_reference(rd, m, kap, h0, dev, a, log)
        fp = pair_readout(Vneg, it["V0"], R2_t, cuts, a, full_norm=("gram" if a.ec_full_norm != "off" else "off"), identity=False)
        fl = {"n_neg_flat": it["n_neg"], "E_ref_flat": it["E_ref"], "cached": it["cached"], "eigh_seconds": it["eigh_seconds"], "nu_vac_flat": fp["trace_norm"].get("nu"),
              "rank_vac_flat": fp["trace_norm"].get("rank"), "frob_vac_flat": fp["trace_norm"].get("frob"), "cuts": fp["cuts"], "sv": fp["sv"]}
        if fp.get("sv_full") is not None:
            arrays["k%d_sv_full_P_minus_P0" % kap] = fp["sv_full"]
        del it, fp
    out["flat"] = {k_: v_ for k_, v_ in fl.items() if k_ not in ("cuts", "sv")} if fl else None
    mr = None
    if Vmask is not None:
        k_inv = int(pk["inverted_levels"])
        cm = pair_readout(Phi_t, Vmask, R2_t, cuts, a, identity=False)
        mp = pair_readout(Vmask, Vneg, R2_t, cuts, a, full_norm=("gram" if a.ec_full_norm != "off" else "off"), identity=False)
        mr = dict(mask_note, k_inverted_levels=k_inv, occupied_at_positive_e=pk["occupied_at_positive_e"], empty_at_negative_e=pk["empty_at_negative_e"], inverted_runs=pk["inverted_runs"],
                  nu_C_mask=cm["trace_norm"].get("nu"), rank_C_mask=cm["trace_norm"].get("rank"), n_pair_C_mask=cm["trace_norm"].get("n_pair"), nu_mask_vac=mp["trace_norm"].get("nu"),
                  nu_mask_vac_minus_k=(mp["trace_norm"]["nu"] - k_inv) if mp["trace_norm"].get("nu") is not None else None, k_log2=k_inv * LOG2, cuts={},
                  label=("P_mask: the stored occupation mask over the channel's Killing eigenlevels at the record (the engine's own continued occupation; EC6's occupations kept across "
                         "zero), differing from 1(e_k < 0) in k levels; ||P_mask - P_-||_1 = k for commuting projectors (the computed value and its deviation from k printed); whether "
                         "P_mask is the theorem's compatible continued P is the lead's and hreduce's reading"))
        for lab, _ in cuts:
            dc = cm["cuts"][lab]
            dm = mp["cuts"][lab]
            d0 = main["cuts"][lab]
            _pop_arrays(dc, arrays, "k%d_cut%s_Cmask" % (kap, lab))
            _pop_arrays(dm, arrays, "k%d_cut%s_maskvac" % (kap, lab))
            mr["cuts"][lab] = {"S_X_Pmask": dc["S_X_vac"], "excess_vs_mask": dc["excess"], "t_X_C_mask": dc.get("t_X"), "bound_ok_C_mask": dc.get("bound_ok"),
                               "dS_mask_minus_vac": dm["excess"], "t_X_mask_vac": dm.get("t_X"), "bound_ok_mask_vac": dm.get("bound_ok"),
                               "bound_ok_mask_vac_klog2": bool(abs(dm["excess"]) <= k_inv * LOG2 + float(a.ec_bound_tol)),
                               "bridge_sum": (dc.get("t_X") or 0.0) + (dm.get("t_X") or 0.0),
                               "bound_ok_bridge": bool(abs(d0["S_X"] - d0["S_X_vac"]) <= (dc.get("t_X") or 0.0) + (dm.get("t_X") or 0.0) + float(a.ec_bound_tol))}
            arrays["k%d_cut%s_sv_block_C_minus_Pmask" % (kap, lab)] = cm["sv"][lab]["sv_block"]
            arrays["k%d_cut%s_sv_block_Pmask_minus_P" % (kap, lab)] = mp["sv"][lab]["sv_block"]
        del cm, mp
    if mr:
        out["mask_reference"] = {k_: v_ for k_, v_ in mr.items() if k_ != "cuts"}
    elif pk is None:
        out["mask_reference"] = {"k_inverted_levels": None, "status": "no stored mask or e_k in this slice"}
    elif pk["mask_equals_negative_set"]:
        out["mask_reference"] = {"k_inverted_levels": 0, "status": "equal to P_- (the stored mask is the negative set)"}
    else:
        out["mask_reference"] = {"k_inverted_levels": int(pk["inverted_levels"]), "status": "off (--ec-mask-reference off): %d inverted levels not resolved" % int(pk["inverted_levels"])}
    out["cuts"] = {}
    for lab, _ in cuts:
        d = main["cuts"][lab]
        _pop_arrays(d, arrays, "k%d_cut%s" % (kap, lab))
        arrays["k%d_cut%s_sv_block_C_minus_P" % (kap, lab)] = main["sv"][lab]["sv_block"]
        if "sv_block_A" in main["sv"][lab]:
            arrays["k%d_cut%s_sv_block_C" % (kap, lab)] = main["sv"][lab]["sv_block_A"]
        d["d_X_complex_modes"] = int(d["n_modes_X"])
        d["d_X_log2"] = float(d["n_modes_X"]) * LOG2
        d["rank_log2"] = (float(tn["rank"]) * LOG2) if tn.get("rank") is not None else None
        terms = [t_ for t_ in (tn.get("nu"), d["rank_log2"], d["d_X_log2"]) if t_ is not None]
        d["min_bound_term"] = min(terms) if terms else None
        d["label"] = ("S_X(C) and S_X(P_-) beyond the cut (pure Gaussian C from the Loewdin-orthonormalized occupied columns; the held Killing vacuum 1(h_m < 0) at the applied fields); "
                      "excess = S_X(C) - S_X(P_-); EC1: |excess| <= t_X = 2||Pi_X (C - P_-) Pi_Xc||_1 <= nu; EC2: min{nu, rank(C - P_-) log 2, d_X log 2}, d_X the complex modes "
                      "beyond the cut (2 per site); the multiplicity 16 kappa is applied in the weighted sums")
        if fl:
            f_ = fl["cuts"][lab]
            _pop_arrays(f_, arrays, "k%d_cut%s_flat" % (kap, lab))
            d["flat"] = {"S_X_P0": f_["S_X_vac"], "S_Xc_P0": f_["S_Xc_vac"], "dS_vac_minus_flat": f_["excess"], "t_X_vac_flat": f_.get("t_X"), "bound_ok": f_.get("bound_ok"),
                         "n_sv_above_1e-8": f_.get("n_sv_above_1e-8"), "label": "P_0 = 1(h_0 < 0) at N = 1, v = 0 on the same chain and cut; S_X(P_-) - S_X(P_0) and 2||Pi_X (P_- - P_0) Pi_Xc||_1 (EC5's reference term on this channel)"}
            arrays["k%d_cut%s_sv_block_P_minus_P0" % (kap, lab)] = fl["sv"][lab]["sv_block"]
        else:
            d["flat"] = None
        d["mask_reference"] = mr["cuts"][lab] if mr else None
        out["cuts"][lab] = d
    out["seconds"] = time.time() - t1
    del Phi, Phi_t, Vneg, Vmask, vac, main, ch, fl, mr
    return out


def accumulate(ws, out, lab, w):
    """the weighted sums (16 kappa) of one channel's figures at one cut into ws."""
    d = out["cuts"][lab]
    tn = out["trace_norm"]

    def add(key, val):
        if val is not None:
            ws[key] = ws.get(key, 0.0) + w * float(val)
    for key, val in (("S_X", d["S_X"]), ("S_Xc", d["S_Xc"]), ("S_X_vac", d["S_X_vac"]), ("excess", d["excess"]), ("abs_dS", d.get("abs_dS")), ("t_X", d.get("t_X")),
                     ("nu", tn.get("nu")), ("rank_log2", d.get("rank_log2")), ("d_X_log2", d["d_X_log2"]), ("min_bound_term", d.get("min_bound_term")), ("n_pair", tn.get("n_pair")),
                     ("n_hole", tn.get("n_hole")), ("frob", tn.get("frob")), ("bound_cauchy_schwarz", tn.get("bound_cauchy_schwarz")), ("bound_rank_sigma", tn.get("bound_rank_sigma"))):
        add(key, val)
    ws["d_X_modes_weighted"] = ws.get("d_X_modes_weighted", 0.0) + w * float(d["n_modes_X"])
    ws["rank_weighted"] = ws.get("rank_weighted", 0.0) + w * float(tn["rank"]) if tn.get("rank") is not None else ws.get("rank_weighted")
    ws["max_abs_purity_gap"] = max(ws.get("max_abs_purity_gap", 0.0), abs(float(d["purity_gap"])))
    if d.get("pure_cut_identity"):
        ws["max_abs_identity_defect"] = max(ws.get("max_abs_identity_defect", 0.0), abs(float(d["pure_cut_identity"]["defect"])))
    ws["bound_ok_all"] = bool(ws.get("bound_ok_all", True) and bool(d.get("bound_ok")))
    ws["min_slack"] = min(ws.get("min_slack", float("inf")), float(d["slack"])) if d.get("slack") is not None else ws.get("min_slack")
    ws["channels_with_dust"] = ws.get("channels_with_dust", 0) + (1 if out["n_dust"] else 0)
    fl = d.get("flat")
    if fl:
        add("S_X_P0", fl["S_X_P0"])
        add("dS_vac_minus_flat", fl["dS_vac_minus_flat"])
        add("t_X_vac_flat", fl["t_X_vac_flat"])
        ws["flat_bound_ok_all"] = bool(ws.get("flat_bound_ok_all", True) and bool(fl.get("bound_ok")))
    kinv = out["mask_reference"].get("k_inverted_levels") or 0
    ws["k_inverted_weighted"] = ws.get("k_inverted_weighted", 0.0) + w * float(kinv)
    ws["k_inverted_total"] = ws.get("k_inverted_total", 0) + int(kinv)
    ws["channels_with_inversions"] = ws.get("channels_with_inversions", 0) + (1 if kinv else 0)
    mr = d.get("mask_reference")
    if mr:
        add("S_X_Pmask", mr["S_X_Pmask"])
        add("excess_vs_mask", mr["excess_vs_mask"])
        add("t_X_C_mask", mr["t_X_C_mask"])
        add("t_X_mask_vac", mr["t_X_mask_vac"])
        add("bridge_sum", mr["bridge_sum"])
        add("dS_mask_minus_vac", mr["dS_mask_minus_vac"])
        ws["bridge_ok_all"] = bool(ws.get("bridge_ok_all", True) and bool(mr.get("bound_ok_bridge")) and bool(mr.get("bound_ok_mask_vac_klog2")))
    else:
        add("S_X_Pmask", d["S_X_vac"])
        add("excess_vs_mask", d["excess"])
        add("t_X_C_mask", d.get("t_X"))
        add("t_X_mask_vac", 0.0)
        add("bridge_sum", d.get("t_X"))
        add("dS_mask_minus_vac", 0.0)
        ws["bridge_ok_all"] = bool(ws.get("bridge_ok_all", True) and bool(d.get("bound_ok")))


def comparison(ws, A_lat, eta, rd):
    ex = ws.get("excess")
    r_ = lambda v_: (float(v_) / float(eta)) if v_ is not None else None  # noqa: E731
    return {"eta_A_nats": float(eta), "A_lat": float(A_lat), "excess_weighted": ex, "ratio_excess_to_eta_A": r_(ex), "abs_ratio_excess_to_eta_A": abs(r_(ex)) if ex is not None else None,
            "excess_per_lattice_area": (ex / float(A_lat)) if ex is not None else None, "t_X_weighted": ws.get("t_X"), "ratio_t_X_to_eta_A": r_(ws.get("t_X")),
            "nu_weighted": ws.get("nu"), "ratio_nu_to_eta_A": r_(ws.get("nu")), "min_bound_term_weighted": ws.get("min_bound_term"), "ratio_min_bound_term_to_eta_A": r_(ws.get("min_bound_term")),
            "bridge_sum_weighted": ws.get("bridge_sum"), "ratio_bridge_to_eta_A": r_(ws.get("bridge_sum")), "S_X_C_weighted_partial": ws.get("S_X"), "S_X_vac_weighted_partial": ws.get("S_X_vac"),
            "S_X_P0_weighted_partial": ws.get("S_X_P0"), "ratio_S_X_P0_partial_to_eta_A": r_(ws.get("S_X_P0")), "channels": rd.kappas, "weights": "16 kappa", "sum_weights": float(sum(rd.weights.values())),
            "label": ("the measured low-channel excess sum_kappa 16 kappa [S_X(C) - S_X(P_-)] at the cut against eta A/a^2 = %g nats (input 01) and per lattice area A/a^2 = 4 pi (R_cut/a)^2; "
                      "the theorem's terms beside (EC1: |excess| <= t_X <= nu; EC2: min{nu, r log 2, d_X log 2}; G6.1's bridge over P_mask); the weighted S_X(C), S_X(P_-), S_X(P_0) are the "
                      "stored channels' partial sums in the chain's reduction, not the 3D cut's values (EC1, G6)" % eta)}


# ------------------------------------------------------------------------------------------------------------ the slice
def process_slice(rd, ctx, entry, a):
    if IO.D13 is None:
        raise RuntimeError("evolvedcut_slices_v1 needs page_d13_readout_v1b (exterior_entropies, killing_vacuum, cross_cut_norms, full_trace_norm, outer_root): not importable (%s)" % IO.SETUP.get("D13"))
    if getattr(ctx, "m", None) is None or getattr(ctx, "g", None) is None:
        raise RuntimeError("the history's records json %s gives no args.m or args.g: h_m needs m and the 2gE cut needs g" % ctx.H.get("records_json"))
    t0 = time.time()
    log = ctx.log
    torch = IO.torch
    dev = torch.device("cpu")
    IO.D13.TRACE_CHUNK[0] = int(a.ec_trace_chunk)
    pj, pz = IO.out_paths(a, READOUT, entry)
    rec = ctx.record_at(entry, rd.T)
    m, g, rs = float(ctx.m), float(ctx.g), float(rd.rs)
    R = np.asarray(rd.R, float)
    F, Finfo = fields_arrays(ctx, entry, rd.T, ("N", "v", "J_tot", "eps_tot"))
    N_app, v_app = rd.N_applied, rd.v_applied
    applied_source = "the slice's N_applied, v_applied (the record's accepted site fields)"
    if N_app is None or v_app is None:
        if F is not None and "N" in F and "v" in F and len(F["N"]) == rd.nsites:
            N_app, v_app = np.asarray(F["N"], float), np.asarray(F["v"], float)
            applied_source = "the fields npz's site fields N, v at the record (the slice carries no N_applied/v_applied)"
        else:
            raise RuntimeError("the slice %s carries no N_applied/v_applied and no fields npz supplies N, v: the channel operator cannot be built" % os.path.basename(rd.path))
    row = IO.slice_header(rd, ctx, entry, rec, READOUT, TITLE, CODE)
    row["c18"] = C18
    E_rec = rec.get("E")
    row["record"] = {"E": E_rec, "E_over_E0": rec.get("E_over_E0"), "two_gE": (2.0 * g * float(E_rec)) if E_rec is not None else None, "trapping_zeros": rec.get("trapping_zeros"),
                     "R_t": rec.get("R_t"), "kappa_T": rec.get("kappa_T"), "r_in": rec.get("r_in"), "min_N_minus_absv": rec.get("min_N_minus_absv"), "r_min_N_minus_absv": rec.get("r_min_N_minus_absv"),
                     "N_min": rec.get("N_min"), "inversions_total": rec.get("inversions_total"), "mask_equals_negative_set": rec.get("mask_equals_negative_set"), "n_refill_R0": rec.get("n_refill_R0"),
                     "n_pocket": rec.get("n_pocket"), "converged": rec.get("converged"), "record_engine": rec.get("engine") or rec.get("record_label")}
    r_rec, r_app, zs = roots_of(rec, N_app, v_app, R)
    x, status, source = choose_cut(a, rec, r_rec, r_app, g, rs)
    row["root"] = {"record_trapping_zeros": zs, "record_outer": r_rec, "record_R_t": rec.get("R_t"), "record_kappa_T": rec.get("kappa_T"), "applied_outer": r_app,
                   "record_minus_applied": (r_rec - r_app) if (r_rec is not None and r_app is not None) else None, "source": source, "R_cut": x, "R_cut_over_rs": (x / rs) if x is not None else None,
                   "status": status, "fields_source": applied_source, "fields_npz": Finfo}
    if x is None:
        row["status"] = status
        row["seconds"] = time.time() - t0
        IO.save_outputs(pj, pz, row, {"R": R, "N_applied": N_app, "v_applied": v_app})
        log("%s T/r_s %.3f (%s; %s): %s; %.0f s" % (TITLE, rd.T_over_rs, rd.dtype_str, entry["history"], status, row["seconds"]))
        return row
    A_lat = 4.0 * math.pi * x * x
    row["root"].update({"A_lat_cut": A_lat, "A_lat_nominal_64": A_LAT_NOMINAL_64, "eta_A_nats": float(a.ec_eta_A_nats),
                        "label": "X = {R > R_cut}; A/a^2 = 4 pi (R_cut/a)^2 at the actual applied root (G5), 16384 pi at R = 64 a beside"})
    cuts = [("root", float(x))] + [(lab, c_ * rs) for lab, c_ in IO.cuts_from_arg(a.ec_extra_cuts_rs)]
    row["cuts"] = {lab: xx for lab, xx in cuts}
    row["premises"] = {"gap_at_root": gap_block(R, N_app, v_app, x, m), "lapse_lower_bound": lapse_block(R, N_app, v_app, x, rec, g, F, Finfo),
                       "slowly_varying": eps_block(R, N_app, v_app, x, rs, a.ec_eps_window_rs, A_lat)}
    R2_t = torch.as_tensor(np.repeat(R, 2))
    arrays = {"R": R, "N_applied": N_app, "v_applied": v_app}
    kappas = rd.kappas if not str(a.ec_channels).strip() else [int(k_) for k_ in str(a.ec_channels).split(",") if k_.strip() and int(k_) in rd.kappas]
    chans = []
    wsum = {lab: {} for lab, _ in cuts}
    gaps_d, gaps_s = {}, {}
    for kap in kappas:
        out = channel_readout(rd, m, kap, N_app, v_app, cuts, R2_t, dev, a, log, arrays)
        chans.append(out)
        gaps_d[str(kap)] = out["vacuum"]["gap_at_zero_dense"]
        gaps_s[str(kap)] = out["vacuum"]["gap_at_zero_stored"]
        for lab, _ in cuts:
            accumulate(wsum[lab], out, lab, out["weight"])
        d = out["cuts"]["root"]
        tn = out["trace_norm"]
        fl = d.get("flat") or {}
        mr = d.get("mask_reference") or {}
        log("  kappa %d (w %g; sea %d, dust %d): vacuum %d occupied (eig vs stored %s; gap at zero %s; %.0f s); root cut %.3f: S_X(C) %.6f (complement gap %.1e, identity defect %s), S_X(P_-) %.6f, excess %+.6f <= t_X %s (%s) <= nu %s (rank %s, n_pair %s, ||.||_2 %s); d_X %d; flat S_X(P_0) %s, P_- against P_0 %s (t_X %s); mask k %s (S_X(C) - S_X(P_mask) %s, bridge %s); %.0f s" % (
            kap, out["weight"], out["n_sea"], out["n_dust"], out["vacuum"]["n_neg"], fx(out["vacuum"]["eig_vs_stored"], "%.1e"), fx(out["vacuum"]["gap_at_zero_dense"], "%.3e"), out["vacuum"]["seconds"], x, d["S_X"], d["purity_gap"],
            fx((d.get("pure_cut_identity") or {}).get("defect"), "%.1e"), d["S_X_vac"], d["excess"], fx(d.get("t_X"), "%.4e"), "ok" if d.get("bound_ok") else "FAIL", fx(tn.get("nu"), "%.4e"), fx(tn.get("rank"), "%d"),
            fx(tn.get("n_pair"), "%.3e"), fx(tn.get("frob"), "%.3e"), d["n_modes_X"], fx(fl.get("S_X_P0"), "%.6f"), fx(fl.get("dS_vac_minus_flat"), "%+.6f"), fx(fl.get("t_X_vac_flat"), "%.3e"),
            fx(out["mask_reference"].get("k_inverted_levels"), "%d"), fx(mr.get("excess_vs_mask"), "%+.6f"), fx(mr.get("bridge_sum"), "%.3e"), out["seconds"]))
    gp = row["premises"]["gap_at_root"]
    gp["channel_gap_at_zero_dense"] = gaps_d
    gp["channel_gap_at_zero_stored"] = gaps_s
    vals = sorted([(v_, k_) for k_, v_ in gaps_d.items() if v_ is not None])
    if vals:
        gp["channel_gap_min"], gp["channel_gap_min_kappa"] = float(vals[0][0]), int(vals[0][1])
    row["channels"] = chans
    row["weighted"] = {lab: dict(wsum[lab], weights="16 kappa", channels=list(kappas), sum_weights=float(sum(rd.weights[k_] for k_ in kappas))) for lab, _ in cuts}
    row["comparison"] = {lab: comparison(wsum[lab], 4.0 * math.pi * xx * xx, a.ec_eta_A_nats, rd) for lab, xx in cuts}
    row["seconds"] = time.time() - t0
    IO.save_outputs(pj, pz, row, arrays)
    w = row["weighted"]["root"]
    c = row["comparison"]["root"]
    lp = row["premises"]["lapse_lower_bound"]
    ep = row["premises"]["slowly_varying"]
    e4 = lp.get("ec4") or {}
    log("%s T/r_s %.3f (%s; %s): %s at R %.3f (%.3f r_s; A/a^2 %.1f): weighted excess %+.6f nats = %+.3e of eta A/a^2 %g (%+.3e per a^2) <= t_X %s <= nu %s; min{nu, r log 2, d_X log 2} %s; bridge over P_mask %s (sum w k %g); flat partial S_X(P_0) %s (P_- against P_0 %s); premises: N(root) %.4f, |v|(root) %.4f, 2 N m %.4f, channel gap at zero min %s (kappa %s), n_* %.4f at R %.1f (record N_min %s), eps(window) %.4e (L/r_s %s; f_loc %s; f_loc(a/r_s) %s), EC4 exp(-B) N_end %s; EC1 holds on every channel: %s; %.0f s" % (
        TITLE, rd.T_over_rs, rd.dtype_str, entry["history"], status, x, x / rs, A_lat, w.get("excess", 0.0), c.get("ratio_excess_to_eta_A") or 0.0, a.ec_eta_A_nats, c.get("excess_per_lattice_area") or 0.0,
        fx(w.get("t_X"), "%.4e"), fx(w.get("nu"), "%.4e"), fx(w.get("min_bound_term"), "%.4e"), fx(w.get("bridge_sum"), "%.4e"), w.get("k_inverted_weighted", 0.0), fx(w.get("S_X_P0"), "%.4f"),
        fx(w.get("dS_vac_minus_flat"), "%+.6f"), gp["N_root"], gp["abs_v_root"], gp["band_separation_lower_bound_2Nm"], fx(gp.get("channel_gap_min"), "%.3e"), fx(gp.get("channel_gap_min_kappa"), "%d"),
        lp["n_star_chain"], lp["at_R"], fx(lp.get("record_N_min"), "%.4f"), ep["eps_window"], fx(ep.get("L_over_rs_window"), "%.3f"), fx(ep.get("f_loc_eps_window"), "%.3e"), fx(ep.get("f_loc_a_over_rs"), "%.3e"),
        fx(e4.get("exp_minus_B_times_N_end"), "%.4f"), w.get("bound_ok_all"), row["seconds"]))
    return row


# ------------------------------------------------------------------------------------------------------------ the tables
def header_lines(a):
    return ["# %s: the evolved trapping cut on the stored slices (%s)" % (TITLE, CODE), "",
            "C18 (reorientation 4; the engine spec 55.4 item 9; the ruling II.4): per evolved channel at the outer root of each stored slice, S_X(C) - S_X(P_-) with the multiplicity 16 kappa, summed, against eta A/a^2 = %g nats (input 01); beside it hreduce's EVOLVEDCUT theorem's terms on the chain's channels and its premises on the records (inputs 03, 04). The reading is the lead's." % a.ec_eta_A_nats, "",
            "- the cut X = {R > R_root}: R_root is %s (the other beside it and used where it is absent); a clock without any root reads %s, labelled; A/a^2 = 4 pi (R_root/a)^2 at the actual applied root (16384 pi at R = 64 a)." % (
                "the record's outermost trapping zero" if a.ec_root_source == "record" else "the applied fields' outer root", "the cut at 2gE of the record" if str(a.ec_no_root_cut).lower() == "2ge" else ("skip" if str(a.ec_no_root_cut).lower() == "skip" else "the cut at %s r_s" % a.ec_no_root_cut)),
            "- S_X(C): Tr s(C_X) of the Loewdin-orthonormalized occupied columns (sea and dust) beyond the cut (page_d13_readout_v1b.exterior_entropies; the complement's S_Xc equal on the pure state, the purity gap the check); S_X(P_-): the held Killing vacuum 1(h_m < 0) at the slice's applied fields; excess = S_X(C) - S_X(P_-); the sums with 16 kappa.",
            "- EC1/G2: |excess| <= t_X = 2||Pi_X (C - P_-) Pi_Xc||_1 <= nu = ||C - P_-||_1 (cross_cut_norms, full_trace_norm); EC2/G3: r log 2 with r = rank(C - P_-), d_X log 2 with d_X the complex modes beyond the cut; min{nu, r log 2, d_X log 2}; n_pair = Tr[(1 - P_-) C] is EC2's comparator, not a bound; the pure-cut identity S_X(C) = sum F(sigma(Pi_X C Pi_Xc)) is the arithmetic's check.",
            "- P_0: the flat sea 1(h_0 < 0) at N = 1, v = 0 on the same chain and cut; S_X(P_-) - S_X(P_0) and 2||Pi_X (P_- - P_0) Pi_Xc||_1 (EC5's reference term on the chain's channels). The weighted S_X(P_0), S_X(P_-), S_X(C) are the stored channels' partial sums in the chain's reduction, not the 3D cut's values (EC1, G6).",
            "- P_mask: the record's continued occupation (the stored mask over the channel's Killing eigenlevels; k inverted levels against 1(e_k < 0)); ||P_mask - P_-||_1 = k; G6.1's bridge |S_X(C) - S_X(P_-)| <= 2||Pi_X (C - P_mask) Pi_Xc||_1 + 2||Pi_X (P_mask - P_-) Pi_Xc||_1 with its second term <= k log 2 (EC6). Whether P_mask is the theorem's compatible continued reference is the lead's and hreduce's reading.",
            "- the premises (EC3/G1, EC4/G5a, EC5/G5): the local gap at the root 2 N(root) m and the conservative zero distance; the channels' own gaps at zero (not the global pathwise delta); n_* = min N (chain, inside, beyond the root; the record's N_min); EC4's profile bound where the fields npz carries J_tot (its convention check beside); eps = a/L from the applied fields' derivatives (the window %g r_s around the root; the chain), f_loc(eps) = eps^2 (1 + |log eps|)^4 which multiplies the unsupplied C_loc, f_loc(a/r_s) beside; f_exc = 1/A_lat." % a.ec_eps_window_rs,
            "- complex64 slices carry the driver's floor (sources %s to %s relative, orthonormality %s; a complex128 slice %s)." % tuple(["%.1e" % v_ for v_ in IO.C64_FLOOR["source_rel_range"]] + ["%.1e" % IO.C64_FLOOR["orthonormality_defect"], "%.1e" % IO.C64_FLOOR["complex128_reference"]]), ""]


def compose(a, hists):
    rows = IO.load_rows(a, READOUT, hists)
    if not rows:
        return
    d_out = os.path.join(a.out, READOUT)
    all_lines = []
    for hk in hists:
        rs_ = [r for r in rows if r["history"] == hk]
        if not rs_:
            continue
        L = header_lines(a)
        L += ["## %s: %s" % (hk, rs_[0]["history_label"]), "", "Engine: %s" % rs_[0]["engine"], "", "### The evolved cut at the root (summed with 16 kappa)", "",
              "| T/r_s | dtype | E/E0 | cut R/r_s (status; applied root) | A/a^2 | S_X(C) | S_X(P_-) | excess | excess / eta A | excess per a^2 | t_X | nu | r log 2 | d_X log 2 | min{nu, r log 2, d_X log 2} | S_X(P_0) partial | S_X(P_-) - S_X(P_0) (t_X) | P_mask: sum w k; S_X(C) - S_X(P_mask); bridge | EC1 ok | purity gap |",
              "|" + "---|" * 20]
        for r in rs_:
            T = r["T_over_rs"]
            if "weighted" not in r:
                L.append("| %.3f | %s | %s | %s |%s" % (T, r["dtype"], fx(r.get("record", {}).get("E_over_E0"), "%.6f"), r.get("status") or (r.get("root") or {}).get("status"), " - |" * 16))
                continue
            w = r["weighted"]["root"]
            c = r["comparison"]["root"]
            ro = r["root"]
            L.append("| %.3f | %s | %s | %.4f (%s; applied %s) | %.1f | %.6f | %.6f | %+.6f | %+.3e | %+.3e | %s | %s | %s | %.4e | %s | %s | %s (%s) | %g; %s; %s | %s | %.1e |" % (
                T, r["dtype"], fx(r["record"].get("E_over_E0"), "%.6f"), ro["R_cut_over_rs"], ro["status"], fx((ro["applied_outer"] / r["rs"]) if ro.get("applied_outer") is not None else None, "%.4f"),
                ro["A_lat_cut"], w.get("S_X", 0.0), w.get("S_X_vac", 0.0), w.get("excess", 0.0), c.get("ratio_excess_to_eta_A") or 0.0, c.get("excess_per_lattice_area") or 0.0, fx(w.get("t_X"), "%.4e"),
                fx(w.get("nu"), "%.4e"), fx(w.get("rank_log2"), "%.4e"), w.get("d_X_log2", 0.0), fx(w.get("min_bound_term"), "%.4e"), fx(w.get("S_X_P0"), "%.6f"),
                fx(w.get("dS_vac_minus_flat"), "%+.6f"), fx(w.get("t_X_vac_flat"), "%.4e"), w.get("k_inverted_weighted", 0.0), fx(w.get("excess_vs_mask"), "%+.6f"), fx(w.get("bridge_sum"), "%.4e"),
                w.get("bound_ok_all"), w.get("max_abs_purity_gap", 0.0)))
        L += ["", "### The theorem's premises on the records (the applied fields of the slice; lattice units a = 1)", "",
              "| T/r_s | N(root); abs v(root) (N - abs v) | 2 N m; zero distance | channel gap at zero min (kappa) | n_* chain (R); inside; beyond | record N_min; min(N - abs v) | eps: N'; v' near the root (chain) | eps_2 | L/r_s; L_2 (a) | f_loc(eps) | f_loc(a/r_s) | f_exc | EC4 exp(-B) N_end (convention ratio) | n_pair; n_hole |",
              "|" + "---|" * 14]
        for r in rs_:
            if "weighted" not in r:
                continue
            T = r["T_over_rs"]
            w = r["weighted"]["root"]
            gp = r["premises"]["gap_at_root"]
            lp = r["premises"]["lapse_lower_bound"]
            ep = r["premises"]["slowly_varying"]
            e4 = lp.get("ec4") or {}
            L.append("| %.3f | %.4f; %.4f (%+.1e) | %.4f; %.4e | %s (kappa %s) | %.4f (R %.1f); %s; %s | %s; %s | %.3e; %.3e (%.3e; %.3e) | %.3e | %s; %s | %s | %s | %.3e | %s (%s) | %s; %s |" % (
                T, gp["N_root"], gp["abs_v_root"], gp["N_minus_absv_root"], gp["band_separation_lower_bound_2Nm"], gp["zero_distance_conservative"], fx(gp.get("channel_gap_min"), "%.4e"), fx(gp.get("channel_gap_min_kappa"), "%d"),
                lp["n_star_chain"], lp["at_R"], fx(lp.get("n_star_inside_root"), "%.4f"), fx(lp.get("n_star_beyond_root"), "%.4f"), fx(lp.get("record_N_min"), "%.4f"), fx(lp.get("record_min_N_minus_absv"), "%+.5f"),
                ep["fields"]["N"]["d1_max_window"] or 0.0, ep["fields"]["v"]["d1_max_window"] or 0.0, ep["fields"]["N"]["d1_max_chain"], ep["fields"]["v"]["d1_max_chain"], ep["eps2_window"],
                fx(ep.get("L_over_rs_window"), "%.3f"), fx(ep["fields"]["N"].get("L2_window"), "%.1f"), fx(ep.get("f_loc_eps_window"), "%.3e"), fx(ep.get("f_loc_a_over_rs"), "%.3e"), ep["f_exc_A_lat_cut"],
                fx(e4.get("exp_minus_B_times_N_end"), "%.4f"), fx(e4.get("convention_check_median_ratio"), "%.3f"), fx(w.get("n_pair"), "%.4e"), fx(w.get("n_hole"), "%.4e")))
        L += ["", "### Per channel at the root cut (unweighted; the multiplicity 16 kappa applies to the sums above)", "",
              "| T/r_s | kappa | n_dust | S_X(C) | S_X(P_-) | excess | t_X | nu | rank | n_pair | gap at zero | S_X(P_0) | P_mask: k; S_X(C) - S_X(P_mask); bridge | identity defect | Loewdin (stored) |",
              "|" + "---|" * 15]
        for r in rs_:
            for c in (r.get("channels") or []):
                d = c["cuts"]["root"]
                tn = c["trace_norm"]
                mr = d.get("mask_reference") or {}
                L.append("| %.3f | %d | %d | %.6f | %.6f | %+.6f | %s | %s | %s | %s | %s | %s | %s; %s; %s | %s | %s (%.1e) |" % (
                    r["T_over_rs"], c["kappa"], c["n_dust"], d["S_X"], d["S_X_vac"], d["excess"], fx(d.get("t_X"), "%.4e"), fx(tn.get("nu"), "%.4e"), fx(tn.get("rank"), "%d"), fx(tn.get("n_pair"), "%.4e"),
                    fx(c["vacuum"].get("gap_at_zero_dense"), "%.4e"), fx((d.get("flat") or {}).get("S_X_P0"), "%.6f"), fx(c["mask_reference"].get("k_inverted_levels"), "%d"), fx(mr.get("excess_vs_mask"), "%+.6f"),
                    fx(mr.get("bridge_sum"), "%.4e"), fx((d.get("pure_cut_identity") or {}).get("defect"), "%.1e"), fx(c["loewdin_orthonormality"], "%.1e"), c["stored_orthonormality"]))
        L += ["", "### Checks", ""]
        for r in rs_:
            if "weighted" not in r:
                L.append("- T/r_s %.3f: %s." % (r["T_over_rs"], r.get("status") or (r.get("root") or {}).get("status")))
                continue
            w = r["weighted"]["root"]
            chs = r.get("channels") or []
            pdef = [c["premise_defects"] for c in chs if c.get("premise_defects")]
            L.append("- T/r_s %.3f: EC1 holds on every channel: %s (smallest slack t_X - |S_X(C) - S_X(P_-)| %s); the pure-cut identity's largest defect %s; purity gap max %.1e; vacuum eigenvalues against the stored e_k max %s; Loewdin after %s (stored %s); premises' defects max ||Phi^dag Phi - 1||_F %s, ||V^dag V - 1||_F %s; vacuum occupied counts %s against N_occ %s (dust in %d channels); the flat pair's EC1 ok %s; P_mask: %d channels with inversions (sum w k %g; degenerate clusters split %s), bridge and k log 2 ok %s; %s; %.0f s." % (
                r["T_over_rs"], w.get("bound_ok_all"), fx(w.get("min_slack"), "%.3e"), fx(w.get("max_abs_identity_defect"), "%.1e"), w.get("max_abs_purity_gap", 0.0),
                fx(max([c["vacuum"]["eig_vs_stored"] for c in chs if c["vacuum"].get("eig_vs_stored") is not None] or [None]), "%.1e") if any(c["vacuum"].get("eig_vs_stored") is not None for c in chs) else "-",
                fx(max([(c["loewdin_orthonormality"] or 0.0) for c in chs] or [0.0]), "%.1e"), fx(max([c["stored_orthonormality"] for c in chs] or [0.0]), "%.1e"),
                fx(max([p_["Phi"] for p_ in pdef if p_.get("Phi") is not None] or [None]), "%.1e") if pdef else "-", fx(max([p_["Vneg"] for p_ in pdef if p_.get("Vneg") is not None] or [None]), "%.1e") if pdef else "-",
                sorted({c["vacuum"]["n_neg"] for c in chs}), sorted({c["N_occ"] for c in chs}), w.get("channels_with_dust", 0), w.get("flat_bound_ok_all", "-"), w.get("channels_with_inversions", 0),
                w.get("k_inverted_weighted", 0.0), any((c["mask_reference"].get("degenerate_clusters_split_by_mask") or []) for c in chs), w.get("bridge_ok_all", "-"),
                "complex64 slice: floor %s" % r["floor"]["source_rel_range"] if r["complex64"] else "complex128", r["seconds"]))
        L += ["", "- every number above is printed by the script from the stored slices and the records; the reading is the lead's."]
        with open(os.path.join(d_out, "%s_%s.md" % (TITLE, hk)), "w") as f:
            f.write("\n".join(L) + "\n")
        all_lines.append(hk)
    present = [hk for hk in hists if any(r["history"] == hk for r in rows)]
    if len(present) >= 2:
        clocks = sorted({round(r["T_over_rs"], 4) for r in rows})
        L = header_lines(a) + ["## The histories at matched clocks (the root cut, summed with 16 kappa)", "",
                               "| T/r_s | " + " | ".join("%s: excess (of eta A); t_X; nu; cut R/r_s" % hk for hk in present) + " |", "|" + "---|" * (1 + len(present))]
        for t in clocks:
            cells = []
            n_have = 0
            for hk in present:
                rr = [r for r in rows if r["history"] == hk and abs(r["T_over_rs"] - t) < 1e-6]
                if not rr:
                    cells.append("-")
                    continue
                r = rr[0]
                if "weighted" not in r:
                    cells.append(str(r.get("status") or "no root"))
                    continue
                n_have += 1
                w, c = r["weighted"]["root"], r["comparison"]["root"]
                cells.append("%+.6f (%+.3e); %s; %s; %.4f" % (w.get("excess", 0.0), c.get("ratio_excess_to_eta_A") or 0.0, fx(w.get("t_X"), "%.3e"), fx(w.get("nu"), "%.3e"), r["root"]["R_cut_over_rs"]))
            if n_have >= 1:
                L.append("| %.3f | %s |" % (t, " | ".join(cells)))
        L += ["", "- the ladder across sigma/a (the production's 4, the member's 8, rung 2's 2) and the relaunch at the clocks they share; every number from the stored slices; the reading is the lead's."]
        with open(os.path.join(d_out, "%s_ladder.md" % TITLE), "w") as f:
            f.write("\n".join(L) + "\n")
    with open(os.path.join(d_out, "%s_all.json" % TITLE), "w") as f:
        json.dump({"code": CODE, "title": TITLE, "c18": C18, "histories": hists, "histories_present": present, "rows": rows}, f, default=IO.jdefault, indent=1)


# ------------------------------------------------------------------------------------------------------------ the self-test
def self_test(a):
    """manufactured slices through the same functions: (A) the closed form on a site-localized reference (S_X(P) = 0 exactly): C = P gives the
    excess 0, nu 0, t_X 0 (also through a unitary mixing of the columns); one occupied mode y in Xc rotated into an empty mode x in X with
    weight p gives S_X(C) - S_X(P) = h_2(p), nu = 2 sqrt p, t_X = 2 sqrt(p (1 - p)), rank 2, n_pair = p (EC2/G2's example); (B) the chain: the
    flat chain's vacuum written as a slice in the stored format and read by slices144_io_v1.SliceReader through process_slice: the excess 0
    to rounding and P_- = P_0; the one-mode rotation of the highest occupied with the lowest empty Killing level (nu = 2 sqrt p exactly, rank
    2, n_pair = p; the excess bounded by t_X and by 2 log 2); a two-level mask inversion (||P_mask - P_-||_1 = 2; C = P_mask; G6.1's bridge)."""
    out_dir = os.path.join(a.out, "selftest")
    os.makedirs(out_dir, exist_ok=True)
    log = IO.Logger(os.path.join(out_dir, "evolvedcut_selftest.log"))
    IO.setup(a.engine_code, a.page_code, a.threads, log)
    log("%s self-test: setup %s" % (CODE, json.dumps(IO.SETUP, default=str)))
    if IO.D13 is None:
        raise RuntimeError("the self-test needs page_d13_readout_v1b: %s" % IO.SETUP.get("D13"))
    torch = IO.torch
    dev = torch.device("cpu")
    a.out = out_dir
    a.ec_full_norm = "svd"
    a.ec_block_route = "svd"
    res = []

    def check(name, value, expected, tol):
        ok = value is not None and abs(float(value) - float(expected)) <= tol
        res.append({"name": name, "value": value, "expected": expected, "tol": tol, "ok": bool(ok)})
        log("  %s: %s against %.12e (tol %.1e): %s" % (name, fx(value, "%.12e"), expected, tol, "ok" if ok else "FAIL"))

    def check_le(name, value, bound, tol):
        ok = value is not None and bound is not None and float(value) <= float(bound) + tol
        res.append({"name": name, "value": value, "bound": bound, "tol": tol, "ok": bool(ok)})
        log("  %s: %s <= %s (tol %.1e): %s" % (name, fx(value, "%.12e"), fx(bound, "%.12e"), tol, "ok" if ok else "FAIL"))
    p = float(a.ec_self_test_p)
    log("(A) the closed form: 6 sites (12 modes), the cut between sites 3 and 4, the reference P = 5 site modes (3 in Xc, 2 in X): S_X(P) = 0 exactly; the full norm and the block by svd")
    R = np.arange(1.0, 7.0)
    R2 = torch.as_tensor(np.repeat(R, 2))
    cuts = [("root", 3.5)]
    occ = [0, 1, 2, 6, 7]
    V = torch.zeros((12, 5), dtype=torch.complex128)
    for j, i in enumerate(occ):
        V[i, j] = 1.0
    r0 = pair_readout(V, V, R2, cuts, a)
    d, tn = r0["cuts"]["root"], r0["trace_norm"]
    check("A0 excess (C = P)", d["excess"], 0.0, 1e-12)
    check("A0 S_X(P)", d["S_X_vac"], 0.0, 1e-12)
    check("A0 nu = ||C - P||_1", tn["nu"], 0.0, 1e-12)
    check("A0 t_X", d["t_X"], 0.0, 1e-12)
    check("A0 rank(C - P)", tn["rank"], 0, 0)
    gen = torch.Generator().manual_seed(1)
    Z = torch.complex(torch.randn(5, 5, generator=gen, dtype=torch.float64), torch.randn(5, 5, generator=gen, dtype=torch.float64))
    U, _ = torch.linalg.qr(Z)
    r1 = pair_readout(V @ U, V, R2, cuts, a)
    check("A1 excess (the same C through a unitary mixing of the columns)", r1["cuts"]["root"]["excess"], 0.0, 1e-12)
    check("A1 nu", r1["trace_norm"]["nu"], 0.0, 1e-10)
    check("A1 identity defect", r1["cuts"]["root"]["pure_cut_identity"]["defect"], 0.0, 1e-12)
    Phi = V.clone()
    Phi[:, 2] = 0.0
    Phi[2, 2] = math.sqrt(1.0 - p)
    Phi[9, 2] = math.sqrt(p)
    r2 = pair_readout(Phi, V, R2, cuts, a)
    d, tn = r2["cuts"]["root"], r2["trace_norm"]
    check("A2 excess = h_2(p)", d["excess"], h2(p), 1e-12)
    check("A2 S_X(C) = h_2(p)", d["S_X"], h2(p), 1e-12)
    check("A2 S_X(P) = 0", d["S_X_vac"], 0.0, 1e-12)
    check("A2 nu = 2 sqrt p", tn["nu"], 2.0 * math.sqrt(p), 1e-10)
    check("A2 t_X = 2 sqrt(p (1 - p))", d["t_X"], 2.0 * math.sqrt(p * (1.0 - p)), 1e-10)
    check("A2 rank(C - P) = 2", tn["rank"], 2, 0)
    check("A2 n_pair = p", tn["n_pair"], p, 1e-12)
    check_le("A2 EC1: |excess| <= t_X", abs(d["excess"]), d["t_X"], 1e-12)
    check_le("A2 EC1: t_X <= nu", d["t_X"], tn["nu"], 1e-12)
    check("A2 identity S_X(C) = sum F(sigma(K_C))", d["pure_cut_identity"]["defect"], 0.0, 1e-12)
    log("(B) the chain: %d sites, kappa 1, m a %g, N = 1, v = 0; the slices written in the stored format and read by slices144_io_v1.SliceReader through process_slice" % (a.ec_self_test_sites, a.ec_self_test_m))
    n = int(a.ec_self_test_sites)
    m = float(a.ec_self_test_m)
    g = float(a.ec_self_test_g)
    rs = 8.0
    kap = 1
    Rn = np.arange(1.0, n + 1.0)
    x_root = float(n // 2) - 0.5
    tmp = tempfile.mkdtemp(prefix="evolvedcut_selftest_", dir=out_dir)
    ones = np.ones(n)
    zeros = np.zeros(n)
    h0, Q, H = IO.operators(Rn, m, kap, ones, zeros)
    e_t, Vt = IO.OPS.eigh_csr(H, dev)
    e_np = np.asarray(e_t.cpu().numpy(), float)
    Vnp = np.asarray(Vt.cpu().numpy(), np.complex128)
    neg = e_np < 0.0
    n_neg = int(neg.sum())
    j = int(np.nonzero(neg)[0][-1])
    u = int(np.nonzero(~neg)[0][0])
    tag = "selftest"
    rec_path = os.path.join(tmp, "collapse_selftest.json")
    recs = []
    for t_, lab_, meq, inv in ((2.0, "vacuum", True, 0), (3.0, "one_mode", True, 0), (4.0, "mask_inversion", False, 2)):
        recs.append({"T": t_ * rs, "T_over_rs": t_, "E": rs / (2.0 * g), "E_over_E0": 1.0, "trapping_zeros": [x_root], "R_t": x_root, "min_N_minus_absv": 1.0, "converged": True,
                     "mask_equals_negative_set": meq, "inversions_total": inv, "engine": "manufactured (self-test: %s)" % lab_})
    with open(rec_path, "w") as f:
        json.dump({"args": {"m": m, "g": g, "rs": rs, "ell": 4.0, "nsites": n, "tau": 8.0}, "records": recs}, f)
    Hh = {"key": tag, "label": "manufactured slices (self-test): the flat chain's vacuum as the sea; a one-mode rotation; a two-level mask inversion", "tag": tag, "nsites": n, "rs": rs,
          "sigma_over_a": 4.0, "engine": "manufactured (self-test)", "fields_module": None, "records_json": rec_path, "fields_npz": None, "data_npz": None, "slices": []}
    ctx = IO.HistoryContext(Hh, a, log)

    def run_slice(name, sea, mask, T_over_rs):
        path = os.path.join(tmp, IO.slice_name(tag, T_over_rs))
        np.savez(path, T=T_over_rs * rs, rs=rs, R=Rn, N_applied=ones, v_applied=zeros, kappas=np.array([kap]), weights=np.array([16.0 * kap]), dtype="complex128",
                 sea_1=np.asarray(sea, np.complex128), dust_1=np.zeros((2 * n, 0), np.complex128), f_1=np.zeros(0), R2_1=np.zeros((2 * n, n), np.complex128), e_1=e_np,
                 mask_1=np.packbits(np.asarray(mask, bool)))
        entry = {"history": tag, "history_label": Hh["label"], "tag": tag, "T_over_rs": T_over_rs, "T": T_over_rs * rs, "dtype": "c128", "nsites": n, "expected_size": None, "box_paths": [path],
                 "hf_path": None, "sha256_prefix": None, "sha_source": None, "records_json": rec_path, "fields_npz": None, "engine": "manufactured (self-test: %s)" % name, "fields_module": None,
                 "note": name, "name": os.path.basename(path)}
        rd = IO.SliceReader(path, entry, log)
        row = process_slice(rd, ctx, entry, a)
        rd.close()
        return row
    row = run_slice("vacuum", Vnp[:, neg], neg, 2.0)
    c = row["channels"][0]
    d, tn = c["cuts"]["root"], c["trace_norm"]
    check("B0 excess (the sea = the vacuum)", d["excess"], 0.0, 1e-9)
    check("B0 nu", tn["nu"], 0.0, 1e-9)
    check("B0 t_X", d["t_X"], 0.0, 1e-9)
    check("B0 identity defect", d["pure_cut_identity"]["defect"], 0.0, 1e-9)
    check("B0 eigenvalues against the stored e_k", c["vacuum"]["eig_vs_stored"], 0.0, 1e-10)
    check("B0 S_X(P_-) - S_X(P_0) (N = 1, v = 0: the same operator)", d["flat"]["dS_vac_minus_flat"], 0.0, 1e-9)
    check("B0 t_X(P_-, P_0)", d["flat"]["t_X_vac_flat"], 0.0, 1e-9)
    check("B0 k inverted levels", c["mask_reference"]["k_inverted_levels"], 0, 0)
    check("B0 weighted excess", row["weighted"]["root"]["excess"], 0.0, 1e-8)
    check("B0 ratio to eta A", row["comparison"]["root"]["ratio_excess_to_eta_A"], 0.0, 1e-12)
    check("B0 N(root) on the manufactured fields", row["premises"]["gap_at_root"]["N_root"], 1.0, 0.0)
    sea1 = Vnp[:, neg].copy()
    sea1[:, n_neg - 1] = math.sqrt(1.0 - p) * Vnp[:, j] + math.sqrt(p) * Vnp[:, u]
    row = run_slice("one_mode", sea1, neg, 3.0)
    c = row["channels"][0]
    d, tn = c["cuts"]["root"], c["trace_norm"]
    check("B1 nu = 2 sqrt p (the highest occupied rotated into the lowest empty level)", tn["nu"], 2.0 * math.sqrt(p), 1e-9)
    check("B1 rank(C - P_-) = 2", tn["rank"], 2, 0)
    check("B1 n_pair = p", tn["n_pair"], p, 1e-10)
    check_le("B1 EC1: |excess| <= t_X", abs(d["excess"]), d["t_X"], 1e-9)
    check_le("B1 EC1: t_X <= nu", d["t_X"], tn["nu"], 1e-9)
    check_le("B1 EC2: |excess| <= 2 log 2 (a particle-hole rotation)", abs(d["excess"]), 2.0 * LOG2, 0.0)
    check("B1 identity defect", d["pure_cut_identity"]["defect"], 0.0, 1e-9)
    log("  B1 measured excess %+.6e (no closed form on the chain: bounded by t_X %.6e and by 2 log 2)" % (d["excess"], d["t_X"]))
    mask2 = neg.copy()
    mask2[j] = False
    mask2[u] = True
    row = run_slice("mask_inversion", Vnp[:, mask2], mask2, 4.0)
    c = row["channels"][0]
    d, tn = c["cuts"]["root"], c["trace_norm"]
    mr, dm = c["mask_reference"], d["mask_reference"]
    check("B2 k inverted levels", mr["k_inverted_levels"], 2, 0)
    check("B2 ||P_mask - P_-||_1 = k", mr["nu_mask_vac"], 2.0, 1e-9)
    check("B2 nu(C, P_-) with C = P_mask", tn["nu"], 2.0, 1e-9)
    check("B2 S_X(C) - S_X(P_mask) (C = P_mask)", dm["excess_vs_mask"], 0.0, 1e-9)
    check("B2 t_X(C, P_mask)", dm["t_X_C_mask"], 0.0, 1e-9)
    check_le("B2 G6.1 bridge: |S_X(C) - S_X(P_-)| <= t_X(C, P_mask) + t_X(P_mask, P_-)", abs(d["excess"]), dm["bridge_sum"], 1e-9)
    check_le("B2 EC2/EC6: |S_X(P_mask) - S_X(P_-)| <= k log 2", abs(dm["dS_mask_minus_vac"]), 2.0 * LOG2, 0.0)
    check("B2 degenerate clusters split by the mask (count)", len(mr.get("degenerate_clusters_split_by_mask") or []), 0, 0)
    all_ok = all(r_["ok"] for r_ in res)
    out = {"code": CODE, "title": TITLE, "all_ok": all_ok, "n_checks": len(res), "n_fail": sum(1 for r_ in res if not r_["ok"]), "p": p, "sites": n, "m": m, "g": g, "results": res,
           "routes": {"full_norm": a.ec_full_norm, "block_route": a.ec_block_route}, "setup": IO.SETUP, "slices_dir": tmp}
    with open(os.path.join(out_dir, "EVOLVEDCUT_selftest.json"), "w") as f:
        json.dump(out, f, default=IO.jdefault, indent=1)
    log("self-test: %d checks, %d FAIL: %s (%s)" % (len(res), out["n_fail"], "PASS" if all_ok else "FAIL", os.path.join(out_dir, "EVOLVEDCUT_selftest.json")))
    return 0 if all_ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    IO.add_common_arguments(ap)
    add_arguments(ap)
    a = ap.parse_args(argv)
    if a.ec_self_test:
        return self_test(a)
    a.readouts = READOUT
    IO.run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
