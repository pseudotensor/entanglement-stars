#!/usr/bin/env python
"""core_well_v1.py (reviewer A for the solve builder, the busi lane). The final plan's item 1 (inbox #162), word for word:
"CORE_well_v1 (box CPU, now, before any new member; it subsumes CORE_kinematics_v1): first name in 1 line which slices of the r_s/a 64,
tau 8, sigma/a 4 history hold stored states (the r11 archive at 12, 15, 16, 20, 24, 25, 28, 30, 35 and 40; the rung reruns on v36d store
states every 0.5 r_s from 11 to 16, which supply 12.5 to 14 where the archive has none); then on the stored slices at the bounce that exist
and on 15 and 16, and on the sigma/a 8 and sigma/a 2 members' slices where they exist, from the stored dense and applied fields and the
stored states, per engine channel without multiplicity on any printed correlation: N(r), v(r), w = v/N, the dust density n(r), the bare
energy density rho(r) and rho/n, the 2-step current density J(r), the enclosed dust count N_<(r), the mass function M(r) = r w^2/(2G) and
its match to the integral of 4 pi r^2 (rho + w J); the slope of ln N against ln N_<(r) inside the core with the fitted exponent beside
(epsilon_c - m a)/(2 m a) and beside epsilon_c/(2 m a), epsilon_c the measured a rho/n at the core; the clock's gradient length N/|N'|
against sigma; the dust's radial-momentum distribution per shell and its share at |a k_r| > pi/4; the Killing spectrum of the dust orbitals
split at 0, m and b = sqrt(m^2 + 1/a^2), per slice and per shell; at T/r_s 16 the exact range-2 dust number current through the applied
outer root (48.5364 a) and through the dense root (50.8371 a) from the stored dust columns, occupations and applied fields, straddling bonds
only, per channel with the declared multiplicities counted once (the one-slice computation both reviewer B routes selected), and the same at 15
and at every stored slice holding a root; the fountain's reach R_90(t) against r_s/(1 - (omega/m)^2) for the bound part; and the root
attribution: between consecutive records holding a dense root (12.125 to 40 on the dense fields; the sigma/a 2 member's 12.0 to 18.125),
Delta M at the root's radius split into the step's collar deficit at that clock, the matter's flux of rho + w J through that radius from the
stored states where they exist with the momentum split at pi/4, and the root's own motion (the sweep term), with each root's radius over
the record's 2GE; everything labelled past the record's own limit; the step scheme the records used named in 1 line (frozen-field
exponential Euler or the midpoint of Eq. dr-midpoint)."

The stored states (input 01, SLICE_INVENTORY_r64_v1; input 02, the builder's inventory of the v36d rungs): the plan's half-r_s states from
11 to 16 do not exist; the 1-line inventory printed at the head of every md says what does (INVENTORY_LINE below).

Where this module sits. It is a readout module of the slices144_io_v1 framework (add_arguments, is_done, process_slice(rd, ctx, entry, a),
compose(a, hists); READOUT 'well'): the reader, the records context, the fetch and the output conventions are the framework's, and the
momentum conventions are core_kinematics_v2's (imported by name: the reduced zone p = k mod pi, its moments). It carries its own driver
loop (run) instead of calling slices144_io_v1.run because two things of item 1 the framework's loop cannot carry: (1) the root attribution
is a pass over every record between the slices (a per-history pass the driver has no hook for; it runs inside compose here, so --compose-only
recomputes it from the jsons present), and (2) the sigma/a 2 member and the v36d rungs are not in slices144_io_v1.inventory and may have no
records json (--extra-history adds them; a history without a records json runs with NoRecordsContext and prints 'no records json' where a
record quantity is asked for). Run through the framework driver (--readouts well) the module behaves identically on the 4 inventory histories.

Per slice (the stored state at the record's clock; the record from the history's records json; the record's stored fields from the fields npz:
eps_tot, J_tot, eps_pairs, J_pairs per site, the dense solution's nodes r_nodes, N_nodes, v_nodes, the applied site fields), per engine
channel kappa without multiplicity on any printed correlation (the additive sums with the slice's declared multiplicities 16 kappa appear only
on counts and currents, labelled):
- the fields: N, v, w = v/N on the stored dense nodes interpolated to the sites and on the applied site fields; the roots: the record's
  trapping zeros (the dense roots) and the applied roots (the zeros of N_bar + v_bar between sites, dust_root_history_v2b's finder); the gate
  sigma |d_r v| on the applied fields against --gate (0.1) gives the label 'past the record's own limit' (computed, never asserted);
- the dust: the number per site from the stored dust columns (n(r) = the shell total over 4 pi r^2 a), the enclosed count N_<(r), R_50, R_90;
  rho(r) = eps_tot/(4 pi r^2 a) (the records' h_0 source, the bare density; eps_tot - eps_pairs, the dust's part, beside), rho/n per shell,
  J(r) = J_tot/(4 pi r^2 a) (the 2-step current density conjugate to v);
- the mass function M(r) = r w^2/(2 g) on the dense and on the applied w, against I(r) = sum_{x <= r} (eps_tot + w J_tot)(x) (the shell
  totals carry the 4 pi r^2 a; a = 1) and against the collar form E - sum_{x > r}; the mismatch profile, its largest value inside R_c, the
  values at the roots; the record's W_over_2g_at_Rt and misner_sharp_at_Rt beside as the convention check;
- the core: ln N against ln N_<(r) on the sites of the core window (--core-window: R <= R_50 by default; R_90, the inner dense root and R_c
  beside) with N_<(r) >= --fit-min-count, the least-squares slope with its standard error, beside (epsilon_c - m)/(2 m) and epsilon_c/(2 m)
  with epsilon_c = sum eps_tot/sum n over the window (a = 1; the dust part's epsilon_c beside), the local slope profile saved;
- the clock's gradient length L_N = N/|N'| (dense nodes, applied sites) against sigma = ell: the minimum over the chain and inside the outer
  root, the values at the roots, the number of sites with L_N < sigma;
- the shells (--shell-sites 128 a = 2 r_s): per shell the dust's radial-momentum distribution n_S(k) (core_kinematics_v2.window_momentum, the
  2 spinor components summed) read in the reduced zone, its share at |ap| > pi/4 (the counter-advected content), <cos 2ak>; and the Killing
  content Re Tr[Pi_S P_bin D] of the 6 bins split at 0, +-m and +-b (half-open (lo, hi]), additive over shells and bins;
- the Killing spectrum: h_m at the slice's applied fields (page_ops_v6) diagonalized per channel with dust (its eigenvalues against the stored
  e_k, the check), the dust's occupation per level, the 6 bins' numbers and energies, the bound part (--bound-lo m < omega <= m): its number,
  R_50, R_90 (of the bound-projected columns' density), its mean omega/m and the turning radius r_s/(1 - (omega/m)^2) with the nominal r_s and
  with the record's 2gE; per dust orbital the mean Killing energy, spread, bins, bound weight and reach;
- the currents through the cuts (the dense outer and inner roots, the applied outer and inner roots, the previous and the next record's dense
  outer root, and at T/r_s 16 of the production the plan's quoted radii 48.5364 a and 50.8371 a): the dust's number current (page_ops_v6.
  number_current_csr, Tr[D i[h_m, Pi_out]], outward positive) and the same by the explicit straddling-bond sum 2 sum_{a in, b out} Im(h_ba D_ab)
  over the bonds crossing the cut (the range-2 check, their count and range printed), the dust's Killing-energy current ((i/2)[h_m^2, Pi_out]),
  the flux of rho + w J through the cut read as the Killing-energy current over N_bar at the cut (F_M; eps_tot + w J_tot = eps_K/N per shell),
  each split by the dust's momentum at pi/4 (the columns projected on |ap| <= pi/4 and its complement; the cross term printed), the occupied
  set C (Loewdin-orthonormalized), the reference R2 R2^dagger and the deviation C - R2 R2^dagger (the matter: the dust and the pairs); the
  16 kappa sums with the multiplicities counted once.
The records pass (compose; every record of the history with a stored root): the dense fields' M(r) match and L_N per record; between
consecutive records both holding a dense outer root, Delta M = (R_t(k) - R_t(k-1))/(2 g) (w = -1 at a root) split as the fixed-radius
change M(R_t(k-1), t_k) - R_t(k-1)/(2 g) plus the sweep term M(R_t(k), t_k) - M(R_t(k-1), t_k) (the dense fields at t_k; the integral form
and the other ordering beside), the fixed-radius change read against the step's collar deficit Delta E (the record's E) and the inward flux
of rho + w J over the step from the stored states where a slice exists at either end (the slice's F_M at the dense outer root, or the next
slice's F_M at the previous record's root; the dust's share and its momentum split beside), the residual printed; R_t/(2gE), R_in/(2gE);
the label per record. The step scheme is named in 1 line from the records' solver key, the fields npz's solver and label, the records'
args and the engine source lines matched on this box (the rule printed).
Outputs: well_<history>_T<clock>.json, .npz, .md per slice; records_<history>.json/.npz; CORE_well_v1_<history>.md; CORE_well_v1.md (the
summary with the slice format for the hreduce lane, SECONDROUTE_CORE_well_v1); CORE_well_v1_all.json. CPU only; no plotting; no mutual
information; no dates (elapsed seconds only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slices144_io_v1 as IO  # noqa: E402  (sets the thread environment before numpy is imported)
import interior_continuation_v1 as IC  # noqa: E402
import core_kinematics_v2 as CK  # noqa: E402
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

try:
    import dust_root_history_v2b as DRH  # noqa: E402
    ROOT_FINDER = "dust_root_history_v2b.applied_outer_inner (DUST_root_history version 2's finder: the outer root the largest rising zero of N + v on the applied site fields, linear between sites; the inner root the largest falling zero below it)"
except Exception as _ex:  # noqa: BLE001
    DRH = None
    ROOT_FINDER = "the same arithmetic as dust_root_history_v2b.applied_outer_inner, local (that module is not importable here: %s: %s)" % (type(_ex).__name__, _ex)

READOUT = "well"
TITLE = "CORE_well_v1"
CODE = "core_well_v1.py"
MODULE = "core_well_v1"
LIMIT_LABEL = IC.LIMIT_LABEL
INVENTORY_LINE = ("Stored states of the r_s/a 64, tau 8, sigma/a 4 history: the r11 archive's slices at T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35 and 40 "
                  "(SLICE_INVENTORY_r64_v1: complex128 at 12, 16, 20, 24, 28; complex64 at 15, 25, 30, 35, 40; D/engine/H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11/); "
                  "no stored state at 12.5, 13, 13.5 or 14 on this history: the v36d rung reruns (tags c32_rs64_A_R2_tau8_ell{2,4,8}_prep4_avg_plain_v36d_from0, another "
                  "engine from T = 0) store states every 1 r_s and not every 0.5 r_s (the builder's inventory of <internal>/v36d_from0), so 12.5 and "
                  "13.5 exist on no history and the rung 4's 13 and 14 are that rerun's states, not this history's; the plan's half-r_s states from 11 to 16 do not exist.")
S8_LINE = ("The sigma/a 8 member (c27_rs64_A_R2_tau8_ell8_prep4_avg, lad_s8_now.json): 8 complex64 slices at 12, 13, 14, 16, 20, 25, 30 and 40, the only copy "
           "(D/engine/H100b/states/); records v25 (engine_fields_v16); roots at 13, 14, 16 and 40, none at 12, 20, 25, 30 (the first episode ends 16.375).")
S2_LINE = ("The sigma/a 2 member: no stored state at the bounce exists in the inputs (the v36d rung 2 holds T/r_s 1 to 4 so far, slices every 1 r_s, "
           "running on pod 3); its records json is not among the inputs: it enters through --extra-history (records=...) and its rows print 'no records json' until then.")
PROVENANCE = ("every number is derived from the stored slice, the records json and the fields npz by this script unless its key or column is labelled 'record' "
              "(read from the records json), 'inbox' (quoted from the plan's item 1) or 'floor' (the complex64 floor of results/engine/C3ext19/slice_restart_check_v1.md)")
CORRELATION_NOTE = ("per engine channel without multiplicity on any printed correlation (the fits, the shares, the spectra per channel); the sums with the slice's "
                    "declared multiplicities 16 kappa appear only on additive counts, densities and currents and are labelled 'weighted' (multiplicities counted once)")
INBOX_T16_ROOTS = {"applied_outer": 48.5364, "dense_outer": 50.8371}
BIN_LABELS = ("omega <= -b", "-b < omega <= -m", "-m < omega <= 0", "0 < omega <= m", "m < omega <= b", "omega > b")
CUT_ORDER = ("dense_outer", "dense_inner", "applied_outer", "applied_inner", "prev_dense_outer", "next_dense_outer", "inbox_applied_outer", "inbox_dense_outer")
CUR_KEYS = ("dust_number", "dust_number_straddle", "dust_energy", "dust_FM", "dust_number_low", "dust_number_high", "dust_number_cross", "dust_energy_low",
            "dust_energy_high", "dust_energy_cross", "dust_FM_low", "dust_FM_high", "dust_FM_cross", "C_number", "R2_number", "dev_number", "C_energy",
            "R2_energy", "dev_energy", "dev_FM", "pairs_number", "pairs_FM")


def add_arguments(ap):
    g = ap.add_argument_group("core_well_v1")
    g.add_argument("--gate", type=float, default=0.1, help="the gate sigma |d_r v| on the applied fields (requirements 5 I1's working value); the label is computed from it")
    g.add_argument("--shell-sites", dest="shell_sites", type=int, default=128, help="the shells' width in sites (128 a = 2 r_s at r_s/a 64)")
    g.add_argument("--k-grid", dest="k_grid", type=int, default=2048, help="the momentum grid (raised to the next power of 2 above n + 2, core_kinematics_v2.k_grid)")
    g.add_argument("--k-threshold-pi", dest="k_threshold_pi", type=float, default=0.25, help="the share at |ap| above this times pi (1/4: where cos 2ak changes sign)")
    g.add_argument("--core-window", dest="core_window", default="R50", choices=["R50", "R90", "inner_root", "Rc"], help="the primary window of the ln N against ln N_< fit (the others are computed beside)")
    g.add_argument("--fit-min-count", dest="fit_min_count", type=float, default=1.0, help="the sites with N_<(r) at or above this count enter the fit")
    g.add_argument("--n-floor", dest="n_floor", type=float, default=1e-9, help="rho/n is formed where the dust number per site exceeds this")
    g.add_argument("--bound-lo", dest="bound_lo", type=float, default=0.0, help="the bound part's lower edge in units of m (0: 0 < omega <= m)")
    g.add_argument("--m", type=float, default=0.2, help="m a for a history without a records json (the records' args otherwise)")
    g.add_argument("--g", type=float, default=0.25, help="g for a history without a records json (the records' args otherwise)")
    g.add_argument("--extra-history", dest="extra_history", action="append", default=[],
                   help="a history outside slices144_io_v1.inventory: 'key=s2;tag=<tag>;dirs=<dir>|<dir>;sigma=2;records=<json>;fields=<npz>;module=<fields module>;label=<text>;engine=<text>;clocks=all;nsites=2048;rs=64' (records and fields may be absent)")
    g.add_argument("--records-pass", dest="records_pass", default="on", choices=["on", "off"], help="the per-record pass (the root attribution) in compose")
    g.add_argument("--store-nk", dest="store_nk", default="on", choices=["on", "off"], help="store every shell's n(k) in the npz")


def is_done(entry, a):
    pj = IO.out_paths(a, READOUT, entry)[0]
    if not os.path.exists(pj):
        return False
    try:
        with open(pj) as f:
            return json.load(f).get("code") == CODE
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------------ the arithmetic
def _zeros_of(R, f):
    """every sign change of f on the sites, linear between sites: (R_zero, +1 rising / -1 falling)."""
    f = np.asarray(f, float)
    R = np.asarray(R, float)
    out = []
    s0, s1 = f[:-1], f[1:]
    idx = np.nonzero(((s0 < 0) & (s1 >= 0)) | ((s0 >= 0) & (s1 < 0)))[0]
    for i in idx:
        if s1[i] == s0[i]:
            continue
        out.append((float(R[i] - s0[i] * (R[i + 1] - R[i]) / (s1[i] - s0[i])), 1 if s1[i] > s0[i] else -1))
    return out


def _applied_outer_inner(R, f):
    z = _zeros_of(R, f)
    rising = [r for r, s in z if s > 0]
    if not rising:
        return None, None, z
    Ro = max(rising)
    falling = [r for r, s in z if s < 0 and r < Ro]
    return Ro, (max(falling) if falling else None), z


def applied_roots(R, N_app, v_app):
    """the applied roots: the zeros of N_bar + v_bar on the sites (dust_root_history_v2b's finder where importable)."""
    f = np.asarray(N_app, float) + np.asarray(v_app, float)
    fn = DRH.applied_outer_inner if DRH is not None else _applied_outer_inner
    Ro, Ri, z = fn(np.asarray(R, float), f)
    return (float(Ro) if Ro is not None else None), (float(Ri) if Ri is not None else None), [{"R": float(r_), "slope_sign": int(s_)} for r_, s_ in z]


def gate_of(R, v_app, ell, thr, inside_R):
    """sigma |d_r v| on the applied site fields (interior_continuation_v1's gate); the label is computed from it."""
    R = np.asarray(R, float)
    s = float(ell) * np.abs(np.gradient(np.asarray(v_app, float), R))
    inside = (R <= inside_R) if inside_R is not None else np.ones(len(R), bool)
    i_all = int(np.argmax(s))
    i_in = int(np.argmax(np.where(inside, s, -1.0)))
    return {"sigma": float(ell), "threshold": float(thr), "sigma_abs_dv_max": float(s[i_all]), "at_R": float(R[i_all]), "sigma_abs_dv_max_inside": float(s[i_in]), "at_R_inside": float(R[i_in]),
            "inside_region": ("R <= %.4f" % inside_R) if inside_R is not None else "the whole chain", "exceeded": bool(s[i_all] > thr), "exceeded_inside": bool(s[i_in] > thr),
            "definition": "sigma |d_r v| on the applied site fields (numpy.gradient), sigma = ell, the records' averaging scale; against requirements 5 I1's working value %g" % thr}


def label_of(gate):
    return LIMIT_LABEL if gate["exceeded"] else "within the record's own limit (the gate is not exceeded on the applied fields)"


def read_nodes(Z, k):
    """the dense solution's stored nodes (r_nodes, N_nodes, v_nodes) of record k, or None."""
    try:
        if not all(key in Z.files for key in ("r_nodes", "N_nodes", "v_nodes")):
            return None
        rn = np.asarray(Z["r_nodes"][k], float).reshape(-1)
        Nn = np.asarray(Z["N_nodes"][k], float).reshape(-1)
        vn = np.asarray(Z["v_nodes"][k], float).reshape(-1)
        if rn.size < 2 or rn.size != Nn.size or rn.size != vn.size:
            return None
        return rn, Nn, vn
    except Exception:  # noqa: BLE001
        return None


def dense_at_sites(nodes, R):
    rn, Nn, vn = nodes
    o = np.argsort(rn)
    return np.interp(R, rn[o], Nn[o]), np.interp(R, rn[o], vn[o])


def fields_at_record(ctx, entry, T):
    """the record's stored fields from the fields npz: the shell totals, the site fields, the applied fields and the dense nodes."""
    out = {"available": False, "k": None}
    try:
        p, Z = ctx.fields_file(entry)
    except Exception as ex:  # noqa: BLE001
        out["error"] = "%s: %s" % (type(ex).__name__, ex)
        return out
    if Z is None:
        out["error"] = "no fields npz"
        return out
    Tz = np.asarray(Z["T"], float).reshape(-1)
    k = int(np.argmin(np.abs(Tz - float(T))))
    if abs(float(Tz[k]) - float(T)) > 1e-6:
        out["error"] = "no record at T %.3f in the fields npz (nearest %.3f)" % (T, Tz[k])
        return out
    out.update({"available": True, "k": k, "T_fields": float(Tz[k]), "path": p})
    for key in ("eps_tot", "J_tot", "eps_pairs", "J_pairs", "N", "v", "N_applied", "v_applied"):
        arr = ctx.fields_key(entry, key)
        out[key] = np.asarray(arr[k], float) if arr is not None else None
    out["nodes"] = read_nodes(Z, k)
    return out


def mass_function(R, w, eps, J, g):
    """M_w(r) = r w^2/(2 g) and I(r) = sum_{x <= r} (eps + w J)(x) (the shell totals carry 4 pi r^2 a; a = 1); the source per shell."""
    R = np.asarray(R, float)
    w = np.asarray(w, float)
    M_w = R * w * w / (2.0 * float(g))
    src = np.asarray(eps, float) + w * np.asarray(J, float)
    return M_w, np.cumsum(src), src


def gradient_length(x, N):
    """L_N = N/|N'| on the grid x (numpy.gradient); inf where N' = 0."""
    N = np.asarray(N, float)
    dN = np.gradient(N, np.asarray(x, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(dN) > 0.0, N / np.abs(dN), np.inf)


def L_stats(L, R, sigma, inside_R, radii):
    L = np.asarray(L, float)
    R = np.asarray(R, float)
    fin = np.isfinite(L)
    out = {"sigma": float(sigma), "sites_L_below_sigma": int(np.sum(fin & (L < sigma)))}
    if fin.any():
        i = int(np.argmin(np.where(fin, L, np.inf)))
        out.update({"min_chain": float(L[i]), "min_chain_R": float(R[i]), "sigma_over_min_chain": float(sigma / L[i]) if L[i] > 0 else None})
    else:
        out.update({"min_chain": None, "min_chain_R": None, "sigma_over_min_chain": None})
    if inside_R is not None:
        sel = fin & (R <= inside_R)
        if sel.any():
            i = int(np.argmin(np.where(sel, L, np.inf)))
            out.update({"min_inside_root": float(L[i]), "min_inside_root_R": float(R[i]), "sigma_over_min_inside_root": float(sigma / L[i]) if L[i] > 0 else None})
    Lc = np.where(fin, L, 1e300)
    out["at_radii"] = {lab: (float(np.interp(x, R, Lc)) if x is not None else None) for lab, x in radii.items()}
    return out


def fit_loglog(x, y, mask):
    """the least-squares slope of ln y against ln x on the masked points (x, y > 0), its standard error and r^2."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sel = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    nn = int(sel.sum())
    out = {"n": nn, "slope": None, "slope_se": None, "intercept": None, "r2": None, "ln_x_range": None}
    if nn < 3:
        return out
    lx, ly = np.log(x[sel]), np.log(y[sel])
    if np.ptp(lx) <= 0.0:
        return out
    A = np.vstack([lx, np.ones(nn)]).T
    coef = np.linalg.lstsq(A, ly, rcond=None)[0]
    resid = ly - A @ coef
    s2 = float(np.sum(resid ** 2) / max(nn - 2, 1))
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = float(math.sqrt(max(float(cov[0, 0]), 0.0)))
    except np.linalg.LinAlgError:
        se = None
    sst = float(np.sum((ly - ly.mean()) ** 2))
    out.update({"slope": float(coef[0]), "slope_se": se, "intercept": float(coef[1]), "r2": (1.0 - float(np.sum(resid ** 2)) / sst) if sst > 0 else None,
                "ln_x_range": [float(lx.min()), float(lx.max())]})
    return out


def turning_radius(rs, omega, m):
    """r_s/(1 - (omega/m)^2) for |omega| < m; None otherwise (no turning point)."""
    if rs is None or omega is None:
        return None
    q = (float(omega) / float(m)) ** 2
    return float(rs) / (1.0 - q) if q < 1.0 else None


def shells_of(nsites, width):
    width = max(1, int(width))
    edges = list(range(0, int(nsites), width)) + [int(nsites)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def momentum_low_mask(nsites, thr_pi):
    """the n-point DFT grid points with |p| <= thr pi in the reduced zone p = k mod pi (an exact integer test where n is divisible by 8 and thr = 1/4)."""
    n = int(nsites)
    j = np.arange(n)
    if n % 8 == 0 and abs(thr_pi - 0.25) < 1e-12:
        jp = np.mod(j + n // 4, n // 2) - n // 4
        return np.abs(jp) <= n // 8
    k = 2.0 * math.pi * j / n
    p = np.mod(k + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    return np.abs(p) <= thr_pi * math.pi + 1e-12


def momentum_split_columns(X, nsites, thr_pi=0.25):
    """the columns projected on the reduced-zone momenta |ap| <= thr pi (X_low) and on the rest (X_high = X - X_low), by the n-point DFT per
    spinor component (a unitary basis on the chain's sites: an exact orthogonal projection; the boundary |ap| = thr pi goes to the low set)."""
    X = np.asarray(X, np.complex128)
    low = momentum_low_mask(nsites, thr_pi)
    Xl = np.empty_like(X)
    for c in (0, 1):
        F = np.fft.fft(X[c::2], axis=0)
        Xl[c::2] = np.fft.ifft(F * low[:, None], axis=0)
    return Xl, X - Xl, low


def straddling_current(H, R, x, X, f):
    """the outward number current through the cut at x from the bonds that straddle it: 2 sum_{a in, b out} Im(h_ba D_ab), D = X diag(f) X^dagger
    (= Tr[D i[h, Pi_{R > x}]] exactly, since i[h, Pi] vanishes on every other pair); returns (current, the number of straddling matrix elements,
    their largest site separation)."""
    R = np.asarray(R, float)
    out_sites = R > float(x)
    rows_out = np.nonzero(np.repeat(out_sites, 2))[0]
    rows_in = np.nonzero(np.repeat(~out_sites, 2))[0]
    if rows_out.size == 0 or rows_in.size == 0:
        return 0.0, 0, 0
    Hio = H[rows_in][:, rows_out].tocoo()
    if Hio.nnz == 0:
        return 0.0, 0, 0
    a_ = rows_in[Hio.row]
    b_ = rows_out[Hio.col]
    h_ab = np.asarray(Hio.data, np.complex128)
    Xf = np.asarray(X)
    Xa = np.asarray(Xf[a_], np.complex128)
    Xb = np.asarray(Xf[b_], np.complex128)
    D_ab = np.sum(Xa * np.conj(Xb) * np.asarray(f, float)[None, :], axis=1)
    rng = int(np.max(np.abs(a_ // 2 - b_ // 2)))
    return float(2.0 * np.sum(np.imag(np.conj(h_ab) * D_ab))), int(Hio.nnz), rng


def expect_rows(J, X, f):
    """Re sum_n f_n x_n^dagger J x_n for a sparse cut operator J (only the rows it touches are read from X; complex64 columns are promoted)."""
    co = J.tocoo()
    if co.nnz == 0:
        return 0.0
    rows = np.unique(np.concatenate([co.row, co.col]))
    Jb = J[rows][:, rows].toarray()
    Xr = np.asarray(np.asarray(X)[rows], np.complex128)
    return float(np.real(np.sum(np.asarray(f, float) * np.sum(np.conj(Xr) * (Jb @ Xr), axis=0))))


def killing_edges(m, b):
    return [-math.inf, -float(b), -float(m), 0.0, float(m), float(b), math.inf]


def bin_indices(omega, m, b):
    """the bin of every level: omega in bin i where edges[i] < omega <= edges[i+1] (half-open (lo, hi])."""
    inner = np.array([-float(b), -float(m), 0.0, float(m), float(b)])
    return np.searchsorted(inner, np.asarray(omega, float), side="left")


def killing_bins(omega, nj, m, b):
    """the 6 bins split at 0, +-m, +-b: number and energy per bin and the level count."""
    omega = np.asarray(omega, float)
    nj = np.asarray(nj, float)
    edges = killing_edges(m, b)
    bidx = bin_indices(omega, m, b)
    bins = []
    for i in range(6):
        sel = bidx == i
        bins.append({"label": BIN_LABELS[i], "lo": edges[i], "hi": edges[i + 1], "N": float(np.sum(nj[sel])), "E": float(np.sum(omega[sel] * nj[sel])), "levels": int(sel.sum())})
    return bins, bidx


def killing_analysis(H, dust, f, R, m, b, e_stored, bound_lo_m, rs_nom, two_gE):
    """the dust in the Killing eigenbasis of h_m at the applied fields: the bins, the per-site bin densities Re Tr[Pi_x P_bin D], the bound part
    and the per-orbital quantities."""
    t1 = time.time()
    Hd = H.toarray()
    om, V = IO.eigh(Hd)
    del Hd
    R = np.asarray(R, float)
    f = np.asarray(f, float)
    out = {"levels": int(om.size), "seconds_eigh": time.time() - t1, "basis": "the eigenbasis of h_m at the slice's applied fields (page_ops_v6.clocked_csr on the stored chain)"}
    out["eig_vs_stored"] = float(np.max(np.abs(np.sort(om) - np.sort(np.asarray(e_stored, float))))) if (e_stored is not None and np.asarray(e_stored).size == om.size) else None
    A = V.conj().T @ dust
    P = np.abs(A) ** 2
    nj = P @ f
    bins, bidx = killing_bins(om, nj, m, b)
    out["bins"] = bins
    out["edges"] = killing_edges(m, b)
    out["completeness_check"] = float(np.sum(nj) - np.sum(f * np.sum(np.abs(dust) ** 2, axis=0)))
    n_site = IC.dust_number_site(dust, f)
    dens = np.zeros((6, len(R)))
    for i in range(6):
        idx = np.nonzero(bidx == i)[0]
        if idx.size:
            Y = V[:, idx] @ A[idx]
            dens[i] = IC.site_sum(np.real(np.conj(dust) * Y) @ f)
    out["site_density_check"] = float(np.max(np.abs(dens.sum(axis=0) - n_site)))
    lo = float(bound_lo_m) * m
    idx_b = np.nonzero((om > lo) & (om <= m))[0]
    bound = {"window": "%g m < omega <= m" % bound_lo_m, "levels": int(idx_b.size), "N": 0.0, "R_50": None, "R_90": None, "mean_omega": None, "mean_omega_over_m": None,
             "r_turn_rs_nominal": None, "r_turn_2gE": None}
    Yb = None
    dens_b = np.zeros(len(R))
    if idx_b.size:
        Yb = V[:, idx_b] @ A[idx_b]
        dens_b = IC.dust_number_site(Yb, f)
        Nb = float(dens_b.sum())
        nb = nj[idx_b]
        mean_om = float(np.sum(om[idx_b] * nb) / np.sum(nb)) if np.sum(nb) > 0 else None
        bound.update({"N": Nb, "R_50": IO.frac_radius(R, np.cumsum(dens_b) / Nb, 0.5) if Nb > 0 else None, "R_90": IO.frac_radius(R, np.cumsum(dens_b) / Nb, 0.9) if Nb > 0 else None,
                      "mean_omega": mean_om, "mean_omega_over_m": (mean_om / m) if mean_om is not None else None,
                      "r_turn_rs_nominal": turning_radius(rs_nom, mean_om, m), "r_turn_2gE": turning_radius(two_gE, mean_om, m)})
    bound["definition"] = "the bound part = the dust projected on the Killing levels in the window; its density |P_b d_n|^2 per site (positive), its mean Killing energy, and the turning radius r_s/(1 - (omega/m)^2) with the nominal r_s and with the record's 2gE"
    out["bound"] = bound
    orbs = []
    d2 = np.abs(dust) ** 2
    for c in range(dust.shape[1]):
        pc = P[:, c]
        wsum = float(pc.sum())
        om_c = float(np.sum(om * pc) / wsum) if wsum > 0 else None
        ns = d2[0::2, c] + d2[1::2, c]
        tot = float(ns.sum())
        o = {"column": c, "f": float(f[c]), "norm": wsum, "mean_omega": om_c, "mean_omega_over_m": (om_c / m) if om_c is not None else None,
             "spread": float(math.sqrt(max(float(np.sum(om ** 2 * pc) / wsum) - om_c ** 2, 0.0))) if wsum > 0 else None,
             "R_50": IO.frac_radius(R, np.cumsum(ns) / tot, 0.5) if tot > 0 else None, "R_90": IO.frac_radius(R, np.cumsum(ns) / tot, 0.9) if tot > 0 else None,
             "bins_N": [float(np.sum(pc[bidx == i])) for i in range(6)], "bound_weight": float(np.sum(pc[idx_b])) if idx_b.size else 0.0,
             "bound_mean_omega_over_m": None, "bound_R_90": None, "r_turn_rs_nominal": None, "r_turn_2gE": None}
        if Yb is not None and o["bound_weight"] > 0:
            om_b = float(np.sum(om[idx_b] * pc[idx_b]) / o["bound_weight"])
            yb = np.abs(Yb[:, c]) ** 2
            nsb = yb[0::2] + yb[1::2]
            totb = float(nsb.sum())
            o.update({"bound_mean_omega_over_m": om_b / m, "bound_R_90": IO.frac_radius(R, np.cumsum(nsb) / totb, 0.9) if totb > 0 else None,
                      "r_turn_rs_nominal": turning_radius(rs_nom, om_b, m), "r_turn_2gE": turning_radius(two_gE, om_b, m)})
        orbs.append(o)
    out["orbitals"] = orbs
    return out, {"omega": om, "n_dust": nj, "bin_density": dens, "bound_density": dens_b}


def momentum_per_shell(dust, f, R, shells, M, k, thr):
    """per shell the dust's n_S(k) (core_kinematics_v2.window_momentum) and its reduced-zone moments."""
    res, nks = [], []
    for lo, hi in shells:
        nk = CK.window_momentum(dust[2 * lo:2 * hi], f, R[lo:hi], k)
        res.append(CK.moments(nk, k, thr, M))
        nks.append(nk)
    return res, nks


def interp_at(x, R, arr):
    if x is None or arr is None:
        return None
    arr = np.asarray(arr, float)
    if not np.all(np.isfinite(arr)):
        i = int(np.argmin(np.abs(R - x)))
        return float(arr[i]) if np.isfinite(arr[i]) else None
    return float(np.interp(float(x), R, arr))


# ------------------------------------------------------------------------------------------------------------ the slice
def process_slice(rd, ctx, entry, a):
    t0 = time.time()
    log = ctx.log
    pj, pz = IO.out_paths(a, READOUT, entry)
    pm = pj[:-5] + ".md"
    rec = ctx.record_at(entry, rd.T) or {}
    prev_rec, next_rec = ctx.record_prev_next(entry, rd.T)
    m, g, ell = float(ctx.m), float(ctx.g), float(ctx.ell)
    rs = float(rd.rs)
    b = math.sqrt(m * m + 1.0)
    R = np.asarray(rd.R, float)
    n = rd.nsites
    if rd.N_applied is None:
        raise RuntimeError("the slice %s has no N_applied/v_applied (a parent c17 slice?): the readout needs the applied fields" % rd.path)
    N_app = np.asarray(rd.N_applied, float)
    v_app = np.asarray(rd.v_applied, float)
    row = IO.slice_header(rd, ctx, entry, rec, READOUT, TITLE, CODE)
    row.update({"inventory_line": INVENTORY_LINE, "s8_line": S8_LINE, "s2_line": S2_LINE, "provenance": PROVENANCE, "correlation_note": CORRELATION_NOTE,
                "momentum_convention": CK.ZONE, "root_finder": ROOT_FINDER, "a": 1.0, "b": b})
    E = float(rec["E"]) if rec.get("E") is not None else None
    two_gE = 2.0 * g * E if E is not None else None
    Rc = float(rec["R_c"]) if rec.get("R_c") is not None else None
    row["record"] = {"source": "record", "E": E, "E_over_E0": rec.get("E_over_E0"), "two_gE": two_gE, "R_c": Rc, "trapping_zeros": rec.get("trapping_zeros"), "kappa_T": rec.get("kappa_T"),
                     "R_t": rec.get("R_t"), "r_in": rec.get("r_in"), "min_N_minus_absv": rec.get("min_N_minus_absv"), "r_min_N_minus_absv": rec.get("r_min_N_minus_absv"),
                     "misner_sharp_at_Rt": rec.get("misner_sharp_at_Rt"), "W_over_2g_at_Rt": rec.get("W_over_2g_at_Rt"), "M1_balance": rec.get("M1_balance"), "H_in": rec.get("H_in"),
                     "dE_dT": rec.get("dE_dT"), "converged": rec.get("converged"), "engine": rec.get("engine") or rec.get("record_label"), "solver": rec.get("solver"),
                     "prev_T_over_rs": (float(prev_rec["T"]) / rs) if prev_rec else None, "next_T_over_rs": (float(next_rec["T"]) / rs) if next_rec else None}
    # the record's stored fields
    FR = fields_at_record(ctx, entry, rd.T)
    eps_tot, J_tot, eps_pairs, J_pairs = FR.get("eps_tot"), FR.get("J_tot"), FR.get("eps_pairs"), FR.get("J_pairs")
    if FR.get("nodes") is not None:
        N_d, v_d = dense_at_sites(FR["nodes"], R)
        dense_src = "the fields npz's stored dense nodes (r_nodes, N_nodes, v_nodes; %d nodes) interpolated linearly to the sites" % FR["nodes"][0].size
    elif FR.get("N") is not None:
        N_d, v_d = FR["N"], FR["v"]
        dense_src = "the fields npz's stored site fields N, v (no dense nodes stored)"
    else:
        N_d, v_d = N_app.copy(), v_app.copy()
        dense_src = "the slice's applied fields (no fields npz record: %s)" % FR.get("error")
    with np.errstate(divide="ignore", invalid="ignore"):
        w_d = v_d / N_d
        w_a = v_app / N_app
    row["fields_source"] = {"dense": dense_src, "applied": "the slice's N_applied, v_applied (the record's accepted site fields)", "fields_npz": FR.get("path"), "record_index": FR.get("k"), "error": FR.get("error")}
    if FR.get("N_applied") is not None:
        row["fields_source"]["slice_vs_fields_npz_applied_max_abs"] = float(max(np.max(np.abs(FR["N_applied"] - N_app)), np.max(np.abs(FR["v_applied"] - v_app))))
    # the roots and the gate
    zeros = [float(z) for z in (rec.get("trapping_zeros") or [])]
    Ro_d = max(zeros) if zeros else None
    Ri_d = min(zeros) if len(zeros) >= 2 else None
    Ro_a, Ri_a, za = applied_roots(R, N_app, v_app)
    prev_Ro = float(max(prev_rec["trapping_zeros"])) if (prev_rec and prev_rec.get("trapping_zeros")) else None
    next_Ro = float(max(next_rec["trapping_zeros"])) if (next_rec and next_rec.get("trapping_zeros")) else None
    row["roots"] = {"dense_outer": Ro_d, "dense_inner": Ri_d, "dense_source": "record (trapping_zeros: the zeros of the dense N + v, identical to N - |v|'s for v <= 0)",
                    "applied_outer": Ro_a, "applied_inner": Ri_a, "applied_zeros": za, "applied_source": ROOT_FINDER,
                    "prev_dense_outer": prev_Ro, "next_dense_outer": next_Ro,
                    "w_dense_at_dense_outer": interp_at(Ro_d, R, w_d), "w_applied_at_applied_outer": interp_at(Ro_a, R, w_a),
                    "dense_outer_over_2gE": (Ro_d / two_gE) if (Ro_d is not None and two_gE) else None, "dense_inner_over_2gE": (Ri_d / two_gE) if (Ri_d is not None and two_gE) else None,
                    "applied_outer_over_2gE": (Ro_a / two_gE) if (Ro_a is not None and two_gE) else None}
    is_prod16 = entry["history"] == "prod" and abs(rd.T_over_rs - 16.0) < 1e-6
    if is_prod16:
        row["roots"]["inbox_T16"] = {"source": "inbox", "applied_outer": INBOX_T16_ROOTS["applied_outer"], "dense_outer": INBOX_T16_ROOTS["dense_outer"],
                                     "computed_minus_inbox_applied": (Ro_a - INBOX_T16_ROOTS["applied_outer"]) if Ro_a is not None else None,
                                     "computed_minus_inbox_dense": (Ro_d - INBOX_T16_ROOTS["dense_outer"]) if Ro_d is not None else None}
    gate = gate_of(R, v_app, ell, a.gate, Ro_d if Ro_d is not None else Rc)
    row["gate"] = gate
    row["limit_label"] = label_of(gate)
    # the dust per channel
    dusts, n_site = {}, {}
    n_w = np.zeros(n)
    for kap in rd.kappas:
        if rd.n_dust[kap]:
            d, f = rd.dust(kap)
            dusts[kap] = (d, f)
            n_site[kap] = IC.dust_number_site(d, f)
            n_w += rd.weights[kap] * n_site[kap]
    tot = float(n_w.sum())
    cum_w = np.cumsum(n_w)
    R50 = IO.frac_radius(R, cum_w / tot, 0.5) if tot > 0 else None
    R90 = IO.frac_radius(R, cum_w / tot, 0.9) if tot > 0 else None
    vol = 4.0 * math.pi * R ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (eps_tot / vol) if eps_tot is not None else None
        rho_dust = ((eps_tot - eps_pairs) / vol) if (eps_tot is not None and eps_pairs is not None) else None
        n_dens = n_w / vol
        J_dens = (J_tot / vol) if J_tot is not None else None
        rho_over_n = np.where(n_w > a.n_floor, eps_tot / np.where(n_w > a.n_floor, n_w, 1.0), np.nan) if eps_tot is not None else None
        rho_dust_over_n = np.where(n_w > a.n_floor, (eps_tot - eps_pairs) / np.where(n_w > a.n_floor, n_w, 1.0), np.nan) if rho_dust is not None else None
    row["dust"] = {"total_weighted": tot, "per_channel_total": {int(k_): float(v_.sum()) for k_, v_ in n_site.items()}, "R_50": R50, "R_90": R90,
                   "definitions": "number per site from the stored dust columns (f all 1), weighted 16 kappa for the total and N_<(r); n(r) = the shell total over 4 pi r^2 a; rho(r) = eps_tot(x)/(4 pi r^2 a), the records' h_0 source (eps_tot - eps_pairs the dust's part); J(r) = J_tot(x)/(4 pi r^2 a); R_50, R_90 the sampled radii holding 50 and 90 percent of the dust (engine_core_slices_v3.frac_radius)"}
    arrays = {"R": R, "N_applied": N_app, "v_applied": v_app, "N_dense_sites": N_d, "v_dense_sites": v_d, "w_dense": w_d, "w_applied": w_a, "dust_number_site": n_w, "dust_enclosed": cum_w,
              "n_density": n_dens}
    for kap, ns in n_site.items():
        arrays["k%d_dust_number_site" % kap] = ns
        arrays["k%d_dust_enclosed" % kap] = np.cumsum(ns)
    for key, arr in (("eps_tot", eps_tot), ("J_tot", J_tot), ("eps_pairs", eps_pairs), ("J_pairs", J_pairs), ("rho_density", rho), ("rho_dust_density", rho_dust), ("J_density", J_dens),
                     ("rho_over_n", rho_over_n), ("rho_dust_over_n", rho_dust_over_n)):
        if arr is not None:
            arrays[key] = arr
    # the mass function
    MF = {"g": g, "definition": "M_w(r) = r w^2/(2 g) on the dense fields (and the applied); I(r) = sum_{x <= r} (eps_tot + w J_tot)(x), the shell totals (a = 1); I_c(r) = E - sum_{x > r}, the collar form; mismatch = M_w - I; the record's W_over_2g_at_Rt and misner_sharp_at_Rt beside as the convention check",
          "available": eps_tot is not None and J_tot is not None}
    M_w_d = I_d = I_c_d = M_w_a = I_a = None
    if MF["available"]:
        M_w_d, I_d, src_d = mass_function(R, w_d, eps_tot, J_tot, g)
        M_w_a, I_a, src_a = mass_function(R, w_a, eps_tot, J_tot, g)
        I_c_d = (E - (I_d[-1] - I_d)) if E is not None else None
        mis_d = M_w_d - I_d
        mis_a = M_w_a - I_a
        inside = (R < Rc) if Rc is not None else np.ones(n, bool)
        MF.update({"max_abs_mismatch_inside_Rc_dense": float(np.max(np.abs(mis_d[inside]))) if inside.any() else None,
                   "max_abs_mismatch_inside_Rc_applied": float(np.max(np.abs(mis_a[inside]))) if inside.any() else None,
                   "max_abs_mismatch_collar_inside_Rc_dense": float(np.max(np.abs((M_w_d - I_c_d)[inside]))) if (I_c_d is not None and inside.any()) else None,
                   "I_chain_end": float(I_d[-1]), "E_record": E, "I_chain_end_minus_E": (float(I_d[-1]) - E) if E is not None else None,
                   "source_total_eps": float(np.sum(eps_tot)), "source_total_wJ_dense": float(np.sum(w_d * J_tot))})
        for lab, x in (("dense_outer", Ro_d), ("dense_inner", Ri_d), ("applied_outer", Ro_a), ("R_c", Rc), ("R_50", R50), ("R_90", R90)):
            if x is None:
                continue
            MF["at_" + lab] = {"R": x, "M_w_dense": interp_at(x, R, M_w_d), "M_w_applied": interp_at(x, R, M_w_a), "I": interp_at(x, R, I_d), "I_collar": interp_at(x, R, I_c_d),
                               "I_applied_w": interp_at(x, R, I_a), "R_over_2g": x / (2.0 * g)}
        MF["record"] = {"source": "record", "W_over_2g_at_Rt": rec.get("W_over_2g_at_Rt"), "misner_sharp_at_Rt": rec.get("misner_sharp_at_Rt"), "M1_balance": rec.get("M1_balance"), "E": E}
        arrays.update({"M_w_dense": M_w_d, "M_w_applied": M_w_a, "I_integral_dense": I_d, "I_integral_applied": I_a, "mismatch_dense": mis_d, "mismatch_applied": mis_a, "source_dense": src_d})
        if I_c_d is not None:
            arrays["I_collar_dense"] = I_c_d
    row["mass_function"] = MF
    # the gradient length
    radii_named = {"dense_outer": Ro_d, "dense_inner": Ri_d, "applied_outer": Ro_a, "applied_inner": Ri_a, "R_50": R50, "R_90": R90, "R_c": Rc}
    if FR.get("nodes") is not None:
        rn, Nn, _ = FR["nodes"]
        o = np.argsort(rn)
        Ln = gradient_length(rn[o], Nn[o])
        L_d = np.interp(R, rn[o], np.where(np.isfinite(Ln), Ln, 1e300))
        L_d_src = "N/|N'| on the stored dense nodes (numpy.gradient on r_nodes) interpolated to the sites"
    else:
        L_d = gradient_length(R, N_d)
        L_d_src = "N/|N'| on the dense-at-sites fields (numpy.gradient on the sites)"
    L_a = gradient_length(R, N_app)
    row["gradient_length"] = {"dense": dict(L_stats(L_d, R, ell, Ro_d, radii_named), source=L_d_src), "applied": dict(L_stats(L_a, R, ell, Ro_d, radii_named), source="N_bar/|N_bar'| on the applied site fields"),
                              "definition": "L_N = N/|N'| against sigma = ell = %g: sigma/L_N > 1 where the clock varies within one averaging length" % ell}
    arrays["L_N_dense"] = np.where(np.isfinite(L_d), L_d, 1e300)
    arrays["L_N_applied"] = np.where(np.isfinite(L_a), L_a, 1e300)
    # the profile at named radii
    prof = {}
    for lab, x in dict(radii_named, two_gE=two_gE).items():
        if x is None:
            continue
        prof[lab] = {"R": x, "N_dense": interp_at(x, R, N_d), "v_dense": interp_at(x, R, v_d), "w_dense": interp_at(x, R, w_d), "N_applied": interp_at(x, R, N_app), "v_applied": interp_at(x, R, v_app),
                     "w_applied": interp_at(x, R, w_a), "n_shell": interp_at(x, R, n_w), "n_density": interp_at(x, R, n_dens), "rho_density": interp_at(x, R, rho), "rho_over_n": interp_at(x, R, rho_over_n),
                     "J_density": interp_at(x, R, J_dens), "N_enclosed": interp_at(x, R, cum_w), "M_w_dense": interp_at(x, R, M_w_d), "I_integral": interp_at(x, R, I_d),
                     "L_N_dense": interp_at(x, R, arrays["L_N_dense"]), "L_N_applied": interp_at(x, R, arrays["L_N_applied"])}
    row["profile_at_radii"] = prof
    # the core fit
    W = {}
    if R50 is not None:
        W["R50"] = ("R <= R_50 = %.3f (the radius holding half the dust)" % R50, R <= R50)
    if R90 is not None:
        W["R90"] = ("R <= R_90 = %.3f" % R90, R <= R90)
    if Ri_d is not None:
        W["inner_root"] = ("R < the record's inner dense root %.3f" % Ri_d, R < Ri_d)
    if Rc is not None:
        W["Rc"] = ("R < the record's R_c = %.3f" % Rc, R < Rc)
    fits = {}
    for name, (lab, mask) in W.items():
        sel = mask & (cum_w >= a.fit_min_count)
        d_ = {"label": lab, "sites": int(mask.sum()), "sites_in_fit": int(sel.sum()), "fit_min_count": a.fit_min_count,
              "fit_dense": fit_loglog(cum_w, N_d, sel), "fit_applied": fit_loglog(cum_w, N_app, sel),
              "per_channel_fit_dense": {int(kap): fit_loglog(np.cumsum(n_site[kap]), N_d, mask & (np.cumsum(n_site[kap]) >= a.fit_min_count)) for kap in n_site}}
        if eps_tot is not None and float(n_w[mask].sum()) > 0:
            eps_c = float(np.sum(eps_tot[mask]) / np.sum(n_w[mask]))
            d_.update({"epsilon_c_total": eps_c, "exponent_eps_minus_m_over_2m": (eps_c - m) / (2.0 * m), "exponent_eps_over_2m": eps_c / (2.0 * m)})
            if eps_pairs is not None:
                eps_cd = float(np.sum((eps_tot - eps_pairs)[mask]) / np.sum(n_w[mask]))
                d_.update({"epsilon_c_dust": eps_cd, "exponent_dust_eps_minus_m_over_2m": (eps_cd - m) / (2.0 * m), "exponent_dust_eps_over_2m": eps_cd / (2.0 * m)})
        fits[name] = d_
    i_in = int(np.argmax(n_w > a.n_floor)) if np.any(n_w > a.n_floor) else None
    row["core_fit"] = {"primary_window": a.core_window if a.core_window in fits else (list(fits)[0] if fits else None), "windows": fits,
                       "rho_over_n_innermost": {"R": float(R[i_in]), "value": float(rho_over_n[i_in]) if rho_over_n is not None else None} if i_in is not None else None,
                       "definition": "the least-squares slope of ln N against ln N_<(r) on the window's sites with N_<(r) >= fit_min_count (N the dense fields at the sites; the applied beside; per channel N_<,kappa without multiplicity), beside (epsilon_c - m a)/(2 m a) and epsilon_c/(2 m a) with epsilon_c = sum eps_tot/sum n over the window (a = 1; the dust part eps_tot - eps_pairs beside)"}
    with np.errstate(divide="ignore", invalid="ignore"):
        lnN, lnC = np.log(N_d), np.log(cum_w)
        ok = np.isfinite(lnN) & np.isfinite(lnC)
        local = np.full(n, np.nan)
        if ok.sum() > 3:
            local[ok] = np.gradient(lnN[ok], lnC[ok])
    arrays["local_slope_lnN_lnNenc_dense"] = local
    # the shells, the momentum grid
    shells = shells_of(n, a.shell_sites)
    M, k = CK.k_grid(n, a.k_grid)
    p = CK.fold_reduced(k, M)
    thr = a.k_threshold_pi * math.pi
    arrays.update({"k": k, "p_reduced": p, "shell_edges_sites": np.array([lo for lo, _ in shells] + [n])})
    row["momentum_grid"] = {"M": M, "k_threshold": thr, "k_threshold_over_pi": a.k_threshold_pi, "zone": CK.ZONE, "convention": "core_kinematics_v2 (its CONVENTION paragraph)",
                            "definition": "per shell n_S(k) = sum_a e_k^dagger (Pi_S D Pi_S)^aa e_k on the grid k_j = 2 pi j/M, read on the fold p = k mod pi; the share at |ap| > k_threshold is the weight where cos 2ak < 0 (the counter-advected content); (1/M) sum_k n_S(k) is the shell's dust number"}
    shell_rows = []
    nk_w = [np.zeros(M) for _ in shells]
    dens_w = np.zeros((6, n))
    dens_b_w = np.zeros(n)
    bound_w = {"N": 0.0, "E": 0.0}
    bins_w = [{"label": BIN_LABELS[i], "N": 0.0, "E": 0.0} for i in range(6)]
    # the cuts
    cuts = []
    for lab, x in (("dense_outer", Ro_d), ("dense_inner", Ri_d), ("applied_outer", Ro_a), ("applied_inner", Ri_a), ("prev_dense_outer", prev_Ro), ("next_dense_outer", next_Ro)):
        if x is not None and R[0] < x < R[-1]:
            cuts.append((lab, float(x)))
    if is_prod16:
        cuts += [("inbox_applied_outer", INBOX_T16_ROOTS["applied_outer"]), ("inbox_dense_outer", INBOX_T16_ROOTS["dense_outer"])]
    cut_src = {"dense_outer": "the record's largest trapping zero", "dense_inner": "the record's smallest trapping zero (2 zeros)", "applied_outer": "the slice's applied outer root", "applied_inner": "the slice's applied inner root",
               "prev_dense_outer": "the previous record's dense outer root (for the attribution step ending at this clock)", "next_dense_outer": "the next record's dense outer root",
               "inbox_applied_outer": "inbox: the plan's quoted applied outer root at 16", "inbox_dense_outer": "inbox: the plan's quoted dense root at 16"}
    currents = {lab: {"R_cut": x, "source": cut_src[lab], "N_applied_at_cut": interp_at(x, R, N_app), "channels": {}, "weighted": {key: 0.0 for key in CUR_KEYS},
                      "straddling_check_max": 0.0, "range_sites_max": 0} for lab, x in cuts}
    chans = []
    for kap in rd.kappas:
        t1 = time.time()
        h0, Q, H = IO.operators(R, m, kap, N_app, v_app)
        ch = rd.channel(kap, orthonormalize=(a.loewdin == "on"))
        Phi, f_all = ch["Phi"], ch["f"]
        R2 = rd.mmap("R2_%d" % kap)
        out = {"kappa": int(kap), "weight": ch["weight"], "n_sea": ch["n_sea"], "n_dust": ch["n_dust"], "stored_orthonormality": ch["stored_orthonormality"], "loewdin_orthonormality": ch["loewdin_orthonormality"],
               "orthonormalization": ch["orthonormalization"], "dust_sea_overlap_max": ch["dust_sea_overlap_max"]}
        dust, f = dusts.get(kap, (None, None))
        if dust is not None:
            d_low, d_high, low_mask = momentum_split_columns(dust, n, a.k_threshold_pi)
            out["momentum_split"] = {"low_weight": float(np.sum(np.abs(d_low) ** 2 @ f)), "high_weight": float(np.sum(np.abs(d_high) ** 2 @ f)), "grid_points_low": int(low_mask.sum()),
                                     "definition": "the dust columns projected on the n-point DFT modes with |ap| <= %g pi in the reduced zone (low) and their complement (high); the currents of each part and the cross term" % a.k_threshold_pi}
            # the Killing spectrum
            KA, karr = killing_analysis(H, dust, f, R, m, b, rd.e(kap), a.bound_lo, rs, two_gE)
            out["killing"] = KA
            for i in range(6):
                bins_w[i]["N"] += ch["weight"] * KA["bins"][i]["N"]
                bins_w[i]["E"] += ch["weight"] * KA["bins"][i]["E"]
            bound_w["N"] += ch["weight"] * KA["bound"]["N"]
            if KA["bound"]["mean_omega"] is not None:
                bound_w["E"] += ch["weight"] * KA["bound"]["mean_omega"] * KA["bound"]["N"]
            dens_w += ch["weight"] * karr["bin_density"]
            dens_b_w += ch["weight"] * karr["bound_density"]
            arrays["k%d_omega" % kap] = karr["omega"]
            arrays["k%d_n_dust_killing" % kap] = karr["n_dust"]
            arrays["k%d_killing_bin_density" % kap] = karr["bin_density"]
            arrays["k%d_bound_density" % kap] = karr["bound_density"]
            # the shells: momentum and Killing content
            mom, nks = momentum_per_shell(dust, f, R, shells, M, k, thr)
            out["shells"] = []
            for s_, (lo, hi) in enumerate(shells):
                mo = mom[s_]
                mo["number_direct"] = float(n_site[kap][lo:hi].sum())
                mo["number_check"] = float(mo["number"] - mo["number_direct"])
                mo["killing_bins_N"] = [float(karr["bin_density"][i, lo:hi].sum()) for i in range(6)]
                mo["bound_N"] = float(karr["bound_density"][lo:hi].sum())
                out["shells"].append(dict(mo, shell=s_, R_lo=float(R[lo]), R_hi=float(R[hi - 1]), sites=hi - lo))
                nk_w[s_] += ch["weight"] * nks[s_]
                if a.store_nk == "on":
                    arrays["k%d_shell%02d_nk" % (kap, s_)] = nks[s_]
            nk_all = np.zeros(M)
            for j in range(dust.shape[1]):
                nk_all += f[j] * CK.orbital_momentum(dust[:, j], M)
            out["dust_whole_chain"] = CK.moments(nk_all, k, thr, M)
            arrays["k%d_dust_whole_chain_nk" % kap] = nk_all
            eps0 = IC.site_density(dust, f, h0)
            out["dust_h0_energy_total"] = float(eps0.sum())
            arrays["k%d_dust_eps_h0_site" % kap] = eps0
        else:
            d_low = d_high = None
        # the currents through the cuts
        out["currents"] = {}
        for lab, x in cuts:
            Jn = IO.OPS.number_current_csr(H, R, x)
            Je = IO.OPS.current_out_csr(H, R, x)
            Nx = currents[lab]["N_applied_at_cut"]
            cur = {"R_cut": x}
            if dust is not None:
                cur["dust_number"] = expect_rows(Jn, dust, f)
                cur["dust_number_straddle"], cur["straddling_nnz"], cur["range_sites_max"] = straddling_current(H, R, x, dust, f)
                cur["straddling_check"] = float(cur["dust_number"] - cur["dust_number_straddle"])
                cur["dust_energy"] = expect_rows(Je, dust, f)
                cur["dust_FM"] = cur["dust_energy"] / Nx if Nx else None
                cur["dust_number_low"] = expect_rows(Jn, d_low, f)
                cur["dust_number_high"] = expect_rows(Jn, d_high, f)
                cur["dust_number_cross"] = cur["dust_number"] - cur["dust_number_low"] - cur["dust_number_high"]
                cur["dust_energy_low"] = expect_rows(Je, d_low, f)
                cur["dust_energy_high"] = expect_rows(Je, d_high, f)
                cur["dust_energy_cross"] = cur["dust_energy"] - cur["dust_energy_low"] - cur["dust_energy_high"]
                for part in ("low", "high", "cross"):
                    cur["dust_FM_" + part] = (cur["dust_energy_" + part] / Nx) if Nx else None
                currents[lab]["straddling_check_max"] = max(currents[lab]["straddling_check_max"], abs(cur["straddling_check"]))
                currents[lab]["range_sites_max"] = max(currents[lab]["range_sites_max"], cur["range_sites_max"])
            else:
                for key in ("dust_number", "dust_number_straddle", "dust_energy", "dust_FM", "dust_number_low", "dust_number_high", "dust_number_cross", "dust_energy_low", "dust_energy_high", "dust_energy_cross", "dust_FM_low", "dust_FM_high", "dust_FM_cross"):
                    cur[key] = 0.0
            ones_R2 = np.ones(R2.shape[1])
            cur["C_number"] = expect_rows(Jn, Phi, f_all)
            cur["R2_number"] = expect_rows(Jn, R2, ones_R2)
            cur["dev_number"] = cur["C_number"] - cur["R2_number"]
            cur["C_energy"] = expect_rows(Je, Phi, f_all)
            cur["R2_energy"] = expect_rows(Je, R2, ones_R2)
            cur["dev_energy"] = cur["C_energy"] - cur["R2_energy"]
            cur["dev_FM"] = (cur["dev_energy"] / Nx) if Nx else None
            cur["pairs_number"] = cur["dev_number"] - cur["dust_number"]
            cur["pairs_FM"] = (cur["dev_FM"] - cur["dust_FM"]) if (cur["dev_FM"] is not None and cur["dust_FM"] is not None) else None
            out["currents"][lab] = cur
            currents[lab]["channels"][str(int(kap))] = cur
            for key in CUR_KEYS:
                if cur.get(key) is not None:
                    currents[lab]["weighted"][key] += ch["weight"] * cur[key]
        out["seconds"] = time.time() - t1
        chans.append(out)
        ka = out.get("killing") or {}
        log("  kappa %d: %d dust columns; %s; Killing bins N %s (eig vs stored %s); bound N %s R_90 %s omega/m %s; dust current at the dense outer root %s (straddling check %s); %.0f s" % (
            kap, rd.n_dust[kap], out["orthonormalization"], ("[%s]" % ", ".join("%.4f" % q["N"] for q in ka["bins"])) if ka else "-", IO.fx(ka.get("eig_vs_stored"), "%.1e") if ka else "-",
            IO.fx((ka.get("bound") or {}).get("N"), "%.4f") if ka else "-", IO.fx((ka.get("bound") or {}).get("R_90"), "%.2f") if ka else "-", IO.fx((ka.get("bound") or {}).get("mean_omega_over_m"), "%.4f") if ka else "-",
            IO.fx((out["currents"].get("dense_outer") or {}).get("dust_number"), "%+.4e"), IO.fx((out["currents"].get("dense_outer") or {}).get("straddling_check"), "%.1e"), out["seconds"]))
        del Phi, ch, R2, dust, d_low, d_high
    row["channels"] = chans
    # the weighted shells and spectra
    for s_, (lo, hi) in enumerate(shells):
        mo = CK.moments(nk_w[s_], k, thr, M)
        mo.update({"shell": s_, "R_lo": float(R[lo]), "R_hi": float(R[hi - 1]), "sites": hi - lo, "number_direct": float(n_w[lo:hi].sum()), "killing_bins_N": [float(dens_w[i, lo:hi].sum()) for i in range(6)],
                   "bound_N": float(dens_b_w[lo:hi].sum())})
        mo["number_check"] = float(mo["number"] - mo["number_direct"])
        shell_rows.append(mo)
        if a.store_nk == "on":
            arrays["shell%02d_nk_weighted" % s_] = nk_w[s_]
    arrays["killing_bin_density_weighted"] = dens_w
    arrays["bound_density_weighted"] = dens_b_w
    row["shells"] = {"width_sites": a.shell_sites, "count": len(shells), "weighted": shell_rows, "label": "the weighted rows sum the channels with the declared multiplicities 16 kappa (additive: counts and distributions, not correlations); the per-channel rows are in channels[].shells"}
    mean_om_b = (bound_w["E"] / bound_w["N"]) if bound_w["N"] > 0 else None
    Nb_w = float(dens_b_w.sum())
    row["killing"] = {"m": m, "b": b, "edges": killing_edges(m, b), "labels": list(BIN_LABELS), "bins_weighted": bins_w, "total_weighted": float(sum(q["N"] for q in bins_w)),
                      "bound_weighted": {"N": bound_w["N"], "mean_omega": mean_om_b, "mean_omega_over_m": (mean_om_b / m) if mean_om_b is not None else None,
                                         "R_50": IO.frac_radius(R, np.cumsum(dens_b_w) / Nb_w, 0.5) if Nb_w > 0 else None, "R_90": IO.frac_radius(R, np.cumsum(dens_b_w) / Nb_w, 0.9) if Nb_w > 0 else None,
                                         "r_turn_rs_nominal": turning_radius(rs, mean_om_b, m), "r_turn_2gE": turning_radius(two_gE, mean_om_b, m)},
                      "definition": "the dust's occupation per Killing level of h_m at the applied fields, binned half-open (lo, hi] at 0, +-m, +-b = sqrt(m^2 + 1/a^2); per shell Re Tr[Pi_S P_bin D], additive over shells and bins (a bin's shell content can be negative; the shell totals are the dust numbers)"}
    row["fountain"] = {"R_50_all": R50, "R_90_all": R90, "R_90_bound": row["killing"]["bound_weighted"]["R_90"], "N_bound_weighted": bound_w["N"], "N_all_weighted": tot,
                       "mean_omega_over_m_bound": row["killing"]["bound_weighted"]["mean_omega_over_m"], "rs_nominal": rs, "two_gE": two_gE,
                       "r_turn_rs_nominal": row["killing"]["bound_weighted"]["r_turn_rs_nominal"], "r_turn_2gE": row["killing"]["bound_weighted"]["r_turn_2gE"],
                       "definition": "the fountain's reach: R_90 of all the dust and of the bound part (0 < omega <= m) against r_s/(1 - (omega/m)^2) at the bound part's mean Killing energy, with the nominal r_s and the record's 2gE"}
    row["currents"] = currents
    row["currents_note"] = ("outward positive; the dust's number current Tr[D i[h_m, Pi_{R > R_cut}]] (page_ops_v6.number_current_csr) and the explicit straddling-bond sum over the bonds crossing the cut "
                            "(range-2: the hopping of h_m reaches 2 sites; the straddling matrix elements' count and range are printed), the Killing-energy current (i/2)[h_m^2, Pi_out] (current_out_csr), "
                            "F_M = the Killing-energy current over N_bar at the cut (the flux of rho + w J: eps_tot + w J_tot = eps_K/N per shell); the momentum split at pi/4 on the dust columns; "
                            "C the Loewdin-orthonormalized occupied set, R2 the continued reference, dev = C - R2 (the matter: the dust and the pairs); the weighted sums count the declared multiplicities once")
    row["seconds"] = time.time() - t0
    IO.save_outputs(pj, pz, row, arrays)
    with open(pm, "w") as fh:
        fh.write("\n".join(slice_md(row)) + "\n")
    co = currents.get("dense_outer", {}).get("weighted", {})
    log("%s T/r_s %.3f (%s; %s): %s; roots dense %s/%s applied %s/%s; R_50 %s R_90 %s bound R_90 %s r_turn %s; eps_c %s slope %s; weighted dust current at the dense outer root %s (F_M %s); %.0f s" % (
        TITLE, rd.T_over_rs, rd.dtype_str, entry["history"], row["limit_label"].split(" (")[0], IO.fx(Ro_d, "%.3f"), IO.fx(Ri_d, "%.3f"), IO.fx(Ro_a, "%.3f"), IO.fx(Ri_a, "%.3f"),
        IO.fx(R50, "%.2f"), IO.fx(R90, "%.2f"), IO.fx(row["fountain"]["R_90_bound"], "%.2f"), IO.fx(row["fountain"]["r_turn_rs_nominal"], "%.2f"),
        IO.fx((fits.get(row["core_fit"]["primary_window"]) or {}).get("epsilon_c_total"), "%.4f"), IO.fx(((fits.get(row["core_fit"]["primary_window"]) or {}).get("fit_dense") or {}).get("slope"), "%.4f"),
        IO.fx(co.get("dust_number"), "%+.4e"), IO.fx(co.get("dev_FM"), "%+.4e"), row["seconds"]))
    return row


# -------------------------------------------------------------------------------------------------- the per-slice md
def slice_md(row):
    fx = IO.fx
    L = ["# %s: %s at T/r_s %.3f (%s, %s)" % (TITLE, row["history"], row["T_over_rs"], row["slice"], row["dtype"]), "", INVENTORY_LINE, "", S8_LINE, "", S2_LINE, "",
         "Engine: %s. Label: %s (gate sigma |d_r v| = %.4f at R %.1f against %g). %s. %s." % (row["engine"], row["limit_label"], row["gate"]["sigma_abs_dv_max"], row["gate"]["at_R"], row["gate"]["threshold"],
                                                                                                  ("complex64 slice: the floor %s relative (sources), %.1e (orthonormality) applies to every column quantity" % (row["floor"]["source_rel_range"], row["floor"]["orthonormality_defect"])) if row["complex64"] else "complex128 slice",
                                                                                                  PROVENANCE), ""]
    rc = row["record"]
    ro = row["roots"]
    L += ["Record: E %s (E/E0 %s), 2gE %s, R_c %s, trapping zeros %s, kappa_T %s, min(N - |v|) %s at %s, converged %s. Roots: dense outer %s (R/(2gE) %s), dense inner %s (R/(2gE) %s); applied outer %s, applied inner %s (every applied zero: %s; finder: %s); previous record's dense outer %s, next %s." % (
        fx(rc["E"], "%.6f"), fx(rc["E_over_E0"], "%.6f"), fx(rc["two_gE"], "%.4f"), fx(rc["R_c"], "%.3f"), rc["trapping_zeros"], fx(rc["kappa_T"], "%.5f"), fx(rc["min_N_minus_absv"], "%+.5f"), fx(rc["r_min_N_minus_absv"], "%.2f"),
        rc["converged"], fx(ro["dense_outer"], "%.4f"), fx(ro["dense_outer_over_2gE"], "%.4f"), fx(ro["dense_inner"], "%.4f"), fx(ro["dense_inner_over_2gE"], "%.4f"), fx(ro["applied_outer"], "%.4f"), fx(ro["applied_inner"], "%.4f"),
        ", ".join("%.4f (%s)" % (z["R"], "rising" if z["slope_sign"] > 0 else "falling") for z in ro["applied_zeros"]) or "none", ro["applied_source"], fx(ro["prev_dense_outer"], "%.4f"), fx(ro["next_dense_outer"], "%.4f"))]
    if ro.get("inbox_T16"):
        ib = ro["inbox_T16"]
        L += ["The plan's quoted roots at 16 (inbox): applied outer %.4f a, dense %.4f a; computed minus quoted: %s, %s." % (ib["applied_outer"], ib["dense_outer"], fx(ib["computed_minus_inbox_applied"], "%+.4e"), fx(ib["computed_minus_inbox_dense"], "%+.4e"))]
    L += ["", "Fields: dense from %s; applied from %s." % (row["fields_source"]["dense"], row["fields_source"]["applied"]), "", "## The profile at the named radii", "",
          "| radius | R | N dense | v dense | w dense | N applied | v applied | w applied | n (shell) | n (density) | rho (density) | rho/n | J (density) | N_< | M_w dense | I integral | L_N dense | L_N applied |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, q in row["profile_at_radii"].items():
        L.append("| %s | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            lab, q["R"], fx(q["N_dense"], "%.5f"), fx(q["v_dense"], "%+.5f"), fx(q["w_dense"], "%+.5f"), fx(q["N_applied"], "%.5f"), fx(q["v_applied"], "%+.5f"), fx(q["w_applied"], "%+.5f"), fx(q["n_shell"], "%.4e"), fx(q["n_density"], "%.4e"),
            fx(q["rho_density"], "%.4e"), fx(q["rho_over_n"], "%.5f"), fx(q["J_density"], "%+.4e"), fx(q["N_enclosed"], "%.4f"), fx(q["M_w_dense"], "%.4f"), fx(q["I_integral"], "%.4f"), fx(q["L_N_dense"], "%.3f"), fx(q["L_N_applied"], "%.3f")))
    d = row["dust"]
    L += ["", "Dust: total %.4f (weighted 16 kappa; per channel %s), R_50 %s, R_90 %s." % (d["total_weighted"], d["per_channel_total"], fx(d["R_50"], "%.3f"), fx(d["R_90"], "%.3f")), "", "## The core: ln N against ln N_<", ""]
    cf = row["core_fit"]
    L += ["Primary window %s; rho/n at the innermost site with dust: %s at R %s." % (cf["primary_window"], fx((cf["rho_over_n_innermost"] or {}).get("value"), "%.5f"), fx((cf["rho_over_n_innermost"] or {}).get("R"), "%.0f")), "",
          "| window | sites (in fit) | epsilon_c total | epsilon_c dust part | slope dense (se; r2; n) | slope applied | (eps_c - m)/(2m) | eps_c/(2m) | dust part: (eps_c - m)/(2m); eps_c/(2m) | per channel slopes (dense) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, w in cf["windows"].items():
        fd, fa = w["fit_dense"], w["fit_applied"]
        L.append("| %s: %s | %d (%d) | %s | %s | %s (%s; %s; %d) | %s | %s | %s | %s; %s | %s |" % (
            name, w["label"], w["sites"], w["sites_in_fit"], fx(w.get("epsilon_c_total"), "%.5f"), fx(w.get("epsilon_c_dust"), "%.5f"), fx(fd["slope"], "%.4f"), fx(fd["slope_se"], "%.4f"), fx(fd["r2"], "%.4f"), fd["n"], fx(fa["slope"], "%.4f"),
            fx(w.get("exponent_eps_minus_m_over_2m"), "%.4f"), fx(w.get("exponent_eps_over_2m"), "%.4f"), fx(w.get("exponent_dust_eps_minus_m_over_2m"), "%.4f"), fx(w.get("exponent_dust_eps_over_2m"), "%.4f"),
            ", ".join("%s: %s" % (kk, fx(v["slope"], "%.3f")) for kk, v in w["per_channel_fit_dense"].items())))
    mf = row["mass_function"]
    L += ["", "## The mass function M(r) = r w^2/(2g) against the integral", ""]
    if mf.get("available"):
        L += ["Largest |M_w - I| inside R_c: dense %s, applied %s (collar form %s); I at the chain end %s against E %s (difference %s). The record's W_over_2g_at_Rt %s and misner_sharp_at_Rt %s (M1_balance %s) beside this readout's values at the dense outer root." % (
            fx(mf.get("max_abs_mismatch_inside_Rc_dense"), "%.4e"), fx(mf.get("max_abs_mismatch_inside_Rc_applied"), "%.4e"), fx(mf.get("max_abs_mismatch_collar_inside_Rc_dense"), "%.4e"), fx(mf.get("I_chain_end"), "%.5f"), fx(mf.get("E_record"), "%.5f"),
            fx(mf.get("I_chain_end_minus_E"), "%+.4e"), fx(mf["record"]["W_over_2g_at_Rt"], "%.5f"), fx(mf["record"]["misner_sharp_at_Rt"], "%.5f"), fx(mf["record"]["M1_balance"], "%s")), "",
              "| radius | R | R/(2g) | M_w dense | M_w applied | I (dense w) | I (applied w) | I collar form |", "|---|---|---|---|---|---|---|---|"]
        for key, q in mf.items():
            if key.startswith("at_"):
                L.append("| %s | %.3f | %.4f | %s | %s | %s | %s | %s |" % (key[3:], q["R"], q["R_over_2g"], fx(q["M_w_dense"], "%.4f"), fx(q["M_w_applied"], "%.4f"), fx(q["I"], "%.4f"), fx(q["I_applied_w"], "%.4f"), fx(q["I_collar"], "%.4f")))
    else:
        L.append("Not available: the fields npz holds no eps_tot, J_tot at this record.")
    gl = row["gradient_length"]
    L += ["", "## The clock's gradient length N/|N'| against sigma = %g" % gl["dense"]["sigma"], "", "| fields | min over the chain (R) | sigma/min | min inside the outer root (R) | sites with L_N < sigma | at dense outer | at dense inner | at applied outer | at R_50 | at R_90 | at R_c |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ("dense", "applied"):
        q = gl[lab]
        ar = q["at_radii"]
        L.append("| %s | %s (%s) | %s | %s (%s) | %d | %s | %s | %s | %s | %s | %s |" % (lab, fx(q.get("min_chain"), "%.3f"), fx(q.get("min_chain_R"), "%.0f"), fx(q.get("sigma_over_min_chain"), "%.4f"), fx(q.get("min_inside_root"), "%.3f"), fx(q.get("min_inside_root_R"), "%.0f"),
                                                                                       q["sites_L_below_sigma"], fx(ar.get("dense_outer"), "%.3f"), fx(ar.get("dense_inner"), "%.3f"), fx(ar.get("applied_outer"), "%.3f"), fx(ar.get("R_50"), "%.3f"), fx(ar.get("R_90"), "%.3f"), fx(ar.get("R_c"), "%.3f")))
    fo = row["fountain"]
    L += ["", "## The fountain's reach", "", "R_90 of all the dust %s; the bound part (0 < omega <= m): N %s of %s, R_90 %s, mean omega/m %s, r_s/(1 - (omega/m)^2) = %s with the nominal r_s %g and %s with 2gE %s." % (
        fx(fo["R_90_all"], "%.3f"), fx(fo["N_bound_weighted"], "%.4f"), fx(fo["N_all_weighted"], "%.4f"), fx(fo["R_90_bound"], "%.3f"), fx(fo["mean_omega_over_m_bound"], "%.5f"), fx(fo["r_turn_rs_nominal"], "%.3f"), fo["rs_nominal"], fx(fo["r_turn_2gE"], "%.3f"), fx(fo["two_gE"], "%.3f"))]
    ki = row["killing"]
    L += ["", "## The Killing spectrum of the dust (bins half-open at 0, +-m = +-%g, +-b = +-%.6f)" % (ki["m"], ki["b"]), "", "| kappa | n_dust | eig vs stored e_k | %s | completeness | site-density check | bound N | bound R_90 | bound omega/m | r_turn nominal | r_turn 2gE |" % " | ".join(BIN_LABELS), "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in row["channels"]:
        ka = c.get("killing")
        if not ka:
            continue
        bd = ka["bound"]
        L.append("| %d | %d | %s | %s | %.1e | %.1e | %.4f | %s | %s | %s | %s |" % (c["kappa"], c["n_dust"], fx(ka["eig_vs_stored"], "%.1e"), " | ".join("%.4f" % q["N"] for q in ka["bins"]), ka["completeness_check"], ka["site_density_check"], bd["N"], fx(bd["R_90"], "%.3f"), fx(bd["mean_omega_over_m"], "%.5f"), fx(bd["r_turn_rs_nominal"], "%.3f"), fx(bd["r_turn_2gE"], "%.3f")))
    L.append("| weighted | - | - | %s | - | - | %.4f | %s | %s | %s | %s |" % (" | ".join("%.4f" % q["N"] for q in ki["bins_weighted"]), ki["bound_weighted"]["N"], fx(ki["bound_weighted"]["R_90"], "%.3f"), fx(ki["bound_weighted"]["mean_omega_over_m"], "%.5f"), fx(ki["bound_weighted"]["r_turn_rs_nominal"], "%.3f"), fx(ki["bound_weighted"]["r_turn_2gE"], "%.3f")))
    L += ["", "Per dust orbital (channel, column): mean omega/m (spread), R_90, bins, bound weight, bound omega/m, bound R_90, r_turn (nominal; 2gE):", ""]
    for c in row["channels"]:
        for o in (c.get("killing") or {}).get("orbitals", []):
            L.append("- kappa %d column %d: omega/m %s (%s), R_90 %s, bins %s, bound weight %.4f, bound omega/m %s, bound R_90 %s, r_turn %s; %s." % (
                c["kappa"], o["column"], fx(o["mean_omega_over_m"], "%.5f"), fx(o["spread"], "%.4f"), fx(o["R_90"], "%.2f"), ["%.4f" % q for q in o["bins_N"]], o["bound_weight"], fx(o["bound_mean_omega_over_m"], "%.5f"), fx(o["bound_R_90"], "%.2f"), fx(o["r_turn_rs_nominal"], "%.2f"), fx(o["r_turn_2gE"], "%.2f")))
    sh = row["shells"]
    L += ["", "## The shells (%d sites each): the dust's radial momentum (reduced zone, share at |ap| > %g pi) and its Killing content; weighted rows" % (sh["width_sites"], row["momentum_grid"]["k_threshold_over_pi"]), "",
          "| shell | R range | n dust | number check | share |ap| > pi/4 | <cos 2ak> | <sin 2ak> | <|ap|> | %s | bound N |" % " | ".join(BIN_LABELS), "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s_ in sh["weighted"]:
        L.append("| %d | %.0f to %.0f | %.4f | %.1e | %s | %s | %s | %s | %s | %.4f |" % (s_["shell"], s_["R_lo"], s_["R_hi"], s_["number"], s_["number_check"], fx(s_["frac_abs_p_above"], "%.4f"), fx(s_["cos2k"], "%+.4f"), fx(s_["sin2k"], "%+.4f"), fx(s_["mean_abs_p"], "%.4f"),
                                                                                        " | ".join("%.4f" % q for q in s_["killing_bins_N"]), s_["bound_N"]))
    L += ["", "Per channel (no multiplicity): shell | n | share |ap| > pi/4 | <cos 2ak> | bins:", ""]
    for c in row["channels"]:
        if not c.get("shells"):
            continue
        L.append("- kappa %d: %s; whole chain share %s, <cos 2ak> %s." % (c["kappa"], "; ".join("%d: %.4f, %s, %s, %s" % (s_["shell"], s_["number"], fx(s_["frac_abs_p_above"], "%.3f"), fx(s_["cos2k"], "%+.3f"), ["%.3f" % q for q in s_["killing_bins_N"]]) for s_ in c["shells"]),
                                                                           fx(c["dust_whole_chain"]["frac_abs_p_above"], "%.4f"), fx(c["dust_whole_chain"]["cos2k"], "%+.4f")))
    L += ["", "## The currents through the cuts (outward positive; per channel without multiplicity; the weighted row counts the declared multiplicities once)", "", row["currents_note"], ""]
    for lab in CUT_ORDER:
        cu = row["currents"].get(lab)
        if not cu:
            continue
        L += ["### %s: R_cut %.4f (%s); N_bar at the cut %s; straddling check max %.1e, range %d sites" % (lab, cu["R_cut"], cu["source"], fx(cu["N_applied_at_cut"], "%.5f"), cu["straddling_check_max"], cu["range_sites_max"]), "",
              "| kappa | dust number (ops) | straddling sum | dust number: low |ap| <= pi/4; high; cross | dust energy | dust F_M (low; high; cross) | C number | R2 number | dev number | pairs number | dev energy | dev F_M |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for kk in sorted(cu["channels"], key=int):
            q = cu["channels"][kk]
            L.append("| %s | %+.5e | %+.5e | %+.4e; %+.4e; %+.4e | %+.5e | %s (%s; %s; %s) | %+.5e | %+.5e | %+.5e | %+.5e | %+.5e | %s |" % (
                kk, q["dust_number"], q["dust_number_straddle"], q["dust_number_low"], q["dust_number_high"], q["dust_number_cross"], q["dust_energy"], fx(q["dust_FM"], "%+.5e"), fx(q["dust_FM_low"], "%+.3e"), fx(q["dust_FM_high"], "%+.3e"), fx(q["dust_FM_cross"], "%+.3e"),
                q["C_number"], q["R2_number"], q["dev_number"], q["pairs_number"], q["dev_energy"], fx(q["dev_FM"], "%+.5e")))
        w = cu["weighted"]
        L.append("| weighted | %+.5e | %+.5e | %+.4e; %+.4e; %+.4e | %+.5e | %+.5e (%+.3e; %+.3e; %+.3e) | %+.5e | %+.5e | %+.5e | %+.5e | %+.5e | %+.5e |" % (
            w["dust_number"], w["dust_number_straddle"], w["dust_number_low"], w["dust_number_high"], w["dust_number_cross"], w["dust_energy"], w["dust_FM"], w["dust_FM_low"], w["dust_FM_high"], w["dust_FM_cross"], w["C_number"], w["R2_number"], w["dev_number"], w["pairs_number"], w["dev_energy"], w["dev_FM"]))
        L.append("")
    L += ["## Checks", "", "- orthonormality: stored %s, after the Loewdin step %s; the fields npz's applied fields against the slice's: %s; %.0f s." % (
        fx(max(c["stored_orthonormality"] for c in row["channels"]), "%.1e"), fx(max((c["loewdin_orthonormality"] or 0.0) for c in row["channels"]), "%.1e"), fx(row["fields_source"].get("slice_vs_fields_npz_applied_max_abs"), "%.1e"), row["seconds"]),
          "- every number above is printed by the script from the stored slice and the records; the reading is the lead's."]
    return L


# ------------------------------------------------------------------------------------------------ the records pass
def scheme_line(ctx, H, a):
    """the step scheme named in 1 line from the records' solver key, the fields npz's solver and label, the records' args and the engine source lines matched on this box."""
    J = {"args": {}, "records": []}
    if H.get("records_json") and os.path.isfile(H["records_json"]):
        try:
            J = ctx.records({"records_json": H["records_json"]})
        except Exception:  # noqa: BLE001
            pass
    args = J.get("args", {}) or {}
    recs = [r for r in J.get("records", []) if r.get("E") is not None]
    native = [r for r in recs if r.get("engine") or r.get("record_label")]
    rec = native[0] if native else (recs[-1] if recs else {})
    solver = rec.get("solver")
    fsolver = flabel = None
    try:
        p, Z = ctx.fields_file({"fields_npz": H.get("fields_npz")})
        if Z is not None:
            fsolver = str(Z["solver"]) if "solver" in Z.files else None
            flabel = str(Z["label"]) if "label" in Z.files else None
    except Exception:  # noqa: BLE001
        pass
    names = sorted(set(re.findall(r"engine_collapse_v\d+[a-z]*", str(H.get("engine", "")) + " " + str(rec.get("engine", "")))))
    dirs = [d for d in str(a.engine_code or "").split(os.pathsep) if d] + [IO.ENGINE_CODE, os.path.join(IO.ENGINE_CODE, "code"), str(a.page_code or "")]
    hits, files = [], []
    for name in names:
        for d in dirs:
            path = os.path.join(d, name + ".py")
            if os.path.isfile(path) and path not in files:
                files.append(path)
                try:
                    with open(path, errors="replace") as fh:
                        for i, line in enumerate(fh):
                            if re.search(r"midpoint|frozen|exponential euler|expm|cheb_apply", line, re.I):
                                hits.append("%s:%d: %s" % (name, i + 1, line.strip()[:140]))
                except Exception:  # noqa: BLE001
                    pass
    hits = hits[:16]
    if any("midpoint" in h.lower() for h in hits):
        named, rule = "the midpoint of Eq. dr-midpoint", "a source line of the records' engine matches 'midpoint'"
    elif hits:
        named, rule = "frozen-field exponential Euler", "the records' engine source matches a propagator ('expm', 'cheb_apply', 'frozen' or 'exponential Euler') and no 'midpoint'"
    else:
        named, rule = "not determinable from the files on this box (no engine source matched)", "no engine source found in %s" % dirs
    return ("Step scheme (1 line): named '%s' by the rule: %s; the record's solver key: %s; the fields npz's solver: %s, label: %s; the records' args: maxit %s, mix %s, stop_E %s, stop_fields %s, "
            "m1_simpson_steps %s, c_LZ %s, window_max %s, tau %s; the engine source lines matched: %s. The naming is mechanical (the rule above); the lead reads the evidence." % (
                named, rule, json.dumps(solver, default=IO.jdefault) if solver is not None else "absent", fsolver, flabel, args.get("maxit"), args.get("mix"), args.get("stop_E"), args.get("stop_fields"),
                args.get("m1_simpson_steps"), args.get("c_LZ"), args.get("window_max"), args.get("tau"), " || ".join(hits) if hits else "none"))


def records_pass(ctx, H, a, slice_rows, log):
    """every record of the history: the dense fields' M(r) match and L_N; the root attribution between consecutive records holding a dense outer root."""
    if not H.get("records_json") or not os.path.isfile(H["records_json"]):
        return {"available": False, "reason": "no records json for this history"}, {}
    entry0 = {"records_json": H["records_json"], "fields_npz": H.get("fields_npz")}
    try:
        J = ctx.records(entry0)
        p, Z = ctx.fields_file(entry0)
    except Exception as ex:  # noqa: BLE001
        return {"available": False, "reason": "%s: %s" % (type(ex).__name__, ex)}, {}
    recs = sorted([r for r in J.get("records", []) if r.get("E") is not None], key=lambda r: float(r["T"]))
    Tz = np.asarray(Z["T"], float).reshape(-1)
    R = np.asarray(Z["R"], float).reshape(-1)
    F = {key: (np.asarray(Z[key], float) if key in Z.files else None) for key in ("N", "v", "N_applied", "v_applied", "eps_tot", "J_tot", "eps_pairs", "J_pairs")}
    nodes_all = None
    if all(key in Z.files for key in ("r_nodes", "N_nodes", "v_nodes")):
        try:
            nodes_all = (Z["r_nodes"], Z["N_nodes"], Z["v_nodes"])
        except Exception:  # noqa: BLE001
            nodes_all = None
    g, ell, rs = float(ctx.g), float(ctx.ell), float(ctx.rs)
    by_T = {round(float(r["T_over_rs"]), 6): r for r in slice_rows}
    rows, pairs = [], []
    prev = None
    t0 = time.time()
    for rec in recs:
        T = float(rec["T"])
        k = int(np.argmin(np.abs(Tz - T)))
        if abs(float(Tz[k]) - T) > 1e-6:
            continue
        E = float(rec["E"])
        two_gE = 2.0 * g * E
        zeros = [float(z) for z in (rec.get("trapping_zeros") or [])]
        Ro = max(zeros) if zeros else None
        Ri = min(zeros) if len(zeros) >= 2 else None
        Rc = float(rec["R_c"]) if rec.get("R_c") is not None else None
        N_app = F["N_applied"][k] if F["N_applied"] is not None else (F["N"][k] if F["N"] is not None else None)
        v_app = F["v_applied"][k] if F["v_applied"] is not None else (F["v"][k] if F["v"] is not None else None)
        gate = gate_of(R, v_app, ell, a.gate, Ro if Ro is not None else Rc) if v_app is not None else None
        nodes = None
        if nodes_all is not None:
            try:
                rn = np.asarray(nodes_all[0][k], float).reshape(-1)
                Nn = np.asarray(nodes_all[1][k], float).reshape(-1)
                vn = np.asarray(nodes_all[2][k], float).reshape(-1)
                if rn.size >= 2 and rn.size == Nn.size == vn.size:
                    nodes = (rn, Nn, vn)
            except Exception:  # noqa: BLE001
                nodes = None
        if nodes is not None:
            N_d, v_d = dense_at_sites(nodes, R)
            src = "nodes"
        elif F["N"] is not None:
            N_d, v_d = F["N"][k], F["v"][k]
            src = "site fields"
        else:
            N_d, v_d = N_app, v_app
            src = "applied"
        with np.errstate(divide="ignore", invalid="ignore"):
            w_d = v_d / N_d
        r = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": rec.get("E_over_E0"), "two_gE": two_gE, "R_t": Ro, "R_in": Ri, "R_t_over_2gE": (Ro / two_gE) if Ro is not None else None,
             "R_in_over_2gE": (Ri / two_gE) if Ri is not None else None, "R_c": Rc, "dense_source": src, "engine": rec.get("engine") or rec.get("record_label") or "absent (the replayed parent's record)",
             "gate_sigma_abs_dv_max": gate["sigma_abs_dv_max"] if gate else None, "limit_label": label_of(gate) if gate else "no applied fields in the fields npz", "converged": rec.get("converged"),
             "kappa_T": rec.get("kappa_T"), "min_N_minus_absv": rec.get("min_N_minus_absv"), "dE_dT": rec.get("dE_dT"), "slice": round(T / rs, 6) in by_T}
        I_d = None
        if F["eps_tot"] is not None and F["J_tot"] is not None:
            M_w, I_d, src_d = mass_function(R, w_d, F["eps_tot"][k], F["J_tot"][k], g)
            inside = (R < Rc) if Rc is not None else np.ones(len(R), bool)
            r.update({"mismatch_max_inside_Rc": float(np.max(np.abs((M_w - I_d)[inside]))) if inside.any() else None, "I_chain_end_minus_E": float(I_d[-1]) - E,
                      "M_w_at_Rt": interp_at(Ro, R, M_w), "I_at_Rt": interp_at(Ro, R, I_d), "I_collar_at_Rt": (E - (float(I_d[-1]) - interp_at(Ro, R, I_d))) if Ro is not None else None,
                      "record_W_over_2g_at_Rt": rec.get("W_over_2g_at_Rt"), "record_misner_sharp_at_Rt": rec.get("misner_sharp_at_Rt")})
        if nodes is not None:
            o = np.argsort(nodes[0])
            Ln = gradient_length(nodes[0][o], nodes[1][o])
            L_d = np.interp(R, nodes[0][o], np.where(np.isfinite(Ln), Ln, 1e300))
        else:
            L_d = gradient_length(R, N_d)
        ls = L_stats(L_d, R, ell, Ro, {"R_t": Ro, "R_in": Ri})
        r.update({"L_N_min_chain": ls["min_chain"], "L_N_min_chain_R": ls["min_chain_R"], "L_N_min_inside_Rt": ls.get("min_inside_root"), "L_N_at_Rt": ls["at_radii"]["R_t"], "L_N_at_Rin": ls["at_radii"]["R_in"],
                  "sites_L_below_sigma": ls["sites_L_below_sigma"]})
        # the attribution step
        if prev is not None and prev["R_t"] is not None and Ro is not None:
            dt = T - prev["T"]
            R_old, R_new = prev["R_t"], Ro
            w_old_at_new = interp_at(R_old, R, w_d)
            M_old_at_new = R_old * w_old_at_new ** 2 / (2.0 * g)
            w_new_at_old = interp_at(R_new, R, prev["w_d"])
            M_new_at_old = R_new * w_new_at_old ** 2 / (2.0 * g)
            pair = {"T_from_over_rs": prev["T_over_rs"], "T_to_over_rs": T / rs, "dt": dt, "R_t_from": R_old, "R_t_to": R_new, "R_t_from_over_2gE": prev["R_t_over_2gE"], "R_t_to_over_2gE": r["R_t_over_2gE"],
                    "dM_root": (R_new - R_old) / (2.0 * g), "dM_fixed_radius": M_old_at_new - R_old / (2.0 * g), "sweep_fields": R_new / (2.0 * g) - M_old_at_new,
                    "sweep_integral": (interp_at(R_new, R, I_d) - interp_at(R_old, R, I_d)) if I_d is not None else None,
                    "alt_sweep_fields_at_old_clock": M_new_at_old - R_old / (2.0 * g), "alt_dM_fixed_radius_at_new_root": R_new / (2.0 * g) - M_new_at_old,
                    "dE": E - prev["E"], "dE_dT_times_dt_record": (float(rec["dE_dT"]) * dt) if isinstance(rec.get("dE_dT"), (int, float)) else None,
                    "limit_label_to": r["limit_label"], "limit_label_from": prev["limit_label"], "flux": None}
            sa = by_T.get(round(prev["T_over_rs"], 6))
            sb = by_T.get(round(T / rs, 6))
            fl = []
            if sa and (sa.get("currents") or {}).get("dense_outer"):
                fl.append(("the slice at the step's start, its dense outer root", sa["currents"]["dense_outer"]))
            if sb and (sb.get("currents") or {}).get("prev_dense_outer"):
                fl.append(("the slice at the step's end, the previous record's dense outer root", sb["currents"]["prev_dense_outer"]))
            if fl:
                def avg(key):
                    vals = [q["weighted"].get(key) for _, q in fl if q["weighted"].get(key) is not None]
                    return float(sum(vals) / len(vals)) if vals else None
                flux = {"source": "; ".join(s for s, _ in fl), "ends": len(fl), "R_cut": [q["R_cut"] for _, q in fl]}
                for key in ("dev_FM", "dust_FM", "dust_FM_low", "dust_FM_high", "dust_FM_cross", "dust_number", "dust_number_low", "dust_number_high", "dust_number_cross", "dev_number", "pairs_FM"):
                    flux[key + "_out"] = avg(key)
                flux["term_matter"] = (-flux["dev_FM_out"] * dt) if flux["dev_FM_out"] is not None else None
                flux["term_dust"] = (-flux["dust_FM_out"] * dt) if flux["dust_FM_out"] is not None else None
                flux["dust_number_crossed_out"] = (flux["dust_number_out"] * dt) if flux["dust_number_out"] is not None else None
                flux["definition"] = "the flux term = minus the outward current of rho + w J (F_M = the Killing-energy current over N_bar at the cut) times dt: the matter (dev = C - R2 R2^dagger) and the dust's part, the dust split by momentum at pi/4; averaged over the ends where both slices exist"
                pair["flux"] = flux
                pair["residual_matter"] = (pair["dM_fixed_radius"] - pair["dE"] - flux["term_matter"]) if flux["term_matter"] is not None else None
                pair["residual_dust"] = (pair["dM_fixed_radius"] - pair["dE"] - flux["term_dust"]) if flux["term_dust"] is not None else None
            else:
                pair["residual_matter"] = pair["residual_dust"] = None
            pair["identity_check"] = pair["dM_root"] - pair["dM_fixed_radius"] - pair["sweep_fields"]
            pairs.append(pair)
        rows.append(r)
        prev = dict(r, w_d=w_d)
    sums = {}
    if pairs:
        for key in ("dM_root", "dM_fixed_radius", "sweep_fields", "dE", "alt_sweep_fields_at_old_clock", "alt_dM_fixed_radius_at_new_root"):
            sums[key] = float(sum(q[key] for q in pairs))
        sums["sweep_integral"] = float(sum(q["sweep_integral"] for q in pairs if q["sweep_integral"] is not None))
        sums["steps"] = len(pairs)
        sums["steps_with_flux"] = int(sum(1 for q in pairs if q["flux"]))
        sums["term_matter_where_present"] = float(sum(q["flux"]["term_matter"] for q in pairs if q["flux"] and q["flux"]["term_matter"] is not None))
        sums["dM_fixed_where_flux_present"] = float(sum(q["dM_fixed_radius"] for q in pairs if q["flux"]))
        sums["dE_where_flux_present"] = float(sum(q["dE"] for q in pairs if q["flux"]))
        sums["residual_matter_where_present"] = float(sum(q["residual_matter"] for q in pairs if q["residual_matter"] is not None))
        sums["max_abs_identity_check"] = float(max(abs(q["identity_check"]) for q in pairs))
        sums["T_first_over_rs"], sums["T_last_over_rs"] = pairs[0]["T_from_over_rs"], pairs[-1]["T_to_over_rs"]
    out = {"available": True, "records": len(rows), "rows": rows, "pairs": pairs, "sums": sums, "seconds": time.time() - t0, "fields_npz": p, "records_json": H["records_json"],
           "definitions": "Delta M at the root = (R_t(k) - R_t(k-1))/(2 g) (w = -1 at a dense root); = the fixed-radius change M(R_t(k-1), t_k) - R_t(k-1)/(2 g) plus the sweep term M(R_t(k), t_k) - M(R_t(k-1), t_k) on the dense fields at t_k (identity_check = 0); the sweep's integral form sum over the swept shells of eps_tot + w J_tot at t_k; the other ordering (the sweep at t_{k-1}, the fixed radius at R_t(k)) beside; the fixed-radius change against dE = E(k) - E(k-1) (the step's collar deficit is -dE) and the inward flux of rho + w J from the stored states where a slice exists at an end; residual = dM_fixed - dE - flux term"}
    arrays = {}
    if rows:
        for key in ("T_over_rs", "E", "two_gE", "R_t", "R_in", "R_t_over_2gE", "R_in_over_2gE", "gate_sigma_abs_dv_max", "mismatch_max_inside_Rc", "M_w_at_Rt", "I_at_Rt", "I_collar_at_Rt", "L_N_min_chain", "L_N_min_inside_Rt", "L_N_at_Rt", "I_chain_end_minus_E"):
            arrays["rec_" + key] = np.array([(q.get(key) if q.get(key) is not None else np.nan) for q in rows], float)
        arrays["rec_past_limit"] = np.array([1 if q["limit_label"] == LIMIT_LABEL else 0 for q in rows])
    if pairs:
        for key in ("T_from_over_rs", "T_to_over_rs", "R_t_from", "R_t_to", "dM_root", "dM_fixed_radius", "sweep_fields", "sweep_integral", "dE", "residual_matter", "residual_dust", "alt_sweep_fields_at_old_clock", "alt_dM_fixed_radius_at_new_root"):
            arrays["pair_" + key] = np.array([(q.get(key) if q.get(key) is not None else np.nan) for q in pairs], float)
        arrays["pair_term_matter"] = np.array([(q["flux"]["term_matter"] if (q["flux"] and q["flux"]["term_matter"] is not None) else np.nan) for q in pairs], float)
        arrays["pair_term_dust"] = np.array([(q["flux"]["term_dust"] if (q["flux"] and q["flux"]["term_dust"] is not None) else np.nan) for q in pairs], float)
    log("records pass %s: %d records, %d attribution steps (%d with a stored-state flux), %.0f s" % (H["key"], len(rows), len(pairs), sums.get("steps_with_flux", 0), out["seconds"]))
    return out, arrays


# ------------------------------------------------------------------------------------------------------------ the tables
def history_md(hk, H, rows, recs, scheme, a):
    fx = IO.fx
    L = ["# %s: %s (%s)" % (TITLE, H["label"], CODE), "", INVENTORY_LINE, "", S8_LINE, "", S2_LINE, "", "Engine: %s." % H["engine"], "", scheme, "", PROVENANCE + ". " + CORRELATION_NOTE + ".", "",
         "The momentum convention is core_kinematics_v2's reduced zone (%s)." % CK.ZONE, ""]
    if rows:
        L += ["## The slices", "", "| T/r_s | dtype | label | gate max | dense roots (outer; inner) | applied roots (outer; inner) | R_c | 2gE | R_t/(2gE) | R_in/(2gE) | E/E0 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ro, rc = r["roots"], r["record"]
            L.append("| %.3f | %s | %s | %.4f | %s; %s | %s; %s | %s | %s | %s | %s | %s |" % (r["T_over_rs"], r["dtype"], r["limit_label"].split(" (")[0], r["gate"]["sigma_abs_dv_max"], fx(ro["dense_outer"], "%.4f"), fx(ro["dense_inner"], "%.4f"),
                                                                                  fx(ro["applied_outer"], "%.4f"), fx(ro["applied_inner"], "%.4f"), fx(rc["R_c"], "%.3f"), fx(rc["two_gE"], "%.4f"), fx(ro["dense_outer_over_2gE"], "%.4f"), fx(ro["dense_inner_over_2gE"], "%.4f"), fx(rc["E_over_E0"], "%.6f")))
        L += ["", "## The fountain's reach R_90(t) against r_s/(1 - (omega/m)^2) for the bound part", "", "| T/r_s | R_50 | R_90 all | R_90 bound | N bound (weighted) of N all | mean omega/m bound | r_turn with r_s %g | r_turn with 2gE | 2gE |" % rows[0]["fountain"]["rs_nominal"], "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            fo = r["fountain"]
            L.append("| %.3f | %s | %s | %s | %s of %s | %s | %s | %s | %s |" % (r["T_over_rs"], fx(fo["R_50_all"], "%.3f"), fx(fo["R_90_all"], "%.3f"), fx(fo["R_90_bound"], "%.3f"), fx(fo["N_bound_weighted"], "%.4f"), fx(fo["N_all_weighted"], "%.4f"),
                                                                                fx(fo["mean_omega_over_m_bound"], "%.5f"), fx(fo["r_turn_rs_nominal"], "%.3f"), fx(fo["r_turn_2gE"], "%.3f"), fx(fo["two_gE"], "%.3f")))
        L += ["", "## The core: the slope of ln N against ln N_< (the primary window; the other windows in the per-slice md)", "", "| T/r_s | window | sites in fit | epsilon_c total | epsilon_c dust part | slope dense (se; r2) | slope applied | (eps_c - m)/(2m) | eps_c/(2m) | dust part: (eps_c - m)/(2m); eps_c/(2m) | per channel slopes |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            cf = r["core_fit"]
            w = cf["windows"].get(cf["primary_window"]) if cf["primary_window"] else None
            if not w:
                L.append("| %.3f | - | - | - | - | - | - | - | - | - | - |" % r["T_over_rs"])
                continue
            fd, fa = w["fit_dense"], w["fit_applied"]
            L.append("| %.3f | %s | %d | %s | %s | %s (%s; %s) | %s | %s | %s | %s; %s | %s |" % (r["T_over_rs"], w["label"], w["sites_in_fit"], fx(w.get("epsilon_c_total"), "%.5f"), fx(w.get("epsilon_c_dust"), "%.5f"), fx(fd["slope"], "%.4f"), fx(fd["slope_se"], "%.4f"), fx(fd["r2"], "%.4f"),
                                                                                      fx(fa["slope"], "%.4f"), fx(w.get("exponent_eps_minus_m_over_2m"), "%.4f"), fx(w.get("exponent_eps_over_2m"), "%.4f"), fx(w.get("exponent_dust_eps_minus_m_over_2m"), "%.4f"), fx(w.get("exponent_dust_eps_over_2m"), "%.4f"),
                                                                                      ", ".join("%s: %s" % (kk, fx(v["slope"], "%.3f")) for kk, v in w["per_channel_fit_dense"].items())))
        L += ["", "## The mass function at the dense outer root and the gradient length", "", "| T/r_s | R_t/(2g) | M_w dense | I integral | I collar | record W/(2g) at R_t | record Misner-Sharp at R_t | max |M_w - I| inside R_c (dense; applied) | I(end) - E | L_N min chain (R) | L_N min inside root | L_N at R_t | sites L_N < sigma |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            mf = r["mass_function"]
            q = mf.get("at_dense_outer") or {}
            gd = r["gradient_length"]["dense"]
            L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s; %s | %s | %s (%s) | %s | %s | %d |" % (r["T_over_rs"], fx(q.get("R_over_2g"), "%.4f"), fx(q.get("M_w_dense"), "%.4f"), fx(q.get("I"), "%.4f"), fx(q.get("I_collar"), "%.4f"), fx((mf.get("record") or {}).get("W_over_2g_at_Rt"), "%.4f"),
                                                                                           fx((mf.get("record") or {}).get("misner_sharp_at_Rt"), "%.4f"), fx(mf.get("max_abs_mismatch_inside_Rc_dense"), "%.3e"), fx(mf.get("max_abs_mismatch_inside_Rc_applied"), "%.3e"), fx(mf.get("I_chain_end_minus_E"), "%+.3e"),
                                                                                           fx(gd.get("min_chain"), "%.3f"), fx(gd.get("min_chain_R"), "%.0f"), fx(gd.get("min_inside_root"), "%.3f"), fx(gd["at_radii"].get("dense_outer"), "%.3f"), gd["sites_L_below_sigma"]))
        L += ["", "## The range-2 dust number current at the roots (per channel without multiplicity; the weighted row counts the declared multiplicities once; outward positive)", "",
              "| T/r_s | cut | R_cut | kappa | dust number (ops) | straddling sum | low |ap| <= pi/4; high; cross | dust F_M | matter (dev) number | matter F_M | straddling check; range |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for lab in CUT_ORDER:
                cu = r["currents"].get(lab)
                if not cu:
                    continue
                for kk in sorted(cu["channels"], key=int):
                    q = cu["channels"][kk]
                    if q["dust_number"] == 0.0 and q["dust_number_straddle"] == 0.0 and q["dev_number"] == 0.0:
                        continue
                    L.append("| %.3f | %s | %.4f | %s | %+.5e | %+.5e | %+.4e; %+.4e; %+.4e | %s | %+.5e | %s | %s; %s |" % (r["T_over_rs"], lab, cu["R_cut"], kk, q["dust_number"], q["dust_number_straddle"], q["dust_number_low"], q["dust_number_high"], q["dust_number_cross"],
                                                                                                                  fx(q["dust_FM"], "%+.4e"), q["dev_number"], fx(q["dev_FM"], "%+.4e"), fx(q.get("straddling_check"), "%.1e"), fx(q.get("range_sites_max"), "%d")))
                w = cu["weighted"]
                L.append("| %.3f | %s | %.4f | weighted | %+.5e | %+.5e | %+.4e; %+.4e; %+.4e | %+.4e | %+.5e | %+.4e | %.1e; %d |" % (r["T_over_rs"], lab, cu["R_cut"], w["dust_number"], w["dust_number_straddle"], w["dust_number_low"], w["dust_number_high"], w["dust_number_cross"], w["dust_FM"], w["dev_number"], w["dev_FM"], cu["straddling_check_max"], cu["range_sites_max"]))
        L += ["", "## The Killing spectrum of the dust (per channel; bins half-open at 0, +-m, +-b)", "", "| T/r_s | kappa | n_dust | eig vs stored e_k | %s | completeness | bound N | bound R_90 | bound omega/m | r_turn nominal; 2gE |" % " | ".join(BIN_LABELS), "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for c in r["channels"]:
                ka = c.get("killing")
                if not ka:
                    continue
                bd = ka["bound"]
                L.append("| %.3f | %d | %d | %s | %s | %.1e | %.4f | %s | %s | %s; %s |" % (r["T_over_rs"], c["kappa"], c["n_dust"], fx(ka["eig_vs_stored"], "%.1e"), " | ".join("%.4f" % q["N"] for q in ka["bins"]), ka["completeness_check"], bd["N"], fx(bd["R_90"], "%.3f"), fx(bd["mean_omega_over_m"], "%.5f"), fx(bd["r_turn_rs_nominal"], "%.3f"), fx(bd["r_turn_2gE"], "%.3f")))
            ki = r["killing"]
            L.append("| %.3f | weighted | - | - | %s | - | %.4f | %s | %s | %s; %s |" % (r["T_over_rs"], " | ".join("%.4f" % q["N"] for q in ki["bins_weighted"]), ki["bound_weighted"]["N"], fx(ki["bound_weighted"]["R_90"], "%.3f"), fx(ki["bound_weighted"]["mean_omega_over_m"], "%.5f"), fx(ki["bound_weighted"]["r_turn_rs_nominal"], "%.3f"), fx(ki["bound_weighted"]["r_turn_2gE"], "%.3f")))
        L += ["", "## The shells (weighted dust; the per-channel rows in the per-slice md): the share at |ap| > pi/4 and the Killing content", "", "| T/r_s | shell R range | n dust | share |ap| > pi/4 | <cos 2ak> | %s | bound N |" % " | ".join(BIN_LABELS), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for s_ in r["shells"]["weighted"]:
                L.append("| %.3f | %.0f to %.0f | %.4f | %s | %s | %s | %.4f |" % (r["T_over_rs"], s_["R_lo"], s_["R_hi"], s_["number"], fx(s_["frac_abs_p_above"], "%.4f"), fx(s_["cos2k"], "%+.4f"), " | ".join("%.4f" % q for q in s_["killing_bins_N"]), s_["bound_N"]))
    else:
        L += ["No slice of this history has been read yet (the per-record pass below runs from the records alone).", ""]
    L += ["", "## The root attribution between consecutive records holding a dense root (the records pass)", ""]
    if not recs.get("available"):
        L += ["Not available: %s." % recs.get("reason"), ""]
    else:
        L += [recs["definitions"] + ".", ""]
        s = recs.get("sums") or {}
        if s:
            L += ["Sums over %d steps (T/r_s %s to %s): Delta M at the root %.5f = fixed-radius change %.5f + sweep %.5f (integral form %.5f; largest identity defect %.1e); dE summed %.5f; the other ordering: sweep %.5f, fixed %.5f. Over the %d steps with a stored-state flux: fixed-radius change %.5f, dE %.5f, matter flux term %.5f, residual %.5f." % (
                s["steps"], fx(s["T_first_over_rs"], "%.3f"), fx(s["T_last_over_rs"], "%.3f"), s["dM_root"], s["dM_fixed_radius"], s["sweep_fields"], s["sweep_integral"], s["max_abs_identity_check"], s["dE"], s["alt_sweep_fields_at_old_clock"], s["alt_dM_fixed_radius_at_new_root"],
                s["steps_with_flux"], s["dM_fixed_where_flux_present"], s["dE_where_flux_present"], s["term_matter_where_present"], s["residual_matter_where_present"]), ""]
        L += ["| T/r_s from -> to | label (to) | R_t from -> to | R_t/(2gE) to | R_in/(2gE) to | dM root | dM fixed radius | sweep (fields; integral) | dE | flux term (matter; dust) | dust low; high; cross | residual (matter) | flux source |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        by_T = {round(q["T_over_rs"], 6): q for q in recs["rows"]}
        for q in recs["pairs"]:
            fl = q.get("flux") or {}
            rt = by_T.get(round(q["T_to_over_rs"], 6), {})
            L.append("| %.3f -> %.3f | %s | %.4f -> %.4f | %s | %s | %+.5f | %+.5f | %+.5f; %s | %+.5f | %s; %s | %s; %s; %s | %s | %s |" % (q["T_from_over_rs"], q["T_to_over_rs"], q["limit_label_to"].split(" (")[0], q["R_t_from"], q["R_t_to"], fx(q["R_t_to_over_2gE"], "%.4f"), fx(rt.get("R_in_over_2gE"), "%.4f"),
                                                                                                         q["dM_root"], q["dM_fixed_radius"], q["sweep_fields"], fx(q["sweep_integral"], "%+.5f"), q["dE"], fx(fl.get("term_matter"), "%+.5f"), fx(fl.get("term_dust"), "%+.5f"),
                                                                                                         fx(fl.get("dust_FM_low_out"), "%+.3e"), fx(fl.get("dust_FM_high_out"), "%+.3e"), fx(fl.get("dust_FM_cross_out"), "%+.3e"), fx(q.get("residual_matter"), "%+.5f"), fl.get("source", "no stored state at either end")))
        L += ["", "### Every record: the mass function and the gradient length on the dense fields", "", "| T/r_s | engine | label | gate max | R_t | R_t/(2gE) | R_in | R_in/(2gE) | M_w(R_t) | I(R_t) | I collar (R_t) | record W/(2g) | record Misner-Sharp | max |M_w - I| inside R_c | I(end) - E | L_N min inside R_t | L_N at R_t | sites L_N < sigma | slice |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for q in recs["rows"]:
            L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s |" % (q["T_over_rs"], q["engine"], q["limit_label"].split(" (")[0], fx(q["gate_sigma_abs_dv_max"], "%.4f"), fx(q["R_t"], "%.4f"), fx(q["R_t_over_2gE"], "%.4f"), fx(q["R_in"], "%.4f"), fx(q["R_in_over_2gE"], "%.4f"),
                                                                                                                 fx(q.get("M_w_at_Rt"), "%.4f"), fx(q.get("I_at_Rt"), "%.4f"), fx(q.get("I_collar_at_Rt"), "%.4f"), fx(q.get("record_W_over_2g_at_Rt"), "%.4f"), fx(q.get("record_misner_sharp_at_Rt"), "%.4f"), fx(q.get("mismatch_max_inside_Rc"), "%.3e"),
                                                                                                                 fx(q.get("I_chain_end_minus_E"), "%+.3e"), fx(q.get("L_N_min_inside_Rt"), "%.3f"), fx(q.get("L_N_at_Rt"), "%.3f"), q["sites_L_below_sigma"], "yes" if q["slice"] else ""))
    L += ["", "## Checks", ""]
    for r in rows:
        checks = [abs(c["killing"]["completeness_check"]) for c in r["channels"] if c.get("killing")] or [0.0]
        dchk = [c["killing"]["site_density_check"] for c in r["channels"] if c.get("killing")] or [0.0]
        eigs = [c["killing"]["eig_vs_stored"] for c in r["channels"] if c.get("killing") and c["killing"]["eig_vs_stored"] is not None]
        str_ = [cu["straddling_check_max"] for cu in r["currents"].values()] or [0.0]
        L.append("- T/r_s %.3f: Killing completeness %.1e, site-density %.1e, eig vs stored e_k %s; straddling against page_ops_v6 %.1e (range %s sites); shell number checks %.1e; Loewdin orthonormality %s (stored %s); %s; %.0f s." % (
            r["T_over_rs"], max(checks), max(dchk), fx(max(eigs), "%.1e") if eigs else "-", max(str_), max([cu["range_sites_max"] for cu in r["currents"].values()] or [0]), max([abs(s_["number_check"]) for s_ in r["shells"]["weighted"]] or [0.0]),
            fx(max((c["loewdin_orthonormality"] or 0.0) for c in r["channels"]), "%.1e"), fx(max(c["stored_orthonormality"] for c in r["channels"]), "%.1e"), "complex64 slice: floor %s" % r["floor"]["source_rel_range"] if r["complex64"] else "complex128", r["seconds"]))
    L += ["", "- every number above is printed by the script from the stored slices and the records; the reading is the lead's."]
    return L


FORMAT_LINE = ("SECONDROUTE_CORE_well_v1 (the slice format for the hreduce lane): per slice <out>/well/well_<history>_T<TTTT.TT>.json, .npz and .md (history in prod, s8, B, B2 or an --extra-history key; T the clock "
               "in r_s, e.g. T0016.00). json keys: the slices144 header (readout, title, code, history, history_label, tag, slice, T, T_over_rs, rs, nsites, chain_end_over_rs, dtype, complex64, floor, engine, "
               "record_found, record_converged, sigma_over_a, m, g, tau, note, setup), inventory_line, s8_line, s2_line, provenance, correlation_note, momentum_convention, root_finder, a, b, record {E, E_over_E0, "
               "two_gE, R_c, trapping_zeros, kappa_T, R_t, r_in, min_N_minus_absv, misner_sharp_at_Rt, W_over_2g_at_Rt, M1_balance, dE_dT, solver, prev_T_over_rs, next_T_over_rs}, fields_source, roots {dense_outer, "
               "dense_inner, applied_outer, applied_inner, applied_zeros, prev_dense_outer, next_dense_outer, *_over_2gE, inbox_T16 at 16 of prod}, gate, limit_label, dust {total_weighted, per_channel_total, R_50, R_90}, "
               "mass_function {max_abs_mismatch_inside_Rc_dense/applied, I_chain_end, I_chain_end_minus_E, at_<radius> {R, M_w_dense, M_w_applied, I, I_collar, I_applied_w, R_over_2g}, record}, gradient_length {dense, "
               "applied: min_chain, min_chain_R, min_inside_root, sites_L_below_sigma, at_radii}, profile_at_radii {<radius>: N_dense, v_dense, w_dense, N_applied, v_applied, w_applied, n_shell, n_density, "
               "rho_density, rho_over_n, J_density, N_enclosed, M_w_dense, I_integral, L_N_dense, L_N_applied}, core_fit {primary_window, windows {R50, R90, inner_root, Rc: epsilon_c_total, epsilon_c_dust, fit_dense "
               "{slope, slope_se, r2, n}, fit_applied, exponent_eps_minus_m_over_2m, exponent_eps_over_2m, exponent_dust_*, per_channel_fit_dense}, rho_over_n_innermost}, momentum_grid {M, k_threshold, zone}, "
               "shells {width_sites, count, weighted: [{shell, R_lo, R_hi, sites, number, number_check, frac_abs_p_above, cos2k, sin2k, mean_abs_p, killing_bins_N[6], bound_N}]}, killing {m, b, edges, labels, "
               "bins_weighted [{label, N, E}], bound_weighted {N, mean_omega_over_m, R_50, R_90, r_turn_rs_nominal, r_turn_2gE}}, fountain {R_50_all, R_90_all, R_90_bound, N_bound_weighted, N_all_weighted, "
               "mean_omega_over_m_bound, r_turn_rs_nominal, r_turn_2gE, two_gE}, channels [{kappa, weight, n_sea, n_dust, stored_orthonormality, loewdin_orthonormality, momentum_split, killing {eig_vs_stored, "
               "bins [{label, lo, hi, N, E, levels}], completeness_check, site_density_check, bound {...}, orbitals [{column, mean_omega_over_m, spread, R_50, R_90, bins_N, bound_weight, bound_mean_omega_over_m, "
               "bound_R_90, r_turn_rs_nominal, r_turn_2gE}]}, shells [...per channel...], dust_whole_chain, currents {<cut>: {R_cut, dust_number, dust_number_straddle, straddling_check, straddling_nnz, range_sites_max, "
               "dust_energy, dust_FM, dust_number_low/high/cross, dust_energy_low/high/cross, dust_FM_low/high/cross, C_number, R2_number, dev_number, C_energy, R2_energy, dev_energy, dev_FM, pairs_number, pairs_FM}}}], "
               "currents {<cut in dense_outer, dense_inner, applied_outer, applied_inner, prev_dense_outer, next_dense_outer, inbox_applied_outer, inbox_dense_outer>: {R_cut, source, N_applied_at_cut, channels {kappa: ...}, "
               "weighted {the same keys summed with 16 kappa}, straddling_check_max, range_sites_max}}, seconds. npz arrays: R, N_applied, v_applied, N_dense_sites, v_dense_sites, w_dense, w_applied, eps_tot, J_tot, "
               "eps_pairs, J_pairs, dust_number_site, dust_enclosed, k<kappa>_dust_number_site, k<kappa>_dust_enclosed, n_density, rho_density, rho_dust_density, J_density, rho_over_n, rho_dust_over_n, M_w_dense, "
               "M_w_applied, I_integral_dense, I_integral_applied, I_collar_dense, mismatch_dense, mismatch_applied, source_dense, L_N_dense, L_N_applied (1e300 where N' = 0), local_slope_lnN_lnNenc_dense, k (the "
               "momentum grid, fftfreq order), p_reduced, shell_edges_sites, k<kappa>_shell<ss>_nk, shell<ss>_nk_weighted, k<kappa>_dust_whole_chain_nk, k<kappa>_omega, k<kappa>_n_dust_killing, "
               "k<kappa>_killing_bin_density (6 x n), k<kappa>_bound_density, killing_bin_density_weighted, bound_density_weighted, k<kappa>_dust_eps_h0_site. Per history: records_<history>.json {rows per record, pairs "
               "per attribution step, sums} and .npz (rec_*, pair_* arrays); CORE_well_v1_<history>.md; the summary CORE_well_v1.md and CORE_well_v1_all.json.")


def compose(a, hists):
    rows_all = IO.load_rows(a, READOUT, hists)
    rows = [r for r in rows_all if r.get("code") == CODE]
    INV = extended_inventory(a)
    log = IO.Logger(os.path.join(a.out, "core_well_v1.log"))
    d = os.path.join(a.out, READOUT)
    os.makedirs(d, exist_ok=True)
    fx = IO.fx
    S = ["# %s: the core's well on the stored slices (%s)" % (TITLE, CODE), "", "The final plan's item 1 (inbox #162). " + PROVENANCE + ". " + CORRELATION_NOTE + ".", "", "## The stored states (1 line per history)", "", "- " + INVENTORY_LINE, "- " + S8_LINE, "- " + S2_LINE, ""]
    summary = {"code": CODE, "title": TITLE, "inventory_line": INVENTORY_LINE, "s8_line": S8_LINE, "s2_line": S2_LINE, "histories": {}, "rows": rows, "format": FORMAT_LINE}
    for hk in hists:
        H = INV.get(hk)
        if H is None:
            continue
        rs_ = [r for r in rows if r["history"] == hk]
        ctx = make_context(H, a, log)
        scheme = scheme_line(ctx, H, a)
        recs, rarr = records_pass(ctx, H, a, rs_, log) if a.records_pass == "on" else ({"available": False, "reason": "--records-pass off"}, {})
        with open(os.path.join(d, "records_%s.json" % hk), "w") as f:
            json.dump(dict(recs, history=hk, scheme=scheme, code=CODE), f, default=IO.jdefault, indent=1)
        if rarr:
            np.savez(os.path.join(d, "records_%s.npz" % hk), **rarr)
        with open(os.path.join(d, "%s_%s.md" % (TITLE, hk)), "w") as f:
            f.write("\n".join(history_md(hk, H, rs_, recs, scheme, a)) + "\n")
        summary["histories"][hk] = {"label": H["label"], "engine": H["engine"], "scheme": scheme, "slices_read": [r["T_over_rs"] for r in rs_], "records_pass_sums": recs.get("sums"), "records_pass_available": recs.get("available"), "records_pass_reason": recs.get("reason")}
        S += ["## %s: %s" % (hk, H["label"]), "", "Engine: %s." % H["engine"], "", scheme, "", "Slices read: %s." % ([r["T_over_rs"] for r in rs_] or "none"), ""]
        if rs_:
            S += ["| T/r_s | dtype | label | dense roots | applied roots | R_t/(2gE) | R_50 | R_90 all | R_90 bound | omega/m bound | r_turn nominal; 2gE | eps_c (primary window) | slope ln N vs ln N_< | (eps_c - m)/(2m); eps_c/(2m) | weighted dust current at the dense outer root (ops; straddling) | matter F_M there |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs_:
                fo, ro, cf = r["fountain"], r["roots"], r["core_fit"]
                w = cf["windows"].get(cf["primary_window"]) if cf["primary_window"] else None
                co = (r["currents"].get("dense_outer") or {}).get("weighted", {})
                S.append("| %.3f | %s | %s | %s; %s | %s; %s | %s | %s | %s | %s | %s | %s; %s | %s | %s | %s; %s | %s; %s | %s |" % (
                    r["T_over_rs"], r["dtype"], r["limit_label"].split(" (")[0], fx(ro["dense_outer"], "%.4f"), fx(ro["dense_inner"], "%.4f"), fx(ro["applied_outer"], "%.4f"), fx(ro["applied_inner"], "%.4f"), fx(ro["dense_outer_over_2gE"], "%.4f"),
                    fx(fo["R_50_all"], "%.3f"), fx(fo["R_90_all"], "%.3f"), fx(fo["R_90_bound"], "%.3f"), fx(fo["mean_omega_over_m_bound"], "%.5f"), fx(fo["r_turn_rs_nominal"], "%.3f"), fx(fo["r_turn_2gE"], "%.3f"),
                    fx((w or {}).get("epsilon_c_total"), "%.5f"), fx(((w or {}).get("fit_dense") or {}).get("slope"), "%.4f"), fx((w or {}).get("exponent_eps_minus_m_over_2m"), "%.4f"), fx((w or {}).get("exponent_eps_over_2m"), "%.4f"),
                    fx(co.get("dust_number"), "%+.5e"), fx(co.get("dust_number_straddle"), "%+.5e"), fx(co.get("dev_FM"), "%+.5e")))
            S.append("")
            r16 = [r for r in rs_ if abs(r["T_over_rs"] - 16.0) < 1e-6]
            if r16 and r16[0]["roots"].get("inbox_T16"):
                r = r16[0]
                ib = r["roots"]["inbox_T16"]
                S += ["At T/r_s 16 (the one-slice computation): the plan's quoted roots (inbox) applied outer %.4f a and dense %.4f a; computed applied outer %s (difference %s), dense outer %s (difference %s). The weighted dust number current (multiplicities counted once): through the applied outer root %s (straddling sum %s), through the dense root %s (%s); through the quoted radii %s and %s; per channel in CORE_well_v1_%s.md." % (
                    ib["applied_outer"], ib["dense_outer"], fx(r["roots"]["applied_outer"], "%.4f"), fx(ib["computed_minus_inbox_applied"], "%+.2e"), fx(r["roots"]["dense_outer"], "%.4f"), fx(ib["computed_minus_inbox_dense"], "%+.2e"),
                    fx((r["currents"].get("applied_outer") or {}).get("weighted", {}).get("dust_number"), "%+.5e"), fx((r["currents"].get("applied_outer") or {}).get("weighted", {}).get("dust_number_straddle"), "%+.5e"),
                    fx((r["currents"].get("dense_outer") or {}).get("weighted", {}).get("dust_number"), "%+.5e"), fx((r["currents"].get("dense_outer") or {}).get("weighted", {}).get("dust_number_straddle"), "%+.5e"),
                    fx((r["currents"].get("inbox_applied_outer") or {}).get("weighted", {}).get("dust_number"), "%+.5e"), fx((r["currents"].get("inbox_dense_outer") or {}).get("weighted", {}).get("dust_number"), "%+.5e"), hk), ""]
        s = recs.get("sums") or {}
        if s:
            S += ["Root attribution over %d steps (T/r_s %s to %s): Delta M at the root %.5f = fixed-radius change %.5f + sweep %.5f (integral form %.5f); dE summed %.5f; over the %d steps with a stored-state flux: fixed-radius change %.5f, dE %.5f, matter flux term %.5f, residual %.5f (the per-step table in CORE_well_v1_%s.md)." % (
                s["steps"], fx(s["T_first_over_rs"], "%.3f"), fx(s["T_last_over_rs"], "%.3f"), s["dM_root"], s["dM_fixed_radius"], s["sweep_fields"], s["sweep_integral"], s["dE"], s["steps_with_flux"], s["dM_fixed_where_flux_present"], s["dE_where_flux_present"], s["term_matter_where_present"], s["residual_matter_where_present"], hk), ""]
        elif not recs.get("available"):
            S += ["Root attribution: not available (%s)." % recs.get("reason"), ""]
    S += ["## The slice format for the hreduce lane", "", FORMAT_LINE, "", "- every number above is printed by the script from the stored slices and the records; the reading is the lead's."]
    with open(os.path.join(d, "%s.md" % TITLE), "w") as f:
        f.write("\n".join(S) + "\n")
    with open(os.path.join(d, "%s_all.json" % TITLE), "w") as f:
        json.dump(summary, f, default=IO.jdefault, indent=1)
    log("%s: the summary and %d history md(s) composed in %s" % (TITLE, len(summary["histories"]), d))


# ----------------------------------------------------------------------------------------------- the inventory and the run
def parse_extra_history(spec):
    kv = {}
    for item in str(spec).split(";"):
        if "=" in item:
            k_, v_ = item.split("=", 1)
            kv[k_.strip()] = v_.strip()
    key, tag = kv["key"], kv["tag"]
    dirs = [x for x in kv.get("dirs", "").split("|") if x]
    H = {"key": key, "label": kv.get("label", "extra history %s (tag %s)" % (key, tag)), "tag": tag, "nsites": int(kv.get("nsites", 2048)), "rs": float(kv.get("rs", 64.0)),
         "sigma_over_a": float(kv["sigma"]) if kv.get("sigma") else None, "engine": kv.get("engine", "not stated (an extra history given on the command line)"), "fields_module": kv.get("module", "not stated"),
         "records_json": kv.get("records") or None, "fields_npz": kv.get("fields") or None, "data_npz": None, "slices": [], "extra": True}
    seen = {}
    for d_ in dirs:
        for p in sorted(glob.glob(os.path.join(d_, "orbitals_%s*_T*.npz" % tag))):
            mt = re.search(r"_T(\d+\.\d+)\.npz$", os.path.basename(p))
            if mt:
                seen.setdefault(round(float(mt.group(1)), 6), []).append(p)
    want = None if kv.get("clocks", "all") == "all" else [float(x) for x in kv["clocks"].split(",") if x]
    for t in sorted(seen):
        if want is not None and not any(abs(t - w) < 1e-6 for w in want):
            continue
        paths = seen[t]
        H["slices"].append({"history": key, "history_label": H["label"], "tag": tag, "T_over_rs": t, "T": t * H["rs"], "dtype": kv.get("dtype", "read from the file"), "nsites": H["nsites"],
                            "expected_size": os.path.getsize(paths[0]), "box_paths": paths, "hf_path": None, "sha256_prefix": None, "sha_source": None, "records_json": H["records_json"], "fields_npz": H["fields_npz"],
                            "engine": H["engine"], "fields_module": H["fields_module"], "name": os.path.basename(paths[0]),
                            "note": "an extra history: no inventory size or hash exists for this file (its size is read from the file); the slice format is checked at read time"})
    return H


def extended_inventory(a):
    INV = IO.inventory(a.box_D, a.box_E)
    for spec in (getattr(a, "extra_history", None) or []):
        H = parse_extra_history(spec)
        INV[H["key"]] = H
    return INV


class NoRecordsContext:
    """a history without a records json: the slice readouts run; every record quantity prints 'no records json'."""

    def __init__(self, H, a, log):
        self.H = H
        self.a = a
        self.log = log
        self.m = float(a.m)
        self.g = float(a.g)
        self.ell = float(H.get("sigma_over_a") or 0.0)
        self.rs = float(H.get("rs", 64.0))
        self.nsites = int(H.get("nsites", 2048))
        self.tau = 8.0
        self.E0 = 0.0

    def records(self, entry):
        return {"records": [], "args": {}}

    def record_at(self, entry, T):
        return {}

    def record_prev_next(self, entry, T):
        return None, None

    def fields_file(self, entry):
        return None, None

    def fields_key(self, entry, key):
        return None

    def fields_index(self, entry, T):
        return None

    def engine_label(self, entry, rec):
        return "%s; no records json given for this history" % entry.get("engine")


def make_context(H, a, log):
    if H.get("records_json") and os.path.isfile(H["records_json"]):
        try:
            return IO.HistoryContext(H, a, log)
        except Exception as ex:  # noqa: BLE001
            log("the records json %s could not be read as a slices144 history (%s: %s): running without records" % (H["records_json"], type(ex).__name__, ex))
    else:
        log("history %s: no records json (%s): running without records" % (H["key"], H.get("records_json")))
    return NoRecordsContext(H, a, log)


def run(a):
    log = IO.Logger(os.path.join(a.out, "core_well_v1.log"))
    IO.setup(a.engine_code, a.page_code, a.threads, log)
    log("%s: setup %s; root finder: %s" % (CODE, json.dumps(IO.SETUP, default=str), ROOT_FINDER))
    log(INVENTORY_LINE)
    INV = extended_inventory(a)
    hists = list(INV) if a.history.strip().lower() == "all" else [h.strip() for h in a.history.split(",") if h.strip()]
    if not a.compose_only:
        for hk in hists:
            H = INV.get(hk)
            if H is None:
                log("unknown history %s (known: %s)" % (hk, list(INV)))
                continue
            ctx = make_context(H, a, log)
            entries = IO.select_entries(H, a.clocks)
            log("history %s (%s): %d slices selected" % (hk, H["label"], len(entries)))
            for e in entries:
                if not a.force and is_done(e, a):
                    log("T/r_s %.3f: the well json is present (--force recomputes)" % e["T_over_rs"])
                    continue
                IO.wait_for_load(a.max_load, log)
                try:
                    with IO.fetch_slice(e, a, log) as path:
                        rd = IO.SliceReader(path, e, log)
                        log("slice %s: T/r_s %.3f, %d sites, %s, kappas %s, dust columns %s" % (os.path.basename(path), rd.T_over_rs, rd.nsites, rd.dtype_str, rd.kappas, rd.n_dust))
                        process_slice(rd, ctx, e, a)
                        rd.close()
                        del rd
                except FileNotFoundError as ex:
                    log("MISSING slice at T/r_s %.3f: %s" % (e["T_over_rs"], ex))
                    if a.stop_on_missing:
                        raise
                except KeyError as ex:
                    log("FORMAT: the slice at T/r_s %.3f of %s lacks the key %s (its member list: %s): skipped" % (e["T_over_rs"], hk, ex, sorted(IO.zip_member_offsets(IO.locate(e)).keys()) if IO.locate(e) else "unreadable"))
    compose(a, hists)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    IO.add_common_arguments(ap)
    add_arguments(ap)
    a = ap.parse_args(argv)
    a.readouts = READOUT
    IO.READOUT_MODULES[READOUT] = MODULE
    run(a)


if __name__ == "__main__":
    main()
