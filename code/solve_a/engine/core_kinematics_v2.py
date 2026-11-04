#!/usr/bin/env python
"""core_kinematics_v2.py (reviewer A for the solve builder, the busi lane). Inbox busi #144 item (2), word for word:
"CORE_kinematics_v1: on the 10 stored slices of the r_s/a 64, τ 8, σ/a 4 history and the σ/a 8 member's slices, per engine channel without
multiplicity on any printed correlation: the radial momentum distribution of the occupied interior orbitals (eigenvectors of Φ†Π_XΦ with
eigenvalue at least 0.99 inside the outer root, the one-object ruling's S4 A.5) and of the whole occupied set, the fraction at |ak| > π/4,
⟨cos 2ak⟩, and the outward number current at the outer root and at the radius holding 50 percent of the dust; the same window stated. This
is requirements §5 I1's "the core's drift against the flow". CPU. Same folder."

Changes against core_kinematics_v1.py (the self-test's one failure on the box: "a pure plane wave at ak = 2.4544 sits below pi/4 with
cos 2ak = 0.1951": frac 0.9961, <cos 2ak> 0.1853). v1 read |ak| > pi/4 in the full zone [-pi, pi); the chain's symbol is pi-periodic in ak,
so the full zone double-counts the staggered pair's zone and misreads the second valley's long-wavelength content (ak near pi) as
lattice-scale content. v2 reads every momentum statistic in the reduced zone p = k mod pi (CONVENTION below, printed at the head of the md
and stored in every json): the fraction at |ap| > pi/4 (= the weight with cos 2ak < 0, the counter-advected content), <cos 2ak> and
<sin 2ak> (the same in either zone), <|ap|>, the share at p > 0 minus p < 0; beside them, for the record, the full-zone fraction at
|ak| > pi/4 (v1's number) and the share of the valley at ak = pi. The 0.1853 against 0.1951 is the finite extent of the 40-site test wave:
<cos 2ak> = (1 - 2/L) cos 2q exactly (the real-space identity), 38/40 x 0.1951 = 0.1853: the window's resolution, not a convention; every
interior orbital now prints its extent L (inverse participation ratio in sites) and 2/L beside its moments. Everything else keeps v1's
arithmetic: the window, the interior orbitals (page_d13_readout_v1b.source_modes), the window-restricted sets, the currents, the checks.
The json keys of the moments are renamed (frac_abs_p_above, frac_counter_advected, frac_abs_k_above_full_zone, valley_pi_share,
mean_abs_p, p_positive_minus_negative; v1's mean_abs_k and outward_minus_inward are gone). The module registers itself as the driver's core
readout when run as a script (slices144_io_v1.READOUT_MODULES maps core to v1 and is not edited); is_done and compose accept v2 rows only
(a v1 json at the same path is recomputed and set aside in the tables' count).

Per slice and per engine channel kappa (no multiplicity on any printed correlation; the 16 kappa sums appear only on the currents, labelled):
- the window: the sites inside the record's dense outer root R_t (the largest stored trapping zero at the slice's clock; --window Rc or 2gE
  where the record holds no root, the fallback named), the same window for every quantity of the row and stated with its site count.
- the interior orbitals: page_d13_readout_v1b.source_modes on the Loewdin-orthonormalized occupied columns Phi: the eigenvectors of
  Phi^dagger Pi_W Phi (through C restricted to the window, the same nonzero spectrum) with eigenvalue at least --interior-weight (0.99),
  as unit modes s = Phi u in ran C; their weight inside (equal to the eigenvalue: printed as a check), occupation <s|C|s>, Killing energy
  and spread (h_m at the applied fields); for each the radial momentum distribution n(k) = sum_a |sum_x e^{-ikx} s_a(x)|^2 over the chain
  (the 2 spinor components summed; the grid k_j = 2 pi j/M, M = max(--k-grid, the next power of 2 above n + 2), exact DTFT samples of the
  zero-padded orbital), read on the fold p = k mod pi: the fraction at |ap| > --k-threshold-pi pi (1/4: where cos 2ak, the two-step
  current's group-velocity factor, changes sign), <cos 2ak> (checked against its real-space form Re sum_x s^dagger(x) s(x + 2), exact for
  M >= n + 2), <sin 2ak>, <|ap|>, the share at p > 0 minus p < 0, the extent L and 2/L; the full-zone fraction and the valley share beside.
- the whole occupied set in the window: the window-restricted covariance C_W = Pi_W C Pi_W (sea and dust), n_W(k) = sum_a e_k^dagger
  C_W^aa e_k on the same grid (e_k the plane wave on the window's sites; (1/M) sum_k n_W(k) = Tr C_W, the number in the window), the same
  moments; the sea alone, the stored (tagged) dust alone, the continued reference R2 R2^dagger alone, and the deviation C - R2 R2^dagger
  (its total can be small or negative: its moments print only with a positive total); the tagged dust over the whole chain.
- the outward number current (page_ops_v6.number_current_csr: Tr[X i[h_m, Pi_{R > R_cut}] X^dagger], outward positive) at the outer root
  R_t and at R_50, the radius holding 50 percent of the weighted dust number, for the tagged dust, the occupied set C, the reference R2 and
  the deviation C - R2, per channel; the dust's Killing-energy current beside; the 16 kappa sums labelled.
Outputs per slice: core_<history>_T<clock>.json and .npz (every n(k) array, the k grid and its fold p_reduced, the interior orbitals'
columns with --store-orbitals); compose writes CORE_kinematics_v2_<history>.md and CORE_kinematics_v2_all.json. CPU only; no plotting;
no mutual information; no dates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slices144_io_v1 as IO  # noqa: E402
import interior_continuation_v1 as IC  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

READOUT = "core"
TITLE = "CORE_kinematics_v2"
CODE = "core_kinematics_v2.py"
MODULE = "core_kinematics_v2"
ZONE = "reduced: p = k mod pi in [-pi/2, pi/2) (p = k - pi sign(k) for |ak| > pi/2; both edge points |ak| = pi/2 to p = -pi/2), folded on the grid index"
MOMENT_KEYS = ("frac_abs_p_above", "frac_counter_advected", "frac_abs_k_above_full_zone", "valley_pi_share", "cos2k", "sin2k", "mean_abs_p", "p_positive_minus_negative")
MEAN_KEYS = ("frac_abs_p_above", "frac_abs_k_above_full_zone", "valley_pi_share", "cos2k", "sin2k", "mean_abs_p", "resolution_2_over_L")

CONVENTION = (
    "Momentum convention: the reduced zone. The chain operator's symbol is pi-periodic in ak: the sine bond -(i/2) sigma_x from site n to n + 1 "
    "has the symbol sin ak sigma_x (omega^2 = m^2 + sin^2 ak + kappa^2/R^2), the two-step current Q = -(i/4) 1 from n to n + 2 has (1/2) sin 2ak, "
    "and the on-site m sigma_z + (kappa/R) sigma_y does not depend on k, so h[N, v] = (1/2){N, h_0} + (1/2){v, Q} cannot tell a plane wave at ak "
    "from the same wave at ak +- pi (the wave carrying the staggered sign (-1)^x): the two valleys ak = 0 and ak = pi are the staggered pair's two "
    "continuum channels (page_ops_v6: 'the chain's own two valleys are the continuum channels +kappa and -kappa'), each long-wavelength about its "
    "own centre with the two-step current at unit slope there, and the lattice-scale content is the content at |ap| -> pi/2, midway between the "
    "valleys, where sin ak saturates at the band's edge (|omega| maximal) and the two-step current's symbol returns to zero. Every momentum "
    "statistic of this readout is therefore read in the reduced zone p = k mod pi in [-pi/2, pi/2) (p = k - pi sign(k) for |ak| > pi/2; the "
    "zone's edge points |ak| = pi/2 both go to p = -pi/2; the fold is exact on the grid): the fraction at |ap| > pi/4 is the weight where "
    "cos 2ak < 0, the content whose advection by the flow term (group velocity v cos 2ak) is reversed against the flow, the lattice-scale, "
    "counter-advected content, and the rest is the long-wavelength content of either valley carried with the flow (at the continuum's unit slope "
    "near either valley centre); <cos 2ak> and <sin 2ak> are the same in either zone; <|ap|> and the share at p > 0 minus the share at p < 0 (the "
    "sign of sin 2ak, the direction of the positive branch's group velocity) use the folded p. The full-zone fraction at |ak| > pi/4 (v1's number) "
    "and the share of the valley at ak = pi (|ak| > pi/2) are printed beside for the record: the full-zone fraction counts the second valley's "
    "long-wavelength content as lattice-scale and contradicts <cos 2ak> (a wave at ak = 2.4544 has cos 2ak = +0.195, carried with the flow, yet "
    "|ak| > pi/4); it is not the readout's number. <cos 2ak> is exact by the real-space identity Re sum_x s^dagger(x) s(x + 2) and carries the "
    "orbital's finite extent, not a zone convention: a plane wave of uniform amplitude on L sites has <cos 2ak> = (1 - 2/L) cos 2q exactly (the "
    "two-step correlator has L - 2 terms over L), so the self-test's 0.1853 against cos 2q = 0.1951 is this factor at L = 40 (38/40 = 0.95); the "
    "same finite extent broadens the line to a width 2 pi/L with 1/(ak)^2 tails, so a fraction read at a threshold a distance Delta from the "
    "line leaks of order 1/(L Delta) across it (0.06 for the 40-site test wave 0.098 from pi/4). Each interior orbital prints its extent L (the "
    "inverse participation ratio of its site density, in sites) and 2/L beside its moments; the window sets' extent is the window's site count.")


def add_arguments(ap):
    g = ap.add_argument_group("core_kinematics_v2")
    g.add_argument("--interior-weight", dest="interior_weight", type=float, default=0.99, help="the interior orbitals' weight inside the window (the one-object ruling's S4 A.5)")
    g.add_argument("--interior-max", dest="interior_max", type=int, default=256, help="at most this many interior orbitals per channel")
    g.add_argument("--k-grid", dest="k_grid", type=int, default=2048, help="the momentum grid (raised to the next power of 2 above n + 2; divisible by 8, so the zone's quarter points are grid points)")
    g.add_argument("--k-threshold-pi", dest="k_threshold_pi", type=float, default=0.25, help="the fraction at |ap| above this times pi in the reduced zone (1/4: where cos 2ak changes sign)")
    g.add_argument("--window", default="root", choices=["root", "Rc", "2gE"], help="the window: inside the record's outer root (Rc or 2gE where the record holds none)")
    g.add_argument("--window-fallback", dest="window_fallback", default="Rc", choices=["Rc", "2gE"])
    g.add_argument("--store-orbitals", dest="store_orbitals", default="on", choices=["on", "off"])


def is_done(entry, a):
    """a json at the slice's path written by this version (a v1 json is recomputed)."""
    pj = IO.out_paths(a, READOUT, entry)[0]
    if not os.path.exists(pj):
        return False
    try:
        with open(pj) as f:
            return json.load(f).get("code") == CODE
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------------------------------------- the momenta
def k_grid(nsites, k_grid_min):
    """M, the smallest power of 2 at or above max(k_grid_min, nsites + 2) (divisible by 8: +-pi/4 and +-pi/2 are grid points), and the grid
    k_j = 2 pi j/M in fftfreq order (k in [-pi, pi))."""
    M = 1
    while M < max(int(k_grid_min), nsites + 2):
        M *= 2
    return M, 2.0 * math.pi * np.fft.fftfreq(M)


def fold_reduced(k, M):
    """the reduced-zone momentum p = k mod pi in [-pi/2, pi/2) of every grid point k_j = 2 pi j/M, folded on the integer index
    j_p = ((j + M/4) mod M/2) - M/4 (exact: k and k +- pi land on the same p; the threshold points +-pi/4 stay on the grid as the same doubles as
    0.25 pi; both k = +-pi/2 go to p = -pi/2; k = -pi goes to p = 0)."""
    j = np.rint(np.asarray(k, float) * M / (2.0 * math.pi)).astype(np.int64)
    jp = np.mod(j + M // 4, M // 2) - M // 4
    return (2.0 * math.pi / M) * jp


def orbital_momentum(s, M):
    """n(k) = sum_a |FFT_M(s_a)|^2 for one chain vector s (2n,), the components interleaved; exact DTFT samples for M >= n."""
    s = np.asarray(s, np.complex128)
    nk = np.zeros(M)
    for a_ in (0, 1):
        nk += np.abs(np.fft.fft(s[a_::2], n=M)) ** 2
    return nk


def window_momentum(Xw, f, xw, k):
    """n_W(k) = sum_a e_k^dagger C_W^aa e_k for the window-restricted columns Xw (2 n_W x ncol; rows interleaved), occupations f, the window's
    radii xw and the plane waves e_k = exp(-i k x)."""
    Xw = np.asarray(Xw, np.complex128)
    E = np.exp(-1j * np.outer(k, np.asarray(xw, float)))
    nk = np.zeros(len(k))
    for a_ in (0, 1):
        Xa = Xw[a_::2]
        C = (Xa * np.asarray(f, float)[None, :]) @ Xa.conj().T
        nk += np.real(np.sum((E @ C) * np.conj(E), axis=1))
    return nk


def window_cos2k_real(Xw, f, xw):
    """Re sum_a sum_{x, x+2 in W} C^aa_{x, x+2}: the real-space form of (1/M) sum_k cos(2k) n_W(k)."""
    Xw = np.asarray(Xw, np.complex128)
    xw = np.asarray(xw, float)
    pos = {float(x): i for i, x in enumerate(xw)}
    tot = 0.0
    for a_ in (0, 1):
        Xa = Xw[a_::2] * np.sqrt(np.clip(np.asarray(f, float), 0.0, None))[None, :]
        for i, x in enumerate(xw):
            j = pos.get(float(x) + 2.0)
            if j is not None:
                tot += float(np.real(np.vdot(Xa[i], Xa[j])))
    return tot


def extent_ipr(s):
    """the extent of a chain vector in sites: the inverse participation ratio (sum_x rho_x)^2/sum_x rho_x^2 of its site density (the 2
    components summed); a plane wave of uniform amplitude on L sites has extent L."""
    s = np.asarray(s, np.complex128)
    rho = np.abs(s[0::2]) ** 2 + np.abs(s[1::2]) ** 2
    q = float(np.sum(rho ** 2))
    return float(np.sum(rho) ** 2 / q) if q > 0 else None


def moments(nk, k, thr, M):
    """the moments of a momentum distribution n(k_j) on the grid k_j = 2 pi j/M, read in the reduced zone p_j = k_j mod pi (CONVENTION):
    number = (1/M) sum_j n; frac_abs_p_above = the weight at |ap| > thr over the total (the readout's fraction); frac_counter_advected = the
    weight at cos 2ak < 0 (the same grid set at thr = pi/4, exactly); frac_abs_k_above_full_zone = the weight at |ak| > thr in the full zone
    (v1's number, for the record); valley_pi_share = the weight at |ak| > pi/2 (the valley at ak = pi); cos2k, sin2k = <cos 2ak>, <sin 2ak>
    (the same in either zone); mean_abs_p = <|ap|>; p_positive_minus_negative = the share at 0 < p < pi/2 minus the share at -pi/2 < p < 0
    (the sign of sin 2ak; the valley centres and the zone's edge, where sin 2ak = 0, count for neither). A non-positive total (the deviation
    C - R2 R2^dagger can have one) returns None for every moment."""
    nk = np.asarray(nk, float)
    k = np.asarray(k, float)
    tot = float(np.sum(nk))
    out = {"number": tot / M, "total_positive": bool(tot > 0), "zone": ZONE}
    if tot <= 0:
        out.update({key: None for key in MOMENT_KEYS})
        return out
    p = fold_reduced(k, M)
    c2 = np.cos(2.0 * p)
    half = 0.5 * math.pi
    out.update({"frac_abs_p_above": float(np.sum(nk[np.abs(p) > thr]) / tot),
                "frac_counter_advected": float(np.sum(nk[c2 < 0.0]) / tot),
                "frac_abs_k_above_full_zone": float(np.sum(nk[np.abs(k) > thr]) / tot),
                "valley_pi_share": float(np.sum(nk[np.abs(k) > half]) / tot),
                "cos2k": float(np.sum(c2 * nk) / tot), "sin2k": float(np.sum(np.sin(2.0 * p) * nk) / tot),
                "mean_abs_p": float(np.sum(np.abs(p) * nk) / tot),
                "p_positive_minus_negative": float((np.sum(nk[(p > 0.0) & (p < half)]) - np.sum(nk[(p < 0.0) & (p > -half)])) / tot)})
    return out


def occ_mean(orbs, key):
    """the occupation-weighted mean of one moment over the interior orbitals (None where any orbital lacks it)."""
    vals = [(o["occ_C"], o.get(key)) for o in orbs]
    if not vals or any(v is None for _, v in vals):
        return None
    w = sum(c for c, _ in vals)
    return float(sum(c * v for c, v in vals) / w) if w > 0 else None


def window_radius(rec, a, g):
    """the window's outer radius and its label."""
    zeros = rec.get("trapping_zeros") or []
    if a.window == "root" and zeros:
        return float(max(zeros)), "inside the record's dense outer root R_t = %.4f" % max(zeros)
    fb = a.window if a.window != "root" else a.window_fallback
    if fb == "Rc" and rec.get("R_c") is not None:
        return float(rec["R_c"]), "inside the record's R_c = %.4f (%s)" % (rec["R_c"], "no root at this clock" if a.window == "root" else "--window Rc")
    r = 2.0 * g * float(rec["E"])
    return r, "inside 2gE = %.4f (%s)" % (r, "no root and no R_c" if a.window == "root" else "--window 2gE")


# ------------------------------------------------------------------------------------------------------------ the slice
def process_slice(rd, ctx, entry, a):
    if IO.D13 is None:
        raise RuntimeError("core_kinematics_v2 needs page_d13_readout_v1b (source_modes): not importable (%s)" % IO.SETUP.get("D13"))
    t0 = time.time()
    log = ctx.log
    torch = IO.torch
    dev = torch.device("cpu")
    pj, pz = IO.out_paths(a, READOUT, entry)
    rec = ctx.record_at(entry, rd.T)
    m, g, ell, rs = ctx.m, ctx.g, ctx.ell, rd.rs
    R, N_app, v_app = rd.R, rd.N_applied, rd.v_applied
    row = IO.slice_header(rd, ctx, entry, rec, READOUT, TITLE, CODE)
    row["correlation_note"] = "per engine channel without multiplicity on any printed correlation; the 16 kappa sums appear only on the currents, labelled"
    row["convention"] = CONVENTION
    dR = np.diff(R)
    spacing_dev = float(np.max(np.abs(dR - 1.0))) if dR.size else 0.0
    row["lattice_spacing"] = {"max_abs_dR_minus_1": spacing_dev, "unit_uniform": bool(spacing_dev < 1e-9),
                              "note": "the lattice momentum ak reads the sites as a uniform chain with a = 1 (page_ops_v6: R_n = n + 1); the fold k -> k +- pi is the staggered sign (-1)^x on integer (or uniformly half-integer) radii"}
    if spacing_dev >= 1e-9:
        log("WARNING: the sites are not a uniform unit chain (max |dR - 1| = %.3e): the lattice momentum reading assumes a = 1" % spacing_dev)
    R_win, wlab = window_radius(rec, a, g)
    Wsites = R < R_win
    Wrows = np.repeat(Wsites, 2)
    xw = R[Wsites]
    n_w = int(Wsites.sum())
    M, k = k_grid(rd.nsites, a.k_grid)
    p = fold_reduced(k, M)
    thr = a.k_threshold_pi * math.pi
    row["window"] = {"R": R_win, "label": wlab, "sites": n_w, "rows": int(Wrows.sum()), "record_zeros": rec.get("trapping_zeros"), "record_R_c": rec.get("R_c"),
                     "resolution_2_over_sites": (2.0 / n_w) if n_w else None,
                     "definition": "the sites with R < R_window for every quantity of this row (the interior orbitals, the window-restricted sets, the window's current); 2/sites the finite-extent scale of the window sets' moments"}
    row["momentum_grid"] = {"M": M, "k_threshold": thr, "k_threshold_over_pi": a.k_threshold_pi, "zone": ZONE,
                            "definition": "n(k) = sum_a |sum_x e^{-ikx} s_a(x)|^2 on k_j = 2 pi j/M in [-pi, pi), a = 1, the 2 spinor components summed; (1/M) sum_j n(k_j) is the number; every moment on the fold p_j = k_j mod pi in [-pi/2, pi/2); the fraction at |ap| > k_threshold; <cos 2ak> checked against Re sum_x s^dagger(x) s(x+2); the full-zone fraction at |ak| > k_threshold and the share at |ak| > pi/2 (the valley at ak = pi) beside, for the record",
                            "threshold_identity": "at k_threshold = pi/4 the grid set |ap| > pi/4 is the grid set cos 2ak < 0 (frac_counter_advected), exactly"}
    arrays = {"R": R, "N_applied": N_app, "v_applied": v_app, "k": k, "p_reduced": p}
    # the weighted dust number per site for R_50 (all channels; the stored dust columns)
    nd_site = np.zeros(rd.nsites)
    dusts = {}
    for kap in rd.kappas:
        if rd.n_dust[kap]:
            d, f = rd.dust(kap)
            dusts[kap] = (d, f)
            nd_site += rd.weights[kap] * IC.dust_number_site(d, f)
    tot = float(nd_site.sum())
    cum = np.cumsum(nd_site) / tot if tot > 0 else None
    R50 = IO.frac_radius(R, cum, 0.5) if cum is not None else None
    R90 = IO.frac_radius(R, cum, 0.9) if cum is not None else None
    row["dust"] = {"total": tot, "R_50": R50, "R_90": R90}
    arrays["dust_number_site"] = nd_site
    cut_list = [("root", R_win), ("R50", R50)] if R50 is not None else [("root", R_win)]
    row["current_cuts"] = {lab: x for lab, x in cut_list}
    R2_t = torch.as_tensor(np.repeat(R, 2))
    chans = []
    wsum = {}
    for kap in rd.kappas:
        t1 = time.time()
        ch = rd.channel(kap, orthonormalize=(a.loewdin == "on"))
        Phi, f = ch["Phi"], ch["f"]
        n_s = ch["n_sea"]
        h0, Q, H = IO.operators(R, m, kap, N_app, v_app)
        out = {"kappa": int(kap), "weight": ch["weight"], "n_sea": n_s, "n_dust": ch["n_dust"], "stored_orthonormality": ch["stored_orthonormality"],
               "loewdin_orthonormality": ch["loewdin_orthonormality"], "orthonormalization": ch["orthonormalization"], "dust_sea_overlap_max": ch["dust_sea_overlap_max"]}
        if a.check_eigs_kappa == kap and ch["e"] is not None:
            out["eig_vs_stored"] = float(np.max(np.abs(np.sort(IO.eigvalsh(H.toarray())) - np.sort(ch["e"]))))
        # the interior orbitals
        Phi_t = torch.as_tensor(Phi)
        src = IO.D13.source_modes(Phi_t, R2_t, R_win, H, dev, None, a.interior_weight, a.interior_max)
        S = src["S"].numpy()
        orbs = []
        for i in range(S.shape[1]):
            s = S[:, i]
            nk = orbital_momentum(s, M)
            mo = moments(nk, k, thr, M)
            rs2 = float(np.real(np.vdot(s[:-4], s[4:])))
            L_ext = extent_ipr(s)
            mo.update({"lam": float(src["lam"][i]), "w_in": float(src["w_in"][i]), "w_in_minus_lam": float(src["w_in"][i] - src["lam"][i]), "occ_C": float(src["occ_C"][i]),
                       "energy": float(src["energy"][i]), "energy_over_m": float(src["energy"][i] / m), "energy_spread": float(src["energy_spread"][i]),
                       "norm": float(np.sum(np.abs(s) ** 2)), "cos2k_real_space": rs2,
                       "cos2k_check": float(mo["cos2k"] * mo["number"] - rs2) if mo["cos2k"] is not None else None,
                       "extent_ipr_sites": L_ext, "resolution_2_over_L": (2.0 / L_ext) if L_ext else None})
            orbs.append(mo)
            arrays["k%d_orbital%d_nk" % (kap, i)] = nk
        if a.store_orbitals == "on" and S.shape[1]:
            arrays["k%d_orbitals" % kap] = S
        out["interior_orbitals"] = {"count": int(S.shape[1]), "n_window_modes": int(src["n_in_modes"]), "n_candidates_above": int(src["n_candidates_above"]), "weight_min": a.interior_weight,
                                    "orbitals": orbs, "lam_all_extremes": [float(np.min(src["lam_all"])), float(np.max(src["lam_all"]))] if len(src["lam_all"]) else None,
                                    "max_abs_w_in_minus_lam": float(max([abs(o["w_in_minus_lam"]) for o in orbs] or [0.0])),
                                    "occupation_weighted_mean": {key: occ_mean(orbs, key) for key in MEAN_KEYS},
                                    "source": "page_d13_readout_v1b.source_modes (the eigenvectors of C restricted to the window with eigenvalue >= weight_min, as unit modes s = Phi u)"}
        arrays["k%d_lam_all" % kap] = np.asarray(src["lam_all"], float)
        # the window-restricted sets
        dust_cols, f_d = dusts.get(kap, (np.zeros((rd.n2, 0), np.complex128), np.zeros(0)))
        R2cols = rd.R2(kap)
        sets = {"occupied": (Phi[Wrows], f), "sea": (Phi[Wrows][:, :n_s], np.ones(n_s)), "dust_stored": (dust_cols[Wrows], f_d) if dust_cols.shape[1] else None,
                "reference_R2": (R2cols[Wrows], np.ones(R2cols.shape[1]))}
        out["window_sets"] = {}
        nks = {}
        for name, item in sets.items():
            if item is None:
                continue
            Xw, ff = item
            nk = window_momentum(Xw, ff, xw, k)
            nks[name] = nk
            mo = moments(nk, k, thr, M)
            mo["cos2k_real_space"] = window_cos2k_real(Xw, ff, xw)
            mo["cos2k_check"] = float(mo["cos2k"] * mo["number"] - mo["cos2k_real_space"]) if mo["cos2k"] is not None else None
            mo["extent_sites"] = n_w
            out["window_sets"][name] = mo
            arrays["k%d_window_%s_nk" % (kap, name)] = nk
        dev_nk = nks["occupied"] - nks["reference_R2"]
        out["window_sets"]["deviation_C_minus_R2"] = moments(dev_nk, k, thr, M)
        arrays["k%d_window_deviation_nk" % kap] = dev_nk
        if dust_cols.shape[1]:
            nk_all = np.zeros(M)
            for j in range(dust_cols.shape[1]):
                nk_all += f_d[j] * orbital_momentum(dust_cols[:, j], M)
            out["dust_whole_chain"] = moments(nk_all, k, thr, M)
            arrays["k%d_dust_whole_chain_nk" % kap] = nk_all
        else:
            out["dust_whole_chain"] = None
        # the currents
        out["currents"] = {}
        for lab, x in cut_list:
            cur = {"R_cut": float(x)}
            cur["dust_number"] = IC.cut_current(H, R, x, dust_cols, f_d, "number") if dust_cols.shape[1] else 0.0
            cur["dust_energy"] = IC.cut_current(H, R, x, dust_cols, f_d, "energy") if dust_cols.shape[1] else 0.0
            cur["C_number"] = IC.cut_current(H, R, x, Phi, f, "number")
            cur["R2_number"] = IC.cut_current(H, R, x, R2cols, np.ones(R2cols.shape[1]), "number")
            cur["deviation_number"] = cur["C_number"] - cur["R2_number"]
            out["currents"][lab] = cur
            for key in ("dust_number", "dust_energy", "C_number", "R2_number", "deviation_number"):
                wsum.setdefault(lab, {}).setdefault(key, 0.0)
                wsum[lab][key] += ch["weight"] * cur[key]
        out["seconds"] = time.time() - t1
        chans.append(out)
        io_ = out["interior_orbitals"]["occupation_weighted_mean"]
        wso = out["window_sets"]["occupied"]
        wsd = out["window_sets"].get("dust_stored") or {}
        log("  kappa %d: window %d sites, %d interior orbitals (lam >= %g; max |w_in - lam| %.1e), orbitals' mean frac(|ap| > pi/4) %s (full zone %s), valley pi share %s, <cos 2ak> %s, 2/L %s; occupied set in the window N %.4f frac %s (full %s) <cos 2ak> %s; dust in the window N %s frac %s <cos 2ak> %s; dust current at the root %+.3e, at R_50 %s; %.1f s" % (
            kap, n_w, S.shape[1], a.interior_weight, out["interior_orbitals"]["max_abs_w_in_minus_lam"], IO.fx(io_["frac_abs_p_above"], "%.4f"), IO.fx(io_["frac_abs_k_above_full_zone"], "%.4f"),
            IO.fx(io_["valley_pi_share"], "%.4f"), IO.fx(io_["cos2k"], "%+.4f"), IO.fx(io_["resolution_2_over_L"], "%.4f"), wso["number"], IO.fx(wso["frac_abs_p_above"], "%.4f"),
            IO.fx(wso["frac_abs_k_above_full_zone"], "%.4f"), IO.fx(wso["cos2k"], "%+.4f"), IO.fx(wsd.get("number"), "%.4f"), IO.fx(wsd.get("frac_abs_p_above"), "%.4f"), IO.fx(wsd.get("cos2k"), "%+.4f"),
            out["currents"]["root"]["dust_number"], IO.fx(out["currents"].get("R50", {}).get("dust_number"), "%+.3e"), out["seconds"]))
        del Phi, Phi_t, S, R2cols, ch
    row["channels"] = chans
    row["weighted_currents"] = {lab: dict(v, label="summed with the multiplicity 16 kappa (currents are additive; not a correlation)") for lab, v in wsum.items()}
    row["seconds"] = time.time() - t0
    IO.save_outputs(pj, pz, row, arrays)
    log("%s T/r_s %.3f (%s; %s): window %s (%d sites), R_50 %s; weighted dust current at the root %+.4e, at R_50 %s; %.0f s" % (
        TITLE, rd.T_over_rs, rd.dtype_str, entry["history"], wlab, n_w, IO.fx(R50, "%.3f"), wsum.get("root", {}).get("dust_number", 0.0), IO.fx(wsum.get("R50", {}).get("dust_number"), "%+.4e"), row["seconds"]))
    return row


# ------------------------------------------------------------------------------------------------------------ the tables
def compose(a, hists):
    rows_all = IO.load_rows(a, READOUT, hists)
    rows = [r for r in rows_all if r.get("code") == CODE]
    if not rows:
        return
    n_other = len(rows_all) - len(rows)
    L = ["# %s: the core's kinematics on the stored slices (%s)" % (TITLE, CODE), "",
         "Inbox busi #144 item (2); requirements 5 I1's 'the core's drift against the flow'. Per engine channel without multiplicity on any printed correlation; the window (the sites inside the record's outer root, or R_c where the record holds no root) is stated in every row and is the same for every quantity of the row.", "",
         CONVENTION, "",
         "- interior orbitals: the eigenvectors of Phi^dagger Pi_W Phi with eigenvalue >= %g (page_d13_readout_v1b.source_modes), unit modes s = Phi u; their weight inside equals the eigenvalue (the check column); n(k) over the chain with the 2 spinor components summed and read on the fold p = k mod pi; the fraction at |ap| > %g pi (the full-zone fraction at |ak| > %g pi in parentheses, for the record), the share of the valley at ak = pi, <cos 2ak>, <sin 2ak>, <|ap|>, and 2/L for the orbital's extent L (inverse participation ratio in sites); the orbitals' means are weighted by their occupation <s|C|s>." % (a.interior_weight, a.k_threshold_pi, a.k_threshold_pi),
         "- the sets in the window: the window-restricted covariance of the occupied set (sea and dust), the sea, the stored dust, the reference R2 R2^dagger and the deviation C - R2 R2^dagger; N is the number in the window (1/M) sum_k n(k); the window's own finite-extent scale is 2/(its site count); the dust over the whole chain beside.",
         "- the currents: Tr[X i[h_m, Pi_{R > R_cut}] X^dagger] (page_ops_v6.number_current_csr), outward positive, at the root and at R_50 (the radius holding 50 percent of the weighted dust); per channel unweighted; the 16 kappa sums in their own table.", ""]
    if n_other:
        L += ["- %d core json(s) of another version in the output folder were set aside (only %s rows are tabulated)." % (n_other, CODE), ""]
    for hk in hists:
        rs_ = [r for r in rows if r["history"] == hk]
        if not rs_:
            continue
        L += ["## %s: %s" % (hk, rs_[0]["history_label"]), "", "Engine: %s" % rs_[0]["engine"], "", "### The interior orbitals (per channel; no multiplicity; the reduced zone)", "",
              "| T/r_s | dtype | window (sites) | kappa | orbitals (lam min) | max abs(w_in - lam) | mean energy/m | frac(abs ap > %g pi) reduced (full zone) | valley pi share | <cos 2ak> | <sin 2ak> | <abs ap> | 2/L | cos2k check (max abs) |" % a.k_threshold_pi,
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs_:
            for c in r["channels"]:
                io = c["interior_orbitals"]
                om = io["occupation_weighted_mean"]
                orbs = io["orbitals"]
                L.append("| %.3f | %s | %s (%d) | %d | %d (%s) | %.1e | %s | %s (%s) | %s | %s | %s | %s | %s | %s |" % (
                    r["T_over_rs"], r["dtype"], r["window"]["label"].split(" = ")[0].replace("inside the record's ", ""), r["window"]["sites"], c["kappa"], io["count"],
                    ("%.4f" % min(o["lam"] for o in orbs)) if orbs else "-", io["max_abs_w_in_minus_lam"],
                    ("%.4f" % (sum(o["occ_C"] * o["energy_over_m"] for o in orbs) / sum(o["occ_C"] for o in orbs))) if orbs else "-",
                    IO.fx(om.get("frac_abs_p_above"), "%.4f"), IO.fx(om.get("frac_abs_k_above_full_zone"), "%.4f"), IO.fx(om.get("valley_pi_share"), "%.4f"), IO.fx(om.get("cos2k"), "%+.4f"),
                    IO.fx(om.get("sin2k"), "%+.4f"), IO.fx(om.get("mean_abs_p"), "%.4f"), IO.fx(om.get("resolution_2_over_L"), "%.4f"),
                    ("%.1e" % max(abs(o.get("cos2k_check") or 0.0) for o in orbs)) if orbs else "-"))
        L += ["", "### The sets in the window and the dust over the chain (per channel; no multiplicity; the reduced zone): N; frac(abs ap > %g pi) reduced (full zone); <cos 2ak>" % a.k_threshold_pi, "",
              "| T/r_s | kappa | window R (sites) | occupied set | sea | stored dust | reference R2 | deviation C - R2 (where N > 0) | dust over the whole chain |",
              "|---|---|---|---|---|---|---|---|---|"]
        fmt4 = lambda s: ("%.4f; %s (%s); %s" % (s["number"], IO.fx(s.get("frac_abs_p_above"), "%.4f"), IO.fx(s.get("frac_abs_k_above_full_zone"), "%.4f"), IO.fx(s.get("cos2k"), "%+.4f"))) if s else "-"  # noqa: E731
        for r in rs_:
            for c in r["channels"]:
                ws = c["window_sets"]
                L.append("| %.3f | %d | %.3f (%d) | %s | %s | %s | %s | %s | %s |" % (
                    r["T_over_rs"], c["kappa"], r["window"]["R"], r["window"]["sites"], fmt4(ws.get("occupied")), fmt4(ws.get("sea")), fmt4(ws.get("dust_stored")), fmt4(ws.get("reference_R2")),
                    fmt4(ws.get("deviation_C_minus_R2")), fmt4(c.get("dust_whole_chain"))))
        L += ["", "### The outward number current (per channel, unweighted; outward positive) at the root and at R_50", "",
              "| T/r_s | kappa | R_t cut | dust number | dust energy | C | R2 | C - R2 | R_50 cut | dust number | dust energy | C | R2 | C - R2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs_:
            for c in r["channels"]:
                cr = c["currents"].get("root", {})
                c5 = c["currents"].get("R50", {})
                L.append("| %.3f | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["T_over_rs"], c["kappa"], IO.fx(cr.get("R_cut"), "%.3f"), IO.fx(cr.get("dust_number"), "%+.4e"), IO.fx(cr.get("dust_energy"), "%+.4e"), IO.fx(cr.get("C_number"), "%+.4e"),
                    IO.fx(cr.get("R2_number"), "%+.4e"), IO.fx(cr.get("deviation_number"), "%+.4e"), IO.fx(c5.get("R_cut"), "%.3f"), IO.fx(c5.get("dust_number"), "%+.4e"), IO.fx(c5.get("dust_energy"), "%+.4e"),
                    IO.fx(c5.get("C_number"), "%+.4e"), IO.fx(c5.get("R2_number"), "%+.4e"), IO.fx(c5.get("deviation_number"), "%+.4e")))
        L += ["", "### The currents summed with the multiplicity 16 kappa (additive quantities, not correlations)", "", "| T/r_s | at the root: dust number; dust energy; C - R2 | at R_50: dust number; dust energy; C - R2 | R_50 | R_90 |", "|---|---|---|---|---|"]
        for r in rs_:
            wr = r["weighted_currents"].get("root", {})
            w5 = r["weighted_currents"].get("R50", {})
            L.append("| %.3f | %s; %s; %s | %s; %s; %s | %s | %s |" % (
                r["T_over_rs"], IO.fx(wr.get("dust_number"), "%+.4e"), IO.fx(wr.get("dust_energy"), "%+.4e"), IO.fx(wr.get("deviation_number"), "%+.4e"), IO.fx(w5.get("dust_number"), "%+.4e"),
                IO.fx(w5.get("dust_energy"), "%+.4e"), IO.fx(w5.get("deviation_number"), "%+.4e"), IO.fx(r["dust"]["R_50"], "%.3f"), IO.fx(r["dust"]["R_90"], "%.3f")))
        L += ["", "### Checks", ""]
        for r in rs_:
            cos_checks = ([abs(o.get("cos2k_check") or 0.0) for c in r["channels"] for o in c["interior_orbitals"]["orbitals"]] +
                          [abs(s.get("cos2k_check") or 0.0) for c in r["channels"] for s in c["window_sets"].values() if isinstance(s, dict)]) or [0.0]
            ls = r.get("lattice_spacing") or {}
            L.append("- T/r_s %.3f: Loewdin orthonormality after %s (stored %s); eig vs stored e_k %s; largest |w_in - lam| %.1e; largest cos2k check %.1e; unit lattice spacing %s (max |dR - 1| %s); %s; %.0f s." % (
                r["T_over_rs"], IO.fx(max((c["loewdin_orthonormality"] or 0.0) for c in r["channels"]), "%.1e"), IO.fx(max(c["stored_orthonormality"] for c in r["channels"]), "%.1e"),
                IO.fx(max([c["eig_vs_stored"] for c in r["channels"] if c.get("eig_vs_stored") is not None] or [None]), "%.1e") if any(c.get("eig_vs_stored") is not None for c in r["channels"]) else "-",
                max(c["interior_orbitals"]["max_abs_w_in_minus_lam"] for c in r["channels"]), max(cos_checks), ls.get("unit_uniform"), IO.fx(ls.get("max_abs_dR_minus_1"), "%.1e"),
                "complex64 slice: floor %s" % r["floor"]["source_rel_range"] if r["complex64"] else "complex128", r["seconds"]))
        L.append("")
    L.append("- every number above is printed by the script from the stored slices and the records; the reading is the lead's.")
    d = os.path.join(a.out, READOUT)
    for hk in hists:
        if any(r["history"] == hk for r in rows):
            with open(os.path.join(d, "%s_%s.md" % (TITLE, hk)), "w") as f:
                f.write("\n".join(L) + "\n")
    with open(os.path.join(d, "%s_all.json" % TITLE), "w") as f:
        json.dump({"code": CODE, "title": TITLE, "convention": CONVENTION, "histories": hists, "rows": rows}, f, default=IO.jdefault, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    IO.add_common_arguments(ap)
    IC.add_arguments(ap)
    add_arguments(ap)
    a = ap.parse_args(argv)
    a.readouts = READOUT
    IO.READOUT_MODULES[READOUT] = MODULE     # the driver's table maps core to core_kinematics_v1; this process runs v2
    IO.run(a)


if __name__ == "__main__":
    main()
