#!/usr/bin/env python
"""page_d13_readout_v1d.py (v6o W1): page_d13_readout_v1b.py with (a''), the massless-sector halving of every dense entropy and the tail-majorant rule
of the kappa sum (built from v1b by build_page_history_v6o_v1d.py's exact replacement list; v1b is not touched; the changed lines carry "v1c" or "v1d"):
    ChannelReadout.checkpoint uses page_w1_sector_v1d.exterior_entropies_sector and cross_cut_readout_sector when the caller sets sector_on (the +
    sector's blocks at half the dimension, the - sector beside at the check clocks, the pairing defects printed); killing_vacuum takes the route
    "sector" (P_-(h) from the eigendecomposition of h_+); set_vacuum carries the sector data; tr_min_h0 returns the pairing audit Tr min(h, 0) =
    -Tr|h|/2 (defect 13); the probes accept a banded operator per sub-step; ksum_table applies the formed-energy tail-majorant rule
    (page_w1_sector_v1d.convergence_rule, defect 10) and prints the retired last-channel share beside, deciding nothing by it; the S_X columns are
    labelled raw. W1_fix_v1c: the caller's w1_floor (the state's impurity, the a priori Chebyshev bound) annotates every cut's excess with its
    propagation floors and resolution on both routes (page_w1_sector_v1d.excess_floors); ksum_table carries the floors; format_line prints them.
    W1_fix_v1d: the a priori bound carries the construction allowance (page_w1_sector_v1d.propagation_floor); nothing of this file's arithmetic changes.
    v1b's docstring follows.

page_d13_readout_v1b.py: page_d13_readout_v1.py with (a'), the cross-cut block trace norms of hreduce's EVOLVEDCUT_theorem_v1 EC1
(built from v1 by build_page_d13_readout_v1b.py's exact replacement list; v1 is not touched). At every clock where the held Killing vacuum
P_-(h(t)) is formed (the baseline clocks, T_0 included) and at every cut X = {R > x}:
    t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1, the exact bound |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 for the pure Gaussian state C = Phi Phi^dagger
    and the projector P_- = V V^dagger, from the singular values of the X-by-Xc block of C - P_- formed from the columns as
    Phi_X Phi_Xc^dagger - V_X V_Xc^dagger (n_X x n_Xc; two GEMMs on the row slices, views when the rows are ordered; never an n2 x n2 or
    N_occ x N_occ object), resolved by torch.linalg.svdvals (cross_cut_norms states the route and its cost at n = 16384 on an H100: the
    block at 48 r_s is 26624 x 6144, 2.6 GiB, its GEMMs about a second, its singular values seconds to tens of seconds; the four cuts under a
    minute and under 8 GiB transient per baseline clock); the check |S_X(C) - S_X(P_-)| <= t_X printed per cut, a violation a FAIL line
    carrying the premises' defects ||Phi^dagger Phi - 1||_F, ||V^dagger V - 1||_F and the purity gap (the theorem's premises are the pure
    Gaussian state and the projector P_-); the consistency of the block with (a)'s spectra (the Frobenius identity and the triangle bounds);
    the full nu = ||C - P_-||_1 by the principal angles between ran C and ran P_- where affordable (full_trace_norm: gram, the N_neg x N_neg
    eigvalsh of 1 - M^dagger M with M = Phi^dagger V, about 16 GiB and tens of seconds at n = 16384; svd, exact to rounding, minutes; auto picks
    gram within --cut-norm-full-GB and the free device memory), else its labelled bounds by the rank and the largest singular value and by
    Cauchy-Schwarz with ||C - P_-||_2 exact; the per-channel values in the cut dicts (t_X, abs_dS, slack, bound_ok, the singular values'
    count and largest) and in sc["trace_norm"], the weighted values (16 kappa) in the CLI's clock summary (with the weighted check) and in
    the ksum table (t_X_cut<x>, nu_trace_weighted, bound_ok_cut<x>); the singular values of every block (sv_block_cut<x>) and of (1 - C) V
    (sv_full) in the npz. Thresholds: --cut-norm, --cut-block-route, --cut-norm-full, --cut-norm-full-GB, --cut-bound-tol, --cut-sigma-tol.
    cross_cut_norms(A, B, R2, cuts) serves any two orthonormal column sets (the reference's vacuum against the current one, for instance).
v1's docstring follows.

page_d13_readout_v1.py (reviewer A 5.1 for the solve builder, the busi lane; reorientation 4, the solve entry's item 1; D13's protocol
REVIEWA_RULING_O1_D13_protocol_v1 sec. 1; review_PAGE_v3b's telescoping identity and rank-16 bound; LIGHTCHANNEL LC5's E_inv identity;
TWOFIELD CF3's source-only root coefficient). The collector-free readouts of Computation A on the pure Gaussian history as post-processing
on the occupied columns Phi (torch complex128, n2 x N_occ per channel kappa, n2 = 2 n, C = Phi Phi^dagger pure), the clocked operator
h(t) = (1/2){N, h_0} + (1/2){v, Q} as page_ops_v6.clocked_csr builds it (host CSR, applied banded on the device), the grid R_n = n + 1 and
the cuts. No mutual information, no selection, no archive. Every function returns plain dicts of floats and numpy arrays; the channel
weights 16 kappa are applied by the caller (one channel per call).

(a) exterior_entropies: S_X = Tr s(C_X) beyond a geometric cut from the exterior block's nonzero spectrum, the eigenvalues of the smaller
    of Phi_X^dagger Phi_X (N_occ x N_occ) and Phi_X Phi_X^dagger (n_X x n_X), never the n2 x n2 covariance; the complement's S_Xc beside it
    (equal on a pure state: the purity check, S_X - S_Xc the unitarity defect's print); the held Killing vacuum P_-(h(t))'s S_X at the same
    cut (its columns Vneg from killing_vacuum: one dense eigendecomposition of h per baseline clock, every --vac-every-rs, cached) and the
    excess S_X - S_X^vac. Cuts 16, 24, 30 and 48 r_s by default.
(b) inventories and ledger_windows: the detector-energy ledger at the cuts 16 and 30 r_s. E_x^C(t) = Tr[{Pi_x, h}/2 C] = Re Tr[Pi_x h C] is
    the Killing energy beyond the cut; on the records' piecewise-constant h it changes within a record by the exact flow (the integral of
    Tr[J_x C], J_x = (i/2)[h^2, Pi_x] of page_ops_v6.current_out_csr; the vacuum carries none: Tr[J_x P_-] = 0 because [h^2, P_-] = 0) and
    at a record boundary by the jump Re Tr[Pi_x (h' - h) C], so E_x^C(t_1) - E_x^C(t_0) = sum(flows) + sum(jumps) exactly (the telescoping
    identity; its residual, printed, is the propagation's own error). Per retarded window of at least --window-rs the 2 terms: the
    inventory beyond the cut at the window's start on D = C - P_-(t_0) and the energy that crossed within it, the current's integral and
    the jumps' sum separately; the reference's change Tr[H_x P_-](t_1) - Tr[H_x P_-](t_0) subtracted where both ends are baseline clocks.
    The per-record flows and jumps at the cuts come from the propagation (page_history_v6n.step_record); a run without them (v6m) gives the
    inventories only.
(c) killing_shells and shell_content: the exterior block of h beyond R_cut (24 r_s), one dense eigendecomposition per --ext-every-rs; its
    Killing energies binned by |e| in units of the FIXED T_H^ref (1/(4 pi 2 g E) at the record of --TH-ref-rs, 40 r_s): bins of --bin-TH up
    to --bin-max-TH, doubling bins up to the band top 1/a, 1 bin above the band top; per sign of e; within each bin the outgoing and
    incoming subspaces by the sign of the radial velocity i[h_X, R]; per checkpoint the number and energy content of C minus a reference
    on each (the block's own ground 1(e < 0) at every checkpoint; P_-(h(t)) at the baseline clocks, their difference the mixing at the
    block's inner edge); the hard class (lower edge at or above 25 T_H^ref and below the band top) against the soft one; the weight above
    the band top printed as bound content, not escape. The hard fluence per retarded window is the hard outgoing inventory's change between
    the window's ends (hard_fluence_windows; the chain's end returns nothing before the first return, printed beside).
(d) probe_vectors, probe_split, ChannelReadout.probes: L_old by back-propagated probes. J_x = (i/2)[h^2, Pi_x] has rank at most 16 on the
    range-2 chain (its support is 4 sites on either side of the cut), J_x = sum_a j_a w_a w_a^dagger, so L_x(t) = Tr[C(t) J_x(t)] =
    sum_a j_a <w_a(t_ref)| C(t_ref) |w_a(t_ref)> with the probes back-propagated through the records' fields (exp(+i h_k tau) applied in
    reverse order with the same Chebyshev steps and sub-steps the sea took, no collection): L_old = sum_a j_a <w_a| C - P_-(t_ref) |w_a>,
    L_sea = sum_a j_a <w_a| P_-(t_ref) |w_a> (the geometry's work after t_ref, the prescribed exterior's record jumps and the join's seam
    included), L_src the part carried by the identified source modes; beside them the rank-16 bound sum_a |j_a| = ||J_x||_1 (the largest
    |L| any state carries through the cut) and the direct Tr[C(t) J_x] as the propagation's check. Readout clocks every 4 r_s from 40 to
    the first return; reference clocks 12, 20 and 40.
(e) source_modes, source_transfer: the eigenvectors of Phi^dagger Pi_in Phi (Pi_in the sites inside the outer root at the reference clock;
    equivalently of C restricted to the interior, the same nonzero spectrum) with weight at least 0.99 inside, s = Phi u (unit, in ran C):
    their occupation <s|C|s> (1 up to rounding, printed), the vacuum's <s|P_-(t_ref)|s>, their Killing energy and spread; propagated with
    the sea (no collection); their same-packet transfer tau(t) = ||Pi_{>R} s(t)||^2 to the collar (16 r_s) and beyond R_cut, the state's
    content on the exterior part w = ||Phi^dagger Pi s||^2 and the inferred occupation w/tau with its excess over <s|C|s> (readout 9: equal
    only where the packet's weight is that mode's alone).
(f) first_return: the continuum light round trip cut -> chain end -> cut in the record's fields (outgoing N - |v|, ingoing N + |v|) from
    the run's start, and the formation's light (--form-rs, from the root) to the end and back to the cut; the join's departure (the
    record's N, v against the vacuum exterior at E_k over --join-rs) is the row's.
(g) killing_density_sites, enclosed_source: the light field's enclosed source I_L(R) = 4 pi int_0^R r^2 N (rho_L + u J_L) dr, which on the
    chain is the Killing energy of D = C - P_- enclosed within R (the sum over sites of Re Tr[Pi_n h D]: N rho + v J is the site's Killing
    energy density, rho the h_0 density conjugate to N, J the Q density conjugate to v; TWOFIELD CF3), at the outer root R, over the collar
    factor kappa_R = (N + v)'(R): delta R_L = G I_L/(R kappa_R) = 2 G I_L/(N D_R); D_R = 1 - 8 pi G R^2 (rho_H - J_H) from the stored heavy
    densities beside it as the convention check (kappa from the densities against kappa from the slope).

The CLI reads v6m/v6n states (.pt) with their record json and fields npz, computes (a) to (c), (e) to (g) at each state's clock (and (d)
with --ref-state T:PATH references), writes <tag>_d13.json (every scalar) and <tag>_d13.npz (every raw per-cut, per-channel, per-bin array;
no plotting here) and is resumable (a clock already in the json is skipped unless --force). Every threshold is a named argument with the
ruling's value as default (add_arguments). Device code in torch as v6m's."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.special import jv

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def code_dirs():
    """where page_ops_v6.py (and page_history_v6m.py for the CLI) live: this directory, $PAGE_CODE_V6, code_v6 beside or above."""
    c = [HERE]
    if os.environ.get("PAGE_CODE_V6"):
        c += os.environ["PAGE_CODE_V6"].split(os.pathsep)
    c += [os.path.join(HERE, "code_v6"), os.path.join(HERE, "..", "code_v6"), os.path.join(HERE, "..")]
    return [os.path.abspath(x) for x in c]


for _d in code_dirs():
    if os.path.isfile(os.path.join(_d, "page_ops_v6.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
import page_ops_v6 as OPS  # noqa: E402
import page_w1_sector_v1d as W1S  # noqa: E402  (v1c, v6o W1 v1d: the sector route, the audits, the majorant rule, the propagation's floor with the construction allowance; W1_fix_v1d.md)

CD = OPS.CD
RD = OPS.RD
T_START = time.time()
CODE = "page_d13_readout_v1d.py"  # v1d (v6o W1)
TRACE_CHUNK = [1024]

RULING = {"cuts_rs": "16,24,30,48", "ledger_cuts_rs": "16,30", "probe_cuts_rs": "16,30", "R_cut_rs": 24.0, "R_collar_rs": 16.0,
          "hard_TH": 25.0, "TH_ref_rs": 40.0, "band_top": 1.0, "bin_TH": 0.5, "bin_max_TH": 40.0, "window_rs": 13.0,
          "vac_every_rs": 10.0, "ext_every_rs": 10.0, "ref_clocks_rs": "12,20,40", "probe_from_rs": 40.0, "probe_every_rs": 4.0,
          "n_probes": 16, "source_weight": 0.99, "source_max": 128, "collar_window": 32, "form_rs": 12.125, "join_rs": "32,64",
          "columns_at_rs": "12,20,40", "columns_every_rs": 10.0}


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T_START, msg), flush=True)


def _f(x, fmt):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def _fl(s):
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _jdefault(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    return str(o)


# ------------------------------------------------------------------------------------------------------------ the arguments
def add_arguments(ap, common=True):
    """the D13 thresholds as named arguments with the ruling's values as defaults; common=True adds the arguments page_history_v6m already
    has (the CLI needs them, page_history_v6n does not)."""
    g = ap.add_argument_group("D13 readouts (page_d13_readout_v1; the ruling's values as defaults)")
    g.add_argument("--cuts-rs", dest="cuts_rs", default=RULING["cuts_rs"], help="(a) the exterior cuts (r_s)")
    g.add_argument("--ledger-cuts-rs", dest="ledger_cuts_rs", default=RULING["ledger_cuts_rs"], help="(b) the ledger's cuts (r_s)")
    g.add_argument("--probe-cuts-rs", dest="probe_cuts_rs", default=RULING["probe_cuts_rs"], help="(d) the probes' cuts (r_s)")
    g.add_argument("--hard-TH", dest="hard_TH", type=float, default=RULING["hard_TH"], help="(c) the hard threshold in units of T_H^ref")
    g.add_argument("--TH-ref-rs", dest="TH_ref_rs", type=float, default=RULING["TH_ref_rs"], help="the record whose 2gE fixes T_H^ref")
    g.add_argument("--TH-ref", dest="TH_ref", type=float, default=None, help="T_H^ref given directly (lattice units; overrides --TH-ref-rs)")
    g.add_argument("--band-top", dest="band_top", type=float, default=RULING["band_top"], help="(c) the band top 1/a; weight above it is bound content")
    g.add_argument("--vac-every-rs", dest="vac_every_rs", type=float, default=RULING["vac_every_rs"], help="the dense vacuum's baseline clocks")
    g.add_argument("--vac-extra-rs", dest="vac_extra_rs", default="", help="extra baseline clocks (r_s)")
    g.add_argument("--vac-route", dest="vac_route", default="dense", choices=["dense", "pair", "sector"],                                                     # v1c (v6o W1)
                   help="dense: torch eigh of h (v6m's); pair: the massless chain's conjugate-pair structure h = i A (the real eigenproblem of h^2, half the work); sector: the + sector block h_+ (v1c: 1/8 of the dense work; page_history_v6o sets it when the state respects S_stag)")   # v1c (v6o W1)
    g.add_argument("--vac-cluster-tol", dest="vac_cluster_tol", type=float, default=1e-9, help="the pair route's cluster tolerance (relative to max e^2)")
    g.add_argument("--d13-vacuum", dest="d13_vacuum", default="on", choices=["on", "off"], help="off: no dense vacuum (the vacuum columns absent)")
    g.add_argument("--cut-norm", dest="cut_norm", default="on", choices=["on", "off"],
                   help="(a') v1b: the cross-cut block trace norms t_X = 2||Pi_X (C - P_-) Pi_Xc||_1 and the check |S_X(C) - S_X(P_-)| <= t_X at the clocks with the vacuum")
    g.add_argument("--cut-block-route", dest="cut_block_route", default="svd", choices=["svd", "gram"],
                   help="(a') the block's singular values: svdvals of the n_X x n_Xc block (exact to rounding) or the smaller Gram's eigvalsh (the small ones to sqrt accuracy)")
    g.add_argument("--cut-norm-full", dest="cut_norm_full", default="auto", choices=["auto", "gram", "svd", "bound", "off"],
                   help="(a') the full ||C - P_-||_1: gram (eigvalsh of 1 - M^dagger M, M = Phi^dagger V), svd (svdvals of (1 - C) V, exact), bound (rank x sigma_max and Cauchy-Schwarz, no N x N object), auto (gram where its memory fits --cut-norm-full-GB and the free device memory, else bound)")
    g.add_argument("--cut-norm-full-GB", dest="cut_norm_full_GB", type=float, default=24.0, help="(a') the device memory allowed to the full trace norm (GiB)")
    g.add_argument("--cut-bound-tol", dest="cut_bound_tol", type=float, default=1e-6, help="(a') the rounding allowance of the check (nats): |S_X(C) - S_X(P_-)| <= t_X + tol")
    g.add_argument("--cut-sigma-tol", dest="cut_sigma_tol", type=float, default=1e-10, help="(a') singular values of (1 - C) V above this count in rank(C - P_-)")
    g.add_argument("--shells", dest="shells", default="on", choices=["on", "off"], help="(c) the exterior block's Killing shells")
    g.add_argument("--ref-clocks-rs", dest="ref_clocks_rs", default=RULING["ref_clocks_rs"], help="(d), (e) the reference clocks")
    g.add_argument("--probe-from-rs", dest="probe_from_rs", type=float, default=RULING["probe_from_rs"])
    g.add_argument("--probe-every-rs", dest="probe_every_rs", type=float, default=RULING["probe_every_rs"])
    g.add_argument("--n-probes", dest="n_probes", type=int, default=RULING["n_probes"], help="(d) at most this many probes (the rank is at most 16)")
    g.add_argument("--source-weight", dest="source_weight", type=float, default=RULING["source_weight"], help="(e) the weight inside the root")
    g.add_argument("--source-max", dest="source_max", type=int, default=RULING["source_max"], help="(e) at most this many source modes per reference")
    g.add_argument("--source-region", dest="source_region", default="root", choices=["root", "2gE"], help="(e) the interior: inside the outer root (2gE where none)")
    g.add_argument("--collar-window", dest="collar_window", type=int, default=RULING["collar_window"], help="the collar flow's windowed split (sites; v6k's reading)")
    g.add_argument("--form-rs", dest="form_rs", type=float, default=RULING["form_rs"], help="(f) the formation's record clock")
    g.add_argument("--ref-cache", dest="ref_cache", default="host", choices=["host", "disk", "recompute"], help="(d) where the references' vacuum columns are kept")
    g.add_argument("--ref-cache-GB", dest="ref_cache_GB", type=float, default=8.0)
    g.add_argument("--columns-kappas", dest="columns_kappas", default="", help="save the occupied columns for these channels only (default all)")
    if common:
        g.add_argument("--R-cut-rs", dest="R_cut_rs", type=float, default=RULING["R_cut_rs"])
        g.add_argument("--R-collar-rs", dest="R_collar_rs", type=float, default=RULING["R_collar_rs"])
        g.add_argument("--window-rs", dest="window_rs", type=float, default=RULING["window_rs"])
        g.add_argument("--ext-every-rs", dest="ext_every_rs", type=float, default=RULING["ext_every_rs"])
        g.add_argument("--bin-TH", dest="bin_TH", type=float, default=RULING["bin_TH"])
        g.add_argument("--bin-max-TH", dest="bin_max_TH", type=float, default=RULING["bin_max_TH"])
        g.add_argument("--join-rs", dest="join_rs", default=RULING["join_rs"])
        g.add_argument("--chunk", type=int, default=2048)
        g.add_argument("--cheb-tol", dest="cheb_tol", type=float, default=1e-14)
        g.add_argument("--trace-chunk", dest="trace_chunk", type=int, default=1024)


def cfg_from_args(a):
    g = lambda k, d: getattr(a, k, d)   # noqa: E731
    return {"cuts_rs": _fl(g("cuts_rs", RULING["cuts_rs"])), "ledger_cuts_rs": _fl(g("ledger_cuts_rs", RULING["ledger_cuts_rs"])),
            "probe_cuts_rs": _fl(g("probe_cuts_rs", RULING["probe_cuts_rs"])), "R_cut_rs": float(g("R_cut_rs", RULING["R_cut_rs"])),
            "R_collar_rs": float(g("R_collar_rs", RULING["R_collar_rs"])), "hard_TH": float(g("hard_TH", RULING["hard_TH"])),
            "TH_ref_rs": float(g("TH_ref_rs", RULING["TH_ref_rs"])), "TH_ref": g("TH_ref", None), "band_top": float(g("band_top", RULING["band_top"])),
            "bin_TH": float(g("bin_TH", RULING["bin_TH"])), "bin_max_TH": float(g("bin_max_TH", RULING["bin_max_TH"])),
            "window_rs": float(g("window_rs", RULING["window_rs"])), "vac_every_rs": float(g("vac_every_rs", RULING["vac_every_rs"])),
            "vac_extra_rs": _fl(g("vac_extra_rs", "")), "vac_route": g("vac_route", "dense"), "vac_cluster_tol": float(g("vac_cluster_tol", 1e-9)),
            "vacuum": g("d13_vacuum", "on"), "shells": g("shells", "on"), "ext_every_rs": float(g("ext_every_rs", RULING["ext_every_rs"])),
            "ref_clocks_rs": _fl(g("ref_clocks_rs", RULING["ref_clocks_rs"])), "probe_from_rs": float(g("probe_from_rs", RULING["probe_from_rs"])),
            "probe_every_rs": float(g("probe_every_rs", RULING["probe_every_rs"])), "n_probes": int(g("n_probes", RULING["n_probes"])),
            "source_weight": float(g("source_weight", RULING["source_weight"])), "source_max": int(g("source_max", RULING["source_max"])),
            "source_region": g("source_region", "root"), "collar_window": int(g("collar_window", RULING["collar_window"])),
            "form_rs": float(g("form_rs", RULING["form_rs"])), "join_rs": str(g("join_rs", RULING["join_rs"])), "ref_cache": g("ref_cache", "host"),
            "ref_cache_GB": float(g("ref_cache_GB", 8.0)), "chunk": int(g("chunk", 2048)), "cheb_tol": float(g("cheb_tol", 1e-14)),
            "trace_chunk": int(g("trace_chunk", 1024)), "columns_at_rs": _fl(g("columns_at_rs", RULING["columns_at_rs"])),
            "columns_every_rs": float(g("columns_every_rs", RULING["columns_every_rs"])), "columns_format": g("columns_format", "pt"),
            "columns_kappas": [int(x) for x in str(g("columns_kappas", "")).split(",") if str(x).strip()],
            "cut_norm": g("cut_norm", "on"), "cut_block_route": g("cut_block_route", "svd"), "cut_norm_full": g("cut_norm_full", "auto"),       # v1b (a')
            "cut_norm_full_GB": float(g("cut_norm_full_GB", 24.0)), "cut_bound_tol": float(g("cut_bound_tol", 1e-6)), "cut_sigma_tol": float(g("cut_sigma_tol", 1e-10))}


# ---------------------------------------------------------------------------------------------------------------- the arithmetic
def tr_blocked(B, X, w=None, chunk=None):
    """Re tr[X^dagger W B X] = Tr[{W, B}/2 C] for a device Banded B and a diagonal weight column W (or None), over column blocks of X."""
    ch = int(chunk or TRACE_CHUNK[0])
    tot = 0.0
    for j0 in range(0, X.shape[1], ch):
        Xb = X[:, j0: j0 + ch]
        Y = B.apply(Xb)
        if w is not None:
            Y = w * Y
        tot += float(torch.sum((Xb.conj() * Y).real))
        del Y
    return tot


def cheb_apply(Bh, X, tau, tol):
    """exp(-i tau h) X for a device Banded h: the Chebyshev series on [-w, w] (w its Gershgorin bound), the order by the Bessel tail below
    tol (page_ops_v6.cheb_order); tau of either sign (the back-propagation). The same arithmetic as page_history_v6m.cheb_apply."""
    if X.shape[1] == 0:
        return X, 0, 0.0
    w = Bh.bound
    z = w * tau
    K, tail = OPS.cheb_order(z, tol)
    k = np.arange(K + 1)
    ck = jv(k, z) * ((-1j) ** k) * np.where(k == 0, 1.0, 2.0)
    T0 = X
    T1 = Bh.apply(X) / w
    Y = complex(ck[0]) * T0 + complex(ck[1]) * T1
    for kk in range(2, K + 1):
        T2 = 2.0 * Bh.apply(T1) / w - T0
        Y = Y + complex(ck[kk]) * T2
        T0, T1 = T1, T2
    return Y, K, tail


def propagate(Bh, X, tau, tol, chunk):
    K = 0; tail = 0.0
    for j0 in range(0, X.shape[1], chunk):
        X[:, j0: j0 + chunk], K, tail = cheb_apply(Bh, X[:, j0: j0 + chunk], tau, tol)
    return X, K, tail


def outer_root(N, v, R):
    """the outermost zero of N - |v| (linear interpolation) or None (page_history_v6m.HistChannel.outer_root)."""
    f = np.asarray(N, float) - np.abs(np.asarray(v, float))
    s = np.nonzero((f[:-1] <= 0) & (f[1:] > 0))[0]
    if len(s) == 0:
        return None
    i = int(s[-1]); R = np.asarray(R, float)
    return float(R[i] + (R[i + 1] - R[i]) * (-f[i]) / (f[i + 1] - f[i]))


# ------------------------------------------------------------------------------------------------------------- (a) the entropies
def gram_eigs(X):
    """the nonzero spectrum of X X^dagger from the smaller of X^dagger X and X X^dagger; (eigenvalues ascending, the route's label)."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return torch.zeros(0, dtype=RD, device=X.device), "empty"
    if X.shape[1] <= X.shape[0]:
        G = X.mH @ X; route = "gram %d x %d" % (X.shape[1], X.shape[1])
    else:
        G = X @ X.mH; route = "block %d x %d" % (X.shape[0], X.shape[0])
    G = 0.5 * (G + G.mH)
    ev = torch.linalg.eigvalsh(G)
    del G
    return ev, route


def entropy_of_eigs(ev):
    return float(OPS.binary_entropy_t(ev).sum()) if ev.numel() else 0.0


# ----------------------------------------------------------- (a') the cross-cut block trace norms (hreduce's EVOLVEDCUT EC1; v1b)
EPS64 = 2.220446049250313e-16


def _h2(x):
    """the binary entropy in nats, elementwise; 0 at 0 and at 1."""
    x = torch.as_tensor(x, dtype=RD).clamp(0.0, 1.0)
    y = torch.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    xm = x[m]
    y[m] = -(xm * torch.log(xm) + (1.0 - xm) * torch.log1p(-xm))
    return y


def pure_cut_F(s):
    """F(s) = h_2((1 - sqrt(1 - 4 s^2))/2) on 0 <= s <= 1/2: a projector Z has S_X(Z) = sum_j F(sigma_j(K_Z)) with K_Z = Pi_X Z Pi_Xc, since
    Z_X (1 - Z_X) = K_Z K_Z^dagger (the paired single-object identity behind EC1; 0 <= F' <= 2 is the bound's constant)."""
    s = torch.as_tensor(s, dtype=RD)
    return _h2(0.5 * (1.0 - torch.sqrt((1.0 - 4.0 * s * s).clamp_min(0.0))))


def cut_index(R2, x):
    """the modes beyond the cut (R2 > x): the slice start i0 when the rows are ordered by R (R2 non-decreasing, the chain's flattened index
    2 n + a), else None; the index tensors (exterior, interior); the counts (n_X, n_Xc)."""
    mask = R2 > x
    n2 = int(R2.numel()); nX = int(mask.sum()); i0 = n2 - nX
    ordered = bool(mask[i0:].all()) and not bool(mask[:i0].any())
    return (i0 if ordered else None), torch.nonzero(mask).flatten(), torch.nonzero(~mask).flatten(), nX, n2 - nX


def _rows(M, i0, idx, exterior):
    """the exterior (or interior) rows of the columns M: a view by slicing when the rows are ordered, else index_select (a copy)."""
    if i0 is not None:
        return M[i0:] if exterior else M[:i0]
    return M.index_select(0, idx)


def cross_block(Phi, Vneg, i0, idx_X, idx_C):
    """K = Pi_X (C - P_-) Pi_Xc = Phi_X Phi_Xc^dagger - V_X V_Xc^dagger as the n_X x n_Xc matrix from the columns: two GEMMs on the row
    slices (views when the rows are ordered), the second accumulated in place (addmm_); no n2 x n2 or N x N object. Returns (K,
    ||Phi_X Phi_Xc^dagger||_F^2), the second for the check against (a)'s spectrum: ||Pi_X C Pi_Xc||_F^2 = sum_j lambda_j (1 - lambda_j) over
    the eigenvalues of C_Xc."""
    K = _rows(Phi, i0, idx_X, True) @ _rows(Phi, i0, idx_C, False).mH
    fro_C = float(torch.linalg.norm(K)) ** 2
    if Vneg is not None and Vneg.shape[1] and min(K.shape):
        K.addmm_(_rows(Vneg, i0, idx_X, True), _rows(Vneg, i0, idx_C, False).mH, alpha=-1.0)
    return K, fro_C


def block_singular_values(K, route="svd"):
    """the block's singular values, descending: svd (torch.linalg.svdvals, exact to rounding: a QR of the block and the bidiagonal SVD of its
    small side) or gram (eigvalsh of the smaller Gram and a square root: the small singular values to sqrt accuracy, about 1e-8). Returns
    (s, the route's label, an indicative rounding bound on 2 sum_j s_j)."""
    m = int(min(K.shape))
    if m == 0:
        return torch.zeros(0, dtype=RD, device=K.device), "empty block", 0.0
    if route == "gram":
        G = K.mH @ K if K.shape[0] >= K.shape[1] else K @ K.mH
        lam = torch.linalg.eigvalsh(G); del G
        s = torch.sqrt(lam.clamp_min(0.0)).flip(0)
        return s, "gram %d x %d (eigvalsh of the smaller Gram; the small singular values to sqrt accuracy)" % (m, m), 2.0 * m * math.sqrt(2.0 * EPS64 * m * max(float(lam[-1]), 0.0))
    s = torch.linalg.svdvals(K)
    return s, "svd (torch.linalg.svdvals of the %d x %d block)" % (K.shape[0], K.shape[1]), 2.0 * m * 8.0 * EPS64 * float(s[0])


def cross_cut_norms(Phi, Vneg, R2, cuts, route="svd"):
    """(a') per cut (label, R_abs): the singular values sigma_j of the X-by-Xc block of C - P_- (X the modes beyond the cut) and t_X =
    2 sum_j sigma_j = 2 ||Pi_X (C - P_-) Pi_Xc||_1, hreduce's exact bound |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 (EVOLVEDCUT EC1), by
    the block route from the columns: the n_X x n_Xc block by two GEMMs on the row slices (cross_block), its singular values by svdvals (or
    the Gram). Any two orthonormal column sets serve as (Phi, Vneg): the same call gives 2 ||Pi_X (A A^dagger - B B^dagger) Pi_Xc||_1.
    Cost at n = 16384 (n2 = 32768, N_occ = N_neg = 16384, r_s = 64) on an H100: the block at the 48 r_s cut is 26624 x 6144 (2.6 GiB); its
    two GEMMs 2 x 26624 x 6144 x 16384 complex multiply-adds (4e13 flops, about a second at the fp64 tensor-core rate); its singular values
    a QR of the block (2e12 complex multiply-adds) and the 6144 x 6144 bidiagonal SVD (6e11), seconds to tens of seconds in cusolver's gesvd
    (the bidiagonal QR iteration is sequential); the four cuts (n_Xc 2048, 3072, 3840, 6144; n_X 30720 down to 26624) together about 1e14
    flops of GEMMs, under a minute, and under 8 GiB transient (the block and svdvals' workspace) per baseline clock. No n2 x n2 (16 GiB) or
    N_occ x N_occ (4 GiB) object is formed; the measured seconds are stored per cut (cross_seconds)."""
    out = {}
    for lab, x in cuts:
        t0 = time.time()
        i0, idx_X, idx_C, nX, nC = cut_index(R2, x)
        K, fro_C = cross_block(Phi, Vneg, i0, idx_X, idx_C)
        fro_D = float(torch.linalg.norm(K)) ** 2
        s, rlab, rb = block_singular_values(K, route)
        del K
        sv = s.cpu().numpy().astype(float); del s
        out[lab] = {"R_abs": float(x), "n_X": nX, "n_Xc": nC, "block_route": rlab, "rows_ordered": i0 is not None, "sv_block": sv,
                    "t_X": 2.0 * float(sv.sum()), "n_sv": int(sv.size), "sigma_max_block": float(sv[0]) if sv.size else 0.0,
                    "n_sv_above_1e-8": int((sv > 1e-8).sum()), "frob2_block": fro_D, "frob2_sv_minus_block": float((sv * sv).sum()) - fro_D,
                    "frob2_block_C": fro_C, "rounding_bound_t_X": rb, "cross_seconds": time.time() - t0,
                    "label": "t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1 = 2 x the sum of the singular values of the X-by-Xc block of C - P_- (EVOLVEDCUT EC1: |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 for a pure Gaussian C and a projector P_-)"}
    if Phi.device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def cross_cut_check(ent, cc, tol=1e-6):
    """the check |S_X(C) - S_X(P_-)| <= t_X per cut (tol in nats for the entropies' rounding), the consistency of the block with (a)'s spectra
    (||Pi_X C Pi_Xc||_F^2 = sum lambda (1 - lambda) over the eigenvalues of C_Xc; the triangle 2 | ||K_C||_1 - ||K_P||_1 | <= t_X <=
    2 (||K_C||_1 + ||K_P||_1) with ||K_Z||_1 = sum sqrt(lambda (1 - lambda)) from the spectra); the results into ent's cut dicts (t_X, abs_dS,
    slack, bound_ok, ...); returns the labels that FAIL."""
    fails = []
    for lab, c in cc.items():
        d = ent.get(lab)
        if d is None:
            continue
        d.update({k_: v_ for k_, v_ in c.items() if k_ != "sv_block"})
        d["bound_tol"] = float(tol)
        for key, nm in (("eig_Xc", "C"), ("eig_Xc_vac", "P")):
            ev = d.get(key)
            if isinstance(ev, np.ndarray) and ev.size:
                lam = np.clip(ev.astype(float), 0.0, 1.0)
                d["K_%s_trace_norm_from_spectrum" % nm] = float(np.sum(np.sqrt(lam * (1.0 - lam))))
                d["K_%s_frob2_from_spectrum" % nm] = float(np.sum(lam * (1.0 - lam)))
        if d.get("K_C_frob2_from_spectrum") is not None:
            d["frob2_C_block_minus_spectrum"] = c["frob2_block_C"] - d["K_C_frob2_from_spectrum"]
        a_, b_ = d.get("K_C_trace_norm_from_spectrum"), d.get("K_P_trace_norm_from_spectrum")
        if a_ is not None and b_ is not None:
            d["t_X_triangle_lower"] = 2.0 * abs(a_ - b_); d["t_X_triangle_upper"] = 2.0 * (a_ + b_)
            d["t_X_triangle_ok"] = bool(d["t_X_triangle_lower"] - 1e-6 <= c["t_X"] <= d["t_X_triangle_upper"] + 1e-6)
        SX, SXv = d.get("S_X"), d.get("S_X_vac")
        if SX is None or SXv is None:
            d.update({"abs_dS": None, "slack": None, "bound_ok": None})
            continue
        d["abs_dS"] = abs(SX - SXv); d["slack"] = c["t_X"] - d["abs_dS"]; d["bound_ok"] = bool(d["slack"] >= -float(tol))
        if not d["bound_ok"]:
            fails.append(lab)
    return fails


def premise_defects(Phi, Vneg, chunk=1024):
    """||Phi^dagger Phi - 1||_F and ||V^dagger V - 1||_F by column blocks (no N x N object): EC1's premises are C = Phi Phi^dagger a projector
    (the pure Gaussian state) and P_- = V V^dagger a projector."""
    out = {}
    for name, X in (("Phi", Phi), ("Vneg", Vneg)):
        if X is None:
            out[name] = None
            continue
        acc = 0.0; n = int(X.shape[1]); ch = max(1, int(chunk))
        for j0 in range(0, n, ch):
            Gb = X.mH @ X[:, j0: j0 + ch]
            jj = torch.arange(j0, min(n, j0 + ch), device=X.device)
            Gb[jj, jj - j0] -= 1.0
            acc += float(torch.linalg.norm(Gb)) ** 2; del Gb
        out[name] = math.sqrt(acc)
    return out


def sigma_max_estimate(Phi, Vneg, iters=60, block=8, seed=0):
    """the largest singular value of C - P_- by a subspace iteration on (C - P_-)^2 from the columns (block vectors, Rayleigh-Ritz; the Ritz
    value is a lower estimate; the certified cap for a projector difference is 1); returns (estimate, iterations)."""
    n2 = int(Phi.shape[0]); dev = Phi.device
    gen = torch.Generator().manual_seed(int(seed))
    X = torch.complex(torch.randn(n2, block, generator=gen, dtype=torch.float64), torch.randn(n2, block, generator=gen, dtype=torch.float64)).to(device=dev, dtype=CD)
    X, _ = torch.linalg.qr(X)

    def D_apply(Y):
        Z = Phi @ (Phi.mH @ Y)
        if Vneg is not None and Vneg.shape[1]:
            Z = Z - Vneg @ (Vneg.mH @ Y)
        return Z
    val = 0.0; it = 0
    for it in range(1, int(iters) + 1):
        Y = D_apply(D_apply(X))
        T = X.mH @ Y
        new = math.sqrt(max(float(torch.linalg.eigvalsh(0.5 * (T + T.mH)).max()), 0.0))
        X, _ = torch.linalg.qr(Y)
        if abs(new - val) <= 1e-12 * max(1.0, new):
            val = new
            break
        val = new
    return val, it


def full_trace_norm(Phi, Vneg, mode="auto", budget_GB=24.0, sigma_tol=1e-10, chunk=1024, power_iters=60):
    """nu = ||C - P_-||_1 for the projectors C = Phi Phi^dagger (N_occ columns) and P_- = V V^dagger (N_neg): with the principal angles theta_j
    between ran C and ran P_- (cos theta_j the singular values of M = Phi^dagger V) the eigenvalues of C - P_- are +-sin theta_j and
    |N_occ - N_neg| of modulus 1, so nu = 2 sum_j sigma_j + (N_occ - N_neg) over the N_neg singular values sigma_j of (1 - C) V (sin theta_j
    and, for N_neg > N_occ, ones); likewise rank(C - P_-) = 2 #{sigma_j > tol} + (N_occ - N_neg) and ||C - P_-||_2^2 = 2 sum sigma_j^2 +
    (N_occ - N_neg) = N_occ + N_neg - 2 Tr[C P_-]; n_pair = Tr[(1 - P_-) C] is EC2's comparator, not a bound. Routes: gram (eigvalsh of
    1 - M^dagger M, N_neg x N_neg: sigma_j^2 to rounding, the small sigma_j to sqrt accuracy, the rounding bound beside), svd
    (torch.linalg.svdvals of W = V - Phi M, n2 x N_neg: exact to rounding), bound (no N x N object: Tr[C P_-] by column blocks, hence
    ||C - P_-||_2 exactly, and the certified bounds nu <= rank(C - P_-) sigma_max <= (N_occ + N_neg) x 1 and nu <= sqrt(N_occ + N_neg) ||C - P_-||_2,
    with sigma_max's subspace-iteration value beside as a lower estimate), auto (gram where its memory estimate fits the budget and the
    device's free memory, else bound). Cost at n = 16384 on an H100: gram forms M (16384 x 16384, 4 GiB, 9e12 complex multiply-adds) and
    M^dagger M (4e12), one eigvalsh of 16384 x 16384 (tens of seconds), about 16 GiB peak; svd forms W (32768 x 16384, 8 GiB) and runs
    cusolver's gesvd on it (minutes: the bidiagonal QR iteration of a 16384-column matrix) with its workspace beside; bound is one pass of
    column-block GEMMs (9e12) and a few GEMV sweeps. The singular values are returned under "sv" (the caller stores them)."""
    t0 = time.time()
    p = int(Phi.shape[1]); q = int(Vneg.shape[1]) if Vneg is not None else 0; n2 = int(Phi.shape[0]); dev = Phi.device
    Gi = 2.0 ** 30
    est = {"gram_GiB": 16.0 * (p * q + 3.0 * q * q) / Gi, "svd_GiB": 16.0 * (p * q + 3.0 * n2 * q) / Gi, "bound_GiB": 16.0 * p * min(q, int(chunk)) / Gi}
    free = None
    if dev.type == "cuda":
        try:
            free = torch.cuda.mem_get_info(dev)[0] / Gi
        except Exception:                                                       # noqa: BLE001
            free = None
    route = mode
    if mode == "auto":
        route = "gram" if (est["gram_GiB"] <= float(budget_GB) and (free is None or est["gram_GiB"] <= 0.85 * free)) else "bound"
    out = {"N_occ": p, "N_neg": q, "rank_difference": p - q, "mode_requested": mode, "route": route, "estimate_GiB": est, "device_free_GiB_before": free,
           "budget_GiB": float(budget_GB), "sigma_tol": float(sigma_tol), "nu": None, "rank": None, "sigma_max": None}
    if route == "off":
        out["label"] = "the full ||C - P_-||_1 not computed (--cut-norm-full off)"; out["seconds"] = time.time() - t0
        return out
    if route in ("gram", "svd") and q == 0:
        route = "bound"; out["route"] = route; out["note"] = "no vacuum columns: the bound route"
    if route in ("gram", "svd"):
        M = Phi.mH @ Vneg
        trCP = float(torch.linalg.norm(M)) ** 2
        if route == "gram":
            G = M.mH @ M; del M
            G.neg_(); G.diagonal().add_(1.0)
            lam = torch.linalg.eigvalsh(G); del G
            s = torch.sqrt(lam.clamp(0.0, 1.0)).flip(0); del lam
        else:
            W = Vneg.clone(); W.addmm_(Phi, M, alpha=-1.0); del M
            s = torch.linalg.svdvals(W); del W
        sv = s.cpu().numpy().astype(float); del s
        out["sv"] = sv
        out.update({"nu": 2.0 * float(sv.sum()) + (p - q), "rank": int(2 * int((sv > float(sigma_tol)).sum()) + (p - q)),
                    "sigma_max": max(float(sv[0]) if sv.size else 0.0, 1.0 if p != q else 0.0), "frob2_from_sv": 2.0 * float((sv * sv).sum()) + (p - q),
                    "n_sigma_below_tol": int((sv <= float(sigma_tol)).sum()), "sigma_min": float(sv[-1]) if sv.size else None,
                    "rounding_bound_nu": (2.0 * q * math.sqrt(2.0 * EPS64 * q)) if route == "gram" else 2.0 * q * 8.0 * EPS64})
    else:
        trCP = 0.0; ch = max(1, int(chunk))
        for j0 in range(0, q, ch):
            Mb = Phi.mH @ Vneg[:, j0: j0 + ch]
            trCP += float(torch.linalg.norm(Mb)) ** 2; del Mb
        out["sigma_max_estimate"], out["power_iterations"] = sigma_max_estimate(Phi, Vneg, power_iters)
    frob2 = p + q - 2.0 * trCP
    out.update({"tr_CP": trCP, "frob2": frob2, "frob": math.sqrt(max(frob2, 0.0)), "n_pair": p - trCP, "n_hole": q - trCP, "rank_bound": p + q, "sigma_max_cap": 1.0,
                "bound_rank_sigma": float(p + q), "bound_cauchy_schwarz": math.sqrt(p + q) * math.sqrt(max(frob2, 0.0))})
    if out["nu"] is not None:
        out["frob2_sv_minus_direct"] = out["frob2_from_sv"] - frob2
        out["bound_rank_sigma_exact"] = out["rank"] * out["sigma_max"]
        out["bound_cauchy_schwarz_exact_rank"] = math.sqrt(max(out["rank"], 0)) * out["frob"]
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    out["seconds"] = time.time() - t0
    out["label"] = ("nu = ||C - P_-||_1 = 2 sum_j sigma_j((1 - C) V) + (N_occ - N_neg) (the principal angles between ran C and ran P_-; the eigenvalues of C - P_- are +-sin theta_j "
                    "and |N_occ - N_neg| of modulus 1); rank(C - P_-) = 2 #{sigma_j > tol} + (N_occ - N_neg); ||C - P_-||_2^2 = N_occ + N_neg - 2 Tr[C P_-]; n_pair = Tr[(1 - P_-) C] is EC2's "
                    "comparator, not a bound; route %s (gram: the small sigma_j to sqrt accuracy, rounding bound beside; svd: exact to rounding; bound: rank <= N_occ + N_neg and "
                    "sigma_max <= 1 certify nu <= bound_rank_sigma, Cauchy-Schwarz certifies nu <= bound_cauchy_schwarz, sigma_max_estimate is a lower estimate)" % route)
    return out


def cross_cut_readout(Phi, Vneg, R2, cuts, ent, cfg, kappa, trs, sc, arrays):
    """(a') at a clock with the vacuum: the block trace norms and the check per cut (printed; a FAIL line carries the premises' defects), the
    full ||C - P_-||_1 or its bounds; the scalars into ent's cut dicts and sc["trace_norm"], the singular values into arrays
    (sv_block_cut<label>, sv_full)."""
    cc = cross_cut_norms(Phi, Vneg, R2, cuts, cfg.get("cut_block_route", "svd"))
    fails = cross_cut_check(ent, cc, float(cfg.get("cut_bound_tol", 1e-6)))
    for lab, c in cc.items():
        arrays["sv_block_cut%s" % lab] = c.pop("sv_block")
    prem = premise_defects(Phi, Vneg, int(cfg.get("trace_chunk", 1024))) if fails else None
    for lab, _ in cuts:
        d = ent.get(lab) or {}
        if d.get("t_X") is None:
            continue
        head = ("D13 kappa %d T/r_s %.3f cross-cut %s r_s: |S_X(C) - S_X(P_-)| %s <= t_X = 2||Pi_X (C - P_-) Pi_Xc||_1 %.6e (slack %s; %d singular values, %d above 1e-8, "
                "largest %.4e; block %d x %d, %s, rows %s; %.1f s)" % (kappa, trs, lab, _f(d.get("abs_dS"), "%.6e"), d["t_X"], _f(d.get("slack"), "%.3e"), d["n_sv"], d["n_sv_above_1e-8"],
                                                                     d["sigma_max_block"], d["n_X"], d["n_Xc"], d["block_route"], "ordered" if d["rows_ordered"] else "indexed", d["cross_seconds"]))
        if d.get("bound_ok") is None:
            log(head + ": no vacuum entropy at this cut")
        elif d["bound_ok"]:
            log(head + ": ok" + ("" if d.get("t_X_triangle_ok", True) else "; WARNING: outside the triangle bounds from (a)'s spectra [%.6e, %.6e]" % (d["t_X_triangle_lower"], d["t_X_triangle_upper"])))
        else:
            log(head + ": FAIL (|S_X(C) - S_X(P_-)| <= 2||Pi_X (C - P_-) Pi_Xc||_1 holds for a pure Gaussian state C and a projector P_-: the premises' defects ||Phi^dagger Phi - 1||_F %s, "
                       "||V^dagger V - 1||_F %s; the purity gap S_X - S_Xc %.2e; the block's Frobenius check against (a)'s spectrum %s)" % (
                           _f(prem.get("Phi"), "%.2e"), _f(prem.get("Vneg"), "%.2e"), d.get("purity_gap", 0.0), _f(d.get("frob2_C_block_minus_spectrum"), "%.2e")))
    tn = full_trace_norm(Phi, Vneg, cfg.get("cut_norm_full", "auto"), float(cfg.get("cut_norm_full_GB", 24.0)), float(cfg.get("cut_sigma_tol", 1e-10)), int(cfg.get("trace_chunk", 1024)))
    if "sv" in tn:
        arrays["sv_full"] = tn.pop("sv")
    tmax = max([c["t_X"] for c in cc.values()], default=None)
    tn["max_t_X"] = tmax
    tn["nu_ge_max_t_X"] = bool(tn["nu"] + 1e-8 >= tmax) if (tn.get("nu") is not None and tmax is not None) else None
    sc["trace_norm"] = tn; sc["cross_cut_fails"] = fails; sc["premise_defects"] = prem
    if tn.get("route") == "off":
        log("D13 kappa %d T/r_s %.3f: the full ||C - P_-||_1 not computed (--cut-norm-full off); max t_X over the cuts %s" % (kappa, trs, _f(tmax, "%.6e")))
    elif tn.get("nu") is not None:
        log("D13 kappa %d T/r_s %.3f: ||C - P_-||_1 = %.6e (%s route%s; N_occ %d, N_neg %d; rank(C - P_-) %d, largest singular value %.6e, ||C - P_-||_2 %.6e, n_pair = Tr[(1 - P_-) C] %.6e; "
            "max t_X over the cuts %s: %s; the Frobenius check %.2e; %.1f s)" % (kappa, trs, tn["nu"], tn["route"], (", rounding bound %.1e" % tn["rounding_bound_nu"]) if tn.get("rounding_bound_nu") else "",
                                                                                   tn["N_occ"], tn["N_neg"], tn["rank"], tn["sigma_max"], tn["frob"], tn["n_pair"], _f(tmax, "%.6e"),
                                                                                   "consistent (2||Pi_X D Pi_Xc||_1 <= ||D||_1)" if tn["nu_ge_max_t_X"] is not False else "WARNING: below a t_X",
                                                                                   tn.get("frob2_sv_minus_direct", 0.0), tn["seconds"]))
    else:
        log("D13 kappa %d T/r_s %.3f: ||C - P_-||_1 by its bounds (%s route, requested %s): <= rank x sigma_max <= %d (rank(C - P_-) <= N_occ + N_neg = %d, sigma_max <= 1 for a projector difference; the subspace "
            "iteration's sigma_max %s after %s steps, a lower estimate), <= sqrt(N_occ + N_neg) ||C - P_-||_2 = %.6e (Cauchy-Schwarz); ||C - P_-||_2 %.6e, n_pair = Tr[(1 - P_-) C] %.6e (EC2's comparator, not a bound); "
            "max t_X over the cuts %s; the gram route's estimate %.1f GiB against the budget %.1f GiB and the free device memory %s GiB; %.1f s" % (
                kappa, trs, tn["route"], tn["mode_requested"], tn["rank_bound"], tn["rank_bound"], _f(tn.get("sigma_max_estimate"), "%.6e"), _f(tn.get("power_iterations"), "%d"), tn["bound_cauchy_schwarz"], tn["frob"], tn["n_pair"],
                _f(tmax, "%.6e"), tn["estimate_GiB"]["gram_GiB"], tn["budget_GiB"], _f(tn.get("device_free_GiB_before"), "%.1f"), tn["seconds"]))
    return cc, tn


def exterior_entropies(Phi, R2, cuts, Vneg=None):
    """(a) per cut (label, R_abs): S_X and the complement's S_Xc (the purity check), the eigenvalues, the vacuum's where Vneg is given and
    the excess S_X - S_X^vac."""
    out = {}
    for lab, x in cuts:
        mask = R2 > x
        XP = Phi[mask]; XC = Phi[~mask]
        evX, rX = gram_eigs(XP); evC, rC = gram_eigs(XC)
        del XP, XC
        SX = entropy_of_eigs(evX); SC = entropy_of_eigs(evC)
        d = {"R_abs": float(x), "n_modes_X": int(mask.sum()), "n_modes_Xc": int((~mask).sum()), "route_X": rX, "route_Xc": rC, "S_X": SX, "S_Xc": SC,
             "purity_gap": SX - SC, "eig_X": evX.cpu().numpy(), "eig_Xc": evC.cpu().numpy(),
             "eig_X_extremes": [float(evX.min()), float(evX.max())] if evX.numel() else None, "S_X_vac": None, "S_Xc_vac": None, "excess": None, "excess_c": None}
        if Vneg is not None:
            evXv, _ = gram_eigs(Vneg[mask]); evCv, _ = gram_eigs(Vneg[~mask])
            SXv = entropy_of_eigs(evXv); SCv = entropy_of_eigs(evCv)
            d.update({"S_X_vac": SXv, "S_Xc_vac": SCv, "excess": SX - SXv, "excess_c": SC - SCv, "eig_X_vac": evXv.cpu().numpy(), "eig_Xc_vac": evCv.cpu().numpy()})
        out[lab] = d
    return out


# --------------------------------------------------------------------------------------------------------- the Killing vacuum
def hermitian_band(h_csr):
    """the LAPACK upper banded storage of the Hermitian CSR h (a_band[u + i - j, j] = h[i, j], i <= j) and its half-bandwidth u."""
    co = h_csr.tocoo(); n = co.shape[0]
    up = co.col >= co.row
    u = int((co.col[up] - co.row[up]).max()) if up.any() else 0
    ab = np.zeros((u + 1, n), dtype=complex)
    np.add.at(ab, (u + co.row[up] - co.col[up], co.col[up]), co.data[up])
    return ab, u


def tr_min_h0(h_csr):
    """Tr min(h, 0) from the banded operator's eigenvalues alone (LAPACK zhbevd through scipy.linalg.eigvals_banded): the reference of
    E_inv = Tr[h C] - Tr min(h, 0) (LIGHTCHANNEL LC5); returns (the value, info with the counts and the +-e asymmetry sum e)."""
    t0 = time.time()
    ab, u = hermitian_band(h_csr)
    e = np.asarray(sla.eigvals_banded(ab, lower=False, check_finite=False), float)
    neg = e < 0
    return float(e[neg].sum()), {"n_neg": int(neg.sum()), "n_pos": int((e > 0).sum()), "n_zero_1e-12": int((np.abs(e) < 1e-12).sum()), "sum_e": float(e.sum()), "audit": W1S.spectral_audit(e),   # v1c (v6o W1) defect 13: Tr min(h, 0) = -Tr|h|/2
                                 "e_min": float(e.min()), "e_max": float(e.max()), "half_bandwidth": u, "seconds": time.time() - t0,
                                 "route": "Tr min(h, 0) = the sum of the negative eigenvalues of the banded Hermitian h (scipy.linalg.eigvals_banded, LAPACK zhbevd)"}


def _apply_real_csr(A, X, device, dmax=16):
    A = A.tocsr(); n = A.shape[0]
    Y = torch.zeros_like(X)
    for d in range(-dmax, dmax + 1):
        vals = A.diagonal(d)
        if not np.any(vals):
            continue
        D = torch.as_tensor(np.asarray(vals, float), dtype=X.dtype, device=device)
        if d >= 0:
            Y[: n - d] += D[:, None] * X[d:]
        else:
            Y[-d:] += D[:, None] * X[: n + d]
    return Y


def killing_vacuum(h_csr, device, route="dense", cluster_tol=1e-9):
    """P_-(h) = 1(h < 0) as its occupied columns Vneg (device, orthonormal): route dense is torch.linalg.eigh of the dense h (v6m's), route
    pair the massless chain's conjugate-pair structure (killing_vacuum_pair). Returns {e (host, ascending), Vneg, n_neg, E_ref, route, ...}."""
    t0 = time.time()
    if route == "pair":
        return killing_vacuum_pair(h_csr, device, cluster_tol)
    if route == "sector":                                                                                                     # v1c (v6o W1)
        return W1S.killing_vacuum_sector(h_csr, device)                                                                     # v1c (v6o W1)
    e, V = OPS.eigh_csr(h_csr, device)
    neg = e < 0
    Vneg = V[:, neg].clone()
    del V
    if device.type == "cuda":
        torch.cuda.empty_cache()
    e_np = e.cpu().numpy()
    return {"e": e_np, "Vneg": Vneg, "n_neg": int(neg.sum()), "n_zero_1e-12": int((np.abs(e_np) < 1e-12).sum()), "E_ref": float(e_np[e_np < 0].sum()),
            "route": "dense (torch.linalg.eigh of h, v6m's)", "seconds": time.time() - t0}


def killing_vacuum_pair(h_csr, device, cluster_tol=1e-9):
    """the massless chain (m = 0): h = i A with A real antisymmetric (zero diagonal, purely imaginary entries), so the spectrum is +-e and
    the eigenvectors conjugate pairs; the real symmetric eigendecomposition of h^2 = A^T A (half the complex one's work) and, per cluster of
    equal eigenvalues (pairs; larger clusters by their small eigenproblem), the occupied vector of i U_c^T A U_c: for a pair (u_1, u_2) with
    eps = u_2^T A u_1 it is (u_1 - i sgn(eps) u_2)/sqrt 2 at energy -|eps|. Falls back to the dense route when h is not of that form."""
    t0 = time.time()
    h = h_csr.tocsr(); co = h.tocoo(); n2 = h.shape[0]
    amax = float(np.max(np.abs(co.data))) if co.nnz else 0.0
    if co.nnz == 0 or float(np.max(np.abs(co.data.real))) > 1e-12 * amax or float(np.max(np.abs(h.diagonal()))) > 1e-12 * amax:
        out = killing_vacuum(h_csr, device, "dense")
        out["note"] = "the pair route needs h = i A with A real antisymmetric (m = 0): not the case here; the dense route was used"
        return out
    A = sp.csr_matrix((co.data.imag, (co.row, co.col)), shape=h.shape)
    B = (A.T @ A).tocoo()
    Bd = torch.zeros((n2, n2), dtype=RD, device=device)
    Bd.index_put_((torch.as_tensor(B.row, dtype=torch.int64, device=device), torch.as_tensor(B.col, dtype=torch.int64, device=device)),
                  torch.as_tensor(np.asarray(B.data, float), dtype=RD, device=device), accumulate=True)
    lam, U = torch.linalg.eigh(Bd)
    del Bd
    AU = _apply_real_csr(A, U, device)
    lam_np = lam.cpu().numpy(); scale = max(float(lam_np[-1]), 1e-300)
    tol = float(cluster_tol); bounds = None; sizes = None
    for _ in range(8):
        brk = np.nonzero(np.diff(lam_np) > tol * scale)[0] + 1
        bounds = np.concatenate([[0], brk, [n2]]).astype(int); sizes = np.diff(bounds)
        if np.all(sizes % 2 == 0):
            break
        tol *= 10.0
    else:
        raise RuntimeError("the pair route found clusters of odd size at every tolerance up to %g" % tol)
    starts = bounds[:-1]
    pairs = starts[sizes == 2]
    cols = []; ens = []
    if len(pairs):
        i1 = torch.as_tensor(pairs, dtype=torch.int64, device=device); i2 = i1 + 1
        U1 = U[:, i1]; U2 = U[:, i2]
        eps = (U2 * AU[:, i1]).sum(dim=0)
        s = torch.where(eps >= 0, torch.ones_like(eps), -torch.ones_like(eps))
        cols.append((U1.to(CD) - 1j * (U2 * s[None, :]).to(CD)) / math.sqrt(2.0))
        ens.append(-eps.abs())
        del U1, U2
    big = [(int(b0), int(m)) for b0, m in zip(starts, sizes) if m > 2]
    for b0, m in big:
        Uc = U[:, b0: b0 + m]
        Hc = (1j * (Uc.T @ AU[:, b0: b0 + m])).to(CD); Hc = 0.5 * (Hc + Hc.mH)
        ec, zc = torch.linalg.eigh(Hc)
        negc = ec < 0
        cols.append(Uc.to(CD) @ zc[:, negc]); ens.append(ec[negc])
    del U, AU
    Vneg = torch.cat(cols, dim=1); e_neg = torch.cat(ens)
    order = torch.argsort(e_neg); Vneg = Vneg[:, order].contiguous(); e_neg = e_neg[order]
    en = e_neg.cpu().numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"e": np.sort(np.concatenate([en, -en])), "Vneg": Vneg, "n_neg": int(Vneg.shape[1]), "n_zero_1e-12": int((np.abs(en) < 1e-12).sum()), "E_ref": float(en.sum()),
            "route": "pair (h = i A: the real eigendecomposition of h^2 = A^T A; the occupied vector per conjugate pair from the 2 x 2 block of i U^T A U; larger clusters by their small eigenproblem)",
            "cluster_tol_used": tol, "n_pairs": int(len(pairs)), "n_big_clusters": len(big), "big_cluster_max": max([m for _, m in big], default=0), "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------------------ (b) the ledger
def inventories(Bh, Phi, R2, cuts, Vneg=None, chunk=None):
    """E_x^C = Tr[{Pi_x, h}/2 C] = Re Tr[Pi_x h C] beyond each cut; the vacuum's and the excess E_D where Vneg is given."""
    out = {}
    for lab, x in cuts:
        chi = (R2 > x).to(CD)[:, None]
        EC = tr_blocked(Bh, Phi, chi, chunk)
        d = {"R_abs": float(x), "E_C": EC, "E_vac": None, "E_D": None}
        if Vneg is not None:
            Ev = tr_blocked(Bh, Vneg, chi, chunk)
            d["E_vac"] = Ev; d["E_D"] = EC - Ev
        out[lab] = d
    return out


def ledger_windows(rows, records, cut_labels, window_rs, rs, R_cuts=None):
    """(b) the 2 terms per window. rows: [{t_over_rs, two_gE, vacuum_dense, inv: {label: {E_C, E_vac}}}] (the checkpoints); records:
    [{t_over_rs, cut_flow: {label: flow}, cut_jump: {label: jump}}] (per record, the flow over the record and the jump into the next
    record's h at its end). For each baseline row t_0 (the dense vacuum) and each cut: the first checkpoint and the first baseline clock at
    or after t_0 + window_rs as the window's end; the inventory on D at the start, the crossing (the current's integral and the jumps),
    the change of the C inventory and the telescoping residual, the reference's change and the second term on D where the end is a
    baseline clock; the retarded clocks u = t - r_* at the cut (page_ops_v6.chart_retarded_time) beside."""
    out = []
    base = [r for r in rows if r.get("vacuum_dense")]
    for r0 in base:
        t0 = r0["t_over_rs"]
        ends = [r for r in rows if r["t_over_rs"] >= t0 + window_rs - 1e-9]
        ends_b = [r for r in base if r["t_over_rs"] >= t0 + window_rs - 1e-9]
        cands = []
        if ends:
            cands.append((ends[0], "first checkpoint at or after t0 + window"))
        if ends_b and (not ends or ends_b[0] is not ends[0]):
            cands.append((ends_b[0], "first baseline clock at or after t0 + window (the vacuum at both ends)"))
        for r1, kind in cands:
            t1 = r1["t_over_rs"]
            recs = [q for q in records if t0 - 1e-9 <= q["t_over_rs"] < t1 - 1e-9]
            for lab in cut_labels:
                i0 = (r0.get("inv") or {}).get(lab); i1 = (r1.get("inv") or {}).get(lab)
                if i0 is None or i1 is None:
                    continue
                have = bool(recs) and all(lab in (q.get("cut_flow") or {}) and lab in (q.get("cut_jump") or {}) for q in recs)
                cur = sum(q["cut_flow"][lab] for q in recs) if have else None
                jmp = sum(q["cut_jump"][lab] for q in recs) if have else None
                EC0, EC1 = i0["E_C"], i1["E_C"]; Ev0 = i0.get("E_vac"); Ev1 = i1.get("E_vac")
                w = {"t0_over_rs": t0, "t1_over_rs": t1, "window_rs": t1 - t0, "kind": kind, "cut": lab,
                     "inventory_D_start": (EC0 - Ev0) if Ev0 is not None else None, "inventory_C_start": EC0, "inventory_C_end": EC1, "change_C": EC1 - EC0,
                     "crossing_current": cur, "crossing_jumps": jmp, "crossing_total": (cur + jmp) if have else None,
                     "telescoping_residual": (EC1 - EC0 - cur - jmp) if have else None, "n_records": len(recs),
                     "inventory_D_end": (EC1 - Ev1) if Ev1 is not None else None, "vacuum_change": (Ev1 - Ev0) if (Ev0 is not None and Ev1 is not None) else None,
                     "second_term_D": (cur + jmp - (Ev1 - Ev0)) if (have and Ev0 is not None and Ev1 is not None) else None}
                if R_cuts is not None and lab in R_cuts:
                    try:
                        w["u_start_rs"] = OPS.chart_retarded_time(t0 * rs, r0["two_gE"], R_cuts[lab]) / rs
                        w["u_end_rs"] = OPS.chart_retarded_time(t1 * rs, r1["two_gE"], R_cuts[lab]) / rs
                    except (ValueError, ZeroDivisionError):
                        w["u_start_rs"] = w["u_end_rs"] = None
                out.append(w)
            if len(cut_labels) >= 2:
                a_, b_ = cut_labels[0], cut_labels[1]
                ia, ib = (r0.get("inv") or {}).get(a_), (r0.get("inv") or {}).get(b_)
                if ia and ib and ia.get("E_D") is not None and ib.get("E_D") is not None:
                    out.append({"t0_over_rs": t0, "t1_over_rs": t1, "kind": "between the cuts at t0", "cut": "%s-%s" % (a_, b_),
                                "inventory_D_between": ia["E_D"] - ib["E_D"], "inventory_C_between": ia["E_C"] - ib["E_C"]})
    return out


def hard_fluence_windows(rows, window_rs, rs, R_cut_abs):
    """(c) the hard fluence per window as the hard outgoing inventory's change beyond R_cut between the window's ends (the block-ground
    reference at every row; the P_- reference where both ends are baseline clocks); rows: [{t_over_rs, two_gE, vacuum_dense, shells: totals}]."""
    out = []
    rows = [r for r in rows if r.get("shells")]
    for r0 in rows:
        t0 = r0["t_over_rs"]
        ends = [r for r in rows if r["t_over_rs"] >= t0 + window_rs - 1e-9]
        if not ends:
            continue
        r1 = ends[0]; T0, T1 = r0["shells"], r1["shells"]
        w = {"t0_over_rs": t0, "t1_over_rs": r1["t_over_rs"], "window_rs": r1["t_over_rs"] - t0}
        for key in ("hard_E_out", "hard_E_in", "soft_E_out", "soft_E_in", "above_E_out", "above_E_in", "hard_N_out", "hard_N_in"):
            w["delta_" + key] = T1[key] - T0[key]
        if r0.get("vacuum_dense") and r1.get("vacuum_dense") and T0.get("hard_E_out_P") is not None and T1.get("hard_E_out_P") is not None:
            for key in ("hard_E_out_P", "hard_E_in_P", "soft_E_out_P", "soft_E_in_P"):
                w["delta_" + key] = T1[key] - T0[key]
        try:
            w["u_start_rs"] = OPS.chart_retarded_time(t0 * rs, r0["two_gE"], R_cut_abs) / rs
            w["u_end_rs"] = OPS.chart_retarded_time(r1["t_over_rs"] * rs, r1["two_gE"], R_cut_abs) / rs
        except (ValueError, ZeroDivisionError):
            pass
        out.append(w)
    return out


# ------------------------------------------------------------------------------------------------------------ (c) the shells
def shell_edges(TH_ref, bin_TH, bin_max_TH, band_top):
    fine = np.arange(0.0, bin_max_TH + 0.5 * bin_TH, bin_TH) * TH_ref
    coarse = []
    x = bin_max_TH * TH_ref
    while x * 2.0 < band_top:
        x *= 2.0; coarse.append(x)
    return np.concatenate([fine, np.array(coarse), [band_top, np.inf]])


def killing_shells(h_csr, R_sites, R_cut_abs, device, TH_ref, bin_TH=0.5, bin_max_TH=40.0, band_top=1.0, hard_TH=25.0):
    """(c) the exterior block beyond R_cut: its dense eigendecomposition (e, U), the bins in |e|/T_H^ref per sign of e and, within each
    bin, the outgoing (v > 0) and incoming subspaces of the radial velocity i[h_X, R] with their energy expectations."""
    t0 = time.time()
    R2 = np.repeat(np.asarray(R_sites, float), 2)
    idx = np.nonzero(R2 > R_cut_abs)[0]
    hX = h_csr[idx][:, idx].tocsr()
    e, U = OPS.eigh_csr(hX, device)
    RX = R2[idx]
    co = hX.tocoo()
    VXd = OPS.dense_from_csr(sp.csr_matrix((1j * co.data * (RX[co.col] - RX[co.row]), (co.row, co.col)), shape=co.shape), device)
    e_np = e.cpu().numpy()
    edges = shell_edges(TH_ref, bin_TH, bin_max_TH, band_top)
    th = hard_TH * TH_ref
    if np.min(np.abs(edges[np.isfinite(edges)] - th)) > 1e-9 * max(th, 1e-300):
        log("WARNING: the hard threshold %.4e is not a bin edge (bin width %.4e): the hard class starts at the first edge at or above it" % (th, bin_TH * TH_ref))
    bins = []
    for sgn in (1, -1):
        sel = (e_np > 0) if sgn > 0 else (e_np < 0)
        for b in range(len(edges) - 1):
            j = np.nonzero(sel & (np.abs(e_np) >= edges[b]) & (np.abs(e_np) < edges[b + 1]))[0]
            if len(j) == 0:
                continue
            jt = torch.as_tensor(j, dtype=torch.int64, device=device)
            Ub = U[:, jt]
            Wb = Ub.mH @ (VXd @ Ub)
            wv, wu = torch.linalg.eigh(0.5 * (Wb + Wb.mH))
            pos = wv > 0
            eb = e[jt]
            cls = "above" if edges[b] >= band_top - 1e-12 else ("hard" if edges[b] >= th - 1e-12 else "soft")
            bins.append({"sign": sgn, "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(len(j)), "class": cls,
                         "O": Ub @ wu[:, pos], "I": Ub @ wu[:, ~pos], "eO": wu[:, pos].abs().pow(2).T @ eb, "eI": wu[:, ~pos].abs().pow(2).T @ eb,
                         "vO": wv[pos], "vI": wv[~pos]})
            del Ub, Wb
    del VXd
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"idx": torch.as_tensor(idx, dtype=torch.int64, device=device), "e": e_np, "U": U, "bins": bins, "TH_ref": TH_ref, "R_cut": R_cut_abs, "nX": int(len(idx)),
            "edges": edges, "band_top": band_top, "hard_TH": hard_TH, "e_extremes": [float(e_np.min()), float(e_np.max())], "n_above_band": int((np.abs(e_np) > band_top).sum()),
            "seconds": time.time() - t0}


def free_shells(sh):
    if sh is None:
        return
    for bn in sh["bins"]:
        for key in ("O", "I", "eO", "eI", "vO", "vI"):
            bn[key] = None
    sh["U"] = None


def shell_content(sh, Phi, Vneg=None):
    """(c) per exterior eigenvector <u_j|C|u_j> and per bin (sign, |e| bin) the number and energy content of C minus the block's own
    ground on the outgoing and incoming subspaces (and minus P_-(h(t)) where Vneg is given); the totals per class (soft, hard, above the
    band top = bound content)."""
    idx = sh["idx"]
    PX = Phi[idx]
    A = sh["U"].mH @ PX
    nj = A.abs().pow(2).sum(dim=1).cpu().numpy(); del A
    out = {"n_j": nj, "excess_j_block": nj - (sh["e"] < 0).astype(float)}
    VXn = None
    if Vneg is not None:
        VXn = Vneg[idx]
        Av = sh["U"].mH @ VXn
        nv = Av.abs().pow(2).sum(dim=1).cpu().numpy(); del Av
        out["nvac_j"] = nv; out["excess_j_Pminus"] = nj - nv
    nb = len(sh["bins"])
    keys = ("lo", "hi", "sign", "n", "n_out", "n_in", "N_out", "N_in", "E_out", "E_in", "N_out_P", "N_in_P", "E_out_P", "E_in_P")
    arr = {k: np.zeros(nb) for k in keys}
    cls = []
    for i, bn in enumerate(sh["bins"]):
        refb = 1.0 if bn["sign"] < 0 else 0.0
        arr["lo"][i] = bn["lo"]; arr["hi"][i] = bn["hi"]; arr["sign"][i] = bn["sign"]; arr["n"][i] = bn["n"]; cls.append(bn["class"])
        for side, key in (("O", "out"), ("I", "in")):
            M = bn[side]
            if M.shape[1] == 0:
                continue
            a_ = M.mH @ PX
            nn = a_.abs().pow(2).sum(dim=1); del a_
            ex = nn - refb
            arr["n_" + key][i] = M.shape[1]
            arr["N_" + key][i] = float(ex.sum()); arr["E_" + key][i] = float((bn["e" + side] * ex).sum())
            if VXn is not None:
                av = M.mH @ VXn
                exv = nn - av.abs().pow(2).sum(dim=1); del av
                arr["N_%s_P" % key][i] = float(exv.sum()); arr["E_%s_P" % key][i] = float((bn["e" + side] * exv).sum())
    cls = np.array(cls)
    tot = {}
    for c_ in ("soft", "hard", "above"):
        m = cls == c_
        for key in ("N_out", "N_in", "E_out", "E_in"):
            tot["%s_%s" % (c_, key)] = float(arr[key][m].sum())
            tot["%s_%s_P" % (c_, key)] = float(arr[key + "_P"][m].sum()) if VXn is not None else None
    tot["above_N_total"] = tot["above_N_out"] + tot["above_N_in"]
    tot["label"] = ("content of C minus the block's own ground 1(e < 0) (and minus P_-(h(t)), the _P keys, at the baseline clocks) on the outgoing (v > 0) and "
                    "incoming subspaces of the exterior block's Killing shells; hard: |e| from %g T_H^ref to the band top; above: |e| above the band top, bound content" % sh["hard_TH"])
    out["bins"] = arr; out["class"] = cls; out["totals"] = tot
    del PX, VXn
    return out


# ------------------------------------------------------------------------------------------------------------ (d) the probes
def probe_vectors(h_csr, R_sites, x_abs, device, n_max=16):
    """J_x = (i/2)[h^2, Pi_x] (page_ops_v6.current_out_csr) on its support (at most 16 modes on the range-2 chain): the eigenpairs
    (w_a, j_a) with j_a nonzero, W (n2 x r) on the device."""
    J = OPS.current_out_csr(h_csr, R_sites, x_abs)
    co = J.tocoo()
    sup = np.unique(np.concatenate([co.row, co.col]))
    Jb = J[sup][:, sup].toarray()
    jv_, U = np.linalg.eigh(0.5 * (Jb + Jb.conj().T))
    keep = np.abs(jv_) > 1e-14 * max(float(np.max(np.abs(jv_))), 1e-300)
    order = np.argsort(-np.abs(jv_[keep]))[:n_max]
    jk = jv_[keep][order]; Uk = U[:, keep][:, order]
    W = np.zeros((J.shape[0], len(jk)), dtype=complex); W[sup] = Uk
    return (torch.as_tensor(W, dtype=CD, device=device), torch.as_tensor(jk, dtype=RD, device=device),
            {"rank": int(keep.sum()), "support_modes": int(len(sup)), "trace_norm": float(np.sum(np.abs(jk))), "op_norm": float(np.max(np.abs(jk))),
             "R_abs": float(x_abs), "label": "J_x = (i/2)[h^2, Pi_x] = sum_a j_a w_a w_a^dagger; rank at most 16 (review_PAGE_v3b)"})


def probe_split(W, j, Phi_ref, Vneg_ref, S_ref=None):
    """at the reference clock: L = sum_a j_a <w_a|C|w_a>, L_sea = sum_a j_a <w_a|P_-|w_a>, L_old = L - L_sea, L_src on the source modes;
    the rank-16 bound sum|j_a| and the pointwise bound sum |j_a| |<w_a|D|w_a>|."""
    nC = (Phi_ref.mH @ W).abs().pow(2).sum(dim=0)
    nV = (Vneg_ref.mH @ W).abs().pow(2).sum(dim=0)
    L_total = float((j * nC).sum()); L_sea = float((j * nV).sum())
    out = {"L_total_backpropagated": L_total, "L_old": L_total - L_sea, "L_sea": L_sea, "L_src": None,
           "bound_rank16_trace_norm": float(j.abs().sum()), "bound_old_pointwise": float((j.abs() * (nC - nV).abs()).sum()),
           "probe_j": j.cpu().numpy(), "probe_nC": nC.cpu().numpy(), "probe_nV": nV.cpu().numpy(), "probe_nD": (nC - nV).cpu().numpy()}
    if S_ref is not None and S_ref.shape[1]:
        nS = (S_ref.mH @ W).abs().pow(2).sum(dim=0)
        out["L_src"] = float((j * nS).sum()); out["probe_nS"] = nS.cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------------ (e) the source modes
def source_modes(Phi, R2, r_region_abs, h_csr, device, Vneg=None, weight_min=0.99, n_max=128):
    """the eigenvectors of C restricted to the interior (R < r_region; the same nonzero spectrum as Phi^dagger Pi_in Phi) with eigenvalue
    at least weight_min, as unit modes s = Phi u in ran C: their weight inside, occupation <s|C|s>, the vacuum's <s|P_-|s>, Killing energy
    and spread."""
    mask = R2 < r_region_abs
    X = Phi[mask]
    Cin = X @ X.mH; Cin = 0.5 * (Cin + Cin.mH)
    lam, Y = torch.linalg.eigh(Cin)
    order = torch.argsort(lam, descending=True)
    keep = order[lam[order] >= weight_min][:n_max]
    out = {"n_in_modes": int(mask.sum()), "n_candidates_above": int((lam >= weight_min).sum()), "region_R_abs": float(r_region_abs), "weight_min": weight_min,
           "lam_all": lam.cpu().numpy()}
    if keep.numel() == 0:
        out.update({"S": torch.zeros((Phi.shape[0], 0), dtype=CD, device=device), "lam": np.zeros(0), "w_in": np.zeros(0), "occ_C": np.zeros(0),
                    "occ_vac": None, "energy": np.zeros(0), "energy_spread": np.zeros(0)})
        return out
    lam_k = lam[keep]
    u = (X.mH @ Y[:, keep]) / torch.sqrt(lam_k).to(CD)[None, :]
    S = Phi @ u
    S = S / torch.linalg.norm(S, dim=0).to(CD)[None, :]
    w_in = S[mask].abs().pow(2).sum(dim=0)
    occ = (Phi.mH @ S).abs().pow(2).sum(dim=0)
    hS = OPS.Banded(h_csr, device).apply(S)
    en = (S.conj() * hS).real.sum(dim=0); en2 = hS.abs().pow(2).sum(dim=0)
    out.update({"S": S, "lam": lam_k.cpu().numpy(), "w_in": w_in.cpu().numpy(), "occ_C": occ.cpu().numpy(),
                "occ_vac": (Vneg.mH @ S).abs().pow(2).sum(dim=0).cpu().numpy() if Vneg is not None else None,
                "energy": en.cpu().numpy(), "energy_spread": torch.sqrt((en2 - en * en).clamp_min(0.0)).cpu().numpy()})
    del X, Cin, hS
    return out


def source_transfer(S, Phi, R2, radii):
    """per propagated source mode and radius (label, R_abs): tau = ||Pi_{>R} s||^2 (the same-packet transfer), w = ||Phi^dagger Pi s||^2
    (the state's content on the exterior part), w/tau (the inferred occupation) and its excess over <s|C|s>."""
    out = {"n_modes": int(S.shape[1])}
    if S.shape[1] == 0:
        return out
    occ = (Phi.mH @ S).abs().pow(2).sum(dim=0)
    out["occ_C_now"] = occ.cpu().numpy()
    for lab, r in radii:
        Se = (R2 > r).to(CD)[:, None] * S
        tau = Se.abs().pow(2).sum(dim=0)
        w = (Phi.mH @ Se).abs().pow(2).sum(dim=0)
        ratio = torch.where(tau > 1e-12, w / tau.clamp_min(1e-300), torch.full_like(tau, float("nan")))
        out["tau_" + lab] = tau.cpu().numpy(); out["w_" + lab] = w.cpu().numpy(); out["occ_inferred_" + lab] = ratio.cpu().numpy()
        out["excess_" + lab] = (ratio - occ).cpu().numpy()
        out["tau_sum_" + lab] = float(tau.sum()); out["tau_max_" + lab] = float(tau.max())
        fin = torch.isfinite(ratio)
        out["occ_inferred_mean_" + lab] = float(ratio[fin].mean()) if bool(fin.any()) else None
        del Se
    return out


# ------------------------------------------------------------------------------------------------- (f) the join and the return
def join_departure(N_e, v_e, R_e, E, g, lo, hi):
    """the largest |N - 1| and |v - v_PG|/|v_PG| of the record's fields over lo <= R <= hi against the vacuum exterior at E."""
    R_e = np.asarray(R_e, float); sel = (R_e >= lo) & (R_e <= hi)
    if not sel.any():
        return None, None
    vpg = -np.sqrt(2.0 * g * E / R_e[sel])
    return float(np.max(np.abs(np.asarray(N_e)[sel] - 1.0))), float(np.max(np.abs(np.asarray(v_e)[sel] - vpg) / np.abs(vpg)))


def light_time(N, v, R, a, b, outgoing):
    """the continuum light travel time between radii a < b (trapezoid over the sites in [a, b]): outgoing speed N - |v|, ingoing N + |v|;
    None where the speed is not positive somewhere in the interval."""
    R = np.asarray(R, float); N = np.asarray(N, float); v = np.asarray(v, float)
    sel = (R >= a) & (R <= b)
    if sel.sum() < 2:
        return 0.0
    c = N[sel] - np.abs(v[sel]) if outgoing else N[sel] + np.abs(v[sel])
    if np.any(c <= 0):
        return None
    return float(np.sum(0.5 * (1.0 / c[1:] + 1.0 / c[:-1]) * np.diff(R[sel])))


def first_return(N, v, R, x_abs, t_now, t_start, rs, t_form=None, r_form=None):
    R_end = float(np.asarray(R, float)[-1])
    t_out = light_time(N, v, R, x_abs, R_end, True); t_in = light_time(N, v, R, x_abs, R_end, False)
    d = {"R_cut_abs": float(x_abs), "R_end_abs": R_end, "t_out_rs": t_out / rs if t_out is not None else None, "t_in_rs": t_in / rs if t_in is not None else None,
         "first_return_from_start_rs": (t_start + t_out + t_in) / rs if (t_out is not None and t_in is not None) else None, "first_return_formation_rs": None}
    if t_form is not None and r_form is not None and t_in is not None:
        to_f = light_time(N, v, R, r_form, R_end, True)
        d["first_return_formation_rs"] = (t_form + to_f + t_in) / rs if to_f is not None else None
    cands = [z for z in (d["first_return_from_start_rs"], d["first_return_formation_rs"]) if z is not None]
    d["first_return_rs"] = min(cands) if cands else None
    d["before_first_return"] = (t_now / rs < d["first_return_rs"]) if d["first_return_rs"] is not None else None
    d["label"] = ("the continuum light round trip in the record's fields (outgoing N - |v|, ingoing N + |v|): from the cut at the run's start, and the formation's light "
                  "from the root to the chain end and back to the cut; the earlier of the two; an estimate, not a measured arrival")
    return d


# ------------------------------------------------------------------------------------------------------ (g) the enclosed source
def killing_density_sites(Bh, Phi, Vneg, chunk=None):
    """the local Killing energy density of D = C - P_- per site, Re Tr[Pi_n h D] (its sum is Tr[h D] = E_inv): N rho_L + v J_L per site."""
    ch = int(chunk or TRACE_CHUNK[0])
    rho = torch.zeros(Phi.shape[0], dtype=RD, device=Phi.device)
    for X, sg in ((Phi, 1.0), (Vneg, -1.0)):
        for j0 in range(0, X.shape[1], ch):
            Xb = X[:, j0: j0 + ch]
            Y = Bh.apply(Xb)
            rho += sg * (Xb.conj() * Y).real.sum(dim=1)
            del Y
    r = rho.cpu().numpy()
    return r[0::2] + r[1::2]


def enclosed_source(rho_sites, R, N, v, g, r_root, eps_H=None, J_H=None, R_dens=None):
    """TWOFIELD CF3's source-only root coefficient on the chain: I_L(R) = sum over sites within R of the Killing energy density of D (the
    channels' weights applied by the caller), kappa_R = (N + v)'(R) at the outer root (the applied fields' slope), delta R_L = G I_L/(R kappa_R)
    = 2 G I_L/(N D_R); D_R = 1 - 8 pi G R^2 (rho_H - J_H) from the stored heavy densities (per unit volume) beside it as the convention check."""
    if r_root is None:
        return None
    R = np.asarray(R, float); N = np.asarray(N, float); v = np.asarray(v, float)
    I_L = float(np.sum(np.asarray(rho_sites, float)[R <= r_root]))
    f = N + v
    i = int(np.clip(np.searchsorted(R, r_root, side="right") - 1, 0, len(R) - 2))
    kap = float((f[i + 1] - f[i]) / (R[i + 1] - R[i]))
    N_root = float(np.interp(r_root, R, N))
    out = {"R_root": float(r_root), "I_L": I_L, "kappa_R_slope": kap, "N_root": N_root, "M_root": r_root / (2.0 * g), "I_L_over_M_root": I_L / (r_root / (2.0 * g)),
           "delta_R_L": (g * I_L / (r_root * kap)) if kap != 0 else None, "delta_R_L_over_R": (g * I_L / (r_root * r_root * kap)) if kap != 0 else None,
           "label": "I_L the Killing energy of D = C - P_- enclosed within the outer root; kappa_R the applied (N + v)'s slope at the root; delta R_L = G I_L/(R kappa_R) (TWOFIELD CF3's source-only coefficient; the heavy response at the same order is not included)"}
    if eps_H is not None and J_H is not None:
        Rd = np.asarray(R_dens if R_dens is not None else R[: len(eps_H)], float)
        rhoH = float(np.interp(r_root, Rd, np.asarray(eps_H, float))); JH = float(np.interp(r_root, Rd, np.asarray(J_H, float)))
        D_R = 1.0 - 8.0 * math.pi * g * r_root * r_root * (rhoH - JH)
        out.update({"rho_H_root": rhoH, "J_H_root": JH, "D_R_from_densities": D_R, "kappa_R_from_densities": N_root * D_R / (2.0 * r_root),
                    "kappa_ratio_densities_over_slope": (N_root * D_R / (2.0 * r_root)) / kap if kap != 0 else None,
                    "delta_R_L_from_D_R": (2.0 * g * I_L / (N_root * D_R)) if D_R != 0 else None,
                    "density_convention_check": "kappa from the stored densities (per unit volume assumed) against kappa from the applied slope; a ratio far from 1 marks a convention or a row defect, not a result"})
    return out


# ----------------------------------------------------------------------------------------------- the collar flow's windowed split
def collar_window_rows(R, R_collar_abs, L):
    R = np.asarray(R, float); n = len(R)
    ic = int(np.searchsorted(R, R_collar_abs, side="right"))
    hL = L // 2; i0 = ic - hL; i1 = i0 + L
    return ic, max(0, i0), min(n, i1), i0


def collar_window_split(Phi, Pm_block, N, v, R, R_collar_abs, kappa, L, omega_th):
    """page_history_v6m.collar_split's windowed signed split of the collar flux (a sine window of L sites centred at the first site beyond
    R_collar, the local spinors of D(q) = sin q sigma_x + (kappa/R_c) sigma_y, the bands omega_+-(q) and group velocities v_+-(q), the
    deviation C - P_- on the window read as n_+-(q)), with the flux split by |omega| below (soft) and at or above (hard) omega_th; a local
    reading, not an identity. Pm_block: the vacuum's block on the window's modes (host complex)."""
    Rf = np.asarray(R, float); Nf = np.asarray(N, float); vf = np.asarray(v, float)
    ic, lo, hi, i0 = collar_window_rows(Rf, R_collar_abs, L)
    Rc, Nc, vc = float(Rf[ic]), float(Nf[ic]), float(vf[ic])
    gw = np.sin(math.pi * (np.arange(L) + 0.5) / L)[lo - i0: hi - i0]
    rows = torch.arange(2 * lo, 2 * hi, device=Phi.device)
    PW = Phi.index_select(0, rows)
    CW = (PW @ PW.mH).cpu().numpy().astype(np.complex128); del PW
    Dl = CW - np.asarray(Pm_block, np.complex128)
    Dl = 0.5 * (Dl + Dl.conj().T)
    M = 4 * L; q = 2.0 * math.pi * np.fft.fftfreq(M); nl = hi - lo
    Fm = np.fft.fft(np.eye(M, nl) * gw[None, :], axis=0, norm="ortho")
    G = {}
    for aa in (0, 1):
        for bb in (0, 1):
            G[(aa, bb)] = np.sum((Fm @ Dl[aa::2, bb::2]) * Fm.conj(), axis=1)
    s_ = np.sin(q); c_ = np.cos(q); kr = kappa / Rc
    E = np.sqrt(s_ ** 2 + kr ** 2); nrm = np.sqrt(2.0 * E * E)
    u0 = E / nrm; u1 = (s_ + 1j * kr) / nrm
    n_tot = (G[(0, 0)] + G[(1, 1)]).real
    n_p = (np.conj(u0) * u0 * G[(0, 0)] + np.conj(u0) * u1 * G[(0, 1)] + np.conj(u1) * u0 * G[(1, 0)] + np.conj(u1) * u1 * G[(1, 1)]).real
    n_m = n_tot - n_p
    om_p = Nc * E + 0.5 * vc * np.sin(2.0 * q); om_m = -Nc * E + 0.5 * vc * np.sin(2.0 * q)
    vp = Nc * s_ * c_ / E + vc * np.cos(2.0 * q); vm = -Nc * s_ * c_ / E + vc * np.cos(2.0 * q)
    W2 = float(np.sum(gw ** 2))
    out = {"L": int(L), "centre_site": ic + 1, "R_c": Rc, "N_c": Nc, "v_c": vc, "sum_w2": W2, "omega_th": float(omega_th), "sites_valid": int(nl)}
    for cls in ("soft", "hard"):
        for side in ("out", "in"):
            Ef = 0.0; nf = 0.0
            for om, vv, nn in ((om_p, vp, n_p), (om_m, vm, n_m)):
                m = (vv > 0) if side == "out" else (vv <= 0)
                m = m & ((np.abs(om) < omega_th) if cls == "soft" else (np.abs(om) >= omega_th))
                Ef += float(np.sum((om * vv * nn)[m])); nf += float(np.sum((vv * nn)[m]))
            out["E_flux_%s_%s" % (side, cls)] = Ef / W2; out["n_flux_%s_%s" % (side, cls)] = nf / W2
    for side in ("out", "in"):
        out["E_flux_" + side] = out["E_flux_%s_soft" % side] + out["E_flux_%s_hard" % side]
        out["n_flux_" + side] = out["n_flux_%s_soft" % side] + out["n_flux_%s_hard" % side]
    out["E_flux_net"] = out["E_flux_out"] + out["E_flux_in"]
    out["E_density"] = (float(np.sum(om_p * n_p)) + float(np.sum(om_m * n_m))) / W2
    out["n_density_plus"] = float(np.sum(n_p)) / W2; out["n_density_minus"] = float(np.sum(n_m)) / W2
    out["label"] = "v6k's windowed signed split of the collar flux at the first site beyond R_collar, soft below and hard at or above omega_th = 25 T_H^ref; a local reading"
    return out


# ----------------------------------------------------------------------------------------------------------------- the store
def split_arrays(d, arrays, path=()):
    """numpy arrays in a (nested) dict moved to arrays under 'path_key' names; the dict keeps {'npz': name, 'shape': ...}."""
    for key in list(d.keys()):
        val = d[key]
        if isinstance(val, np.ndarray):
            name = "_".join(list(path) + [str(key)])
            arrays[name] = val
            d[key] = {"npz": name, "shape": list(val.shape)}
        elif isinstance(val, dict):
            split_arrays(val, arrays, tuple(path) + (str(key),))
        elif isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            for i_, x in enumerate(val):
                split_arrays(x, arrays, tuple(path) + (str(key), str(i_)))
    return d


class Store:
    """<prefix>.json (every scalar per clock, channel, window and probe) and <prefix>.npz (every raw array); resumable."""

    def __init__(self, prefix, settings=None, label=None, code=CODE):
        self.prefix = prefix; self.pj = prefix + ".json"; self.pz = prefix + ".npz"
        self.J = {"code": code, "settings": settings, "label": label, "clocks": {}, "probes": [], "windows": {}, "hard_windows": {}, "sources": {}, "notes": []}
        self.Z = {}
        if os.path.exists(self.pj):
            try:
                self.J.update(json.load(open(self.pj)))
            except Exception as ex:                                        # noqa: BLE001
                log("the store's json %s could not be read (%s): started afresh" % (self.pj, ex))
        if os.path.exists(self.pz):
            try:
                with np.load(self.pz, allow_pickle=False) as z:
                    self.Z = {k: z[k] for k in z.files}
            except Exception as ex:                                        # noqa: BLE001
                log("the store's npz %s could not be read (%s): started afresh" % (self.pz, ex))
        self.J["code"] = code
        if settings is not None:
            self.J["settings"] = settings
        if label is not None:
            self.J["label"] = label

    @staticmethod
    def ckey(trs):
        return "%010.4f" % float(trs)

    def has_clock(self, trs, kappa):
        c = self.J["clocks"].get(self.ckey(trs))
        return bool(c and str(kappa) in (c.get("per_channel") or {}))

    def put_clock(self, trs, kappa, scalars, arrays=None, common=None):
        key = self.ckey(trs)
        c = self.J["clocks"].setdefault(key, {"t_over_rs": float(trs), "per_channel": {}})
        if common:
            c.update(common)
        c["per_channel"][str(kappa)] = scalars
        for name, arr in (arrays or {}).items():
            self.Z["T%s_k%d_%s" % (key, int(kappa), name)] = np.asarray(arr)

    def put_probe(self, rec):
        self.J["probes"] = [p for p in self.J["probes"] if not (p["kappa"] == rec["kappa"] and abs(p["t_over_rs"] - rec["t_over_rs"]) < 1e-6 and p["cut"] == rec["cut"])] + [rec]

    def put_sources(self, kappa, k_ref, d, arrays=None):
        self.J["sources"].setdefault(str(kappa), {})[str(k_ref)] = d
        for name, arr in (arrays or {}).items():
            self.Z["src_k%d_ref%d_%s" % (int(kappa), int(k_ref), name)] = np.asarray(arr)

    def flush(self):
        tmp = self.pj + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.J, f, default=_jdefault, indent=1)
        os.replace(tmp, self.pj)
        if self.Z:
            tmpz = self.pz + ".tmp.npz"
            np.savez(tmpz, **self.Z)
            os.replace(tmpz, self.pz)


def save_columns(path, X, meta, key="Phi", fmt="pt"):
    """the occupied (or vacuum) columns to a named file: pt (torch.save of {key: cpu complex128 tensor, meta}) or npz."""
    tmp = path + ".tmp"
    if fmt == "npz":
        np.savez(tmp, **{key: X.cpu().numpy()}, **{k_: np.asarray(v_) for k_, v_ in meta.items() if not isinstance(v_, (dict, list))})
        os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)
    else:
        torch.save(dict(meta, **{key: X.cpu()}), tmp)
        os.replace(tmp, path)
    with open(path + ".json", "w") as f:
        json.dump(dict(meta, key=key, shape=list(X.shape), dtype=str(X.dtype), format=fmt), f, default=_jdefault, indent=1)


def load_columns(path, device, key="Phi"):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return torch.as_tensor(z[key], dtype=CD, device=device)
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st[key].to(device)


def memory_figure(nsites=8192, n_occ=None, chunk=2048, R_cut_rs=24.0, rs=64.0, source_max=128, cut_max_rs=48.0):
    """the device memory of one channel at n sites (GiB, estimated from the shapes; the measured peaks are in the rows)."""
    n2 = 2 * nsites; n0 = n_occ or nsites
    G = lambda b: b / 2.0 ** 30   # noqa: E731
    cols = 16 * n2 * n0; cheb = 4 * 16 * n2 * chunk
    dense = 16 * n2 * n2; nX = 2 * (nsites - int(R_cut_rs * rs)); shells = 16 * nX * nX; gram = 16 * n0 * n0
    d = {"columns_GiB": G(cols), "chebyshev_workspace_GiB": G(cheb), "vacuum_dense_h_GiB": G(dense), "vacuum_V_GiB": G(dense), "vacuum_eigh_workspace_GiB": G(dense),
         "Vneg_GiB": G(cols), "shells_U_GiB": G(shells), "shells_bins_GiB": G(shells), "shells_velocity_transient_GiB": G(shells), "gram_GiB": G(gram),
         "exterior_projection_transient_GiB": G(16 * nX * n0), "source_modes_GiB": G(16 * n2 * 3 * source_max), "references_transient_GiB": 2 * G(cols)}
    nXc = 2 * min(nsites, int(cut_max_rs * rs)); blk = 16 * (n2 - nXc) * nXc                                                  # v1b (a'): the largest cross-cut block
    d.update({"cross_cut_block_GiB": G(blk), "cross_cut_svd_peak_GiB": 3 * G(blk), "trace_norm_gram_peak_GiB": 4 * G(gram),
              "trace_norm_svd_peak_GiB": G(cols) + G(gram) + 2 * G(cols)})
    d["cross_cut_checkpoint_peak_GiB"] = d["columns_GiB"] + d["Vneg_GiB"] + 2 * d["shells_U_GiB"] + max(d["cross_cut_svd_peak_GiB"], d["trace_norm_gram_peak_GiB"])
    d["steady_propagation_GiB"] = d["columns_GiB"] + d["chebyshev_workspace_GiB"] + d["source_modes_GiB"]
    d["baseline_checkpoint_peak_GiB"] = d["columns_GiB"] + 3 * G(dense) + d["source_modes_GiB"] + 2 * d["shells_U_GiB"]
    d["shells_rebuild_peak_GiB"] = d["columns_GiB"] + d["Vneg_GiB"] + 3 * G(shells) + d["source_modes_GiB"] + 2 * G(dense)
    d["probe_clock_peak_GiB"] = d["columns_GiB"] + d["references_transient_GiB"] + 2 * d["shells_U_GiB"] + d["source_modes_GiB"]
    d["pair_route_vacuum_peak_GiB"] = d["columns_GiB"] + 4 * G(8 * n2 * n2) + G(cols) + 2 * d["shells_U_GiB"]
    d["label"] = ("one channel at n = %d sites (n2 = %d, N_occ = %d): the columns, the Chebyshev workspace (chunk %d), the dense vacuum (h, V, the eigh workspace), "
                  "the exterior block beyond R_cut (%d modes: U and the bins' bases resident, i[h_X, R] transient), the 8192-Gram; peaks: the baseline checkpoint (dense "
                  "eigendecomposition with the shells resident), the shells' rebuild, a probe clock (2 references on the device)" % (nsites, n2, n0, chunk, nX))
    return d


def format_memory_figure(d):
    keys = ("columns_GiB", "chebyshev_workspace_GiB", "Vneg_GiB", "shells_U_GiB", "gram_GiB", "steady_propagation_GiB", "baseline_checkpoint_peak_GiB",
            "shells_rebuild_peak_GiB", "probe_clock_peak_GiB", "pair_route_vacuum_peak_GiB", "cross_cut_block_GiB", "cross_cut_svd_peak_GiB", "trace_norm_gram_peak_GiB", "cross_cut_checkpoint_peak_GiB")
    return "memory figure (GiB, estimated): " + ", ".join("%s %.2f" % (k_.replace("_GiB", ""), d[k_]) for k_ in keys) + "; " + d["label"]


# ------------------------------------------------------------------------------------------------------------ one channel
class ChannelReadout:
    """one channel's D13 readouts with their caches: the dense vacuum at the baseline clocks, the exterior shells per --ext-every-rs, the
    source modes per reference clock (propagated with the sea by propagate_sources), the references' columns and vacua for the probes."""

    def __init__(self, kappa, R_sites, device, rs_nom, cfg):
        self.kappa = int(kappa); self.w = 16.0 * self.kappa
        self.R = np.asarray(R_sites, float); self.n = len(self.R); self.n2 = 2 * self.n
        self.R2 = torch.as_tensor(np.repeat(self.R, 2), dtype=RD, device=device)
        self.dev = device; self.rs = float(rs_nom)
        self.cfg = dict(cfg)
        self.cuts = [("%g" % x, x * self.rs) for x in cfg["cuts_rs"]]
        self.ledger_cuts = [("%g" % x, x * self.rs) for x in cfg["ledger_cuts_rs"]]
        self.probe_cuts = [("%g" % x, x * self.rs) for x in cfg["probe_cuts_rs"]]
        self.R_cut = cfg["R_cut_rs"] * self.rs; self.R_collar = cfg["R_collar_rs"] * self.rs
        self.TH_ref = None
        self.vac = None; self.shells = None; self.shells_t = None
        self.src = {}; self.refs = {}; self.ref_vac = {}; self.ref_vac_bytes = 0
        self.collar_Pm = None; self.collar_Pm_clock = None
        self._masks = None
        self.ref_loader = None
        self.sector_on = False; self.sector_check_now = False                                                                 # v1c (v6o W1): set by page_history_v6o per checkpoint
        self.w1_floor = None                                                                                                  # v1c (W1_fix_v1c): the propagation's floor (the impurity, the a priori bound), set by page_history_v6o per checkpoint

    # ------------------------------------------------------------------------------------------------------ the masks
    def masks(self):
        if self._masks is None:
            self._masks = {lab: (self.R2 > x).to(CD)[:, None] for lab, x in self.ledger_cuts}
        return self._masks

    # ----------------------------------------------------------------------------------------------------- the vacuum
    def vacuum(self, h_csr, t, k, due):
        """the dense vacuum at a baseline clock (computed and cached; the route of the configuration), None elsewhere."""
        if self.vac is not None and abs(self.vac["t"] - t) < 1e-9 and self.vac.get("Vneg") is not None:
            return self.vac
        if not due or self.cfg.get("vacuum", "on") == "off":
            return None
        if self.vac is not None:
            self.vac["Vneg"] = None
            self.vac["V_plus"] = None                                                                                         # v1c (v6o W1)
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        d = killing_vacuum(h_csr, self.dev, self.cfg["vac_route"], self.cfg["vac_cluster_tol"])
        d["t"] = float(t); d["k"] = int(k)
        self.vac = d
        return d

    def set_vacuum(self, t, k, e_np, Vneg, route, E_ref, extra=None):                                                        # v1c (v6o W1): extra carries the sector data
        self.vac = {"t": float(t), "k": int(k), "e": e_np, "Vneg": Vneg, "n_neg": int(Vneg.shape[1]), "E_ref": float(E_ref), "route": route, "seconds": 0.0,
                    "n_zero_1e-12": int((np.abs(e_np) < 1e-12).sum())}
        if extra:                                                                                                             # v1c (v6o W1)
            self.vac.update(extra)                                                                                            # v1c (v6o W1)

    def release_vacuum(self):
        if self.vac is not None:
            self.vac["Vneg"] = None
            self.vac["V_plus"] = None                                                                                         # v1c (v6o W1)

    def vacuum_clock(self):
        return (self.vac["t"] / self.rs) if self.vac is not None else None

    # ----------------------------------------------------------------------------------------------------- the shells
    def shells_at(self, h_csr, t):
        if self.TH_ref is None or self.cfg.get("shells", "on") == "off":
            return None
        if self.shells is None or (t - self.shells_t) >= self.cfg["ext_every_rs"] * self.rs - 1e-9:
            free_shells(self.shells); self.shells = None
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
            self.shells = killing_shells(h_csr, self.R, self.R_cut, self.dev, self.TH_ref, self.cfg["bin_TH"], self.cfg["bin_max_TH"], self.cfg["band_top"], self.cfg["hard_TH"])
            self.shells_t = float(t)
            log("D13 kappa %d: the exterior block beyond R_cut %.1f r_s at T/r_s %.3f: %d modes, Killing energies %.4e to %.4e (band top %g: %d above it), %d bins (%d hard, %d above the band), %.1f s" % (
                self.kappa, self.cfg["R_cut_rs"], t / self.rs, self.shells["nX"], self.shells["e_extremes"][0], self.shells["e_extremes"][1], self.cfg["band_top"], self.shells["n_above_band"],
                len(self.shells["bins"]), sum(1 for b in self.shells["bins"] if b["class"] == "hard"), sum(1 for b in self.shells["bins"] if b["class"] == "above"), self.shells["seconds"]))
        return self.shells

    # ----------------------------------------------------------------------------------------------- the source modes
    def add_sources(self, k_ref, t, Phi, h_csr, vac, r_region, region_label):
        d = source_modes(Phi, self.R2, r_region, h_csr, self.dev, vac["Vneg"] if vac is not None else None, self.cfg["source_weight"], self.cfg["source_max"])
        S = d.pop("S")
        meta = {k_: v_ for k_, v_ in d.items() if not isinstance(v_, np.ndarray)}
        arrays = {k_: v_ for k_, v_ in d.items() if isinstance(v_, np.ndarray)}
        meta.update({"t_ref": float(t), "t_ref_over_rs": t / self.rs, "k_ref": int(k_ref), "region": region_label, "n_modes": int(S.shape[1]),
                     "occ_C_min": float(np.min(d["occ_C"])) if len(d["occ_C"]) else None, "occ_vac_extremes": ([float(np.min(d["occ_vac"])), float(np.max(d["occ_vac"]))] if d.get("occ_vac") is not None and len(d["occ_vac"]) else None),
                     "lam_min_kept": float(np.min(d["lam"])) if len(d["lam"]) else None,
                     "label": "the eigenvectors of C restricted to the sites inside the region with eigenvalue >= %g, as unit modes s = Phi u in ran C; occupation <s|C|s> = 1 up to rounding (printed); the vacuum's <s|P_-(t_ref)|s>; propagated with the sea without collection" % self.cfg["source_weight"]})
        self.src[int(k_ref)] = {"S": S, "S0": S.cpu(), "t": float(t), "k": int(k_ref), "meta": meta, "arrays": arrays}
        return meta, arrays

    def propagate_sources(self, Bh, dt, tol, chunk):
        for d in self.src.values():
            if d["S"].shape[1]:
                d["S"], _, _ = propagate(Bh, d["S"], dt, tol, chunk)

    # --------------------------------------------------------------------------------------------------- the references
    def register_reference(self, k_ref, t_ref, columns=None, vacuum_path=None):
        r = self.refs.setdefault(int(k_ref), {"k": int(k_ref), "t": float(t_ref), "t_over_rs": t_ref / self.rs})
        if columns:
            r["columns"] = columns
        if vacuum_path:
            r["vacuum"] = vacuum_path

    def cache_ref_vacuum(self, k_ref, Vneg):
        nb = Vneg.numel() * 16
        if self.cfg["ref_cache"] == "host" and self.ref_vac_bytes + nb <= self.cfg["ref_cache_GB"] * 1e9:
            self.ref_vac[int(k_ref)] = Vneg.cpu(); self.ref_vac_bytes += nb
            return True
        return False

    def ref_vacuum_host(self, k_ref):
        x = self.ref_vac.get(int(k_ref))
        return x.to(self.dev) if x is not None else None

    def load_reference(self, k_ref, Phi_live=None):
        if Phi_live is not None:
            Vneg = self.vac["Vneg"] if (self.vac is not None and self.vac.get("k") == int(k_ref) and self.vac.get("Vneg") is not None) else self.ref_vacuum_host(k_ref)
            if Vneg is not None:
                return Phi_live, Vneg
        if self.ref_loader is None:
            raise RuntimeError("no reference loader set for the probes")
        return self.ref_loader(int(k_ref))

    # ---------------------------------------------------------------------------------------------------- (d) probes
    def probes(self, t_now, k_now, Phi_now, h_csr, record_steps, refs):
        """(d) at a readout clock: per probe cut the probes of J_x(t_now) back-propagated through the records (record_steps(k) returns the
        record's [(h_csr, dt)] in forward order) to each reference record (refs: [(k_ref, t_ref)], descending), the split there."""
        cfg = self.cfg; out = []
        refs = sorted([(int(k_), float(t_)) for k_, t_ in refs if int(k_) <= int(k_now)], key=lambda z: -z[0])
        for lab, x in self.probe_cuts:
            W, j, jinfo = probe_vectors(h_csr, self.R, x, self.dev, cfg["n_probes"])
            Jb = OPS.Banded(OPS.current_out_csr(h_csr, self.R, x), self.dev)
            L_direct = tr_blocked(Jb, Phi_now, None, cfg["trace_chunk"]); del Jb
            k = int(k_now); res = []; n_rec = 0; Kmax = 0; tail = 0.0
            for (k_ref, t_ref) in refs:
                for kk in range(k - 1, k_ref - 1, -1):
                    for (hk, dt) in reversed(record_steps(kk)):
                        W, K_, tl = propagate(hk if isinstance(hk, OPS.Banded) else OPS.Banded(hk, self.dev), W, -dt, cfg["cheb_tol"], cfg["chunk"])   # v1c (v6o W1): a banded operator per sub-step accepted
                        Kmax = max(Kmax, K_); tail += tl; n_rec += 1
                k = k_ref
                Phi_ref, Vneg_ref = self.load_reference(k_ref, Phi_now if k_ref == int(k_now) else None)
                S0 = self.src.get(k_ref, {}).get("S0")
                S0d = S0.to(self.dev) if S0 is not None else None
                spl = probe_split(W, j, Phi_ref, Vneg_ref, S0d)
                spl.update({"k_ref": int(k_ref), "t_ref_over_rs": t_ref / self.rs, "records_back_propagated": n_rec, "cheb_K_max": Kmax, "cheb_tail_acc": tail,
                            "check_L_minus_direct": spl["L_total_backpropagated"] - L_direct, "n_source_modes": int(S0.shape[1]) if S0 is not None else 0})
                res.append(spl)
                if Phi_ref is not Phi_now:
                    del Phi_ref
                del Vneg_ref, S0d
                if self.dev.type == "cuda":
                    torch.cuda.empty_cache()
            out.append({"cut": lab, "R_abs": float(x), "rank": jinfo["rank"], "support_modes": jinfo["support_modes"], "trace_norm_J": jinfo["trace_norm"], "op_norm_J": jinfo["op_norm"],
                        "L_direct": L_direct, "refs": res,
                        "label": "L(t) = Tr[C(t) J_x(t)] = sum_a j_a <w_a(t_ref)|C(t_ref)|w_a(t_ref)>, the probes back-propagated through the stored fields without collection; "
                                 "L_old on C - P_-(t_ref), L_sea on P_-(t_ref), L_src on the source modes; the rank-16 bound ||J_x||_1 beside"})
        return out

    # -------------------------------------------------------------------------------------------------- the checkpoint
    def checkpoint(self, Phi, h_csr, Bh, t, k, N, v, E_k, rs_k, TH_now, vac, dens=None, T_start=0.0, join=None, form=None):
        """(a), (b)'s inventories, (c), the collar flow's soft/hard split, (e)'s transfer, (f), (g) at one checkpoint; returns (scalars,
        arrays). vac: this clock's dense vacuum (killing_vacuum's dict) or None."""
        t0 = time.time()
        Vneg = vac["Vneg"] if vac is not None else None
        trs = t / self.rs
        sc = {"t_over_rs": trs, "k": int(k), "E": float(E_k), "two_gE": float(rs_k), "T_H_now": float(TH_now), "T_H_ref": self.TH_ref,
              "hard_threshold_energy": (self.cfg["hard_TH"] * self.TH_ref) if self.TH_ref else None, "vacuum_dense": vac is not None,
              "vacuum_route": vac["route"] if vac is not None else None, "vacuum_n_neg": vac["n_neg"] if vac is not None else None, "vacuum_E_ref": vac["E_ref"] if vac is not None else None,
              "cuts_rs": [x for _, x in self.cuts], "join": join}
        arrays = {}
        if getattr(self, "sector_on", False) and (vac is None or vac.get("V_plus") is not None):                                 # v1c (v6o W1): the halved route (the + sector; the - beside at the check clocks)
            ent = W1S.exterior_entropies_sector(Phi, self.R, self.cuts, vac, check_minus=bool(getattr(self, "sector_check_now", False)), floor=getattr(self, "w1_floor", None))   # v1c (v6o W1; W1_fix_v1c: the floors beside every excess)
            sc["sector"] = ent.pop("_sector")                                                                                   # v1c (v6o W1)
            if vac is not None and self.cfg.get("cut_norm", "on") == "on" and bool(getattr(self, "sector_check_now", False)):     # v1c (v6o W1): the cross-cut norms at the check clocks
                W1S.cross_cut_readout_sector(Phi, vac, self.R, self.cuts, ent, self.cfg, self.kappa, trs, sc, arrays)             # v1c (v6o W1)
            sc["sector_line"] = W1S.sector_line(sc["sector"], ent)                                                              # v1c (v6o W1)
        else:                                                                                                                   # v1c (v6o W1): v1b's full route
            ent = exterior_entropies(Phi, self.R2, self.cuts, Vneg)
            if Vneg is not None and self.cfg.get("cut_norm", "on") == "on":                                 # v1b: the cross-cut block trace norms (EVOLVEDCUT EC1)
                cross_cut_readout(Phi, Vneg, self.R2, self.cuts, ent, self.cfg, self.kappa, trs, sc, arrays)
            if getattr(self, "w1_floor", None):                                                                                 # v1c (W1_fix_v1c): the floors on the full route (modes, no doubling)
                W1S.excess_floors(ent, self.R, self.cuts, self.w1_floor, sector=False)                                          # v1c (W1_fix_v1c)
        sc["cuts"] = {}
        for lab, d in ent.items():
            for key in [k_ for k_, v_ in d.items() if isinstance(v_, np.ndarray)]:                                              # v1c (v6o W1): every spectrum array (the sector route's included)
                arrays["%s_cut%s" % (key, lab)] = d.pop(key)
            sc["cuts"][lab] = d
        sc["inventory"] = inventories(Bh, Phi, self.R2, self.cuts, Vneg, self.cfg["trace_chunk"])
        sh = self.shells_at(h_csr, t)
        if sh is not None:
            cont = shell_content(sh, Phi, Vneg)
            sc["shells"] = {"totals": cont["totals"], "nX": sh["nX"], "block_clock_over_rs": self.shells_t / self.rs, "e_extremes": sh["e_extremes"], "n_above_band": sh["n_above_band"],
                            "n_bins": len(sh["bins"]), "reference_at_this_clock": "P_-(h(t)) and the block's own ground" if Vneg is not None else "the block's own ground 1(e < 0)"}
            arrays["shell_e"] = sh["e"]; arrays["shell_n_j"] = cont["n_j"]; arrays["shell_excess_j_block"] = cont["excess_j_block"]
            if "excess_j_Pminus" in cont:
                arrays["shell_nvac_j"] = cont["nvac_j"]; arrays["shell_excess_j_Pminus"] = cont["excess_j_Pminus"]
            for key, val in cont["bins"].items():
                arrays["shellbin_" + key] = val
            arrays["shellbin_class"] = np.array([{"soft": 0, "hard": 1, "above": 2}[c_] for c_ in cont["class"]])
        if Vneg is not None:
            ic, lo, hi, _ = collar_window_rows(self.R, self.R_collar, self.cfg["collar_window"])
            rows = torch.arange(2 * lo, 2 * hi, device=self.dev)
            VW = Vneg.index_select(0, rows)
            self.collar_Pm = (VW @ VW.mH).cpu().numpy(); self.collar_Pm_clock = trs; del VW
        if self.collar_Pm is not None and self.TH_ref is not None:
            cs = collar_window_split(Phi, self.collar_Pm, N, v, self.R, self.R_collar, self.kappa, self.cfg["collar_window"], self.cfg["hard_TH"] * self.TH_ref)
            cs["Pm_clock_over_rs"] = self.collar_Pm_clock
            sc["collar_split"] = cs
        if self.src:
            radii = [("collar", self.R_collar), ("cut", self.R_cut), ("far", max(x for _, x in self.cuts))]
            sc["sources"] = {}
            for k_ref, d in sorted(self.src.items()):
                tr = source_transfer(d["S"], Phi, self.R2, radii)
                scal = {k_: v_ for k_, v_ in tr.items() if not isinstance(v_, np.ndarray)}
                scal.update({"t_ref_over_rs": d["t"] / self.rs, "radii": {lab: float(r_) for lab, r_ in radii}})
                sc["sources"][str(k_ref)] = scal
                for k_, v_ in tr.items():
                    if isinstance(v_, np.ndarray):
                        arrays["src_ref%d_%s" % (k_ref, k_)] = v_
        sc["first_return"] = {}
        tf = form.get("t") if form else None; rf = form.get("r_root") if form else None
        for lab, x in self.cuts:
            sc["first_return"][lab] = first_return(N, v, self.R, x, t, T_start, self.rs, tf, rf)
        r_t = outer_root(N, v, self.R)
        sc["r_t"] = r_t
        if Vneg is not None:
            rho = killing_density_sites(Bh, Phi, Vneg, self.cfg["trace_chunk"])
            arrays["rho_sites"] = rho
            sc["rho_sum_check"] = float(np.sum(rho))
            sc["enclosed"] = enclosed_source(rho, self.R, N, v, dens.get("g") if dens else self.cfg.get("g", 0.25), r_t,
                                             dens.get("eps") if dens else None, dens.get("J") if dens else None, dens.get("R") if dens else None)
        sc["seconds"] = time.time() - t0
        return sc, arrays

    # ------------------------------------------------------------------------------------------------------- the state
    def state_host(self):
        return {"src": {k_: {"S": d["S"].cpu(), "S0": d["S0"], "t": d["t"], "k": d["k"], "meta": d["meta"], "arrays": d["arrays"]} for k_, d in self.src.items()},
                "refs": self.refs, "collar_Pm": self.collar_Pm, "collar_Pm_clock": self.collar_Pm_clock, "shells_t": self.shells_t,
                "vac_meta": ({k_: v_ for k_, v_ in self.vac.items() if k_ not in ("Vneg", "e")} if self.vac is not None else None)}

    def load_state_host(self, d, device):
        self.src = {int(k_): {"S": x["S"].to(device), "S0": x["S0"], "t": x["t"], "k": x["k"], "meta": x["meta"], "arrays": x["arrays"]} for k_, x in (d.get("src") or {}).items()}
        self.refs = {int(k_): v_ for k_, v_ in (d.get("refs") or {}).items()}
        self.collar_Pm = d.get("collar_Pm"); self.collar_Pm_clock = d.get("collar_Pm_clock")


def format_line(kappa, sc, E_inv=None):
    """one labelled line of a checkpoint's readouts."""
    cuts = sc.get("cuts", {})
    sx = ", ".join("%s r_s %.6f%s%s%s" % (lab, d["S_X"], (" (vacuum %.6f, excess %.6f)" % (d["S_X_vac"], d["excess"])) if d.get("S_X_vac") is not None else "",   # v1c (W1_fix_v1c): the floor and the resolution beside the excess
                                          (", t_X %.6f %s" % (d["t_X"], "ok" if d.get("bound_ok") else ("FAIL" if d.get("bound_ok") is False else "-"))) if d.get("t_X") is not None else "",   # v1c (W1_fix_v1c)
                                          (", floor %.1e (low %.1e) %s" % (d["excess_floor_high"], d.get("excess_floor_low") or 0.0, d.get("excess_resolution") or "-")) if d.get("excess_floor_high") is not None else "") for lab, d in cuts.items())   # v1c (W1_fix_v1c)
    tn = sc.get("trace_norm") or {}                                                                             # v1b (a')
    if not tn:
        tns = "-"
    elif tn.get("nu") is not None:
        tns = "%.6e (%s route; rank %d, largest singular value %.4e, ||.||_2 %.4e)" % (tn["nu"], tn.get("route"), int(tn.get("rank") or 0), float(tn.get("sigma_max") or 0.0), float(tn.get("frob") or 0.0))
    else:
        tns = "<= %s (rank x sigma_max; rank <= N_occ + N_neg, sigma_max <= 1), <= %s (Cauchy-Schwarz); ||.||_2 %s, sigma_max estimate %s" % (
            _f(tn.get("bound_rank_sigma"), "%.4e"), _f(tn.get("bound_cauchy_schwarz"), "%.4e"), _f(tn.get("frob"), "%.4e"), _f(tn.get("sigma_max_estimate"), "%.4e"))
    gap = max([abs(d["purity_gap"]) for d in cuts.values()], default=0.0)
    inv = sc.get("inventory", {})
    iv = ", ".join("%s r_s: C %.6e%s" % (lab, d["E_C"], (", D %.6e" % d["E_D"]) if d.get("E_D") is not None else "") for lab, d in inv.items())
    sh = sc.get("shells"); shs = "off"
    if sh and sh.get("totals"):
        T = sh["totals"]
        shs = "hard out %.3e in %.3e, soft out %.3e in %.3e, above the band top (bound content) N %.3e%s" % (
            T["hard_E_out"], T["hard_E_in"], T["soft_E_out"], T["soft_E_in"], T["above_N_total"],
            ("; against P_-: hard out %s in %s" % (_f(T.get("hard_E_out_P"), "%.3e"), _f(T.get("hard_E_in_P"), "%.3e"))) if T.get("hard_E_out_P") is not None else "")
    cs = sc.get("collar_split")
    css = "-" if not cs else "out soft %.3e hard %.3e, in soft %.3e hard %.3e (P_- block of T/r_s %s)" % (cs["E_flux_out_soft"], cs["E_flux_out_hard"], cs["E_flux_in_soft"], cs["E_flux_in_hard"], _f(cs.get("Pm_clock_over_rs"), "%.3f"))
    src = sc.get("sources") or {}
    srcs = "; ".join("ref %s: %d modes, tau to the collar sum %.4f max %.4f, w/tau mean %s" % (k_, d["n_modes"], d.get("tau_sum_collar", 0.0), d.get("tau_max_collar", 0.0), _f(d.get("occ_inferred_mean_collar"), "%.4f")) for k_, d in src.items()) or "-"
    fr = sc.get("first_return", {})
    frs = ", ".join("%s r_s: %s" % (lab, _f(d.get("first_return_rs"), "%.1f")) for lab, d in fr.items())
    en = sc.get("enclosed")
    ens = "-" if not en else "I_L %.4e at the root %.3f, kappa_R %.4e, delta R_L %s (over R %s), I_L/M_root %.3e%s" % (
        en["I_L"], en["R_root"], en["kappa_R_slope"], _f(en.get("delta_R_L"), "%.4e"), _f(en.get("delta_R_L_over_R"), "%.3e"), en["I_L_over_M_root"],
        ("; kappa from the densities over the slope %s" % _f(en.get("kappa_ratio_densities_over_slope"), "%.3f")) if en.get("kappa_ratio_densities_over_slope") is not None else "")
    return ("D13 kappa %d T/r_s %.3f: E_inv %s; S_X raw [%s] nats (complement gap max %.1e); ||C - P_-||_1 %s; energy beyond the cut [%s]; shells %s; collar window %s; source modes %s; "   # v1c (v6o W1): 'raw'
            "first return (r_s) [%s]; enclosed source %s; sector %s; %.0f s" % (kappa, sc["t_over_rs"], _f(E_inv, "%.6e"), sx, gap, tns, iv, shs, css, srcs, frs, ens, sc.get("sector_line", "-"), sc.get("seconds", 0.0)))   # v1c (v6o W1)


# ------------------------------------------------------------------------------------------- the convergence table in kappa
def ksum_table(out_dir, tag, kappas, hard_frac=0.01, from_rs=40.0, write=True):
    """the partial sums over the channels run so far (weights 16 kappa, already in the rows' _weighted keys and applied here to the D13
    columns) of E_inv, the collar flow (soft and hard outgoing, the windowed reading), the interval current and its comparators, the
    exterior-cut entropies and their excess, the hard shells; the extension rule: the last channel's hard part against the total."""
    per = {}
    for kap in kappas:
        sj = os.path.join(out_dir, "%s_k%d_summary.json" % (tag, kap)); dj = os.path.join(out_dir, "%s_k%d_d13.json" % (tag, kap))
        if not os.path.exists(sj):
            continue
        S = json.load(open(sj)); D = json.load(open(dj)) if os.path.exists(dj) else {"clocks": {}}
        per[kap] = {"rows": {round(r["t_over_rs"], 4): r for r in S.get("rows", [])}, "clocks": {round(c["t_over_rs"], 4): c for c in D.get("clocks", {}).values()},
                    "w": 16.0 * kap, "final": S.get("final")}
    kap_sorted = sorted(per)
    if not kap_sorted:
        return None
    clocks = sorted(set.intersection(*[set(per[k]["rows"]) for k in kap_sorted]))
    cut_labels = []
    for k_ in kap_sorted:
        for c in per[k_]["clocks"].values():
            pc = (c.get("per_channel") or {}).get(str(k_)) or {}
            if pc.get("cuts"):
                cut_labels = list(pc["cuts"].keys()); break
        if cut_labels:
            break

    def d13_of(kap, t):
        return ((per[kap]["clocks"].get(t) or {}).get("per_channel") or {}).get(str(kap)) or {}

    table = []
    for i, K in enumerate(kap_sorted):
        sub = kap_sorted[: i + 1]
        for t in clocks:
            e = {"K": K, "kappas": sub, "t_over_rs": t}
            for key in ("E_inv_weighted", "collar_flow_acc_weighted", "L_interval", "L_H_kappaT", "L_H_TH", "E_out_weighted"):
                vals = [per[k_]["rows"][t].get(key) for k_ in sub]
                e[key] = sum(vals) if all(v_ is not None for v_ in vals) else None
            e["L_over_LH_kappaT"] = (e["L_interval"] / e["L_H_kappaT"]) if (e["L_interval"] is not None and e["L_H_kappaT"]) else None
            e["L_over_LH_TH"] = (e["L_interval"] / e["L_H_TH"]) if (e["L_interval"] is not None and e["L_H_TH"]) else None
            for key in ("E_flux_out_soft", "E_flux_out_hard", "E_flux_in_soft", "E_flux_in_hard"):
                vals = [(d13_of(k_, t).get("collar_split") or {}).get(key) for k_ in sub]
                e["collar_" + key] = sum(per[k_]["w"] * v_ for k_, v_ in zip(sub, vals)) if all(v_ is not None for v_ in vals) else None
            for key in ("hard_E_out", "hard_E_in", "soft_E_out", "soft_E_in", "above_N_total"):
                vals = [((d13_of(k_, t).get("shells") or {}).get("totals") or {}).get(key) for k_ in sub]
                e["shells_" + key] = sum(per[k_]["w"] * v_ for k_, v_ in zip(sub, vals)) if all(v_ is not None for v_ in vals) else None
            for lab in cut_labels:
                for key in ("S_X", "excess", "S_X_vac", "t_X", "abs_dS", "excess_floor_high", "excess_floor_low"):             # v1c (W1_fix_v1c): the floors summed with the weights
                    vals = [((d13_of(k_, t).get("cuts") or {}).get(lab) or {}).get(key) for k_ in sub]
                    e["%s_cut%s" % (key, lab)] = sum(per[k_]["w"] * v_ for k_, v_ in zip(sub, vals)) if all(v_ is not None for v_ in vals) else None
            for key in ("nu", "bound_rank_sigma", "bound_cauchy_schwarz", "frob"):                                              # v1b (a')
                vals = [((d13_of(k_, t).get("trace_norm") or {}).get(key)) for k_ in sub]
                e["%s_trace_weighted" % key] = sum(per[k_]["w"] * v_ for k_, v_ in zip(sub, vals)) if all(v_ is not None for v_ in vals) else None
            for lab in cut_labels:
                ex_, tx_ = e.get("excess_cut%s" % lab), e.get("t_X_cut%s" % lab)
                e["bound_ok_cut%s" % lab] = bool(abs(ex_) <= tx_ + 1e-6 * sum(per[k_]["w"] for k_ in sub)) if (ex_ is not None and tx_ is not None) else None
            I_vals = [((d13_of(k_, t).get("enclosed") or {}).get("I_L")) for k_ in sub]
            e["I_L_weighted"] = sum(per[k_]["w"] * v_ for k_, v_ in zip(sub, I_vals)) if all(v_ is not None for v_ in I_vals) else None
            enc = d13_of(sub[0], t).get("enclosed") or {}
            if e["I_L_weighted"] is not None and enc.get("kappa_R_slope") and enc.get("R_root"):
                g_ = enc["M_root"] and enc["R_root"] / (2.0 * enc["M_root"])
                e["delta_R_L_weighted"] = g_ * e["I_L_weighted"] / (enc["R_root"] * enc["kappa_R_slope"]) if g_ else None
            table.append(e)
    K = kap_sorted[-1]
    shares = []
    for t in clocks:
        if t < from_rs - 1e-9:
            continue
        tot = [x for x in table if x["K"] == K and x["t_over_rs"] == t]
        if not tot or tot[0].get("collar_E_flux_out_hard") is None:
            continue
        last = per[K]["w"] * ((d13_of(K, t).get("collar_split") or {}).get("E_flux_out_hard") or 0.0)
        tt = tot[0]["collar_E_flux_out_hard"]
        shares.append({"t_over_rs": t, "share": (abs(last) / abs(tt)) if tt != 0 else (0.0 if last == 0 else float("inf")), "last": last, "total": tt})
    smax = max([s["share"] for s in shares], default=0.0)
    w1 = W1S.convergence_rule(per, kap_sorted, clocks, cut_labels, hard_frac, d13_of, from_rs)                                   # v1c (v6o W1): the tail-majorant rule (defect 10)
    res = {"kappas": kap_sorted, "clocks": clocks, "table": table, "last_channel": K, "last_channel_hard_share_max": smax, "hard_moves_above_frac": bool(smax > hard_frac),
           "hard_frac": hard_frac, "shares": shares, "cut_labels": cut_labels,
           "retired_rule": "v6n's: the sum extended while the last channel's hard outgoing collar flow exceeded hard_frac of the total at a readout clock from %g r_s; printed, deciding nothing (it passes on the flat held vacuum with an infinite remainder of the raw sum)" % from_rs,   # v1c (v6o W1)
           "rule": w1["rule"], "w1": w1, "extend": bool(w1["extend"]), "verdict_line": w1["verdict_line"],                      # v1c (v6o W1)
           "label": "partial sums over kappa <= K with the weights 16 kappa; E_inv (= E_kappa, the formed-energy majorant's term), the collar flow accumulated, the interval current and its comparators, the collar flow's soft/hard split, the raw exterior-cut entropies (regulated), the held vacuum's and their excess (the readout) with the excess's propagation floors (v1c), the hard shells, the enclosed source"}   # v1c (v6o W1)
    if write:
        pj = os.path.join(out_dir, tag + "_ksum.json")
        with open(pj + ".tmp", "w") as f:
            json.dump(res, f, default=_jdefault, indent=1)
        os.replace(pj + ".tmp", pj)
        L = ["# %s: the convergence table in kappa (channels %s; weights 16 kappa)" % (tag, kap_sorted), "", res["label"], "", res["verdict_line"], "",
             "the retired rule (v6n): the last channel's hard collar-flow share %.4f (%s by its 1-percent criterion); it decides nothing here" % (smax, "moves" if res["hard_moves_above_frac"] else "passes"), ""]   # v1c (v6o W1)
        show = [t for t in clocks if t >= from_rs - 1e-9 or abs(t - round(t / 10.0) * 10.0) < 1e-6 or any(abs(t - z) < 1e-6 for z in (12.0, 14.0, 20.0, 30.0))]
        cols = ["E_inv_weighted", "L_interval", "L_over_LH_kappaT", "collar_E_flux_out_soft", "collar_E_flux_out_hard", "shells_hard_E_out", "shells_above_N_total"] + ["excess_cut%s" % lab for lab in cut_labels] + ["S_X_cut%s" % lab for lab in cut_labels] + ["t_X_cut%s" % lab for lab in cut_labels] + ["nu_trace_weighted"]
        for K_ in kap_sorted:
            L += ["", "## K = %d (kappa 1 to %d)" % (K_, K_), "", "| T/r_s | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for t in show:
                e = [x for x in table if x["K"] == K_ and x["t_over_rs"] == t]
                if e:
                    L.append("| %.3f | %s |" % (t, " | ".join(_f(e[0].get(c_), "%.6e") for c_ in cols)))
        L += W1S.convergence_markdown(res["w1"])                                                                              # v1c (v6o W1): the majorant's tables (with the floors, v1c)
        with open(os.path.join(out_dir, tag + "_ksum.md"), "w") as f:
            f.write("\n".join(L) + "\n")
    return res


def compare_runs(out_dir, tag_a, tag_b):
    """the record-interpolation control: the printable columns of two runs' per-channel outputs at common clocks and their differences."""
    def load(tag):
        S = json.load(open(os.path.join(out_dir, tag + "_summary.json")))
        dj = os.path.join(out_dir, tag + "_d13.json")
        D = json.load(open(dj)) if os.path.exists(dj) else {"clocks": {}}
        return {round(r["t_over_rs"], 4): r for r in S["rows"]}, {round(c["t_over_rs"], 4): c for c in D.get("clocks", {}).values()}
    ra, da = load(tag_a); rb, db = load(tag_b)
    common = sorted(set(ra) & set(rb))
    out = []
    for t in common:
        e = {"t_over_rs": t}
        for key in ("E_inv_weighted", "collar_flow_acc_weighted", "L_interval"):
            e[key + "_a"] = ra[t].get(key); e[key + "_b"] = rb[t].get(key)
            e[key + "_diff"] = (ra[t][key] - rb[t][key]) if (ra[t].get(key) is not None and rb[t].get(key) is not None) else None
        for kap in (da.get(t) or {}).get("per_channel", {}):
            ca = da[t]["per_channel"][kap].get("cuts", {}); cb = ((db.get(t) or {}).get("per_channel", {}).get(kap) or {}).get("cuts", {})
            for lab in ca:
                if lab in cb:
                    e["S_X_cut%s_diff" % lab] = ca[lab]["S_X"] - cb[lab]["S_X"]
                    if ca[lab].get("t_X") is not None and cb[lab].get("t_X") is not None:                          # v1b
                        e["t_X_cut%s_diff" % lab] = ca[lab]["t_X"] - cb[lab]["t_X"]
                    if ca[lab].get("excess") is not None and cb[lab].get("excess") is not None:
                        e["excess_cut%s_diff" % lab] = ca[lab]["excess"] - cb[lab]["excess"]
        out.append(e)
    res = {"tag_a": tag_a, "tag_b": tag_b, "rows": out, "label": "differences a - b at common clocks: E_inv, the accumulated collar flow, the interval current, the exterior-cut entropies and their excess"}
    pj = os.path.join(out_dir, "%s_vs_%s.json" % (tag_a, tag_b))
    with open(pj, "w") as f:
        json.dump(res, f, default=_jdefault, indent=1)
    keys = [k_ for k_ in (out[0].keys() if out else []) if k_.endswith("_diff")]
    L = ["# %s against %s" % (tag_a, tag_b), "", res["label"], "", "| T/r_s | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for e in out:
        L.append("| %.3f | %s |" % (e["t_over_rs"], " | ".join(_f(e.get(k_), "%.3e") for k_ in keys)))
    with open(os.path.join(out_dir, "%s_vs_%s.md" % (tag_a, tag_b)), "w") as f:
        f.write("\n".join(L) + "\n")
    return res


# ------------------------------------------------------------------------------------------------------------------- the CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", action="append", required=True, help="a v6m/v6n state .pt (repeatable; each clock is one entry)")
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True)
    ap.add_argument("--fields-key", dest="fields_key", default="applied", choices=["applied", "stored"])
    ap.add_argument("--g", type=float, default=None); ap.add_argument("--rs-nominal", dest="rs_nominal", type=float, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--channels", default=None, help="a subset of the state's channels (default: all)")
    ap.add_argument("--ref-state", dest="ref_state", action="append", default=[], help="(d) T_rs:PATH of a state at a reference clock (repeatable)")
    ap.add_argument("--force", action="store_true", help="recompute a clock already in the json")
    add_arguments(ap, common=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    import importlib
    V6M = importlib.import_module("page_history_v6m")
    TRACE_CHUNK[0] = int(a.trace_chunk)
    os.makedirs(a.out, exist_ok=True)
    states = []
    for p in a.state:
        st = torch.load(p, map_location="cpu", weights_only=False)
        states.append((int(st["k"]), p, st))
    states.sort(key=lambda z: z[0])
    first = states[0][2]
    nsites = int(next(iter(first["channels"].values()))["Phi"].shape[0]) // 2
    H = V6M.History(a.record, a.fields, nsites, a.g, a.fields_key)
    rs = float(a.rs_nominal or H.rs_nom or 2.0 * H.g * H.E[0])
    cfg = cfg_from_args(a); cfg["g"] = H.g
    k_TH = int(np.argmin(np.abs(H.T - cfg["TH_ref_rs"] * rs)))
    TH_ref = float(cfg["TH_ref"]) if cfg.get("TH_ref") else 1.0 / (4.0 * math.pi * 2.0 * H.g * float(H.E[k_TH]))
    k_form = int(np.argmin(np.abs(H.T - cfg["form_rs"] * rs)))
    Nf, vf = H.fields(k_form); r_form = outer_root(Nf, vf, H.R)
    form = {"t": float(H.T[k_form]), "k": k_form, "r_root": r_form if r_form is not None else 2.0 * H.g * float(H.E[k_form]), "root_present": r_form is not None}
    dens_all = None
    try:
        Z = np.load(a.fields, allow_pickle=True)
        if "eps_tot" in Z.files and "J_tot" in Z.files:
            dens_all = {"eps": np.asarray(Z["eps_tot"], float), "J": np.asarray(Z["J_tot"], float), "R": np.asarray(Z["R"], float)}
    except Exception as ex:                                                 # noqa: BLE001
        log("the fields' densities could not be read (%s): (g)'s D_R check is skipped" % ex)
    store = Store(os.path.join(a.out, a.tag + "_d13"), settings={"args": vars(a), "d13": {k_: (list(v_) if isinstance(v_, (list, tuple)) else v_) for k_, v_ in cfg.items()}},
                  label="D13 readouts (page_d13_readout_v1b: with the cross-cut block trace norms of EVOLVEDCUT EC1) on saved states of %s; no collector, no selection, no archive" % (H.label or a.record))
    refs_given = {}
    for s_ in a.ref_state:
        trs_, path_ = s_.split(":", 1)
        k_ref = int(np.argmin(np.abs(H.T - float(trs_) * rs)))
        refs_given[k_ref] = path_
    lo_j, hi_j = [float(x) * rs for x in cfg["join_rs"].split(",")]
    log("D13 standalone: the history %s (%d records, tau %g, g %g, r_s nominal %g, the chain %d sites); T_H^ref %.6e from the record at T/r_s %.3f (2gE %.4f); cuts %s r_s; device %s; %d states" % (
        H.label or a.record, len(H), H.tau, H.g, rs, nsites, TH_ref, H.T[k_TH] / rs, 2.0 * H.g * H.E[k_TH], cfg["cuts_rs"], dev, len(states)))
    log(format_memory_figure(memory_figure(nsites, None, cfg["chunk"], cfg["R_cut_rs"], rs, cfg["source_max"])))
    ref_states_loaded = {}
    for k, path, st in states:
        t = float(H.T[k]); trs = t / rs
        N, v = H.fields(k); E_k = float(H.E[k]); rs_k = 2.0 * H.g * E_k; TH_now = 1.0 / (4.0 * math.pi * rs_k)
        jN, jv_ = H.join_check(k, lo_j, hi_j)
        chans = [kk for kk in st["channels"].keys() if (a.channels is None or kk in a.channels.split(","))]
        row_last = (st.get("rows") or [{}])[-1] if st.get("rows") else {}
        log("state %s: record %d (T/r_s %.4f), channels %s; the state's last row at T/r_s %s" % (path, k, trs, chans, _f(row_last.get("t_over_rs"), "%.4f")))
        dens = dict(dens_all, eps=dens_all["eps"][k], J=dens_all["J"][k], g=H.g) if (dens_all is not None and k < len(dens_all["eps"])) else {"g": H.g}
        weighted = {}
        for kk in chans:
            kappa = int(kk)
            if store.has_clock(trs, kappa) and not a.force:
                log("kappa %d at T/r_s %.4f is in %s already: skipped (--force recomputes)" % (kappa, trs, store.pj)); continue
            ops = V6M.ChannelOps(kappa, H.R, dev)
            Phi = st["channels"][kk]["Phi"].to(dev)
            cr = ChannelReadout(kappa, H.R, dev, rs, cfg); cr.TH_ref = TH_ref
            h = ops.h_csr(N, v); Bh = ops.banded(h)
            E_hC = tr_blocked(Bh, Phi); E_ref, minfo = tr_min_h0(h); E_inv = E_hC - E_ref
            vac = None
            if cfg["vacuum"] == "on":
                vac = cr.vacuum(h, t, k, due=True)
                log("kappa %d: the dense vacuum at T/r_s %.4f: %d occupied, E_ref (dense) %.10e against Tr min(h, 0) (banded) %.10e (difference %.2e); %s; %.1f s" % (
                    kappa, trs, vac["n_neg"], vac["E_ref"], E_ref, vac["E_ref"] - E_ref, vac["route"], vac["seconds"]))
            sc, arrays = cr.checkpoint(Phi, h, Bh, t, k, N, v, E_k, rs_k, TH_now, vac, dens=dens, T_start=float(H.T[0]), join={"max_dN": jN, "max_rel_dv": jv_, "window_rs": cfg["join_rs"]}, form=form)
            sc.update({"E_inv": E_inv, "E_hC": E_hC, "E_ref_banded": E_ref, "E_ref_info": minfo, "state_path": path})
            pc_row = (row_last.get("per_channel") or {}).get(kk) or {}
            if row_last.get("k") == k and pc_row.get("E_inv") is not None:
                sc["E_inv_state_row"] = pc_row["E_inv"]; sc["E_out_state_row"] = pc_row.get("E_out")
                log("kappa %d: E_inv %.10e here against the state's own row %.10e (difference %.2e: the banded against the dense reference)" % (kappa, E_inv, pc_row["E_inv"], E_inv - pc_row["E_inv"]))
            log(format_line(kappa, sc, E_inv))
            if refs_given:
                refs = sorted([(k_, float(H.T[k_])) for k_ in refs_given if k_ <= k], key=lambda z: -z[0])

                def loader(k_ref, kappa=kappa, ops=ops, cr=cr):
                    if (k_ref, kappa) not in ref_states_loaded:
                        ref_states_loaded[(k_ref, kappa)] = torch.load(refs_given[k_ref], map_location="cpu", weights_only=False)["channels"][str(kappa)]["Phi"]
                    Phi_r = ref_states_loaded[(k_ref, kappa)].to(dev)
                    Vn = cr.ref_vacuum_host(k_ref)
                    if Vn is None:
                        vr = killing_vacuum(ops.h_csr(*H.fields(k_ref)), dev, cfg["vac_route"], cfg["vac_cluster_tol"]); Vn = vr["Vneg"]; cr.cache_ref_vacuum(k_ref, Vn)
                    return Phi_r, Vn
                cr.ref_loader = loader
                steps = lambda kk_, ops=ops: [(ops.h_csr(*H.fields(kk_)), float(H.T[kk_ + 1] - H.T[kk_]) if kk_ + 1 < len(H) else H.tau)]   # noqa: E731
                res = cr.probes(t, k, Phi, h, steps, refs)
                for r_ in res:
                    store.put_probe(dict(r_, t_over_rs=trs, k=k, kappa=kappa))
                    for q in r_["refs"]:
                        log("D13 kappa %d probes at T/r_s %.3f, cut %s r_s, reference T/r_s %.3f: L_direct %.6e, back-propagated L %.6e (check %.1e), L_old %.6e, L_sea %.6e, L_src %s; rank %d, ||J||_1 %.3e, pointwise bound %.3e" % (
                            kappa, trs, r_["cut"], q["t_ref_over_rs"], r_["L_direct"], q["L_total_backpropagated"], q["check_L_minus_direct"], q["L_old"], q["L_sea"], _f(q.get("L_src"), "%.6e"), r_["rank"], q["bound_rank16_trace_norm"], q["bound_old_pointwise"]))
                sc["probes"] = [{k_: v_ for k_, v_ in r_.items() if k_ != "refs"} | {"refs": [{k2: v2 for k2, v2 in q.items() if not isinstance(v2, np.ndarray)} for q in r_["refs"]]} for r_ in res]
            split_arrays(sc, arrays, ())
            store.put_clock(trs, kappa, sc, arrays, common={"k": k, "t": t, "E": E_k, "two_gE": rs_k, "T_H": TH_now, "join_max_dN": jN, "join_max_rel_dv": jv_, "min_N_minus_absv": float(np.min(N - np.abs(v)))})
            for lab, d in sc["cuts"].items():
                for key in ("S_X", "excess", "t_X", "abs_dS"):
                    if d.get(key) is not None:
                        weighted.setdefault("%s_cut%s" % (key, lab), 0.0); weighted["%s_cut%s" % (key, lab)] += 16.0 * kappa * d[key]
            weighted["E_inv"] = weighted.get("E_inv", 0.0) + 16.0 * kappa * E_inv
            tn_ = sc.get("trace_norm") or {}                                                                      # v1b (a'): the weighted trace norm and its bounds
            for key in ("nu", "bound_rank_sigma", "bound_cauchy_schwarz", "frob"):
                if tn_.get(key) is not None:
                    weighted.setdefault("%s_trace" % key, 0.0); weighted["%s_trace" % key] += 16.0 * kappa * tn_[key]
            for lab, d in sc["inventory"].items():
                weighted.setdefault("E_C_cut%s" % lab, 0.0); weighted["E_C_cut%s" % lab] += 16.0 * kappa * d["E_C"]
                if d.get("E_D") is not None:
                    weighted.setdefault("E_D_cut%s" % lab, 0.0); weighted["E_D_cut%s" % lab] += 16.0 * kappa * d["E_D"]
            del Phi, Bh, vac; cr.release_vacuum(); free_shells(cr.shells)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            store.flush()
        if weighted:
            store.J["clocks"][Store.ckey(trs)]["weighted_over_state_channels"] = dict(weighted, channels=chans, weights="16 kappa")
            log("T/r_s %.4f weighted over the state's channels %s (16 kappa): %s" % (trs, chans, ", ".join("%s %.6e" % (k_, v_) for k_, v_ in sorted(weighted.items()))))
            for lab in sorted({k_[len("t_X_cut"):] for k_ in weighted if k_.startswith("t_X_cut")}, key=float):             # v1b (a'): the weighted bound check
                ex_, ab_, tx_ = weighted.get("excess_cut" + lab), weighted.get("abs_dS_cut" + lab), weighted["t_X_cut" + lab]
                if ex_ is not None:
                    tol_w = cfg["cut_bound_tol"] * 16.0 * sum(int(c_) for c_ in chans)
                    log("T/r_s %.4f weighted cross-cut bound at %s r_s: |sum_kappa w excess| %.6e <= sum_kappa w |S_X(C) - S_X(P_-)| %s <= sum_kappa w t_X %.6e: %s" % (
                        trs, lab, abs(ex_), _f(ab_, "%.6e"), tx_, "ok" if abs(ex_) <= tx_ + tol_w else "FAIL"))
        store.flush()
    log("done: %s and %s" % (store.pj, store.pz))


if __name__ == "__main__":
    main()
