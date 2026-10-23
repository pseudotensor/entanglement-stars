#!/usr/bin/env python
"""raymap_solved_flow_v1.py (reviewer A for the solve builder, the busi lane; reorientation 4 C13 (i) and the solve entry's item 6; reviewer A's
annals ruling, inbox #144 item 4: ahead of items 7 to 9, with the dense root's slope kappa_R against 1/(4GE) and the retarded clock at
which it first reaches 0.9 of it as the window's start). The ray map on the solved flow: outgoing light-channel probes launched in the
exterior at 16 r_s, back-propagated through the stored records to T = 0. Reuses page_d13_readout_v1 (its readout (d)'s Chebyshev
back-propagation, the root, the light times, the first return, the join check, the source coefficient) and page_ops_v6 by import.

Definitions (every threshold a named argument; raymap_selftest_v1 checks them on manufactured exteriors):
  the chain: page_ops_v6's channel kappa (2 components per site, R_n = n + 1, massless by default), h[N, v] = (1/2){N, h_0} + (1/2){v, Q}
    (page_ops_v6.clocked_csr) on --nsites sites: the record's stored fields (--fields-key, the applied site fields by default) on its own
    chain and the vacuum exterior of the record's charge appended beyond it (N = 1, v = -sqrt(2gE_k/R)); page_history_v6m's History and
    ChannelOps build it when they are importable (--fields-source), the own construction otherwise, the two compared when both exist.
    h is piecewise constant over each record; the propagation is page_d13_readout_v1's Chebyshev step per record (cheb_apply/propagate,
    the order by the Bessel tail below --cheb-tol): exp(+i dt_k h_k) in reverse record order for the back-propagation, exp(-i dt_k h_k)
    forward for the check.
  the retarded clock: u = T - r_*(R) in the stage's chart, page_ops_v6.chart_retarded_time(T, 2gE_k, R) with the record's charge; the
    launch clocks u/r_s from --u-from-rs to --u-to-rs in steps of --u-step-rs (r_s = --rs-nominal, the run's nominal 64 a), each moved to
    the nearest record clock T_k (a probe launched at T_k passes exactly the records k - 1, ..., 0), extended beyond --u-to-rs in the same
    step to the first return at the launch radius (page_d13's first_return) or to the history's last record when --extend is on.
  the probe: a unit wave packet on the local outgoing branch at the launch site R_L = --launch-rs r_s. With the record's (N_L, v_L) there,
    omega_+(k) = N_L sqrt(sin^2 k + (kappa/R_L)^2 + m^2) + v_L sin(2k)/2, v_+(k) = N_L sin k cos k/E + v_L cos 2k; k_turn is the smallest
    k > 0 with v_+ > 0, k_peak the branch's maximum, the carrier k_0 solves omega_+(k_0) = omega on [k_turn, k_peak]. The packet is
    psi_n = sum_m w(k_m) e^{i k_m (R_n - R_L)} chi_n(k_m) over the chain's wavenumber grid k_m = 2 pi m/n with the Gaussian amplitude
    w(k) = exp(-(k - k_c)^2/(2 sigma_k^2)), zero at and below k_turn (a C^1 ramp of width --k-cut-width-frac k_0 above it: no ingoing
    content) and above k_peak, chi_n(k) the local positive-branch spinor ((E + m), sin k + i kappa/R_n)/sqrt(2E(E + m)) at each site; the
    centre k_c is adjusted so that the spectral mean of omega_+ over w^2 equals omega (energy-matched). The width by --width-rule:
    "rel-spread": sigma_k = --rel-spread k_0 with the spatial width 1/sigma_k capped at --sigma-x-max-rs r_s (the probe stays localized
    at the launch radius); "sigma-u": sigma_x = --sigma-u-rs r_s v_+(k_0) (a fixed width in the retarded clock). Each row prints the
    packet's measured centroid, width, width in u (sigma_x/v_g), Killing energy <h> and spread, group velocity <i[h, X]> and spectral
    content (the weight at k < 0 is the ingoing admixture, zero by construction up to the chain's periodicity).
  U(u): the proper radial position of the packet's centroid on the initial flat slice (the first record, T = 0; the flat-slice gauge's x,
    so dU is proper distance along the slice): U = sum_n R_n rho_n/sum_n rho_n, rho_n the packet's weight at site n; sigma_U its spread.
    This is the affine parameter whose compression the paper's T_d counts: in geometric optics the traced-back wavenumber at T = 0 is
    k_0 = omega/|dU/du| (the WKB phase omega u = int k_0 dU), so kappa(u) = -d ln|dU/du|/du = d ln k_0/du is the wavenumber's blueshift
    exponent, whose e-folds T_d = (2/3) r_s ln(1/(2 chi q)) counts (chi = a/r_s, q = omega r_s, hence T_d = --Td-prefactor r_s
    ln(1/(--Td-inner a omega)): 2.31 r_s at q = 1 and 4.00 r_s at omega = T_H for r_s/a = 64, the paper's Table I). The measured mean
    |k| at T = 0 is printed beside omega/|dU/du| as the check. Two further parametrizations are printed beside, not primary: U_0(u) =
    -r_*^(0)(U), the initial configuration's own outgoing-light-time parameter (r_*^(0)(R) = int_{R_L}^{R} dr/(N_0 + v_0) on the T = 0
    slice, defined where N_0 + v_0 > 0), whose compression is the in/out frequency ratio; and the packet's Killing energy w.r.t. h(T = 0),
    <h_0>(u), with kappa_E = d ln<h_0>/du. On a static exterior U - r_s is proportional to e^{-u/(4GE)} while <h_0> is conserved; on an
    exponential segment of a collapse the three exponents agree, and they differ where the initial slice is not stationary.
  kappa(u): with the clock grid u_j and D_j = (U_{j+1} - U_j)/(u_{j+1} - u_j) attributed to m_j = (u_j + u_{j+1})/2,
    kappa^(1)_j = -(ln|D_j| - ln|D_{j-1}|)/(m_j - m_{j-1}) at u_j; kappa^(2)_j the same from the 2-step differences (U_{j+2} - U_j).
    Both are exact for an exponential map on a uniform grid at any step, so |kappa^(1) - kappa^(2)| is the error column (the departure
    from the exponential form over 2 steps and the readout's noise); dU/du by centred differences at both steps; the spread estimate
    sigma_U/sigma_u beside |dU/du|.
  the plateau and the segment: kappa_p is the value about which the longest contiguous run of clocks has |kappa^(1) - kappa_p| <=
    --plateau-frac |kappa_p| and the 2-step error <= --segment-err-frac |kappa_p| (every clock's kappa^(1) is a candidate; the plateau is
    the chosen run's mean; ties by the smaller mean error); the segment is that run's covered interval, its extent (u_last - u_first + the
    step) against T_d(omega), against 2 r_s (e-folds of 1/(2 r_s)) and against 4 pi r_s (thermal periods), with r_s = 2gE at the segment's
    start and the nominal r_s both. Beside it the Hawking band: the longest run within --plateau-frac of 1/(4GE) at the launch record.
    A segment shorter than 2 sigma_u (the probe's width in u) is marked resolution-limited. The same finder runs on kappa_E and on the
    geometric-optics ray map.
  the record table: per record the trapping root of N - |v| on the dense solver nodes (r_nodes, N_nodes, v_nodes) and its slope
    kappa_R = (N - |v|)'(R_t) (the secant of the bracketing nodes and a quadratic fit through --slope-fit-nodes nearest nodes), the site
    fields' slope and the ruling's kappa_R = (N_R/2R)[1 - 8 pi G R^2 (rho_R - J_R)] from the stored densities (page_d13_readout_v1.
    enclosed_source), the engine's own kappa_T, each against 1/(4GE_k); the first record at which each reaches --kappaR-frac of 1/(4GE)
    and its clocks: T_k, and the retarded clock at the launch radius of light leaving --window-launch-over-2gE x 2gE then (by the chart,
    and by the light time in the record's fields): the window's start on the solved history.
  checks per row: the packet's norm at T = 0 and its extremes along the path, the forward re-propagation from T = 0 to the launch record
    with |<psi_L|psi_fwd>| and ||psi_fwd - psi_L|| (--fb-tol, --norm-tol flag the row), the Chebyshev order and accumulated tail bound, the
    geometric-optics ray on the same dispersion traced back through the same records (--ray-compare: the central ray and the Wigner
    average over --ray-wigner-nodes^2 Gauss-Hermite nodes of the launch packet's (x, k) Gaussian; its T = 0 position and wavenumber and
    the ray map's kappa beside the packet's), the join of the stored fields with the vacuum exterior at the launch record, the first-return
    estimate. Optional (--in-vacuum on): the packet's weight on P_-(h(T = 0)), the light channel's initial Killing vacuum (one dense
    eigendecomposition of h(T = 0) per channel), and the Fermi temperature it implies, omega/ln(1/w - 1).
Outputs: raymap_<tag>.json (every scalar), raymap_<tag>.npz (every raw array: the T = 0 density, spectrum and state per probe, the launch
density, the map arrays per channel and frequency, the record table), raymap_<tag>.md; resumable per (u, omega): completed rows are skipped,
rows with a T = 0 state but no forward check get the check only (--force recomputes). No plotting, no mutual information."""
import argparse
import cmath
import json
import math
import os
import sys
import time

import numpy as np
import scipy.optimize as sopt
from numpy.polynomial.hermite import hermgauss

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "raymap_solved_flow_v1.py"


def code_dirs():
    """where page_d13_readout_v1.py (and page_ops_v6.py beside it) lives: this directory, $PAGE_CODE_V6, code_v6 beside or above."""
    c = [HERE]
    if os.environ.get("PAGE_CODE_V6"):
        c += os.environ["PAGE_CODE_V6"].split(os.pathsep)
    c += [os.path.join(HERE, "code_v6"), os.path.join(HERE, "..", "code_v6"), os.path.join(HERE, "..")]
    return [os.path.abspath(x) for x in c]


for _d in code_dirs():
    if os.path.isfile(os.path.join(_d, "page_d13_readout_v1.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
import page_d13_readout_v1 as D13  # noqa: E402  (puts page_ops_v6's directory on sys.path and imports it)
import page_ops_v6 as OPS  # noqa: E402

CD = OPS.CD
RD = OPS.RD
log = D13.log
_fl = D13._fl
_jdefault = D13._jdefault

RULING = {"launch_rs": 16.0, "u_from_rs": 12.0, "u_to_rs": 40.0, "u_step_rs": 0.5, "omega_rs": "0.25,0.5,1,2", "kappas": "1",
          "plateau_frac": 0.1, "segment_err_frac": 0.1, "kappaR_frac": 0.9, "Td_prefactor": 2.0 / 3.0, "Td_inner": 2.0,
          "k_dispersive": 1.0 / 3.0, "k_super": math.pi / 4.0, "window_launch_over_2gE": "1.5,2.0", "form_rs": D13.RULING["form_rs"],
          "join_rs": D13.RULING["join_rs"], "rel_spread": 0.25, "sigma_x_max_rs": 4.0, "sigma_u_rs": 1.0, "k_cut_width_frac": 0.2,
          "cheb_tol": 1e-14, "norm_tol": 1e-10, "fb_tol": 1e-10, "ray_dt": 0.5, "ray_wigner_nodes": 5, "slope_fit_nodes": 4}
KGRID = np.linspace(1e-7, 0.5 * math.pi, 40001)


def _g(x, fmt="%.6g"):
    """a formatted number or '-' for None and nan."""
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return "-"
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


# ------------------------------------------------------------------------------------------------------------ the arguments
def add_arguments(ap):
    g = ap.add_argument_group("the ray map (raymap_solved_flow_v1; the ruling's values as defaults)")
    g.add_argument("--launch-rs", dest="launch_rs", type=float, default=RULING["launch_rs"], help="the launch radius (r_s nominal)")
    g.add_argument("--u-from-rs", dest="u_from_rs", type=float, default=RULING["u_from_rs"], help="the first launch clock u/r_s")
    g.add_argument("--u-to-rs", dest="u_to_rs", type=float, default=RULING["u_to_rs"], help="the last ruled launch clock u/r_s")
    g.add_argument("--u-step-rs", dest="u_step_rs", type=float, default=RULING["u_step_rs"], help="the clock step (r_s); each clock moves to the nearest record")
    g.add_argument("--extend", default="on", choices=["on", "off"], help="extend the clocks beyond --u-to-rs to the first return or the last record")
    g.add_argument("--u-max-rs", dest="u_max_rs", type=float, default=None, help="a hard cap on the extended clocks (r_s)")
    g.add_argument("--omega-rs", dest="omega_rs", default=RULING["omega_rs"], help="the probe frequencies omega r_s (r_s nominal)")
    g.add_argument("--kappas", default=RULING["kappas"], help="the channels (kappa 2 by argument)")
    g.add_argument("--m", type=float, default=0.0, help="the light channel's mass (0: massless)")
    g.add_argument("--width-rule", dest="width_rule", default="rel-spread", choices=["rel-spread", "sigma-u"])
    g.add_argument("--rel-spread", dest="rel_spread", type=float, default=RULING["rel_spread"], help="sigma_k/k_0 under rel-spread")
    g.add_argument("--sigma-x-max-rs", dest="sigma_x_max_rs", type=float, default=RULING["sigma_x_max_rs"], help="the spatial width's cap under rel-spread (r_s)")
    g.add_argument("--sigma-u-rs", dest="sigma_u_rs", type=float, default=RULING["sigma_u_rs"], help="the width in the retarded clock under sigma-u (r_s)")
    g.add_argument("--k-cut-width-frac", dest="k_cut_width_frac", type=float, default=RULING["k_cut_width_frac"], help="the C^1 cutoff's width above k_turn (units of k_0)")
    g.add_argument("--cheb-tol", dest="cheb_tol", type=float, default=RULING["cheb_tol"], help="the Chebyshev tail per record")
    g.add_argument("--chunk", type=int, default=512, help="probe columns per Chebyshev chunk")
    g.add_argument("--plateau-frac", dest="plateau_frac", type=float, default=RULING["plateau_frac"], help="the band about the plateau (and about 1/(4GE))")
    g.add_argument("--segment-err-frac", dest="segment_err_frac", type=float, default=RULING["segment_err_frac"], help="the 2-step error admitted on the segment (fraction of the plateau)")
    g.add_argument("--Td-prefactor", dest="Td_prefactor", type=float, default=RULING["Td_prefactor"], help="T_d = prefactor r_s ln(1/(inner a omega))")
    g.add_argument("--Td-inner", dest="Td_inner", type=float, default=RULING["Td_inner"])
    g.add_argument("--k-dispersive", dest="k_dispersive", type=float, default=RULING["k_dispersive"], help="a k beyond which the mode left the continuum description (the paper's 3 spacings)")
    g.add_argument("--k-super", dest="k_super", type=float, default=RULING["k_super"], help="the superluminal regime's lower edge (pi/4)")
    g.add_argument("--kappaR-frac", dest="kappaR_frac", type=float, default=RULING["kappaR_frac"], help="the window's start: kappa_R at this fraction of 1/(4GE)")
    g.add_argument("--window-launch-over-2gE", dest="window_launch_over_2gE", default=RULING["window_launch_over_2gE"], help="the window's light leaves from these multiples of 2gE")
    g.add_argument("--slope-fit-nodes", dest="slope_fit_nodes", type=int, default=RULING["slope_fit_nodes"], help="dense nodes in the quadratic fit of the root's slope")
    g.add_argument("--form-rs", dest="form_rs", type=float, default=RULING["form_rs"], help="the formation's record clock (the first return's second route)")
    g.add_argument("--join-rs", dest="join_rs", default=RULING["join_rs"], help="the join check's window (r_s)")
    g.add_argument("--ray-compare", dest="ray_compare", default="on", choices=["on", "off"])
    g.add_argument("--ray-dt", dest="ray_dt", type=float, default=RULING["ray_dt"], help="the ray integrator's RK4 step (lattice time)")
    g.add_argument("--ray-wigner-nodes", dest="ray_wigner_nodes", type=int, default=RULING["ray_wigner_nodes"], help="Gauss-Hermite nodes per phase-space direction (1: the central ray only)")
    g.add_argument("--in-vacuum", dest="in_vacuum", default="off", choices=["on", "off"], help="the packet's weight on P_-(h(T = 0)) (one dense eigendecomposition per channel)")
    g.add_argument("--save-states", dest="save_states", default="on", choices=["on", "off"], help="save the T = 0 states in the npz (needed to resume the forward check)")
    g.add_argument("--norm-tol", dest="norm_tol", type=float, default=RULING["norm_tol"])
    g.add_argument("--fb-tol", dest="fb_tol", type=float, default=RULING["fb_tol"])
    g.add_argument("--md-every-rs", dest="md_every_rs", type=float, default=1.0, help="the record table's row spacing in the md")


def cfg_from_args(a):
    g = lambda k, d: getattr(a, k, d)   # noqa: E731
    return {"launch_rs": float(g("launch_rs", RULING["launch_rs"])), "u_from_rs": float(g("u_from_rs", RULING["u_from_rs"])),
            "u_to_rs": float(g("u_to_rs", RULING["u_to_rs"])), "u_step_rs": float(g("u_step_rs", RULING["u_step_rs"])), "extend": g("extend", "on") == "on",
            "u_max_rs": g("u_max_rs", None), "omega_rs": _fl(g("omega_rs", RULING["omega_rs"])), "kappas": [int(x) for x in str(g("kappas", "1")).split(",") if x.strip()],
            "m": float(g("m", 0.0)), "width_rule": g("width_rule", "rel-spread"), "rel_spread": float(g("rel_spread", RULING["rel_spread"])),
            "sigma_x_max_rs": float(g("sigma_x_max_rs", RULING["sigma_x_max_rs"])), "sigma_u_rs": float(g("sigma_u_rs", RULING["sigma_u_rs"])),
            "k_cut_width_frac": float(g("k_cut_width_frac", RULING["k_cut_width_frac"])), "cheb_tol": float(g("cheb_tol", RULING["cheb_tol"])), "chunk": int(g("chunk", 512)),
            "plateau_frac": float(g("plateau_frac", RULING["plateau_frac"])), "segment_err_frac": float(g("segment_err_frac", RULING["segment_err_frac"])),
            "Td_prefactor": float(g("Td_prefactor", RULING["Td_prefactor"])), "Td_inner": float(g("Td_inner", RULING["Td_inner"])),
            "k_dispersive": float(g("k_dispersive", RULING["k_dispersive"])), "k_super": float(g("k_super", RULING["k_super"])),
            "kappaR_frac": float(g("kappaR_frac", RULING["kappaR_frac"])), "window_launch_over_2gE": _fl(g("window_launch_over_2gE", RULING["window_launch_over_2gE"])),
            "slope_fit_nodes": int(g("slope_fit_nodes", RULING["slope_fit_nodes"])), "form_rs": float(g("form_rs", RULING["form_rs"])), "join_rs": _fl(g("join_rs", RULING["join_rs"])),
            "ray_compare": g("ray_compare", "on") == "on", "ray_dt": float(g("ray_dt", RULING["ray_dt"])), "ray_wigner_nodes": int(g("ray_wigner_nodes", RULING["ray_wigner_nodes"])),
            "in_vacuum": g("in_vacuum", "off") == "on", "save_states": g("save_states", "on") == "on", "norm_tol": float(g("norm_tol", RULING["norm_tol"])),
            "fb_tol": float(g("fb_tol", RULING["fb_tol"])), "md_every_rs": float(g("md_every_rs", 1.0)), "force": bool(g("force", False)), "ball_rs": None}


# -------------------------------------------------------------------------------------------------------------- the records
def load_records(path):
    """the records' json: a list of record dicts, or a dict holding the list under some key and the run's args; (records, args)."""
    J = json.load(open(path))
    args = None; recs = None
    if isinstance(J, list):
        recs = J
    elif isinstance(J, dict):
        args = J.get("args") if isinstance(J.get("args"), dict) else None
        for val in J.values():
            if isinstance(val, list) and val and isinstance(val[0], dict) and "T_over_rs" in val[0]:
                recs = val
                break
    if recs is None:
        raise ValueError("no list of records (dicts with T_over_rs) found in %s" % path)
    return recs, args


def nodes_full(row):
    """one record's dense node row (object dtype: floats or None) as a float array with nan for the missing entries."""
    try:
        return np.asarray(row, dtype=float)
    except (TypeError, ValueError):
        return np.array([float(e) if (e is not None and np.ndim(e) == 0) else np.nan for e in row], float)


class FieldsBase:
    """the common interface of the fields: T, E, R (the chain), g, tau, rs_nom, label, m; fields(k) -> (N, v) on the chain; two_gE(k);
    dt(k); h_csr(kappa, N, v); banded(kappa, h, device)."""
    m = 0.0

    def __len__(self):
        return len(self.T)

    def two_gE(self, k):
        return 2.0 * self.g * float(self.E[k])

    def dt(self, k):
        return float(self.T[k + 1] - self.T[k])

    def chain(self, kappa):
        if kappa not in self._chain:
            self._chain[kappa] = OPS.chain_sparse(self.R, self.m, float(kappa))
        return self._chain[kappa]

    def h_csr(self, kappa, N, v):
        h0, Q = self.chain(kappa)
        return OPS.clocked_csr(h0, Q, N, v)

    def banded(self, kappa, h, device):
        return OPS.Banded(h, device)


class OwnFields(FieldsBase):
    """the stored records on the light channel's chain with the vacuum exterior of the record's charge appended (the own construction),
    plus the records' json, the dense nodes and the densities for the record table."""
    kind = "own"

    def __init__(self, record_json, fields_npz, nsites, g=None, fields_key="applied", m=0.0):
        self._chain = {}
        self.m = float(m)
        self.path_record = record_json; self.path_fields = fields_npz
        self.notes = []
        Z = np.load(fields_npz, allow_pickle=True)
        self.R_rec = np.asarray(Z["R"], float)
        self.T = np.asarray(Z["T"], float)
        self.g = float(Z["g"]) if (g is None and "g" in Z.files) else float(g if g is not None else 0.25)
        keyN, keyv = ("N_applied", "v_applied") if fields_key == "applied" else ("N", "v")
        if keyN not in Z.files or keyv not in Z.files:
            self.notes.append("the fields key %s is absent in %s: the stored N, v are used" % (fields_key, fields_npz))
            keyN, keyv = "N", "v"
        self.fields_key = keyN
        self.N_rec = np.asarray(Z[keyN], float); self.v_rec = np.asarray(Z[keyv], float)
        self.eps = np.asarray(Z["eps_tot"], float) if "eps_tot" in Z.files else None
        self.J = np.asarray(Z["J_tot"], float) if "J_tot" in Z.files else None
        self.nodes = {k_: Z[k_] for k_ in ("r_nodes", "N_nodes", "v_nodes") if k_ in Z.files}
        self.tau = float(np.median(np.diff(self.T))) if len(self.T) > 1 else 8.0
        self.label = str(Z["label"]) if "label" in Z.files else ""
        recs, args = load_records(record_json)
        self.records = recs; self.args = args or {}
        Trec = np.array([float(r_.get("T", np.nan)) for r_ in recs])
        if not np.all(np.isfinite(Trec)):
            Trec = np.array([float(r_["T_over_rs"]) * float(self.args.get("rs", 64.0)) for r_ in recs])
        self.rec_index = np.array([int(np.argmin(np.abs(Trec - t))) for t in self.T])
        dT = float(np.max(np.abs(Trec[self.rec_index] - self.T)))
        if dT > 1e-6:
            self.notes.append("the records' T and the fields' T differ by up to %.3e" % dT)
        self.E = np.array([float(recs[i]["E"]) for i in self.rec_index])
        self.rs_nom = float(self.args["rs"]) if "rs" in self.args else None
        self.nsites = int(nsites)
        n_rec = len(self.R_rec)
        if self.nsites < n_rec:
            raise ValueError("nsites %d below the record's chain %d" % (self.nsites, n_rec))
        self.R = np.arange(self.nsites, dtype=float) + float(self.R_rec[0])
        self.R_defect = float(np.max(np.abs(self.R[:n_rec] - self.R_rec)))
        if self.R_defect > 1e-9:
            self.notes.append("the chain's R differs from the record's by up to %.2e" % self.R_defect)

    def record(self, k):
        return self.records[int(self.rec_index[k])]

    def fields(self, k):
        N = np.ones(self.nsites); v = -np.sqrt(self.two_gE(k) / self.R)
        n = len(self.R_rec)
        N[:n] = self.N_rec[k]; v[:n] = self.v_rec[k]
        return N, v

    def join_check(self, k, lo, hi):
        return D13.join_departure(self.N_rec[k], self.v_rec[k], self.R_rec, self.E[k], self.g, lo, hi)

    def dense_nodes(self, k):
        """(r, N, v) on the record's dense solver nodes, sorted by r, the missing entries dropped jointly; None without nodes."""
        if not all(k_ in self.nodes for k_ in ("r_nodes", "N_nodes", "v_nodes")):
            return None
        r = nodes_full(self.nodes["r_nodes"][k]); Nn = nodes_full(self.nodes["N_nodes"][k]); vn = nodes_full(self.nodes["v_nodes"][k])
        n = min(len(r), len(Nn), len(vn))
        r, Nn, vn = r[:n], Nn[:n], vn[:n]
        ok = np.isfinite(r) & np.isfinite(Nn) & np.isfinite(vn)
        r, Nn, vn = r[ok], Nn[ok], vn[ok]
        o = np.argsort(r)
        return r[o], Nn[o], vn[o]


class V6MFields(FieldsBase):
    """page_history_v6m's History and ChannelOps (the light channel's own construction), with the own backend beside for the records."""
    kind = "v6m"

    def __init__(self, record_json, fields_npz, nsites, g, fields_key, m, device, own):
        import page_history_v6m as V6M
        self.V6M = V6M
        self._chain = {}
        self.m = float(m)
        self.H = V6M.History(record_json, fields_npz, int(nsites), g, fields_key)
        self.T = np.asarray(self.H.T, float); self.E = np.asarray(self.H.E, float); self.R = np.asarray(self.H.R, float)
        self.g = float(self.H.g); self.tau = float(self.H.tau)
        self.rs_nom = getattr(self.H, "rs_nom", None); self.label = getattr(self.H, "label", "") or ""
        self.device = device; self.ops = {}
        self.own = own; self.records = own.records; self.args = own.args; self.nsites = int(nsites); self.notes = []
        self.rec_index = own.rec_index; self.R_rec = own.R_rec; self.N_rec = own.N_rec; self.v_rec = own.v_rec; self.eps = own.eps; self.J = own.J
        if self.m != 0.0:
            self.notes.append("--m %g is not passed to page_history_v6m's ChannelOps (its own mass applies)" % self.m)

    def record(self, k):
        return self.own.record(k)

    def fields(self, k):
        N, v = self.H.fields(k)
        return np.asarray(N, float), np.asarray(v, float)

    def _ops(self, kappa):
        if kappa not in self.ops:
            self.ops[kappa] = self.V6M.ChannelOps(int(kappa), self.H.R, self.device)
        return self.ops[kappa]

    def h_csr(self, kappa, N, v):
        return self._ops(kappa).h_csr(N, v)

    def banded(self, kappa, h, device):
        return self._ops(kappa).banded(h)

    def join_check(self, k, lo, hi):
        return self.H.join_check(k, lo, hi)

    def dense_nodes(self, k):
        return self.own.dense_nodes(k)


def make_fields(a, device, kappa_check=1):
    """the own backend always (the records), page_history_v6m's when importable and asked; the two compared at 3 records."""
    own = OwnFields(a.record, a.fields, a.nsites, a.g, a.fields_key, a.m)
    info = {"own_notes": own.notes, "fields_key": own.fields_key, "nsites": own.nsites, "record_chain": int(len(own.R_rec)), "g": own.g, "tau": own.tau,
            "rs_nom_args": own.rs_nom, "source": "own", "v6m_error": None, "compare": None}
    F = own
    if a.fields_source in ("auto", "v6m"):
        try:
            F = V6MFields(a.record, a.fields, a.nsites, a.g, a.fields_key, a.m, device, own)
            info["source"] = "v6m"
            cmp_ = []
            for k in sorted(set([0, len(own) // 2, len(own) - 1])):
                N1, v1 = own.fields(k); N2, v2 = F.fields(k)
                row = {"k": k, "max_dN": float(np.max(np.abs(N1 - N2))), "max_dv": float(np.max(np.abs(v1 - v2)))}
                h1 = own.h_csr(kappa_check, N1, v1); h2 = F.h_csr(kappa_check, N2, v2)
                dh = (h1 - h2).tocoo()
                row["max_dh"] = float(np.max(np.abs(dh.data))) if dh.nnz else 0.0
                cmp_.append(row)
            info["compare"] = cmp_
            log("fields: page_history_v6m's History and ChannelOps (own construction compared at records %s: max |dN| %.2e, |dv| %.2e, |dh| %.2e)" % (
                [c_["k"] for c_ in cmp_], max(c_["max_dN"] for c_ in cmp_), max(c_["max_dv"] for c_ in cmp_), max(c_["max_dh"] for c_ in cmp_)))
        except Exception as ex:                                            # noqa: BLE001
            info["v6m_error"] = "%s: %s" % (type(ex).__name__, ex)
            F = own
            if a.fields_source == "v6m":
                raise
            log("fields: page_history_v6m not usable (%s); the own construction is used (the stored fields with the vacuum exterior appended)" % info["v6m_error"])
    else:
        log("fields: the own construction (--fields-source own)")
    info["v6m_notes"] = getattr(F, "notes", [])
    return F, own, info


# -------------------------------------------------------------------------------------------------------- the launch clocks
def clock_records(F, R_L, rs, u_list_rs):
    """the record nearest each requested clock u at the launch radius: T_k = u r_s + r_*(R_L; 2gE_k), iterated for the record's own charge;
    [(k, u_actual/r_s)] without repeated records."""
    T = np.asarray(F.T, float)
    out = []; seen = set()
    for u in u_list_rs:
        k = 0
        rstar = T[0] - OPS.chart_retarded_time(T[0], F.two_gE(0), R_L)
        for _ in range(4):
            k2 = int(np.argmin(np.abs(T - (u * rs + rstar))))
            rstar = T[k2] - OPS.chart_retarded_time(T[k2], F.two_gE(k2), R_L)
            if k2 == k:
                break
            k = k2
        if k in seen:
            continue
        seen.add(k)
        out.append((k, OPS.chart_retarded_time(T[k], F.two_gE(k), R_L) / rs))
    return out


# -------------------------------------------------------------------------------------------------------------- the packet
def local_branch(N, v, R, kappa, m, k):
    """omega_+(k), v_+(k), E(k) of the local positive branch at (N, v, R)."""
    k = np.asarray(k, float)
    E = np.sqrt(np.sin(k) ** 2 + (kappa / R) ** 2 + m * m)
    om = N * E + 0.5 * v * np.sin(2.0 * k)
    vg = N * np.sin(k) * np.cos(k) / E + v * np.cos(2.0 * k)
    return om, vg, E


def outgoing_branch(N, v, R, kappa, m, omega):
    """k_turn (the smallest k > 0 with v_+ > 0), k_peak (the outgoing branch's maximum), k_0 with omega_+(k_0) = omega."""
    om, vg, _ = local_branch(N, v, R, kappa, m, KGRID)
    pos = np.nonzero(vg > 0)[0]
    if len(pos) == 0:
        raise ValueError("no outgoing wavenumber at the launch site (v_+ <= 0 everywhere)")
    i0 = int(pos[0]); i1 = i0
    while i1 + 1 < len(KGRID) and vg[i1 + 1] > 0:
        i1 += 1
    k_turn, k_peak = float(KGRID[i0]), float(KGRID[i1])
    if not (om[i0] <= omega <= om[i1]):
        raise ValueError("omega %.4e outside the outgoing branch [%.4e, %.4e] at the launch site" % (omega, om[i0], om[i1]))
    k0 = float(sopt.brentq(lambda kk: float(local_branch(N, v, R, kappa, m, kk)[0]) - omega, k_turn, k_peak, xtol=1e-14, maxiter=200))
    return {"k_0": k0, "k_turn": k_turn, "k_peak": k_peak, "omega_turn": float(om[i0]), "omega_peak": float(om[i1]),
            "v_g_k0": float(local_branch(N, v, R, kappa, m, k0)[1])}


_ramp = np.vectorize(OPS.ramp_c1)


def packet_amplitudes(kgrid, k_c, sigma_k, k_turn, k_peak, cut_width):
    """the Gaussian amplitude about k_c, zero at and below k_turn through a C^1 ramp of width cut_width, zero above k_peak."""
    w = np.exp(-(kgrid - k_c) ** 2 / (2.0 * sigma_k ** 2))
    w = w * _ramp((kgrid - k_turn) / max(cut_width, 1e-300))
    w[kgrid > k_peak] = 0.0
    return w


def match_energy(om_grid, kgrid, sigma_k, k_turn, k_peak, cut_width, omega, k_0):
    """the centre k_c at which the spectral mean of omega_+ over the amplitudes squared equals omega (brentq on [k_turn, k_peak]);
    k_0 with matched = False when no centre in the bracket does."""
    def mean_om(kc):
        w = packet_amplitudes(kgrid, kc, sigma_k, k_turn, k_peak, cut_width); w2 = w * w
        s = float(np.sum(w2))
        return float(np.sum(w2 * om_grid) / s) if s > 0 else float("nan")
    f = lambda kc: mean_om(kc) - omega   # noqa: E731
    fa, fb = f(k_turn), f(k_peak)
    if math.isfinite(fa) and math.isfinite(fb) and fa <= 0.0 <= fb:
        kc = float(sopt.brentq(f, k_turn, k_peak, xtol=1e-13, maxiter=200))
        return kc, True, mean_om(kc)
    return k_0, False, mean_om(k_0)


def build_packet(F, k_rec, kappa, omega, n_L, cfg, rs, device):
    """the unit outgoing packet at the launch site at record k_rec; (psi (n2,) device, info, (N, v) of the record)."""
    N, v = F.fields(k_rec)
    R = np.asarray(F.R, float); n = len(R)
    R_L = float(R[n_L]); N_L, v_L = float(N[n_L]), float(v[n_L])
    br = outgoing_branch(N_L, v_L, R_L, kappa, F.m, omega)
    k0 = br["k_0"]
    capped = False
    if cfg["width_rule"] == "rel-spread":
        sigma_k = cfg["rel_spread"] * k0
        sigma_x = 1.0 / sigma_k
        if sigma_x > cfg["sigma_x_max_rs"] * rs:
            sigma_x = cfg["sigma_x_max_rs"] * rs; sigma_k = 1.0 / sigma_x; capped = True
    else:
        sigma_x = cfg["sigma_u_rs"] * rs * abs(br["v_g_k0"]); sigma_k = 1.0 / sigma_x
    kgrid = 2.0 * math.pi * np.arange(1, n // 2 + 1) / n
    om_grid = local_branch(N_L, v_L, R_L, kappa, F.m, kgrid)[0]
    cut_width = cfg["k_cut_width_frac"] * k0
    k_c, matched, mean_om = match_energy(om_grid, kgrid, sigma_k, br["k_turn"], br["k_peak"], cut_width, omega, k0)
    w = packet_amplitudes(kgrid, k_c, sigma_k, br["k_turn"], br["k_peak"], cut_width)
    idx = np.nonzero(w > 1e-16 * float(w.max()))[0]
    km = kgrid[idx]; wm = w[idx]
    x = R - R_L
    phase = np.exp(1j * np.outer(x, km))
    E = np.sqrt(np.sin(km)[None, :] ** 2 + (kappa / R[:, None]) ** 2 + F.m ** 2)
    nrm = np.sqrt(2.0 * E * (E + F.m))
    chi0 = (E + F.m) / nrm
    chi1 = (np.sin(km)[None, :] + 1j * kappa / R[:, None]) / nrm
    psi0 = (phase * chi0) @ wm; psi1 = (phase * chi1) @ wm
    psi = np.empty(2 * n, complex); psi[0::2] = psi0; psi[1::2] = psi1
    psi /= np.linalg.norm(psi)
    info = dict(br, R_L=R_L, N_L=N_L, v_L=v_L, v_pg_L=-math.sqrt(F.two_gE(k_rec) / R_L), k_c=k_c, energy_matched=bool(matched), energy_mean_branch=mean_om,
                sigma_k=sigma_k, sigma_x_rule=sigma_x, width_rule=cfg["width_rule"], rel_spread=cfg["rel_spread"], capped=bool(capped), n_k_components=int(len(idx)),
                k_cut_width=cut_width)
    return torch.as_tensor(psi, dtype=CD, device=device), info, (N, v)


def spectrum_stats(psi, n, k_super, k_disp):
    """the packet's wavenumber spectrum (both components' FFT summed, k in (-pi, pi]): the mean |k|, the signed mean, the spread, the
    weight above k_super and k_disp, the weight at k < 0; (dict, P fftshifted, k fftshifted)."""
    Z = torch.fft.fft(psi.view(n, 2), dim=0)
    P = (Z.abs() ** 2).sum(dim=1)
    kf = 2.0 * math.pi * torch.fft.fftfreq(n, d=1.0, dtype=RD, device=psi.device)
    tot = float(P.sum())
    ak = kf.abs()
    d = {"k_mean_abs": float((P * ak).sum() / tot), "k_mean_signed": float((P * kf).sum() / tot), "k_rms": math.sqrt(max(float((P * kf * kf).sum() / tot), 0.0)),
         "frac_k_super": float(P[ak > k_super].sum() / tot), "frac_k_dispersive": float(P[ak > k_disp].sum() / tot), "frac_k_negative": float(P[kf < 0].sum() / tot)}
    d["k_spread"] = math.sqrt(max(d["k_rms"] ** 2 - d["k_mean_signed"] ** 2, 0.0))
    d["reduced_wavelength"] = (1.0 / d["k_mean_abs"]) if d["k_mean_abs"] > 0 else None
    return d, torch.fft.fftshift(P).cpu().numpy(), torch.fft.fftshift(kf).cpu().numpy()


def packet_diagnostics(psi, R_t, Bh, Vb, n, k_super, k_disp):
    """rho per site, the centroid and spread, <h> and its spread, <i[h, X]> (Vb, optional), the one-site phase advance and the spectrum."""
    rho = (psi.abs() ** 2).view(n, 2).sum(dim=1)
    w = float(rho.sum())
    cen = float((R_t * rho).sum() / w)
    var = float((R_t * R_t * rho).sum() / w) - cen * cen
    hpsi = Bh.apply(psi[:, None])[:, 0]
    Eh = float(torch.vdot(psi, hpsi).real) / w
    Eh2 = float(torch.vdot(hpsi, hpsi).real) / w
    vg = (float(torch.vdot(psi, Vb.apply(psi[:, None])[:, 0]).real) / w) if Vb is not None else None
    p2 = psi.view(n, 2)
    c1 = complex(torch.sum(p2[:-1].conj() * p2[1:]))
    sp, Ps, ks = spectrum_stats(psi, n, k_super, k_disp)
    d = {"norm": math.sqrt(w), "centroid": cen, "sigma_x": math.sqrt(max(var, 0.0)), "energy_h": Eh, "energy_spread_h": math.sqrt(max(Eh2 - Eh * Eh, 0.0)),
         "group_velocity": vg, "k_corr": cmath.phase(c1), "coherence": abs(c1) / w}
    d.update(sp)
    return d, rho.cpu().numpy(), Ps, ks


# ----------------------------------------------------------------------------------------------------- the T = 0 readouts
def slice_chart(N0, v0, R, n_L):
    """r_*^(0)(R) = int_{R_L}^{R} dr/(N_0 + v_0) on the first slice (trapezoid over the sites), on the connected run about the launch site
    where N_0 + v_0 > 0; (r_* with nan outside the run, c_0 = N_0 + v_0, the run's radii)."""
    c = np.asarray(N0, float) + np.asarray(v0, float)
    n = len(R)
    ok = c > 0
    lo = n_L
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = n_L
    while hi + 1 < n and ok[hi + 1]:
        hi += 1
    rstar = np.full(n, np.nan)
    rr = np.asarray(R, float)[lo: hi + 1]; ii = 1.0 / c[lo: hi + 1]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (ii[1:] + ii[:-1]) * np.diff(rr))])
    cum = cum - cum[n_L - lo]
    rstar[lo: hi + 1] = cum
    return rstar, c, (float(R[lo]), float(R[hi]))


def readouts_T0(F, kappa, X, probes, cfg, rs, n_L, device, Vneg=None):
    """the T = 0 readouts of the columns X (one per probe): the density and its moments (U, sigma_U), the weights inside 2gE_0, the ball
    and beyond the launch radius, the spectrum, <h_0> and its spread, U_0 (the slice's outgoing-light-time parameter at U), the weight on
    P_-(h_0) when Vneg is given; (list of dicts, list of (rho, P) arrays)."""
    N0, v0 = F.fields(0)
    h0 = F.h_csr(kappa, N0, v0); B0 = F.banded(kappa, h0, device)
    R = np.asarray(F.R, float); n = len(R)
    R_t = torch.as_tensor(R, dtype=RD, device=device)
    rstar0, c0, run = slice_chart(N0, v0, R, n_L)
    fin = np.isfinite(rstar0)
    two0 = F.two_gE(0)
    ball = cfg["ball_rs"] * rs if cfg.get("ball_rs") else None
    R_L = float(R[n_L])
    out = []; arrays = []
    for i, p in enumerate(probes):
        psi = X[:, i]
        d, rho, Ps, ks = packet_diagnostics(psi, R_t, B0, None, n, cfg["k_super"], cfg["k_dispersive"])
        U = d["centroid"]
        r = {"U": U, "U_over_rs": U / rs, "sigma_U": d["sigma_x"], "sigma_U_over_rs": d["sigma_x"] / rs, "norm_T0": d["norm"],
             "weight_inside_2gE0": float(np.sum(rho[R <= two0]) / d["norm"] ** 2), "weight_inside_ball": (float(np.sum(rho[R <= ball]) / d["norm"] ** 2) if ball else None),
             "weight_beyond_RL": float(np.sum(rho[R > R_L]) / d["norm"] ** 2), "energy_h0": d["energy_h"], "energy_spread_h0": d["energy_spread_h"],
             "energy_h0_over_omega": d["energy_h"] / p["omega"], "k_mean_abs": d["k_mean_abs"], "k_mean_signed": d["k_mean_signed"], "k_spread": d["k_spread"],
             "k_corr": d["k_corr"], "coherence": d["coherence"], "frac_k_super": d["frac_k_super"], "frac_k_dispersive": d["frac_k_dispersive"],
             "frac_k_negative": d["frac_k_negative"], "reduced_wavelength": d["reduced_wavelength"], "two_gE_0": two0,
             "U0_chart": (float(-np.interp(U, R[fin], rstar0[fin])) if (fin.any() and run[0] <= U <= run[1]) else None),
             "c0_at_U": float(np.interp(U, R, c0)), "slice_chart_run": list(run), "w_neg_in": None, "T_eff_fermi": None, "T_eff_fermi_over_TH": None}
        if Vneg is not None:
            wneg = float((Vneg.mH @ psi[:, None]).abs().pow(2).sum()) / d["norm"] ** 2
            r["w_neg_in"] = wneg
            if 0.0 < wneg < 0.5:
                r["T_eff_fermi"] = p["omega"] / math.log(1.0 / wneg - 1.0)
                r["T_eff_fermi_over_TH"] = r["T_eff_fermi"] * 4.0 * math.pi * p["two_gE"]
        out.append(r); arrays.append((rho, Ps, ks))
    return out, arrays


# ---------------------------------------------------------------------------------------------- the two propagations
def backward_pass(F, kappa, probes, cfg, device):
    """every probe's launch state exp(+i dt h) through the records k - 1, ..., 0 (inserted at its launch record, all probes in one column
    set); returns X (n2 x P at T_0, in the probes' order) and per-probe norm extremes, Chebyshev order and tail along its own records."""
    P = len(probes); n2 = 2 * len(F.R)
    kmax = max(p["k"] for p in probes)
    X = torch.zeros((n2, 0), dtype=CD, device=device); cols = []
    nmin = np.ones(P); nmax = np.ones(P)
    tails = np.zeros(max(kmax, 1)); Kmax = 0
    t0 = time.time()
    for krec in range(kmax - 1, -1, -1):
        new = [i for i in range(P) if probes[i]["k"] == krec + 1]
        if new:
            X = torch.cat([X] + [probes[i]["psiL"][:, None] for i in new], dim=1); cols += new
        if X.shape[1] == 0:
            continue
        N, v = F.fields(krec)
        Bh = F.banded(kappa, F.h_csr(kappa, N, v), device)
        X, K_, tl = D13.propagate(Bh, X, -F.dt(krec), cfg["cheb_tol"], cfg["chunk"])
        Kmax = max(Kmax, K_); tails[krec] = tl
        nr = torch.linalg.norm(X, dim=0).cpu().numpy()
        for c_, i in enumerate(cols):
            nmin[i] = min(nmin[i], nr[c_]); nmax[i] = max(nmax[i], nr[c_])
        if krec % 100 == 0:
            log("backward: record %d of %d done, %d columns, Chebyshev order %d, %.0f s" % (krec, kmax, X.shape[1], K_, time.time() - t0))
    new = [i for i in range(P) if probes[i]["k"] == 0]
    if new:
        X = torch.cat([X] + [probes[i]["psiL"][:, None] for i in new], dim=1); cols += new
    perm = torch.as_tensor(np.argsort(np.asarray(cols)), dtype=torch.int64, device=device)
    X = X[:, perm]
    tail_acc = np.array([float(np.sum(tails[: probes[i]["k"]])) for i in range(P)])
    return X, {"norm_min": nmin, "norm_max": nmax, "cheb_K_max": Kmax, "cheb_tail_acc": tail_acc, "records_max": kmax, "seconds": time.time() - t0}


def forward_pass(F, kappa, probes, X0, cfg, device):
    """the T_0 states exp(-i dt h) through the records 0, ..., k - 1 back to each probe's launch record, compared with the launch packet."""
    P = len(probes); kmax = max(p["k"] for p in probes)
    X = X0.clone(); cols = list(range(P)); res = [None] * P
    tails = []

    def finish(i, vec):
        psiL = probes[i]["psiL"]
        ov = complex(torch.vdot(psiL, vec)); nf = float(torch.linalg.norm(vec)); nl = float(torch.linalg.norm(psiL))
        res[i] = {"overlap_abs": abs(ov) / (nf * nl), "overlap_defect": 1.0 - abs(ov) / (nf * nl), "defect_norm": float(torch.linalg.norm(vec - psiL)),
                  "norm_fwd": nf, "phase": cmath.phase(ov), "cheb_tail_acc_fwd": float(np.sum(tails))}
    done = [c for c in range(len(cols)) if probes[cols[c]]["k"] == 0]
    for c in done:
        finish(cols[c], X[:, c])
    if done:
        keep = [c for c in range(len(cols)) if c not in done]
        X = X[:, torch.as_tensor(keep, dtype=torch.int64, device=device)]; cols = [cols[c] for c in keep]
    t0 = time.time()
    for krec in range(0, kmax):
        if X.shape[1] == 0:
            break
        N, v = F.fields(krec)
        Bh = F.banded(kappa, F.h_csr(kappa, N, v), device)
        X, K_, tl = D13.propagate(Bh, X, F.dt(krec), cfg["cheb_tol"], cfg["chunk"])
        tails.append(tl)
        done = [c for c in range(len(cols)) if probes[cols[c]]["k"] == krec + 1]
        for c in done:
            finish(cols[c], X[:, c])
        if done:
            keep = [c for c in range(len(cols)) if c not in done]
            X = X[:, torch.as_tensor(keep, dtype=torch.int64, device=device)]; cols = [cols[c] for c in keep]
        if krec % 100 == 0:
            log("forward: record %d of %d done, %d columns left, %.0f s" % (krec, kmax, X.shape[1], time.time() - t0))
    return res


# ---------------------------------------------------------------------------------------------- the geometric-optics ray
def wigner_nodes(x0, k0, sigma_x, sigma_k, nn):
    """Gauss-Hermite nodes and weights of the launch packet's (x, k) Gaussian (std sigma_x, sigma_k, no correlation); nn = 1: the centre."""
    xi, wi = hermgauss(max(1, int(nn)))
    xs = x0 + math.sqrt(2.0) * sigma_x * xi; ks = k0 + math.sqrt(2.0) * sigma_k * xi
    Xg, Kg = np.meshgrid(xs, ks, indexing="ij"); W = np.outer(wi, wi) / math.pi
    return Xg.ravel(), Kg.ravel(), W.ravel()


class RayTracer:
    """geometric optics on the lattice dispersion omega(k, r; T) = N sqrt(sin^2 k + (kappa/r)^2 + m^2) + v sin(2k)/2 with the record's
    fields (constant in T over the record, linear in r between the sites, radial derivatives by np.gradient): dr/dT = d omega/dk,
    dk/dT = -d omega/dr, RK4 at a fixed step dividing each record, backward from every launch (T_k, r, k) to the first record in one
    vectorized pass (rays activate at their launch record)."""

    def __init__(self, F, kappa, dt):
        self.F = F; self.kappa = float(kappa); self.m = float(F.m); self.dt = float(dt)

    def rhs(self, r, k, N, v, dN, dv, R):
        Nr = np.interp(r, R, N); vr = np.interp(r, R, v); dNr = np.interp(r, R, dN); dvr = np.interp(r, R, dv)
        s, c = np.sin(k), np.cos(k)
        E = np.sqrt(s * s + (self.kappa / r) ** 2 + self.m ** 2)
        drdT = Nr * s * c / E + vr * np.cos(2.0 * k)
        dkdT = -(dNr * E - Nr * self.kappa ** 2 / (r ** 3 * E) + 0.5 * dvr * np.sin(2.0 * k))
        return drdT, dkdT

    def trace_back(self, k_launch, r_launch, k_wave):
        kl = np.asarray(k_launch, int); r = np.asarray(r_launch, float).copy(); k = np.asarray(k_wave, float).copy()
        L = len(kl); active = np.zeros(L, bool); lost = np.zeros(L, bool)
        R = np.asarray(self.F.R, float)
        kmax = int(kl.max()) if L else 0
        for krec in range(kmax - 1, -1, -1):
            active |= (kl == krec + 1)
            idx = np.nonzero(active & ~lost)[0]
            if len(idx) == 0:
                continue
            N, v = self.F.fields(krec); dN = np.gradient(N, R); dv = np.gradient(v, R)
            dtk = self.F.dt(krec); nsub = max(1, int(round(dtk / self.dt))); h = -dtk / nsub
            rr, kk = r[idx], k[idx]
            for _ in range(nsub):
                a1, b1 = self.rhs(rr, kk, N, v, dN, dv, R)
                a2, b2 = self.rhs(rr + 0.5 * h * a1, kk + 0.5 * h * b1, N, v, dN, dv, R)
                a3, b3 = self.rhs(rr + 0.5 * h * a2, kk + 0.5 * h * b2, N, v, dN, dv, R)
                a4, b4 = self.rhs(rr + h * a3, kk + h * b3, N, v, dN, dv, R)
                rr = rr + h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4); kk = kk + h / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
                bad = ~np.isfinite(rr) | ~np.isfinite(kk) | (rr < R[0]) | (rr > R[-1])
                if bad.any():
                    lost[idx[bad]] = True
                    rr = np.where(bad, np.clip(np.nan_to_num(rr, nan=R[0]), R[0], R[-1]), rr); kk = np.where(bad, np.nan_to_num(kk), kk)
            r[idx] = rr; k[idx] = kk
        return r, k, lost


def ray_compare(F, kappa, probes, cfg, rs):
    """per probe the central ray and the Wigner average over the launch packet's (x, k) Gaussian (measured widths): T = 0 position and
    wavenumber, the average's spread and the lost fraction."""
    nn = max(1, int(cfg["ray_wigner_nodes"]))
    kl = []; rl = []; kw = []; wl = []; owner = []
    for i, p in enumerate(probes):
        L = p["launch"]
        xs, ks, ws = wigner_nodes(L["centroid"], L["k_mean_signed"], L["sigma_x"], L["k_spread"], nn)
        kl += [p["k"]] * len(xs); rl += list(xs); kw += list(ks); wl += list(ws); owner += [i] * len(xs)
    tr = RayTracer(F, kappa, cfg["ray_dt"])
    t0 = time.time()
    r0, k0, lost = tr.trace_back(kl, rl, kw)
    owner = np.asarray(owner); wl = np.asarray(wl)
    out = []
    centre = (nn * nn) // 2
    for i in range(len(probes)):
        sel = np.nonzero(owner == i)[0]
        ok = sel[~lost[sel]]
        w = wl[ok]; sw = float(np.sum(w)) if len(ok) else 0.0
        d = {"nodes": int(nn * nn), "lost_frac": float(np.sum(wl[sel][lost[sel]])) if len(sel) else None, "dt": cfg["ray_dt"]}
        if sw > 0:
            Um = float(np.sum(w * r0[ok]) / sw); km = float(np.sum(w * k0[ok]) / sw)
            d.update({"U_ray": Um, "U_ray_over_rs": Um / rs, "sigma_ray": math.sqrt(max(float(np.sum(w * r0[ok] ** 2) / sw) - Um * Um, 0.0)), "k_ray": km,
                      "k_ray_max": float(np.max(np.abs(k0[ok])))})
        else:
            d.update({"U_ray": None, "U_ray_over_rs": None, "sigma_ray": None, "k_ray": None, "k_ray_max": None})
        c = sel[centre] if (nn % 2 == 1 and len(sel) > centre) else None
        d["U_ray_central"] = float(r0[c]) if (c is not None and not lost[c]) else None
        d["k_ray_central"] = float(k0[c]) if (c is not None and not lost[c]) else None
        out.append(d)
    log("ray compare: %d rays (%d probes x %d nodes) traced back in %.1f s, %d lost" % (len(kl), len(probes), nn * nn, time.time() - t0, int(lost.sum())))
    return out


# ------------------------------------------------------------------------------------------------- the map's derivatives
def map_derivatives(u, U):
    """dU/du by centred differences at 1 and 2 steps, kappa by the log-difference of first differences at 1 and 2 steps (exact for an
    exponential map on a uniform grid at either step), their difference as the error; nan where undefined."""
    u = np.asarray(u, float); U = np.asarray(U, float); n = len(u)
    out = {k_: np.full(n, np.nan) for k_ in ("dUdu1", "dUdu2", "kappa1", "kappa2", "err", "sign_change")}
    if n < 3:
        return out
    D1 = np.diff(U) / np.diff(u); m1 = 0.5 * (u[:-1] + u[1:])
    for j in range(1, n - 1):
        out["dUdu1"][j] = (U[j + 1] - U[j - 1]) / (u[j + 1] - u[j - 1])
        if np.isfinite(D1[j]) and np.isfinite(D1[j - 1]) and D1[j] != 0 and D1[j - 1] != 0:
            out["kappa1"][j] = -(math.log(abs(D1[j])) - math.log(abs(D1[j - 1]))) / (m1[j] - m1[j - 1])
            out["sign_change"][j] = float(np.sign(D1[j]) != np.sign(D1[j - 1]))
    if n >= 5:
        D2 = (U[2:] - U[:-2]) / (u[2:] - u[:-2]); m2 = 0.5 * (u[:-2] + u[2:])
        for j in range(2, n - 2):
            out["dUdu2"][j] = (U[j + 2] - U[j - 2]) / (u[j + 2] - u[j - 2])
            if np.isfinite(D2[j]) and np.isfinite(D2[j - 2]) and D2[j] != 0 and D2[j - 2] != 0:
                out["kappa2"][j] = -(math.log(abs(D2[j])) - math.log(abs(D2[j - 2]))) / (m2[j] - m2[j - 2])
    out["err"] = np.abs(out["kappa1"] - out["kappa2"])
    return out


def centred_log_derivative(u, E):
    """d ln E/du at 1 and 2 steps (centred), nan where E <= 0 or undefined."""
    u = np.asarray(u, float); E = np.asarray(E, float); n = len(u)
    lnE = np.where(E > 0, np.log(np.where(E > 0, E, 1.0)), np.nan)
    d1 = np.full(n, np.nan); d2 = np.full(n, np.nan)
    for j in range(1, n - 1):
        d1[j] = (lnE[j + 1] - lnE[j - 1]) / (u[j + 1] - u[j - 1])
    for j in range(2, n - 2):
        d2[j] = (lnE[j + 2] - lnE[j - 2]) / (u[j + 2] - u[j - 2])
    return d1, d2


def plateau_segment(u, kappa, err, frac, err_frac, ref=None, step=None):
    """the longest contiguous run of clocks within frac of a centre with the error within err_frac of it: every clock's kappa as the centre
    when ref is None (the plateau rule), the given per-clock reference otherwise (the Hawking band); the run's mean, its covered interval."""
    u = np.asarray(u, float); kappa = np.asarray(kappa, float); err = np.asarray(err, float); n = len(u)
    if step is None:
        step = float(np.median(np.diff(u))) if n > 1 else 0.0
    valid = np.isfinite(kappa) & np.isfinite(err)
    best = None
    cands = [(float(kappa[j]), None) for j in np.nonzero(valid)[0]] if ref is None else [(None, np.asarray(ref, float))]
    for c, refarr in cands:
        centre = np.full(n, c) if refarr is None else refarr
        scale = np.abs(centre)
        ok = valid & np.isfinite(centre) & (scale > 0) & (np.abs(kappa - centre) <= frac * scale) & (err <= err_frac * scale)
        i = 0
        while i < n:
            if not ok[i]:
                i += 1; continue
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            key = (j - i + 1, -float(np.mean(err[i: j + 1])))
            if best is None or key > best[0]:
                best = (key, i, j, c)
            i = j + 1
    if best is None:
        return {"found": False, "rule": "plateau" if ref is None else "reference band", "frac": frac, "err_frac": err_frac, "step": step}
    _, i, j, c = best
    run = slice(i, j + 1)
    return {"found": True, "rule": "plateau" if ref is None else "reference band", "frac": frac, "err_frac": err_frac, "step": step,
            "plateau": float(np.mean(kappa[run])), "candidate": c, "reference_mean": (float(np.mean(np.asarray(ref, float)[run])) if ref is not None else None),
            "i0": int(i), "i1": int(j), "u_start": float(u[i]), "u_end": float(u[j]), "n_clocks": int(j - i + 1), "extent": float(u[j] - u[i] + step),
            "mean_err": float(np.mean(err[run])), "kappa_min": float(np.min(kappa[run])), "kappa_max": float(np.max(kappa[run]))}


def T_d_over_rs(q, cfg, rs_eff_over_rs=1.0):
    """T_d/r_s (nominal units) = prefactor (r_s,eff/r_s) ln(r_s/(inner q)) = prefactor (r_s,eff/r_s) ln(1/(inner a omega))."""
    return cfg["Td_prefactor"] * rs_eff_over_rs * math.log(cfg["rs_nominal"] / (cfg["Td_inner"] * q))


def extents(seg, rows, rs, cfg, q):
    """the segment's extent against T_d(omega), 2 r_s (e-folds) and 4 pi r_s (thermal periods), with r_s = 2gE at the segment's start and
    the nominal r_s both; the resolution flag against the probe's width in u."""
    if not seg.get("found"):
        return {}
    two = rows[seg["i0"]]["two_gE_launch"]; f = two / rs
    sig_u = float(np.median([r["launch"]["sigma_u_rs"] for r in rows]))
    ex = seg["extent"]
    return {"T_d_nominal_rs": T_d_over_rs(q, cfg, 1.0), "T_d_2gE_rs": T_d_over_rs(q, cfg, f), "extent_over_T_d_nominal": ex / T_d_over_rs(q, cfg, 1.0),
            "extent_over_T_d_2gE": ex / T_d_over_rs(q, cfg, f), "e_folds_nominal": ex / 2.0, "e_folds_2gE": ex / (2.0 * f), "thermal_periods_nominal": ex / (4.0 * math.pi),
            "thermal_periods_2gE": ex / (4.0 * math.pi * f), "sigma_u_rs_median": sig_u, "resolution_limited": bool(ex < 2.0 * sig_u), "two_gE_at_start": two}


def analyse_map(rows, rs, cfg):
    """the derived map of one (channel, frequency): the arrays over the clocks, the plateau and the segment (the position map, the in-energy
    exponent, the ray map), the Hawking bands, the extents."""
    rows = sorted(rows, key=lambda r: r["u_over_rs"])
    u = np.array([r["u_over_rs"] for r in rows]); n = len(u)
    q = rows[0]["omega_rs"]; omega = rows[0]["omega"]; kappa_ch = rows[0]["kappa"]
    U = np.array([r["T0"]["U"] for r in rows]); sU = np.array([r["T0"]["sigma_U"] for r in rows])
    kH = np.array([rs / (2.0 * r["two_gE_launch"]) for r in rows])                          # 1/(4GE) in units of 1/r_s
    d = map_derivatives(u, U)
    U0 = np.array([r["T0"]["U0_chart"] if r["T0"].get("U0_chart") is not None else np.nan for r in rows]); d0 = map_derivatives(u, U0)
    E0 = np.array([r["T0"]["energy_h0"] for r in rows]); kE1, kE2 = centred_log_derivative(u, E0)
    sig_u = np.array([r["launch"]["sigma_u_rs"] for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        k_pred = omega * rs / np.abs(d["dUdu1"])
        spread_est = sU / sig_u
    k_meas = np.array([r["T0"]["k_mean_abs"] for r in rows])
    arrays = {"u_over_rs": u, "T_launch_over_rs": np.array([r["T_launch_over_rs"] for r in rows]), "k_launch": np.array([r["k_launch"] for r in rows]),
              "U": U, "sigma_U": sU, "kappa_H_per_rs": kH, "dUdu1": d["dUdu1"], "dUdu2": d["dUdu2"], "kappa1": d["kappa1"], "kappa2": d["kappa2"], "err": d["err"],
              "sign_change": d["sign_change"], "U0_chart": U0, "kappa0_1": d0["kappa1"], "kappa0_2": d0["kappa2"], "energy_h0": E0, "kappaE_1": kE1, "kappaE_2": kE2,
              "k_pred_from_dUdu": k_pred, "k_mean_abs_T0": k_meas, "spread_estimate_dUdu": spread_est, "sigma_u_rs": sig_u,
              "frac_k_super_T0": np.array([r["T0"]["frac_k_super"] for r in rows]), "frac_k_negative_T0": np.array([r["T0"]["frac_k_negative"] for r in rows]),
              "norm_T0": np.array([r["T0"]["norm_T0"] for r in rows]),
              "fb_defect": np.array([r["fb"]["defect_norm"] if r.get("fb") else np.nan for r in rows]),
              "w_neg_in": np.array([r["T0"]["w_neg_in"] if r["T0"].get("w_neg_in") is not None else np.nan for r in rows])}
    res = {"kappa": kappa_ch, "omega_rs": q, "omega": omega, "n_clocks": n, "u_first": float(u[0]), "u_last": float(u[-1]),
           "omega_over_TH_first": omega * 4.0 * math.pi * rows[0]["two_gE_launch"], "arrays": arrays}
    pf, ef = cfg["plateau_frac"], cfg["segment_err_frac"]
    res["plateau_x"] = plateau_segment(u, d["kappa1"], d["err"], pf, ef)
    res["hawking_band_x"] = plateau_segment(u, d["kappa1"], d["err"], pf, ef, ref=kH)
    res["plateau_E"] = plateau_segment(u, kE1, np.abs(kE1 - kE2), pf, ef)
    res["hawking_band_E"] = plateau_segment(u, kE1, np.abs(kE1 - kE2), pf, ef, ref=kH)
    res["plateau_0"] = plateau_segment(u, d0["kappa1"], d0["err"], pf, ef)
    for key in ("plateau_x", "hawking_band_x", "plateau_E", "hawking_band_E", "plateau_0"):
        res[key + "_extents"] = extents(res[key], rows, rs, cfg, q)
        if res[key].get("found"):
            res[key]["plateau_x4GE"] = float(np.mean(d["kappa1"][res[key]["i0"]: res[key]["i1"] + 1] / kH[res[key]["i0"]: res[key]["i1"] + 1])) if key.endswith("_x") else \
                float(np.mean((kE1 if key.endswith("_E") else d0["kappa1"])[res[key]["i0"]: res[key]["i1"] + 1] / kH[res[key]["i0"]: res[key]["i1"] + 1]))
            res[key]["plateau_x2rs_nominal"] = res[key]["plateau"] * 2.0
    if all(r.get("ray") and r["ray"].get("U_ray") is not None for r in rows):
        Ur = np.array([r["ray"]["U_ray"] for r in rows]); dr = map_derivatives(u, Ur)
        arrays.update({"U_ray": Ur, "k_ray": np.array([r["ray"]["k_ray"] for r in rows]), "kappa_ray1": dr["kappa1"], "kappa_ray2": dr["kappa2"], "err_ray": dr["err"],
                       "U_ray_central": np.array([r["ray"]["U_ray_central"] if r["ray"].get("U_ray_central") is not None else np.nan for r in rows]),
                       "sigma_ray": np.array([r["ray"]["sigma_ray"] for r in rows])})
        res["plateau_ray"] = plateau_segment(u, dr["kappa1"], dr["err"], pf, ef)
        res["hawking_band_ray"] = plateau_segment(u, dr["kappa1"], dr["err"], pf, ef, ref=kH)
        res["plateau_ray_extents"] = extents(res["plateau_ray"], rows, rs, cfg, q)
        res["packet_minus_ray_max"] = float(np.nanmax(np.abs(U - Ur)))
        with np.errstate(invalid="ignore"):
            res["kappa_packet_minus_ray_over_H_max"] = float(np.nanmax(np.abs(d["kappa1"] - dr["kappa1"]) / kH))
    res["kappa_x_max_x4GE"] = float(np.nanmax(d["kappa1"] / kH)) if np.isfinite(d["kappa1"]).any() else None
    res["kappa_E_max_x4GE"] = float(np.nanmax(kE1 / kH)) if np.isfinite(kE1).any() else None
    res["first_clock_k_dispersive"] = next((float(u[j]) for j in range(n) if k_meas[j] >= cfg["k_dispersive"]), None)
    res["first_clock_frac_super_above_half"] = next((float(u[j]) for j in range(n) if arrays["frac_k_super_T0"][j] >= 0.5), None)
    return res


# ------------------------------------------------------------------------------------------------------ the record table
def kappa_R_table(own, cfg, rs):
    """per record the roots and slopes against 1/(4GE_k): the dense nodes' root and slope (secant, quadratic fit), the site fields' slope
    and the densities' kappa_R (page_d13_readout_v1.enclosed_source), the engine's kappa_T."""
    g = own.g
    rows = []
    for k in range(len(own)):
        rec = own.record(k)
        E = float(own.E[k]); two = 2.0 * g * E; kH = 1.0 / (2.0 * two)
        N_rec, v_rec, R = own.N_rec[k], own.v_rec[k], own.R_rec
        root_site = D13.outer_root(N_rec, v_rec, R)
        row = {"k": k, "T_over_rs": float(own.T[k]) / rs, "E": E, "two_gE": two, "kappa_H": kH, "R_t_json": rec.get("R_t"), "kappa_T_json": rec.get("kappa_T"),
               "four_gE_kappa_T_json": rec.get("four_gE_kappa_T"), "min_N_minus_absv_json": rec.get("min_N_minus_absv"), "root_site": root_site,
               "R_t_dense": None, "r_in_dense": None, "kappa_R_dense_secant": None, "kappa_R_dense_fit": None, "n_nodes": None, "min_c_dense": None,
               "kappa_R_site_slope": None, "kappa_R_densities": None, "kappa_ratio_densities_over_slope": None}
        dn = own.dense_nodes(k)
        if dn is not None and len(dn[0]) >= 4:
            r, Nn, vn = dn
            c = Nn - np.abs(vn)
            row["n_nodes"] = int(len(r)); row["min_c_dense"] = float(np.min(c))
            s = np.nonzero((c[:-1] <= 0) & (c[1:] > 0))[0]
            if len(s):
                i = int(s[-1])
                rt = float(r[i] + (r[i + 1] - r[i]) * (-c[i]) / (c[i + 1] - c[i]))
                row["R_t_dense"] = rt
                row["kappa_R_dense_secant"] = float((c[i + 1] - c[i]) / (r[i + 1] - r[i]))
                nf = max(3, int(cfg["slope_fit_nodes"]))
                near = np.argsort(np.abs(r - rt))[:nf]
                co = np.polyfit(r[near] - rt, c[near], 2)
                row["kappa_R_dense_fit"] = float(co[1])
            d_ = np.nonzero((c[:-1] > 0) & (c[1:] <= 0))[0]
            if len(d_):
                j = int(d_[0]); row["r_in_dense"] = float(r[j] + (r[j + 1] - r[j]) * c[j] / (c[j] - c[j + 1]))
        if root_site is not None:
            eps = own.eps[k] if own.eps is not None else None; J = own.J[k] if own.J is not None else None
            enc = D13.enclosed_source(np.zeros(len(R)), R, N_rec, v_rec, g, root_site, eps, J, R)
            if enc:
                row["kappa_R_site_slope"] = enc.get("kappa_R_slope"); row["kappa_R_densities"] = enc.get("kappa_R_from_densities")
                row["kappa_ratio_densities_over_slope"] = enc.get("kappa_ratio_densities_over_slope")
        for key in ("kappa_R_dense_secant", "kappa_R_dense_fit", "kappa_R_site_slope", "kappa_R_densities", "kappa_T_json"):
            val = row.get(key)
            row[key + "_x4GE"] = (val / kH) if val is not None else None
        rows.append(row)
    return rows


def window_start(table, F, cfg, rs, R_L):
    """the first record at which each slope column reaches --kappaR-frac of 1/(4GE) and its clocks: T_k, the retarded clock at the launch
    radius of light leaving f x 2gE then by the chart and by the light time in the record's fields."""
    frac = cfg["kappaR_frac"]
    out = {"kappaR_frac": frac, "columns": {}, "launch_radius": R_L}
    for key in ("kappa_R_dense_fit", "kappa_R_dense_secant", "kappa_R_site_slope", "kappa_R_densities", "kappa_T_json"):
        hit = next((row for row in table if row.get(key + "_x4GE") is not None and row[key + "_x4GE"] >= frac), None)
        if hit is None:
            out["columns"][key] = {"found": False}
            continue
        k = hit["k"]; T_k = float(F.T[k]); two = hit["two_gE"]
        N, v = F.fields(k)
        d = {"found": True, "k": k, "T_over_rs": T_k / rs, "ratio": hit[key + "_x4GE"], "R_t": hit.get("R_t_dense") if hit.get("R_t_dense") is not None else hit.get("root_site"), "launches": []}
        for f_ in cfg["window_launch_over_2gE"]:
            R_w = f_ * two
            try:
                u_chart = OPS.chart_retarded_time(T_k, two, R_w) / rs
            except (ValueError, ZeroDivisionError):
                u_chart = None
            t_out = D13.light_time(N, v, F.R, R_w, R_L, True)
            u_arr = (OPS.chart_retarded_time(T_k + t_out, two, R_L) / rs) if t_out is not None else None
            d["launches"].append({"over_2gE": f_, "R_w": R_w, "u_chart_rs": u_chart, "t_out_rs": (t_out / rs) if t_out is not None else None, "u_at_launch_radius_rs": u_arr})
        out["columns"][key] = d
    return out


# --------------------------------------------------------------------------------------------------------------- the store
class RayStore:
    """raymap_<tag>.json (every scalar: settings, rows per probe, maps per channel and frequency, the record table) and raymap_<tag>.npz
    (every raw array); resumable."""

    def __init__(self, prefix, settings=None, label=None):
        self.prefix = prefix; self.pj = prefix + ".json"; self.pz = prefix + ".npz"
        self.J = {"code": CODE, "settings": settings, "label": label, "rows": {}, "maps": {}, "kappa_R": None, "fields": None, "clocks": None, "notes": []}
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
        self.J["code"] = CODE
        if settings is not None:
            self.J["settings"] = settings
        if label is not None:
            self.J["label"] = label

    @staticmethod
    def row_key(kappa, u_rs, q):
        return "k%d_u%08.4f_q%06.3f" % (int(kappa), float(u_rs), float(q))

    def has_T0(self, key):
        r = self.J["rows"].get(key)
        return bool(r and r.get("T0") is not None)

    def has_fb(self, key):
        r = self.J["rows"].get(key)
        return bool(r and r.get("fb") is not None)

    def psi0(self, key, device):
        a, b = key + "_psi0_re", key + "_psi0_im"
        if a in self.Z and b in self.Z:
            return torch.as_tensor(self.Z[a] + 1j * self.Z[b], dtype=CD, device=device)
        return None

    def flush(self):
        tmp = self.pj + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.J, f, default=_jdefault, indent=1)
        os.replace(tmp, self.pj)
        if self.Z:
            tmpz = self.pz + ".tmp.npz"
            np.savez(tmpz, **self.Z)
            os.replace(tmpz, self.pz)


# -------------------------------------------------------------------------------------------------------- one channel's run
def run_channel(F, kappa, specs, cfg, rs, n_L, device, store, extra_row=None):
    """specs: [{"k", "u_rs", "q", "omega"}]; the pending probes are built at their launch records, back-propagated to the first record, read
    out, re-propagated forward and compared with their launch; the rows join the store. Returns the rows of every spec (from the store)."""
    keys = [store.row_key(kappa, s["u_rs"], s["q"]) for s in specs]
    pend = [i for i, key in enumerate(keys) if cfg["force"] or not store.has_T0(key)]
    pend_fb = [i for i, key in enumerate(keys) if (not cfg["force"]) and store.has_T0(key) and not store.has_fb(key)]
    R = np.asarray(F.R, float); n = len(R)
    R_t = torch.as_tensor(R, dtype=RD, device=device)
    probes = []
    t0 = time.time()
    launch_cache = {}
    for i in pend + pend_fb:
        s = specs[i]
        psiL, info, (N, v) = build_packet(F, s["k"], kappa, s["omega"], n_L, cfg, rs, device)
        if s["k"] not in launch_cache:
            h = F.h_csr(kappa, N, v)
            launch_cache[s["k"]] = (F.banded(kappa, h, device), OPS.Banded(OPS.velocity_csr(h, R), device))
        Bh, Vb = launch_cache[s["k"]]
        d, rhoL, PsL, ksL = packet_diagnostics(psiL, R_t, Bh, Vb, n, cfg["k_super"], cfg["k_dispersive"])
        info.update(d)
        info["sigma_u_rs"] = d["sigma_x"] / max(abs(d["group_velocity"]), 1e-300) / rs
        info["energy_h_over_omega"] = d["energy_h"] / s["omega"]; info["energy_spread_over_omega"] = d["energy_spread_h"] / s["omega"]
        info["centroid_minus_RL"] = d["centroid"] - info["R_L"]
        probes.append({"i": i, "key": keys[i], "k": s["k"], "u_rs": s["u_rs"], "q": s["q"], "omega": s["omega"], "psiL": psiL, "launch": info, "rhoL": rhoL, "PsL": PsL,
                       "two_gE": F.two_gE(s["k"]), "need_T0": i in pend})
    del launch_cache
    log("kappa %d: %d probes built (%d need the back-propagation, %d the forward check only) in %.1f s" % (kappa, len(probes), len(pend), len(pend_fb), time.time() - t0))
    if not probes:
        return [store.J["rows"][key] for key in keys if key in store.J["rows"]]
    Vneg = None
    if cfg["in_vacuum"]:
        N0, v0 = F.fields(0)
        t1 = time.time()
        vac = D13.killing_vacuum(F.h_csr(kappa, N0, v0), device, "dense")
        Vneg = vac["Vneg"]
        log("kappa %d: P_-(h(T = 0)) by %s: %d occupied, %.0f s" % (kappa, vac["route"], vac["n_neg"], time.time() - t1))
    # the back-propagation of the probes that need it
    need = [p for p in probes if p["need_T0"]]
    if need:
        X0, bst = backward_pass(F, kappa, need, cfg, device)
        T0s, arrs = readouts_T0(F, kappa, X0, need, cfg, rs, n_L, device, Vneg)
        rays = ray_compare(F, kappa, need, cfg, rs) if cfg["ray_compare"] else [None] * len(need)
        for j, p in enumerate(need):
            tz = T0s[j]
            tz.update({"norm_min_along": float(bst["norm_min"][j]), "norm_max_along": float(bst["norm_max"][j]), "cheb_K_max": int(bst["cheb_K_max"]),
                       "cheb_tail_acc": float(bst["cheb_tail_acc"][j]), "records_back_propagated": int(p["k"]), "norm_flag": bool(abs(tz["norm_T0"] - 1.0) > cfg["norm_tol"])})
            L = {k_: v_ for k_, v_ in p["launch"].items()}
            row = {"kappa": int(kappa), "u_over_rs": p["u_rs"], "omega_rs": p["q"], "omega": p["omega"], "k_launch": int(p["k"]), "T_launch": float(F.T[p["k"]]),
                   "T_launch_over_rs": float(F.T[p["k"]]) / rs, "E_launch": float(F.E[p["k"]]), "two_gE_launch": p["two_gE"], "kappa_H_lattice": 1.0 / (2.0 * p["two_gE"]),
                   "omega_over_TH": p["omega"] * 4.0 * math.pi * p["two_gE"], "launch": L, "T0": tz, "ray": rays[j], "fb": None}
            if extra_row:
                row.update(extra_row(p["k"]))
            store.J["rows"][p["key"]] = row
            rho, Ps, ks = arrs[j]
            store.Z[p["key"] + "_rho0"] = rho; store.Z[p["key"] + "_spec0"] = Ps; store.Z[p["key"] + "_rhoL"] = p["rhoL"]; store.Z[p["key"] + "_specL"] = p["PsL"]
            if "spec_k_grid" not in store.Z:
                store.Z["spec_k_grid"] = ks; store.Z["R_sites"] = R
            if cfg["save_states"]:
                x = X0[:, j].cpu().numpy(); store.Z[p["key"] + "_psi0_re"] = x.real.copy(); store.Z[p["key"] + "_psi0_im"] = x.imag.copy()
            log("raymap kappa %d u/r_s %.3f q %.3f (T_L/r_s %.3f, %d records): U/r_s %.4f (sigma %.4f), <|k|> %.4f (super %.3f, k<0 %.3f), <h0>/omega %.3f, norm-1 %.1e%s; launch sigma_u %.2f r_s, <h>/omega %.4f (spread %.3f), k<0 %.1e%s" % (
                kappa, p["u_rs"], p["q"], row["T_launch_over_rs"], p["k"], tz["U_over_rs"], tz["sigma_U_over_rs"], tz["k_mean_abs"], tz["frac_k_super"], tz["frac_k_negative"],
                tz["energy_h0_over_omega"], tz["norm_T0"] - 1.0, (" w_neg %.3e" % tz["w_neg_in"]) if tz["w_neg_in"] is not None else "", L["sigma_u_rs"], L["energy_h_over_omega"],
                L["energy_spread_over_omega"], L["frac_k_negative"], (", ray U/r_s %s" % _g(rays[j]["U_ray_over_rs"], "%.4f")) if rays[j] else ""))
        store.flush()
        xmap = {p["key"]: X0[:, j] for j, p in enumerate(need)}
    else:
        xmap = {}
    # the forward check
    fb_probes = []; X0_list = []
    for p in probes:
        x = xmap.get(p["key"])
        if x is None:
            x = store.psi0(p["key"], device)
        if x is None:
            store.J["notes"].append("row %s: no T = 0 state saved; its forward check needs --force" % p["key"]); continue
        fb_probes.append(p); X0_list.append(x)
    if fb_probes:
        X0f = torch.stack(X0_list, dim=1)
        res = forward_pass(F, kappa, fb_probes, X0f, cfg, device)
        for p, r in zip(fb_probes, res):
            r["fb_flag"] = bool(r["defect_norm"] > cfg["fb_tol"] or abs(r["norm_fwd"] - 1.0) > cfg["norm_tol"])
            store.J["rows"][p["key"]]["fb"] = r
            log("fb kappa %d u/r_s %.3f q %.3f: ||psi_fwd - psi_L|| %.2e, 1 - |overlap| %.2e, norm - 1 %.2e%s" % (
                kappa, p["u_rs"], p["q"], r["defect_norm"], r["overlap_defect"], r["norm_fwd"] - 1.0, " FLAG" if r["fb_flag"] else ""))
        store.flush()
    del Vneg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return [store.J["rows"][key] for key in keys if key in store.J["rows"]]


# ------------------------------------------------------------------------------------------------------------------ the md
def write_md(path, store, cfg, rs, maps, info_lines):
    J = store.J
    L = ["# RAYMAP_solved_flow_v1: %s" % J["settings"].get("tag", ""), "", J.get("label") or "", ""]
    L += ["## Settings (every named argument)", "", "```", json.dumps({k_: v_ for k_, v_ in cfg.items() if k_ != "force"}, default=_jdefault, indent=None), "```", ""]
    L += ["## Definitions", "",
          "- The probe: a unit outgoing wave packet on the local positive branch at the launch site (the carrier from omega_+(k) = omega, the Gaussian spectral amplitude "
          "about the energy-matched centre k_c with no component below k_turn), width rule %s (rel-spread %g, sigma_x cap %g r_s; sigma-u %g r_s); its measured "
          "width, width in u, energy and spectral content are in every row." % (cfg["width_rule"], cfg["rel_spread"], cfg["sigma_x_max_rs"], cfg["sigma_u_rs"]),
          "- U(u): the proper radial position of the packet's centroid on the first slice (T = 0), the affine parameter whose compression is the traced-back "
          "wavenumber's blueshift (k_0 = omega/|dU/du| in geometric optics), the quantity T_d counts; U_0 (the slice's outgoing-light-time parameter) and <h_0> "
          "(the in-energy, kappa_E = d ln<h_0>/du) beside it.",
          "- kappa(u) = -d ln|dU/du|/du by the log-difference of first differences at 1 and 2 clock steps (exact for an exponential map at any step); the error "
          "column is their difference. 1/(4GE) at the launch record is the comparator (x4GE columns); 1/(2 r_s nominal) beside.",
          "- The plateau: the value about which the longest run of clocks stays within %g with the 2-step error within %g of it; the segment is that run; the "
          "Hawking band is the same about 1/(4GE). Extents against T_d = %g r_s ln(1/(%g a omega)), 2 r_s and 4 pi r_s (r_s = 2gE at the segment's start, and the "
          "nominal r_s). A segment shorter than 2 sigma_u is resolution-limited." % (cfg["plateau_frac"], cfg["segment_err_frac"], cfg["Td_prefactor"], cfg["Td_inner"]),
          "- The geometric-optics ray: the lattice dispersion's ray traced back through the same records (the Wigner average over the launch packet, %d^2 nodes)." % cfg["ray_wigner_nodes"], ""]
    L += ["## The history, the fields and the clocks", ""] + ["- " + s for s in info_lines] + [""]
    KR = J.get("kappa_R")
    if KR:
        ws = KR["window_start"]
        L += ["## The record table: kappa_R against 1/(4GE)", "",
              "The first record at which each slope column reaches %g of 1/(4GE), and the window's start clocks (the light leaving f x 2gE then, at the launch radius %.1f):" % (ws["kappaR_frac"], ws["launch_radius"]), ""]
        for key, d in ws["columns"].items():
            if not d.get("found"):
                L.append("- %s: never reaches the fraction" % key); continue
            la = "; ".join("f %g: u(chart) %s, light time %s r_s, u(arrival) %s" % (x["over_2gE"], _g(x["u_chart_rs"], "%.3f"), _g(x["t_out_rs"], "%.3f"), _g(x["u_at_launch_radius_rs"], "%.3f")) for x in d["launches"])
            L.append("- %s: record %d, T/r_s %.3f (ratio %.3f, root %s); %s" % (key, d["k"], d["T_over_rs"], d["ratio"], _g(d["R_t"], "%.3f"), la))
        L += ["", "| T/r_s | R_t json | R_t dense | kappa_R dense secant x4GE | dense fit x4GE | site slope x4GE | densities x4GE | kappa_T x4GE | min(N-abs v) | r_in dense |", "|---|---|---|---|---|---|---|---|---|---|"]
        every = max(1, int(round(cfg["md_every_rs"] / max(KR["table"][1]["T_over_rs"] - KR["table"][0]["T_over_rs"], 1e-9)))) if len(KR["table"]) > 1 else 1
        hits = set(d.get("k") for d in ws["columns"].values() if d.get("found"))
        for row in KR["table"]:
            if row["k"] % every == 0 or row["k"] in hits or (row["k"] - 1) in hits:
                L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (row["T_over_rs"], _g(row["R_t_json"], "%.3f"), _g(row["R_t_dense"], "%.3f"),
                                                                          _g(row["kappa_R_dense_secant_x4GE"], "%.4f"), _g(row["kappa_R_dense_fit_x4GE"], "%.4f"),
                                                                          _g(row["kappa_R_site_slope_x4GE"], "%.4f"), _g(row["kappa_R_densities_x4GE"], "%.4f"),
                                                                          _g(row["kappa_T_json_x4GE"], "%.4f"), _g(row["min_N_minus_absv_json"], "%.4f"), _g(row["r_in_dense"], "%.2f")))
        L.append("")
    L += ["## The ray map per channel and frequency", ""]
    for key, m in maps.items():
        A = m["arrays"]
        L += ["### kappa %d, omega r_s = %g (omega/T_H %.2f at the first launch; %d clocks from u/r_s %.3f to %.3f)" % (m["kappa"], m["omega_rs"], m["omega_over_TH_first"], m["n_clocks"], m["u_first"], m["u_last"]), ""]
        for name, lab in (("plateau_x", "position map U: plateau"), ("hawking_band_x", "position map U: Hawking band"), ("plateau_E", "in-energy <h_0>: plateau"),
                          ("hawking_band_E", "in-energy <h_0>: Hawking band"), ("plateau_0", "slice chart U_0: plateau"), ("plateau_ray", "geometric-optics ray: plateau"),
                          ("hawking_band_ray", "geometric-optics ray: Hawking band")):
            s = m.get(name)
            if not s:
                continue
            if not s.get("found"):
                L.append("- %s: no run of clocks satisfies the rule" % lab); continue
            ex = m.get(name + "_extents", {})
            L.append("- %s: kappa %.5f per r_s = %.4f x 1/(4GE) (%.4f x 1/(2 r_s nominal)) over u/r_s [%.3f, %.3f], %d clocks, extent %.3f r_s = %s T_d(2gE) = %s T_d(nominal) = %s e-folds (2gE) = %s thermal periods (2gE); mean 2-step error %.2e; sigma_u %s r_s%s" % (
                lab, s["plateau"], s.get("plateau_x4GE", float("nan")), s["plateau_x2rs_nominal"], s["u_start"], s["u_end"], s["n_clocks"], s["extent"],
                _g(ex.get("extent_over_T_d_2gE"), "%.3f"), _g(ex.get("extent_over_T_d_nominal"), "%.3f"), _g(ex.get("e_folds_2gE"), "%.3f"), _g(ex.get("thermal_periods_2gE"), "%.4f"),
                s["mean_err"], _g(ex.get("sigma_u_rs_median"), "%.2f"), " (resolution-limited)" if ex.get("resolution_limited") else ""))
        L += ["- largest kappa x 4GE over the clocks: position map %s, in-energy %s; first clock with <|k|> at T = 0 at or above %g: %s; first clock with half the weight beyond k = %g: %s%s" % (
            _g(m.get("kappa_x_max_x4GE"), "%.4f"), _g(m.get("kappa_E_max_x4GE"), "%.4f"), cfg["k_dispersive"], _g(m.get("first_clock_k_dispersive"), "%.3f"), cfg["k_super"],
            _g(m.get("first_clock_frac_super_above_half"), "%.3f"), ("; packet minus ray max %.2f spacings, kappa difference max %s x 1/(4GE)" % (m["packet_minus_ray_max"], _g(m.get("kappa_packet_minus_ray_over_H_max"), "%.3f"))) if "packet_minus_ray_max" in m else ""), ""]
        hdr = "| u/r_s | T_L/r_s | U/r_s | sigma_U/r_s | dU/du (1) | kappa1 x4GE | kappa2 x4GE | err x4GE | kappa_0 x4GE | kappa_E x4GE | <abs k> T0 | omega/abs(dU/du) | f(k>pi/4) | f(k<0) | <h0>/omega | norm-1 | fb defect | U_ray/r_s | kappa_ray x4GE |"
        L += [hdr, "|" + "---|" * 19]
        E0 = A["energy_h0"]
        for j in range(m["n_clocks"]):
            kH = A["kappa_H_per_rs"][j]
            L.append("| %.3f | %.3f | %.4f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %.3f | %.1e | %s | %s | %s |" % (
                A["u_over_rs"][j], A["T_launch_over_rs"][j], A["U"][j] / rs, A["sigma_U"][j] / rs, _g(A["dUdu1"][j], "%.4e"), _g(A["kappa1"][j] / kH, "%.4f"), _g(A["kappa2"][j] / kH, "%.4f"),
                _g(A["err"][j] / kH, "%.3f"), _g(A["kappa0_1"][j] / kH, "%.4f"), _g(A["kappaE_1"][j] / kH, "%.4f"), _g(A["k_mean_abs_T0"][j], "%.4f"), _g(A["k_pred_from_dUdu"][j], "%.4f"),
                A["frac_k_super_T0"][j], A["frac_k_negative_T0"][j], E0[j] / m["omega"], A["norm_T0"][j] - 1.0, _g(A["fb_defect"][j], "%.1e"),
                _g(A["U_ray"][j] / rs, "%.4f") if "U_ray" in A else "-", _g(A["kappa_ray1"][j] / kH, "%.4f") if "kappa_ray1" in A else "-"))
        L.append("")
    flags = [k_ for k_, r in J["rows"].items() if (r.get("T0") or {}).get("norm_flag") or (r.get("fb") or {}).get("fb_flag")]
    L += ["## Checks", "", "- rows flagged by the norm (> %g) or the forward check (> %g): %d%s" % (cfg["norm_tol"], cfg["fb_tol"], len(flags), (": " + ", ".join(flags[:20])) if flags else ""),
          "- notes: %s" % ("; ".join(J.get("notes") or []) or "none"), "",
          "## What a failure looks like", "",
          "- a norm or forward-check defect above the tolerance: the Chebyshev step did not resolve a record (raise the order's tolerance or check the record's fields for nan);",
          "- a packet whose launch centroid is off R_L by more than a tenth of its width, whose <h>/omega is far from 1 or whose weight at k < 0 is not tiny: the branch solve or the cutoff failed at that record;",
          "- a kappa column of nan: the map's first differences vanished or changed sign (the packet stalled or turned around); the sign-change array says which;",
          "- a large packet-minus-ray difference with a small 2-step error: wave effects (reflection off the centrifugal term or the geometry); with a large error: the readout is in the stalled regime.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True)
    ap.add_argument("--nsites", type=int, default=8192, help="the light channel's chain (the record's chain plus the appended vacuum exterior)")
    ap.add_argument("--g", type=float, default=None); ap.add_argument("--rs-nominal", dest="rs_nominal", type=float, default=None)
    ap.add_argument("--fields-key", dest="fields_key", default="applied", choices=["applied", "stored"])
    ap.add_argument("--fields-source", dest="fields_source", default="auto", choices=["auto", "own", "v6m"])
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="recompute rows already in the store")
    add_arguments(ap)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    os.makedirs(a.out, exist_ok=True)
    cfg = cfg_from_args(a); cfg["force"] = bool(a.force)
    F, own, finfo = make_fields(a, dev, kappa_check=cfg["kappas"][0])
    rs = float(a.rs_nominal or own.rs_nom or (2.0 * own.g * own.E[0]))
    cfg["rs_nominal"] = rs
    if own.args.get("R0_over_rs") is not None:
        cfg["ball_rs"] = float(own.args["R0_over_rs"])
    n_L = int(np.argmin(np.abs(np.asarray(F.R, float) - cfg["launch_rs"] * rs))); R_L = float(F.R[n_L])
    store = RayStore(os.path.join(a.out, "raymap_" + a.tag), settings=dict(vars(a), cfg={k_: v_ for k_, v_ in cfg.items()}),
                     label="the ray map on the solved flow (raymap_solved_flow_v1) on %s; light channel %s on %d sites" % (own.label or a.record, "kappa " + ",".join(str(k_) for k_ in cfg["kappas"]), F.nsites))
    store.J["fields"] = finfo
    log("raymap: %d records (T/r_s %.3f to %.3f, tau %g), g %g, r_s nominal %g, chain %d sites, launch site %d (R %.1f = %.3f r_s), device %s; fields: %s (%s)" % (
        len(F), F.T[0] / rs, F.T[-1] / rs, F.tau, F.g, rs, F.nsites, n_L, R_L, R_L / rs, dev, finfo["source"], own.fields_key))
    # the formation record and the first return at the launch radius
    T = np.asarray(F.T, float)
    k_form = int(np.argmin(np.abs(T - cfg["form_rs"] * rs)))
    r_form = D13.outer_root(own.N_rec[k_form], own.v_rec[k_form], own.R_rec)
    if r_form is None:
        r_form = F.two_gE(k_form)
    u_grid = np.arange(cfg["u_from_rs"], cfg["u_to_rs"] + 1e-9, cfg["u_step_rs"])
    clocks = clock_records(F, R_L, rs, u_grid)
    k_to = clocks[-1][0]
    N_to, v_to = F.fields(k_to)
    fr = D13.first_return(N_to, v_to, F.R, R_L, float(T[k_to]), float(T[0]), rs, float(T[k_form]), r_form)
    ext = []
    if cfg["extend"]:
        seen = set(k_ for k_, _ in clocks); u = cfg["u_to_rs"] + cfg["u_step_rs"]
        while True:
            cr = clock_records(F, R_L, rs, [u])
            if not cr:
                break
            k, ua = cr[0]
            if k in seen or (cfg["u_max_rs"] is not None and u > cfg["u_max_rs"] + 1e-9):
                break
            if fr.get("first_return_rs") is not None and T[k] / rs > fr["first_return_rs"]:
                break
            seen.add(k); ext.append((k, ua)); u += cfg["u_step_rs"]
    clocks = clocks + ext
    lo_j, hi_j = cfg["join_rs"][0] * rs, cfg["join_rs"][1] * rs
    joins = {k_: F.join_check(k_, lo_j, hi_j) for k_, _ in clocks}
    store.J["clocks"] = {"n_ruled": int(len(clocks) - len(ext)), "n_extended": len(ext), "records": [int(k_) for k_, _ in clocks], "u_over_rs": [float(u_) for _, u_ in clocks],
                         "first_return_at_launch_radius": fr, "formation": {"k": k_form, "T_over_rs": float(T[k_form]) / rs, "r_root": float(r_form)},
                         "launch_site": n_L, "R_L": R_L}
    log("clocks: %d ruled (u/r_s %.3f to %.3f on records %d to %d) + %d extended to u/r_s %s; first return at the launch radius %s r_s (from the start %s, the formation's light %s); join over %s r_s at the first launch: max |N - 1| %s, max |dv|/|v| %s" % (
        len(clocks) - len(ext), clocks[0][1], clocks[len(clocks) - len(ext) - 1][1], clocks[0][0], clocks[len(clocks) - len(ext) - 1][0], len(ext), _g(clocks[-1][1], "%.3f") if ext else "-",
        _g(fr.get("first_return_rs"), "%.1f"), _g(fr.get("first_return_from_start_rs"), "%.1f"), _g(fr.get("first_return_formation_rs"), "%.1f"), cfg["join_rs"], _g(joins[clocks[0][0]][0], "%.2e"), _g(joins[clocks[0][0]][1], "%.2e")))

    def extra_row(k):
        jn = joins.get(k) or F.join_check(k, lo_j, hi_j)
        return {"join_max_dN": jn[0], "join_max_rel_dv": jn[1], "join_window_rs": cfg["join_rs"]}
    maps = {}
    for kappa in cfg["kappas"]:
        specs = [{"k": k_, "u_rs": u_, "q": q, "omega": q / rs} for (k_, u_) in clocks for q in cfg["omega_rs"]]
        rows = run_channel(F, kappa, specs, cfg, rs, n_L, dev, store, extra_row)
        for q in cfg["omega_rs"]:
            rq = [r for r in rows if abs(r["omega_rs"] - q) < 1e-12 and r.get("T0") is not None]
            if len(rq) < 3:
                continue
            m = analyse_map(rq, rs, cfg)
            dkey = "k%d_q%06.3f" % (kappa, q)
            maps[dkey] = m
            store.J["maps"][dkey] = {k_: v_ for k_, v_ in m.items() if k_ != "arrays"}
            for name, arr in m["arrays"].items():
                store.Z["map_%s_%s" % (dkey, name)] = np.asarray(arr)
            px = m["plateau_x"]; hb = m["hawking_band_x"]
            log("map kappa %d q %.3f: plateau %s (%s x 1/(4GE)) over u/r_s [%s, %s], extent %s r_s (%s T_d, %s e-folds); Hawking band %s clocks over [%s, %s]; kappa max %s x 1/(4GE); k dispersive from u/r_s %s" % (
                kappa, q, _g(px.get("plateau"), "%.4f"), _g(px.get("plateau_x4GE"), "%.3f"), _g(px.get("u_start"), "%.3f"), _g(px.get("u_end"), "%.3f"), _g(px.get("extent"), "%.3f"),
                _g(m["plateau_x_extents"].get("extent_over_T_d_2gE"), "%.3f"), _g(m["plateau_x_extents"].get("e_folds_2gE"), "%.3f"), _g(hb.get("n_clocks"), "%d"), _g(hb.get("u_start"), "%.3f"),
                _g(hb.get("u_end"), "%.3f"), _g(m.get("kappa_x_max_x4GE"), "%.3f"), _g(m.get("first_clock_k_dispersive"), "%.3f")))
        store.flush()
    t0 = time.time()
    table = kappa_R_table(own, cfg, rs)
    ws = window_start(table, F, cfg, rs, R_L)
    store.J["kappa_R"] = {"table": table, "window_start": ws, "label": "kappa_R = (N - |v|)' at the outer root of the dense nodes (secant and quadratic fit), the site fields' slope and the densities' "
                          "(N_R/2R)[1 - 8 pi G R^2 (rho_R - J_R)] (page_d13_readout_v1.enclosed_source), the engine's kappa_T; each against 1/(4GE_k)"}
    for key in ("T_over_rs", "two_gE", "kappa_H", "R_t_json", "R_t_dense", "r_in_dense", "kappa_R_dense_secant", "kappa_R_dense_fit", "kappa_R_site_slope", "kappa_R_densities", "kappa_T_json",
                "kappa_R_dense_secant_x4GE", "kappa_R_dense_fit_x4GE", "kappa_R_site_slope_x4GE", "kappa_R_densities_x4GE", "kappa_T_json_x4GE", "min_N_minus_absv_json", "min_c_dense"):
        store.Z["kappaR_" + key] = np.array([row.get(key) if row.get(key) is not None else np.nan for row in table], float)
    for key, d in ws["columns"].items():
        if d.get("found"):
            log("window start (%s): record %d, T/r_s %.3f, ratio %.3f; %s" % (key, d["k"], d["T_over_rs"], d["ratio"], "; ".join("f %g: u chart %s, arrival %s" % (
                x["over_2gE"], _g(x["u_chart_rs"], "%.3f"), _g(x["u_at_launch_radius_rs"], "%.3f")) for x in d["launches"])))
        else:
            log("window start (%s): the fraction %g is never reached" % (key, cfg["kappaR_frac"]))
    log("record table: %d records in %.1f s" % (len(table), time.time() - t0))
    info_lines = ["history %s" % (own.label or a.record), "fields: %s (%s; v6m error: %s; compare: %s)" % (finfo["source"], own.fields_key, finfo.get("v6m_error"), finfo.get("compare")),
                  "clocks: %s" % json.dumps({k_: v_ for k_, v_ in store.J["clocks"].items() if k_ != "first_return_at_launch_radius"}, default=_jdefault),
                  "first return at the launch radius: %s" % json.dumps(fr, default=_jdefault), "device %s" % dev]
    write_md(os.path.join(a.out, "raymap_%s.md" % a.tag), store, cfg, rs, maps, info_lines)
    store.flush()
    log("done: %s, %s and %s" % (store.pj, store.pz, os.path.join(a.out, "raymap_%s.md" % a.tag)))


if __name__ == "__main__":
    main()
