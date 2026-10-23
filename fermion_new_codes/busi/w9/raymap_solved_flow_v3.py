#!/usr/bin/env python
"""raymap_solved_flow_v3.py (reviewer A for the busi lane; REVIEWA_REVIEW_web40i section 4, busi W9): the segment readout on the solved flow,
kappa(u)/kappa_H from 3 receiver jets propagated backward through the stored records, the onset and the dispersive end per frequency, the
e-fold count within 5 and within 20 percent of kappa_H. The second route on the solved flow for the window sentence's record values (Y1).

Form: a new module importing raymap_solved_flow_v2b (the code that wrote the B2 files B2_k1, B2_k2 and B2_k1_vac that RAYMAP_solved_flow_v1.3
read), not a set of replacements over it. Why. (1) Nothing numerical in v2b changes: the launch packet, the branch-edge rule, the
per-record Chebyshev propagation (page_d13_readout_v1), the T = 0 readouts, the geometric-optics ray and the record table are called by
import, so W9 reads the same records and fields through the same propagation product, byte for byte, and v2b's unitarity and
forward-backward checks transfer to every W9 row. (2) The W9 readout is a different object from v2b's plateau tables: its own clock grid
(from 6 r_s, to cover the files' window start 9.48 to 11.25 r_s that v2b's 12 r_s did not, v1.3 flag 4), its own probes (the 3 jets), a
band rule with two fractions, an onset, a dispersive end and an e-fold count; replacements inside v2b's run_channel, analyse_map and
write_md would have changed the B2 outputs the reading rests on, and about 15 OLD/NEW pairs would have been needed. (3) v2b's stores
(raymap_B2_k1 and the others) stay untouched; the W9 rows live in their own stores raymap_w9_<tag>_<jet> and are resumable there.
(4) raymap_w9_selftest_v1 imports both and tests the W9 analysis on manufactured inputs and on manufactured static flows through v2b's
propagation. v2b is found beside this file, in $RAYMAP_V2B_DIR, in the $PAGE_CODE_V6 directories, or in the solve lane's code directory
(read-only); the module's own name is printed at the start of every run.

Definitions (every threshold a named argument; raymap_w9_selftest_v1 checks them).
  The receiver jets (--jets "q@R_L[:kappa][:cap=<r_s>][:step=<r_s>]", default "TH@16:1:cap=8,1@16:1,2@16:1"). A jet is the family of
    outgoing probes of one frequency received at one radius in one radial channel, launched at the retarded clocks of the grid and
    propagated backward through the stored records to T = 0: each probe is v2b's unit outgoing packet on the local positive branch at the
    receiver, read out at T = 0 as v2b reads it. The three default jets, taken from the review's text: the receiver at 16 r_s, the collar
    of the D13 protocol's detector-energy ledger ("the cuts 16 and 30 r_s"), v2b's launch radius and the receiver radius of the review's
    arrival statements (Y16, "a receiver at radius R"); the frequencies omega = T_H, i.e. omega r_s = 1/(4 pi) = 0.0796 at the nominal
    r_s (the review's "lasts 0.3 r_s at omega = T_H"; omega over T_H at the launch record's charge prints per row), omega r_s = 1 (the
    review's "empty at omega r_s = 1", T_d = 2.31 r_s at r_s/a 64) and omega r_s = 2, the highest frequency the stored records resolved in
    the v1.3 reading (the review's "every frequency the records resolve"; its sine cutoff alpha_d = 0.397 per spacing is the nearest);
    the radial channel kappa 1 (the channel the v1.3 reading read; kappa 2 by argument). The thermal jet's position-width cap is 8 r_s
    instead of v2b's 4 r_s: at omega = T_H the carrier k_0 = 1.2e-3 per spacing lies half a spectral width above the kappa 1 centrifugal
    edge at 16 r_s under the 4 r_s cap (the cuts remove about a third of the Gaussian), about one width under 8 r_s; a probe at that
    frequency is not monochromatic on this chain (its wavelength is 79 r_s), which is itself a record value: the cut fraction, the edge
    distance and the duration in u print per row and per jet (v2b's near-edge rule, cut fraction above 0.1, flags the rows), and the
    comparator's segment at that frequency prints against them. A receiver at 30 r_s runs by --jets (RUN_W9_v1.md).
  The clocks: u/r_s from --u-from-rs 6 to --u-to-rs 40 in steps of --u-step-rs 0.5 (each moved to the nearest record clock, v2b's
    clock_records with the launch record's own charge), extended beyond 40 to the first return at the receiver or the last stored record
    (--extend on; on B2 the first return lies beyond the records, so the extension runs to T/r_s 100, about 139 clocks per jet).
  kappa_H on the solved flow: kappa_H(u) = 1/(4 G E_k) = 1/(2 x 2gE_k) with E_k the collar charge of the probe's launch record k (v2b's
    two_gE_launch), the surface gravity of the Painleve-Gullstrand exterior at that record's charge: v1.3's comparator for the probes'
    exponents (section 1.4, analyse_map's kH = rs/(2 two_gE_launch) per row; 0.602 per nominal r_s at the first launches on B2, 0.6035 to
    0.6038 from u 24 on). Beside it 1/(2 r_s nominal) and, in the record table, the record's own dense surface slope kappa_T in units of
    the record's own 1/(4GE_k) (section 2 item 10), the local conversion scale against which Y1 tests the computed sizes.
  The peeling rate kappa(u), per jet and clock, read 4 ways from the traced-back packet, each by centred differences at 1 and 2 clock
    steps with their difference as the error column:
    (k) kappa_k = d ln <|k|>_T0/du, the exponent of the traced-back mean wavenumber at T = 0: the blueshift's local rate, the primary
        readout (in geometric optics it is the ray map's -d ln|dU/du|/du; it carries no lapse-shift factor; v1.3 section 6 and flag 14);
    (E) kappa_E = d ln <h_0>/du, v2b's in-energy exponent, equal to kappa_k plus d ln(N_0 + |v_0|)/du for a narrow ingoing massless
        packet (v1.3 section 6), so the de-lapsed kappa_Ede = kappa_E - d ln c_in/du prints beside it, c_in = N_0 + |v_0| at the centroid
        (new in v3), with the identity's ratio <h_0>/(c_in <|k|>) per row;
    (U) kappa_U = -d ln|dU/du|/du, v2b's position map by log_diff_kappa (exact for an exponential map on a uniform grid); the incoming
        chart Uin = int_{R_L}^{U} dr/(N_0 + |v_0|) on the T = 0 slice (v1.3 flag 12) beside v2b's outgoing U_0;
    (ray) the Wigner-averaged geometric-optics ray's exponents from U_ray and from |k_ray| (v2b's ray_compare).
    The ratio r(u) = kappa(u)/kappa_H(u) per readout.
  The band rule: a clock is in the band(frac) when |r(u) - 1| <= frac and the 1-step/2-step difference is at most --band-err-frac x
    kappa_H ("same": frac itself); --band-fracs 0.05,0.20,0.50 (Y1's 5 and 20 percent; 0.50 the review's third onset constant). A run
    is a maximal set of consecutive in-band clocks with at least --band-min-clocks 2 (a one-clock crossing is not a plateau, v1.3 flag 2);
    every run prints, the one-clock crossings included, in the all-runs table.
  The onset(frac): the first clock of the first run that begins before the dispersive end (of the first run at all, where the
    dispersive end is not reached). The comparator's onset is the surface position f_s(frac) at which its ratio reaches 1 - frac.
  The dispersive end: the first clock at which <|k|>_T0 reaches alpha_d = (--cutoff-inner a omega)^(1/3) per spacing, --cutoff-inner 2
    (the review's sine cutoff; v1.3 section 1.4: 0.198, 0.250, 0.315, 0.397 at omega r_s 0.25, 0.5, 1, 2 and 0.136 at 1/(4 pi) for
    r_s/a 64); where it is not reached, the largest <|k|>_T0 prints as a fraction of alpha_d with the blueshift it carries. Beside it:
    the ray's |k_ray| against alpha_d, v2b's fixed thresholds (1/3 per spacing; half the weight beyond pi/4). The comparator's end is
    f_d = (2 a omega)^(2/3).
  The e-fold count: the sum over the segment's clocks of kappa^(1)_j h_j, h_j = (u_{j+1} - u_{j-1})/2, the integral of the peeling rate
    over the segment; for the k readout it telescopes to (1/2)[ln k_{i1+1} + ln k_{i1} - ln k_{i0} - ln k_{i0-1}], the log blueshift of
    <|k|>_T0 across the segment with half-step ends, printed beside with the plain ln(k_T0(end)/k_T0(start)). Beside it kappa_H's own
    count over the segment (sum kappa_H h_j) and the sum over every in-band clock, the isolated crossings included. The segment(frac) is
    the first run truncated at the dispersive end (the band's exit or the dispersive end, whichever is earlier); a segment shorter than
    2 sigma_u (the jet's median launch width in u, sigma_x/|v_g|) is resolution-limited.
  The comparator (the review's marginally bound comparator, section 1 A1; icloud X1 rederives it): kappa/kappa_H = 6/[(1 + f)^4 (2 + f)
    (3 + f)] with f = (R - r_s)/r_s; f_s(frac) the root of that ratio = 1 - frac; f_d = (2 a omega)^(2/3) at r_s/a = the nominal r_s;
    the retarded extent 2 r_s ln(f_s/f_d) where f_s > f_d, else empty; the e-fold count int_{f_d}^{f_s} (kappa/kappa_H) df/f, kappa_H's
    count ln(f_s/f_d) beside; the constant 2 r_s ln f_s against the leading (4/3) r_s ln(1/(2 a omega)). The constants the review quotes
    (f_s 0.021 at 95 percent and -7.6 r_s, -6.0 r_s at 80, -3.7 r_s at 50, 0.63 at f_d for omega r_s = 1, 0.3 r_s at omega = T_H) print
    beside the formula's own values with the difference where there is one (the formula's 95-percent root is 0.0107, not 0.021).
Outputs (data apart from any plot; this code plots nothing): <out>/raymap_w9_<tag>.md (the readout), raymap_w9_<tag>.json (every
scalar), raymap_w9_<tag>.npz (every array), raymap_w9_<tag>_<jet>.csv (the per-clock table per jet) and, per jet, v2b's store
raymap_w9_<tag>_<jet>.json/.npz (the rows and the T = 0 states; resumable per (u, omega), --force recomputes)."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.integrate as sint
import scipy.optimize as sopt

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "raymap_solved_flow_v3.py"
SOLVE_PAGE_DIR = "<internal>/page"
V2B_NAMES = ("raymap_solved_flow_v2b", "raymap_solved_flow_v2")


def _v2b_dirs():
    """where raymap_solved_flow_v2b.py lives: this directory, $RAYMAP_V2B_DIR, the $PAGE_CODE_V6 directories, code_v6 beside or above,
    the solve lane's code directory (read-only)."""
    c = [HERE]
    for key in ("RAYMAP_V2B_DIR", "PAGE_CODE_V6"):
        if os.environ.get(key):
            c += os.environ[key].split(os.pathsep)
    c += [os.path.join(HERE, "code_v6"), os.path.join(HERE, ".."), SOLVE_PAGE_DIR]
    return [os.path.abspath(x) for x in c]


def _import_v2b():
    for d in _v2b_dirs():
        if any(os.path.isfile(os.path.join(d, nm + ".py")) for nm in V2B_NAMES):
            if d not in sys.path:
                sys.path.insert(0, d)
            break
    err = None
    for nm in V2B_NAMES:
        try:
            return __import__(nm)
        except ImportError as ex:                                          # noqa: PERF203
            err = ex
    raise ImportError("raymap_solved_flow_v2b.py (or raymap_solved_flow_v2.py) not found beside this file, in $RAYMAP_V2B_DIR, in "
                      "$PAGE_CODE_V6 or in %s: %s" % (SOLVE_PAGE_DIR, err))


RM = _import_v2b()
OPS = RM.OPS
D13 = RM.D13
log = RM.log
_g = RM._g
_jdefault = RM._jdefault

Q_TH = 1.0 / (4.0 * math.pi)
W9_DEFAULTS = {"jets": "TH@16:1:cap=8,1@16:1,2@16:1", "u_from_rs": 6.0, "u_to_rs": 40.0, "u_step_rs": 0.5, "band_fracs": "0.05,0.20,0.50",
               "band_err_frac": "same", "band_min_clocks": 2, "cutoff_inner": 2.0}
REVIEW_QUOTED = {"f_s_at_0.95": 0.021, "constant_rs_at_0.95": -7.6, "constant_rs_at_0.80": -6.0, "constant_rs_at_0.50": -3.7,
                 "f_d_q1_rs64": 0.099, "ratio_at_f_d_q1_rs64": 0.63, "segment_rs_at_TH_0.95_rs64": 0.27,
                 "source": "REVIEWA_REVIEW_web40i, section 1 A1 and section 3 Y1; the comparator is rederived by icloud X1"}
READOUT_LABELS = {"k": "d ln <|k|>_T0/du, the traced-back wavenumber's exponent (primary)", "E": "d ln <h_0>/du, v2b's in-energy exponent",
                  "Ede": "the de-lapsed in-energy exponent, kappa_E - d ln(N_0 + |v_0|)/du", "U": "-d ln|dU/du|/du, v2b's position map",
                  "Uin": "the incoming chart int dr/(N_0 + |v_0|) at the centroid", "rayU": "the Wigner-averaged ray's position map",
                  "rayk": "the Wigner-averaged ray's wavenumber exponent"}
KAPPA_H_DEFINITION = ("kappa_H(u) = 1/(4 G E_k) = 1/(2 x 2gE_k), E_k the collar charge of the probe's launch record k (v2b's two_gE_launch): the "
                      "surface gravity of the Painleve-Gullstrand exterior at that record's charge, v1.3's comparator for the probes' exponents "
                      "(section 1.4: analyse_map's kH = rs/(2 two_gE_launch) per row; 0.602 per nominal r_s at the first launches on B2, 0.6035 to "
                      "0.6038 from u 24 on). Beside it 1/(2 r_s nominal) = 0.5 per nominal r_s and, in the record table, the record's own dense "
                      "surface slope kappa_T in units of the record's own 1/(4GE_k) (section 2 item 10), the local conversion scale against which "
                      "Y1 tests the computed sizes.")


# ------------------------------------------------------------------------------------------------------------------- the jets
def parse_jets(spec):
    """v3 W9: the receiver jets from "q@R_L[:kappa][:cap=<r_s>][:step=<r_s>]" items, q a number (omega r_s at the nominal r_s) or TH
    (omega = T_H at the nominal r_s, q = 1/(4 pi)); R_L the receiver radius in nominal r_s; kappa the radial channel (1); cap this jet's
    position-width cap of the rel-spread rule (v2b's --sigma-x-max-rs otherwise); step this jet's clock step (--u-step-rs otherwise)."""
    jets = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(":")]
        if "@" not in parts[0]:
            raise ValueError("jet %r: expected q@R_L[:kappa][:cap=<r_s>][:step=<r_s>]" % item)
        qs, Rs = [x.strip() for x in parts[0].split("@", 1)]
        thermal = qs.upper() in ("TH", "T_H", "THERMAL")
        q = Q_TH if thermal else float(qs)
        kappa = 1
        opts = {}
        for p in parts[1:]:
            if not p:
                continue
            if "=" in p:
                k_, v_ = p.split("=", 1)
                opts[k_.strip().lower()] = float(v_)
            else:
                kappa = int(p)
        R_L_rs = float(Rs)
        qname = "TH" if thermal else ("q%g" % q).replace(".", "p").replace("-", "m")
        name = "%s_R%s_k%d" % (qname, ("%g" % R_L_rs).replace(".", "p"), kappa)
        jets.append({"name": name, "spec": item, "q": q, "thermal": thermal, "q_over_q_TH": q / Q_TH, "R_L_rs": R_L_rs, "kappa": kappa,
                     "sigma_x_max_rs": opts.get("cap"), "u_step_rs": opts.get("step"), "omega": None})
    if not jets:
        raise ValueError("no jets in %r" % spec)
    return jets


def jet_clocks(F, own, R_L, rs, cfg, step=None):
    """v3 W9 (v2b's main, as a function): the ruled clocks u/r_s from --u-from-rs to --u-to-rs in the given step, each moved to the nearest
    record (v2b's clock_records with the launch record's own charge), the extension to the first return at the receiver or the last record,
    the first-return estimate and the formation record; ([(k, u)], [(k, u)] extended, first_return, formation)."""
    T = np.asarray(F.T, float)
    step = float(cfg["u_step_rs"] if step is None else step)
    k_form = int(np.argmin(np.abs(T - cfg["form_rs"] * rs)))
    r_form = D13.outer_root(own.N_rec[k_form], own.v_rec[k_form], own.R_rec)
    if r_form is None:
        r_form = F.two_gE(k_form)
    u_grid = np.arange(cfg["u_from_rs"], cfg["u_to_rs"] + 1e-9, step)
    clocks = RM.clock_records(F, R_L, rs, u_grid)
    if not clocks:
        raise ValueError("no record clock for u/r_s %g to %g at R_L %g" % (cfg["u_from_rs"], cfg["u_to_rs"], R_L))
    k_to = clocks[-1][0]
    N_to, v_to = F.fields(k_to)
    fr = D13.first_return(N_to, v_to, F.R, R_L, float(T[k_to]), float(T[0]), rs, float(T[k_form]), r_form)
    ext = []
    if cfg["extend"]:
        seen = set(k_ for k_, _ in clocks)
        u = cfg["u_to_rs"] + step
        while True:
            cr = RM.clock_records(F, R_L, rs, [u])
            if not cr:
                break
            k, ua = cr[0]
            if k in seen or (cfg["u_max_rs"] is not None and u > cfg["u_max_rs"] + 1e-9):
                break
            if fr.get("first_return_rs") is not None and T[k] / rs > fr["first_return_rs"]:
                break
            seen.add(k)
            ext.append((k, ua))
            u += step
    return clocks, ext, fr, {"k": k_form, "T_over_rs": float(T[k_form]) / rs, "r_root": float(r_form)}


# ---------------------------------------------------------------------------------------------------------- the W9 readouts
def add_w9_readouts(rows, F, n_L):
    """v3 W9: per row at T = 0 the ingoing lapse-shift factor c_in = N_0 + |v_0| at the centroid U (the identity <h_0> = c_in <|k|> for a
    narrow ingoing massless packet, v1.3 section 6), the outgoing factor c_out = N_0 - |v_0| there, the identity's ratio, and the incoming
    chart Uin = int_{R_L}^{U} dr/(N_0 + |v_0|) on the T = 0 slice (v1.3 flag 12: v2b's U_0 uses N_0 + v_0, the outgoing speed, while the
    traced-back packets are ingoing). Modifies the rows' T0 dicts in place."""
    N0, v0 = F.fields(0)
    R = np.asarray(F.R, float)
    cin = np.asarray(N0, float) + np.abs(np.asarray(v0, float))
    cout = np.asarray(N0, float) - np.abs(np.asarray(v0, float))
    inv = 1.0 / np.maximum(cin, 1e-300)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(R))])
    cum = cum - cum[int(n_L)]
    for r in rows:
        tz = r.get("T0")
        if tz is None:
            continue
        U = float(tz["U"])
        tz["c_in_at_U"] = float(np.interp(U, R, cin))
        tz["c_out_at_U"] = float(np.interp(U, R, cout))
        tz["Uin_chart"] = float(np.interp(U, R, cum))
        k = tz.get("k_mean_abs")
        tz["identity_h0_over_cin_k"] = (float(tz["energy_h0"]) / (tz["c_in_at_U"] * float(k))) if (k is not None and k > 0) else None


def jet_arrays(rows, rs):
    """the per-clock arrays of one jet from v2b's rows (sorted by u); (A, rows sorted)."""
    rows = sorted([r for r in rows if r.get("T0") is not None], key=lambda r: r["u_over_rs"])
    n = len(rows)

    def col(fn):
        out = np.full(n, np.nan)
        for i, r in enumerate(rows):
            try:
                v = fn(r)
                out[i] = float(v) if v is not None else np.nan
            except (KeyError, TypeError, ValueError):
                out[i] = np.nan
        return out
    A = {"u": col(lambda r: r["u_over_rs"]), "T_launch_over_rs": col(lambda r: r["T_launch_over_rs"]), "k_launch_record": col(lambda r: r["k_launch"]),
         "two_gE_launch": col(lambda r: r["two_gE_launch"]), "omega": col(lambda r: r["omega"]), "omega_over_TH": col(lambda r: r["omega_over_TH"]),
         "U": col(lambda r: r["T0"]["U"]), "sigma_U": col(lambda r: r["T0"]["sigma_U"]), "k_abs": col(lambda r: r["T0"]["k_mean_abs"]),
         "k_signed": col(lambda r: r["T0"]["k_mean_signed"]), "k_spread": col(lambda r: r["T0"]["k_spread"]), "E0": col(lambda r: r["T0"]["energy_h0"]),
         "E0_spread": col(lambda r: r["T0"]["energy_spread_h0"]), "c_in": col(lambda r: r["T0"]["c_in_at_U"]), "c_out": col(lambda r: r["T0"]["c_out_at_U"]),
         "Uin": col(lambda r: r["T0"]["Uin_chart"]), "U0_chart": col(lambda r: r["T0"]["U0_chart"]), "identity": col(lambda r: r["T0"]["identity_h0_over_cin_k"]),
         "frac_k_neg": col(lambda r: r["T0"]["frac_k_negative"]), "frac_k_super": col(lambda r: r["T0"]["frac_k_super"]), "frac_k_disp": col(lambda r: r["T0"]["frac_k_dispersive"]),
         "weight_inside_2gE0": col(lambda r: r["T0"]["weight_inside_2gE0"]), "norm_T0": col(lambda r: r["T0"]["norm_T0"]), "w_neg_in": col(lambda r: r["T0"]["w_neg_in"]),
         "U_ray": col(lambda r: r["ray"]["U_ray"]), "k_ray": col(lambda r: r["ray"]["k_ray"]), "ray_lost_frac": col(lambda r: r["ray"]["lost_frac"]),
         "sigma_u_rs": col(lambda r: r["launch"]["sigma_u_rs"]), "sigma_x_launch": col(lambda r: r["launch"]["sigma_x"]), "cut_frac": col(lambda r: r["launch"]["weight_cut_fraction"]),
         "edge_dist": col(lambda r: r["launch"]["k0_minus_kturn_over_sigma_k"]), "k_launch_abs": col(lambda r: r["launch"]["k_mean_abs"]),
         "launch_h_over_omega": col(lambda r: r["launch"]["energy_h_over_omega"]), "fb_defect": col(lambda r: r["fb"]["defect_norm"]),
         "join_max_dN": col(lambda r: r["join_max_dN"]), "join_max_rel_dv": col(lambda r: r["join_max_rel_dv"])}
    with np.errstate(divide="ignore", invalid="ignore"):
        A["kH_rs"] = rs / (2.0 * A["two_gE_launch"])
        A["kH_lat"] = 1.0 / (2.0 * A["two_gE_launch"])
        A["E0_over_omega"] = A["E0"] / A["omega"]
        A["blueshift_k"] = A["k_abs"] / A["k_launch_abs"]
    return A, rows


def exponents(A):
    """v3 W9: the peeling-rate readouts per clock, 1- and 2-step, in per r_s: k = d ln <|k|>_T0/du, E = d ln <h_0>/du, Ede = E minus
    d ln c_in/du, U = -d ln|dU/du|/du (v2b's log_diff_kappa), Uin the incoming chart's, rayU and rayk the Wigner-averaged ray's;
    {name: (kappa1, kappa2)}."""
    u = np.asarray(A["u"], float)
    n = len(u)
    ex = {}
    ex["k"] = RM.centred_log_derivative(u, np.asarray(A["k_abs"], float))
    ex["E"] = RM.centred_log_derivative(u, np.asarray(A["E0"], float))
    c1, c2 = RM.centred_log_derivative(u, np.asarray(A["c_in"], float))
    ex["Ede"] = (ex["E"][0] - c1, ex["E"][1] - c2)
    m = RM.map_derivatives(u, np.asarray(A["U"], float))
    ex["U"] = (m["kappa1"], m["kappa2"])
    A["dUdu1"] = m["dUdu1"]
    A["sign_change_U"] = m["sign_change"]
    Uin = np.asarray(A.get("Uin", np.full(n, np.nan)), float)
    if np.isfinite(Uin).sum() >= 5:
        m = RM.map_derivatives(u, Uin)
        ex["Uin"] = (m["kappa1"], m["kappa2"])
    Ur = np.asarray(A.get("U_ray", np.full(n, np.nan)), float)
    if np.isfinite(Ur).sum() >= 5:
        m = RM.map_derivatives(u, Ur)
        ex["rayU"] = (m["kappa1"], m["kappa2"])
    kr = np.abs(np.asarray(A.get("k_ray", np.full(n, np.nan)), float))
    if np.isfinite(kr).sum() >= 5:
        ex["rayk"] = RM.centred_log_derivative(u, kr)
    return ex


# ------------------------------------------------------------------------------------------------- the band and the segment
def band_runs(ratio, err, frac, err_frac, min_clocks):
    """the in-band clocks (|ratio - 1| <= frac, err <= err_frac, both finite) and their maximal consecutive runs of at least min_clocks;
    (runs [(i0, i1)], in-band mask)."""
    ratio = np.asarray(ratio, float)
    err = np.asarray(err, float)
    ok = np.isfinite(ratio) & np.isfinite(err) & (np.abs(ratio - 1.0) <= frac) & (err <= err_frac)
    runs = []
    n = len(ok)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        if j - i + 1 >= max(1, int(min_clocks)):
            runs.append((i, j))
        i = j + 1
    return runs, ok


def band_readout(u, h, step, kappa1, kH, ratio, err, k_abs, frac, err_frac, min_clocks, j_d, sigma_u_med):
    """v3 W9: the band(frac) readout of one peeling-rate series: the runs (min_clocks and all), the segment (the first run that begins
    before the dispersive end j_d, truncated at j_d), its e-fold count sum kappa1 h over its clocks (kappa_H's own count beside, the log
    blueshift of <|k|>_T0 across it plain and telescoped), the resolution flag against 2 sigma_u, the sum over every in-band clock."""
    u = np.asarray(u, float)
    n = len(u)
    kappa1 = np.asarray(kappa1, float)
    kH = np.asarray(kH, float)
    k_abs = np.asarray(k_abs, float)
    h = np.asarray(h, float)
    runs_all, ok = band_runs(ratio, err, frac, err_frac, 1)
    runs, _ = band_runs(ratio, err, frac, err_frac, min_clocks)

    def lnk(i):
        return math.log(k_abs[i]) if (0 <= i < n and np.isfinite(k_abs[i]) and k_abs[i] > 0) else None

    def stats(i0, i1):
        sl = slice(i0, i1 + 1)
        ext = float(u[i1] - u[i0] + step)
        ef = float(np.nansum(kappa1[sl] * h[sl]))
        efH = float(np.nansum(kH[sl] * h[sl]))
        a, b = lnk(i0), lnk(i1)
        am, bp = lnk(i0 - 1), lnk(i1 + 1)
        tele = 0.5 * (bp + b - a - am) if None not in (a, b, am, bp) else None
        rat = np.asarray(ratio, float)[sl]
        er = np.asarray(err, float)[sl]
        return {"i0": int(i0), "i1": int(i1), "u_start": float(u[i0]), "u_end": float(u[i1]), "n_clocks": int(i1 - i0 + 1), "extent_rs": ext,
                "efolds": ef, "efolds_at_kappa_H": efH, "efolds_over_kappa_H_count": (ef / efH) if efH else None,
                "log_blueshift_k": ((b - a) if None not in (a, b) else None), "log_blueshift_k_telescoped": tele,
                "ratio_mean": (float(np.nanmean(rat)) if np.isfinite(rat).any() else None), "ratio_min": (float(np.nanmin(rat)) if np.isfinite(rat).any() else None),
                "ratio_max": (float(np.nanmax(rat)) if np.isfinite(rat).any() else None), "err_mean": (float(np.nanmean(er)) if np.isfinite(er).any() else None),
                "resolution_limited": (bool(ext < 2.0 * sigma_u_med) if (sigma_u_med is not None and np.isfinite(sigma_u_med)) else None),
                "two_sigma_u_rs": (2.0 * float(sigma_u_med) if (sigma_u_med is not None and np.isfinite(sigma_u_med)) else None),
                "before_dispersive_end": bool(j_d is None or i1 < j_d)}
    seg = None
    truncated = False
    after = []
    for (i0, i1) in runs:
        if j_d is not None and i0 >= j_d:
            after.append(stats(i0, i1))
            continue
        if seg is None:
            j1 = i1
            if j_d is not None and i1 >= j_d:
                j1 = j_d - 1
                truncated = True
            if j1 >= i0:
                seg = stats(i0, j1)
    tot = float(np.nansum((kappa1 * h)[ok])) if ok.any() else 0.0
    totH = float(np.nansum((kH * h)[ok])) if ok.any() else 0.0
    return {"frac": float(frac), "err_frac": float(err_frac), "min_clocks": int(min_clocks), "n_inband_clocks": int(ok.sum()), "inband": ok,
            "onset_u": (seg["u_start"] if seg else None), "segment": seg, "segment_truncated_at_dispersive_end": bool(truncated),
            "runs": [stats(a_, b_) for a_, b_ in runs], "runs_all": [stats(a_, b_) for a_, b_ in runs_all], "runs_after_dispersive_end": after,
            "total_inband_efolds": tot, "total_inband_efolds_at_kappa_H": totH}


def analyse_jet_arrays(A, cfg, omega):
    """v3 W9: the segment readout of one jet from its per-clock arrays: the 7 peeling-rate readouts and their ratios to kappa_H, the
    dispersive end (the first clock with <|k|>_T0 >= alpha_d = (cutoff_inner omega)^(1/3)), and per readout and band fraction the onset,
    the segment, its e-fold count and the runs."""
    u = np.asarray(A["u"], float)
    n = len(u)
    kH = np.asarray(A["kH_rs"], float)
    step = float(np.median(np.diff(u))) if n > 1 else float(cfg.get("u_step_rs", 0.5))
    h = np.full(n, np.nan)
    if n >= 3:
        h[1:-1] = 0.5 * (u[2:] - u[:-2])
    if n >= 2:
        h[0] = u[1] - u[0]
        h[-1] = u[-1] - u[-2]
    omega = float(omega)
    alpha_d = (float(cfg["cutoff_inner"]) * omega) ** (1.0 / 3.0)
    ex = exponents(A)
    sig = np.asarray(A["sigma_u_rs"], float)
    sig_med = float(np.nanmedian(sig)) if np.isfinite(sig).any() else float("nan")
    k_abs = np.asarray(A["k_abs"], float)
    fin = np.isfinite(k_abs)
    j_d = next((j for j in range(n) if fin[j] and k_abs[j] >= alpha_d), None)
    kmax_j = int(np.nanargmax(k_abs)) if fin.any() else None
    kL = np.asarray(A.get("k_launch_abs", np.full(n, np.nan)), float)
    kL_med = float(np.nanmedian(kL)) if np.isfinite(kL).any() else float("nan")
    disp = {"alpha_d": alpha_d, "alpha_d_over_omega": (alpha_d / omega) if omega > 0 else None, "reached": j_d is not None, "j": j_d,
            "u": (float(u[j_d]) if j_d is not None else None), "k_T0_max": (float(k_abs[kmax_j]) if kmax_j is not None else None),
            "u_at_k_T0_max": (float(u[kmax_j]) if kmax_j is not None else None),
            "k_T0_max_over_alpha_d": (float(k_abs[kmax_j] / alpha_d) if kmax_j is not None else None),
            "blueshift_max": (float(k_abs[kmax_j] / kL[kmax_j]) if (kmax_j is not None and np.isfinite(kL[kmax_j]) and kL[kmax_j] > 0) else None),
            "blueshift_to_alpha_d": ((alpha_d / kL_med) if (np.isfinite(kL_med) and kL_med > 0) else None), "k_launch_abs_median": kL_med}
    kr = np.abs(np.asarray(A.get("k_ray", np.full(n, np.nan)), float))
    j_dr = next((j for j in range(n) if np.isfinite(kr[j]) and kr[j] >= alpha_d), None)
    disp["ray"] = {"reached": j_dr is not None, "u": (float(u[j_dr]) if j_dr is not None else None), "k_ray_max": (float(np.nanmax(kr)) if np.isfinite(kr).any() else None)}
    disp["v2b_k_dispersive"] = {"threshold": float(cfg["k_dispersive"]), "u": next((float(u[j]) for j in range(n) if fin[j] and k_abs[j] >= cfg["k_dispersive"]), None)}
    fs = np.asarray(A.get("frac_k_super", np.full(n, np.nan)), float)
    disp["v2b_half_weight_beyond_k_super"] = {"k_super": float(cfg["k_super"]), "u": next((float(u[j]) for j in range(n) if np.isfinite(fs[j]) and fs[j] >= 0.5), None)}
    fracs = [float(x) for x in cfg["band_fracs"]]
    em = cfg["band_err_frac"]
    minc = int(cfg["band_min_clocks"])
    out = {"n_clocks": n, "u_first": float(u[0]), "u_last": float(u[-1]), "step_rs": step, "sigma_u_rs_median": sig_med, "omega": omega,
           "kappa_H_per_rs_min": (float(np.nanmin(kH)) if np.isfinite(kH).any() else None), "kappa_H_per_rs_max": (float(np.nanmax(kH)) if np.isfinite(kH).any() else None),
           "dispersive_end": disp, "readouts": {}, "arrays": {"h": h, "u": u, "kH_rs": kH}}
    for name, (k1, k2) in ex.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = k1 / kH
            ratio2 = k2 / kH
            err = np.abs(k1 - k2) / kH
        fin_r = np.isfinite(ratio)
        R = {"label": READOUT_LABELS.get(name, name), "kappa1_per_rs": k1, "kappa2_per_rs": k2, "ratio1": ratio, "ratio2": ratio2, "err": err,
             "ratio_max": (float(np.nanmax(ratio)) if fin_r.any() else None), "u_ratio_max": (float(u[int(np.nanargmax(ratio))]) if fin_r.any() else None),
             "ratio_min": (float(np.nanmin(ratio)) if fin_r.any() else None), "u_ratio_min": (float(u[int(np.nanargmin(ratio))]) if fin_r.any() else None),
             "n_finite": int(fin_r.sum()), "bands": {}}
        for frac in fracs:
            ef = frac if (isinstance(em, str) and em == "same") else float(em)
            R["bands"]["%.2f" % frac] = band_readout(u, h, step, k1, kH, ratio, err, k_abs, frac, ef, minc, j_d, sig_med)
        out["readouts"][name] = R
    return out


# -------------------------------------------------------------------------------------------------------- the comparator
def comparator_ratio(f):
    """the review's marginally bound comparator: kappa/kappa_H = 6/[(1 + f)^4 (2 + f)(3 + f)], f = (R - r_s)/r_s (section 1 A1; X1 rederives)."""
    f = np.asarray(f, float)
    return 6.0 / ((1.0 + f) ** 4 * (2.0 + f) * (3.0 + f))


def comparator_onset(frac):
    """f_s(frac): the surface position at which the comparator's ratio reaches 1 - frac (brentq on [0, 100])."""
    target = 1.0 - float(frac)
    if not (0.0 < target < 1.0):
        return None
    return float(sopt.brentq(lambda f: float(comparator_ratio(f)) - target, 0.0, 100.0, xtol=1e-13, maxiter=500))


def comparator_cutoff(q, rs_over_a, inner=2.0):
    """f_d = (inner a omega)^(2/3) with a omega = q/(r_s/a): the review's sine cutoff."""
    return (float(inner) * float(q) / float(rs_over_a)) ** (2.0 / 3.0)


def comparator_segment(q, rs_over_a, frac, inner=2.0):
    """the comparator's segment within frac of kappa_H at omega r_s = q and r_s/a: f_s, f_d, the retarded extent 2 r_s ln(f_s/f_d) (in r_s;
    empty where f_s <= f_d), the e-fold count int_{f_d}^{f_s} (kappa/kappa_H) df/f and kappa_H's count ln(f_s/f_d), the constant 2 ln f_s
    against the leading (4/3) ln(1/(inner a omega))."""
    f_s = comparator_onset(frac)
    f_d = comparator_cutoff(q, rs_over_a, inner)
    a_omega = float(q) / float(rs_over_a)
    d = {"frac": float(frac), "f_s": f_s, "f_d": f_d, "ratio_at_f_d": float(comparator_ratio(f_d)), "a_omega": a_omega,
         "constant_rs": (2.0 * math.log(f_s) if f_s else None), "leading_rs": (4.0 / 3.0) * math.log(1.0 / (float(inner) * a_omega)),
         "empty": bool((f_s is None) or (f_s <= f_d))}
    if f_s is not None and f_s > f_d:
        d["extent_rs"] = 2.0 * math.log(f_s / f_d)
        d["efolds_at_kappa_H"] = math.log(f_s / f_d)
        d["efolds"] = float(sint.quad(lambda f: float(comparator_ratio(f)) / f, f_d, f_s)[0])
    else:
        d["extent_rs"] = 0.0
        d["efolds_at_kappa_H"] = 0.0
        d["efolds"] = 0.0
    return d


def comparator_table(q, rs_over_a, fracs, inner=2.0):
    """the comparator at one frequency and size, per band fraction, with the review's quoted constants and the formula's differences from them."""
    bands = {("%.2f" % f): comparator_segment(q, rs_over_a, f, inner) for f in fracs}
    b95 = comparator_segment(q, rs_over_a, 0.05, inner)
    b80 = comparator_segment(q, rs_over_a, 0.20, inner)
    b50 = comparator_segment(q, rs_over_a, 0.50, inner)
    q1 = comparator_segment(1.0, 64.0, 0.05, inner)
    th = comparator_segment(Q_TH, 64.0, 0.05, inner)
    quoted = REVIEW_QUOTED
    diff = {"f_s_at_0.95": b95["f_s"] - quoted["f_s_at_0.95"], "constant_rs_at_0.95": b95["constant_rs"] - quoted["constant_rs_at_0.95"],
            "constant_rs_at_0.80": b80["constant_rs"] - quoted["constant_rs_at_0.80"], "constant_rs_at_0.50": b50["constant_rs"] - quoted["constant_rs_at_0.50"],
            "f_d_q1_rs64": q1["f_d"] - quoted["f_d_q1_rs64"], "ratio_at_f_d_q1_rs64": q1["ratio_at_f_d"] - quoted["ratio_at_f_d_q1_rs64"],
            "segment_rs_at_TH_0.95_rs64": th["extent_rs"] - quoted["segment_rs_at_TH_0.95_rs64"],
            "segment_rs_at_TH_0.95_rs64_with_quoted_f_s": (2.0 * math.log(quoted["f_s_at_0.95"] / th["f_d"]) if quoted["f_s_at_0.95"] > th["f_d"] else 0.0) - quoted["segment_rs_at_TH_0.95_rs64"]}
    note = ("the formula's 95-percent root is f_s = %.4f (the constant %.2f r_s), not the review's quoted 0.021 (-7.6 r_s): with the formula's root the "
            "5-percent segment at omega = T_H and r_s/a 64 is empty (f_s %.4f against f_d %.4f), with the quoted 0.021 it is %.2f r_s, the review's 0.27; "
            "the 80- and 50-percent constants (%.2f, %.2f r_s) and the ratio %.3f at f_d for omega r_s = 1 agree with the quoted -6.0, -3.7 and 0.63. "
            "A note for icloud X1, which rederives the comparator; the solved-flow readout above does not depend on it." % (
                b95["f_s"], b95["constant_rs"], th["f_s"], th["f_d"], 2.0 * math.log(quoted["f_s_at_0.95"] / th["f_d"]) if quoted["f_s_at_0.95"] > th["f_d"] else 0.0,
                b80["constant_rs"], b50["constant_rs"], q1["ratio_at_f_d"]))
    return {"formula": "kappa/kappa_H = 6/[(1 + f)^4 (2 + f)(3 + f)], f = (R - r_s)/r_s; f_s(frac): ratio = 1 - frac; f_d = (inner a omega)^(2/3); extent 2 r_s ln(f_s/f_d); e-folds int (kappa/kappa_H) df/f",
            "inner": float(inner), "q": float(q), "rs_over_a": float(rs_over_a), "a_omega": float(q) / float(rs_over_a), "bands": bands,
            "quoted": quoted, "formula_minus_quoted": diff, "note": note}


# --------------------------------------------------------------------------------------------------------- one jet's run
def run_jet(F, own, jet, cfg, rs, dev, out, tag, settings):
    """build this jet's clocks, run v2b's run_channel on its probes in its own store, add the W9 readouts, analyse; (A, res, info, rows)."""
    t0 = time.time()
    R = np.asarray(F.R, float)
    n_L = int(np.argmin(np.abs(R - jet["R_L_rs"] * rs)))
    R_L = float(R[n_L])
    step = jet["u_step_rs"] if jet["u_step_rs"] is not None else cfg["u_step_rs"]
    clocks, ext, fr, form = jet_clocks(F, own, R_L, rs, cfg, step)
    allc = clocks + ext
    cfg_j = dict(cfg)
    if jet["sigma_x_max_rs"] is not None:
        cfg_j["sigma_x_max_rs"] = float(jet["sigma_x_max_rs"])
    prefix = os.path.join(out, "raymap_w9_%s_%s" % (tag, jet["name"]))
    label = "W9 jet %s: omega r_s %.5g (%.3g T_H at the nominal r_s), receiver %.4g r_s (R %.1f), kappa %d, on %s; light channel on %d sites" % (
        jet["name"], jet["q"], jet["q_over_q_TH"], R_L / rs, R_L, jet["kappa"], own.label or settings.get("record"), F.nsites)
    store = RM.RayStore(prefix, settings=dict(settings, jet=jet, cfg={k_: v_ for k_, v_ in cfg_j.items()}), label=label)
    lo_j, hi_j = cfg["join_rs"][0] * rs, cfg["join_rs"][1] * rs
    joins = {}

    def extra_row(k):
        if k not in joins:
            joins[k] = F.join_check(k, lo_j, hi_j)
        jn = joins[k]
        return {"join_max_dN": jn[0], "join_max_rel_dv": jn[1], "join_window_rs": cfg["join_rs"], "jet": jet["name"], "R_L_rs": R_L / rs}
    branch = []
    for k_, u_ in (allc[0], allc[-1]):
        N_, v_ = F.fields(k_)
        be = RM.branch_edges(float(N_[n_L]), float(v_[n_L]), R_L, jet["kappa"], F.m)
        branch.append({"k": int(k_), "u_rs": float(u_), "omega_turn": be["omega_turn"], "q_turn": (be["omega_turn"] * rs if be["omega_turn"] is not None else None),
                       "omega_peak": be["omega_peak"], "q_peak": (be["omega_peak"] * rs if be["omega_peak"] is not None else None), "gap_small_k": be["gap_small_k"]})
    log("jet %s: %d ruled + %d extended clocks (u/r_s %.3f to %.3f, step %g, records %d to %d), receiver site %d (R %.1f = %.3f r_s), q %.5g (omega %.4e; %.3g T_H nominal), kappa %d, cap %s r_s; "
        "branch at the receiver: q_turn %s to %s; first return %s r_s (an estimate)" % (
            jet["name"], len(clocks), len(ext), allc[0][1], allc[-1][1], step, allc[0][0], allc[-1][0], n_L, R_L, R_L / rs, jet["q"], jet["omega"], jet["q_over_q_TH"], jet["kappa"],
            _g(cfg_j["sigma_x_max_rs"], "%g"), _g(branch[0]["q_turn"], "%.4f"), _g(branch[1]["q_turn"], "%.4f"), _g(fr.get("first_return_rs"), "%.1f")))
    specs = [{"k": k_, "u_rs": u_, "q": jet["q"], "omega": jet["omega"]} for (k_, u_) in allc]
    rows = RM.run_channel(F, jet["kappa"], specs, cfg_j, rs, n_L, dev, store, extra_row)
    good = [r for r in rows if r.get("T0") is not None]
    add_w9_readouts(good, F, n_L)
    store.flush()
    A, good = jet_arrays(good, rs)
    if len(good) < 5:
        raise RuntimeError("jet %s: only %d analysable rows" % (jet["name"], len(good)))
    res = analyse_jet_arrays(A, cfg, jet["omega"])
    info = {"n_L": n_L, "R_L": R_L, "R_L_rs": R_L / rs, "n_ruled": len(clocks), "n_extended": len(ext), "records": [int(k) for k, _ in allc], "u_over_rs": [float(x) for _, x in allc],
            "first_return": fr, "formation": form, "branch_at_receiver": branch, "n_rows": len(rows), "n_analysed": len(good),
            "n_refused": int(sum(1 for r in rows if r.get("refused"))), "n_near_edge": int(sum(1 for r in good if ((r.get("launch") or {}).get("weight_cut_fraction") or 0.0) > 0.1)),
            "n_flagged": int(sum(1 for r in good if ((r.get("T0") or {}).get("norm_flag") or (r.get("fb") or {}).get("fb_flag")))),
            "cut_frac_max": (float(np.nanmax(A["cut_frac"])) if np.isfinite(A["cut_frac"]).any() else None), "edge_dist_min": (float(np.nanmin(A["edge_dist"])) if np.isfinite(A["edge_dist"]).any() else None),
            "sigma_x_max_rs_used": cfg_j["sigma_x_max_rs"], "omega_over_TH_first": (float(A["omega_over_TH"][0]) if len(A["omega_over_TH"]) else None),
            "store_prefix": prefix, "store_notes": list(store.J.get("notes") or []), "seconds": time.time() - t0}
    return A, res, info, good


def record_values(jet, res, comp, info):
    """the compact record values of one jet for Y1's second route: per band fraction and readout the onset, the segment, its e-fold count
    and resolution, the dispersive end, the comparator beside."""
    rv = {"jet": jet["name"], "q": jet["q"], "q_over_q_TH": jet["q_over_q_TH"], "omega_over_TH_first_launch": info.get("omega_over_TH_first"), "R_L_rs": info["R_L_rs"],
          "kappa": jet["kappa"], "n_clocks": res["n_clocks"], "u_first": res["u_first"], "u_last": res["u_last"], "step_rs": res["step_rs"],
          "sigma_u_rs_median": res["sigma_u_rs_median"], "cut_frac_max": info.get("cut_frac_max"), "edge_dist_min": info.get("edge_dist_min"), "n_near_edge": info.get("n_near_edge"),
          "kappa_H_per_rs": [res["kappa_H_per_rs_min"], res["kappa_H_per_rs_max"]], "kappa_H_definition": KAPPA_H_DEFINITION,
          "dispersive_end": {k_: v_ for k_, v_ in res["dispersive_end"].items()}, "bands": []}
    for key in sorted(res["readouts"]["k"]["bands"].keys(), key=float):
        for name in ("k", "E", "Ede", "U", "Uin", "rayk", "rayU"):
            if name not in res["readouts"]:
                continue
            b = res["readouts"][name]["bands"][key]
            seg = b["segment"]
            row = {"frac": b["frac"], "readout": name, "onset_u": b["onset_u"], "n_inband_clocks": b["n_inband_clocks"], "total_inband_efolds": b["total_inband_efolds"],
                   "total_inband_efolds_at_kappa_H": b["total_inband_efolds_at_kappa_H"], "truncated_at_dispersive_end": b["segment_truncated_at_dispersive_end"],
                   "n_runs": len(b["runs"]), "n_runs_after_dispersive_end": len(b["runs_after_dispersive_end"]), "ratio_max": res["readouts"][name]["ratio_max"], "u_ratio_max": res["readouts"][name]["u_ratio_max"]}
            if seg:
                row.update({"u_end": seg["u_end"], "n_clocks": seg["n_clocks"], "extent_rs": seg["extent_rs"], "efolds": seg["efolds"], "efolds_at_kappa_H": seg["efolds_at_kappa_H"],
                            "log_blueshift_k": seg["log_blueshift_k"], "ratio_mean": seg["ratio_mean"], "resolution_limited": seg["resolution_limited"], "two_sigma_u_rs": seg["two_sigma_u_rs"]})
            else:
                row.update({"u_end": None, "n_clocks": 0, "extent_rs": 0.0, "efolds": 0.0, "efolds_at_kappa_H": 0.0, "log_blueshift_k": None, "ratio_mean": None, "resolution_limited": None, "two_sigma_u_rs": None})
            cb = comp["bands"].get(key)
            if cb:
                row["comparator"] = {"f_s": cb["f_s"], "f_d": cb["f_d"], "empty": cb["empty"], "extent_rs": cb["extent_rs"], "efolds": cb["efolds"], "efolds_at_kappa_H": cb["efolds_at_kappa_H"], "ratio_at_f_d": cb["ratio_at_f_d"]}
            rv["bands"].append(row)
    return rv


def jet_sentence(jet, res, comp, info):
    """the computed sentence of one jet (the k readout; the record values in words)."""
    k = res["readouts"]["k"]
    de = res["dispersive_end"]
    s = ["jet %s (omega r_s %.4g = %.3g T_H at the nominal r_s, %s T_H at the first launch's charge; receiver %.3g r_s, kappa %d; %d clocks u/r_s %.3f to %.3f at step %g; "
         "probe duration sigma_u median %.2f r_s, cut fraction max %s, edge distance min %s sigma_k, %d rows near the edge): the traced-back wavenumber's exponent peaks at %s of kappa_H "
         "at u/r_s %s" % (jet["name"], jet["q"], jet["q_over_q_TH"], _g(info.get("omega_over_TH_first"), "%.2f"), info["R_L_rs"], jet["kappa"], res["n_clocks"], res["u_first"], res["u_last"],
                           res["step_rs"], res["sigma_u_rs_median"], _g(info.get("cut_frac_max"), "%.3f"), _g(info.get("edge_dist_min"), "%.2f"), info.get("n_near_edge", 0),
                           _g(k["ratio_max"], "%.3f"), _g(k["u_ratio_max"], "%.3f"))]
    for key in sorted(k["bands"].keys(), key=float):
        b = k["bands"][key]
        seg = b["segment"]
        pct = int(round(100 * b["frac"]))
        cb = comp["bands"].get(key)
        comps = ("comparator: empty (f_s %.4f below f_d %.4f, ratio %.2f at f_d)" % (cb["f_s"], cb["f_d"], cb["ratio_at_f_d"])) if (cb and cb["empty"]) else \
                ("comparator: %.2f r_s, %.3f e-folds (%.3f at kappa_H), f_s %.4f to f_d %.4f" % (cb["extent_rs"], cb["efolds"], cb["efolds_at_kappa_H"], cb["f_s"], cb["f_d"])) if cb else ""
        if seg:
            s.append("within %d percent: onset u/r_s %.3f, segment to %.3f (%d clocks, %.2f r_s%s%s), e-folds %.3f (kappa_H's count %.3f; ln k_end/k_start %s), ratio mean %.3f; %d in-band clocks in all carrying %.3f e-folds; %s" % (
                pct, seg["u_start"], seg["u_end"], seg["n_clocks"], seg["extent_rs"], ", resolution-limited against 2 sigma_u %.2f r_s" % seg["two_sigma_u_rs"] if seg["resolution_limited"] else "",
                ", truncated at the dispersive end" if b["segment_truncated_at_dispersive_end"] else "", seg["efolds"], seg["efolds_at_kappa_H"], _g(seg["log_blueshift_k"], "%.3f"),
                seg["ratio_mean"], b["n_inband_clocks"], b["total_inband_efolds"], comps))
        else:
            s.append("within %d percent: no run of %d clocks (%d isolated in-band clocks carrying %.3f e-folds in all); %s" % (pct, b["min_clocks"], b["n_inband_clocks"], b["total_inband_efolds"], comps))
    if de["reached"]:
        s.append("the dispersive end (<|k|>_T0 reaching alpha_d %.3f per spacing, %.1f times the launch wavenumber) at u/r_s %.3f" % (de["alpha_d"], de["blueshift_to_alpha_d"] or float("nan"), de["u"]))
    else:
        s.append("the dispersive end not reached: <|k|>_T0 peaks at %s per spacing at u/r_s %s, %s of alpha_d %.3f (a blueshift of %s against the %s the cutoff needs)" % (
            _g(de["k_T0_max"], "%.4f"), _g(de["u_at_k_T0_max"], "%.3f"), _g(de["k_T0_max_over_alpha_d"], "%.3f"), de["alpha_d"], _g(de["blueshift_max"], "%.2f"), _g(de["blueshift_to_alpha_d"], "%.1f")))
    return "; ".join(s) + "."


def strip_arrays(obj, sink=None, prefix=""):
    """a copy of obj without numpy arrays (they go to sink under prefix + key, a marker stays); the json's scalars."""
    if isinstance(obj, dict):
        out = {}
        for k_, v_ in obj.items():
            if isinstance(v_, np.ndarray):
                if sink is not None:
                    sink[prefix + str(k_)] = v_
                out[k_] = {"array_in_npz": prefix + str(k_), "shape": list(v_.shape)}
            else:
                out[k_] = strip_arrays(v_, sink, prefix + str(k_) + "_")
        return out
    if isinstance(obj, (list, tuple)):
        return [strip_arrays(x, sink, prefix) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------------------------------------------- the files
def write_csv(path, A, res, rs):
    """the per-clock table of one jet as plain data (apart from the md and from any plot)."""
    Rd = res["readouts"]
    alpha = res["dispersive_end"]["alpha_d"]
    cols = [("u_over_rs", A["u"]), ("T_launch_over_rs", A["T_launch_over_rs"]), ("launch_record", A["k_launch_record"]), ("two_gE_launch", A["two_gE_launch"]),
            ("kappa_H_per_rs", A["kH_rs"]), ("U_over_rs", A["U"] / rs), ("sigma_U_over_rs", A["sigma_U"] / rs), ("k_abs_T0", A["k_abs"]), ("k_signed_T0", A["k_signed"]),
            ("k_abs_over_alpha_d", A["k_abs"] / alpha), ("blueshift_k", A["blueshift_k"]), ("h0_over_omega", A["E0_over_omega"]), ("c_in_at_U", A["c_in"]), ("identity_h0_over_cin_k", A["identity"]),
            ("Uin_chart", A["Uin"]), ("U_ray_over_rs", A["U_ray"] / rs), ("k_ray", A["k_ray"]), ("sigma_u_rs", A["sigma_u_rs"]), ("cut_frac", A["cut_frac"]), ("edge_dist", A["edge_dist"]),
            ("frac_k_neg_T0", A["frac_k_neg"]), ("frac_k_super_T0", A["frac_k_super"]), ("norm_T0_minus_1", A["norm_T0"] - 1.0), ("fb_defect", A["fb_defect"])]
    for name in ("k", "E", "Ede", "U", "Uin", "rayU", "rayk"):
        if name in Rd:
            cols += [("kappa1_%s_per_rs" % name, Rd[name]["kappa1_per_rs"]), ("ratio_%s" % name, Rd[name]["ratio1"]), ("err_%s" % name, Rd[name]["err"])]
            for key, b in Rd[name]["bands"].items():
                cols.append(("inband_%s_%s" % (name, key), b["inband"].astype(float)))
    n = len(A["u"])
    with open(path, "w") as f:
        f.write(",".join(c for c, _ in cols) + "\n")
        for j in range(n):
            f.write(",".join(("%.10g" % float(arr[j])) if np.isfinite(float(arr[j])) else "nan" for _, arr in cols) + "\n")


DEFINITIONS_MD = [
    "- A receiver jet: the family of outgoing probes of one frequency received at one radius in one radial channel, launched at the retarded clocks of the grid and propagated backward through the stored records to T = 0 (v2b's unit outgoing packet on the local positive branch at the receiver, v2b's per-record Chebyshev propagation and T = 0 readouts, by import). The default jets: the receiver 16 r_s (the D13 ledger's collar and v2b's launch radius; Y16's receiver radius), the frequencies omega = T_H (omega r_s = 1/(4 pi) at the nominal r_s), omega r_s = 1 and omega r_s = 2, the channel kappa 1; the thermal jet's position-width cap 8 r_s (v2b's 4 r_s otherwise).",
    "- kappa_H on the solved flow: " + KAPPA_H_DEFINITION,
    "- kappa(u), the peeling rate, 7 readouts of the traced-back packet, each by centred differences at 1 and 2 clock steps with their difference as the error column: k = d ln <|k|>_T0/du, the traced-back mean wavenumber's exponent (the blueshift's local rate; the primary readout: in geometric optics the ray map's -d ln|dU/du|/du, with no lapse-shift factor; v1.3 section 6, flag 14); E = d ln <h_0>/du (v2b's in-energy, = k plus d ln(N_0 + |v_0|)/du for a narrow ingoing massless packet); Ede = E minus d ln c_in/du, c_in = N_0 + |v_0| at the centroid (new in v3; the identity <h_0>/(c_in <|k|>) prints per row); U = -d ln|dU/du|/du (v2b's position map, log_diff_kappa, exact for an exponential map on a uniform grid); Uin from the incoming chart int_{R_L}^{U} dr/(N_0 + |v_0|) (flag 12); rayU and rayk from the Wigner-averaged geometric-optics ray (v2b's ray_compare). The ratio r(u) = kappa(u)/kappa_H(u).",
    "- The band(frac): a clock is in the band when |r(u) - 1| <= frac and the 1-step/2-step difference is at most band_err_frac x kappa_H (same: frac). A run: consecutive in-band clocks, at least band_min_clocks (a one-clock crossing is not a plateau, v1.3 flag 2); the all-runs table lists every run down to 1 clock.",
    "- The onset(frac): the first clock of the first run that begins before the dispersive end (of the first run, where the dispersive end is not reached). The comparator's onset: the surface position f_s(frac) at which its ratio reaches 1 - frac.",
    "- The dispersive end: the first clock at which <|k|>_T0 reaches alpha_d = (cutoff_inner a omega)^(1/3) per spacing (the review's sine cutoff; v1.3 section 1.4); where not reached, the largest <|k|>_T0 as a fraction of alpha_d with the blueshift it carries; beside it the ray's |k_ray|, v2b's fixed 1/3 per spacing and half the weight beyond pi/4. The comparator's end: f_d = (2 a omega)^(2/3).",
    "- The e-fold count: the sum over the segment's clocks of kappa^(1)_j h_j, h_j = (u_{j+1} - u_{j-1})/2 (the integral of the peeling rate; for the k readout the telescoped log blueshift (1/2)[ln k_{i1+1} + ln k_{i1} - ln k_{i0} - ln k_{i0-1}], printed with the plain ln(k_end/k_start)); kappa_H's own count sum kappa_H h_j beside, and the sum over every in-band clock (the isolated crossings included). The segment(frac): the first run truncated at the dispersive end; shorter than 2 sigma_u (the jet's median launch width in u) it is resolution-limited.",
    "- The comparator: the review's marginally bound comparator kappa/kappa_H = 6/[(1 + f)^4 (2 + f)(3 + f)], f = (R - r_s)/r_s (section 1 A1; icloud X1 rederives it); f_s(frac) its onset, f_d = (2 a omega)^(2/3) its end at r_s/a = the nominal r_s; the retarded extent 2 r_s ln(f_s/f_d) where positive; the e-fold count int_{f_d}^{f_s} (kappa/kappa_H) df/f, kappa_H's count ln(f_s/f_d) beside; the review's quoted constants beside the formula's values with the differences.",
]


def write_md(path, J, jets_out, cfg, rs, table, ws):
    S = J["settings"]
    L = ["# RAYMAP_W9_v1: %s (raymap_solved_flow_v3 over %s)" % (S.get("tag", ""), RM.CODE), "", J.get("label") or "", ""]
    L += ["## The form of the code", "",
          "A new module importing %s from %s; nothing numerical in v2b changed: the packet, the branch-edge rule, the propagation, the T = 0 readouts, the ray comparator and the record table are v2b's by import, so this readout is on the same records and fields through the same propagation product as the files RAYMAP_solved_flow_v1.3 read. New in v3: the jets, the clock grid from %g r_s, the ingoing lapse-shift factor and the incoming chart at T = 0, the 7 peeling-rate readouts, the band rule, the onset, the dispersive end, the e-fold count, the comparator; v2b's stores are untouched, the W9 rows live in their own stores (%s)." % (
              RM.CODE, J["v2b"]["file"], cfg["u_from_rs"], ", ".join(os.path.basename(jo["info"]["store_prefix"]) for jo in jets_out)), ""]
    L += ["## Settings (every named argument)", "", "    " + json.dumps({k_: v_ for k_, v_ in cfg.items() if k_ != "force"}, default=_jdefault), ""]
    L += ["## Definitions", ""] + DEFINITIONS_MD + [""]
    L += ["## The history, the fields and the clocks", "", "- history %s" % J["history"],
          "- fields: %s (%s; v6m error: %s; compare: %s)" % (J["fields"]["source"], J["fields"].get("fields_key"), J["fields"].get("v6m_error"), J["fields"].get("compare")),
          "- r_s nominal %g spacings; device %s; v2b arguments --omega-rs, --kappas and --launch-rs are not used by v3 (the jets define them)" % (rs, S.get("device"))]
    for jo in jets_out:
        inf = jo["info"]
        L.append("- jet %s: %d ruled + %d extended clocks, u/r_s %.3f to %.3f on records %d to %d; receiver site %d (R %.1f = %.3f r_s); first return at the receiver %s r_s (%s); formation record %d at T/r_s %.3f (root %.3f spacings); the branch at the receiver: q_turn %s (first clock) to %s (last), q_peak %s" % (
            jo["jet"]["name"], inf["n_ruled"], inf["n_extended"], inf["u_over_rs"][0], inf["u_over_rs"][-1], inf["records"][0], inf["records"][-1], inf["n_L"], inf["R_L"], inf["R_L_rs"],
            _g(inf["first_return"].get("first_return_rs"), "%.1f"), inf["first_return"].get("label", "")[:60], inf["formation"]["k"], inf["formation"]["T_over_rs"], inf["formation"]["r_root"],
            _g(inf["branch_at_receiver"][0]["q_turn"], "%.4f"), _g(inf["branch_at_receiver"][1]["q_turn"], "%.4f"), _g(inf["branch_at_receiver"][0]["q_peak"], "%.3f")))
    L.append("")
    L += ["## The jets", "", "| jet | omega r_s | omega/T_H (nominal) | omega/T_H (first launch) | receiver r_s | kappa | cap r_s | clocks | u/r_s range | sigma_u median r_s | cut fraction max | edge distance min (sigma_k) | near-edge rows | refused | flagged | alpha_d (per spacing) | T_d nominal r_s | seconds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for jo in jets_out:
        j_, inf, res = jo["jet"], jo["info"], jo["res"]
        L.append("| %s | %.5g | %.3g | %s | %.3g | %d | %s | %d | %.3f to %.3f | %.2f | %s | %s | %d | %d | %d | %.4f | %.3f | %.0f |" % (
            j_["name"], j_["q"], j_["q_over_q_TH"], _g(inf.get("omega_over_TH_first"), "%.2f"), inf["R_L_rs"], j_["kappa"], _g(inf["sigma_x_max_rs_used"], "%g"), res["n_clocks"], res["u_first"], res["u_last"],
            res["sigma_u_rs_median"], _g(inf.get("cut_frac_max"), "%.3f"), _g(inf.get("edge_dist_min"), "%.2f"), inf["n_near_edge"], inf["n_refused"], inf["n_flagged"],
            res["dispersive_end"]["alpha_d"], j_.get("T_d_nominal_rs", float("nan")), inf["seconds"]))
    L.append("")
    L += ["## Record values for Y1 (the window sentence's second route on the solved flow)", "",
          "Computed values, labelled computed (RAYMAP_W9_v1); 1 history; the light radial channel; the receiver and frequencies of the jets table; the probe durations against the segments as printed; the e-fold count is the integral of the peeling rate over the segment within the stated fraction of kappa_H, kappa_H = 1/(4GE) at the probe's launch record.", ""]
    for jo in jets_out:
        j_, res, comp, inf = jo["jet"], jo["res"], jo["comp"], jo["info"]
        de = res["dispersive_end"]
        L += ["### Jet %s" % j_["name"], "", "- " + jo["sentence"], "",
              "- kappa_H = 1/(4GE_launch) runs %s to %s per nominal r_s over the clocks (1/(2 r_s nominal) = 0.5); the k readout's ratio to it: max %s at u/r_s %s, min %s at %s" % (
                  _g(res["kappa_H_per_rs_min"], "%.4f"), _g(res["kappa_H_per_rs_max"], "%.4f"), _g(res["readouts"]["k"]["ratio_max"], "%.3f"), _g(res["readouts"]["k"]["u_ratio_max"], "%.3f"),
                  _g(res["readouts"]["k"]["ratio_min"], "%.3f"), _g(res["readouts"]["k"]["u_ratio_min"], "%.3f")),
              "- the dispersive end: alpha_d %.4f per spacing (%.1f times the launch wavenumber %s): %s; the ray's |k_ray| %s alpha_d (max %s); v2b's 1/3 threshold %s; half the weight beyond pi/4 %s" % (
                  de["alpha_d"], de["blueshift_to_alpha_d"] or float("nan"), _g(de["k_launch_abs_median"], "%.4f"),
                  ("reached at u/r_s %.3f" % de["u"]) if de["reached"] else ("not reached; max <|k|>_T0 %s at u/r_s %s = %s of alpha_d" % (_g(de["k_T0_max"], "%.4f"), _g(de["u_at_k_T0_max"], "%.3f"), _g(de["k_T0_max_over_alpha_d"], "%.3f"))),
                  "reaches" if de["ray"]["reached"] else "does not reach", _g(de["ray"]["k_ray_max"], "%.4f"), ("at u/r_s %.3f" % de["v2b_k_dispersive"]["u"]) if de["v2b_k_dispersive"]["u"] is not None else "never reached",
                  ("at u/r_s %.3f" % de["v2b_half_weight_beyond_k_super"]["u"]) if de["v2b_half_weight_beyond_k_super"]["u"] is not None else "never reached"), "",
              "| frac | readout | onset u/r_s | segment end u/r_s | clocks | extent r_s | e-folds | kappa_H count | ln(k_end/k_start) | ratio mean | 2 sigma_u r_s | resolution | truncated at disp. end | in-band clocks (all) | total in-band e-folds | comparator f_s | f_d | comparator extent r_s | comparator e-folds | comparator kappa_H count |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in jo["record_values"]["bands"]:
            cb = row.get("comparator") or {}
            L.append("| %.2f | %s | %s | %s | %d | %.3f | %.4f | %.4f | %s | %s | %s | %s | %s | %d | %.4f | %s | %s | %s | %s | %s |" % (
                row["frac"], row["readout"], _g(row["onset_u"], "%.3f"), _g(row["u_end"], "%.3f"), row["n_clocks"], row["extent_rs"], row["efolds"], row["efolds_at_kappa_H"], _g(row["log_blueshift_k"], "%.4f"),
                _g(row["ratio_mean"], "%.3f"), _g(row["two_sigma_u_rs"], "%.2f"), ("limited" if row["resolution_limited"] else ("resolved" if row["resolution_limited"] is not None else "-")),
                "yes" if row["truncated_at_dispersive_end"] else "no", row["n_inband_clocks"], row["total_inband_efolds"], _g(cb.get("f_s"), "%.4f"), _g(cb.get("f_d"), "%.4f"),
                ("empty" if cb.get("empty") else _g(cb.get("extent_rs"), "%.3f")) if cb else "-", _g(cb.get("efolds"), "%.3f") if cb else "-", _g(cb.get("efolds_at_kappa_H"), "%.3f") if cb else "-"))
        L.append("")
    comp0 = jets_out[0]["comp"]
    L += ["## The comparator (the review's marginally bound comparator; icloud X1 rederives it)", "", "- " + comp0["formula"], "- " + comp0["note"], "",
          "| jet | omega r_s | a omega | frac | f_s (formula) | f_d | ratio at f_d | constant 2 r_s ln f_s | leading (4/3) r_s ln(1/(2 a omega)) | extent r_s | e-folds | kappa_H count |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for jo in jets_out:
        c = jo["comp"]
        for key in sorted(c["bands"].keys(), key=float):
            b = c["bands"][key]
            L.append("| %s | %.5g | %.4e | %.2f | %.5f | %.5f | %.3f | %.3f | %.3f | %s | %.4f | %.4f |" % (
                jo["jet"]["name"], c["q"], c["a_omega"], b["frac"], b["f_s"], b["f_d"], b["ratio_at_f_d"], b["constant_rs"], b["leading_rs"], "empty" if b["empty"] else "%.3f" % b["extent_rs"], b["efolds"], b["efolds_at_kappa_H"]))
    L += ["", "The review's quoted constants and the formula's values at r_s/a 64: " + "; ".join("%s: quoted %s, formula minus quoted %s" % (k_, _g(comp0["quoted"].get(k_), "%.4g"), _g(v_, "%.4g")) for k_, v_ in comp0["formula_minus_quoted"].items()), ""]
    L += ["## The record table: kappa_R against the record's own 1/(4GE_k), with the record's charge", "",
          "v2b's kappa_R_table: every x4GE column is the slope over the record's own 1/(4GE_k), 2gE_k = 2 g E_k with E_k the record's collar charge (v1.3 section 2 item 10); the 2gE_k column and R_t/2gE_k are new in v3 (v1.3 flag 16); the densities column is dropped (flag 6, a units mismatch). The window's start (the first record at which each slope column reaches %g of 1/(4GE_k); the light leaving f x 2gE_k then, at the launch radius %.1f):" % (ws["kappaR_frac"], ws["launch_radius"]), ""]
    for key, d in ws["columns"].items():
        if key == "kappa_R_densities":
            continue
        if not d.get("found"):
            L.append("- %s: never reaches the fraction" % key)
            continue
        la = "; ".join("f %g: u(chart) %s, light time %s r_s, u(arrival) %s" % (x["over_2gE"], _g(x["u_chart_rs"], "%.3f"), _g(x["t_out_rs"], "%.3f"), _g(x["u_at_launch_radius_rs"], "%.3f")) for x in d["launches"])
        L.append("- %s: record %d, T/r_s %.3f (ratio %.3f, root %s); %s" % (key, d["k"], d["T_over_rs"], d["ratio"], _g(d["R_t"], "%.3f"), la))
    L += ["", "| T/r_s | 2gE_k (a) | R_t json | R_t dense | R_t dense/2gE_k | dense secant x4GE_k | dense fit x4GE_k | site slope x4GE_k | kappa_T x4GE_k | min(N-abs v) | r_in dense |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    every = max(1, int(round(cfg["md_every_rs"] / max(table[1]["T_over_rs"] - table[0]["T_over_rs"], 1e-9)))) if len(table) > 1 else 1
    hits = set(d.get("k") for d in ws["columns"].values() if d.get("found"))
    for row in table:
        if row["k"] % every == 0 or row["k"] in hits or (row["k"] - 1) in hits or row.get("R_t_dense") is not None and row["k"] <= hits.__len__() * 0:
            rt = row.get("R_t_dense")
            L.append("| %.3f | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["T_over_rs"], row["two_gE"], _g(row.get("R_t_json"), "%.3f"), _g(rt, "%.3f"), _g((rt / row["two_gE"]) if rt is not None else None, "%.3f"),
                _g(row.get("kappa_R_dense_secant_x4GE"), "%.4f"), _g(row.get("kappa_R_dense_fit_x4GE"), "%.4f"), _g(row.get("kappa_R_site_slope_x4GE"), "%.4f"),
                _g(row.get("kappa_T_json_x4GE"), "%.4f"), _g(row.get("min_N_minus_absv_json"), "%.4f"), _g(row.get("r_in_dense"), "%.2f")))
    L.append("")
    for jo in jets_out:
        j_, A, res = jo["jet"], jo["A"], jo["res"]
        Rd = res["readouts"]
        alpha = res["dispersive_end"]["alpha_d"]
        L += ["## Jet %s: the runs and the per-clock table" % j_["name"], "", "Every run of every readout and fraction, down to 1 clock (the all-runs listing; a run of fewer than %d clocks is no plateau by the rule):" % cfg["band_min_clocks"], "",
              "| readout | frac | u start | u end | clocks | extent r_s | e-folds | kappa_H count | ratio mean | err mean | resolution | before the dispersive end |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in ("k", "E", "Ede", "U", "Uin", "rayk", "rayU"):
            if name not in Rd:
                continue
            for key in sorted(Rd[name]["bands"].keys(), key=float):
                for r_ in Rd[name]["bands"][key]["runs_all"]:
                    L.append("| %s | %s | %.3f | %.3f | %d | %.3f | %.4f | %.4f | %s | %s | %s | %s |" % (
                        name, key, r_["u_start"], r_["u_end"], r_["n_clocks"], r_["extent_rs"], r_["efolds"], r_["efolds_at_kappa_H"], _g(r_["ratio_mean"], "%.3f"), _g(r_["err_mean"], "%.3f"),
                        ("limited" if r_["resolution_limited"] else ("resolved" if r_["resolution_limited"] is not None else "-")), "yes" if r_["before_dispersive_end"] else "no"))
        L += ["", "| u/r_s | T_L/r_s | U/r_s | sigma_U/r_s | <abs k> T0 | k/alpha_d | r_k | err_k | r_E | r_Ede | r_U | r_Uin | r_rayk | <h0>/omega | c_in | <h0>/(c_in <abs k>) | f(k<0) | norm-1 | fb |", "|" + "---|" * 19]

        def rr(name, j):
            return _g(Rd[name]["ratio1"][j], "%.4f") if name in Rd else "-"
        for j in range(res["n_clocks"]):
            L.append("| %.3f | %.3f | %.4f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                A["u"][j], A["T_launch_over_rs"][j], A["U"][j] / rs, A["sigma_U"][j] / rs, _g(A["k_abs"][j], "%.4f"), _g(A["k_abs"][j] / alpha, "%.3f"), rr("k", j), _g(Rd["k"]["err"][j], "%.3f"),
                rr("E", j), rr("Ede", j), rr("U", j), rr("Uin", j), rr("rayk", j), _g(A["E0_over_omega"][j], "%.3f"), _g(A["c_in"][j], "%.4f"), _g(A["identity"][j], "%.3f"),
                _g(A["frac_k_neg"][j], "%.3f"), _g(A["norm_T0"][j] - 1.0, "%.1e"), _g(A["fb_defect"][j], "%.1e")))
        L.append("")
    L += ["## Checks", ""]
    for jo in jets_out:
        inf = jo["info"]
        L.append("- jet %s: rows flagged by the norm (> %g) or the forward check (> %g): %d; refused at the branch edge: %d; near the edge (the cuts removed more than a tenth of the Gaussian): %d of %d; store notes: %s" % (
            jo["jet"]["name"], cfg["norm_tol"], cfg["fb_tol"], inf["n_flagged"], inf["n_refused"], inf["n_near_edge"], inf["n_analysed"], "; ".join(inf["store_notes"]) or "none"))
    L += ["", "## What a failure looks like", "",
          "- a norm or forward-check defect above the tolerance: v2b's Chebyshev step did not resolve a record (the record is in the log line);",
          "- a probe near the branch edge (cut fraction above 0.1, the thermal jet at 16 r_s by construction): the probe is not monochromatic, read its exponents as smeared over its spectral width and its duration in u, never as the frequency's own;",
          "- a band run in the U readout while the k readout's ratio stays low or the in-energy falls there: a centroid swing inside a broad packet, not a blueshift (v1.3 section 6); the record value is the k readout's;",
          "- a segment shorter than 2 sigma_u: resolution-limited, the smeared exponent is the measurement, the comparator's segment beside it says what the probe could not resolve;",
          "- the dispersive end read while the packet's forward check holds but the folds of the position map have begun: the k readout stays meaningful (the spectral mean), the U readout is in v2b's stalled regime;",
          "- a comparator difference from the review's quoted constant: a note for icloud X1, not a defect of this readout.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------------- main
def w9_cfg(a):
    em = str(a.band_err_frac).strip()
    return {"jets_spec": a.jets, "band_fracs": [float(x) for x in str(a.band_fracs).split(",") if x.strip()], "band_err_frac": ("same" if em.lower() == "same" else float(em)),
            "band_min_clocks": int(a.band_min_clocks), "cutoff_inner": float(a.cutoff_inner), "csv": a.csv == "on", "code": CODE, "v2b_code": RM.CODE}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--nsites", type=int, default=8192, help="the light channel's chain (the record's chain plus the appended vacuum exterior)")
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--rs-nominal", dest="rs_nominal", type=float, default=None)
    ap.add_argument("--fields-key", dest="fields_key", default="applied", choices=["applied", "stored"])
    ap.add_argument("--fields-source", dest="fields_source", default="auto", choices=["auto", "own", "v6m"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cpu", help="W9 runs on the box CPU (cuda:0 admitted)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="recompute rows already in the jets' stores")
    RM.add_arguments(ap)
    ap.set_defaults(u_from_rs=W9_DEFAULTS["u_from_rs"], u_to_rs=W9_DEFAULTS["u_to_rs"], u_step_rs=W9_DEFAULTS["u_step_rs"])
    gw = ap.add_argument_group("the W9 segment readout (raymap_solved_flow_v3)")
    gw.add_argument("--jets", default=W9_DEFAULTS["jets"], help="the receiver jets q@R_L[:kappa][:cap=<r_s>][:step=<r_s>], q a number (omega r_s nominal) or TH (omega = T_H nominal)")
    gw.add_argument("--band-fracs", dest="band_fracs", default=W9_DEFAULTS["band_fracs"], help="the band fractions about kappa_H")
    gw.add_argument("--band-err-frac", dest="band_err_frac", default=W9_DEFAULTS["band_err_frac"], help="the 1-step/2-step difference admitted in the band, x kappa_H ('same': the band fraction)")
    gw.add_argument("--band-min-clocks", dest="band_min_clocks", type=int, default=W9_DEFAULTS["band_min_clocks"], help="the shortest run that counts as a segment")
    gw.add_argument("--cutoff-inner", dest="cutoff_inner", type=float, default=W9_DEFAULTS["cutoff_inner"], help="alpha_d = (inner a omega)^(1/3)")
    gw.add_argument("--csv", default="on", choices=["on", "off"], help="the per-clock table per jet as csv")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    jets = parse_jets(a.jets)
    cfg = RM.cfg_from_args(a)
    cfg["force"] = bool(a.force)
    cfg.update(w9_cfg(a))
    log("raymap_solved_flow_v3 (W9): v2b imported as %s from %s" % (RM.CODE, os.path.abspath(RM.__file__)))
    F, own, finfo = RM.make_fields(a, dev, kappa_check=jets[0]["kappa"])
    rs = float(a.rs_nominal or own.rs_nom or (2.0 * own.g * own.E[0]))
    cfg["rs_nominal"] = rs
    if own.args.get("R0_over_rs") is not None:
        cfg["ball_rs"] = float(own.args["R0_over_rs"])
    for jet in jets:
        jet["omega"] = jet["q"] / rs
        jet["T_d_nominal_rs"] = RM.T_d_over_rs(jet["q"], cfg, 1.0)
        jet["alpha_d"] = (cfg["cutoff_inner"] * jet["omega"]) ** (1.0 / 3.0)
    settings = dict(vars(a), cfg={k_: v_ for k_, v_ in cfg.items()})
    label = "the W9 segment readout on the solved flow (raymap_solved_flow_v3 over %s) on %s; jets %s; light channel on %d sites" % (RM.CODE, own.label or a.record, ", ".join(j_["spec"] for j_ in jets), F.nsites)
    log("raymap W9: %d records (T/r_s %.3f to %.3f, tau %g), g %g, r_s nominal %g, chain %d sites, device %s; fields %s (%s); jets %s; bands %s (err %s, min clocks %d); cutoff inner %g" % (
        len(F), F.T[0] / rs, F.T[-1] / rs, F.tau, F.g, rs, F.nsites, dev, finfo["source"], own.fields_key, [j_["spec"] for j_ in jets], cfg["band_fracs"], cfg["band_err_frac"], cfg["band_min_clocks"], cfg["cutoff_inner"]))
    jets_out = []
    for jet in jets:
        A, res, info, rows = run_jet(F, own, jet, cfg, rs, dev, a.out, a.tag, settings)
        comp = comparator_table(jet["q"], rs, cfg["band_fracs"], cfg["cutoff_inner"])
        rv = record_values(jet, res, comp, info)
        sent = jet_sentence(jet, res, comp, info)
        jets_out.append({"jet": jet, "A": A, "res": res, "info": info, "comp": comp, "record_values": rv, "sentence": sent})
        log("W9 " + sent)
    t1 = time.time()
    table = RM.kappa_R_table(own, cfg, rs)
    ws = RM.window_start(table, F, cfg, rs, jets_out[0]["info"]["R_L"])
    log("record table: %d records in %.1f s" % (len(table), time.time() - t1))
    J = {"code": CODE, "v2b": {"code": RM.CODE, "file": os.path.abspath(RM.__file__)}, "label": label, "history": own.label or a.record, "settings": settings, "fields": finfo,
         "kappa_H_definition": KAPPA_H_DEFINITION, "definitions": DEFINITIONS_MD, "jets": [],
         "kappa_R": {"table": table, "window_start": ws, "label": "v2b's kappa_R_table (each slope against the record's own 1/(4GE_k); two_gE the record's own charge) and window_start"},
         "notes": ["v2b's --omega-rs, --kappas and --launch-rs are not used: the jets define them"]}
    Z = {}
    for jo in jets_out:
        name = jo["jet"]["name"]
        for k_, v_ in jo["A"].items():
            Z["jet_%s_A_%s" % (name, k_)] = np.asarray(v_)
        analysis = strip_arrays(jo["res"], Z, "jet_%s_" % name)
        J["jets"].append({"jet": jo["jet"], "info": jo["info"], "comparator": jo["comp"], "record_values": jo["record_values"], "sentence": jo["sentence"], "analysis": analysis})
        if cfg["csv"]:
            pc = os.path.join(a.out, "raymap_w9_%s_%s.csv" % (a.tag, name))
            write_csv(pc, jo["A"], jo["res"], rs)
    for key in ("T_over_rs", "two_gE", "kappa_H", "R_t_json", "R_t_dense", "r_in_dense", "kappa_R_dense_secant", "kappa_R_dense_fit", "kappa_R_site_slope", "kappa_T_json",
                "kappa_R_dense_secant_x4GE", "kappa_R_dense_fit_x4GE", "kappa_R_site_slope_x4GE", "kappa_T_json_x4GE", "min_N_minus_absv_json", "min_c_dense"):
        Z["kappaR_" + key] = np.array([row.get(key) if row.get(key) is not None else np.nan for row in table], float)
    pj = os.path.join(a.out, "raymap_w9_%s.json" % a.tag)
    pz = os.path.join(a.out, "raymap_w9_%s.npz" % a.tag)
    pm = os.path.join(a.out, "raymap_w9_%s.md" % a.tag)
    with open(pj, "w") as f:
        json.dump(J, f, default=_jdefault, indent=1)
    np.savez(pz, **Z)
    write_md(pm, J, jets_out, cfg, rs, table, ws)
    log("done in %.0f s: %s, %s, %s%s" % (time.time() - t0, pj, pz, pm, (" and the csv per jet" if cfg["csv"] else "")))


if __name__ == "__main__":
    main()
