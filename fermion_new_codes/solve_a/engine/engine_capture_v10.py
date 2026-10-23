#!/usr/bin/env python
"""engine_capture_v10.py (builder analyst; engine spec 38.5 (3): "engine_capture_v10 (new; version 9 with an outer absorbing port on the
last 16 sites, the same construction as the inner port, the absorbed weight accumulated into R(T) as outward weight that left; nothing
else changed)"): the 300 r_s run of version 9 met the chain's reflecting outer end at 56 to 58 r_s (38.5), so the main state is
propagated here with the operator h_m - i Gamma_out(R), Gamma_out = --outer-gamma (default: --gamma, 0.2, the inner port's) times
clip((R - R_max)/--outer-ramp, 0, 1) (default ramp 8 sites, the inner port's), nonzero on the last 16 sites R > R_max = R_N - 16 only
(version 9's R_max, so the counted region [R_t + 2, R_max) and every reading region are unchanged); the propagator is the inner port's,
scipy's expm_multiply, one monitor-grid step (1 r_s) at a time (a Chebyshev series in h_m alone does not hold for the non-Hermitian
operator). The weight the outer port has absorbed by the reading time, I - ||psi(T)||^2 (the only non-Hermitian term of the main
operator), is outward weight that left: R(T) = the outward-branch weight on [R_t + 2, R_max) plus it, A = 1 - R(T)/I; both parts are
stored per channel (R_out_region, outer_absorbed) and the absorbed part is printed summed as sigma is, over pi b_c(omega, m)^2, at every
reading time (log, JSON, md). Nothing else changes: the packet, its positive-frequency projection (Chebyshev in h_m), the reading times,
the settle rule with version 9's in-transit gate, the channel rule, the inner absorbing-port comparator (version 6's operator, the
outer end reflecting there as in version 9; its readings after the chain end's return reaches the inner port carry that return) and
every other reading. --no-outer-port propagates the main state by version 9's Chebyshev steps and adds nothing to R(T): version 9's
run to the bit. New args: --no-outer-port --outer-gamma --outer-ramp. Version 9's docstring follows.
engine_capture_v9.py (builder analyst; engine spec 34.4 (2): "a stop counts as settled only if the inward-branch weight on [R_t + 2,
R_max) is below 10 percent of sigma at the stop; otherwise ... that probe on that background is rerun once on engine_capture_v8 (a new file:
version 7 with the settle rule gated on the in-transit share below 10 percent, nothing else changed)"; the name engine_capture_v8 was taken
by the one-point-branch guard, so this file is version 9 = version 8 with that gate): a reading time settles only when version 7's rule
holds and the inward-branch weight on [R_t + 2, R_max), summed over the channels as sigma is, is below --transit-share (0.10) of sigma
there; the share is printed at every reading time (log, JSON, md); the JSON records this file's name (version 8 recorded
"engine_capture_v7.py"). The propagation and every reading are version 8's, so a version 9 run reproduces version 7's or 8's readings to
the bit up to their stop. Nothing else changes. Version 8:
engine_capture_v8.py (builder analyst; the version 7 run at slice 283 stopped at 14:28:15 UTC: flow_chart_v4.branch_momenta returned
one momentum point inside the probe's frequency window for a channel near its threshold at R_0 (the normalization divided by zero) and
build_packet's np.gradient raised): version 7 with that channel excluded at launch by the rule versions 6 and 7 already apply to a channel
with no momenta in the window (no incident packet, no contribution to sigma): a branch with fewer than 2 momentum points or a non-finite
measure is recorded as an error entry with its reason; every excluded channel is listed per background and probe in the log, the JSON
("excluded_channels") and, when any exists, the md. Nothing else changes (the md title names this file). Version 7:
engine_capture_v7.py (builder analyst; engine spec 32.3 (4) and (5), 32.6 item 8; the lead's turn 40): version 6 with the reading
times every 2 r_s to 40 r_s after launch, or until settled, and the inward-branch weight inside R_t + 2 at each reading time.
(1) The reading times: T_k = 2 k r_s after launch, k = 1, ..., 20 (2 to 40 r_s, r_s = 2 g E of the slice; the same for every probe;
version 6's three times 2 r_s ln(1/(a omega)) + {2, 4, 8} r_s are replaced), taken on version 6's monitor grid (every r_s,
np.round(np.arange(r_s, T_max, r_s), 6)), so every channel is propagated by version 6's Chebyshev steps of 1 r_s. Settled: the readings
of a background and probe stop at the first reading time T >= 6 r_s at which the summed sigma/(pi b_c^2) changes by less than 10 percent
over the last 4 r_s, read as: the readings at T - 2 r_s and at T both differ from the reading at T - 4 r_s by less than 10 percent of it
(|s(T') - s(T - 4 r_s)| < 0.10 s(T - 4 r_s) for T' = T - 2 r_s and T); s is sigma over a constant of the background and probe, so the
rule is the same for sigma/(pi b_c(omega, m)^2) and for the massless sigma/(pi b_c^2). A probe without that by 40 r_s is "not settled
within 40 r_s", its last change over 4 r_s printed (percent and the factor s(T - 4 r_s)/s(T)). The rule's numbers are arguments
(--T-step-over-rs 2, --T-max-over-rs 40, --settle-window-over-rs 4, --settle-rel 0.10, the defaults) and are printed on each row.
(2) To read until settled, the channels of a background and probe run in lockstep from one reading time to the next, each channel's state
kept between them: each channel's arithmetic is version 6's (the same packet, positive-frequency projection, Chebyshev steps, block-wise
Gabor projections and monitor); only the order of the channels' steps changes. The channel list is version 6's rule (|kappa| from 1 until
A < 1e-3 at both signs and at every reading time so far for two consecutive |kappa| beyond omega b_c; the cap --kappa-cap), re-checked at
each reading time; a later reading that breaks it extends the list, the new channels run from launch to the current reading time. The
absorbing-port variant (version 6's operator) is propagated by scipy's expm_multiply from one reading time to the next (version 6
restarted it from launch at each of its three times).
(3) At each reading time, per channel (the JSON) and summed per probe as sigma is ((pi/p^2) sum_{kappa>0} kappa [W(+kappa) + W(-kappa)],
printed over pi b_c(omega, m)^2): the inward-branch weight inside R_t + 2 (version 6's block-wise Gabor projection on [1, R_t + 2) kept on
the momenta of negative local group velocity); beside it, of the same form, the plain norm inside R_t + 2, the inward-branch weight on
[R_t + 2, R_max) (incident flux still in transit) and the plain norm beyond R_max = R_N - 16 (outside version 6's counted region).
(4) The run's own length is printed (wall clock per reading step, per probe and in all); nothing is estimated.
(5) --workers N (default 1, the serial path) runs each step's channels in N forked processes; the channels the list needs are added in
batches of 4 N pairs and those beyond version 6's stop dropped, so the list is the serial one. On pod 1 (CAP7TEST tests 2 and 3: serial
against --workers 2 and 8) the channel lists, every unitary reading and monitor value, sigma and the settle decisions equal the serial path's
to the bit; the absorbing port's values differ by at most 9.437e-15 (equal in test 3): scipy's norm estimator inside expm_multiply draws
unseeded random vectors (_onenormest.py), so version 6's port was not bit-reproducible either. The BLAS setting is version 6's (left as the
launcher sets it): the background re-solve's E moves by 1.902e-11 relative between 1 and 8 BLAS threads on the last record (CAP7TEST
tests 1 and 1b; at 8, the lane runner's setting, E and the monitor equal version 6's to the bit), and the workers inherit it.
The massive comparator b_c(omega, m) and its self-check (e = 1.0001 and 1000) are version 6's, unchanged.
New args: --T-step-over-rs --T-max-over-rs --settle-window-over-rs --settle-rel --workers. Version 6's docstring follows.
engine_capture_v6.py (builder analyst): version 5 with the geometric capture radius b_c^2 = min R^2/(1 - v^2) (and its lapse form)
taken over each background's own exterior: version 5 used the solved slice's trapping radius for both backgrounds, so on the exact
profile (whose trapping radius 2 g E lies outside the slice's) the minimum ran over 1 - v^2 < 0 and the run stopped at a negative b_c^2
(pod 1 lane 3, 08:11 UTC, after the solved slice's rows); here each background's trapping zero is found first and the minimum is over
the radii beyond it where 1 - v^2 > 0 (N^2 - v^2 > 0 for the lapse form). Nothing else changes; version 5's docstring follows.
engine_capture_v5.py (builder analyst; version 5 = version 4 with the slice re-solved by the production's own field module
(engine_fields_v14.solve_constraints_ivp, its trapping radius and E at the collar) in place of engine_fields_v8's Picard solve, which did not
converge on the production's last slice (200 iterations, dN 2.5e-11, dW 1.15e-10; version 4's run on pod 1 lane 3). Nothing else changes.
Version 4's docstring: engine spec 28.3, S13 of REVIEWA_REVIEW_web40e: "C-capture: the queued omega r_s in {2, 4, 8} lie below the
gap m r_s = 12.8 and are dropped; the dry run uses omega/m = 1.1, 1.25, 1.5, 2 (a omega 0.22 to 0.40, cos 2 a p 0.98 to 0.77; each labelled
by its asymptotic speed and cos 2 a p); f_nr the unitary return deficit including the core-mediated return, R_object = r + t' R_c (1 - r'
R_c)^-1 t, no separate Pauli factor, at 13.3's reading times; the massive comparator pi b_c(omega, m)^2 times f_nr, b_c from the exterior's
effective potential V(r) = (1 - r_s/r)(1 + l^2/r^2), l_c solving e^2 = max V at e = omega/m, b_c = l_c/sqrt(e^2 - 1) (a bisection; the code's
controls: b_c^2 to 27 r_s^2/4 as e grows ... and to 4 r_s^2/(e^2 - 1) as e tends to 1)"). Version 4 = version 3 with: the massless probes
dropped by default (--omega-rs empty), the massive probes given as --omega-over-m (default 1.1, 1.25, 1.5, 2; a omega = (omega/m) m a); each
probe labelled by its lattice asymptotic momentum a p_lat = arcsin(sqrt((a omega)^2 - (m a)^2)), group speed sin(a p_lat) cos(a p_lat)/(a
omega) and cos 2 a p_lat; f_nr per channel and time is version 3's A = 1 - R(T)/I (R(T) the whole outward-branch weight outside R_t + 2 at
the reading time, so every return the core mediates by then is in it: the composition R_object = r + t' R_c (1 - r' R_c)^-1 t is what the
propagation does; no Pauli factor anywhere); the massive comparator b_c(omega, m) by bisection on l with the Schwarzschild peak radius
r = (l^2 - sqrt(l^4 - 3 l^2 r_s^2))/r_s (r_s = 2 g E of the slice), sigma/(pi b_c(omega, m)^2) beside version 3's massless sigma/(pi b_c^2),
and the comparator's two limits printed beside it as controls (27 r_s^2/4 and 4 r_s^2/(e^2 - 1)) with a self-check at e = 1.0001 and 1000.
Version 3's docstring follows.
engine_capture_v3.py (builder analyst; engine spec 19.2: the production's consumers propagate on the applied (kernel-averaged) site fields:
version 3 = version 2 with --fields-key applied | resolved (default applied): applied takes the slice's N_applied, v_applied (the fields of
the h_m that propagated the run's state) as the background's site fields, the re-solve from the stored shell totals kept for E and the
trapping radius (the dense functions); resolved is version 2's background (the re-solved site fields). The record names the key and prints
the largest difference between the two site-field sets. Nothing else changes. Version 2's docstring follows.
engine_capture_v2.py (builder analyst; C-capture as restated in SPEC_final4b_engine.md sec. 13.3: the non-return fraction; version 2 of the
lead's engine_capture_v1.py, whose first-passage fluence is kept only as a crossing monitor).

On the frozen late background (a collapse slice's dense functions re-solved from its stored shell totals; and, as the control, the same interior
with the exterior replaced by N = 1, v = -sqrt(2 g E/R) through a tanh junction at --Rx r_s of width 4), per channel kappa (both signs: the
chain's two valleys) and Killing frequency omega: an incident packet of the exterior's ingoing branch (flow_chart_v4.branch_momenta with the
channel's transverse mass kappa/R_0, the frequency width --sigma-omega-rel omega) built at R_0 = --R0-over-rs r_s (build_packet with the local
spinor) and projected on positive Killing frequency (the erf step of h_m at width --ws, Chebyshev); the incident weight I; the state propagated
on the frozen h_m (Chebyshev); at the report times T = 2 r_s ln(1/(a omega)) + {2, 4, 8} r_s the returned weight R(T): the norm at R > R_t + 2
on the outward branch (a block-wise Gabor projection: blocks of --block sites, each kept on the momenta of positive local group velocity of the
channel symbol at the block's N, v, kappa/R), beside the plain outside norm; A(omega, kappa, T) = 1 - R(T)/I; the conversion monitor: the
weight inside R_t on positive local group velocity (the escape branch) against time; the cross section sigma(omega, T) = (pi/p^2) sum_{kappa>0}
kappa [A_{+kappa} + A_{-kappa}], the geometric check b_c^2 = min_{R > R_t} R^2/(1 - v^2) (and with the lapse, R^2 N^2/(N^2 - v^2), labelled),
pi b_c^2 against 27 pi (g E)^2, sigma/(pi b_c^2); the absorbing-port variant (--absorber: -i Gamma(R) inside R_t - 2, a smooth ramp to --gamma,
propagated by scipy's expm_multiply) as the ideal-absorber comparator. The probes: the massless fermion at omega r_s in --omega-rs and the
massive at a omega in --fermion-omega-a with m a = --m. Channels from |kappa| = 1 until A < 1e-3 at both signs and at every T for two
consecutive |kappa| beyond omega b_c. CPU (sparse). Args: --fields --records --slice --g --Rx --omega-rs --fermion-omega-a --m --R0-over-rs
--sigma-omega-rel --ws --block --absorber --gamma --kappa-cap --control --out-json --md --tag --gpu (accepted, unused)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy.sparse as sp
from scipy.fft import dct
from scipy.sparse.linalg import expm_multiply
from scipy.special import erfc, jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "flow", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import engine_vacflow_v6 as VF  # noqa: E402  (chain_sparse: m sz + (kappa/R) sy on site, -(i/2) sx bond, the two-step Q)
import flow_chart_v4 as FC  # noqa: E402


class SB:
    def __init__(self, H):
        self.H = H.tocsr()
        self.bound = float(abs(self.H).sum(axis=1).max())

    def apply(self, X):
        return self.H @ X


def cheb_expm(B, X, tau, tol=1e-15):
    if tau == 0:
        return X
    r = B.bound; a = r * tau
    K = int(abs(a) + 12)
    while abs(jv(K, abs(a))) > tol or K < abs(a) + 4:
        K += 4
    T0, T1 = X, B.apply(X) / r
    Y = jv(0, a) * T0 + 2 * (-1j) * jv(1, a) * T1
    for k in range(2, K + 1):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + 2 * ((-1j) ** k) * jv(k, a) * T2
        T0, T1 = T1, T2
    return Y


def cheb_neg_step(B, X, ws):
    """f(h) X with f(E) = erfc(E/(sqrt2 ws))/2 (the negative-energy step)."""
    r = B.bound
    M = 1
    while M < 8 * r / ws:
        M *= 2
    th = (np.arange(M) + 0.5) * math.pi / M
    c = dct(0.5 * erfc(r * np.cos(th) / (math.sqrt(2.0) * ws)), type=2) / M
    c[0] /= 2.0
    K = int(np.nonzero(np.abs(c) > 1e-13)[0][-1]) + 1
    T0, T1 = X, B.apply(X) / r
    Y = c[0] * T0 + c[1] * T1
    for k in range(2, K):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
    return Y


def clocked_sparse(hs, Qs, N, v):
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ hs + hs @ Nd) + 0.5 * (Vd @ Qs + Qs @ Vd)).tocsr()


def outward_weight(psi, R, N, v, m, kappa, lo, hi, block):
    """the norm on [lo, hi) kept on the momenta of positive local group velocity, block by block (a Gabor projection with a cosine-squared
    window per block, the blocks overlapping by half, the windows summing to one)."""
    a = psi.reshape(-1, 2)
    tot = 0.0
    starts = np.arange(lo, hi, block // 2)
    for s0 in starts:
        sel = np.nonzero((R >= s0) & (R < s0 + block) & (R >= lo) & (R < hi))[0]
        if len(sel) < 4:
            continue
        xw = (R[sel] - s0) / block
        win = np.sin(math.pi * xw) ** 2                                  # half-overlapping sin^2 windows sum to one
        seg = a[sel] * np.sqrt(win)[:, None]
        n = 4 * len(sel)
        k = 2 * np.pi * np.fft.fftfreq(n)
        ak = np.fft.fft(seg, n=n, axis=0)
        mid = int(sel[len(sel) // 2])
        vg = N[mid] * np.sin(k) * np.cos(k) / np.sqrt(m * m + (kappa / R[mid]) ** 2 + np.sin(k) ** 2) + v[mid] * np.cos(2 * k)
        keep = vg > 0
        tot += float(np.sum(np.abs(ak[keep]) ** 2)) / n
    return tot


def inward_weight(psi, R, N, v, m, kappa, lo, hi, block):
    """version 6's outward_weight kept on the momenta of negative local group velocity (the inward branch); nothing else differs."""
    a = psi.reshape(-1, 2)
    tot = 0.0
    starts = np.arange(lo, hi, block // 2)
    for s0 in starts:
        sel = np.nonzero((R >= s0) & (R < s0 + block) & (R >= lo) & (R < hi))[0]
        if len(sel) < 4:
            continue
        xw = (R[sel] - s0) / block
        win = np.sin(math.pi * xw) ** 2
        seg = a[sel] * np.sqrt(win)[:, None]
        n = 4 * len(sel)
        k = 2 * np.pi * np.fft.fftfreq(n)
        ak = np.fft.fft(seg, n=n, axis=0)
        mid = int(sel[len(sel) // 2])
        vg = N[mid] * np.sin(k) * np.cos(k) / np.sqrt(m * m + (kappa / R[mid]) ** 2 + np.sin(k) ** 2) + v[mid] * np.cos(2 * k)
        keep = vg < 0
        tot += float(np.sum(np.abs(ak[keep]) ** 2)) / n
    return tot


# ---- the channels in lockstep: version 6's run_channel cut at the reading times (the state kept between them) ----
BG = {}                                   # the frozen background and probe of the current (background, probe); set before any channel runs
_CACHE = {"id": None, "mats": {}}         # per process: the channel matrices of the current (background, probe)


def _matrices(kappa):
    if _CACHE["id"] != BG["id"]:
        _CACHE["id"] = BG["id"]; _CACHE["mats"] = {}
    hit = _CACHE["mats"].get(kappa)
    if hit is None:
        hs, Qs = VF.chain_sparse(BG["R"], BG["m"], float(kappa))
        hm = clocked_sparse(hs, Qs, BG["N"], BG["v"])
        Hc = None
        if BG["gamma"] is not None:
            gam = BG["gamma"] * np.clip((BG["Rt"] - 2.0 - BG["R"]) / 8.0, 0.0, 1.0)
            Hc = (hm - 1j * sp.diags(np.repeat(gam, 2))).tocsc()
        Ho = None
        if BG["outer"] is not None:                                             # version 10: the outer port on R > R_max, the inner port's form
            gam_o = BG["outer"]["gamma"] * np.clip((BG["R"] - BG["Rmax"]) / BG["outer"]["ramp"], 0.0, 1.0)
            Ho = (hm - 1j * sp.diags(np.repeat(gam_o, 2))).tocsc()
        hit = (SB(hm), Hc, Ho)
        _CACHE["mats"][kappa] = hit
    return hit


def chan_init(kappa):
    """version 6's run_channel up to the incident weight: the packet on the exterior's ingoing branch at R_0, projected on positive
    Killing frequency; the state at launch."""
    R, N, v, m, omega, R0 = BG["R"], BG["N"], BG["v"], BG["m"], BG["omega"], BG["R0"]
    B, Hc, Ho = _matrices(kappa)
    mc = math.sqrt(m * m + (kappa / R0) ** 2)
    vext = float(np.interp(R0, R, v))
    try:
        k, om, a2 = FC.branch_momenta(omega, BG["sig_om"], mc, vext, outgoing=False)
    except ValueError as err:
        return {"kappa": kappa, "error": str(err)}
    if len(k) < 2 or not np.all(np.isfinite(a2)):                            # version 8's guard
        return {"kappa": kappa, "error": "the branch holds %d momentum point(s) inside the frequency window, its measure %s (at threshold: excluded as a channel with no momenta in the window; version 8)" % (len(k), "finite" if np.all(np.isfinite(a2)) else "not finite")}
    psi0 = FC.build_packet(len(R), int(round(R0 - R[0])), k, a2, mc, m=m, a=kappa / R0)
    psi0 = psi0 - cheb_neg_step(B, psi0, BG["ws"])                          # positive Killing frequency
    I = float(np.vdot(psi0, psi0).real)
    st = {"kappa": kappa, "I": I, "bound": B.bound, "A": {}, "R_out": {}, "outside_plain": {}, "inward_inside": {}, "plain_inside": {},
          "inward_transit": {}, "beyond_Rmax": {}, "monitor": [], "_psi": psi0, "_j": 0}
    if Hc is not None:
        st["absorbed"] = {}; st["_pT"] = psi0; st["_jT"] = 0
    if Ho is not None:
        st["outer_absorbed"] = {}; st["R_out_region"] = {}
    return st


def chan_advance(st, j_to):
    """the state from its grid point to BG["tgrid"][j_to] in version 6's steps (Chebyshev, one grid step at a time), the monitor at every
    grid point and the readings at the reading points; the absorbing port from the last reading point to each new one."""
    if "error" in st:
        return st
    R, N, v, m, block = BG["R"], BG["N"], BG["v"], BG["m"], BG["block"]
    Rt, Rmax, tg, reads = BG["Rt"], BG["Rmax"], BG["tgrid"], BG["read_js"]
    kappa, I = st["kappa"], st["I"]
    B, Hc, Ho = _matrices(kappa)
    psi, j = st["_psi"], st["_j"]
    while j < j_to:
        if Ho is None:
            psi = cheb_expm(B, psi, tg[j + 1] - tg[j])                         # version 9's step
        else:
            psi = expm_multiply(-1j * Ho * (tg[j + 1] - tg[j]), psi)          # version 10: h_m - i Gamma_out, the inner port's propagator
        j += 1
        t = tg[j]
        inside_out = outward_weight(psi, R, N, v, m, kappa, 1.0, Rt, block)
        st["monitor"].append({"t": t, "inside_outward": inside_out / I, "inside_total": float(np.sum(np.abs(psi.reshape(-1, 2)[R < Rt]) ** 2)) / I})
        if j in reads:
            Rw = outward_weight(psi, R, N, v, m, kappa, Rt + 2.0, Rmax, block)
            plain = float(np.sum(np.abs(psi.reshape(-1, 2)[(R > Rt + 2.0)]) ** 2))
            key = "%.4f" % t
            if Ho is None:
                st["R_out"][key] = Rw / I; st["outside_plain"][key] = plain / I; st["A"][key] = 1.0 - Rw / I
            else:                                                               # version 10: the absorbed weight is outward weight that left
                gone = 1.0 - float(np.vdot(psi, psi).real) / I
                st["outer_absorbed"][key] = gone; st["R_out_region"][key] = Rw / I
                st["R_out"][key] = Rw / I + gone; st["outside_plain"][key] = plain / I; st["A"][key] = 1.0 - (Rw / I + gone)
            st["inward_inside"][key] = inward_weight(psi, R, N, v, m, kappa, 1.0, Rt + 2.0, block) / I
            st["plain_inside"][key] = float(np.sum(np.abs(psi.reshape(-1, 2)[R < Rt + 2.0]) ** 2)) / I
            st["inward_transit"][key] = inward_weight(psi, R, N, v, m, kappa, Rt + 2.0, Rmax, block) / I
            st["beyond_Rmax"][key] = float(np.sum(np.abs(psi.reshape(-1, 2)[R >= Rmax]) ** 2)) / I
            if Hc is not None:
                pT = expm_multiply(-1j * Hc * (t - tg[st["_jT"]]), st["_pT"])
                st["_pT"] = pT; st["_jT"] = j
                st["absorbed"][key] = 1.0 - float(np.vdot(pT, pT).real) / I
    st["_psi"], st["_j"] = psi, j
    return st


def _task(args):
    kind, payload, j_to = args
    st = chan_init(payload) if kind == "init" else payload
    return chan_advance(st, j_to)


def run_tasks(pool, tasks):
    if pool is None:
        return [_task(t) for t in tasks]
    return pool.map(_task, tasks, chunksize=1)


def rule_stop(chans, om, bc2):
    """version 6's channel rule on the readings so far: the number of channels kept (|kappa| = 1 to the second of the first two consecutive
    |kappa| beyond omega b_c with A < 1e-3 at both signs and at every reading), or None when the list holds no such pair."""
    quiet = 0
    for i, c in enumerate(chans):
        pair = (c["plus"], c["minus"])
        Amax = max([x for r_ in pair if "A" in r_ for x in r_["A"].values()] + [0.0])
        quiet = quiet + 1 if (Amax < 1e-3 and c["kappa"] > om * math.sqrt(bc2)) else 0
        if quiet >= 2:
            return i + 1
    return None


def summed(chans, field, key, p):
    """(pi/p^2) sum_{kappa>0} kappa [X(+kappa) + X(-kappa)] at one reading (field "A": version 6's sigma)."""
    return math.pi / p ** 2 * sum(c["kappa"] * (c["plus"].get(field, {}).get(key, 0.0) + c["minus"].get(field, {}).get(key, 0.0)) for c in chans)


def public(st):
    return {k: x for k, x in st.items() if not k.startswith("_")}


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def write_md(rec, md, a, done):
    rs = rec["r_s"]
    chk = rec["comparator_selfcheck"]
    f4 = lambda x, fm="%.4f": (fm % x) if x is not None else "-"  # noqa: E731
    pc = lambda x: ("%+.1f" % (100.0 * x)) if x is not None else "-"  # noqa: E731
    op = rec["outer_port"]
    L = ["## C-capture (engine_capture_v10.py, engine 13.3, 28.3, 32.3 (4), 34.4 (2) and 38.5 (3)): slice %d (T %.1f), E %.6f, r_s %.4f, R_t %.4f" % (rec["slice"], rec["T"], rec["E"], rs, rec["R_t"]), "",
         ("- the outer absorbing port (version 10, engine 38.5 (3)): -i Gamma_out(R) on the last 16 sites R > R_max = %g, Gamma_out = %g clip((R - R_max)/%g, 0, 1), the inner port's construction; the main state propagated with it (scipy's expm_multiply, one r_s per step); the weight it has absorbed by T is outward weight that left and is part of R(T), A = 1 - R(T)/I; its column below is that weight summed as sigma is." % (op["R_max"], op["gamma"], op["ramp"]))
         if op["on"] else "- no outer port (--no-outer-port): the main state propagated by version 9's Chebyshev steps, R(T) version 9's.",
         "- the massive comparator's self-check (version 6's, unchanged): b_c^2 over 4 r_s^2/(e^2 - 1) at e 1.0001: %.6f; b_c^2 over 27 r_s^2/4 at e 1000: %.6f." % (chk["bc2_over_4rs2_over_e2m1"], chk["bc2_over_27rs2_over_4"]),
         "- the reading times: every %g r_s after launch to %g r_s, the same for every probe; settled: the first reading time T >= %g r_s at which the readings at T - %g r_s and at T both differ from the reading at T - %g r_s by less than %g percent of it (s = sigma, f_nr summed; sigma/(pi b_c^2) is sigma over a constant) and the inward-branch weight on [R_t + 2, R_max) is below %g percent of sigma there (version 9, engine 34.4 (2)); the readings stop there."
         % (a.T_step_over_rs, a.T_max_over_rs, (rec["settle_window_steps"] + 1) * a.T_step_over_rs, a.T_step_over_rs, a.settle_window_over_rs, 100.0 * a.settle_rel, 100.0 * a.transit_share),
         ("- the run's length: %.0f s wall clock (%s to %s UTC), --workers %d." % (rec["run_length_s"], rec["start_utc"], rec["end_utc"], a.workers)) if done else
         ("- the run is in progress (started %s UTC, --workers %d): the rows below are the probes done by %.0f s." % (rec["start_utc"], a.workers, rec["elapsed_s"])), "",
         "### Per background and probe", "",
         "| background | probe | a omega | e = omega/m | a p_lat | group speed | cos 2 a p | pi b_c(omega, m)^2 | its limits 27 r_s^2/4 and 4 r_s^2/(e^2 - 1) (times pi) | pi b_c^2, massless geometry | channels | reading times (T/r_s) | settled (the rule above) | sigma/(pi b_c(omega, m)^2) there (settled) or at the last reading | its change against the reading 4 r_s earlier (percent; the factor s(T - 4 r_s)/s(T)) | length (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for res in rec["results"]:
        mm = res.get("massive") or {}
        st_ = res["settle"]
        L.append("| %s | %s | %.5f | %s | %s | %s | %s | %s | %s, %s | %.4e | %d | %g to %g (%d) | %s | %s | %s; %s | %.0f |" % (
            res["background"], res["probe"], res["omega"], f4(mm.get("e")), f4(mm.get("a_p_lat"), "%.5f"), f4(mm.get("group_speed"), "%.5f"),
            f4(mm.get("cos_2ap"), "%.5f"), f4(mm.get("pi_bc2_massive"), "%.4e"), f4(math.pi * mm["limit_27rs2_over_4"] if mm else None, "%.4e"),
            f4(math.pi * mm["limit_4rs2_over_e2m1"] if mm else None, "%.4e"), res["pi_bc2"], len(res["channels"]), res["T_list"][0] / rs, res["T_list"][-1] / rs,
            len(res["T_list"]), ("yes, at T %g r_s" % st_["T_over_rs"]) if st_["settled"] else ("not settled within %g r_s" % (res["T_list"][-1] / rs)),
            f4(st_["ratio_at"]), pc(st_["change_at"]), f4(st_["factor_at"], "%.3f"), res["length_s"]))
    L += ["", "### At each reading time (the weights summed over the channels as sigma is, (pi/p^2) sum kappa [W(+kappa) + W(-kappa)], over pi b_c(omega, m)^2)", "",
          "| background | probe | T (r_s) | sigma (f_nr summed) | sigma/(pi b_c(omega, m)^2) | sigma/(pi b_c^2), massless geometry | the readings at T - %g r_s and T against T - %g r_s (percent; the rule: both under %g) | settled here | inward-branch weight inside R_t + 2 | plain norm inside R_t + 2 | inward-branch weight on [R_t + 2, R_max), in transit | plain norm beyond R_max | outer port: weight absorbed by T (outward, left; part of R(T)) | absorbing port, over pi b_c(omega, m)^2 | channels at this step | step (s) |"
          % (a.T_step_over_rs, a.settle_window_over_rs, 100.0 * a.settle_rel),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for res in rec["results"]:
        nrm = res["norm_ratio"]
        for i, T in enumerate(res["T_list"]):
            key = "%.4f" % T
            sp_ = res["steps"][i]
            devs = sp_["devs"]
            L.append("| %s | %s | %.0f | %.4e | %.4f | %.4f | %s | %s | %.4f | %.4f | %.4f | %.4f | %s | %s | %d | %.0f |" % (
                res["background"], res["probe"], T / rs, res["sigma"][key], res["sigma"][key] / nrm, res["sigma_over_pi_bc2"][key],
                (", ".join(pc(d) for d in devs)) if devs else "-", "yes" if sp_["settled"] else "no",
                res["inward_inside_summed"][key] / nrm, res["plain_inside_summed"][key] / nrm, res["inward_transit_summed"][key] / nrm,
                res["beyond_Rmax_summed"][key] / nrm, f4(res["outer_absorbed_summed"][key] / nrm if res["outer_absorbed_summed"] else None),
                f4(res["sigma_absorber"][key] / nrm if res["sigma_absorber"] else None),
                sp_["channels"], sp_["step_s"]))
    grew = [(res["background"], res["probe"], sp_["T_over_rs"], sp_["channels"]) for res in rec["results"] for sp_ in res["steps"] if sp_["channels"] != len(res["channels"])]
    L += ["", "- every number above is printed by the script; f_nr = A = 1 - R(T)/I with R(T) the whole outward-branch weight on [R_t + 2, R_max) at the reading time (block-wise Gabor projection; every core-mediated return by then included; no Pauli factor)%s;" % (
              " plus the weight the outer port has absorbed by then (version 10)" if op["on"] else ""),
          "  the inward-branch weights are version 6's projection kept on negative local group velocity; the per-channel values, the absorbing port and the conversion monitor (every r_s) are in the JSON.",
          "- the rule's numbers on each row are those of its own step; the sigma columns are summed over the final channel list%s." % (
              "" if not grew else (" (the list grew after these steps: " + "; ".join("%s | %s at T %g r_s: %d" % g for g in grew) + ")"))]
    exc = [(res["background"], res["probe"], res.get("excluded_channels") or []) for res in rec["results"]]
    if any(x[2] for x in exc):
        L.append("- channels excluded at launch (version 8's guard; no incident packet, no contribution to sigma): " + "; ".join(
            "%s | %s: %s" % (b_, p_, ", ".join("kappa %+d (%s)" % (kk, why) for kk, why in e_)) for b_, p_, e_ in exc if e_))
    open(md, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--Rx", type=float, default=0.5)
    ap.add_argument("--omega-rs", type=float, nargs="*", default=[])
    ap.add_argument("--omega-over-m", type=float, nargs="*", default=[1.1, 1.25, 1.5, 2.0])
    ap.add_argument("--fermion-omega-a", type=float, nargs="*", default=[0.25, 0.3, 0.4])
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--R0-over-rs", type=float, default=8.0)
    ap.add_argument("--sigma-omega-rel", type=float, default=0.1)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--absorber", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--kappa-cap", type=int, default=200)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--collar-policy", default="flag")
    ap.add_argument("--fields-key", default="applied", choices=["applied", "resolved"], help="engine spec 19.2: the applied (averaged) site fields")
    ap.add_argument("--T-step-over-rs", type=float, default=2.0, help="the reading times every this many r_s after launch (a whole number)")
    ap.add_argument("--T-max-over-rs", type=float, default=40.0, help="the last reading time (r_s after launch)")
    ap.add_argument("--settle-window-over-rs", type=float, default=4.0, help="settled: the change over this many r_s (a multiple of the step)")
    ap.add_argument("--settle-rel", type=float, default=0.10, help="settled: every reading in the window within this fraction of its first")
    ap.add_argument("--transit-share", type=float, default=0.10, help="version 9: settled also needs the in-transit inward weight below this share of sigma")
    ap.add_argument("--workers", type=int, default=1, help="each step's channels in this many forked single-threaded processes (1: serial)")
    ap.add_argument("--no-outer-port", action="store_true", help="version 10: no outer port (version 9's propagation and R(T), to the bit)")
    ap.add_argument("--outer-gamma", type=float, default=None, help="version 10: the outer port's strength (default: --gamma, the inner port's)")
    ap.add_argument("--outer-ramp", type=float, default=8.0, help="version 10: the outer port's ramp in sites (default 8, the inner port's)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--tag", default="capture")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    t_run0 = time.time()
    step_j = int(round(a.T_step_over_rs)); wsteps = int(round(a.settle_window_over_rs / a.T_step_over_rs))
    if abs(a.T_step_over_rs - step_j) > 1e-9 or step_j < 1 or abs(a.settle_window_over_rs - wsteps * a.T_step_over_rs) > 1e-9 or wsteps < 1:
        raise SystemExit("the reading step must be a whole number of r_s (the monitor grid) and the settle window a multiple of it")
    out_json = a.out_json or "CAP/%s.json" % a.tag
    md = a.md or "CAP/%s.md" % a.tag
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(md) or ".", exist_ok=True)
    import engine_fields_v14 as F8
    Z = np.load(a.fields, allow_pickle=True)
    idx = a.slice if a.slice >= 0 else len(Z["T"]) + a.slice
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints_ivp(np.asarray(Z["eps_tot"][idx], float), np.asarray(Z["J_tot"][idx], float), a.g, R, ell,
                                                collar_policy=a.collar_policy)
    if info.get("failed"):
        raise SystemExit("the re-solve failed: %s" % info["failed"])
    E = float(info["E_collar"]); rs = 2 * a.g * E
    tz = F8.trapping_radius(Ns, vs, R, info)
    Rt = tz["zeros"][-1] if tz["zeros"] else None
    if Rt is None:
        raise SystemExit("no trapping surface on this slice")
    key_diff = None
    if a.fields_key == "applied":
        Na = np.asarray(Z["N_applied"][idx], float); va = np.asarray(Z["v_applied"][idx], float)
        key_diff = {"max_dN": float(np.max(np.abs(Na - Ns))), "max_dv": float(np.max(np.abs(va - vs)))}
        print("the applied site fields against the re-solved ones: max |dN| %.2e, max |dv| %.2e" % (key_diff["max_dN"], key_diff["max_dv"]), flush=True)
        Ns, vs = Na, va
    bgs = [("solved slice (%s site fields)" % a.fields_key, Ns, vs)]
    if a.control:
        s = 0.5 * (1 + np.tanh((R - a.Rx * rs) / 4.0))
        Nc = (1 - s) * Ns + s * 1.0
        vc = (1 - s) * vs + s * (-np.sqrt(rs / R))
        bgs.append(("control: the exact exterior from %g r_s, the same interior" % a.Rx, Nc, vc))
    probes = [("massless", 0.0, w / rs) for w in a.omega_rs] + [("fermion m a %g, omega/m %g" % (a.m, x), a.m, x * a.m) for x in a.omega_over_m]

    def bc_massive(e, rs_):
        """the critical impact parameter of a massive geodesic at energy per mass e > 1 on the Schwarzschild exterior of radius rs_."""
        def vmax(l):
            disc = l ** 4 - 3.0 * l * l * rs_ * rs_
            if disc <= 0:
                return -1.0
            r = (l * l - math.sqrt(disc)) / rs_
            return (1.0 - rs_ / r) * (1.0 + l * l / (r * r))
        lo, hi = math.sqrt(3.0) * rs_ * (1 + 1e-12), 2.0 * rs_
        while vmax(hi) < e * e:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if vmax(mid) < e * e:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi) / math.sqrt(e * e - 1.0)
    chk = {"e": [1.0001, 1000.0], "bc2_over_4rs2_over_e2m1": bc_massive(1.0001, 1.0) ** 2 / (4.0 / (1.0001 ** 2 - 1.0)), "bc2_over_27rs2_over_4": bc_massive(1000.0, 1.0) ** 2 / 6.75}
    print("massive comparator self-check: b_c^2 / (4 r_s^2/(e^2 - 1)) at e 1.0001 = %.6f; b_c^2 / (27 r_s^2/4) at e 1000 = %.6f" % (chk["bc2_over_4rs2_over_e2m1"], chk["bc2_over_27rs2_over_4"]), flush=True)
    # version 6's monitor grid (every r_s) to the last reading time; the reading times every --T-step-over-rs r_s on it
    tgrid = [0.0] + [float(x) for x in np.round(np.arange(rs, a.T_max_over_rs * rs + 1e-9, rs), 6)]
    read_js = list(range(step_j, len(tgrid), step_j))
    if not read_js:
        raise SystemExit("no reading time: --T-max-over-rs is below --T-step-over-rs")
    rule = {"T_step_over_rs": a.T_step_over_rs, "T_max_over_rs": a.T_max_over_rs, "window_over_rs": a.settle_window_over_rs, "rel": a.settle_rel,
            "transit_share": a.transit_share,
            "text": "settled at the first reading time T at which the readings at T - %g r_s and T both differ from the reading at T - %g r_s by less than %g percent of it and the inward-branch weight on [R_t + 2, R_max) is below %g percent of sigma at T (version 9, engine 34.4 (2))"
                    % (a.T_step_over_rs, a.settle_window_over_rs, 100.0 * a.settle_rel, 100.0 * a.transit_share)}
    outer = None if a.no_outer_port else {"gamma": a.gamma if a.outer_gamma is None else a.outer_gamma, "ramp": a.outer_ramp}
    outer_rec = {"on": outer is not None, "R_max": float(R[-1]) - 16.0, "R_N": float(R[-1]), "gamma": outer["gamma"] if outer else None,
                 "ramp": outer["ramp"] if outer else None, "text": ("-i Gamma_out(R), Gamma_out = %g clip((R - R_max)/%g, 0, 1) on R > R_max = %g (the last 16 sites); "
                                                                     "expm_multiply; the absorbed weight is part of R(T)" % (outer["gamma"], outer["ramp"], float(R[-1]) - 16.0)) if outer else "none (version 9's run)"}
    print("the outer port (version 10): %s" % outer_rec["text"], flush=True)
    rec = {"script": os.path.basename(__file__), "comparator_selfcheck": chk, "args": vars(a), "slice": idx, "T": float(Z["T"][idx]), "E": E, "r_s": rs, "R_t": Rt, "outer_port": outer_rec,
           "fields_key": a.fields_key, "applied_minus_resolved": key_diff, "reading_times": [tgrid[j] for j in read_js], "settle_rule": rule,
           "settle_window_steps": wsteps, "start_utc": utc(t_run0), "results": []}
    print("the reading times: every %g r_s after launch to %g r_s (%d); %s; --workers %d; start %s UTC" % (a.T_step_over_rs, a.T_max_over_rs, len(read_js), rule["text"], a.workers, rec["start_utc"]), flush=True)
    ctx = mp.get_context("fork") if a.workers > 1 else None
    for ib, (bl, N, v) in enumerate(bgs):
        Rt_b = Rt
        if bl.startswith("control"):
            zz = np.nonzero((N[:-1] + v[:-1]) * (N[1:] + v[1:]) <= 0)[0]
            Rt_b = float(R[zz[-1]]) if len(zz) else Rt
        ext = (R > Rt_b) & (1 - v ** 2 > 0)
        extN = (R > Rt_b) & (N ** 2 - v ** 2 > 0)
        bc2 = float(np.min(R[ext] ** 2 / (1 - v[ext] ** 2)))
        bc2N = float(np.min(R[extN] ** 2 * N[extN] ** 2 / (N[extN] ** 2 - v[extN] ** 2)))
        for ip, (pl, m, om) in enumerate(probes):
            if om <= m:
                continue
            p = math.sqrt(om * om - m * m)
            lab = {}
            if m > 0:
                e_ = om / m; bcm = bc_massive(e_, rs)
                aplat = math.asin(min(1.0, math.sqrt(max(om * om - m * m, 0.0))))
                lab = {"e": e_, "a_p_lat": aplat, "group_speed": math.sin(aplat) * math.cos(aplat) / om, "cos_2ap": math.cos(2.0 * aplat), "b_c_massive": bcm,
                       "pi_bc2_massive": math.pi * bcm * bcm, "limit_27rs2_over_4": 6.75 * rs * rs, "limit_4rs2_over_e2m1": 4.0 * rs * rs / (e_ * e_ - 1.0)}
            nrm = lab["pi_bc2_massive"] if lab else math.pi * bc2            # the printed ratio: the massive comparator (massless: the geometric pi b_c^2)
            t_probe0 = time.time()
            BG.clear()
            BG.update({"id": (ib, ip), "R": R, "N": N, "v": v, "m": m, "omega": om, "Rt": Rt_b, "R0": a.R0_over_rs * rs, "sig_om": a.sigma_omega_rel * om,
                       "ws": a.ws, "block": a.block, "gamma": a.gamma if a.absorber else None, "tgrid": tgrid, "read_js": frozenset(read_js),
                       "Rmax": float(R[-1]) - 16.0, "outer": outer})
            pool = ctx.Pool(a.workers) if ctx is not None else None
            chans, steps = [], []
            sig, settle = {}, None
            try:
                for k, j_to in enumerate(read_js, start=1):
                    t_step0 = time.time()
                    T = tgrid[j_to]
                    st_ = run_tasks(pool, [("advance", c[s_], j_to) for c in chans for s_ in ("plus", "minus")])
                    for i, c in enumerate(chans):
                        c["plus"], c["minus"] = st_[2 * i], st_[2 * i + 1]
                    n_use = rule_stop(chans, om, bc2)
                    while n_use is None and len(chans) < a.kappa_cap:
                        nb = 1 if pool is None else 4 * a.workers
                        kas = list(range(len(chans) + 1, min(a.kappa_cap, len(chans) + nb) + 1))
                        st_ = run_tasks(pool, [("init", sgn * ka, j_to) for ka in kas for sgn in (1, -1)])
                        n0 = len(chans)
                        for i, ka in enumerate(kas):
                            chans.append({"kappa": ka, "plus": st_[2 * i], "minus": st_[2 * i + 1]})
                        n_use = rule_stop(chans, om, bc2)
                        for c in chans[n0:(n_use if n_use is not None else len(chans))]:
                            print("%s | %s omega %.5f: |kappa| %d: A(+) %s A(-) %s (%.0f s)" % (bl[:12], pl, om, c["kappa"], {kk: round(x, 5) for kk, x in c["plus"].get("A", {}).items()},
                                  {kk: round(x, 5) for kk, x in c["minus"].get("A", {}).items()}, time.time() - t_run0), flush=True)
                    if n_use is not None and n_use < len(chans):
                        del chans[n_use:]                                          # beyond the stop (a batch's speculative channels): version 6's list
                    keys = ["%.4f" % tgrid[j] for j in read_js[:k]]
                    sig = {kk: summed(chans, "A", kk, p) for kk in keys}
                    s_list = [sig[kk] for kk in keys]
                    devs, settled = [], False
                    if k > wsteps:
                        s0 = s_list[k - 1 - wsteps]
                        devs = [((x - s0) / s0) if s0 > 0 else float("inf") for x in s_list[k - wsteps:]]
                        settled = s0 > 0 and all(abs(d) < a.settle_rel for d in devs)
                    w_in = summed(chans, "inward_inside", keys[-1], p)
                    w_tr = summed(chans, "inward_transit", keys[-1], p)
                    w_out = summed(chans, "outer_absorbed", keys[-1], p) if outer is not None else None
                    share = (w_tr / s_list[-1]) if s_list[-1] > 0 else float("inf")
                    rule7 = settled
                    settled = settled and share < a.transit_share                   # version 9's gate
                    now = time.time()
                    steps.append({"k": k, "T": T, "T_over_rs": T / rs, "sigma": s_list[-1], "ratio": s_list[-1] / nrm, "devs": devs,
                                  "factor": (s_list[k - 1 - wsteps] / s_list[-1]) if (k > wsteps and s_list[-1] > 0) else None, "settled": settled,
                                  "channels": len(chans), "inward_inside_ratio": w_in / nrm, "transit_share": share, "rule7_settled": rule7,
                                  "outer_absorbed_ratio": (w_out / nrm) if w_out is not None else None,
                                  "step_s": now - t_step0, "probe_s": now - t_probe0, "run_s": now - t_run0})
                    print("%s | %s omega %.5f: T %g r_s: sigma/(pi b_c(omega, m)^2) %.4f, the readings at T - %g and T against T - %g r_s %s percent (rule < %g: %s), inward-branch weight inside R_t + 2 (summed as sigma) over pi b_c(omega, m)^2 %.4f, in transit over sigma %.4f (gate < %g; version 7's rule %s), outer port absorbed (summed as sigma) over pi b_c(omega, m)^2 %s, channels %d, step %.0f s, probe %.0f s, run %.0f s"
                          % (bl[:12], pl, om, T / rs, s_list[-1] / nrm, a.T_step_over_rs, a.settle_window_over_rs, ", ".join("%+.1f" % (100 * d) for d in devs) if devs else "-",
                             100 * a.settle_rel, "settled" if settled else "not settled", w_in / nrm, share, a.transit_share, "met" if rule7 else "not met",
                             ("%.3e" % (w_out / nrm)) if w_out is not None else "-", len(chans), now - t_step0, now - t_probe0, now - t_run0), flush=True)
                    if settled:
                        settle = {"settled": True, "T": T, "T_over_rs": T / rs}
                        break
            finally:
                if pool is not None:
                    pool.close(); pool.join()
            keys = ["%.4f" % tgrid[j] for j in read_js[:len(steps)]]
            last = steps[-1]
            if settle is None:
                settle = {"settled": False, "T": None, "T_over_rs": None}
            settle.update({"rule": rule["text"], "ratio_at": last["ratio"], "change_at": last["devs"][-1] if last["devs"] else None, "factor_at": last["factor"],
                           "at_T_over_rs": last["T_over_rs"]})
            sig = {kk: summed(chans, "A", kk, p) for kk in keys}
            sig_abs = {kk: summed(chans, "absorbed", kk, p) for kk in keys} if a.absorber else {}
            if lab:
                lab["sigma_over_pi_bc2_massive"] = {kk: s_ / (math.pi * lab["b_c_massive"] ** 2) for kk, s_ in sig.items()}
            res = {"background": bl, "probe": pl, "m": m, "omega": om, "p": p, "T_list": [tgrid[j] for j in read_js[:len(steps)]], "sigma": sig,
                   "sigma_absorber": sig_abs, "massive": lab, "pi_bc2": math.pi * bc2, "pi_bc2_lapse": math.pi * bc2N, "27_pi_gE2": 27 * math.pi * (a.g * E) ** 2,
                   "sigma_over_pi_bc2": {kk: s_ / (math.pi * bc2) for kk, s_ in sig.items()}, "norm_ratio": nrm,
                   "inward_inside_summed": {kk: summed(chans, "inward_inside", kk, p) for kk in keys},
                   "plain_inside_summed": {kk: summed(chans, "plain_inside", kk, p) for kk in keys},
                   "inward_transit_summed": {kk: summed(chans, "inward_transit", kk, p) for kk in keys},
                   "beyond_Rmax_summed": {kk: summed(chans, "beyond_Rmax", kk, p) for kk in keys},
                   "outer_absorbed_summed": {kk: summed(chans, "outer_absorbed", kk, p) for kk in keys} if outer is not None else {},
                   "R_out_region_summed": {kk: summed(chans, "R_out_region", kk, p) for kk in keys} if outer is not None else {},
                   "settle": settle, "steps": steps, "length_s": time.time() - t_probe0,
                   "channels": [{"kappa": c["kappa"], "plus": public(c["plus"]), "minus": public(c["minus"])} for c in chans], "R_t_background": Rt_b,
                   "excluded_channels": [(int(c[s_]["kappa"]), c[s_]["error"]) for c in chans for s_ in ("plus", "minus") if "error" in c[s_]]}
            for kk, why in res["excluded_channels"]:
                print("%s | %s omega %.5f: channel kappa %+d excluded at launch: %s" % (bl[:12], pl, om, kk, why), flush=True)
            rec["results"].append(res)
            print("%s | %s omega %.5f: %s; length %.0f s (run %.0f s)" % (bl[:12], pl, om, ("settled at T %g r_s" % settle["T_over_rs"]) if settle["settled"] else
                  ("not settled within %g r_s" % last["T_over_rs"]), res["length_s"], time.time() - t_run0), flush=True)
            rec["elapsed_s"] = time.time() - t_run0
            json.dump(rec, open(out_json, "w"), indent=1, default=float)
            write_md(rec, md, a, done=False)
    t_end = time.time()
    rec["run_length_s"] = t_end - t_run0; rec["end_utc"] = utc(t_end); rec["elapsed_s"] = t_end - t_run0
    json.dump(rec, open(out_json, "w"), indent=1, default=float)
    write_md(rec, md, a, done=True)
    print("the run's length: %.0f s wall clock (%s to %s UTC)" % (rec["run_length_s"], rec["start_utc"], rec["end_utc"]), flush=True)


if __name__ == "__main__":
    main()
