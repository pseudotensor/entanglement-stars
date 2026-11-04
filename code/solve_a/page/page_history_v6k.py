#!/usr/bin/env python
"""page_history_v6k.py (the lane's analyst builder; engine spec 43.3 (1)): page_history_v6j.py with option (v), off by default, built as a
new file under the new-files rule: --collar-split adds, at every checkpoint row and per channel, the collar flux's incoming and outgoing
parts by the sign of the local group velocity within the collar shell, the windowed sign split engine_massive_channel_D_v5b's readout uses
(a sine window of L sites centred at the first site beyond R_collar, its orthonormal DFT zero-padded to 4 L, the local spinors u_+-(q) of
D(q) = sin q sigma_x + (kappa/R_c) sigma_y at the window centre's R_c (m = 0), the band energies omega_+-(q) = +-N_c E(q) + (v_c/2) sin 2q and
group velocities v_+-(q) = +-N_c sin q cos q/E(q) + v_c cos 2q; the deviation Delta = C - P_- on the window, C = Phi Phi^dagger the propagated
state and P_- = 1(h < 0) of the checkpoint's h, read as n_+-(q) = u^dagger (F Delta F^dagger)(q) u, which equals D's
sum_k lambda_k |<u, F x_k>|^2); the energy flux parts sum omega v n over v > 0 (outgoing) and v <= 0 (incoming) per band, divided by the
window's sum of squared weights, likewise the number flux and the local energy density; --collar-split-windows 32,16,64 (32 the reading,
16 and 64 the controls); saved per row beside the net flux (row L_at_t, the exact instantaneous collar current at the checkpoint clock, the
check of the local reading, not an identity), per channel and weighted (16 kappa); and the collar shell's covariance block (C and P_- on
the largest window's sites, with their sites, R, N and v) saved at every checkpoint to <out>/<tag>_collar_blocks/. With the option off
the code paths are v6j's and every output is v6j's (the gate: the history oracle bitwise against v6j's oracle run).
v6j's docstring follows.
page_history_v6j.py (the lane's analyst builder; engine spec 42.3 (3)): page_history_v6i.py with (i) the cubic sub-steps only when the record
is not held (v6i's step_record selected them for a held record inside the window, #65 item 7, confirmed at its lines 800 to 806); (ii)
--collector-select gen (off by default): per half, the candidates selected by the outside weight of their normalized new part, the
generalized eigenproblem G u = g H_D u (G = (chi K D)^dag (chi K D), H_D = (K D)^dag (K D) the new part's Gram, solved on H_D's range above
1e-12 of its largest eigenvalue), in place of the raw eigenvalues of G (#65 item 2); (iii) the joint outside weight lambda_min(Q^dag chi_cut Q)
of each block's collected set printed per channel beside the per-mode weights; (iv) the archive's Gram deviation ||F^dag F - I||_F after
each block printed as a test independent of the identity residual. With --collector-select raw (the default) every number is v6i's except
the added keys. v6i's docstring follows.
page_history_v6i.py (builder analyst, 2026-10-17; engine spec 39.1 (2)): page_history_v6h.py with the partners' Gram brought to the identity
(Lowdin, B (B^dagger B)^{-1/2}) after their construction on C_q and F_q. v6h's oracle (2026-10-17 01:58 UTC, box CPU) left ||B^dagger B - I||_F
at about 1e-15/min f(1 - f) at every ledger start (2.99e-06 at f(1 - f) 2.14e-10, 3.69e-09 at 5.82e-07): the occupations f near 1 come from
G with an absolute rounding of about 1e-15, so f(1 - f), the partners' normalization, carries it divided by f(1 - f), and so do their
overlaps; the pure state and the orthonormal archive keep the overlaps at that rounding, so the symmetric orthonormalization moves each
partner only by it. The Gram before the orthonormalization and its smallest eigenvalue are printed with the ledger. Everything else is v6h's.

v6h's docstring: page_history_v6h.py (builder analyst, 2026-10-17; engine spec 39.1 (2)): page_history_v6g.py with the partners of both transfer ledgers
built on the pure state closest to the propagated one and on an orthonormal archive. The sea's columns drift from orthonormality by rounding
(A s4: ||Phi^dagger Phi - I||_F 2.1e-11 at T/r_s 12, 7.2e-11 at 40) and so does the archive (2e-13), so C = Phi Phi^dagger is a projector
only to that level, and P14 sec. 7's partner b_j = K C F u_j/(f_j(1 - f_j))^(1/2) carries it divided by f_j(1 - f_j): B^dagger B - I =
u^dagger F^dagger (C^2 - C) F u/(f(1 - f)) (s4tr's kept ledger at T/r_s 40: the norms of its hole-like partners fall 6e-7 to 3.5e-5 below 1
as f(1 - f) falls from 6e-7 to 1.5e-8, defect times f(1 - f) about 4.5e-13 throughout), and at the formation's clock, where the collected
sector is nearly pure (S_esc 2.3e-7 nats on kappa 1 at T/r_s 12), f(1 - f) reaches the cut 1e-12. v6h builds the partners from
C_q = Phi (Phi^dagger Phi)^{-1} Phi^dagger (the projector on the sea's span: one Gram and its Cholesky factor per ledger start) and
F_q = F (F^dagger F)^{-1/2} (Lowdin): G = F_q^dagger C_q F_q = U diag(f) U^dagger, b_j = (1 - F_q F_q^dagger) C_q F_q u_j/(f_j(1 - f_j))^(1/2),
so B^dagger B = I up to rounding over f(1 - f). The sea and the archive themselves are not changed (S_esc and every row stay v6f's and
v6g's bit for bit); the two Gram deviations at each start are printed with the ledger. Everything else is v6g's.

v6g's docstring: page_history_v6g.py (builder analyst, 2026-10-17; engine spec 39.1 (2) and the lane's choice (a) of 01:08 UTC for A to T/r_s 100):
page_history_v6f.py with the transfer ledger anchored at a stated clock beside the peak-anchored one, both printed at every checkpoint, and
the history's continuation read as its records land. The state, the collector and every v6f readout are unchanged (S_esc and every row of
v6f bitwise on the same history: the oracle checks it).
(1) --ledger-anchor-rs X: P14 sec. 7's transfer ledger (v6f's construction, unchanged) started at the record nearest X r_s (A: the
    formation's record clock 12.125) on the archive collected by then, whether or not that record is a checkpoint (no row is added there):
    the p partners b_j = K C F u_j/(f_j(1 - f_j))^(1/2) in the retained complement (K = 1 - F F^dagger), carried with the sea, W_R
    accumulated at every later collection. With --transfer-at-peak beside it, v6f's peak-anchored ledger (restarted at every checkpoint
    whose weighted S_esc exceeds every earlier one) is kept as before.
(2) Both ledgers at every checkpoint after their start (after the checkpoint's collection, before a peak restart): per channel the
    retained part T_K = (1 - F F^dagger) B, diag and trace of T_K^dagger T_K, of the accumulated W_R and of W_H = I_p - T_K^dagger T_K - W_R
    (the identity's residual), the accumulation's own check |B^dagger F F^dagger B - W_R| (B is orthogonal to the archive at the ledger's
    start and both are carried by the same propagation, so the two agree up to rounding and the propagation's tail), the partners' Gram
    |B^dagger B - I|, and T_K's weight inside r_t, from r_t to R_cut and beyond R_cut. Printed in the log, stored in the rows
    ("ledger_anchored", "ledger_peak") and tabulated.
(3) The history's continuation (--record-ext JSON --fields-ext NPZ --ext-after-rs T1): the primary's records up to T1 r_s and the
    continuation's after it (the continuation's own records up to T1, where it holds them, compared with the primary's at T1 and printed
    as the seam: the largest |dN|, |dv| of the applied fields and |dE|/E there). With --follow the continuation's files are read again
    whenever the run needs a record past the loaded end (the next record only; a record is used once it is in both the json and the fields
    npz, the files read whole and parsed from the bytes whose md5 is printed; a read that fails, a partial file, is retried at the next
    poll), polling every --follow-poll-s seconds with the device cache released while waiting; a record already used that changes in a later
    read stops the run. The run ends at --T-end-rs (required with --follow), or at the continuation's last record when its done marker
    (--follow-done-marker) exists and a fresh read holds no later record; --follow-timeout-h without a new record saves the state and stops
    (resumable with --resume). --cubic-window is not combined with --follow.
(4) The ledgers' partners take the sea's own sub-steps inside step_record (v6f carried them after the record with the record's piecewise h,
    so inside a cubic window they missed the sub-steps: s4cubic's ledger columns are not consistent inside its window; its S_esc and every
    other readout are not affected). Outside a cubic window the partners' arithmetic is v6f's (the same banded h and step).
(5) The label reads "over the run". C (--freeze-at-rs) is not combined with --ledger-anchor-rs (C is not re-run).

v6f's docstring: page_history_v6f.py (builder analyst, 2026-10-16; engine spec 38.1 (3), for A extended to T/r_s 100 and A on the sigma/a 8
history): page_history_v6e.py with two readouts added; the state, the collector and every v6e readout unchanged.
(1) The geometry's work accumulated per record: the fields are piecewise constant per record (and per sub-step inside the cubic control),
    so Tr[h C] is conserved inside each and jumps by Tr[(h_new - h_old) C] where they change. Every jump is summed per channel (the jump
    into the next record's piecewise h at the record's end, on the propagated state; the sub-step jumps inside the cubic control), each
    record's jump stored with the record; at a checkpoint W_acc = (the jumps since the previous checkpoint - the change of the reference
    energy sum(e < 0) between the two checkpoints) / dt, the geometry's work on C - P_- accumulated per record, printed beside the
    finite-difference W (dE_inv,inside/dt + L) and the checkpoints' trapezoid of Tr[h_dot (C - P)]; the jump check (the jumps against
    the change of Tr[h C] between the checkpoints, the propagation's own error) printed. W_acc equals dE_inv/dt of the whole chain by
    construction; W - W_acc is L minus the change of the energy beyond the collar, the open collar's bookkeeping.
(2) The collected occupation's bins carry a resolution flag: readable only where 2 pi/omega at the bin's lower edge is at most a quarter of
    the exterior segment beyond R_cut (the chain's end minus R_cut); the composer prints kappa 2's soft counts beside kappa 1's as the null.

v6e's docstring: page_history_v6e.py (builder analyst, 2026-10-16): page_history_v6d.py with the device footprint of its traces bounded. v6d formed
every Re tr[X^dagger W B X] over a whole n2 x n0 block at once (B X, W B X, X^dagger and the product, each 2 GB at 8192 sites) and the
energy identity's K Phi and K P_- as whole blocks, so C (whose frozen eigenbasis already holds 2 x 4.3 GB) ran out of device memory at the
identity beside A s4 and lane 4's member (v6dA_c15, pod 1, 2026-10-16 21:43 UTC: 36.41 GiB held, 2 GiB more asked). v6e sums every such
trace over column blocks of --trace-chunk columns (default 1024; the same sums in another order, roundoff only) and forms K Phi and K P_-
block by block inside the identity; each checkpoint row carries the device peak since the previous checkpoint and within it
(torch.cuda.max_memory_allocated and max_memory_reserved, GiB), so the launch gates can use measured peaks. Nothing else changes.

v6d's docstring: page_history_v6d.py (builder analyst, 2026-10-16): page_history_v6c.py with one fix. The collection step's W_R accumulation read
self.transfer[kappa] for every channel with a new batch, so a ledger restart (--transfer-at-peak) that gave one channel no partner mode
(every f(1 - f) of its collected sector at or below 1e-12, which start_transfer skips) raised KeyError at that channel's next collection
(v6cA_s4far on the second H100, 2026-10-16 21:20 UTC, kappa 2 after the restart at T/r_s 8). v6d reads the channel's ledger with .get and
accumulates W_R only for a channel that has one; start_transfer logs the channel it skips. Nothing else changes.

v6c's docstring: page_history_v6c.py (builder analyst, 2026-10-16): page_history_v6b.py with five changes; the state and the collector of A are v6b's.
(1) The continuum comparator's greybody table on its 48 fixed nodes is computed once per channel and kept (v6b recomputed it at every
    checkpoint, about 19 s of CPU per channel pair): the same function on the same nodes, the numbers unchanged.
(2) --transfer-at-peak (S1 A.11, "at the S_esc peak"): the transfer ledger of P14 sec. 7 restarts at every checkpoint whose weighted S_esc
    exceeds every earlier one (the previous ledger dropped), so the one kept at the end starts at the run's peak; the restart clocks are
    printed (transfer_starts). With --transfer-at-rs the v6b behaviour is kept.
(3) C (#36 C, "the pilot's stage machinery, 1 dense eigendecomposition per control"): with --frozen-prop eig (the default) the sea, the
    archive and the ledger's partner modes are carried in the eigenbasis of the frozen record's h by exact phases from the freeze clock
    and materialized at each checkpoint (one dense product per channel and checkpoint, no Chebyshev series on the frozen h); the collar
    energy flow is evaluated per checkpoint interval (on a frozen h the per-record flows telescope to it) and the collar current at the
    checkpoint clock. --frozen-prop cheb keeps v6b's per-record Chebyshev propagation.
(5) A run continued from a saved state of A (--init-state without --freeze-at-rs, e.g. the ledger run from T/r_s 15) keeps A's collection
    clocks: the schedule's origin is the history's start (--T-start-rs), not the saved record (v6b counted from the saved record); the
    saved record itself is not collected again (A collected it before saving).
(4) C prints the chain end's reflection clock: the continuum light round trip from R_collar to the chain end and back in the frozen
    record's fields (outgoing N - |v|, ingoing N + |v|), with |v|/N at R_collar; each frozen row carries after_reflection_clock. The
    drain rule is S1's, unchanged.

v6b's docstring: page_history_v6b.py (builder analyst, 2026-10-16): page_history_v6.py with inbox #41's changes to S1 (REVIEWA_RULING_oneobject_MI.md E
and D15, owner-approved): no mutual information is computed or printed. v6's slab mutual information at --mi-at-rs (S1 A.12) is removed
(replaced by S4's A.3, another builder's post-processing of the stored states); its checkpoint clocks are kept by --extra-ck-rs (default
15,30), so the collection schedule is v6's; the collection step prints S(M) - S(M') = S_old - S_new, the archive's entropy before minus
after the new batch (minus the conditional entropy S(B|M) of the batch given the archive; positive when the escaping entropy falls), in
place of v6's S_old + S_B - S_new. Nothing else changes (the state, the collector, the identities, the transfer ledger W_R).

v6's docstring: page_history_v6.py (builder analyst, 2026-10-16; inbox #36 S1 revised, Computation A; REVIEWA_RULING_quasifrozen.md; the ideas of
P01_theorems.md secs. 2, 4, 5, 7, P07's DERIVATION.md sec. 10 and P14's response sec. 7 re-derived, not used as sources). Imports
page_ops_v6 (the operators, the Chebyshev order, the entropies, the hull and the ledgers), page_core_v6 (principal_delta, the principal
angles with the sharp dropped-pair ledger) and page_scatter_v5 (the continuum greybody for the comparators).

Computation A: the massless channel of the lattice Dirac operator propagated on the solved history of the massive collapse, its sources
not fed back. The background (#36 A 1): per stored record k of the history (record json + fields npz in the engine's layout), the applied
(averaged-back) site fields N_k, v_k on the engine's sites R = 1 ... n_e and the collar charge E_k; beyond the engine's domain the vacuum
Painleve-Gullstrand exterior at E_k (N = 1, v = -(2 g E_k / R)^(1/2)) appended to --nsites; the fields piecewise constant over
[T_k, T_k + tau) (the engine propagates its sea with record k's accepted fields over that interval: engine_collapse_v19's loop); the join
checked per record (the largest relative difference of N and v from the vacuum exterior at E_k over --join-rs, 16 to 32 r_s by default).
The channel (A 2): h[N, v] = (1/2){N, h_0} + (1/2){v, Q} of page_ops_v6 at m = 0, kappa = 1, 2 with weights 16 kappa (kappa 3 the guard).
The initial state (A 3): the ground state of h(T_0) per channel (one dense eigendecomposition), the occupied columns Phi (n2 x n0);
min(N - |v|) of the record printed. The propagation (A 4): per record, the Chebyshev action of exp(-i h_k tau) on Phi and on the archive of
collected modes F (the order set by the Bessel tail below --cheb-tol; printed), no dense eigendecomposition in the propagation; --cubic-window
T0,T1 (in r_s) replaces the piecewise-constant fields inside the window by the cubic (Catmull-Rom) time interpolation of the records with
sub-steps of 1 lattice unit (the control).

The state is propagated whole (the sea's every occupied column), so the global state C = Phi Phi^dagger is pure and known at every clock
and the escaping entropy is the direct route on the cumulative collected set (S1 repair 4): S_esc = tr s(F^dagger C F) with F every
collected mode propagated to the clock, exact up to the propagation's Chebyshev tail and rounding (the band printed: the accumulated tail
bound and the sea's unitarity defect). The collector (repairs 1 to 3 in their time-dependent form; collection every --collect-every-rs): at
a checkpoint the dense eigendecomposition of h(T_k) gives P_-(T_k) = 1(h(T_k) < 0); the deviation C - P_- is represented by its principal
pairs (page_core_v6.principal_delta on V_k^dagger Phi: the split hull in h(T_k)'s halves with the sharp dropped-pair ledger, repairs 1 and
2); per half h the memory's weight beyond R_cut outside the archive, G_h = c_h^dagger K chi K c_h with K = 1 - F F^dagger, is diagonalized
and its directions with weight at least 1 - eps_loc are the candidates (the active-subspace rule, repair 3); each candidate y = K c_h u is
split in the Killing shells of the exterior block of h(T_k) beyond R_cut (one dense eigendecomposition of the exterior block per
--ext-every-rs; bins of --bin-TH T_H up to --bin-max-TH T_H and one overflow bin, particles and holes by the sign of the Killing energy) by
the sign of the radial velocity i[h, R] within each shell: the outgoing part is collected (escaped), the incoming part stays in the system
and is printed as the bin's incoming fraction; the outgoing parts are orthonormalized against the archive (twice) and appended to F. The
archive is never pruned; its full Gram is checked every block (the new block against every earlier mode at one clock).

Readouts per checkpoint, per channel and weighted (A 7 to 12): E_inv(T_k) = Tr[h(T_k)(C - P_-)]; the creation rate W = dE_inv/dt + L
against Tr[h_dot (C - P_-)] from the records' field differences (h_dot = ({dN, h_0} + {dv, Q})/(2 tau) at the record); the collar current L
= Tr[C j_chi] with j_chi = (i/2)[h^2, chi_c] (chi_c beyond --R-collar-rs) per record and its time integral; the exact collar identity at
the checkpoint, E_out = Tr[{chi_c, h}/2 (C - P_-)] = E_remove + M_chi with E_remove the collected sector's energy (the mode ledger) and
M_chi = Tr[({chi_c, h}/2 - h + K h K)(C - P_-)] printed; the comparators L_H at the record's dense surface slope kappa_T and at 1/(4 g E_k)
(the run's channels' continuum greybody, page_scatter_v5), L/L_H; the collected occupation per bin per block, the window-averaged spectrum
over --window-rs, the implied partner occupation nu = n_out/(Gamma f_FD) in the soft bins at T_m from kappa_T; S_esc with its band against
eta A/a^2 and A/(4G) at the run's g; the transfer ledger (P14 sec. 7's construction) at --transfer-at-rs; (v6b: no slab readout, no
mutual information); the label on every output. The run ends at the last
record of the history (or --T-end-rs).

Usage: python page_history_v6g.py --record JSON --fields NPZ --out DIR --tag TAG [--nsites 8192 --channels 1,2 --R-cut-rs 24 ...]
       [--ledger-anchor-rs 12.125 --transfer-at-peak] [--record-ext JSON --fields-ext NPZ --ext-after-rs 40 --follow --T-end-rs 100
       --follow-poll-s 60 --follow-timeout-h 6 --follow-done-marker PATH]; the oracle: page_history_oracle_v6i.py"""
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v6 as OPS            # noqa: E402
import page_core_v6 as CORE          # noqa: E402

CD = OPS.CD
RD = OPS.RD
T_START = time.time()
LABEL_A = ("the massless channel of the lattice Dirac operator on the solved history of the massive collapse, its sources not fed back, "
           "E_rad^(0)/E = %s over the run; the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit")


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T_START, msg), flush=True)


def _f(x, fmt):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------------------------------------------- the history
class History:
    """the stored records of a solved collapse history in the engine's layout: the fields npz (R, T, N_applied, v_applied, tau, ell) and the
    record json (records with T and E, args with g and rs). The site fields of the chain of --nsites sites: the record's applied fields on
    the engine's sites, the vacuum exterior at the record's E beyond them. v6g (3): an extension (another run's json and fields npz in the
    same layout, e.g. a continuation that keeps the stored records up to its restart and appends its own) supplies the records after the
    clock ext["after_rs"] r_s; with ext["follow"] it is read again whenever a record past the loaded end is needed (ensure)."""

    def __init__(self, record, fields, nsites, g=None, fields_key="applied", synthetic=None, ext=None):
        if synthetic is not None:
            self.D = synthetic["json"]; Z = synthetic["npz"]
        else:
            self.D = json.load(open(record))
            Z = np.load(fields, allow_pickle=True)
        args = self.D.get("args", {})
        self.g = float(g if g is not None else args.get("g", 0.25))
        self.rs_nom = float(args.get("rs", 0.0)) or None
        recs = [r for r in self.D["records"] if r.get("E") is not None]
        self.T = np.asarray(Z["T"], float)
        self.Re = np.asarray(Z["R"], float)
        self.ne = len(self.Re)
        kN, kv = ("N_applied", "v_applied") if fields_key == "applied" else ("N", "v")
        self.kN, self.kv = kN, kv
        self.N_e = np.asarray(Z[kN], float); self.v_e = np.asarray(Z[kv], float)
        self.tau = float(Z["tau"]); self.ell = float(Z["ell"]) if "ell" in Z else None
        byT = {round(float(r["T"]), 6): r for r in recs}
        self.rec = [byT.get(round(float(t), 6)) or min(recs, key=lambda q: abs(q["T"] - t)) for t in self.T]
        self.E = np.array([float(r["E"]) for r in self.rec])
        self.nsites = int(nsites)
        self.R = np.arange(self.nsites, dtype=float) + 1.0
        if self.nsites < self.ne:
            raise ValueError("the chain (%d sites) is shorter than the engine's domain (%d)" % (self.nsites, self.ne))
        if not np.allclose(self.Re, self.R[: self.ne]):
            raise ValueError("the engine's sites are not R = 1 ... n_e")
        self.label = str(Z["label"]) if "label" in Z else ""
        self.fields_key = fields_key
        self.ext = None; self.follow = False; self.seam = None; self.loads = []; self.n_primary = len(self.T)
        self.ext_error = None; self.ext_stat = None; self.ext_label = None; self.ext_md5 = None; self.n_failed_reads = 0
        if ext is not None:
            self._init_ext(ext)

    # ---------------------------------------------------------------------------------------------- v6g (3): the continuation
    def _init_ext(self, ext):
        self.ext = dict(ext)
        self.follow = bool(ext.get("follow"))
        rs = float(ext.get("rs_nominal") or self.rs_nom or 0.0)
        if rs <= 0:
            raise ValueError("the extension's clock needs r_s (the record's args or --rs-nominal)")
        Ta = float(ext["after_rs"]) * rs
        keep = self.T <= Ta + 1e-9
        if not keep.any() or abs(self.T[keep][-1] - Ta) > 1e-6:
            raise ValueError("the primary history holds no record at the extension's clock T = %g (T/r_s %g)" % (Ta, ext["after_rs"]))
        n = int(keep.sum())
        self.T = self.T[:n]; self.N_e = self.N_e[:n]; self.v_e = self.v_e[:n]; self.rec = self.rec[:n]; self.E = self.E[:n]
        self.n_primary = n; self.T_after = Ta
        got = self._read_ext()
        if got is None:
            if not self.follow:
                raise ValueError("the extension's files could not be read: %s" % self.ext_error)
            log("v6g: the extension's files are not readable yet (%s): the run waits for them" % self.ext_error)
            return
        self._seam_check(got)
        self._append(got)

    def _read_ext(self):
        """one read of the extension's json and fields npz: the bytes read whole, their md5, the parse from those bytes; the records after
        the seam that are in both files, consecutive from the first; None (the reason in ext_error) when either cannot be parsed."""
        try:
            pj, pf = self.ext["record"], self.ext["fields"]
            st = (os.stat(pj).st_mtime_ns, os.stat(pj).st_size, os.stat(pf).st_mtime_ns, os.stat(pf).st_size)
            bj = open(pj, "rb").read(); bf = open(pf, "rb").read()
            D = json.loads(bj.decode("utf-8"))
            Z = np.load(io.BytesIO(bf), allow_pickle=True)
            T = np.asarray(Z["T"], float); Re = np.asarray(Z["R"], float)
            N_all = np.asarray(Z[self.kN], float); v_all = np.asarray(Z[self.kv], float)
            tau = float(Z["tau"]); ell = float(Z["ell"]) if "ell" in Z else None
            lab = str(Z["label"]) if "label" in Z else ""
        except Exception as ex:                                               # noqa: BLE001 (a partial file: retried at the next poll)
            self.ext_error = "%s: %s" % (type(ex).__name__, ex); self.n_failed_reads += 1
            return None
        if len(Re) != self.ne or not np.array_equal(Re, self.Re):
            raise ValueError("the extension's engine sites differ from the primary's")
        if abs(tau - self.tau) > 1e-12 or (ell is not None and self.ell is not None and abs(ell - self.ell) > 1e-12):
            raise ValueError("the extension's tau %g or ell %s differs from the primary's (%g, %s)" % (tau, ell, self.tau, self.ell))
        if N_all.shape[0] != len(T) or v_all.shape[0] != len(T):
            self.ext_error = "the fields' arrays and T differ in length (%d, %d, %d)" % (N_all.shape[0], v_all.shape[0], len(T)); self.n_failed_reads += 1
            return None
        byT = {round(float(r["T"]), 6): r for r in D.get("records", []) if isinstance(r, dict) and r.get("E") is not None}
        idx = []
        for i, t in enumerate(T):
            if t <= self.T_after + 1e-9:
                continue
            if round(float(t), 6) not in byT:
                break
            idx.append(i)
        seam_i = [i for i, t in enumerate(T) if abs(t - self.T_after) < 1e-9]
        return {"T": T[idx], "N": N_all[idx], "v": v_all[idx], "rec": [byT[round(float(T[i]), 6)] for i in idx], "stat": st,
                "md5_json": hashlib.md5(bj).hexdigest(), "md5_fields": hashlib.md5(bf).hexdigest(), "label": lab, "args": D.get("args", {}),
                "seam": ({"N": N_all[seam_i[0]], "v": v_all[seam_i[0]], "rec": byT.get(round(float(T[seam_i[0]]), 6))} if seam_i else None),
                "n_file": int(len(T)), "T_file_last": float(T[-1]) if len(T) else None}

    def _seam_check(self, got):
        s = got.get("seam")
        rs = float(self.ext.get("rs_nominal") or self.rs_nom)
        k = self.n_primary - 1
        if s is None:
            self.seam = {"T_over_rs": self.T_after / rs, "present": False}
            log("v6g: the seam at T/r_s %.4f: the extension holds no record there (its records start after it)" % (self.T_after / rs))
            return
        dN = float(np.max(np.abs(s["N"] - self.N_e[k]))); dv = float(np.max(np.abs(s["v"] - self.v_e[k])))
        Ee = float(s["rec"]["E"]) if (s.get("rec") and s["rec"].get("E") is not None) else None
        dE = (abs(Ee - self.E[k]) / abs(self.E[k])) if Ee is not None else None
        self.seam = {"T_over_rs": self.T_after / rs, "present": True, "max_abs_dN": dN, "max_abs_dv": dv, "rel_dE": dE,
                     "label": "the extension's own record at the seam against the primary's (the run uses the primary's up to the seam, the extension's after it)"}
        log("v6g: the seam at T/r_s %.4f: the extension's record there against the primary's: max |dN| %.3e, max |dv| %.3e, |dE|/E %s" % (
            self.T_after / rs, dN, dv, ("%.3e" % dE) if dE is not None else "-"))

    def _append(self, got):
        """the read's records past the loaded end appended, after the records already taken from the extension are found unchanged."""
        n_used = len(self.T) - self.n_primary
        if n_used > len(got["T"]):
            raise RuntimeError("the extension now holds %d records after the seam, fewer than the %d already used" % (len(got["T"]), n_used))
        if n_used:
            T_old = self.T[self.n_primary:]
            same = (np.array_equal(T_old, got["T"][:n_used]) and np.array_equal(self.N_e[self.n_primary:], got["N"][:n_used])
                    and np.array_equal(self.v_e[self.n_primary:], got["v"][:n_used])
                    and all(float(self.E[self.n_primary + i]) == float(got["rec"][i]["E"]) for i in range(n_used)))
            if not same:
                raise RuntimeError("the extension changed a record already used (one of its first %d records after the seam)" % n_used)
        new = slice(n_used, len(got["T"]))
        Tn = got["T"][new]
        if len(Tn):
            prev = self.T[-1]
            for t in Tn:
                if abs((t - prev) - self.tau) > 1e-9 * max(1.0, self.tau):
                    raise RuntimeError("the extension's record at T %.6f does not follow T %.6f by tau %g" % (t, prev, self.tau))
                prev = t
            self.T = np.concatenate([self.T, Tn]); self.N_e = np.concatenate([self.N_e, got["N"][new]], axis=0)
            self.v_e = np.concatenate([self.v_e, got["v"][new]], axis=0)
            self.rec = self.rec + list(got["rec"][new]); self.E = np.concatenate([self.E, np.array([float(r["E"]) for r in got["rec"][new]])])
            rs = float(self.ext.get("rs_nominal") or self.rs_nom)
            self.loads.append({"wall": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "added": int(len(Tn)), "records": int(len(self.T)),
                               "last_T_over_rs": float(self.T[-1]) / rs, "md5_json": got["md5_json"], "md5_fields": got["md5_fields"],
                               "file_records": got["n_file"]})
        self.ext_stat = got["stat"]; self.ext_label = got["label"]; self.ext_md5 = (got["md5_json"], got["md5_fields"])

    def ensure(self, k, on_wait=None):
        """v6g (3): True when record k is loaded (the extension read again while it is not); False when the extension has ended (its done
        marker exists and a fresh read holds no record k). A read whose files have not changed since the last good read is skipped."""
        if k < len(self.T):
            return True
        if self.ext is None or not self.follow:
            return False
        rs = float(self.ext.get("rs_nominal") or self.rs_nom)
        poll = float(self.ext.get("poll_s", 60.0)); tmax = float(self.ext.get("timeout_h", 6.0)) * 3600.0
        t_w = time.time(); last_log = time.time(); waited = False
        while True:
            try:
                pj, pf = self.ext["record"], self.ext["fields"]
                st = (os.stat(pj).st_mtime_ns, os.stat(pj).st_size, os.stat(pf).st_mtime_ns, os.stat(pf).st_size)
            except OSError:
                st = None
            if st is not None and st != self.ext_stat:
                got = self._read_ext()
                if got is not None:
                    if self.seam is None:
                        self._seam_check(got)
                    self._append(got)
                    if k < len(self.T):
                        if waited:
                            log("v6g: record %d (T/r_s %.4f) is in; waited %.0f s; the extension's last record now T/r_s %.4f (md5 json %s, fields %s)" % (
                                k, self.T[k] / rs, time.time() - t_w, self.T[-1] / rs, self.ext_md5[0], self.ext_md5[1]))
                        return True
            mk = self.ext.get("done_marker")
            if mk and os.path.exists(mk):
                got = self._read_ext()
                if got is not None:
                    self._append(got)
                if k < len(self.T):
                    return True
                log("v6g: the extension's done marker %s exists and a fresh read holds no record %d: the history ends at record %d (T/r_s %.4f)" % (
                    mk, k, len(self.T) - 1, self.T[-1] / rs))
                return False
            if time.time() - t_w > tmax:
                raise TimeoutError("no record %d within %.1f h (the extension's last record T/r_s %.4f; last read error: %s)" % (
                    k, tmax / 3600.0, self.T[-1] / rs, self.ext_error))
            if not waited:
                waited = True
                log("v6g: waiting for record %d (T/r_s %.4f): the extension's last record is T/r_s %.4f; polling every %.0f s" % (
                    k, (self.T[-1] + (k - len(self.T) + 1) * self.tau) / rs, self.T[-1] / rs, poll))
                if on_wait is not None:
                    on_wait()
            elif time.time() - last_log > 1800.0:
                last_log = time.time()
                log("v6g: still waiting for record %d (%.0f s so far; failed reads %d, the last: %s)" % (k, time.time() - t_w, self.n_failed_reads, self.ext_error))
            time.sleep(poll)

    def __len__(self):
        return len(self.T)

    def fields(self, k):
        """(N, v) on the chain at record k: the record's applied fields, the vacuum exterior at E_k beyond the engine's sites."""
        N = np.ones(self.nsites); v = -np.sqrt(2.0 * self.g * self.E[k] / self.R)
        N[: self.ne] = self.N_e[k]; v[: self.ne] = self.v_e[k]
        return N, v

    def join_check(self, k, lo, hi):
        """the largest relative differences of the record's N and v from the vacuum exterior at E_k over lo <= R <= hi (engine sites)."""
        sel = (self.Re >= lo) & (self.Re <= hi)
        if not sel.any():
            return None, None
        vpg = -np.sqrt(2.0 * self.g * self.E[k] / self.Re[sel])
        return float(np.max(np.abs(self.N_e[k][sel] - 1.0))), float(np.max(np.abs(self.v_e[k][sel] - vpg) / np.abs(vpg)))

    def cubic_fields(self, t):
        """the Catmull-Rom interpolation in time of the chain's fields at clock t (the control of A 4)."""
        k = int(np.clip(np.searchsorted(self.T, t, side="right") - 1, 0, len(self.T) - 2))
        u = (t - self.T[k]) / (self.T[k + 1] - self.T[k])
        ks = [max(k - 1, 0), k, k + 1, min(k + 2, len(self.T) - 1)]
        F = [self.fields(q) for q in ks]
        w = [-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0, -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2]
        N = sum(wi * Fi[0] for wi, Fi in zip(w, F)); v = sum(wi * Fi[1] for wi, Fi in zip(w, F))
        return N, v


# --------------------------------------------------------------------------------------------------------------- the operators
class ChannelOps:
    """one channel kappa: the chain h_0 and Q (page_ops_v6.chain_sparse, m = 0), the clocked operator at given fields as a host CSR and a
    device banded operator, the collar current, the radial-velocity operator."""

    def __init__(self, kappa, R, device):
        self.kappa = int(kappa); self.w = 16.0 * self.kappa
        self.R = R; self.n2 = 2 * len(R); self.device = device
        self.h0, self.Q = OPS.chain_sparse(R, 0.0, float(kappa))
        self.R2 = torch.as_tensor(np.repeat(R, 2), dtype=RD, device=device)

    def h_csr(self, N, v):
        return OPS.clocked_csr(self.h0, self.Q, N, v)

    def banded(self, h):
        return OPS.Banded(h, self.device)

    def velocity_apply(self, h, X):
        """i[h, R] X for the clocked host h (CSR) on the device: (i h_ab (R_b - R_a)) applied banded."""
        co = h.tocoo()
        r2 = np.repeat(self.R, 2)
        V = sp.csr_matrix((1j * co.data * (r2[co.col] - r2[co.row]), (co.row, co.col)), shape=co.shape)
        return OPS.Banded(V, self.device).apply(X)

    def hdot_csr(self, dN, dv, dt):
        """({dN, h_0} + {dv, Q})/(2 dt): the operator derivative of the clocked h at record field differences dN, dv over dt."""
        Nd = sp.diags(np.repeat(np.asarray(dN, float), 2)); Vd = sp.diags(np.repeat(np.asarray(dv, float), 2))
        return ((Nd @ self.h0 + self.h0 @ Nd) + (Vd @ self.Q + self.Q @ Vd)).tocsr() * (0.5 / dt)


def cheb_apply(Bh, X, tau, tol):
    """exp(-i tau h) X for a device Banded h (Gershgorin bound w): the Chebyshev series on [-w, w], the order by the Bessel tail below tol
    (page_ops_v6.cheb_order); returns (Y, K, tail)."""
    if X.shape[1] == 0:
        return X, 0, 0.0
    w = Bh.bound
    z = w * tau
    K, tail = OPS.cheb_order(z, tol)
    k = np.arange(K + 1)
    from scipy.special import jv
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
    """cheb_apply over column chunks (device memory); returns (X, K, tail)."""
    K = 0; tail = 0.0
    for j0 in range(0, X.shape[1], chunk):
        X[:, j0: j0 + chunk], K, tail = cheb_apply(Bh, X[:, j0: j0 + chunk], tau, tol)
    return X, K, tail


TRACE_CHUNK = [1024]                                                    # v6e: --trace-chunk sets it


def tr_blocked(B, X, w=None, chunk=None):
    """v6e: Re tr[X^dagger W B X] for a device Banded B (W a diagonal weight column, or None) summed over column blocks of X."""
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


# ---------------------------------------------------------------------------------------------------------------- the state
class HistChannel:
    """one channel's state on the history: the sea Phi (n2 x n0, every occupied column, propagated), the archive F of collected modes
    (n2 x m, propagated with the sea, never pruned) with each mode's collection clock, outside weight and bin, and the accumulators."""

    def __init__(self, ops):
        self.ops = ops
        self.Phi = None
        self.F = None
        self.F_t = []; self.F_g = []; self.F_bin = []
        self.cheb_K_max = 0; self.cheb_tail_acc = 0.0
        self.collar_flow_acc = 0.0                      # sum over records of the exact collar energy flow (frozen h_k over each record)
        self.gram_F2 = 0.0; self.gram_max = 0.0
        self.work_jump_since = 0.0; self.E_hC_last = None; self.E_ref_last = None   # v6f (1)
        self.unit_defect = 0.0
        self.ext = None                                 # the exterior block's shells (cached; recomputed every ext_every)
        self.ext_t = None

    def trace_h(self, Bh, X):
        """Re tr[X^dagger h X] for a device Banded h (v6e: over column blocks)."""
        return tr_blocked(Bh, X)

    def outer_root(self, N, v):
        """the outermost zero of N - |v| (linear interpolation) or None."""
        f = np.asarray(N, float) - np.abs(np.asarray(v, float))
        s = np.nonzero((f[:-1] <= 0) & (f[1:] > 0))[0]
        if len(s) == 0:
            return None
        i = int(s[-1]); R = self.ops.R
        return float(R[i] + (R[i + 1] - R[i]) * (-f[i]) / (f[i + 1] - f[i]))


def exterior_shells(ops, h, R_cut, TH, bin_TH, bin_max_TH, device):
    """the exterior block beyond R_cut of the clocked h (dense eigendecomposition): its Killing energies and eigenvectors; per bin of
    |omega|/T_H (width bin_TH up to bin_max_TH, one overflow bin) and per sign of omega (particles omega > 0, holes omega < 0) the outgoing
    and incoming subspaces by the sign of the radial velocity i[h_X, R] within the bin; returns the index set, the eigenvalues and the per-bin
    (O, I) bases in the exterior coordinates."""
    idx = np.nonzero(np.repeat(ops.R, 2) > R_cut)[0]
    hX = h[idx][:, idx]
    e, U = OPS.eigh_csr(hX.tocsr(), device)
    RX = ops.R[idx // 2]
    co = hX.tocoo()
    r2 = RX
    VX = sp.csr_matrix((1j * co.data * (r2[co.col] - r2[co.row]), (co.row, co.col)), shape=co.shape)   # i[h_X, R]
    VXd = OPS.dense_from_csr(VX, device)
    nb = int(round(bin_max_TH / bin_TH))
    edges = [b * bin_TH for b in range(nb + 1)] + [1e30]
    eabs = (e.abs() / TH).cpu().numpy()
    bins = []
    for sgn in (1, -1):
        sel_sign = (e > 0) if sgn > 0 else (e < 0)
        ss = sel_sign.cpu().numpy()
        for b in range(len(edges) - 1):
            j = np.nonzero(ss & (eabs >= edges[b]) & (eabs < edges[b + 1]))[0]
            if len(j) == 0:
                continue
            Ub = U[:, torch.as_tensor(j, device=device)]
            Wb = Ub.mH @ (VXd @ Ub)
            wv, wu = torch.linalg.eigh(0.5 * (Wb + Wb.mH))
            O = Ub @ wu[:, wv > 0]; I_ = Ub @ wu[:, wv <= 0]
            bins.append({"sign": sgn, "bin": b, "lo": edges[b], "hi": edges[b + 1], "n": len(j), "O": O, "I": I_})
    del VXd
    return {"idx": torch.as_tensor(idx, device=device), "e": e, "bins": bins, "TH": TH, "R_cut": R_cut, "nX": len(idx)}


def split_outgoing(ext, Y):
    """the candidates Y (n2 x c, site basis) split in the exterior shells by the sign of the Killing energy and, within each shell, by the sign
    of the radial velocity: per sign s the outgoing part sum over the bins of sign s of O O^dagger Y_X (embedded in the full space), so that
    every collected column lies in one spectral half of the exterior block (its vacuum occupation 0 or 1 up to the spectral tails at R_cut;
    an unsplit outgoing part would carry the vacuum's entanglement across R_cut as escaped entropy, P01 sec. 3); per bin the outgoing and
    incoming squared norms (c candidates summed). Returns (the 2c columns [the positive-energy parts, the negative-energy parts], rows)."""
    idx = ext["idx"]
    YX = Y[idx]
    rows = []
    out_X = {1: torch.zeros_like(YX), -1: torch.zeros_like(YX)}
    for bn in ext["bins"]:
        a_o = bn["O"].mH @ YX; a_i = bn["I"].mH @ YX
        out_X[bn["sign"]] += bn["O"] @ a_o
        rows.append({"sign": bn["sign"], "bin": bn["bin"], "out": float(a_o.abs().pow(2).sum()), "in": float(a_i.abs().pow(2).sum())})
    Yout = torch.zeros((Y.shape[0], 2 * Y.shape[1]), dtype=Y.dtype, device=Y.device)
    Yout[idx, : Y.shape[1]] = out_X[1]; Yout[idx, Y.shape[1]:] = out_X[-1]
    return Yout, rows


def orthonormal_against(Y, F, eps):
    """the orthonormal basis of span(Y) projected out of span(F) (classical Gram-Schmidt twice, then an SVD with relative threshold eps)."""
    if Y.shape[1] == 0:
        return Y
    for _ in range(2):
        if F is not None and F.shape[1]:
            Y = Y - F @ (F.mH @ Y)
    U, s, _ = torch.linalg.svd(Y, full_matrices=False)
    if s.numel() == 0:
        return Y[:, :0]
    keep = s > eps * max(float(s.max()), 1e-300)
    Q = U[:, keep]
    if F is not None and F.shape[1]:
        Q = Q - F @ (F.mH @ Q)
        U2, s2, _ = torch.linalg.svd(Q, full_matrices=False)
        Q = U2[:, s2 > 0.5]
    return Q


def entropy_of(Phi, F):
    """tr s(F^dagger C F) with C = Phi Phi^dagger (the direct route): (S, the covariance's eigenvalues)."""
    if F is None or F.shape[1] == 0:
        return 0.0, None
    A = Phi.mH @ F
    G = A.mH @ A
    ev = torch.linalg.eigvalsh(0.5 * (G + G.mH))
    return float(OPS.binary_entropy_t(ev).sum()), ev


# v6b (#41 D15): v6's slabs_p07 (the slab mutual information) is removed.


# --------------------------------------------------------------------------------------------------------------- the driver
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=None); ap.add_argument("--fields", default=None)
    ap.add_argument("--fields-key", dest="fields_key", default="applied", choices=["applied", "stored"])
    ap.add_argument("--g", type=float, default=None, help="G/a^2 (default: the record's)")
    ap.add_argument("--rs-nominal", dest="rs_nominal", type=float, default=None, help="the r_s of the settings in units of r_s (default: the record's rs)")
    ap.add_argument("--nsites", type=int, default=8192)
    ap.add_argument("--channels", default="1,2")
    ap.add_argument("--T-start-rs", dest="T_start_rs", type=float, default=0.0)
    ap.add_argument("--T-end-rs", dest="T_end_rs", type=float, default=None, help="default: the history's last record")
    ap.add_argument("--collect-every-rs", dest="collect_every_rs", type=float, default=2.0)
    ap.add_argument("--R-cut-rs", dest="R_cut_rs", type=float, default=24.0)
    ap.add_argument("--R-collar-rs", dest="R_collar_rs", type=float, default=16.0)
    ap.add_argument("--join-rs", dest="join_rs", default="16,32")
    ap.add_argument("--eps-loc", dest="eps_loc", type=float, default=1e-2)
    ap.add_argument("--eps-form", dest="eps_form", type=float, default=1e-7)
    ap.add_argument("--eps-orth", dest="eps_orth", type=float, default=1e-12)
    ap.add_argument("--eps-coll", dest="eps_coll", type=float, default=1e-5, help="a candidate's outgoing part below this squared norm is not collected")
    ap.add_argument("--pair-ledger", dest="pair_ledger", default="sharp", choices=["sharp", "v5"])
    ap.add_argument("--collector-select", dest="collector_select", default="raw", choices=["raw", "gen"], help="v6j (ii)")
    ap.add_argument("--collector-mode", dest="collector_mode", default="velocity", choices=["velocity", "shells"],
                    help="velocity: the outgoing directions of the outside-weight candidate subspace by i[h, R] within it (the collected modes stay "
                         "combinations of one half's memory directions); shells: the candidates' outgoing parts in the exterior block's shells (they "
                         "carry the vacuum's entanglement across R_cut; kept as the comparison)")
    ap.add_argument("--cheb-tol", dest="cheb_tol", type=float, default=1e-14)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--ext-every-rs", dest="ext_every_rs", type=float, default=10.0)
    ap.add_argument("--bin-TH", dest="bin_TH", type=float, default=0.5)
    ap.add_argument("--bin-max-TH", dest="bin_max_TH", type=float, default=40.0)
    ap.add_argument("--window-rs", dest="window_rs", type=float, default=13.0)
    ap.add_argument("--soft-max-TH", dest="soft_max_TH", type=float, default=5.0)
    ap.add_argument("--extra-ck-rs", dest="extra_ck_rs", default="15,30",
                    help="v6b: extra checkpoint clocks (r_s), v6's --mi-at-rs schedule kept; no mutual information is computed (#41 D15)")
    ap.add_argument("--transfer-at-rs", dest="transfer_at_rs", type=float, default=None)
    ap.add_argument("--transfer-modes", dest="transfer_modes", type=int, default=16)
    ap.add_argument("--transfer-at-peak", dest="transfer_at_peak", action="store_true",
                    help="v6c: the transfer ledger restarts at each new maximum of the weighted S_esc at a checkpoint (S1 A.11, at the peak)")
    ap.add_argument("--trace-chunk", dest="trace_chunk", type=int, default=1024, help="v6e: the column block of every trace on the device")
    ap.add_argument("--frozen-prop", dest="frozen_prop", default="eig", choices=["eig", "cheb"],
                    help="v6c: C's propagation, the frozen h's eigenbasis by exact phases (eig) or v6b's per-record Chebyshev series (cheb)")
    ap.add_argument("--cubic-window", dest="cubic_window", default=None, help="T0,T1 in r_s: cubic time interpolation with 1-unit sub-steps inside")
    ap.add_argument("--eta", type=float, default=0.9192)
    ap.add_argument("--save-every", dest="save_every", type=int, default=4, help="checkpoints between saved states")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-wall-hours", dest="max_wall_hours", type=float, default=24.0)
    # the frozen-record controls (#36 C) and their starting states
    ap.add_argument("--save-at-rs", dest="save_at_rs", default="", help="clocks (r_s) at which the state is also saved to <tag>_state_at_<T>.pt")
    ap.add_argument("--init-state", dest="init_state", default=None, help="start from a saved state of run A (the archive kept) instead of the ground state")
    ap.add_argument("--freeze-at-rs", dest="freeze_at_rs", type=float, default=None, help="C: h frozen at the record of this clock from there on")
    ap.add_argument("--freeze-duration-rs", dest="freeze_duration_rs", type=float, default=300.0)
    ap.add_argument("--end-flux-frac", dest="end_flux_frac", type=float, default=1e-3)
    ap.add_argument("--end-window", dest="end_window", type=int, default=10)
    ap.add_argument("--end-entropy-frac", dest="end_entropy_frac", type=float, default=0.01)
    # v6g: the anchored ledger and the history's continuation
    ap.add_argument("--ledger-anchor-rs", dest="ledger_anchor_rs", type=float, default=None, help="v6g (1): the transfer ledger anchored at the record nearest this clock (r_s)")
    ap.add_argument("--record-ext", dest="record_ext", default=None, help="v6g (3): the continuation's record json")
    ap.add_argument("--fields-ext", dest="fields_ext", default=None, help="v6g (3): the continuation's fields npz")
    ap.add_argument("--ext-after-rs", dest="ext_after_rs", type=float, default=None, help="v6g (3): the continuation's records after this clock (r_s), the primary's up to it")
    ap.add_argument("--follow", action="store_true", help="v6g (3): read the continuation again whenever a record past the loaded end is needed")
    ap.add_argument("--follow-poll-s", dest="follow_poll_s", type=float, default=60.0)
    ap.add_argument("--follow-timeout-h", dest="follow_timeout_h", type=float, default=6.0)
    ap.add_argument("--follow-done-marker", dest="follow_done_marker", default=None)
    # v6k (v): the collar flux's signed split and the collar shell's covariance block (engine spec 43.3 (1)); off by default
    ap.add_argument("--collar-split", dest="collar_split", action="store_true", help="v6k (v): the windowed signed split of the collar flux per row and the collar block per checkpoint")
    ap.add_argument("--collar-split-windows", dest="collar_split_windows", default="32,16,64", help="v6k (v): window sizes in sites, the reading first")
    a = ap.parse_args(argv)
    if a.record_ext and (not a.fields_ext or a.ext_after_rs is None):
        ap.error("--record-ext needs --fields-ext and --ext-after-rs")
    if a.follow and not a.record_ext:
        ap.error("--follow needs --record-ext")
    if a.follow and a.T_end_rs is None:
        ap.error("--follow needs --T-end-rs")
    if a.follow and a.cubic_window:
        ap.error("--cubic-window is not combined with --follow")
    if a.ledger_anchor_rs is not None and a.freeze_at_rs is not None:
        ap.error("--ledger-anchor-rs is not combined with --freeze-at-rs (C is not re-run)")
    return a


_GAMMA_TABLE = {}                                                       # v6c (1): kappa -> Gamma on the 48 fixed nodes


def continuum_comparator(channels, rs, T):
    """sum over the run's channels of (64 kappa/(2 pi)) int domega omega Gamma_kappa f(omega/T), omega = x/r_s (page_recursion_v5's
    comparator: the continuum massless greybody of page_scatter_v5 on 48 Gauss-Legendre nodes of x in [0, 1.5])."""
    import page_scatter_v5 as SC
    if T is None or T <= 0:
        return None
    xs, ws = np.polynomial.legendre.leggauss(48)
    x = 0.75 * (xs + 1.0); w = 0.75 * ws
    tot = 0.0
    for k in channels:
        G = _GAMMA_TABLE.get(k)
        if G is None:
            G = SC.gamma_table(k, x); _GAMMA_TABLE[k] = G
        arg = np.clip(x / (rs * T), -700, 700)
        tot += 64.0 * float(k) / (2.0 * math.pi) * float(np.sum(w * x * G / (np.exp(arg) + 1.0))) / (rs * rs)
    return tot


class HistRun:
    def __init__(self, a, history=None):
        self.a = a
        self.dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
        torch.set_num_threads(a.threads)
        os.makedirs(a.out, exist_ok=True)
        self.prefix = os.path.join(a.out, a.tag)
        self._cs_blocks = {}                                                   # v6k (v)
        TRACE_CHUNK[0] = max(1, int(getattr(a, "trace_chunk", 1024)))
        ext = None
        if getattr(a, "record_ext", None):                                    # v6g (3)
            ext = {"record": a.record_ext, "fields": a.fields_ext, "after_rs": a.ext_after_rs, "follow": a.follow, "poll_s": a.follow_poll_s,
                   "timeout_h": a.follow_timeout_h, "done_marker": a.follow_done_marker, "rs_nominal": a.rs_nominal}
        self.H = history if history is not None else History(a.record, a.fields, a.nsites, a.g, a.fields_key, ext=ext)
        H = self.H
        self.g = H.g
        self.rs_nom = float(a.rs_nominal or H.rs_nom or 2.0 * H.g * H.E[0])
        self.tau = H.tau
        self.channels = [int(x) for x in a.channels.split(",") if x.strip()]
        self.chs = [HistChannel(ChannelOps(k, H.R, self.dev)) for k in self.channels]
        self.R_cut_abs = a.R_cut_rs * self.rs_nom; self.R_collar_abs = a.R_collar_rs * self.rs_nom
        self.k0 = int(np.argmin(np.abs(H.T - a.T_start_rs * self.rs_nom)))
        self.k_sched0 = self.k0                       # v6c (5): the collection schedule's origin, kept when a run continues from a saved state
        self.k_init = None                            # v6c (5): the saved state's record (no second collection there)
        self.kend = (self._k_of(a.T_end_rs * self.rs_nom) if a.T_end_rs is not None else len(H) - 1)
        every = max(1, int(round(a.collect_every_rs * self.rs_nom / self.tau)))
        self.ck = set(range(self.k0, self.kend + 1, every)) | {self.kend}
        self.xk = set(self._k_of(float(x) * self.rs_nom) for x in a.extra_ck_rs.split(",") if x.strip())
        self.ck |= {k for k in self.xk if self.k0 <= k <= self.kend}
        self.tr_k = self._k_of(a.transfer_at_rs * self.rs_nom) if a.transfer_at_rs is not None else None
        if self.tr_k is not None:
            self.ck.add(self.tr_k)
        self.cw = [float(x) * self.rs_nom for x in a.cubic_window.split(",")] if a.cubic_window else None
        self.rows = []; self.k = self.k0; self.n_ck = 0; self.records = []
        self.last_ck = None
        self.transfer = None
        self.label = None
        self.save_at_k = set(self._k_of(float(x) * self.rs_nom) for x in a.save_at_rs.split(",") if x.strip())
        self.kf = self._k_of(a.freeze_at_rs * self.rs_nom) if a.freeze_at_rs is not None else None
        self.frozen = False; self.fz = {}; self.t_now = None
        self.end_count = 0; self.S_peak = 0.0; self.end_reason = None
        self.tr_peak_S = 0.0; self.transfer_starts = []; self.transfer_epoch = 0          # v6c (2)
        self.refl = None; self.t_freeze = None                                         # v6c (4)
        self.k_anchor = self._k_of(a.ledger_anchor_rs * self.rs_nom) if getattr(a, "ledger_anchor_rs", None) is not None else None   # v6g (1)
        self.anchor = None; self.anchor_started = False; self.follow_end = None

    def _k_of(self, T_abs):
        """v6g (3): the record index of the clock T_abs: the nearest loaded record (v6f's rule), or, following an extension past its loaded
        end, the index that record will have at the records' spacing tau."""
        H = self.H
        if H.follow and T_abs > H.T[-1] + 0.5 * self.tau:
            return len(H) - 1 + int(round((T_abs - H.T[-1]) / self.tau))
        return int(np.argmin(np.abs(H.T - T_abs)))

    def _T_of_k(self, k):
        H = self.H
        return float(H.T[k]) if k < len(H) else float(H.T[-1] + (k - len(H) + 1) * self.tau)

    def _release_cache(self):
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------------------------------------------- start
    def start(self):
        a, H = self.a, self.H
        N, v = H.fields(self.k0)
        mnv = float(np.min(N - np.abs(v)))
        log("Computation A: %s; the history %s (%d records, T/r_s %.3f to %.3f, tau %g, ell %s, g %g, the engine's %d sites), the chain %d sites "
            "(%.1f r_s nominal); r_s nominal %g; channels %s; R_cut %.1f r_s, collar %.1f r_s" % (
                "start", H.label or a.record, len(H), H.T[self.k0] / self.rs_nom, self._T_of_k(self.kend) / self.rs_nom, self.tau, H.ell, self.g, H.ne,
                H.nsites, H.nsites / self.rs_nom, self.rs_nom, self.channels, a.R_cut_rs, a.R_collar_rs))
        if H.ext is not None:
            log("v6g: the history's continuation %s after T/r_s %g: %d primary records, %d of the continuation loaded (to T/r_s %.4f)%s; the run to T/r_s %.4f (record %d)" % (
                a.record_ext, a.ext_after_rs, H.n_primary, len(H) - H.n_primary, H.T[-1] / self.rs_nom, ", followed as its records land" if H.follow else "",
                self._T_of_k(self.kend) / self.rs_nom, self.kend))
        if self.k_anchor is not None:
            log("v6g: the transfer ledger anchored at record %d (T/r_s %.4f)%s" % (self.k_anchor, self._T_of_k(self.k_anchor) / self.rs_nom,
                                                                                  ", the peak-anchored ledger beside it" if a.transfer_at_peak else ""))
        log("the initial record T/r_s %.4f: min(N - |v|) %.6f (A 3: subcritical, the continued sea's start with no pockets when positive); E %.6f, 2 g E %.4f" % (
            H.T[self.k0] / self.rs_nom, mnv, H.E[self.k0], 2 * self.g * H.E[self.k0]))
        for c in self.chs:
            h = c.ops.h_csr(N, v)
            e, V = OPS.eigh_csr(h, self.dev)
            neg = e < 0
            c.Phi = V[:, neg].clone()
            c.F = torch.zeros((c.ops.n2, 0), dtype=CD, device=self.dev)
            log("kappa %d: the ground state of h(T_0): %d occupied of %d, %d levels within 1e-12 of zero" % (c.ops.kappa, int(neg.sum()), c.ops.n2, int((e.abs() < 1e-12).sum())))
            del V, e
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.min_N_minus_absv_T0 = mnv

    # ------------------------------------------------------------------------------------------------------------ records
    def step_record(self, k, t0=None, t1=None):
        """propagate every channel over [T_k, T_k + tau): record k's fields (or the cubic interpolation inside the control window); the
        exact collar energy flow over the record (frozen h over each sub-step: Re tr[C chi h] at both ends with the same h) and the
        instantaneous collar current at T_k. C: t0, t1 given (the frozen record's fields over any interval)."""
        a, H = self.a, self.H
        held = t0 is not None                                                  # v6f: a held-fixed record has no jump
        if t0 is None:
            t0 = H.T[k]; t1 = H.T[k + 1] if k + 1 < len(H) else t0 + self.tau
        sub = [(t0, t1, None)]
        if self.cw is not None and self.cw[0] <= t0 < self.cw[1] and not held:     # v6j (i)
            nsub = max(1, int(round(t1 - t0)))
            sub = [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic") for i in range(nsub)]
        rec = {"k": k, "t": t0, "per_channel": {}}
        for c in self.chs:
            flow = 0.0; L0 = None; Kmax = 0; tail = 0.0; jump = 0.0
            h_cur = c.ops.h_csr(*H.fields(k))                                    # v6f (1): the record's piecewise h (the checkpoint's)
            for (sa, sb, mode) in sub:
                if mode == "cubic":
                    N, v = H.cubic_fields(0.5 * (sa + sb))
                else:
                    N, v = H.fields(k)
                h = c.ops.h_csr(N, v)
                if mode == "cubic" and not held:                                 # v6f (1): the sub-step's jump on the state
                    jump += tr_blocked(c.ops.banded(h - h_cur), c.Phi); h_cur = h
                Bh = c.ops.banded(h)
                chi = (c.ops.R2 > self.R_collar_abs).to(CD)[:, None]
                Eo0 = tr_blocked(Bh, c.Phi, chi)
                if L0 is None:
                    J = OPS.Banded(OPS.current_out_csr(h, c.ops.R, self.R_collar_abs), self.dev)
                    L0 = tr_blocked(J, c.Phi)
                    del J
                c.Phi, K_, tl = propagate(Bh, c.Phi, sb - sa, a.cheb_tol, a.chunk)
                if c.F.shape[1]:
                    c.F, _, _ = propagate(Bh, c.F, sb - sa, a.cheb_tol, a.chunk)
                for L_ in (self.transfer, self.anchor):                          # v6g (4): the ledgers' partners take the sea's own sub-steps
                    tr_ = (L_ or {}).get(str(c.ops.kappa))
                    if tr_ is not None:
                        tr_["B"], _, _ = propagate(Bh, tr_["B"], sb - sa, a.cheb_tol, a.chunk)
                Eo1 = tr_blocked(Bh, c.Phi, chi)
                flow += Eo1 - Eo0
                Kmax = max(Kmax, K_); tail += tl
            if not held and k + 1 < len(H):                                      # v6f (1): the jump into the next record's h at its start
                jump += tr_blocked(c.ops.banded(c.ops.h_csr(*H.fields(k + 1)) - h_cur), c.Phi)
            c.work_jump_since += jump
            c.collar_flow_acc += flow
            c.cheb_K_max = max(c.cheb_K_max, Kmax); c.cheb_tail_acc += tail
            rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L0, "cheb_K": Kmax, "cheb_tail": tail, "work_jump": jump}
        rec["collar_flow_weighted"] = sum(c.ops.w * rec["per_channel"][str(c.ops.kappa)]["collar_flow"] for c in self.chs)
        rec["L_weighted"] = sum(c.ops.w * rec["per_channel"][str(c.ops.kappa)]["L_at_t"] for c in self.chs)
        rec["work_jump_weighted"] = sum(c.ops.w * rec["per_channel"][str(c.ops.kappa)]["work_jump"] for c in self.chs)
        rec["sub_steps"] = len(sub)
        self.records.append(rec)

    # --------------------------------------------------------------------------------------------------------- checkpoints
    def bin_occupation(self, c, Q, Phi):
        """the block's excess occupation per exterior shell bin in its outgoing subspace: particles N_b = tr[P G P^dagger] (omega > 0) and
        holes tr[P P^dagger] - tr[P G P^dagger] (omega < 0) with P = O_b^dagger Q_X and G = Q^dagger C Q."""
        A = Phi.mH @ Q
        GQ = A.mH @ A
        QX = Q[c.ext["idx"]]
        out = []
        for bn in c.ext["bins"]:
            if bn["O"].shape[1] == 0:
                continue
            P = bn["O"].mH @ QX
            occ = float(torch.trace(P @ GQ @ P.mH).real); w_ = float(torch.trace(P @ P.mH).real)
            val = occ if bn["sign"] > 0 else (w_ - occ)
            if abs(val) > 0 or w_ > 0:
                out.append({"sign": bn["sign"], "bin": bn["bin"], "lo_TH": bn["lo"], "hi_TH": bn["hi"], "excess": val, "weight": w_})
        return out

    def checkpoint(self, k, t=None, collect_override=None):
        a, H = self.a, self.H
        t = float(H.T[k]) if t is None else float(t)
        E_k = float(H.E[k]); rs_k = 2.0 * self.g * E_k; TH = 1.0 / (4.0 * math.pi * rs_k)
        N, v = H.fields(k)
        if self.frozen:
            dN = np.zeros_like(N); dv = np.zeros_like(v)                      # C: h frozen, h_dot = 0
        else:
            kn = k + 1 if k + 1 < len(H) else k - 1
            Nn, vn = H.fields(kn)
            sgn_d = 1.0 if kn > k else -1.0
            dN, dv = sgn_d * (Nn - N), sgn_d * (vn - v)
        lo, hi = [float(x) * self.rs_nom for x in a.join_rs.split(",")]
        jN, jv = H.join_check(k, lo, hi)
        rj = H.rec[k] or {}
        collect = (((k - self.k_sched0) % max(1, int(round(a.collect_every_rs * self.rs_nom / self.tau))) == 0) and k != self.k_sched0
                   and k != self.k_init)                                  # v6c (5): the saved record was collected by A before its save
        if collect_override is not None:
            collect = collect_override
        row = {"k": k, "t": t, "t_over_rs": t / self.rs_nom, "E": E_k, "two_gE": rs_k, "T_H": TH, "join_max_dN": jN, "join_max_rel_dv": jv,
               "kappa_T": rj.get("kappa_T"), "four_gE_kappa_T": rj.get("four_gE_kappa_T"), "R_t_record": rj.get("R_t"), "n_pocket_record": rj.get("n_pocket"),
               "min_N_minus_absv": float(np.min(N - np.abs(v))), "collect": collect, "per_channel": {}}
        t_ck = time.time()
        if self.dev.type == "cuda":                                          # v6e: the device peak since the previous checkpoint, then reset
            row["device_peak_before_GiB"] = {"allocated": torch.cuda.max_memory_allocated(self.dev) / 2.0 ** 30, "reserved": torch.cuda.max_memory_reserved(self.dev) / 2.0 ** 30}
            torch.cuda.reset_peak_memory_stats(self.dev)
        for c in self.chs:
            tc = time.time()
            h = c.ops.h_csr(N, v); Bh = c.ops.banded(h)
            fzc = self.fz.get(str(c.ops.kappa)) if self.frozen else None
            if fzc is not None:
                e, V = fzc["e"], fzc["V"]                                       # C: the one eigendecomposition of the frozen h
            else:
                e, V = OPS.eigh_csr(h, self.dev)
            neg = e < 0; nneg = int(neg.sum())
            Vneg = V[:, neg]
            Phi = c.Phi; n0 = Phi.shape[1]
            E_hC = c.trace_h(Bh, Phi); E_ref = float(e[neg].sum())
            E_inv = E_hC - E_ref
            chi_c = (c.ops.R2 > self.R_collar_abs).to(CD)[:, None]
            E_out = tr_blocked(Bh, Phi, chi_c) - tr_blocked(Bh, Vneg, chi_c)
            Bd = OPS.Banded(c.ops.hdot_csr(dN, dv, self.tau), self.dev)
            W_hdot = c.trace_h(Bd, Phi) - c.trace_h(Bd, Vneg)
            Gs = Phi.mH @ Phi
            Gs.diagonal().sub_(1.0)
            unit = float(torch.linalg.norm(Gs)); del Gs
            c.unit_defect = max(c.unit_defect, unit)
            # the deviation's principal pairs in h(T_k)'s halves (repairs 1, 2 in the representation; the state itself is not truncated); C: computed
            # once at the freeze, then rotated by the frozen phases (the deviation's span is invariant under the frozen h)
            method = "sine" if n0 == nneg else "cos+sin"
            if fzc is not None:
                c_mem = torch.exp(-1j * e.to(CD) * (t - fzc["t0"]))[:, None] * fzc["c0"]; half_mem = fzc["half0"]; pinfo = fzc["pinfo"]
            else:
                Phi_c = V.mH @ Phi
                c_mem, half_mem, MD, lam, pinfo = CORE.principal_delta(Phi_c, neg, a.eps_form, a.eps_orth, self.dev, method=method, ledger=a.pair_ledger)
                del Phi_c
                if self.frozen:
                    self.fz[str(c.ops.kappa)] = {"e": e, "V": V, "c0": c_mem.clone(), "half0": half_mem.clone(), "pinfo": pinfo, "t0": t}
            D = V @ c_mem
            del c_mem
            pc = {"n0": n0, "n_neg": nneg, "E_hC": E_hC, "E_ref": E_ref, "E_inv": E_inv, "E_out": E_out, "E_inv_inside": E_inv - E_out, "W_hdot": W_hdot,
                  "sea_unitarity_F": unit, "memory_rank": int(D.shape[1]), "principal_method": method, "sin_max": pinfo["sin_theta_max"],
                  "dropped_pairs": pinfo["n_dropped_pairs"], "dropped_pair_ledger": pinfo["entropy_bound_dropped_pairs"], "pair_ledger": pinfo["pair_ledger"],
                  "collar_flow_acc": c.collar_flow_acc, "cheb_K_max": c.cheb_K_max, "cheb_tail_acc": c.cheb_tail_acc}
            pc["work_jumps_since_last"] = c.work_jump_since                     # v6f (1)
            if a.collar_split:                                                  # v6k (v)
                pc["collar_split"] = self.collar_split(c, Phi, Vneg, N, v, t)
            if c.E_hC_last is not None:
                pc["E_hC_change"] = E_hC - c.E_hC_last
                pc["jump_check"] = (E_hC - c.E_hC_last) - c.work_jump_since
                pc["W_acc_interval"] = c.work_jump_since - (E_ref - c.E_ref_last)
            c.E_hC_last = E_hC; c.E_ref_last = E_ref; c.work_jump_since = 0.0
            F = c.F
            if collect:
                KD = (D - F @ (F.mH @ D)) if F.shape[1] else D
                chi_cut = (c.ops.R2 > self.R_cut_abs).to(CD)[:, None]
                X = chi_cut * KD
                Gw = X.mH @ X
                Gw = 0.5 * (Gw + Gw.mH)
                del X
                cand = []; gsel = []; n_unsel = 0; w_unsel = 0.0; g_unsel_max = 0.0
                for hs in (1, -1):
                    idx = (half_mem == hs).nonzero().ravel()
                    if idx.numel() == 0:
                        continue
                    if a.collector_select == "gen":                                  # v6j (ii): the normalized new part's outside weight
                        Kd_ = KD[:, idx]; Hd = Kd_.mH @ Kd_; Hd = 0.5 * (Hd + Hd.mH)
                        sh, Vh = torch.linalg.eigh(Hd)
                        keep = sh > 1e-12 * float(sh.max().clamp_min(1e-300))
                        Wd = Vh[:, keep] / torch.sqrt(sh[keep])[None, :]
                        Gg = Wd.mH @ Gw[idx][:, idx] @ Wd
                        g_, Ug = torch.linalg.eigh(0.5 * (Gg + Gg.mH))
                        U_ = Wd @ Ug
                    else:
                        g_, U_ = torch.linalg.eigh(Gw[idx][:, idx])
                    sel = g_ >= 1.0 - a.eps_loc
                    n_unsel += int((~sel).sum()); w_unsel += float(g_[~sel].clamp_min(0.0).sum())
                    if bool((~sel).any()):
                        g_unsel_max = max(g_unsel_max, float(g_[~sel].max()))
                    if bool(sel.any()):
                        cand.append(KD[:, idx] @ U_[:, sel]); gsel.append(g_[sel])
                del KD
                bin_rows = []; Q = torch.zeros((c.ops.n2, 0), dtype=CD, device=self.dev); n_cand = 0
                if cand:
                    Y = torch.cat(cand, dim=1); g_all = torch.cat(gsel)
                    n_cand = int(Y.shape[1])
                    Y = Y / torch.linalg.norm(Y, dim=0).clamp_min(1e-300)[None, :]
                    if c.ext is None or (t - c.ext_t) >= a.ext_every_rs * self.rs_nom - 1e-9:
                        c.ext = None
                        if self.dev.type == "cuda":
                            torch.cuda.empty_cache()
                        c.ext = exterior_shells(c.ops, h, self.R_cut_abs, TH, a.bin_TH, a.bin_max_TH, self.dev); c.ext_t = t
                    if a.collector_mode == "shells":
                        Yout, bin_rows = split_outgoing(c.ext, Y)
                        no = torch.linalg.norm(Yout, dim=0) ** 2
                        nc_ = Y.shape[1]
                        parts = []
                        for half_cols in (slice(0, nc_), slice(nc_, 2 * nc_)):              # per sign of the exterior Killing energy
                            Ys = Yout[:, half_cols]; ks = no[half_cols] > a.eps_coll
                            if bool(ks.any()):
                                base = torch.cat([F] + parts, dim=1) if (F.shape[1] or parts) else None
                                q_ = orthonormal_against(Ys[:, ks], base, 1e-12)
                                if q_.shape[1]:
                                    parts.append(q_)
                        if parts:
                            Q = torch.cat(parts, dim=1)
                        del Yout
                    else:
                        # per half: the candidate subspace orthonormalized against the archive, the radial velocity i[h, R] diagonalized in it, its
                        # positive directions collected (outgoing), the negative kept in the system (incoming); the shells classify the collected
                        parts = []; n_in_dirs = 0; vel_rows = []
                        offs = 0
                        for blk, gb in zip(cand, gsel):
                            nb_ = blk.shape[1]
                            Yb = Y[:, offs: offs + nb_]; offs += nb_
                            base = torch.cat([F] + parts, dim=1) if (F.shape[1] or parts) else None
                            Qb = orthonormal_against(Yb, base, 1e-12)
                            if Qb.shape[1] == 0:
                                continue
                            VQ = c.ops.velocity_apply(h, Qb)
                            Wv = Qb.mH @ VQ
                            wv, wu = torch.linalg.eigh(0.5 * (Wv + Wv.mH))
                            pos = wv > 0
                            n_in_dirs += int((~pos).sum())
                            vel_rows.append({"candidates": int(Qb.shape[1]), "outgoing": int(pos.sum()), "velocity_min": float(wv.min()), "velocity_max": float(wv.max())})
                            if bool(pos.any()):
                                parts.append(Qb @ wu[:, pos])
                        if parts:
                            Q = torch.cat(parts, dim=1)
                        pc_vel = {"velocity_split": vel_rows, "incoming_directions_kept": n_in_dirs}
                        if Q.shape[1]:
                            _, bin_rows = split_outgoing(c.ext, Q)
                    del Y
                pc_j = {"collector_select": a.collector_select}                   # v6j (iii)
                if Q.shape[1]:
                    Mq = Q.mH @ (chi_cut * Q)
                    pc_j["joint_outside_weight"] = float(torch.linalg.eigvalsh(0.5 * (Mq + Mq.mH)).min())
                S_old, _ = entropy_of(Phi, F)
                Fn = torch.cat([F, Q], dim=1) if Q.shape[1] else F
                S_new, evn = entropy_of(Phi, Fn)
                S_vac, _ = entropy_of(Vneg, Fn)                                       # tr s(F^dagger P_- F): the archive's vacuum entropy now
                pc["S_vac_archive"] = S_vac; pc["S_esc_minus_S_vac"] = S_new - S_vac
                S_B, _ = entropy_of(Phi, Q) if Q.shape[1] else (0.0, None)
                inc = 0.0; ov_max = 0.0
                if Q.shape[1]:
                    g1 = Q.mH @ Q
                    g1.diagonal().sub_(1.0)
                    inc = float(torch.linalg.norm(g1)) ** 2
                    if F.shape[1]:
                        ov = F.mH @ Q
                        ov_max = float(ov.abs().max()); inc += 2.0 * float(torch.linalg.norm(ov)) ** 2
                c.gram_F2 += inc; c.gram_max = max(c.gram_max, math.sqrt(c.gram_F2))
                wout = ((Q.abs() ** 2) * (c.ops.R2 > self.R_cut_abs).to(RD)[:, None]).sum(dim=0) if Q.shape[1] else torch.zeros(0, dtype=RD, device=self.dev)
                occ = self.bin_occupation(c, Q, Phi) if Q.shape[1] else []
                L_ext = float(np.asarray(c.ops.R)[-1]) - self.R_cut_abs              # v6f (2): the exterior segment beyond R_cut
                for b_ in occ:
                    b_["readable"] = bool(b_["lo_TH"] > 0 and 2.0 * math.pi / (b_["lo_TH"] * TH) <= 0.25 * L_ext)
                n_in = sum(r_["in"] for r_ in bin_rows); n_out = sum(r_["out"] for r_ in bin_rows)
                for L_ in (self.transfer, self.anchor):                      # v6d: a channel without a ledger accumulates nothing; v6g: both ledgers
                    tr_c = (L_ or {}).get(str(c.ops.kappa))
                    if tr_c is not None and Q.shape[1]:
                        tb = Q.mH @ tr_c["B"]
                        tr_c["W_R"] += tb.mH @ tb
                c.F = Fn
                c.F_t += [t] * int(Q.shape[1]); c.F_g += [float(x) for x in wout]
                F = c.F
                if a.collector_mode == "velocity" and cand:
                    pc.update(pc_vel)
                pc.update(pc_j)
                pc["archive_gram_dev_F"] = float(torch.linalg.norm(Fn.mH @ Fn - torch.eye(Fn.shape[1], dtype=Fn.dtype, device=Fn.device))) if Fn.shape[1] else 0.0   # v6j (iv)
                pc.update({"candidates": n_cand, "collected": int(Q.shape[1]), "archive": int(F.shape[1]), "S_old": S_old, "S_esc": S_new, "S_B": S_B,
                           "S_M_minus_S_Mprime": S_old - S_new, "gram_all_history_F": math.sqrt(c.gram_F2), "gram_new_vs_old_max": ov_max,
                           "outside_weight_min": float(wout.min()) if wout.numel() else None, "outside_weight_max": float(wout.max()) if wout.numel() else None,
                           "incoming_fraction": (n_in / (n_in + n_out)) if (n_in + n_out) > 0 else None, "bins_split": bin_rows, "bins_occupation": occ,
                           "unselected_count": n_unsel, "unselected_weight_beyond_cut": w_unsel, "unselected_g_max": g_unsel_max,
                           "covariance_eigs_extremes": [float(evn.min()), float(evn.max())] if evn is not None else None})
            else:
                S_now, _ = entropy_of(Phi, F)
                S_vac, _ = entropy_of(Vneg, F)
                pc.update({"S_esc": S_now, "archive": int(F.shape[1]), "collected": 0, "S_vac_archive": S_vac, "S_esc_minus_S_vac": S_now - S_vac})
            # the energy identity and the collar identity with the archive at this clock
            if F.shape[1]:
                FPhi = F.mH @ Phi; FV = F.mH @ Vneg
                hF = Bh.apply(F)
                E_K = 0.0; Ex1 = 0.0; chk = TRACE_CHUNK[0]
                for X_, FX_, sg_ in ((Phi, FPhi, 1.0), (Vneg, FV, -1.0)):           # v6e: K X by column blocks
                    for j0 in range(0, X_.shape[1], chk):
                        KXb = X_[:, j0: j0 + chk] - F @ FX_[:, j0: j0 + chk]
                        E_K += sg_ * tr_blocked(Bh, KXb, chunk=chk)
                        Ex1 += sg_ * float(torch.trace((hF.mH @ KXb) @ FX_[:, j0: j0 + chk].mH).real)
                        del KXb
                E_F = float(torch.trace((F.mH @ hF) @ (FPhi @ FPhi.mH - FV @ FV.mH)).real)
                E_x = 2.0 * Ex1
                del FPhi, FV, hF
            else:
                E_K = E_inv; E_F = 0.0; E_x = 0.0
            pc.update({"E_K": E_K, "E_F": E_F, "E_cross": E_x, "E_remove": E_F + E_x, "M_chi": E_out - E_inv + E_K,
                       "energy_identity_residual": E_inv - (E_K + E_F + E_x), "collar_identity_residual": E_out - (E_F + E_x) - (E_out - E_inv + E_K)})
            # the outer root, the transfer ledger's start
            r_t = c.outer_root(N, v)
            pc["r_t"] = r_t
            if self.tr_k is not None and k == self.tr_k and F.shape[1] and (self.transfer is None or str(c.ops.kappa) not in self.transfer):
                self.start_transfer(c, Phi, F, r_t)
            pc["seconds"] = time.time() - tc
            row["per_channel"][str(c.ops.kappa)] = pc
            if not self.frozen:
                del V, e
            del Vneg, D
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        if self.dev.type == "cuda":
            row["device_peak_GiB"] = {"allocated": torch.cuda.max_memory_allocated(self.dev) / 2.0 ** 30, "reserved": torch.cuda.max_memory_reserved(self.dev) / 2.0 ** 30}
            log("device peak (GiB, torch): this checkpoint allocated %.2f, reserved %.2f; since the previous checkpoint allocated %.2f, reserved %.2f" % (
                row["device_peak_GiB"]["allocated"], row["device_peak_GiB"]["reserved"], row["device_peak_before_GiB"]["allocated"], row["device_peak_before_GiB"]["reserved"]))
        row["ledger_anchored"] = self.ledger_snapshot(self.anchor, row)               # v6g (2): after the collection, before a peak restart
        row["ledger_peak"] = self.ledger_snapshot(self.transfer, row)
        self.finish_row(row, t_ck)
        self.ledger_log(row)
        if a.transfer_at_peak:
            self.transfer_peak_check(row)

    def transfer_peak_check(self, row):
        """v6c (2): a checkpoint whose weighted S_esc exceeds every earlier one restarts the ledger here (the previous one dropped)."""
        S = row.get("S_esc_weighted") or 0.0
        if S <= self.tr_peak_S + 1e-12 or not any(c.F is not None and c.F.shape[1] for c in self.chs):
            return
        self.tr_peak_S = S
        self.transfer = None
        for c in self.chs:
            if c.F is not None and c.F.shape[1]:
                self.start_transfer(c, c.Phi, c.F, row["per_channel"][str(c.ops.kappa)].get("r_t"), t0=row["t"])
        self.transfer_epoch += 1
        self.transfer_starts.append({"t": row["t"], "t_over_rs": row["t_over_rs"], "S_esc_weighted": S})
        row["transfer_restart"] = True

    def _partners(self, c, Phi, F, r_t, t0):
        """P14 sec. 7's transfer ledger re-derived (v6f's start_transfer, its arithmetic unchanged): the collected sector's covariance
        G = F^dagger C F = U diag(f) U^dagger; for the p modes with the largest f(1 - f) the partner b_j = K C F u_j/(f_j(1 - f_j))^(1/2) in
        the retained complement (K = 1 - F F^dagger; unit norm for a pure state); their weights inside r_t, between r_t and R_cut and beyond.
        Returns the ledger's dict (B, W_R = 0, f, t0, the norms and weights), or None when no f(1 - f) exceeds 1e-12."""
        a = self.a
        # v6h: the pure state closest to the propagated one, C_q = Phi (Phi^dagger Phi)^{-1} Phi^dagger, and the orthonormal archive
        # F_q = F (F^dagger F)^{-1/2}; the sea and the archive themselves are not changed
        Gs = Phi.mH @ Phi
        Gs = 0.5 * (Gs + Gs.mH)
        Gs.diagonal().sub_(1.0)
        sea_gram = float(torch.linalg.norm(Gs))
        Gs.diagonal().add_(1.0)
        Lc = torch.linalg.cholesky(Gs)
        del Gs
        gF = F.mH @ F
        gF = 0.5 * (gF + gF.mH)
        arch_gram = float(torch.linalg.norm(gF - torch.eye(gF.shape[0], dtype=CD, device=self.dev)))
        eF, VF = torch.linalg.eigh(gF)
        Fq = F @ (VF @ (eF.clamp_min(1e-300).rsqrt().to(CD)[:, None] * VF.mH))
        del gF, eF, VF
        A = Phi.mH @ Fq
        Ai = torch.cholesky_solve(A, Lc)
        del Lc
        G = A.mH @ Ai
        del A
        f, U = torch.linalg.eigh(0.5 * (G + G.mH))
        ff = (f * (1.0 - f)).clamp_min(0.0)
        order = torch.argsort(ff, descending=True)[: a.transfer_modes]
        order = order[ff[order] > 1e-12]
        if order.numel() == 0:
            return None
        CFu = Phi @ (Ai @ U[:, order])
        del Ai
        B = CFu - Fq @ (Fq.mH @ CFu)
        B = B / torch.sqrt(ff[order]).to(CD)[None, :]
        del CFu, Fq
        # v6i: the partners' Gram to the identity (Lowdin): their norms and overlaps carry G's rounding over f(1 - f)
        BB0 = B.mH @ B
        BB0 = 0.5 * (BB0 + BB0.mH)
        b_gram_raw = float(torch.linalg.norm(BB0 - torch.eye(BB0.shape[0], dtype=CD, device=self.dev)))
        eB, VB = torch.linalg.eigh(BB0)
        eB_min = float(eB.min())
        if eB_min < 0.5:
            log("v6i: kappa %d: the partners' Gram has an eigenvalue %.3e (below 0.5): orthonormalized all the same; the ledger's partners are not independent there" % (
                c.ops.kappa, eB_min))
        B = B @ (VB @ (eB.clamp_min(1e-300).rsqrt().to(CD)[:, None] * VB.mH))
        del BB0, VB
        R2 = c.ops.R2
        rt = r_t if r_t is not None else 0.0
        w_in = ((B.abs() ** 2) * (R2 < rt).to(RD)[:, None]).sum(dim=0); w_mid = ((B.abs() ** 2) * ((R2 >= rt) & (R2 <= self.R_cut_abs)).to(RD)[:, None]).sum(dim=0)
        w_out = ((B.abs() ** 2) * (R2 > self.R_cut_abs).to(RD)[:, None]).sum(dim=0)
        BB = B.mH @ B
        b_gram = float(torch.linalg.norm(BB - torch.eye(BB.shape[0], dtype=CD, device=self.dev)))
        del BB
        return {"B": B, "W_R": torch.zeros((B.shape[1], B.shape[1]), dtype=CD, device=self.dev), "f": [float(x) for x in f[order]],
                "f_one_minus_f": [float(x) for x in ff[order]], "sea_gram_F": sea_gram, "archive_gram_F": arch_gram, "B_gram_F_at_start": b_gram,
                "B_gram_F_before_lowdin": b_gram_raw, "B_gram_min_eig_before_lowdin": eB_min,
                "construction": "v6i: C_q = Phi (Phi^dagger Phi)^{-1} Phi^dagger, F_q = F (F^dagger F)^{-1/2}, B (B^dagger B)^{-1/2}",
                "t0": float(t0), "norms": [float(x) for x in torch.linalg.norm(B, dim=0)], "r_t": r_t, "archive_at_start": int(F.shape[1]),
                "w_inside_rt": [float(x) for x in w_in], "w_rt_to_cut": [float(x) for x in w_mid], "w_beyond_cut": [float(x) for x in w_out]}

    def start_transfer(self, c, Phi, F, r_t, t0=None):
        """v6f's ledger (at --transfer-at-rs, or restarted at each new S_esc peak with --transfer-at-peak): _partners at the clock t0, then B is
        carried with the sea (v6g (4): inside step_record) and W_R accumulates (Q^dagger B)^dagger (Q^dagger B) over the later blocks."""
        t0 = float(t0) if t0 is not None else float(self.H.T[self.tr_k])
        d = self._partners(c, Phi, F, r_t, t0)
        if d is None:
            log("kappa %d: no partner mode (every f(1 - f) of the collected sector at or below 1e-12): no ledger for this channel from T/r_s %.3f" % (
                c.ops.kappa, t0 / self.rs_nom))
            return
        if self.transfer is None:
            self.transfer = {}
        self.transfer[str(c.ops.kappa)] = d
        log("kappa %d: the transfer ledger starts at T/r_s %.3f with %d partner modes (f(1 - f) %s; their weights inside r_t %s, r_t to R_cut %s, beyond %s; "
            "v6h: ||Phi^dagger Phi - I||_F %.2e, ||F^dagger F - I||_F %.2e; v6i: ||B^dagger B - I||_F %.2e before the orthonormalization, %.2e after)" % (
            c.ops.kappa, d["t0"] / self.rs_nom, d["B"].shape[1], ["%.2e" % x for x in d["f_one_minus_f"]], ["%.3f" % x for x in d["w_inside_rt"]],
            ["%.3f" % x for x in d["w_rt_to_cut"]], ["%.3f" % x for x in d["w_beyond_cut"]], d["sea_gram_F"], d["archive_gram_F"], d["B_gram_F_before_lowdin"],
            d["B_gram_F_at_start"]))

    def start_anchor(self, k):
        """v6g (1): the ledger anchored at record k's clock (--ledger-anchor-rs), on the archive collected by then, whether or not record k
        is a checkpoint; carried with the sea and accumulated at every later collection like v6f's."""
        N, v = self.H.fields(k)
        t0 = float(self.H.T[k])
        self.anchor = {}
        for c in self.chs:
            if c.F is None or not c.F.shape[1]:
                log("v6g: kappa %d: the archive is empty at the anchor T/r_s %.4f: no anchored ledger for this channel" % (c.ops.kappa, t0 / self.rs_nom))
                continue
            r_t = c.outer_root(N, v)
            d = self._partners(c, c.Phi, c.F, r_t, t0)
            if d is None:
                log("v6g: kappa %d: no partner mode at the anchor T/r_s %.4f (every f(1 - f) at or below 1e-12): no anchored ledger for this channel" % (
                    c.ops.kappa, t0 / self.rs_nom))
                continue
            self.anchor[str(c.ops.kappa)] = d
            log("v6g: kappa %d: the ledger anchored at T/r_s %.4f (record %d) with %d partner modes of the archive's %d (f(1 - f) %s; their weights inside r_t %s, "
                "r_t to R_cut %s, beyond %s; r_t %s; v6h: ||Phi^dagger Phi - I||_F %.2e, ||F^dagger F - I||_F %.2e; v6i: ||B^dagger B - I||_F %.2e before the "
                "orthonormalization, %.2e after)" % (
                    c.ops.kappa, t0 / self.rs_nom, k, d["B"].shape[1], d["archive_at_start"], ["%.2e" % x for x in d["f_one_minus_f"]],
                    ["%.3f" % x for x in d["w_inside_rt"]], ["%.3f" % x for x in d["w_rt_to_cut"]], ["%.3f" % x for x in d["w_beyond_cut"]], _f(r_t, "%.3f"),
                    d["sea_gram_F"], d["archive_gram_F"], d["B_gram_F_before_lowdin"], d["B_gram_F_at_start"]))
        self.anchor_started = True

    def ledger_snapshot(self, led, row):
        """v6g (2): a ledger at this checkpoint (after its collection): per channel T_K = (1 - F F^dagger) B, T_K^dagger T_K, the accumulated
        W_R, W_H = I_p - T_K^dagger T_K - W_R (the identity's residual), the accumulation's check |B^dagger F F^dagger B - W_R|_F, the partners'
        Gram |B^dagger B - I|_F, and T_K's weight (summed over the p modes) inside r_t, from r_t to R_cut and beyond R_cut."""
        if not led:
            return None
        out = {}
        for c in self.chs:
            tr = led.get(str(c.ops.kappa))
            if tr is None:
                continue
            B = tr["B"]; F = c.F; p = int(B.shape[1])
            I_ = torch.eye(p, dtype=CD, device=self.dev)
            if F is not None and F.shape[1]:
                FB = F.mH @ B
                TK = B - F @ FB
                direct = FB.mH @ FB
                del FB
            else:
                TK = B; direct = torch.zeros((p, p), dtype=CD, device=self.dev)
            TT = TK.mH @ TK
            WR = tr["W_R"]
            WH = I_ - TT - WR
            BB = B.mH @ B
            pc = (row.get("per_channel") or {}).get(str(c.ops.kappa)) or {}
            rt = pc.get("r_t") or 0.0
            R2 = c.ops.R2
            wT = TK.abs() ** 2
            out[str(c.ops.kappa)] = {
                "t0": tr["t0"], "t0_over_rs": tr["t0"] / self.rs_nom, "p": p, "r_t": pc.get("r_t"),
                "trace_W_R_over_p": float(torch.trace(WR).real) / p, "trace_TK_TK_over_p": float(torch.trace(TT).real) / p,
                "diag_W_R": [float(x) for x in torch.diagonal(WR).real], "diag_TK_TK": [float(x) for x in torch.diagonal(TT).real],
                "diag_W_H": [float(x) for x in torch.diagonal(WH).real], "W_H_F": float(torch.linalg.norm(WH)),
                "W_R_check_F": float(torch.linalg.norm(direct - WR)), "B_gram_F": float(torch.linalg.norm(BB - I_)),
                "TK_weight_inside_rt": float((wT * (R2 < rt).to(RD)[:, None]).sum()),
                "TK_weight_rt_to_cut": float((wT * ((R2 >= rt) & (R2 <= self.R_cut_abs)).to(RD)[:, None]).sum()),
                "TK_weight_beyond_cut": float((wT * (R2 > self.R_cut_abs).to(RD)[:, None]).sum())}
            del TK, TT, WH, BB, direct, wT
        return out or None

    def ledger_log(self, row):
        for key, name in (("ledger_anchored", "the ledger anchored"), ("ledger_peak", "the peak-anchored ledger")):
            d = row.get(key)
            if not d:
                continue
            log("v6g: T/r_s %.3f: %s: %s" % (row["t_over_rs"], name, "; ".join(
                "kappa %s from T/r_s %.4f (p %d): tr W_R/p %.4e, tr T_K^dagger T_K/p %.6f, ||W_H||_F %.2e, the accumulation's check %.2e, ||B^dagger B - I||_F %.2e; "
                "T_K's weight (summed over the p modes) inside r_t %.4f, r_t to R_cut %.4f, beyond %.4f" % (
                    k_, v_["t0_over_rs"], v_["p"], v_["trace_W_R_over_p"], v_["trace_TK_TK_over_p"], v_["W_H_F"], v_["W_R_check_F"], v_["B_gram_F"],
                    v_["TK_weight_inside_rt"], v_["TK_weight_rt_to_cut"], v_["TK_weight_beyond_cut"]) for k_, v_ in sorted(d.items()))))

    SPLIT_KEYS = ("E_flux_plus_out", "E_flux_plus_in", "E_flux_minus_out", "E_flux_minus_in", "E_flux_out", "E_flux_in", "E_flux_net",
                  "n_flux_out", "n_flux_in", "n_flux_net", "E_density", "n_density_plus", "n_density_minus")

    def collar_split(self, c, Phi, Vneg, N, v, t):
        """v6k (v): the windowed signed split of the collar flux at the checkpoint (engine_massive_channel_D_v5b.signed_energy's local
        reading on one sine window centred at the first site beyond R_collar), per window size; the largest window's C and P_- blocks kept
        for the checkpoint's block file."""
        R = np.asarray(c.ops.R, float); n = len(R)
        Nf = np.asarray(N, float); vf = np.asarray(v, float)
        ic = int(np.searchsorted(R, self.R_collar_abs, side="right"))              # the first site beyond the collar (0-based)
        Rc, Nc, vc = float(R[ic]), float(Nf[ic]), float(vf[ic])
        kap = float(c.ops.kappa)
        wins = [int(x) for x in str(self.a.collar_split_windows).split(",") if x.strip()]
        Lmax = max(wins)
        out = {"centre_site": ic + 1, "R_c": Rc, "N_c": Nc, "v_c": vc, "windows": {}}
        for L in wins:
            hL = L // 2
            i0 = ic - hL; i1 = i0 + L
            lo, hi = max(0, i0), min(n, i1)
            gw = np.sin(math.pi * (np.arange(L) + 0.5) / L)[lo - i0:hi - i0]       # the sine window on the valid sites
            rows = torch.arange(2 * lo, 2 * hi, device=Phi.device)
            PW = Phi.index_select(0, rows); VW = Vneg.index_select(0, rows)
            CW = (PW @ PW.mH).cpu().numpy().astype(np.complex128)
            PmW = (VW @ VW.mH).cpu().numpy().astype(np.complex128)
            del PW, VW
            Dl = CW - PmW
            Dl = 0.5 * (Dl + Dl.conj().T)
            M = 4 * L
            q = 2.0 * math.pi * np.fft.fftfreq(M)
            nl = hi - lo
            Fm = np.fft.fft(np.eye(M, nl) * gw[None, :], axis=0, norm="ortho")        # M x nl: column i the window's DFT of site i
            G = {}
            for aa in (0, 1):
                for bb in (0, 1):
                    G[(aa, bb)] = np.sum((Fm @ Dl[aa::2, bb::2]) * Fm.conj(), axis=1)   # (F Delta_ab F^dagger)(q, q)
            s_ = np.sin(q); c_ = np.cos(q)
            kr = kap / Rc
            E = np.sqrt(s_ ** 2 + kr ** 2)
            nrm = np.sqrt(2.0 * E * E)
            u0 = E / nrm; u1 = (s_ + 1j * kr) / nrm
            n_tot = (G[(0, 0)] + G[(1, 1)]).real
            n_p = (np.conj(u0) * u0 * G[(0, 0)] + np.conj(u0) * u1 * G[(0, 1)] + np.conj(u1) * u0 * G[(1, 0)] + np.conj(u1) * u1 * G[(1, 1)]).real
            n_m = n_tot - n_p
            om_p = Nc * E + 0.5 * vc * np.sin(2.0 * q); om_m = -Nc * E + 0.5 * vc * np.sin(2.0 * q)
            vp = Nc * s_ * c_ / E + vc * np.cos(2.0 * q); vm = -Nc * s_ * c_ / E + vc * np.cos(2.0 * q)
            W2 = float(np.sum(gw ** 2))
            po = vp > 0; mo = vm > 0
            w_ = {"L": L, "sites_valid": int(nl), "sum_w2": W2,
                  "E_flux_plus_out": float(np.sum((om_p * vp * n_p)[po])) / W2, "E_flux_plus_in": float(np.sum((om_p * vp * n_p)[~po])) / W2,
                  "E_flux_minus_out": float(np.sum((om_m * vm * n_m)[mo])) / W2, "E_flux_minus_in": float(np.sum((om_m * vm * n_m)[~mo])) / W2,
                  "n_flux_out": (float(np.sum((vp * n_p)[po])) + float(np.sum((vm * n_m)[mo]))) / W2,
                  "n_flux_in": (float(np.sum((vp * n_p)[~po])) + float(np.sum((vm * n_m)[~mo]))) / W2,
                  "E_density": (float(np.sum(om_p * n_p)) + float(np.sum(om_m * n_m))) / W2,
                  "n_density_plus": float(np.sum(n_p)) / W2, "n_density_minus": float(np.sum(n_m)) / W2,
                  }
            w_["E_flux_out"] = w_["E_flux_plus_out"] + w_["E_flux_minus_out"]
            w_["E_flux_in"] = w_["E_flux_plus_in"] + w_["E_flux_minus_in"]
            w_["E_flux_net"] = w_["E_flux_out"] + w_["E_flux_in"]
            w_["n_flux_net"] = w_["n_flux_out"] + w_["n_flux_in"]
            w_["in_over_out"] = (abs(w_["E_flux_in"]) / abs(w_["E_flux_out"])) if w_["E_flux_out"] != 0 else None
            out["windows"][str(L)] = w_
            if L == Lmax:
                self._cs_blocks["k%d_C" % c.ops.kappa] = CW; self._cs_blocks["k%d_Pm" % c.ops.kappa] = PmW
                self._cs_blocks["k%d_sites" % c.ops.kappa] = np.arange(lo + 1, hi + 1)
                self._cs_blocks["R"] = R[lo:hi]; self._cs_blocks["N"] = Nf[lo:hi]; self._cs_blocks["v"] = vf[lo:hi]
        return out

    def finish_row(self, row, t_ck):
        a = self.a
        if a.collar_split:                                                      # v6k (v): weighted over the channels; the block file
            wins = [str(int(x)) for x in str(a.collar_split_windows).split(",") if x.strip()]
            agg = {}
            for L in wins:
                agg[L] = {key: sum(c.ops.w * row["per_channel"][str(c.ops.kappa)]["collar_split"]["windows"][L][key] for c in self.chs) for key in self.SPLIT_KEYS}
                agg[L]["in_over_out"] = (abs(agg[L]["E_flux_in"]) / abs(agg[L]["E_flux_out"])) if agg[L]["E_flux_out"] != 0 else None
            row["collar_split_weighted"] = agg
            row["collar_split_label"] = ("v6k (v): the collar flux's outgoing and incoming parts by the sign of the local group velocity in a sine window "
                                         "centred at the first site beyond R_collar (engine_massive_channel_D_v5b's windowed reading of C - P_-), "
                                         "per window size; a local reading beside the exact net current L_at_t, not an identity")
            bdir = self.prefix + "_collar_blocks"
            os.makedirs(bdir, exist_ok=True)
            blk = dict(self._cs_blocks); blk["t"] = np.array(row["t"]); blk["k"] = np.array(row["k"])
            blk["kappas"] = np.array([c.ops.kappa for c in self.chs]); blk["weights"] = np.array([c.ops.w for c in self.chs])
            tmp = os.path.join(bdir, ".tmp_%d.npz" % os.getpid())
            np.savez(tmp, **blk)
            os.replace(tmp, os.path.join(bdir, "ck_T%010.4f.npz" % (row["t"] / self.rs_nom)))
            self._cs_blocks = {}
        wsum = lambda key: sum(c.ops.w * (row["per_channel"][str(c.ops.kappa)].get(key) or 0.0) for c in self.chs)
        for key in ("E_inv", "E_out", "E_inv_inside", "W_hdot", "S_esc", "S_B", "S_M_minus_S_Mprime", "E_K", "E_F", "E_cross", "E_remove", "M_chi", "S_vac_archive", "S_esc_minus_S_vac",
                    "energy_identity_residual", "collar_identity_residual", "collar_flow_acc", "dropped_pair_ledger",
                    "work_jumps_since_last", "W_acc_interval", "jump_check", "E_hC_change"):
            row[key + "_weighted"] = wsum(key)
        rs_k = row["two_gE"]; TH = row["T_H"]
        kT = row.get("kappa_T")
        row["L_H_kappaT"] = continuum_comparator(self.channels, rs_k, kT / (2.0 * math.pi)) if kT else None
        row["L_H_TH"] = continuum_comparator(self.channels, rs_k, TH)
        recs_since = [r for r in self.records if self.last_ck is None or (self.last_ck["t"] <= r["t"] < row["t"])]
        if self.last_ck is not None and row["t"] > self.last_ck["t"]:
            dt = row["t"] - self.last_ck["t"]
            flow = sum(r["collar_flow_weighted"] for r in recs_since)
            row["L_interval"] = flow / dt
            row["W_creation"] = (row["E_inv_inside_weighted"] - self.last_ck["E_inv_inside_weighted"]) / dt + row["L_interval"]
            row["W_hdot_trapezoid"] = 0.5 * (row["W_hdot_weighted"] + self.last_ck["W_hdot_weighted"])
            row["W_acc_rate"] = row["W_acc_interval_weighted"] / dt                # v6f (1): the work accumulated per record
            log("v6f: T/r_s %.3f: W (finite difference) %s, W_acc (accumulated per record) %.4e, the checkpoints' trapezoid %.4e; the jump check %.3e (weighted)" % (
                row["t_over_rs"], _f(row.get("W_creation"), "%.4e"), row["W_acc_rate"], row["W_hdot_trapezoid"], row["jump_check_weighted"]))
            row["dE_inv_dt"] = (row["E_inv_weighted"] - self.last_ck["E_inv_weighted"]) / dt
        row["L_at_t"] = recs_since[-1]["L_weighted"] if recs_since else None
        if row.get("L_interval") is not None:
            row["L_over_LH_kappaT"] = row["L_interval"] / row["L_H_kappaT"] if row.get("L_H_kappaT") else None
            row["L_over_LH_TH"] = row["L_interval"] / row["L_H_TH"] if row.get("L_H_TH") else None
        row["t_inf_collar"] = OPS.chart_retarded_time(row["t"], rs_k, self.R_collar_abs) - OPS.chart_retarded_time(self.H.T[self.k0], 2 * self.g * self.H.E[self.k0], self.R_collar_abs)
        A = math.pi * rs_k * rs_k
        row["levels"] = {"eta_A_over_a2": 4.0 * a.eta * A, "A_over_4G": A / self.g}
        row["band"] = sum(c.ops.w * (c.cheb_tail_acc + c.unit_defect) for c in self.chs)
        row["seconds"] = time.time() - t_ck
        self.rows.append(row); self.last_ck = row; self.n_ck += 1
        log("T/r_s %.3f (k %d; 2gE %.4f, T_H %.4e; join dN %s dv/v %s; min(N-|v|) %.4f; r_t %s): E_inv %.6e (inside the collar %.6e, beyond %.6e), W %s against "
            "Tr[h_dot (C - P)] %s (trapezoid %s); L %s (L_H at kappa_T %s, at T_H %s; ratios %s, %s); S_esc %.6f (band %.2e; the archive's vacuum entropy %.4e, the excess %.6f) S_B %s, S(M) - S(M') %s (the archive's entropy before minus after the batch: minus the conditional entropy S(B|M)); collected %s (archive %s; "
            "outside weight min %s; incoming fraction %s); E_remove %.6e, M_chi %.6e, the identities %.2e, %.2e; collar flow %.6e; %.0f s" % (
                row["t_over_rs"], row["k"], rs_k, TH, _f(row["join_max_dN"], "%.2e"), _f(row["join_max_rel_dv"], "%.2e"), row["min_N_minus_absv"],
                ", ".join("%s" % _f(row["per_channel"][str(c.ops.kappa)].get("r_t"), "%.3f") for c in self.chs),
                row["E_inv_weighted"], row["E_inv_inside_weighted"], row["E_out_weighted"], _f(row.get("W_creation"), "%.4e"), _f(row["W_hdot_weighted"], "%.4e"),
                _f(row.get("W_hdot_trapezoid"), "%.4e"), _f(row.get("L_interval"), "%.4e"), _f(row["L_H_kappaT"], "%.4e"), _f(row["L_H_TH"], "%.4e"),
                _f(row.get("L_over_LH_kappaT"), "%.3f"), _f(row.get("L_over_LH_TH"), "%.3f"), row["S_esc_weighted"], row["band"], row["S_vac_archive_weighted"],
                row["S_esc_minus_S_vac_weighted"], _f(row.get("S_B_weighted"), "%.4e"),
                _f(row.get("S_M_minus_S_Mprime_weighted"), "%.4e"), ", ".join("%s" % row["per_channel"][str(c.ops.kappa)].get("collected") for c in self.chs),
                ", ".join("%s" % row["per_channel"][str(c.ops.kappa)].get("archive") for c in self.chs),
                ", ".join(_f(row["per_channel"][str(c.ops.kappa)].get("outside_weight_min"), "%.5f") for c in self.chs),
                ", ".join(_f(row["per_channel"][str(c.ops.kappa)].get("incoming_fraction"), "%.3f") for c in self.chs),
                row["E_remove_weighted"], row["M_chi_weighted"], row["energy_identity_residual_weighted"], row["collar_identity_residual_weighted"],
                row["collar_flow_acc_weighted"], row["seconds"]))

    # ----------------------------------------------------------------------------------------------------------- the loop
    def propagate_transfer(self, k, dt):
        """v6g (4): folded into step_record (the ledgers' partners take the sea's own sub-steps there); kept as a no-op."""
        return

    def run(self, resumed=False):
        a = self.a
        if not resumed:
            if a.init_state:
                self.load_init_state(a.init_state)
            else:
                self.start()
        done = {r_["k"] for r_ in self.rows if not r_.get("frozen")}
        k = self.k
        while not self.frozen:
            if self.H.follow and k < self.kend:                              # v6g (3): the next record is in before record k is used
                try:
                    have = self.H.ensure(k + 1, on_wait=self._release_cache)
                except TimeoutError as ex:
                    log("v6g: %s: the state saved at record %d; resumable with --resume" % (ex, k))
                    self.save_state(k); self.write_outputs(final=False)
                    return
                if not have:
                    self.kend = len(self.H) - 1; self.ck.add(self.kend)
                    self.follow_end = "the continuation's done marker: the history ends at record %d (T/r_s %.4f)" % (self.kend, self.H.T[self.kend] / self.rs_nom)
                    log("v6g: %s" % self.follow_end)
            if k in self.ck and k not in done:
                self.checkpoint(k); done.add(k)
                if self.n_ck % max(1, a.save_every) == 0:
                    self.save_state(k)
            if k in self.save_at_k:
                self.save_state(k, path="%s_state_at_%07.3f.pt" % (self.prefix, self.H.T[k] / self.rs_nom))
            if self.k_anchor is not None and k == self.k_anchor and not self.anchor_started:     # v6g (1)
                self.start_anchor(k)
            if self.kf is not None and k >= self.kf:
                self.frozen = True
                break
            if k >= self.kend:
                break
            self.step_record(k)
            k += 1; self.k = k
            if (time.time() - T_START) / 3600.0 > a.max_wall_hours:
                log("the wall cap: the state saved at record %d" % k)
                self.save_state(k); self.write_outputs(final=False)
                return
        if self.frozen:
            self.run_frozen()
        self.finish()

    def run_frozen(self):
        """C (#36): h frozen at the record kf from the clock T_kf on for --freeze-duration-rs, checkpoints and collections every
        --collect-every-rs with the one cached eigendecomposition and the phase-rotated memory; S1's drain rule: |L| below --end-flux-frac
        of the comparator (L_H at kappa_T of the frozen record, else at T_H) for --end-window consecutive blocks after S_esc has fallen below
        --end-entropy-frac of its peak; else the duration."""
        a, H = self.a, self.H
        kf = self.kf
        every = max(1, int(round(a.collect_every_rs * self.rs_nom / self.tau)))
        nsteps = int(round(a.freeze_duration_rs * self.rs_nom / self.tau))
        t = float(H.T[kf])
        log("C: h frozen at the record T/r_s %.4f (k %d) for %.1f r_s (%d steps of tau %g), collection every %d steps; propagation %s" % (
            t / self.rs_nom, kf, a.freeze_duration_rs, nsteps, self.tau, every, a.frozen_prop))
        self.t_freeze = t
        self.refl = self.reflection_clock(kf)
        if self.refl is not None:
            log("C: the chain end's reflection clock (v6c (4)): %.1f r_s after the freeze (out %.1f r_s, in %.1f r_s, R_collar %.1f r_s to the chain end %.1f r_s; |v|/N at R_collar %.4f)" % (
                self.refl["T_refl_rs"], self.refl["t_out_rs"], self.refl["t_in_rs"], self.refl["R_from_rs"], self.refl["R_end_rs"], self.refl["s_at_R_collar"]))
        self.checkpoint(kf, t=t, collect_override=False)
        self.rows[-1]["frozen"] = True
        self.rows[-1]["after_reflection_clock"] = False
        eig = (a.frozen_prop == "eig")
        if eig:
            self.frozen_eig_init(t)
        t_last = t
        for j in range(1, nsteps + 1):
            t += self.tau
            if eig:
                if j % every == 0 or j == nsteps:
                    self.frozen_materialize(kf, t_last, t); t_last = t
            else:
                self.step_record(kf, t0=t - self.tau, t1=t)
            self.t_now = t
            if j % every == 0:
                n_before = {str(c.ops.kappa): int(c.F.shape[1]) for c in self.chs}
                epoch = self.transfer_epoch
                self.checkpoint(kf, t=t, collect_override=True)
                if eig:
                    self.frozen_eig_update(t, n_before, epoch)
                row = self.rows[-1]; row["frozen"] = True
                row["after_reflection_clock"] = bool(self.refl is not None and (t - self.t_freeze) > self.refl["T_refl"])
                S = row["S_esc_weighted"]; self.S_peak = max(self.S_peak, S)
                comp = row.get("L_H_kappaT") or row.get("L_H_TH")
                L = row.get("L_interval")
                armed = self.S_peak > 0 and S < a.end_entropy_frac * self.S_peak
                small = (L is not None and comp is not None and abs(L) < a.end_flux_frac * comp)
                self.end_count = (self.end_count + 1) if (armed and small) else 0
                row["end_rule"] = {"armed": bool(armed), "small_flux": bool(small), "count": self.end_count, "S_peak": self.S_peak}
                if self.end_count >= a.end_window:
                    self.end_reason = "drained (S1's rule: %d blocks with |L| below %g of the comparator after S_esc below %g of its peak)" % (a.end_window, a.end_flux_frac, a.end_entropy_frac)
                    break
            if (time.time() - T_START) / 3600.0 > a.max_wall_hours:
                self.end_reason = "the wall cap"
                if eig and t_last != t:
                    self.frozen_materialize(kf, t_last, t); t_last = t
                break
        if eig and t_last != t:
            self.frozen_materialize(kf, t_last, t); t_last = t
        if self.end_reason is None:
            self.end_reason = "the duration (%.1f r_s) reached without the drain" % a.freeze_duration_rs
        log("C: the frozen control ends: %s" % self.end_reason)

    def reflection_clock(self, k):
        """v6c (4): the continuum light round trip R_collar -> the chain end -> R_collar in record k's site fields (trapezoid over the sites)."""
        N, v = self.H.fields(k)
        N = np.asarray(N, float); v = np.asarray(v, float); R = np.asarray(self.H.R, float)
        sel = R >= self.R_collar_abs
        if sel.sum() < 2:
            return None
        cp = N[sel] - np.abs(v[sel]); cm = N[sel] + np.abs(v[sel]); Rs = R[sel]
        if np.any(cp <= 0):
            return None
        trap = lambda f: float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(Rs)))
        t_out = trap(1.0 / cp); t_in = trap(1.0 / cm)
        return {"T_refl": t_out + t_in, "T_refl_rs": (t_out + t_in) / self.rs_nom, "t_out_rs": t_out / self.rs_nom, "t_in_rs": t_in / self.rs_nom,
                "R_from_rs": float(Rs[0]) / self.rs_nom, "R_end_rs": float(Rs[-1]) / self.rs_nom, "s_at_R_collar": float(abs(v[sel][0]) / N[sel][0]),
                "label": "the continuum light round trip R_collar -> the chain end -> R_collar in the frozen record's fields (outgoing N - |v|, ingoing N + |v|); "
                         "a frozen row after it may carry the chain end's reflection in its collar readouts"}

    def _frozen_ops(self, c, kf):
        N, v = self.H.fields(kf)
        h = c.ops.h_csr(N, v)
        return h, c.ops.banded(h), (c.ops.R2 > self.R_collar_abs).to(CD)[:, None]

    def frozen_eig_init(self, t):
        """v6c (3): the sea, the archive and the ledger's partner modes in the frozen h's eigenbasis, referred to the freeze clock fz t0."""
        for c in self.chs:
            fz = self.fz[str(c.ops.kappa)]; V = fz["V"]
            ph = torch.exp(1j * fz["e"].to(CD) * (t - fz["t0"]))[:, None]
            c.Phi_eig = ph * (V.mH @ c.Phi)
            c.F_eig = (ph * (V.mH @ c.F)) if c.F.shape[1] else torch.zeros((c.ops.n2, 0), dtype=CD, device=self.dev)
            h, Bh, chi = self._frozen_ops(c, self.kf)
            c.Eo_last = tr_blocked(Bh, c.Phi, chi)
            tr = (self.transfer or {}).get(str(c.ops.kappa))
            if tr is not None:
                tr["B_eig"] = ph * (V.mH @ tr["B"])
        log("C: the frozen eigenbasis carries the sea (%s columns) and the archives (%s modes) by exact phases from the freeze clock" % (
            ", ".join(str(int(c.Phi.shape[1])) for c in self.chs), ", ".join(str(int(c.F.shape[1])) for c in self.chs)))

    def frozen_materialize(self, kf, ta, tb):
        """v6c (3): Phi, F and the ledger's B at tb from the eigenbasis; the collar energy flow over [ta, tb] (Re tr[C chi h] at tb minus at ta,
        the per-record flows' telescoped sum on a frozen h) and the collar current at tb, as one interval record."""
        rec = {"k": kf, "t": ta, "t_end": tb, "frozen_interval": True, "per_channel": {}}
        for c in self.chs:
            fz = self.fz[str(c.ops.kappa)]; V = fz["V"]
            ph = torch.exp(-1j * fz["e"].to(CD) * (tb - fz["t0"]))[:, None]
            c.Phi = V @ (ph * c.Phi_eig)
            if c.F_eig.shape[1]:
                c.F = V @ (ph * c.F_eig)
            tr = (self.transfer or {}).get(str(c.ops.kappa))
            if tr is not None and "B_eig" in tr:
                tr["B"] = V @ (ph * tr["B_eig"])
            h, Bh, chi = self._frozen_ops(c, kf)
            Eo1 = tr_blocked(Bh, c.Phi, chi)
            J = OPS.Banded(OPS.current_out_csr(h, c.ops.R, self.R_collar_abs), self.dev)
            L1 = tr_blocked(J, c.Phi)
            del J
            flow = Eo1 - c.Eo_last; c.Eo_last = Eo1
            c.collar_flow_acc += flow
            rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L1, "L_clock": "the interval's end", "cheb_K": 0, "cheb_tail": 0.0}
        rec["collar_flow_weighted"] = sum(c.ops.w * rec["per_channel"][str(c.ops.kappa)]["collar_flow"] for c in self.chs)
        rec["L_weighted"] = sum(c.ops.w * rec["per_channel"][str(c.ops.kappa)]["L_at_t"] for c in self.chs)
        rec["sub_steps"] = 0
        self.records.append(rec)
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

    def frozen_eig_update(self, t, n_before, epoch):
        """v6c (3): after a frozen checkpoint, its newly collected modes (and a restarted ledger's B) enter the eigenbasis at the clock t."""
        for c in self.chs:
            fz = self.fz[str(c.ops.kappa)]; V = fz["V"]
            ph = torch.exp(1j * fz["e"].to(CD) * (t - fz["t0"]))[:, None]
            m0 = n_before[str(c.ops.kappa)]
            if int(c.F.shape[1]) > m0:
                c.F_eig = torch.cat([c.F_eig, ph * (V.mH @ c.F[:, m0:])], dim=1)
            tr = (self.transfer or {}).get(str(c.ops.kappa))
            if tr is not None and (self.transfer_epoch != epoch or "B_eig" not in tr):
                tr["B_eig"] = ph * (V.mH @ tr["B"])

    def load_init_state(self, path):
        """C: start from run A's saved state (the sea, the archive, the accumulators) at its record; the rows and records start empty."""
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.k = st["k"]; self.k0 = st["k"]; self.k_init = st["k"]
        for c in self.chs:
            s = st["channels"][str(c.ops.kappa)]
            c.Phi = s["Phi"].to(self.dev); c.F = s["F"].to(self.dev); c.F_t = list(s["F_t"]); c.F_g = list(s["F_g"])
            c.cheb_K_max = s["cheb_K_max"]; c.cheb_tail_acc = s["cheb_tail_acc"]; c.collar_flow_acc = s["collar_flow_acc"]
            c.gram_F2 = s["gram_F2"]; c.gram_max = s["gram_max"]; c.unit_defect = s["unit_defect"]
        self.min_N_minus_absv_T0 = st.get("min_N_minus_absv_T0")
        log("C: started from the saved state %s at record %d (T/r_s %.4f): archives %s" % (path, self.k, self.H.T[self.k] / self.rs_nom,
                                                                                            ", ".join("kappa %d %d modes" % (c.ops.kappa, c.F.shape[1]) for c in self.chs)))

    def save_state(self, k, path=None):
        st = {"k": k, "rows": self.rows, "records": self.records, "n_ck": self.n_ck, "last_ck": self.last_ck,
              "min_N_minus_absv_T0": getattr(self, "min_N_minus_absv_T0", None), "channels": {}}
        for c in self.chs:
            st["channels"][str(c.ops.kappa)] = {"Phi": c.Phi.cpu(), "F": c.F.cpu(), "F_t": c.F_t, "F_g": c.F_g, "cheb_K_max": c.cheb_K_max,
                                                 "cheb_tail_acc": c.cheb_tail_acc, "collar_flow_acc": c.collar_flow_acc, "gram_F2": c.gram_F2,
                                                 "gram_max": c.gram_max, "unit_defect": c.unit_defect,
                                                 "work_jump_since": c.work_jump_since, "E_hC_last": c.E_hC_last, "E_ref_last": c.E_ref_last}
        if self.transfer is not None:
            st["transfer"] = {k_: {kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv) for kk, vv in tr.items() if kk != "B_eig"} for k_, tr in self.transfer.items()}
        st["v6c"] = {"tr_peak_S": self.tr_peak_S, "transfer_starts": self.transfer_starts, "transfer_epoch": self.transfer_epoch}
        if self.anchor is not None:                                            # v6g (1)
            st["anchor"] = {k_: {kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv) for kk, vv in tr.items()} for k_, tr in self.anchor.items()}
        st["v6g"] = {"anchor_started": self.anchor_started, "follow_end": self.follow_end}
        dest = path or (self.prefix + "_state.pt")
        tmp = dest + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, dest)
        self.write_outputs(final=False)
        log("state saved at record %d (T/r_s %.3f)" % (k, self.H.T[min(k, len(self.H) - 1)] / self.rs_nom))

    def load_state(self):
        st = torch.load(self.prefix + "_state.pt", map_location="cpu", weights_only=False)
        self.k = st["k"]; self.rows = st["rows"]; self.records = st["records"]; self.n_ck = st["n_ck"]; self.last_ck = st["last_ck"]
        self.min_N_minus_absv_T0 = st.get("min_N_minus_absv_T0")
        for c in self.chs:
            s = st["channels"][str(c.ops.kappa)]
            c.Phi = s["Phi"].to(self.dev); c.F = s["F"].to(self.dev); c.F_t = s["F_t"]; c.F_g = s["F_g"]
            c.cheb_K_max = s["cheb_K_max"]; c.cheb_tail_acc = s["cheb_tail_acc"]; c.collar_flow_acc = s["collar_flow_acc"]
            c.gram_F2 = s["gram_F2"]; c.gram_max = s["gram_max"]; c.unit_defect = s["unit_defect"]
            c.work_jump_since = s.get("work_jump_since", 0.0); c.E_hC_last = s.get("E_hC_last"); c.E_ref_last = s.get("E_ref_last")   # v6g: kept on resume
        if st.get("transfer"):
            self.transfer = {k_: {kk: (vv.to(self.dev) if isinstance(vv, torch.Tensor) else vv) for kk, vv in tr.items()} for k_, tr in st["transfer"].items()}
        v6c = st.get("v6c") or {}
        self.tr_peak_S = v6c.get("tr_peak_S", 0.0); self.transfer_starts = v6c.get("transfer_starts", []); self.transfer_epoch = v6c.get("transfer_epoch", 0)
        if st.get("anchor") is not None:                                      # v6g (1)
            self.anchor = {k_: {kk: (vv.to(self.dev) if isinstance(vv, torch.Tensor) else vv) for kk, vv in tr.items()} for k_, tr in st["anchor"].items()}
        v6g = st.get("v6g") or {}
        self.anchor_started = bool(v6g.get("anchor_started", False)); self.follow_end = v6g.get("follow_end")
        log("resumed at record %d (T/r_s %.3f), %d checkpoints done" % (self.k, self.H.T[self.k] / self.rs_nom, len(self.rows)))

    def finish(self):
        a = self.a
        if self.transfer is not None:
            for c in self.chs:
                tr = self.transfer.get(str(c.ops.kappa))
                if tr is None:
                    continue
                B = tr["B"]; F = c.F
                TK = B - F @ (F.mH @ B)
                TT = TK.mH @ TK
                WR = tr["W_R"]
                p = B.shape[1]
                I_ = torch.eye(p, dtype=CD, device=self.dev)
                WH = I_ - TT - WR
                tr["end"] = {"t_end": float(self.t_now) if (self.frozen and self.t_now is not None) else float(self.H.T[self.k]), "diag_TK_TK": [float(x) for x in torch.diagonal(TT).real], "diag_W_R": [float(x) for x in torch.diagonal(WR).real],
                             "diag_W_H": [float(x) for x in torch.diagonal(WH).real], "trace_W_R_over_p": float(torch.trace(WR).real) / p,
                             "identity_residual_F": float(torch.linalg.norm(WH)), "label": "T_K^dagger T_K + W_R + W_H = I_p; W_H is the residual of the identity here"}
                log("kappa %d: the transfer ledger at the end: tr W_R/p %.4f, diag W_R %s, diag T_K^dagger T_K %s, ||W_H||_F %.3e" % (
                    c.ops.kappa, tr["end"]["trace_W_R_over_p"], ["%.3f" % x for x in tr["end"]["diag_W_R"]], ["%.3f" % x for x in tr["end"]["diag_TK_TK"]],
                    tr["end"]["identity_residual_F"]))
        if self.anchor and self.rows and self.rows[-1].get("ledger_anchored"):           # v6g (1): the last checkpoint's snapshot
            for k_, d in self.rows[-1]["ledger_anchored"].items():
                if k_ in self.anchor:
                    self.anchor[k_]["end"] = dict(d, t_end=float(self.rows[-1]["t"]))
            log("v6g: the ledger anchored at T/r_s %s at the end (T/r_s %.3f): %s" % (
                ", ".join(sorted({"%.4f" % d["t0_over_rs"] for d in self.rows[-1]["ledger_anchored"].values()})), self.rows[-1]["t_over_rs"],
                "; ".join("kappa %s: tr W_R/p %.4e, tr T_K^dagger T_K/p %.6f, ||W_H||_F %.2e" % (k_, d["trace_W_R_over_p"], d["trace_TK_TK_over_p"], d["W_H_F"])
                          for k_, d in sorted(self.rows[-1]["ledger_anchored"].items()))))
        last = self.rows[-1] if self.rows else {}
        E_rad0 = last.get("collar_flow_acc_weighted")
        self.label = LABEL_A % (("%.4e (the collar flow over E at the end)" % (E_rad0 / last["E"])) if (E_rad0 is not None and last.get("E")) else "-")
        if self.kf is not None:
            self.label += "; C: h frozen at T/r_s %.4f, %s" % (self.H.T[self.kf] / self.rs_nom, self.end_reason)
        if self.follow_end:
            self.label += "; " + self.follow_end
        self.write_outputs(final=True)
        log("done: %s" % self.label)

    def write_outputs(self, final):
        a = self.a
        rows = self.rows
        H = self.H
        anc = ({k_: {kk: vv for kk, vv in tr.items() if kk not in ("B", "W_R")} for k_, tr in self.anchor.items()} if self.anchor else None)
        extd = ({"record": a.record_ext, "fields": a.fields_ext, "after_rs": a.ext_after_rs, "follow": bool(a.follow), "n_primary": int(H.n_primary),
                 "n_extension_used": int(len(H) - H.n_primary), "seam": H.seam, "loads": H.loads, "md5_last": H.ext_md5, "label": H.ext_label,
                 "failed_reads": H.n_failed_reads, "follow_end": self.follow_end} if H.ext is not None else None)
        summary = {"code": "page_history_v6k.py", "settings": vars(a), "ledger_anchor": anc, "extension": extd,
                   "label_v6i": "the partners' Gram brought to the identity after their construction on C_q and F_q (Lowdin)",
                   "label_v6h": "both ledgers' partners built on C_q = Phi (Phi^dagger Phi)^{-1} Phi^dagger and F_q = F (F^dagger F)^{-1/2}; the sea and the archive unchanged",
                   "label_v6g": ("the transfer ledger anchored at --ledger-anchor-rs beside the peak-anchored one, both at every checkpoint after the collection "
                                 "(rows ledger_anchored, ledger_peak); the history's continuation after --ext-after-rs, read as its records land with --follow"), "final": final, "label": self.label or (LABEL_A % "-"),
                   "transfer_starts": self.transfer_starts, "reflection_clock": self.refl, "frozen_prop": a.frozen_prop if self.kf is not None else None,
                   "label_v6c": "the greybody table cached (the numbers unchanged); --transfer-at-peak restarts the ledger at each new S_esc maximum; C's propagation in the frozen h's eigenbasis by exact phases (one eigendecomposition), the collar flow per checkpoint interval and the collar current at the checkpoint clock; the reflection clock printed",
                   "label_d15": "no mutual information computed or printed (#41 D15); S(M) - S(M') = S_old - S_new is minus the conditional entropy S(B|M) of each collected batch given the archive", "history": {
            "label": self.H.label, "records": len(self.H), "tau": self.tau, "ell": self.H.ell, "g": self.g, "rs_nominal": self.rs_nom, "engine_sites": self.H.ne,
            "T_start": float(self.H.T[self.k0]), "T_end": self._T_of_k(self.kend), "min_N_minus_absv_T0": getattr(self, "min_N_minus_absv_T0", None)},
            "rows": rows, "records": self.records, "end_reason": self.end_reason, "transfer": ({k_: {kk: vv for kk, vv in tr.items() if kk not in ("B", "W_R", "B_eig")} for k_, tr in self.transfer.items()} if self.transfer else None),
            "archive": {str(c.ops.kappa): {"modes": int(c.F.shape[1]) if c.F is not None else 0, "t": c.F_t, "outside_weight": c.F_g, "gram_all_history_max": c.gram_max,
                                           "sea_unitarity_max": c.unit_defect, "cheb_K_max": c.cheb_K_max, "cheb_tail_acc": c.cheb_tail_acc} for c in self.chs},
            "wall_s": time.time() - T_START}
        with open(self.prefix + "_summary.json", "w") as f:
            json.dump(summary, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o), indent=1)
        L = ["# %s: Computation A (%s)\n" % (a.tag, "final" if final else "running"), "Label: **%s**.\n" % summary["label"],
             "History: %s; %d records, tau %g, ell %s, g %g, r_s nominal %g, the engine's %d sites, the chain %d sites; channels %s (weights 16 kappa); R_cut %g r_s, collar %g r_s; eps_loc %g, eps_form %g, the pair ledger %s; collection every %g r_s.\n" % (
                 self.H.label, len(self.H), self.tau, self.H.ell, self.g, self.rs_nom, self.H.ne, self.H.nsites, self.channels, a.R_cut_rs, a.R_collar_rs, a.eps_loc, a.eps_form,
                 a.pair_ledger, a.collect_every_rs),
             "| T/r_s | 2gE | E_inv | inside the collar | beyond | W (creation) | W_acc (per record) | Tr[h_dot (C - P)] | L (interval) | L_H(kappa_T) | L_H(T_H) | L/L_H(kappa_T) | L/L_H(T_H) | S_esc | band | S_vac (archive) | S_esc - S_vac | S_B | S(M) - S(M') | collected | archive | outside weight min | incoming fraction | E_remove | M_chi | energy identity | collar identity | join dN | join dv/v |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            pcs = [r["per_channel"][str(c.ops.kappa)] for c in self.chs]
            L.append("| %.3f | %.4f | %.6e | %.6e | %.6e | %s | %s | %.4e | %s | %s | %s | %s | %s | %.6f | %.2e | %.4e | %.6f | %s | %s | %s | %s | %s | %s | %.6e | %.6e | %.2e | %.2e | %s | %s |" % (
                r["t_over_rs"], r["two_gE"], r["E_inv_weighted"], r["E_inv_inside_weighted"], r["E_out_weighted"], _f(r.get("W_creation"), "%.4e"), _f(r.get("W_acc_rate"), "%.4e"), r["W_hdot_weighted"],
                _f(r.get("L_interval"), "%.4e"), _f(r.get("L_H_kappaT"), "%.4e"), _f(r.get("L_H_TH"), "%.4e"), _f(r.get("L_over_LH_kappaT"), "%.3f"), _f(r.get("L_over_LH_TH"), "%.3f"),
                r["S_esc_weighted"], r["band"], r["S_vac_archive_weighted"], r["S_esc_minus_S_vac_weighted"], _f(r.get("S_B_weighted"), "%.4e"), _f(r.get("S_M_minus_S_Mprime_weighted"), "%.4e"), ", ".join(str(p.get("collected")) for p in pcs),
                ", ".join(str(p.get("archive")) for p in pcs), ", ".join(_f(p.get("outside_weight_min"), "%.5f") for p in pcs), ", ".join(_f(p.get("incoming_fraction"), "%.3f") for p in pcs),
                r["E_remove_weighted"], r["M_chi_weighted"], r["energy_identity_residual_weighted"], r["collar_identity_residual_weighted"], _f(r.get("join_max_dN"), "%.2e"),
                _f(r.get("join_max_rel_dv"), "%.2e")))
        L += ["", "## The transfer ledgers at the checkpoints (v6g (2), after each checkpoint's collection; per channel: tr W_R/p, tr T_K^dagger T_K/p, "
              "||W_H||_F, the accumulation's check |B^dagger F F^dagger B - W_R|_F; T_K's weight summed over the p modes inside r_t, r_t to R_cut, beyond R_cut)", "",
              "| T/r_s | anchored at T/r_s | the anchored ledger | peak-anchored from T/r_s | the peak-anchored ledger |", "|---|---|---|---|---|"]
        for r in rows:
            cells = []
            for key in ("ledger_anchored", "ledger_peak"):
                d = r.get(key) or {}
                t0s = sorted({"%.4f" % v_["t0_over_rs"] for v_ in d.values()})
                cells.append(", ".join(t0s) if t0s else "-")
                cells.append("; ".join("kappa %s: %.4e, %.6f, %.1e, %.1e; %.4f, %.4f, %.4f" % (
                    k_, v_["trace_W_R_over_p"], v_["trace_TK_TK_over_p"], v_["W_H_F"], v_["W_R_check_F"], v_["TK_weight_inside_rt"], v_["TK_weight_rt_to_cut"],
                    v_["TK_weight_beyond_cut"]) for k_, v_ in sorted(d.items())) or "-")
            L.append("| %.3f | %s | %s | %s | %s |" % (r["t_over_rs"], cells[0], cells[1], cells[2], cells[3]))
        with open(self.prefix + "_tables.md", "w") as f:
            f.write("\n".join(L) + "\n")


def main(argv=None):
    a = parse(argv)
    run = HistRun(a)
    if a.resume and os.path.exists(run.prefix + "_state.pt"):
        run.load_state()
        run.run(resumed=True)
    else:
        run.run()


if __name__ == "__main__":
    main()
