#!/usr/bin/env python
"""page_recursion_v6.py. Changes against page_recursion_v5.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 37.1 "Ruled" (2) to
(4), inbox #31 S1; imports page_ops_v6, page_core_v6, page_selftest_v6 and page_scatter_v5 unchanged). The four repairs are switches with
the repaired settings as defaults, so that the regression attributes every change to one of them: --hull exact|v5 (repair 1),
--pair-ledger sharp|v5 (repair 2), --collector active|v5 with --eps-loc (repair 3; default 1e-2), --direct-every 1 (repair 4: the direct
route on the cumulative collected set and the exact Z identity at every checkpoint); with --hull v5 --pair-ledger v5 --collector v5
--direct-every 5 --end-rule v5 every v5 number is reproduced (the new readouts are added beside).
(S1) the end rule (--end-rule v6): the run ends when the collected flux |L| has stayed below --end-flux-frac (1e-3) of the constant-
    luminosity comparator L_cont for --end-window (10) consecutive blocks after S_esc has fallen below --end-entropy-frac (1 percent) of its
    peak, and at --t-stage-max in any case; --end-rule v5 keeps v5's stall rule.
(S1) per checkpoint (every direct checkpoint): the exact collar identity E_out = E_remove + M_chi per channel and weighted (page_core_v6
    collar_identity), with E_out - E_out(0) beside the collar current's trapezoid integral (v5's); the slab mutual information in P07 sec.
    10's definition per unweighted channel (formed state and Killing ground, the bound 4 ||K||_F^2) beside the code's own; at the stage
    start (t = 0) both are printed too; S_esc with its summed error band (the accumulated ledger: the sharp dropped pairs, the compressions'
    r' s(eps_rank), the Gram drops, the clipping).
(S1) checks and validation: the exact Z identity (the active collector's pre-compression Z) below 1e-8; the all-history Gram defect below
    1e-10 at every block; the energy identity below 1e-6 per checkpoint (v5's); the direct route by 36.1 (3)'s revised criterion
    (relative 1e-6 or absolute 1e-9 nats, whichever is larger); the collected modes' smallest outside weight (at least 1 - eps_loc by
    construction); with the active collector the uncollected weight beyond R_cut is printed as collection efficiency with no criterion
    (the uncollected state stays in the memory), the FIFO items do not apply; the collar identity's residual is printed, no criterion.
(S1) labels: every output carries "collected-mode entropy of the formation transient on a prescribed exterior-plus-ball background; the
    interior at H ell = 0.35 (ell 4) and 0.18 (ell 2) is outside the hydrodynamic gate" (S1's words).
(v6 restriction) --collector active is certified for one stage: it keeps no archive transported through a ramp (P01 sec. 5's own
    premise), and the self-test's T4j in v6 mode prints 0.287 nats between the memory route after a transition and the dense entropy of the
    true escaping set (3.05e-06 with v5's collector); main() refuses --collector active with --max-stages above 1.
(37.1 (3)) --keep-start-ckpt keeps the block-0 checkpoint (after the formation) as <tag>_ckpt_start.pt; --start-ckpt PATH starts a run
    from such a checkpoint (copied to the run's checkpoint when the run has none), so the regression's variants share one formation; at
    block 0 the dropped-pair ledger is recomputed from the stored sines under the run's --pair-ledger.

page_recursion_v5.py. Changes against page_recursion_v4.py (builder analyst; SPEC_final4b_engine.md 33.1 "Ruled" (1) to (7) and 33.6
BUILDER item 1), each with the 33.1 item it implements; imports page_ops_v5, page_core_v5, page_selftest_v5 and page_scatter_v5 (the
latter page_scatter_v4 with its imports renamed). The 31.4 corrections of v4 stand.
(33.1 (1)) 8.3 (4)'s energy criterion is the uncompleted identity |tr[Delta_k h_k] - (E_mem + E_coll + E_x)|/|E_init| below 1e-6 per
    stage (every evaluated checkpoint, every channel) while the direct route is kept (the stage record complete); the residual is printed
    per checkpoint; the completed form is withdrawn; E_trunc,acc and the coverage identity are printed with no criterion. The stage table
    prints, per stage, the largest relative residual over its evaluated checkpoints, E_init,k (weighted), the error bound
    sum over channels of 16 kappa max |E_init - E_mem - E_coll - E_x| (energy units) and its accumulation over the stages, the charge
    sequence's error bound (a stage without an evaluated identity is printed so and marks the accumulation incomplete).
(33.1 (2)) the transition's log line and record carry page_core_v5's continuation line (certified unless a Weyl failure, dh/g above 1/8
    or the nsub cap; h_dot/g^2 and the non-adiabatic creation printed); only a flagged continuation enters the validation status.
(33.1 (3)) the FIFO criterion is 1e-4 on each entry's inward weight at its exit (fifo_exit_weight_max over the run, with the exit count);
    the oldest live entry's weight is printed per checkpoint as a depth diagnostic (fifo_oldest_weight_max, no criterion).
(33.1 (4)) the first block after a transition collects uncapped (page_core_v5); the 1e-6 criterion on the uncollected weight is
    unchanged from that block on; the uncollected content (||.||_F, energy) is printed beside the weight per checkpoint and in Checks; the
    first block after a transition is always logged.
(33.1 (5)) a clipping excursion at or below eps_rank is budgeted in the turnover uncertainty (as in v4, R6) and not flagged; above
    eps_rank it is flagged.
(33.1 (6)) the lower-memory transition (page_core_v5): do_transition's snapshot holds no old eigenvectors (Channel.transition_snapshot),
    a restore rebuilds the old stage (Channel.restore_snapshot, its exactness printed); the device peak is reset at each transition's
    start and read with torch.cuda.max_memory_allocated at its end, printed per transition with the allocation at its start and the run's
    peak before it, and every channel's memory phases; the curve rows' gpu_mem_GB stays the run's peak.
(33.1 (7)) the pilot's line is v4's with --max-wall-hours 40 (launch_page_pilot_v5.sh), labelled "on the exact exterior sequence".
R5's localization (the collected modes' post-split inward weight) stays printed with no criterion.

page_recursion_v4.py. Changes against page_recursion_v3.py (builder analyst; SPEC_final4b_engine.md 31.4 "Ruled" (1) to (9) and
31.6 BUILDER item 1; the GAP03 review review_GAP03_page_curve.tex, findings R3 to R11), each with the item it implements; imports
page_ops_v4, page_core_v4, page_selftest_v4 and page_scatter_v4.
(31.4 (1); R3, R7.1) every transition is page_core_v4's: the complete occupied sea through the ramp, the continued reference P^cont by
    engine_ops_v3's rule with its Weyl certificate and the 1/8 sub-step rule, the new memory C_exact - P^cont compressed; the log line
    per channel gives the continuation's record (certified or flagged, nsub, ||dH||_G, the gap, the Weyl margin, the largest sub-step
    ratios), the R3 reference offset ||P^cont - 1(h < 0)||_F^2, the sea part's norm (now in the memory), the leak over every old occupied
    mode, the FIFO's half defect and the memory's weight removed on it, the compression's dropped energy, and the wall times (CF4 of the
    full sea, the sub-step eigendecompositions, the principal angles). New options --nsub-max (64), --tr-chunk (4096 columns),
    --timing-transition (one transition in the timing run, labelled "timing, not the pilot"); --engine-path also locates engine_ops_v3.
(31.4 (2); R7.2) the collar flux and the cross-surface mutual information across the trapping zero come from the exact stage state (the
    stage-initial memory propagated), the latter at every --direct-every checkpoint and at the end of every stage.
(31.4 (3)) 8.3 (4)'s energy criterion is the completed identity tr[Delta_k h_k] = E_mem + E_coll + E_x + E_trunc,acc
    (energy_identity_max_rel, below 1e-8); the uncompleted residual is printed beside it (energy_identity_uncompleted_max_rel, the
    truncation's energy share, not a criterion); E_trunc,acc is printed beside E_rad and in the budget line.
(31.4 (5); R5, R7.3) per checkpoint the FIFO's oldest inward weight (8.3 (4)'s orthogonality bound to earlier radiation, 1e-4), its depth
    and doublings, the collected modes' post-split inward weight, and the uncollected weight beyond R_cut over the active norm (8.3 (4)'s
    1e-6) are printed, their maxima in Checks.
(31.4 (6); R8) the checkpoint keeps the stage record's collected modes (<tag>_cQ/ chunk files, site basis), the stage reference mask and
    its rotated blocks, the FIFO in the frames its timestamps name, the FIFO depth and every v4 accumulator; resume restores them, so the
    resumed record is complete (cQ_from (0, 0)); the empty memory is handled throughout.
(31.4 (7); R4) the stage rule E_{k+1} = E_k - E_rad is labelled "the declared exterior-sequence rule (P28's balance)" in the log, the stage
    table and the summary, with the collar current's time integral beside it per checkpoint.
(31.4 (9); R4, R5, R8, R9) the pilot's readouts: the curve against t_inf, labelled "the joined per-stage chart retarded time at R_cut, not
    a global clock (R4)"; the turnover test raw (v3's I > S(B) + 1e-9) and established only above the combined uncertainty (per block:
    the decrease S_old - S_new against the block's D8 increment plus the clipping term; over the run: the decrease from the maximum
    against the accumulated bounds at both ends); the saturation against eta A/a^2 and A/(4G) at G/a^2 1/4, labelled a comparison with
    supplied levels (R9); the mutual information across the trapping zero of the stage state; per stage the back-averaged root r_H,m and
    slope kappa_m of N + v on the stage's site fields (J3' version 6's five-site quadratic fit) and the thermal comparator at
    T_m = kappa_m/(2 pi) for the run's channels with the continuum Gamma, sum over kappa of (64 kappa/(2 pi)) int omega Gamma f(omega/T_m),
    beside the stage's L_mean and the same comparator at T_H (massless only). The validation status ("validated", or "unresolved:" with the
    failing 8.3 items, and the items this run does not evaluate named) is written to the log, the summary, the tables and the npz (R8,
    R11).
(R3) the formation record states the preparation as declared (the initial configuration's Killing sea 1(h_i < 0); the stage-1 reference
    P_1 = 1(h_1 < 0) declared).
(R6) the Chebyshev tail bound accumulated over the formation and over each transition is printed; the averaging quadrature check (ell/16
    and 8 ell against ell/32 and 10 ell) is printed once at the start.
Every v3 readout is kept.

page_recursion_v3.py. Changes against page_recursion_v2.py (builder analyst, BRIEF_page_v3_builder.md), each with the item of
SPEC_final4b_engine.md 30.1 it implements: imports page_ops_v3, page_core_v3, page_selftest_v3, page_scatter_v3. (3) the emitted energy is
the memory route, E_rad,k(t) = tr[Delta_k h_k] - E_mem(t), per channel and weighted; 3.4's charge update, the stage rule and the collected
flux (the drop of E_rad over a block) use it; E_diag (the diagonal block sum, v2's E_rad) is printed beside it with E_diag - E_rad the
blocks' coherence energy; at the direct route's checkpoints, while the stage record holds every collected mode, the full identity
tr[Delta_k h_k] = E_mem + E_coll + E_x (page_core_v3.energy_identity) is printed and its residual is 8.3 (4)'s energy criterion
(energy_identity_max_rel, below 1e-8; v2's energy_balance_max_rel of the withdrawn D10 balance is gone); per checkpoint the time-integrated
collar current (trapezoid on the checkpoints) is printed against E_rad, their difference reported and not gated; the direct route counts
only complete records. (4) and 8.3 (8): each transition prints the dropped non-adiabatic coherence and checks the accumulated D8 bound
against 1 percent of S_esc, the adiabatic leak against 1e-4 and the S-continuity defect against 1e-6; on a failure it restores the
channels and repeats the transition with tau_tr doubled (--tau-tr-doublings, default 3; 0 only prints). (6) D7's partner identity is
read at the end of stage 1 and again at the end, labelled; the Checks block labels fifo_overlap_after_max as 8.3 (4)'s 1e-4 criterion and
fifo_overlap_max as the candidates' redundancy; --timing-blocks N runs the formation, the stage-1 eigendecomposition and N checkpoints
labelled "timing, not the pilot" and prints each one's wall time. The checkpoint and resume carry the new state. Not a 30.1 item:
the direct route's entropy check (tr s(G_R) against tr s(M)) is made in stage 1 only; after a transition the stage record holds only the
stage's own collected modes, whose entropy is not S_esc (the earlier stages' radiation is outside the record), so v1 and v2 compared
different quantities there (0.0144 against 0.0036 in the v3 smoke run's third stage); it is printed as S_stage_collected instead.

page_recursion_v2.py. Changes against page_recursion_v1.py (builder analyst, BRIEF_page_debug_v1.md): imports page_ops_v2,
page_core_v2, page_selftest_v2 and page_scatter_v2 (the numerical fixes behind T4b, T4d, T4e and T4g live there); new option --engine-path
(T1a: the directory of engine_ops_v1.py for the self-test's operator comparison; without it: $PAGE_ENGINE_PATH, then ../engine,
../engine/code and ../../engine/code beside this file, then <internal>/code); the Checks block adds fifo_overlap_after_max
(the largest overlap with the FIFO after the projection; fifo_overlap_max stays the overlap before it); checkpoint stores each FIFO mode
at its own collection time tq (v1 stored it propagated to t_ref under the label tq, so a resumed run advanced its phase a second time; the
icloud lane's finding, SPEC_final4b_engine.md 29.9; the transition's time origin, the other finding, is fixed in page_core_v2); resume
marks where the stage record's stored modes restart (rec["cQ_from"], read by page_core_v2.stage_direct); the file names in the printed
text. Nothing else changes.

page_recursion_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the Page curve of the model's radiating object by
the retained-channel recursion (REVIEWA_REVIEW_web40e.md S5; GAP03_integrated_v1.md). The sea of the initial configuration is carried through
the declared formation and through the exact exterior sequence (N = 1, v = -sqrt(2GE/r) outside a declared regular ball) by one-particle
propagation; per channel kappa (weight 16 kappa) the memory covariance obeys M_{k+1} = D_k M_k D_k^dagger + Xi_k, S_esc = tr s(M), the
new block's entropy S(B_k) = tr s(Z_k), I(R_<k : B_k) = tr s(M_k) + tr s(Z_k) - tr s(M_{k+1}), turnover iff I > S(B_k). A stage ends when
the collected Killing energy reaches dE_frac E_k (the review's E_{n+1} = 0.995 E_n), the charge drops by the measured energy and the state is
carried through a smooth ramp. Nothing thermal is put in. Outputs: <out>/<tag>_summary.json, <tag>_tables.md, <tag>_curve.npz, the
checkpoint <tag>_ckpt.pt (resume with --resume). Defaults are the r_s/a = 32 pilot (G/a^2 = 1/4, massless).

Usage: python page_recursion_v6.py --out DIR --tag TAG [--rs 32 --g 0.25 --nsites 8192 --channels 1,2 ...]
       python page_recursion_v6.py --selftest [--device cpu] [--engine-path DIR]
       python page_recursion_v6.py --validate-scattering --out DIR --tag TAG [--rs 32 ...]
       python page_recursion_v6.py --timing-blocks 3 --out DIR --tag TAG [pilot settings]   (timing, not the pilot)"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v6 as OPS            # noqa: E402
import page_core_v6 as CORE          # noqa: E402

CD = OPS.CD
RD = OPS.RD
L_CONT_RS2 = 0.0052373      # REVIEWA_REVIEW_web40e.md 1.11 (continuum massless luminosity, 16 two-component species)
B_CONT = 1.6391             # REVIEWA_REVIEW_web40e.md 1.11 (continuum stationary entropy ratio)
T_START = time.time()
LABEL_SEQUENCE = "on the exact exterior sequence"
LABEL_CLOCK = "t_inf: the joined per-stage chart retarded time at R_cut (R4: not a global clock for the time-dependent history)"
LABEL_RULE = "E_{k+1} = E_k - E_rad: the declared exterior-sequence rule (P28's balance; R4)"
LABEL_LEVELS = "eta A/a^2 and A/(4G) are supplied levels; the comparison proves no saturation (R9)"
LABEL_S1 = ("collected-mode entropy of the formation transient on a prescribed exterior-plus-ball background; the interior at H ell = 0.35 "
            "(ell 4) and 0.18 (ell 2) is outside the hydrodynamic gate")


def label_s1(a):
    """v6 (S1): S1's label; S1 gives H ell at ell 4 and 2 only."""
    return LABEL_S1 + ("" if float(a.ell) in (2.0, 4.0) else " (S1 gives H ell at ell 4 and 2 only; this run's ell is %g)" % a.ell)


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T_START, msg), flush=True)


# ------------------------------------------------------------------------------------------------------------------------ config
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs", type=float, default=32.0, help="r_s/a of the first stage")
    ap.add_argument("--g", type=float, default=0.25, help="G/a^2")
    ap.add_argument("--m", type=float, default=0.0, help="m a (the pilot is massless; 4 pi m r_s is printed beside the collapse's 160)")
    ap.add_argument("--nsites", type=int, default=8192)
    ap.add_argument("--half-offset", dest="half_offset", type=int, default=0)
    ap.add_argument("--channels", default="1,2", help="kappa list, weights 16 kappa; 3 is the truncation check")
    ap.add_argument("--ell", type=float, default=4.0)
    ap.add_argument("--fields", default="averaged", choices=["averaged", "point"])
    ap.add_argument("--Rb", type=float, default=0.5, help="ball radius over r_s (0.8 control)")
    ap.add_argument("--Rb0", type=float, default=4.0, help="formation start radius over r_s (8 control)")
    ap.add_argument("--formation", default="dust", choices=["dust", "ramp"])
    ap.add_argument("--t-ramp", dest="t_ramp", type=float, default=2.0, help="ramp formation duration over r_s")
    ap.add_argument("--dt-form", dest="dt_form", type=float, default=0.5)
    ap.add_argument("--form-chunk", dest="form_chunk", type=int, default=4096, help="column chunk of the formation propagation (device memory)")
    ap.add_argument("--R-cut", dest="R_cut", type=float, default=32.0, help="collection radius over r_s,1 (fixed through the sequence)")
    ap.add_argument("--R-collar", dest="R_collar", type=float, default=16.0, help="collar radius over r_s,1 for the current readout")
    ap.add_argument("--tau-s", dest="tau_s", type=float, default=4.0, help="checkpoint spacing over r_s,1")
    ap.add_argument("--m-cap", dest="m_cap", type=int, default=64, help="largest number of collected directions per checkpoint and channel")
    ap.add_argument("--eps-coll", dest="eps_coll", type=float, default=1e-5)
    ap.add_argument("--eps-rank", dest="eps_rank", type=float, default=1e-9)
    ap.add_argument("--eps-form", dest="eps_form", type=float, default=1e-7)
    ap.add_argument("--eps-orth", dest="eps_orth", type=float, default=1e-12)
    ap.add_argument("--fifo-depth", dest="fifo_depth", type=int, default=12, help="the FIFO's initial depth (v4: doubled at the cap, 31.4 (5))")
    ap.add_argument("--fifo-prune-weight", dest="fifo_prune_weight", type=float, default=1e-10, help="a collected mode leaves the FIFO below this weight near the cut")
    ap.add_argument("--dE-frac", dest="dE_frac", type=float, default=0.005, help="stage ends when E_rad reaches dE_frac E_k")
    ap.add_argument("--interior", default="selfsimilar", choices=["selfsimilar", "fixed"])
    ap.add_argument("--tau-tr", dest="tau_tr", type=float, default=4.0, help="ramp duration over r_s,k at a transition")
    ap.add_argument("--dt-tr", dest="dt_tr", type=float, default=0.5)
    ap.add_argument("--max-stages", dest="max_stages", type=int, default=200)
    ap.add_argument("--E-stop-frac", dest="E_stop_frac", type=float, default=0.5)
    ap.add_argument("--t-stage-max", dest="t_stage_max", type=float, default=3000.0, help="stage cap over r_s,1")
    ap.add_argument("--stall-frac", dest="stall_frac", type=float, default=0.01)
    ap.add_argument("--stall-window", dest="stall_window", type=int, default=10)
    ap.add_argument("--t-stall-min", dest="t_stall_min", type=float, default=200.0, help="no stall verdict before this time over r_s,1")
    ap.add_argument("--force-steps", dest="force_steps", type=int, default=0, help="labelled control: step the charge by dE_frac E_k on a stall")
    ap.add_argument("--store-cap", dest="store_cap", type=int, default=6000)
    ap.add_argument("--store-modes", dest="store_modes", type=int, default=1)
    ap.add_argument("--direct-every", dest="direct_every", type=int, default=1, help="v6 (S1 repair 4): 1 (v5: 5)")
    ap.add_argument("--eta", type=float, default=0.9145, help="the lane's sphere coefficient at mesh 4 (REVIEWA_REVIEW_web40e.md)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-wall-hours", dest="max_wall_hours", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--engine-path", dest="engine_path", default=None,
                    help="directory holding engine_ops_v1.py (T1a) and engine_ops_v2/v3.py (the transition's continuation, 31.4 (1))")
    ap.add_argument("--obs-path", dest="obs_path", default=None,
                    help="v4 (31.4 (8)): directory of obs_plateau_v4/v5.py and obs_greybody_radial_v1.py (default: $PAGE_OBS_PATH, ../obs, ../obs/code, ../../obs/code, <internal>/code)")
    ap.add_argument("--nsub-max", dest="nsub_max", type=int, default=64, help="v4 (31.4 (1)): the cap of the continuation's sub-steps (a capped transition is flagged)")
    ap.add_argument("--tr-chunk", dest="tr_chunk", type=int, default=4096, help="v4 (31.4 (1)): column chunk of the full sea's propagation through the ramp")
    ap.add_argument("--tau-tr-doublings", dest="tau_tr_doublings", type=int, default=3, help="8.3 (8): repeat a failing transition with tau_tr doubled, at most this often")
    ap.add_argument("--timing-blocks", dest="timing_blocks", type=int, default=0, help="timing, not the pilot: the formation, stage 1 and this many checkpoints")
    ap.add_argument("--timing-transition", dest="timing_transition", type=int, default=0, help="v4: with --timing-blocks, then one transition (dE_frac E) and one block, timed")
    ap.add_argument("--validate-scattering", dest="validate_scattering", action="store_true")
    # v6 (S1): the four repairs (repaired defaults), the end rule, the shared-formation start
    ap.add_argument("--hull", default="exact", choices=["exact", "v5"], help="v6 (S1 repair 1): the compression's split hull")
    ap.add_argument("--pair-ledger", dest="pair_ledger", default="sharp", choices=["sharp", "v5"], help="v6 (S1 repair 2): the dropped-pair ledger")
    ap.add_argument("--collector", default="active", choices=["active", "v5"], help="v6 (S1 repair 3): the active-subspace collector or v5's")
    ap.add_argument("--eps-loc", dest="eps_loc", type=float, default=1e-2, help="v6 (S1 repair 3): collect memory directions with outside weight >= 1 - eps_loc")
    ap.add_argument("--end-rule", dest="end_rule", default="v6", choices=["v6", "v5"], help="v6 (S1): the drain end rule, or v5's stall rule")
    ap.add_argument("--end-flux-frac", dest="end_flux_frac", type=float, default=1e-3)
    ap.add_argument("--end-window", dest="end_window", type=int, default=10)
    ap.add_argument("--end-entropy-frac", dest="end_entropy_frac", type=float, default=0.01)
    ap.add_argument("--keep-start-ckpt", dest="keep_start_ckpt", action="store_true", help="v6: keep the block-0 checkpoint as <tag>_ckpt_start.pt")
    ap.add_argument("--start-ckpt", dest="start_ckpt", default=None, help="v6: start from this block-0 checkpoint (copied when the run has none)")
    ap.add_argument("--v5-selftest-results", dest="v5_selftest_results", default=None, help="v6 self-test: the stored v5 self-test results for T-V5REPRO")
    a = ap.parse_args(argv)
    finish_cfg(a)
    return a


def finish_cfg(a):
    a.E1 = a.rs / (2.0 * a.g)
    a.TH = 1.0 / (4.0 * math.pi * a.rs)
    a.kappa_surf = 1.0 / (2.0 * a.rs)
    a.R_cut_abs = a.R_cut * a.rs
    a.R_collar_abs = a.R_collar * a.rs
    a.tau_s_abs = a.tau_s * a.rs
    a.m_over_TH = 4.0 * math.pi * a.m * a.rs
    a.Rbf = a.Rb
    a.channel_list = [int(x) for x in str(a.channels).split(",") if x.strip()]
    a.store_modes = bool(a.store_modes)
    return a


def levels(a, rs):
    return {"A_over_4G": math.pi * rs * rs / a.g, "eta_A_over_a2": 4.0 * math.pi * a.eta * rs * rs,
            "memory_budget_modes": 4.0 * math.pi * a.eta / math.log(2.0) * rs * rs, "L_cont": L_CONT_RS2 / (rs * rs),
            "S_dot_cont": B_CONT * L_CONT_RS2 / (rs * rs) * 4.0 * math.pi * rs}


def Rb_of(a, E):
    rs = 2.0 * a.g * E
    return a.Rb * rs if a.interior == "selfsimilar" else a.Rb * a.rs


# ---------------------------------------------------------------------------------------------------------------------- driver
class Run:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
        torch.set_num_threads(a.threads)
        os.makedirs(a.out, exist_ok=True)
        self.prefix = os.path.join(a.out, a.tag)
        self.chans = [CORE.Channel(k, a.m, a.nsites, bool(a.half_offset), self.dev) for k in a.channel_list]
        for ch in self.chans:
            ch.set_averager(a.ell, a.fields)
        self.curve = []; self.stages = []; self.transitions = []; self.formation = {}; self.inversion = {}
        self.k = 0; self.E = a.E1; self.t = 0.0; self.t_stage0 = 0.0; self.n_block = 0; self.E_rad_stage = 0.0; self.E_rad_total = 0.0
        self.L_hist = []; self.L_peak = 0.0; self.Rb_now = Rb_of(a, a.E1); self.stop_reason = None
        # v3 (30.1 (3)): the memory route's emission of the completed stages, the diagonal sum, the collar current's integral
        self.E_rad_prev_stages = 0.0; self.E_diag_stage = 0.0; self.E_collar_stage = 0.0; self.Lc_prev = 0.0; self.partners_stage1 = None
        self.timings = {}
        self.checks = {"Z_identity_max": 0.0, "energy_identity_max_rel": 0.0, "energy_identity_checkpoints": 0, "clip_excursion_max": 0.0,
                       "fifo_overlap_max": 0.0, "fifo_overlap_after_max": 0.0, "Xi_min_eig_min": 0.0, "I_minus_DD_minus_Xi_min": 0.0,
                       "direct_vs_memory_max_rel": 0.0, "direct_checkpoints": 0, "D8_bound_over_S_esc_at_transitions_max": 0.0,
                       # v4
                       "gram_energy_bound_rel_max": 0.0, "fifo_oldest_weight_max": 0.0,
                       "post_split_inward_weight_max": 0.0, "uncollected_over_active_max": 0.0, "fifo_depth_doublings": 0,
                       "transitions_flagged": 0, "transition_leak_max": 0.0, "transition_S_continuity_max": 0.0,
                       # v5 (33.1 (1), (3), (4), (6))
                       "coverage_identity_max_rel": 0.0, "memory_mismatch_max_rel": 0.0, "fifo_exit_weight_max": 0.0, "fifo_exits": 0,
                       "uncollected_content_F_max": 0.0, "uncollected_content_E_max_abs": 0.0, "transition_device_peak_GB_max": 0.0,
                       "restores": 0, "restores_inexact": 0}
        # v4 (31.4 (9); R5, R8): the run-level turnover test against the combined uncertainty
        self.S_max = None; self.t_S_max = None; self.U_at_max = 0.0; self.turnover_established_any = False
        self.stage_readouts = []; self.quadrature = None; self.cont_tables = None; self.validation = None
        # v5 (33.1 (1)): the stage's identity record (per channel the largest |E_init - E_mem - E_coll - E_x| over its evaluated
        # checkpoints, the largest relative residual, the count) and the charge sequence's accumulated error bound
        self.ident_stage = {}; self.ident_stage_rel_max = 0.0; self.ident_stage_n = 0
        self.E_bound_acc = 0.0; self.E_bound_incomplete = False
        # v5 (33.1 (6)): the run's device peak (the transitions reset torch's peak; this keeps the run's maximum)
        self.gpu_peak_run = 0.0
        # v6 (S1): the end rule's consecutive count, the stage-start readouts (the collar identity's E_out(0), M_chi(0) and the P07 slabs at
        # t = 0), and the new checks
        self.end_count = 0; self.end_armed_t = None; self.start_readouts = {}
        self.checks.update({"Z_identity_exact_max": 0.0, "gram_all_history_max": 0.0, "g_sel_min": None, "outside_weight_remaining_max": 0.0,
                            "DDdag_defect_max": 0.0, "Xi_norm_max": 0.0, "direct_revised_ratio_max": 0.0, "direct_abs_max": 0.0,
                            "collar_identity_residual_max_rel": 0.0, "collar_identity_checkpoints": 0, "reset_bound_acc": 0.0})

    # ------------------------------------------------------------------------------------------------- v4 readouts
    def continuum_tables(self):
        """v4 (31.4 (9)): the continuum Gamma_kappa(x) on the Gauss-Legendre nodes of [0, 1.5] (48 nodes; page_scatter_v4's comparator) for
        the run's channels, for the thermal comparator at T_m; massless only."""
        if self.cont_tables is not None or self.a.m != 0.0:
            return self.cont_tables
        import page_scatter_v5 as SC
        xs, ws = np.polynomial.legendre.leggauss(48)
        x = 0.75 * (xs + 1.0); w = 0.75 * ws
        self.cont_tables = {str(k): {"x": x, "w": w, "Gamma": SC.gamma_table(k, x)} for k in self.a.channel_list}
        return self.cont_tables

    def comparator_L(self, rs, T):
        """sum over the run's channels of (64 kappa/(2 pi)) int domega omega Gamma_kappa f(omega/T), omega = x/r_s (lattice units); None when
        massive (the comparator is the massless continuum's, R10)."""
        tabs = self.continuum_tables()
        if tabs is None or T is None or T <= 0:
            return None
        tot = 0.0
        for k_, tb in tabs.items():
            arg = np.clip(tb["x"] / (rs * T), -700, 700)
            tot += 64.0 * float(k_) / (2.0 * math.pi) * float(np.sum(tb["w"] * tb["x"] * tb["Gamma"] / (np.exp(arg) + 1.0))) / (rs * rs)
        return tot

    def stage_readout(self):
        """v4 (31.4 (9)): the stage's frozen record (its back-averaged site fields, N = 1, v the averaged flow): the outer root r_H,m and
        slope kappa_m of N + v by J3' version 6's five-site quadratic fit (page_scatter_v4.outer_root), the inner root r_in,m, T_m =
        kappa_m/(2 pi) against T_H of the stage's charge, and the thermal comparator at T_m and at T_H for the run's channels."""
        import page_scatter_v5 as SC
        a = self.a
        ch = self.chans[0]; st = ch.stage
        f = 1.0 + np.asarray(st["v_sites"], float)
        rHm, km = SC.outer_root(ch.R, f)
        rin = SC.inner_root(ch.R, f, rHm) if rHm is not None else None
        rs_k = st["rs"]; TH_k = 1.0 / (4.0 * math.pi * rs_k)
        Tm = (km / (2.0 * math.pi)) if km else None
        out = {"k": self.k, "E": self.E, "rs": rs_k, "T_H": TH_k, "r_Hm": rHm, "kappa_m": km, "r_in_m": rin, "r_t": st["r_t"], "r_in": st["r_in"],
               "kappa_m_2rs": (km * 2.0 * rs_k) if km else None, "T_m": Tm, "T_m_over_T_H": (Tm / TH_k) if Tm else None,
               "L_comp_T_m": self.comparator_L(rs_k, Tm), "L_comp_T_H": self.comparator_L(rs_k, TH_k), "channels": a.channel_list,
               "label": "the thermal comparator at T_m = kappa_m/(2 pi) of the frozen stage record, continuum Gamma, the run's channels (31.4 (9); massless)"}
        self.stage_readouts.append(out)
        log("stage %d frozen record: r_H,m %s, kappa_m %s (2 r_s kappa_m %s), r_in,m %s; T_m/T_H %s; thermal comparator over channels %s: L(T_m) %s, L(T_H) %s" % (
            self.k, _f(rHm, "%.4f"), _f(km, "%.6e"), _f(out["kappa_m_2rs"], "%.5f"), _f(rin, "%.4f"), _f(out["T_m_over_T_H"], "%.5f"), a.channel_list,
            _f(out["L_comp_T_m"], "%.4e"), _f(out["L_comp_T_H"], "%.4e")))
        return out

    # ------------------------------------------------------------------------------------------------------------ start
    def start(self):
        a = self.a
        self.quadrature = self.chans[0].averager.quadrature_check(OPS.flow_dense(a.rs, Rb_of(a, a.E1)))     # v4 (R6)
        log("averaging quadrature check (R6): max |v(ell/16, 8 ell) - v(ell/32, 10 ell)| %.3e (max |v| %.4f); l = 0 norms in the bulk %s; the Gaussian tail beyond 8 ell %.2e" % (
            self.quadrature["max_abs_dv"], self.quadrature.get("max_abs_v", 0.0), _f(self.quadrature.get("max_abs_norm0_minus_1_bulk"), "%.3e"),
            self.quadrature.get("gaussian_tail_beyond_8_ell", 0.0)))
        for ch in self.chans:
            t_f0 = time.time()
            cs0 = OPS.cheb_stats()
            Phi0, frec = ch.formation(a)
            frec["chebyshev"] = OPS.cheb_delta(cs0)                                                         # v4 (R6)
            frec["preparation"] = "declared: the initial configuration's Killing sea 1(h_i < 0), zero modes empty (R3; ||v||_inf < 1 alone does not establish it)"
            t_f1 = time.time()
            log("kappa %d: formation %s t_form %.2f r_s, n0 %d, zero modes %d, u_max %.3f, unitarity %.2e (eigh %.0f s, cf4 %.0f s); Chebyshev: %d exponentials, accumulated tail bound %.2e, largest order %d; preparation %s" % (
                ch.kappa, a.formation, frec["t_form_over_rs"], frec["n0"], frec["n_zero_modes_initial"], frec["u_max_initial"], frec["unitarity"], frec["eigh_s"], frec["cf4_s"],
                frec["chebyshev"]["calls"], frec["chebyshev"]["tail_bound_acc"], frec["chebyshev"]["K_max"], frec["preparation"]))
            st = ch.build_stage(a, a.E1, self.Rb_now, label="stage 1")
            t_f2 = time.time()
            irec = ch.initial_memory(Phi0, a)
            t_f3 = time.time()
            self.timings[str(ch.kappa)] = {"formation_s": t_f1 - t_f0, "formation_cf4_s": frec["cf4_s"], "stage1_eigh_s": st["eigh_s"],
                                           "stage1_build_s": t_f2 - t_f1, "initial_memory_s": t_f3 - t_f2,
                                           "device_after_initial_memory": OPS.device_memory(self.dev)}            # v5 (33.1 (6)): printed below
            dm_ = self.timings[str(ch.kappa)]["device_after_initial_memory"]
            if dm_ is not None:
                log("kappa %d: device memory after the formation and the initial memory (33.1 (6)): %.3f GB allocated, the run's peak so far %.3f GB" % (
                    ch.kappa, dm_["allocated_GB"], dm_["peak_GB"]))
            log("kappa %d: wall times: formation %.1f s (CF4 %.1f s), stage-1 build %.1f s (eigendecomposition %.1f s), initial memory %.1f s; "
                "dropped-pair share %.4e, its entropy bound %.3e nats (D8 lemma, v2's term, %.3e)" % (
                    ch.kappa, t_f1 - t_f0, frec["cf4_s"], t_f2 - t_f1, st["eigh_s"], t_f3 - t_f2, irec["dropped_pair_share"],
                    irec["entropy_bound_dropped_pairs"], irec["D8_lemma_dropped_pairs_v2"]))
            del Phi0
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
            log("kappa %d: stage 1 r_s %.3f r_t %s r_in %s n_neg %d (eigh %.0f s); Delta_0 rank %d (sin theta max %.3e), memory rank %d; "
                "N+ %.4f N- %.4f E_inv/E %.3e; E_inv per bin of T_H %s; reference %s" % (
                    ch.kappa, st["rs"], st["r_t"], st["r_in"], st["nneg"], st["eigh_s"], irec["rank_kept"], irec["sin_theta_max"], irec["memory_rank"],
                    irec["N_plus"], irec["N_minus"], irec["E_inv_over_E"], ["%.3f" % x for x in irec["inverted_by_bin_TH"]], irec["reference"]))
            self.formation[str(ch.kappa)] = frec; self.inversion[str(ch.kappa)] = irec
        self.t = 0.0; self.t_stage0 = 0.0; self.k = 0; self.E = a.E1; self.E_rad_stage = 0.0
        self.Lc_prev = sum(ch.w * ch.collar_flux(self.t) for ch in self.chans); self.E_collar_stage = 0.0
        self.stage_readout()
        self.stage_start_readouts()

    def stage_start_readouts(self):
        """v6 (S1): at the stage start (t = 0, nothing collected): per channel the collar identity's E_out(0), E_tot and M_chi(0) (= E_out(0)),
        the P07 slab mutual information of the formed state and the Killing ground, and the code's own cross_surface."""
        a = self.a
        out = {"t": self.t, "per_channel": {}}
        for ch in self.chans:
            ci = ch.collar_identity(a, t=self.t)
            p7 = ch.cross_surface_p07(a, t=self.t)
            own = ch.cross_surface(a, t=self.t)
            out["per_channel"][str(ch.kappa)] = {"collar_identity": ci, "slab_p07": p7, "cross_surface_own": own}
            log("kappa %d stage start (t 0): E_out(0) beyond R_collar %s, tr[h Delta] %s, M_chi(0) %s; P07 slabs (unweighted; formed / Killing ground): %s; "
                "the code's own: %s" % (ch.kappa, _f(ci["E_out"] if ci else None, "%.6e"), _f(ci["E_tot"] if ci else None, "%.6e"),
                                        _f(ci["M_chi"] if ci else None, "%.6e"),
                                        "; ".join("d %d: %.6e / %.6e (excess %.3e, X %s, Y %s)" % (x["d"], x["formed"]["I"], x["killing_ground"]["I"], x["excess"],
                                                                                                     x["X_R"], x["Y_R"]) for x in p7) or "-",
                                        "; ".join("d %d: %.6e / %.6e" % (x["d"], x["physical"]["I"], x["reference_vacuum"]["I"]) for x in own) or "-"))
        out["E_out0_weighted"] = sum(ch.w * (out["per_channel"][str(ch.kappa)]["collar_identity"] or {}).get("E_out", 0.0) for ch in self.chans)
        self.start_readouts = out
        return out

    # ------------------------------------------------------------------------------------------------------------- block
    def block(self):
        """one checkpoint over every channel. v5: (33.1 (1)) the uncompleted identity is the criterion and the stage's identity record
        is kept for the stage table's error bound; (33.1 (3)) the FIFO's exit weights; (33.1 (4)) the uncollected content beside the weight,
        the first block after a transition always logged; (33.1 (6)) gpu_mem_GB is the run's peak (the transitions reset torch's peak)."""
        a = self.a
        self.t += a.tau_s_abs
        self.n_block += 1
        rs_k = 2.0 * a.g * self.E
        row = {"t": self.t, "t_over_rs1": self.t / a.rs, "t_inf": OPS.chart_retarded_time(self.t, rs_k, a.R_cut_abs) - OPS.chart_retarded_time(0.0, a.rs, a.R_cut_abs),
               "stage": self.k, "E": self.E, "E_over_E1": self.E / a.E1, "per_channel": {}}
        S_tot = S_B_tot = I_tot = E_B_tot = Lc_tot = E_mem_tot = 0.0; rank_tot = 0.0; wall = 0.0; S_dir_tot = 0.0; have_dir = True
        E_rad_st = E_diag_st = 0.0; ident = {"E_init": 0.0, "E_mem": 0.0, "E_coll": 0.0, "E_x": 0.0, "E_trunc": 0.0}; have_ident = True
        S_old_tot = 0.0; inc_tot = 0.0; clip_tot = 0.0; E_trunc_tot = 0.0; mi_tot = {}; have_mi = False
        unc_ratio_max = 0.0; unc_F_max = 0.0; unc_E_max = 0.0; first_after = False; cov_max = None
        direct_now = (self.n_block % a.direct_every == 0)
        ci_tot = {"E_out": 0.0, "E_remove": 0.0, "M_chi": 0.0, "residual": 0.0, "E_K": 0.0, "E_tot": 0.0, "E_remove_exact": 0.0}; have_ci = False; ci_complete = True
        for ch in self.chans:
            r = ch.block_step(a, self.t)
            w = ch.w
            first_after = first_after or bool(r.get("first_block_after_transition"))
            # v6 (S1): the active collector's checks (the exact Z, the all-history Gram, the outside weights, D D^dagger, Xi)
            if r["m_coll"] and "S_B_Z_exact" in r:
                self.checks["Z_identity_exact_max"] = max(self.checks["Z_identity_exact_max"], abs(r["S_B_Z_exact"] - r["S_B"]))
            if "gram_all_history_F" in r:
                self.checks["gram_all_history_max"] = max(self.checks["gram_all_history_max"], r["gram_all_history_F"])
            if r.get("g_sel_min") is not None:
                gm = self.checks.get("g_sel_min")
                self.checks["g_sel_min"] = r["g_sel_min"] if gm is None else min(gm, r["g_sel_min"])
            if r.get("collector") == "active":
                self.checks["outside_weight_remaining_max"] = max(self.checks["outside_weight_remaining_max"], r.get("weight_beyond_cut_uncollected", 0.0))
            self.checks["DDdag_defect_max"] = max(self.checks["DDdag_defect_max"], r.get("DDdag_defect_F", 0.0))
            self.checks["Xi_norm_max"] = max(self.checks["Xi_norm_max"], r.get("Xi_norm", 0.0) if r.get("collector") == "active" else 0.0)
            self.checks["reset_bound_acc"] += w * r.get("reset_bound_rank_sensitive", 0.0)
            S_tot += w * r["S_new"]; S_B_tot += w * r["S_B"]; I_tot += w * r["I_RB"]; E_B_tot += w * r["E_B"]
            Lc_tot += w * r["L_collar"]; E_mem_tot += w * r["E_mem"]; rank_tot += w * r["r_after"]; wall += r["wall_s"]
            S_old_tot += w * r["S_old"]; inc_tot += w * r.get("ent_bound_increment", 0.0)
            clip_tot += w * (r["r_before"] + r["r_after"]) * OPS.binary_entropy_scalar(r.get("clip_excursion_max", 0.0))   # R5, 33.1 (5): clipping budgeted
            if r["m_coll"]:
                self.checks["Z_identity_max"] = max(self.checks["Z_identity_max"], abs(r["S_B_Z"] - r["S_B"]))
                self.checks["clip_excursion_max"] = max(self.checks["clip_excursion_max"], r.get("clip_excursion_max", 0.0))
                self.checks["fifo_overlap_max"] = max(self.checks["fifo_overlap_max"], r.get("fifo_overlap_max", 0.0))
                self.checks["fifo_overlap_after_max"] = max(self.checks.get("fifo_overlap_after_max", 0.0), r.get("fifo_overlap_after", 0.0))
                self.checks["Xi_min_eig_min"] = min(self.checks["Xi_min_eig_min"], r.get("Xi_min_eig", 0.0))
                self.checks["I_minus_DD_minus_Xi_min"] = min(self.checks["I_minus_DD_minus_Xi_min"], r.get("I_minus_DD_minus_Xi_min_eig", 0.0))
                self.checks["post_split_inward_weight_max"] = max(self.checks["post_split_inward_weight_max"], r.get("post_split_inward_weight_max", 0.0))
            self.checks["fifo_oldest_weight_max"] = max(self.checks["fifo_oldest_weight_max"], r.get("fifo_oldest_weight", 0.0))      # v5: a depth diagnostic
            ratio = r.get("weight_beyond_cut_uncollected", 0.0) / max(r.get("active_norm", 1.0), 1.0)
            unc_ratio_max = max(unc_ratio_max, ratio)
            unc_F_max = max(unc_F_max, r.get("uncollected_content_F", 0.0)); unc_E_max = max(unc_E_max, abs(r.get("uncollected_content_E", 0.0)))
            self.checks["uncollected_over_active_max"] = max(self.checks["uncollected_over_active_max"], ratio)
            self.checks["uncollected_content_F_max"] = max(self.checks["uncollected_content_F_max"], r.get("uncollected_content_F", 0.0))
            self.checks["uncollected_content_E_max_abs"] = max(self.checks["uncollected_content_E_max_abs"], abs(r.get("uncollected_content_E", 0.0)))
            self.checks["fifo_depth_doublings"] = sum(len(c_.fifo_doublings) for c_ in self.chans)
            self.checks["fifo_exit_weight_max"] = max(c_.fifo_exit_weight_max for c_ in self.chans)                                # v5 (33.1 (3))
            self.checks["fifo_exits"] = sum(c_.fifo_exits for c_ in self.chans)
            # v3 (30.1 (3)): the emitted energy by the memory route, E_rad = tr[Delta_k h_k] - E_mem(t); E_diag (the diagonal block sum) beside it
            rec = ch.rec
            r["E_rad_memory_route"] = rec["E_delta_init"] - r["E_mem"]; r["E_diag"] = rec["E_rad"]; r["E_trunc_acc"] = rec.get("E_trunc", 0.0)
            E_rad_st += w * r["E_rad_memory_route"]; E_diag_st += w * rec["E_rad"]; E_trunc_tot += w * rec.get("E_trunc", 0.0)
            if direct_now:
                d = ch.stage_direct(a)
                if d is not None and d.get("complete") and self.k == 0:        # v3: a complete record, and stage 1 only (D1, D2)
                    r["S_direct"] = d["S_direct"]; S_dir_tot += w * d["S_direct"]
                    self.checks["direct_vs_memory_max_rel"] = max(self.checks["direct_vs_memory_max_rel"], abs(d["S_direct"] - r["S_new"]) / max(r["S_new"], 1e-12))
                    self.checks["direct_checkpoints"] += 1
                    dabs = abs(d["S_direct"] - r["S_new"])                     # v6: 36.1 (3)'s revised criterion, relative 1e-6 or absolute 1e-9
                    self.checks["direct_abs_max"] = max(self.checks["direct_abs_max"], dabs)
                    self.checks["direct_revised_ratio_max"] = max(self.checks["direct_revised_ratio_max"], dabs / max(1e-6 * r["S_new"], 1e-9))
                else:
                    if d is not None and d.get("complete"):                     # after a transition: the stage's own collected set, not S_esc
                        r["S_stage_collected"] = d["S_direct"]
                    have_dir = False
                ei = ch.energy_identity(a)                                    # v5 (33.1 (1)): the uncompleted identity, 8.3 (4)'s criterion
                if ei is not None:
                    r["energy_identity_rel"] = ei["residual_rel"]; r["energy_identity_abs"] = ei["residual_abs"]
                    r["coverage_identity_rel"] = ei["coverage_residual_rel"]; r["memory_mismatch_rel"] = ei["memory_mismatch_rel"]
                    r["E_coll"] = ei["E_coll"]; r["E_x"] = ei["E_x"]
                    self.checks["energy_identity_max_rel"] = max(self.checks["energy_identity_max_rel"], abs(ei["residual_rel"]))
                    self.checks["coverage_identity_max_rel"] = max(self.checks["coverage_identity_max_rel"], abs(ei["coverage_residual_rel"]))
                    self.checks["memory_mismatch_max_rel"] = max(self.checks["memory_mismatch_max_rel"], abs(ei["memory_mismatch_rel"]))
                    self.checks["gram_energy_bound_rel_max"] = max(self.checks["gram_energy_bound_rel_max"], ei["gram_energy_bound_rel"])
                    self.checks["energy_identity_checkpoints"] += 1
                    cov_max = max(cov_max or 0.0, abs(ei["coverage_residual_rel"]))
                    # v5 (33.1 (1)): the stage's identity record for the error bound (per channel, energy units, unweighted)
                    key_ch = str(ch.kappa)
                    self.ident_stage[key_ch] = max(self.ident_stage.get(key_ch, 0.0), abs(ei["residual_abs"]))
                    self.ident_stage_rel_max = max(self.ident_stage_rel_max, abs(ei["residual_rel"]))
                    for key_ in ident:
                        ident[key_] += w * ei[key_]
                else:
                    have_ident = False
                # v6 (S1): the exact collar identity and the P07 slabs at every direct checkpoint
                ci = ch.collar_identity(a, t=self.t)
                r["collar_identity"] = ci
                if ci is not None:
                    have_ci = True
                    if not ci.get("complete"):
                        ci_complete = False
                    for key_ in ci_tot:
                        if ci.get(key_) is not None:
                            ci_tot[key_] += w * ci[key_]
                    if ci.get("residual") is not None:
                        self.checks["collar_identity_residual_max_rel"] = max(self.checks["collar_identity_residual_max_rel"], abs(ci["residual"]) / max(abs(ci["E_init"]), 1e-300))
                        self.checks["collar_identity_checkpoints"] += 1
                r["slab_p07"] = ch.cross_surface_p07(a, t=self.t)
                # v4 (31.4 (2), (9)): the mutual information across the trapping zero of the exact stage state
                cs = ch.cross_surface(a, t=self.t)
                r["cross_surface"] = cs
                for c_ in cs:
                    have_mi = True
                    mi_tot[c_["d"]] = mi_tot.get(c_["d"], 0.0) + w * c_["physical"]["I"]
            else:
                have_dir = False; have_ident = False
            row["per_channel"][str(ch.kappa)] = {k_: v_ for k_, v_ in r.items() if not isinstance(v_, (torch.Tensor,))}
        if direct_now and have_ident:
            self.ident_stage_n += 1
        L_now = (E_rad_st - self.E_rad_stage) / a.tau_s_abs                  # v3: the memory route's flux over the block
        self.E_rad_stage = E_rad_st; self.E_rad_total = self.E_rad_prev_stages + E_rad_st; self.E_diag_stage = E_diag_st
        self.E_collar_stage += 0.5 * (self.Lc_prev + Lc_tot) * a.tau_s_abs; self.Lc_prev = Lc_tot   # the collar current's time integral (stage)
        self.L_hist.append(L_now)
        # v5 (33.1 (1)): the weighted identity is the uncompleted form (the completed form is withdrawn)
        ident_rel = ((ident["E_init"] - ident["E_mem"] - ident["E_coll"] - ident["E_x"]) / max(abs(ident["E_init"]), 1e-300)) if have_ident else None
        # v4 (31.4 (9); R5, R8): the turnover test raw and against the combined uncertainty
        U_now = sum(ch.w * (ch.ent_bound + ch.ent_bound_gram) for ch in self.chans) + clip_tot
        dec = S_old_tot - S_tot
        est_block = bool(dec > inc_tot + clip_tot and dec > 0.0)
        if self.S_max is None or S_tot > self.S_max:
            self.S_max, self.t_S_max, self.U_at_max = S_tot, self.t, U_now
        est_run = bool((self.S_max - S_tot) > (self.U_at_max + U_now))
        self.turnover_established_any = self.turnover_established_any or est_run
        dm = OPS.device_memory(self.dev)
        if dm is not None:
            self.gpu_peak_run = max(self.gpu_peak_run, dm["peak_GB"])
        E_out0 = self.start_readouts.get("E_out0_weighted") if self.start_readouts else None
        row["collar_identity_total"] = ({k_: v_ for k_, v_ in ci_tot.items()} if have_ci else None)
        if have_ci:
            row["collar_identity_total"]["complete"] = ci_complete
            row["collar_identity_total"]["E_out_minus_E_out0"] = (ci_tot["E_out"] - E_out0) if E_out0 is not None else None
        row["S_esc_band"] = U_now                                            # v6 (S1): the summed error band (the ledger, labelled)
        row.update({"S_esc_total": S_tot, "S_direct_total": S_dir_tot if have_dir else None, "S_B_total": S_B_tot, "I_total": I_tot,
                    "turnover_total": bool(I_tot > S_B_tot + 1e-9), "turnover_raw": bool(I_tot > S_B_tot + 1e-9),
                    "decrease_block": dec, "uncertainty_block": inc_tot + clip_tot, "decrease_established_block": est_block,
                    "uncertainty_total": U_now, "S_max": self.S_max, "decrease_from_max": self.S_max - S_tot, "turnover_established_run": est_run,
                    "E_rad_stage": self.E_rad_stage, "E_rad_total": self.E_rad_total, "E_trunc_stage": E_trunc_tot,
                    "E_diag_stage": E_diag_st, "E_coherence_blocks_stage": E_diag_st - E_rad_st, "E_collar_integrated_stage": self.E_collar_stage,
                    "E_rad_minus_collar_stage": E_rad_st - self.E_collar_stage, "energy_identity_rel_total": ident_rel,
                    "coverage_identity_max_rel_block": cov_max,
                    "E_coll_total": ident["E_coll"] if have_ident else None, "E_x_total": ident["E_x"] if have_ident else None,
                    "MI_cross_total": {str(k_): v_ for k_, v_ in mi_tot.items()} if have_mi else None,
                    "L_collected": L_now, "L_collar": Lc_tot, "L_cont": levels(a, rs_k)["L_cont"], "E_mem_total": E_mem_tot,
                    "memory_rank_weighted": rank_tot, "wall_s": wall, "first_block_after_transition": first_after,
                    "uncollected_over_active_max": unc_ratio_max, "uncollected_content_F_max": unc_F_max, "uncollected_content_E_max_abs": unc_E_max,
                    "gpu_mem_GB": self.gpu_peak_run if dm is not None else None,
                    "gpu_peak_since_reset_GB": dm["peak_GB"] if dm is not None else None})
        self.curve.append(row)
        if self.n_block % max(1, a.direct_every) == 0 or self.n_block <= 3 or a.timing_blocks or first_after:
            log("stage %d block %d t %.1f r_s%s: S_esc %.5f (direct %s) S_B %.3e I %.3e turnover raw %s, established %s (block decrease %.3e against %.3e; from the maximum %.3e against %.3e); "
                "E_rad(stage, memory route) %.4e of dE %.4e (E_trunc,acc %.3e; E_diag %.4e; identity %s (33.1 (1), 1e-6), coverage %s; collar integral %.4e, difference %.3e); "
                "uncollected weight over the active norm %.3e (content ||.||_F %.3e, energy %.3e; 33.1 (4)); "
                "L %.3e (collar %.3e, cont %.3e); MI across r_t %s; rank %.0f; %.2f s" % (
                    self.k, self.n_block, self.t / a.rs, " (the first block after a transition, uncapped)" if first_after else "", S_tot,
                    ("%.5f" % S_dir_tot) if have_dir else "-", S_B_tot, I_tot, row["turnover_raw"], est_run,
                    dec, inc_tot + clip_tot, self.S_max - S_tot, self.U_at_max + U_now,
                    self.E_rad_stage, a.dE_frac * self.E, E_trunc_tot, E_diag_st, _f(ident_rel, "%.2e"), _f(cov_max, "%.2e"), self.E_collar_stage,
                    E_rad_st - self.E_collar_stage, unc_ratio_max, unc_F_max, unc_E_max, L_now, Lc_tot, row["L_cont"],
                    (", ".join("d %s: %.3e" % (k_, v_) for k_, v_ in sorted(mi_tot.items()))) if have_mi else "-", rank_tot, wall))
            # v6 (S1): the collar identity, the band, the collector's certificates, the P07 slabs
            cit = row.get("collar_identity_total") or {}
            p7 = "; ".join("kappa %s %s" % (k_, ", ".join("d %d %.4e/%.4e" % (x["d"], x["formed"]["I"], x["killing_ground"]["I"]) for x in (pc.get("slab_p07") or [])))
                           for k_, pc in row["per_channel"].items() if pc.get("slab_p07"))
            log("stage %d block %d t %.1f r_s (v6, %s): E_remove (mode ledger) %.5e, E_out - E_out(0) (collar, exact) %s, the collar integral (trapezoid) %.5e, M_chi %s, "
                "identity residual %s; S_esc %.5f with band %.3e (labels: the ledger); collected %s (smallest outside weight %s, run %s); remaining outside weight %s; "
                "all-history Gram %s; exact Z %s; P07 slabs (formed/ground) %s" % (
                    self.k, self.n_block, self.t / a.rs, a.collector, self.E_rad_stage, _f(cit.get("E_out_minus_E_out0"), "%.5e"), self.E_collar_stage,
                    _f(cit.get("M_chi"), "%.5e"), _f(cit.get("residual"), "%.3e"), S_tot, U_now,
                    ", ".join("kappa %s %s" % (k_, pc.get("m_coll")) for k_, pc in row["per_channel"].items()),
                    ", ".join("%s" % _f(pc.get("g_sel_min"), "%.6f") for pc in row["per_channel"].values()), _f(self.checks.get("g_sel_min"), "%.6f"),
                    ", ".join("%s" % _f(pc.get("weight_beyond_cut_uncollected"), "%.3e") for pc in row["per_channel"].values()),
                    _f(self.checks.get("gram_all_history_max"), "%.2e"), _f(self.checks.get("Z_identity_exact_max"), "%.2e"), p7 or "-"))
        return row

    def stage_verdict(self):
        """normal: E_rad >= dE; stall: late and the recent flux below stall_frac of its peak; cap: the stage duration cap.
        v6 (S1, --end-rule v6): "drained" when the collected flux |L| has stayed below end_flux_frac of the constant-luminosity comparator
        L_cont for end_window consecutive blocks after S_esc has fallen below end_entropy_frac of its peak; else the cap."""
        a = self.a
        dE = a.dE_frac * self.E
        t_stage = self.t - self.t_stage0
        if self.E_rad_stage >= dE:
            return "reached dE"
        if getattr(a, "end_rule", "v5") == "v6":
            row = self.curve[-1] if self.curve else None
            if row is not None and self.S_max is not None and self.S_max > 0:
                armed = row["S_esc_total"] < a.end_entropy_frac * self.S_max
                small = abs(row["L_collected"]) < a.end_flux_frac * row["L_cont"]
                if armed and self.end_armed_t is None:
                    self.end_armed_t = self.t
                self.end_count = (self.end_count + 1) if (armed and small) else 0
                if self.end_count >= a.end_window:
                    return "drained"
            if t_stage >= a.t_stage_max * a.rs:
                return "stage cap"
            return None
        n = a.stall_window
        if len(self.L_hist) >= n + 2:
            L_recent = float(np.mean(self.L_hist[-n:]))
            self.L_peak = max(self.L_peak, L_recent)
            if t_stage >= a.t_stall_min * a.rs and self.L_peak > 0 and L_recent < a.stall_frac * self.L_peak:
                return "stalled"
        if t_stage >= a.t_stage_max * a.rs:
            return "stage cap"
        return None

    def end_stage(self, reason):
        """the stage's row. v5 (33.1 (1)): the identity's largest relative residual over the stage's evaluated checkpoints, E_init,k
        (weighted over the channels), the stage's error bound sum_ch 16 kappa max |E_init - E_mem - E_coll - E_x| (energy units) and its
        accumulation over the stages, the charge sequence's error bound (None, and the accumulation marked incomplete, when a channel has
        no evaluated checkpoint in the stage)."""
        a = self.a
        t_stage = self.t - self.t_stage0
        rs_k = 2.0 * a.g * self.E
        rd = self.stage_readouts[-1] if self.stage_readouts else {}
        mi_end = {str(ch.kappa): ch.cross_surface(a, t=self.t) for ch in self.chans}                   # v4 (31.4 (2), (9))
        E_init_w = sum(ch.w * ch.rec["E_delta_init"] for ch in self.chans if ch.rec is not None)
        evaluated = all(str(ch.kappa) in self.ident_stage for ch in self.chans)
        bound = sum(ch.w * self.ident_stage[str(ch.kappa)] for ch in self.chans) if evaluated else None
        if evaluated:
            self.E_bound_acc += bound
        else:
            self.E_bound_incomplete = True
        row = {"k": self.k, "E_k": self.E, "rs_k": rs_k, "duration": t_stage, "duration_over_rs1": t_stage / a.rs, "E_rad": self.E_rad_stage,
               "dE_target": a.dE_frac * self.E, "L_mean": self.E_rad_stage / max(t_stage, 1e-300), "L_cont": levels(a, rs_k)["L_cont"],
               "L_peak_recent": self.L_peak, "end": reason, "n_blocks": self.n_block, "S_esc_end": self.curve[-1]["S_esc_total"] if self.curve else 0.0,
               "levels_now": levels(a, rs_k), "E_diag": self.E_diag_stage, "E_collar_integrated": self.E_collar_stage,
               "E_trunc": self.curve[-1].get("E_trunc_stage") if self.curve else None,
               "D8_bound_weighted": sum(ch.w * (ch.ent_bound + ch.ent_bound_gram) for ch in self.chans), "charge_rule": LABEL_RULE,
               "T_m_over_T_H": rd.get("T_m_over_T_H"), "kappa_m": rd.get("kappa_m"), "r_Hm": rd.get("r_Hm"),
               "L_comp_T_m": rd.get("L_comp_T_m"), "L_comp_T_H": rd.get("L_comp_T_H"), "cross_surface_end": mi_end,
               # v5 (33.1 (1))
               "identity_max_rel": self.ident_stage_rel_max if evaluated else None, "identity_checkpoints": self.ident_stage_n,
               "E_init_weighted": E_init_w, "energy_error_bound": bound, "energy_error_bound_acc": self.E_bound_acc,
               "energy_error_bound_acc_complete": not self.E_bound_incomplete}
        self.stages.append(row)
        if self.k == 0:                                                        # v3 (30.1 (6)): D7 read within stage 1
            self.partners_stage1 = {str(ch.kappa): ch.partner_readout(a) for ch in self.chans}
        log("stage %d ends (%s): emitted %.4e of %.4e over %.1f r_s (E_trunc,acc %s; the collar integral %.4e beside it); L_mean %.3e against the continuum thermal %.3e "
            "(all channels, T_H) and the run's channels' comparator at T_m %s (T_m/T_H %s) and at T_H %s; %s; the identity (33.1 (1)): largest residual %s over %d checkpoints, "
            "E_init,k %.4e, the stage's error bound %s, accumulated %.4e%s" % (
                self.k, reason, self.E_rad_stage, a.dE_frac * self.E, t_stage / a.rs, _f(row["E_trunc"], "%.3e"), self.E_collar_stage, row["L_mean"], row["L_cont"],
                _f(row["L_comp_T_m"], "%.3e"), _f(row["T_m_over_T_H"], "%.4f"), _f(row["L_comp_T_H"], "%.3e"), LABEL_RULE,
                _f(row["identity_max_rel"], "%.3e"), self.ident_stage_n, E_init_w, _f(bound, "%.3e"), self.E_bound_acc,
                "" if not self.E_bound_incomplete else " (incomplete: a stage without an evaluated identity)"))

    def do_transition(self, E_new, forced=False):
        """v5 (33.1 (2), (6)); v4 (31.4 (1)) with v3's 8.3 (8) checks (the accumulated D8 bound below 1 percent of S_esc, the adiabatic
        leak below 1e-4 over every old occupied mode, the S-continuity defect below 1e-6); on a failure the channels are restored and the
        transition repeated with tau_tr doubled, at most --tau-tr-doublings times (the last attempt is kept and flagged). v5 (33.1 (6)):
        the snapshot holds no old eigenvectors (Channel.transition_snapshot); a restore rebuilds the old stage (Channel.restore_snapshot,
        its exactness printed); the device peak is reset at the transition's start and read with torch.cuda.max_memory_allocated at its
        end, printed with the allocation at the start and the run's peak before it; every channel's memory phases are printed. v5 (33.1 (2)):
        the continuation line is page_core_v5's; only a flagged continuation enters the validation status."""
        a = self.a
        Rb_new = Rb_of(a, E_new)
        S_esc_now = self.curve[-1]["S_esc_total"] if self.curve else 0.0
        t_tr0 = time.time()
        peak_before = OPS.reset_device_peak(self.dev)                          # v5 (33.1 (6)): the transition's own peak from here
        if peak_before is not None:
            self.gpu_peak_run = max(self.gpu_peak_run, peak_before)
        dm0 = OPS.device_memory(self.dev)
        alloc_start = dm0["allocated_GB"] if dm0 is not None else None
        snaps = [ch.transition_snapshot(self.E, self.Rb_now) for ch in self.chans]
        bound_before = sum(ch.w * (ch.ent_bound + ch.ent_bound_gram) for ch in self.chans)
        factor = 1.0; attempts = []; restores = []
        for attempt in range(max(0, int(a.tau_tr_doublings)) + 1):
            trs = {}
            for ch in self.chans:
                tr = ch.transition(a, E_new, Rb_new, tau_factor=factor)
                trs[str(ch.kappa)] = tr
                c_ = tr["continuation"]; tm = tr["timings"]
                log("kappa %d transition (tau_tr x %g): r_s %.3f -> %.3f; %s; nsub %d (the 1/8 rule needs %s), ||dH||_G %.3e, gap %s, Weyl margin %s, "
                    "max dh/g %.3f, max h_dot/g^2 %.3f; R3 reference offset ||P^cont - 1(h<0)||_F^2 %s (pushed up %s, new pockets %s), ambiguous %s; "
                    "sea part ||U P U^+ - P^cont||_F %.3e (%s of ||Delta_0||_F; in the memory), leak max %.2e over %d occupied modes, sea unitarity %.1e, dropped-pair share %.2e; "
                    "FIFO %d entries (%d -> %d columns; %d emptied), half defect max %.2e, memory removed on it %.2e (share) %.3e (energy); compression E_trunc %.3e (share %.2e); "
                    "S continuity %.2e; rank %d; Chebyshev tail %.1e; walls: CF4 %.0f s, sub-step eighs %.0f s (%d), new stage %.0f s, principal angles %.0f s, total %.0f s" % (
                        ch.kappa, factor, tr["rs_old"], tr["rs_new"], tr["continuation_label"], c_["nsub"], c_["nsub_needed"], c_["norm_dH_G"], _f(c_["gap_old"], "%.3e"),
                        _f(c_["margin_min"], "%.3e"), c_["max_dh_over_gap"], c_["max_hdot_over_gap2"], tr["reference_offset_F2"], tr["released_pushed_up"],
                        tr["released_new_pockets"], tr["ambiguous_continuations"], tr["sea_coherence_F"], _f(tr["sea_coherence_over_Delta0"], "%.3e"),
                        tr["adiabatic_leak_max"], tr["n_sea"], tr["sea_unitarity"], tr["sea_dropped_pair_share"], tr["fifo_entries"], tr["fifo_columns_before"],
                        tr["fifo_columns_after"], tr["fifo_entries_emptied"], tr["fifo_half_defect_max"], tr["memory_removed_on_fifo_share"], tr["memory_removed_on_fifo_E"],
                        tr["compression_E_trunc"], tr["compression_dropped_share"], tr["S_continuity_defect"], tr["memory_rank"],
                        tr["chebyshev"]["tail_bound_acc"], tm["cf4_full_sea_and_memory_s"], tm["substep_eigh_s"], tm["substep_eighs"], tm["eigh_new_stage_s"],
                        tm["principal_angles_s"], tm["total_s"]))
                if any(m_.get("allocated_GB") is not None for m_ in tr["memory_phases"]):
                    log("kappa %d transition memory phases (33.1 (6); allocated / peak since the transition began, GB): %s" % (
                        ch.kappa, "; ".join("%s %.2f / %.2f" % (m_["phase"], m_["allocated_GB"], m_["peak_GB"]) for m_ in tr["memory_phases"])))
                if self.dev.type == "cuda":
                    torch.cuda.empty_cache()
            bound_w = sum(ch.w * (ch.ent_bound + ch.ent_bound_gram) for ch in self.chans)
            leak_max = max(t_["adiabatic_leak_max"] for t_ in trs.values())
            S_def = max(abs(t_["S_continuity_defect"]) for t_ in trs.values())
            ok_b = bound_w < 0.01 * S_esc_now; ok_l = leak_max < 1e-4; ok_s = S_def < 1e-6
            attempts.append({"tau_factor": factor, "D8_bound_weighted": bound_w, "S_esc": S_esc_now, "bound_over_S_esc": bound_w / max(S_esc_now, 1e-300),
                             "leak_max": leak_max, "S_continuity_defect_max": S_def, "pass": bool(ok_b and ok_l and ok_s),
                             "certified": {k_: t_["continuation"]["certified"] for k_, t_ in trs.items()}})
            log("transition check (8.3 (8)), tau_tr x %g: accumulated D8 bound %.3e nats against 1 percent of S_esc %.3e (%s); leak max %.2e against 1e-4 (%s); "
                "S-continuity defect %.2e against 1e-6 (%s); continuation %s" % (factor, bound_w, 0.01 * S_esc_now, "pass" if ok_b else "FAIL", leak_max, "pass" if ok_l else "FAIL",
                                                                   S_def, "pass" if ok_s else "FAIL",
                                                                   ", ".join("kappa %s %s" % (k_, "certified" if v_ else "flagged") for k_, v_ in attempts[-1]["certified"].items())))
            if (ok_b and ok_l and ok_s) or attempt >= int(a.tau_tr_doublings):
                break
            # v4: a D8 failure alone repeats the transition only when the transition's own increment is at least the excess (the earlier
            # blocks' compressions are not reduced by a longer ramp); the check stays failed and enters the validation status
            if ok_l and ok_s and not ok_b and (bound_w - bound_before) < (bound_w - 0.01 * S_esc_now):
                log("transition check: the accumulated D8 bound's excess %.3e is not the transition's own increment %.3e; a longer ramp cannot remove it, so tau_tr is not doubled (the check stays failed)" % (
                    bound_w - 0.01 * S_esc_now, bound_w - bound_before))
                attempts[-1]["no_doubling_reason"] = "the excess is the earlier blocks' compressions"
                break
            for ch, sn in zip(self.chans, snaps):                              # v5 (33.1 (6)): the old stage rebuilt, the state restored
                rinfo = ch.restore_snapshot(a, sn)
                restores.append({"kappa": ch.kappa, "tau_factor": factor, **rinfo})
                self.checks["restores"] += 1
                if not rinfo["exact"]:
                    self.checks["restores_inexact"] += 1
                log("kappa %d restored for tau_tr x %g (33.1 (6): the old stage rebuilt by its eigendecomposition, %.1f s): %s" % (
                    ch.kappa, 2.0 * factor, rinfo["seconds"], "bitwise the original stage (exact)" if rinfo["exact"] else
                    "not bitwise (max |de| %.3e, max |d colsum| %.3e): re-expressed from the site-basis copies" % (rinfo["max_dev_e"], rinfo["max_dev_colsum"])))
            factor *= 2.0
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        del snaps
        dm1 = OPS.device_memory(self.dev)
        tr_peak = dm1["peak_GB"] if dm1 is not None else None
        if tr_peak is not None:
            self.gpu_peak_run = max(self.gpu_peak_run, tr_peak)
            self.checks["transition_device_peak_GB_max"] = max(self.checks["transition_device_peak_GB_max"], tr_peak)
        tr_wall = time.time() - t_tr0
        tau_used = a.tau_tr * 2.0 * a.g * self.E * factor
        self.checks["D8_bound_over_S_esc_at_transitions_max"] = max(self.checks.get("D8_bound_over_S_esc_at_transitions_max", 0.0), attempts[-1]["bound_over_S_esc"])
        self.checks["transition_leak_max"] = max(self.checks["transition_leak_max"], attempts[-1]["leak_max"])
        self.checks["transition_S_continuity_max"] = max(self.checks["transition_S_continuity_max"], attempts[-1]["S_continuity_defect_max"])
        self.checks["transitions_flagged"] += sum(1 for v_ in attempts[-1]["certified"].values() if not v_)
        self.transitions.append({"k_from": self.k, "E_from": self.E, "E_to": E_new, "forced": forced, "per_channel": trs,
                                 "t": self.t, "tau_tr": tau_used, "tau_factor": factor, "attempts": attempts, "check_pass": attempts[-1]["pass"],
                                 "device_peak_GB": tr_peak, "device_allocated_start_GB": alloc_start, "run_peak_before_GB": peak_before,
                                 "wall_s": tr_wall, "restores": restores})
        log("transition %d -> %d (33.1 (6)): device peak %s GB (torch.cuda.max_memory_allocated from the transition's start; %s GB allocated at its start; the run's peak before it %s GB); "
            "wall %.1f s over %d attempt(s)" % (self.k, self.k + 1, _f(tr_peak, "%.3f"), _f(alloc_start, "%.3f"), _f(peak_before, "%.3f"), tr_wall, len(attempts)))
        self.t += tau_used
        self.E = E_new; self.Rb_now = Rb_new; self.k += 1
        self.E_rad_prev_stages += self.E_rad_stage                             # v3: the completed stage's memory-route emission
        self.t_stage0 = self.t; self.E_rad_stage = 0.0; self.E_diag_stage = 0.0; self.E_collar_stage = 0.0; self.L_hist = []; self.L_peak = 0.0
        self.ident_stage = {}; self.ident_stage_rel_max = 0.0; self.ident_stage_n = 0  # v5 (33.1 (1)): the new stage's identity record
        self.Lc_prev = sum(ch.w * ch.collar_flux(self.t) for ch in self.chans)
        self.stage_readout()
        self.stage_start_readouts()                                           # v6: E_out(0) and the P07 slabs at the new stage's start

    # -------------------------------------------------------------------------------------------------------------- loop
    def main_loop(self):
        a = self.a
        while True:
            self.block()
            if self.n_block % a.ckpt_every == 0:
                self.checkpoint(); self.write_outputs(final=False)
            if (time.time() - T_START) / 3600.0 > a.max_wall_hours:
                self.stop_reason = "wall clock cap"; self.end_stage("wall clock cap"); break
            verdict = self.stage_verdict()
            if verdict is None:
                continue
            self.end_stage(verdict)
            E_new = self.E - self.E_rad_stage
            if verdict != "reached dE":
                if a.force_steps:
                    E_new = self.E - a.dE_frac * self.E
                    log("forced step (labelled control, not the model's balance): E -> %.6f" % E_new)
                    forced = True
                else:
                    self.stop_reason = verdict; break
            else:
                forced = False
            if self.k + 1 >= a.max_stages or E_new <= a.E_stop_frac * a.E1 or 2.0 * a.g * E_new <= 1.05 * (self.Rb_now if a.interior == "fixed" else 0.0):
                self.stop_reason = "sequence end (stages %d, E/E1 %.4f)" % (self.k + 1, E_new / a.E1); break
            self.do_transition(E_new, forced=forced)
            self.checkpoint()
        self.finish()

    # --------------------------------------------------------------------------------------------- checkpoint, resume
    def checkpoint(self):
        a = self.a
        state = {"cfg": {k_: v_ for k_, v_ in vars(a).items() if not callable(v_)}, "k": self.k, "E": self.E, "t": self.t, "t_stage0": self.t_stage0,
                 "n_block": self.n_block, "E_rad_stage": self.E_rad_stage, "E_rad_total": self.E_rad_total, "L_hist": self.L_hist, "L_peak": self.L_peak,
                 "Rb_now": self.Rb_now, "curve": self.curve, "stages": self.stages, "transitions": self.transitions, "formation": self.formation,
                 "inversion": self.inversion, "checks": self.checks, "channels": {},
                 "E_rad_prev_stages": self.E_rad_prev_stages, "E_diag_stage": self.E_diag_stage, "E_collar_stage": self.E_collar_stage,
                 "Lc_prev": self.Lc_prev, "partners_stage1": self.partners_stage1, "timings": self.timings,
                 # v4
                 "S_max": self.S_max, "t_S_max": self.t_S_max, "U_at_max": self.U_at_max, "turnover_established_any": self.turnover_established_any,
                 "stage_readouts": self.stage_readouts, "quadrature": self.quadrature,
                 # v5 (33.1 (1), (6))
                 "ident_stage": self.ident_stage, "ident_stage_rel_max": self.ident_stage_rel_max, "ident_stage_n": self.ident_stage_n,
                 "E_bound_acc": self.E_bound_acc, "E_bound_incomplete": self.E_bound_incomplete, "gpu_peak_run": self.gpu_peak_run,
                 # v6 (S1)
                 "end_count": self.end_count, "end_armed_t": self.end_armed_t, "start_readouts": self.start_readouts}
        cqdir = self.prefix + "_cQ"
        for ch in self.chans:
            V = ch.stage["V"]
            D = (V @ ch.c).cpu()
            # v2: each FIFO mode is stored at its own collection time tq, the time saved beside it (the frame its timestamp names, R8.2)
            fifo = [((V @ cq).cpu(), hq.cpu(), tq) for cq, hq, tq in ch.fifo]
            rec = ch.rec
            recs = None
            if rec is not None:
                # v4 (R8.3, 31.4 (6)): the stage record's stored modes in per-stage chunk files, the site basis; the modes stored since the
                # last checkpoint are written, the list of files is the checkpoint's
                nb = len(rec["cQ"])
                if a.store_modes and nb > rec.get("cQ_saved", 0):
                    os.makedirs(cqdir, exist_ok=True)
                    fn = os.path.join(cqdir, "k%d_st%d_b%06d.pt" % (ch.kappa, ch.n_transitions, rec.get("cQ_saved", 0)))
                    torch.save([(V @ x).cpu() for x in rec["cQ"][rec.get("cQ_saved", 0):]], fn + ".tmp")
                    os.replace(fn + ".tmp", fn)
                    rec["cQ_files"] = list(rec.get("cQ_files", [])) + [(fn, nb - rec.get("cQ_saved", 0))]
                    rec["cQ_saved"] = nb
                recs = {"D_init": (V @ rec["c_init"]).cpu(), "half_init": None, "MD_init": rec["MD_init"].cpu(), "t_init": rec["t_init"],
                        "a": [x.cpu() for x in rec["a"]], "halfQ": [x.cpu() for x in rec["halfQ"]], "tQ": rec["tQ"], "n_coll": rec["n_coll"],
                        "E_rad": rec["E_rad"], "E_delta_init": rec["E_delta_init"], "n_coll_all": rec.get("n_coll_all", rec["n_coll"]),
                        "E_trunc": rec.get("E_trunc", 0.0), "gram_energy_bound": rec.get("gram_energy_bound", 0.0),
                        "cQ_files": list(rec.get("cQ_files", [])), "cQ_saved": rec.get("cQ_saved", 0), "cQ_from": rec.get("cQ_from", (0, 0)),
                        "gQ": [x.cpu() for x in rec.get("gQ", [])]}                    # v6 (S1 repair 3): the collected modes' outside weights
            rot = [(i, j, w_, V[:, i:j].cpu()) for (i, j, w_) in ch.stage.get("rot_blocks", [])]
            state["channels"][str(ch.kappa)] = {"D": D, "half": ch.half.cpu(), "MD": ch.MD.cpu(), "t_ref": ch.t_ref, "fifo": fifo, "rec": recs,
                                                 "ent_bound": ch.ent_bound, "n_transitions": ch.n_transitions, "norm_Delta0": ch.norm_Delta0,
                                                 # v4: the stage reference (31.4 (1)), its rotated blocks, the FIFO depth (31.4 (5)), the accumulators
                                                 "neg": ch.stage["neg"].cpu(), "reference": ch.stage.get("reference"), "rot_blocks": rot,
                                                 "ent_bound_gram": ch.ent_bound_gram, "fifo_depth": ch.fifo_depth, "fifo_doublings": ch.fifo_doublings,
                                                 "gram_energy_bound": ch.gram_energy_bound, "fifo_oldest_weight": ch.fifo_oldest_weight,
                                                 # v5 (33.1 (3), (4)): the FIFO's exit record, the blocks since the last transition
                                                 "fifo_exit_weight_max": ch.fifo_exit_weight_max, "fifo_exit_sum_max": ch.fifo_exit_sum_max,
                                                 "fifo_exits": ch.fifo_exits, "blocks_since_transition": ch.blocks_since_transition,
                                                 # v6 (S1 repair 3)
                                                 "gram_all_F2": ch.gram_all_F2, "gram_all_max": ch.gram_all_max, "g_sel_min": ch.g_sel_min}
        tmp = self.prefix + "_ckpt.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.prefix + "_ckpt.pt")

    def resume(self):
        a = self.a
        state = torch.load(self.prefix + "_ckpt.pt", map_location="cpu", weights_only=False)
        for key in ("k", "E", "t", "t_stage0", "n_block", "E_rad_stage", "E_rad_total", "L_hist", "L_peak", "Rb_now", "curve", "stages", "transitions",
                    "formation", "inversion", "checks"):
            setattr(self, key, state[key])
        for key in ("E_rad_prev_stages", "E_diag_stage", "E_collar_stage", "Lc_prev", "partners_stage1", "timings",          # v3
                    "S_max", "t_S_max", "U_at_max", "turnover_established_any", "stage_readouts", "quadrature",          # v4
                    "ident_stage", "ident_stage_rel_max", "ident_stage_n", "E_bound_acc", "E_bound_incomplete", "gpu_peak_run",   # v5
                    "end_count", "end_armed_t", "start_readouts"):                                                          # v6
            if key in state:
                setattr(self, key, state[key])
        for ch in self.chans:
            s = state["channels"][str(ch.kappa)]
            st = ch.build_stage(a, self.E, self.Rb_now, label="stage %d (resumed)" % (self.k + 1))
            V = st["V"]
            # v4: the stage reference and its rotated blocks first (every coefficient below is expressed against them)
            for (i, j, w_, Vb) in s.get("rot_blocks", []):
                W = V[:, i:j].mH @ Vb.to(self.dev)
                Uw, _, Whw = torch.linalg.svd(W)
                V[:, i:j] = V[:, i:j] @ (Uw @ Whw)
            st["rot_blocks"] = [(i, j, w_) for (i, j, w_, _) in s.get("rot_blocks", [])]
            if s.get("neg") is not None:
                st["neg"] = s["neg"].to(self.dev); st["nneg"] = int(st["neg"].sum())
            if s.get("reference"):
                st["reference"] = s["reference"]
            neg = st["neg"]

            def to_eig(Dcpu, half):
                c = V.mH @ Dcpu.to(self.dev)
                half = half.to(self.dev)
                for hsign in (1, -1):
                    cols = (half == hsign).nonzero().ravel()
                    if cols.numel():
                        rows = neg if hsign > 0 else ~neg
                        sub = c[:, cols]; sub[rows] = 0.0; c[:, cols] = sub
                nrm = torch.linalg.norm(c, dim=0).clamp_min(1e-300)
                return c / nrm[None, :].to(CD), half
            ch.c, ch.half = to_eig(s["D"], s["half"])
            ch.MD = s["MD"].to(self.dev); ch.t_ref = s["t_ref"]; ch.ent_bound = s["ent_bound"]
            ch.n_transitions = s.get("n_transitions", 0); ch.norm_Delta0 = s.get("norm_Delta0")
            ch.ent_bound_gram = s.get("ent_bound_gram", 0.0); ch.fifo_depth = s.get("fifo_depth"); ch.fifo_doublings = list(s.get("fifo_doublings", []))
            ch.gram_energy_bound = s.get("gram_energy_bound", 0.0); ch.fifo_oldest_weight = s.get("fifo_oldest_weight", 0.0)
            ch.fifo_exit_weight_max = s.get("fifo_exit_weight_max", 0.0); ch.fifo_exit_sum_max = s.get("fifo_exit_sum_max", 0.0)   # v5 (33.1 (3))
            ch.fifo_exits = s.get("fifo_exits", 0); ch.blocks_since_transition = s.get("blocks_since_transition")                    # v5 (33.1 (4))
            ch.gram_all_F2 = s.get("gram_all_F2", 0.0); ch.gram_all_max = s.get("gram_all_max", 0.0); ch.g_sel_min = s.get("g_sel_min", math.inf)   # v6
            ch.fifo = []
            for Dq, hq, tq in s["fifo"]:
                cq, hq2 = to_eig(Dq, hq)
                ch.fifo.append((cq, hq2, tq))
            r = s["rec"]
            if r is not None:
                # the stored D_init columns are split; recover their halves from the support
                c_raw = V.mH @ r["D_init"].to(self.dev)
                wneg = (c_raw.abs().pow(2) * neg.to(RD)[:, None]).sum(dim=0)
                hinit = torch.where(wneg > 0.5, -torch.ones_like(wneg, dtype=torch.int64), torch.ones_like(wneg, dtype=torch.int64))
                c_init, _ = to_eig(r["D_init"], hinit.cpu())
                # v4 (R8.3, 31.4 (6)): the stored collected modes from the chunk files, so the resumed record is complete (cQ_from (0, 0))
                cQ = []
                if a.store_modes:
                    blocks = []
                    for fn, nb in r.get("cQ_files", []):
                        blocks += torch.load(fn, map_location="cpu", weights_only=False)
                    for Dq, hq in zip(blocks, r["halfQ"]):
                        cq, _ = to_eig(Dq, hq)
                        cQ.append(cq)
                complete = (len(cQ) == len(r["tQ"])) and tuple(r.get("cQ_from", (0, 0))) == (0, 0)
                ch.rec = {"c_init": c_init, "MD_init": r["MD_init"].to(self.dev), "t_init": r["t_init"], "a": [x.to(self.dev) for x in r["a"]],
                          "cQ": cQ, "cQ_from": (0, 0) if complete else (len(r["tQ"]), r["n_coll"]),
                          "halfQ": [x.to(self.dev) for x in r["halfQ"]], "tQ": r["tQ"], "n_coll": r["n_coll"], "E_rad": r["E_rad"],
                          "E_delta_init": r["E_delta_init"], "n_coll_all": r.get("n_coll_all", r["n_coll"]), "E_trunc": r.get("E_trunc", 0.0),
                          "gram_energy_bound": r.get("gram_energy_bound", 0.0), "cQ_files": list(r.get("cQ_files", [])), "cQ_saved": r.get("cQ_saved", 0),
                          "gQ": [x.to(self.dev) for x in r.get("gQ", [])]}                                                    # v6
            else:
                ch.init_stage_record()
        log("resumed at stage %d, block %d, t %.1f r_s, E %.6f (the stage record's stored modes restored: %s)" % (
            self.k, self.n_block, self.t / a.rs, self.E, ", ".join("kappa %d %d blocks" % (ch.kappa, len(ch.rec["cQ"])) for ch in self.chans)))

    def recompute_pair_ledger(self):
        """v6 (37.1 (3)): at block 0 of a run started from a stored checkpoint, the dropped-pair ledger is recomputed from the stored
        sines under this run's --pair-ledger (the formation's other products do not depend on the repairs); printed."""
        a = self.a
        for ch in self.chans:
            irec = self.inversion.get(str(ch.kappa)) or {}
            sines = irec.get("dropped_sines")
            if sines is None:
                log("kappa %d: the stored inversion record has no dropped sines (a v5 checkpoint?): the ledger is left as stored (%.6e)" % (ch.kappa, ch.ent_bound))
                continue
            new = OPS.pair_ledger_sharp(sines) if a.pair_ledger == "sharp" else OPS.pair_ledger_v5(sines)
            log("kappa %d: the dropped-pair ledger at block 0 recomputed under --pair-ledger %s from %d stored sines: %.6e (stored %.6e; v5 %.6e, sharp %.6e)" % (
                ch.kappa, a.pair_ledger, len(sines), new, ch.ent_bound, OPS.pair_ledger_v5(sines), OPS.pair_ledger_sharp(sines)))
            ch.ent_bound = new
            irec["entropy_bound_dropped_pairs"] = new; irec["pair_ledger"] = a.pair_ledger; irec["ent_bound_after_formation"] = new

    # ------------------------------------------------------------------------------------------------------------ outputs
    def finish(self):
        a = self.a
        self.readouts = {}
        for ch in self.chans:
            d = ch.stage_direct(a, with_spectrum=True)
            self.readouts[str(ch.kappa)] = {"stage_direct": {k_: v_ for k_, v_ in (d or {}).items() if k_ != "spectrum_omega_over_TH_occupation_half"},
                                            "spectrum": (d or {}).get("spectrum_omega_over_TH_occupation_half"),
                                            "partners": ch.partner_readout(a), "cross_surface": ch.cross_surface(a, t=self.t), "ent_bound": ch.ent_bound,
                                            "ent_bound_gram": ch.ent_bound_gram, "partners_stage1": (self.partners_stage1 or {}).get(str(ch.kappa)),
                                            "fifo_doublings": ch.fifo_doublings, "fifo_depth": ch.fifo_depth}
        self.checkpoint()
        self.write_outputs(final=True)
        log("done: %s; validation status: %s; outputs %s_summary.json, _tables.md, _curve.npz" % (self.stop_reason, self.validation["status"], self.prefix))

    def validation_status(self):
        """v4 (R8, R11): "validated" when every 8.3 item this run evaluates passes, else "unresolved:" with the failing items; the items the run
        does not evaluate are named. (3) the formation; (4) the identities at production size; (8) the transitions; (2) blocks nothing; (5) to
        (7) are other runs. v5: (33.1 (1)) the energy identity is the uncompleted form below 1e-6 per stage; (33.1 (2)) a continuation enters
        only when flagged (a Weyl failure, dh/g above 1/8 or the nsub cap); (33.1 (3)) the FIFO's 1e-4 applies to the entries' inward weight
        at their exit (the oldest live entry's weight is a printed diagnostic); (33.1 (4)) the 1e-6 criterion on the uncollected weight is
        unchanged; (33.1 (5)) a clipping excursion at or below eps_rank is budgeted, not flagged."""
        a = self.a; c = self.checks
        fails = []; notev = ["8.3 (1) the self-test (its own run)", "8.3 (2) the scattering check (blocks nothing; its own run)",
                             "8.3 (5) the box and collection controls (other runs)", "8.3 (7) the interior, formation and averaging controls (other runs)"]
        for k_, f in self.formation.items():
            if f["unitarity"] >= 1e-10:
                fails.append("8.3 (3) formation unitarity kappa %s %.2e" % (k_, f["unitarity"]))
            if f.get("pockets_initial"):
                fails.append("8.3 (3) pockets at the initial configuration kappa %s (u_max %.4f)" % (k_, f["u_max_initial"]))
        for k_, i in self.inversion.items():
            if i["entropy_bound_dropped_pairs"] >= 1e-3:
                fails.append("8.3 (3) principal-angle truncation bound kappa %s %.3e nats" % (k_, i["entropy_bound_dropped_pairs"]))
        tests = [("Z_identity_max", "<", 1e-8, "8.3 (4) |S(B) - tr s(Z)|"), ("clip_excursion_max", "<=", float(a.eps_rank), "8.3 (4) clipping excursion above eps_rank (33.1 (5): at or below it budgeted)"),
                 ("Xi_min_eig_min", ">", -1e-10, "8.3 (4) smallest eigenvalue of Xi"), ("I_minus_DD_minus_Xi_min", ">", -1e-10, "8.3 (4) smallest eigenvalue of I - D D^+ - Xi"),
                 ("fifo_overlap_after_max", "<", 1e-4, "8.3 (4) FIFO overlap after the projection"),
                 ("uncollected_over_active_max", "<", 1e-6, "8.3 (4) uncollected weight beyond R_cut over the active norm"),
                 ("fifo_exit_weight_max", "<", 1e-4, "33.1 (3) the FIFO entries' inward weight at their exit")]
        if getattr(a, "collector", "v5") == "active":
            # v6 (S1, 37.1 (3)): the exact Z identity (pre-compression Z), the all-history Gram defect at every block, the collected modes'
            # outside weight; the uncollected weight is collection efficiency (printed, no criterion: the state stays in the memory); no FIFO
            tests = [("Z_identity_exact_max", "<", 1e-8, "S1 repair 4: |S(B) - tr s(Z)| with the pre-compression Z"),
                     ("clip_excursion_max", "<=", float(a.eps_rank), "8.3 (4) clipping excursion above eps_rank (33.1 (5): at or below it budgeted)"),
                     ("Xi_min_eig_min", ">", -1e-10, "8.3 (4) smallest eigenvalue of Xi"), ("I_minus_DD_minus_Xi_min", ">", -1e-10, "8.3 (4) smallest eigenvalue of I - D D^+ - Xi"),
                     ("gram_all_history_max", "<=", 1e-10, "37.1 (3): the all-history Gram defect of the cumulative collected set")]
            notev.append("the uncollected weight beyond R_cut (the active collector: collection efficiency, printed as outside_weight_remaining_max, no criterion)")
            notev.append("the FIFO items (the active collector keeps no FIFO)")
            gmin = c.get("g_sel_min")
            if gmin is not None and gmin < 1.0 - float(a.eps_loc) - 1e-12:
                fails.append("S1 repair 3: a collected mode's outside weight %.6f below 1 - eps_loc" % gmin)
        for key, op, thr, lab in tests:
            v = c.get(key, 0.0)
            if (op == "<" and not v < thr) or (op == ">" and not v > thr) or (op == "<=" and not v <= thr):
                fails.append("%s %.3e (criterion %s %g)" % (lab, v, op, thr))
        if c.get("direct_checkpoints", 0):
            if getattr(a, "collector", "v5") == "active":
                if not c.get("direct_revised_ratio_max", 0.0) <= 1.0:            # v6: 36.1 (3)'s revised criterion
                    fails.append("8.3 (4) direct route against the memory route: abs %.3e nats, the revised criterion max(1e-6 S, 1e-9) exceeded by %.3e (36.1 (3))" % (
                        c.get("direct_abs_max", 0.0), c.get("direct_revised_ratio_max", 0.0)))
            elif not c["direct_vs_memory_max_rel"] < 1e-6:
                fails.append("8.3 (4) direct route against the memory route %.3e (1e-6)" % c["direct_vs_memory_max_rel"])
        else:
            notev.append("8.3 (4) the direct route (no complete stage-1 record at a direct checkpoint)")
        if c.get("energy_identity_checkpoints", 0):
            if not c["energy_identity_max_rel"] < 1e-6:
                fails.append("8.3 (4) the energy identity (the uncompleted form, 33.1 (1)) %.3e (1e-6 per stage)" % c["energy_identity_max_rel"])
        else:
            notev.append("8.3 (4) the energy identity (no complete stage record at a direct checkpoint)")
        if 3 in a.channel_list and self.curve:
            last = self.curve[-1]; w3 = 48.0
            S3 = w3 * last["per_channel"]["3"]["S_new"]
            share = S3 / max(last["S_esc_total"], 1e-300)
            if not share < 0.05:
                fails.append("8.3 (6) kappa 3's share of S_esc %.3e (0.05)" % share)
        else:
            notev.append("8.3 (6) the channel cutoff (kappa 3 not run)")
        for tr in self.transitions:
            if not tr["check_pass"]:
                at = tr["attempts"][-1]
                fails.append("8.3 (8) transition %d -> %d at tau_tr x %g: D8 bound/S_esc %.3e, leak %.2e, S continuity %.2e" % (
                    tr["k_from"], tr["k_from"] + 1, tr["tau_factor"], at["bound_over_S_esc"], at["leak_max"], at["S_continuity_defect_max"]))
            for k_, t_ in tr["per_channel"].items():
                if not t_["continuation"]["certified"]:
                    fails.append("33.1 (2) transition %d -> %d kappa %s: %s" % (tr["k_from"], tr["k_from"] + 1, k_, t_["continuation_label"]))
        if not self.transitions:
            notev.append("8.3 (8) the transitions (none occurred)")
        status = "validated" if not fails else "unresolved: " + "; ".join(fails)
        self.validation = {"status": status, "failed": fails, "not_evaluated": notev}
        return self.validation

    def write_outputs(self, final):
        a = self.a
        lv1 = levels(a, a.rs)
        val = self.validation_status()
        summary = {"settings": {k_: v_ for k_, v_ in vars(a).items() if not callable(v_)},
                   "derived": {"E1": a.E1, "T_H": a.TH, "kappa_surface": a.kappa_surf, "m": a.m, "m_over_TH": a.m_over_TH, "collapse_m_over_TH": 160.0,
                               "levels_initial": lv1, "L_cont_rs2": L_CONT_RS2, "b_cont": B_CONT, "device": str(self.dev)},
                   "formation": self.formation, "inversion": self.inversion, "stages": self.stages, "transitions": self.transitions,
                   "checks": self.checks, "stop_reason": self.stop_reason, "n_blocks": self.n_block, "wall_s": time.time() - T_START,
                   "final": final, "readouts": getattr(self, "readouts", None), "curve": self.curve, "timings": self.timings,
                   "label": "timing, not the pilot" if a.timing_blocks else (LABEL_SEQUENCE if a.max_stages > 1 else label_s1(a)),
                   "label_s1": label_s1(a), "code": "page_recursion_v6.py",
                   "repairs": {"hull": a.hull, "pair_ledger": a.pair_ledger, "collector": a.collector, "eps_loc": a.eps_loc, "direct_every": a.direct_every,
                               "end_rule": a.end_rule, "label": "S1's four repairs as switches (37.1 (2)); the regression attributes each change to one"},
                   "start_readouts": self.start_readouts, "end_rule_state": {"end_count": self.end_count, "armed_at_t": self.end_armed_t,
                                                                              "end_flux_frac": a.end_flux_frac, "end_window": a.end_window, "end_entropy_frac": a.end_entropy_frac},
                   "label_clock": LABEL_CLOCK, "label_rule": LABEL_RULE,
                   "label_levels": LABEL_LEVELS, "validation": val, "stage_readouts": self.stage_readouts, "quadrature_check": self.quadrature,
                   "turnover": {"S_max": self.S_max, "t_S_max": self.t_S_max, "U_at_max": self.U_at_max, "established_any": self.turnover_established_any},
                   # v5 (33.1 (1), (6)): the charge sequence's error bound; the device peaks (the run's, and per transition from its start)
                   "energy_error_bound": {"accumulated": self.E_bound_acc, "complete": not self.E_bound_incomplete,
                                          "label": "sum over stages of sum_ch 16 kappa max |E_init - E_mem - E_coll - E_x| over the stage's evaluated checkpoints (33.1 (1))"},
                   "device": {"gpu_peak_run_GB": self.gpu_peak_run if self.dev.type == "cuda" else None,
                              "transitions": [{"k_from": t_["k_from"], "device_peak_GB": t_.get("device_peak_GB"), "device_allocated_start_GB": t_.get("device_allocated_start_GB"),
                                               "run_peak_before_GB": t_.get("run_peak_before_GB"), "wall_s": t_.get("wall_s"), "attempts": len(t_.get("attempts", [])),
                                               "restores": t_.get("restores")} for t_ in self.transitions],
                              "label": "33.1 (6): torch.cuda.max_memory_allocated, reset at each transition's start"}}
        with open(self.prefix + "_summary.json", "w") as f:
            json.dump(summary, f, default=_jsonable, indent=1)
        self.write_npz(val)
        self.write_md(summary)

    def write_npz(self, val):
        a = self.a
        rows = self.curve
        if not rows:
            return
        arr = {"t": np.array([r["t"] for r in rows]), "t_inf": np.array([r["t_inf"] for r in rows]), "stage": np.array([r["stage"] for r in rows]),
               "E": np.array([r["E"] for r in rows]), "S_esc_total": np.array([r["S_esc_total"] for r in rows]),
               "S_direct_total": np.array([r["S_direct_total"] if r["S_direct_total"] is not None else np.nan for r in rows]),
               "S_B_total": np.array([r["S_B_total"] for r in rows]), "I_total": np.array([r["I_total"] for r in rows]),
               "E_rad_total": np.array([r["E_rad_total"] for r in rows]), "L_collected": np.array([r["L_collected"] for r in rows]),
               "L_collar": np.array([r["L_collar"] for r in rows]), "L_cont": np.array([r["L_cont"] for r in rows]),
               "memory_rank_weighted": np.array([r["memory_rank_weighted"] for r in rows]),
               "E_rad_stage": np.array([r["E_rad_stage"] for r in rows]), "E_diag_stage": np.array([r["E_diag_stage"] for r in rows]),
               "E_collar_integrated_stage": np.array([r["E_collar_integrated_stage"] for r in rows]),
               "A_over_4G_initial": levels(a, a.rs)["A_over_4G"], "eta_A_over_a2_initial": levels(a, a.rs)["eta_A_over_a2"],
               "memory_budget_initial": levels(a, a.rs)["memory_budget_modes"], "rs1": a.rs, "g": a.g, "eta": a.eta, "TH": a.TH, "E1": a.E1,
               # v4 (31.4 (9); R4, R5, R8, R11): the uncertainty, the turnover test, the truncation, the labels and the validation status
               "E_trunc_stage": np.array([r.get("E_trunc_stage", np.nan) for r in rows]),
               "uncertainty_total": np.array([r.get("uncertainty_total", np.nan) for r in rows]),
               "turnover_raw": np.array([bool(r.get("turnover_raw", False)) for r in rows]),
               "decrease_established_block": np.array([bool(r.get("decrease_established_block", False)) for r in rows]),
               "turnover_established_run": np.array([bool(r.get("turnover_established_run", False)) for r in rows]),
               "MI_cross_d2_total": np.array([(r.get("MI_cross_total") or {}).get("2", np.nan) for r in rows]),
               "m": a.m, "m_over_TH": a.m_over_TH, "channels": np.array(a.channel_list),
               "validation_status": np.array(val["status"]), "label_sequence": np.array(LABEL_SEQUENCE if not a.timing_blocks else "timing, not the pilot"),
               "label_clock": np.array(LABEL_CLOCK), "label_levels": np.array(LABEL_LEVELS)}
        for ch in self.chans:
            key = str(ch.kappa)
            arr["S_esc_kappa%s" % key] = np.array([r["per_channel"][key]["S_new"] for r in rows])
            arr["rank_kappa%s" % key] = np.array([r["per_channel"][key]["r_after"] for r in rows])
        arr["stage_E"] = np.array([s["E_k"] for s in self.stages]); arr["stage_duration"] = np.array([s["duration"] for s in self.stages])
        arr["stage_L_mean"] = np.array([s["L_mean"] for s in self.stages]); arr["stage_E_rad"] = np.array([s["E_rad"] for s in self.stages])
        arr["stage_L_comp_T_m"] = np.array([s.get("L_comp_T_m") if s.get("L_comp_T_m") is not None else np.nan for s in self.stages])
        arr["stage_T_m_over_T_H"] = np.array([s.get("T_m_over_T_H") if s.get("T_m_over_T_H") is not None else np.nan for s in self.stages])
        # v5 (33.1 (1), (6))
        arr["stage_identity_max_rel"] = np.array([s.get("identity_max_rel") if s.get("identity_max_rel") is not None else np.nan for s in self.stages])
        arr["stage_energy_error_bound"] = np.array([s.get("energy_error_bound") if s.get("energy_error_bound") is not None else np.nan for s in self.stages])
        arr["stage_energy_error_bound_acc"] = np.array([s.get("energy_error_bound_acc", np.nan) for s in self.stages])
        arr["transition_device_peak_GB"] = np.array([t_.get("device_peak_GB") if t_.get("device_peak_GB") is not None else np.nan for t_ in self.transitions])
        # v6 (S1): the band, the collar identity, the P07 slabs (d 2 per channel, formed and ground)
        arr["S_esc_band"] = np.array([r.get("S_esc_band", np.nan) for r in rows])
        cit = [r.get("collar_identity_total") or {} for r in rows]
        for key_ in ("E_out", "E_remove", "M_chi", "residual", "E_out_minus_E_out0", "E_remove_exact"):
            arr["collar_%s" % key_] = np.array([x.get(key_) if x.get(key_) is not None else np.nan for x in cit])
        for ch in self.chans:
            key = str(ch.kappa)
            for which in ("formed", "killing_ground"):
                arr["slab_p07_d2_%s_kappa%s" % (which, key)] = np.array([next((x[which]["I"] for x in (r["per_channel"][key].get("slab_p07") or []) if x["d"] == 2), np.nan) for r in rows])
            arr["m_coll_kappa%s" % key] = np.array([r["per_channel"][key].get("m_coll", 0) for r in rows])
            arr["g_sel_min_kappa%s" % key] = np.array([r["per_channel"][key].get("g_sel_min") if r["per_channel"][key].get("g_sel_min") is not None else np.nan for r in rows])
        arr["label_s1"] = np.array(label_s1(a))
        arr["transition_wall_s"] = np.array([t_.get("wall_s", np.nan) for t_ in self.transitions])
        np.savez_compressed(self.prefix + "_curve.npz", **arr)

    def write_md(self, summary):
        a = self.a
        lv = levels(a, a.rs)
        val = summary["validation"]
        L = []
        L.append("# %s: the Page curve by the retained-channel recursion (%s; %s)\n" % (a.tag, "final" if summary["final"] else "running", summary.get("label")))
        L.append("Validation status (8.3 thresholds as evaluated; R8, R11): **%s**. Not evaluated by this run: %s.\n" % (val["status"], "; ".join(val["not_evaluated"])))
        L.append("Settings: r_s/a %g, G/a^2 %g, m a %g (m/T_H = 4 pi m r_s = %.3f beside the collapse's 160), channels %s (weights 16 kappa), n_sites %d, ell %g (%s), ball R_b %g r_s (%s), formation %s from %g r_s, R_cut %g r_s, tau_s %g r_s, dE/E %g, eta %g, device %s. Exterior: %s (no certified solved exterior exists)." % (
            a.rs, a.g, a.m, a.m_over_TH, a.channel_list, a.nsites, a.ell, a.fields, a.Rb, a.interior, a.formation, a.Rb0, a.R_cut, a.tau_s, a.dE_frac, a.eta, summary["derived"]["device"], LABEL_SEQUENCE))
        L.append("Derived: E a %.4f, T_H a %.4e, kappa a %.4e, A/(4G) %.1f, eta A/a^2 %.1f, memory budget %.0f modes, L_cont %.4e (0.0052373/r_s^2), S_dot_cont %.4e per unit time (b = 1.6391).\n" % (
            a.E1, a.TH, a.kappa_surf, lv["A_over_4G"], lv["eta_A_over_a2"], lv["memory_budget_modes"], lv["L_cont"], lv["S_dot_cont"]))
        L.append("Labels: %s. %s. %s.\n" % (LABEL_CLOCK, LABEL_RULE, LABEL_LEVELS))
        # v6 (S1)
        rp = summary.get("repairs") or {}
        L.append("S1's label: **%s**.\n" % summary.get("label_s1"))
        L.append("Repairs (37.1 (2); each a switch): hull %s (repair 1), dropped-pair ledger %s (repair 2), collector %s with eps_loc %g (repair 3), the direct route every %s block(s) with the exact Z identity (repair 4); end rule %s (S1: |L| below %g of L_cont for %d blocks after S_esc below %g of its peak, else t_stage_max).\n" % (
            rp.get("hull"), rp.get("pair_ledger"), rp.get("collector"), rp.get("eps_loc", float("nan")), rp.get("direct_every"), rp.get("end_rule"),
            a.end_flux_frac, a.end_window, a.end_entropy_frac))
        sr = summary.get("start_readouts") or {}
        for k_, pc in (sr.get("per_channel") or {}).items():
            ci = pc.get("collar_identity") or {}
            L.append("- Stage start (t 0), kappa %s: E_out(0) beyond R_collar %s, tr[h Delta] %s, M_chi(0) %s; the P07 slabs (per unweighted copy; formed / Killing ground / excess; bound 4||K||_F^2 of the formed state): %s; the code's own (physical / vacuum): %s" % (
                k_, _f(ci.get("E_out"), "%.6e"), _f(ci.get("E_tot"), "%.6e"), _f(ci.get("M_chi"), "%.6e"),
                "; ".join("d %d (X %s, Y %s): %.8f / %.8f / %+.3e (bound %.3e)" % (x["d"], x["X_R"], x["Y_R"], x["formed"]["I"], x["killing_ground"]["I"], x["excess"], x["formed"]["bound_4KF2"])
                          for x in (pc.get("slab_p07") or [])) or "-",
                "; ".join("d %d: %.8f / %.8f" % (x["d"], x["physical"]["I"], x["reference_vacuum"]["I"]) for x in (pc.get("cross_surface_own") or [])) or "-"))
        L.append("")
        if summary.get("quadrature_check"):
            q = summary["quadrature_check"]
            L.append("Averaging quadrature check (R6): max |v(ell/16, 8 ell) - v(ell/32, 10 ell)| %.3e; l = 0 norms in the bulk within %s of 1.\n" % (q["max_abs_dv"], _f(q.get("max_abs_norm0_minus_1_bulk"), "%.3e")))
        L.append("## Formation and inversion per channel\n")
        L.append("| kappa | t_form/r_s | n0 | unitarity | Chebyshev tail bound | Delta_0 rank | memory rank | N+ | N- | E_inv/E | E_inv by bin of T_H [0,1,2,3,4,6,8,12,25,inf) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k_, f in summary["formation"].items():
            i = summary["inversion"].get(k_, {})
            L.append("| %s | %.3f | %d | %.2e | %.2e | %s | %s | %.4f | %.4f | %.3e | %s |" % (k_, f["t_form_over_rs"], f["n0"], f["unitarity"], (f.get("chebyshev") or {}).get("tail_bound_acc", float("nan")),
                                                                                     i.get("rank_kept"), i.get("memory_rank"), i.get("N_plus", 0), i.get("N_minus", 0), i.get("E_inv_over_E", 0),
                                                                                     ["%.3f" % x for x in i.get("inverted_by_bin_TH", [])]))
        if summary["formation"]:
            L.append("\nPreparation (R3): %s; the stage-1 reference: %s.\n" % (list(summary["formation"].values())[0].get("preparation"), list(summary["inversion"].values())[0].get("reference")))
        if summary.get("timings"):
            L.append("Wall times per channel (%s): %s\n" % (summary.get("label"), json.dumps(summary["timings"], default=_jsonable)))
        L.append("\n## Stage table (E_rad by the memory route; %s; E_trunc,acc beside it; the collar integral beside it)\n" % LABEL_RULE)
        L.append("| k | E_k | r_s,k | duration/r_s1 | E_rad | E_trunc,acc | E_diag | collar integral | dE target | L_mean | L_cont (all, T_H) | T_m/T_H | L comparator at T_m (run's channels) | at T_H | end | S_esc at end | D8 bound (weighted) | A/(4G) now | eta A/a^2 now | identity max rel (33.1 (1)) | checkpoints | E_init,k (weighted) | the stage's energy error bound | accumulated (the charge sequence's error bound) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in summary["stages"]:
            L.append("| %d | %.5f | %.4f | %.1f | %.4e | %s | %.4e | %.4e | %.4e | %.3e | %.3e | %s | %s | %s | %s | %.4f | %.3e | %.1f | %.1f | %s | %s | %s | %s | %s%s |" % (
                s["k"], s["E_k"], s["rs_k"], s["duration_over_rs1"], s["E_rad"], _f(s.get("E_trunc"), "%.3e"), s.get("E_diag", float("nan")), s.get("E_collar_integrated", float("nan")),
                s["dE_target"], s["L_mean"], s["L_cont"], _f(s.get("T_m_over_T_H"), "%.4f"), _f(s.get("L_comp_T_m"), "%.3e"), _f(s.get("L_comp_T_H"), "%.3e"), s["end"],
                s["S_esc_end"], s.get("D8_bound_weighted", float("nan")), s["levels_now"]["A_over_4G"], s["levels_now"]["eta_A_over_a2"],
                _f(s.get("identity_max_rel"), "%.3e"), _f(s.get("identity_checkpoints"), "%d"), _f(s.get("E_init_weighted"), "%.4e"), _f(s.get("energy_error_bound"), "%.3e"),
                _f(s.get("energy_error_bound_acc"), "%.3e"), "" if s.get("energy_error_bound_acc_complete", True) else " (incomplete)"))
        if summary.get("stage_readouts"):
            L.append("\nFrozen stage records (31.4 (9)): " + "; ".join("stage %d r_H,m %s kappa_m %s T_m/T_H %s" % (r_["k"], _f(r_["r_Hm"], "%.3f"), _f(r_["kappa_m"], "%.5e"), _f(r_["T_m_over_T_H"], "%.4f"))
                                                                    for r_ in summary["stage_readouts"]) + ".")
        L.append("\n## Transitions (31.4 (1): the full sea through the ramp, the continued reference, the new memory C_exact - P^cont; 33.1 (2) the continuation line; 33.1 (6) the lower-memory transition and its device peak)\n")
        for tr in summary["transitions"]:
            L.append("- stage %d -> %d: device peak %s GB (torch.cuda.max_memory_allocated from the transition's start), %s GB allocated at its start, the run's peak before it %s GB; wall %s s; restores %s" % (
                tr["k_from"], tr["k_from"] + 1, _f(tr.get("device_peak_GB"), "%.3f"), _f(tr.get("device_allocated_start_GB"), "%.3f"), _f(tr.get("run_peak_before_GB"), "%.3f"),
                _f(tr.get("wall_s"), "%.1f"), json.dumps(tr.get("restores") or [], default=_jsonable)))
            for k_, t in tr["per_channel"].items():
                c_ = t["continuation"]
                L.append("- stage %d -> %d, kappa %s: r_s %.4f -> %.4f; %s; nsub %d (needed %s), ||dH||_G %.3e, gap %s, Weyl margin %s, max dh/g %.3f, max h_dot/g^2 %.3f; "
                         "reference offset %s (pushed up %s, new pockets %s), ambiguous %s; sea part %.3e (%s of ||Delta_0||_F, in the memory), leak max %.2e, dropped-pair share %.2e (bound %.2e nats); "
                         "FIFO %d entries, half defect %.2e, memory removed on it %.2e; compression E_trunc %.3e (share %.2e); S continuity %.2e; rank %d; walls CF4 %.0f s, sub-step eighs %.0f s, principal angles %.0f s%s" % (
                    tr["k_from"], tr["k_from"] + 1, k_, t["rs_old"], t["rs_new"], t["continuation_label"], c_["nsub"], c_["nsub_needed"], c_["norm_dH_G"], _f(c_["gap_old"], "%.3e"),
                    _f(c_["margin_min"], "%.3e"), c_["max_dh_over_gap"], c_["max_hdot_over_gap2"], t["reference_offset_F2"], t["released_pushed_up"], t["released_new_pockets"],
                    t["ambiguous_continuations"], t["sea_coherence_F"], _f(t["sea_coherence_over_Delta0"], "%.3e"), t["adiabatic_leak_max"], t["sea_dropped_pair_share"],
                    t["sea_entropy_bound_dropped_pairs"], t["fifo_entries"], t["fifo_half_defect_max"], t["memory_removed_on_fifo_share"], t["compression_E_trunc"],
                    t["compression_dropped_share"], t["S_continuity_defect"], t["memory_rank"], t["timings"]["cf4_full_sea_and_memory_s"], t["timings"]["substep_eigh_s"],
                    t["timings"]["principal_angles_s"], " (forced step, labelled control)" if tr["forced"] else ""))
            for at in tr.get("attempts", []):
                L.append("  - 8.3 (8) check at tau_tr x %g: accumulated D8 bound %.3e against 1 percent of S_esc %.3e; leak max %.2e (1e-4); S-continuity defect %.2e (1e-6): %s" % (
                    at["tau_factor"], at["D8_bound_weighted"], 0.01 * at["S_esc"], at["leak_max"], at["S_continuity_defect_max"], "pass" if at["pass"] else "FAIL"))
        L.append("\n## v6 (S1): per checkpoint the collar identity, the error band and the collector's certificates (weighted; per channel in the summary; %s)\n" % summary.get("label_s1"))
        L.append("| t/r_s1 | S_esc total | band (the ledger) | E_remove (mode ledger) | E_out - E_out(0) (collar, exact) | collar integral (trapezoid) | M_chi | identity residual | collected per channel | smallest outside weight per channel | remaining outside weight per channel | P07 slab d 2, formed / ground, per channel |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows6 = summary["curve"]
        step6 = 1 if len(rows6) <= 200 else 10
        for r in rows6[::step6]:
            cit = r.get("collar_identity_total") or {}
            pcs = r["per_channel"]
            L.append("| %.1f | %.6f | %.3e | %.5e | %s | %.5e | %s | %s | %s | %s | %s | %s |" % (
                r["t_over_rs1"], r["S_esc_total"], r.get("S_esc_band", float("nan")), r["E_rad_stage"], _f(cit.get("E_out_minus_E_out0"), "%.5e"),
                r["E_collar_integrated_stage"], _f(cit.get("M_chi"), "%.5e"), _f(cit.get("residual"), "%.2e"),
                ", ".join("%s: %s" % (k_, pc.get("m_coll")) for k_, pc in pcs.items()),
                ", ".join("%s: %s" % (k_, _f(pc.get("g_sel_min"), "%.6f")) for k_, pc in pcs.items()),
                ", ".join("%s: %s" % (k_, _f(pc.get("weight_beyond_cut_uncollected"), "%.3e")) for k_, pc in pcs.items()),
                "; ".join("%s: %s" % (k_, " / ".join("%.5f" % x[w_]["I"] for x in (pc.get("slab_p07") or []) if x["d"] == 2 for w_ in ("formed", "killing_ground")) or "-") for k_, pc in pcs.items())))
        L.append("\n## The curve (every row while there are at most 200, else every 10th; %s)\n" % LABEL_CLOCK)
        L.append("| t/r_s1 | t_inf/r_s1 | stage | E/E1 | S_esc total | S_direct | S_B total | I total | turnover raw | block decrease established | decrease from max established | uncertainty | E_rad total | E_rad stage | E_trunc stage | identity (33.1 (1)) | coverage (printed) | MI across r_t (d 2) | collar integral (stage) | E_rad - collar | L collected | L collar | L_cont | rank (weighted) | s per block |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = summary["curve"]
        step = 1 if len(rows) <= 200 else 10
        for r in rows[::step]:
            L.append("| %.1f | %.1f | %d | %.5f | %.5f | %s | %.3e | %.3e | %s | %s | %s | %.2e | %.4e | %.4e | %s | %s | %s | %s | %.4e | %.3e | %.3e | %.3e | %.3e | %.0f | %.2f |" % (
                r["t_over_rs1"], r["t_inf"] / a.rs, r["stage"], r["E_over_E1"], r["S_esc_total"], ("%.5f" % r["S_direct_total"]) if r["S_direct_total"] is not None else "-",
                r["S_B_total"], r["I_total"], r.get("turnover_raw"), r.get("decrease_established_block"), r.get("turnover_established_run"), r.get("uncertainty_total", float("nan")),
                r["E_rad_total"], r["E_rad_stage"], _f(r.get("E_trunc_stage"), "%.3e"), _f(r.get("energy_identity_rel_total"), "%.2e"), _f(r.get("coverage_identity_max_rel_block"), "%.2e"),
                _f((r.get("MI_cross_total") or {}).get("2"), "%.3e"), r["E_collar_integrated_stage"], r["E_rad_minus_collar_stage"], r["L_collected"], r["L_collar"], r["L_cont"],
                r["memory_rank_weighted"], r["wall_s"]))
        L.append("\n## Saturation comparison (%s)\n" % LABEL_LEVELS)
        if rows:
            smax = max(r["S_esc_total"] for r in rows)
            L.append("Largest S_esc total %.4f nats at t %.1f r_s1; against eta A/a^2 (initial) %.1f: ratio %.4e; against A/(4G) (initial) %.1f: ratio %.4e. The thermal comparator over the same time would give S_dot_cont t = %.4f nats (continuum, stationary channel; not an input). Turnover: the run's decrease from its maximum is %s above the combined uncertainty (R5, R8)." % (
                smax, [r for r in rows if r["S_esc_total"] == smax][0]["t_over_rs1"], lv["eta_A_over_a2"], smax / lv["eta_A_over_a2"], lv["A_over_4G"], smax / lv["A_over_4G"], lv["S_dot_cont"] * rows[-1]["t"],
                "established" if summary["turnover"]["established_any"] else "never"))
        L.append("\n## Checks\n")
        labels = {"energy_identity_max_rel": "8.3 (4): the uncompleted identity |tr[Delta_k h_k] - (E_mem + E_coll + E_x)|/|E_init|, below 1e-6 per stage (33.1 (1); the completed form withdrawn)",
                  "coverage_identity_max_rel": "the coverage identity (the exact energy in the memory's span in place of E_mem), printed, no criterion (33.1 (1))",
                  "memory_mismatch_max_rel": "E_mem against the exact energy in the memory's span, printed, no criterion",
                  "energy_identity_checkpoints": "checkpoints with the full identity evaluated",
                  "clip_excursion_max": "8.3 (4): flagged above eps_rank, budgeted at or below it (33.1 (5))",
                  "fifo_exit_weight_max": "33.1 (3): the FIFO entries' inward weight at their exit, below 1e-4 (below the prune threshold by construction)",
                  "fifo_exits": "33.1 (3): entries that left the FIFO (count)",
                  "uncollected_content_F_max": "33.1 (4): the uncollected beyond-cut directions' content ||.||_F, printed",
                  "uncollected_content_E_max_abs": "33.1 (4): the uncollected content's energy (largest magnitude), printed",
                  "transition_device_peak_GB_max": "33.1 (6): the largest transition device peak (GB, torch.cuda.max_memory_allocated from the transition's start)",
                  "restores": "33.1 (6): restores of the old stage after a failed 8.3 (8) check (count)",
                  "restores_inexact": "33.1 (6): restores whose rebuilt stage was not bitwise the original (count)",
                  "fifo_overlap_after_max": "8.3 (4): below 1e-4 (30.1 (6))",
                  "fifo_overlap_max": "the candidates' redundancy before the projection, printed (30.1 (6))",
                  "fifo_oldest_weight_max": "the oldest live FIFO entry's inward weight, a printed depth diagnostic (33.1 (3); no criterion)",
                  "post_split_inward_weight_max": "R5: the collected modes' largest weight inside R_cut after the spectral split, printed",
                  "uncollected_over_active_max": "8.3 (4): the uncollected weight beyond R_cut over the active norm, below 1e-6",
                  "D8_bound_over_S_esc_at_transitions_max": "8.3 (8): below 0.01 at every transition (30.1 (4))",
                  "transition_leak_max": "8.3 (8): below 1e-4 (v4: over every old occupied mode)",
                  "transition_S_continuity_max": "8.3 (8): below 1e-6",
                  "transitions_flagged": "33.1 (2): continuations flagged (a Weyl failure, dh/g above 1/8 or the nsub cap; count over channels)",
                  "fifo_depth_doublings": "31.4 (5): FIFO depth doublings (count)"}
        for k_, v_ in summary["checks"].items():
            L.append("- %s: %.3e%s" % (k_, v_, ("  (" + labels[k_] + ")") if k_ in labels else ""))
        if summary.get("readouts"):
            for k_, rd in summary["readouts"].items():
                L.append("- kappa %s: accumulated truncation entropy bound %.3e nats (Gram drops %.3e); stage direct route %s" % (k_, rd["ent_bound"], rd.get("ent_bound_gram", 0.0), json.dumps(rd["stage_direct"], default=_jsonable)))
                for lab_, plist in (("at the end of stage 1", rd.get("partners_stage1") or []), ("at the end", rd["partners"])):
                    for p in plist[:4]:
                        L.append("  - partner (%s; %s): n %.4e, ||w||^2 %.4e against n(1-n) %.4e; weights core %.3e annulus %.3e outside r_t %.3e beyond cut %.3e; negative-half %.3e" % (
                            lab_, p.get("label"), p["n"], p["norm2_w"], p["n_1_minus_n"], p["w_core"], p["w_annulus"], p["w_outside_rt"], p["w_beyond_cut"], p["w_negative_half"]))
                for cs in rd["cross_surface"]:
                    L.append("  - cross-surface d %d (the exact stage state, 31.4 (2)): I physical %.4e (bound %.3e), reference vacuum %.4e (bound %.3e)" % (
                        cs["d"], cs["physical"]["I"], cs["physical"]["bound_2K2"], cs["reference_vacuum"]["I"], cs["reference_vacuum"]["bound_2K2"]))
                spec = rd.get("spectrum")
                if spec:
                    L.append("  - energy-resolved collected occupations of the last stage (omega/T_H bins; mean excess occupation; count; per-bin marginals sum to an upper bound on the joint, R9):")
                    edges = [0, 1, 2, 3, 4, 6, 8, 12, 25, 1e9]
                    for i in range(len(edges) - 1):
                        sel = [s for s in spec if edges[i] <= abs(s[0]) < edges[i + 1]]
                        if sel:
                            L.append("    - [%g, %g): particles %.3e (n %d), holes %.3e (n %d)" % (edges[i], edges[i + 1], np.mean([s[1] for s in sel if s[2] > 0] or [0.0]), len([s for s in sel if s[2] > 0]),
                                                                                           np.mean([s[1] for s in sel if s[2] < 0] or [0.0]), len([s for s in sel if s[2] < 0])))
        L.append("\nStop: %s. Wall %.0f s. Every number above is computed by page_recursion_v6.py; nothing thermal is an input. Validation status: %s." % (summary["stop_reason"], summary["wall_s"], val["status"]))
        with open(self.prefix + "_tables.md", "w") as f:
            f.write("\n".join(L) + "\n")


def _f(x, fmt):
    """v4: fmt % x, or '-' for None."""
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, torch.device):
        return str(o)
    return str(o)


def main(argv=None):
    a = parse(argv)
    if a.selftest:
        import page_selftest_v6 as ST
        ok = ST.run_all(a)
        sys.exit(0 if ok else 1)
    if a.out is None or a.tag is None:
        raise SystemExit("--out and --tag are required")
    if a.validate_scattering:
        import page_scatter_v5 as SC
        SC.validate(a)
        return
    # v6: the active collector keeps no transported archive across a transition (self-test T4j in v6 mode: after a transition the memory
    # route differs from the dense entropy of the true escaping set by 0.287 nats, against 3.05e-06 with v5's collector), so it is
    # certified for one stage only
    if a.collector == "active" and a.max_stages > 1 and not a.timing_blocks:
        raise SystemExit("v6: --collector active is certified for a single stage (--max-stages 1): across a transition the collected sector is "
                         "not transported (self-test T4j, v6 mode); use --max-stages 1 or --collector v5")
    run = Run(a)
    log("settings: r_s/a %g, g %g, m a %g (m/T_H %.3f; collapse 160), channels %s, nsites %d, ell %g %s, R_b %g r_s (%s), formation %s, R_cut %g r_s, tau_s %g r_s; E a %.4f, T_H a %.4e, A/(4G) %.1f, eta A/a^2 %.1f, budget %.0f modes, L_cont %.3e; %s" % (
        a.rs, a.g, a.m, a.m_over_TH, a.channel_list, a.nsites, a.ell, a.fields, a.Rb, a.interior, a.formation, a.R_cut, a.tau_s, a.E1, a.TH,
        levels(a, a.rs)["A_over_4G"], levels(a, a.rs)["eta_A_over_a2"], levels(a, a.rs)["memory_budget_modes"], levels(a, a.rs)["L_cont"],
        "timing, not the pilot" if a.timing_blocks else LABEL_SEQUENCE))
    if a.timing_blocks > 0:                                                    # v3 (30.1 (6)): timing, not the pilot
        log("timing, not the pilot: the formation, the stage-1 eigendecomposition and %d checkpoints%s" % (a.timing_blocks, ", then one transition and one block" if a.timing_transition else ""))
        run.start()
        walls = []
        for _ in range(a.timing_blocks):
            t_b = time.time()
            row = run.block()
            walls.append(time.time() - t_b)
            log("timing, not the pilot: checkpoint %d wall %.2f s (block steps %.2f s, the rest the direct route and the identity)" % (run.n_block, walls[-1], row["wall_s"]))
        run.timings["checkpoint_wall_s"] = walls
        if a.timing_transition:                                                 # v4: one transition, timed (31.4 (1): "timed first")
            run.end_stage("timing, not the pilot")
            t_tr = time.time()
            run.do_transition(run.E - a.dE_frac * run.E, forced=True)
            run.timings["transition_wall_s"] = time.time() - t_tr
            run.timings["transition_per_channel"] = {k_: t_["timings"] for k_, t_ in run.transitions[-1]["per_channel"].items()}
            log("timing, not the pilot: transition wall %.1f s; per channel %s" % (run.timings["transition_wall_s"], json.dumps(run.timings["transition_per_channel"], default=_jsonable)))
            t_b = time.time()
            row = run.block()
            run.timings["block_after_transition_s"] = time.time() - t_b
            log("timing, not the pilot: the block after the transition %.2f s" % run.timings["block_after_transition_s"])
        run.stop_reason = "timing, not the pilot (%d checkpoints%s)" % (a.timing_blocks, ", one transition" if a.timing_transition else "")
        run.write_outputs(final=True)
        log("timing, not the pilot: done; per-channel wall times %s; checkpoint walls %s s; validation status %s" % (
            json.dumps(run.timings, default=_jsonable), ["%.2f" % x for x in walls], run.validation["status"]))
        return
    # v6 (37.1 (3)): a run may start from a stored block-0 checkpoint (the regression's shared formation)
    if a.start_ckpt and not os.path.exists(run.prefix + "_ckpt.pt"):
        import shutil
        shutil.copyfile(a.start_ckpt, run.prefix + "_ckpt.pt")
        log("v6: the run starts from the stored checkpoint %s (copied to %s_ckpt.pt)" % (a.start_ckpt, run.prefix))
        a.resume = True
    if a.resume and os.path.exists(run.prefix + "_ckpt.pt"):
        run.resume()
        if run.n_block == 0:
            run.recompute_pair_ledger()
    else:
        run.start()
        run.checkpoint()
        if a.keep_start_ckpt:
            import shutil
            shutil.copyfile(run.prefix + "_ckpt.pt", run.prefix + "_ckpt_start.pt")
            log("v6: the block-0 checkpoint kept as %s_ckpt_start.pt" % run.prefix)
    log("v6 (S1): %s; repairs: hull %s, pair ledger %s, collector %s (eps_loc %g), direct route every %d block(s), end rule %s" % (
        label_s1(a), a.hull, a.pair_ledger, a.collector, a.eps_loc, a.direct_every, a.end_rule))
    run.main_loop()


if __name__ == "__main__":
    main()
