#!/usr/bin/env python
"""page_selftest_v6b.py (builder analyst, 2026-10-16): page_selftest_v6.py on page_recursion_v6b.py (inbox #41 D15: T8's P07 slab
check is removed with the slabs; no mutual information computed or printed); results with code page_selftest_v6b.py.

page_selftest_v6.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 37.1 "Ruled" (3), inbox #31 S1's validation): imports
page_ops_v6, page_core_v6, page_recursion_v6 and page_scatter_v5 (unchanged). Every v5 test runs twice on the v6 code: in v5 mode
(--hull v5 --pair-ledger v5 --collector v5 --direct-every 5 --end-rule v5; with --v5-selftest-results its printed numbers are compared with
the stored v5 run, T-V5REPRO, every listed number equal to every digit) and in v6 mode (the repairs on: the exact hull, the sharp
dropped-pair ledger, the active-subspace collector at eps_loc 1e-2; T4e takes the exact pre-compression Z; T4d and T4f take the collected
modes from the stage record, the active collector keeping no FIFO; T4j's dense comparisons take the stage records' modes). New counted
tests: T7a the hull counterexample of P01 sec. 2 re-derived (P = diag(0, 1), C = [[1/2, b], [b, 1 - 2 b^2]], b = 2e-7, eps_rank 1e-9,
eps_orth 1e-12; the probe q = (0.2, 0.96^(1/2)): v5's hull drops the kept eigenvector's minor half and errs by about 3e-7 nats against
its logged s(1e-9), the exact hull errs at roundoff); T7b the sharp dropped-pair ledger against the exact F(s) of SHARP_ENTROPY_THEOREM's
statement at seven sines (the difference evaluated as the integral of its derivative, no cancellation), F(s) at most the envelope and
v5's h(s^2) below F(s) at small s; T8 the dense oracle on a
fixture with a trapping zero (the builder's fixture: r_s/a 12, G/a^2 1/4, massless, 256 sites, kappa 1, ell 2 (ell/2GE = 1/6 below 0.26),
R_b 0.5 r_s, dust from 3 r_s, R_cut 2.5 r_s, R_collar 1.5 r_s, tau_s 1 r_s, 14 blocks, eps_loc 1e-2; every check against the dense
evolution of the recursion's own initial memory, the exact formation state's differences printed beside: the formation's dropped pairs,
linear in the sines for energies), at eps_rank 0 (exact: the memory
route against the dense entropy of the cumulative collected set below 1e-8, the exact Z identity below 1e-10, the all-history Gram below
1e-10, the energy identity below 1e-8, the collar identity's E_out, E_K and M_chi against the dense state below 1e-10 relative, the P07
slabs against the dense covariance below 1e-10, every collected mode's dense outside weight at least 1 - eps_loc) and at eps_rank 1e-9
(the memory route within its ledger band of the dense entropy). The results file carries "code": "page_selftest_v6.py".

page_selftest_v5.py. Changes against page_selftest_v4.py (builder analyst; SPEC_final4b_engine.md 33.1 "Ruled" (1), (2), (6), (7) and
33.6 BUILDER item 1), each with the 33.1 item it implements; imports page_ops_v5, page_core_v5, page_recursion_v5, page_scatter_v5.
(33.1 (1)) T4f's criterion as ruled: the uncompleted identity |tr[Delta_k h_k] - (E_mem + E_coll + E_x)|/E_init (E_coll and E_x by the
    direct dense route) below 1e-6 at every block at the production thresholds (eps_coll 1e-5, eps_rank 1e-9, eps_form 1e-7, eps_orth
    1e-12), and, as the new counted test T4f-roundoff, below 1e-8 at every block with all four thresholds at roundoff (eps_coll 1e-10,
    eps_rank 1e-13, eps_form 1e-9, eps_orth 1e-15); the completed form is withdrawn; E_trunc,acc, the dense coverage identity and the
    memory's mismatch inside its span are printed with no criterion, beside the stage record's own identity.
(33.1 (2)) T4i prints the continuation's v5 record line (flagged only by a Weyl failure, dh/g above 1/8 or the nsub cap); its criterion is
    v4's.
(33.1 (6)) new counted test T4k, the lower-memory transition's restore: on the T4 small case after 3 blocks the channel is snapshotted the
    driver's way (no old eigenvectors held: checked), one transition made (the old eigenvectors released: checked), the channel restored
    (the old stage rebuilt by its eigendecomposition) and compared with a full copy taken before; then the transition repeated at tau_tr
    x 2 and compared with a second channel carried through the same blocks and transitioned at tau_tr x 2 directly. It passes when the
    restored state equals the copy (bitwise on the exact path; within 1e-12 on the re-expressed path) and the repeated transition's
    P^cont + Delta agrees with the direct one within 1e-10 of ||Delta||_F.
(33.1 (4)) T4j prints, after the transition, the first block's cap (the memory's rank) and the uncollected content (no criterion).
Every other test, parameter and tolerance is v4's.

page_selftest_v4.py. Changes against page_selftest_v3.py (builder analyst; SPEC_final4b_engine.md 31.4 (3), (4), (6) and 31.6 BUILDER
item 1 (i); the GAP03 review review_GAP03_page_curve.tex, R11), the lead's rulings made as written; imports page_ops_v4, page_core_v4,
page_recursion_v4, page_scatter_v4.
(R11, 31.4 (6)) T1a reports a skipped comparison as SKIP (printed SKIP, recorded with pass None and skipped True); a SKIP among T1 to T5
    makes the self-test fail (a skipped comparison is not an executed passing one); --engine-path reaches T1a and the transitions.
(31.4 (3), 31.6 (i)) T4f tests the completed identity tr[Delta_k h_k] = E_mem + E_coll + E_x + E_trunc,acc by the direct dense route
    (E_coll and E_x from the dense state and the collected modes, E_trunc,acc the energy the code's compressions dropped, accumulated in
    the stage record) below 1e-8 relative at the production thresholds at every block; the uncompleted residual is printed beside it as
    the truncation's energy share (the 3.42e-7 of v3; not a criterion), with the stage record's own completed and uncompleted identities.
(31.4 (4)) T4i: on a fresh channel right after initial_memory, the exact transition of the recursion's own state P_1 + Delta_0 (the
    memory route's Delta_0) through the ramp, U (P_1 + Delta_0) U^dagger, against the code's P^cont + Delta_{k+1} (the full-sea transition
    of page_core_v4); it passes when the relative error is at most twice the transition's printed dropped share (the sea part's dropped
    pairs, the compression; plus 1e-10 for rounding) and the S-continuity defect is below v3's tolerance max(1e-3, 10 x leak); printed
    beside: the continuation's certificate line, the leak over every old occupied mode, the sea part's norm. Two labelled diagnostics
    (DIAG, not counted; R11): the CF4 step-halving (the transition and the dense propagator at dt_tr/2 against dt_tr) and the
    no-production control (the formation of zero duration, R_b0 = R_b: Delta_0 = 0, the empty memory through 3 blocks and one transition).
(31.4 (6), R8, R11) T4j: the driver (page_recursion_v4.Run) on the T4 small case: 3 blocks, one transition (2 percent charge step), 4
    blocks, uninterrupted; and the same run checkpointed after the first block after the transition and resumed by a new Run; it passes
    when the resumed S_esc agrees with the uninterrupted one at every later block below 1e-6. Printed beside (no criterion): the memory
    route's S_esc after the transition (the channel's own, unweighted) against the dense entropy of every collected mode carried exactly
    through the ramp (the true escaping set) and against the dense entropy of the code's FIFO (each propagated column on its labelled
    half), and the stage identities of both stages.
(R10, 31.4 (8)) T6 stays a labelled diagnostic at r_s/a 8 and prints the stationary Gamma_lat where the stage has a trapped annulus (at
    r_s/a 8, ell 2 it has none: the reason is printed) and the finite-window branch-current diagnostic.
Every other test, parameter and tolerance is v3's.

page_selftest_v3.py. Changes against page_selftest_v2.py (builder analyst, BRIEF_page_v3_builder.md), the lead's rulings of
SPEC_final4b_engine.md 30.1 made as written: imports page_ops_v3, page_core_v3, page_recursion_v3, page_scatter_v3. (2) T4b keeps eps_form
1e-7; its criterion is the printed truncation: rel_err_Delta within a factor 2 of the dropped-pair share sqrt(2 sum s^2)/||Delta_0||_F that
initial_memory prints (and ||C_0^2 - C_0|| below 1e-10 as before); the dropped pairs' entropy bound sum s(s^2) is printed. (3) T4f tests the
full identity tr[Delta_k h_k] = E_mem + E_coll + E_x at every block by the direct dense route (E_coll and E_x from the dense state and the
collected modes; tr[Delta_k h_k] the stage's initial excess, E_mem the memory's) below 1e-8, and prints the stage record's own identity
(page_core_v3.energy_identity), E_rad (the memory route) and E_diag. (4) T4i is split: the code route, the recursion against
U Delta_0 U^dagger + (P^cont - P_2) with Delta_0 the recursion's own initial Delta, passes below 10 times the leak (the S-continuity
criterion as before); printed beside, with no tolerance: the same with the dense Delta_0 (it adds T4b's truncation), the dropped
non-adiabatic coherence ||U P_1 U^dagger - P^cont||_F/||Delta_0||_F by the dense route and by the transition's test modes, and the
entropy bound sum_i s(leak_i). (5) T6 at r_s/a 8 stays a labelled diagnostic and prints the branch-resolved measure of page_scatter_v3.
Every other test, parameter and tolerance is v2's.

page_selftest_v2.py. Changes against page_selftest_v1.py (builder analyst, BRIEF_page_debug_v1.md): imports page_ops_v2,
page_core_v2, page_recursion_v2 and page_scatter_v2; T1a finds engine_ops_v1.py through --engine-path, then $PAGE_ENGINE_PATH, then
../engine, ../engine/code and ../../engine/code beside this file, then <internal>/code (the pod's layout; v1 looked only in
../engine, where the pod has no engine_ops_v1.py) and prints the directory used or the directories tried.
The lead's amendments of SPEC_final4b_engine.md 29.9 (turn 37; the method unchanged), made here as the lead ruled: (a) the T4 small case
runs at the production thresholds of GAP03 sec. 1, eps_coll 1e-5, eps_rank 1e-9, eps_form 1e-7, eps_orth 1e-12 (v1: 1e-10, 1e-13, 1e-9,
1e-15; every other T4 setting is v1's); (b) T4d adds the check n_collected below 2n at every checkpoint; (c) the Gram-defect criterion in
T4d and T4g is 1e-6 (v1: 1e-8), the entropy criteria unchanged; (d) T4i runs on a fresh channel right after initial_memory against the
full dense Delta (v1: after T4's 12 collections); (e) T6 at r_s/a 8 is a labelled diagnostic (printed as DIAG, not counted in SELFTEST;
the gate for the lattice transmission is --validate-scattering at r_s/a 32 with flatness above 0.9 on the counted points); (f) T1a imports
the engine operator by path (importlib from the file found). Every other test, parameter and tolerance is v1's.

page_selftest_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the --selftest of page_recursion_v1.py, minutes on a
CPU. T1 the operator against the engine's chain_parts when importable, and the symbols sin k sigma_x and sin(2k)/2 on a bulk plane wave;
T2 the l = 1 kernel average reproduces a linear flow; T3 the recursion identities on a random pure Gaussian state (purity, S(B) = tr s(Z),
the mutual-information identity, the increment identity); T4 the physical small case: the formation, Delta_0 by principal angles against the
dense difference of projectors, 12 block steps with S_esc = tr s(M) against the direct dense entropy of the collected modes, the Z identity,
the energy balance, the stage direct route, the partner identity, and one transition against the exact dense evolution through the ramp;
T5 the continuum luminosity and b against 0.0052373 and 1.6391; T6 the lattice first-passage transmission at r_s/a 8 against the continuum
average (coarse; stated tolerance). Every check prints PASS or FAIL with its number; run_all returns True when all pass."""
import importlib.util
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
RESULTS = []


MODE = "v6"                                                                   # v6: the mode the T4 tests run in
MODE_ARGS = {"v5": ["--hull", "v5", "--pair-ledger", "v5", "--collector", "v5", "--direct-every", "5", "--end-rule", "v5"],
             "v6": ["--hull", "exact", "--pair-ledger", "sharp", "--collector", "active", "--eps-loc", "1e-2", "--direct-every", "1", "--end-rule", "v6"]}


def mode_argv():
    return list(MODE_ARGS[MODE])


def tagged(name):
    return "%s [%s mode]" % (name, MODE) if name.startswith("T4") else name


def last_collected(ch):
    """v6: the last collected block (Q, halves, t): the FIFO's last entry (v5's collector) or the channel's last block (the active one)."""
    return ch.fifo[-1] if ch.fifo else ch.last_block


def rep(name, ok, detail):
    name = tagged(name)
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print("%s  %s: %s" % ("PASS" if ok else "FAIL", name, json.dumps(detail, default=float)), flush=True)
    return bool(ok)


def skip(name, detail):
    """v4 (R11): a comparison that did not execute: printed SKIP, recorded with pass None and skipped True; run_all counts it as not passed."""
    RESULTS.append({"test": name, "pass": None, "skipped": True, "detail": detail})
    print("SKIP  %s: %s" % (name, json.dumps(detail, default=float)), flush=True)
    return False


def diag(name, detail):
    """a labelled diagnostic (the lead's amendment 29.9 for T6): printed as DIAG, recorded with pass None, not counted."""
    RESULTS.append({"test": name, "pass": None, "diagnostic": True, "detail": detail})
    print("DIAG  %s: %s" % (name, json.dumps(detail, default=float)), flush=True)


def s_np(x):
    return OPS.binary_entropy_np(x)


# --------------------------------------------------------------------------------------------------------------------- T1
def engine_dirs(engine_path=None):
    """v2: the candidate directories for engine_ops_v1.py, in order (v4: page_core_v4.engine_dirs)."""
    return CORE.engine_dirs(engine_path)


def t1_operator(engine_path=None):
    ok = True
    dirs = engine_dirs(engine_path)
    found = [d for d in dirs if os.path.isfile(os.path.join(d, "engine_ops_v1.py"))]
    try:
        if not found:
            raise ImportError("engine_ops_v1.py not found in %s" % dirs)
        spec = importlib.util.spec_from_file_location("engine_ops_v1", os.path.join(found[0], "engine_ops_v1.py"))   # by path (29.9)
        EOPS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(EOPS)
        h_ref, Q_ref, R_ref = EOPS.chain_parts(64, 0.2, 2.0)
        h0, Q = OPS.chain_sparse(R_ref, 0.2, 2.0)
        dh = float(np.max(np.abs(h0.toarray() - h_ref))); dQ = float(np.max(np.abs(Q.toarray() - Q_ref)))
        ok &= rep("T1a operator against engine_ops_v1.chain_parts (m 0.2, kappa 2, 64 sites)", dh < 1e-14 and dQ < 1e-14,
                  {"max_diff_h": dh, "max_diff_Q": dQ, "engine_dir": found[0]})
    except Exception as ex:                                          # v4 (R11): a skipped comparison is SKIP, not PASS
        ok &= skip("T1a operator against the engine (not executed)", {"reason": str(ex)[:400]})
    n = 400
    R = np.arange(n, dtype=float) + 1.0
    h0, Q = OPS.chain_sparse(R, 0.0, 0.0)                              # no on-site term: the pure sine chain
    k = 0.7
    u = np.array([1.0, 1.0]) / math.sqrt(2.0)                           # sigma_x eigenvector (+1)
    psi = (np.exp(1j * k * np.arange(n))[:, None] * u[None, :]).reshape(-1)
    hpsi = (h0 @ psi).reshape(n, 2); Qpsi = (Q @ psi).reshape(n, 2)
    mid = slice(100, 300)
    err_h = float(np.max(np.abs(hpsi[mid] - math.sin(k) * psi.reshape(n, 2)[mid])))
    err_Q = float(np.max(np.abs(Qpsi[mid] - 0.5 * math.sin(2 * k) * psi.reshape(n, 2)[mid])))
    ok &= rep("T1b symbols sin k sigma_x and sin(2k)/2 on a bulk plane wave", err_h < 1e-12 and err_Q < 1e-12, {"err_h": err_h, "err_Q": err_Q})
    # the clocked operator with uniform v: energies +-sin k + v sin(2k)/2
    v = -0.3
    h = OPS.clocked_csr(h0, Q, np.ones(n), np.full(n, v))
    hp = (h @ psi).reshape(n, 2)
    err_c = float(np.max(np.abs(hp[mid] - (math.sin(k) + v * 0.5 * math.sin(2 * k)) * psi.reshape(n, 2)[mid])))
    ok &= rep("T1c clocked symbol sin k + v sin(2k)/2 at uniform v", err_c < 1e-12, {"err": err_c})
    return ok


# --------------------------------------------------------------------------------------------------------------------- T2, T3
def t2_kernel():
    R = np.arange(200, dtype=float) + 1.0
    av = OPS.FlowAverager(R, 2.0, "averaged")
    out = av(lambda r: np.asarray(r, float))
    sel = (R > 20) & (R < 170)
    err = float(np.max(np.abs(out[sel] - R[sel])))
    nrm = float(np.max(np.abs(av.norm0[sel] - 1.0)))
    return rep("T2 l = 1 kernel average of v = r returns R_n (bulk), l = 0 norms 1", err < 1e-8 and nrm < 1e-8, {"max_err_v": err, "max_err_norm0": nrm})


# --------------------------------------------------------------------------------------------------------------------- T3
def t3_recursion_identities(d=24, m1=5, m2=4, seed=3):
    rng = np.random.default_rng(seed)
    def unitary(k):
        q, _ = np.linalg.qr(rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k)))
        return q
    Uc = unitary(d); Phi = Uc[:, : d // 2]; C = Phi @ Phi.conj().T
    W = unitary(d); Q1 = W[:, :m1]; Q2 = W[:, m1: m1 + m2]; A1 = W[:, m1:]; A2 = W[:, m1 + m2:]
    S = lambda M: s_np(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))
    M1 = A1.conj().T @ C @ A1; M2 = A2.conj().T @ C @ A2
    Dk = A2.conj().T @ A1
    purity = abs(S(Q1.conj().T @ C @ Q1) - S(M1))
    rec = float(np.linalg.norm(M2 - Dk @ M1 @ Dk.conj().T))
    ev, U = np.linalg.eigh(M1); P = (U * np.sqrt(np.clip(ev * (1 - ev), 0, None))) @ U.conj().T
    Z = np.block([[np.eye(d - m1) - M1, P @ Dk.conj().T], [Dk @ P, M2]])
    SB_direct = S(Q2.conj().T @ C @ Q2); SB_Z = S(Z)
    I_direct = S(Q1.conj().T @ C @ Q1) + SB_direct - S(np.concatenate([Q1, Q2], axis=1).conj().T @ C @ np.concatenate([Q1, Q2], axis=1))
    I_rec = S(M1) + SB_Z - S(M2)
    incr = (S(M2) - S(M1)) - (SB_direct - I_direct)
    ok = purity < 1e-10 and rec < 1e-12 and abs(SB_direct - SB_Z) < 1e-9 and abs(I_direct - I_rec) < 1e-9 and abs(incr) < 1e-9
    return rep("T3 recursion identities on a random pure Gaussian state", ok, {"purity": purity, "M2_minus_DMD": rec, "S_B_direct_minus_Z": SB_direct - SB_Z,
                                                                            "I_direct_minus_rec": I_direct - I_rec, "increment_identity": incr})


# --------------------------------------------------------------------------------------------------------------------- T4
T4_ARGV = ["--rs", "4", "--g", "0.25", "--nsites", "192", "--channels", "1", "--ell", "2", "--Rb", "0.5", "--Rb0", "3", "--R-cut", "12", "--R-collar", "6",
           "--tau-s", "2", "--m-cap", "1000", "--eps-coll", "1e-5", "--eps-rank", "1e-9", "--eps-form", "1e-7", "--eps-orth", "1e-12", "--fifo-depth", "100",
           "--fifo-prune-weight", "0", "--store-cap", "100000", "--tau-tr", "8", "--dt-tr", "0.25"]


def dense_state(V, e, C0, t):
    """C(t) = V exp(-i e t) V^dagger C0 V exp(+i e t) V^dagger (one stage, from t = 0)."""
    ph = torch.exp(-1j * e.to(CD) * t)
    return V @ ((ph[:, None] * (V.mH @ C0 @ V)) * ph.conj()[None, :]) @ V.mH


def ramp_unitary(ch, v_old, v_new, tau, dt, dev):
    """the CF4 propagator of the ramp on the identity (the same scheme as the code; the step-halving DIAG bounds its time-step error)."""
    vfun_t = lambda tt: v_old + (v_new - v_old) * OPS.ramp_c1(tt / tau)
    return OPS.cf4_propagate(ch.B, torch.eye(ch.n2, dtype=CD, device=dev), vfun_t, 0.0, tau, dt)


def t4_small_case(device, engine_path=None):
    import page_recursion_v6b as PR
    # the lead's amendment (29.9 (a)): the production thresholds of GAP03 sec. 1 (v1: eps_coll 1e-10, eps_rank 1e-13, eps_form 1e-9, eps_orth 1e-15)
    argv = T4_ARGV + mode_argv() + ["--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t4"] + (["--engine-path", engine_path] if engine_path else [])
    a = PR.parse(argv)
    dev = torch.device(str(device))
    ch = CORE.Channel(1, a.m, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    Phi0, frec = ch.formation(a)
    ok = rep("T4a formation unitarity (r_s 4, n 192, dust from 3 r_s)", frec["unitarity"] < 1e-10, {"unitarity": frec["unitarity"], "t_form_over_rs": frec["t_form_over_rs"], "n0": frec["n0"]})
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    irec = ch.initial_memory(Phi0, a)
    V, e, neg = st["V"], st["e"], st["neg"]
    C0 = Phi0 @ Phi0.mH
    pur = float(torch.linalg.norm(C0 @ C0 - C0))
    P1 = V[:, neg] @ V[:, neg].mH
    Delta_exact = C0 - P1
    Drec = (V @ ch.c) @ ch.MD @ (V @ ch.c).mH
    d_delta = float(torch.linalg.norm(Delta_exact - Drec)) / max(float(torch.linalg.norm(Delta_exact)), 1e-300)
    share = irec["dropped_pair_share"]                                     # 30.1 (2): the printed truncation
    in_window = (0.5 * share <= d_delta <= 2.0 * share) if share > 1e-12 else (d_delta < 1e-10)
    ok &= rep("T4b purity of C_0 and Delta_0 reconstructed from the split hull of its principal angles (30.1 (2): within a factor 2 of the printed truncation)",
              pur < 1e-10 and in_window,
              {"||C0^2 - C0||": pur, "rel_err_Delta": d_delta, "dropped_pair_share": share, "ratio_to_share": d_delta / max(share, 1e-300),
               "norm_Delta0_dense": float(torch.linalg.norm(Delta_exact)), "norm_Delta0_from_sines": irec["norm_Delta0_from_sines"],
               "n_dropped_pairs": irec["n_dropped_pairs"], "entropy_bound_dropped_pairs": irec["entropy_bound_dropped_pairs"],
               "D8_lemma_dropped_pairs_v2": irec["D8_lemma_dropped_pairs_v2"], "memory_rank": irec["memory_rank"], "Delta_rank": irec["rank_kept"], "E_inv": irec["E_inv"]})
    E_inv = float(torch.trace(Delta_exact @ OPS.dense_from_csr(st["h"], dev)).real)
    ok &= rep("T4c E_inv = tr[Delta_0 h_1] from the memory against the dense trace", abs(irec["E_inv"] - E_inv) < 1e-8 * max(abs(E_inv), 1.0), {"E_inv_memory": irec["E_inv"], "E_inv_dense": E_inv})
    # the block steps against the dense state
    worst_S = worst_Z = worst_bal = worst_gram = 0.0
    h_dense = OPS.dense_from_csr(st["h"], dev)
    worst_unc = 0.0; worst_unc_code = 0.0; id_rows = []; worst_cov = 0.0; worst_mis = 0.0
    collected = []
    n_coll_max = 0; n_coll_below = True
    t = 0.0
    for n in range(12):
        t += a.tau_s_abs
        r = ch.block_step(a, t)
        if r["m_coll"]:
            cq, hq, tq = last_collected(ch)
            collected.append((cq.clone(), tq))
        n_coll = sum(c.shape[1] for c, _ in collected)                      # the lead's amendment (29.9 (b)): n_collected below 2n
        n_coll_max = max(n_coll_max, n_coll); n_coll_below = n_coll_below and n_coll < ch.n2
        Ct = dense_state(V, e, C0, t)
        if collected:
            Q = torch.cat([V @ (torch.exp(-1j * e.to(CD) * (t - tq))[:, None] * cq) for cq, tq in collected], dim=1)
            gram = float(torch.linalg.norm(Q.mH @ Q - torch.eye(Q.shape[1], dtype=CD, device=dev)))
            S_direct, _ = OPS.entropy_cov(Q.mH @ Ct @ Q)
            worst_gram = max(worst_gram, gram)
            worst_S = max(worst_S, abs(S_direct - r["S_new"]))
        if r["m_coll"]:
            worst_Z = max(worst_Z, abs(r.get("S_B_Z_exact", r["S_B_Z"]) - r["S_B"]))      # v6: the exact pre-compression Z (active)
        # v5 (33.1 (1)): the uncompleted identity tr[Delta_k h_k] = E_mem + E_coll + E_x by the direct dense route (the criterion)
        Dt_dense = Ct - P1
        Dm = V @ ch.c
        E_init = ch.rec["E_delta_init"]; E_tr = ch.rec.get("E_trunc", 0.0)
        E_coll = float(torch.trace((Q.mH @ Dt_dense @ Q) @ (Q.mH @ h_dense @ Q)).real) if collected else 0.0
        E_x = 2.0 * float(torch.trace((Q.mH @ Dt_dense @ Dm) @ (Dm.mH @ h_dense @ Q)).real) if (collected and Dm.shape[1]) else 0.0
        res_unc = (E_init - (r["E_mem"] + E_coll + E_x)) / max(abs(E_init), 1e-300)
        worst_unc = max(worst_unc, abs(res_unc))
        # printed beside (no criterion): the dense coverage identity with the exact energy in the memory's span in place of E_mem
        E_md = float(torch.trace((Dm.mH @ Dt_dense @ Dm) @ (Dm.mH @ h_dense @ Dm)).real) if Dm.shape[1] else 0.0
        worst_cov = max(worst_cov, abs((E_init - (E_md + E_coll + E_x)) / max(abs(E_init), 1e-300)))
        worst_mis = max(worst_mis, abs((r["E_mem"] - E_md) / max(abs(E_init), 1e-300)))
        ei = ch.energy_identity(a)
        if ei is not None:
            worst_unc_code = max(worst_unc_code, abs(ei["residual_rel"]))           # v5: the stage record's uncompleted identity
        id_rows.append({"block": n + 1, "E_mem": r["E_mem"], "E_coll": E_coll, "E_x": E_x, "E_trunc_acc": E_tr, "uncompleted_residual_rel": res_unc,
                        "stage_record_residual_rel": ei["residual_rel"] if ei else None, "E_rad_memory_route": E_init - r["E_mem"], "E_diag": ch.rec["E_rad"]})
        worst_bal = max(worst_bal, abs(r["E_mem"] + ch.rec["E_rad"] - ch.rec["E_delta_init"]) / max(abs(ch.rec["E_delta_init"]), 1e-300))
    ok &= rep("T4d S_esc = tr s(M) against the direct dense entropy of the collected modes over 12 blocks (Gram defect below 1e-6, n_collected below 2n: 29.9)",
              worst_S < 1e-6 and worst_gram < 1e-6 and n_coll_below,
              {"max_|S_direct - S_memory|": worst_S, "max_gram_defect": worst_gram, "S_esc_final": r["S_new"], "n_collected": sum(c.shape[1] for c, _ in collected),
               "n_collected_max": n_coll_max, "2n": ch.n2})
    ok &= rep("T4e S(B_k) from Z_k against the direct block entropy", worst_Z < 1e-9, {"max_|S_B_Z - S_B|": worst_Z})
    ok &= rep("T4f the energy identity |tr[Delta_k h_k] - (E_mem + E_coll + E_x)|/E_init within the stage by the direct dense route at the production thresholds (33.1 (1): below 1e-6 at every block; the completed form withdrawn)",
              worst_unc < 1e-6,
              {"max_rel_residual_uncompleted": worst_unc, "criterion": 1e-6,
               "thresholds": {"eps_coll": a.eps_coll, "eps_rank": a.eps_rank, "eps_form": a.eps_form, "eps_orth": a.eps_orth},
               "E_trunc_acc_final_printed_no_criterion": ch.rec.get("E_trunc", 0.0), "E_trunc_eigs_acc_final_printed": ch.rec.get("E_trunc_eigs", 0.0),
               "dense_coverage_identity_max_rel_printed_no_criterion": worst_cov, "memory_mismatch_E_mem_minus_dense_span_max_rel": worst_mis,
               "stage_record_route_max_rel": worst_unc_code,
               "E_init": ch.rec["E_delta_init"], "gram_energy_bound_final": ch.rec.get("gram_energy_bound", 0.0),
               "E_rad_memory_route_final": id_rows[-1]["E_rad_memory_route"], "E_diag_final": id_rows[-1]["E_diag"],
               "blocks_coherence_energy_final": id_rows[-1]["E_diag"] - id_rows[-1]["E_rad_memory_route"], "withdrawn_D10_diagonal_balance_max_rel": worst_bal,
               "per_block_uncompleted": [x["uncompleted_residual_rel"] for x in id_rows],
               "per_block_E_trunc_acc": [x["E_trunc_acc"] for x in id_rows], "per_block_stage_record": [x["stage_record_residual_rel"] for x in id_rows]})
    d = ch.stage_direct(a, with_spectrum=True)
    ok &= rep("T4g stage direct route G_R = Pi + A^dagger M_init A against the memory route (Gram defect below 1e-6: 29.9)", d is not None and abs(d["S_direct"] - r["S_new"]) < 1e-6 and d.get("gram_defect", 0.0) < 1e-6,
              {"S_direct": d["S_direct"] if d else None, "S_memory": r["S_new"], "gram_defect": d.get("gram_defect") if d else None})
    prt = ch.partner_readout(a, n_modes=4)
    werr = max([abs(p["norm2_w"] - p["n_1_minus_n"]) for p in prt] or [0.0])
    ok &= rep("T4h partner identity ||C q - n q||^2 = n(1 - n)", werr < 1e-8, {"max_identity_error": werr, "n_values": [p["n"] for p in prt]})
    ok &= t4i_transition(a, dev)
    return ok


def fresh_channel(a, dev):
    ch = CORE.Channel(1, a.m, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    Phi0, _ = ch.formation(a)
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    ch.initial_memory(Phi0, a)
    return ch, st, Phi0


def t4i_transition(a, dev):
    """31.4 (4): the code route on a fresh channel right after initial_memory, the exact transition of the recursion's own state."""
    E_new = 0.98 * a.E1
    Rb_new = a.Rb * 2.0 * a.g * E_new
    ch, st, Phi0 = fresh_channel(a, dev)
    V, e, neg = st["V"], st["e"], st["neg"]
    S_before, _ = ch.memory_entropy()
    v_old = st["v_sites"]; rs_old = st["rs"]
    tau = a.tau_tr * rs_old
    v_new = np.asarray(ch.averager(OPS.flow_dense(2.0 * a.g * E_new, Rb_new)), float)
    Dmem0 = (V @ ch.c) @ ch.MD @ (V @ ch.c).mH                                 # the recursion's own Delta_0 (the memory route)
    P1 = V[:, neg] @ V[:, neg].mH
    ph = torch.exp(-1j * e.to(CD) * ch.t_ref)
    C_rec = V @ ((ph[:, None] * (V.mH @ (P1 + Dmem0) @ V)) * ph.conj()[None, :]) @ V.mH
    U1 = ramp_unitary(ch, v_old, v_new, tau, a.dt_tr, dev)
    C_after = U1 @ C_rec @ U1.mH
    tr = ch.transition(a, E_new, Rb_new)
    st2 = ch.stage; V2, occ = st2["V"], st2["neg"]
    Pcont = V2[:, occ] @ V2[:, occ].mH
    Drec2 = (V2 @ ch.c) @ ch.MD @ (V2 @ ch.c).mH
    nA = max(float(torch.linalg.norm(C_after - Pcont)), 1e-300)
    d_code = float(torch.linalg.norm(C_after - Pcont - Drec2)) / nA
    S_after, _ = ch.memory_entropy()
    tol = max(1e-3, 10.0 * tr["adiabatic_leak_max"])
    share = tr["transition_dropped_share"]
    ok = rep("T4i transition on a fresh channel right after initial_memory (31.4 (4)): U (P_1 + Delta_0) U^dagger against the code's P^cont + Delta_2, at most twice the transition's printed dropped share",
             d_code <= 2.0 * share + 1e-10 and abs(S_after - S_before) < tol,
             {"rel_err": d_code, "transition_dropped_share": share, "twice_share_plus_1e-10": 2.0 * share + 1e-10, "S_continuity_defect": S_after - S_before, "S_tolerance": tol,
              "continuation": tr["continuation_label"], "nsub": tr["continuation"]["nsub"], "nsub_needed": tr["continuation"]["nsub_needed"],
              "weyl_margin": tr["continuation"]["margin_min"], "max_dh_over_gap": tr["continuation"]["max_dh_over_gap"], "max_hdot_over_gap2": tr["continuation"]["max_hdot_over_gap2"],
              "reference_offset_F2": tr["reference_offset_F2"], "adiabatic_leak_max_all_occupied": tr["adiabatic_leak_max"], "sea_coherence_F": tr["sea_coherence_F"],
              "sea_coherence_over_Delta0": tr["sea_coherence_over_Delta0"], "sea_dropped_pair_share": tr["sea_dropped_pair_share"],
              "compression_dropped_share": tr["compression_dropped_share"], "compression_E_trunc": tr["compression_E_trunc"], "sea_unitarity": tr["sea_unitarity"],
              "memory_rank_after": tr["memory_rank"], "chebyshev_tail_bound": tr["chebyshev"]["tail_bound_acc"]})
    # DIAG (R11): the CF4 step-halving, the code's transition and the dense propagator at dt_tr/2 against dt_tr
    chh, sth, _ = fresh_channel(a, dev)
    trh = chh.transition(a, E_new, Rb_new, dt_factor=0.5)
    st3 = chh.stage; V3, occ3 = st3["V"], st3["neg"]
    Ch = V3[:, occ3] @ V3[:, occ3].mH + (V3 @ chh.c) @ chh.MD @ (V3 @ chh.c).mH
    Cc = Pcont + Drec2
    U2 = ramp_unitary(ch, v_old, v_new, tau, 0.5 * a.dt_tr, dev)
    C_after_h = U2 @ C_rec @ U2.mH
    diag("T4i CF4 step-halving (R11): the transition at dt_tr/2 against dt_tr (the code and the dense propagator)",
         {"code_rel_change": float(torch.linalg.norm(Ch - Cc)) / nA, "dense_rel_change": float(torch.linalg.norm(C_after_h - C_after)) / nA,
          "code_half_step_vs_dense_half_step": float(torch.linalg.norm(C_after_h - Ch)) / nA, "masks_equal": bool(torch.equal(occ3, occ)),
          "dt_tr": a.dt_tr, "transition_dropped_share_half_step": trh["transition_dropped_share"]})
    # DIAG (R8.4, R11): the no-production control, R_b0 = R_b (a formation of zero duration): Delta_0 = 0, the empty memory
    import page_recursion_v6b as PR
    argv0 = [x for x in T4_ARGV] + mode_argv()
    argv0[argv0.index("--Rb0") + 1] = "0.5"
    a0 = PR.parse(argv0 + ["--device", str(dev), "--out", "<internal>/page_selftest", "--tag", "t4np"] + (["--engine-path", a.engine_path] if a.engine_path else []))
    ch0, st0, _ = fresh_channel(a0, dev)
    rank0 = int(ch0.c.shape[1]); S0 = []
    tt = 0.0
    for _ in range(3):
        tt += a0.tau_s_abs
        S0.append(ch0.block_step(a0, tt)["S_new"])
    tr0 = ch0.transition(a0, E_new, Rb_new)
    diag("T4i no-production control (R8.4): R_b0 = R_b, Delta_0 = 0, 3 blocks and one transition",
         {"memory_rank_after_formation": rank0, "S_esc_blocks": S0, "after_transition_S": tr0["S_after"], "after_transition_rank": tr0["memory_rank"],
          "sea_coherence_F": tr0["sea_coherence_F"], "adiabatic_leak_max": tr0["adiabatic_leak_max"], "continuation": tr0["continuation_label"]})
    return ok


def t4j_driver_resume(device, engine_path=None, out_dir="<internal>/t4j"):
    """31.4 (6), R8, R11: the driver through a transition, uninterrupted and resumed."""
    import page_recursion_v6b as PR
    os.makedirs(out_dir, exist_ok=True)
    base = T4_ARGV + mode_argv() + ["--direct-every", "1", "--tau-tr-doublings", "0", "--ckpt-every", "1000", "--device", str(device), "--out", out_dir] + (["--engine-path", engine_path] if engine_path else [])
    dev = torch.device(str(device))
    # run A, uninterrupted, with the dense state carried beside it
    A = PR.Run(PR.parse(base + ["--tag", "t4jA"]))
    A.start()
    chA = A.chans[0]; st1 = chA.stage
    V1, e1 = st1["V"], st1["e"]
    for _ in range(3):
        A.block()
    t_start = chA.t_ref
    src1 = list(chA.fifo) if chA.fifo else [(cq, None, tq) for cq, tq in zip(chA.rec["cQ"], chA.rec["tQ"])]   # v6: the active collector's record
    old = [(V1 @ (torch.exp(-1j * e1.to(CD) * (t_start - tq))[:, None] * cq)) for cq, hq, tq in src1]   # every stage-1 collected mode (prune weight 0)
    n_old = sum(int(x.shape[1]) for x in old)
    # the dense state at t_start: the formation state on stage 1, from the formation record's Phi0 (recomputed: deterministic)
    chD = CORE.Channel(1, A.a.m, A.a.nsites, False, dev); chD.set_averager(A.a.ell, A.a.fields)
    Phi0, _ = chD.formation(A.a)
    C0 = Phi0 @ Phi0.mH
    C_ts = dense_state(V1, e1, C0, t_start)
    A.end_stage("T4j")
    v_old = st1["v_sites"]; tau = A.a.tau_tr * st1["rs"]
    A.do_transition(0.98 * A.E, forced=True)
    st2 = chA.stage; V2, e2 = st2["V"], st2["e"]
    U1 = ramp_unitary(chA, v_old, st2["v_sites"], tau, A.a.dt_tr, dev)
    C_after = U1 @ C_ts @ U1.mH
    old_after = [U1 @ x for x in old]
    t2 = chA.t_ref
    n_fifo_hull = sum(int(cq.shape[1]) for cq, _, _ in chA.fifo)
    S_A = []; dense_rows = []; first_after = None
    for _ in range(4):
        row = A.block()
        S_A.append(row["S_esc_total"])
        if first_after is None:                                                 # v5 (33.1 (4)): the first block after the transition
            pc = row["per_channel"]["1"]
            first_after = {k_: pc.get(k_) for k_ in ("first_block_after_transition", "m_cap_used", "r_before", "m_coll", "weight_beyond_cut_uncollected",
                                                     "uncollected_count", "uncollected_content_F", "uncollected_content_E", "uncollected_g_max")}
        tnow = A.t
        ph2 = torch.exp(-1j * e2.to(CD) * (tnow - t2))
        Ct = V2 @ ((ph2[:, None] * (V2.mH @ C_after @ V2)) * ph2.conj()[None, :]) @ V2.mH
        Q_old = torch.cat([V2 @ (ph2[:, None] * (V2.mH @ x)) for x in old_after], dim=1) if old_after else None
        src2 = list(chA.fifo) if chA.fifo else [(cq, None, tq) for cq, tq in zip(chA.rec["cQ"], chA.rec["tQ"])]   # v6: the new stage's record
        new = [(V2 @ (torch.exp(-1j * e2.to(CD) * (tnow - tq))[:, None] * cq)) for cq, hq, tq in src2 if tq > t2 + 1e-9]
        F_hull = torch.cat([V2 @ (torch.exp(-1j * e2.to(CD) * (tnow - tq))[:, None] * cq) for cq, hq, tq in chA.fifo], dim=1) if chA.fifo else None
        parts = ([Q_old] if Q_old is not None else []) + new
        Q_true = torch.cat(parts, dim=1) if parts else None
        S_true = OPS.entropy_cov(Q_true.mH @ Ct @ Q_true)[0] if Q_true is not None else 0.0
        S_hull = OPS.entropy_cov(F_hull.mH @ Ct @ F_hull)[0] if F_hull is not None else 0.0
        gram_true = float(torch.linalg.norm(Q_true.mH @ Q_true - torch.eye(Q_true.shape[1], dtype=CD, device=dev))) if Q_true is not None else 0.0
        dense_rows.append({"t": tnow, "S_memory": row["S_esc_total"] / chA.w, "S_dense_true_escaping_set": S_true, "S_dense_code_fifo": S_hull, "gram_true": gram_true,
                           "identity_33_1_1": row.get("energy_identity_rel_total")})
    # run B: the same, checkpointed after the first block after the transition, resumed by a new Run
    B = PR.Run(PR.parse(base + ["--tag", "t4jB"]))
    B.start()
    for _ in range(3):
        B.block()
    B.end_stage("T4j")
    B.do_transition(0.98 * B.E, forced=True)
    S_B = [B.block()["S_esc_total"]]
    B.checkpoint()
    del B
    B2 = PR.Run(PR.parse(base + ["--tag", "t4jB", "--resume"]))
    B2.resume()
    for _ in range(3):
        S_B.append(B2.block()["S_esc_total"])
    cont = max(abs(x - y) for x, y in zip(S_A, S_B))
    st_ids = [r_.get("energy_identity_rel_total") for r_ in A.curve]
    trA = A.transitions[-1]["per_channel"]["1"]
    ok_out = t_out_writers(B2)                                              # v6b: the output writers on the resumed run
    return ok_out & rep("T4j the driver through a transition, uninterrupted and checkpoint-resumed (31.4 (6): resumed S_esc against uninterrupted below 1e-6)", cont < 1e-6,
               {"max_|S_resumed - S_uninterrupted|": cont, "S_uninterrupted": S_A, "S_resumed": S_B,
                "resumed_record_complete": tuple(B2.chans[0].rec.get("cQ_from", (0, 0))) == (0, 0),
                "dense_after_transition": dense_rows, "max_|S_memory - S_dense_true|": max(abs(r_["S_memory"] - r_["S_dense_true_escaping_set"]) for r_ in dense_rows),
                "max_|S_memory - S_dense_code_fifo|": max(abs(r_["S_memory"] - r_["S_dense_code_fifo"]) for r_ in dense_rows),
                "note": "S_memory is the channel's own S_esc (the weighted total over the weight 16 kappa); the code's FIFO after the transition holds each propagated collected column on its labelled half (the minor half dropped)",
                "stage1_collected_columns": n_old, "fifo_hull_columns_after_transition": n_fifo_hull, "transition_fifo_half_defect_max": trA["fifo_half_defect_max"],
                "transition_memory_removed_on_fifo_share": trA["memory_removed_on_fifo_share"], "transition_continuation": trA["continuation_label"],
                "identity_per_block": st_ids, "first_block_after_transition_33_1_4": first_after,
                "transition_continuation_33_1_2": A.transitions[-1]["per_channel"]["1"]["continuation_label"],
                "validation_status_run_A": A.validation_status()["status"]})



D15_KEYS = ("I_RB", "I_total", "MI_cross_total", "slab_p07", "cross_surface", "cross_surface_own", "cross_surface_end")


def t_out_writers(run):
    """v6b (the regression's failure of 2026-10-16; inbox #41 D15): the driver's output writers on a run of the small case in this mode: the
    summary json, the curve npz and the tables md are written (v6's write_md raised TypeError on a None check in v5 mode, which no v6 test
    reached); no mutual-information entry in the json (no key I and none of D15_KEYS), the npz or the md."""
    err = None; hits = []
    try:
        run.write_outputs(final=False)
    except Exception as ex:                                                   # noqa: BLE001 (recorded as the failure)
        err = "%s: %s" % (type(ex).__name__, ex)
    pre = run.prefix
    files = {s_: os.path.exists(pre + s_) for s_ in ("_summary.json", "_curve.npz", "_tables.md")}
    if err is None and all(files.values()):
        d = json.load(open(pre + "_summary.json"))

        def walk(o, p_):
            if isinstance(o, dict):
                for k_, v_ in o.items():
                    if k_ in D15_KEYS or k_ == "I":
                        hits.append(p_ + "/" + str(k_))
                    walk(v_, p_ + "/" + str(k_))
            elif isinstance(o, list):
                for x in o:
                    walk(x, p_ + "[]")
        walk(d, "")
        z = np.load(pre + "_curve.npz")
        hits += ["npz:" + k_ for k_ in z.files if k_ in ("I_total", "MI_cross_d2_total") or k_.startswith("slab_p07")]
        md = open(pre + "_tables.md").read()
        hits += ["md:" + w_ for w_ in ("MI across", "I total", "mutual information across", "P07 slab", "cross-surface d") if w_ in md]
    ok = err is None and all(files.values()) and not hits
    return rep("T-OUT (v6b) the driver's output writers in %s mode on the resumed T4j run: summary json, curve npz and tables md written; no mutual-information entry (#41 D15)" % MODE,
               ok, {"error": err, "files": files, "d15_hits": sorted(set(hits))[:20], "checks_None": [k_ for k_, v_ in run.checks.items() if v_ is None]})

T4_ROUNDOFF = {"--eps-coll": "1e-10", "--eps-rank": "1e-13", "--eps-form": "1e-9", "--eps-orth": "1e-15"}   # v5 (33.1 (1)): the four at roundoff


def t4f_roundoff(device, engine_path=None):
    """v5 (33.1 (1)): T4f with all four thresholds at roundoff (eps_coll 1e-10, eps_rank 1e-13, eps_form 1e-9, eps_orth 1e-15), every other
    T4 setting the same (12 blocks): the uncompleted identity |tr[Delta_k h_k] - (E_mem + E_coll + E_x)|/E_init by the direct dense route
    below 1e-8 at every block (it certifies the identity's code); E_trunc,acc, the dense coverage identity and the memory's mismatch inside
    its span printed with no criterion, beside the stage record's own identity."""
    import page_recursion_v6b as PR
    argv = list(T4_ARGV) + mode_argv()
    for k_, v_ in T4_ROUNDOFF.items():
        argv[argv.index(k_) + 1] = v_
    argv += ["--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t4f_roundoff"] + (["--engine-path", engine_path] if engine_path else [])
    a = PR.parse(argv)
    dev = torch.device(str(device))
    ch = CORE.Channel(1, a.m, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    Phi0, _ = ch.formation(a)
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    ch.initial_memory(Phi0, a)
    V, e, neg = st["V"], st["e"], st["neg"]
    C0 = Phi0 @ Phi0.mH
    P1 = V[:, neg] @ V[:, neg].mH
    h_dense = OPS.dense_from_csr(st["h"], dev)
    E_init = ch.rec["E_delta_init"]
    collected = []; t = 0.0; per_block = []; worst = 0.0; worst_cov = 0.0; worst_mis = 0.0; worst_code = 0.0
    r = None
    for n in range(12):
        t += a.tau_s_abs
        r = ch.block_step(a, t)
        if r["m_coll"]:
            cq, hq, tq = last_collected(ch)
            collected.append((cq.clone(), tq))
        Dt_dense = dense_state(V, e, C0, t) - P1
        Dm = V @ ch.c
        if collected:
            Q = torch.cat([V @ (torch.exp(-1j * e.to(CD) * (t - tq))[:, None] * cq) for cq, tq in collected], dim=1)
            E_coll = float(torch.trace((Q.mH @ Dt_dense @ Q) @ (Q.mH @ h_dense @ Q)).real)
            E_x = 2.0 * float(torch.trace((Q.mH @ Dt_dense @ Dm) @ (Dm.mH @ h_dense @ Q)).real) if Dm.shape[1] else 0.0
        else:
            E_coll = E_x = 0.0
        E_md = float(torch.trace((Dm.mH @ Dt_dense @ Dm) @ (Dm.mH @ h_dense @ Dm)).real) if Dm.shape[1] else 0.0
        res = (E_init - (r["E_mem"] + E_coll + E_x)) / max(abs(E_init), 1e-300)
        worst = max(worst, abs(res))
        worst_cov = max(worst_cov, abs((E_init - (E_md + E_coll + E_x)) / max(abs(E_init), 1e-300)))
        worst_mis = max(worst_mis, abs((r["E_mem"] - E_md) / max(abs(E_init), 1e-300)))
        ei = ch.energy_identity(a)
        if ei is not None:
            worst_code = max(worst_code, abs(ei["residual_rel"]))
        per_block.append(res)
    return rep("T4f-roundoff the same identity with all four thresholds at roundoff (33.1 (1): below 1e-8 at every block; it certifies the identity's code)",
               worst < 1e-8,
               {"max_rel_residual_uncompleted": worst, "criterion": 1e-8,
                "thresholds": {"eps_coll": a.eps_coll, "eps_rank": a.eps_rank, "eps_form": a.eps_form, "eps_orth": a.eps_orth},
                "per_block_uncompleted": per_block, "E_trunc_acc_final_printed_no_criterion": ch.rec.get("E_trunc", 0.0),
                "dense_coverage_identity_max_rel_printed_no_criterion": worst_cov, "memory_mismatch_E_mem_minus_dense_span_max_rel": worst_mis,
                "stage_record_route_max_rel": worst_code, "E_init": E_init, "S_esc_final": r["S_new"] if r else None,
                "n_collected": sum(int(c.shape[1]) for c, _ in collected)})


def _state_dense(ch):
    """the channel's state P_ref + Delta on the site basis (for T4k's comparisons)."""
    st = ch.stage
    V, occ = st["V"], st["neg"]
    D = V @ ch.c
    return V[:, occ] @ V[:, occ].mH + D @ ch.MD @ D.mH, D @ ch.MD @ D.mH


def t4k_restore(device, engine_path=None):
    """v5 (33.1 (6)): the lower-memory transition's restore (see the module docstring)."""
    import page_recursion_v6b as PR
    argv = T4_ARGV + mode_argv() + ["--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t4k"] + (["--engine-path", engine_path] if engine_path else [])
    a = PR.parse(argv)
    dev = torch.device(str(device))

    def prepared():
        ch_, _, _ = fresh_channel(a, dev)
        tt = 0.0
        for _ in range(3):
            tt += a.tau_s_abs
            ch_.block_step(a, tt)
        return ch_
    E_new = 0.98 * a.E1
    Rb_new = a.Rb * 2.0 * a.g * E_new
    ch = prepared()
    st = ch.stage
    n2 = ch.n2
    ref = {"e": st["e"].clone(), "V": st["V"].clone(), "neg": st["neg"].clone(), "neg_sign": st["neg_sign"].clone(), "c": ch.c.clone(), "MD": ch.MD.clone(),
           "half": ch.half.clone(), "t_ref": ch.t_ref, "fifo": [(cq.clone(), hq.clone(), tq) for cq, hq, tq in ch.fifo], "ent_bound": ch.ent_bound,
           "c_init": ch.rec["c_init"].clone(), "fifo_depth": ch.fifo_depth, "n_transitions": ch.n_transitions}
    snap = ch.transition_snapshot(a.E1, a.Rb * a.rs)
    big = [k_ for k_, v_ in snap.items() if isinstance(v_, torch.Tensor) and v_.dim() == 2 and v_.shape[0] == n2 and v_.shape[1] == n2]
    old_stage = ch.stage
    tr1 = ch.transition(a, E_new, Rb_new, tau_factor=1.0)
    released = old_stage.get("V") is None
    rinfo = ch.restore_snapshot(a, snap)
    st2 = ch.stage
    exact_eq = {"e": bool(torch.equal(st2["e"], ref["e"])), "V": bool(torch.equal(st2["V"], ref["V"])), "neg": bool(torch.equal(st2["neg"], ref["neg"])),
                "neg_sign": bool(torch.equal(st2["neg_sign"], ref["neg_sign"])), "c": bool(torch.equal(ch.c, ref["c"])), "MD": bool(torch.equal(ch.MD, ref["MD"])),
                "half": bool(torch.equal(ch.half, ref["half"])), "t_ref": ch.t_ref == ref["t_ref"], "ent_bound": ch.ent_bound == ref["ent_bound"],
                "fifo": len(ch.fifo) == len(ref["fifo"]) and all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(ch.fifo, ref["fifo"])),
                "c_init": bool(torch.equal(ch.rec["c_init"], ref["c_init"])), "fifo_depth": ch.fifo_depth == ref["fifo_depth"],
                "n_transitions": ch.n_transitions == ref["n_transitions"]}
    dev_max = {"e": float((st2["e"] - ref["e"]).abs().max()), "V": float((st2["V"] - ref["V"]).abs().max()), "c_site": float(((st2["V"] @ ch.c) - (ref["V"] @ ref["c"])).abs().max()),
               "MD": float((ch.MD - ref["MD"]).abs().max()) if ch.MD.numel() else 0.0}
    restored_ok = all(exact_eq.values()) if rinfo["exact"] else (max(dev_max["c_site"], dev_max["MD"]) < 1e-12)
    tr2 = ch.transition(a, E_new, Rb_new, tau_factor=2.0)
    C_rep, D_rep = _state_dense(ch)
    chB = prepared()
    trB = chB.transition(a, E_new, Rb_new, tau_factor=2.0)
    C_dir, D_dir = _state_dense(chB)
    rel = float(torch.linalg.norm(C_rep - C_dir)) / max(float(torch.linalg.norm(D_dir)), 1e-300)
    ok = restored_ok and rel < 1e-10 and released and not big
    return rep("T4k the lower-memory transition's restore (33.1 (6)): the old stage rebuilt by its eigendecomposition equals the state before the transition, and the transition repeated at tau_tr x 2 equals a direct one",
               ok, {"snapshot_holds_n2_x_n2_tensors": big, "old_eigenvectors_released_by_the_transition": released, "restore": rinfo,
                    "bitwise_equal": exact_eq, "max_deviation": dev_max, "repeated_vs_direct_rel": rel, "criterion_repeated": 1e-10,
                    "transition_1_label": tr1["continuation_label"], "transition_2_label": tr2["continuation_label"],
                    "transition_2_S_after": tr2["S_after"], "direct_S_after": trB["S_after"],
                    "memory_rank_repeated": tr2["memory_rank"], "memory_rank_direct": trB["memory_rank"]})


def t5_continuum():
    import page_scatter_v5 as SC
    t0 = time.time()
    c = SC.continuum_totals(kmax=6, n_nodes=40, x_max=1.5)
    ok = abs(c["rel_dev_L"]) < 2e-3 and abs(c["rel_dev_b"]) < 2e-3 and c["unitarity_error_max"] < 1e-6
    return rep("T5 continuum L r_s^2 and b against 0.0052373 and 1.6391 (tolerance 2e-3)", ok,
               {"L_rs2": c["L_rs2"], "rel_dev_L": c["rel_dev_L"], "b": c["b"], "rel_dev_b": c["rel_dev_b"], "mean_omega_over_TH": c["mean_omega_over_TH"],
                "unitarity_max": c["unitarity_error_max"], "seconds": time.time() - t0})


# --------------------------------------------------------------------------------------------------------------------- T6
def t6_lattice_gamma(device, obs_path=None):
    import page_recursion_v6b as PR
    import page_scatter_v5 as SC
    argv = ["--rs", "8", "--g", "0.25", "--nsites", "768", "--channels", "1", "--ell", "2", "--Rb", "0.5", "--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t6"]
    if obs_path:
        argv += ["--obs-path", obs_path]
    a = PR.parse(argv)
    dev = torch.device(str(device))
    ch = CORE.Channel(1, 0.0, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    try:
        rows = SC.lattice_gamma(ch, st, a, (0.6, 1.2))
    except ImportError as ex:
        diag("T6 the stationary Gamma_lat at r_s/a 8 (labelled diagnostic): the obs libraries were not found", {"reason": str(ex)[:400]})
        return True
    # 29.9 (e), 30.1 (5), 31.4 (8): a labelled diagnostic; the stationary Gamma needs a trapped annulus on the sites
    diag("T6 the stationary Gamma_lat at r_s/a 8 against the continuum average, with the finite-window branch-current diagnostic (labelled diagnostic, 29.9 and 31.4 (8))",
         {"Gamma_lat_stationary": [r["Gamma_lat"] for r in rows], "Gamma_cont_avg": [r["Gamma_cont_avg"] for r in rows], "ratios": [r["ratio_lat_over_cont"] for r in rows],
          "stationary_available": [r["stationary"].get("ok") for r in rows], "stationary_reason": [r["stationary"].get("reason") for r in rows],
          "end_norm_inside_rt": [r["end_norm_inside_rt"] for r in rows], "reconversion_fraction": [r["reconversion_fraction"] for r in rows],
          "branch_current_diagnostic": [r["diagnostic_branch_current"]["Gamma_branch"] for r in rows],
          "calibration_ratio": [r["diagnostic_branch_current"]["calibration_ratio"] for r in rows],
          "x_mean": [r["x_mean"] for r in rows], "x_std": [r["x_std"] for r in rows], "label": "particle half, massless",
          "r_t_found_on_the_sites": [r["r_t_found"] for r in rows], "r_t_used": [r["r_t_used"] for r in rows]})
    return True


# --------------------------------------------------------------------------------------------------------------------- v6
V5_REPRO_KEYS = {"T4b": ["rel_err_Delta", "dropped_pair_share", "entropy_bound_dropped_pairs"], "T4c": ["E_inv_memory", "E_inv_dense"],
                 "T4d": ["max_|S_direct - S_memory|", "max_gram_defect", "S_esc_final"], "T4e": ["max_|S_B_Z - S_B|"],
                 "T4f ": ["max_rel_residual_uncompleted"], "T4g": ["S_direct", "S_memory", "gram_defect"], "T4h": ["max_identity_error"],
                 "T4i transition": ["rel_err", "S_continuity_defect"], "T4f-roundoff": ["max_rel_residual_uncompleted"],
                 "T4j": ["max_|S_resumed - S_uninterrupted|"], "T4k": ["repeated_vs_direct_rel"]}


def t_v5_repro(path):
    """v6: the v5-mode results on v6 against the stored v5 self-test (every listed number; equal to every digit, or within 1e-12 relative)."""
    if not path or not os.path.isfile(path):
        return skip("T-V5REPRO the v5-mode numbers against the stored v5 self-test", {"reason": "no --v5-selftest-results file: %s" % path})
    ref = json.load(open(path))
    refmap = {r_["test"]: r_ for r_ in ref.get("results", [])}
    mine = {r_["test"][: -len(" [v5 mode]")]: r_ for r_ in RESULTS if r_["test"].endswith(" [v5 mode]")}
    rows = []; n_eq = 0; n_cmp = 0; worst = 0.0
    for prefix, keys in V5_REPRO_KEYS.items():
        for name, r5 in refmap.items():
            if not name.startswith(prefix) or r5.get("diagnostic"):
                continue
            r6 = mine.get(name)
            if r6 is None:
                rows.append({"test": name, "missing_in_v6": True}); continue
            for k_ in keys:
                x5 = (r5.get("detail") or {}).get(k_); x6 = (r6.get("detail") or {}).get(k_)
                if not isinstance(x5, (int, float)) or not isinstance(x6, (int, float)):
                    continue
                n_cmp += 1; n_eq += int(x5 == x6)
                rel = abs(x6 - x5) / max(abs(x5), 1e-300)
                worst = max(worst, rel)
                rows.append({"test": name[:40], "key": k_, "v5": x5, "v6_v5_mode": x6, "equal": x5 == x6, "rel": rel})
    ok = n_cmp > 0 and worst <= 1e-12 and not any(r_.get("missing_in_v6") for r_ in rows)
    return rep("T-V5REPRO v6 in v5 mode against the stored v5 self-test (every listed number equal, or within 1e-12 relative)", ok,
               {"compared": n_cmp, "equal_to_every_digit": n_eq, "max_rel": worst, "rows": rows, "reference": path})


def t7a_hull_counterexample(device):
    """v6 (S1 repair 1): P01 sec. 2's counterexample re-derived on the split-memory form: two modes, P = diag(0, 1) (mode 1 occupied),
    C = [[1/2, b], [b, 1 - 2 b^2]], b = 2e-7, the memory c = 1 (column 0 the empty half, column 1 the occupied half), M_Delta = C - P;
    eps_rank 1e-9 keeps one eigenvector whose occupied-half part has squared norm about 1.6e-13 < eps_orth 1e-12; the probe q = (0.2,
    0.96^(1/2)): the entropy error |s(q^+ C q) - s(q^+ C' q)| of the reset C' by v5's hull against the exact hull's."""
    dev = torch.device(str(device))
    b = 2e-7
    C = torch.tensor([[0.5, b], [b, 1.0 - 2.0 * b * b]], dtype=CD, device=dev)
    P = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=CD, device=dev)
    MD = C - P
    c = torch.eye(2, dtype=CD, device=dev); half = torch.tensor([1, -1], dtype=torch.int64, device=dev)
    lam, uu = torch.linalg.eigh(MD)
    keep = lam.abs() > 1e-9
    q = torch.tensor([0.2, math.sqrt(0.96)], dtype=CD, device=dev)
    def s_(x):
        return OPS.binary_entropy_np([x])
    n_exact = float((q.conj() @ C @ q).real)
    out = {"eigenvalues": [float(x) for x in lam], "kept": int(keep.sum())}
    for mode in ("v5", "exact"):
        cf, hf = OPS.hull_in_coords(c, half, uu[:, keep], 1e-12, mode=mode)
        T = cf.mH @ c
        MDf = T @ MD @ T.mH
        Cr = P + cf @ MDf @ cf.mH
        n_r = float((q.conj() @ Cr @ q).real)
        out[mode] = {"hull_columns": int(cf.shape[1]), "entropy_error": abs(s_(n_exact) - s_(n_r)), "n_exact": n_exact, "n_reset": n_r}
    allowance = 2 * OPS.binary_entropy_scalar(1e-9)
    ok = out["exact"]["entropy_error"] < 1e-14 and out["v5"]["entropy_error"] > allowance
    out["v5_allowance_2_s(eps_rank)"] = allowance
    return rep("T7a the hull counterexample (P01 sec. 2): v5's hull errs beyond its allowance, the exact hull (S1 repair 1) at roundoff", ok, out)


def t7b_pair_ledger():
    """v6 (S1 repair 2): the exact F(s) = 2 max over 0 <= t <= pi/4 - theta/2 of h[sin^2(t + theta/2)] - h(sin^2 t), theta = arcsin s
    (SHARP_ENTROPY_THEOREM's statement), against pair_ledger_sharp's envelope and v5's h(s^2). The difference is evaluated without
    cancellation as the integral of its derivative, h[sin^2(t + theta/2)] - h(sin^2 t) = int_t^(t + theta/2) 2 sin(2u) ln(cot u) du
    (scipy quad), maximized over t by a grid and a bounded refinement (the first run's direct difference of order-one h values lost
    1e-16 absolute, 1.6e-7 relative at s = 1e-9)."""
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar

    def G(t, th):
        val, _ = quad(lambda u: 2.0 * math.sin(2.0 * u) * math.log(1.0 / math.tan(u)) if u > 0 else 0.0, t, t + th / 2.0, epsabs=0.0, epsrel=1e-13, limit=200)
        return val

    def F_exact(s):
        th = math.asin(s); tmax = math.pi / 4.0 - th / 2.0
        ts = np.linspace(0.0, tmax, 801)
        vals = [G(t, th) for t in ts]
        i = int(np.argmax(vals))
        lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
        res = minimize_scalar(lambda t: -G(t, th), bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
        return 2.0 * max(max(vals), -res.fun)
    rows = []; ok = True
    for s_ in (1e-9, 1e-7, 1e-4, 1e-2, 0.2, 0.5, 0.9):
        Fe = F_exact(s_); Fu = OPS.pair_ledger_sharp([s_]); hv5 = OPS.pair_ledger_v5([s_])
        good = Fe <= Fu * (1.0 + 1e-9) + 1e-300 and (s_ > 1e-2 or hv5 < Fe)
        ok &= good
        rows.append({"s": s_, "F_exact": Fe, "envelope": Fu, "v5_h_s2": hv5, "ok": good})
    return rep("T7b the sharp dropped-pair ledger (S1 repair 2): F(s) at most min(C* arcsin s, 2h(min(s, 1/2))) at seven sines; v5's h(s^2) below F(s) at s <= 1e-2", ok,
               {"C_star": OPS.C_STAR, "rows": rows})


T8_ARGV = ["--rs", "12", "--g", "0.25", "--nsites", "256", "--channels", "1", "--ell", "2", "--Rb", "0.5", "--Rb0", "3", "--R-cut", "2.5", "--R-collar", "1.5",
           "--tau-s", "1", "--m-cap", "100000", "--eps-coll", "1e-5", "--eps-form", "1e-7", "--eps-orth", "1e-12", "--store-cap", "100000",
           "--hull", "exact", "--pair-ledger", "sharp", "--collector", "active", "--eps-loc", "1e-2", "--direct-every", "1"]


def t8_dense_oracle(device, engine_path=None, eps_rank="0", n_blocks=14):
    """v6 (37.1 (3)): the dense oracle on the builder's fixture with a trapping zero (the module docstring): the recursion's every readout
    against the dense covariance C(t) = U(t) C_0 U(t)^dagger of the one stage, every collected mode propagated to the block's clock."""
    import page_recursion_v6b as PR
    argv = T8_ARGV + ["--eps-rank", eps_rank, "--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t8"] + (["--engine-path", engine_path] if engine_path else [])
    a = PR.parse(argv)
    dev = torch.device(str(device))
    ch = CORE.Channel(1, a.m, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    Phi0, frec = ch.formation(a)
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    irec = ch.initial_memory(Phi0, a)
    V, e, neg = st["V"], st["e"], st["neg"]
    fixture = {"r_s": a.rs, "nsites": a.nsites, "ell": a.ell, "ell_over_2GE": a.ell / a.rs, "r_t": st["r_t"], "r_in": st["r_in"],
               "max_abs_v": float(np.max(np.abs(st["v_sites"]))), "R_cut_sites": a.R_cut_abs, "R_collar_sites": a.R_collar_abs, "tau_s": a.tau_s_abs,
               "memory_rank": irec["memory_rank"], "E_inv": irec["E_inv"], "eps_rank": a.eps_rank, "eps_loc": a.eps_loc, "formation_unitarity": frec["unitarity"]}
    P1 = V[:, neg] @ V[:, neg].mH
    C0_exact = Phi0 @ Phi0.mH
    Dinit = V @ ch.rec["c_init"]
    C0 = P1 + Dinit @ ch.rec["MD_init"] @ Dinit.mH                             # the recursion's own initial state (the memory route's Delta_0)
    fixture["formation_truncation_rel"] = float(torch.linalg.norm(C0 - C0_exact)) / max(float(torch.linalg.norm(C0_exact - P1)), 1e-300)
    fixture["dropped_pair_share_printed"] = irec["dropped_pair_share"]
    h_d = OPS.dense_from_csr(st["h"], dev)
    R2 = ch.R2
    chi_col = (R2 > a.R_collar_abs).to(CD)
    out_mask = (R2 > a.R_cut_abs)
    E_init = ch.rec["E_delta_init"]
    worst = {"S_mem_vs_dense": 0.0, "S_B_vs_dense": 0.0, "Z_exact": 0.0, "gram": 0.0, "energy_identity": 0.0, "collar_vs_dense": 0.0,
             "outside_weight_min": 1.0, "band_last": 0.0, "S_mem_vs_exact_formation_state": 0.0, "E_out_vs_exact_formation_state_rel": 0.0}
    rows = []; collected = []; n_coll = 0

    # the stage start
    ci0 = ch.collar_identity(a, t=0.0)
    Dt0 = C0 - P1
    E_out0_d = float(torch.trace(chi_col[:, None] * (h_d @ Dt0)).real)
    worst["collar_vs_dense"] = max(worst["collar_vs_dense"], abs(ci0["E_out"] - E_out0_d) / max(abs(E_init), 1e-300))
    t = 0.0
    for n in range(n_blocks):
        t += a.tau_s_abs
        r = ch.block_step(a, t)
        Ct = dense_state(V, e, C0, t); Dt = Ct - P1
        if r["m_coll"]:
            cq, hq, tq = ch.last_block
            collected.append((cq.clone(), tq)); n_coll += int(cq.shape[1])
            Qb = V @ cq
            SBd = OPS.entropy_cov(Qb.mH @ Ct @ Qb)[0]
            worst["S_B_vs_dense"] = max(worst["S_B_vs_dense"], abs(SBd - r["S_B"]))
            worst["Z_exact"] = max(worst["Z_exact"], abs(r["S_B_Z_exact"] - r["S_B"]))
            wout = (Qb.abs().pow(2) * out_mask.to(RD)[:, None]).sum(dim=0)
            worst["outside_weight_min"] = min(worst["outside_weight_min"], float(wout.min()))
        if collected:
            Q = torch.cat([V @ (torch.exp(-1j * e.to(CD) * (t - tq_))[:, None] * cq_) for cq_, tq_ in collected], dim=1)
            gram = float(torch.linalg.norm(Q.mH @ Q - torch.eye(Q.shape[1], dtype=CD, device=dev)))
            Sd = OPS.entropy_cov(Q.mH @ Ct @ Q)[0]
            Ct_x = dense_state(V, e, C0_exact, t)
            worst["S_mem_vs_exact_formation_state"] = max(worst["S_mem_vs_exact_formation_state"], abs(OPS.entropy_cov(Q.mH @ Ct_x @ Q)[0] - r["S_new"]))
            worst["gram"] = max(worst["gram"], gram)
            worst["S_mem_vs_dense"] = max(worst["S_mem_vs_dense"], abs(Sd - r["S_new"]))
            Dm = V @ ch.c
            E_coll = float(torch.trace((Q.mH @ Dt @ Q) @ (Q.mH @ h_d @ Q)).real)
            E_x = 2.0 * float(torch.trace((Q.mH @ Dt @ Dm) @ (Dm.mH @ h_d @ Q)).real) if Dm.shape[1] else 0.0
            worst["energy_identity"] = max(worst["energy_identity"], abs(E_init - (r["E_mem"] + E_coll + E_x)) / max(abs(E_init), 1e-300))
            K = torch.eye(ch.n2, dtype=CD, device=dev) - Q @ Q.mH
            E_K_d = float(torch.trace(h_d @ K @ Dt @ K).real)
        else:
            Sd = 0.0; E_K_d = float(torch.trace(h_d @ Dt).real)
        E_out_d = float(torch.trace(chi_col[:, None] * (h_d @ Dt)).real); E_tot_d = float(torch.trace(h_d @ Dt).real)
        E_out_x = float(torch.trace(chi_col[:, None] * (h_d @ (dense_state(V, e, C0_exact, t) - P1))).real)
        worst["E_out_vs_exact_formation_state_rel"] = max(worst["E_out_vs_exact_formation_state_rel"], abs(E_out_x - E_out_d) / max(abs(E_init), 1e-300))
        ci = ch.collar_identity(a, t=t)
        M_chi_d = E_out_d - E_tot_d + E_K_d
        dc = max(abs(ci["E_out"] - E_out_d), abs(ci["E_tot"] - E_tot_d), abs((ci["E_K"] if ci["E_K"] is not None else 0.0) - E_K_d),
                 abs((ci["M_chi"] if ci["M_chi"] is not None else 0.0) - M_chi_d)) / max(abs(E_init), 1e-300)
        worst["collar_vs_dense"] = max(worst["collar_vs_dense"], dc)
        band = ch.ent_bound + ch.ent_bound_gram
        worst["band_last"] = band
        rows.append({"block": n + 1, "t_over_rs": t / a.rs, "m_coll": r["m_coll"], "S_memory": r["S_new"], "S_dense": Sd, "S_B": r["S_B"],
                     "E_out": ci["E_out"], "E_remove": ci["E_remove"], "M_chi": ci["M_chi"], "identity_residual": ci["residual"], "band": band,
                     "outside_weight_remaining": r.get("weight_beyond_cut_uncollected")})
    exact = float(a.eps_rank) == 0.0
    if exact:
        ok = (st["r_t"] is not None and n_coll > 0 and worst["S_mem_vs_dense"] < 1e-8 and worst["S_B_vs_dense"] < 1e-8 and worst["Z_exact"] < 1e-10
              and worst["gram"] < 1e-10 and worst["energy_identity"] < 1e-8 and worst["collar_vs_dense"] < 1e-10
              and worst["outside_weight_min"] >= 1.0 - a.eps_loc - 1e-12)
        name = "T8 the dense oracle (37.1 (3)) on a fixture with a trapping zero at eps_rank 0: every readout of the v6 recursion against the dense state of its own initial memory (the exact formation state's differences printed)"
    else:
        ok = (st["r_t"] is not None and n_coll > 0 and worst["S_mem_vs_dense"] <= max(worst["band_last"], 1e-8) and worst["gram"] < 1e-10
              and worst["outside_weight_min"] >= 1.0 - a.eps_loc - 1e-12)
        name = "T8 the dense oracle at eps_rank 1e-9: the memory route within its ledger band of the dense entropy"
    return rep(name, ok, {"fixture": fixture, "n_collected": n_coll, "worst": worst, "stage_start": {"E_out0": ci0["E_out"], "E_out0_dense": E_out0_d}, "rows": rows})


def run_all(a):
    """v6: the v5 tests in v5 mode (with T-V5REPRO against the stored v5 self-test) and in v6 mode, then T7a, T7b and T8 (two eps_rank)."""
    global MODE
    device = a.device if (str(a.device).startswith("cpu") or torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(a.threads)
    t0 = time.time()
    ok = True
    ok &= t1_operator(getattr(a, "engine_path", None))
    ok &= t2_kernel()
    ok &= t3_recursion_identities()
    for mode in ("v5", "v6"):
        MODE = mode
        ok &= t4_small_case(device, getattr(a, "engine_path", None))
        ok &= t4f_roundoff(device, getattr(a, "engine_path", None))
        ok &= t4j_driver_resume(device, getattr(a, "engine_path", None), os.path.join(a.out or "<internal>/page_selftest", "t4j_" + mode))
        ok &= t4k_restore(device, getattr(a, "engine_path", None))
        if mode == "v5":
            ok &= t_v5_repro(getattr(a, "v5_selftest_results", None))
    MODE = "v6"
    ok &= t5_continuum()
    ok &= t7a_hull_counterexample(device)
    ok &= t7b_pair_ledger()
    ok &= t8_dense_oracle(device, getattr(a, "engine_path", None), eps_rank="0")
    ok &= t8_dense_oracle(device, getattr(a, "engine_path", None), eps_rank="1e-9")
    t6_lattice_gamma(device, getattr(a, "obs_path", None))            # a labelled diagnostic (29.9 (e)), not counted
    out = a.out or "<internal>/page_selftest"
    os.makedirs(out, exist_ok=True)
    skipped = [r["test"] for r in RESULTS if r.get("skipped")]
    with open(os.path.join(out, "selftest_results.json"), "w") as f:
        json.dump({"all_pass": bool(ok), "skipped": skipped, "results": RESULTS, "seconds": time.time() - t0, "device": str(device),
                   "code": "page_selftest_v6b.py"}, f, indent=1, default=float)
    print("SELFTEST %s in %.0f s%s (results in %s)" % ("PASS" if ok else "FAIL", time.time() - t0, (" (skipped: %s)" % skipped) if skipped else "",
                                                       os.path.join(out, "selftest_results.json")), flush=True)
    return bool(ok)
