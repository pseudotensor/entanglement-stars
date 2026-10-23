#!/usr/bin/env python
"""test_pt3_stage1.py (busi, PAGE design v3 stage 1: unit tests, no physics claim). One test per v3 piece, known answers where they exist:
  T1  continuation arms on the cavity-box two-level model, lambda = 0.1, 1, 10 (A1): the physical Landau-Zener probability e^-lambda; R-an keeps
      the branch (hands the cavity occupation to the exterior level); R-loc transports it with a jump of -2|V| (step convergence); R-code (the
      engine's rule) equals R-an when the record step resolves the crossing; R-code at the production c_LZ ladder reported per lambda.
  T2  one level on a ladder (Demkov-Osherov): survival exp(-sum 2 pi V^2/s); R-an hands the occupation to the first exterior level crossed;
      R-loc keeps it, its jumps summing to 2 d_omega (Gamma/(2 pi Delta))^(1/2).
  T3  the symmetry sectors (A1): [S_stag, h(t)] = 0 along a formation-like history, the sector route's spectrum equal to the dense one, the
      sector counts of R-an preserved, the exact-crossing flag on a constructed degeneracy.
  T4  the preparations (A2): pure P_-(h_0) by the sector route against dense; f_ws diagnostics exact against KPM; the sharp-step projector at a
      width below the measured min |e| against P_-; its idempotency defect.
  T5  the constructed depleted arm (F5): 0 <= C_dep <= 1, idempotent, Tr s = 0.
  T6  the packet-charge calibration (A3, F6) on the paper's parent at T/r_s 100 (record P64_s4_B2n4096, the engine's pure propagated sea):
      particle and hole, one copy and 16 copies, read through the engine's Gauss rows on the enclosing surface; the packet's energy.
  T7  the finite-radius ledger (A4) on the exactly solvable leaking-orbital control with a sweep: every arm's identity closed to the cadence
      floor, the held control's floor, R-star's driven identity, the interface term zero for commuting arms, R-Q's block identity, cadence
      convergence; the production KPM sea trace against dense; the absorber split and the eigenstate arms' refusal of a non-Hermitian h.
  T8  the nested orthonormal outgoing modes (A8, F9): S(R_n), S(b|R) against the closed form s(q) on a pure single-particle drain; earlier bins
      unchanged by later ones; certified intervals containing the truth under a perturbation; the pure-state floor.
  T9  the certified unitary exterior >= 1e4 r_s at r_s/a 64 (644,096 sites on the record's exterior): exact unitarity (forward-backward),
      norm drift, Hermiticity, the first return from the symbol's fastest carrier; the flat chain's group velocity 1.
  T10 the level tracking (A6, A5): a static exterior's level count against the standing-wave rule; the resonance width from the Lorentzian fit
      against the golden rule; the three endpoints and lambda_j on a synthetic trajectory; on the stored record the k = 1 inverted pair at
      +-3.95 T_H (v1's pilot value) found by the tracker.
  T11 R-an against v1's held control on the stored record (T/r_s 100): h(T_hold) from the slice's applied fields reproduces the stored spectrum;
      R-an initialized from the stored reference's occupations (0/1, equal to the engine's mask); held steps leave P unchanged (Tr(h dP) = 0); the
      four inverted levels (2 rising, 2 pocket); <chi|P_an - P_-|chi> on v1's exterior packets below v1's table precision, so R-an's held read is
      v1's held control.
  T12 the readers and the conflict options: every line of v3 sec. 10 on synthetic inputs, every C1 to C6 side enumerated.
Usage (box): nice -n 10 python test_pt3_stage1.py --out <internal>/stage1 [--only T1,T2] [--skip-record]"""
import argparse
import math
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K  # noqa: E402
import pt3_refs as RF  # noqa: E402
import pt3_prep as PR  # noqa: E402
import pt3_ledger as LG  # noqa: E402
import pt3_calib as CB  # noqa: E402
import pt3_output as OUT  # noqa: E402
import pt3_levels as LV  # noqa: E402
import pt3_readers as RD  # noqa: E402
import pt3_options as OPT  # noqa: E402

SLICE = "<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_T0100.00.npz"
LABEL, TREC = "P64_s4_B2n4096", 100.0
_REC = {}


def chk(name, ok, **kw):
    d = {"check": name, "verdict": "PASS" if bool(ok) else "FAIL"}
    d.update(kw)
    return d


# --------------------------------------------------------------------------------------------------------- the fixtures
def prop_2level(Hf, t0, t1, dt, psi0):
    """exact-to-O(dt^2) propagation with the midpoint exponential (2 x 2 or small dense)."""
    psi = np.array(psi0, complex)
    n = int(math.ceil((t1 - t0) / dt)); dt = (t1 - t0) / n
    for k in range(n):
        tm = t0 + (k + 0.5) * dt
        e, U = np.linalg.eigh(Hf(tm))
        psi = U @ (np.exp(-1j * e * dt) * (U.conj().T @ psi))
    return psi


def spec_of(H, t, f=None, sector_op=None):
    e, U, sec = K.dense_spectrum(sp.csr_matrix(H), sector_op)
    return RF.Spec(e, U, sec, t=t, f=f, h=sp.csr_matrix(H))


def run_arms_sequence(Hf, ts, occ0, arms, ff=None):
    s0 = spec_of(Hf(ts[0]), ts[0], ff(ts[0]) if ff else None)
    for a in arms:
        a.start(s0, occ0(s0))
    for t in ts[1:]:
        s = spec_of(Hf(t), t, ff(t) if ff else None)
        for a in arms:
            a.step(s)
    return s


# ---------------------------------------------------------------------------------------------------------------- T1
def T1():
    out = {"rows": [], "checks": []}
    s = 1.0
    for lam in (0.1, 1.0, 10.0):
        V = math.sqrt(lam * s / (2 * math.pi))
        T = 1500.0                                             # the finite-time correction ~ V/(s T) kept below 1e-3
        Hf = lambda t, V=V: np.array([[-s * t, V], [V, 0.0]])
        psi = prop_2level(Hf, -T, T, 0.01, [1.0, 0.0])
        P_dia = float(abs(psi[0]) ** 2)
        row = {"lambda": lam, "V": V, "P_diabatic": P_dia, "exp_minus_lambda": math.exp(-lam)}
        out["checks"].append(chk("T1 physical LZ lambda %g" % lam, abs(P_dia - math.exp(-lam)) < 2e-3, P=P_dia, expected=math.exp(-lam)))
        qmask = np.array([True, False])
        jumps = []
        for dt_ref in (0.2, 0.1, 0.05):
            ts = np.arange(-40.0, 40.0 + 1e-9, dt_ref)
            occ0 = lambda sp_: np.array([0.0, 1.0])            # at t = -40 the cavity (energy +40) is the upper branch, filled
            ran = RF.RAn("rank"); rov = RF.RAn("overlap"); rl = RF.RLoc(qmask, delta_c=3.0 * V, R_Q="cavity")
            rl_small = RF.RLoc(qmask, delta_c=0.5 * V, R_Q="cavity")
            rl_ee = RF.RLoc(qmask, delta_c=3.0 * V, R_Q="cavity")         # equal occupations (empty-empty): an event, no exchange, zero jump
            fam = RF.LinearFamily([np.array([[0.0, V], [V, 0.0]]), np.array([[-1.0, 0.0], [0.0, 0.0]])])
            ff = lambda t: (np.array([1.0, s * t]), np.zeros(2))
            rc = RF.RCode(fam, tau=dt_ref, c_LZ=1.0, window_max=4)
            sl = run_arms_sequence(Hf, ts, occ0, [ran, rov, rl, rl_small, rc], ff)
            run_arms_sequence(Hf, ts, lambda sp_: np.array([0.0, 0.0]), [rl_ee])
            wc = lambda a: float(RF.w_in(sl.U, qmask) @ a.occ)
            jumps.append(rl.jump_sum())
            row.setdefault("by_step", []).append({"dt_ref": dt_ref, "Ran_final_cavity_weight": wc(ran), "Ran_overlap_final_cavity_weight": wc(rov),
                                                  "Rloc_final_cavity_weight": wc(rl), "Rloc_jump_sum": rl.jump_sum(), "Rloc_jump_applied": rl.jump_sum_applied(), "Rloc_n_jumps": len(rl.jumps),
                                                  "Rloc_empty_empty_events": len(rl_ee.jumps), "Rloc_empty_empty_jump": rl_ee.jump_sum(),
                                                  "Rloc_small_delta_c_kept_analytic": len(rl_small.kept_analytic), "Rloc_small_final_cavity_weight": wc(rl_small),
                                                  "Rcode_final_cavity_weight": float(RF.w_in(sl.U, qmask) @ rc.occ),
                                                  "Ran_min_gap_occ_empty": min(x["min_gap_occ_empty"] for x in ran.log),
                                                  "Ran_sum_tr_h_dP": float(sum(x.get("tr_h_dP") or 0.0 for x in ran.log[1:]))})
        b = row["by_step"][-1]
        out["checks"].append(chk("T1 R-an hands the occupation to the exterior level, lambda %g" % lam, b["Ran_final_cavity_weight"] < 0.01,
                                 value=b["Ran_final_cavity_weight"]))
        out["checks"].append(chk("T1 R-an(overlap) = R-an at resolved steps, lambda %g" % lam, abs(b["Ran_overlap_final_cavity_weight"] - b["Ran_final_cavity_weight"]) < 1e-6))
        out["checks"].append(chk("T1 R-loc keeps the cavity occupation, lambda %g" % lam, b["Rloc_final_cavity_weight"] > 0.99, value=b["Rloc_final_cavity_weight"]))
        out["checks"].append(chk("T1 R-loc jump = -2|V| (step converged to 1%%), lambda %g" % lam, abs(jumps[-1] + 2 * V) < 0.01 * 2 * V,
                                 jumps=jumps, expected=-2 * V))
        out["checks"].append(chk("T1 R-loc empty-empty crossing: event located, zero jump, lambda %g" % lam,
                                 b["Rloc_empty_empty_events"] == 1 and b["Rloc_empty_empty_jump"] == 0.0, events=b["Rloc_empty_empty_events"]))
        out["checks"].append(chk("T1 R-loc with delta_c below the minigap stays analytic, lambda %g" % lam,
                                 b["Rloc_small_kept_analytic"] if False else (b["Rloc_small_delta_c_kept_analytic"] >= 1 and b["Rloc_small_final_cavity_weight"] < 0.01)))
        out["checks"].append(chk("T1 R-code (engine rule, record step resolving the crossing) = R-an, lambda %g" % lam,
                                 abs(b["Rcode_final_cavity_weight"] - b["Ran_final_cavity_weight"]) < 1e-6, value=b["Rcode_final_cavity_weight"]))
        out["checks"].append(chk("T1 R-an Tr(h dP) -> 0 along the branch, lambda %g" % lam, abs(b["Ran_sum_tr_h_dP"]) < 1e-3 * V, value=b["Ran_sum_tr_h_dP"]))
        # R-code at coarse records (the c_LZ ladder 1/2, 1, 2), reported: which convention the engine's rule realizes per lambda
        rep = []
        for tau in (8.0, 32.0):
            for cLZ in (0.5, 1.0, 2.0):
                ts = np.arange(-64.0, 64.0 + 1e-9, tau)
                fam = RF.LinearFamily([np.array([[0.0, V], [V, 0.0]]), np.array([[-1.0, 0.0], [0.0, 0.0]])])
                rc = RF.RCode(fam, tau=tau, c_LZ=cLZ)
                sl = run_arms_sequence(Hf, ts, lambda sp_: np.array([0.0, 1.0]), [rc], lambda t: (np.array([1.0, s * t]), np.zeros(2)))
                rep.append({"tau": tau, "c_LZ": cLZ, "final_cavity_weight": float(RF.w_in(sl.U, qmask) @ rc.occ),
                            "modes": sorted(set(str(x.get("mode")) for x in rc.log[1:]))})
        row["Rcode_coarse_records_reported"] = rep
        out["rows"].append(row)
    return out


# ---------------------------------------------------------------------------------------------------------------- T2
def T2():
    out = {"checks": []}
    M, Dl, V, lam = 10, 1.0, 0.05, 0.05
    s = 2 * math.pi * V * V / lam
    n = 2 * M + 1
    ej = Dl * np.arange(-M, M + 1)
    def Hf(t):
        H = np.zeros((n + 1, n + 1))
        H[0, 0] = -s * t
        H[0, 1:] = V; H[1:, 0] = V
        H[np.arange(1, n + 1), np.arange(1, n + 1)] = ej
        return H
    T0 = (M + 4) * Dl / s
    psi = prop_2level(Hf, -T0, T0, 0.01, np.eye(n + 1)[0])
    surv = float(abs(psi[0]) ** 2)
    DO = math.exp(-n * 2 * math.pi * V * V / s)
    out["checks"].append(chk("T2 physical Demkov-Osherov survival", abs(surv - DO) < 0.02, survival=surv, DO=DO, n_cross=n))
    qmask = np.zeros(n + 1, bool); qmask[0] = True
    ts = np.linspace(-T0, T0, 4001)
    occ0 = lambda s0: (np.arange(n + 1) == n).astype(float)     # the top eigenvector (the cavity at +s T0) filled, the ladder empty
    ran = RF.RAn("rank"); rl = RF.RLoc(qmask, delta_c=4 * V, R_Q="cavity")
    sl = run_arms_sequence(Hf, ts, occ0, [ran, rl])
    occ_vec_an = sl.U @ (ran.occ * 0 + 0) if False else sl.U[:, ran.occ > 0.5][:, 0]
    first = np.zeros(n + 1); first[n] = 1.0                       # the first exterior level crossed by the descending cavity: e_j = +M
    ov_first = float(abs(np.vdot(first, occ_vec_an)) ** 2)
    out["checks"].append(chk("T2 R-an hands the occupation to the first exterior level crossed", ov_first > 0.99, overlap=ov_first))
    wl = float(RF.w_in(sl.U, qmask) @ rl.occ)
    out["checks"].append(chk("T2 R-loc keeps the occupation in the cavity", wl > 0.99, cavity_weight=wl))
    Gam = 2 * math.pi * V * V / Dl
    bound = 2 * (n * Dl) * math.sqrt(Gam / (2 * math.pi * Dl))
    js = abs(rl.jump_sum())
    out["checks"].append(chk("T2 R-loc jump sum = 2 d_omega sqrt(Gamma/2 pi Delta) within 5%", abs(js - bound) < 0.05 * bound, jump_sum=js, formula=bound,
                             n_jumps=len(rl.jumps)))
    return out


# ---------------------------------------------------------------------------------------------------------------- T3
def chain_fields(R, t, rs=8.0):
    """a formation-like history on a small chain: a lapse well and a flow deepening with t in [0, 1]."""
    N = 1.0 - 0.5 * t * np.exp(-(R / (1.5 * rs)) ** 2)
    v = -t * np.sqrt(2 * 0.25 * rs * 2 / np.maximum(R, 1.0)) * (1 - np.exp(-R / rs))
    return N, v


def T3():
    out = {"checks": []}
    ops = K.ChainOps(64, 1.0)
    import page_w1_sector_v1d as W1S
    S = W1S.stag_sparse(ops.n)
    ts = np.linspace(0, 1, 11)
    comm = []; dspec = []; lab = []
    seq = []
    for t in ts:
        h = ops.h(*chain_fields(ops.R, t))
        comm.append(float(abs(S @ h - h @ S).max()))
        e1, U1, s1 = K.full_spectrum_from_sector(h)
        e2, U2, s2 = K.dense_spectrum(h, S)
        dspec.append(float(np.max(np.abs(e1 - e2))))
        lab.append(int(np.sum(np.sort(s1[e1 < 0]) != np.sort(s2[e2 < 0]))))
        seq.append(RF.Spec(e1, U1, s1, t=t, h=h))
    out["checks"].append(chk("T3 [S_stag, h(t)] = 0 along the history", max(comm) == 0.0, max_comm=max(comm)))
    out["checks"].append(chk("T3 sector-route spectrum = dense", max(dspec) < 1e-10, max_diff=max(dspec)))
    out["checks"].append(chk("T3 sector labels agree", sum(lab) == 0, mismatches=sum(lab)))
    ran = RF.RAn("rank")
    ran.start(seq[0], (seq[0].e < 0).astype(float))
    for sq in seq[1:]:
        ran.step(sq)
    cnt0 = {s_: int(((seq[0].sec == s_) & (seq[0].e < 0)).sum()) for s_ in (1, -1)}
    cnt1 = {s_: int(ran.occ[seq[-1].sec == s_].sum()) for s_ in (1, -1)}
    out["checks"].append(chk("T3 R-an sector counts preserved", cnt0 == cnt1, start=cnt0, end=cnt1,
                             min_gap_occ_empty=min(x["min_gap_occ_empty"] for x in ran.log), flags=len(ran.flags)))
    # constructed exact crossing in one sector (flagged) and across sectors (not flagged)
    def mk(e, sec, t):
        return RF.Spec(np.array(e), np.eye(len(e)), np.array(sec), t=t)
    a = RF.RAn("rank"); a.start(mk([-1.0, 1.0], [1, 1], 0), np.array([1.0, 0.0])); a.step(mk([0.0, 0.0], [1, 1], 1))
    b = RF.RAn("rank"); b.start(mk([-1.0, 1.0], [1, -1], 0), np.array([1.0, 0.0])); b.step(mk([0.0, 0.0], [1, -1], 1))
    out["checks"].append(chk("T3 exact same-sector crossing flagged, cross-sector not", len(a.flags) == 1 and len(b.flags) == 0))
    return out


# ---------------------------------------------------------------------------------------------------------------- T4, T5
def T4():
    out = {"checks": []}
    ops = K.ChainOps(200, 1.0)
    N, v = chain_fields(ops.R, 1.0, rs=20.0)
    h = ops.h(N, v)
    pr = PR.pure_sector(h)
    e, U = np.linalg.eigh(h.toarray())
    Pd = U[:, e < 0] @ U[:, e < 0].conj().T
    Ps = pr["V"] @ pr["V"].conj().T
    out["checks"].append(chk("T4 pure P_-(h_0): sector route = dense", float(np.max(np.abs(Ps - Pd))) < 1e-10, max_diff=float(np.max(np.abs(Ps - Pd))),
                             n_neg=pr["n_neg"], zero_modes=pr["n_zero_1e-12"]))
    out["checks"].append(chk("T4 pure: idempotent (Gram trace-norm defect)", pr["gram_defect_1"] < 1e-10, defect=pr["gram_defect_1"]))
    ws = 0.02
    ex = PR.mixed_diagnostics_exact(e, ws)
    kp = PR.mixed_diagnostics_kpm(h, ws, n_probe=48)
    ok1 = abs(kp["trace_norm_C0_minus_P"] - ex["trace_norm_C0_minus_P"]) <= 4 * kp["trace_norm_se"] + 1e-6
    ok2 = abs(kp["S_C0"] - ex["S_C0"]) <= 4 * kp["S_C0_se"] + 1e-6
    out["checks"].append(chk("T4 mixed f_ws: ||C0 - P||_1 KPM = exact within 4 se", ok1, exact=ex["trace_norm_C0_minus_P"], kpm=kp["trace_norm_C0_minus_P"], se=kp["trace_norm_se"]))
    out["checks"].append(chk("T4 mixed f_ws: S(C0) KPM = exact within 4 se", ok2, exact=ex["S_C0"], kpm=kp["S_C0"], se=kp["S_C0_se"],
                             near_half=ex["n_near_half_(0.1,0.9)"]))
    mn, cnt = PR.min_abs_eig_banded(h, 0.2)
    out["checks"].append(chk("T4 min |e| from the banded eigenvalues = dense", abs(mn - float(np.min(np.abs(e)))) < 1e-12, banded=mn, dense=float(np.min(np.abs(e)))))
    ws_s = mn / 8.0
    rng = np.random.default_rng(5)
    X = rng.standard_normal((h.shape[0], 3)) + 1j * rng.standard_normal((h.shape[0], 3))
    Y, M = PR.sharp_step_apply(h, X, ws_s)
    err = float(np.max(np.abs(Y - Pd @ X)))
    out["checks"].append(chk("T4 sharp-step projector at ws = min|e|/8 = P_- on probes", err < 1e-8, max_err=err, cheb_terms=M,
                             defect_exact=PR.idempotency_defect_exact(e, ws_s)))
    return out


def T5():
    out = {"checks": []}
    ops = K.ChainOps(200, 1.0)
    h = ops.h(*chain_fields(ops.R, 1.0, rs=20.0))
    e, U = np.linalg.eigh(h.toarray())
    Cn = U[:, e < 0] @ U[:, e < 0].conj().T
    Cd, rep = PR.depleted_projector(h, ops.site_mask(40.0), Cn)
    out["checks"].append(chk("T5 C_dep: 0 <= C <= 1", rep["min_eig"] > -1e-12 and rep["max_eig"] < 1 + 1e-12, **rep))
    out["checks"].append(chk("T5 C_dep idempotent and pure", rep["idempotency_defect_F"] < 1e-10 and rep["Tr_s"] < 1e-8, defect=rep["idempotency_defect_F"], Tr_s=rep["Tr_s"]))
    return out


# --------------------------------------------------------------------------------------------------------- the record
def record():
    if "H" in _REC:
        return _REC
    sys.path.insert(0, os.path.dirname(HERE))
    import pt_fieldsA_v1 as FA
    import read_history_v1 as RH
    H = FA.load_record(LABEL, TREC)
    Z = np.load(SLICE, allow_pickle=True)
    FM = RH.field_module()
    _, _, _, info = FM.solve_constraints_ivp(H["eps_tot"], H["J_tot"], H["g"], H["R_sites"], H["ell"], collar_policy="flag", support_rel=1e-14)
    Na, va, _ = FM.site_fields_kernel_average(info, H["R_sites"], H["ell"])
    _REC.update({"H": H, "Z": Z, "FM": FM, "N_avg": np.asarray(Na, float), "v_avg": np.asarray(va, float), "info0": info,
                 "N_app": np.asarray(Z["N_applied"], float), "v_app": np.asarray(Z["v_applied"], float)})
    return _REC


def T_H_rec(H):
    return 1.0 / (8 * math.pi * H["g"] * H["E_rec"])


def probe_vacuum():
    """the massless k = 1 probe on the record's applied fields at T_hold (v1's Test A convention) and its exact Killing vacuum P_-(h(T_hold))
    by the sector route: the pure held preparation of the calibration (sec. 1(b) on the held operator)."""
    rc = record()
    if "probe" not in _REC:
        H = rc["H"]
        ops = K.ChainOps(len(H["R_sites"]), 1.0)
        h = ops.h(rc["N_app"], rc["v_app"])
        e, U, sec = K.full_spectrum_from_sector(h)
        _REC["probe"] = {"ops": ops, "h": h, "e": e, "U": U, "sec": sec}
    return _REC["probe"]


def T6():
    out = {"checks": []}
    rc = record()
    H = rc["H"]
    import pt_lib_v1 as L1
    pv = probe_vacuum()
    ops = pv["ops"]
    # the calibration's operator is h[N_avg, v_avg] (the averaged rows' generator); its exact vacuum is the pure preparation
    h_avg = ops.h(rc["N_avg"], rc["v_avg"])
    e, U, _ = K.full_spectrum_from_sector(h_avg)
    Phi = U[:, e < 0]
    out["pure_vacuum"] = {"n_neg": int((e < 0).sum()), "n_zero_1e-12": int((np.abs(e) < 1e-12).sum()), "min_abs_e": float(np.min(np.abs(e))),
                          "N_avg_minus_N_app_max": float(np.max(np.abs(rc["N_avg"] - rc["N_app"]))), "v_avg_minus_v_app_max": float(np.max(np.abs(rc["v_avg"] - rc["v_app"])))}
    TH = T_H_rec(H)
    Nf = lambda R: np.interp(R, ops.R, rc["N_avg"]); vf = lambda R: np.interp(R, ops.R, rc["v_avg"])
    gauss = {"FM": rc["FM"], "eps0": H["eps_tot"], "J0": H["J_tot"], "g": H["g"], "R_sites": H["R_sites"], "ell": H["ell"]}
    R_col = 16.0 * H["rs_nom"]
    for om_TH in (6.0, -6.0):
        om = om_TH * TH
        psi, inf = L1.wkb_packet(Nf, vf, 1.0, om, abs(om) / 3.0, 2600.0, ops.n)
        r = CB.calibrate(ops, rc["N_avg"], rc["v_avg"], Phi, psi, "particle" if om > 0 else "hole", om, R_enc=float(ops.R[-1]), R_col=R_col,
                         gauss=gauss, weight=16.0, s_list=(1.0, 16.0))
        r["packet_info"] = {k: inf[k] for k in ("lo", "hi", "edge_weight", "dropped_weight")}
        out[("particle" if om > 0 else "hole")] = r
        out["checks"].append(chk("T6 the mode is available (kept norm > 0.99) (%s)" % r["kind"], r["kept_norm"] > 0.99, kept=r["kept_norm"]))
        out["checks"].append(chk("T6 packet energy <psi'|h|psi'>/|omega| = 1 within 2e-3 (%s)" % r["kind"], abs(r["target_over_omega"] - 1) < 2e-3,
                                 value=r["target_over_omega"]))
        # check v1.1 (added after the as-written run): the WKB packet's known energy is its own spectral mean sum rho omega, not the nominal omega
        mean_om = float(np.sum(inf["rho"] * inf["oms"]))
        out["checks"].append(chk("T6 [check v1.1] packet energy = its spectral mean sum rho omega within 1e-3 (%s)" % r["kind"],
                                 abs(r["target_one_copy"] * (1 if om > 0 else -1) / mean_om - 1) < 1e-3,
                                 value=r["target_one_copy"] * (1 if om > 0 else -1) / mean_om, mean_over_omega=mean_om / om))
        out["checks"].append(chk("T6 direct read on the enclosing surface = <psi'|h|psi'> (%s)" % r["kind"], r["direct_pass"],
                                 Tr_dC_h_Renc=r["Tr_dC_h_Renc"], target=r["target_one_copy"], Tr_dC_h_Rcol=r["Tr_dC_h_Rcol"]))
        for g_ in r["gauss"]:
            out["checks"].append(chk("T6 Gauss rows: dE = <psi'|h|psi'> within max(1e-3 omega, row residual), %s, %g copies" % (r["kind"], g_["scale_copies"]),
                                     g_["pass"], dE=g_["dE_one_copy"], target=r["target_one_copy"], tol=g_["tol"]))
            out["checks"].append(chk("T6 collar charge unchanged (Gauss rows), %s, %g copies" % (r["kind"], g_["scale_copies"]), g_["collar_unchanged"],
                                     dM_collar=g_["dM_collar_one_copy"], tol=g_["tol"]))
    return out



def leaking_model(L=120, eps0=0.6, s=0.004, V=0.25, J=0.5):
    """orbital 0 (energy eps0 - s t) coupled by V to a tight-binding lead 1..L (hopping J, band [-2J, 2J])."""
    n = L + 1
    A0 = np.zeros((n, n))
    for i in range(1, L):
        A0[i, i + 1] = A0[i + 1, i] = -J
    A0[0, 1] = A0[1, 0] = V
    A1 = np.zeros((n, n)); A1[0, 0] = 1.0
    Hf = lambda t: A0 + (eps0 - s * t) * A1
    dHf = lambda t: -s * A1
    return Hf, dHf, A0, A1, n


def T7():
    out = {"checks": []}
    res = {}
    for label, s in (("swept", 0.004), ("held", 0.0)):
        Hf, dHf, A0, A1, n = leaking_model(s=s)
        Rn = 6
        Pi = np.zeros(n); Pi[: Rn + 1] = 1.0
        hR = lambda H: 0.5 * (np.diag(Pi) @ H + H @ np.diag(Pi))
        e0, U0 = np.linalg.eigh(Hf(0.0))
        w0 = np.abs(U0[0, :]) ** 2
        exc = int(np.argmax(np.where(e0 > 0, w0, -1)))
        occ0 = (e0 < 0).astype(float); occ0[exc] = 1.0
        C0 = (U0 * occ0) @ U0.conj().T
        T = 150.0
        dt_f = 0.01
        for cad in (1.0, 0.5):
            ts = np.arange(0.0, T + 1e-9, cad)
            qmask = np.zeros(n, bool); qmask[: Rn + 1] = True
            fam = RF.LinearFamily([A0, A1])
            arms = {"R-an": RF.RAn("rank"), "R-loc": RF.RLoc(qmask, delta_c=0.2, R_Q=Rn), "R-code": RF.RCode(fam, tau=cad, c_LZ=1.0),
                    "R-star": RF.RStar(C0 * 0 + (U0[:, e0 < 0] @ U0[:, e0 < 0].conj().T))}
            rq = RF.RQ(np.r_[np.ones(4, bool), np.zeros(n - 4, bool)], R_Q=3)
            led = LG.Ledger(list(arms) + ["R-Q"])
            C = C0.copy()
            Ucum = np.eye(n, dtype=complex)
            tprev = 0.0
            rq_tr = []
            for k, t in enumerate(ts):
                if k > 0:
                    nsub = int(round((t - tprev) / dt_f))
                    for j in range(nsub):
                        tm = tprev + (j + 0.5) * dt_f
                        e_, U_ = np.linalg.eigh(Hf(tm))
                        Ucum = (U_ * np.exp(-1j * e_ * dt_f)) @ U_.conj().T @ Ucum
                    tprev = t
                C = Ucum @ C0 @ Ucum.conj().T
                H = Hf(t)
                spc = RF.Spec(*np.linalg.eigh(H), t=t, f=(np.array([1.0, 0.6 - s * t]), np.zeros(2)), h=sp.csr_matrix(H))
                if k == 0:
                    for a in arms.values():
                        a.start(spc, occ0 if not isinstance(a, RF.RStar) else None)
                    rq.start(H, C0)
                else:
                    Pq_old = rq.P()
                    for a in arms.values():
                        a.step(spc)
                    rq.step(H)
                    Hm = 0.5 * (H + Hf(ts[k - 1]))
                    rq_tr.append(float(np.real(np.trace(Hm @ (rq.P() - Pq_old)))))
                ops_t = {"h": sp.csr_matrix(H), "h_R": sp.csr_matrix(hR(H)), "hdot": sp.csr_matrix(dHf(t)), "hdot_R": sp.csr_matrix(hR(dHf(t)))}
                Ps = {a: arms[a].P() for a in arms}
                Ps["R-Q"] = rq.P()
                led.sample(t, ops_t, C, Ps)
            # E_emit: the physical energy current out of the region, reference-free (P_-(h) commutes: its current vanishes)
            J_out = [-v_ for v_ in led.S["R-an"]["Fin_D"]]
            Iout = led._trap(ts, np.array(J_out) + np.array(led.S["R-an"]["Int"]) * 0)
            E_emit = abs(Iout) if s > 0 else 1.0
            cl = led.close(E_scale=E_emit)
            res[(label, cad)] = {"E_emit": E_emit, "ledger": cl, "RQ_sum_tr_h_dP": float(np.sum(rq_tr)), "RQ_max_abs_step": float(np.max(np.abs(rq_tr))),
                                 "RQ_cut_bond_energy_t0": RF.RQ.cut_bond_energy(Hf(0.0), rq.qmask), "Rloc_jumps": len(arms["R-loc"].jumps),
                                 "Rloc_jump_sum": arms["R-loc"].jump_sum()}
    sw, hd = res[("swept", 1.0)], res[("held", 1.0)]
    sw2 = res[("swept", 0.5)]
    for a, r in sw["ledger"].items():
        out["checks"].append(chk("T7 ledger closes, %s (residual/E_emit < 1e-2, A4)" % a, r["residual_over_E"] < 1e-2, residual=r["residual_retained"],
                                 E_emit=sw["E_emit"], over_E=r["residual_over_E"], total_residual=r["residual_total"]))
    floor = max(abs(r["residual_retained"]) for r in hd["ledger"].values())
    out["checks"].append(chk("T7 held control floor < 1e-10 (all arms)", floor < 1e-10, floor=floor))
    ratio = abs(sw["ledger"]["R-an"]["residual_retained"]) / max(abs(sw2["ledger"]["R-an"]["residual_retained"]), 1e-300)
    out["checks"].append(chk("T7 cadence convergence (trapezoid, ratio ~ 4 on halving)", 2.5 < ratio < 6.0, ratio=ratio))
    for a in ("R-an", "R-loc", "R-code"):
        out["checks"].append(chk("T7 interface term zero for the commuting arm %s" % a, sw["ledger"][a]["max_abs_Int_rate"] < 1e-10,
                                 value=sw["ledger"][a]["max_abs_Int_rate"]))
    rs_ = sw["ledger"]["R-star"]
    out["checks"].append(chk("T7 R-star: no reference response; driven identity d/dt Tr[(C - P*) h] = Tr[(C - P*) dh/dt]",
                             abs(rs_["reference_response_in"] + rs_["reference_response_out"]) < 1e-12 and abs(rs_["residual_total"]) < 1e-2 * sw["E_emit"],
                             response=rs_["reference_response_in"] + rs_["reference_response_out"], residual_total=rs_["residual_total"],
                             Int=rs_["interface_Int"]))
    out["checks"].append(chk("T7 R-Q block identity: sum Tr(h dP_D) -> 0", abs(sw["RQ_sum_tr_h_dP"]) < 1e-2 * sw["E_emit"], value=sw["RQ_sum_tr_h_dP"],
                             cut_bond_energy_t0=sw["RQ_cut_bond_energy_t0"]))
    cbe = sw["RQ_cut_bond_energy_t0"]
    out["checks"].append(chk("T7 R-Q cut-bond energy: projector route = eigenvalue sums, negative", abs(cbe[0] - cbe[1]) < 1e-12 and cbe[0] < 0, value=cbe))
    out["summary"] = {"%s_cad%g" % k: {"E_emit": v["E_emit"], "Rloc_jumps": v["Rloc_jumps"], "Rloc_jump_sum": v["Rloc_jump_sum"],
                                       "residuals": {a: r["residual_retained"] for a, r in v["ledger"].items()},
                                       "terms_R-an": {x: v["ledger"]["R-an"][x] for x in ("dM_retained", "outflow_Delta_physical_current", "imposed_work_in",
                                                                                            "reference_response_in", "interface_Int", "dX_exterior_storage")}}
                      for k, v in res.items()}
    # the production KPM sea trace + window correction against dense, on a Dirac chain
    ops = K.ChainOps(120, 1.0)
    N, v = chain_fields(ops.R, 1.0, rs=12.0)
    h = ops.h(N, v)
    e, U, sec = K.full_spectrum_from_sector(h)
    W = 0.08; ws = W / 10.0
    win = np.abs(e) < W
    occ = (e < 0).astype(float)
    j0 = np.where(win & (e > 0))[0][0]; occ[j0] = 1.0             # one window level occupied against the Killing state
    Pd = (U * occ) @ U.conj().T
    hR = ops.h_R(N, v, 30.0)
    sea = LG.kpm_local_sea_trace(h, ws)
    wp = LG.WindowPKPM(sea, U[:, win], occ[win], e[win], ws)
    t_k = LG.tr(wp, hR); t_d = LG.tr(Pd, hR)
    out["checks"].append(chk("T7 production KPM sea trace + window correction = dense", abs(t_k - t_d) < 1e-9, kpm=t_k, dense=t_d,
                             outside_window_min_abs_e=float(np.min(np.abs(e[~win])))))
    # the absorber: Strang split against the dense non-Hermitian exponential; the eigenstate arms refuse h - iW
    Wd = np.zeros(h.shape[0]); Wd[-40:] = 0.05 * np.linspace(0, 1, 40) ** 2
    Hd = h.toarray()
    dt = 0.5
    Ue = expm(-1j * (Hd - 1j * np.diag(Wd)) * dt)
    eh, Uh = np.linalg.eigh(Hd)
    Us = np.diag(np.exp(-Wd * dt / 2)) @ ((Uh * np.exp(-1j * eh * dt)) @ Uh.conj().T) @ np.diag(np.exp(-Wd * dt / 2))
    err_split = float(np.max(np.abs(Ue - Us)))
    try:
        RF.require_hermitian(sp.csr_matrix(Hd - 1j * np.diag(Wd)))
        refused = False
    except ValueError:
        refused = True
    out["checks"].append(chk("T7 absorber Strang step O(dt^3) error, eigenstate arms refuse h - iW", err_split < 1e-3 and refused, split_err=err_split, refused=refused))
    return out


# ---------------------------------------------------------------------------------------------------------------- T8
def T8():
    out = {"checks": []}
    Hf, dHf, A0, A1, n = leaking_model(L=300, eps0=0.3, s=0.0, V=0.3)
    H = Hf(0.0)
    e, U = np.linalg.eigh(H)
    psi = U @ (np.exp(-1j * e * 200.0) * (U.conj().T @ np.eye(n)[0]))
    C = np.outer(psi, psi.conj())
    # bins in arrival order: the farthest lead sites arrived first
    edges = list(range(n - 1, 0, -25))
    bins = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        B = np.zeros((n, a_ - b_)); B[np.arange(b_ + 1, a_ + 1), np.arange(a_ - b_)] = 1.0
        bins.append(B)
    Ws, rep = OUT.nested_modes(bins)
    rows, _ = OUT.nested_entropies(C, Ws, eps=0.0)
    q = np.cumsum([float(np.sum(np.abs(W.conj().T @ psi) ** 2)) for W in Ws])
    exact = K.s_bin(q)
    dev = float(np.max(np.abs(np.array([r["S_R"] for r in rows]) - exact)))
    out["checks"].append(chk("T8 nested S(R_n) = s(q_n) on the pure single-particle drain", dev < 1e-10, max_dev=dev, q_final=float(q[-1])))
    negs = [r["n"] for r in rows if r["S_b_given_R"] < -1e-6]
    out["checks"].append(chk("T8 S(b|R) < 0 for the later bins once q > 1/2", len(negs) > 0 and all(q[k - 1] > 0.5 for k in negs), bins_negative=negs))
    # overlapping Gaussian bins: earlier modes unchanged by later bins
    rng = np.random.default_rng(2)
    xs = np.arange(n)
    gb = [np.stack([np.exp(-((xs - c) / 9.0) ** 2) * np.exp(1j * k * xs) for k in (0.5, 1.0)], axis=1) for c in range(280, 60, -20)]
    W_all, rep_all = OUT.nested_modes(gb)
    W_less, _ = OUT.nested_modes(gb[:-1])
    same = max(float(np.max(np.abs(np.abs(W_all[i].conj().T @ W_less[i]) - np.eye(W_less[i].shape[1])))) for i in range(len(W_less)))
    out["checks"].append(chk("T8 nested modes: earlier bins unchanged when a later bin is added; orthonormal", same < 1e-10 and rep_all["orthonormality_defect"] < 1e-12,
                             unchanged_dev=same, orthonormality=rep_all["orthonormality_defect"], gram_offdiag_max=float(np.max(np.abs(rep_all["gram_raw"] - np.diag(np.diag(rep_all["gram_raw"])))))))
    # certified intervals contain the truth under a Hermitian perturbation of norm eps
    eps = 1e-6
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); X = 0.5 * (X + X.conj().T)
    X *= eps / np.linalg.norm(X, 2)
    rows_p, _ = OUT.nested_entropies(C + X, Ws, eps=eps)
    contain = all(rp["S_R_lo"] - 1e-15 <= r["S_R"] <= rp["S_R_hi"] + 1e-15 and rp["S_b_given_R_lo"] - 1e-15 <= r["S_b_given_R"] <= rp["S_b_given_R_hi"] + 1e-15
                  for r, rp in zip(rows, rows_p))
    out["checks"].append(chk("T8 certified intervals contain the exact entropies (perturbation 1e-6, no clipping)", contain,
                             widest=max(rp["S_R_hi"] - rp["S_R_lo"] for rp in rows_p)))
    # the pure-state floor on a Dirac chain (both bands): the complete mode set has S = 0; S(A) = S(complement)
    ops = K.ChainOps(150, 1.0)
    h = ops.h(*chain_fields(ops.R, 1.0, rs=15.0))
    pr = PR.pure_sector(h)
    Cp = pr["V"] @ pr["V"].conj().T
    fl = OUT.floor_pure_state(Cp, Cp.shape[0])
    A = ops.site_mask(40.0)
    SA = K.cov_entropy(Cp[np.ix_(A, A)])["S"]; SB = K.cov_entropy(Cp[np.ix_(~A, ~A)])["S"]
    out["checks"].append(chk("T8 pure-state floor (complete mode set) and S(A) = S(A^c)", fl < 1e-8 and abs(SA - SB) < 1e-8, floor=fl, S_A=SA, S_Ac=SB))
    rd = RD.purification(rows, floor=fl)
    out["checks"].append(chk("T8 purification reader on the drain: PASS", rd["verdict"] == "PASS", reader=rd))
    return out


# ---------------------------------------------------------------------------------------------------------------- T9
def T9():
    out = {"checks": []}
    rc = record()
    H = rc["H"]
    rs = H["rs_nom"]; ne = len(H["R_sites"])
    n = ne + int(1e4 * rs)
    R = np.arange(1, n + 1, dtype=float)
    N = np.ones(n); v = -np.sqrt(2.0 * H["g"] * H["E_rec"] / R)
    N[:ne] = rc["N_app"]; v[:ne] = rc["v_app"]
    flat = OUT.group_velocity_max(np.ones(500), np.zeros(500), 1.0, 1e8 + np.arange(500, dtype=float))     # kappa/R -> 0: |cos k|, max 1
    out["checks"].append(chk("T9 flat chain (kappa/R = 1e-8): max |d omega/dk| = 1", abs(flat["v_max"] - 1.0) < 1e-5, v_max=flat["v_max"]))
    T_needed = (3.0 / (3.6e-4) + 2100.0) * rs                       # >= 3/Gamma_res of readout (F9) plus a 2000 r_s sweep and the formation
    cert = OUT.certify_exterior(N, v, 1.0, R, R_read=16.0 * rs, T_needed=T_needed, rs=rs)
    out["certification"] = cert
    out["checks"].append(chk("T9 exterior length >= 1e4 r_s and first return (bound N + |v|) after the readout", cert["certified"] and cert["length_ge_1e4_rs"], **cert))
    t0 = time.time()
    u = OUT.unitarity_check(n, 1.0, N, v, T=2000.0, n_probe=4)
    u["seconds"] = time.time() - t0
    out["unitarity"] = u
    out["checks"].append(chk("T9 exact unitarity on the 1e4 r_s parent (forward-backward < 1e-10, norm drift < 1e-12, h Hermitian)",
                             u["forward_backward_max_err"] < 1e-10 and u["norm_drift_max"] < 1e-12 and u["hermiticity_defect"] == 0.0, **u))
    return out


# ---------------------------------------------------------------------------------------------------------------- T10
def T10(skip_record=False):
    out = {"checks": []}
    ops = K.ChainOps(2000, 1.0)
    h = ops.h(np.ones(ops.n), np.zeros(ops.n))
    e, _, _ = K.full_spectrum_from_sector(h)
    rows = []
    for E in (0.05, 0.1, 0.2):
        cnt = int(((e > 0) & (e < E)).sum())
        w = LV.wkb_level_count(E, ops.R, 1.0)
        rows.append({"E": E, "count": cnt, "wkb": w})
    out["static_counts"] = rows
    out["checks"].append(chk("T10 static exterior level count = standing-wave rule (within 2 levels)", all(abs(r["count"] - r["wkb"]) <= 2 for r in rows), rows=rows))
    sel = np.sort(e[(e > 0.1) & (e < 0.2)])
    sp_meas = float(np.mean(np.diff(sel)))
    sp_wkb = 0.1 / (LV.wkb_level_count(0.2, ops.R, 1.0) - LV.wkb_level_count(0.1, ops.R, 1.0))
    out["checks"].append(chk("T10 static exterior level spacing (0.1 < e < 0.2) = standing-wave rule within 2%", abs(sp_meas / sp_wkb - 1) < 0.02,
                             measured=sp_meas, wkb=sp_wkb))
    # resonance width: one level on a dense ladder (Gamma > Delta) against the golden rule
    M, Dl, V = 200, 0.05, 0.08
    n = 2 * M + 2
    Hm = np.zeros((n, n)); Hm[0, 0] = 0.013
    Hm[np.arange(1, n), np.arange(1, n)] = Dl * np.arange(-M, M + 1)
    Hm[0, 1:] = V; Hm[1:, 0] = V
    e2, U2 = np.linalg.eigh(Hm)
    qm = np.zeros(n, bool); qm[0] = True
    rr = LV.resonance_at(e2, U2, qm, -1.5, 1.5)
    Gam = 2 * math.pi * V * V / Dl
    f = rr["fit"]
    out["checks"].append(chk("T10 resonance width (Lorentzian fit) = 2 pi V^2/Delta within 5%, centre within Delta/2", abs(f["Gamma"] / Gam - 1) < 0.05 and
                             abs(f["e_c"] - 0.013) < Dl / 2, fit=f, golden_rule=Gam))
    out["checks"].append(chk("T10 V from the fit = coupling within 5%", abs(f["V"] / V - 1) < 0.05, V_fit=f["V"], V=V))
    # endpoints on a synthetic trajectory
    t = np.linspace(0, 10, 1001); TH = 1.0; r_ = 0.5
    ec = 3.95 * TH - r_ * t
    ep = LV.endpoints(t, ec, TH, +1, ladder=[[3.5, 2.0, 0.5]] * len(t))
    exp_ = {"E1": 2.95 / r_, "E2": sorted([(3.95 - x) / r_ for x in (3.5, 2.0, 0.5)]), "E3": 3.95 / r_}
    ok = abs(ep["E1_entry_thermal_band"] - exp_["E1"]) < 0.011 and abs(ep["E3_zero_crossing"] - exp_["E3"]) < 1e-9 and \
        max(abs(a - b) for a, b in zip(sorted(ep["E2_crossings"]), exp_["E2"])) < 1e-9
    out["checks"].append(chk("T10 endpoints E1, E2, E3 on a synthetic slide", ok, got=ep, expected=exp_))
    cp = LV.crossing_params(t, ec, np.full(len(t), 2.0), V=0.01, i_cross=int(np.argmin(np.abs(ec - 2.0))))
    out["checks"].append(chk("T10 crossing parameters: s_j and lambda_j = 2 pi V^2/s", abs(cp["s_j"] - r_) < 1e-9 and abs(cp["lambda_j"] - 2 * math.pi * 1e-4 / r_) < 1e-12, **cp))
    rg = LV.regime_checks(Dl, Gam, T_obs=1.0, s_j=r_)
    out["regime_example"] = rg
    if not skip_record:
        rc = record()
        Hh = rc["H"]
        ops4 = K.ChainOps(len(Hh["R_sites"]), 1.0)
        h4 = ops4.h(rc["N_app"], rc["v_app"])
        e4, U4, s4 = K.full_spectrum_from_sector(h4)
        _REC["spec100"] = (e4, U4, s4)
        TH4 = T_H_rec(Hh)
        root = max(Hh["rec"]["trapping_zeros"]) if Hh["rec"].get("trapping_zeros") else 36.31
        qm4 = ops4.site_mask(root)
        up = LV.resonance_at(e4, U4, qm4, 3.0 * TH4, 5.0 * TH4, outmask=ops4.site_mask(root, inside=False))
        dn = LV.resonance_at(e4, U4, qm4, -5.0 * TH4, -3.0 * TH4, outmask=ops4.site_mask(root, inside=False))
        out["record_pair"] = {"T_H": TH4, "root": root, "rising": up, "pocket": dn}
        okp = up is not None and dn is not None and abs(up["e_top"] / TH4 - 3.95) < 0.15 and abs(dn["e_top"] / TH4 + 3.95) < 0.15
        out["checks"].append(chk("T10 tracker finds the k = 1 inverted pair at +-3.95 T_H on the stored record (v1's pilot value)", okp,
                                 rising_over_TH=None if up is None else up["e_top"] / TH4, pocket_over_TH=None if dn is None else dn["e_top"] / TH4,
                                 w_top=[None if up is None else up["w_top"], None if dn is None else dn["w_top"]]))
    return out


# ---------------------------------------------------------------------------------------------------------------- T11
def T11():
    out = {"checks": []}
    rc = record()
    H = rc["H"]; Z = rc["Z"]
    # (a) the engine's own sea channel (kappa 1, m = 0.2, engine_collapse_v19 --m default; no staggered symmetry with a mass): the stored
    #     record's reference R2_1 and the R-an start from it
    import page_ops_v6 as OPS
    R = np.arange(1, len(H["R_sites"]) + 1, dtype=float)
    h0m, Qm = OPS.chain_sparse(R, 0.2, 1.0)
    hm = OPS.clocked_csr(h0m, Qm, rc["N_app"], rc["v_app"])
    e, U = np.linalg.eigh(hm.toarray())
    e_st = np.asarray(Z["e_1"], float)
    out["checks"].append(chk("T11 h(T_hold) (the engine's m = 0.2 sea channel) from the slice's applied fields reproduces the stored spectrum",
                             float(np.max(np.abs(e - e_st))) < 1e-10, max_diff=float(np.max(np.abs(e - e_st)))))
    R2 = np.asarray(Z["R2_1"])
    occ = np.real((np.abs(U.conj().T @ R2) ** 2).sum(axis=1))
    frac = float(np.max(np.minimum(occ, 1 - occ)))
    mask = np.unpackbits(np.asarray(Z["mask_1"]))[: len(e)].astype(bool)
    occ_r = occ > 0.5
    out["checks"].append(chk("T11 the stored reference's occupations in the eigenbasis of h(T_hold) are 0/1 and equal the engine's mask", frac < 1e-8 and
                             bool(np.all(occ_r == mask)), max_fractional=frac, mask_mismatch=int(np.sum(occ_r != mask)),
                             occupied_positive=int(np.sum(occ_r & (e > 0))), empty_negative=int(np.sum(~occ_r & (e < 0)))))
    del R2
    sp0 = RF.Spec(e, U, None, t=0.0, h=hm)
    ran = RF.RAn("rank", source="the stored reference R2_1 at T/r_s 100 in the eigenbasis of h(T_hold)")
    ran.start(sp0, occ_r.astype(float))
    for k in range(1, 4):
        ran.step(RF.Spec(e, U, None, t=float(k * 8.0), h=hm))
    trh = [x.get("tr_h_dP") for x in ran.log[1:]]
    out["checks"].append(chk("T11 R-an on held fields (engine sea): occupations unchanged, Tr(h dP) = 0", bool(np.all(ran.occ == occ_r)) and max(abs(x) for x in trh) < 1e-12,
                             tr_h_dP=trh, min_gap_occ_empty=ran.log[-1]["min_gap_occ_empty"]))
    del U
    # (b) v1's held control: the massless k = 1 probe on the same applied fields; its inverted levels by the band rule of pt_testA_levels_v1
    pv = probe_vacuum()
    ops, hp, ep, Up = pv["ops"], pv["h"], pv["e"], pv["U"]
    hN = ops.h(rc["N_app"], np.zeros(ops.n))
    band = np.sign(np.real(np.einsum("ij,ij->j", Up.conj(), hN @ Up)))
    rising = (band < 0) & (ep > 0); pocket = (band > 0) & (ep < 0)
    TH = T_H_rec(H)
    out["probe_inverted"] = {"rising_over_TH": [float(x) for x in ep[rising] / TH], "pocket_over_TH": [float(x) for x in ep[pocket] / TH]}
    out["checks"].append(chk("T11 the probe's four inverted levels (2 rising, 2 pocket) at +-3.95 T_H (v1's census)", rising.sum() == 2 and pocket.sum() == 2 and
                             all(abs(abs(x) - 3.95) < 0.1 for x in out["probe_inverted"]["rising_over_TH"] + out["probe_inverted"]["pocket_over_TH"]), **out["probe_inverted"]))
    # check v1.1 (added after the as-written run): the band rule over the whole 4096-site domain also marks a non-interior level; v1's
    # census was on the interior block, so the inverted set is the band rule AND interior localization (w_in(R <= 64) > 1/2)
    win = np.real((np.abs(Up[ops.site_mask(64.0), :]) ** 2).sum(axis=0)) > 0.5
    ri, pi_ = rising & win, pocket & win
    out["probe_inverted_interior"] = {"rising_over_TH": [float(x) for x in ep[ri] / TH], "pocket_over_TH": [float(x) for x in ep[pi_] / TH]}
    out["checks"].append(chk("T11 [check v1.1] the probe's interior inverted levels: 2 rising, 2 pocket at +-3.95 T_H", ri.sum() == 2 and pi_.sum() == 2 and
                             all(abs(abs(x) - 3.95) < 0.1 for x in out["probe_inverted_interior"]["rising_over_TH"] + out["probe_inverted_interior"]["pocket_over_TH"]),
                             **out["probe_inverted_interior"]))
    rising, pocket = ri, pi_
    # R-an's start for the probe (the record's de facto P: rising filled, pocket empty); held: unchanged
    occp = (ep < 0).astype(float); occp[rising] = 1.0; occp[pocket] = 0.0
    rp = RF.RAn("rank", source="the record's de facto P (rising filled, pocket empty); v1 stored no probe C(T_hold)")
    rp.start(RF.Spec(ep, Up, pv["sec"], t=0.0, h=hp), occp)
    for k in range(1, 4):
        rp.step(RF.Spec(ep, Up, pv["sec"], t=float(k * 8.0), h=hp))
    out["checks"].append(chk("T11 R-an on held fields (probe): occupations unchanged, Tr(h dP) = 0", bool(np.all(rp.occ == occp)) and
                             max(abs(x["tr_h_dP"]) for x in rp.log[1:]) < 1e-12))
    # <chi|P_an - P_-|chi> lives on the inverted levels only, for ANY start occupation: bounded by sum_inv |<alpha|chi>|^2
    import pt_lib_v1 as L1
    Nf = lambda R_: np.interp(R_, ops.R, rc["N_app"]); vf = lambda R_: np.interp(R_, ops.R, rc["v_app"])
    inv = rising | pocket
    rows = []
    for om_TH in (3.0, 6.0, -3.0, -6.0):
        om = om_TH * TH
        chi, _ = L1.wkb_packet(Nf, vf, 1.0, om, abs(om) / 3.0, 2048.0, ops.n)
        bnd = float(np.sum(np.abs(Up[:, inv].conj().T @ chi) ** 2))
        Gf = float(L1.hawking_n({"oms": np.array([abs(om)]), "rho": np.array([1.0])}, 1.0, H["rs_nom"], TH))
        rows.append({"omega_over_TH": om_TH, "bound_chi_Pan_minus_Pminus_chi": bnd, "chi_Pminus_chi": float(np.sum(np.abs(Up[:, ep < 0].conj().T @ chi) ** 2)),
                     "Gamma_f_H": Gf})
    out["v1_packets"] = rows
    v1_min = {3.0: 0.39, 6.0: 0.24}                                # v1 sec. A.2: the smallest held-control f_out/Gamma f_H at 3 and 6 T_H
    ok = all(r["bound_chi_Pan_minus_Pminus_chi"] < 1e-3 * v1_min[abs(r["omega_over_TH"])] * r["Gamma_f_H"] for r in rows)
    out["checks"].append(chk("T11 [check v1.1, interior inverted set] R-an's read of v1's exterior packets = the Killing read to below v1's table precision (any start)", ok, rows=rows))
    return out



# ---------------------------------------------------------------------------------------------------------------- T12
def T12():
    out = {"checks": []}
    exp = []
    exp.append((RD.reference_charges_emission(1.0, -1.0, 0.001, 1.0)["verdict"], "PASS"))
    exp.append((RD.reference_charges_emission(1.0, -0.5, 0.001, 1.0)["verdict"], "FAIL"))
    exp.append((RD.reference_charges_emission(1.0, -1.0, 0.05, 1.0)["verdict"], "INCONCLUSIVE"))
    exp.append((RD.ran_artefact_F4(0.03, 0.6, 1.0, 0.4, 0.38, 0.38, 1.0, 1.0, 30.0, 0.0)["verdict"], "SHOWN"))
    exp.append((RD.ran_artefact_F4(0.03, 1.0, 1.0, 0.4, 0.0, 0.38, 1.0, 1.0, None, 0.0)["verdict"], "REFUTED"))
    exp.append((RD.ran_artefact_F4(3.0, 1.0, 1.0, 0.4, 0.0, 0.38, 1.0, 1.0, None, 0.0)["verdict"], "INCONCLUSIVE"))
    exp.append((RD.ran_artefact_A9(True, True, True)["verdict"], "SHOWN"))
    exp.append((RD.ran_artefact_A9(True, True, True, small_C_minus_P_only=True)["verdict"], "INCONCLUSIVE"))
    b = 2 * 1.0 * math.sqrt(0.1 / (2 * math.pi * 1.0))
    exp.append((RD.rloc_declaration([[1, 0], [1, 0]], b, 1.0, 0.1, 1.0)["verdict"], "PASS"))
    exp.append((RD.rloc_declaration([[1, 0], [0, 1]], b, 1.0, 0.1, 1.0)["verdict"], "FAIL"))
    exp.append((RD.slide_fed_transfer(True, True, True, 5.0, 0.8)["verdict"], "PASS"))
    exp.append((RD.slide_fed_transfer(True, False, True, 0.0, 0.0)["verdict"], "INCONCLUSIVE"))
    exp.append((RD.kinematics(-1e-8, 1e-8, 0.03)["verdict"], "PASS"))
    exp.append((RD.kinematics(-1e-8, 1e-8, 1e-4)["verdict"], "FAIL"))
    exp.append((RD.qft_agreement(0.01, 0.05, True, True)["verdict"], "PASS"))
    exp.append((RD.qft_agreement(0.01, 0.05, False, True)["verdict"], "INCONCLUSIVE"))
    late = RD.qft_agreement(0.5, 0.05, True, True, late_interval=True, c6="both")
    exp.append((tuple(x["verdict"] for x in late), ("UNRESOLVED", "DISPERSION EFFECT")))
    rows = [{"n": 1, "S_R": 0.5, "S_b_given_R_hi": 0.6, "valid": True}, {"n": 2, "S_R": 0.6, "S_b_given_R_hi": 0.2, "valid": True},
            {"n": 3, "S_R": 0.1, "S_b_given_R_hi": -0.4, "valid": True}]
    exp.append((RD.purification(rows, 1e-6)["verdict"], "PASS"))
    rows_bad = [dict(r, valid=False) for r in rows]
    exp.append((RD.purification(rows_bad, 1e-6)["verdict"], "INCONCLUSIVE"))
    exp.append((RD.complete_drain(4 * math.log(2) * 1.0003, 4, 1e-6, 1e-6, 1e-6)["verdict"], "PASS"))
    exp.append((RD.complete_drain(3.0, 4, 1e-6, 1e-6, 1e-6)["verdict"], "FAIL"))
    exp.append((RD.no_turnover([0, 2000])["verdict"], "BOUND ONLY"))
    exp.append((RD.charge_calibration(1.0005, 1.0, 1.0, 0.0)["verdict"], "PASS"))
    bad = [(i, g, w) for i, (g, w) in enumerate(exp) if g != w]
    out["checks"].append(chk("T12 every sec. 10 reader returns its verdict on synthetic inputs", not bad, n=len(exp), mismatches=bad))
    combos = 0
    import itertools
    for vals in itertools.product(*[OPT.CHOICES[k] for k in sorted(OPT.CHOICES)]):
        o = dict(zip(sorted(OPT.CHOICES), vals))
        a = OPT.arms_for(o)
        assert a["reference_arms"] and a["parents"] and a["regime_grid"] and a["testB_late_verdicts"]
        combos += 1
    out["checks"].append(chk("T12 every C1 to C6 side is constructible (all option combinations)", combos == int(np.prod([len(v) for v in OPT.CHOICES.values()])),
                             combinations=combos, defaults=OPT.DEFAULTS, default_arms=OPT.arms_for(OPT.DEFAULTS)))
    return out


TESTS = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6, "T7": T7, "T8": T8, "T9": T9, "T10": T10, "T11": T11, "T12": T12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-record", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = [x for x in a.only.split(",") if x] or list(TESTS)
    if a.skip_record:
        names = [x for x in names if x not in ("T6", "T9", "T11")]
    res = {"script": "test_pt3_stage1.py", "engine_md5": K.engine_md5s(), "own_md5": K.own_md5s(), "tests": {}}
    for nm in names:
        t0 = time.time()
        K.log("start %s" % nm)
        try:
            r = TESTS[nm](a.skip_record) if nm == "T10" else TESTS[nm]()
        except Exception as ex:                                        # a crash is a FAIL as written
            r = {"checks": [chk("%s crashed" % nm, False, error=repr(ex), tb=traceback.format_exc())]}
        r["seconds"] = time.time() - t0
        res["tests"][nm] = r
        for c in r["checks"]:
            K.log("  %-5s %s" % (c["verdict"], c["check"]))
        K.dump_json(res, os.path.join(a.out, "stage1_results%s.json" % ("_" + "_".join(names) if a.only else "")))
    tot = [c for r in res["tests"].values() for c in r["checks"]]
    K.log("%d checks: %d PASS, %d FAIL" % (len(tot), sum(c["verdict"] == "PASS" for c in tot), sum(c["verdict"] == "FAIL" for c in tot)))


if __name__ == "__main__":
    main()
