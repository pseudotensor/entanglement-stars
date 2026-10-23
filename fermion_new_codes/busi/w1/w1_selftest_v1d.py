#!/usr/bin/env python
"""w1_selftest_v1d.py (reviewer A, the busi lane, W1; v1d of w1_selftest_v1c.py, W1_fix_v1d.md): the gates before any entropy number of page_history_v6o
prints. Stages (--stages, default all):
  pairing : a 128-site chain (kappa 2, random smooth N, v): the operator audit (max|h + conj h|, max|[S_stag, h]| = 0), the dense spectrum's pairing
            Tr min(h, 0) = -Tr|h|/2, spec(h_-) = -spec(h_+); the negative control m = 0.2 with the flow (the audit fails, the residual printed).
  sector  : the sector vacuum against the dense one on the fixture's record with a trapping zero (the projector difference, E_ref, the exterior
            entropies of a state and of the vacuum, t_X, ||C - P_-||_1 by the exact routes on both sides; the Gram route's defect against its
            indicative bound beside).
  oracle  : the dense oracle fixture with a trapping zero (the marginally bound dust flow, 192 sites, r_s 16, tau 2, 61 records; the zero at R = r_s
            from t about 75): the full 384 x 384 covariance propagated by the exact dense exponential of every sub-step against page_history_v6o's
            column recursion in 3 representations (the + sector propagation; the full propagation with the halved entropies; v6n's full route) for
            the production (--interp linear) and the control (--interp none): S_X raw, the held vacuum's S_X, the excess, E_inv, E_kappa =
            Tr[|h|(C - P)^2] (the identity with E_inv), t_X, ||C - P_-||_1 (exact routes), the accumulated collar flow, the probes' back-propagation
            check; each run's representation is checked against the requested one; the worst differences print against the propagation's floor
            (page_w1_sector_v1d.excess_floors, information beside the stage's own tolerances); v1d: the floor premise failures are expected 0 (the
            construction allowance covers the first checkpoint).
  held    : v1d (W1_fix_v1d.md): the derived gate. A held record with a trapping zero (the ball's edge fixed at 0.5 r_s), kappa 1 and 2, at the
            production's Chebyshev tolerance (1e-16) and at the control tolerance 1e-13: at every checkpoint and cut |excess| <= the low floor
            2 m h2(T_meas/m) + 2 m h2(8 eps), T_meas = ||Phi_+^dagger Phi_+ - 1||_1 the state's measured impurity (on a held record the whole
            propagation error; the pairing defect |S_Xc(C_+) - S_Xc(P_C)| is its entropy), m = min(n_Xc, n_+) the paired eigenvalues (Lidskii-Mirsky's
            pairing and the binary entropy's modulus of continuity); the premises: T_meas within the a priori bound 4 n_+^(3/2) eps (the construction
            allowance, v1d: v1c's bound was 0 at the first checkpoint and failed there against the state's own construction impurity) + 2 n_+ sum of
            the Chebyshev tails + the rounding allowance, the first checkpoint's impurity within the construction allowance, |E_inv(t) - E_inv(0)| <=
            2 w T_meas + 64 n_+ eps (the quench tail absent: defect 11's gate), the exact route's sum of singular values of (1 - C_+) V_{+,neg} equal to
            T_meas (the drift is a norm drift of the eigencolumns), the pairing defect equal to half the excess (the excess is the impurity's entropy and
            nothing else), the excess present at every checkpoint and cut; the control at 1e-13 passes the same checks and its excess is at least 10
            times the production's (the excess responds to the Chebyshev tolerance: the production's residual is the truncation fallen below the
            rounding); the impurity ratio prints with the Chebyshev orders' parity and the truncation bounds' ratio and decides nothing (v1d: the
            production's impurity is the rounding floor by design, so the ratio is (rounding + truncation_ctrl)/rounding, 2.5 for kappa 1 whose
            control order K + 1 = 11 is odd, a phase error, and 57 for kappa 2; v1c's requirement of 10 on it was mis-specified).
  degree  : the Chebyshev degree of the record step against the sub-steps (the cost statement) at the tolerances 1e-14 and 1e-16, from cheb_order.
  flat    : the flat held vacuum (N = 1, v = 0) at kappa 1 on the 8192-site chain: the cut entropy at 2, 16, 24 and 48 r_s against icloud X3's
            2.05436, 2.74643, 2.88147 and 3.11088 nats at 1e-5 (v1d: the reconciliation's values; X3's own arithmetic allowance below 4.2e-6); the
            30 r_s value for the relabel; --flat-full-check runs v1b's dense route beside, gated against the same values at 1e-5, its difference from
            the sector route printed against the derived arithmetic allowance from the two routes' measured Gram defects (a WARNING above it, not a
            verdict: v1c's 1e-9 on that difference was a round number below the two eigensolves' arithmetic at 8192 and 16384 rows, and failed on
            1.82e-9 after the stage had logged PASS); the verdict prints after every check.
  flat41  : the flat control at K = 41 (kappa 1 to 41 on the 8192-site chain): the retired rule passes, the new rule on the raw entropy fails (no
            majorant), on the excess converges to 0 (defect 10's gate).
Writes <out>/w1_selftest_v1.json and .md (the names RUN_W1_v1d.md and the post-pass read) with PASS/FAIL per gate and the relabel text (defect 12).
CPU suffices (the flat stages take minutes to an hour on 16 threads); --device cuda:0 where a GPU is free.
Usage: PYTHONPATH=<code_v6>:<code_d13> python w1_selftest_v1d.py --out <dir> [--stages ...]"""
import argparse
import glob
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

import torch

for _p in (os.environ.get("W1_CODE_V6"), os.environ.get("W1_CODE_D13")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import page_ops_v6 as OPS  # noqa: E402
import page_w1_sector_v1d as W1S  # noqa: E402  (v1d)
import page_d13_readout_v1d as D13  # noqa: E402  (v1d)
import page_history_v6o as V6O  # noqa: E402

T0 = time.time()
GATES_FLAT = {2.0: 2.05436, 16.0: 2.74643, 24.0: 2.88147, 48.0: 3.11088}    # v1d: icloud X3's values (the reconciliation; X3's own arithmetic allowance below 4.2e-6 nats)
FLAT_TOL = 1e-5                                                              # v1d: the reconciliation's gate
C_EIG_ROUTE = 8.0                                                            # v1d: the eigensolves' indicative rounding per eigenvalue in the route allowance (page_w1_sector_v1d.excess_floors's c_eig)
PROD_CHEB_TOL = 1e-16                                                        # v1c: the production's Chebyshev tolerance (RUN_W1_v1d.md)
CTRL_CHEB_TOL = 1e-13                                                        # v1c: the held gate's scaling control
HELD_RULE = ("held gate (v1d, W1_fix_v1d.md): at every checkpoint and cut |excess| <= 2 m h2(T_meas/m) + 2 m h2(8 eps), T_meas = ||Phi_+^dagger Phi_+ - 1||_1 the state's "
             "measured impurity, m = min(n_Xc, n_+) (Lidskii-Mirsky and the binary entropy's modulus); premises: T_meas <= 4 n_+^(3/2) eps (the construction allowance, "
             "indicative) + 2 n_+ sum tails (the Chebyshev truncation, a theorem) + n_+ n_sub 16 K eps (the rounding allowance, indicative), the first checkpoint's impurity "
             "within the construction allowance, |dE_inv| <= 2 w T_meas + 64 n_+ eps, sum sigma_j((1 - C_+) V_{+,neg}) = T_meas within 0.2 T_meas + 64 n_+ eps, the pairing "
             "defect = |excess|/2 within a quarter plus the rounding term, the excess present at every checkpoint and cut; the control at the 1000 times larger Chebyshev "
             "tolerance passes the same checks and its excess is at least 10 times the production's (the excess responds to the tolerance); its impurity ratio prints with the "
             "Chebyshev orders' parity and the truncation bounds' ratio and decides nothing (the production's impurity is the rounding floor by design)")


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T0, msg), flush=True)


def ent_dense(M):
    ev = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
    return OPS.binary_entropy_np(ev)


def _parity(K):
    """v1d: the Chebyshev order's parity and what its leading truncation coefficient 2 (-i)^(K+1) J_(K+1)(z) does to a norm (page_w1_sector_v1d.propagation_floor)."""
    if K is None:
        return "-"
    k1 = int(K) + 1
    return "%d (%s)" % (k1, "even: a real leading coefficient, a norm error" if k1 % 2 == 0 else "odd: an imaginary leading coefficient, a phase error whose norm drift carries sin(e dt)")


# ------------------------------------------------------------------------------------------------------------------ the fixture
def dust_fixture(nsites=192, rs=16.0, g=0.25, tau=2.0, nrec=61, Rb0_over_rs=4.0, held_Rb_over_rs=None, label=""):
    """the marginally bound dust flow of page_ops_v6 (N = 1, v = -sqrt(r_s/R) outside the ball, -sqrt(r_s/R_b^3) R inside), the ball's edge on
    dust_radius (floor 2 sites), released at Rb0; held_Rb_over_rs fixes the edge (a held record). Synthetic history dicts for page_history_v6o.History."""
    R = np.arange(nsites, dtype=float) + 1.0
    E = rs / (2.0 * g); T = tau * np.arange(nrec, dtype=float)
    N = np.ones((nrec, nsites)); v = np.zeros((nrec, nsites))
    for i, t in enumerate(T):
        Rb = held_Rb_over_rs * rs if held_Rb_over_rs is not None else max(OPS.dust_radius(t, Rb0_over_rs * rs, rs), 2.0)
        v[i] = OPS.flow_dense(rs, Rb)(R)
    Z = {"T": T, "R": R, "N_applied": N, "v_applied": v, "tau": tau, "ell": 1.0, "label": label or "W1 fixture: the marginally bound dust flow, r_s %g, %d sites, tau %g" % (rs, nsites, tau)}
    D = {"args": {"g": g, "rs": rs}, "records": [{"T": float(t), "E": float(E)} for t in T]}
    return D, Z


def run_fixture(D, Z, out, tag, interp, sector, sector_prop, device, threads, nsub=8, T_end_rs=7.5, extra=(), cheb_tol=PROD_CHEB_TOL):
    for f in glob.glob(os.path.join(out, tag + "_*")):                      # v1b: no stale store, columns or state of an earlier run under this tag
        try:
            os.remove(f)
        except OSError:
            pass
    # busi (analyst): str(device), argparse takes strings. v1b: the comment stands on its own line.
    argv = ["--out", out, "--tag", tag, "--nsites", str(len(Z["R"])), "--g", str(D["args"]["g"]), "--rs-nominal", str(D["args"]["rs"]),
            "--channels", "1", "--kappas", "1", "--R-cut-rs", "6", "--R-collar-rs", "4", "--join-rs", "8,11", "--collect-every-rs", "2", "--extra-ck-rs", "",
            "--ext-every-rs", "4", "--vac-every-rs", "4", "--T-end-rs", str(T_end_rs), "--cuts-rs", "2,4,6,9", "--ledger-cuts-rs", "4,6", "--probe-cuts-rs", "4,6",
            "--TH-ref-rs", "6", "--form-rs", "5", "--probe-from-rs", "6", "--probe-every-rs", "2", "--ref-clocks-rs", "4", "--columns-at-rs", "4", "--columns-every-rs", "0",
            "--device", str(device), "--threads", str(threads), "--interp", interp,
            "--interp-sub", str(nsub), "--sector", sector, "--sector-propagate", sector_prop,                                 # v1b: restored
            "--sector-check-every-rs", "0", "--cut-norm-full", "svd", "--force",                                                 # v1b: the exact trace-norm route; rewrite the columns
            "--cheb-tol", "%g" % cheb_tol,                                                                                      # v1c: the production's tolerance by default, the control's for the scaling check
            "--save-every", "1000", "--max-wall-hours", "4", "--kappa-hard-max", "1", "--no-kappa-extend"] + list(extra)
    a = V6O.parse(argv)
    H = V6O.History(None, None, len(Z["R"]), D["args"]["g"], "applied", synthetic={"json": D, "npz": Z})
    run = V6O.HistRun(a, history=H)
    run.run()
    S = json.load(open(os.path.join(out, tag + "_summary.json"))); Dj = json.load(open(os.path.join(out, tag + "_d13.json")))
    return run, S, Dj


def dense_oracle(H, kappa, ck, interp, nsub, cuts, R_collar_abs):
    """the full covariance propagated by the exact dense exponential of every sub-step (the same sub-steps as page_history_v6o); the readouts at the
    checkpoints ck (record indices) before each is stepped."""
    R = np.asarray(H.R, float); h0, Q = OPS.chain_sparse(R, 0.0, float(kappa))
    R2 = np.repeat(R, 2); chi_c = (R2 > R_collar_abs).astype(float)

    def dense_h(N, v):
        return OPS.clocked_csr(h0, Q, N, v).toarray()

    N0, v0 = H.fields(0); h = dense_h(N0, v0)
    e, V = np.linalg.eigh(h)
    C = V[:, e < 0] @ V[:, e < 0].conj().T
    flow_acc = 0.0; out = {}
    kmax = max(ck)

    def readout(k, C, flow_acc):
        N, v = H.fields(k); hk = dense_h(N, v)
        e, V = np.linalg.eigh(hk)
        Vn = V[:, e < 0]; P = Vn @ Vn.conj().T
        absh = (V * np.abs(e)) @ V.conj().T
        Dm = C - P
        r = {"k": k, "E_inv": float(np.real(np.trace(hk @ C))) - float(e[e < 0].sum()), "E_kappa": float(np.real(np.trace(absh @ Dm @ Dm))),
             "nu": float(np.abs(np.linalg.eigvalsh(0.5 * (Dm + Dm.conj().T))).sum()), "collar_flow_acc": flow_acc, "cuts": {}}
        for lab, x in cuts:
            m = R2 > x
            SX = ent_dense(C[m][:, m]); SXc = ent_dense(C[~m][:, ~m]); Sv = ent_dense(P[m][:, m])
            sv = np.linalg.svd(Dm[m][:, ~m], compute_uv=False)
            r["cuts"][lab] = {"S_X": SX, "S_Xc": SXc, "S_X_vac": Sv, "excess": SX - Sv, "t_X": 2.0 * float(sv.sum())}
        return r
    for k in range(0, kmax):
        if k in ck:
            out[k] = readout(k, C, flow_acc)
        for (sa, sb, mode, (N, v)) in W1S.record_substeps(H, k, H.T[k], H.T[k + 1], False, None, interp, nsub):
            hs = dense_h(N, v)
            Eo0 = float(np.real(np.sum(chi_c * np.einsum("ij,ji->i", hs, C))))
            es, Vs = np.linalg.eigh(hs)
            U = (Vs * np.exp(-1j * es * (sb - sa))) @ Vs.conj().T
            C = U @ C @ U.conj().T
            flow_acc += float(np.real(np.sum(chi_c * np.einsum("ij,ji->i", hs, C)))) - Eo0
    out[kmax] = readout(kmax, C, flow_acc)
    return out


# ------------------------------------------------------------------------------------------------------------------- the stages
def stage_pairing(dev):
    n = 128; R = np.arange(n, dtype=float) + 1.0
    rng = np.random.default_rng(1)
    N = 1.0 - 0.5 * np.exp(-((R - 40.0) / 20.0) ** 2) * rng.uniform(0.8, 1.2); v = -1.4 * np.exp(-((R - 50.0) / 25.0) ** 2) * rng.uniform(0.9, 1.1)
    out = {"pass": True, "rows": {}}
    for m, name in ((0.0, "massless"), (0.2, "massive_negative_control")):
        h0, Q = OPS.chain_sparse(R, m, 2.0); h = OPS.clocked_csr(h0, Q, N, v)
        aud = W1S.operator_audit(h)
        e = np.linalg.eigvalsh(h.toarray())
        sa = W1S.spectral_audit(e)
        row = {"operator_audit": aud, "spectral_audit": sa}
        if m == 0.0:
            hp, hm, off = W1S.sector_blocks_csr(h)
            ep = np.linalg.eigvalsh(hp.toarray()); em = np.linalg.eigvalsh(hm.toarray())
            row["spec_minus_plus_negspec_plus_max"] = float(np.max(np.abs(np.sort(em) - np.sort(-ep))))
            row["offblock_max"] = off
            row["full_spec_vs_union_max"] = float(np.max(np.abs(np.sort(e) - np.sort(np.concatenate([ep, -ep])))))
            ok = aud["ok"] and sa["ok"] and row["spec_minus_plus_negspec_plus_max"] < 1e-11 and off < 1e-13 and row["full_spec_vs_union_max"] < 1e-11
            fb = W1S.fast_banded_matches(type("O", (), {"h_csr": lambda self, N_, v_: OPS.clocked_csr(h0, Q, N_, v_), "kappa": 2, "R": R})(), N, v, dev)
            row["fast_banded"] = fb; ok = ok and fb["ok"]
        else:
            ok = (not aud["ok"])
        row["pass"] = bool(ok); out["rows"][name] = row; out["pass"] = out["pass"] and bool(ok)
        log("pairing %s: audit ok %s (max|h + conj h| %.2e, max|[S, h]| %.2e), spectral pairing residual %.2e, %s" % (name, aud["ok"], aud["max_h_plus_conj_h"], aud["max_commutator_S_stag"], sa["pairing_residual_rel"], "PASS" if ok else "FAIL"))
    return out


def stage_sector(dev):
    D, Z = dust_fixture()
    H = V6O.History(None, None, 192, 0.25, "applied", synthetic={"json": D, "npz": Z})
    cuts = [("2", 32.0), ("4", 64.0), ("6", 96.0), ("9", 144.0)]
    ops = V6O.ChannelOps(1, H.R, dev)
    k_state, k_vac = 30, 50
    hs = ops.h_csr(*H.fields(k_state)); hv = ops.h_csr(*H.fields(k_vac))
    st = D13.killing_vacuum(hs, dev, "dense"); Phi = st["Vneg"]
    vd = D13.killing_vacuum(hv, dev, "dense"); vs = W1S.killing_vacuum_sector(hv, dev)
    Pd = vd["Vneg"] @ vd["Vneg"].mH; Ps = vs["Vneg"] @ vs["Vneg"].mH
    out = {"proj_diff_F": float(torch.linalg.norm(Pd - Ps)), "E_ref_diff": abs(vd["E_ref"] - vs["E_ref"]), "n_neg": (vd["n_neg"], vs["n_neg"]), "cuts": {}}
    R2 = torch.as_tensor(np.repeat(H.R, 2), dtype=OPS.RD, device=dev)
    ent_full = D13.exterior_entropies(Phi, R2, cuts, vd["Vneg"])
    ent_sec = W1S.exterior_entropies_sector(Phi, H.R, cuts, vs, check_minus=True); sec = ent_sec.pop("_sector")
    cc = D13.cross_cut_norms(Phi, vd["Vneg"], R2, cuts)
    sc = {"sector": sec}; arrays = {}
    cfg_svd = {"cut_bound_tol": 1e-6, "cut_norm_full": "svd", "cut_sigma_tol": 1e-10}                                            # v1b: the exact route on the sector
    W1S.cross_cut_readout_sector(Phi, vs, H.R, cuts, ent_sec, cfg_svd, 1, 0.0, sc, arrays)
    tn_sec_svd = sc["trace_norm"]
    tn_full_svd = D13.full_trace_norm(Phi, vd["Vneg"], "svd")                                                                    # exact on the full chain
    tn_full_gram = D13.full_trace_norm(Phi, vd["Vneg"], "gram")                                                                  # v1b: v1b's Gram route on the full chain, for the record
    ep = torch.as_tensor(np.asarray(vs["e_plus"], float), dtype=OPS.RD, device=dev)
    Pp = W1S.to_sector(Phi, 1); Vp = vs["V_plus"][:, ep < 0]
    tn_sec_gram = W1S.full_trace_norm_sector(Pp, Vp, {"cut_norm_full": "gram"}, 1e-10)                                           # v1b: the production's route, against its bound
    worst = 0.0
    for lab, _ in cuts:
        d = {"S_X_full": ent_full[lab]["S_X"], "S_Xc_full": ent_full[lab]["S_Xc"], "S_X_sector": ent_sec[lab]["S_X"], "S_X_sector_Xblocks": ent_sec[lab].get("S_X_both_Xblocks"),
             "S_vac_full": ent_full[lab]["S_X_vac"], "S_vac_sector": ent_sec[lab]["S_X_vac"], "t_X_full": cc[lab]["t_X"], "t_X_sector": ent_sec[lab]["t_X"],
             "pairing_defect_state": ent_sec[lab].get("pairing_defect_state"), "pairing_defect_vacuum": ent_sec[lab].get("pairing_defect_vacuum"),
             "purity_gap_full": ent_full[lab]["purity_gap"], "purity_gap_sector": ent_sec[lab]["purity_gap"], "pairing_defect_t_X": ent_sec[lab].get("pairing_defect_t_X")}
        for a_, b_ in (("S_X_full", "S_X_sector"), ("S_Xc_full", "S_X_sector"), ("S_X_full", "S_X_sector_Xblocks"), ("S_vac_full", "S_vac_sector"), ("t_X_full", "t_X_sector")):
            worst = max(worst, abs(d[a_] - d[b_]))
        worst = max(worst, d["pairing_defect_state"], d["pairing_defect_vacuum"], d["pairing_defect_t_X"] or 0.0)
        out["cuts"][lab] = d
    out["nu_full_svd"] = tn_full_svd["nu"]; out["nu_sector_svd"] = tn_sec_svd["nu"]; out["nu_full_gram"] = tn_full_gram["nu"]; out["nu_sector_gram"] = tn_sec_gram["nu"]
    out["nu_diff_exact_routes"] = abs(tn_full_svd["nu"] - tn_sec_svd["nu"])                                                                  # v1b
    out["gram_sector_defect"] = abs(tn_sec_gram["nu"] - tn_sec_svd["nu"]); out["gram_sector_bound"] = tn_sec_gram["rounding_bound_nu"]                 # v1b
    out["gram_full_defect"] = abs(tn_full_gram["nu"] - tn_full_svd["nu"]); out["gram_full_bound"] = tn_full_gram.get("rounding_bound_nu")                 # v1b (printed, not gated)
    out["gram_sector_within_bound"] = bool(out["gram_sector_defect"] <= out["gram_sector_bound"])
    out["worst_entropy_diff"] = worst; out["nu_diff"] = out["nu_diff_exact_routes"]
    out["reading"] = ("the first gate's nu difference 7.35e-6 was the Gram route's square-root accuracy (the sector's Gram against the full chain's exact route); v1b compares "
                      "exact against exact and checks the Gram route against its indicative bound 4 q sqrt(2 eps q)")
    out["pass"] = bool(out["proj_diff_F"] < 1e-10 and out["E_ref_diff"] < 1e-9 and worst < 1e-9 and out["nu_diff"] < 1e-8 and out["gram_sector_within_bound"])
    log("sector: projector difference %.2e, E_ref difference %.2e, worst entropy/t_X/pairing difference %.2e, nu difference (exact routes) %.2e; the Gram route's defect on the sector %.2e against its bound %.2e (%s), on the full chain %.2e against %s: %s" % (
        out["proj_diff_F"], out["E_ref_diff"], worst, out["nu_diff"], out["gram_sector_defect"], out["gram_sector_bound"], "within" if out["gram_sector_within_bound"] else "OUTSIDE",
        out["gram_full_defect"], ("%.2e" % out["gram_full_bound"]) if out["gram_full_bound"] is not None else "-", "PASS" if out["pass"] else "FAIL"))
    return out


def _floor_max(Dj, kappa):
    """v1c: the largest high floor over the clocks and cuts of a run's D13 json (information for the oracle stage)."""
    fm = 0.0
    for c in (Dj.get("clocks") or {}).values():
        for cut in (((c.get("per_channel") or {}).get(str(kappa)) or {}).get("cuts") or {}).values():
            if cut.get("excess_floor_high") is not None:
                fm = max(fm, float(cut["excess_floor_high"]))
    return fm


def stage_oracle(dev, threads, tmp):
    D, Z = dust_fixture()
    H = V6O.History(None, None, 192, 0.25, "applied", synthetic={"json": D, "npz": Z})
    cuts = [("2", 32.0), ("4", 64.0), ("6", 96.0), ("9", 144.0)]
    out = {"pass": True, "runs": {}, "cheb_tol": PROD_CHEB_TOL}
    tol = {"S": 1e-8, "E": 1e-9, "flow": 1e-9, "t": 1e-8, "nu": 1e-8, "probe": 1e-9, "identity": 1e-10}                        # v1b: nu exact on both sides (--cut-norm-full svd)
    for interp in ("linear", "none"):
        oracle = None
        for (sector, sprop) in (("on", "on"), ("on", "off"), ("off", "off")):
            tag = "fx_%s_s%s_p%s" % (interp, sector, sprop)
            run, S, Dj = run_fixture(D, Z, tmp, tag, interp, sector, sprop, dev, threads)
            ck = sorted(run.ck)
            if oracle is None:
                oracle = dense_oracle(H, 1, ck, interp, 8, cuts, 64.0)
            w1 = S.get("w1") or {}
            n_vac = sum(1 for r in S["rows"] if r.get("vacuum_dense"))
            rep = {"sector_used": bool(w1.get("sector_used")), "sector_propagate": bool(all(w1.get("sector_propagate") or [False])), "interp": w1.get("interp"),
                   "vacuum_clocks": n_vac, "rows": len(S["rows"]), "checkpoints": len(ck)}
            rep_ok = (rep["sector_used"] == (sector == "on")) and (rep["sector_propagate"] == (sector == "on" and sprop == "on")) and (rep["interp"] == interp) \
                and (rep["rows"] == rep["checkpoints"]) and (n_vac == (len(ck) if sector == "on" else len([k for k in ck if k in run.vac_k])))   # v1b: the requested representation is the one run
            worst = {k_: 0.0 for k_ in ("S", "Svac", "excess", "E", "flow", "t", "nu", "identity", "probe", "Ekappa_spectral")}
            impurity_max = 0.0; premise_fail = 0; T_init = None                                                                  # v1c; v1d: T_init
            for r in S["rows"]:
                k = r["k"]; pc = r["per_channel"]["1"]; o = oracle[k]
                worst["E"] = max(worst["E"], abs(pc["E_inv"] - o["E_inv"]))
                worst["flow"] = max(worst["flow"], abs(pc["collar_flow_acc"] - o["collar_flow_acc"]))
                worst["identity"] = max(worst["identity"], abs(o["E_kappa"] - o["E_inv"]))
                fe = ((pc.get("w1") or {}).get("E_kappa_spectral") or {})
                if fe.get("E_kappa") is not None:
                    worst["Ekappa_spectral"] = max(worst["Ekappa_spectral"], abs(fe["E_kappa"] - o["E_inv"]))
                w1r = pc.get("w1") or {}                                                                                           # v1c
                if w1r.get("sea_unitarity_trace") is not None:                                                                   # v1c
                    impurity_max = max(impurity_max, float(w1r["sea_unitarity_trace"]))                                           # v1c
                    if ((w1r.get("propagation_floor") or {}).get("premise_ok")) is False:                                       # v1c
                        premise_fail += 1                                                                                        # v1c
                    if T_init is None:                                                                                           # v1d
                        T_init = w1r.get("T_init")                                                                               # v1d
                for pr in pc.get("d13_probes") or []:
                    for q in pr["refs"]:
                        worst["probe"] = max(worst["probe"], abs(q["check_L_minus_direct"]))
                key = D13.Store.ckey(r["t_over_rs"]); cl = Dj["clocks"][key]["per_channel"]["1"]
                for lab, _ in cuts:
                    c = cl["cuts"][lab]; oc = o["cuts"][lab]
                    worst["S"] = max(worst["S"], abs(c["S_X"] - oc["S_X"]))
                    if c.get("S_X_vac") is not None:
                        worst["Svac"] = max(worst["Svac"], abs(c["S_X_vac"] - oc["S_X_vac"])); worst["excess"] = max(worst["excess"], abs(c["excess"] - oc["excess"]))
                    if c.get("t_X") is not None:
                        worst["t"] = max(worst["t"], abs(c["t_X"] - oc["t_X"]))
                tn = cl.get("trace_norm") or {}
                if tn.get("nu") is not None:
                    worst["nu"] = max(worst["nu"], abs(tn["nu"] - o["nu"]))
            ok = (worst["S"] < tol["S"] and worst["Svac"] < tol["S"] and worst["excess"] < tol["S"] and worst["E"] < tol["E"] and worst["flow"] < tol["flow"] and worst["t"] < tol["t"]
                  and worst["nu"] < tol["nu"] and worst["identity"] < tol["identity"] and worst["probe"] < tol["probe"] and worst["Ekappa_spectral"] < tol["E"] and rep_ok)
            Ks = [q["per_channel"]["1"].get("cheb_K_sum") for q in S["records"]]
            fmax = _floor_max(Dj, 1)                                                                                             # v1c
            out["runs"][tag] = {"worst": worst, "pass": bool(ok), "representation": rep, "representation_ok": bool(rep_ok), "cheb_K_sum_mean_per_record": float(np.mean(Ks)) if Ks else None,
                                "E_inv_end": S["rows"][-1]["per_channel"]["1"]["E_inv"], "sector_used": w1.get("sector_used"), "wall_s": S.get("wall_s"),
                                "audit": w1.get("audit"), "trace_norm_route": ((Dj["clocks"][D13.Store.ckey(S["rows"][-1]["t_over_rs"])]["per_channel"]["1"].get("trace_norm") or {}).get("route")),
                                "propagation_floor_high_max": fmax, "S_within_floor": bool(worst["S"] <= fmax and worst["excess"] <= fmax), "impurity_max": impurity_max,
                                "floor_premise_failures": premise_fail, "T_init": T_init}                                        # v1c: information beside the stage's tolerances; v1d: the first checkpoint's impurity
            out["pass"] = out["pass"] and bool(ok)
            log("oracle %s: %s; representation %s (sector_used %s, propagate %s, interp %s, vacuum at %d of %d checkpoints); worst differences %s; against the propagation's floor (high, max over the clocks and cuts) %.2e: %s; the impurity's max %.2e (at the first checkpoint %s), floor premise failures %d (expected 0: the construction allowance)" % (
                tag, "PASS" if ok else "FAIL", "as requested" if rep_ok else "NOT as requested", rep["sector_used"], rep["sector_propagate"], rep["interp"], n_vac, len(ck),
                ", ".join("%s %.1e" % (k_, v_) for k_, v_ in worst.items()), fmax, "within" if out["runs"][tag]["S_within_floor"] else "ABOVE (the oracle's own rounding or a defect)", impurity_max,
                ("%.2e" % T_init) if T_init is not None else "-", premise_fail))
    e_lin = out["runs"]["fx_linear_son_pon"]["E_inv_end"]; e_none = out["runs"]["fx_none_son_pon"]["E_inv_end"]
    out["E_inv_end_production_vs_control"] = {"linear": e_lin, "none": e_none, "difference_none_minus_linear": e_none - e_lin}
    log("oracle: E_inv at the end, production (linear) %.6e against the control (none) %.6e: the control's excess %.3e" % (e_lin, e_none, e_none - e_lin))
    return out


def _held_run(D, Z, tmp, kappa, tol, dev, threads):
    """v1c: one held run and its derived-gate checks (W1_fix_v1c.md section 3); v1d: the premise with the construction allowance, the construction check
    at the first checkpoint, the Chebyshev order's parity recorded for the scaling control's print."""
    tag = "held_k%d_tol%02d" % (kappa, int(round(-math.log10(tol))))
    run, S, Dj = run_fixture(D, Z, tmp, tag, "linear", "on", "on", dev, threads, T_end_rs=4.0, extra=["--channels", str(kappa), "--kappas", str(kappa)], cheb_tol=tol)
    rows = S["rows"]; kap = str(kappa)
    E0 = rows[0]["per_channel"][kap]["E_inv"]
    w = {"excess": 0.0, "excess_over_floor": 0.0, "floor_low": 0.0, "T_meas": 0.0, "T_meas_over_cheb": 0.0, "dE_inv": 0.0, "dE_over_tol": 0.0, "consistency_rel": 0.0, "pairing_identity_rel": 0.0,
         "T_init": None, "construction_allowance": None, "construction_ratio": None}                                               # v1d
    ok = {"floor": True, "premise": True, "construction": True, "E_inv": True, "consistency": True, "pairing": True, "present": True, "inputs": True}   # v1d: construction
    n_exc = 0; checks = []
    for r in rows:
        pc = r["per_channel"][kap]; w1 = pc.get("w1") or {}; fl = w1.get("propagation_floor") or {}
        T = w1.get("sea_unitarity_trace"); Tc = fl.get("T_cheb"); wb = w1.get("h_bound"); n_plus = w1.get("n0_plus") or fl.get("n_plus")
        if T is None or Tc is None or wb is None or not n_plus:
            ok["inputs"] = False; continue
        w["T_meas"] = max(w["T_meas"], T); w["T_meas_over_cheb"] = max(w["T_meas_over_cheb"], T / Tc if Tc > 0 else float("inf"))
        if T > Tc:
            ok["premise"] = False
        if w["T_init"] is None:                                                                                                   # v1d: the first checkpoint's impurity against the construction allowance
            w["T_init"] = fl.get("T_init"); w["construction_allowance"] = fl.get("T_cheb_construction")
            if w["T_init"] is not None and w["construction_allowance"]:
                w["construction_ratio"] = w["T_init"] / w["construction_allowance"]
                if w["T_init"] > w["construction_allowance"]:
                    ok["construction"] = False
            elif w["T_init"] is None or w["construction_allowance"] is None:
                ok["inputs"] = False
        dE = abs(pc["E_inv"] - E0); tolE = 2.0 * wb * T + 64.0 * n_plus * W1S.EPS64
        w["dE_inv"] = max(w["dE_inv"], dE); w["dE_over_tol"] = max(w["dE_over_tol"], dE / tolE if tolE > 0 else float("inf"))
        if dE > tolE:
            ok["E_inv"] = False
        cl = Dj["clocks"][D13.Store.ckey(r["t_over_rs"])]["per_channel"][kap]
        tn = cl.get("trace_norm") or {}
        if tn.get("nu_plus") is not None and tn.get("N_occ_plus") is not None:
            sig = 0.5 * (tn["nu_plus"] - (tn["N_occ_plus"] - tn["N_neg_plus"]))
            tolc = 0.2 * T + 64.0 * n_plus * W1S.EPS64
            w["consistency_rel"] = max(w["consistency_rel"], abs(sig - T) / tolc if tolc > 0 else float("inf"))
            if abs(sig - T) > tolc:
                ok["consistency"] = False
        else:
            ok["inputs"] = False
        for lab, c in cl["cuts"].items():
            if c.get("excess") is None:
                continue
            n_exc += 1; exc = abs(c["excess"]); low = c.get("excess_floor_low"); rnd = c.get("excess_floor_round") or 0.0
            if low is None:
                ok["inputs"] = False; continue
            w["excess"] = max(w["excess"], exc); w["floor_low"] = max(w["floor_low"], low); w["excess_over_floor"] = max(w["excess_over_floor"], exc / low if low > 0 else float("inf"))
            if exc > low:
                ok["floor"] = False
            pd = c.get("pairing_defect_state")
            if pd is not None:
                tolp = 0.25 * (exc / 2.0) + rnd
                w["pairing_identity_rel"] = max(w["pairing_identity_rel"], abs(pd - exc / 2.0) / tolp if tolp > 0 else float("inf"))
                if abs(pd - exc / 2.0) > tolp:
                    ok["pairing"] = False
            checks.append({"t_over_rs": r["t_over_rs"], "cut": lab, "excess": c["excess"], "floor_low": low, "floor_high": c.get("excess_floor_high"), "round": rnd, "m_pairs": c.get("m_pairs"),
                           "pairing_defect": pd, "resolution": c.get("excess_resolution")})
    ok["present"] = (n_exc == len(rows) * 4)
    passed = all(ok.values())
    last = rows[-1]["per_channel"][kap].get("w1") or {}
    lfl = last.get("propagation_floor") or {}
    res = {"tag": tag, "cheb_tol": tol, "pass": bool(passed), "ok": ok, "worst": w, "n_excess_values": n_exc, "n_rows": len(rows), "E_inv_T0": E0, "checks": checks,
           "n_substeps": last.get("n_substeps"), "cheb_K_max": lfl.get("cheb_K_max"), "cheb_tail_acc": lfl.get("cheb_tail_acc"),
           "T_cheb_truncation": lfl.get("T_cheb_truncation"), "T_cheb_rounding_allowance": lfl.get("T_cheb_rounding_allowance"), "T_cheb_construction": lfl.get("T_cheb_construction"),   # v1d
           "h_bound": last.get("h_bound"), "K_plus_1_parity": _parity(lfl.get("cheb_K_max"))}                                     # v1d
    log("held kappa %d cheb_tol %g (%d sub-steps, K max %s, K + 1 %s, sum of tails %s): worst |excess| %.2e against its floor (max ratio %.2f, floor max %.2e); the impurity ||Phi^dagger Phi - 1||_1 %.2e against the a priori "
        "bound (max ratio %.2f; construction allowance %s, truncation %s, rounding allowance %s); at the first checkpoint %s against the construction allowance (ratio %s); |dE_inv| %.2e against 2 w T + 64 n eps "
        "(max ratio %.2f, w %s); the exact route's drift against the impurity (worst, in tolerance units) %.2f; the pairing defect against half the excess (worst, in tolerance units) %.2f; %d of %d excess values present: %s%s" % (
            kappa, tol, res["n_substeps"] or 0, res["cheb_K_max"], res["K_plus_1_parity"], ("%.2e" % res["cheb_tail_acc"]) if res["cheb_tail_acc"] is not None else "-", w["excess"], w["excess_over_floor"], w["floor_low"], w["T_meas"], w["T_meas_over_cheb"],
            ("%.2e" % res["T_cheb_construction"]) if res["T_cheb_construction"] is not None else "-", ("%.2e" % res["T_cheb_truncation"]) if res["T_cheb_truncation"] is not None else "-",
            ("%.2e" % res["T_cheb_rounding_allowance"]) if res["T_cheb_rounding_allowance"] is not None else "-",
            ("%.2e" % w["T_init"]) if w["T_init"] is not None else "-", ("%.2f" % w["construction_ratio"]) if w["construction_ratio"] is not None else "-",
            w["dE_inv"], w["dE_over_tol"], ("%.3f" % res["h_bound"]) if res["h_bound"] is not None else "-", w["consistency_rel"], w["pairing_identity_rel"], n_exc, len(rows) * 4,
            "PASS" if passed else "FAIL", "" if passed else (" (" + ", ".join(k_ for k_, v_ in ok.items() if not v_) + ")")))
    return res


def stage_held(dev, threads, tmp):
    """v1c: the derived held gate (HELD_RULE) for kappa 1 and 2 at the production's Chebyshev tolerance and at the control tolerance; v1d: the scaling
    control gates the excess ratio and prints the impurity ratio with the orders' parity and the truncation bounds' ratio."""
    out = {"pass": True, "channels": {}, "rule": HELD_RULE, "prod_cheb_tol": PROD_CHEB_TOL, "ctrl_cheb_tol": CTRL_CHEB_TOL}
    for kappa in (1, 2):
        D, Z = dust_fixture(nrec=33, held_Rb_over_rs=0.5, label="W1 held fixture: the dust flow held with the edge at 0.5 r_s (a trapping zero at R = r_s)")
        rp = _held_run(D, Z, tmp, kappa, PROD_CHEB_TOL, dev, threads)
        rc = _held_run(D, Z, tmp, kappa, CTRL_CHEB_TOL, dev, threads)
        ratio_T = (rc["worst"]["T_meas"] / rp["worst"]["T_meas"]) if rp["worst"]["T_meas"] > 0 else float("inf")
        ratio_x = (rc["worst"]["excess"] / rp["worst"]["excess"]) if rp["worst"]["excess"] > 0 else float("inf")
        ratio_trunc = (rc["T_cheb_truncation"] / rp["T_cheb_truncation"]) if (rc.get("T_cheb_truncation") and rp.get("T_cheb_truncation")) else None   # v1d
        scale_ok = bool(ratio_x >= 10.0)                                                                                          # v1d: the excess ratio gates; the impurity ratio prints (information)
        ok = bool(rp["pass"] and rc["pass"] and scale_ok)
        out["channels"][kappa] = {"production": rp, "control": rc, "scaling": {"impurity_ratio": ratio_T, "excess_ratio": ratio_x, "truncation_bound_ratio": ratio_trunc, "ok": scale_ok,
                                                                                "K_plus_1_production": rp["K_plus_1_parity"], "K_plus_1_control": rc["K_plus_1_parity"],
                                                                                "rule": "the excess ratio at least 10 gates (the excess responds to the Chebyshev tolerance); the impurity ratio is information: the production's impurity is the rounding floor by design, the control's the same floor plus its truncation, whose norm drift depends on the Chebyshev order's parity (v1d)"}, "pass": ok}
        out["pass"] = out["pass"] and ok
        log("held kappa %d: the scaling control at cheb_tol %g against the production's %g: excess ratio %.1f (at least 10 required: the excess responds to the Chebyshev tolerance, so the production's residual is the truncation fallen below the rounding): %s; "
            "the impurity ratio %.1f prints as information (the production's impurity is the rounding floor by design; the a priori truncation bounds' ratio %s; K + 1 at the production %s, at the control %s); channel %s" % (
                kappa, CTRL_CHEB_TOL, PROD_CHEB_TOL, ratio_x, "PASS" if scale_ok else "FAIL", ratio_T, ("%.0f" % ratio_trunc) if ratio_trunc is not None else "-",
                rp["K_plus_1_parity"], rc["K_plus_1_parity"], "PASS" if ok else "FAIL"))
    return out


def stage_degree():
    out = {"rows": []}
    for kappa in (1, 4, 8, 16):
        w = kappa + 1.5
        Kf, _ = OPS.cheb_order(w * 8.0, 1e-14)
        Kf16, _ = OPS.cheb_order(w * 8.0, 1e-16)                                                                                # v1c
        row = {"kappa": kappa, "bound": w, "K_record_step": Kf, "K_record_step_1e-16": Kf16}
        for n in (2, 4, 8):
            Ks, _ = OPS.cheb_order(w * 8.0 / n, 1e-14)
            row["K_sum_%d_substeps" % n] = n * Ks; row["ratio_%d" % n] = n * Ks / Kf
        Ks16, _ = OPS.cheb_order(w * 8.0 / 8, 1e-16)                                                                            # v1c
        row["K_sum_8_substeps_1e-16"] = 8 * Ks16; row["ratio_8_1e-16"] = 8 * Ks16 / Kf; row["cost_1e-16_over_1e-14"] = (8 * Ks16) / row["K_sum_8_substeps"]   # v1c
        out["rows"].append(row)
        log("degree kappa %d (bound %.1f): record step K %d (1e-16: %d); sub-steps 2/4/8 at 1e-14: %d/%d/%d (ratios %.2f/%.2f/%.2f); 8 sub-steps at 1e-16: %d (ratio %.2f to the record step, %.3f to the 1e-14 sub-steps)" % (
            kappa, w, Kf, Kf16, row["K_sum_2_substeps"], row["K_sum_4_substeps"], row["K_sum_8_substeps"], row["ratio_2"], row["ratio_4"], row["ratio_8"], row["K_sum_8_substeps_1e-16"], row["ratio_8_1e-16"], row["cost_1e-16_over_1e-14"]))
    out["note"] = ("the propagation's degree per record for --interp linear with 8 sub-steps against the record step's is the ratio_8 column per kappa; the production's tolerance 1e-16 "
                   "(v1c: the truncation part of the floor falls by 100) costs the cost_1e-16_over_1e-14 column; the Gershgorin bound stands in as kappa + 1.5; the + sector propagation "
                   "applies each term at about 1/5 of the full apply's memory traffic")
    out["pass"] = True
    return out


def flat_vacuum_entropies(kappa, nsites, rs, cuts_rs, dev, want_defect=False):
    """the flat held vacuum's kappa cut entropies by the sector route; v1d: returns (values, h, info) with info the + sector's column count, the paired
    eigenvalue counts per cut and, with want_defect, the vacuum columns' construction impurity ||V_+^dagger V_+ - 1||_1."""
    R = np.arange(nsites, dtype=float) + 1.0
    h0, Q = OPS.chain_sparse(R, 0.0, float(kappa))
    h = OPS.clocked_csr(h0, Q, np.ones(nsites), np.zeros(nsites))
    vac = W1S.killing_vacuum_sector(h, dev)
    ep = torch.as_tensor(vac["e_plus"], dtype=OPS.RD, device=dev)
    Vp = vac["V_plus"][:, ep < 0]
    out = {}; m_sites = {}
    for x in cuts_rs:
        mask = torch.as_tensor(R > x * rs, dtype=torch.bool, device=dev)
        ev, _ = W1S.gram_eigs(Vp[~mask])
        out[float(x)] = 2.0 * W1S.entropy_of_eigs(ev)
        m_sites[float(x)] = int(min(int((~mask).sum()), int(Vp.shape[1])))                                                       # v1d: the paired eigenvalues of the sector route
    info = {"n_plus": int(Vp.shape[1]), "m_sites": m_sites, "T_sec": (W1S.impurity_trace_norm(Vp) if want_defect else None)}       # v1d
    del Vp, vac
    return out, h, info


def route_allowance(m_s, T_s, m_f, T_f, c_eig=C_EIG_ROUTE):
    """v1d: the derived allowance for the difference of the two routes' flat held-vacuum entropies at one cut: the sector route's 2 m_s h2(T_s/m_s) (the
    entropy modulus of its vacuum columns' impurity T_s = ||V_+^dagger V_+ - 1||_1 over m_s = min(n_Xc sites, n_+) paired eigenvalues, doubled by the
    pairing) + 2 m_s h2(c_eig eps) (its Gram eigensolve's rounding, indicative) + the full route's m_f h2(T_f/m_f) (T_f = ||V^dagger V - 1||_1 of the dense
    route's occupied columns over m_f = min(n_X modes, N_occ) eigenvalues) + m_f h2(c_eig eps). The eigenvalue-rounding constant is indicative, so the
    allowance prints and warns and is not the gate (W1_fix_v1d.md section 2)."""
    return (2.0 * W1S.entropy_modulus(m_s, T_s) + 2.0 * W1S.entropy_modulus(m_s, m_s * c_eig * W1S.EPS64)
            + W1S.entropy_modulus(m_f, T_f) + W1S.entropy_modulus(m_f, m_f * c_eig * W1S.EPS64))


def _implied_c_eig(diff, m_s, T_s, m_f, T_f):
    """v1d: the smallest eigenvalue-rounding constant in a geometric list for which the derived route allowance covers the measured difference."""
    for c in (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0):
        if route_allowance(m_s, T_s, m_f, T_f, c) >= diff:
            return c
    return None


def stage_flat(dev, full_check):
    t0 = time.time()
    cuts = (2.0, 16.0, 24.0, 30.0, 48.0)
    vals, h, info = flat_vacuum_entropies(1, 8192, 64.0, cuts, dev, want_defect=True)                                            # v1d
    n_plus = info["n_plus"]; T_s = info["T_sec"]
    out = {"values": vals, "gates": GATES_FLAT, "tol": FLAT_TOL, "diffs": {}, "pass": True, "seconds_sector": time.time() - t0,
           "sector_route": {"n_plus": n_plus, "impurity_T": T_s, "impurity_over_n32eps": (T_s / (n_plus ** 1.5 * W1S.EPS64)) if n_plus else None,
                            "construction_allowance": W1S.construction_allowance(n_plus)},                                       # v1d
           "gate_label": "icloud X3's own run of the flat held vacuum (the reconciliation): 2.05436, 2.74643, 2.88147 and 3.11088 nats at 2, 16, 24 and 48 r_s, its arithmetic allowance below 4.2e-6 nats; the gate 1e-5 on the sector route and, with --flat-full-check, on the dense route (v1d)"}
    for x, g in GATES_FLAT.items():
        d = abs(vals[x] - g); out["diffs"][x] = d
        out["pass"] = out["pass"] and d <= FLAT_TOL
    out["relabel_text"] = W1S.relabel_text(vals)
    log("flat held vacuum kappa 1 (8192 sites, r_s 64), the sector route: %s; X3's gates %s at %g: %s; the vacuum columns' construction impurity ||V_+^dagger V_+ - 1||_1 %.2e over %d columns (%.2f n_+^(3/2) eps; the allowance %.2e)" % (
        ", ".join("%g r_s %.6f" % (x, vals[x]) for x in sorted(vals)), ", ".join("%g r_s %.5f (diff %.1e)" % (x, GATES_FLAT[x], out["diffs"][x]) for x in sorted(GATES_FLAT)), FLAT_TOL,
        "within" if out["pass"] else "OUTSIDE", T_s, n_plus, out["sector_route"]["impurity_over_n32eps"] or 0.0, out["sector_route"]["construction_allowance"]))
    if full_check:
        t1 = time.time()
        vd = D13.killing_vacuum(h, dev, "dense")
        Vn = vd["Vneg"]; N_occ = int(Vn.shape[1])
        T_f = W1S.impurity_trace_norm(Vn)                                                                                      # v1d: the dense route's construction impurity
        R2 = torch.as_tensor(np.repeat(np.arange(8192, dtype=float) + 1.0, 2), dtype=OPS.RD, device=dev)
        ent = D13.exterior_entropies(Vn, R2, [("%g" % x, x * 64.0) for x in cuts])
        full = {x: ent["%g" % x]["S_X"] for x in cuts}
        out["full_route"] = {"values": full, "n_occ": N_occ, "impurity_T": T_f, "impurity_over_n32eps": T_f / (N_occ ** 1.5 * W1S.EPS64), "construction_allowance": W1S.construction_allowance(N_occ),
                             "diffs": {x: abs(full[x] - g) for x, g in GATES_FLAT.items()}, "purity_gap": {x: ent["%g" % x]["purity_gap"] for x in cuts}}   # v1d
        full_ok = all(d <= FLAT_TOL for d in out["full_route"]["diffs"].values())                                               # v1d: the dense route against X3 too
        per = {}
        for x in cuts:
            m_s = info["m_sites"][x]; m_f = min(int(ent["%g" % x]["n_modes_X"]), N_occ)
            diff = abs(full[x] - vals[x]); allow = route_allowance(m_s, T_s, m_f, T_f)
            per[x] = {"diff": diff, "allowance": allow, "within": bool(diff <= allow), "m_sector": m_s, "m_full": m_f, "implied_c_eig": _implied_c_eig(diff, m_s, T_s, m_f, T_f),
                      "purity_gap_full": ent["%g" % x]["purity_gap"]}
        out["route_comparison"] = per
        out["full_vs_sector_max"] = max(p["diff"] for p in per.values())
        out["route_within_allowance"] = all(p["within"] for p in per.values())
        out["seconds_full"] = time.time() - t1
        out["pass"] = out["pass"] and full_ok                                                                                    # v1d: the route difference gates nothing (its allowance is indicative); X3's values do
        log("flat: v1b's dense route beside (%.0f s): %s; against X3's gates at %g: %s; its columns' construction impurity ||V^dagger V - 1||_1 %.2e over %d columns (%.2f N^(3/2) eps); the two routes' difference %s (max %.2e) against the derived allowance from the measured Gram defects at c_eig %g: %s" % (
            out["seconds_full"], ", ".join("%g r_s %.6f" % (x, full[x]) for x in sorted(full)), FLAT_TOL, "within" if full_ok else "OUTSIDE", T_f, N_occ, out["full_route"]["impurity_over_n32eps"],
            ", ".join("%g r_s %.1e (allowance %.1e, %s)" % (x, per[x]["diff"], per[x]["allowance"], "within" if per[x]["within"] else "above, c_eig %s needed" % per[x]["implied_c_eig"]) for x in sorted(per)),
            out["full_vs_sector_max"], C_EIG_ROUTE, "within" if out["route_within_allowance"] else "ABOVE"))
        if not out["route_within_allowance"]:
            log("flat WARNING: the two routes differ by more than the derived arithmetic allowance at c_eig %g (the eigensolves' indicative rounding per eigenvalue): the implied c_eig prints per cut; "
                "the gate stands on X3's independent values at %g, which both routes meet, so the warning decides nothing (v1c's verdict on this difference at a round 1e-9 was the test's error, W1_fix_v1d.md section 2)" % (C_EIG_ROUTE, FLAT_TOL))
        del Vn, vd
    out["verdict"] = "PASS" if out["pass"] else "FAIL"
    log("flat: %s (the gate: X3's values at %g on the sector route%s)" % (out["verdict"], FLAT_TOL, " and on the dense route" if full_check else ""))   # v1d: the verdict after every check
    return out


def stage_flat41(dev, K=41):
    raw = {x: [] for x in (2.0, 16.0, 24.0, 48.0)}
    for kappa in range(1, K + 1):
        vals, _, _ = flat_vacuum_entropies(kappa, 8192, 64.0, (2.0, 16.0, 24.0, 48.0), dev)                                     # v1d: 3 returns
        for x in raw:
            raw[x].append(16.0 * kappa * vals[x])
        if kappa in (1, 2, 6, 10, 20, 30, K):
            log("flat41: kappa %d done; raw S_kappa at 16 r_s %.6f (weighted %.3f)" % (kappa, vals[16.0], raw[16.0][-1]))
    zeros = [0.0] * K
    out = {"K": K, "per_cut": {}, "pass": True}
    for x, terms in raw.items():
        res = W1S.flat_control_rule(terms, zeros, zeros, zeros, 0.01)
        ok = res["retired_rule_passes"] and (not res["new_rule_raw"]["converged"]) and res["new_rule_excess"]["converged"]
        res["pass"] = bool(ok); res["partial_sums"] = {k_: float(sum(terms[:k_])) for k_ in (6, 10, 20, 41) if k_ <= K}; res["raw_terms"] = terms
        out["per_cut"][x] = res; out["pass"] = out["pass"] and bool(ok)
        log("flat41 cut %g r_s: raw partial sums %s; retired rule passes %s (hard share %.3f; its raw share %.4f); new rule on the raw sum converged %s (%s); on the excess converged %s: %s" % (
            x, res["partial_sums"], res["retired_rule_passes"], res["retired_rule_hard_share"], res["retired_rule_raw_share"], res["new_rule_raw"]["converged"], res["new_rule_raw"]["premise"]["reason"],
            res["new_rule_excess"]["converged"], "PASS" if ok else "FAIL"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="pairing,sector,oracle,held,degree,flat,flat41")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--flat-full-check", dest="flat_full", action="store_true", help="flat: run v1b's dense route beside (16384 x 16384 eigh; gated against X3's values too, its difference from the sector route against the derived allowance)")
    ap.add_argument("--flat41-K", dest="flat41_K", type=int, default=41)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "fixture_runs"); os.makedirs(tmp, exist_ok=True)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    res = {"device": str(dev), "selftest": "w1_selftest_v1d.py", "stages": {}, "code": {"v6o": V6O.__file__, "v1d": D13.__file__, "sector": W1S.__file__}, "prod_cheb_tol": PROD_CHEB_TOL,
           "flat_gates": GATES_FLAT, "flat_tol": FLAT_TOL}                                                                       # v1d
    for s in stages:
        t0 = time.time()
        log("=== stage %s ===" % s)
        if s == "pairing":
            res["stages"][s] = stage_pairing(dev)
        elif s == "sector":
            res["stages"][s] = stage_sector(dev)
        elif s == "oracle":
            res["stages"][s] = stage_oracle(dev, a.threads, tmp)
        elif s == "held":
            res["stages"][s] = stage_held(dev, a.threads, tmp)
        elif s == "degree":
            res["stages"][s] = stage_degree()
        elif s == "flat":
            res["stages"][s] = stage_flat(dev, a.flat_full)
        elif s == "flat41":
            res["stages"][s] = stage_flat41(dev, a.flat41_K)
        else:
            raise SystemExit("unknown stage %s" % s)
        res["stages"][s]["seconds"] = time.time() - t0
    res["pass_all"] = all(bool(v.get("pass", False)) for v in res["stages"].values())
    with open(os.path.join(a.out, "w1_selftest_v1.json"), "w") as f:
        json.dump(res, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)), indent=1)
    L = ["# W1 self-test v1d (page_history_v6o, page_d13_readout_v1d, page_w1_sector_v1d): %s" % ("ALL GATES PASS" if res["pass_all"] else "A GATE FAILS"), "", "| stage | pass | seconds | note |", "|---|---|---|---|"]
    for s, v in res["stages"].items():
        note = ""
        if s == "flat":
            note = "; ".join("%g r_s %.6f (X3 %.5f, diff %.1e)" % (x, v["values"][x], v["gates"][x], v["diffs"][x]) for x in sorted(v["gates"])) + "; 30 r_s %.6f (the relabel); the gate %g" % (v["values"][30.0], v["tol"])   # v1d
            if v.get("full_route"):                                                                                             # v1d
                fr = v["full_route"]
                note += "; the dense route beside: %s; the routes' difference max %.1e against the derived allowance: %s" % (
                    ", ".join("%g r_s %.6f (diff %.1e)" % (x, fr["values"][x], fr["diffs"][x]) for x in sorted(fr["diffs"])), v["full_vs_sector_max"],
                    "within" if v["route_within_allowance"] else "ABOVE (a WARNING, not a verdict)")
        elif s == "oracle":
            note = "; ".join("%s %s (within floor %s; premise failures %d)" % (k_, "pass" if r["pass"] else "FAIL", "yes" if r.get("S_within_floor") else "no", r.get("floor_premise_failures", 0)) for k_, r in v["runs"].items())   # v1d
        elif s == "sector":
            note = "nu exact routes %.2e; Gram defect %.2e within %.2e (%s)" % (v["nu_diff"], v["gram_sector_defect"], v["gram_sector_bound"], "yes" if v["gram_sector_within_bound"] else "no")
        elif s == "held":
            parts = []                                                                                                           # v1d: within or OUTSIDE by the verdict, the construction, the scaling's two ratios
            for k_, r in v["channels"].items():
                rp = r["production"]; sc_ = r["scaling"]; wp = rp["worst"]; okp = rp["ok"]
                parts.append("kappa %s: excess %.2e against its floor %.2e (ratio %.2f, %s), impurity %.2e against the a priori bound (worst ratio %.2f, %s; at the first checkpoint %s against the construction allowance %s, %s), dE_inv %.2e (ratio %.2f, %s); "
                             "control run %s; scaling: excess ratio %.1f (gate 10, %s), impurity ratio %.1f (information; truncation bounds' ratio %s; K + 1 %s at the production, %s at the control)" % (
                                 k_, wp["excess"], wp["floor_low"], wp["excess_over_floor"], "within" if okp["floor"] else "OUTSIDE", wp["T_meas"], wp["T_meas_over_cheb"], "within" if okp["premise"] else "OUTSIDE",
                                 ("%.2e" % wp["T_init"]) if wp.get("T_init") is not None else "-", ("%.2e" % wp["construction_allowance"]) if wp.get("construction_allowance") is not None else "-", "within" if okp["construction"] else "OUTSIDE",
                                 wp["dE_inv"], wp["dE_over_tol"], "within" if okp["E_inv"] else "OUTSIDE", "pass" if r["control"]["pass"] else "FAIL", sc_["excess_ratio"], "pass" if sc_["ok"] else "FAIL", sc_["impurity_ratio"],
                                 ("%.0f" % sc_["truncation_bound_ratio"]) if sc_.get("truncation_bound_ratio") is not None else "-", sc_["K_plus_1_production"], sc_["K_plus_1_control"]))
            note = "; ".join(parts)
        elif s == "flat41":
            note = "; ".join("cut %g: retired passes %s, new raw %s, new excess %s" % (x, r["retired_rule_passes"], r["new_rule_raw"]["converged"], r["new_rule_excess"]["converged"]) for x, r in v["per_cut"].items())
        L.append("| %s | %s | %.0f | %s |" % (s, "yes" if v.get("pass") else "NO", v.get("seconds", 0.0), note))
    if "held" in res["stages"]:
        L += ["", res["stages"]["held"]["rule"]]
    if "flat" in res["stages"]:
        L += ["", res["stages"]["flat"]["gate_label"], "", res["stages"]["flat"]["relabel_text"]]                               # v1d
    with open(os.path.join(a.out, "w1_selftest_v1.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("done: %s (%s)" % (os.path.join(a.out, "w1_selftest_v1.md"), "ALL GATES PASS" if res["pass_all"] else "A GATE FAILS"))
    sys.exit(0 if res["pass_all"] else 1)


if __name__ == "__main__":
    main()
