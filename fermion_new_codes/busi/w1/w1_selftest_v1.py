#!/usr/bin/env python
"""w1_selftest_v1.py (reviewer A, the busi lane, W1): the gates before any entropy number of page_history_v6o prints. Stages (--stages, default all):
  pairing : a 128-site chain (kappa 2, random smooth N, v): the operator audit (max|h + conj h|, max|[S_stag, h]| = 0), the dense spectrum's pairing
            Tr min(h, 0) = -Tr|h|/2, spec(h_-) = -spec(h_+); the negative control m = 0.2 with the flow (the audit fails, the residual printed).
  sector  : the sector vacuum against the dense one on the fixture's record with a trapping zero (the projector difference, E_ref, the exterior
            entropies of a state and of the vacuum, t_X, ||C - P_-||_1 by both routes).
  oracle  : the dense oracle fixture with a trapping zero (the marginally bound dust flow, 192 sites, r_s 16, tau 2, 61 records; the zero at R = r_s
            from t about 75): the full 384 x 384 covariance propagated by the exact dense exponential of every sub-step against page_history_v6o's
            column recursion in 3 representations (the + sector propagation; the full propagation with the halved entropies; v6n's full route) for
            the production (--interp linear) and the control (--interp none): S_X raw, the held vacuum's S_X, the excess, E_inv, E_kappa =
            Tr[|h|(C - P)^2] (the identity with E_inv), t_X, ||C - P_-||_1, the accumulated collar flow, the probes' back-propagation check.
  held    : a held record with a trapping zero (the ball's edge fixed at 0.5 r_s): the excess 0 to 1e-10 per channel at every checkpoint and cut,
            E_inv constant (the quench tail absent, defect 11's gate), for kappa 1 and 2.
  degree  : the Chebyshev degree of the record step against the sub-steps (the cost statement), from the fixture's records and from cheb_order.
  flat    : the flat held vacuum (N = 1, v = 0) at kappa 1 on the 8192-site chain: the cut entropy at 2, 16, 24 and 48 r_s against 2.0544, 2.7464,
            2.8815 and 3.1109 nats (tolerance 1e-4); the 30 r_s value for the relabel; --flat-full-check runs v1b's dense route beside.
  flat41  : the flat control at K = 41 (kappa 1 to 41 on the 8192-site chain): the retired rule passes, the new rule on the raw entropy fails (no
            majorant), on the excess converges to 0 (defect 10's gate).
Writes <out>/w1_selftest_v1.json and .md with PASS/FAIL per gate and the relabel text (defect 12). CPU suffices (the flat stages take minutes to an
hour on 16 threads); --device cuda:0 where a GPU is free. Usage: PYTHONPATH=<code_v6>:<code_d13> python w1_selftest_v1.py --out <dir> [--stages ...]"""
import argparse
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
import page_w1_sector_v1 as W1S  # noqa: E402
import page_d13_readout_v1c as D13  # noqa: E402
import page_history_v6o as V6O  # noqa: E402

T0 = time.time()
GATES_FLAT = {2.0: 2.0544, 16.0: 2.7464, 24.0: 2.8815, 48.0: 3.1109}


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T0, msg), flush=True)


def ent_dense(M):
    ev = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
    return OPS.binary_entropy_np(ev)


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


def run_fixture(D, Z, out, tag, interp, sector, sector_prop, device, threads, nsub=8, T_end_rs=7.5, extra=()):
    argv = ["--out", out, "--tag", tag, "--nsites", str(len(Z["R"])), "--g", str(D["args"]["g"]), "--rs-nominal", str(D["args"]["rs"]),
            "--channels", "1", "--kappas", "1", "--R-cut-rs", "6", "--R-collar-rs", "4", "--join-rs", "8,11", "--collect-every-rs", "2", "--extra-ck-rs", "",
            "--ext-every-rs", "4", "--vac-every-rs", "4", "--T-end-rs", str(T_end_rs), "--cuts-rs", "2,4,6,9", "--ledger-cuts-rs", "4,6", "--probe-cuts-rs", "4,6",
            "--TH-ref-rs", "6", "--form-rs", "5", "--probe-from-rs", "6", "--probe-every-rs", "2", "--ref-clocks-rs", "4", "--columns-at-rs", "4", "--columns-every-rs", "0",
            "--device", str(device), "--threads", str(threads), "--interp", interp,   # busi (analyst): str(device), argparse takes strings "--interp-sub", str(nsub), "--sector", sector, "--sector-propagate", sector_prop,
            "--sector-check-every-rs", "0", "--save-every", "1000", "--max-wall-hours", "4", "--kappa-hard-max", "1", "--no-kappa-extend"] + list(extra)
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
    rs = 16.0; cuts = [("2", 32.0), ("4", 64.0), ("6", 96.0), ("9", 144.0)]
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
    W1S.cross_cut_readout_sector(Phi, vs, H.R, cuts, ent_sec, {"cut_bound_tol": 1e-6, "cut_norm_full": "auto"}, 1, 0.0, sc, arrays)
    tn_full = D13.full_trace_norm(Phi, vd["Vneg"], "svd")
    worst = 0.0
    for lab, _ in cuts:
        d = {"S_X_full": ent_full[lab]["S_X"], "S_X_sector": ent_sec[lab]["S_X"], "S_vac_full": ent_full[lab]["S_X_vac"], "S_vac_sector": ent_sec[lab]["S_X_vac"],
             "t_X_full": cc[lab]["t_X"], "t_X_sector": ent_sec[lab]["t_X"], "pairing_defect_state": ent_sec[lab].get("pairing_defect_state"), "pairing_defect_vacuum": ent_sec[lab].get("pairing_defect_vacuum")}
        for a_, b_ in (("S_X_full", "S_X_sector"), ("S_vac_full", "S_vac_sector"), ("t_X_full", "t_X_sector")):
            worst = max(worst, abs(d[a_] - d[b_]))
        worst = max(worst, d["pairing_defect_state"], d["pairing_defect_vacuum"])
        out["cuts"][lab] = d
    out["nu_full"] = tn_full["nu"]; out["nu_sector"] = sc["trace_norm"]["nu"]; worst_nu = abs(tn_full["nu"] - sc["trace_norm"]["nu"])
    out["worst_entropy_diff"] = worst; out["nu_diff"] = worst_nu
    out["pass"] = bool(out["proj_diff_F"] < 1e-10 and out["E_ref_diff"] < 1e-9 and worst < 1e-9 and worst_nu < 1e-6)
    log("sector: projector difference %.2e, E_ref difference %.2e, worst entropy/t_X/pairing difference %.2e, nu difference %.2e: %s" % (out["proj_diff_F"], out["E_ref_diff"], worst, worst_nu, "PASS" if out["pass"] else "FAIL"))
    return out


def stage_oracle(dev, threads, tmp):
    D, Z = dust_fixture()
    H = V6O.History(None, None, 192, 0.25, "applied", synthetic={"json": D, "npz": Z})
    cuts = [("2", 32.0), ("4", 64.0), ("6", 96.0), ("9", 144.0)]
    out = {"pass": True, "runs": {}}
    tol = {"S": 1e-8, "E": 1e-9, "flow": 1e-9, "t": 1e-8, "nu": 1e-6, "probe": 1e-9, "identity": 1e-10}
    for interp in ("linear", "none"):
        oracle = None
        for (sector, sprop) in (("on", "on"), ("on", "off"), ("off", "off")):
            tag = "fx_%s_s%s_p%s" % (interp, sector, sprop)
            run, S, Dj = run_fixture(D, Z, tmp, tag, interp, sector, sprop, dev, threads)
            ck = sorted(run.ck)
            if oracle is None:
                oracle = dense_oracle(H, 1, ck, interp, 8, cuts, 64.0)
            worst = {k_: 0.0 for k_ in ("S", "Svac", "excess", "E", "flow", "t", "nu", "identity", "probe", "Ekappa_spectral")}
            for r in S["rows"]:
                k = r["k"]; pc = r["per_channel"]["1"]; o = oracle[k]
                worst["E"] = max(worst["E"], abs(pc["E_inv"] - o["E_inv"]))
                worst["flow"] = max(worst["flow"], abs(pc["collar_flow_acc"] - o["collar_flow_acc"]))
                worst["identity"] = max(worst["identity"], abs(o["E_kappa"] - o["E_inv"]))
                fe = ((pc.get("w1") or {}).get("E_kappa_spectral") or {})
                if fe.get("E_kappa") is not None:
                    worst["Ekappa_spectral"] = max(worst["Ekappa_spectral"], abs(fe["E_kappa"] - o["E_inv"]))
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
                  and worst["nu"] < tol["nu"] and worst["identity"] < tol["identity"] and worst["probe"] < tol["probe"] and worst["Ekappa_spectral"] < tol["E"])
            Ks = [q["per_channel"]["1"].get("cheb_K_sum") for q in S["records"]]
            out["runs"][tag] = {"worst": worst, "pass": bool(ok), "cheb_K_sum_mean_per_record": float(np.mean(Ks)) if Ks else None, "E_inv_end": S["rows"][-1]["per_channel"]["1"]["E_inv"],
                                "sector_used": S.get("w1", {}).get("sector_used"), "wall_s": S.get("wall_s")}
            out["pass"] = out["pass"] and bool(ok)
            log("oracle %s: %s; worst differences %s" % (tag, "PASS" if ok else "FAIL", ", ".join("%s %.1e" % (k_, v_) for k_, v_ in worst.items())))
    e_lin = out["runs"]["fx_linear_son_pon"]["E_inv_end"]; e_none = out["runs"]["fx_none_son_pon"]["E_inv_end"]
    out["E_inv_end_production_vs_control"] = {"linear": e_lin, "none": e_none, "difference_none_minus_linear": e_none - e_lin}
    log("oracle: E_inv at the end, production (linear) %.6e against the control (none) %.6e: the control's excess %.3e" % (e_lin, e_none, e_none - e_lin))
    return out


def stage_held(dev, threads, tmp):
    out = {"pass": True, "channels": {}}
    for kappa in (1, 2):
        D, Z = dust_fixture(nrec=33, held_Rb_over_rs=0.5, label="W1 held fixture: the dust flow held with the edge at 0.5 r_s (a trapping zero at R = r_s)")
        tag = "held_k%d" % kappa
        run, S, Dj = run_fixture(D, Z, tmp, tag, "linear", "on", "on", dev, threads, T_end_rs=4.0, extra=["--channels", str(kappa), "--kappas", str(kappa)])
        worst_exc = 0.0; E0 = S["rows"][0]["per_channel"][str(kappa)]["E_inv"]; worst_E = 0.0
        for r in S["rows"]:
            pc = r["per_channel"][str(kappa)]; worst_E = max(worst_E, abs(pc["E_inv"] - E0))
            cl = Dj["clocks"][D13.Store.ckey(r["t_over_rs"])]["per_channel"][str(kappa)]
            for lab, c in cl["cuts"].items():
                if c.get("excess") is not None:
                    worst_exc = max(worst_exc, abs(c["excess"]))
        ok = worst_exc <= 1e-10 and worst_E <= 1e-10
        out["channels"][kappa] = {"worst_excess": worst_exc, "E_inv_T0": E0, "worst_E_inv_change": worst_E, "pass": bool(ok)}
        out["pass"] = out["pass"] and bool(ok)
        log("held kappa %d: worst |excess| %.2e, worst |E_inv(t) - E_inv(0)| %.2e (the quench tail absent): %s" % (kappa, worst_exc, worst_E, "PASS" if ok else "FAIL"))
    return out


def stage_degree():
    out = {"rows": []}
    for kappa in (1, 4, 8, 16):
        w = kappa + 1.5
        Kf, _ = OPS.cheb_order(w * 8.0, 1e-14)
        row = {"kappa": kappa, "bound": w, "K_record_step": Kf}
        for n in (2, 4, 8):
            Ks, _ = OPS.cheb_order(w * 8.0 / n, 1e-14)
            row["K_sum_%d_substeps" % n] = n * Ks; row["ratio_%d" % n] = n * Ks / Kf
        out["rows"].append(row)
        log("degree kappa %d (bound %.1f): record step K %d; sub-steps 2/4/8: %d/%d/%d (ratios %.2f/%.2f/%.2f)" % (
            kappa, w, Kf, row["K_sum_2_substeps"], row["K_sum_4_substeps"], row["K_sum_8_substeps"], row["ratio_2"], row["ratio_4"], row["ratio_8"]))
    out["note"] = "the propagation's degree per record for --interp linear with 8 sub-steps is 1.5 (kappa 16) to 4 (kappa 1) times the record step's; the + sector propagation applies each term at about 1/5 of the full apply's memory traffic"
    out["pass"] = True
    return out


def flat_vacuum_entropies(kappa, nsites, rs, cuts_rs, dev):
    R = np.arange(nsites, dtype=float) + 1.0
    h0, Q = OPS.chain_sparse(R, 0.0, float(kappa))
    h = OPS.clocked_csr(h0, Q, np.ones(nsites), np.zeros(nsites))
    vac = W1S.killing_vacuum_sector(h, dev)
    ep = torch.as_tensor(vac["e_plus"], dtype=OPS.RD, device=dev)
    Vp = vac["V_plus"][:, ep < 0]
    out = {}
    for x in cuts_rs:
        mask = torch.as_tensor(R > x * rs, dtype=torch.bool, device=dev)
        ev, _ = W1S.gram_eigs(Vp[~mask])
        out[float(x)] = 2.0 * W1S.entropy_of_eigs(ev)
    del Vp, vac
    return out, h


def stage_flat(dev, full_check):
    t0 = time.time()
    vals, h = flat_vacuum_entropies(1, 8192, 64.0, (2.0, 16.0, 24.0, 30.0, 48.0), dev)
    out = {"values": vals, "gates": GATES_FLAT, "diffs": {}, "pass": True, "seconds_sector": time.time() - t0}
    for x, g in GATES_FLAT.items():
        d = abs(vals[x] - g); out["diffs"][x] = d
        out["pass"] = out["pass"] and d <= 1e-4
    out["relabel_text"] = W1S.relabel_text(vals)
    log("flat held vacuum kappa 1 (8192 sites, r_s 64): %s; gates %s; %s" % (", ".join("%g r_s %.6f" % (x, vals[x]) for x in sorted(vals)), GATES_FLAT, "PASS" if out["pass"] else "FAIL"))
    if full_check:
        t1 = time.time()
        vd = D13.killing_vacuum(h, dev, "dense")
        R2 = torch.as_tensor(np.repeat(np.arange(8192, dtype=float) + 1.0, 2), dtype=OPS.RD, device=dev)
        ent = D13.exterior_entropies(vd["Vneg"], R2, [("%g" % x, x * 64.0) for x in (2.0, 16.0, 24.0, 30.0, 48.0)])
        out["full_route"] = {x: ent["%g" % x]["S_X"] for x in (2.0, 16.0, 24.0, 30.0, 48.0)}
        out["full_vs_sector_max"] = max(abs(out["full_route"][x] - vals[x]) for x in vals)
        out["seconds_full"] = time.time() - t1
        out["pass"] = out["pass"] and out["full_vs_sector_max"] < 1e-9
        log("flat: v1b's dense route beside: max difference %.2e (%.0f s)" % (out["full_vs_sector_max"], out["seconds_full"]))
    return out


def stage_flat41(dev, K=41):
    raw = {x: [] for x in (2.0, 16.0, 24.0, 48.0)}
    for kappa in range(1, K + 1):
        vals, _ = flat_vacuum_entropies(kappa, 8192, 64.0, (2.0, 16.0, 24.0, 48.0), dev)
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
    ap.add_argument("--flat-full-check", dest="flat_full", action="store_true", help="flat: run v1b's dense route beside (16384 x 16384 eigh)")
    ap.add_argument("--flat41-K", dest="flat41_K", type=int, default=41)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "fixture_runs"); os.makedirs(tmp, exist_ok=True)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    res = {"device": str(dev), "stages": {}, "code": {"v6o": V6O.__file__, "v1c": D13.__file__, "sector": W1S.__file__}}
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
    L = ["# W1 self-test (page_history_v6o, page_d13_readout_v1c, page_w1_sector_v1): %s" % ("ALL GATES PASS" if res["pass_all"] else "A GATE FAILS"), "", "| stage | pass | seconds | note |", "|---|---|---|---|"]
    for s, v in res["stages"].items():
        note = ""
        if s == "flat":
            note = "; ".join("%g r_s %.6f (gate %s)" % (x, v["values"][x], v["gates"].get(x, "-")) for x in sorted(v["values"]))
        elif s == "oracle":
            note = "; ".join("%s %s" % (k_, "pass" if r["pass"] else "FAIL") for k_, r in v["runs"].items())
        elif s == "flat41":
            note = "; ".join("cut %g: retired passes %s, new raw %s, new excess %s" % (x, r["retired_rule_passes"], r["new_rule_raw"]["converged"], r["new_rule_excess"]["converged"]) for x, r in v["per_cut"].items())
        L.append("| %s | %s | %.0f | %s |" % (s, "yes" if v.get("pass") else "NO", v.get("seconds", 0.0), note))
    if "flat" in res["stages"]:
        L += ["", res["stages"]["flat"]["relabel_text"]]
    with open(os.path.join(a.out, "w1_selftest_v1.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("done: %s (%s)" % (os.path.join(a.out, "w1_selftest_v1.md"), "ALL GATES PASS" if res["pass_all"] else "A GATE FAILS"))
    sys.exit(0 if res["pass_all"] else 1)


if __name__ == "__main__":
    main()
