#!/usr/bin/env python
"""page_history_oracle_v6d.py (builder analyst, 2026-10-16): page_history_oracle_v6c.py on page_history_v6d.py, with a fifth run: the
fixture with channels 1 and 2 and --transfer-at-peak (the plain driver, no dense comparison), which must reach the last record (the case
that raised KeyError in v6c: a ledger restart with no partner mode in one channel); its ledger starts and the per-channel ledgers printed.
Results in history_oracle_v6d_results.json.

v6c's docstring: page_history_oracle_v6c.py (builder analyst, 2026-10-16): page_history_oracle_v6b.py on page_history_v6c.py. The frozen control runs
v6c's eigenbasis propagation (--frozen-prop eig): the oracle carries its dense covariance and archive record by record with its own per-record
dense propagator over each checkpoint interval (independent of the code's single phase step from the freeze clock) and checks the interval's
collar flow against the sum of its per-record dense flows; a fourth run, the same frozen control with --frozen-prop cheb (v6b's path), is
compared with it; the main run adds --transfer-at-peak (the ledger's identity residual printed). Results in history_oracle_v6c_results.json.

v6b's docstring: page_history_oracle_v6b.py (builder analyst, 2026-10-16): page_history_oracle_v6.py on page_history_v6b.py (inbox #41 D15: no mutual
information; v6's slab check is removed with the slabs, the run's --mi-at-rs 6 becomes --extra-ck-rs 6, the same checkpoint schedule).
Results in history_oracle_v6b_results.json.

v6's docstring: page_history_oracle_v6.py (builder analyst, 2026-10-16; inbox #36 E): the dense oracle of page_history_v6 (Computation A) on a small
time-dependent fixture with a trapping zero, and the piecewise-constant against cubic fields control on one window.

The builder's fixture: r_s/a 12, G/a^2 1/4 (E a 24), massless, kappa 1, a 256-site chain (n2 512), the prescribed moving profile N = 1,
v(r, t) = the l = 1 average (ell 2: ell/2GE = 1/6, below 0.26) of the dust ball's exterior-plus-ball flow with the ball's edge R_b(t) falling
by the marginally bound law from 3 r_s to 0.5 r_s and then held (page_ops_v6.dust_radius), stored as records every tau = 2 lattice units over
t in [0, 12 r_s] (72 records) on the engine's first 160 sites, the vacuum exterior at the same E beyond (the join exact by construction);
R_cut 2.5 r_s (30 sites), the collar 1.5 r_s (18 sites), collection every 1 r_s, the exterior shells every 4 r_s, eps_loc 1e-2.

The oracle propagates the full covariance densely, C <- U_k C U_k^dagger with U_k = V_k exp(-i e_k tau) V_k^dagger from each record's dense
eigendecomposition (independent of the code's Chebyshev series), every collected mode of the code propagated densely from its collection clock,
and checks at every checkpoint: ||Phi Phi^dagger - C||_F (the code's state); S_esc of the code against the dense tr s(F^dagger C F) with the
densely propagated modes (criterion 1e-10); E_inv against the dense Tr[h(C - P_-)]; the code's collar flow per record against the dense
Re tr[chi h_k (C(T_k + tau) - C(T_k))]; the energy identity and the collar identity (E_K, E_F, E_cross, M_chi) against their dense values;
Tr[h_dot (C - P_-)] against its dense value; the all-history Gram of the code's archive (criterion 1e-10); the energy identity's residual
(criterion 1e-6 of |E_inv| per checkpoint). Then the control: the same run with the cubic interpolation of the fields with 1-unit sub-steps on
the window [3, 7] r_s (the formation), its S_esc and collar flow at the window's end against the piecewise-constant run's. Prints PASS or FAIL
per check and writes a json. Usage: python page_history_oracle_v6d.py --out DIR [--device cpu --threads 16]"""
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
import page_history_v6d as PH        # noqa: E402

CD = OPS.CD
RD = OPS.RD


def synthetic_history(rs=12.0, g=0.25, ell=2.0, ne=160, nsites=256, tau=2.0, t_end_rs=12.0, Rb0_rs=3.0, Rbf_rs=0.5):
    E = rs / (2.0 * g)
    R = np.arange(nsites, dtype=float) + 1.0
    av = OPS.FlowAverager(R, ell, "averaged")
    T = np.arange(0.0, t_end_rs * rs + 1e-9, tau)
    Rb0, Rbf = Rb0_rs * rs, Rbf_rs * rs
    Nl, vl = [], []
    for t in T:
        Rb = max(OPS.dust_radius(t, Rb0, rs), Rbf)
        v = np.asarray(av(OPS.flow_dense(rs, Rb)), float)
        Nl.append(np.ones(ne)); vl.append(v[:ne])
    npz = {"R": R[:ne], "T": T, "N_applied": np.array(Nl), "v_applied": np.array(vl), "N": np.array(Nl), "v": np.array(vl), "tau": tau, "ell": ell,
           "label": "synthetic moving dust-ball profile (the oracle's fixture)"}
    js = {"args": {"g": g, "rs": rs}, "records": [{"T": float(t), "E": E, "kappa_T": None} for t in T]}
    return {"npz": npz, "json": js}


class OracleRun(PH.HistRun):
    """HistRun with the dense covariance carried beside it and the comparisons at every checkpoint."""

    def start(self):
        super().start()
        c = self.chs[0]
        N, v = self.H.fields(self.k0)
        e, V = OPS.eigh_csr(c.ops.h_csr(N, v), self.dev)
        P = V[:, e < 0]
        self.C = P @ P.mH
        self.F_dense = []                                                     # (dense-propagated column, t_collected)
        self.worst = {"state_F": 0.0, "S_esc": 0.0, "E_inv": 0.0, "collar_flow": 0.0, "E_K": 0.0, "E_F": 0.0, "E_cross": 0.0, "M_chi": 0.0, "W_hdot": 0.0,
                      "energy_identity_rel": 0.0, "gram": 0.0, "mode_propagation": 0.0}
        self.flows = []

    def step_record(self, k, t0=None, t1=None):
        H = self.H; c = self.chs[0]
        tt0, tt1 = t0, t1
        if t0 is None:
            t0 = H.T[k]; t1 = H.T[k + 1] if k + 1 < len(H) else t0 + self.tau
        sub = [(t0, t1, None)]
        if self.cw is not None and self.cw[0] <= t0 < self.cw[1]:
            nsub = max(1, int(round(t1 - t0)))
            sub = [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic") for i in range(nsub)]
        flow_d = 0.0
        for (sa, sb, mode) in sub:
            N, v = H.cubic_fields(0.5 * (sa + sb)) if mode == "cubic" else H.fields(k)
            hd = OPS.dense_from_csr(c.ops.h_csr(N, v), self.dev)
            e, V = torch.linalg.eigh(hd)
            U = V @ (torch.exp(-1j * e.to(CD) * (sb - sa))[:, None] * V.mH)
            chi = (c.ops.R2 > self.R_collar_abs).to(CD)
            Ea = float(torch.trace((chi[:, None] * hd) @ self.C).real)
            self.C = U @ self.C @ U.mH
            Eb = float(torch.trace((chi[:, None] * hd) @ self.C).real)
            flow_d += Eb - Ea
            self.F_dense = [(U @ q, tq) for q, tq in self.F_dense]
        super().step_record(k, tt0, tt1)
        flow_c = self.records[-1]["per_channel"][str(c.ops.kappa)]["collar_flow"]
        self.worst["collar_flow"] = max(self.worst["collar_flow"], abs(flow_c - flow_d))
        self.flows.append((t0, flow_c, flow_d))

    def frozen_materialize(self, kf, ta, tb):
        """v6c: the dense covariance and archive carried over [ta, tb] record by record with the oracle's own dense propagator of the frozen h."""
        H = self.H; c = self.chs[0]
        N, v = H.fields(kf)
        hd = OPS.dense_from_csr(c.ops.h_csr(N, v), self.dev)
        e, V = torch.linalg.eigh(hd)
        nst = int(round((tb - ta) / self.tau))
        U = V @ (torch.exp(-1j * e.to(CD) * self.tau)[:, None] * V.mH)
        chi = (c.ops.R2 > self.R_collar_abs).to(CD)
        flow_d = 0.0
        for _ in range(nst):
            Ea = float(torch.trace((chi[:, None] * hd) @ self.C).real)
            self.C = U @ self.C @ U.mH
            Eb = float(torch.trace((chi[:, None] * hd) @ self.C).real)
            flow_d += Eb - Ea
            self.F_dense = [(U @ q, tq) for q, tq in self.F_dense]
        super().frozen_materialize(kf, ta, tb)
        flow_c = self.records[-1]["per_channel"][str(c.ops.kappa)]["collar_flow"]
        self.worst["collar_flow"] = max(self.worst["collar_flow"], abs(flow_c - flow_d))
        self.flows.append((ta, flow_c, flow_d))

    def checkpoint(self, k, t=None, collect_override=None):
        c = self.chs[0]
        n_before = int(c.F.shape[1])
        super().checkpoint(k, t=t, collect_override=collect_override)
        row = self.rows[-1]; pc = row["per_channel"][str(c.ops.kappa)]
        H = self.H
        N, v = H.fields(k)
        hd = OPS.dense_from_csr(c.ops.h_csr(N, v), self.dev)
        e, V = torch.linalg.eigh(hd)
        P = V[:, e < 0] @ V[:, e < 0].mH
        Dl = self.C - P
        # the code's state against the dense one
        Cc = c.Phi @ c.Phi.mH
        self.worst["state_F"] = max(self.worst["state_F"], float(torch.linalg.norm(Cc - self.C)))
        # the new modes enter the dense archive at their collection clock (the code's columns at that clock)
        if c.F.shape[1] > n_before:
            for j in range(n_before, int(c.F.shape[1])):
                self.F_dense.append((c.F[:, j: j + 1].clone(), float(H.T[k]) if t is None else float(t)))
        if self.F_dense:
            Fd = torch.cat([q for q, _ in self.F_dense], dim=1)
            self.worst["mode_propagation"] = max(self.worst["mode_propagation"], float(torch.linalg.norm(Fd - c.F)))
            Sd = float(OPS.binary_entropy_t(torch.linalg.eigvalsh(0.5 * ((Fd.mH @ self.C @ Fd) + (Fd.mH @ self.C @ Fd).mH))).sum())
            self.worst["S_esc"] = max(self.worst["S_esc"], abs(Sd - pc["S_esc"]))
            K = torch.eye(c.ops.n2, dtype=CD, device=self.dev) - Fd @ Fd.mH
            E_K = float(torch.trace(hd @ K @ Dl @ K).real)
            FF = Fd @ Fd.mH
            E_F = float(torch.trace(hd @ FF @ Dl @ FF).real)
            E_x = float(torch.trace(hd @ (FF @ Dl @ K + K @ Dl @ FF)).real)
            g_ = Fd.mH @ Fd
            self.worst["gram"] = max(self.worst["gram"], float(torch.linalg.norm(g_ - torch.eye(g_.shape[0], dtype=CD, device=self.dev))))
        else:
            E_K = float(torch.trace(hd @ Dl).real); E_F = 0.0; E_x = 0.0
        E_inv = float(torch.trace(hd @ Dl).real)
        chi = (c.ops.R2 > self.R_collar_abs).to(CD)
        E_out = float(torch.trace((chi[:, None] * hd) @ Dl).real)
        M_chi = E_out - E_inv + E_K
        kn = k + 1 if k + 1 < len(H) else k - 1
        Nn, vn = H.fields(kn)
        sg = 1.0 if kn > k else -1.0
        hdot = OPS.dense_from_csr(c.ops.hdot_csr(sg * (Nn - N), sg * (vn - v), self.tau), self.dev)
        W_d = 0.0 if self.frozen else float(torch.trace(hdot @ Dl).real)        # C: h frozen, h_dot = 0
        for key, dense in (("E_inv", E_inv), ("E_K", E_K), ("E_F", E_F), ("E_cross", E_x), ("M_chi", M_chi), ("W_hdot", W_d)):
            code = pc.get(key)
            if code is not None:
                self.worst[key] = max(self.worst[key], abs(code - dense))          # absolute (E_inv starts at zero on the ground state)
        self.worst["energy_identity_rel"] = max(self.worst["energy_identity_rel"], abs(pc["energy_identity_residual"]) / max(abs(pc["E_inv"]), 1e-300))


def run_case(out, device, threads, cubic=None, tag="oracle", freeze=None, extra=None):
    hist = synthetic_history()
    Hs = PH.History(None, None, 256, synthetic=hist)
    argv = ["--nsites", "256", "--channels", "1", "--R-cut-rs", "2.5", "--R-collar-rs", "1.5", "--collect-every-rs", "1", "--ext-every-rs", "4",
            "--eps-loc", "1e-2", "--extra-ck-rs", "6", "--join-rs", "8,13", "--device", device, "--threads", str(threads), "--out", out, "--tag", tag,
            "--chunk", "512"] + (["--cubic-window", cubic] if cubic else []) + (["--freeze-at-rs", str(freeze[0]), "--freeze-duration-rs", str(freeze[1])] if freeze else []) + (extra or [])
    a = PH.parse(argv)
    R = OracleRun(a, history=Hs)
    R.run()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = run_case(a.out, a.device, a.threads, extra=["--transfer-at-peak"])
    w = R.worst
    last = R.rows[-1]
    n_coll = int(R.chs[0].F.shape[1])
    crit = {"S_esc": 1e-10, "gram": 1e-10, "energy_identity_rel": 1e-6, "state_F": 1e-10, "mode_propagation": 1e-10, "collar_flow": 1e-10,
            "E_inv": 1e-10, "E_K": 1e-10, "E_F": 1e-10, "E_cross": 1e-10, "M_chi": 1e-10, "W_hdot": 1e-10}
    res = {k_: {"worst": w[k_], "criterion": crit[k_], "pass": bool(w[k_] <= crit[k_])} for k_ in crit}
    ok = all(v_["pass"] for v_ in res.values()) and n_coll > 0 and R.rows and any(r_["per_channel"]["1"].get("r_t") for r_ in R.rows)
    print("%s  the dense oracle of page_history_v6d on the moving fixture (r_s/a 12, 256 sites, ell 2, a trapping zero): %d collected modes, S_esc at the end %.10f; %s" % (
        "PASS" if ok else "FAIL", n_coll, last["S_esc_weighted"] / 16.0, json.dumps(res)), flush=True)
    Rc = run_case(a.out, a.device, a.threads, cubic="3,7", tag="oracle_cubic")
    kw = [i for i, r_ in enumerate(R.rows) if abs(r_["t_over_rs"] - 7.0) < 1e-9]
    kc = [i for i, r_ in enumerate(Rc.rows) if abs(r_["t_over_rs"] - 7.0) < 1e-9]
    ctl = None
    if kw and kc:
        r1, r2 = R.rows[kw[0]], Rc.rows[kc[0]]
        ctl = {"t_over_rs": 7.0, "S_esc_piecewise": r1["S_esc_weighted"], "S_esc_cubic": r2["S_esc_weighted"], "dS": r2["S_esc_weighted"] - r1["S_esc_weighted"],
               "collar_flow_piecewise": r1["collar_flow_acc_weighted"], "collar_flow_cubic": r2["collar_flow_acc_weighted"],
               "dflow": r2["collar_flow_acc_weighted"] - r1["collar_flow_acc_weighted"], "E_inv_piecewise": r1["E_inv_weighted"], "E_inv_cubic": r2["E_inv_weighted"],
               "cubic_oracle_worst": Rc.worst}
    print("DIAG  the piecewise-constant against cubic fields on the window [3, 7] r_s: %s" % json.dumps(ctl), flush=True)
    okc = Rc.worst["S_esc"] <= 1e-10 and Rc.worst["state_F"] <= 1e-10
    print("%s  the dense oracle on the cubic run (its sub-steps propagated densely the same way): S_esc %.3e, state %.3e" % ("PASS" if okc else "FAIL", Rc.worst["S_esc"], Rc.worst["state_F"]), flush=True)
    Rf = run_case(a.out, a.device, a.threads, tag="oracle_frozen", freeze=(6.0, 40.0), extra=["--transfer-at-peak"])
    fr = [r_ for r_ in Rf.rows if r_.get("frozen")]
    okf = Rf.worst["S_esc"] <= 1e-10 and Rf.worst["state_F"] <= 1e-10 and Rf.worst["gram"] <= 1e-10
    print("%s  the dense oracle on the frozen-record control (C's machinery; h frozen at T/r_s 6 for 40 r_s): S_esc %.3e, state %.3e, Gram %.3e; %d frozen checkpoints; the end: %s; the late |L|/L_H(T_H) over the last 10 blocks %s; S_esc peak %.6f, end %.6f" % (
        "PASS" if okf else "FAIL", Rf.worst["S_esc"], Rf.worst["state_F"], Rf.worst["gram"], len(fr), Rf.end_reason,
        ["%.2e" % (abs(r_["L_interval"]) / r_["L_H_TH"]) for r_ in fr[-10:] if r_.get("L_interval") is not None and r_.get("L_H_TH")],
        max([r_["S_esc_weighted"] for r_ in fr] or [0.0]), fr[-1]["S_esc_weighted"] if fr else float("nan")), flush=True)
    # v6c: the frozen control on v6b's per-record Chebyshev path, against the eigenbasis path
    Rq = run_case(a.out, a.device, a.threads, tag="oracle_frozen_cheb", freeze=(6.0, 40.0), extra=["--transfer-at-peak", "--frozen-prop", "cheb"])
    fq = [r_ for r_ in Rq.rows if r_.get("frozen")]
    pairs = list(zip(fr, fq))
    cmp_ = {"rows": len(pairs), "S_esc_max_abs": max((abs(x["S_esc_weighted"] - y["S_esc_weighted"]) for x, y in pairs), default=None),
            "E_inv_max_abs": max((abs(x["E_inv_weighted"] - y["E_inv_weighted"]) for x, y in pairs), default=None),
            "collar_flow_acc_max_abs": max((abs(x["collar_flow_acc_weighted"] - y["collar_flow_acc_weighted"]) for x, y in pairs), default=None),
            "collected_equal": all([p_.get("collected") for p_ in x["per_channel"].values()] == [p_.get("collected") for p_ in y["per_channel"].values()] for x, y in pairs),
            "cheb_oracle_worst": Rq.worst, "eig_reflection_clock": Rf.refl, "eig_transfer_starts": Rf.transfer_starts,
            "eig_transfer_end": {k_: tr.get("end") for k_, tr in (Rf.transfer or {}).items()}, "main_transfer_end": {k_: tr.get("end") for k_, tr in (R.transfer or {}).items()},
            "main_transfer_starts": R.transfer_starts}
    print("DIAG  the frozen control on the eigenbasis path against the Chebyshev path (the same oracle): %s" % json.dumps(cmp_, default=float), flush=True)
    # v6d: the two-channel ledger smoke run (the plain driver; the case that raised KeyError in v6c)
    hist2 = synthetic_history()
    Hs2 = PH.History(None, None, 256, synthetic=hist2)
    a2 = PH.parse(["--nsites", "256", "--channels", "1,2", "--R-cut-rs", "2.5", "--R-collar-rs", "1.5", "--collect-every-rs", "1", "--ext-every-rs", "4",
                   "--eps-loc", "1e-2", "--extra-ck-rs", "6", "--join-rs", "8,13", "--device", a.device, "--threads", str(a.threads), "--out", a.out,
                   "--tag", "oracle_two_channel_ledger", "--chunk", "512", "--transfer-at-peak"])
    R2c = PH.HistRun(a2, history=Hs2)
    err2 = None
    try:
        R2c.run()
    except Exception as ex:                                               # noqa: BLE001 (recorded as the failure)
        err2 = "%s: %s" % (type(ex).__name__, ex)
    two = {"error": err2, "rows": len(R2c.rows), "last_T_over_rs": R2c.rows[-1]["t_over_rs"] if R2c.rows else None,
           "transfer_starts": R2c.transfer_starts, "ledger_channels_at_end": sorted((R2c.transfer or {}).keys()),
           "transfer_end": {k_: tr.get("end") for k_, tr in (R2c.transfer or {}).items()}}
    ok2 = err2 is None and bool(R2c.rows) and abs(R2c.rows[-1]["t_over_rs"] - 12.0) < 1e-9
    print("%s  the two-channel ledger run (channels 1, 2, --transfer-at-peak, to the last record): %s" % ("PASS" if ok2 else "FAIL", json.dumps(two, default=float)), flush=True)
    out = {"code": "page_history_oracle_v6d.py", "two_channel_ledger": dict(two, ok=ok2), "pass": bool(ok and okc and okf), "checks": res, "n_collected": n_coll, "rows": R.rows, "flows": R.flows, "cubic_control": ctl,
           "frozen_eig_vs_cheb": cmp_,
           "frozen_control": {"worst": Rf.worst, "end_reason": Rf.end_reason, "rows": fr, "flows": Rf.flows}, "seconds": time.time() - t0}
    with open(os.path.join(a.out, "history_oracle_v6d_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("ORACLE %s in %.0f s" % ("PASS" if out["pass"] else "FAIL", time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
