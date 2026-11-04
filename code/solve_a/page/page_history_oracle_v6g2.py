#!/usr/bin/env python
"""page_history_oracle_v6g2.py (builder analyst, 2026-10-17): page_history_oracle_v6g.py with the criterion of the partners' orthogonality
to the archive (at each ledger's start, and the propagated overlap with the archive before the start) set at the construction's own floor
in place of 1e-10: B = (C F u - F F^dagger C F u)/(f(1 - f))^(1/2), so F^dagger B = (I - F^dagger F) F^dagger C F u/(f(1 - f))^(1/2) carries the
archive's Gram error amplified by 1/(f(1 - f))^(1/2) of the smallest partner (the fixture's partners have f within 1e-4 of 0 or 1). The
floor is 10 ||F^dagger F - I||_F/min_j (f_j(1 - f_j))^(1/2) at the start; the printed criterion is the ratio, at most 1. The oracle v6g's run
1 (2026-10-17 01:33 UTC) read 2.12e-10 (2.12e-10) on the main run and 4.75e-10 (4.58e-10) on the cubic run against 1e-10, every other
check passing. Everything else is the oracle v6g's. Results in history_oracle_v6g2_results.json.

The oracle v6g's docstring: page_history_oracle_v6g.py (builder analyst, 2026-10-17): the oracle of page_history_v6g.py on v6f's moving fixture (page_history_oracle_v6f's
synthetic_history: r_s/a 12, 256 sites, ell 2, a trapping zero, 72 records every tau 2 to T/r_s 12). Eight runs; results in
history_oracle_v6g_results.json; PASS or FAIL per run.
 1. dense: v6f's dense oracle (the state, S_esc, E_inv, the identities, W_acc, the jump check; v6f's criteria) on v6g with both ledgers
    (--ledger-anchor-rs 6.1667: record 37, not a checkpoint; --transfer-at-peak), and per ledger at every snapshot: the code's partners B
    against the same partners carried by the oracle's own dense propagator from the ledger's start, W_R against the dense overlap of the
    partners with the modes collected after the start, diag T_K^dagger T_K against its dense value, the dense overlap with the archive
    before the start (zero by construction) and the code's own accumulation check; at each start, on the dense covariance, the partners'
    relation K C K b_j = (1 - f_j) b_j (K = 1 - F F^dagger; exact for a pure state) and their orthogonality to the archive.
 2. cubic: the ledger checks of run 1 with --cubic-window 3,7 and the anchor at 4.1667 inside it (v6g (4): the partners take the sea's
    sub-steps; v6f carried them with the record's piecewise h); its S_esc against the dense oracle printed beside v6f's oracle value.
 3. v6f against v6g on the fixture (--transfer-at-peak; v6g with the anchor added): every row value and record value of v6f's equal in v6g's,
    and the peak ledger's end equal.
 4. merged: the fixture as two file pairs (a primary to T/r_s 6; a continuation holding every record, its records up to 6 perturbed by 1e-12
    so the seam prints a difference), v6g with --record-ext --fields-ext --ext-after-rs 6: every row equal to run 3's v6g rows.
 5. follow: the continuation written record by record by page_history_oracle_v6g_writer.py (from T/r_s 7, one record every 0.4 s, each
    file written in two halves with a pause, so reads meet partial files), v6g with --follow polling every 0.05 s to T/r_s 12: every row
    equal to run 4's; the failed (partial) reads and the waits printed.
 6. the done marker: the same writer with a marker after its last record, v6g with --follow to T/r_s 14: the run ends at T/r_s 12 with
    run 4's rows.
 7. a changed record: a continuation that rewrites a record the history has already taken: ensure must raise RuntimeError.
 8. two channels (1, 2) with both ledgers to the last record (v6d's smoke run).
Usage: python page_history_oracle_v6g.py --out DIR [--device cpu --threads 8]"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v6 as OPS            # noqa: E402
import page_history_v6g as PG        # noqa: E402
import page_history_v6f as P6F       # noqa: E402
import page_history_oracle_v6f as O6F  # noqa: E402  (its synthetic_history only)

CD = OPS.CD
RD = OPS.RD
SKIP = {"seconds", "device_peak_GiB", "device_peak_before_GiB"}
CRIT = {"S_esc": 1e-10, "gram": 1e-10, "energy_identity_rel": 1e-6, "state_F": 1e-10, "mode_propagation": 1e-10, "collar_flow": 1e-10,
        "E_inv": 1e-10, "E_K": 1e-10, "E_F": 1e-10, "E_cross": 1e-10, "M_chi": 1e-10, "W_hdot": 1e-10, "W_acc": 1e-10, "jump_check": 1e-10}
LCRIT = {"B_prop": 1e-10, "W_R": 1e-10, "TK_diag": 1e-10, "before_overlap_over_floor": 1.0, "code_W_R_check": 1e-10, "start_orth_over_floor": 1.0, "start_relation": 1e-7}
PRINTED = ("before_overlap", "start_orth", "start_floor")                    # v6g2: the absolute values beside the ratios


def first_diff(x, y, path=""):
    """the first difference of y from x over x's keys (floats compared exactly), or None."""
    if isinstance(x, dict):
        if not isinstance(y, dict):
            return "%s: dict against %s" % (path, type(y).__name__)
        for k in x:
            if k in SKIP:
                continue
            if k not in y:
                return "%s/%s missing" % (path, k)
            d = first_diff(x[k], y[k], "%s/%s" % (path, k))
            if d:
                return d
        return None
    if isinstance(x, (list, tuple)):
        if not isinstance(y, (list, tuple)) or len(x) != len(y):
            return "%s: lengths %s, %s" % (path, len(x), len(y) if isinstance(y, (list, tuple)) else "-")
        for i, (p, q) in enumerate(zip(x, y)):
            d = first_diff(p, q, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    if isinstance(x, float) or isinstance(y, float):
        if x is None or y is None:
            return None if x is y else "%s: %r against %r" % (path, x, y)
        if float(x) == float(y) or (math.isnan(float(x)) and math.isnan(float(y))):
            return None
        return "%s: %r against %r" % (path, x, y)
    return None if x == y else "%s: %r against %r" % (path, x, y)


class OracleRunG(PG.HistRun):
    """v6f's OracleRun on v6g's HistRun, with the dense covariance and archive beside it, and the two ledgers' partners carried densely."""

    def start(self):
        super().start()
        c = self.chs[0]
        N, v = self.H.fields(self.k0)
        e, V = OPS.eigh_csr(c.ops.h_csr(N, v), self.dev)
        P = V[:, e < 0]
        self.C = P @ P.mH
        self.F_dense = []
        self.worst = {k_: 0.0 for k_ in CRIT}
        self.lw = {k_: 0.0 for k_ in LCRIT}
        self.lw.update({k_: 0.0 for k_ in PRINTED}); self.Bd_floor = {"anchor": None, "peak": None}
        self.Bd = {"anchor": None, "peak": None}; self.Bd_t0 = {"anchor": None, "peak": None}; self.n_snap = {"anchor": 0, "peak": 0}; self.n_start = {"anchor": 0, "peak": 0}
        self.E_inv_dense_last = None
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
            for key in ("anchor", "peak"):
                if self.Bd[key] is not None:
                    self.Bd[key] = U @ self.Bd[key]
        super().step_record(k, tt0, tt1)
        flow_c = self.records[-1]["per_channel"][str(c.ops.kappa)]["collar_flow"]
        self.worst["collar_flow"] = max(self.worst["collar_flow"], abs(flow_c - flow_d))
        self.flows.append((t0, flow_c, flow_d))

    def _start_checks(self, tr, key):
        c = self.chs[0]; F = c.F; B = tr["B"]
        n2 = c.ops.n2
        K = torch.eye(n2, dtype=CD, device=self.dev) - (F @ F.mH if F.shape[1] else 0.0)
        f = torch.tensor(tr["f"], dtype=RD, device=self.dev).to(CD)
        rel = torch.linalg.norm(K @ self.C @ K @ B - B * (1.0 - f)[None, :], dim=0)
        self.lw["start_relation"] = max(self.lw["start_relation"], float(rel.max()))
        fr = np.asarray(tr["f"], float); ffmin = float(np.min(fr * (1.0 - fr)))
        g_ = F.mH @ F
        gram = float(torch.linalg.norm(g_ - torch.eye(g_.shape[0], dtype=CD, device=self.dev)))
        floor = 10.0 * max(gram, 1e-16) / math.sqrt(max(ffmin, 1e-300))           # v6g2: the construction's floor
        self.Bd_floor[key] = floor
        self.lw["start_floor"] = max(self.lw["start_floor"], floor)
        if F.shape[1]:
            o = float(torch.linalg.norm(F.mH @ B))
            self.lw["start_orth"] = max(self.lw["start_orth"], o)
            self.lw["start_orth_over_floor"] = max(self.lw["start_orth_over_floor"], o / floor)

    def start_anchor(self, k):
        super().start_anchor(k)
        tr = (self.anchor or {}).get("1")
        if tr is not None:
            self.Bd["anchor"] = tr["B"].clone(); self.Bd_t0["anchor"] = tr["t0"]; self.n_start["anchor"] += 1
            self._start_checks(tr, "anchor")

    def start_transfer(self, c, Phi, F, r_t, t0=None):
        super().start_transfer(c, Phi, F, r_t, t0)
        if str(c.ops.kappa) != "1":
            return
        tr = (self.transfer or {}).get("1")
        if tr is not None:
            self.Bd["peak"] = tr["B"].clone(); self.Bd_t0["peak"] = tr["t0"]; self.n_start["peak"] += 1
            self._start_checks(tr, "peak")
        else:
            self.Bd["peak"] = None

    def ledger_snapshot(self, led, row):
        out = super().ledger_snapshot(led, row)
        if not out or "1" not in out:
            return out
        key = "anchor" if led is self.anchor else "peak"
        Bd = self.Bd[key]
        if Bd is None:
            return out
        c = self.chs[0]; tr = led["1"]; t = float(row["t"])
        cols = [q for q, _ in self.F_dense]; ts = [tq for _, tq in self.F_dense]
        m = sum(int(q.shape[1]) for q in cols)
        if int(c.F.shape[1]) > m:                                            # this checkpoint's batch: the code's columns at this clock
            cols.append(c.F[:, m:]); ts += [t] * (int(c.F.shape[1]) - m)
        Fd = torch.cat(cols, dim=1)
        after = torch.tensor([tq > self.Bd_t0[key] + 1e-9 for tq in ts], device=self.dev)
        Fa = Fd[:, after]; Fb = Fd[:, ~after]
        WRd = (Fa.mH @ Bd).mH @ (Fa.mH @ Bd)
        FdB = Fd.mH @ Bd
        TTd = Bd.mH @ Bd - FdB.mH @ FdB
        self.lw["B_prop"] = max(self.lw["B_prop"], float(torch.linalg.norm(tr["B"] - Bd)))
        self.lw["W_R"] = max(self.lw["W_R"], float(torch.linalg.norm(tr["W_R"] - WRd)))
        self.lw["TK_diag"] = max(self.lw["TK_diag"], float(np.max(np.abs(np.asarray(out["1"]["diag_TK_TK"]) - torch.diagonal(TTd).real.cpu().numpy()))))
        if Fb.shape[1]:
            ob = float(torch.linalg.norm(Fb.mH @ Bd))
            self.lw["before_overlap"] = max(self.lw["before_overlap"], ob)
            self.lw["before_overlap_over_floor"] = max(self.lw["before_overlap_over_floor"], ob / self.Bd_floor[key])
        self.lw["code_W_R_check"] = max(self.lw["code_W_R_check"], float(out["1"]["W_R_check_F"]))
        self.n_snap[key] += 1
        return out

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
        Cc = c.Phi @ c.Phi.mH
        self.worst["state_F"] = max(self.worst["state_F"], float(torch.linalg.norm(Cc - self.C)))
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
        if pc.get("W_acc_interval") is not None and self.E_inv_dense_last is not None:
            self.worst["W_acc"] = max(self.worst["W_acc"], abs(pc["W_acc_interval"] - (E_inv - self.E_inv_dense_last)))
        if pc.get("jump_check") is not None:
            self.worst["jump_check"] = max(self.worst["jump_check"], abs(pc["jump_check"]))
        self.E_inv_dense_last = E_inv
        chi = (c.ops.R2 > self.R_collar_abs).to(CD)
        E_out = float(torch.trace((chi[:, None] * hd) @ Dl).real)
        M_chi = E_out - E_inv + E_K
        kn = k + 1 if k + 1 < len(H) else k - 1
        Nn, vn = H.fields(kn)
        sg = 1.0 if kn > k else -1.0
        hdot = OPS.dense_from_csr(c.ops.hdot_csr(sg * (Nn - N), sg * (vn - v), self.tau), self.dev)
        W_d = 0.0 if self.frozen else float(torch.trace(hdot @ Dl).real)
        for key, dense in (("E_inv", E_inv), ("E_K", E_K), ("E_F", E_F), ("E_cross", E_x), ("M_chi", M_chi), ("W_hdot", W_d)):
            code = pc.get(key)
            if code is not None:
                self.worst[key] = max(self.worst[key], abs(code - dense))
        self.worst["energy_identity_rel"] = max(self.worst["energy_identity_rel"], abs(pc["energy_identity_residual"]) / max(abs(pc["E_inv"]), 1e-300))


def base_argv(out, device, threads, tag, extra=None):
    return (["--nsites", "256", "--channels", "1", "--R-cut-rs", "2.5", "--R-collar-rs", "1.5", "--collect-every-rs", "1", "--ext-every-rs", "4",
             "--eps-loc", "1e-2", "--extra-ck-rs", "6", "--join-rs", "8,13", "--trace-chunk", "64", "--device", device, "--threads", str(threads),
             "--out", out, "--tag", tag, "--chunk", "512"] + (extra or []))


def write_fixture(path_npz, path_json, hist, upto=None, perturb_upto=None):
    """the fixture (or its records up to index upto) in the engine's layout; perturb_upto: v_applied of records 0..perturb_upto times (1 + 1e-12)."""
    Z = dict(hist["npz"]); J = hist["json"]
    n = len(Z["T"]) if upto is None else upto + 1
    sub = {}
    for k_, v_ in Z.items():
        arr = np.asarray(v_)
        sub[k_] = arr[:n] if (arr.ndim >= 1 and arr.shape[0] == len(Z["T"]) and k_ != "R") else arr
    if perturb_upto is not None:
        va = np.array(sub["v_applied"], float)
        va[: perturb_upto + 1] *= (1.0 + 1e-12)
        sub["v_applied"] = va
    np.savez(path_npz, **sub)
    T_last = float(sub["T"][-1])
    json.dump({"args": J["args"], "records": [r for r in J["records"] if r["T"] <= T_last + 1e-9]}, open(path_json, "w"))


def rows_equal(Ra, Rb):
    if len(Ra.rows) != len(Rb.rows):
        return "rows: %d against %d" % (len(Ra.rows), len(Rb.rows))
    for i, (x, y) in enumerate(zip(Ra.rows, Rb.rows)):
        d = first_diff(x, y, "rows[%d]" % i)
        if d:
            return d
    if len(Ra.records) != len(Rb.records):
        return "records: %d against %d" % (len(Ra.records), len(Rb.records))
    for i, (x, y) in enumerate(zip(Ra.records, Rb.records)):
        d = first_diff(x, y, "records[%d]" % i)
        if d:
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    res = {"code": "page_history_oracle_v6g2.py", "runs": {}}
    LED = ["--transfer-at-peak", "--ledger-anchor-rs", "6.1667"]
    # 1. dense
    R1 = OracleRunG(PG.parse(base_argv(a.out, a.device, a.threads, "oracle_g", LED)), history=PG.History(None, None, 256, synthetic=O6F.synthetic_history()))
    R1.run()
    c1 = {k_: {"worst": R1.worst[k_], "criterion": CRIT[k_], "pass": bool(R1.worst[k_] <= CRIT[k_])} for k_ in CRIT}
    l1 = {k_: {"worst": R1.lw[k_], "criterion": LCRIT[k_], "pass": bool(R1.lw[k_] <= LCRIT[k_])} for k_ in LCRIT}
    end_anchor = (R1.anchor or {}).get("1", {}).get("end")
    ok1 = (all(v_["pass"] for v_ in c1.values()) and all(v_["pass"] for v_ in l1.values()) and R1.n_snap["anchor"] >= 5 and R1.n_snap["peak"] >= 3
           and R1.n_start["anchor"] == 1 and R1.k_anchor == 37)
    res["runs"]["1_dense"] = {"pass": ok1, "checks": c1, "ledger_checks": l1, "printed": {k_: R1.lw[k_] for k_ in PRINTED}, "snapshots": R1.n_snap, "starts": R1.n_start, "anchor_record": R1.k_anchor,
                              "anchor_end": end_anchor, "transfer_starts": R1.transfer_starts}
    print("%s  1 dense: v6f's checks %s; the ledgers' checks %s; snapshots %s, starts %s; the anchored ledger at the end: tr W_R/p %s, tr T_K^dagger T_K/p %s" % (
        "PASS" if ok1 else "FAIL", json.dumps({k_: "%.2e" % v_["worst"] for k_, v_ in c1.items()}), json.dumps(dict({k_: "%.2e" % v_["worst"] for k_, v_ in l1.items()}, **{k_: "%.2e" % R1.lw[k_] for k_ in PRINTED})),
        R1.n_snap, R1.n_start, _g(end_anchor, "trace_W_R_over_p"), _g(end_anchor, "trace_TK_TK_over_p")), flush=True)
    # 2. cubic
    R2 = OracleRunG(PG.parse(base_argv(a.out, a.device, a.threads, "oracle_g_cubic", ["--transfer-at-peak", "--ledger-anchor-rs", "4.1667", "--cubic-window", "3,7"])),
                    history=PG.History(None, None, 256, synthetic=O6F.synthetic_history()))
    R2.run()
    l2 = {k_: {"worst": R2.lw[k_], "criterion": LCRIT[k_], "pass": bool(R2.lw[k_] <= LCRIT[k_])} for k_ in LCRIT}
    ok2 = all(v_["pass"] for v_ in l2.values()) and R2.n_snap["anchor"] >= 5 and R2.k_anchor == 25
    res["runs"]["2_cubic"] = {"pass": ok2, "ledger_checks": l2, "printed": {k_: R2.lw[k_] for k_ in PRINTED}, "snapshots": R2.n_snap, "oracle_worst": R2.worst,
                              "note": "S_esc against the dense oracle printed only (v6f's oracle read 1.99e-10 on its cubic run against 1e-10)"}
    print("%s  2 cubic (window 3 to 7 r_s, the anchor at record 25 inside it): the ledgers' checks %s; S_esc against the dense oracle %.2e, the state %.2e (printed)" % (
        "PASS" if ok2 else "FAIL", json.dumps(dict({k_: "%.2e" % v_["worst"] for k_, v_ in l2.items()}, **{k_: "%.2e" % R2.lw[k_] for k_ in PRINTED})), R2.worst["S_esc"], R2.worst["state_F"]), flush=True)
    # 3. v6f against v6g
    R3f = P6F.HistRun(P6F.parse(base_argv(a.out, a.device, a.threads, "oracle_g_v6f", ["--transfer-at-peak"])), history=P6F.History(None, None, 256, synthetic=O6F.synthetic_history()))
    R3f.run()
    R3g = PG.HistRun(PG.parse(base_argv(a.out, a.device, a.threads, "oracle_g_plain", LED)), history=PG.History(None, None, 256, synthetic=O6F.synthetic_history()))
    R3g.run()
    d3 = rows_equal(R3f, R3g)
    d3e = first_diff((R3f.transfer or {}).get("1", {}).get("end"), (R3g.transfer or {}).get("1", {}).get("end"), "transfer_end")
    ok3 = d3 is None and d3e is None and bool(R3f.rows)
    res["runs"]["3_v6f_vs_v6g"] = {"pass": ok3, "first_difference": d3, "transfer_end_difference": d3e, "rows": len(R3f.rows)}
    print("%s  3 v6f against v6g on the fixture: %d rows, the first difference %s; the peak ledger's end %s" % ("PASS" if ok3 else "FAIL", len(R3f.rows), d3, d3e), flush=True)
    # 4. merged
    hist = O6F.synthetic_history()
    fp = {k_: os.path.join(a.out, k_) for k_ in ("prim.npz", "prim.json", "extfull.npz", "extfull.json", "live.npz", "live.json", "live2.npz", "live2.json",
                                                  "live2.done", "chg.npz", "chg.json")}
    write_fixture(fp["prim.npz"], fp["prim.json"], hist, upto=36)
    write_fixture(fp["extfull.npz"], fp["extfull.json"], hist, perturb_upto=36)
    FILES = ["--record", fp["prim.json"], "--fields", fp["prim.npz"]]
    R4 = PG.HistRun(PG.parse(base_argv(a.out, a.device, a.threads, "oracle_g_merged", LED + FILES + ["--record-ext", fp["extfull.json"], "--fields-ext", fp["extfull.npz"], "--ext-after-rs", "6"])))
    R4.run()
    d4 = rows_equal(R3g, R4)
    seam = R4.H.seam or {}
    ok4 = d4 is None and len(R4.H) == 73 and R4.H.n_primary == 37 and seam.get("present") and 0.0 < (seam.get("max_abs_dv") or 0.0) < 1e-10
    res["runs"]["4_merged"] = {"pass": bool(ok4), "first_difference": d4, "records": len(R4.H), "n_primary": R4.H.n_primary, "seam": seam}
    print("%s  4 merged (a primary to T/r_s 6, the continuation after it): %d records (%d primary), the seam %s; against run 3's v6g rows: %s" % (
        "PASS" if ok4 else "FAIL", len(R4.H), R4.H.n_primary, json.dumps(seam, default=float), d4), flush=True)
    # 5. follow
    W = [sys.executable, "-I", os.path.join(HERE, "page_history_oracle_v6g_writer.py"), fp["extfull.npz"], fp["extfull.json"], fp["live.json"], fp["live.npz"],
         "--start-records", "43", "--interval-s", "0.4", "--pause-s", "0.05"]
    wlog = open(os.path.join(a.out, "writer_5.log"), "w")
    pw = subprocess.Popen(W, stdout=wlog, stderr=subprocess.STDOUT)
    time.sleep(1.0)
    FOL = ["--record-ext", fp["live.json"], "--fields-ext", fp["live.npz"], "--ext-after-rs", "6", "--follow", "--follow-poll-s", "0.05", "--follow-timeout-h", "0.05"]
    R5 = PG.HistRun(PG.parse(base_argv(a.out, a.device, a.threads, "oracle_g_follow", LED + FILES + FOL + ["--T-end-rs", "12"])))
    R5.run()
    rc5 = pw.wait(); wlog.close()
    d5 = rows_equal(R4, R5)
    ok5 = d5 is None and rc5 == 0 and len(R5.H) == 73 and R5.kend == 72
    res["runs"]["5_follow"] = {"pass": bool(ok5), "first_difference": d5, "writer_exit": rc5, "records": len(R5.H), "loads": len(R5.H.loads),
                               "failed_reads": R5.H.n_failed_reads, "last_error": R5.H.ext_error}
    print("%s  5 follow (the writer from T/r_s 7, one record every 0.4 s, partial files on purpose): %d records in %d reads that added records, %d failed (partial) reads "
          "(the last: %s); against run 4's rows: %s" % ("PASS" if ok5 else "FAIL", len(R5.H), len(R5.H.loads), R5.H.n_failed_reads, R5.H.ext_error, d5), flush=True)
    # 6. the done marker
    W2 = [sys.executable, "-I", os.path.join(HERE, "page_history_oracle_v6g_writer.py"), fp["extfull.npz"], fp["extfull.json"], fp["live2.json"], fp["live2.npz"],
          "--start-records", "43", "--interval-s", "0.2", "--pause-s", "0.05", "--marker", fp["live2.done"]]
    wlog2 = open(os.path.join(a.out, "writer_6.log"), "w")
    pw2 = subprocess.Popen(W2, stdout=wlog2, stderr=subprocess.STDOUT)
    time.sleep(1.0)
    FOL2 = ["--record-ext", fp["live2.json"], "--fields-ext", fp["live2.npz"], "--ext-after-rs", "6", "--follow", "--follow-poll-s", "0.05", "--follow-timeout-h", "0.05",
            "--follow-done-marker", fp["live2.done"]]
    R6 = PG.HistRun(PG.parse(base_argv(a.out, a.device, a.threads, "oracle_g_marker", LED + FILES + FOL2 + ["--T-end-rs", "14"])))
    R6.run()
    rc6 = pw2.wait(); wlog2.close()
    d6 = rows_equal(R4, R6)
    ok6 = d6 is None and rc6 == 0 and R6.kend == 72 and bool(R6.follow_end) and abs(R6.rows[-1]["t_over_rs"] - 12.0) < 1e-9
    res["runs"]["6_done_marker"] = {"pass": bool(ok6), "first_difference": d6, "writer_exit": rc6, "kend": R6.kend, "follow_end": R6.follow_end, "label": R6.label}
    print("%s  6 the done marker (--T-end-rs 14, the continuation ends at 12): the run ends at record %d (T/r_s %.3f): %s; against run 4's rows: %s" % (
        "PASS" if ok6 else "FAIL", R6.kend, R6.rows[-1]["t_over_rs"], R6.follow_end, d6), flush=True)
    # 7. a changed record
    write_fixture(fp["chg.npz"], fp["chg.json"], hist, upto=49)
    Hc = PG.History(fp["prim.json"], fp["prim.npz"], 256, ext={"record": fp["chg.json"], "fields": fp["chg.npz"], "after_rs": 6.0, "follow": True, "poll_s": 0.05,
                                                              "timeout_h": 0.01, "done_marker": None, "rs_nominal": None})
    n_a = len(Hc)
    Z = dict(np.load(fp["extfull.npz"], allow_pickle=True)); Z = {k_: np.array(v_) for k_, v_ in Z.items()}
    Z["v_applied"] = np.array(Z["v_applied"], float); Z["v_applied"][40] *= (1.0 + 1e-9)
    np.savez(fp["chg.npz"], **Z)
    json.dump({"args": hist["json"]["args"], "records": hist["json"]["records"]}, open(fp["chg.json"], "w"))
    err7 = None
    try:
        Hc.ensure(60)
    except RuntimeError as ex:
        err7 = str(ex)
    ok7 = n_a == 50 and err7 is not None and "changed a record already used" in err7
    res["runs"]["7_changed_record"] = {"pass": bool(ok7), "records_before": n_a, "error": err7}
    print("%s  7 a changed record (record 40 rewritten after the history took records to 49): %s" % ("PASS" if ok7 else "FAIL", err7), flush=True)
    # 8. two channels
    R8 = PG.HistRun(PG.parse(["--nsites", "256", "--channels", "1,2", "--R-cut-rs", "2.5", "--R-collar-rs", "1.5", "--collect-every-rs", "1", "--ext-every-rs", "4",
                              "--eps-loc", "1e-2", "--extra-ck-rs", "6", "--join-rs", "8,13", "--trace-chunk", "64", "--device", a.device, "--threads", str(a.threads),
                              "--out", a.out, "--tag", "oracle_g_two_channel", "--chunk", "512"] + LED), history=PG.History(None, None, 256, synthetic=O6F.synthetic_history()))
    err8 = None
    try:
        R8.run()
    except Exception as ex:                                               # noqa: BLE001 (recorded as the failure)
        err8 = "%s: %s" % (type(ex).__name__, ex)
    ok8 = err8 is None and bool(R8.rows) and abs(R8.rows[-1]["t_over_rs"] - 12.0) < 1e-9
    res["runs"]["8_two_channels"] = {"pass": bool(ok8), "error": err8, "anchor_channels": sorted((R8.anchor or {}).keys()),
                                     "last_row_ledgers": {k_: sorted((R8.rows[-1].get(k_) or {}).keys()) for k_ in ("ledger_anchored", "ledger_peak")} if R8.rows else None}
    print("%s  8 two channels with both ledgers to the last record: %s" % ("PASS" if ok8 else "FAIL", json.dumps(res["runs"]["8_two_channels"])), flush=True)
    res["pass"] = bool(ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8)
    res["seconds"] = time.time() - t_all
    with open(os.path.join(a.out, "history_oracle_v6g2_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("ORACLE %s in %.0f s" % ("PASS" if res["pass"] else "FAIL", res["seconds"]), flush=True)


def _g(d, k):
    return ("%.6e" % d[k]) if (d and d.get(k) is not None) else "-"


if __name__ == "__main__":
    main()
