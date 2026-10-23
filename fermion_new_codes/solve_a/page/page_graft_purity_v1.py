#!/usr/bin/env python
"""page_graft_purity_v1.py (the solve lane's analyst builder 11; engine spec 53.2 (2), (3), (4)): the graft's purity readout beside its
join null readout. page_graft_pullback_v2 is imported read-only (step_coeffs, cheb_filter_pair, walled, split_weight; version 1's
cheb_plus and split_from_block through it); the propagation and the readouts' arithmetic follow version 2's main.

The graft's state is P_long + Delta0 (P_long the long chain's connected vacuum, Delta0 = Phi Phi^dagger - P_-^short on the short chain).
With D = P_walled - P_long (the null readout's D_sigma = f(H_walled) - f(H_connected) is its smoothed form) and Phi pure on the short chain,
(P_long + Delta0)^2 - (P_long + Delta0) = -(D Delta0 + Delta0 D) exactly: the construction's one defect. The purity readout is
Tr(O(t) (D_sigma Delta0 + Delta0 D_sigma)) contracted as the null readout is: with X = W(t)'s rows on the join's window (lo_j..hi_j),
Dx = D_sigma X (the filter on the window, exact for the polynomial) and Z = (Delta0 W_s)'s rows on the window (zero at rows >= 2 ns),
the block B_P = Dx^dagger Z + Z^dagger Dx = W^dagger (D_sigma Delta0 + Delta0 D_sigma) W; L_pur = Re Tr(J_SS B_P), the windowed split's
entries from B_P (E_flux_out(32)_pur printed), and the inside energy I_pur = Re Tr((h W_in)^dagger (D_sigma Delta0 + Delta0 D_sigma) W_in)
= Re Tr((D_sigma (h W_in))^dagger (Delta0 W_in)) + Re Tr((h W_in)_s^dagger Delta0 (D_sigma W_in)) (one more filter, on the inside columns).
The construction's defect is minus these. Remainders: the filter's error 2 eps in D_sigma gives |dB_P| <= 4 eps ||X||_F ||Z||_F, so
rem_L_pur = 4 eps ||J_SS|| ||X||_F ||Z||_F. L_pur and E_flux_out(32)_pur at 2 sigma beside (the resolution's dependence).
Readouts at every row from --readouts-from-rs (default 115) to the append's end, the null readout beside each; rows before are propagated
only. At the hold's start per channel: the deviation's weight in the window (||Delta0[win, win]||_F, ||Delta0[win, :]||_F, its largest
|eigenvalue| on the window block). With --trunc 1 Delta0 is version 2's certified truncation.
The gate (53.2 (4)), printed at the end (<tag>_purity_gate.md): per channel the first row from which |L_pur| and |E_flux_out(32)_pur| stay under
the digit 1e-3 L_H to the last row; GATE PASS when that clock exists in every channel (rows before it labelled join-dominated, out of every
emptying fit); else GATE FAIL (nothing launches; the owner is told).
Outputs (rule 10): <tag>_rows.json (every computed row: L_at_t, I_inside, collar_split, null, purity), <tag>_purity_gate.md.
Usage: as page_graft_pullback_v2 plus --readouts-from-rs."""
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
import page_graft_pullback_v2 as G2   # noqa: E402  (read-only)
G1 = G2.G1; PK = G2.PK; OPS = G2.OPS; CD = G2.CD
T0 = time.time()


def log(m):
    print("[%7.1f s] %s" % (time.time() - T0, m), flush=True)


def purity_block(Dx, Z):
    """B_P = Dx^dagger Z + Z^dagger Dx (W^dagger (D Delta0 + Delta0 D) W from the window's rows)."""
    return Dx.mH @ Z + Z.mH @ Dx


def gate(rows_by_ch, digit):
    out = {}
    for kap, rows in rows_by_ch.items():
        rr = [r for r in rows if r.get("purity")]
        ok = [abs(r["purity"]["L_pur"]) < digit and abs(r["purity"]["E_flux_out_32_pur"]) < digit for r in rr]
        start = None
        for i in range(len(rr)):
            if all(ok[i:]):
                start = i; break
        out[kap] = {"rows": len(rr), "clock_under_to_end": (rr[start]["t_over_rs"] if start is not None else None),
                    "join_dominated_rows": (start if start is not None else len(rr)),
                    "max_abs_L_pur": max((abs(r["purity"]["L_pur"]) for r in rr), default=None),
                    "last_L_pur": rr[-1]["purity"]["L_pur"] if rr else None, "last_E_flux_out_32_pur": rr[-1]["purity"]["E_flux_out_32_pur"] if rr else None}
    passed = bool(out) and all(v["clock_under_to_end"] is not None and v["rows"] > 0 for v in out.values())
    return passed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--fields-key", default="applied")
    ap.add_argument("--init-state", required=True); ap.add_argument("--nsites-short", type=int, default=8192); ap.add_argument("--append", type=int, required=True)
    ap.add_argument("--channels", default="1,2"); ap.add_argument("--freeze-at-rs", type=float, required=True)
    ap.add_argument("--R-collar-rs", type=float, default=16.0); ap.add_argument("--collect-every-rs", type=float, default=2.0)
    ap.add_argument("--duration-rs", type=float, default=300.0); ap.add_argument("--windows", default="32,16,64")
    ap.add_argument("--cheb-tol", type=float, default=1e-14); ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--null-sigma-TH", type=float, default=8.0); ap.add_argument("--null-tol", type=float, default=1e-12)
    ap.add_argument("--trunc", type=int, default=0); ap.add_argument("--trunc-drop-sum", type=float, default=0.0)
    ap.add_argument("--trunc-digit-frac", type=float, default=1e-3); ap.add_argument("--LH", type=float, required=True)
    ap.add_argument("--readouts-from-rs", type=float, default=115.0)
    a = ap.parse_args()
    assert a.append > 0
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True); pre = os.path.join(a.out, a.tag)
    Hs = PK.History(a.record, a.fields, a.nsites_short, None, a.fields_key)
    Hl = PK.History(a.record, a.fields, a.nsites_short + a.append, None, a.fields_key)
    rs = float(Hs.rs_nom or 2.0 * Hs.g * Hs.E[0]); tau = Hs.tau
    kf = int(np.argmin(np.abs(Hs.T - a.freeze_at_rs * rs))); tf = float(Hs.T[kf])
    every = max(1, int(round(a.collect_every_rs * rs / tau))); nsteps = int(round(a.duration_rs * rs / tau))
    trow = every * tau; nrows = nsteps // every
    Rc_abs = a.R_collar_rs * rs
    wins = [int(x) for x in a.windows.split(",") if x.strip()]; Lmax = max(wins)
    st = torch.load(a.init_state, map_location="cpu", weights_only=False)
    Nl, vl = Hl.fields(kf); Ns, vs = Hs.fields(kf)
    Rl = Hl.R; ns = a.nsites_short; n2s = 2 * ns
    ic = int(np.searchsorted(Rl, Rc_abs, side="right"))
    lo_w = max(0, ic - Lmax // 2); hi_w = min(ns, lo_w + Lmax)
    T_H = 1.0 / (4.0 * math.pi * rs)
    digit = a.trunc_digit_frac * a.LH
    log("graft purity v1: record kf %d T/r_s %.4f, short %d, appended %d, %d rows of %.1f a; readouts from T/r_s %.1f; digit %.3e" % (
        kf, tf / rs, ns, a.append, nrows, trow, a.readouts_from_rs, digit))
    res = {"settings": vars(a), "kf": kf, "T_freeze": tf, "rs": rs, "tau": tau, "row_time": trow, "T_H": T_H, "digit": digit, "channels": {},
           "purity_readout": "Tr(O(t)(D_sigma Delta0 + Delta0 D_sigma)), D_sigma = f(H_walled) - f(H_connected); the construction's defect is minus it",
           "label": "the graft's purity readout beside its join null readout (53.2)"}
    for kap in [int(x) for x in a.channels.split(",") if x.strip()]:
        t_c = time.time()
        ops_s = PK.ChannelOps(kap, Hs.R, dev); ops_l = PK.ChannelOps(kap, Rl, dev)
        hs = ops_s.h_csr(Ns, vs); hl = ops_l.h_csr(Nl, vl)
        e, V = OPS.eigh_csr(hs, dev)
        Vn = V[:, e < 0].clone(); del V, e
        Phi = st["channels"][str(kap)]["Phi"].to(dev)
        J = OPS.current_out_csr(hl, Rl, Rc_abs).tocoo()
        jsites = np.unique(np.concatenate([J.row // 2, J.col // 2]))
        inside = np.nonzero(Rl[:ns] <= Rc_abs)[0]
        S = np.unique(np.concatenate([inside, jsites, np.arange(lo_w, hi_w)]))
        idx = np.sort(np.concatenate([2 * S, 2 * S + 1]))
        pos2 = {int(r): j for j, r in enumerate(idx)}
        Jss = np.zeros((len(idx), len(idx)), complex)
        for r_, c_, d_ in zip(J.row, J.col, J.data):
            Jss[pos2[int(r_)], pos2[int(c_)]] += d_
        Jss_t = torch.as_tensor(Jss, dtype=CD, device=dev); J_norm = float(np.linalg.norm(Jss, 2))
        inn = np.array([pos2[int(r)] for r in np.sort(np.concatenate([2 * inside, 2 * inside + 1]))]); inn_t = torch.as_tensor(inn, device=dev)
        Bh = OPS.Banded(hl, dev)
        # the truncation (version 2's rule)
        trunc = {"taken": False}; Vr = Lr = None
        if a.trunc:
            Dl0 = Phi @ Phi.mH - Vn @ Vn.mH; Dl0 = 0.5 * (Dl0 + Dl0.mH)
            lam, U = torch.linalg.eigh(Dl0); del Dl0
            lam_np = lam.cpu().numpy(); order = np.argsort(np.abs(lam_np)); cs = np.cumsum(np.abs(lam_np[order]))
            wsplit = G2.split_weight(Rl, Nl, vl, ic, kap, max(wins))
            budget = a.trunc_drop_sum if a.trunc_drop_sum > 0 else 0.5 * digit / max(J_norm, Bh.bound, wsplit)
            ndrop = int(np.searchsorted(cs, budget, side="right")); sdrop = float(cs[ndrop - 1]) if ndrop else 0.0
            keep = np.sort(order[ndrop:])
            cert = {"L_at_t": sdrop * J_norm, "I_inside": sdrop * Bh.bound, "split_E_flux": sdrop * wsplit}
            ok = all(x < digit for x in cert.values())
            trunc = {"taken": bool(ok), "kept_pairs": int(len(keep)), "dropped_pairs": int(ndrop), "dropped_sum": sdrop, "certificate": cert}
            if ok:
                kt = torch.as_tensor(keep, device=dev); Vr = U[:, kt].contiguous(); Lr = lam[kt].to(CD)
            del U, lam

        def delta(Ys):
            """Delta0 Ys for Ys on the short chain's rows."""
            if Vr is not None:
                return Vr @ (Lr[:, None] * (Vr.mH @ Ys))
            return Phi @ (Phi.mH @ Ys) - Vn @ (Vn.mH @ Ys)
        # the filter on the join's window (version 2's construction)
        sig = a.null_sigma_TH * T_H; lam_b = Bh.bound * (1.0 + 1e-6)
        c1, M1, eps1 = G2.step_coeffs(sig, lam_b, a.null_tol)
        c2, M2, eps2 = G2.step_coeffs(2.0 * sig, lam_b, a.null_tol)
        c2p = np.zeros(M1 + 1, complex); c2p[:min(M1, M2) + 1] = c2[:min(M1, M2) + 1]
        eps2 = eps2 + (float(np.sum(np.abs(c2[M1 + 1:]))) if M2 > M1 else 0.0)
        A = hl.tocsr(); rr_ = max(abs(d) for d, _ in Bh.diags)
        half = 2 * (M1 + 1) * rr_
        lo_j = max(0, n2s - half); hi_j = min(A.shape[0], n2s + half); hs_j = n2s - lo_j     # window rows on the short chain: lo_j..n2s
        Aw = A[lo_j:hi_j, lo_j:hi_j]
        Bc = OPS.Banded(Aw, dev, dmax=max(16, rr_)); Bwl = OPS.Banded(G2.walled(Aw, ns - lo_j // 2), dev, dmax=max(16, rr_))
        # the deviation's weight in the window at the hold's start
        Ewin = torch.zeros((n2s, hs_j), dtype=CD, device=dev); Ewin[lo_j + torch.arange(hs_j, device=dev), torch.arange(hs_j, device=dev)] = 1.0
        Dcol = delta(Ewin); del Ewin                                                     # Delta0[:, win]
        blk_ww = Dcol[lo_j:n2s]
        weight = {"window_rows": [int(lo_j), int(n2s)], "fro_window_block": float(torch.linalg.norm(blk_ww)), "fro_window_rows": float(torch.linalg.norm(Dcol)),
                  "max_abs_eig_window_block": float(torch.linalg.eigvalsh(0.5 * (blk_ww + blk_ww.mH)).abs().max()) if hs_j else 0.0}
        del Dcol, blk_ww
        nul = {"sigma": sig, "M": M1, "eps": eps1, "M_2sigma": M2, "eps_2sigma": eps2, "index_range": rr_, "window_rows": [int(lo_j), int(hi_j)], "Lambda": lam_b}
        log("kappa %d: filter sigma %.3e (%.1f T_H) M %d eps %.1e; window rows %d..%d (short to %d); Delta0's window weight: block %.3e, rows %.3e, max |eig| %.3e; trunc %s" % (
            kap, sig, a.null_sigma_TH, M1, eps1, lo_j, hi_j, n2s, weight["fro_window_block"], weight["fro_window_rows"], weight["max_abs_eig_window_block"], trunc.get("taken")))
        rr = np.concatenate([[pos2[2 * s], pos2[2 * s] + 1] for s in range(lo_w, hi_w)])
        W = torch.zeros((2 * len(Rl), len(idx)), dtype=CD, device=dev)
        W[torch.as_tensor(idx, device=dev), torch.arange(len(idx), device=dev)] = 1.0
        rows = []; K_max = 0

        def readouts(W, j):
            t_r = time.time()
            Ws = W[:n2s]
            DW = delta(Ws)                                                      # Delta0 W_s
            B = Ws.mH @ DW
            L = float(torch.trace(Jss_t @ B).real)
            Y = Bh.apply(W[:, inn_t])                                           # h W_in (all rows)
            I = float(torch.sum(Y[:n2s].conj() * DW[:, inn_t]).real)
            Bn = B.cpu().numpy()
            spj = G1.split_from_block(Bn[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
            X = W[lo_j:hi_j]; xn = float(torch.linalg.norm(X))
            Dx = [torch.zeros_like(X), torch.zeros_like(X)]
            for j0 in range(0, X.shape[1], a.chunk):
                Xc = X[:, j0:j0 + a.chunk]
                if float(torch.linalg.norm(Xc)) == 0.0:
                    continue
                r = G2.cheb_filter_pair(Bwl, Bc, Xc, [c1, c2p], lam_b)
                Dx[0][:, j0:j0 + a.chunk] = r[0]; Dx[1][:, j0:j0 + a.chunk] = r[1]
            Yh = Y[lo_j:hi_j]
            nullr = {"X_window_norm2": xn ** 2}
            Z = torch.zeros_like(X); Z[:hs_j] = DW[lo_j:n2s]                    # (Delta0 W)'s window rows
            zn = float(torch.linalg.norm(Z))
            pur = {"Z_window_norm": zn}
            for tagz, D_ in (("", Dx[0]), ("_2sigma", Dx[1])):
                BD = X.mH @ D_
                nullr["L_null" + tagz] = float(torch.trace(Jss_t @ BD).real)
                nullr["I_null" + tagz] = float(torch.sum(Yh.conj() * D_[:, inn_t]).real)
                spn = G1.split_from_block(BD.cpu().numpy()[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
                nullr["E_flux_out_32_null" + tagz] = spn["windows"]["32"]["E_flux_out"]
                BP = purity_block(D_, Z)
                pur["L_pur" + tagz] = float(torch.trace(Jss_t @ BP).real)
                spp = G1.split_from_block(BP.cpu().numpy()[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
                pur["E_flux_out_32_pur" + tagz] = spp["windows"]["32"]["E_flux_out"]
                if tagz == "":
                    pur["split_pur"] = {L_: {k_: v_ for k_, v_ in w_.items() if k_.startswith("E_flux") or k_.startswith("n_flux")} for L_, w_ in spp["windows"].items()}
            nullr["rem_L"] = 2 * eps1 * J_norm * xn ** 2
            # I_pur: Re Tr((D (h W_in))^dagger (Delta0 W_in)) + Re Tr((h W_in)_s^dagger Delta0 (D W_in))
            DY = torch.zeros_like(Yh)
            for j0 in range(0, Yh.shape[1], a.chunk):
                Yc = Yh[:, j0:j0 + a.chunk]
                if float(torch.linalg.norm(Yc)) == 0.0:
                    continue
                DY[:, j0:j0 + a.chunk] = G2.cheb_filter_pair(Bwl, Bc, Yc, [c1], lam_b)[0]
            t1 = float(torch.sum(DY.conj() * Z[:, inn_t]).real)
            DWin_s = torch.zeros((n2s, len(inn)), dtype=CD, device=dev); DWin_s[lo_j:n2s] = Dx[0][:hs_j][:, inn_t]
            t2 = float(torch.sum(Y[:n2s].conj() * delta(DWin_s)).real)
            pur["I_pur"] = t1 + t2
            pur["rem_L_pur"] = 4 * eps1 * J_norm * xn * zn
            return {"j": j, "t": tf + j * trow, "t_over_rs": (tf + j * trow) / rs, "L_at_t": L, "I_inside": I, "collar_split": spj,
                    "null": nullr, "purity": pur, "seconds": time.time() - t_r}

        j_from = int(math.ceil(((a.readouts_from_rs * rs) - tf) / trow - 1e-9))
        for j in range(1, nrows + 1):
            for j0 in range(0, W.shape[1], a.chunk):
                W[:, j0:j0 + a.chunk], K, tail = G1.cheb_plus(Bh, W[:, j0:j0 + a.chunk], trow, a.cheb_tol)
                K_max = max(K_max, K)
            if j >= j_from:
                row = readouts(W, j); rows.append(row)
                p_ = row["purity"]; n_ = row["null"]
                log("kappa %d row %d/%d T/r_s %.2f: L %.6e; null L %.3e; purity L %.3e (2s %.3e), E_flux_out(32) %.3e, I %.3e, rem %.1e; digit %.2e (%.1f s)" % (
                    kap, j, nrows, row["t_over_rs"], row["L_at_t"], n_["L_null"], p_["L_pur"], p_["L_pur_2sigma"], p_["E_flux_out_32_pur"], p_["I_pur"],
                    p_["rem_L_pur"], digit, row["seconds"]))
        res["channels"][str(kap)] = {"w": ops_s.w, "rows": rows, "cheb_K_max": K_max, "n_columns": len(idx), "seconds": time.time() - t_c,
                                     "filter": nul, "trunc": trunc, "J_SS_norm": J_norm, "deviation_window_weight_at_hold_start": weight}
        json.dump(res, open(pre + "_rows.json", "w"))
        del W, Phi, Vn, Bh
        if Vr is not None:
            del Vr, Lr
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    passed, g = gate({k: v["rows"] for k, v in res["channels"].items()}, digit)
    res["gate"] = {"pass": passed, "channels": g}
    json.dump(res, open(pre + "_rows.json", "w"))
    L_ = ["# %s: the purity readout's gate (engine spec 53.2 (4))" % a.tag, "",
          "digit 1e-3 L_H = %.3e; rows from T/r_s %.1f to the append's end; criterion |L_pur| and |E_flux_out(32)_pur| under the digit from a clock to the last row." % (digit, a.readouts_from_rs), ""]
    for k, v in g.items():
        L_.append("- kappa %s: %d rows; under the digit from T/r_s %s to the end (%d rows before it join-dominated); max |L_pur| %.3e; last L_pur %.3e, E_flux_out(32)_pur %.3e" % (
            k, v["rows"], ("%.2f" % v["clock_under_to_end"]) if v["clock_under_to_end"] is not None else "NEVER", v["join_dominated_rows"],
            v["max_abs_L_pur"] or 0.0, v["last_L_pur"] or 0.0, v["last_E_flux_out_32_pur"] or 0.0))
    L_ += ["", "GATE %s" % ("PASS: the long chains launch" if passed else "FAIL: nothing launches; for the owner")]
    open(pre + "_purity_gate.md", "w").write("\n".join(L_) + "\n")
    log("GATE %s; written %s_rows.json and _purity_gate.md" % ("PASS" if passed else "FAIL", pre))


if __name__ == "__main__":
    main()
