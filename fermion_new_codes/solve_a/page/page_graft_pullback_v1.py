#!/usr/bin/env python
"""page_graft_pullback_v1.py (the solve lane's analyst builder 9; engine spec 51.2 (1), the owner's approved graft, #88): the held-field
control's readouts on the grafted long chain as pulled-back detector columns. New code; page_history_v6k is imported read-only for its
History (the record's fields, the vacuum exterior N = 1, v = -sqrt(2 g E_k / R) beyond the engine's sites), its ChannelOps (the light
channel's clocked h, m = 0) and its operators; nothing of it is changed.

Under the fields held at the record kf (T/r_s --freeze-at-rs), the hold's state is C(t) = P_- + U(t) Delta0 U(t)^dagger, U = exp(-i h t), with
Delta0 = C(t_f) - P_-^short on the 8192-site chain (the hold's saved state Phi: C = Phi Phi^dagger; P_-^short the negative-energy projector
of the frozen short-chain h) and, on the graft, Delta0 (+) 0 on the long chain of --nsites-short + --append sites (the long chain's own
Killing vacuum P_-^long in place of P_-). Every readout is linear in the deviation: with W(t) = exp(+i h t) E_S, E_S the unit columns of
the detector sites S (the sites inside R_collar, the collar current's support, the largest split window), the block of C(t) - P_- on S is
W_s^dagger Delta0 W_s (W_s the rows of W on the short chain); from it
  L_at_t     = Re Tr(J_SS B)        (J = (i/2)[h^2, Pi_{R > R_collar}], page_ops_v6.current_out_csr; Tr(J P_-) = 0 for an eigenprojector),
  collar_flow over an interval = -(I(t_b) - I(t_a)),  I(t) = Re Tr((h W_in)_s^dagger Delta0 W_in,s)  (Tr(h Delta) conserved),
  the windowed signed split at every window (page_history_v6k.collar_split's arithmetic on the window's block of C - P_-).
The columns are carried by Chebyshev steps exp(+i h tau_row) on the long chain's banded h (the Bessel order by the tail below --cheb-tol).
Gate (51.2 (1)): with --append 0 the same closed system as the stored hold, every record's L_at_t and collar_flow and every row's split
compared with the stored summary (--compare): PASS when the largest absolute difference of L_at_t, collar_flow and the windows' E_flux
entries is below 1e-6 of the row's comparator L_H(kappa_T) (1e-3 of the digit 1e-3 L_H), stated before the run.
The join's vacuum null readout (51.2 (2)) is not in this version (its filter is the next build); the rows say so.
Outputs (rule 10): <out>/<tag>_rows.json (every row and record, per channel), <tag>_blocks.npz (the largest window's block per row and
channel), <tag>_compare.md when --compare is given. Usage: see --help."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_history_v6k as PK   # noqa: E402
OPS = PK.OPS
CD = PK.CD
T0 = time.time()


def log(m):
    print("[%7.1f s] %s" % (time.time() - T0, m), flush=True)


def cheb_plus(Bh, X, tau, tol):
    """exp(+i tau h) X for a device Banded h (bound w): e^{i z x} = J0(z) + 2 sum_k i^k J_k(z) T_k(x), z = w tau, x = h/w."""
    w = Bh.bound; z = w * tau
    K, tail = OPS.cheb_order(z, tol)
    k = np.arange(K + 1)
    ck = jv(k, z) * ((1j) ** k) * np.where(k == 0, 1.0, 2.0)
    T0_ = X; T1_ = Bh.apply(X) / w
    Y = complex(ck[0]) * T0_ + complex(ck[1]) * T1_
    for kk in range(2, K + 1):
        T2_ = 2.0 * Bh.apply(T1_) / w - T0_
        Y = Y + complex(ck[kk]) * T2_
        T0_, T1_ = T1_, T2_
    return Y, K, tail


def split_from_block(Dfull, sites_full, ic, R, N, v, kappa, wins):
    """page_history_v6k.collar_split's windowed signed split, from the block D = (C - P_-) on the largest window's sites (0-based site
    indices sites_full; D in the interleaved 2-component order)."""
    n = len(R); Rc, Nc, vc = float(R[ic]), float(N[ic]), float(v[ic])
    pos = {s: j for j, s in enumerate(sites_full)}
    out = {"centre_site": ic + 1, "R_c": Rc, "N_c": Nc, "v_c": vc, "windows": {}}
    for L in wins:
        hL = L // 2; i0 = ic - hL; i1 = i0 + L
        lo, hi = max(0, i0), min(n, i1)
        gw = np.sin(math.pi * (np.arange(L) + 0.5) / L)[lo - i0:hi - i0]
        rr = np.concatenate([[2 * pos[s], 2 * pos[s] + 1] for s in range(lo, hi)])
        Dl = Dfull[np.ix_(rr, rr)]
        Dl = 0.5 * (Dl + Dl.conj().T)
        M = 4 * L; q = 2.0 * math.pi * np.fft.fftfreq(M); nl = hi - lo
        Fm = np.fft.fft(np.eye(M, nl) * gw[None, :], axis=0, norm="ortho")
        G = {}
        for aa in (0, 1):
            for bb in (0, 1):
                G[(aa, bb)] = np.sum((Fm @ Dl[aa::2, bb::2]) * Fm.conj(), axis=1)
        s_ = np.sin(q); c_ = np.cos(q); kr = kappa / Rc
        E = np.sqrt(s_ ** 2 + kr ** 2); nrm = np.sqrt(2.0 * E * E)
        u0 = E / nrm; u1 = (s_ + 1j * kr) / nrm
        n_tot = (G[(0, 0)] + G[(1, 1)]).real
        n_p = (np.conj(u0) * u0 * G[(0, 0)] + np.conj(u0) * u1 * G[(0, 1)] + np.conj(u1) * u0 * G[(1, 0)] + np.conj(u1) * u1 * G[(1, 1)]).real
        n_m = n_tot - n_p
        om_p = Nc * E + 0.5 * vc * np.sin(2.0 * q); om_m = -Nc * E + 0.5 * vc * np.sin(2.0 * q)
        vp = Nc * s_ * c_ / E + vc * np.cos(2.0 * q); vm = -Nc * s_ * c_ / E + vc * np.cos(2.0 * q)
        W2 = float(np.sum(gw ** 2)); po = vp > 0; mo = vm > 0
        w_ = {"L": L, "sites_valid": int(nl), "sum_w2": W2,
              "E_flux_plus_out": float(np.sum((om_p * vp * n_p)[po])) / W2, "E_flux_plus_in": float(np.sum((om_p * vp * n_p)[~po])) / W2,
              "E_flux_minus_out": float(np.sum((om_m * vm * n_m)[mo])) / W2, "E_flux_minus_in": float(np.sum((om_m * vm * n_m)[~mo])) / W2,
              "n_flux_out": (float(np.sum((vp * n_p)[po])) + float(np.sum((vm * n_m)[mo]))) / W2,
              "n_flux_in": (float(np.sum((vp * n_p)[~po])) + float(np.sum((vm * n_m)[~mo]))) / W2,
              "E_density": (float(np.sum(om_p * n_p)) + float(np.sum(om_m * n_m))) / W2,
              "n_density_plus": float(np.sum(n_p)) / W2, "n_density_minus": float(np.sum(n_m)) / W2}
        w_["E_flux_out"] = w_["E_flux_plus_out"] + w_["E_flux_minus_out"]
        w_["E_flux_in"] = w_["E_flux_plus_in"] + w_["E_flux_minus_in"]
        w_["E_flux_net"] = w_["E_flux_out"] + w_["E_flux_in"]
        w_["n_flux_net"] = w_["n_flux_out"] + w_["n_flux_in"]
        out["windows"][str(L)] = w_
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--fields-key", default="applied")
    ap.add_argument("--init-state", required=True); ap.add_argument("--nsites-short", type=int, default=8192); ap.add_argument("--append", type=int, default=0)
    ap.add_argument("--channels", default="1,2"); ap.add_argument("--freeze-at-rs", type=float, required=True)
    ap.add_argument("--R-collar-rs", type=float, default=16.0); ap.add_argument("--collect-every-rs", type=float, default=2.0)
    ap.add_argument("--duration-rs", type=float, default=300.0); ap.add_argument("--windows", default="32,16,64")
    ap.add_argument("--cheb-tol", type=float, default=1e-14); ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True); pre = os.path.join(a.out, a.tag)
    Hs = PK.History(a.record, a.fields, a.nsites_short, None, a.fields_key)
    Hl = PK.History(a.record, a.fields, a.nsites_short + a.append, None, a.fields_key) if a.append else Hs
    rs = float(Hs.rs_nom or 2.0 * Hs.g * Hs.E[0]); tau = Hs.tau
    kf = int(np.argmin(np.abs(Hs.T - a.freeze_at_rs * rs))); tf = float(Hs.T[kf])
    every = max(1, int(round(a.collect_every_rs * rs / tau))); nsteps = int(round(a.duration_rs * rs / tau))
    trow = every * tau; nrows = nsteps // every
    Rc_abs = a.R_collar_rs * rs
    wins = [int(x) for x in a.windows.split(",") if x.strip()]; Lmax = max(wins)
    st = torch.load(a.init_state, map_location="cpu", weights_only=False)
    Ns, vs = Hs.fields(kf); Nl, vl = Hl.fields(kf)
    Rl = Hl.R; ns = a.nsites_short; n2s = 2 * ns
    ic = int(np.searchsorted(Rl, Rc_abs, side="right"))
    lo_w = max(0, ic - Lmax // 2); hi_w = min(ns, lo_w + Lmax)
    log("graft pull-back: record kf %d T/r_s %.4f, short chain %d sites, appended %d (long chain %d), row every %d steps (%.1f a), %d rows; R_collar %.1f, centre site %d" % (
        kf, tf / rs, ns, a.append, len(Rl), every, trow, nrows, Rc_abs, ic + 1))
    res = {"settings": vars(a), "kf": kf, "T_freeze": tf, "rs": rs, "tau": tau, "row_time": trow, "channels": {},
           "join_null_readout": "not computed in version 1 (the filter is the next build)",
           "label": "the held-field control's readouts as pulled-back detector columns" + (" on the graft (short chain's deviation in the long chain's Killing vacuum)" if a.append else " on the short chain (the gate)")}
    blocks = {}
    for kap in [int(x) for x in a.channels.split(",") if x.strip()]:
        t_c = time.time()
        ops_s = PK.ChannelOps(kap, Hs.R, dev); ops_l = PK.ChannelOps(kap, Rl, dev) if a.append else ops_s
        hs = ops_s.h_csr(Ns, vs); hl = ops_l.h_csr(Nl, vl) if a.append else hs
        e, V = OPS.eigh_csr(hs, dev)
        Vn = V[:, e < 0].clone(); del V, e
        Phi = st["channels"][str(kap)]["Phi"].to(dev)
        log("kappa %d: short-chain vacuum %d occupied of %d; the hold's state %d columns" % (kap, Vn.shape[1], n2s, Phi.shape[1]))
        J = OPS.current_out_csr(hl, Rl, Rc_abs).tocoo()
        jsites = np.unique(np.concatenate([J.row // 2, J.col // 2]))
        inside = np.nonzero(Rl[:ns] <= Rc_abs)[0]
        S = np.unique(np.concatenate([inside, jsites, np.arange(lo_w, hi_w)]))
        assert S.max() < ns
        idx = np.sort(np.concatenate([2 * S, 2 * S + 1]))
        pos2 = {int(r): j for j, r in enumerate(idx)}
        Jss = np.zeros((len(idx), len(idx)), complex)
        for r_, c_, d_ in zip(J.row, J.col, J.data):
            Jss[pos2[int(r_)], pos2[int(c_)]] += d_
        Jss_t = torch.as_tensor(Jss, dtype=CD, device=dev)
        inn = np.array([pos2[int(r)] for r in np.sort(np.concatenate([2 * inside, 2 * inside + 1]))])
        inn_t = torch.as_tensor(inn, device=dev)
        Bh = OPS.Banded(hl, dev)
        W = torch.zeros((2 * len(Rl), len(idx)), dtype=CD, device=dev)
        W[torch.as_tensor(idx, device=dev), torch.arange(len(idx), device=dev)] = 1.0
        log("kappa %d: detector sites %d (inside %d, current %d, window %d), %d columns on %d rows; h bound %.4f" % (
            kap, len(S), len(inside), len(jsites), hi_w - lo_w, len(idx), W.shape[0], Bh.bound))

        def readout(W):
            Ws = W[:n2s]
            DW = Phi @ (Phi.mH @ Ws) - Vn @ (Vn.mH @ Ws)                        # Delta0 W_s
            B = Ws.mH @ DW                                                       # the block of C(t) - P_- on S
            L = float(torch.trace(Jss_t @ B).real)
            Win = W[:, inn_t]
            Y = Bh.apply(Win)[:n2s]                                              # (h W_in)_s
            I = float(torch.sum(Y.conj() * DW[:, inn_t]).real)
            return B, L, I

        rows = []; K_max = 0
        B, L, I = readout(W); I_last = I
        Bn = B.cpu().numpy()
        rr = np.concatenate([[pos2[2 * s], pos2[2 * s] + 1] for s in range(lo_w, hi_w)])
        sp0 = split_from_block(Bn[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
        rows.append({"j": 0, "t": tf, "t_over_rs": tf / rs, "L_at_t": L, "I_inside": I, "collar_split": sp0})
        blk = [Bn[np.ix_(rr, rr)]]
        for j in range(1, nrows + 1):
            for j0 in range(0, W.shape[1], a.chunk):
                W[:, j0:j0 + a.chunk], K, tail = cheb_plus(Bh, W[:, j0:j0 + a.chunk], trow, a.cheb_tol)
                K_max = max(K_max, K)
            B, L, I = readout(W)
            Bn = B.cpu().numpy()
            spj = split_from_block(Bn[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
            rows.append({"j": j, "t": tf + j * trow, "t_over_rs": (tf + j * trow) / rs, "L_at_t": L, "I_inside": I,
                         "collar_flow": -(I - I_last), "collar_split": spj, "cheb_K": K})
            I_last = I
            blk.append(Bn[np.ix_(rr, rr)])
            if j % 10 == 0:
                log("kappa %d row %d/%d T/r_s %.2f: L %.6e, flow %.6e, E_flux_out(32) %.6e (%.0f s)" % (
                    kap, j, nrows, rows[-1]["t_over_rs"], L, rows[-1]["collar_flow"], spj["windows"]["32"]["E_flux_out"], time.time() - t_c))
        res["channels"][str(kap)] = {"w": ops_s.w, "rows": rows, "cheb_K_max": K_max, "n_columns": len(idx), "seconds": time.time() - t_c}
        blocks["k%d" % kap] = np.stack(blk)
        del W, Phi, Vn, Bh
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    blocks["window_sites"] = np.arange(lo_w, hi_w) + 1
    np.savez_compressed(pre + "_blocks.npz", **blocks)
    json.dump(res, open(pre + "_rows.json", "w"))
    log("written %s_rows.json and _blocks.npz" % pre)
    if a.compare:
        compare(res, a.compare, pre)


def compare(res, path, pre):
    D = json.load(open(path))
    recs = D["records"]; rws = [r for r in D["rows"] if r.get("frozen")]
    lines = ["# %s against the stored hold %s" % (os.path.basename(pre), path), "",
             "Criterion (stated before the run): every |difference| of L_at_t, collar_flow and the windows' E_flux entries below 1e-6 of the row's L_H(kappa_T).", ""]
    worst = {"L_at_t": 0.0, "collar_flow": 0.0, "split": 0.0}; worst_rel = 0.0; n = 0
    for kap, ch in res["channels"].items():
        mine = ch["rows"]
        byt = {round(r["t"], 4): r for r in mine}
        for rec in recs:
            if rec.get("t_end") is None:
                continue
            m = byt.get(round(rec["t_end"], 4))
            pc = rec["per_channel"].get(kap)
            if m is None or pc is None or "collar_flow" not in m:
                continue
            for key in ("L_at_t", "collar_flow"):
                d = abs(m[key] - pc[key]); worst[key] = max(worst[key], d)
        for rw in rws:
            m = byt.get(round(rw["t"], 4)); pc = rw["per_channel"].get(kap, {}).get("collar_split")
            if m is None or not pc:
                continue
            comp = rw.get("L_H_kappaT") or rw.get("L_H_TH") or 1.0
            for L, wv in pc["windows"].items():
                for k_, x in wv.items():
                    if k_.startswith("E_flux") and k_ in m["collar_split"]["windows"][L] and x is not None:
                        d = abs(m["collar_split"]["windows"][L][k_] - x); worst["split"] = max(worst["split"], d)
                        worst_rel = max(worst_rel, d / comp); n += 1
    comp = min((r.get("L_H_kappaT") or r.get("L_H_TH") or 1.0) for r in rws)
    rel = max(worst["L_at_t"], worst["collar_flow"]) / comp
    ok = max(rel, worst_rel) < 1e-6
    lines.append("records compared per channel: %d; split entries compared: %d" % (len(recs), n))
    lines.append("largest |dL_at_t| %.3e, |d collar_flow| %.3e, |d split E_flux| %.3e; the smallest comparator L_H %.4e; largest relative %.3e" % (
        worst["L_at_t"], worst["collar_flow"], worst["split"], comp, max(rel, worst_rel)))
    lines.append(""); lines.append("VERDICT: %s" % ("PASS" if ok else "FAIL"))
    open(pre + "_compare.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
