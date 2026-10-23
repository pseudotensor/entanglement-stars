#!/usr/bin/env python
"""page_graft_pullback_v2.py (the solve lane's analyst builder 10; engine spec 52.4 (2), (3)): page_graft_pullback_v1 (imported read-only
for cheb_plus and split_from_block; its readout arithmetic repeated here in the same order, so that with the options below off every row is
version 1's bitwise) plus:

(a) the join's null readout (51.2 (2), 52.4 (2)): Tr(O(t) D_sigma) beside every row, D_sigma = f(H_walled) - f(H_connected), H_connected
    the long chain's h, H_walled the same with every entry coupling a site below --nsites-short to one at or above it removed (the two walled
    vacua, short and tail, against the connected one), f(E) = erfc(E/(2 sigma))/2 the smoothed negative-energy step whose derivative is
    P03 A8's Gaussian exp(-E^2/(4 sigma^2)) (unit weight), applied by its degree-M Chebyshev interpolant on [-Lambda, Lambda] (Lambda the
    Gershgorin bound of the long h). A polynomial of degree M in a banded h of index range r changes only within M r indices of the join,
    so p(H_walled) - p(H_connected) is computed exactly on the window of the join +- 2 (M + 1) r indices (the true chain ends where nearer).
    Printed beside every row: sigma (in lattice units and in T_H of the record), M, the filter's remainder eps = sum_{n > M} |c_n| (the
    Chebyshev coefficients computed to 4 M on a 16 M-point grid), each readout's remainder bound (2 eps ||O|| ||X||_F^2 with X the window's
    rows of the pulled-back columns), and the same readouts at 2 sigma (the resolution's dependence). The readouts: L_null = Re Tr(J_SS B_D),
    the inside energy I_null, the windowed split's entries, B_D = X^dagger D_sigma X. With --append 0 the walled and connected chains are one
    and the readout is 0 exactly (the gate's zero).
(b) the truncation (52.4 (2)): Delta0 = Phi Phi^dagger - P_-^short = V Lambda V^dagger once per channel; the pairs with the smallest |lambda|
    are dropped while their sum stays below --trunc-drop-sum; the readouts then use W^dagger Delta0 W = (V_r^dagger W)^dagger Lambda_r
    (V_r^dagger W). Its certificate per readout (W an isometry): |dL| <= s ||J_SS||_2, |dI| <= s ||h||, the split's |dE_flux| <= s times the
    window's sum of |omega v| weights (2/sum w^2), s the dropped sum; the truncation is taken for a channel only where every certificate is
    under --trunc-digit-frac (1e-3) times L_H(kappa_T) (--LH, or the smallest L_H_kappaT of --compare's frozen rows), else the full product
    is used and the row says so.
(c) W's columns saved every --ckpt-every rows (one copy, <tag>_ckpt.pt, written to a temporary file then renamed), with the rows and blocks
    so far; --resume continues from it (a channel already finished is kept).

Outputs (rule 10): <tag>_rows.json (every row: version 1's keys, plus "null" and "trunc"), <tag>_blocks.npz (version 1's), and with
--compare the version 1 compare (<tag>_compare.md); --compare-rows a version 1 rows json to compare bitwise (<tag>_v1cmp.md).
Usage: as version 1 plus the options below."""
import argparse
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
import page_graft_pullback_v1 as G1   # noqa: E402  (read-only: cheb_plus, split_from_block, compare)
PK = G1.PK
OPS = G1.OPS
CD = G1.CD
T0 = time.time()


def log(m):
    print("[%7.1f s] %s" % (time.time() - T0, m), flush=True)


def step_coeffs(sigma, lam, tol=1e-13):
    """Chebyshev coefficients of f(E) = erfc(E/(2 sigma))/2 on [-lam, lam]; returns (c[0..M], M, eps = sum_{n>M} |c_n| from the
    coefficients computed to 4 M on a 16 M-point grid)."""
    from scipy.special import erfc
    M0 = int(16 * lam / sigma) + 64
    for _ in range(4):
        N = 16 * M0
        th = math.pi * (np.arange(N) + 0.5) / N
        fx = 0.5 * erfc(lam * np.cos(th) / (2.0 * sigma))
        c = _dct_coeffs(fx, N, 4 * M0)
        c[0] *= 0.5
        tail = np.concatenate([np.cumsum(np.abs(c[::-1]))[::-1][1:], [0.0]])
        ok = np.nonzero(tail < tol)[0]
        if len(ok) and ok[0] < 2 * M0:
            M = int(ok[0]); return c[:M + 1], M, float(tail[M])
        M0 *= 2
    M = len(c) - 1
    return c, M, float(tail[-1])


def _dct_coeffs(fx, N, nmax):
    from scipy.fft import dct
    c = dct(fx, type=2) / N           # sum_k f_k cos(pi n (k + 1/2)/N) * 2 / N
    return c[:nmax + 1].copy()


def cheb_filter_pair(Bw, Bc, X, coeffs_list, lam):
    """sum_n c_n (T_n(Hw/lam) - T_n(Hc/lam)) X for each coefficient vector in coeffs_list (same length); returns the list of results."""
    K = len(coeffs_list[0]) - 1
    out = []
    for B in (Bw, Bc):
        T0_ = X; T1_ = B.apply(X) / lam
        Ys = [complex(c[0]) * T0_ + complex(c[1]) * T1_ for c in coeffs_list]
        for k in range(2, K + 1):
            T2_ = 2.0 * B.apply(T1_) / lam - T0_
            for Y, c in zip(Ys, coeffs_list):
                Y.add_(complex(c[k]) * T2_)
            T0_, T1_ = T1_, T2_
        out.append(Ys)
    return [a - b for a, b in zip(out[0], out[1])]


def walled(A, ns):
    """A (CSR on interleaved 2-component sites) with every entry between a site < ns and a site >= ns removed."""
    A = A.tocoo()
    keep = (A.row < 2 * ns) == (A.col < 2 * ns)
    return sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape)


def split_weight(R, N, v, ic, kappa, L):
    """the windowed split's weight sum_q (|omega_+ v_+| + |omega_- v_-|) * 2 / sum w^2 (the certificate factor of an operator-norm error)."""
    Rc, Nc, vc = float(R[ic]), float(N[ic]), float(v[ic])
    n = len(R); hL = L // 2; i0 = ic - hL; i1 = i0 + L; lo, hi = max(0, i0), min(n, i1)
    gw = np.sin(math.pi * (np.arange(L) + 0.5) / L)[lo - i0:hi - i0]
    M = 4 * L; q = 2.0 * math.pi * np.fft.fftfreq(M)
    s_ = np.sin(q); c_ = np.cos(q); kr = kappa / Rc; E = np.sqrt(s_ ** 2 + kr ** 2)
    om_p = Nc * E + 0.5 * vc * np.sin(2.0 * q); om_m = -Nc * E + 0.5 * vc * np.sin(2.0 * q)
    vp = Nc * s_ * c_ / E + vc * np.cos(2.0 * q); vm = -Nc * s_ * c_ / E + vc * np.cos(2.0 * q)
    return float(np.sum(np.abs(om_p * vp) + np.abs(om_m * vm))) * 2.0 / float(np.sum(gw ** 2))


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
    ap.add_argument("--compare-rows", default=None, help="a version 1 (or 2) rows json: every common row compared number by number")
    ap.add_argument("--null", type=int, default=1, help="1: the join's null readout beside every row; 0: off")
    ap.add_argument("--null-every", type=int, default=1, help="the null readout on every this many rows (1: every row)")
    ap.add_argument("--null-sigma-TH", type=float, default=8.0, help="the filter's resolution sigma in T_H of the frozen record")
    ap.add_argument("--null-tol", type=float, default=1e-12)
    ap.add_argument("--trunc", type=int, default=0, help="1: the certified truncation of Delta0")
    ap.add_argument("--trunc-drop-sum", type=float, default=0.0, help="the dropped pairs' |lambda| sum; 0: half the digit over the largest certificate factor")
    ap.add_argument("--trunc-digit-frac", type=float, default=1e-3); ap.add_argument("--LH", type=float, default=None)
    ap.add_argument("--ckpt-every", type=int, default=100); ap.add_argument("--resume", type=int, default=0)
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
    T_H = 1.0 / (4.0 * math.pi * rs)
    LH = a.LH
    if LH is None and a.compare:
        D_ = json.load(open(a.compare))
        LH = min((r.get("L_H_kappaT") or r.get("L_H_TH") or 1.0) for r in D_["rows"] if r.get("frozen"))
    digit = a.trunc_digit_frac * LH if LH else None
    log("graft pull-back v2: record kf %d T/r_s %.4f, short chain %d sites, appended %d (long chain %d), row every %d steps (%.1f a), %d rows; R_collar %.1f, centre site %d; null %d (sigma %.3g T_H), trunc %d (drop sum %.1e, digit %s)" % (
        kf, tf / rs, ns, a.append, len(Rl), every, trow, nrows, Rc_abs, ic + 1, a.null, a.null_sigma_TH, a.trunc, a.trunc_drop_sum, digit))
    res = {"settings": vars(a), "kf": kf, "T_freeze": tf, "rs": rs, "tau": tau, "row_time": trow, "T_H": T_H, "LH_digit": digit, "channels": {},
           "join_null_readout": ("D_sigma = f(H_walled) - f(H_connected), f(E) = erfc(E/(2 sigma))/2 (P03 A8's Gaussian as its derivative), by its Chebyshev interpolant on the join's window; remainder and the 2 sigma readout printed" if a.null else "off"),
           "label": "the held-field control's readouts as pulled-back detector columns" + (" on the graft (short chain's deviation in the long chain's Killing vacuum)" if a.append else " on the short chain (the gate)")}
    blocks = {}
    ck = None; ckp = pre + "_ckpt.pt"
    if a.resume and os.path.exists(ckp):
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        res["channels"] = ck["done"]; blocks.update(ck["done_blocks"])
        log("resumed from %s: channels done %s; channel %s at row %d" % (ckp, list(ck["done"].keys()), ck.get("kap"), ck.get("j", -1)))
    for kap in [int(x) for x in a.channels.split(",") if x.strip()]:
        if str(kap) in res["channels"]:
            continue
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
        J_norm = float(np.linalg.norm(Jss, 2))
        inn = np.array([pos2[int(r)] for r in np.sort(np.concatenate([2 * inside, 2 * inside + 1]))])
        inn_t = torch.as_tensor(inn, device=dev)
        Bh = OPS.Banded(hl, dev)
        log("kappa %d: detector sites %d (inside %d, current %d, window %d), %d columns on %d rows; h bound %.4f; ||J_SS|| %.4e" % (
            kap, len(S), len(inside), len(jsites), hi_w - lo_w, len(idx), 2 * len(Rl), Bh.bound, J_norm))
        # (b) the truncation
        trunc = {"taken": False}
        Vr = Lr = None
        if a.trunc:
            Dl0 = Phi @ Phi.mH - Vn @ Vn.mH
            Dl0 = 0.5 * (Dl0 + Dl0.mH)
            lam, U = torch.linalg.eigh(Dl0); del Dl0
            lam_np = lam.cpu().numpy(); order = np.argsort(np.abs(lam_np)); cs = np.cumsum(np.abs(lam_np[order]))
            wsplit = split_weight(Rl, Nl, vl, ic, kap, max(wins))
            budget = a.trunc_drop_sum if a.trunc_drop_sum > 0 else (0.5 * digit / max(J_norm, Bh.bound, wsplit) if digit else 0.0)
            ndrop = int(np.searchsorted(cs, budget, side="right")); sdrop = float(cs[ndrop - 1]) if ndrop else 0.0
            keep = np.sort(order[ndrop:])
            cert = {"L_at_t": sdrop * J_norm, "I_inside": sdrop * Bh.bound, "split_E_flux": sdrop * wsplit}
            ok = digit is not None and all(x < digit for x in cert.values())
            trunc = {"taken": bool(ok), "kept_pairs": int(len(keep)), "dropped_pairs": int(ndrop), "dropped_sum": sdrop,
                     "largest_dropped": float(np.abs(lam_np[order[ndrop - 1]])) if ndrop else 0.0, "certificate": cert, "digit": digit,
                     "split_weight": wsplit}
            if ok:
                kt = torch.as_tensor(keep, device=dev)
                Vr = U[:, kt].contiguous(); Lr = lam[kt].to(CD)
            del U, lam
            log("kappa %d: truncation %s: keep %d of %d pairs, dropped sum %.3e; certificates L %.3e, I %.3e, split %.3e against the digit %s" % (
                kap, "TAKEN" if ok else "NOT taken", len(keep), n2s, sdrop, cert["L_at_t"], cert["I_inside"], cert["split_E_flux"], digit))
        # (a) the null readout's operators
        nul = None
        if a.null:
            sig = a.null_sigma_TH * T_H; lam_b = Bh.bound * (1.0 + 1e-6)
            c1, M1, eps1 = step_coeffs(sig, lam_b, a.null_tol)
            c2, M2, eps2 = step_coeffs(2.0 * sig, lam_b, a.null_tol)
            c2p = np.zeros(M1 + 1, complex); c2p[:min(M1, M2) + 1] = c2[:min(M1, M2) + 1]
            eps2 = eps2 + (float(np.sum(np.abs(c2[M1 + 1:]))) if M2 > M1 else 0.0)
            A = hl.tocsr(); rr_ = max(abs(d) for d, _ in Bh.diags)
            half = 2 * (M1 + 1) * rr_
            lo_j = max(0, n2s - half); hi_j = min(A.shape[0], n2s + half)
            Aw = A[lo_j:hi_j, lo_j:hi_j]
            Bc = OPS.Banded(Aw, dev, dmax=max(16, rr_)); Bwl = OPS.Banded(walled(Aw, ns - lo_j // 2), dev, dmax=max(16, rr_))
            nul = {"sigma": sig, "sigma_TH": a.null_sigma_TH, "M": M1, "eps": eps1, "M_2sigma": M2, "eps_2sigma": eps2, "index_range": rr_,
                   "window_rows": [int(lo_j), int(hi_j)], "Lambda": lam_b, "walled_equals_connected": bool(a.append == 0)}
            dshort = None
            if a.append:
                dshort = float(abs(walled(A, ns)[:n2s, :n2s] - hs).max())
                nul["short_block_vs_short_h_max_abs"] = dshort
            log("kappa %d: null readout: sigma %.4e (%.2f T_H), M %d, eps %.2e (2 sigma: M %d, eps %.2e); index range %d, window rows %d..%d%s" % (
                kap, sig, a.null_sigma_TH, M1, eps1, M2, eps2, rr_, lo_j, hi_j, "" if dshort is None else ", walled short block vs short h %.2e" % dshort))
            cfs = [c1, c2p]

        def readout(W):
            Ws = W[:n2s]
            if Vr is not None:
                Z = Vr.mH @ Ws
                DW = Vr @ (Lr[:, None] * Z)                                   # Delta0_r W_s
                B = Z.mH @ (Lr[:, None] * Z)
            else:
                DW = Phi @ (Phi.mH @ Ws) - Vn @ (Vn.mH @ Ws)                        # Delta0 W_s (version 1's order)
                B = Ws.mH @ DW                                                       # the block of C(t) - P_- on S
            L = float(torch.trace(Jss_t @ B).real)
            Win = W[:, inn_t]
            Y = Bh.apply(Win)[:n2s]                                              # (h W_in)_s
            I = float(torch.sum(Y.conj() * DW[:, inn_t]).real)
            return B, L, I

        def null_readout(W, rrw):
            X = W[lo_j:hi_j]
            xn = float(torch.linalg.norm(X))
            if xn == 0.0 or a.append == 0:
                z = {"L_null": 0.0, "I_null": 0.0, "L_null_2sigma": 0.0, "I_null_2sigma": 0.0, "E_flux_out_32_null": 0.0,
                     "X_window_norm2": xn ** 2, "rem_L": 2 * eps1 * J_norm * xn ** 2, "rem_block": 2 * eps1 * xn ** 2,
                     "exact_zero": a.append == 0 or xn == 0.0}
                return z
            DX = [torch.zeros_like(X), torch.zeros_like(X)]
            for j0 in range(0, X.shape[1], a.chunk):
                Xc = X[:, j0:j0 + a.chunk]
                if float(torch.linalg.norm(Xc)) == 0.0:
                    continue
                r = cheb_filter_pair(Bwl, Bc, Xc, cfs, lam_b)
                DX[0][:, j0:j0 + a.chunk] = r[0]; DX[1][:, j0:j0 + a.chunk] = r[1]
            Yh = Bh.apply(W[:, inn_t])[lo_j:hi_j]
            out = {"X_window_norm2": xn ** 2}
            for tagz, Dx in (("", DX[0]), ("_2sigma", DX[1])):
                BD = X.mH @ Dx
                out["L_null" + tagz] = float(torch.trace(Jss_t @ BD).real)
                out["I_null" + tagz] = float(torch.sum(Yh.conj() * Dx[:, inn_t]).real)
                BDn = BD.cpu().numpy()
                spn = G1.split_from_block(BDn[np.ix_(rrw, rrw)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
                out["E_flux_out_32_null" + tagz] = spn["windows"]["32"]["E_flux_out"]
                out["split_null" + tagz] = {L_: {k_: v_ for k_, v_ in w_.items() if k_.startswith("E_flux") or k_.startswith("n_flux")} for L_, w_ in spn["windows"].items()}
            out["rem_L"] = 2 * eps1 * J_norm * xn ** 2
            out["rem_I"] = 2 * eps1 * float(torch.linalg.norm(Yh)) * float(torch.linalg.norm(X[:, inn_t]))
            out["rem_block"] = 2 * eps1 * xn ** 2
            return out

        rr = np.concatenate([[pos2[2 * s], pos2[2 * s] + 1] for s in range(lo_w, hi_w)])
        if ck is not None and ck.get("kap") == kap:
            W = ck["W"].to(dev); rows = ck["rows"]; blk = list(ck["blk"]); I_last = ck["I_last"]; K_max = ck["K_max"]; j_start = ck["j"] + 1
            log("kappa %d: resumed at row %d" % (kap, j_start))
        else:
            W = torch.zeros((2 * len(Rl), len(idx)), dtype=CD, device=dev)
            W[torch.as_tensor(idx, device=dev), torch.arange(len(idx), device=dev)] = 1.0
            rows = []; K_max = 0
            B, L, I = readout(W); I_last = I
            Bn = B.cpu().numpy()
            sp0 = G1.split_from_block(Bn[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
            row0 = {"j": 0, "t": tf, "t_over_rs": tf / rs, "L_at_t": L, "I_inside": I, "collar_split": sp0}
            if nul is not None:
                row0["null"] = null_readout(W, rr)
            row0["trunc"] = bool(Vr is not None)
            rows.append(row0)
            blk = [Bn[np.ix_(rr, rr)]]
            j_start = 1
        for j in range(j_start, nrows + 1):
            t_r = time.time()
            for j0 in range(0, W.shape[1], a.chunk):
                W[:, j0:j0 + a.chunk], K, tail = G1.cheb_plus(Bh, W[:, j0:j0 + a.chunk], trow, a.cheb_tol)
                K_max = max(K_max, K)
            B, L, I = readout(W)
            Bn = B.cpu().numpy()
            spj = G1.split_from_block(Bn[np.ix_(rr, rr)], list(range(lo_w, hi_w)), ic, Rl, Nl, vl, kap, wins)
            row = {"j": j, "t": tf + j * trow, "t_over_rs": (tf + j * trow) / rs, "L_at_t": L, "I_inside": I,
                   "collar_flow": -(I - I_last), "collar_split": spj, "cheb_K": K}
            t_n = time.time()
            if nul is not None and j % a.null_every == 0:
                row["null"] = null_readout(W, rr)
            row["trunc"] = bool(Vr is not None)
            row["seconds"] = time.time() - t_r; row["seconds_null"] = time.time() - t_n
            rows.append(row)
            I_last = I
            blk.append(Bn[np.ix_(rr, rr)])
            if j % 10 == 0:
                nz = row.get("null", {})
                log("kappa %d row %d/%d T/r_s %.2f: L %.6e, flow %.6e, E_flux_out(32) %.6e; null L %.3e (2s %.3e, rem %.1e) (%.1f s per row, null %.1f s; %.0f s)" % (
                    kap, j, nrows, row["t_over_rs"], L, row["collar_flow"], spj["windows"]["32"]["E_flux_out"],
                    nz.get("L_null", float("nan")), nz.get("L_null_2sigma", float("nan")), nz.get("rem_L", float("nan")),
                    row["seconds"], row["seconds_null"], time.time() - t_c))
            if a.ckpt_every and j % a.ckpt_every == 0 and j < nrows:
                tmp = ckp + ".tmp"
                torch.save({"kap": kap, "j": j, "W": W.cpu(), "rows": rows, "blk": np.stack(blk), "I_last": I_last, "K_max": K_max,
                            "done": res["channels"], "done_blocks": blocks}, tmp)
                os.replace(tmp, ckp)
                log("kappa %d: checkpoint at row %d written (%s)" % (kap, j, ckp))
        res["channels"][str(kap)] = {"w": ops_s.w, "rows": rows, "cheb_K_max": K_max, "n_columns": len(idx), "seconds": time.time() - t_c,
                                     "null": nul, "trunc": trunc, "J_SS_norm": J_norm}
        blocks["k%d" % kap] = np.stack(blk)
        ck = None
        del W, Phi, Vn, Bh
        if Vr is not None:
            del Vr, Lr
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    blocks["window_sites"] = np.arange(lo_w, hi_w) + 1
    np.savez_compressed(pre + "_blocks.npz", **blocks)
    json.dump(res, open(pre + "_rows.json", "w"))
    log("written %s_rows.json and _blocks.npz" % pre)
    if os.path.exists(ckp):
        os.remove(ckp)
    if a.compare:
        G1.compare(res, a.compare, pre)
    if a.compare_rows:
        compare_rows(res, a.compare_rows, pre, digit)


def compare_rows(res, path, pre, digit):
    """every common row's L_at_t, I_inside, collar_flow and window E_flux entries against another rows json: the largest |difference|,
    the count of exactly equal numbers, and (for the truncation) the largest difference against the digit."""
    D = json.load(open(path))
    n = 0; neq = 0; worst = 0.0; where = None
    for kap, ch in res["channels"].items():
        oth = {r["j"]: r for r in D["channels"].get(kap, {}).get("rows", [])}
        for r in ch["rows"]:
            o = oth.get(r["j"])
            if o is None:
                continue
            pairs = [(k, r.get(k), o.get(k)) for k in ("L_at_t", "I_inside", "collar_flow")]
            for L_, w_ in r["collar_split"]["windows"].items():
                for k_, x in w_.items():
                    if isinstance(x, float):
                        pairs.append(("split %s %s" % (L_, k_), x, o["collar_split"]["windows"][L_][k_]))
            for k, x, y in pairs:
                if x is None or y is None:
                    continue
                n += 1; d = abs(x - y)
                neq += (x == y)
                if d > worst:
                    worst = d; where = (kap, r["j"], k)
    lines = ["# %s against %s, row by row" % (os.path.basename(pre), path), "",
             "numbers compared: %d; exactly equal: %d; largest |difference| %.3e at %s; the digit 1e-3 L_H %s" % (n, neq, worst, where, digit),
             "", "BITWISE: %s" % ("yes" if n and neq == n else "no"),
             "WITHIN THE DIGIT: %s" % ("yes" if digit is not None and worst < digit else "no")]
    open(pre + "_v1cmp.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
