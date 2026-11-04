#!/usr/bin/env python
"""engine_dust_root_history_v1b.py (the lane's analyst builder; engine spec 42.3 (4)): version 1 with (1) the applied outer root taken from the
record's applied site fields (the largest zero of N + v on the engine's sites with positive slope, linear between sites), where version 1 took
the record's trapping zeros, which are the dense fields' roots (checked: at 12.125 and 15.0 the dense roots 38.5562 and 51.7604 equal the
record's zeros, so version 1's two cuts coincided; no fallback occurred); (2) the jump ledger sampled every --jump-every-rs: at a record k
the dust's weight above m in h(k+1)'s Killing basis minus that in h(k)'s, both of the state at t_{k+1} (the spectral projector's jump at the
field jump), occupation-weighted and per column, labelled "sampled every J r_s, not summed over every jump". HORIZON version 2's columns 1
and 7 remain outside this version. Version 1's docstring follows.
engine_dust_root_history_v1.py (the lane's analyst builder; inbox #62 item 4, engine spec 42.3 (4)): the shed dust's history against the
moving root, the producer of DUST_root_history_v1 (saves every per-column, per-record array; the compose is a separate script). On the
production's solved history (C3prod19 r11 to T/r_s 40, B's continuation C3ext19 after it; engine_single_object_B_v2's CombinedHistory), from
HORIZON B.5 version 2's dust-tagged occupied columns (engine_single_object_B5_prep_v2's npz: the stored slices' dust columns with their
occupations f at T/r_s 12.125 and 15.0, kappa 1 and 2), per launch:
- the tags: each column split at the launch record by the dense outer root (B v1's dense_roots on the record's sources) and by the applied
  outer root (the record's largest trapping zero): X_in = Pi_{R < R_o} X and X_out = X - X_in for each cut; X, X_in(dense) and X_in(applied)
  propagated record by record on the stored history by B's Chebyshev action (piecewise-constant applied fields) to --T-end-rs (56), X_out by
  linearity (U X - U X_in), so the dust's weight at a later clock splits exactly into the part from inside the root, the part from outside it
  and their cross term (printed separately);
- per record and per column (arrays saved): the weight beyond 16 and 24 r_s of X, X_in and X_out for both cuts; the weight beyond the
  applied outer root R_t(t) and inside it; the clock of the record and the root;
- per record (occupation-weighted, X): the fixed-cut number currents Tr[X^dag i[h, Pi_{R > c}] X] at the outer and inner applied roots and at
  16 r_s, and the moving-cut reassignment over the step at R_t (the weight beyond R_t(t_{k+1}) minus beyond R_t(t_k), both of the state at
  t_{k+1}), separate columns, never summed;
- the Killing weights at the launch record and at the end record (the eigendecomposition of the record's h on the chain) per column in the
  five bins split at -b, -m, m and b, b = sqrt(m^2 + 1/a^2) (the sine chain's asymptotic band edge; weight above b has no propagating partner
  at infinity and is never read as unbound), and the mean Killing energy against m;
- per column the outward crossing clock (the first record at which its weight beyond R_t(t) exceeds 1/2) beside the first closing clock
  after the launch (the first record with no applied trapping zero).
Not in version 1 (said so in the json): the per-record change of the weight above m at the field jumps (the spectral projector's jump ledger)
and the same tagging on HORIZON version 2's columns 1 and 7. No mutual information. Refuses to overwrite.
Usage (box): python engine_dust_root_history_v1.py --columns-npz S4one/v2/B5v2_cols_prod.npz <B v2's history args> --T-end-rs 56 --out-json J --out-npz Z"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_B_v2 as BV2   # noqa: E402  (never edited)

B1, OPS, PH, CD = BV2.B1, BV2.OPS, BV2.PH, BV2.CD


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg), flush=True)


def site_w(X, n):
    return (X.abs() ** 2).reshape(n, 2, -1).sum(dim=1).cpu().numpy()      # sites x columns


def killing_bins(e, U, X, m, b):
    c = U.conj().T @ X
    w = (c.abs() ** 2).cpu().numpy(); ee = e.cpu().numpy()
    edges = [-np.inf, -b, -m, m, b, np.inf]
    bins = np.array([w[(ee > edges[i]) & (ee <= edges[i + 1])].sum(axis=0) for i in range(5)])   # 5 x columns
    mean = (ee[:, None] * w).sum(axis=0)
    return bins, mean


def applied_roots(rec):
    z = rec.get("trapping_zeros") or []
    if z:
        return max(z), (min(z) if len(z) > 1 else None)
    return rec.get("R_t"), rec.get("r_in")


def applied_outer_root(H, k):
    """the largest positive-slope zero of N + v of the record's applied site fields on the engine's sites (linear between sites)."""
    N, v = H.fields(k); ne = H.ne; f = N[:ne] + v[:ne]; R = H.R[:ne]
    idx = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(idx) == 0:
        return None
    i = idx[-1]
    return float(R[i] - f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns-npz", required=True)
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True)
    ap.add_argument("--record2", default=None); ap.add_argument("--fields2", default=None); ap.add_argument("--switch-T-rs", type=float, default=40.0)
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--T-end-rs", type=float, default=56.0); ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--jump-every-rs", type=float, default=4.0)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--out-json", required=True); ap.add_argument("--out-npz", required=True)
    a = ap.parse_args()
    for p in (a.out_json, a.out_npz):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    H = BV2.CombinedHistory(a.record, a.fields, a.nsites, a.record2, a.fields2, a.switch_T_rs if a.record2 else None)
    rs = H.rs_nom; g = H.g; m = a.m; b = math.sqrt(m * m + 1.0); R = H.R; n = len(R)
    TT = np.asarray(H.T, float) / rs
    kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
    O = np.load(a.columns_npz)
    tags = sorted({k_.split("_")[2] for k_ in O.files if k_.startswith("cols_dust_")})
    out = {"script": "engine_dust_root_history_v1b.py", "columns_npz": a.columns_npz, "columns_npz_md5": BV2.md5(a.columns_npz), "files": H.files, "m": m, "b": b,
           "T_end_over_rs": TT[kend], "not_in_v1b": ["the tagging on HORIZON version 2's columns 1 and 7", "the jump ledger at every record (sampled here)"],
           "bins": ["(-inf, -b]", "(-b, -m]", "(-m, m]", "(m, b]", "(b, inf)"], "launches": [], "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    arrays = {}
    first_closing = {}
    for tg in tags:
        T_rs = float(O["T_%s" % tg]); k0 = int(np.argmin(np.abs(TT - T_rs)))
        kc = next((k for k in range(k0 + 1, len(TT)) if not (H.rec[k].get("trapping_zeros") or H.rec[k].get("R_t"))), None)
        first_closing[tg] = TT[kc] if kc is not None else None
        rd = BV2.roots_at(H, k0, a.fields_module)
        Ro_d, Ri_d, _ = BV2.outer_inner_of(rd, H.rec[k0])
        Ro_a = applied_outer_root(H, k0); Ri_a = applied_roots(H.rec[k0])[1]
        for kap in [int(x) for x in a.kappas.split(",")]:
            kd = "cols_dust_%s_k%d" % (tg, kap)
            if kd not in O.files:
                continue
            t_c = time.time(); key = "%s_k%d" % (tg, kap)
            A_ = np.asarray(O[kd], np.complex128); f = np.asarray(O["f_dust_%s_k%d" % (tg, kap)], float)
            W = np.zeros((2 * n, A_.shape[1]), complex); W[: A_.shape[0]] = A_
            X = torch.as_tensor(W, dtype=CD, device=dev)
            S = {"all": X}
            for cut, Ro in (("dense", Ro_d), ("applied", Ro_a)):
                msk = torch.as_tensor(np.repeat(R < (Ro or 0.0), 2), device=dev)
                S["in_" + cut] = X * msk[:, None]
            inside0 = {cut: float(np.dot(f, site_w(S["in_" + cut], n).sum(axis=0))) for cut in ("dense", "applied")}
            h0, Q = OPS.chain_sparse(R, m, float(kap))
            N0, v0 = H.fields(k0)
            e, U = OPS.eigh_csr(OPS.clocked_csr(h0, Q, N0, v0), dev)
            bins0, mean0 = killing_bins(e, U, X, m, b); del e, U
            nr = kend - k0 + 1; nc = X.shape[1]
            arr = {nm: {c: np.zeros((nr, nc)) for c in ("b16", "b24")} for nm in ("all", "in_dense", "out_dense", "in_applied", "out_applied")}
            beyond_Rt = np.full((nr, nc), np.nan); Rt_arr = np.full(nr, np.nan); Ri_arr = np.full(nr, np.nan)
            cur = {c: np.full(nr, np.nan) for c in ("outer", "inner", "r16")}; mov = np.full(nr, np.nan)
            Sprev = None; Kmax = 0; jumps = []
            for j, kk in enumerate(range(k0, kend + 1)):
                if kk > k0:
                    Bk = OPS.Banded(OPS.clocked_csr(h0, Q, *H.fields(kk - 1)), dev); tau = float(H.T[kk] - H.T[kk - 1])
                    for nm in S:
                        S[nm], K, _ = PH.cheb_apply(Bk, S[nm], tau, a.cheb_tol); Kmax = max(Kmax, K)
                if kk > k0 and kk < kend and ((kk - k0) % max(1, int(round(a.jump_every_rs * rs / H.tau))) == 0):
                    ea, Ua = OPS.eigh_csr(OPS.clocked_csr(h0, Q, *H.fields(kk - 1)), dev); ba, _ = killing_bins(ea, Ua, S["all"], m, b); del ea, Ua
                    eb, Ub = OPS.eigh_csr(OPS.clocked_csr(h0, Q, *H.fields(kk)), dev); bb, _ = killing_bins(eb, Ub, S["all"], m, b); del eb, Ub
                    jumps.append({"t_over_rs": float(TT[kk]), "above_m_jump_weighted": float(np.dot(f, (bb[3] + bb[4]) - (ba[3] + ba[4]))),
                                  "above_m_jump_per_column": ((bb[3] + bb[4]) - (ba[3] + ba[4])).tolist(), "bins_before": ba.T.tolist(), "bins_after": bb.T.tolist()})
                Wall = site_w(S["all"], n)
                for cut in ("dense", "applied"):
                    Win = site_w(S["in_" + cut], n); Wout = site_w(S["all"] - S["in_" + cut], n)
                    for nm, Wx in (("in_" + cut, Win), ("out_" + cut, Wout)):
                        arr[nm]["b16"][j] = Wx[R > 16 * rs].sum(axis=0); arr[nm]["b24"][j] = Wx[R > 24 * rs].sum(axis=0)
                arr["all"]["b16"][j] = Wall[R > 16 * rs].sum(axis=0); arr["all"]["b24"][j] = Wall[R > 24 * rs].sum(axis=0)
                Rt, Ri = applied_roots(H.rec[kk]); Rt_arr[j] = Rt if Rt else np.nan; Ri_arr[j] = Ri if Ri else np.nan
                if Rt:
                    beyond_Rt[j] = Wall[R > Rt].sum(axis=0)
                    if Sprev is not None and not np.isnan(Rt_arr[j - 1]):
                        mov[j] = float(np.dot(f, Wall[R > Rt].sum(axis=0) - Wall[R > Rt_arr[j - 1]].sum(axis=0)))
                hn = OPS.clocked_csr(h0, Q, *H.fields(kk))
                for c, cutR in (("outer", Rt), ("inner", Ri), ("r16", 16 * rs)):
                    if cutR:
                        JX = OPS.Banded(OPS.number_current_csr(hn, R, cutR), dev).apply(S["all"])
                        cur[c][j] = float(np.dot(f, (S["all"].conj() * JX).sum(dim=0).real.cpu().numpy()))
                Sprev = Wall
            Nn, vn = H.fields(kend)
            e, U = OPS.eigh_csr(OPS.clocked_csr(h0, Q, Nn, vn), dev)
            binsE, meanE = killing_bins(e, U, S["all"], m, b); del e, U
            tt = TT[k0: kend + 1]
            cross = [next((float(tt[i]) for i in range(nr) if beyond_Rt[i, c] > 0.5), None) for c in range(nc)]
            for nm, d in arr.items():
                for c, v in d.items():
                    arrays["%s_%s_%s" % (key, nm, c)] = v
            arrays["%s_beyond_Rt" % key] = beyond_Rt; arrays["%s_T" % key] = tt; arrays["%s_Rt" % key] = Rt_arr; arrays["%s_Ri" % key] = Ri_arr
            arrays["%s_bins_launch" % key] = bins0; arrays["%s_bins_end" % key] = binsE; arrays["%s_f" % key] = f
            for c, v in cur.items():
                arrays["%s_current_%s" % (key, c)] = v
            arrays["%s_moving_cut_step" % key] = mov
            fw = lambda v: float(np.dot(f, v))
            L_ = {"key": key, "tag": tg, "kappa": kap, "T_launch_over_rs": TT[k0], "k0": k0, "k_end": kend, "f": f.tolist(),
                  "roots_launch": {"dense_outer": Ro_d, "dense_inner": Ri_d, "applied_outer": Ro_a, "applied_inner": Ri_a},
                  "first_closing_after_launch_over_rs": first_closing[tg],
                  "inside_root_at_launch": inside0, "occupation_sum": float(f.sum()),
                  "end": {nm: {"b16": fw(arr[nm]["b16"][-1]), "b24": fw(arr[nm]["b24"][-1])} for nm in arr},
                  "cross_term_b16_end": {cut: fw(arr["all"]["b16"][-1] - arr["in_" + cut]["b16"][-1] - arr["out_" + cut]["b16"][-1]) for cut in ("dense", "applied")},
                  "killing_launch_per_column": bins0.T.tolist(), "killing_end_per_column": binsE.T.tolist(),
                  "mean_killing_launch": mean0.tolist(), "mean_killing_end": meanE.tolist(),
                  "outward_crossing_clock_per_column": cross, "jump_ledger_sampled": {"every_rs": a.jump_every_rs, "label": "sampled every %g r_s, not summed over every jump" % a.jump_every_rs, "rows": jumps}, "cheb_K_max": Kmax, "seconds": time.time() - t_c}
            out["launches"].append(L_)
            log("%s: f %s; roots dense %.3f applied %s; the dust beyond 16 r_s at %.3f: all %.4e, from inside the dense root %.4e, outside %.4e, cross %.2e; "
                "from inside the applied root %.4e, outside %.4e; crossing clocks %s; first closing %s; %.0f s" % (
                    key, np.round(f, 4).tolist(), Ro_d or float("nan"), Ro_a, TT[kend], L_["end"]["all"]["b16"], L_["end"]["in_dense"]["b16"], L_["end"]["out_dense"]["b16"],
                    L_["cross_term_b16_end"]["dense"], L_["end"]["in_applied"]["b16"], L_["end"]["out_applied"]["b16"], cross, first_closing[tg], L_["seconds"]))
            json.dump(out, open(a.out_json, "w"), default=float)
            del S, X
            torch.cuda.empty_cache()
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    np.savez(a.out_npz, **arrays)
    json.dump(out, open(a.out_json, "w"), default=float)


if __name__ == "__main__":
    main()
