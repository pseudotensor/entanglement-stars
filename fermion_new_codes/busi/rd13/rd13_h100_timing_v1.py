#!/usr/bin/env python
"""rd13_h100_timing_v1.py (busi, analyst; RD-1 + RD-3, REVIEWA_algorithm_redesign_v1 sec. 7): the card factor of the banded eigensolver and of the
Sternheimer J.v: one channel (kappa 1) at 2n = 4096 (the W2 r2b T0013.50 fields), 8192 (the ISO_128 nkqh2 T0005.00 fields) and 16384 (the
prototype's synthetic 8192-site chain), each: torch.linalg.eigh (2 warm calls, then 3 timed, the minimum) against
engine_banded_eigh_v1.eigT_banded on the same dense input (1 warm, 3 timed, the minimum), and the engine's eigenbasis J.v formula against
sternheimer_X (on the banded basis; 1 warm, 3 timed, the minimum; the occupied set the stored mask or, synthetic, the negative set; the
perturbation the applied fields' change over the record before, or a 1e-3 bump), with the device synchronized around every timed call.
Sanity numbers beside the walls: eigenvalues max |d|, the tracked set (the stored mask's) equal, the density response's relative difference.
    python rd13_h100_timing_v1.py INPUTS_NPZ CODE_DIR OUT_STEM [--reps 3]"""
import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch


def clocked_csr(h0, Q, N, v):
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs"); ap.add_argument("code_dir"); ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_vacflow_v1 as VF
    import engine_banded_eigh_v1 as EBE
    import hashlib
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    sync = lambda: torch.cuda.synchronize(dev)
    lines, out = [], []

    def say(s):
        lines.append(s); print(s, flush=True)
    tw = EBE.warmup(dev)
    say("rd13_h100_timing_v1: %s md5 %s; %s; torch %s, numba %s; warmup (JIT) %.1f s" % (EBE.VERSION, hashlib.md5(open(EBE.__file__, "rb").read()).hexdigest(),
        torch.cuda.get_device_name(dev), torch.__version__, EBE.numba.__version__, tw))
    Z = np.load(a.inputs, allow_pickle=False)
    for case in [c for c in ("w2", "iso", "syn") if c + "_R" in Z.files]:
        R = Z[case + "_R"]; N = Z[case + "_N"]; v = Z[case + "_v"]; dN = Z[case + "_dN"]; dv = Z[case + "_dv"]
        h0, Q = VF.chain_sparse(R, 0.2, 1.0)
        hm = clocked_csr(h0, Q, N, v); dh = clocked_csr(h0, Q, dN, dv)
        n = hm.shape[0]
        band = EBE.band_from_csr(hm); dhb = torch.as_tensor(EBE.band_from_csr(dh), device=dev); hb = torch.as_tensor(EBE.band_from_csr(h0.tocsr()), device=dev)
        hT = torch.as_tensor(hm.toarray(), device=dev)
        rec = {"case": case, "n_mat": n}
        td = []
        for r in range(2 + a.reps):
            sync(); t0 = time.time(); e_d, U_d = torch.linalg.eigh(hT); sync()
            if r >= 2:
                td.append(time.time() - t0)
        e_d = e_d.real
        tb = []
        for r in range(1 + a.reps):
            EBE.stats_snapshot(reset=True)
            sync(); t0 = time.time(); e_b, U_b = EBE.eigT_banded(hT); sync()
            if r >= 1:
                tb.append(time.time() - t0)
        st = EBE.stats_snapshot(reset=True)
        rec.update({"dense_s": min(td), "dense_all_s": td, "banded_s": min(tb), "banded_all_s": tb, "factor": min(td) / min(tb), "banded_stats_last": st,
                    "e_max_abs_diff": float((e_b - e_d).abs().max())})
        if case + "_mask" in Z.files:
            m = np.unpackbits(Z[case + "_mask"])[:n].astype(bool)
        else:
            m = (e_d < 0).cpu().numpy()
        mo = torch.as_tensor(m, device=dev)
        # tracked set on the stored mask's columns of the dense basis (the reference), from the banded basis
        Ref = U_d[:, mo]
        wb = ((U_b.conj().T @ Ref).abs() ** 2).sum(dim=1).cpu().numpy()
        sb = np.zeros(n, bool); sb[np.argsort(-wb, kind="stable")[:int(m.sum())]] = True
        rec["tracked_equal"] = bool(np.array_equal(sb, m)); del Ref

        def dens_pair(Uo, X):
            d_ = (Uo.conj() * EBE.band_matvec(hb, X)).real.sum(dim=1) + (X.conj() * EBE.band_matvec(hb, Uo)).real.sum(dim=1)
            return d_.reshape(-1, 2).sum(dim=1).cpu().numpy()
        Uo = U_d[:, mo]; Uu = U_d[:, ~mo]; eo = e_d[mo]; eu = e_d[~mo]
        te = []
        for r in range(1 + a.reps):
            sync(); t0 = time.time()
            Hm = Uu.conj().T @ EBE.band_matvec(dhb, Uo)
            den = eo[None, :] - eu[:, None]
            small = den.abs() < 1e-12
            M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
            X = Uu @ M
            d_ref = dens_pair(Uo, X)
            sync()
            if r >= 1:
                te.append(time.time() - t0)
            del Hm, den, small, M, X
        Uob = U_b[:, mo]; eob = e_b[mo]; eub = e_b[~mo]
        ts = []
        for r in range(1 + a.reps):
            EBE.stats_snapshot(reset=True)
            sync(); t0 = time.time()
            Y, inf_ = EBE.sternheimer_X(band, eob, eub, Uob, -EBE.band_matvec(dhb, Uob))
            d_st = dens_pair(Uob, Y) if Y is not None else None
            sync()
            if r >= 1:
                ts.append(time.time() - t0)
            del Y
        rec.update({"jv_eigenbasis_s": min(te), "jv_eigenbasis_all_s": te, "jv_sternheimer_s": min(ts), "jv_sternheimer_all_s": ts, "jv_factor": min(te) / min(ts),
                    "jv_info": inf_, "jv_rel_diff": (float(np.max(np.abs(d_st - d_ref))) / float(np.max(np.abs(d_ref)))) if d_st is not None else None, "n_occ": int(m.sum())})
        say("%s n %d: eigh dense %.3f s, banded %.3f s (%s): %.2fx; e max |d| %.1e; tracked set equal %s; J.v eigenbasis %.3f s, Sternheimer %.3f s: %.2fx (occupied %d, rel diff %s, partner offsets %s, solve residual %s); banded stats %s" % (
            case, n, rec["dense_s"], rec["banded_s"], "banded" if st["banded"] == 1 else "FALLBACK %s" % st["fallback_why"], rec["factor"], rec["e_max_abs_diff"],
            rec["tracked_equal"], rec["jv_eigenbasis_s"], rec["jv_sternheimer_s"], rec["jv_factor"], rec["n_occ"],
            "%.1e" % rec["jv_rel_diff"] if rec["jv_rel_diff"] is not None else "FALLBACK %s" % inf_.get("fallback"), inf_.get("partner_offsets"),
            "%.1e" % inf_.get("solve_residual_rel", float("nan")),
            {k: (round(st[k], 3) if isinstance(st[k], float) else st[k]) for k in ("kernel_seconds", "post_seconds", "guard_seconds", "max_residual_rel", "clusters")}))
        out.append(rec)
        del hT, U_d, U_b, Uo, Uu, Uob
        torch.cuda.empty_cache()
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
