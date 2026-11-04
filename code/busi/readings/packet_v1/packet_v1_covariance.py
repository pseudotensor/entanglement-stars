#!/usr/bin/env python
"""packet_v1_covariance.py (busi lane, reviewer A slot, inbox #354): the per-site source bound checked on a stored covariance itself, on the engine's
orbital slices (orbitals_c19_..._T00xx.npz: per channel the propagated sea's columns, the dust's columns and occupations f, the reference R2's
columns, the eigenvalues e and the occupation mask at the applied fields of that record).

Per channel kappa: D = C - P = sea sea^dag + dust diag(f) dust^dag - R2 R2^dag.
  (a) my h_m[N_applied, v_applied] against the engine's: R2's columns are eigenvectors of the engine's h_m at these fields with eigenvalues e[mask],
      so ||h_m R2 - R2 diag(e[mask])||_max / max|e| measures the difference of the two operators (0 to rounding if they are the same matrix);
  (b) the stored eps_tot and J_tot of the record against the sources recomputed from the slice with my chain:
      eps_n = sum_kappa w_kappa (dens(h0, sea) - dens(h0, R2) + dens(h0, dust, f)), the same for Q;
  (c) ||D||_2 (torch eigvalsh of the dense D on the device) against 1, and tr D = n_sea + sum f - n_R2;
  (d) per site the Hoelder inequality itself: |Re tr Pi_n (h0 + W Q) D| <= ||Pi_n (h0 + W Q)||_1 ||D||_2 with W = v_applied[n], and the ratio.
Usage: python packet_v1_covariance.py --slice ORB.npz --fields F_fields.npz --json F.json --out DIR [--gpu N | --cpu] [--channels 1 2 ... 10]
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packet_v1_common as PC  # noqa: E402
from packet_v1_bounds import site_blocks, trace_norm_2x2  # noqa: E402


def dens(A, Phi, f=None):
    AP = A @ Phi
    if f is not None:
        AP = AP * np.asarray(f, float)[None, :]
    d = np.real(np.sum(np.conj(Phi) * AP, axis=1))
    return d.reshape(-1, 2).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--channels", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    device = torch.device("cpu") if (a.cpu or a.gpu is None) else torch.device("cuda:%d" % a.gpu)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = np.load(a.slice, allow_pickle=True)
    Z = np.load(a.fields, allow_pickle=True)
    J = json.load(open(a.json)); m = float(J["args"]["m"]); rs = float(J["args"]["rs"])
    R = np.asarray(S["R"], float); n = len(R); Tq = float(S["T"]); Napp = np.asarray(S["N_applied"], float); vapp = np.asarray(S["v_applied"], float)
    T = np.asarray(Z["T"], float); k = int(np.nonzero(np.abs(T - Tq) < 1e-9)[0][0])
    eps_st = np.asarray(Z["eps_tot"][k], float); J_st = np.asarray(Z["J_tot"][k], float)
    dN = float(np.max(np.abs(np.asarray(Z["N_applied"][k], float) - Napp))); dv = float(np.max(np.abs(np.asarray(Z["v_applied"][k], float) - vapp)))
    out = {"meta": {"slice": a.slice, "slice_md5": PC.md5(a.slice), "fields_md5": PC.md5(a.fields), "json_md5": PC.md5(a.json), "script_md5": PC.md5(os.path.abspath(__file__)),
                    "record": k, "T": Tq, "T_over_rs": Tq / rs, "dtype": str(S["dtype"]), "device": str(device),
                    "applied_fields_slice_vs_fields_npz_max_abs_diff": {"N": dN, "v": dv}}, "channels": {}}
    vol = 4.0 * np.pi * R ** 2
    eps_my = np.zeros(n); J_my = np.zeros(n)
    kappas = [int(x) for x in S["kappas"]]; weights = [float(x) for x in S["weights"]]
    for kappa in a.channels:
        tk = time.time()
        w = weights[kappas.index(kappa)]
        h0, Q = PC.chain(R, m, kappa)
        Hm = PC.clocked(h0, Q, Napp, vapp)
        sea = np.asarray(S["sea_%d" % kappa]).astype(np.complex128); dust = np.asarray(S["dust_%d" % kappa]).astype(np.complex128); f = np.asarray(S["f_%d" % kappa], float)
        R2 = np.asarray(S["R2_%d" % kappa]).astype(np.complex128); e = np.asarray(S["e_%d" % kappa], float); mask = np.unpackbits(np.asarray(S["mask_%d" % kappa]))[:2 * n].astype(bool)
        # (a) the operator check
        eocc = e[mask]
        resid = Hm @ R2 - R2 * eocc[None, :]
        ra = float(np.max(np.abs(resid)) / np.max(np.abs(e)))
        orth = float(np.max(np.abs(R2.conj().T @ R2 - np.eye(R2.shape[1]))))
        # (b) the sources
        d_eps = dens(h0, sea) - dens(h0, R2) + (dens(h0, dust, f) if dust.shape[1] else 0.0)
        d_J = dens(Q, sea) - dens(Q, R2) + (dens(Q, dust, f) if dust.shape[1] else 0.0)
        eps_my += w * d_eps; J_my += w * d_J
        # (c) D and its norm
        Phi = np.concatenate([sea, dust, R2], axis=1); ff = np.concatenate([np.ones(sea.shape[1]), f, -np.ones(R2.shape[1])])
        P_ = torch.as_tensor(Phi, dtype=torch.complex128, device=device); F_ = torch.as_tensor(ff, dtype=torch.complex128, device=device)
        D = (P_ * F_[None, :]) @ P_.conj().T
        lam = torch.linalg.eigvalsh(0.5 * (D + D.conj().T)).real.cpu().numpy()
        normD = float(np.max(np.abs(lam))); trD = float(np.real(torch.trace(D).cpu().numpy())); trD_expected = float(sea.shape[1] + f.sum() - R2.shape[1])
        del D, P_, F_
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # (d) the per-site Hoelder inequality with W = v_applied
        Vd = sp.diags(np.repeat(vapp, 2))
        A = (h0 + 0.5 * (Q @ Vd + Vd @ Q)).tocsr()                                         # h0 + (1/2){W, Q} with W = v_applied
        lhs = np.abs(dens(A, Phi, ff))                                                     # |Re tr Pi_n A D| per site
        Ghh = site_blocks(h0, h0, n); Ghq = site_blocks(h0, Q, n); Gqq = site_blocks(Q, Q, n)
        Wf = vapp[:, None, None]
        tn = trace_norm_2x2(Ghh + Wf * (Ghq + np.conj(np.transpose(Ghq, (0, 2, 1)))) + Wf ** 2 * Gqq)   # ||Pi_n (h0 + W Q)||_1 (fixed W per site)
        # note: with W varying per site, (1/2){W, Q} differs from W Q by the hop's half-difference of W; the fixed-W trace norm is reviewer A's bound's object
        ratio = lhs / (tn * normD)
        ch = {"weight": w, "n_sea": int(sea.shape[1]), "n_dust": int(dust.shape[1]), "f_sum": float(f.sum()), "n_R2": int(R2.shape[1]), "n_mask": int(mask.sum()),
              "operator_check_resid_rel": ra, "R2_orthonormality_dev": orth, "max_abs_e": float(np.max(np.abs(e))),
              "normD": normD, "trD": trD, "trD_expected": trD_expected, "lam_min": float(lam.min()), "lam_max": float(lam.max()),
              "hoelder_max_ratio": float(np.max(ratio)), "site_of_max_ratio": float(R[int(np.argmax(ratio))]),
              "max_lhs_site_density_a4": float(np.max(lhs / vol)), "seconds": time.time() - tk}
        out["channels"][str(kappa)] = ch
        print("kappa %d: operator resid %.2e, orth %.1e; ||D|| %.6f tr D %.3f (expected %.3f); Hoelder max ratio %.4f at R %.0f (%.0f s)" % (
            kappa, ra, orth, normD, trD, trD_expected, ch["hoelder_max_ratio"], ch["site_of_max_ratio"], ch["seconds"]), flush=True)
        del sea, dust, R2, Phi
    if set(a.channels) == set(kappas):
        sc = max(float(np.max(np.abs(eps_st))), 1e-300); sj = max(float(np.max(np.abs(J_st))), 1e-300)
        out["sources_check"] = {"max_abs_diff_eps_over_max_abs_eps": float(np.max(np.abs(eps_my - eps_st)) / sc), "max_abs_diff_J_over_max_abs_J": float(np.max(np.abs(J_my - J_st)) / sj),
                                "max_abs_eps_stored": sc, "max_abs_J_stored": sj, "sum_eps_stored": float(eps_st.sum()), "sum_eps_mine": float(eps_my.sum())}
        print("sources: eps rel max diff %.2e, J rel max diff %.2e" % (out["sources_check"]["max_abs_diff_eps_over_max_abs_eps"], out["sources_check"]["max_abs_diff_J_over_max_abs_J"]), flush=True)
    out["seconds"] = time.time() - t0
    tag = os.path.basename(a.slice).replace(".npz", "")
    with open(os.path.join(a.out, "packet_v1_covariance_%s.json" % tag), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
