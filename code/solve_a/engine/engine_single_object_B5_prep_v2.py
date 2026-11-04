#!/usr/bin/env python
"""engine_single_object_B5_prep_v2.py (builder analyst, the HORIZON version 2 builder; engine spec 38.4 (1) (ii) and 39.5 (1) (vii); brief
lead/BRIEF_horizon_v2_builder.md): the columns of B.5 version 2, written for engine_single_object_B_v2.py --mode dust (never edited here).
(ii) "B.5 on the dust-tagged occupied columns only (the sector with occupation f), ... the sea orbitals' spreading beside as the comparator":
- the state per channel: the stored slice's occupied columns Phi = [sea, dust] with occupations f = [1, f_dust] (printed), re-orthonormalized
  in float64 together (Loewdin: Phi G^(-1/2), as part A and B.5 version 1 do); on B's chain (page_history_v6b.History, --nsites 8192) padded
  with zeros beyond the engine's sites; when the target clock is after the slice's, the whole occupied set is advanced record by record by B's
  Chebyshev action (piecewise-constant applied fields; labelled "B's propagation of the stored state, not an engine slice"), as version 1 did;
- the dust set: the dust-tagged columns of the re-orthonormalized set (its last n_dust columns) with their stored occupations f, all of them
  (inside and outside the root: their weight beyond the root at launch is printed and is part of the readout's start);
- the comparator: the sea-tagged columns' orbitals localized inside the outer root, the right singular vectors v_j of Pi_X Phi_sea,o with
  sigma_j^2 >= --lam-min (0.99), the columns Phi_sea,o v_j with their tails (version 1's selection on the sea sector alone), occupation 1;
- X: the sites with R < R_outer, R_outer the largest positive-slope root of the dense N + v at the target record (B v1's dense_roots).
Output: one npz (cols_dust_<tag>_k<kappa>, f_dust_..., cols_sealoc_<tag>_k<kappa>, f_sealoc_..., lam_sealoc_..., T_<tag>) and a json of the
checks: the stored purity defect (the eigenvalues of f^(1/2) G f^(1/2)), the Loewdin orthonormality, the propagation's Chebyshev order and the
occupied set's orthonormality after it, the dust columns' weight inside X and beyond R_outer and beyond 2gE at the target, the sea-localized
orbitals' count, smallest lambda and the next below, their orthonormality and inside-X weight against lambda. No mutual information (D15);
the massive channel only (D14). Every number is printed by the script. Refuses to overwrite."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_B_v1 as B1   # noqa: E402  (never edited; its import puts page_*_v6 on the path)

OPS = B1.OPS
PH = B1.PH
CD = OPS.CD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True)
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--slices", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--tags", required=True)
    ap.add_argument("--kappas", default="1,2"); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--lam-min", type=float, default=0.99); ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--out-npz", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    for p in (a.out_npz, a.out_json):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    t_all = time.time()
    H = PH.History(a.record, a.fields, a.nsites)
    Zf = np.load(a.fields, allow_pickle=True)
    rs = H.rs_nom; R = np.asarray(H.R, float); n = len(R); TT = np.asarray(H.T, float) / rs
    out = {"script": "engine_single_object_B5_prep_v2.py", "record": a.record, "fields": a.fields, "fields_module": a.fields_module, "nsites": n, "m": a.m,
           "lam_min": a.lam_min, "device": str(dev), "label": B1.LABEL, "no_mutual_information": "D15", "items": [],
           "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    arrays = {}
    for sp_, Tt, tag in zip(a.slices.split(","), [float(x) for x in a.targets.split(",")], a.tags.split(",")):
        Z = np.load(sp_)
        Ts = float(Z["T"]) / rs; ks = int(np.argmin(np.abs(TT - Ts))); kt = int(np.argmin(np.abs(TT - Tt)))
        Re = np.asarray(Z["R"], float); n0 = len(Re)
        site_diff = float(np.max(np.abs(Re - R[:n0])))
        rd = B1.dense_roots(H, Zf, kt, a.fields_module, H.ell)
        pos = [q for q in rd if q["slope"] > 0]
        Ro = max(q["R"] for q in pos) if pos else None
        two_gE = 2.0 * H.g * float(H.E[kt])
        print("%s: slice %s (T/rs %.3f, record %d, dtype %s; %d engine sites, the sites' largest difference from B's chain %.1e); target T/rs %.3f (record %d); the dense roots %s; outer %s; 2gE %.4f" % (
            tag, os.path.basename(sp_), TT[ks], ks, Z["sea_1"].dtype, n0, site_diff, TT[kt], kt, ", ".join("%.4f (%+.5f)" % (q["R"], q["slope"]) for q in rd) or "none",
            B1.fmt(Ro, "%.4f"), two_gE), flush=True)
        arrays["T_%s" % tag] = np.array(TT[kt])
        if Ro is None:
            out["items"].append({"tag": tag, "slice": sp_, "T_target_over_rs": TT[kt], "roots_dense": rd, "R_outer": None, "note": "no root of the dense N + v at the target record"})
            continue
        iX = int(np.searchsorted(R, Ro, side="left")); rx = np.arange(2 * iX)
        for kap in [int(x) for x in a.kappas.split(",")]:
            t_c = time.time()
            sea = np.asarray(Z["sea_%d" % kap], np.complex128); dust = np.asarray(Z["dust_%d" % kap], np.complex128); fd = np.asarray(Z["f_%d" % kap], float)
            ns, nd = sea.shape[1], dust.shape[1]
            Phi = np.concatenate([sea, dust], axis=1)
            f = np.concatenate([np.ones(ns), fd])
            G = Phi.conj().T @ Phi
            fh = np.sqrt(np.clip(f, 0.0, None))
            mu = np.linalg.eigvalsh((fh[:, None] * G) * fh[None, :])
            stored_idem = float(np.max(np.abs(mu * mu - mu)))
            gw, gv = np.linalg.eigh(G)
            Phi_o = Phi @ ((gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T)
            loew = float(np.max(np.abs(Phi_o.conj().T @ Phi_o - np.eye(Phi.shape[1]))))
            Xb = np.zeros((2 * n, Phi.shape[1]), np.complex128); Xb[: 2 * n0] = Phi_o
            Kmax = 0; tailacc = 0.0; steps = []
            if kt > ks:
                Xt = torch.as_tensor(Xb, dtype=CD, device=dev)
                h0, Q = OPS.chain_sparse(R, a.m, float(kap))
                for kk in range(ks, kt):
                    Nk, vk = H.fields(kk)
                    Xt, K, tail = PH.cheb_apply(OPS.Banded(OPS.clocked_csr(h0, Q, Nk, vk), dev), Xt, float(H.T[kk + 1] - H.T[kk]), a.cheb_tol)
                    Kmax = max(Kmax, K); tailacc += tail; steps.append({"from_T_over_rs": TT[kk], "to_T_over_rs": TT[kk + 1], "tau": float(H.T[kk + 1] - H.T[kk]), "cheb_K": int(K)})
                Xb = Xt.cpu().numpy(); del Xt
            orth_prop = float(np.max(np.abs(Xb.conj().T @ Xb - np.eye(Xb.shape[1]))))
            D = Xb[:, ns:]
            Sd = (np.abs(D) ** 2).reshape(n, 2, -1).sum(axis=1)
            d_in = Sd[R < Ro].sum(axis=0); d_out = Sd[R > Ro].sum(axis=0); d_2g = Sd[R > two_gE].sum(axis=0)
            Sea = Xb[:, :ns]
            U_, s_, Vh = np.linalg.svd(Sea[rx], full_matrices=False)
            lam = s_ ** 2; keep = lam >= a.lam_min
            W = Sea @ Vh[keep].conj().T
            orth_W = float(np.max(np.abs(W.conj().T @ W - np.eye(W.shape[1])))) if W.shape[1] else None
            inside = (np.abs(W[rx]) ** 2).sum(axis=0)
            ins_diff = float(np.max(np.abs(inside - lam[keep]))) if W.shape[1] else None
            arrays["cols_dust_%s_k%d" % (tag, kap)] = D
            arrays["f_dust_%s_k%d" % (tag, kap)] = fd
            arrays["cols_sealoc_%s_k%d" % (tag, kap)] = W
            arrays["f_sealoc_%s_k%d" % (tag, kap)] = np.ones(W.shape[1])
            arrays["lam_sealoc_%s_k%d" % (tag, kap)] = lam[keep]
            it = {"tag": tag, "kappa": kap, "slice": sp_, "slice_dtype": str(Z["sea_1"].dtype), "T_slice_over_rs": TT[ks], "T_target_over_rs": TT[kt], "engine_sites": n0,
                  "site_diff": site_diff, "roots_dense": rd, "R_outer": Ro, "two_gE": two_gE, "iX": iX, "n_sea": ns, "n_dust": nd, "dust_occupations": fd.tolist(),
                  "stored_idempotency": stored_idem, "loewdin_orthonormality": loew, "propagated_records": steps, "cheb_K_max": Kmax, "cheb_tail_acc": tailacc,
                  "occupied_orthonormality_after_propagation": orth_prop,
                  "dust_inside_X": d_in.tolist(), "dust_beyond_Rt": d_out.tolist(), "dust_beyond_2gE": d_2g.tolist(),
                  "dust_beyond_Rt_sum": float(np.dot(fd, d_out)), "dust_beyond_2gE_sum": float(np.dot(fd, d_2g)),
                  "sealoc_n": int(keep.sum()), "sealoc_lambda_min": float(lam[keep].min()) if keep.any() else None,
                  "sealoc_lambda_next_below": float(lam[~keep].max()) if (~keep).any() else None, "sealoc_orthonormality": orth_W, "sealoc_inside_X_minus_lambda_max": ins_diff,
                  "seconds": time.time() - t_c,
                  "provenance": ("the stored slice" if ks == kt else "the T/rs %.3f slice advanced %d record(s) by B's Chebyshev propagation (piecewise-constant applied fields); not an engine slice" % (TT[ks], kt - ks))}
            out["items"].append(it)
            print("  kappa %d: %d occupied (sea %d, dust %d, occupations %s; stored purity defect %.1e; Loewdin %.1e); propagated %d record(s) (K max %d; orthonormality after %.1e); "
                  "the dust's weight inside X %s, beyond R_t %s, beyond 2gE %s; %d sea-localized orbitals (smallest lambda %s, next below %s; orthonormality %s; inside-X minus lambda %s); %.1f s" % (
                      kap, ns + nd, ns, nd, ", ".join("%.3f" % x for x in fd), stored_idem, loew, len(steps), Kmax, orth_prop, ", ".join("%.4f" % x for x in d_in),
                      ", ".join("%.4e" % x for x in d_out), ", ".join("%.4e" % x for x in d_2g), int(keep.sum()), B1.fmt(it["sealoc_lambda_min"], "%.6f"),
                      B1.fmt(it["sealoc_lambda_next_below"], "%.6f"), B1.fmt(orth_W, "%.1e"), B1.fmt(ins_diff, "%.1e"), it["seconds"]), flush=True)
            json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    np.savez(a.out_npz, **arrays)
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()); out["seconds"] = time.time() - t_all
    out["npz_keys"] = sorted(arrays.keys())
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("done: %d items in %.0f s; npz keys %s" % (len(out["items"]), out["seconds"], sorted(arrays.keys())), flush=True)


if __name__ == "__main__":
    main()
