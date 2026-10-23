#!/usr/bin/env python
"""engine_single_object_B5_prep_v1.py (builder analyst; inbox #41's revised S4 part B, item 5's columns): "5. The state's own interior
orbitals (eigenvalue >= 0.99 of Phi^dagger Pi_X Phi inside the outer root at T/r_s 12.125 and 15.0), propagated on the solved history to
T/r_s 40". This script writes those columns; engine_single_object_B_v1.py --mode orbitals propagates them (never edited here).
- The state per channel: the stored slice's occupied columns Phi = [sea, dust] (the slice's occupations printed), re-orthonormalized in
  float64 (Loewdin: Phi G^(-1/2), as part A does); on B's chain (page_history_v6b.History: the record's applied fields on the engine's sites,
  the vacuum Painleve-Gullstrand exterior beyond them, --nsites 8192), the columns padded with zeros beyond the engine's sites.
- T/r_s 15.0: the production's stored slice at that record. T/r_s 12.125: no slice is stored there and the record at T/r_s 12.0 has no root
  of N + v; the T/r_s 12.0 slice is advanced one record by B's own machinery (the Chebyshev action of exp(-i tau h) with h the record's
  clocked operator at its applied fields, piecewise constant, page_history_v6b.cheb_apply to --cheb-tol), labelled: B's propagation of the
  stored state, not an engine slice.
- X: the sites with R < R_outer, R_outer the largest positive-slope root of the dense N + v at the target record (B's dense_roots: the record's
  stored shell totals re-solved by --fields-module). The orbitals: the right singular vectors v_j of Pi_X Phi_o with sigma_j^2 >= --lam-min,
  the columns Phi_o v_j (the actual orbitals with their tails beyond X, strand R SO4's physical readout, never their projections).
Output: one npz in the layout engine_single_object_B_v1.py --mode orbitals reads (cols_<tag>_k<kappa>, lam_<tag>_k<kappa>, T_<tag>) and a json of
the checks (the orthonormality of the columns, their weight inside X against lambda, the one-record propagation's norm defect and Chebyshev
order). No mutual information (D15); the massive channel only (D14). Every number is printed by the script."""
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
import engine_single_object_B_v1 as B1   # noqa: E402  (its import puts the page_*_v6 path in place)

OPS = B1.OPS
PH = B1.PH
CD = OPS.CD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True)
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--slices", required=True, help="slice npz paths, comma-separated, one per tag")
    ap.add_argument("--targets", required=True, help="the target T/r_s per slice")
    ap.add_argument("--tags", required=True, help="the tag per slice (no underscores)")
    ap.add_argument("--kappas", default="1,2"); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--lam-min", type=float, default=0.99); ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--out-npz", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    t_all = time.time()
    H = PH.History(a.record, a.fields, a.nsites)
    Zf = np.load(a.fields, allow_pickle=True)
    rs = H.rs_nom; R = np.asarray(H.R, float); n = len(R); TT = np.asarray(H.T, float) / rs
    out = {"script": "engine_single_object_B5_prep_v1.py", "record": a.record, "fields": a.fields, "fields_module": a.fields_module, "nsites": n, "m": a.m,
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
        print("%s: slice %s (T/rs %.3f, record %d; %d engine sites, the sites' largest difference from B's chain %.1e); target T/rs %.3f (record %d); the dense roots %s; outer %s" % (
            tag, os.path.basename(sp_), TT[ks], ks, n0, site_diff, TT[kt], kt, ", ".join("%.4f (%+.5f)" % (q["R"], q["slope"]) for q in rd) or "none",
            B1.fmt(Ro, "%.4f")), flush=True)
        arrays["T_%s" % tag] = np.array(TT[kt])
        if Ro is None:
            out["items"].append({"tag": tag, "slice": sp_, "T_slice_over_rs": TT[ks], "T_target_over_rs": TT[kt], "roots_dense": rd, "R_outer": None,
                                 "note": "no root of the dense N + v at the target record: no X, no orbitals"})
            continue
        iX = int(np.searchsorted(R, Ro, side="left")); rx = np.arange(2 * iX)
        for kap in [int(x) for x in a.kappas.split(",")]:
            t_c = time.time()
            sea = np.asarray(Z["sea_%d" % kap], np.complex128); dust = np.asarray(Z["dust_%d" % kap], np.complex128); fd = np.asarray(Z["f_%d" % kap], float)
            Phi = np.concatenate([sea, dust], axis=1)
            f = np.concatenate([np.ones(sea.shape[1]), fd])
            G = Phi.conj().T @ Phi
            fh = np.sqrt(np.clip(f, 0.0, None))
            mu = np.linalg.eigvalsh((fh[:, None] * G) * fh[None, :])
            stored_idem = float(np.max(np.abs(mu * mu - mu)))
            gw, gv = np.linalg.eigh(G)
            Phi_o = Phi @ ((gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T)
            X = np.zeros((2 * n, Phi.shape[1]), np.complex128); X[: 2 * n0] = Phi_o
            Xt = torch.as_tensor(X, dtype=CD, device=dev)
            h0, Q = OPS.chain_sparse(R, a.m, float(kap))
            Kmax = 0; tailacc = 0.0; steps = []
            for kk in range(ks, kt):
                Nk, vk = H.fields(kk)
                Xt, K, tail = PH.cheb_apply(OPS.Banded(OPS.clocked_csr(h0, Q, Nk, vk), dev), Xt, float(H.T[kk + 1] - H.T[kk]), a.cheb_tol)
                Kmax = max(Kmax, K); tailacc += tail; steps.append({"from_T_over_rs": TT[kk], "to_T_over_rs": TT[kk + 1], "tau": float(H.T[kk + 1] - H.T[kk]), "cheb_K": int(K)})
            G2 = (Xt.mH @ Xt).cpu().numpy()
            orth_prop = float(np.max(np.abs(G2 - np.eye(G2.shape[0]))))
            Xn = Xt.cpu().numpy(); del Xt
            U_, s_, Vh = np.linalg.svd(Xn[rx], full_matrices=False)
            lam = s_ ** 2; keep = lam >= a.lam_min
            W = Xn @ Vh[keep].conj().T
            orth_W = float(np.max(np.abs(W.conj().T @ W - np.eye(W.shape[1])))) if W.shape[1] else None
            inside = (np.abs(W[rx]) ** 2).sum(axis=0)
            ins_diff = float(np.max(np.abs(inside - lam[keep]))) if W.shape[1] else None
            arrays["cols_%s_k%d" % (tag, kap)] = W
            arrays["lam_%s_k%d" % (tag, kap)] = lam[keep]
            it = {"tag": tag, "kappa": kap, "slice": sp_, "T_slice_over_rs": TT[ks], "T_target_over_rs": TT[kt], "engine_sites": n0, "site_diff": site_diff,
                  "roots_dense": rd, "R_outer": Ro, "iX": iX, "n_occ": int(Phi.shape[1]), "n_sea": int(sea.shape[1]), "n_dust": int(dust.shape[1]),
                  "dust_occupations": fd.tolist(), "stored_idempotency": stored_idem, "propagated_records": steps, "cheb_K_max": Kmax, "cheb_tail_acc": tailacc,
                  "occupied_orthonormality_after_propagation": orth_prop, "n_orbitals": int(keep.sum()), "lambda_min": float(lam[keep].min()) if keep.any() else None,
                  "lambda_next_below": float(lam[~keep].max()) if (~keep).any() else None, "lambdas": lam[keep].tolist(), "orbitals_orthonormality": orth_W,
                  "inside_X_minus_lambda_max": ins_diff, "seconds": time.time() - t_c,
                  "provenance": ("the stored slice" if ks == kt else "the T/rs %.3f slice advanced %d record(s) by B's Chebyshev propagation (piecewise-constant applied fields); not an engine slice" % (TT[ks], kt - ks))}
            out["items"].append(it)
            print("  kappa %d: %d occupied (sea %d, dust %d, dust occupations %s; stored ||C^2 - C|| %.1e); propagated %d record(s) (Cheb K max %d, tail %.1e; the occupied set's orthonormality after %.1e); X: %d sites (R < %.4f); %d orbitals with lambda >= %.2f (smallest %s; the next below %s); their orthonormality %s, inside-X weight minus lambda %s; %.1f s" % (
                kap, Phi.shape[1], sea.shape[1], dust.shape[1], ", ".join("%.3f" % x for x in fd), stored_idem, len(steps), Kmax, tailacc, orth_prop, iX, Ro,
                int(keep.sum()), a.lam_min, B1.fmt(it["lambda_min"], "%.6f"), B1.fmt(it["lambda_next_below"], "%.6f"), B1.fmt(orth_W, "%.1e"), B1.fmt(ins_diff, "%.1e"),
                it["seconds"]), flush=True)
            json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    np.savez(a.out_npz, **arrays)
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()); out["seconds"] = time.time() - t_all
    out["npz_keys"] = sorted(arrays.keys())
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("done: %d items in %.0f s; npz keys %s" % (len(out["items"]), out["seconds"], sorted(arrays.keys())), flush=True)


if __name__ == "__main__":
    main()
