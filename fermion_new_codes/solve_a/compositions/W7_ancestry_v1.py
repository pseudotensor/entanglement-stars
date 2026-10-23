#!/usr/bin/env python
"""W7_ancestry_v1.py (the solve lane; reorientation 6 composition 2, W7's five items; box CPU). The data script for items 1, 4 and 5 on all
five dust channels kappa 1..5, per copy (f per column, no multiplicity; the composer applies the 16 kappa weights where it prints counts).

Per kappa, two launches, as DUST_root_history v2b (dust_root_history_v2b.py, imported read-only for its Backend and fields):
- T12p125: the stored slice T/r_s 12.000's dust columns (engine_single_object_B5_prep_v2's Loewdin of [sea, dust]) advanced one record to
  the formation record 12.125 (record 97, the first record with a trapping zero) by B's Chebyshev action;
- T15p000: the stored slice 15.000's dust columns (complex64 on disk; Loewdin in float64), record 120;
each split at its launch record by the dense outer root (B v2's roots_at / outer_inner_of, the record's trapping zero) and by the applied outer
root (the largest rising zero of N + v on the applied site fields): psi = psi_in + psi_out. psi, psi_in(dense), psi_in(applied) are propagated
record by record on the production's stored history (C3prod19 r11 to 40, then the ext100 continuation; 8192 sites) by v2b's propagator (the
previous record's clocked operator over the record step, Chebyshev tolerance 1e-14) to T/r_s 56 (record 448).
Per record and per column (arrays saved): the norm; the weights beyond 16 and 24 r_s of psi, psi_in and psi_out for both cuts; the weight
beyond the record's dense and applied roots; the roots; the moving-cut reassignment per column (the weight beyond the new root minus beyond the
old root, both at the new record); the number currents i[h, Pi_(R > c)] through 16 r_s of psi, psi_in(dense), psi_in(applied) and psi_out(dense),
through 24 r_s of psi and through the dense and applied outer roots of psi. The state psi of T12p125 at the slice clocks 15 .. 56 (for the
same-set check against the stored slices) and every state at 56.
At 56 (the held record): one dense eigendecomposition of the held operator's exterior block beyond 16 r_s (export_horizon_refcol_v2's block:
the restriction of h(N(56), v(56)) to the sites R > 16 r_s, both components; 14336 levels) and one of the full held operator (16384 levels,
dust_ledger_postpass_v1b's alternative, with each eigenvector's weight inside 2 r_s and 16 r_s for the extended/bound proxy). The weights of
every final state's exterior part on both eigenbases are saved (levels x columns) with the eigenvalues; the composer bins them
(horizon3_bins_v1). Outputs <out-dir>/W7_ancestry_k<kappa>.{npz,json} per kappa; a kappa whose outputs exist is skipped.
Usage (box): nice -n 10 python W7_ancestry_v1.py --kappas 1,4,5 --threads 4 --out-dir <internal>/W7_v1
"""
import os
import sys

_t = "4"
for _i, _s in enumerate(sys.argv):
    if _s == "--threads" and _i + 1 < len(sys.argv):
        _t = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _t   # before numpy loads its BLAS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import W7_common_v1 as C   # noqa: E402

import argparse   # noqa: E402
import time       # noqa: E402
import types      # noqa: E402

import numpy as np            # noqa: E402
import scipy.linalg as sla    # noqa: E402

T0 = time.time()


def log(msg):
    print("[+%7.0f s] %s" % (time.time() - T0, msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", default="1,2,3,4,5")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="<internal>/W7_v1")
    ap.add_argument("--T-end-rs", type=float, default=56.0)
    ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--no-eigh", action="store_true", help="skip the held-record eigendecompositions (a dry run)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch, BV2, HB, D2 = C.load_engine()
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    ns_ = types.SimpleNamespace(propagator="ops", cheb_tol=a.cheb_tol, eig_backend="scipy", eig_driver="evr")
    BE = D2.Backend(ns_, torch, BV2, dev)
    OPS = BE.OPS
    H = BV2.CombinedHistory(C.REC1, C.FLD1, 8192, C.REC2, C.FLD2, 40.0)
    rs = float(H.rs_nom); R = np.asarray(H.R, float); n = len(R); ne = int(H.ne); m = C.M_MASS; b = HB.band_top(m)
    TT = np.asarray(H.T, float) / rs
    kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
    k_form = next(k for k in range(len(TT)) if H.rec[k].get("trapping_zeros"))
    k15 = int(np.argmin(np.abs(TT - 15.0)))
    save_T = [12.125, 15.0, 16.0, 20.0, 24.0, 25.0, 28.0, 30.0, 35.0, 40.0, 48.0, 50.0, 52.0, 54.0, 56.0]
    save_k = {int(np.argmin(np.abs(TT - t))): t for t in save_T}
    log("history: %d records, %d sites (ne %d), rs %g; formation record %d (T/r_s %.3f); 15 -> record %d; end record %d (T/r_s %.3f); threads %d" % (
        len(TT), n, ne, rs, k_form, TT[k_form], k15, kend, TT[kend], a.threads))
    rows16 = C.block_rows(R, 16.0 * rs)
    mask16 = np.repeat(R > 16.0 * rs, 2)
    for kap in [int(x) for x in a.kappas.split(",") if x]:
        pj = os.path.join(a.out_dir, "W7_ancestry_k%d.json" % kap); pz = pj[:-5] + ".npz"
        if os.path.exists(pj) and os.path.exists(pz):
            log("kappa %d: outputs exist, skipped" % kap); continue
        t_k = time.time()
        h0, Q = OPS.chain_sparse(R, m, float(kap))
        out = {"script": "W7_ancestry_v1.py", "script_md5": C.md5(os.path.abspath(__file__)), "common_md5": C.md5(C.__file__),
               "engine_module_md5": C.md5(BV2.__file__), "ops_module": OPS.__file__, "ops_module_md5": C.md5(OPS.__file__),
               "v2b_md5": C.md5(D2.__file__), "bins_module_md5": C.md5(HB.__file__), "history": [C.REC1, C.FLD1, C.REC2, C.FLD2], "switch_T_rs": 40.0,
               "kappa": kap, "m": m, "b": b, "rs": rs, "n_sites": n, "ne": ne, "k_form": k_form, "T_form": float(TT[k_form]), "k15": k15, "k_end": kend,
               "T_end": float(TT[kend]), "cheb_tol": a.cheb_tol, "threads": a.threads, "launches": {}}
        arrays = {}
        L = {}
        for tag, Ts, kt in (("T12p125", 12.0, k_form), ("T15p000", 15.0, k15)):
            sp_ = C.SLICE_FMT % Ts
            Z = np.load(sp_)
            ks = int(np.argmin(np.abs(TT - float(Z["T"]) / rs)))
            D, fd, idem, orth, raw = C.loewdin_dust(Z, kap)
            nd = D.shape[1]
            W = np.zeros((2 * n, nd), np.complex128); W[:D.shape[0]] = D
            Kp = 0
            for kk in range(ks, kt):
                W, K = BE.propagate(OPS.clocked_csr(h0, Q, *D2.fields(H, kk)), W, float(H.T[kk + 1] - H.T[kk])); Kp = max(Kp, K)
            rd = BV2.roots_at(H, kt, "engine_fields_v14")
            Ro_d, Ri_d, _ = BV2.outer_inner_of(rd, H.rec[kt])
            N0, v0 = D2.fields(H, kt)
            Ro_a, Ri_a, za = C.applied_outer_inner(R[:ne], (N0 + v0)[:ne])
            md = np.repeat(R < (Ro_d or 0.0), 2); ma = np.repeat(R < (Ro_a or 0.0), 2)
            L[tag] = {"k0": kt, "nd": nd, "f": fd, "S": {"all": W.copy(), "in_dense": W * md[:, None], "in_applied": W * ma[:, None]},
                      "nr": kend - kt + 1}
            out["launches"][tag] = {"slice": sp_, "slice_T_over_rs": float(Z["T"]) / rs, "slice_record": ks, "launch_record": kt, "T_launch": float(TT[kt]),
                                    "slice_dtype": str(Z["dust_%d" % kap].dtype), "n_dust": nd, "f": fd.tolist(), "stored_purity_defect": idem,
                                    "loewdin_dust_orthonormality": orth, "loewdin_minus_stored_max": float(np.max(np.abs(D - raw))) if nd else 0.0,
                                    "propagated_records_to_launch": kt - ks, "cheb_K_to_launch": Kp,
                                    "roots": {"dense_outer": Ro_d, "dense_inner": Ri_d, "record_zeros": list(H.rec[kt].get("trapping_zeros") or []),
                                              "applied_outer": Ro_a, "applied_inner": Ri_a, "applied_zeros": [[r_, s_] for r_, s_ in za]},
                                    "inside_dense_per_column": C.site_w(W * md[:, None], n).sum(axis=0).tolist(),
                                    "inside_applied_per_column": C.site_w(W * ma[:, None], n).sum(axis=0).tolist(),
                                    "norm_per_column": C.site_w(W, n).sum(axis=0).tolist()}
            arrays["%s_launch_cols" % tag] = W.copy()
            del Z, D, raw
            log("kappa %d %s: slice %s (%s), %d dust columns, purity defect %.1e, Loewdin %.1e, %d record(s) to launch (K %d); dense root %.6f, applied %s; inside dense %s" % (
                kap, tag, os.path.basename(sp_), out["launches"][tag]["slice_dtype"], nd, idem, orth, kt - ks, Kp, Ro_d, Ro_a,
                np.round(out["launches"][tag]["inside_dense_per_column"], 4).tolist()))
        # per-record arrays
        for tag, Lt in L.items():
            nr, nd = Lt["nr"], Lt["nd"]
            A = {}
            for nm in ("all", "in_dense", "out_dense", "in_applied", "out_applied"):
                A["b16_" + nm] = np.full((nr, nd), np.nan); A["b24_" + nm] = np.full((nr, nd), np.nan)
            for cut in ("dense", "applied"):
                A["beyond_Rt_" + cut] = np.full((nr, nd), np.nan); A["Rt_" + cut] = np.full(nr, np.nan)
                A["mcs_col_" + cut] = np.full((nr, nd), np.nan)
                A["J_outer_%s_all" % cut] = np.full((nr, nd), np.nan)
            for nm in ("all", "in_dense", "in_applied", "out_dense"):
                A["J16_" + nm] = np.full((nr, nd), np.nan)
            A["J24_all"] = np.full((nr, nd), np.nan)
            A["norm"] = np.full((nr, nd), np.nan); A["T"] = TT[Lt["k0"]: kend + 1].copy()
            Lt["A"] = A
        active = ["T12p125"]
        Kmax = 0; h_prev = None
        for kk in range(k_form, kend + 1):
            if kk > k_form:
                tau = float(H.T[kk] - H.T[kk - 1])
                cols = [(tg, nm) for tg in active for nm in ("all", "in_dense", "in_applied")]
                big = np.concatenate([L[tg]["S"][nm] for tg, nm in cols], axis=1)
                big, K = BE.propagate(h_prev, big, tau); Kmax = max(Kmax, K)
                i0 = 0
                for tg, nm in cols:
                    w_ = L[tg]["nd"]; L[tg]["S"][nm] = big[:, i0:i0 + w_]; i0 += w_
                del big
            if kk == k15 and "T15p000" not in active:
                active.append("T15p000")
            Nk, vk = D2.fields(H, kk)
            hn = OPS.clocked_csr(h0, Q, Nk, vk)
            rt_d, _ = C.record_roots(H.rec[kk])
            rt_a, _, _ = C.applied_outer_inner(R[:ne], (Nk + vk)[:ne])
            Jops = {"16": OPS.number_current_csr(hn, R, 16.0 * rs), "24": OPS.number_current_csr(hn, R, 24.0 * rs)}
            if rt_d:
                Jops["od"] = OPS.number_current_csr(hn, R, rt_d)
            if rt_a:
                Jops["oa"] = OPS.number_current_csr(hn, R, rt_a)

            def cur(key, X):
                return ((X.conj() * (Jops[key] @ X)).sum(axis=0)).real

            for tg in active:
                Lt = L[tg]; A = Lt["A"]; j = kk - Lt["k0"]; S = Lt["S"]
                Wall = C.site_w(S["all"], n); A["norm"][j] = Wall.sum(axis=0)
                A["b16_all"][j] = Wall[R > 16 * rs].sum(axis=0); A["b24_all"][j] = Wall[R > 24 * rs].sum(axis=0)
                for cut in ("dense", "applied"):
                    Win = C.site_w(S["in_" + cut], n); Wout = C.site_w(S["all"] - S["in_" + cut], n)
                    A["b16_in_" + cut][j] = Win[R > 16 * rs].sum(axis=0); A["b24_in_" + cut][j] = Win[R > 24 * rs].sum(axis=0)
                    A["b16_out_" + cut][j] = Wout[R > 16 * rs].sum(axis=0); A["b24_out_" + cut][j] = Wout[R > 24 * rs].sum(axis=0)
                for cut, rt in (("dense", rt_d), ("applied", rt_a)):
                    if rt:
                        A["Rt_" + cut][j] = rt
                        A["beyond_Rt_" + cut][j] = Wall[R > rt].sum(axis=0)
                        if j > 0 and np.isfinite(A["Rt_" + cut][j - 1]):
                            A["mcs_col_" + cut][j] = Wall[R > rt].sum(axis=0) - Wall[R > A["Rt_" + cut][j - 1]].sum(axis=0)
                        A["J_outer_%s_all" % cut][j] = cur("od" if cut == "dense" else "oa", S["all"])
                A["J16_all"][j] = cur("16", S["all"]); A["J24_all"][j] = cur("24", S["all"])
                A["J16_in_dense"][j] = cur("16", S["in_dense"]); A["J16_in_applied"][j] = cur("16", S["in_applied"])
                A["J16_out_dense"][j] = cur("16", S["all"] - S["in_dense"])
                if tg == "T12p125" and kk in save_k:
                    arrays["T12p125_state_all_T%06.3f" % save_k[kk]] = S["all"].copy()
            h_prev = hn
            if (kk - k_form) % 32 == 0 or kk == kend:
                Lt = L["T12p125"]; j = kk - Lt["k0"]; f = Lt["f"]
                log("kappa %d: record %d (T/r_s %.3f); K max %d; T12p125 beyond 16 r_s all %.5e, from inside dense %.5e; J16 all %+.3e in %+.3e; norm drift %.1e" % (
                    kap, kk, TT[kk], Kmax, f @ Lt["A"]["b16_all"][j], f @ Lt["A"]["b16_in_dense"][j], f @ Lt["A"]["J16_all"][j], f @ Lt["A"]["J16_in_dense"][j],
                    float(np.nanmax(np.abs(Lt["A"]["norm"][j] - Lt["A"]["norm"][0])))))
        out["cheb_K_max"] = Kmax
        for tg, Lt in L.items():
            for k_, v_ in Lt["A"].items():
                arrays["%s_%s" % (tg, k_)] = v_
            arrays["%s_f" % tg] = Lt["f"]
            for nm in ("all", "in_dense", "in_applied"):
                arrays["%s_end_%s" % (tg, nm)] = Lt["S"][nm]
        # the held record at 56: the exterior block beyond 16 r_s and the full held operator
        Nn, vn = D2.fields(H, kend)
        hn = OPS.clocked_csr(h0, Q, Nn, vn)
        sets = []
        for tg, Lt in L.items():
            for nm in ("all", "in_dense", "in_applied"):
                sets.append((tg, nm, Lt["S"][nm]))
            sets.append((tg, "out_dense", Lt["S"]["all"] - Lt["S"]["in_dense"]))
            sets.append((tg, "out_applied", Lt["S"]["all"] - Lt["S"]["in_applied"]))
        Xall = np.concatenate([X for _, _, X in sets], axis=1)
        out["held_sets"] = [[tg, nm, int(X.shape[1])] for tg, nm, X in sets]
        if not a.no_eigh:
            t1 = time.time()
            Hb = np.asarray(hn[rows16][:, rows16].toarray(), np.complex128)
            herm_b = float(np.max(np.abs(Hb - Hb.conj().T)))
            eB, UB = sla.eigh(Hb, driver="evr", overwrite_a=True, check_finite=False); del Hb
            arrays["held_block16_e"] = eB
            arrays["held_block16_w"] = C.weights_in_basis(UB, Xall[rows16])
            out["held_block16"] = {"dim": int(len(eB)), "e_min": float(eB[0]), "e_max": float(eB[-1]), "hermiticity": herm_b, "seconds": time.time() - t1,
                                   "cut_a": 16.0 * rs, "n_sites": int(len(rows16) // 2), "R_max": float(R[-1])}
            del UB
            log("kappa %d: block beyond 16 r_s, %d levels in %.0f s (e %.6f .. %.6f)" % (kap, len(eB), time.time() - t1, eB[0], eB[-1]))
            t1 = time.time()
            Hf = np.asarray(hn.toarray(), np.complex128)
            eA, UA = sla.eigh(Hf, driver="evr", overwrite_a=True, check_finite=False); del Hf
            arrays["held_full_e"] = eA
            arrays["held_full_w_ext16"] = C.weights_in_basis(UA, Xall * mask16[:, None])
            arrays["held_full_w_whole"] = C.weights_in_basis(UA, Xall)
            for Rc in (2.0, 16.0):
                n_in = int(np.sum(R <= Rc * rs))
                arrays["held_full_win_%g" % Rc] = (np.abs(UA[:2 * n_in, :]) ** 2).sum(axis=0)
            out["held_full"] = {"dim": int(len(eA)), "e_min": float(eA[0]), "e_max": float(eA[-1]), "seconds": time.time() - t1}
            del UA
            log("kappa %d: full held operator, %d levels in %.0f s" % (kap, len(eA), time.time() - t1))
        out["seconds"] = time.time() - t_k
        C.save_npz(arrays, pz)
        C.dump_json(out, pj)
        log("kappa %d done in %.0f s: %s, %s" % (kap, out["seconds"], pz, pj))


if __name__ == "__main__":
    main()
