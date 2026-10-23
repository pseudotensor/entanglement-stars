#!/usr/bin/env python
"""W7_record_items_v1_2.py (the solve lane; COMP2 v1.2, reviewer A's ruling 1a: two record items, not gates; box CPU).
1. What in the propagator's input changes at T/r_s 40: dust_root_history_v2b's propagator (engine_single_object_B_v2.CombinedHistory) takes
   the r11 records and fields up to 40 and the ext100 ones after; both files' field grids, tau, sites, ell and run args are compared, the
   records' solver counters (halvings, iterations) and the step between records on each side, and the applied fields of the two files at
   the records both hold (28 to 40, the rerun from the T28 slice).
2. Where along the chain the slice-vs-propagated difference sits at 48 and 56: the weighted (16 kappa) per-site difference
   sum_k 16k f . |w_slice - w_propagated| binned by radius, its peak, its median radius, and the propagated weight beyond the production
   chain's end (32 r_s).
3. A direct attribution test: the stored 48 slice's dust columns (Loewdin, as B.5) advanced to 50 by v2b's propagator on the stored ext100
   fields twice, on the production's own chain (2048 sites, ending at 32 r_s) and on B's chain (8192 sites, vacuum beyond 2048), each
   against the stored 50 slice (1 - overlap and site-weight L1, per kappa).
Outputs <out-dir>/W7_record_items_v1_2.json. Read-only on every input.
Usage (box): NUMPY_MADVISE_HUGEPAGE=0 nice -n 10 python W7_record_items_v1_2.py --threads 4
"""
import os
import sys

_t = "4"
for _i, _s in enumerate(sys.argv):
    if _s == "--threads" and _i + 1 < len(sys.argv):
        _t = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _t
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import W7_common_v1 as C   # noqa: E402

import argparse   # noqa: E402
import json       # noqa: E402
import types      # noqa: E402

import numpy as np   # noqa: E402

V1 = "<internal>/W7_v1"
V11 = "<internal>/W7_v1_1"
MULT = {k: 16.0 * k for k in C.KAPPAS}
RS = 64.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="<internal>/W7_v1_2")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch, BV2, HB, D2 = C.load_engine()
    torch.set_num_threads(a.threads)
    OUT = {"script": "W7_record_items_v1_2.py", "script_md5": C.md5(os.path.abspath(__file__))}
    # ---------------- 1. the inputs at 40
    H = BV2.CombinedHistory(C.REC1, C.FLD1, 8192, C.REC2, C.FLD2, 40.0)
    H1, H2 = H.H1, H.H2
    J1 = json.load(open(C.REC1)); J2 = json.load(open(C.REC2))
    a1, a2 = J1.get("args", {}), J2.get("args", {})
    diff = {k: [a1.get(k), a2.get(k)] for k in sorted(set(a1) | set(a2)) if a1.get(k) != a2.get(k)}
    T1 = np.asarray(H1.T) / RS; T2 = np.asarray(H2.T) / RS

    def stats(J, lo, hi):
        rs_ = [r for r in J["records"] if r.get("T_over_rs") is not None and lo <= r["T_over_rs"] <= hi]
        out = {"n": len(rs_)}
        for key in ("halvings", "iterations"):
            v = [r.get(key) for r in rs_ if isinstance(r.get(key), (int, float))]
            out[key] = {"min": min(v), "max": max(v), "mean": float(np.mean(v))} if v else None
        out["converged_all"] = all(bool(r.get("converged", True)) for r in rs_)
        out["intersample_labels"] = sorted({str(r.get("intersample_label")) for r in rs_})
        out["solvers"] = sorted({str(r.get("solver")) if not isinstance(r.get("solver"), dict) else json.dumps(r.get("solver"), sort_keys=True)[:200] for r in rs_})
        return out
    Zf1 = np.load(C.FLD1, allow_pickle=True); Zf2 = np.load(C.FLD2, allow_pickle=True)
    common = [t for t in T1 if 28.0 - 1e-9 <= t <= 40.0 + 1e-9]
    fd = []
    for t in common:
        i1 = int(np.argmin(np.abs(T1 - t))); i2 = int(np.argmin(np.abs(T2 - t)))
        if abs(T2[i2] - t) < 1e-9:
            fd.append(max(float(np.abs(H1.N_e[i1] - H2.N_e[i2]).max()), float(np.abs(H1.v_e[i1] - H2.v_e[i2]).max())))
    OUT["item1_inputs_at_40"] = {
        "switch": "CombinedHistory: records with T <= 40 r_s from the r11 files, records after from the ext100 files (record json and applied-field npz); "
                  "the chain (8192 sites, vacuum fields at the record's E beyond the engine's 2048 sites) and the propagator are unchanged",
        "r11": {"fields": C.FLD1, "n_records": len(T1), "T_range": [float(T1[0]), float(T1[-1])], "dT_over_rs_unique": sorted({round(float(x), 6) for x in np.diff(T1)}),
                "tau": H1.tau, "ne": H1.ne, "ell": H1.ell, "label": H1.label, "fields_keys": list(Zf1.files), "script": J1.get("script")},
        "ext100": {"fields": C.FLD2, "n_records": len(T2), "T_range": [float(T2[0]), float(T2[-1])], "dT_over_rs_unique": sorted({round(float(x), 6) for x in np.diff(T2)}),
                   "tau": H2.tau, "ne": H2.ne, "ell": H2.ell, "label": H2.label, "fields_keys": list(Zf2.files), "script": J2.get("script"), "runner": J2.get("runner")},
        "args_differing": diff,
        "records_r11_32_40": stats(J1, 32.0, 40.0), "records_ext100_32_40": stats(J2, 32.0, 40.0), "records_ext100_40_56": stats(J2, 40.125, 56.0),
        "applied_fields_r11_vs_ext100_28_40": {"n_common": len(fd), "max": max(fd) if fd else None, "at_40": fd[-1] if fd else None},
        "combined_dT_after_40_unique": sorted({round(float(x), 6) for x in np.diff(np.asarray(H.T) / RS)[np.asarray(H.T)[1:] / RS > 40.0]})}
    print(json.dumps(OUT["item1_inputs_at_40"], indent=1, default=C.jdefault)[:6000], flush=True)
    # ---------------- 2. where along the chain the difference sits
    loc = {}
    for t in (48.0, 56.0):
        key = "T%06.3f" % t
        Zs = np.load(os.path.join(V11, "s%d" % int(t), "W7_slice_%s.npz" % key))
        R = Zs["R"]; n = len(R)
        dsum = np.zeros(n); tail = 0.0; dfar = 0.0
        for k in C.KAPPAS:
            Z = np.load(os.path.join(V1, "W7_ancestry_k%d.npz" % k))
            Sp = Z["T12p125_state_all_T%06.3f" % t]; f = Z["T12p125_f"]
            wp = C.site_w(Sp, Sp.shape[0] // 2)
            dsum += MULT[k] * (np.abs(wp[:n] - Zs["k%d_sitew" % k]) @ f)
            tail += MULT[k] * float(wp[n:].sum(axis=0) @ f)
        tot = float(dsum.sum())
        edges = [0, 8, 16, 20, 24, 28, 30, 31, 32]
        bins = {"%g-%g r_s" % (edges[i], edges[i + 1]): float(dsum[(R > edges[i] * RS) & (R <= edges[i + 1] * RS)].sum()) for i in range(len(edges) - 1)}
        cs = np.cumsum(dsum)
        loc[key] = {"weighted_L1_total": tot, "by_radius": bins, "peak_R_over_rs": float(R[int(np.argmax(dsum))] / RS), "median_R_over_rs": float(R[int(np.searchsorted(cs, 0.5 * tot))] / RS),
                    "last_64_sites_share": float(dsum[-64:].sum() / tot), "propagated_weight_beyond_32rs_weighted": tail}
    OUT["item2_location"] = loc
    print(json.dumps(loc, indent=1), flush=True)
    # ---------------- 3. the attribution test 48 -> 50 on the two chains
    TT = np.asarray(H.T) / RS
    k48 = int(np.argmin(np.abs(TT - 48.0))); k50 = int(np.argmin(np.abs(TT - 50.0)))
    ns_ = types.SimpleNamespace(propagator="ops", cheb_tol=1e-14, eig_backend="scipy", eig_driver="evr")
    BE = D2.Backend(ns_, torch, BV2, torch.device("cpu"))
    OPS = BE.OPS
    Z48 = np.load(os.path.join(V11, "slices", C.EXT_FMT % 48.0)); Z50 = np.load(os.path.join(V11, "slices", C.EXT_FMT % 50.0))
    test = {"from_T": float(TT[k48]), "to_T": float(TT[k50]), "records": k50 - k48, "kappas": {}}
    Rfull = np.asarray(H.R, float)
    for k in C.KAPPAS:
        D48, f, _, _, _ = C.loewdin_dust(Z48, k)
        D50, _, _, _, _ = C.loewdin_dust(Z50, k)
        n = D48.shape[0] // 2
        res = {}
        for name, nsites in (("chain_2048", n), ("chain_8192", len(Rfull))):
            R = Rfull[:nsites]
            h0, Q = OPS.chain_sparse(R, C.M_MASS, float(k))
            X = np.zeros((2 * nsites, D48.shape[1]), np.complex128); X[:2 * n] = D48
            for kk in range(k48, k50):
                Nk, vk = D2.fields(H, kk)
                X, _ = BE.propagate(OPS.clocked_csr(h0, Q, Nk[:nsites], vk[:nsites]), X, float(H.T[kk + 1] - H.T[kk]))
            ov = np.abs((D50.conj() * X[:2 * n]).sum(axis=0)) ** 2
            l1 = np.abs(C.site_w(X[:2 * n], n) - C.site_w(D50, n)).sum(axis=0)
            res[name] = {"one_minus_overlap_max": float((1 - ov).max()), "site_weight_L1_max": float(l1.max()),
                         "weight_beyond_32rs_max": float(C.site_w(X, nsites)[n:].sum(axis=0).max()) if nsites > n else 0.0}
        test["kappas"][str(k)] = res
        print("kappa %d: %s" % (k, json.dumps(res)), flush=True)
    OUT["item3_attribution_48_to_50"] = test
    C.dump_json(OUT, os.path.join(a.out_dir, "W7_record_items_v1_2.json"))


if __name__ == "__main__":
    main()
