#!/usr/bin/env python
"""W7_slices_v1.py (the solve lane; reorientation 6 composition 2, W7's five items; box CPU). The data script for the stored slices.
The slices the definition names: the production's r11 orbital slices at T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35, 40 and the ext100
continuation's 48, 50, 52, 54, 56. Per slice: its presence on the box, its md5 (and, for the ext100 slices, the md5 the box's manifest
recorded), its five dust channels (dust_k, f_k, sea_k for kappa 1..5) and dtype. For every present slice and kappa 1..5 (per copy):
- the dust columns as engine_single_object_B5_prep_v2 builds them (Loewdin of [sea, dust] in float64), their per-site weights and the columns
  themselves (for the composer's same-set check against the propagated set and its slice-to-slice ledger);
- on the slice's own chain (2048 sites, R_j = j a) and its stored applied fields (N_applied, v_applied; equal to the history's fields at the
  record, checked): the held operator h = clocked_csr(chain_sparse(R, m, kappa), N, v);
- the number currents i[h, Pi_(R > c)] per column through 16 r_s, 24 r_s, the record's dense outer root and the applied outer root;
- the Killing split of the exterior part beyond 16 r_s: on the held exterior block (export_horizon_refcol_v2's block: h restricted to the
  sites R > 16 r_s; dense eigendecomposition) and, as the alternative, on the full held operator's eigenvectors (dust_ledger_postpass_v1b),
  with each eigenvector's weight inside 2 r_s and 16 r_s for the extended/bound proxy; the weights (levels x columns) and the levels saved.
Outputs <out-dir>/W7_slices.json and W7_slice_T<T>.npz per slice. Refuses to overwrite a slice's npz (skips it).
Usage (box): nice -n 10 python W7_slices_v1.py --threads 8 --out-dir <internal>/W7_v1
"""
import os
import sys

_t = "8"
for _i, _s in enumerate(sys.argv):
    if _s == "--threads" and _i + 1 < len(sys.argv):
        _t = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import W7_common_v1 as C   # noqa: E402

import argparse   # noqa: E402
import json       # noqa: E402
import time       # noqa: E402

import numpy as np          # noqa: E402
import scipy.linalg as sla  # noqa: E402

T0 = time.time()


def log(msg):
    print("[+%7.0f s] %s" % (time.time() - T0, msg), flush=True)


def records_by_T(path, rs=64.0):
    with open(path) as fh:
        J = json.load(fh)
    recs = J["records"]
    return {round(float(r["T_over_rs"]), 6): r for r in recs if r.get("T_over_rs") is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default="<internal>/W7_v1")
    ap.add_argument("--no-eigh", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of slice clocks (a dry run)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch, BV2, HB, D2 = C.load_engine()
    torch.set_num_threads(a.threads)
    OPS = BV2.OPS
    m = C.M_MASS; b = HB.band_top(m); rs = 64.0
    rec1 = records_by_T(C.REC1); rec2 = records_by_T(C.REC2)
    H = BV2.CombinedHistory(C.REC1, C.FLD1, 8192, C.REC2, C.FLD2, 40.0)
    TT = np.asarray(H.T, float) / float(H.rs_nom)
    pj = os.path.join(a.out_dir, "W7_slices.json")
    out = json.load(open(pj)) if os.path.exists(pj) else {"script": "W7_slices_v1.py", "slices": {}}
    out.update({"script_md5": C.md5(os.path.abspath(__file__)), "common_md5": C.md5(C.__file__), "ops_module": OPS.__file__, "ops_module_md5": C.md5(OPS.__file__),
                "bins_module_md5": C.md5(HB.__file__), "m": m, "b": b, "rs": rs, "G": C.G_NEWTON, "records": [C.REC1, C.REC2]})
    # the ext100 slices: the manifest's md5s and every place on the box they could be
    man = {}
    if os.path.exists(C.EXT_MANIFEST):
        for line in open(C.EXT_MANIFEST):
            p = line.split()
            if len(p) == 2 and p[1].endswith(".npz") and "/solve/B_ext100_slices/" in p[1]:
                man[os.path.basename(p[1])] = p[0]
    ext_dirs = ["<internal>/B_ext100_slices", "<internal>/c19_rs64_A_R2_tau8_ell4_avg_ext100",
                "<internal>/states", "<internal>/states"]
    plan = []
    for T in C.SLICE_T_R11:
        plan.append((T, C.SLICE_FMT % T, rec1.get(round(T, 6)), None))
    for T in C.SLICE_T_EXT:
        nm = C.EXT_FMT % T
        found = [os.path.join(d, nm) for d in ext_dirs if os.path.exists(os.path.join(d, nm))]
        plan.append((T, found[0] if found else os.path.join(ext_dirs[0], nm), rec2.get(round(T, 6)), man.get(nm)))
    only = [float(x) for x in a.only.split(",")] if a.only else None
    for T, path, rec, man_md5 in plan:
        if only is not None and T not in only:
            continue
        key = "T%06.3f" % T
        S = out["slices"].get(key, {})
        S.update({"T": T, "path": path, "present": os.path.exists(path), "manifest_md5": man_md5,
                  "record": {"E": rec.get("E") if rec else None, "trapping_zeros": rec.get("trapping_zeros") if rec else None, "R_t": rec.get("R_t") if rec else None}})
        out["slices"][key] = S
        pz = os.path.join(a.out_dir, "W7_slice_%s.npz" % key)
        if not S["present"]:
            log("%s: MISSING on the box (%s); manifest md5 %s" % (key, path, man_md5))
            C.dump_json(out, pj); continue
        if os.path.exists(pz):
            log("%s: npz exists, skipped" % key); continue
        t_s = time.time()
        S["md5"] = C.md5(path); S["md5_matches_manifest"] = (S["md5"] == man_md5) if man_md5 else None
        S["bytes"] = os.path.getsize(path)
        Z = np.load(path)
        S["keys_present"] = {str(k): all(("%s_%d" % (p, k)) in Z.files for p in ("sea", "dust", "f")) for k in C.KAPPAS}
        Rs = np.asarray(Z["R"], float); n = len(Rs); N = np.asarray(Z["N_applied"], float); v = np.asarray(Z["v_applied"], float)
        k = int(np.argmin(np.abs(TT - float(Z["T"]) / rs)))
        Nh, vh = D2.fields(H, k)
        S.update({"T_stored_over_rs": float(Z["T"]) / rs, "n_sites": n, "weights": np.asarray(Z["weights"]).tolist(), "dtype": str(Z["dust_1"].dtype),
                  "history_record": k, "fields_vs_history_max": float(max(np.abs(N - Nh[:n]).max(), np.abs(v - vh[:n]).max())), "R_max": float(Rs[-1])})
        rt_d, ri_d = C.record_roots(rec) if rec else (None, None)
        rt_a, ri_a, za = C.applied_outer_inner(Rs, N + v)
        S["roots"] = {"dense_outer": rt_d, "dense_inner": ri_d, "applied_outer": rt_a, "applied_inner": ri_a}
        arrays = {"R": Rs, "N": N, "v": v}
        rows16 = C.block_rows(Rs, 16.0 * rs); mask16 = np.repeat(Rs > 16.0 * rs, 2)
        S["kappas"] = {}
        for kap in C.KAPPAS:
            t_k = time.time()
            D, fd, idem, orth, raw = C.loewdin_dust(Z, kap)
            nd = D.shape[1]
            h0, Q = OPS.chain_sparse(Rs, m, float(kap))
            h = OPS.clocked_csr(h0, Q, N, v)
            sw = C.site_w(D, n)
            arrays["k%d_cols" % kap] = D; arrays["k%d_f" % kap] = fd; arrays["k%d_sitew" % kap] = sw
            arrays["k%d_sitew_stored" % kap] = C.site_w(raw, n)
            cur = {}
            for cname, cR in (("16", 16.0 * rs), ("24", 24.0 * rs), ("outer_dense", rt_d), ("outer_applied", rt_a)):
                if cR:
                    J = OPS.number_current_csr(h, Rs, cR)
                    cur[cname] = ((D.conj() * (J @ D)).sum(axis=0)).real
                    arrays["k%d_J_%s" % (kap, cname)] = cur[cname]
            K = {"n_dust": nd, "f": fd.tolist(), "stored_purity_defect": idem, "loewdin_orthonormality": orth,
                 "loewdin_minus_stored_b16_max": float(np.abs(sw[Rs > 16 * rs].sum(axis=0) - arrays["k%d_sitew_stored" % kap][Rs > 16 * rs].sum(axis=0)).max()) if nd else 0.0,
                 "b16_per_column": sw[Rs > 16 * rs].sum(axis=0).tolist(), "b24_per_column": sw[Rs > 24 * rs].sum(axis=0).tolist(),
                 "J16_per_column": cur["16"].tolist(), "hermiticity": float(abs(h - h.getH()).max())}
            if not a.no_eigh and nd:
                Hb = np.asarray(h[rows16][:, rows16].toarray(), np.complex128)
                eB, UB = sla.eigh(Hb, driver="evr", overwrite_a=True, check_finite=False); del Hb
                arrays["k%d_block16_e" % kap] = eB; arrays["k%d_block16_w" % kap] = C.weights_in_basis(UB, D[rows16]); del UB
                eA, UA = sla.eigh(np.asarray(h.toarray(), np.complex128), driver="evr", overwrite_a=True, check_finite=False)
                arrays["k%d_full_e" % kap] = eA
                arrays["k%d_full_w_ext16" % kap] = C.weights_in_basis(UA, D * mask16[:, None])
                arrays["k%d_full_w_whole" % kap] = C.weights_in_basis(UA, D)
                for Rc in (2.0, 16.0):
                    arrays["k%d_full_win_%g" % (kap, Rc)] = (np.abs(UA[:2 * int(np.sum(Rs <= Rc * rs)), :]) ** 2).sum(axis=0)
                del UA
            K["seconds"] = time.time() - t_k
            S["kappas"][str(kap)] = K
            log("%s kappa %d: %d dust columns (%s), purity %.1e, Loewdin-minus-stored beyond 16 r_s %.1e; beyond 16 r_s %s; J16 %s; %.0f s" % (
                key, kap, nd, S["dtype"], idem, K["loewdin_minus_stored_b16_max"], np.round(K["b16_per_column"], 5).tolist(), np.round(K["J16_per_column"], 6).tolist(), K["seconds"]))
        del Z
        S["seconds"] = time.time() - t_s
        C.save_npz(arrays, pz)
        C.dump_json(out, pj)
        log("%s done in %.0f s (md5 %s)" % (key, S["seconds"], S["md5"]))
    C.dump_json(out, pj)


if __name__ == "__main__":
    main()
