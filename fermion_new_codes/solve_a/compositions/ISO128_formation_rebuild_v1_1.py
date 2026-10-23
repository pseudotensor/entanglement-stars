#!/usr/bin/env python
"""ISO128_formation_readout v1.1 (inbox #255): rebuild a stored record of ISO_128_s4 (H2) from its complex128 slice.

The record at clock T in engine_collapse_v36nq (run()):
  base[k]  = (dens(h, sea), dens(Q, sea), dens(h, dust, f), dens(Q, dust, f))   from the orbitals at T (st[k], before the step's propagation)
  sources  = sum_k w_k (base - dens(., R2_k)) with R2_k the continued reference at the kept iterate   (run().sources / sources_with)
  fields   = solve(eps, J, g, R, ell, None)   (engine's solve: FLD.solve_constraints_ivp + site_fields_kernel_average)
  readout  = FLD.charge, FLD.trapping_radius, surface_jets, R_c/R_sh from eps_dust, N_min
The slice written at T holds exactly sea_k, dust_k, f_k, R2_k (= the kept iterate's reference) and N_applied, v_applied (the kept iterate).
So the record's own field solve is one call on the slice; nothing is iterated. This driver imports the engine's modules from an unchanged copy
of the code dir (v36nq 8549c542, engine_fields_v16 = v16b d7250a37) and calls the same functions in the same order; no engine file is edited.

usage: python -I ISO128_formation_rebuild_v1_1.py --code CODEDIR --slice SLICE.npz --run-json RUN.json --fields-npz FIELDS.npz --out OUT_PREFIX
env  : H2_PW_ONE=1 H2_LOCAL_SPLINE=1 (as the run), thread limits, NUMPY_MADVISE_HUGEPAGE=0. CPU torch (the run used the card: rounding-level differences).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--fields-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    o = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, o.code)
    import torch
    torch.set_num_threads(o.threads)
    import engine_collapse_v36nq as E
    J = json.load(open(o.run_json))
    a = J["args"]
    # main()'s configuration, from the run json's args block
    E.OPT["collar_policy"] = a["collar_policy"]; E.OPT["support_rel"] = a["support_rel"]; E.OPT["site_fields"] = a["site_fields"]
    E.SOLVE_TOL["rtol"] = a["field_rtol"]; E.SOLVE_TOL["atol"] = a["field_atol"]
    assert not a.get("fields_v17")
    if a.get("field_kernel_cache"):
        import engine_fields_kcache_v1 as KC
        import engine_fields_v4 as _F4, engine_fields_v12 as _F12, engine_fields_v14 as _F14, engine_fields_v16 as _F16
        KC.install([_F4, _F12, _F14, _F16, E.FLD]); E.V36["kcache"] = KC.STATS
    E.V36["fixes"] = bool(a.get("v36_fixes"))
    E.C5.backend(None)                                    # CPU
    FLD = E.FLD
    import engine_fields_v16 as F16
    env = {k: os.environ.get(k) for k in ("H2_PW_ONE", "H2_LOCAL_SPLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMPY_MADVISE_HUGEPAGE")}
    print("engine %s md5 %s; fields module %s md5 %s; H2 flags in module: PW_ONE %s LOCAL_SPLINE %s; env %s" % (
        E.__file__, md5(E.__file__), FLD.__name__, md5(F16.__file__), F16._H2_PW_ONE, F16._H2_LOCAL_SPLINE, env), flush=True)

    rs = a["rs"]; g = a["g"]; ell = a["ell"]; nsites = a["nsites"]
    chs = [E.Chan(k, a["m"], nsites, bool(a["half_offset"])) for k in range(1, a["kappa_sea"] + 1)]
    R = chs[0].R
    Z = np.load(o.slice)
    T = float(Z["T"]); assert str(Z["dtype"]) == "complex128"
    assert np.array_equal(np.asarray(Z["R"], float), R)
    assert np.array_equal(np.asarray(Z["kappas"]), np.array([ch.kappa for ch in chs]))
    assert np.array_equal(np.asarray(Z["weights"], float), np.array([ch.w for ch in chs]))
    N_app = np.asarray(Z["N_applied"], float); v_app = np.asarray(Z["v_applied"], float)
    print("slice T/rs %.4f loaded header (%.0f s)" % (T / rs, time.time() - t0), flush=True)

    # run().sources(cache_) with the record's base and the kept reference, channel by channel in the engine's order
    eps = np.zeros(nsites); Jt = np.zeros(nsites); eps_pairs = np.zeros(nsites); eps_dust = np.zeros(nsites); J_pairs = np.zeros(nsites)
    per_ch = {}
    for ch in chs:
        k = ch.kappa; tc = time.time()
        sea = torch.as_tensor(np.asarray(Z["sea_%d" % k]), dtype=torch.complex128)
        dust = torch.as_tensor(np.asarray(Z["dust_%d" % k]), dtype=torch.complex128)
        f = np.asarray(Z["f_%d" % k], float)
        ref = torch.as_tensor(np.asarray(Z["R2_%d" % k]), dtype=torch.complex128)
        hs_, Qs_, dd, Jd = E.dens(ch.h, sea), E.dens(ch.Q, sea), E.dens(ch.h, dust, f), E.dens(ch.Q, dust, f)
        ds = hs_ - E.dens(ch.h, ref); Js = Qs_ - E.dens(ch.Q, ref)
        eps += ch.w * (ds + dd); Jt += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
        per_ch[k] = {"n_sea": int(sea.shape[1]), "n_ref": int(ref.shape[1]), "n_dust": int(dust.shape[1])}
        del sea, dust, ref
        print("  channel %d: sea %d ref %d dust %d (%.0f s)" % (k, per_ch[k]["n_sea"], per_ch[k]["n_ref"], per_ch[k]["n_dust"], time.time() - tc), flush=True)

    # run().do_solve: the engine's solve (FLD.solve_constraints_ivp, then the kernel average of the site fields)
    ts = time.time()
    N, v, W, info = E.solve(eps, Jt, g, R, ell, None)
    print("field solve %.1f s; failed %s" % (time.time() - ts, info.get("failed")), flush=True)
    Ech = FLD.charge(info, g)
    E0 = float(J["head"]["E0"])
    tz = FLD.trapping_radius(N, v, R, info)
    Rt = tz["zeros"][-1] if tz["zeros"] else None
    r_in = tz["zeros"][0] if len(tz["zeros"]) >= 2 else None
    jets = E.surface_jets(info, Rt, g, Ech)
    Rc = Rsh = None
    tot = eps_dust.sum()
    if tot > 0:
        cum = np.cumsum(eps_dust); Rsh = float(R[min(int(np.searchsorted(cum, 0.99 * tot)), nsites - 1)]); Rc = float(eps_dust @ R / tot)
    rn = np.asarray(info["r_nodes"], float); Nn = np.asarray(info["N_nodes"], float); vn = np.asarray(info["v_nodes"], float)
    dn = Nn - np.abs(vn); i_n = int(np.argmin(dn))
    dsite = N - np.abs(v); i_s = int(np.argmin(dsite))
    out = {
        "slice": o.slice, "T": T, "T_over_rs": T / rs, "E": Ech, "E0": E0, "E_over_E0": Ech / E0,
        "E_dust_shells": float(eps_dust.sum()), "E_pairs_shells": float(eps_pairs.sum()),
        "R_t": Rt, "R_t_over_rs": (Rt / rs if Rt else None), "two_gE": 2 * g * Ech, "R_t_over_2gE": (Rt / (2 * g * Ech) if Rt else None),
        "r_in": r_in, "r_in_over_rs": (r_in / rs if r_in else None), "trapping_zeros": tz["zeros"], "tangencies": tz["tangencies"],
        "min_N_minus_absv": tz["min"], "r_min_N_minus_absv": tz["r_min"],
        "min_N_minus_absv_nodes": float(dn[i_n]), "r_at_min_nodes": float(rn[i_n]),
        "min_N_minus_absv_sites": float(dsite[i_s]), "r_at_min_sites": float(R[i_s]),
        "N_min": float(N.min()), "R_c": Rc, "R_sh": Rsh,
        "d1_d4_over_exact": (jets["over_exact"] if jets else None), "surface_jets": jets,
        "kept_residual_rebuilt": max(float(np.max(np.abs(N - N_app))), float(np.max(np.abs(v - v_app)))),
        "gauge_label": (info.get("gauge") or {}).get("label"), "regime": info.get("regime"),
        "channels": per_ch, "seconds": time.time() - t0, "engine_md5": md5(E.__file__), "fields_md5": md5(F16.__file__), "env": env,
    }
    # the stored record and fields row at this clock, if any
    recs = [r for r in J["records"] if r.get("T") is not None and abs(r["T"] - T) < 1e-9]
    if recs:
        r = recs[0]; s32 = r.get("solve_v32") or {}
        out["stored_record"] = {k: r.get(k) for k in ("E", "E_over_E0", "E_dust_shells", "E_pairs_shells", "R_t", "R_t_over_2gE", "r_in", "trapping_zeros",
                                                      "min_N_minus_absv", "r_min_N_minus_absv", "N_min", "R_c", "R_sh")}
        out["stored_record"]["kept_residual"] = s32.get("kept_residual"); out["stored_record"]["confirming_residual"] = s32.get("confirming_residual")
        out["stored_record"]["surface_jets"] = r.get("surface_jets")
    Zf = np.load(o.fields_npz, allow_pickle=True)
    Tz = np.asarray(Zf["T"], float); hit = np.where(np.abs(Tz - T) < 1e-9)[0]
    if len(hit):
        i = int(hit[0]); cmp_ = {}
        for nm, arr in (("eps_tot", eps), ("J_tot", Jt), ("eps_pairs", eps_pairs), ("J_pairs", J_pairs), ("N", N), ("v", v),
                        ("N_applied", N_app), ("v_applied", v_app)):
            s_ = np.asarray(Zf[nm][i], float)
            cmp_[nm] = {"max_abs_diff": float(np.max(np.abs(arr - s_))), "max_abs_stored": float(np.max(np.abs(s_)))}
        for nm, arr in (("r_nodes", rn), ("N_nodes", Nn), ("v_nodes", vn)):
            s_ = np.asarray(Zf[nm][i], float)
            cmp_[nm] = {"max_abs_diff": float(np.max(np.abs(arr - s_))) if s_.shape == arr.shape else "shape %s vs %s" % (s_.shape, arr.shape),
                        "max_abs_stored": float(np.max(np.abs(s_)))}
        out["stored_fields_row"] = cmp_
    else:
        out["stored_fields_row"] = None
    np.savez_compressed(o.out + "_nodefields.npz", T=T, R=R, r_nodes=rn, N_nodes=Nn, v_nodes=vn, N=N, v=v,
                        N_point=np.asarray(info.get("N_point", N), float), v_point=np.asarray(info.get("v_point", v), float),
                        eps_tot=eps, J_tot=Jt, eps_pairs=eps_pairs, J_pairs=J_pairs, eps_dust=eps_dust, N_applied=N_app, v_applied=v_app)
    json.dump(out, open(o.out + "_readout.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(json.dumps({k: out[k] for k in ("T_over_rs", "E_over_E0", "R_t", "R_t_over_rs", "two_gE", "R_t_over_2gE", "r_in", "min_N_minus_absv", "r_min_N_minus_absv",
                                          "min_N_minus_absv_nodes", "r_at_min_nodes", "min_N_minus_absv_sites", "N_min", "R_c", "R_sh", "kept_residual_rebuilt",
                                          "E_dust_shells", "E_pairs_shells")}, default=str), flush=True)
    if out.get("stored_record"):
        print("stored record:", json.dumps({k: v_ for k, v_ in out["stored_record"].items() if k != "surface_jets"}, default=str), flush=True)
    if out["stored_fields_row"]:
        print("stored fields row:", json.dumps(out["stored_fields_row"]), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
