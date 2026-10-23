#!/usr/bin/env python
"""w256end_slices_v1.py (busi; inbox #477/#478): W256_s4 run_f's record sources and fields re-solved from the stored complex128 slices,
zero-card (box CPU, torch on the host). Per slice clock T:
  eps_tot = sum_k w_k [dens(h_k, sea_k) - dens(h_k, R2_k) + dens(h_k, dust_k, f_k)],  J_tot the same with Q_k
(engine_collapse_v36nq_rd13mfd's sources() at the record's converged cache: the slice's sea is the record's state, R2 its accepted
reference; Chan and dens are engine_collapse_v5's own). Then the record's site fields by the engine's own solve (engine_fields_v16,
rtol 1e-12, atol 1e-16, collar flag, support 1e-14, N0 None, kernel-averaged site fields), on BOTH routes: collocation (the run's
--fields collocation) and the ivp (the gate code's six-term route). Writes one npz per slice and a json of the checks.
  w256end_slices_v1.py OUT_DIR CODE_DIR SLICE [SLICE ...]
"""
import json, os, sys, time, hashlib
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
import numpy as np

OUT, CODE = sys.argv[1:3]; SL = sys.argv[3:]
sys.path.insert(0, CODE)
import torch  # noqa: E402
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import engine_collapse_v5 as C5  # noqa: E402
import engine_fields_v16 as FLD  # noqa: E402
C5.backend(None)
m, g, ell, nsites, K = 0.2, 0.25, 4.0, 4096, 14
chs = [C5.Chan(k, m, nsites, False) for k in range(1, K + 1)]
R = chs[0].R


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def site_solve(eps, J, route):
    FLD.rd2_configure(route)
    Nn, vn, Wn, info = FLD.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        return None
    Na, va, norms = FLD.site_fields_kernel_average(info, R, ell)
    return Na, va, Nn, vn, float(FLD.charge(info, g)), info


out = {}
for p in SL:
    t0 = time.time()
    Z = np.load(p)
    T = float(Z["T"]); rs = float(Z["rs"]); assert str(Z["dtype"]) == "complex128", Z["dtype"]
    assert np.allclose(np.asarray(Z["R"], float), R)
    w = np.asarray(Z["weights"], float); assert np.allclose(w, [ch.w for ch in chs])
    eps = np.zeros(nsites); J = np.zeros(nsites); eps_pairs = np.zeros(nsites); J_pairs = np.zeros(nsites); eps_dust = np.zeros(nsites)
    for ch in chs:
        k = ch.kappa
        sea = torch.as_tensor(np.asarray(Z["sea_%d" % k]), dtype=torch.complex128)
        ref = torch.as_tensor(np.asarray(Z["R2_%d" % k]), dtype=torch.complex128)
        dust = torch.as_tensor(np.asarray(Z["dust_%d" % k]), dtype=torch.complex128); f = np.asarray(Z["f_%d" % k], float)
        hs_, Qs_ = C5.dens(ch.h, sea), C5.dens(ch.Q, sea)
        dd, Jd = C5.dens(ch.h, dust, f), C5.dens(ch.Q, dust, f)
        ds = hs_ - C5.dens(ch.h, ref); Js = Qs_ - C5.dens(ch.Q, ref)
        eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
        del sea, ref, dust
        print("T %.4f channel %d done (%.0f s)" % (T / rs, k, time.time() - t0), flush=True)
    Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
    res = {}
    arr = {"T": T, "R": R, "eps_tot": eps, "J_tot": J, "eps_pairs": eps_pairs, "J_pairs": J_pairs, "eps_dust": eps_dust, "N_applied": Napp, "v_applied": vapp}
    for route in ("collocation", "ivp"):
        s = site_solve(eps, J, route)
        if s is None:
            res[route] = {"failed": True}; continue
        Na, va, Np, vp, E, info = s
        arr["N_%s" % route] = Na; arr["v_%s" % route] = va
        f_ = Na - np.abs(va); neg = np.nonzero(f_ < 0)[0]
        res[route] = {"E": E, "max_abs_N_minus_Napplied": float(np.max(np.abs(Na - Napp))), "max_abs_v_minus_vapplied": float(np.max(np.abs(va - vapp))),
                      "min_N_minus_absv": float(f_.min()), "r_min": float(R[int(np.argmin(f_))]),
                      "outer_neg_site": (float(R[neg[-1]]) if len(neg) else None), "inner_neg_site": (float(R[neg[0]]) if len(neg) else None)}
    fa = Napp - np.abs(vapp)
    res["applied"] = {"min_N_minus_absv": float(fa.min()), "r_min": float(R[int(np.argmin(fa))])}
    np.savez(os.path.join(OUT, "resolved_T%07.4f.npz" % (T / rs)), **arr)
    out["%.4f" % (T / rs)] = {"slice": p, "slice_md5": md5(p), "checks": res, "seconds": time.time() - t0}
    print(json.dumps({"T": T / rs, "checks": res}, default=str), flush=True)
    del Z
json.dump(out, open(os.path.join(OUT, "w256end_slices_v1.json"), "w"), indent=1, default=str)
print("done", flush=True)
