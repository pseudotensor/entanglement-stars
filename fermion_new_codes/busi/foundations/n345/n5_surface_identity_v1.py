#!/usr/bin/env python
"""n5_surface_identity_v1.py (busi, foundations item N5; ReviewA_foundations_v1.md sec. 3.2 N5, Chain 2, P14 'The energy as a surface
term').

DATA script (box CPU, no card, no evolution). On one stored slice: rebuild the shell totals from the slice's own columns
(engine_ops_v1.sources_dynamic: sum_kappa 16 kappa Tr[(Phi Phi^dag - R2 R2^dag) e_n] and the same with the momentum allocation),
compare them with the run's record at the slice's clock, re-solve the rows with the field module (engine_fields_v16.solve_constraints_ivp,
ivp route, collar_policy flag, its collar margin printed), and evaluate on a fine radial grid to the collar r_c:

  I_1 = int_0^{r_c} 4 pi r^2 (2 N rho_ell + v J_ell) dr             against 2E   (E the collar charge E_collar)
  I_2 = E_m - int A d^3y, E_m = int 4 pi r^2 (N rho_ell + v J_ell) dr, A = (N/16 pi G)[tr K^2 - (tr K)^2] from the dense fields'
        derivatives alone (K_r = v'/N, K_t = v/(r N): tr K^2 - (tr K)^2 = -2 K_t (2 K_r + K_t)), against 2E (Euler's identity, Chain 2)
  I_3 = sum_n (2 N_n eps_n + v_n J_n) on the sites with the back-averaged site fields (engine_fields_v16.site_fields_kernel_average; the
        adjoint of the averaging, Chain 9), and the same with the slice's stored applied fields N_applied, v_applied, against 2E.
The identity is the rows' consequence d/dr[r v^2/N] = 4 pi G r^2 (2 N rho + v J) with N(r_c) = 1, r_c v^2 = 2 G E.
Output: <outdir>/n5_<tag>.json. Reads the slice memory-mapped (slices144_io_v1.SliceReader); never writes beside its inputs."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

VERSION = "n5_surface_identity_v1"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--code", default="<internal>/code16d")
    ap.add_argument("--slices-io", default="<internal>/slices144")
    ap.add_argument("--tag", required=True); ap.add_argument("--outdir", default="<internal>/n345")
    ap.add_argument("--no-slice-md5", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.outdir, "n5_%s.json" % a.tag)
    if os.path.exists(out):
        sys.exit("refusing to overwrite " + out)
    sys.path.insert(0, a.code); sys.path.insert(1, a.slices_io)
    import engine_ops_v1 as OPS
    import engine_fields_v16 as FLD
    import slices144_io_v1 as SIO
    t0 = time.time()
    prov = {"script_md5": md5(os.path.abspath(__file__)), "engine_ops_v1": md5(OPS.__file__), "engine_fields_v16": md5(FLD.__file__),
            "slices144_io_v1": md5(SIO.__file__), "fields_npz_md5": md5(a.fields), "run_json_md5": md5(a.json),
            "slice": a.slice, "slice_md5": None if a.no_slice_md5 else md5(a.slice)}
    print(json.dumps(prov), flush=True)
    run = json.load(open(a.json)); args = run["args"]
    m = float(args["m"]); g = float(args["g"]); ell = int(args["ell"]); half = bool(int(args.get("half_offset", 0) or 0))
    SR = SIO.SliceReader(a.slice)
    R = SR.R; ns = SR.nsites
    eps = np.zeros(ns); J = np.zeros(ns)
    ortho = {}
    for k in SR.kappas:
        h, Q, Rk = OPS.chain_parts(ns, m, float(k), half_offset=half)
        assert np.allclose(Rk, R)
        ch = SR.channel(k, orthonormalize=False)           # the stored columns [sea | dust], f (sea 1, dust its occupations)
        Phi = ch["Phi"]; f = ch["f"]
        R2 = np.asarray(SR.R2(k))
        ortho[k] = float(ch["stored_orthonormality"])
        h = sp.csr_matrix(h); Q = sp.csr_matrix(Q)          # sparse: the products A @ Phi without dense BLAS
        e_k, J_k = OPS.sources_dynamic(h, Q, Phi, f, R2, w=SR.weights[k])
        eps += e_k; J += J_k
        print("kappa %d: n_occ %d, orthonormality defect %.2e (%.0f s)" % (k, Phi.shape[1], ortho[k], time.time() - t0), flush=True)
    Z = np.load(a.fields, allow_pickle=True)
    Td = np.asarray(Z["T"], float)
    kT = int(np.argmin(np.abs(Td - SR.T)))
    rec_eps = np.asarray(Z["eps_tot"][kT], float); rec_J = np.asarray(Z["J_tot"][kT], float)
    cmp_rec = {"record_T": float(Td[kT]), "slice_T": SR.T, "eps_max_rel": float(np.max(np.abs(eps - rec_eps)) / np.max(np.abs(rec_eps))),
               "J_max_rel": float(np.max(np.abs(J - rec_J)) / max(float(np.max(np.abs(rec_J))), 1e-300))}
    print("record comparison", cmp_rec, flush=True)
    Ns, vs, Wn, info = FLD.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        sys.exit("rows failed: %s" % info["failed"])
    E = float(info["E_collar"]); rc = float(info["r_collar"])
    tot_e = float(np.abs(eps).sum())
    supp = {}
    for thr in (1e-6, 1e-10, 1e-14):
        s = np.where((np.abs(eps) > thr * np.abs(eps).max()) | (np.abs(J) > thr * max(float(np.abs(J).max()), 1e-300)))[0]
        supp["R_last_rel_%g" % thr] = float(R[s.max()]) if len(s) else 0.0
    collar = {"collar_margin": info.get("collar_margin"), "r_collar": rc, "chain_end": float(R[-1]), "support": supp,
              "eps_share_beyond_r_collar_minus_8ell": float(np.abs(eps[R > rc - 8 * ell]).sum()) / tot_e}
    # fine grid (Simpson) to the collar
    nfine = int(round(rc / (ell / 64.0)))
    if nfine % 2:
        nfine += 1
    r = np.linspace(0.0, rc, nfine + 1); hh = r[1] - r[0]
    wS = np.ones(nfine + 1); wS[1:-1:2] = 4.0; wS[2:-1:2] = 2.0; wS *= hh / 3.0
    N = info["N_dense"](r); v = info["v_dense"](r); rho = info["rho_ell"](r); Jl = info["J_ell"](r)
    I1 = float(np.sum(wS * 4 * np.pi * r ** 2 * (2 * N * rho + v * Jl)))
    Em = float(np.sum(wS * 4 * np.pi * r ** 2 * (N * rho + v * Jl)))
    d = ell / 128.0
    vp = (info["v_dense"](r + d) - info["v_dense"](r - d)) / (2 * d)
    Kr = vp / N
    with np.errstate(invalid="ignore", divide="ignore"):
        Kt = np.where(r > 0, v / (np.where(r > 0, r, 1.0) * N), Kr)
    Adens = (N / (16 * np.pi * g)) * (-2.0 * Kt * (2.0 * Kr + Kt))
    intA = float(np.sum(wS * 4 * np.pi * r ** 2 * Adens))
    I2 = Em - intA
    # the energy row pointwise: N rho vs -A (A = -N rho on a solution)
    m_in = (r > ell) & (r < rc - ell)
    row_res = float(np.max(np.abs(Adens[m_in] + N[m_in] * rho[m_in])) / max(float(np.max(np.abs(N * rho))), 1e-300))
    # site sums with the back-averaged fields and with the slice's stored applied fields
    Na, va, norms = FLD.site_fields_kernel_average(info, R, ell)
    I3 = float(np.sum(2 * Na * eps + va * J))
    I3s = float(np.sum(2 * SR.N_applied * eps + SR.v_applied * J)) if SR.has_applied else None
    # the boundary value r v^2/(G N) at the collar
    bval = float(rc * info["v_dense"](np.array([rc]))[0] ** 2 / (g * info["N_dense"](np.array([rc]))[0]))
    rec = [x for x in run["records"] if abs(float(x["T"]) - SR.T) < 1e-9]
    res = {"version": VERSION, "provenance": prov, "slice_T": SR.T, "slice_T_over_rs": SR.T_over_rs, "kappas": SR.kappas,
           "orthonormality_defect_max": max(ortho.values()), "record_comparison": cmp_rec, "collar": collar,
           "E_collar": E, "two_E": 2 * E, "E_record": (rec[0].get("E") if rec else None),
           "I1_int_2Nrho_plus_vJ": I1, "I1_over_2E_minus_1": I1 / (2 * E) - 1.0,
           "E_m": Em, "int_A": intA, "I2_Em_minus_intA": I2, "I2_over_2E_minus_1": I2 / (2 * E) - 1.0,
           "energy_row_pointwise_rel": row_res,
           "I3_site_sum_backaveraged": I3, "I3_over_2E_minus_1": I3 / (2 * E) - 1.0,
           "I3_site_sum_stored_applied": I3s, "I3s_over_2E_minus_1": (I3s / (2 * E) - 1.0) if I3s is not None else None,
           "boundary_r_v2_over_GN_at_collar": bval, "boundary_over_2E_minus_1": bval / (2 * E) - 1.0,
           "terms": {"int_2Nrho": float(np.sum(wS * 4 * np.pi * r ** 2 * 2 * N * rho)), "int_vJ": float(np.sum(wS * 4 * np.pi * r ** 2 * v * Jl))},
           "fields": {"N_min": float(N.min()), "max_abs_v": float(np.max(np.abs(v))), "min_N_minus_abs_v": float(np.min(N - np.abs(v))),
                      "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "regime": info.get("regime", {}).get("label"),
                      "gauge": info.get("gauge", {}).get("label")},
           "kernel_norms_minmax": [float(norms.min()), float(norms.max())], "elapsed_s": time.time() - t0}
    with open(out, "w") as fo:
        json.dump(res, fo, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
