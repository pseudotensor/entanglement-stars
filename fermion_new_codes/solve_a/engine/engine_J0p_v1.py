#!/usr/bin/env python
"""engine_J0p_v1.py (builder analyst; SPEC_final4b_engine.md sec. 11.4, job J0'): the checks of the hydrodynamic solver engine_fields_v2.py.
(1) the l = 0 kernel's charge per shell (Simpson on a fine grid) and the l = 1 kernel's continuity at the series switch z = 1e-2; (2) the
regular centre (rho_ell'(0), J_ell(0), W/r^3 near 0, N'(0)); (3) the exterior (N = 1, r v^2 = 2 g E beyond the collar); (4) the charge at the
collar, + 2 ell, + 4 ell and with R_max doubled, against the hydrodynamic Gauss sum; (5) the ell-scan on the prescribed Gaussian source of the
draft's main block (J = 0: E = sum eps and N = 1; then the iterated static-frame momentum J_n = |v_n| eps_n/N_n to convergence: E(ell),
N_min(ell), the quadratic trend in ell, the sign check N < 1); (6) the spectra of the interpolated fields and the interpolation floors; (7) the
pair iteration count and the independence of rtol (1e-10 against 1e-12); (8) C1.9, the OV-PG hydrodynamic check: J-OVPG's continuum
maximum members at (m a, g) = (0.1, 1/4) and (0.1, 1/16), their sources from the continuum identities rho = (r v^2)'/(8 pi g r^2 N^2),
J = (v/(4 pi g r)) (1/N)' on a fine areal grid, binned to shell totals (shell n = [n + 1/2, n + 3/2) about R_n = n + 1, the centre [0, 1/2)
added to shell 0), fed to the solver at ell = 4, 8, 16: the recovered (N, v) at the sites against the continuum profile and E against M;
(9) the timing of one solve at n_sites 2048 and 4096.  Writes <out>/J0p_checks.json and J0p_checks.md (every number printed by the script).
Args: --eos (eos_v4_tables.npz) --ovpg (ovpg_families.json) --out."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import make_interp_spline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_fields_v2 as FLD  # noqa: E402
import engine_ovpg_v1 as OV  # noqa: E402

ELLS = (2, 4, 8, 16)


def check_kernels(ells, R):
    out = {}
    for ell in ells:
        dev = 0.0
        for Rn in R:
            r = np.linspace(0.0, Rn + 14.0 * ell, int((Rn + 14.0 * ell) / (ell / 64.0)) + 1)
            K0, _ = FLD.kernels(r, [Rn], ell)
            dev = max(dev, abs(simpson(4.0 * np.pi * r ** 2 * K0[:, 0], x=r) - 1.0))
        # the l = 1 kernel at the series switch: z = r R/ell^2 = 1e-2 from both sides
        Rn = 1.0
        r_sw = 1e-2 * ell ** 2 / Rn
        K_lo = FLD.kernels([r_sw * (1 - 1e-9)], [Rn], ell)[1][0, 0]
        K_hi = FLD.kernels([r_sw * (1 + 1e-9)], [Rn], ell)[1][0, 0]
        K0_lo = FLD.kernels([r_sw * (1 - 1e-9)], [Rn], ell)[0][0, 0]
        K0_hi = FLD.kernels([r_sw * (1 + 1e-9)], [Rn], ell)[0][0, 0]
        out[str(ell)] = {"max_abs_charge_minus_1": dev, "K1_rel_jump_at_switch": abs(K_hi / K_lo - 1.0), "K0_rel_jump_at_switch": abs(K0_hi / K0_lo - 1.0)}
    return out


def iterate_static(eps, g, R, ell, rtol=1e-12, maxouter=80, tol=1e-12):
    N, v, W, info = FLD.solve_constraints(eps, 0 * eps, g, R, ell, rtol=rtol)
    hist = []
    for k in range(maxouter):
        J = np.abs(v) * eps / N
        N2, v2, W, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N, rtol=rtol)
        d = float(max(np.max(np.abs(N2 - N)), np.max(np.abs(v2 - v))))
        N, v = N2, v2
        hist.append(d)
        if d < tol:
            break
    return N, v, W, info, J, hist


def centre_checks(eps, J, g, R, ell, info):
    src = FLD.SmoothedSources(eps, J, R, ell, info["r_nodes"][-1])
    N_spl = FLD.parity_spline(info["r_nodes"], info["N_nodes"], +1)
    rhsW = lambda r, W: [8.0 * np.pi * g * r * r * float(N_spl(r)) ** 2 * float(src.rho(r))]
    sol = solve_ivp(rhsW, (0.0, 3.0 * ell), [0.0], method="DOP853", dense_output=True, rtol=1e-12, atol=1e-30)
    rs = np.array([0.05, 0.1, 0.2, 0.4]) * ell
    Wr3 = [float(sol.sol(x)[0] / x ** 3) for x in rs]
    return {"rho_ell_prime_0": float(src.rho.derivative()(0.0)), "J_ell_0": float(src.J(0.0)), "rho_ell_0": float(src.rho(0.0)),
            "W_over_r3_at_0.05_0.1_0.2_0.4_ell": Wr3, "W_over_r3_spread_rel": float(np.ptp(Wr3) / abs(np.mean(Wr3))) if np.mean(Wr3) else None,
            "W_over_r3_expected": float(8.0 * np.pi * g * float(N_spl(0.0)) ** 2 * float(src.rho(0.0)) / 3.0),
            "N_prime_0": float(N_spl.derivative()(0.0))}


def ovpg_profile(eos, m, g, z_c, dr=0.01, r_out_factor=3.0):
    e_of, P_of, n_of = OV.load_eos(eos, m)
    res = OV.shoot(m, m, g, e_of, P_of, n_of, m * np.sqrt(1 + z_c * z_c))
    Rs, M = res["R_star"], res["M"]
    rr = np.arange(dr, r_out_factor * Rs, dr)
    inside = rr <= Rs
    Y = res["sol"].sol(np.minimum(rr, Rs))
    mm = np.where(inside, Y[0], M)
    Phi = np.where(inside, Y[1] + res["c"], 0.5 * np.log(1 - 2 * g * M / rr))
    F = 1 - 2 * g * mm / rr
    N = np.exp(Phi) / np.sqrt(F)
    v = -N * np.sqrt(2 * g * mm / rr)
    return rr, N, v, M, Rs, res


def c19(eos, m, g, z_c, ells):
    rr, N, v, M, Rs, res = ovpg_profile(eos, m, g, z_c)
    W = rr * v ** 2
    sW = make_interp_spline(rr, W, k=5)
    sI = make_interp_spline(rr, 1.0 / N, k=5)
    rho = sW.derivative()(rr) / (8 * np.pi * g * rr ** 2 * N ** 2)
    Jd = (v / (4 * np.pi * g * rr)) * sI.derivative()(rr)
    nsites = int(np.ceil(rr[-1])) - 2
    R = np.arange(1, nsites + 1, dtype=float)
    edges = np.concatenate([[0.0], R + 0.5])
    cr = np.concatenate([[0.0], np.cumsum(0.5 * (4 * np.pi * rr[1:] ** 2 * rho[1:] + 4 * np.pi * rr[:-1] ** 2 * rho[:-1]) * np.diff(rr))])
    cJ = np.concatenate([[0.0], np.cumsum(0.5 * (4 * np.pi * rr[1:] ** 2 * Jd[1:] + 4 * np.pi * rr[:-1] ** 2 * Jd[:-1]) * np.diff(rr))])
    rr0 = np.concatenate([[0.0], rr])
    eps_n = np.diff(np.interp(edges, rr0, np.concatenate([[0.0], cr])))
    J_n = np.diff(np.interp(edges, rr0, np.concatenate([[0.0], cJ])))
    Nc = np.interp(R, rr, N); vc = np.interp(R, rr, v)
    out = {"m": m, "g": g, "z_c": z_c, "M_cont": M, "R_surf": Rs, "nsites": nsites, "sum_eps_n": float(eps_n.sum()), "by_ell": {}}
    for ell in ells:
        t0 = time.time()
        Nr, vr, Wn, info = FLD.solve_constraints(eps_n, J_n, g, R, ell)
        inner = R < Rs
        out["by_ell"][str(ell)] = {"E": FLD.charge(info, g), "E_over_M_minus_1": FLD.charge(info, g) / M - 1.0,
                                   "max_abs_dN": float(np.max(np.abs(Nr - Nc))), "max_abs_dv": float(np.max(np.abs(vr - vc))),
                                   "max_rel_dN_inside": float(np.max(np.abs(Nr[inner] / Nc[inner] - 1.0))),
                                   "max_rel_dv_inside_R_gt_ell": float(np.max(np.abs(vr[inner & (R > ell)] / vc[inner & (R > ell)] - 1.0))),
                                   "N_centre_rec": float(Nr[0]), "N_centre_cont": float(N[0]), "it": info["it"], "converged": info["converged"],
                                   "spectrum": info["spectrum"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
                                   "seconds": time.time() - t0}
        print("C1.9 (m %g, g %g) ell %d: E/M - 1 %.3e, max|dN| %.3e, max|dv| %.3e, rel dN inside %.3e, it %d (%.0f s)" % (
            m, g, ell, out["by_ell"][str(ell)]["E_over_M_minus_1"], out["by_ell"][str(ell)]["max_abs_dN"], out["by_ell"][str(ell)]["max_abs_dv"],
            out["by_ell"][str(ell)]["max_rel_dN_inside"], info["it"], time.time() - t0), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos", required=True)
    ap.add_argument("--ovpg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-c19", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {"script": "engine_J0p_v1.py", "args": vars(a)}
    t0 = time.time()
    # (1)
    rec["kernels"] = check_kernels(ELLS, np.arange(1, 401, dtype=float)[::7])
    print("(1) kernels:", json.dumps(rec["kernels"]), flush=True)
    # (5) with (2), (3), (4), (6), (7)
    R = np.arange(1, 401, dtype=float)
    eps = 0.5 * np.exp(-((R - 20.0) / 6.0) ** 2)
    g = 0.25
    rec["prescribed"] = {"source": "eps_n = 0.5 exp(-((R_n - 20)/6)^2), R_n = 1..400, g = 0.25", "sum_eps": float(eps.sum()), "by_ell": {}}
    for ell in ELLS:
        N0, v0, W0, info0 = FLD.solve_constraints(eps, 0 * eps, g, R, ell)
        N, v, W, info, J, hist = iterate_static(eps, g, R, ell)
        E = FLD.charge(info, g)
        R2 = np.arange(1, 801, dtype=float); eps2 = np.concatenate([eps, np.zeros(400)]); J2 = np.concatenate([J, np.zeros(400)])
        _, _, _, info2 = FLD.solve_constraints(eps2, J2, g, R2, ell, N0=np.concatenate([N, np.ones(400)]))
        Nb, vb, Wb, infob = FLD.solve_constraints(eps, J, g, R, ell, N0=N, rtol=1e-10)
        gs = FLD.gauss_sum(eps, R, ell, FLD.parity_spline(info["r_nodes"], info["N_nodes"], +1), R[-1])
        row = {"J0": {"E": FLD.charge(info0, g), "E_minus_sum_eps": FLD.charge(info0, g) - float(eps.sum()), "max_abs_N_minus_1": float(np.max(np.abs(N0 - 1.0))),
                      "charge_conservation_quadrature": info0["charge_conservation_quadrature"]},
               "static": {"E": E, "E_collar_plus_2ell": FLD.charge(info, g, 2), "E_collar_plus_4ell": FLD.charge(info, g, 4),
                          "E_Rmax_doubled": FLD.charge(info2, g), "gauss_hydro": gs, "E_minus_gauss": E - gs, "N_min": float(N.min()),
                          "sign_check_N_lt_1": bool(N.min() < 1.0), "outer_iterations": len(hist), "last_outer_change": hist[-1] if hist else None,
                          "pair_iterations_last_solve": info["it"], "converged": info["converged"], "exterior": FLD.exterior_check(N, v, R, info, g),
                          "spectrum": info["spectrum"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
                          "rtol_1e-10_minus_1e-12": {"max_abs_dN": float(np.max(np.abs(Nb - N))), "max_abs_dv": float(np.max(np.abs(vb - v))),
                                                     "dE": FLD.charge(infob, g) - E},
                          "centre": centre_checks(eps, J, g, R, ell, info), "r_collar": info["r_collar"]}}
        rec["prescribed"]["by_ell"][str(ell)] = row
        st = row["static"]
        print("(5) ell %d: J=0 E - sum eps %.2e max|N-1| %.1e | static E %.10f (+2ell %.2e, +4ell %.2e, Rmax x2 %.2e, gauss %.2e) N_min %.8f outer %d "
              "floor %.1e/%.1e spectrum N %.1e v %.1e rtol dN %.1e" % (
                  ell, row["J0"]["E_minus_sum_eps"], row["J0"]["max_abs_N_minus_1"], E, st["E_collar_plus_2ell"] - E, st["E_collar_plus_4ell"] - E,
                  st["E_Rmax_doubled"] - E, st["E_minus_gauss"], st["N_min"], st["outer_iterations"], st["interp_floor_N"] or 0, st["interp_floor_v"] or 0,
                  st["spectrum"]["N"]["above_pi_over_ell"], st["spectrum"]["v"]["above_pi_over_ell"], st["rtol_1e-10_minus_1e-12"]["max_abs_dN"]), flush=True)
    Es = np.array([rec["prescribed"]["by_ell"][str(l)]["static"]["E"] for l in ELLS]); Ns = np.array([rec["prescribed"]["by_ell"][str(l)]["static"]["N_min"] for l in ELLS])
    x = np.array(ELLS, float) ** 2
    rec["prescribed"]["quadratic_trend"] = {"E_vs_ell2": np.polyfit(x, Es, 1).tolist(), "N_min_vs_ell2": np.polyfit(x, Ns, 1).tolist(),
                                            "note": "fits a + b ell^2 over ell = 2, 4, 8, 16 (the source's width 6 is sub-regime for ell >= 4 by design)"}
    # (8) C1.9
    if not a.skip_c19:
        fam = json.load(open(a.ovpg))["families"]
        rec["C1.9"] = []
        for (m, g_) in ((0.1, 0.25), (0.1, 0.0625)):
            f = [F for F in fam if abs(F["m"] - m) < 1e-12 and abs(F["g"] - g_) < 1e-12][0]
            rec["C1.9"].append(c19(a.eos, m, g_, f["maximum"]["z_c"], (4, 8, 16)))
    # (9) timing
    rec["timing"] = {}
    for ns in (2048, 4096):
        Rt = np.arange(1, ns + 1, dtype=float)
        et = 0.5 * np.exp(-((Rt - ns / 6.0) / (ns / 20.0)) ** 2)
        t1 = time.time()
        Nt, vt, Wt, it_ = FLD.solve_constraints(et, 0 * et, 0.25, Rt, 8)
        Jt = np.abs(vt) * et / Nt
        t2 = time.time()
        _, _, _, it2 = FLD.solve_constraints(et, Jt, 0.25, Rt, 8, N0=Nt)
        t3 = time.time()
        rec["timing"][str(ns)] = {"seconds_J0": t2 - t1, "seconds_with_J": t3 - t2, "pair_iterations": it2["it"]}
        print("(9) n_sites %d ell 8: %.1f s (J = 0), %.1f s (with J, %d pair iterations)" % (ns, t2 - t1, t3 - t2, it2["it"]), flush=True)
    rec["seconds"] = time.time() - t0
    json.dump(rec, open(os.path.join(a.out, "J0p_checks.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    L = ["# J0' (engine_J0p_v1.py): the hydrodynamic solver's checks (SPEC_final4b_engine.md sec. 11.4)", ""]
    L.append("(1) kernels: " + "; ".join("ell %s: max |charge - 1| %.2e, K1 jump %.1e, K0 jump %.1e" % (k, v["max_abs_charge_minus_1"], v["K1_rel_jump_at_switch"], v["K0_rel_jump_at_switch"]) for k, v in rec["kernels"].items()))
    L.append("")
    L.append("| ell | J = 0: E - sum eps | J = 0: max abs(N - 1) | E (static) | E(+2 ell) - E | E(+4 ell) - E | E(R_max x 2) - E | E - Gauss sum | N_min | outer it. | floor N | floor v | power above pi/ell (N, v) | rtol change of N | W/r^3 spread near 0 | N'(0) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ell in ELLS:
        r_ = rec["prescribed"]["by_ell"][str(ell)]; s = r_["static"]
        L.append("| %d | %.2e | %.1e | %.10f | %.1e | %.1e | %.1e | %.1e | %.8f | %d | %.1e | %.1e | %.1e, %.1e | %.1e | %.1e | %.1e |" % (
            ell, r_["J0"]["E_minus_sum_eps"], r_["J0"]["max_abs_N_minus_1"], s["E"], s["E_collar_plus_2ell"] - s["E"], s["E_collar_plus_4ell"] - s["E"],
            s["E_Rmax_doubled"] - s["E"], s["E_minus_gauss"], s["N_min"], s["outer_iterations"], s["interp_floor_N"] or 0, s["interp_floor_v"] or 0,
            s["spectrum"]["N"]["above_pi_over_ell"], s["spectrum"]["v"]["above_pi_over_ell"], s["rtol_1e-10_minus_1e-12"]["max_abs_dN"],
            s["centre"]["W_over_r3_spread_rel"] or 0, s["centre"]["N_prime_0"]))
    L.append("")
    L.append("- quadratic trends over ell = 2, 4, 8, 16 (a + b ell^2): E %s, N_min %s." % (rec["prescribed"]["quadratic_trend"]["E_vs_ell2"], rec["prescribed"]["quadratic_trend"]["N_min_vs_ell2"]))
    for c in rec.get("C1.9", []):
        L.append("- C1.9 (m a %g, g %g; M_cont %.6f, R_surf %.2f): " % (c["m"], c["g"], c["M_cont"], c["R_surf"]) + "; ".join(
            "ell %s: E/M - 1 %.2e, max abs dN %.2e, max abs dv %.2e, max rel dN inside %.2e" % (k, v["E_over_M_minus_1"], v["max_abs_dN"], v["max_abs_dv"], v["max_rel_dN_inside"])
            for k, v in c["by_ell"].items()))
    L.append("- timing at ell = 8: " + "; ".join("n_sites %s: %.1f s (J = 0), %.1f s (with J)" % (k, v["seconds_J0"], v["seconds_with_J"]) for k, v in rec["timing"].items()))
    open(os.path.join(a.out, "J0p_checks.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
