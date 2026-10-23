#!/usr/bin/env python
"""engine_star_v1.py (lead's draft; SPEC_final4b_engine.md sec. 4): the self-consistent static star of the declared model in the radial
reduction.  Per channel kappa the eigenmodes of h_m^{(kappa)}[N, v]; the positive band filled with f = fermi(E; E_F, T_s) (E the global
eigenvalue; E_F < m); the sources by form (i); the constraints re-solved; damped mixing to convergence.  Outputs per member and the
family against E_F.  Usage: python engine_star_v1.py --m 0.2 --g 0.25 --EF 0.9 --nsites 512 --out <dir> --tag <tag> [--gpu 0]."""
import argparse
import json
import os
import time

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v1 as FLD


def ovpg_start(eos, m, g, EF, R):
    """BUILDER: the start profile from the continuum control (engine_ovpg_v1) at the same Fermi level on the stable branch: z_c by
    bisection on EF/m(z_c) between 0.05 and the mass maximum's z_c (EF/m falls monotonically there), the shot's (N, v) on the lattice radii."""
    import engine_ovpg_v1 as OV
    e_of, P_of, n_of = OV.load_eos(eos, m)
    def ef(z):
        try:
            return OV.shoot(m, m, g, e_of, P_of, n_of, m * np.sqrt(1 + z * z))["EF_consistent"] / m
        except (IndexError, ValueError):                      # the most dilute shots find no surface within the range
            return np.nan
    zs = np.geomspace(0.05, 3.0, 40)
    vals = np.array([ef(z) for z in zs])
    good = np.isfinite(vals)
    i_min = int(np.nanargmin(vals))
    lo, hi = float(zs[good][0]), float(zs[i_min])
    if not (ef(hi) <= EF / m <= ef(lo)):
        return None, None, {"note": "EF outside the continuum family's stable range", "EF_over_m_range": [ef(hi), ef(lo)]}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ef(mid) > EF / m else (lo, mid)
    z = 0.5 * (lo + hi)
    res = OV.shoot(m, m, g, e_of, P_of, n_of, m * np.sqrt(1 + z * z))
    N, v, mm = OV.profile_on_lattice(res, g, R)
    return N, v, {"z_c": z, "EF_over_m": ef(z), "M_cont": res["M"], "R_surf_cont": res["R_star"]}


def star_member(EF, m, g, nsites, Ts=None, N=None, v=None, alpha=0.3, tol=1e-11, maxpass=200, kappa_cap=200, half_offset=False, log=None):
    Ts = 0.01 * m if Ts is None else Ts
    parts = {}
    R = OPS.chain_parts(2, m, 1.0, half_offset)[2]
    R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
    N = np.ones(nsites) if N is None else np.array(N, float)
    v = np.zeros(nsites) if v is None else np.array(v, float)
    hist = []
    for ip in range(maxpass):
        eps = np.zeros(nsites); J = np.zeros(nsites); NB = 0.0; Esum = 0.0; kmax = 0; levels = {}
        for kappa in range(1, kappa_cap + 1):
            if kappa not in parts:
                parts[kappa] = OPS.chain_parts(nsites, m, float(kappa), half_offset)
            h, Q, _ = parts[kappa]
            e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
            pos = e > 0
            f = OPS.fermi(e[pos], EF, Ts)
            if f.max() < 1e-12:
                break
            kmax = kappa
            w = 16.0 * kappa
            e1, J1 = OPS.sources_static(h, Q, U[:, pos], f, w)
            eps += e1; J += J1; NB += w * f.sum(); Esum += w * float(f @ e[pos])
            levels[kappa] = e[pos][f > 1e-6].tolist()
        Nn, vn, Wf, info = FLD.solve_constraints(eps, J, g, R, N0=N)
        dN = float(np.max(np.abs(Nn - N))); dv = float(np.max(np.abs(vn - v)))
        N = N + alpha * (Nn - N); v = v + alpha * (vn - v)
        hist.append((dN, dv))
        if log:
            print("pass %d dN %.2e dv %.2e NB %.3f E %.6f kmax %d" % (ip, dN, dv, NB, FLD.charge(Wf, g, nsites - 1), kmax), file=log, flush=True)
        if dN < tol and dv < tol:
            break
    # final: the un-mixed solve at the converged sources, residuals, outputs
    N, v, Wf, info = FLD.solve_constraints(eps, J, g, R, N0=N)
    r1, r2 = FLD.residuals(eps, J, g, R, N, v, Wf)
    cum = np.cumsum(N ** 2 * eps)
    E = float(cum[-1])
    r_star = float(R[np.searchsorted(cum, 0.999 * E)])
    collar = min(nsites - 1, int(r_star) + 20)
    out = {"EF": EF, "Ts": Ts, "m": m, "g": g, "nsites": nsites, "passes": len(hist), "last_dN_dv": hist[-1], "kappa_max": kmax,
           "N_B": NB, "E_collar": FLD.charge(Wf, g, collar), "E_gauss": E, "E_collar_end": FLD.charge(Wf, g, nsites - 1),
           "sum_f_E": Esum, "binding": E - NB * m, "r_star": r_star, "compactness": 2 * g * E / r_star, "N_centre": float(N[0]),
           "min_N_minus_absv": float(np.min(N - np.abs(v))), "central_density": float(eps[0] / (4 * np.pi * R[0] ** 2)),
           "residual_H_max": float(np.max(np.abs(r1))), "residual_M_max": float(np.max(np.abs(r2))),
           "exterior": FLD.exterior_check(N, v, R, Wf, g, E, r_star), "solver": info, "levels": levels,
           "N": N.tolist(), "v": v.tolist(), "eps_tot": eps.tolist(), "J_tot": J.tolist()}
    return out


def family(args):
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "star_%s.log" % args.tag), "a")
    members = []
    N = v = None
    EFs = np.arange(args.EF_start, args.EF_stop - 1e-12, -args.EF_step) * args.m
    start_info = None
    if args.start_eos:
        R0 = np.arange(args.nsites, dtype=float) + (0.5 if args.half_offset else 1.0)
        N, v, start_info = ovpg_start(args.start_eos, args.m, args.g, float(EFs[0]), R0)
        print("start profile: %s" % start_info, flush=True)
    for EF in EFs:
        t0 = time.time()
        mem = star_member(float(EF), args.m, args.g, args.nsites, N=N, v=v, alpha=args.alpha, half_offset=bool(args.half_offset), log=log)
        mem["seconds"] = time.time() - t0
        N, v = np.array(mem["N"]), np.array(mem["v"])                # continuation
        members.append(mem)
        summary = family_summary(members)
        with open(os.path.join(args.out, "star_%s.json" % args.tag), "w") as fh:
            json.dump({"script": "engine_star_v1.py", "args": vars(args), "start": start_info, "summary": summary, "members": members}, fh, default=float)
        print("EF/m %.4f E %.4f NB %.1f R* %.1f C %.4f minN-|v| %.4f passes %d (%.0f s)" % (
            EF / args.m, mem["E_gauss"], mem["N_B"], mem["r_star"], mem["compactness"], mem["min_N_minus_absv"], mem["passes"], mem["seconds"]), flush=True)
    # BUILDER (completed in family_summary): the first-law residual per interior member and the parabola maximum.  The scan stops at
    # --EF-stop (0.845 by default: the continuum family's Fermi level turns at about 0.843 m beyond the mass maximum, below which no member
    # exists; the central-density continuation past the turning point is not run, labelled).


def family_summary(members):
    ok = [mm for mm in members if mm.get("E_gauss") is not None]
    out = {"n": len(ok), "first_law": []}
    for a_, b_, c_ in zip(ok[:-2], ok[1:-1], ok[2:]):
        dN = c_["N_B"] - a_["N_B"]
        if dN != 0:
            out["first_law"].append({"EF": b_["EF"], "residual": (c_["E_gauss"] - a_["E_gauss"]) / (b_["EF"] * dN) - 1})
    if len(ok) >= 3:
        Es = [mm["E_gauss"] for mm in ok]
        i = int(np.argmax(Es))
        out["max_member"] = {k: ok[i][k] for k in ("EF", "E_gauss", "N_B", "r_star", "compactness", "N_centre", "min_N_minus_absv")}
        if 0 < i < len(ok) - 1:
            x = np.array([ok[i - 1]["EF"], ok[i]["EF"], ok[i + 1]["EF"]]); y = np.array(Es[i - 1:i + 2])
            c2, c1, c0 = np.polyfit(x, y, 2)
            xm = -c1 / (2 * c2)
            out["E_max_parabola"] = float(c0 + c1 * xm + c2 * xm * xm)
            out["EF_at_max"] = float(xm)
        out["max_compactness"] = max(mm["compactness"] for mm in ok)
        out["min_over_family_of_min_N_minus_absv"] = min(mm["min_N_minus_absv"] for mm in ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--nsites", type=int, default=512)
    ap.add_argument("--EF-start", type=float, default=0.995)
    ap.add_argument("--EF-stop", type=float, default=0.845)
    ap.add_argument("--start-eos", default=None, help="eos_v4_tables.npz: start from the continuum member at the first E_F (BUILDER)")
    ap.add_argument("--EF-step", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    OPS.GPU["dev"] = a.gpu
    family(a)


if __name__ == "__main__":
    main()
