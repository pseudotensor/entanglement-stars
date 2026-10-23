#!/usr/bin/env python
"""engine_star_v3.py (builder analyst; version 3 = version 2 plus --mixing anderson (engine_mixing_v1: Anderson type II on the
concatenated fields (N, v), depth 6, beta 0.5, with the safeguard), the fixed point unchanged, and the sources from the occupied
columns only (identical sums; version 2 multiplied every positive mode); version 2 of the lead's engine_star_v1 per SPEC_final4b_engine.md sec. 11.10 items (1) to (6), with
the loop corrections the reviewer B red team named (sec. 6 of its report) that do not change the physics; the filling rule is a switch until the
lead rules on it (turn 19)).

The self-consistent static star of the declared model at the Hartree level in the radial reduction: per channel kappa the eigenmodes of
h_m^{(kappa)}[N, v] on the sites; the positive band filled (--filling: "fermi" the spec's smearing T_s; "zeroT_EF" every mode below E_F
fully, none above; "zeroT_NB" the lowest N_B states with the last degenerate set filled fractionally, E_F its energy); the sources by form
(i); the hydrodynamic constraint solve (engine_fields_v2: the averages over ell, the coarse grid, the smooth interpolation); damped mixing.
Version 2's changes against version 1: (1) engine_fields_v2 and --ell (default 8) in every solve; (2) the charge from the solver's collar
(FLD.charge(info, g)) with the collar controls, the hydrodynamic Gauss identity, the interpolation floors, the spectra and the pair
iterations recorded per member; (3) E and r_star from the matter (r_star encloses 0.999 of sum eps_tot; R_surf, where the last occupied
level's classically allowed region ends, is not computed here: BUILDER-pending), compactness 2 g E/r_star, the exterior check at the
collar, the residuals of the continuum equations at the nodes with the final sources, the sign check min N < 1; (4) the ell-control at the
maximum member and at E_F/m = 0.95 (re-solved from the production fields at each ell of --ell-control), the three-ell table in the summary;
(5) the red team's loop items: after the loop one more diagonalization at the final fields and the sources recomputed from those eigenstates
(the outer residual: the change of the fields when re-solved with them is printed and recorded); a member is flagged "failed" when the
outer loop or the solver does not converge, when the channel cap is reached with a nonnegligible last channel, when the solver flags a
negative W or a nonpositive lapse, or when the omitted-channel bound (the last kept channel's weighted occupation times the number of
channels it would take to reach the zero of the tail by a linear extrapolation) exceeds 1e-8 of N_B; (6) no dense operator cache across
channels (version 1 kept two dense 2N x 2N arrays per channel: about 97 GB at 80 channels and 2N = 6144); the chain parts are rebuilt per
channel per pass (milliseconds).  Lattice units.  Usage: python engine_star_v2.py --m 0.1 --g 0.25 --ell 8 --filling zeroT_EF
--EF-start 0.95 --EF-stop 0.86 --EF-step 0.01 --start-eos eos_v4_tables.npz --nsites 720 --out <dir> --tag <tag> [--gpu 2]."""
import argparse
import json
import os
import time

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v2 as FLD
import engine_mixing_v1 as MIX


def ovpg_start(eos, m, g, EF, R):
    """the continuum control's member at the same Fermi level on the stable branch, its (N, v) on the lattice radii (version 1's routine)."""
    import engine_ovpg_v1 as OV
    e_of, P_of, n_of = OV.load_eos(eos, m)

    def ef(z):
        try:
            return OV.shoot(m, m, g, e_of, P_of, n_of, m * np.sqrt(1 + z * z))["EF_consistent"] / m
        except (IndexError, ValueError):
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


def occupations(ep_all, filling, EF, Ts, NB_target=None, tol=1e-9):
    """ep_all: list over channels of (kappa, positive energies). Returns per channel the occupations and the E_F used.
    fermi: f = 1/(e^((E - E_F)/T_s) + 1); zeroT_EF: f = 1 below E_F; zeroT_NB: the lowest NB_target states (weights 16 kappa), the last
    degenerate set (energies within tol) filled fractionally."""
    if filling == "fermi":
        return [OPS.fermi(ep, EF, Ts) for _, ep in ep_all], EF
    if filling == "zeroT_EF":
        return [(ep < EF).astype(float) for _, ep in ep_all], EF
    # zeroT_NB
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ = [np.zeros(len(ep)) for _, ep in ep_all]
    left = float(NB_target)
    i = 0
    E_last = None
    while i < len(lev) and left > 1e-12:
        grp = [lev[i]]
        while i + len(grp) < len(lev) and abs(lev[i + len(grp)][0] - lev[i][0]) < tol:
            grp.append(lev[i + len(grp)])
        w = sum(gg[1] for gg in grp)
        frac = min(1.0, left / w)
        for gg in grp:
            occ[gg[2]][gg[3]] = frac
        left -= frac * w
        E_last = lev[i][0]
        i += len(grp)
    return occ, E_last


def star_member(EF, m, g, nsites, ell, filling="zeroT_EF", Ts=None, NB_target=None, N=None, v=None, alpha=0.3, tol=1e-11, maxpass=200,
                kappa_cap=400, half_offset=False, log=None, mixing="anderson"):
    Ts = 0.01 * m if Ts is None else Ts
    R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
    N = np.ones(nsites) if N is None else np.array(N, float)
    v = np.zeros(nsites) if v is None else np.array(v, float)
    hist = []
    flags = []

    def channel_pass(N, v):
        ep_all, store = [], {}
        for kappa in range(1, kappa_cap + 1):
            h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half_offset)
            e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
            pos = e > 0
            ep = e[pos]
            ep_all.append((kappa, ep))
            store[kappa] = (h, Q, U[:, pos])
            # stop when the channel's lowest positive level lies above every level that can be occupied (E_F + 40 T_s for fermi;
            # E_F for zeroT_EF; for zeroT_NB the energy at which the weighted count of the levels found so far reaches N_B)
            if filling == "zeroT_NB":
                lv = np.sort(np.concatenate([np.repeat(x_[1], 1) for x_ in ep_all]))
                wv = np.concatenate([np.full(len(x_[1]), 16.0 * x_[0]) for x_ in ep_all])[np.argsort(np.concatenate([x_[1] for x_ in ep_all]))]
                cw = np.cumsum(wv)
                top = float(lv[min(int(np.searchsorted(cw, NB_target)), len(lv) - 1)]) if cw[-1] >= NB_target else np.inf
            else:
                top = EF + (40 * Ts if filling == "fermi" else 0.0)
            if ep.min() > top and kappa > 1:
                break
        occ, EF_used = occupations(ep_all, filling, EF, Ts, NB_target)
        eps = np.zeros(nsites); J = np.zeros(nsites); NB = 0.0; Esum = 0.0; kmax = 0; levels = {}; tail = 0.0
        for (kappa, ep), f in zip(ep_all, occ):
            if f.max() < 1e-15:
                continue
            h, Q, Up = store[kappa]
            w = 16.0 * kappa
            occ = f > 1e-15                                       # version 3: the sources from the occupied columns only (the rest carry f = 0)
            e1, J1 = OPS.sources_static(h, Q, Up[:, occ], f[occ], w)
            eps += e1; J += J1; NB += w * f.sum(); Esum += w * float(f @ ep); kmax = kappa
            levels[kappa] = ep[f > 1e-6].tolist()
            tail = w * f.sum()
        return eps, J, NB, Esum, kmax, levels, tail, len(ep_all), EF_used

    info = {}
    mixer = MIX.Anderson() if mixing == "anderson" else None
    for ip in range(maxpass):
        eps, J, NB, Esum, kmax, levels, tail, nch, EF_used = channel_pass(N, v)
        Nn, vn, Wn, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N)
        dN = float(np.max(np.abs(Nn - N))); dv = float(np.max(np.abs(vn - v)))
        if mixer is not None:
            x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
            N, v = x[:nsites], x[nsites:]
        else:
            N = N + alpha * (Nn - N); v = v + alpha * (vn - v)
        hist.append((dN, dv))
        if log:
            print("pass %d dN %.2e dv %.2e NB %.3f E %.6f kmax %d E_F %.6f it %d" % (ip, dN, dv, NB, FLD.charge(info, g), kmax, EF_used, info["it"]),
                  file=log, flush=True)
        if dN < tol and dv < tol:
            break
    if hist[-1][0] >= tol or hist[-1][1] >= tol:
        flags.append("outer loop not converged (maxpass %d)" % maxpass)
    # the red team's final recompute: one more diagonalization at the final fields, the sources from those eigenstates, the fields re-solved
    eps, J, NB, Esum, kmax, levels, tail, nch, EF_used = channel_pass(N, v)
    N2, v2, Wn, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N)
    outer = {"max_abs_dN": float(np.max(np.abs(N2 - N))), "max_abs_dv": float(np.max(np.abs(v2 - v)))}
    N, v = N2, v2
    if not info.get("converged", False):
        flags.append("constraint solver not converged")
    if info.get("negative_W") is not None:
        flags.append("negative W %.3e" % info["negative_W"])
    if info.get("nonpositive_lapse"):
        flags.append("nonpositive lapse (clipped by the solver)")
    if kmax >= kappa_cap - 1:
        flags.append("channel cap reached")
    if tail > 1e-8 * max(NB, 1e-300) and filling == "fermi":
        flags.append("omitted-channel tail %.2e of N_B" % (tail / NB))
    E = FLD.charge(info, g)
    cum = np.cumsum(eps)
    r_star = float(R[min(int(np.searchsorted(cum, 0.999 * cum[-1])), nsites - 1)]) if cum[-1] > 0 else None
    r1, r2 = FLD.residuals(info, g, eps, J, R)
    out = {"EF": EF, "EF_used": EF_used, "filling": filling, "Ts": Ts if filling == "fermi" else 0.0, "NB_target": NB_target, "m": m, "g": g,
           "ell": ell, "nsites": nsites, "passes": len(hist), "last_dN_dv": hist[-1], "outer_residual": outer, "kappa_max": kmax,
           "channels_diagonalized": nch, "N_B": NB, "E": E, "E_collar_plus_2ell": FLD.charge(info, g, 2), "E_collar_plus_4ell": FLD.charge(info, g, 4),
           "gauss_hydro": FLD.gauss_sum(eps, R, ell, FLD.parity_spline(info["r_nodes"], info["N_nodes"], +1), R[-1]),
           "sum_f_E": Esum, "binding": E - NB * m, "r_star": r_star, "compactness": (2 * g * E / r_star) if r_star else None,
           "N_centre": float(N[0]), "N_min": float(N.min()), "sign_check_N_lt_1": bool(N.min() < 1.0), "min_N_minus_absv": float(np.min(N - np.abs(v))),
           "trapping_radii": FLD.trapping_radius(N, v, R), "residual_H_max": float(np.max(np.abs(r1))), "residual_M_max": float(np.max(np.abs(r2))),
           "exterior": FLD.exterior_check(N, v, R, info, g), "r_collar": info["r_collar"], "solver_it": info["it"],
           "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"), "spectrum": info["spectrum"],
           "flags": flags, "failed": bool(flags), "levels": levels, "N": N.tolist(), "v": v.tolist(), "eps_tot": eps.tolist(), "J_tot": J.tolist(),
           "ell_line": "ell = %d" % ell}
    return out


def family_summary(members):
    ok = [mm for mm in members if not mm.get("failed")]
    out = {"n": len(members), "n_ok": len(ok), "first_law": []}
    for a_, b_, c_ in zip(ok[:-2], ok[1:-1], ok[2:]):
        dN = c_["N_B"] - a_["N_B"]
        if dN != 0:
            out["first_law"].append({"EF": b_["EF_used"], "residual": (c_["E"] - a_["E"]) / (b_["EF_used"] * dN) - 1})
    if len(ok) >= 3:
        Es = [mm["E"] for mm in ok]
        i = int(np.argmax(Es))
        out["max_member"] = {k: ok[i][k] for k in ("EF", "EF_used", "E", "N_B", "r_star", "compactness", "N_centre", "min_N_minus_absv")}
        if 0 < i < len(ok) - 1:
            x = np.array([ok[i - 1]["EF_used"], ok[i]["EF_used"], ok[i + 1]["EF_used"]]); y = np.array(Es[i - 1:i + 2])
            c2, c1, c0 = np.polyfit(x, y, 2)
            xm = -c1 / (2 * c2)
            out["E_max_parabola"] = float(c0 + c1 * xm + c2 * xm * xm)
            out["EF_at_max"] = float(xm)
        out["max_compactness"] = max(mm["compactness"] for mm in ok if mm["compactness"] is not None)
        out["min_over_family_of_min_N_minus_absv"] = min(mm["min_N_minus_absv"] for mm in ok)
    return out


def ell_control(member, ells, args, log):
    """re-solve the member at each ell from its production fields; the three-ell table (spread per output, the quadratic coefficient in ell)."""
    rows = {}
    for ell in ells:
        mem = star_member(member["EF"], args.m, args.g, args.nsites, ell, args.filling, args.Ts, member.get("NB_target"), N=member["N"], v=member["v"],
                          alpha=args.alpha, half_offset=bool(args.half_offset), log=log, mixing=args.mixing)
        rows[str(ell)] = {k: mem[k] for k in ("E", "N_B", "r_star", "compactness", "N_centre", "N_min", "min_N_minus_absv", "failed", "flags", "passes")}
    keys = ("E", "N_B", "compactness", "N_centre", "min_N_minus_absv")
    spread = {k: float(np.ptp([rows[str(l)][k] for l in ells if rows[str(l)][k] is not None])) for k in keys}
    quad = {k: (np.polyfit(np.array(ells, float) ** 2, [rows[str(l)][k] for l in ells], 1).tolist() if len(ells) >= 2 else None) for k in keys}
    return {"EF": member["EF"], "rows": rows, "spread": spread, "fit_a_plus_b_ell2": quad}


def family(args):
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "star_%s.log" % args.tag), "a")
    members = []
    N = v = None
    EFs = np.arange(args.EF_start, args.EF_stop - 1e-12, -args.EF_step) * args.m
    start_info = None
    R0 = np.arange(args.nsites, dtype=float) + (0.5 if args.half_offset else 1.0)
    if args.start_eos:
        N, v, start_info = ovpg_start(args.start_eos, args.m, args.g, float(EFs[0]), R0)
        print("start profile: %s" % start_info, flush=True)
    path = os.path.join(args.out, "star_%s.json" % args.tag)
    NBs = [float(x) for x in args.NB_list.split(",")] if args.NB_list else [None] * len(EFs)
    if args.NB_list:
        assert args.filling == "zeroT_NB", "--NB-list drives the family by the particle number: --filling zeroT_NB"
        EFs = [float(EFs[0])] * len(NBs)           # E_F is then an output (the last filled level); EFs[0] only fixes the start profile
    for EF, NBt in zip(EFs, NBs):
        t0 = time.time()
        mem = star_member(float(EF), args.m, args.g, args.nsites, args.ell, args.filling, args.Ts, NBt, N=N, v=v, alpha=args.alpha,
                          half_offset=bool(args.half_offset), log=log, mixing=args.mixing)
        mem["seconds"] = time.time() - t0
        if not mem["failed"]:
            N, v = np.array(mem["N"]), np.array(mem["v"])          # continuation from the last good member only
        members.append(mem)
        with open(path, "w") as fh:
            json.dump({"script": "engine_star_v3.py", "args": vars(args), "start": start_info, "summary": family_summary(members), "members": members}, fh, default=float)
        print("EF/m %.4f (used %.5f) E %.6f NB %.2f r* %.1f C %.4f N_min %.5f minN-|v| %.5f passes %d outer dN %.1e %s (%.0f s) ell = %d" % (
            EF / args.m, mem["EF_used"] / args.m, mem["E"], mem["N_B"], mem["r_star"] or -1, mem["compactness"] or -1, mem["N_min"], mem["min_N_minus_absv"],
            mem["passes"], mem["outer_residual"]["max_abs_dN"], ("FAILED " + "; ".join(mem["flags"])) if mem["failed"] else "ok", mem["seconds"], args.ell), flush=True)
    summ = family_summary(members)
    if args.ell_control:
        ells = [int(x) for x in args.ell_control.split(",")]
        ok = [mm for mm in members if not mm["failed"]]
        picks = []
        if "max_member" in summ:
            picks.append(min(ok, key=lambda mm: abs(mm["E"] - summ["max_member"]["E"])))
        dil = min(ok, key=lambda mm: abs(mm["EF"] / args.m - 0.95)) if ok else None
        if dil is not None and all(dil is not p for p in picks):
            picks.append(dil)
        summ["ell_control"] = [ell_control(p, ells, args, log) for p in picks]
    with open(path, "w") as fh:
        json.dump({"script": "engine_star_v3.py", "args": vars(args), "start": start_info, "summary": summ, "members": members}, fh, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--ell", type=int, default=8)
    ap.add_argument("--filling", default="zeroT_EF", choices=["fermi", "zeroT_EF", "zeroT_NB"])
    ap.add_argument("--Ts", type=float, default=None, help="the smearing for --filling fermi (default 0.01 m)")
    ap.add_argument("--nsites", type=int, default=512)
    ap.add_argument("--EF-start", type=float, default=0.995)
    ap.add_argument("--EF-stop", type=float, default=0.845)
    ap.add_argument("--EF-step", type=float, default=0.005)
    ap.add_argument("--start-eos", default=None)
    ap.add_argument("--NB-list", default=None, help="zeroT_NB families: comma list of particle numbers (E_F is then an output)")
    ap.add_argument("--ell-control", default="4,8,16")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--mixing", default="anderson", choices=["anderson", "linear"])
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    OPS.GPU["dev"] = a.gpu
    family(a)


if __name__ == "__main__":
    main()
