#!/usr/bin/env python
"""engine_star_v5.py (builder analyst; version 5 = version 4 with two fixes from its first family run (Sfam4, rejected at its first member):
a member below the continuum seed's range (the most dilute shot that finds a surface) is seeded from the most dilute available continuum
member (labelled in the record), and a rejection before any converged member moves on to the next N_B instead of ending the scan.
Version 4's docstring: engine_star_v4.py (builder analyst; the static star per SPEC_final4b_engine.md secs. 11.10 and 12.1, 12.9 on engine_fields_v3, from
the builder's engine_star_v3).

The self-consistent static star of the declared model at the Hartree level: zero-temperature filling at fixed particle number N_B (the
family's parameter): the positive levels of all channels pooled with the weights 16 kappa and sorted, filled from the bottom until the
weight reaches N_B, at most one multiplet fractional (f = remainder/(16 kappa) on all its states); E_top the last filled level, E_next the
first unfilled, E_F := (E_top + E_next)/2 with the half gap as its error; the channel bound: channels are taken until two consecutive kappa
have their lowest positive level above E_top; E_top < m required (bound filling). The reference is the ground state of H_m[N, v] (the
transported sea coincides with it below the pocket threshold, which a static member never reaches: the goal of 09:07 UTC). The seed: the
continuum Oppenheimer-Volkoff-PG member at the same N_B (bisection on z_c along the stable branch), later members continuing from the
previous one. The loop: Anderson mixing (engine_mixing_v1) to max |dN|, |dv| < 1e-11 and |dE/E| < 1e-12; then the final pass: eigh at the
final fields, the sources from those eigenstates, the constraints solved without mixing, the outer residual max |N_out - N_in| (below 10 tol)
and the two constraint residuals at those sources. A member is rejected (recorded with its flag, not in the family) on: N_B not met,
E_top >= m, passes exhausted, the channel bound not reached at the cap, a solver failure (engine_fields_v3's info["failed"]), the outer
residual above 10 tol. The radii: R_surf where m sqrt(N^2 - v^2) = E_top (the turning point of the last level; the compactness by it) and
R_999 (0.999 of sum eps_tot). The family: N_B at the fractions --fractions of the continuum N_B,max of J-OVPG (default 0.1, 0.2, 0.35, 0.5,
0.7, 0.85, 0.95, 0.98, 0.99), then bisection between the last converged and the first rejected member to --bisect-tol (0.2 percent) in N_B;
the maximum mass is the last converged member's E with the bisection interval as its error. The ell-control at the maximum member and at the
member nearest E_F/m = 0.95 (re-solved from the production fields at each ell of --ell-control). --closed-count: every member also re-solved
at the nearest closed count (the last multiplet full) for engine_massfluct. Every record carries ell and the dictionary line.
Usage: python engine_star_v4.py --m 0.1 --g 0.25 --ell 8 --nsites 1536 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json
--out <dir> --tag <tag> [--gpu 0]"""
import argparse
import json
import os
import time

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v3 as FLD
import engine_mixing_v1 as MIX


def continuum_member_at_NB(eos, m, g, NB, z_lo=0.05, z_hi=None):
    """the continuum OV-PG member with the given N_B on the stable branch: bisection on z_c between z_lo and the mass maximum's z_c."""
    import engine_ovpg_v1 as OV
    e_of, P_of, n_of = OV.load_eos(eos, m)
    def nb(z):
        try:
            return OV.member(m, g, e_of, P_of, n_of, z)["N_B"]
        except (IndexError, ValueError):                      # the most dilute shots find no surface within the range
            return np.nan
    lo, hi = z_lo, z_hi
    while not np.isfinite(nb(lo)) and lo < hi:
        lo *= 1.5
    seed_note = None
    if NB < nb(lo):                                     # version 5: below the range, seed from the most dilute available member
        seed_note = "seeded from the most dilute continuum member (N_B %.3f > target %.3f)" % (nb(lo), NB)
        NB = nb(lo) * (1 + 1e-9)
    if not (nb(lo) <= NB <= nb(hi)):
        return None, {"note": "N_B outside the continuum stable branch", "NB_range": [nb(lo), nb(hi)]}
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if nb(mid) < NB else (lo, mid)
    z = 0.5 * (lo + hi)
    mem = OV.member(m, g, e_of, P_of, n_of, z)
    mem["seed_note"] = seed_note
    res = OV.shoot(m, m, g, e_of, P_of, n_of, m * np.sqrt(1 + z * z))
    return (res, z), mem


def channel_pass(N, v, m, nsites, NB, half_offset, kappa_cap=600):
    """all channels to the bound of sec. 12.1; the zero-temperature filling at N_B; the sources from the occupied columns."""
    ep_all, store = [], {}
    above = 0
    E_top_est = None
    for kappa in range(1, kappa_cap + 1):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half_offset)
        e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
        pos = e > 0
        ep = e[pos]
        ep_all.append((kappa, ep))
        store[kappa] = (h, Q, U[:, pos])
        # the running E_top: the N_B-th level of the pooled list so far (it can only fall as channels are added)
        lv = np.concatenate([x[1] for x in ep_all]); wv = np.concatenate([np.full(len(x[1]), 16.0 * x[0]) for x in ep_all])
        o = np.argsort(lv); cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][min(int(np.searchsorted(cw, NB - 1e-9)), len(lv) - 1)]) if cw[-1] >= NB else np.inf
        above = above + 1 if ep.min() > E_top_est else 0
        if above >= 2:
            break
    capped = above < 2
    # the filling: pooled sorted levels, the last multiplet fractional
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ = [np.zeros(len(ep)) for _, ep in ep_all]
    left = float(NB); i = 0; E_top = None; E_next = None
    while i < len(lev):
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        if left <= 1e-12:
            E_next = lev[i][0]
            break
        w = sum(x[1] for x in lev[i:j])
        frac = min(1.0, left / w)
        for x in lev[i:j]:
            occ[x[2]][x[3]] = frac
        left -= frac * w
        E_top = lev[i][0]
        if frac < 1.0:
            E_next = lev[j][0] if j < len(lev) else None
            # a fractional multiplet: the next level is the same multiplet's continuation; E_next is the next distinct level
        i = j
    eps = np.zeros(nsites); J = np.zeros(nsites); NBs = 0.0; Esum = 0.0; kmax = 0; levels = {}
    for (kappa, ep), f in zip(ep_all, occ):
        if f.max() < 1e-15:
            continue
        h, Q, Up = store[kappa]
        w = 16.0 * kappa
        o_ = f > 1e-15
        e1, J1 = OPS.sources_static(h, Q, Up[:, o_], f[o_], w)
        eps += e1; J += J1; NBs += w * f.sum(); Esum += w * float(f @ ep); kmax = kappa
        levels[kappa] = [(float(x), float(fx)) for x, fx in zip(ep[o_], f[o_])]
    fractional = any(((f > 1e-12) & (f < 1 - 1e-12)).any() for f in occ)
    return {"eps": eps, "J": J, "NB": NBs, "Esum": Esum, "kmax": kmax, "levels": levels, "E_top": E_top, "E_next": E_next,
            "channels": len(ep_all), "capped": capped, "fractional": fractional}


def star_member(NB, m, g, nsites, ell, N, v, half_offset=False, tol=1e-11, maxpass=300, log=None):
    R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
    mixer = MIX.Anderson()
    flags, hist = [], []
    E_prev = None
    info = {}
    for ip in range(maxpass):
        cp = channel_pass(N, v, m, nsites, NB, half_offset)
        Nn, vn, Wn, info = FLD.solve_constraints(cp["eps"], cp["J"], g, R, ell, N0=N)
        if info.get("failed"):
            flags.append("solver: " + info["failed"])
            break
        E = FLD.charge(info, g)
        dN = float(np.max(np.abs(Nn - N))); dv = float(np.max(np.abs(vn - v)))
        dE = abs(E - E_prev) / abs(E) if (E_prev is not None and E != 0) else np.inf
        E_prev = E
        hist.append((dN, dv, dE))
        if log:
            print("pass %d dN %.2e dv %.2e dE/E %.2e NB %.3f E %.8f kmax %d E_top %.6f it %d ell %d" % (ip, dN, dv, dE, cp["NB"], E, cp["kmax"],
                  cp["E_top"] or -1, info["it"], ell), file=log, flush=True)
        if dN < tol and dv < tol and dE < 1e-12:
            break
        x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
        N, v = x[:nsites], x[nsites:]
    else:
        flags.append("passes exhausted (%d)" % maxpass)
    out = {"NB_target": NB, "m": m, "g": g, "ell": ell, "nsites": nsites, "passes": len(hist), "dictionary": FLD.DICTIONARY}
    if any(f.startswith("solver") for f in flags):
        out.update({"failed": True, "flags": flags})
        return out
    # the final pass: eigh at the final fields, the sources from those eigenstates, the constraints solved without mixing
    cp = channel_pass(N, v, m, nsites, NB, half_offset)
    N2, v2, Wn, info = FLD.solve_constraints(cp["eps"], cp["J"], g, R, ell, N0=N)
    if info.get("failed"):
        flags.append("solver (final pass): " + info["failed"])
        out.update({"failed": True, "flags": flags})
        return out
    outer = {"max_abs_dN": float(np.max(np.abs(N2 - N))), "max_abs_dv": float(np.max(np.abs(v2 - v)))}
    N, v = N2, v2
    if outer["max_abs_dN"] > 10 * tol:
        flags.append("outer residual %.2e above 10 tol" % outer["max_abs_dN"])
    if abs(cp["NB"] - NB) > 1e-8 * max(NB, 1.0):
        flags.append("N_B not met (%.6f)" % cp["NB"])
    if cp["E_top"] is None or cp["E_top"] >= m:
        flags.append("unbound filling (E_top %s >= m)" % cp["E_top"])
    if cp["capped"]:
        flags.append("channel bound not reached at the cap")
    E = FLD.charge(info, g)
    r1, r2 = FLD.residuals(info, g, cp["eps"], cp["J"], R)
    cum = np.cumsum(cp["eps"])
    r999 = float(R[min(int(np.searchsorted(cum, 0.999 * cum[-1])), nsites - 1)]) if cum[-1] > 0 else None
    Ns = np.sqrt(np.maximum(N * N - v * v, 0.0))
    target = (cp["E_top"] or m) / m
    s = np.nonzero((Ns[:-1] - target) * (Ns[1:] - target) < 0)[0]
    R_surf = float(R[s[-1]] + (target - Ns[s[-1]]) * (R[s[-1] + 1] - R[s[-1]]) / (Ns[s[-1] + 1] - Ns[s[-1]])) if len(s) else None
    E_F = 0.5 * (cp["E_top"] + cp["E_next"]) if (cp["E_top"] is not None and cp["E_next"] is not None) else None
    tr = FLD.trapping_radius(N, v, R, info)
    out.update({"failed": bool(flags), "flags": flags, "N_B": cp["NB"], "E": E, "E_collar_plus_2ell": FLD.charge(info, g, 2), "E_collar_plus_4ell": FLD.charge(info, g, 4),
                "gauss_hydro": FLD.gauss_sum(cp["eps"], R, ell, FLD.parity_spline(info["r_nodes"], info["N_nodes"], +1), R[-1]),
                "E_top": cp["E_top"], "E_next": cp["E_next"], "E_F": E_F, "E_F_half_gap": (0.5 * (cp["E_next"] - cp["E_top"]) if E_F is not None else None),
                "fractional_multiplet": cp["fractional"], "binding": E - cp["NB"] * m, "R_surf": R_surf, "R_999": r999,
                "compactness_surf": (2 * g * E / R_surf) if R_surf else None, "compactness_999": (2 * g * E / r999) if r999 else None,
                "invariant_M_g15_ma2": E * g ** 1.5 * m * m, "N_centre": float(N[0]), "N_min": float(N.min()), "sign_check_N_lt_1": bool(N.min() < 1.0),
                "min_N_minus_absv": tr["min"], "r_min_N_minus_absv": tr["r_min"], "trapping_zeros": tr["zeros"],
                "residual_H_max": float(np.max(np.abs(r1))), "residual_M_max": float(np.max(np.abs(r2))), "outer_residual": outer,
                "exterior": "analytic by construction beyond r_collar = %g (N = 1, r v^2 = 2 g E)" % info["r_collar"], "r_collar": info["r_collar"],
                "channels": cp["channels"], "kappa_max_occupied": cp["kmax"], "solver_it": info["it"], "spectrum": info["spectrum"],
                "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"), "levels": cp["levels"],
                "N": N.tolist(), "v": v.tolist(), "eps_tot": cp["eps"].tolist(), "J_tot": cp["J"].tolist(), "hist_last": hist[-1] if hist else None})
    return out


def closed_count(levels, NB):
    """the nearest closed count to N_B from a member's occupied levels: the fractional multiplet (channel kappa, fraction f, weight 16 kappa)
    emptied (N_B - f 16 kappa) or completed (N_B + (1 - f) 16 kappa), whichever is nearer; None when no multiplet is fractional."""
    for k, lv in levels.items():
        for e, f in lv:
            if 1e-12 < f < 1 - 1e-12:
                w = 16.0 * float(k)
                down, up = NB - f * w, NB + (1 - f) * w
                return down if (NB - down) <= (up - NB) else up
    return None


def summary(members, m, g, NB_max_cont):
    ok = [mm for mm in members if not mm.get("failed")]
    out = {"n": len(members), "n_ok": len(ok), "NB_max_continuum": NB_max_cont}
    if ok:
        i = int(np.argmax([mm["E"] for mm in ok]))
        out["max_member"] = {k: ok[i][k] for k in ("NB_target", "E", "N_B", "R_surf", "R_999", "compactness_surf", "N_centre", "min_N_minus_absv", "E_F", "invariant_M_g15_ma2")}
        out["max_compactness_surf"] = max((mm["compactness_surf"] or 0) for mm in ok)
        out["min_over_family_of_min_N_minus_absv"] = min(mm["min_N_minus_absv"] for mm in ok)
        rej = [mm for mm in members if mm.get("failed") and mm["NB_target"] > ok[i]["NB_target"]]
        if rej:
            out["max_mass_bracket_NB"] = [ok[i]["NB_target"], min(mm["NB_target"] for mm in rej)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--ell", type=int, default=8)
    ap.add_argument("--nsites", type=int, default=1536)
    ap.add_argument("--eos", required=True)
    ap.add_argument("--ovpg", required=True)
    ap.add_argument("--fractions", default="0.1,0.2,0.35,0.5,0.7,0.85,0.95,0.98,0.99")
    ap.add_argument("--NB-list", default=None, help="explicit N_B list in place of the fractions")
    ap.add_argument("--bisect-tol", type=float, default=0.002)
    ap.add_argument("--ell-control", default="4,8,16")
    ap.add_argument("--closed-count", action="store_true", help="each converged member with a fractional multiplet also at its nearest closed count")
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    OPS.GPU["dev"] = a.gpu
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "star_%s.log" % a.tag), "a")
    path = os.path.join(a.out, "star_%s.json" % a.tag)
    fam = [F for F in json.load(open(a.ovpg))["families"] if abs(F["m"] - a.m) < 1e-12 and abs(F["g"] - a.g) < 1e-12][0]
    import engine_ovpg_v1 as OV
    e_of, P_of, n_of = OV.load_eos(a.eos, a.m)
    z_max = fam["maximum"]["z_c"]
    NB_max = OV.member(a.m, a.g, e_of, P_of, n_of, z_max)["N_B"]
    NBs = [float(x) for x in a.NB_list.split(",")] if a.NB_list else [round(float(f) * NB_max, 3) for f in a.fractions.split(",")]
    R0 = np.arange(a.nsites, dtype=float) + (0.5 if a.half_offset else 1.0)
    members = []
    N = v = None
    t_all = time.time()

    def run_member(NB, N, v, ell):
        t0 = time.time()
        if N is None:
            seed, cinfo = continuum_member_at_NB(a.eos, a.m, a.g, NB, z_hi=z_max)
            if seed is None:
                return {"NB_target": NB, "failed": True, "flags": ["no continuum seed: %s" % cinfo]}
            Ns, vs, _ = OV.profile_on_lattice(seed[0], a.g, R0)
        else:
            Ns, vs = N, v
            cinfo = None
        mem = star_member(NB, a.m, a.g, a.nsites, ell, np.array(Ns), np.array(vs), bool(a.half_offset), log=log)
        mem["seconds"] = time.time() - t0
        mem["continuum_seed"] = cinfo
        return mem

    def record():
        with open(path, "w") as fh:
            json.dump({"script": "engine_star_v5.py", "args": vars(a), "dictionary": FLD.DICTIONARY, "summary": summary(members, a.m, a.g, NB_max),
                       "members": members}, fh, default=float)

    def report(mem):
        if mem.get("failed") and "E" not in mem:
            print("N_B %.2f: REJECTED %s (%.0f s)" % (mem["NB_target"], "; ".join(mem["flags"]), mem.get("seconds", 0)), flush=True)
            return
        print("N_B %.2f E %.6f E_F/m %s R_surf %s R_999 %s C_surf %s N_min %.5f min(N-|v|) %.5f passes %d %s (%.0f s) ell = %d" % (
            mem["NB_target"], mem["E"], ("%.5f" % (mem["E_F"] / a.m)) if mem.get("E_F") else "-", mem["R_surf"], mem["R_999"],
            ("%.4f" % mem["compactness_surf"]) if mem["compactness_surf"] else "-", mem["N_min"], mem["min_N_minus_absv"], mem["passes"],
            ("REJECTED " + "; ".join(mem["flags"])) if mem["failed"] else "ok", mem["seconds"], mem["ell"]), flush=True)

    last_ok = None
    first_bad = None
    for NB in NBs:
        mem = run_member(NB, N, v, a.ell)
        members.append(mem); record(); report(mem)
        if not mem.get("failed"):
            N, v = np.array(mem["N"]), np.array(mem["v"]); last_ok = mem
            if a.closed_count and mem.get("fractional_multiplet"):
                NBc = closed_count(mem["levels"], NB)
                if NBc:
                    mc_ = run_member(NBc, N, v, a.ell)
                    mc_["closed_count_of"] = NB
                    members.append(mc_); record(); report(mc_)
        else:
            if last_ok is None:                         # version 5: a rejection before any converged member does not end the scan
                continue
            first_bad = mem
            break
    # bisection between the last converged and the first rejected member
    while last_ok is not None and first_bad is not None and (first_bad["NB_target"] - last_ok["NB_target"]) > a.bisect_tol * last_ok["NB_target"]:
        NB = 0.5 * (first_bad["NB_target"] + last_ok["NB_target"])
        mem = run_member(NB, np.array(last_ok["N"]), np.array(last_ok["v"]), a.ell)
        members.append(mem); record(); report(mem)
        if not mem.get("failed"):
            last_ok = mem
        else:
            first_bad = mem
    # the ell-control at the maximum member and at the member nearest E_F/m = 0.95
    ok = [mm for mm in members if not mm.get("failed")]
    ctrl = []
    if ok and a.ell_control:
        picks = [max(ok, key=lambda mm: mm["E"])]
        cand = [mm for mm in ok if mm.get("E_F")]
        if cand:
            dil = min(cand, key=lambda mm: abs(mm["E_F"] / a.m - 0.95))
            if dil is not picks[0]:
                picks.append(dil)
        for p in picks:
            rows = {}
            for ell in [int(x) for x in a.ell_control.split(",")]:
                mm = star_member(p["NB_target"], a.m, a.g, a.nsites, ell, np.array(p["N"]), np.array(p["v"]), bool(a.half_offset), log=log)
                rows[str(ell)] = {k: mm.get(k) for k in ("E", "R_surf", "compactness_surf", "N_centre", "N_min", "min_N_minus_absv", "E_F", "failed", "flags", "passes")}
                print("ell-control N_B %.2f ell %d: E %s R_surf %s N_min %s %s" % (p["NB_target"], ell, mm.get("E"), mm.get("R_surf"), mm.get("N_min"),
                      "REJECTED" if mm.get("failed") else "ok"), flush=True)
            Es = {k: r["E"] for k, r in rows.items() if r.get("E") is not None}
            ctrl.append({"NB_target": p["NB_target"], "rows": rows, "spread_E": float(np.ptp(list(Es.values()))) if Es else None,
                         "fit_E_a_plus_b_ell2": (np.polyfit(np.array([int(k) for k in Es], float) ** 2, list(Es.values()), 1).tolist() if len(Es) >= 2 else None)})
    with open(path, "w") as fh:
        json.dump({"script": "engine_star_v5.py", "args": vars(a), "dictionary": FLD.DICTIONARY, "summary": summary(members, a.m, a.g, NB_max),
                   "ell_control": ctrl, "members": members, "seconds": time.time() - t_all}, fh, default=float)


if __name__ == "__main__":
    main()
