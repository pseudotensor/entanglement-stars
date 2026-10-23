#!/usr/bin/env python
"""engine_collapse_v2.py (builder analyst; version 2 of the lead's engine_collapse_v1 per SPEC_final4b_engine.md sec. 11.10 item (5) and
sec. 11.2's settings, with the builder's mechanical corrections; the parts that wait for the lead's rulings on the reviewer B engine red team
(turn 19) are switches marked PENDING and run in their version-1 form or not at all, labelled in every record).

The model's dust under H_m[N(t), v(t)] with (N, v) re-solved from the hydrodynamic constraint solver (engine_fields_v2: the sources
averaged over ell, the coarse grid of spacing ell, the smooth interpolation) at every re-solve: quasi-static (the fields frozen over the
coupling interval K dt; every record says "quasi-static (K dt = ...)").
Changes against version 1: (1) engine_fields_v2 with --ell (default 4), the charge from the solver's collar, the records' ell; (2) the
initial data at zero temperature: the lowest N_B levels of h_m[N, v] + U (the well) with the last degenerate set filled fractionally
(version 1's T_s = 0.01 m smearing is the defect the builder's diagnosis and the red team found in the static star), projected on the
positive band of h_m[N, v], Loewdin-orthonormalized, the fields self-consistent by Anderson mixing (engine_mixing_v1); (3) the sources from
the occupied columns only; (4) the trapping surface from the smooth fields (the zero of N - |v| of the parity splines on a fine grid, the
outermost), with R_t against both 2 g E (the collar charge) and 2 g M(R_t), M(r) = r v^2/(2 g N^2) (the red team's M2: only the latter is
an identity of the constraints at the surface); (5) M10: the spectral fractions of every re-solve; (6) the coldness as
sum f (<h_D> - m)/(N_B m) (version 1 divided by N_B) and, beside it, the same in the bulk frame for reading A (the bulk momentum's share
removed: BUILDER-pending, printed as None until the lead rules); (7) the collar current is labelled the number current (the red team's M1:
it is not the energy current; the energy flux: PENDING); (8) the fields npz stores the node values, ell and the label; (9) the sea's
channel count by --kappa-sea with the last channel's created energy printed per re-solve (the red team: a cutoff ladder is needed;
PENDING). PENDING (turn 19): the evolution control (the energy-flux law of the metric mass with a lattice stress, or the lead's M9), the
chart map M4 (version 1's exterior offset diverges), the capture frequencies (above m), the r_s label (2 g N_B m is a rest-charge label).
Usage: python engine_collapse_v2.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --dt 1 --K 4 --T-over-rs 40
--out <dir> --tag <tag> [--gpu 0]"""
import argparse
import json
import os
import time

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v2 as FLD
import engine_mixing_v1 as MIX


class Channel:
    def __init__(self, kappa, m, nsites, half_offset=False):
        self.kappa = kappa
        self.w = 16.0 * kappa
        self.h, self.Q, self.R = OPS.chain_parts(nsites, m, float(kappa), half_offset)


def well(R, R0, U0, wU):
    return U0 / (1.0 + np.exp(-(R - R0) / wU))


def fill_lowest(levels, NB, tol=1e-9):
    """levels: list of (energy, weight, channel index, mode index); the lowest NB states, the last degenerate set fractional."""
    lv = sorted(levels)
    occ = {}
    left = float(NB)
    i = 0
    while i < len(lv) and left > 1e-12:
        j = i
        while j < len(lv) and abs(lv[j][0] - lv[i][0]) < tol:
            j += 1
        w = sum(x[1] for x in lv[i:j])
        frac = min(1.0, left / w)
        for x in lv[i:j]:
            occ[(x[2], x[3])] = frac
        left -= frac * w
        i = j
    return occ


def prepare_dust(chs, N, v, g, R, ell, NB_target, R0, U0, wU, reading="A", passes=200, tol=1e-10):
    """the confined cold ball at zero temperature (spec 5.1 as amended): N, v, per-channel (Phi_dust, f_dust) and the diagnostics."""
    Ud = np.repeat(well(R, R0, U0, wU), 2)
    mixer = MIX.Anderson()
    nsites = len(R)
    for ip in range(passes):
        spec = {}
        levels = []
        for ic, ch in enumerate(chs):
            hw = OPS.clocked(ch.h, ch.Q, N, v if reading == "A" else 0 * v) + np.diag(Ud)
            ew, Uw = OPS.spectrum(hw)
            em, Um = OPS.spectrum(OPS.clocked(ch.h, ch.Q, N, v))
            pos = ew > 0
            spec[ch.kappa] = (ew[pos], Uw[:, pos], Um[:, em > 0])
            levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ew[pos][:400])]
        occ = fill_lowest(levels, NB_target)
        eps = np.zeros(nsites); J = np.zeros(nsites); dust = {}; kin = 0.0; NB = 0.0
        for ic, ch in enumerate(chs):
            ew, Uw, Pplus = spec[ch.kappa]
            keep = [j for j in range(min(400, len(ew))) if occ.get((ic, j), 0.0) > 0]
            if not keep:
                continue
            fk = np.array([occ[(ic, j)] for j in keep])
            Phi = Pplus @ (Pplus.conj().T @ Uw[:, keep])
            Phi = OPS.lowdin(Phi)
            e1, J1 = OPS.sources_static(ch.h, ch.Q, Phi, fk, ch.w)
            eps += e1; J += J1; NB += ch.w * fk.sum()
            hD = np.real(np.einsum("ik,ik->k", Phi.conj(), ch.h @ Phi))
            kin += ch.w * float(fk @ hD)
            dust[ch.kappa] = (Phi, fk)
        Nn, vn, Wn, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N)
        d = max(float(np.max(np.abs(Nn - N))), float(np.max(np.abs(vn - v))))
        x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
        N, v = x[:nsites], x[nsites:]
        if d < tol:
            break
    N, v, Wn, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N)
    m = abs(chs[0].h[0, 0].real)
    return N, v, dust, info, {"N_B": NB, "passes": ip + 1, "last_change": d, "E0": FLD.charge(info, g), "kin_fraction": (kin - NB * m) / max(NB * m, 1e-300),
                              "kin_fraction_bulk_frame": None, "J_tot_sum": float(J.sum()), "eps_tot": eps.tolist(), "J_tot": J.tolist()}


def dust_profile(chs, state):
    R = chs[0].R
    rho = np.zeros(len(R))
    for ch in chs:
        Phi, f, nsea = state[ch.kappa]
        if Phi.shape[1] > nsea:
            rho += ch.w * OPS.site_density(np.eye(Phi.shape[0]), Phi[:, nsea:], f[nsea:])
    tot = rho.sum()
    cum = np.cumsum(rho)
    Rsh = float(R[np.searchsorted(cum, 0.99 * tot)]) if tot > 0 else None
    return rho, (float(rho @ R / tot) if tot > 0 else None), Rsh


def smooth_trapping(info, g):
    """the outermost zero of N - |v| on the smooth fields (the parity splines of the node values on a grid of ell/64)."""
    r = info["r_nodes"]
    Ns = FLD.parity_spline(r, info["N_nodes"], +1); vs = FLD.parity_spline(r, info["v_nodes"], -1)
    rr = np.linspace(r[1] * 0.5, r[-1], int(64 * (r[-1] / info["ell"])) + 1)
    f = Ns(rr) - np.abs(vs(rr))
    s = np.nonzero(f[:-1] * f[1:] < 0)[0]
    if not len(s):
        return None, None
    i = s[-1]
    Rt = float(rr[i] - f[i] * (rr[i + 1] - rr[i]) / (f[i + 1] - f[i]))
    Nt = float(Ns(Rt)); vt = float(vs(Rt))
    return Rt, Rt * vt * vt / (2 * g * Nt * Nt)                     # R_t and M(R_t) = r v^2/(2 g N^2)


def run(a):
    os.makedirs(a.out, exist_ok=True)
    label = "quasi-static (K dt = %g: K = %d, dt = %g; lagged reference); ell = %d" % (a.K * a.dt, a.K, a.dt, a.ell)
    rs = a.rs
    NB_target = rs / (2.0 * a.g * a.m)
    chs = [Channel(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    N = np.ones(a.nsites); v = np.zeros(a.nsites)
    N, v, dust, info, head = prepare_dust(chs, N, v, a.g, R, a.ell, NB_target, a.R0_over_rs * rs, 0.5 * a.m, 4.0, a.reading)
    head.update({"label": label, "reading": a.reading, "rs_rest_label": rs, "R0": a.R0_over_rs * rs, "ell": a.ell,
                 "pending": ["the evolution control (turn 19)", "M4 chart map", "the energy flux at the collar", "the bulk-frame coldness", "the sea-channel ladder"]})
    E0 = head["E0"]
    print("initial data: N_B %.2f E0 %.6f passes %d kin fraction %.4f (%s)" % (head["N_B"], E0, head["passes"], head["kin_fraction"], label), flush=True)
    state, cache = {}, {}
    for ch in chs:
        e, U = OPS.spectrum(OPS.clocked(ch.h, ch.Q, N, v))
        sea = U[:, e < 0]
        Phi_d, f_d = dust.get(ch.kappa, (np.zeros((2 * a.nsites, 0), complex), np.zeros(0)))
        state[ch.kappa] = (np.hstack([sea, Phi_d]), np.concatenate([np.ones(sea.shape[1]), f_d]), sea.shape[1])
        cache[ch.kappa] = (e, U)
    nsteps = int(round(a.T_over_rs * rs / a.dt))
    recs, fields = [], {"T": [], "N": [], "v": [], "r_nodes": [], "N_nodes": [], "v_nodes": []}
    T = 0.0
    t0 = time.time()
    for step in range(0, nsteps + 1, a.K):
        eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_sea = np.zeros(a.nsites); last_sea = 0.0
        for ch in chs:
            Phi, f, nsea = state[ch.kappa]
            e, U = cache[ch.kappa]
            Qneg = U[:, e < 0]
            e1, J1 = OPS.sources_dynamic(ch.h, ch.Q, Phi, f, Qneg, ch.w)
            eps += e1; J += J1
            es = ch.w * (OPS.site_density(ch.h, Phi[:, :nsea]) - OPS.site_density(ch.h, Qneg))
            eps_sea += es
            if ch.kappa == a.kappa_sea:
                last_sea = float(es.sum())
        N, v, Wn, info = FLD.solve_constraints(eps, J, a.g, R, a.ell, N0=N)
        for ch in chs:
            cache[ch.kappa] = OPS.spectrum(OPS.clocked(ch.h, ch.Q, N, v))
        E = FLD.charge(info, a.g)
        rho_d, Rc, Rsh = dust_profile(chs, state)
        Rt, Mt = smooth_trapping(info, a.g)
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "R_trap": Rt, "R_trap_over_2gE": (Rt / (2 * a.g * E) if Rt else None),
               "M_at_Rt": Mt, "R_trap_over_2gM_Rt": (Rt / (2 * a.g * Mt) if (Rt and Mt) else None),
               "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(np.interp(Rc, R, v)) if Rc else None), "N_at_Rsh": (float(np.interp(Rsh, R, N)) if Rsh else None),
               "absv_at_Rsh": (float(np.abs(np.interp(Rsh, R, v))) if Rsh else None), "N_min": float(N.min()),
               "sea_energy_inside_Rt": (float(eps_sea[R < Rt].sum()) if Rt else None), "sea_energy_outside_Rt": (float(eps_sea[R >= Rt].sum()) if Rt else 0.0),
               "sea_last_channel_energy": last_sea,
               "collar_number_current": sum(OPS.cut_current(OPS.clocked(ch.h, ch.Q, N, v), state[ch.kappa][0], state[ch.kappa][1],
                                                             int(min(max(round(info["r_collar"]) - 1, 1), a.nsites - 2))) * ch.w for ch in chs),
               "spectrum": info["spectrum"], "solver_it": info["it"], "r_collar": info["r_collar"], "label": label, "seconds": time.time() - t0}
        if recs and Rc and recs[-1]["R_c"]:
            rec["dRc_dT"] = (Rc - recs[-1]["R_c"]) / (T - recs[-1]["T"])
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy())
        fields["r_nodes"].append(info["r_nodes"]); fields["N_nodes"].append(info["N_nodes"]); fields["v_nodes"].append(info["v_nodes"])
        print("T/rs %.2f E/E0 %.6f Rt %s (2gE %.3f, 2gM(Rt) %s) Rsh %s Rc %s N_min %.4f (%.0f s)" % (
            T / rs, E / E0, ("%.3f" % Rt) if Rt else None, 2 * a.g * E, ("%.3f" % (2 * a.g * Mt)) if Mt else None, Rsh, ("%.2f" % Rc) if Rc else None, N.min(), time.time() - t0), flush=True)
        if step >= nsteps:
            break
        for ch in chs:
            e, U = cache[ch.kappa]
            Phi, f, nsea = state[ch.kappa]
            Phi = U @ (np.exp(-1j * e * a.K * a.dt)[:, None] * (U.conj().T @ Phi))
            state[ch.kappa] = (Phi, f, nsea)
        T += a.K * a.dt
    np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),
                        r_nodes=np.array(fields["r_nodes"], dtype=object), N_nodes=np.array(fields["N_nodes"], dtype=object), v_nodes=np.array(fields["v_nodes"], dtype=object),
                        ell=a.ell, label=label)
    with open(os.path.join(a.out, "collapse_%s.json" % a.tag), "w") as fh:
        json.dump({"script": "engine_collapse_v2.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    OPS.GPU["dev"] = a.gpu
    run(a)


if __name__ == "__main__":
    main()
