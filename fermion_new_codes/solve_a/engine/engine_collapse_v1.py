#!/usr/bin/env python
"""engine_collapse_v1.py (lead's draft; SPEC_final4b_engine.md sec. 5): the model's dust under H_m[N(t), v(t)] with (N, v) re-solved from
the constraints at every re-solve (quasi-static, exact in spherical symmetry up to the matter's time discretization; the Bianchi control).
Initial data: the cold ball of the well (reading A: at rest in the static frame; reading B: zero lattice momentum), projected on the
positive band of h_m[N, v], the fields self-consistent.  Step: per channel Phi <- U exp(-i e K dt) U^dag Phi with (e, U) at the current
fields; sources by form (ii) with the lagged reference; the constraints re-solved.  Every record carries "quasi-static (K = ...)".
Usage: python engine_collapse_v1.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 1024 --dt 1 --K 4 --T-over-rs 40 --out <dir> --tag <tag>"""
import argparse
import json
import os
import time

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v1 as FLD


class Channel:
    def __init__(self, kappa, m, nsites, half_offset=False):
        self.kappa = kappa
        self.w = 16.0 * kappa
        self.h, self.Q, self.R = OPS.chain_parts(nsites, m, float(kappa), half_offset)


def well(R, R0, U0, wU):
    return U0 / (1.0 + np.exp(-(R - R0) / wU))


def prepare_dust(chs, N, v, g, NB_target, Ts, R0, U0, wU, reading="A", passes=80, alpha=0.3, tol=1e-10):
    """the confined cold ball (spec 5.1); returns N, v, per-channel (Phi_dust, f_dust), E_F and the diagnostics."""
    R = chs[0].R
    Ud = np.repeat(well(R, R0, U0, wU), 2)
    for ip in range(passes):
        spec = {}
        for ch in chs:
            hw = OPS.clocked(ch.h, ch.Q, N, v if reading == "A" else 0 * v) + np.diag(Ud)
            ew, Uw = OPS.spectrum(hw)
            em, Um = OPS.spectrum(OPS.clocked(ch.h, ch.Q, N, v))
            spec[ch.kappa] = (ew, Uw, Um[:, em > 0])
        # the Fermi level by bisection on the weighted count
        lo, hi = 0.0, 2.0 * max(1.0, np.max([spec[k][0].max() for k in spec]))
        count = lambda EF: sum(ch.w * OPS.fermi(spec[ch.kappa][0][spec[ch.kappa][0] > 0], EF, Ts).sum() for ch in chs)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if count(mid) < NB_target else (lo, mid)
        EF = 0.5 * (lo + hi)
        eps = np.zeros(len(R)); J = np.zeros(len(R)); dust = {}; kin = 0.0; NB = 0.0
        for ch in chs:
            ew, Uw, Pplus = spec[ch.kappa]
            pos = ew > 0
            f = OPS.fermi(ew[pos], EF, Ts)
            keep = f > 1e-10
            if not keep.any():
                continue
            Phi = Pplus @ (Pplus.conj().T @ Uw[:, pos][:, keep])
            Phi = OPS.lowdin(Phi)
            fk = f[keep]
            e1, J1 = OPS.sources_static(ch.h, ch.Q, Phi, fk, ch.w)
            eps += e1; J += J1; NB += ch.w * fk.sum()
            kin += ch.w * float(fk @ np.real(np.einsum("ik,ik->k", Phi.conj(), ch.h @ Phi))) - ch.w * fk.sum() * chs[0].h[0, 0].real  # BUILDER: the kinetic fraction properly: sum f (<h> - m) / (NB m)
            dust[ch.kappa] = (Phi, fk)
        Nn, vn, Wf, info = FLD.solve_constraints(eps, J, g, R, N0=N)
        d = max(float(np.max(np.abs(Nn - N))), float(np.max(np.abs(vn - v))))
        N = N + alpha * (Nn - N); v = v + alpha * (vn - v)
        if d < tol:
            break
    N, v, Wf, info = FLD.solve_constraints(eps, J, g, R, N0=N)
    return N, v, dust, {"EF": EF, "N_B": NB, "passes": ip + 1, "E0": FLD.charge(Wf, g, len(R) - 1), "kin_fraction_BUILDER": kin / max(NB * 1.0, 1e-300),
                        "J_tot_sum": float(J.sum()), "eps_tot": eps.tolist(), "J_tot": J.tolist()}


def dust_profile(chs, state):
    """the dust's density per shell and its centroid (the dust block tracked separately from the sea)."""
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


def run(a):
    os.makedirs(a.out, exist_ok=True)
    label = "quasi-static (K = %d, dt = %g, lagged reference)" % (a.K, a.dt)
    rs = a.rs
    NB_target = rs / (2.0 * a.g * a.m)
    chs = [Channel(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    N = np.ones(a.nsites); v = np.zeros(a.nsites)
    Ts = 0.01 * a.m
    N, v, dust, head = prepare_dust(chs, N, v, a.g, NB_target, Ts, a.R0_over_rs * rs, 0.5 * a.m, 4.0, a.reading)
    head.update({"label": label, "reading": a.reading, "rs_rest": rs, "R0": a.R0_over_rs * rs})
    E0 = head["E0"]
    # the orbital blocks: sea (reference at t = 0) + dust
    state, cache = {}, {}
    for ch in chs:
        e, U = OPS.spectrum(OPS.clocked(ch.h, ch.Q, N, v))
        sea = U[:, e < 0]
        Phi_d, f_d = dust.get(ch.kappa, (np.zeros((2 * a.nsites, 0), complex), np.zeros(0)))
        state[ch.kappa] = (np.hstack([sea, Phi_d]), np.concatenate([np.ones(sea.shape[1]), f_d]), sea.shape[1])
        cache[ch.kappa] = (e, U)
    nsteps = int(round(a.T_over_rs * rs / a.dt))
    collar = a.nsites - 2
    recs, fields = [], {"T": [], "N": [], "v": []}
    T = 0.0
    t0 = time.time()
    for step in range(0, nsteps + 1, a.K):
        # sources with the lagged reference (the eigendecomposition the propagation used), then the re-solve
        eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_sea = np.zeros(a.nsites)
        for ch in chs:
            Phi, f, nsea = state[ch.kappa]
            e, U = cache[ch.kappa]
            Qneg = U[:, e < 0]
            e1, J1 = OPS.sources_dynamic(ch.h, ch.Q, Phi, f, Qneg, ch.w)
            eps += e1; J += J1
            eps_sea += ch.w * (OPS.site_density(ch.h, Phi[:, :nsea]) - OPS.site_density(ch.h, Qneg))
        N, v, Wf, info = FLD.solve_constraints(eps, J, a.g, R, N0=N)
        for ch in chs:
            cache[ch.kappa] = OPS.spectrum(OPS.clocked(ch.h, ch.Q, N, v))
        E = FLD.charge(Wf, a.g, collar)
        rho_d, Rc, Rsh = dust_profile(chs, state)
        traps = FLD.trapping_radius(N, v, R)
        Rt = traps[-1] if traps else None
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "R_trap": Rt, "R_trap_over_2gE": (Rt / (2 * a.g * E) if Rt else None),
               "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(np.interp(Rc, R, v)) if Rc else None), "N_at_Rsh": (float(np.interp(Rsh, R, N)) if Rsh else None),
               "absv_at_Rsh": (float(np.abs(np.interp(Rsh, R, v))) if Rsh else None), "N_min": float(N.min()),
               "sea_energy_inside_Rt": (float(eps_sea[R < Rt].sum()) if Rt else None), "sea_energy_outside_Rt": (float(eps_sea[R >= Rt].sum()) if Rt else 0.0),
               "collar_current": sum(OPS.cut_current(OPS.clocked(ch.h, ch.Q, N, v), state[ch.kappa][0], state[ch.kappa][1], collar) * ch.w for ch in chs),
               "solver": info, "seconds": time.time() - t0}
        if recs:
            rec["dRc_dT"] = (Rc - recs[-1]["R_c"]) / (T - recs[-1]["T"]) if (Rc and recs[-1]["R_c"]) else None
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy())
        # BUILDER: C5.2 (the Bianchi control) every 10th re-solve: the face increments of W over the interval against
        # 2 g N^2 (d eps/dt) with d eps_n/dt = Tr[(C - C_0) i[h_m, hcal_n]] - Tr[dC_0/dt hcal_n]; the same for 1/N and J.
        print("T/rs %.2f E/E0 %.6f Rt %s Rsh %s Rc %s" % (T / rs, E / E0, Rt, Rsh, Rc), flush=True)
        if step >= nsteps:
            break
        # propagate K steps with the frozen fields
        for ch in chs:
            e, U = cache[ch.kappa]
            Phi, f, nsea = state[ch.kappa]
            Phi = U @ (np.exp(-1j * e * a.K * a.dt)[:, None] * (U.conj().T @ Phi))
            state[ch.kappa] = (Phi, f, nsea)
        T += a.K * a.dt
    np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]))
    with open(os.path.join(a.out, "collapse_%s.json" % a.tag), "w") as fh:
        json.dump({"script": "engine_collapse_v1.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)


def chart_map_redshift(npz, Rsh_of_T, g, collar):
    """M4 (post-processing): outgoing rays dr/dt = N + v from (t_e, R_sh(t_e)) to the collar on the stored fields; t_s = t_arr + the
    exterior PG map; 1 + z = dt_arr/dtau_e, dtau_e = sqrt(N^2 - (u - v)^2) dt_e; the fits of ln(1+z) and ln(R_sh - 2gE) against t_s.
    BUILDER: implement with scipy.integrate.solve_ivp on the bilinear interpolant of (N, v) in (r, T); the exterior map
    t_s = t + int_collar^inf v/(1 - v^2) dr with v = -sqrt(2 g E/r) (a constant offset per E); report kappa_fit against 1/(4 g E)."""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=32.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--kappa-sea", type=int, default=8, help="BUILDER: fix by the created energy of the last channel (spec 5.2)")
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
