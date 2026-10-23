#!/usr/bin/env python
"""packet_v1_propagate.py (busi lane, reviewer A slot, inbox #354: the quantum packet completeness test on the computed collapse).

Localized one-particle fermion packets propagated with the engine's own clocked chain Hamiltonian h_m(T) = (1/2){N, h0_kappa} + (1/2){v, Q}
(packet_v1_common.chain / clocked, the engine's operator rebuilt and checked) on the stored applied (back-averaged) fields N_applied, v_applied
of the reference collapse (sigma/a 4, tau 8, r_s/a 64: collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz), from the formation record on.

Two propagation rules, both run on every packet:
  E (the engine's rule): the state at record k is carried to record k+1 by exp(-i tau h_m[N_applied_k, v_applied_k]), the frozen accepted
    Hamiltonian of the record held over the whole step (engine_collapse_v19.run, the propagation block at the end of the record loop:
    ph = exp(-i e tau), U ph U^dag). Here the same exponential is applied by scipy's expm_multiply (Al-Mohy and Higham), exact to rounding.
  I (interpolated): the fields linear in T between records k and k+1 (the engine's continuation interpolates h(s) = h_prev + s (h_cur - h_prev)),
    nsub sub-steps per record, each the exponential of h_m at the sub-step's midpoint fields.
Packets: a Gaussian envelope of width sigma_p about r0 times a plane wave exp(i k0 r) times the positive-energy spinor of m s3 + s1 sin k0,
projected on the positive band of the free chain h0_kappa (the slice-frame particle), normalized; k0 > 0 moves outward (p_r = sin k).
Measured at every record (with h_m of that record): the norm; <h_m> (the energy per lattice tick) and its spread against ||h_m||;
<h0_kappa> (the slice-frame energy, the energy per tick of the local clock) and its spread against the radial band top b = sqrt(m^2 + 1) and
||h0_kappa||; <Q>; the momentum distribution's shares at |k| > pi/4 and |k| > 1.2 and <|k|>; <r>, the spread; the shares inside the inner
trapping root, in the trapped shell, outside the outer root and inside r < 12a; <N>, <v>, <N - |v|> at the packet; the packet's clock integral
sum_k tau <N>/<h0> (a diagnostic only: the lattice theory has no affine parameter for a spreading state).
Beside each ray-like packet (|k0| = k_ray) the classical radial null ray on the same barred fields: dr/dT = v + s N, d ln eps/dT = -(v' + s N'),
d lambda = N dT/eps (CURVATURE_readout_v1 sec. 5), the fields linear in r (parity-extended through the centre: N even, v odd) and linear in T,
RK4 with dT = tau/8.

Usage: python packet_v1_propagate.py --fields F_fields.npz --json F.json --out DIR [--engine-code DIR] [--gpu N | --cpu] [--channels 1 5]
       [--rec0 97] [--rec1 320] [--nsub 16] [--sigma 4] [--k-ray 0.45] [--k-top 1.2] [--quick]
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
from scipy.sparse.linalg import eigsh, expm_multiply  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packet_v1_common as PC  # noqa: E402


def positive_band_projector(h0, device):
    """the eigenvectors of h0_kappa with positive energy (torch eigh; the one dense operation), as a function phi -> P_+ phi (block capable)."""
    import torch
    H = torch.as_tensor(h0.toarray(), dtype=torch.complex128, device=device)
    e, U = torch.linalg.eigh(H)
    e = e.real
    Up = U[:, e > 0].contiguous()
    del H, U
    info = {"n_positive": int((e > 0).sum()), "n_negative": int((e < 0).sum()), "n_zero": int((e == 0).sum()),
            "band_top_free": float(e.abs().max()), "lowest_positive": float(e[e > 0].min()), "device": str(device)}

    def proj(phi):
        X = torch.as_tensor(np.ascontiguousarray(phi), dtype=torch.complex128, device=device)
        Y = Up @ (Up.conj().T @ X)
        return Y.cpu().numpy()
    return proj, info


def make_packet(R, m, r0, k0, sigma):
    g = np.exp(-(R - r0) ** 2 / (4.0 * sigma ** 2))
    s = np.sin(k0); E = np.sqrt(m * m + s * s)
    u = np.array([E + m, s], complex); u /= np.linalg.norm(u)
    phi = (g * np.exp(1j * k0 * R))[:, None] * u[None, :]
    return phi.reshape(-1)


def momentum_shares(psi, n, k_lat=1.2):
    """momentum distribution by the DFT of both components: shares at |k| > pi/4 and |k| > k_lat, and <|k|>, per column."""
    X = psi.reshape(n, 2, -1)
    F = np.fft.fft(X, axis=0)
    P = (np.abs(F) ** 2).sum(axis=1)                       # (n, ncol)
    P /= np.maximum(P.sum(axis=0, keepdims=True), 1e-300)
    k = 2.0 * np.pi * np.fft.fftfreq(n)
    ak = np.abs(k)[:, None]
    return (P * (ak > np.pi / 4)).sum(axis=0), (P * (ak > k_lat)).sum(axis=0), (P * ak).sum(axis=0)


class Fields:
    """the barred site fields of the records, linear in T between records and linear in r, parity-extended through the centre."""

    def __init__(self, R, T, N, v):
        self.R = R; self.T = T; self.N = N; self.v = v; self.n = len(R); self.tau = float(T[1] - T[0])
        # parity extension on r = -n_ch .. n_ch with r = 0 included: N even (N(0) by the average of the two neighbours), v odd (v(0) = 0)
        self.rx = np.concatenate([-R[::-1], [0.0], R])

    def at(self, Tq):
        k = int(min(max(np.floor(Tq / self.tau), 0), len(self.T) - 2))
        th = (Tq - self.T[k]) / self.tau
        Nk = (1 - th) * self.N[k] + th * self.N[k + 1]; vk = (1 - th) * self.v[k] + th * self.v[k + 1]
        Nx = np.concatenate([Nk[::-1], [Nk[0]], Nk]); vx = np.concatenate([-vk[::-1], [0.0], vk])
        return Nx, vx

    def rhs(self, Tq, r, s):
        Nx, vx = self.at(Tq)
        rr = float(np.clip(r, self.rx[0] + 1.0, self.rx[-1] - 1.0))
        Nq = np.interp(rr, self.rx, Nx); vq = np.interp(rr, self.rx, vx)
        dN = np.interp(rr, self.rx, np.gradient(Nx, self.rx)); dv = np.interp(rr, self.rx, np.gradient(vx, self.rx))
        return vq + s * Nq, -(dv + s * dN), Nq


def trace_ray(F, r0, s, T0, T1, dt):
    """RK4 for y = (r, ln eps, lambda) from T0 to T1; eps = 1 at launch; returns the samples at every record time and the end state."""
    y = np.array([r0, 0.0, 0.0]); T = T0
    recs_T = []; recs_y = []
    escaped = False

    def f(Tq, y_):
        drdT, dlneps, Nq = F.rhs(Tq, y_[0], s)
        return np.array([drdT, dlneps, Nq * np.exp(-y_[1])])
    nstep = int(round((T1 - T0) / dt))
    rec_every = int(round(F.tau / dt))
    recs_T.append(T); recs_y.append(y.copy())
    for i in range(1, nstep + 1):
        if abs(y[0]) >= F.R[-1] - 2.0:
            escaped = True
            y_end = y.copy()
            # hold the escaped ray's values on the remaining records
            while len(recs_T) < nstep // rec_every + 1:
                recs_T.append(recs_T[-1] + F.tau); recs_y.append(y_end.copy())
            break
        k1 = f(T, y); k2 = f(T + dt / 2, y + dt / 2 * k1); k3 = f(T + dt / 2, y + dt / 2 * k2); k4 = f(T + dt, y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4); T += dt
        if i % rec_every == 0:
            recs_T.append(T); recs_y.append(y.copy())
    Y = np.array(recs_y)
    return {"T": np.array(recs_T), "r": np.abs(Y[:, 0]), "r_signed": Y[:, 0], "ln_eps": Y[:, 1], "lambda": Y[:, 2], "escaped": escaped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--engine-code", default=None); ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 5]); ap.add_argument("--rec0", type=int, default=97); ap.add_argument("--rec1", type=int, default=320)
    ap.add_argument("--nsub", type=int, default=16); ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--k-ray", type=float, default=0.45); ap.add_argument("--k-top", type=float, default=1.2)
    ap.add_argument("--k-list", type=float, nargs="*", default=[0.1, 0.3])
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    device = torch.device("cpu") if (a.cpu or a.gpu is None) else torch.device("cuda:%d" % a.gpu)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
    tau = float(Z["tau"]); n = len(R)
    J = json.load(open(a.json))
    m = float(J["args"]["m"]); g = float(J["args"]["g"]); rs = float(J["args"]["rs"])
    recs = J["records"]
    zeros = [r_.get("trapping_zeros") or [] for r_ in recs]
    rec1 = min(a.rec1, len(T) - 1)
    if a.quick:
        rec1 = min(a.rec0 + 6, rec1)
    ks = list(range(a.rec0, rec1 + 1))
    meta = {"fields": a.fields, "fields_md5": PC.md5(a.fields), "json": a.json, "json_md5": PC.md5(a.json), "script_md5": PC.md5(os.path.abspath(__file__)),
            "common_md5": PC.md5(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packet_v1_common.py")),
            "m": m, "g": g, "rs": rs, "tau": tau, "nsites": n, "rec0": a.rec0, "rec1": rec1, "T_over_rs": [float(T[k] / rs) for k in (a.rec0, rec1)],
            "nsub": a.nsub, "sigma_p": a.sigma, "k_ray": a.k_ray, "k_top": a.k_top, "channels": a.channels, "device": str(device),
            "formation_zeros": zeros[a.rec0], "engine_check": PC.check_against_engine(a.engine_code, R, m, a.channels)}
    print("engine check:", json.dumps(meta["engine_check"]), flush=True)
    b_free = float(np.sqrt(m * m + 1.0))
    z0 = zeros[a.rec0]
    r_in0, r_out0 = (z0[0], z0[-1]) if len(z0) >= 2 else (23.24, 38.56)
    r_mid0 = 0.5 * (r_in0 + r_out0)
    kr, kt = a.k_ray, a.k_top
    kset = sorted(set([float(x) for x in a.k_list] + [kr, kt]))
    launches = []
    for lab, r0 in (("centre", 4.0), ("r_in", r_in0), ("mid", r_mid0), ("r_out", r_out0), ("outside", 60.0)):
        for k0 in (([0.0] + kset) if lab == "centre" else ([0.0] + kset + [-x for x in kset])):
            launches.append((lab, r0, k0))
    launches_ctrl = [("centre", 4.0, kr), ("r_in", r_in0, -kr), ("r_in", r_in0, kr), ("r_in", r_in0, -0.3), ("r_in", r_in0, 0.3),
                     ("r_out", r_out0, -kr), ("r_out", r_out0, kr), ("r_out", r_out0, -0.3), ("r_out", r_out0, 0.3)]

    def dispersion(k0):
        """the free chain's dispersion at k0 (N = 1, v = 0, large r): eps, the group velocity eps'(k), the flow coupling q'(k) = cos 2k of the
        two-step current, and the lattice trapping threshold |v|/N = eps'/q' above which a packet at k0 cannot move outward (None if q' <= 0)."""
        s, c = np.sin(k0), np.cos(k0); e = np.sqrt(m * m + s * s)
        vg = s * c / e; qp = np.cos(2.0 * k0)
        return {"eps": float(e), "group_velocity": float(vg), "flow_coupling": float(qp), "trap_threshold_v_over_N": (float(abs(vg) / qp) if qp > 1e-12 else None)}
    vN_max = np.array([float(np.max(np.abs(vapp[k]) / Napp[k])) for k in range(len(T))])
    vN_argmax = np.array([float(R[int(np.argmax(np.abs(vapp[k]) / Napp[k]))]) for k in range(len(T))])
    F = Fields(R, T, Napp, vapp)
    obs_names = ["norm", "Em", "dEm", "normHm", "E0", "dE0", "Qm", "rmean", "rspread", "Nm", "vm", "NmV", "share_in", "share_shell", "share_out", "share_core",
                 "mom_pi4", "mom_top", "kabs", "clock"]
    out = {"meta": meta, "channels": {}}
    arrays = {}
    for kappa in a.channels:
        tk = time.time()
        h0, Q = PC.chain(R, m, kappa)
        norm_h0 = float(abs(eigsh(h0, k=1, which="LM", return_eigenvectors=False)[0]))
        norm_Q = float(abs(eigsh(Q, k=1, which="LM", return_eigenvectors=False)[0]))
        proj, pinfo = positive_band_projector(h0, device)
        L = launches if kappa == a.channels[0] else launches_ctrl
        Phi = np.stack([make_packet(R, m, r0, k0, a.sigma) for (_, r0, k0) in L], axis=1)
        Psi0 = proj(Phi)
        proj_frac = (np.abs(Psi0) ** 2).sum(axis=0) / (np.abs(Phi) ** 2).sum(axis=0)
        Psi0 /= np.sqrt((np.abs(Psi0) ** 2).sum(axis=0, keepdims=True))
        npk = len(L)
        print("kappa %d: ||h0|| %.6f ||Q|| %.6f band top sqrt(m^2+1) %.6f; positive band %s; %d packets, projected fractions %s (%.0f s)" % (
            kappa, norm_h0, norm_Q, b_free, json.dumps(pinfo), npk, np.array2string(proj_frac, precision=3), time.time() - tk), flush=True)
        series = {sch: {nm: np.full((npk, len(ks)), np.nan) for nm in obs_names} for sch in ("E", "I")}
        normHm_rec = np.full(len(ks), np.nan)
        PsiE = Psi0.copy(); PsiI = Psi0.copy()
        clockE = np.zeros(npk); clockI = np.zeros(npk)
        tp = time.time()
        for ik, k in enumerate(ks):
            Hm = PC.clocked(h0, Q, Napp[k], vapp[k])
            normHm = float(abs(eigsh(Hm, k=1, which="LM", return_eigenvectors=False, tol=1e-10)[0]))
            normHm_rec[ik] = normHm
            zk = zeros[k]
            for sch, Psi, clock in (("E", PsiE, clockE), ("I", PsiI, clockI)):
                d = PC.site_density(Psi, n); nrm = d.sum(axis=0)
                Em, Em2 = PC.expectation(Hm, Psi); E0, E02 = PC.expectation(h0, Psi); Qm, _ = PC.expectation(Q, Psi)
                Em /= nrm; Em2 /= nrm; E0 /= nrm; E02 /= nrm; Qm /= nrm
                dn = d / nrm
                rmean = (R[:, None] * dn).sum(axis=0); rsp = np.sqrt(np.maximum((R[:, None] ** 2 * dn).sum(axis=0) - rmean ** 2, 0.0))
                Nm = (Napp[k][:, None] * dn).sum(axis=0); vm = (vapp[k][:, None] * dn).sum(axis=0); NmV = ((Napp[k] - np.abs(vapp[k]))[:, None] * dn).sum(axis=0)
                if len(zk) >= 2:
                    rin, rout = zk[0], zk[-1]
                    sin_ = dn[R < rin].sum(axis=0); ssh = dn[(R >= rin) & (R <= rout)].sum(axis=0); sout = dn[R > rout].sum(axis=0)
                else:
                    sin_ = np.full(npk, np.nan); ssh = np.full(npk, np.nan); sout = np.full(npk, np.nan)
                score = dn[R < 12.0].sum(axis=0)
                mp4, mtop, kabs = momentum_shares(Psi, n, a.k_top)
                clock += tau * Nm / np.maximum(E0, 1e-300)
                vals = [nrm, Em, np.sqrt(np.maximum(Em2 - Em ** 2, 0)), np.full(npk, normHm), E0, np.sqrt(np.maximum(E02 - E0 ** 2, 0)), Qm, rmean, rsp, Nm, vm, NmV,
                        sin_, ssh, sout, score, mp4, mtop, kabs, clock.copy()]
                for nm, val in zip(obs_names, vals):
                    series[sch][nm][:, ik] = val
            if k < rec1:
                PsiE = expm_multiply(-1j * tau * Hm, PsiE, traceA=0.0)
                for s_ in range(a.nsub):
                    th = (s_ + 0.5) / a.nsub
                    Hs = PC.clocked(h0, Q, (1 - th) * Napp[k] + th * Napp[k + 1], (1 - th) * vapp[k] + th * vapp[k + 1])
                    PsiI = expm_multiply(-1j * (tau / a.nsub) * Hs, PsiI, traceA=0.0)
            if ik % 20 == 0 or k == rec1:
                print("  kappa %d record %d (T/rs %.3f): ||h_m|| %.5f; E: norm dev max %.1e, <r> %s; I: <r> %s (%.0f s)" % (
                    kappa, k, T[k] / rs, normHm, np.max(np.abs(series["E"]["norm"][:, ik] - 1)),
                    np.array2string(series["E"]["rmean"][:, ik], precision=1, max_line_width=300), np.array2string(series["I"]["rmean"][:, ik], precision=1, max_line_width=300),
                    time.time() - tp), flush=True)
        # the classical null rays, one per launch radius and direction, beside every moving packet of that direction
        rays = {}
        for (lab, r0, k0) in L:
            key = (lab, 1.0 if k0 > 0 else -1.0)
            if k0 != 0.0 and key not in rays:
                rays[key] = trace_ray(F, r0, key[1], T[a.rec0], T[rec1], tau / 8.0)
                sfx = "out" if key[1] > 0 else "in"
                arrays["ray_k%d_%s_%s_r" % (kappa, lab, sfx)] = rays[key]["r"]; arrays["ray_k%d_%s_%s_lneps" % (kappa, lab, sfx)] = rays[key]["ln_eps"]
                arrays["ray_k%d_%s_%s_lambda" % (kappa, lab, sfx)] = rays[key]["lambda"]
        # summary per packet
        ch = {"kappa": kappa, "weight": 16.0 * kappa, "norm_h0": norm_h0, "norm_Q": norm_Q, "band_top_free": b_free, "positive_band": pinfo, "packets": []}
        for ip, (lab, r0, k0) in enumerate(L):
            p = {"launch": lab, "r0": r0, "k0": k0, "projected_fraction": float(proj_frac[ip]), "dispersion": dispersion(k0)}
            for sch in ("E", "I"):
                S = series[sch]
                p[sch] = {"norm_dev_max": float(np.max(np.abs(S["norm"][ip] - 1))),
                          "E0_launch": float(S["E0"][ip, 0]), "E0_max": float(np.max(S["E0"][ip])), "E0_min": float(np.min(S["E0"][ip])), "E0_end": float(S["E0"][ip, -1]),
                          "E0_max_over_b": float(np.max(S["E0"][ip]) / b_free), "dE0_max": float(np.max(S["dE0"][ip])),
                          "Em_launch": float(S["Em"][ip, 0]), "Em_min": float(np.min(S["Em"][ip])), "Em_max": float(np.max(S["Em"][ip])),
                          "absEm_over_normHm_max": float(np.max(np.abs(S["Em"][ip]) / S["normHm"][ip])),
                          "rmean_launch": float(S["rmean"][ip, 0]), "rmean_end": float(S["rmean"][ip, -1]), "rmean_min": float(np.min(S["rmean"][ip])), "rmean_max": float(np.max(S["rmean"][ip])),
                          "rspread_launch": float(S["rspread"][ip, 0]), "rspread_end": float(S["rspread"][ip, -1]),
                          "share_in_end": float(S["share_in"][ip, -1]), "share_shell_end": float(S["share_shell"][ip, -1]), "share_out_end": float(S["share_out"][ip, -1]),
                          "share_core_max": float(np.nanmax(S["share_core"][ip])), "share_core_end": float(S["share_core"][ip, -1]),
                          "share_in_max": float(np.nanmax(S["share_in"][ip])),
                          "T_over_rs_share_in_first_half": (float(T[ks[int(np.argmax(S["share_in"][ip] > 0.5))]] / rs) if np.any(S["share_in"][ip] > 0.5) else None),
                          "mom_top_max": float(np.max(S["mom_top"][ip])), "mom_pi4_max": float(np.max(S["mom_pi4"][ip])), "kabs_launch": float(S["kabs"][ip, 0]), "kabs_end": float(S["kabs"][ip, -1]),
                          "Nm_min": float(np.min(S["Nm"][ip])), "Nm_end": float(S["Nm"][ip, -1]), "NmV_min": float(np.min(S["NmV"][ip])),
                          "clock_end": float(S["clock"][ip, -1]), "clock_rate_last5rs": float((S["clock"][ip, -1] - S["clock"][ip, max(0, len(ks) - 1 - 40)]) / (T[ks[-1]] - T[ks[max(0, len(ks) - 1 - 40)]]))}
            key = (lab, 1.0 if k0 > 0 else -1.0)
            if k0 != 0.0 and key in rays:
                ry = rays[key]
                p["ray"] = {"escaped": bool(ry["escaped"]), "r_end": float(ry["r"][-1]), "ln_eps_end": float(ry["ln_eps"][-1]), "lambda_end": float(ry["lambda"][-1]),
                            "lambda_growth_last5rs": float(ry["lambda"][-1] - ry["lambda"][max(0, len(ry["lambda"]) - 1 - 40)]),
                            "T_over_rs_first_r_below_1": (float(ry["T"][int(np.argmax(ry["r"] < 1.0))] / rs) if np.any(ry["r"] < 1.0) else None),
                            "r_min": float(np.min(ry["r"])), "ln_eps_max": float(np.max(ry["ln_eps"]))}
            # the packet's difference between the two rules
            p["E_vs_I"] = {"rmean_max_abs_diff": float(np.max(np.abs(series["E"]["rmean"][ip] - series["I"]["rmean"][ip]))),
                           "E0_max_abs_diff": float(np.max(np.abs(series["E"]["E0"][ip] - series["I"]["E0"][ip]))),
                           "overlap_end": float(abs(np.vdot(PsiE[:, ip], PsiI[:, ip])) ** 2)}
            ch["packets"].append(p)
        out["channels"][str(kappa)] = ch
        for sch in ("E", "I"):
            for nm in obs_names:
                arrays["%s_k%d_%s" % (sch, kappa, nm)] = series[sch][nm]
        arrays["normHm_k%d" % kappa] = normHm_rec
        arrays["launch_r0_k%d" % kappa] = np.array([r0 for (_, r0, _) in L]); arrays["launch_k0_k%d" % kappa] = np.array([k0 for (_, _, k0) in L])
        arrays["launch_label_k%d" % kappa] = np.array([lab for (lab, _, _) in L])
        print("kappa %d done (%.0f s)" % (kappa, time.time() - tk), flush=True)
    arrays["T"] = T[ks]; arrays["T_over_rs"] = T[ks] / rs; arrays["records"] = np.array(ks)
    arrays["r_in"] = np.array([zeros[k][0] if len(zeros[k]) >= 2 else np.nan for k in ks]); arrays["r_out"] = np.array([zeros[k][-1] if len(zeros[k]) >= 2 else np.nan for k in ks])
    arrays["N_min"] = np.array([float(recs[k]["N_min"]) for k in ks])
    arrays["vN_max"] = vN_max[ks]; arrays["vN_argmax"] = vN_argmax[ks]
    out["meta"]["vN_max_over_window"] = {"max": float(vN_max[ks].max()), "T_over_rs_at_max": float(T[ks[int(np.argmax(vN_max[ks]))]] / rs), "r_at_max": float(vN_argmax[ks[int(np.argmax(vN_max[ks]))]]),
                                         "min_N_applied_over_window": float(Napp[ks].min())}
    np.savez_compressed(os.path.join(a.out, "packet_v1_series.npz"), **arrays)
    out["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "packet_v1_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
