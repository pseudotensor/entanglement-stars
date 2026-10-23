#!/usr/bin/env python3
"""flow_page_v2.py (version 2 of flow_page_v1.py; the changes at the end of this docstring): task 6 of final4 (item 7, the page curve of the quasi-frozen radiating object on the flow chart), the lead's sec. 8 drafts
of solve/specs/SPEC_task6_page_curve.md completed by the builder analyst.  Imports flow_chart_v4.py (the operator, the packets) and
flow_chart_v6.py (the cut index, the roots) unchanged.  Every record: "prescribed flow (simpler version); PG radial; v declared; formation
history declared (the ramp)" and the script name, never a hash.

One call = one or more radial channels (m, j, sign of kappa_j) on one GPU:
  the flat sea P_0 of h_0 (v = 0) from one dense eigendecomposition; the formation ramp v(R, t) = sin^2(pi t/(2 t_r)) v_final(R) applied to the
  orbital matrix V_0 by a fourth-order propagator; one dense eigendecomposition of h_f; the hold exactly in h_f's eigenbasis.  Observables
  at the hold times t in {0, 6, 12, 24, 36, 48}/kappa after the ramp (kappa = 1/(2 r_s)):
  P1 S_A for A = [1, R_c] (the sites R = 1 .. R_c), R_c = r_s and 2 r_s, formed and flat, the excess, the growth rate by a linear fit over
     t in [12, 48]/kappa;
  P2 (i) I(A_in : A_out), A_in = [r_s - d, r_s), A_out = [r_s, r_s + d), d in {2, 4, 8, 16}, formed and flat; (ii) the cross block
     G = P_in C P_out: singular values, ||G||_HS^2, and the excess block's; (iii) the partner w_p = (1 - |w><w|) C w of outgoing packets at
     omega/kappa in {1/4, 1/2, 1, 2}: n, ||w_p||^2 against n(1 - n), w_p's weights in [1, R_inner), [R_inner, r_s), [r_s, R_max], its
     Killing mean in h_f's eigenbasis and its weight at omega < 0 (at 24/kappa and 48/kappa);
  P3 (a) the back-trace exp(+i h_f tau) w for tau in [0, 8/kappa] (step 1/(4 kappa)): the packet's position and wavenumber, the fit of
     ln(x - r_s) against tau over 2 < x - r_s < r_s/4; (b) the trace-back through the ramp, U_ramp^dagger exp(+i h_f (t_l - t_r)) w, its
     centre at t = 0 against t_l - t_r in [0, 6/kappa];
  P4 the occupations n = w^dagger C(t) w of the outgoing packets at omega/kappa in {1/4, 1/2, 1, 2, 4} at every hold time, and theta from the
     packet-averaged Fermi function with the greybody factor set to 1 (the factor from task 5's E2 is divided out in the report);
  checks (P0, P6): Phi^dagger Phi = 1 (C^2 = C), the energy balance E(t_r) - E(0) against the drive's work int s'(t) tr(h_flow C) dt, the
  pure-state identity S_A from the singular values of G_{A A'} and S_{A'} (with --p0).

BUILDER choices (also in the records):
  (C1) the package's S4 split propagator (input 7, eq. 3) is replaced on the dense chain by the commutator-free fourth-order Magnus scheme
       (two exponentials per step at the Gauss points' combinations; Blanes and Moan's CF4) with each exponential by the Chebyshev expansion of
       the banded h (Gershgorin bounds, terms to 1e-15); both are fourth order in dt with exact or near-exact substeps; the package itself is
       P5's cross-check.
  (C2) A = [1, R_c] includes the site R = R_c (the spec's literal interval); the shells as in P2.
  (C3) the outgoing packets: version 4's branch_momenta on the local outgoing branch at x_0 (task 3 B's packet measure), sigma_omega =
       min(omega/4, (omega - m)/3) at m > 0, the local spinor; x_0 = max(4 r_s, r_s + 4 sigma_x) so that the packet's 4 sigma tail stays outside
       r_s (at omega/kappa = 1/4 sigma_x exceeds 4 r_s); the packet's Killing mean and spread are printed.
  (C4) P3(a): the packet's position is the density-weighted centre within 3 widths of the density's peak in R > r_s, its wavenumber the
       FFT-weighted mean over k > 0 of the state masked to that window (task 5 E3's annulus method).
  (C5) the hold keeps X = Q_f^dagger Phi(t_r) and forms only the rows needed (the sites R <= 2 r_s) and matrix-vector products; Phi(t) is never
       formed in full except for --p0.
Version 2 (after version 1's first channel, r_s = 32, j = 1/2, kappa_j = -1): (i) the back-trace of P3(a) runs to tau_max = 8/kappa +
1.5 (x_0 - 4 r_s) (version 1 stopped at 8/kappa while the packets of (C3) start at x_0 up to 27 r_s, so the trace never reached the window
2 < x - r_s < r_s/4); (ii) the hold times add 64, 80 and 96/kappa (the default list; the wall's echo returns to the cut at about 2 R_max = 128/kappa
and to the packets at about 2 R_max - x_0), the late growth fit over [48, 96]/kappa beside the spec's [12, 48]/kappa, the shells and partners
at 96/kappa beside 24 and 48, and the raw-temperature plateau over the holds from 48/kappa on (BUILDER: version 1's occupations at omega/kappa >= 2 were still falling at 48/kappa, the formation transient).
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.special import jv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flow_chart_v4 as V4  # noqa: E402

LABEL = "prescribed flow (simpler version); PG radial; v declared; formation history declared (the ramp)"
T0 = time.time()


def log(msg):
    print("[%7.0f s] %s" % (time.time() - T0, msg), flush=True)


def binary_entropy(lam):
    lam = np.clip(np.asarray(lam, float), 0.0, 1.0)
    q = lam[(lam > 1e-16) & (lam < 1 - 1e-16)]
    return float(np.sum(-q * np.log(q) - (1 - q) * np.log1p(-q)))


class Banded:
    """h(s) = h_0 + s h_flow as component diagonals (offsets -5 .. 5) on the device."""

    def __init__(self, h0, hfl, torch, dev):
        self.torch, self.n = torch, h0.shape[0]
        self.d0, self.dfl = [], []
        for d in range(-5, 6):
            a, b = np.diagonal(h0, d).copy(), np.diagonal(hfl, d).copy()
            if np.max(np.abs(a)) > 0 or np.max(np.abs(b)) > 0:
                self.d0.append((d, torch.as_tensor(a, device=dev)))
                self.dfl.append((d, torch.as_tensor(b, device=dev)))
        g = np.zeros(self.n)
        for d in range(-5, 6):
            for M in (h0, hfl):
                x = np.abs(np.diagonal(M, d))
                if d >= 0:
                    g[:self.n - d] += x
                else:
                    g[-d:] += x
        self.bound = float(g.max()) * 1.0001

    def apply(self, X, s=1.0, flow_only=False):
        Y = self.torch.zeros_like(X)
        n = self.n
        for (d, D0), (_, Df) in zip(self.d0, self.dfl):
            D = Df if flow_only else (D0 + s * Df if s != 0 else D0)
            if d >= 0:
                Y[:n - d] += D[:, None] * X[d:]
            else:
                Y[-d:] += D[:, None] * X[:n + d]
        return Y


def cheb_expm(B, X, tau, s, tol=1e-15):
    """exp(-i tau h(s)) X by the Chebyshev expansion on [-R, R] (R the Gershgorin bound of h(s) for |s| <= 1)."""
    r = B.bound
    a = r * tau
    K = int(abs(a) + 12)
    while abs(jv(K, abs(a))) > tol or K < abs(a) + 4:
        K += 4
    T0_, T1_ = X, B.apply(X, s) / r
    Y = jv(0, a) * T0_ + 2 * (-1j) * jv(1, a) * T1_
    for k in range(2, K + 1):
        T2_ = 2 * B.apply(T1_, s) / r - T0_
        Y = Y + 2 * ((-1j) ** k) * jv(k, a) * T2_
        T0_, T1_ = T1_, T2_
    return Y


SQ3 = math.sqrt(3.0)
CF_C = (0.5 - SQ3 / 6, 0.5 + SQ3 / 6)
CF_A = (0.25 + SQ3 / 6, 0.25 - SQ3 / 6)


def envelope(t, tr):
    return 0.0 if t <= 0 else (1.0 if t >= tr else math.sin(math.pi * t / (2 * tr)) ** 2)


def ramp_forward(B, X, tr, dt, work=None):
    """CF4 over the ramp: per step exp(-i dt/2 h(s_b)) exp(-i dt/2 h(s_a)), s_a = 2(a1 s1 + a2 s2), s_b = 2(a2 s1 + a1 s2)."""
    nst = max(1, int(math.ceil(tr / dt - 1e-9)))
    h = tr / nst
    trs = []
    for i in range(nst + 1):
        if work is not None:
            trs.append(float(B.torch.real(B.torch.sum(X.conj() * B.apply(X, flow_only=True)))))
        if i == nst:
            break
        t = i * h
        s1, s2 = envelope(t + CF_C[0] * h, tr), envelope(t + CF_C[1] * h, tr)
        sa, sb = 2 * (CF_A[0] * s1 + CF_A[1] * s2), 2 * (CF_A[1] * s1 + CF_A[0] * s2)
        X = cheb_expm(B, X, h / 2, sa)
        X = cheb_expm(B, X, h / 2, sb)
    if work is not None:
        ts = np.arange(nst + 1) * h
        sdot = (math.pi / (2 * tr)) * np.sin(math.pi * ts / tr)
        f = sdot * np.array(trs)
        if nst % 2 == 0:
            W = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
        else:
            W = float(np.trapz(f, ts))
        work["W"] = float(W)
        work["steps"] = nst
    return X


def ramp_backward(B, X, tr, dt):
    """U_ramp^dagger X: the steps in reverse, each step's two exponentials reversed with +i."""
    nst = max(1, int(math.ceil(tr / dt - 1e-9)))
    h = tr / nst
    for i in reversed(range(nst)):
        t = i * h
        s1, s2 = envelope(t + CF_C[0] * h, tr), envelope(t + CF_C[1] * h, tr)
        sa, sb = 2 * (CF_A[0] * s1 + CF_A[1] * s2), 2 * (CF_A[1] * s1 + CF_A[0] * s2)
        X = cheb_expm(B, X, -h / 2, sb)
        X = cheb_expm(B, X, -h / 2, sa)
    return X


def comps(sites):
    sites = np.asarray(sites, int)
    return np.stack([2 * sites, 2 * sites + 1], axis=1).reshape(-1)


def outgoing_packet(N, R, rs, Rm, m, kj, omega):
    """(C3) task 3 B's packet on the local outgoing branch, centred at x_0 = max(4 r_s, r_s + 4 sigma_x)."""
    sg = min(omega / 4, (omega - m) / 3) if m > 0 else omega / 4
    if sg <= 0 or omega - 3 * sg <= m:
        return None, "below the threshold window"
    x0 = 4 * rs
    for _ in range(4):
        vext, aext = float(V4.pg_profile(x0, rs, Rm)), kj / x0
        me = float(np.sqrt(m * m + aext * aext))
        try:
            k, om, a2 = V4.branch_momenta(omega, sg, me, vext, outgoing=True)
        except ValueError as ex:
            return None, str(ex)
        vg0 = abs(float(np.sum(a2 * V4.group_velocity(k, me, vext) * np.gradient(k))))
        sx = vg0 / (2 * sg)
        x1 = max(4 * rs, rs + 4 * sx)
        if abs(x1 - x0) < 1e-9:
            break
        x0 = x1
    if x0 + 4 * sx > N - 2:
        return None, "the packet does not fit (x0 %.0f, sigma_x %.0f)" % (x0, sx)
    w = V4.build_packet(N, int(round(x0 - 1)), k, a2, me, m=m, a=kj / x0)
    return {"w": w, "x0": x0, "sigma_x": sx, "sigma_omega": sg, "omega": omega, "k": k, "om": om, "a2": a2}, None


def window_centre_k(psi, R, rs):
    """(C4) the packet's position (within 3 widths of the peak in R > r_s) and its wavenumber (FFT over the window, k > 0)."""
    rho = V4.density(psi)
    ext = R > rs
    if rho[ext].sum() <= 0:
        return None, None, None
    ip = int(np.argmax(np.where(ext, rho, 0.0)))
    sel = ext.copy()
    for _ in range(3):
        rr = rho * sel
        c = float((rr * R).sum() / rr.sum())
        wd = float(np.sqrt((rr * (R - c) ** 2).sum() / rr.sum()))
        sel = ext & (np.abs(R - R[ip]) <= max(3 * wd, 2.0))
    rr = rho * sel
    c = float((rr * R).sum() / rr.sum())
    a = np.where(sel[:, None], psi.reshape(-1, 2), 0.0)
    nfft = 1 << int(math.ceil(math.log2(max(4 * len(R), 1024))))
    kf = 2 * np.pi * np.fft.fftfreq(nfft)
    pk = (np.abs(np.fft.fft(a, n=nfft, axis=0)) ** 2).sum(1)
    pos = kf > 0
    kbar = float((kf[pos] * pk[pos]).sum() / max(pk[pos].sum(), 1e-300))
    return c, kbar, float(rr.sum())


class Formation:
    def __init__(self, a, m, kj, gpu):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.dev = torch.device("cuda:%d" % gpu) if gpu is not None else torch.device("cpu")
        self.rs, self.Rm, self.N, self.m, self.kj = a.rs, a.Rm, a.nsites, m, kj
        self.kappa = 1.0 / (2 * a.rs)
        self.tr = a.ramp_over_rs * a.rs
        self.dt = a.dt
        mz, yy, R = V4.radial_fields(self.N, m, kj)
        self.R = R
        self.vf = V4.pg_profile(R, a.rs, a.Rm)
        h0 = V4.chain_hamiltonian(mz, yy, np.zeros(self.N))
        hf = V4.chain_hamiltonian(mz, yy, self.vf)
        self.B = Banded(h0, hf - h0, torch, self.dev)
        t0 = time.time()
        e0, Q0 = torch.linalg.eigh(torch.as_tensor(h0, device=self.dev))
        del h0
        occ = e0 < 0
        self.E0 = float(e0[occ].sum())
        self.nocc = int(occ.sum())
        self.V0 = Q0[:, occ].contiguous()
        del Q0
        self.eig0_s = time.time() - t0
        t0 = time.time()
        work = {}
        self.Phi_r = ramp_forward(self.B, self.V0.clone(), self.tr, self.dt, work)
        self.ramp_s = time.time() - t0
        self.work = work
        t0 = time.time()
        ef, Qf = torch.linalg.eigh(torch.as_tensor(hf, device=self.dev))
        del hf
        self.ef, self.Qf = ef, Qf
        self.eigf_s = time.time() - t0
        self.X = Qf.conj().T @ self.Phi_r
        self.Er = float(torch.sum(ef[:, None] * (self.X.abs() ** 2)).real)
        G = self.Phi_r.conj().T @ self.Phi_r
        G.diagonal().sub_(1.0)
        self.orth = float(G.abs().max())
        del G
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

    def rows_at(self, rows, tau):
        ph = self.torch.exp(-1j * self.ef * float(tau))
        return (self.Qf[list(rows), :] @ (ph[:, None] * self.X)).cpu().numpy()

    def flat_rows(self, rows):
        return self.V0[list(rows), :].cpu().numpy()

    def apply_C(self, w, tau):
        """C(t) w = Phi (Phi^dagger w) with Phi(t_r + tau) = Q_f e^{-i e_f tau} X."""
        torch = self.torch
        wt = torch.as_tensor(np.ascontiguousarray(w), device=self.dev)
        ph = torch.exp(-1j * self.ef * float(tau))
        y = self.X.conj().T @ (ph.conj() * (self.Qf.conj().T @ wt))
        return (self.Qf @ (ph * (self.X @ y))).cpu().numpy(), float((y.abs() ** 2).sum())

    def killing(self, v):
        c = (self.Qf.conj().T @ self.torch.as_tensor(np.ascontiguousarray(v), device=self.dev)).abs() ** 2
        tot = float(c.sum())
        c = c / tot
        return {"omega_mean": float((self.ef * c).sum()), "omega_spread": float(math.sqrt(max(float((self.ef ** 2 * c).sum()) - float((self.ef * c).sum()) ** 2, 0))),
                "weight_omega_neg": float(c[self.ef < 0].sum())}

    def evolve_f(self, w, tau):
        torch = self.torch
        c = self.Qf.conj().T @ torch.as_tensor(np.ascontiguousarray(w), device=self.dev)
        return (self.Qf @ (torch.exp(-1j * self.ef * float(tau)) * c)).cpu().numpy()

    def free(self):
        for k in ("V0", "Phi_r", "ef", "Qf", "X"):
            if hasattr(self, k):
                delattr(self, k)
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()


def shell_blocks(Cfull_rows, rs, d):
    """C restricted to A_in u A_out from the rows of the sites R <= 2 r_s (indices 0 .. 2 r_s - 1)."""
    ins = np.arange(rs - d - 1, rs - 1)
    outs = np.arange(rs - 1, rs + d - 1)
    ci, co = comps(ins), comps(outs)
    return ci, co


def run_channel(a, m, j, sign, gpu):
    kj = sign * (j + 0.5)
    rs = a.rs
    F = Formation(a, m, kj, gpu)
    log("channel m %g j %g kappa_j %g: eig0 %.0f s, ramp %.0f s (%d steps), eigf %.0f s, orth %.2e" % (m, j, kj, F.eig0_s, F.ramp_s, F.work["steps"],
                                                                                                    F.eigf_s, F.orth))
    kappa = F.kappa
    out = {"mass": m, "j": j, "kappa_j": kj, "N": F.N, "rs": rs, "Rm": a.Rm, "R_inner": math.sqrt(a.Rm ** 3 / rs), "kappa": kappa, "t_r": F.tr,
           "dt": F.dt, "n_occ": F.nocc, "timing_s": {"eig0": F.eig0_s, "ramp": F.ramp_s, "eigf": F.eigf_s},
           "checks": {"PhiH_Phi_minus_1_max": F.orth, "E0": F.E0, "E_tr": F.Er, "work": F.work["W"], "energy_balance": F.Er - F.E0 - F.work["W"],
                      "energy_balance_rel": (F.Er - F.E0 - F.work["W"]) / max(abs(F.work["W"]), 1e-300)}}
    Rc_list = [int(round(rs)), int(round(2 * rs))]
    ds = [d for d in (2, 4, 8, 16) if int(round(rs)) - d - 1 >= 0]
    nsA = max(max(Rc_list), int(round(rs)) + max(ds))
    rowsA = comps(np.arange(0, nsA))
    flatA = F.flat_rows(rowsA)
    CflatA = flatA @ flatA.conj().T
    holds = [float(x) for x in a.holds]
    S = {"flat": {}, "formed": {}}
    for Rc in Rc_list:
        S["flat"][str(Rc)] = binary_entropy(np.linalg.eigvalsh(CflatA[:2 * Rc, :2 * Rc]))
    shells = {}
    blocks = {"flat": CflatA}
    for th in holds:
        PA = F.rows_at(rowsA, th / kappa)
        CA = PA @ PA.conj().T
        blocks["formed_%g" % th] = CA
        S["formed"][str(th)] = {str(Rc): binary_entropy(np.linalg.eigvalsh(CA[:2 * Rc, :2 * Rc])) for Rc in Rc_list}
        if th in (24.0, 48.0, 96.0):
            shells[str(th)] = {}
            for d in ds:
                ci, co = shell_blocks(None, int(round(rs)), d)
                both = np.concatenate([ci, co])
                rec = {}
                for name, C in (("formed", CA), ("flat", CflatA)):
                    Si = binary_entropy(np.linalg.eigvalsh(C[np.ix_(ci, ci)]))
                    So = binary_entropy(np.linalg.eigvalsh(C[np.ix_(co, co)]))
                    Sb = binary_entropy(np.linalg.eigvalsh(C[np.ix_(both, both)]))
                    G = C[np.ix_(ci, co)]
                    sv = np.linalg.svd(G, compute_uv=False)
                    rec[name] = {"S_in": Si, "S_out": So, "S_both": Sb, "I": Si + So - Sb, "G_sv": sv.tolist(), "G_HS2": float(np.sum(sv ** 2))}
                dG = CA[np.ix_(ci, co)] - CflatA[np.ix_(ci, co)]
                svd = np.linalg.svd(dG, compute_uv=False)
                rec["excess"] = {"I": rec["formed"]["I"] - rec["flat"]["I"], "dG_sv": svd.tolist(), "dG_HS2": float(np.sum(svd ** 2))}
                shells[str(th)][str(d)] = rec
    out["S_A"] = S
    out["excess"] = {str(Rc): {str(th): S["formed"][str(th)][str(Rc)] - S["flat"][str(Rc)] for th in holds} for Rc in Rc_list}
    fit = {}
    for lo_, hi_, key_ in ((12.0, 48.0, ""), (48.0, 96.0, "_late")):
        for Rc in Rc_list:
            xs = [th / kappa for th in holds if lo_ <= th <= hi_]
            ys = [out["excess"][str(Rc)][str(th)] for th in holds if lo_ <= th <= hi_]
            if len(xs) >= 2:
                A_ = np.vstack([xs, np.ones(len(xs))]).T
                coef, *_ = np.linalg.lstsq(A_, ys, rcond=None)
                res = np.array(ys) - A_ @ coef
                fit[str(Rc) + key_] = {"rate": float(coef[0]), "rate_over_kappa": float(coef[0] / kappa), "intercept": float(coef[1]),
                                       "rms_residual": float(np.sqrt(np.mean(res ** 2))), "n": len(xs), "window_over_kappa": [lo_, hi_]}
    out["growth_fit"] = fit
    out["shells"] = shells
    log("  S_A excess at r_s: %s; rate/kappa %s" % ({k: round(v, 5) for k, v in out["excess"][str(Rc_list[0])].items()},
                                                    {k: round(v["rate_over_kappa"], 5) for k, v in fit.items()}))
    # packets: P2(iii), P4 (j <= 5/2), P3 (j = 1/2, m = 0)
    pk_rows = []
    if j <= a.packet_jmax + 1e-9:
        for r_ in a.omega_over_kappa:
            pk, why = outgoing_packet(F.N, F.R, rs, a.Rm, m, kj, r_ * kappa)
            if pk is None:
                pk_rows.append({"omega_over_kappa": r_, "skipped": why})
                continue
            w = pk["w"]
            row = {"omega_over_kappa": r_, "omega": pk["omega"], "x0": pk["x0"], "sigma_x": pk["sigma_x"], "sigma_omega": pk["sigma_omega"],
                   "killing_w": F.killing(w), "F_w(1)": V4.fermi_average(pk["k"], pk["om"], pk["a2"], 1.0, kappa), "hold": {}}
            for th in holds:
                Cw, n = F.apply_C(w, th / kappa)
                h = {"n": n}
                try:
                    h["theta_raw"] = V4.invert_theta(n, pk["k"], pk["om"], pk["a2"], kappa)
                except ValueError:
                    h["theta_raw"] = None
                if th in (24.0, 48.0, 96.0) and r_ in a.partner_omegas:
                    wp = Cw - n * w
                    nrm2 = float(np.real(np.vdot(wp, wp)))
                    rho = V4.density(wp)
                    Rin = math.sqrt(a.Rm ** 3 / rs)
                    h["partner"] = {"norm2": nrm2, "n_1_minus_n": n * (1 - n), "identity_error": nrm2 - n * (1 - n),
                                    "weight_core": float(rho[F.R < Rin].sum()), "weight_trapped": float(rho[(F.R >= Rin) & (F.R < rs)].sum()),
                                    "weight_outside": float(rho[F.R >= rs].sum()),
                                    "fraction_trapped": float(rho[(F.R >= Rin) & (F.R < rs)].sum()) / nrm2 if nrm2 > 0 else None,
                                    **({"killing": F.killing(wp)} if nrm2 > 1e-24 else {})}
                row["hold"][str(th)] = h
            th_late = [row["hold"][str(th)]["theta_raw"] for th in holds if th >= 48.0 and row["hold"][str(th)]["theta_raw"] is not None]
            row["theta_raw_plateau"] = float(np.mean(th_late)) if th_late else None
            row["theta_raw_spread"] = float(np.ptp(th_late)) if th_late else None
            if m == 0 and abs(j - 0.5) < 1e-9 and r_ in a.raymap_omegas:
                taus = np.arange(0, 8.0 / kappa + 1.5 * max(pk["x0"] - 4 * rs, 0.0) + 1e-9, 0.25 / kappa)
                tr_ = []
                for tau in taus:
                    psi = F.evolve_f(w, -tau)
                    c, kb, wt = window_centre_k(psi, F.R, rs)
                    tr_.append({"tau": float(tau), "x": c, "k": kb, "window_weight": wt})
                pts = [q for q in tr_ if q["x"] is not None and 2.0 < q["x"] - rs < rs / 4]
                rm = {"trace": tr_}
                if len(pts) >= 3:
                    tt = np.array([q["tau"] for q in pts])
                    rm["fit_ln_x_minus_rs"] = np.polyfit(tt, np.log([q["x"] - rs for q in pts]), 1).tolist()
                    rm["kappa_p_from_x"] = -rm["fit_ln_x_minus_rs"][0]
                    kk = [q["k"] for q in pts if q["k"] and q["k"] > 0]
                    if len(kk) == len(pts):
                        rm["fit_ln_k"] = np.polyfit(tt, np.log(kk), 1).tolist()
                        rm["kappa_p_from_k"] = rm["fit_ln_k"][0]
                    rm["n_points"] = len(pts)
                tls = np.arange(0, 6.0 / kappa + 1e-9, 0.25 / kappa)
                torch = F.torch
                cols = np.stack([F.evolve_f(w, -tl) for tl in tls], axis=1)
                back = ramp_backward(F.B, torch.as_tensor(cols, device=F.dev), F.tr, F.dt).cpu().numpy()
                rb = []
                for i_, tl in enumerate(tls):
                    c, kb, wt = window_centre_k(back[:, i_], F.R, rs)
                    rb.append({"t_l_minus_t_r": float(tl), "x_at_t0": c, "k_at_t0": kb, "window_weight": wt,
                               "norm": float(np.real(np.vdot(back[:, i_], back[:, i_])))})
                rm["through_ramp"] = rb
                ok = [q for q in rb if q["x_at_t0"] is not None and q["x_at_t0"] - rs > 0]
                if len(ok) >= 3:
                    tt = np.array([q["t_l_minus_t_r"] for q in ok])
                    rm["fit_through_ramp_ln_x_minus_rs"] = np.polyfit(tt, np.log([q["x_at_t0"] - rs for q in ok]), 1).tolist()
                row["raymap"] = rm
                log("  raymap omega/kappa %g: kappa_p/kappa from x %s, from k %s" % (r_, (rm.get("kappa_p_from_x") or float("nan")) / kappa,
                                                                                    (rm.get("kappa_p_from_k") or float("nan")) / kappa))
            pk_rows.append(row)
    out["packets"] = pk_rows
    if a.p0:
        out["p0"] = p0_checks(F, a, int(round(rs)))
    np.savez_compressed(os.path.join(a.outdir, "blocks_m%g_j%g_k%+g.npz" % (m, j, kj)), **{k: v for k, v in blocks.items()},
                        rowsA_sites=np.arange(0, nsA))
    F.free()
    return out


def p0_checks(F, a, rsi):
    """C^2 = C (Phi^dagger Phi = 1, in the record), S_A from the singular values of G_{A A'} (the pure-state identity), S_A = S_{A'} at
    t = 12/kappa for A = [1, r_s]; the partner identity is in every packet row."""
    torch = F.torch
    tau = 12.0 / F.kappa
    ph = torch.exp(-1j * F.ef * tau)
    Phi = F.Qf @ (ph[:, None] * F.X)
    nA = 2 * rsi
    PA, PB = Phi[:nA], Phi[nA:]
    CA = (PA @ PA.conj().T).cpu().numpy()
    G = (PA @ PB.conj().T)
    sv = torch.linalg.svdvals(G).cpu().numpy()
    CB_eig = torch.linalg.eigvalsh(PB @ PB.conj().T).cpu().numpy()
    lam = np.linalg.eigvalsh(CA)
    S_direct = binary_entropy(lam)
    lam_sv = 0.5 * (1 - np.sqrt(np.clip(1 - 4 * sv ** 2, 0, 1)))
    S_sv = binary_entropy(lam_sv)
    S_B = binary_entropy(CB_eig)
    del Phi, PA, PB, G
    if F.dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"t": "12/kappa", "S_A_direct": S_direct, "S_A_from_G_singular_values": S_sv, "difference": S_direct - S_sv, "S_complement": S_B,
            "S_A_minus_S_complement": S_direct - S_B}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rs", type=float, default=32.0)
    p.add_argument("--Rm", type=float, default=8.0)
    p.add_argument("--nsites", type=int, default=4096)
    p.add_argument("--ramp-over-rs", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--holds", type=float, nargs="+", default=[0.0, 6.0, 12.0, 24.0, 36.0, 48.0, 64.0, 80.0, 96.0])
    p.add_argument("--channels", required=True, help="m:j:sign,... e.g. 0:0.5:-1,0:0.5:1")
    p.add_argument("--omega-over-kappa", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--partner-omegas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--raymap-omegas", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--packet-jmax", type=float, default=2.5)
    p.add_argument("--p0", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res = {"label": LABEL, "script": os.path.basename(__file__), "args": vars(a), "channels": []}
    path = os.path.join(a.outdir, "page_%s.json" % a.tag)
    for spec in a.channels.split(","):
        m, j, sg = (float(x) for x in spec.split(":"))
        res["channels"].append(run_channel(a, m, j, sg, a.gpu))
        res["wall_s"] = time.time() - T0
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
