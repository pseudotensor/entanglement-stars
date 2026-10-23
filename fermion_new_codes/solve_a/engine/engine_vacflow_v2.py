#!/usr/bin/env python
"""engine_vacflow_v2.py (builder analyst; job J-VF of SPEC_final4b_engine.md sec. 13.2; version 2 of engine_vacflow_v1.py).

Version 2: (1) the production loop's bug fixed: version 1 keyed the totals by the ramp strings of --tr-list ("2") and each channel's reads
by str(float) ("2.0"), so its production run stopped with a KeyError after the first channel (10:16 UTC, rs16_exact_m0; the benchmark
path used one key and was correct); one key function now; (2) the propagation's matrix product fused: h(s) = h0 + s V assembled once per
exponential on the union sparsity pattern (the same operator, one sparse product per Chebyshev term instead of two); (3) --kstride K
(default 1, the spec's every channel): channels 1..k_full every one, beyond k_full every K-th with weight K (the sum's trapezoid estimate);
labelled in the record; (4) per-channel records at the probe radii (the spec's "any truncation re-evaluated from the per-channel records").
Version 1's docstring follows.

engine_vacflow_v1.py (builder analyst; job J-VF of SPEC_final4b_engine.md sec. 13.2, task 1's first control under the goal of 09:07 UTC).

The fermion vacuum in the exact Painleve-Gullstrand flow at N = 1, switched on slowly from the flat sea: per channel kappa of the radial chain
(weights 16 kappa, s-wave bookkeeping), the flat sea's columns propagated through the ramp v(R, t) = s(t) v(R), s = sin^2(pi t/(2 t_r)), by
the commutator-free fourth-order scheme with Chebyshev exponentials on the banded chain (torch sparse on the GPU), read at the ramp's end and
after a hold of 2 r_s; three densities per site (shell totals): rho_raw = Tr[(C_tr - C_flat) hcal] (the transported sea against the flat sea:
what the declared subtraction removes), rho_NO = Tr[(C_tr - R0) hcal] (against the instantaneous ground state at the same v: the declared
source of the vacuum), rho_gs = rho_raw - rho_NO (R0's own polarization); hcal = (1/2){P_n, h_0} with h_0 the flat chain; R0 and C_flat by
eigh; the pocket count per channel n_neg(R0) - n_sea. Profiles: exact (v = -sqrt(r_s/R) for R >= R_m = r_s/8, linear inside) and the no-trap
v_0 (v for R >= 2 r_s, inside the odd cubic matched in value and slope at 2 r_s, |v_0| <= 0.73). Reported: the shell totals summed over the
channels; the running channel sums at probe radii (the convergence; any truncation re-evaluated from the per-channel records); the l = 0
smoothing at ell = 4, 8, 16 (engine_fields_v3's kernels); E_raw(R_out), E_NO(R_out) on a logarithmic R_out grid from 1.5 r_s to R_read and the
log-slope fit c of E_raw over the last factor 8 (against (9 pi/2) B_L r_s, P31's form, with B_L the lead's to supply), E_NO's slope beside it.
Builder decision (labelled): each channel's chain starts at the first site where kappa/R_n < E_cut (default 6): the sites inside carry only
centrifugal states of energy above E_cut, which the flow (|v| <= 2.8, the two-step amplitude |v|/4) moves adiabatically; C_flat, R0 and the
propagated sea are computed on the same truncated chain, and E_cut = 9 is the control (--Ecut); without it the spectral radius kappa/R_0 = kappa
would set the Chebyshev degree. Sizes per the spec: r_s in {16, 32}, N = 64 r_s sites, R_read = 16 r_s, kappa = 1..2 R_read, t_r in {2, 8, 32} r_s,
dt = 0.5 (0.25 the control), m a in {0, 0.2}.  --benchmark K runs channel K only and prints its timing and the dt control.
Args: --rs --profile (exact|notrap) --m --Ecut --kmax --tr-list --dt --gpu --out --tag --benchmark."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v3 as FLD  # noqa: E402

SQ3 = math.sqrt(3.0)
CF_C = (0.5 - SQ3 / 6, 0.5 + SQ3 / 6)
CF_A = (0.25 + SQ3 / 6, 0.25 - SQ3 / 6)


def profile(R, rs, kind):
    R = np.asarray(R, float)
    if kind == "exact":
        Rm = rs / 8.0
        return np.where(R >= Rm, -np.sqrt(rs / np.maximum(R, 1e-300)), -np.sqrt(rs / Rm) * R / Rm)
    x = R / (2 * rs)
    A, B = -1.2374368670764582, 0.5303300858899106          # v = A x + B x^3 inside 2 r_s: value -1/sqrt 2 and slope matched at x = 1
    return np.where(x >= 1.0, -np.sqrt(rs / np.maximum(R, 1e-300)), A * x + B * x ** 3)


def chain_sparse(R, m, kappa):
    """the flat chain h_0 (on-site m sz + (kappa/R) sy, sine bond -(i/2) sx) and the two-step Q (-(i/4) 1 to n + 2), sparse 2n x 2n."""
    n = len(R)
    rows, cols, vals = [], [], []
    for i in range(n):
        b = 2 * i
        rows += [b, b + 1, b, b + 1]; cols += [b, b + 1, b + 1, b]
        vals += [m, -m, -1j * kappa / R[i], 1j * kappa / R[i]]
    for i in range(n - 1):
        b, c = 2 * i, 2 * i + 2
        # -(i/2) sx from site i to i+1 and its conjugate
        for (r_, c_, v_) in ((b, c + 1, -0.5j), (b + 1, c, -0.5j)):
            rows += [r_, c_]; cols += [c_, r_]; vals += [v_, np.conj(v_)]
    h0 = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    rq, cq, vq = [], [], []
    for i in range(n - 2):
        for a in range(2):
            r_, c_ = 2 * i + a, 2 * i + 4 + a
            rq += [r_, c_]; cq += [c_, r_]; vq += [-0.25j, 0.25j]
    Q = sp.csr_matrix((vq, (rq, cq)), shape=(2 * n, 2 * n))
    return h0, Q


def flow_term(Q, v):
    vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return 0.5 * (vd @ Q + Q @ vd)


def site_density(A, Phi):
    """per site: sum over the columns of Re(Phi^dag P_n A Phi) with A sparse (the site-symmetrized density of A)."""
    AP = A @ Phi
    d = np.real(np.sum(np.conj(Phi) * AP, axis=1))
    return d.reshape(-1, 2).sum(axis=1)


class Prop:
    """the CF4 propagation of a block of columns under h(s) = h0 + s V on the GPU (torch sparse CSR), Chebyshev exponentials."""

    def __init__(self, h0, V, dev):
        import torch
        self.torch = torch
        self.dev = dev
        U = (abs(h0) + abs(V)).tocsr(); U.sort_indices()                    # the union pattern
        rows = np.repeat(np.arange(U.shape[0]), np.diff(U.indptr)); cols = U.indices
        d0 = np.asarray(h0.tocsr()[rows, cols]).ravel().astype(complex); dV = np.asarray(V.tocsr()[rows, cols]).ravel().astype(complex)
        import scipy.sparse as _sp
        chk = max(abs(_sp.csr_matrix((d0, cols, U.indptr), shape=U.shape) - h0).max(), abs(_sp.csr_matrix((dV, cols, U.indptr), shape=U.shape) - V).max())
        if chk > 1e-14:
            raise RuntimeError("the union-pattern copy of h0 and V differs by %g" % chk)
        self.crow = torch.as_tensor(U.indptr, dtype=torch.int64, device=dev)
        self.col = torch.as_tensor(U.indices, dtype=torch.int64, device=dev)
        self.d0 = torch.as_tensor(d0, dtype=torch.complex128, device=dev)
        self.dV = torch.as_tensor(dV, dtype=torch.complex128, device=dev)
        self.shape = U.shape
        self._s = None
        self.bound = float(max(np.abs(h0).sum(axis=1).max() + np.abs(V).sum(axis=1).max(), 1e-6)) * 1.01

    def _hs(self, s):
        if self._s != s:
            self._H = self.torch.sparse_csr_tensor(self.crow, self.col, self.d0 + s * self.dV, size=self.shape, device=self.dev)
            self._s = s
        return self._H

    def hmul(self, X, s):
        return self._hs(s) @ X

    def expm(self, X, tau, s):
        from scipy.special import jv
        r = self.bound
        a = r * tau
        K = int(abs(a) + 12)
        while abs(jv(K, abs(a))) > 1e-15 or K < abs(a) + 4:
            K += 4
        T0 = X; T1 = self.hmul(X, s) / r
        Y = jv(0, a) * T0 + 2 * (-1j) * jv(1, a) * T1
        for k in range(2, K + 1):
            T2 = 2 * self.hmul(T1, s) / r - T0
            Y = Y + 2 * ((-1j) ** k) * jv(k, a) * T2
            T0, T1 = T1, T2
        return Y

    def ramp(self, X, t_r, dt, hold):
        nst = max(1, int(math.ceil(t_r / dt - 1e-9)))
        h = t_r / nst
        env = lambda t: math.sin(math.pi * t / (2 * t_r)) ** 2
        for i in range(nst):
            t = i * h
            s1, s2 = env(t + CF_C[0] * h), env(t + CF_C[1] * h)
            X = self.expm(X, h / 2, 2 * (CF_A[0] * s1 + CF_A[1] * s2))
            X = self.expm(X, h / 2, 2 * (CF_A[1] * s1 + CF_A[0] * s2))
        X_end = X
        if hold > 0:
            nh = max(1, int(math.ceil(hold / (8 * dt))))
            for _ in range(nh):
                X = self.expm(X, hold / nh, 1.0)
        return X_end, X


def trkey(tr):
    return "%g" % float(tr)


def run_channel(kappa, Rfull, rs, kind, m, Ecut, tr_list, dt, dev):
    import torch
    n0 = int(np.searchsorted(Rfull, kappa / Ecut))           # the first site with kappa/R < Ecut
    R = Rfull[n0:]
    h0, Q = chain_sparse(R, m, kappa)
    V = flow_term(Q, profile(R, rs, kind))
    hd = torch.as_tensor(h0.toarray(), device=dev)
    e0, U0 = torch.linalg.eigh(hd)
    nsea = int((e0 < 0).sum())
    Phi = U0[:, e0 < 0]
    hf = torch.as_tensor((h0 + V).toarray(), device=dev)
    ef, Uf = torch.linalg.eigh(hf)
    R0 = Uf[:, ef < 0]
    nneg = int((ef < 0).sum())
    del hd, hf, U0, Uf
    Phi_np = Phi.cpu().numpy(); R0_np = R0.cpu().numpy()
    d_flat = site_density(h0, Phi_np)
    d_R0 = site_density(h0, R0_np)
    P = Prop(h0, V, dev)
    out = {"kappa": kappa, "n0": n0, "nsea": nsea, "pocket": nneg - nsea, "bound": P.bound, "reads": {}}
    for tr in tr_list:
        t0 = time.time()
        Xe, Xh = P.ramp(Phi, tr * rs, dt, 2 * rs)
        de = site_density(h0, Xe.cpu().numpy()); dh = site_density(h0, Xh.cpu().numpy())
        pad = lambda a: np.concatenate([np.zeros(n0), a])
        out["reads"][trkey(tr)] = {"raw_end": pad(de - d_flat), "NO_end": pad(de - d_R0), "raw_hold": pad(dh - d_flat), "NO_hold": pad(dh - d_R0),
                                 "seconds": time.time() - t0}
        del Xe, Xh
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=16.0)
    ap.add_argument("--profile", default="exact", choices=["exact", "notrap"])
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--Ecut", type=float, default=6.0)
    ap.add_argument("--kmax", type=int, default=None, help="default 2 R_read = 32 r_s")
    ap.add_argument("--tr-list", default="2,8,32")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--benchmark", type=int, default=None)
    ap.add_argument("--kstride", type=int, default=1, help="beyond --kfull every K-th channel with weight K (labelled)")
    ap.add_argument("--kfull", type=int, default=64, help="channels 1..kfull always every one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda:%d" % a.gpu) if a.gpu is not None else torch.device("cpu")
    rs = a.rs
    nsites = int(64 * rs)
    Rfull = np.arange(nsites, dtype=float) + 1.0
    R_read = 16 * rs
    kmax = a.kmax or int(2 * R_read)
    trs = [float(x) for x in a.tr_list.split(",")]
    os.makedirs(a.out, exist_ok=True)
    if a.benchmark:
        t0 = time.time()
        c1 = run_channel(a.benchmark, Rfull, rs, a.profile, a.m, a.Ecut, trs[:1], a.dt, dev)
        t1 = time.time()
        c2 = run_channel(a.benchmark, Rfull, rs, a.profile, a.m, a.Ecut, trs[:1], a.dt / 2, dev)
        c3 = run_channel(a.benchmark, Rfull, rs, a.profile, a.m, a.Ecut * 1.5, trs[:1], a.dt, dev)
        k = trkey(trs[0])
        sel = Rfull < R_read
        dd = lambda x, y, key: float(np.max(np.abs(x["reads"][k][key][sel] - y["reads"][k][key][sel])))
        print("benchmark kappa %d: %.1f s per ramp t_r = %g r_s (bound %.2f, n0 %d, pocket %d); dt/2 change raw %.2e NO %.2e; Ecut x1.5 change raw %.2e NO %.2e; max |raw| %.2e |NO| %.2e" % (
            a.benchmark, t1 - t0, trs[0], c1["bound"], c1["n0"], c1["pocket"], dd(c1, c2, "raw_hold"), dd(c1, c2, "NO_hold"), dd(c1, c3, "raw_hold"), dd(c1, c3, "NO_hold"),
            float(np.max(np.abs(c1["reads"][k]["raw_hold"][sel]))), float(np.max(np.abs(c1["reads"][k]["NO_hold"][sel])))), flush=True)
        return
    tot = {trkey(tr): {key: np.zeros(nsites) for key in ("raw_end", "NO_end", "raw_hold", "NO_hold")} for tr in trs}
    probes = [r for r in (1.5 * rs, 2 * rs, 4 * rs, 8 * rs, 16 * rs) if r < nsites]
    running = {tr: [] for tr in tot}
    pockets = []
    t_all = time.time()
    path = os.path.join(a.out, "vacflow_%s.json" % a.tag)
    klist = list(range(1, min(a.kfull, kmax) + 1)) + list(range(min(a.kfull, kmax) + a.kstride, kmax + 1, a.kstride))
    per_channel = []
    for kappa in klist:
        c = run_channel(kappa, Rfull, rs, a.profile, a.m, a.Ecut, trs, a.dt, dev)
        w = 16.0 * kappa * (a.kstride if kappa > a.kfull else 1)
        per_channel.append({"kappa": kappa, "weight": w, "pocket": c["pocket"], "n0": c["n0"],
                            "at_probes": {tr: {key: [float(c["reads"][tr][key][int(p) - 1]) for p in probes] for key in ("raw_hold", "NO_hold")} for tr in tot}})
        pockets.append((kappa, c["pocket"]))
        for tr in tot:
            for key in tot[tr]:
                tot[tr][key] += w * c["reads"][tr][key]
            cs = np.cumsum(tot[tr]["raw_hold"]); cn = np.cumsum(tot[tr]["NO_hold"])
            running[tr].append({"kappa": kappa, "E_raw_at": [float(cs[int(p) - 1]) for p in probes], "E_NO_at": [float(cn[int(p) - 1]) for p in probes]})
        if kappa % 16 == 0 or kappa == klist[-1]:
            print("kappa %d (%.0f s): pocket %d, E_raw(R_read) %s, E_NO(R_read) %s" % (kappa, time.time() - t_all, c["pocket"],
                  {tr: "%.4e" % float(np.cumsum(tot[tr]["raw_hold"])[int(R_read) - 1]) for tr in tot},
                  {tr: "%.4e" % float(np.cumsum(tot[tr]["NO_hold"])[int(R_read) - 1]) for tr in tot}), flush=True)
            json.dump({"script": "engine_vacflow_v2.py", "args": vars(a), "kappa_done": kappa, "klist": klist, "pockets": pockets, "running": running, "per_channel": per_channel,
                       "totals": {tr: {k_: v_.tolist() for k_, v_ in d.items()} for tr, d in tot.items()}}, open(path, "w"), default=float)
    # smoothing, the integrated energies, the log slopes
    Rout = np.geomspace(1.5 * rs, R_read, 24)
    summ = {}
    for tr in tot:
        summ[tr] = {}
        for key in ("raw_hold", "NO_hold", "raw_end", "NO_end"):
            cum = np.cumsum(tot[tr][key])
            E_R = [float(cum[int(r) - 1]) for r in Rout]
            sel = Rout >= R_read / 8
            c_fit = np.polyfit(np.log(Rout[sel]), np.array(E_R)[sel], 1)[0]
            sm = {}
            for ell in (4, 8, 16):
                rr = np.arange(0.0, R_read, ell / 4.0)
                K0, _ = FLD.kernels(rr, Rfull[:int(2 * R_read)], ell)
                sm[str(ell)] = {"r": rr.tolist(), "rho_ell": (K0 @ tot[tr][key][:int(2 * R_read)]).tolist()}
            summ[tr][key] = {"R_out": Rout.tolist(), "E_of_Rout": E_R, "log_slope_c": float(c_fit), "smoothed": sm}
        print("t_r = %s r_s: log slope c of E_raw %.4e (hold), of E_NO %.4e; E_NO(R_read) %.4e" % (tr, summ[tr]["raw_hold"]["log_slope_c"], summ[tr]["NO_hold"]["log_slope_c"],
              summ[tr]["NO_hold"]["E_of_Rout"][-1]), flush=True)
    json.dump({"script": "engine_vacflow_v2.py", "args": vars(a), "kappa_done": kmax, "klist": klist, "pockets": pockets, "running": running, "per_channel": per_channel, "summary": summ,
               "totals": {tr: {k_: v_.tolist() for k_, v_ in d.items()} for tr, d in tot.items()}, "seconds": time.time() - t_all}, open(path, "w"), default=float)


if __name__ == "__main__":
    main()
