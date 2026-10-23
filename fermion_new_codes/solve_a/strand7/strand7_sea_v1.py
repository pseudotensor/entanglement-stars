#!/usr/bin/env python
"""Strand 7 S7b, the probe sea's evolution and its produced quanta (SPEC_strand7_collapse.md sec. 3 S7b and sec. 1.6; builder analyst).
The transmission Gamma_j(omega) and the temperature fit are not in this script (they wait for the lead's re-specification of the packets);
this script writes the occupations they need.

Background: an S7a run's stored fields (strand7_shell_v1.py --save-fields 1: u = psi - 1, v = chi - 1 on S7a's grid at every output time),
interpolated in the external clock T by cubic splines through the stored times (held at the last stored fields beyond the last stored time,
stated per run), in r by cubic splines (zero slope at r = 0) inside S7a's R_max and by the exterior forms psi = 1 + A/r, chi = 1 - B/r beyond
it (A, B from the tails at that T).  L = N psi = chi.
Channels: the massless light packet (m = 0, one packet, eta = +1), per channel j the radial operator of strand5_w0fast_v1.channel_matrix on
the sea's grid (integer nodes r_i = i h and half nodes (i - 1/2) h, i = 1..n, R_max = n h; no cut for j <= 9/2):
A' = [[0, P], [Q, 0]] with P = diag(psi_int^-3) D^T and Q = -diag(psi_half^-3) D.  With m = 0 the chiral h = [[0, B_0 U^T], [U B_0, 0]] of the
spec (E_0 = (A'^T A')^(1/2), U = A' E_0^-1, B_0 = E_0^(1/2) L E_0^(1/2)) splits exactly into two sectors: alpha (the integer-node domain and
the half-node range, X = Q) and beta (the half-node domain and the integer-node range, X = P), each h_s = [[0, T^T], [T, 0]] with
T = X G^-1/4 L_dom G^1/4, G = X^T X (n x n), so that U_s = X G^-1/2 and B_0,s = G^1/4 L_dom G^1/4 (2n x 2n per sector instead of 4n x 4n).
Evolution: the sea at T = 0 is the n negative-energy eigenvectors of each sector's h_s(0); the clock is cut into refresh intervals of K
steps of delta_t (and at the output times), and each interval applies the exact exponential of h at the interval's midpoint,
Phi <- V e^(-i Lambda tau) V^T Phi (tau the interval's length; K steps with the same V and Lambda are this one exponential).
Output times: the positive-energy eigenvectors |+a> of h_s(T) (energies omega_a), n_a = sum_b |<+a|Phi_b>|^2, the mode's mean radius
<r>_a = sum_i r_i |<i|+a>|^2 (r_i the node's radius in its half); T_H(T) = kappa/(2 pi), kappa = 1/(2 r_s(T)), r_s(T) = 4 A(T); the exterior
modes (<r>_a > 2 r_s(T)) binned in omega with bins of 0.25 T_H (both sectors pooled; per bin the mode count, the summed and the mean
occupation), the near-surface modes (<r>_a < 2 r_s(T)) as a count and a total occupation.
Controls built here: --flat (the vacuum background psi = chi = 1: no production; C-S7b-2), --static-from (the background frozen from that T
on: the occupations of h's own modes then stay put; C-S7b-3 with --T-end beyond it), --K (C-S7b-4: K = 1 and 100 against 10 over one r_s),
--h (C-S7b-5), and at T = 0 per sector the frame identity V h V^T = sigma_3 x B_0 with V = (1/sqrt2)[[1, U^T], [1, -U^T]] (the exact part of
sec. 1.6's relation, which holds when U is orthogonal; C-S7b-1's connection term is not built here) and U's orthogonality.
Writes <out>/s7b_<tag>.json (the binned spectra and meta) and <out>/s7b_<tag>_modes.npz (omega, n_a, <r>_a per channel, sector, time)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "strand5"))
import strand5_w0fast_v1 as V1  # noqa: E402


class Background:
    def __init__(self, path, flat=False, static_from=None):
        z = np.load(path)
        self.r, self.T, self.u, self.v = z["r"], z["T"], z["u"], z["v"]
        self.Rmax = float(self.r[-1])
        self.flat, self.static_from = flat, static_from
        self.Tlast = float(self.T[-1])
        if len(self.T) >= 2:
            self.su = CubicSpline(self.T, self.u, axis=0)
            self.sv = CubicSpline(self.T, self.v, axis=0)

    def at(self, T, rr):
        """psi, chi at the radii rr and the tails A, B at the clock T."""
        if self.flat:
            return np.ones_like(rr), np.ones_like(rr), 0.0, 0.0
        Tc = min(T, self.Tlast)
        if self.static_from is not None:
            Tc = min(Tc, self.static_from)
        u = self.su(Tc) if len(self.T) >= 2 else self.u[0]
        v = self.sv(Tc) if len(self.T) >= 2 else self.v[0]
        A, B = self.Rmax * u[-1], -self.Rmax * v[-1]
        cu = CubicSpline(self.r, u, bc_type=((1, 0.0), "not-a-knot"))
        cv = CubicSpline(self.r, v, bc_type=((1, 0.0), "not-a-knot"))
        ins = rr <= self.Rmax
        psi = np.where(ins, 1 + cu(np.minimum(rr, self.Rmax)), 1 + A / rr)
        chi = np.where(ins, 1 + cv(np.minimum(rr, self.Rmax)), 1 - B / rr)
        return psi, chi, float(A), float(B)


class Channel:
    def __init__(self, j, n, h, dev):
        self.j, self.n, self.h, self.dev = j, n, h, dev
        self.r_int = h * np.arange(1, n + 1, dtype=float)
        self.r_half = h * (np.arange(n, dtype=float) + 0.5)

    def sectors(self, bg, T, parts=False):
        psi_i, chi_i, A, B = bg.at(T, self.r_int)
        psi_h, chi_h, _, _ = bg.at(T, self.r_half)
        Ap, k0 = V1.channel_matrix(self.r_int, self.r_half, psi_i, psi_h, 0.0, self.j, self.h)
        if k0 != 0:
            raise SystemExit("channel cut k0 = %d at j = %g (not expected for j <= 9/2)" % (k0, self.j))
        n = self.n
        P = torch.as_tensor(Ap[:n, n:], dtype=torch.float64, device=self.dev)
        Q = torch.as_tensor(Ap[n:, :n], dtype=torch.float64, device=self.dev)
        out, prt = {}, {}
        for name, X, Ldom in (("alpha", Q, chi_i), ("beta", P, chi_h)):
            G = X.T @ X
            w, Z = torch.linalg.eigh(G)
            Gm = (Z * w ** -0.25) @ Z.T
            Gp = (Z * w ** 0.25) @ Z.T
            L = torch.as_tensor(Ldom, dtype=torch.float64, device=self.dev)
            Tm = X @ Gm @ (L[:, None] * Gp)
            H = torch.zeros(2 * n, 2 * n, dtype=torch.float64, device=self.dev)
            H[:n, n:] = Tm.T
            H[n:, :n] = Tm
            out[name] = H
            if parts:
                prt[name] = (X @ ((Z * w ** -0.5) @ Z.T), Gp @ (L[:, None] * Gp))
        return (out, A, B, prt) if parts else (out, A, B)

    def radii(self, name):
        return np.concatenate([self.r_int, self.r_half]) if name == "alpha" else np.concatenate([self.r_half, self.r_int])


def schedule(T_end, dt, K, outs):
    """the refresh boundaries: multiples of K dt, with the output times inserted."""
    b = set(np.round(np.arange(0.0, T_end + 1e-9, K * dt), 12).tolist()) | set(np.round(outs, 12).tolist())
    b = sorted(x for x in b if x <= T_end + 1e-9)
    return b


def run_channel(a, bg, j, dev, outs, rs0):
    ch = Channel(j, a.n, a.h, dev)
    Hs, A0, B0, prt = ch.sectors(bg, 0.0, parts=True)
    Phi, rec, modes = {}, [], {}
    ident = {}
    for name, H in Hs.items():
        lam, V = torch.linalg.eigh(H)
        Phi[name] = V[:, :a.n].to(torch.complex128)
        U, B0m = prt[name]
        I = torch.eye(a.n, dtype=torch.float64, device=dev)
        Vf = torch.cat([torch.cat([I, U.T], 1), torch.cat([I, -U.T], 1)], 0) / math.sqrt(2)
        D = Vf @ H @ Vf.T
        Zb = torch.zeros_like(B0m)
        tgt = torch.cat([torch.cat([B0m, Zb], 1), torch.cat([Zb, -B0m], 1)], 0)
        ident[name] = {"frame_identity_rel": (torch.linalg.norm(D - tgt) / torch.linalg.norm(B0m)).item(),
                       "U_orthogonality": torch.linalg.norm(U.T @ U - I).item()}
    bounds = schedule(a.T_end, a.dt, a.K, outs)
    t0 = time.time()
    nref = 0
    for ta, tb in zip(bounds[:-1], bounds[1:]):
        tm = 0.5 * (ta + tb)
        Hs, _, _ = ch.sectors(bg, tm)
        for name, H in Hs.items():
            lam, V = torch.linalg.eigh(H)
            ph = torch.exp(-1j * lam * (tb - ta)).to(torch.complex128)
            Vc = V.to(torch.complex128)
            Phi[name] = Vc @ (ph[:, None] * (Vc.T @ Phi[name]))
        nref += 1
        if any(abs(tb - o) < 1e-9 for o in outs):
            Hs, A, B = ch.sectors(bg, tb)
            rs = 4 * A if A > 0 else rs0
            TH = 1.0 / (4 * math.pi * rs)
            row = {"j": j, "T": tb, "T_over_rs0": tb / rs0, "A": A, "B": B, "r_s": rs, "T_H": TH, "refreshes": nref, "seconds": time.time() - t0}
            om_all, na_all, rr_all = [], [], []
            for name, H in Hs.items():
                lam, V = torch.linalg.eigh(H)
                Vp = V[:, a.n:]
                ov = Vp.T.to(torch.complex128) @ Phi[name]
                na = torch.sum(torch.abs(ov) ** 2, dim=1).cpu().numpy()
                om = lam[a.n:].cpu().numpy()
                rr = (torch.as_tensor(ch.radii(name), dtype=torch.float64, device=dev)[:, None] * Vp ** 2).sum(0).cpu().numpy()
                norm = torch.linalg.norm(Phi[name].conj().T @ Phi[name] - torch.eye(a.n, dtype=torch.complex128, device=dev)).item()
                row["sea_orthonormality_" + name] = norm
                row["occupation_total_" + name] = float(na.sum())
                modes["j%g_%s_T%g" % (j, name, tb)] = np.stack([om, na, rr])
                om_all.append(om)
                na_all.append(na)
                rr_all.append(rr)
            om, na, rr = np.concatenate(om_all), np.concatenate(na_all), np.concatenate(rr_all)
            ext = rr > 2 * rs
            row["near_surface_count"] = int(np.sum(~ext))
            row["near_surface_occupation"] = float(np.sum(na[~ext]))
            edges = np.arange(0.0, a.bins_max_TH + 1e-9, a.bin_TH) * TH
            idx = np.digitize(om[ext], edges) - 1
            bins = []
            for b in range(len(edges) - 1):
                sel = idx == b
                bins.append({"omega_lo_TH": float(edges[b] / TH), "omega_hi_TH": float(edges[b + 1] / TH), "modes": int(np.sum(sel)),
                             "occupation_sum": float(np.sum(na[ext][sel])), "occupation_mean": float(np.mean(na[ext][sel])) if np.any(sel) else None})
            row["bins"] = bins
            row["exterior_occupation_total"] = float(np.sum(na[ext]))
            rec.append(row)
            print("j %g T/rs0 %.3g: exterior occupation %.6e, near-surface %.6e (%d modes), %.0f s" % (
                j, tb / rs0, row["exterior_occupation_total"], row["near_surface_occupation"], row["near_surface_count"], time.time() - t0), flush=True)
    return rec, modes, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--s7a-json", required=True)
    ap.add_argument("--channels", default="0.5,1.5,2.5")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--Rmax-over-rs", type=float, default=125.0)
    ap.add_argument("--dt-over-h", type=float, default=0.25)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--T-end-over-rs", type=float, default=40.0)
    ap.add_argument("--out-times-over-rs", default="5,10,20,30,40")
    ap.add_argument("--bin-TH", type=float, default=0.25)
    ap.add_argument("--bins-max-TH", type=float, default=10.0)
    ap.add_argument("--flat", type=int, default=0)
    ap.add_argument("--static-from-over-rs", type=float, default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    with open(a.s7a_json) as fh:
        S7 = json.load(fh)
    rs0 = float(S7["meta"]["r_s"])
    a.n = int(round(a.Rmax_over_rs * rs0 / a.h))
    a.dt = a.dt_over_h * a.h
    a.T_end = a.T_end_over_rs * rs0
    outs = [float(x) * rs0 for x in a.out_times_over_rs.split(",")]
    bg = Background(a.fields, flat=bool(a.flat), static_from=None if a.static_from_over_rs is None else a.static_from_over_rs * rs0)
    os.makedirs(a.out, exist_ok=True)
    rec = {"meta": dict(vars(a), r_s0=rs0, device_used=str(dev), background_last_T_over_rs=bg.Tlast / rs0, builder="analyst",
                        script="strand7_sea_v1.py"), "channels": {}}
    allmodes = {}
    t0 = time.time()
    for j in [float(x) for x in a.channels.split(",")]:
        r_, m_, id_ = run_channel(a, bg, j, dev, outs, rs0)
        rec["channels"][str(j)] = r_
        rec.setdefault("frame_identity_T0", {})[str(j)] = id_
        allmodes.update(m_)
    rec["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "s7b_%s.json" % a.tag), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(a.out, "s7b_%s_modes.npz" % a.tag), **allmodes)
    print("done %s %.0f s" % (a.tag, time.time() - t0))


if __name__ == "__main__":
    main()
