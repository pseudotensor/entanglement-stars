#!/usr/bin/env python
"""Strand 6 (i), spec (2) (SPEC_strand6_validation.md, the final3 section (i); builder analyst): the control rectangles by two routes.

Rectangles x = (a m)^2 = (0.01, 0.01 + H^2, 0.01 + H^2, 0.01 + 2 H^2), eta = (+, -, -, +), at H = 1, 0.5 (both routes) and 0.25 (the quadrature
route only; the spec's grid route is at H = 1 and 0.5).
Route 1 (the quadrature): strand6i_kernel_sign_v1.Rkern, F_R(x) per packet at the spec's ten R, grouped by eta.
Route 2 (the grid): strand 2c's packet routine (strand2c_kernel_v1.packet_terms, the Daleckii-Krein second order of W_0 = sum_s eta_s Tr L
(E_s1 - E_s0) in psi = 1 + eps cos(q.x) on the periodic L^3 box, twist 0) generalized from q along [100] to every lattice q: with
d_k = (sin k_1, sin k_2, sin k_3), s_k^2 = |d_k|^2 and k' = k +- q, X'_kk' = c_kk' [a0 + i a.sigma], a0 = -2 m^2 - 6 d_k.d_k',
|a|^2 = 16 m^2 |d_k - d_k'|^2 + 36 |d_k x d_k'|^2 (the cross term vanishes: (d_k - d_k') is orthogonal to d_k x d_k'), X~'_kk' = c_kk'
[b0 + i b.sigma], b0 = -2 m^2 - 3 (s_k^2 + s_k'^2), |b|^2 = 4 m^2 |d_k - d_k'|^2, and X'', X~'' as in strand2c_kernel_v1 (sum_k' c_kk'^2 =
1/2 and (c^2)_kk = 1/2 for a generic q; 1 and 1 when 2q = 0 mod 2 pi in every component, the two neighbours then one); the [100] reduction
of strand2c_kernel_v1 is these forms at q = (q_1, 0, 0) (|d_k - d_k'|^2 = (s_1 - s_1')^2, |d_k x d_k'|^2 = s_perp^2 (s_1 - s_1')^2), checked
here against packet_terms itself at three q.  Per packet F_s(q) = -(U_s'' - T_s'')/L^3 for a generic q (2c's k_a q_lat^2 = -2 W^(2)/L^3 with
W^(2) = (1/2) sum eta_s (U_s'' - T_s'')) and -(U_s'' - T_s'')/(2 L^3) when 2q = 0 (the cosine mode's norm L^3 instead of L^3/2; strand5_kernel_v2's
zone-edge halving); computed on the wedge 0 <= n_3 <= n_2 <= n_1 <= L/2 (cubic symmetry and q -> -q), filled to the full grid, and
F_R^(L) = (1/L^3) sum_q F(q) e^(iq.R) by the inverse FFT (F(0) the routine's own uniform-mode value, reported); L = 64 and 128 (the difference
is the finite-size error: the light packet m = 0.1 wraps as e^(-m L)).  On a GPU (torch, float64; --device) or the CPU.
Checks: the generalized routine against packet_terms at q = (1, 0, 0), (5, 0, 0), (L/2, 0, 0) (T'', U'', W11, relative); sum_R F_R^(L) against
F(0).  Writes --out-json and the markdown table I2 (--md): per H, R and packet the two routes, their relative difference, and L = 64 against
128.  Numbers emitted, not retyped."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand6i_kernel_sign_v1 as QR  # noqa: E402
import strand2c_kernel_v1 as K2C  # noqa: E402


def wedge(L):
    h = L // 2
    return [(a, b, c) for a in range(h + 1) for b in range(a + 1) for c in range(b + 1)]


class GridPacket:
    def __init__(self, m, L, dev):
        self.m, self.L, self.dev = m, L, dev
        self.s = torch.tensor(np.sin(2 * np.pi * np.arange(L) / L), dtype=torch.float64, device=dev)
        s2 = self.s ** 2
        self.SS = s2.view(L, 1, 1) + s2.view(1, L, 1) + s2.view(1, 1, L)
        self.sq = torch.sqrt(m * m + self.SS)
        self.Fp = 0.5 / self.sq
        m2 = m * m
        self.FpA = float(torch.sum(self.Fp * 4 * (m2 + 9 * self.SS)))          # sum_k F'(x_k) 2 * 2 (m^2 + 9 s_k^2)
        self.FpB = float(torch.sum(self.Fp * 4 * (2 * m2 + 12 * self.SS)))     # sum_k F'(x_k) 2 * 2 (2 m^2 + 12 s_k^2)

    def terms(self, n):
        L, m2, s, SS, sq, Fp = self.L, self.m * self.m, self.s, self.SS, self.sq, self.Fp
        zero2 = all((2 * c) % L == 0 for c in n)
        nbs = [(tuple(c % L for c in n), 1.0)] if zero2 else [(tuple(c % L for c in n), 0.5), (tuple((-c) % L for c in n), 0.5)]
        sum_c2 = sum(w * w for _, w in nbs)
        c2d = 1.0 if zero2 else 0.5
        T2 = torch.tensor(sum_c2 * self.FpA + c2d * self.FpB, dtype=torch.float64, device=self.dev)
        Ut = torch.zeros_like(SS)
        parts = []
        for sh, w in nbs:
            sp = [torch.roll(s, -sh[j]) for j in range(3)]
            SSp = (sp[0] ** 2).view(L, 1, 1) + (sp[1] ** 2).view(1, L, 1) + (sp[2] ** 2).view(1, 1, L)
            Ut = Ut + w * w * (m2 + 9 * SSp)
            parts.append((sp, SSp, w))
        U2 = torch.sum(Fp * 2 * (2 * (Ut + c2d * (2 * m2 + 12 * SS))))
        W11 = torch.zeros((), dtype=torch.float64, device=self.dev)
        for sp, SSp, w in parts:
            sqp = torch.sqrt(m2 + SSp)
            F1 = 1.0 / (sq + sqp)
            Fp1 = -1.0 / (2 * sq * sqp * (sq + sqp))
            dd = (s * sp[0]).view(L, 1, 1) + (s * sp[1]).view(1, L, 1) + (s * sp[2]).view(1, 1, L)
            dm = ((s - sp[0]) ** 2).view(L, 1, 1) + ((s - sp[1]) ** 2).view(1, L, 1) + ((s - sp[2]) ** 2).view(1, 1, L)
            a0 = -2 * m2 - 6 * dd
            a2 = 16 * m2 * dm + 36 * (SS * SSp - dd * dd)
            trX = 2 * w * a0
            T2 = T2 + torch.sum(2 * F1 * w * trX + Fp1 * (2 * w * w * (a0 * a0 + a2)))
            b0 = -2 * m2 - 3 * (SS + SSp)
            trXt = 2 * w * b0
            U2 = U2 + torch.sum(2 * F1 * w * trXt + Fp1 * (2 * w * w * (b0 * b0 + 4 * m2 * dm)))
            W11 = W11 + torch.sum(F1 * w * (trXt - trX))
        return float(T2), float(U2), float(W11), zero2


def grid_FR(m, L, dev, Rs):
    """per packet: F_R^(L) at the given R by the inverse FFT of F(q) on the full grid, F(0), and sum_R F_R against F(0)."""
    P = GridPacket(m, L, dev)
    h = L // 2
    Wd = np.zeros((h + 1, h + 1, h + 1))
    for n in wedge(L):
        T2, U2, _, z2 = P.terms(n)
        Wd[n] = -(U2 - T2) / (L ** 3) / (2.0 if z2 else 1.0)
    idx = np.arange(L)
    f = np.minimum(idx, L - idx)
    A, B, C = np.meshgrid(f, f, f, indexing="ij")
    S = np.sort(np.stack([A, B, C], axis=-1), axis=-1)[..., ::-1]
    Ff = Wd[S[..., 0], S[..., 1], S[..., 2]]
    FR = np.fft.ifftn(Ff)
    out = {str(R): float(FR[R[0] % L, R[1] % L, R[2] % L].real) for R in Rs}
    return out, float(Wd[0, 0, 0]), float(FR.real.sum()), float(np.max(np.abs(FR.imag)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--Ls", default="64,128")
    ap.add_argument("--Hs-grid", default="1,0.5")
    ap.add_argument("--Hs-quad", default="1,0.5,0.25")
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    t0 = time.time()
    etas = (1, -1, -1, 1)
    Rs = QR.TEN_R
    rec = {"meta": dict(vars(a), device_used=str(dev)), "checks": {}, "H": {}}
    # check: the generalized routine against strand2c_kernel_v1.packet_terms along [100]
    chk = []
    L = 64
    sp_, cnt_ = K2C.perp_classes(L, 0.0, 0.0)
    for m in (0.1, float(np.sqrt(1.01))):
        P = GridPacket(m, L, dev)
        for nq in (1, 5, L // 2):
            t2, u2, w11 = K2C.packet_terms(m, L, nq, (0.0, 0.0, 0.0), sp_, cnt_)
            T2, U2, W11, _ = P.terms((nq, 0, 0))
            chk.append({"m": m, "n_q": nq, "T2_rel": T2 / t2 - 1, "U2_rel": U2 / u2 - 1, "W11_rel": (W11 / w11 - 1) if w11 != 0 else None,
                        "U_minus_T_rel": (U2 - T2) / (u2 - t2) - 1})
    rec["checks"]["general_vs_100"] = chk
    print("check done %.0f s" % (time.time() - t0), flush=True)
    Hq = [float(x) for x in a.Hs_quad.split(",")]
    Hg = [float(x) for x in a.Hs_grid.split(",")]
    Ls = [int(x) for x in a.Ls.split(",")]
    K10 = {R: QR.Rkern(R) for R in Rs}
    print("quadrature kernels done %.0f s" % (time.time() - t0), flush=True)
    for H in Hq:
        xs = (0.01, 0.01 + H * H, 0.01 + H * H, 0.01 + 2 * H * H)
        quad = {str(R): [K10[R].F(x) for x in xs] for R in Rs}
        hrec = {"x": xs, "quad": quad, "grid": {}}
        if H in Hg:
            for L in Ls:
                per, F0s, sums, imags = [], [], [], []
                for x in (xs[0], xs[1], xs[3]):
                    FR, F0, sm, im = grid_FR(float(np.sqrt(x)), L, dev, Rs)
                    per.append(FR)
                    F0s.append(F0)
                    sums.append(sm)
                    imags.append(im)
                    print("H %g L %d m %.6g done %.0f s" % (H, L, np.sqrt(x), time.time() - t0), flush=True)
                g = {str(R): [per[0][str(R)], per[1][str(R)], per[1][str(R)], per[2][str(R)]] for R in Rs}
                hrec["grid"][str(L)] = {"per_packet": g, "F0_per_packet": [F0s[0], F0s[1], F0s[1], F0s[2]], "sum_R_FR": [sums[0], sums[1], sums[1], sums[2]],
                                        "max_imag": max(imags)}
        rec["H"][str(H)] = hrec
    rec["meta"]["seconds"] = time.time() - t0
    with open(a.out_json, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    # tables
    def grp(v):
        return sum(e * p for e, p in zip(etas, v))
    L_ = ["# Strand 6 (i), spec (2): the control rectangles by two routes (strand6i_kernel_fft_v1.py)", "",
          "## I2a The generalized grid routine against strand2c_kernel_v1.packet_terms along [100] (L = 64; relative differences)", "",
          QR.table(["m", "n_q", "T''", "U''", "W11", "U'' - T''"], [["%.8g" % c["m"], str(c["n_q"]), "%.2e" % c["T2_rel"], "%.2e" % c["U2_rel"],
                                                                 "%.2e" % c["W11_rel"] if c["W11_rel"] is not None else "-", "%.2e" % c["U_minus_T_rel"]] for c in chk]), ""]
    rows = []
    for H, hrec in rec["H"].items():
        for R in Rs:
            q = hrec["quad"][str(R)]
            row = [H, str(R), "%.10e" % grp(q)]
            for L in Ls:
                gr = hrec["grid"].get(str(L))
                if gr is None:
                    row += ["-", "-"]
                    continue
                gv = grp(gr["per_packet"][str(R)])
                row += ["%.10e" % gv, "%.2e" % (gv / grp(q) - 1) if grp(q) != 0 else "-"]
            rows.append(row)
    head = ["H", "R", "grouped (quadrature)"]
    for L in Ls:
        head += ["grouped (grid L = %d)" % L, "grid/quad - 1 (L = %d)" % L]
    L_ += ["## I2b The grouped F_R at the control rectangles: the quadrature (infinite volume) against the grid route", "", QR.table(head, rows), ""]
    rows = []
    for H, hrec in rec["H"].items():
        for R in Rs:
            for i, x in enumerate(hrec["x"]):
                if i == 2:
                    continue
                q = hrec["quad"][str(R)][i]
                row = [H, str(R), "%.6g" % x, "%.10e" % q]
                for L in Ls:
                    gr = hrec["grid"].get(str(L))
                    if gr is None:
                        row += ["-", "-"]
                        continue
                    gv = gr["per_packet"][str(R)][i]
                    row += ["%.10e" % gv, "%.2e" % (gv / q - 1) if q != 0 else ("%.2e" % gv)]
                rows.append(row)
    head = ["H", "R", "x", "per packet (quadrature)"]
    for L in Ls:
        head += ["grid L = %d" % L, "grid/quad - 1 (L = %d)" % L]
    L_ += ["## I2c Per packet (x = 0.01, 0.01 + H^2, 0.01 + 2 H^2)", "", QR.table(head, rows), ""]
    rows = []
    for H, hrec in rec["H"].items():
        for L, gr in hrec["grid"].items():
            rows.append([H, L, " ".join("%.6e" % v for v in gr["F0_per_packet"]), "%.6e" % grp(gr["F0_per_packet"]),
                         " ".join("%.2e" % (s_ - f0) for s_, f0 in zip(gr["sum_R_FR"], gr["F0_per_packet"])), "%.1e" % gr["max_imag"]])
    L_ += ["## I2d The grid route's F(0) (the uniform mode, per packet and grouped) and sum_R F_R^(L) - F(0); the largest imaginary part of the inverse FFT", "",
           QR.table(["H", "L", "F(0) per packet", "F(0) grouped", "sum_R F_R - F(0) per packet", "max |Im|"], rows), "",
           "Seconds: %.0f (device %s)." % (rec["meta"]["seconds"], rec["meta"]["device_used"]), ""]
    with open(a.md, "w") as fh:
        fh.write("\n".join(L_) + "\n")
    print("done %.0f s" % rec["meta"]["seconds"])


if __name__ == "__main__":
    main()
