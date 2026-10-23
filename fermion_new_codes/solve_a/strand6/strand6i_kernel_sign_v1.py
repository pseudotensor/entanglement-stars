#!/usr/bin/env python
"""Strand 6 (i) (SPEC_strand6_validation.md, the final3 section (i); builder analyst): the grouped finite-sine kernel's sign at even and odd
lattice separations, by the lead's exact infinite-volume representation (LEAD.tex, "The lane under final3", (i); GAP_stability (77) to (79)
with the odd separations added).

Per packet with x = (a m)^2 and a lattice displacement R = (R_1, R_2, R_3):
  h_R(u) = e^(-u/2) I_(R/2)(u/2) for even R, 0 for odd R;  sigma_R(u) = e^(-u/2) [I_((R-1)/2)(u/2) - I_((R+1)/2)(u/2)]/2 for odd R
  (scipy.special.ive; |R| used for negative components);
  every component even:   F_R(x) = -(6/sqrt pi) Int_0^inf dS S^(-3/2) e^(-xS) C_R'(S),  C_R(S) = Int_0^S H_R(u) H_R(S - u) du,
                          H_R = prod_j h_(R_j); evaluated in the integrated-by-parts form
                          F_R(x) = -(6/sqrt pi) Int_0^inf dS (3/2 S^(-5/2) + x S^(-3/2)) e^(-xS) C_R(S)
                          (the boundary terms vanish for R != 0: C_R ~ S^(|R|_1 + 1) at small S and ~ S^(-3/2) at large S), the derivative
                          form beside at two R as a check;
  exactly one component odd (i):  F_R(x) = -(6/sqrt pi) Int_0^inf dS S^(-3/2) e^(-xS) G_R(S),
                          G_R(S) = Int_0^S sigma_(R_i)(u) sigma_(R_i)(S - u) prod_(j != i) h_(R_j)(u) h_(R_j)(S - u) du;
  two or three odd components: F_R = 0;
  grouped: F_R^group = sum_s eta_s F_R(x_s) (eta = +, -, -, +).
Quadrature: S on log-spaced panels from 1e-10 to 1e8 (16 Gauss-Legendre nodes per panel, 12 panels per decade), the head below 1e-10 by the
power law of the first two nodes (C_R ~ S^(|R|_1 + 1), G_R ~ S^(|R|_1) at small S; the head matters only for |R|_1 = 1, where the integrand is
S^(-1/2)), the tail beyond 1e8 by the asymptotic S^(-3/2) form of C_R and G_R (negligible); for each S the u-integral by symmetry over [0, S/2] (the integrands are symmetric
under u -> S - u) on panels graded geometrically from min(S/2, 1e-3) with 16 nodes each; 40-digit mpmath spot checks of the u-integral at two
(R, S) pairs.
Jobs (spec (1), (3), (4)): (1) F_R per packet and grouped at the ten R of the spec for the diagnostic assignment x = (0, mu^2, mu^2, 2 mu^2),
mu = 0.0130903478, and the window edge (mu scaled by 0.0055/0.0213); F_0 = -sum_(R != 0, |R| <= 12) F_R (by octahedral classes, with the
truncation's change from |R| <= 10); (3) the ratios F_R^group/F_0 and the sign-change scan of F_(2,0,0)^group over 16 geometric mu from 0.5
down to the window edge; (4) every per-packet F_R(x_s) < 0 at the ten R (all masses used), and the 1D projection: the [100] line kernel
K_n = -sum over the transverse components of F_(n, R_2, R_3), the transverse sums done exactly inside the S-representation by Parseval
(sum_R h_R(u) h_R(S - u) = h_0(S), sum_(R odd) sigma_R(u) sigma_R(S - u) = -h_0'(S)), against strand 2c's stored per-packet symbols (J2b's
packets_A_m*_L256.json: the line kernel built from k(m; q) on the 256-ring with the zone-edge entry halved, as the rows build it) at the control
masses 0.1 and the heavies at H = 1 and 0.5 (n = 1..8).  The spec's 3D FFT route of (2) is not in this script.  Numbers emitted, not retyped."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import itertools
import json
import math

import mpmath as mp
import numpy as np
from scipy.special import ive

XG, WG = np.polynomial.legendre.leggauss(16)
PREF = 6.0 / math.sqrt(math.pi)
TEN_R = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (2, 2, 0), (2, 2, 2), (1, 2, 0), (1, 2, 2), (4, 2, 0), (6, 0, 0)]
MU = 0.0130903478


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def h_even(R, u):
    return ive(abs(R) / 2.0, u / 2.0)


def sig_odd(R, u):
    R = abs(R)
    return 0.5 * (ive((R - 1) / 2.0, u / 2.0) - ive((R + 1) / 2.0, u / 2.0))


def S_nodes(smin=1e-10, smax=1e8, per_decade=12):
    edges = np.geomspace(smin, smax, int(round(math.log10(smax / smin) * per_decade)) + 1)
    s = (0.5 * (edges[1:] - edges[:-1])[:, None] * XG[None, :] + 0.5 * (edges[1:] + edges[:-1])[:, None]).ravel()
    w = (0.5 * (edges[1:] - edges[:-1])[:, None] * WG[None, :]).ravel()
    return s, w


def u_nodes(S, umin=1e-3, per_decade=8):
    """Gauss-Legendre nodes on [0, S/2], panels graded geometrically from min(S/2, umin)."""
    half = 0.5 * S
    lo = min(half, umin)
    if half <= lo * 1.000001:
        edges = np.array([0.0, half])
    else:
        edges = np.concatenate([[0.0], np.geomspace(lo, half, max(2, int(math.ceil(math.log10(half / lo) * per_decade))) + 1)])
    u = (0.5 * (edges[1:] - edges[:-1])[:, None] * XG[None, :] + 0.5 * (edges[1:] + edges[:-1])[:, None]).ravel()
    w = (0.5 * (edges[1:] - edges[:-1])[:, None] * WG[None, :]).ravel()
    return u, w


def kernel_function(R, S):
    """C_R(S) (all even) or G_R(S) (one odd); None for two or three odd components."""
    odd = [i for i, c in enumerate(R) if c % 2]
    if len(odd) >= 2:
        return None
    u, w = u_nodes(S)
    v = S - u
    if not odd:
        Hu = np.prod([h_even(c, u) for c in R], axis=0)
        Hv = np.prod([h_even(c, v) for c in R], axis=0)
        return 2.0 * float(np.sum(w * Hu * Hv))
    i = odd[0]
    prod = sig_odd(R[i], u) * sig_odd(R[i], v)
    for j, c in enumerate(R):
        if j != i:
            prod = prod * h_even(c, u) * h_even(c, v)
    return 2.0 * float(np.sum(w * prod))


class Rkern:
    """F_R(x) for one R at any x, from C_R or G_R tabulated on the S nodes once."""

    def __init__(self, R):
        self.R = tuple(abs(c) for c in R)
        self.odd = sum(c % 2 for c in self.R)
        self.s, self.w = S_nodes()
        if self.odd >= 2:
            self.vals = None
            return
        self.vals = np.array([kernel_function(self.R, s_) for s_ in self.s])
        # the asymptotic tail beyond the last node: vals ~ c S^(-3/2); the head below the first node: vals ~ c0 S^p
        self.c_tail = float(self.vals[-1] * self.s[-1] ** 1.5)
        v0, v1 = self.vals[0], self.vals[1]
        self.p_head = float(math.log(v1 / v0) / math.log(self.s[1] / self.s[0])) if v0 > 0 and v1 > 0 else None
        self.c_head = float(v0 / self.s[0] ** self.p_head) if self.p_head is not None else 0.0

    def F(self, x):
        if self.vals is None:
            return 0.0
        s, w, v = self.s, self.w, self.vals
        e = np.exp(-x * s)
        smax = s[-1]
        s0 = s[0]
        if self.odd == 0:
            integ = float(np.sum(w * (1.5 * s ** -2.5 + x * s ** -1.5) * e * v))
            tail = self.c_tail * (0.5 * smax ** -3) if x == 0 else 0.0
            head = 1.5 * self.c_head * s0 ** (self.p_head - 1.5) / (self.p_head - 1.5) if self.p_head is not None and self.p_head > 1.5 else 0.0
        else:
            integ = float(np.sum(w * s ** -1.5 * e * v))
            tail = self.c_tail * smax ** -2 / 2 if x == 0 else 0.0
            head = self.c_head * s0 ** (self.p_head - 0.5) / (self.p_head - 0.5) if self.p_head is not None and self.p_head > 0.5 else 0.0
        return -PREF * (integ + tail + head)

    def F_derivform(self, x):
        """the derivative form for the even case: -(6/sqrt pi) Int S^(-3/2) e^(-xS) C_R'(S), C_R' by a centred difference on the nodes."""
        if self.vals is None or self.odd:
            return None
        ls = np.log(self.s)
        dC = np.gradient(self.vals, ls) / self.s
        return -PREF * float(np.sum(self.w * self.s ** -1.5 * np.exp(-x * self.s) * dC))


def mp_check(R, S):
    R = tuple(abs(c) for c in R)
    odd = [i for i, c in enumerate(R) if c % 2]

    def hh(c, u):
        z = mp.mpf(u) / 2
        return mp.exp(-z) * mp.besseli(mp.mpf(c) / 2, z)

    def ss(c, u):
        z = mp.mpf(u) / 2
        return mp.exp(-z) * (mp.besseli(mp.mpf(c - 1) / 2, z) - mp.besseli(mp.mpf(c + 1) / 2, z)) / 2

    def integrand(u):
        v = mp.mpf(S) - u
        if not odd:
            return mp.fprod(hh(c, u) for c in R) * mp.fprod(hh(c, v) for c in R)
        i = odd[0]
        p = ss(R[i], u) * ss(R[i], v)
        for j, c in enumerate(R):
            if j != i:
                p *= hh(c, u) * hh(c, v)
        return p
    mp.mp.dps = 40
    return float(2 * mp.quad(integrand, [0, mp.mpf(S) / 4, mp.mpf(S) / 2]))


def line_kernel_quad(n, x):
    """the [100] line kernel of one packet, K_n = -sum_(R_2, R_3) F_(n, R_2, R_3), n >= 1, with the transverse sums done inside the S-representation
    by Parseval: sum_R h_R(u) h_R(S - u) = h_0(S) and sum_(R odd) sigma_R(u) sigma_R(S - u) = -h_0'(S) = (h_0(S) - e^(-S/2) I_1(S/2))/2."""
    s, w = S_nodes()
    h0 = ive(0, s / 2)
    mh0p = 0.5 * (ive(0, s / 2) - ive(1, s / 2))
    Cl = np.zeros_like(s)
    Gl = np.zeros_like(s)
    for i, S in enumerate(s):
        u, wu = u_nodes(S)
        v = S - u
        if n % 2 == 0:
            Ihh = 2.0 * float(np.sum(wu * h_even(n, u) * h_even(n, v)))
            Cl[i] = h0[i] ** 2 * Ihh
            Gl[i] = 2.0 * mh0p[i] * h0[i] * Ihh
        else:
            Iss = 2.0 * float(np.sum(wu * sig_odd(n, u) * sig_odd(n, v)))
            Gl[i] = h0[i] ** 2 * Iss
    e = np.exp(-x * s)
    Fl = -PREF * (float(np.sum(w * (1.5 * s ** -2.5 + x * s ** -1.5) * e * Cl)) + float(np.sum(w * s ** -1.5 * e * Gl)))
    return -Fl


def stored_line_kernel(path, nmax):
    """strand 2c's per-packet line kernel from its stored symbol k(m; q) on the L-ring (the zone-edge entry halved), as the rows build it."""
    P = json.load(open(path))
    L = int(P["L"])
    rows = sorted(P["rows"], key=lambda r_: r_["n"])
    q = np.array([r_["q"] for r_ in rows])
    k = np.array([r_["k"] for r_ in rows])
    k[-1] *= 0.5
    C = -k * 2 * (1 - np.cos(q))
    out = []
    for n in range(1, nmax + 1):
        out.append((2 * np.sum(C[:-1] * np.cos(q[:-1] * n)) + C[-1] * np.cos(q[-1] * n)) / L)
    return np.array(out), float(P["m"])


def classes(rmax):
    out = {}
    for a in range(0, rmax + 1):
        for b in range(0, a + 1):
            for c in range(0, b + 1):
                if (a, b, c) != (0, 0, 0) and a * a + b * b + c * c <= rmax * rmax:
                    mult = len(set(itertools.permutations((a, b, c)))) * 2 ** sum(1 for t in (a, b, c) if t)
                    out[(a, b, c)] = mult
    return out


def grouped(F, xs, etas):
    return sum(e * F(x) for x, e in zip(xs, etas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--rmax", type=int, default=12)
    ap.add_argument("--packets-dir", default="<internal>/J2b_run1", help="strand 2c's per-packet symbol files (J2b)")
    a = ap.parse_args()
    etas = (1, -1, -1, 1)
    mu_e = MU * 0.0055 / 0.0213
    assign = {"diagnostic": (0.0, MU ** 2, MU ** 2, 2 * MU ** 2), "window edge": (0.0, mu_e ** 2, mu_e ** 2, 2 * mu_e ** 2)}
    summ, rec = [], {"assignments": assign}
    out = ["# Strand 6 (i): the grouped finite-sine kernel's sign at lattice separations (generated by solve/code/strand6/strand6i_kernel_sign_v1.py)", ""]
    # the ten R
    K10 = {R: Rkern(R) for R in TEN_R}
    # F_0 by the zero-sum rule over octahedral classes
    cl = classes(a.rmax)
    Kc = {}
    for R in cl:
        Kc[R] = K10[R] if R in K10 else Rkern(R)
    F0 = {}
    for name, xs in assign.items():
        tot12 = sum(m * grouped(Kc[R].F, xs, etas) for R, m in cl.items())
        tot10 = sum(m * grouped(Kc[R].F, xs, etas) for R, m in cl.items() if R[0] ** 2 + R[1] ** 2 + R[2] ** 2 <= 100)
        F0[name] = (-tot12, -tot10)
    rows1 = []
    for name, xs in assign.items():
        for R in TEN_R:
            per = [K10[R].F(x) for x in xs]
            g = sum(e * p for e, p in zip(etas, per))
            rows1.append([name, str(R), " ".join("%.6e" % p for p in per), "%.6e" % g, "+" if g > 0 else ("-" if g < 0 else "0"), "%.5e" % (g / F0[name][0])])
        pos = [str(R) for R in TEN_R if sum(e * K10[R].F(x) for x, e in zip(xs, etas)) > 0]
        summ.append("I1 (%s, mu = %s): F_0 = %s (|R| <= %d; %s at |R| <= 10); the grouped F_R is positive (the M-matrix sign fails) at %s and negative or zero "
                    "at the rest of the ten R." % (name, "%.6e" % math.sqrt(xs[1]), "%.6e" % F0[name][0], a.rmax, "%.6e" % F0[name][1], ", ".join(pos) if pos else "none"))
    # (3) the scan of F_(2,0,0)^group
    scan = []
    for mu_ in np.geomspace(0.5, mu_e, 16):
        xs = (0.0, mu_ ** 2, mu_ ** 2, 2 * mu_ ** 2)
        scan.append((float(mu_), grouped(K10[(2, 0, 0)].F, xs, etas), grouped(K10[(1, 0, 0)].F, xs, etas)))
    ch = [(scan[i][0], scan[i + 1][0]) for i in range(len(scan) - 1) if scan[i][1] * scan[i + 1][1] < 0]
    summ.append("I3: F_(2,0,0)^group over mu from 0.5 to the window edge (16 geometric points) changes sign between %s; F_(1,0,0)^group's signs %s." % (
        "; ".join("%.5g and %.5g" % c for c in ch) if ch else "no adjacent pair (no sign change)", "".join("+" if s_[2] > 0 else "-" for s_ in scan)))
    # (4) per-packet negativity and the checks
    allneg = all(K10[R].F(x) <= 0 for R in TEN_R for xs in assign.values() for x in xs)
    chk = []
    for R, S in (((2, 0, 0), 3.0), ((1, 2, 0), 40.0)):
        q = kernel_function(R, S)
        m_ = mp_check(R, S)
        chk.append([str(R), "%g" % S, "%.15e" % q, "%.15e" % m_, f(q / m_ - 1, 2)])
    dchk = [[str(R), "%.10e" % K10[R].F(MU ** 2), "%.10e" % K10[R].F_derivform(MU ** 2), f(K10[R].F_derivform(MU ** 2) / K10[R].F(MU ** 2) - 1, 2)]
            for R in ((2, 0, 0), (2, 2, 0))]
    summ.append("I4: every per-packet F_R(x_s) at the ten R and both assignments is negative or zero: %s; the u-integral against 40 digits: %s; the "
                "integrated-by-parts form against the derivative form (centred differences on the S nodes): %s." % (
                    "yes" if allneg else "NO", "; ".join("%s at S = %s: %s" % (c[0], c[1], c[4]) for c in chk), "; ".join("%s: %s" % (c[0], c[3]) for c in dchk)))
    # (4) the [100] line kernel against strand 2c's stored per-packet symbols (control masses: 0.1 and the heavies at H = 1, 0.5)
    lrows = []
    if a.packets_dir:
        for mname in ("0.1", "1.0049876", "1.4177447", "0.50990195", "0.71414284"):
            pth = os.path.join(a.packets_dir, "packets_A_m%s_L256.json" % mname)
            if not os.path.exists(pth):
                continue
            Ks, mm = stored_line_kernel(pth, 8)
            Kq = np.array([line_kernel_quad(n, mm * mm) for n in range(1, 9)])
            for n in range(1, 9):
                lrows.append(["%.8g" % mm, n, "%.12e" % Ks[n - 1], "%.12e" % Kq[n - 1], "%.6e" % (Ks[n - 1] / Kq[n - 1]) if Kq[n - 1] != 0 else "-"])
        if lrows:
            rat = [float(r_[4]) for r_ in lrows if r_[4] != "-" and abs(float(r_[2])) > 1e-14]
            summ.append("I4 (line kernel): strand 2c's stored per-packet [100] line kernel over the quadrature's (transverse sums by Parseval) at n = 1..8 and "
                        "masses %s: ratios from %s to %s." % (", ".join(sorted({r_[0] for r_ in lrows})), f(min(rat), 8) if rat else "-", f(max(rat), 8) if rat else "-"))
    out += ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""]
    out += ["## I1 The production assignments: per R the per-packet F_R(x_s) (packets in the order x_0 = 0, mu^2, mu^2, 2 mu^2), the grouped value, its sign and "
            "its ratio to F_0", "", table(["assignment", "R", "per packet", "grouped", "sign", "grouped/F_0"], rows1), "",
            "## I3 The scan: F_(2,0,0)^group and F_(1,0,0)^group against mu (the rectangle (0, mu^2, mu^2, 2 mu^2))", "",
            table(["mu", "F_(2,0,0)^group", "F_(1,0,0)^group"], [["%.6g" % s_[0], "%.6e" % s_[1], "%.6e" % s_[2]] for s_ in scan]), "",
            "## I4 Checks: the u-integral against 40-digit quadrature; the two forms of the even case", "",
            table(["R", "S", "u-integral", "40 digits", "relative difference"], chk), "",
            table(["R", "F_R(mu^2) by parts", "derivative form", "relative difference"], dchk), "",
            "## I4b The [100] line kernel per packet: strand 2c's stored symbol's line kernel (L = 256, the zone edge halved) against the quadrature's "
            "-sum_(R_2, R_3) F_(n, R_2, R_3) (transverse sums by Parseval inside the S-representation)", "",
            table(["m", "n", "stored (2c, L = 256)", "quadrature (infinite volume)", "ratio"], lrows), ""]
    rec["F0"] = F0
    rec["scan"] = scan
    rec["ten"] = {str(R): {name: [K10[R].F(x) for x in xs] for name, xs in assign.items()} for R in TEN_R}
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
