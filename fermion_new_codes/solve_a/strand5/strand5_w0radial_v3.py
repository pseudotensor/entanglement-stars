#!/usr/bin/env python
"""Strand 5 J5 (spec solve/specs/SPEC_strand5_kernel_star.md sec. 4 J5, 10(p), 10(p'); builder analyst): the nonlinear vacuum energy
W_0 = sum_s eta_s Tr (N psi)(E_s1 - E_s0) on a kernel-level star by the radial partial-wave spectral evaluation of the continuum polar
operator, against the same operator's quadratic form.

Operator (the lattice polar operator's placement, strand2/strand0_lib.py and strand2c_kernel_v1.packet_terms: psi^-1 on the mass, psi^-3
to the LEFT of the gradient): A_s = m_s psi^-1 + i psi^-3 sigma . grad on one two-component cone; E_s0 = (A^dag A)^(1/2), E_s1 = (A A^dag)^(1/2).
The lattice's eight valleys: every W and k is reported per cone and times 8.  Masses (m_l, (m_l^2 + aH^2)^(1/2) twice, (m_l^2 + 2 aH^2)^(1/2)),
orientations (+, -, -, +).
Partial waves: for j = 1/2, 3/2, ... the pair kappa = -(j + 1/2) (u, on the integer nodes r_i = i h, i = 1..n; u_0 = 0) and +(j + 1/2)
(v, on the half nodes (i - 1/2) h).  With sigma . grad [(u/r) Omega_kappa] = -(1/r)(u' + kappa u/r) Omega_-kappa and the diagonal unitary
diag(1, i), the radial operator is the real 2n x 2n matrix
    A' = [[diag(m psi_int^-1), -diag(psi_int^-3) D^T], [-diag(psi_half^-3) D, diag(m psi_half^-1)]],
    (D u)_{i-1/2} = (u_i - u_{i-1})/h + kappa (u_i + u_{i-1})/(2 r_{i-1/2}),   kappa = -(j + 1/2),
exactly symmetric (normal) at psi = 1 (no radial doublers: one first-order difference between staggered nodes).  Per (packet, j) the
symmetric chiral matrix H = [[0, A'], [A'^T, 0]] is diagonalized (torch.linalg.eigh, float64, GPU): its positive eigenpairs are
(sigma_k, (u_k, v_k)/sqrt 2), so (E_1 - E_0)_ii = 2 sum_{lambda > 0} lambda (top_ik^2 - bottom_ik^2) (one decomposition gives both polar
factors).  The four packets' diagonals are summed with eta_s at each j before accumulating (spec 10(p')(ii)): d_j = sum_s eta_s (E_s1 - E_s0)_ii.
W_0 = sum_j (2j + 1) sum_i (L_i - c) d_j,i with L = N psi at the nodes and c = L(R_max): Tr (E_1 - E_0) = 0 exactly on the finite domain, so
the constant c leaves W unchanged, and c = L at the wall multiplies by about zero the wall's compensation of the 1/r tail's Gauss flux.
What remains of the truncation is W_c = W_bulk - 4 pi k A_psi beta/R_max (k the operator's own coefficient, A_psi the 1/r coefficient of
psi - 1, beta that of 1 - L), exactly quadratic in the amplitude; W_bulk = W_c + 4 pi k R_max (psi(R_max) - 1)(1 - L(R_max)) is reported beside
W_c (k from the Gaussian profiles of this run, else the lattice k_op/8, stated in the record).
The clock-row density: x(r) = sum_j (2j + 1) d_j(r)/(4 pi r^2 h) per unit volume per cone (integer and half nodes), w(r) = psi x(psi).

Profiles: the stage-1 member (r = 1..J, psi, N from <family>_profiles.npz) interpolated by a cubic spline on the mirrored data (even in r),
the exterior psi = 1 + A/r, N = (1 - B/r)/(1 + A/r) beyond r_J (the member's A, B); --interp akima gives the alternative interpolation
(its effect through W_0^(2) is the interpolation error).  Configurations (one psi each; the L variants share its decompositions):
  flat     psi = 1 with L = the star's N psi: W and max |d_j| per j are the exact-normality control (zero in exact arithmetic);
  star_e   psi_e = 1 + e (psi - 1), N_e = 1 + e (N - 1) for e in +-{1e-3, 2e-3, 4e-3} (the specification's) and +-{1e-2, 2e-2, 4e-2}
           (builder's addition, the round-off check), and e = 1 (the star); L variants: primary N_e psi_e, psi_only psi_e (N = 1);
  gauss_w  psi = 1 + e f_w, f_w = exp(-r^2/(2 w^2)), w = 20 and 60 lattice units, e in +-{1e-3, 2e-3, 4e-3}, +-{1e-2, 2e-2, 4e-2}; L = psi;
           on its own grid R_max = max(R_max of the star, 8 w).
Records per configuration and L variant: W_c, W_bulk, the per-j values (packet-summed) and their partial sums, the per-packet totals of the
primary variant (the cancellation), the timing; the per-node densities of the star and of the e = +-1e-3, +-1e-2 star configurations
(npz).  The analysis (W_0^(2), the eps^3 term, k and g_4, J5's number, the lattice comparison) is strand5_w0radial_report_v1.py's.
The lattice kernel's quadratic forms on the same profiles (the stage-1 stencil, strand5_lattice_star_v3.Radial on r = 1..J + R with the
exterior tails): sum 4 pi r^2 (p + n) C_lat p and sum 4 pi r^2 p C_lat p (p = psi - 1, n = N - 1) for the star, sum 4 pi r^2 f C_lat f for the
Gaussians, and the stage-1 record's W = sum 4 pi r^2 (chi - 1) C_lat (psi - 1) beside them.  Writes <out>/w0radial_<tag>.json and .npz.
(version 2, spec 10(p''): the dilation family psi_lam(r) = lam^-1/2 psi(r/lam), N_lam(r) = N(r/lam) at lam = 1 -+ 1e-3, 1 -+ 2e-3 (--what dil):
the nonlinear W_0 (L = N_lam psi_lam) and the quadratic form by the same steps, W at psi = 1 + e (psi_lam - 1), N = 1 + e (N_lam - 1),
e = +-2e-2 (the even part over e^2; its e^2 correction is the star series' at lam = 1, stated).  Every configuration records its asymptotic
ruler s (1, or lam^-1/2, or 1 + e (lam^-1/2 - 1)), psi and L at the wall: the wall term generalizes to W_c = W_bulk - 4 pi k R_max (psi(R_max)
- s)(s - L(R_max)) (the report's correction).  The exact continuum trace is invariant under the dilation (A_lam = lam^1/2 U A U^dag,
L_lam = lam^-1/2 U L U^dag), so the nonlinear derivative measures the radial grid's and the wall's breaking of it beside the frozen form's.)
(version 3: the radial derivative in the integrating-factor form D_kappa = r^-kappa d/dr r^kappa,
    (D u)_{i-1/2} = [(r_i/r_{i-1/2})^kappa u_i - (r_{i-1}/r_{i-1/2})^kappa u_{i-1}]/h,
second-order like versions 1 and 2's (u_i - u_{i-1})/h + kappa (u_i + u_{i-1})/(2 r_{i-1/2}), but without that form's spurious mode: the
averaged centrifugal term annihilates the alternating half-node vector, and D^T then has a near-zero mode localized at r = (h/2)(j + 1/2)^1/2
for every j >= 5/2 (the local balance -2/h + (j + 1/2) h/(2 r^2) = 0), whose singular value is the packet's mass, so the four packets'
traces did not cancel and the per-j values grew as j^2 in the pilot's first run (J5pilot1, stopped).  In the integrating-factor form D and
D^T act on the alternating vectors by sums of positive terms (no such zero).  Per channel the grid starts at the integer node i_0 = max(1,
floor((j + 1/2)/cut)) (u = 0 below; cut = 4 by default, --cut): r_0 = (j + 1/2) h/cut lies inside the centrifugal barrier of every state
the grid carries (the turning point (j + 1/2)/p >= (j + 1/2) h/2 for p <= 2/h), and the factors stay within about e^(+-cut/2) there; the
dropped nodes carry d = 0.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.interpolate import Akima1DInterpolator, CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_lattice_star_v3 as LS  # noqa: E402

EPS_SPEC = (1e-3, 2e-3, 4e-3)
EPS_BIG = (1e-2, 2e-2, 4e-2)


def packet_masses(ml, aH):
    a = math.sqrt(ml * ml + aH * aH)
    return [ml, a, a, math.sqrt(ml * ml + 2 * aH * aH)], [1, -1, -1, 1]


class Member:
    def __init__(self, family, profiles, zc, interp="cubic"):
        F = json.load(open(family))
        P = np.load(profiles)
        keys = sorted({k.rsplit("_", 1)[0] for k in P.files})
        kz = min(keys, key=lambda k: abs(float(k[1:]) - zc))
        self.key = kz
        self.r, self.psi, self.N = P[kz + "_r"], P[kz + "_psi"], P[kz + "_N"]
        self.meta = F["meta"]
        self.mem = min(F["members"], key=lambda m: abs(m["z_c"] - float(kz[1:])))
        self.A, self.B, self.rs = float(self.mem["A"]), float(self.mem["B"]), float(self.mem["r_s"])
        rr = np.concatenate([-self.r[::-1], self.r])
        mk = CubicSpline if interp == "cubic" else Akima1DInterpolator
        self.sp = mk(rr, np.concatenate([self.psi[::-1], self.psi]))
        self.sn = mk(rr, np.concatenate([self.N[::-1], self.N]))
        self.rJ = float(self.r[-1])

    def fields(self, x):
        x = np.asarray(x, float)
        inside = x <= self.rJ
        xi = np.minimum(x, self.rJ)
        ps = np.where(inside, self.sp(xi), 1 + self.A / x)
        N = np.where(inside, self.sn(xi), (1 - self.B / x) / (1 + self.A / x))
        return ps, N


class RadialOp:
    def __init__(self, h, n, dev, cut=4.0):
        self.h, self.n, self.dev, self.cut = h, n, dev, cut
        self.r_int = h * np.arange(1, n + 1, dtype=float)
        self.r_half = h * (np.arange(n, dtype=float) + 0.5)
        self.rh_t = torch.tensor(self.r_half, dtype=torch.float64, device=dev)
        self.idx = torch.arange(n, device=dev)

    def d_diag(self, psi_int, psi_half, m, j):
        """(E_1 - E_0)_ii on the 2n nodes (integer then half) for one packet and one j (zero on the nodes below the channel's cut); the
        count of positive eigenvalues against the kept size."""
        n, h, dev = self.n, self.h, self.dev
        K = j + 0.5
        kap = -K
        k0 = max(1, int(math.floor(K / self.cut))) - 1
        mm = n - k0
        ri = torch.as_tensor(self.r_int[k0:], dtype=torch.float64, device=dev)
        rh = torch.as_tensor(self.r_half[k0:], dtype=torch.float64, device=dev)
        pi_ = torch.as_tensor(psi_int[k0:], dtype=torch.float64, device=dev)
        ph_ = torch.as_tensor(psi_half[k0:], dtype=torch.float64, device=dev)
        idx = torch.arange(mm, device=dev)
        D = torch.zeros((mm, mm), dtype=torch.float64, device=dev)
        D[idx, idx] = torch.exp(kap * torch.log(ri / rh)) / h
        D[idx[1:], idx[:-1]] = -torch.exp(kap * torch.log(ri[:-1] / rh[1:])) / h
        Ap = torch.zeros((2 * mm, 2 * mm), dtype=torch.float64, device=dev)
        Ap[idx, idx] = m / pi_
        Ap[mm + idx, mm + idx] = m / ph_
        Ap[:mm, mm:] = -(pi_ ** -3)[:, None] * D.T
        Ap[mm:, :mm] = -(ph_ ** -3)[:, None] * D
        H = torch.zeros((4 * mm, 4 * mm), dtype=torch.float64, device=dev)
        H[:2 * mm, 2 * mm:] = Ap
        H[2 * mm:, :2 * mm] = Ap.T
        lam, Q = torch.linalg.eigh(H)
        pos = lam > 0
        lp = lam[pos]
        top = Q[:2 * mm][:, pos]
        bot = Q[2 * mm:][:, pos]
        dk = 2.0 * ((top * top) @ lp - (bot * bot) @ lp)
        d = torch.zeros(2 * n, dtype=torch.float64, device=dev)
        d[k0:n] = dk[:mm]
        d[n + k0:] = dk[mm:]
        return d, int(pos.sum().item()) - 2 * mm + 2 * n


def run_config(op, psi_fn, Lvars, masses, etas, jmax_n, keep_density=False, s_inf=1.0):
    """One psi configuration: per j the packet-summed diagonal; W per L variant (with c = L at the wall) per j; per-packet totals for the
    first L variant; the density sum_j (2j + 1) d_j if asked."""
    psi_int, psi_half = psi_fn(op.r_int), psi_fn(op.r_half)
    names = list(Lvars)
    Lc = {}
    for nm in names:
        Li, Lh = Lvars[nm]
        c = float(Li[-1])
        Lc[nm] = (torch.tensor(np.concatenate([Li, Lh]) - c, dtype=torch.float64, device=op.dev), c)
    Wj = {nm: [] for nm in names}
    maxd = []
    Wpk = [0.0] * len(masses)
    dens = torch.zeros(2 * op.n, dtype=torch.float64, device=op.dev) if keep_density else None
    npos_bad = 0
    for ji in range(jmax_n):
        j = ji + 0.5
        dsum = torch.zeros(2 * op.n, dtype=torch.float64, device=op.dev)
        for s, (m, e) in enumerate(zip(masses, etas)):
            d, npos = op.d_diag(psi_int, psi_half, m, j)
            if npos != 2 * op.n:
                npos_bad += 1
            dsum += e * d
            Wpk[s] += (2 * j + 1) * float(torch.dot(Lc[names[0]][0], d))
        maxd.append(float(dsum.abs().max()))
        for nm in names:
            Wj[nm].append((2 * j + 1) * float(torch.dot(Lc[nm][0], dsum)))
        if keep_density:
            dens += (2 * j + 1) * dsum
    out = {}
    for nm in names:
        w = np.array(Wj[nm])
        ps = np.cumsum(w)
        tot = float(math.fsum(w))
        nlast = max(1, len(w) // 10)
        out[nm] = {"W_c": tot, "c": Lc[nm][1], "W_j": w.tolist(), "last_tenth_over_total": float(math.fsum(w[-nlast:]) / tot) if tot != 0 else None,
                   "partial_at_quarters": [float(ps[min(len(ps) - 1, int(q * len(ps)) - 1)]) for q in (0.25, 0.5, 0.75, 1.0)]}
    rec = {"L_variants": out, "per_packet_W_c_first_variant": Wpk, "max_abs_d_j": maxd, "eigen_count_mismatches": npos_bad,
           "psi_at_wall": float(psi_int[-1]), "s_inf": s_inf, "R_wall": float(op.r_int[-1])}
    if keep_density:
        rec["_density"] = dens.cpu().numpy()
    return rec


def lattice_forms(stencil, J, R, p_in, v_in, A_tail, p_tail_fn, v_tail_fn):
    """sum 4 pi r^2 v (C_lat p) on r = 1..J + R, p and v given on 1..J and by the tail functions beyond (u = r p; the Radial tail constant
    A_tail is the value of u beyond J + R)."""
    Jx = J + R
    rx = np.arange(1, Jx + 1, dtype=float)
    p = np.concatenate([p_in, p_tail_fn(rx[J:])])
    v = np.concatenate([v_in, v_tail_fn(rx[J:])])
    opx = LS.Radial(Jx, stencil)
    Cu = opx.apply(rx * p, A_tail)
    return float(np.sum(4 * np.pi * rx * rx * v * Cu))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--zc", type=float, required=True)
    ap.add_argument("--kernel", default=None, help="the stage-1 kernel table (for the lattice forms)")
    ap.add_argument("--packets", default=None, help="the stage-1 per-packet symbols (version 3 kernels)")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--rmax-over-rs", type=float, default=4.0)
    ap.add_argument("--jmax", type=int, default=0, help="number of j values (default: the resolution limit n)")
    ap.add_argument("--what", default="flat,star,gauss")
    ap.add_argument("--eps", default="spec,big")
    ap.add_argument("--widths", default="20,60")
    ap.add_argument("--interp", default="cubic", choices=["cubic", "akima"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cut", type=float, default=4.0, help="per channel the grid starts at the integer node max(1, floor((j + 1/2)/cut))")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda:%d" % a.gpu) if a.gpu >= 0 else torch.device("cpu")
    M = Member(a.family, a.profiles, a.zc, a.interp)
    ml, aH = float(M.meta["mlight"]), float(M.meta["aH"])
    masses, etas = packet_masses(ml, aH)
    Rmax = a.rmax_over_rs * M.rs
    n = int(round(Rmax / a.h))
    Rmax = n * a.h
    jn = a.jmax if a.jmax > 0 else n
    epsl = []
    if "spec" in a.eps:
        epsl += list(EPS_SPEC)
    if "big" in a.eps:
        epsl += list(EPS_BIG)
    res = {"meta": {"family": a.family, "profiles": a.profiles, "member_key": M.key, "z_c": M.mem["z_c"], "A": M.A, "B": M.B, "r_s": M.rs,
                    "J_stage1": int(M.mem["J"]), "mlight": ml, "aH": aH, "masses": masses, "etas": etas, "h": a.h, "n": n, "R_max": Rmax,
                    "j_count": jn, "interp": a.interp, "k_op": M.meta["k_op"], "eps": epsl, "device": str(dev), "tag": a.tag,
                    "torch": torch.__version__, "cut": a.cut, "version": 3}, "configs": {}}
    out = os.path.join(a.out, "w0radial_%s.json" % a.tag)
    npz = {}
    op = RadialOp(a.h, n, dev, a.cut)
    star_fields_int = M.fields(op.r_int)
    star_fields_half = M.fields(op.r_half)
    what = a.what.split(",")

    def star_L(e, variant):
        res_ = []
        for (ps, N) in (star_fields_int, star_fields_half):
            pe = 1 + e * (ps - 1)
            Ne = 1 + e * (N - 1)
            res_.append(Ne * pe if variant == "primary" else pe)
        return res_[0], res_[1]

    def star_psi(e):
        return lambda x: 1 + e * (M.fields(x)[0] - 1)

    def save():
        json.dump(res, open(out, "w"), indent=1)

    if "flat" in what:
        Lv = {"star_L": star_L(1.0, "primary")}
        rec = run_config(op, lambda x: np.ones_like(x), Lv, masses, etas, jn)
        res["configs"]["flat"] = rec
        print("flat: W_c %.3e, max |d_j| %.3e, eigen-count mismatches %d (%.0f s)" % (rec["L_variants"]["star_L"]["W_c"], max(rec["max_abs_d_j"]),
                                                                                  rec["eigen_count_mismatches"], time.time() - t0), flush=True)
        save()
    if "star" in what:
        for e in [1.0] + [s * x for x in epsl for s in (1, -1)]:
            Lv = {"primary": star_L(e, "primary"), "psi_only": star_L(e, "psi_only")}
            keep = e == 1.0 or abs(abs(e) - 1e-3) < 1e-15 or abs(abs(e) - 1e-2) < 1e-15
            rec = run_config(op, star_psi(e), Lv, masses, etas, jn, keep_density=keep)
            if keep:
                npz["star_density_e%+.0e" % e] = rec.pop("_density")
            res["configs"]["star_e%+.0e" % e] = dict(rec, eps=e)
            print("star e %+.0e: W_c primary %.12e psi_only %.12e; per-packet %s; last tenth %.2e (%.0f s)" % (
                e, rec["L_variants"]["primary"]["W_c"], rec["L_variants"]["psi_only"]["W_c"], ["%.4e" % x for x in rec["per_packet_W_c_first_variant"]],
                rec["L_variants"]["primary"]["last_tenth_over_total"] or 0, time.time() - t0), flush=True)
            save()
    if "dil" in what:
        for lam in (1 - 2e-3, 1 - 1e-3, 1 + 1e-3, 1 + 2e-3):
            sl = lam ** -0.5
            fi = M.fields(op.r_int / lam)
            fh = M.fields(op.r_half / lam)
            Li, Lh = fi[1] * sl * fi[0], fh[1] * sl * fh[0]
            rec = run_config(op, lambda x, lam=lam, sl=sl: sl * M.fields(x / lam)[0], {"primary": (Li, Lh)}, masses, etas, jn, s_inf=sl)
            res["configs"]["dil_l%+.0e" % (lam - 1)] = dict(rec, lam=lam)
            print("dil lam - 1 %+.0e: W_c %.15e (%.0f s)" % (lam - 1, rec["L_variants"]["primary"]["W_c"], time.time() - t0), flush=True)
            save()
            for e in (2e-2, -2e-2):
                pi_, ph_ = 1 + e * (sl * fi[0] - 1), 1 + e * (sl * fh[0] - 1)
                ni_, nh_ = 1 + e * (fi[1] - 1), 1 + e * (fh[1] - 1)
                rec = run_config(op, lambda x, lam=lam, sl=sl, e=e: 1 + e * (sl * M.fields(x / lam)[0] - 1), {"primary": (ni_ * pi_, nh_ * ph_)},
                                 masses, etas, jn, s_inf=1 + e * (sl - 1))
                res["configs"]["dil_l%+.0e_e%+.0e" % (lam - 1, e)] = dict(rec, lam=lam, eps=e)
                print("dil lam - 1 %+.0e e %+.0e: W_c %.15e (%.0f s)" % (lam - 1, e, rec["L_variants"]["primary"]["W_c"], time.time() - t0), flush=True)
                save()
    if "gauss" in what:
        for w in [float(x) for x in a.widths.split(",")]:
            ng = max(n, int(math.ceil(8 * w / a.h)))
            opg = op if ng == n else RadialOp(a.h, ng, dev, a.cut)
            fw = lambda x, w=w: np.exp(-x * x / (2 * w * w))
            for e in [s * x for x in epsl for s in (1, -1)]:
                Li = 1 + e * fw(opg.r_int)
                Lh = 1 + e * fw(opg.r_half)
                rec = run_config(opg, lambda x, e=e, fw=fw: 1 + e * fw(x), {"psi_only": (Li, Lh)}, masses, etas, ng)
                rec["n"] = ng
                rec["R_max"] = ng * a.h
                res["configs"]["gauss_w%g_e%+.0e" % (w, e)] = dict(rec, eps=e, width=w)
                print("gauss w %g e %+.0e: W_c %.12e (n %d; %.0f s)" % (w, e, rec["L_variants"]["psi_only"]["W_c"], ng, time.time() - t0), flush=True)
                save()
    # the lattice kernel's forms on the same profiles
    if a.kernel or a.packets:
        args = SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=ml, aH=aH, operator="kernel")
        stencil, kmom, info = LS.setup_kernel(args)
        R = max(stencil)
        J = int(M.mem["J"])
        r = M.r
        p, Nn = M.psi - 1, M.N - 1
        ptail = lambda x: M.A / x
        ntail = lambda x: (1 - M.B / x) / (1 + M.A / x) - 1
        lat = {"k_moment": kmom, "stencil_half_width": R,
               "pCp_plus_nCp": lattice_forms(stencil, J, R, p, p + Nn, M.A, ptail, lambda x: ptail(x) + ntail(x)),
               "pCp": lattice_forms(stencil, J, R, p, p, M.A, ptail, ptail),
               "chi_minus_1_C_psi_minus_1": lattice_forms(stencil, J, R, p, M.psi * M.N - 1, M.A, ptail, lambda x: -M.B / x),
               "stage1_W": M.mem.get("W")}
        for w in [float(x) for x in a.widths.split(",")]:
            Jg = int(math.ceil(10 * w))
            rg = np.arange(1, Jg + 1, dtype=float)
            f = np.exp(-rg * rg / (2 * w * w))
            lat["gauss_w%g_fCf" % w] = lattice_forms(stencil, Jg, R, f, f, 0.0, lambda x: 0 * x, lambda x: 0 * x)
        res["lattice"] = lat
        print("lattice forms: %s (%.0f s)" % (json.dumps({k: v for k, v in lat.items()}), time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    save()
    if npz:
        np.savez(os.path.join(a.out, "w0radial_%s_density.npz" % a.tag), r_int=op.r_int, r_half=op.r_half, **npz)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
