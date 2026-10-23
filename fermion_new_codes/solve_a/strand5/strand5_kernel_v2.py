#!/usr/bin/env python
"""Strand 5 J2 (spec solve/specs/SPEC_strand5_kernel_star.md sec. 3 J2; builder analyst): the kernel ingredients from the 2c part A table.

Input: kernel_L256.json (strand2c_kernel_v1.py: k_a(q; aH) = -2 W^(2)/(q_lat^2 L^3) along [100] at q = 2 pi n/256, n = 1..128, the
equal rectangle with the light packet at m = 0.1: masses (m, sqrt(m^2 + H^2) twice, sqrt(m^2 + 2 H^2)), orientations (+, -, -, +), H = aH;
and the physical row).  The full symbol along [100] is C(q) = -k_a(q) q_lat^2, q_lat^2 = 2 (1 - cos q) (C 1 = 0).
Per aH:
(1) k_a(0; aH) and k_2 by the least-squares fit a + b q^2 over the table's q <= aH/4 (point count reported; one point: its value, no k_2);
    the [100] q^4 coefficient of F(q) = -C(q) = k_a(q) q_lat^2, c_4 = k_2 - k_a(0)/12, against P14's F(q) = k_a q^2 + b_iso q^4 + b_cubic
    sum_i q_i^4 with b_iso = g_4/P^4 + k a^2/8, b_cubic = -k a^2/6 (P = a = 1, k := k_a(0; aH) as measured), g_4 = L/(2 pi^2),
    L = sum_s eta_s log m_s^2: the predicted [100] coefficient b_iso + b_cubic and b_iso alone.
(2) The real-space kernel along the line, K_n = (1/256) sum_{m=-127}^{128} C(q_m) cos(q_m n) (the [100] symbol's inverse transform on the
    256-ring, i.e. the 3D kernel summed over the transverse planes), with its checks sum_n K_n = C(0) = 0, (1/2) sum_n n^2 K_n against
    k_a(0) (exact for a kernel decaying within the ring), and its decay |K_n|/|K_0| at n = 8, 16, 32, 64, 128 (the range; 128 is the
    ring's half-length, so a kernel not decayed there is aliased).
(3) The radial Green function by the spec's formula G_C(r) = (2 pi^2 r)^-1 int_0^pi dq sin(qr)/(q k_a(q)) (the table's q-range; k_a by a
    cubic spline on the table and the fit a + b q^2 (or the smallest-q value) below the first point), at r = 1, 2, 5, 10, 20, 50, ... up to
    20/(aH), against the Newtonian tail 1/(4 pi k_a(0) r); beside it the same integral with k_a(q) -> k_a(0) (the local row's Green
    function with the same cut at q = pi, whose deviation is the cut alone) and the difference of the two (the kernel's nonlocal part).
(4) The radial convolution operator on the uniform radial grid of lattice spacing, r_j = j a: (C f)_j = (1/r_j) sum_n K_n u_{j-n},
    u = r f extended oddly (the 3D radial action of an isotropic symbol from its line kernel); checks: C 1 = 0 (exact by sum K_n = 0 and
    the symmetry), C (1/r) beyond the range (sizes per j against k_a(0)), and the Gauss law 4 pi sum_j r_j^2 (C(1/r))_j = -4 pi k_a(0).
Writes <out>/kernel_v2.json and <out>/kernel_v2_K.npz.
(version 2: the table's value at the zone edge n = L/2 (q = pi) is halved before use: strand2c_kernel_v1.py defines k_a = -2 W^(2)/(q_lat^2
L^3) with the norm sum_x cos^2(q x_1) = L^3/2 of a cosine mode, but the zone-edge mode cos(pi x_1) = (-1)^x_1 has norm L^3, so the tabulated
k_a(pi) is twice the symbol (it is also twice its neighbours); version 1 used it as is, which put a non-decaying alternating component
K_n ~ (-1)^n into the line kernel; the line stencil is |n| <= L/2 - 1 plus K_{L/2}/2 at +-L/2, so that sum_n K_n = 0 and the symmetry hold
exactly.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.special import sici


def masses_eq(m, H):
    return [m, np.sqrt(m * m + H * H), np.sqrt(m * m + H * H), np.sqrt(m * m + 2 * H * H)], [1, -1, -1, 1]


def analyse(q, ka, aH, masses, etas, label, Lring=256, rmax_cap=None):
    out = {"set": label, "aH": aH, "k_a_zone_edge_tabulated": float(ka[-1]), "k_a_zone_edge_used": float(ka[-1] / 2),
           "k_a_next_to_edge": float(ka[-2])}
    ka = ka.copy()
    ka[-1] = ka[-1] / 2
    # (1) fit
    sel = q <= aH / 4 if aH else np.zeros_like(q, bool)
    npts = int(np.sum(sel))
    out["fit_points"] = npts
    if npts >= 2:
        A = np.vstack([np.ones(npts), q[sel] ** 2]).T
        coef, *_ = np.linalg.lstsq(A, ka[sel], rcond=None)
        k0, k2 = float(coef[0]), float(coef[1])
        out["fit_max_residual"] = float(np.max(np.abs(A @ coef - ka[sel])))
    else:
        k0, k2 = float(ka[0]), None
        out["fit_max_residual"] = None
    out["k_a0"], out["k_2"] = k0, k2
    Lsum = float(sum(e * np.log(m * m) for m, e in zip(masses, etas) if m > 0)) if all(m > 0 for m in masses) else None
    out["L_sum_eta_log_m2"] = Lsum
    if Lsum is not None:
        g4 = Lsum / (2 * np.pi ** 2)
        out["g_4"] = g4
        out["b_iso_pred"] = g4 + k0 / 8
        out["b_cubic_pred"] = -k0 / 6
        out["c4_100_pred"] = g4 + k0 / 8 - k0 / 6
    out["c4_100_fit"] = (k2 - k0 / 12) if k2 is not None else None
    # (2) line kernel on the ring
    n_tab = np.rint(q * Lring / (2 * np.pi)).astype(int)
    assert np.all(n_tab == np.arange(1, Lring // 2 + 1))
    Cq = np.zeros(Lring)
    ql2 = 2 * (1 - np.cos(q))
    for j, nn in enumerate(n_tab):
        Cq[nn % Lring] = -ka[j] * ql2[j]
        Cq[(-nn) % Lring] = -ka[j] * ql2[j]
    K = np.real(np.fft.ifft(Cq))            # K_n for n = 0..255, n > 128 meaning n - 256
    nn_all = np.arange(Lring)
    nn_all = np.where(nn_all > Lring // 2, nn_all - Lring, nn_all)
    out["sum_K"] = float(np.sum(K))
    out["half_second_moment"] = float(0.5 * np.sum(nn_all ** 2 * K))
    out["half_second_moment_over_k_a0"] = out["half_second_moment"] / k0
    out["K0"] = float(K[0])
    out["decay"] = {str(n): float(abs(K[n]) / abs(K[0])) for n in (1, 2, 4, 8, 16, 32, 64, 128)}
    # (3) Green function by the spec's integral
    q_ext = np.concatenate([[0.0], q])
    ka_ext = np.concatenate([[k0], ka])
    if k2 is not None:   # below the first grid point use the fit
        qq = np.linspace(0, q[0], 6)[1:-1]
        q_ext = np.concatenate([[0.0], qq, q])
        ka_ext = np.concatenate([[k0], k0 + k2 * qq ** 2, ka])
    spl = CubicSpline(q_ext, ka_ext, bc_type=((1, 0.0), "not-a-knot"))
    rows = []
    rmax = 20.0 / aH if aH else 2000.0
    if rmax_cap:
        rmax = min(rmax, rmax_cap)
    rlist = [r for r in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000) if r <= rmax] + ([rmax] if rmax not in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000) else [])
    for r in rlist:
        # J(r) = int_0^pi sin(qr) (1/k_a(q) - 1/k_a(0))/q dq, panel Gauss-Legendre (panels of width <= 1/r)
        npan = int(max(8, np.ceil(np.pi * r / 1.0) * 2))
        edges = np.linspace(0, np.pi, npan + 1)
        xg, wg = leggauss(24)
        mid = (edges[:-1] + edges[1:]) / 2
        hw = (edges[1:] - edges[:-1]) / 2
        Q = (mid[:, None] + hw[:, None] * xg[None, :]).ravel()
        WQ = (hw[:, None] * wg[None, :]).ravel()
        h = (1 / spl(Q) - 1 / k0) / Q
        J = float(np.sum(WQ * np.sin(Q * r) * h))
        Si = float(sici(np.pi * r)[0])
        GC_rel = (Si + k0 * J) / (np.pi / 2) - 1
        Gloc_rel = Si / (np.pi / 2) - 1
        rows.append({"r": float(r), "G_C_over_Newton_minus_1": GC_rel, "G_local_cut_over_Newton_minus_1": Gloc_rel,
                     "kernel_part_over_Newton": GC_rel - Gloc_rel})
    out["green"] = rows
    # (4) the radial operator on r_j = j: C 1 and C(1/r) and the Gauss law
    N = 4 * Lring
    j = np.arange(1, N + 1)
    R = Lring // 2
    Kc = {n: float(K[n % Lring]) for n in range(-R, R + 1)}
    Kc[R] *= 0.5
    Kc[-R] *= 0.5

    def apply(fvals):        # fvals at r = 1..N; u = r f, u_0 = 0, odd extension, zero beyond N + R treated by extension of f
        u = np.concatenate([-(j * fvals)[::-1], [0.0], j * fvals])   # index offset N: u at r = -N..N
        res = np.zeros(N)
        for n in range(-R, R + 1):
            idx = (j - n) + N
            ok = (idx >= 0) & (idx < len(u))
            res[ok] += Kc[n] * u[idx[ok]]
        return res / j
    one = apply(np.ones(N))
    cou = apply(1.0 / j)
    inner = j <= N - R          # nodes whose stencil stays on the grid
    out["C1_max_abs_inner"] = float(np.max(np.abs(one[inner])))
    out["C1_scale_k_a0"] = k0
    dec = {}
    for jj in (1, 2, 4, 8, 16, 32, 64, 128):
        dec[str(jj)] = float(cou[jj - 1])
    out["C_coulomb_at_r"] = dec
    out["C_coulomb_max_abs_beyond_128"] = float(np.max(np.abs(cou[(j > R) & inner])))
    out["C_coulomb_max_abs_beyond_128_over_k_a0"] = out["C_coulomb_max_abs_beyond_128"] / k0
    gauss = float(4 * np.pi * np.sum((j ** 2 * cou)[inner]))
    out["gauss_4pi_sum_r2_C_coulomb"] = gauss
    out["gauss_over_minus_4pi_k_a0_minus_1"] = gauss / (-4 * np.pi * k0) - 1
    out["_K"] = K
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    D = json.load(open(a.kernel))
    res = {"meta": {"kernel": os.path.abspath(a.kernel), "L": D["meta"]["L"], "light_mass": 0.1}, "rows": []}
    Ks = {}
    aHs = sorted({r["aH"] for r in D["equal"]}, reverse=True)
    for aH in aHs:
        rr = sorted([r for r in D["equal"] if r["aH"] == aH], key=lambda r: r["n"])
        q = np.array([r["q"] for r in rr])
        ka = np.array([r["k_a"] for r in rr])
        ms, etas = masses_eq(0.1, aH)
        o = analyse(q, ka, aH, ms, etas, "equal rectangle, m = 0.1")
        Ks["aH%g" % aH] = o.pop("_K")
        res["rows"].append(o)
        print("aH %g: k_a0 %.8g k_2 %s c4 fit %s pred %s; sumK %.2e; 2nd moment/k_a0 %.6f; decay128 %.2e; C1 %.1e; coulomb>128 %.1e; gauss %.2e" % (
            aH, o["k_a0"], o["k_2"], o["c4_100_fit"], o.get("c4_100_pred"), o["sum_K"], o["half_second_moment_over_k_a0"],
            o["decay"]["128"], o["C1_max_abs_inner"], o["C_coulomb_max_abs_beyond_128_over_k_a0"], o["gauss_over_minus_4pi_k_a0_minus_1"]), flush=True)
    P = D["physical"]
    rr = sorted(P["rows"], key=lambda r: r["n"])
    q = np.array([r["q"] for r in rr])
    ka = np.array([r["k_a"] for r in rr])
    o = analyse(q, ka, float(P["masses"][1]), P["masses"], [1, -1, -1, 1], "physical a/l_P 0.0213 (light massless, twist 0.371)", rmax_cap=2000)
    Ks["physical"] = o.pop("_K")
    res["rows"].append(o)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "kernel_v2.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(a.out, "kernel_v2_K.npz"), **Ks)
    print("done %.1f s" % res["meta"]["seconds"], flush=True)


if __name__ == "__main__":
    main()
