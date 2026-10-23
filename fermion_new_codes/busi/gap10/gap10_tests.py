#!/usr/bin/env python
"""gap10_tests.py -- unit tests and small-box controls (run before any production run; --quick for the box CPU)."""
import argparse
import math
import sys
import numpy as np
import scipy.sparse as sp
import torch
import gap10_core as C

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true', help="smaller boxes for a CPU pass")
ap.add_argument('--device', default='auto')
ARGS = ap.parse_args()
QUICK = ARGS.quick
dev = torch.device(ARGS.device) if ARGS.device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FAIL = []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAIL.append(name)


def sparse_h(m, v, N=None):
    """Independent construction of h from the definition, by site loops (layout (4, L, L, L))."""
    L = v[0].shape[0]
    Nn = 4 * L ** 3

    def flat(comp, ix, iy, iz):
        return ((comp * L + ix) * L + iy) * L + iz

    def Nf(x):
        return 1.0 if N is None else float(N[x])

    rows, cols, vals = [], [], []
    for ix in range(L):
        for iy in range(L):
            for iz in range(L):
                x = (ix, iy, iz)
                if m != 0.0:
                    for comp in range(4):
                        rows.append(flat(comp, *x)); cols.append(flat(comp, *x)); vals.append(m * Nf(x) * C.BETA_DIAG[comp])
                for i in range(3):
                    for sgn in (+1, -1):
                        y = list(x); y[i] += sgn
                        if 0 <= y[i] < L:
                            nw = 0.5 * (Nf(x) + Nf(tuple(y)))
                            for r_ in range(4):
                                for c_ in range(4):
                                    a = C.ALPHA[i][r_, c_]
                                    if a != 0:
                                        rows.append(flat(r_, *x)); cols.append(flat(c_, *y)); vals.append(a * nw * sgn / 2j)
                        y2 = list(x); y2[i] += 2 * sgn
                        if 0 <= y2[i] < L:
                            w = 0.5 * (v[i][x] + v[i][tuple(y2)])
                            for comp in range(4):
                                rows.append(flat(comp, *x)); cols.append(flat(comp, *y2)); vals.append(w * sgn / 4j)
    return sp.csr_matrix((np.array(vals, dtype=np.complex128), (rows, cols)), shape=(Nn, Nn))


def make_radial(L, rs, Rb, ell, m, sampling='averaged', prof=None):
    c = L // 2
    prof = prof or C.RadialProfile.ball(rs, Rb)
    v_full, N_full, comp = C.build_site_fields(L + 4, c + 2, ell, prof, sampling)
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    N = None if N_full is None else N_full[2:-2, 2:-2, 2:-2].copy()
    ghost = C.ghost_weights_np(v_full, L, N_full)
    wp, wm = C.bond_weights_np(v)
    lp, lm = (None, None) if N is None else C.link_weights_np(N)
    dE = 1.01 * C.spectral_bound(m, wp, wm, N)
    return c, v, N, v_full, N_full, C.LatticeH(m, wp, wm, dev, ghost_np=ghost, N_np=N, lp_np=lp, lm_np=lm), dE


def lapse_profile(rs, Rb):
    """A record-like table with N < 1 inside: N = 1 - 0.3 exp(-(r/Rb)^2), u the ball magnitude."""
    r = np.linspace(0.0, 3.0 * rs, 601)
    ball = C.RadialProfile.ball(rs, Rb)
    N = 1.0 - 0.3 * np.exp(-(r / Rb) ** 2)
    return C.RadialProfile(rs, Rb, 'file', 'test lapse table', False, (r, N, ball.u(r)))


def test_stencil_vs_sparse():
    for label, prof in (("N = 1 (ball)", None), ("N < 1 (record-like table)", lapse_profile(3.0, 1.8))):
        L, rs, Rb, ell, m = 9, 3.0, 1.8, 2.0, 0.2
        c, v, N, v_full, N_full, Hop, dE = make_radial(L, rs, Rb, ell, m, prof=prof)
        Hs = sparse_h(m, v, N)
        herm = abs(Hs - Hs.conj().T).max()
        check(f"sparse h Hermitian [{label}]", herm < 1e-14, f"max|H - H^dag| = {herm:.2e}")
        rng = np.random.default_rng(0)
        psi = rng.normal(size=(2, 4, L, L, L)) + 1j * rng.normal(size=(2, 4, L, L, L))
        out = Hop.apply(torch.tensor(psi, dtype=C.CD, device=dev)).cpu().numpy()
        ref = np.stack([(Hs @ psi[b].reshape(-1)).reshape(4, L, L, L) for b in range(2)])
        err = np.abs(out - ref).max()
        check(f"torch stencil = independent sparse matrix [{label}]", err < 1e-12, f"max diff {err:.2e}")
        lam = np.linalg.eigvalsh(Hs.toarray())
        wp, wm = C.bond_weights_np(v)
        bnd = C.spectral_bound(m, wp, wm, N)
        check(f"rigorous spectral bound >= max |eigenvalue| [{label}]", bnd >= np.abs(lam).max(), f"bound {bnd:.4f}, max|lambda| {np.abs(lam).max():.4f}")
        Lb = L + 4
        Hbig = sparse_h(m, v_full, N_full)
        emb = np.zeros((2, 4, Lb, Lb, Lb), dtype=np.complex128)
        emb[:, :, 2:-2, 2:-2, 2:-2] = psi
        resid = []
        for b in range(2):
            o = (Hbig @ emb[b].reshape(-1)).reshape(4, Lb, Lb, Lb)
            o[:, 2:-2, 2:-2, 2:-2] = 0.0
            resid.append(np.linalg.norm(o))
        got = Hop.boundary_residual(torch.tensor(psi, dtype=C.CD, device=dev)).cpu().numpy()
        err = np.abs(got - np.array(resid)).max()
        check(f"face residual ||(1 - Pi) h Pi psi|| = sparse matrix on the halo box [{label}]", err < 1e-12, f"max diff {err:.2e}, values {np.round(resid, 6)}")


def test_kernel_profile():
    ell = 4.0
    r = np.linspace(0, ell, 200001)
    Kn = 4 * math.pi * C._trapz(r * r * C.kernel_K(r, ell), r)
    check("kernel normalization", abs(Kn - 1) < 1e-6, f"int K = {Kn:.8f}")
    rg, V = C.radial_projection(lambda x: np.asarray(x, dtype=float), ell, 12.0, 1)
    err = np.abs(V[1:] - rg[1:]).max()
    check("l = 1 projection preserves the linear field u = r", err < 2e-5, f"max |V - r| = {err:.2e}")
    rg0, N0 = C.radial_projection(lambda x: np.ones_like(np.asarray(x, dtype=float)), ell, 12.0, 0)
    check("l = 0 projection preserves a constant", np.abs(N0 - 1.0).max() < 2e-5, f"max |K*1 - 1| = {np.abs(N0 - 1.0).max():.2e}")
    rs, Rb = 32.0, 19.2
    prof = C.RadialProfile.ball(rs, Rb)
    rg, V = C.radial_projection(prof.u, ell, 40.0, 1)
    j = int(np.argmin(np.abs(rg - 36.0)))
    r0 = rg[j]
    ratio = V[j] / math.sqrt(rs / r0)
    asym = 1 - 3 * ell ** 2 / (40 * r0 * r0)
    check("l = 1 projection against 1 - 3 ell^2/(40 r^2) (sigma_K^2 = ell^2/15, K only)", abs(ratio - asym) < 5e-5, f"ratio {ratio:.7f}, asymptotic {asym:.7f}")
    h = 0.08
    g = np.arange(-ell, ell + h / 2, h)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    K = C.kernel_K(np.sqrt(X * X + Y * Y + Z * Z), ell)
    yx = r0 + X
    ry = np.sqrt(yx * yx + Y * Y + Z * Z)
    Vbf = (K * prof.u(ry) * (yx / ry)).sum() / K.sum()
    check("l = 1 projection against a brute-force 3-dimensional convolution", abs(Vbf - V[j]) < 1e-4, f"brute force {Vbf:.7f}, projection {V[j]:.7f}")
    rg2, V2 = C.radial_projection(prof.u, ell, 40.0, 1, dr=0.0625, drp=1.0 / 128.0)
    jj = int(np.argmin(np.abs(rg2 - r0)))
    check("radial quadrature converged (step and node halving)", abs(V2[jj] - V[j]) < 1e-6, f"difference {abs(V2[jj] - V[j]):.2e}")
    r_n, u3, _, _ = C.axis_allocated(prof, ell, 'averaged', '100', 40, ngauss=3)
    r_n, u5, _, _ = C.axis_allocated(prof, ell, 'averaged', '100', 40, ngauss=5)
    check("cell-average quadrature converged (3 against 5 Gauss nodes per axis)", np.abs(u3 - u5).max() < 1e-6, f"max difference {np.abs(u3 - u5).max():.2e}")
    cr = C.crossings_between(lambda x: 1.0 - np.interp(np.asarray(x, float), rg, V), 1.0, 40.0)
    pred = rs - 9 * (ell ** 2 / 15.0) / (4 * rs)
    check("outer |v| = 1 crossing of the K-smoothed (not cell-averaged) exterior near rs - 9 sigma_K^2/(4 rs)", len(cr) == 2 and abs(cr[-1][0] - pred) < 0.01,
          f"crossings {cr}, predicted outer {pred:.4f}; predicted inner {Rb ** 1.5 / math.sqrt(rs):.3f}")
    # allocated axis field = the full build on the axes (review R3)
    L, c = 25, 12
    for sampling in ('averaged', 'cell', 'point'):
        v_full, N_full, comp = C.build_site_fields(L + 4, c + 2, 2.0, C.RadialProfile.ball(6.0, 3.6), sampling)
        ch = c + 2
        nax = np.arange(0, c + 3)
        full100 = -v_full[0][ch + nax, ch, ch]
        full111 = -(v_full[0][ch + nax, ch + nax, ch + nax] + v_full[1][ch + nax, ch + nax, ch + nax] + v_full[2][ch + nax, ch + nax, ch + nax]) / C.SQRT3
        _, a100, _, _ = C.axis_allocated(C.RadialProfile.ball(6.0, 3.6), 2.0, sampling, '100', c + 2)
        _, a111, _, _ = C.axis_allocated(C.RadialProfile.ball(6.0, 3.6), 2.0, sampling, '111', c + 2)
        e1, e2 = np.abs(full100 - a100).max(), np.abs(full111 - a111).max()
        check(f"axis_allocated equals the full build on [100] and [111] ({sampling})", e1 < 1e-12 and e2 < 1e-12, f"{e1:.1e}, {e2:.1e}")
        if sampling == 'averaged':
            print(f"[info] allocated against radial K comparator at r = 6 ({sampling}): [100] {a100[6]:.6f}, [111] {np.interp(6.0, nax * C.SQRT3, a111):.6f}, "
                  f"K profile {np.interp(6.0, comp['rg'], comp['V']):.6f}")


def test_dispersion_and_rays():
    n100 = np.array([1.0, 0.0, 0.0])
    n111 = np.ones(3) / math.sqrt(3.0)
    bad = []
    for u in (0.0, 0.8, 1.2):
        for m in (0.0, 0.2):
            for om in (0.25, 0.5, 0.8):
                for n in (n100, n111):
                    k, nchg = C.outgoing_root(om, u, n, m, check_unique=True)
                    res = abs(C.disp_omega(k, u, n, m)[0] - om)
                    h = 1e-6
                    fd = (C.disp_omega(k + h, u, n, m)[0] - C.disp_omega(k - h, u, n, m)[0]) / (2 * h)
                    vg = C.disp_vg(k, u, n, m)[0]
                    if not (res < 1e-12 and abs(fd - vg) < 1e-7 and vg > 0 and nchg == 1):
                        bad.append((u, m, om, res, fd, vg, nchg))
    check("dispersion roots, group velocities, positivity and uniqueness on the half-cell", not bad, str(bad[:3]))
    k0 = C.outgoing_root(1e-6, 1.3, n100, 0.0)
    check("for u > 1 the massless root lies above arccos(1/u) (review R4: a lower endpoint, no cap)", k0 > math.acos(1.0 / 1.3) and k0 < math.acos(1.0 / 1.3) + 0.01,
          f"k(omega -> 0) = {k0:.5f}, arccos(1/u) = {math.acos(1.0 / 1.3):.5f}")
    k, u = 0.9, 1.1
    lhs = C.disp_omega(k, u, n111, 0.0)[0]
    rhs = math.sqrt(3) * math.sin(k / math.sqrt(3)) * (1 - u * math.cos(k / math.sqrt(3)))
    check("[111] dispersion identity", abs(lhs - rhs) < 1e-12, f"{lhs:.12f} vs {rhs:.12f}")
    for m in (0.0, 0.2):
        for n in (n100, n111):
            up, um = C.helicity_spinors(0.7, n, m)
            p = 0.7 * n
            h0 = m * C.BETA + sum(C.ALPHA[i] * math.sin(p[i]) for i in range(3))
            eps = math.sqrt(m * m + sum(math.sin(p[i]) ** 2 for i in range(3)))
            Sn = sum(n[i] * C.SIGMA4[i] for i in range(3))
            ok = (np.linalg.norm(h0 @ up - eps * up) < 1e-12 and np.linalg.norm(h0 @ um - eps * um) < 1e-12
                  and np.linalg.norm(Sn @ up - up) < 1e-12 and np.linalg.norm(Sn @ um + um) < 1e-12
                  and abs(np.vdot(up, um)) < 1e-12 and abs(np.linalg.norm(up) - 1) < 1e-12)
            check(f"helicity spinors are positive-band eigenvectors with Sigma.n = +-1 (m={m}, n={np.round(n, 2)})", ok)
    prof = C.RadialProfile.ball(32.0, 19.2)
    u_fn = prof.u
    cr = C.crossings_between(lambda r: 1.0 - u_fn(r), 8.0, 64.0)
    b = 0.6
    check("inner and outer crossings of the point-sampled profile with inverse slopes b^{3/2} rs and 2 rs",
          len(cr) == 2 and abs(cr[0][0] - b ** 1.5 * 32) < 1e-6 and abs(cr[0][1] - b ** 1.5 * 32) < 1e-3 and abs(cr[1][1] - 64.0) < 1e-2, str(cr))
    cross_r = [x[0] for x in cr]
    T = C.ray_time(24.0, 64.0, 0.1, n100, 0.0, u_fn, crossings=cross_r)
    print(f"[info] ray flight time 24 -> 64 at a w = 0.1, [100], point-sampled exterior rs = 32: {T:.3f} a = {T / 32:.3f} rs (asymptote 3.16 rs)")
    T111 = C.ray_time(24.0, 64.0, 0.1, n111, 0.0, u_fn, crossings=cross_r)
    T100b = C.ray_time(24.0, 64.0, 0.1 / math.sqrt(3), n100, 0.0, u_fn, crossings=cross_r)
    check("T_111(w) = T_100(w/sqrt3) on the same scalar profile (m = 0; a characteristic comparator)", abs(T111 - T100b) < 1e-6, f"{T111:.6f} vs {T100b:.6f}; lag {(T111 - T) / 32:.3f} rs")
    T111m = C.ray_time(24.0, 64.0, 0.3, n111, 0.2, u_fn, crossings=cross_r)
    T100m = C.ray_time(24.0, 64.0, 0.3 / math.sqrt(3), n100, 0.2 / math.sqrt(3), u_fn, crossings=cross_r)
    check("T_111(w, m) = T_100(w/sqrt3, m/sqrt3) on the same scalar profile", abs(T111m - T100m) < 1e-6, f"{T111m:.6f} vs {T100m:.6f}")
    # the delay law (B5): coefficient 1 for a deep source, 2/3 for a source at the zero, in the exact sine rays and the normal form
    gamma = 1.0 / 64.0
    oms = (1e-4, 3e-4, 1e-3)
    t_deep = [C.ray_time(25.6, 64.0, om, n100, 0.0, u_fn, crossings=cross_r) for om in oms]
    t_zero = [C.ray_time(32.0, 64.0, om, n100, 0.0, u_fn, crossings=cross_r) for om in oms]
    x = np.log(1.0 / np.array(oms))
    s_deep = np.polyfit(x, t_deep, 1)[0] / 64.0
    s_zero = np.polyfit(x, t_zero, 1)[0] / 64.0
    check("exact sine rays: log coefficient / (2 rs) = 1 for a source at 0.8 rs", abs(s_deep - 1.0) < 0.01, f"{s_deep:.5f}")
    check("exact sine rays: log coefficient / (2 rs) = 2/3 for a source at the zero", abs(s_zero - 2.0 / 3.0) < 0.02, f"{s_zero:.5f}")
    nf_deep = [C.nf_delay(om, 6.4, 32.0, gamma) for om in oms]
    nf_zero = [C.nf_delay(om, 0.0, 32.0, gamma) for om in oms]
    sn_deep = np.polyfit(x, nf_deep, 1)[0] / 64.0
    sn_zero = np.polyfit(x, nf_zero, 1)[0] / 64.0
    check("normal form: coefficients 1 (deep) and 2/3 (at the zero)", abs(sn_deep - 1.0) < 0.01 and abs(sn_zero - 2.0 / 3.0) < 0.01, f"{sn_deep:.5f}, {sn_zero:.5f}")
    print("[info] normal-form local coefficient at d/delta_w = 0, 0.25, 0.5, 1, 2 (L = 32, a w = 0.1): " + ", ".join(
        f"{(C.nf_delay(0.1 * 1.01, dd * C.nf_delta(0.1, gamma), 32.0, gamma) - C.nf_delay(0.1 * 0.99, dd * C.nf_delta(0.1, gamma), 32.0, gamma)) / math.log(0.99 / 1.01) / 64.0:.3f}"
        for dd in (0.0, 0.25, 0.5, 1.0, 2.0)))
    print(f"[info] delta_w at rs = 32 for a w = 0.05, 0.1, 0.2: {[round(C.nf_delta(w, gamma), 2) for w in (0.05, 0.1, 0.2)]} a")
    print(f"[info] chart map at r = 2 rs (rs = 32): T - t = {C.chart_T(0.0, 64.0, 32.0):.3f}, u - t = {C.chart_u(0.0, 64.0, 32.0):.3f}; Fu' = 1/(1 - S_0) checked below")
    r1, r2 = 60.0, 60.001
    Fu1, Fu2 = C.clock_offsets(r1, 32.0)[1], C.clock_offsets(r2, 32.0)[1]
    check("dFu/dr = 1/(1 - S_0) (review eq. G10.7)", abs((Fu2 - Fu1) / (r2 - r1) - 1.0 / (1.0 - math.sqrt(32.0 / r1))) < 1e-4)


def test_packets_receivers_geometry():
    L, c, rs = (41, 20, 7.0) if QUICK else (61, 30, 10.0)
    n = np.array([1.0, 0.0, 0.0])
    k = 0.4
    up, um = C.helicity_spinors(k, n, 0.0)
    pos = np.array([int(0.5 * rs), 0, 0])
    sig = 3.0 if QUICK else 4.0
    idx, v1, coords, prob, meta = C.packet_mode(L, c, pos, n, k, up, sig, 5.0, rs, 'raw')
    _, v2, _, _, _ = C.packet_mode(L, c, pos, n, k, um, sig, 5.0, rs, 'raw')
    check("packet: normalized, unclipped, retained fraction 1", abs(np.linalg.norm(v1) - 1) < 1e-12 and not meta['clipped'] and abs(meta['retained'] - 1) < 1e-12,
          f"retained {meta['retained']:.6f}, exterior prob {meta['exterior_probability']:.4f}")
    idx_in, vin, cin, pin, min_ = C.packet_mode(L, c, pos, n, k, up, sig, 5.0, rs, 'in')
    idx_out, vout, cout, pout, mout = C.packet_mode(L, c, pos + np.array([int(rs), 0, 0]), n, k, up, sig, 5.0, rs, 'out')
    check("packet restrictions: 'in' entirely inside rs, 'out' entirely outside with kinematic clock spreads",
          np.all(np.linalg.norm(cin, axis=1) < rs) and np.all(np.linalg.norm(cout, axis=1) > rs) and mout['T_rms'] > 0 and mout['u_rms'] > mout['T_rms'],
          f"retained in {min_['retained']:.4f}, out {mout['retained']:.4f}, T_rms {mout['T_rms']:.3f}, u_rms {mout['u_rms']:.3f}")
    _, vcl, _, _, mcl = C.packet_mode(L, c, pos, n, k, up, sig, 5.0, rs, 'raw', common_half=int(pos[0]) + 4)
    check("common clipping records a retained fraction below 1", mcl['clipped'] and mcl['retained'] < 1.0, f"retained {mcl['retained']:.4f}")
    bw = C.band_weights(v1, coords, pos, 0.0, np.array([-0.8, 0.0, 0.0]), n, int(math.ceil(5.0 * sig)))
    check("source band weights: norm 1 including the zero mode, upper-band weight above 0.95, outgoing positive frequency above 0.9",
          abs(bw['norm'] - 1) < 1e-10 and bw['upper_band'] > 0.95 and bw['outgoing_positive_frequency'] > 0.9, str({k_: round(v_, 5) for k_, v_ in bw.items()}))
    rec = C.Receivers(L, c, dev)
    rec.add_site(np.array([-8, 2, 1]), 'x', 8.3)
    rec.add_packet_group(idx, [v1, v2], [dict(kind='pkt', role='x', r=5.0, omega=0.1, sigma=sig, res='raw', hel=+1, label='a'),
                                         dict(kind='pkt', role='x', r=5.0, omega=0.1, sigma=sig, res='raw', hel=-1, label='b')])
    rec.finalize()
    S = rec.groups[0]['S']
    check("packet overlap matrix: unit diagonal, orthogonal helicities", abs(S[0, 0] - 1) < 1e-12 and abs(S[1, 1] - 1) < 1e-12 and abs(S[0, 1]) < 1e-12)
    psi = torch.zeros((2, 4, L, L, L), dtype=C.CD, device=dev)
    psi[0].view(4, -1).index_copy_(1, torch.tensor(idx, dtype=torch.int64, device=dev), torch.tensor(v1, dtype=C.CD, device=dev))
    psi[1, 2, c - 8, c + 2, c + 1] = 1.0
    P = rec.project(psi).cpu().numpy()
    check("receiver projections: packet on itself = 1, on the other helicity = 0, site component = 1, site outside the packet = 0",
          abs(P[0, 4] - 1) < 1e-12 and abs(P[0, 5]) < 1e-12 and abs(P[1, 2] - 1) < 1e-12 and abs(P[1, 4]) < 1e-12 and abs(P[1, 0]) < 1e-12)
    geo = C.Geometry(L, c, rs, 0.6 * rs, dev, region_fracs=(0.5, 1.5))
    nm, hist, sh, bnd, pz, pd = geo.measure(psi)
    G = geo.gram(psi)
    check("geometry: norms, shell sums and shares", np.allclose(nm, 1.0) and np.allclose(hist.sum(1), 1.0) and hist[1, 8] == 1.0 and sh[1, 0] == 1.0 and sh[1, 1] == 0.0 and bnd[1] == 0.0,
          f"norms {nm}, shell of the site {np.argmax(hist[1])}, shares {sh[1]}")
    lam, vec = C.smax2_from_gram(G[0])
    check("region Gram: diagonal = share of each source beyond r_h, lambda_max over 2 orthogonal sources = their larger share",
          abs(G[0, 0, 0].real - sh[0, 0]) < 1e-12 and abs(G[0, 1, 1].real - sh[1, 0]) < 1e-12 and abs(lam - max(sh[0, 0], sh[1, 0])) < 1e-12,
          f"diag {np.real(np.diag(G[0]))}, lambda_max {lam:.6f}")
    W, rank = C.lowdin_inverse_sqrt(np.array([[1.0, 0.5], [0.5, 1.0]]))
    check("Lowdin S^{-1/2}: W S W = 1, full rank", np.allclose(W @ np.array([[1.0, 0.5], [0.5, 1.0]]) @ W, np.eye(2)) and rank == 2)


def test_fft_exact():
    L, c = (41, 20) if QUICK else (97, 48)
    tau = 4.0 if QUICK else 10.0
    for m in (0.0, 0.2):
        for vc in ((0.0, 0.0, 0.0), (0.0, 0.0, -0.5)):
            mode = 'flat' if vc == (0.0, 0.0, 0.0) else 'uniform'
            v_full, _, _ = C.build_site_fields(L + 4, c + 2, 1.0, None, 'point', mode=mode, uniform_v=vc)
            v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
            wp, wm = C.bond_weights_np(v)
            dE = 1.01 * C.spectral_bound(m, wp, wm)
            Hop = C.LatticeH(m, wp, wm, dev, ghost_np=C.ghost_weights_np(v_full, L))
            psi0 = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
            psi0[0, 0, c, c, c] = 1.0
            n, tr, Cm, Ik, tail = C.cheb_coefficients(dE, tau, 0.5)
            proj = lambda phi: phi[:, 0, c, c, c].reshape(1, 1)
            psi, P, R, Nr = C.cheb_sweep(Hop, psi0, dE, Cm[-1], n, proj)
            ref = C.free_propagate_fft(psi0, tau, m, vc)
            err = (psi - ref).abs().max().item()
            check(f"Chebyshev = exact Fourier propagator (m={m}, v={vc})", err < 1e-11, f"max diff {err:.2e}, n={n}, tail bound {tail:.1e}")
            A = (torch.tensor(Cm, dtype=C.CD, device=dev) @ P.reshape(n + 1, -1))[-1].item()
            check(f"in-sweep readout at tau equals the state (m={m}, v={vc})", abs(A - psi[0, 0, c, c, c].item()) < 1e-12)
            tp = 0.37 * tau
            Cp = C.cheb_readout(dE, n, [tp])[0]
            A_mid = complex(Cp @ P.reshape(n + 1, -1).cpu().numpy()[:, 0])
            ref_mid = C.free_propagate_fft(psi0, tp, m, vc)[0, 0, c, c, c].item()
            check(f"readout at an arbitrary in-sweep time from the stored projections (m={m}, v={vc})", abs(A_mid - ref_mid) < 1e-11, f"diff {abs(A_mid - ref_mid):.2e}")
            cert = float(Ik[-1] @ R.cpu().numpy()[:, 0])
            check(f"certificate small while the wave is far from the faces (m={m}, v={vc})", cert < 1e-6, f"eps_B({tau:g}) = {cert:.2e}")
            check(f"Chebyshev vector norms stay <= 1 + 1e-12 (m={m}, v={vc})", float(Nr.max().item()) < 1.0 + 1e-12, f"max {float(Nr.max().item()) - 1.0:.1e}")
            J = C.taylor_order(dE, 1.0)
            ps = psi0
            for _ in range(int(tau)):
                ps, _, _ = C.taylor_step(Hop, ps, 1.0, J, residual=False)
            errT = (ps - ref).abs().max().item()
            check(f"Taylor = exact Fourier propagator (m={m}, v={vc})", errT < 1e-11, f"max diff {errT:.2e}, J={J}")
            acc, inc, Pst = C.taylor_step(Hop, psi0, 1.0, J, project=proj)
            frac = 0.43
            A_t = complex((frac ** np.arange(J + 1)) @ Pst.reshape(J + 1, -1).cpu().numpy()[:, 0])
            ref_t = C.free_propagate_fft(psi0, frac, m, vc)[0, 0, c, c, c].item()
            check(f"Taylor in-step readout from the term projections (m={m}, v={vc})", abs(A_t - ref_t) < 1e-11, f"diff {abs(A_t - ref_t):.2e}")


def test_cheb_vs_taylor_radial():
    L, rs, Rb, ell, m = (25, 5.0, 3.0, 2.0, 0.2) if QUICK else (41, 8.0, 4.8, 2.0, 0.2)
    c, v, N, v_full, N_full, Hop, dE = make_radial(L, rs, Rb, ell, m)
    rng = np.random.default_rng(1)
    psi0 = torch.tensor(rng.normal(size=(1, 4, L, L, L)) + 1j * rng.normal(size=(1, 4, L, L, L)), dtype=C.CD, device=dev)
    psi0 /= psi0.norm()
    e_in, _ = Hop.energy(psi0)
    tau = 8.0
    n, tr, Cm, Ik, tail = C.cheb_coefficients(dE, tau, 0.5)
    proj = lambda phi: phi[:, 0, c, c, c].reshape(1, 1)
    pc = psi0
    for _ in range(2):
        pc, _P, _R, _N = C.cheb_sweep(Hop, pc, dE, Cm[-1], n, proj, residual=False)
    J = C.taylor_order(dE, 1.0)
    pt = psi0
    for _ in range(16):
        pt, _, _ = C.taylor_step(Hop, pt, 1.0, J, residual=False)
    err = (pc - pt).abs().max().item()
    nerr = abs(pc.norm().item() - 1.0)
    e_out, _ = Hop.energy(pc)
    check("Chebyshev = Taylor on the radial field (t = 16)", err < 1e-11, f"max diff {err:.2e}")
    check("unitarity over 16 time units", nerr < 1e-12, f"|norm - 1| = {nerr:.2e}")
    check("energy conservation over 16 time units", abs(e_out.item() - e_in.item()) < 1e-10, f"|E - E0| = {abs(e_out.item() - e_in.item()):.2e}")


def test_drift_identity():
    """Exact identity: the centroid velocity difference between a uniform-flow run and a flat run of the same packet
    equals v_z <cos 2 p_z>_0 (uniform v: h = h_0 + v.Q commutes with h_0; drift +v)."""
    L, c, m, vc, tau = (41, 20, 0.0, (0.0, 0.0, -0.5), 5.0) if QUICK else (97, 48, 0.0, (0.0, 0.0, -0.5), 10.0)
    n = np.array([0.0, 0.0, 1.0])
    k = 0.6
    up, _ = C.helicity_spinors(k, n, m)
    idx, vals, coords, prob, meta = C.packet_mode(L, c, np.array([0, 0, 0]), n, k, up, 3.0 if QUICK else 4.0, 5.0, 1.0, 'raw')
    psi0 = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
    psi0[0].view(4, -1).index_copy_(1, torch.tensor(idx, dtype=torch.int64, device=dev), torch.tensor(vals, dtype=C.CD, device=dev))
    zc = torch.arange(L, dtype=C.FD, device=dev) - c

    def centroid(p):
        dens = (p.real ** 2 + p.imag ** 2).sum(1)[0]
        return (dens.sum((0, 1)) * zc).sum().item() / dens.sum().item()

    res = {}
    for mode, vv in (('flat', (0.0, 0.0, 0.0)), ('uniform', vc)):
        v_full, _, _ = C.build_site_fields(L + 4, c + 2, 1.0, None, 'point', mode=mode, uniform_v=vv)
        v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
        wp, wm = C.bond_weights_np(v)
        dE = 1.01 * C.spectral_bound(m, wp, wm)
        Hop = C.LatticeH(m, wp, wm, dev)
        nn, tr, Cm, Ik, tail = C.cheb_coefficients(dE, tau, 0.5)
        psi, _, _, _ = C.cheb_sweep(Hop, psi0, dE, Cm[-1], nn, lambda phi: phi[:, 0, c, c, c].reshape(1, 1), residual=False)
        res[mode] = (centroid(psi) - centroid(psi0)) / tau
    ph = torch.fft.fftn(psi0, dim=(2, 3, 4))
    w = (ph.real ** 2 + ph.imag ** 2).sum((0, 1, 2, 3))
    pz = 2 * math.pi * torch.fft.fftfreq(L, d=1.0, dtype=C.FD, device=dev)
    cos2 = ((w * torch.cos(2 * pz)).sum() / w.sum()).item()
    dv_exact = vc[2] * cos2
    dv = res['uniform'] - res['flat']
    check("flow drift identity (uniform v = -0.5 z, packet k = +0.6 z): d<z>/dt difference = v_z <cos 2 p_z>",
          abs(dv - dv_exact) < 1e-6, f"measured {dv:.8f}, exact {dv_exact:.8f}; drift without flow {res['flat']:.5f}")


def test_flat_light_cone():
    L, c, m = (41, 20, 0.0) if QUICK else (65, 32, 0.0)
    dist = 12 if QUICK else 20
    v_full, _, _ = C.build_site_fields(L + 4, c + 2, 1.0, None, 'point', mode='flat')
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    wp, wm = C.bond_weights_np(v)
    dE = 1.01 * C.spectral_bound(m, wp, wm)
    Hop = C.LatticeH(m, wp, wm, dev)
    psi = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
    psi[0, 0, c, c, c] = 1.0
    tau = 10.0 if not QUICK else 6.0
    n, tr, Cm, Ik, tail = C.cheb_coefficients(dE, tau, 0.5)
    Cg = torch.tensor(Cm, dtype=C.CD, device=dev)
    proj = lambda phi: phi[:, :, c + dist, c, c].reshape(1, 4)
    series, times = [], []
    t = 0.0
    for _ in range(3):
        psi, P, _, _ = C.cheb_sweep(Hop, psi, dE, Cm[-1], n, proj, residual=False)
        series.append(torch.einsum('jk,kbr->jbr', Cg, P).cpu().numpy()[:, 0, :])
        times.append(t + tr)
        t += tau
    A = np.concatenate(series)
    tt = np.concatenate(times)
    amp = np.linalg.norm(A, axis=1)
    i10 = int(np.flatnonzero(amp > 0.1 * amp.max())[0])
    check(f"flat-space front arrives near t = r = {dist} (10 % crossing within [{dist - 3}, {dist + 1}])", dist - 3 <= tt[i10] <= dist + 1, f"t_10 = {tt[i10]:.1f}")


def test_certificate():
    """The certificate (rigorous integrals + truncation tail) must bound the actual open-box error: small box against a box
    whose own boundary is unreached, radial field, site source at the centre."""
    LS, LB = (17, 49) if QUICK else (33, 97)
    rs, Rb, ell, m = (4.0, 2.4, 1.5, 0.0) if QUICK else (8.0, 4.8, 2.0, 0.0)
    cS, _, _, _, _, HS, dES = make_radial(LS, rs, Rb, ell, m)
    cB, _, _, _, _, HB, dEB = make_radial(LB, rs, Rb, ell, m)
    psiS = torch.zeros((1, 4, LS, LS, LS), dtype=C.CD, device=dev)
    psiS[0, 0, cS, cS, cS] = 1.0
    psiB = torch.zeros((1, 4, LB, LB, LB), dtype=C.CD, device=dev)
    psiB[0, 0, cB, cB, cB] = 1.0
    tau = 4.0 if QUICK else 8.0
    nS, trS, CmS, IkS, tailS = C.cheb_coefficients(dES, tau, 0.5)
    nB, trB, CmB, IkB, tailB = C.cheb_coefficients(dEB, tau, 0.5)
    # rigor direction of the integrals and of the tail
    kk = np.arange(nS + 1)
    tf = np.linspace(0.0, tau, 20001)
    plain = np.where(kk == 0, 1.0, 2.0) * C._trapz(np.abs(sps_jv(kk[None, :], dES * tf[:, None])), tf, axis=0)
    check("rigorous Bessel integrals >= plain fine trapezoid for every k", np.all(IkS[-1] >= plain - 1e-15), f"min margin {np.min(IkS[-1] - plain):.2e}")
    actual_tail = 2.0 * np.sum(np.abs(sps_jv(np.arange(nS + 1, nS + 200), dES * tau)))
    check("truncation tail bound >= actual Bessel tail", tailS >= actual_tail, f"bound {tailS:.2e}, actual {actual_tail:.2e}")
    dummy = lambda phi: phi[:, 0, 0, 0, 0].reshape(1, 1)
    hw = LS // 2
    eps, ok_all, info = 0.0, True, []
    for s in range(3):
        psiS, _, R, _ = C.cheb_sweep(HS, psiS, dES, CmS[-1], nS, dummy)
        eps = eps + float(IkS[-1] @ R.cpu().numpy()[:, 0]) + tailS
        psiB, _, _, _ = C.cheb_sweep(HB, psiB, dEB, CmB[-1], nB, dummy, residual=False)
        sub = psiB[:, :, cB - hw:cB + hw + 1, cB - hw:cB + hw + 1, cB - hw:cB + hw + 1]
        diff = (psiS - sub).norm().item()
        info.append((tau * (s + 1), diff, eps))
        ok_all = ok_all and (diff <= eps)
    check(f"parent certificate bounds the actual open-box error at t = {tau:g}, {2 * tau:g}, {3 * tau:g}", ok_all and info[0][2] < 1e-3 and info[-1][1] > 1e-7,
          "(t, actual error, certificate) = " + str([(a, f"{b:.2e}", f"{c_:.2e}") for a, b, c_ in info]))


def sps_jv(k, x):
    import scipy.special as sps
    return sps.jv(k, x)


def test_reflection_symmetry():
    """The lattice reflections of review R5 commute with h on the open parent with the cell-averaged radial field and flip the
    helicity along the fixed axis: z -> -z with W = beta Sigma_z for [100]; x <-> y with W = beta (Sigma_x - Sigma_y)/sqrt2 for [111]."""
    L, rs, Rb, ell, m = 9, 3.0, 1.8, 2.0, 0.2
    c, v, N, v_full, N_full, Hop, dE = make_radial(L, rs, Rb, ell, m)
    Hs = sparse_h(m, v, N)
    Nn = 4 * L ** 3

    def flat(comp, ix, iy, iz):
        return ((comp * L + ix) * L + iy) * L + iz

    for d, perm in (('100', lambda x, y, z: (x, y, L - 1 - z)), ('111', lambda x, y, z: (y, x, z))):
        W = C.reflection_spinor(d)
        rows, cols, vals = [], [], []
        for ix in range(L):
            for iy in range(L):
                for iz in range(L):
                    jx, jy, jz = perm(ix, iy, iz)
                    for cr in range(4):
                        for cc in range(4):
                            if abs(W[cr, cc]) > 0:
                                rows.append(flat(cr, jx, jy, jz)); cols.append(flat(cc, ix, iy, iz)); vals.append(W[cr, cc])
        R = sp.csr_matrix((np.array(vals, dtype=np.complex128), (rows, cols)), shape=(Nn, Nn))
        unit = abs(R @ R.conj().T - sp.identity(Nn)).max()
        comm = abs(R @ Hs @ R.conj().T - Hs).max()
        nvec = C.direction_vector(d)
        Sn = sum(nvec[i] * C.SIGMA4[i] for i in range(3))
        flip = np.linalg.norm(W @ Sn @ np.linalg.inv(W) + Sn)
        check(f"reflection symmetry of h and helicity flip along [{d}]", unit < 1e-12 and comm < 1e-12 and flip < 1e-12, f"unitarity {unit:.1e}, |R h R^dag - h| {comm:.1e}, flip {flip:.1e}")


def test_deterministic_swap():
    """P31 section 2 re-derived and checked on a 6-mode Fock space (5 system modes + 1 apparatus mode): swapping the interior
    mode f with the apparatus mode prepared empty or filled gives detector occupation difference ||P_D U f||^2 in the optimized
    detector mode, for every Gaussian state; the exterior reduced state is the same for both bits at t = 0."""
    nm = 6
    I2, Z, a = np.eye(2), np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = []
    for j in range(nm):
        mats = [Z] * j + [a] + [I2] * (nm - j - 1)
        op = mats[0]
        for M in mats[1:]:
            op = np.kron(op, M)
        ops.append(op.astype(np.complex128))
    car = max(np.abs(ops[i] @ ops[j].conj().T + ops[j].conj().T @ ops[i] - (1.0 if i == j else 0.0) * np.eye(2 ** nm)).max() for i in range(nm) for j in range(nm))
    check("Jordan-Wigner CAR", car < 1e-14, f"{car:.1e}")
    rng = np.random.default_rng(3)
    h1 = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    h1 = h1 + h1.conj().T
    dG = sum(h1[i, j] * ops[i].conj().T @ ops[j] for i in range(5) for j in range(5))
    tt = 0.9
    evals, evecs = np.linalg.eigh(dG)
    UF = evecs @ np.diag(np.exp(-1j * evals * tt)) @ evecs.conj().T
    U1 = np.linalg.matrix_power(np.eye(5), 1)
    w1, V1 = np.linalg.eigh(h1)
    U1 = V1 @ np.diag(np.exp(-1j * w1 * tt)) @ V1.conj().T
    f = np.zeros(5, dtype=np.complex128)
    f[:3] = rng.normal(size=3) + 1j * rng.normal(size=3)
    f /= np.linalg.norm(f)
    cf = sum(f[i] * ops[i] for i in range(5))          # annihilation of mode f: c_f = sum_i f_i c_i
    dop = ops[5]
    Hsw = dop.conj().T @ cf + cf.conj().T @ dop
    ws, Vs = np.linalg.eigh(Hsw)
    Usw = Vs @ np.diag(np.exp(-1j * ws * (math.pi / 2.0))) @ Vs.conj().T
    # a Gaussian system state: the Slater determinant of 2 filled random orthonormal modes, apparatus empty or filled
    Q, _ = np.linalg.qr(rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2)))
    vac = np.zeros(2 ** nm, dtype=np.complex128)
    vac[0] = 1.0
    psi_sys = vac.copy()
    for col in range(2):
        cre = sum(Q[i, col] * ops[i].conj().T for i in range(5))
        psi_sys = cre @ psi_sys
    psi_sys /= np.linalg.norm(psi_sys)
    D = [3, 4]
    PD_Uf = (U1 @ f)[D]
    eta = float(np.vdot(PD_Uf, PD_Uf).real)
    gvec = np.zeros(5, dtype=np.complex128)
    gvec[D] = PD_Uf / math.sqrt(eta)
    n_g = sum(gvec[i].conj() * gvec[j] * ops[i].conj().T @ ops[j] for i in range(5) for j in range(5))
    occ = {}
    occ0 = {}
    for bit in (0, 1):
        st = psi_sys if bit == 0 else dop.conj().T @ psi_sys
        st = Usw @ st
        occ0[bit] = [float(np.vdot(st, ops[i].conj().T @ ops[j] @ st)) for i in D for j in D]
        st = UF @ st
        occ[bit] = float(np.vdot(st, n_g @ st).real)
    diff0 = max(abs(x - y) for x, y in zip(occ0[0], occ0[1]))
    check("deterministic swap: detector occupation difference in the optimized mode = ||P_D U f||^2",
          abs((occ[1] - occ[0]) - eta) < 1e-12, f"difference {occ[1] - occ[0]:.12f}, ||P_D U f||^2 = {eta:.12f}")
    check("deterministic swap: the detector block of the covariance is identical for both bits at t = 0", diff0 < 1e-12, f"max |Delta C_D(0)| = {diff0:.1e}")
    # the postselected insertion, for contrast: availability p_f = f^dag (1 - C) f depends on the state
    Cmat = np.array([[np.vdot(psi_sys, ops[i].conj().T @ ops[j] @ psi_sys) for j in range(5)] for i in range(5)])
    p_f = float((f.conj() @ (np.eye(5) - Cmat.T) @ f).real)
    print(f"[info] postselected insertion availability p_f = f^dag(1 - C)f = {p_f:.4f} (state dependent); the swap needs no conditioning")


if __name__ == '__main__':
    print(f"[tests] device {dev}, quick={QUICK}, in-place complex axpy {'available' if C.AXPY_OK else 'emulated'}")
    test_stencil_vs_sparse()
    test_kernel_profile()
    test_dispersion_and_rays()
    test_packets_receivers_geometry()
    test_fft_exact()
    test_cheb_vs_taylor_radial()
    test_drift_identity()
    test_flat_light_cone()
    test_certificate()
    test_reflection_symmetry()
    test_deterministic_swap()
    print(f"[tests] {len(FAIL)} failures" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
