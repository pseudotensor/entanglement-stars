#!/usr/bin/env python
"""csr_tests.py -- unit tests and small-box controls (run before any production run)."""
import math
import sys
import numpy as np
import scipy.sparse as sp
import torch
import csr_core as C

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FAIL = []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAIL.append(name)


def sparse_h(m, v):
    """Independent construction of h from the definition, by site loops (layout (4, L, L, L))."""
    L = v[0].shape[0]
    N = 4 * L ** 3

    def flat(comp, ix, iy, iz):
        return ((comp * L + ix) * L + iy) * L + iz

    rows, cols, vals = [], [], []
    for ix in range(L):
        for iy in range(L):
            for iz in range(L):
                x = (ix, iy, iz)
                if m != 0.0:
                    for comp in range(4):
                        rows.append(flat(comp, *x)); cols.append(flat(comp, *x)); vals.append(m * C.BETA_DIAG[comp])
                for i in range(3):
                    for sgn in (+1, -1):
                        y = list(x); y[i] += sgn
                        if 0 <= y[i] < L:
                            for r_ in range(4):
                                for c_ in range(4):
                                    a = C.ALPHA[i][r_, c_]
                                    if a != 0:
                                        rows.append(flat(r_, *x)); cols.append(flat(c_, *y)); vals.append(a * sgn / 2j)
                        y2 = list(x); y2[i] += 2 * sgn
                        if 0 <= y2[i] < L:
                            w = 0.5 * (v[i][x] + v[i][tuple(y2)])
                            for comp in range(4):
                                rows.append(flat(comp, *x)); cols.append(flat(comp, *y2)); vals.append(w * sgn / 4j)
    return sp.csr_matrix((np.array(vals, dtype=np.complex128), (rows, cols)), shape=(N, N))


def make_radial(L, rs, Rb, ell, m, averaged=True):
    c = L // 2
    v_full, _ = C.build_site_flow(L + 4, c + 2, ell, lambda r: C.u_continuum(r, rs, Rb), averaged)
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    ghost = C.ghost_weights_np(v_full, L)
    wp, wm = C.bond_weights_np(v)
    dE = 1.01 * C.spectral_bound(m, wp, wm)
    return c, v, v_full, C.LatticeH(m, wp, wm, dev, ghost_np=ghost), dE


def test_stencil_vs_sparse():
    L, rs, Rb, ell, m = 9, 3.0, 1.8, 2.0, 0.2
    c, v, v_full, Hop, dE = make_radial(L, rs, Rb, ell, m)
    Hs = sparse_h(m, v)
    herm = abs(Hs - Hs.conj().T).max()
    check("sparse h Hermitian", herm < 1e-14, f"max|H - H^dag| = {herm:.2e}")
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(2, 4, L, L, L)) + 1j * rng.normal(size=(2, 4, L, L, L))
    out = Hop.apply(torch.tensor(psi, dtype=C.CD, device=dev)).cpu().numpy()
    ref = np.stack([(Hs @ psi[b].reshape(-1)).reshape(4, L, L, L) for b in range(2)])
    err = np.abs(out - ref).max()
    check("torch stencil = independent sparse matrix", err < 1e-12, f"max diff {err:.2e}")
    lam = np.linalg.eigvalsh(Hs.toarray())
    wp, wm = C.bond_weights_np(v)
    bnd = C.spectral_bound(m, wp, wm)
    check("rigorous spectral bound >= max |eigenvalue|", bnd >= np.abs(lam).max(), f"bound {bnd:.4f}, max|lambda| {np.abs(lam).max():.4f}")
    Lb = L + 4
    Hbig = sparse_h(m, v_full)
    emb = np.zeros((2, 4, Lb, Lb, Lb), dtype=np.complex128)
    emb[:, :, 2:-2, 2:-2, 2:-2] = psi
    resid = []
    for b in range(2):
        o = (Hbig @ emb[b].reshape(-1)).reshape(4, Lb, Lb, Lb)
        o[:, 2:-2, 2:-2, 2:-2] = 0.0
        resid.append(np.linalg.norm(o))
    got = Hop.boundary_residual(torch.tensor(psi, dtype=C.CD, device=dev)).cpu().numpy()
    err = np.abs(got - np.array(resid)).max()
    check("face residual ||(1 - Pi) h Pi psi|| = sparse matrix on the halo box", err < 1e-12, f"max diff {err:.2e}, values {resid}")


def test_kernel_profile():
    ell = 4.0
    r = np.linspace(0, ell, 200001)
    Kn = 4 * math.pi * C._trapz(r * r * C.kernel_K(r, ell), r)
    check("kernel normalization", abs(Kn - 1) < 1e-6, f"int K = {Kn:.8f}")
    rg, V = C.l1_projected_profile(lambda x: np.asarray(x, dtype=float), ell, 12.0)
    err = np.abs(V[1:] - rg[1:]).max()
    check("l = 1 projection preserves the linear field u = r", err < 2e-5, f"max |V - r| = {err:.2e}")
    rs, Rb = 32.0, 19.2
    rg, V = C.l1_projected_profile(lambda x: C.u_continuum(x, rs, Rb), ell, 40.0)
    j = int(np.argmin(np.abs(rg - 36.0)))
    r0 = rg[j]
    ratio = V[j] / math.sqrt(rs / r0)
    asym = 1 - 3 * ell ** 2 / (40 * r0 * r0)
    check("l = 1 projection against 1 - 3 ell^2/(40 r^2) (sigma^2 = ell^2/15)", abs(ratio - asym) < 5e-5, f"ratio {ratio:.7f}, asymptotic {asym:.7f}")
    h = 0.08
    g = np.arange(-ell, ell + h / 2, h)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    K = C.kernel_K(np.sqrt(X * X + Y * Y + Z * Z), ell)
    yx = r0 + X
    ry = np.sqrt(yx * yx + Y * Y + Z * Z)
    Vbf = (K * C.u_continuum(ry, rs, Rb) * (yx / ry)).sum() / K.sum()
    check("l = 1 projection against a brute-force 3-dimensional convolution", abs(Vbf - V[j]) < 1e-4, f"brute force {Vbf:.7f}, projection {V[j]:.7f}")
    cr = C.crossings_between(lambda x: np.interp(np.asarray(x, float), rg, V), 1.0, 40.0)
    sig2 = ell ** 2 / 15.0
    pred = rs - 9 * sig2 / (4 * rs)
    check("outer |v| = 1 crossing of the K-smoothed exterior at rs - 9 sigma^2/(4 rs)", len(cr) == 2 and abs(cr[-1][0] - pred) < 0.01,
          f"crossings {cr}, predicted outer {pred:.4f}; predicted inner {Rb ** 1.5 / math.sqrt(rs):.3f}")


def test_dispersion_and_rays():
    n100 = np.array([1.0, 0.0, 0.0])
    n111 = np.ones(3) / math.sqrt(3.0)
    bad = []
    for u in (0.0, 0.8, 1.2):
        for m in (0.0, 0.2):
            for om in (0.25, 0.5, 0.8):
                for n in (n100, n111):
                    k = C.outgoing_root(om, u, n, m)
                    res = abs(C.disp_omega(k, u, n, m)[0] - om)
                    h = 1e-6
                    fd = (C.disp_omega(k + h, u, n, m)[0] - C.disp_omega(k - h, u, n, m)[0]) / (2 * h)
                    vg = C.disp_vg(k, u, n, m)[0]
                    if not (res < 1e-12 and abs(fd - vg) < 1e-7 and vg > 0):
                        bad.append((u, m, om, res, fd, vg))
    check("dispersion roots, group velocities and their positivity", not bad, str(bad[:3]))
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
    u_fn = lambda r: C.u_continuum(r, 32.0, 19.2)
    T = C.ray_time(24.0, 64.0, 0.1, n100, 0.0, u_fn)
    print(f"[info] ray flight time 24 -> 64 at a w = 0.1, [100], point-sampled exterior rs = 32: {T:.2f} a = {T / 32:.3f} rs (hand estimate about 3.1 rs; asymptote 3.16 rs)")
    T111 = C.ray_time(24.0, 64.0, 0.1, n111, 0.0, u_fn)
    T100b = C.ray_time(24.0, 64.0, 0.1 / math.sqrt(3), n100, 0.0, u_fn)
    check("T_111(w) = T_100(w/sqrt3) on the same profile (m = 0)", abs(T111 - T100b) < 1e-6, f"{T111:.6f} vs {T100b:.6f}; lag {(T111 - T) / 32:.3f} rs")
    T111m = C.ray_time(24.0, 64.0, 0.3, n111, 0.2, u_fn)
    T100m = C.ray_time(24.0, 64.0, 0.3 / math.sqrt(3), n100, 0.2 / math.sqrt(3), u_fn)
    check("T_111(w, m) = T_100(w/sqrt3, m/sqrt3) on the same profile", abs(T111m - T100m) < 1e-6, f"{T111m:.6f} vs {T100m:.6f}")
    cr = C.crossings_between(u_fn, 8.0, 64.0)
    b = 0.6
    check("inner and outer crossings of the point-sampled profile with inverse slopes b^{3/2} rs and 2 rs",
          len(cr) == 2 and abs(cr[0][0] - b ** 1.5 * 32) < 1e-6 and abs(cr[0][1] - b ** 1.5 * 32) < 1e-3 and abs(cr[1][1] - 64.0) < 1e-2, str(cr))
    print(f"[info] chart map at r = 2 rs (rs = 32): T - t = {C.chart_T(0.0, 64.0, 32.0):.3f}, u - t = {C.chart_u(0.0, 64.0, 32.0):.3f}")


def test_packets_receivers_geometry():
    L, c, rs = 61, 30, 10.0
    n = np.array([1.0, 0.0, 0.0])
    k = 0.4
    up, um = C.helicity_spinors(k, n, 0.0)
    pos = np.array([5, 0, 0])
    idx, v1, coords, prob, meta = C.packet_mode(L, c, pos, n, k, up, 4.0, 5.0, rs, 'raw')
    _, v2, _, _, _ = C.packet_mode(L, c, pos, n, k, um, 4.0, 5.0, rs, 'raw')
    check("packet: normalized, unclipped, retained fraction 1", abs(np.linalg.norm(v1) - 1) < 1e-12 and not meta['clipped'] and abs(meta['retained'] - 1) < 1e-12,
          f"retained {meta['retained']:.6f}, exterior prob {meta['exterior_probability']:.4f}")
    idx_in, vin, cin, pin, min_ = C.packet_mode(L, c, pos, n, k, up, 4.0, 5.0, rs, 'in')
    idx_out, vout, cout, pout, mout = C.packet_mode(L, c, pos + np.array([10, 0, 0]), n, k, up, 4.0, 5.0, rs, 'out')
    check("packet restrictions: 'in' entirely inside rs, 'out' entirely outside with clock spreads",
          np.all(np.linalg.norm(cin, axis=1) < rs) and np.all(np.linalg.norm(cout, axis=1) > rs) and mout['T_rms'] > 0 and mout['u_rms'] > mout['T_rms'],
          f"retained in {min_['retained']:.4f}, out {mout['retained']:.4f}, T_rms {mout['T_rms']:.3f}, u_rms {mout['u_rms']:.3f}")
    bw = C.band_weights(v1, coords, pos, 0.0, np.array([-0.8, 0.0, 0.0]), n, int(math.ceil(20.0)))
    check("source band weights: norm 1, upper-band weight above 0.95, outgoing positive frequency above 0.9",
          abs(bw['norm'] - 1) < 1e-10 and bw['upper_band'] > 0.95 and bw['outgoing_positive_frequency'] > 0.9, str({k_: round(v_, 4) for k_, v_ in bw.items()}))
    rec = C.Receivers(L, c, dev)
    rec.add_site(np.array([-8, 2, 1]), 'x', 8.3)
    rec.add_packet_group(idx, [v1, v2], [dict(kind='pkt', role='x', r=5.0, omega=0.1, sigma=4.0, res='raw', hel=+1, label='pkt|x|r=5.000|w=0.1|sig=4|res=raw|h=+'),
                                         dict(kind='pkt', role='x', r=5.0, omega=0.1, sigma=4.0, res='raw', hel=-1, label='pkt|x|r=5.000|w=0.1|sig=4|res=raw|h=-')])
    rec.finalize()
    psi = torch.zeros((2, 4, L, L, L), dtype=C.CD, device=dev)
    psi[0].view(4, -1).index_copy_(1, torch.tensor(idx, dtype=torch.int64, device=dev), torch.tensor(v1, dtype=C.CD, device=dev))
    psi[1, 2, c - 8, c + 2, c + 1] = 1.0
    P = rec.project(psi).cpu().numpy()
    check("receiver projections: packet on itself = 1, on the other helicity = 0, site component = 1, site outside the packet = 0",
          abs(P[0, 4] - 1) < 1e-12 and abs(P[0, 5]) < 1e-12 and abs(P[1, 2] - 1) < 1e-12 and abs(P[1, 4]) < 1e-12 and abs(P[1, 0]) < 1e-12)
    geo = C.Geometry(L, c, rs, 4.0, dev)
    nm, hist, sh, bnd, pz, pd = geo.measure(psi)
    check("geometry: norms, shell sums and shares", np.allclose(nm, 1.0) and np.allclose(hist.sum(1), 1.0) and hist[1, 8] == 1.0
          and sh[1, 0] == 1.0 and sh[1, 1] == 0.0 and bnd[1] == 0.0, f"norms {nm}, shell of the site {np.argmax(hist[1])}, shares {sh[1]}")


def test_fft_exact():
    L, c = 97, 48
    for m in (0.0, 0.2):
        for vc in ((0.0, 0.0, 0.0), (0.0, 0.0, -0.5)):
            mode = 'flat' if vc == (0.0, 0.0, 0.0) else 'uniform'
            v_full, _ = C.build_site_flow(L + 4, c + 2, 1.0, None, False, mode=mode, uniform_v=vc)
            v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
            wp, wm = C.bond_weights_np(v)
            dE = 1.01 * C.spectral_bound(m, wp, wm)
            Hop = C.LatticeH(m, wp, wm, dev, ghost_np=C.ghost_weights_np(v_full, L))
            psi0 = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
            psi0[0, 0, c, c, c] = 1.0
            tau = 10.0
            n, tr, Cm, Ik = C.cheb_coefficients(dE, tau, 0.5)
            proj = lambda phi: phi[:, 0, c, c, c].reshape(1, 1)
            psi, P, R = C.cheb_sweep(Hop, psi0, dE, Cm[-1], n, proj)
            ref = C.free_propagate_fft(psi0, tau, m, vc)
            err = (psi - ref).abs().max().item()
            check(f"Chebyshev = exact Fourier propagator (m={m}, v={vc})", err < 1e-11, f"max diff {err:.2e}, n={n}")
            A = (torch.tensor(Cm, dtype=C.CD, device=dev) @ P.reshape(n + 1, -1))[-1].item()
            check(f"in-sweep readout at tau equals the state (m={m}, v={vc})", abs(A - psi[0, 0, c, c, c].item()) < 1e-12)
            cert = float((Ik[-1] @ R.cpu().numpy()[:, 0]))
            check(f"certificate small while the wave is far from the faces (m={m}, v={vc})", cert < 1e-6, f"eps_B(10) = {cert:.2e}")
            J = C.taylor_terms(dE, 1.0)
            ps = psi0
            for _ in range(10):
                ps, _ = C.taylor_step(Hop, ps, 1.0, J, residual=False)
            errT = (ps - ref).abs().max().item()
            check(f"Taylor = exact Fourier propagator (m={m}, v={vc})", errT < 1e-11, f"max diff {errT:.2e}, J={J}")


def test_cheb_vs_taylor_radial():
    L, rs, Rb, ell, m = 41, 8.0, 4.8, 2.0, 0.2
    c, v, v_full, Hop, dE = make_radial(L, rs, Rb, ell, m)
    rng = np.random.default_rng(1)
    psi0 = torch.tensor(rng.normal(size=(1, 4, L, L, L)) + 1j * rng.normal(size=(1, 4, L, L, L)), dtype=C.CD, device=dev)
    psi0 /= psi0.norm()
    e_in, _ = Hop.energy(psi0)
    tau = 8.0
    n, tr, Cm, Ik = C.cheb_coefficients(dE, tau, 0.5)
    proj = lambda phi: phi[:, 0, c, c, c].reshape(1, 1)
    pc = psi0
    for _ in range(2):
        pc, _P, _R = C.cheb_sweep(Hop, pc, dE, Cm[-1], n, proj, residual=False)
    J = C.taylor_terms(dE, 1.0)
    pt = psi0
    for _ in range(16):
        pt, _ = C.taylor_step(Hop, pt, 1.0, J, residual=False)
    err = (pc - pt).abs().max().item()
    nerr = abs(pc.norm().item() - 1.0)
    e_out, _ = Hop.energy(pc)
    check("Chebyshev = Taylor on the radial field (t = 16)", err < 1e-11, f"max diff {err:.2e}")
    check("unitarity over 16 time units", nerr < 1e-12, f"|norm - 1| = {nerr:.2e}")
    check("energy conservation over 16 time units", abs(e_out.item() - e_in.item()) < 1e-10, f"|E - E0| = {abs(e_out.item() - e_in.item()):.2e}")


def test_drift_identity():
    """Exact identity: the centroid velocity difference between a uniform-flow run and a flat run of the same
    packet equals v_z <cos 2 p_z>_0 (uniform v: h = h_0 + v.Q commutes with h_0; drift +v)."""
    L, c, m, vc, tau = 97, 48, 0.0, (0.0, 0.0, -0.5), 10.0
    n = np.array([0.0, 0.0, 1.0])
    k = 0.6
    up, _ = C.helicity_spinors(k, n, m)
    idx, vals, coords, prob, meta = C.packet_mode(L, c, np.array([0, 0, 0]), n, k, up, 4.0, 5.0, 1.0, 'raw')
    psi0 = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
    psi0[0].view(4, -1).index_copy_(1, torch.tensor(idx, dtype=torch.int64, device=dev), torch.tensor(vals, dtype=C.CD, device=dev))
    zc = torch.arange(L, dtype=C.FD, device=dev) - c

    def centroid(p):
        dens = (p.real ** 2 + p.imag ** 2).sum(1)[0]
        return (dens.sum((0, 1)) * zc).sum().item() / dens.sum().item()

    res = {}
    for mode, vv in (('flat', (0.0, 0.0, 0.0)), ('uniform', vc)):
        v_full, _ = C.build_site_flow(L + 4, c + 2, 1.0, None, False, mode=mode, uniform_v=vv)
        v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
        wp, wm = C.bond_weights_np(v)
        dE = 1.01 * C.spectral_bound(m, wp, wm)
        Hop = C.LatticeH(m, wp, wm, dev)
        nn, tr, Cm, Ik = C.cheb_coefficients(dE, tau, 0.5)
        psi, _, _ = C.cheb_sweep(Hop, psi0, dE, Cm[-1], nn, lambda phi: phi[:, 0, c, c, c].reshape(1, 1), residual=False)
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
    L, c, m = 65, 32, 0.0
    v_full, _ = C.build_site_flow(L + 4, c + 2, 1.0, None, False, mode='flat')
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    wp, wm = C.bond_weights_np(v)
    dE = 1.01 * C.spectral_bound(m, wp, wm)
    Hop = C.LatticeH(m, wp, wm, dev)
    psi = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
    psi[0, 0, c, c, c] = 1.0
    tau = 10.0
    n, tr, Cm, Ik = C.cheb_coefficients(dE, tau, 0.5)
    Cg = torch.tensor(Cm, dtype=C.CD, device=dev)
    proj = lambda phi: phi[:, :, c + 20, c, c].reshape(1, 4)
    series, times = [], []
    t = 0.0
    for _ in range(3):
        psi, P, _ = C.cheb_sweep(Hop, psi, dE, Cm[-1], n, proj, residual=False)
        series.append(torch.einsum('jk,kbr->jbr', Cg, P).cpu().numpy()[:, 0, :])
        times.append(t + tr)
        t += tau
    A = np.concatenate(series)
    tt = np.concatenate(times)
    amp = np.linalg.norm(A, axis=1)
    for tq in (10.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0):
        j = int(np.argmin(np.abs(tt - tq)))
        print(f"[info] flat space, site at distance 20 on [100]: |A|(t={tt[j]:.1f}) = {amp[j]:.3e}")
    i10 = int(np.flatnonzero(amp > 0.1 * amp.max())[0])
    check("flat-space front arrives near t = r = 20 (10 % crossing within [17, 21])", 17.0 <= tt[i10] <= 21.0, f"t_10 = {tt[i10]:.1f}")


def test_certificate():
    """The Duhamel certificate must bound the actual open-box error: 33^3 box against the 97^3 box (whose own
    boundary is unreached by t = 24), radial field rs = 8, site source at the centre."""
    rs, Rb, ell, m = 8.0, 4.8, 2.0, 0.0
    cS, _, _, HS, dES = make_radial(33, rs, Rb, ell, m)
    cB, _, _, HB, dEB = make_radial(97, rs, Rb, ell, m)
    psiS = torch.zeros((1, 4, 33, 33, 33), dtype=C.CD, device=dev)
    psiS[0, 0, cS, cS, cS] = 1.0
    psiB = torch.zeros((1, 4, 97, 97, 97), dtype=C.CD, device=dev)
    psiB[0, 0, cB, cB, cB] = 1.0
    tau = 8.0
    nS, trS, CmS, IkS = C.cheb_coefficients(dES, tau, 0.5)
    nB, trB, CmB, IkB = C.cheb_coefficients(dEB, tau, 0.5)
    dummy = lambda phi: phi[:, 0, 0, 0, 0].reshape(1, 1)
    eps, ok_all, info = 0.0, True, []
    for s in range(3):
        psiS, _, R = C.cheb_sweep(HS, psiS, dES, CmS[-1], nS, dummy)
        eps = eps + float(IkS[-1] @ R.cpu().numpy()[:, 0])
        psiB, _, _ = C.cheb_sweep(HB, psiB, dEB, CmB[-1], nB, dummy, residual=False)
        sub = psiB[:, :, cB - 16:cB + 17, cB - 16:cB + 17, cB - 16:cB + 17]
        diff = (psiS - sub).norm().item()
        info.append((8.0 * (s + 1), diff, eps))
        ok_all = ok_all and (diff <= eps)
    check("parent certificate bounds the actual open-box error at t = 8, 16, 24", ok_all and info[0][2] < 1e-3 and info[-1][1] > 1e-6,
          "(t, actual error, certificate) = " + str([(a, f"{b:.2e}", f"{c_:.2e}") for a, b, c_ in info]))


if __name__ == '__main__':
    print(f"[tests] device {dev}, in-place complex axpy {'available' if C.AXPY_OK else 'emulated'}")
    test_stencil_vs_sparse()
    test_kernel_profile()
    test_dispersion_and_rays()
    test_packets_receivers_geometry()
    test_fft_exact()
    test_cheb_vs_taylor_radial()
    test_drift_identity()
    test_flat_light_cone()
    test_certificate()
    print(f"[tests] {len(FAIL)} failures" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
