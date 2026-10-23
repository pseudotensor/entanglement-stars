"""gap10_sources_wl.py -- the wavelength-class sources of GAP10 T7 (busi/notes/GAP10_wavelength_test_v1.md) and their readouts:
the 2 long-wavelength carriers on the preparation field (the incoming-branch root at a Killing energy above m; the outgoing-
branch minimum, in the gap), the positive-band projected packet, the semiclassical (local-symbol) Killing-energy split, the
Chebyshev coefficients of the smoothed threshold filter F_>(h), the Jackson-KPM evaluation of the Killing spectrum from the
stored moments, the box-exactness bound of the moments, and the axis WKB kinematic clock. Imported by gap10_run_v2_wl.py and
gap10_analysis_v3.py. Every function is pure numpy/scipy except nothing: no torch here. The lattice core (gap10_core_v2,
or gap10_core_opt2 which re-exports it) supplies the dispersion, the spinors and the packet geometry.

Conventions (gap10_core_v2): the axial branch H(k) = N eps(k n) - u sum_i n_i sin(2 k n_i)/2 with u = |v| >= 0 the inward
flow; k > 0 is momentum along +n (outward), k < 0 inward (the drag term then raises the Killing energy). The Killing energy
of a mode on the held fields is its eigenvalue under h (h generates the record's time translation while the fields are
held), so "above m" means spectral weight of the source on eigenvalues of h above m.
"""
import math
import numpy as np
import scipy.optimize as spo
import scipy.special as sps

import gap10_core_v2 as C

DEFAULT_E_IN = 0.26      # the incoming-branch label: 1.3 m at ma 0.2 (HORIZON v2's near-threshold long-wavelength class)


# ------------------------------------------------------------------ carriers on the preparation field
def _bracket_root(f, lo, hi, n=4001):
    ks = np.linspace(lo, hi, n)
    g = np.array([f(k) for k in ks])
    idx = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
    if idx.size == 0:
        return None, 0
    i = idx[-1]   # the root nearest k = 0 from below (lo < 0 < hi never happens here: lo, hi have one sign)
    return spo.brentq(f, ks[i], ks[i + 1], xtol=1e-14, rtol=1e-14), int(idx.size)


def incoming_root(E, u, nhat, m, N=1.0):
    """The incoming-branch carrier k < 0 with H(k; u, N) = E along nhat: the root nearest k = 0 on (-k_max, 0).
    Returns (k, number of sign changes). Raises ValueError if the branch does not reach E."""
    nhat = np.asarray(nhat, dtype=float)
    kmax = (math.pi / 2.0) / float(np.max(np.abs(nhat)))
    f = lambda k: float(C.disp_omega(k, u, nhat, m, N)[0]) - E
    k, nchg = _bracket_root(f, -kmax + 1e-9, -1e-9)
    if k is None:
        raise ValueError(f"no incoming-branch root: E={E}, u={u:.4f}, N={N:.4f}, m={m}")
    return k, nchg


def outgoing_minimum(u, nhat, m, N=1.0, kcap=None):
    """The outgoing-branch minimum (k_min, E_min) over 0 < k < pi/4 / max|n_i| (the long-wavelength half of the branch):
    the carrier of the sub-threshold source (zero group velocity on the preparation field, energy N m or below)."""
    nhat = np.asarray(nhat, dtype=float)
    kc = (math.pi / 4.0) / float(np.max(np.abs(nhat))) if kcap is None else kcap
    kk = np.linspace(1e-4, kc, 8001)
    hh = C.disp_omega(kk, u, nhat, m, N)
    i = int(np.argmin(hh))
    lo, hi = kk[max(i - 1, 0)], kk[min(i + 1, kk.size - 1)]
    res = spo.minimize_scalar(lambda k: float(C.disp_omega(k, u, nhat, m, N)[0]), bounds=(lo, hi), method='bounded', options=dict(xatol=1e-12))
    return float(res.x), float(res.fun)


# ------------------------------------------------------------------ the positive-band projected packet
def packet_mode_band(L, c, pos, nhat, k, spinor, sigma, cut, rs, m, restriction='raw', common_half=None, project=True):
    """gap10_core_v2.packet_mode, then (project=True) the projection onto the positive band of h_0 momentum by momentum on
    the packet's bounding cube (the eigenvectors of the local symbol with flow are those of h_0(p): the flow term is a
    scalar q(p).v and N multiplies h_0), the ball/box/restriction mask applied again, and renormalization. Returns the
    same tuple (flat indices, values (4, n), coords, probabilities, meta) with meta['band_projected'] and the upper-band
    weights before and after. The index set is the one of packet_mode (common to every packet at this site and width)."""
    idx, values, coords, prob, meta = C.packet_mode(L, c, pos, nhat, k, spinor, sigma, cut, rs, restriction=restriction, common_half=common_half)
    hw = int(math.ceil(cut * sigma))
    v_vec = np.zeros(3)
    bw0 = C.band_weights(values, coords, pos, m, v_vec, np.asarray(nhat, dtype=float), hw, 1.0)
    meta.update(band_projected=False, upper_band_raw=bw0['upper_band'])
    if not project:
        return idx, values, coords, prob, meta
    n = 2 * hw + 1
    cube = np.zeros((4, n, n, n), dtype=np.complex128)
    rel = (np.asarray(coords, dtype=int) - np.asarray(pos, dtype=int)) + hw
    cube[:, rel[:, 0], rel[:, 1], rel[:, 2]] = values
    phi = np.fft.fftn(cube, axes=(1, 2, 3), norm='ortho')
    p = 2.0 * math.pi * np.fft.fftfreq(n)
    P = list(np.meshgrid(p, p, p, indexing='ij'))
    d = [np.sin(x) for x in P]
    eps = np.sqrt(m * m + d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    zero = eps == 0
    safe = np.where(zero, 1.0, eps)
    h0phi = m * C.BETA_DIAG[:, None, None, None] * phi
    for i in range(3):
        for (row, col, val) in C.ALPHA_NZ[i]:
            h0phi[row] += val * d[i] * phi[col]
    plus = 0.5 * (phi + np.where(zero, 0.0, h0phi / safe))      # P_+ phi; at eps = 0 (massless only) phi/2
    cube_p = np.fft.ifftn(plus, axes=(1, 2, 3), norm='ortho')
    vals_p = cube_p[:, rel[:, 0], rel[:, 1], rel[:, 2]]          # back on the retained sites (the mask re-applied)
    nrm = math.sqrt(float(np.sum(np.abs(vals_p) ** 2)))
    if nrm == 0.0:
        raise ValueError("empty band-projected packet")
    vals_p = vals_p / nrm
    prob_p = (np.abs(vals_p) ** 2).sum(0)
    bw1 = C.band_weights(vals_p, coords, pos, m, v_vec, np.asarray(nhat, dtype=float), hw, 1.0)
    r = np.linalg.norm(coords, axis=1)
    mean = prob_p @ coords
    meta.update(band_projected=True, upper_band_projected=bw1['upper_band'], projection_retained=nrm * nrm,
                exterior_probability=float(prob_p[r > rs].sum()), interior_probability=float(prob_p[r < rs].sum()),
                mean_r=float(prob_p @ r), mean_pos=[float(x) for x in mean],
                position_std=[float(x) for x in np.sqrt(prob_p @ (coords - mean) ** 2)])
    return idx, vals_p, coords, prob_p, meta


def exterior_tails(coords, prob, radii):
    """The packet's initial weight beyond each radius (the initial exterior tail of a column, review of HORIZON v2)."""
    r = np.linalg.norm(coords, axis=1)
    return {f"{float(R):.4f}": float(prob[r > R].sum()) for R in radii}


# ------------------------------------------------------------------ the semiclassical (local-symbol) Killing split
def band_tops(m):
    """The band tops of the free operator: b1 = sqrt(m^2 + 1) along an axis (the radial chain's, HORIZON's bins), b3 = sqrt(m^2 + 3)
    of the cubic operator (the top of its essential spectrum on the frozen exterior; WEB40G P19 item 1: the flow lifts the band top,
    so Killing weight above b3 is bound content, and the unbound class is (m, b3))."""
    return math.sqrt(m * m + 1.0), math.sqrt(m * m + 3.0)


def five_bins(E, w, m, b):
    """Weights of the distribution (E, w) in the Killing bins (-inf, -b), (-b, -m), (-m, m), (m, b), (b, inf)."""
    E = np.asarray(E, dtype=float)
    w = np.asarray(w, dtype=float)
    edges = [-np.inf, -b, -m, m, b, np.inf]
    names = ['below_-b', '-b_to_-m', 'gap', 'm_to_b', 'above_b']
    return {nm: float(w[(E > lo) & (E <= hi)].sum()) for nm, lo, hi in zip(names, edges[:-1], edges[1:])}


def semiclassical_split(pos, nhat, k, sigma, u_fn, N_fn, m, nsamp=200000, seed=1, shift=0.05):
    """The Killing-energy distribution of the packet by the local symbol: x ~ |f|^2 (the Gaussian of standard deviation
    sigma per axis), p ~ |f^|^2 (standard deviation 1/(2 sigma) per axis about k nhat), E(x, p) = N(r) eps(p) + v(x).q(p)
    with v = -u(r) x/r (radial inflow). Returns the mean and spread of E, the weights above m, in the gap, below -m, the
    weights above m + shift and below m - shift, and the lattice-scale fraction (folded momentum above pi/4). An
    estimate (phase-space, not the spectral measure); the KPM split is the computed one."""
    rng = np.random.default_rng(seed)
    nhat = np.asarray(nhat, dtype=float)
    x = np.asarray(pos, dtype=float) + rng.normal(0.0, sigma, (nsamp, 3))
    p = k * nhat + rng.normal(0.0, 1.0 / (2.0 * sigma), (nsamp, 3))
    r = np.maximum(np.linalg.norm(x, axis=1), 1e-9)
    Nx = np.asarray(N_fn(r), dtype=float)
    ux = np.asarray(u_fn(r), dtype=float)
    sp = np.sin(p)
    eps = np.sqrt(m * m + (sp ** 2).sum(1))
    q = 0.5 * np.sin(2.0 * p)
    E = Nx * eps + ((-(ux / r)[:, None] * x) * q).sum(1)
    fold = np.max(np.minimum(np.abs(p), math.pi - np.abs(p)), axis=1)
    b1, b3 = band_tops(m)
    return dict(E_mean=float(E.mean()), E_std=float(E.std()), w_above=float(np.mean(E > m)), w_gap=float(np.mean(np.abs(E) < m)),
                w_below=float(np.mean(E < -m)), w_above_shift=float(np.mean(E > m + shift)), w_below_shift=float(np.mean(E < m - shift)),
                shift=float(shift), lattice_scale=float(np.mean(fold > math.pi / 4.0)), nsamp=int(nsamp),
                bins_b3=five_bins(E, np.full(E.size, 1.0 / E.size), m, b3), bins_b1=five_bins(E, np.full(E.size, 1.0 / E.size), m, b1), b1=b1, b3=b3)


# ------------------------------------------------------------------ the threshold filter and the Killing moments
def filter_coefficients(dE, m, s, tol=1e-10, nmax=1200, M=16384):
    """Chebyshev coefficients c_k (k = 0 .. n_f) of F_>(E) = (1 + erf((E - m)/(sqrt2 s)))/2 (m here is any threshold: m or the band top b) as a function of x = E/dE on
    [-1, 1], by Gauss-Chebyshev quadrature on M nodes; n_f is the smallest order beyond which every coefficient is below
    tol (the Gaussian smoothing makes them decay as exp(-(k s/dE)^2/2)). F_<(h) = 1 - F_>(h), so psi_< = psi - psi_>."""
    th = math.pi * (np.arange(M) + 0.5) / M
    fx = 0.5 * (1.0 + sps.erf((dE * np.cos(th) - m) / (math.sqrt(2.0) * s)))
    kk = np.arange(nmax + 1)
    ck = (2.0 / M) * (np.cos(np.outer(kk, th)) @ fx)
    ck[0] *= 0.5
    big = np.flatnonzero(np.abs(ck) >= tol)
    n_f = int(big[-1]) + 1 if big.size else 1
    n_f = min(max(n_f, 2), nmax)
    return ck[:n_f + 1].astype(np.complex128), n_f


def moments_box_bound(resid, dE):
    """The accumulated box error of the Chebyshev vectors from the face residuals r_j = ||B T_j psi||: with T_k^box - T_k^inf
    obeying the 3-term recursion driven by -2 B T_j psi/dE, |U_n| <= n + 1 gives ||T_k^box psi - T_k^inf psi|| <= (2/dE)
    sum_{j<k} (k - j) r_j (a bound on the moments' box error |mu_k^box - mu_k^inf| as well, since |<psi|delta>| <= ||delta||).
    The residuals are the box's own (the infinite ones differ at second order in the same quantity)."""
    resid = np.asarray(resid, dtype=float)
    n = resid.size
    out = np.zeros(n)
    for k in range(1, n):
        j = np.arange(k)
        out[k] = (2.0 / dE) * float(np.sum((k - j) * resid[:k]))
    return out


def kpm_split(mu, dE, m, n_use=None, ntheta=40001):
    """The Jackson-KPM spectral density of psi on the box's h from the moments mu_k = <psi|T_k(h/dE)|psi>, k = 0 .. n, and
    its integrals: the weights above m, in (-m, m), below -m, the weights within +-delta of the 2 thresholds (delta = pi dE/N,
    the kernel's width: the ambiguous part of the split at this resolution), the mean energy mu_1 dE and the density's
    normalization. x = cos(theta): rho(x) dx = (1/pi)[g_0 mu_0 + 2 sum_k g_k mu_k cos(k theta)] dtheta, integrated on a
    fine theta grid (the integrand is a trigonometric polynomial of degree N)."""
    mu = np.asarray(mu, dtype=float)
    N = int(mu.size - 1 if n_use is None else min(n_use, mu.size - 1))
    kk = np.arange(N + 1)
    a = math.pi / (N + 1.0)
    g = ((N + 1 - kk) * np.cos(kk * a) + np.sin(kk * a) / math.tan(a)) / (N + 1.0)   # the Jackson kernel
    th = np.linspace(0.0, math.pi, ntheta)
    fac = np.where(kk == 0, 1.0, 2.0) * g * mu[:N + 1]
    dens = (fac @ np.cos(np.outer(kk, th))) / math.pi            # rho(theta) = rho(x) |dx/dtheta|
    E = dE * np.cos(th)
    w = np.full(ntheta, th[1] - th[0])          # trapezoid weights (half at the ends): exact for the trigonometric polynomial
    w[0] *= 0.5
    w[-1] *= 0.5
    tot = float(np.sum(dens * w))
    delta = math.pi * dE / N
    sel = lambda mask: float(np.sum(dens[mask] * w[mask]))
    b1, b3 = band_tops(m)
    return dict(n_moments=N, resolution=delta, norm=tot, w_above=sel(E > m), w_gap=sel(np.abs(E) < m), w_below=sel(E < -m),
                w_trans_plus=sel(np.abs(E - m) < delta), w_trans_minus=sel(np.abs(E + m) < delta),
                bins_b3=five_bins(E, dens * w, m, b3), bins_b1=five_bins(E, dens * w, m, b1), b1=b1, b3=b3,
                w_trans_b3=sel(np.abs(E - b3) < delta),
                E_mean=float(mu[1] * dE) if mu.size > 1 else float('nan'),
                E_rms=float(dE * math.sqrt(max(0.5 * (mu[2] + mu[0]), 0.0))) if mu.size > 2 else float('nan'),
                density_E=E, density=dens / np.where(np.abs(np.sin(th)) > 1e-12, dE * np.abs(np.sin(th)), np.inf))


# ------------------------------------------------------------------ the axis WKB kinematic clock
def kinematic_clock(E, r_src, radii, u_fn, N_fn, nhat, m, dr=0.1, vg_floor=2e-3):
    """The flight time of Killing energy E launched inward at r_src on the axis nhat: the inward branch (k < 0) to the
    turning radius (where that branch no longer reaches E) or to the centre, then the outgoing branch (k > 0) outward to
    each radius in radii, int dr/|v_g| on both legs (the trapezoid at step dr, |v_g| floored at vg_floor at the turning
    point, where the integrand is integrable but the trapezoid is not). The fields are the allocated axis fields (their
    continuation u = u(1) r, N = N(1) below the first node is the core's). An estimate on the axis, not a 3-dimensional
    ray; the receivers see a wave. Returns dict(t_in, r_turn (None if the centre is reached), t_total={R: t}, ok={R: bool})."""
    nhat = np.asarray(nhat, dtype=float)
    kmax = (math.pi / 2.0) / float(np.max(np.abs(nhat)))

    def fields(r):
        return float(np.asarray(u_fn(r), dtype=float).reshape(-1)[0]), float(np.asarray(N_fn(r), dtype=float).reshape(-1)[0])

    def vg_in(r):
        u, N = fields(r)
        f = lambda k: float(C.disp_omega(k, u, nhat, m, N)[0]) - E
        k, _ = _bracket_root(f, -kmax + 1e-9, -1e-9, n=801)
        if k is None:
            return None
        return abs(float(C.disp_vg(k, u, nhat, m, N)[0]))

    def vg_out(r):
        u, N = fields(r)
        try:
            k = C.outgoing_root(E, u, nhat, m, N)
        except ValueError:
            return None
        return float(C.disp_vg(k, u, nhat, m, N)[0])

    rr = np.arange(r_src, 0.05, -dr)
    t_in, r_turn = 0.0, None
    for a, b in zip(rr[:-1], rr[1:]):
        v = vg_in(0.5 * (a + b))
        if v is None:
            r_turn = float(0.5 * (a + b))
            break
        t_in += dr / max(v, vg_floor)
    # the outward leg starts where the outgoing branch first reaches E (after a turning point the band top must rise above E
    # again: the first radius with a root, searched outward from r_turn; the stretch between carries no flight time here)
    r0 = (r_turn + dr) if r_turn is not None else 0.05
    r_start = None
    for r in np.arange(r0, max(radii) + dr, dr):
        if vg_out(r) is not None:
            r_start = float(r)
            break
    out, ok = {}, {}
    for R in radii:
        if r_start is None or r_start > float(R):
            out[f"{float(R):.4f}"], ok[f"{float(R):.4f}"] = float('nan'), False
            continue
        rr2 = np.arange(r_start, float(R) + dr, dr)
        t_out, good = 0.0, True
        for a, b in zip(rr2[:-1], rr2[1:]):
            v = vg_out(0.5 * (a + b))
            if v is None or v <= 0.0:
                good = False
                break
            t_out += dr / max(v, vg_floor)
        out[f"{float(R):.4f}"] = float(t_in + t_out) if good else float('nan')
        ok[f"{float(R):.4f}"] = bool(good)
    return dict(t_in=float(t_in), r_turn=r_turn, r_start_out=r_start, t_total=out, ok=ok)


def reach_radius(E, u_fn, N_fn, nhat, m, r_max, dr=0.25):
    """The largest radius at which the outgoing branch has a mode of Killing energy E (min_k H(k; r) <= E): the farthest
    excursion of sub-threshold content on that branch (bound content turns around there). nan if none up to r_max."""
    nhat = np.asarray(nhat, dtype=float)
    kmax = (math.pi / 2.0) / float(np.max(np.abs(nhat)))
    kk = np.linspace(1e-4, kmax, 600)
    rr = np.arange(1.0, r_max + dr, dr)
    reach = float('nan')
    for r in rr:
        u = float(np.asarray(u_fn(r), dtype=float).reshape(-1)[0])
        N = float(np.asarray(N_fn(r), dtype=float).reshape(-1)[0])
        if float(np.min(C.disp_omega(kk, u, nhat, m, N))) <= E:
            reach = float(r)
    return reach
