# m3r2_engine_v2.py -- coupled Hartree history of manifest Secs. 2-4 (v3 governing). v2: light propagated at the heavy-converged fields from an extrapolated light part; E-residual floor stall rule (mode E_floor, logged).
# Reference (lift): within each symmetry block the continued occupied set is the lowest-Nocc set of
# levels as long as the occupied/empty boundary (the Fermi pair) stays gapped along the straight
# field path; certified by Sturm counts with a Weyl bound ||h(s')-h(s)|| <= |s'-s| ||h_{j+1}-h_j||.
# A path on which the Fermi gap cannot be certified is logged as an escalation event (not repaired).
import time
import numpy as np
import m3r2_chain_v1 as ch
import m3r2_linalg_v1 as la
import m3r2_rows_v1 as rw

GAMMA1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
GAMMA2 = 1.0 - 2.0 * GAMMA1


class Chain:
    def __init__(self, kind, ns, kch, m, weight, a=1.0):
        self.kind, self.ns, self.kch, self.m, self.weight = kind, ns, kch, m, weight
        if kind == 'heavy':
            self.H0, self.Q, self.site_of, self.r = ch.heavy_h0_Q_bands(ns, kch, m, a)
            self.bw, self.b, self.factor = 4, 4, 1.0
        elif kind == 'light':
            self.H0, self.Q, self.site_of, self.r = ch.light_sector_h0_Q_bands(ns, kch, a)
            self.bw, self.b, self.factor = 2, 2, 2.0
        else:
            raise ValueError(kind)
        self.dim = self.H0.shape[1]
        self.Psi = None          # orbitals: C = Psi Psi^dag (sector + for the light)
        self.Nocc = None         # reference count (lowest-Nocc lift)
        self.PB = None           # band of P at the current accepted fields
        self.HB = None           # band of h at the current accepted fields
        self.refinfo = None
        self.quad = None

    def hband(self, Nbar, vbar):
        return ch.weighted_band(self.H0, self.Q, self.site_of,
                                np.ascontiguousarray(Nbar[:self.ns]), np.ascontiguousarray(vbar[:self.ns]))


def absbound(dHB):
    """Max absolute row sum of a Hermitian band (bounds the operator norm)."""
    bw1, dim = dHB.shape
    rad = np.abs(dHB[0]).astype(float)
    for d in range(1, bw1):
        av = np.abs(dHB[d, :dim - d])
        rad[:dim - d] += av
        rad[d:] += av
    return float(np.max(rad))


def fermi_pair(D, U, Nocc, lo, hi, tol=1e-13):
    """eps_{Nocc-1} (top occupied) and eps_{Nocc} (bottom empty), 0-based ascending."""
    e_occ = la.eig_by_index(D, U, Nocc - 1, lo, hi, tol=tol)
    e_emp = la.eig_by_index(D, U, Nocc, lo, hi, tol=tol)
    return e_occ, e_emp


def projector_band(HB, b, Nocc, ds=0.25, tail=39.0, pair=None):
    """Band (offsets 0..bw) of P = 1_{(-inf,mu)}(h), mu in the Fermi gap above level Nocc-1."""
    D, U = la.band_to_blocks(HB, b)
    lo, hi = la._gersh(HB)
    if pair is None:
        e_occ, e_emp = fermi_pair(D, U, Nocc, lo - 1e-6, hi + 1e-6)
    else:
        e_occ, e_emp = pair
    mu = 0.5 * (e_occ + e_emp)
    g = 0.5 * (e_emp - e_occ)
    wmax = max(abs(lo - mu), abs(hi - mu)) * 1.05
    t, cs, lin_c = la.sign_quadrature(g, wmax, ds=ds, tail=tail)
    zs = (mu + 1j * t).astype(np.complex128)
    SD, SU = la.sign_band_rgf(D, U, zs, cs.astype(np.float64), float(lin_c), float(mu))
    bw = HB.shape[0] - 1
    SB = la.blocks_to_band(SD, SU, bw)
    PB = -0.5 * SB
    PB[0] += 0.5
    info = dict(mu=mu, half_gap=g, e_occ=e_occ, e_emp=e_emp, npole=len(t), lo=lo, hi=hi)
    return PB, info, (D, U, zs, cs, lin_c, mu)


def projector_band_tangent(quad, dHB, b):
    """Eq. dm-contact route: dP band along dh at fixed occupations (tangent RGF of the quadrature)."""
    D, U, zs, cs, lin_c, mu = quad
    dD, dU = la.band_to_blocks(dHB, b)
    SD, SU = la.sign_band_tangent_rgf(D, U, dD, dU, zs, cs.astype(np.float64), float(lin_c))
    bw = dHB.shape[0] - 1
    return -0.5 * la.blocks_to_band(SD, SU, bw)


def certify_path(HB0, HB1, b, Nocc, pair0, max_samples=200000):
    """Weyl/Sturm certificate that the Fermi pair stays separated along h(s) = (1-s) h0 + s h1.
    Returns dict(ok, nsamples, min_half_gap, s_fail, delta)."""
    dHB = HB1 - HB0
    delta = absbound(dHB)
    a, bb = pair0
    if delta == 0:
        return dict(ok=True, nsamples=0, min_half_gap=0.5 * (bb - a), s_fail=None, delta=0.0)
    s = 0.0
    nsamp = 0
    gmin = 0.5 * (bb - a)
    while s < 1.0:
        rad = (bb - a) / (2.0 * delta)
        if rad <= 1e-10:
            return dict(ok=False, nsamples=nsamp, min_half_gap=gmin, s_fail=s, delta=delta)
        snew = min(1.0, s + 0.9 * rad)
        HBs = (1.0 - snew) * HB0 + snew * HB1
        D, U = la.band_to_blocks(HBs, b)
        step_rad = delta * (snew - s) * 1.01 + 1e-14
        e_occ = la.eig_by_index(D, U, Nocc - 1, a - step_rad, bb + step_rad, tol=1e-12)
        e_emp = la.eig_by_index(D, U, Nocc, a - step_rad, bb + step_rad, tol=1e-12)
        nsamp += 1
        a, bb = e_occ, e_emp
        gmin = min(gmin, 0.5 * (bb - a))
        s = snew
        if nsamp > max_samples:
            return dict(ok=False, nsamples=nsamp, min_half_gap=gmin, s_fail=s, delta=delta)
    return dict(ok=True, nsamples=nsamp, min_half_gap=gmin, s_fail=None, delta=delta)


class Fields:
    """Rows on the dense grid from site sources (common radii); back-averaged site fields."""

    def __init__(self, kern, G):
        self.k, self.G = kern, G

    def solve(self, rho, J):
        k = self.k
        rhol, Jl = k.fwd(rho, J)
        tr, tt = k.tail_terms(rho, J)
        M, w, s, E, flag = rw.rows_forward(k.y, k.hd, rhol, Jl, self.G, tr, tt)
        N, v, lam = rw.rows_adjoint(k.y, k.hd, M, w, Jl, self.G, tt)
        Nb, vb = k.adj(N, v, M[-1], self.G)
        return dict(rhol=rhol, Jl=Jl, M=M, w=w, s=s, E=E, flag=flag, N=N, v=v, Nb=Nb, vb=vb,
                    lam=lam, tail=(tr, tt))


def chain_sources(chain, DB, V):
    """Per-site source contributions of one field: w_f * factor * Tr(Delta e_n) / V_n, likewise J."""
    tr_e = ch.alloc_trace(DB, chain.H0, chain.site_of, chain.ns)
    tr_p = ch.alloc_trace(DB, chain.Q, chain.site_of, chain.ns)
    f = chain.weight * chain.factor
    return f * tr_e / V[:chain.ns], f * tr_p / V[:chain.ns]


class Anderson:
    def __init__(self, m=6, beta=1.0):
        self.m, self.beta = m, beta
        self.X, self.F = [], []

    def step(self, x, g):
        f = g - x
        self.X.append(x.copy()); self.F.append(f.copy())
        if len(self.X) > self.m + 1:
            self.X.pop(0); self.F.pop(0)
        if len(self.X) == 1:
            return x + self.beta * f
        dX = np.array([self.X[i + 1] - self.X[i] for i in range(len(self.X) - 1)]).T
        dF = np.array([self.F[i + 1] - self.F[i] for i in range(len(self.F) - 1)]).T
        gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
        return x + self.beta * f - (dX + self.beta * dF) @ gam


def field_diff(fa, fb):
    return max(float(np.max(np.abs(fa['N'] - fb['N']))), float(np.max(np.abs(fa['v'] - fb['v']))))


class History:
    """Chains, kernel, fields at the accepted time; coupled self-consistent midpoint substeps."""

    def __init__(self, chains, kern, G, log=print):
        self.chains, self.k, self.G = chains, kern, G
        self.fs = Fields(kern, G)
        self.nsmax = kern.ns
        self.log = log
        self.t = 0.0
        self.src = None
        self.fld = None
        self.stats = []
        self._last_rate = None

    def init_from_state(self, rho, J):
        self.src = (rho.copy(), J.copy())
        self.fld = self.fs.solve(rho, J)
        for c in self.chains:
            c.HB = c.hband(self.fld['Nb'], self.fld['vb'])

    def _eval_chain(self, c, fh, f1, dt):
        HBh = c.hband(fh['Nb'], fh['vb'])
        Psi1, K = la.propagate(HBh, c.Psi, dt)
        HB1 = c.hband(f1['Nb'], f1['vb'])
        PB1, info, quad = projector_band(HB1, c.b, c.Nocc)
        CB = ch.band_from_orbitals(Psi1, c.bw)
        r_, j_ = chain_sources(c, CB - PB1, self.k.V)
        return r_, j_, (Psi1, PB1, info, quad, HB1, K)

    def set_light_part(self, lp):
        """Light contribution (site sources rho|J concatenated) of the accepted state (restart use)."""
        self._lpart = lp.copy()

    def _inner(self, x, xj, lpart, dt, tol, tolE_floor, stall_n, maxit, res, heavy, outer, hist, verbose):
        ns = self.nsmax
        rho_j, J_j = xj[:ns], xj[ns:]
        acc = Anderson(m=6)
        best_dE = np.inf
        since_best = 0
        nprop = 0
        for it in range(maxit):
            fh = self.fs.solve(0.5 * (rho_j + x[:ns]), 0.5 * (J_j + x[ns:]))
            f1 = self.fs.solve(x[:ns], x[ns:])
            g = lpart.copy()
            for i in heavy:
                r_, j_, pack = self._eval_chain(self.chains[i], fh, f1, dt)
                nprop += 1
                g[:self.chains[i].ns] += r_
                g[ns:ns + self.chains[i].ns] += j_
                res[i] = pack
            fg1 = self.fs.solve(g[:ns], g[ns:])
            fgh = self.fs.solve(0.5 * (rho_j + g[:ns]), 0.5 * (J_j + g[ns:]))
            dfield = max(field_diff(fg1, f1), field_diff(fgh, fh))
            dE = abs(fg1['E'] - f1['E']) / abs(fg1['E'])
            hist.append((outer, it, dfield, dE))
            if verbose:
                self.log(f'   outer {outer} inner {it} dfield {dfield:.3e} dE {dE:.3e}')
            if dfield < tol and dE < tol:
                return g, nprop, 'tol', dfield, dE
            if dE < best_dE * 0.7:
                best_dE = dE; since_best = 0
            else:
                since_best += 1
            if dfield < tol and dE < tolE_floor and since_best >= stall_n:
                return g, nprop, 'E_floor', dfield, dE
            x = acc.step(x, g)
        raise RuntimeError(f'inner fixed point not converged: dfield {dfield:.3e} dE {dE:.3e}')

    def substep(self, dt, tol=1e-12, tolE_floor=1e-11, stall_n=4, maxit=60, verbose=False):
        """Self-consistent midpoint substep (manifest v2 Sec. 4). Unknown x = site sources at j+1.
        u_{j+1} = Rows[x], u_{1/2} = Rows[(src_j + x)/2]. The heavy chains are iterated (Anderson) with
        the light contribution frozen at its extrapolated value; the light is then propagated at the
        converged midpoint fields and its reference lifted at the converged u_{j+1}; the heavy loop is
        re-run with the new light contribution, and the light re-propagated only if the fields moved by
        >= tol. Acceptance: field residual < tol (sup norm, dense grid, u_{j+1} and u_{1/2}) and relative
        E residual < tol; or, when the E residual has stalled at its floating-point floor for stall_n
        iterations, field residual < tol and E residual < tolE_floor (mode 'E_floor', logged)."""
        T0 = time.time()
        rho_j, J_j = self.src
        ns = self.nsmax
        heavy = [i for i, c in enumerate(self.chains) if c.kind == 'heavy']
        light = [i for i, c in enumerate(self.chains) if c.kind == 'light']
        xj = np.concatenate([rho_j, J_j])
        x = xj.copy() if self._last_rate is None else xj + dt * self._last_rate
        lp0 = getattr(self, '_lpart', None)
        if lp0 is None:
            lp0 = np.zeros(2 * ns)
        lrate = getattr(self, '_lrate', None)
        lpart = lp0 + (dt * lrate if lrate is not None else 0.0)
        res = [None] * len(self.chains)
        nprop_h = nprop_l = 0
        hist = []
        modes = []
        used = None
        outer = 0
        while True:
            outer += 1
            x, nph, mode, dfield, dE = self._inner(x, xj, lpart, dt, tol, tolE_floor, stall_n, maxit, res,
                                                   heavy, outer, hist, verbose)
            nprop_h += nph
            modes.append(mode)
            if not light:
                break
            fh = self.fs.solve(0.5 * (rho_j + x[:ns]), 0.5 * (J_j + x[ns:]))
            f1 = self.fs.solve(x[:ns], x[ns:])
            if used is not None:
                dl = max(field_diff(fh, used[0]), field_diff(f1, used[1]))
                dEl = abs(f1['E'] - used[1]['E']) / abs(f1['E'])
                hist.append((outer, -1, dl, dEl))
                if verbose:
                    self.log(f'  outer {outer}: fields moved {dl:.3e} (E rel {dEl:.3e}) since the light propagation')
                if dl < tol and dEl < tolE_floor:
                    break
            if outer > 8:
                raise RuntimeError('outer light loop not converged')
            lpart = np.zeros(2 * ns)
            for i in light:
                r_, j_, pack = self._eval_chain(self.chains[i], fh, f1, dt)
                nprop_l += 1
                lpart[:self.chains[i].ns] += r_
                lpart[ns:ns + self.chains[i].ns] += j_
                res[i] = pack
            used = (fh, f1)
        # accept
        self._last_rate = (x - xj) / dt
        if light:
            self._lrate = (lpart - lp0) / dt
            self._lpart = lpart
        Ks = []
        for i, c in enumerate(self.chains):
            Psi1, PB1, info, quad, HB1, K = res[i]
            c.HB_prev = c.HB
            c.Psi, c.PB, c.refinfo, c.quad, c.HB = Psi1, PB1, info, quad, HB1
            Ks.append(K)
        self.src = (x[:ns].copy(), x[ns:].copy())
        self.fld = self.fs.solve(x[:ns], x[ns:])
        self.t += dt
        inner_res = [h_ for h_ in hist if h_[1] >= 0]
        st = dict(t=self.t, dt=dt, outer=outer, inner=len(inner_res), nprop_h=nprop_h, nprop_l=nprop_l,
                  wall=time.time() - T0, E=self.fld['E'], cheb_terms=Ks, modes=modes,
                  final_dfield=inner_res[-1][2], final_dE=inner_res[-1][3],
                  light_moved=[h_[2] for h_ in hist if h_[1] < 0])
        self.stats.append(st)
        return st

    def certify_last(self):
        """Fermi-pair certificate for every chain along the accepted path of the last substep."""
        out = []
        for c in self.chains:
            pair0 = getattr(c, 'pair_prev', None)
            if pair0 is None:
                D, U = la.band_to_blocks(c.HB_prev, c.b)
                lo, hi = la._gersh(c.HB_prev)
                pair0 = fermi_pair(D, U, c.Nocc, lo - 1e-6, hi + 1e-6)
            cert = certify_path(c.HB_prev, c.HB, c.b, c.Nocc, pair0)
            c.pair_prev = (c.refinfo['e_occ'], c.refinfo['e_emp'])
            out.append(cert)
        return out

    def yoshida_step(self, tau, **kw):
        return [self.substep(g * tau, **kw) for g in (GAMMA1, GAMMA2, GAMMA1)]
