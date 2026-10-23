#!/usr/bin/env python3
"""
firstlaw_selftest_v1.py -- manufactured checks for firstlaw_records_v1.py

Checks (each prints PASS/FAIL with the numbers; exit code 1 on any FAIL):

  S1  static Schwarzschild-like stored profile (N = 1, v = -sqrt(2GM/r), rho = J = 0,
      E = M): the 3 columns and the residual are 0 between identical records and below
      --tol-static-noisy when one record carries 1e-10 noise; kappa_formula = 1/(4GM),
      the refit root = 2GM and the refit slope = 1/(4GM).
  F1  a manufactured quasi-static accretion family that solves the declared rows exactly
      (M(r, lam) and ln N(r, lam) analytic; rho, J defined by the rows): the identity
      closes to the discretization order.  Per pair |W_script - W_exact| and the residual
      (Simpson W) fall by 2^order when the record step halves (order >= --order-min);
      the cumulative residual by 2^(order-1); the finest residual is below
      --tol-closure-rel relative to the rms delta E.  W_exact is the nested quadrature
      of the exact one-form (Gauss-Legendre in lam, Simpson on a 4x finer radial grid).
      Also: kappa_formula = the analytic kappa_T to --tol-kappa-rel, R_refit = the
      analytic root to --tol-root, M_collar_fields = E to --tol-E-rel, exterior row
      defects below --tol-rows, J sign +1 found, the identity with exact lam-quadrature
      of both terms closes to --tol-identity, the radial rules behave as expected
      ((W_half - W_trapz)/(W_trapz - W_simpson) in [--rule-ratio-min, --rule-ratio-max]).
  J1  the same family with J_tot stored with the opposite sign: the audit picks -1 and
      kappa_formula still matches.
  R1  the root finder on manufactured N + v: a single zero (root and slope), four zeros
      (two outer-type -> not simple), a near tangency (no zero, flagged).
  X1  the exclusion logic: a root appearing, a regime flag, a stored tangency.

Writes the manufactured records (json + npz in the engine's layout) and a summary json
to --out.  CPU only; at most 8 threads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firstlaw_records_v1 as fl  # noqa: E402  (sets the thread caps from --threads)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

try:
    from scipy.special import erfc as _erfc
except Exception:  # pragma: no cover
    _erfc = np.vectorize(math.erfc)

FOUR_PI = 4.0 * np.pi
SQRT2 = math.sqrt(2.0)


def Phi(x):
    return 0.5 * _erfc(-np.asarray(x, float) / SQRT2)


def phi(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='manufactured checks for firstlaw_records_v1')
    p.add_argument('--out', default='<internal>/firstlaw_selftest', help='output directory')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--n-sites', dest='n_sites', type=int, default=2048, help='sites of the manufactured grid (spacing 1)')
    p.add_argument('--G', type=float, default=0.25)
    p.add_argument('--n-coarse', dest='n_coarse', type=int, default=9, help='records of the coarse family (lam = 0..1)')
    p.add_argument('--fine-factor', dest='fine_factor', type=int, default=2, help='the fine family has (n_coarse-1)*factor + 1 records')
    p.add_argument('--n-gl', dest='n_gl', type=int, default=6, help='Gauss-Legendre nodes per pair for W_exact')
    p.add_argument('--dlam', type=float, default=1e-4, help='centred step for the exact lam-derivatives')
    p.add_argument('--fine-radial-factor', dest='fine_radial_factor', type=int, default=4, help='radial refinement for W_exact')
    p.add_argument('--tol-static', dest='tol_static', type=float, default=1e-12)
    p.add_argument('--tol-static-noisy', dest='tol_static_noisy', type=float, default=1e-6)
    p.add_argument('--tol-kappa-rel', dest='tol_kappa_rel', type=float, default=1e-6)
    p.add_argument('--tol-root', dest='tol_root', type=float, default=1e-5, help='sites')
    p.add_argument('--tol-E-rel', dest='tol_E_rel', type=float, default=1e-9)
    p.add_argument('--tol-rows', dest='tol_rows', type=float, default=1e-3, help='relative exterior row defect on the exact family')
    p.add_argument('--tol-identity', dest='tol_identity', type=float, default=1e-6, help='|dE - int kappa dA/8piG - W_exact| per pair with exact lam-quadrature of both')
    p.add_argument('--tol-closure-rel', dest='tol_closure_rel', type=float, default=1e-3, help='rms residual (Simpson W) over rms delta E at the fine step')
    p.add_argument('--order-min', dest='order_min', type=float, default=2.5, help='minimum observed per-pair convergence order (3 expected)')
    p.add_argument('--order-cum-min', dest='order_cum_min', type=float, default=1.5, help='minimum observed cumulative order (2 expected)')
    p.add_argument('--rule-ratio-min', dest='rule_ratio_min', type=float, default=1.5)
    p.add_argument('--rule-ratio-max', dest='rule_ratio_max', type=float, default=6.0)
    p.add_argument('--tol-root-finder', dest='tol_root_finder', type=float, default=1e-7, help='sites, on the manufactured single zero')
    p.add_argument('--tol-slope-rel', dest='tol_slope_rel', type=float, default=1e-5)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# the manufactured family (solves the declared rows exactly)
# ----------------------------------------------------------------------------
class Family(object):
    """M(r,lam) = (M_inf - mu)(1 - exp(-(r/Lc)^3)) + mu (1 - Phi((r_s - r)/w)) + m_b (1 - exp(-(r/L)^2)),
       ln N(r,lam) = -nu Phi((r_s - r)/w) - nu_h exp(-(r/L)^2); u = -sqrt(2GM/r), v = uN;
       J = -u (ln N)'/(4 pi G r), rho = M'/(4 pi r^2) - u J  (the rows of input 02, exactly)."""

    def __init__(self, G, M_inf0=100.0, dM_inf=-0.5, mu0=10.0, dmu=-4.0, rs0=300.0, drs=-50.0, w=20.0,
                 nu0=0.2, dnu=0.05, nuh0=0.05, dnuh=0.01, mb0=20.0, dmb=2.0, L=100.0, Lc=10.0):
        self.G = G
        self.M_inf0, self.dM_inf, self.mu0, self.dmu = M_inf0, dM_inf, mu0, dmu
        self.rs0, self.drs, self.w = rs0, drs, w
        self.nu0, self.dnu, self.nuh0, self.dnuh = nu0, dnu, nuh0, dnuh
        self.mb0, self.dmb, self.L, self.Lc = mb0, dmb, L, Lc

    def par(self, lam):
        return dict(M_inf=self.M_inf0 + self.dM_inf * lam, mu=self.mu0 + self.dmu * lam, rs=self.rs0 + self.drs * lam,
                    nu=self.nu0 + self.dnu * lam, nuh=self.nuh0 + self.dnuh * lam, mb=self.mb0 + self.dmb * lam)

    def M(self, r, lam):
        p = self.par(lam)
        r = np.asarray(r, float)
        core = (p['M_inf'] - p['mu']) * (1.0 - np.exp(-(r / self.Lc) ** 3))
        shell = p['mu'] * (1.0 - Phi((p['rs'] - r) / self.w))
        halo = p['mb'] * (1.0 - np.exp(-(r / self.L) ** 2))
        return core + shell + halo

    def Mp(self, r, lam):
        p = self.par(lam)
        r = np.asarray(r, float)
        core = (p['M_inf'] - p['mu']) * 3.0 * r * r / self.Lc ** 3 * np.exp(-(r / self.Lc) ** 3)
        shell = p['mu'] / self.w * phi((p['rs'] - r) / self.w)
        halo = p['mb'] * 2.0 * r / self.L ** 2 * np.exp(-(r / self.L) ** 2)
        return core + shell + halo

    def lnN(self, r, lam):
        p = self.par(lam)
        r = np.asarray(r, float)
        return -p['nu'] * Phi((p['rs'] - r) / self.w) - p['nuh'] * np.exp(-(r / self.L) ** 2)

    def lnNp(self, r, lam):
        p = self.par(lam)
        r = np.asarray(r, float)
        return p['nu'] / self.w * phi((p['rs'] - r) / self.w) + p['nuh'] * 2.0 * r / self.L ** 2 * np.exp(-(r / self.L) ** 2)

    def fields(self, r, lam):
        r = np.asarray(r, float)
        M = self.M(r, lam)
        Mp = self.Mp(r, lam)
        N = np.exp(self.lnN(r, lam))
        lnNp = self.lnNp(r, lam)
        u = -np.sqrt(2.0 * self.G * M / r)
        v = u * N
        J = -u * lnNp / (FOUR_PI * self.G * r)
        rho = Mp / (FOUR_PI * r * r) - u * J
        return dict(N=N, v=v, rho=rho, J=J, M=M, Mp=Mp, u=u)

    def zeros(self, lam, r_lo=0.5, r_hi=2048.0, n_scan=40000):
        """all zeros of 2GM(r) - r (the zeros of N + v = N(1+u)), bisected to 1e-13"""
        r = np.linspace(r_lo, r_hi, n_scan)
        hfun = lambda x: 2.0 * self.G * self.M(x, lam) - x  # noqa: E731
        hv = hfun(r)
        out = []
        for i in np.where(np.sign(hv[:-1]) * np.sign(hv[1:]) < 0)[0]:
            a, b = float(r[i]), float(r[i + 1])
            fa = float(hfun(a))
            for _ in range(200):
                m = 0.5 * (a + b)
                fm = float(hfun(m))
                if (fm > 0) == (fa > 0):
                    a, fa = m, fm
                else:
                    b = m
                if b - a < 1e-13 * max(1.0, m):
                    break
            out.append(0.5 * (a + b))
        return out

    def root(self, lam):
        return max(self.zeros(lam))

    def kappa(self, lam, R=None):
        """the analytic dense slope d_r(N+v) at the outer root = N_R (1 - 2 G M'_R)/(2R)"""
        if R is None:
            R = self.root(lam)
        N_R = float(np.exp(self.lnN(R, lam)))
        return N_R * (1.0 - 2.0 * self.G * float(self.Mp(R, lam))) / (2.0 * R)


def make_record(lam, E, R_zeros, kappa_T, N_t, r_collar, T_scale=1.0):
    return {'T': lam * T_scale, 'T_over_rs': lam, 'E': float(E), 'r_collar': float(r_collar),
            'trapping_zeros': [float(z) for z in R_zeros], 'tangencies': [], 'kappa_T': kappa_T,
            'surface_jets': {'N_t': N_t}, 'converged': True,
            'regime': {'label': 'in the regime', 'intervals_above_1': []}, 'gauge': {'label': 'intact'},
            'late_surface': {'flag': False}, 'interp_floor_N': 1e-12, 'interp_floor_v': 1e-12,
            'indicator': {'collar_screen': {'delta_E_signed': 0.0}, 'collar_label': 'passes'},
            'solver': {'W_tail': {'E_tail_end': float(E), 'E_collar': float(E)}}}


def family_records(fam, lams, R, flip_J=False):
    n = len(lams)
    N = np.empty((n, R.size))
    v = np.empty_like(N)
    rho = np.empty_like(N)
    J = np.empty_like(N)
    records = []
    for i, lam in enumerate(lams):
        f = fam.fields(R, lam)
        N[i], v[i], rho[i], J[i] = f['N'], f['v'], f['rho'], f['J']
        zs = fam.zeros(lam)
        Rr = max(zs)
        rec = make_record(lam, f['M'][-1], zs, fam.kappa(lam, Rr), float(np.exp(fam.lnN(Rr, lam))), R[-1])
        records.append(rec)
    F = {'R': R, 'T': np.asarray(lams, float), 'N': N, 'v': v, 'rho': rho, 'J': (-1.0 if flip_J else 1.0) * J,
         'rho_pairs': None, 'J_pairs': None, 'G': fam.G, 'keys': ['R', 'T', 'N', 'v', 'eps_tot', 'J_tot', 'g'],
         'N_key': 'N', 'v_key': 'v', 'rho_key': 'eps_tot', 'J_key': 'J_tot'}
    return records, F


def static_records(R, G, M0, noise=0.0, seed=1):
    rng = np.random.RandomState(seed)
    N = np.ones((3, R.size))
    v = np.tile(-np.sqrt(2.0 * G * M0 / R), (3, 1))
    if noise > 0:
        N[2] += noise * rng.randn(R.size)
        v[2] += noise * rng.randn(R.size)
    zero = np.zeros((3, R.size))
    records = [make_record(float(i), M0, [2.0 * G * M0], 1.0 / (4.0 * G * M0), 1.0, R[-1]) for i in range(3)]
    F = {'R': R, 'T': np.arange(3, dtype=float), 'N': N, 'v': v, 'rho': zero, 'J': zero.copy(),
         'rho_pairs': None, 'J_pairs': None, 'G': G, 'keys': ['R', 'T', 'N', 'v', 'eps_tot', 'J_tot', 'g'],
         'N_key': 'N', 'v_key': 'v', 'rho_key': 'eps_tot', 'J_key': 'J_tot'}
    return records, F


def save_layout(out_dir, name, records, F):
    """write the manufactured records in the engine's layout (collapse_<tag>.json + _fields.npz)"""
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, 'collapse_%s.json' % name)
    npath = os.path.join(out_dir, 'collapse_%s_fields.npz' % name)
    with open(jpath, 'w') as fh:
        json.dump({'args': {'g': F['G'], 'rs': 1.0, 'stop_E': 1e-12, 'tag': name}, 'passes': fl._jsonable(records)}, fh)
    np.savez(npath, R=F['R'], T=F['T'], N=F['N'], v=F['v'], eps_tot=F['rho'], J_tot=F['J'], g=np.array(F['G']), ell=np.array(1))
    return jpath, npath


def run(records, F, G, **kw):
    opts = fl.make_opts(G=G, rs=1.0, stop_E=1e-12, tag='selftest', **kw)
    return fl.analyse(records, F, {}, opts)


def gl_integral(fun, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return half * sum(wi * fun(mid + half * xi) for xi, wi in zip(x, w))


def exact_pair_terms(fam, lam_a, lam_b, R_fine, h_fine, n_gl, dlam):
    """W_exact = int dlam 4 pi int_{R(lam)}^{r_end} r^2 (N d_lam rho + v d_lam J) dr   (Simpson on the fine grid)
       K_exact = int dlam kappa(lam) dA/dlam / (8 pi G)"""
    base = FOUR_PI * R_fine * R_fine

    def Wdot(lam):
        fp = fam.fields(R_fine, lam + dlam)
        fm = fam.fields(R_fine, lam - dlam)
        f0 = fam.fields(R_fine, lam)
        drho = (fp['rho'] - fm['rho']) / (2.0 * dlam)
        dJ = (fp['J'] - fm['J']) / (2.0 * dlam)
        integrand = base * (f0['N'] * drho + f0['v'] * dJ)
        return fl.integral_from(R_fine, integrand, fam.root(lam), h_fine, 3)[0]['simpson']

    def Kdot(lam):
        Rr = fam.root(lam)
        dR = (fam.root(lam + dlam) - fam.root(lam - dlam)) / (2.0 * dlam)
        return fam.kappa(lam, Rr) * 8.0 * np.pi * Rr * dR / (8.0 * np.pi * fam.G)

    return gl_integral(Wdot, lam_a, lam_b, n_gl), gl_integral(Kdot, lam_a, lam_b, n_gl)


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    G = a.G
    R = np.arange(1, a.n_sites + 1, dtype=float)
    results = []

    def check(name, ok, detail):
        results.append({'name': name, 'pass': bool(ok), 'detail': detail})
        print('%s  %s  %s' % ('PASS' if ok else 'FAIL', name, detail))

    # ---------------- S1 static ----------------
    M0 = 100.0
    recs, F = static_records(R, G, M0, noise=0.0)
    save_layout(a.out, 'static', recs, F)
    res = run(recs, F, G)
    p0 = res['pairs'][0] if res['pairs'] else None
    ok = p0 is not None and abs(p0['dE']) <= a.tol_static and abs(p0['col2']) <= a.tol_static and abs(p0['W_ext']) <= a.tol_static and abs(p0['res']) <= a.tol_static
    check('S1a static identical records: columns 0', ok, 'pairs %d; dE %s col2 %s W %s res %s' % (len(res['pairs']), fl._fmt(p0 and p0['dE']), fl._fmt(p0 and p0['col2']), fl._fmt(p0 and p0['W_ext']), fl._fmt(p0 and p0['res'])))
    r0 = res['records'][0]
    kf = r0['kappa_formula']
    ok = abs(kf - 1.0 / (4.0 * G * M0)) <= a.tol_kappa_rel / (4.0 * G * M0) and abs(r0['R_refit'] - 2.0 * G * M0) <= a.tol_root and abs(r0['kappa_dense'] - 1.0 / (4.0 * G * M0)) <= a.tol_slope_rel / (4.0 * G * M0)
    check('S1b static kappa_formula = 1/(4GM), refit root = 2GM, refit slope', ok, 'kappa_f %s (1/4GM %s), R_refit - 2GM %s, kappa_dense %s, flags %s' % (
        fl._fmt(kf, 10), fl._fmt(1.0 / (4.0 * G * M0), 10), fl._fmt(r0['R_refit'] - 2.0 * G * M0, 3), fl._fmt(r0['kappa_dense'], 10), r0['flags']))
    recs, F = static_records(R, G, M0, noise=1e-10)
    res = run(recs, F, G)
    p1 = res['pairs'][1] if len(res['pairs']) > 1 else None
    ok = p1 is not None and abs(p1['res']) <= a.tol_static_noisy and abs(p1['col2']) <= a.tol_static_noisy and abs(p1['W_ext']) <= a.tol_static_noisy
    check('S1c static with 1e-10 noise on one record: columns below tol', ok, 'dE %s col2 %s W %s res %s res_applied %s' % (
        fl._fmt(p1 and p1['dE']), fl._fmt(p1 and p1['col2']), fl._fmt(p1 and p1['W_ext']), fl._fmt(p1 and p1['res']), fl._fmt(p1 and p1['res_applied'])))

    # ---------------- F1 family ----------------
    fam = Family(G)
    n_c = int(a.n_coarse)
    n_f = (n_c - 1) * int(a.fine_factor) + 1
    lam_c = np.linspace(0.0, 1.0, n_c)
    lam_f = np.linspace(0.0, 1.0, n_f)
    recs_c, F_c = family_records(fam, lam_c, R)
    recs_f, F_f = family_records(fam, lam_f, R)
    save_layout(a.out, 'family_coarse', recs_c, F_c)
    save_layout(a.out, 'family_fine', recs_f, F_f)
    res_c = run(recs_c, F_c, G)
    res_f = run(recs_f, F_f, G)
    ok = len(res_c['pairs']) == n_c - 1 and len(res_f['pairs']) == n_f - 1 and res_c['sJ'] == 1
    check('F1a all family pairs included, J sign +1', ok, 'coarse %d/%d fine %d/%d, sJ %d, excluded %s' % (
        len(res_c['pairs']), n_c - 1, len(res_f['pairs']), n_f - 1, res_c['sJ'], [(e['T_a'], e['reasons']) for e in res_c['excluded_pairs']]))
    kdiff = max(abs(r['kappa_formula'] - r['kappa_T_stored']) / abs(r['kappa_T_stored']) for r in res_f['records'])
    rdiff = max(abs(r['R_refit'] - r['R_stored']) for r in res_f['records'])
    ediff = max(abs(r['M_collar_fields_minus_E']) / abs(r['E_stored']) for r in res_f['records'])
    ok = kdiff <= a.tol_kappa_rel and rdiff <= a.tol_root and ediff <= a.tol_E_rel
    check('F1b kappa_formula = analytic kappa_T, refit root = analytic root, M_collar = E', ok,
          'max rel kappa diff %s, max |R_refit - R| %s, max rel E diff %s' % (fl._fmt(kdiff, 3), fl._fmt(rdiff, 3), fl._fmt(ediff, 3)))
    rows_e = max(r['row_e_rel_ext'] for r in res_f['records'])
    rows_g = max(r['row_g_rel_ext'] for r in res_f['records'])
    pred = max(abs(r['kappa_prediction_check']) / abs(r['kappa_T_stored']) for r in res_f['records'])
    ok = rows_e <= a.tol_rows and rows_g <= a.tol_rows and pred <= a.tol_kappa_rel * 10
    check('F1c exterior row defects small on the exact family, kappa prediction check', ok,
          'rel e %s, rel g %s (tol %s); prediction check rel %s' % (fl._fmt(rows_e, 3), fl._fmt(rows_g, 3), a.tol_rows, fl._fmt(pred, 3)))

    R_fine = np.arange(1, a.n_sites * a.fine_radial_factor + 1, dtype=float) / a.fine_radial_factor
    h_fine = 1.0 / a.fine_radial_factor

    def pair_errors(res, lams):
        errW, errres, errid, ratios = [], [], [], []
        for p in res['pairs']:
            W_ex, K_ex = exact_pair_terms(fam, p['T_a'], p['T_b'], R_fine, h_fine, a.n_gl, a.dlam)
            errW.append(p['W_simpson'] - W_ex)
            errres.append(p['res_simpson'])
            errid.append(p['dE'] - K_ex - W_ex)
            dts = p['W_ext'] - p['W_simpson']
            ratios.append((p['W_coarse'] - p['W_ext']) / dts if dts != 0 else float('nan'))
        return np.array(errW), np.array(errres), np.array(errid), np.array(ratios)

    eW_c, er_c, eid_c, rat_c = pair_errors(res_c, lam_c)
    eW_f, er_f, eid_f, rat_f = pair_errors(res_f, lam_f)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))  # noqa: E731
    order_W = math.log(rms(eW_c) / rms(eW_f), 2) if rms(eW_f) > 0 else float('inf')
    order_r = math.log(rms(er_c) / rms(er_f), 2) if rms(er_f) > 0 else float('inf')
    cum_c, cum_f = abs(float(np.sum(er_c))), abs(float(np.sum(er_f)))
    order_cum = math.log(cum_c / cum_f, 2) if cum_f > 0 else float('inf')
    ok = order_W >= a.order_min and order_r >= a.order_min and order_cum >= a.order_cum_min
    check('F1d convergence order in the record step (per pair ~3, cumulative ~2)', ok,
          'order |W_script - W_exact| %s, order res %s, cumulative order %s; rms per pair coarse/fine: W err %s/%s, res %s/%s; cum res %s/%s' % (
              fl._fmt(order_W, 3), fl._fmt(order_r, 3), fl._fmt(order_cum, 3), fl._fmt(rms(eW_c), 3), fl._fmt(rms(eW_f), 3),
              fl._fmt(rms(er_c), 3), fl._fmt(rms(er_f), 3), fl._fmt(cum_c, 3), fl._fmt(cum_f, 3)))
    dE_rms = res_f['stats']['dE_rms']
    ok = rms(er_f) <= a.tol_closure_rel * dE_rms
    check('F1e closure at the fine step (Simpson W) relative to rms delta E', ok,
          'rms res %s / rms dE %s = %s (tol %s); sum W_script %s, sum W_exact %s' % (
              fl._fmt(rms(er_f), 3), fl._fmt(dE_rms, 3), fl._fmt(rms(er_f) / dE_rms, 3), a.tol_closure_rel,
              fl._fmt(res_f['stats']['sum_W_simpson'], 8), fl._fmt(res_f['stats']['sum_W_simpson'] - float(np.sum(eW_f)), 8)))
    idmax = float(np.max(np.abs(np.concatenate([eid_c, eid_f]))))
    check('F1f the identity with exact lam-quadrature of both terms', idmax <= a.tol_identity, 'max |dE - int kappa dA/8piG - W_exact| %s (tol %s)' % (fl._fmt(idmax, 3), a.tol_identity))
    rat_med = float(np.nanmedian(np.concatenate([rat_c, rat_f])))
    ok = a.rule_ratio_min <= rat_med <= a.rule_ratio_max
    check('F1g radial rules: (W_half - W_trapz)/(W_trapz - W_simpson) ~ 3', ok, 'median %s (window [%s, %s]); disc_W_rule median %s' % (
        fl._fmt(rat_med, 4), a.rule_ratio_min, a.rule_ratio_max, fl._fmt(np.median([p['disc_W_rule'] for p in res_f['pairs']]), 3)))

    # ---------------- J1 flipped sign ----------------
    recs_j, F_j = family_records(fam, lam_c, R, flip_J=True)
    res_j = run(recs_j, F_j, G)
    kdiff_j = max(abs(r['kappa_formula'] - r['kappa_T_stored']) / abs(r['kappa_T_stored']) for r in res_j['records'])
    ok = res_j['sJ'] == -1 and kdiff_j <= a.tol_kappa_rel and abs(res_j['stats']['sum_res_simpson'] - res_c['stats']['sum_res_simpson']) <= 1e-9
    check('J1 flipped J_tot: audit picks -1, kappa and residual unchanged', ok, 'sJ %d, e(+) %s e(-) %s, kappa rel diff %s, sum res_simpson %s vs %s' % (
        res_j['sJ'], fl._fmt(res_j['audit']['rel_e_plus'], 3), fl._fmt(res_j['audit']['rel_e_minus'], 3), fl._fmt(kdiff_j, 3),
        fl._fmt(res_j['stats']['sum_res_simpson'], 6), fl._fmt(res_c['stats']['sum_res_simpson'], 6)))

    # ---------------- R1 root finder ----------------
    R_star = 37.3
    f1 = 0.01 * (R - R_star) * (1.0 + 0.3 * np.tanh((R - 60.0) / 10.0))
    slope1 = 0.01 * (1.0 + 0.3 * math.tanh((R_star - 60.0) / 10.0))
    z1 = fl.classify_zeros(R, f1, 2.0, 4)
    ok = len(z1) == 1 and abs(z1[0]['r'] - R_star) <= a.tol_root_finder and abs(z1[0]['slope'] - slope1) <= a.tol_slope_rel * slope1 and z1[0]['type'] == 'outer'
    check('R1a single zero: root and slope', ok, 'zeros %s; root err %s slope err rel %s' % (
        [(fl._fmt(z['r'], 10), z['type']) for z in z1], fl._fmt(z1[0]['r'] - R_star if z1 else float('nan'), 3), fl._fmt((z1[0]['slope'] - slope1) / slope1 if z1 else float('nan'), 3)))
    f2 = 1e-6 * (R - 20.0) * (R - 30.0) * (R - 50.0) * (R - 70.0)
    z2 = fl.classify_zeros(R, f2, 2.0, 4)
    n_out = sum(1 for z in z2 if z['type'] == 'outer')
    n_in = sum(1 for z in z2 if z['type'] == 'inner')
    ok = len(z2) == 4 and n_out == 2 and n_in == 2 and max(abs(z['r'] - t) for z, t in zip(z2, (20.0, 30.0, 50.0, 70.0))) <= a.tol_root_finder
    check('R1b four zeros: two outer-type (not simple)', ok, 'zeros %s' % [(fl._fmt(z['r'], 8), z['type']) for z in z2])
    f3 = 5e-5 + 1e-5 * (R - 40.0) ** 2
    z3 = fl.classify_zeros(R, f3, 2.0, 4)
    t3 = fl.near_tangencies(R, f3, 1e-4, 2.0)
    ok = len(z3) == 0 and len(t3) == 1 and abs(t3[0]['r'] - 40.0) <= 1.0
    check('R1c near tangency: no zero, flagged', ok, 'zeros %d, near tangencies %s' % (len(z3), t3))

    # ---------------- X1 exclusion logic ----------------
    recs_x, F_x = family_records(fam, lam_c, R)
    recs_x[0]['trapping_zeros'] = []
    recs_x[0]['kappa_T'] = None
    recs_x[3]['regime'] = {'label': 'above the regime', 'intervals_above_1': [[10.0, 12.0]]}
    recs_x[6]['tangencies'] = [30.0]
    res_x = run(recs_x, F_x, G)
    exc = {(round(e['T_a'], 6), round(e['T_b'], 6)): e['reasons'] for e in res_x['excluded_pairs']}
    want = {(round(lam_c[0], 6), round(lam_c[1], 6)): 'root_appearing',
            (round(lam_c[2], 6), round(lam_c[3], 6)): 'B:regime', (round(lam_c[3], 6), round(lam_c[4], 6)): 'A:regime',
            (round(lam_c[5], 6), round(lam_c[6], 6)): 'B:tangency_stored', (round(lam_c[6], 6), round(lam_c[7], 6)): 'A:tangency_stored'}
    ok = len(exc) == len(want) and all(k in exc and w in exc[k] for k, w in want.items()) and len(res_x['pairs']) == n_c - 1 - len(want)
    check('X1 exclusion logic: root appearing, regime flag, stored tangency', ok, 'excluded %s; included %d' % (
        sorted((k, v) for k, v in exc.items()), len(res_x['pairs'])))

    # ---------------- summary ----------------
    n_fail = sum(1 for r in results if not r['pass'])
    summary = {'results': results, 'n_fail': n_fail, 'seconds': time.time() - t0, 'options': vars(a),
               'family_stats_fine': fl._jsonable(res_f['stats'])}
    with open(os.path.join(a.out, 'firstlaw_selftest_v1.json'), 'w') as fh:
        json.dump(fl._jsonable(summary), fh, indent=1)
    # the main script's writers on the fine family (exercises the json/npz/md path)
    opts_w = fl.make_opts(G=G, rs=1.0, stop_E=1e-12, tag='selftest_family_fine')
    arr = {}
    for r in res_f['records']:
        row = r['row']
        e, g, M, _, _ = fl.row_defects(R, F_f['N'][row], F_f['v'][row], F_f['rho'][row], F_f['J'][row], G, 1.0)
        arr[r['k']] = {'e': e, 'g': g, 'M': M, 'Nplusv': F_f['N'][row] + F_f['v'][row]}
    base = fl.write_outputs(res_f, R, arr, a.out, 'selftest_family_fine', 'manufactured', 'manufactured', opts_w)
    ok = all(os.path.exists(base + ext) for ext in ('.json', '.npz', '.md'))
    check('W1 writers produce json/npz/md', ok, base)
    n_fail = sum(1 for r in results if not r['pass'])
    print('firstlaw_selftest_v1: %d checks, %d failed, %.1f s, outputs in %s' % (len(results), n_fail, time.time() - t0, a.out))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
