#!/usr/bin/env python3
"""
firstlaw_selftest_v2.py -- manufactured checks for firstlaw_records_v2.py

Two groups (each line prints PASS/FAIL with the numbers; exit code 1 on any FAIL):

  A. the pair machinery on an exact-rows family without the engine (version 1's analytic family M(r, lam), ln N(r, lam)
     with rho, J defined by the rows, sampled on the dense grid and fed to analyse() as if re-solved):
     F1a all pairs included; F1b the dense row audit of the analytic family (Richardson differences) is small;
     F1c per-pair |W - W_exact| and the residual fall by 2^order when the record step halves (order >= --order-min),
         the cumulative residual by 2^(order-1); F1d closure at the fine step; F1e the identity with exact
         lam-quadrature; F1f the radial rules; X1 the exclusion logic (root appearing, regime flag, stored tangency,
         a stored zero that the dense solution does not reproduce).
  B. the engine (imported read-only from --engine-path, the module --engine-module):
     E0 the import; E1 a compact manufactured profile (a Gaussian core; the outer root in vacuum): the engine's zeros
        include the outer zero at 2 g E_collar exactly, E_collar = sum eps_tot, kappa_eq = 1/(2R), the own scan equals
        the engine's zeros, kappa_fd = kappa_eq;
     E2 the rows close on the dense solution (compact and broad profiles, exterior RMS defects <= --tol-rows-dense;
        the engine's own residuals printed);
     E3 the broad profile (core + halo, the root inside the matter): two zeros (inner, outer); kappa_eq, kappa_fd and
        the row formula agree; the re-solve reproduces the manufactured record's zeros (a second solve of the same
        sources, through the script's reproduction check, max |difference| <= --tol-repro); sources perturbed by
        --perturb-rel are NOT reproduced (the check detects them);
     E4 a lam family through the engine: all pairs included, the per-pair residual's convergence order in the record
        step (>= --order-min), closure at the fine step, res_applied = res_collarE up to the defect integral;
     E5 the site-array comparisons on the engine family: the dense solution sampled at the sites reproduces the dense
        root to --tol-site-refit; with the engine's kernel average available, its root displacement against the
        prediction is printed (informational);
     W1 the writers.

CPU only; the engine solves run in-process (one at a time).  Writes the manufactured records (json + npz in the
engine's layout) and a summary json to --out.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firstlaw_records_v2 as fl  # noqa: E402  (sets the thread caps from --threads/--workers)

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
NAN = float('nan')


def Phi(x):
    return 0.5 * _erfc(-np.asarray(x, float) / SQRT2)


def phi(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='manufactured checks for firstlaw_records_v2')
    p.add_argument('--out', default='<internal>/firstlaw_selftest_v2', help='output directory')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--workers', type=int, default=1, help='accepted for symmetry; the self-test solves in-process')
    p.add_argument('--engine-path', dest='engine_path', default=None, help="the engine's directory (default: this script's directory)")
    p.add_argument('--engine-module', dest='engine_module', default='engine_fields_v16')
    p.add_argument('--skip-engine', dest='skip_engine', action='store_true', help='group B is skipped (the SKIP lines count as failures unless --allow-skip)')
    p.add_argument('--allow-skip', dest='allow_skip', action='store_true')
    p.add_argument('--G', type=float, default=0.25)
    p.add_argument('--ell', type=float, default=2.0)
    p.add_argument('--fine', type=int, default=16)
    p.add_argument('--n-sites-family', dest='n_sites_family', type=int, default=2048, help='sites of the analytic family grid (spacing 1)')
    p.add_argument('--n-sites-engine', dest='n_sites_engine', type=int, default=320, help='sites of the engine profiles (spacing 1)')
    p.add_argument('--n-coarse', dest='n_coarse', type=int, default=5, help='records of the coarse families (lam = 0..1)')
    p.add_argument('--fine-factor', dest='fine_factor', type=int, default=2)
    p.add_argument('--n-gl', dest='n_gl', type=int, default=6)
    p.add_argument('--dlam', type=float, default=1e-4)
    p.add_argument('--fine-radial-factor', dest='fine_radial_factor', type=int, default=4)
    p.add_argument('--rtol', type=float, default=1e-12)
    p.add_argument('--atol', type=float, default=1e-16)
    p.add_argument('--support-rel', dest='support_rel', type=float, default=1e-14)
    p.add_argument('--perturb-rel', dest='perturb_rel', type=float, default=1e-3, help='relative perturbation of eps_tot for the non-reproduction check')
    p.add_argument('--tol-kappa-rel', dest='tol_kappa_rel', type=float, default=1e-6)
    p.add_argument('--tol-root', dest='tol_root', type=float, default=1e-6, help='sites')
    p.add_argument('--tol-E-rel', dest='tol_E_rel', type=float, default=1e-6)
    p.add_argument('--tol-vacuum-root-rel', dest='tol_vacuum_root_rel', type=float, default=1e-8, help='|R - 2 g E_collar|/R on the compact profile')
    p.add_argument('--tol-rows-dense', dest='tol_rows_dense', type=float, default=1e-6, help='exterior RMS row defects on the dense solution')
    p.add_argument('--tol-rows-family', dest='tol_rows_family', type=float, default=1e-6)
    p.add_argument('--tol-scan', dest='tol_scan', type=float, default=1e-8, help='sites: own scan against the engine zeros')
    p.add_argument('--tol-repro', dest='tol_repro', type=float, default=1e-9, help='sites: the second solve against the first')
    p.add_argument('--tol-site-refit', dest='tol_site_refit', type=float, default=5e-3, help='sites: the refit root of the dense solution sampled at the sites against the dense root')
    p.add_argument('--tol-identity', dest='tol_identity', type=float, default=1e-6)
    p.add_argument('--tol-closure-rel', dest='tol_closure_rel', type=float, default=1e-3)
    p.add_argument('--order-min', dest='order_min', type=float, default=2.5)
    p.add_argument('--order-cum-min', dest='order_cum_min', type=float, default=1.5)
    p.add_argument('--rule-ratio-min', dest='rule_ratio_min', type=float, default=1.5)
    p.add_argument('--rule-ratio-max', dest='rule_ratio_max', type=float, default=6.0)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# A. the exact-rows family (version 1's), sampled on the dense grid
# ----------------------------------------------------------------------------
class Family(object):
    """M(r,lam) = (M_inf - mu)(1 - exp(-(r/Lc)^3)) + mu (1 - Phi((r_s - r)/w)) + m_b (1 - exp(-(r/L)^2)),
       ln N(r,lam) = -nu Phi((r_s - r)/w) - nu_h exp(-(r/L)^2); u = -sqrt(2GM/r), v = uN;
       J = -u (ln N)'/(4 pi G r), rho = M'/(4 pi r^2) - u J  (the rows, exactly)."""

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
        return ((p['M_inf'] - p['mu']) * (1.0 - np.exp(-(r / self.Lc) ** 3)) + p['mu'] * (1.0 - Phi((p['rs'] - r) / self.w))
                + p['mb'] * (1.0 - np.exp(-(r / self.L) ** 2)))

    def Mp(self, r, lam):
        p = self.par(lam)
        r = np.asarray(r, float)
        return ((p['M_inf'] - p['mu']) * 3.0 * r * r / self.Lc ** 3 * np.exp(-(r / self.Lc) ** 3) + p['mu'] / self.w * phi((p['rs'] - r) / self.w)
                + p['mb'] * 2.0 * r / self.L ** 2 * np.exp(-(r / self.L) ** 2))

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
        rr = np.where(r > 0, r, 1e-300)
        M = self.M(rr, lam)
        Mp = self.Mp(rr, lam)
        N = np.exp(self.lnN(rr, lam))
        lnNp = self.lnNp(rr, lam)
        u = -np.sqrt(np.maximum(2.0 * self.G * M / rr, 0.0))
        v = u * N
        J = -u * lnNp / (FOUR_PI * self.G * rr)
        rho = Mp / (FOUR_PI * rr * rr) - u * J
        v = np.where(r > 0, v, 0.0)
        J = np.where(r > 0, J, 0.0)
        rho = np.where(r > 0, rho, rho)
        return dict(N=N, v=v, rho=rho, J=J, M=M, Mp=Mp, u=u, W=rr * v * v)

    def zeros(self, lam, r_lo=0.5, r_hi=2048.0, n_scan=40000):
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

    def slope(self, z, lam):
        """d_r(N + v) at a zero of N + v: N_z (1 - 2 G M'(z))/(2z)"""
        return float(np.exp(self.lnN(z, lam))) * (1.0 - 2.0 * self.G * float(self.Mp(z, lam))) / (2.0 * z)

    def kappa(self, lam, R=None):
        if R is None:
            R = self.root(lam)
        return self.slope(R, lam)


def make_record(lam, E, zeros, kappa_T, N_t, r_collar, extra=None):
    rec = {'T': lam, 'T_over_rs': lam, 'E': float(E), 'r_collar': float(r_collar),
           'trapping_zeros': [float(z) for z in zeros], 'tangencies': [], 'kappa_T': kappa_T,
           'surface_jets': {'N_t': N_t}, 'converged': True,
           'regime': {'label': 'in the regime', 'intervals_above_1': []}, 'gauge': {'label': 'intact'},
           'late_surface': {'flag': False}, 'interp_floor_N': 1e-12, 'interp_floor_v': 1e-12,
           'indicator': {'collar_screen': {'delta_E_signed': 0.0}, 'collar_label': 'passes'},
           'solver': {'W_tail': {'E_tail_end': float(E), 'E_collar': float(E)}}}
    if extra:
        rec.update(extra)
    return rec


def family_dense(fam, lam, r_fine, G, ell, r_collar, h_fd):
    """The analytic family as a dense result dict (what solve_record_dense returns), the row audit by the same
    Richardson rule on the analytic functions."""
    f = fam.fields(r_fine, lam)
    zeros = fam.zeros(lam)
    details = [{'r': z, 'slope_eq': fam.slope(z, lam), 'slope_fd': fam.slope(z, lam), 'type': 'outer' if fam.slope(z, lam) > 0 else 'inner'} for z in zeros]
    Nf = lambda y: fam.fields(y, lam)['N']  # noqa: E731
    vf = lambda y: fam.fields(y, lam)['v']  # noqa: E731
    Wf = lambda y: fam.fields(y, lam)['W']  # noqa: E731
    rhof = lambda y: fam.fields(y, lam)['rho']  # noqa: E731
    Jf = lambda y: fam.fields(y, lam)['J']  # noqa: E731
    root = fl.root_quantities(max(zeros), Nf, vf, Wf, rhof, Jf, G, ell, h_fd) if zeros else None
    lo = (max(zeros) if zeros else 2.0) + 2.0 * h_fd
    mask = (r_fine > lo) & (r_fine < r_collar - 2.0 * h_fd)
    audit = fl.dense_row_audit(r_fine[mask], Nf, vf, Wf, rhof, Jf, G, h_fd)
    E = float(fam.M(r_collar, lam))
    eng = {'module': 'analytic family', 'converged': True, 'regime': {'label': 'in the regime', 'intervals_above_1': []},
           'gauge': {'label': 'intact', 'intervals': []}, 'ivp': {'nfev': 0, 'segments': 1, 'N_centre': float(np.exp(fam.lnN(1e-6, lam)))},
           'collar_margin': {'deficit_ell': 0}, 'W_tail': {'E_tail_end': E, 'E_collar': E}}
    return {'ok': True, 'failed': None, 'seconds': 0.0, 'E_collar': E, 'r_collar': float(r_collar), 'engine': eng,
            'zeros_engine': list(zeros), 'zeros_scan': list(zeros), 'zeros_dense': list(zeros), 'zeros_source': 'analytic', 'tangencies_engine': [],
            'zeros_detail': details, 'root': root, 'audit': audit, 'audit_2h': audit, 'sites': {},
            'arrays': {'N': f['N'], 'v': f['v'], 'rho': f['rho'], 'J': f['J']}}


def family_run(fam, lams, R, r_fine, G, ell, h_fd):
    r_collar = float(r_fine[-1])
    records, dense = [], {}
    n = len(lams)
    eps = np.zeros((n, R.size))
    J = np.zeros((n, R.size))
    Ns = np.zeros((n, R.size))
    vs = np.zeros((n, R.size))
    for i, lam in enumerate(lams):
        d = family_dense(fam, lam, r_fine, G, ell, r_collar, h_fd)
        d['row'] = i
        dense[i] = d
        fs = fam.fields(R, lam)
        eps[i], J[i], Ns[i], vs[i] = fs['rho'] * FOUR_PI * R * R, fs['J'] * FOUR_PI * R * R, fs['N'], fs['v']
        Rr = max(d['zeros_dense'])
        records.append(make_record(lam, d['E_collar'], d['zeros_dense'], fam.kappa(lam, Rr), float(np.exp(fam.lnN(Rr, lam))), r_collar))
    F = {'R': R, 'T': np.asarray(lams, float), 'eps_tot': eps, 'J_tot': J, 'eps_pairs': None, 'J_pairs': None, 'N_site': Ns, 'v_site': vs,
         'N_applied': None, 'v_applied': None, 'G': G, 'ell': ell, 'keys': ['R', 'T', 'N', 'v', 'eps_tot', 'J_tot', 'g', 'ell'], 'solver_string': None, 'label': None}
    return records, F, dense


def run_analysis(records, F, dense, r_fine, G, ell, **kw):
    opts = fl.make_opts(G=G, ell=ell, rs=1.0, stop_E=1e-12, tag='selftest', **kw)
    return fl.analyse(records, F, {}, opts, dense, r_fine)


def gl_integral(fun, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return half * sum(wi * fun(mid + half * xi) for xi, wi in zip(x, w))


def exact_pair_terms(fam, lam_a, lam_b, R_fine, h_fine, n_gl, dlam, r_collar):
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
# B. the engine profiles
# ----------------------------------------------------------------------------
def shell(R, r0, w):
    return R * R * np.exp(-(R - r0) ** 2 / (2.0 * w * w))


def compact_sources(R, E=100.0, r0=20.0, w=4.0, jfrac=0.0):
    s = shell(R, r0, w)
    eps = E * s / s.sum()
    return eps, jfrac * eps


def broad_sources(R, lam=0.0):
    """core (fraction fc at r0c) + halo (the rest at r0h): the outer root inside the halo's matter"""
    E = 100.0 + 6.0 * lam
    fc = 0.62 + 0.06 * lam
    r0h = 60.0 - 8.0 * lam
    jf = 0.10 + 0.05 * lam
    sc = shell(R, 20.0, 4.0)
    sh = shell(R, r0h, 25.0)
    eps = E * (fc * sc / sc.sum() + (1.0 - fc) * sh / sh.sum())
    return eps, jf * eps


def record_from_dense(d, lam):
    eng = d.get('engine') or {}
    root = d.get('root') or {}
    E = d['E_collar']
    extra = {'regime': {'label': (eng.get('regime') or {}).get('label') or 'in the regime', 'intervals_above_1': (eng.get('regime') or {}).get('intervals_above_1') or []},
             'gauge': {'label': (eng.get('gauge') or {}).get('label') or 'intact'},
             'interp_floor_N': eng.get('interp_floor_N'), 'interp_floor_v': eng.get('interp_floor_v'),
             'solver': {'W_tail': eng.get('W_tail') or {'E_tail_end': E, 'E_collar': E}, 'module': eng.get('module'), 'rtol': eng.get('rtol'), 'atol': eng.get('atol')}}
    return make_record(lam, E, d['zeros_dense'], root.get('kappa_eq'), root.get('N_R'), d['r_collar'], extra=extra)


def engine_family(FLD, R, lams, source_fn, g, ell, sargs, r_fine, cfg, with_kernel=False):
    records, dense = [], {}
    n = len(lams)
    eps_all = np.zeros((n, R.size))
    J_all = np.zeros((n, R.size))
    Ns = np.zeros((n, R.size))
    vs = np.zeros((n, R.size))
    Nk = np.zeros((n, R.size)) if with_kernel else None
    vk = np.zeros((n, R.size)) if with_kernel else None
    for i, lam in enumerate(lams):
        eps, J = source_fn(R, lam)
        d = fl.solve_record_dense(FLD, R, eps, J, g, ell, sargs, r_fine, cfg)
        d['row'] = i
        dense[i] = d
        eps_all[i], J_all[i] = eps, J
        if d.get('ok'):
            Ns[i], vs[i] = d['arrays']['N_sites'], d['arrays']['v_sites']
            if with_kernel and 'N_kernel' in d['arrays']:
                Nk[i], vk[i] = d['arrays']['N_kernel'], d['arrays']['v_kernel']
            records.append(record_from_dense(d, lam))
        else:
            records.append(make_record(lam, NAN, [], None, None, NAN))
    F = {'R': R, 'T': np.asarray(lams, float), 'eps_tot': eps_all, 'J_tot': J_all, 'eps_pairs': None, 'J_pairs': None, 'N_site': Ns, 'v_site': vs,
         'N_applied': Nk, 'v_applied': vk, 'G': g, 'ell': ell, 'keys': ['R', 'T', 'N', 'v', 'eps_tot', 'J_tot', 'g', 'ell'], 'solver_string': None, 'label': None}
    return records, F, dense


def save_layout(out_dir, name, records, F):
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, 'collapse_%s.json' % name)
    npath = os.path.join(out_dir, 'collapse_%s_fields.npz' % name)
    with open(jpath, 'w') as fh:
        json.dump({'args': {'g': F['G'], 'rs': 1.0, 'stop_E': 1e-12, 'tag': name, 'ell': F['ell']}, 'passes': fl._jsonable(records)}, fh)
    arrays = dict(R=F['R'], T=F['T'], eps_tot=F['eps_tot'], J_tot=F['J_tot'], g=np.array(F['G']), ell=np.array(int(round(F['ell']))))
    if F.get('N_site') is not None:
        arrays['N'] = F['N_site']
        arrays['v'] = F['v_site']
    if F.get('N_applied') is not None:
        arrays['N_applied'] = F['N_applied']
        arrays['v_applied'] = F['v_applied']
    np.savez(npath, **arrays)
    return jpath, npath


# ----------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    G, ell = a.G, a.ell
    results = []
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))  # noqa: E731

    def check(name, ok, detail, skipped=False):
        status = 'SKIP' if skipped else ('PASS' if ok else 'FAIL')
        results.append({'name': name, 'pass': bool(ok) and not skipped, 'skipped': bool(skipped), 'detail': detail})
        print('%s  %s  %s' % (status, name, detail))

    h_fd = ell / 64.0

    # =========================== A. the exact-rows family ===========================
    fam = Family(G)
    R_sites = np.arange(1, a.n_sites_family + 1, dtype=float)
    h_fine = ell / float(a.fine)
    n_fine = int(round(a.n_sites_family / h_fine))
    r_fine = np.arange(n_fine + 1, dtype=float) * h_fine
    n_c = int(a.n_coarse)
    n_f = (n_c - 1) * int(a.fine_factor) + 1
    lam_c = np.linspace(0.0, 1.0, n_c)
    lam_f = np.linspace(0.0, 1.0, n_f)
    recs_c, F_c, dense_c = family_run(fam, lam_c, R_sites, r_fine, G, ell, h_fd)
    recs_f, F_f, dense_f = family_run(fam, lam_f, R_sites, r_fine, G, ell, h_fd)
    save_layout(a.out, 'family_coarse', recs_c, F_c)
    save_layout(a.out, 'family_fine', recs_f, F_f)
    res_c = run_analysis(recs_c, F_c, dense_c, r_fine, G, ell)
    res_f = run_analysis(recs_f, F_f, dense_f, r_fine, G, ell)
    ok = len(res_c['pairs']) == n_c - 1 and len(res_f['pairs']) == n_f - 1
    check('F1a all family pairs included (coarse and fine)', ok, 'coarse %d/%d fine %d/%d; excluded %s' % (
        len(res_c['pairs']), n_c - 1, len(res_f['pairs']), n_f - 1, [(e['T_a'], e['reasons']) for e in res_c['excluded_pairs'] + res_f['excluded_pairs']]))
    rows_e = max(r['row_e_rel_ext'] for r in res_f['records'])
    rows_g = max(r['row_g_rel_ext'] for r in res_f['records'])
    kd = max(abs(r['kappa_fd_minus_kappa_eq']) / abs(r['kappa_eq']) for r in res_f['records'])
    kf = max(abs(r['kappa_formula_minus_kappa_eq']) / abs(r['kappa_eq']) for r in res_f['records'])
    ok = rows_e <= a.tol_rows_family and rows_g <= a.tol_rows_family and kd <= a.tol_kappa_rel and kf <= a.tol_kappa_rel
    check('F1b dense row audit of the exact family (Richardson h = ell/64) and the three kappas', ok,
          'rel e %s, rel g %s (tol %s); |kappa_fd - kappa_eq|/kappa %s, |kappa_formula - kappa_eq|/kappa %s (tol %s)' % (
              fl._fmt(rows_e, 3), fl._fmt(rows_g, 3), a.tol_rows_family, fl._fmt(kd, 3), fl._fmt(kf, 3), a.tol_kappa_rel))
    R_q = np.arange(1, a.n_sites_family * a.fine_radial_factor + 1, dtype=float) / a.fine_radial_factor
    h_q = 1.0 / a.fine_radial_factor

    def pair_errors(res):
        errW, errres, errid, ratios = [], [], [], []
        for p in res['pairs']:
            W_ex, K_ex = exact_pair_terms(fam, p['T_a'], p['T_b'], R_q, h_q, a.n_gl, a.dlam, float(r_fine[-1]))
            errW.append(p['W_simpson'] - W_ex)
            errres.append(p['res_simpson'])
            errid.append(p['dE'] - K_ex - W_ex)
            dts = p['W_ext'] - p['W_simpson']
            ratios.append((p['W_coarse'] - p['W_ext']) / dts if dts != 0 else NAN)
        return np.array(errW), np.array(errres), np.array(errid), np.array(ratios)

    eW_c, er_c, eid_c, rat_c = pair_errors(res_c)
    eW_f, er_f, eid_f, rat_f = pair_errors(res_f)
    order_W = math.log(rms(eW_c) / rms(eW_f), 2) if rms(eW_f) > 0 else float('inf')
    order_r = math.log(rms(er_c) / rms(er_f), 2) if rms(er_f) > 0 else float('inf')
    cum_c, cum_f = abs(float(np.sum(er_c))), abs(float(np.sum(er_f)))
    order_cum = math.log(cum_c / cum_f, 2) if cum_f > 0 else float('inf')
    ok = order_W >= a.order_min and order_r >= a.order_min and order_cum >= a.order_cum_min
    check('F1c convergence order in the record step (per pair ~3, cumulative ~2)', ok,
          'order |W - W_exact| %s, order res %s, cumulative order %s; rms per pair coarse/fine: W err %s/%s, res %s/%s; cum res %s/%s' % (
              fl._fmt(order_W, 3), fl._fmt(order_r, 3), fl._fmt(order_cum, 3), fl._fmt(rms(eW_c), 3), fl._fmt(rms(eW_f), 3),
              fl._fmt(rms(er_c), 3), fl._fmt(rms(er_f), 3), fl._fmt(cum_c, 3), fl._fmt(cum_f, 3)))
    dE_rms = res_f['stats']['dE_rms']
    ok = rms(er_f) <= a.tol_closure_rel * dE_rms
    check('F1d closure at the fine step (Simpson W) relative to rms delta E', ok, 'rms res %s / rms dE %s = %s (tol %s)' % (
        fl._fmt(rms(er_f), 3), fl._fmt(dE_rms, 3), fl._fmt(rms(er_f) / dE_rms, 3), a.tol_closure_rel))
    idmax = float(np.max(np.abs(np.concatenate([eid_c, eid_f]))))
    check('F1e the identity with exact lam-quadrature of both terms', idmax <= a.tol_identity, 'max |dE - int kappa dA/8piG - W_exact| %s (tol %s)' % (fl._fmt(idmax, 3), a.tol_identity))
    rat_med = float(np.nanmedian(np.concatenate([rat_c, rat_f])))
    ok = a.rule_ratio_min <= rat_med <= a.rule_ratio_max
    check('F1f radial rules on the dense grid: (W_half - W_trapz)/(W_trapz - W_simpson) ~ 3', ok, 'median %s (window [%s, %s])' % (fl._fmt(rat_med, 4), a.rule_ratio_min, a.rule_ratio_max))

    # X1 exclusion logic
    recs_x = [dict(r) for r in recs_c]
    recs_x[0] = dict(recs_x[0], trapping_zeros=[], kappa_T=None)
    recs_x[1] = dict(recs_x[1], regime={'label': 'above the regime', 'intervals_above_1': [[10.0, 12.0]]})
    recs_x[3] = dict(recs_x[3], tangencies=[30.0])
    recs_x[4] = dict(recs_x[4], trapping_zeros=[z + (0.01 if i == len(recs_x[4]['trapping_zeros']) - 1 else 0.0) for i, z in enumerate(recs_x[4]['trapping_zeros'])])
    res_x = run_analysis(recs_x, F_c, dense_c, r_fine, G, ell)
    exc = {(round(e['T_a'], 6), round(e['T_b'], 6)): e['reasons'] for e in res_x['excluded_pairs']}
    want = {(round(lam_c[0], 6), round(lam_c[1], 6)): 'root_appearing',
            (round(lam_c[1], 6), round(lam_c[2], 6)): 'A:regime',
            (round(lam_c[2], 6), round(lam_c[3], 6)): 'B:tangency_stored', (round(lam_c[3], 6), round(lam_c[4], 6)): 'A:tangency_stored'}
    k34 = (round(lam_c[3], 6), round(lam_c[4], 6))
    ok = (len(exc) == len(want) and all(k in exc and w in exc[k] for k, w in want.items()) and 'B:zeros_not_reproduced' in exc.get(k34, [])
          and len(res_x['pairs']) == n_c - 1 - len(want))
    check('X1 exclusion logic: root appearing, regime flag, stored tangency, stored zero not reproduced', ok, 'excluded %s; included %d' % (
        sorted((k, v) for k, v in exc.items()), len(res_x['pairs'])))

    # =========================== B. the engine ===========================
    FLD = None
    if a.skip_engine:
        check('E0 engine import', False, 'skipped by --skip-engine', skipped=True)
    else:
        try:
            FLD, eprov = fl.import_engine(a.engine_path or os.path.dirname(os.path.abspath(__file__)), a.engine_module)
            check('E0 engine import', True, '%s from %s; files %s' % (a.engine_module, eprov['module_file'], sorted(eprov['files'].keys())))
        except SystemExit as exc:
            check('E0 engine import', False, str(exc))
    if FLD is not None:
        R_e = np.arange(1, a.n_sites_engine + 1, dtype=float)
        ell_pass = fl.ell_for_engine(ell)
        sargs, aspec, _sf = fl.resolve_solver_args(FLD, {}, {}, {'kv': {}}, fl.make_opts(rtol=a.rtol, atol=a.atol, collar_policy='flag', support_rel=a.support_rel))
        r_fine_e, r_last_e, gnote = fl.build_fine_grid(FLD, R_e, ell_pass, a.fine)
        cfg = fl.worker_cfg(fl.make_opts(check_applied=True), r_last_e, ell / float(a.fine))
        print('engine solve args %s; dense grid to %s step %s%s' % (json.dumps(fl._jsonable(sargs)), fl._fmt(r_last_e), fl._fmt(ell / a.fine), ('; ' + gnote) if gnote else ''))

        # E1 compact profile
        eps1, J1 = compact_sources(R_e)
        d1 = fl.solve_record_dense(FLD, R_e, eps1, J1, G, ell_pass, sargs, r_fine_e, cfg)
        if not d1.get('ok'):
            check('E1 compact profile: the engine solve', False, str(d1.get('failed')) + ' ' + str(d1.get('traceback', ''))[-600:])
        else:
            E1 = d1['E_collar']
            R_vac = 2.0 * G * E1
            zd = d1['zeros_dense']
            root = d1['root'] or {}
            scan = d1.get('zeros_scan') or []
            dz_scan = max(abs(x - y) for x, y in zip(sorted(scan), sorted(zd))) if (scan and len(scan) == len(zd)) else NAN
            ok = (bool(zd) and abs(max(zd) - R_vac) <= a.tol_vacuum_root_rel * R_vac and abs(E1 - eps1.sum()) <= a.tol_E_rel * eps1.sum()
                  and abs(root.get('kappa_eq', NAN) - 1.0 / (2.0 * R_vac)) <= a.tol_kappa_rel / (2.0 * R_vac)
                  and abs(root.get('kappa_fd', NAN) - root.get('kappa_eq', NAN)) <= a.tol_kappa_rel * abs(root.get('kappa_eq', 1.0))
                  and np.isfinite(dz_scan) and dz_scan <= a.tol_scan)
            check('E1 compact profile: outer zero = 2 g E_collar, E_collar = sum eps, kappa_eq = 1/(2R), kappa_fd = kappa_eq, own scan = engine zeros', ok,
                  'zeros %s (%s); R - 2gE = %s (rel %s); E_collar - sum eps = %s; kappa_eq %s vs 1/(2R) %s; kappa_fd - kappa_eq %s; scan - engine max %s; r_collar %s; %.1f s' % (
                      json.dumps([round(z, 8) for z in zd]), d1.get('zeros_source'), fl._fmt(max(zd) - R_vac if zd else NAN, 3), fl._fmt((max(zd) - R_vac) / R_vac if zd else NAN, 3),
                      fl._fmt(E1 - eps1.sum(), 3), fl._fmt(root.get('kappa_eq'), 10), fl._fmt(1.0 / (2.0 * R_vac), 10), fl._fmt(root.get('kappa_fd', NAN) - root.get('kappa_eq', NAN), 3),
                      fl._fmt(dz_scan, 3), fl._fmt(d1['r_collar']), d1['seconds']))
        # E3 broad profile (also E2 on both)
        eps3, J3 = broad_sources(R_e, 0.0)
        d3 = fl.solve_record_dense(FLD, R_e, eps3, J3, G, ell_pass, sargs, r_fine_e, cfg)
        if not d3.get('ok'):
            check('E3 broad profile: the engine solve', False, str(d3.get('failed')) + ' ' + str(d3.get('traceback', ''))[-600:])
        else:
            au1 = (d1.get('audit') or {}) if d1.get('ok') else {}
            au3 = d3.get('audit') or {}
            e1 = fl._float(au1.get('rel_e'))
            g1 = fl._float(au1.get('rel_g'))
            g1_abs = fl._float(au1.get('max_abs_g'))
            e3 = fl._float(au3.get('rel_e'))
            g3 = fl._float(au3.get('rel_g'))
            g1_ok = (g1 <= a.tol_rows_dense) if np.isfinite(g1) else (g1_abs <= 1e-9)
            ok = e1 <= a.tol_rows_dense and g1_ok and e3 <= a.tol_rows_dense and g3 <= a.tol_rows_dense
            check('E2 the rows close on the dense solution (exterior RMS, Richardson h = ell/64)', ok,
                  'compact: e %s, g %s (max |g| %s; J flipped e %s); broad: e %s, g %s (J flipped e %s, g %s); engine residual_W %s / %s, residual_invN %s / %s (tol %s)' % (
                      fl._fmt(e1, 3), fl._fmt(g1, 3), fl._fmt(g1_abs, 3), fl._fmt(au1.get('rel_e_flipJ'), 3), fl._fmt(e3, 3), fl._fmt(g3, 3), fl._fmt(au3.get('rel_e_flipJ'), 3),
                      fl._fmt(au3.get('rel_g_flipJ'), 3), fl._fmt((d1.get('engine') or {}).get('residual_W'), 3), fl._fmt((d3.get('engine') or {}).get('residual_W'), 3),
                      fl._fmt((d1.get('engine') or {}).get('residual_invN'), 3), fl._fmt((d3.get('engine') or {}).get('residual_invN'), 3), a.tol_rows_dense))
            zd = d3['zeros_dense']
            det = d3['zeros_detail']
            root = d3['root'] or {}
            types = [z['type'] for z in det]
            kd_ = abs(root.get('kappa_fd', NAN) - root.get('kappa_eq', NAN)) / abs(root.get('kappa_eq', 1.0))
            kf_ = abs(root.get('kappa_formula', NAN) - root.get('kappa_eq', NAN)) / abs(root.get('kappa_eq', 1.0))
            d3b = fl.solve_record_dense(FLD, R_e, eps3, J3, G, ell_pass, sargs, r_fine_e, cfg)
            rec3 = record_from_dense(d3, 0.0)
            F3 = {'R': R_e, 'T': np.array([0.0, 1.0]), 'eps_tot': np.array([eps3, eps3]), 'J_tot': np.array([J3, J3]), 'eps_pairs': None, 'J_pairs': None,
                  'N_site': np.array([d3['arrays']['N_sites'], d3['arrays']['N_sites']]), 'v_site': np.array([d3['arrays']['v_sites'], d3['arrays']['v_sites']]),
                  'N_applied': None, 'v_applied': None, 'G': G, 'ell': ell, 'keys': [], 'solver_string': None, 'label': None}
            res3 = run_analysis([rec3, dict(rec3, T=1.0, T_over_rs=1.0)], F3, {0: d3b, 1: d3b}, r_fine_e, G, ell)
            r0 = res3['records'][0]
            eps3p = eps3 * (1.0 + a.perturb_rel)
            d3p = fl.solve_record_dense(FLD, R_e, eps3p, J3, G, ell_pass, sargs, r_fine_e, cfg)
            res3p = run_analysis([rec3, dict(rec3, T=1.0, T_over_rs=1.0)], F3, {0: d3p, 1: d3p}, r_fine_e, G, ell)
            rp = res3p['records'][0]
            ok = (len(zd) == 2 and types == ['inner', 'outer'] and kd_ <= a.tol_kappa_rel and kf_ <= a.tol_kappa_rel and root.get('margin_eq', 0) < 0.98
                  and r0.get('zeros_reproduced') is True and fl._float(r0.get('max_abs_zero_diff')) <= a.tol_repro and len(res3['pairs']) == 1
                  and rp.get('zeros_reproduced') is False and 'zeros_not_reproduced' in (rp.get('flags') or []) and len(res3p['pairs']) == 0)
            check('E3 broad profile: two zeros (inner, outer), kappa_eq = kappa_fd = formula, the root inside the matter; the re-solve reproduces the record (second solve), the perturbed sources do not', ok,
                  'zeros %s types %s; kappa_eq %s, fd-eq rel %s, formula-eq rel %s, s = 2 R kappa/N_R %s, rho_ell(R) %s, J_ell(R) %s; second solve max|dz| %s (reproduced %s, pairs %d); perturbed (%g) max|dz| %s, count %s, flags %s, pairs %d; %.1f s per solve' % (
                      json.dumps([round(z, 6) for z in zd]), types, fl._fmt(root.get('kappa_eq'), 8), fl._fmt(kd_, 3), fl._fmt(kf_, 3), fl._fmt(root.get('margin_eq'), 5),
                      fl._fmt(root.get('rho_R'), 4), fl._fmt(root.get('J_R'), 4), fl._fmt(r0.get('max_abs_zero_diff'), 3), r0.get('zeros_reproduced'), len(res3['pairs']),
                      a.perturb_rel, fl._fmt(rp.get('max_abs_zero_diff'), 3), rp.get('n_zeros_dense'), rp.get('flags'), len(res3p['pairs']), d3['seconds']))
            # E4 the lam family through the engine
            lam_ce = np.linspace(0.0, 1.0, n_c)
            lam_fe = np.linspace(0.0, 1.0, n_f)
            recs_ce, F_ce, dense_ce = engine_family(FLD, R_e, lam_ce, broad_sources, G, ell_pass, sargs, r_fine_e, cfg, with_kernel=True)
            recs_fe, F_fe, dense_fe = engine_family(FLD, R_e, lam_fe, broad_sources, G, ell_pass, sargs, r_fine_e, cfg, with_kernel=True)
            save_layout(a.out, 'engine_family_coarse', recs_ce, F_ce)
            save_layout(a.out, 'engine_family_fine', recs_fe, F_fe)
            res_ce = run_analysis(recs_ce, F_ce, dense_ce, r_fine_e, G, ell)
            res_fe = run_analysis(recs_fe, F_fe, dense_fe, r_fine_e, G, ell)
            n_ok = all(d.get('ok') for d in list(dense_ce.values()) + list(dense_fe.values()))
            er_ce = np.array([p['res_simpson'] for p in res_ce['pairs']])
            er_fe = np.array([p['res_simpson'] for p in res_fe['pairs']])
            order_e = math.log(rms(er_ce) / rms(er_fe), 2) if (er_ce.size and er_fe.size and rms(er_fe) > 0) else NAN
            cum_ce, cum_fe = abs(float(np.sum(er_ce))) if er_ce.size else NAN, abs(float(np.sum(er_fe))) if er_fe.size else NAN
            order_cum_e = math.log(cum_ce / cum_fe, 2) if (np.isfinite(cum_fe) and cum_fe > 0) else NAN
            dE_rms_e = res_fe['stats'].get('dE_rms', NAN)
            app = max((abs(p['res_applied'] - p['res_collarE']) for p in res_fe['pairs']), default=NAN)
            ok = (n_ok and len(res_ce['pairs']) == n_c - 1 and len(res_fe['pairs']) == n_f - 1 and order_e >= a.order_min and order_cum_e >= a.order_cum_min
                  and er_fe.size > 0 and rms(er_fe) <= a.tol_closure_rel * dE_rms_e and app <= 1e-6 * max(1.0, dE_rms_e))
            check('E4 engine lam family: all pairs included, convergence order (per pair ~3, cumulative ~2), closure at the fine step, res_applied = res_collarE', ok,
                  'solves ok %s; pairs coarse %d/%d fine %d/%d (excluded %s); rms res_simpson coarse %s fine %s -> order %s; cum %s/%s -> order %s; rms res / rms dE = %s (tol %s); max |res_applied - res_collarE| %s; sum dE %s sum col2 %s sum W %s; W/dE median %s; E_collar - E max %s; solves %.1f s' % (
                      n_ok, len(res_ce['pairs']), n_c - 1, len(res_fe['pairs']), n_f - 1, [(e['T_a'], e['reasons']) for e in res_ce['excluded_pairs'] + res_fe['excluded_pairs']],
                      fl._fmt(rms(er_ce) if er_ce.size else NAN, 3), fl._fmt(rms(er_fe) if er_fe.size else NAN, 3), fl._fmt(order_e, 3), fl._fmt(cum_ce, 3), fl._fmt(cum_fe, 3),
                      fl._fmt(order_cum_e, 3), fl._fmt(rms(er_fe) / dE_rms_e if (er_fe.size and dE_rms_e) else NAN, 3), a.tol_closure_rel, fl._fmt(app, 3),
                      fl._fmt(res_fe['stats'].get('sum_dE'), 6), fl._fmt(res_fe['stats'].get('sum_col2'), 6), fl._fmt(res_fe['stats'].get('sum_W_ext'), 6),
                      fl._fmt(res_fe['stats'].get('W_over_dE_median'), 4), fl._fmt(res_fe['stats'].get('E_collar_rel_diff_max_abs'), 3),
                      sum(d['seconds'] for d in list(dense_ce.values()) + list(dense_fe.values()))))
            # E5 the site-array comparisons
            rr = [r for r in res_fe['records'] if r.get('has_root') and r.get('solve_ok')]
            dsr = max((abs(fl._float(r.get('R_densesites_refit_minus_R_dense'))) for r in rr), default=NAN)
            dss = max((fl._float(r.get('max_abs_N_site_minus_dense')) for r in rr), default=NAN)
            ok = np.isfinite(dsr) and dsr <= a.tol_site_refit and np.isfinite(dss) and dss == 0.0
            kern = [(fl._float(r.get('R_applied_minus_R_dense')), fl._float(r.get('shift_pred_3d')), fl._float(r.get('shift_pred_1d')), fl._float(r.get('max_abs_N_applied_minus_dense'))) for r in rr]
            kern_txt = ('kernel average (as N_applied): root - dense root %s, predicted 3D %s, 1D %s, max |N_kernel - N_dense| %s (informational)' % (
                fl._fmt(np.nanmedian([k[0] for k in kern]), 3), fl._fmt(np.nanmedian([k[1] for k in kern]), 3), fl._fmt(np.nanmedian([k[2] for k in kern]), 3),
                fl._fmt(np.nanmax([k[3] for k in kern]), 3))) if kern and np.any(np.isfinite([k[0] for k in kern])) else 'kernel average unavailable (%s)' % (
                    (dense_fe.get(0) or {}).get('kernel_average_error'))
            check('E5 site arrays: the dense solution sampled at the sites reproduces the dense root (lattice interpolation only)', ok,
                  'max |R(dense at sites refit) - R_dense| %s (tol %s); max |N_site - N_dense| %s (0 by construction); %s' % (fl._fmt(dsr, 3), a.tol_site_refit, fl._fmt(dss, 3), kern_txt))

    # ---------------- W1 writers ----------------
    opts_w = fl.make_opts(G=G, ell=ell, rs=1.0, stop_E=1e-12, tag='selftest_family_fine')
    base = fl.write_outputs(res_f, r_fine, a.out, 'selftest_family_fine', 'manufactured', 'manufactured', opts_w)
    ok = all(os.path.exists(base + ext) for ext in ('.json', '.npz', '.md'))
    check('W1 writers produce json/npz/md', ok, base)

    n_fail = sum(1 for r in results if not r['pass'] and not r['skipped'])
    n_skip = sum(1 for r in results if r['skipped'])
    summary = {'results': results, 'n_fail': n_fail, 'n_skip': n_skip, 'seconds': time.time() - t0, 'options': vars(a),
               'family_stats_fine': fl._jsonable(res_f['stats'])}
    with open(os.path.join(a.out, 'firstlaw_selftest_v2.json'), 'w') as fh:
        json.dump(fl._jsonable(summary), fh, indent=1)
    print('firstlaw_selftest_v2: %d checks, %d failed, %d skipped, %.1f s, outputs in %s' % (len(results), n_fail, n_skip, time.time() - t0, a.out))
    if n_fail:
        return 1
    if n_skip and not a.allow_skip:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
