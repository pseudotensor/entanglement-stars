#!/usr/bin/env python3
"""
firstlaw_selftest_v3.py -- manufactured checks for firstlaw_records_v2.py (self-test version 3; the records script stays
version 2, unchanged, and is the module imported here)

Version 2 of this self-test ran on the box with 4 of 14 checks failing.  Read against the engine's code and the output, every
failure is in the test's construction, none in the records script (the builder notes give the decision per check):

  E2  the compact profile's exterior is vacuum: the exterior-relative row defects divide roundoff by a vanishing denominator
      (e = 8.7e-4 of nothing: the row's terms 4 pi r^2 rho_ell are 1e-9 and smaller beyond the root), and the momentum row with
      J = 0 is 0/0 (N = 1 identically, so g = nan with max |g| = 0).  Version 3 audits the compact profiles against the profile's
      own scale (max |defect| over the exterior / max of the row's terms over the whole dense grid), audits the INTERIOR (where
      the matter is) on the dense functions of a direct engine call, adds a compact profile WITH a current so the momentum row
      has content, and keeps the exterior-relative audit (the records' measure) for the broad profile, whose exterior has matter.
  E3  version 2's broad profile (a Gaussian core of 62 at r = 20, w = 4; a halo at r = 60, w = 25; J = 0.1 eps) does not trap
      under the rows the engine integrates: the current reduces the mass function by jf |q| (about 9 percent) and the core's
      width spreads it, so 2 g M_eff(r) < r everywhere and the engine finds no zero (the zero-count comparison 0 = 0 then reads
      "reproduced", correctly).  Version 3 builds the profile from the rows: a compact core (sites 15 to 25) that traps with a
      margin h = 2 g M_eff - r of about +10 at its edge, and a plateau halo of A energy units per site on [32, 70] (C^2 tapers
      on [24, 32] and [70, 90], exactly zero at and beyond 90) on which h decreases at the rate 2 g A (1 - jf) - 1 and crosses
      zero near r = 42 + 6 lam with the simple margin s = 1 - 2 g A (1 - jf) = 0.55 to 0.45: an inner root inside the core and
      an outer root on the plateau, both simple for every lam in [0, 1]; the margin is checked against this design value, which
      also checks the reading "4 pi R^2 rho_ell = the energy per site" of the engine's smoothed sources.
  E4  version 2's family moved the halo's position with lam, so the sources' support and with it the engine's collar node moved
      (collar_changed on every second pair, correctly excluded as it would be for records), and the marginal profile acquired
      its root inside the window (root_appearing, correctly excluded).  Version 3's family keeps the support fixed (the last
      nonzero site is 89 for every lam, so the collar is one node), traps at every lam, and is solved at three record steps on
      the same lam values (5, 9, 17 records; 17 solves): the orders are read from the two finer levels (the 4-pair level is
      pre-asymptotic), the per-pair order from the rms residual, the cumulative order from sum |res| (the signed sum's leading
      term is proportional to the difference of a second lam-derivative at the two ends of the family and can cancel; the
      signed order is printed beside it; the analytic family's F1c keeps version 2's signed measure, which passed).
  E5  version 2 never passed the site arrays into the comparison (solve_record_dense ran before the arrays existed, so
      max |N_site - N_dense| was nan) and read the kernel average's root from the 'applied' key that was never filled.
      Version 3 makes the comparison as the records script's _worker makes it, after the solve, with the dense fields at the
      sites as the stored 'N', 'v' and the engine's kernel average as 'N_applied', 'v_applied': E5a the dense fields at the sites
      (the lattice refit against the dense root), E5b the kernel average's plumbing and its root displacement against the
      leading-order prediction.
  Also: the analytic family's r = 0 sample no longer divides by zero (its rho has an integrable 1/r at the centre; the sample is
  never used, the pair integrals start at R_mid).

Two groups (each line prints PASS/FAIL with the numbers; exit code 1 on any FAIL):
  A. the pair machinery on an exact-rows family without the engine (version 1's analytic family M(r, lam), ln N(r, lam) with
     rho, J defined by the rows, sampled on the dense grid and fed to analyse() as if re-solved): F1a all pairs included; F1b the
     dense row audit of the analytic family; F1c the convergence orders in the record step; F1d closure at the fine step; F1e the
     identity with exact lam-quadrature; F1f the radial rules; X1 the exclusion logic.  Unchanged from version 2.
  B. the engine (imported read-only from --engine-path, the module --engine-module):
     E0 the import; E1 the compact profile (a Gaussian shell, no current; the outer root in vacuum): the engine's zeros include
        the outer zero at 2 g E_collar, E_collar = sum eps_tot, kappa_eq = 1/(2R), kappa_fd = kappa_eq, the own scan = the
        engine's zeros;
     E2 the rows close on the dense solution: compact (J = 0) and compact with a current against the profile's scale (exterior)
        and relative (interior); broad: exterior-relative (the records' measure; the J-flipped exterior defect must exceed
        --flip-min: the sign is fixed by the rows) and interior-relative; all <= --tol-rows-dense; the engine's residuals printed;
     E3 the broad profile: two zeros (inner in the core, outer on the plateau), kappa_eq = kappa_fd = formula, s against the
        design value; the re-solve (a second solve of the same sources) reproduces the manufactured record's zeros to
        --tol-repro; sources perturbed by --perturb-rel are NOT reproduced;
     E4 the lam family through the engine at three record steps: all pairs included at every level, one collar, per-pair order
        >= --order-min and cumulative (sum |res|) order >= --order-cum-min between the two finer levels, closure at the finest
        step, res_applied = res_collarE up to the defect integral;
     E5a the dense fields at the sites refit the dense root to --tol-site-refit and compare exactly with the dense solution;
     E5b the engine's kernel average: its arrays compare with themselves exactly through the 'applied' and 'kernel' paths, its
        refit root is within --tol-kernel-shift of the dense root, the displacement against the prediction is printed;
     W1 the writers.

CPU only; the engine solves run in-process (one at a time).  Writes the manufactured records (json + npz in the engine's layout)
and a summary json to --out.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firstlaw_records_v2 as fl  # noqa: E402  (the records script under test; sets the thread caps from --threads/--workers)

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

# the manufactured broad profile (E3, E4, E5): lattice units (site spacing 1), energies in the units of eps_tot
CORE_R0, CORE_W = 20.0, 6.0               # the core: R^2 (1 - ((R - 20)/6)^2)^4 on the sites 15 to 25 (zero at 14 and 26)
E_CORE0, DE_CORE = 80.0, 8.0              # the core's energy 80 + 8 lam
RISE, FALL = (24.0, 32.0), (70.0, 90.0)   # the plateau's C^2 tapers; exactly zero at and beyond R = 90 (last nonzero site 89)
A_PLATEAU0, DA_PLATEAU = 1.0, 0.3         # the plateau's energy per site A = 1 + 0.3 lam
JF0, DJF = 0.10, 0.05                     # the current fraction J_tot = jf eps_tot, jf = 0.10 + 0.05 lam
CENTRE_E, CENTRE_W = 0.02, 12.0           # a weak smooth central density (0.09 energy units in total: the centre's series
#                                           coefficient a = (8 pi g/3) rho_ell(0) stays far above the solver's atol)


def Phi(x):
    return 0.5 * _erfc(-np.asarray(x, float) / SQRT2)


def phi(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='manufactured checks for firstlaw_records_v2 (self-test version 3)')
    p.add_argument('--out', default='<internal>/firstlaw_selftest_v3', help='output directory')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--workers', type=int, default=1, help='accepted for symmetry; the self-test solves in-process')
    p.add_argument('--engine-path', dest='engine_path', default=None, help="the engine's directory (default: this script's directory)")
    p.add_argument('--engine-module', dest='engine_module', default='engine_fields_v16')
    p.add_argument('--skip-engine', dest='skip_engine', action='store_true', help='group B is skipped (the SKIP line counts as a failure unless --allow-skip)')
    p.add_argument('--allow-skip', dest='allow_skip', action='store_true')
    p.add_argument('--G', type=float, default=0.25)
    p.add_argument('--ell', type=float, default=2.0)
    p.add_argument('--fine', type=int, default=16)
    p.add_argument('--n-sites-family', dest='n_sites_family', type=int, default=2048, help='sites of the analytic family grid (spacing 1)')
    p.add_argument('--n-sites-engine', dest='n_sites_engine', type=int, default=320, help='sites of the engine profiles (spacing 1)')
    p.add_argument('--n-coarse', dest='n_coarse', type=int, default=5, help='records of the coarse families (lam = 0..1); the engine family adds the levels 2(n-1)+1 and 4(n-1)+1')
    p.add_argument('--fine-factor', dest='fine_factor', type=int, default=2, help='the analytic family: records of the fine level = (n_coarse - 1) * fine_factor + 1')
    p.add_argument('--n-gl', dest='n_gl', type=int, default=6)
    p.add_argument('--dlam', type=float, default=1e-4)
    p.add_argument('--fine-radial-factor', dest='fine_radial_factor', type=int, default=4)
    p.add_argument('--rtol', type=float, default=1e-12)
    p.add_argument('--atol', type=float, default=1e-16)
    p.add_argument('--support-rel', dest='support_rel', type=float, default=1e-14)
    p.add_argument('--compact-jfrac', dest='compact_jfrac', type=float, default=0.1, help='the current fraction J_tot/eps_tot of the second compact profile (E2)')
    p.add_argument('--perturb-rel', dest='perturb_rel', type=float, default=1e-3, help='relative perturbation of eps_tot for the non-reproduction check')
    p.add_argument('--tol-kappa-rel', dest='tol_kappa_rel', type=float, default=1e-6)
    p.add_argument('--tol-root', dest='tol_root', type=float, default=1e-6, help='sites')
    p.add_argument('--tol-E-rel', dest='tol_E_rel', type=float, default=1e-6)
    p.add_argument('--tol-vacuum-root-rel', dest='tol_vacuum_root_rel', type=float, default=1e-8, help='|R - 2 g E_collar|/R on the compact profile')
    p.add_argument('--tol-rows-dense', dest='tol_rows_dense', type=float, default=1e-6, help='row defects on the dense solution (relative, or scale-normalized where the row terms vanish)')
    p.add_argument('--tol-g-abs', dest='tol_g_abs', type=float, default=1e-14, help='absolute bound on the momentum-row defect where its terms vanish identically (J = 0, N constant: 0 expected exactly)')
    p.add_argument('--flip-min', dest='flip_min', type=float, default=1e-2, help="the broad profile's exterior Hamiltonian defect with J_ell -> -J_ell must exceed this (the rows fix the sign of J)")
    p.add_argument('--tol-rows-family', dest='tol_rows_family', type=float, default=1e-6)
    p.add_argument('--tol-scan', dest='tol_scan', type=float, default=1e-8, help='sites: own scan against the engine zeros')
    p.add_argument('--tol-repro', dest='tol_repro', type=float, default=1e-9, help='sites: the second solve against the first')
    p.add_argument('--s-design-tol', dest='s_design_tol', type=float, default=0.05, help='|s - (1 - 2 g A (1 - jf))| on the broad profile (A the plateau energy per site, jf the current fraction)')
    p.add_argument('--tol-site-refit', dest='tol_site_refit', type=float, default=5e-3, help='sites: the refit root of the dense solution sampled at the sites against the dense root')
    p.add_argument('--tol-kernel-shift', dest='tol_kernel_shift', type=float, default=1.0, help="sites: |refit root of the engine's kernel average - R_dense| bound (the displacement is a fraction of a site)")
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
        """the fields at r > 0; the r = 0 sample is evaluated at r = 1e-3 (the family's rho has an integrable 1/r at the centre from the
        m_b term; the sample is never used: the pair integrals start at R_mid) with v = J = 0 there."""
        r = np.asarray(r, float)
        pos = r > 0
        rr = np.where(pos, r, 1e-3)
        M = self.M(rr, lam)
        Mp = self.Mp(rr, lam)
        N = np.exp(self.lnN(rr, lam))
        lnNp = self.lnNp(rr, lam)
        u = -np.sqrt(np.maximum(2.0 * self.G * M / rr, 0.0))
        v = u * N
        J = -u * lnNp / (FOUR_PI * self.G * rr)
        rho = Mp / (FOUR_PI * rr * rr) - u * J
        v = np.where(pos, v, 0.0)
        J = np.where(pos, J, 0.0)
        u = np.where(pos, u, 0.0)
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
    """a Gaussian shell (the engine's support cut at --support-rel ends it near R = 53 for these numbers); the outer root in vacuum"""
    s = shell(R, r0, w)
    eps = E * s / s.sum()
    return eps, jfrac * eps


def smoothstep5(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def bump(x):
    x = np.asarray(x, float)
    return np.where(np.abs(x) < 1.0, (1.0 - x * x) ** 4, 0.0)


def plateau(R):
    """1 on [RISE[1], FALL[0]], the quintic smoothstep (C^2) on the tapers, exactly 0 at and beyond the taper ends"""
    R = np.asarray(R, float)
    P = np.ones_like(R)
    P = np.where(R <= RISE[0], 0.0, P)
    m = (R > RISE[0]) & (R < RISE[1])
    P = np.where(m, smoothstep5((R - RISE[0]) / (RISE[1] - RISE[0])), P)
    m = (R > FALL[0]) & (R < FALL[1])
    P = np.where(m, 1.0 - smoothstep5((R - FALL[0]) / (FALL[1] - FALL[0])), P)
    P = np.where(R >= FALL[1], 0.0, P)
    return P


def broad_sources(R, lam=0.0):
    """the per-site sources of the broad profile (module constants): the core E_c = 80 + 8 lam on the sites 15 to 25 (it traps:
    2 g M_eff - r is about +10 at its edge), the plateau A = 1 + 0.3 lam energy units per site on [32, 70] with the tapers [24, 32]
    and [70, 90] (zero at and beyond 90: the last nonzero site 89 fixes the engine's collar node for every lam), a weak smooth
    central density, and the current J_tot = jf eps_tot with jf = 0.10 + 0.05 lam.  On the plateau 4 pi r^2 rho_ell = A, so
    h = 2 g M_eff - r decreases at the rate 2 g A (1 - jf) - 1 and crosses zero near r = 42 + 6 lam with the simple margin
    s = 1 - 2 g A (1 - jf) = 0.55 (lam 0) to 0.45 (lam 1); the inner root sits inside the core near r = 20."""
    R = np.asarray(R, float)
    Ec = E_CORE0 + DE_CORE * lam
    A = A_PLATEAU0 + DA_PLATEAU * lam
    jf = JF0 + DJF * lam
    wc = R * R * bump((R - CORE_R0) / CORE_W)
    eps = Ec * wc / wc.sum() + A * plateau(R) + CENTRE_E * bump(R / CENTRE_W)
    return eps, jf * eps


def broad_margin_design(g, lam):
    A = A_PLATEAU0 + DA_PLATEAU * lam
    jf = JF0 + DJF * lam
    return 1.0 - 2.0 * g * A * (1.0 - jf)


def broad_description(g):
    return ('core %g + %g lam on sites %d..%d; plateau A = %g + %g lam per site on [%g, %g], tapers [%g, %g] and [%g, %g], zero beyond; '
            'centre %g x bump(R/%g); jf = %g + %g lam; design s = 1 - 2 g A (1 - jf) = %s (lam 0) to %s (lam 1)' % (
                E_CORE0, DE_CORE, int(CORE_R0 - CORE_W) + 1, int(CORE_R0 + CORE_W) - 1, A_PLATEAU0, DA_PLATEAU, RISE[1], FALL[0], RISE[0], RISE[1],
                FALL[0], FALL[1], CENTRE_E, CENTRE_W, JF0, DJF, fl._fmt(broad_margin_design(g, 0.0), 4), fl._fmt(broad_margin_design(g, 1.0), 4)))


def record_from_dense(d, lam):
    eng = d.get('engine') or {}
    root = d.get('root') or {}
    E = d['E_collar']
    extra = {'regime': {'label': (eng.get('regime') or {}).get('label') or 'in the regime', 'intervals_above_1': (eng.get('regime') or {}).get('intervals_above_1') or []},
             'gauge': {'label': (eng.get('gauge') or {}).get('label') or 'intact'},
             'interp_floor_N': eng.get('interp_floor_N'), 'interp_floor_v': eng.get('interp_floor_v'),
             'solver': {'W_tail': eng.get('W_tail') or {'E_tail_end': E, 'E_collar': E}, 'module': eng.get('module'), 'rtol': eng.get('rtol'), 'atol': eng.get('atol')}}
    return make_record(lam, E, d['zeros_dense'], root.get('kappa_eq'), root.get('N_R'), d['r_collar'], extra=extra)


def sites_post_hoc(R, r_fine, d, eps, J, g, cfg, kern):
    """the site comparison as the records script's _worker makes it, made after the solve: the stored 'N', 'v' are the dense fields
    at the sites (the engine's own N_sites), the stored 'N_applied', 'v_applied' the engine's kernel average when it ran; the smoothed
    sources at the sites by interpolation of the dense arrays (exact: the sites lie on the dense grid)."""
    a = d['arrays']
    r_fine = np.asarray(r_fine, float)
    rhof = lambda x: np.interp(np.abs(np.asarray(x, float)), r_fine, np.asarray(a['rho'], float))  # noqa: E731
    Jf = lambda x: np.interp(np.abs(np.asarray(x, float)), r_fine, np.asarray(a['J'], float))  # noqa: E731
    root = d.get('root') or {}
    N_app, v_app = (kern[0], kern[1]) if kern is not None else (None, None)
    return fl.site_comparisons(np.asarray(R, float), a['N_sites'], a['v_sites'], a['N_sites'], a['v_sites'], N_app, v_app, np.asarray(eps, float),
                               np.asarray(J, float), rhof, Jf, g, float(d['r_collar']), root.get('R'), cfg, kernel=kern)


def engine_family(FLD, R, lams, source_fn, g, ell, sargs, r_fine, cfg):
    """the family solved record by record through solve_record_dense (the records script's path), then the site comparison per
    record (sites_post_hoc).  Returns (records, F, dense, kernel_ok, kernel_error)."""
    records, dense = [], {}
    n = len(lams)
    eps_all = np.zeros((n, R.size))
    J_all = np.zeros((n, R.size))
    Ns = np.full((n, R.size), np.nan)
    vs = np.full((n, R.size), np.nan)
    Nk = np.full((n, R.size), np.nan)
    vk = np.full((n, R.size), np.nan)
    kernel_ok, kernel_err = True, None
    for i, lam in enumerate(lams):
        eps, J = source_fn(R, lam)
        d = fl.solve_record_dense(FLD, R, eps, J, g, ell, sargs, r_fine, cfg)
        d['row'] = i
        dense[i] = d
        eps_all[i], J_all[i] = eps, J
        if not d.get('ok'):
            records.append(make_record(lam, NAN, [], None, None, NAN))
            continue
        a = d['arrays']
        Ns[i], vs[i] = a['N_sites'], a['v_sites']
        kern = None
        if 'N_kernel' in a and 'v_kernel' in a:
            Nk[i], vk[i] = a['N_kernel'], a['v_kernel']
            kern = (np.asarray(a['N_kernel'], float), np.asarray(a['v_kernel'], float))
        else:
            kernel_ok = False
            kernel_err = d.get('kernel_average_error') or 'no kernel arrays returned by solve_record_dense (check_applied set)'
        try:
            d['sites'] = sites_post_hoc(R, r_fine, d, eps, J, g, cfg, kern)
        except Exception as exc:
            d['sites'] = {'error': '%s: %s' % (type(exc).__name__, exc)}
        records.append(record_from_dense(d, lam))
    F = {'R': R, 'T': np.asarray(lams, float), 'eps_tot': eps_all, 'J_tot': J_all, 'eps_pairs': None, 'J_pairs': None, 'N_site': Ns, 'v_site': vs,
         'N_applied': Nk if kernel_ok else None, 'v_applied': vk if kernel_ok else None, 'G': g, 'ell': ell,
         'keys': ['R', 'T', 'N', 'v', 'eps_tot', 'J_tot', 'g', 'ell'] + (['N_applied', 'v_applied'] if kernel_ok else []), 'solver_string': None, 'label': None}
    return records, F, dense, kernel_ok, kernel_err


def subfamily(records, F, dense, step):
    """every step-th record of a family (the same lam values at a coarser record step; no new solves)"""
    idx = list(range(0, len(records), step))
    recs = [records[k] for k in idx]
    Fs = dict(F)
    for key in ('T', 'eps_tot', 'J_tot', 'N_site', 'v_site', 'N_applied', 'v_applied'):
        if F.get(key) is not None:
            Fs[key] = np.asarray(F[key])[idx]
    dn = {}
    for i, k in enumerate(idx):
        d = dict(dense[k])
        d['row'] = i
        dn[i] = d
    return recs, Fs, dn


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


def engine_info(FLD, R, eps, J, g, ell, sargs):
    """a direct call of the engine's solver for its dense functions (solve_record_dense keeps only arrays); the same arguments"""
    try:
        res = FLD.solve_constraints_ivp(np.asarray(eps, float), np.asarray(J, float), g, np.asarray(R, float), ell, **sargs)
        N_s, v_s, W_nodes, info = res
    except Exception as exc:
        return None, 'solve_constraints_ivp raised %s: %s' % (type(exc).__name__, exc)
    if N_s is None or info.get('failed'):
        return None, "the engine's solve failed: %s" % info.get('failed')
    return info, None


def interior_audit(FLD, R, eps, J, g, ell, sargs, d, r_fine, h_fd, r_min):
    """the records script's dense_row_audit on the interior (r_min + 2h, R_outer - 2h) of a direct engine call's dense functions"""
    info, err = engine_info(FLD, R, eps, J, g, ell, sargs)
    if info is None:
        return None, err
    zeros = d.get('zeros_dense') or []
    hi = (max(zeros) if zeros else float(d['r_collar'])) - 2.0 * h_fd
    mask = (r_fine > r_min + 2.0 * h_fd) & (r_fine < hi)
    au = fl.dense_row_audit(r_fine[mask], info['N_dense'], info['v_dense'], info['W_field'], info['rho_ell'], info['J_ell'], g, h_fd)
    return au, None


def scaled_defects(d, au, r_fine, h_fine):
    """the audit's max |e| and max |g| over its points against the profile's own scale: the maxima over the WHOLE dense grid of the
    rows' terms 4 pi r^2 (|rho_ell| + |u J_ell|) and |N'/N| (the 4th-order stencil on the dense arrays).  Returns
    (e_scaled, g_scaled, scale_e, scale_g); scale_g is 0 on a current-free profile (N constant), g_scaled nan then."""
    a = d.get('arrays') or {}
    if not all(k in a for k in ('N', 'v', 'rho', 'J')):
        return NAN, NAN, NAN, NAN
    N, v, rho, J = (np.asarray(a[k], float) for k in ('N', 'v', 'rho', 'J'))
    r = np.asarray(r_fine, float)
    Ns = np.where(N != 0.0, N, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(N != 0.0, v / Ns, 0.0)
        terms_e = FOUR_PI * r * r * (np.abs(rho) + np.abs(u * J))
        Np = fl.derivative(N, r, h_fine)
        terms_g = np.abs(np.where(N != 0.0, Np / Ns, 0.0))
    scale_e = fl._nan(np.max, terms_e)
    scale_g = fl._nan(np.max, terms_g)
    e_abs, g_abs = fl._float(au.get('max_abs_e')), fl._float(au.get('max_abs_g'))
    e_sc = e_abs / scale_e if (np.isfinite(scale_e) and scale_e > 0) else NAN
    g_sc = g_abs / scale_g if (np.isfinite(scale_g) and scale_g > 0) else NAN
    return e_sc, g_sc, scale_e, scale_g


# ----------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    G, ell = a.G, a.ell
    results = []

    def rms(x):
        x = np.asarray(x, float)
        return float(np.sqrt(np.mean(x ** 2))) if x.size else NAN

    def l1(x):
        x = np.asarray(x, float)
        return float(np.sum(np.abs(x))) if x.size else NAN

    def order2(coarse, fine):
        return math.log(coarse / fine, 2) if (np.isfinite(coarse) and np.isfinite(fine) and coarse > 0 and fine > 0) else NAN

    def med(vals):
        return fl._nan(np.median, vals)

    def mx(vals):
        return fl._nan(np.max, vals)

    def mn(vals):
        return fl._nan(np.min, vals)

    def check(name, ok, detail, skipped=False):
        status = 'SKIP' if skipped else ('PASS' if ok else 'FAIL')
        results.append({'name': name, 'pass': bool(ok) and not skipped, 'skipped': bool(skipped), 'detail': detail})
        print('%s  %s  %s' % (status, name, detail))

    h_fd = ell / 64.0

    # =========================== A. the exact-rows family (version 2's checks) ===========================
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
    order_W = order2(rms(eW_c), rms(eW_f))
    order_r = order2(rms(er_c), rms(er_f))
    cum_c, cum_f = abs(float(np.sum(er_c))), abs(float(np.sum(er_f)))
    order_cum = order2(cum_c, cum_f)
    ok = order_W >= a.order_min and order_r >= a.order_min and order_cum >= a.order_cum_min
    check('F1c convergence order in the record step (per pair ~3, cumulative ~2)', ok,
          'order |W - W_exact| %s, order res %s, cumulative order %s (sum |res| order %s); rms per pair coarse/fine: W err %s/%s, res %s/%s; cum res %s/%s' % (
              fl._fmt(order_W, 3), fl._fmt(order_r, 3), fl._fmt(order_cum, 3), fl._fmt(order2(l1(er_c), l1(er_f)), 3), fl._fmt(rms(eW_c), 3), fl._fmt(rms(eW_f), 3),
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
    engine_names = ['E2 the rows close on the dense solution (compact profiles against the profile scale; broad relative; interior from a direct call)',
                    'E3 broad profile: inner and outer zeros, the three kappas, s against the design; the second solve reproduces the record, the perturbed sources do not',
                    'E4 engine lam family at three record steps: all pairs included, one collar, per-pair order ~3 and sum|res| order ~2 (finer levels), closure, res_applied = res_collarE',
                    'E5a site arrays: the dense fields at the sites refit the dense root (lattice interpolation only) and compare exactly with the dense solution',
                    'E5b the engine\'s kernel average: its arrays through the applied and kernel paths, its root displacement against the prediction']
    if a.skip_engine:
        check('E0 engine import', False, 'skipped by --skip-engine', skipped=True)
    else:
        try:
            FLD, eprov = fl.import_engine(a.engine_path or os.path.dirname(os.path.abspath(__file__)), a.engine_module)
            check('E0 engine import', True, '%s from %s; files %s' % (a.engine_module, eprov['module_file'], sorted(eprov['files'].keys())))
        except SystemExit as exc:
            check('E0 engine import', False, str(exc))
    res_ff = None
    if FLD is not None:
        R_e = np.arange(1, a.n_sites_engine + 1, dtype=float)
        ell_pass = fl.ell_for_engine(ell)
        sargs, aspec, _sf = fl.resolve_solver_args(FLD, {}, {}, {'kv': {}}, fl.make_opts(rtol=a.rtol, atol=a.atol, collar_policy='flag', support_rel=a.support_rel))
        r_fine_e, r_last_e, gnote = fl.build_fine_grid(FLD, R_e, ell_pass, a.fine)
        h_fine_e = ell / float(a.fine)
        cfg = fl.worker_cfg(fl.make_opts(check_applied=True), r_last_e, h_fine_e)
        print('engine solve args %s; dense grid to %s step %s%s' % (json.dumps(fl._jsonable(sargs)), fl._fmt(r_last_e), fl._fmt(h_fine_e), ('; ' + gnote) if gnote else ''))
        print('broad profile: %s' % broad_description(G))

        # ---------------- E1 the compact profile (no current) ----------------
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

        # ---------------- the second compact profile (with a current) and the broad profile ----------------
        eps2, J2 = compact_sources(R_e, jfrac=a.compact_jfrac)
        d2 = fl.solve_record_dense(FLD, R_e, eps2, J2, G, ell_pass, sargs, r_fine_e, cfg)
        eps3, J3 = broad_sources(R_e, 0.0)
        d3 = fl.solve_record_dense(FLD, R_e, eps3, J3, G, ell_pass, sargs, r_fine_e, cfg)
        failed = [(lab, d) for lab, d in (('compact (J = 0)', d1), ('compact with current', d2), ('broad', d3)) if not d.get('ok')]
        if failed:
            msg = 'a solve failed: ' + '; '.join('%s: %s %s' % (lab, d.get('failed'), str(d.get('traceback', ''))[-400:]) for lab, d in failed)
            for nm in engine_names:
                check(nm, False, msg)
        else:
            # ---------------- E2 the rows on the dense solution ----------------
            tol = a.tol_rows_dense
            ok2 = True
            parts = []
            for lab, eps_, J_, d_, mode in (('compact (J = 0)', eps1, J1, d1, 'scale'), ('compact with current %g' % a.compact_jfrac, eps2, J2, d2, 'scale'),
                                            ('broad', eps3, J3, d3, 'relative')):
                au = d_.get('audit') or {}
                e_sc, g_sc, scale_e, scale_g = scaled_defects(d_, au, r_fine_e, h_fine_e)
                aui, err = interior_audit(FLD, R_e, eps_, J_, G, ell_pass, sargs, d_, r_fine_e, h_fd, float(cfg['r_min_zero']))
                aui = aui or {}
                rel_e, rel_g = fl._float(au.get('rel_e')), fl._float(au.get('rel_g'))
                g_abs = fl._float(au.get('max_abs_g'))
                flip_e, flip_g = fl._float(au.get('rel_e_flipJ')), fl._float(au.get('rel_g_flipJ'))
                rel_e_i, rel_g_i, g_abs_i = fl._float(aui.get('rel_e')), fl._float(aui.get('rel_g')), fl._float(aui.get('max_abs_g'))
                eng = d_.get('engine') or {}
                if mode == 'scale':
                    g_ok = (g_sc <= tol) if (np.isfinite(g_sc) and scale_g > 0) else (np.isfinite(g_abs) and g_abs <= a.tol_g_abs)
                    gi_ok = (rel_g_i <= tol) if np.isfinite(rel_g_i) else (np.isfinite(g_abs_i) and g_abs_i <= a.tol_g_abs)
                    ok_p = (np.isfinite(e_sc) and e_sc <= tol and g_ok and np.isfinite(rel_e_i) and rel_e_i <= tol and gi_ok and err is None)
                    parts.append('%s: exterior (%s pts) scale-normalized e %s, %s (scales e %s, g %s; the relative e %s, g %s there are roundoff over a vacuum: not gated); '
                                 'interior (%s pts) relative e %s, %s; engine residual_W %s, residual_invN %s%s' % (
                                     lab, au.get('n'), fl._fmt(e_sc, 3),
                                     ('g %s' % fl._fmt(g_sc, 3)) if (np.isfinite(g_sc) and scale_g > 0) else ('max|g| %s (N = 1: the row is 0 identically, tol %g)' % (fl._fmt(g_abs, 3), a.tol_g_abs)),
                                     fl._fmt(scale_e, 3), fl._fmt(scale_g, 3), fl._fmt(rel_e, 3), fl._fmt(rel_g, 3), aui.get('n'), fl._fmt(rel_e_i, 3),
                                     ('g %s' % fl._fmt(rel_g_i, 3)) if np.isfinite(rel_g_i) else ('max|g| %s' % fl._fmt(g_abs_i, 3)),
                                     fl._fmt(eng.get('residual_W'), 3), fl._fmt(eng.get('residual_invN'), 3), ('; interior audit unavailable: %s' % err) if err else ''))
                else:
                    ok_p = (np.isfinite(rel_e) and rel_e <= tol and np.isfinite(rel_g) and rel_g <= tol and np.isfinite(rel_e_i) and rel_e_i <= tol
                            and np.isfinite(rel_g_i) and rel_g_i <= tol and np.isfinite(flip_e) and flip_e >= a.flip_min and err is None)
                    parts.append('%s: exterior (%s pts) relative e %s, g %s (J_ell -> -J_ell: e %s, g %s; must exceed %g), W form %s, scale-normalized e %s g %s; '
                                 'interior (%s pts) relative e %s, g %s; engine residual_W %s, residual_invN %s%s' % (
                                     lab, au.get('n'), fl._fmt(rel_e, 3), fl._fmt(rel_g, 3), fl._fmt(flip_e, 3), fl._fmt(flip_g, 3), a.flip_min, fl._fmt(au.get('rel_W'), 3),
                                     fl._fmt(e_sc, 3), fl._fmt(g_sc, 3), aui.get('n'), fl._fmt(rel_e_i, 3), fl._fmt(rel_g_i, 3),
                                     fl._fmt(eng.get('residual_W'), 3), fl._fmt(eng.get('residual_invN'), 3), ('; interior audit unavailable: %s' % err) if err else ''))
                ok2 = ok2 and ok_p
            check(engine_names[0], ok2, '(Richardson h = ell/64; tol %g) ' % tol + ' | '.join(parts))

            # ---------------- E3 the broad profile ----------------
            zd = d3['zeros_dense']
            det = d3['zeros_detail']
            root = d3['root'] or {}
            types = [z['type'] for z in det]
            R_in = min(zd) if zd else NAN
            R_out = max(zd) if zd else NAN
            s_eq = fl._float(root.get('margin_eq'))
            s_des = broad_margin_design(G, 0.0)
            k_eq = fl._float(root.get('kappa_eq'))
            kd_ = abs(fl._float(root.get('kappa_fd')) - k_eq) / abs(k_eq) if (np.isfinite(k_eq) and k_eq != 0) else NAN
            kf_ = abs(fl._float(root.get('kappa_formula')) - k_eq) / abs(k_eq) if (np.isfinite(k_eq) and k_eq != 0) else NAN
            fourpiR2rho = FOUR_PI * R_out ** 2 * fl._float(root.get('rho_R')) if np.isfinite(R_out) else NAN
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
            in_lo, in_hi = CORE_R0 - CORE_W, CORE_R0 + CORE_W
            out_lo, out_hi = RISE[1] + ell, FALL[0] - ell
            ok3 = (len(zd) == 2 and types == ['inner', 'outer'] and in_lo <= R_in <= in_hi and out_lo <= R_out <= out_hi
                   and kd_ <= a.tol_kappa_rel and kf_ <= a.tol_kappa_rel and np.isfinite(s_eq) and abs(s_eq - s_des) <= a.s_design_tol
                   and r0.get('zeros_reproduced') is True and fl._float(r0.get('max_abs_zero_diff')) <= a.tol_repro and len(res3['pairs']) == 1
                   and d3b.get('ok') is True and d3p.get('ok') is True
                   and rp.get('zeros_reproduced') is False and 'zeros_not_reproduced' in (rp.get('flags') or []) and len(res3p['pairs']) == 0)
            check(engine_names[1], ok3,
                  'zeros %s types %s (design: inner in [%g, %g], outer on the plateau [%g, %g]); kappa_eq %s, fd-eq rel %s, formula-eq rel %s (tol %g); s = 2 R kappa/N_R %s vs design %s (tol %g); '
                  '4 pi R^2 rho_ell(R) = %s vs A = %g, J_ell/rho_ell at R %s vs jf %g; r_collar %s; E_collar %s (sum eps %s); regime %s; gauge %s; '
                  'second solve max|dz| %s (reproduced %s, pairs %d); perturbed (%g) max|dz| %s, count %s vs 2, flags %s, pairs %d; %.1f s per solve' % (
                      json.dumps([round(z, 6) for z in zd]), types, in_lo, in_hi, out_lo, out_hi, fl._fmt(k_eq, 8), fl._fmt(kd_, 3), fl._fmt(kf_, 3), a.tol_kappa_rel,
                      fl._fmt(s_eq, 5), fl._fmt(s_des, 5), a.s_design_tol, fl._fmt(fourpiR2rho, 5), A_PLATEAU0,
                      fl._fmt(fl._float(root.get('J_R')) / fl._float(root.get('rho_R')) if fl._float(root.get('rho_R')) else NAN, 4), JF0,
                      fl._fmt(d3['r_collar']), fl._fmt(d3['E_collar'], 9), fl._fmt(eps3.sum(), 9), ((d3.get('engine') or {}).get('regime') or {}).get('label'),
                      ((d3.get('engine') or {}).get('gauge') or {}).get('label'), fl._fmt(r0.get('max_abs_zero_diff'), 3), r0.get('zeros_reproduced'), len(res3['pairs']),
                      a.perturb_rel, fl._fmt(rp.get('max_abs_zero_diff'), 3), rp.get('n_zeros_dense'), rp.get('flags'), len(res3p['pairs']), d3['seconds']))

            # ---------------- E4 the lam family through the engine, three record steps on the same lam values ----------------
            n_ff = 4 * (n_c - 1) + 1
            lam_ff = np.linspace(0.0, 1.0, n_ff)
            t_fam = time.time()
            recs_ff, F_ff, dense_ff, kern_ok, kern_err = engine_family(FLD, R_e, lam_ff, broad_sources, G, ell_pass, sargs, r_fine_e, cfg)
            fam_secs = time.time() - t_fam
            recs_fe, F_fe, dense_fe = subfamily(recs_ff, F_ff, dense_ff, 2)
            recs_ce, F_ce, dense_ce = subfamily(recs_ff, F_ff, dense_ff, 4)
            save_layout(a.out, 'engine_family_coarse', recs_ce, F_ce)
            save_layout(a.out, 'engine_family_fine', recs_fe, F_fe)
            save_layout(a.out, 'engine_family_finest', recs_ff, F_ff)
            res_ce = run_analysis(recs_ce, F_ce, dense_ce, r_fine_e, G, ell)
            res_fe = run_analysis(recs_fe, F_fe, dense_fe, r_fine_e, G, ell)
            res_ff = run_analysis(recs_ff, F_ff, dense_ff, r_fine_e, G, ell)
            n_ok = all(d.get('ok') for d in dense_ff.values())
            collars = sorted(set(round(float(d['r_collar']), 9) for d in dense_ff.values() if d.get('ok')))

            def resid(res):
                return np.array([p['res_simpson'] for p in res['pairs']], float)
            er_ce, er_fe, er_ff = resid(res_ce), resid(res_fe), resid(res_ff)
            o_pair_cf, o_pair_fff = order2(rms(er_ce), rms(er_fe)), order2(rms(er_fe), rms(er_ff))
            o_l1_cf, o_l1_fff = order2(l1(er_ce), l1(er_fe)), order2(l1(er_fe), l1(er_ff))
            sgn = [abs(float(np.sum(x))) if x.size else NAN for x in (er_ce, er_fe, er_ff)]
            o_sgn_cf, o_sgn_fff = order2(sgn[0], sgn[1]), order2(sgn[1], sgn[2])
            dE_rms_ff = fl._float(res_ff['stats'].get('dE_rms'))
            app = mx([abs(p['res_applied'] - p['res_collarE']) for p in res_ff['pairs']])
            n_pairs_want = (n_c - 1, 2 * (n_c - 1), 4 * (n_c - 1))
            n_pairs_got = (len(res_ce['pairs']), len(res_fe['pairs']), len(res_ff['pairs']))
            excl = [(round(e['T_a'], 4), e['reasons']) for e in res_ce['excluded_pairs'] + res_fe['excluded_pairs'] + res_ff['excluded_pairs']]
            closure = rms(er_ff) / dE_rms_ff if (np.isfinite(dE_rms_ff) and dE_rms_ff > 0) else NAN
            ok4 = (n_ok and len(collars) == 1 and n_pairs_got == n_pairs_want and o_pair_fff >= a.order_min and o_l1_fff >= a.order_cum_min
                   and np.isfinite(closure) and closure <= a.tol_closure_rel and np.isfinite(app) and app <= 1e-6 * max(1.0, dE_rms_ff))
            stf = res_ff['stats']
            check(engine_names[2], ok4,
                  'solves ok %s (%d records, %.1f s); collar %s; pairs %d/%d, %d/%d, %d/%d (excluded %s); rms res_simpson %s / %s / %s -> per-pair orders %s (4->8), %s (8->16; min %g); '
                  'sum |res| %s / %s / %s -> orders %s, %s (min %g); signed sum %s / %s / %s -> orders %s, %s (printed; cancellation %s); '
                  'finest step: rms res / rms dE = %s (tol %g), max |res_applied - res_collarE| %s, sum dE %s, sum col2 %s, sum W %s, W/dE median %s, '
                  's from %s to %s, R from %s to %s, kappa_eq from %s to %s, E_collar - E max %s' % (
                      n_ok, n_ff, fam_secs, collars, n_pairs_got[0], n_pairs_want[0], n_pairs_got[1], n_pairs_want[1], n_pairs_got[2], n_pairs_want[2], excl,
                      fl._fmt(rms(er_ce), 3), fl._fmt(rms(er_fe), 3), fl._fmt(rms(er_ff), 3), fl._fmt(o_pair_cf, 3), fl._fmt(o_pair_fff, 3), a.order_min,
                      fl._fmt(l1(er_ce), 3), fl._fmt(l1(er_fe), 3), fl._fmt(l1(er_ff), 3), fl._fmt(o_l1_cf, 3), fl._fmt(o_l1_fff, 3), a.order_cum_min,
                      fl._fmt(sgn[0], 3), fl._fmt(sgn[1], 3), fl._fmt(sgn[2], 3), fl._fmt(o_sgn_cf, 3), fl._fmt(o_sgn_fff, 3),
                      fl._fmt(sgn[2] / l1(er_ff) if (np.isfinite(l1(er_ff)) and l1(er_ff) > 0) else NAN, 3),
                      fl._fmt(closure, 3), a.tol_closure_rel, fl._fmt(app, 3), fl._fmt(stf.get('sum_dE'), 6), fl._fmt(stf.get('sum_col2'), 6), fl._fmt(stf.get('sum_W_ext'), 6),
                      fl._fmt(stf.get('W_over_dE_median'), 4), fl._fmt(stf.get('simple_margin_eq_min'), 4), fl._fmt(stf.get('simple_margin_eq_max'), 4),
                      fl._fmt(stf.get('R_first'), 6), fl._fmt(stf.get('R_last'), 6), fl._fmt(mn([r.get('kappa_eq') for r in res_ff['records']]), 5),
                      fl._fmt(mx([r.get('kappa_eq') for r in res_ff['records']]), 5), fl._fmt(stf.get('E_collar_rel_diff_max_abs'), 3)))

            # ---------------- E5a the dense fields at the sites ----------------
            rr = [r for r in res_ff['records'] if r.get('has_root') and r.get('solve_ok')]
            dsr = mx([abs(fl._float(r.get('R_densesites_refit_minus_R_dense'))) for r in rr])
            dss = mx([fl._float(r.get('max_abs_N_site_minus_dense')) for r in rr])
            dsv = mx([fl._float(r.get('max_abs_v_site_minus_dense')) for r in rr])
            dsite = mx([abs(fl._float(r.get('R_site_minus_R_dense'))) for r in rr])
            site_err = [d.get('sites', {}).get('error') for d in dense_ff.values() if isinstance(d.get('sites'), dict) and d.get('sites', {}).get('error')]
            ok5a = (bool(rr) and np.isfinite(dsr) and dsr <= a.tol_site_refit and dss == 0.0 and dsv == 0.0 and not site_err)
            check(engine_names[3], ok5a,
                  '%d rooted records; max |R(dense at sites refit) - R_dense| %s (tol %g; the stored arrays\' refit %s); max |N_site - N_dense(R_n)| %s, |v_site - v_dense| %s '
                  '(0: the stored arrays are the engine\'s own dense fields at the sites); row defects at the sites with the smoothed sources (4th-order stencil, spacing 1): '
                  'e median %s, g median %s; the raw reading (eps_tot, J_tot as densities) e median %s%s' % (
                      len(rr), fl._fmt(dsr, 3), a.tol_site_refit, fl._fmt(dsite, 3), fl._fmt(dss, 3), fl._fmt(dsv, 3),
                      fl._fmt(stf.get('dense_at_sites_row_e_rel_ext_median'), 3), fl._fmt(stf.get('dense_at_sites_row_g_rel_ext_median'), 3),
                      fl._fmt(stf.get('site_row_e_rel_ext_raw_median'), 3), ('; site comparison errors: %s' % site_err[:2]) if site_err else ''))

            # ---------------- E5b the engine's kernel average ----------------
            if kern_ok:
                sh_app = [fl._float(r.get('R_applied_minus_R_dense')) for r in rr]
                sh_ker = [fl._float(r.get('R_kernel_minus_R_dense')) for r in rr]
                pred3 = [fl._float(r.get('shift_pred_3d')) for r in rr]
                pred1 = [fl._float(r.get('shift_pred_1d')) for r in rr]
                dNk = mx([fl._float(r.get('max_abs_N_applied_minus_dense')) for r in rr])
                dvk = mx([fl._float(r.get('max_abs_v_applied_minus_dense')) for r in rr])
                dak = mx([fl._float(r.get('max_abs_N_applied_minus_kernel')) for r in rr])
                davk = mx([fl._float(r.get('max_abs_v_applied_minus_kernel')) for r in rr])
                dpath = mx([abs(x - y) for x, y in zip(sh_app, sh_ker)])
                norms = [d.get('kernel_norms') for d in dense_ff.values() if d.get('ok') and d.get('kernel_norms')]
                nmin = mn([n_['min'] for n_ in norms])
                nmax = mx([n_['max'] for n_ in norms])
                ok5b = (bool(rr) and all(np.isfinite(x) for x in sh_app) and all(np.isfinite(x) for x in sh_ker) and mx([abs(x) for x in sh_app]) <= a.tol_kernel_shift
                        and dak == 0.0 and davk == 0.0 and np.isfinite(dpath) and dpath <= 1e-12 and np.isfinite(dNk))
                check(engine_names[4], ok5b,
                      'kernel average root - dense root: median %s (min %s, max %s; bound %g); predicted -(ell^2/2)[(N+v)\'\'/(N+v)\' + 2/R] median %s (1D form %s); '
                      'max |N_kernel - N_dense(R_n)| %s, |v_kernel - v_dense| %s; applied (:= kernel) against kernel: N %s, v %s (0); |R_applied - R_kernel| max %s; '
                      'kernel norms [%s, %s]; row defects of the kernel-averaged arrays at the sites e median %s, g median %s' % (
                          fl._fmt(med(sh_app), 4), fl._fmt(mn(sh_app), 4), fl._fmt(mx(sh_app), 4), a.tol_kernel_shift, fl._fmt(med(pred3), 4), fl._fmt(med(pred1), 4),
                          fl._fmt(dNk, 3), fl._fmt(dvk, 3), fl._fmt(dak, 3), fl._fmt(davk, 3), fl._fmt(dpath, 3), fl._fmt(nmin, 6), fl._fmt(nmax, 6),
                          fl._fmt(stf.get('kernel_row_e_rel_ext_median'), 3), fl._fmt(stf.get('kernel_row_g_rel_ext_median'), 3)))
            else:
                check(engine_names[4], False, "the engine's site_fields_kernel_average did not provide the kernel arrays: %s (the records' --check-applied depends on it)" % kern_err)

    # ---------------- W1 writers ----------------
    opts_w = fl.make_opts(G=G, ell=ell, rs=1.0, stop_E=1e-12, tag='selftest_family_fine')
    base = fl.write_outputs(res_f, r_fine, a.out, 'selftest_family_fine', 'manufactured', 'manufactured', opts_w)
    ok = all(os.path.exists(base + ext) for ext in ('.json', '.npz', '.md'))
    check('W1 writers produce json/npz/md', ok, base)

    n_fail = sum(1 for r in results if not r['pass'] and not r['skipped'])
    n_skip = sum(1 for r in results if r['skipped'])
    summary = {'results': results, 'n_fail': n_fail, 'n_skip': n_skip, 'seconds': time.time() - t0, 'options': vars(a),
               'broad_profile': broad_description(G), 'family_stats_fine': fl._jsonable(res_f['stats']),
               'engine_family_stats_finest': fl._jsonable(res_ff['stats']) if res_ff is not None else None}
    with open(os.path.join(a.out, 'firstlaw_selftest_v3.json'), 'w') as fh:
        json.dump(fl._jsonable(summary), fh, indent=1)
    print('firstlaw_selftest_v3: %d checks, %d failed, %d skipped, %.1f s, outputs in %s' % (len(results), n_fail, n_skip, time.time() - t0, a.out))
    if n_fail:
        return 1
    if n_skip and not a.allow_skip:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
