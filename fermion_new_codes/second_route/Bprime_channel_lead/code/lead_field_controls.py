"""Manufactured source/kernel/field checks; invoked only after source admission."""
from __future__ import annotations
import math
import numpy as np
from types import SimpleNamespace
from lead_gaussian import gaussian_kernels
from lead_fields import RadialProfile, back_average, field_majorants, gaussian_tail
from lead_field_core import GaussianSources, SolvedField


def run_field_controls():
    checks, arrays = [], {}
    def check(name, measured, threshold):
        checks.append({'name': name, 'measured': float(measured),
                       'threshold': float(threshold), 'passed': bool(measured <= threshold)})

    # An independent angular integral of the unnormalized 3D Gaussian.
    z, w = np.polynomial.legendre.leggauss(96)
    pairs = [(0., 2.), (1e-6, 4.), (.5, 1.), (8., 4.), (20., 17.)]
    own, reference = [], []
    for r, s in pairs:
        k0, k1 = gaussian_kernels(r, np.array([s]), 2.)
        angular = (2*np.pi*4.)**(-1.5)*np.exp(-(r*r+s*s-2*r*s*z)/8.)
        own.append([k0[0], k1[0]])
        reference.append([np.dot(w, angular)/2, np.dot(w, z*angular)/2])
    own, reference = np.array(own), np.array(reference)
    arrays['kernel_own'], arrays['kernel_angular_oracle'] = own, reference
    check('scalar_vector_kernel_independent_angular_rule', np.max(np.abs(own-reference)), 5e-12)

    class ConstantProfile:
        family = 'manufactured_constant'
        def __call__(self, radius):
            return np.ones_like(radius), np.zeros_like(radius)
        def majorants(self):
            return {'N_upper': 1., 'v_abs_upper': 0., 'method': 'exact constant'}
    queries = np.array([0., 1e-5, .25, 2., 8., 16., 100.])
    constant, metadata = back_average(ConstantProfile(), queries)
    arrays['constant_queries'], arrays['constant_average'] = queries, constant
    check('Gaussian_constant_without_face_normalization', np.max(np.abs(constant[:, 0]-1)),
          metadata['gaussian_tail']['mass_upper']+5e-12)
    check('Gaussian_zero_vector_at_origin_and_everywhere', np.max(np.abs(constant[:, 1])), 0.)
    tail = gaussian_tail()
    check('Gaussian_tail_upper_covers_analytic_mass', tail['mass_estimate']-tail['mass_upper'], 0.)

    f2 = RadialProfile('F2')
    near = np.array([8.-1e-9, 8., 8.+1e-9])
    n, v = f2(near)
    arrays['F2_kink_queries'], arrays['F2_kink_v'] = near, v
    check('F2_N_exact_one', np.max(np.abs(n-1)), 0.)
    declared_square = np.where(near <= 8., 8*(near/16.)**2, 16./near)
    check('F2_declared_squared_flow_across_kink', np.max(np.abs(v*v-declared_square)), 5e-12)
    base, _ = back_average(f2, queries)
    tight, _ = back_average(f2, queries, tight=True)
    arrays['F2_average_base'], arrays['F2_average_tight'] = base, tight
    check('F2_average_fixed_quadrature_refinement', np.max(np.abs(base-tight)), 1e-10)
    check('F2_averaged_vector_regular_centre', abs(base[0, 1]), 5e-12)

    # Exercise the actual reused regular solver on an independent synthetic
    # source. These are not original-collapse fields or record consequences.
    synthetic = GaussianSources(np.array([1., 2., 4.]), np.array([.1, .2, .1]), np.zeros(3), 4.)
    p = SolvedField(synthetic, .25, 10., 10., 4., variant='P').integrate()
    h = SolvedField(synthetic, .25, 10., 10., 4., variant='H').integrate()
    if p.status != 'complete' or h.status != 'complete':
        raise ValueError('Manufactured regular field unavailable')
    rr = np.linspace(0., 10., 101)
    pf, hf = p.fields(rr), h.fields(rr)
    arrays['synthetic_r'] = rr
    arrays['synthetic_P_Nv'] = np.array([pf['N'], pf['v']]).T
    arrays['synthetic_H_Nv'] = np.array([hf['N'], hf['v']]).T
    check('own_source_P_H_manufactured_refinement',
          np.max(np.abs(arrays['synthetic_P_Nv']-arrays['synthetic_H_Nv'])), 1e-10)
    check('J_zero_lapse_exact_constant', np.max(np.abs(pf['N']-1)), 5e-12)
    check('regular_centre_zero_flow', abs(pf['v'][0]), 0.)
    bounds = field_majorants(p)
    check('actual_dense_majorant_N_covers_samples', np.max(pf['N'])-bounds['N_upper'], 0.)
    check('actual_dense_majorant_v_covers_samples', np.max(np.abs(pf['v']))-bounds['v_abs_upper'], 0.)

    # A deliberately nonmonotone nested polynomial distinguishes coefficient
    # majorants from maxima on endpoints. Its basis is evaluated independently.
    coeff = np.array([[.03, -.4], [-.02, .6], [.07, -.8],
                      [-.05, 1.2], [.01, -.5], [.04, .7], [-.03, -.2]])
    fake_segment = SimpleNamespace(t_old=1., t=2., y_old=np.array([.2, -.3]), F=coeff)
    fake = SimpleNamespace(status='complete', vacuum=False, r0=.01,
        coefficients={'A': .01, 'B': 0., 'C': -.01, 'D': 0.},
        y_collar=-.2, collar=10., interpolants=[fake_segment],
        state=lambda radius: np.array([.5, -.2]))
    upper = field_majorants(fake)
    xi = np.linspace(0., 1., 1001)
    result = np.zeros((len(xi), 2))
    for i, coefficient in enumerate(coeff[::-1]):
        result += coefficient
        result *= (xi if i % 2 == 0 else 1-xi)[:, None]
    result += fake_segment.y_old
    fn = np.exp(result[:, 1]+.2)
    fv = fn*np.sqrt(result[:, 0]/(1+xi))
    arrays['nested_polynomial_samples'] = result
    check('nested_coefficient_N_bound', np.max(fn)-upper['N_upper'], 0.)
    check('nested_coefficient_v_bound', np.max(fv)-upper['v_abs_upper'], 0.)
    refused = False
    try:
        f2(np.array([1.+1j]))
    except ValueError:
        refused = True
    check('complex_profile_query_refused', 0. if refused else 1., 0.)
    return {'checks': checks, 'all_passed': all(x['passed'] for x in checks),
            'gaussian_tail': tail, 'synthetic_majorants': bounds,
            'synthetic_source_evaluations': synthetic.evaluations}, arrays
