"""Own radial source reconstruction and Gaussian back-average; box callers only.

No foreign lane is a code dependency. Entry points must admit the runtime
before importing this numerical module. This module never reads consequences.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_CEILING
import math
import numpy as np
from lead_field_core import GaussianSources, SolvedField
from lead_gaussian import gaussian_kernels

SOURCE_E = 106.67530815426768
SOURCE_L = 53.33765407713384
SCALE = 16.0 / SOURCE_L
SOURCE_ROW = 240
SOURCE_MD5 = '876dd5c6dee34b1c290b33eaeb4d25b1'
SOURCE_JSON_MD5 = 'e27e457d37615cff8d5216f05a494f89'
SOURCE_PATH = '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz'


def _positive_upper(value):
    """Convert a positive Decimal upper bound outward to binary64."""
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError('Finite nonnegative majorant required')
    return float(np.nextafter(out, math.inf)) if value else 0.0


def field_majorants(field, minimum_radius=0.0):
    """Bounds for the actual saved dense polynomial, not continuum ODE error.

    DOP853's nested basis alternates xi and 1-xi, each bounded by one on
    the accepted segment. No polynomial is fitted to point samples.
    """
    if field.status != 'complete' or minimum_radius < 0:
        raise ValueError('Complete field and nonnegative bound domain required')
    if field.vacuum:
        return {'N_upper': 1.0, 'v_abs_upper': 0.0, 'segments': [],
                'minimum_radius': float(minimum_radius), 'method': 'exact vacuum'}
    rows = []
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        d = lambda x: Decimal.from_float(float(x))
        gauge = d(field.y_collar)
        if minimum_radius < field.r0:
            c = field.coefficients
            r0 = d(field.r0)
            y_up = abs(d(c['C']))*r0**2 + abs(d(c['D']))*r0**4
            speed2 = abs(d(c['A']))*r0**2 + abs(d(c['B']))*r0**4
            n = (y_up-gauge).exp().next_plus()
            speed = speed2.sqrt().next_plus()
            rows.append({'kind': 'centre', 'N_upper': _positive_upper(n),
                         'v_abs_upper': _positive_upper(n*speed)})
        for index, segment in enumerate(field.interpolants):
            if segment.t < minimum_radius:
                continue
            if segment.F.shape != (7, 2) or segment.y_old.shape != (2,):
                raise ValueError('Unreviewed DOP853 dense basis layout')
            left = max(float(segment.t_old), float(minimum_radius))
            if left <= 0 or segment.t <= segment.t_old:
                raise ValueError('Invalid dense majorant segment')
            u_abs = abs(d(segment.y_old[0]))
            y_up = d(segment.y_old[1])
            for coefficient in segment.F:
                u_abs += abs(d(coefficient[0]))
                y_up += abs(d(coefficient[1]))
            n = (y_up-gauge).exp().next_plus()
            speed = (u_abs/d(left)).sqrt().next_plus()
            rows.append({'kind': 'dense', 'index': index, 'left': left,
                         'right': float(segment.t), 'N_upper': _positive_upper(n),
                         'v_abs_upper': _positive_upper(n*speed)})
        u_collar = d(field.state(field.collar)[0])
        if u_collar <= 0:
            raise ValueError('Nonpositive exterior charge')
        exterior_left = d(max(field.collar, float(minimum_radius)))
        rows.append({'kind': 'exterior', 'N_upper': 1.0,
                     'v_abs_upper': _positive_upper((u_collar/exterior_left).sqrt().next_plus())})
    return {'N_upper': max(x['N_upper'] for x in rows),
            'v_abs_upper': max(x['v_abs_upper'] for x in rows),
            'minimum_radius': float(minimum_radius), 'segments': rows,
            'method': 'Decimal80 outward nested-basis absolute coefficient majorants; actual numerical field only'}


def gaussian_tail():
    """Analytic 3D Gaussian mass estimate and a conservative upper bound.

    erfc(q/sqrt(2)) <= sqrt(2/pi)*exp(-q*q/2)/q, q=8.
    The explicit decimal pi constant is a lower bound, increasing the result.
    """
    estimate = math.erfc(8/math.sqrt(2)) + math.sqrt(2/math.pi)*8*math.exp(-32)
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        pi_lower = Decimal('3.14159265358979323846264338327950288419716939937510')
        upper = (Decimal(2)/pi_lower).sqrt().next_plus()
        upper *= Decimal(8)+Decimal(1)/Decimal(8)
        upper *= Decimal(-32).exp().next_plus()
        value = _positive_upper(upper)
    return {'radius_in_sigma': 8.0, 'mass_estimate': estimate,
            'mass_upper': value, 'decimal_upper': str(upper),
            'meaning': 'omitted radial domain is contained in |Z|>8sigma'}


def reconstruct_source(path=SOURCE_PATH, variant='P'):
    """Load only numeric row240 source inputs, never pickled node arrays."""
    if variant not in ('P', 'H'):
        raise ValueError('Only frozen P/H field variants are allowed')
    with np.load(path, allow_pickle=False) as archive:
        radius = archive['R']
        clocks = archive['T']
        if (radius.dtype != np.float64 or not np.array_equal(radius, np.arange(1., 2049.))
                or clocks.shape != (321,) or float(clocks[SOURCE_ROW]) != 1920.):
            raise ValueError('Wrong F1 original source grid/clock')
        if float(archive['g']) != .25 or float(archive['ell']) != 4.:
            raise ValueError('Wrong F1 source coupling/kernel')
        energy = archive['eps_tot'][SOURCE_ROW].copy()
        current = archive['J_tot'][SOURCE_ROW].copy()
    source = GaussianSources(radius, energy, current, 4.)
    field = SolvedField(source, .25, 2048., 2048., 4., variant=variant).integrate()
    if field.status != 'complete':
        raise ValueError('Own F1 field unavailable: '+str(field.failure))
    charge_u = float(field.state(2048.)[0])
    charge_e = charge_u / .5
    profile = RadialProfile('F1', field)
    report = {'variant': variant, 'row': SOURCE_ROW, 'T': 1920., 'g': .25,
              'source_length': SOURCE_L, 'target_scale': SCALE,
              'own_U_collar': charge_u, 'own_E_collar': charge_e,
              'source_dense_E_collar': SOURCE_E,
              'charge_relative_difference': (charge_e-SOURCE_E)/SOURCE_E,
              'charge_gate_passed': abs(charge_e-SOURCE_E) <= 1e-8*SOURCE_E,
              'global_field_majorants': field_majorants(field),
              'steps': field.steps, 'trials': field.trials,
              'source_evaluations': source.evaluations,
              'centre': source.centre, 'centre_omissions': source.centre_omissions,
              'coefficients': field.coefficients, 'atol': field.atol,
              'nfev': field.nfev, 'status': field.status}
    return profile, report


class RadialProfile:
    def __init__(self, family, field=None):
        if family not in ('F1', 'F2') or (family == 'F1' and field is None):
            raise ValueError('Declared F1/F2 profile required')
        self.family, self.field = family, field

    def __call__(self, radius):
        if np.iscomplexobj(radius):
            raise ValueError('Real profile query required')
        r = np.asarray(radius, dtype=float)
        if np.any(r < 0) or np.any(~np.isfinite(r)):
            raise ValueError('Finite nonnegative profile query required')
        if self.family == 'F1':
            return self.field.applied_exterior(r/SCALE)
        n = np.ones_like(r)
        v = np.empty_like(r)
        inside = r <= 8.
        v[inside] = -4*r[inside]/(8.**1.5)
        v[~inside] = -np.sqrt(16./r[~inside])
        return n, v

    def majorants(self, minimum_radius=0.):
        if not math.isfinite(minimum_radius) or minimum_radius < 0:
            raise ValueError('Finite nonnegative majorant domain required')
        if self.family == 'F1':
            return field_majorants(self.field, minimum_radius/SCALE)
        with localcontext() as ctx:
            ctx.prec = 80
            ctx.rounding = ROUND_CEILING
            squared_upper = Decimal(2) if minimum_radius <= 8. else Decimal(16)/Decimal.from_float(float(minimum_radius))
            speed_upper = _positive_upper(squared_upper.sqrt().next_plus())
        return {'N_upper': 1., 'v_abs_upper': speed_upper,
                'minimum_radius': float(minimum_radius), 'method': 'exact dust-ball maximum'}


def back_average(profile, radius, tight=False, sigma=2., chunk=64):
    """Frozen 16/24-point panel rule using own stable angular Gaussian kernels."""
    r = np.asarray(radius, dtype=float)
    if np.iscomplexobj(radius) or np.any(r < 0) or not np.all(np.isfinite(r)) or sigma != 2.:
        raise ValueError('Frozen nonnegative radial queries and sigma=2 required')
    nodes, weights = np.polynomial.legendre.leggauss(24 if tight else 16)
    width = sigma/(4 if tight else 2)
    output = np.empty((r.size, 2))
    flat = r.ravel()
    for start in range(0, len(flat), chunk):
        # Each radius retains its exact declared bounds and kink split.
        for j, query in enumerate(flat[start:start+chunk], start):
            low, high = max(0., query-8*sigma), query+8*sigma
            cuts = [low]
            if profile.family == 'F2' and low < 8. < high:
                cuts.append(8.)
            cuts.append(high)
            ss, ww = [], []
            for left, right in zip(cuts[:-1], cuts[1:]):
                count = int(math.ceil((right-left)/width))
                edges = np.linspace(left, right, count+1)
                centre = (edges[:-1]+edges[1:])/2
                half = (edges[1:]-edges[:-1])/2
                ss.append((centre[:, None]+half[:, None]*nodes).ravel())
                ww.append((half[:, None]*weights).ravel())
            s, w = np.concatenate(ss), np.concatenate(ww)
            n, v = profile(s)
            k0, k1 = gaussian_kernels(query, s, sigma)
            measure = 4*np.pi*s*s*w
            output[j] = [np.dot(measure*k0, n), np.dot(measure*k1, v)]
    if not np.all(np.isfinite(output)) or np.any(output[:, 0] <= 0):
        raise ValueError('Nonfinite/nonpositive averaged fields')
    majorant = profile.majorants()
    tail = gaussian_tail()
    tail['N_absolute_upper'] = float(np.nextafter(tail['mass_upper']*majorant['N_upper'], math.inf))
    tail['v_absolute_upper'] = float(np.nextafter(tail['mass_upper']*majorant['v_abs_upper'], math.inf))
    tail['scope'] = ('Omitted 8sigma integration domain of the frozen numerical field as a real function only; '
                     'excludes continuum ODE error, quadrature, kernel approximation and ordinary evaluator roundoff')
    return output.reshape(r.shape+(2,)), {'tight': bool(tight), 'sigma': sigma,
              'order': len(nodes), 'maximum_panel_width': width,
              'gaussian_tail': tail, 'global_majorants': majorant}


def dense_artifact(field):
    """All coefficients necessary to reload the actual chosen numerical field."""
    return {'dense_left': np.array([x.t_old for x in field.interpolants]),
            'dense_right': np.array([x.t for x in field.interpolants]),
            'dense_y_old': np.array([x.y_old for x in field.interpolants]),
            'dense_F': np.array([x.F for x in field.interpolants]),
            'y_collar': np.asarray(field.y_collar),
            'coefficients': np.array([field.coefficients[k] for k in ('A','B','C','D')]),
            'r0': np.asarray(field.r0), 'collar': np.asarray(field.collar)}


def averaged_exterior_majorants(profile, minimum_radius):
    """Bound averaged N and vector magnitude outside the declared radius.

    Retained Gaussian nodes have source radius >= minimum_radius-8sigma.
    The omitted domain uses the global numerical-field bound and tail mass.
    Component magnitudes are bounded by the vector-magnitude majorant.
    """
    if not math.isfinite(minimum_radius) or minimum_radius < 0:
        raise ValueError('Nonnegative exterior-bound radius required')
    near = profile.majorants(max(0., minimum_radius-16.))
    global_ = profile.majorants()
    tail = gaussian_tail()
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        d = lambda x: Decimal.from_float(float(x))
        n = d(near['N_upper'])+d(tail['mass_upper'])*d(global_['N_upper'])
        v = d(near['v_abs_upper'])+d(tail['mass_upper'])*d(global_['v_abs_upper'])
        n_upper, v_upper = _positive_upper(n), _positive_upper(v)
    return {'minimum_radius': float(minimum_radius), 'N_upper': n_upper,
            'v_abs_upper': v_upper, 'retained_source_majorants': near,
            'global_source_majorants': global_, 'tail_mass': tail,
            'scope': 'actual numerical-field Gaussian average; excludes continuum ODE and evaluator errors'}
