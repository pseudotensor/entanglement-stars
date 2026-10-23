"""The original reviewer B pair's 15 arbitrary-field H4 assertions, remote CPU only."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import sympy as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (180, 190))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    started = time.perf_counter()
    t, r, grav = sp.symbols('t r G', nonzero=True)
    coords = [t, r]
    lapse, flow, sr, st = [sp.Function(name)(t, r) for name in ['N', 'v', 's_r', 's_t']]
    scalar = 4 * sp.pi * grav
    a = sp.diff(flow, r) / lapse
    b = flow / (r * lapse)
    rho = (2 * a * b + b**2) / (2 * scalar)
    current = -b * sp.diff(lapse, r) / (scalar * lapse)

    def along(expr):
        return sp.diff(expr, t) + flow * sp.diff(expr, r)

    rr = (along(a) - sp.diff(lapse, r, 2)) / lapse + a * (a + 2 * b) - scalar * (rho + sr - 2 * st)
    rt = (along(b) - sp.diff(lapse, r) / r) / lapse + b * (a + 2 * b) - scalar * (rho - sr)
    w0 = sp.diff(rho, t) + sp.diff(r**2 * (flow * rho + lapse * current), r) / r**2
    w0 += current * sp.diff(lapse, r) + sr * sp.diff(flow, r) + 2 * st * flow / r
    wr = sp.diff(current, t) + sp.diff(r**2 * (flow * current + lapse * sr), r) / r**2
    wr += -2 * lapse * st / r + rho * sp.diff(lapse, r) + current * sp.diff(flow, r)

    q = sp.Matrix([[-lapse**2 + flow**2, -flow], [-flow, 1]])
    qi = sp.Matrix([[-1 / lapse**2, -flow / lapse**2],
                    [-flow / lapse**2, 1 - flow**2 / lapse**2]])
    gamma = [[[sp.cancel(sum(qi[i, k] * (
        sp.diff(q[k, j], coords[l]) + sp.diff(q[k, l], coords[j])
        - sp.diff(q[j, l], coords[k])) for k in range(2)) / 2)
        for l in range(2)] for j in range(2)] for i in range(2)]
    ricci = sp.zeros(2)
    for i in range(2):
        for j in range(2):
            ricci[i, j] = sum(
                sp.diff(gamma[k][i][j], coords[k]) - sp.diff(gamma[k][i][k], coords[j])
                + sum(gamma[k][i][j] * gamma[l][k][l]
                      - gamma[l][i][k] * gamma[k][j][l] for l in range(2))
                for k in range(2))
    curvature2 = sp.cancel(sum(qi[i, j] * ricci[i, j] for i in range(2) for j in range(2)))
    box_radius = (sp.diff(-flow / lapse, t) + sp.diff(lapse - flow**2 / lapse, r)) / lapse
    einstein = sp.Matrix(2, 2, lambda i, j: 2 * gamma[1][i][j] / r
                         + (2 * box_radius / r - flow**2 / (r**2 * lapse**2)) * q[i, j])
    angular_einstein = box_radius / r - curvature2 / 2
    normal = sp.Matrix([1 / lapse, flow / lapse])
    radial = sp.Matrix([0, 1])
    stress = rho * normal * normal.T + current * (normal * radial.T + radial * normal.T)
    stress += sr * radial * radial.T
    divergence = [
        sum(sp.diff(lapse * r**2 * stress[j, i], coords[j]) for j in range(2)) / (lapse * r**2)
        + sum(gamma[i][j][k] * stress[j, k] for j in range(2) for k in range(2))
        - 2 * st * qi[i, 1] / r for i in range(2)
    ]
    radial_einstein_residual = einstein[1, 1] - 2 * scalar * sr
    checks = [
        ('01_normal_constraint', (normal.T * einstein * normal)[0] - 2 * scalar * rho),
        ('02_momentum_constraint', (normal.T * einstein * radial)[0] + 2 * scalar * current),
        ('03_radial_einstein_vs_tangent_ADM', radial_einstein_residual + 2 * rt),
        ('04_angular_einstein_vs_ADM_trace', angular_einstein - 2 * scalar * st + rr + rt),
        ('05_coordinate_energy_divergence', divergence[0] - w0 / lapse**2),
        ('06_coordinate_momentum_divergence', divergence[1] - flow * divergence[0] - wr / lapse),
        ('07_first_source_defect_identity', scalar * w0 / lapse - b * rr - (a + b) * rt),
        ('08_second_source_defect_identity', scalar * wr - sp.diff(lapse * rt, r) - lapse * (rt - rr) / r),
        ('09_quadrature_differential_identity', sp.diff(lapse * r**2 * flow * rt, r)
         - scalar * r**2 * (lapse * w0 + flow * wr)),
        ('10_covariant_quadrature', sp.diff(lapse * r**2 * flow * radial_einstein_residual, r)
         + 2 * scalar * lapse * r**2 * (flow * divergence[1] + (lapse**2 - flow**2) * divergence[0])),
        ('11_zero_flow_radial_exception', (rr + sp.diff(lapse, r, 2) / lapse).subs({flow: 0, sr: 0, st: 0}).doit()),
        ('12_zero_flow_tangent_exception', (rt + sp.diff(lapse, r) / (r * lapse)).subs({flow: 0, sr: 0, st: 0}).doit()),
        ('13_zero_flow_energy_defect', w0.subs({flow: 0, sr: 0, st: 0}).doit()),
        ('14_zero_flow_momentum_defect', wr.subs({flow: 0, sr: 0, st: 0}).doit()),
        ('15_mass_balance_normalization', sp.diff(r**3 * b**2 / (2 * grav), t)
         + scalar * r**2 * (flow * (rho + sr) + lapse * (1 + (flow / lapse)**2) * current) / grav
         - r**2 * flow * rt / grav),
    ]
    result = {
        'status': 'running', 'serves': 'WEB40E H4 / duty8 / insert5 DM6',
        'conventions': {'metric': '-N^2 dt^2+(dr-v dt)^2+r^2 dOmega^2',
                        'residuals': 'ADM equation residuals divided by N',
                        'fields': 'arbitrary smooth N(t,r),v(t,r),s_r(t,r),s_t(t,r)',
                        'constraints': 'rho and J substituted from both coherent constraints'},
        'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sympy_version': sp.__version__, 'assertions': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, expr in checks:
        derivatives = sorted(expr.atoms(sp.Derivative), key=str)
        replacements = {term: sp.Symbol(f'jet_{i}') for i, term in enumerate(derivatives)}
        replacements.update({field: sp.Symbol(f'field_{i}') for i, field in enumerate([lapse, flow, sr, st])})
        polynomial = sp.cancel(expr.xreplace(replacements))
        numerator, _ = sp.fraction(polynomial)
        residual = sp.expand(numerator)
        result['assertions'].append({'name': name, 'zero': residual == 0, 'residual_numerator': str(residual)})
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    result.update({
        'status': 'complete', 'passed': all(row['zero'] for row in result['assertions']),
        'elapsed_seconds': time.perf_counter() - started,
        'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        'maximum_resident_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope': 'Exact arbitrary-field identities only; no native history, stress or time/ell error certification.',
    })
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
