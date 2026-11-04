#!/usr/bin/env python3
"""Required H1 positive-mass native controls, specified by the original pair.

Full four-component projector, half of the two-face slab entropy, reduced
unimodular transverse basis and pi-zone/|p| normalization. The algorithm is
adapted from the actual busi eta_common/eta_direction routines read in place;
it does not run their massless shortcut or import their error extrapolation.
Only scalar summaries are results; resumable per-point checkpoints stay on
the box and are excluded from the repository. Run in qi2, nice10, BLAS1.
"""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import warnings

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import mpmath as mp
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.linalg import eigvalsh, toeplitz
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from massive_axial_control import axial_chain

SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)
ALPHA = np.zeros((3, 4, 4), dtype=complex)
ALPHA[:, :2, :2] = SIGMA
ALPHA[:, 2:, 2:] = -SIGMA
BETA = np.zeros((4, 4), dtype=complex)
BETA[:2, 2:] = np.eye(2)
BETA[2:, :2] = np.eye(2)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def det(a, b, c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
            + a[2]*(b[0]*c[1]-b[1]*c[0]))


def xgcd(a, b):
    rr, r, ss, s, tt, t = abs(a), abs(b), 1, 0, 0, 1
    while r:
        q = rr//r
        rr, r, ss, s, tt, t = r, rr-q*r, s, ss-q*s, t, tt-q*t
    return rr, ss*(1 if a >= 0 else -1), tt*(1 if b >= 0 else -1)


def plane_basis(p):
    a, b, c = p
    g, x, y = xgcd(a, b)
    if g == 0:
        r, s = (1, 0, 0), (0, c, 0)
    else:
        one, aa, bb = xgcd(g, c)
        if one != 1:
            raise ValueError("normal must be primitive")
        r, s = (-y, x, 0), (-bb*(a//g), -bb*(b//g), aa)
    pp = dot(p, p)

    def gram(u, v):
        return pp*dot(u, v)-dot(u, p)*dot(v, p)

    for _ in range(10000):
        g11, g12, g22 = gram(r, r), gram(r, s), gram(s, s)
        if g11 > g22:
            r, s = s, tuple(-v for v in r)
            continue
        shift = (2*g12+g11)//(2*g11)
        if not shift:
            break
        s = tuple(v-shift*u for u, v in zip(r, s))
    else:
        raise ArithmeticError("transverse reduction did not terminate")
    g11, g12, g22 = gram(r, r), gram(r, s), gram(s, s)
    if det(p, r, s) != 1 or g11 > g22 or 2*abs(g12) > g11:
        raise ArithmeticError("invalid reduced unimodular basis")
    return r, s, np.array([[g11, g12], [g12, g22]], dtype=float)/pp


def next_pow2(n):
    return 1 << (int(n)-1).bit_length()


def slab(p, r, s, phi, psi, y, length, memory_cap, fft_multiplier=1):
    started = time.monotonic()
    n = 4*length
    nt = fft_multiplier*max(8192, next_pow2(4*length))
    estimate = 6*16*n*n + 8*16*nt*16
    if estimate > memory_cap*0.8:
        raise MemoryError("planned full-four slab exceeds declared worker memory allowance")
    th = 2*math.pi*np.arange(nt)/nt
    phase = np.asarray(r)*phi+np.asarray(s)*psi
    sine = np.sin(np.outer(th, p)+phase)
    energy = np.sqrt(y+np.sum(sine*sine, axis=1))
    projector = np.einsum('ki,iab->kab', sine, ALPHA)
    projector += math.sqrt(y)*BETA
    projector /= energy[:, None, None]
    projector *= -0.5
    for a in range(4):
        projector[:, a, a] += 0.5
    coefficients = np.fft.ifft(projector, axis=0)
    del projector, sine, energy, th
    covariance = np.empty((n, n), dtype=complex, order='F')
    rows = (-np.arange(length)) % nt
    for a in range(4):
        for b in range(4):
            covariance[a::4, b::4] = toeplitz(coefficients[:length, a, b],
                                               coefficients[rows, a, b])
    del coefficients
    hermitian_residual = float(np.max(np.abs(covariance-covariance.conj().T)))
    values = eigvalsh(covariance, overwrite_a=True, check_finite=False, driver='evr')
    del covariance
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("nonfinite covariance spectrum")
    minimum, maximum = float(values.min()), float(values.max())
    outside = int(np.count_nonzero((values < 0) | (values > 1)))
    clipped = np.clip(values, 0, 1)
    entropy = float(np.sum(-xlogy(clipped, clipped)-xlogy(1-clipped, 1-clipped)))/2
    return {"one_face_entropy": entropy, "L": length, "matrix_dimension": n,
            "ntheta": nt, "n_components": 4, "slab_faces": 2,
            "minimum_eigenvalue": minimum, "maximum_eigenvalue": maximum,
            "eigenvalues_outside_unit_interval": outside,
            "maximum_spectrum_excursion": max(0., -minimum, maximum-1),
            "endpoint_rule": "clip only outside [0,1]; no interior eigenvalue cutoff",
            "eigenvalues_within_1e_minus12_of_endpoints": int(np.count_nonzero((clipped < 1e-12) | (clipped > 1-1e-12))),
            "hermitian_residual": hermitian_residual,
            "estimated_peak_working_bytes": estimate,
            "elapsed_seconds": time.monotonic()-started}


def numerical_gap(p, r, s, phi, psi, y):
    count = max(4096, 256*max(map(abs, p)))
    theta = 2*math.pi*np.arange(count)/count
    phase = np.asarray(r)*phi+np.asarray(s)*psi
    e2 = y+np.sum(np.sin(np.outer(theta, p)+phase)**2, axis=1)
    index = int(np.argmin(e2))

    def value(t):
        return y+float(np.sum(np.sin(np.asarray(p)*t+phase)**2))

    answer = minimize_scalar(value, bounds=(theta[index]-2*math.pi/count,
                                           theta[index]+2*math.pi/count),
                             method='bounded', options={'xatol': 1e-12})
    return math.sqrt(max(y, min(float(answer.fun), float(e2[index]))))


def reference_integral(gram, y):
    g11, g12, g22 = gram[0, 0], gram[0, 1], gram[1, 1]

    def inner(phi):
        sine, cosine = math.sin(phi), math.cos(phi)
        a0 = g11*sine*sine+y
        disc = a0*a0+a0*g22-(g12*sine*cosine)**2
        return math.log((a0+0.5*g22+math.sqrt(max(0., disc)))/2)

    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', IntegrationWarning)
        for tolerance in (1e-12, 1e-13):
            vals = [quad(inner, a, b, epsabs=tolerance, epsrel=tolerance, limit=300)
                    for a, b in ((0, math.pi/2), (math.pi/2, math.pi))]
            rows.append({"value": math.fsum(v for v, e in vals)/math.pi,
                         "reported_error": math.fsum(e for v, e in vals)/math.pi,
                         "tolerance": tolerance})
    return rows, [str(w.message) for w in caught]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def run_fibers(result, args, memory, save):
    p, r, s = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    result['fibers'] = []
    ys = [Fraction(x) for x in args.fiber_y]
    for yq in ys:
        y = float(yq)
        for phi, psi, transverse_sq in ((0., 0., Fraction(0)),
                                         (math.pi/4, math.pi/3, Fraction(5, 4))):
            gap = math.sqrt(y+float(transverse_sq))
            length = 24+math.ceil(10/gap)
            with mp.workdps(60):
                exact_s = mp.mpf((yq+transverse_sq).numerator)/(yq+transverse_sq).denominator
                reference = mp.nstr(axial_chain(exact_s), 68)
            base = slab(p, r, s, phi, psi, y, length, memory)
            record = {"y": str(yq), "transverse": [phi, psi],
                      "exact_transverse_gap_squared": str(transverse_sq),
                      "exact_chain_reference": reference, "reference_digits": 60,
                      "base": base,
                      "base_minus_reference": base['one_face_entropy']-float(reference)}
            result['fibers'].append(record)
            save()
            if yq == min(ys) and transverse_sq == 0:
                long = slab(p, r, s, phi, psi, y, 24+math.ceil(15/gap), memory)
                fft = slab(p, r, s, phi, psi, y, length, memory, fft_multiplier=2)
                record.update(longer_slab=long, doubled_fft=fft,
                              longer_minus_base=long['one_face_entropy']-base['one_face_entropy'],
                              fft_minus_base=fft['one_face_entropy']-base['one_face_entropy'])
                save()


def run_level(result, args, memory, save):
    p, yq = tuple(args.p), Fraction(args.y)
    y, level = float(yq), args.level
    r, s, gram = plane_basis(p)
    pn = math.sqrt(dot(p, p))
    n1 = max(1, math.ceil(math.pi*math.sqrt(gram[0, 0])/args.h0)) << level
    n2 = max(1, math.ceil(math.pi*math.sqrt(gram[1, 1])/args.h0)) << level
    signature = {"p": list(p), "y": str(yq), "level": level, "h0": args.h0,
                 "N1": n1, "N2": n2, "cL": 10, "Lmin": 24,
                 "components": 4, "precision": "complex128", "ntheta_minimum": 8192}
    checkpoint = args.output.with_suffix('.checkpoint.json')
    points, checks = {}, {}
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        if old['signature'] != signature:
            raise ValueError("checkpoint has different case parameters")
        points, checks = old['points'], old['checks']
    result['case'] = signature | {"r": list(r), "s": list(s), "G": gram.tolist(),
                                 "pnorm": pn, "checkpoint_box_only": str(checkpoint)}
    plans = []
    for i in range(n1):
        for j in range(n2):
            partner = (n1-1-i, n2-1-j)
            if (i, j) > partner:
                continue
            phi, psi = math.pi*(i+0.5)/n1, math.pi*(j+0.5)/n2
            gap = numerical_gap(p, r, s, phi, psi, y)
            plans.append({"i": i, "j": j, "phi": phi, "psi": psi,
                          "weight": 1 if (i, j) == partner else 2,
                          "numerical_gap": gap, "L": 24+math.ceil(10*pn/gap)})
    if sum(q['weight'] for q in plans) != n1*n2:
        raise ArithmeticError("pi-zone inversion weights do not cover the grid")
    order = sorted(range(len(plans)), key=lambda i: (-plans[i]['L'], i))
    result['case'].update(n_points=len(plans), L_max=max(q['L'] for q in plans),
                          numerical_gap_min=min(q['numerical_gap'] for q in plans),
                          gap_scope="4096-or-more theta scan and local minimization; not a certified global gap",
                          numerical_error_bound=None)

    def save_checkpoint():
        write_json(checkpoint, {"kind": "box-only resumable checkpoint cache",
                                "signature": signature, "points": points, "checks": checks})
        result['progress'] = {"completed_points": len(points), "expected_points": len(plans),
                              "completed_sampled_checks": len(checks)}
        save()

    try:
        for idx in order:
            key = str(idx)
            if key in points:
                continue
            q = plans[idx]
            values = slab(p, r, s, q['phi'], q['psi'], y, q['L'], memory)
            points[key] = q | values
            if len(points) % 16 == 0:
                save_checkpoint()
        save_checkpoint()
        near = min(range(len(plans)), key=lambda i: plans[i]['numerical_gap'])
        ordinary = sorted(range(len(plans)), key=lambda i: plans[i]['numerical_gap'])[len(plans)//2]
        control_requests = [('near_longer', near, True, False),
                            ('ordinary_longer', ordinary, True, False),
                            ('near_doubled_fft', near, False, True)]
        for name, idx, longer, doubled in control_requests:
            if name in checks:
                continue
            q, base = plans[idx], points[str(idx)]
            length = 24+math.ceil(15*pn/q['numerical_gap']) if longer else q['L']
            row = slab(p, r, s, q['phi'], q['psi'], y, length, memory,
                       fft_multiplier=2 if doubled else 1)
            checks[name] = {"index": idx, "transverse": [q['phi'], q['psi']],
                            "base_one_face_entropy": base['one_face_entropy'],
                            "variation": row,
                            "difference": row['one_face_entropy']-base['one_face_entropy']}
            save_checkpoint()
        reference, ref_warnings = reference_integral(gram, y)
        adjusted, raw = [], []
        for idx, q in enumerate(plans):
            value = points[str(idx)]['one_face_entropy']
            sp, cp = math.sin(q['phi']), math.cos(q['phi'])
            sq, cq = math.sin(q['psi']), math.cos(q['psi'])
            model = gram[0, 0]*sp*sp+2*gram[0, 1]*sp*cp*sq*cq+gram[1, 1]*sq*sq+y
            if model <= 0:
                raise ArithmeticError("nonpositive native node model")
            raw.append(q['weight']*value)
            adjusted.append(q['weight']*(value+math.log(model)/3))
        case = result['case']
        case.update(eta=(math.fsum(adjusted)/(n1*n2)-reference[-1]['value']/3)/pn,
                    eta_raw_unsubtracted=math.fsum(raw)/(n1*n2*pn),
                    reference_integrals=reference, reference_warnings=ref_warnings,
                    reference_change_per_area=abs(reference[1]['value']-reference[0]['value'])/(3*pn),
                    sampled_controls=checks,
                    sampled_length_change_per_area=max(abs(checks[k]['difference']) for k in ('near_longer', 'ordinary_longer'))/pn,
                    sampled_fft_change_per_area=abs(checks['near_doubled_fft']['difference'])/pn,
                    maximum_spectrum_excursion=max(q['maximum_spectrum_excursion'] for q in points.values()),
                    maximum_hermitian_residual=max(q['hermitian_residual'] for q in points.values()),
                    maximum_point_seconds=max(q['elapsed_seconds'] for q in points.values()),
                    total_point_seconds=math.fsum(q['elapsed_seconds'] for q in points.values()),
                    numerical_indicator_scope="sampled changes and quadrature indicators, not uniform error bounds",
                    point_arrays_copied_to_result=False)
    finally:
        save_checkpoint()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['fibers', 'level'], required=True)
    ap.add_argument('--fiber-y', nargs='+', default=[])
    ap.add_argument('--p', type=int, nargs=3)
    ap.add_argument('--y')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--h0', type=float, default=0.1)
    ap.add_argument('--cpu', type=int)
    ap.add_argument('--wall-seconds', type=int, required=True)
    ap.add_argument('--memory-mib', type=int, default=4096)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.mode == 'fibers':
        if not args.fiber_y or any(Fraction(x) <= 0 for x in args.fiber_y):
            ap.error('fiber controls require positive squared masses')
    elif (args.p is None or args.y is None or Fraction(args.y) <= 0 or
          math.gcd(*args.p) != 1 or not any(args.p)):
        ap.error('native level requires a primitive normal and strictly positive squared mass')
    if args.level < 0 or args.h0 <= 0 or args.wall_seconds <= 0 or args.memory_mib < 256:
        ap.error('invalid numerical/resource controls')
    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))
    available = os.sched_getaffinity(0)
    cpu = min(available) if args.cpu is None else args.cpu
    if cpu not in available:
        ap.error('requested CPU is outside the allowed affinity')
    os.sched_setaffinity(0, {cpu})
    memory = args.memory_mib*1024**2
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (args.wall_seconds, args.wall_seconds+1))

    def timeout(signum, frame):
        raise TimeoutError('native massive control reached its declared time cap')

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGXCPU, timeout)
    signal.alarm(args.wall_seconds)
    start = time.monotonic()
    result = {"status": "incomplete", "purpose": "required H1 b_sphere native positive-mass control",
              "mode": args.mode, "units": "a=1; y=(ma)^2; nats; full4; one face per area a^2",
              "scope": "flat positive-mass native sea only; numerical indicators, no angular/mass-tail certificate",
              "pid": os.getpid(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
              "cpu_affinity": [cpu], "threads": 1,
              "wall_cap_seconds": args.wall_seconds, "memory_cap_bytes": memory,
              "numpy_version": np.__version__, "mpmath_version": mp.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(elapsed_seconds=time.monotonic()-start,
                      cpu_seconds=usage.ru_utime+usage.ru_stime, peak_rss_KiB=usage.ru_maxrss)
        write_json(args.output, result)

    save()
    try:
        if args.mode == 'fibers':
            run_fibers(result, args, memory, save)
        else:
            run_level(result, args, memory, save)
        signal.alarm(0)
        result['status'] = 'completed_numerical_control'
        save()
    except Exception as exc:
        result['failure'] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    print(json.dumps({k: result[k] for k in ('status', 'mode', 'elapsed_seconds', 'cpu_seconds', 'peak_rss_KiB')}, indent=2))


if __name__ == '__main__':
    main()
