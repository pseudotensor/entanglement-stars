#!/usr/bin/env python3
"""H1.12 scalar readout of the required positive-mass and angular controls.

Outward arithmetic here concerns printed numerical estimates only. Signed
grid changes, absolute amplification and sampled slab/FFT changes are
indicators, not certified native, mass-tail or angular error bounds.
"""
import argparse
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import enclose_support_table as geometry
from flint import arb, ctx
from certify_eta111 import ball_record


def value(x):
    return geometry.ball(Q(str(x)))


def weights(mesh):
    rays, triangles, _ = geometry.fundamental_mesh(mesh)
    answer = {}
    for ids in triangles:
        omega = geometry.solid_angle([rays[i] for i in ids])
        for i in ids:
            p = tuple(x//math.gcd(*rays[i]) for x in rays[i])
            answer[p] = answer.get(p, arb(0)) + (4/arb.pi())*omega
    if not sum(answer.values(), arb(0)).contains(1) or any(not w > 0 for w in answer.values()):
        raise ArithmeticError('angular weights fail positivity/normalization')
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))

    def timeout(signum, frame):
        raise TimeoutError('massive scalar readout time cap reached')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    ctx.prec, ctx.threads = 160, 1
    start = time.monotonic()
    axial = json.loads((args.input_dir/'massive_axial_control.json').read_text())
    if axial['status'] != 'completed_numerical_control':
        raise ValueError('exact axial numerical control is incomplete')
    chosen = max(axial['evaluations'], key=lambda r: (r['digits'], r['integration_partition']=='halves'))
    rows, native_sources = {}, []
    for path in sorted(args.input_dir.glob('native_massive_p*_y*_level[0-9].json')):
        d = json.loads(path.read_text())
        if d['status'] != 'completed_numerical_control':
            continue
        c = d['case']; key = (tuple(c['p']), Q(c['y']), c['level'])
        if key in rows or c['components'] != 4 or Q(c['y']) <= 0:
            raise ValueError('duplicate or incompatible native scalar row')
        rows[key] = c
        native_sources.append({'source': str(path), 'p': c['p'], 'y': c['y'],
                               'level': c['level'], 'eta': c['eta']})
    for row in chosen['eta100']:
        for level in (0, 1):
            rows[((1, 0, 0), Q(row['y']), level)] = {'eta': row['eta']}
    ww = {mesh: weights(mesh) for mesh in (2, 4)}
    normals = sorted(ww[4], key=lambda p: (sum(x*x for x in p), p))
    records, by_normal = [], {}
    for p in normals:
        record = {'p': list(p), 'angular_weights': {str(m): ball_record(ww[m][p]) if p in ww[m] else None for m in ww},
                  'filters': {}}
        for y in (Q(1, 25), Q(1, 50)):
            masses = (2*y, y, y/2, y/4)
            coefficients = [Q(a)/y for a in (1, -9, 24, -16)]
            if not all((p, z, l) in rows for z in masses for l in (0, 1)):
                continue
            estimates = []
            for level in (0, 1):
                estimate = sum((geometry.ball(a)*value(rows[p, z, level]['eta'])
                                for a, z in zip(coefficients, masses)), arb(0))
                estimate -= (geometry.ball(y).log()-2*arb(2).log())/(3*arb.pi())
                estimates.append(estimate)
            differences = [value(rows[p, z, 1]['eta'])-value(rows[p, z, 0]['eta']) for z in masses]
            absolute = sum((geometry.ball(abs(a))*abs(d) for a, d in zip(coefficients, differences)), arb(0))
            item = {'level0': ball_record(estimates[0]), 'level1': ball_record(estimates[1]),
                    'signed_grid_change': ball_record(estimates[1]-estimates[0]),
                    'absolute_grid_change_amplification': ball_record(absolute),
                    'weighted_absolute_grid_change': {str(m): ball_record(ww[m][p]*absolute) for m in ww if p in ww[m]},
                    'native_error_bound': None, 'mass_remainder_bound': None}
            selections = []
            for z in masses:
                available = sorted(level for normal, mass, level in rows if normal == p and mass == z)
                selections.append({'y': str(z), 'selected_level': available[-1],
                                   'predecessor_level': available[-2]})
            selected_estimates = []
            for which in ('predecessor_level', 'selected_level'):
                estimate = sum((geometry.ball(a)*value(rows[p, z, selected[which]]['eta'])
                                for a, z, selected in zip(coefficients, masses, selections)), arb(0))
                estimate -= (geometry.ball(y).log()-2*arb(2).log())/(3*arb.pi())
                selected_estimates.append(estimate)
            selected_absolute = sum((geometry.ball(abs(a))*abs(
                value(rows[p, z, selected['selected_level']]['eta'])-
                value(rows[p, z, selected['predecessor_level']]['eta']))
                for a, z, selected in zip(coefficients, masses, selections)), arb(0))
            item.update(selected_levels=selections,
                        selected_level_policy='finest completed level per mass, with its actual predecessor; not a uniform grid level',
                        selected_estimate=ball_record(selected_estimates[1]),
                        selected_predecessor_estimate=ball_record(selected_estimates[0]),
                        selected_predecessor_change=ball_record(selected_estimates[1]-selected_estimates[0]),
                        selected_absolute_change_amplification=ball_record(selected_absolute),
                        weighted_selected_absolute_change={str(m): ball_record(ww[m][p]*selected_absolute)
                                                           for m in ww if p in ww[m]})
            for field, label in (('sampled_length_change_per_area', 'sampled_length'),
                                 ('sampled_fft_change_per_area', 'sampled_fft')):
                item[label+'_amplification'] = (ball_record(sum(
                    (geometry.ball(abs(a))*value(rows[p, z, 1][field]) for a, z in zip(coefficients, masses)), arb(0)))
                    if all(field in rows[p, z, 1] for z in masses) else None)
                item['selected_'+label+'_amplification'] = (ball_record(sum(
                    (geometry.ball(abs(a))*value(rows[p, z, selected['selected_level']][field])
                     for a, z, selected in zip(coefficients, masses, selections)), arb(0)))
                    if all(field in rows[p, z, selected['selected_level']] for z, selected in zip(masses, selections)) else None)
            record['filters'][str(y)] = item
            by_normal[p, y] = (estimates, absolute, selected_estimates, selected_absolute)
        records.append(record)
    sphere = {}
    for mesh, angular in ww.items():
        sphere[str(mesh)] = {}
        for y in (Q(1, 25), Q(1, 50)):
            missing = [list(p) for p in angular if (p, y) not in by_normal]
            entry = {'missing_normals': missing, 'angular_or_mass_error_bound': None}
            if not missing:
                means = [sum((w*by_normal[p, y][0][level] for p, w in angular.items()), arb(0)) for level in (0, 1)]
                entry.update(level0=ball_record(means[0]), level1=ball_record(means[1]),
                             signed_grid_change=ball_record(means[1]-means[0]),
                             absolute_grid_change_amplification=ball_record(sum(
                                 (w*by_normal[p, y][1] for p, w in angular.items()), arb(0))),
                             selected_estimate=ball_record(sum(
                                 (w*by_normal[p, y][2][1] for p, w in angular.items()), arb(0))),
                             selected_predecessor_estimate=ball_record(sum(
                                 (w*by_normal[p, y][2][0] for p, w in angular.items()), arb(0))),
                             selected_absolute_change_amplification=ball_record(sum(
                                 (w*by_normal[p, y][3] for p, w in angular.items()), arb(0))),
                             selected_level_policy='per-mass selected/predecessor levels are explicit in directions; not a uniform level2 result')
            sphere[str(mesh)][str(y)] = entry
    signal.alarm(0)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {'status': 'numerical_mass_control_readout', 'purpose': 'required H1.12 E/Q and propagated-change readout',
              'scope': 'outward arithmetic on printed estimates only; no certified b_sphere or native/angular/mass-tail bound',
              'bits': 160, 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'wall_cap_seconds': 120, 'memory_cap_bytes': 512*1024**2,
              'elapsed_seconds': time.monotonic()-start, 'cpu_seconds': usage.ru_utime+usage.ru_stime,
              'peak_rss_KiB': usage.ru_maxrss, 'axial_source': str(args.input_dir/'massive_axial_control.json'),
              'native_sources': native_sources, 'directions': records, 'sphere_estimates': sphere,
              'certified_b_sphere_interval': None}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'cpu_seconds', 'peak_rss_KiB')}, indent=2))


if __name__ == '__main__':
    main()
