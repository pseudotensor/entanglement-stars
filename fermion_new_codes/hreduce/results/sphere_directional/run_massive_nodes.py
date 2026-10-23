#!/usr/bin/env python3
"""Execute explicit H1 mass-control cases under one foreground CPU budget.

The original pair supplies the normals, positive squared masses and levels.
This runner starts the unchanged native producer sequentially, reuses completed
case results, and bounds cumulative child CPU time on one assigned CPU.
"""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys


def normal(text):
    p = tuple(int(x) for x in text.split(','))
    if len(p) != 3 or math.gcd(*p) != 1:
        raise argparse.ArgumentTypeError('need three primitive integer components')
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--normals', type=normal, nargs='+', required=True)
    ap.add_argument('--squared-masses', nargs='+', required=True)
    ap.add_argument('--levels', type=int, nargs='+', required=True)
    ap.add_argument('--cpu', type=int, required=True)
    ap.add_argument('--cpu-budget-seconds', type=int, required=True)
    ap.add_argument('--case-seconds', type=int, required=True)
    ap.add_argument('--memory-mib', type=int, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    masses = [Fraction(y) for y in args.squared_masses]
    if any(y <= 0 for y in masses) or min(args.levels) < 0:
        ap.error('require strictly positive masses and nonnegative levels')
    if min(args.cpu_budget_seconds, args.case_seconds, args.memory_mib) <= 0:
        ap.error('resource budgets must be positive')
    if args.cpu not in os.sched_getaffinity(0):
        ap.error('CPU unavailable')
    os.sched_setaffinity(0, {args.cpu})
    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    producer = Path(__file__).with_name('massive_directional_control.py')
    completed = skipped = 0
    print(json.dumps({'event': 'worker_started', 'pid': os.getpid(), 'cpu': args.cpu,
                      'cpu_budget_seconds': args.cpu_budget_seconds}), flush=True)
    for p in args.normals:
        for y in masses:
            for level in args.levels:
                stem = 'native_massive_p'+''.join(map(str, p))+'_y'+str(y).replace('/', '-')+'_level'+str(level)
                output = args.output_dir/(stem+'.json')
                if output.exists():
                    previous = json.loads(output.read_text())
                    c = previous.get('case', {})
                    if previous['status'] == 'completed_numerical_control':
                        if (c.get('p') != list(p) or Fraction(c['y']) != y or
                                c.get('level') != level or c.get('h0') != 0.1 or c.get('components') != 4):
                            raise ValueError('existing completed filename has different scientific inputs')
                        skipped += 1
                        continue
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                used = usage.ru_utime+usage.ru_stime
                remaining = int(args.cpu_budget_seconds-used)
                if remaining < 1:
                    print(json.dumps({'status': 'worker_cpu_budget_reached', 'cpu': args.cpu,
                                      'completed': completed, 'skipped': skipped, 'child_cpu_seconds': used}), flush=True)
                    return 2
                cmd = [sys.executable, str(producer), '--mode', 'level', '--p', *map(str, p),
                       '--y', str(y), '--level', str(level), '--h0', '0.1', '--cpu', str(args.cpu),
                       '--wall-seconds', str(min(args.case_seconds, remaining)),
                       '--memory-mib', str(args.memory_mib), '--output', str(output)]
                code = subprocess.run(cmd).returncode
                if code:
                    print(json.dumps({'status': 'native_case_incomplete', 'case': stem, 'exit_code': code}), flush=True)
                    return code
                completed += 1
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(json.dumps({'status': 'explicit_cases_complete', 'cpu': args.cpu, 'completed': completed,
                      'skipped': skipped, 'child_cpu_seconds': usage.ru_utime+usage.ru_stime}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
