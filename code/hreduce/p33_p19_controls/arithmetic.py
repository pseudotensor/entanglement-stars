"""Elementary numerical labels specified by the reviewer B priority-3 author.

Execute only on compute-host, qi2 Python, nice 10, one CPU thread.
Decimal inputs are exact as supplied; measurement uncertainty is not modeled.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform

import mpmath as mp


def expressions(context):
    b = (2 - 1 / context.sqrt(2)) * context.sqrt(context.pi) / (2 * context.sqrt(6))
    z = (3 * context.mpf('0.068955') + 2 * context.mpf('0.077728')) / 5
    old_k = 8 * z
    old_d = 4 * (b / old_k - 1)
    old_a = context.sqrt(2 * context.pi * old_k)
    epsilon = context.mpf('1e-5')
    required_d = 12 * epsilon / (1 - epsilon) ** 2
    return {'B': b, 'Z': z, 'Kold': old_k, 'dold': old_d, 'aold_over_lP': old_a, 'epsilon': epsilon, 'dreq': required_d}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert os.environ[key] == '1'
    mp.mp.dps = 80
    mp.iv.dps = 50
    values = expressions(mp.mp)
    intervals = expressions(mp.iv)
    data = {
        'metadata': {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': platform.node(), 'python': platform.python_version(), 'mpmath': mp.__version__, 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'threads': 1, 'gpu_used': False, 'value_digits': 80, 'interval_digits': 50, 'input_scope': 'Decimal inputs treated as exact specified labels; no measurement uncertainty modeled.'},
        'values': {key: {'value_50_digits': mp.nstr(value, 50), 'interval_50_digits': str(intervals[key])} for key, value in values.items()},
        'requested_conservative_checks': {'B_between_0.467_and_0.469': bool(intervals['B'] > mp.iv.mpf('0.467') and intervals['B'] < mp.iv.mpf('0.469')), 'dold_below_minus_0.76': bool(intervals['dold'] < mp.iv.mpf('-0.76'))},
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'arithmetic.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# P19 numerical labels', '', 'Author-specified elementary arithmetic, evaluated remotely with qi2 Python, nice 10 and one CPU thread; no GPU.', '', '80-digit values and 50-digit interval arithmetic. Decimal inputs are exact as specified; these enclosures do not include measurement uncertainty and do not replace the analytic proof.', '', '| Quantity | Value (50 significant digits) | Arithmetic interval |', '|---|---|---|']
    for key, row in data['values'].items():
        lines.append(f"| {key} | {row['value_50_digits']} | {row['interval_50_digits']} |")
    lines.extend(['', 'B in (0.467, 0.469): ' + str(data['requested_conservative_checks']['B_between_0.467_and_0.469']), 'dold < -0.76: ' + str(data['requested_conservative_checks']['dold_below_minus_0.76']), ''])
    (output / 'ARITHMETIC.md').write_text('\n'.join(lines))
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
