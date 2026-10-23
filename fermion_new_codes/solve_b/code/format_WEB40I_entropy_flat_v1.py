#!/usr/bin/env python3
"""Render saved X3 flat-check fields; never run a solver or load raw arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def cell(value):
    if isinstance(value, float):
        return format(value, '.17g')
    if value is None:
        return 'not supplied'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
                '| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data_path = args.directory / 'flat_gate.json'
    execution_path = args.directory / 'execution.json'
    data = json.loads(data_path.read_text())
    execution = json.loads(execution_path.read_text())
    lines = ['# X3 native flat check: saved-field readout', '',
             'This report formats the saved producer and execution records only. '
             'It performs no eigensolve, propagation, residual calculation or raw-array load.', '',
             'The producer labels its numerical error ingredients as observed and uncertified. '
             'The held-phase check is an arithmetic invariance check; its result does not '
             'certify a production propagator, history tail or printed decimal.', '',
             '## Execution', '']
    lines += table(['Field', 'Saved value'], [(key, execution.get(key)) for key in (
        'state', 'start_utc', 'end_utc', 'returncode', 'timed_out',
        'whole_outer_wall_seconds', 'child_user_cpu_seconds',
        'child_system_cpu_seconds', 'child_peak_rss_kib',
        'pre_python_limits', 'hard_whole_process_group_limit_seconds',
        'code_origin_commit', 'source_sha256')])
    lines += ['', '## Producer', '']
    lines += table(['Field', 'Saved value'], [(key, data.get(key)) for key in (
        'complete', 'reason', 'stage', 'scope', 'L', 'negative_count_required',
        'cut_sites', 'time_over_a', 'copy_weight_applied', 'host', 'python',
        'numpy', 'scipy', 'nice', 'cpu_affinity', 'threads',
        'startup_address_space_limits_observed', 'resources')])
    lines += ['', '## Identity fixture', '']
    lines += table(['Field', 'Saved value'], data.get('fixture', {}).items())
    lines += ['', '## Parent error ingredients', '']
    lines += table(['Field', 'Saved value'], data.get('parent_error', {}).items())
    lines += ['', '## Four prescribed cuts', '']
    lines += table(['Cut sites', 'Cut / r_s', 'State', 'Target', 'Raw entropy / nats',
                    'Raw minus target', 'Held entropy / nats', 'Held minus raw',
                    'Saved absolute 1e-10 check'], [(
                        cut.get('cut_sites'), cut.get('cut_over_rs'), cut.get('state'),
                        cut.get('target'), cut.get('raw_entropy', {}).get('entropy_nats'),
                        cut.get('target_difference'),
                        cut.get('held_phase_entropy', {}).get('entropy_nats'),
                        cut.get('held_entropy_difference'), cut.get('held_abs_1e-10_check'))
                    for cut in data.get('cuts', [])])
    for cut in data.get('cuts', []):
        lines += ['', '### Cut ' + cell(cut.get('cut_sites')) + ': saved details', '']
        lines += table(['Field', 'Saved value'], cut.items())
    lines += ['', '## Stored products', '',
              'Every raw-array path, size and hash supplied by the producer is recorded '
              'in LARGE_FILES.tsv; the raw arrays remain on the box.', '']
    lines += table(['Input', 'SHA256'], [(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                                        for path in (data_path, execution_path)])
    (args.directory / 'flat_gate.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
