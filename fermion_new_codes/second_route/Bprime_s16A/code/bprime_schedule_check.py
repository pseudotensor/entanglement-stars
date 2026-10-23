"""Verify unchanged frozen inputs and bitwise short endpoints after scheduling-only repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bprime_runtime import admit, identity, require_identity, verify_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Saved-only scheduling-repair identity and short endpoint check', 1)
    if any(args.outdir.iterdir()):
        raise ValueError('Use a fresh empty scheduling-check output directory')
    import numpy as np
    bindings, records, inputs = [], [], []
    source = identity(__file__)
    for name in ('old', 'new'):
        path = getattr(args, name)
        bindings.append(require_identity(path, getattr(args, name+'_sha256')))
        record = json.loads(path.read_text())
        if record['mode'] != 'benchmark' or not record['all_passed']:
            raise ValueError('Both short runs must be completed benchmarks')
        ib = record['input_summary_binding']
        bindings.append(require_identity(ib['path'], ib['sha256']))
        inputs.append(json.loads(Path(ib['path']).read_text()))
        records.append(record)
    checks = {'same_frozen_physical_block': inputs[0]['canonical_benchmark_identity'] == inputs[1]['canonical_benchmark_identity'],
              'same_source_values': inputs[0]['benchmark_source_values'] == inputs[1]['benchmark_source_values'],
              'same_spectral_ranks': records[0]['benchmark_ranks'] == records[1]['benchmark_ranks'],
              'same_endpoint': records[0]['endpoint'] == records[1]['endpoint'],
              'same_short_interval': records[0]['benchmark']['macro_step_interval'] == records[1]['benchmark']['macro_step_interval'],
              'same_coarse_steps': records[0]['benchmark']['coarse_steps'] == records[1]['benchmark']['coarse_steps'],
              'same_fine_steps': records[0]['benchmark']['fine_steps'] == records[1]['benchmark']['fine_steps']}
    for name in ('benchmark_integrated_endpoint.npy', 'benchmark_plain_endpoint.npy'):
        arrays = []
        for record in records:
            binding = record['output_bindings'][name]
            bindings.append(require_identity(binding['path'], binding['sha256']))
            arrays.append(np.load(binding['path'], mmap_mode='r', allow_pickle=False))
        checks[name+'_bitwise_equal'] = arrays[0].shape == arrays[1].shape and arrays[0].dtype == arrays[1].dtype and np.array_equal(arrays[0].view(np.uint8), arrays[1].view(np.uint8))
    for binding in [source]+bindings:
        require_identity(binding['path'], binding['sha256'])
    verify_threads(runtime, 'after_saved_endpoint_comparison')
    result = {'runtime': runtime, 'source_binding': source, 'bindings': bindings, 'checks': checks,
              'all_passed': all(checks.values()),
              'scope': 'Unchanged benchmark inputs and bitwise short endpoint equality; no full endpoint or physical variation.'}
    (args.outdir/'schedule_check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'all_passed': result['all_passed'], 'checks': checks}, sort_keys=True), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
