"""Read small input metadata, without sources, field solves, or state contractions."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from cpu_runtime import require_remote_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output.parent)
    raw_inventory = args.inventory.read_bytes()
    inventory = json.loads(raw_inventory)
    run = json.loads(Path(inventory['run_identity']['path']).read_text())
    result = {'scope': 'small input metadata only; no source or field evaluation',
              'runtime': runtime,
              'inventory_sha256': hashlib.sha256(raw_inventory).hexdigest(),
              'run_head': run['head'], 'args': run['args'], 'slices': {}}
    result['record_clock_catalog'] = [
        {'index': i, 'clock_fields': {key: value for key, value in record.items()
                                    if key in ['T', 'T_over_rs', 'T_over_r_s', 'time']}}
        for i, record in enumerate(run['records'])]
    record = run['records'][80]
    result['control_index_80_metadata'] = {
        'keys': list(record),
        'selected_metadata': {
            key: value for key, value in record.items()
            if re.search(r'solver|gauge|collar|support|label|kernel|centre|converged|iteration|newton|residual|^solve_|^T$|^T_over_rs$', key,
                         flags=re.IGNORECASE)}}
    fields_path = inventory['archives']['stored_fields']['identity']['path']
    with np.load(fields_path, allow_pickle=False) as archive:
        result['stored_fields_scalar_metadata'] = {
            key: archive[key].tolist() for key in ['solver', 'label', 'g', 'ell', 'tau']}
    for label in ['T0005.00', 'T0011.50']:
        identity = inventory['archives'][label]['identity']
        path = Path(identity['path'])
        if path.stat().st_size != identity['bytes']:
            raise RuntimeError('Input size changed since frozen identity read')
        with np.load(path, allow_pickle=False) as archive:
            keys = ['T', 'rs', 'R', 'kappas', 'weights', 'dtype']
            keys += ['f_' + str(k) for k in range(1, 11)]
            result['slices'][label] = {
                'identity_from_inventory': identity,
                'values': {key: archive[key].tolist() for key in keys}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'runtime': runtime,
                      'run_head_keys': list(run['head'])}))


if __name__ == '__main__':
    main()
