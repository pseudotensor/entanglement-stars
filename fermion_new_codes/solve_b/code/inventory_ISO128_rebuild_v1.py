"""Inventory input identities and NPY headers only; no source contraction/field solve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from cpu_runtime import require_remote_cpu

STEM = 'lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq'
ROOT = Path('<internal>/iso128_h2')


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def headers(path: Path) -> dict:
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
                result[name] = {'shape': list(shape), 'dtype': str(dtype),
                                'fortran_order': bool(fortran_order),
                                'uncompressed_bytes': archive.getinfo(name).file_size}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output.parent)
    run_path = ROOT / 'run_h2' / ('collapse_' + STEM + '.json')
    raw = json.loads(run_path.read_text())
    result = {'scope': 'file identities, run configuration, and archive headers only',
              'runtime': runtime, 'run_json_keys': list(raw), 'args': raw.get('args'),
              'run_identity': digest(run_path), 'archives': {}}
    for label, path in [
        ('T0005.00', ROOT / 'states_h2' / ('orbitals_' + STEM + '_T0005.00.npz')),
        ('T0011.50', ROOT / 'states_h2' / ('orbitals_' + STEM + '_T0011.50.npz')),
        ('stored_fields', ROOT / 'run_h2' / ('collapse_' + STEM + '_fields.npz')),
    ]:
        result['archives'][label] = {'identity': digest(path), 'headers': headers(path)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'runtime': runtime,
                      'run_json_keys': result['run_json_keys'], 'args': result['args']}))


if __name__ == '__main__':
    main()
