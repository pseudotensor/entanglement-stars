"""Input metadata only for packet route; NPZ object payloads are never unpickled."""
import argparse
import json
import hashlib
from pathlib import Path
import zipfile
import numpy as np
from cpu_runtime import require_remote_cpu

BASE = '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output.parent)
    raw = json.loads(Path(BASE+'.json').read_text())
    result = {'args': raw['args'], 'record_97': raw['records'][97], 'source_bindings': {}}
    for label, suffix in [('fields', '_fields.npz'), ('spectra', '_spectra.npz')]:
        result[label] = {}
        with zipfile.ZipFile(BASE+suffix) as archive:
            for name in archive.namelist():
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _, dtype = np.lib.format._read_array_header(stream, version)
                    result[label][name] = {'shape': list(shape), 'dtype': str(dtype)}
    for suffix in ('.json', '_fields.npz', '_spectra.npz'):
        path = Path(BASE+suffix)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        result['source_bindings'][str(path)] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    args.output.with_name('inventory_runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'runtime': runtime}))

if __name__ == '__main__':
    main()
