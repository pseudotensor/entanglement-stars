"""Render bound setup/audit/benchmark scalar records and output headers; stdlib only."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import struct
import zipfile
from lead_runtime import admit, identity, require_identity, code_bindings, verify_threads
from lead_prepare import recheck_artifacts, save_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--run-sha256', required=True)
    p.add_argument('--outdir', type=Path, required=True)
    a = p.parse_args()
    runtime = admit(a.outdir, __file__, 'Saved setup/benchmark scalar/header evidence', 1, 0, mode='dense', blas_threads=1)
    if any(a.outdir.iterdir()):
        raise ValueError('Fresh evidence directory required')
    binding = require_identity(a.run, a.run_sha256)
    v = json.loads(a.run.read_text())
    if v['stage'] not in ('setup-benchmark', 'setup', 'audit', 'apply-benchmark') or v['bindings']['code'] != code_bindings():
        raise ValueError('Only exact-source setup/audit/benchmark records are admitted')
    recheck_artifacts(v)
    rows = []
    omitted = {'bindings', 'runtime', 'output_bindings', 'code_bindings', 'core_bindings'}
    def walk(value, key):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                return
            for name, child in value.items():
                if name not in omitted:
                    walk(child, key+'.'+str(name) if key else str(name))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, key+'.'+str(index))
        else:
            rows.append((key, value))
    walk(v, '')
    headers = []
    outputs = {}
    def collect(node, context='', in_outputs=False):
        if isinstance(node, dict):
            if 'path' in node and 'sha256' in node:
                if in_outputs:
                    key = str(Path(node['path']).resolve())
                    if key in outputs and outputs[key]['sha256'] != node['sha256']:
                        raise ValueError('Conflicting output identities')
                    outputs[key] = node
                return
            for key, child in node.items():
                if key not in ('bindings', 'code_bindings', 'core_bindings'):
                    collect(child, context+'.'+key, in_outputs or key == 'output_bindings')
        elif isinstance(node, list):
            for child in node:
                collect(child, context, in_outputs)
    collect(v)
    for arrays in outputs.values():
        path = Path(arrays['path'])
        if path.suffix not in ('.npz', '.npy'):
            continue
        require_identity(path, arrays['sha256'])
        def read_header(f, name):
            if f.read(6) != b'\x93NUMPY':
                raise ValueError('Expected numeric NPY archive entry')
            version = f.read(2)
            if version[0] not in (1, 2, 3):
                raise ValueError('Unexpected NPY header version')
            size = 2 if version[0] == 1 else 4
            count = struct.unpack('<H' if size == 2 else '<I', f.read(size))[0]
            header = ast.literal_eval(f.read(count).decode('utf8' if version[0] == 3 else 'latin1'))
            headers.append({'artifact': str(path), 'name': name, **header})
        if path.suffix == '.npy':
            with path.open('rb') as f:
                read_header(f, path.name)
        else:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    with z.open(name) as f:
                        read_header(f, name)
    original_runtime = v['runtime']
    resources = {k: val for k, val in original_runtime.items()
                 if k.startswith(('resident_threads_', 'memory_')) or k in
                 ('host', 'device', 'pid', 'thread_cap', 'library_thread_cap', 'numba_workers',
                  'actual_niceness', 'actual_executable', 'root_filesystem_available_bytes_at_admission')}
    output = {'self': identity(__file__), 'input': binding, 'runtime': runtime,
              'source_code': code_bindings(), 'original_resources': resources,
              'array_headers': headers, 'original_outputs': outputs,
              'scalar_rows': rows, 'saved_only': True, 'array_values_read': False,
              'new_numerical_evaluation': False}
    recheck_artifacts(v)
    verify_threads(runtime, 'before_save')
    save_record(a.outdir/'extraction.json', output)
    lines = ['# Saved '+v['stage']+' evidence', '',
             'Literal scalar records and array headers only; no array values or new numerical evaluation.', '',
             'Input record SHA256: '+binding['sha256']+'.',
             'Reader source SHA256: '+output['self']['sha256']+'.', '',
             '| Saved field | Literal value |', '|---|---|']
    for key, value in rows:
        lines.append('| '+key.replace('|', '\\|')+' | '+json.dumps(value).replace('|', '\\|')+' |')
    lines += ['', 'Array header inventory:', '', '```json', json.dumps(headers, indent=2, sort_keys=True), '```',
              '', 'Original resource observations:', '', '```json', json.dumps(resources, indent=2, sort_keys=True), '```',
              '', 'Original output identities:', '', '```json',
              json.dumps(outputs, indent=2, sort_keys=True), '```', '']
    with (a.outdir/'TABLE.md').open('x') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
