#!/usr/bin/env python3
"""Verify completed source file/column bytes, without numerical propagation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np


BASE = Path('<internal>/k3_source')
REPORT = BASE / 'recovery_v1/source_construction_checks.json'
EXPECTED_REPORT = '19be36cdf080b34aac7cfcf0f454a3c5caeb9374a6a3aada182579e628d87a52'
OUTPUT = BASE / 'hreduce_checks_v1/source_content_checks.json'


def stat_key(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def main():
    if OUTPUT.exists() or OUTPUT.with_suffix('.partial.json').exists():
        raise FileExistsError('Existing verification must be read, never repeated implicitly')
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Insufficient root-disk reserve')
    before = time.monotonic()
    consumed = {}

    def identity(path):
        path = Path(path).resolve(strict=True)
        if str(path) in consumed:
            return consumed[str(path)]
        first = stat_key(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 2**20), b''):
                digest.update(block)
        if first != stat_key(path):
            raise RuntimeError('A bound file changed while hashing')
        value = dict(path=str(path), size=first[2], sha256=digest.hexdigest(), stat=list(first))
        consumed[str(path)] = value
        return value

    def verify(expected):
        actual = identity(expected['path'])
        if any(actual[key] != expected[key] for key in ('path', 'size', 'sha256')):
            raise ValueError('Bound content differs: ' + expected['path'])
        return actual

    report_id = identity(REPORT)
    if report_id['sha256'] != EXPECTED_REPORT:
        raise ValueError('The completed construction snapshot differs')
    report = json.loads(REPORT.read_text())
    if report['completed_blocks'] != list(range(32)) or not report['completed']:
        raise ValueError('The expected complete block inventory is unavailable')
    for item in report['artifacts'].values():
        verify(item)
    blocks = {}
    for index in range(32):
        entry = verify(report['block_result_identities'][str(index)])
        block = json.loads(Path(entry['path']).read_text())
        if (block['index'], block['columns'], block['completed']) != (
                index, [256 * index, 256 * (index + 1)], True):
            raise ValueError('A block identity/range differs')
        blocks[index] = block
    checked_columns = []
    for endpoint in ('120', '240'):
        source = verify(report['raw_sources'][endpoint])
        array = np.load(source['path'], mmap_mode='r', allow_pickle=False)
        if (array.shape != (16384, 8192) or array.dtype != np.complex128
                or not array.dtype.isnative or not array.flags.c_contiguous or array.flags.writeable):
            raise ValueError('A completed source has another stored layout')
        for index in range(32):
            expected = blocks[index]['endpoints'][endpoint]['source_columns']
            start, stop = expected['columns']
            if (start, stop) != (256 * index, 256 * (index + 1)):
                raise ValueError('Logical source-column range differs')
            digest = hashlib.sha256(array[:, start:stop].tobytes(order='C')).hexdigest()
            if digest != expected['sha256']:
                raise ValueError('A logical endpoint block differs from its saved producer bytes')
            checked_columns.append(dict(endpoint=endpoint, index=index, columns=[start, stop],
                                        sha256=digest, matched=True))
        del array
    for value in consumed.values():
        if list(stat_key(Path(value['path']))) != value['stat']:
            raise RuntimeError('A consumed file changed during verification')
    result = dict(completed=True, content_verified=True, source_report=report_id,
        block_reports_verified=32, logical_endpoint_blocks_verified=64,
        complete_source_files_verified=2, unique_input_files_verified=len(consumed),
        files=list(consumed.values()), logical_blocks=checked_columns,
        elapsed_seconds=time.monotonic() - before,
        scope='content/layout only; no numerical source-bound, physical-state or held-row acceptance')
    OUTPUT.parent.mkdir(exist_ok=True)
    temporary = OUTPUT.with_suffix('.partial.json')
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, OUTPUT)
    temporary.unlink()
    print(json.dumps({key: result[key] for key in ('completed', 'content_verified',
        'block_reports_verified', 'logical_endpoint_blocks_verified', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
