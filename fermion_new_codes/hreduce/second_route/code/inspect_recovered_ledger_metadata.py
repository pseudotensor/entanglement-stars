#!/usr/bin/env python3
"""Inspect NPY headers/text scalars only; preserve the stopped ledger payload."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import zipfile


SOURCE = Path('<internal>/hreduce_readouts_v1'
    'c15/diagnostics/ledger_16_arrays.partial.npz')
OUTPUT = SOURCE.with_suffix('.metadata.json')
EXPECTED = '428ff35783f873743d53af9b0f8a5f23ed98afe870c6e9a3b3f09b8c941ababd'


def identity():
    stat = SOURCE.stat()
    return {'path': str(SOURCE), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
        'device': stat.st_dev, 'inode': stat.st_ino}


def main():
    before = identity()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise RuntimeError('Stopped payload hash differs; do not infer or repair it')
    result = {'purpose': 'Necessary held-ledger serialization repair: headers/text only',
        'source': before, 'source_sha256': digest, 'members': [], 'scalar_texts': {},
        'numerical_array_readout_or_reduction': False}
    with zipfile.ZipFile(SOURCE) as archive:
        for member in archive.infolist():
            with archive.open(member) as stream:
                magic = stream.read(8)
                if magic[:6] != b'\x93NUMPY' or magic[6] not in (1, 2, 3):
                    raise ValueError('Unsupported NPY header')
                version = magic[6]
                width = 2 if version == 1 else 4
                length = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
                header = ast.literal_eval(stream.read(length).decode('utf-8' if version == 3 else 'latin1'))
                shape, dtype = header['shape'], header['descr']
                result['members'].append({'member': member.filename, 'shape': list(shape),
                    'dtype': dtype, 'fortran_order': header['fortran_order'],
                    'uncompressed_bytes': member.file_size, 'compressed_bytes': member.compress_size,
                    'zip_crc32': format(member.CRC, '08x')})
                if shape == () and isinstance(dtype, str) and len(dtype) > 2 and dtype[1] in 'US':
                    payload = stream.read()
                    codec = ('utf-32-be' if dtype[0] == '>' else 'utf-32-le') if dtype[1] == 'U' else 'utf-8'
                    result['scalar_texts'][member.filename] = payload.decode(codec).rstrip('\x00')
    if identity() != before:
        raise RuntimeError('Stopped payload changed during metadata inspection')
    raw = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != raw:
            raise RuntimeError('Refusing to replace a different saved metadata inspection')
    else:
        with OUTPUT.open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'output': str(OUTPUT), 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(), 'members': len(result['members']),
        'text_scalar_lengths': {k: len(v) for k, v in result['scalar_texts'].items()}}))


if __name__ == '__main__':
    main()
