"""Immutable persistence and scalar helpers for HOLD specification section 19."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


BASE = Path('<internal>/second_route_bulk')
METHOD = 'held-saved-spectrum-localization-split-1'
CASES = ('c15_k1', 'c15_k2', 'c30_k1', 'c30_k2')
CONTROL_NAMES = ('initial_gate.json', 'spectral_checks.json', 'precision_checks.json')
PINS = dict(zip(CASES, (
    ('ee9ccdeea8f520443a0f91dc8b32f30fb586310d67fec812312a6490a1a49dfa',
     'd2ed67b7fe76c8ed778cc037533f183c785241a1ded75091b02e46a926e7c961',
     'd17e45937ceecc3e38d6193f53eb12296909717591c6b77267a6ccedad0c02fc'),
    ('9510bc33c1267edc885d2ec15cc3f508dc3dc43248cbe9b26282f1d8da2ed6b5',
     '1ee28b29225d6b137028b58dcd5606ad04e5ee655e428070ae9d30b850c95da9',
     '97aec0b1fdad3a6fd715478694d7900a92020b14531c2fa62b8558ec834be4f9'),
    ('24653af0597de640e71888e079a5da3fd626fb1df05796371292db9c3f39b163',
     '8258145330424e4c2a6e667390fd315ebe503c73b1467b12c8c99dd7169e1ee8',
     '358afca7e4b0a36c1bdb89cfe2598ac45ca71c55afbbc7a2ab644495747b87a7'),
    ('cdedf2da642543b3e3abfea9bb1cd31f30934bded5880ee9d3bb2e18fdbe8c99',
     'c65240777d94fe6f7b96e860c8c83fc303bb1cbf6cda5b9a443aa55d46138a46',
     '2aeddafcc447e35069b4b6a8a4dc65ae6546d5b1a35ac9d66a790650f1faa014')),
    strict=True))
ARGS_PINS = {'c15': '8b031c689264ccce0c17efe0ee231492c20368b5d7b730f7111d3b308ee552a0',
             'c30': '9dbc9a1239a089dc473b450a125c55a06a41ca8d1e651567c883faf9a562af31'}
STAGES = {
    'raw': ('input_snapshot.json', *(f'norm_block_{b:02d}.npz' for b in range(16)), 'raw_arrays.npz'),
    'assessment': ('split_arrays.npz', 'split_checks.json')}
JOURNAL = 'part_a_progress.json'


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            digest.update(chunk)
    after = path.stat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if key(before) != key(after):
        raise RuntimeError('An input changed during content verification')
    return dict(path=str(path), sha256=digest.hexdigest(), size=after.st_size,
                device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns)


def same_content(left, right):
    return all(left[k] == right[k] for k in ('path', 'size', 'sha256'))


def unchanged(artifacts):
    for item in artifacts.values():
        stat = Path(item['path']).stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                item['device'], item['inode'], item['size'], item['mtime_ns']):
            raise RuntimeError('A bound input changed during use: '+item['path'])


def product(path, payload, arrays=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    if path.exists():
        raise FileExistsError('A completed product must never be overwritten')
    with temporary.open('xb') as stream:
        if arrays:
            np.savez(stream, **payload)
        else:
            stream.write((json.dumps(payload, indent=2, allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    if arrays:
        with np.load(temporary, allow_pickle=False) as saved:
            if set(saved.files) != set(payload):
                raise ValueError('Serialized array inventory differs')
            for key, expected in payload.items():
                actual = saved[key]
                expected = np.asarray(expected)
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise ValueError('Serialized array layout differs')
                if actual.tobytes(order='C') != expected.tobytes(order='C'):
                    raise ValueError('Serialized array bytes/values differ')
    os.link(temporary, path)
    temporary.unlink()
    return identity(path)


def journal_write(root, value):
    path = root/JOURNAL
    temporary = root/(JOURNAL+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inventory(root, journal):
    allowed = {JOURNAL}
    stages = journal['stages']
    if list(stages) != list(STAGES)[:len(stages)]:
        raise ValueError('The journal is not an exact completed prefix')
    for stage, saved in stages.items():
        if saved['status'] != 'completed' or set(saved['outputs']) != set(STAGES[stage]):
            raise RuntimeError('A failed, started or uncertain stage needs review')
        for name, previous in saved['outputs'].items():
            actual = identity(root/name)
            if not same_content(previous, actual):
                raise ValueError('A completed stage product changed')
            allowed.add(name)
    if {p.name for p in root.iterdir()} != allowed:
        raise FileExistsError('Orphan final, partial, temporary or unexpected output exists')


def outward(value, upper):
    value = Fraction(value)
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError('A scalar conversion is nonfinite')
    if (upper and Fraction(result) < value) or (not upper and Fraction(result) > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    return result


def exact_record(value):
    value = Fraction(value)
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def print_cell(value):
    """Open %.6e cell, including the asymmetric decade boundary at 1.000000."""
    text = format(float(value), '.6e')
    mantissa, exponent = text.split('e')
    exponent = int(exponent)
    magnitude = abs(Fraction(mantissa))
    integer = magnitude*10**6
    unit = Fraction(10)**(exponent-6)
    center = integer*unit
    previous = ((integer-1)*unit if integer > 10**6
                else Fraction(9999999)*Fraction(10)**(exponent-7))
    following = (integer+1)*unit
    lower,upper = (previous+center)/2,(center+following)/2
    return (lower,upper) if Fraction(value) > 0 else (-upper,-lower)


def cell_status(value, lower, upper):
    value, lower, upper = map(Fraction, (value, lower, upper))
    if value == lower == upper == 0:
        return 'exact zero'
    if value == 0:
        return 'unresolved; interval is not exact zero'
    lo,hi = print_cell(value)
    return 'resolved' if lo < lower <= upper < hi else 'unresolved'


def scalar(value, lower=None, upper=None):
    value = Fraction(value)
    rounded = float(value)
    rounding = abs(Fraction(rounded)-value)
    lower = value if lower is None else Fraction(lower)
    upper = value if upper is None else Fraction(upper)
    if not lower <= value <= upper:
        raise ValueError('A scalar is outside its declared interval')
    return dict(value=rounded, exact=exact_record(value), declared_6e=format(rounded, '.6e'),
        reporting_conversion_error_upper=outward(rounding, True),
        interval=[outward(lower, False), outward(upper, True)],
        exact_interval=[exact_record(lower), exact_record(upper)],
        fixed_array_cell_status=cell_status(value, lower, upper))


def environment():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('Part A requires the one-thread numerical environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Part A requires CPU-only execution')


def resources():
    import shutil
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available-16*2**30 < 64*2**30:
        raise RuntimeError('The scan would violate the 64 GiB memory reserve')
    if shutil.disk_usage('/').free < 100*10**9 or shutil.disk_usage(BASE).free < 2*2**30:
        raise RuntimeError('Insufficient root/bulk storage reserve')


def register(path):
    with (BASE/'LARGE_FILES.tsv').open('a') as stream:
        stream.write(f'{path}\t{path.stat().st_size}\theld Part A saved-spectrum split\n')
