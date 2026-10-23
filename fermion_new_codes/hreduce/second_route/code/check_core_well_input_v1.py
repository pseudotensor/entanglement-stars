"""Compare the L archive reader with NumPy; no scientific contraction."""

import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
import warnings
import zipfile

import numpy as np

from core_well_input_v1 import StoredNpz, file_identity


ROOT = Path('<internal>/profiles_v1')
SOURCE = Path('<internal>/states'
              'c19_rs64_A_R2_tau8_ell4_avg_r11')
OUTPUT = ROOT / 'input_reader_check.json'


def require_refusal(operation, kind=ValueError):
    try:
        operation()
    except kind:
        return
    raise AssertionError('Invalid input was not refused')


def equal_member(reader, reference, key):
    actual, expected = reader.array(key), reference[key]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes(order='C') == expected.tobytes(order='C')
    assert not actual.flags.writeable


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box output folder')
    if shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('The root-disk reserve is unavailable')
    start = time.monotonic()
    record = dict(status='running', purpose='owner66 L readonly input correctness',
                  scientific_evaluations=0, actual_archives=[], synthetic_checks=[])
    with OUTPUT.open('x') as stream:
        json.dump(record, stream)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
        resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
        with tempfile.TemporaryDirectory(prefix='reader_check_', dir=ROOT) as scratch:
            folder = Path(scratch)
            source = folder / 'numeric.npz'
            values = dict(c=np.arange(12, dtype='float64').reshape(3, 4),
                          f=np.asfortranarray(np.arange(12).reshape(3, 4)).astype('complex64'),
                          endian=np.arange(6, dtype='>f8'),
                          empty=np.empty((2, 0), dtype='float32'), scalar=np.array(7))
            np.savez(source, **values)
            with StoredNpz(source) as reader, np.load(source, allow_pickle=False) as native:
                for key in values:
                    equal_member(reader, native, key)
                    record['synthetic_checks'].append(key + ':exact_bytes_dtype_shape_readonly')
                require_refusal(lambda: reader.array('c').__setitem__((0, 0), 9))
                record['synthetic_checks'].append('readonly_write_refused')
            compressed = folder / 'compressed.npz'
            np.savez_compressed(compressed, c=values['c'])
            with StoredNpz(compressed) as reader:
                require_refusal(lambda: reader.array('c'))
            record['synthetic_checks'].append('compressed_refused')
            objects = folder / 'objects.npz'
            np.savez(objects, c=np.array([{}], dtype=object))
            with StoredNpz(objects) as reader:
                require_refusal(lambda: reader.array('c'))
            record['synthetic_checks'].append('object_refused')
            duplicate = folder / 'duplicate.npz'
            data = io.BytesIO()
            np.save(data, values['c'], allow_pickle=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(duplicate, 'w') as archive:
                    archive.writestr('c.npy', data.getvalue())
                    archive.writestr('c.npy', data.getvalue())
            require_refusal(lambda: StoredNpz(duplicate))
            record['synthetic_checks'].append('duplicate_refused')
            replacement = folder / 'replacement.npz'
            np.savez(replacement, **values)
            with StoredNpz(source) as reader:
                replacement.replace(source)
                require_refusal(reader.verify_identity, RuntimeError)
            record['synthetic_checks'].append('changed_path_binding_refused')
        for clock in ('0012.00', '0015.00', '0016.00'):
            path = SOURCE / ('orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T' + clock + '.npz')
            keys = ['R', 'T', 'rs', 'N_applied', 'v_applied', 'kappas', 'weights', 'dtype']
            keys += [prefix + str(k) for k in range(1, 11) for prefix in ('dust_', 'f_')]
            with StoredNpz(path) as reader, np.load(path, allow_pickle=False) as native:
                for key in keys:
                    equal_member(reader, native, key)
                large_headers = {}
                for key in ('sea_1', 'R2_1'):
                    array = reader.array(key)
                    assert array.shape == (4096, 2048)
                    assert not array.flags.writeable
                    mapped_bytes = len(array._mmap)
                    assert mapped_bytes <= array.nbytes + mmap.ALLOCATIONGRANULARITY
                    large_headers[key] = dict(shape=array.shape, dtype=str(array.dtype),
                                              mapped_bytes=mapped_bytes)
                    del array
                reader.verify_identity()
            record['actual_archives'].append(dict(input=file_identity(path),
                exact_small_members=keys, large_members_header_and_readonly_only=large_headers))
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
    record.update(elapsed_seconds=time.monotonic() - start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('core_well_input_v1.py').read_bytes()).hexdigest(),
        check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='exact byte/dtype/shape checks of small members; no whole-array check of sea/R2 payloads')
    temporary = OUTPUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(OUTPUT)
    print(json.dumps({key: record[key] for key in ('status', 'elapsed_seconds', 'peak_rss_kib')}))
    if record['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
