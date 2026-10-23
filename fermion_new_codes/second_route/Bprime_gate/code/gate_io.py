"""Save numerical reductions once; renderers consume the resulting artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def save_record(outdir, name, record):
    import numpy as np
    from gate_runtime import enforce_storage, identity
    outdir = Path(outdir)
    arrays = {}

    def extract(value, trail):
        if isinstance(value, np.ndarray):
            key = '__'.join(trail)
            if key in arrays:
                raise ValueError('Duplicate array path')
            arrays[key] = value
            return {'array_key': key, 'shape': list(value.shape), 'dtype': value.dtype.str}
        if isinstance(value, np.generic):
            return extract(value.item(), trail)
        if isinstance(value, complex):
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, dict):
            return {str(k): extract(v, trail+[str(k)]) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [extract(v, trail+[str(k)]) for k, v in enumerate(value)]
        if isinstance(value, Path):
            return str(value)
        return value

    literal = extract(record, ['record'])
    archive, target = outdir/(name+'.npz'), outdir/(name+'.json')
    if archive.exists() or target.exists():
        raise ValueError('Fresh record filenames required')
    estimate = sum(value.nbytes for value in arrays.values())+1024**2
    enforce_storage(2*estimate)
    if arrays:
        # File object prevents accidental filename extension changes.
        with archive.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        literal['output_arrays'] = identity(archive)
    encoded = json.dumps(literal, indent=2, sort_keys=True, allow_nan=False)+'\n'
    enforce_storage(len(encoded.encode()))
    with target.open('x') as stream:
        stream.write(encoded)
    enforce_storage()
    return identity(target)


def logical_identity(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(memoryview(value).cast('B'))
    return {'sha256': digest.hexdigest(), 'shape': list(value.shape), 'dtype': value.dtype.str}


def restore_record(path):
    """Only called by a guarded saved-artifact renderer."""
    import numpy as np
    from gate_runtime import require_identity
    path = Path(path)
    record = json.loads(path.read_text())
    binding = record.get('output_arrays')
    archive = None
    if binding:
        require_identity(binding['path'], binding['sha256'])
        archive = np.load(binding['path'], allow_pickle=False)

    def restore(value):
        if isinstance(value, dict):
            if set(value) == {'array_key', 'shape', 'dtype'}:
                array = archive[value['array_key']]
                if list(array.shape) != value['shape'] or array.dtype.str != value['dtype']:
                    raise ValueError('Saved array descriptor mismatch')
                return array
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value
    try:
        return restore(record)
    finally:
        if archive is not None:
            archive.close()
