"""Strict representation-only repair for NumPy boolean report leaves."""
import json
import os
from pathlib import Path

import numpy as np


def native_booleans(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: native_booleans(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native_booleans(item) for item in value]
    if isinstance(value, tuple):
        return tuple(native_booleans(item) for item in value)
    return value


def write_new_json(path, value):
    path = Path(path)
    text = json.dumps(native_booleans(value), indent=2, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.partial')
    if path.exists():
        raise FileExistsError('A completed report must not be replaced')
    with temporary.open('x') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write_stage_json(path, value):
    # Preserve the inherited stage journal's replacement semantics exactly.
    from hold_exact import write_json
    write_json(path, native_booleans(value))
