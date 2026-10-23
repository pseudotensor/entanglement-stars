"""Read original-r11 numeric NPY members in place without extracting the NPZ.

File handling for owner66 L only. This module computes no scientific
observable and never modifies an input. Object-valued dense-field members
belong to the separate, explicitly trusted historical-fields reader.
"""

from math import prod
import os
from pathlib import Path
import struct
import zipfile

import numpy as np
from numpy.lib import format as npy


LOCAL_HEADER = struct.Struct('<4s5H3I2H')


def file_identity(path):
    state = Path(path).stat()
    return dict(path=str(Path(path)), size=state.st_size,
                mtime_ns=state.st_mtime_ns, device=state.st_dev,
                inode=state.st_ino)


def stat_fields(state):
    return state.st_size, state.st_mtime_ns, state.st_dev, state.st_ino


class StoredNpz:
    """Numeric ZIP_STORED members, mapped read-only at checked byte offsets."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('rb')
        self.archive = None
        try:
            self.initial = os.fstat(self.stream.fileno())
            self.archive = zipfile.ZipFile(self.stream)
            entries = self.archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive member names')
            self.entries = {entry.filename: entry for entry in entries}
            ordered = sorted(entries, key=lambda entry: entry.header_offset)
            self.ends = {entry.filename: (ordered[index + 1].header_offset
                         if index + 1 < len(ordered) else self.archive.start_dir)
                         for index, entry in enumerate(ordered)}
            self.verify_identity()
        except Exception:
            self.close()
            raise

    def verify_identity(self):
        """Check both the open descriptor and its current pathname binding."""
        initial = stat_fields(self.initial)
        if (stat_fields(os.fstat(self.stream.fileno())) != initial or
                stat_fields(self.path.stat()) != initial):
            raise RuntimeError('Input archive changed while being read')

    @property
    def keys(self):
        return tuple(name[:-4] for name in self.entries if name.endswith('.npy'))

    def member_header(self, key):
        if not key or '/' in key or '\\' in key:
            raise ValueError('Expected a flat, explicitly named array key')
        self.verify_identity()
        entry = self.entries[key + '.npy']
        if entry.compress_type != zipfile.ZIP_STORED:
            raise ValueError('Memory mapping requires an uncompressed member')
        if entry.flag_bits & 1:
            raise ValueError('Encrypted archive member')
        self.stream.seek(entry.header_offset)
        encoded = self.stream.read(LOCAL_HEADER.size)
        if len(encoded) != LOCAL_HEADER.size:
            raise ValueError('Truncated ZIP local header')
        header = LOCAL_HEADER.unpack(encoded)
        if (header[0] != b'PK\x03\x04' or header[2] != entry.flag_bits or
                header[3] != entry.compress_type):
            raise ValueError('ZIP local and central headers disagree')
        raw_name = self.stream.read(header[9])
        encoding = 'utf-8' if header[2] & 0x800 else 'cp437'
        if raw_name.decode(encoding) != entry.filename:
            raise ValueError('ZIP member name mismatch')
        offset = entry.header_offset + LOCAL_HEADER.size + header[9] + header[10]
        if (offset < self.stream.tell() or
                offset + entry.file_size > self.ends[entry.filename] or
                offset + entry.file_size > self.initial.st_size):
            raise ValueError('ZIP member exceeds its archive bounds')
        self.stream.seek(offset)
        version = npy.read_magic(self.stream)
        if version == (1, 0):
            shape, fortran, dtype = npy.read_array_header_1_0(self.stream)
        elif version == (2, 0):
            shape, fortran, dtype = npy.read_array_header_2_0(self.stream)
        else:
            raise ValueError('Unsupported NPY header version: ' + str(version))
        if dtype.hasobject:
            raise ValueError('Object-valued NPY data cannot be memory mapped')
        data_offset = self.stream.tell()
        size = prod(shape) * dtype.itemsize
        if size < 0 or data_offset + size != offset + entry.file_size:
            raise ValueError('NPY shape/dtype does not match member length')
        self.verify_identity()
        return dict(shape=shape, dtype=dtype, fortran_order=fortran,
                    offset=data_offset, data_bytes=size,
                    zip_crc32=entry.CRC)

    def array(self, key):
        header = self.member_header(key)
        order = 'F' if header['fortran_order'] else 'C'
        if header['data_bytes'] == 0:
            result = np.empty(header['shape'], dtype=header['dtype'], order=order)
            result.flags.writeable = False
            return result
        return np.memmap(self.stream, dtype=header['dtype'], mode='r',
                         offset=header['offset'], shape=header['shape'], order=order)

    def close(self):
        if self.archive is not None:
            self.archive.close()
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
