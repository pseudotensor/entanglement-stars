#!/usr/bin/env python3
"""
core_hf_c64_round_slice_v1_1.py: the direct storage-precision control of CORE_HF_full v1.1 (reviewer A R2, reviewer B R7). Writes a copy of one
complex128 slice with its column arrays (sea_k, dust_k, R2_k) rounded to complex64 (numpy astype, round to nearest), every other member
byte-identical and in the same order, the dtype member set to 'complex64': the same layout the complex64 slices of the record carry.
    python3 core_hf_c64_round_slice_v1_1.py SRC.npz OUTDIR
Prints the member list, the output size and md5, and the max |rounded - original| / max |original| per column array.
"""
import hashlib
import os
import sys
import zipfile

import numpy as np


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, os.path.basename(src))
    assert os.path.abspath(dst) != os.path.abspath(src)
    names = [i.filename[:-4] for i in zipfile.ZipFile(src).infolist()]
    Z = np.load(src, allow_pickle=False)
    out = {}
    worst = 0.0
    for n in names:
        x = Z[n]
        if n.split('_')[0] in ('sea', 'dust', 'R2') and np.iscomplexobj(x):
            y = x.astype(np.complex64)
            d = float(np.max(np.abs(y.astype(np.complex128) - x)) / max(float(np.max(np.abs(x))), 1e-300)) if x.size else 0.0
            worst = max(worst, d)
            out[n] = y
        elif n == 'dtype':
            out[n] = np.array('complex64')
        else:
            out[n] = x
    np.savez(dst, **out)
    h = hashlib.md5()
    with open(dst, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    print('members', len(names), 'first', names[:10])
    print('written', dst, 'size', os.path.getsize(dst), 'md5', h.hexdigest(), 'worst column rounding (max abs / max abs)', '%.3e' % worst)


if __name__ == '__main__':
    main()
