#!/usr/bin/env python
"""o5_control_fields_v1.py -- RUN_O5_controls_v1 control 1: writes the 3 constant-field records on the T7 record's own radial grid, in
record_slice.npz's form (keys r, N, v; every other key of the record copied unchanged), for both codes to read by path.
  S1  N = 1,   v = 0     the free lattice;
  S2  N = 0.5, v = 0     the lapse alone: h = N h_0 exactly on both codes, the known answer being the free propagation at the clock scaled by 0.5;
  S3  N = 1,   v = +0.5  the shift alone, outward in the chain export's sign convention ("v <= 0 inward"), chosen so that nothing reaches the origin
                         inside the window.
No root of N - |v| exists on these tables: the chain export takes the exchange's cut with --rh 22.250080 and the cube driver with --o5_rh 22.250080.
A json beside each npz records the setting, the grid, the source record's md5 and the written file's md5, with empty roots and null axes r_h for any
reader of record_slice.json's keys. Nothing here is a result of the object.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="RUN_O5_controls_v1 control 1: the constant-field records S1, S2, S3 on the record's grid")
    ap.add_argument('--record', required=True, help="record_slice.npz (keys r, N, v)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--N2', type=float, default=0.5, help="S2's constant lapse (0.5: the S2 sweep ends map to S1 readout times)")
    ap.add_argument('--v3', type=float, default=0.5, help="S3's constant shift (+0.5: outward in the chain export's convention; nothing reaches the origin)")
    args = ap.parse_args(argv)
    Z = np.load(args.record)
    keys = list(Z.files)
    for k in ('r', 'N', 'v'):
        if k not in keys:
            raise SystemExit(f"{args.record}: key {k} missing (keys {keys})")
    r = np.asarray(Z['r'], dtype=float)
    settings = [('S1', 1.0, 0.0, "the free lattice: N = 1, v = 0"),
                ('S2', float(args.N2), 0.0, f"the lapse alone: N = {args.N2:g}, v = 0 (h = N h_0; the free propagation at the clock scaled by {args.N2:g})"),
                ('S3', 1.0, float(args.v3), f"the shift alone: N = 1, v = {args.v3:+g} (outward in the chain export's convention v <= 0 inward)")]
    os.makedirs(args.out, exist_ok=True)
    src_md5 = md5(args.record)
    print(f"[source] {args.record} (md5 {src_md5}): keys {keys}; n = {r.size}, r = {r[0]:g} .. {r[-1]:g} a; the record's N in [{float(np.min(Z['N'])):.6f}, {float(np.max(Z['N'])):.6f}], "
          f"v in [{float(np.min(Z['v'])):.6f}, {float(np.max(Z['v'])):.6f}]", flush=True)
    for tag, Nc, vc, desc in settings:
        arrays = {k: np.asarray(Z[k]) for k in keys}
        arrays['N'] = np.full(r.shape, Nc, dtype=float)
        arrays['v'] = np.full(r.shape, vc, dtype=float)
        path = os.path.join(args.out, f"record_control_{tag}.npz")
        np.savez_compressed(path, **arrays)
        meta = dict(script='o5_control_fields_v1.py', setting=tag, description=desc, N=Nc, v=vc, N_minus_abs_v=Nc - abs(vc),
                    source_record=args.record, source_md5=src_md5, n=int(r.size), r_first=float(r[0]), r_last=float(r[-1]),
                    keys_copied=[k for k in keys if k not in ('N', 'v')], md5_written_npz=md5(path),
                    roots_on_table=dict(roots=[], note="no root of N - |v|: the cut is given explicitly (--rh / --o5_rh 22.250080 a, the exchange's common cut)"),
                    slice=dict(two_gE_tail_last=None), fields_written=dict(N_max=Nc, v_const=vc),
                    axes={'100': dict(r_h=None), '111': dict(r_h=None)})
        with open(path[:-4] + '.json', 'w') as fh:
            json.dump(meta, fh, indent=1)
        print(f"[written] {path} (md5 {meta['md5_written_npz']}): {desc}; N - |v| = {Nc - abs(vc):g} everywhere; json beside", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
