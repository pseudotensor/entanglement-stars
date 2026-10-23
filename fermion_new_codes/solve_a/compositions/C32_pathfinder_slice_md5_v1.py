#!/usr/bin/env python
"""C32_pathfinder_slice_md5_v1.py: every .npz path the readout json names (the slices read), md5 on the box (4 parallel), compared
against the .md5 file beside it, busi's compC_keep MANIFEST_*.md5 lines, and our R2 md5 list. Writes slice_md5_v1.tsv."""
import json, re, sys, os, hashlib, glob
from concurrent.futures import ThreadPoolExecutor
J = json.load(open(sys.argv[1])); out = sys.argv[2]; r2 = sys.argv[3]
paths = sorted(set(p for p in re.findall(r'"(/[^"]+?\.npz)"', json.dumps(J)) if os.path.exists(p) and ('orbitals_' in p or 'light_' in p)))
ref = {}
for m in glob.glob('<internal>/MANIFEST_*.md5'):
    for l in open(m):
        s = l.split()
        if len(s) >= 2: ref.setdefault(os.path.basename(s[1]), set()).add(s[0])
for l in open(r2):
    s = l.split()
    if len(s) >= 2: ref.setdefault(os.path.basename(s[1]), set()).add(s[0])
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
with ThreadPoolExecutor(4) as ex: sums = list(ex.map(md5, paths))
n_ok = n_none = n_bad = 0
with open(out, 'w') as f:
    for p, s in zip(paths, sums):
        side = []
        if os.path.exists(p + '.md5'): side.append(open(p + '.md5').read().split()[0])
        side += sorted(ref.get(os.path.basename(p), []))
        st = 'NOREF' if not side else ('OK' if all(x == s for x in side) else 'MISMATCH')
        n_ok += st == 'OK'; n_none += st == 'NOREF'; n_bad += st == 'MISMATCH'
        f.write('%s\t%s\t%d\t%s\t%s\n' % (p, s, os.path.getsize(p), st, ','.join(side)))
print('slices', len(paths), 'OK', n_ok, 'NOREF', n_none, 'MISMATCH', n_bad)
