#!/usr/bin/env python3
"""route2_neumann_check.py (busi lane, ROUTE2 v1): the Neumann series of the dressing on a main S3 member, from its saved npz
(B, b at the radii) and route 2's cached channels: term norms ||(B P^A)^n b||/||b|| and Var_C, Var_P of the partial sums to
order --nmax. Information on the convergence of beta = (I + B P^A)^-1 b (the spec's first-order form and its remainder
criterion are not a convergence certificate when rho(B P^A) > 1, as the compact regression showed).
Interface as route2_variance.py: --star (default the family file), --member, --tag (default m<member>; reads
<out>/route2_variance_<tag>.npz, writes <out>/route2_neumann_<tag>.json), --out, --nmax, --threads. A diagnostic, not a
certificate (reviewer A red team ruling 4)."""
import os, sys
_thr = sys.argv[sys.argv.index('--threads') + 1] if '--threads' in sys.argv else '3'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = _thr
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC

ap = argparse.ArgumentParser()
ap.add_argument('--star', default=RC.FAMILY)
ap.add_argument('--member', type=int, default=1)
ap.add_argument('--tag', default=None)
ap.add_argument('--nmax', type=int, default=6)
ap.add_argument('--out', required=True)
ap.add_argument('--threads', default='3')
a = ap.parse_args()
tag = a.tag or 'm%d' % a.member
mem, _ = RC.load_member(a.star, a.member)
ns = mem['nsites']
Z = np.load(os.path.join(a.out, 'route2_variance_%s.npz' % tag))
B = Z['B']
chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], ns, mem['N_B'], mem['half'], logf=lambda *x: None)
res = dict(star=a.star, member=a.member, tag=tag, N_B=mem['N_B'], ell=mem['ell'], nmax=a.nmax)
for k in ('R_surf', 'R_999'):
    b = Z['b_%s' % k]
    t = b.copy(); beta = b.copy(); tn = [1.0]; VC = []; VP = []
    v = RC.variances(chans, beta, ns); VC.append(v[0]); VP.append(v[1])
    for n in range(1, a.nmax + 1):
        t = -(B @ RC.response_apply(chans, t, ns))
        beta = beta + t
        tn.append(float(np.linalg.norm(t) / np.linalg.norm(b)))
        v = RC.variances(chans, beta, ns); VC.append(v[0]); VP.append(v[1])
        RC.log('%s order %d: term %.3e Var_C %.12e Var_P %.12e excess %.12e' % (k, n, tn[-1], VC[-1], VP[-1], VC[-1] - VP[-1]))
    res[k] = dict(term_norms=tn, VarC=VC, VarP=VP)
RC.dump_json(os.path.join(a.out, 'route2_neumann_%s.json' % tag), res)
