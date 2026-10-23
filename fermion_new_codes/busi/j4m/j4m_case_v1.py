#!/usr/bin/env python3
"""
j4m_case_v1.py -- one case of the massive J4 rerun (busi lane, task 4; spec J4_massive_spec_v1.md): the digital ball
{|x|^2 <= n} (canonicalized) in the antiperiodic box L at mass mu, by the box route (production) and the box-free
infinite-lattice route, written as one atomic record: <id>.json and <id>_eig.npz (the eigenvalues or singular values
of every block of every route; the record's S is rebuilt from them before writing).  Resumable: a complete record is
not recomputed (--force overrides).  At mu = 0 the box route is j4lib_v2's validated sublattice reduction and the
full-block code runs as a check for n <= 20,000 sites (or --fullblock-massless on).

Usage: j4m_case_v1.py --n2max 145 --L 192 --mu 0.2 --window-centre 12 --offset 1 --kind prod
                      --outdir records --cache-dir cache [--threads 4] [--method eigh|torch] [--no-inf]
                      [--kernel-check auto|on|off] [--fullblock-massless auto|on|off] [--force] [--quiet]
"""
import argparse
import json
import os
import resource
import socket
import sys
import time


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=1)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
HERE = os.path.dirname(os.path.abspath(__file__))
J4_DIR = os.path.normpath(os.path.join(HERE, '..', 'j4'))
for _p in (HERE, J4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                 # noqa: E402
import scipy                       # noqa: E402
import j4lib as jl                 # noqa: E402
import j4lib_v2 as jl2             # noqa: E402
import j4m_lib_v1 as jm            # noqa: E402
import j4m_kernel_v1 as jk         # noqa: E402


def case_id(R, n, L, mu):
    pre = ('R%g_' % float(R)) if R is not None else ''
    return '%sn%d_L%d_mu%s' % (pre, int(n), int(L), jm.mu_tag(mu))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def record_complete(path):
    try:
        with open(path) as fh:
            r = json.load(fh)
        S = r['routes']['box']['S']
        return S == S
    except Exception:
        return False


def rebuild_S(irreps, arrays, prefix):
    """S from the saved per-irrep spectra: eigenvalues ('eig') or singular values with the imbalance term ('sv')."""
    S = 0.0
    for r in irreps:
        nm, d = r['irrep'], r['d']
        if prefix + 'eig_' + nm in arrays:
            S += d * float(np.sum(jl.binary_entropy(arrays[prefix + 'eig_' + nm])))
        elif prefix + 'sv_' + nm in arrays:
            S += d * (float(np.sum(jl.pair_entropy_from_singular(arrays[prefix + 'sv_' + nm]))) + abs(r['m_e'] - r['m_o']) * jl.LN2)
        else:
            raise KeyError('no saved spectrum for %s%s' % (prefix, nm))
    return 4.0 * S


def run_case(n2max, L, mu, outdir, cache_dir, window_centre=None, offset=None, kind='prod', method='eigh', threads=1,
             do_inf=True, kernel_check='auto', fullblock_massless='auto', chunk_elems=1.6e7, force=False, log=None):
    def say(msg):
        if log is not None:
            log(msg)
    t0 = time.time()
    L, mu = int(L), float(mu)
    meta, sites, sym = jm.setup_symmetry(n2max)
    n_can = int(meta['n2_ball'])
    cid = case_id(window_centre, n_can, L, mu)
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, cid + '.json')
    zpath = os.path.join(outdir, cid + '_eig.npz')
    if not force and record_complete(jpath) and os.path.exists(zpath):
        say('case %s already complete; skipping' % cid)
        return 'skipped', jpath
    if 2.0 * meta['R_hi'] >= L:
        raise ValueError('the ball must fit in the box: need L > 2 R_hi')
    if sym.M >= L:
        raise ValueError('displacement range exceeds the box')
    n = sym.n
    kc = kernel_check == 'on' or (kernel_check == 'auto' and n <= 15000)
    fbm = fullblock_massless == 'on' or (fullblock_massless == 'auto' and n <= 20000)
    R_eff = None if (window_centre is None or offset is None) else float(window_centre) + float(offset) / 16.0
    say('case %s: ball %d (requested %d), sites %d (e %d, o %d), R in (%.5f, %.5f], R_vol %.5f, orbits %d, M %d'
        % (cid, n_can, int(n2max), n, sym.n_e, sym.n_o, meta['R_lo'], meta['R_hi'], meta['R_vol'], len(sym.reps), sym.M))
    rec = dict(version=1, id=cid, kind=kind, n2max=n_can, n2max_requested=int(n2max), R_lo=meta['R_lo'], R_hi=meta['R_hi'],
               R_mid=meta['R_mid'], R_vol=meta['R_vol'], window_centre=window_centre, offset_k=offset, R_eff=R_eff, L=L, mu=mu,
               n=n, n_e=sym.n_e, n_o=sym.n_o, sites_hash=jl.sites_hash(sites), M_disp=int(sym.M), orbits=int(len(sym.reps)),
               method=method, threads=int(threads), routes={}, checks={}, timing={})
    arrays = {}
    # ---- the box route ----------------------------------------------------------------------
    tb = time.time()
    if mu == 0.0:
        v2rec, svals, _sym2 = jl2.reduced_ball_entropy_v2(n_can, L, method='gram', chunk_elems=chunk_elems,
                                                          cache_dir=os.path.join(cache_dir, 'v2_massless'),
                                                          window_centre=window_centre, keep_values=True, log=log)
        del _sym2
        rec['routes']['box'] = dict(route='j4lib_v2 sublattice SVD (massless)', S=v2rec['S'], S_scalar=v2rec['S_scalar'],
                                    kernel_source=v2rec['kernel_source'], kernel_junk=v2rec['kernel_structure_junk'],
                                    irreps=v2rec['irreps'], largest_block=v2rec['largest_block'],
                                    dimension_check=v2rec['dimension_check'])
        for r in v2rec['irreps']:
            arrays['box_sv_' + r['irrep']] = np.asarray(svals[r['irrep']], dtype=float)
        if fbm:
            gtab, mtab, diag, src = jm.box_tables_cached(L, 0.0, sym.M, cache_dir, log=log)
            fb, fbv = jm.massive_entropy_from_tables(sym, gtab, mtab, 0.0, method=method, chunk_elems=chunk_elems, log=log)
            rec['routes']['box_fullblock'] = dict(route='full block at mu 0 (check)', S=fb['S'], S_scalar=fb['S_scalar'],
                                                  kernel=diag, kernel_source=src, irreps=fb['irreps'], largest_block=fb['largest_block'])
            rec['checks']['fullblock_minus_sublattice'] = fb['S'] - v2rec['S']
            for r in fb['irreps']:
                arrays['boxfb_eig_' + r['irrep']] = fbv[r['irrep']]
    else:
        gtab, mtab, diag, src = jm.box_tables_cached(L, mu, sym.M, cache_dir, log=log)
        if diag['junk'] > 1e-13:
            raise ArithmeticError('kernel parity structure violated: %g' % diag['junk'])
        fb, fbv = jm.massive_entropy_from_tables(sym, gtab, mtab, mu, method=method, chunk_elems=chunk_elems, log=log)
        rec['routes']['box'] = dict(route='full block massive', S=fb['S'], S_scalar=fb['S_scalar'], kernel=diag, kernel_source=src,
                                    irreps=fb['irreps'], largest_block=fb['largest_block'], dimension_check=fb['dimension_check'])
        for r in fb['irreps']:
            arrays['box_eig_' + r['irrep']] = fbv[r['irrep']]
    rec['timing']['box_s'] = time.time() - tb
    # ---- the box-free route ----------------------------------------------------------------
    if do_inf:
        ti = time.time()
        gti, mti, alt, info, srci = jk.inf_tables_cached(mu, sym.M, cache_dir, log=log)
        if mu == 0.0:
            ri, vi = jm.massless_entropy_from_gtab(sym, gti, method='gram', chunk_elems=chunk_elems, log=log)
            tag = 'inf_sv_'
        else:
            ri, vi = jm.massive_entropy_from_tables(sym, gti, mti, mu, method=method, chunk_elems=chunk_elems, log=log)
            tag = 'inf_eig_'
        rec['routes']['inf'] = dict(route='box-free Schwinger-Bessel kernel', S=ri['S'], S_scalar=ri['S_scalar'], kernel=info,
                                    kernel_source=srci, irreps=ri['irreps'])
        for r in ri['irreps']:
            arrays[tag + r['irrep']] = vi[r['irrep']]
        rec['checks']['box_minus_inf'] = rec['routes']['box']['S'] - ri['S']
        if kc and alt is not None:
            if mu == 0.0:
                ra, _ = jm.massless_entropy_from_gtab(sym, alt[0], method='gram', chunk_elems=chunk_elems, keep_values=False)
            else:
                ra, _ = jm.massive_entropy_from_tables(sym, alt[0], alt[1], mu, method=method, chunk_elems=chunk_elems, keep_values=False)
            rec['checks']['inf_alt_minus_inf'] = ra['S'] - ri['S']
        rec['timing']['inf_s'] = time.time() - ti
    # ---- rule 10: the saved spectra rebuild the record --------------------------------------
    for key, prefix in (('box', 'box_'), ('box_fullblock', 'boxfb_'), ('inf', 'inf_')):
        if key in rec['routes']:
            S_re = rebuild_S(rec['routes'][key]['irreps'], arrays, prefix)
            rec['checks']['rebuild_%s_minus_record' % key] = S_re - rec['routes'][key]['S']
            if abs(S_re - rec['routes'][key]['S']) > 1e-9 * max(1.0, abs(rec['routes'][key]['S'])):
                raise ArithmeticError('entropy rebuilt from the saved spectra differs from the record (%s)' % key)
    rec['timing']['total_s'] = time.time() - t0
    rec.update(wall_s=time.time() - t0, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               host=socket.gethostname(), numpy=np.__version__, scipy=scipy.__version__)
    arrays.update(S_box=np.array(rec['routes']['box']['S']), n2max=np.array(n_can), L=np.array(L), mu=np.array(mu),
                  irrep_names=np.array(jm.IRREP_NAMES))
    if 'inf' in rec['routes']:
        arrays['S_inf'] = np.array(rec['routes']['inf']['S'])
    tmpz = zpath[:-4] + '.tmp.npz'
    np.savez_compressed(tmpz, **arrays)
    os.replace(tmpz, zpath)
    tmpj = jpath + '.tmp'
    with open(tmpj, 'w') as fh:
        json.dump(jsonable(rec), fh, indent=1)
    os.replace(tmpj, jpath)
    s_inf = rec['routes']['inf']['S'] if 'inf' in rec['routes'] else float('nan')
    print('j4m case %s: sites %d  S_box = %.12f  S_inf = %.12f  box-inf = %.3e  [%s, %.0f s, peak %.2f GB]'
          % (cid, n, rec['routes']['box']['S'], s_inf, rec['checks'].get('box_minus_inf', float('nan')), method, rec['wall_s'],
             rec['peak_rss_gb']), flush=True)
    return 'done', jpath


def main():
    ap = argparse.ArgumentParser(description='one case of the massive J4 rerun (box and box-free routes)')
    ap.add_argument('--n2max', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--mu', type=float, required=True)
    ap.add_argument('--window-centre', type=float, default=None)
    ap.add_argument('--offset', type=int, default=None, help='the offset k of R_eff = R + k/16 (recorded)')
    ap.add_argument('--kind', default='prod')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--method', choices=['eigh', 'torch'], default='eigh')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--chunk-elems', type=float, default=1.6e7)
    ap.add_argument('--no-inf', action='store_true')
    ap.add_argument('--kernel-check', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--fullblock-massless', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    log = None if args.quiet else (lambda s: print(s, flush=True))
    run_case(args.n2max, args.L, args.mu, args.outdir, args.cache_dir, window_centre=args.window_centre, offset=args.offset,
             kind=args.kind, method=args.method, threads=args.threads, do_inf=not args.no_inf, kernel_check=args.kernel_check,
             fullblock_massless=args.fullblock_massless, chunk_elems=args.chunk_elems, force=args.force, log=log)


if __name__ == '__main__':
    main()
