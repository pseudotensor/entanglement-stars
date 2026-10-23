#!/usr/bin/env python3
"""
j4_prefix_tests_v3b.py -- diagnostics of the prefix assembly on the pod (busi lane, task 1; lead's task B4;
busi/notes/J4_v2_B4_judgment_v1.md section 2).  CPU only, a few minutes with 4 threads.  Whatever it prints, the
prefix driver is NOT used in version 2 (the version-2 driver runs stages D and E); the output locates the defect.

  build  the R 12 window (L 96), largest ball 182, its 10 blocks;
  (0)    the entropy of the largest ball itself by the PrefixWindow object against its record at L 96 (|Delta S| <= 1e-9:
         isolates the assembly path from the prefix hypothesis);
  (a)    whether the orbit representatives of the largest ball are in non-decreasing |x|^2 (the shell-order hypothesis);
  (i)    entrywise: the leading submatrices against the direct assembly of 4 balls of the window, every irrep;
  (ii)   the entropies of all 61 balls of the window by the prefix route against the records (diagnostic size);
  (i-b)  the same entrywise test with the sites of the largest ball sorted by |x|^2 before BallSymmetry is built;
  (ii-b) the 61 entropies with the shell-sorted variant, if (i-b) passed;
  (iii)  3 balls of the R 34 window at L 272 with whichever variant passed (i) or (i-b); skipped if none.

Usage: j4_prefix_tests_v3b.py --v1-dir prod --v2-dir prod_v2 --cache-dir prod_v2/kcache [--skip-34] [--threads 4]
"""
import argparse
import os
import sys
import time


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=4)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                           # noqa: E402
import j4window_v2 as jw                     # noqa: E402
import j4prefix_v3b as jp                    # noqa: E402
from j4production_scan import scan_done      # noqa: E402
import json                                  # noqa: E402

RESULTS = []


def run(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        print('[PASS] %s: %s' % (name, msg), flush=True)
        RESULTS.append(True)
        ok = True
    except Exception as e:
        print('[FAIL] %s: %s: %s' % (name, type(e).__name__, e), flush=True)
        RESULTS.append(False)
        ok = False
    print('      (%s: %.1f s)' % (name, time.time() - t0), flush=True)
    return ok


def record_S(path):
    with open(path) as fh:
        return float(json.load(fh)['S'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1-dir', required=True)
    ap.add_argument('--v2-dir', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--skip-34', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    cache = args.cache_dir or os.path.join(args.v2_dir, 'kcache')
    done = scan_done([args.v1_dir, args.v2_dir])
    log = lambda s: print(s, flush=True)  # noqa: E731
    state = {}

    def build(key, shell_sort):
        pw = jp.PrefixWindow(12, 3.0, 96, cache_dir=cache, log=log, shell_sort=shell_sort)
        pw.assemble()
        state[key] = pw
        ro = pw.reps_order()
        state[key + '_reps'] = ro
        rep_msg = 'reps order not recognised' if ro is None else ('reps in non-decreasing |x|^2' if ro[0] else 'reps NOT in |x|^2 order (%d descents; first norms %s)' % (int(np.sum(np.diff(ro[1]) < 0)), list(ro[1][:12])))
        return 'largest ball %d, blocks %s; %s' % (pw.N, {nm: list(pw.blocks[nm].shape) for nm in pw.names}, rep_msg)

    def t_build():
        return build('pw', False)

    def t_largest():
        pw = state['pw']
        n = pw.N
        assert (n, 96) in done, 'no record for the largest ball %d at L 96' % n
        S_prefix = pw.entropy(n, method='gram')['S']
        S_rec = record_S(done[(n, 96)])
        assert abs(S_prefix - S_rec) <= 1e-9, (S_prefix, S_rec, S_prefix - S_rec)
        return 'largest ball %d: prefix object %.9f, record %.9f (the assembly path agrees; what fails below is the prefix hypothesis)' % (n, S_prefix, S_rec)

    def t_reps():
        ro = state['pw_reps']
        assert ro is not None, 'the representation of symN.reps is not recognised (neither (n, 3) coordinates nor site indices)'
        mono, norms = ro
        assert mono, 'orbit representatives not in |x|^2 order: %d descents, first norms %s (the remedy is a shell-ordered basis in j4lib or an exposed row-to-orbit map)' % (int(np.sum(np.diff(norms) < 0)), list(norms[:12]))
        return 'reps in non-decreasing |x|^2 (first norms %s): the orbit order is the shell order; a failure of (i) then means the basis vectors are not orbit-local' % list(norms[:12])

    def entrywise(key):
        pw = state[key]
        probe = [pw.balls[0], jw.v1_balls(12)[0], pw.balls[len(pw.balls) // 2], pw.balls[-1]]
        worst, msgs = 0.0, []
        for n in probe:
            rep = jp.compare_blocks(pw, n, cache_dir=cache)
            bad = {nm: r for nm, r in rep.items() if not r['ok']}
            worst = max([worst] + [r.get('max_diff', 0.0) for r in rep.values() if 'max_diff' in r])
            msgs.append('ball %d: %s' % (n, 'all 10 irreps agree (max %.1e)' % max(r.get('max_diff', 0.0) for r in rep.values()) if not bad else bad))
            assert not bad, 'ball %d: %s' % (n, bad)
        return '; '.join(msgs) + '; worst entry %.1e' % worst

    def entropies_12(key):
        pw = state[key]
        worst, count, missing = 0.0, 0, []
        for n in pw.balls:
            if (n, 96) not in done:
                missing.append(n)
                continue
            S_prefix = pw.entropy(n, method='gram')['S']
            worst = max(worst, abs(S_prefix - record_S(done[(n, 96)])))
            count += 1
        assert worst <= 1e-9, 'max |Delta S| %.3e nats over %d balls' % (worst, count)
        assert count >= 50, count
        return '%d balls of the R 12 window against the records at L 96: max |Delta S| %.1e%s' % (count, worst, ('; no record for %s' % missing) if missing else '')

    def t_34(shell_sort):
        pw = jp.PrefixWindow(34, 3.0, 272, cache_dir=cache, log=log, shell_sort=shell_sort)
        pw.assemble()
        worst, used = 0.0, []
        for n in (pw.balls[0], pw.balls[len(pw.balls) // 2], pw.balls[-1]):
            if (n, 272) not in done:
                continue
            S_prefix = pw.entropy(n, method='gram')['S']
            worst = max(worst, abs(S_prefix - record_S(done[(n, 272)])))
            used.append(n)
        assert used and worst <= 1e-9, (used, worst)
        return 'balls %s of the R 34 window against the records at L 272: max |Delta S| %.1e' % (used, worst)

    t0 = time.time()
    run('build the R 12 window (L 96) and assemble the largest ball', t_build)
    run('(0) the largest ball itself against its record', t_largest)
    run('(a) orbit representatives in shell order', t_reps)
    ok_i = run('(i) entrywise leading submatrices against the direct assembly', lambda: entrywise('pw'))
    run('(ii) entropies of the R 12 window against the records', lambda: entropies_12('pw'))
    run('(i-b) build with the sites of the largest ball shell-sorted', lambda: build('pws', True))
    ok_ib = ('pws' in state) and run('(i-b) entrywise with the shell-sorted variant', lambda: entrywise('pws'))
    if ok_ib:
        run('(ii-b) entropies of the R 12 window, shell-sorted variant', lambda: entropies_12('pws'))
    else:
        print('[SKIP] (ii-b): the shell-sorted variant did not pass (i-b)', flush=True)
    if not args.skip_34:
        if ok_i or ok_ib:
            run('(iii) 3 balls of the R 34 window against the records (%s)' % ('original' if ok_i else 'shell-sorted'), lambda: t_34(not ok_i))
        else:
            print('[SKIP] (iii): no variant passed the entrywise test at R 12', flush=True)
    print('\nSUMMARY: %d PASS, %d FAIL  (%.0f s); the prefix driver is not used in version 2 whatever this summary says' % (sum(RESULTS), len(RESULTS) - sum(RESULTS), time.time() - t0))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
