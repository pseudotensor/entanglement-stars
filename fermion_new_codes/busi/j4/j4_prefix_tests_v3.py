#!/usr/bin/env python3
"""
j4_prefix_tests_v3.py -- the exact test of the prefix assembly on the pod (busi lane, task 1; lead's task B4;
busi/notes/J4_v2_B4_v1.md section 4.3).  CPU only, about 20 minutes with 4 threads.

  (i)   entrywise: the leading submatrices of the largest ball of the R 12 window (L 96) against the direct version-1
        assembly of 4 balls of that window, every irrep: equal shapes and max |difference| <= 1e-12;
  (ii)  the entropies of all balls of the R 12 window by the prefix route (gram) against the records (stage B in
        --v2-dir, the version-1 balls in --v1-dir) at L 96: |Delta S| <= 1e-9;
  (iii) 3 balls of the R 34 window at L 272 against the records (optional, --skip-34 to omit).
If (i) fails with equal shapes and equal sorted row and column norms, j4lib's basis is not in shell order (remedy in
the note); the prefix route is then not used.

Usage: j4_prefix_tests_v3.py --v1-dir prod --v2-dir prod_v2 --cache-dir prod_v2/kcache [--skip-34] [--threads 4]
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
import j4window_v2 as jw                     # noqa: E402
import j4prefix_v3 as jp                     # noqa: E402
from j4production_scan import scan_done      # noqa: E402
import json                                  # noqa: E402

RESULTS = []


def run(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        print('[PASS] %s: %s' % (name, msg), flush=True)
        RESULTS.append(True)
    except Exception as e:
        print('[FAIL] %s: %s: %s' % (name, type(e).__name__, e), flush=True)
        RESULTS.append(False)
    print('      (%s: %.1f s)' % (name, time.time() - t0), flush=True)


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

    def t_build():
        state['pw'] = jp.PrefixWindow(12, 3.0, 96, cache_dir=cache, log=log)
        state['pw'].assemble()
        return 'largest ball %d, blocks %s' % (state['pw'].N, {nm: list(state['pw'].blocks[nm].shape) for nm in state['pw'].names})

    def t_entrywise():
        pw = state['pw']
        probe = [pw.balls[0], jw.v1_balls(12)[0], pw.balls[len(pw.balls) // 2], pw.balls[-1]]
        worst, msgs = 0.0, []
        for n in probe:
            rep = jp.compare_blocks(pw, n, cache_dir=cache)
            bad = {nm: r for nm, r in rep.items() if not r['ok']}
            worst = max([worst] + [r.get('max_diff', 0.0) for r in rep.values() if 'max_diff' in r])
            msgs.append('ball %d: %s' % (n, 'all 10 irreps agree (max %.1e)' % max(r.get('max_diff', 0.0) for r in rep.values()) if not bad else bad))
            assert not bad, 'ball %d: %s' % (n, bad)
        return '; '.join(msgs) + '; worst entry %.1e' % worst

    def t_entropies_12():
        pw = state['pw']
        worst, count, missing = 0.0, 0, []
        for n in pw.balls:
            if (n, 96) not in done:
                missing.append(n)
                continue
            S_prefix = pw.entropy(n, method='gram')['S']
            S_rec = record_S(done[(n, 96)])
            worst = max(worst, abs(S_prefix - S_rec))
            count += 1
        assert worst <= 1e-9, worst
        assert count >= 50, count
        return '%d balls of the R 12 window against the records at L 96: max |Delta S| %.1e%s' % (count, worst, ('; no record for %s' % missing) if missing else '')

    def t_34():
        pw = jp.PrefixWindow(34, 3.0, 272, cache_dir=cache, log=log)
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
    run('(i) entrywise leading submatrices against the direct assembly', t_entrywise)
    run('(ii) entropies of the R 12 window against the records', t_entropies_12)
    if not args.skip_34:
        run('(iii) 3 balls of the R 34 window against the records', t_34)
    print('\nSUMMARY: %d PASS, %d FAIL  (%.0f s)' % (sum(RESULTS), len(RESULTS) - sum(RESULTS), time.time() - t0))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
