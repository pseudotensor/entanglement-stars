#!/usr/bin/env python
"""build_engine_twofield_v9c4.py (busi lane, PERF tracker rows H3-A1, H3-B1, H3-C1, owner rule #334; reviewer B perf_host_twofield, box
<internal>/result.md): engine_twofield_v9c4.py over
engine_twofield_v9c3c.py (md5 c6f15b2e234cf9653ad374710c5cd14c) in the lane's patch form: exact replacements, each OLD occurring once in v9c3c;
every line of v9c4 not in v9c3c carries "[v9c4 perf]"; --reverse restores v9c3c byte for byte; the source md5 is checked; an existing output is
never overwritten.

Three options, each off by default, in the review's spelling (sec. 1 plumbing: the options are consumed by an argparse action into the module's _H3
dict and never reach the parsed namespace, so the json's args equal v9c3c's; at the defaults every statement executed is v9c3c's):

  --perf-h3-band-gather (A1, review sec. 2): band_upper_v36 through a CSR-data-index gather plan (one plan per thread, the structure verified on
      every call, rebuilt when it changes; non-CSR, non-complex128, non-square or duplicate upper destinations fall back to v9c3c's scatter);
      the gather adds complex(+0) to every entry as the scatter's np.add.at does, the diagonal made real by the original statement.
  --perf-h3-anchor-workers N (B1, review sec. 3; 0..20, 0 off): the probe-point Cholesky ladders (light_sector_walk_v9's "for j in others" block)
      and the reopening tests ("for j in cand") of the anchors at one fixed probe run as host jobs in a thread pool of N workers, BLAS limited to
      1 thread (threadpoolctl) only while the coordinator waits; results consumed in the original order; everything else (path.eig, the
      passages, crossings, re-anchoring) serial and unchanged. N 1, or a single job, runs the same helpers serially in the coordinator.
  --perf-h3-weight-gather (C1, review sec. 4): _v9c_weights on a V9cPlus whose U is a torch tensor on a non-CPU device gathers the plus-sector
      columns on the device (index_select) after the unchanged full GEMM, before the D2H copy, and returns the Fortran-ordered array numpy's
      advanced column indexing gives. On a CPU tensor (the box) the option never takes this path.

Deviation from the review's sec. 1 table: only these three options are registered (A2 pd_workspace, A3 sector_signs, A4 walk_locals are not built,
so their flags are absent rather than inert). Additions to the review's code, diagnostics only: counters (band plan hits, rebuilds, fallbacks;
pool maps, jobs, parallel maps and their seconds; Gram gathers and avoided D2H bytes) printed on one "v9c4 perf (the run)" line at the end when an
option is on, and a "v9c4 perf:" settings line at the start.

Line references (the review read v9c2; in v9c3c): band_upper_v36 313 (v9c2 313), pd_banded_v36 326, v9c_add_arguments 714, _v9c_gram 809 (v9c2
about 818), _v9c_weights 861 (v9c2 about 865), light_sector_walk_v9 1524, its ladder "for j in others" 1605 (v9c2 about 1606), its reopening
"for j in cand" 1633.

Usage. python build_engine_twofield_v9c4.py engine_twofield_v9c3c.py [engine_twofield_v9c4.py]
         --check: the build in memory with the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c4.py restored.py [--against engine_twofield_v9c3c.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v9c4 perf]"
SOURCE_MD5 = "c6f15b2e234cf9653ad374710c5cd14c"   # engine_twofield_v9c3c.py

BLOCK = r'''# >>> v9c4 perf: PERF H3 (owner rule #334; reviewer B perf_host_twofield 20261020_105808, secs. 1, 2 A1, 3 B1, 4 C1): the host CSR-to-band gather, a bounded host pool for the anchor tests at a fixed probe, the device-side selected Gram gather; all off by default, and then every statement executed is v9c3c's
# H3 performance options. All off by default.
import argparse as _h3_argparse
import threading as _h3_threading

V9C4_VERSION = "engine_twofield_v9c4"
_H3 = {
    "band_gather": False,
    "anchor_workers": 0,
    "weight_gather": False,
}
_H3_TLS = _h3_threading.local()
_H3_POOL = None
_H3_STATS = {}
_H3_STATS_LOCK = _h3_threading.Lock()


def _h3_inc(k, n=1):
    # diagnostic counters only (no numerical effect); printed on the run's last v9c4 perf line when an option is on
    with _H3_STATS_LOCK:
        _H3_STATS[k] = _H3_STATS.get(k, 0) + n


class _H3Action(_h3_argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if self.dest == "anchor_workers":
            if not 0 <= values <= 20:
                raise _h3_argparse.ArgumentError(
                    self, "expected 0..20: 22-core quota, two cores reserved"
                )
            _H3[self.dest] = values
        else:
            _H3[self.dest] = True
        # Deliberately do not modify namespace: output args stay unchanged.


def _h3_add_arguments(ap):
    # the review's sec. 1 plumbing for the three options built here (A1 band_gather, C1 weight_gather, B1 anchor_workers); A2 to A4 not built
    for name in (
        "band_gather", "weight_gather",
    ):
        ap.add_argument(
            "--perf-h3-" + name.replace("_", "-"),
            dest=name, nargs=0, action=_H3Action,
            default=_h3_argparse.SUPPRESS,
            help="H3 bitwise-gated performance option; default off",
        )
    ap.add_argument(
        "--perf-h3-anchor-workers",
        dest="anchor_workers", type=int, action=_H3Action,
        default=_h3_argparse.SUPPRESS,
        help="H3 host anchor-test workers, 0..20; default 0/off",
    )


def h3_report():
    if _H3["band_gather"] or _H3["anchor_workers"] or _H3["weight_gather"]:
        print("v9c4 perf: %s; band_gather %s, anchor_workers %d, weight_gather %s (bitwise-gated; the options are kept out of the parsed namespace, so the json's args equal v9c3c's)" % (
            V9C4_VERSION, "on" if _H3["band_gather"] else "off", _H3["anchor_workers"], "on" if _H3["weight_gather"] else "off"), flush=True)


def h3_close():
    if (_H3["band_gather"] or _H3["anchor_workers"] or _H3["weight_gather"]) and _H3_STATS:
        print("  v9c4 perf (the run): %s" % ", ".join("%s %s" % (k, (("%.2f" % v) if isinstance(v, float) else v)) for k, v in sorted(_H3_STATS.items())), flush=True)


def _h3_band_gather(A):
    # Restrict the fast path to the actual double-complex CSR representation.
    if (getattr(A, "format", None) != "csr"
            or A.shape[0] != A.shape[1]
            or A.dtype != np.dtype(np.complex128)):
        return None

    n = A.shape[0]
    plan = getattr(_H3_TLS, "band_plan", None)
    hit = (
        plan is not None
        and plan[0] == A.shape
        and np.array_equal(plan[1], A.indptr)
        and np.array_equal(plan[2], A.indices)
    )
    _h3_inc("band_plan_hit" if hit else "band_plan_rebuild")

    if not hit:
        col = np.asarray(A.indices, dtype=np.intp)
        row = np.repeat(
            np.arange(n, dtype=np.intp), np.diff(A.indptr)
        )
        off = col - row
        u = int(off.max()) if off.size else 0
        if u < 0:
            return None

        src = np.flatnonzero(off >= 0)
        dst = (u - off[src]) * n + col[src]

        # Assignment/gather is valid only for unique upper-band destinations.
        if np.unique(dst).size != dst.size:
            return None

        # A.data.size is the index of an appended positive complex zero.
        gather = np.full((u + 1) * n, A.data.size, dtype=np.intp)
        gather[dst] = src
        gather = gather.reshape(u + 1, n)

        plan = (
            A.shape, A.indptr.copy(), A.indices.copy(), gather, u
        )
        _H3_TLS.band_plan = plan

    gather, u = plan[3], plan[4]
    data = np.empty(A.data.size + 1, dtype=np.complex128)
    data[:-1] = A.data
    data[-1] = 0.0 + 0.0j
    ab = data[gather]
    np.add(np.complex128(0.0 + 0.0j), ab, out=ab)
    ab[u, :] = ab[u, :].real
    return ab, u


def _h3_host_map(fn, jobs):
    global _H3_POOL

    p = _H3["anchor_workers"]
    _h3_inc("pool_maps"); _h3_inc("pool_jobs", len(jobs))
    t_h3 = time.time()
    if p <= 1 or len(jobs) <= 1:
        out_h3 = [fn(job) for job in jobs]
        _h3_inc("pool_serial_s", time.time() - t_h3)
        return out_h3

    from concurrent.futures import ThreadPoolExecutor, wait
    from threadpoolctl import threadpool_limits
    import atexit

    if _H3_POOL is None:
        _H3_POOL = ThreadPoolExecutor(
            max_workers=p, thread_name_prefix="h3-anchor"
        )
        atexit.register(_H3_POOL.shutdown, wait=True)

    # Called only by the serial coordinator; never nest this region.
    with threadpool_limits(limits=1, user_api="blas"):
        futures = [_H3_POOL.submit(fn, job) for job in jobs]
        # Finish all jobs before restoring native limits, including on error.
        wait(futures)
        _h3_inc("pool_parallel_maps"); _h3_inc("pool_parallel_s", time.time() - t_h3)
        # Exceptions and results are observed in the original job order.
        return [future.result() for future in futures]


def _h3_anchor_ladder(job):
    K0, dK, t1, d0, d_eff = job
    Ks = (K0 + t1 * dK).tocsr()
    ab, _u = band_upper_v36(Ks @ Ks)
    d_new = None
    d_try = d0
    nchol = 0

    while d_try >= d_eff:
        nchol += 1
        if pd_banded_v36(ab, d_try):
            d_new = d_try
            break
        d_try *= 0.5

    if d_new is not None and d_new == d0 and d_new * 2.0 < 1.0:
        nchol += 1
        if pd_banded_v36(ab, 2.0 * d_new):
            d_new = 2.0 * d_new

    return d_new, nchol


def _h3_anchor_reopen(job):
    K0, dK, t1, d_eff, slack, tol_block = job
    Ks = (K0 + t1 * dK).tocsr()
    ab, _u = band_upper_v36(Ks @ Ks)
    return pd_banded_v36(
        ab, max(2.0 * (d_eff + slack), 4.0 * float(tol_block))
    )


def _h3_gram_columns(A, B, order):
    import torch

    if not _v9c_is_torch(A):
        A = torch.as_tensor(
            _v9c_np.asarray(A), device=B.device, dtype=B.dtype
        )

    # Exact original GEMM and elementwise expressions, with original shapes.
    W = (A.conj().transpose(0, 1) @ B).abs() ** 2
    idx = torch.as_tensor(order, dtype=torch.int64, device=W.device)
    selected = W.index_select(1, idx)
    _h3_inc("gram_gathers"); _h3_inc("gram_d2h_bytes_avoided", 8 * int(W.shape[0]) * (int(W.shape[1]) - int(idx.numel())))

    # Match numpy's original advanced-column-indexing layout so subsequent
    # numpy reductions retain their traversal/reduction path.
    return _v9c_np.asfortranarray(
        selected.detach().cpu().numpy(), dtype=float
    )
# <<< v9c4 perf'''

OLD_BAND = 'def band_upper_v36(A):\n    """the upper band storage of a Hermitian host sparse matrix for scipy.linalg.cholesky_banded(lower=False): ab[u + i - j, j] = A[i, j]."""\n'
BAND_INS = '    if _H3["band_gather"]:\n        out = _h3_band_gather(A)\n        if out is not None:\n            return out\n        _h3_inc("band_fallback")\n'
OLD_ARGS = '    """the v9c arguments (the builder\'s R2 places this call before --cut-transport); every default reproduces v9b; the run\'s wrapper reads them off the parsed namespace"""  # v9c prim\n'
OLD_LADDER = '        for j in others:                                                                                 # v9 sector: the ladder at the probe point for the anchors in the step rule\n            Ks = (seg["K0"][j] + t1 * seg["dK"][j]).tocsr()                                              # v9 sector\n            ab, _u = band_upper_v36(Ks @ Ks)                                                             # v9 sector\n            d_new = None; d_try = seg["d"][j]                                                            # v9 sector\n            while d_try >= seg["d_eff"][j]:                                                              # v9 sector\n                nchol += 1                                                                               # v9 sector\n                if pd_banded_v36(ab, d_try):                                                             # v9 sector\n                    d_new = d_try                                                                        # v9 sector\n                    break                                                                                # v9 sector\n                d_try *= 0.5                                                                             # v9 sector\n            if d_new is None:                                                                            # v9 sector: the gap at the probe point below the floor: the anchor\'s crossing phase opens, s_lo the probe point certified before it\n                newly.append(j)                                                                          # v9 sector\n                continue                                                                                 # v9 sector\n            if d_new == seg["d"][j] and d_new * 2.0 < 1.0:                                               # v9 sector\n                nchol += 1                                                                               # v9 sector\n                if pd_banded_v36(ab, 2.0 * d_new):                                                       # v9 sector\n                    d_new = 2.0 * d_new                                                                  # v9 sector\n            new_d[j] = d_new                                                                             # v9 sector\n'
LADDER_HEAD = '        if _H3["anchor_workers"]:\n            jobs = [\n                (\n                    seg["K0"][j], seg["dK"][j], t1,\n                    seg["d"][j], seg["d_eff"][j],\n                )\n                for j in others\n            ]\n            results = _h3_host_map(_h3_anchor_ladder, jobs)\n            for j, (d_new, calls) in zip(others, results):\n                nchol += calls\n                if d_new is None:\n                    newly.append(j)\n                    continue\n                new_d[j] = d_new\n        else:\n'
OLD_REOPEN = '                for j in cand:                                                                           # v9 sector\n                    Ks = (seg["K0"][j] + t1 * seg["dK"][j]).tocsr()                                      # v9 sector\n                    ab, _u = band_upper_v36(Ks @ Ks); nchol += 1                                         # v9 sector\n                    if pd_banded_v36(ab, max(2.0 * (seg["d_eff"][j] + seg["slack"][j]), 4.0 * float(tol_block))):   # v9 sector: the nearest level at least twice the floor from the anchor: the pair may have reopened\n                        ready.append(j)                                                                  # v9 sector\n'
REOPEN_HEAD = '                if _H3["anchor_workers"]:\n                    jobs = [\n                        (\n                            seg["K0"][j], seg["dK"][j], t1,\n                            seg["d_eff"][j], seg["slack"][j], tol_block,\n                        )\n                        for j in cand\n                    ]\n                    results = _h3_host_map(_h3_anchor_reopen, jobs)\n                    for j, reopened in zip(cand, results):\n                        nchol += 1\n                        if reopened:\n                            ready.append(j)\n                else:\n'
OLD_W = '    if isinstance(B, V9cPlus):  # v9c prim\n        return _v9c_gram(A, B.U)[:, B.order]  # v9c prim\n'
W_INS = '        if (_H3["weight_gather"]\n                and _v9c_is_torch(B.U)\n                and str(B.U.device) != "cpu"):\n            return _h3_gram_columns(A, B.U, B.order)\n'


class BuildStop(Exception):
    pass


class Rep(object):
    def __init__(self, name, old, new):
        self.name = name; self.old = old; self.new = new


def mark(text):
    out = []
    for line in text.split("\n"):
        out.append(line if (line.strip() == "" or MARK in line) else line + "  # " + MARK)
    return "\n".join(out)


def indent4(t):
    return "\n".join(("    " + x) if x.strip() else x for x in t.rstrip("\n").split("\n")) + "\n"


def replacements():
    M = "  # " + MARK
    w_lines = OLD_W.split("\n")
    return [
        Rep("P1 block", "\n\n" + "def band_upper_v36(A):\n", "\n\n" + mark(BLOCK) + "\n\n\n" + "def band_upper_v36(A):\n"),
        Rep("A1 band", OLD_BAND, OLD_BAND + mark(BAND_INS.rstrip("\n")) + "\n"),
        Rep("P2 args", OLD_ARGS, OLD_ARGS + "    _h3_add_arguments(ap)" + M + "\n"),
        Rep("P3 report", "    v9c3_set(a)  # [v9c3 perf]\n", "    v9c3_set(a)  # [v9c3 perf]\n    h3_report()" + M + "\n"),
        Rep("P3 close", "        v9c3_close()  # [v9c3 perf]\n", "        h3_close()" + M + "\n        v9c3_close()  # [v9c3 perf]\n"),
        Rep("C1 weights", OLD_W, w_lines[0] + "\n" + mark(W_INS.rstrip("\n")) + "\n" + w_lines[1] + "\n"),
        Rep("B1 ladder", OLD_LADDER, mark(LADDER_HEAD.rstrip("\n")) + "\n" + mark(indent4(OLD_LADDER).rstrip("\n")) + "\n"),
        Rep("B1 reopen", OLD_REOPEN, mark(REOPEN_HEAD.rstrip("\n")) + "\n" + mark(indent4(OLD_REOPEN).rstrip("\n")) + "\n"),
    ]


def precheck(text, reps, reverse=False):
    return [(r, text.count(r.new if reverse else r.old)) for r in reps if text.count(r.new if reverse else r.old) != 1]


def apply(text, reps, reverse=False):
    bad = precheck(text, reps, reverse)
    if bad:
        raise BuildStop("; ".join("%s: %s occurs %d time(s), not once" % (r.name, "NEW" if reverse else "OLD", n) for r, n in bad))
    for r in (reversed(reps) if reverse else reps):
        old, new = (r.new, r.old) if reverse else (r.old, r.new)
        text = text.replace(old, new, 1)
    return text


def marks_check(src, out):
    old = set(src.split("\n")); bad = []
    for q, line in enumerate(out.split("\n"), 1):
        if line.strip() == "" or line in old:
            continue
        if MARK not in line:
            bad.append((q, line))
    return bad


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def write_new(path, text):
    if os.path.exists(path):
        raise BuildStop("the output %s exists; it is never overwritten" % path)
    with open(path, "xb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="engine_twofield_v9c4 over engine_twofield_v9c3c (exact replacements; fails closed)")
    ap.add_argument("source"); ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)
    src = read(a.source); reps = replacements()
    try:
        if a.reverse:
            back = apply(src, reps, reverse=True)
            out = a.out or (os.path.splitext(a.source)[0] + ".restored.py")
            write_new(out, back)
            print("reverse: %s (md5 %s) -> %s (md5 %s)" % (a.source, md5(src), out, md5(back)))
            if a.against:
                ok = read(a.against) == back
                print("against %s: %s" % (a.against, "BYTE FOR BYTE" if ok else "DIFFERS"))
                return 0 if ok else 2
            return 0
        if md5(src) != SOURCE_MD5:
            print("STOP: the source md5 is %s, not v9c3c's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r:" % (len(marks), MARK))
        for q, line in marks[:20]:
            print("  %6d: %s" % (q, line[:160]))
        return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v9c3c byte for byte"); return 2
    try:
        compile(out_text, "engine_twofield_v9c4.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c4 (md5 %s); %d replacements applied; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c4.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
