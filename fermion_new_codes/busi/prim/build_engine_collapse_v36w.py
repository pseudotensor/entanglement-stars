#!/usr/bin/env python
"""build_engine_collapse_v36w.py (busi lane PERF build, owner order #299; final4/PERF/PERF_TRACKER.md row W2-P3; deliveries/lead_gaps/REVIEWA_perf_w2_v1.md
sec. 2 P3): writes engine_collapse_v36w.py from engine_collapse_v36u.py (md5 062d7dd4 = fixv4c b623d3b0 + W2-P1/P2) by exact string replacements in
build_engine_collapse_v36u.py's form. (v36v is reserved elsewhere; this build is v36w.) Each OLD is copied exactly from engine_collapse_v36u.py and
must occur exactly once (else the build aborts); each NEW contains the OLD's text plus inserted lines, so replacing every NEW by its OLD restores
v36u byte for byte (checked here). Every new line carries the marker "v36w perf" (checked here).

W2-P3: `--streams 1` made safe, no new option (the json's "args" dict is v36u's). Two W10 fixes inlined into eigh_all's threaded branch, which the
default (--streams 0) and every CPU run never enter, so the defaults are v36u (and fixv4c) bitwise:
  (1) the W10 warm-up (W10_warmup_block_v1b): one torch.linalg.eigh of the 2 x 2 identity on the device, in the first channel matrix's dtype, on the
      calling thread, once per device per process, result discarded, before the channel threads start (the cold-process race in PyTorch's lazy
      CUDA linear-algebra loader, "lazy wrapper should be called at most once": 4 of 8 cold A6000 launches failed without it, 8 of 8 passed with it);
  (2) the W10 re-raise (w10_lib_v1.run_threads_reraise): every channel thread joined; a thread's exception is no longer swallowed (which left the
      output dict partial while TM["eigh_n"] counted every channel). One deliberate difference from w10_lib's RuntimeError wrapper, following
      reviewer A P3's own requirement ("a cusolver failure in a thread must surface as the same exception the serial path raises"): the exception
      object of the first failed channel in the channels' order (the one the serial loop would have hit first) is re-raised unchanged, so the
      callers' handlers ("stopped: %s" % str(ex)[:120]) write the same text as on the serial path; the failed and missing channels are printed in
      one "v36w perf:" line first. An output lacking a channel although no thread raised is a RuntimeError (no serial analogue).
Usage: python build_engine_collapse_v36w.py [engine_collapse_v36u.py path] [engine_collapse_v36w.py path]   (defaults: beside this script)."""
import hashlib
import os
import py_compile
import sys

MARK = "v36w perf"
SRC_MD5 = "062d7dd4700b839470fe3c5b70143740"

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36u.py (busi PERF build; REVIEWA_perf_w2_v1 P1 and P2; built by build_engine_collapse_v36u.py over engine_collapse_v36p_fixv4c.py md5 b623d3b0) [v36u perf]
''',
r'''"""engine_collapse_v36w.py (busi PERF build; REVIEWA_perf_w2_v1 P3; built by build_engine_collapse_v36w.py over engine_collapse_v36u.py md5 062d7dd4) [v36w perf]
version 36w = version 36u with `--streams 1` made safe and no new option (the json's "args" is v36u's): eigh_all's threaded branch runs the W10 [v36w perf]
warm-up (one 2 x 2 eigh on the device on the calling thread, once per device per process, result discarded) before its channel threads start, and [v36w perf]
joins them through run_threads_reraise (a thread's exception re-raised after the join, the first failed channel's in the channels' order, unchanged, [v36w perf]
as the serial loop would raise it; an output lacking a channel is an error). The default (--streams 0) and every CPU run never enter that branch, [v36w perf]
so they are v36u (and fixv4c) bitwise; with --streams 1 every channel's eigh is the same cusolver call on the same matrix as before. [v36w perf]
[v36w perf: v36u's docstring follows] engine_collapse_v36u.py (busi PERF build; REVIEWA_perf_w2_v1 P1 and P2; built by build_engine_collapse_v36u.py over engine_collapse_v36p_fixv4c.py md5 b623d3b0) [v36u perf]
'''),
("the warm-up and the re-raise helpers (before eigT_t)",
r'''def eigT_t(hmT):
''',
r'''EIGH_WARMED = set()   # v36w perf (W2-P3, W10 close): the (device type, index) keys on which the one-time main-thread eigh warm-up has run in this process


def eigh_warm_up(like):   # v36w perf (W10_warmup_block_v1b's w10_eigh_warm_up)
    """v36w perf (W2-P3; W10 close, reading B): PyTorch loads its CUDA linear-algebra library at the first call in a process through stubs that admit one first caller ("lazy wrapper should be called at most once"); eigh_all's threaded branch makes that first call from several threads at once in a cold process (4 of 8 cold A6000 launches failed without this, 8 of 8 passed with it). One eigh of the 2 x 2 identity on the device and in the dtype of `like` (the first channel's matrix), on the calling thread, once per device per process; the device synchronized and the result discarded; no counter, record field, channel stream or random state touched. Returns True when it ran."""   # v36w perf
    torch = BK["torch"]; dev = getattr(like, "device", None)   # v36w perf
    if dev is None or dev.type != "cuda" or (dev.type, dev.index) in EIGH_WARMED:   # v36w perf
        return False   # v36w perf
    torch.linalg.eigh(torch.eye(2, dtype=like.dtype, device=dev)); torch.cuda.synchronize(dev); EIGH_WARMED.add((dev.type, dev.index))   # v36w perf
    return True   # v36w perf


def run_threads_reraise(jobs, out, where="threads"):   # v36w perf (w10_lib_v1.run_threads_reraise, with the serial path's exception)
    """v36w perf (W2-P3; W10 defect 3): jobs = {key: callable()} in the channels' order, each callable writing out[key]; every job in its own thread, all joined (threading.Thread swallows a target's exception, which left eigh_all's output partial and its count wrong). Then, if any job raised, one "v36w perf:" line names the failed and missing keys and the exception object of the first failed key in the jobs' order is re-raised unchanged (the serial loop would have raised that one first, so the callers' "stopped: ..." texts match the serial path's); a key missing although no job raised is a RuntimeError. Returns the number of completed keys."""   # v36w perf
    errors = {}   # v36w perf

    def wrap(k, fn):   # v36w perf
        try:   # v36w perf
            fn()   # v36w perf
        except BaseException as ex:  # noqa: BLE001 (re-raised after the join)   # v36w perf
            errors[k] = ex   # v36w perf
    th = [threading.Thread(target=wrap, args=(k, fn)) for k, fn in jobs.items()]   # v36w perf
    for t in th:   # v36w perf
        t.start()   # v36w perf
    for t in th:   # v36w perf
        t.join()   # v36w perf
    missing = [k for k in jobs if k not in out]   # v36w perf
    if errors:   # v36w perf
        k0 = next(k for k in jobs if k in errors)   # v36w perf: the first failed key in the jobs' (the channels') order
        print("v36w perf: %s: %d of %d threads failed (channels %s; the output lacks %s); re-raising channel %s's %s: %s" % (where, len(errors), len(jobs), sorted(errors, key=str), sorted(missing, key=str), k0, type(errors[k0]).__name__, str(errors[k0])[:300]), flush=True)   # v36w perf
        raise errors[k0]   # v36w perf
    if missing:   # v36w perf
        raise RuntimeError("%s: the output lacks the channels %s although no thread raised" % (where, sorted(missing, key=str)))   # v36w perf
    return len(out)   # v36w perf


def eigT_t(hmT):
'''),
("eigh_all's threaded branch: the warm-up and the re-raise",
r'''    th = [threading.Thread(target=work, args=(ch.kappa,)) for ch, _k in todo]   # v36u perf
    for t in th:
        t.start()
    for t in th:
        t.join()
    sync_dev()
    hms.clear()
''',
r'''    eigh_warm_up(hms[todo[0][0].kappa])   # v36w perf (W2-P3): the W10 warm-up on this thread before the channel threads start (once per device per process)
    run_threads_reraise({ch.kappa: (lambda k=ch.kappa: work(k)) for ch, _k in todo}, out, "eigh_all --streams 1")   # v36w perf (W2-P3): replaces v36u's bare start/join, which swallowed a thread's exception
    sync_dev()
    hms.clear()
'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36w: the OLD string of '%s' occurs %d times in the source (1 required): the source is not engine_collapse_v36u.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36w: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36u.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36w.py")
    raw = open(src, "rb").read()
    if md5(raw) != SRC_MD5:
        raise SystemExit("build_engine_collapse_v36w: the source md5 is %s, not v36u's %s" % (md5(raw), SRC_MD5))
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text or md5(back.encode("utf-8")) != SRC_MD5:
        raise SystemExit("build_engine_collapse_v36w: the reverse replacements do not restore v36u byte for byte")
    srcset = set(text.splitlines())
    unmarked = [l_ for l_ in out.splitlines() if l_ not in srcset and l_.strip() and MARK not in l_]
    if unmarked:
        raise SystemExit("build_engine_collapse_v36w: %d new lines without the marker, the first: %r" % (len(unmarked), unmarked[0][:200]))
    marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines, every new non-blank line marked; source md5 %s; built md5 %s; the reverse replacements restore v36u byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
