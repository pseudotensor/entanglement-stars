#!/usr/bin/env python
"""build_engine_banded_eigh_v2.py (busi, analyst; PERF RD-1d, the memory-safe dense fallback; busi/notes/PERF_RD1d_build_v1.md):
engine_banded_eigh_v2.py from engine_banded_eigh_v1.py (md5 73ee46fa..., RD-1, ADOPTED) by marked replacement blocks, in the form of
build_engine_collapse_v36nq_rd13mfb.py (busi/code/gatecomp/flat).
Rules (checked): the source's md5 begins with 73ee46fa (--expect-src-md5); every OLD anchor is copied verbatim from the source and occurs
exactly once; every line a NEW block adds or changes carries "rd1d memsafe"; the reverse pass (NEW -> OLD, each NEW once) restores the
source byte for byte (in memory before writing and again on the written file; available as --reverse); the output compiles; nothing is
overwritten.
    python build_engine_banded_eigh_v2.py --src .../engine_banded_eigh_v1.py [--dst .../engine_banded_eigh_v2.py] [--check-only]
    python build_engine_banded_eigh_v2.py --reverse --src .../engine_banded_eigh_v2.py --dst .../restored.py --expect .../engine_banded_eigh_v1.py

What changes (only the dense route's out-of-memory case; every call that does not raise torch.cuda.OutOfMemoryError in the dense
torch.linalg.eigh runs v1's statements unchanged, so the banded route, the guards and a dense call that fits are v1's bitwise; STATS gains
no key unless the new path acted):
  the dense route _dense(hmT): torch.linalg.eigh(hmT) as v1; on torch.cuda.OutOfMemoryError (i) synchronize, collect Python's unreachable
  objects (gc.collect: tensors held only by reference cycles are freed, never parked), empty the caching allocator's free blocks and call
  it again; (ii) if that fails too, PARK: the device's other resident tensors (every strided CUDA tensor on hmT's device
  reachable from Python's gc, by storage, at least 64 MiB, resizable, not hmT's storage) are copied to host memory byte for byte and their
  device storage released (UntypedStorage.resize_(0), as FSDP frees parameters), largest first, one at a time, the same call retried after
  each, until it succeeds; then every parked storage is reallocated and copied back byte for byte (every tensor and view keeps its size,
  strides, offset and values; only the device address changes), before _dense returns. The call that succeeds is the same
  torch.linalg.eigh on the same input on the same device, so the result is v1's dense result (bitwise in the self-test). If nothing is left
  to park, the original error is raised (v1's outcome) with every parked storage restored. If the result and the residents cannot be kept
  together, the result is dropped, every storage restored (gc.collect and empty_cache, two attempts) and the OOM raised; a storage that still
  cannot come back raises MemsafeRestoreFailed, a BaseException, so that no engine handler (they catch Exception) continues with an emptied
  tensor.
  Counters in MEMSAFE (events, retries, tensors and bytes parked, seconds), one log line per event, and a "memsafe" entry in
  stats_snapshot only when an event happened.
"""
import argparse
import hashlib
import os
import py_compile

MARK = "rd1d memsafe"
SRC_MD5_PREFIX = "73ee46fa"
SRC_NAME = "engine_banded_eigh_v1.py"
OUT_NAME = "engine_banded_eigh_v2.py"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

EDITS.append(("A docstring", blk(r'''
#!/usr/bin/env python
"""engine_banded_eigh_v1.py (busi, analyst; PERF RD-1 + RD-3, REVIEWA_algorithm_redesign_v1 sec. 1 and 3): the production module of the banded
'''), blk(r'''
#!/usr/bin/env python
"""engine_banded_eigh_v2.py (busi, analyst; PERF RD-1d; built by build_engine_banded_eigh_v2.py over engine_banded_eigh_v1, md5 73ee46fa) [rd1d memsafe]
v2 = v1 with a memory-safe dense route: _dense(hmT) runs v1's torch.linalg.eigh(hmT); only when that raises torch.cuda.OutOfMemoryError [rd1d memsafe]
does it (i) empty the caching allocator and retry, then (ii) park the device's other resident tensors (strided CUDA tensors reachable [rd1d memsafe]
from gc, by storage, >= 64 MiB, largest first) on the host byte for byte, retry after each, and restore every parked storage byte for [rd1d memsafe]
byte before returning (the same call on the same input: v1's dense result). Nothing else changes; MEMSAFE counts the events, and [rd1d memsafe]
stats_snapshot carries "memsafe" only when one happened. The flat member F256_s4's OOM (22 Oct) is the case. v1's docstring follows. [rd1d memsafe]
engine_banded_eigh_v1.py (busi, analyst; PERF RD-1 + RD-3, REVIEWA_algorithm_redesign_v1 sec. 1 and 3): the production module of the banded
''')))

EDITS.append(("B version", blk(r'''
VERSION = "engine_banded_eigh_v1"
'''), blk(r'''
VERSION = "engine_banded_eigh_v2"   # rd1d memsafe
''')))

EDITS.append(("C dense", blk(r'''
def _dense(hmT):
    e, U = torch.linalg.eigh(hmT)                                    # engine_collapse_v5.eigT's code
    return e.real, U
'''), blk(r'''
MEMSAFE = {"events": 0, "ok_after_empty_cache": 0, "ok_after_park": 0, "failed": 0, "retries": 0, "tensors_parked": 0, "bytes_parked": 0,   # rd1d memsafe
           "park_s": 0.0, "restore_s": 0.0, "seconds": 0.0, "max_parked_bytes_one_call": 0}   # rd1d memsafe
PARK_MIN_BYTES = 64 << 20   # rd1d memsafe: storages below 64 MiB are never parked


class MemsafeRestoreFailed(BaseException):   # rd1d memsafe: a parked storage that cannot come back stops the process (not catchable by except Exception)
    pass   # rd1d memsafe


def _park_candidates(hmT):   # rd1d memsafe: the device storages that may leave: (nbytes, storage), largest first
    import gc   # rd1d memsafe
    dev = hmT.device   # rd1d memsafe
    keep = hmT.untyped_storage().data_ptr()   # rd1d memsafe
    seen = {}   # rd1d memsafe
    for o in gc.get_objects():   # rd1d memsafe
        try:   # rd1d memsafe
            if not isinstance(o, torch.Tensor) or o.layout != torch.strided or o.device != dev:   # rd1d memsafe
                continue   # rd1d memsafe
            s = o.untyped_storage()   # rd1d memsafe
            p = s.data_ptr(); nb = s.nbytes()   # rd1d memsafe
        except Exception:   # rd1d memsafe: a tensor without storage (sparse, meta, functional): never parked
            continue   # rd1d memsafe
        if p == 0 or p == keep or nb < PARK_MIN_BYTES or p in seen or not s.resizable():   # rd1d memsafe
            continue   # rd1d memsafe
        seen[p] = (nb, s)   # rd1d memsafe
    return sorted(seen.values(), key=lambda q: -q[0])   # rd1d memsafe


def _park(s, nb):   # rd1d memsafe: the storage's bytes to a pageable host buffer, then its device memory released
    h = torch.empty(nb, dtype=torch.uint8, device="cpu")   # rd1d memsafe
    v = torch.empty(0, dtype=torch.uint8, device=s.device).set_(s, 0, (nb,), (1,))   # rd1d memsafe: a byte view of the whole storage
    h.copy_(v)   # rd1d memsafe: synchronous (pageable)
    del v   # rd1d memsafe
    s.resize_(0)   # rd1d memsafe
    return h   # rd1d memsafe


def _unpark(s, h):   # rd1d memsafe: the storage reallocated at its size and its bytes copied back
    nb = h.numel()   # rd1d memsafe
    s.resize_(nb)   # rd1d memsafe
    v = torch.empty(0, dtype=torch.uint8, device=s.device).set_(s, 0, (nb,), (1,))   # rd1d memsafe
    v.copy_(h)   # rd1d memsafe
    del v   # rd1d memsafe


def _dense_memsafe(hmT, err, log=None):   # rd1d memsafe: v1's dense call after empty_cache, then with the resident storages parked one by one
    t0 = time.time()   # rd1d memsafe
    dev = hmT.device   # rd1d memsafe
    MEMSAFE["events"] += 1   # rd1d memsafe
    free0 = _free_bytes(dev)   # rd1d memsafe
    import gc   # rd1d memsafe
    torch.cuda.synchronize(dev); gc.collect(); torch.cuda.empty_cache()   # rd1d memsafe: unreachable tensors (reference cycles) freed first, never parked
    MEMSAFE["retries"] += 1   # rd1d memsafe
    try:   # rd1d memsafe
        e, U = torch.linalg.eigh(hmT)   # rd1d memsafe: engine_collapse_v5.eigT's code
        MEMSAFE["ok_after_empty_cache"] += 1; MEMSAFE["seconds"] += time.time() - t0   # rd1d memsafe
        if log is not None:   # rd1d memsafe
            log("%s: dense eigh out of memory (%.2f GiB free): done after emptying the allocator's cache" % (VERSION, free0 / 2**30))   # rd1d memsafe
        return e, U   # rd1d memsafe
    except torch.cuda.OutOfMemoryError as ex2:   # rd1d memsafe
        err = ex2   # rd1d memsafe
    parked = []   # rd1d memsafe
    out = None   # rd1d memsafe
    cands = _park_candidates(hmT)   # rd1d memsafe
    try:   # rd1d memsafe
        for nb, s in cands:   # rd1d memsafe
            t_p = time.time()   # rd1d memsafe
            torch.cuda.synchronize(dev)   # rd1d memsafe: no queued work reads the storage while it leaves
            parked.append((s, _park(s, nb)))   # rd1d memsafe
            torch.cuda.empty_cache()   # rd1d memsafe
            MEMSAFE["park_s"] += time.time() - t_p   # rd1d memsafe
            MEMSAFE["retries"] += 1   # rd1d memsafe
            try:   # rd1d memsafe
                out = torch.linalg.eigh(hmT)   # rd1d memsafe: engine_collapse_v5.eigT's code
                break   # rd1d memsafe
            except torch.cuda.OutOfMemoryError as ex3:   # rd1d memsafe
                err = ex3   # rd1d memsafe
    finally:   # rd1d memsafe: every parked storage back, byte for byte, before anything else runs
        t_r = time.time()   # rd1d memsafe
        torch.cuda.synchronize(dev)   # rd1d memsafe
        bad = None   # rd1d memsafe
        for s, h in reversed(parked):   # rd1d memsafe
            try:   # rd1d memsafe
                _unpark(s, h)   # rd1d memsafe
            except torch.cuda.OutOfMemoryError as ex4:   # rd1d memsafe: the result and the residents do not fit together
                bad = ex4   # rd1d memsafe
                break   # rd1d memsafe
        if bad is not None:   # rd1d memsafe: give the result's memory back and restore the rest; then v1's outcome (the error)
            out = None   # rd1d memsafe
            for attempt in range(2):   # rd1d memsafe
                gc.collect(); torch.cuda.empty_cache()   # rd1d memsafe
                left = 0   # rd1d memsafe
                for s, h in reversed(parked):   # rd1d memsafe
                    if s.nbytes() == 0:   # rd1d memsafe
                        try:   # rd1d memsafe
                            _unpark(s, h)   # rd1d memsafe
                        except torch.cuda.OutOfMemoryError:   # rd1d memsafe
                            left += 1   # rd1d memsafe
                if left == 0:   # rd1d memsafe
                    break   # rd1d memsafe
            if left:   # rd1d memsafe: never continue with a resident tensor emptied: a BaseException, which the engine's broad handlers do not catch
                raise MemsafeRestoreFailed("%s: %d parked storages could not be restored; the process stops" % (VERSION, left))   # rd1d memsafe
            err = bad   # rd1d memsafe
        torch.cuda.synchronize(dev)   # rd1d memsafe
        MEMSAFE["restore_s"] += time.time() - t_r   # rd1d memsafe
    nbp = sum(h.numel() for _s, h in parked); npk = len(parked)   # rd1d memsafe
    MEMSAFE["tensors_parked"] += len(parked); MEMSAFE["bytes_parked"] += nbp   # rd1d memsafe
    MEMSAFE["max_parked_bytes_one_call"] = max(MEMSAFE["max_parked_bytes_one_call"], nbp)   # rd1d memsafe
    MEMSAFE["seconds"] += time.time() - t0   # rd1d memsafe
    del parked   # rd1d memsafe
    if out is None:   # rd1d memsafe
        MEMSAFE["failed"] += 1   # rd1d memsafe
        if log is not None:   # rd1d memsafe
            log("%s: dense eigh out of memory (%.2f GiB free): %d of %d storages parked (%.2f GiB) did not make room; restored; raising" % (   # rd1d memsafe
                VERSION, free0 / 2**30, npk, len(cands), nbp / 2**30))   # rd1d memsafe
        raise err   # rd1d memsafe
    MEMSAFE["ok_after_park"] += 1   # rd1d memsafe
    if log is not None:   # rd1d memsafe
        log("%s: dense eigh out of memory (%.2f GiB free): done with %.2f GiB of resident storages parked on the host and restored (%.1f s)" % (   # rd1d memsafe
            VERSION, free0 / 2**30, nbp / 2**30, time.time() - t0))   # rd1d memsafe
    return out   # rd1d memsafe


def _dense(hmT, log=None):   # rd1d memsafe
    try:   # rd1d memsafe
        e, U = torch.linalg.eigh(hmT)                                    # engine_collapse_v5.eigT's code [rd1d memsafe]
    except torch.cuda.OutOfMemoryError as ex:   # rd1d memsafe: only then the memory-safe path
        e, U = _dense_memsafe(hmT, ex, log)   # rd1d memsafe
    return e.real, U
''')))

EDITS.append(("D call", blk(r'''
    out = _dense(hmT)
    STATS["seconds"] += time.time() - t0
    return out
'''), blk(r'''
    out = _dense(hmT, log)   # rd1d memsafe
    STATS["seconds"] += time.time() - t0
    return out
''')))

EDITS.append(("E log name", blk(r'''
        log("engine_banded_eigh_v1: the dense route for this call (%s)" % why)
'''), blk(r'''
        log("%s: the dense route for this call (%s)" % (VERSION, why))   # rd1d memsafe
''')))

EDITS.append(("F snapshot", blk(r'''
    snap = {k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()}
    if reset:
'''), blk(r'''
    snap = {k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()}
    if MEMSAFE["events"]:   # rd1d memsafe: only when the memory-safe path acted
        snap["memsafe"] = dict(MEMSAFE)   # rd1d memsafe
        if reset:   # rd1d memsafe
            for k in MEMSAFE:   # rd1d memsafe
                MEMSAFE[k] = 0.0 if isinstance(MEMSAFE[k], float) else 0   # rd1d memsafe
    if reset:
''')))


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text):
    for name, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100]))
        text = text.replace(old, new)
    return text


def reverse(text):
    for name, old, new in reversed(EDITS):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check():
    """every line a NEW block adds or changes carries MARK; a line of OLD is not new (nor is one moved out from behind the opening triple
    quote); blank lines are neutral."""
    bad, n_marked = [], 0
    for name, old, new in EDITS:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}   # the parent's first docstring line moved out from behind the quotes
        for ln in new.splitlines():
            if ln in olds or ln.strip() == "":
                continue
            if MARK in ln:
                n_marked += 1
            else:
                bad.append((name, ln))
    return bad, n_marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_NAME)
    ap.add_argument("--dst", default=None, help="default: %s beside --src" % OUT_NAME)
    ap.add_argument("--reverse", action="store_true", help="src is %s: restore %s into dst" % (OUT_NAME, SRC_NAME))
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), OUT_NAME if not a.reverse else "engine_banded_eigh_v1_restored.py")
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text).encode("utf-8")
        if not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if a.check_only else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not %s; nothing written" % (a.expect_src_md5, SRC_NAME))
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-14s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_marked = marker_check()
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln))
        raise SystemExit("new lines without the marker")
    print("every new or changed line carries %r (%d marked lines)" % (MARK, n_marked))
    built = forward(text)
    out = built.encode("utf-8")
    back = reverse(built).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst, "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    if os.path.exists(a.dst):
        if open(a.dst, "rb").read() == out:
            print("%s exists and equals the build byte for byte (md5 %s); not rewritten" % (a.dst, md5(out)))
            return
        raise SystemExit("%s exists and differs from the build; nothing written (never overwritten)" % a.dst)
    open(a.dst, "wb").write(out)
    written = open(a.dst, "rb").read()
    if written != out or reverse(written.decode("utf-8")).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    py_compile.compile(a.dst, doraise=True)
    n_lines = sum(1 for ln in built.splitlines() if MARK in ln)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))
    print("check: grep -c '%s' %s  ->  %d" % (MARK, a.dst, n_lines))


if __name__ == "__main__":
    main()
