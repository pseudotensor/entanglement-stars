#!/usr/bin/env python
"""build_engine_collapse_v36nq_aw.py (busi, analyst; PERF H4(a), inbox #348): engine_collapse_v36nq_aw.py from engine_collapse_v36nq.py
(md5 8549c542..., pod6's ISO_128_s4 engine) by marked replacement blocks, in the form of build_engine_collapse_v36r.py (busi/code/iso).
Rules: the source's md5 must begin with 8549c542 (--expect-src-md5); every OLD anchor is copied verbatim from v36nq and occurs exactly once
(checked); every line a NEW block adds or changes carries "v36aw slice" (checked); the reverse pass (NEW -> OLD, each NEW once) restores v36nq
byte for byte (checked in memory before writing and again on the written file; available as --reverse); the output compiles (py_compile).
    python build_engine_collapse_v36nq_aw.py [--src .../engine_collapse_v36nq.py] [--dst .../engine_collapse_v36nq_aw.py] [--check-only]
    python build_engine_collapse_v36nq_aw.py --reverse --src .../engine_collapse_v36nq_aw.py --dst .../restored.py --expect .../engine_collapse_v36nq.py

The write path (v36nq, run(), the record loop after the handoff block): at a record whose T/rs is in --slices-c128/--slices-c64 (with
--slice-dir), the dict arrs is built (the sea's, the dust's and R2's columns per channel by .cpu().numpy().astype(dtype), the eigenvalues,
f, the packed mask, the applied fields, R, the weights) and written by np.savez(saved, **arrs) on the main thread, single threaded; the
GPU idles for the copies and the write (pod6 ISO_128_s4: 10.7 GB per complex128 slice to the network volume).
The patch (default off: with --slice-async off, the default, the slice is written exactly as v36nq writes it; the one addition is a log
line with that write's wall; the records, fields, spectra, handoff and the slice are v36nq's bitwise):
  A  the module docstring's lineage line;
  B  the helpers before run(): SLW_V36AW (the one write in flight), slice_gib_v36aw, slice_join_v36aw (joins the write in flight and raises
     its error: the run stops), slice_worker_v36aw (np.savez into ".partial_<name>" in the slice dir, then os.replace onto the slice's name:
     the slice's name only ever holds a complete file), slice_submit_v36aw (the small arrays copied; the big columns are already fresh
     host copies made by astype; the thread started, non-daemon);
  C  the slice block's head: with the option on, the previous write joined before the next slice's arrays are built (one slice in host
     memory at a time); the wall's clock started;
  D  the write: with the option on, slice_submit_v36aw; off, v36nq's np.savez and the wall line;
  E  the run's end (after the final json and save_fields): with the option on, the last write joined (its error raised);
  F  the option --slice-async off | on (after --slice-dir).
"""
import argparse
import hashlib
import os
import py_compile

MARK = "v36aw slice"
SRC_MD5_PREFIX = "8549c542"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

# A: the lineage (the first line of the module docstring)
EDITS.append(("A lineage", blk(r'''
"""engine_collapse_v36nq.py (busi, reviewer A; ISO_ruling_v1): version 36nq = version 36q with options, each off by default, so that the defaults run'''), blk(r'''
"""engine_collapse_v36nq_aw.py (busi, analyst; PERF H4(a), #348; built by build_engine_collapse_v36nq_aw.py over v36nq, md5 8549c542) [v36aw slice]
version 36nq_aw = version 36nq with one option, off by default (off: v36nq bitwise; one log line per slice with its write's wall): [v36aw slice]
--slice-async on: the slice (--slices-c128/--slices-c64 into --slice-dir) is built on the main thread as before, its small arrays copied, and [v36aw slice]
    np.savez runs in a background thread into ".partial_<name>" beside it, renamed onto the slice's name when complete (os.replace), so the [v36aw slice]
    bytes are v36nq's and the slice's name only ever holds a complete file; one write in flight: the next slice point and the run's end join [v36aw slice]
    it first; a write error is raised at the join and the run stops. Version 36nq's docstring follows. [v36aw slice]
engine_collapse_v36nq.py (busi, reviewer A; ISO_ruling_v1): version 36nq = version 36q with options, each off by default, so that the defaults run''')))

# B: the helpers, before run()
EDITS.append(("B helpers", blk(r'''
def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)'''), blk(r'''
SLW_V36AW = {"thread": None, "error": None, "path": None}   # v36aw slice: the one background slice write in flight


def slice_gib_v36aw(arrs):   # v36aw slice: the slice's array bytes in GiB
    return sum(int(np.asarray(v_).nbytes) for v_ in arrs.values()) / 2.0 ** 30   # v36aw slice


def slice_join_v36aw():   # v36aw slice: join the write in flight (if any); its error is raised here and the run stops
    th = SLW_V36AW["thread"]   # v36aw slice
    if th is None:   # v36aw slice
        return   # v36aw slice
    t0 = time.time()   # v36aw slice
    th.join()   # v36aw slice
    SLW_V36AW["thread"] = None   # v36aw slice
    w_ = time.time() - t0   # v36aw slice
    if w_ >= 0.1:   # v36aw slice
        print("  v36aw slice: waited %.1f s for the previous background write (%s)" % (w_, os.path.basename(SLW_V36AW["path"] or "")), flush=True)   # v36aw slice
    err = SLW_V36AW["error"]   # v36aw slice
    SLW_V36AW["error"] = None   # v36aw slice
    if err is not None:   # v36aw slice
        raise RuntimeError("v36aw slice: the background write of %s failed (%r); the run stops" % (SLW_V36AW["path"], err)) from err   # v36aw slice


def slice_worker_v36aw(path, tmp, arrs, T_rs, gib, t_main):   # v36aw slice: the thread's body: np.savez (v36nq's call) into tmp, then onto path
    t0 = time.time()   # v36aw slice
    try:   # v36aw slice
        np.savez(tmp, **arrs)   # v36aw slice
        os.replace(tmp, path)   # v36aw slice
        print("  v36aw slice: T/rs %.2f written in %.1f s in the background (%.2f GiB; the host copies %.1f s on the main thread)" % (T_rs, time.time() - t0, gib, t_main), flush=True)   # v36aw slice
    except BaseException as e_:   # v36aw slice
        SLW_V36AW["error"] = e_   # v36aw slice
        print("  v36aw slice: the background write of %s FAILED after %.1f s: %r (raised at the next join)" % (path, time.time() - t0, e_), flush=True)   # v36aw slice
    finally:   # v36aw slice
        arrs.clear()   # v36aw slice: the host copies released


def slice_submit_v36aw(path, arrs, T_rs, t_build):   # v36aw slice: the write handed to a background thread (the caller joined the previous one)
    import threading as _th   # v36aw slice
    t0 = time.time()   # v36aw slice
    own = {k_: (np.array(v_, copy=True) if (isinstance(v_, np.ndarray) and not k_.startswith(("sea_", "dust_", "R2_"))) else v_) for k_, v_ in arrs.items()}   # v36aw slice: the small arrays copied (the columns are astype's fresh copies)
    gib = slice_gib_v36aw(own)   # v36aw slice
    tmp = os.path.join(os.path.dirname(path), ".partial_" + os.path.basename(path))   # v36aw slice: ends in .npz, so np.savez keeps the name
    SLW_V36AW["path"] = path; SLW_V36AW["error"] = None   # v36aw slice
    th = _th.Thread(target=slice_worker_v36aw, args=(path, tmp, own, T_rs, gib, t_build + time.time() - t0), name="v36aw_slice", daemon=False)   # v36aw slice: non-daemon: an exception elsewhere still lets the write finish
    SLW_V36AW["thread"] = th   # v36aw slice
    th.start()   # v36aw slice


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)''')))

# C: the slice block's head: the previous write joined first (option on), the wall's clock
EDITS.append(("C slice head", blk(r'''
        if tslice and a.slice_dir:
            dt_ = np.complex128 if any(abs(T / rs - x) < 1e-6 for x in sl128) else np.complex64'''), blk(r'''
        if tslice and a.slice_dir:
            if getattr(a, "slice_async", "off") == "on":   # v36aw slice: one write in flight: the previous one joined before this slice's arrays are built
                slice_join_v36aw()   # v36aw slice
            t_sw = time.time()   # v36aw slice: the slice's wall (the host copies and the write)
            dt_ = np.complex128 if any(abs(T / rs - x) < 1e-6 for x in sl128) else np.complex64''')))

# D: the write
EDITS.append(("D write", blk(r'''
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(saved, **arrs)
            del arrs'''), blk(r'''
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            if getattr(a, "slice_async", "off") == "on":   # v36aw slice
                slice_submit_v36aw(saved, arrs, T / rs, time.time() - t_sw)   # v36aw slice: written in the background (the record's slice_saved names the same path)
            else:   # v36aw slice
                np.savez(saved, **arrs)   # v36aw slice: v36nq's write
                print("  v36aw slice: T/rs %.2f written in %.1f s (%.2f GiB; synchronous: the host copies and np.savez)" % (T / rs, time.time() - t_sw, slice_gib_v36aw(arrs)), flush=True)   # v36aw slice
            del arrs''')))

# E: the run's end
EDITS.append(("E run end", blk(r'''
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
    save_fields()'''), blk(r'''
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
    save_fields()
    if getattr(a, "slice_async", "off") == "on":   # v36aw slice: the run's end joins the last background write (its error raised)
        slice_join_v36aw()   # v36aw slice''')))

# F: the option
EDITS.append(("F option", blk(r'''
    ap.add_argument("--slice-dir", default=None)'''), blk(r'''
    ap.add_argument("--slice-dir", default=None)
    ap.add_argument("--slice-async", dest="slice_async", default="off", choices=["off", "on"], help="(v36aw) on: the slice is written by np.savez in a background thread after the host copies (one in flight; joined at the next slice and at the end; an error stops the run), into .partial_<name> then renamed; off: v36nq's write; bitwise either way")   # v36aw slice''')))


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
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}
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
    ap.add_argument("--src", default="engine_collapse_v36nq.py")
    ap.add_argument("--dst", default=None, help="default: engine_collapse_v36nq_aw.py beside --src")
    ap.add_argument("--reverse", action="store_true", help="src is engine_collapse_v36nq_aw.py: restore v36nq into dst")
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36nq_aw.py" if not a.reverse else "engine_collapse_v36nq_restored.py")
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
        raise SystemExit("the source's md5 does not start with %s: not engine_collapse_v36nq.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-22s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
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
