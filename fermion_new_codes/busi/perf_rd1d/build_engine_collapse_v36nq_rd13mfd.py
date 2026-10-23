#!/usr/bin/env python
"""build_engine_collapse_v36nq_rd13mfd.py (busi, analyst; PERF RD-1d, the memory-safe dense fallback and the stage release;
busi/notes/PERF_RD1d_build_v1.md): engine_collapse_v36nq_rd13mfd.py from the flat engine engine_collapse_v36nq_rd13mfb.py (md5 02a1729a...,
code16d_flat) by marked replacement blocks, in the form of build_engine_collapse_v36nq_rd13mfb.py (busi/code/gatecomp/flat).
Rules (checked): the source's md5 begins with 02a1729a (--expect-src-md5); every OLD anchor is copied verbatim from the source and occurs
exactly once; every line a NEW block adds or changes carries "rd1d"; the reverse pass (NEW -> OLD, each NEW once) restores the source
byte for byte (in memory before writing and again on the written file; available as --reverse); the output compiles; nothing is overwritten.
    python build_engine_collapse_v36nq_rd13mfd.py --src .../engine_collapse_v36nq_rd13mfb.py [--dst .../engine_collapse_v36nq_rd13mfd.py] [--check-only]
    python build_engine_collapse_v36nq_rd13mfd.py --reverse --src .../engine_collapse_v36nq_rd13mfd.py --dst .../restored.py --expect .../engine_collapse_v36nq_rd13mfb.py

Two options, both absent by default (argparse.SUPPRESS: the json's args gain no key unless one is given, and nothing of the patch runs but
one dictionary read per call site, so the records, fields, spectra, handoff and slices are 02a1729a's bitwise):
--eigh-module v1 | v2 (with --eigh banded): the banded eigensolver module the engine imports; v1 = engine_banded_eigh_v1 (the parent's);
    v2 = engine_banded_eigh_v2 (RD-1d: v1 with the memory-safe dense route; v1 bitwise wherever v1's dense call fits).
--nk-stage-release off | on (with --mix newton): in newton_record_v32's two-stage source continuation, the direct solve's kept evaluation
    (e, U and the continued R2 of every channel, 96 kappa_sea nsites^2 bytes) leaves the device for host memory (v36m's HostCache_v36m, an
    exact copy with sizes, strides and views) before stage 1 starts, and comes back by exact copy just before its repeated confirming step
    (when stage 2 is not kept; when stage 2 is kept it is dropped as the parent drops it). The parent holds it on the device through both
    stages, so a continuation record carries one evaluation cache more than any other record (the direct solve's kept, plus the stage's kept
    and the one in progress); at 4096 sites and K 15 that is 22.5 GiB, which is the F256_s4 OOM of 22 Oct (record 0.375, stage 1). Nothing
    computed changes: the data come back by copy, never by recomputation. With the option, each finished stage's kept evaluation is also
    released in place (its dict's "cache" set to None) just before drop_arrays_v32 replaces the dict: the stage's closures form a reference
    cycle (F <-> F.jvp) that holds the old dict, and so the device tensors, until the record's gc.collect, so the parent carries stage 1's
    kept evaluation through stage 2 as well (the replay's U1: stage 2 enters with two extra caches). One log line per release, restore and drop.
Blocks:
  A  the module docstring's lineage lines;
  B  after memo_drop_v36m: RD1D, rd1d_configure, rd1d_release, rd1d_restore;
  C  rd13_configure: the module chosen by --eigh-module;
  D  newton_record_v32: the release before stage 1;
  E  newton_record_v32: the restore before the direct solve's repeated confirming step;
  F  the options (after --prep-taper);
  G  main(): rd1d_configure before rd13_configure;
  H, I, J  newton_record_v32: rd1d_drop before each drop_arrays_v32 (stage 1 before stage 2; the direct solve when stage 2 is kept; stage 2
     before the direct solve's restore).
"""
import argparse
import hashlib
import os
import py_compile

MARK = "rd1d"
SRC_MD5_PREFIX = "02a1729a"
SRC_NAME = "engine_collapse_v36nq_rd13mfb.py"
OUT_NAME = "engine_collapse_v36nq_rd13mfd.py"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

EDITS.append(("A lineage", blk(r'''
#!/usr/bin/env python
"""engine_collapse_v36nq_rd13mfb.py (busi, analyst; GATECOMP, the flat preparation; built by build_engine_collapse_v36nq_rd13mfb.py over v36nq_rd13mf, md5 248f25f3) [flat prep]
'''), blk(r'''
#!/usr/bin/env python
"""engine_collapse_v36nq_rd13mfd.py (busi, analyst; PERF RD-1d; built by build_engine_collapse_v36nq_rd13mfd.py over v36nq_rd13mfb, md5 02a1729a) [rd1d]
version 36nq_rd13mfd = version 36nq_rd13mfb with two options, absent by default, so that the defaults are 02a1729a bitwise (no json key is added): [rd1d]
--eigh-module v1|v2 (with --eigh banded): v2 = engine_banded_eigh_v2, v1 with a memory-safe dense route (on an out-of-memory error the [rd1d]
    other resident storages are parked on the host byte for byte and restored after the same torch.linalg.eigh call); [rd1d]
--nk-stage-release off|on (with --mix newton): in the two-stage source continuation the direct solve's kept evaluation leaves the device [rd1d]
    (v36m's exact host copy) before stage 1 and comes back by exact copy before its repeated confirming step, and each finished stage's [rd1d]
    kept evaluation is released in place before drop_arrays_v32 (a closure cycle otherwise holds it to the record's gc), so a continuation [rd1d]
    record holds no more evaluation caches than a direct one (F256_s4's OOM, 22 Oct, record 0.375). Version 36nq_rd13mfb's docstring follows. [rd1d]
engine_collapse_v36nq_rd13mfb.py (busi, analyst; GATECOMP, the flat preparation; built by build_engine_collapse_v36nq_rd13mfb.py over v36nq_rd13mf, md5 248f25f3) [flat prep]
''')))

EDITS.append(("B helpers", blk(r'''
def memo_drop_v36m(F):   # v36m offload: the exact J.v's memo emptied through the hook nk_stage_v32 sets on F (a no-op without --nk-jvp exact)
    d_ = getattr(F, "v36m_drop", None)   # v36m offload
    if d_ is not None:   # v36m offload
        d_()   # v36m offload
'''), blk(r'''
def memo_drop_v36m(F):   # v36m offload: the exact J.v's memo emptied through the hook nk_stage_v32 sets on F (a no-op without --nk-jvp exact)
    d_ = getattr(F, "v36m_drop", None)   # v36m offload
    if d_ is not None:   # v36m offload
        d_()   # v36m offload
RD1D = {"module": "v1", "stage_release": False, "released": 0, "restored": 0, "bytes": 0, "seconds": 0.0, "pinned": None}   # rd1d: the options' state (absent: the parent's path)
def rd1d_configure(a, ap):   # rd1d: main(), before rd13_configure (which imports the module RD1D names)
    RD1D["module"] = getattr(a, "eigh_module", "v1")   # rd1d
    if RD1D["module"] == "v2" and getattr(a, "eigh", "dense") != "banded":   # rd1d
        ap.error("rd1d: --eigh-module v2 is the banded eigensolver's module: --eigh banded is required")   # rd1d
    RD1D["stage_release"] = getattr(a, "nk_stage_release", "off") == "on"   # rd1d
    if RD1D["stage_release"] and (a.mix != "newton" or getattr(a, "nk_cache_host", "off") == "on"):   # rd1d
        ap.error("rd1d: --nk-stage-release on acts in the Newton record solve (--mix newton) with the caches on the device (--nk-cache-host off)")   # rd1d
    if RD1D["module"] != "v1" or RD1D["stage_release"]:   # rd1d
        print("rd1d: --eigh-module %s --nk-stage-release %s" % (RD1D["module"], "on" if RD1D["stage_release"] else "off"), flush=True)   # rd1d
def rd1d_alloc():   # rd1d: the device's allocated and reserved GiB (the log line's numbers)
    torch = BK["torch"]   # rd1d
    if BK["dev"] is None or BK["dev"].type != "cuda":   # rd1d
        return 0.0, 0.0   # rd1d
    return torch.cuda.memory_allocated(BK["dev"]) / 2**30, torch.cuda.memory_reserved(BK["dev"]) / 2**30   # rd1d
def rd1d_release(T, b):   # rd1d: the direct solve's kept evaluation to the host before the continuation's stage 1 (v36m's exact copy)
    if not RD1D["stage_release"] or b is None or b.get("cache") is None or isinstance(b.get("cache"), HostCache_v36m):   # rd1d
        return   # rd1d
    t_ = time.time(); a0_ = rd1d_alloc()   # rd1d
    p_ = OFF["pinned"]   # rd1d: v36m's pinned flag is a record key (the json's v36m/pinned): left as the parent's path leaves it
    hc_ = HostCache_v36m(b["cache"])   # rd1d
    sync_dev()   # rd1d: the copies complete before the device tensors can be released
    RD1D["pinned"] = OFF["pinned"]; OFF["pinned"] = p_   # rd1d
    b["cache"] = hc_   # rd1d: the device tensors die with their last holder
    gc.collect()   # rd1d
    a1_ = rd1d_alloc()   # rd1d
    RD1D["released"] += 1; RD1D["bytes"] += hc_.bytes; RD1D["seconds"] += time.time() - t_   # rd1d
    print("  rd1d T/rs %.3f: the direct solve's kept evaluation to the host before stage 1: %.2f GiB in %.1f s; device allocated %.2f -> %.2f GiB" % (   # rd1d
        T, hc_.bytes / 2**30, time.time() - t_, a0_[0], a1_[0]), flush=True)   # rd1d
def rd1d_restore(T, b):   # rd1d: back on the device by exact copy before the repeated confirming step (a no-op when it never left)
    if b is None or not isinstance(b.get("cache"), HostCache_v36m) or not RD1D["stage_release"]:   # rd1d
        return   # rd1d
    t_ = time.time(); a0_ = rd1d_alloc()   # rd1d
    hc_ = b["cache"]   # rd1d
    b["cache"] = hc_.to_device()   # rd1d
    sync_dev()   # rd1d
    RD1D["restored"] += 1; RD1D["seconds"] += time.time() - t_   # rd1d
    print("  rd1d T/rs %.3f: the direct solve's kept evaluation back on the device: %.2f GiB in %.1f s; device allocated %.2f -> %.2f GiB" % (   # rd1d
        T, hc_.bytes / 2**30, time.time() - t_, a0_[0], rd1d_alloc()[0]), flush=True)   # rd1d
def rd1d_drop(T, s, name):   # rd1d: a finished stage's kept evaluation released IN PLACE before drop_arrays_v32 replaces the dict: the stage's closures (F <-> jvp, a reference cycle) keep the old dict alive until the record's gc, and with it the device tensors
    if not RD1D["stage_release"] or s is None or s.get("best") is None or s["best"].get("cache") is None:   # rd1d
        return   # rd1d
    a0_ = rd1d_alloc()   # rd1d
    s["best"]["cache"] = None   # rd1d: the same dict object nk["best"] holds; nothing reads it after its stage (the stage's evalG is never called again)
    sync_dev()   # rd1d
    print("  rd1d T/rs %.3f: %s's kept evaluation released in place: device allocated %.2f -> %.2f GiB" % (T, name, a0_[0], rd1d_alloc()[0]), flush=True)   # rd1d
''')))

EDITS.append(("C module", blk(r'''
    import engine_banded_eigh_v1 as EBE   # rd13 banded
'''), blk(r'''
    if RD1D["module"] == "v2":   # rd1d: --eigh-module v2
        import engine_banded_eigh_v2 as EBE   # rd1d
    else:   # rd1d: the parent's module
        import engine_banded_eigh_v1 as EBE   # rd13 banded [rd1d]
''')))

EDITS.append(("D release", blk(r'''
        base_mid = {k: tuple(0.5 * (p_ + q_) for p_, q_ in zip(base_prev[k], base_new[k])) for k in base_new}
        s1 = nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with(base_mid, c_)), do_solve, f_prev, it_hist, T,
'''), blk(r'''
        base_mid = {k: tuple(0.5 * (p_ + q_) for p_, q_ in zip(base_prev[k], base_new[k])) for k in base_new}
        rd1d_release(T / a.rs, d["best"])   # rd1d: --nk-stage-release on (a no-op otherwise)
        s1 = nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with(base_mid, c_)), do_solve, f_prev, it_hist, T,
''')))

EDITS.append(("E restore", blk(r'''
            rc = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)   # the kept iterate's confirming step again, the history's last entry
'''), blk(r'''
            rd1d_restore(T / a.rs, d["best"])   # rd1d: --nk-stage-release on (a no-op otherwise)
            rc = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)   # the kept iterate's confirming step again, the history's last entry
''')))

EDITS.append(("F options", blk(r'''
    ap.add_argument("--prep-taper", dest="prep_taper", type=float, default=None, help="(flat prep) the taper width W in units of r_s (with --prep-edge taper)")   # flat prep
'''), blk(r'''
    ap.add_argument("--prep-taper", dest="prep_taper", type=float, default=None, help="(flat prep) the taper width W in units of r_s (with --prep-edge taper)")   # flat prep
    ap.add_argument("--eigh-module", dest="eigh_module", default=argparse.SUPPRESS, choices=["v1", "v2"], help="(rd1d; with --eigh banded) v1 = engine_banded_eigh_v1 (absent: v1); v2 = engine_banded_eigh_v2, the memory-safe dense route")   # rd1d
    ap.add_argument("--nk-stage-release", dest="nk_stage_release", default=argparse.SUPPRESS, choices=["off", "on"], help="(rd1d; --mix newton, --nk-cache-host off) on: the direct solve's kept evaluation on the host during the two-stage continuation, back by exact copy (absent: off)")   # rd1d
''')))

EDITS.append(("G configure", blk(r'''
    rd13_configure(a, ap)   # rd13 banded
'''), blk(r'''
    rd1d_configure(a, ap)   # rd1d
    rd13_configure(a, ap)   # rd13 banded
''')))

EDITS.append(("H drop s1", blk(r'''
            drop_arrays_v32(s1)
'''), blk(r'''
            rd1d_drop(T / a.rs, s1, "stage 1")   # rd1d: --nk-stage-release on (a no-op otherwise)
            drop_arrays_v32(s1)
''')))

EDITS.append(("I drop d", blk(r'''
            drop_arrays_v32(d)
'''), blk(r'''
            rd1d_drop(T / a.rs, d, "the direct solve")   # rd1d: --nk-stage-release on (a no-op otherwise; its host copy freed)
            drop_arrays_v32(d)
''')))

EDITS.append(("J drop s2", blk(r'''
                drop_arrays_v32(s2)
'''), blk(r'''
                rd1d_drop(T / a.rs, s2, "stage 2")   # rd1d: --nk-stage-release on (a no-op otherwise), before the direct solve's comes back
                drop_arrays_v32(s2)
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
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), OUT_NAME if not a.reverse else "engine_collapse_v36nq_rd13mfb_restored.py")
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
