#!/usr/bin/env python
"""build_engine_twofield_v9c3.py (busi lane, PERF tracker row C-A7, owner order #299; the v9c2 build's finding in the C-A6 row): engine_twofield_v9c3.py
over engine_twofield_v9c2.py (md5 6f4a0e2841b8cd7c40bc843671ea4c38) in the lane's patch form: exact replacements, each OLD occurring once in v9c2;
every line of v9c3 not in v9c2 carries "[v9c3 perf]"; --reverse restores v9c2 byte for byte; the source md5 is checked; an existing output is never
overwritten.

One option, off by default; at the default every statement executed is v9c2's (the wrapper calls v9c2's own solve expression first thing) and the
option is removed from the parsed namespace, so the json's args equal v9c2's.

  --perf-solve-memo on (C-A7): the field solve of run()'s do_solve (V35.solve: FLD.solve_constraints_ivp and the kernel average) memoized within a
      record, keyed by the exact bytes, dtype and shape of eps, J and R, the exact repr of g and ell, N0 (None only), the resolved rtol and atol
      (SOLVE_TOL), every OPT entry (collar policy, support_rel, site fields) and the identities of the solve function and of the field module
      (V35.FLD, the engine's FLD); a hit returns a deep copy of a deep copy taken at the computation (so a caller's later change to a returned
      array or info dict never reaches the memo), at most 4 entries, cleared at every record start. Where it hits: in --reference-solve lagged the
      references are held through the record's solve, so G(x0) = x0 bitwise and the confirming plain step (newton_record's / confirm_v32's
      evalG at the kept iterate) builds the sources of evaluation 1 on identical bytes and repeats its field solve (2 of the 4 solves of an S5
      record, about 0.8 s of 15.5 s). The counters stay logical: TM solve_n and solve_nfev count requests as v9c2's records (the info's nfev is
      the copy's), the field kernel cache's hits are advanced by the kernel calls the original solve made (a repeat finds every kernel cached;
      the cache is 4 GiB, the runs use < 0.1 GiB; "kernel_cache" is outside the G0'' compare anyway); the physical counts go to the per-record
      "v9c3 perf" line only. Not memoized: the indicator's delta_rtol solve, the padded collar and r0_check (they call FLD directly, other tolerances).

Usage. python build_engine_twofield_v9c3.py engine_twofield_v9c2.py [engine_twofield_v9c3.py]
         --check: the build in memory with the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c3.py restored.py [--against engine_twofield_v9c2.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v9c3 perf]"
SOURCE_MD5 = "6f4a0e2841b8cd7c40bc843671ea4c38"

BLOCK = r'''# >>> v9c3 perf: PERF C-A7 (owner order #299; the v9c2 build's finding): the record's repeated field solve on identical sources memoized; off by default, and then every statement executed is v9c2's
import copy as _v9c3_copy
V9C3_VERSION = "engine_twofield_v9c3"
V9C3 = {"memo": False}
_V9C3_MEMO = _v9c2_collections.OrderedDict()
_V9C3_STATS = {}
_V9C3_CAP = 4


class _V9c3NoMemo(Exception):
    pass


def _v9c3_inc(k, n=1):
    _V9C3_STATS[k] = _V9C3_STATS.get(k, 0) + n


def v9c3_add_arguments(ap):
    ap.add_argument("--perf-solve-memo", default="off", choices=["off", "on"], help="v9c3 perf (C-A7): the record's field solves memoized by the exact bytes of the sources and every solve input (off: v9c2)")


def v9c3_set(a):
    val = getattr(a, "perf_solve_memo", "off")
    V9C3["memo"] = val == "on"
    if hasattr(a, "perf_solve_memo") and a.perf_solve_memo == "off":
        delattr(a, "perf_solve_memo")
    if V9C3["memo"]:
        print("v9c3 perf: %s; field solve memo on (bitwise: exact-bytes keys, deep copies; at most %d entries per record; the counters in the records stay logical, the physical counts on the per-record v9c3 perf line)" % (V9C3_VERSION, _V9C3_CAP), flush=True)


def _v9c3_scalar(x):
    if isinstance(x, (bool, int, float, str, type(None))) or isinstance(x, np.generic):
        return (type(x).__name__, repr(x))
    raise _V9c3NoMemo()


def _v9c3_key(solve_fn, eps, J, g, R, ell, N0):
    if N0 is not None:
        raise _V9c3NoMemo()
    try:
        ak = (_v9c2_akey(eps), _v9c2_akey(J), _v9c2_akey(R))
    except _V9c2NoCache:
        raise _V9c3NoMemo()
    return (id(solve_fn), id(V35.FLD), id(FLD), _v9c3_scalar(g), _v9c3_scalar(ell), _v9c3_scalar(SOLVE_TOL["rtol"]), _v9c3_scalar(SOLVE_TOL["atol"]),
            tuple(sorted((str(k), _v9c3_scalar(v)) for k, v in OPT.items()))) + ak


def v9c3_solve(solve_fn, eps, J, g, R, ell, N0):
    if not V9C3["memo"]:
        return solve_fn(eps, J, g, R, ell, N0)
    try:
        key = _v9c3_key(solve_fn, eps, J, g, R, ell, N0)
    except _V9c3NoMemo:
        _v9c3_inc("solve_uncacheable")
        return solve_fn(eps, J, g, R, ell, N0)
    kc = V35.V36.get("kcache") or None
    hit = _V9C3_MEMO.get(key)
    if hit is not None:
        _V9C3_MEMO.move_to_end(key)
        t_ = time.time()
        out = _v9c3_copy.deepcopy(hit[0])
        _v9c3_inc("solve_copy_s", time.time() - t_)
        if kc is not None and hit[1]:
            kc["hits"] += hit[1]
        _v9c3_inc("solve_hit")
        _v9c3_inc("solve_saved_s", hit[2])
        return out
    k0 = (kc["hits"] + kc["misses"]) if kc is not None else 0
    t_ = time.time()
    out = solve_fn(eps, J, g, R, ell, N0)
    dt = time.time() - t_
    _v9c3_inc("solve_computed")
    calls = ((kc["hits"] + kc["misses"]) - k0) if kc is not None else 0
    try:
        t_ = time.time()
        saved = _v9c3_copy.deepcopy(out)
        _v9c3_inc("solve_copy_s", time.time() - t_)
        _V9C3_MEMO[key] = (saved, calls, dt)
        while len(_V9C3_MEMO) > _V9C3_CAP:
            _V9C3_MEMO.popitem(last=False)
            _v9c3_inc("solve_evicted")
    except Exception:
        _v9c3_inc("solve_uncacheable")
    return out


def _v9c3_line():
    return ", ".join("%s %s" % (k, (("%.2f" % v) if isinstance(v, float) else v)) for k, v in sorted(_V9C3_STATS.items()))


def v9c3_record_reset(trs):
    if V9C3["memo"] and _V9C3_STATS:
        print("  v9c3 perf (the record before T/rs %.4f): %s" % (trs, _v9c3_line()), flush=True)
    _V9C3_STATS.clear()
    _V9C3_MEMO.clear()


def v9c3_close():
    if V9C3["memo"] and _V9C3_STATS:
        print("  v9c3 perf (the last record): %s" % _v9c3_line(), flush=True)
    _V9C3_STATS.clear()
    _V9C3_MEMO.clear()
# <<< v9c3 perf'''

MAIN_SEP = "# ------------------------------------------------------------------------------------------------------- version 35's main, with v36 args\ndef main():\n"


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


def replacements():
    M = "  # " + MARK
    return [
        Rep("P1 block", "\n\n" + MAIN_SEP, "\n\n" + mark(BLOCK) + "\n\n\n" + MAIN_SEP),
        Rep("P2 args", "    v9c2_add_arguments(ap)  # [v9c2 perf]\n", "    v9c2_add_arguments(ap)  # [v9c2 perf]\n    v9c3_add_arguments(ap)" + M + "\n"),
        Rep("P3 set", "    v9c2_set(a)  # [v9c2 perf]\n", "    v9c2_set(a)  # [v9c2 perf]\n    v9c3_set(a)" + M + "\n"),
        Rep("P3 close", "        v9c2_close()  # [v9c2 perf]\n", "        v9c3_close()" + M + "\n        v9c2_close()  # [v9c2 perf]\n"),
        Rep("P4 record reset", "        v9c2_record_reset(T / rs)  # [v9c2 perf]\n", "        v9c2_record_reset(T / rs)  # [v9c2 perf]\n        v9c3_record_reset(T / rs)" + M + "\n"),
        Rep("A7 do_solve", "            out = solve(srcs_[0], srcs_[1], a.g, R, a.ell, None)\n",
            "            out = v9c3_solve(solve, srcs_[0], srcs_[1], a.g, R, a.ell, None)" + M + "\n"),
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
    ap = argparse.ArgumentParser(description="engine_twofield_v9c3 over engine_twofield_v9c2 (exact replacements; fails closed)")
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
            print("STOP: the source md5 is %s, not v9c2's %s" % (md5(src), SOURCE_MD5)); return 2
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
        print("STOP: the reverse does not restore v9c2 byte for byte"); return 2
    try:
        compile(out_text, "engine_twofield_v9c3.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c3 (md5 %s); %d replacements applied; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c3.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
