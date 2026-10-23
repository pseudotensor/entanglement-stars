#!/usr/bin/env python
"""build_engine_collapse_rd2.py (busi, analyst; PERF rows RD-2 and RD-5; busi/notes/PERF_RD2_build_v1.md): the Newton engines with the field
module's RD-2 collocation route (--fields) and the RD-5 warm start (--fields-warm-start) as options, off by default. One set of marked blocks
applied to a host:
  --host v36nq (engine_collapse_v36nq.py md5 8549c542..., the ISO composite)            -> engine_collapse_v36nq_rd2.py
  --host v36r  (engine_collapse_v36r.py  md5 7b6d0f4f..., v36n + the plain fallback)    -> engine_collapse_v36r_rd2.py
  --host v37a  (engine_collapse_v37a.py  md5 2580dbfe..., v36z + rct, the srf/ptc line) -> engine_collapse_v37a_rd2.py
  --host v37n  (engine_collapse_v37n.py  md5 a68b046e..., v36nq + rct)                  -> engine_collapse_v37n_rd2.py
The options (the field module must be engine_fields_v16d's content under the name engine_fields_v16.py; the engine refuses otherwise):
  --fields ivp | collocation (default ivp): collocation calls engine_fields_v16.rd2_configure, which sets the module's route and
      QI2_FIELDS_ROUTE in os.environ (the pair workers are spawned processes and read it when they import the module); every solve at the
      engine's field tolerance (rtol 1e-12, atol 1e-16) is then the collocation solve, with the ivp route as the automatic, logged fallback.
  --fields-warm-start off | on (default off; on needs --fields collocation): the collocation Newton starts from the latest solve on the same mesh.
  Under a non-default option each record carries "fields_rd2" (the route and the field module's counters since the previous record, this
  process) and prints one "rd2 fields:" line; the run prints one "rd2 fields: route ..." line at the start.
At the defaults no new line runs: the engine is the host bitwise (the json's args carry the 2 new keys).
Blocks (each OLD occurs exactly once in the host): A lineage (the docstring's first line), B the settings and the record helper (after the
field module's import), C the options (after --fields-v17), D the configuration (after the parse), E the record entry (after the record's
"label"/"seconds"/"indicator" line). Rules (checked): the host's md5 prefix; every OLD anchor once; every line a NEW block adds carries
"rd2 fields" (a host line kept verbatim is not new); the reverse pass (NEW -> OLD) restores the host byte for byte, in memory and on the
written file; the result compiles; nothing is overwritten.
    python build_engine_collapse_rd2.py --host v36nq --src .../engine_collapse_v36nq.py [--dst .../engine_collapse_v36nq_rd2.py] [--check-only]
    python build_engine_collapse_rd2.py --host v36nq --reverse --src .../engine_collapse_v36nq_rd2.py --dst .../restored.py --expect .../engine_collapse_v36nq.py
"""
import argparse
import hashlib
import os

MARK = "rd2 fields"
HOSTS = {"v36nq": {"md5": "8549c542", "name": "engine_collapse_v36nq.py", "out": "engine_collapse_v36nq_rd2.py"},
         "v36r": {"md5": "7b6d0f4f", "name": "engine_collapse_v36r.py", "out": "engine_collapse_v36r_rd2.py"},
         "v37a": {"md5": "2580dbfe", "name": "engine_collapse_v37a.py", "out": "engine_collapse_v37a_rd2.py"},
         "v37n": {"md5": "a68b046e", "name": "engine_collapse_v37n.py", "out": "engine_collapse_v37n_rd2.py"}}


def edits(text, host):
    H = HOSTS[host]
    E = []
    old = '#!/usr/bin/env python\n"""' + H["name"] + " ("
    E.append(("A lineage", old, '#!/usr/bin/env python\n"""' + H["out"] + " (busi, analyst; PERF RD-2 and RD-5: the field solve by collocation under --fields, default ivp; built by build_engine_collapse_rd2.py over " + H["name"] + " md5 " + H["md5"] + ") [rd2 fields]\n"
              "version rd2 = the host with two options, off by default, so that the defaults are the host bitwise (the json's args carry the 2 new keys): [rd2 fields]\n"
              "    --fields collocation: every field solve at the engine's tolerance by engine_fields_v16d's Gauss-Legendre collocation (the ivp route as [rd2 fields]\n"
              "    the logged automatic fallback, e.g. the flat-slice continuation); --fields-warm-start on: its Newton warm-started from the latest solve [rd2 fields]\n"
              "    on the same mesh. Needs engine_fields_v16d's content as engine_fields_v16.py in the code dir. The host's docstring follows. [rd2 fields]\n"
              + H["name"] + " ("))
    old = "import engine_fields_v16 as FLD\n"
    E.append(("B settings", old, old +
              'RD2F = {"on": False, "route": "ivp", "warm": "off", "prev": None}   # rd2 fields: --fields / --fields-warm-start (off = the host bitwise)\n'
              '\n'
              '\n'
              'def rd2_record_v1():   # rd2 fields: the record entry under a non-default option: the route and the field module counters since the previous record\n'
              '    import engine_fields_v16 as F16_   # rd2 fields\n'
              '    s_ = F16_.rd2_stats()   # rd2 fields\n'
              '    p_ = RD2F["prev"] or {}   # rd2 fields\n'
              '    d_ = {k_: s_[k_] - p_.get(k_, 0) for k_ in s_ if isinstance(s_[k_], (int, float))}   # rd2 fields\n'
              '    RD2F["prev"] = s_   # rd2 fields\n'
              '    print("  rd2 fields: route %s, warm start %s; since the previous record: collocation %d, fallbacks %d, ivp by tolerance %d, warm starts %d, Newton iterations %d, collocation %.2f s, ivp %.2f s (this process)" % (   # rd2 fields\n'
              '        RD2F["route"], RD2F["warm"], d_["collocation"], d_["fallback"], d_["ivp_by_tolerance"], d_["warm_used"], d_["newton_iterations"], d_["collocation_s"], d_["ivp_s"]), flush=True)   # rd2 fields\n'
              '    return {"fields_rd2": {"route": RD2F["route"], "warm_start": RD2F["warm"], "since_previous_record": d_, "fallback_reasons_total": s_["fallback_reasons"],   # rd2 fields\n'
              '                           "scope": "this process (pair workers count apart; their fallback lines print to the same log)"}}   # rd2 fields\n'))
    old_c = 'ap.add_argument("--fields-v17", dest="fields_v17", action="store_true", help="(e) engine_fields_v17 for every solve")\n'
    lines = [ln for ln in text.splitlines(True) if ln.endswith(old_c)]
    if len(lines) != 1:
        raise SystemExit("the --fields-v17 line occurs %d times" % len(lines))
    ind = lines[0][:len(lines[0]) - len(old_c)]
    E.append(("C options", lines[0], lines[0] +
              ind + 'ap.add_argument("--fields", dest="fields", default="ivp", choices=["ivp", "collocation"], help="(RD-2) collocation: the field solve by engine_fields_v16d\'s Gauss-Legendre collocation, the ivp route as the logged fallback; ivp: the host")   # rd2 fields\n'
              + ind + 'ap.add_argument("--fields-warm-start", dest="fields_warm_start", default="off", choices=["off", "on"], help="(RD-5) on: the collocation Newton warm-started from the latest solve on the same mesh (needs --fields collocation)")   # rd2 fields\n'))
    old = "    a = ap.parse_args()\n"
    E.append(("D configuration", old, old +
              '    if a.fields != "ivp" or a.fields_warm_start != "off":   # rd2 fields\n'
              '        if a.fields_warm_start == "on" and a.fields != "collocation":   # rd2 fields\n'
              '            ap.error("--fields-warm-start on needs --fields collocation")   # rd2 fields\n'
              '        import engine_fields_v16 as _F16rd2   # rd2 fields\n'
              '        if not hasattr(_F16rd2, "rd2_configure"):   # rd2 fields\n'
              '            ap.error("--fields %s needs engine_fields_v16d\'s content as engine_fields_v16.py in the code dir (the RD-2 kit)" % a.fields)   # rd2 fields\n'
              '        _F16rd2.rd2_configure(a.fields, a.fields_warm_start)   # rd2 fields: the module and os.environ (spawned pair workers inherit it)\n'
              '        RD2F.update(on=True, route=a.fields, warm=a.fields_warm_start)   # rd2 fields\n'
              '        print("rd2 fields: route %s, warm start %s; field module %s (%s)" % (a.fields, a.fields_warm_start, _F16rd2.__name__, _F16rd2.__file__), flush=True)   # rd2 fields\n'))
    old = '               "label": label, "seconds": time.time() - tr, "indicator": ind,\n'
    E.append(("E record", old, old + '               **(rd2_record_v1() if RD2F["on"] else {}),   # rd2 fields\n'))
    return E


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text, E):
    for name, old, new in E:
        c = text.count(old)
        if c != 1:
            raise SystemExit("block %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100] if old.strip() else old))
        text = text.replace(old, new)
    return text


def reverse(text, E):
    for name, old, new in reversed(E):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check(E):
    bad, n = [], 0
    for name, old, new in E:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}
        for ln in new.splitlines():
            if ln in olds or ln.strip() == "":
                continue
            if MARK in ln:
                n += 1
            else:
                bad.append((name, ln))
    return bad, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, choices=sorted(HOSTS))
    ap.add_argument("--src", required=True, help="the host (build) or the rd2 engine (--reverse)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None, help="(reverse) the host the restored text must equal byte for byte; the blocks are derived from it")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    H = HOSTS[a.host]
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        if not a.expect:
            raise SystemExit("--reverse needs --expect (the host, from which the blocks are derived)")
        host_text = open(a.expect, "rb").read().decode("utf-8")
        E = edits(host_text, a.host)
        out = reverse(text, E).encode("utf-8")
        if a.dst and not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        ok = out == host_text.encode("utf-8")
        print("reverse: %s %s (md5 %s); restored equals %s byte for byte: %s" % ("would write" if (a.check_only or not a.dst) else "wrote", a.dst, md5(out), a.expect, ok))
        raise SystemExit(0 if ok else 1)
    print("host %s %s: md5 %s, %d bytes" % (a.host, a.src, md5(src), len(src)))
    if not md5(src).startswith(H["md5"]):
        raise SystemExit("the host's md5 does not start with %s: not %s; nothing written" % (H["md5"], H["name"]))
    if MARK in text or "RD2F = {" in text or '"--fields"' in text:
        raise SystemExit("the host already carries the option or %r; nothing written" % MARK)
    E = edits(text, a.host)
    for name, old, new in E:
        print("  block %-16s OLD occurs %d time(s) (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n = marker_check(E)
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln[:160]))
        raise SystemExit("new lines without the marker")
    print("every new or changed line carries %r (%d lines)" % (MARK, n))
    built = forward(text, E)
    out = built.encode("utf-8")
    back = reverse(built, E).encode("utf-8")
    print("reversible (NEW -> OLD restores the host byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the host; nothing written")
    try:
        compile(built, a.dst or H["out"], "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), H["out"])
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
    if written != out or reverse(written.decode("utf-8"), E).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the host")
    compile(written.decode("utf-8"), a.dst, "exec")
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the host byte for byte; grep -c '%s' -> %d" % (
        a.dst, len(out), md5(out), MARK, sum(1 for ln in built.splitlines() if MARK in ln)))


if __name__ == "__main__":
    main()
