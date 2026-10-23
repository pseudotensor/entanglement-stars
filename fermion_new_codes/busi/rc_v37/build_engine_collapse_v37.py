#!/usr/bin/env python
"""build_engine_collapse_v37.py (busi builder, analyst; rc_ruling_v1 = busi/notes/REVIEWA_rc_floor_v1.md section 6.1 (a) to (d): the confirming
residual tested under a marked option, default off, at the 2 certification levels). One set of marked replacement blocks applied to a host:
  --host v36z  (engine_collapse_v36z.py md5 924eefed..., the Newton production lineage v36q -> v36y srf -> v36z ptcfb)  -> engine_collapse_v37a.py
  --host v36nq (engine_collapse_v36nq.py md5 8549c542..., the ISO composite v36q + the v36n options; the vehicle of the v36n self-test's
               D, UD and L lines, which need --iter-cache and --prep-memory, options v36z does not carry)                  -> engine_collapse_v37n.py
The option (spelling: --help governs):
  --confirm-test off | on (default off): off, nothing below runs: confirm_v32 and the v31 inline copy compute, print and store r_c and test
      nothing, as the host. On (needs --mix newton):
      (a) a solve's conv is kept < --stop-fields and dE_c < --stop-E (the host's) AND r_c at most --confirm-above, in confirm_v32 and in the
          inline copy, so a stored state in the band or worse makes the solve, hence the record, unconverged and the rounds (rounds_v36c) and
          the source continuation respond as to an unconverged kept; a line "  v37 rct: ..." is printed when r_c alone refused the stop rule;
      (b) rounds_v36c, when no round meets the stop rule: the kept round is the one with the smallest larger-of-the-2 residuals (a missing r_c
          counts as infinite); with a round meeting it, the host's choice;
      (c) the end rule (engine spec 32.1 (3), --end-above) reads the larger of the 2 residuals of the record's solve_v32 (a missing r_c
          counts as infinite); on the srf host also the step refinement's verdict (srf_verdict_v36q: "a record failing the end rule") and
          its give-up entry;
      (d) the record line carries "; v37 rct [kept, confirming, ratio, the 2 levels, the end rule: fires on kept | confirming | both | does
          not fire]"; when the end rule fires, its line names the residual that fired it and end_v32 carries "fired_by" and both residuals;
          the run label (head and records) carries "; v37 rct: ..." and solve_v32.options and the fields file's solver entry "confirm_test".
  --confirm-above X (default 1e-4, the band edge): the level of (a). Nothing tests r_c at --stop-fields.
Blocks (each OLD occurs exactly once in the host; required on both hosts unless marked srf):
  A lineage (the module docstring's first lines); B the settings dict CONF; C the helpers; D confirm_v32 (a); E the v31 inline copy (a);
  F rounds_v36c (b); G the run label; H v32_options; I the record line (d); J the end rule (c), (d); K srf give-up entry (c) [srf];
  L srf verdict (c) [srf]; M the options; N the validation.
Rules (checked): the host's md5 prefix; every OLD anchor once; every line a NEW block adds or changes carries "v37 rct" (blank lines neutral; a
host line kept verbatim is not new); the reverse pass (NEW -> OLD, in reverse order) restores the host byte for byte, in memory and on the
written file; the result compiles; nothing is overwritten.
    python build_engine_collapse_v37.py --host v36z --src .../engine_collapse_v36z.py [--dst .../engine_collapse_v37a.py] [--check-only]
    python build_engine_collapse_v37.py --host v36z --reverse --src .../engine_collapse_v37a.py --dst .../restored.py --expect .../engine_collapse_v36z.py
"""
import argparse
import hashlib
import os
import py_compile

MARK = "v37 rct"
HOSTS = {"v36z": {"md5": "924eefed", "name": "engine_collapse_v36z.py", "out": "engine_collapse_v37a.py", "srf": True},
         "v36nq": {"md5": "8549c542", "name": "engine_collapse_v36nq.py", "out": "engine_collapse_v37n.py", "srf": False}}


def blk(s):
    return s[1:] if s.startswith("\n") else s


def full_line(text, start):
    """the host's one line beginning with `start` (verbatim, with its newline)."""
    hits = [ln for ln in text.splitlines(True) if ln.startswith(start)]
    if len(hits) != 1:
        raise SystemExit("anchor line %r occurs %d times" % (start[:80], len(hits)))
    return hits[0]


def edits(text, host):
    H = HOSTS[host]
    out_name = H["out"]
    E = []
    # A: the lineage. The host's docstring opens on line 2 with '"""<name> ('; v37's lines go in front, the opening quotes move to them.
    old = '#!/usr/bin/env python\n"""' + H["name"] + " ("
    E.append(("A lineage", old, '#!/usr/bin/env python\n"""' + out_name + " (busi builder, analyst; rc_ruling_v1 6.1: the confirming residual tested under --confirm-test, default off; built by build_engine_collapse_v37.py over " + H["name"] + " md5 " + H["md5"] + ") [v37 rct]\n"
              "version 37 = the host with one option, off by default, so that the defaults are the host bitwise (the json's args carry the 2 new keys): --confirm-test on [v37 rct]\n"
              "    (needs --mix newton): (a) a solve meets the stop rule only when, beside the host's tests (kept < --stop-fields, confirming |dE/E| < --stop-E), [v37 rct]\n"
              "    its confirming residual r_c is at most --confirm-above (1e-4), so that a stored state in the band or worse is unconverged and the rounds [v37 rct]\n"
              "    respond; (b) with no round meeting it, the round kept is the one with the smallest larger-of-the-2 residuals; (c) the end rule [v37 rct]\n"
              "    (--end-above) reads the larger of the 2 residuals (and on the srf host the step refinement's verdict reads what the end rule reads); [v37 rct]\n"
              "    (d) the record line prints both residuals, the ratio and the end rule's reading, its end line which residual fired it; the run label [v37 rct]\n"
              "    and solve_v32.options record the option. Nothing tests r_c at --stop-fields (rc_ruling_v1 6.1). The host's docstring follows. [v37 rct]\n"
              + H["name"] + " ("))

    # B: the settings
    old = 'OFF = {"on": False, "test": False, "pinned": None, "rec": None}'
    line = full_line(text, old)
    E.append(("B settings", line, line + 'CONF = {"on": False, "above": 1e-4}   # v37 rct: --confirm-test (on: r_c enters the stop rule at --confirm-above, the kept round by the larger residual, the end rule by the larger residual)\n'))

    # C: the helpers, before cross_lower_v36d
    E.append(("C helpers", "\ndef cross_lower_v36d(steps):", "\n" + blk(r'''
def rct_larger_v37(s32):   # v37 rct: the larger of a solve_v32 entry's 2 residuals (a missing confirming residual counts as infinite)
    k_ = s32.get("kept_residual"); c_ = s32.get("confirming_residual")   # v37 rct
    return max(k_, c_ if c_ is not None else float("inf"))   # v37 rct


def rct_by_v37(s32, thr):   # v37 rct: which residual is above thr: "kept", "confirming", "both" or "" (a missing confirming residual is above)
    k_ = s32.get("kept_residual"); c_ = s32.get("confirming_residual")   # v37 rct
    ka = k_ is not None and k_ > thr; ca = c_ is None or c_ > thr   # v37 rct
    return "both" if (ka and ca) else ("kept" if ka else ("confirming" if ca else ""))   # v37 rct


def rct_desc_v37():   # v37 rct: the option in words (the run label, solve_v32.options)
    return ("on (rc_ruling_v1 6.1: a solve meets the stop rule only with its confirming residual at most %g; with no round meeting it the round with the "   # v37 rct
            "smallest larger-of-the-2 residuals is kept; the end rule reads the larger of the 2 residuals against --end-above)" % CONF["above"])   # v37 rct


def rct_segment_v37(a, s32):   # v37 rct (d): the record line's part: both residuals, the ratio, the 2 levels, the end rule's reading
    k_ = s32.get("kept_residual"); c_ = s32.get("confirming_residual")   # v37 rct
    ratio = ("%.3g" % (c_ / k_)) if (k_ not in (None, 0) and c_ is not None) else "-"   # v37 rct
    if a.end_above is None:   # v37 rct
        er = "the end rule off"   # v37 rct
    else:   # v37 rct
        by = rct_by_v37(s32, a.end_above)   # v37 rct
        er = ("the end rule (%g) fires on %s" % (a.end_above, ("both residuals" if by == "both" else "the %s residual" % by))) if by else ("the end rule (%g) does not fire" % a.end_above)   # v37 rct
    return "; v37 rct [kept %s, confirming %s, ratio %s; --confirm-above %g: %s; %s]" % (   # v37 rct
        ("%.2e" % k_) if k_ is not None else "none", ("%.2e" % c_) if c_ is not None else "none", ratio, CONF["above"],   # v37 rct
        ("met" if (c_ is not None and c_ <= CONF["above"]) else "above"), er)   # v37 rct


def rct_end_v37(a, s32):   # v37 rct (c), (d): the end entry's fields when the end rule (the larger residual) fired
    return {"confirming_residual": s32.get("confirming_residual"), "larger_residual": rct_larger_v37(s32), "fired_by": rct_by_v37(s32, a.end_above),   # v37 rct
            "rule_v37": "rc_ruling_v1 6.1 (c): the end rule reads the larger of the kept and the confirming residual against --end-above"}   # v37 rct


def rct_refused_v37(where, r_c, base_conv):   # v37 rct (a): a line when the confirming residual alone refused the stop rule
    if base_conv and not (r_c is not None and r_c <= CONF["above"]):   # v37 rct
        print("  v37 rct %s: the confirming residual %s is above --confirm-above %g: the solve does not meet the stop rule" % (   # v37 rct
            where, ("%.2e" % r_c) if r_c is not None else "none", CONF["above"]), flush=True)   # v37 rct


def cross_lower_v36d(steps):''')))

    # D: confirm_v32 (a)
    old = blk(r'''
    conv = bool(b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E)
    return {"r_c": r_c, "dE_c": dE_c, "conv": conv}
''')
    E.append(("D confirm_v32", old, blk(r'''
    conv = bool(b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E)
    if CONF["on"]:   # v37 rct (a): the stored state's residual enters the stop rule at --confirm-above
        rct_refused_v37("confirm_v32", r_c, conv)   # v37 rct
        conv = bool(conv and r_c is not None and r_c <= CONF["above"])   # v37 rct
    return {"r_c": r_c, "dE_c": dE_c, "conv": conv}
''')))

    # E: the v31 inline copy (a)
    old = blk(r'''
    conv = b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E
    b["dE"] = dE_c if dE_c is not None else b["dE"]
''')
    E.append(("E inline copy", old, blk(r'''
    conv = b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E
    if CONF["on"]:   # v37 rct (a): the same test in the v31 path's inline copy
        rct_refused_v37("v31 inline", r_c, conv)   # v37 rct
        conv = bool(conv and r_c is not None and r_c <= CONF["above"])   # v37 rct
    b["dE"] = dE_c if dE_c is not None else b["dE"]
''')))

    # F: rounds_v36c (b)
    old = blk(r'''
        chosen = min(cands, key=lambda r: r["best"]["resid"]) if cands else rounds[-1]
''')
    E.append(("F rounds", old, blk(r'''
        chosen = min(cands, key=lambda r: r["best"]["resid"]) if cands else rounds[-1]
        if CONF["on"] and cands:   # v37 rct (b): no round meets the stop rule: the round with the smallest larger-of-the-2 residuals (a missing r_c infinite)
            chosen = min(cands, key=lambda r: max(r["best"]["resid"], r["r_c"] if r["r_c"] is not None else float("inf")))   # v37 rct
''')))

    # G: the run label
    old = blk(r'''
    if v32:
        label += "; v32: " + v32_desc(a)
''')
    E.append(("G label", old, blk(r'''
    if v32:
        label += "; v32: " + v32_desc(a)
    if CONF["on"]:   # v37 rct (d): the option's state in the run label (the head and every record)
        label += "; v37 rct: --confirm-test " + rct_desc_v37()   # v37 rct
''')))

    # H: v32_options
    old = '            "end_above": a.end_above,   # v36q fallback\n'
    E.append(("H options entry", old, old + '            **({"confirm_test": rct_desc_v37()} if CONF["on"] else {}),   # v37 rct (d): the option in solve_v32.options and the fields file\'s solver entry\n'))

    # I: the record line (d)
    old = '        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)\n'
    E.append(("I record line", old, '        print(_line + (v32_record_segment(solve32) if solve32 is not None else "") + (rct_segment_v37(a, solve32) if (CONF["on"] and solve32 is not None) else ""), flush=True)   # v37 rct (d): both residuals, the ratio, the end rule\'s reading\n'))

    # J: the end rule (c), (d)
    cond = '        if solve32 is not None and a.end_above is not None and solve32["kept_residual"] > a.end_above:\n'
    pr = full_line(text, '            print("T/rs %.3f: the member ends at this record (engine spec 32.1 (3)): the kept residual')
    pr2 = full_line(text, '                T / rs, solve32["kept_residual"], solve32["kept_from"], a.end_above, solve32["continuation"]), flush=True)')
    old = cond + '            # engine spec 32.1 (3)'
    E.append(("J1 end rule condition", old, '        if solve32 is not None and a.end_above is not None and (solve32["kept_residual"] > a.end_above or (CONF["on"] and rct_larger_v37(solve32) > a.end_above)):   # v37 rct (c): on, the larger of the 2 residuals\n            # engine spec 32.1 (3)'))
    old = pr + pr2 + "            break\n"
    E.append(("J2 end rule line", old, blk(r'''
            if CONF["on"]:   # v37 rct (c), (d): the end entry and line name the residual that fired the rule
                end32.update(rct_end_v37(a, solve32))   # v37 rct
                print("T/rs %.3f: the member ends at this record (engine spec 32.1 (3), v37 rct (c)): the larger of the kept residual %s and the confirming residual %s (%s) exceeds %g, fired by %s, after the source continuation (%s); the record is written and no record is propagated from an uncertified state" % (   # v37 rct
                    T / rs, "%.2e" % solve32["kept_residual"], ("%.2e" % solve32["confirming_residual"]) if solve32.get("confirming_residual") is not None else "none", solve32["kept_from"], a.end_above, end32["fired_by"], solve32["continuation"]), flush=True)   # v37 rct
            else:   # v37 rct: the host's line
''') + "    " + pr.rstrip("\n") + "   # v37 rct\n" + "    " + pr2.rstrip("\n") + "   # v37 rct\n" + "            break\n"))

    if H["srf"]:
        # K: the srf give-up's end entry (c)
        old = full_line(text, '            if solve32 is not None and a.end_above is not None and solve32["kept_residual"] > a.end_above:   # srf')
        assert old.rstrip("\n").endswith(":   # srf: the version 32 end rule's entry beside it where it applies"), old
        new = old.replace('solve32["kept_residual"] > a.end_above:   # srf', '(solve32["kept_residual"] > a.end_above or (CONF["on"] and rct_larger_v37(solve32) > a.end_above)):   # v37 rct (c) and srf', 1)
        E.append(("K srf end entry", old, new))
        old = full_line(text, '                         "rule": "engine spec 32.1 (3): a record whose kept residual exceeds the bound after the continuation ends the member at that record; no record is propagated from an uncertified state"}   # srf')
        E.append(("K2 srf end entry fields", old, old + '                if CONF["on"]:   # v37 rct (c): the residual that fired the end rule\n                    end32.update(rct_end_v37(a, solve32))   # v37 rct\n'))
        # L: the srf verdict (c)
        old = blk(r'''
        if kept > thr:
            return {"reason": "the kept residual %.3e exceeds %g" % (kept, thr), "kept_residual": kept}
''')
        E.append(("L srf verdict", old, old + blk(r'''
        if CONF["on"] and rct_larger_v37(solve32) > thr:   # v37 rct (c): the refinement reads what the end rule reads, the larger of the 2 residuals
            return {"reason": "the confirming residual %s exceeds %g (the kept %.3e) [v37 rct]" % (("%.3e" % solve32["confirming_residual"]) if solve32.get("confirming_residual") is not None else "none", thr, kept), "kept_residual": kept, "confirming_residual": solve32.get("confirming_residual")}   # v37 rct
''')))

    # M: the options
    old = full_line(text, '    ap.add_argument("--nk-plain-fallback"')
    E.append(("M argparse", old, old + blk(r'''
    ap.add_argument("--confirm-test", dest="confirm_test", default="off", choices=["off", "on"], help="(v37, rc_ruling_v1 6.1) on: a Newton solve meets the stop rule only with its confirming residual (the plain step at the kept iterate's output, the state the record stores) at most --confirm-above; with no round meeting it the round with the smallest larger-of-the-2 residuals is kept; the end rule (--end-above) reads the larger of the 2 residuals; the record line prints both, the ratio and which residual fired the end rule; needs --mix newton; off: bitwise the host")   # v37 rct
    ap.add_argument("--confirm-above", dest="confirm_above", type=float, default=1e-4, help="(v37) the level of --confirm-test on's stop-rule test of the confirming residual (default 1e-4, the reading rule's band edge; never --stop-fields)")   # v37 rct
''')))

    # N: the validation
    old = full_line(text, '        ap.error("version 36q: --nk-plain-fallback acts in the set-resolving solve')
    E.append(("N validation", old, old + blk(r'''
    if a.confirm_test == "on":   # v37 rct
        if a.mix != "newton":   # v37 rct
            ap.error("version 37: --confirm-test on tests the Newton record's confirming residual: --mix newton is required")   # v37 rct
        if not (a.confirm_above > 0.0):   # v37 rct
            ap.error("version 37: --confirm-above must be positive")   # v37 rct
        CONF.update({"on": True, "above": float(a.confirm_above)})   # v37 rct
''')))
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
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}   # the host's docstring line, its opening quotes moved to the v37 lines
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
    ap.add_argument("--src", required=True, help="the host (build) or the v37 engine (--reverse)")
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
    if MARK in text or "CONF = {" in text or "--confirm-test" in text:
        raise SystemExit("the host already carries the option or %r; nothing written" % MARK)
    E = edits(text, a.host)
    for name, old, new in E:
        print("  block %-26s OLD occurs %d time(s) (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
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
    py_compile.compile(a.dst, doraise=True)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the host byte for byte; grep -c '%s' -> %d" % (
        a.dst, len(out), md5(out), MARK, sum(1 for ln in built.splitlines() if MARK in ln)))


if __name__ == "__main__":
    main()
