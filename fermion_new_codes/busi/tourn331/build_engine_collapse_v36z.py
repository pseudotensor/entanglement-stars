#!/usr/bin/env python
"""build_engine_collapse_v36z.py (busi, analyst; #331, TOURNAMENT_reading_v1 sec. 4.2: BUILD the PTC fallback form, not PTC as the loop):
engine_collapse_v36z.py from engine_collapse_v36y.py (md5 517d4466..., v36q + the merged substep option) by marked replacement blocks, in the
form of build_engine_collapse_v36r.py, with the pseudo-transient code copied verbatim from engine_collapse_v36t.py (md5 40249674..., the
engine of the P2_ptc_RUNG2c pilot) at build time.

The patch (defaults bitwise v36y, hence v36q: with --nk-ptc off, the default, nothing of it runs):
  v36q's Newton loop (nk_loop_v36c) is unchanged. Only when, at a Newton step, the halving search found no trial that lowers the residual on the
  held set AND the plain-direction fallback (--nk-plain-fallback on) found none either (or is off), the round hands over to v36t's
  pseudo-transient loop nk_loop_ptc_v36t from the current iterate with delta0 = --nk-ptc-delta0 (1 by default): backward Euler on
  dx/dt = G(x) - x with the shifted exact J.v - v/delta, SER growth, the two-sided ladder on a rejected trial, the Newton limit delta_max the
  last rung. Its status is the round's (rounds_v36c's rounds, far-side starts and choice unchanged). The dispatch of v36t at nk_loop_v36c's head
  is NOT ported (PTC is not the loop: the reading's 4.2, v36t stops RUNG2 at 0.5 from T0 and costs 2x on quiet records); v36t's --nk-ptc on
  is not offered; the exclusion of --nk-plain-fallback with --nk-ptc is lifted (the fallback value runs after the plain direction).
Blocks:
  A  the module docstring's lineage line;
  B  the PTC settings dict (v36t's, plus "fallback" and "test");
  C  nk_loop_v36c's failure branch: --nk-ptc test (the self-test's exercise in the whole engine: at Newton step 1 of every round the ladder
     runs whatever the Newton trials did), then after the plain fallback, the ladder when acc is still None;
  D  v36t's ptc_desc_v36t, ptc_ladder_v36t and nk_loop_ptc_v36t verbatim, and nk_ptc_fallback_v36z (the hand-over: marks the step entry
     "ptc_fallback", every ladder step entry "ptc_fallback_from_newton_step", prints one line before and one after, appends a clause to a
     non-converged status);
  E  v32_options: v36t's "ptc" entry verbatim plus "ptc_fallback" (only with the option on);
  F  rounds_v36c: v36t's --nk-ptc-one-cache release before the confirming step, verbatim;
  G  nk_stage_v32: v36t's three hooks (v36t_rebind, v36t_release, v36t_restore) verbatim;
  H  nk_stage_v32: v36t's record label RS["v36t_record"] verbatim;
  I  the options: --nk-ptc off | fallback | test (new), v36t's eight sub-options verbatim (delta0 1, delta-min 1e-3, delta-max 1e6, ladder 4,
     pivot 16, trials 8, dx-max 0.1, one-cache off; one-cache kept: the reading's 4.2, the only thing that let a Newton solve fit ISO on an H100);
  J  the validation: --nk-set resolve required, v36t's range checks and its one-cache rule; PTC.update.
Rules: the source's md5 begins with 517d4466 (--expect-src-md5) and v36t's with 40249674 (--expect-ptc-md5); every OLD anchor and every v36t
excerpt occurs exactly once (checked); every line a NEW block adds or changes carries "v36z ptcfb" or is a v36t line carrying "v36t ptc" that
occurs verbatim in v36t (checked); the reverse pass restores v36y byte for byte (in memory and on the written file; --reverse); it compiles.
    python build_engine_collapse_v36z.py --src .../engine_collapse_v36y.py --ptc .../engine_collapse_v36t.py [--dst .../engine_collapse_v36z.py] [--check-only]
    python build_engine_collapse_v36z.py --reverse --src .../engine_collapse_v36z.py --ptc .../engine_collapse_v36t.py --dst .../restored.py --expect .../engine_collapse_v36y.py
"""
import argparse
import hashlib
import os
import py_compile

MARK = "v36z ptcfb"
TMARK = "v36t ptc"
SRC_MD5_PREFIX = "517d4466"
PTC_MD5_PREFIX = "40249674"


def blk(s):
    return s[1:] if s.startswith("\n") else s


def one(text, start, end=None, include_end=False, label=""):
    """the excerpt of v36t from the line starting with `start` (once) to the line before `end` (or through it)."""
    if text.count(start) != 1:
        raise SystemExit("v36t excerpt %s: its start occurs %d times" % (label, text.count(start)))
    i = text.index(start)
    if end is None:
        j = text.index("\n", i) + 1
        return text[i:j]
    j = text.index(end, i)
    if include_end:
        j = text.index("\n", j) + 1
    return text[i:j]


def edits(T):
    """the blocks; T is engine_collapse_v36t.py's text (the verbatim excerpts)."""
    E = []
    funcs = one(T, "def ptc_desc_v36t():   # v36t ptc\n", "\n\n\ndef tm_reset():", label="functions")
    hooks = one(T, '        if PTC["on"]:   # v36t ptc: the pseudo-transient loop\'s three hooks', "            F.v36t_rebind = v36t_rebind; F.v36t_release = v36t_release; F.v36t_restore = v36t_restore   # v36t ptc\n", include_end=True, label="hooks")
    optline = one(T, '            **({"ptc": ptc_desc_v36t()} if PTC["on"] else {}),   # v36t ptc', label="options entry")
    conf = one(T, '            if PTC["one_cache"] and getattr(F, "v36t_release", None) is not None:   # v36t ptc', "                F.v36t_release()   # v36t ptc\n", include_end=True, label="confirm release")
    lab = one(T, '    if PTC["on"]:   # v36t ptc: the record and the solve the ptc lines name', '        RS["v36t_record"] = "T/rs %.3f %s" % (T / a.rs, name)   # v36t ptc\n', include_end=True, label="record label")
    subargs = one(T, '    ap.add_argument("--nk-ptc-delta0"', '    ap.add_argument("--nk-ptc-one-cache"', include_end=True, label="sub-options")
    rng = one(T, '        if not (0.0 < a.nk_ptc_delta_min <= a.nk_ptc_delta0 <= a.nk_ptc_delta_max)', '            ap.error("version 36t: need 0 < --nk-ptc-delta-min', include_end=True, label="range check")
    occ = one(T, '        if a.nk_ptc_one_cache == "on" and not OFF["on"]:   # v36t ptc', '            ap.error("version 36t: --nk-ptc-one-cache needs --nk-host-release on', include_end=True, label="one-cache rule")
    if subargs.count("ap.add_argument") != 8:
        raise SystemExit("v36t's sub-options: expected 8 lines")

    # A: the lineage
    E.append(("A lineage", blk(r'''
"""engine_collapse_v36y.py (busi, analyst; #331, TOURNAMENT_reading_v1 sec. 4.1: the substep patch merged into the Newton engine lineage;'''), blk(r'''
"""engine_collapse_v36z.py (busi, analyst; #331, TOURNAMENT_reading_v1 sec. 4.2: the PTC ladder as a FALLBACK stage behind the Newton halving search and the plain direction; built by build_engine_collapse_v36z.py over v36y md5 517d4466, the ladder verbatim from engine_collapse_v36t.py md5 40249674) [v36z ptcfb]
version 36z = version 36y with one option group, off by default, so that the defaults are version 36y bitwise, hence version 36q bitwise with --step-refine off [v36z ptcfb]
(with --nk-ptc off nothing of it runs; the json's "args" carries the new keys at their defaults): --nk-ptc fallback (needs --nk-set resolve): the [v36z ptcfb]
    Newton loop nk_loop_v36c runs unchanged; when at a Newton step no trial of the halving search lowers the residual on the held set and the [v36z ptcfb]
    plain-direction fallback (--nk-plain-fallback on) lowers it at no omega either (or is off), the round hands over to v36t's pseudo-transient [v36z ptcfb]
    loop nk_loop_ptc_v36t from the current iterate with delta0 = --nk-ptc-delta0 (1): backward Euler on dx/dt = G(x) - x with the shifted exact [v36z ptcfb]
    J.v - v/delta, SER growth to the Newton limit, the two-sided ladder on a rejected trial (v36t's docstring below); its status is the round's. [v36z ptcfb]
    The step entry where the hand-over happens carries "ptc_fallback" (the Newton step, why, delta0, the ladder's steps and status); the ladder's [v36z ptcfb]
    own step entries carry "ptc" and "ptc_fallback_from_newton_step"; one line before and one after the ladder print the record. With the option [v36z ptcfb]
    on, a record where Newton or the plain direction is accepted at every step is version 36y's bitwise (the option is inert there). v36t's [v36z ptcfb]
    --nk-ptc on (the ladder in place of the Newton loop) is not offered (the reading's 4.2: it stops RUNG2 at 0.5 from T0 where Newton's half [v36z ptcfb]
    step passes). --nk-ptc test: fallback, and at Newton step 1 of every round the ladder runs whatever the Newton trials did (the self-test's [v36z ptcfb]
    exercise of the hand-over in the whole engine; never for a member). v36t's sub-options are kept verbatim (--nk-ptc-one-cache included). [v36z ptcfb]
engine_collapse_v36y.py (busi, analyst; #331, TOURNAMENT_reading_v1 sec. 4.1: the substep patch merged into the Newton engine lineage;''')))

    # B: the settings
    E.append(("B settings", blk(r'''
OFF = {"on": False, "test": False, "pinned": None, "rec": None}   # v36m offload: --nk-host-release (on | test): the host release of finished rounds' evaluations
'''), blk(r'''
OFF = {"on": False, "test": False, "pinned": None, "rec": None}   # v36m offload: --nk-host-release (on | test): the host release of finished rounds' evaluations
PTC = {"on": False, "fallback": False, "test": False, "delta0": 1.0, "delta_min": 1e-3, "delta_max": 1e6, "ladder": 4.0, "pivot": 16.0, "trials": 8, "dx_max": 0.1, "one_cache": False}   # v36z ptcfb: v36t's settings dict plus "fallback" (the ladder behind the Newton search and the plain direction) and "test"; "on" means the ladder is armed (it never replaces the loop in v36z)
''')))

    # C: the failure branch
    E.append(("C failure branch", blk(r'''
        if acc is None and fb_on:   # v36q fallback: no Newton trial fell on the held set: the JVP check, then the plain direction G on the same schedule
            acc = nk_plain_fallback_v36q(a, F, RS, x, Fx, r0, dx, trials, fb_fd, e_, n_, H, HMAX)   # v36q fallback: (omega, x_t, G_t, gap) or None; the Jacobian, eta and norm updates below run on the plain iterate as on a Newton one
        if acc is None:
            RS["by_halving"] = True'''), blk(r'''
        if PTC["test"] and n_ == 0:   # v36z ptcfb: --nk-ptc test: the ladder from the iterate of Newton step 1 whatever its trials did (the hand-over exercised in the whole engine)
            return nk_ptc_fallback_v36z(a, F, x, RS, e_, n_, "test mode (Newton step 1 of the round; its trials %s)" % ("accepted %g" % acc[0] if acc is not None else "all rejected"))   # v36z ptcfb
        if acc is None and fb_on:   # v36q fallback: no Newton trial fell on the held set: the JVP check, then the plain direction G on the same schedule
            acc = nk_plain_fallback_v36q(a, F, RS, x, Fx, r0, dx, trials, fb_fd, e_, n_, H, HMAX)   # v36q fallback: (omega, x_t, G_t, gap) or None; the Jacobian, eta and norm updates below run on the plain iterate as on a Newton one
        if acc is None and PTC["fallback"]:   # v36z ptcfb: the halving search and the plain direction both failed (or the plain fallback is off): v36t's ladder from this iterate, its status the round's
            return nk_ptc_fallback_v36z(a, F, x, RS, e_, n_, ("no Newton trial and no plain trial fell on the held set" if fb_on else "no Newton trial fell on the held set (--nk-plain-fallback off)"))   # v36z ptcfb
        if acc is None:
            RS["by_halving"] = True''')))

    # D: the functions (v36t's verbatim, then the hand-over)
    E.append(("D functions", blk(r'''

def tm_reset():'''), "\n" + funcs + "\n" + blk(r'''


def nk_ptc_fallback_v36z(a, F, x, RS, e_, n_, why):   # v36z ptcfb: the hand-over from nk_loop_v36c's failed Newton step n_ + 1 to v36t's ladder
    # v36z ptcfb: x is the loop's current iterate (the last accepted one, or the round's start), e_ the failed step's json entry (already in
    # v36z ptcfb: RS["steps"] and printed). nk_loop_ptc_v36t starts from x with delta0 (it evaluates F at x once: its own residual, the memo and
    # v36z ptcfb: RS["last"] on x again, as at a round's start), sets RS["by_halving"] by its own rule and appends its steps; its status is returned
    # v36z ptcfb: as the round's ("converged" verbatim, else with a clause naming the hand-over).
    s0 = len(RS["steps"])   # v36z ptcfb
    e_["ptc_fallback"] = {"from_newton_step": n_ + 1, "why": why, "delta0": PTC["delta0"], "ladder_steps": None, "status": None}   # v36z ptcfb
    print("  v36z ptcfb %s: round %d Newton step %d: %s; the pseudo-transient ladder (nk_loop_ptc_v36t) runs from this iterate, delta0 %g" % (   # v36z ptcfb
        RS.get("v36t_record", "record ?"), RS["round"], n_ + 1, why, PTC["delta0"]), flush=True)   # v36z ptcfb
    st = nk_loop_ptc_v36t(a, F, x, RS)   # v36z ptcfb
    for q in RS["steps"][s0:]:   # v36z ptcfb
        q["ptc_fallback_from_newton_step"] = n_ + 1   # v36z ptcfb
    e_["ptc_fallback"]["ladder_steps"] = len(RS["steps"]) - s0; e_["ptc_fallback"]["status"] = st   # v36z ptcfb
    print("  v36z ptcfb %s: round %d: the ladder from Newton step %d ended after %d step(s): %s" % (RS.get("v36t_record", "record ?"), RS["round"], n_ + 1, len(RS["steps"]) - s0, st), flush=True)   # v36z ptcfb
    return st if st == "converged" else (st + " [v36z: the ladder from Newton step %d]" % (n_ + 1))   # v36z ptcfb


def tm_reset():''')))

    # E: the options entry
    E.append(("E options entry", blk(r'''
            "end_above": a.end_above,   # v36q fallback
'''), blk(r'''
            "end_above": a.end_above,   # v36q fallback
''') + optline + blk(r'''
            **({"ptc_fallback": "on (v36z: v36t's ladder from the iterate where the Newton halving search and the plain direction both fail, delta0 %g%s)" % (PTC["delta0"], ("; test mode" if PTC["test"] else ""))} if PTC["fallback"] else {}),   # v36z ptcfb
''')))

    # F: the confirming step's release (one cache)
    E.append(("F confirm release", blk(r'''
        if nk["best"] is not None:
            ent.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))'''), blk(r'''
        if nk["best"] is not None:
''') + conf + blk(r'''
            ent.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))''')))

    # G: the hooks
    E.append(("G hooks", blk(r'''
            F.v36m_drop = v36m_drop   # v36m offload
    t0 = time.time()'''), blk(r'''
            F.v36m_drop = v36m_drop   # v36m offload
''') + hooks + blk(r'''
    t0 = time.time()''')))

    # H: the record label
    E.append(("H record label", blk(r'''
        RS["v36q_record"] = "T/rs %.3f %s" % (T / a.rs, name)   # v36q fallback
    if NSR["on"]:'''), blk(r'''
        RS["v36q_record"] = "T/rs %.3f %s" % (T / a.rs, name)   # v36q fallback
''') + lab + blk(r'''
    if NSR["on"]:''')))

    # I: the options
    E.append(("I argparse", blk(r'''
    ap.add_argument("--nk-plain-fallback", dest="nk_plain_fallback", default="off", choices=["off", "on"], help="(v36q) in the held-set Newton loop, when no trial of the halving search lowers the residual, the plain direction G = F(x) - x on the same schedule (omega 1, 1/2, ...), the first fall taken, with a JVP check line; needs --nk-set resolve; off: bitwise v36m")   # v36q fallback
'''), blk(r'''
    ap.add_argument("--nk-plain-fallback", dest="nk_plain_fallback", default="off", choices=["off", "on"], help="(v36q) in the held-set Newton loop, when no trial of the halving search lowers the residual, the plain direction G = F(x) - x on the same schedule (omega 1, 1/2, ...), the first fall taken, with a JVP check line; needs --nk-set resolve; off: bitwise v36m")   # v36q fallback
    ap.add_argument("--nk-ptc", dest="nk_ptc", default="off", choices=["off", "fallback", "test"], help="(v36z) fallback: when at a Newton step the halving search and the plain direction (--nk-plain-fallback on) both fail to lower the residual on the held set, v36t's pseudo-transient ladder (nk_loop_ptc_v36t) from the current iterate with delta0 = --nk-ptc-delta0; the Newton loop is otherwise unchanged; needs --nk-set resolve; test: fallback, and the ladder at Newton step 1 of every round (self-test only); off: bitwise v36y")   # v36z ptcfb
''') + subargs))

    # J: the validation
    E.append(("J validation", blk(r'''
    if a.nk_plain_fallback == "on" and a.nk_set != "resolve":   # v36q fallback
        ap.error("version 36q: --nk-plain-fallback acts in the set-resolving solve's loop (nk_loop_v36c): --nk-set resolve is required")   # v36q fallback
'''), blk(r'''
    if a.nk_plain_fallback == "on" and a.nk_set != "resolve":   # v36q fallback
        ap.error("version 36q: --nk-plain-fallback acts in the set-resolving solve's loop (nk_loop_v36c): --nk-set resolve is required")   # v36q fallback
    if a.nk_ptc != "off":   # v36z ptcfb
        if a.nk_set != "resolve":   # v36z ptcfb
            ap.error("version 36z: --nk-ptc acts in the set-resolving solve's loop (nk_loop_v36c's failure branch): --nk-set resolve is required")   # v36z ptcfb
''') + rng + occ + blk(r'''
        PTC.update({"on": True, "fallback": True, "test": (a.nk_ptc == "test"), "delta0": a.nk_ptc_delta0, "delta_min": a.nk_ptc_delta_min, "delta_max": a.nk_ptc_delta_max, "ladder": a.nk_ptc_ladder, "pivot": a.nk_ptc_pivot, "trials": a.nk_ptc_trials, "dx_max": a.nk_ptc_dx_max, "one_cache": (a.nk_ptc_one_cache == "on")})   # v36z ptcfb
''')))
    return E


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text, E):
    for name, old, new in E:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100]))
        text = text.replace(old, new)
    return text


def reverse(text, E):
    for name, old, new in reversed(E):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check(E, T):
    """every line a NEW block adds or changes carries MARK, or carries TMARK and occurs verbatim in v36t (as a whole line); a line of OLD is not
    new (nor one moved out from behind the opening triple quote); blank lines are neutral."""
    tl = set(T.splitlines())
    bad, n_marked, n_v36t = [], 0, 0
    for name, old, new in E:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}
        for ln in new.splitlines():
            if ln in olds or ln.strip() == "":
                continue
            if MARK in ln:
                n_marked += 1
            elif ln in tl and (TMARK in ln):
                n_v36t += 1
            elif ln in tl and name == "D functions":
                n_v36t += 1   # a line of v36t's verbatim functions without the marker (none expected; counted)
            else:
                bad.append((name, ln))
    return bad, n_marked, n_v36t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_collapse_v36y.py (build) or engine_collapse_v36z.py (--reverse)")
    ap.add_argument("--ptc", required=True, help="engine_collapse_v36t.py (md5 40249674): the verbatim excerpts")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--expect-ptc-md5", default=PTC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    tb = open(a.ptc, "rb").read()
    print("v36t %s: md5 %s" % (a.ptc, md5(tb)))
    if a.expect_ptc_md5 and not md5(tb).startswith(a.expect_ptc_md5):
        raise SystemExit("v36t's md5 does not start with %s; nothing written" % a.expect_ptc_md5)
    T = tb.decode("utf-8")
    E = edits(T)
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text, E).encode("utf-8")
        if a.dst and not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if (a.check_only or not a.dst) else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not engine_collapse_v36y.py; nothing written" % a.expect_src_md5)
    if MARK in text or TMARK in text:
        raise SystemExit("the source already carries %r or %r; nothing written" % (MARK, TMARK))
    for name, old, new in E:
        print("  anchor %-20s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_marked, n_v36t = marker_check(E, T)
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln[:160]))
        raise SystemExit("new lines without the marker")
    print("every new or changed line carries %r (%d lines) or is a v36t line, verbatim, carrying %r (%d lines)" % (MARK, n_marked, TMARK, n_v36t))
    built = forward(text, E)
    if 'return nk_loop_ptc_v36t(a, F, x0, RS)' in built:
        raise SystemExit("v36t's head dispatch (PTC as the loop) is in the build; nothing written")
    out = built.encode("utf-8")
    back = reverse(built, E).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst or "engine_collapse_v36z.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36z.py")
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
        raise SystemExit("the written file does not reverse to the input")
    py_compile.compile(a.dst, doraise=True)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))
    print("check: grep -c '%s' %s  ->  %d; grep -c '%s'  ->  %d" % (MARK, a.dst, sum(1 for l in built.splitlines() if MARK in l), TMARK, sum(1 for l in built.splitlines() if TMARK in l)))


if __name__ == "__main__":
    main()
