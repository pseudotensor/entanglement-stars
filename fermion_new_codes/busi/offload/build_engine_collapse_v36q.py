#!/usr/bin/env python
"""build_engine_collapse_v36q.py (busi, reviewer A; S4_128_nk_stop_ruling_v1 section 3, the plain-direction fallback of the Newton line
search): engine_collapse_v36q.py from engine_collapse_v36m.py (md5 4535a6f7...) by marked replacement blocks, in the form of
build_engine_collapse_v36m.py.
Rules: the source's md5 must begin with 4535a6f7 (--expect-src-md5); every OLD anchor is copied verbatim from v36m and occurs exactly once
(checked); every line a NEW block adds or changes carries "v36q fallback" (checked; inside the module docstring as "[v36q fallback]"); the
reverse pass (NEW -> OLD, each NEW once) restores v36m byte for byte (checked in memory before writing and again on the written file;
available as --reverse); the output compiles (py_compile).
    python build_engine_collapse_v36q.py [--src .../engine_collapse_v36m.py] [--dst .../engine_collapse_v36q.py] [--check-only]
    python build_engine_collapse_v36q.py --reverse --src .../engine_collapse_v36q.py --dst .../restored.py --expect .../engine_collapse_v36m.py

The patch (defaults bitwise v36m: with --nk-plain-fallback off, the default, nothing of it runs):
  A  the module docstring's lineage line;
  B  v32_options: the record's "options" entry carries "plain_fallback" with the option on;
  C  nk_loop_v36c, the loop's head: fb_on (the option), fb_fd (the smallest Newton trial's G for the JVP check's finite difference);
  D  nk_loop_v36c, the trial loop: fb_fd kept (the smallest non-crossing trial when there is one);
  E  nk_loop_v36c, after the step's line: with the option on and no Newton trial accepted, nk_plain_fallback_v36q runs; its accepted plain
     trial becomes acc, so the loop's own jac.update, eta and norm updates (and gap_x, RS["last"] through F) run on it as on a Newton iterate;
     when it accepts nothing the status says so and RS["by_halving"] is set as before;
  F  nk_plain_fallback_v36q (before tm_reset): the JVP check and the plain direction G = F(x) - x on the same halving schedule;
  G  nk_stage_v32: RS["v36q_record"], the record and solve the fallback's line names (option on only);
  H  the option --nk-plain-fallback off | on;  I  its validation (needs --nk-set resolve: the fallback lives in the held-set loop).
"""
import argparse
import hashlib
import os
import py_compile

MARK = "v36q fallback"
SRC_MD5_PREFIX = "4535a6f7"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

# A: the lineage (the first line of the module docstring)
EDITS.append(("A lineage", blk(r'''
"""engine_collapse_v36m.py (busi, reviewer A; reorientation 5, the host-offload Newton engine; built by build_engine_collapse_v36m.py) [v36m offload]'''), blk(r'''
"""engine_collapse_v36q.py (busi, reviewer A; S4_128_nk_stop_ruling_v1 section 3, the plain-direction fallback of the Newton line search; built by build_engine_collapse_v36q.py over v36m, md5 4535a6f7) [v36q fallback]
version 36q = version 36m with one option, off by default, so that the defaults are version 36m bitwise (with it off nothing of it runs): [v36q fallback]
--nk-plain-fallback on (needs --nk-set resolve): in the held-set Newton loop (nk_loop_v36c), when no trial of the halving search lowers the [v36q fallback]
    residual on the held set, the step falls back to the plain direction G = F(x) - x (the plain iteration's step is x + G: run() applies N_app, [v36q fallback]
    v_app = Nn, vn, the docstring's a = 1, so omega = 1), tried on the same schedule as the Newton trials (omega = 1, 1/2, ..., 2^-H, continued [v36q fallback]
    to 2^-12 while the trials cross; a crossing trial is not taken and is the far-side start when none is kept yet, marked "plain"); the first [v36q fallback]
    trial that lowers the residual becomes the next iterate and the loop's own Jacobian (jac.update), forcing-term (eta), norm and gap updates [v36q fallback]
    run on it as on a Newton iterate (the memo holds its evaluation, as after an accepted Newton trial); the solve ends only when neither [v36q fallback]
    direction lowers the residual (the status says so; RS["by_halving"] as before). Before the plain trials the JVP check runs: the slope of |G|_2 [v36q fallback]
    along the failed Newton direction from the finite difference the smallest non-crossing trial already paid for, (|G(x + t dx)|_2 - |G(x)|_2)/t, [v36q fallback]
    against the exact J.v's, G.(J dx)/|G|_2 with J dx = F.jvp(x, dx) (one evaluation of F at the iterate, since the memo holds the last trial, [v36q fallback]
    plus the product's two field solves; under the host release that evaluation leaves the memo before the plain trials, so at most two [v36q fallback]
    evaluation caches are on the device, as in v36m), and against the Newton value -|G|_2, with |FD - exact|_2/|FD|_2 of the two J dx vectors and [v36q fallback]
    the max-norm slope (r_t - r0)/t the line search saw in units of r0. One line per fallback prints the record, the round and Newton step, the [v36q fallback]
    residuals, the check and the plain trials; the step's json entry carries it all under "plain_fallback" (its Newton "accepted_step" stays None [v36q fallback]
    there, so the record line's step list shows none where the plain direction carried the step). A rejected plain trial leaves the memo as a [v36q fallback]
    rejected Newton trial does. The record's "options" entry carries "plain_fallback" with the option on. Version 36m's docstring follows. [v36q fallback]
engine_collapse_v36m.py (busi, reviewer A; reorientation 5, the host-offload Newton engine; built by build_engine_collapse_v36m.py) [v36m offload]''')))

# B: v32_options: the record's options entry (unchanged with the option off: ** of an empty dict adds nothing)
EDITS.append(("B options entry", blk(r'''
            "end_above": a.end_above}'''), blk(r'''
            "end_above": a.end_above,   # v36q fallback
            **({"plain_fallback": "on (v36q: the plain direction G on the halving schedule when no Newton trial falls on the held set, with the JVP check)"} if getattr(a, "nk_plain_fallback", "off") == "on" else {})}   # v36q fallback''')))

# C: nk_loop_v36c, the loop's head
EDITS.append(("C loop head", blk(r'''
    H = nk_halvings_eff(a); HMAX = 12
    RS["by_halving"] = False'''), blk(r'''
    H = nk_halvings_eff(a); HMAX = 12
    fb_on = getattr(a, "nk_plain_fallback", "off") == "on"   # v36q fallback: --nk-plain-fallback on
    fb_fd = None   # v36q fallback: the smallest Newton trial's (step, G, crosses), a non-crossing one when there is one, for the JVP check's finite difference
    RS["by_halving"] = False''')))

# D: nk_loop_v36c, the trial loop: the smallest trial's G kept for the check
EDITS.append(("D trial kept", blk(r'''
            trials.append({"step": s, "residual": rt, "set": L["set_id"], "crosses": cross, "flips_vs_held": L["flips"], "gap": L["gap"]})
            if cross:'''), blk(r'''
            trials.append({"step": s, "residual": rt, "set": L["set_id"], "crosses": cross, "flips_vs_held": L["flips"], "gap": L["gap"]})
            if fb_on and (not cross or fb_fd is None or fb_fd[2]):   # v36q fallback: the smallest trial's G (a non-crossing one when there is one) for the finite difference
                fb_fd = (s, Ft, cross)   # v36q fallback
            if cross:''')))

# E: nk_loop_v36c, after the step's line: the fallback, and the status when neither direction falls
EDITS.append(("E fallback call", blk(r'''
        if acc is None:
            RS["by_halving"] = True
            return "the residual did not fall on the held set in the halvings (to 2^-%d while the trials crossed) at Newton step %d (the solve ends there)" % (
                (len(trials) - 1), n_ + 1)
        s, x, Fx, gap_x = acc'''), blk(r'''
        if acc is None and fb_on:   # v36q fallback: no Newton trial fell on the held set: the JVP check, then the plain direction G on the same schedule
            acc = nk_plain_fallback_v36q(a, F, RS, x, Fx, r0, dx, trials, fb_fd, e_, n_, H, HMAX)   # v36q fallback: (omega, x_t, G_t, gap) or None; the Jacobian, eta and norm updates below run on the plain iterate as on a Newton one
        if acc is None:
            RS["by_halving"] = True
            return "the residual did not fall on the held set in the halvings (to 2^-%d while the trials crossed) at Newton step %d (the solve ends there)" % (
                (len(trials) - 1), n_ + 1) + ((" [v36q: nor along the plain direction, %d trials]" % len(e_["plain_fallback"]["trials"])) if fb_on else "")   # v36q fallback
        s, x, Fx, gap_x = acc''')))

# F: the fallback itself, before tm_reset
EDITS.append(("F fallback function", blk(r'''
def tm_reset():'''), blk(r'''
def nk_plain_fallback_v36q(a, F, RS, x, Fx, r0, dx, trials, fb_fd, e_, n_, H, HMAX):   # v36q fallback: the Newton line search's fallback (S4_128_nk_stop_ruling_v1 section 3)
    # v36q fallback: called by nk_loop_v36c when no trial of the halving search lowered the residual on the held set at Newton step n_ + 1 (x the
    # v36q fallback: iterate, Fx = G(x) its residual vector, r0 = max|G|, dx the failed Newton direction, trials the step's Newton trials, fb_fd the
    # v36q fallback: smallest trial's (step, G, crosses), e_ the step's json entry).
    # v36q fallback: (1) The JVP check: the slope of |G|_2 along dx from the finite difference that trial paid for, (|G(x + t dx)|_2 - |G(x)|_2)/t,
    # v36q fallback: against the exact J.v's, G.(J dx)/|G|_2 with J dx = F.jvp(x, dx) (one evaluation of F at x, since the memo holds the last trial,
    # v36q fallback: plus the product's two field solves; nothing when F has no jvp; a failure inside it is reported and the fallback goes on), and
    # v36q fallback: against the Newton value -|G|_2; also |FD - exact|_2/|FD|_2 of the two J dx vectors and the max-norm slope (r_t - r0)/t the line
    # v36q fallback: search saw, in units of r0. Under the host release the check's evaluation leaves the memo before the plain trials.
    # v36q fallback: (2) The plain direction G = F(x) - x (v36m's F returns G; the plain iteration's step is x + G, the docstring's a = 1: omega = 1) on
    # v36q fallback: the same schedule as the Newton trials (omega, omega/2, ..., omega 2^-H, continued to 2^-HMAX while the trials cross) under the
    # v36q fallback: held-set rule (a crossing trial is not taken and is RS["far"] when none is kept yet, marked "plain"; a rejected trial leaves the
    # v36q fallback: memo under the host release); the first trial that lowers the residual is returned as (omega, x_t, G_t, gap) for the loop's own
    # v36q fallback: Jacobian (jac.update), eta and norm updates, None when none falls. e_["plain_fallback"] carries the check and the trials; one
    # v36q fallback: line prints the record, the round and Newton step, the residuals, the check and the trials.
    from numpy.linalg import norm   # v36q fallback
    g2 = float(norm(Fx))   # v36q fallback: |G|_2 at the iterate
    chk = {"G_norm2": g2, "newton_slope": -g2, "fd_step": None, "fd_crosses": None, "fd_slope": None, "fd_slope_maxnorm": None, "exact_slope": None, "fd_vs_exact": None, "exact_error": None}   # v36q fallback
    fd = None   # v36q fallback
    if fb_fd is not None:   # v36q fallback
        t_, Ft_, cr_ = fb_fd   # v36q fallback
        fd = (Ft_ - Fx) / t_   # v36q fallback: the finite-difference J dx
        chk["fd_step"] = float(t_); chk["fd_crosses"] = bool(cr_)   # v36q fallback
        chk["fd_slope"] = (float(norm(Ft_)) - g2) / t_   # v36q fallback
        chk["fd_slope_maxnorm"] = (float(np.max(np.abs(Ft_))) - r0) / t_   # v36q fallback
    jvp_ = getattr(F, "jvp", None)   # v36q fallback
    if jvp_ is not None:   # v36q fallback
        try:   # v36q fallback
            jd = np.asarray(jvp_(np.asarray(x, dtype=float), np.asarray(dx, dtype=float)), dtype=float).flatten()   # v36q fallback: the exact J dx
            chk["exact_slope"] = (float(np.dot(Fx, jd)) / g2) if g2 > 0.0 else 0.0   # v36q fallback
            if fd is not None:   # v36q fallback
                chk["fd_vs_exact"] = float(norm(fd - jd)) / max(float(norm(fd)), 1e-300)   # v36q fallback
        except Exception as ex:   # v36q fallback: a failed field solve inside the exact J.v, or the like: reported, and the fallback goes on
            chk["exact_error"] = str(ex)[:120]   # v36q fallback
        if OFF["on"]:   # v36q fallback: the check's evaluation of x leaves the memo before the plain trials (two caches on the device at most, as in v36m)
            memo_drop_v36m(F)   # v36q fallback
    omega = 1.0   # v36q fallback: the plain iteration's mixing (run(): N_app, v_app = Nn, vn, the docstring's a = 1; --mix damped's a cannot be set with --mix newton)
    w, ptrials, acc, crossed_last = omega, [], None, False   # v36q fallback
    e_["plain_fallback"] = {"omega": omega, "trials": ptrials, "accepted_step": None, "jvp_check": chk}   # v36q fallback
    for h_ in range(HMAX + 1):   # v36q fallback
        if h_ > H and not crossed_last:   # v36q fallback
            break   # v36q fallback
        xt = x + w * Fx   # v36q fallback: the plain step x + omega G
        Ft = np.asarray(F(xt), dtype=float).flatten()   # v36q fallback
        rt = float(np.max(np.abs(Ft)))   # v36q fallback
        L = RS["last"]   # v36q fallback
        cross = not L["same"]   # v36q fallback
        ptrials.append({"step": w, "residual": rt, "set": L["set_id"], "crosses": cross, "flips_vs_held": L["flips"], "gap": L["gap"]})   # v36q fallback
        if cross:   # v36q fallback
            if RS["far"] is None:   # v36q fallback
                RS["far"] = {"x": xt.copy(), "set_id": L["set_id"], "residual": rt, "newton_step": n_ + 1, "step": w, "plain": True}   # v36q fallback
            crossed_last = True   # v36q fallback
        else:   # v36q fallback
            crossed_last = False   # v36q fallback
            if rt < r0:   # v36q fallback
                acc = (w, xt, Ft, L["gap"])   # v36q fallback
                break   # v36q fallback
        if OFF["on"]:   # v36q fallback: this plain trial is rejected (crossing, or no fall): its evaluation leaves the memo, as a rejected Newton trial's does
            memo_drop_v36m(F)   # v36q fallback
        w *= 0.5   # v36q fallback
    e_["plain_fallback"]["accepted_step"] = (acc[0] if acc is not None else None)   # v36q fallback
    q_ = lambda v, f: (f % v) if v is not None else "n/a"   # v36q fallback
    ratio_ = lambda v, d: (v / d) if (v is not None and d != 0.0) else None   # v36q fallback
    chk_s = "JVP check at the smallest trial (step %s%s): |G|_2 %.3e, slope of |G|_2 along dx: FD %s, exact %s, Newton -|G|_2 %+.3e (ratios to Newton: FD %s, exact %s), |FD - exact|/|FD| %s%s, max-norm slope %s = %s r0" % (   # v36q fallback
        q_(chk["fd_step"], "%g"), (", crosses" if chk["fd_crosses"] else ""), g2, q_(chk["fd_slope"], "%+.3e"), q_(chk["exact_slope"], "%+.3e"), -g2,   # v36q fallback
        q_(ratio_(chk["fd_slope"], -g2), "%.3f"), q_(ratio_(chk["exact_slope"], -g2), "%.3f"), q_(chk["fd_vs_exact"], "%.2e"),   # v36q fallback
        ((" (exact J.v failed: %s)" % chk["exact_error"]) if chk["exact_error"] else ("" if jvp_ is not None else " (no exact J.v)")),   # v36q fallback
        q_(chk["fd_slope_maxnorm"], "%+.3e"), q_(ratio_(chk["fd_slope_maxnorm"], r0), "%+.2f"))   # v36q fallback
    print("  v36q fallback %s: round %d Newton step %d, held set %s, r0 %.2e, no Newton trial fell (%d trials, smallest residual %.2e); %s; plain direction omega %g: trials %s; accepted %s" % (   # v36q fallback
        RS.get("v36q_record", "record ?"), RS["round"], n_ + 1, e_["held_set"], r0, len(trials), min(q["residual"] for q in trials), chk_s, omega,   # v36q fallback
        " ".join("[%g %.2e set %d%s gap %.2e]" % (q["step"], q["residual"], q["set"], (" crosses, %d flips" % q["flips_vs_held"]) if q["crosses"] else "", q["gap"]) for q in ptrials),   # v36q fallback
        ("%g (residual %.2e)" % (acc[0], float(np.max(np.abs(acc[2]))))) if acc is not None else "none (the solve ends there)"), flush=True)   # v36q fallback
    return acc   # v36q fallback


def tm_reset():''')))

# G: nk_stage_v32: the record and solve the fallback's line names
EDITS.append(("G record label", blk(r'''
    RS = {"held": None, "held_m": None, "first_m": None, "sets": {}, "evals": [], "steps": [], "far": None, "round": 0, "last": None, "by_halving": False}
    if NSR["on"]:'''), blk(r'''
    RS = {"held": None, "held_m": None, "first_m": None, "sets": {}, "evals": [], "steps": [], "far": None, "round": 0, "last": None, "by_halving": False}
    if getattr(a, "nk_plain_fallback", "off") == "on":   # v36q fallback: the record and the solve the fallback's line names
        RS["v36q_record"] = "T/rs %.3f %s" % (T / a.rs, name)   # v36q fallback
    if NSR["on"]:''')))

# H: the option
EDITS.append(("H argparse", blk(r'''
    ap.add_argument("--nk-host-release", dest="nk_host_release", default="off", choices=["off", "on", "test"], help="(v36m) finished rounds' eigendecompositions and references to pinned host memory, the memo emptied of rejected trials; bitwise; test: the kept round round-trips too")   # v36m offload'''), blk(r'''
    ap.add_argument("--nk-host-release", dest="nk_host_release", default="off", choices=["off", "on", "test"], help="(v36m) finished rounds' eigendecompositions and references to pinned host memory, the memo emptied of rejected trials; bitwise; test: the kept round round-trips too")   # v36m offload
    ap.add_argument("--nk-plain-fallback", dest="nk_plain_fallback", default="off", choices=["off", "on"], help="(v36q) in the held-set Newton loop, when no trial of the halving search lowers the residual, the plain direction G = F(x) - x on the same schedule (omega 1, 1/2, ...), the first fall taken, with a JVP check line; needs --nk-set resolve; off: bitwise v36m")   # v36q fallback''')))

# I: the validation
EDITS.append(("I validation", blk(r'''
        OFF["on"] = True; OFF["test"] = (a.nk_host_release == "test")   # v36m offload
    C5.backend(a.gpu)'''), blk(r'''
        OFF["on"] = True; OFF["test"] = (a.nk_host_release == "test")   # v36m offload
    if a.nk_plain_fallback == "on" and a.nk_set != "resolve":   # v36q fallback
        ap.error("version 36q: --nk-plain-fallback acts in the set-resolving solve's loop (nk_loop_v36c): --nk-set resolve is required")   # v36q fallback
    C5.backend(a.gpu)''')))


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
    ap.add_argument("--src", default="engine_collapse_v36m.py")
    ap.add_argument("--dst", default=None, help="default: engine_collapse_v36q.py beside --src")
    ap.add_argument("--reverse", action="store_true", help="src is engine_collapse_v36q.py: restore v36m into dst")
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36q.py" if not a.reverse else "engine_collapse_v36m_restored.py")
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text).encode("utf-8")
        if not a.check_only:
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if a.check_only else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not engine_collapse_v36m.py; nothing written" % a.expect_src_md5)
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
