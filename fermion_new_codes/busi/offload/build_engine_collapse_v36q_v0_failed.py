#!/usr/bin/env python3
"""build_engine_collapse_v36q.py (busi): engine_collapse_v36q.py over engine_collapse_v36m.py (the host release), S4_128_nk_stop_ruling_v1 section 3.

Form: exact OLD -> NEW replacements, every changed or added line marked 'v36q nk'; the reverse replacements must restore v36m byte for byte or nothing
is written. With the defaults (--nk-plain-fallback -1, --nk-jvp-check off) v36q is bitwise v36m; with the member's flags it is bitwise v36m on every
record where the Newton line search accepts a trial at every step, because the only changed path is the 'accepted none' branch of nk_loop_v32.

What this builder cannot see: v36m's source. The two OLD anchors and the loop's names come from v36q_anchors.json beside the engine, written by the
lane from nk_loop_v32 in v36m (PRIM 3.2's precedent). Keys:
  "argparse_halvings": the exact add_argument line for --nk-halvings (the two new flags are added after it, same parser, same indent)
  "stop_line": the exact line (or lines) in nk_loop_v32's 'accepted none' branch that records the stop reason; it carries the literal
               "the residual did not fall on the held set in the halvings"; the fallback is inserted before it, at its indent, inside the Newton loop body
  "names": the engine's names, as visible at the stop line, for
     args (the parsed arguments), x (the iterate), fx (its plain image F(x), in hand from the residual evaluation), r (its residual),
     resid_fn (the trial evaluation, x -> (F(x), residual)), dx (the Newton direction of the failed line search), jvp_fn (the exact JVP, (x, v) -> J v
     on G = F - I, or None), t_min (the smallest trial's step), fx_tmin (that trial's F, or None), r_stop (the solve's stop residual, 0.0 if the stop
     rule is not a residual), nevals (the evaluation counter), log (the step-line logger, a callable of one string).
If the stop sits in a helper rather than in the loop body, the lane raises the fallback's tuple to the loop in the same marked form (the 'continue').
"""
import hashlib, json, pathlib, sys

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "engine_collapse_v36m.py")
DST = SRC.with_name("engine_collapse_v36q.py")
ANCH = SRC.with_name("v36q_anchors.json")
LITERAL = "the residual did not fall on the held set in the halvings"
NAMES = ["args", "x", "fx", "r", "resid_fn", "dx", "jvp_fn", "t_min", "fx_tmin", "r_stop", "nevals", "log"]

FLAG_LINES = [
    ".add_argument('--nk-plain-fallback', type=int, default=-1, help='v36q nk: halvings of the plain direction F(x) - x tried when the Newton line search accepts no trial; -1 off (bitwise v36m)')  # v36q nk",
    ".add_argument('--nk-plain-fallback-steps', type=int, default=12, help='v36q nk: plain steps taken after a fallback acceptance while each lowers the residual')  # v36q nk",
    ".add_argument('--nk-jvp-check', choices=['on', 'off'], default='off', help='v36q nk: at each fallback print the slope of |G| along the failed direction, finite difference against the exact JVP (needs the fallback on)')  # v36q nk",
]

CALL = (
    "{indent}_q = _v36q_plain_fallback({x}, {fx}, {r}, {resid_fn}, {dx}, {jvp_fn}, {t_min}, {fx_tmin}, {r_stop}, {args}.nk_plain_fallback, {args}.nk_plain_fallback_steps, {args}.nk_jvp_check == 'on', {log})  # v36q nk\n"
    "{indent}if _q is not None:  # v36q nk: the plain direction carried the step; the Newton loop continues from its iterate on the held set\n"
    "{indent}    {x}, {fx}, {r}, _qn = _q  # v36q nk\n"
    "{indent}    {nevals} += _qn  # v36q nk\n"
    "{indent}    continue  # v36q nk\n"
)

FALLBACK_SRC = '''
def _v36q_plain_fallback(x, fx, r, resid_fn, dx, jvp_fn, t_min, fx_tmin, r_stop, k, steps, check, log):  # v36q nk
    """v36q nk: the plain direction F(x) - x as the Newton line search's fallback (S4_128_nk_stop_ruling_v1 section 3).
    Called in the 'accepted none' branch. Tries omega = 1, 1/2, ..., 2^-k, accepts the first trial that lowers the residual, then takes plain
    steps at that omega while each lowers it (at most `steps`, or to r_stop). Returns (x, F(x), residual, evaluations) or None (no trial fell).
    With check on, prints the slope of |G| along the failed Newton direction from the finite difference the smallest trial already paid for
    against the exact JVP's (Newton gives -1 in |G|^2 units) and |FD - exact|/|FD|. k < 0: off, nothing evaluated."""  # v36q nk
    if k < 0:  # v36q nk
        return None  # v36q nk
    import math  # v36q nk
    def _dot(a, b):  # v36q nk
        return float((a.conj() * b).real.sum())  # v36q nk
    def _norm(a):  # v36q nk
        return math.sqrt(max(_dot(a, a), 0.0))  # v36q nk
    n = 0  # v36q nk
    G = fx - x  # v36q nk
    if check and jvp_fn is not None and fx_tmin is not None and t_min:  # v36q nk
        fd = ((fx_tmin - (x + t_min * dx)) - G) / t_min  # v36q nk
        jd = jvp_fn(x, dx)  # v36q nk
        g2 = max(_dot(G, G), 1e-300)  # v36q nk
        log("  v36q jvp check: slope of |G| along the failed direction in |G|^2 units, FD %+.3f exact %+.3f (Newton gives -1); |FD - exact|/|FD| %.2e"  # v36q nk
            % (_dot(G, fd) / g2, _dot(G, jd) / g2, _norm(fd - jd) / max(_norm(fd), 1e-300)))  # v36q nk
    best = None  # v36q nk
    for j in range(k + 1):  # v36q nk
        w = 0.5 ** j  # v36q nk
        xt = x + w * G  # v36q nk
        ft, rt = resid_fn(xt)  # v36q nk
        n += 1  # v36q nk
        if rt < r:  # v36q nk
            log("  v36q accepted plain omega %g: residual %.2e -> %.2e" % (w, r, rt))  # v36q nk
            best = (xt, ft, rt, w)  # v36q nk
            break  # v36q nk
        log("  v36q plain trial omega %g: residual %.2e not below %.2e" % (w, rt, r))  # v36q nk
    if best is None:  # v36q nk
        return None  # v36q nk
    xt, ft, rt, w = best  # v36q nk
    for s in range(steps):  # v36q nk
        if rt <= r_stop:  # v36q nk
            break  # v36q nk
        xn = xt + w * (ft - xt)  # v36q nk
        fn, rn = resid_fn(xn)  # v36q nk
        n += 1  # v36q nk
        if not (rn < rt):  # v36q nk
            log("  v36q plain step %d: residual %.2e not below %.2e; back to Newton" % (s + 1, rn, rt))  # v36q nk
            break  # v36q nk
        xt, ft, rt = xn, fn, rn  # v36q nk
        log("  v36q plain step %d: residual %.2e" % (s + 1, rt))  # v36q nk
    return xt, ft, rt, n  # v36q nk
'''


def _lead(s):
    return s[: len(s) - len(s.lstrip())]


def main():
    if not SRC.exists() or not ANCH.exists():
        sys.exit("v36q: need %s and %s" % (SRC, ANCH))
    src = SRC.read_text()
    a = json.loads(ANCH.read_text())
    missing = [k for k in ("argparse_halvings", "stop_line", "names") if k not in a]
    missing += ["names." + k for k in NAMES if k not in a.get("names", {})]
    if missing:
        sys.exit("v36q: anchors missing: " + ", ".join(missing))
    if "--nk-halvings" not in a["argparse_halvings"] or ".add_argument" not in a["argparse_halvings"]:
        sys.exit("v36q: argparse_halvings is not the --nk-halvings add_argument line")
    if LITERAL not in a["stop_line"]:
        sys.exit("v36q: stop_line does not carry the literal %r" % LITERAL)
    if "nk_loop_v32" not in src:
        sys.exit("v36q: %s has no nk_loop_v32" % SRC.name)
    for k in ("argparse_halvings", "stop_line"):
        c = src.count(a[k])
        if c != 1:
            sys.exit("v36q: anchor %s occurs %d times in %s (needs exactly 1)" % (k, c, SRC.name))
    parser = a["argparse_halvings"].split(".add_argument")[0].strip()
    ind_arg = _lead(a["argparse_halvings"])
    new_flags = a["argparse_halvings"] + "".join("\n" + ind_arg + parser + f for f in FLAG_LINES)
    new_call = CALL.format(indent=_lead(a["stop_line"]), **a["names"]) + a["stop_line"]
    reps = [(a["argparse_halvings"], new_flags), (a["stop_line"], new_call)]
    out = src
    for old, new in reps:
        out = out.replace(old, new, 1)
    tail = "\n\n" + FALLBACK_SRC
    out += tail
    back = out[: -len(tail)]
    for old, new in reversed(reps):
        if back.count(new) != 1:
            sys.exit("v36q: reverse anchor ambiguous for %r; nothing written" % old[:60])
        back = back.replace(new, old, 1)
    if back != src:
        sys.exit("v36q: the reverse replacements do not restore %s byte for byte; nothing written" % SRC.name)
    try:
        compile(out, str(DST), "exec")
    except SyntaxError as e:
        sys.exit("v36q: the result does not compile (%s); check the anchors' indentation and names; nothing written" % e)
    DST.write_text(out)
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
    marked = sum(1 for ln in out.splitlines() if "v36q nk" in ln)
    print("wrote %s (md5 %s) over %s (md5 %s): %d marked lines, 2 anchors replaced, 1 function appended; defaults --nk-plain-fallback -1 --nk-jvp-check off are bitwise v36m"
          % (DST.name, md5(out), SRC.name, md5(src), marked))
    print("member flags: the route's NK string plus --nk-plain-fallback 2 --nk-plain-fallback-steps 12 --nk-jvp-check on; gate Q0: the v36c round lines and record lines 0.000 to 2.625 identical to the v36m log's")


if __name__ == "__main__":
    main()
