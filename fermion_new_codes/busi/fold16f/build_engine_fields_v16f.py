#!/usr/bin/env python
"""build_engine_fields_v16f.py (busi, analyst; ReviewA_s4fail_v1 sec. 2, inbox #444): engine_fields_v16f.py = engine_fields_v16d.py (md5
7a69faad...) + the FOLD EXIT, the near-tangency rule v16's docstring already states ("a vanishing U' before s = 0 is a near-tangency, counted,
the integration returning to r there"), made able to fire. Ruled a fix to be gated, not a new field rule; the file name v16f only names the build.
Three marked insertion blocks:
  FOLD-def    before "def integrate_UL(": the helper _fold_exit_v16f;
  FOLD-exit   in integrate_UL, inside the branch where the DOWNWARD s = sqrt(U) stretch has failed (v16d returned info["failed"] there):
              the r-integration is re-run from the switch event (r_e, U = U_sw, L) with two terminal events, U' = 0 crossed upward (a positive
              minimum of U: the fold of r(s) at s_m = sqrt(U_m) > 0) and U = 0 crossed downward; when the minimum comes first with U > 0, that
              r-segment is kept, counted as a near-tangency, and the loop goes on in r from the minimum; otherwise v16d's failure is returned
              unchanged;
  FOLD-gauge  in solve_constraints_ivp, after the gauge dict: info["gauge"]["fold_exits"] = [[r_e, r_m, U_m/U_scale], ...] only when one fired.
Every other solve runs v16d's lines exactly: the new code is reached only where v16d's downward stretch had already failed (the run's end), so
the build is bitwise to v16d wherever the fold is not met. Every new line lies between a "# >>> v16f FOLD-<X>" and a "# <<< v16f FOLD-<X>"
line (checked); the reverse pass restores v16d byte for byte (in memory and on the written file; --reverse --expect); it compiles; the file
is never overwritten. Deployed as engine_fields_v16.py in a NEW code dir (like v16c and v16d).
    python build_engine_fields_v16f.py --src .../engine_fields_v16d.py [--dst .../engine_fields_v16f.py] [--check-only]
    python build_engine_fields_v16f.py --reverse --src .../engine_fields_v16f.py --dst .../restored.py --expect .../engine_fields_v16d.py
"""
import argparse
import hashlib
import os

MARK = "v16f FOLD"
SRC_MD5_PREFIX = "7a69faad"

OLD_DEF = "\n\ndef integrate_UL("
NEW_DEF = '''

# >>> v16f FOLD-def: engine_fields_v16f = engine_fields_v16d + the fold exit (build_engine_fields_v16f.py; ReviewA_s4fail_v1 sec. 2).
# A downward approach (U crossing U_sw = 1e-6 U_scale with U' < 0) is integrated in s = sqrt(U) to s = 0. When U instead has a positive
# minimum U_m < U_sw, r(s) folds at s_m = sqrt(U_m) > 0 (dr/ds = 2s/U' -> infinity with s > 0), real s cannot pass it and DOP853 fails at
# roundoff; ev_flat (a sign change of U') never fires at a smooth fold. Only then (the stretch failed) _fold_exit_v16f re-runs the
# r-integration from the switch event with terminal events U' = 0 upward (the minimum) and U = 0 downward; in r the equations are regular
# at U > 0 (q = -sqrt(U/r) != 0). Returns (r_m, dense solution, nfev, y at r_m) when the minimum comes first with U > 0, else None.
def _fold_exit_v16f(rhs_pos, rho_, J_, fpi, r_e, ye, r_collar, rtol, atol):
    def ev_min(r, y):
        return 2.0 * fpi * r * r * (float(rho_(r)) - np.sqrt(max(float(y[0]), 0.0) / r) * float(J_(r)))
    ev_min.terminal = True; ev_min.direction = 1

    def ev_zero(r, y):
        return y[0]
    ev_zero.terminal = True; ev_zero.direction = -1
    try:
        pr = solve_ivp(rhs_pos, (r_e, r_collar), np.array(ye, float), method="DOP853", dense_output=True, rtol=rtol, atol=atol,
                       events=[ev_min, ev_zero])
    except Exception:
        return None
    if not (pr.success and pr.status == 1 and len(pr.t_events[0]) and not len(pr.t_events[1])):
        return None
    r_m = float(pr.t_events[0][0]); y_m = np.array(pr.y_events[0][0], float)
    if not (y_m[0] > 0.0 and np.all(np.isfinite(y_m))):
        return None
    return r_m, pr.sol, int(pr.nfev), y_m
# <<< v16f FOLD-def


def integrate_UL('''

OLD_EXIT = ('                st = s_stretch(r_e, float(ye[1]), float(np.sqrt(max(float(ye[0]), 0.0))), 0.0)\n'
            '                if not st.success or len(st.t) < 2:\n'
            '                    return segs, "the downward stretch at r = %.6g failed: %s" % (r_e, st.message), counts, zeros\n')
NEW_EXIT = ('                st = s_stretch(r_e, float(ye[1]), float(np.sqrt(max(float(ye[0]), 0.0))), 0.0)\n'
            '                if not st.success or len(st.t) < 2:\n'
            '                    # >>> v16f FOLD-exit: the stretch met a fold (a positive minimum below the switch): back to r there (FOLD-def)\n'
            '                    fx_ = _fold_exit_v16f(rhs_pos, rho_, J_, fpi, r_e, ye, r_collar, rtol, atol)\n'
            '                    if fx_ is not None:\n'
            '                        segs.append((r_e, fx_[0], fx_[1], 1, fx_[2])); counts["near_tangencies"] += 1\n'
            '                        counts.setdefault("fold_exits", []).append([r_e, fx_[0], float(fx_[3][0]) / max(U_scale, 1e-300)])\n'
            '                        r = fx_[0]; y = fx_[3]\n'
            '                        continue\n'
            '                    # <<< v16f FOLD-exit\n'
            '                    return segs, "the downward stretch at r = %.6g failed: %s" % (r_e, st.message), counts, zeros\n')

OLD_GAUGE = '             "U_scale": counts["U_scale"], "U_switch": counts["U_switch"]}\n'
NEW_GAUGE = ('             "U_scale": counts["U_scale"], "U_switch": counts["U_switch"]}\n'
             '    # >>> v16f FOLD-gauge: the fold exits, only when one fired (FOLD-exit)\n'
             '    if counts.get("fold_exits"):\n'
             '        gauge["fold_exits"] = counts["fold_exits"]\n'
             '    # <<< v16f FOLD-gauge\n')


def edits():
    return [("FOLD-def", OLD_DEF, NEW_DEF), ("FOLD-exit", OLD_EXIT, NEW_EXIT), ("FOLD-gauge", OLD_GAUGE, NEW_GAUGE)]


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text, E):
    for name, old, new in E:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1)" % (name, c))
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
    """every line a NEW block adds (a line not in its OLD anchor) lies between a '# >>> v16f FOLD-<X>' and a '# <<< v16f FOLD-<X>' line."""
    bad, n_new = [], 0
    for name, old, new in E:
        olds = set(old.splitlines())
        inside = False
        for ln in new.splitlines():
            s = ln.strip()
            if s.startswith("# >>> " + MARK):
                if inside:
                    bad.append((name, "<nested block>"))
                inside = True; n_new += 1; continue
            if s.startswith("# <<< " + MARK):
                inside = False; n_new += 1; continue
            if inside:
                n_new += 1; continue
            if s == "" or ln in olds:
                continue
            bad.append((name, ln))
        if inside:
            bad.append((name, "<block not closed>"))
    return bad, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_fields_v16d.py (build) or engine_fields_v16f.py (--reverse)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    E = edits()
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
        raise SystemExit("the source's md5 does not start with %s: not engine_fields_v16d.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in E:
        print("  anchor %-10s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_new = marker_check(E)
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln[:160]))
        raise SystemExit("new lines outside the marked blocks")
    print("every new line lies in a '# >>> %s-X' ... '# <<< %s-X' block (%d lines incl. the markers)" % (MARK, MARK, n_new))
    built = forward(text, E)
    out = built.encode("utf-8")
    back = reverse(built, E).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst or "engine_fields_v16f.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_fields_v16f.py")
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
    compile(written.decode("utf-8"), a.dst, "exec")
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))


if __name__ == "__main__":
    main()
