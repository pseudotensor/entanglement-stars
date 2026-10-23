#!/usr/bin/env python
"""build_engine_fields_v16b.py (busi, analyst; PERF tracker rows H2-A, H2-B (H2-C), inbox #333/#334/#338): engine_fields_v16b.py from
engine_fields_v16.py (md5 1ffcf5e8...) by marked insertion blocks, the code of the reviewer B review perf_host_newton (box
<internal>/result.md, sec. 2) verbatim.

Options (environment variables, read once at import; unset or "0" = off; off is v16's arithmetic and control flow exactly):
  H2_PW_ONE=1        Option A: the length-one fast path of Piecewise.__call__ and of solve_constraints_ivp's nested Nfin (the final W
                     integration's RHS calls Nfin(r) per scalar r). Keeps v16's array arithmetic and the SciPy dense-output call shape.
  H2_LOCAL_SPLINE=1  Option B: _h2_scalar_spline, a local-support scalar BSpline evaluator (knots t[i-k:i+k+2], coefficients c[i-k:i+1],
                     SciPy's own evaluator) for the float scalar calls of rho_/J_ in integrate_UL and of rhoW in the W integration; vector
                     calls, exact knots, the end intervals, non-finite and unusual inputs go to the original spline; info["src"] untouched.
  H2_RESIDUAL_PAIR=1 Option C (only with --with-c): the two residuals_at resolutions overlapped (one worker thread + the caller); needs
                     H2_CPU_QUOTA (the verified effective cgroup quota).
The review's code is inserted, not rewritten: the replacement methods it gives for Piecewise.__call__ and Nfin are v16's lines with the fast
path in front, so each is an insertion between v16's first lines and the original path; nothing of v16 is deleted or edited (C's two
call lines excepted, with --with-c: a marked if/else around each). info["module"] stays "engine_fields_v16" (the bytewise gate compares it).
Every inserted block opens with a "# >>> v16b H2-<X>" line and closes with "# <<< v16b H2-<X>"; every new line lies between such a pair
(checked); the reverse pass (NEW -> OLD) restores v16 byte for byte (in memory and on the written file; --reverse); it compiles.
    python build_engine_fields_v16b.py --src .../engine_fields_v16.py [--dst .../engine_fields_v16b.py] [--with-c] [--check-only]
    python build_engine_fields_v16b.py --reverse --src .../engine_fields_v16b.py [--with-c] --dst .../restored.py --expect .../engine_fields_v16.py
"""
import argparse
import hashlib
import os
import py_compile

MARK = "v16b H2"
SRC_MD5_PREFIX = "1ffcf5e8"


def blk(s):
    return s[1:] if s.startswith("\n") else s


def edits(with_c):
    E = []
    # S: the module note and the switches (review sec. 2, the common switch patch; C's switch only with --with-c)
    E.append(("S0 import", "import numpy as np\nfrom scipy.integrate import solve_ivp\n", blk(r'''
# >>> v16b H2-S: engine_fields_v16b = engine_fields_v16 + the reviewer B perf_host_newton options (build_engine_fields_v16b.py); all off by default
import os
# <<< v16b H2-S
import numpy as np
from scipy.integrate import solve_ivp
''')))
    sw = blk(r'''
import engine_fields_v12 as F12
# >>> v16b H2-S
_H2_PW_ONE = os.environ.get("H2_PW_ONE", "0") == "1"
_H2_LOCAL_SPLINE = os.environ.get("H2_LOCAL_SPLINE", "0") == "1"
''')
    if with_c:
        sw += '_H2_RESIDUAL_PAIR = os.environ.get("H2_RESIDUAL_PAIR", "0") == "1"\n'
    sw += "# <<< v16b H2-S\n"
    E.append(("S1 switches", "import engine_fields_v12 as F12\n", sw))

    # A: Piecewise.__call__ (review sec. 2 Option A, "Replacement for Piecewise.__call__": the fast path in front of the original path)
    E.append(("A1 Piecewise", "        x = np.abs(r)\n        U = np.empty_like(x); L = np.empty_like(x)\n", blk(r'''
        x = np.abs(r)
        # >>> v16b H2-A

        # H2 option A: same length-one arithmetic and dense-call shape.
        if _H2_PW_ONE and x.shape == (1,):
            if x[0] < self.r0:
                U = self.cc["a"] * x ** 3 + self.cc["b"] * x ** 5
                L = self.cc["L2"] * x ** 2 + self.cc["L4"] * x ** 4
                return U, L

            nseg = len(self.segs)
            if nseg == 1:
                j = 0
            else:
                j = int(np.searchsorted(
                    self.starts, x[0], side="right")) - 1
                j = min(max(j, 0), nseg - 1)
            s0, s1, sol = self.segs[j][:3]
            y = sol(np.clip(x, s0, s1))
            # Match the original independent float64 result arrays.
            return (np.array(y[0], dtype=float, copy=True),
                    np.array(y[1], dtype=float, copy=True))

        # Original path, unchanged.
        # <<< v16b H2-A
        U = np.empty_like(x); L = np.empty_like(x)
''')))
    # A: the nested Nfin (review sec. 2 Option A, "Replacement for the nested Nfin")
    E.append(("A2 Nfin", "        x = np.abs(np.atleast_1d(np.asarray(r, float)))\n        inside = x <= r_collar\n", blk(r'''
        x = np.abs(np.atleast_1d(np.asarray(r, float)))
        # >>> v16b H2-A

        # H2 option A: retain array subtraction and array np.exp.
        if _H2_PW_ONE and x.shape == (1,):
            if x[0] <= r_collar:
                # minimum(x, r_collar) would have the same value here.
                _u, lnN = pw(x)
                return np.exp(lnN - _c)
            if x[0] > r_collar:
                # Original np.where selects exactly 1.0 here.
                return np.ones_like(x)
            # NaN: retain the original path and its behavior.

        # Original path, unchanged.
        # <<< v16b H2-A
        inside = x <= r_collar
''')))

    # B: the module-level evaluator before integrate_UL and the two bindings (review sec. 2 Option B)
    E.append(("B1 evaluator", "\n\ndef integrate_UL(", blk(r'''


# >>> v16b H2-B
def _h2_scalar_spline(spline):
    """H2 option B: local-support BSpline calls for float64 scalars.

    Vector calls, endpoints, exact knots, nonfinite inputs, and unsupported
    spline layouts use the original callable. The original spline is never
    modified, including the one exported through info["src"].
    """
    from scipy.interpolate import BSpline

    if (type(spline) is not BSpline or spline.k != 5
            or spline.axis != 0 or spline.c.ndim != 1
            or spline.extrapolate == "periodic"):
        return spline

    t = spline.t
    c = spline.c
    k = spline.k
    n = len(t) - k - 1
    cache = {}
    last_i = None
    last_spline = None

    def evaluate(x):
        nonlocal last_i, last_spline

        # Keep all vector calls and unusual scalar types on today's path.
        if type(x) not in (float, np.float64):
            return spline(x)

        if last_i is not None and t[last_i] < x < t[last_i + 1]:
            return last_spline(x)

        # Also catches NaN and infinities without changing their treatment.
        if not (t[k] < x < t[n]):
            return spline(x)

        i = int(np.searchsorted(t, x, side="right")) - 1

        # Do not depend on the native evaluator's exact-knot convention.
        if x == t[i]:
            return spline(x)

        local = cache.get(i)
        if local is None:
            local = BSpline.construct_fast(
                t[i - k:i + k + 2],
                c[i - k:i + 1],
                k,
                extrapolate=spline.extrapolate,
                axis=spline.axis,
            )
            cache[i] = local

        last_i, last_spline = i, local
        return local(x)

    return evaluate
# <<< v16b H2-B


def integrate_UL(''')))
    E.append(("B2 rho_ J_", "    rho_ = src.rho; J_ = src.J\n", blk(r'''
    rho_ = src.rho; J_ = src.J
    # >>> v16b H2-B
    if _H2_LOCAL_SPLINE:
        rho_ = _h2_scalar_spline(rho_); J_ = _h2_scalar_spline(J_)
    # <<< v16b H2-B
''')))
    E.append(("B3 rhoW", "    rhoW = src.rho\n", blk(r'''
    rhoW = src.rho
    # >>> v16b H2-B
    if _H2_LOCAL_SPLINE:
        rhoW = _h2_scalar_spline(rhoW)
    # <<< v16b H2-B
''')))

    if with_c:
        # C: review sec. 2 Option C; the two call lines become marked if/else blocks (the else branch is v16's line)
        E.append(("C1 pair fn", "\n\ndef solve_constraints_ivp(", blk(r'''


# >>> v16b H2-C
def _h2_residual_pair(fn, ell):
    """H2 option C: at most two simultaneously running residual tasks.

    H2_CPU_QUOTA must be the verified effective quota, not os.cpu_count().
    Two CPU slots are reserved for the rest of the process/container.
    """
    from concurrent.futures import ThreadPoolExecutor
    from math import floor

    text = os.environ.get("H2_CPU_QUOTA")
    if text is None:
        raise RuntimeError(
            "H2_RESIDUAL_PAIR requires H2_CPU_QUOTA set to the verified "
            "effective cgroup CPU quota")
    quota = float(text)
    affinity = len(os.sched_getaffinity(0))
    available = min(floor(quota), affinity) - 2
    if available < 2:
        raise RuntimeError("H2 residual pair needs two available CPU slots")

    # Per-call lifetime: no persistent executor to inherit across a fork.
    # One worker plus the caller, not two workers plus a working caller.
    with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="h2-residual") as pool:
        future = pool.submit(fn, ell / 32.0)
        fine = fn(ell / 64.0)
        coarse = future.result()
    return fine, coarse
# <<< v16b H2-C


def solve_constraints_ivp(''')))
        E.append(("C2 fine", "    rW, rN, rW8, rN8 = residuals_at(ell / 64.0)\n", blk(r'''
    # >>> v16b H2-C
    if _H2_RESIDUAL_PAIR:
        _h2_fine, _h2_coarse = _h2_residual_pair(residuals_at, ell)
        rW, rN, rW8, rN8 = _h2_fine
    else:
        # <<< v16b H2-C
        rW, rN, rW8, rN8 = residuals_at(ell / 64.0)
''')))
        E.append(("C3 coarse", "    rW2, rN2, _, _ = residuals_at(ell / 32.0)\n", blk(r'''
    # >>> v16b H2-C
    if _H2_RESIDUAL_PAIR:
        rW2, rN2, _, _ = _h2_coarse
    else:
        # <<< v16b H2-C
        rW2, rN2, _, _ = residuals_at(ell / 32.0)
''')))
    return E


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text, E):
    for name, old, new in E:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100] if old.strip() else old))
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
    """every line a NEW block adds lies between a '# >>> v16b H2-<X>' and a '# <<< v16b H2-<X>' line; the lines of OLD are kept (an
    OLD line indented one level deeper inside C's else branch counts as kept: it is v16's line, the only re-indented ones); blank lines
    are neutral."""
    bad, n_new = [], 0
    for name, old, new in E:
        olds = set(old.splitlines())
        inside = False
        for ln in new.splitlines():
            s = ln.strip()
            if s.startswith("# >>> " + MARK):
                inside = True; n_new += 1; continue
            if s.startswith("# <<< " + MARK):
                inside = False; n_new += 1; continue
            if inside:
                n_new += 1; continue
            if ln in olds or s == "" or (name.startswith("C") and ("    " + ln[8:]) in olds and ln.startswith("        ")):
                continue
            bad.append((name, ln))
        if inside:
            bad.append((name, "<block not closed>"))
    return bad, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_fields_v16.py (build) or engine_fields_v16b.py (--reverse)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--with-c", action="store_true", help="also Option C (H2_RESIDUAL_PAIR)")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    E = edits(a.with_c)
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
        raise SystemExit("the source's md5 does not start with %s: not engine_fields_v16.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in E:
        print("  anchor %-14s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
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
        compile(built, a.dst or "engine_fields_v16b.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_fields_v16b.py")
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


if __name__ == "__main__":
    main()
