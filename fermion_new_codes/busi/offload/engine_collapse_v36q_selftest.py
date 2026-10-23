#!/usr/bin/env python
"""engine_collapse_v36q_selftest.py (busi, reviewer A; S4_128_nk_stop_ruling_v1 section 3): the plain-direction fallback's fixture on the CPU.
nk_loop_v36c is imported from the built engine (engine_collapse_v36q.py) and from the base (engine_collapse_v36m.py) and run in isolation on a
toy residual with stub arguments and a stub RS: G(x) = c (x* - x) on 8 components (the plain step x + omega G contracts by |1 - omega c|), with
F.jvp the exact J v = -c v where the Jacobian is right and the sign-flipped +c v where it is deliberately wrong (the Krylov direction is then
x - x*, along which |G| grows linearly: no halving lowers the residual, while the plain step does). Scenarios: newton (right everywhere), wrong
(wrong everywhere), mixed (wrong while max|x - x*| > 0.3), cross (mixed with c = 1.2 and a second set where x[0] > x*[0] + 0.05: the plain
trial at omega 1 crosses and is the far-side start, omega 1/2 is taken), offload (cross with OFF["on"] and the memo hook, the drops counted),
fail (a repelling fixed point, G = c (x - x*): neither direction lowers the residual).
Checks: with the option off the v36q loop is the v36m loop (status, steps, far, by_halving, every evaluation's x, the jvp calls, the memo drops,
the printed lines: all equal, no "v36q" printed); with it on and Newton never failing, the same; where Newton fails the fallback fires, the JVP
check reads FD +|G|_2 against the (wrong) exact -|G|_2 (|FD - exact|/|FD| = 2, the max-norm slope +1 r0), the plain step is taken and the solve
converges, Newton taking over once the Jacobian is right (mixed, cross: the Jacobian updated at the plain iterate); in fail the status says
neither direction fell. The engine's command line is checked too (--help lists the option; on without --nk-set resolve is refused).
    python engine_collapse_v36q_selftest.py [--code DIR] [--engine .../engine_collapse_v36q.py] [--base .../engine_collapse_v36m.py] [--verbose]
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import types

import numpy as np

MARK = "v36q fallback"
XSTAR = np.linspace(0.5, 1.5, 8)
D0 = np.array([-1.0, 0.7, -0.4, 0.9, -0.2, 0.5, -0.8, 0.3])            # x0 = x* + D0: max|x0 - x*| = 1, x0[0] below x*[0]

SCEN = {                                                               # c, the sign of G, the Jacobian wrong while max|x - x*| > wrong_above, the other set where x[0] > x*[0] + cross_above, OFF["on"]
    "newton": dict(c=0.6, sign=+1, wrong_above=None, cross_above=None, off=False),
    "wrong": dict(c=0.6, sign=+1, wrong_above=0.0, cross_above=None, off=False),
    "mixed": dict(c=0.6, sign=+1, wrong_above=0.3, cross_above=None, off=False),
    "cross": dict(c=1.2, sign=+1, wrong_above=0.3, cross_above=0.05, off=False),
    "offload": dict(c=1.2, sign=+1, wrong_above=0.3, cross_above=0.05, off=True),
    "fail": dict(c=0.6, sign=-1, wrong_above=0.0, cross_above=None, off=False),
}
ORDER = ["newton", "wrong", "mixed", "cross", "offload", "fail"]


def load(path, code_dir, name):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


class ToyF(object):
    """the stub residual: G(x) = sign c (x* - x); the evaluation sets RS["last"] (the held set 0, or set 1 beyond x[0] > x*[0] + cross_above) as
    nk_stage_v32's F does; jvp is the exact J v where the Jacobian is right and -J v where it is deliberately wrong; v36m_drop counts the memo drops."""

    def __init__(self, RS, c, sign, wrong_above, cross_above):
        self.RS, self.c, self.sign, self.wrong_above, self.cross_above = RS, c, sign, wrong_above, cross_above
        self.calls, self.jvp_calls, self.drops = [], 0, 0

    def __call__(self, x):
        x = np.asarray(x, float).flatten()
        self.calls.append(x.copy())
        cross = self.cross_above is not None and float(x[0]) > XSTAR[0] + self.cross_above
        if cross and "s1" not in self.RS["sets"]:
            self.RS["sets"]["s1"] = {"id": 1}
        self.RS["last"] = {"set_id": 1 if cross else 0, "same": not cross, "gap": 0.5 - 0.1 * float(np.max(np.abs(x - XSTAR))),
                           "sig": "s1" if cross else "s0", "flips": 1 if cross else 0}
        return self.sign * self.c * (XSTAR - x)

    def jvp(self, x, v):
        self.jvp_calls += 1
        x = np.asarray(x, float).flatten(); v = np.asarray(v, float).flatten()
        wrong = self.wrong_above is not None and float(np.max(np.abs(x - XSTAR))) > self.wrong_above
        J = -self.sign * self.c                                        # the true Jacobian of G
        return (-J if wrong else J) * v

    def v36m_drop(self):
        self.drops += 1


def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x, float)).tobytes()).hexdigest()


def run_loop(E, scen, flag):
    p = SCEN[scen]
    E.OFF.update({"on": bool(p["off"]), "test": False, "rec": None})
    E.JV["dropped_pairs"] = 0
    RS = {"held": "s0", "held_m": None, "first_m": None, "sets": {"s0": {"id": 0}}, "evals": [], "steps": [], "far": None, "round": 1, "last": None,
          "by_halving": False, "v36q_record": "T/rs 2.938 direct (toy)"}
    a = types.SimpleNamespace(nk_krylov=10, nk_inner=15, nk_restarts=None, nk_halvings=6, nk_maxiter=40, stop_fields=1e-10, nk_plain_fallback=flag,
                              mix="newton", mix_alpha=0.5, rs=64.0)
    F = ToyF(RS, p["c"], p["sign"], p["wrong_above"], p["cross_above"])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            status = E.nk_loop_v36c(a, F, XSTAR + D0, RS)
    finally:
        E.OFF.update({"on": False, "test": False})
    far = None if RS["far"] is None else dict({k: v for k, v in RS["far"].items() if k != "x"}, x=h(RS["far"]["x"]))
    return {"status": status, "steps": json.dumps(RS["steps"], sort_keys=True, default=float), "far": json.dumps(far, sort_keys=True, default=float),
            "by_halving": bool(RS["by_halving"]), "calls": [h(c) for c in F.calls], "jvp_calls": F.jvp_calls, "drops": F.drops, "out": buf.getvalue(),
            "raw_steps": RS["steps"], "raw_far": far}


def differing(A, B):
    return [k for k in ("status", "steps", "far", "by_halving", "calls", "jvp_calls", "drops", "out") if A[k] != B[k]]


def expected_drops(steps, off):
    n = 0
    if off:
        for s in steps:
            n += len(s["trials"]) - (1 if s["accepted_step"] is not None else 0)
            pf = s.get("plain_fallback")
            if pf is not None:
                n += 1 + len(pf["trials"]) - (1 if pf["accepted_step"] is not None else 0)
    return n


def close(x, y, rel=1e-9):
    return x is not None and abs(x - y) <= rel * max(abs(x), abs(y), 1e-300)


def check_jvp(pf, r0, fails, where):
    """the check on the toy's wrong Jacobian: FD +|G|_2 (the trials ascend), exact -|G|_2 (the wrong J.v claims descent), |FD - exact|/|FD| 2."""
    c = pf["jvp_check"]
    g2 = c["G_norm2"]
    for name, ok in (("fd_step 1/64", c["fd_step"] == 1.0 / 64), ("fd on the held set", c["fd_crosses"] is False), ("FD slope +|G|_2", close(c["fd_slope"], g2)),
                     ("exact slope -|G|_2", close(c["exact_slope"], -g2)), ("|FD - exact|/|FD| 2", close(c["fd_vs_exact"], 2.0)),
                     ("max-norm slope +r0", close(c["fd_slope_maxnorm"], r0)), ("no exact error", c["exact_error"] is None)):
        if not ok:
            fails.append("%s: JVP check %s (got %r)" % (where, name, c))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--code", default=here, help="the engines' directory (engine_collapse_v5, engine_fields_v16, ... importable from it)")
    ap.add_argument("--engine", default=None, help="engine_collapse_v36q.py (default: --code/engine_collapse_v36q.py)")
    ap.add_argument("--base", default=None, help="engine_collapse_v36m.py (default: --code/engine_collapse_v36m.py)")
    ap.add_argument("--verbose", action="store_true", help="print every captured engine line")
    ap.add_argument("--no-cli", action="store_true", help="skip the engine's command-line checks (subprocesses)")
    a = ap.parse_args()
    eng = a.engine or os.path.join(a.code, "engine_collapse_v36q.py")
    base = a.base or os.path.join(a.code, "engine_collapse_v36m.py")
    for p_ in (eng, base):
        print("%s md5 %s" % (p_, hashlib.md5(open(p_, "rb").read()).hexdigest()))
    n_marked = sum(1 for ln in open(eng, encoding="utf-8") if MARK in ln)
    print("marked lines in the engine (%r): %d; numpy %s; python %s" % (MARK, n_marked, np.__version__, sys.version.split()[0]))
    Eq = load(eng, a.code, "engine_collapse_v36q_under_test")
    Em = load(base, a.code, "engine_collapse_v36m_base")
    fails = []
    if not hasattr(Eq, "nk_plain_fallback_v36q"):
        fails.append("the built engine has no nk_plain_fallback_v36q")
    if hasattr(Em, "nk_plain_fallback_v36q"):
        fails.append("the base engine has nk_plain_fallback_v36q (wrong --base?)")
    if n_marked == 0:
        fails.append("the engine carries no marked line")
    R = {}
    for scen in ORDER:
        R[scen] = {"m": run_loop(Em, scen, "off"), "off": run_loop(Eq, scen, "off"), "on": run_loop(Eq, scen, "on")}
        m, off, on = R[scen]["m"], R[scen]["off"], R[scen]["on"]
        d = differing(m, off)
        if d:
            fails.append("%s: v36q with the option off differs from v36m in %s" % (scen, d))
        if "v36q" in off["out"] or "v36q" in m["out"]:
            fails.append("%s: 'v36q' printed with the option off" % scen)
        n_fb = sum(1 for s in on["raw_steps"] if "plain_fallback" in s)
        print("%-8s v36m/off: %s (%d steps, by_halving %s, %d evaluations, %d drops); on: %s (%d steps, %d fallbacks, %d evaluations, %d drops); off == v36m: %s" % (
            scen, m["status"][:70], len(m["raw_steps"]), m["by_halving"], len(m["calls"]), m["drops"], on["status"][:70], len(on["raw_steps"]), n_fb, len(on["calls"]), on["drops"],
            "yes" if not d else "NO"))
        steps = on["raw_steps"]
        pf = [s.get("plain_fallback") for s in steps]
        if scen == "newton":
            if differing(on, off):
                fails.append("newton: the option on differs from off although the fallback never fires: %s" % differing(on, off))
            if on["status"] != "converged" or n_fb != 0 or len(steps) != 1 or steps[0]["accepted_step"] != 1.0:
                fails.append("newton: expected converged in one Newton step without a fallback")
        if scen == "wrong":
            if m["status"] != "the residual did not fall on the held set in the halvings (to 2^-6 while the trials crossed) at Newton step 1 (the solve ends there)" or not m["by_halving"]:
                fails.append("wrong: v36m did not stop by the halving rule at step 1 (%s)" % m["status"])
            if on["status"] != "converged" or len(steps) != 25 or n_fb != 25 or any(p is None or p["accepted_step"] != 1.0 for p in pf):
                fails.append("wrong: expected 25 plain steps at omega 1 then converged (got %s, %d steps, %d fallbacks)" % (on["status"], len(steps), n_fb))
            if pf[0] is not None:
                check_jvp(pf[0], steps[0]["residual_before"], fails, "wrong step 1")
            if "accepted none" not in on["out"].splitlines()[0] or not on["out"].splitlines()[1].startswith("  v36q fallback T/rs 2.938 direct (toy): round 1 Newton step 1, held set 0"):
                fails.append("wrong: the fallback line does not follow the step's line with the record label")
        if scen == "mixed":
            if on["status"] != "converged" or len(steps) != 3 or [p is not None for p in pf] != [True, True, False] or [p["accepted_step"] for p in pf[:2]] != [1.0, 1.0] or steps[2]["accepted_step"] != 1.0:
                fails.append("mixed: expected two plain steps at omega 1, then a Newton step at 1, then converged (got %s, %s)" % (on["status"], [(s["accepted_step"], (s.get("plain_fallback") or {}).get("accepted_step")) for s in steps]))
            if steps[0]["accepted_step"] is not None:
                fails.append("mixed: the Newton accepted_step is not None on a fallback step")
            if not m["by_halving"] or on["by_halving"]:
                fails.append("mixed: by_halving expected True for v36m and False for the converged v36q")
        if scen in ("cross", "offload"):
            good = (on["status"] == "converged" and len(steps) == 3 and [p is not None for p in pf] == [True, True, False] and all(p["accepted_step"] == 0.5 for p in pf[:2])
                    and all([q["crosses"] for q in p["trials"]] == [True, False] for p in pf[:2]) and all(p["trials"][0]["set"] == 1 for p in pf[:2]) and steps[2]["accepted_step"] == 1.0)
            if not good:
                fails.append("%s: expected two fallbacks accepted at omega 1/2 after a crossing trial at 1, then a Newton step (got %s, %s)" % (scen, on["status"], json.dumps(pf, default=float)[:400]))
            far = on["raw_far"]
            if far is None or far.get("plain") is not True or far.get("set_id") != 1 or far.get("newton_step") != 1 or far.get("step") != 1.0 or not close(far.get("residual"), 1.2 * 0.2, 1e-6):
                fails.append("%s: RS['far'] expected the plain trial at omega 1 of Newton step 1 on set 1 (got %r)" % (scen, far))
            if m["raw_far"] is not None:
                fails.append("%s: v36m's RS['far'] expected None (the Newton trials do not cross)" % scen)
            if pf[0] is not None:
                check_jvp(pf[0], steps[0]["residual_before"], fails, "%s step 1" % scen)
        if scen == "offload":
            for lab, r_ in (("v36m", m), ("off", off), ("on", on)):
                want = expected_drops(r_["raw_steps"], True)
                if r_["drops"] != want:
                    fails.append("offload %s: %d memo drops, expected %d (every rejected trial, and the check's evaluation)" % (lab, r_["drops"], want))
            if m["drops"] != 7:
                fails.append("offload v36m: expected 7 drops (the seven rejected Newton trials)")
        if scen == "fail":
            if on["status"] != m["status"] + " [v36q: nor along the plain direction, 7 trials]" or not on["by_halving"] or len(steps) != 1 or pf[0] is None or pf[0]["accepted_step"] is not None or len(pf[0]["trials"]) != 7:
                fails.append("fail: expected the v36m stop with the plain-direction clause after 7 plain trials (got %s)" % on["status"])
            if pf[0] is not None:
                check_jvp(pf[0], steps[0]["residual_before"], fails, "fail step 1")
            if len(on["calls"]) != 15 or len(m["calls"]) != 8:                # the iterate's evaluation, 7 Newton trials, then 7 plain trials
                fails.append("fail: expected 8 evaluations (v36m) and 15 (v36q on); got %d and %d" % (len(m["calls"]), len(on["calls"])))
        if scen in ("mixed", "fail") or a.verbose:
            for ln in on["out"].splitlines():
                if ln.startswith("  v36q fallback") or a.verbose:
                    print("    " + ln)
    if not a.no_cli:
        py = sys.executable
        r1 = subprocess.run([py, eng, "--help"], cwd=a.code, capture_output=True, text=True, timeout=120)
        r2 = subprocess.run([py, eng, "--out", os.path.join(a.code, "v36q_selftest_never_written"), "--tag", "x", "--nk-plain-fallback", "on"], cwd=a.code, capture_output=True, text=True, timeout=120)
        ok1 = r1.returncode == 0 and "--nk-plain-fallback" in r1.stdout and "{off,on}" in r1.stdout
        ok2 = r2.returncode == 2 and "version 36q: --nk-plain-fallback" in r2.stderr
        print("cli: --help lists --nk-plain-fallback {off,on}: %s; on without --nk-set resolve refused (exit %d): %s" % (ok1, r2.returncode, ok2))
        if not ok1:
            fails.append("cli: --help does not list the option (exit %d): %s" % (r1.returncode, (r1.stderr or r1.stdout)[-300:]))
        if not ok2:
            fails.append("cli: the option without --nk-set resolve was not refused (exit %d): %s" % (r2.returncode, (r2.stderr or r2.stdout)[-300:]))
    for f_ in fails:
        print("FAIL: " + f_)
    print("SELFTEST %s (%d checks failed)" % ("PASS" if not fails else "FAIL", len(fails)))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
