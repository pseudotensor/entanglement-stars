#!/usr/bin/env python
"""engine_collapse_v36r_selftest.py (busi, analyst; #303): the self-test of engine_collapse_v36r.py (v36q's plain-direction fallback on v36n,
built by build_engine_collapse_v36r.py) on the CPU. Ported from engine_collapse_v36q_selftest.py (busi/code/offload, md5 3ca4a2bd): the base is
v36n in place of v36m, the "offload" scenario (v36m's host release) is dropped since v36n has no release, the names say v36r; plus a whole-engine
Newton leg on the v36n self-test's 512-site fixture.
Part 1 (the toy loop, seconds): nk_loop_v36c is imported from the built engine (engine_collapse_v36r.py) and from the base (engine_collapse_v36n.py)
and run in isolation on a toy residual with stub arguments and a stub RS: G(x) = c (x* - x) on 8 components (the plain step x + omega G contracts
by |1 - omega c|), with F.jvp the exact J v = -c v where the Jacobian is right and the sign-flipped +c v where it is deliberately wrong (the Krylov
direction is then x - x*, along which |G| grows linearly: no halving lowers the residual, while the plain step does). Scenarios: newton (right
everywhere), wrong (wrong everywhere), mixed (wrong while max|x - x*| > 0.3), cross (mixed with c = 1.2 and a second set where x[0] > x*[0] + 0.05:
the plain trial at omega 1 crosses and is the far-side start, omega 1/2 is taken), fail (a repelling fixed point, G = c (x - x*): neither direction
lowers the residual).
Checks: with the option off the v36r loop is the v36n loop (status, steps, far, by_halving, every evaluation's x, the jvp calls, the printed lines:
all equal, no "v36r" printed); with it on and Newton never failing, the same; where Newton fails the fallback fires, the JVP check reads FD +|G|_2
against the (wrong) exact -|G|_2 (|FD - exact|/|FD| = 2, the max-norm slope +1 r0), the plain step is taken and the solve converges, Newton taking
over once the Jacobian is right (mixed, cross: the Jacobian updated at the plain iterate); in fail the status says neither direction fell. The
engine's command line is checked too (--help lists the option; on without --nk-set resolve is refused).
Part 2 (--whole, the engine as a process; minutes on the box CPU): the v36n self-test's fixture (engine_collapse_v36n_selftest.py: m 0.2, r_s 16,
512 sites, kappa_sea 10, T/r_s 0 to 0.25) with M3c's option set (--dust-set isotropic --iter-cache drop) and the Newton line (--mix newton and
J2's NK flags, ready_jobs_v3b's NK without --end-above): NN v36n, RN v36r with the option off, RF v36r with --nk-plain-fallback on.
G-R: RN bitwise with NN (the v36n self-test's compare_runs: the records but the walls, the head, the fields, spectra and handoff npz, the T = 0
slice); the args differ by the key nk_plain_fallback only. RF: exit 0; the records equal to RN's with the "plain_fallback" entries removed when
the fallback never fires (READ otherwise: the fallback lines are printed).
    python engine_collapse_v36r_selftest.py [--code DIR] [--engine .../engine_collapse_v36r.py] [--base .../engine_collapse_v36n.py] [--verbose]
        [--whole --v36n-selftest .../engine_collapse_v36n_selftest.py --out DIR --threads 8]
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

MARK = "v36r fallback"
XSTAR = np.linspace(0.5, 1.5, 8)
D0 = np.array([-1.0, 0.7, -0.4, 0.9, -0.2, 0.5, -0.8, 0.3])            # x0 = x* + D0: max|x0 - x*| = 1, x0[0] below x*[0]

SCEN = {                                                               # c, the sign of G, the Jacobian wrong while max|x - x*| > wrong_above, the other set where x[0] > x*[0] + cross_above
    "newton": dict(c=0.6, sign=+1, wrong_above=None, cross_above=None),
    "wrong": dict(c=0.6, sign=+1, wrong_above=0.0, cross_above=None),
    "mixed": dict(c=0.6, sign=+1, wrong_above=0.3, cross_above=None),
    "cross": dict(c=1.2, sign=+1, wrong_above=0.3, cross_above=0.05),
    "fail": dict(c=0.6, sign=-1, wrong_above=0.0, cross_above=None),
}
ORDER = ["newton", "wrong", "mixed", "cross", "fail"]
NK = ["--nk-krylov", "40", "--nk-halvings", "6", "--nk-inner", "15", "--nk-jvp", "exact", "--nk-line-search", "halving", "--nk-set", "resolve",
      "--nk-maxiter", "30", "--nk-restarts", "2"]
M3C_OPTS = ["--dust-set", "isotropic", "--iter-cache", "drop"]


def load(path, code_dir, name):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


class ToyF(object):
    """the stub residual: G(x) = sign c (x* - x); the evaluation sets RS["last"] (the held set 0, or set 1 beyond x[0] > x*[0] + cross_above) as
    nk_stage_v32's F does; jvp is the exact J v where the Jacobian is right and -J v where it is deliberately wrong."""

    def __init__(self, RS, c, sign, wrong_above, cross_above):
        self.RS, self.c, self.sign, self.wrong_above, self.cross_above = RS, c, sign, wrong_above, cross_above
        self.calls, self.jvp_calls = [], 0

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


def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x, float)).tobytes()).hexdigest()


def run_loop(E, scen, flag):
    p = SCEN[scen]
    RS = {"held": "s0", "held_m": None, "first_m": None, "sets": {"s0": {"id": 0}}, "evals": [], "steps": [], "far": None, "round": 1, "last": None,
          "by_halving": False, "v36r_record": "T/rs 2.833 direct (toy)"}
    a = types.SimpleNamespace(nk_krylov=10, nk_inner=15, nk_restarts=None, nk_halvings=6, nk_maxiter=40, stop_fields=1e-10, nk_plain_fallback=flag,
                              mix="newton", mix_alpha=0.5, rs=64.0)
    F = ToyF(RS, p["c"], p["sign"], p["wrong_above"], p["cross_above"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = E.nk_loop_v36c(a, F, XSTAR + D0, RS)
    far = None if RS["far"] is None else dict({k: v for k, v in RS["far"].items() if k != "x"}, x=h(RS["far"]["x"]))
    return {"status": status, "steps": json.dumps(RS["steps"], sort_keys=True, default=float), "far": json.dumps(far, sort_keys=True, default=float),
            "by_halving": bool(RS["by_halving"]), "calls": [h(c) for c in F.calls], "jvp_calls": F.jvp_calls, "out": buf.getvalue(),
            "raw_steps": RS["steps"], "raw_far": far}


def differing(A, B):
    return [k for k in ("status", "steps", "far", "by_halving", "calls", "jvp_calls", "out") if A[k] != B[k]]


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


def toy_part(a, eng, base, fails):
    Er = load(eng, a.code, "engine_collapse_v36r_under_test")
    En = load(base, a.code, "engine_collapse_v36n_base")
    if not hasattr(Er, "nk_plain_fallback_v36r"):
        fails.append("the built engine has no nk_plain_fallback_v36r")
    if hasattr(En, "nk_plain_fallback_v36r") or hasattr(En, "nk_plain_fallback_v36q"):
        fails.append("the base engine has a plain fallback (wrong --base?)")
    if not hasattr(En, "iso_configure"):
        fails.append("the base engine has no iso_configure (not v36n?)")
    n_fb_total = 0
    for scen in ORDER:
        m, off, on = run_loop(En, scen, "off"), run_loop(Er, scen, "off"), run_loop(Er, scen, "on")
        d = differing(m, off)
        if d:
            fails.append("%s: v36r with the option off differs from v36n in %s" % (scen, d))
        if "v36r" in off["out"] or "v36r" in m["out"]:
            fails.append("%s: 'v36r' printed with the option off" % scen)
        n_fb = sum(1 for s in on["raw_steps"] if "plain_fallback" in s)
        n_fb_total += n_fb
        print("%-8s v36n/off: %s (%d steps, by_halving %s, %d evaluations); on: %s (%d steps, %d fallbacks, %d evaluations); off == v36n: %s" % (
            scen, m["status"][:70], len(m["raw_steps"]), m["by_halving"], len(m["calls"]), on["status"][:70], len(on["raw_steps"]), n_fb, len(on["calls"]),
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
                fails.append("wrong: v36n did not stop by the halving rule at step 1 (%s)" % m["status"])
            if on["status"] != "converged" or len(steps) != 25 or n_fb != 25 or any(p is None or p["accepted_step"] != 1.0 for p in pf):
                fails.append("wrong: expected 25 plain steps at omega 1 then converged (got %s, %d steps, %d fallbacks)" % (on["status"], len(steps), n_fb))
            if pf and pf[0] is not None:
                check_jvp(pf[0], steps[0]["residual_before"], fails, "wrong step 1")
            ol = on["out"].splitlines()
            if len(ol) < 2 or "accepted none" not in ol[0] or not ol[1].startswith("  v36r fallback T/rs 2.833 direct (toy): round 1 Newton step 1, held set 0"):
                fails.append("wrong: the fallback line does not follow the step's line with the record label")
        if scen == "mixed":
            if on["status"] != "converged" or len(steps) != 3 or [p is not None for p in pf] != [True, True, False] or [p["accepted_step"] for p in pf[:2]] != [1.0, 1.0] or steps[2]["accepted_step"] != 1.0:
                fails.append("mixed: expected two plain steps at omega 1, then a Newton step at 1, then converged (got %s, %s)" % (on["status"], [(s["accepted_step"], (s.get("plain_fallback") or {}).get("accepted_step")) for s in steps]))
            if steps and steps[0]["accepted_step"] is not None:
                fails.append("mixed: the Newton accepted_step is not None on a fallback step")
            if not m["by_halving"] or on["by_halving"]:
                fails.append("mixed: by_halving expected True for v36n and False for the converged v36r")
        if scen == "cross":
            good = (on["status"] == "converged" and len(steps) == 3 and [p is not None for p in pf] == [True, True, False] and all(p["accepted_step"] == 0.5 for p in pf[:2])
                    and all([q["crosses"] for q in p["trials"]] == [True, False] for p in pf[:2]) and all(p["trials"][0]["set"] == 1 for p in pf[:2]) and steps[2]["accepted_step"] == 1.0)
            if not good:
                fails.append("cross: expected two fallbacks accepted at omega 1/2 after a crossing trial at 1, then a Newton step (got %s, %s)" % (on["status"], json.dumps(pf, default=float)[:400]))
            far = on["raw_far"]
            if far is None or far.get("plain") is not True or far.get("set_id") != 1 or far.get("newton_step") != 1 or far.get("step") != 1.0 or not close(far.get("residual"), 1.2 * 0.2, 1e-6):
                fails.append("cross: RS['far'] expected the plain trial at omega 1 of Newton step 1 on set 1 (got %r)" % (far,))
            if m["raw_far"] is not None:
                fails.append("cross: v36n's RS['far'] expected None (the Newton trials do not cross)")
            if pf and pf[0] is not None:
                check_jvp(pf[0], steps[0]["residual_before"], fails, "cross step 1")
        if scen == "fail":
            if on["status"] != m["status"] + " [v36r: nor along the plain direction, 7 trials]" or not on["by_halving"] or len(steps) != 1 or pf[0] is None or pf[0]["accepted_step"] is not None or len(pf[0]["trials"]) != 7:
                fails.append("fail: expected the v36n stop with the plain-direction clause after 7 plain trials (got %s)" % on["status"])
            if pf and pf[0] is not None:
                check_jvp(pf[0], steps[0]["residual_before"], fails, "fail step 1")
            if len(on["calls"]) != 15 or len(m["calls"]) != 8:                # the iterate's evaluation, 7 Newton trials, then 7 plain trials
                fails.append("fail: expected 8 evaluations (v36n) and 15 (v36r on); got %d and %d" % (len(m["calls"]), len(on["calls"])))
        if scen in ("mixed", "cross", "fail") or a.verbose:
            for ln in on["out"].splitlines():
                if ln.startswith("  v36r fallback") or a.verbose:
                    print("    " + ln[:400])
    print("toy part: the fallback fired %d times over the scenarios (option on)" % n_fb_total)
    if n_fb_total == 0:
        fails.append("toy part: the fallback branch was never exercised")


def cli_part(a, eng, fails):
    py = sys.executable
    r1 = subprocess.run([py, eng, "--help"], cwd=a.code, capture_output=True, text=True, timeout=300)
    r2 = subprocess.run([py, eng, "--out", os.path.join(a.code, "v36r_selftest_never_written"), "--tag", "x", "--nk-plain-fallback", "on"], cwd=a.code, capture_output=True, text=True, timeout=300)
    ok1 = r1.returncode == 0 and "--nk-plain-fallback" in r1.stdout and "{off,on}" in r1.stdout and "--dust-set" in r1.stdout
    ok2 = r2.returncode == 2 and "version 36r: --nk-plain-fallback" in r2.stderr
    print("cli: --help lists --nk-plain-fallback {off,on} beside v36n's --dust-set: %s; on without --nk-set resolve refused (exit %d): %s" % (ok1, r2.returncode, ok2))
    if not ok1:
        fails.append("cli: --help does not list the option (exit %d): %s" % (r1.returncode, (r1.stderr or r1.stdout)[-300:]))
    if not ok2:
        fails.append("cli: the option without --nk-set resolve was not refused (exit %d): %s" % (r2.returncode, (r2.stderr or r2.stdout)[-300:]))


def drop_fb(o):
    if isinstance(o, dict):
        return {k: drop_fb(v) for k, v in o.items() if k != "plain_fallback"}
    if isinstance(o, list):
        return [drop_fb(v) for v in o]
    return o


def whole_part(a, eng, base, fails):
    """the engine as a process on the v36n self-test's fixture, Newton line, M3c's options: NN v36n, RN v36r off, RF v36r on."""
    S = load(a.v36n_selftest, a.code, "engine_collapse_v36n_selftest_lib")
    ns = types.SimpleNamespace(code_dir=a.code, no_kcache=False, gpu=None, threads=a.threads)
    root = a.out
    os.makedirs(root, exist_ok=True)
    newton = ["--mix", "newton"] + NK + M3C_OPTS
    cases = [("NN", base, newton), ("RN", eng, newton), ("RF", eng, newton + ["--nk-plain-fallback", "on"])]
    runs = {}
    for name, e_, extra in cases:
        if os.path.exists(os.path.join(root, name, "collapse_fx_%s.json" % name)):
            raise SystemExit("%s exists under %s: use a fresh --out (nothing overwritten)" % (name, root))
        print("running %s (%s %s) ..." % (name, os.path.basename(e_), " ".join(extra)), flush=True)
        runs[name] = S.run_case(ns, name, os.path.basename(e_), extra, root)
        print("  exit %d in %.0f s (%s)" % (runs[name]["rc"], runs[name]["seconds"], runs[name]["log"]), flush=True)
        if runs[name]["rc"] != 0:
            fails.append("whole: %s exit %d (%s)" % (name, runs[name]["rc"], runs[name]["log"]))
    if any(r["rc"] != 0 for r in runs.values()):
        return
    d = S.compare_runs(runs["RN"], runs["NN"])
    print("G-R: RN (v36r, option off) bitwise with NN (v36n) (records but the walls, head, fields, spectra, handoff, T = 0 slice): %s" % ("; ".join(d) or "identical"))
    if d:
        fails.append("whole: RN differs from NN: %s" % "; ".join(d))
    JN, JR, JF = (json.load(open(runs[k]["json"])) for k in ("NN", "RN", "RF"))
    ka = set(JR["args"]) ^ set(JN["args"])
    va = [k for k in set(JR["args"]) & set(JN["args"]) if JR["args"][k] != JN["args"][k] and k not in ("out", "tag", "slice_dir")]
    print("args: keys only in one of RN/NN %s; values differing (but out, tag, slice_dir) %s" % (sorted(ka), sorted(va)))
    if ka != {"nk_plain_fallback"} or JR["args"].get("nk_plain_fallback") != "off" or va:
        fails.append("whole: the args of RN and NN differ beyond nk_plain_fallback: %s %s" % (sorted(ka), va))
    fb_lines = [ln.rstrip() for ln in open(runs["RF"]["log"], errors="replace") if ln.startswith("  v36r fallback")]
    nk_lines = [ln for ln in open(runs["NN"]["log"], errors="replace") if "Newton" in ln or "newton" in ln]
    print("NN log lines mentioning Newton: %d (the Newton path ran); RF fallback lines: %d" % (len(nk_lines), len(fb_lines)))
    if not nk_lines:
        fails.append("whole: NN's log shows no Newton line (the Newton path did not run?)")
    for ln in fb_lines:
        print("    RF: " + ln[:400])
    rr = [json.dumps(drop_fb(S.strip(q)), sort_keys=True, default=float) for q in JR["records"]]
    rf = [json.dumps(drop_fb(S.strip(q)), sort_keys=True, default=float) for q in JF["records"]]
    same = rr == rf
    if not fb_lines:
        print("RF: the fallback never fired; the records equal RN's with the plain_fallback entries removed: %s" % same)
        if not same:
            fails.append("whole: RF's records differ from RN's although the fallback never fired")
    else:
        print("READ RF: the fallback fired %d times; records equal to RN's (plain_fallback removed): %s" % (len(fb_lines), same))
    nE = [q.get("T_over_rs") for q in JF["records"]]
    print("RF records T/rs %s; unconverged %s" % (nE, [q.get("T_over_rs") for q in JF["records"] if q.get("unconverged")]))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--code", default=here, help="the engines' directory (engine_collapse_v5, engine_fields_v16, ... importable from it)")
    ap.add_argument("--engine", default=None, help="engine_collapse_v36r.py (default: --code/engine_collapse_v36r.py)")
    ap.add_argument("--base", default=None, help="engine_collapse_v36n.py (default: --code/engine_collapse_v36n.py)")
    ap.add_argument("--verbose", action="store_true", help="print every captured engine line")
    ap.add_argument("--no-cli", action="store_true", help="skip the engine's command-line checks (subprocesses)")
    ap.add_argument("--whole", action="store_true", help="also part 2: the engine as a process on the v36n self-test's fixture, Newton line")
    ap.add_argument("--v36n-selftest", dest="v36n_selftest", default=None, help="engine_collapse_v36n_selftest.py (its fixture, run_case and compare_runs)")
    ap.add_argument("--out", default=None, help="part 2's scratch directory (fresh)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    eng = os.path.abspath(a.engine or os.path.join(a.code, "engine_collapse_v36r.py"))
    base = os.path.abspath(a.base or os.path.join(a.code, "engine_collapse_v36n.py"))
    for p_ in (eng, base):
        print("%s md5 %s" % (p_, hashlib.md5(open(p_, "rb").read()).hexdigest()))
    n_marked = sum(1 for ln in open(eng, encoding="utf-8") if MARK in ln)
    print("marked lines in the engine (%r): %d; numpy %s; python %s" % (MARK, n_marked, np.__version__, sys.version.split()[0]))
    fails = []
    if n_marked == 0:
        fails.append("the engine carries no marked line")
    toy_part(a, eng, base, fails)
    if not a.no_cli:
        cli_part(a, eng, fails)
    if a.whole:
        if not a.v36n_selftest or not a.out:
            raise SystemExit("--whole needs --v36n-selftest and --out")
        whole_part(a, eng, base, fails)
    for f_ in fails:
        print("FAIL: " + f_)
    print("SELFTEST %s (%d checks failed)" % ("PASS" if not fails else "FAIL", len(fails)))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
