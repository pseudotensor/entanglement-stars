#!/usr/bin/env python
"""engine_collapse_v36z_selftest.py (busi, analyst; #331): the self-test of engine_collapse_v36y.py (v36q + the merged substep option) and
engine_collapse_v36z.py (v36y + v36t's PTC ladder as a fallback behind the Newton halving search and the plain direction), on the box CPU.
Part 1 (the toy loop, seconds): nk_loop_v36c imported from v36y and from v36z, run on a toy residual with stub arguments and a stub RS (the
form of engine_collapse_v36r_selftest.py). Scenarios newton, wrong, mixed, cross, fail (v36q's toy: G(x) = c (x* - x), the J.v deliberately
sign-flipped where "wrong") and ladder: G = diag(lam) (x - x*) with lam = +3 on four components (the plain step x + omega G expands there at
every omega) and -0.6 on four (the plain step contracts), the J.v diag(mu) v with mu = 4 / 0.1 while max|x - x*| > 0.05 (so the Newton
direction grows the second group at every halving: no Newton trial falls) and mu = lam below it; the starting residuals are equal (0.06) on
all components, so neither the halving search nor the plain direction lowers the max norm, while the backward-Euler step at delta 1 does.
Checks: (T1) --nk-ptc off: v36z's loop is v36y's in every scenario, plain fallback off and on (status, steps, far, by_halving, every
evaluation's x, the jvp calls, the printed lines; no "v36z" printed); (T2) --nk-ptc fallback where Newton or the plain direction is accepted
at every step (newton; wrong, mixed, cross with the plain fallback on): the same as off, bitwise; (T3) ladder, plain fallback on: v36y stops at
Newton step 1 "nor along the plain direction, 7 trials"; v36z hands over at Newton step 1 (the step entry's ptc_fallback: from step 1, delta0
1), the ladder's first step accepts delta 1, the round converges, every ladder entry carries "ptc" and ptc_fallback_from_newton_step 1, and the
evaluations up to the hand-over are v36y's; (T4) ladder, plain fallback off: the hand-over says so and converges; (T5) fail: the ladder finds
no rung either: by_halving True and the status carries "[v36z: the ladder from Newton step 1]"; (T6) --nk-ptc test on newton: the ladder from
Newton step 1 converges.
Part 2 (--cli): --help lists --nk-ptc {off,fallback,test} and the sub-options; --nk-ptc fallback without --nk-set resolve is refused; --nk-ptc
on is refused (PTC as the loop is not offered).
Part 3 (--whole, the engines as processes; minutes on the box CPU): the solve lane's srf smoke fixture (selftest_srf_v1b: m 0.2, r_s 16, 256
sites, 3 channels, tau 8 a, to T/r_s 1.0) on RUNG2's Newton line (the member's NK flags, --end-above 1e-3, --nk-plain-fallback on,
--nk-host-release on) and on the plain line: QN, QP v36q; YN, YP v36y; ZN, ZP v36z (defaults): bitwise QN / QP (the json's records and head
and end entries but walls, the fields, spectra, handoff and the T/r_s 0.5 slice byte for byte; the args differ by the new keys only); YS v36y
with "--step-refine on --step-refine-max-halvings 3" (inert where no record fails: the records equal QN's with the "srf" entries removed);
ZF v36z with "--nk-ptc fallback --nk-ptc-delta0 1" (inert where Newton never fails: the records equal QN's with the options' ptc entries
removed); ZT v36z with "--nk-ptc test" (the hand-over in the whole engine with the host release on: exit 0, ladder lines printed, every record
within the 1e-3 end rule; E/E0 against QN printed).
    python engine_collapse_v36z_selftest.py --code DIR [--y .../engine_collapse_v36y.py] [--z .../engine_collapse_v36z.py] [--q .../engine_collapse_v36q.py]
        [--cli] [--whole --out DIR --threads 8]
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

XSTAR = np.linspace(0.5, 1.5, 8)
D0 = np.array([-1.0, 0.7, -0.4, 0.9, -0.2, 0.5, -0.8, 0.3])
LAM = np.array([3.0, -0.6, 3.0, -0.6, 3.0, -0.6, 3.0, -0.6])
MU_WRONG = np.array([4.0, 0.1, 4.0, 0.1, 4.0, 0.1, 4.0, 0.1])
D_LAD = np.array([0.02, 0.1, -0.02, -0.1, 0.02, -0.1, -0.02, 0.1])    # |lam d| = 0.06 on every component

SCEN = {
    "newton": dict(kind="c", c=0.6, sign=+1, wrong_above=None, cross_above=None),
    "wrong": dict(kind="c", c=0.6, sign=+1, wrong_above=0.0, cross_above=None),
    "mixed": dict(kind="c", c=0.6, sign=+1, wrong_above=0.3, cross_above=None),
    "cross": dict(kind="c", c=1.2, sign=+1, wrong_above=0.3, cross_above=0.05),
    "fail": dict(kind="c", c=0.6, sign=-1, wrong_above=0.0, cross_above=None),
    "ladder": dict(kind="diag", wrong_above=0.05, cross_above=None),
}
ORDER = ["newton", "wrong", "mixed", "cross", "fail", "ladder"]


def load(path, code_dir, name):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


class ToyF(object):
    def __init__(self, RS, p):
        self.RS, self.p = RS, p
        self.calls, self.jvp_calls = [], 0

    def __call__(self, x):
        x = np.asarray(x, float).flatten()
        self.calls.append(x.copy())
        p = self.p
        cross = p["cross_above"] is not None and float(x[0]) > XSTAR[0] + p["cross_above"]
        if cross and "s1" not in self.RS["sets"]:
            self.RS["sets"]["s1"] = {"id": 1}
        self.RS["last"] = {"set_id": 1 if cross else 0, "same": not cross, "gap": 0.5 - 0.1 * float(np.max(np.abs(x - XSTAR))),
                           "sig": "s1" if cross else "s0", "flips": 1 if cross else 0}
        if p["kind"] == "diag":
            return LAM * (x - XSTAR)
        return p["sign"] * p["c"] * (XSTAR - x)

    def jvp(self, x, v):
        self.jvp_calls += 1
        x = np.asarray(x, float).flatten(); v = np.asarray(v, float).flatten()
        p = self.p
        wrong = p["wrong_above"] is not None and float(np.max(np.abs(x - XSTAR))) > p["wrong_above"]
        if p["kind"] == "diag":
            return (MU_WRONG if wrong else LAM) * v
        J = -p["sign"] * p["c"]
        return (-J if wrong else J) * v


def h(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x, float)).tobytes()).hexdigest()


def set_ptc(E, mode):
    """--nk-ptc as main() sets it (v36z's PTC dict; v36y has none)."""
    if not hasattr(E, "PTC"):
        return
    E.PTC.update({"on": mode != "off", "fallback": mode != "off", "test": mode == "test", "delta0": 1.0, "delta_min": 1e-3, "delta_max": 1e6,
                  "ladder": 4.0, "pivot": 16.0, "trials": 8, "dx_max": 0.1, "one_cache": False})


def run_loop(E, scen, fb, ptc="off"):
    p = SCEN[scen]
    set_ptc(E, ptc)
    RS = {"held": "s0", "held_m": None, "first_m": None, "sets": {"s0": {"id": 0}}, "evals": [], "steps": [], "far": None, "round": 1, "last": None,
          "by_halving": False, "v36q_record": "T/rs 2.833 direct (toy)", "v36t_record": "T/rs 2.833 direct (toy)"}
    a = types.SimpleNamespace(nk_krylov=10, nk_inner=15, nk_restarts=None, nk_halvings=6, nk_maxiter=40, stop_fields=1e-10, nk_plain_fallback=fb,
                              mix="newton", mix_alpha=0.5, rs=64.0)
    F = ToyF(RS, p)
    x0 = XSTAR + (D_LAD if p["kind"] == "diag" else D0)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            status = E.nk_loop_v36c(a, F, x0, RS)
    finally:
        set_ptc(E, "off")
    far = None if RS["far"] is None else dict({k: v for k, v in RS["far"].items() if k != "x"}, x=h(RS["far"]["x"]))
    return {"status": status, "steps": json.dumps(RS["steps"], sort_keys=True, default=float), "far": json.dumps(far, sort_keys=True, default=float),
            "by_halving": bool(RS["by_halving"]), "calls": [h(c) for c in F.calls], "jvp_calls": F.jvp_calls, "out": buf.getvalue(), "raw_steps": RS["steps"]}


def differing(A, B):
    return [k for k in ("status", "steps", "far", "by_halving", "calls", "jvp_calls", "out") if A[k] != B[k]]


def toy_part(a, fails):
    Ey = load(a.y, a.code, "engine_collapse_v36y_under_test")
    Ez = load(a.z, a.code, "engine_collapse_v36z_under_test")
    for nm in ("nk_ptc_fallback_v36z", "nk_loop_ptc_v36t", "ptc_ladder_v36t", "PTC"):
        if not hasattr(Ez, nm):
            fails.append("v36z has no %s" % nm)
    if hasattr(Ey, "PTC") or not hasattr(Ey, "SRF"):
        fails.append("v36y: expected SRF and no PTC")
    import inspect
    if "nk_loop_ptc_v36t(a, F, x0, RS)" in inspect.getsource(Ez.nk_loop_v36c):
        fails.append("v36z's nk_loop_v36c dispatches to the ladder at its head (PTC as the loop)")
    # T1, T2
    for scen in ORDER:
        for fb in ("off", "on"):
            y = run_loop(Ey, scen, fb)
            z0 = run_loop(Ez, scen, fb, "off")
            d = differing(y, z0)
            if d:
                fails.append("T1 %s fb %s: v36z --nk-ptc off differs from v36y in %s" % (scen, fb, d))
            if "v36z" in z0["out"]:
                fails.append("T1 %s fb %s: 'v36z' printed with the option off" % (scen, fb))
            zf = run_loop(Ez, scen, fb, "fallback")
            inert = (scen == "newton") or (fb == "on" and scen in ("wrong", "mixed", "cross"))
            fired = sum(1 for s in zf["raw_steps"] if "ptc_fallback" in s)
            print("%-7s fb %-3s v36y: %-60s (%2d steps, %3d evals) | v36z fallback: %-60s (%2d steps, %3d evals, hand-overs %d)%s" % (
                scen, fb, y["status"][:60], len(y["raw_steps"]), len(y["calls"]), zf["status"][:60], len(zf["raw_steps"]), len(zf["calls"]), fired,
                "" if not d else "  OFF DIFFERS"))
            if inert:
                d2 = differing(y, zf)
                if d2 or fired:
                    fails.append("T2 %s fb %s: v36z --nk-ptc fallback differs from v36y where Newton or the plain direction always falls: %s" % (scen, fb, d2))
            if scen == "ladder":
                st = zf["raw_steps"]
                if fb == "on" and not (y["status"].endswith("[v36q: nor along the plain direction, 7 trials]") and y["by_halving"] and len(y["raw_steps"]) == 1):
                    fails.append("T3: v36y did not stop at Newton step 1 on both directions (%s)" % y["status"])
                ok = (zf["status"] == "converged" and fired == 1 and st[0].get("ptc_fallback", {}).get("from_newton_step") == 1 and st[0]["ptc_fallback"]["delta0"] == 1.0
                      and st[0]["ptc_fallback"]["status"] == "converged" and len(st) >= 3 and all("ptc" in q and q.get("ptc_fallback_from_newton_step") == 1 for q in st[1:])
                      and st[1]["ptc"]["accepted_delta"] == 1.0 and not zf["by_halving"])
                if not ok:
                    fails.append("T%s: ladder fb %s: expected the hand-over at Newton step 1, delta 1 accepted first, converged (got %s, %s)" % (
                        "3" if fb == "on" else "4", fb, zf["status"], json.dumps([(q.get("ptc_fallback"), (q.get("ptc") or {}).get("accepted_delta")) for q in st], default=float)[:600]))
                n_pre = len(y["calls"])
                if zf["calls"][:n_pre] != y["calls"]:
                    fails.append("T3/T4 ladder fb %s: the evaluations before the hand-over are not v36y's" % fb)
                if fb == "off" and "(--nk-plain-fallback off)" not in zf["out"]:
                    fails.append("T4: the hand-over line does not say the plain fallback is off")
                for ln in zf["out"].splitlines():
                    if ln.startswith("  v36z ptcfb") or ln.startswith("  v36t ptc"):
                        print("    " + ln[:330])
            if scen == "fail":
                if not (zf["by_halving"] and zf["status"].endswith("[v36z: the ladder from Newton step 1]") and fired == 1):
                    fails.append("T5 fail fb %s: expected the ladder to find no rung (by_halving, the clause): %s" % (fb, zf["status"]))
    zt = run_loop(Ez, "newton", "on", "test")
    stt = zt["raw_steps"]
    print("T6 newton, --nk-ptc test: %s (%d steps, %d evals; the step-1 entry: %s)" % (zt["status"], len(stt), len(zt["calls"]), json.dumps(stt[0].get("ptc_fallback"), default=float)))
    if not (zt["status"] == "converged" and stt and stt[0].get("ptc_fallback", {}).get("why", "").startswith("test mode") and len(stt) >= 2 and "ptc" in stt[1]):
        fails.append("T6: the test mode did not hand over at Newton step 1 and converge")


def cli_part(a, fails):
    py = sys.executable
    r1 = subprocess.run([py, a.z, "--help"], cwd=a.code, capture_output=True, text=True, timeout=600)
    r2 = subprocess.run([py, a.z, "--out", os.path.join(a.out or a.code, "v36z_never_written"), "--tag", "x", "--nk-ptc", "fallback"], cwd=a.code, capture_output=True, text=True, timeout=600)
    r3 = subprocess.run([py, a.z, "--out", os.path.join(a.out or a.code, "v36z_never_written"), "--tag", "x", "--nk-ptc", "on"], cwd=a.code, capture_output=True, text=True, timeout=600)
    ok1 = r1.returncode == 0 and "{off,fallback,test}" in r1.stdout and "--nk-ptc-delta0" in r1.stdout and "--nk-ptc-one-cache" in r1.stdout and "--step-refine" in r1.stdout
    ok2 = r2.returncode == 2 and "version 36z: --nk-ptc" in r2.stderr
    ok3 = r3.returncode == 2 and "invalid choice" in r3.stderr
    print("cli: --help lists --nk-ptc {off,fallback,test}, the sub-options and --step-refine: %s; fallback without --nk-set resolve refused: %s; --nk-ptc on refused: %s" % (ok1, ok2, ok3))
    for ok, r, what in ((ok1, r1, "help"), (ok2, r2, "fallback without resolve"), (ok3, r3, "--nk-ptc on")):
        if not ok:
            fails.append("cli %s (exit %d): %s" % (what, r.returncode, (r.stderr or r.stdout)[-300:]))


BASE = ["--m", "0.2", "--g", "0.25", "--rs", "16", "--R0-over-rs", "4", "--reading", "A", "--nsites", "256", "--kappa-sea", "3", "--ell", "4", "--prep-ell", "4",
        "--tau", "8", "--T-over-rs", "1.0", "--reference", "R2", "--site-fields", "averaged", "--field-kernel-cache", "--v36-fixes", "--slices-c128", "0.5"]
NEWTON = ["--mix", "newton", "--nk-krylov", "40", "--nk-restarts", "2", "--nk-maxiter", "30", "--nk-halvings", "6", "--nk-inner", "15", "--nk-jvp", "exact",
          "--nk-line-search", "halving", "--nk-set", "resolve", "--end-above", "0.001", "--nk-plain-fallback", "on", "--nk-host-release", "on"]
PLAIN = ["--mix", "plain", "--maxit", "20"]
SRF_ARGS = {"step_refine", "step_refine_max_halvings", "step_refine_above", "step_refine_gauge", "step_refine_recover", "step_refine_growth"}
PTC_ARGS = {"nk_ptc", "nk_ptc_delta0", "nk_ptc_delta_min", "nk_ptc_delta_max", "nk_ptc_ladder", "nk_ptc_pivot", "nk_ptc_trials", "nk_ptc_dx_max", "nk_ptc_one_cache"}


def strip(x, drop=()):
    if isinstance(x, dict):
        return {k: strip(v, drop) for k, v in x.items() if k not in drop and k != "slice_saved" and not k.endswith("_s") and not any(w in k for w in ("second", "timing", "wall"))}
    if isinstance(x, list):
        return [strip(v, drop) for v in x]
    return x


def same_npz(p, q):
    A = np.load(p, allow_pickle=True); B = np.load(q, allow_pickle=True)
    if set(A.files) != set(B.files):
        return ["keys %s" % sorted(set(A.files) ^ set(B.files))]
    bad = []
    for k in A.files:
        a_, b_ = A[k], B[k]
        if a_.dtype == object:
            ok = len(a_) == len(b_) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a_, b_))
        else:
            ok = a_.shape == b_.shape and a_.dtype == b_.dtype and a_.tobytes() == b_.tobytes()
        if not ok:
            bad.append(k)
    return bad


def run_engine(a, engine, tag, extra):
    d = os.path.join(a.out, tag)
    if os.path.exists(os.path.join(d, "collapse_%s.json" % tag)):
        raise SystemExit("%s exists under %s: use a fresh --out (nothing overwritten)" % (tag, a.out))
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, engine] + BASE + extra + ["--out", d, "--tag", tag, "--slice-dir", os.path.join(d, "states")]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(a.threads)
    log = os.path.join(d, tag + ".log")
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n"); fh.flush()
        rc = subprocess.call(cmd, cwd=a.code, stdout=fh, stderr=subprocess.STDOUT, env=env)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag))) if rc == 0 else None
    return {"tag": tag, "rc": rc, "dir": d, "json": J, "log": open(log, errors="replace").read()}


def bitwise(A, B, extra_keys, fails, label):
    diffs = []
    JA, JB = A["json"], B["json"]
    ka = set(JA["args"]) ^ set(JB["args"])
    va = [k for k in set(JA["args"]) & set(JB["args"]) if JA["args"][k] != JB["args"][k] and k not in ("out", "tag", "slice_dir")]
    if ka != extra_keys or va:
        diffs.append("args: keys only in one %s, values %s" % (sorted(ka), va))
    for key in ["head", "records"] + sorted(k for k in set(JA) | set(JB) if k.startswith("end_")):
        if json.dumps(strip(JA.get(key)), sort_keys=True, default=float) != json.dumps(strip(JB.get(key)), sort_keys=True, default=float):
            diffs.append(key)
    for f in ("fields", "spectra", "handoff"):
        b = same_npz(os.path.join(A["dir"], "collapse_%s_%s.npz" % (A["tag"], f)), os.path.join(B["dir"], "collapse_%s_%s.npz" % (B["tag"], f)))
        if b:
            diffs.append("%s %s" % (f, b))
    b = same_npz(os.path.join(A["dir"], "states", "orbitals_%s_T0000.50.npz" % A["tag"]), os.path.join(B["dir"], "states", "orbitals_%s_T0000.50.npz" % B["tag"]))
    if b:
        diffs.append("slice 0.5 %s" % b)
    print("%s: %s" % (label, "; ".join(diffs) or "bitwise (records and head and end entries but walls; fields, spectra, handoff and the 0.5 slice byte for byte; args differ by the new keys only)"))
    if diffs:
        fails.append("%s: %s" % (label, "; ".join(diffs)))


def whole_part(a, fails):
    os.makedirs(a.out, exist_ok=True)
    cases = [("QN", a.q, NEWTON), ("YN", a.y, NEWTON), ("ZN", a.z, NEWTON), ("QP", a.q, PLAIN), ("YP", a.y, PLAIN), ("ZP", a.z, PLAIN),
             ("YS", a.y, NEWTON + ["--step-refine", "on", "--step-refine-max-halvings", "3"]),
             ("ZF", a.z, NEWTON + ["--nk-ptc", "fallback", "--nk-ptc-delta0", "1"]),
             ("ZT", a.z, NEWTON + ["--nk-ptc", "test"])]
    R = {}
    for tag, eng, extra in cases:
        print("running %s (%s %s) ..." % (tag, os.path.basename(eng), " ".join(extra[-6:])), flush=True)
        R[tag] = run_engine(a, eng, tag, extra)
        print("  exit %d" % R[tag]["rc"], flush=True)
        if R[tag]["rc"] != 0:
            fails.append("whole: %s exit %d (%s)" % (tag, R[tag]["rc"], R[tag]["dir"]))
    if any(r["rc"] != 0 for r in R.values()):
        return
    bitwise(R["YN"], R["QN"], SRF_ARGS, fails, "G-Y Newton: YN (v36y, --step-refine off) against QN (v36q)")
    bitwise(R["ZN"], R["QN"], SRF_ARGS | PTC_ARGS, fails, "G-Z Newton: ZN (v36z, defaults) against QN (v36q)")
    bitwise(R["YP"], R["QP"], SRF_ARGS, fails, "G-Y plain: YP against QP")
    bitwise(R["ZP"], R["QP"], SRF_ARGS | PTC_ARGS, fails, "G-Z plain: ZP against QP")
    for tag, R_ in (("YN", R["YN"]), ("ZN", R["ZN"]), ("YP", R["YP"]), ("ZP", R["ZP"])):
        body = R_["log"].split("\n", 1)[1] if "\n" in R_["log"] else ""   # the log's first line is the command (it names the engine file)
        if "srf" in body or "v36z" in body or "v36t ptc" in body:
            fails.append("whole: %s's log mentions srf / v36z / v36t ptc with the options off" % tag)
    nq = [json.dumps(strip(q), sort_keys=True, default=float) for q in R["QN"]["json"]["records"]]
    fired_s = "srf T/rs" in R["YS"]["log"] and "the record fails" in R["YS"]["log"]
    # with the option on, v36q_srf copies every accepted state to pinned host memory (its snapshot), so the record's v36m offload counters say
    # pinned True where v36q's say None (bookkeeping, no number): the "pinned" key is dropped from both sides for this comparison only
    nqp = [json.dumps(strip(q, ("pinned",)), sort_keys=True, default=float) for q in R["QN"]["json"]["records"]]
    ys = [json.dumps(strip(q, ("srf", "pinned")), sort_keys=True, default=float) for q in R["YS"]["json"]["records"]]
    print("YS (v36y --step-refine on --step-refine-max-halvings 3): the refinement fired: %s; records equal QN's with the srf entries (and the offload's pinned flag) removed: %s" % (fired_s, ys == nqp))
    if not fired_s and ys != nqp:
        fails.append("whole: YS's records differ from QN's although the refinement never fired")
    fired_f = "v36z ptcfb" in R["ZF"]["log"]
    zf = [json.dumps(strip(q, ("ptc", "ptc_fallback")), sort_keys=True, default=float) for q in R["ZF"]["json"]["records"]]
    print("ZF (v36z --nk-ptc fallback --nk-ptc-delta0 1): the hand-over fired: %s; records equal QN's with the options' ptc entries removed: %s" % (fired_f, zf == nq))
    if not fired_f and zf != nq:
        fails.append("whole: ZF's records differ from QN's although the ladder never ran")
    zl = [ln for ln in R["ZT"]["log"].splitlines() if ln.startswith("  v36z ptcfb") or ln.startswith("  v36t ptc")]
    EQ = {round(q["T_over_rs"], 6): q.get("E_over_E0") for q in R["QN"]["json"]["records"]}
    rows, bad = [], []
    for q in R["ZT"]["json"]["records"]:
        t_ = round(q["T_over_rs"], 6)
        un = q.get("unconverged")
        kept = (un or {}).get("residual") if un else None
        rows.append("T/rs %.4f converged %s%s E/E0 %s (QN %s)" % (t_, q.get("converged"), ("" if kept is None else " kept %.2e" % kept), q.get("E_over_E0"), EQ.get(t_)))
        if not (q.get("converged") or (kept is not None and kept < 1e-3)):
            bad.append(t_)
    print("ZT (v36z --nk-ptc test, the hand-over in the whole engine with --nk-host-release on): %d v36z/v36t lines; records:" % len(zl))
    for r_ in rows:
        print("    " + r_)
    for ln in zl[:12]:
        print("    ZT: " + ln[:300])
    if not any(ln.startswith("  v36z ptcfb") for ln in zl) or not any(ln.startswith("  v36t ptc") for ln in zl):
        fails.append("whole: ZT printed no hand-over or no ladder line")
    if bad:
        fails.append("whole: ZT records outside the 1e-3 end rule at %s" % bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="the engines' directory (engine_collapse_v5, engine_fields_v16, ... importable from it)")
    ap.add_argument("--q", default=None)
    ap.add_argument("--y", default=None)
    ap.add_argument("--z", default=None)
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--whole", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.q = os.path.abspath(a.q or os.path.join(a.code, "engine_collapse_v36q.py"))
    a.y = os.path.abspath(a.y or os.path.join(a.code, "engine_collapse_v36y.py"))
    a.z = os.path.abspath(a.z or os.path.join(a.code, "engine_collapse_v36z.py"))
    for p_ in (a.q, a.y, a.z):
        print("%s md5 %s" % (p_, hashlib.md5(open(p_, "rb").read()).hexdigest()))
    print("numpy %s; python %s" % (np.__version__, sys.version.split()[0]))
    fails = []
    toy_part(a, fails)
    if a.cli:
        cli_part(a, fails)
    if a.whole:
        if not a.out:
            raise SystemExit("--whole needs --out")
        whole_part(a, fails)
    for f_ in fails:
        print("FAIL: " + f_)
    print("SELFTEST %s (%d checks failed)" % ("PASS" if not fails else "FAIL", len(fails)))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
