#!/usr/bin/env python
"""engine_collapse_v37_selftest.py (busi builder, analyst; rc_ruling_v1 6.1 and 6.2): the self-test of engine_collapse_v37a.py
(engine_collapse_v36z.py + --confirm-test, built by build_engine_collapse_v37.py) on the box CPU.
Part 1 (toy, seconds): rounds_v36c imported from v36z and v37a, run with a stub Newton loop (each round's kept residual given) and a stub
evaluation (the confirming step's residual given; |dE/E| 1e-12), so that confirm_v32 itself runs. Scenarios: "quiet" (kept 1e-7, r_c 5e-5:
nothing fires), "band" (kept 1e-7 in 3 rounds, r_c 2e-4, 3e-4, 5e-5), "choice" (kept 1e-5, 2e-5, 3e-5 above --stop-fields, r_c 5e-4, 3e-5,
1e-3), "refused" (kept 1e-7, r_c 2e-4, 3e-4, 1.5e-4). Checks: (T1) v37a off = v36z in every scenario (rounds, kept round, conv, r_c, status,
evaluations, printed lines); (T2) on, "quiet" = off bitwise; (T3) on, "band": rounds 1 and 2 refused by r_c (a "v37 rct" line each), round 3
kept and converged (off: round 1, converged, 1 round); (T4) on, "choice": the kept round is round 2 (the smallest larger-of-the-2, 3e-5) where
off keeps round 1 (the smallest kept); (T5) on, "refused": no round meets the stop rule, the round kept is round 3 (1.5e-4), conv False.
Part 2 (--cli): --help lists --confirm-test {off,on} and --confirm-above; --confirm-test on with --mix plain is refused.
Part 3 (--whole, the engines as processes, about a minute each): engine_collapse_v36z_selftest.py's srf smoke fixture (m 0.2, r_s 16, 256
sites, 3 channels, tau 8, to T/r_s 1.0; its NEWTON line and PLAIN line; its run_engine, bitwise and strip, imported): ZN, ZP v36z; AN, AP
v37a defaults (G-A: bitwise ZN / ZP: records, head and end entries but walls, the fields, spectra and handoff npz and the 0.5 slice byte for
byte; the args differ by confirm_test and confirm_above only; no "v37 rct" in the logs); AO v37a --confirm-test on (nothing fires on this fixture:
every record's r_c at most 1e-4 and the end rule silent; the records equal AN's with the option's label part and solve_v32.options'
"confirm_test" removed, the numeric npz arrays and the 0.5 slice byte for byte); AF v37a --confirm-test on --confirm-above 1e-15 (the
fixture's confirming residuals are 0 at T = 0 and 1e-14, 2e-13 after: the 2 records above it are marked unconverged, meets_stop_rule False,
where AN's converged, one "v37 rct" refusal line each; the T = 0 record as AN's); EZ v36z, EA0 v37a off and EA v37a --confirm-test on
--confirm-above 1, each under the harness v37_inject_rc.py, which adds 5e-3 to confirm_v32's returned r_c from the T/r_s 0.5 record on (the
fixture's r_c is below its kept everywhere, ratio 0.02 to 0.08, so the end rule cannot see the confirming residual alone otherwise): EZ and
EA0 run to 1.0 with no end entry and equal records (the host's end rule reads the kept only); EA ends at 0.5 (end_v32 fired_by "confirming",
the end line says so, the record line's segment says the end rule fires on the confirming residual; the T = 0 record equal EZ's).
    python engine_collapse_v37_selftest.py --code DIR [--z .../engine_collapse_v36z.py] [--a .../engine_collapse_v37a.py] [--cli] [--whole --out DIR --threads 8]
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import types

import numpy as np

NEW_ARGS = {"confirm_test", "confirm_above"}
LABEL_RE = re.compile(r"; v37 rct: --confirm-test on \(rc_ruling_v1 6\.1: .*? against --end-above\)")


def load(path, code_dir, name):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


SCEN = {"quiet": ([1e-7, 1e-7, 1e-7], [5e-5, 5e-5, 5e-5]),
        "band": ([1e-7, 1e-7, 1e-7], [2e-4, 3e-4, 5e-5]),
        "choice": ([1e-5, 2e-5, 3e-5], [5e-4, 3e-5, 1e-3]),
        "refused": ([1e-7, 1e-7, 1e-7], [2e-4, 3e-4, 1.5e-4])}


def toy_rounds(E, scen, on):
    kepts, rcs = SCEN[scen]
    if hasattr(E, "CONF"):
        E.CONF.update({"on": bool(on), "above": 1e-4})
    elif on:
        raise SystemExit("the host has no CONF")
    cur = {"k": 0}
    nk = {"best": None, "failed": None, "n": 0}
    N0 = np.linspace(0.5, 1.0, 6); v0 = np.linspace(-0.2, 0.1, 6)

    def loop(a, F, xs, RS):
        k = RS["round"]; cur["k"] = k
        nk["best"] = {"resid": kepts[k - 1], "k_it": 10 * k, "cache": None, "srcs": None, "sol": (N0.copy(), v0.copy(), None, {"E": 1.0}),
                      "app": (N0.copy(), v0.copy()), "dE": 0.0, "E": 1.0}
        RS["by_halving"] = True
        RS["far"] = {"x": np.asarray(xs, float).copy(), "set_id": k, "step": 0.5, "newton_step": 1, "residual": 1.0}
        return "no convergence within 1 Newton steps (toy round %d)" % k

    def evalG(N, v):
        k = cur["k"]
        return None, None, (np.asarray(N, float) + rcs[k - 1], np.asarray(v, float), None, {"E": 1.0 + 1e-12})

    saved = (E.nk_loop_v36c, getattr(E, "FLD", None))
    E.nk_loop_v36c = loop
    E.FLD = types.SimpleNamespace(charge=lambda info, g: info["E"])
    a = types.SimpleNamespace(stop_fields=1e-6, stop_E=1e-8, g=0.25, rs=16.0)
    RS = {"held": None, "held_m": None, "far": None, "round": 1, "by_halving": False, "steps": [], "sets": {}, "evals": [], "last": None}
    ctr = {"n": 0, "E_prev": 1.0}
    it_hist = []
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out = E.rounds_v36c(a, "direct", "toy", nk, RS, None, evalG, np.concatenate([N0, v0]), it_hist, ctr, 0.0, 0)
    finally:
        E.nk_loop_v36c, E.FLD = saved
        if hasattr(E, "CONF"):
            E.CONF.update({"on": False, "above": 1e-4})
    v = out["v36c"]
    return {"kept_round": v["kept_round"], "n_rounds": len(v["rounds"]), "conv": out["conv"], "r_c": out["r_c"], "status": out["status"],
            "kept": out["best"]["resid"], "evaluations": out["evaluations"], "it_hist": json.dumps(it_hist),
            "rounds": json.dumps([{k: r[k] for k in ("round", "status", "r_c", "dE_c", "conv")} for r in v["rounds"]], default=float),
            "out": buf.getvalue()}


def toy_part(a, fails):
    Z = load(a.z, a.code, "eng_v36z_t37"); A = load(a.a, a.code, "eng_v37a_t37")
    for s in SCEN:
        rz, ra = toy_rounds(Z, s, False), toy_rounds(A, s, False)
        same = rz == ra and "v37" not in ra["out"]
        print("T1 %-8s off: v37a = v36z: %s (kept round %d of %d, conv %s, r_c %.1e)" % (s, same, ra["kept_round"], ra["n_rounds"], ra["conv"], ra["r_c"]))
        if not same:
            fails.append("T1 %s: v37a off differs from v36z" % s)
    off = {s: toy_rounds(A, s, False) for s in SCEN}
    on = {s: toy_rounds(A, s, True) for s in SCEN}
    for s in SCEN:
        nlines = sum(1 for ln in on[s]["out"].splitlines() if ln.startswith("  v37 rct confirm_v32"))
        print("    %-8s on: kept round %d of %d, conv %s, kept %.1e, r_c %.1e, %d refusal line(s); off: kept round %d of %d, conv %s" % (
            s, on[s]["kept_round"], on[s]["n_rounds"], on[s]["conv"], on[s]["kept"], on[s]["r_c"], nlines, off[s]["kept_round"], off[s]["n_rounds"], off[s]["conv"]))
    if on["quiet"] != off["quiet"]:
        fails.append("T2 quiet: on differs from off although nothing fires")
    print("T2 quiet: on = off bitwise: %s" % (on["quiet"] == off["quiet"]))
    b = on["band"]
    okb = (off["band"]["kept_round"] == 1 and off["band"]["n_rounds"] == 1 and off["band"]["conv"] and b["kept_round"] == 3 and b["n_rounds"] == 3 and b["conv"]
           and sum(1 for ln in b["out"].splitlines() if ln.startswith("  v37 rct confirm_v32")) == 2)
    print("T3 band: rounds 1, 2 refused by r_c, round 3 kept and converged (off: round 1 of 1): %s" % okb)
    if not okb:
        fails.append("T3 band")
    c = on["choice"]
    okc = off["choice"]["kept_round"] == 1 and c["kept_round"] == 2 and not c["conv"] and c["n_rounds"] == 3
    print("T4 choice: on keeps round 2 (larger-of-the-2 3e-5), off round 1 (kept 1e-5): %s" % okc)
    if not okc:
        fails.append("T4 choice")
    r = on["refused"]
    okr = r["kept_round"] == 3 and not r["conv"] and r["n_rounds"] == 3 and off["refused"]["conv"]
    print("T5 refused: no round meets the stop rule on; round 3 kept (r_c 1.5e-4), conv False: %s" % okr)
    if not okr:
        fails.append("T5 refused")


def cli_part(a, fails):
    py = sys.executable
    r1 = subprocess.run([py, a.a, "--help"], cwd=a.code, capture_output=True, text=True, timeout=600)
    r2 = subprocess.run([py, a.a, "--out", os.path.join(a.out or a.code, "v37a_never_written"), "--tag", "x", "--mix", "plain", "--confirm-test", "on"], cwd=a.code, capture_output=True, text=True, timeout=600)
    ok1 = r1.returncode == 0 and "{off,on}" in r1.stdout and "--confirm-test" in r1.stdout and "--confirm-above" in r1.stdout and "--nk-ptc" in r1.stdout
    ok2 = r2.returncode == 2 and "version 37: --confirm-test on" in r2.stderr
    print("cli: --help lists --confirm-test {off,on} and --confirm-above: %s; --confirm-test on with --mix plain refused: %s" % (ok1, ok2))
    if not ok1:
        fails.append("cli help: %s" % (r1.stderr or r1.stdout)[-300:])
    if not ok2:
        fails.append("cli refusal (exit %d): %s" % (r2.returncode, (r2.stderr or r2.stdout)[-300:]))


def norm(x, drop=("confirm_test",)):
    """the on-state's records with the option's own entries removed: the label part and solve_v32.options' confirm_test."""
    if isinstance(x, dict):
        return {k: norm(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, list):
        return [norm(v, drop) for v in x]
    if isinstance(x, str):
        return LABEL_RE.sub("", x)
    return x


def npz_numeric_same(p, q):
    A = np.load(p, allow_pickle=True); B = np.load(q, allow_pickle=True)
    if set(A.files) != set(B.files):
        return ["keys %s" % sorted(set(A.files) ^ set(B.files))], []
    bad, strs = [], []
    for k in A.files:
        a_, b_ = A[k], B[k]
        if a_.dtype.kind in "USO":
            if repr(a_.tolist()) != repr(b_.tolist()):
                strs.append(k)
            continue
        if not (a_.shape == b_.shape and a_.dtype == b_.dtype and a_.tobytes() == b_.tobytes()):
            bad.append(k)
    return bad, strs


def rec_sig(q, T, drop=("confirm_test",)):
    return json.dumps(T.strip(norm(q, drop)), sort_keys=True, default=float)


def whole_part(a, fails):
    T = load(os.path.join(a.code, "engine_collapse_v36z_selftest.py"), a.code, "selftest_v36z_t37")
    os.makedirs(a.out, exist_ok=True)
    R = {}

    def run(tag, eng, extra):
        print("running %s (%s %s) ..." % (tag, os.path.basename(eng), " ".join(extra[-6:])), flush=True)
        R[tag] = T.run_engine(a, eng, tag, extra)
        print("  exit %d" % R[tag]["rc"], flush=True)
        if R[tag]["rc"] != 0:
            fails.append("whole: %s exit %d (%s)" % (tag, R[tag]["rc"], R[tag]["dir"]))
        return R[tag]["rc"] == 0

    ok = all([run("ZN", a.z, T.NEWTON), run("AN", a.a, T.NEWTON), run("ZP", a.z, T.PLAIN), run("AP", a.a, T.PLAIN),
              run("AO", a.a, T.NEWTON + ["--confirm-test", "on"]), run("AF", a.a, T.NEWTON + ["--confirm-test", "on", "--confirm-above", "1e-15"])])
    if not ok:
        return
    T.bitwise(R["AN"], R["ZN"], NEW_ARGS, fails, "G-A Newton: AN (v37a, defaults) against ZN (v36z)")
    T.bitwise(R["AP"], R["ZP"], NEW_ARGS, fails, "G-A plain: AP against ZP")
    for tag in ("AN", "AP"):
        body = R[tag]["log"].split("\n", 1)[1] if "\n" in R[tag]["log"] else ""
        if "v37 rct" in body:
            fails.append("whole: %s's log mentions v37 rct with the option off" % tag)
    AN = R["AN"]["json"]["records"]
    nsv = sum(1 for q in AN if isinstance(q.get("solve_v32"), dict))
    rc_max = max((q["solve_v32"].get("confirming_residual") or 0.0) for q in AN if isinstance(q.get("solve_v32"), dict))
    print("AN: %d records, %d with solve_v32; confirming residual max %.2e; kept max %.2e" % (len(AN), nsv, rc_max,
          max(q["solve_v32"]["kept_residual"] for q in AN if isinstance(q.get("solve_v32"), dict))))
    for q in AN:
        sv = q.get("solve_v32") or {}
        print("    AN T/rs %.4f converged %s kept %s confirming %s E/E0 %s" % (q["T_over_rs"], q.get("converged"), sv.get("kept_residual"), sv.get("confirming_residual"), q.get("E_over_E0")))
    # AO: on, nothing fires
    ao = R["AO"]
    same_rec = [rec_sig(q, T) for q in ao["json"]["records"]] == [rec_sig(q, T) for q in AN]
    same_head = json.dumps(T.strip(norm(ao["json"]["head"])), sort_keys=True, default=float) == json.dumps(T.strip(R["AN"]["json"]["head"]), sort_keys=True, default=float)
    ends = sorted(k for k in ao["json"] if k.startswith("end_")) == sorted(k for k in R["AN"]["json"] if k.startswith("end_"))
    diffs = []
    for f in ("fields", "spectra", "handoff"):
        bad, strs = npz_numeric_same(os.path.join(ao["dir"], "collapse_AO_%s.npz" % f), os.path.join(R["AN"]["dir"], "collapse_AN_%s.npz" % f))
        if bad:
            diffs.append("%s %s" % (f, bad))
        if strs:
            print("    AO %s npz: string entries differing (the option's label/solver text): %s" % (f, strs))
    bad = T.same_npz(os.path.join(ao["dir"], "states", "orbitals_AO_T0000.50.npz"), os.path.join(R["AN"]["dir"], "states", "orbitals_AN_T0000.50.npz"))
    if bad:
        diffs.append("slice 0.5 %s" % bad)
    seg = [ln for ln in ao["log"].splitlines() if "; v37 rct [" in ln]
    quiet = all("--confirm-above 0.0001: met; the end rule (0.001) does not fire" in ln for ln in seg) and "v37 rct confirm_v32" not in ao["log"]
    print("AO (v37a --confirm-test on; nothing fires): records = AN's (the option's label part and options entry removed): %s; head: %s; end entries: %s; numeric npz and the 0.5 slice: %s; %d record lines carry the v37 segment, all 'met' and 'does not fire': %s" % (
        same_rec, same_head, ends, "; ".join(diffs) or "byte for byte", len(seg), quiet))
    for ln in seg[:3]:
        print("    AO: ..." + ln[ln.index("; v32 solve"):][:400] if "; v32 solve" in ln else ln[-400:])
    if not (same_rec and same_head and ends and not diffs and quiet and len(seg) == nsv):
        fails.append("whole: AO is not bitwise AN with nothing firing")
    # AF: fires on every Newton record whose confirming residual exceeds 1e-15 (the fixture's r_c: 0 at T = 0, 1e-14 and 2e-13 after)
    af = R["AF"]["json"]["records"]
    n_ref = sum(1 for ln in R["AF"]["log"].splitlines() if ln.startswith("  v37 rct confirm_v32"))
    rows, okf, n_fire = [], len(af) == len(AN), 0
    for q, p in zip(af, AN):
        sv, sp = q.get("solve_v32") or {}, p.get("solve_v32") or {}
        rc_ = sv.get("confirming_residual")
        fire = rc_ is not None and rc_ > 1e-15
        n_fire += bool(fire and sv)
        rows.append("T/rs %.4f AF converged %s meets_stop_rule %s (AN %s %s); kept %s confirming %s (fires: %s); E/E0 %s (AN %s)" % (
            q["T_over_rs"], q.get("converged"), sv.get("meets_stop_rule"), p.get("converged"), sp.get("meets_stop_rule"), sv.get("kept_residual"), rc_, fire,
            q.get("E_over_E0"), p.get("E_over_E0")))
        if sv and fire and (q.get("converged") or sv.get("meets_stop_rule") or not p.get("converged")):
            okf = False
        if sv and not fire and (q.get("converged") != p.get("converged")):
            okf = False
    print("AF (v37a --confirm-test on --confirm-above 1e-15): %d records (AN %d); %d Newton records with r_c above 1e-15, %d refusal lines; those marked unconverged (AN's converged), the others as AN: %s" % (len(af), len(AN), n_fire, n_ref, okf))
    for r_ in rows:
        print("    " + r_)
    if not okf or n_fire == 0 or n_ref != n_fire:
        fails.append("whole: AF does not show the records marked unconverged")
    # AE: the end rule fired by the confirming residual alone. On this fixture r_c is below the kept on every record (ratio 0.02 to 0.08), so
    # the harness v37_inject_rc.py adds 5e-3 to confirm_v32's returned r_c from its 2nd call on (the T/r_s 0.5 record), after the engine's own
    # stop-rule test; run on v36z (the host), v37a off and v37a on (--confirm-above 1, so (a) cannot fire; --end-above 1e-3)
    inj = os.path.join(a.code, "v37_inject_rc.py")

    def run_inj(tag, eng, extra):
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, inj, eng, "--rc-add", "5e-3", "--rc-from-call", "2", "--"] + T.BASE + extra + ["--out", d, "--tag", tag, "--slice-dir", os.path.join(d, "states")]
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(a.threads)
        log = os.path.join(d, tag + ".log")
        print("running %s (injected; %s %s) ..." % (tag, os.path.basename(eng), " ".join(extra[-4:])), flush=True)
        with open(log, "w") as fh:
            fh.write(" ".join(cmd) + "\n"); fh.flush()
            rc = subprocess.call(cmd, cwd=a.code, stdout=fh, stderr=subprocess.STDOUT, env=env)
        print("  exit %d" % rc, flush=True)
        R[tag] = {"tag": tag, "rc": rc, "dir": d, "json": json.load(open(os.path.join(d, "collapse_%s.json" % tag))) if rc == 0 else None, "log": open(log, errors="replace").read()}
        if rc != 0:
            fails.append("whole: %s exit %d" % (tag, rc))
        return rc == 0
    if not all([run_inj("EZ", a.z, T.NEWTON), run_inj("EA0", a.a, T.NEWTON), run_inj("EA", a.a, T.NEWTON + ["--confirm-test", "on", "--confirm-above", "1"])]):
        return
    EZ, EA0, EA = R["EZ"]["json"], R["EA0"]["json"], R["EA"]["json"]
    sig = lambda J: [json.dumps(T.strip(q), sort_keys=True, default=float) for q in J["records"]]  # noqa: E731
    host_runs = ("end_v32" not in EZ and "end_v32" not in EA0 and len(EZ["records"]) == len(AN) and sig(EZ) == sig(EA0))
    print("EZ (v36z, injected) and EA0 (v37a off, injected): no end entry, %d records each, records equal: %s (the host's end rule reads the kept only)" % (len(EZ["records"]), host_runs))
    e32 = EA.get("end_v32") or {}
    ln_ = [ln for ln in R["EA"]["log"].splitlines() if "the member ends at this record (engine spec 32.1 (3), v37 rct (c))" in ln]
    seg = [ln for ln in R["EA"]["log"].splitlines() if "; v37 rct [" in ln]
    before = rec_sig(EA["records"][0], T) == rec_sig(EZ["records"][0], T)
    oke = (e32.get("fired_by") == "confirming" and abs(e32.get("T_over_rs", -1) - 0.5) < 1e-9 and len(EA["records"]) == 2 and len(ln_) == 1
           and "fired by confirming" in ln_[0] and any("the end rule (0.001) fires on the confirming residual" in s for s in seg) and before)
    print("EA (v37a on, injected): ends at T/rs %s (%d records); end_v32 fired_by %s (kept %s, confirming %s, larger %s); T = 0 record equal EZ's (normalized): %s" % (
        e32.get("T_over_rs"), len(EA["records"]), e32.get("fired_by"), e32.get("kept_residual"), e32.get("confirming_residual"), e32.get("larger_residual"), before))
    for s in seg:
        print("    EA: ..." + s[s.index("; v37 rct ["):][:300])
    print("    EA: " + (ln_[0][:400] if ln_ else "no end line"))
    if not host_runs:
        fails.append("whole: EZ / EA0 (the host and v37a off under the injection) differ or ended")
    if not oke:
        fails.append("whole: EA did not end on the confirming residual as expected")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--z", default=None)
    ap.add_argument("--a", default=None)
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--whole", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.z = os.path.abspath(a.z or os.path.join(a.code, "engine_collapse_v36z.py"))
    a.a = os.path.abspath(a.a or os.path.join(a.code, "engine_collapse_v37a.py"))
    for p_ in (a.z, a.a, os.path.join(a.code, "engine_collapse_v36z_selftest.py")):
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
