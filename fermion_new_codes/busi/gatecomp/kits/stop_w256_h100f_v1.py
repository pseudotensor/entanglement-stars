#!/usr/bin/env python
"""stop_w256_h100f_v1.py (busi, GATECOMP, ReviewA_flatq_v1 sec. 2, inbox #408): W256_s4's end conditions, ON THE CARD (h100-f), read only
until it acts. Every PERIOD s it reads the member's run json (and, after the birth, the fields npz) and stops the engine by PID when the
first of these holds:
  (a) the plan's end: a first root exists (min(N - |v|) crossed zero; T_birth by gate_common_v1.birth, reviewer A 3.2) AND the gate has failed
      (T_gate: the first record from T_birth - 1 with sigma ||K||_op > 0.1 or the code of record's six-term on the nodes r >= sigma > 0.1,
      gate_common_v1.gate_profile with the field module's ivp re-solve) AND the json holds a record past max(T_birth's bracket, T_gate):
      "T_gate + 1 record" (the json and npz are rewritten every 5 records, so the stop can come up to 4 records later; it acts at once);
  (b) the cap in clock: a record at T/r_s >= 12.5 in the log;
  (c) the cap in card-hours: 20 h after the member's first start (seg0 at 2026-10-21T21:05:38Z), i.e. 2026-10-22T17:05:38Z.
On acting: the run's json and npz are copied to run_stop_backup/ (after they load), the reading (T_birth, R_birth, T_gate, window, the gate
rows, which condition) goes to STOP.json and STOP.txt, then the engine python and its children (the pair workers) get TERM by PID (KILL
after 120 s). The birth is also written to BIRTH.txt as soon as it is seen, with ReviewA_flatq_v1's rule for the well's second width:
8a if R_birth >= 80 a (sigma/R_birth <= 0.1 at 8a), else 2a.
  stop_w256_h100f_v1.py MEMBER_DIR TAG CODE_DIR [--period 300] [--sigma 4] [--dry]
"""
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_common_v1 as GC  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("member"); ap.add_argument("tag"); ap.add_argument("code")
ap.add_argument("--period", type=float, default=300.0); ap.add_argument("--sigma", type=float, default=4.0)
ap.add_argument("--t-cap", dest="t_cap", type=float, default=12.5); ap.add_argument("--rs", type=float, default=256.0)
ap.add_argument("--wall-cap", dest="wall_cap", default="2026-10-22T17:05:38Z"); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
M = a.member; RUN = os.path.join(M, "run"); rs = a.rs
cap_wall = dt.datetime.strptime(a.wall_cap, "%Y-%m-%dT%H:%M:%SZ")
six = GC.load_sixterm(a.code)


def say(s):
    line = "%s %s" % (dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), s)
    open(os.path.join(M, "STOP_WATCH.txt"), "a").write(line + "\n"); print(line, flush=True)


def engine_pid():
    """the python engine of the member's current run: run/pid_T3's subshell child after the T 3 restart, else the chain's run/pid child."""
    for f in ("pid_T3", "pid"):
        p = os.path.join(RUN, f)
        if os.path.exists(p):
            sh = open(p).read().strip()
            kids = subprocess.run(["ps", "-o", "pid=,cmd=", "--ppid", sh], capture_output=True, text=True).stdout.split("\n")
            for k in kids:
                if "engine_collapse" in k:
                    return int(k.split()[0])
    return None


def stop(why, reading):
    pid = engine_pid()
    bk = os.path.join(M, "run_stop_backup"); os.makedirs(bk, exist_ok=True)
    for f in os.listdir(RUN):
        if f.startswith("collapse_%s" % a.tag):
            shutil.copy2(os.path.join(RUN, f), bk)
    reading.update({"why": why, "engine_pid": pid, "utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    json.dump(reading, open(os.path.join(M, "STOP.json"), "w"), indent=1, default=str)
    say("STOP (%s): engine pid %s; reading %s" % (why, pid, json.dumps({k: reading.get(k) for k in ("T_birth", "R_birth", "T_gate", "window", "last_T")}, default=str)))
    if a.dry or pid is None:
        say("dry or no pid: not signalled"); return
    kids = [int(x) for x in subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout.split()]
    os.kill(pid, signal.SIGTERM)
    for _ in range(24):
        try:
            os.kill(pid, 0); time.sleep(5)
        except OSError:
            break
    else:
        os.kill(pid, signal.SIGKILL); say("KILL sent to %d" % pid)
    for k in kids:
        try:
            os.kill(k, signal.SIGTERM); say("child %d TERM" % k)
        except OSError:
            pass
    say("engine stopped")


birth_posted = False
say("watching %s (tag %s); caps T %.3f and %s; sigma %g; six-term %s" % (M, a.tag, a.t_cap, a.wall_cap, a.sigma, "on" if six else "OFF"))
while True:
    now = dt.datetime.utcnow()
    last_T = None
    try:
        L = [x for x in open(os.path.join(RUN, "log.txt"), errors="replace") if x.startswith("T/rs ") and " E/E0 " in x]
        last_T = float(L[-1].split()[1]) if L else None
    except OSError:
        pass
    if last_T is not None and last_T >= a.t_cap - 1e-9:
        stop("the clock cap T/r_s %.3f" % a.t_cap, {"last_T": last_T}); break
    if now >= cap_wall:
        stop("the card-hour cap (20 h from seg0)", {"last_T": last_T}); break
    try:
        J = json.load(open(os.path.join(RUN, "collapse_%s.json" % a.tag)))
    except Exception:
        time.sleep(30); continue
    b = GC.birth(J, rs)
    if b and not birth_posted:
        R = b["R_birth"]; w = "8a" if (R is not None and R >= 80.0) else "2a"
        txt = ("T_birth %.5f r_s (bracket %.4f to %.4f, quadratic %s), R_birth %s a = %s of 2GE0 (%.3f a); ReviewA_flatq_v1's rule: the well's second "
               "width is %s (8a if R_birth >= 80 a)" % (b["T_birth"], b["T_lo"], b["T_hi"], GC.fmt(b["T_quad"], "%.5f"), GC.fmt(R, "%.3f"),
                                                     GC.fmt(b["R_birth_over_2gE"], "%.4f"), b["two_gE0"], w))
        open(os.path.join(M, "BIRTH.txt"), "w").write(txt + "\n"); say("BIRTH: " + txt); birth_posted = True
    if b:
        try:
            shutil.copy2(os.path.join(RUN, "collapse_%s_fields.npz" % a.tag), os.path.join(M, ".stopwatch_fields.npz"))
            F = np.load(os.path.join(M, ".stopwatch_fields.npz"), allow_pickle=True)
            g = GC.gate_profile(J, F, rs, a.sigma, b, sixterm=six)
        except Exception as e:
            say("gate read failed (%s); retry" % e); time.sleep(60); continue
        Tj = max(r["T"] / rs for r in J["records"] if r.get("E") is not None)
        if g["T_gate"] is not None:
            past = max(b["T_hi"], g["T_gate"])
            say("gate: T_gate %.4f, window %s, json to %.4f" % (g["T_gate"], GC.fmt(g["window"], "%.4f"), Tj))
            if Tj > past + 1e-9:
                stop("the plan's end: T_gate + 1 record", {"T_birth": b["T_birth"], "R_birth": b["R_birth"], "birth": b, "T_gate": g["T_gate"],
                                                           "window": g["window"], "rows": g["rows"], "last_T": last_T, "json_T": Tj}); break
    time.sleep(a.period)
