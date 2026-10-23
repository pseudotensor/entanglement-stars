#!/usr/bin/env python
"""flat_stall_watch_v1.py (busi, GATECOMP; ReviewA_w256end_v1 sec. 3 (ii)): the stall watcher every flat member (and the W256_s4
restart) carries with --source-continuation two-stage. ON THE CARD, beside the engine; reads the engine's log as it grows.
Per record (the record line "T/rs X E/E0 ...", with v32's segment "continuation: ...; kept from ...: residual R"): whether the record took
the continuation path ("continuation: ran"), and its kept residual. The member is STOPPED FOR A READ (the engine by its PID, TERM then KILL
after 120 s, and its children; never pkill) when:
  (a) two consecutive records take the path;
  (b) the path's rate exceeds one record in ten: two path records within any 10 consecutive records;
  (c) the kept residual rises a decade within 1 r_s: the largest kept residual over the last 1 r_s of records is at least 10 times the
      largest over the 1 r_s before it, read only where the larger is at or above 1e-6 (below that every record meets the engine's stop
      rule, 1e-7 on E and 1e-6 on the fields; the floor is the watcher's reading, for reviewer A).
A record whose polish fails ends the member by the engine's own --end-above (32.1 (3)); the watcher logs it. Every record is logged to
OUT (tsv: T, path, kept, trigger); a trigger writes OUT.STOP with the reason.
    flat_stall_watch_v1.py LOG ENGINE_PID OUT --rs-records 16 [--from-T T0]"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time

REC = re.compile(r"^T/rs ([0-9.]+) E/E0 ")
CONT = re.compile(r"\); continuation: ([^;\]]*)|; continuation: (ran[^;\]]*|not [^;\]]*|off)")   # the v32 segment's own entry (the record line also names the LZ continuation)
KEPT = re.compile(r"kept from [^:]*: residual ([0-9.eE+-]+|none)")


def stop(pid, why, out):
    open(out + ".STOP", "w").write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), why))
    def desc(p_):   # every descendant (the engine's pair workers; the engine itself where pid is a launching subshell)
        k_ = subprocess.run(["ps", "-o", "pid=", "--ppid", str(p_)], capture_output=True, text=True).stdout.split()
        return k_ + [g for c in k_ for g in desc(c)]
    kids = desc(pid)
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(24):
            time.sleep(5); os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for k in kids:
        try:
            os.kill(int(k), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    print("STOPPED engine %d: %s" % (pid, why), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("pid", type=int); ap.add_argument("out")
    ap.add_argument("--rs-records", type=int, default=16, help="records per r_s (tau = r_s/16)")
    ap.add_argument("--from-T", type=float, default=-1.0, help="records at or below this clock (before the restart) are not counted")
    ap.add_argument("--floor", type=float, default=1e-6)
    a = ap.parse_args()
    recs = []; seen = set(); pos = 0
    fo = open(a.out, "a"); fo.write("# T_over_rs\tpath\tkept\ttrigger (flat_stall_watch_v1, pid %d)\n" % a.pid); fo.flush()
    while True:
        alive = True
        try:
            os.kill(a.pid, 0)
        except ProcessLookupError:
            alive = False
        if os.path.exists(a.log):
            with open(a.log) as f:
                f.seek(pos); chunk = f.read(); pos = f.tell()
            for ln in chunk.splitlines():
                m = REC.match(ln)
                if not m:
                    if "the member ends at this record" in ln:
                        fo.write("# engine end: %s\n" % ln[:200]); fo.flush()
                    continue
                T = float(m.group(1))
                if T <= a.from_T + 1e-9 or T in seen:
                    continue
                seen.add(T)
                cs = [g for mm in CONT.finditer(ln) for g in mm.groups() if g]; k = KEPT.search(ln)
                path = bool(cs and cs[-1].startswith("ran"))
                kept = float(k.group(1)) if (k and k.group(1) != "none") else float("nan")
                recs.append((T, path, kept))
                trig = ""
                if len(recs) >= 2 and recs[-1][1] and recs[-2][1]:
                    trig = "(a) two consecutive records took the continuation path (%.4f, %.4f)" % (recs[-2][0], recs[-1][0])
                elif sum(r[1] for r in recs[-10:]) >= 2:
                    trig = "(b) %d of the last %d records took the path" % (sum(r[1] for r in recs[-10:]), len(recs[-10:]))
                elif len(recs) >= 2 * a.rs_records:
                    w1 = [r[2] for r in recs[-a.rs_records:] if r[2] == r[2]]; w0 = [r[2] for r in recs[-2 * a.rs_records:-a.rs_records] if r[2] == r[2]]
                    if w1 and w0 and max(w1) >= a.floor and max(w1) >= 10.0 * max(w0):
                        trig = "(c) the kept residual rose a decade within 1 r_s: max %.2e over the last 1 r_s against %.2e before" % (max(w1), max(w0))
                fo.write("%.4f\t%d\t%.3e\t%s\n" % (T, path, kept, trig)); fo.flush()
                if trig:
                    stop(a.pid, trig, a.out)
                    return
        if not alive:
            fo.write("# engine pid %d exited\n" % a.pid); fo.flush()
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
