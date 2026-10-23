#!/usr/bin/env python
"""memwatch_v1.py (busi, GATECOMP, the coordinator's memory watch on h100-f): per record of a running member, its wall split and the
device memory during it. ON THE CARD, read only (the run's log, json, the chain's nvidia-smi log); appends to OUT (csv).
Per record (as its summary line 'T/rs X E/E0' appears in the log, polled every 10 s): the UTC time it appeared and the largest
nvidia-smi memory.used (MiB, the device total: torch's allocations plus its cache and the CUDA context; torch's own per-record peak is not
exposed by the engine, its peak hook prints only at exit) between the previous record's line and this one. When the json is rewritten
(every 5 records): per record seconds, Newton evaluations, eigh_s (eigh_n), continuation_s, solve_s (solve_n), and the rest = seconds - those.
  memwatch_v1.py RUN_DIR TAG LOG SMI_CSV OUT
"""
import csv
import datetime as dt
import json
import os
import re
import sys
import time

run, tag, log, smi, out = sys.argv[1:6]
J = os.path.join(run, "collapse_%s.json" % tag)
seen_lines = None; last_t = None; done_split = set()
pat = re.compile(r"^T/rs ([0-9.]+) E/E0")
new = not os.path.exists(out)
f = open(out, "a"); w = csv.writer(f)
if new:
    w.writerow(["kind", "T_over_rs", "utc", "smi_max_MiB", "seconds", "evaluations", "eigh_s", "eigh_n", "continuation_s", "solve_s", "solve_n", "rest_s"])


def smi_max(t0, t1):
    m = 0
    try:
        for row in open(smi):
            p = [x.strip() for x in row.split(",")]
            if len(p) < 2:
                continue
            try:
                ts = dt.datetime.strptime(p[0][:19], "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue
            if (t0 is None or ts >= t0) and ts <= t1:
                m = max(m, int(p[1].split()[0]))
    except OSError:
        pass
    return m


while True:
    try:
        L = open(log, errors="replace").read().splitlines()
    except OSError:
        L = []
    recs = [m.group(1) for m in (pat.match(x) for x in L) if m]
    now = dt.datetime.utcnow()
    if seen_lines is None:   # records already in the log at the start: no per-record window, marked
        for T in recs:
            w.writerow(["record_before_watch", T, "", ""] + [""] * 8)
        seen_lines = len(recs); last_t = now
    for T in recs[seen_lines:]:
        w.writerow(["record", T, now.strftime("%Y-%m-%dT%H:%M:%SZ"), smi_max(last_t, now)] + [""] * 8); last_t = now
    seen_lines = len(recs)
    try:
        d = json.load(open(J))
        for r in d["records"]:
            T = round(r["T_over_rs"], 6)
            if T in done_split or r.get("seconds") is None:
                continue
            tm = r.get("timing") or {}; sec = float(r["seconds"])
            parts = [float(tm.get(k) or 0.0) for k in ("eigh_s", "continuation_s", "solve_s")]
            w.writerow(["split", T, "", "", "%.1f" % sec, (r.get("solve_v32") or {}).get("evaluations"), "%.1f" % parts[0], tm.get("eigh_n"),
                        "%.1f" % parts[1], "%.1f" % parts[2], tm.get("solve_n"), "%.1f" % (sec - sum(parts))])
            done_split.add(T)
    except Exception:
        pass
    f.flush()
    time.sleep(10)
