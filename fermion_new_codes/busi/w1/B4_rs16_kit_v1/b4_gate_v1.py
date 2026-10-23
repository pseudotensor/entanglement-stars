"""b4_gate_v1.py (busi f52b1152, 20 Oct; REVIEWA_B4_rs16_v1.md, "Gate and stop rule"): the 32 r_s gate of B4's small member, run beside the
engine by launch_B4_history_v1.sh. It polls the engine's record json (written every 5 records, so the 256th record, T/r_s 32, is first
readable in the dump after record 260) and decides once:
  PASS  R_t non-null at any record with T/r_s <= the gate clock: the run continues untouched; the first such record is the ladder's clock origin.
  FAIL  a record at T/r_s >= the gate clock exists and R_t is null at every record up to it: the engine (its own pid, checked by its cmdline)
        is stopped (SIGTERM, SIGKILL after 120 s), the min(N - |v|) trace with its floor and flat clock is written, exit 3.
  ENDED the engine exited before a decision: exit 4 (the launcher reports the engine's rc).
Outputs: <prefix>_gate.json and <prefix>_gate.md (the decision; on FAIL the trace). Exit 0 on PASS.
Usage: python -I b4_gate_v1.py --json <record json> --pid <engine pid> --tag <tag> --prefix <out prefix> [--gate-rs 32] [--poll-s 30] [--flat-tol 0.01]"""
import argparse, json, os, signal, time


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        st = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[0]
        return st != "Z"
    except OSError:
        return False


def cmdline(pid):
    try:
        return open("/proc/%d/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def read_records(path):
    try:
        with open(path) as fh:
            return json.load(fh).get("records", [])
    except (OSError, ValueError):
        return None


def trace_summary(recs, flat_tol):
    tr = [(float(r["T_over_rs"]), r.get("min_N_minus_absv"), r.get("r_min_N_minus_absv")) for r in recs if r.get("min_N_minus_absv") is not None]
    if not tr:
        return {"n": 0}
    m = [x[1] for x in tr]
    i_floor = min(range(len(m)), key=lambda i: m[i])
    flat = None
    for i in range(len(m)):
        rest = m[i:]
        if max(rest) - min(rest) <= flat_tol:
            flat = tr[i][0]; break
    return {"n": len(tr), "floor": m[i_floor], "floor_T_over_rs": tr[i_floor][0], "floor_r": tr[i_floor][2], "last": m[-1],
            "flat_from_T_over_rs": flat, "flat_tol": flat_tol, "trace": tr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--pid", type=int, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--gate-rs", type=float, default=32.0)
    ap.add_argument("--poll-s", type=float, default=30.0); ap.add_argument("--flat-tol", type=float, default=0.01)
    a = ap.parse_args()
    eps = 1e-9
    while True:
        recs = read_records(a.json) if os.path.exists(a.json) else None
        if recs:
            upto = [r for r in recs if float(r["T_over_rs"]) <= a.gate_rs + eps]
            roots = [r for r in upto if r.get("R_t") is not None]
            reached = any(float(r["T_over_rs"]) >= a.gate_rs - eps for r in recs)
            if roots:
                r0 = roots[0]
                out = {"decision": "PASS", "gate_rs": a.gate_rs, "first_root_T_over_rs": r0["T_over_rs"], "first_root_R_t": r0["R_t"],
                       "records_read": len(recs), "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                json.dump(out, open(a.prefix + "_gate.json", "w"), indent=1)
                open(a.prefix + "_gate.md", "w").write("B4 gate %s: PASS at T/r_s %.4f (R_t %.4f), the ladder's clock origin; the run continues untouched (%s)\n" % (
                    a.tag, r0["T_over_rs"], r0["R_t"], out["wall"]))
                print("B4 gate PASS: first root at T/r_s %.4f" % r0["T_over_rs"], flush=True)
                return 0
            if reached:
                cl = cmdline(a.pid)
                killed = False
                if alive(a.pid) and "engine_collapse_v19.py" in cl and a.tag in cl:
                    os.kill(a.pid, signal.SIGTERM); killed = True
                    for _ in range(120):
                        if not alive(a.pid):
                            break
                        time.sleep(1)
                    if alive(a.pid):
                        os.kill(a.pid, signal.SIGKILL)
                s = trace_summary(upto, a.flat_tol)
                later = [r["T_over_rs"] for r in recs if float(r["T_over_rs"]) > a.gate_rs + eps and r.get("R_t") is not None]
                out = {"decision": "FAIL", "gate_rs": a.gate_rs, "records_to_gate": len(upto), "killed_pid": a.pid if killed else None,
                       "root_after_gate_in_dump": later, "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "min_N_minus_absv": s}
                json.dump(out, open(a.prefix + "_gate.json", "w"), indent=1)
                lines = ["# B4 gate %s: FAIL, R_t null at every record to T/r_s %g (%d records); engine pid %s stopped (%s)" % (
                    a.tag, a.gate_rs, len(upto), a.pid if killed else "(already gone)", out["wall"]), "",
                    "The ladder's resolution floor at the paper's mass (the ruling's card entry): min(N - |v|) floor %.6f at T/r_s %.4f (r %.2f); "
                    "flat (within %.3f to the gate) from T/r_s %s; last %.6f." % (s["floor"], s["floor_T_over_rs"], s["floor_r"] or float("nan"), a.flat_tol,
                                                                                 s["flat_from_T_over_rs"], s["last"]), "",
                    "| T/r_s | min(N - abs v) | at r |", "|---|---|---|"]
                lines += ["| %.4f | %.6f | %s |" % (t, m, ("%.2f" % r) if r is not None else "-") for t, m, r in s["trace"]]
                open(a.prefix + "_gate.md", "w").write("\n".join(lines) + "\n")
                print("B4 gate FAIL: no root to T/r_s %g; floor %.6f; engine stopped" % (a.gate_rs, s["floor"]), flush=True)
                return 3
        if not alive(a.pid):
            recs = read_records(a.json) or []
            json.dump({"decision": "ENDED", "records_read": len(recs)}, open(a.prefix + "_gate.json", "w"), indent=1)
            return 4
        time.sleep(a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
