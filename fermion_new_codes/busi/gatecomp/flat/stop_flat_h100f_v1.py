#!/usr/bin/env python
"""stop_flat_h100f_v1.py (busi, GATECOMP; the flat member's live gate read and stop rule; ReviewA_gatecomp_v1 sec. 3 items 1, 2, 7,
ReviewA_flatq_v1 sec. 3; the coordinator's three orders for F256_s8). From stop_w256_h100f_v1.py (the W256 stopper), ON THE CARD, read
only until it acts. Every PERIOD s it reads the member's run json and (from the birth) its fields npz, both rewritten by the engine every
5 records, and:
 (1) BIRTH (plan item 2; gate_common_v1.birth): T_birth by the linear zero of min(N - |v|) through the bracketing records, R_birth at the
     crossing; written to BIRTH.txt once.
 (2) THE GATE on every record from T_birth - 1 (gate_common_v1.gate_profile: the dense fields re-solved from each record's eps_tot, J_tot
     by the field module's ivp, the code of record's six-term max(|v'/N|, |v/(N r)|, |N'/N|, sqrt|D k_r|, sqrt|D k_t|, sqrt|N''/N|) on the
     nodes r = sigma, 2 sigma, ... up to 2 R_t (gated), the r < sigma values reported beside; sigma ||K||_op over (0, 2 R_t]; the one-term
     sigma |H|): T_gate per test (the first record with the test above 0.1), the window T_gate - T_birth per test; T_gate of the STOP rule
     = the first of the six-term (nodes) and sigma ||K||_op (as the W256 stopper). GATE_LIVE.json and GATE_LIVE.tsv every poll.
 (3) THE COLLAR CHARGE (plan item 7; the kit reader's definition: E is the collar charge, the line max |E/E0 - 1| <= 1e-2), per record,
     with E_dust_shells/E0 and E_pairs_shells beside (CHARGE.tsv). If |E/E0 - 1| crosses 0.01 before T_gate: CHARGE_ALERT.txt with the
     numbers, once; the member is NOT stopped for it (reviewer A rules).
 (4) THE STOP: when the json holds a record past T_gate (T_gate + 1 record; with the json's 5-record cadence the stop can come up to 4
     records later), the run files are copied to run_stop_backup/, STOP.json/STOP.txt written, and the engine stopped by PID (TERM, KILL
     after 120 s, its children; never pkill). The engine writes slices only at its listed clocks (every 0.25 r_s from 3.25 to 6.0 here): the
     slice at or before the stop is the resumable state, named in STOP.json. Caps: --t-cap (clock), --wall-cap (UTC).
  stop_flat_h100f_v1.py MEMBER_DIR TAG CODE_DIR [--sigma 8] [--period 120] [--t-cap 6.5] [--wall-cap ...] [--dry] [--once]
"""
import argparse, datetime as dt, glob, json, os, re, shutil, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(os.path.dirname(HERE), "code"), "<internal>/code"):
    if os.path.exists(os.path.join(p, "gate_common_v1.py")):
        sys.path.insert(0, p); break
import gate_common_v1 as GC  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("member"); ap.add_argument("tag"); ap.add_argument("code")
ap.add_argument("--period", type=float, default=120.0); ap.add_argument("--sigma", type=float, default=8.0)
ap.add_argument("--t-cap", dest="t_cap", type=float, default=6.5); ap.add_argument("--rs", type=float, default=256.0)
ap.add_argument("--wall-cap", dest="wall_cap", default="2026-10-23T12:00:00Z"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--once", action="store_true", help="one reading and exit (the dry test)"); ap.add_argument("--line", type=float, default=0.1)
ap.add_argument("--charge-line", dest="charge_line", type=float, default=0.01)
ap.add_argument("--run", default="run", help="the run dir under MEMBER_DIR"); ap.add_argument("--states", default="states")
a = ap.parse_args()
M = a.member; RUN = os.path.join(M, a.run); rs = a.rs; OUT = (M if a.run == "run" else os.path.join(M, a.run)) if not a.dry else os.path.join(M, "dry_stopflat"); os.makedirs(OUT, exist_ok=True)
cap_wall = dt.datetime.strptime(a.wall_cap, "%Y-%m-%dT%H:%M:%SZ")
six = GC.load_sixterm(a.code)


def say(s):
    line = "%s %s" % (dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), s)
    open(os.path.join(OUT, "STOP_WATCH.txt"), "a").write(line + "\n"); print(line, flush=True)


def engine_pid():
    p = os.path.join(RUN, "pid")
    if not os.path.exists(p):
        return None
    sh = int(open(p).read().strip())
    try:
        if "engine_collapse" in open("/proc/%d/cmdline" % sh).read():
            return sh
    except OSError:
        return None
    for k in subprocess.run(["ps", "-o", "pid=,cmd=", "--ppid", str(sh)], capture_output=True, text=True).stdout.split("\n"):
        if "engine_collapse" in k:
            return int(k.split()[0])
    return None


def desc(p):
    k = subprocess.run(["ps", "-o", "pid=", "--ppid", str(p)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in k] + [g for c in k for g in desc(int(c))]


def last_slice(T):
    ts = sorted(float(re.search(r"_T(\d+\.\d+)\.npz$", f).group(1)) for f in glob.glob(os.path.join(M, a.states, "orbitals_%s_T*.npz" % a.tag)))
    ok = [t for t in ts if t <= T + 1e-9]
    return ok[-1] if ok else None


def stop(why, reading):
    pid = engine_pid()
    bk = os.path.join(M, a.run + "_stop_backup")
    if not a.dry:
        os.makedirs(bk, exist_ok=True)
        for f in os.listdir(RUN):
            if f.startswith("collapse_%s" % a.tag) or f == "log.txt":
                shutil.copy2(os.path.join(RUN, f), bk)
    reading.update({"why": why, "engine_pid": pid, "utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "resumable_slice": last_slice(reading.get("json_T") or 99.0)})
    json.dump(reading, open(os.path.join(OUT, "STOP.json"), "w"), indent=1, default=str)
    txt = "STOP (%s): engine pid %s; T_birth %s, R_birth %s, T_gate %s (%s), window %s, json to %s; the resumable slice T %s" % (
        why, pid, reading.get("T_birth"), reading.get("R_birth"), reading.get("T_gate"), reading.get("binding"), reading.get("window"),
        reading.get("json_T"), reading["resumable_slice"])
    open(os.path.join(OUT, "STOP.txt"), "w").write(txt + "\n"); say(txt)
    if a.dry or pid is None:
        say("dry or no pid: not signalled"); return
    kids = desc(pid)
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
            os.kill(k, signal.SIGTERM)
        except OSError:
            pass
    say("engine stopped")


birth_posted = False; alert_posted = os.path.exists(os.path.join(OUT, "CHARGE_ALERT.txt")); seen_charge = set()
say("watching %s (tag %s); sigma %g; six-term %s; caps T %.3f and %s; %s" % (M, a.tag, a.sigma, "on" if six else "OFF", a.t_cap, a.wall_cap, "DRY" if a.dry else "live"))
while True:
    try:
        J = json.load(open(os.path.join(RUN, "collapse_%s.json" % a.tag)))
    except Exception:
        time.sleep(30); continue
    E0 = float(J["head"]["E0"]); recs = sorted([r for r in J["records"] if r.get("E") is not None], key=lambda r: r["T"])
    Tj = recs[-1]["T"] / rs
    with open(os.path.join(OUT, "CHARGE.tsv"), "a") as fc:
        if not seen_charge:
            fc.write("# T_over_rs\tE/E0 (collar)\t|E/E0-1|\tE_dust_shells/E0\tE_pairs_shells\tR_sh\n")
        for r in recs:
            t = round(r["T"] / rs, 6)
            if t in seen_charge:
                continue
            seen_charge.add(t)
            fc.write("%.4f\t%.7f\t%.2e\t%.6f\t%.3e\t%s\n" % (t, r["E"] / E0, abs(r["E"] / E0 - 1), (r.get("E_dust_shells") or float("nan")) / E0,
                                                           r.get("E_pairs_shells") or float("nan"), r.get("R_sh")))
    b = GC.birth(J, rs)
    if b and not birth_posted:
        txt = "T_birth %.5f r_s (bracket %.4f to %.4f, quadratic %s), R_birth %s a = %s of 2GE0 (%.3f a)" % (
            b["T_birth"], b["T_lo"], b["T_hi"], GC.fmt(b["T_quad"], "%.5f"), GC.fmt(b["R_birth"], "%.3f"), GC.fmt(b["R_birth_over_2gE"], "%.4f"), b["two_gE0"])
        open(os.path.join(OUT, "BIRTH.txt"), "w").write(txt + "\n"); say("BIRTH: " + txt); birth_posted = True
    g = None; tg = {}
    if b:
        try:
            shutil.copy2(os.path.join(RUN, "collapse_%s_fields.npz" % a.tag), os.path.join(OUT, ".stopflat_fields.npz"))
            F = np.load(os.path.join(OUT, ".stopflat_fields.npz"), allow_pickle=True)
            g = GC.gate_profile(J, F, rs, a.sigma, b, sixterm=six, line=a.line)
        except Exception as e:
            say("gate read failed (%s); retry" % e); time.sleep(60); continue
        rows = g["rows"]
        first = lambda idx: next((row[0] for row in rows if row[idx] is not None and row[idx] > a.line), None)  # noqa: E731
        tg = {"six_term_nodes": first(5), "sigma_Kop": first(2), "one_term_sigma_H": first(4), "six_term_r_below_sigma (reported)": first(9)}
        cands = [t for t in (tg["six_term_nodes"], tg["sigma_Kop"]) if t is not None]
        T_gate = min(cands) if cands else None
        binding = None if T_gate is None else ("six-term" if tg["six_term_nodes"] == T_gate else "sigma ||K||_op")
        win = {k: (v - b["T_birth"] if v is not None else None) for k, v in tg.items()}
        live = {"T_birth": b["T_birth"], "R_birth": b["R_birth"], "R_birth_over_2gE": b["R_birth_over_2gE"], "birth": b, "T_gate_per_test": tg,
                "window_per_test": win, "T_gate": T_gate, "binding": binding, "window": (T_gate - b["T_birth"]) if T_gate is not None else None,
                "json_T": Tj, "sigma": a.sigma,
                "rows": [{"T": r[0], "rmax": r[1], "sigma_Kop": r[2], "r_Kop": r[3], "sigma_H": r[4], "six_nodes": r[5], "R_t": r[6], "r_six": r[7],
                          "six_dominant": r[8], "six_r_below_sigma": r[9], "r_six_below": r[10], "dT_mode": r[11]} for r in rows]}
        json.dump(live, open(os.path.join(OUT, "GATE_LIVE.json"), "w"), indent=1, default=str)
        with open(os.path.join(OUT, "GATE_LIVE.tsv"), "w") as ft:
            ft.write("# T\tR_t\tsigma||K||_op\tsigma|H|\tsix(nodes)\tr_six\tdominant\tsix(r<sigma, reported)\n")
            for r in rows:
                ft.write("%.4f\t%s\t%.4f\t%.4f\t%s\t%s\t%s\t%s\n" % (r[0], GC.fmt(r[6], "%.2f"), r[2], r[4], GC.fmt(r[5], "%.4f"), GC.fmt(r[7], "%.1f"), r[8], GC.fmt(r[9], "%.4f")))
        say("gate: json to %.4f; T_gate per test %s; binding %s" % (Tj, {k: GC.fmt(v, "%.4f") for k, v in tg.items()}, binding))
        bad = [r for r in recs if abs(r["E"] / E0 - 1) > a.charge_line and r["T"] / rs >= b["T_birth"] - 1 - 1e-9]
        if bad and not alert_posted and (T_gate is None or bad[0]["T"] / rs <= T_gate + 1e-9):
            r0 = bad[0]
            txt = ("CHARGE ALERT (for reviewer A; the member is NOT stopped): |E/E0 - 1| = %.4f at T %.4f (E %.4f, E0 %.4f), before T_gate (%s); "
                   "E_dust_shells/E0 %.5f, E_pairs_shells %.3e, R_sh %s; T_birth %.4f" % (abs(r0["E"] / E0 - 1), r0["T"] / rs, r0["E"], E0, GC.fmt(T_gate, "%.4f"),
                   (r0.get("E_dust_shells") or float("nan")) / E0, r0.get("E_pairs_shells") or float("nan"), r0.get("R_sh"), b["T_birth"]))
            open(os.path.join(OUT, "CHARGE_ALERT.txt"), "w").write(txt + "\n"); say(txt); alert_posted = True
        if T_gate is not None and Tj > max(b["T_hi"], T_gate) + 1e-9:
            live["last_T"] = Tj
            stop("the plan's end: T_gate + 1 record (%s binding)" % binding, live); break
    if Tj >= a.t_cap - 1e-9:
        stop("the clock cap T/r_s %.3f" % a.t_cap, {"json_T": Tj}); break
    if dt.datetime.utcnow() >= cap_wall:
        stop("the wall cap %s" % a.wall_cap, {"json_T": Tj}); break
    if a.once:
        say("--once: one reading done"); break
    time.sleep(a.period)
