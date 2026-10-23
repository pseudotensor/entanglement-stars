#!/usr/bin/env python3
"""Per-record flags of an engine_collapse_v36c run (spec 52.1 (2)); light parsing of the run's json only.

For each record at T/r_s >= --from: the solve's round and held set, evaluations, kept and confirming
residuals, the record's set signature against the previous record's, and the flags:
  KEPT>1e-07   the kept residual above 1e-07 (the record's own solve error, 52.1 (2))
  ROUND>=2     the record was met in round 2 or 3 (a set-change round; posted with its gaps)
  TWO-SETS     the engine's label "two self-consistent sets"
  FAR=PREV     the round's far-side start set is the previous record's set (the record left it)
  CROSS-LOWER  a crossing trial (not taken) had a residual below the held-set trial accepted at that
               Newton step (a far-side set the round did not solve; a fact, not a verdict)
  UNCONVERGED  converged False or the stop rule not met
Usage: flags_v36c_records_v1.py RUN.json [--from 18.25] [--out PREFIX]
"""
import argparse, json


def rec_flags(x):
    s = x.get("solve_v32", {}).get("solves", [{}])
    s = s[-1] if s else {}
    v = x.get("v36c", {})
    rounds = s.get("v36c", {}).get("rounds", []) if isinstance(s.get("v36c"), dict) else []
    kept = s.get("kept_residual")
    conf = s.get("confirming_residual")
    flags = []
    if kept is not None and float(kept) > 1e-07:
        flags.append("KEPT>1e-07")
    nround = max([r.get("round", 1) for r in rounds] or [1])
    status = str(s.get("status", ""))
    if nround >= 2 or "round 2" in status or "round 3" in status:
        flags.append("ROUND>=2")
    blob = json.dumps(x)
    if "two self-consistent sets" in blob:
        flags.append("TWO-SETS")
    gaps = []
    for ls in s.get("line_search", []) or []:
        trials = ls.get("trials", [])
        acc = [t for t in trials if not t.get("crosses")]
        crossing = [t for t in trials if t.get("crosses")]
        if ls.get("gap_at_iterate") is not None:
            gaps.append(ls["gap_at_iterate"])
        if acc and crossing:
            ra = min(t["residual"] for t in acc)
            rc = min(t["residual"] for t in crossing)
            if rc < ra and "CROSS-LOWER" not in flags:
                flags.append("CROSS-LOWER")
    if not x.get("converged", True) or str(s.get("meets_stop_rule")) == "False":
        flags.append("UNCONVERGED")
    vc = s.get("v36c", {}) if isinstance(s.get("v36c"), dict) else {}
    fs = [r.get("far_start") for r in rounds if r.get("far_start")]
    far = ("set %s res %.2e" % (fs[0].get("set_id"), fs[0].get("residual"))) if fs else "none"
    if fs:
        sig = {q["id"]: q["signature"] for q in vc.get("sets", [])}
        if sig.get(fs[0].get("set_id")) == v.get("previous_record_set"):
            far += " (= previous record's set)"
            flags.append("FAR=PREV")
    return dict(T=x["T_over_rs"], far_start=far, label=vc.get("label"), status=status, rounds=nround,
                held=[r.get("held_set") for r in rounds], evaluations=s.get("evaluations"),
                kept=kept, confirming=conf, confirming_dE=s.get("confirming_dE"),
                set=v.get("record_set"), equals_previous=v.get("equals_previous_record"),
                smallest_gap=(min(gaps) if gaps else None), E_over_E0=x.get("E_over_E0"),
                R_t=x.get("R_t"), min_N_minus_absv=x.get("min_N_minus_absv"),
                n_pocket=x.get("n_pocket"), inversions=x.get("inversions_total"), flags=flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--from", dest="t0", type=float, default=18.25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.run_json))
    rows = [rec_flags(x) for x in d["records"] if x["T_over_rs"] >= a.t0 - 1e-9]
    lines = ["# v36c per-record flags (spec 52.1 (2)), from T/r_s %.4f" % a.t0, "",
             "Source: %s; flags_v36c_records_v1.py; every number the run json's." % a.run_json, "",
             "| T/r_s | status | far-side start (not solved) | rounds; held sets | evaluations | kept | confirming | confirming dE/E | set equals previous | smallest gap | E/E0 | R_t | min(N-|v|) | flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda z: "-" if z is None else ("%.2e" % float(z))
    for r in rows:
        lines.append("| %.4f | %s | %s | %d; %s | %s | %s | %s | %s | %s | %s | %.7f | %.3f | %+.4f | %s |" % (
            r["T"], r["status"], r["far_start"], r["rounds"], r["held"], r["evaluations"], f(r["kept"]), f(r["confirming"]),
            f(r["confirming_dE"]), r["equals_previous"], f(r["smallest_gap"]), r["E_over_E0"], r["R_t"],
            r["min_N_minus_absv"], ", ".join(r["flags"]) or "none"))
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        open(a.out + ".md", "w").write(txt)
        json.dump(rows, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
