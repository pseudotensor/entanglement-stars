#!/usr/bin/env python
"""Strand 7 S7a addendum (builder analyst): the records at each run's D2 flag (strand7_shell_v1.py's JSON), with the shell's coordinate extent
in grid spacings and in units of A (the median, 1st and 99th percentile radii r_sh, r_p01, r_p99), N and N psi at the median and the outer
radius, both characteristic speeds, the exterior form's lapse-zero areal radius (A + B)^2/B over r_s, B/A - 1, the mean u^2 and the solve's
counts; and the same quantities at the last record before the flag.  Numbers emitted, not retyped."""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    head = ["tag", "record", "T/r_s", "h", "r_sh/h", "r_p01/h", "r_p99/h", "r_sh/A", "R_sh/r_s", "4A/r_s", "(A+B)^2/B / r_s", "B/A-1", "N(r_sh)",
            "N psi(r_sh)", "N(r_p99)", "c_+^(r)", "c_+^(R)", "<u^2>", "outer", "newton"]
    rows = []
    for p in sorted(glob.glob(os.path.join(a.dir, "s7a_*.json"))):
        d = json.load(open(p))
        if "run" not in d or "D2" not in d["run"]["kills"]:
            continue
        tag = os.path.basename(p)[4:-5]
        rs, h = d["meta"]["r_s"], d["meta"]["h"]
        recs = d["run"]["records"]
        Tf = d["run"]["kills"]["D2"]["T_over_rs"]
        k = min(range(len(recs)), key=lambda i: abs(recs[i]["T_over_rs"] - Tf))
        for lab, q in (("at the flag", recs[k]), ("one record before", recs[k - 1])):
            rows.append([tag, lab, "%.4f" % q["T_over_rs"], "%g" % h, "%.4f" % (q["r_sh"] / h), "%.4f" % (q["r_p01"] / h), "%.4f" % (q["r_p99"] / h),
                         "%.4f" % (q["r_sh"] / q["A"]), "%.6f" % (q["R_sh"] / rs), "%.6f" % (q["R_trap"] / rs),
                         "%.6f" % (q["R_lapse0_ext"] / rs) if q["R_lapse0_ext"] else "-", "%.5f" % q["B_over_A_minus_1"], "%.5f" % q["N_sh"],
                         "%.5f" % q["Npsi_sh"], "%.5f" % q["N_out"], "%.5f" % q["c_plus_r"], "%.5f" % q["c_plus_R"], "%.5f" % q["u2"], str(q["outer"]),
                         str(q["newton"])])
    out = ["# Strand 7 S7a addendum: the records at the D2 flag (strand7_shell_d2_v1.py over %s)" % a.dir, "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
