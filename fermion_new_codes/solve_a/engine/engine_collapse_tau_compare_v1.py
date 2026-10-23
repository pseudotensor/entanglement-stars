#!/usr/bin/env python
"""engine_collapse_tau_compare_v1.py (builder analyst; spec 18.5's tau rule: "Tau 2 runs only if the tau 4 and tau 8 runs differ at matched
times by more than 1e-4 relative in E, 0.1 lattice unit in R_t, or 1 percent in the fitted late rate"): two or more collapse runs of the
same configuration at different tau, compared at the times all of them recorded (multiples of the largest tau): E (relative), R_t (where
both have one, else which has one), min(N - |v|), R_c, R_sh, the pair energy; per pair of runs the largest differences and the first
time each threshold is exceeded; the times at which each run has a trapping surface. The late rate (the chart map's exponent) is not a
record entry and is not compared here (labelled). Args: --runs dir/tag ... (the first is the reference) --md."""
import argparse
import json
import os



def load(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    return tag, J, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = [load(r) for r in a.runs]
    rs = float(runs[0][1]["args"]["rs"])
    common = sorted(set.intersection(*[set(r[2].keys()) for r in runs]))
    L = ["## The tau ladder at matched times (engine_collapse_tau_compare_v1.py; %s)" % ", ".join("%s (tau %g, %d records to T/rs %.2f)" % (
        t, J["args"]["tau"], len(R), max(R) / rs if R else 0) for t, J, R in runs), "",
         "Thresholds of spec 18.5: 1e-4 relative in E, 0.1 lattice unit in R_t (the late rate is not compared here: it is not a record entry).", ""]
    hdr = "| T/rs | " + " | ".join("E/E0 %s" % J["args"]["tau"] for _, J, _ in runs) + " | " + " | ".join("R_t %s" % J["args"]["tau"] for _, J, _ in runs) + \
          " | " + " | ".join("min(N-abs v) %s" % J["args"]["tau"] for _, J, _ in runs) + " | " + " | ".join("R_c %s" % J["args"]["tau"] for _, J, _ in runs) + " |"
    L += [hdr, "|" + "---|" * (1 + 4 * len(runs))]
    E0 = [J["head"]["E0"] for _, J, _ in runs]
    for T in common:
        row = [runs[i][2][T] for i in range(len(runs))]
        f = lambda x, s: (s % x) if x is not None else "-"
        L.append("| %.3f | %s | %s | %s | %s |" % (T / rs, " | ".join("%.7f" % (q["E"] / E0[i]) for i, q in enumerate(row)),
                                                   " | ".join(f(q.get("R_t"), "%.3f") for q in row), " | ".join(f(q.get("min_N_minus_absv"), "%.4f") for q in row),
                                                   " | ".join(f(q.get("R_c"), "%.2f") for q in row)))
    L += ["", "### Largest differences against the reference (%s, tau %g)" % (runs[0][0], runs[0][1]["args"]["tau"]), ""]
    for j in range(1, len(runs)):
        dE = []; dRt = []; mism = []; first_E = first_R = None
        for T in common:
            p, q = runs[0][2][T], runs[j][2][T]
            e = abs(q["E"] - p["E"]) / abs(p["E"]); dE.append(e)
            if first_E is None and e > 1e-4:
                first_E = T / rs
            if p.get("R_t") and q.get("R_t"):
                r = abs(q["R_t"] - p["R_t"]); dRt.append(r)
                if first_R is None and r > 0.1:
                    first_R = T / rs
            elif bool(p.get("R_t")) != bool(q.get("R_t")):
                mism.append(T / rs)
        L.append("- %s (tau %g): max E rel %.2e (first above 1e-4 at T/rs %s); max R_t difference %s (first above 0.1 at T/rs %s); a trapping surface in only one of the two at T/rs %s." % (
            runs[j][0], runs[j][1]["args"]["tau"], max(dE) if dE else float("nan"), ("%.2f" % first_E) if first_E is not None else "never",
            ("%.3f" % max(dRt)) if dRt else "- (no common trapping surface)", ("%.2f" % first_R) if first_R is not None else "never",
            ", ".join("%.2f" % x for x in mism) if mism else "none"))
    L += ["", "### When each run has a trapping surface", ""]
    for t, J, R in runs:
        Ts = sorted(T / rs for T, r in R.items() if r.get("R_t"))
        L.append("- %s (tau %g): %d of %d records with R_t%s" % (t, J["args"]["tau"], len(Ts), len(R), (", T/rs %.2f to %.2f" % (Ts[0], Ts[-1])) if Ts else ""))
    L += ["", "- every number above is read from the runs' json by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(runs) + 8):]))


if __name__ == "__main__":
    main()
