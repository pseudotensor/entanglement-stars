#!/usr/bin/env python
"""engine_iteration_diag_v1.py (builder analyst; for the lead's reading of engine spec 30.3's sigma/a 2 floor): prints, for each given
engine_collapse run json (label=path) and its first n records, the self-consistent iteration's history per record: per iteration
max |N_out - N_in|, max |v_out - v_in| and |dE/E| (the record's iteration_history, the solve's output against the iteration's applied
fields), the smallest max(dN, dv) and its iteration, the last iteration's, and the growth after the minimum: g = (r_last/r_min)^(1/(k_last
- k_min)), the geometric mean factor per iteration of max(dN, dv) from the minimum to the last iterate (g > 1: the iteration moves away
after its minimum). The run's mixer, its step and the field solver's tolerances are read from its args. Every number is printed by the
script. Args: --run label=json (repeated) --n (5) --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    out = ["## The self-consistent iteration at sigma/a 2: histories per record (engine_iteration_diag_v1.py)", ""]
    for s in a.run:
        lab, p = s.split("=", 1)
        J = json.load(open(p))
        A = J["args"]
        out.append("### %s: %s (script %s; mix %s%s; maxit %s; stop |dE/E| < %s, fields < %s; field rtol %s, atol %s; ell %s, prep-ell %s)" % (
            lab, p.split("/")[-1], J.get("script"), A.get("mix"), (" step %s" % A.get("mix_alpha")) if A.get("mix") == "damped" else "", A.get("maxit"),
            A.get("stop_E"), A.get("stop_fields"), A.get("field_rtol", "1e-12 (default)"), A.get("field_atol", "1e-16 (default)"), A.get("ell"), A.get("prep_ell")))
        out.append("")
        out.append("| T/rs | stop rule | min max(dN, dv) (iteration) | last (iteration) | growth per iteration after the min | max dN per iteration | max dv per iteration | abs dE/E per iteration |")
        out.append("|---|---|---|---|---|---|---|---|")
        R = [r for r in J["records"] if r.get("iteration_history") and "E" in r][:a.n]
        for r in R:
            ih = r["iteration_history"]
            m = [max(h[1], h[2]) for h in ih]
            k0 = min(range(len(m)), key=lambda i: m[i])
            kl = len(m) - 1
            g = (m[kl] / m[k0]) ** (1.0 / (kl - k0)) if kl > k0 else None
            out.append("| %.3f | %s | %.2e (%d) | %.2e (%d) | %s | %s | %s | %s |" % (
                r["T_over_rs"], "met" if r.get("converged") else "not met", m[k0], k0, m[kl], kl, "%.3f" % g if g else "none (the minimum is the last)",
                " ".join("%.1e" % h[1] for h in ih), " ".join("%.1e" % h[2] for h in ih), " ".join(("%.0e" % h[0]) if h[0] is not None else "-" for h in ih)))
        out.append("")
    out.append("- every number above is printed by the script.")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
