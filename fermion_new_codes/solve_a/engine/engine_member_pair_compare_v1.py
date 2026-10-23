#!/usr/bin/env python
"""engine_member_pair_compare_v1.py (builder analyst; inbox #9 item 3: "stop the flagged plain run once the Newton member has reproduced
its first records"): two engine_collapse run jsons of the same member (--a label=json, --b label=json) compared record by record at the
record times both hold: E/E0, the dust and pairs' shell energies, min(N - |v|) and its radius, R_c, R_sh, R_t (or none), the pockets, the
stop rule and each run's returned residual (max(dN, dv) of the kept iterate); then per quantity the largest absolute difference over the
shared records, and beside it the larger of the two runs' residuals over the same records (the scale at which a difference can come from
the unconverged run's iterate). Every number is printed by the script. Args: --a --b --md."""
import argparse
import json


def resid(r):
    ih = r.get("iteration_history") or []
    if not ih:
        return None
    if r.get("converged"):
        return max(ih[-1][1], ih[-1][2])
    return (r.get("unconverged") or {}).get("residual")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--md", default=None)
    a = ap.parse_args()
    runs = []
    for s in (a.a, a.b):
        lab, p = s.split("=", 1)
        J = json.load(open(p))
        runs.append((lab, p, {round(r["T_over_rs"], 6): r for r in J["records"] if "E" in r and "T_over_rs" in r}, J.get("script"), J["args"].get("mix")))
    (la, pa, Ra, sa, ma), (lb, pb, Rb, sb, mb) = runs
    Ts = sorted(set(Ra) & set(Rb))
    keys = [("E_over_E0", "E/E0"), ("E_dust_shells", "dust"), ("E_pairs_shells", "pairs"), ("min_N_minus_absv", "min(N-|v|)"), ("r_min_N_minus_absv", "its radius"), ("R_c", "R_c"), ("R_sh", "R_sh")]
    out = ["## The same member on two solvers, record by record (engine_member_pair_compare_v1.py)", "",
           "- A: %s (%s, script %s, mix %s); B: %s (%s, script %s, mix %s); %d shared record times" % (la, pa.split("/")[-1], sa, ma, lb, pb.split("/")[-1], sb, mb, len(Ts)), "",
           "| T/rs | " + " | ".join("%s A; B" % k[1] for k in keys) + " | R_t A; B | pockets A; B | stop rule A; B | residual A; B |", "|---|" + "---|" * (len(keys) + 4)]
    dmax = {k[0]: 0.0 for k in keys}
    rmax = 0.0
    for t in Ts:
        ra, rb = Ra[t], Rb[t]
        cells = []
        for k, _ in keys:
            x, y = ra.get(k), rb.get(k)
            cells.append("%.7g; %.7g" % (x, y) if x is not None and y is not None else "%s; %s" % (x, y))
            if x is not None and y is not None:
                dmax[k] = max(dmax[k], abs(x - y))
        pa_ = sum((ra.get("n_pocket") or {}).values()); pb_ = sum((rb.get("n_pocket") or {}).values())
        resa, resb = resid(ra), resid(rb)
        rmax = max(rmax, resa or 0.0, resb or 0.0)
        out.append("| %.3f | %s | %s; %s | %d; %d | %s; %s | %s; %s |" % (t, " | ".join(cells), "%.3f" % ra["R_t"] if ra.get("R_t") is not None else "none",
                   "%.3f" % rb["R_t"] if rb.get("R_t") is not None else "none", pa_, pb_, "met" if ra.get("converged") else "not met", "met" if rb.get("converged") else "not met",
                   "%.2e" % resa if resa is not None else "", "%.2e" % resb if resb is not None else ""))
    out += ["", "- the largest absolute difference over the shared records: " + "; ".join("%s %.3e" % (lab_, dmax[k]) for k, lab_ in keys) +
            "; the larger of the two runs' residuals over the same records %.2e." % rmax, "- every number above is printed by the script."]
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
