#!/usr/bin/env python
"""engine_fields_pair_compare_v1.py (builder analyst; engine spec 31.1 (1): "One solver-independence check (minutes): Newton on three
converged records of the sigma 8 member ..., the fields' maximum difference against the stored records printed (expected below the stop
tolerance 1e-6; a larger difference is reported, not read)"): two engine_collapse runs' fields npz files (--a label=npz, --b label=npz),
compared at the record times both hold: per record max |N_a - N_b| and max |v_a - v_b| of the solved fields at the sites, the same for
the accepted (applied) fields, and the largest of each over the shared records; with each run's json (--a-json, --b-json) the stop rule
and the returned residual per record beside. Every number is printed by the script. Args: --a --b --a-json --b-json --md."""
import argparse
import json

import numpy as np


def resid(r):
    ih = r.get("iteration_history") or []
    if not ih:
        return None
    if r.get("converged"):
        return max(ih[-1][1], ih[-1][2])
    return (r.get("unconverged") or {}).get("residual")


def main():
    ap = argparse.ArgumentParser()
    for k in ("a", "b", "a-json", "b-json"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    la, pa = a.a.split("=", 1); lb, pb = a.b.split("=", 1)
    A = np.load(pa, allow_pickle=True); B = np.load(pb, allow_pickle=True)
    Ja = {round(r["T_over_rs"], 6): r for r in json.load(open(a.a_json))["records"] if "T_over_rs" in r}
    Jb = {round(r["T_over_rs"], 6): r for r in json.load(open(a.b_json))["records"] if "T_over_rs" in r}
    rs = float(json.load(open(a.a_json))["args"]["rs"])
    Ta = {round(t / rs, 6): k for k, t in enumerate(np.asarray(A["T"], float))}
    Tb = {round(t / rs, 6): k for k, t in enumerate(np.asarray(B["T"], float))}
    Ts = sorted(set(Ta) & set(Tb))
    assert np.array_equal(np.asarray(A["R"]), np.asarray(B["R"])), "different site grids"
    out = ["## Two solves of the same records, field by field (engine_fields_pair_compare_v1.py; engine spec 31.1 (1))", "",
           "- A: %s (%s); B: %s (%s); %d shared record times" % (la, pa.split("/")[-1], lb, pb.split("/")[-1], len(Ts)), "",
           "| T/rs | max abs dN (solved) | max abs dv (solved) | max abs dN (applied) | max abs dv (applied) | stop rule A; B | residual A; B |", "|---|---|---|---|---|---|---|"]
    worst = [0.0, 0.0, 0.0, 0.0]
    for t in Ts:
        i, j = Ta[t], Tb[t]
        d = [float(np.max(np.abs(np.asarray(A[k][i], float) - np.asarray(B[k][j], float)))) for k in ("N", "v", "N_applied", "v_applied")]
        worst = [max(x, y) for x, y in zip(worst, d)]
        ra, rb = Ja.get(t, {}), Jb.get(t, {})
        out.append("| %.3f | %.3e | %.3e | %.3e | %.3e | %s; %s | %s; %s |" % (t, d[0], d[1], d[2], d[3], "met" if ra.get("converged") else "not met",
                   "met" if rb.get("converged") else "not met", "%.2e" % resid(ra) if resid(ra) is not None else "-", "%.2e" % resid(rb) if resid(rb) is not None else "-"))
    out += ["", "- the largest over the shared records: solved N %.3e, solved v %.3e, applied N %.3e, applied v %.3e (the stop rule's field tolerance 1e-6)." % tuple(worst),
            "- every number above is printed by the script."]
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
