#!/usr/bin/env python
"""engine_records_summary_v1.py (builder analyst): a collapse run's records from T/rs --from on, one line each, printed from its json: E/E0,
dE/dT, R_t against 2gE, r_in, min(N - |v|), R_c, R_sh, the pairs' shell energy, the pockets and inversions summed over the channels, whether
the occupation masks equal the negative-energy set, the continuation's label, the unconverged flag, the gauge label. Args: --run dir/tag --from --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    rs = float(J["args"]["rs"]); E0 = float(J["head"]["E0"])
    L = ["## %s from T/rs %g (engine_records_summary_v1.py; %s)" % (tag, a.t0, J.get("script")), "",
         "| T/rs | E/E0 | dE/dT | R_t | 2gE | R_t/2gE | r_in | min(N - abs v) | R_c | R_sh | pairs | pockets | inversions | mask = neg | continuation | unconverged | gauge |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g = float(J["args"]["g"])
    for r in J["records"]:
        if r.get("E") is None or r["T"] / rs < a.t0 - 1e-9:
            continue
        f2 = lambda x, f: (f % x) if x is not None else "-"
        L.append("| %.3f | %.7f | %s | %s | %.3f | %s | %s | %.4f | %s | %s | %.3e | %s | %s | %s | %s | %s | %s |" % (
            r["T"] / rs, r["E"] / E0, f2(r.get("dE_dT"), "%.3e"), f2(r.get("R_t"), "%.3f"), 2 * g * r["E"], f2(r.get("R_t_over_2gE"), "%.4f"), f2(r.get("r_in"), "%.3f"),
            r["min_N_minus_absv"], f2(r.get("R_c"), "%.2f"), f2(r.get("R_sh"), "%.1f"), r["E_pairs_shells"], sum((r.get("n_pocket") or {}).values()),
            r.get("inversions_total", "-"), r.get("mask_equals_negative_set", "-"), (r.get("continuation_label") or (r.get("intersample_label") or "-"))[:70],
            ("%.1e" % r["unconverged"]["residual"]) if r.get("unconverged") else "-", (r.get("gauge") or {}).get("label", "-") if isinstance(r.get("gauge"), dict) else "-"))
    L += ["", "- every number above is read from the run's json by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-14:]))


if __name__ == "__main__":
    main()
