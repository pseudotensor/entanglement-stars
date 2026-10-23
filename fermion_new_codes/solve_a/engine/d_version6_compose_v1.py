#!/usr/bin/env python
"""d_version6_compose_v1.py (the lane's analyst builder; engine spec 41.2 (2): "D version 6 (a compose, no new run): per slice the dust columns
of D_dust_tail_v1 beside the pairs' signed energies and numbers by band at the three windows, with the energy per particle of the incoming and
the outgoing positive-band parts (a reflection's signature: equal)"): from the dust-tail jsons (dust_tail_v1.py), D version 5's jsons
(engine_massive_channel_D_v5) and D_return_marker_v2's json (the labels from its marker), per continuation slice (clock order; the production's
slices 20 to 40 beside where the dust table has them): the dust's weight beyond 16 and 24 r_s, its energy per particle (m), its number current
through 16 r_s; the pairs beyond 24 r_s at each window (16, 32, 64): the positive band's outgoing and incoming energy and number and each part's
energy per particle E/n in units of m, the negative band's two energies; the label. Every number printed by the script. Usage: --dust a.json,...
--v5 a.json,... --marker m.json --m 0.2 --md --out-json."""
import argparse
import json
import os
import time

WINDOWS = ("16", "32", "64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dust", required=True); ap.add_argument("--v5", required=True); ap.add_argument("--marker", required=True)
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    for p in (a.md, a.out_json):
        if os.path.exists(p):
            raise SystemExit("refusing to overwrite %s" % p)
    dust = {}
    for p in a.dust.split(","):
        for r in json.load(open(p))["rows"]:
            src = "continuation" if "ext" in os.path.basename(r["slice"]) else "production"
            dust[(src, round(r["T_over_rs"], 6))] = r
    pairs = {}
    for p in a.v5.split(","):
        for r in json.load(open(p))["slices"]:
            src = "production" if r["baseline"] else "continuation"
            pairs[(src, round(r["T_over_rs"], 6))] = r
    M = json.load(open(a.marker)); mk = M["marker_T_over_rs"]
    keys = sorted(set(dust) | set(pairs), key=lambda k: (k[1], k[0] == "continuation"))
    out = {"script": "d_version6_compose_v1.py", "marker_T_over_rs": mk, "rows": []}
    T = ["# D version 6: the dust's tail beside the pairs' signed band energies beyond 24 r_s (engine spec 41.2 (2); a compose, no new run)", "",
         "Composed %s by code/engine/d_version6_compose_v1.py from %s; %s; %s; every number printed by the script. Weighted over the channels (16 kappa). The dust: D's stored "
         "readouts beyond 16 and 24 r_s and its exact number current through 16 r_s (dust_tail_v1). The pairs: D version 5's signed projection beyond 24 r_s (the positive band's "
         "outgoing and incoming parts, energy E and number n, E/n in units of m; a reflection's signature is equal E/n in the two parts). The label from D_return_marker_v2 "
         "(the marker T/r_s %s)." % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), a.dust, a.v5, a.marker, mk), "",
         "| T/r_s | source | label | dust beyond 16: weight; E/particle (m) | dust beyond 24: weight | dust F through 16 r_s | L | + out: E; n; E/n (m) | + in: E; n; E/n (m) | - out E; - in E |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        src, t = k
        d = dust.get(k) or {}
        b16 = d.get("beyond_16") or {}; b24 = d.get("beyond_24") or {}
        lab = "after the chain end's first return" if (mk is not None and src == "continuation" and t >= mk - 1e-9) else "-"
        pr = pairs.get(k)
        row = {"T_over_rs": t, "source": src, "label": lab, "dust": {"beyond_16": b16, "beyond_24": b24, "F_16": d.get("F_dust_16_weighted")}, "pairs": {}}
        for L in WINDOWS:
            if pr is None:
                cells = "- | - | -"
            else:
                s = pr["weighted"][L]
                eo, ei, no, ni = s["E_plus_out"], s["E_plus_in"], s["n_plus_out"], s["n_plus_in"]
                epo = (eo / no / a.m) if no else None; epi = (ei / ni / a.m) if ni else None
                row["pairs"][L] = {"E_plus_out": eo, "n_plus_out": no, "E_per_n_out_over_m": epo, "E_plus_in": ei, "n_plus_in": ni, "E_per_n_in_over_m": epi,
                                   "E_minus_out": s["E_minus_out"], "E_minus_in": s["E_minus_in"]}
                f = lambda x: ("%.3f" % x) if x is not None else "-"  # noqa: E731
                cells = "%+.4e; %+.4e; %s | %+.4e; %+.4e; %s | %+.4e; %+.4e" % (eo, no, f(epo), ei, ni, f(epi), s["E_minus_out"], s["E_minus_in"])
            T.append("| %.3f | %s | %s | %s; %s | %s | %s | %s | %s |" % (
                t, src, lab, ("%.4e" % b16["dust_weight"]) if b16.get("dust_weight") is not None else "-",
                ("%.4f" % b16["E_per_particle_over_m"]) if b16.get("E_per_particle_over_m") is not None else "-",
                ("%.4e" % b24["dust_weight"]) if b24.get("dust_weight") is not None else "-",
                ("%+.4e" % d["F_dust_16_weighted"]) if d.get("F_dust_16_weighted") is not None else "-", L, cells))
        out["rows"].append(row)
    T += ["", "Signed numbers on a band are windowed effects of the deviation, not counts of particles; E/n is printed only as the ratio of the two printed sums. No mutual information (D15)."]
    open(a.md, "x").write("\n".join(T) + "\n")
    json.dump(out, open(a.out_json, "x"), indent=1, default=float)
    print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
