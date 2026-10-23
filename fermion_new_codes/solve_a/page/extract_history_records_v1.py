#!/usr/bin/env python
"""extract_history_records_v1.py (builder analyst, 2026-10-16; inbox #36 F): the per-record trapping readouts of the solved history that
Computation A runs on (the r_s/a 64, tau 8, sigma/a 4 production C3prod19 r11, its record json), copied from the json into a small file for
the report and the figure: per record T, T/r_s, E, the outer trapping root R_t, kappa_T, 4 g E kappa_T, min(N - |v|), n_pocket; the
formation clock (the first record with a trapping root) and the trapped episodes (maximal runs of consecutive records with a root). Nothing
is computed beyond the episode grouping. Usage: python extract_history_records_v1.py RECORD_JSON OUT_JSON [ORIGIN (the pod path of the json)]"""
import hashlib
import json
import sys


def main():
    src, out = sys.argv[1:3]
    origin = sys.argv[3] if len(sys.argv) > 3 else src
    raw = open(src, "rb").read()
    d = json.loads(raw)
    recs = d["records"]
    keep = ("T", "T_over_rs", "E", "R_t", "kappa_T", "four_gE_kappa_T", "min_N_minus_absv", "n_pocket")
    rows = [{k: r.get(k) for k in keep} for r in recs]
    eps = []; cur = None
    for i, r in enumerate(rows):
        if r["R_t"] is not None:
            if cur is None:
                cur = {"first": i, "last": i}
            else:
                cur["last"] = i
        elif cur is not None:
            eps.append(cur); cur = None
    if cur is not None:
        eps.append(cur)
    episodes = [{"T_over_rs_first": rows[e["first"]]["T_over_rs"], "T_over_rs_last": rows[e["last"]]["T_over_rs"], "records": e["last"] - e["first"] + 1,
                 "R_t_first": rows[e["first"]]["R_t"], "R_t_max": max(rows[j]["R_t"] for j in range(e["first"], e["last"] + 1))} for e in eps]
    form = next((r for r in rows if r["R_t"] is not None), None)
    res = {"source": origin, "copied_from": src, "source_md5": hashlib.md5(raw).hexdigest(), "script": d.get("script"), "n_records": len(rows), "rows": rows,
           "formation": form, "episodes": episodes,
           "label": "copied from the record json; the formation clock is the first record with a trapping root R_t; an episode is a maximal run of consecutive records with a root"}
    json.dump(res, open(out, "w"), indent=1)
    print("wrote %s: %d records, formation at T/r_s %s (R_t %s), %d episodes: %s" % (out, len(rows), form and form["T_over_rs"], form and form["R_t"], len(episodes),
                                                                                     ["%s-%s" % (e["T_over_rs_first"], e["T_over_rs_last"]) for e in episodes]))


if __name__ == "__main__":
    main()
