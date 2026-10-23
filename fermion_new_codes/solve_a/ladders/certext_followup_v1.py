#!/usr/bin/env python
"""certext_followup_v1.py (solve lane builder): the check ReviewA_cert_rule_v1.md asks for before the ladder tables recompose.
For each unconverged plain record inside a member's production-rule certified extent (certext_production_v1.json), read the
iteration_history entry after the kept iterate (index unconverged.iteration + 1: the map's change out of the kept iterate's fields)
and report it; the extent shortens where it exceeds 1e-3; a record with no following entry stays kept-only.
The index is verified: iteration_history[unconverged.iteration] must reproduce the stored unconverged.residual as max(dN, dv).
Reads stored json only. Usage: certext_followup_v1.py --certext certext_production_v1.json --out OUT.json"""
import argparse
import json
import math

CAP = 1e-3
BAND = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certext", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = json.load(open(a.certext))
    out = {"script": "certext_followup_v1.py", "certext": a.certext, "members": []}
    for M in C["members"]:
        if M["solver"] != "plain":
            continue
        ext = M["new_extent"]
        J = json.load(open(M["run_json"]))
        R = [r for r in J["records"] if "E" in r and "T_over_rs" in r]
        rows = []
        for r in R:
            if r.get("converged") is True or ext is None or r["T_over_rs"] > ext + 1e-12:
                continue
            u = r.get("unconverged") or {}
            ih = r.get("iteration_history") or []
            k = u.get("iteration")
            kept = u.get("residual")
            idx_ok = (k is not None and k < len(ih) and abs(max(ih[k][1], ih[k][2]) - kept) <= 1e-15 * max(1.0, abs(kept)))
            nxt = (max(ih[k + 1][1], ih[k + 1][2]) if (k is not None and k + 1 < len(ih)) else None)
            rows.append({"T": r["T_over_rs"], "kept": kept, "kept_iteration": k, "history_len": len(ih), "index_verified": idx_ok,
                         "following": nxt, "following_above_1e-3": (nxt is not None and nxt > CAP),
                         "banded_kept": (kept is not None and kept > BAND)})
        first_bad = next((x["T"] for x in rows if x["following_above_1e-3"]), None)
        prev = None
        if first_bad is not None:
            Ts = [r["T_over_rs"] for r in R]
            j = Ts.index(first_bad)
            prev = Ts[j - 1] if j > 0 else None
        out["members"].append({"name": M["name"], "extent": ext, "unconverged_inside": rows, "first_following_above_1e-3": first_bad,
                               "shortened_extent": (prev if first_bad is not None else ext)})
        for x in rows:
            print("%-16s ext %-8s T %-9s kept %.3e it %s/%s idx_ok %s following %s" % (
                M["name"], ext, x["T"], x["kept"], x["kept_iteration"], x["history_len"], x["index_verified"],
                ("%.3e" % x["following"]) if x["following"] is not None else "none"))
    json.dump(out, open(a.out, "w"), indent=1)
    print("shortened:", [(m["name"], m["extent"], m["shortened_extent"]) for m in out["members"] if m["shortened_extent"] != m["extent"]])


if __name__ == "__main__":
    main()
