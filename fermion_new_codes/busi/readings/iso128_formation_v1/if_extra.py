#!/usr/bin/env python
"""if_extra.py (busi reading of ISO128_formation_readout_v1): the H2 (nkq) history against the plain run's at every common clock
(E/E0, min(N - |v|), R_c), the onset of their separation and the continuation labels of both runs (run-length encoded), read
against the busi route's cross-solver thresholds (ISO_128_nk_route_v2 4.3: |dE/E0| at most 6e-7, |d min(N - |v|)| at most 1e-5).
Own reader (if_common). Writes OUT/extra.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import if_common as C  # noqa: E402


def rle(pairs):
    runs = []
    for t, v in pairs:
        if runs and runs[-1]["value"] == v:
            runs[-1]["last"], runs[-1]["n"] = t, runs[-1]["n"] + 1
        else:
            runs.append({"value": v, "first": t, "last": t, "n": 1})
    return runs


def main():
    H = C.records(C.load(C.H2_JSON))
    P = C.records(C.load(C.MEMBERS["ISO_128_s4_plain"]))
    rows = []
    for r in H:
        t = float(r["T_over_rs"])
        j = C.at_clock(P, t)
        if j is None:
            continue
        p = P[j]
        rows.append({"T": t, "dEE0": float(r["E_over_E0"]) - float(p["E_over_E0"]), "dminNv": float(r["min_N_minus_absv"]) - float(p["min_N_minus_absv"]),
                     "dRc": float(r["R_c"]) - float(p["R_c"]), "h2_label": str(r.get("continuation_label")), "plain_label": str(p.get("continuation_label")),
                     "h2_kept": C.residuals(r, True)["kept"], "h2_conf": C.residuals(r, True)["conf"], "plain_kept": C.residuals(p, False)["kept"],
                     "plain_converged": p.get("converged")})
    out = {"n_common": len(rows), "thresholds": {"dEE0": 6e-7, "dminNv": 1e-5},
           "first_T_dEE0_above_6e-7": next((x["T"] for x in rows if abs(x["dEE0"]) > 6e-7), None),
           "first_T_dminNv_above_1e-5": next((x["T"] for x in rows if abs(x["dminNv"]) > 1e-5), None),
           "first_T_labels_differ": next((x["T"] for x in rows if x["h2_label"] != x["plain_label"]), None),
           "max_abs_dEE0_0_to_3p375": max(abs(x["dEE0"]) for x in rows if x["T"] <= 3.375 + 1e-9),
           "max_abs_dminNv_0_to_3p375": max(abs(x["dminNv"]) for x in rows if x["T"] <= 3.375 + 1e-9),
           "max_abs_dEE0_0_to_1p5": max(abs(x["dEE0"]) for x in rows if x["T"] <= 1.5 + 1e-9),
           "max_abs_dEE0_all": max(abs(x["dEE0"]) for x in rows), "max_abs_dminNv_all": max(abs(x["dminNv"]) for x in rows),
           "n_records_dEE0_above_6e-7_0_to_3p375": sum(1 for x in rows if x["T"] <= 3.375 + 1e-9 and abs(x["dEE0"]) > 6e-7),
           "h2_labels_rle": rle([(x["T"], x["h2_label"][:90]) for x in rows]), "plain_labels_rle": rle([(float(p["T_over_rs"]), str(p.get("continuation_label"))[:90]) for p in P]),
           "rows_1p375_to_2p0": [x for x in rows if 1.375 - 1e-9 <= x["T"] <= 2.0 + 1e-9],
           "rows_selected": [x for x in rows if x["T"] in (2.75, 3.0, 3.375, 5.0, 8.0, 11.0, 11.25)]}
    C.jdump(out, C.OUT + "/extra.json")
    for k, v in out.items():
        if k not in ("rows_1p375_to_2p0", "rows_selected", "h2_labels_rle", "plain_labels_rle"):
            print(k, v)
    print("h2 labels:", [(r["first"], r["last"], r["n"], r["value"]) for r in out["h2_labels_rle"]])
    print("plain labels:", [(r["first"], r["last"], r["n"], r["value"]) for r in out["plain_labels_rle"]])
    for x in out["rows_1p375_to_2p0"]:
        print("  T %.4f dEE0 %+.3e dminNv %+.3e dRc %+.3e h2 kept %.2e plain kept %.2e | %s | %s" % (x["T"], x["dEE0"], x["dminNv"], x["dRc"], x["h2_kept"], x["plain_kept"], x["h2_label"][:70], x["plain_label"][:40]))


if __name__ == "__main__":
    main()
