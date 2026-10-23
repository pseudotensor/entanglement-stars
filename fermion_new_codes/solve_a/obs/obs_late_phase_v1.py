#!/usr/bin/env python3
"""obs_late_phase_v1.py (builder analyst; engine spec 31.3 ruling (1) and (2), 31.6 BUILDER item 2): the late phase of each history's
exterior over S1's certified late window, for choosing the frozen record of J3' version 7.

Per history (--run label:record_json:fields_npz, the production first): every stored record with T/rs inside --window (28.6 to 39.9, S1's
certified late window on the production, 30.5; the same window applied to the tau 4 record, labelled so with --window-note), with T/rs, R_t,
kappa_T (the record's stored d_R(N + v) at R_t of its dense solve) and E, the difference of the stored kappa_T against the previous stored
record, and kappa_m (the slope of the propagated applied site fields' outermost root, obs_j3p_scope_v2's five-site fit) beside. A record
where the stored kappa_T's difference changes sign is marked: a maximum where the difference to it is positive and the difference to the
next record negative, a minimum for the reverse; at the window's two edges the neighbouring stored records outside the window enter the
differences (labelled). The last maximum and the last minimum inside the window are named on one line per history in the fixed form
"- stationary phases of <label>: last maximum T/rs <x> (kappa_T <y>), last minimum T/rs <z> (kappa_T <w>)", which obs_raymap_solved_v7
reads. The version 6 frozen record (the last stored one) is printed with its label of 31.3 (1). Every number printed by the script; CPU,
seconds. Args: --run (repeatable) --window lo hi --window-note label:text (repeatable) --md --json."""
import argparse
import json
import math
import os

import numpy as np


def outer_root(R, f):
    """obs_j3p_scope_v2's: the outermost negative-to-positive sign change of f on R, refined by a quadratic fit through the nearest five
    sites; (root, slope)."""
    s = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(s) == 0:
        return None, None
    i = int(s[-1])
    lo = max(i - 2, 0); hi = min(i + 3, len(R))
    x = R[lo:hi] - R[i]; c = np.polyfit(x, f[lo:hi], 2)
    roots = np.roots(c); roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= -1e-9) & (roots <= R[i + 1] - R[i] + 1e-9)]
    r0 = float(roots[0]) if len(roots) else -f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i])
    return float(R[i] + r0), float(2 * c[0] * r0 + c[1])


def history_phase(label, record, fields, lo, hi, note):
    D = json.load(open(record))
    rs0 = float(D.get("args", {}).get("rs", 64.0))
    recs = [r for r in D["records"] if r.get("E") is not None]
    Z = np.load(fields, allow_pickle=True)
    Tz = np.asarray(Z["T"], float); Rs = np.asarray(Z["R"], float)
    rows = []
    for r in recs:
        t_ = float(r["T"]) / rs0
        k = int(np.argmin(np.abs(Tz - float(r["T"]))))
        km = None
        if abs(Tz[k] - float(r["T"])) < 1e-6:
            _, km = outer_root(Rs, np.asarray(Z["N_applied"][k], float) + np.asarray(Z["v_applied"][k], float))
        rows.append({"T_over_rs": t_, "R_t": r.get("R_t"), "kappa_T": r.get("kappa_T"), "E": float(r["E"]), "kappa_m": km,
                     "in_window": lo - 1e-9 <= t_ <= hi + 1e-9})
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        r["d_kappa_T"] = (r["kappa_T"] - prev["kappa_T"]) if (prev and r["kappa_T"] is not None and prev["kappa_T"] is not None) else None
    marks = []
    for i, r in enumerate(rows):
        if not r["in_window"] or i + 1 >= len(rows):
            r["mark"] = None
            continue
        d0, d1 = r["d_kappa_T"], rows[i + 1]["d_kappa_T"]
        mark = None
        if d0 is not None and d1 is not None:
            if d0 > 0 > d1:
                mark = "maximum"
            elif d0 < 0 < d1:
                mark = "minimum"
        r["mark"] = mark
        r["edge"] = (not rows[i - 1]["in_window"]) if i > 0 else True
        r["edge"] = r["edge"] or not rows[i + 1]["in_window"]
        if mark:
            marks.append((r["T_over_rs"], mark, r["kappa_T"]))
    maxima = [m for m in marks if m[1] == "maximum"]; minima = [m for m in marks if m[1] == "minimum"]
    win = [r for r in rows if r["in_window"]]
    last = rows[-1]
    return {"label": label, "record": os.path.basename(record), "rs_nominal": rs0, "window": [lo, hi], "note": note, "rows": win,
            "n_window": len(win), "maxima": maxima, "minima": minima, "last_maximum": maxima[-1] if maxima else None,
            "last_minimum": minima[-1] if minima else None, "last_record": last,
            "before_window": next((rows[i - 1] for i, r in enumerate(rows) if r["in_window"] and i > 0), None),
            "after_window": next((rows[i + 1] for i, r in enumerate(rows) if r["in_window"] and i + 1 < len(rows) and not rows[i + 1]["in_window"]), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label:record_json:fields_npz")
    ap.add_argument("--window", type=float, nargs=2, default=[28.6, 39.9])
    ap.add_argument("--window-note", action="append", default=[], help="label:text")
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    notes = dict(n.split(":", 1) for n in a.window_note)
    H = []
    for spec in a.run:
        label, rec, fld = spec.split(":", 2)
        H.append(history_phase(label, rec, fld, a.window[0], a.window[1], notes.get(label, "")))
    json.dump({"script": "obs_late_phase_v1.py", "args": vars(a), "histories": H}, open(a.json, "w"), indent=1, default=float)
    f_ = lambda x, fmt: (fmt % x) if isinstance(x, (int, float)) and x is not None and math.isfinite(x) else "-"
    L = ["## The late phase of the exterior over the certified late window (obs_late_phase_v1.py; engine spec 31.3 (1), (2))", "",
         "- per history: the stored records with T/rs in [%g, %g] (S1's certified late window on the production, 30.5); T/rs, R_t, kappa_T (the record's stored d_R(N + v) at R_t of its dense solve), E, the difference of the stored kappa_T against the previous stored record, and kappa_m (the propagated applied fields' outermost root slope) beside; a record is marked where the stored kappa_T's difference changes sign (a maximum: positive to it, negative to the next record; a minimum: the reverse); at the window's edges the neighbouring stored records outside it enter the differences." % tuple(a.window), ""]
    for h in H:
        lm, ln = h["last_maximum"], h["last_minimum"]
        L += ["### %s (%s)%s" % (h["label"], h["record"], (": " + h["note"]) if h["note"] else ""), "",
              "- stationary phases of %s: last maximum T/rs %s (kappa_T %s), last minimum T/rs %s (kappa_T %s)" % (
                  h["label"], f_(lm[0] if lm else None, "%.4f"), f_(lm[2] if lm else None, "%.6e"), f_(ln[0] if ln else None, "%.4f"), f_(ln[2] if ln else None, "%.6e")),
              "- %d records in the window; the marked records: %s." % (h["n_window"], "; ".join("T/rs %.4f %s (kappa_T %.6e)" % m for m in sorted(h["maxima"] + h["minima"])) or "none"),
              "- the record before the window T/rs %s (kappa_T %s) and after it T/rs %s (kappa_T %s) enter the edge differences; the version 6 frozen record, the last stored one, T/rs %s (kappa_T %s, kappa_m %s): \"frozen at the uncertified one-sided last record, near a maximum of kappa_m\" (31.3 (1))." % (
                  f_((h["before_window"] or {}).get("T_over_rs"), "%.4f"), f_((h["before_window"] or {}).get("kappa_T"), "%.6e"), f_((h["after_window"] or {}).get("T_over_rs"), "%.4f"),
                  f_((h["after_window"] or {}).get("kappa_T"), "%.6e"), f_(h["last_record"]["T_over_rs"], "%.4f"), f_(h["last_record"]["kappa_T"], "%.6e"), f_(h["last_record"]["kappa_m"], "%.6e")), "",
              "| T/rs | R_t | kappa_T | d kappa_T | E | kappa_m (beside) | mark |", "|---|---|---|---|---|---|---|"]
        for r in h["rows"]:
            nm = ""
            if r.get("mark"):
                nm = r["mark"]
                if lm and abs(r["T_over_rs"] - lm[0]) < 1e-9:
                    nm += " (the last maximum)"
                if ln and abs(r["T_over_rs"] - ln[0]) < 1e-9:
                    nm += " (the last minimum)"
                if r.get("edge"):
                    nm += "; at the window's edge"
            L.append("| %.4f | %s | %s | %s | %s | %s | %s |" % (r["T_over_rs"], f_(r["R_t"], "%.4f"), f_(r["kappa_T"], "%.6e"), f_(r["d_kappa_T"], "%+.3e"), f_(r["E"], "%.6f"),
                     f_(r["kappa_m"], "%.6e"), nm))
        L.append("")
    L.append("- every number above is printed by the script from the runs' json records and fields npz.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(l_ for l_ in L if l_.startswith("- stationary") or l_.startswith("### ")))


if __name__ == "__main__":
    main()
