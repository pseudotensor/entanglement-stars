#!/usr/bin/env python
"""cmp_interior_v6s.py (busi lane, inbox #309, item 2's gate): the interior-side readout of a page_history_v6s.py run (--interior-readout check, or a
--lean-interior run) against (i) its own checks at the same clocks (the state's exterior block, the dense vacuum's interior and exterior blocks,
the production readout of the same run) and (ii) a stored production run's D13 readout (e.g. v6o_B2_k1_d13.json) at the common clocks.
The gate per clock and cut: |excess_interior - excess_stored| <= floor_low(this run) + floor_low(stored) + slack (1e-10 by default, the entropies'
rounding at the spectrum's edges); the purity identity's sides within the run's low floor + slack.
Usage: python cmp_interior_v6s.py RUN_SUMMARY_JSON STORED_D13_JSON [--kappa 1] [--slack 1e-10] [--out md]"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary"); ap.add_argument("stored"); ap.add_argument("--kappa", default="1"); ap.add_argument("--slack", type=float, default=1e-10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = json.load(open(a.summary)); D = json.load(open(a.stored))
    clocks = D.get("clocks") or {}
    st = {round(float(v["t_over_rs"]), 6): v for v in clocks.values()}
    L = ["# interior readout against the stored production readout", "", "run: %s; stored: %s; kappa %s; slack %g" % (a.summary, a.stored, a.kappa, a.slack), "",
         "| T/r_s | cut | excess interior (this run) | excess stored | diff | tol | gate | state I-E | tol | vacuum dense I-E | contour-dense (I) | interior - own production | interior - own exterior |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    n = 0; bad = 0; worst = {"stored": 0.0, "state": 0.0, "vac": 0.0, "contour": 0.0, "own": 0.0}
    for r in S.get("rows", []):
        pc = (r.get("per_channel") or {}).get(str(a.kappa)) or {}
        ir = pc.get("interior")
        if not ir:
            continue
        trs = round(float(r["t_over_rs"]), 6)
        sv = st.get(trs)
        scuts = ((sv or {}).get("per_channel") or {}).get(str(a.kappa), {}).get("cuts", {}) if sv else {}
        chk = (ir.get("check") or {}).get("cuts") or {}
        for lab, d in ir["cuts"].items():
            q = scuts.get(lab) or {}
            fl_me = d.get("excess_floor_low") or 0.0
            fl_st = q.get("excess_floor_low") or 0.0
            cell_st = "-"; cell_d = "-"; cell_tol = "-"; gate = "-"
            if q.get("excess") is not None:
                diff = d["excess"] - float(q["excess"]); tol = fl_me + fl_st + a.slack
                ok = abs(diff) <= tol; n += 1; bad += (not ok)
                worst["stored"] = max(worst["stored"], abs(diff))
                cell_st = "%.12f" % q["excess"]; cell_d = "%.2e" % diff; cell_tol = "%.2e" % tol; gate = "PASS" if ok else "FAIL"
            c = chk.get(lab) or {}
            cells = []
            if c:
                tol2 = fl_me + a.slack
                for key, wk in (("state_I_minus_E", "state"),):
                    ok2 = abs(c[key]) <= tol2; n += 1; bad += (not ok2); worst[wk] = max(worst[wk], abs(c[key]))
                    cells += ["%.2e" % c[key], "%.2e %s" % (tol2, "PASS" if ok2 else "FAIL")]
                for key, wk, tl in (("vacuum_dense_I_minus_E", "vac", tol2), ("vacuum_contour_minus_dense_I", "contour", a.slack * 10)):
                    ok2 = abs(c[key]) <= tl; n += 1; bad += (not ok2); worst[wk] = max(worst[wk], abs(c[key]))
                    cells.append("%.2e %s" % (c[key], "PASS" if ok2 else "FAIL"))
                if c.get("excess_interior_minus_production") is not None:
                    v = c["excess_interior_minus_production"]; ok2 = abs(v) <= tol2; n += 1; bad += (not ok2); worst["own"] = max(worst["own"], abs(v))
                    cells.append("%.2e %s" % (v, "PASS" if ok2 else "FAIL"))
                else:
                    cells.append("-")
                cells.append("%.2e" % c["excess_interior_minus_exterior"])
            else:
                cells = ["-"] * 6
            L.append("| %.3f | %s | %.12f | %s | %s | %s | %s | %s |" % (trs, lab, d["excess"], cell_st, cell_d, cell_tol, gate, " | ".join(cells)))
    L += ["", "VERDICT: %d of %d comparisons PASS; worst |diff| against stored %.2e, state I-E %.2e, vacuum dense I-E %.2e, contour-dense %.2e, interior-own production %.2e" % (
        n - bad, n, worst["stored"], worst["state"], worst["vac"], worst["contour"], worst["own"])]
    txt = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
