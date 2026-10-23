#!/usr/bin/env python
"""page_v6i_s4to100_regression_v1.py (builder analyst, 2026-10-17; the lane's choice (a) of 01:08 UTC: "its first 40 r_s printed against s4
as a regression (S_esc expected bitwise, as in s4tr)"): A to T/r_s 100 (v6iA_s4to100, page_history_v6i.py, from T = 0 on the production's
records to T/r_s 40 and B's continuation after it) against A s4 (v6bA_s4, page_history_v6b.py, the production's records to T/r_s 40) at
their common checkpoints up to T/r_s 40. Per checkpoint: S_esc weighted and per channel, equal bit for bit or not, and the largest
differences of E_inv, the collar flow, L, E_remove, M_chi, the energy identity's residual, the archive and the batch per channel. The
traces that v6e sums by column blocks (E_inv, E_out, E_K, the identities) differ from v6b's by rounding only; Tr[h_dot (C - P)] at T/r_s 40
is a forward difference in v6iA (record 321, B's first record after the seam exists) and a backward one in v6bA (record 320 was s4's last),
so it is printed apart. No mutual information is computed or printed (#41 D15). Usage: python page_v6i_s4to100_regression_v1.py
S4_SUMMARY A100_SUMMARY OUT_MD (the json beside it)"""
import hashlib
import json
import sys

KEYS = ("E_inv_weighted", "collar_flow_acc_weighted", "L_interval", "E_remove_weighted", "M_chi_weighted", "energy_identity_residual_weighted",
        "S_vac_archive_weighted", "E_out_weighted")


def g(x, fmt):
    return "-" if x is None else (fmt % x)


def main():
    p4, pa, pout = sys.argv[1:4]
    b4 = open(p4, "rb").read(); ba = open(pa, "rb").read()
    S4 = json.loads(b4); SA = json.loads(ba)
    r4 = {round(r["t_over_rs"], 6): r for r in S4["rows"]}
    ra = {round(r["t_over_rs"], 6): r for r in SA["rows"]}
    common = sorted(t for t in r4 if t in ra and t <= 40.0 + 1e-9)
    rows = []; n_eq = 0; n_eq_ch = 0; n_ch = 0
    worst = {k: 0.0 for k in KEYS}
    arch_eq = True
    for t in common:
        x, y = r4[t], ra[t]
        eq = (x["S_esc_weighted"] == y["S_esc_weighted"])
        n_eq += int(eq)
        per = {}
        for ch, pc in x["per_channel"].items():
            qc = y["per_channel"].get(ch) or {}
            e_ = (pc.get("S_esc") == qc.get("S_esc"))
            n_ch += 1; n_eq_ch += int(e_)
            a_eq = (pc.get("archive") == qc.get("archive") and pc.get("collected") == qc.get("collected"))
            arch_eq = arch_eq and a_eq
            per[ch] = {"S_esc_equal": e_, "S_esc_s4": pc.get("S_esc"), "S_esc_A100": qc.get("S_esc"), "archive": [pc.get("archive"), qc.get("archive")],
                       "collected": [pc.get("collected"), qc.get("collected")], "archive_equal": a_eq}
        d = {}
        for k in KEYS:
            if x.get(k) is not None and y.get(k) is not None:
                d[k] = abs(x[k] - y[k]); worst[k] = max(worst[k], d[k])
            else:
                d[k] = None
        rows.append({"t_over_rs": t, "S_esc_s4": x["S_esc_weighted"], "S_esc_A100": y["S_esc_weighted"], "S_esc_equal": eq, "per_channel": per, "abs_diff": d,
                     "W_hdot_s4": x.get("W_hdot_weighted"), "W_hdot_A100": y.get("W_hdot_weighted")})
    out = {"code": "page_v6i_s4to100_regression_v1.py", "s4": {"path": p4, "md5": hashlib.md5(b4).hexdigest(), "code": S4.get("code"), "final": S4.get("final")},
           "A100": {"path": pa, "md5": hashlib.md5(ba).hexdigest(), "code": SA.get("code"), "final": SA.get("final"),
                    "last_T_over_rs": SA["rows"][-1]["t_over_rs"] if SA["rows"] else None},
           "common_checkpoints": len(common), "S_esc_weighted_equal": n_eq, "S_esc_per_channel_equal": [n_eq_ch, n_ch], "archives_and_batches_equal": arch_eq,
           "largest_abs_diff": worst, "rows": rows}
    with open(pout.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(out, f, indent=1)
    L = ["# A to T/r_s 100 against A s4 over T/r_s 0 to 40 (the regression of the lane's choice (a), 01:08 UTC)\n",
         "Written by code/page/page_v6i_s4to100_regression_v1.py from %s (md5 %s; %s, final %s) and %s (md5 %s; %s, final %s, its last checkpoint T/r_s %s).\n" % (
             p4, out["s4"]["md5"], S4.get("code"), S4.get("final"), pa, out["A100"]["md5"], SA.get("code"), SA.get("final"), g(out["A100"]["last_T_over_rs"], "%.3f")),
         "Both runs: s4's settings, from the ground state of h(T = 0); the same production records to T/r_s 40 (A to 100 reads B's continuation only after T/r_s 40). "
         "**S_esc (weighted) is equal bit for bit at %d of %d common checkpoints** (per channel %d of %d); the archives and batches %s.\n" % (
             n_eq, len(common), n_eq_ch, n_ch, "are equal at every common checkpoint" if arch_eq else "differ at some checkpoint (see the table)"),
         "The largest differences over the common checkpoints (the traces v6e sums by column blocks differ from v6b's by rounding only): %s.\n" % (
             ", ".join("%s %s" % (k, g(v, "%.3e")) for k, v in worst.items())),
         "| T/r_s | S_esc s4 | S_esc A to 100 | equal | per channel equal | archive (s4, A to 100) per channel | |dE_inv| | |d collar flow| | |dL| | |dE_remove| | |dM_chi| |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["abs_diff"]
        L.append("| %.3f | %.12f | %.12f | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["t_over_rs"], r["S_esc_s4"], r["S_esc_A100"], "yes" if r["S_esc_equal"] else "no",
            ", ".join("%s %s" % (ch, "yes" if v["S_esc_equal"] else "no") for ch, v in sorted(r["per_channel"].items())),
            "; ".join("%s: %s" % (ch, v["archive"]) for ch, v in sorted(r["per_channel"].items())),
            g(d["E_inv_weighted"], "%.2e"), g(d["collar_flow_acc_weighted"], "%.2e"), g(d["L_interval"], "%.2e"), g(d["E_remove_weighted"], "%.2e"), g(d["M_chi_weighted"], "%.2e")))
    last = rows[-1] if rows else None
    if last and abs(last["t_over_rs"] - 40.0) < 1e-9:
        L.append("\nTr[h_dot (C - P)] (weighted) at T/r_s 40: s4 %s (the backward difference: record 320 was its last), A to 100 %s (the forward difference to B's record at 40.125)." % (
            g(last["W_hdot_s4"], "%.6e"), g(last["W_hdot_A100"], "%.6e")))
    with open(pout, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s: %d common checkpoints, S_esc equal at %d" % (pout, len(common), n_eq))


if __name__ == "__main__":
    main()
