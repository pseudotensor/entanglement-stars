#!/usr/bin/env python
"""item7_fig_page_draft_v1.py (builder analyst, 2026-10-16; inbox #36 S1 F, the item 7 figure draft): plots, from the "figure" block of
PAGE_solved_history_v1_summary.json (page_solved_history_report_v1.py; nothing is computed here), S_esc(t_inf) and L(t_inf)/L_H of
Computation A's primary run against the collar's arrival clock t_inf (in nominal r_s), with the formation clock, the formation clock plus
T_d and the formation's arrival clock marked, and writes the png and its SOURCES file. Colours: black, red, blue (no yellow, no viridis).
Usage: python item7_fig_page_draft_v1.py SUMMARY_JSON OUT_PNG OUT_SOURCES"""
import datetime
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402


def main():
    p_json, p_png, p_src = sys.argv[1:4]
    raw = open(p_json, "rb").read()
    J = json.loads(raw)
    F = J.get("figure")
    if not F:
        sys.exit("no figure block in %s" % p_json)
    x = F["t_inf_over_rs"]; S = F["S_esc"]
    mk = F["marks"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
    a1.plot(x, S, "k-o", ms=3, lw=1.2, label="S_esc (weighted, nats)")
    a1.set_ylabel("S_esc (nats)")
    for key, val in (("L_over_LH_kappaT", "L/L_H at kappa_T"), ("L_over_LH_TH", "L/L_H at T_H = 1/(4 pi 2gE)")):
        col = "r" if key.endswith("kappaT") else "b"
        pts = [(xx, abs(v), v < 0) for xx, v in zip(x, F[key]) if v is not None and v != 0]
        if pts:
            a2.semilogy([p[0] for p in pts], [p[1] for p in pts], col + "-", lw=1.0, label=val + " (|.|; open: negative)")
            a2.semilogy([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], col + "o", ms=3)
            neg = [p for p in pts if p[2]]
            if neg:
                a2.semilogy([p[0] for p in neg], [p[1] for p in neg], col + "o", ms=4, mfc="none")
    a2.set_ylabel("|L| / L_H")
    a2.set_xlabel("t_inf / r_s (the collar's arrival clock, nominal r_s)")
    marks = [("formation_t_inf_over_rs", "k", "--", "formation T/r_s %s" % mk.get("formation_T_over_rs")),
             ("formation_plus_T_d_t_inf_over_rs", "b", "-.", "formation + T_d (T_d/r_s %s)" % mk.get("T_d_over_rs")),
             ("arrival_t_inf_over_rs", "r", ":", "the formation's arrival at the collar")]
    for key, col, ls, lab in marks:
        v = mk.get(key)
        if v is not None:
            for ax in (a1, a2):
                ax.axvline(v, color=col, ls=ls, lw=1.0, label=lab if ax is a1 else None)
    a1.legend(fontsize=7, loc="upper left"); a2.legend(fontsize=7, loc="lower right")
    a1.set_title("DRAFT (%s): %s" % ("final" if F.get("final") else "running", "the collapse epoch's escaping entropy and flux, a test channel (D14)"), fontsize=8)
    fig.text(0.01, 0.005, F.get("label", "")[:220], fontsize=5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(p_png, dpi=160)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(p_src, "w") as f:
        f.write("ITEM7_fig_page_draft_v1 sources (written %s by code/page/item7_fig_page_draft_v1.py)\n" % now)
        f.write("data: %s (md5 %s), its figure block, copied from the run summaries by code/page/page_solved_history_report_v1.py\n" % (p_json, hashlib.md5(raw).hexdigest()))
        f.write("run inputs (from the summary json's inputs): %s\n" % json.dumps({k: v for k, v in (J.get("inputs") or {}).items() if v.get("landed")}))
        f.write("x: t_inf_collar / 64 (the summary rows' retarded clock at the collar R_collar, from the run's first record)\n")
        f.write("top: S_esc_weighted (the direct route on the cumulative collected set; band printed in the summary, %s at the end)\n" % (F["band"][-1] if F.get("band") else "-"))
        f.write("bottom: |L_interval| / L_H for both comparators (L_H(kappa_T): the record's dense surface slope; L_H(T_H): 1/(4 pi 2gE)); open markers: L < 0\n")
        for k_ in ("formation_T_over_rs", "formation_t_inf_over_rs", "T_d_over_rs", "T_d_source", "formation_plus_T_d_t_inf_over_rs", "arrival_t_over_rs",
                   "arrival_t_inf_over_rs", "arrival_travel_over_2gE", "two_gE_formation", "arrival_definition", "half_collar_flow_t_over_rs", "half_collar_flow_definition"):
            f.write("mark %s: %s\n" % (k_, mk.get(k_)))
        f.write("label: %s\n" % F.get("label"))
    print("wrote %s and %s" % (p_png, p_src))


if __name__ == "__main__":
    main()
