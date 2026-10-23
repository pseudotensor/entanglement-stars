#!/usr/bin/env python
"""heavy_cut_plot_v1.py: the figures of HEAVYCUT_W8_v1 from its composed data (W8; the Page panel's heavy curve, panel data and not the
paper's figure). Reads HEAVYCUT_W8_v1_all.json from --data (the data folder, <out>/heavycut) and writes images only into --plot-out
(a separate folder; no data is written there and nothing is written into the data folder).

Per history 2 figures (png and pdf):
- <TITLE>_<history>_decomposition: one column per cut; top: the heavy excess S_H - S_P (the panel's curve) with its decomposition
  S_Q - S_P (the sea's excess, signed), S_d and -K_sd against T/r_s; middle: the raw S_H and S_P (regulated tables, not converged in kappa);
  bottom: K_sd with its band [0, 2 min(S_Q, S_d)]; the formation clock and the first-return estimate marked where the json carries them.
- <TITLE>_<history>_against_energy (only where page_own_object_v1's jsons are found in --page-dir): the heavy excess against the heavy
  field's Killing energy beyond the cut E_D from page_own_object_v1's weighted rows at the matched clocks (the panel's second axis).
Every figure carries the print status of the data; no dates."""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TITLE = "HEAVYCUT_W8_v1"


def load_rows(data):
    p = os.path.join(data, "%s_all.json" % TITLE)
    if os.path.isfile(p):
        with open(p) as f:
            J = json.load(f)
        return J.get("rows", []), J.get("print_status", ""), J.get("cuts_rs")
    rows = []
    for fn in sorted(os.listdir(data)):
        if fn.startswith("heavycut_") and fn.endswith(".json"):
            with open(os.path.join(data, fn)) as f:
                rows.append(json.load(f))
    rows.sort(key=lambda r: (r["history"], r["T_over_rs"]))
    status = rows[0].get("print_status", "") if rows else ""
    return rows, status, None


def page_energy(page_dir, hist, trs, lab):
    p = os.path.join(page_dir, "page_%s_T%07.2f.json" % (hist, trs))
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            J = json.load(f)
        return ((J.get("weighted") or {}).get(lab) or {}).get("E_D")
    except Exception:  # noqa: BLE001
        return None


def plot_history(rows, hist, labels, plot_out, status, page_dir):
    rs_ = [r for r in rows if r["history"] == hist]
    if not rs_:
        return []
    T = np.array([r["T_over_rs"] for r in rs_])
    label_h = rs_[0].get("history_label", hist)
    written = []
    fig, axes = plt.subplots(3, len(labels), figsize=(5.2 * len(labels), 11.0), squeeze=False)
    for j, lab in enumerate(labels):
        W = [(r.get("weighted") or {}).get(lab) or {} for r in rs_]
        get = lambda key: np.array([w.get(key, np.nan) if w else np.nan for w in W], float)  # noqa: E731
        ax = axes[0, j]
        ax.plot(T, get("heavy_excess"), "k-o", lw=2.0, ms=4, label="S_H - S_P (the heavy excess; the panel's curve)")
        ax.plot(T, get("excess_sea"), "b-s", ms=3, label="S_Q - S_P (the sea's excess, signed)")
        ax.plot(T, get("S_d"), "g-^", ms=3, label="S_d (the dust alone)")
        ax.plot(T, -get("K_sd"), "r-v", ms=3, label="-K_sd (the restriction term)")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title("beyond %s r_s" % lab)
        ax.set_ylabel("nats (summed with 16 kappa; whole multiplets assumed)")
        ax = axes[1, j]
        ax.plot(T, get("S_H"), "k-o", ms=3, label="S_H (raw; regulated, not converged in kappa)")
        ax.plot(T, get("S_P"), "0.5", marker="s", ms=3, ls="--", label="S_P (the held Killing vacuum; raw)")
        if np.isfinite(get("S_mask")).any():
            ax.plot(T, get("S_mask"), "m:", marker="d", ms=3, label="S_mask (the mask's projector; diagnostic)")
        ax.set_ylabel("nats")
        ax = axes[2, j]
        up = get("K_sd_upper")
        ax.fill_between(T, 0.0, up, color="r", alpha=0.12, label="[0, 2 min(S_Q, S_d)]")
        ax.plot(T, get("K_sd"), "r-v", ms=3, label="K_sd")
        ax.set_ylabel("nats")
        ax.set_xlabel("T / r_s (the clock at infinity)")
        for ax_ in axes[:, j]:
            form = rs_[0].get("formation") or {}
            if form.get("T_over_rs") is not None:
                ax_.axvline(form["T_over_rs"], color="0.3", ls=":", lw=0.8)
            frs = [((r.get("first_return") or {}).get(lab) or {}).get("first_return_rs") for r in rs_]
            frs = [x for x in frs if x is not None]
            if frs:
                ax_.axvline(min(frs), color="0.7", ls="-.", lw=0.8)
            ax_.grid(alpha=0.25)
        axes[0, j].legend(fontsize=7, loc="best")
        axes[1, j].legend(fontsize=7, loc="best")
        axes[2, j].legend(fontsize=7, loc="best")
    fig.suptitle("%s: %s\nNOT FOR PRINT: %s" % (TITLE, label_h, status[:160]), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        p = os.path.join(plot_out, "%s_%s_decomposition.%s" % (TITLE, hist, ext))
        fig.savefig(p, dpi=130)
        written.append(p)
    plt.close(fig)
    if page_dir and os.path.isdir(page_dir):
        got = False
        fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.2), squeeze=False)
        for j, lab in enumerate(labels):
            E = np.array([page_energy(page_dir, hist, r["T_over_rs"], lab) or np.nan for r in rs_], float)
            W = [(r.get("weighted") or {}).get(lab) or {} for r in rs_]
            y = np.array([w.get("heavy_excess", np.nan) if w else np.nan for w in W], float)
            ok = np.isfinite(E) & np.isfinite(y)
            if ok.any():
                got = True
                axes[0, j].plot(E[ok], y[ok], "k-o", ms=4)
                for t_, e_, y_ in zip(T[ok], E[ok], y[ok]):
                    axes[0, j].annotate("%.1f" % t_, (e_, y_), fontsize=6, textcoords="offset points", xytext=(3, 3))
            axes[0, j].set_title("beyond %s r_s" % lab)
            axes[0, j].set_xlabel("E_D: the heavy Killing energy beyond the cut (page_own_object_v1, lattice units)")
            axes[0, j].set_ylabel("S_H - S_P (nats)")
            axes[0, j].grid(alpha=0.25)
        fig.suptitle("%s: %s, the heavy excess against the energy beyond the cut (T/r_s at the points)\nNOT FOR PRINT" % (TITLE, label_h), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        if got:
            for ext in ("png", "pdf"):
                p = os.path.join(plot_out, "%s_%s_against_energy.%s" % (TITLE, hist, ext))
                fig.savefig(p, dpi=130)
                written.append(p)
        plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the data folder <out>/heavycut (HEAVYCUT_W8_v1_all.json or the per-slice jsons)")
    ap.add_argument("--plot-out", dest="plot_out", required=True, help="the plot folder (separate from --data)")
    ap.add_argument("--page-dir", dest="page_dir", default=None, help="page_own_object_v1's outputs (<out>/page) for the energy axis; optional")
    ap.add_argument("--histories", default=None, help="comma list (default: every history in the data)")
    a = ap.parse_args(argv)
    if os.path.abspath(a.plot_out) == os.path.abspath(a.data):
        raise SystemExit("the plot folder must differ from the data folder")
    os.makedirs(a.plot_out, exist_ok=True)
    rows, status, cuts = load_rows(a.data)
    if not rows:
        raise SystemExit("no %s rows in %s" % (TITLE, a.data))
    labels = cuts or sorted({lab for r in rows for lab in (r.get("weighted") or {})}, key=float)
    hists = [h for h in a.histories.split(",") if h] if a.histories else sorted({r["history"] for r in rows})
    written = []
    for h in hists:
        written += plot_history(rows, h, labels, a.plot_out, status, a.page_dir)
    for p in written:
        print(p)
    with open(os.path.join(a.plot_out, "%s_plots_index.txt" % TITLE), "w") as f:
        f.write("\n".join(["images drawn from %s; NOT FOR PRINT: %s" % (a.data, status)] + written) + "\n")


if __name__ == "__main__":
    main()
