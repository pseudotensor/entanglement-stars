#!/usr/bin/env python
"""engine_capture_read_v5.py (builder analyst): version 4 with the barrier-top fit's window corrected, nothing else changed: the predicted
edge (eta_c ln 9)/(pi L_c) is about 0.7 of one channel at omega/m 2 (L_c about 53), so version 4's window 0.02 <= A <= 0.98 held one channel
on a synthetic barrier-top profile and could not fit; version 5 fits every channel with 0.5 L_c <= L <= 1.5 L_c (L_c the predicted p b_c),
L_c and eta free (eta > 0), and prints, beside the fit, the width in channels (eta ln 9/pi, L units) with "the edge narrower than one
channel" where it is below 1 (eta then constrained only loosely by the channel grid). Version 4's docstring follows.
engine_capture_read_v4.py (builder analyst; engine spec 38.5 (3): "A probe that settles takes 37.6 (3)'s barrier-top fit and 35.2 (3)'s
settled form; otherwise the unsettled form with the new run's last reading", read "by a new reading script version (code/engine/
engine_capture_read_v3.py shows the pattern)"): version 3's reading (the massive critical disk and C_m, the inward-classified wording, the
uncertified tails, the lattice-kinematic label, the settle state), for C-capture version 9, 10 and 11 JSONs, with three changes:
(1) a run with an outer absorber (version 10's port, version 11's appended absorber; the JSON's outer_port record, printed) prints, per
probe, the weight the absorber has taken by the last reading (summed as sigma, over pi b_c^2(omega, m)); version 3's chain-end arrival
column stays a reading aid for a run without one (with an absorber the weight beyond R_max is weight on its way into it);
(2) a settled probe prints the settled form's numbers (the reading time, sigma/(pi b_c^2) there: the impact-averaged non-return fraction
over the massive critical disk, with the in-transit share), and 37.6 (3)'s barrier-top fit at that reading: per channel A(L) at L = kappa
(the mean of the two signs), fitted by least squares to [1 + exp(2 pi (L - L_c)/eta)]^-1 over the channels with 0.02 <= A <= 0.98 (both
L_c and eta free), beside the massive critical orbit's prediction (x_c = 1 + 2/(1 + sqrt(9 - 8 m^2/omega^2)), b_c^2/r_s^2 = x_c^3/(2 -
x_c), L_c = p b_c with p = sqrt(omega^2 - m^2), eta_c = sqrt((3 - x_c)/x_c)) and the edge widths (eta ln 9)/(pi L_c) fitted and predicted;
no Airy width, no Pauli factor; fewer than 4 channels in the window prints "no fit";
(3) an unsettled probe prints the unsettled form's numbers at the run's last reading (sigma/(pi b_c^2), its change over 4 r_s, the
in-transit share). Every number printed by the script from the JSONs. Args: --json (repeatable) --md [--arrival 0.01]."""
import argparse
import json
import math

import numpy as np


def tails(r, cap):
    ex = {}
    for k, why in r.get("excluded_channels") or []:
        ex.setdefault(abs(k), set()).add(1 if k > 0 else -1)
    s = ", ".join(("+-%d" % k) if len(v) == 2 else ("%+d" % (k * next(iter(v)))) for k, v in sorted(ex.items()))
    n = sum(len(v) for v in ex.values())
    kmax = max((c["kappa"] for c in r["channels"]), default=0)
    return ("%s (%d channels)" % (s, n)) if n else "none", ("at the cap %d" % cap) if kmax >= cap else "below the cap %d (largest kappa %d)" % (cap, kmax)


def barrier_fit(r, key, rs):
    """37.6 (3): A(L) near L_c at the reading key, fitted to the barrier-top form; the prediction from the massive critical orbit."""
    from scipy.optimize import curve_fit
    om, m = r["omega"], r["m"]
    p = math.sqrt(om * om - m * m)
    xc = 1.0 + 2.0 / (1.0 + math.sqrt(9.0 - 8.0 * m * m / (om * om)))
    bc = rs * math.sqrt(xc ** 3 / (2.0 - xc)); Lc_pred = p * bc; eta_c = math.sqrt((3.0 - xc) / xc)
    LL, AA = [], []
    for c in r["channels"]:
        vals = [c[s_]["A"][key] for s_ in ("plus", "minus") if key in (c[s_].get("A") or {})]
        if vals:
            LL.append(float(c["kappa"])); AA.append(float(np.mean(vals)))
    LL, AA = np.array(LL), np.array(AA)
    win = (LL >= 0.5 * Lc_pred) & (LL <= 1.5 * Lc_pred)
    out = {"x_c": xc, "b_c_over_rs": bc / rs, "L_c_pred": Lc_pred, "eta_c": eta_c, "width_pred": eta_c * math.log(9.0) / (math.pi * Lc_pred), "n_window": int(win.sum())}
    if win.sum() < 4:
        out["fit"] = None
        return out
    f = lambda L, Lc, eta: 1.0 / (1.0 + np.exp(2.0 * math.pi * (L - Lc) / eta))  # noqa: E731
    try:
        (Lc_f, eta_f), cov = curve_fit(f, LL[win], AA[win], p0=(Lc_pred, eta_c), bounds=([0.0, 1e-6], [np.inf, np.inf]), maxfev=20000)
        out["fit"] = {"L_c": float(Lc_f), "eta": float(eta_f), "width": float(eta_f * math.log(9.0) / (math.pi * Lc_f)),
                      "width_channels": float(eta_f * math.log(9.0) / math.pi),
                      "rms": float(np.sqrt(np.mean((f(LL[win], Lc_f, eta_f) - AA[win]) ** 2)))}
    except Exception as err:  # noqa: BLE001
        out["fit"] = {"error": str(err)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True); ap.add_argument("--md", required=True)
    ap.add_argument("--arrival", type=float, default=0.01)
    a = ap.parse_args()
    L = ["## C-capture read against the massive critical disk, versions 9 to 11 (engine_capture_read_v5.py; engine spec 37.6 (1) to (4), 38.5 (3))", "",
         "- the comparator is pi b_c^2(omega, m), the massive critical disk (b_c from each run's own label), C_m = pi b_c^2(omega, m)/(27 pi (g E)^2) times the massless disk; sigma/(pi b_c^2) is a summed ratio, (pi/p^2) sum_kappa kappa [f(+kappa) + f(-kappa)] over that constant, not a packet probability.",
         "- \"inward-classified weight\": the positive-band group-velocity sign applied to both spin components; not a spinor-band projection or an integrated flux (37.6 (1)); the runs' columns headed \"inward-branch\" carry this weight.",
         "- uncertified tails: the launch channels the engine excluded (taken as A = 0), and the channels above the cap where a probe's channel list reaches it; lattice-kinematic: the lattice group speed below the continuum sqrt(1 - 1/e^2) by more than 5 percent (the lead's threshold, 37.6 (2)).",
         "- an outer absorber (version 10's port on the last 16 sites, version 11's on the declared exterior appended beyond the chain): the weight it has taken by a reading is outward weight that left and is part of R(T); its column is that weight summed as sigma, over pi b_c^2(omega, m). Without one (version 9), the chain-end arrival (the builder's reading aid, threshold %g of pi b_c^2(omega, m)) marks the first reading at which the plain norm beyond R_max, summed as sigma, exceeds it." % a.arrival,
         "- settled (the run's own rule, version 9's in-transit gate): the settled form's numbers and 37.6 (3)'s barrier-top fit at the settled reading (A(L) at L = kappa, the two signs' mean, over the channels with 0.5 L_c <= L <= 1.5 L_c (version 5), fitted to [1 + exp(2 pi (L - L_c)/eta)]^-1; the prediction x_c = 1 + 2/(1 + sqrt(9 - 8 m^2/omega^2)), b_c^2/r_s^2 = x_c^3/(2 - x_c), L_c = p b_c, eta_c = sqrt((3 - x_c)/x_c), the edge width (eta ln 9)/(pi L_c)); unsettled: the unsettled form's numbers at the run's last reading.", ""]
    for jp in a.json:
        d = json.load(open(jp))
        sr = d["settle_rule"]; cap = d["args"]["kappa_cap"]; rs = d["r_s"]
        op = d.get("outer_port") or {"on": False, "text": "none (version 9)"}
        state = ("complete, %s to %s UTC" % (d["start_utc"], d["end_utc"])) if d.get("end_utc") else ("in progress (started %s UTC): %d probes done" % (d["start_utc"], len(d["results"])))
        L += ["### %s" % jp, "",
              "- %s; slice %s (T %.1f), E %.6f, r_s %.4f, R_t %.4f; readings every %g r_s to %g r_s; the run %s; the outer absorber: %s." % (
                  d["script"], d["slice"], d["T"], d["E"], rs, d["R_t"], sr["T_step_over_rs"], sr["T_max_over_rs"], state, op.get("text")), "",
              "| background | probe | e = omega/m | C_m | group speed, lattice | sqrt(1 - 1/e^2) | departure (percent) | label | channels | excluded at launch (uncertified, A = 0) | the cap | settled (the run's rule) | sigma/(pi b_c^2) there or at the last reading (T/r_s) | its change over 4 r_s (percent) | in-transit share there | outer absorber's weight by then, over pi b_c^2 | chain-end arrival (no absorber: T/r_s; the plain norm beyond R_max there) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        fits = []
        for r in d["results"]:
            mm = r["massive"]; e = mm["e"]; vl = mm["group_speed"]; vc = math.sqrt(1.0 - 1.0 / e ** 2)
            dep = 100.0 * (1.0 - vl / vc); Cm = mm["pi_bc2_massive"] / r["27_pi_gE2"]; nrm = r["norm_ratio"]
            ex, cp = tails(r, cap)
            st = r["settle"]; steps = r["steps"]
            last = steps[-1]
            kkey = "%.4f" % last["T"]
            ab = r.get("outer_absorbed_summed") or {}
            abs_txt = ("%.4e" % (ab[kkey] / nrm)) if kkey in ab else "-"
            arr = None
            if not op.get("on"):
                for i, s_ in enumerate(steps):
                    b = r["beyond_Rmax_summed"]["%.4f" % s_["T"]] / nrm
                    if b > a.arrival:
                        arr = (s_["T_over_rs"], b); break
            L.append("| %s | %s | %.4f | %.4f | %.5f | %.5f | %.2f | %s | %d | %s | %s | %s | %.4f (%.1f) | %s | %.4f | %s | %s |" % (
                r["background"], r["probe"], e, Cm, vl, vc, dep, "lattice-kinematic" if dep > 5.0 else "-", len(r["channels"]), ex, cp,
                ("settled at %.1f r_s" % st["T_over_rs"]) if st.get("settled") else "unsettled", last["ratio"], last["T_over_rs"],
                ("%+.1f" % (100.0 * last["devs"][-1])) if last.get("devs") else "-", last["transit_share"], abs_txt,
                ("%.1f; %.4f" % arr) if arr else ("none" if not op.get("on") else "(with an absorber: not applicable)")))
            if st.get("settled"):
                fits.append((r, barrier_fit(r, kkey, rs)))
        if fits:
            L += ["", "#### The settled probes: 37.6 (3)'s barrier-top fit at the settled reading", "",
                  "| background | probe | x_c | b_c/r_s | L_c predicted (p b_c) | L_c fitted | eta_c | eta fitted | edge width predicted (eta_c ln 9)/(pi L_c) | fitted | fitted width in channels | rms | channels in the window |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r, fz in fits:
                ff = fz.get("fit")
                if not ff or "error" in ff:
                    L.append("| %s | %s | %.4f | %.4f | %.2f | no fit (%s) | %.4f | - | %.4f | - | - | - | %d |" % (r["background"], r["probe"], fz["x_c"], fz["b_c_over_rs"], fz["L_c_pred"],
                                                                                      (ff or {}).get("error", "fewer than 4 channels"), fz["eta_c"], fz["width_pred"], fz["n_window"]))
                else:
                    L.append("| %s | %s | %.4f | %.4f | %.2f | %.2f | %.4f | %.4f | %.4f | %.4f | %.3f%s | %.2e | %d |" % (r["background"], r["probe"], fz["x_c"], fz["b_c_over_rs"], fz["L_c_pred"], ff["L_c"],
                                                                                         fz["eta_c"], ff["eta"], fz["width_pred"], ff["width"], ff["width_channels"],
                                                                                         " (the edge narrower than one channel)" if ff["width_channels"] < 1.0 else "", ff["rms"], fz["n_window"]))
        L.append("")
    L.append("- every number above is printed by the script from %s; no propagation." % ", ".join(a.json))
    open(a.md, "x").write("\n".join(L) + "\n")
    print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
