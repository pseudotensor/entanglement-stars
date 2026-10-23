#!/usr/bin/env python
"""fill_item7_sentence_v1.py (builder analyst, 2026-10-17; engine spec 39.1 (4), FILE 2): fills the lead's template
ITEM7_solved_history_sentence_v1_template.md from PAGE_solved_history_v2_summary.json into ITEM7_solved_history_sentence_v1.md: every bracket
is replaced by a value copied from the json at two significant figures (the composer's arithmetic for the percentages of E, E_inv over the
record's E at the clock; every selection printed in the file's values table); the template's text is copied, never edited; the sec. 4 check
(requirements_new.txt: no "frozen", "prediction", run-sense "window"; the template's own: no "dark", "defect") is asserted on the filled
sentence. Selections, from engine spec 39.1's reading: A's formation pair is the checkpoint at T/r_s 14; the pulses are the checkpoints from
15 to 40; the arrival L ratios are the checkpoints 36 to 40 at the record's dense slope; c15's steady flux is its checkpoints at 77 + 16 n,
c30's late flux its checkpoints at 148 + 20 n, both before their reflection clocks. It prints only after the icloud lane's R2 has read
version 2. Usage: python fill_item7_sentence_v1.py SUMMARY_JSON TEMPLATE_MD OUT_MD"""
import datetime
import hashlib
import json
import re
import sys


def sig2(x):
    """two significant figures, trailing zeros kept (2 -> 2.0, 0.3 -> 0.30, 2e-05 -> 2.0e-05)."""
    s = "%.2g" % x
    mant, _, exp = s.partition("e")
    digits = mant.replace("-", "").replace(".", "").lstrip("0")
    if len(digits) < 2:
        mant = mant + ("0" if "." in mant else ".0")
    return mant + ("e" + exp if exp else "")


def main():
    p_json, p_tpl, p_out = sys.argv[1:4]
    raw = open(p_json, "rb").read()
    J = json.loads(raw)
    tpl = open(p_tpl, encoding="utf-8").read()
    m = re.search(r'"(On the model\'s own solved collapse history.*?)"\n', tpl, re.S)
    if not m:
        sys.exit("the template's sentence was not found")
    sentence = m.group(1)
    rows = J["A_s4"]["rows"]
    byT = {round(r["t_over_rs"], 6): r for r in rows}
    V = {}; SRC = {}

    def put(k, x, src):
        V[k] = x; SRC[k] = src
    lab = J.get("label") or ""
    mm = re.search(r"E_rad\^\(0\)/E = ([0-9.eE+-]+)", lab)
    put("Erad_over_E", float(mm.group(1)), "the run's label (s4's summary label, copied into the json)")
    r14 = byT[14.0]
    put("Einv_14_pct", 100.0 * r14["E_inv_weighted"] / r14["E"], "A's checkpoint T/r_s 14: E_inv %.6e over E %.6f" % (r14["E_inv_weighted"], r14["E"]))
    pul = [r for r in rows if 15.0 - 1e-9 <= r["t_over_rs"] <= 40.0 + 1e-9]
    pct = [(100.0 * r["E_inv_weighted"] / r["E"], r["t_over_rs"]) for r in pul]
    put("Einv_min_pct", min(pct)[0], "A's checkpoints 15 to 40: the smallest E_inv/E, at T/r_s %.3f" % min(pct)[1])
    put("Einv_max_pct", max(pct)[0], "A's checkpoints 15 to 40: the largest E_inv/E, at T/r_s %.3f" % max(pct)[1])
    put("energy_identity_max", max(abs(r["energy_identity_residual_weighted"]) for r in rows), "A's checkpoints: the largest |energy identity residual| (weighted)")
    mk = J["figure"]["marks"]["v2"]
    put("arrival_halfflow", mk["arrival_half_flow_t_inf_over_rs"], "the figure's mark: the half-collar-flow clock on the collar's clock")
    put("band_lo", min(mk["light_travel_band_t_inf_over_rs"]), "the figure's mark: the light-travel band from 2 x 2gE_f on the collar's clock (#48's correction)")
    put("band_hi", max(mk["light_travel_band_t_inf_over_rs"]), "the figure's mark: the light-travel band from 1.5 x 2gE_f on the collar's clock")
    arr = [r["L_over_LH_kappaT"] for r in rows if 36.0 - 1e-9 <= r["t_over_rs"] <= 40.0 + 1e-9 and r.get("L_over_LH_kappaT") is not None]
    put("L_ratio_min", min(arr), "A's checkpoints 36 to 40: the smallest L/L_H(kappa_T)")
    put("L_ratio_max", max(arr), "A's checkpoints 36 to 40: the largest L/L_H(kappa_T)")
    put("S_col_final", rows[-1]["S_esc_weighted"], "A's last checkpoint T/r_s %.3f: the selected-sector entropy (weighted)" % rows[-1]["t_over_rs"])
    put("S_over_etaA", J["A_s4"]["peak"]["ratio_eta"], "the largest S_esc over eta A/a^2 at its clock (the composer's script ratio)")
    for n in ("c15", "c30"):
        C = J["C"][n]; hr = C["hold_readouts"]
        put("%s_Einv0_pct" % n, 100.0 * hr["E_inv0_over_E"], "%s: E_inv at the hold's start over the record's E %.6f" % (n, hr["E_at_hold"]))
        put("%s_frac_left" % n, C["fraction_leaving_script"], "%s: the collar flow over the hold over E_inv at the hold's start" % n)
    T_refl = {n: J["C"][n]["hold_readouts"]["T_refl"] for n in ("c15", "c30")}

    def pick(n, t0, step):
        out = []
        for r in J["C"][n]["rows"]:
            t = r["t_over_rs"]
            if t >= t0 - 1e-9 and t < T_refl[n] and abs((t - t0) / step - round((t - t0) / step)) < 1e-9 and r.get("L_over_LH_kappaT") is not None:
                out.append((t, r["L_over_LH_kappaT"]))
        return out
    c15s = pick("c15", 77.0, 16.0); c30s = pick("c30", 148.0, 20.0)
    put("c15_L_lo", min(v for _, v in c15s), "c15: L/L_H(kappa_T) at T/r_s %s (77 + 16 n before the reflection clock %.1f): the smallest" % ([t for t, _ in c15s], T_refl["c15"]))
    put("c15_L_hi", max(v for _, v in c15s), "c15: the same checkpoints: the largest")
    put("c30_L_late_lo", min(v for _, v in c30s), "c30: L/L_H(kappa_T) at T/r_s %s (148 + 20 n before the reflection clock %.1f): the smallest" % ([t for t, _ in c30s], T_refl["c30"]))
    put("c30_L_late_hi", max(v for _, v in c30s), "c30: the same checkpoints: the largest")
    filled = sentence
    for k in re.findall(r"\[([A-Za-z0-9_]+)\]", sentence):
        if k not in V:
            sys.exit("no value for [%s]" % k)
        filled = filled.replace("[%s]" % k, sig2(V[k]))
    left = re.findall(r"\[[A-Za-z0-9_]+\]", filled)
    bad = [w for w in ("frozen", "freeze", "prediction", "dark", "defect") if w in filled.lower()]
    bad += ["window (run sense)"] if re.search(r"\bwindow\b", filled.lower()) else []
    if left or bad:
        sys.exit("unfilled %s or forbidden words %s" % (left, bad))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# Item 7, the radiation paragraph: the one sentence this lane supplies on Computation A (FILE 2, filled; engine spec 39.1 (4))\n",
         "Filled %s by code/page/fill_item7_sentence_v1.py from %s (md5 %s) and the lead's template %s (md5 %s): every bracket replaced by the json's value at two significant figures; the template's text copied, never edited. **It prints only after the icloud lane's R2 has read version 2.**\n" % (
             now, p_json, hashlib.md5(raw).hexdigest(), p_tpl, hashlib.md5(open(p_tpl, "rb").read()).hexdigest()),
         "\"%s\"\n" % filled,
         "The sec. 4 check (no \"frozen\", \"prediction\", run-sense \"window\", \"dark\", \"defect\" in the sentence): passed.\n",
         "## The values (two significant figures, then the json's value and its selection)\n",
         "| bracket | printed | the json's value | selection |", "|---|---|---|---|"]
    for k in re.findall(r"\[([A-Za-z0-9_]+)\]", sentence):
        L.append("| %s | %s | %.10g | %s |" % (k, sig2(V[k]), V[k], SRC[k]))
    with open(p_out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s: %d brackets filled" % (p_out, len(V)))


if __name__ == "__main__":
    main()
