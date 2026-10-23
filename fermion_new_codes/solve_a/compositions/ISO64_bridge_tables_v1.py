#!/usr/bin/env python
"""ISO64_bridge_tables_v1.py (composition 4, the solve lane): the ladder tables of ISO_64_s4 (and its plain original) beside P64_s4, written by
the v1.3 composer's own table methods (ladders_compose_v1_3.Doc) from ISO64_bridge_ladder_v1.py's readout json; plus a check that the 27 v1.3
members' readouts equal the delivered v1.3 readout json, and a dense matched-clock table (every record, 11.5 to 16 r_s, and every 2 r_s to 40)
of ISO_64_s4 against P64_s4 from the saved npz arrays. Computes nothing new beyond differences.
Usage: python ISO64_bridge_tables_v1.py LADDER_DIR OUT_MD"""
import json
import sys

import numpy as np

sys.path.insert(0, "<internal>/qi2_ladders_code")
import ladders_compose_v1_3 as C  # noqa: E402

D_DIR, OUT = sys.argv[1], sys.argv[2]
D = json.load(open(D_DIR + "/ladders_readout_v1_3.json"))
V13 = json.load(open("<internal>/ladders_readout_v1_3.json"))
A = np.load(D_DIR + "/ladders_readout_v1_3.npz")
d = C.Doc(D, {})
names = ["P64_s4", "ISO_64_s4", "ISO_64_s4_plain"]
L = []
# reproduction check against the delivered v1.3 readout
bad = []
for nm, M in V13["members"].items():
    N = D["members"][nm]
    for k in ("formation", "convergence", "episodes", "Rc_min", "matched", "matched_from_formation", "counts"):
        a = json.dumps(M.get(k), sort_keys=True, default=str); b = json.dumps(N.get(k), sort_keys=True, default=str)
        if a != b:
            bad.append((nm, k))
fits_same = json.dumps(V13["fits"], sort_keys=True, default=str) == json.dumps(D["fits"], sort_keys=True, default=str)
L.append("## reproduction check\n\n- the %d v1.3 members' blocks (formation, convergence, episodes, Rc_min, matched, matched_from_formation, counts) against "
         "the delivered ladders_readout_v1_3.json: %d differ %s; fits identical: %s" % (len(V13["members"]), len(bad), bad[:10], fits_same))
L.append("\n## member table\n\n" + d.member_table(names, {}))
L.append("\n## formation table\n\n" + d.formation_table(names))
L.append("\n## episodes\n\n" + d.episodes_text(names))
L.append("\n## centroid\n\n" + d.compression_table(names))
L.append("\n## from formation\n\n" + d.from_formation_table(names))
L.append("\n## convergence\n\n" + d.convergence_table(names))
L.append("\n## excluded\n\n" + d.excluded_text(names))
L.append("\n## spreads\n\n" + d.spreads_table({("P64_s4", "ISO_64_s4"), ("ISO_64_s4", "ISO_64_s4_plain")}))
for key, title in (("minNv", "### min(N - |v|)"), ("EE0", "### E/E0"), ("R_c_over_rs", "### R_c/r_s"), ("R_t_over_rs", "### R_t/r_s"), ("R_t_over_2gE", "### R_t/(2gE)")):
    L.append("\n" + d.matched_tables(names, key, title))
L.append("\n## newton records (the composition's 13.625 to 14.5)\n\n" + json.dumps(D["members"]["ISO_64_s4"]["meta"].get("newton_records_in_plain_args"), indent=0))
# dense matched clocks from the arrays
def arr(nm, k):
    return A["%s__%s" % (nm, k)]
TP, TI = arr("P64_s4", "T"), arr("ISO_64_s4", "T")
ip = {round(float(t), 6): i for i, t in enumerate(TP)}
rows = []
for j, t in enumerate(TI):
    t6 = round(float(t), 6)
    if t6 not in ip or not ((11.5 <= t6 <= 16.0) or abs(t6 / 2 - round(t6 / 2)) < 1e-9):
        continue
    i = ip[t6]
    def g(nm, k, idx):
        v = arr(nm, k)[idx]
        return v
    zP, zI = arr("P64_s4", "nzeros")[i] > 0, arr("ISO_64_s4", "nzeros")[j] > 0
    rtp = g("P64_s4", "Rt_rs", i) if zP else np.nan; rti = g("ISO_64_s4", "Rt_rs", j) if zI else np.nan
    rows.append(["%.4f" % t6, "%.5f" % g("P64_s4", "EE0", i), "%.5f" % g("ISO_64_s4", "EE0", j), "%+.5f" % (g("ISO_64_s4", "EE0", j) - g("P64_s4", "EE0", i)),
                 "%.4f" % g("P64_s4", "minNv", i), "%.4f" % g("ISO_64_s4", "minNv", j),
                 ("%.4f" % rtp) if zP else "none", ("%.4f" % rti) if zI else "none",
                 ("%+.4f" % ((rti - rtp) / rtp)) if (zP and zI) else "-",
                 ("%.4f" % g("P64_s4", "Rt_2gE", i)) if zP else "-", ("%.4f" % g("ISO_64_s4", "Rt_2gE", j)) if zI else "-",
                 "%.3f" % (g("P64_s4", "Rc", i) / 64.0), "%.3f" % (g("ISO_64_s4", "Rc", j) / 64.0),
                 "%s/%s" % ("c" if arr("P64_s4", "counts")[i] and arr("P64_s4", "certified")[i] else "x", "c" if arr("ISO_64_s4", "counts")[j] and arr("ISO_64_s4", "certified")[j] else "x"),
                 "%.1e" % g("ISO_64_s4", "res", j)])
L.append("\n## dense matched clocks (P64_s4 | ISO_64_s4)\n\n" + C.table(["T/r_s", "E/E0 P64", "E/E0 ISO", "dE/E0 (ISO-P64)", "minNv P64", "minNv ISO", "R_t/r_s P64", "R_t/r_s ISO",
                                                                        "dR_t/R_t", "R_t/2gE P64", "R_t/2gE ISO", "R_c/r_s P64", "R_c/r_s ISO", "certified (P64/ISO)", "ISO kept res"], rows))
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, "bad", len(bad), "fits_same", fits_same)
