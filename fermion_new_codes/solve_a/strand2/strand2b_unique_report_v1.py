#!/usr/bin/env python
"""Strand 2b sec. 14 item 2 report (builder analyst): tables from strand2b_c8_unique_v1.py (unique_<case>.json) and
strand2b_c8_unique_post_v1.py (post_<case>.json) for the two cube enclosure cases.  Numbers emitted, not retyped."""
import argparse
import json
import os

import numpy as np


def f(v, sig=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    esc = lambda c: str(c).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-unique", required=True)
    ap.add_argument("--dir-post", required=True)
    ap.add_argument("--dir-records", required=True, help="the enclosure case records")
    ap.add_argument("--tags", default="C8_b20_m0.1_gM0.2_gB1_L1_k1,C8_b20_m0.1_gM0.2_gB1_L1_k10000")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    out = []
    rows1, rows2, rows3 = [], [], []
    notes = []
    for tag in a.tags.split(","):
        U = json.load(open(os.path.join(a.dir_unique, "unique_%s.json" % tag)))
        Pp = json.load(open(os.path.join(a.dir_post, "post_%s.json" % tag)))
        rec = json.load(open(os.path.join(a.dir_records, "case_%s.json" % tag)))
        k = U["k"]
        orb = Pp["orbits"]
        reached = any(o["is_continuation_solution"] for o in orb)
        rows1.append([f(k), U["starts"], "[%s, %s]^16" % tuple(U["box"]), U["seed"], U["converged"], f(rec["residual"]), f(Pp["noise_floor_threshold"]),
                      Pp["limits_at_or_below_floor_positive"], len(Pp["rows_not_at_floor"]),
                      " ".join("%d: %s" % (r["trial"], f(r["best"])) for r in Pp["rows_not_at_floor"]) or "-", len(orb),
                      " ".join(str(len(o["trials"])) for o in orb), "yes" if reached else "no"])
        notes.append((k, rec["residual"], Pp["noise_floor_threshold"]))
        sets = []
        if "continuation_checks" in Pp:
            sets.append(("continuation solution (recomputed)", None, Pp["continuation_checks"]))
        elif reached:
            sets.append(("continuation solution (record)", [o for o in orb if o["is_continuation_solution"]][0],
                         {"residual": rec["residual"], "noise_floor": rec["residual_noise_floor"], "sigma_min_J": rec["sigma_min_J"],
                          "cond_J": rec["cond_J"], "E": rec["E"], "gap": rec["gap_block"], "d0_at_tolerance": rec["d0_at_tolerance"],
                          "E_sel": rec["E_sel"], "min_component": min(min(rec["psi"]), min(rec["N"])), "psi": rec["psi"], "N": rec["N"],
                          "enclosure": {"radii": rec["enclosure"]["radii"], "largest_passing_r_in_domain": rec["enclosure"]["largest_passing_r_in_domain"]}}))
        for j, o in enumerate(orb):
            if o["is_continuation_solution"]:
                if "continuation_checks" in Pp:
                    sets[0] = (sets[0][0], o, sets[0][2])
                continue
            sets.append(("orbit %d" % (j + 1), o, o["checks"]))
        for name, o, c in sets:
            enc = c["enclosure"]["radii"]
            rows2.append([f(k), name, f(o["stabilizer_order"]) if o else "48", f(o["orbit_size"]) if o else "1",
                          len(o["trials"]) if o else 0, f(o["distance_to_continuation"]) if o else "0",
                          f(c["residual"]), f(c["noise_floor"]), f(c["sigma_min_J"]), f(c["cond_J"]), f(c["E"], 6), f(c["gap"]),
                          c["d0_at_tolerance"], f(c["E_sel"], 6), f(c["min_component"]),
                          " ".join("%s: %s %s" % (r, f(v["q"]), "pass" if v["passed"] else "fail") for r, v in enc.items()),
                          f(c["enclosure"]["largest_passing_r_in_domain"])])
            rows3.append([f(k), name, " ".join(f(x, 6) for x in c["psi"]), " ".join(f(x, 6) for x in c["N"])])
    out += ["### U1 The starts (16 per case, every component of z = (psi, N) uniform in the box, seed 11, damped Newton with the FD Jacobian, "
            "at most 40 iterations); the run accepted at residual 1e-11; the post-processing counts every positive limit at or below the case's "
            "noise floor (the record's residual_noise_floor) and groups them into orbits of the cube's 48 symmetries", "",
            table(["k", "starts", "box", "seed", "accepted at 1e-11", "continuation solution's residual", "noise floor",
                   "positive limits at or below the floor", "starts not at the floor", "their best residuals (start: residual)",
                   "orbits", "starts per orbit", "continuation solution reached"], rows1), ""]
    out += ["### U2 Each solution class with the case driver's checks (strand2b_row_C8_v6.py's objects; the sampled enclosure: radii 1e-3, 1e-2, "
            "1e-1, the 16 axes of z and 24 random directions, seed 3, q = sup ||1 - Y J(z + r v)||, pass if |Y F| + r q < r)", "",
            table(["k", "solution", "stabilizer order", "orbit size", "starts reaching it", "distance to the continuation solution", "residual",
                   "noise floor", "sigma_min(J)", "cond(J)", "E", "gap", "d0 at tolerance", "E_sel", "smallest component",
                   "enclosure per radius (q, pass)", "largest passing radius in the positive domain"], rows2), ""]
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.join(here, "frozen"))
    import strand2a_lib as S2
    pos = S2.lattice("C8")["positions"]
    out += ["### U3 The representatives' fields per site (sites in order: %s)" % ", ".join("%d (%s)" % (i, ",".join("%g" % c for c in p_)) for i, p_ in enumerate(pos)), "",
            table(["k", "solution", "psi per site", "N per site"], rows3), ""]
    out += ["Noise floors: " + "; ".join("k = %s: the continuation solution's residual %s, the record's noise floor %s" % (f(k), f(r_), f(n_))
                                          for k, r_, n_ in notes) + ".", ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("wrote", a.md)


if __name__ == "__main__":
    main()
