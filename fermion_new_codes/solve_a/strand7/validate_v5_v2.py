#!/usr/bin/env python
"""Item 7, version 5's validation re-emitted under SPEC_strand7_page.md sec. 15.2 (builder analyst; no run: the existing F0 and S0 records on
version 5 against their version 4 records).  V1, the item's measures: |dS0|, |dSad|, |g - 1|, N_ext at every output and cut below 1e-9.  V2,
the (d, e)-frame diagnostic: S'_in of the evolved state (S_de_int_state) below 1e-8 at every output.  V3, vac_projector_diff: no gate,
reported at every output (expected below 1e-6 where the edge pair is resolved).  V4, C-pg-5 (the frame identity, U's orthogonality) and the
orthonormality drift, printed.  V5, eigh_cpu_fallbacks, printed.  Sec. 15.7 (i), printed per record, sector and output: S_de_int_vacad
(computed in version 5), U_orthogonality, orthonormality beside S_de_int_state.  Writes --out."""
import argparse
import json


def rows_of(R):
    return [q for v in R["channels"].values() for q in v]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--v5-f0", "--v4-f0", "--v5-s0", "--v4-s0", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    L, ok = [], True
    for lab, p5, p4 in (("F0", a.v5_f0, a.v4_f0), ("S0 on L", a.v5_s0, a.v4_s0)):
        R5, R4 = json.load(open(p5)), json.load(open(p4))
        for tag, R in (("v5", R5), ("v4", R4)):
            m = {"dS0": 0.0, "dSad": 0.0, "g_minus_1": 0.0, "N_ext": 0.0}
            for q in rows_of(R):
                for s in q["sectors"].values():
                    for c in s["cuts"].values():
                        m["dS0"] = max(m["dS0"], abs(c["dS0"]))
                        m["dSad"] = max(m["dSad"], abs(c["dSad"]))
                        for k, v in c.items():
                            if k.startswith("g_d") and v is not None:
                                m["g_minus_1"] = max(m["g_minus_1"], abs(v - 1))
                    m["N_ext"] = max(m["N_ext"], abs(s["spectrum"]["N_ext"]))
            if tag == "v5":
                good = all(v < 1e-9 for v in m.values())
                ok &= good
                L.append("%s V1 (v5): %s against 1e-9: %s" % (lab, ", ".join("%s %.3e" % (k, v) for k, v in m.items()), "pass" if good else "FAIL"))
            else:
                L.append("%s V1 (v4, beside): %s" % (lab, ", ".join("%s %.3e" % (k, v) for k, v in m.items())))
        sde = max(c["S_de_int_state"] for q in rows_of(R5) for s in q["sectors"].values() for c in s["cuts"].values())
        good = sde < 1e-8
        ok &= good
        L.append("%s V2: the largest S'_in of the evolved state %.3e against 1e-8: %s" % (lab, sde, "pass" if good else "FAIL"))
        vpd = [s["controls"].get("vac_projector_diff") for q in rows_of(R5) for s in q["sectors"].values()]
        vpd0 = [v for cm in R5.get("channel_meta", {}).values() for v in cm.get("vac_projector_diff_T0", {}).values()]
        L.append("%s V3 (reported, no gate): vac_projector_diff at the outputs from %.3e to %.3e; at T = 0 %s" % (
            lab, min(vpd), max(vpd), ", ".join("%.3e" % v for v in vpd0)))
        fr = max(s["controls"]["frame_identity_rel"] for q in rows_of(R5) for s in q["sectors"].values())
        uo = max(s["controls"]["U_orthogonality"] for q in rows_of(R5) for s in q["sectors"].values())
        on = max(s["controls"]["orthonormality"] for q in rows_of(R5) for s in q["sectors"].values())
        L.append("%s V4: the frame identity %.3e, U^T U - 1 %.3e, the orthonormality drift %.3e; V5: eigh_cpu_fallbacks %s" % (
            lab, fr, uo, on, R5["meta"].get("eigh_cpu_fallbacks")))
        L.append("%s 15.7 (i), per output and sector (T/r_s0, sector: S_de_int_state at the radiation cut, S_de_int_vacad there, U^T U - 1, orthonormality):" % lab)
        for q in rows_of(R5):
            for nm, s in q["sectors"].items():
                c = s["cuts"]["rad"]
                L.append("  %g %s: %.3e, %.3e, %.3e, %.3e" % (q["T_over_rs0"], nm, c["S_de_int_state"], c["S_de_int_vacad"], s["controls"]["U_orthogonality"],
                                                          s["controls"]["orthonormality"]))
    L.append("VERDICT (sec. 15.2, V1 and V2 gated): %s" % ("passed" if ok else "failed"))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
