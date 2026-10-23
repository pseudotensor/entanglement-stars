#!/usr/bin/env python
"""Item 7, the validation of strand7_sea_page_v5.py before production use (SPEC_strand7_page.md sec. 14.4; builder analyst): F0 (flat) and
S0 (L, static) on v5 against their v4 records: C-pg-3 and C-pg-4's measures (the largest |dS0|, |dSad|, S'_in, |g - 1| and N_ext over the
outputs, sectors and cuts) at their previous level (here: at most max(10 x the v4 value, 1e-9), the builder's operational reading), and
vac_projector_diff at every output and at T = 0 (the channel meta) below 1e-6, the lead's expectation where the pair is resolved (sec.
14.4 (4)); the validation sentence's 1e-9 is reported beside it (the builder's choice, labelled: the repair pass's recomputation on S0 at
j = 1/2 gives the eigh route's projector 4.9e-9 from the polar one in the Frobenius norm over 2560 modes, the old route's rounding).  Writes --out (a text verdict) and creates <dir>/v5_validated or
<dir>/v5_validation_failed.  Args: --v5-f0 --v4-f0 --v5-s0 --v4-s0 --dir --out."""
import argparse
import json
import os


def measures(R):
    rows = [q for v in R["channels"].values() for q in v]
    m = {"dS0": 0.0, "dSad": 0.0, "S_in": 0.0, "g_minus_1": 0.0, "N_ext": 0.0}
    for q in rows:
        for s in q["sectors"].values():
            for c in s["cuts"].values():
                m["dS0"] = max(m["dS0"], abs(c["dS0"]))
                m["dSad"] = max(m["dSad"], abs(c["dSad"]))
                m["S_in"] = max(m["S_in"], abs(c["S_de_int_state"]))
                for k, v in c.items():
                    if k.startswith("g_d") and v is not None:
                        m["g_minus_1"] = max(m["g_minus_1"], abs(v - 1))
            m["N_ext"] = max(m["N_ext"], abs(s["spectrum"]["N_ext"]))
    vpd = [s["controls"].get("vac_projector_diff") for q in rows for s in q["sectors"].values()]
    vpd0 = [v for cm in R.get("channel_meta", {}).values() for v in cm.get("vac_projector_diff_T0", {}).values()]
    return m, vpd, vpd0


def main():
    ap = argparse.ArgumentParser()
    for k in ("--v5-f0", "--v4-f0", "--v5-s0", "--v4-s0", "--dir", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    lines, ok = [], True
    for lab, p5, p4 in (("F0", a.v5_f0, a.v4_f0), ("S0 on L", a.v5_s0, a.v4_s0)):
        m5, vpd, vpd0 = measures(json.load(open(p5)))
        m4, _, _ = measures(json.load(open(p4)))
        for k in m5:
            lim = max(10 * m4[k], 1e-9)
            good = m5[k] <= lim
            ok &= good
            lines.append("%s %s: v5 %.3e, v4 %.3e, limit %.3e: %s" % (lab, k, m5[k], m4[k], lim, "pass" if good else "FAIL"))
        vm = max(v for v in vpd if v is not None) if any(v is not None for v in vpd) else None
        v0 = max(vpd0) if vpd0 else None
        good = vm is not None and vm < 1e-6 and (v0 is None or v0 < 1e-6)
        tight = vm is not None and vm < 1e-9 and (v0 is None or v0 < 1e-9)
        ok &= good
        lines.append("%s vac_projector_diff: largest over the outputs %s, at T = 0 %s (the gate 1e-6): %s; against the validation sentence's 1e-9: %s" % (
            lab, "%.3e" % vm if vm is not None else "-", "%.3e" % v0 if v0 is not None else "-", "pass" if good else "FAIL", "holds" if tight else "does not hold"))
    lines.append("VERDICT: %s" % ("validated" if ok else "failed"))
    open(a.out, "w").write("\n".join(lines) + "\n")
    open(os.path.join(a.dir, "v5_validated" if ok else "v5_validation_failed"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
