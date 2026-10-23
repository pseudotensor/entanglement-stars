#!/usr/bin/env python
"""engine_ell16_line_v1.py (builder analyst; engine spec 36.4 (3): "P21 (4): on the ell 16 record at its closest approach, min(N - |v|)
of the dense and of the applied fields and ell/(2GE), one line"). Record-only. Reads the ell 16 run's json (the records' min_N_minus_absv, the
dense fields' minimum by engine_fields' trapping_radius; E; E0) and its fields npz (N_applied, v_applied: the site fields the Hamiltonian
saw; N, v: the record's own kernel-averaged returns of its solve). The closest approach is the record with the smallest dense minimum; at
that record the applied fields' minimum over the sites of N_app - |v_app| (and the record's averaged returns' beside it); the applied
fields' own smallest minimum over the run and its record; ell/(2 g E) at that record's E and at the run's E0. Quoted beside, as the
returns' numbers and not this lane's criterion: the comparison bounds of WEB40F P21/P20 (ell/(2GE) > 0.260875404864 forbids a zero of the
returned field for N = 1, 0 <= M <= E), P34 (0.228195 with both averagings on a positive J = 0 source) and the reviewer's dense-field bound
(a zero of the dense field unless ell > 0.371 (2GE)). Args: --json --fields --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    J = json.load(open(a.json))
    R = [r for r in J["records"] if r.get("E") is not None and "T_over_rs" in r]
    Z = np.load(a.fields, allow_pickle=True)
    Rs = np.asarray(Z["R"], float); Tz = np.asarray(Z["T"], float)
    g = float(J["args"]["g"]); ell = float(J["args"]["ell"]); rs = float(J["args"]["rs"]); E0 = float(J["head"]["E0"])
    k = min(range(len(R)), key=lambda i: R[i]["min_N_minus_absv"])
    r = R[k]
    kz = int(np.argmin(np.abs(Tz - r["T"])))
    if abs(Tz[kz] - r["T"]) > 1e-9:
        raise SystemExit("the fields npz holds no record at T/rs %.3f" % r["T_over_rs"])
    fa = np.asarray(Z["N_applied"][kz], float) - np.abs(np.asarray(Z["v_applied"][kz], float))
    fr = np.asarray(Z["N"][kz], float) - np.abs(np.asarray(Z["v"][kz], float))
    ia, ir = int(np.argmin(fa)), int(np.argmin(fr))
    allA = np.asarray(Z["N_applied"], float) - np.abs(np.asarray(Z["v_applied"], float))
    kmin = np.min(allA, axis=1); kk = int(np.argmin(kmin)); isite = int(np.argmin(allA[kk]))
    nz = sum(1 for q in R if q.get("trapping_zeros"))
    E = float(r["E"])
    line = ("- the ell %g record (%s, %s, prepared at ell %s; %d records to T/rs %.3f, %d holding a zero of the dense N + v): its closest approach (the smallest dense "
            "min(N - |v|)) at T/rs %.3f: the dense fields' min(N - |v|) %.6f at r %.3f; the applied site fields' min(N_app - |v_app|) %.6f at site R %.0f "
            "(the record's averaged returns N, v: %.6f at R %.0f); the applied fields' own smallest minimum over the run %.6f at T/rs %.3f (site R %.0f); "
            "ell/(2 g E) %.6f at that record's E %.6f (2 g E %.4f), %.6f at the run's E0 %.6f (2 g E0 %.4f); quoted, the returns' numbers: P21/P20 0.260875 "
            "(no zero of the returned field for N = 1, 0 <= M <= E), P34 0.228195 (both averagings, positive J = 0 source), the dense-field bound 0.371 "
            "(the reviewer's; a zero of the dense field unless ell exceeds it times 2GE)." % (
                ell, J.get("script"), J["args"].get("tag"), J["head"].get("prep_ell", J["args"].get("prep_ell") or ell), len(R), R[-1]["T_over_rs"], nz,
                r["T_over_rs"], r["min_N_minus_absv"], r["r_min_N_minus_absv"], float(fa[ia]), Rs[ia], float(fr[ir]), Rs[ir],
                float(kmin[kk]), Tz[kk] / rs, Rs[isite], ell / (2.0 * g * E), E, 2.0 * g * E, ell / (2.0 * g * E0), E0, 2.0 * g * E0))
    hdr = "## The ell 16 record at its closest approach (engine_ell16_line_v1.py; engine spec 36.4 (3), P21 (4); record-only)"
    open(a.md, "w").write(hdr + "\n\n" + line + "\n\n- every number above is printed by the script from the run's json and fields npz.\n")
    print(line)


if __name__ == "__main__":
    main()
