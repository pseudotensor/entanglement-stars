#!/usr/bin/env python
"""engine_ext_overlap_check_v1.py (builder analyst; inbox #36 B): the continuation's restart check. The continuation restarts from the
production's complex128 slice at T/rs 28 (run_collapse_v19_from_slice_v1.py), so its records from T/rs 28.125 to 40 recompute stored
production records: per common record the relative difference of E, the differences of R_t (and the number of trapping zeros), of
min(N - |v|), the iterations and eigendecompositions of the two, the largest difference of the applied fields (fields npz), and the
occupation masks per channel bitwise (spectra npz; the continuation's spectra hold its own records only). Read as the records stand
(a running continuation is read to its last record). Every number is printed by the script. Args: --new dir/tag --stored dir/tag --md."""
import argparse
import json
import os

import numpy as np


def load(dt):
    d, tag = os.path.split(dt)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    sp = os.path.join(d, "collapse_%s_spectra.npz" % tag)
    S = np.load(sp, allow_pickle=True) if os.path.exists(sp) else None
    return J, Z, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True); ap.add_argument("--stored", required=True); ap.add_argument("--md", required=True)
    ap.add_argument("--from-T", type=float, default=None, help="T/rs of the restart (records after it are compared)")
    a = ap.parse_args()
    Jn, Zn, Sn = load(a.new); Js, Zs, Ss = load(a.stored)
    rs = float(Jn["args"]["rs"])
    t_from = a.from_T if a.from_T is not None else float((Jn.get("head") or {}).get("resume", {}).get("T_over_rs", 0.0))
    byS = {round(r["T"], 6): r for r in Js["records"] if r.get("E") is not None}
    newr = [r for r in Jn["records"] if r.get("E") is not None and r["T_over_rs"] > t_from + 1e-9]
    TzN = np.asarray(Zn["T"], float); TzS = np.asarray(Zs["T"], float)
    TsN = np.asarray(Sn["T"], float) if Sn is not None else None; TsS = np.asarray(Ss["T"], float) if Ss is not None else None
    kap = [int(k.split("_")[1]) for k in (Sn.files if Sn is not None else []) if k.startswith("mask_")]
    L = ["## The continuation's restart check: its records from T/rs %g against the stored production's (engine_ext_overlap_check_v1.py; inbox #36 B)" % t_from, "",
         "- new: %s (%d records after T/rs %g); stored: %s." % (a.new, len(newr), t_from, a.stored), "",
         "| T/rs | E/E0 new | rel. diff of E | R_t new | R_t diff | zeros new, stored | min(N - abs v) diff | it new, stored | eigh new, stored | applied fields max diff N, v | masks equal (channels) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    worst = {"E": 0.0, "Rt": 0.0, "m": 0.0, "f": 0.0}; nm_eq = nm_tot = 0; n = 0
    for r in newr:
        s = byS.get(round(r["T"], 6))
        if s is None:
            continue
        n += 1
        dE = abs(r["E"] - s["E"]) / abs(s["E"])
        zr, zs = r.get("trapping_zeros") or [], s.get("trapping_zeros") or []
        dR = abs(r["R_t"] - s["R_t"]) if (r.get("R_t") and s.get("R_t")) else None
        dm = abs(r["min_N_minus_absv"] - s["min_N_minus_absv"])
        kn = int(np.argmin(np.abs(TzN - r["T"]))); ks = int(np.argmin(np.abs(TzS - r["T"])))
        dN = float(np.max(np.abs(np.asarray(Zn["N_applied"][kn], float) - np.asarray(Zs["N_applied"][ks], float))))
        dv = float(np.max(np.abs(np.asarray(Zn["v_applied"][kn], float) - np.asarray(Zs["v_applied"][ks], float))))
        meq = "-"
        if TsN is not None and TsS is not None:
            i = np.where(np.abs(TsN - r["T"]) < 1e-9)[0]; j = np.where(np.abs(TsS - r["T"]) < 1e-9)[0]
            if len(i) and len(j):
                eq = [bool(np.array_equal(Sn["mask_%d" % k][i[0]], Ss["mask_%d" % k][j[0]])) for k in kap]
                nm_eq += sum(eq); nm_tot += len(eq); meq = "%d of %d" % (sum(eq), len(eq))
        worst["E"] = max(worst["E"], dE); worst["m"] = max(worst["m"], dm); worst["f"] = max(worst["f"], dN, dv)
        if dR is not None:
            worst["Rt"] = max(worst["Rt"], dR)
        L.append("| %.3f | %.7f | %.2e | %s | %s | %d, %d | %.2e | %s, %s | %s, %s | %.2e, %.2e | %s |" % (
            r["T_over_rs"], r["E_over_E0"], dE, ("%.3f" % r["R_t"]) if r.get("R_t") else "-", ("%.2e" % dR) if dR is not None else "-", len(zr), len(zs), dm,
            r.get("iterations"), s.get("iterations"), r.get("eigendecompositions"), s.get("eigendecompositions"), dN, dv, meq))
    L += ["", "- over %d common records: the largest relative difference of E %.2e, of R_t %.2e (a), of min(N - |v|) %.2e, of the applied fields %.2e; masks equal on %d of %d channel-records." % (
        n, worst["E"], worst["Rt"], worst["m"], worst["f"], nm_eq, nm_tot),
          "- every number above is printed by the script from the two runs' json, fields and spectra npz files."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-2:]))


if __name__ == "__main__":
    main()
