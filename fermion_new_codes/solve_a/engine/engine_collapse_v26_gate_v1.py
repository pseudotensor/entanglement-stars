#!/usr/bin/env python
"""engine_collapse_v26_gate_v1.py (builder analyst; the brief BRIEF_v26_builder_S4_S9: "v26 on a stored v25 record with the new items switched
off reproduces v25 (the gate)"). Compares runs record by record: per record (matched by T) the fields npz arrays (N, v, eps_tot, J_tot,
eps_pairs, J_pairs, N_applied, v_applied), the spectra npz (the eigenvalues and the packed occupation masks per channel) and the json's
scalars (E, E_over_E0, iterations, eigendecompositions, R_t, min_N_minus_absv, R_c, n_pocket, the M1 terms); bitwise where every
difference is exactly zero. --ref is the run every other --run is compared with (dir/tag; repeatable --run), --stored a stored run of the
same member (another machine: differences printed, not bitwise); a stored npz unreadable at the time (a run still writing it, or its sync) is
reported and the readable parts compared. Every number is printed by the script. CPU, seconds.
Args: --ref dir/tag --run label=dir/tag (repeatable) --stored label=dir/tag --md --json."""
import argparse
import json
import os

import numpy as np

ARR = ("N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied")
SCAL = ("E", "E_over_E0", "iterations", "eigendecompositions", "R_t", "min_N_minus_absv", "R_c", "R_sh", "E_dust_shells", "E_pairs_shells", "M1_I_E", "H_in", "R2_minus_R0_F")


def load(spec):
    d, tag = os.path.split(spec)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    out = [J]
    for kind in ("fields", "spectra"):
        try:
            out.append(np.load(os.path.join(d, "collapse_%s_%s.npz" % (tag, kind)), allow_pickle=True))
        except Exception as e:  # a stored run still being written or synced
            out.append(None)
            print("unreadable %s npz of %s: %s" % (kind, spec, e))
    return tuple(out)


def compare(A, B, n_ch):
    JA, ZA, SA = A; JB, ZB, SB = B
    TA = np.asarray(ZA["T"], float)
    TB = np.asarray(ZB["T"], float) if ZB is not None else np.array([float(r["T"]) for r in JB["records"] if r.get("E") is not None])
    rows = []
    recA = {round(float(r["T"]), 9): r for r in JA["records"] if r.get("E") is not None}
    recB = {round(float(r["T"]), 9): r for r in JB["records"] if r.get("E") is not None}
    for i, T in enumerate(TA):
        j = np.where(np.abs(TB - T) < 1e-9)[0]
        if not len(j):
            continue
        j = int(j[0])
        row = {"T": float(T), "arrays": {}, "scalars": {}, "spectra": {}, "fields_npz_readable": ZB is not None, "spectra_npz_readable": SB is not None}
        for k in ARR:
            if ZB is not None and k in ZA and k in ZB:
                row["arrays"][k] = float(np.max(np.abs(np.asarray(ZA[k][i], float) - np.asarray(ZB[k][j], float))))
        ra, rb = recA.get(round(float(T), 9), {}), recB.get(round(float(T), 9), {})
        for k in SCAL:
            x, y = ra.get(k), rb.get(k)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                row["scalars"][k] = abs(float(x) - float(y))
            elif x != y:
                row["scalars"][k] = "differs (%r, %r)" % (x, y)
        ia = np.where(np.abs(np.asarray(SA["T"], float) - T) < 1e-9)[0]
        ib = np.where(np.abs(np.asarray(SB["T"], float) - T) < 1e-9)[0] if SB is not None else np.array([], dtype=np.int64)
        if len(ia) and len(ib):
            de = 0.0; dm = 0
            for c in range(1, n_ch + 1):
                ea = np.asarray(SA["e_%d" % c][int(ia[0])], float); eb = np.asarray(SB["e_%d" % c][int(ib[0])], float)
                de = max(de, float(np.max(np.abs(ea - eb))))
                dm += int(not np.array_equal(np.asarray(SA["mask_%d" % c][int(ia[0])]), np.asarray(SB["mask_%d" % c][int(ib[0])])))
            row["spectra"] = {"max_eigenvalue_difference": de, "channels_with_different_masks": dm}
        vals = list(row["arrays"].values()) + [v for v in row["scalars"].values() if isinstance(v, float)] + ([row["spectra"]["max_eigenvalue_difference"], float(row["spectra"]["channels_with_different_masks"])] if row["spectra"] else [])
        row["bitwise"] = bool(all(v == 0.0 for v in vals) and not any(isinstance(v, str) for v in row["scalars"].values()))
        row["largest"] = max(vals) if vals else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--run", action="append", default=[]); ap.add_argument("--stored", action="append", default=[])
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    R = load(a.ref)
    n_ch = int(R[0]["args"]["kappa_sea"])
    out = {"script": "engine_collapse_v26_gate_v1.py", "ref": a.ref, "comparisons": []}
    L = ["## The v26 gate: version 26 against version 25 on one member and one machine (engine_collapse_v26_gate_v1.py)", "",
         "- the reference run: %s (%s; args: m %s, r_s %s, %s sites, tau %s, ell %s, %s channels, site fields %s, reading %s, R0/r_s %s)" % (
             a.ref, R[0]["script"], R[0]["args"]["m"], R[0]["args"]["rs"], R[0]["args"]["nsites"], R[0]["args"]["tau"], R[0]["args"]["ell"], n_ch,
             R[0]["args"]["site_fields"], R[0]["args"]["reading"], R[0]["args"]["R0_over_rs"]), ""]
    for spec in a.run + a.stored:
        lab, path = spec.split("=", 1)
        B = load(path)
        rows = compare(R, B, n_ch)
        kind = "stored (another machine)" if spec in a.stored else "this machine"
        out["comparisons"].append({"label": lab, "path": path, "kind": kind, "script": B[0]["script"],
                                   "v26_args": {k: B[0]["args"].get(k) for k in ("ops", "fields", "empty_rel", "stop_rule", "step_diag", "v26_items_off", "stall_fd_dim")}, "rows": rows})
        L += ["### %s: %s (%s; %s)%s" % (lab, path, B[0]["script"], kind, "" if (B[1] is not None and B[2] is not None) else " (its %s unreadable at the time: being written or synced; the readable parts compared)" % " and ".join(x for x, y in (("fields npz", B[1]), ("spectra npz", B[2])) if y is None)), "",
              "- its v26 switches: %s" % json.dumps({k: B[0]["args"].get(k) for k in ("ops", "fields", "empty_rel", "stop_rule", "step_diag", "v26_items_off", "stall_fd_dim")}), "",
              "| T | bitwise | largest difference | N | v | eps_tot | J_tot | N_applied | v_applied | E | iterations | eigenvalues | channels with different masks |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            A_ = r["arrays"]; S_ = r["scalars"]; P_ = r["spectra"] or {}
            f = lambda x: ("%.3e" % x) if isinstance(x, float) else (str(x) if x is not None else "-")
            L.append("| %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["T"], "yes" if r["bitwise"] else "no", f(r["largest"]), f(A_.get("N")), f(A_.get("v")), f(A_.get("eps_tot")), f(A_.get("J_tot")), f(A_.get("N_applied")),
                f(A_.get("v_applied")), f(S_.get("E")), f(S_.get("iterations")), f(P_.get("max_eigenvalue_difference")), f(P_.get("channels_with_different_masks"))))
        L.append("")
    L += ["- every number above is printed by the script; a row is bitwise when every listed difference is exactly zero."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
