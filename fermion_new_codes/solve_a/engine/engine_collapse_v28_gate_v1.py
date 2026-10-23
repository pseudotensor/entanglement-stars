#!/usr/bin/env python
"""engine_collapse_v28_gate_v1.py (builder analyst; engine spec 30.2 item 7: "engine_collapse_v28 (= v26; a new file; the gate bitwise v25
with the items off and bitwise v26 on the smoke's T = 0 record up to the stop rule)"). Part 1: a version-25 run (--v25) against a version-28
run with --v26-items-off of the same member on the same machine and threads, record by record: the fields npz arrays (N, v, eps_tot,
J_tot, eps_pairs, J_pairs, N_applied, v_applied), the spectra npz (eigenvalues, packed masks per channel) and the json's scalars (E,
E_over_E0, iterations, eigendecompositions, R_t, min_N_minus_absv, R_c, R_sh, E_dust_shells, E_pairs_shells, M1_I_E, H_in,
R2_minus_R0_F); bitwise when every difference is exactly zero. Part 2: a version-26 run (--v26) against a version-28 run (--v28) of the
same member, machine and threads, at --t2 (the smoke's T = 0 record by default): the same arrays and spectra and the scalars that do not
count the stop rule's iterations (iterations and eigendecompositions are printed apart, as the stop rule's own difference), and the
unmixed residual history up to version 28's accepted iteration against version 26's (equal when the two rules saw the same iterates);
the records after --t2 are printed as the stop rule's effect, not gated. Every number is printed by the script. CPU, seconds.
Args: --v25 dir/tag --v28off dir/tag --v26 dir/tag --v28 dir/tag [--t2 0] --md --json."""
import argparse
import json
import os

import numpy as np

ARR = ("N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied")
SCAL = ("E", "E_over_E0", "R_t", "min_N_minus_absv", "R_c", "R_sh", "E_dust_shells", "E_pairs_shells", "M1_I_E", "H_in", "R2_minus_R0_F")
STOP = ("iterations", "eigendecompositions")


def load(spec):
    d, tag = os.path.split(spec)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    return J, Z, S


def compare_at(A, B, T, n_ch, scal):
    JA, ZA, SA = A; JB, ZB, SB = B
    ia = np.where(np.abs(np.asarray(ZA["T"], float) - T) < 1e-9)[0]; ib = np.where(np.abs(np.asarray(ZB["T"], float) - T) < 1e-9)[0]
    if not len(ia) or not len(ib):
        return None
    i, j = int(ia[0]), int(ib[0])
    row = {"T": float(T), "arrays": {}, "scalars": {}, "stop": {}, "spectra": {}}
    for k in ARR:
        if k in ZA and k in ZB:
            row["arrays"][k] = float(np.max(np.abs(np.asarray(ZA[k][i], float) - np.asarray(ZB[k][j], float))))
    ra = [r for r in JA["records"] if r.get("E") is not None and abs(float(r["T"]) - T) < 1e-9]
    rb = [r for r in JB["records"] if r.get("E") is not None and abs(float(r["T"]) - T) < 1e-9]
    ra, rb = (ra[0] if ra else {}), (rb[0] if rb else {})
    for k in scal:
        x, y = ra.get(k), rb.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            row["scalars"][k] = abs(float(x) - float(y))
        elif x != y:
            row["scalars"][k] = "differs (%r, %r)" % (x, y)
    for k in STOP:
        row["stop"][k] = [ra.get(k), rb.get(k)]
    sa = np.where(np.abs(np.asarray(SA["T"], float) - T) < 1e-9)[0]; sb = np.where(np.abs(np.asarray(SB["T"], float) - T) < 1e-9)[0]
    if len(sa) and len(sb):
        de = 0.0; dm = 0
        for c in range(1, n_ch + 1):
            de = max(de, float(np.max(np.abs(np.asarray(SA["e_%d" % c][int(sa[0])], float) - np.asarray(SB["e_%d" % c][int(sb[0])], float)))))
            dm += int(not np.array_equal(np.asarray(SA["mask_%d" % c][int(sa[0])]), np.asarray(SB["mask_%d" % c][int(sb[0])])))
        row["spectra"] = {"max_eigenvalue_difference": de, "channels_with_different_masks": dm}
    vals = list(row["arrays"].values()) + [v for v in row["scalars"].values() if isinstance(v, float)]
    if row["spectra"]:
        vals += [row["spectra"]["max_eigenvalue_difference"], float(row["spectra"]["channels_with_different_masks"])]
    row["bitwise"] = bool(all(v == 0.0 for v in vals) and not any(isinstance(v, str) for v in row["scalars"].values()))
    row["largest"] = max(vals) if vals else None
    row["records"] = (ra, rb)
    return row


def fmt(x):
    return ("%.3e" % x) if isinstance(x, float) else (str(x) if x is not None else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v25", required=True); ap.add_argument("--v28off", required=True); ap.add_argument("--v26", required=True); ap.add_argument("--v28", required=True)
    ap.add_argument("--t2", type=float, default=0.0); ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    out = {"script": "engine_collapse_v28_gate_v1.py", "part1": [], "part2": {}}
    L = ["## The v28 gate (engine_collapse_v28_gate_v1.py; engine spec 30.2 item 7)", ""]
    A = load(a.v25); B = load(a.v28off)
    n_ch = int(A[0]["args"]["kappa_sea"])
    L += ["### Part 1: version 28 with the items off against version 25", "",
          "- version 25: %s (%s); version 28: %s (%s; its switches %s); member: m %s, r_s %s, %s sites, tau %s, ell %s, %s channels, site fields %s." % (
              a.v25, A[0]["script"], a.v28off, B[0]["script"], json.dumps({k: B[0]["args"].get(k) for k in ("ops", "fields", "stop_rule", "step_diag", "stall_fd_dim", "anderson_switch", "force_nsub", "v26_items_off")}),
              A[0]["args"]["m"], A[0]["args"]["rs"], A[0]["args"]["nsites"], A[0]["args"]["tau"], A[0]["args"]["ell"], n_ch, A[0]["args"]["site_fields"]), "",
          "| T | bitwise | largest difference | N | v | eps_tot | J_tot | N_applied | v_applied | E | iterations (v25, v28) | eigenvalues | channels with different masks |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    allbit = True
    for T in np.asarray(A[1]["T"], float):
        r = compare_at(A, B, float(T), n_ch, SCAL)
        if r is None:
            continue
        it_same = r["stop"]["iterations"][0] == r["stop"]["iterations"][1] and r["stop"]["eigendecompositions"][0] == r["stop"]["eigendecompositions"][1]
        r["bitwise"] = bool(r["bitwise"] and it_same)
        allbit = allbit and r["bitwise"]
        A_ = r["arrays"]; P_ = r["spectra"] or {}
        L.append("| %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s | %s | %s |" % (
            r["T"], "yes" if r["bitwise"] else "no", fmt(r["largest"]), fmt(A_.get("N")), fmt(A_.get("v")), fmt(A_.get("eps_tot")), fmt(A_.get("J_tot")), fmt(A_.get("N_applied")),
            fmt(A_.get("v_applied")), fmt(r["scalars"].get("E")), r["stop"]["iterations"][0], r["stop"]["iterations"][1], fmt(P_.get("max_eigenvalue_difference")), fmt(P_.get("channels_with_different_masks"))))
        r.pop("records"); out["part1"].append(r)
    L += ["", "- part 1 bitwise at every record (the arrays, the spectra, the scalars and the iteration counts): %s." % ("yes" if allbit else "NO"), ""]
    C = load(a.v26); D = load(a.v28)
    n2 = int(C[0]["args"]["kappa_sea"])
    r = compare_at(C, D, a.t2, n2, SCAL)
    L += ["### Part 2: version 28 against version 26 on the smoke's record at T = %g, up to the stop rule" % a.t2, "",
          "- version 26: %s (%s; stop rule %s); version 28: %s (%s; stop rule %s, mix %s, anderson switch %s)." % (
              a.v26, C[0]["script"], C[0]["args"].get("stop_rule"), a.v28, D[0]["script"], D[0]["args"].get("stop_rule"), D[0]["args"].get("mix"), D[0]["args"].get("anderson_switch")), ""]
    if r is None:
        L.append("- no common record at T = %g." % a.t2)
    else:
        ra, rb = r.pop("records")
        sa = ((ra.get("v26") or {}).get("stop") or {}); sb = ((rb.get("v26") or {}).get("stop") or {})
        rn_a = sa.get("scaled_residual_norms") or []; rn_b = sb.get("scaled_residual_norms") or []
        acc = (sb.get("accepted") or {}).get("iteration")
        best_a = (ra.get("unconverged") or {}).get("iteration")
        k = acc if acc is not None else len(rn_b) - 1
        same_hist = bool(len(rn_a) > k and rn_a[:k + 1] == rn_b[:k + 1])
        A_ = r["arrays"]; P_ = r["spectra"] or {}
        L += ["| quantity | difference |", "|---|---|"]
        for key in ARR:
            L.append("| %s | %s |" % (key, fmt(A_.get(key))))
        for key in SCAL:
            L.append("| %s | %s |" % (key, fmt(r["scalars"].get(key))))
        L += ["| eigenvalues (all channels) | %s |" % fmt(P_.get("max_eigenvalue_difference")), "| channels with different masks | %s |" % fmt(P_.get("channels_with_different_masks")), "",
              "- the state at T = %g bitwise (the arrays, the spectra, the scalars above): %s; the stop rule's own entries: version 26 %s iterations and %s eigendecompositions, version 28 %s and %s; version 28's record line: \"%s\"; version 26's returned iterate %s, version 28's accepted iteration %s." % (
                  a.t2, "yes" if r["bitwise"] else "NO", r["stop"]["iterations"][0], r["stop"]["eigendecompositions"][0], r["stop"]["iterations"][1], r["stop"]["eigendecompositions"][1],
                  (sb.get("accepted") or {}).get("label") or (rb.get("record_label") or "")[:120], best_a, acc),
              "- the scaled unmixed residual norms: version 26 %s; version 28 %s; equal up to version 28's last iteration: %s." % (
                  ", ".join("%.6e" % x for x in rn_a), ", ".join("%.6e" % x for x in rn_b), "yes" if same_hist else "NO"), ""]
        out["part2"] = {"T": a.t2, "row": r, "rn_v26": rn_a, "rn_v28": rn_b, "accepted_iteration_v28": acc, "returned_iteration_v26": best_a, "history_equal": same_hist}
        later = []
        for T in np.asarray(D[1]["T"], float):
            if T <= a.t2 + 1e-9:
                continue
            rr = [q for q in D[0]["records"] if q.get("E") is not None and abs(float(q["T"]) - T) < 1e-9]
            r6 = [q for q in C[0]["records"] if q.get("E") is not None and abs(float(q["T"]) - T) < 1e-9]
            if rr:
                q = rr[0]; s_ = ((q.get("v26") or {}).get("stop") or {})
                later.append("| %.4f | %s | %s | %s | %s | %s |" % (T / float(D[0]["args"]["rs"]), q["iterations"], (r6[0]["iterations"] if r6 else "-"),
                             ("iteration %d" % s_["switched_to_anderson"]["iteration"]) if s_.get("switched_to_anderson") else "no",
                             ("converged at iteration %d" % s_["accepted"]["iteration"]) if s_.get("accepted") else ("unconverged (residual %.2e)" % (q.get("unconverged") or {}).get("residual", float("nan"))),
                             "%.8f / %s" % (q["E_over_E0"], ("%.8f" % r6[0]["E_over_E0"]) if r6 else "-")))
        if later:
            L += ["### The records after T = %g: the stop rule's effect (not gated)" % a.t2, "", "| T/rs | iterations v28 | iterations v26 | switched to Anderson | v28's stop | E/E0 v28 / v26 |", "|---|---|---|---|---|---|"] + later + [""]
    L += ["- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
