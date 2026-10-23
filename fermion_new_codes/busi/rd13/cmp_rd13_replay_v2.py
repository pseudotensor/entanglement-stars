#!/usr/bin/env python
"""cmp_rd13_replay_v2.py (busi, analyst; RD-1/RD-3 self-test part (c)): v1 with the arm names as options (the 8-record extension arms Bx,
Dx), and a second verdict by REVIEWA_perf_tolerance_ruling_v1 sec. 4 (A: E/E0 3.0e-8, roots 3.4e-7 a, min(N - |v|) 4.6e-9, fields 1e-7,
decisions identical, at least 8 records, no monotone growth (the last quarter's median E/E0 difference at most 10x the first quarter's);
B: C = E/E0 3.0e-7, roots 3.4e-6 a, min(N - |v|) 4.6e-8, fields 2.5e-5, the change arm at most twice the default arm's distance or below
the A gate). The v1 verdicts (my pre-written budget) are printed unchanged beside it.
v1: the replay arms of rd13_replay_v1.sh against each other and against the
stored production records, by the budget of busi/notes/PERF_RD1_build_v1.md sec. 2 (written before the runs).
  (A) B (--eigh banded --jvp sternheimer) against D (the defaults), per replayed record: E/E0 |d| <= 1e-9; every root (R_t, r_in, R_c, each
      trapping zero, the threshold radii) |d| <= 3.4e-7 a, R_sh equal; min(N - |v|) |d| <= 4.6e-9; the site fields (N, v, N_applied, v_applied
      rows) max |d| <= 1e-7; every set decision identical (the continuation labels per channel, n_pocket, n_refill_R0, inversions_total,
      mask_equals_negative_set, the trapping-zero and tangency counts, late_surface, converged, unconverged, gauge and regime labels, the v36c
      record set, every solve's name, status and meets_stop_rule, the spectra npz masks); at the end slice the masks of every channel.
      Reported only: iterations, eigendecompositions, products, solve counts, walls, residual-class floats.
  (B) each arm against the production (json and npz rows at the same T): the cross-device floor (E/E0 3.0e-7, roots 3.4e-6 a, min(N - |v|)
      4.6e-8, fields 1e-5) and the set decisions identical; B's distance from production within 10x of D's or below 1e-3 of the floor.
  (P) the parent arm P against D where both have the record: bitwise (the records without the walls, the fields, spectra and handoff rows).
    python cmp_rd13_replay_v1.py --replay DIR --prod RUN_DIR/TAG [--arms D,B] [--parent P] [--t0 5.0] --out FILE_STEM
"""
import argparse
import json
import os

import numpy as np

WALL_KEYS = ("seconds", "timing", "slice_saved", "rd13", "v36m", "v36b", "jvp_v35", "solver", "v36")


def load_arm(d, tag):
    p = os.path.join(d, "collapse_%s" % tag)
    J = json.load(open(p + ".json"))
    return {"json": J, "fields": np.load(p + "_fields.npz", allow_pickle=True), "spectra": np.load(p + "_spectra.npz", allow_pickle=True),
            "handoff": np.load(p + "_handoff.npz", allow_pickle=True), "dir": d, "tag": tag}


def recs_after(A, t0):
    return {round(r["T_over_rs"], 6): r for r in A["json"]["records"] if r.get("T_over_rs") is not None and r["T_over_rs"] > t0 + 1e-9}


def row(Z, T):
    Ts = np.asarray(Z["T"], float)
    i = int(np.argmin(np.abs(Ts - T)))
    return i if abs(Ts[i] - T) < 1e-9 else None


def roots(r):
    out = {}
    for k in ("R_t", "r_in", "R_c"):
        out[k] = r.get(k)
    for i, z in enumerate(r.get("trapping_zeros") or []):
        if isinstance(z, dict):
            for kk, vv in z.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    out["tz%d.%s" % (i, kk)] = vv
        else:
            out["tz%d" % i] = z
    for k, v in (r.get("threshold_radius") or {}).items():
        out["thr%s" % k] = v
    return out


CONT_LABELS = ("mode", "nsub", "m", "m_eff", "verified", "n_intermediate", "inversions", "n_pocket", "pockets_occupied_positive", "holes_empty_negative",
               "rotated_blocks", "partly_occupied_blocks", "mask_equals_negative_set")


def decisions(r):
    d = {}
    for kk, c in sorted((r.get("continuation") or {}).items()):
        for f in CONT_LABELS:
            d["cont%s.%s" % (kk, f)] = c.get(f)
    for k in ("n_pocket", "n_refill_R0", "inversions_total", "mask_equals_negative_set", "converged", "unconverged", "R_sh"):
        d[k] = json.dumps(r.get(k), sort_keys=True)
    d["n_trapping_zeros"] = len(r.get("trapping_zeros") or []); d["n_tangencies"] = len(r.get("tangencies") or [])
    ls = r.get("late_surface"); d["late_surface"] = json.dumps(ls.get("flag") if isinstance(ls, dict) else ls)
    d["gauge"] = (r.get("gauge") or {}).get("label") if isinstance(r.get("gauge"), dict) else r.get("gauge")
    d["regime"] = (r.get("regime") or {}).get("label") if isinstance(r.get("regime"), dict) else r.get("regime")
    v = r.get("v36c") or {}
    d["v36c"] = json.dumps({k: v.get(k) for k in ("record_set", "previous_record_set", "equals_previous_record")}, sort_keys=True)
    for i, s in enumerate((r.get("solve_v32") or {}).get("solves", [])):
        d["solve%d" % i] = json.dumps([s.get("name"), s.get("status"), s.get("meets_stop_rule"), (s.get("v36c") or {}).get("label")])
    return d


def counters(r):
    s = r.get("solve_v32") or {}
    return {"iterations": r.get("iterations"), "eigendecompositions": r.get("eigendecompositions"), "products": (r.get("jvp_v35") or {}).get("products"),
            "solves": len(s.get("solves", [])), "evaluations": [q.get("evaluations") for q in s.get("solves", [])], "seconds": r.get("seconds"),
            "eigh_s": (r.get("timing") or {}).get("eigh_s"), "eigh_n": (r.get("timing") or {}).get("eigh_n"), "jvp_s": (r.get("jvp_v35") or {}).get("seconds"),
            "solve_s": (r.get("timing") or {}).get("solve_s"), "solve_n": (r.get("timing") or {}).get("solve_n"),
            "rd13_eigh_s": (r.get("rd13") or {}).get("seconds"), "rd13_st_s": (r.get("rd13") or {}).get("st_seconds")}


def fdiff(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(float(a) - float(b))


def compare(X, Y, T, lim, label):
    """X, Y: (record, arm) pairs; returns (lines, ok, numbers)."""
    rx, ax = X; ry, ay = Y
    num = {}
    num["E_over_E0"] = fdiff(rx.get("E_over_E0"), ry.get("E_over_E0"))
    rx_, ry_ = roots(rx), roots(ry)
    ks = sorted(set(rx_) | set(ry_))
    num["roots"] = max([fdiff(rx_.get(k), ry_.get(k)) for k in ks] or [0.0])
    num["roots_detail"] = {k: fdiff(rx_.get(k), ry_.get(k)) for k in ks}
    num["min_N_minus_absv"] = fdiff(rx.get("min_N_minus_absv"), ry.get("min_N_minus_absv"))
    num["r_min_N_minus_absv (reported)"] = fdiff(rx.get("r_min_N_minus_absv"), ry.get("r_min_N_minus_absv"))
    fmax = 0.0
    ix, iy = row(ax["fields"], T * ax["json"]["args"]["rs"]), row(ay["fields"], T * ay["json"]["args"]["rs"])
    if ix is None or iy is None:
        fmax = float("inf")
    else:
        for k in ("N", "v", "N_applied", "v_applied"):
            fmax = max(fmax, float(np.max(np.abs(np.asarray(ax["fields"][k][ix], float) - np.asarray(ay["fields"][k][iy], float)))))
    num["fields"] = fmax
    dx, dy = decisions(rx), decisions(ry)
    dd = sorted(k for k in set(dx) | set(dy) if dx.get(k) != dy.get(k))
    sx, sy = row(ax["spectra"], T * ax["json"]["args"]["rs"]), row(ay["spectra"], T * ay["json"]["args"]["rs"])
    mk = [k for k in ax["spectra"].files if k.startswith("mask_")]
    mdiff = [k for k in mk if sx is None or sy is None or not np.array_equal(ax["spectra"][k][sx], ay["spectra"][k][sy])]
    num["decisions_differing"] = dd + ["spectra." + k for k in mdiff]
    ok = {"E_over_E0": num["E_over_E0"] <= lim["E"], "roots": num["roots"] <= lim["roots"], "min_N_minus_absv": num["min_N_minus_absv"] <= lim["mNv"],
          "fields": num["fields"] <= lim["fields"], "decisions": not num["decisions_differing"]}
    line = "%s T/rs %.4f: E/E0 %.2e [%s] roots %.2e [%s] min(N-|v|) %.2e [%s] fields %.2e [%s] decisions %s [%s]; r_min(N-|v|) %.1e (reported)" % (
        label, T, num["E_over_E0"], "ok" if ok["E_over_E0"] else "OVER", num["roots"], "ok" if ok["roots"] else "OVER", num["min_N_minus_absv"], "ok" if ok["min_N_minus_absv"] else "OVER",
        num["fields"], "ok" if ok["fields"] else "OVER", "identical" if ok["decisions"] else num["decisions_differing"][:12], "ok" if ok["decisions"] else "DIFFER",
        num["r_min_N_minus_absv (reported)"])
    return line, ok, num


def strip(r):
    """the record without its walls, recursively (keys seconds, timing, slice_saved and any key ending in _s or _seconds)."""
    if isinstance(r, dict):
        return {k: strip(v) for k, v in r.items() if not (k in ("seconds", "timing", "slice_saved") or k.endswith("_s") or k.endswith("_seconds"))}
    if isinstance(r, list):
        return [strip(v) for v in r]
    return r


def row_bytes(x, i):
    v = x[i]
    if x.dtype == object:
        return b"".join(np.asarray(q, dtype=float).tobytes() for q in np.atleast_1d(v))
    return np.asarray(v).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True); ap.add_argument("--prod", required=True)
    ap.add_argument("--arms", default="D,B"); ap.add_argument("--parent", default="P"); ap.add_argument("--t0", type=float, default=5.0)
    ap.add_argument("--arm-d", default="D"); ap.add_argument("--arm-b", default="B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = []; res = {"A": [], "B": [], "P": [], "counters": {}, "slice": {}}

    def say(s):
        lines.append(s); print(s, flush=True)
    pd_, pt_ = os.path.split(a.prod)
    PROD = {"json": json.load(open(os.path.join(pd_, "collapse_%s.json" % pt_))), "fields": np.load(os.path.join(pd_, "collapse_%s_fields.npz" % pt_), allow_pickle=True),
            "spectra": np.load(os.path.join(pd_, "collapse_%s_spectra.npz" % pt_), allow_pickle=True)}
    arms = {}
    for nm in a.arms.split(",") + ([a.parent] if a.parent else []):
        d = os.path.join(a.replay, nm)
        if os.path.exists(os.path.join(d, "collapse_rd13_%s.json" % nm)):
            arms[nm] = load_arm(d, "rd13_" + nm)
    if a.arm_d in arms and a.arm_d != "D":
        arms["D"] = arms.pop(a.arm_d)
    if a.arm_b in arms and a.arm_b != "B":
        arms["B"] = arms.pop(a.arm_b)
    say("cmp_rd13_replay_v2: arms %s (D = %s, B = %s); production %s" % (sorted(arms), a.arm_d, a.arm_b, a.prod))
    RA = {nm: recs_after(A, a.t0) for nm, A in arms.items()}
    RP = recs_after(PROD, a.t0)
    for nm, R in RA.items():
        for T, r in sorted(R.items()):
            res["counters"].setdefault(nm, {})[str(T)] = counters(r)
            c = counters(r)
            say("  %s T/rs %.4f: iterations %s, eigh %s (%.0f s), products %s (%.0f s), solves %s (%.0f s), record %.0f s%s" % (
                nm, T, c["iterations"], c["eigh_n"], c["eigh_s"] or 0, c["products"], c["jvp_s"] or 0, c["solve_n"], c["solve_s"] or 0, c["seconds"] or 0,
                ("; rd13 eigh %.0f s, Sternheimer %.0f s" % (c["rd13_eigh_s"] or 0, c["rd13_st_s"] or 0)) if c["rd13_eigh_s"] is not None else ""))
    okA = True; okB = True; okP = True
    limA = {"E": 1e-9, "roots": 3.4e-7, "mNv": 4.6e-9, "fields": 1e-7}
    limB = {"E": 3.0e-7, "roots": 3.4e-6, "mNv": 4.6e-8, "fields": 1e-5}
    if "D" in arms and "B" in arms:
        Ts = sorted(set(RA["D"]) & set(RA["B"]))
        if sorted(RA["D"]) != sorted(RA["B"]):
            okA = False; say("(A) the arms' record sets differ: D %s B %s" % (sorted(RA["D"]), sorted(RA["B"])))
        for T in Ts:
            line, ok, num = compare((RA["B"][T], arms["B"]), (RA["D"][T], arms["D"]), T, limA, "(A) B vs D")
            okA &= all(ok.values()); say(line); res["A"].append({"T": T, "ok": ok, "num": num})
    else:
        okA = False; say("(A) arm D or B missing")
    dist = {}
    for nm in ("D", "B"):
        if nm not in RA:
            continue
        for T in sorted(RA[nm]):
            if T not in RP:
                say("(B) %s T/rs %.4f: no production record" % (nm, T)); continue
            line, ok, num = compare((RA[nm][T], arms[nm]), (RP[T], PROD), T, limB, "(B) %s vs production" % nm)
            okB &= all(ok.values()); say(line); res["B"].append({"arm": nm, "T": T, "ok": ok, "num": num})
            dist[(nm, T)] = num
    for T in sorted({t for (n_, t) in dist if n_ == "B"}):
        if ("D", T) in dist:
            for k, fl in (("E_over_E0", 3.0e-7), ("roots", 3.4e-6), ("min_N_minus_absv", 4.6e-8), ("fields", 1e-5)):
                b_, d_ = dist[("B", T)][k], dist[("D", T)][k]
                fine = b_ <= max(10.0 * d_, 1e-3 * fl)
                okB &= fine
                if not fine:
                    say("(B) T/rs %.4f %s: B's distance from production %.2e is over 10x D's %.2e and over 1e-3 of the floor" % (T, k, b_, d_))
    if a.parent and a.parent in RA and "D" in RA:
        for T in sorted(set(RA[a.parent]) & set(RA["D"])):
            sp_ = json.dumps(strip(RA[a.parent][T]), sort_keys=True, default=float); sd_ = json.dumps(strip(RA["D"][T]), sort_keys=True, default=float)
            same = sp_ == sd_
            for key in ("fields", "spectra", "handoff"):
                ZP, ZD = arms[a.parent][key], arms["D"][key]
                rs_ = float(arms["D"]["json"]["args"]["rs"]); ip, id_ = row(ZP, T * rs_), row(ZD, T * rs_)
                for k in ZP.files:
                    xp, xd = np.asarray(ZP[k]), np.asarray(ZD[k])
                    if xp.ndim >= 1 and xp.shape[0] == len(np.asarray(ZP["T"])) and k != "T":
                        if ip is None or id_ is None or row_bytes(xp, ip) != row_bytes(xd, id_):
                            same = False; say("(P) T/rs %.4f %s.%s differs" % (T, key, k))
            jp, jd = arms[a.parent]["json"]["args"], arms["D"]["json"]["args"]
            dk = sorted(k for k in set(jp) | set(jd) if jp.get(k) != jd.get(k) and k not in ("out", "tag", "slice_dir", "slices_c128", "segment_end_T_over_rs"))
            say("(P) parent P vs D at T/rs %.4f: record (walls removed) and the fields, spectra, handoff rows byte for byte: %s; args differ by %s" % (T, same, dk))
            okP &= same and dk == ["eigh", "jvp"]
            res["P"].append({"T": T, "bitwise": same, "args_diff": dk})
    # the end slices: the masks
    sl = {}
    for nm in ("D", "B"):
        if nm in arms:
            sd = os.path.join(arms[nm]["dir"], "states")
            fs = sorted(f for f in os.listdir(sd) if f.endswith(".npz")) if os.path.isdir(sd) else []
            if fs:
                sl[nm] = np.load(os.path.join(sd, fs[-1]))
    if "D" in sl and "B" in sl:
        ks = sorted(k for k in sl["D"].files if k.startswith("mask_"))
        meq = {k: bool(np.array_equal(sl["D"][k], sl["B"][k])) for k in ks}
        ede = {k: float(np.max(np.abs(sl["D"]["e_" + k[5:]] - sl["B"]["e_" + k[5:]]))) for k in ks}
        okA &= all(meq.values())
        say("(A) the end slice: masks equal %s of %d channels; eigenvalues max |d| %.1e" % (sum(meq.values()), len(meq), max(ede.values())))
        res["slice"] = {"masks_equal": meq, "e_max_diff": ede}
    else:
        say("(A) the end slice: missing (%s)" % sorted(sl)); okA = False
    # REVIEWA_perf_tolerance_ruling_v1 sec. 4
    gA = {"E": 3.0e-8, "roots": 3.4e-7, "mNv": 4.6e-9, "fields": 1e-7}
    okRA = True; nrec = len(res["A"]); eds = []
    for q in res["A"]:
        n_ = q["num"]
        ok_ = n_["E_over_E0"] <= gA["E"] and n_["roots"] <= gA["roots"] and n_["min_N_minus_absv"] <= gA["mNv"] and n_["fields"] <= gA["fields"] and not n_["decisions_differing"]
        okRA &= ok_; eds.append(n_["E_over_E0"])
        say("(ruling A) T/rs %.4f: E/E0 %.2e roots %.2e min(N-|v|) %.2e fields %.2e decisions %s: %s" % (q["T"], n_["E_over_E0"], n_["roots"], n_["min_N_minus_absv"], n_["fields"],
            "identical" if not n_["decisions_differing"] else n_["decisions_differing"][:8], "inside" if ok_ else "OUTSIDE"))
    if nrec >= 4:
        k4 = max(1, nrec // 4)
        m1, m4 = float(np.median(eds[:k4])), float(np.median(eds[-k4:]))
        grow = m4 > 10.0 * max(m1, 1e-300)
        say("(ruling A) growth: first-quarter median E/E0 diff %.2e, last-quarter %.2e: %s" % (m1, m4, "GROWTH" if grow else "no growth"))
        okRA &= not grow
    enough = nrec >= 8
    say("(ruling A) replayed records %d (the gate asks at least 8): %s" % (nrec, "met" if enough else "NOT MET"))
    gC = {"E_over_E0": 3.0e-7, "roots": 3.4e-6, "min_N_minus_absv": 4.6e-8, "fields": 2.5e-5}
    okRB = True; nB = 0
    for T in sorted({t for (n_, t) in dist if n_ == "B"}):
        for k, fl in gC.items():
            b_ = dist[("B", T)][k]
            if b_ > fl:
                okRB = False; say("(ruling B) T/rs %.4f %s: B against production %.2e over C %.2e" % (T, k, b_, fl))
            if ("D", T) in dist:
                d_ = dist[("D", T)][k]
                if d_ > fl:
                    okRB = False; say("(ruling B) T/rs %.4f %s: D against production %.2e over C %.2e" % (T, k, d_, fl))
                gk = {"E_over_E0": gA["E"], "roots": gA["roots"], "min_N_minus_absv": gA["mNv"], "fields": gA["fields"]}[k]
                if not (b_ <= 2.0 * d_ or b_ <= gk):
                    okRB = False; say("(ruling B) T/rs %.4f %s: B's distance %.2e over twice D's %.2e and over the A gate" % (T, k, b_, d_))
        if dist[("B", T)]["decisions_differing"]:
            okRB = False
        nB += 1
    say("RULING sec. 4 (A) thresholds and decisions on every replayed record: %s" % ("PASS" if okRA else "FAIL"))
    say("RULING sec. 4 (A) as written, with the 8-record requirement: %s" % ("PASS" if (okRA and enough) else "FAIL"))
    say("RULING sec. 4 (B) against production (%d records): %s" % (nB, "PASS" if (okRB and nB) else "FAIL"))
    res["ruling"] = {"A_thresholds": okRA, "A_as_written": okRA and enough, "B": bool(okRB and nB), "records": nrec}
    say("VERDICT (A) B vs D within the budget on every record: %s" % ("PASS" if okA else "FAIL"))
    say("VERDICT (B) each arm within the cross-device floor of production, B's distance of D's size: %s" % ("PASS" if okB else "FAIL"))
    if a.parent:
        say("VERDICT (P) the defaults bitwise with the parent on the shared records: %s" % ("PASS" if (okP and res["P"]) else "FAIL"))
    say("VERDICT (c) [(A) and (B)]: %s" % ("PASS" if (okA and okB) else "FAIL"))
    res["verdict"] = {"A": okA, "B": okB, "P": bool(okP and res["P"]), "c": okA and okB}
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
