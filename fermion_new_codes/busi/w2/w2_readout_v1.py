#!/usr/bin/env python3
"""W2 readout v1 (busi; Reorientation 6 item 6): the data script.

Composes W2's history 0 -> 40 from the runs that supply it and saves every per-record quantity to one json
(the table script w2_readout_v1_table.py prints the readout from that json; nothing here is printed but a
short summary).

  certified records before 12.5 : the v36f lineage (inherited unchanged in every W2 json; checked equal)
  12.625 .. 14.5                : J1_W2_anc (dg_assign=anchor on fixv4c: the maximum-overlap convention's
                                   trajectory; not printable, RUN_W2_v4 0.2)
  14.625 .. end                 : J1_W2_v36v_anc (v36v, dg_assign=prim, from the anc 14.5 slice)

Per record: the supplying run, the printable status and its cause, E/E0, the record's charge move
dE/E0 and its bins (start, chord, record-solve reassignment) with the closure residual, and every charge
move attributed by the Reorientation-5 criterion:
  resolved reassignment  : a primitive crossing decided and resolved with a jump that falls under
                           refinement (diabatic: its fitted jump; adiabatic or exact: zero, its
                           would-be jump listed)
  unresolved reassignment: a primitive crossing undecided / unresolved / jump not falling (its
                           would-be jump listed), every maximum-overlap switch of the anchor convention
                           (no halving test exists for it), a record-solve reassignment above the floor,
                           and a v36f record move with no bins
  no reassignment        : the remainder (chord less the attributed jumps) and the start bin, the
                           three-node rule's defect; printed against S2's 2e-5 threshold
Also: the rule dependence anchor vs prim at the shared clocks 12.625 .. 13.625 (R2' _r2b and v36v), the
anchor epoch against the v36l comparator, and from the v36l maximum-overlap comparator the formation
clock and the first trapped episode's close, labelled as that convention's.

Usage (box, read-only on its inputs):
  python -I w2_readout_v1.py --prim P.json --anc A.json --r2b R.json --v36v V.json --v36l L.json \
      [--prim-log P.log] [--t-end 40] --out w2_readout_v1.json
"""
import argparse, hashlib, json, math, os, re, sys, time

FLOOR = 1e-9        # E/E0 equality floor of RUN_W2_v4 0.2
FLOOR_REC = 5e-9    # the record solve's floor (reviewer A's reading of the v36l readout: none above 5e-9 after 23.875);
                    # a record-solve move below it is the solve's floor, not a reassignment
S2_REM = 2e-5       # S2's remainder threshold (of E)
RS = 64.0           # r_s / a of the member (rs64)
EP0, EP1 = 12.625, 14.5      # the anchor epoch (this composition)
PRIM0 = 14.625
SHARED = 13.625     # the last shared clock of anchor and prim (rule dependence headline)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    J = json.load(open(p))
    return J, {round(r["T_over_rs"], 6): r for r in J["records"]}


def pockets(r):
    d = r.get("n_pocket") or {}
    return int(sum(v for v in d.values() if v is not None))


def trap(r):
    z = r.get("trapping_zeros") or []
    return {"n_zeros": len(z), "R_t_over_rs": (r.get("R_t") / RS) if z and r.get("R_t") is not None else None,
            "min_N_minus_absv": r.get("min_N_minus_absv"), "zeros_over_rs": [x / RS for x in z]}


def intervals(recs):
    iv, cur = [], None
    for r in recs:
        z = bool(r.get("trapping_zeros"))
        T = r["T_over_rs"]
        if z and cur is None:
            cur = [T, T]
        elif z:
            cur[1] = T
        elif cur is not None:
            cur.append(T); iv.append(cur); cur = None     # [first, last, first clock without zeros]
    if cur is not None:
        cur.append(None); iv.append(cur)
    return iv


def crossing_rows(d, E0):
    """Every primitive crossing of the accepted attempt, with its class under the criterion."""
    rows = []
    for s in d.get("substeps") or []:
        p = s.get("prim") or {}
        v = p.get("verify") or p.get("decision") or {}
        src = "verify" if p.get("verify") else ("decision" if p.get("decision") else None)
        for c in v.get("crossings") or []:
            dec, decided, resolved, falls = c.get("decision"), c.get("decided"), c.get("resolved"), c.get("falls")
            if decided and resolved and (falls is True or (dec != "diabatic")):
                cls = "resolved"
            else:
                cls = "unresolved"
            jf = c.get("jump_fit") or 0.0
            jg = c.get("jump_grid") or 0.0
            ja = c.get("jump_alt") or 0.0
            would = jf if jf else (jg if dec == "diabatic" else ja)
            rows.append({"sub": s.get("sub"), "nsub": s.get("nsub"), "src": src, "channel": c.get("channel"),
                         "i": c.get("i"), "n": c.get("n"), "kind": c.get("kind"), "decision": dec, "decided": decided,
                         "resolved": resolved, "falls": falls, "block": c.get("block"), "mirrored": c.get("mirrored"),
                         "self_mirror": c.get("self_mirror"), "s_star": c.get("s_star"), "gap_min": c.get("gap_min"),
                         "jump_fit_over_E0": jf / E0, "jump_grid_over_E0": jg / E0, "jump_alt_over_E0": ja / E0,
                         "would_be_over_E0": would / E0, "n_flags": len(c.get("flags") or []), "class": cls})
    return rows


def sub_rows(d):
    out = []
    for s in d.get("substeps") or []:
        p = s.get("prim") or {}
        out.append({"sub": s.get("sub"), "nsub": s.get("nsub"), "level": s.get("level"), "converged": s.get("converged"),
                    "iterations": s.get("iterations"), "r_chain_over_E_rec": s.get("r_chain_over_E_rec"),
                    "switch_sum_over_E_rec": s.get("switch_sum_over_E_rec"), "n_switch": s.get("n_switch"),
                    "prim_printable": p.get("printable"), "prim_stable": p.get("stable"), "prim_held": p.get("held"),
                    "prim_jumps_fall": p.get("jumps_fall"), "prim_jf_over_E_rec": p.get("jump_fit_sum_over_E_rec")})
    return out


def record(r, prevE, E0, segment, source, prev_pockets=None):
    """One composed record: bins, attribution, status."""
    d = r.get("dg_v36f") or {}
    T = r["T_over_rs"]
    E = r["E"]
    out = {"T_over_rs": T, "segment": segment, "source": source, "E": E, "E_over_E0": E / E0,
           "E_over_E0_minus_1": E / E0 - 1.0, "dE_over_E0": None if prevE is None else (E - prevE) / E0,
           "converged": r.get("converged"), "iterations": r.get("iterations"), "pockets": pockets(r),
           "inversions": r.get("inversions_total"), "trap": trap(r), "level": d.get("level"), "nsub": d.get("nsub"),
           "attempts": d.get("attempts"), "dg_seconds": d.get("seconds")}
    E_rec, E_start, E_end, E_pred = d.get("E_rec"), d.get("E_start"), d.get("E_end"), d.get("E_pred")
    out["E_rec_prev_matches"] = None if (prevE is None or E_rec is None) else abs(E_rec - prevE) / E0
    bins = {}
    if E_start is not None and E_end is not None and E_rec is not None:
        bins = {"start": (E_start - E_rec) / E0, "chord": (E_end - E_start) / E0, "record_reassign": (E - E_end) / E0}
        bins["closure"] = (out["dE_over_E0"] - (bins["start"] + bins["chord"] + bins["record_reassign"])) if out["dE_over_E0"] is not None else None
        rc = d.get("r_chain_over_E_rec")
        bins["r_chain_check"] = None if rc is None else rc * E_rec / E0 - bins["chord"]
    elif E_rec is not None and E_pred is not None:        # v36f records: the step's prediction, then the record solve
        bins = {"start": None, "chord": (E_pred - E_rec) / E0, "record_reassign": (E - E_pred) / E0, "closure": None, "r_chain_check": None}
        if out["dE_over_E0"] is not None:
            bins["closure"] = out["dE_over_E0"] - bins["chord"] - bins["record_reassign"]
    out["bins"] = bins
    att = {"resolved_jump_over_E0": 0.0, "n_resolved": 0, "n_resolved_diabatic": 0, "n_resolved_adiabatic": 0,
           "unresolved_would_be_over_E0": 0.0, "n_unresolved": 0, "overlap_switch_over_E0": 0.0, "n_overlap_switch": 0,
           "record_reassign_over_E0": bins.get("record_reassign"), "remainder_over_E0": None, "unattributed_over_E0": None}
    pm = d.get("prim") or {}
    cr = crossing_rows(d, E0) if pm else []
    out["crossings"] = cr
    out["substeps"] = sub_rows(d)
    if pm:
        for c in cr:
            if c["class"] == "resolved":
                att["resolved_jump_over_E0"] += c["jump_fit_over_E0"]; att["n_resolved"] += 1
                if c["decision"] == "diabatic" and c["jump_fit_over_E0"] != 0.0:
                    att["n_resolved_diabatic"] += 1
                else:
                    att["n_resolved_adiabatic"] += 1
            else:
                att["unresolved_would_be_over_E0"] += c["would_be_over_E0"]; att["n_unresolved"] += 1
        att["prim_jf_sum_check"] = None if pm.get("jump_fit_sum_over_E_rec") is None else \
            pm["jump_fit_sum_over_E_rec"] * E_rec / E0 - sum(c["jump_fit_over_E0"] for c in cr)
        if bins.get("chord") is not None:
            att["remainder_over_E0"] = bins["chord"] - att["resolved_jump_over_E0"] - sum(c["jump_fit_over_E0"] for c in cr if c["class"] != "resolved")
    elif d.get("n_switch") is not None:           # the anchor (maximum-overlap) convention: switches have no halving test
        att["overlap_switch_over_E0"] = (d.get("switch_sum_over_E_rec") or 0.0) * E_rec / E0
        att["n_overlap_switch"] = int(d.get("n_switch") or 0)
        if bins.get("chord") is not None:
            att["remainder_over_E0"] = bins["chord"] - att["overlap_switch_over_E0"]
    else:                                          # v36f: no bins beyond the record solve
        if out["dE_over_E0"] is not None:
            att["unattributed_over_E0"] = out["dE_over_E0"]
    out["attribution"] = att
    # the record's class under the criterion
    unres = []
    if att["n_unresolved"]:
        unres.append("%d unresolved primitive crossing(s)" % att["n_unresolved"])
    if att["n_overlap_switch"]:
        unres.append("%d maximum-overlap switch(es), no halving test" % att["n_overlap_switch"])
    if segment != "certified" and att["record_reassign_over_E0"] is not None and abs(att["record_reassign_over_E0"]) > FLOOR_REC:
        unres.append("record-solve reassignment %+.3e" % att["record_reassign_over_E0"])
    if segment == "certified" and att["unattributed_over_E0"] is not None and abs(att["unattributed_over_E0"]) > FLOOR_REC \
            and ((out["pockets"] or 0) > 0 or (prev_pockets or 0) > 0):
        unres.append("v36f record move with pockets, no bins")
    rem = att["remainder_over_E0"]
    rem_big = rem is not None and abs(rem) > S2_REM and (bins.get("chord") is None or abs(rem) > abs(bins["chord"]) / 2)
    if unres:
        cls = "unresolved" + (" + resolved" if att["n_resolved_diabatic"] else "")
    elif att["n_resolved_diabatic"]:
        cls = "resolved"
    elif att["n_resolved"]:
        cls = "resolved (adiabatic/exact, zero jump)"
    else:
        cls = "no reassignment"
    out["class"] = cls
    out["unresolved_items"] = unres
    out["remainder_above_S2"] = bool(rem_big)
    # printable status and cause
    if segment == "certified":
        st, cause = "certified", "before 12.5, the v36f lineage (no primitive; pre-epoch)"
    elif segment == "anchor":
        st = "no"
        cause = "the maximum-overlap (anchor) convention's trajectory (RUN_W2_v4 0.2)"
        if not r.get("converged"):
            cause += "; record solve unconverged"
    elif segment == "prim_shared":
        st = "yes" if pm.get("printable") and r.get("converged") else "no"
        cause = "" if st == "yes" else ", ".join(x for x in (
            "set not stable under halving" if pm.get("stable") is False else "",
            "set not held through the solve" if pm.get("held") is False else "",
            "a jump does not fall" if pm.get("jumps_fall") is False else "",
            "%d unresolved" % pm.get("n_unresolved") if pm.get("n_unresolved") else "",
            "record solve unconverged" if not r.get("converged") else "") if x) + " (prim run of the epoch; comparator for the rule dependence)"
    else:
        ok = bool(pm.get("printable")) and bool(r.get("converged"))
        st = "yes" if ok else "no"
        cause = "" if ok else ", ".join(x for x in (
            "no prim block" if not pm else "",
            "set not stable under halving" if pm.get("stable") is False else "",
            "set not held through the solve" if pm.get("held") is False else "",
            "a jump does not fall" if pm.get("jumps_fall") is False else "",
            "%d unresolved" % pm.get("n_unresolved") if pm.get("n_unresolved") else "",
            "record solve unconverged" if not r.get("converged") else "") if x)
        if ok and rem_big:
            cause = "printable; remainder above S2's threshold"
    out["printable"] = st
    out["cause"] = cause
    out["prim"] = {k: pm.get(k) for k in ("stable", "held", "jumps_fall", "converged", "printable_set", "printable", "level_stable",
                                          "n_crossings", "n_diabatic", "n_adiabatic", "n_unresolved", "n_reverted", "n_cycle",
                                          "end_gap_max", "tau_lz_min")} if pm else None
    return out


def log_state(p):
    if not p or not os.path.exists(p):
        return None
    last_T, stops, tbs, last_line = None, 0, 0, None
    with open(p, errors="replace") as f:
        for line in f:
            m = re.search(r"T/rs ([0-9]+\.[0-9]+)", line)
            if m:
                last_T = float(m.group(1))
            if "STOP" in line:
                stops += 1
            if "Traceback" in line:
                tbs += 1
            if line.strip():
                last_line = line.strip()[:300]
    return {"last_T_over_rs_in_log": last_T, "n_STOP": stops, "n_Traceback": tbs, "last_line": last_line}


def main():
    ap = argparse.ArgumentParser()
    for k in ("prim", "anc", "r2b", "v36v", "v36l", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--prim-log")
    ap.add_argument("--t-end", type=float, default=40.0)
    a = ap.parse_args()
    t0 = time.time()
    inputs = {k: {"path": os.path.abspath(getattr(a, k)), "md5": md5(getattr(a, k))} for k in ("prim", "anc", "r2b", "v36v", "v36l")}
    if a.prim_log:
        inputs["prim_log"] = {"path": os.path.abspath(a.prim_log), "md5": md5(a.prim_log)}
    JP, P = load(a.prim); JA, A = load(a.anc); JR, R2 = load(a.r2b); JV, V = load(a.v36v); JL, L = load(a.v36l)
    E0 = JP["head"]["E0"]
    checks = {"E0": {k: J["head"].get("E0") for k, J in (("prim", JP), ("anc", JA), ("r2b", JR), ("v36v", JV), ("v36l", JL))},
              "dg_assign": {k: J["args"].get("dg_assign") for k, J in (("prim", JP), ("anc", JA), ("r2b", JR), ("v36v", JV), ("v36l", JL))},
              "tag": {k: J["args"].get("tag") for k, J in (("prim", JP), ("anc", JA), ("r2b", JR), ("v36v", JV), ("v36l", JL))}}

    def maxdiff(X, Y, lo, hi):
        ks = [k for k in X if lo - 1e-9 <= k <= hi + 1e-9 and k in Y]
        return {"n": len(ks), "max_abs_dE_over_E0": max((abs(X[k]["E"] - Y[k]["E"]) / E0 for k in ks), default=None),
                "missing": [k for k in X if lo - 1e-9 <= k <= hi + 1e-9 and k not in Y]}
    checks["pre12.5_prim_vs_anc"] = maxdiff(A, P, 0.0, 12.5)
    checks["pre12.5_prim_vs_r2b"] = maxdiff(R2, P, 0.0, 12.5)
    checks["pre12.5_prim_vs_v36l"] = maxdiff(L, P, 0.0, 12.5)
    checks["epoch_prim_json_vs_anc"] = maxdiff(A, P, EP0, EP1)
    checks["epoch_anc_vs_v36l"] = maxdiff(A, L, EP0, EP1)
    checks["r2b_vs_v36v_shared"] = maxdiff(R2, V, EP0, 13.5)

    comp = []
    prevE = None
    for T in sorted(P):
        if T > a.t_end + 1e-9:
            break
        if T <= 12.5 + 1e-9:
            seg, src, r = "certified", "v36f lineage (J1; inherited, equal in every W2 json)", P[T]
        elif T <= EP1 + 1e-9:
            seg, src, r = "anchor", "J1_W2_anc (_v36p_anc, dg_assign anchor)", A.get(T, P[T])
        else:
            seg, src, r = "prim", "J1_W2_v36v_anc (_v36p_v36v_anc, v36v prim)", P[T]
        comp.append(record(r, prevE, E0, seg, src, prev_pockets=comp[-1]["pockets"] if comp else None))
        prevE = r["E"]

    # the rule dependence: anchor vs prim at the shared clocks
    rd = []
    prevE = A[12.5]["E"] if 12.5 in A else None
    pp = R2[12.5]["E"] if 12.5 in R2 else None
    for T in sorted(set(k for k in A if EP0 - 1e-9 <= k <= EP1 + 1e-9)):
        pr = R2.get(T) if T <= 13.5 + 1e-9 else V.get(T)
        prsrc = "R2' _r2b" if T <= 13.5 + 1e-9 else "v36v"
        if T > 13.5 + 1e-9 and pr is None:
            pr = None
        row = {"T_over_rs": T, "anchor_E_over_E0": A[T]["E"] / E0, "anchor_converged": A[T].get("converged"),
               "anchor_pockets": pockets(A[T])}
        if pr is not None:
            pm = (pr.get("dg_v36f") or {}).get("prim") or {}
            row.update({"prim_source": prsrc, "prim_E_over_E0": pr["E"] / E0, "prim_minus_anchor": (pr["E"] - A[T]["E"]) / E0,
                        "prim_converged": pr.get("converged"), "prim_printable": pm.get("printable"),
                        "prim_level": (pr.get("dg_v36f") or {}).get("level"), "prim_pockets": pockets(pr),
                        "prim_record": record(pr, pp, E0, "prim_shared", prsrc)})
            pp = pr["E"]
        rd.append(row)
    shared = next((x for x in rd if abs(x["T_over_rs"] - SHARED) < 1e-9), None)

    # the v36l maximum-overlap comparator: formation and the first episode's close (that convention's)
    Lrecs = [L[k] for k in sorted(L)]
    ivL = intervals(Lrecs)
    fL = next((r for r in Lrecs if r.get("trapping_zeros")), None)
    v36l = {"convention": "the maximum-overlap convention (v36l, dg_assign anchor; the comparator, RUN_W2_v4 0.4)",
            "records": len(Lrecs), "T_last": Lrecs[-1]["T_over_rs"],
            "formation_clock": fL and fL["T_over_rs"], "formation_R_t_over_rs": fL and fL["R_t"] / RS,
            "formation_E_over_E0": fL and fL["E"] / E0,
            "first_episode": ivL[0] if ivL else None,
            "first_episode_close": ivL[0][2] if ivL else None,
            "close_min_N_minus_absv": (L[ivL[0][2]].get("min_N_minus_absv") if ivL and ivL[0][2] is not None else None),
            "intervals_with_zeros": ivL,
            "E_over_E0_at": {str(t): (L[t]["E"] / E0 if t in L else None) for t in (12.0, 16.0, 20.0, 30.0, 40.0)},
            "min_N_minus_absv_at": {str(t): (L[t].get("min_N_minus_absv") if t in L else None) for t in (15.75, 15.875, 16.0, 20.0)}}
    # the composed history's own trapped intervals (anchor epoch, then prim), for the same comparison
    ivC = intervals([P[k] if k > EP1 + 1e-9 or k <= 12.5 + 1e-9 else A.get(k, P[k]) for k in sorted(P) if k <= a.t_end + 1e-9])
    composed_trap = {"convention": "the composed history (anchor epoch 12.625 to 14.5, prim from 14.625)",
                     "formation_clock": ivC[0][0] if ivC else None, "first_episode": ivC[0] if ivC else None,
                     "intervals_with_zeros": ivC}

    # the prim continuation against the v36l comparator from the shared 14.5 state (RUN_W2_v4 (e6), 0.2's test)
    pv = []
    for T in sorted(k for k in P if k > EP1 + 1e-9 and k <= a.t_end + 1e-9):
        if T in L:
            pv.append({"T_over_rs": T, "prim_minus_v36l": (P[T]["E"] - L[T]["E"]) / E0, "prim_pockets": pockets(P[T]),
                       "v36l_pockets": pockets(L[T]), "v36l_switches": (L[T].get("dg_v36f") or {}).get("n_switch")})
    run, rejoin = 0, None
    for x in pv:
        run = run + 1 if abs(x["prim_minus_v36l"]) <= FLOOR else 0
        if run == 3 and rejoin is None:
            rejoin = x["T_over_rs"]
    first_off = next((x["T_over_rs"] for x in pv if abs(x["prim_minus_v36l"]) > FLOOR), None)
    prim_vs_v36l = {"rows": pv, "floor": FLOOR, "third_consecutive_within_floor": rejoin, "first_clock_off_floor": first_off,
                    "max_abs": max((abs(x["prim_minus_v36l"]) for x in pv), default=None),
                    "last": pv[-1] if pv else None,
                    "note": "E/E0 only (masks not compared here; the spectra npz are not read)"}

    # sums by segment and class
    def seg_sum(sel):
        s = {"n": len(sel), "dE_over_E0": sum(x["dE_over_E0"] or 0.0 for x in sel)}
        for k in ("resolved_jump_over_E0", "unresolved_would_be_over_E0", "overlap_switch_over_E0"):
            s[k] = sum(x["attribution"][k] or 0.0 for x in sel)
        s["record_reassign_over_E0"] = sum(x["attribution"]["record_reassign_over_E0"] or 0.0 for x in sel)
        s["remainder_over_E0"] = sum(x["attribution"]["remainder_over_E0"] or 0.0 for x in sel)
        s["start_over_E0"] = sum((x["bins"].get("start") or 0.0) for x in sel)
        s["unattributed_over_E0"] = sum(x["attribution"]["unattributed_over_E0"] or 0.0 for x in sel)
        s["n_resolved"] = sum(x["attribution"]["n_resolved"] for x in sel)
        s["n_resolved_diabatic"] = sum(x["attribution"]["n_resolved_diabatic"] for x in sel)
        s["n_unresolved"] = sum(x["attribution"]["n_unresolved"] for x in sel)
        s["n_overlap_switch"] = sum(x["attribution"]["n_overlap_switch"] for x in sel)
        s["printable_yes"] = sum(1 for x in sel if x["printable"] == "yes")
        s["printable_no"] = sum(1 for x in sel if x["printable"] == "no")
        s["classes"] = {}
        for x in sel:
            s["classes"][x["class"]] = s["classes"].get(x["class"], 0) + 1
        s["max_abs_remainder_over_E0"] = max((abs(x["attribution"]["remainder_over_E0"]) for x in sel if x["attribution"]["remainder_over_E0"] is not None), default=None)
        s["max_abs_closure"] = max((abs(x["bins"]["closure"]) for x in sel if x["bins"].get("closure") is not None), default=None)
        s["max_E_rec_prev_mismatch"] = max((x["E_rec_prev_matches"] for x in sel if x["E_rec_prev_matches"] is not None), default=None)
        s["unconverged"] = [x["T_over_rs"] for x in sel if not x["converged"]]
        s["E_over_E0_first_last"] = [sel[0]["E_over_E0"], sel[-1]["E_over_E0"]] if sel else None
        return s
    segs = {k: seg_sum([x for x in comp if x["segment"] == k]) for k in ("certified", "anchor", "prim")}
    segs["certified_from_12"] = seg_sum([x for x in comp if x["segment"] == "certified" and x["T_over_rs"] >= 12.0 - 1e-9])
    T_last = comp[-1]["T_over_rs"] if comp else None
    out = {"script": "w2_readout_v1.py", "md5_self": md5(os.path.abspath(__file__)), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "inputs": inputs, "E0": E0, "floor": FLOOR, "floor_record_solve": FLOOR_REC, "S2_remainder": S2_REM, "t_end": a.t_end,
           "complete": bool(T_last is not None and T_last >= a.t_end - 1e-9), "T_last": T_last,
           "prim_log": log_state(a.prim_log), "checks": checks,
           "composition": [{"segment": "certified", "range": [0.0, 12.5]}, {"segment": "anchor", "range": [EP0, EP1]},
                           {"segment": "prim", "range": [PRIM0, T_last]}],
           "records": comp, "segments": segs,
           "rule_dependence": {"rows": rd, "shared_clock": SHARED,
                               "at_shared": None if shared is None else {k: shared.get(k) for k in ("T_over_rs", "anchor_E_over_E0", "prim_E_over_E0", "prim_minus_anchor", "prim_source", "prim_printable", "prim_level", "anchor_converged", "prim_converged")},
                               "anchor_at_14.5_E_over_E0": A[EP1]["E"] / E0 if EP1 in A else None},
           "v36l_comparator": v36l, "composed_trap": composed_trap, "prim_vs_v36l": prim_vs_v36l, "seconds": None}
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("w2_readout_v1: %d records 0 .. %.4f (complete to %.1f: %s); prim printable %d/%d; wrote %s in %.1f s" % (
        len(comp), T_last, a.t_end, out["complete"], segs["prim"]["printable_yes"], segs["prim"]["n"], a.out, out["seconds"]))


if __name__ == "__main__":
    main()
