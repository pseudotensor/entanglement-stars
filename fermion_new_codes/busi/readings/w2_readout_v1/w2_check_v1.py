#!/usr/bin/env python3
"""Cold correctness reading of W2_readout_v1 (checking lane, reviewer A): recompute every printed number from the
stored run jsons and logs with independent code, compare to the delivery's json at full precision and to its
printed table line by line, and run the extra checks the reading needs.

Runs on the box only (json arithmetic, 1 thread, no GPU):
  nice -n 10 python -I w2_check_v1.py --out-dir <internal>/w2_readout_v1
Inputs are read-only; nothing of the delivery's scripts is imported or executed.
"""
import argparse, hashlib, json, math, os, re, sys, time

E0_PRINT = None
RS = 64.0
FLOOR = 1e-9
FLOOR_REC = 5e-9
S2 = 2e-5
EPOCH = (12.625, 14.5)
B = "<internal>/final_40"
TAG = "lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p"
PATHS = {
    "prim": "%s/runs/%s_v36v_anc/collapse_%s_v36v_anc.json" % (B, TAG, TAG),
    "anc": "%s/runs/%s_anc/collapse_%s_anc.json" % (B, TAG, TAG),
    "r2b": "%s/runs/%s_r2b/collapse_%s_r2b.json" % (B, TAG, TAG),
    "v36v": "%s/runs/%s_v36v/collapse_%s_v36v.json" % (B, TAG, TAG),
    "v36l": "<internal>/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f.json",
}
LOGS = {k: "%s/logs/%s_%s.log" % (B, TAG, s) for k, s in (("prim", "v36v_anc"), ("anc", "anc"), ("r2b", "r2b"), ("v36v", "v36v"))}
DELIV_JSON = B + "/w2_readout_v1.json"
DELIV_TABLE = B + "/w2_readout_v1_table.out"
# the md5s the delivery states (W2_readout_v1.md and its table's inputs block)
STATED_MD5 = {"prim": "c40c10690dccd65a19bc89d7583b41c6", "anc": "bd3b26959df5214e146522eb174c3658",
              "r2b": "a73316c2b65ee24574cdcf9f045efe76", "v36v": "696ea929d0bd5fe92e61db2a72461283",
              "v36l": "1d4b69ef8ebed850f04f66a94b40b680", "prim_log": "ccf306b79a475d6ceeef7eaf20cb1320",
              "deliv_json": "0c3370d7ca0753a40d5337fe21586bb9", "deliv_table": "96ec99318daaa1cf8899777e863e6cc6"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fnum(x, fm="%+.3e"):
    return "-" if x is None else fm % x


def rk(T):
    return round(float(T), 6)


def load(p):
    J = json.load(open(p))
    return J, {rk(r["T_over_rs"]): r for r in J["records"]}


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def diff_keys(a, b):
    """top-level keys whose canonical json differs"""
    ks = set(a) | set(b)
    return sorted(k for k in ks if canon(a.get(k)) != canon(b.get(k)))


def npock(r):
    d = r.get("n_pocket") or {}
    return int(sum(v for v in d.values() if v is not None))


def has_zero(r):
    return bool(r.get("trapping_zeros"))


def intervals(recs):
    """[first clock with a zero, last clock with a zero, first clock after without a zero or None]"""
    out, cur = [], None
    for r in recs:
        T = r["T_over_rs"]
        if has_zero(r):
            if cur is None:
                cur = [T, T]
            else:
                cur[1] = T
        elif cur is not None:
            out.append(cur + [T]); cur = None
    if cur is not None:
        out.append(cur + [None])
    return out


# ----------------------------------------------------------------------------------------------------------
# the criterion, implemented afresh from RUN_W2_v4 sec. 7 and the Reorientation-5 wording
# ----------------------------------------------------------------------------------------------------------
def crossings_of(dg, E0):
    rows = []
    for s in dg.get("substeps") or []:
        p = s.get("prim") or {}
        blk = p.get("verify") if p.get("verify") else p.get("decision")
        src = "verify" if p.get("verify") else ("decision" if p.get("decision") else None)
        for c in (blk or {}).get("crossings") or []:
            dec = c.get("decision")
            resolved_cls = bool(c.get("decided")) and bool(c.get("resolved")) and (c.get("falls") is True or dec != "diabatic")
            jf = float(c.get("jump_fit") or 0.0); jg = float(c.get("jump_grid") or 0.0); ja = float(c.get("jump_alt") or 0.0)
            would = jf if jf != 0.0 else (jg if dec == "diabatic" else ja)
            rows.append({"sub": s.get("sub"), "nsub": s.get("nsub"), "src": src, "channel": c.get("channel"), "i": c.get("i"), "n": c.get("n"),
                         "kind": c.get("kind"), "decision": dec, "decided": c.get("decided"), "resolved": c.get("resolved"), "falls": c.get("falls"),
                         "gap_min": c.get("gap_min"), "jf": jf / E0, "jg": jg / E0, "ja": ja / E0, "would": would / E0,
                         "cls": "resolved" if resolved_cls else "unresolved", "mirrored": c.get("mirrored"), "block": c.get("block"),
                         "flags": len(c.get("flags") or [])})
    return rows


def one_record(r, prevE, prev_pk, E0, seg):
    dg = r.get("dg_v36f") or {}
    E = r["E"]
    o = {"T": r["T_over_rs"], "seg": seg, "E": E, "EE0": E / E0, "dE": None if prevE is None else (E - prevE) / E0,
         "conv": r.get("converged"), "it": r.get("iterations"), "pk": npock(r), "inv": r.get("inversions_total"),
         "lvl": dg.get("level"), "nsub": dg.get("nsub"), "att": dg.get("attempts"), "zero": has_zero(r),
         "R_t": r.get("R_t"), "minN": r.get("min_N_minus_absv")}
    E_rec, E_start, E_end, E_pred = dg.get("E_rec"), dg.get("E_start"), dg.get("E_end"), dg.get("E_pred")
    o["E_rec_mis"] = None if (prevE is None or E_rec is None) else abs(E_rec - prevE) / E0
    start = chord = rec = closure = rchk = None
    unatt = None
    if E_start is not None and E_end is not None and E_rec is not None:
        start = (E_start - E_rec) / E0; chord = (E_end - E_start) / E0; rec = (E - E_end) / E0
        if o["dE"] is not None:
            closure = o["dE"] - (start + chord + rec)
        rc = dg.get("r_chain_over_E_rec")
        rchk = None if rc is None else rc * E_rec / E0 - chord
    elif E_rec is not None and E_pred is not None:
        chord = (E_pred - E_rec) / E0; rec = (E - E_pred) / E0
        if o["dE"] is not None:
            closure = o["dE"] - chord - rec
    o.update({"start": start, "chord": chord, "rec": rec, "closure": closure, "rchain_check": rchk})
    pm = dg.get("prim") or {}
    cr = crossings_of(dg, E0) if pm else []
    o["cr"] = cr
    res_j = sum(c["jf"] for c in cr if c["cls"] == "resolved")
    n_res = sum(1 for c in cr if c["cls"] == "resolved")
    n_res_d = sum(1 for c in cr if c["cls"] == "resolved" and c["decision"] == "diabatic" and c["jf"] != 0.0)
    unres_w = sum(c["would"] for c in cr if c["cls"] != "resolved")
    n_unres = sum(1 for c in cr if c["cls"] != "resolved")
    sw = 0.0; n_sw = 0; rem = None
    if pm:
        if chord is not None:
            rem = chord - sum(c["jf"] for c in cr)
        o["engine_jf_sum"] = None if pm.get("jump_fit_sum_over_E_rec") is None else pm["jump_fit_sum_over_E_rec"] * E_rec / E0
        o["engine_n_dia"] = pm.get("n_diabatic"); o["engine_n_adi"] = pm.get("n_adiabatic"); o["engine_n_unres"] = pm.get("n_unresolved")
        o["engine_n_cross"] = pm.get("n_crossings")
        o["pflags"] = {k: pm.get(k) for k in ("stable", "held", "jumps_fall", "converged", "printable_set", "printable", "level_stable", "n_reverted", "n_cycle")}
    elif dg.get("n_switch") is not None:
        sw = float(dg.get("switch_sum_over_E_rec") or 0.0) * E_rec / E0; n_sw = int(dg.get("n_switch") or 0)
        if chord is not None:
            rem = chord - sw
        o["engine_rest"] = None if dg.get("rest_over_E_rec") is None else dg["rest_over_E_rec"] * E_rec / E0
    else:
        unatt = o["dE"]
    o.update({"res_j": res_j, "n_res": n_res, "n_res_d": n_res_d, "unres_w": unres_w, "n_unres": n_unres, "sw": sw, "n_sw": n_sw,
              "rem": rem, "unatt": unatt})
    items = []
    if n_unres:
        items.append("%d unresolved primitive crossing(s)" % n_unres)
    if n_sw:
        items.append("%d maximum-overlap switch(es), no halving test" % n_sw)
    if seg != "certified" and rec is not None and abs(rec) > FLOOR_REC:
        items.append("record-solve reassignment %+.3e" % rec)
    if seg == "certified" and unatt is not None and abs(unatt) > FLOOR_REC and ((o["pk"] or 0) > 0 or (prev_pk or 0) > 0):
        items.append("v36f record move with pockets, no bins")
    rem_big = rem is not None and abs(rem) > S2 and (chord is None or abs(rem) > abs(chord) / 2)
    if items:
        cls = "unresolved" + (" + resolved" if n_res_d else "")
    elif n_res_d:
        cls = "resolved"
    elif n_res:
        cls = "resolved (adiabatic/exact, zero jump)"
    else:
        cls = "no reassignment"
    o["cls"] = cls; o["items"] = items; o["rem_big"] = bool(rem_big)
    if seg == "certified":
        st, cause = "certified", "before 12.5, the v36f lineage (no primitive; pre-epoch)"
    elif seg == "anchor":
        st = "no"; cause = "the maximum-overlap (anchor) convention's trajectory (RUN_W2_v4 0.2)" + ("" if r.get("converged") else "; record solve unconverged")
    else:
        ok = bool(pm.get("printable")) and bool(r.get("converged"))
        st = "yes" if ok else "no"
        cause = "" if ok else ", ".join(x for x in ("no prim block" if not pm else "", "set not stable under halving" if pm.get("stable") is False else "",
                                                    "set not held through the solve" if pm.get("held") is False else "",
                                                    "a jump does not fall" if pm.get("jumps_fall") is False else "",
                                                    ("%d unresolved" % pm.get("n_unresolved")) if pm.get("n_unresolved") else "",
                                                    "record solve unconverged" if not r.get("converged") else "") if x)
        if ok and rem_big:
            cause = "printable; remainder above S2's threshold"
        # independent: printable from the engine's components
        o["printable_components"] = bool(pm) and all(bool(pm.get(k)) for k in ("stable", "held", "jumps_fall", "converged", "printable_set")) and bool(r.get("converged"))
    o["printable"] = st; o["cause"] = cause
    return o


def seg_sum(sel):
    s = {"n": len(sel), "dE": sum(x["dE"] or 0.0 for x in sel), "res_j": sum(x["res_j"] for x in sel), "unres_w": sum(x["unres_w"] for x in sel),
         "sw": sum(x["sw"] for x in sel), "rec": sum(x["rec"] or 0.0 for x in sel), "rem": sum(x["rem"] or 0.0 for x in sel),
         "start": sum(x["start"] or 0.0 for x in sel), "unatt": sum(x["unatt"] or 0.0 for x in sel),
         "n_res": sum(x["n_res"] for x in sel), "n_res_d": sum(x["n_res_d"] for x in sel), "n_unres": sum(x["n_unres"] for x in sel),
         "n_sw": sum(x["n_sw"] for x in sel), "p_yes": sum(1 for x in sel if x["printable"] == "yes"), "p_no": sum(1 for x in sel if x["printable"] == "no"),
         "classes": {}, "max_rem": max((abs(x["rem"]) for x in sel if x["rem"] is not None), default=None),
         "max_closure": max((abs(x["closure"]) for x in sel if x["closure"] is not None), default=None),
         "max_E_rec_mis": max((x["E_rec_mis"] for x in sel if x["E_rec_mis"] is not None), default=None),
         "unconverged": [x["T"] for x in sel if not x["conv"]], "EE0_first_last": [sel[0]["EE0"], sel[-1]["EE0"]] if sel else None,
         "max_rchain_check": max((abs(x["rchain_check"]) for x in sel if x["rchain_check"] is not None), default=None)}
    for x in sel:
        s["classes"][x["cls"]] = s["classes"].get(x["cls"], 0) + 1
    return s


def log_scan(p):
    lastT, stops, tbs, nrec, last = None, 0, 0, 0, None
    stop_lines = []
    with open(p, errors="replace") as f:
        for line in f:
            m = re.search(r"T/rs ([0-9]+\.[0-9]+)", line)
            if m:
                lastT = float(m.group(1))
            if "STOP" in line:
                stops += 1; stop_lines.append(line.strip()[:260])
            if "Traceback" in line:
                tbs += 1
            if re.match(r"^T/rs [0-9.]+ E/E0", line):
                nrec += 1
            if line.strip():
                last = line.strip()[:200]
    return {"last_T": lastT, "STOP": stops, "Traceback": tbs, "record_lines": nrec, "last_line": last, "stop_lines": stop_lines[-2:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="<internal>/w2_readout_v1")
    ap.add_argument("--t-end", type=float, default=40.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_txt = open(os.path.join(a.out_dir, "w2_check_v1.out"), "w")

    def P(*s):
        line = " ".join(str(x) for x in s)
        print(line); out_txt.write(line + "\n")

    t0 = time.time()
    P("# w2_check_v1.py (reviewer A cold reading of W2_readout_v1); md5_self %s; %s" % (md5(os.path.abspath(__file__)), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    R = {"script_md5": md5(os.path.abspath(__file__))}
    # ---------------- 1. provenance -----------------
    P("\n## 1. provenance (md5 of every input against the delivery's statement)")
    prov = {}
    for k, p in list(PATHS.items()) + [("prim_log", LOGS["prim"]), ("deliv_json", DELIV_JSON), ("deliv_table", DELIV_TABLE)]:
        h = md5(p); prov[k] = {"path": p, "md5": h, "stated": STATED_MD5.get(k), "match": h == STATED_MD5.get(k)}
        P("  %-11s %s stated %s %s" % (k, h, STATED_MD5.get(k), "MATCH" if h == STATED_MD5.get(k) else "MISMATCH"))
    for k in ("anc", "r2b", "v36v"):
        prov[k + "_log"] = {"path": LOGS[k], "md5": md5(LOGS[k])}
        P("  %-11s %s (not stated in the delivery)" % (k + "_log", prov[k + "_log"]["md5"]))
    R["provenance"] = prov
    D = json.load(open(DELIV_JSON))
    J = {}; M = {}
    for k, p in PATHS.items():
        J[k], M[k] = load(p)
    E0s = {k: J[k]["head"]["E0"] for k in J}
    E0 = E0s["prim"]
    P("  E0 per json:", E0s, "all equal:", len(set(E0s.values())) == 1)
    P("  args: dg_assign", {k: J[k]["args"].get("dg_assign") for k in J}, "| v36v-only args in prim json: dg_stall", J["prim"]["args"].get("dg_stall"), "dg_sub_counts", J["prim"]["args"].get("dg_sub_counts"))
    P("  prim json resume_slice:", J["prim"]["args"].get("resume_slice"))
    P("  records per json:", {k: len(J[k]["records"]) for k in J}, "last clocks:", {k: max(M[k]) for k in M})
    R["E0"] = E0; R["records_per_json"] = {k: len(J[k]["records"]) for k in J}; R["last_clock"] = {k: max(M[k]) for k in M}

    # ---------------- 2. composition -----------------
    P("\n## 2. composition: record equality (E exactly, and the whole record dict)")
    comp = {}

    def cmp(X, Y, lo, hi, name):
        ks = sorted(k for k in M[X] if lo - 1e-9 <= k <= hi + 1e-9)
        miss = [k for k in ks if k not in M[Y]]
        ks = [k for k in ks if k in M[Y]]
        dEmax = max((abs(M[X][k]["E"] - M[Y][k]["E"]) / E0 for k in ks), default=None)
        nE = sum(1 for k in ks if M[X][k]["E"] != M[Y][k]["E"])
        full_diff = [(k, diff_keys(M[X][k], M[Y][k])) for k in ks if canon(M[X][k]) != canon(M[Y][k])]
        keys_union = sorted(set(kk for _, d in full_diff for kk in d))
        comp[name] = {"n": len(ks), "missing": miss, "n_E_unequal": nE, "max_abs_dE_over_E0": dEmax, "n_records_not_bitwise": len(full_diff), "differing_keys": keys_union}
        P("  %-28s n %3d missing %s | E unequal on %d (max |dE|/E0 %s) | whole record differs on %d; keys: %s" % (
            name, len(ks), miss, nE, fnum(dEmax, "%.2e"), len(full_diff), keys_union[:12]))
    for Y in ("anc", "r2b", "v36v", "v36l"):
        cmp("prim", Y, 0.0, 12.5, "pre12.5 prim vs " + Y)
    cmp("anc", "v36l", EPOCH[0], EPOCH[1], "epoch anc vs v36l")
    cmp("anc", "prim", EPOCH[0], EPOCH[1], "epoch anc vs prim-json")
    cmp("r2b", "v36v", EPOCH[0], 13.5, "r2b vs v36v 12.625..13.5")
    cmp("prim", "v36l", 14.625, a.t_end, "prim vs v36l 14.625..40 (E only matters)")
    R["composition"] = comp

    # ---------------- 3. the composed history -----------------
    hist = []
    prevE = None; prev_pk = None
    for T in sorted(M["prim"]):
        if T > a.t_end + 1e-9:
            break
        if T <= 12.5 + 1e-9:
            seg, r = "certified", M["prim"][T]
        elif T <= EPOCH[1] + 1e-9:
            seg, r = "anchor", M["anc"][T]
        else:
            seg, r = "prim", M["prim"][T]
        o = one_record(r, prevE, prev_pk, E0, seg)
        hist.append(o); prevE = r["E"]; prev_pk = o["pk"]
    H = {rk(o["T"]): o for o in hist}
    segs = {k: seg_sum([o for o in hist if o["seg"] == k]) for k in ("certified", "anchor", "prim")}
    segs["certified_from_12"] = seg_sum([o for o in hist if o["seg"] == "certified" and o["T"] >= 12.0 - 1e-9])
    R["segments"] = segs
    P("\n## 3. the composed history: %d records, 0 .. %.4f" % (len(hist), hist[-1]["T"]))
    for k, s in segs.items():
        P("  %-18s n %3d dE %+.6e res %+.6e (%d dia of %d) unres %+.3e (%d) sw %+.6e (%d) rec %+.6e rem %+.6e start %+.6e unatt %+.6e" % (
            k, s["n"], s["dE"], s["res_j"], s["n_res_d"], s["n_res"], s["unres_w"], s["n_unres"], s["sw"], s["n_sw"], s["rec"], s["rem"], s["start"], s["unatt"]))
        P("  %-18s printable yes %d no %d; classes %s; max|rem| %s; max|closure| %s; max E_rec mismatch %s; max |r_chain*E_rec/E0 - chord| %s; unconverged %s; E/E0 first/last %s" % (
            "", s["p_yes"], s["p_no"], s["classes"], fnum(s["max_rem"], "%.4e"), fnum(s["max_closure"], "%.1e"), fnum(s["max_E_rec_mis"], "%.1e"), fnum(s["max_rchain_check"], "%.1e"), s["unconverged"], s["EE0_first_last"]))

    # ---------------- 4. per-record comparison with the delivery's json -----------------
    P("\n## 4. per-record comparison with the delivery's json (max |mine - delivery| over the 321 records)")
    DR = {rk(x["T_over_rs"]): x for x in D["records"]}
    keymap = {"EE0": ("E_over_E0",), "dE": ("dE_over_E0",), "start": ("bins", "start"), "chord": ("bins", "chord"), "rec": ("bins", "record_reassign"),
              "closure": ("bins", "closure"), "res_j": ("attribution", "resolved_jump_over_E0"), "unres_w": ("attribution", "unresolved_would_be_over_E0"),
              "sw": ("attribution", "overlap_switch_over_E0"), "rem": ("attribution", "remainder_over_E0"), "unatt": ("attribution", "unattributed_over_E0")}
    cmpres = {}
    for mk, path in keymap.items():
        mx = 0.0; nnone = 0; worst = None
        for T, o in H.items():
            x = DR[T]
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
            mine = o[mk]
            if mine is None or x is None:
                if (mine is None) != (x is None):
                    nnone += 1
                continue
            d = abs(mine - x)
            if d > mx:
                mx = d; worst = T
        cmpres[mk] = {"max_abs_diff": mx, "worst_T": worst, "none_mismatch": nnone}
        P("  %-8s max |diff| %.2e at %s; None-pattern mismatches %d" % (mk, mx, worst, nnone))
    for mk, dk in (("n_res", "n_resolved"), ("n_res_d", "n_resolved_diabatic"), ("n_unres", "n_unresolved"), ("n_sw", "n_overlap_switch")):
        bad = [T for T, o in H.items() if o[mk] != DR[T]["attribution"][dk]]
        cmpres[mk] = bad; P("  %-8s count mismatches at %s" % (mk, bad))
    for mk, dk in (("cls", "class"), ("printable", "printable"), ("cause", "cause"), ("pk", "pockets"), ("lvl", "level"), ("nsub", "nsub"), ("rem_big", "remainder_above_S2"), ("seg", "segment")):
        bad = [T for T, o in H.items() if o[mk] != DR[T][dk]]
        cmpres[mk] = bad; P("  %-8s string/int mismatches at %s" % (mk, bad))
    R["vs_delivery_json_records"] = cmpres
    # segments
    P("  segment sums vs delivery:")
    smap = {"dE": "dE_over_E0", "res_j": "resolved_jump_over_E0", "unres_w": "unresolved_would_be_over_E0", "sw": "overlap_switch_over_E0", "rec": "record_reassign_over_E0",
            "rem": "remainder_over_E0", "start": "start_over_E0", "unatt": "unattributed_over_E0", "n_res": "n_resolved", "n_res_d": "n_resolved_diabatic", "n_unres": "n_unresolved",
            "n_sw": "n_overlap_switch", "p_yes": "printable_yes", "p_no": "printable_no", "max_rem": "max_abs_remainder_over_E0", "max_closure": "max_abs_closure", "n": "n"}
    sdiff = {}
    for k in segs:
        dd = {}
        for mk, dk in smap.items():
            mine, theirs = segs[k][mk], D["segments"][k].get(dk)
            if mine is None or theirs is None:
                dd[mk] = None if mine is theirs else "NONE-MISMATCH"
            else:
                dd[mk] = abs(mine - theirs)
        bigs = {mk: v for mk, v in dd.items() if v is None or v == "NONE-MISMATCH" or v > 1e-15}
        sdiff[k] = dd
        P("   %-18s diffs above 1e-15 or None-mismatch: %s; classes equal %s; unconverged equal %s" % (k, bigs, segs[k]["classes"] == D["segments"][k]["classes"], segs[k]["unconverged"] == D["segments"][k]["unconverged"]))
    R["vs_delivery_segments"] = sdiff

    # ---------------- 5. the prim segment: every crossing, the engine's own counts, the flags -----------------
    P("\n## 5. the prim segment 14.625 .. 40: crossings, the engine's own counts and flags")
    prim = [o for o in hist if o["seg"] == "prim"]
    P("  n %d; converged all %s; printable yes %d; printable (from the engine's components stable&held&jumps_fall&converged&printable_set & record converged) %d; disagreements %s" % (
        len(prim), all(o["conv"] for o in prim), sum(1 for o in prim if o["printable"] == "yes"), sum(1 for o in prim if o.get("printable_components")),
        [o["T"] for o in prim if (o["printable"] == "yes") != bool(o.get("printable_components"))]))
    P("  engine sums over the prim records: n_crossings %d, n_diabatic %d, n_adiabatic %d, n_unresolved %d; my crossing rows %d (resolved %d, of which diabatic with a nonzero fitted jump %d, zero-jump %d)" % (
        sum(o["engine_n_cross"] or 0 for o in prim), sum(o["engine_n_dia"] or 0 for o in prim), sum(o["engine_n_adi"] or 0 for o in prim), sum(o["engine_n_unres"] or 0 for o in prim),
        sum(len(o["cr"]) for o in prim), sum(o["n_res"] for o in prim), sum(o["n_res_d"] for o in prim), sum(1 for o in prim for c in o["cr"] if c["cls"] == "resolved" and c["jf"] == 0.0)))
    P("  engine jump_fit sums vs my crossing sums: max |diff| %.2e" % max(abs((o["engine_jf_sum"] or 0.0) - sum(c["jf"] for c in o["cr"])) for o in prim))
    P("  levels used: %s; attempts: %s; max level %s" % (sorted(set(o["lvl"] for o in prim)), sorted(set(o["att"] for o in prim)), max(o["lvl"] for o in prim)))
    P("  records with a reverted hold (n_reverted>0): %s; with a cycle: %s" % ([o["T"] for o in prim if (o["pflags"].get("n_reverted") or 0) > 0], [o["T"] for o in prim if (o["pflags"].get("n_cycle") or 0) > 0]))
    P("  records with pockets: %s" % [(o["T"], o["pk"]) for o in prim if o["pk"]])
    P("  records with |record-solve move| > %.0e: %s" % (FLOOR_REC, [(o["T"], "%+.3e" % o["rec"]) for o in prim if o["rec"] is not None and abs(o["rec"]) > FLOOR_REC]))
    P("  records with |start bin| > 1e-9: %s" % [(o["T"], "%+.3e" % o["start"]) for o in prim if o["start"] is not None and abs(o["start"]) > 1e-9])
    P("  records with class != 'no reassignment': %s" % [(o["T"], o["cls"]) for o in prim if o["cls"] != "no reassignment"])
    P("  every crossing:")
    for o in prim:
        for c in o["cr"]:
            P("   %.3f sub %s/%s src %s ch %s %s->%s %s %s decided %s resolved %s falls %s mirrored %s block %s flags %d gap_min %s jf %+.4e jg %+.4e ja %+.4e -> %s" % (
                o["T"], c["sub"], c["nsub"], c["src"], c["channel"], c["i"], c["n"], c["kind"], c["decision"], c["decided"], c["resolved"], c["falls"], c["mirrored"], c["block"], c["flags"],
                fnum(c["gap_min"], "%.3e"), c["jf"], c["jg"], c["ja"], c["cls"]))
    ep2 = [o for o in prim if 23.25 - 1e-9 <= o["T"] <= 23.875 + 1e-9]
    P("  second episode 23.25..23.875: sum dE %+.4e, sum resolved jumps %+.4e, sum remainder %+.4e, sum start %+.3e, sum rec %+.3e; (E(23.875)-E(23.125))/E0 %+.4e" % (
        sum(o["dE"] for o in ep2), sum(o["res_j"] for o in ep2), sum(o["rem"] for o in ep2), sum(o["start"] for o in ep2), sum(o["rec"] for o in ep2), (H[23.875]["E"] - H[23.125]["E"]) / E0))
    # S2 as a stop rule: 3 consecutive printable records with |rem| > 2e-5 of E (E, not E0) and > |r_chain|/2
    s2 = []
    for o in prim:
        Eprev = o["E"] - (o["dE"] or 0.0) * E0
        remE = (o["rem"] or 0.0) * E0 / Eprev
        s2.append(abs(remE) > S2 and abs(o["rem"] or 0.0) > abs(o["chord"] or 0.0) / 2)
    trip = [prim[i]["T"] for i in range(2, len(prim)) if s2[i] and s2[i - 1] and s2[i - 2]]
    P("  S2 (|rem| > 2e-5 of E and > |r_chain|/2) holds on records %s; triplets %s; max |rem|/E %.4e at %s" % (
        [o["T"] for o, f in zip(prim, s2) if f], trip, max(abs((o["rem"] or 0.0) * E0 / (o["E"] - (o["dE"] or 0.0) * E0)) for o in prim), max(prim, key=lambda o: abs(o["rem"] or 0.0))["T"]))
    R["prim_segment"] = {"n": len(prim), "printable_yes": sum(1 for o in prim if o["printable"] == "yes"), "n_cross": sum(len(o["cr"]) for o in prim),
                         "engine_n_dia": sum(o["engine_n_dia"] or 0 for o in prim), "engine_n_adi": sum(o["engine_n_adi"] or 0 for o in prim),
                         "classes": segs["prim"]["classes"], "rec_above_floor": [(o["T"], o["rec"]) for o in prim if abs(o["rec"] or 0) > FLOOR_REC],
                         "second_episode": {"dE": sum(o["dE"] for o in ep2), "res_j": sum(o["res_j"] for o in ep2), "rem": sum(o["rem"] for o in ep2)}, "S2_records": [o["T"] for o, f in zip(prim, s2) if f]}

    # ---------------- 6. the anchor epoch -----------------
    P("\n## 6. the anchor epoch 12.625 .. 14.5")
    anc = [o for o in hist if o["seg"] == "anchor"]
    P("  switch counts per record: %s (sum %d); switch sums %s" % ([o["n_sw"] for o in anc], sum(o["n_sw"] for o in anc), ["%+.3e" % o["sw"] for o in anc]))
    P("  engine rest (rest_over_E_rec*E_rec/E0) vs my remainder: max |diff| %.2e" % max(abs((o.get("engine_rest") or 0.0) - (o["rem"] or 0.0)) for o in anc))
    P("  record solves unconverged: %s; their record-solve moves: %s" % ([o["T"] for o in anc if not o["conv"]], ["%+.3e" % o["rec"] for o in anc if not o["conv"]]))
    P("  net (E(14.5)-E(12.5))/E0 = %+.6e; sum of bins start %+.4e + sw %+.4e + rem %+.4e + rec %+.4e = %+.6e" % (
        (M["anc"][14.5]["E"] - M["anc"][12.5]["E"]) / E0, segs["anchor"]["start"], segs["anchor"]["sw"], segs["anchor"]["rem"], segs["anchor"]["rec"],
        segs["anchor"]["start"] + segs["anchor"]["sw"] + segs["anchor"]["rem"] + segs["anchor"]["rec"]))
    P("  12.625 alone: dE %+.4e, sw %+.4e (%d), rem %+.4e, rec %+.4e, start %+.3e" % (anc[0]["dE"], anc[0]["sw"], anc[0]["n_sw"], anc[0]["rem"], anc[0]["rec"], anc[0]["start"]))
    P("  records with |rem| > 2e-5 and > |chord|/2 (the * flag): %s" % [o["T"] for o in anc if o["rem_big"]])
    P("  pockets: %s" % [o["pk"] for o in anc])

    # ---------------- 7. the deficit at 40 by segment -----------------
    P("\n## 7. the deficit at 40 and where it is set")
    dtot = H[40.0]["EE0"] - 1.0
    parts = {"certified 0..12.25": (M["prim"][12.25]["E"] - E0) / E0, "v36f 12.375": H[12.375]["dE"], "v36f 12.5": H[12.5]["dE"],
             "anchor epoch 12.625..14.5": segs["anchor"]["dE"], "prim 14.625..40": segs["prim"]["dE"]}
    P("  E/E0(40) - 1 = %+.6e; parts: %s; sum %+.6e" % (dtot, {k: "%+.4e (%.1f%%)" % (v, 100 * v / dtot) for k, v in parts.items()}, sum(parts.values())))
    R["deficit_40"] = {"total": dtot, "parts": parts}

    # ---------------- 8. the rule dependence in the epoch -----------------
    P("\n## 8. the rule dependence at the shared clocks (anchor vs R2' to 13.5, v36v at 13.625)")
    rd = []
    for T in sorted(k for k in M["anc"] if EPOCH[0] - 1e-9 <= k <= EPOCH[1] + 1e-9):
        pr = M["r2b"].get(T) if T <= 13.5 + 1e-9 else M["v36v"].get(T)
        row = {"T": T, "anc": M["anc"][T]["E"] / E0, "anc_conv": M["anc"][T].get("converged"), "anc_pk": npock(M["anc"][T])}
        if pr is not None:
            pm = (pr.get("dg_v36f") or {}).get("prim") or {}
            row.update({"src": "R2' _r2b" if T <= 13.5 + 1e-9 else "v36v", "prim": pr["E"] / E0, "diff": (pr["E"] - M["anc"][T]["E"]) / E0, "prim_conv": pr.get("converged"),
                        "prim_printable": pm.get("printable"), "lvl": (pr.get("dg_v36f") or {}).get("level"), "att": (pr.get("dg_v36f") or {}).get("attempts"), "prim_pk": npock(pr),
                        "stable": pm.get("stable"), "held": pm.get("held"), "jumps_fall": pm.get("jumps_fall"), "n_unres": pm.get("n_unresolved")})
        rd.append(row)
        P("  %.3f anchor %.9f conv %s pk %d | prim %s %s diff %s printable %s lvl %s att %s conv %s pk %s stable %s held %s jumps_fall %s n_unres %s" % (
            T, row["anc"], row["anc_conv"], row["anc_pk"], row.get("src", "-"), fnum(row.get("prim"), "%.9f"), fnum(row.get("diff")), row.get("prim_printable", "-"), row.get("lvl", "-"), row.get("att", "-"),
            row.get("prim_conv", "-"), row.get("prim_pk", "-"), row.get("stable", "-"), row.get("held", "-"), row.get("jumps_fall", "-"), row.get("n_unres", "-")))
    DRD = {rk(x["T_over_rs"]): x for x in D["rule_dependence"]["rows"]}
    mx = max(abs(r["anc"] - DRD[r["T"]]["anchor_E_over_E0"]) for r in rd)
    mx2 = max(abs(r["diff"] - DRD[r["T"]]["prim_minus_anchor"]) for r in rd if "diff" in r)
    P("  vs delivery: max |anchor E/E0 diff| %.1e, max |prim-anchor diff| %.1e; at 13.625 mine %+.6e delivery %+.6e" % (mx, mx2, [r for r in rd if r["T"] == 13.625][0]["diff"], D["rule_dependence"]["at_shared"]["prim_minus_anchor"]))
    P("  prim records of the epoch printable (engine flag): %s" % [(r["T"], r.get("prim_printable")) for r in rd if "prim" in r])
    R["rule_dependence"] = rd

    # ---------------- 9. the v36l comparator and the composed history's trapping intervals -----------------
    P("\n## 9. trapping zeros: the v36l comparator and the composed history")
    Lrecs = [M["v36l"][k] for k in sorted(M["v36l"])]
    ivL = intervals(Lrecs)
    fL = next(r for r in Lrecs if has_zero(r))
    P("  v36l: %d records to %.3f; formation (first record with a trapping zero) %.3f (R_t %.4f, R_t/r_s %.4f, E/E0 %.9f); that record's segment in the composed history: %s" % (
        len(Lrecs), Lrecs[-1]["T_over_rs"], fL["T_over_rs"], fL["R_t"], fL["R_t"] / RS, fL["E"] / E0, H[rk(fL["T_over_rs"])]["seg"]))
    P("  v36l intervals [first, last, first clock after]: %s" % ivL)
    P("  v36l min(N-|v|) at 15.75 %+.5f, 15.875 %+.5f, 16.0 %+.5f, 20.0 %+.5f; zeros at 15.75 %d, 15.875 %d" % (
        M["v36l"][15.75]["min_N_minus_absv"], M["v36l"][15.875]["min_N_minus_absv"], M["v36l"][16.0]["min_N_minus_absv"], M["v36l"][20.0]["min_N_minus_absv"], len(M["v36l"][15.75]["trapping_zeros"] or []), len(M["v36l"][15.875]["trapping_zeros"] or [])))
    P("  v36l E/E0 at 12, 16, 20, 30, 40: %s" % ["%.10f" % (M["v36l"][t]["E"] / E0) for t in (12.0, 16.0, 20.0, 30.0, 40.0)])
    comp_recs = []
    for T in sorted(M["prim"]):
        if T > a.t_end + 1e-9:
            break
        comp_recs.append(M["anc"][T] if EPOCH[0] - 1e-9 <= T <= EPOCH[1] + 1e-9 else M["prim"][T])
    ivC = intervals(comp_recs)
    P("  composed intervals: %s" % ivC)
    P("  composed: 15.75 zeros %d (seg %s, printable %s), 15.875 zeros %d (seg %s, printable %s), min(N-|v|) at 15.875 %+.5f; 37.25 zeros prim %d v36l %d" % (
        len(M["prim"][15.75]["trapping_zeros"] or []), H[15.75]["seg"], H[15.75]["printable"], len(M["prim"][15.875]["trapping_zeros"] or []), H[15.875]["seg"], H[15.875]["printable"],
        M["prim"][15.875]["min_N_minus_absv"], len(M["prim"][37.25]["trapping_zeros"] or []), len(M["v36l"][37.25]["trapping_zeros"] or [])))
    P("  vs delivery: v36l intervals equal %s; composed intervals equal %s; formation %s; close %s; close min(N-|v|) diff %.1e" % (
        ivL == D["v36l_comparator"]["intervals_with_zeros"], ivC == D["composed_trap"]["intervals_with_zeros"], fL["T_over_rs"] == D["v36l_comparator"]["formation_clock"],
        ivL[0][2] == D["v36l_comparator"]["first_episode_close"], abs(M["v36l"][15.875]["min_N_minus_absv"] - D["v36l_comparator"]["close_min_N_minus_absv"])))
    R["trap"] = {"v36l_intervals": ivL, "composed_intervals": ivC, "formation": fL["T_over_rs"], "formation_R_t_over_rs": fL["R_t"] / RS}

    # ---------------- 10. prim vs v36l from the shared 14.5 state -----------------
    P("\n## 10. the prim continuation against v36l from 14.5 (E/E0 only)")
    pv = []
    for T in sorted(k for k in M["prim"] if k > EPOCH[1] + 1e-9 and k <= a.t_end + 1e-9 and k in M["v36l"]):
        pv.append({"T": T, "d": (M["prim"][T]["E"] - M["v36l"][T]["E"]) / E0, "pk_p": npock(M["prim"][T]), "pk_l": npock(M["v36l"][T]), "sw_l": (M["v36l"][T].get("dg_v36f") or {}).get("n_switch")})
    run = 0; rejoin = None
    for x in pv:
        run = run + 1 if abs(x["d"]) <= FLOOR else 0
        if run == 3 and rejoin is None:
            rejoin = x["T"]
    first_off = next((x["T"] for x in pv if abs(x["d"]) > FLOOR), None)
    last_on = max((x["T"] for x in pv if x["T"] < first_off), default=None) if first_off else None
    P("  n %d; third consecutive within 1e-9: %s; last clock within the floor before the separation: %s; first clock off the floor: %s; max |d| %.4e; last %.3f %+.4e" % (
        len(pv), rejoin, last_on, first_off, max(abs(x["d"]) for x in pv), pv[-1]["T"], pv[-1]["d"]))
    P("  max |d| on 14.625..23.25: %.2e; d at 23.375 %+.4e, 23.5 %+.4e, 23.625 %+.4e, 23.75 %+.4e; from 23.75 to 40 min %+.4e max %+.4e" % (
        max(abs(x["d"]) for x in pv if x["T"] <= 23.25 + 1e-9), [x["d"] for x in pv if x["T"] == 23.375][0], [x["d"] for x in pv if x["T"] == 23.5][0], [x["d"] for x in pv if x["T"] == 23.625][0], [x["d"] for x in pv if x["T"] == 23.75][0],
        min(x["d"] for x in pv if x["T"] >= 23.75 - 1e-9), max(x["d"] for x in pv if x["T"] >= 23.75 - 1e-9)))
    prev = 0.0; moved = []
    for x in pv:
        if abs(x["d"] - prev) > FLOOR or x["pk_p"] or x["pk_l"]:
            moved.append(x["T"])
        prev = x["d"]
    P("  clocks where d moves by more than 1e-9 or either has pockets: %s" % moved)
    DPV = {rk(x["T_over_rs"]): x["prim_minus_v36l"] for x in D["prim_vs_v36l"]["rows"]}
    P("  vs delivery: max |d diff| %.1e; rejoin %s/%s; first_off %s/%s; max_abs diff %.1e" % (max(abs(x["d"] - DPV[x["T"]]) for x in pv), rejoin, D["prim_vs_v36l"]["third_consecutive_within_floor"], first_off, D["prim_vs_v36l"]["first_clock_off_floor"], abs(max(abs(x["d"]) for x in pv) - D["prim_vs_v36l"]["max_abs"])))
    R["prim_vs_v36l"] = {"rejoin": rejoin, "first_off": first_off, "last_on": last_on, "max_abs": max(abs(x["d"]) for x in pv), "last": pv[-1], "moved": moved}
    # the v36l second-episode moves beside the prim's
    P("  v36l's own second episode: dE/E0 at 23.25..23.875: %s; (E_l(23.875)-E_l(23.125))/E0 %+.4e vs prim %+.4e" % (
        ["%+.3e" % ((M["v36l"][T]["E"] - M["v36l"][rk(T - 0.125)]["E"]) / E0) for T in (23.25, 23.375, 23.5, 23.625, 23.75, 23.875)], (M["v36l"][23.875]["E"] - M["v36l"][23.125]["E"]) / E0, (M["prim"][23.875]["E"] - M["prim"][23.125]["E"]) / E0))

    # ---------------- 11. the logs and the run ends -----------------
    P("\n## 11. the logs: ends, STOP and Traceback counts (fixed-string counts)")
    logs = {}
    for k, p in LOGS.items():
        logs[k] = log_scan(p)
        P("  %-5s last T/rs %s, STOP %d, Traceback %d, record lines %d; last json clock %s" % (k, logs[k]["last_T"], logs[k]["STOP"], logs[k]["Traceback"], logs[k]["record_lines"], max(M[k])))
        for s in logs[k]["stop_lines"]:
            P("        STOP line: %s" % s)
    R["logs"] = logs

    # ---------------- 12. regenerate the printed table from my values and diff it -----------------
    P("\n## 12. the printed table regenerated from my values, diffed line by line against the delivery's")
    lines = []
    lines.append("E0 %.12g; records %d, T/r_s 0 .. %.4f; complete to %.1f: %s; floor %.0e, S2 remainder %.0e" % (E0, len(hist), hist[-1]["T"], a.t_end, "YES" if hist[-1]["T"] >= a.t_end - 1e-9 else "NO (test on the records that exist)", FLOOR, S2))
    lines.append("record-solve floor %.0e (a record-solve move above it is counted as an unresolved reassignment)" % FLOOR_REC)
    lines.append("prim log: last T/rs %s, STOP %d, Traceback %d" % (logs["prim"]["last_T"], logs["prim"]["STOP"], logs["prim"]["Traceback"]))
    lines.append("")
    lines.append("## inputs")
    for k in ("prim", "anc", "r2b", "v36v", "v36l"):
        lines.append("  %-8s %s  %s" % (k, prov[k]["md5"], PATHS[k]))
    lines.append("  %-8s %s  %s" % ("prim_log", prov["prim_log"]["md5"], LOGS["prim"]))
    lines.append("")
    lines.append("## composition checks")
    lines.append("  %-26s %s" % ("E0", E0s))
    lines.append("  %-26s %s" % ("dg_assign", {k: J[k]["args"].get("dg_assign") for k in ("prim", "anc", "r2b", "v36v", "v36l")}))
    lines.append("  %-26s %s" % ("tag", {k: J[k]["args"].get("tag") for k in ("prim", "anc", "r2b", "v36v", "v36l")}))
    for name, key in (("pre12.5 prim vs anc", "pre12.5_prim_vs_anc"), ("pre12.5 prim vs r2b", "pre12.5_prim_vs_r2b"), ("pre12.5 prim vs v36l", "pre12.5_prim_vs_v36l"),
                      ("epoch anc vs prim-json", "epoch_prim_json_vs_anc"), ("epoch anc vs v36l", "epoch_anc_vs_v36l"), ("r2b vs v36v 12.625..13.5", "r2b_vs_v36v_shared")):
        c = comp[name]
        lines.append("  %-26s n %3d  max |dE|/E0 %s  missing %s" % (key, c["n"], fnum(c["max_abs_dE_over_E0"], "%.2e"), c["missing"][:6]))
    lines.append("")
    lines.append("## the composed history from T/r_s %.3f (moves of E0; class by the Reorientation-5 criterion)" % 11.5)
    lines.append("T/rs | source | printable | E/E0 | dE | start | resolved (n) | unresolved: prim would-be (n) / overlap sw (n) / rec. reassign | remainder | pk | lvl | class | cause")
    for o in hist:
        if o["T"] < 11.5 - 1e-9:
            continue
        src = {"certified": "v36f", "anchor": "anc", "prim": "prim"}[o["seg"]]
        lines.append("%.3f | %s | %s | %.9f | %s | %s | %s (%d) | %s (%d) / %s (%d) / %s | %s%s | %d | %s/%s | %s | %s" % (
            o["T"], src, o["printable"], o["EE0"], fnum(o["dE"]), fnum(o["start"]), fnum(o["res_j"]), o["n_res_d"], fnum(o["unres_w"]), o["n_unres"], fnum(o["sw"]), o["n_sw"], fnum(o["rec"]),
            fnum(o["rem"] if o["rem"] is not None else o["unatt"]), "*" if o["rem_big"] else ("u" if o["unatt"] is not None else ""), o["pk"],
            o["lvl"] if o["lvl"] is not None else "-", o["nsub"] if o["nsub"] is not None else "-", o["cls"], o["cause"]))
    lines.append("  (* remainder above S2's threshold; u: v36f move with no bins, printed in the remainder column)")
    lines.append("")
    lines.append("## sums by segment (of E0)")
    for k, s in segs.items():
        lines.append("  %-18s n %3d  dE %s  resolved %s (%d diabatic of %d)  unresolved prim %s (%d)  overlap sw %s (%d)  rec. reassign %s  remainder %s  start %s  unattributed %s" % (
            k, s["n"], fnum(s["dE"]), fnum(s["res_j"]), s["n_res_d"], s["n_res"], fnum(s["unres_w"]), s["n_unres"], fnum(s["sw"]), s["n_sw"], fnum(s["rec"]), fnum(s["rem"]), fnum(s["start"]), fnum(s["unatt"])))
        lines.append("  %-18s printable yes %d no %d; classes %s; max |remainder| %s; max |closure| %s; max E_rec/prev mismatch %s; unconverged %s; E/E0 %s" % (
            "", s["p_yes"], s["p_no"], s["classes"], fnum(s["max_rem"], "%.2e"), fnum(s["max_closure"], "%.1e"), fnum(s["max_E_rec_mis"], "%.1e"), s["unconverged"], s["EE0_first_last"] and ["%.9f" % v for v in s["EE0_first_last"]]))
    lines.append("")
    lines.append("## crossings of the composed history's prim records (every one)")
    for o in hist:
        for c in o["cr"]:
            lines.append("  %.3f sub %s/%s ch %s %s->%s %s %s dec %s res %s falls %s gap %s jf %s jg %s alt %s %s" % (
                o["T"], c["sub"], c["nsub"], c["channel"], c["i"], c["n"], c["kind"], c["decision"], c["decided"], c["resolved"], c["falls"], fnum(c["gap_min"], "%.2e"), fnum(c["jf"]), fnum(c["jg"]), fnum(c["ja"]), c["cls"]))
    lines.append("")
    lines.append("## the rule dependence in the epoch: anchor (maximum overlap) against prim (R2' _r2b to 13.5, v36v at 13.625)")
    lines.append("T/rs | anchor E/E0 | conv | pk | prim E/E0 | src | prim - anchor | prim printable | lvl | conv | pk")
    for r in rd:
        lines.append("%.3f | %.9f | %s | %d | %s | %s | %s | %s | %s | %s | %s" % (
            r["T"], r["anc"], r["anc_conv"], r["anc_pk"], fnum(r.get("prim"), "%.9f"), r.get("src", "-"), fnum(r.get("diff")), r.get("prim_printable", "-"), r.get("lvl", "-"), r.get("prim_conv", "-"), r.get("prim_pk", "-")))
    s13 = [r for r in rd if r["T"] == 13.625][0]
    lines.append("  rule dependence at %.3f: prim %.7f - anchor %.7f = %+.2e of E0 (prim record printable %s, level %s; the anchor solve converged %s)" % (
        13.625, s13["prim"], s13["anc"], s13["diff"], s13["prim_printable"], s13["lvl"], s13["anc_conv"]))
    lines.append("")
    lines.append("## the maximum-overlap comparator (v36l), labelled as that convention's")
    lines.append("  the maximum-overlap convention (v36l, dg_assign anchor; the comparator, RUN_W2_v4 0.4); %d records to %.3f" % (len(Lrecs), Lrecs[-1]["T_over_rs"]))
    lines.append("  formation (first record with a trapping zero): %.3f (R_t/r_s %.4f, E/E0 %.8f)" % (fL["T_over_rs"], fL["R_t"] / RS, fL["E"] / E0))
    lines.append("  first episode %.3f .. %.3f; its close (first record without a zero) %s (min(N - |v|) there %s)" % (ivL[0][0], ivL[0][1], ivL[0][2], fnum(M["v36l"][ivL[0][2]]["min_N_minus_absv"], "%+.4f")))
    lines.append("  intervals with zeros: %s" % ["%.3f-%.3f" % (i[0], i[1]) for i in ivL])
    lines.append("  min(N - |v|): %s" % {str(t): fnum(M["v36l"][t]["min_N_minus_absv"], "%+.4f") for t in (15.75, 15.875, 16.0, 20.0)})
    lines.append("  for comparison, %s: formation %s, first episode %s, intervals %s" % ("the composed history (anchor epoch 12.625 to 14.5, prim from 14.625)", ivC[0][0], ivC[0], ["%.3f-%.3f" % (i[0], i[1]) for i in ivC]))
    lines.append("")
    lines.append("## the prim continuation against the v36l comparator from the shared 14.5 state (E/E0 only)")
    lines.append("  third consecutive clock within %.0e: %s; first clock off the floor: %s; max |prim - v36l| %s; last %s" % (FLOOR, rejoin, first_off, fnum(max(abs(x["d"]) for x in pv), "%.3e"), "%.3f %+.3e" % (pv[-1]["T"], pv[-1]["d"])))
    lines.append("  (clocks where the difference moves by more than the floor, or either run has pockets)")
    prev = 0.0
    for x in pv:
        mv = abs(x["d"] - prev) > FLOOR; prev = x["d"]
        if mv or x["pk_p"] or x["pk_l"]:
            lines.append("  %.3f prim - v36l %+.3e  pockets prim %d v36l %d  v36l switches %s" % (x["T"], x["d"], x["pk_p"], x["pk_l"], x["sw_l"]))
    mine_path = os.path.join(a.out_dir, "w2_check_v1_regenerated_table.out")
    with open(mine_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    theirs = open(DELIV_TABLE).read().splitlines()
    theirs_body = theirs[1:]     # line 1 carries their script md5 and timestamp
    nd = 0; first_diffs = []
    import difflib
    for i, (x, y) in enumerate(zip(lines, theirs_body)):
        if x != y:
            nd += 1
            if len(first_diffs) < 40:
                first_diffs.append((i + 2, y, x))
    P("  my lines %d, delivery lines (after line 1) %d; differing lines %d (beyond the per-line compare: length difference %d)" % (len(lines), len(theirs_body), nd, len(lines) - len(theirs_body)))
    for ln, y, x in first_diffs:
        P("   line %d\n     delivery: %s\n     mine    : %s" % (ln, y, x))
    with open(os.path.join(a.out_dir, "w2_check_v1_table.diff"), "w") as f:
        f.write("\n".join(difflib.unified_diff(theirs_body, lines, "delivery_table_from_line2", "regenerated", lineterm="")) + "\n")
    R["table_diff"] = {"n_lines_mine": len(lines), "n_lines_theirs": len(theirs_body), "differing": nd, "regenerated_table": mine_path}

    # ---------------- 13. the headline numbers as the delivery's md states them -----------------
    P("\n## 13. the delivery's headline numbers, recomputed")
    hd = {
        "prim n": len(prim), "prim printable": segs["prim"]["p_yes"], "prim dE": segs["prim"]["dE"], "prim n_cross": sum(len(o["cr"]) for o in prim),
        "prim n diabatic nonzero jump": segs["prim"]["n_res_d"], "prim n zero-jump": segs["prim"]["n_res"] - segs["prim"]["n_res_d"], "prim resolved jumps": segs["prim"]["res_j"],
        "prim max |rem|": segs["prim"]["max_rem"], "prim sum rem": segs["prim"]["rem"], "E/E0 at 40": H[40.0]["EE0"], "E/E0-1 at 40": H[40.0]["EE0"] - 1,
        "epoch n": segs["anchor"]["n"], "epoch n_switch": segs["anchor"]["n_sw"], "epoch sw": segs["anchor"]["sw"], "epoch rem": segs["anchor"]["rem"], "epoch dE": segs["anchor"]["dE"],
        "epoch rec 12.625": anc[0]["rec"], "rule dep 13.625": s13["diff"], "rule dep 12.625": rd[0]["diff"], "prim-v36l at 40": pv[-1]["d"], "prim-v36l max": max(abs(x["d"]) for x in pv),
        "formation": fL["T_over_rs"], "formation R_t/r_s": fL["R_t"] / RS, "first episode close": ivL[0][2], "close min(N-|v|)": M["v36l"][ivL[0][2]]["min_N_minus_absv"],
        "v36f 12.375 dE": H[12.375]["dE"], "v36f 12.5 dE": H[12.5]["dE"], "certified sum dE": segs["certified"]["dE"]}
    for k, v in hd.items():
        P("  %-30s %s" % (k, ("%+.6e" % v) if isinstance(v, float) else v))
    R["headline"] = hd
    R["seconds"] = time.time() - t0
    with open(os.path.join(a.out_dir, "w2_check_v1.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    P("\nwrote %s (%.1f s)" % (a.out_dir, R["seconds"]))
    out_txt.close()


if __name__ == "__main__":
    main()
