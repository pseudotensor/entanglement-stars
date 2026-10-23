#!/usr/bin/env python
"""comp5_tower_2rs_data_v1.py (solve lane, composition 5): the Page tower's 2 r_s table, data step. Box CPU only (nice 10); reads stored files only.

The quantity (PAGE v4.1 conventions, page_v4_data_v1b.py): x_kappa is the unweighted excess S_X(C) - S_X(P_-(h)) of one channel at one clock at the
2 r_s cut (the d13 'excess' under clocks[t].per_channel[kappa].cuts['2']); the weight is 16 kappa; the sum is S = sum_k 16 k x_k; F_meas per channel is
the post-pass's f_meas (w1_postpass_v1f.f_meas), weighted and summed. 2c at the 2 r_s cut is 64.25 (C11 reading, ReviewA_reading_web40j_C11_v1.md).

Inputs: the production k1 to 18 (page_history_v6o) and the tower finals (page_history_v6p, runs v6q_B2_k<k>to<k>) on the box; kappa <= 80 only.
Side data, never summed into the 54-channel table: the R2 k64 final mirrored to page_v6q/R2_k64 (found, not in the task's input list); the ruling2
probes k24, k32, k48 (v6o, lean, cut_norm off); the resume-mirror partial states of k19, 20, 44 to 47, 49 (not final, to their last mirrored clock).
Writes OUT/comp5_tower_2rs_v1.json and OUT/MANIFEST.md5. Usage: python comp5_tower_2rs_data_v1.py OUT_DIR"""
import glob
import hashlib
import json
import os
import sys

CODE_D13 = "<internal>/code_d13"
CODE_V6 = "<internal>/code_v6"
PP_DIR = "<internal>/w1"
os.environ.setdefault("W1_CODE_D13", CODE_D13)
os.environ.setdefault("W1_CODE_V6", CODE_V6)
for _p in (CODE_D13, CODE_V6, PP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import w1_postpass_v1f as PP  # noqa: E402

CUT = "2"
TWO_C = 64.25
KMAX = 80
PROD = ("<internal>/prod", "v6o_B2")
TF = "<internal>/tower_finals"
CARDS = ["bench1b", "pod6", "pod8b", "tourn2_r", "R1_r"]
K64 = ("<internal>/v6q_B2_k64to64", "v6q_B2_k64to64")
PROBES = {k: ("<internal>/v6o_B2r2b_k%d" % k, "v6o_B2r2b_k%d" % k) for k in (24, 32, 48)}
RM = "<internal>/resume_mirror"
PARTIAL_CARDS = {19: "R2", 20: "R2", 49: "R2", 64: "R2", 44: "tourn1b", 45: "tourn1b", 46: "tourn1b", 47: "tourn1b"}
ROWCHECK = ["E", "two_gE", "T_H", "R_t_record", "min_N_minus_absv", "join_max_dN", "join_max_rel_dv"]
SETTINGS_IGNORE = {"channels", "kappas", "kappa_hard_max", "out", "tag", "device", "threads", "max_wall_hours", "kappa_extend", "no_ksum"}
MANIFEST = []


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files_of(d, tag, k):
    return [os.path.join(d, "%s_k%d_%s" % (tag, k, s)) for s in ("summary.json", "d13.json")]


def card_md5(d):
    """the card-side md5 of the channel's files where the pull recorded one (MD5_card.txt beside the run dirs)."""
    out = {}
    mf = os.path.join(os.path.dirname(d), "MD5_card.txt")
    if os.path.exists(mf):
        for line in open(mf):
            parts = line.split()
            if len(parts) >= 2:
                out[os.path.basename(parts[-1])] = parts[0]
    return out


def read_channel(d, tag, k, role):
    ch = PP.load_channel(d, tag, k)
    if ch is None:
        return None
    S = json.load(open(files_of(d, tag, k)[0]))
    cm = card_md5(d)
    ent = {"kappa": k, "role": role, "dir": d, "tag": tag, "code": S.get("code"), "final": bool(S.get("final")), "settings": S.get("settings") or {},
           "history": {a: b for a, b in (S.get("history") or {}).items() if a != "label"}, "history_label": (S.get("history") or {}).get("label"),
           "propagation": (S.get("w1") or {}).get("propagation"), "interp": (S.get("w1") or {}).get("interp"), "interp_sub": (S.get("w1") or {}).get("interp_sub"),
           "n_records": len(S.get("records", [])), "files": []}
    for p in files_of(d, tag, k):
        m = md5(p)
        cmv = cm.get(os.path.basename(p))
        ent["files"].append({"path": p, "md5": m, "bytes": os.path.getsize(p), "card_md5": cmv, "card_equal": (cmv == m) if cmv else None})
        MANIFEST.append((m, p))
    clocks = sorted(ch["clocks"])
    ent["clocks"] = clocks
    ent["x"] = {}
    ent["fmeas"] = {}
    for t in clocks:
        c = ((ch["clocks"].get(t) or {}).get("cuts") or {}).get(CUT) or {}
        if c.get("excess") is not None:
            ent["x"][t] = float(c["excess"])
            fm = PP.f_meas(ch, t, CUT)
            ent["fmeas"][t] = None if fm is None else float(fm)
    ent["rows"] = {t: {q: ch["rows"][t].get(q) for q in ROWCHECK} for t in sorted(ch["rows"])}
    return ent


def settings_diff(a, b):
    keys = (set(a) | set(b)) - SETTINGS_IGNORE
    return {q: [str(a.get(q))[:90], str(b.get(q))[:90]] for q in sorted(keys) if a.get(q) != b.get(q)}


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    present = {}
    for k in range(1, 19):
        present[k] = read_channel(PROD[0], PROD[1], k, "production v6o (composed_B2/prod)")
    for card in CARDS:
        for d in sorted(glob.glob(os.path.join(TF, card, "v6q_B2_k*to*"))):
            tag = os.path.basename(d)
            k0, k1 = tag.split("_k")[1].split("to")
            if int(k0) != int(k1) or int(k0) > KMAX:
                continue
            k = int(k0)
            if k in present:
                raise SystemExit("duplicate channel %d" % k)
            present[k] = read_channel(d, tag, k, "tower final v6q (%s)" % card)
    side = {"k64_R2_final": read_channel(K64[0], K64[1], 64, "tower final v6q (R2), mirrored to page_v6q/R2_k64; not in the task's input list")}
    for k, (d, tag) in PROBES.items():
        side["probe_k%d" % k] = read_channel(d, tag, k, "ruling2 run (b) probe v6o, lean, cut_norm off")
    for k, card in PARTIAL_CARDS.items():
        dd = sorted(glob.glob(os.path.join(RM, card, "v6q_B2_k%dto%d" % (k, k), "k%d_T*" % k)))
        if dd:
            side["partial_k%d" % k] = read_channel(dd[-1], "v6q_B2_k%dto%d" % (k, k), k, "resume-mirror partial state (%s, %s), NOT final" % (card, os.path.basename(dd[-1])))
    # consistency: final flags, settings, history, the history-derived row columns against k1
    ref_o = present[1]
    ref_q = present[min(k for k in present if k > 18)]
    checks = {"non_final": [k for k, e in present.items() if not e["final"]], "settings_vs_ref": {}, "history_vs_k1": {}, "rows_vs_k1_maxabs": {},
              "codes": sorted(set(e["code"] for e in present.values())), "prod_vs_tower_settings": settings_diff(ref_o["settings"], ref_q["settings"]),
              "card_md5": {}}
    for name, e in list(present.items()) + list(side.items()):
        if e is None:
            continue
        ref = ref_o if e["code"] == "page_history_v6o.py" else ref_q
        sd = settings_diff(ref["settings"], e["settings"])
        if sd:
            checks["settings_vs_ref"][str(name)] = sd
        hd = {q: [ref_o["history"].get(q), e["history"].get(q)] for q in set(ref_o["history"]) | set(e["history"]) if ref_o["history"].get(q) != e["history"].get(q)}
        if hd or e["history_label"] != ref_o["history_label"]:
            checks["history_vs_k1"][str(name)] = {"fields": hd, "label_equal": e["history_label"] == ref_o["history_label"]}
        mx = {}
        for t, r in e["rows"].items():
            r0 = ref_o["rows"].get(t)
            if not r0:
                continue
            for q in ROWCHECK:
                if r.get(q) is not None and r0.get(q) is not None:
                    mx[q] = max(mx.get(q, 0.0), abs(float(r[q]) - float(r0[q])))
        checks["rows_vs_k1_maxabs"][str(name)] = mx
        checks["card_md5"][str(name)] = [f["card_equal"] for f in e["files"]]
    # the table: weighted terms at every clock
    ks = sorted(present)
    clocks = sorted(set.intersection(*[set(present[k]["x"]) for k in ks]))
    W = lambda k: 16.0 * k  # noqa: E731
    table = {}
    for k in range(1, KMAX + 1):
        if k in present:
            e = present[k]
            table[k] = {"status": "present", "x": [e["x"][t] for t in clocks], "wx": [W(k) * e["x"][t] for t in clocks],
                        "wF": [None if e["fmeas"][t] is None else W(k) * e["fmeas"][t] for t in clocks],
                        "resolved": [None if e["fmeas"][t] is None else abs(e["x"][t]) > e["fmeas"][t] for t in clocks]}
        else:
            table[k] = {"status": "missing"}
    sums = {"clocks": clocks, "S_1to18": [], "S_tower_present": [], "S_54": [], "F_54": [], "S_19to63_present": [], "S_64to80_present": []}
    for i, t in enumerate(clocks):
        sums["S_1to18"].append(sum(table[k]["wx"][i] for k in ks if k <= 18))
        sums["S_tower_present"].append(sum(table[k]["wx"][i] for k in ks if k > 18))
        sums["S_54"].append(sum(table[k]["wx"][i] for k in ks))
        sums["F_54"].append(sum(table[k]["wF"][i] for k in ks))
        sums["S_19to63_present"].append(sum(table[k]["wx"][i] for k in ks if 19 <= k <= 63))
        sums["S_64to80_present"].append(sum(table[k]["wx"][i] for k in ks if k >= 64))
    # the brackets for the missing channels from the present neighbours (weighted terms; never a value)
    missing = [k for k in range(1, KMAX + 1) if k not in present]

    def neigh(k, n):
        lo = [q for q in ks if q < k][-n:]
        hi = [q for q in ks if q > k][:n]
        return lo, hi
    br = {"A": {"definition": "per missing channel: [min, max] of the weighted terms 16 kappa x_kappa of the nearest present channel below and the nearest above, at the same clock"},
          "B": {"definition": "per missing channel: [min, max] over the two nearest present channels below and the two nearest above, at the same clock"}}
    br["C"] = {"definition": "per missing channel: [min (16 kappa x - 16 kappa F_meas), max (16 kappa x + 16 kappa F_meas)] over the two nearest present channels below and the two nearest above, at the same clock (bracket B widened by each neighbour's own propagation floor)"}
    for lab, n in (("A", 1), ("B", 2), ("C", 2)):
        wide = lab == "C"

        def lo_of(q, i):
            return table[q]["wx"][i] - (table[q]["wF"][i] if wide else 0.0)

        def hi_of(q, i):
            return table[q]["wx"][i] + (table[q]["wF"][i] if wide else 0.0)
        per = {}
        lo_sum = [0.0] * len(clocks)
        hi_sum = [0.0] * len(clocks)
        for k in missing:
            lo, hi = neigh(k, n)
            nb = lo + hi
            mn = [min(lo_of(q, i) for q in nb) for i in range(len(clocks))]
            mxv = [max(hi_of(q, i) for q in nb) for i in range(len(clocks))]
            per[k] = {"neighbours": nb, "lo": mn, "hi": mxv}
            lo_sum = [a + b for a, b in zip(lo_sum, mn)]
            hi_sum = [a + b for a, b in zip(hi_sum, mxv)]
        br[lab]["per_channel"] = per
        br[lab]["sum_lo"] = lo_sum
        br[lab]["sum_hi"] = hi_sum
        # leave-one-out test of the rule on the present channels that have present neighbours on both sides
        inside = 0
        tot = 0
        fails = []
        for k in ks:
            ks_wo = [q for q in ks if q != k]
            lo = [q for q in ks_wo if q < k][-n:]
            hi = [q for q in ks_wo if q > k][:n]
            if not lo or not hi:
                continue
            for i, t in enumerate(clocks):
                if t < 40.0:
                    continue
                v = table[k]["wx"][i]
                a = min(lo_of(q, i) for q in lo + hi)
                b = max(hi_of(q, i) for q in lo + hi)
                tot += 1
                if a <= v <= b:
                    inside += 1
                else:
                    fails.append([k, t])
        br[lab]["leave_one_out_from40"] = {"inside": inside, "total": tot, "fails_at_100": [f[0] for f in fails if f[1] == 100.0]}
    # side data placed against the brackets
    side_out = {}
    for name, e in side.items():
        if e is None:
            continue
        k = e["kappa"]
        rows = []
        for i, t in enumerate(clocks):
            if t in e["x"]:
                v = W(k) * e["x"][t]
                entry = {"t": t, "wx": v, "wF": None if e["fmeas"].get(t) is None else W(k) * e["fmeas"][t]}
                if k in br["A"]["per_channel"]:
                    entry["in_A"] = br["A"]["per_channel"][k]["lo"][i] <= v <= br["A"]["per_channel"][k]["hi"][i]
                    entry["in_B"] = br["B"]["per_channel"][k]["lo"][i] <= v <= br["B"]["per_channel"][k]["hi"][i]
                    entry["in_C"] = br["C"]["per_channel"][k]["lo"][i] <= v <= br["C"]["per_channel"][k]["hi"][i]
                if k in present:
                    entry["present_wx"] = table[k]["wx"][i]
                    entry["minus_present"] = v - table[k]["wx"][i]
                rows.append(entry)
        side_out[name] = {"kappa": k, "role": e["role"], "final": e["final"], "code": e["code"], "dir": e["dir"], "last_clock": max(e["x"]) if e["x"] else None,
                          "files": e["files"], "rows": rows}
    # the resume-mirror partial of k64 against the R2 final of k64 at the common clocks (is a mirrored partial the run's own numbers?)
    if side.get("k64_R2_final") and side.get("partial_k64"):
        a_, b_ = side["k64_R2_final"]["x"], side["partial_k64"]["x"]
        com = sorted(set(a_) & set(b_))
        checks["k64_partial_vs_final"] = {"n_clocks": len(com), "max_abs_dx": max(abs(a_[t] - b_[t]) for t in com) if com else None}
    meta = {k: {q: present[k][q] for q in ("role", "dir", "tag", "code", "final", "n_records", "propagation", "interp", "interp_sub", "files")} for k in ks}
    out = {"script": os.path.abspath(__file__), "cut": CUT, "two_c": TWO_C, "kmax": KMAX, "present": ks, "missing": missing,
           "missing_groups": {"R2 volume (stopped)": [19, 20, 49, 64], "tourn1b volume (stopped)": [44, 45, 46, 47],
                              "never run": [k for k in missing if k not in (19, 20, 49, 64, 44, 45, 46, 47)]},
           "clocks": clocks, "table": table, "sums": sums, "brackets": br, "side": side_out, "checks": checks, "meta": meta}
    jp = os.path.join(out_dir, "comp5_tower_2rs_v1.json")
    json.dump(out, open(jp, "w"), indent=1, default=float)
    with open(os.path.join(out_dir, "MANIFEST.md5"), "w") as f:
        for m, p in MANIFEST:
            f.write("%s  %s\n" % (m, p))
        f.write("%s  %s\n" % (md5(os.path.abspath(__file__)), os.path.abspath(__file__)))
        f.write("%s  %s\n" % (md5(jp), jp))
    print("present", len(ks), "missing", len(missing), "clocks", len(clocks))
    i100 = clocks.index(100.0)
    print("S_1to18 at 100", sums["S_1to18"][i100], "S_54", sums["S_54"][i100], "tower", sums["S_tower_present"][i100])
    print("non_final", checks["non_final"], "codes", checks["codes"])


if __name__ == "__main__":
    main()
