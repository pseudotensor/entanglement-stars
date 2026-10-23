#!/usr/bin/env python
"""cmp_rd2_scope_audit_v1.py (busi, analyst, 21 Oct): ruling v2 sec. 4 (a) (deliveries/lead_gaps/ReviewA_tolv2_v1.md), the key-difference
audit of the two identical-class scopes of cmp_rd2_ruling_v1.py: the FIRST scope (commit e9fbc9b0f2b: any key under /v36c/ matching
"/v36c/.*\\bset\\b" or starting "/v36c/set", the scope that wrote cmp_IC_ruling_firstscope.txt) against the NARROWED scope (ec05a1f0, RD-2 notes
sec. 8.2: /v36c/nk_set, /v36c/record_set, each solve's kept_round, kept_set, kept_sig, sets[#len]), and beside them the v2 scope of
cmp_rd2_ruling_v2.py. Over every compared record of one segment: each change arm against the default arm (the (A) pairs) and each arm against
the stored production (the (B) pairs). Per key pattern (indices as [*]): which scopes gate it, how many leaves exist, how many differ in each
pair; every key the narrowing DROPPED (first scope gated, narrowed not) and ADDED, each classified against the ruling's table. Read only.
usage: cmp_rd2_scope_audit_v1.py --label L --arm I=DIR:TAG --arm C=DIR:TAG [...] --prod-json J --t-min X --out JSON"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cmp_rd2_replay_v1 import leaves  # noqa: E402
import cmp_rd2_ruling_v1 as V1  # noqa: E402
import cmp_rd2_ruling_v2 as V2  # noqa: E402


def identical_key_first(k):
    """cmp_rd2_ruling_v1.identical_key as of e9fbc9b0f2b (the first scope), verbatim."""
    m = re.match(r"^/continuation/[^/]+/([^/\[#]+)$", k)
    if m:
        return m.group(1) in V1.CONT
    if re.match(r"^/(n_pocket|n_refill_R0)/[^/]+$", k):
        return True
    if k in ("/inversions_total", "/trapping_zeros#len", "/tangencies#len", "/late_surface/flag", "/converged", "/unconverged", "/gauge/label",
             "/regime/label", "/mask_equals_negative_set"):
        return True
    if k.startswith("/srf/") or k.startswith("/v36c/set") or re.search(r"/v36c/.*\bset\b", k):
        return True
    if k.startswith("/solve_v32/") and (k.endswith("/status") or k.endswith("meets_stop_rule") or k.endswith("/converged")):
        return True
    return False


def scope_v2(k):
    return V2.identical_key_v2(k) or bool(V2.RESUME_RE.match(k))


def table_class(p):
    """the ruling's sec. 2.3 table (with erratum v2 2(a)) for a key pattern."""
    if re.search(r"/(v36c/)?steps\[\*\]/|/line_search\[\*\]/", p):
        return "iterate trace (steps[*] / trials[*] / line_search[*], incl. plain_fallback trials): reported"
    if "/rounds[*]/" in p or p.endswith("/rounds#len"):
        return "set-resolution rounds (histories): reported"
    if "evaluation_sets" in p or "iteration_history" in p:
        return "per-evaluation trace: reported"
    if re.search(r"(previous_record_set|equals_previous_record)$", p):
        return "comparison with the previous record: compared where both sides carry a value (v2 2(a))"
    if re.search(r"/v36c/(nk_set|record_set)$|/v36c/(kept_round|kept_set|kept_sig)$|/v36c/sets", p):
        return "the v36c set entry: identical"
    return "OUTSIDE THE TRACES (identical class candidate)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--prod-json", required=True)
    ap.add_argument("--t-min", type=float, default=-1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        n, rest = s.split("=", 1); d, t = rest.rsplit(":", 1); arms.append((n, d, t))
    names = [n for n, _, _ in arms]; ref = names[0]
    J = {}
    for n, d, t in arms:
        J[n] = {round(r["T_over_rs"], 6): r for r in json.load(open(os.path.join(d, "collapse_%s.json" % t)))["records"] if r["T_over_rs"] > a.t_min + 1e-9}
    J["prod"] = {round(r["T_over_rs"], 6): r for r in json.load(open(a.prod_json))["records"] if r["T_over_rs"] > a.t_min + 1e-9}
    ts = sorted(set.intersection(*[set(J[n]) for n in names]))
    pairs = [(n, ref) for n in names[1:]] + [(n, "prod") for n in names]
    pat = {}   # pattern -> {"F","N","V2","leaves":{pair:n},"differ":{pair:n},"examples":[...]}
    nrec = {}
    for x, y in pairs:
        pk = "%s-%s" % (x, y)
        nrec[pk] = 0
        for t in ts:
            if t not in J[y]:
                continue
            nrec[pk] += 1
            La, Lb = dict(leaves(J[x][t])), dict(leaves(J[y][t]))
            for k in set(La) | set(Lb):
                xa, yb = La.get(k, "<absent>"), Lb.get(k, "<absent>")
                discrete = any(isinstance(v, (bool, str)) or v is None for v in (xa, yb)) or (isinstance(xa, int) and isinstance(yb, int))
                p = re.sub(r"\[\d+\]", "[*]", k)
                e = pat.setdefault(p, {"F": identical_key_first(k), "N": V1.identical_key(k), "V2": scope_v2(k), "leaves": {}, "differ": {}, "examples": []})
                e["leaves"][pk] = e["leaves"].get(pk, 0) + 1
                if discrete and xa != yb:
                    e["differ"][pk] = e["differ"].get(pk, 0) + 1
                    if len(e["examples"]) < 6:
                        e["examples"].append([pk, t, k, str(xa)[:40], str(yb)[:40]])
    out = {"label": a.label, "records": ts, "pairs": nrec, "dropped": {}, "added": {}, "v2_added_over_narrowed": {}, "v2_dropped_from_narrowed": {},
           "ungated_differing": {}}
    for p, e in sorted(pat.items()):
        row = {"class": table_class(p), "leaves": e["leaves"], "differ": e["differ"], "examples": e["examples"]}
        if e["F"] and not e["N"]:
            out["dropped"][p] = row
        if e["N"] and not e["F"]:
            out["added"][p] = row
        if e["V2"] and not e["N"]:
            out["v2_added_over_narrowed"][p] = row
        if e["N"] and not e["V2"]:
            out["v2_dropped_from_narrowed"][p] = row
        if not (e["F"] or e["N"] or e["V2"]) and e["differ"]:
            out["ungated_differing"][p] = row
    outside = [p for p, r in out["dropped"].items() if r["class"].startswith("OUTSIDE")]
    out["dropped_outside_traces"] = outside
    print("scope audit %s: records %s; pairs (records compared) %s" % (a.label, ts, nrec))
    for sec, title in (("dropped", "DROPPED by the narrowing (first scope gated, narrowed not)"), ("added", "ADDED by the narrowing (narrowed gated, first not)"),
                       ("v2_added_over_narrowed", "v2 gates, narrowed v1 does not"), ("v2_dropped_from_narrowed", "narrowed v1 gates, v2 does not"),
                       ("ungated_differing", "gated by no scope and differing in some pair (reported keys)")):
        print("-- %s: %d patterns" % (title, len(out[sec])))
        for p, r in out[sec].items():
            nl = sum(r["leaves"].values()); nd = sum(r["differ"].values())
            print("   %s  [%s]  leaves %d, differing %d %s%s" % (p, r["class"], nl, nd, json.dumps(r["differ"]) if nd else "",
                                                             ("  e.g. " + "; ".join("%s %.4g %s: %s vs %s" % tuple(x) for x in r["examples"][:2])) if nd else ""))
    print("-- dropped keys outside the traces: %s" % (outside or "none"))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
