#!/usr/bin/env python
"""cmp_rd2_ruling_v2.py (busi, analyst, 21 Oct): the RD-2 / RD-2b replays judged by REVIEWA_perf_tolerance_ruling_v1 sec. 4 with its sec. 2.3
table as corrected by ruling v2 (deliveries/lead_gaps/ReviewA_tolv2_v1.md, erratum v2 secs. 2 (a) to (c)). cmp_rd2_ruling_v1.py and its
outputs are unchanged; this file imports v1's table constants and its identical_key (the narrowed scope of RD-2 notes sec. 8.2) and adds:

2(a) the identical class: v1's list (the continuation entries, n_pocket, n_refill_R0, inversions_total, the trapping-zero and tangency counts,
     late_surface flag, converged/unconverged, gauge and regime labels, srf decisions, each solve's status / meets_stop_rule / converged, R_sh,
     the spectra masks) with the v36c set entry as erratum v2 lists it: /v36c/nk_set, /v36c/record_set, each solve's kept_round, kept_set,
     kept_sig and sets (every leaf under /solve_v32/solves[i]/v36c/sets; v1 gated only sets#len). previous_record_set and
     equals_previous_record (and the solve-level kept_set_equals_previous_record) are compared where both sides carry a value (neither None
     nor absent) and reported otherwise (a resume's first record). Every entry under steps[*] / trials[*] / line_search[*], the rounds and
     histories, evaluation_sets and node residuals are reported (they are not in the class).
2(b) at a record where the Newton iteration counts of the change arm, the default arm and the stored production (where it has the record)
     are all equal, every v1 line holds as written; fields 1e-7.
2(c) stop flip: the first record where those counts are not all equal, and every later record, keep E/E0 and E rel 3.0e-8, roots 3.4e-7 a,
     depth 4.6e-9, R_sh, the identical class and the masks as (A) lines. The fields (1e-7), named observables, source/handoff rows (1e-6),
     spectra e rows (1e-8) and the end slice's densities and applied fields there are decided by:
       (1) each arm against the stored production within C (E/E0 3.0e-7, roots 3.4e-6 a, depth 4.6e-8, fields 2.5e-5), identical class 0
           and masks equal;
       (2) the change arm's distance from the production at most twice the default arm's, per class (fields max |d|; named: the largest
           |d| / (1e-6 |x_prod| + 1e-8); rows max |d|; spectra max |d e| / max |e|), at each post-flip record;
       (3) the per-call effect at the flipped record's stored inputs below the table's line (fields 1e-7, E 3.0e-8), read from the field
           gate's output (--percall ARM=gate:JSON, the B1 arm "Ce against P12" at the flipped clock) or, where the change route was not used at
           the flipped record (every solve a logged fallback to the default's ivp route), from the arm's own fields_rd2 counters
           (--percall ARM=route-identity: the per-call effect is 0 by construction);
       (4) the three counts printed. A post-flip record the production lacks is INCONCLUSIVE for these classes.
     The end slice: masks and f_k identical always. After a flip its e_k and applied fields are checked bitwise equal to the arm's own record
     rows at the slice clock (then the record's three-part test at that clock decides them); its sea / R2 densities have no stored production
     slice to be read against (the production writes slices every 2.0 r_s), so (2) is read through the density-built handoff rows eps_sea and
     eps_R2 at the slice clock (stated in the output as that reading).
The flip rule and the (2) rule are tolcmp_v2.py's flip_records() and three_way() (busi/code/readings/perf_tolerance_v1/, written by the RD-1
entry builder), imported from --shared (its md5 is printed), so the two comparators share them.
(B) as in v1: each arm against the stored production within C at every record it has, set decisions identical (v2 class), masks equal; the
v1 same-class test reported.
usage: cmp_rd2_ruling_v2.py --arm I=DIR:TAG --arm C=DIR:TAG [--arm W=DIR:TAG] --rs RS --prod-json J --prod-dir D --prod-tag T --t-min X
       [--slice I=PATH --slice C=PATH ...] [--percall C=gate:JSON | C=route-identity ...] [--shared PATH] [--gate-extra REGEX] --out JSON"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cmp_rd2_replay_v1 import cmp_record, rows_by_T  # noqa: E402
import cmp_rd2_ruling_v1 as V1  # noqa: E402  (unchanged; its constants and its identical_key)

GATE = dict(V1.GATE)          # E/E0 3.0e-8, E 3.0e-8, root 3.4e-7, min 4.6e-9, fields 1e-7, rows 1e-6, spectra 1e-8
FLOORB = dict(V1.FLOORB)      # C: E/E0 3.0e-7, root 3.4e-6, min 4.6e-8, fields 2.5e-5
NAMED, FIELDS, HROWS = V1.NAMED, V1.FIELDS, V1.HROWS
RESUME_RE = re.compile(r"^(/v36c/(previous_record_set|equals_previous_record)|/solve_v32/solves\[\d+\]/v36c/kept_set_equals_previous_record)$")
SETS_RE = re.compile(r"^/solve_v32/solves\[\d+\]/v36c/sets(\[\d+\]/[^/\[]+)?(#len)?$")
EXTRA = None   # --gate-extra (sec. 4 (a) of ruling v2: a dropped key outside the traces is re-gated)


def identical_key_v2(k):
    if V1.identical_key(k) or SETS_RE.match(k):
        return True
    if EXTRA is not None and EXTRA.search(k):
        return True
    return False


def load_shared(path):
    spec = importlib.util.spec_from_file_location("tolcmp_v2_shared", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, hashlib.md5(open(path, "rb").read()).hexdigest()


def row(Z, name, t, rs):
    ix = rows_by_T(Z, rs)
    if t not in ix or name not in Z.files:
        return None
    return np.asarray(Z[name][ix[t]], float)


def metrics(x, y, t, J, Z, rs):
    """every class distance of arm x against arm (or production) y at record t."""
    fl, disc = cmp_record(J[x][t], J[y][t])
    m = {"E_over_E0": fl.get("/E_over_E0", (0.0, 0.0))[0], "E_rel": fl.get("/E", (0.0, 0.0))[1],
         "root": max([fl[k][0] for k in fl if k in ("/R_t", "/r_in", "/R_c") or k.startswith("/trapping_zeros[") or k.startswith("/threshold_radius/")] or [0.0]),
         "min": fl.get("/min_N_minus_absv", (0.0, 0.0))[0], "R_sh": fl.get("/R_sh", (0.0, 0.0))[0],
         "argmin": fl.get("/r_min_N_minus_absv", (0.0, 0.0))[0]}
    ident, resume, resume_gated, other, ib = [], [], [], [], []
    for k, a_, b_ in disc:
        if RESUME_RE.match(k):
            if a_ in (None, "<absent>") or b_ in (None, "<absent>"):
                resume.append((k, a_, b_))
            else:
                ident.append((k, a_, b_)); resume_gated.append(k)
        elif identical_key_v2(k):
            ident.append((k, a_, b_))
        elif k.endswith("indicator_pass"):
            ib.append((k, a_, b_))
        else:
            other.append((k, a_, b_))
    m["ident"], m["resume"], m["other"] = ident, resume, other
    m["indicator_hard"] = None
    if ib:
        ba = (J[x][t].get("indicator") or {}).get("B_estimate"); bb = (J[y][t].get("indicator") or {}).get("B_estimate")
        m["indicator_hard"] = any(v is None or v < 1e-10 or v > 1e-8 for v in (ba, bb))
        m["indicator"] = (ib[0][1], ib[0][2], ba, bb)
    # named: the largest |d| / (1e-6 |x_y| + 1e-8) (the allowance on the second operand, the default arm or the production)
    nm = {}; bad_nonnum = []
    for k_ in NAMED:
        a_, b_ = J[x][t].get(k_), J[y][t].get(k_)
        if isinstance(a_, (int, float)) and isinstance(b_, (int, float)) and not isinstance(a_, bool):
            nm[k_] = (abs(a_ - b_), abs(a_ - b_) / (1e-6 * abs(b_) + 1e-8))
        elif a_ != b_:
            bad_nonnum.append((k_, a_, b_))
    m["named_keys"] = nm
    m["named"] = max([v[1] for v in nm.values()] or [0.0])
    m["named_nonnum"] = bad_nonnum
    fmax = 0.0; parts = {}
    for f in FIELDS:
        ra, rb = row(Z[x]["fields"], f, t, rs), row(Z[y]["fields"], f, t, rs)
        if ra is None or rb is None:
            continue
        d = float(np.max(np.abs(ra - rb))); fmax = max(fmax, d); parts[f] = d
    m["fields"], m["fields_parts"] = fmax, parts
    rmax = 0.0; rk = None; rparts = {}
    for src, names in (("fields", [n for n in Z[x]["fields"].files if n.startswith("eps_") or n.startswith("J_")]), ("handoff", HROWS)):
        for n in names:
            ra, rb = row(Z[x][src], n, t, rs), row(Z[y][src], n, t, rs)
            if ra is None or rb is None:
                continue
            d = float(np.max(np.abs(ra - rb))); rparts[n] = d
            if d > rmax:
                rmax, rk = d, n
    m["rows"], m["rows_key"], m["rows_parts"] = rmax, rk, rparts
    sa = rows_by_T(Z[x]["spectra"], rs); sb = rows_by_T(Z[y]["spectra"], rs)
    ks = [k for k in Z[x]["spectra"].files if k.startswith("mask_")]
    m["masks_equal"] = bool(all(np.array_equal(Z[x]["spectra"][k][sa[t]], Z[y]["spectra"][k][sb[t]]) for k in ks))
    m["spectra"] = max(float(np.max(np.abs(Z[x]["spectra"][k.replace("mask", "e")][sa[t]] - Z[y]["spectra"][k.replace("mask", "e")][sb[t]])))
                       / max(float(np.max(np.abs(Z[y]["spectra"][k.replace("mask", "e")][sb[t]]))), 1e-300) for k in ks)
    return m


def fmt_disc(lst, n=6):
    return ("; ".join("%s: %r vs %r" % s for s in lst[:n])) if lst else ""


def kept_lines(m):
    """the lines that hold at every record, before and after a flip: E/E0, E rel, roots, depth, R_sh, identical class, masks, indicator."""
    L = [("E/E0", m["E_over_E0"], GATE["E_over_E0"], m["E_over_E0"] <= GATE["E_over_E0"]),
         ("E rel", m["E_rel"], GATE["E"], m["E_rel"] <= GATE["E"]),
         ("roots", m["root"], GATE["root"], m["root"] <= GATE["root"]),
         ("depth", m["min"], GATE["min"], m["min"] <= GATE["min"]),
         ("R_sh", m["R_sh"], 0.0, m["R_sh"] == 0.0),
         ("identical class", len(m["ident"]), 0, not m["ident"]),
         ("spectra masks", "equal" if m["masks_equal"] else "DIFFER", "equal", m["masks_equal"])]
    if m["indicator_hard"] is not None:
        L.append(("indicator_pass flip", "%r vs %r (B %s vs %s)" % m["indicator"], "inside a decade of 1e-9", not m["indicator_hard"]))
    return L


def field_lines(m):
    """the table's field-derived lines (2(b)); after a flip they are reported and the three-part test decides."""
    return [("fields", m["fields"], GATE["fields"], m["fields"] <= GATE["fields"]),
            ("named (largest d / allowance)", m["named"], 1.0, m["named"] <= 1.0 and not m["named_nonnum"]),
            ("rows", m["rows"], GATE["rows"], m["rows"] <= GATE["rows"]),
            ("spectra e rel", m["spectra"], GATE["spectra"], m["spectra"] <= GATE["spectra"])]


def floor_lines(m):
    return [("E/E0", m["E_over_E0"], FLOORB["E_over_E0"], m["E_over_E0"] <= FLOORB["E_over_E0"]),
            ("roots", m["root"], FLOORB["root"], m["root"] <= FLOORB["root"]),
            ("depth", m["min"], FLOORB["min"], m["min"] <= FLOORB["min"]),
            ("fields", m["fields"], FLOORB["fields"], m["fields"] <= FLOORB["fields"]),
            ("R_sh", m["R_sh"], 0.0, m["R_sh"] == 0.0),
            ("identical class", len(m["ident"]), 0, not m["ident"]),
            ("spectra masks", "equal" if m["masks_equal"] else "DIFFER", "equal", m["masks_equal"])]


def pline(name, val, lim, ok):
    v = ("%.2e" % val) if isinstance(val, float) else str(val)
    l_ = ("%.1e" % lim) if isinstance(lim, float) else str(lim)
    return "%s %s (line %s) %s" % (name, v, l_, "ok" if ok else "BREACH")


def percall_read(spec, arm, t_flip, J, rs, post_ts):
    """2(c)(3) at the flipped record t_flip. spec 'gate:PATH' or 'route-identity'."""
    if spec is None:
        return {"ok": None, "text": "not supplied (2(c)(3) OPEN)"}
    if spec == "route-identity":
        fr = J[arm][t_flip].get("fields_rd2") or {}
        sp = fr.get("since_previous_record") or {}
        ok = sp.get("collocation") == 0 and (sp.get("fallback") or 0) > 0 and (sp.get("ivp_by_tolerance") or 0) == 0
        return {"ok": bool(ok), "fields": 0.0 if ok else None, "E": 0.0 if ok else None,
                "text": "route identity at %.6f: collocation %s, fallbacks %s (%s), ivp by tolerance %s: %s" % (
                    t_flip, sp.get("collocation"), sp.get("fallback"), fr.get("fallback_reasons_total"), sp.get("ivp_by_tolerance"),
                    "every solve on the default's ivp route, per-call effect 0 by construction" if ok else "the change route ran: route identity does not hold")}
    if spec.startswith("gate:"):
        G = json.load(open(spec[5:]))
        recs = {round(float(r["T"]) / rs, 6): r for r in G["records"]}
        if t_flip not in recs:
            return {"ok": None, "text": "the field gate output has no record at %.6f (2(c)(3) OPEN)" % t_flip}

        def one(r):
            b = r["B1"]
            return max(b["field"], b.get("N_avg", 0.0), b.get("v_avg", 0.0)), b["E_rel"]
        f0, e0 = one(recs[t_flip])
        rest = [one(recs[t]) for t in post_ts if t in recs]
        b2 = recs[t_flip].get("B2", {})
        ok = f0 <= GATE["fields"] and e0 <= GATE["E"]
        return {"ok": bool(ok), "fields": f0, "E": e0, "route": recs[t_flip].get("route_Ce"), "fallback": recs[t_flip].get("fallback_Ce"),
                "post_flip_max": (max([r[0] for r in rest] or [0.0]), max([r[1] for r in rest] or [0.0]), len(rest)),
                "text": "field gate %s at %.6f (record index %s, the stored sources): change route %s (fallbacks %s) against the default route P12: "
                        "fields %.2e (line 1e-7), E rel %.2e (line 3.0e-8): %s; against P14 fields %.2e, E %.2e; over the %d post-flip records of the "
                        "file: fields at most %.2e, E at most %.2e" % (
                            os.path.basename(spec[5:]), t_flip, recs[t_flip].get("i"), recs[t_flip].get("route_Ce"), recs[t_flip].get("fallback_Ce"),
                            f0, e0, "below the line" if ok else "ABOVE THE LINE", b2.get("field", float("nan")), b2.get("E_rel", float("nan")), len(rest),
                            max([r[0] for r in rest] or [0.0]), max([r[1] for r in rest] or [0.0]))}
    raise SystemExit("unknown --percall spec %r" % spec)


def main():
    global EXTRA
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--prod-json", required=True); ap.add_argument("--prod-dir", required=True); ap.add_argument("--prod-tag", required=True)
    ap.add_argument("--t-min", type=float, default=-1.0)
    ap.add_argument("--slice", action="append", default=[], help="NAME=PATH: the arm's end slice")
    ap.add_argument("--percall", action="append", default=[], help="NAME=gate:JSON or NAME=route-identity (2(c)(3))")
    ap.add_argument("--shared", default=os.path.join(HERE, "..", "readings", "perf_tolerance_v1", "tolcmp_v2.py"))
    ap.add_argument("--gate-extra", default=None, help="REGEX: keys added to the identical class (ruling v2 sec. 4 (a) re-gating)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gate_extra:
        EXTRA = re.compile(a.gate_extra)
    SH, sh_md5 = load_shared(a.shared)
    flip_records, three_way = SH.flip_records, SH.three_way
    arms = []
    for s in a.arm:
        n, rest = s.split("=", 1); d, t = rest.rsplit(":", 1); arms.append((n, d, t))
    names = [n for n, _, _ in arms]
    ref = names[0]
    pc = dict(x.split("=", 1) for x in a.percall)
    J = {}; Z = {}
    for n, d, t in arms:
        J[n] = {round(r["T_over_rs"], 6): r for r in json.load(open(os.path.join(d, "collapse_%s.json" % t)))["records"] if r["T_over_rs"] > a.t_min + 1e-9}
        Z[n] = {k: np.load(os.path.join(d, "collapse_%s_%s.npz" % (t, k)), allow_pickle=True) for k in ("fields", "spectra", "handoff")}
    J["prod"] = {round(r["T_over_rs"], 6): r for r in json.load(open(a.prod_json))["records"] if r["T_over_rs"] > a.t_min + 1e-9}
    Z["prod"] = {k: np.load(os.path.join(a.prod_dir, "collapse_%s_%s.npz" % (a.prod_tag, k)), allow_pickle=True) for k in ("fields", "spectra", "handoff")}
    ts = sorted(set.intersection(*[set(J[n]) for n in names]))
    seteq = all(sorted(J[n]) == sorted(J[ref]) for n in names)
    rep = {"label": a.label, "shared": {"path": os.path.abspath(a.shared), "md5": sh_md5, "functions": ["flip_records", "three_way"]},
           "gate_extra": a.gate_extra, "records": ts, "record_sets_equal": seteq, "pairs": {}, "B": {}, "verdict": {}}
    print("ruling v2 gate (ReviewA_tolv2_v1 erratum v2 to REVIEWA_perf_tolerance_ruling_v1 secs. 2.3 and 4) %s" % a.label)
    print("records %s; record sets equal %s; shared flip/three-way rule: %s md5 %s%s" % (ts, seteq, os.path.basename(a.shared), sh_md5,
                                                                                    ("; extra gated keys /%s/" % a.gate_extra) if a.gate_extra else ""))
    M = {}  # (x, y, t) -> metrics
    for t in ts:
        for n in names[1:]:
            M[(n, ref, t)] = metrics(n, ref, t, J, Z, a.rs)
        if t in J["prod"]:
            for n in names:
                M[(n, "prod", t)] = metrics(n, "prod", t, J, Z, a.rs)
    # the counts (2(c)(4)) and (B)
    print("Newton iteration counts per record (%s, production): %s" % (", ".join(names), "; ".join(
        "%.4g: %s / %s" % (t, "/".join(str(J[n][t].get("iterations")) for n in names), J["prod"][t].get("iterations") if t in J["prod"] else "-") for t in ts)))
    for n in names:
        okB = True; lines = {}
        for t in ts:
            if t not in J["prod"]:
                continue
            m = M[(n, "prod", t)]
            L = floor_lines(m)
            ok = all(x[3] for x in L); okB &= ok
            lines["%.6f" % t] = {"ok": ok, "lines": [pline(*x) for x in L], "resume": m["resume"], "ident": m["ident"], "other_n": len(m["other"]),
                                 "nums": {k: m[k] for k in ("E_over_E0", "root", "min", "fields", "named", "rows", "spectra")}}
        rep["B"][n] = {"ok": bool(okB), "records": lines}
    for n in names[1:]:
        counts = {t: (J[n][t].get("iterations"), J[ref][t].get("iterations"), J["prod"][t].get("iterations") if t in J["prod"] else None) for t in ts}
        first, post = flip_records(counts)
        print("=" * 100)
        print("(A) %s against %s; first stop flip: %s%s" % (n, ref, first, (" (counts %s / %s / %s)" % counts[first]) if first is not None else " (none: 2(b), v1's lines everywhere)"))
        okA = True; incon = []; recs = {}
        pcr = None
        if first is not None:
            post_ts = [t for t in ts if post[t]]
            pcr = percall_read(pc.get(n), n, first, J, a.rs, post_ts)
            print("  2(c)(3) per-call at %.6f: %s" % (first, pcr["text"]))
            if pcr["ok"] is None:
                incon.append("2(c)(3) per-call evidence at %.6f: %s" % (first, pcr["text"]))
            elif not pcr["ok"]:
                okA = False
        for t in ts:
            m = M[(n, ref, t)]
            r = {"counts": counts[t], "post_flip": post[t]}
            K = kept_lines(m); Fl = field_lines(m)
            ok_t = all(x[3] for x in K)
            print("== T/rs %.6f  counts %s/%s/%s  %s" % ((t,) + counts[t] + ("POST-FLIP" if post[t] else "equal counts",)))
            print("   (A) kept lines: " + "; ".join(pline(*x) for x in K))
            if m["ident"]:
                print("       identical-class differences: " + fmt_disc(m["ident"]))
            if not post[t]:
                ok_t &= all(x[3] for x in Fl)
                print("   (A) field-derived lines (2(b)): " + "; ".join(pline(*x) for x in Fl))
            else:
                print("   (A) field-derived lines (reported; 2(c) decides): " + "; ".join(pline(*x) for x in Fl))
                if t not in J["prod"]:
                    incon.append("post-flip record %.6f has no production record" % t)
                    print("   2(c): INCONCLUSIVE (no production record at this clock)")
                else:
                    mx, md = M[(n, "prod", t)], M[(ref, "prod", t)]
                    one = []
                    for arm_, mm in ((n, mx), (ref, md)):
                        L1 = floor_lines(mm)
                        one.append((arm_, all(x[3] for x in L1), L1))
                    two = []
                    for c in ("fields", "named", "rows", "spectra"):
                        okc, ratio = three_way(mx[c], md[c], None)
                        two.append((c, mx[c], md[c], ratio, okc))
                    ok1 = all(o[1] for o in one); ok2 = all(x[4] for x in two)
                    ok_t &= ok1 and ok2
                    for arm_, o, L1 in one:
                        print("   2(c)(1) %s against the production (C): %s: %s" % (arm_, "PASS" if o else "FAIL", "; ".join(pline(*x) for x in L1)))
                    print("   2(c)(2) %s's distance from the production / %s's (at most 2): %s: %s" % (n, ref, "PASS" if ok2 else "FAIL", "; ".join(
                        "%s %.2e / %.2e = %.2f%s" % (c, xv, dv, ratio, "" if okc else " BREACH") for c, xv, dv, ratio, okc in two)))
                    # per named key (reported)
                    pk = []
                    for k_ in NAMED:
                        if k_ in mx["named_keys"] and k_ in md["named_keys"]:
                            xv, dv = mx["named_keys"][k_][0], md["named_keys"][k_][0]
                            pk.append((k_, xv, dv, (xv / dv) if dv > 0 else (0.0 if xv == 0 else float("inf"))))
                    worst = sorted(pk, key=lambda q: -q[3])[:3]
                    print("       named per key, largest ratios (reported): " + "; ".join("%s %.2e / %.2e = %.2f" % q for q in worst))
                    r["three_part"] = {"one": {o[0]: o[1] for o in one}, "two": {c: {"x": xv, "d": dv, "ratio": ratio, "ok": okc} for c, xv, dv, ratio, okc in two},
                                       "named_per_key": {q[0]: [q[1], q[2], q[3]] for q in pk}}
            if m["resume"]:
                print("   reported resume labels (one side without a value): " + fmt_disc(m["resume"]))
            print("   reported other discrete differences: %d%s" % (len(m["other"]), (" " + fmt_disc(m["other"], 4)) if m["other"] else ""))
            print("   reported: argmin r_min_N_minus_absv %.2e; rows max at %s" % (m["argmin"], m["rows_key"]))
            print("   record %s" % ("PASS" if ok_t else "FAIL"))
            okA &= ok_t
            r.update({"ok": bool(ok_t), "kept": [pline(*x) for x in K], "field": [pline(*x) for x in Fl], "ident": m["ident"], "resume": m["resume"],
                      "other": m["other"][:50], "other_n": len(m["other"]),
                      "nums": {k: m[k] for k in ("E_over_E0", "E_rel", "root", "min", "R_sh", "fields", "named", "rows", "spectra")}})
            recs["%.6f" % t] = r
        # segment conditions
        dA = [M[(n, ref, t)]["E_over_E0"] for t in ts]
        k = max(1, len(ts) // 4)
        g = (float(np.median(dA[-k:])), float(np.median(dA[:k])))
        nogrowth = (g[0] <= 10.0 * g[1]) if g[1] > 0 else (g[0] == 0.0)
        cond8 = len(ts) >= 8
        setchange = any(J[ref][t].get("inversions_total") != J[ref][ts[i - 1]].get("inversions_total") or J[ref][t].get("n_pocket") != J[ref][ts[i - 1]].get("n_pocket")
                        for i, t in enumerate(ts) if i > 0)
        # the end slice
        sl = dict(x.split("=", 1) for x in a.slice); slr = None
        if ref in sl and n in sl:
            S0 = np.load(sl[ref], allow_pickle=False); S1 = np.load(sl[n], allow_pickle=False)
            tsl = round(float(S0["T"]) / a.rs, 6)
            kk = sorted(int(q.split("_")[1]) for q in S0.files if q.startswith("mask_"))
            slr = {"clock": tsl, "masks_equal": True, "f_identical": True, "e_rel": 0.0, "dens_sea": 0.0, "dens_R2": 0.0, "dens_dust": 0.0,
                   "applied": max(float(np.max(np.abs(S1["N_applied"] - S0["N_applied"]))), float(np.max(np.abs(S1["v_applied"] - S0["v_applied"]))))}
            for q in kk:
                slr["masks_equal"] &= bool(np.array_equal(S1["mask_%d" % q], S0["mask_%d" % q]))
                slr["f_identical"] &= bool(S1["f_%d" % q].shape == S0["f_%d" % q].shape and S1["f_%d" % q].tobytes() == S0["f_%d" % q].tobytes())
                e0 = S0["e_%d" % q]; slr["e_rel"] = max(slr["e_rel"], float(np.max(np.abs(S1["e_%d" % q] - e0))) / max(float(np.max(np.abs(e0))), 1e-300))
                for nm_, key in (("dens_sea", "sea_%d"), ("dens_R2", "R2_%d"), ("dens_dust", "dust_%d")):
                    A_, B_ = S1[key % q], S0[key % q]
                    if A_.shape != B_.shape:
                        slr["masks_equal"] = False; continue
                    if A_.size:
                        slr[nm_] = max(slr[nm_], float(np.max(np.abs(np.sum(np.abs(A_) ** 2, axis=1) - np.sum(np.abs(B_) ** 2, axis=1)))))
            dec_ok = slr["masks_equal"] and slr["f_identical"]
            print("end slice %s vs %s at %.6f: masks %s, f_k %s; e rel %.2e (1e-8), sea %.2e, R2 %.2e (1e-9), dust %.2e (reported), applied fields %.2e (1e-7)" % (
                n, ref, tsl, "equal" if slr["masks_equal"] else "DIFFER", "identical" if slr["f_identical"] else "DIFFER", slr["e_rel"], slr["dens_sea"],
                slr["dens_R2"], slr["dens_dust"], slr["applied"]))
            if tsl in post and post[tsl]:
                # the slice's e_k and applied fields equal the arm's record rows at the slice clock (then the record's 2(c) decides them)
                ident_rows = {}
                for arm_, S_ in ((n, S1), (ref, S0)):
                    ix = rows_by_T(Z[arm_]["fields"], a.rs)[tsl]; jx = rows_by_T(Z[arm_]["spectra"], a.rs)[tsl]
                    ident_rows[arm_] = bool(np.array_equal(S_["N_applied"], Z[arm_]["fields"]["N_applied"][ix]) and np.array_equal(S_["v_applied"], Z[arm_]["fields"]["v_applied"][ix])
                                            and all(np.array_equal(S_["e_%d" % q], Z[arm_]["spectra"]["e_%d" % q][jx]) for q in kk))
                rec_ok = recs.get("%.6f" % tsl, {}).get("ok")
                dens = None
                if tsl in J["prod"]:
                    mx, md = M[(n, "prod", tsl)], M[(ref, "prod", tsl)]
                    dx = max(mx["rows_parts"].get("eps_sea", 0.0), mx["rows_parts"].get("eps_R2", 0.0))
                    dd = max(md["rows_parts"].get("eps_sea", 0.0), md["rows_parts"].get("eps_R2", 0.0))
                    okd, ratio = three_way(dx, dd, None)
                    dens = {"x": dx, "d": dd, "ratio": ratio, "ok": bool(okd)}
                slr.update({"post_flip": True, "rows_identity": ident_rows, "record_at_clock_ok": rec_ok, "densities_2_via_eps_rows": dens,
                            "production_slice": "none stored at this clock"})
                s_ok = dec_ok and all(ident_rows.values()) and bool(rec_ok) and (dens is not None and dens["ok"])
                print("   post-flip slice (2(c)): decisions %s; e_k and applied fields bitwise the arms' record rows at %.6f: %s (so the record's three-part test "
                      "decides them: record %s); densities: no production slice is stored at this clock, (2) read through the density-built handoff rows "
                      "eps_sea, eps_R2: %s; (3) as the record's" % (
                          "identical" if dec_ok else "DIFFER", tsl, ident_rows, "PASS" if rec_ok else "FAIL",
                          ("%.2e / %.2e = %.2f %s" % (dens["x"], dens["d"], dens["ratio"], "ok" if dens["ok"] else "BREACH")) if dens else "not readable"))
            else:
                s_ok = dec_ok and slr["e_rel"] <= GATE["spectra"] and slr["dens_sea"] <= 1e-9 and slr["dens_R2"] <= 1e-9 and slr["applied"] <= GATE["fields"]
                slr["post_flip"] = False
            slr["ok"] = bool(s_ok)
            print("   end slice %s" % ("PASS" if s_ok else "FAIL"))
        A_ok = okA and nogrowth and cond8 and (slr is None or slr["ok"])
        verdict_A = "FAIL" if not A_ok else ("INCONCLUSIVE" if incon else "PASS")
        rep["pairs"][n] = {"first_flip": first, "counts": {("%.6f" % t): counts[t] for t in ts}, "percall": pcr, "records": recs, "slice": slr,
                           "no_monotone_growth": bool(nogrowth), "growth_last_first": g, "n_records": len(ts), "at_least_8": cond8,
                           "set_change_in_segment": bool(setchange), "inconclusive": incon, "A_lines_every_record": bool(okA)}
        rep["verdict"][n] = {"A": verdict_A, "B": "PASS" if rep["B"][n]["ok"] else "FAIL", "B_default_arm": "PASS" if rep["B"][ref]["ok"] else "FAIL"}
        print("VERDICT %s (v2): (A) %s [lines at every record %s; no monotone growth %s (last/first quarter median E/E0 %.2e / %.2e); records %d (at least 8: %s); "
              "set change in the segment %s; end slice %s; first flip %s; inconclusive %s]; (B) %s (the default arm %s)" % (
                  n, verdict_A, "PASS" if okA else "FAIL", nogrowth, g[0], g[1], len(ts), cond8, setchange,
                  ("n/a" if slr is None else ("PASS" if slr["ok"] else "FAIL")), first, incon or "none", rep["verdict"][n]["B"], rep["verdict"][n]["B_default_arm"]))
    for n in names:
        bad = [(t, r["lines"]) for t, r in rep["B"][n]["records"].items() if not r["ok"]]
        print("(B) %s against the production: %s at %d records%s" % (n, "PASS" if rep["B"][n]["ok"] else "FAIL", len(rep["B"][n]["records"]),
                                                                    ("; failing " + str(bad[:3])) if bad else ""))
        for t, r in rep["B"][n]["records"].items():
            print("    %s: %s%s" % (t, "; ".join(r["lines"]), ("; resume labels reported: " + fmt_disc(r["resume"], 3)) if r["resume"] else ""))
    for n in names[1:]:
        print("FINAL %s: (A) %s, (B) %s" % (n, rep["verdict"][n]["A"], rep["verdict"][n]["B"]))
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
