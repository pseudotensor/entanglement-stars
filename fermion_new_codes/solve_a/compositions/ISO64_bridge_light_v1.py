#!/usr/bin/env python
"""ISO64_bridge_light_v1.py (composition 4, the solve lane): the J5 light-channel readout on the isotropic bridge history isoB (ISO_64_s4 composed),
read from the channel files on the box with the post-pass's own reader (w1_postpass_v1f.load_channel, f_meas), the same per-cut excess
conventions as PAGE_solved_history_v4.1 (x_kappa the d13 'excess' S_X(C) - S_X(P_-(h)), weight 16 kappa, S = sum 16 kappa x_kappa, F_meas the
post-pass's measured floor weighted and summed, F_apriori the 'excess_floor_high' weighted and summed), against B2's channel files
(<internal>/prod, v6o_B2, kappa 1 to 18) at matched clocks. Runs no propagation.
Channels on the box: kappa 1 final (<internal>/isoB_k1); kappa 2, 6, 9, 13 as resume-mirror snapshots (their own rows to
their snapshot clock, labelled partial). Writes OUT/ISO64_light_v1.json and OUT/ISO64_light_tables_v1.md.
Usage: python ISO64_bridge_light_v1.py OUT_DIR"""
import hashlib
import json
import os
import sys

CODE_D13 = "<internal>/code_d13"
PP_DIR = "<internal>/w1"
os.environ.setdefault("W1_CODE_D13", CODE_D13)
for _p in (CODE_D13, "<internal>/code_v6", PP_DIR):
    sys.path.insert(0, _p)
import w1_postpass_v1f as PP  # noqa: E402

CUTS = ["2", "16", "24", "30", "48"]
RM = "<internal>/resume_mirror"
ISO = {1: ("<internal>/isoB_k1", "isoB_k1", "final"),
       2: (RM + "/tourn2/isoB_k2/k2_T14.000_1792441914", "isoB_k2", "partial (resume-mirror snapshot, tourn2)"),
       6: (RM + "/R2/isoB_k6/k6_T6.000_1792441819", "isoB_k6", "partial (resume-mirror snapshot, R2)"),
       9: (RM + "/tourn2/isoB_k9/k9_T36.000_1792446050", "isoB_k9", "partial (resume-mirror snapshot, tourn2)"),
       13: (RM + "/R2/isoB_k13/k13_T14.000_1792445964", "isoB_k13", "partial (resume-mirror snapshot, R2)")}
B2 = ("<internal>/prod", "v6o_B2")
MAN = []


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(d, tag, k, label):
    ch = PP.load_channel(d, tag, k)
    if ch is None:
        raise SystemExit("missing %s %s %d" % (d, tag, k))
    for suf in ("summary.json", "d13.json"):
        p = os.path.realpath(os.path.join(d, "%s_k%d_%s" % (tag, k, suf)))
        MAN.append({"label": "%s kappa %d %s" % (label, k, suf), "path": p, "md5": md5(p)})
    return ch


def cell(ch, t, lab):
    c = (((ch["clocks"].get(t) or {}).get("cuts") or {}).get(lab)) or {}
    if c.get("excess") is None:
        return None
    return {"x": c["excess"], "fmeas": PP.f_meas(ch, t, lab), "fhigh": c.get("excess_floor_high"), "S_X": c.get("S_X"), "S_X_vac": c.get("S_X_vac")}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    iso = {k: load(d, tag, k, "isoB " + st) for k, (d, tag, st) in ISO.items()}
    b2 = {k: load(B2[0], B2[1], k, "B2 production") for k in range(1, 19)}
    for p in (CODE_D13 + "/page_history_v6o.py", CODE_D13 + "/page_w1_sector_v1d.py", PP_DIR + "/w1_postpass_v1f.py"):
        MAN.append({"label": "code", "path": p, "md5": md5(p)})
    res = {"channels": {}, "b2": {}, "inventory": {}, "manifest": MAN}
    for k, ch in iso.items():
        cl = sorted(t for t in ch["clocks"] if (ch["clocks"][t].get("cuts")))
        res["inventory"][k] = {"status": ISO[k][2], "final_flag": ch["final"], "clocks_first": cl[0] if cl else None, "clocks_last": cl[-1] if cl else None,
                               "n_clocks": len(cl), "rows_last": max(ch["rows"]) if ch["rows"] else None, "audit": (ch["w1"] or {}).get("audit")}
        res["channels"][k] = {"%g" % t: {lab: cell(ch, t, lab) for lab in CUTS} for t in cl}
    for k, ch in b2.items():
        cl = sorted(t for t in ch["clocks"] if (ch["clocks"][t].get("cuts")) and t <= 40.0 + 1e-9)
        res["b2"][k] = {"%g" % t: {lab: cell(ch, t, lab) for lab in CUTS} for t in cl}
    json.dump(res, open(os.path.join(out, "ISO64_light_v1.json"), "w"), indent=0, default=float)
    # tables
    L = []
    L.append("## inventory\n")
    for k, v in res["inventory"].items():
        L.append("- kappa %d: %s; final flag %s; d13 clocks %s to %s (%d); audit %s" % (k, v["status"], v["final_flag"], v["clocks_first"], v["clocks_last"], v["n_clocks"],
                 json.dumps({a: v["audit"].get(a) for a in ("records_audited", "records_failed", "fast_banded_mismatches", "floor_premise_failures")} if v["audit"] else None)))
    def fmt(v, f="%.4e"):
        return "-" if v is None else (f % v)
    iso1 = res["channels"][1]
    clocks = sorted(float(t) for t in iso1)
    # T-A: finished channels' sum (kappa 1 only) per cut with floors, beside B2's kappa 1 term and B2's K = 18 sum
    L.append("\n## A. the finished-channel sum on isoB (kappa 1 only: S = 16 x_1) per cut, its floors, against B2 at the same clock: B2's kappa 1 term 16 x_1 (the same channel set) and B2's K = 18 sum (the delivered readout's set)\n")
    L.append("| T/r_s | cut r_s | isoB S (K=1) | isoB F_meas | isoB F_apriori | B2 16 x_1 | B2 F_meas (k1) | isoB - B2 (k1) | B2 S (K=18) | B2 F_meas (K=18) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in clocks:
        tk = "%g" % t
        for lab in CUTS:
            a = iso1[tk][lab]
            b = (res["b2"][1].get(tk) or {}).get(lab)
            allb = [(res["b2"][k].get(tk) or {}).get(lab) for k in range(1, 19)]
            SK = sum(16 * k * c["x"] for k, c in zip(range(1, 19), allb)) if all(allb) else None
            FK = sum(16 * k * c["fmeas"] for k, c in zip(range(1, 19), allb)) if all(allb) and all(c["fmeas"] is not None for c in allb) else None
            L.append("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, fmt(16 * a["x"]) if a else "-", fmt(16 * a["fmeas"] if a and a["fmeas"] is not None else None, "%.2e"),
                     fmt(16 * a["fhigh"] if a and a["fhigh"] is not None else None, "%.2e"), fmt(16 * b["x"]) if b else "-",
                     fmt(16 * b["fmeas"] if b and b["fmeas"] is not None else None, "%.2e"), fmt(16 * (a["x"] - b["x"])) if (a and b) else "-", fmt(SK), fmt(FK, "%.2e")))
    # T-B: per channel weighted terms for every channel on the box, at every clock it holds, beside B2's same channel
    L.append("\n## B. every isoB channel on the box, weighted term 16 kappa x_kappa per cut (resolved: |x| > F_meas marked *), beside B2's same channel at the same clock\n")
    L.append("| kappa | status | T/r_s | " + " | ".join("isoB %s / B2 %s" % (c, c) for c in CUTS) + " | isoB 16k F_meas (2 r_s cut) |")
    L.append("|---|---|---|" + "---|" * len(CUTS) + "---|")
    for k in sorted(res["channels"]):
        for tk in sorted(res["channels"][k], key=float):
            cells = []
            for lab in CUTS:
                a = res["channels"][k][tk][lab]; b = (res["b2"][k].get(tk) or {}).get(lab)
                sa = ("%.3e%s" % (16 * k * a["x"], "*" if (a["fmeas"] is not None and abs(a["x"]) > a["fmeas"]) else "")) if a else "-"
                sb = ("%.3e%s" % (16 * k * b["x"], "*" if (b["fmeas"] is not None and abs(b["x"]) > b["fmeas"]) else "")) if b else "-"
                cells.append("%s / %s" % (sa, sb))
            a2 = res["channels"][k][tk]["2"]
            L.append("| %d | %s | %s | %s | %s |" % (k, ISO[k][2].split(" (")[0], tk, " | ".join(cells), fmt(16 * k * a2["fmeas"] if a2 and a2["fmeas"] is not None else None, "%.2e")))
    # T-C: the common-set sums (labelled diagnostics): isoB over the channels it holds at the clock, B2 over the same kappa set
    L.append("\n## C. labelled diagnostic: the sum over the isoB channels present at the clock (a partial, non-contiguous set; not a readout) against B2's sum over the same kappa set\n")
    L.append("| T/r_s | kappa set | " + " | ".join("isoB %s / B2 same set" % c for c in CUTS) + " |")
    L.append("|---|---|" + "---|" * len(CUTS))
    allt = sorted({float(t) for k in res["channels"] for t in res["channels"][k]})
    for t in allt:
        tk = "%g" % t
        ks = [k for k in sorted(res["channels"]) if tk in res["channels"][k]]
        if len(ks) < 2:
            continue
        cells = []
        for lab in CUTS:
            a = [res["channels"][k][tk][lab] for k in ks]; b = [(res["b2"][k].get(tk) or {}).get(lab) for k in ks]
            sa = sum(16 * k * c["x"] for k, c in zip(ks, a)) if all(a) else None
            sb = sum(16 * k * c["x"] for k, c in zip(ks, b)) if all(b) else None
            cells.append("%s / %s" % (fmt(sa, "%.3e"), fmt(sb, "%.3e")))
        L.append("| %g | %s | %s |" % (t, ",".join(str(k) for k in ks), " | ".join(cells)))
    L.append("\n## manifest\n")
    L += ["- %s  %s  [%s]" % (m["md5"], m["path"], m["label"]) for m in MAN]
    open(os.path.join(out, "ISO64_light_tables_v1.md"), "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
