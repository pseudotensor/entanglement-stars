#!/usr/bin/env python3
"""core_well_pendings_v12.py (the busi lane as solve builder; ReviewA_reading_batch6_v1 (S) lane line: "reads the five pendings on the box CPU
in this order: the npz centroid against the R_c column, n(R) at 1 to 8 a, the local-slope array at the named radii, the json's bin energies,
the halvings"; the commands are CORE_well_v1.1 section 0 items 2 to 5 and section 8.1, run as written over the 18 stored slices). Reads stored
outputs only; writes one md and one json. No engine run, no judgment: every value printed as read.
Usage (box): python core_well_pendings_v12.py --well <internal>/well --prod-json <records json> --s8-json <...>
    --out <internal>/pendings_v12"""
import argparse, collections, json, os, re
import numpy as np

SLICES = [("prod", t) for t in (12, 15, 16, 20, 24, 25, 28, 30, 35, 40)] + [("s8", t) for t in (12, 13, 14, 16, 20, 25, 30, 40)]


def named_radii(md_path):
    """The rows of the per-slice md's '## The profile at the named radii' table: (label, R)."""
    rows, on = [], False
    for line in open(md_path):
        if line.startswith("## The profile at the named radii"):
            on = True
            continue
        if on and line.startswith("## "):
            break
        if on and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                try:
                    rows.append((cells[0], float(cells[1])))
                except ValueError:
                    pass
    return rows


def record_rc(md_path):
    m = re.search(r"^Record:.*?R_c ([0-9.eE+-]+)", open(md_path).read(), re.M)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--well", required=True)
    ap.add_argument("--prod-json", required=True)
    ap.add_argument("--s8-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L, J = ["# CORE_well v1.1 pendings, read on the box (stored outputs only; ReviewA_reading_batch6_v1 (S) order)", ""], {}
    # 1. the npz centroid against the R_c column
    L += ["## 1. The dust centroid sum(R n_site)/sum(n_site) from each slice npz against the record's R_c (the per-slice md's Record line)", "",
          "| history | T/r_s | centroid (a) | record R_c (a) | centroid - R_c | sum n_site |", "|---|---|---|---|---|---|"]
    J["centroid"] = []
    for h, t in SLICES:
        base = os.path.join(a.well, "well_%s_T%07.2f" % (h, t))
        d = np.load(base + ".npz")
        R, n = d["R"], d["dust_number_site"]
        c = float((R * n).sum() / n.sum())
        rc = record_rc(base + ".md")
        J["centroid"].append({"history": h, "T": t, "centroid": c, "R_c": rc, "sum_n": float(n.sum())})
        L.append("| %s | %g | %.4f | %s | %s | %.4f |" % (h, t, c, "%.3f" % rc if rc is not None else "absent", "%+.4f" % (c - rc) if rc is not None else "-", n.sum()))
    # 2. n(R) at R = 1 to 8 a
    L += ["", "## 2. n(R) at the first 8 sites (R, n_density, n/2, n/0.131), every slice", ""]
    J["n_R"] = []
    for h, t in SLICES:
        d = np.load(os.path.join(a.well, "well_%s_T%07.2f.npz" % (h, t)))
        R, nd = d["R"][:8], d["n_density"][:8]
        J["n_R"].append({"history": h, "T": t, "R": R.tolist(), "n_density": nd.tolist()})
        L.append("- %s %g: R %s; n %s; n/2 %s; n/0.131 %s" % (h, t, np.array2string(R, precision=3), np.array2string(nd, precision=5),
                                                             np.array2string(nd / 2, precision=5), np.array2string(nd / 0.131, precision=4)))
    # 3. the local-slope array at the named radii
    L += ["", "## 3. local_slope_lnN_lnNenc_dense at each slice's named radii (the nearest site, its R beside)", ""]
    J["local_slope"] = []
    for h, t in SLICES:
        base = os.path.join(a.well, "well_%s_T%07.2f" % (h, t))
        d = np.load(base + ".npz")
        R, s = d["R"], d["local_slope_lnN_lnNenc_dense"]
        rows = []
        for lab, r in named_radii(base + ".md"):
            i = int(np.abs(R - r).argmin())
            rows.append({"label": lab, "R_named": r, "R_site": float(R[i]), "slope": float(s[i])})
        J["local_slope"].append({"history": h, "T": t, "rows": rows})
        L.append("- %s %g: " % (h, t) + "; ".join("%s %.3f (site %.3f) %s" % (x["label"], x["R_named"], x["R_site"], "%.5g" % x["slope"]) for x in rows))
    # 4. the json's bin energies
    L += ["", "## 4. The Killing bins from each per-slice json (killing.bins_weighted; per channel killing.bins), as stored", ""]
    J["bins"] = []
    for h, t in SLICES:
        d = json.load(open(os.path.join(a.well, "well_%s_T%07.2f.json" % (h, t))))
        bw = d["killing"]["bins_weighted"]
        ch = [{"kappa": c.get("kappa"), "n_dust": c.get("n_dust"), "bins": (c.get("killing") or {}).get("bins", "absent (channel keys: %s)" % sorted(c))} for c in d["channels"]]
        J["bins"].append({"history": h, "T": t, "bins_weighted": bw, "channels": ch})
        L.append("- %s %g weighted: %s" % (h, t, json.dumps(bw)))
        for c in ch:
            L.append("  - kappa %s (n_dust %s): %s" % (c["kappa"], c["n_dust"], json.dumps([{k: v for k, v in b.items() if k != "levels"} for b in c["bins"]]) if isinstance(c["bins"], list) else c["bins"]))
    # 5. the halvings
    L += ["", "## 5. The per-record halvings (key 'halvings' of the engine's records json; the top-level keys first)", ""]
    J["halvings"] = {}
    for h, p in (("prod", a.prod_json), ("s8", a.s8_json)):
        r = json.load(open(p))
        top = list(r.keys()) if isinstance(r, dict) else "list"
        rows = r["records"] if isinstance(r, dict) and "records" in r else r
        cnt = collections.Counter(x.get("halvings") for x in rows)
        per = [(x.get("T_over_rs"), x.get("halvings")) for x in rows]
        J["halvings"][h] = {"json": p, "top_keys": top, "n_records": len(rows), "counter": {str(k): v for k, v in cnt.items()}, "per_record": per}
        L.append("- %s (%s): top-level keys %s; %d records; Counter of halvings %s" % (h, p, top, len(rows), dict(cnt)))
        L.append("  - per record (T/r_s: halvings), the records with halvings not 0 or None: %s" % ", ".join("%g: %s" % (T, v) for T, v in per if v not in (0, None)))
    open(os.path.join(a.out, "core_well_pendings_v12.md"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(a.out, "core_well_pendings_v12.json"), "w"), indent=1, default=str)
    print("\n".join(L[:40]))
    print("written:", os.path.join(a.out, "core_well_pendings_v12.md"))


if __name__ == "__main__":
    main()
