#!/usr/bin/env python3
"""ladders_diff_v1_3.py (solve, the ladders v1.3): every value of the producer's json that differs between version 1.2 and version 1.3
(ladders_readout_v1_2.json against ladders_readout_v1_3.json), with the document(s) that use it. Members tag by the composer's member lists
(compose_text_v1_3: MASS + MASS_CTRL for MASS; SIZE + SIZE_CTRL + S4_128_s8/s16 + PREP + M3a_s4 for SCALING); fits, spreads and the
rest tag by whether their key's name occurs in the composer's mass_doc or scaling_doc source (both when in a shared helper or in neither).
Numbers compare exactly (any change is listed, with the relative change); md5/bytes/path/seconds fields are listed separately as provenance.
Usage: ladders_diff_v1_3.py OLD.json NEW.json CODE_DIR OUT.json"""
import inspect, json, sys

PROV = ("md5", "bytes", "read_from", "path", "seconds", "load_seconds", "snapshot_of", "label", "running", "mtime")


def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append({"path": p + [k], "old": a.get(k, "<absent>"), "new": b.get(k, "<absent>")})
            else:
                walk(a[k], b[k], p + [k], out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": p, "old": "list of %d" % len(a), "new": "list of %d" % len(b), "lists": [a, b]})
            n = min(len(a), len(b))
            for i in range(n):
                walk(a[i], b[i], p + [i], out)
        else:
            for i in range(len(a)):
                walk(a[i], b[i], p + [i], out)
    else:
        if a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
            out.append({"path": p, "old": a, "new": b})


def short(v):
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, (dict, list)):
        s = json.dumps(v, default=str)
        return s if len(s) <= 160 else s[:157] + "..."
    return str(v)


def main():
    old, new, code, out = sys.argv[1:5]
    sys.path.insert(0, code)
    import compose_text_v1_3 as CT
    src_m = inspect.getsource(CT.mass_doc); src_s = inspect.getsource(CT.scaling_doc)
    massN = set(CT.MASS + CT.MASS_CTRL); scalN = set(CT.SIZE + CT.SIZE_CTRL + ["S4_128_s8", "S4_128_s16"] + CT.PREP + ["M3a_s4"])
    A = json.load(open(old)); B = json.load(open(new))
    raw = []
    walk(A, B, [], raw)
    rows, prov = [], []
    for r in raw:
        p = r["path"]
        docs = []
        if p and p[0] == "members" and len(p) > 1:
            nm = p[1]
            if nm in massN:
                docs.append("MASS")
            if nm in scalN:
                docs.append("SCALING")
        elif p and p[0] == "spreads" and len(p) > 1 and isinstance(p[1], int):
            sp = B["spreads"][p[1]]
            for nm in (sp.get("member"), sp.get("control")):
                if nm in massN and "MASS" not in docs:
                    docs.append("MASS")
                if nm in scalN and "SCALING" not in docs:
                    docs.append("SCALING")
            r["path"] = p[:2] + ["%s vs %s" % (sp.get("member"), sp.get("control"))] + p[2:]
        elif p and len(p) > 1:
            key = str(p[1])
            inm, ins = key in src_m, key in src_s
            if inm:
                docs.append("MASS")
            if ins:
                docs.append("SCALING")
            if not inm and not ins:
                docs = ["MASS", "SCALING"] if p[0] != "fits" else ["SCALING"] if key.startswith(("size", "width", "design")) else ["MASS"]
        row = {"path": "/".join(str(x) for x in r["path"]), "old": short(r["old"]), "new": short(r["new"]), "docs": docs}
        if isinstance(r["old"], (int, float)) and isinstance(r["new"], (int, float)) and r["old"]:
            row["rel"] = (r["new"] - r["old"]) / abs(r["old"])
        (prov if (p and str(p[-1]) in PROV) else rows).append(row)
    res = {"old": old, "new": new, "n_changed": len(rows), "n_provenance": len(prov),
           "changed": rows, "provenance": prov,
           "by_top": {k: sum(1 for r in rows if r["path"].split("/")[0] == k) for k in ("members", "fits", "spreads", "order", "definitions", "certext_check")}}
    json.dump(res, open(out, "w"), indent=1, default=str)
    print("changed %d values (+%d provenance); by top %s; MASS %d, SCALING %d" % (len(rows), len(prov), res["by_top"],
          sum("MASS" in r["docs"] for r in rows), sum("SCALING" in r["docs"] for r in rows)))


if __name__ == "__main__":
    main()
