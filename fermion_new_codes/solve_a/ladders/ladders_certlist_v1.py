#!/usr/bin/env python3
"""ladders_certlist_v1.py (solve, the ladders' certification list): one row per ladder member of ladders_readout_v1_2.json: the engine (from
the run tag's version suffix, else the json's script field), every box copy of that engine file with its md5, the requested and reached end
clock, the certification as the readout (and so the deliveries) states it (certified prefix end, first non-counting record and its gate, the
formation record's class), the member's status file on its card mirror (last lines, read verbatim), and its final run files on the box (path,
bytes, md5). Reads stored files only. Usage: ladders_certlist_v1.py READOUT.json OUT.json"""
import hashlib, json, os, re, sys

ENGINE_DIRS = ["<internal>/engine", "<internal>/engine",
               "<internal>/compC_engine_v36", "<internal>/code", "<internal>/qi2_ladders_code"]
L = "<internal>/ladders"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def engine_of(meta):
    tag = os.path.basename(meta["path"])
    m = re.findall(r"_(v36[a-z]+)", tag)
    if m:
        return "engine_collapse_%s.py" % m[-1].rstrip("c") if m[-1] == "v36cc" else "engine_collapse_%s.py" % m[-1], "the run tag's suffix %s" % m[-1]
    return meta.get("script"), "the run json's script field"


def main():
    D = json.load(open(sys.argv[1]))
    eng_cache = {}
    rows = []
    for nm, M in D["members"].items():
        me, c, fo = M["meta"], M["convergence"], (M.get("formation") or {})
        eng, how = engine_of(me)
        if eng not in eng_cache:
            eng_cache[eng] = [{"path": os.path.join(d, eng), "md5": md5(os.path.join(d, eng))} for d in ENGINE_DIRS if os.path.exists(os.path.join(d, eng))]
        d = os.path.dirname(me["path"]); tag = os.path.basename(me["path"])[len("collapse_"):-len(".json")]
        files = []
        for f in sorted(os.listdir(d)):
            if f.startswith("collapse_%s" % tag):
                p = os.path.join(d, f)
                files.append({"path": p, "bytes": os.path.getsize(p), "md5": md5(p)})
        short = None; status = None
        mm = re.match(r"collapse_lad_(.+?)_m0p", os.path.basename(me["path"]))
        if mm:
            short = mm.group(1)
            card = me["path"].split("/ladders/")[1].split("/")[0] if "/ladders/" in me["path"] else None
            sp = os.path.join(L, card, "status_%s.txt" % short) if card else None
            if sp and os.path.exists(sp):
                lines = [x.rstrip() for x in open(sp, errors="replace") if x.strip()]
                status = {"path": sp, "md5": md5(sp), "last_lines": [x[:400] for x in lines[-3:]]}
        T_req = float(me["args"].get("T_over_rs") or 0)
        rows.append({"name": nm, "role": me["role"], "label": me["label"], "engine": eng, "engine_from": how, "engine_box_copies": eng_cache[eng],
                     "json_script": me.get("script"), "engine_entries": me.get("engine_entries"),
                     "rs_a": me["rs"], "sigma_a": me["sigma"], "tau": me["tau"], "m": me["m"], "nsites": me["nsites"], "solver": "newton" if me.get("newton") else "plain",
                     "T_requested": T_req, "T_last": c["last"], "reached_request": (c["last"] is not None and c["last"] + 1e-9 >= T_req),
                     "certified_prefix_end": c.get("certified_prefix_end"), "last_converged": c.get("last_converged"),
                     "first_noncounting": c.get("first_noncounting"), "first_noncounting_gate": c.get("first_noncounting_gate"),
                     "first_above_1e-3": c.get("first_above_1e-3"), "conf_fail_count": c.get("conf_fail_count"),
                     "formation_T": fo.get("T"), "formation_class": fo.get("class"), "formation_label": fo.get("label"),
                     "run_json": me["path"], "run_json_md5": me["md5"], "run_files": files, "status": status})
    json.dump({"source": sys.argv[1], "source_md5": md5(sys.argv[1]), "rows": rows}, open(sys.argv[2], "w"), indent=1)
    for r in rows:
        print(r["name"], r["engine"], [x["md5"][:8] for x in r["engine_box_copies"]], r["T_requested"], r["T_last"], r["certified_prefix_end"], r["formation_class"],
              (r["status"] or {}).get("last_lines", [""])[-1][:150])


if __name__ == "__main__":
    main()
