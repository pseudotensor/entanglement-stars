"""Second reader: J-B-vertex from the on-site controls report (run spec sec. 1.1). Read-only. Output JSON argv[1]."""
import json, sys, hashlib, math
from fractions import Fraction as Fr
P = "<internal>/written_report.md"
md5 = hashlib.md5(open(P, "rb").read()).hexdigest()
probes, tangents, local, wprobe_checks, tan_checks = [], [], [], [], []
for l in open(P):
    if not l.strip(): continue
    if '"kind":"weighted_probe"' in l[:40]: probes.append(json.loads(l)["data"])
    elif '"kind":"weighted_tangent"' in l[:40]: tangents.append(json.loads(l)["data"])
    elif '"kind":"check"' in l[:30]:
        d = json.loads(l)["data"]
        if d["name"].startswith("local_amputated_n"): local.append(d)
        elif d["name"].startswith("weighted_probe_"): wprobe_checks.append(d)
        elif d["name"].startswith("weighted_tangent_"): tan_checks.append(d)
def C(x): return Fr(x["re"]), Fr(x["im"])
def SN(n, z): return sum((Fr(1, 2) if x == 0 else Fr(1)) * Fr(z) ** x for x in range(n))
rows = []; bad = []
for p in probes:
    n = p["model"]["n"]; g = Fr(p["model"]["g"]).limit_denominator(8)
    lab = p["label"]
    z = -1 if "valley_0001" in lab else 1
    raw = C(p["raw"]); tgt = C(p["target"]); s = C(p["S_N"]); lc = C(p["local_C"])
    ok_SN = s == (SN(n, z), 0)
    ok_fact = raw == (g * s[0] * lc[0], g * s[0] * lc[1])  # S_N real
    ok_tgt = raw == tgt
    QH, QL = Fr(p["QH"]), Fr(p["QL"])
    nv = float(raw[0]) / math.sqrt(float(QH * QL))
    lo, hi = p["normalized"]["re"]
    ok_norm = lo - 1e-12 <= nv <= hi + 1e-12 and hi - lo <= 1e-12
    # spec sec 1.1 analytic normalized targets
    spec = None
    if "collinear_1_(1, 1)_(1, 1)" in lab or "collinear_1_(-1, -1)_(-1, -1)" in lab:
        spec = float(g) * (-12 / 5) / math.sqrt(544 / 625) if n == 2 else float(g) * (-16 / 9) / math.sqrt(15504 / 15625)
    elif "valley_0101" in lab: spec = float(g) * (1.5 if n == 2 else 10 / 9)
    elif "valley_0001" in lab: spec = float(g) * (-0.5 if n == 2 else 2 / 9)
    ok_spec = spec is None or abs(nv - spec) < 1e-12
    r = {"n": n, "g": str(g), "label": lab, "raw": str(raw[0]), "S_N": str(s[0]), "local_C": str(lc[0]), "norm": nv, "spec": spec,
         "ok": all([ok_SN, ok_fact, ok_tgt, ok_norm, ok_spec])}
    rows.append(r)
    if not r["ok"]: bad.append(r)
tan = []
for t in tangents:
    tan.append({"spin": t["spin"], "raw": str(C(t["raw"])[0]), "target": str(C(t["target"])[0]), "eq": t["raw"] == t["target"], "g_zero": t.get("g_zero")})
loc_vals = {}
for d in local:
    a = d["detail"]["actual"]; t = d["detail"]["target"]
    key = d["name"].split("_", 3)[3].split("_(")[0] if False else None
    loc_vals.setdefault((a["re"], a["im"]), 0); loc_vals[(a["re"], a["im"])] += 1
out = {"md5": md5, "probes": len(probes), "probe_ok": sum(r["ok"] for r in rows), "bad": bad,
       "probe_checks": len(wprobe_checks), "probe_checks_pass": sum(c["pass"] for c in wprobe_checks),
       "tangent_checks": len(tan_checks), "tangent_checks_pass": sum(c["pass"] for c in tan_checks),
       "tangents": tan, "local_checks": len(local), "local_pass": sum(c["pass"] for c in local),
       "local_actual_eq_target": sum(c["detail"]["actual"] == c["detail"]["target"] for c in local),
       "local_value_counts": {"%s+%si" % k: v for k, v in loc_vals.items()},
       "nonzero_rows": [r for r in rows if r["raw"] != "0"]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("nonzero_rows",)}, indent=0)[:3000])
