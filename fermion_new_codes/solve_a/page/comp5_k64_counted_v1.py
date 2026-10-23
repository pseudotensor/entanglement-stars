# COMP5 v1.1: count k64 (reviewer A ruling (a), solve_cold8 section (1)); re-sum on 55 channels and recompute
# rule C (neighbour bracket, per missing channel only) for the 25 gaps, same rule as comp5_tower_2rs_data_v1.py.
# Input: the v1 json only (k64's terms from its side-data rows, which the data script read from the k64 final).
import json, sys
J = "<internal>/comp5_tower_2rs_v1.json"
OUT = "<internal>/comp5_tower_2rs_k64counted_v1_1.json"
d = json.load(open(J))
clocks = d["clocks"]
tab = {int(k): v for k, v in d["table"].items()}
r64 = {row["t"]: row for row in d["side"]["k64_R2_final"]["rows"]}
assert all(t in r64 for t in clocks)
tab[64] = {"status": "present", "wx": [r64[t]["wx"] for t in clocks], "wF": [r64[t]["wF"] for t in clocks]}
ks = sorted(k for k, v in tab.items() if v["status"] == "present")
assert len(ks) == 55, len(ks)
missing = [k for k in range(1, 81) if k not in ks]
S55 = [sum(tab[k]["wx"][i] for k in ks) for i in range(len(clocks))]
F55 = [sum(tab[k]["wF"][i] for k in ks) for i in range(len(clocks))]
S54 = d["sums"]["S_54"]
lo_of = lambda q, i: tab[q]["wx"][i] - tab[q]["wF"][i]
hi_of = lambda q, i: tab[q]["wx"][i] + tab[q]["wF"][i]
per = {}
for k in missing:
    lo = [q for q in ks if q < k][-2:]; hi = [q for q in ks if q > k][:2]
    nb = lo + hi
    per[k] = {"neighbours": nb, "lo": [min(lo_of(q, i) for q in nb) for i in range(len(clocks))],
              "hi": [max(hi_of(q, i) for q in nb) for i in range(len(clocks))]}
inside = tot = 0
for k in ks:
    wo = [q for q in ks if q != k]
    lo = [q for q in wo if q < k][-2:]; hi = [q for q in wo if q > k][:2]
    if not lo or not hi:
        continue
    for i, t in enumerate(clocks):
        if t < 40.0:
            continue
        v = tab[k]["wx"][i]; tot += 1
        if min(lo_of(q, i) for q in lo + hi) <= v <= max(hi_of(q, i) for q in lo + hi):
            inside += 1
i40, i100 = clocks.index(40.0), clocks.index(100.0)
print(f"channels present {len(ks)}, missing {len(missing)}: {missing}")
for t in (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
    i = clocks.index(t)
    print(f"T {t:5.1f}: S_55 {S55[i]:.6f} (S_54 {S54[i]:.6f}, k64 {tab[64]['wx'][i]:+.4e}), F_55 {F55[i]:.3e}, S_19to63 {sum(tab[k]['wx'][i] for k in ks if 19<=k<=63):+.4e}")
print(f"rule C leave-one-out from 40 on 55 channels: {inside} of {tot} ({100*inside/tot:.1f} percent)")
print("rule C per missing channel at 100 r_s:")
for k in missing:
    print(f"  k{k}: [{per[k]['lo'][i100]:+.3e}, {per[k]['hi'][i100]:+.3e}] from {','.join(map(str, per[k]['neighbours']))}")
json.dump({"input": J, "present": ks, "missing": missing, "clocks": clocks, "S_55": S55, "F_55": F55,
           "ruleC_per_channel": per, "ruleC_leave_one_out_from40": {"inside": inside, "total": tot},
           "k64": {"wx": tab[64]["wx"], "wF": tab[64]["wF"], "provenance": "box-mirrored final, no card-side md5"}},
          open(OUT, "w"), indent=1)
print("wrote", OUT)
