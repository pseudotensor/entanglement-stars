import json
B="<internal>/solve_a_bulk"
H=B+"/iso128_h2/run_h2/collapse_lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq.json"
P=B+"/ladders/p6/runs/lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n/collapse_lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n.json"
h={round(r["T_over_rs"],6):r for r in json.load(open(H))["records"] if r.get("E") is not None}
p={round(r["T_over_rs"],6):r for r in json.load(open(P))["records"] if r.get("E") is not None}
cs=sorted(set(h)&set(p))
d=[(t,abs(h[t]["E_over_E0"]-p[t]["E_over_E0"]),abs(h[t]["min_N_minus_absv"]-p[t]["min_N_minus_absv"])) for t in cs if t>0]
print("common",len(cs))
for t in (1.5,1.5625,1.625,1.6875,1.75,2.0,2.75,3.0,3.5625,10.75,11.25):
    if t in h and t in p: print(t,"%.2e"%abs(h[t]["E_over_E0"]-p[t]["E_over_E0"]), h[t]["E_over_E0"], p[t]["E_over_E0"])
m=max(d,key=lambda x:x[1]); print("max dE/E0 %.3e at %s"%(m[1],m[0]))
print("max over 0.0625-3.375 %.3e"%max(x[1] for x in d if x[0]<=3.375), "max to 1.5 %.2e"%max(x[1] for x in d if x[0]<=1.5))
print("max dmin %.2e at %s"%max((x[2],x[0]) for x in d))
print("first >6e-7 at", min(x[0] for x in d if x[1]>6e-7), "count>6e-7 in 0.0625-3.375:", sum(1 for x in d if x[0]<=3.375 and x[1]>6e-7), "of", sum(1 for x in d if x[0]<=3.375))
def onset(R):
    for t in sorted(R):
        if "LZ" in (R[t].get("continuation_label") or ""): return t, R[t]["continuation_label"][:80]
print("H2 LZ onset", onset(h)); print("plain LZ onset", onset(p))
