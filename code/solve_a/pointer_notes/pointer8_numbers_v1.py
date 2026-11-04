"""pointer8_numbers_v1 (s2-vert; lead #1482, POINTER_GAPS item 8): the two numbers of 06_scaling:216, from the stored record.
R_t: the stored r11 history's record with R_t = 49.48a (json e27e457d; the closed BRK-9 sentence's 'computed R_t = 49.5a'); f = 0.887 is
the paper's capacity fraction (declared input, as in BRK9_CLOSED_SENTENCE_v1.md line 13). K = 8 interior bound 32 K (K+1) (R_t/a) ln 2;
assumed capacity 4 pi f (R_t/a)^2 (= f A/(4G) at G/a^2 = 1/4). Usage: python pointer8_numbers_v1.py OUT.json"""
import json, math, sys, os, hashlib
OUT = sys.argv[1]; assert not os.path.exists(OUT)
JN = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json"
j = json.load(open(JN)); rec = [r for r in j["records"] if r.get("R_t") and abs(r["R_t"] - 49.48) < 0.06]
r = rec[0]; Rt = float(r["R_t"]); K = 8; f = 0.887
res = {"script": "pointer8_numbers_v1", "md5_self": hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest(), "json": JN,
       "json_md5": hashlib.md5(open(JN, "rb").read()).hexdigest(), "record_T_over_rs": r["T_over_rs"], "R_t_over_a": Rt, "n_matching_records": len(rec),
       "K": K, "f_declared": f, "interior_bound_nats": 32 * K * (K + 1) * Rt * math.log(2), "capacity_nats": 4 * math.pi * f * Rt ** 2,
       "G_over_a2": j["args"]["g"]}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
