import json, sys
d = json.load(open(sys.argv[1]))
print("keys", list(d.keys()))
print("T", d["T"], "E", d["E"], "r_s", d["r_s"], "R_t", d["R_t"])
for res in d["results"]:
    rs = d["r_s"]
    print("==", res["background"][:20], res["probe"], "omega", res["omega"], "n_ch", len(res["channels"]), "T_list/rs", [round(t/rs, 3) for t in res["T_list"]], "R_t_bg", res["R_t_background"])
    ch = res["channels"]
    for c in [ch[0], ch[len(ch)//4], ch[len(ch)//2], ch[(3*len(ch))//4], ch[-3], ch[-2], ch[-1]]:
        p = c["plus"]
        if "error" in p:
            print("  kappa", c["kappa"], "ERROR", p["error"][:60]); continue
        mon = p["monitor"]
        print("  kappa", c["kappa"], "I %.6f" % p["I"], "A", {k: round(v, 5) for k, v in p["A"].items()}, "abs", {k: round(v, 5) for k, v in p.get("absorbed", {}).items()})
        print("     inside_total(t/rs):", " ".join("%.0f:%.3f" % (m["t"]/rs, m["inside_total"]) for m in mon))
    nerr = sum(1 for c in ch if "error" in c["plus"] or "error" in c["minus"])
    print("  channels with an error:", nerr)
