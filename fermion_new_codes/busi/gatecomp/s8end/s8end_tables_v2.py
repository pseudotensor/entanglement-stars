#!/usr/bin/env python
"""s8end_tables_v2.py (busi, inbox #457; v2 = v1 with pipes in cells escaped): emits the tables of F256_s8_end_reads_v1.md from s8end_reads_v1.json and fills a template:
{{TABLE:name}} -> a markdown table; {{a/b/c|fmt}} -> a number from the json (list items by index). Nothing is retyped by hand.
  s8end_tables_v2.py READS_JSON TEMPLATE_MD OUT_MD
"""
import json, re, sys

o = json.load(open(sys.argv[1])); tpl = open(sys.argv[2]).read()


def f(x, p="%.4f"):
    return "-" if x is None else (p % x)


def tab(head, rows):
    e = lambda c: c.replace("|", "\\|")  # noqa: E731
    return "\n".join(["| " + " | ".join(map(e, head)) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(e, r)) + " |" for r in rows])


T = {}
T["charge"] = tab(["T/r_s", "E/E0 (record)", "1 - E/E0", "re-solve - record (rel)", "Gauss rows - record (rel)", "tail control - record (rel)",
                   "M(R_t)/E", "M(2R_t)/E", "M(2000a)/E", "E0/E - 1 (kappa_H shift)", "certified"],
                  [[f(c["T"]), f(c["E_over_E0"], "%.7f"), f(c["dev"], "%.2e"), f(c["E_resolve_minus_E_rel"], "%.1e"), f(c["E_gauss_minus_E_rel"], "%.1e"),
                    f((c["E_tail_over_E0"] * o["E0"] - c["E"]) / c["E"] if c["E_tail_over_E0"] is not None else None, "%.1e"),
                    f(c["M_over_E"]["R_t"]), f(c["M_over_E"]["2R_t"], "%.5f"), f(c["M_over_E"]["2000a"], "%.5f"), f(c["kappaH_ratio_E0_over_E_minus1"], "%.4f"),
                    ("yes" if c["certified"] else "**NO (5.0, not a record)**")] for c in o["charge"]])
T["crossings"] = tab(["line abs(1 - E/E0)", "T (linear)", "bracket T_lo (1 - E/E0)", "bracket T_hi (1 - E/E0)"],
                     [[f(c["line"], "%g"), f(c["T"]), "%s (%s)" % (f(c.get("T_lo")), f(c.get("dev_lo"), "%.2e")), "%s (%s)" % (f(c.get("T_hi")), f(c.get("dev_hi"), "%.2e"))]
                      for c in o["crossings"]])
T["s16crossings"] = tab(["line abs(1 - E/E0), 16a", "T (linear)", "bracket T_lo (1 - E/E0)", "bracket T_hi (1 - E/E0)"],
                        [[f(c["line"], "%g"), f(c["T"]), "%s (%s)" % (f(c.get("T_lo")), f(c.get("dev_lo"), "%.2e")), "%s (%s)" % (f(c.get("T_hi")), f(c.get("dev_hi"), "%.2e"))]
                         for c in o["s16"]["crossings"]])
T["rate"] = tab(["T/r_s", "kappa_T", "4 g E kappa_T (record E)", "4 g E0 kappa_T (E0)", "dE/dT", "dE/dT - dH_in/dT"],
                [[f(c["T"]), f(c["kappa_T"], "%.6f"), f(c["four_gE_kappa_T"]), f(c["four_gE0_kappa_T"]), f(c["dE_dT"], "%.4f"), f(c["balance_dE_minus_dHin"], "%.4f")]
                 for c in o["charge"] if c["kappa_T"] is not None])
T["s16"] = tab(["T/r_s", "16a E/E0", "16a R_t (a)", "16a 2gE (a)", "16a R_t - 2gE", "16a kept", "8a E/E0", "8a R_t (a)", "8a 2gE (a)", "8a certified"],
               [[f(m["T"]), f(m["E_over_E0_16"]), f(m["R_t_16"], "%.2f"), f(m["two_gE_16"], "%.2f"), f(m["R_t_minus_2gE_16"], "%.2f"), f(m["kept_16"], "%.1e"),
                 f(m["E_over_E0_8"]), f(m["R_t_8"], "%.2f"), f(m["two_gE_8"], "%.2f"),
                 ("-" if m["certified_8"] is None else ("yes" if m["certified_8"] else "**NO**"))] for m in o["s16"]["matched"]])
T["tail"] = tab(["T/r_s", "R_t (a)", "2gE (a)", "R_t - 2gE (a)", "kept residual", "evaluations", "min(N - abs v)", "at r (a)", "pockets", "inversions", "regime max", "wall s"],
                [[f(t["T"]), f(t["R_t"], "%.2f"), f(t["two_gE"], "%.2f"), f(t["R_t_minus_2gE"], "%.2f"), f(t["kept_residual"], "%.2e") + ("" if t["meets_stop_rule"] else " **FAIL**"),
                  str(t["evaluations"]), f(t["min_N_minus_absv"], "%.4f"), f(t["r_min_N_minus_absv"], "%.1f"), str(t["n_pocket"]), str(t["inversions_total"]),
                  f(t["regime_max_ratio"], "%.3f"), f(t["seconds"], "%.0f")] for t in o["tail"]])
T["gate"] = tab(["T/r_s", "R_t (a)", "six-term [R_t,2R_t]", "binding (r)", "six-term, d_T backward", "sigma K_op", "sigma abs N'/N", "sigma abs H", "six-term [R_t,4R_t]", "live ROOTGATE.tsv (K_op, N'/N, six)"],
                [[f(q["T"]), f(q["R_t"], "%.2f"), f(q["six"], "%.5f"), "%s (%s)" % (q["binding"], f(q["r_six"], "%.0f")), f(q["six_backward"], "%.5f"),
                  f(q["sigma_Kop"], "%.5f"), f(q["sigma_absNp_over_N"], "%.1e"), f(q["sigma_H_R_2R"], "%.5f"), f(q["six_R_4R"], "%.5f"),
                  ", ".join((q["live_ROOTGATE"] or ["-"] * 4)[1:4])] for q in o["gate"]])
T["terms"] = tab(["T/r_s"] + list(o["gate"][0]["per_term"]),
                 [[f(q["T"])] + ["%s (%s)" % (f(v[0], "%.4f"), f(v[1], "%.0f")) for v in q["per_term"].values()] for q in o["gate"]])
T["trail"] = tab(["json read to T/r_s", "polls", "first (UTC)", "last (UTC)", "T_gate^R", "root closed"],
                 [[k, str(v["n"]), v["first_utc"], v["last_utc"], v["T_gate_R"], v["closed"]] for k, v in sorted(o["monitor_trail"].items(), key=lambda x: float(x[0]))])
T["restart"] = tab(["T/r_s", "pre-gate run E", "end run E", "bitwise", "kept (pre / end)"],
                   [[f(c["T"]), f(c["E_pre"], "%.10f"), f(c["E_end"], "%.10f"), "yes" if c["bitwise_E"] else "NO", "%s / %s" % (f(c["kept_pre"], "%.3e"), f(c["kept_end"], "%.3e"))]
                    for c in o["restart_vs_pregate"]])
T["crash"] = tab(["T/r_s (log)", "first post-gate run (cache off): E/E0, R_t, min, kept, wall s", "end run (cache on): E/E0, R_t, min, kept, wall s", "equal (printed digits)"],
                 [[c["T"], "%s, %s, %s, %s, %d" % (c["crash_run"]["E_over_E0"], c["crash_run"]["R_t"], c["crash_run"]["min"], c["crash_run"]["kept"], c["crash_run"]["seconds"]),
                   "%s, %s, %s, %s, %d" % (c["end_run"]["E_over_E0"], c["end_run"]["R_t"], c["end_run"]["min"], c["end_run"]["kept"], c["end_run"]["seconds"]),
                   "yes" if all(c["crash_run"][k] == c["end_run"][k] for k in ("E_over_E0", "R_t", "min", "kept", "pockets", "inversions")) else "NO"]
                  for c in o["restart_vs_crashrun"]])
T["slices"] = tab(["slice T/r_s", "written to (card path)", "on the box mirror"],
                  [[f(s["T"]), s["path"].split("/F256_s8/")[-1].split("/orbitals")[0] + "/", (lambda k: ("yes, md5 %s, RESUMABLE %s%s" % (o["slices"]["box_mirror"][k]["orbitals_md5_listed"][:8], o["slices"]["box_mirror"][k]["RESUMABLE"],
                    ", NOT A RESTART POINT" if o["slices"]["box_mirror"][k]["not_a_restart_point"] else "")) if k in o["slices"]["box_mirror"] else "no")("T%07.2f" % s["T"])]
                   for s in o["slices"]["written_per_json"]])
w = o["wall"]; cum_on = sum(x["seconds_end_run_cache_on"] for x in w if x["T"] < 4.99); cum_off = sum(x["seconds_crash_run_cache_off"] for x in w if x["seconds_crash_run_cache_off"])
o["derived"] = {"wall_on_to_4.9375_h": cum_on / 3600.0, "wall_off_to_4.9375_h": cum_off / 3600.0, "wall_on_5.0_h": [x for x in w if x["T"] > 4.99][0]["seconds_end_run_cache_on"] / 3600.0,
                "wall_on_per_rs_h": cum_on / 3600.0 / 0.4375, "wall_pre_4.0_4.5_s": sum(o["wall_pre_gate_4.0_4.5"]) / len(o["wall_pre_gate_4.0_4.5"]),
                "gauss_rel_4.9375": [c for c in o["charge"] if abs(c["T"] - 4.9375) < 1e-9][0]["E_gauss_minus_E_rel"],
                "gate_max_six": max(q["six"] for q in o["gate"]), "gate_max_three": max(q["max_of_three"] for q in o["gate"]),
                "gate_max_Np": max(q["sigma_absNp_over_N"] for q in o["gate"]), "dev_4.9375_16": [m for m in o["s16"]["matched"] if abs(m["T"] - 4.9375) < 1e-9][0]["E_over_E0_16"]}


def val(path, fm):
    x = o
    for k in path.split("/"):
        x = x[int(k)] if isinstance(x, list) else x[k]
    return fm % x if fm else str(x)


s = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: T[m.group(1)], tpl)
s = re.sub(r"\{\{([A-Za-z0-9_./\-]+?)(?:\|([^}]+))?\}\}", lambda m: val(m.group(1), m.group(2)), s)
assert "{{" not in s, "unfilled placeholder"
open(sys.argv[3], "w").write(s)
print("wrote", sys.argv[3])
