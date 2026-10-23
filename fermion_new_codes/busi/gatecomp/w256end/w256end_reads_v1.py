#!/usr/bin/env python
"""w256end_reads_v1.py (busi; inbox #477/#478): the zero-card end reads of W256_s4 run_f (sigma/a 4, r_s/a 256, RD-1d engine 59a641d2).
The run json/fields stop at T 9.375 (before birth); the records 9.4375 .. 9.625 exist only as log lines; the only slices in 9.25 .. 9.625
are 9.25 and 9.50 (the run's slice list: every 0.25). Inputs: run_f's end files (solve's end_0056 copy), the re-solved slice rows of
w256end_slices_v1.py, the stopper's own code (stop_flat_h100f_v2.py, gate_common_v1, gate_profiles_v1). Writes one JSON; tables by
w256end_tables_v1.py. Nothing here is ruled.
  (1) validation: the 9.25 slice's re-solved eps_tot, J_tot, N, v against the stored row 9.25; the 9.50 re-solve against its own applied
      fields and the 9.50 log line (E/E0, R_t, min(N - |v|)).
  (2) birth: gate_common_v1.birth on the json + the log's records 9.4375 .. 9.625 (min(N-|v|) from the log; r_min is not logged).
  (3) global gates (gate_common_v1.gate_profile, the stopper's v1/gate mode; sigma 4, line 0.1) on the stored rows plus the 9.50 row;
      root gate (stop_flat_h100f_v2.root_row, postgate) on the 9.50 row. d_T of the six-term: the code's own rule on the rows present,
      so 9.375 is central over (9.3125, 9.50) and 9.50 backward over (9.375, 9.50); the stored-only 9.375 (backward) beside.
  w256end_reads_v1.py OUT_JSON RUN_DIR RESOLVED_DIR GC_CODE_DIR FM_DIR STOPPER
"""
import json, os, re, sys, hashlib
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
import numpy as np

OUTJ, RUN, RES, GCD, FM, STOPPER = sys.argv[1:7]
TAG = "lad_W256_s4_m0p2_rs256_n4096_R04_k14_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mf"
rs = 256.0; SIG = 4.0; LINE = 0.1


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


src = open(STOPPER).read(); head = src.split("\nt_start = dt.datetime.utcnow()")[0]
G = {"__file__": os.path.join(GCD, "stop_flat_h100f_v2.py"), "__name__": "stopper_defs"}
sys.argv = ["stop_flat_h100f_v2.py", os.path.dirname(os.path.abspath(OUTJ)), TAG, FM, "--sigma", str(SIG), "--mode", "postgate", "--dry", "--once", "--line", str(LINE)]
exec(compile(head, STOPPER, "exec"), G)
GC = G["GC"]; six = G["six"]; root_row = G["root_row"]

pj = os.path.join(RUN, "collapse_%s.json" % TAG); pf = os.path.join(RUN, "collapse_%s_fields.npz" % TAG); pl = os.path.join(RUN, "log.txt")
J = json.load(open(pj)); F0 = np.load(pf, allow_pickle=True)
pr = {t: os.path.join(RES, "resolved_T%07.4f.npz" % t) for t in (9.25, 9.5)}
SJ = json.load(open(os.path.join(RES, "w256end_slices_v1.json")))
out = {"inputs": {k: {"path": p, "md5": md5(p)} for k, p in (("json", pj), ("fields", pf), ("log", pl), ("resolved_9.25", pr[9.25]), ("resolved_9.50", pr[9.5]),
                                                            ("stopper", STOPPER), ("gate_common", os.path.join(GCD, "gate_common_v1.py")),
                                                            ("gate_profiles", os.path.join(GCD, "gate_profiles_v1.py")),
                                                            ("engine_fields_v16", os.path.join(FM, "engine_fields_v16.py")))},
       "slices": {k: {"path": v["slice"], "md5": v["slice_md5"]} for k, v in SJ.items()}}
E0 = float(J["head"]["E0"]); g = float(J["args"]["g"]); out["E0"] = E0; out["two_gE0"] = 2 * g * E0
RB0 = GC.recs_by_T(J, rs); IB0 = GC.rows_by_T(F0, rs); out["json_last"] = max(RB0); out["fields_last"] = max(IB0)

# ---- log records (the json does not hold 9.4375 .. 9.625)
LOG = {}
for s in open(pl, errors="replace"):
    m = re.match(r"T/rs (\S+) E/E0 (\S+) dust \S+ pairs \S+ R_t (\S+) \(2gE (\S+);.*?r_in (\S+)\) min\(N-\|v\|\) (\S+) .*?pockets (\d+) inversions (\d+).*? it (\d+) \(eigh", s)
    if m:
        t = round(float(m.group(1)) * 16) / 16.0
        LOG[t] = {"E_over_E0": float(m.group(2)), "R_t": (None if m.group(3) == "None" else float(m.group(3))), "two_gE": float(m.group(4)),
                  "r_in": (None if m.group(5) == "None" else float(m.group(5))), "min": float(m.group(6)), "pockets": int(m.group(7)), "inversions": int(m.group(8)), "it": int(m.group(9))}
    m = re.match(r"T/rs (\S+) newton .*?kept residual (\S+) .*?; (converged|not converged)", s)
    if m:
        t = round(float(m.group(1)) * 16) / 16.0; LOG.setdefault(t, {}); LOG[t]["kept"] = float(m.group(2)); LOG[t]["newton"] = m.group(3)
out["log"] = {"%.4f" % t: LOG[t] for t in sorted(LOG) if t >= 9.0 - 1e-9}

# ---- (1) validation
v = {}
Z25 = np.load(pr[9.25]); i25 = IB0[9.25]
for k in ("eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied"):
    a_ = np.asarray(F0[k][i25], float); b_ = np.asarray(Z25[k], float)
    v[k] = {"max_abs": float(np.max(np.abs(a_ - b_))), "scale": float(np.max(np.abs(a_))), "bitwise": bool(np.array_equal(a_, b_))}
for rt in ("collocation", "ivp"):
    for k in ("N", "v"):
        a_ = np.asarray(F0[k][i25], float); b_ = np.asarray(Z25["%s_%s" % (k, rt)], float)
        v["%s_%s" % (k, rt)] = {"max_abs": float(np.max(np.abs(a_ - b_))), "scale": float(np.max(np.abs(a_))), "bitwise": bool(np.array_equal(a_, b_))}
v["E_resolve_over_record"] = {rt: SJ["9.2500"]["checks"][rt]["E"] / RB0[9.25]["E"] - 1 for rt in ("collocation", "ivp")}
v["min_N_minus_absv_record"] = RB0[9.25].get("min_N_minus_absv"); v["r_min_record"] = RB0[9.25].get("r_min_N_minus_absv")
v["min_N_minus_absv_resolve"] = SJ["9.2500"]["checks"]["collocation"]["min_N_minus_absv"]; v["r_min_resolve"] = SJ["9.2500"]["checks"]["collocation"]["r_min"]
c50 = SJ["9.5000"]["checks"]
v["T9.50"] = {"E_over_E0_resolve": {rt: c50[rt]["E"] / E0 for rt in ("collocation", "ivp")}, "E_over_E0_log": LOG[9.5]["E_over_E0"],
              "N_minus_Napplied": c50["collocation"]["max_abs_N_minus_Napplied"], "v_minus_vapplied": c50["collocation"]["max_abs_v_minus_vapplied"],
              "min_N_minus_absv_resolve": c50["collocation"]["min_N_minus_absv"], "min_log": LOG[9.5]["min"], "r_min_resolve": c50["collocation"]["r_min"],
              "outer_neg_site": c50["collocation"]["outer_neg_site"], "inner_neg_site": c50["collocation"]["inner_neg_site"], "R_t_log": LOG[9.5]["R_t"], "r_in_log": LOG[9.5]["r_in"]}
out["validation"] = v

# ---- augmented json and fields: stored rows + the 9.50 re-solve (collocation site fields, the run's route)
Z50 = np.load(pr[9.5])
J2 = dict(J); recs = [r for r in J["records"]]
for t in sorted(LOG):
    if t > out["json_last"] + 1e-9 and "E_over_E0" in LOG[t]:
        L = LOG[t]; rr = {"T": t * rs, "E": L["E_over_E0"] * E0, "R_t": L["R_t"], "min_N_minus_absv": L["min"], "from_log": True}
        if abs(t - 9.5) < 1e-9:
            rr["r_min_N_minus_absv"] = c50["collocation"]["r_min"]
        recs.append(rr)
J2["records"] = recs
keys = ("eps_tot", "J_tot", "N", "v", "N_applied", "v_applied")
F2 = {"R": np.asarray(F0["R"], float), "g": F0["g"], "ell": F0["ell"]}
F2["T"] = np.concatenate([np.asarray(F0["T"], float), [9.5 * rs]])
for k in keys:
    extra = np.asarray(Z50[k + ("_collocation" if k in ("N", "v") else "")], float)
    F2[k] = np.concatenate([np.asarray(F0[k], float), extra[None, :]], axis=0)

# ---- (2) birth
b = GC.birth(J2, rs); out["birth"] = b
r375 = RB0[9.375].get("r_min_N_minus_absv"); r50 = c50["collocation"]["r_min"]
tb = b["T_birth"]
out["R_b"] = {"r_min_9.375_json": r375, "r_min_9.50_resolve": r50, "R_t_9.4375_log": LOG[9.4375]["R_t"], "r_in_9.4375_log": LOG[9.4375]["r_in"],
              "R_b_lin_9.375_9.50": (r375 + (r50 - r375) * (tb - 9.375) / 0.125) if r375 is not None else None,
              "two_gE_9.4375": LOG[9.4375]["two_gE"], "two_gE_9.375": 2 * g * RB0[9.375]["E"]}
El, Eh = RB0[9.375]["E"], LOG[9.4375]["E_over_E0"] * E0; Eb = El + (Eh - El) * (tb - 9.375) / 0.0625
out["R_b"]["E_Tb_over_E0"] = Eb / E0; out["R_b"]["two_gE_Tb"] = 2 * g * Eb
for lab in ("R_t_9.4375_log", "R_b_lin_9.375_9.50", "r_min_9.375_json"):
    x = out["R_b"][lab]
    out["R_b"][lab + "_over"] = {"2gE0": x / (2 * g * E0), "2gE_Tb": x / (2 * g * Eb), "2gE_9.4375": x / LOG[9.4375]["two_gE"]} if x is not None else None

# ---- (3) global gates
bb = dict(b); bb["R_birth"] = out["R_b"]["R_b_lin_9.375_9.50"]
gp = GC.gate_profile(J2, F2, rs, SIG, bb, sixterm=six, line=LINE)
cols = ("T", "rmax", "sigma_Kop", "r_Kop", "sigma_H", "six_nodes", "R_t", "r_six", "six_dominant", "six_centre", "r_six_centre", "dT_mode")
out["global"] = {"rows": [dict(zip(cols, r)) for r in gp["rows"]], "T_gate": gp["T_gate"], "window": gp["window"], "R_birth_used": bb["R_birth"]}
six.cache.clear()
# the alternative r_max before the root: 2 R_t(9.4375) (the first logged root)
bb2 = dict(b); bb2["R_birth"] = LOG[9.4375]["R_t"]
gp2 = GC.gate_profile(J2, F2, rs, SIG, bb2, sixterm=six, line=LINE)
out["global_alt_rmax"] = {"rows": [dict(zip(cols, r)) for r in gp2["rows"]], "T_gate": gp2["T_gate"], "R_birth_used": bb2["R_birth"]}
six.cache.clear()
# stored only (what the run's stopper could see): 9.375 one-sided
F0d = {k: F0[k] for k in F0.files}
gp0 = GC.gate_profile(J, F0d, rs, SIG, bb, sixterm=six, line=LINE, t_from=9.0)
out["global_stored_only"] = {"rows": [dict(zip(cols, r)) for r in gp0["rows"]], "T_gate": gp0["T_gate"]}
six.cache.clear()
first = lambda rows, k: next((r["T"] for r in rows if r[k] is not None and r[k] > LINE), None)  # noqa: E731
out["T_gate_per_test"] = {"sigma_Kop": first(out["global"]["rows"], "sigma_Kop"), "six_nodes": first(out["global"]["rows"], "six_nodes"),
                          "sigma_H (one-term, reported)": first(out["global"]["rows"], "sigma_H"), "six_centre r<sigma (reported)": first(out["global"]["rows"], "six_centre")}
# ---- root gate at 9.50 (the only re-solved record with a root)
RB2 = GC.recs_by_T(J2, rs); IB2 = GC.rows_by_T(F2, rs); Rg = F2["R"]
rr = root_row(F2, RB2, IB2, Rg, 9.5)
out["root"] = [{"T": rr[0], "R_t": rr[1], "sigma_Kop_R_2R": rr[2], "sigma_absNp_over_N": rr[3], "six_R_2R": rr[4], "r_six": rr[5], "dT_mode": "backward (9.375 -> 9.50)",
                "max_of_three": max(x for x in rr[2:5] if x is not None)}]
# per-term six at 9.50 on the global nodes, for the reader
NAMES = ["|v'/N|", "|v/(N r)|", "|N'/N|", "sqrt|D k_r|", "sqrt|D k_t|", "sqrt|N''/N|"]


def terms(F, i, rlo, rhi):
    S0 = six.solve(F, i); Sp = six.solve(F, i - 1); Sn = six.solve(F, i + 1) if i + 1 < len(F["T"]) else None
    r = np.arange(rlo, rhi + 1e-9, SIG); GP = six.GP; h = SIG / 32.0; Td = np.asarray(F["T"], float)
    k0 = GP.kfields(S0[0], S0[1], r, h); kp = GP.kfields(Sp[0], Sp[1], r, h)
    if Sn is not None:
        kn = GP.kfields(Sn[0], Sn[1], r, h); dTr = (kn["kr"] - kp["kr"]) / (Td[i + 1] - Td[i - 1]); dTt = (kn["kt"] - kp["kt"]) / (Td[i + 1] - Td[i - 1])
    else:
        dTr = (k0["kr"] - kp["kr"]) / (Td[i] - Td[i - 1]); dTt = (k0["kt"] - kp["kt"]) / (Td[i] - Td[i - 1])
    Dkr = (dTr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dTt + k0["v"] * k0["kt1"]) / k0["N"]
    T6 = np.array([np.abs(k0["kr"]), np.abs(k0["kt"]), np.abs(k0["N1"] / k0["N"]), np.sqrt(np.abs(Dkr)), np.sqrt(np.abs(Dkt)), np.sqrt(np.abs(k0["N2"] / k0["N"]))])
    return {NAMES[q]: [float(SIG * T6[q].max()), float(r[int(np.argmax(T6[q]))])] for q in range(6)}


out["per_term"] = {"9.5000 nodes sigma..2R_t": terms(F2, IB2[9.5], SIG, 2 * RB2[9.5]["R_t"]),
                   "9.3750 nodes sigma..2R_b": terms(F2, IB2[9.375], SIG, 2 * bb["R_birth"])}
# ---- answers
tg = out["global"]["T_gate"]
out["answers"] = {"T_b": tb, "T_gate_global": tg, "T_b_le_T_gate": (None if tg is None else bool(tb <= tg + 1e-12)),
                  "gate_closed_by_9.50": bool(tg is not None and tg <= 9.5 + 1e-9), "root_gate_9.50_crossed": bool(out["root"][0]["max_of_three"] > LINE)}
json.dump(out, open(OUTJ, "w"), indent=1, default=str)
print(json.dumps({"birth": b, "R_b": out["R_b"], "T_gate_per_test": out["T_gate_per_test"], "root": out["root"], "answers": out["answers"]}, default=str, indent=1), flush=True)
for r in out["global"]["rows"]:
    if r["T"] >= 9.0 - 1e-9:
        print("T %.4f rmax %.1f sKop %.4f (r %.0f) sH %.4f six %s (%s, %s, r %s) centre %s" % (r["T"], r["rmax"], r["sigma_Kop"], r["r_Kop"], r["sigma_H"], r["six_nodes"], r["six_dominant"], r["dT_mode"], r["r_six"], r["six_centre"]), flush=True)
print("validation", json.dumps(v, default=str), flush=True)
print("wrote", OUTJ, flush=True)
