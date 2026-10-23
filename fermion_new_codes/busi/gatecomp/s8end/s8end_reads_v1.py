#!/usr/bin/env python
"""s8end_reads_v1.py (busi; inbox #457; ReviewA_s8end_v1 sec. 3 (a)-(e) and ReviewB_s8end_v1 sec. 3): the zero-card reads on F256_s8's
md5-checked end files (sigma/a 8, r_s/a 256), with F256_s16's end json for the matched clocks. Box CPU only. Writes one JSON; the tables
are emitted from it by s8end_tables_v1.py. Nothing here is ruled.

Definitions (every one printed in the tables' captions):
  E (collar charge)  the engine's record E = W(r_collar)/(2g), W' = 8 pi g r^2 N^2 rho_ell (engine_fields_v16; the collar r_collar is the
                     chain's last node 4096 a, flagged): the charge every CHARGE.tsv and log line prints. Re-read here three ways:
     E_resolve        W(r_collar)/(2g) from a fresh engine_fields_v16.solve_constraints_ivp on the record's stored eps_tot, J_tot (collar flag,
                      support 1e-14): the record's charge reproduced from its sources;
     E_gauss          sum_n eps_tot_n <N^2>_n with <N^2>_n the handoff's N2avg (the engine's kernel average of N^2): the Gauss identity, the
                      lapse-weighted rows form of the same charge;
     E_tail           info["W_tail"]["E_tail_end"]: W integrated 8 ell past the collar (the engine's stated collar control, charge(offset));
     shell sum        E_dust_shells + E_pairs_shells (the unweighted sum of the shells' energies; NOT a conserved charge, printed for scale).
  correction (d)     per record, the exterior's constancy beyond the matter: M(r) = W(r)/(2g) at r = 2 R_t and 4 R_t and at the collar, against E;
                     and the kappa_H normalization: kappa_H(E0)/kappa_H(E) - 1 = E/E0 - 1 ... printed as E0/E - 1 (kappa_H = 1/(4 g E)).
                     The ray form of (d) (ReviewA_redshift_v1 diag (d): (psi_D - psi_vac) + ln(f_D/f_vac) along fronts) needs the fronts;
                     it is the web lane's and is not computed here.
  crossings          |E/E0 - 1| at 1e-3, 1e-2, 3e-2, 6e-2: linear in T between the bracketing records (both printed).
  birth              gate_common_v1.birth (the plan's item 2) on the json; E at T_b linear between the bracketing records; R_b against
                     2 g E0 and against 2 g E(T_b) (the 64 convention).
  root gate          stop_flat_h100f_v2.py's own root_row (executed from its source): on the nodes r = ceil(R_t/sigma) sigma ... 2 R_t:
                     sigma ||K||_op (sites in [R_t, 2R_t]), sigma max|N'/N| and the six-term of record (sigma Lambda_geom, h = sigma/32,
                     d_T central between the adjacent records); per term apart; the same with d_T backward (no successor) beside, which
                     matters at 4.9375, whose successor 5.0 is UNCONVERGED; sigma |H| = sigma |d_r v| on [R_t, 2R_t] reported.
  certification      the record's solve_v32: kept residual, evaluations, meets_stop_rule, converged; the restart reproduction: the pre-gate
                     run's json (run_stop_backup, to 4.6875) and the first post-gate run's log (log_crash_0756, cache off) against the end json.
  s8end_reads_v1.py OUT_JSON END_DIR S16_RUN_DIR PREGATE_RUN_DIR GC_CODE_DIR FM_DIR STOPPER
"""
import json, os, re, sys, glob, hashlib
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
import numpy as np

OUTJ, END, S16, PRE, GCD, FM, STOPPER = sys.argv[1:8]
TAG8 = "lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb"
TAG16 = "lad_F256_s16_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell16_prep4_A_R2_avg_newton_v36nq_rd13mfb"
rs = 256.0; SIG = 8.0


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# the stopper's own definitions (as s16_rootgate_test_v1)
src = open(STOPPER).read(); head = src.split("\nt_start = dt.datetime.utcnow()")[0]
G = {"__file__": os.path.join(GCD, "stop_flat_h100f_v2.py"), "__name__": "stopper_defs"}
sys.argv = ["stop_flat_h100f_v2.py", os.path.dirname(os.path.abspath(OUTJ)), TAG8, FM, "--sigma", str(SIG), "--mode", "postgate", "--dry", "--once"]
exec(compile(head, STOPPER, "exec"), G)
GC = G["GC"]; six = G["six"]; root_row = G["root_row"]; FMm = six.FM

RUN = os.path.join(END, "run")
pj = os.path.join(RUN, "collapse_%s.json" % TAG8); pf = os.path.join(RUN, "collapse_%s_fields.npz" % TAG8); ph = os.path.join(RUN, "collapse_%s_handoff.npz" % TAG8)
J = json.load(open(pj)); F = np.load(pf, allow_pickle=True); H = np.load(ph, allow_pickle=True)
p16 = os.path.join(S16, "collapse_%s.json" % TAG16); J16 = json.load(open(p16))
ppre = os.path.join(PRE, "collapse_%s.json" % TAG8); JP = json.load(open(ppre))
out = {"inputs": {k: {"path": p, "md5": md5(p)} for k, p in (("json8", pj), ("fields8", pf), ("handoff8", ph), ("json16", p16), ("json8_pregate", ppre),
                                                                ("stopper", STOPPER), ("gate_common", os.path.join(GCD, "gate_common_v1.py")),
                                                                ("gate_profiles", os.path.join(GCD, "gate_profiles_v1.py")),
                                                                ("engine_fields_v16", os.path.join(FM, "engine_fields_v16.py")))}}
E0 = float(J["head"]["E0"]); g = float(J["args"]["g"]); ell = float(J["args"]["ell"])
out["E0"] = E0; out["g"] = g; out["two_gE0"] = 2 * g * E0
RB = GC.recs_by_T(J, rs); IB = GC.rows_by_T(F, rs); Rg = np.asarray(F["R"], float)
HB = {round(float(t) / rs, 6): i for i, t in enumerate(np.asarray(H["T"], float))}
Ts = sorted(RB)


def npocket(r):
    m = re.search(r"n_pocket (\d+)", r.get("continuation_label") or "")
    return int(m.group(1)) if m else None


def kept(r):
    s = r.get("solve_v32") or {}
    return s.get("kept_residual"), s.get("evaluations"), s.get("meets_stop_rule"), s.get("continuation"), s.get("kept_from")


# ---------- (a) and reviewer B's charge audit: per record 4.0 .. 5.0 (5.0 flagged)
charge = []
for t in Ts:
    if t < 4.0 - 1e-9:
        continue
    r = RB[t]; i = IB[t]
    _, _, _, info = FMm.solve_constraints_ivp(np.asarray(F["eps_tot"][i], float), np.asarray(F["J_tot"][i], float), float(F["g"]), Rg,
                                              float(F["ell"]), collar_policy="flag", support_rel=1e-14)
    Eres = float(FMm.charge(info, g)); Etail = float(info["W_tail"]["E_tail_end"]) if info.get("W_tail") else None
    Eg = float(np.sum(np.asarray(F["eps_tot"][i], float) * np.asarray(H["N2avg"][HB[t]], float))) if t in HB else None
    Rt = r.get("R_t")
    Mr = {}
    for lab, rr in (("2R_t", 2 * Rt if Rt else None), ("4R_t", 4 * Rt if Rt else None), ("R_t", Rt), ("1000a", 1000.0), ("2000a", 2000.0)):
        Mr[lab] = float(FMm.mass_function(info, g, rr)) if rr else None
    kr, ev, ms, cont, kf = kept(r)
    charge.append({"T": t, "E": r["E"], "E_over_E0": r["E"] / E0, "dev": 1 - r["E"] / E0, "E_resolve_over_E0": Eres / E0, "E_resolve_minus_E_rel": (Eres - r["E"]) / r["E"],
                   "E_gauss_over_E0": (Eg / E0 if Eg is not None else None), "E_gauss_minus_E_rel": ((Eg - r["E"]) / r["E"] if Eg is not None else None),
                   "E_tail_over_E0": (Etail / E0 if Etail is not None else None), "shell_sum_over_E0": (r["E_dust_shells"] + r["E_pairs_shells"]) / E0,
                   "M_over_E": {k: (v / r["E"] if v is not None else None) for k, v in Mr.items()}, "kappaH_ratio_E0_over_E_minus1": E0 / r["E"] - 1,
                   "kappa_T": r.get("kappa_T"), "four_gE_kappa_T": r.get("four_gE_kappa_T"), "four_gE0_kappa_T": (4 * g * E0 * r["kappa_T"] if r.get("kappa_T") is not None else None),
                   "dE_dT": r.get("dE_dT"), "dHin_dT": r.get("dHin_dT"), "balance_dE_minus_dHin": r.get("balance_d_E_minus_Hin_dT"), "H_in": r.get("H_in"),
                   "pairs_inside_Rt": r.get("pairs_inside_Rt"), "pairs_outside_Rt": r.get("pairs_outside_Rt"), "W_over_2g_at_Rt": r.get("W_over_2g_at_Rt"),
                   "kept_residual": kr, "certified": bool(r.get("converged")) and bool(ms), "resolve_failed": bool(info.get("failed"))})
    print("charge T %.4f E/E0 %.7f resolve %.7f gauss %s tail %s" % (t, r["E"] / E0, Eres / E0, charge[-1]["E_gauss_over_E0"], charge[-1]["E_tail_over_E0"]), flush=True)
out["charge"] = charge

cert = [c for c in charge if c["certified"]]


def crossing(rows, line):
    for a, b in zip(rows[:-1], rows[1:]):
        if abs(a["dev"]) < line <= abs(b["dev"]):
            return {"line": line, "T": a["T"] + (line - abs(a["dev"])) * (b["T"] - a["T"]) / (abs(b["dev"]) - abs(a["dev"])), "T_lo": a["T"], "dev_lo": a["dev"], "T_hi": b["T"], "dev_hi": b["dev"]}
    return {"line": line, "T": None}


allrecs = [{"T": t, "dev": 1 - RB[t]["E"] / E0} for t in Ts]
out["crossings"] = [crossing(allrecs, x) for x in (1e-3, 1e-2, 3e-2, 6e-2)]
b = GC.birth(J, rs); out["birth"] = b
tl, th = b["T_lo"], b["T_hi"]; El, Eh = RB[round(tl, 6)]["E"], RB[round(th, 6)]["E"]
Eb = El + (Eh - El) * (b["T_birth"] - tl) / (th - tl)
out["at_birth"] = {"E_over_E0": Eb / E0, "E_lo_over_E0": El / E0, "E_hi_over_E0": Eh / E0, "R_birth": b["R_birth"], "R_b_over_2gE0": b["R_birth"] / (2 * g * E0),
                   "R_b_over_2gE_Tb": b["R_birth"] / (2 * g * Eb), "R_b_over_2gE_lo": b["R_birth"] / (2 * g * El), "R_b_over_2gE_hi": b["R_birth"] / (2 * g * Eh),
                   "reviewa_test_E_over_E0_ge_0p97_at_4.3125_4.375": bool(El / E0 >= 0.97 and Eh / E0 >= 0.97)}
out["at_gate_4.5625"] = {"E_over_E0": RB[4.5625]["E"] / E0}
r49, r50 = RB[4.9375], RB[5.0]
out["arith"] = {"E50_over_E0": r50["E"] / E0, "below_E0_pct": 100 * (1 - r50["E"] / E0), "below_4.9375_pct": 100 * (1 - r50["E"] / r49["E"]),
                "two_gE_4.9375": 2 * g * r49["E"], "R_b": b["R_birth"], "two_gE_minus_R_b_rel": (2 * g * r49["E"] - b["R_birth"]) / b["R_birth"],
                "kappaH_E0_vs_E_4.9375_pct": 100 * (E0 / r49["E"] - 1)}

# ---------- (c): per record 4.5 .. 5.0
tail = []
for t in Ts:
    if t < 4.5 - 1e-9:
        continue
    r = RB[t]; kr, ev, ms, cont, kf = kept(r)
    tail.append({"T": t, "R_t": r.get("R_t"), "two_gE": 2 * g * r["E"], "R_t_minus_2gE": (r["R_t"] - 2 * g * r["E"]) if r.get("R_t") else None,
                 "R_t_over_2gE": r.get("R_t_over_2gE"), "trapping_zeros": r.get("trapping_zeros"), "kept_residual": kr, "evaluations": ev, "meets_stop_rule": ms,
                 "converged": r.get("converged"), "continuation": cont, "kept_from": kf, "min_N_minus_absv": r.get("min_N_minus_absv"),
                 "r_min_N_minus_absv": r.get("r_min_N_minus_absv"), "n_pocket": npocket(r), "inversions_total": r.get("inversions_total"),
                 "nsub_max": r.get("nsub_max"), "iterations": r.get("iterations"), "eigendecompositions": r.get("eigendecompositions"), "seconds": r.get("seconds"),
                 "regime_max_ratio": (r.get("regime") or {}).get("max_ratio"), "min_occupied_weight": r.get("min_occupied_weight"), "N_min": r.get("N_min"),
                 "R_sh": r.get("R_sh"), "dust_over_E0": r["E_dust_shells"] / E0, "E_over_E0": r["E"] / E0})
out["tail"] = tail

# ---------- (d) and reviewer B's root-gate audit: 4.5 .. 4.9375
NAMES = ["|v'/N|", "|v/(N r)|", "|N'/N|", "sqrt|D k_r|", "sqrt|D k_t|", "sqrt|N''/N|"]


def terms(t, force=None, rfac=2.0):
    rec = RB[t]; i = IB[t]; Rt = rec["R_t"]; n_ = len(F["T"])
    S0 = six.solve(F, i); Sp = six.solve(F, i - 1) if i > 0 else None; Sn = six.solve(F, i + 1) if i + 1 < n_ else None
    if force == "backward":
        Sn = None
    r = np.arange(np.ceil(Rt / SIG) * SIG, rfac * Rt + 1e-9, SIG); GP = six.GP; h = SIG / 32.0; Td = np.asarray(F["T"], float)
    k0 = GP.kfields(S0[0], S0[1], r, h)
    kp = GP.kfields(Sp[0], Sp[1], r, h) if Sp else None; kn = GP.kfields(Sn[0], Sn[1], r, h) if Sn else None
    if kp is not None and kn is not None:
        dTr = (kn["kr"] - kp["kr"]) / (Td[i + 1] - Td[i - 1]); dTt = (kn["kt"] - kp["kt"]) / (Td[i + 1] - Td[i - 1]); mode = "central"
    elif kp is not None:
        dTr = (k0["kr"] - kp["kr"]) / (Td[i] - Td[i - 1]); dTt = (k0["kt"] - kp["kt"]) / (Td[i] - Td[i - 1]); mode = "backward"
    else:
        dTr = (kn["kr"] - k0["kr"]) / (Td[i + 1] - Td[i]); dTt = (kn["kt"] - k0["kt"]) / (Td[i + 1] - Td[i]); mode = "forward"
    Dkr = (dTr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dTt + k0["v"] * k0["kt1"]) / k0["N"]
    T6 = np.array([np.abs(k0["kr"]), np.abs(k0["kt"]), np.abs(k0["N1"] / k0["N"]), np.sqrt(np.abs(Dkr)), np.sqrt(np.abs(Dkt)), np.sqrt(np.abs(k0["N2"] / k0["N"]))])
    L6 = T6.max(axis=0); j = int(np.argmax(L6))
    return {"six": float(SIG * L6[j]), "r_six": float(r[j]), "binding": NAMES[int(np.argmax(T6[:, j]))], "mode": mode,
            "per_term": {NAMES[q]: [float(SIG * T6[q].max()), float(r[int(np.argmax(T6[q]))])] for q in range(6)}}


rg_live = {}
for line in open(os.path.join(END, "member", "ROOTGATE.tsv")):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split("\t"); rg_live[round(float(p[0]), 6)] = [x.strip() for x in p[1:]]
gate = []
for t in Ts:
    if t < 4.5 - 1e-9 or t > 4.9375 + 1e-9:
        continue
    rr = root_row(F, RB, IB, Rg, t)
    tc = terms(t); assert abs(tc["six"] - rr[4]) < 1e-12 * max(1.0, rr[4]), (tc["six"], rr[4])
    tb = terms(t, force="backward"); t4 = terms(t, rfac=4.0)
    i = IB[t]; Rt = RB[t]["R_t"]; K, Hh = GC.kop_rows(Rg, F["N"][i], F["v"][i]); sel = (Rg >= Rt) & (Rg <= 2 * Rt)
    gate.append({"T": t, "R_t": rr[1], "sigma_Kop": rr[2], "sigma_absNp_over_N": rr[3], "six": rr[4], "r_six": rr[5], "binding": tc["binding"], "dT_mode": tc["mode"],
                 "per_term": tc["per_term"], "six_backward": tb["six"], "six_backward_binding": tb["binding"], "six_R_4R": t4["six"], "r_six_R_4R": t4["r_six"],
                 "sigma_H_R_2R": float(SIG * np.max(Hh[sel])), "max_of_three": max(rr[2], rr[3], rr[4]), "live_ROOTGATE": rg_live.get(t),
                 "outer_root": (RB[t].get("trapping_zeros") or [None])[-1]})
    print("gate T %.4f six %.5f (%s, %s) back %.5f sKop %.5f sN'/N %.2e" % (t, rr[4], tc["binding"], tc["mode"], tb["six"], rr[2], rr[3]), flush=True)
out["gate"] = gate
# the monitor's evaluation trail
sw = open(os.path.join(END, "member", "STOP_WATCH.txt")).read().splitlines()
trail = {}
for s in sw:
    m = re.match(r"(\S+) postgate: json to (\S+); T_gate\^R (\S+); root closed (\S+)", s)
    if m:
        k = m.group(2); trail.setdefault(k, {"n": 0, "first_utc": m.group(1), "last_utc": m.group(1), "T_gate_R": m.group(3), "closed": m.group(4)})
        trail[k]["n"] += 1; trail[k]["last_utc"] = m.group(1)
out["monitor_trail"] = trail

# ---------- certification and the restart reproduction
pre = GC.recs_by_T(JP, rs)
cmp_pre = []
for t in sorted(pre):
    if t < 4.5 - 1e-9:
        continue
    a, e = pre[t], RB.get(t)
    if e is None:
        continue
    cmp_pre.append({"T": t, "E_pre": a["E"], "E_end": e["E"], "dE_rel": (e["E"] - a["E"]) / a["E"], "R_t_pre": a.get("R_t"), "R_t_end": e.get("R_t"),
                    "kept_pre": (a.get("solve_v32") or {}).get("kept_residual"), "kept_end": (e.get("solve_v32") or {}).get("kept_residual"),
                    "min_pre": a.get("min_N_minus_absv"), "min_end": e.get("min_N_minus_absv"), "bitwise_E": a["E"] == e["E"]})
out["restart_vs_pregate"] = cmp_pre


def parse_log(p):
    recs = {}
    for s in open(p, errors="replace"):
        m = re.match(r"T/rs (\S+) E/E0 (\S+) dust (\S+) pairs (\S+) R_t (\S+) .*?min\(N-\|v\|\) (\S+) .*?pockets (\d+) inversions (\d+).*?\((\d+) s; eigh.*?kept (\S+), confirming", s)
        if m:
            recs[m.group(1)] = {"E_over_E0": float(m.group(2)), "R_t": m.group(5), "min": float(m.group(6)), "pockets": int(m.group(7)), "inversions": int(m.group(8)),
                                "seconds": int(m.group(9)), "kept": m.group(10)}
    return recs


L1 = parse_log(os.path.join(RUN, "log_crash_0756.txt")); L2 = parse_log(os.path.join(RUN, "log.txt"))
out["restart_vs_crashrun"] = [{"T": k, "crash_run": L1[k], "end_run": L2.get(k)} for k in sorted(L1, key=float)]
restart_lines = []
for nm in ("log_crash_0756.txt", "log.txt"):
    for s in open(os.path.join(RUN, nm), errors="replace"):
        if s.startswith("version 34 (a): restarted"):
            m = re.search(r"\(T/rs (\S+), record (\d+)\).*?the slice's applied fields against the stored row (\S+); per channel the recomputed eigenvalues against the stored ones (.*?); the state", s)
            restart_lines.append({"log": nm, "T": m.group(1) if m else None, "applied_vs_stored": m.group(3) if m else None,
                                  "eig_max": (max(float(x.split(": ")[1]) for x in m.group(4).split(", ")) if m else None)})
out["restart_lines"] = restart_lines
out["certification"] = [{"T": t, "kept_residual": kept(RB[t])[0], "evaluations": kept(RB[t])[1], "meets_stop_rule": kept(RB[t])[2], "converged": RB[t].get("converged"),
                         "continuation": kept(RB[t])[3]} for t in Ts if t >= 4.0 - 1e-9]
out["end_v32"] = J.get("end_v32")

# ---------- (b): the 16a at the matched clocks
R16 = GC.recs_by_T(J16, rs); E016 = float(J16["head"]["E0"]); g16 = float(J16["args"]["g"]); b16 = GC.birth(J16, rs)
m16 = []
for t in (4.8125, 4.875, 4.9375, 5.0, 5.0625, 5.5625):
    r = R16.get(t)
    if r is None:
        continue
    r8 = RB.get(t)
    m16.append({"T": t, "E_over_E0_16": r["E"] / E016, "R_t_16": r.get("R_t"), "two_gE_16": 2 * g16 * r["E"], "R_t_minus_2gE_16": (r["R_t"] - 2 * g16 * r["E"]) if r.get("R_t") else None,
                "kept_16": (r.get("solve_v32") or {}).get("kept_residual"), "certified_16": bool(r.get("converged")) and bool((r.get("solve_v32") or {}).get("meets_stop_rule", True)),
                "E_over_E0_8": (r8["E"] / E0 if r8 else None), "R_t_8": (r8.get("R_t") if r8 else None), "two_gE_8": (2 * g * r8["E"] if r8 else None),
                "certified_8": (bool(r8.get("converged")) and bool((r8.get("solve_v32") or {}).get("meets_stop_rule")) if r8 else None)})
tl, th = b16["T_lo"], b16["T_hi"]; El, Eh = R16[round(tl, 6)]["E"], R16[round(th, 6)]["E"]; Eb16 = El + (Eh - El) * (b16["T_birth"] - tl) / (th - tl)
out["s16"] = {"E0": E016, "birth": b16, "E_at_birth_over_E0": Eb16 / E016, "R_b_over_2gE_Tb": b16["R_birth"] / (2 * g16 * Eb16), "matched": m16,
              "last_record": max(R16), "crossings": [crossing([{"T": t, "dev": 1 - R16[t]["E"] / E016} for t in sorted(R16)], x) for x in (1e-2, 3e-2, 6e-2)]}

# ---------- (e): slice inventory and wall time
sl = [(t, RB[t].get("slice_saved")) for t in Ts if RB[t].get("slice_saved")]
mir = "<internal>/%s" % TAG8
mirrored = {}
for d in sorted(glob.glob(os.path.join(mir, "T*"))):
    nm = os.path.basename(d); m5 = {}
    if os.path.exists(os.path.join(d, "MD5SUMS")):
        for s in open(os.path.join(d, "MD5SUMS")):
            p = s.split()
            if len(p) == 2 and "orbitals" in p[1]:
                m5 = p[0]
    mirrored[nm] = {"orbitals_md5_listed": m5, "RESUMABLE": open(os.path.join(d, "RESUMABLE")).read().strip() if os.path.exists(os.path.join(d, "RESUMABLE")) else None,
                    "not_a_restart_point": os.path.exists(os.path.join(d, "NOT_A_RESTART_POINT.txt"))}
out["slices"] = {"written_per_json": [{"T": t, "path": p} for t, p in sl], "box_mirror": mirrored}
wall = []
for k in sorted(L2, key=float):
    wall.append({"T": float(k), "seconds_end_run_cache_on": L2[k]["seconds"], "seconds_crash_run_cache_off": (L1.get(k) or {}).get("seconds")})
out["wall"] = wall
pre_rec = [RB[t].get("seconds") for t in Ts if 4.0 - 1e-9 <= t <= 4.5 + 1e-9]
out["wall_pre_gate_4.0_4.5"] = pre_rec
json.dump(out, open(OUTJ, "w"), indent=1, default=str)
print("wrote", OUTJ, flush=True)
