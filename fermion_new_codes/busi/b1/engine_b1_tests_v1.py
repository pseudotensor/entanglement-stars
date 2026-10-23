#!/usr/bin/env python
"""engine_b1_tests_v1.py (busi lane, B1 build): the self-tests of B1_design_reviewa_v1.md section 4 with the review's amendments, and the D2 and
D4 decision rules as code. Subcommands (each prints PASS/FAIL lines and appends to --md, default B1_selftests_<date>.md in --out):

  t41      4.1 the copy regression: engine_collapse_v19 and engine_b1_v1 --light off (--io v19 and --io chunked), identical arguments; E, R_t,
           N_min per record and every fields array bitwise (np.array_equal).
  light    the light module's gates on a standalone driver (no heavy engine; the light set fed the fields directly): 4.2 the conventions (Chan
           against chain_sparse, the audits, Delta_0's size and smoothness, the two density routes, M8's sector-entropy known answer), 4.3 the
           flat gate (N = 1, v = 0, E = 0), 4.3b the held-fields gate (the initial star's fields from prepare_dust held), 4.8 the purity identity
           at the readouts, and the light's energy closure on a driven small chain (fields ramped per record: the in-energy change against the
           exact flow, Re Tr[h C] conserved under each record's frozen h).
  coupled  a small coupled run and its no-light control (engine_b1_v1 --light on/off at reduced size): the run's ledger (E_end, the M1 light
           closures, the lag residual, the stop rule) and the control's |E - E_0|/E_0.
  resume   the b1 state's resume: a coupled run continuous against the same run resumed from its saved state; E per record bitwise.
  d2       the D2 go/no-go table from the card runs (criteria 1 to 7 of the design's sec. 5 and the review's D2 AND D4).
  d4       the D4 rule from the coupled run's first 200 r_s (the review's M4): the fits, the three t_P values, the branch.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_b1_v1 as B1      # noqa: E402  (sets the import paths)
import numpy as np             # noqa: E402

RESULTS = []


def say(md, line):
    print(line, flush=True)
    if md:
        with open(md, "a") as fh:
            fh.write(line + "\n")


def verdict(md, name, ok, detail):
    RESULTS.append((name, ok))
    say(md, "- %s: **%s** %s" % (name, "PASS" if ok else "FAIL", detail))


def py():
    return sys.executable


# ---------------------------------------------------------------------------------------------------------------- 4.1
def t41(a, md):
    v19 = os.path.join(a.engine_dir, "engine_collapse_v19.py")
    common = ["--m", str(a.m), "--g", "0.25", "--rs", str(a.rs), "--R0-over-rs", "4", "--nsites", str(a.nsites), "--ell", "2", "--kappa-sea", str(a.kappa_sea),
              "--tau", "1.05", "--T-over-rs", str(a.T_over_rs), "--out", a.out]
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([a.engine_dir] + [p for p in env.get("PYTHONPATH", "").split(":") if p])
    runs = [("v19", [py(), v19] + common + ["--tag", "t41_v19"]),
            ("b1_io_v19", [py(), os.path.join(HERE, "engine_b1_v1.py")] + common + ["--tag", "t41_b1v", "--light", "off", "--io", "v19"]),
            ("b1_io_chunked", [py(), os.path.join(HERE, "engine_b1_v1.py")] + common + ["--tag", "t41_b1c", "--light", "off", "--io", "chunked", "--fields-chunk-records", "8",
                                                                                          "--b1-save-every-rs", "0"])]
    for lab, cmd in runs:
        t0 = time.time()
        with open(os.path.join(a.out, "t41_%s.log" % lab), "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        say(md, "  4.1 run %s: rc %d (%.0f s)" % (lab, rc, time.time() - t0))
        if rc != 0:
            verdict(md, "4.1 copy regression", False, "the %s run failed (log t41_%s.log)" % (lab, lab))
            return False
    J0 = json.load(open(os.path.join(a.out, "collapse_t41_v19.json")))["records"]
    Z0 = np.load(os.path.join(a.out, "collapse_t41_v19_fields.npz"), allow_pickle=True)
    ok_all = True
    for tag, layout in (("t41_b1v", "v19"), ("t41_b1c", "chunked")):
        J1 = B1.load_records(a.out, tag)
        n = len(J0)
        same_n = (len(J1) == n)
        bad = []
        for r0, r1 in zip(J0, J1):
            for k_ in ("T", "E", "R_t", "N_min", "min_N_minus_absv", "iterations", "H_in", "M1_I_E"):
                if r0.get(k_) != r1.get(k_):
                    bad.append((r0["T"], k_, r0.get(k_), r1.get(k_)))
        if layout == "v19":
            Z1 = {k_: v_ for k_, v_ in np.load(os.path.join(a.out, "collapse_%s_fields.npz" % tag), allow_pickle=True).items()}
        else:
            Z1 = {}
            for p in sorted(glob.glob(os.path.join(a.out, "collapse_%s_fields_*.npz" % tag))):
                z = np.load(p, allow_pickle=True)
                for k_ in z.files:
                    if k_ in ("R", "N_cinit", "v_cinit", "g", "ell", "tau", "solver", "label"):
                        Z1.setdefault(k_, z[k_])
                    else:
                        Z1.setdefault(k_, []).append(z[k_])
            for k_ in list(Z1):
                if isinstance(Z1[k_], list):
                    Z1[k_] = np.concatenate(Z1[k_]) if Z1[k_][0].dtype != object else np.concatenate([np.asarray(x, dtype=object) for x in Z1[k_]])
        arr_bad = []
        for k_ in ("R", "T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "N_cinit", "v_cinit"):
            if not np.array_equal(np.asarray(Z0[k_]), np.asarray(Z1[k_])):
                arr_bad.append(k_)
        for k_ in ("r_nodes", "N_nodes", "v_nodes"):
            x0, x1 = Z0[k_], Z1[k_]
            if len(x0) != len(x1) or not all(np.array_equal(np.asarray(p, float), np.asarray(q, float)) for p, q in zip(x0, x1)):
                arr_bad.append(k_)
        ok = same_n and not bad and not arr_bad
        ok_all = ok_all and ok
        verdict(md, "4.1 copy regression (%s layout)" % layout, ok, "records %d vs %d; record keys differing %d%s; arrays differing %s (E/E0 last %.10f)" % (
            n, len(J1), len(bad), (" first %s" % (bad[0],)) if bad else "", arr_bad or "none", J0[-1]["E"] / J0[0]["E"]))
    return ok_all


# ---------------------------------------------------------------------------------------------------------------- the light gates
def light_args(over):
    a = B1.build_parser().parse_args(["--out", over.get("out", "/tmp"), "--tag", "x"])
    for k_, v_ in over.items():
        setattr(a, k_, v_)
    return a


def light_gates(a, md):
    import torch
    import engine_b1_light_v1 as B1L
    B1.C5.backend(None if a.device == "cpu" else int(a.device.split(":")[1]))
    torch.set_num_threads(a.threads)
    E19 = B1.E19
    n_e = a.nsites
    R = np.arange(n_e, dtype=float) + 1.0
    ok_all = True
    base = {"out": a.out, "g": 0.25, "rs": a.rs, "tau": 1.05, "light_nsites": a.n_L, "light_nv": 0, "light_pad": 128, "light_kappas": "1,2,3",
            "light_guard_kappas": "4", "light_kappa_zero": "on", "light_init": "device", "light_init_cache": None, "light_threads": a.threads,
            "light_cuts_rs": "2,4,8,16", "light_comparator": "off", "c_LZ": 1.0, "window_max": 4}
    # ---- 4.2 and 4.3: the flat gate
    la = light_args(base)
    L = B1L.LightSet(la, R, [], E19)
    f0 = (np.ones(n_e), np.zeros(n_e))
    head = {}
    conv = L.start(f0, 0.0, head)
    for k_, cc in conv["channels"].items():
        sec = max(abs(v_["diff"]) for v_ in cc["sector_entropy_known_answer"].values())
        okc = (cc["Chan_vs_chain_sparse_h_max"] <= 1e-15 and cc["Chan_vs_chain_sparse_Q_max"] <= 1e-15 and cc["audit_h0"] and cc["audit_Q"] and cc["audit_h_T0"]
               and cc["Delta0_max_abs"] <= 1e-6 and cc["density_routes_max_abs"] <= 1e-12 and sec <= 1e-10)
        ok_all = ok_all and okc
        verdict(md, "4.2 conventions kappa %s (flat T_0)" % k_, okc, "Chan vs chain_sparse h %.1e Q %.1e; audits %s %s %s; Delta_0 max %.2e (first sites %s, last %s); density routes %.1e; "
                "M8 sector entropy max |2 S(C_+) - S(full)| %.1e; Delta S_0 %s; n_+ %d; T_init %s" % (
                    cc["Chan_vs_chain_sparse_h_max"], cc["Chan_vs_chain_sparse_Q_max"], cc["audit_h0"], cc["audit_Q"], cc["audit_h_T0"], cc["Delta0_max_abs"],
                    ["%.1e" % x for x in cc["Delta0_first8"][:3]], ["%.1e" % x for x in cc["Delta0_last8"][-3:]], cc["density_routes_max_abs"], sec,
                    {kk: "%.2e" % vv for kk, vv in cc["dS0_at_cuts"].items()}, cc["n_plus"], cc["T_init_impurity"]))
    ok_flat, worst = run_light_records(L, f0, 0.0, a.records, n_e, md, "4.3 flat gate", eps_tol=1e-10, dS_tol=1e-6, flow_tol=1e-10, E0_ledger=1.0)
    ok_all = ok_all and ok_flat
    # ---- 4.3b: the held-fields gate on the initial star's fields (prepare_dust at the B1 point, reduced size)
    chs = [E19.Chan(k, a.m, n_e, False) for k in range(1, a.kappa_sea + 1)]
    B1.OPT["site_fields"] = "averaged"
    N, v, dust, info, hd = E19.prepare_dust(chs, 0.25, R, 2, a.rs / (2 * 0.25 * a.m), 4 * a.rs, a.m, "A", None)
    E0 = hd["E0"]
    say(md, "  4.3b fields: prepare_dust at m %g, r_s %g, %d sites: E0 %.6f, min N %.4f, max |v| %.3e" % (a.m, a.rs, n_e, E0, N.min(), np.abs(v).max()))
    L2 = B1L.LightSet(light_args(base), R, chs, E19)
    conv2 = L2.start((N, v), E0, {})
    for k_, cc in conv2["channels"].items():
        okc = cc["audit_h_T0"] and cc["Delta0_max_abs"] <= 1e-6 and cc["density_routes_max_abs"] <= 1e-12
        ok_all = ok_all and okc
        verdict(md, "4.2 conventions kappa %s (the star's T_0 fields)" % k_, okc, "Delta_0 max %.2e (sum %.2e); density routes %.1e; audit h(T0) %s" % (
            cc["Delta0_max_abs"], cc["Delta0_sum"], cc["density_routes_max_abs"], cc["audit_h_T0"]))
    ok_held, _ = run_light_records(L2, (N, v), E0, a.records, n_e, md, "4.3b held-fields gate", eps_tol=1e-9, dS_tol=1e-5, flow_tol=1e-9, E0_ledger=E0)
    ok_all = ok_all and ok_held
    # ---- the light's energy closure on a driven chain: the star's fields deepened by a ramp per record (radiation produced)
    L3 = B1L.LightSet(light_args(base), R, chs, E19)
    L3.start((N, v), E0, {})
    ok_dr = driven_closure(L3, (N, v), E0, max(20, a.records // 5), n_e, md)
    ok_all = ok_all and ok_dr
    return ok_all


def run_light_records(L, f, E, nrec, n_e, md, name, eps_tol, dS_tol, flow_tol, E0_ledger):
    cut = n_e - 10
    worst = {"eps": 0.0, "flow": 0.0, "dS": 0.0, "purity": 0.0, "closure": 0.0, "lag": 0.0}
    T = 0.0
    for k in range(nrec + 1):
        eps_l, J_l, dg = L.sources(f, E, T, k)
        worst["eps"] = max(worst["eps"], float(np.max(np.abs(eps_l))), max(abs(d_["E_shells"]) for d_ in dg["per_channel"].values()))
        worst["lag"] = max(worst["lag"], abs(dg["lag_residual_over_E0"]))
        if k > 0:
            L.m1_old(f, E, cut)
        m1 = L.m1_new(f, E, cut)
        b = m1.get("balance")
        if b and b.get("closure_light_evolution_plus_flow") is not None:
            worst["closure"] = max(worst["closure"], abs(b["closure_light_evolution_plus_flow"]))
        if k in (0, nrec // 2, nrec):
            ro = L.readout(T, E, None, None, f, k, None, "x", impurity=(k == nrec), contour=(k == nrec), rows=False)
            for kk, pc in ro["per_channel"].items():
                worst["purity"] = max(worst["purity"], pc["purity_residual"])
                for lab, d_ in pc["cuts"].items():
                    worst["dS"] = max(worst["dS"], abs(d_["excess"]))
                if pc.get("contour"):
                    say(md, "  %s kappa %s contour certificate: %s" % (name, kk, {lab: "%.1e" % d_["truncated_plus_dS0_minus_contour"] for lab, d_ in pc["contour"]["cuts"].items()}))
            if k == nrec:
                say(md, "  %s readout T %.1f: impurity %s; floors %s" % (name, T, {kk: "%.1e" % (pc["impurity"] or 0) for kk, pc in ro["per_channel"].items()},
                                                                       {kk: "%.1e" % pc["floor"]["T_cheb"] for kk, pc in ro["per_channel"].items()}))
        if k == nrec:
            break
        L.step(f, E, cut, 1.05, 2)
        for c in L.chans:
            for lab, v_ in c.last_flow.items():
                worst["flow"] = max(worst["flow"], abs(c.w * v_) if c.kappa else abs(v_))
        T += 1.05
    worst["E_rad"] = abs(L.E_rad_acc)
    ok = worst["eps"] <= eps_tol and worst["dS"] <= dS_tol and worst["flow"] <= flow_tol and worst["E_rad"] <= flow_tol * max(1, nrec) and worst["purity"] <= 1e-9
    verdict(md, name, ok, "%d records: max |eps^L| per site %.1e (tol %.0e); max |excess| %.1e nats (tol %.0e); max |weighted flow| per record %.1e, |E_rad| %.1e (tol %.0e); "
            "purity residual %.1e; light closure (evolution + flow) %.1e; lag residual %.1e" % (
                nrec, worst["eps"], eps_tol, worst["dS"], dS_tol, worst["flow"], worst["E_rad"], flow_tol, worst["purity"], worst["closure"], worst["lag"]))
    return ok, worst


def driven_closure(L, f0, E0, nrec, n_e, md):
    """fields deepened per record (N -> 1 - s (1 - N), v -> s v with s growing 2 percent per record): radiation leaves; per record and channel
    the inside Killing energy's change under the frozen h (Y1 - X1_prev) against minus the exact flow through the M1 cut; the total
    Re Tr[h C] over the chain conserved under the frozen h (the flows' suffix sum at the first site)."""
    cut = n_e - 10
    T = 0.0; worst = 0.0; worst_tot = 0.0; scale = 0.0; E = E0
    for k in range(nrec + 1):
        s = 1.0 + 0.02 * k
        f = (1.0 - s * (1.0 - f0[0]), s * f0[1])
        L.sources(f, E, T, k)
        if k > 0:
            L.m1_old(fprev, E, cut)
        m1 = L.m1_new(f, E, cut)
        for kk, d_ in m1["per_channel"].items():
            b = d_.get("balance")
            if b and b.get("evolution_plus_flow") is not None:
                worst = max(worst, abs(b["evolution_plus_flow"])); scale = max(scale, abs(b["evolution_exact"]))
        if k == nrec:
            break
        L.step(f, E, cut, 1.05, 2)
        for c in L.chans:
            worst_tot = max(worst_tot, abs(c.last_flow.get("total", 0.0)))
        fprev = f
        T += 1.05
    ok = worst <= 1e-9 * max(1.0, scale) + 1e-11
    verdict(md, "light energy closure on a driven chain", ok, "%d records, fields ramped 2 percent per record: max |evolution_exact + flow_M1cut| %.1e against max |evolution| %.1e; E_rad %.3e" % (
        nrec, worst, scale, L.E_rad_acc))
    return ok


# ---------------------------------------------------------------------------------------------------------------- coupled and resume
def b1_cmd(a, tag, extra):
    return [py(), os.path.join(HERE, "engine_b1_v1.py"), "--m", str(a.m), "--g", "0.25", "--rs", str(a.rs), "--R0-over-rs", "4", "--nsites", str(a.nsites),
            "--ell", "2", "--kappa-sea", str(a.kappa_sea), "--tau", "1.05", "--T-over-rs", str(a.T_over_rs), "--out", a.out, "--tag", tag] + extra


def run_cmd(a, md, lab, cmd):
    t0 = time.time()
    with open(os.path.join(a.out, "%s.log" % lab), "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    say(md, "  run %s: rc %d (%.0f s)" % (lab, rc, time.time() - t0))
    return rc


def coupled(a, md):
    lightopts = ["--light", "on", "--light-nsites", str(a.n_L), "--light-threads", str(a.threads), "--light-readout-every-rs", "1", "--light-impurity-every-rs", "2",
                 "--light-contour-every-rs", "0", "--light-rows-every-rs", "0", "--light-comparator", a.comparator, "--io", "chunked", "--fields-chunk-records", "16",
                 "--b1-save-every-rs", "0", "--light-kappa-zero", "on"]
    rc1 = run_cmd(a, md, "coupled_on", b1_cmd(a, "cpl_on", lightopts))
    rc0 = run_cmd(a, md, "coupled_off", b1_cmd(a, "cpl_off", ["--light", "off", "--io", "chunked", "--b1-save-every-rs", "0"]))
    ok = (rc1 == 0 and rc0 == 0)
    if not ok:
        verdict(md, "coupled small run", False, "a run failed (logs coupled_on.log, coupled_off.log)")
        return False
    R1 = [r_ for r_ in B1.load_records(a.out, "cpl_on") if r_.get("E") is not None]
    R0 = [r_ for r_ in B1.load_records(a.out, "cpl_off") if r_.get("E") is not None]
    E0 = R1[0]["E"]
    dE0 = max(abs(r_["E"] - R0[0]["E"]) / R0[0]["E"] for r_ in R0)
    Ee = np.array([r_["light"]["E_end_vac"] for r_ in R1]); Ec = np.array([r_["E"] for r_ in R1])
    dEend = float(np.max(np.abs(Ee - Ee[0])) / E0)
    lag = max(abs(r_["light"].get("lag_residual_over_E0") or 0.0) for r_ in R1)
    clos = [((r_.get("M1_light") or {}).get("balance") or {}).get("closure_light_evolution_plus_flow") for r_ in R1]
    clos = [abs(x) for x in clos if x is not None]
    stops = [r_["light"].get("stop_reason") for r_ in R1 if r_["light"].get("stop_reason")]
    conv = sum(1 for r_ in R1 if r_.get("converged")) / len(R1)
    ro = [r_["light"]["readout"] for r_ in R1 if r_.get("light", {}).get("readout")]
    pur = max((pc["purity_residual"] for x in ro for pc in x["per_channel"].values()), default=None)
    exc = ro[-1]["cuts"] if ro else {}
    walls = [r_.get("b1_wall_s") for r_ in R1 if r_.get("b1_wall_s")]
    tm = R1[-1].get("light", {}).get("timing", {})
    verdict(md, "no-light control (reduced size)", dE0 <= 1e-4, "%d records to T/r_s %.2f: max |E - E_0|/E_0 %.2e (D2 criterion 1's threshold 1e-4 per 100 r_s, here over %.1f r_s); converged %d of %d" % (
        len(R0), R0[-1]["T_over_rs"], dE0, R0[-1]["T_over_rs"], sum(1 for r_ in R0 if r_.get("converged")), len(R0)))
    okc = (not stops) and (max(clos) if clos else 0.0) <= 1e-8 and pur is not None and pur <= 1e-9
    verdict(md, "coupled small run (light on, n_L %d)" % a.n_L, okc, "%d records to T/r_s %.2f: converged %.2f; E/E_0 last %.8f; max |E_end - E_end(0)|/E_0 %.2e; light closure max %.1e; "
            "lag residual max %.1e E_0; stop rule %s; purity max %s; last readout excess (weighted) %s; capacities 2r_s %s; wall %.1f s/record (light prop %.2f s, ref %.2f s, flow %.2f s, stress %.2f s, m1 %.2f s)" % (
                len(R1), R1[-1]["T_over_rs"], conv, Ec[-1] / E0, dEend, max(clos) if clos else float("nan"), lag, stops or "not tripped", ("%.1e" % pur) if pur is not None else None,
                {k_: "%.3e" % v_["excess"] for k_, v_ in exc.items()}, ("%.0f" % exc["2"]["capacity_ln2"]) if "2" in exc else "-", float(np.mean(walls)) if walls else float("nan"),
                tm.get("light_prop_s", 0), tm.get("light_ref_s", 0), tm.get("light_flow_s", 0), tm.get("light_stress_s", 0), tm.get("light_m1_s", 0)))
    return ok and okc


def resume_test(a, md):
    lightopts = ["--light", "on", "--light-nsites", str(a.n_L), "--light-threads", str(a.threads), "--light-readout-every-rs", "1", "--light-contour-every-rs", "0",
                 "--light-rows-every-rs", "0", "--light-comparator", "off", "--io", "chunked", "--fields-chunk-records", "16"]
    half = round(a.T_over_rs / 2.0 / (1.05 / a.rs)) * (1.05 / a.rs)
    rc1 = run_cmd(a, md, "res_full", b1_cmd(a, "res_full", lightopts + ["--b1-save-every-rs", "%.6f" % half]))
    sts = sorted(glob.glob(os.path.join(a.out, "collapse_res_full_b1state_T*.pt")))
    if rc1 != 0 or not sts:
        verdict(md, "resume (b1 state)", False, "the full run failed or saved no state")
        return False
    rc2 = run_cmd(a, md, "res_resumed", b1_cmd(a, "res_resumed", lightopts + ["--b1-save-every-rs", "0", "--b1-resume", sts[0]]))
    A = {round(r_["T"], 6): r_ for r_ in B1.load_records(a.out, "res_full") if r_.get("E") is not None}
    Bq = [r_ for r_ in B1.load_records(a.out, "res_resumed") if r_.get("E") is not None]
    diffs = [abs(A[round(r_["T"], 6)]["E"] - r_["E"]) for r_ in Bq if round(r_["T"], 6) in A]
    dl = [abs(A[round(r_["T"], 6)]["light"]["E_end_vac"] - r_["light"]["E_end_vac"]) for r_ in Bq if round(r_["T"], 6) in A]
    ok = rc2 == 0 and len(diffs) >= 2 and max(diffs) == 0.0 and max(dl) == 0.0
    verdict(md, "resume (b1 state)", ok, "resumed from %s: %d records compared, max |dE| %s, max |dE_end| %s" % (os.path.basename(sts[0]), len(diffs), max(diffs) if diffs else None, max(dl) if dl else None))
    return ok


# ---------------------------------------------------------------------------------------------------------------- D2 and D4
def recs_of(path):
    d, tag = os.path.split(path)
    return [r_ for r_ in B1.load_records(d, tag) if r_.get("E") is not None]


def d2(a, md):
    """the D2 table: criterion 1 (no-light control), 2 (coupled convergence), 3 (wall per r_s0 at the production schedule, M3 d), 4 (coupled
    conservation, M2), 5 (the build gates, from --gates PASS/FAIL), 6 (the fed-light ledger after 40 r_s), 7 (the guard's share)."""
    rows = []; go = True
    nl = recs_of(a.nolight) if a.nolight else None
    if nl:
        E0 = nl[0]["E"]; T = np.array([r_["T_over_rs"] for r_ in nl]); E = np.array([r_["E"] for r_ in nl])
        f1 = float(np.max(np.abs(E - E0)) / E0); f1a = float(np.max(np.abs(E[T <= 40] - E0)) / E0)
        late = T >= 40
        f1b = float((np.max(E[late]) - np.min(E[late])) / E0) if late.any() else None
        ok1 = (f1b is not None and f1b <= 1e-4 and f1a <= 1e-3)
        rows.append(("1 no-light control", ok1, "0-40 r_s max |E - E_0|/E_0 %.2e (<= 1e-3); 40-100 r_s spread %s (<= 1e-4); over the run %.2e; %d records" % (
            f1a, ("%.2e" % f1b) if f1b is not None else "-", f1, len(nl))))
        go = go and ok1
    cp = recs_of(a.coupled) if a.coupled else None
    if cp:
        n = len(cp); T = np.array([r_["T_over_rs"] for r_ in cp])
        conv = np.array([bool(r_.get("converged")) for r_ in cp])
        res = [((r_.get("unconverged") or {}).get("residual") or 0.0) for r_ in cp]
        run3 = any(not conv[i] and not conv[i + 1] and not conv[i + 2] for i in range(n - 2))
        ok2 = conv[:800].mean() >= 0.95 and conv[T > 40].all() and max(res) <= 1e-5 and not run3
        rows.append(("2 coupled convergence", ok2, "converged %.3f of records 1-800, all after 40 r_s %s, max unconverged residual %.1e, three in a row %s" % (
            conv[:800].mean(), bool(conv[T > 40].all()), max(res), run3)))
        go = go and ok2
        w = np.array([r_.get("b1_wall_s") or np.nan for r_ in cp]); tau = np.array([r_.get("tau_record") or 1.05 for r_ in cp])
        sel = (np.arange(n) >= 320) & (np.arange(n) < 800)
        if sel.any():
            rs = a.rs0
            sec_per_rs = float(np.nanmean(w[sel] / tau[sel]) * rs)
            g3 = "GO-A" if sec_per_rs <= 24 else ("GO-B" if sec_per_rs <= 48 else "GO-C")
            rows.append(("3 wall", True, "mean over records 320-800: %.2f s per record, %.1f s per r_s0 of physical time (GO-A <= 24, GO-B <= 48): %s" % (
                float(np.nanmean(w[sel])), sec_per_rs, g3)))
        E0 = cp[0]["E"]
        Ee = np.array([r_["light"]["E_end_vac"] for r_ in cp])
        stops = [r_["light"].get("stop_reason") for r_ in cp if r_["light"].get("stop_reason")]
        form = T <= 40
        a1 = float(np.max(np.abs(Ee[form] - E0)) / E0)
        late = (T >= 40) & (T <= 100)
        drift = float((Ee[late].max() - Ee[late].min()) / E0) if late.any() else None
        clos = [abs(((r_.get("M1_light") or {}).get("balance") or {}).get("closure_light_evolution_plus_flow") or 0.0) for r_ in cp]
        lag = [abs(r_["light"].get("lag_residual_over_E0") or 0.0) for r_ in cp if r_["T_over_rs"] > 40]
        ctrl = None
        if a.ref_iterate:
            ri = recs_of(a.ref_iterate)
            byT = {round(r_["T_over_rs"], 4): r_["E"] for r_ in cp}
            m = [abs(r_["E"] - byT[round(r_["T_over_rs"], 4)]) / E0 for r_ in ri if round(r_["T_over_rs"], 4) in byT and r_["T_over_rs"] <= 20.0001]
            ctrl = max(m) if m else None
        ok4 = (not stops and a1 <= 1e-3 and (drift is None or drift <= 1e-4) and max(clos) <= 1e-6 * E0 and (max(lag) if lag else 0.0) <= 1e-3
               and (ctrl is None or ctrl <= 1e-3))
        rows.append(("4 coupled conservation (M2)", ok4, "stop rule %s; 0-40 r_s max |E_end - E_0|/E_0 %.2e (<= 1e-3); 40-100 r_s E_end drift %s (<= 1e-4); light closure max %.1e (<= 1e-6 E_0); "
                     "lag residual after 40 r_s max %.1e E_0 (<= 1e-3); the 4.11 control |dE(<= 20 r_s)| %s (<= 1e-3 E_0)" % (
                         stops[:1] or "not tripped", a1, ("%.2e" % drift) if drift is not None else "-", max(clos), max(lag) if lag else 0.0, ("%.1e" % ctrl) if ctrl is not None else "not run")))
        go = go and ok4
        led = [abs(r_["light"].get("ledger_Hin_plus_Erad_over_E0") or 0.0) for r_ in cp if r_["T_over_rs"] > 40]
        ins = [r_["light"].get("E_shells_inside_root") for r_ in cp]
        pk = max((sum((d_.get("n_pocket") or 0) for d_ in r_["light"]["per_channel"].values()) for r_ in cp), default=0)
        ok6 = (max(led) if led else 0.0) <= 1e-3
        rows.append(("6 fed-light ledger", ok6, "after 40 r_s max |H_in^L + E_rad|/E_0 %.2e (<= 1e-3); the bare inside-root total range %s; pockets max %d" % (
            max(led) if led else 0.0, ("[%.3e, %.3e]" % (min(x for x in ins if x is not None), max(x for x in ins if x is not None))) if any(x is not None for x in ins) else "-", pk)))
        go = go and ok6
        gs = [r_["light"].get("guard_share_collar_flow") for r_ in cp if r_["T_over_rs"] <= 100 and r_["light"].get("guard_share_collar_flow") is not None]
        if gs:
            ok7 = abs(gs[-1]) <= 0.02
            rows.append(("7 guard share", ok7, "kappa 4's share of the collar flow over 0-100 r_s %.3f (<= 0.02)" % gs[-1]))
            go = go and ok7
    if a.gates:
        g = dict(x.split("=") for x in a.gates.split(","))
        ok5 = all(v_.upper() == "PASS" for v_ in g.values())
        rows.append(("5 build gates", ok5, ", ".join("%s %s" % kv for kv in g.items())))
        go = go and ok5
    say(md, "\n## D2 table (%s)\n" % time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    say(md, "| criterion | pass | numbers |\n|---|---|---|")
    for name, ok, txt in rows:
        say(md, "| %s | %s | %s |" % (name, "PASS" if ok else "FAIL", txt))
    w3 = [r_ for r_ in rows if r_[0].startswith("3")]
    say(md, "\nD2 verdict: %s" % (("GO (%s)" % w3[0][2].split(": ")[-1]) if go and w3 else ("NO-GO" if not go else "GO pending the wall row")))
    return go


def d4(a, md):
    """M4: from the coupled run's records over 50-200 r_s, the flow at the 4 r_s0 cut fitted as L(t) = L_inf + (L_50 - L_inf) exp(-(t - 50)/t_d) and as
    a power law; L_inf/L_H at T_H and at kappa_T/(2 pi); beta_eff over the window from the 4 r_s0 cut's excess against its flow; t_P by (i) the
    thermal rate rescaled, (iii) the measured beta, (iv) the interior's own rate; the branch: no sustained evaporation (L_inf < 0.3 L_H, or t_d <
    1000 r_s0 with L_inf consistent with zero) -> Outcome VI, run to 500 r_s0 and stop; else the chain length by t_P(iv) against 3000 r_s0."""
    from scipy.optimize import curve_fit
    cp = recs_of(a.coupled)
    rs = a.rs0; g = 0.25
    T = np.array([r_["T_over_rs"] for r_ in cp]); tau = np.array([r_.get("tau_record") or 1.05 for r_ in cp])
    f4 = np.array([r_["light"].get("flow_cut_4", np.nan) for r_ in cp])
    L4 = f4 / tau
    sel = (T >= 50) & (T <= 200)
    E = np.array([r_["E"] for r_ in cp]); E0 = E[0]
    if sel.sum() < 20:
        say(md, "D4: too few records in 50-200 r_s (%d)" % sel.sum())
        return None
    t = T[sel]; y = L4[sel]
    bins = np.arange(50, 201, 5.0)
    tb = []; yb = []
    for lo in bins[:-1]:
        m = (t >= lo) & (t < lo + 5)
        if m.any():
            tb.append(lo + 2.5); yb.append(float(np.mean(y[m])))
    tb = np.array(tb); yb = np.array(yb)
    expf = lambda x, Li, L50, td: Li + (L50 - Li) * np.exp(-(x - 50.0) / td)
    try:
        pe, ce = curve_fit(expf, tb, yb, p0=[yb[-1], yb[0], 100.0], maxfev=20000)
        se = np.sqrt(np.diag(ce)); res_e = float(np.sqrt(np.mean((expf(tb, *pe) - yb) ** 2)))
    except Exception:                                                     # noqa: BLE001
        pe, se, res_e = [np.nan] * 3, [np.nan] * 3, np.nan
    pos = yb > 0
    try:
        pp = np.polyfit(np.log(tb[pos]), np.log(yb[pos]), 1); res_p = float(np.sqrt(np.mean((np.exp(np.polyval(pp, np.log(tb[pos]))) - yb[pos]) ** 2)))
    except Exception:                                                     # noqa: BLE001
        pp, res_p = [np.nan, np.nan], np.nan
    Ef = float(np.mean(E[sel])); TH = 1.0 / (8 * math.pi * g * Ef)
    ros = [r_["light"]["readout"] for r_ in cp if r_.get("light", {}).get("readout") and 50 <= r_["T_over_rs"] <= 200]
    LH = np.nanmean([x.get("L_H_TH") or np.nan for x in ros]) if ros else np.nan
    LHk = np.nanmean([x.get("L_H_kappaT") or np.nan for x in ros]) if ros else np.nan
    Li = float(pe[0])
    beta = None
    if len(ros) >= 2:
        dS = ros[-1]["cuts"].get("4", {}).get("excess", np.nan) - ros[0]["cuts"].get("4", {}).get("excess", np.nan)
        T0r, T1r = ros[0]["T_over_rs"], ros[-1]["T_over_rs"]
        m = (T > T0r) & (T <= T1r)
        Eflow = float(np.sum(f4[m]))
        beta = dS * TH / Eflow if Eflow else None
    rs0 = 2 * g * E0
    S0 = 4 * math.pi * g * E0 ** 2
    ell_cal = LH * rs0 ** 2 if np.isfinite(LH) else np.nan
    t_star = rs0 ** 3 / (6 * g * ell_cal) if np.isfinite(ell_cal) and ell_cal > 0 else np.nan
    t_star_meas = t_star * (LH / Li) if (Li > 0 and np.isfinite(t_star)) else np.inf
    tP_i = 0.511 * t_star_meas
    tP_iii = t_star_meas * (1 - (beta / (1 + beta)) ** 1.5) if (beta and beta > 0) else None
    tP_iv = None
    if len(ros) >= 2:
        Sdot = (ros[-1]["cuts"].get("4", {}).get("excess", 0) - ros[0]["cuts"].get("4", {}).get("excess", 0)) / ((ros[-1]["T_over_rs"] - ros[0]["T_over_rs"]) * rs)
        if Sdot > 0 and np.isfinite(t_star_meas):
            ts = np.linspace(0, t_star_meas, 20001)
            cross = np.where(Sdot * ts >= S0 * (1 - ts / t_star_meas) ** (2.0 / 3.0))[0]
            tP_iv = float(ts[cross[0]]) if len(cross) else None
    none_branch = (np.isfinite(LH) and Li < 0.3 * LH) or (pe[2] < 1000 * 1.0 and abs(Li) <= 2 * (se[0] if np.isfinite(se[0]) else 0))
    branch = "Outcome VI (no sustained evaporation at this size): run on to 500 r_s0 for the drain's curve and stop" if none_branch else (
        "evaporation: the full run on the %s chain" % ("33,792-site" if (tP_iv or tP_i) / rs > 3000 else "16,896-site"))
    out = {"fit_exp": {"L_inf": Li, "L_50": float(pe[1]), "t_d_rs": float(pe[2]), "se": [float(x) for x in se], "rms": res_e}, "fit_power": {"slope": float(pp[0]), "rms": res_p},
           "L_inf_over_LH_TH": Li / LH if np.isfinite(LH) else None, "L_inf_over_LH_kappaT": Li / LHk if np.isfinite(LHk) else None, "beta_eff": beta,
           "t_star_rs": t_star_meas / rs if np.isfinite(t_star_meas) else None, "tP_i_rs": tP_i / rs if np.isfinite(tP_i) else None,
           "tP_iii_rs": (tP_iii / rs) if tP_iii else None, "tP_iv_rs": (tP_iv / rs) if tP_iv else None, "S0": S0, "branch": branch}
    say(md, "\n## D4 (%s)\n\n%s\n" % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), json.dumps(out, indent=1, default=B1._jdefault)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["t41", "light", "coupled", "resume", "d2", "d4", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--engine-dir", default="<internal>/compC_engine")
    ap.add_argument("--m", type=float, default=0.4)
    ap.add_argument("--rs", type=float, default=8.4)
    ap.add_argument("--nsites", type=int, default=512)
    ap.add_argument("--kappa-sea", type=int, default=4)
    ap.add_argument("--T-over-rs", type=float, default=2.5)
    ap.add_argument("--n-L", type=int, default=4096)
    ap.add_argument("--records", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--comparator", default="off")
    ap.add_argument("--nolight", default=None); ap.add_argument("--coupled", default=None); ap.add_argument("--ref-iterate", default=None)
    ap.add_argument("--gates", default=None, help="e.g. 4.1=PASS,4.2=PASS,4.3=PASS,4.3b=PASS")
    ap.add_argument("--rs0", type=float, default=8.44)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = a.md or os.path.join(a.out, "B1_selftests_%s.md" % time.strftime("%Y%m%d"))
    say(md, "\n### engine_b1_tests_v1 %s (%s UTC; host %s)\n" % (a.cmd, time.strftime("%Y-%m-%d %H:%M", time.gmtime()), os.uname()[1]))
    fn = {"t41": t41, "light": light_gates, "coupled": coupled, "resume": resume_test, "d2": d2, "d4": d4}
    if a.cmd == "all":
        for k_ in ("t41", "light", "coupled", "resume"):
            fn[k_](a, md)
    else:
        fn[a.cmd](a, md)
    say(md, "\nsummary: %s" % ", ".join("%s %s" % (n_, "PASS" if o_ else "FAIL") for n_, o_ in RESULTS))


if __name__ == "__main__":
    main()
