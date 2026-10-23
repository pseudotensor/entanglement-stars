#!/usr/bin/env python
"""engine_twofield_v5.py (the busi lane, Computation C): the replacement list that builds engine_twofield_v5 over engine_twofield_v4 (the built engine),
with the second sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v2.md, section 2); every change against version 4 is marked "v5:".
Build: python engine_twofield_v5.py [--src engine_twofield_v4.py] [--out engine_twofield_v5.py]; without arguments it reads engine_twofield_v4.py beside
this file and writes the engine over this file (gate_v5_s2_v1's ensure_built does the same). Each replacement's current text must occur exactly once in
the source (else the build stops and names it); the result is compiled before it is written. A version 4 trailing mark may follow a version 5 mark on
the few lines that extend a version 4 line.
The option (off by default; off, every code path and every record is version 4's, bitwise): --light-coupling-ramp S1,...,1 with --light-ramp-bisect B.
At a record whose direct coupled Newton solve misses the stop rule the record's fixed point is sought along the homotopy G_s = Solve(heavy sources +
s x light sources): the base stage s = 0 is the uncoupled map (the light channels not evaluated; the light sources absent from eps, J and the exact J.v;
the heavy sum bitwise version 35's) from version 31's start; each listed s is version 32's Newton stage on the coupled map with the light sources and
their exact J.v weighted by s (LS["s"]), from the previous stage's kept applied fields; a stage that misses the stop rule is retried over the half interval
from the last converged s, at most B times per target; a stall is recorded with its interval (the fold readout) and a closing stage at s = 1 runs from the
last converged iterate; a base that misses the stop rule is not walked, only closed. Only an s = 1 stage can be kept, by version 32's kept rule (the
smaller kept residual of the direct solve, version 32's stage 2 where it ran and the s = 1 stage); G, the stop rule, the end rule, the records' content
and the propagation are unchanged, so the fixed point is unchanged. Per stage the record carries s, the status, the kept and confirming residuals, the
largest trial displacement (v4's readout), the kept applied fields' change from the stage's start with its field and site, and the central cell at the
kept iterate (N_dense(0), W/2g at r = 1, 2 and ell, rho_ell(0), J_ell/v_dense at r = 1). The first record is written when its kept residual is below
--end-above, the production's practice on its sigma/a 2 member (labelled; nothing else about it changes)."""
import argparse
import os
import sys


def S(x):
    """a triple-quoted block without its leading newline."""
    return x[1:] if x.startswith("\n") else x


REPLACEMENTS = [
    # 1. the docstring head
    ('''"""engine_twofield_v4.py (the busi lane, Computation C): version 3 with the sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v1.md);''',
     S('''
"""engine_twofield_v5.py (the busi lane, Computation C): version 4 with the second sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v2.md);
every change against version 4 is marked "v5:". One option, --light-coupling-ramp S1,...,1 with --light-ramp-bisect B (default off; off, every code path and
every record is version 4's, bitwise): at a record whose direct coupled Newton solve misses the stop rule, the record's fixed point is sought along the
homotopy G_s = Solve(heavy sources + s x light sources): the base stage s = 0 is the uncoupled map (the light channels not evaluated; the heavy sum bitwise
version 35's) from version 31's start; each listed s is version 32's Newton stage on the coupled map with the light sources and their exact J.v weighted by
s, from the previous stage's kept applied fields; a stage that misses the stop rule is retried over the half interval from the last converged s, at most B
times per target; a stall is recorded with its interval (the fold readout) and a closing stage at s = 1 runs from the last converged iterate; a base that
misses the stop rule is not walked, only closed. Only an s = 1 stage can be kept, by version 32's kept rule; G, the stop rule, the end rule, the records'
content and the propagation are unchanged, so the fixed point is unchanged. Per stage the record carries the status, the kept and confirming residuals, the
largest trial displacement, the kept applied fields' change from the stage's start with its field and site, and the central cell at the kept iterate
(N_dense(0), W/2g at r = 1, 2 and ell, rho_ell(0), J_ell/v_dense at r = 1). The first record is written when its kept residual is below --end-above, the
production's practice on its sigma/a 2 member (labelled).
Version 4's docstring follows.
engine_twofield_v4.py (the busi lane, Computation C): version 3 with the sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v1.md);''')),
    # 2. the ramp's weight
    ("# v36: the light field's per-record accounting",
     "# v36: the light field's per-record accounting\n"
     'LS = {"s": 1.0}                                        # v5: the light sources\' weight inside a ramp stage (1.0 everywhere else, so every other path is bitwise version 4\'s)'),
    # 3. sources_with: the light sum weighted by the ramp's s
    ("                        ref = cache_[chL.key][3]; hs_, Qs_ = base_[chL.key]\n"
     "                        eps += chL.w * (hs_ - dens(chL.h, ref)); J += chL.w * (Qs_ - dens(chL.Q, ref))",
     "                        ref = cache_[chL.key][3]; hs_, Qs_ = base_[chL.key]\n"
     '                        wL = chL.w if LS["s"] == 1.0 else chL.w * LS["s"]                                       # v5: the ramp\'s weight (1.0 outside a ramp stage: bitwise)\n'
     "                        eps += wL * (hs_ - dens(chL.h, ref)); J += wL * (Qs_ - dens(chL.Q, ref))"),
    # 4. the exact J.v: the light block weighted by the ramp's s
    ("acc -= chL.w * d_.reshape(-1, 2).sum(dim=1).cpu().numpy()",
     'acc -= (chL.w if LS["s"] == 1.0 else chL.w * LS["s"]) * d_.reshape(-1, 2).sum(dim=1).cpu().numpy()   # v5: the ramp\'s weight'),
    # 5. run(): the label and the launch notice
    ("a.light_start_continuation, a.light_coupling, len(chl)), flush=True)",
     S('''
a.light_start_continuation, a.light_coupling, len(chl)), flush=True)
    if getattr(a, "light_ramp", None):                                                                  # v5: the light-coupling ramp (C_s2_start_judgment_v2.md, section 2)
        if light_on:
            label += ("; v5: the light-coupling ramp at a record whose direct solve misses the stop rule (base s = 0 the uncoupled map from version 31's start, then s = %s by "
                      "version 32's Newton stage from the previous stage's kept applied fields with the light sources and their exact J.v weighted by s; a missed stage retried "
                      "over the half interval, at most %d bisections per target; a stall recorded with its interval and closed by a stage at s = 1; only an s = 1 stage kept, by "
                      "version 32's rule); the first record written when its kept residual is below the end rule %s, the production's practice on its sigma/a 2 member" % (
                          a.light_coupling_ramp, a.light_ramp_bisect, a.end_above))
        else:
            print("v5: --light-coupling-ramp %s acts on the coupled solve only; with the coupling %s and %d light channel(s) it changes nothing" % (
                a.light_coupling_ramp, a.light_coupling, len(chl)), flush=True)''')),
    # 6. run(): the Newton call carries the ramp
    ("sources_with_heavy=sources_with_heavy, start_cont=a.light_start_continuation",
     'ramp=getattr(a, "light_ramp", None), ramp_bisect=getattr(a, "light_ramp_bisect", 2),                                   # v5: the light-coupling ramp\n'
     "                sources_with_heavy=sources_with_heavy, start_cont=a.light_start_continuation"),
    # 7. the new functions, and newton_record_v32's signature
    ("def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None,",
     S('''
def central_cell_v5(a, info):
    """v5: the central cell at a kept iterate (C_s2_start_judgment_v2.md, section 1): the dense lapse at the centre, W/(2g) at r = 1, 2 and ell (the form the
    record's W_over_2g readouts use), the smoothed energy density at the centre and J_ell/v_dense at r = 1 (the momentum equation fixes the lapse through
    it, N'/N proportional to r N J/v in the flat-slice chart, and v in the cell is set by the cell's mass). An error entry where the solve carries no dense fields."""
    out = {}
    try:
        r0 = np.array([0.0]); r1 = np.array([1.0]); r2 = np.array([2.0]); rl = np.array([float(a.ell)])
        W = info["W_field"]
        first = lambda f_, r_: float(np.asarray(f_(r_), dtype=float).ravel()[0])
        out["N_dense_0"] = first(info["N_dense"], r0)
        out["W_over_2g_at_1"] = first(W, r1) / (2.0 * a.g); out["W_over_2g_at_2"] = first(W, r2) / (2.0 * a.g); out["W_over_2g_at_ell"] = first(W, rl) / (2.0 * a.g)
        out["rho_ell_0"] = first(info["rho_ell"], r0)
        J1 = first(info["J_ell"], r1); v1 = first(info["v_dense"], r1)
        out.update({"J_ell_1": J1, "v_dense_1": v1, "J_over_v_at_1": (J1 / v1) if v1 != 0.0 else None})
    except Exception as ex:                                                                             # a solve without these callables
        out["error"] = str(ex)[:120]
    return out


def coupling_ramp_v5(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, x0, it_hist, T, ctr, solves, kw, ramp, bisect_max):
    """v5: the light-coupling ramp (C_s2_start_judgment_v2.md, section 2): the record's fixed point sought along the homotopy G_s = Solve(heavy sources +
    s x light sources). The base stage s = 0 is the uncoupled map (the light channels not evaluated, the light sources absent from eps, J and the J.v; the
    heavy sum bitwise version 35's) from version 31's start; each target s of the list is version 32's Newton stage (nk_stage_v32) on the coupled map with
    the light sources and their exact J.v weighted by s (LS["s"]), started from the previous stage's kept iterate's applied fields; a stage that misses the
    stop rule is retried over the half interval from the last converged s, at most bisect_max times per target, and the ramp stops at a stall (recorded
    with its interval: the fold readout); a closing stage at s = 1 then runs from the last converged iterate unless one already ran from it; a base that
    misses the stop rule is not walked (its stages would start off the branch), only closed. Only an s = 1 stage can be kept (the caller applies version
    32's kept rule); every other stage's arrays are dropped once its applied fields are taken. Per stage the entry carries s, the status, the kept and
    confirming residuals, the largest trial displacement (v4's readout), the kept applied fields' change from the stage's start with its field and site,
    and the central cell at the kept iterate (central_cell_v5)."""
    n = a.nsites
    t_r = time.time()

    def stage_at(s, start, name, start_desc):
        LS["s"] = float(s)
        try:
            if s == 0.0:
                kwA = dict(kw); kwA.update(chl=(), light_on=False)
                return nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with_heavy(base_new, c_)), do_solve, start, it_hist, T, name, start_desc, ctr, **kwA)
            return nk_stage_v32(a, chs, st, cache0, src_new, do_solve, start, it_hist, T, name, start_desc, ctr, **kw)
        finally:
            LS["s"] = 1.0

    def app_of(stage):
        return (np.array(stage["best"]["app"][0], dtype=float), np.array(stage["best"]["app"][1], dtype=float))

    def entry(stage, s, s_from, x_from):
        b = stage.get("best")
        e = {"s": s, "s_from": s_from, "name": stage["name"], "status": stage["status"], "converged": bool(stage.get("conv")), "evaluations": stage.get("evaluations"),
             "seconds": stage.get("seconds"), "kept_residual": (b["resid"] if b is not None else None), "confirming_residual": stage.get("r_c"), "confirming_dE": stage.get("dE_c"),
             "displacement": stage.get("v4_displacement"), "change_from_start": None, "central_cell": None}
        if b is not None:
            xa = np.concatenate([np.asarray(b["app"][0], dtype=float), np.asarray(b["app"][1], dtype=float)]); xf = np.concatenate([x_from[0], x_from[1]])
            d_ = np.abs(xa - xf); i_ = int(np.argmax(d_)); site = int(i_ if i_ < n else i_ - n)
            e["change_from_start"] = {"max_abs": float(d_[i_]), "field": ("N" if i_ < n else "v"), "site": site, "R_over_rs": (site + (0.5 if a.half_offset else 1.0)) / a.rs,
                                      "N_site0": float(xa[0]), "v_site0": float(xa[n])}
            e["central_cell"] = central_cell_v5(a, b["sol"][3])
        cc = e["central_cell"] or {}; ch_ = e["change_from_start"] or {}
        print("T/rs %.3f v5 ramp s = %.6g (from s = %.6g): %s; kept %s, confirming %s; the kept applied fields moved %s from the stage's start (in %s at site %s, R/rs %s; N at site 0 %s); "
              "central cell at the kept iterate: N_dense(0) %s, W/2g at r = 1, 2, ell %s %s %s, rho_ell(0) %s, J_ell/v at r = 1 %s" % (
                  T / a.rs, s, s_from, ("converged" if e["converged"] else "NOT converged (%s)" % stage["status"]), fmt(e["kept_residual"], "%.2e"), fmt(e["confirming_residual"], "%.2e"),
                  fmt(ch_.get("max_abs"), "%.3e"), ch_.get("field"), ch_.get("site"), fmt(ch_.get("R_over_rs"), "%.3f"), fmt(ch_.get("N_site0"), "%.7f"),
                  fmt(cc.get("N_dense_0"), "%.7f"), fmt(cc.get("W_over_2g_at_1"), "%.4e"), fmt(cc.get("W_over_2g_at_2"), "%.4e"), fmt(cc.get("W_over_2g_at_ell"), "%.4e"),
                  fmt(cc.get("rho_ell_0"), "%.3e"), fmt(cc.get("J_over_v_at_1"), "%.4e")), flush=True)
        return e

    targets = [float(s_) for s_ in ramp]
    out = {"targets": targets, "bisect_max": int(bisect_max), "base": None, "walked": False, "stages": [], "bisections": 0, "reached_1": False, "stalled": None,
           "closing_stage": None, "final_kept_residual": None, "final_converged": None, "label": None, "ctr_end": None, "seconds": 0.0, "final": None}
    base = stage_at(0.0, x0, "v5 ramp base, s = 0 (the light coupling off)", "the first solve's output (version 31's start)")
    solves.append(base)
    print_stage_v32(a, T, base)
    if base["best"] is None:
        out.update({"label": "v5: the light-coupling ramp has no base stage evaluation (%s); the kept rule runs over the other stages" % base["failed"], "ctr_end": ctr["n"],
                    "seconds": time.time() - t_r})
        return out
    out["base"] = entry(base, 0.0, 0.0, (np.asarray(x0[0], dtype=float), np.asarray(x0[1], dtype=float)))
    base_conv = bool(base["conv"])
    x_cur = app_of(base); s_cur = 0.0
    drop_arrays_v32(base)
    final = None; stalled = None; s1_from = None
    cap = 3 * len(targets) + 2; n_st = 0; i = 0
    if base_conv:
        out["walked"] = True
        while i < len(targets) and stalled is None and n_st < cap:
            s_t = targets[i]; s_try = s_t; nb = 0
            while True:
                n_st += 1
                sg = stage_at(s_try, x_cur, "v5 ramp s = %.6g" % s_try, "the ramp's kept iterate at s = %.6g" % s_cur)
                solves.append(sg)
                print_stage_v32(a, T, sg)
                out["stages"].append(entry(sg, s_try, s_cur, x_cur))
                if s_try >= 1.0:
                    s1_from = s_cur
                if sg["best"] is not None and bool(sg["conv"]):
                    x_cur = app_of(sg); s_cur = s_try
                    if s_try >= 1.0:
                        if final is not None:
                            drop_arrays_v32(final)
                        final = sg
                    else:
                        drop_arrays_v32(sg)
                    break
                if s_try >= 1.0 and sg["best"] is not None and (final is None or sg["best"]["resid"] < final["best"]["resid"]):
                    if final is not None:
                        drop_arrays_v32(final)
                    final = sg
                elif sg["best"] is not None:
                    drop_arrays_v32(sg)
                if nb >= bisect_max or n_st >= cap:
                    stalled = {"s_from": s_cur, "s_to": s_try, "bisections": nb, "stages": n_st, "cap_reached": bool(n_st >= cap)}
                    break
                nb += 1; out["bisections"] += 1; s_try = 0.5 * (s_cur + s_try)
            if stalled is None and s_cur >= s_t:
                i += 1
        if stalled is None and s_cur < 1.0:
            stalled = {"s_from": s_cur, "s_to": (targets[i] if i < len(targets) else 1.0), "bisections": out["bisections"], "stages": n_st, "cap_reached": True}
    out["reached_1"] = bool(s_cur >= 1.0); out["stalled"] = stalled
    if (not base_conv or stalled is not None) and s_cur < 1.0 and s1_from != s_cur:
        sg = stage_at(1.0, x_cur, "v5 ramp closing stage, s = 1", "the ramp's kept iterate at s = %.6g" % s_cur)
        solves.append(sg)
        print_stage_v32(a, T, sg)
        e_ = entry(sg, 1.0, s_cur, x_cur); e_["closing"] = True; out["stages"].append(e_)
        out["closing_stage"] = {k_: e_.get(k_) for k_ in ("converged", "kept_residual", "confirming_residual", "status")}
        if sg["best"] is not None and (final is None or bool(sg["conv"]) or sg["best"]["resid"] < final["best"]["resid"]):
            if final is not None:
                drop_arrays_v32(final)
            final = sg
        elif sg["best"] is not None:
            drop_arrays_v32(sg)
    out["final"] = final
    out["final_kept_residual"] = final["best"]["resid"] if (final is not None and final.get("best") is not None) else None
    out["final_converged"] = bool(final["conv"]) if final is not None else None
    tgt = ",".join(("%g" % s_) for s_ in targets)
    if not base_conv:
        how = "the base did not meet the stop rule, so the ramp was not walked (its stages would start off the branch) and only the closing stage at s = 1 ran from the base's kept iterate"
    elif out["reached_1"]:
        how = "reached s = 1"
    else:
        how = ("stalled between s = %g and s = %g after %d bisection(s)%s: the fixed points' branch has no continuation there within the stop rule (a fold) or needs finer steps" % (
            stalled["s_from"], stalled["s_to"], stalled["bisections"], (" (the stage cap %d reached)" % cap) if stalled.get("cap_reached") else ""))
    fin = (("the s = 1 stage met the stop rule (kept %.2e)" % out["final_kept_residual"]) if out["final_converged"] else
           (("the s = 1 stage missed the stop rule (kept %.2e)" % out["final_kept_residual"]) if out["final_kept_residual"] is not None else "no s = 1 stage"))
    out["label"] = "v5: the light-coupling ramp ran (base s = 0 %s, kept %s; targets %s; %d stage(s), %d bisection(s); %s; %s)" % (
        "met the stop rule" if base_conv else "did not meet the stop rule", fmt(out["base"]["kept_residual"], "%.2e"), tgt, len(out["stages"]), out["bisections"], how, fin)
    out["ctr_end"] = ctr["n"]; out["seconds"] = time.time() - t_r
    print("T/rs %.3f %s (%.0f s)" % (T / a.rs, out["label"], out["seconds"]), flush=True)
    return out


def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None, ramp=None, ramp_bisect=2,   # v5: the light-coupling ramp''')),
    # 8. the displacement readout also with the ramp on
    ('    kw["v4_report"] = bool(start_cont != "off" and light_on)',
     '    kw["v4_report"] = bool((start_cont != "off" or bool(ramp)) and light_on)   # v5: the ramp too;'),
    # 9. the ramp runs after a direct solve that misses the stop rule
    ("    final = d\n    v4_stages = None",
     S('''
    ramp_v5 = None                                                                                      # v5: the light-coupling ramp (C_s2_start_judgment_v2.md, section 2)
    if (not d["conv"]) and ramp and light_on and sources_with_heavy is not None:
        ramp_v5 = coupling_ramp_v5(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, (Nn0, vn0), it_hist, T, ctr, solves, kw, ramp, ramp_bisect)
    final = d
    v4_stages = None''')),
    # 10. the ramp's s = 1 stage, when it meets the stop rule, is the record
    ('        cont = "not needed (the direct solve meets the stop rule)"\n'
     '    elif (base_prev is None or f_prev is None) and start_cont == "coupling" and light_on and sources_with_heavy is not None:',
     S('''
        cont = "not needed (the direct solve meets the stop rule)"
    elif ramp_v5 is not None and ramp_v5.get("final") is not None and ramp_v5["final"].get("conv"):          # v5: the ramp's s = 1 stage meets the stop rule
        final = ramp_v5["final"]; drop_arrays_v32(d); cont = ramp_v5["label"]
    elif (base_prev is None or f_prev is None) and start_cont == "coupling" and light_on and sources_with_heavy is not None:''')),
    # 11. version 32's kept rule over an s = 1 stage that missed the stop rule
    ('    b = final["best"]\n    b["dE"] = final["dE_c"] if final["dE_c"] is not None else b["dE"]',
     S('''
    if ramp_v5 is not None and ramp_v5.get("final") is not None and ramp_v5["final"] is not final:          # v5: version 32's kept rule over the ramp's s = 1 stage
        rf = ramp_v5["final"]
        if rf.get("best") is not None and (final.get("best") is None or rf["best"]["resid"] < final["best"]["resid"]):
            drop_arrays_v32(final); d.pop("confirm_repeated", None)
            if ctr["n"] > ramp_v5["ctr_end"]:
                rf["confirm_repeated"] = confirm_v32(a, rf["best"], rf["evalG"], it_hist, ctr)                  # the kept iterate's confirming step again, the history's last entry
            final = rf; cont = "%s; kept from the ramp's s = 1 stage: %s (v5)" % (cont, ramp_v5["label"])
        else:
            if rf.get("best") is not None:
                drop_arrays_v32(rf)
            cont = "%s; the ramp's s = 1 stage not kept: %s (v5)" % (cont, ramp_v5["label"])
    elif ramp_v5 is not None and ramp_v5.get("final") is None:
        cont = "%s; %s (v5)" % (cont, ramp_v5["label"])
    b = final["best"]
    b["dE"] = final["dE_c"] if final["dE_c"] is not None else b["dE"]''')),
    # 12. the record's entry
    ('"displacements": {q["name"]: q.get("v4_displacement") for q in solves}}',
     S('''
"displacements": {q["name"]: q.get("v4_displacement") for q in solves}}
    if ramp_v5 is not None:                                                                              # v5: the ramp's record entry
        e32["light_coupling_ramp_v5"] = {k_: v_ for k_, v_ in ramp_v5.items() if k_ != "final"}
        if final is not d and "confirm_repeated" in final:
            e32["confirming_step_repeated"] = {"residual": final["confirm_repeated"]["r_c"], "dE": final["confirm_repeated"]["dE_c"]}
    if ramp and light_on and (base_prev is None or f_prev is None):                                        # v5: the first record's rule, the production's practice made explicit
        e32["first_record_rule_v5"] = ("the first record is written when its kept residual is below --end-above (%s), the production's practice on its sigma/a 2 member "
                                       "(member2 record 0: unconverged, 1.857e-04 at iteration 15; its 0.0625 converged); nothing propagated differs from version 4" % (a.end_above,))''')),
    # 13. main: the arguments
    ('    ap.add_argument("--light-start-plain-steps"',
     S('''
    ap.add_argument("--light-coupling-ramp", dest="light_coupling_ramp", default="",
                    help="(v5) at a record whose direct coupled Newton solve misses the stop rule: the light sources' weights s of the homotopy from the uncoupled map (s = 0) to the coupled map (s = 1), increasing, ending at 1, for example 0.25,0.5,0.75,1; empty: off (version 4 bitwise)")   # v5
    ap.add_argument("--light-ramp-bisect", dest="light_ramp_bisect", type=int, default=2, help="(v5) a ramp stage that misses the stop rule is retried over the half interval from the last converged weight, at most this many times per target")   # v5
    ap.add_argument("--light-start-plain-steps"''')),
    # 14. main: the ramp's parsing and checks
    ("    if a.light_start_plain_steps < 0:",
     S('''
    a.light_ramp = []                                                                                                                    # v5: the parsed ramp (empty: off)
    if a.light_coupling_ramp:
        try:
            a.light_ramp = [float(x_) for x_ in a.light_coupling_ramp.split(",") if x_.strip()]
        except ValueError:
            ap.error("--light-coupling-ramp: a comma-separated list of weights")
        if not a.light_ramp or any(not (0.0 < s_ <= 1.0) for s_ in a.light_ramp) or any(q_ >= p_ for q_, p_ in zip(a.light_ramp, a.light_ramp[1:])) or a.light_ramp[-1] != 1.0:
            ap.error("--light-coupling-ramp: increasing weights in (0, 1] ending at 1 (for example 0.25,0.5,0.75,1)")
        if a.mix != "newton" or not v32_own_loop(a) or a.light_follow:
            ap.error("v5: --light-coupling-ramp acts inside version 32's own Newton loop (--mix newton with --nk-line-search, --nk-krylov or --nk-restarts) and not under --light-follow")
    if a.light_ramp_bisect < 0:                                                                                                          # v5
        ap.error("--light-ramp-bisect must be nonnegative")
    if a.light_start_plain_steps < 0:''')),
    # 15 to 18. the script's name in the json files
    ('{"script": "engine_twofield_v4.py (light-follow)"', '{"script": "engine_twofield_v5.py (light-follow)"'),
    ('json.dump({"script": "engine_twofield_v4.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)',
     'json.dump({"script": "engine_twofield_v5.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)'),
    ('json.dump({"script": "engine_twofield_v4.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)',
     'json.dump({"script": "engine_twofield_v5.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)'),
    ('json.dump(dict({"script": "engine_twofield_v4.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32}',
     'json.dump(dict({"script": "engine_twofield_v5.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32}'),
]


def build(src_path, out_path):
    text = open(src_path).read()
    for i, (old, new) in enumerate(REPLACEMENTS, 1):
        c = text.count(old)
        if c != 1:
            raise SystemExit("v5 build: replacement %d's current text occurs %d time(s) in %s (exactly 1 required): %r" % (i, c, src_path, old[:140]))
        text = text.replace(old, new)
    compile(text, out_path, "exec")
    for needle in ("coupling_ramp_v5(", "central_cell_v5(", "--light-coupling-ramp", 'LS = {"s": 1.0}', "light_coupling_ramp_v5", "first_record_rule_v5"):
        if needle not in text:
            raise SystemExit("v5 build: the built text lacks %r" % needle)
    with open(out_path, "w") as fh:
        fh.write(text)
    print("engine_twofield_v5: %d replacements applied to %s -> %s (%d lines)" % (len(REPLACEMENTS), src_path, out_path, text.count("\n")), flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="builds engine_twofield_v5.py from the built engine_twofield_v4.py by exact replacements (each current text unique)")
    ap.add_argument("--src", default=os.path.join(here, "engine_twofield_v4.py"), help="the built engine_twofield_v4.py (default: beside this file)")
    ap.add_argument("--out", default=os.path.abspath(__file__), help="where the engine is written (default: over this file)")
    a = ap.parse_args()
    src = open(a.src).read()
    if "coupling_ramp_v5(" in src:
        raise SystemExit("v5 build: %s already carries version 5" % a.src)
    if "def newton_record_v32(" not in src or "start_continuation_v4(" not in src:
        raise SystemExit("v5 build: %s is not the built engine_twofield_v4 (build it first: python engine_twofield_v4.py)" % a.src)
    build(a.src, a.out)


if __name__ == "__main__":
    main()
