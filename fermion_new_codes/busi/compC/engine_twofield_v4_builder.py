#!/usr/bin/env python
"""engine_twofield_v4.py (the busi lane, Computation C): delivered as the list of exact replacements that turn engine_twofield_v3.py into
version 4 (busi/notes/C_s2_start_judgment_v1.md, section 2); every new or changed line of the engine is marked "v4:".
Build: python engine_twofield_v4.py [--src engine_twofield_v3.py] [--out engine_twofield_v4.py] applies the replacements in order (each
current text must occur exactly once in the source, else the build stops and names it), writes the engine (over this file by default) and
prints its md5. Replacement 1 is the engine's docstring, which states the option."""
import argparse
import hashlib
import os
import sys

REPLACEMENTS = [
("1 the docstring", r'''
"""engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every
''', r'''
"""engine_twofield_v4.py (the busi lane, Computation C): version 3 with the sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v1.md);
every change against version 3 is marked "v4:". One option, --light-start-continuation {off,coupling} (default off; with it off every code path
and every record is version 3's, bitwise): at the first record of a run (no previous record, no resume), when the direct coupled Newton solve
does not meet the stop rule, the record is re-solved in stages: A, the uncoupled map (the light sources removed from eps, J and the exact J.v,
the light channels not evaluated; the heavy sum bitwise version 35's) from version 31's start, the solve the production is known to finish;
B, the coupled map from stage A's kept iterate's applied fields (version 32's stage 2 rule); C, when B misses the stop rule, plain fixed-point
steps from B's kept iterate while they lower the residual (at most --light-start-plain-steps). The kept rule is version 32's (the smaller kept
residual of the direct solve and stage B/C; stage A is never kept, it is not the coupled fixed point); the map G, the stop rule, the end rule,
the records' content and the propagation are unchanged, so the fixed point is unchanged. With the option on and the coupling on every Newton
stage prints the largest displacement of any trial from its start (the ill-conditioning readout). With the coupling off the option does
nothing (said at launch).
Version 3's docstring follows.
engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every
'''),
("2 nk_stage_v32's signature", r'''
def nk_stage_v32(a, chs, st, cache0, sources_fn, do_solve, x0, it_hist, T, name, start, ctr, chl=(), light_on=False, sgn_dev=None, S_host=None):
    """version 32's nk_stage_v32 (its docstring) with the light channels in every evaluation and the light sources in the exact J.v (v36)."""
''', r'''
def nk_stage_v32(a, chs, st, cache0, sources_fn, do_solve, x0, it_hist, T, name, start, ctr, chl=(), light_on=False, sgn_dev=None, S_host=None, v4_report=False):   # v4
    """version 32's nk_stage_v32 (its docstring) with the light channels in every evaluation and the light sources in the exact J.v (v36).
    v4: with v4_report the stage records and prints the largest displacement of any evaluation from its start (displacement_report_v4)."""
'''),
("3 nk_stage_v32's state", r'''
    nk = {"best": None, "failed": None}
    n_start = ctr["n"]
''', r'''
    nk = {"best": None, "failed": None, "disp_max": 0.0, "disp_index": None}                             # v4: the displacement readout's state
    n_start = ctr["n"]
'''),
("4 nk_stage_v32's F", r'''
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        ctr["n"] += 1
        if jv_on:
''', r'''
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        ctr["n"] += 1
        if v4_report:                                                                                   # v4: the largest displacement from the stage's start
            dxa = np.abs(np.asarray(x, dtype=float) - x0v); i_ = int(np.argmax(dxa))
            if dxa[i_] > nk["disp_max"]:
                nk["disp_max"] = float(dxa[i_]); nk["disp_index"] = i_
        if jv_on:
'''),
("5 nk_stage_v32's return", r'''
    out["evaluations"] = ctr["n"] - n_start
    out["seconds"] = time.time() - t0
    out["history"] = [h0, len(it_hist)]
    return out
''', r'''
    out["evaluations"] = ctr["n"] - n_start
    out["seconds"] = time.time() - t0
    out["history"] = [h0, len(it_hist)]
    if v4_report:                                                                                       # v4
        out["v4_displacement"] = displacement_report_v4(a, T, name, nk, x0v)
    return out
'''),
("6 the v4 helpers and newton_record_v32's signature", r'''
def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None):
    """version 32's record solve (its docstring), calling this file's nk_stage_v32 (v36); the midpoint base of stage 1 covers every key of base_new,
    the light channels' included."""
    ctr = {"n": 0, "E_prev": E0}
''', r'''
def displacement_report_v4(a, T, name, nk, x0v):
    """v4: the largest displacement of any evaluation of a Newton stage from the stage's start, the field and the site that carry it (the note's
    ill-conditioning readout: a direction orders above the residual marks a near-null direction of I - G' that the inner solve left unresolved);
    printed, and returned for the stage's record entry (the option on and the coupling on only)."""
    i_ = nk.get("disp_index")
    if i_ is None:
        return None
    n = a.nsites
    field = "N" if i_ < n else "v"; site = int(i_ if i_ < n else i_ - n)
    out = {"max_abs_displacement": float(nk["disp_max"]), "field": field, "site": site, "R_over_rs": (site + (0.5 if a.half_offset else 1.0)) / a.rs,
           "start_norm_inf": float(np.max(np.abs(x0v)))}
    print("T/rs %.3f newton (%s) v4: the trial fields ranged up to %.3e from the stage's start, in %s at site %d (R/rs %.3f)" % (
        T / a.rs, name, out["max_abs_displacement"], field, site, out["R_over_rs"]), flush=True)
    return out


def plain_closure_v4(a, stage, it_hist, T, ctr, max_steps):
    """v4 stage C: plain fixed-point steps x <- G(x) from the stage's kept iterate's output while each lowers the residual max(dN, dv) (the coupled
    map contracted by 2.7 at the sigma/a 2 stall), at most max_steps; the bookkeeping is nk_stage_v32's F (ctr, it_hist, E_prev under version 36's
    fix (b)); the kept rule unchanged (the smallest residual); the confirming step is redone on the new kept iterate."""
    evalG = stage["evalG"]; b = stage["best"]
    path = [b["resid"]]; taken = 0
    while taken < max_steps:
        N_a = np.array(b["sol"][0], dtype=float); v_a = np.array(b["sol"][1], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        ctr["n"] += 1; taken += 1
        Nn_, vn_, Wn_, info_ = sol_
        if info_.get("failed"):
            path.append(None)
            break
        E_ = FLD.charge(info_, a.g)
        dN_ = float(np.max(np.abs(Nn_ - N_a))); dv_ = float(np.max(np.abs(vn_ - v_a)))
        dE_ = abs(E_ - ctr["E_prev"]) / abs(E_)
        r_ = max(dN_, dv_)
        it_hist.append([float(dE_), dN_, dv_])
        path.append(r_)
        better = r_ < b["resid"]
        if better or not V35.V36["fixes"]:
            ctr["E_prev"] = E_
        if not better:
            break
        b = {"resid": r_, "k_it": ctr["n"], "cache": cache_, "srcs": srcs_, "sol": sol_, "app": (N_a.copy(), v_a.copy()), "dE": dE_, "E": E_}
        if r_ < a.stop_fields:
            break
    stage["best"] = b
    stage.update(confirm_v32(a, b, evalG, it_hist, ctr))
    stage["v4_plain_closure"] = {"steps": taken, "residual_path": path, "kept_evaluation": b["k_it"], "kept_residual": b["resid"]}
    print("T/rs %.3f newton (v4 stage C, plain steps from stage B's kept iterate): %d step(s); residual path %s; kept %.2e (evaluation %d), the confirming plain step's residual %s and abs dE/E %s; %s" % (
        T / a.rs, taken, ", ".join((("%.2e" % q) if q is not None else "failed") for q in path), b["resid"], b["k_it"],
        fmt(stage.get("r_c"), "%.2e"), fmt(stage.get("dE_c"), "%.1e"), "converged" if stage.get("conv") else "not converged"), flush=True)
    return stage


def start_continuation_v4(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, x0, it_hist, T, ctr, d, solves, kw, plain_steps):
    """v4: the first record's continuation in the light coupling (the docstring): stage A, the uncoupled map from version 31's start (the light
    channels not evaluated, the light sources absent from eps, J and the J.v; the heavy sum bitwise version 35's); stage B, the coupled map from
    stage A's kept iterate's applied fields; stage C (plain_closure_v4) when B misses the stop rule. The kept rule is version 32's: the smaller
    kept residual of the direct solve and stage B/C; stage A is never kept. Returns (the continuation's label, the final stage, the stage list)."""
    kwA = dict(kw); kwA.update(chl=(), light_on=False)
    sA = nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with_heavy(base_new, c_)), do_solve, x0, it_hist, T,
                      "v4 stage A, the light coupling off", "the first solve's output (version 31's start)", ctr, **kwA)
    solves.append(sA)
    print_stage_v32(a, T, sA)
    if sA["best"] is None:
        d["confirm_repeated"] = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)
        return "v4: ran at the first record; stage A (the light coupling off) has no evaluation (%s); kept from the direct solve" % sA["failed"], d, [sA]
    xA = (np.array(sA["best"]["app"][0], dtype=float), np.array(sA["best"]["app"][1], dtype=float))
    tagA = "stage A %s (kept residual %.2e, confirming %s)" % ("met the stop rule" if sA["conv"] else "did not meet the stop rule", sA["best"]["resid"], fmt(sA.get("r_c"), "%.2e"))
    drop_arrays_v32(sA)
    sB = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, xA, it_hist, T, "v4 stage B, the light coupling on", "stage A's kept iterate's applied fields", ctr, **kw)
    solves.append(sB)
    print_stage_v32(a, T, sB)
    if sB["best"] is not None and not sB["conv"] and plain_steps > 0:
        plain_closure_v4(a, sB, it_hist, T, ctr, plain_steps)
    if sB["best"] is not None and sB["best"]["resid"] < d["best"]["resid"]:
        drop_arrays_v32(d)
        return ("v4: ran at the first record (the light coupling continuation; %s); kept from stage B%s" % (tagA, " after stage C's plain steps" if sB.get("v4_plain_closure") else ""), sB, [sA, sB])
    if sB["best"] is not None:
        drop_arrays_v32(sB)
    d["confirm_repeated"] = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)
    return "v4: ran at the first record (the light coupling continuation; %s); kept from the direct solve (stage B's kept residual is not smaller)" % tagA, d, [sA, sB]


def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None,
                      sources_with_heavy=None, start_cont="off", plain_steps=12):                                                      # v4
    """version 32's record solve (its docstring), calling this file's nk_stage_v32 (v36); the midpoint base of stage 1 covers every key of base_new,
    the light channels' included. v4: at the first record (no previous record), with start_cont "coupling" and the coupling on, a direct solve that
    misses the stop rule is followed by start_continuation_v4 (stages A, B, C); with start_cont "off" nothing here differs from version 3."""
    ctr = {"n": 0, "E_prev": E0}
'''),
("7 the stages' keyword arguments", r'''
    kw = dict(chl=chl, light_on=light_on, sgn_dev=sgn_dev, S_host=S_host)                               # v36
''', r'''
    kw = dict(chl=chl, light_on=light_on, sgn_dev=sgn_dev, S_host=S_host)                               # v36
    kw["v4_report"] = bool(start_cont != "off" and light_on)                                            # v4: the displacement readout, the option on and the coupling on only
'''),
("8 the continuation branch", r'''
    final = d
    if d["conv"]:
        cont = "not needed (the direct solve meets the stop rule)"
    elif a.source_continuation != "two-stage":
        cont = "off"
    elif base_prev is None or f_prev is None:
        cont = "not run: no previous record in this run"
    else:
''', r'''
    final = d
    v4_stages = None                                                                                    # v4
    if d["conv"]:
        cont = "not needed (the direct solve meets the stop rule)"
    elif (base_prev is None or f_prev is None) and start_cont == "coupling" and light_on and sources_with_heavy is not None:   # v4: the first record's continuation
        cont, final, v4_stages = start_continuation_v4(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, (Nn0, vn0), it_hist, T, ctr, d, solves, kw, plain_steps)
    elif a.source_continuation != "two-stage":
        cont = "off"
    elif base_prev is None or f_prev is None:
        cont = "not run: no previous record in this run"
    else:
'''),
("9 the record's solve entry", r'''
    if "confirm_repeated" in d:
        e32["confirming_step_repeated"] = {"residual": d["confirm_repeated"]["r_c"], "dE": d["confirm_repeated"]["dE_c"]}
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
''', r'''
    if "confirm_repeated" in d:
        e32["confirming_step_repeated"] = {"residual": d["confirm_repeated"]["r_c"], "dE": d["confirm_repeated"]["dE_c"]}
    if v4_stages is not None:                                                                            # v4: the record's entry (the option on and run only)
        sA_, sB_ = v4_stages[0], (v4_stages[1] if len(v4_stages) > 1 else None)
        e32["light_start_continuation_v4"] = {"option": start_cont, "plain_steps_max": plain_steps, "stages": [q["name"] for q in v4_stages],
                                              "stage_A": {"converged": sA_.get("conv"), "kept_residual": (sA_.get("best") or {}).get("resid"), "confirming_residual": sA_.get("r_c"),
                                                          "status": sA_.get("status")},
                                              "stage_B": ({"converged": sB_.get("conv"), "kept_residual": (sB_.get("best") or {}).get("resid"), "confirming_residual": sB_.get("r_c"),
                                                           "status": sB_.get("status"), "plain_closure": sB_.get("v4_plain_closure")} if sB_ is not None else None),
                                              "displacements": {q["name"]: q.get("v4_displacement") for q in solves}}
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
'''),
("10 the launch note and the label", r'''
    light_on = bool(chl) and a.light_coupling == "on"                                                  # v36: the source coupling's switch
''', r'''
    light_on = bool(chl) and a.light_coupling == "on"                                                  # v36: the source coupling's switch
    if getattr(a, "light_start_continuation", "off") != "off":                                           # v4: the first record's continuation (the note's section 2)
        if light_on:
            label += ("; v4: the first record's continuation in the light coupling when the direct solve misses the stop rule (stage A the uncoupled map from version 31's "
                      "start, stage B the coupled map from A's kept applied fields, stage C at most %d plain steps while they lower the residual; version 32's kept rule)" % a.light_start_plain_steps)
        else:
            print("v4: --light-start-continuation %s acts on the coupled solve only; with the coupling %s and %d light channel(s) it changes nothing" % (
                a.light_start_continuation, a.light_coupling, len(chl)), flush=True)
'''),
("11 the heavy-only sources (stage A's map)", r'''
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def do_solve(srcs_):
''', r'''
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def sources_with_heavy(base_, cache_):
            """v4: sources_with(base_, cache_) without the light sum (the uncoupled map of stage A); the heavy loop is sources_with's first loop verbatim,
            so its sum is bitwise version 35's."""
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base_[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def do_solve(srcs_):
'''),
("12 the record solve's call", r'''
                do_solve, Nn, vn, E_prev, it_hist, T, chl=chl, light_on=light_on, sgn_dev=sgn_dev, S_host=S_host)                       # v36
''', r'''
                do_solve, Nn, vn, E_prev, it_hist, T, chl=chl, light_on=light_on, sgn_dev=sgn_dev, S_host=S_host,                       # v36
                sources_with_heavy=sources_with_heavy, start_cont=a.light_start_continuation, plain_steps=a.light_start_plain_steps)      # v4
'''),
("13 the light-follow json's script name", r'''
                json.dump({"script": "engine_twofield_v3.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v3
''', r'''
                json.dump({"script": "engine_twofield_v4.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v4
'''),
("14 the periodic json's script name", r'''
                json.dump({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v3
''', r'''
                json.dump({"script": "engine_twofield_v4.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v4
'''),
("15 the P2 stop json's script name", r'''
                json.dump({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v3
''', r'''
                json.dump({"script": "engine_twofield_v4.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v4
'''),
("16 the final json's script name", r'''
        json.dump(dict({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v3
''', r'''
        json.dump(dict({"script": "engine_twofield_v4.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v4
'''),
("17 the arguments", r'''
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    a = ap.parse_args()
''', r'''
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    ap.add_argument("--light-start-continuation", dest="light_start_continuation", default="off", choices=["off", "coupling"],
                    help="(v4) at the first record, when the direct coupled Newton solve misses the stop rule: stage A the uncoupled map, stage B the coupled map from A's kept fields, stage C plain steps; off: version 3 bitwise")   # v4
    ap.add_argument("--light-start-plain-steps", dest="light_start_plain_steps", type=int, default=12, help="(v4) stage C: at most this many plain fixed-point steps (0: none)")   # v4
    a = ap.parse_args()
'''),
("18 the arguments' checks", r'''
    if a.weyl_oracle and a.reference != "R2":
        ap.error("version 34 (d): the oracle compares R2's continuation")
''', r'''
    if a.weyl_oracle and a.reference != "R2":
        ap.error("version 34 (d): the oracle compares R2's continuation")
    if a.light_start_continuation != "off" and (a.mix != "newton" or not v32_own_loop(a) or a.light_follow):                              # v4
        ap.error("v4: --light-start-continuation acts inside version 32's own Newton loop (--mix newton with --nk-line-search, --nk-krylov or --nk-restarts) and not under --light-follow")
    if a.light_start_plain_steps < 0:                                                                                                    # v4
        ap.error("--light-start-plain-steps must be nonnegative")
'''),
]


def build(src, out):
    txt = open(src).read()
    for name, cur, new in REPLACEMENTS:
        k = txt.count(cur)
        if k != 1:
            first = cur.strip("\n").splitlines()[0]
            hint = [ln for ln in txt.splitlines() if ln.strip()[:40] == first.strip()[:40]]
            raise SystemExit("replacement %s: the current text occurs %d times in %s (1 expected); lines opening alike: %s" % (name, k, src, hint[:3]))
        txt = txt.replace(cur, new)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(txt)
    os.replace(tmp, out)
    print("engine_twofield_v4.py built from %s: %d replacements, %d lines, md5 %s -> %s" % (src, len(REPLACEMENTS), txt.count("\n"), hashlib.md5(txt.encode()).hexdigest(), out))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(here, "engine_twofield_v3.py"))
    ap.add_argument("--out", default=os.path.abspath(__file__))
    a = ap.parse_args()
    build(a.src, a.out)
