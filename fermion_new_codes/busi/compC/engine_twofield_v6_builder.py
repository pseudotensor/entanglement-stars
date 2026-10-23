#!/usr/bin/env python
"""engine_twofield_v6.py (the busi lane, Computation C): the replacement list that builds version 6 over engine_twofield_v5.py
(busi/notes/C_s2_start_judgment_v3.md, section 4) and its builder. Every change against version 5 is marked "v6:". The engine is built over this file by
    python engine_twofield_v6.py [--src engine_twofield_v5.py] [--out engine_twofield_v6.py] [--check]
(the gate's ensure_built does this; a file that already carries the engine body is left alone). Each current text is matched exactly once in the source
(runs of 2 or more spaces taken as any such run: the alignment of trailing comments is not load-bearing); the result is compiled before it is written.
What version 6 changes:
(1) the memory: nk_stage_v32 breaks the F <-> jvp reference cycle, empties the exact J.v's memo and drops nk's reference to the kept iterate at the
    stage's end; every stage not the candidate of version 32's kept rule is released at once (release_stage_v6); the kept rule runs as soon as the
    homotopy returns, so 1 candidate's arrays are held; the device memory is printed per stage and per record and the run's peak at its end (mem_v6);
(2) the homotopy (homotopy_ramp_v6 in place of coupling_ramp_v5, under --light-coupling-ramp and --light-ramp-bisect): the light coupling at the first
    record only; the propagated covariance from the previous record's kept state at every later record (version 32's two-stage continuation walked with
    bisection); the base rule --light-ramp-base-above (default --end-above) and the floor rule --light-ramp-floor-factor (2), labelled per stage;
(3) the soft gain per stage (soft_gain_v6; --light-ramp-gain-vectors Arnoldi products of the exact J.v at the kept iterate, 0 off): g, 1 - g, its site.
With the ramp off every code path and every record is version 5's, hence version 4's, bitwise (gate v6's G1 and G2)."""
import argparse
import os
import re
import sys


def _t(s):
    """one leading and one trailing newline of a text literal removed (the literals below start and end on their own lines)."""
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


# (name, the current text (version 5), the new text (version 6), how many times the current text occurs)
REPLACEMENTS = [
    ("the docstring", r'''
"""engine_twofield_v5.py (the busi lane, Computation C): version 4 with the second sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v2.md);
''', r'''
"""engine_twofield_v6.py (the busi lane, Computation C): version 5 with the third sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v3.md);
every change against version 5 is marked "v6:". (1) The memory: nk_stage_v32 breaks the F <-> jvp reference cycle, empties the exact J.v's memo and drops
nk's reference to the kept iterate at the stage's end (the cycle held each stage's last evaluation's cache, 4 to 5 GiB, until a generation-2 collection:
the 78 GiB of gate v5's G3); every Newton stage not the candidate of version 32's kept rule is released at once; the kept rule over the direct solve and
the homotopy's final stage runs as soon as the homotopy returns, so 1 candidate's arrays are held; the device memory is printed per stage, per record
and at the run's end. (2) The homotopy (homotopy_ramp_v6, under --light-coupling-ramp and --light-ramp-bisect): at the first record of a run the light
coupling, G_s = Solve(heavy sources + s x light sources), base s = 0 the uncoupled map from version 31's start, the 2 maps sharing the state there; at
every later record the propagated covariance, G_t = Solve(sources at (1 - t) base_prev + t base_new), base t = 0 the previous record's kept state, which
is version 32's two-stage continuation walked with bisection (the coupling ramp's base after the first record, the uncoupled sources on the coupled
history's seas, is a third problem and is not run). A base below --light-ramp-base-above (default --end-above, the production's practice at T = 0) is
walked; a stage advances by the stop rule or, labelled "floor rule", within --light-ramp-floor-factor (2) times the base's kept residual; a stage above
both is bisected; a stall is recorded with its interval. (3) The soft gain per stage (--light-ramp-gain-vectors K, default 0): K Arnoldi products of the
exact J.v at the kept iterate from the unit vector of N at site 0 give g, the largest eigenvalue of G', 1 - g the smallest eigenvalue of I - G' (0 at a
fold), with its eigenvector's site. With the ramp off every path is version 5's, bitwise.
Version 5's docstring follows.
engine_twofield_v5.py (the busi lane, Computation C): version 4 with the second sigma/a 2 start judgment applied (busi/notes/C_s2_start_judgment_v2.md);
''', 1),

    ("the imports", r'''
import argparse
import json
''', r'''
import argparse
import gc                                             # v6: the cyclic collector is run at the stage boundaries (free_device_v6)
import json
''', 1),

    ("the memory helpers", r'''
def tml_reset_v36():
''', r'''
MEM = {"peak_gib": 0.0}                                # v6: the run's peak device memory allocated (GiB), printed per record and at the end (mem_v6)


def mem_v6(reset=False):
    """v6: the device memory in GiB as (allocated now, the peak since the last reset, reserved); None off CUDA. The run's peak is kept in MEM."""
    torch = BK["torch"]
    try:
        dev = torch.device(BK["dev"])
        if dev.type != "cuda":
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats(dev)
        now = torch.cuda.memory_allocated(dev) / 2.0 ** 30
        peak = torch.cuda.max_memory_allocated(dev) / 2.0 ** 30
        reserved = torch.cuda.memory_reserved(dev) / 2.0 ** 30
        MEM["peak_gib"] = max(MEM["peak_gib"], peak)
        return now, peak, reserved
    except Exception:
        return None


def free_device_v6():
    """v6: the cyclic collector run and the caching allocator's free blocks returned to the device; no arithmetic. nk_stage_v32 breaks its own F <-> jvp
    cycle at the stage's end; this collects what else a stage left and returns the released stages' blocks."""
    gc.collect()
    torch = BK["torch"]
    try:
        if torch.device(BK["dev"]).type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _has_tensor_v6(v, torch):
    if torch.is_tensor(v):
        return True
    if isinstance(v, (tuple, list)):
        return any(torch.is_tensor(q) for q in v)
    if isinstance(v, dict):
        return any(_has_tensor_v6(q, torch) for q in list(v.values())[:3])
    return False


def release_stage_v6(stage):
    """v6: drop_arrays_v32(stage), the kept iterate's cache (and any other tensor-bearing entry of it) nulled where the stage dict still references it,
    and the device memory returned: a stage holds its arrays only while it is the candidate of version 32's kept rule or its applied fields are still to
    be taken. The scalars the record entry reads (stage_entry_v32) are untouched."""
    drop_arrays_v32(stage)
    torch = BK["torch"]
    b = stage.get("best")
    if isinstance(b, dict):
        for k_ in list(b.keys()):
            if k_ == "cache" or (k_ not in ("sol", "app", "srcs") and _has_tensor_v6(b[k_], torch)):
                b[k_] = None
    free_device_v6()


def tml_reset_v36():
''', 1),

    ("nk_stage_v32's signature", r'''
S_host=None, v4_report=False):
''', r'''
S_host=None, v4_report=False, gain_vectors=0):
''', 1),

    ("nk_stage_v32: the soft gain after the confirming step", r'''
        out.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))
    out["evaluations"] = ctr["n"] - n_start
''', r'''
        out.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))
    if gain_vectors and jv_on and nk["best"] is not None:                                                # v6: the soft gain at the kept iterate (soft_gain_v6; the stage's own exact J.v; its first product re-evaluates F at the kept applied fields, counted)
        out["soft_gain_v6"] = soft_gain_v6(a, jvp, nk["best"], int(gain_vectors), T, name)
    out["evaluations"] = ctr["n"] - n_start
''', 1),

    ("nk_stage_v32: the cycle broken and the memo emptied at the stage's end", r'''
        out["v4_displacement"] = displacement_report_v4(a, T, name, nk, x0v)
    return out
''', r'''
        out["v4_displacement"] = displacement_report_v4(a, T, name, nk, x0v)
    if jv_on:                                                                                           # v6: F.jvp = jvp while jvp calls F is a reference cycle; it kept each stage's memo (the last evaluation's cache, 4 to 5 GiB) and nk alive until a generation-2 collection (gate v5's 78 GiB); broken here
        try:
            del F.jvp
        except AttributeError:
            pass
    memo["x"] = None; memo["cache"] = None; memo["srcs"] = None                                        # v6: the memo emptied (the confirming step and the kept rule do not read it)
    nk["best"] = None                                                                                   # v6: the kept iterate lives in out["best"] alone from here
    return out
''', 1),

    ("soft_gain_v6", r'''
def displacement_report_v4(a, T, name, nk, x0v):
''', r'''
def soft_gain_v6(a, jvp, b, K, T, name):
    """v6: the soft gain at a Newton stage's kept iterate (C_s2_start_judgment_v3.md, sections 3 and 5): K steps of Arnoldi on G'(x) = I + dF/dx at the kept
    iterate's applied fields x, the products the stage's exact J.v (jvp(x, w) = (G' - I) w; its first call re-evaluates F at x, counted as an evaluation),
    the start vector the unit vector of N at site 0 (the soft direction's carrier), 2 passes of orthogonalization. Returns the Ritz value of largest real
    part g (the gain along the soft direction; 1 - g the smallest eigenvalue of I - G' seen, 0 at a fold), the largest Ritz modulus, the Arnoldi residual
    of g, the 3 largest Ritz real parts and the g Ritz vector's dominant component (field, site, its share of the norm). The solve's arithmetic is not touched."""
    n = a.nsites; m = 2 * n
    t0 = time.time()
    x = np.concatenate([np.asarray(b["app"][0], dtype=float), np.asarray(b["app"][1], dtype=float)])
    V = np.zeros((m, K + 1)); H = np.zeros((K + 1, K))
    V[0, 0] = 1.0
    k_done = 0; why = None
    for k in range(K):
        try:
            w = np.asarray(jvp(x, V[:, k].copy()), dtype=float) + V[:, k]
        except Exception as ex:
            why = "product %d failed: %s" % (k + 1, str(ex)[:80])
            break
        if w.shape != (m,) or not np.all(np.isfinite(w)):
            why = "product %d not finite" % (k + 1)
            break
        for _pass in range(2):
            for j in range(k + 1):
                c_ = float(V[:, j] @ w)
                H[j, k] += c_
                w = w - c_ * V[:, j]
        H[k + 1, k] = float(np.linalg.norm(w)); k_done = k + 1
        if H[k + 1, k] <= 1e-13:
            break
        V[:, k + 1] = w / H[k + 1, k]
    out = {"products": k_done, "requested": int(K), "why": why, "gain": None, "gain_imag": None, "one_minus_gain": None, "largest_modulus": None, "residual": None,
           "ritz_real_top3": None, "field": None, "site": None, "R_over_rs": None, "component": None, "seconds": None}
    if k_done > 0:
        ev, Y = np.linalg.eig(H[:k_done, :k_done])
        i_ = int(np.argmax(ev.real)); g = ev[i_]; y = Y[:, i_]
        vec = V[:, :k_done] @ y; j_ = int(np.argmax(np.abs(vec))); site = int(j_ if j_ < n else j_ - n)
        out.update({"gain": float(g.real), "gain_imag": float(g.imag), "one_minus_gain": float(1.0 - g.real), "largest_modulus": float(np.max(np.abs(ev))),
                    "residual": float(abs(H[k_done, k_done - 1] * y[-1])), "ritz_real_top3": [float(q) for q in np.sort(ev.real)[::-1][:3]],
                    "field": ("N" if j_ < n else "v"), "site": site, "R_over_rs": (site + (0.5 if a.half_offset else 1.0)) / a.rs,
                    "component": float(abs(vec[j_]) / max(float(np.linalg.norm(vec)), 1e-300))})
    out["seconds"] = time.time() - t0
    print("T/rs %.3f newton (%s) v6 soft gain at the kept iterate: g %s (1 - g %s; imag %s; Arnoldi residual %s after %d products; largest |Ritz| %s; Ritz re top 3 %s); its eigenvector's largest component %s of the norm in %s at site %s (R/rs %s)%s (%.0f s)" % (
        T / a.rs, name, fmt(out["gain"], "%.6f"), fmt(out["one_minus_gain"], "%.3e"), fmt(out["gain_imag"], "%.1e"), fmt(out["residual"], "%.1e"), k_done,
        fmt(out["largest_modulus"], "%.4f"), out["ritz_real_top3"], fmt(out["component"], "%.3f"), out["field"], out["site"], fmt(out["R_over_rs"], "%.3f"),
        (" [%s]" % why) if why else "", out["seconds"]), flush=True)
    return out


def displacement_report_v4(a, T, name, nk, x0v):
''', 1),

    ("homotopy_ramp_v6 and newton_record_v32's signature", r'''
def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None, ramp=None, ramp_bisect=2,
''', r'''
def homotopy_ramp_v6(a, chs, st, cache0, sources_with, sources_with_heavy, base_new, base_prev, f_prev, prev_kept, do_solve, x0, it_hist, T, ctr, solves, kw, ramp, bisect_max):
    """v6: the homotopy walk (C_s2_start_judgment_v3.md, section 2): coupling_ramp_v5's walk with 3 changes.
    (i) The family. At the first record of a run (no previous record) the light coupling, G_s = Solve(heavy sources + s x light sources), base s = 0 the
    uncoupled map from version 31's start (the 2 maps share the state there, so the homotopy is well-defined). At every later record the propagated
    covariance, G_t = Solve(sources at the base densities (1 - t) base_prev + t base_new, every channel's), base t = 0 the previous record's kept state
    with its kept residual (no stage run: G_0 at the previous applied fields returns the previous solve): version 32's two-stage continuation walked with
    bisection. The coupling ramp does not run after the first record: its base there is the uncoupled sources on the coupled history's seas, a third
    problem whose fixed point nobody holds (gate v5's 0.0625).
    (ii) The base rule and the floor rule. A base whose kept residual is below --light-ramp-base-above (default --end-above: the production's practice at
    T = 0, member2's record 0 written at 1.86e-04) is walked; a stage advances when it meets the stop rule, else, labelled "floor rule", when its kept
    residual is not above --light-ramp-floor-factor times the base's; a stage above both is retried over the half interval from the last advanced
    parameter, at most bisect_max times per target; a stall is recorded with its interval (the fold readout, read with the soft gain and the central
    cell); a closing stage at 1 runs from the last advanced iterate.
    (iii) Per stage the soft gain (soft_gain_v6, --light-ramp-gain-vectors products) beside version 5's central cell and displacement; every stage's arrays
    released at once (release_stage_v6) but the final stage's, the candidate of version 32's kept rule, which the caller applies; the device memory
    printed per stage. Inside a coupling stage the light sources and their exact J.v carry the weight LS["s"]; it is 1.0 everywhere else, so every other
    path's arithmetic is version 5's."""
    n = a.nsites
    t_r = time.time()
    first = base_prev is None or f_prev is None
    family = "coupling" if first else "covariance"
    pn = "s" if first else "t"
    base_above = a.light_ramp_base_above if a.light_ramp_base_above is not None else (a.end_above if a.end_above is not None else float("inf"))
    floor_factor = float(a.light_ramp_floor_factor)

    def stage_at(s, start, name, start_desc):
        if first:
            LS["s"] = float(s)
            try:
                if s == 0.0:
                    kwA = dict(kw); kwA.update(chl=(), light_on=False)
                    return nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with_heavy(base_new, c_)), do_solve, start, it_hist, T, name, start_desc, ctr, **kwA)
                return nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with(base_new, c_)), do_solve, start, it_hist, T, name, start_desc, ctr, **kw)
            finally:
                LS["s"] = 1.0
        base_t = {k_: tuple((1.0 - s) * p_ + s * q_ for p_, q_ in zip(base_prev[k_], base_new[k_])) for k_ in base_new}
        return nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with(base_t, c_)), do_solve, start, it_hist, T, name, start_desc, ctr, **kw)

    def app_of(stage):
        return (np.array(stage["best"]["app"][0], dtype=float), np.array(stage["best"]["app"][1], dtype=float))

    def adv_rule(sg, floor):
        if sg.get("best") is None:
            return None
        if bool(sg.get("conv")):
            return "stop rule"
        if np.isfinite(floor) and sg["best"]["resid"] <= floor:
            return "floor rule"
        return None

    def entry(stage, s, s_from, x_from, advanced_by):
        b = stage.get("best")
        m6 = mem_v6()
        e = {"family": family, "parameter": pn, "s": s, "s_from": s_from, "name": stage["name"], "status": stage["status"], "converged": bool(stage.get("conv")),
             "advanced_by": advanced_by, "evaluations": stage.get("evaluations"), "seconds": stage.get("seconds"), "kept_residual": (b["resid"] if b is not None else None),
             "confirming_residual": stage.get("r_c"), "confirming_dE": stage.get("dE_c"), "displacement": stage.get("v4_displacement"), "soft_gain": stage.get("soft_gain_v6"),
             "change_from_start": None, "central_cell": None, "memory_gib": ({"allocated": m6[0], "peak": m6[1], "reserved": m6[2]} if m6 is not None else None)}
        if b is not None:
            xa = np.concatenate([np.asarray(b["app"][0], dtype=float), np.asarray(b["app"][1], dtype=float)]); xf = np.concatenate([x_from[0], x_from[1]])
            d_ = np.abs(xa - xf); i_ = int(np.argmax(d_)); site = int(i_ if i_ < n else i_ - n)
            e["change_from_start"] = {"max_abs": float(d_[i_]), "field": ("N" if i_ < n else "v"), "site": site, "R_over_rs": (site + (0.5 if a.half_offset else 1.0)) / a.rs,
                                      "N_site0": float(xa[0]), "v_site0": float(xa[n])}
            e["central_cell"] = central_cell_v5(a, b["sol"][3])
        if advanced_by == "base":
            if e["converged"]:
                verdict = "the base met the stop rule"
            elif e["kept_residual"] is not None and e["kept_residual"] <= base_above:
                verdict = "the base is below the base rule %s" % fmt(base_above if np.isfinite(base_above) else None, "%g")
            else:
                verdict = "the base is ABOVE the base rule %s" % fmt(base_above if np.isfinite(base_above) else None, "%g")
        elif e["converged"]:
            verdict = "converged (the stop rule)"
        elif advanced_by:
            verdict = "advanced by the %s (%s)" % (advanced_by, stage["status"])
        else:
            verdict = "NOT advanced (%s)" % stage["status"]
        cc = e["central_cell"] or {}; ch_ = e["change_from_start"] or {}; sg_ = e["soft_gain"] or {}
        print("T/rs %.3f v6 ramp (%s) %s = %.6g (from %s = %.6g): %s; kept %s, confirming %s; soft gain g %s (1 - g %s; in %s at site %s); the kept applied fields moved %s from the stage's start (in %s at site %s, R/rs %s; N at site 0 %s); "
              "central cell at the kept iterate: N_dense(0) %s, W/2g at r = 1, 2, ell %s %s %s, rho_ell(0) %s, J_ell/v at r = 1 %s; device memory %s GiB (peak %s)" % (
                  T / a.rs, family, pn, s, pn, s_from, verdict, fmt(e["kept_residual"], "%.2e"), fmt(e["confirming_residual"], "%.2e"), fmt(sg_.get("gain"), "%.6f"),
                  fmt(sg_.get("one_minus_gain"), "%.3e"), sg_.get("field"), sg_.get("site"), fmt(ch_.get("max_abs"), "%.3e"), ch_.get("field"), ch_.get("site"),
                  fmt(ch_.get("R_over_rs"), "%.3f"), fmt(ch_.get("N_site0"), "%.7f"), fmt(cc.get("N_dense_0"), "%.7f"), fmt(cc.get("W_over_2g_at_1"), "%.4e"),
                  fmt(cc.get("W_over_2g_at_2"), "%.4e"), fmt(cc.get("W_over_2g_at_ell"), "%.4e"), fmt(cc.get("rho_ell_0"), "%.3e"), fmt(cc.get("J_over_v_at_1"), "%.4e"),
                  fmt(m6[0] if m6 is not None else None, "%.1f"), fmt(m6[1] if m6 is not None else None, "%.1f")), flush=True)
        return e

    targets = [float(s_) for s_ in ramp]
    out = {"family": family, "parameter": pn, "targets": targets, "bisect_max": int(bisect_max), "base_above": (base_above if np.isfinite(base_above) else None),
           "floor_factor": floor_factor, "gain_vectors": int(kw.get("gain_vectors", 0) or 0), "base": None, "walked": False, "stages": [], "bisections": 0,
           "reached_1": False, "stalled": None, "closing_stage": None, "final_kept_residual": None, "final_converged": None, "final_advanced_by": None, "label": None,
           "ctr_end": None, "seconds": 0.0, "final": None}
    if first:
        base = stage_at(0.0, x0, "v6 ramp base, s = 0 (the light coupling off)", "the first solve's output (version 31's start)")
        solves.append(base)
        print_stage_v32(a, T, base)
        if base["best"] is None:
            out.update({"label": "v6: the homotopy (coupling) has no base stage evaluation (%s); the kept rule runs over the direct solve" % base["failed"],
                        "ctr_end": ctr["n"], "seconds": time.time() - t_r})
            return out
        base_kept = float(base["best"]["resid"]); base_conv = bool(base["conv"])
        out["base"] = entry(base, 0.0, 0.0, (np.asarray(x0[0], dtype=float), np.asarray(x0[1], dtype=float)), "base")
        x_cur = app_of(base)
        release_stage_v6(base)
    else:
        base_kept = float(prev_kept) if prev_kept is not None else float("inf"); base_conv = bool(np.isfinite(base_kept) and base_kept < a.stop_fields)
        x_cur = (np.array(f_prev[0], dtype=float), np.array(f_prev[1], dtype=float))
        out["base"] = {"family": family, "parameter": pn, "s": 0.0, "s_from": 0.0, "name": "the previous record's kept state (no stage run)", "converged": base_conv,
                       "status": ("the previous record met the stop rule" if base_conv else "the previous record's kept residual"), "advanced_by": "base",
                       "kept_residual": (base_kept if np.isfinite(base_kept) else None), "confirming_residual": None, "soft_gain": None, "change_from_start": None, "central_cell": None}
        print("T/rs %.3f v6 ramp (covariance) t = 0: the previous record's kept state (kept residual %s, %s the base rule %s; its soft gain is the previous record's)" % (
            T / a.rs, fmt(out["base"]["kept_residual"], "%.2e"), "below" if base_kept <= base_above else "ABOVE", fmt(base_above if np.isfinite(base_above) else None, "%g")), flush=True)
    base_ok = base_conv or base_kept <= base_above
    floor = floor_factor * base_kept
    s_cur = 0.0; final = None; stalled = None; s1_from = None
    cap = 3 * len(targets) + 2; n_st = 0; i = 0
    if base_ok:
        out["walked"] = True
        while i < len(targets) and stalled is None and n_st < cap:
            s_t = targets[i]; s_try = s_t; nb = 0
            while True:
                n_st += 1
                sg = stage_at(s_try, x_cur, "v6 ramp (%s) %s = %.6g" % (family, pn, s_try), "the homotopy's kept iterate at %s = %.6g" % (pn, s_cur))
                solves.append(sg)
                print_stage_v32(a, T, sg)
                adv = adv_rule(sg, floor)
                out["stages"].append(entry(sg, s_try, s_cur, x_cur, adv))
                if s_try >= 1.0:
                    s1_from = s_cur
                if adv is not None:
                    x_cur = app_of(sg); s_cur = s_try
                    if s_try >= 1.0:
                        if final is not None:
                            release_stage_v6(final)
                        final = sg
                    else:
                        release_stage_v6(sg)
                    break
                if s_try >= 1.0 and sg.get("best") is not None and (final is None or sg["best"]["resid"] < final["best"]["resid"]):
                    if final is not None:
                        release_stage_v6(final)
                    final = sg
                elif sg.get("best") is not None:
                    release_stage_v6(sg)
                if nb >= bisect_max or n_st >= cap:
                    stalled = {"s_from": s_cur, "s_to": s_try, "bisections": nb, "stages": n_st, "cap_reached": bool(n_st >= cap)}
                    break
                nb += 1; out["bisections"] += 1; s_try = 0.5 * (s_cur + s_try)
            if stalled is None and s_cur >= s_t:
                i += 1
        if stalled is None and s_cur < 1.0:
            stalled = {"s_from": s_cur, "s_to": (targets[i] if i < len(targets) else 1.0), "bisections": out["bisections"], "stages": n_st, "cap_reached": True}
    out["reached_1"] = bool(s_cur >= 1.0); out["stalled"] = stalled
    if (not base_ok or stalled is not None) and s_cur < 1.0 and s1_from != s_cur:
        sg = stage_at(1.0, x_cur, "v6 ramp (%s) closing stage, %s = 1" % (family, pn), "the homotopy's kept iterate at %s = %.6g" % (pn, s_cur))
        solves.append(sg)
        print_stage_v32(a, T, sg)
        e_ = entry(sg, 1.0, s_cur, x_cur, adv_rule(sg, floor)); e_["closing"] = True; out["stages"].append(e_)
        out["closing_stage"] = {k_: e_.get(k_) for k_ in ("converged", "advanced_by", "kept_residual", "confirming_residual", "status")}
        if sg.get("best") is not None and (final is None or bool(sg.get("conv")) or sg["best"]["resid"] < final["best"]["resid"]):
            if final is not None:
                release_stage_v6(final)
            final = sg
        elif sg.get("best") is not None:
            release_stage_v6(sg)
    out["final"] = final
    out["final_kept_residual"] = final["best"]["resid"] if (final is not None and final.get("best") is not None) else None
    out["final_converged"] = bool(final["conv"]) if final is not None else None
    out["final_advanced_by"] = adv_rule(final, floor) if final is not None else None
    tgt = ",".join(("%g" % s_) for s_ in targets)
    if not base_ok:
        how = "the base is above the base rule %s, so the homotopy was not walked and only the closing stage at %s = 1 ran from the base" % (fmt(base_above if np.isfinite(base_above) else None, "%g"), pn)
    elif out["reached_1"]:
        how = "reached %s = 1 (%s)" % (pn, "; ".join("%s %g by the %s" % (pn, q["s"], q["advanced_by"]) for q in out["stages"] if q.get("advanced_by")))
    else:
        how = ("stalled between %s = %g and %s = %g after %d bisection(s)%s: the branch of fixed points has no continuation there within the stop rule or the floor rule (the fold, if the soft gain rises to 1 there and the central cell turns) or needs finer steps" % (
            pn, stalled["s_from"], pn, stalled["s_to"], stalled["bisections"], (" (the stage cap %d reached)" % cap) if stalled.get("cap_reached") else ""))
    if out["final_converged"]:
        fin = "the %s = 1 stage met the stop rule (kept %.2e)" % (pn, out["final_kept_residual"])
    elif out["final_kept_residual"] is not None:
        fin = "the %s = 1 stage kept %.2e (%s)" % (pn, out["final_kept_residual"], "within the floor rule" if out["final_advanced_by"] else "above the floor rule")
    else:
        fin = "no %s = 1 stage" % pn
    out["label"] = "v6: the homotopy (%s) ran (base %s = 0 %s, kept %s; targets %s; %d stage(s), %d bisection(s); %s; %s)" % (
        family, pn, ("met the stop rule" if base_conv else ("below the base rule" if base_kept <= base_above else "ABOVE the base rule")), fmt(out["base"]["kept_residual"], "%.2e"), tgt,
        len(out["stages"]), out["bisections"], how, fin)
    out["ctr_end"] = ctr["n"]; out["seconds"] = time.time() - t_r
    print("T/rs %.3f %s (%.0f s)" % (T / a.rs, out["label"], out["seconds"]), flush=True)
    return out


def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T, chl=(), light_on=False, sgn_dev=None, S_host=None, ramp=None, ramp_bisect=2, prev_kept=None,
''', 1),

    ("newton_record_v32: the soft gain's products per stage when the ramp is on", r'''
    d = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, (Nn0, vn0), it_hist, T, "direct", "the first solve's output (version 31's start)", ctr, **kw)
''', r'''
    if ramp and light_on:                                                                               # v6: the soft gain's products per Newton stage (the direct stage included), the ramp on and the coupling on only
        kw["gain_vectors"] = int(getattr(a, "light_ramp_gain_vectors", 0) or 0)
    d = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, (Nn0, vn0), it_hist, T, "direct", "the first solve's output (version 31's start)", ctr, **kw)
''', 1),

    ("newton_record_v32: homotopy_ramp_v6 in place of coupling_ramp_v5", r'''
        ramp_v5 = coupling_ramp_v5(a, chs, st, cache0, sources_with_heavy, src_new, base_new, do_solve, (Nn0, vn0), it_hist, T, ctr, solves, kw, ramp, ramp_bisect)
''', r'''
        ramp_v5 = homotopy_ramp_v6(a, chs, st, cache0, sources_with, sources_with_heavy, base_new, base_prev, f_prev, prev_kept, do_solve, (Nn0, vn0), it_hist, T, ctr, solves, kw, ramp, ramp_bisect)   # v6: the homotopy (the coupling at the first record, the covariance after it); the name ramp_v5 kept for the lines below
''', 1),

    ("newton_record_v32: the converged final stage kept, the direct solve released", r'''
        final = ramp_v5["final"]; drop_arrays_v32(d); cont = ramp_v5["label"]
''', r'''
        final = ramp_v5["final"]; release_stage_v6(d); cont = "%s; kept from the homotopy's final stage, which meets the stop rule (v6)" % ramp_v5["label"]   # v6
''', 1),

    ("newton_record_v32: the kept rule at once, 1 candidate held", r'''
    elif (base_prev is None or f_prev is None) and start_cont == "coupling" and light_on and sources_with_heavy is not None:
''', r'''
    elif ramp_v5 is not None:                                                                            # v6: the homotopy ran: version 32's kept rule over the direct solve and its final stage at once (1 candidate's arrays held); the two-stage continuation is the covariance homotopy's special case and does not run beside it
        rf = ramp_v5.get("final")
        if rf is not None and rf.get("best") is not None and rf["best"]["resid"] < d["best"]["resid"]:
            final = rf; release_stage_v6(d); cont = "%s; kept from the homotopy's final stage (v6)" % ramp_v5["label"]
        else:
            if rf is not None and rf.get("best") is not None:
                release_stage_v6(rf)
            d["confirm_repeated"] = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)                   # the kept iterate's confirming step again, the history's last entry
            cont = "%s; kept from the direct solve (%s) (v6)" % (ramp_v5["label"], ("no final stage" if (rf is None or rf.get("best") is None) else "its final stage's kept residual is not smaller"))
    elif (base_prev is None or f_prev is None) and start_cont == "coupling" and light_on and sources_with_heavy is not None:
''', 1),

    ("newton_record_v32: version 5's late kept-rule block retired", r'''
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
''', r'''
    # v6: version 5's kept-rule block over the ramp's s = 1 stage is retired here: the kept rule ran at once after homotopy_ramp_v6 above, so that 1 candidate's arrays are held
''', 1),

    ("newton_record_v32: the record entry", r'''
        e32["light_coupling_ramp_v5"] = {k_: v_ for k_, v_ in ramp_v5.items() if k_ != "final"}
''', r'''
        e32["homotopy_ramp_v6"] = {k_: v_ for k_, v_ in ramp_v5.items() if k_ != "final"}                 # v6: the homotopy's record entry (version 5's key light_coupling_ramp_v5 retired)
        e32["soft_gain_direct_v6"] = d.get("soft_gain_v6")                                              # v6: the direct stage's soft gain at its kept iterate
''', 1),

    ("run: the previous record's kept residual passed (the covariance homotopy's base)", r'''
                ramp=getattr(a, "light_ramp", None), ramp_bisect=getattr(a, "light_ramp_bisect", 2),                                   # v5: the light-coupling ramp
''', r'''
                ramp=getattr(a, "light_ramp", None), ramp_bisect=getattr(a, "light_ramp_bisect", 2),   # v5: the light-coupling ramp
                prev_kept=(((recs[-1].get("solve_v32") or {}).get("kept_residual")) if (recs and recs[-1].get("E") is not None) else None),   # v6: the previous record's kept residual, the covariance homotopy's base
''', 1),

    ("run: the label (1)", r'''
            label += ("; v5: the light-coupling ramp at a record whose direct solve misses the stop rule (base s = 0 the uncoupled map from version 31's start, then s = %s by "
''', r'''
            label += ("; v6: the homotopy walk at a record whose direct solve misses the stop rule: at the first record the light coupling (base s = 0 the uncoupled map from version 31's start, then s = %s by "
''', 1),

    ("run: the label (2)", r'''
                          a.light_coupling_ramp, a.light_ramp_bisect, a.end_above))
''', r'''
                          a.light_coupling_ramp, a.light_ramp_bisect, a.end_above))
            label += ("; v6: after the first record the homotopy is in the propagated covariance from the previous record's kept state (version 32's two-stage continuation walked with the same bisection); "
                      "a base below the base rule %s is walked, a stage advances by the stop rule or within %g times the base's kept residual (the floor rule, labelled); the soft gain by %d Arnoldi products per stage" % (
                          a.light_ramp_base_above if a.light_ramp_base_above is not None else a.end_above, a.light_ramp_floor_factor, a.light_ramp_gain_vectors))
''', 1),

    ("run: the record's memory peak counted from its start", r'''
        tml_reset_v36()                                                                                 # v36
        TM["propagation_prev_s"] = last_prop_s
''', r'''
        tml_reset_v36()                                                                                 # v36
        mem_v6(reset=True)                                                                              # v6: the record's device memory peak counted from here
        TM["propagation_prev_s"] = last_prop_s
''', 1),

    ("run: the memory printed per record", r'''
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
''', r'''
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        m6_ = mem_v6()                                                                                  # v6: the device memory per record (the gate reads these lines)
        if m6_ is not None:
            print("T/rs %.3f v6 memory: allocated now %.2f GiB, this record's peak %.2f GiB, reserved %.2f GiB; the run's peak %.2f GiB" % (T / rs, m6_[0], m6_[1], m6_[2], MEM["peak_gib"]), flush=True)
''', 1),

    ("run: the collector after the propagation", r'''
        TML["propagation_s"] = time.time() - t_pl
        T += a.tau
''', r'''
        TML["propagation_s"] = time.time() - t_pl
        free_device_v6()                                                                                # v6: the record's stages collected, the allocator's free blocks returned
        T += a.tau
''', 1),

    ("run: the run's peak at its end", r'''
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
    save_fields()
''', r'''
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
    save_fields()
    m6_ = mem_v6()                                                                                      # v6: the run's peak device memory (the gate reads this line)
    print("v6 memory: the run's peak device memory allocated %.2f GiB%s" % (MEM["peak_gib"], ("; allocated now %.2f GiB, reserved %.2f GiB" % (m6_[0], m6_[2])) if m6_ is not None else ""), flush=True)
''', 1),

    ("main: the options", r'''
    ap.add_argument("--light-ramp-bisect", dest="light_ramp_bisect", type=int, default=2,
''', r'''
    ap.add_argument("--light-ramp-base-above", dest="light_ramp_base_above", type=float, default=None, help="(v6) the homotopy's base is walked when its kept residual is below this (default --end-above, the production's practice at T = 0)")   # v6
    ap.add_argument("--light-ramp-floor-factor", dest="light_ramp_floor_factor", type=float, default=2.0, help="(v6) the floor rule: a stage advances when its kept residual is not above this factor times the base's (the stop rule where met)")   # v6
    ap.add_argument("--light-ramp-gain-vectors", dest="light_ramp_gain_vectors", type=int, default=0, help="(v6) the soft gain's Arnoldi products of the exact J.v per Newton stage at its kept iterate (0: off; 24 in the gate)")   # v6
    ap.add_argument("--light-ramp-bisect", dest="light_ramp_bisect", type=int, default=2,
''', 1),

    ("main: the options' checks", r'''
        ap.error("--light-ramp-bisect must be nonnegative")
''', r'''
        ap.error("--light-ramp-bisect must be nonnegative")
    if a.light_ramp_floor_factor < 1.0:                                                                                                  # v6
        ap.error("--light-ramp-floor-factor must be at least 1")
    if a.light_ramp_gain_vectors < 0:                                                                                                    # v6
        ap.error("--light-ramp-gain-vectors must be nonnegative")
    if a.light_ramp_gain_vectors > 0 and a.nk_jvp != "exact":                                                                            # v6
        ap.error("--light-ramp-gain-vectors needs --nk-jvp exact (the products are the exact J.v)")
''', 1),

    ("the script name in the json", r'''
"script": "engine_twofield_v5.py
''', r'''
"script": "engine_twofield_v6.py
''', 4),
]


def _pattern(old):
    """the current text as a regular expression: every run of 2 or more spaces matches any such run, everything else literally."""
    return "\n".join(r" {2,}".join(re.escape(p) for p in re.split(r" {2,}", ln)) for ln in old.split("\n"))


def build(src_text):
    text = src_text
    for name, old, new, count in REPLACEMENTS:
        old = _t(old); new = _t(new)
        ms = list(re.finditer(_pattern(old), text))
        if len(ms) != count:
            raise SystemExit("engine_twofield_v6 builder: the current text of %r is found %d time(s) in the source, %d expected" % (name, len(ms), count))
        for m in reversed(ms):
            text = text[:m.start()] + new + text[m.end():]
    for needle in ("def homotopy_ramp_v6(", "def soft_gain_v6(", "def release_stage_v6(", "def mem_v6(", "gain_vectors=0):", "prev_kept=None,",
                   "ramp_v5 = homotopy_ramp_v6(", 'e32["homotopy_ramp_v6"]', "--light-ramp-gain-vectors"):
        if needle not in text:
            raise SystemExit("engine_twofield_v6 builder: %r is missing from the result" % needle)
    if "ramp_v5 = coupling_ramp_v5(" in text:
        raise SystemExit("engine_twofield_v6 builder: coupling_ramp_v5 is still called")
    compile(text, "engine_twofield_v6.py", "exec")
    return text


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="builds engine_twofield_v6.py over engine_twofield_v5.py from the replacement list in this file")
    ap.add_argument("--src", default=os.path.join(here, "engine_twofield_v5.py"), help="the version 5 engine (built)")
    ap.add_argument("--out", default=os.path.abspath(__file__), help="where the built engine is written (default: over this file)")
    ap.add_argument("--check", action="store_true", help="build and compile without writing")
    a = ap.parse_args()
    with open(a.src) as fh:
        src = fh.read()
    if "def homotopy_ramp_v6(" in src:
        raise SystemExit("engine_twofield_v6 builder: %s already carries version 6" % a.src)
    if "def coupling_ramp_v5(" not in src or "def nk_stage_v32(" not in src:
        raise SystemExit("engine_twofield_v6 builder: %s is not the built engine_twofield_v5.py" % a.src)
    text = build(src)
    print("engine_twofield_v6: %d replacements applied over %s; %d v6 marks; the result compiles" % (len(REPLACEMENTS), a.src, text.count("v6:") + text.count("# v6")), flush=True)
    if a.check:
        return
    with open(a.out, "w") as fh:
        fh.write(text)
    print("written %s (%d lines)" % (a.out, text.count("\n") + 1), flush=True)


if __name__ == "__main__":
    main()
