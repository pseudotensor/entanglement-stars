#!/usr/bin/env python
"""build_engine_twofield_v8.py (the busi lane, Computation C; the fix ruling busi/notes/C64_fix_ruling_v4.md section I.2, inbox busi #193 item 2):
writes engine_twofield_v8.py, the ruling's engine_twofield_v4 (filed as version 8: engine_twofield_v4.py to v7.py exist in this lane with other
contents; the run tags ..._twofield_v4L and the directories results/C64v4L, results/C32v4L stay the ruling's), from engine_twofield_v3.py (input
02) by exact string replacements, the form of build_engine_twofield_v7.py. Each OLD below is copied from version 3; the only tolerance in the
match is the run of spaces the file uses to align a trailing comment (2 or more spaces before "#" match any such run; the code and the comment
text are exact). Every OLD must occur exactly once in version 3 or the build stops and writes nothing; the result is compiled (a syntax check)
before it is written. Every line the replacements add or change carries the mark "v8 lagged (the ruling's v4)" (engine_twofield_v8_selftest.py
marks checks it line by line). The implicit path keeps every statement of version 3: where a lagged branch is inserted in front of version 3's
`if a.reference == "R2":` that line becomes the `elif` of the same statements, and the version 3 calls that move under an `else` are re-indented
unchanged; nothing else on that path is touched.
The changes, by function and place (the build prints each with its version 3 and version 8 line numbers):
  1. the module docstring (version 8's paragraph before version 3's);
  2. the v8 block before light_eval_v36: lagged_v8, lagged_desc_v8, held_cinfo_v8, light_eval_held_v8 (item (a)'s light part),
     transport_references_v8 (item (c)), lag_evaluation_v8 (item (d)), lagged_close_v8, lag_line_v8;
  3. run(): the run label; the head's entry; the plain iteration loop's heavy continue_ref_v34 call at (N_app, v_app) and its light_eval_v36 call
     (item (a) on that path); the transport and the lag evaluation after E = FLD.charge(info, a.g) (items (c), (d)); the record's keys and label
     after rec["engine"] (items (d), (f)); the exact J.v's engine_v35 label (item (b)); the lag line after the record line; the 3 json "script"
     names; the fields npz's light_meta (the two references of a lagged run named); the P2 readout's heavy current with the coupling on (J minus the
     light source as it entered the solve, the held reference's; the red team's R2);
  4. light_follow_v36(): the json "script" name;
  5. newton_record(): evalG's heavy continue_ref_v34 call and its light_eval_v36 call (item (a));
  6. nk_stage_v32(): evalG's heavy continue_ref_v34 call and its light_eval_v36 call (item (a)); jvp's heavy and light reference-derivative
     loops (the Uo, Uu, M, X terms; item (b));
  7. main(): the argument --reference-solve {implicit, lagged} and its 2 checks (--reference R2 required; no --light-follow).
What the lagged mode is in this engine: the record's covariance (the base densities) is the state propagated at the end of the previous record
and is fixed within the record, so the sources' only within-record dependence on the trial fields is the references' continuation; with the
references held at P_-(n-1) the record's map is constant, the direct solve meets the stop rule at its first evaluation (the residual 0 at the
lagged first solve's output, version 3's start) and its confirming step is the second; the lag residual of item (d), ||G(x_n; P_-(n)) - x_n||,
is then the implicit solve's first Newton residual from the same start. The build changes none of this arithmetic; it states it.
Usage: python build_engine_twofield_v8.py --v3 PATH/engine_twofield_v3.py --out PATH/engine_twofield_v8.py"""
import argparse
import re
import sys

MARK = "v8 lagged (the ruling's v4)"


def _s(x):
    """the snippets below start on the line after the opening quotes: that first newline is not part of them."""
    return x[1:] if x.startswith("\n") else x


def pattern_of(old):
    """old as a regex: exact everywhere except that a run of 2 or more spaces before '#' matches any such run (the comment alignment)."""
    parts = re.split(r"( {2,}#)", old)
    out = []
    for p in parts:
        if re.fullmatch(r" {2,}#", p):
            out.append("[ ]{2,}#")
        else:
            out.append(re.escape(p))
    return re.compile("".join(out))


# ------------------------------------------------------------------------------------------------------------------------- the replacements
R = []   # (name, where, old, new): every old occurs exactly once in version 3

# 1. the docstring
R.append(("docstring", "the module docstring: version 8's paragraph before version 3's", _s(r'''
"""engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every'''), _s(r'''
"""engine_twofield_v8.py (the busi lane, Computation C; the fix ruling busi/notes/C64_fix_ruling_v4.md section I.2, inbox busi #193 item 2): the
ruling's engine_twofield_v4, lagged, filed as version 8 because engine_twofield_v4.py to v7.py exist in this lane with other contents (the run tags
..._twofield_v4L and the directories results/C64v4L, results/C32v4L stay the ruling's). Version 3 with one run argument, --reference-solve
{implicit, lagged}, default implicit, under which every number this file writes is version 3's (the json names this script and its args carry the
argument; nothing else differs: no statement of the implicit path changes, the 3 `if a.reference == "R2"` of the continuation sites becoming the
`elif` of the lagged branch). Every line the build adds or changes carries the mark "v8 lagged (the ruling's v4)" (build_engine_twofield_v8.py;
engine_twofield_v8_selftest.py marks checks it line by line). In lagged mode, the ruling's items (a) to (f):
(a) the per-evaluation continuation calls are not made: the heavy channels' continue_ref_v34 inside the 3 evaluations (run()'s plain iteration
    loop, newton_record's evalG, nk_stage_v32's evalG) and light_continue_v36 inside light_eval_v36 at the same 3 places (light_eval_held_v8
    takes the light eigendecompositions alone); every evaluation computes the sources with the previous accepted record's references P_-(n-1)
    (st[...]["ref"], the stored ref columns) against the record's covariance step (the base densities of the state propagated at the end of the
    previous record, as in version 3). In this engine the sources' only within-record dependence on the trial fields is the references', so with
    them held the record's map x -> G(x) is constant: the direct solve meets the stop rule at its first evaluation (the residual 0 at the lagged
    first solve's output, version 3's start) and its confirming plain step is the second; the two-stage continuation, the line search, the stop
    rule and the end rule run unchanged on that solve (item (e)); an end by the end rule cannot arise from the kept residual (0): a failed field
    solve or the P2 stop end a lagged run as they end a version 3 run.
(b) the exact J.v (nk_stage_v32.jvp) omits the references' derivative terms (the heavy and the light loops over Uo, Uu, M, X); the field solve's
    central difference, version 3's response to a source change, is then not entered (version 3's own sc == 0 branch: J.v = -dx, the Jacobian of
    the held map), and the record's engine_v35 label reads "exact in the covariance step and the field solve, the reference held".
(c) after the record's acceptance (run(), right after E = FLD.charge(info, a.g)), transport_references_v8 transports each field's reference once
    from h(x_(n-1)) (the window's last entry) to h(x_n) (the accepted applied fields) by the existing continuation on the accepted evaluation's own
    eigendecomposition: continue_ref_v34 (the Weyl test, the LZ convention as the fallback) on the heavy channels, light_continue_v36 (the mu S
    walk with the Cholesky ladder, the LZ convention as the fallback, the cut check) on the light; the cache entries, the counters, the record's
    continuation and light entries, the labels, the window, the slices, the handoff and the P2 readout then read the transported reference exactly
    as version 3 reads its accepted iterate's.
(d) lag_evaluation_v8: one extra evaluation at the accepted fields with the transported references, the plain map applied once (the sources with
    P_-(n) and one field solve, no eigendecomposition): its field change max(max|N - N_app|, max|v - v_app|) is the lag residual
    ||G(x_n; P_-(n)) - x_n|| (the distance of the lagged solution from the implicit map's fixed point; from the same start it equals the implicit
    solve's first Newton residual, so it exceeds the stop rule wherever the implicit solve needed a Newton step), its collar charge E_cons; the json
    record gets "reference_solve": "lagged" and "lag": {"residual", "E_cons", "d_eps_total": {"heavy", "light"}, "d_J_total": {"heavy", "light"},
    "d_E_inv_light", "max_site_abs_d_eps_outside_cell"}: the consistent minus the solve's sources at x_n per field (the light's over its sourcing
    channels, 16 kappa weighted, computed with the coupling off too), the light's E_inv change Tr[h(x_n) (P_-(n-1) - P_-(n))] summed likewise, and
    the largest per-site |d eps| of the fed total outside the central cell (R > ell); the record's timing carries lag_s; the line "lag v8 T/rs ..."
    prints it. The fields npz rows carry the solve's sources (the held reference's), the handoff's eps_R2 and the stresses the transported
    reference's: their difference is the lag.
(e) nothing else changes: the stop rule and the end rule on the solve's kept residual, the two-stage continuation, the halving line search, the
    argument set, the slices (the references in the handoff as now), the P2 readout, the collar readouts.
(f) the record label carries "reference transported, held in the solve"; the run label and the head carry the scheme's sentence; the LABEL file
    (pod_launch_C_v11.sh) carries section I.2's sentence.
Lagged mode needs --reference R2 (the held reference is the continued sea) and excludes --light-follow (no record solve there).
Version 3's docstring follows.
engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every''')))

# 2. the v8 block, before light_eval_v36 (after light_continue_v36, which it calls at the transport)
BLOCK8 = r'''
# -------------------------------------------------- v8 lagged (the ruling's v4): the transported-reference scheme (busi/notes/C64_fix_ruling_v4.md I.2)
LAG8_SITES = {}                                                                                          # v8 lagged (the ruling's v4): the record's per-site light lag (the consistent P_-(n) minus the solve's P_-(n-1) fed light sources), read by the P2 readout's heavy split; never written to the json


def lagged_v8(a):                                                                                        # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4): whether --reference-solve lagged is on; False on every path version 3 ran, whose statements are then the ones executed."""   # v8 lagged (the ruling's v4)
    return getattr(a, "reference_solve", "implicit") == "lagged"                                        # v8 lagged (the ruling's v4)


def lagged_desc_v8():                                                                                    # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4): the scheme's sentence (the run label and the head)."""               # v8 lagged (the ruling's v4)
    return ("the references transported along the accepted path once per record and held through the record's solve: every evaluation of the "   # v8 lagged (the ruling's v4)
            "record's map computes the sources with the previous accepted record's references P_-(n-1) (the stored ref columns) against the "   # v8 lagged (the ruling's v4)
            "record's covariance step; after acceptance each field's reference is transported once from h(x_(n-1)) to h(x_n) by the existing "   # v8 lagged (the ruling's v4)
            "continuation (the mu S walk with the LZ convention as the fallback on the light, Weyl on the heavy) and stored as the record's "   # v8 lagged (the ruling's v4)
            "reference with its route, pockets, inversions, weights and cinfo as in version 3; one extra evaluation at the accepted fields with "   # v8 lagged (the ruling's v4)
            "the transported references gives the lag residual ||G(x_n; P_-(n)) - x_n|| and E_cons; the exact J.v omits the references' "   # v8 lagged (the ruling's v4)
            "derivative terms (exact in the covariance step and the field solve, the reference held); the stop rule and the end rule apply to "   # v8 lagged (the ruling's v4)
            "the solve's kept residual unchanged (engine_twofield_v8 --reference-solve lagged, the ruling's engine_twofield_v4 lagged)")   # v8 lagged (the ruling's v4)


def held_cinfo_v8():                                                                                     # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4): the evaluation's continuation entry while the reference is held; the transport's replaces it after acceptance."""   # v8 lagged (the ruling's v4)
    return {"mode": "held: the previous accepted record's reference P_-(n-1), transported after acceptance (v8 lagged, the ruling's v4)",   # v8 lagged (the ruling's v4)
            "nsub": 1, "verified": None, "fallback": False, "_mask": None, "_sigma": None}             # v8 lagged (the ruling's v4)


def light_eval_held_v8(a, chl, st, cache_, N_a, v_a):                                                    # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4), item (a): light_eval_v36 without light_continue_v36: the light channels' eigendecompositions at the trial fields (the same call and accounting) and the previous accepted record's references into cache_."""   # v8 lagged (the ruling's v4)
    for chL in chl:                                                                                      # v8 lagged (the ruling's v4)
        t_ = time.time()                                                                                 # v8 lagged (the ruling's v4)
        e, U = eigT(clocked_gpu(chL, N_a, v_a))                                                          # v8 lagged (the ruling's v4)
        sync_dev()                                                                                       # v8 lagged (the ruling's v4)
        TML["eigh_s"] += time.time() - t_; TML["eigh_n"] += 1                                            # v8 lagged (the ruling's v4)
        cache_[chL.key] = (None, e, U, st[chL.key]["ref"], held_cinfo_v8())                             # v8 lagged (the ruling's v4)


def transport_references_v8(a, chs, chl, st, cache, N_app, v_app, sgn_dev, S_host):                      # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4), item (c): after the record's acceptance each field's reference is transported once from h(x_(n-1)) (the window's last entry) to h(x_n) (the accepted applied fields) by the existing continuation on the accepted evaluation's own eigendecomposition: continue_ref_v34 on the heavy channels (Weyl, the LZ convention as the fallback), light_continue_v36 on the light (the mu S walk, the convention as the fallback); the cache entries and the counters as the evaluation writes them in implicit mode."""   # v8 lagged (the ruling's v4)
    for ch in chs:                                                                                       # v8 lagged (the ruling's v4)
        _hm, e, U, _held, _ci = cache[ch.kappa]                                                          # v8 lagged (the ruling's v4)
        t_c = time.time()                                                                                # v8 lagged (the ruling's v4)
        s_ = st[ch.kappa]                                                                                # v8 lagged (the ruling's v4)
        ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])      # v8 lagged (the ruling's v4)
        cinfo["_mask"] = mask                                                                            # v8 lagged (the ruling's v4)
        TM["continuation_s"] += time.time() - t_c                                                        # v8 lagged (the ruling's v4)
        TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1                                     # v8 lagged (the ruling's v4)
        cache[ch.kappa] = (None, e, U, ref, cinfo)                                                       # v8 lagged (the ruling's v4)
    for chL in chl:                                                                                      # v8 lagged (the ruling's v4)
        _hm, e, U, _held, _ci = cache[chL.key]                                                           # v8 lagged (the ruling's v4)
        t_c = time.time()                                                                                # v8 lagged (the ruling's v4)
        s_ = st[chL.key]                                                                                 # v8 lagged (the ruling's v4)
        ref, cinfo, mask = light_continue_v36(a, chL, s_["window"], (N_app, v_app), e, U, s_["n_sea"], sgn_dev, S_host)   # v8 lagged (the ruling's v4)
        cinfo["_mask"] = mask                                                                            # v8 lagged (the ruling's v4)
        TML["continuation_s"] += time.time() - t_c                                                       # v8 lagged (the ruling's v4)
        c_ = cinfo.get("certified")                                                                      # v8 lagged (the ruling's v4)
        if isinstance(c_, dict):                                                                         # v8 lagged (the ruling's v4)
            TML["walk_pieces"] += int(c_.get("pieces") or 0); TML["walk_cholesky"] += int(c_.get("cholesky_calls") or 0)   # v8 lagged (the ruling's v4)
        if cinfo.get("fallback"):                                                                        # v8 lagged (the ruling's v4)
            TML["walk_fallbacks"] += 1                                                                   # v8 lagged (the ruling's v4)
        if cinfo.get("cut_crossing"):                                                                    # v8 lagged (the ruling's v4)
            TML["cut_crossings"] += 1                                                                    # v8 lagged (the ruling's v4)
        cache[chL.key] = (None, e, U, ref, cinfo)                                                        # v8 lagged (the ruling's v4)


def lag_evaluation_v8(a, chl, st, cache, srcs, N_app, v_app, sources, do_solve, R):                      # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4), item (d): the plain map applied once at the accepted fields with the transported references (the sources with P_-(n) and one field solve; no eigendecomposition): its field change max(max|N - N_app|, max|v - v_app|) is the lag residual ||G(x_n; P_-(n)) - x_n||, its collar charge E_cons; per field the totals of the consistent minus the solve's sources (the light's over its sourcing channels, 16 kappa weighted, with the coupling off too), the light's E_inv change Tr[h(x_n) (P_-(n-1) - P_-(n))] summed likewise, and the largest per-site |d eps| of the fed total outside the central cell (R > ell); the seconds into the record's timing as lag_s."""   # v8 lagged (the ruling's v4)
    torch = BK["torch"]                                                                                  # v8 lagged (the ruling's v4)
    t0 = time.time()                                                                                     # v8 lagged (the ruling's v4)
    srcs_c = sources(cache)                                                                              # v8 lagged (the ruling's v4)
    eps_c, J_c, eps_pairs_c, _eps_dust_c, J_pairs_c = srcs_c                                             # v8 lagged (the ruling's v4)
    eps_s, J_s, eps_pairs_s, _eps_dust_s, J_pairs_s = srcs                                               # v8 lagged (the ruling's v4)
    d_eps_L = np.zeros(a.nsites); d_J_L = np.zeros(a.nsites); d_Einv = 0.0                               # v8 lagged (the ruling's v4)
    for chL in chl:                                                                                      # v8 lagged (the ruling's v4)
        if not chL.sourcing:                                                                             # v8 lagged (the ruling's v4)
            continue                                                                                     # v8 lagged (the ruling's v4)
        _hm, e, U, ref_c, _ci = cache[chL.key]; ref_h = st[chL.key]["ref"]                               # v8 lagged (the ruling's v4)
        d_eps_L += chL.w * (dens(chL.h, ref_h) - dens(chL.h, ref_c)); d_J_L += chL.w * (dens(chL.Q, ref_h) - dens(chL.Q, ref_c))   # v8 lagged (the ruling's v4)
        e64 = e.to(torch.float64)                                                                        # v8 lagged (the ruling's v4)
        E_ref_h = float((e64 * ((U.conj().T @ ref_h).abs() ** 2).sum(dim=1)).sum())                      # v8 lagged (the ruling's v4)
        E_ref_c = float((e64 * ((U.conj().T @ ref_c).abs() ** 2).sum(dim=1)).sum())                      # v8 lagged (the ruling's v4)
        d_Einv += chL.w * (E_ref_h - E_ref_c)                                                            # v8 lagged (the ruling's v4)
    LAG8_SITES["d_eps_light"] = d_eps_L; LAG8_SITES["d_J_light"] = d_J_L                                # v8 lagged (the ruling's v4): per site, over the sourcing channels, 16 kappa weighted (J_light_fed - d_J_light is the light current as it entered the solve)
    d_tot = np.asarray(eps_c, float) - np.asarray(eps_s, float)                                          # v8 lagged (the ruling's v4)
    out_cell = np.asarray(R, float) > a.ell                                                              # v8 lagged (the ruling's v4)
    Nc, vc, _Wc, info_c = do_solve(srcs_c)                                                               # v8 lagged (the ruling's v4)
    failed = info_c.get("failed")                                                                        # v8 lagged (the ruling's v4)
    out = {"residual": (None if failed else max(float(np.max(np.abs(np.asarray(Nc, float) - np.asarray(N_app, float)))), float(np.max(np.abs(np.asarray(vc, float) - np.asarray(v_app, float)))))),   # v8 lagged (the ruling's v4)
           "E_cons": (None if failed else FLD.charge(info_c, a.g)),                                     # v8 lagged (the ruling's v4)
           "d_eps_total": {"heavy": float(np.sum(np.asarray(eps_pairs_c, float) - np.asarray(eps_pairs_s, float))), "light": float(d_eps_L.sum())},   # v8 lagged (the ruling's v4)
           "d_J_total": {"heavy": float(np.sum(np.asarray(J_pairs_c, float) - np.asarray(J_pairs_s, float))), "light": float(d_J_L.sum())},   # v8 lagged (the ruling's v4)
           "d_E_inv_light": d_Einv, "max_site_abs_d_eps_outside_cell": (float(np.max(np.abs(d_tot[out_cell]))) if out_cell.any() else None)}   # v8 lagged (the ruling's v4)
    TM["lag_s"] = time.time() - t0                                                                       # v8 lagged (the ruling's v4)
    if failed:                                                                                           # v8 lagged (the ruling's v4)
        print("lag v8: the plain map's field solve at the accepted fields with the transported references failed (%s); the lag residual and E_cons are None on this record" % failed, flush=True)   # v8 lagged (the ruling's v4)
    return out                                                                                           # v8 lagged (the ruling's v4)


def lagged_close_v8(a, chs, chl, st, cache, srcs, N_app, v_app, sources, do_solve, sgn_dev, S_host, R):  # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4): items (c) and (d) after the record's acceptance, the transport then the lag evaluation; returns the record's "lag" entry."""   # v8 lagged (the ruling's v4)
    transport_references_v8(a, chs, chl, st, cache, N_app, v_app, sgn_dev, S_host)                       # v8 lagged (the ruling's v4)
    return lag_evaluation_v8(a, chl, st, cache, srcs, N_app, v_app, sources, do_solve, R)                # v8 lagged (the ruling's v4)


def lag_line_v8(trs, a, E, lag):                                                                         # v8 lagged (the ruling's v4)
    """v8 lagged (the ruling's v4): the per-record line of item (d)."""                                 # v8 lagged (the ruling's v4)
    Ec = lag.get("E_cons")                                                                               # v8 lagged (the ruling's v4)
    return ("  lag v8 T/rs %.3f [reference transported, held in the solve]: lag residual %s (the stop rule %g); E_cons %s, E %.10f, E_cons - E %s; "   # v8 lagged (the ruling's v4)
            "d eps total heavy %s light %s; d J total heavy %s light %s; d E_inv light (sourcing) %s; max |d eps| outside the cell %s (%.2f s)") % (   # v8 lagged (the ruling's v4)
        trs, fmt(lag.get("residual"), "%.3e"), a.stop_fields, fmt(Ec, "%.10f"), E, fmt((Ec - E) if Ec is not None else None, "%+.3e"),   # v8 lagged (the ruling's v4)
        fmt(lag["d_eps_total"]["heavy"], "%+.3e"), fmt(lag["d_eps_total"]["light"], "%+.3e"), fmt(lag["d_J_total"]["heavy"], "%+.3e"),   # v8 lagged (the ruling's v4)
        fmt(lag["d_J_total"]["light"], "%+.3e"), fmt(lag.get("d_E_inv_light"), "%+.3e"), fmt(lag.get("max_site_abs_d_eps_outside_cell"), "%.3e"), TM.get("lag_s", 0.0))   # v8 lagged (the ruling's v4)
'''
R.append(("v8 block", "the module: lagged_v8, lagged_desc_v8, held_cinfo_v8, light_eval_held_v8, transport_references_v8, lag_evaluation_v8, lagged_close_v8, lag_line_v8 before light_eval_v36",
          "\n\ndef light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host):\n",
          "\n\n" + BLOCK8.strip("\n") + "\n\n\ndef light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host):\n"))

# 3. run(): the run label
R.append(("run label", "run(): the run label after v36_desc (item (f))", _s(r'''
    label += "; v36: " + v36_desc(a)                                                                   # v36'''), _s(r'''
    label += "; v36: " + v36_desc(a)                                                                   # v36
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): (f) the run label
        label += "; v8 lagged (the ruling's engine_twofield_v4 lagged): " + lagged_desc_v8()           # v8 lagged (the ruling's v4)''')))

# 4. run(): the head's entry
R.append(("head entry", "run(): the head's reference_solve_v8 entry before the stress objects", _s(r'''
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    cstrL = {chL.key: STR.ChanStress(chL, torch, BK["dev"]) for chL in chl}                               # v36'''), _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): the head names the scheme
        head["reference_solve_v8"] = {"mode": "lagged", "description": lagged_desc_v8()}                # v8 lagged (the ruling's v4)
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    cstrL = {chL.key: STR.ChanStress(chL, torch, BK["dev"]) for chL in chl}                               # v36''')))

# 4b. run(): the fields npz's light_meta names the two references of a lagged run (the red team's R2)
R.append(("light_meta", "run(): light_meta's reference_solve_v8 note (the solve's sources against the stored reference's)", _s(r'''
                  "eps_tot_includes_fed_light": True, "note": "eps_heavy = eps_tot - eps_light_fed; eps_L<k>, J_L<k> are the channel's unweighted sources (times 16 kappa when fed)"}'''), _s(r'''
                  "eps_tot_includes_fed_light": True, "note": "eps_heavy = eps_tot - eps_light_fed; eps_L<k>, J_L<k> are the channel's unweighted sources (times 16 kappa when fed)"}
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): the fields npz's two references named
        light_meta["reference_solve_v8"] = ("lagged: eps_tot, J_tot, eps_pairs, J_pairs are the solve's sources (the held reference P_-(n-1)); eps_light_fed, J_light_fed, eps_L<k>, J_L<k> "   # v8 lagged (the ruling's v4)
                                            "are the stored reference's P_-(n), transported after acceptance; eps_heavy = eps_tot - eps_light_fed holds up to the light lag, the record's lag.d_eps_total.light")   # v8 lagged (the ruling's v4)''')))

# 5. run(): the plain iteration loop (item (a) on that path)
R.append(("plain loop", "run(): the plain iteration loop's heavy continue_ref_v34 call at (N_app, v_app) and its light_eval_v36 call (item (a))", _s(r'''
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)
            for ch in chs:
                e, U = eig.pop(ch.kappa)
                if a.reference == "R2":
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])
                    cinfo["_mask"] = mask
                    TM["continuation_s"] += time.time() - t_c
                    TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
                else:
                    ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
                cache[ch.kappa] = (None, e, U, ref, cinfo)
            del eig
            light_eval_v36(a, chl, st, cache, N_app, v_app, sgn_dev, S_host)                            # v36
            srcs = sources(cache)'''), _s(r'''
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)
            for ch in chs:
                e, U = eig.pop(ch.kappa)
                if lagged_v8(a):                                                                         # v8 lagged (the ruling's v4): (a) the heavy reference held through the iteration, no continuation here
                    ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                   # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
                elif a.reference == "R2":                                                               # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])
                    cinfo["_mask"] = mask
                    TM["continuation_s"] += time.time() - t_c
                    TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
                else:
                    ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
                cache[ch.kappa] = (None, e, U, ref, cinfo)
            del eig
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the light eigendecompositions without light_continue_v36
                light_eval_held_v8(a, chl, st, cache, N_app, v_app)                                      # v8 lagged (the ruling's v4)
            else:                                                                                        # v8 lagged (the ruling's v4)
                light_eval_v36(a, chl, st, cache, N_app, v_app, sgn_dev, S_host)                            # v36; v8 lagged (the ruling's v4): version 3's call, under the else
            srcs = sources(cache)''')))

# 6. run(): the transport and the lag evaluation after the acceptance (items (c), (d))
R.append(("transport and lag", "run(): after E = FLD.charge(info, a.g), before the reference's diagnostics (items (c), (d))", _s(r'''
        best = None
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)
        # the reference's diagnostics on the accepted state; ||[h_m, R2]||_F per channel; the continuation's record'''), _s(r'''
        best = None
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)
        lag_v8 = None                                                                                   # v8 lagged (the ruling's v4)
        if lagged_v8(a):                                                                                # v8 lagged (the ruling's v4): (c) the references transported once along the accepted path, (d) the lag evaluation
            lag_v8 = lagged_close_v8(a, chs, chl, st, cache, srcs, N_app, v_app, sources, do_solve, sgn_dev, S_host, R)   # v8 lagged (the ruling's v4)
        # the reference's diagnostics on the accepted state; ||[h_m, R2]||_F per channel; the continuation's record''')))

# 7. run(): the record's keys and label (items (d), (f))
R.append(("record keys", "run(): after rec[\"engine\"] (items (d), (f))", _s(r'''
        rec["engine"] = "engine_collapse_v25"'''), _s(r'''
        rec["engine"] = "engine_collapse_v25"
        if lagged_v8(a):                                                                                # v8 lagged (the ruling's v4): (d) the record's keys, (f) the record label
            rec["reference_solve"] = "lagged"; rec["lag"] = lag_v8                                      # v8 lagged (the ruling's v4)
            rec["record_label"] += "; reference transported, held in the solve (engine_twofield_v8 --reference-solve lagged, the ruling's engine_twofield_v4 lagged)"   # v8 lagged (the ruling's v4)''')))

# 8. run(): the exact J.v's label (item (b))
R.append(("engine_v35 label", "run(): the exact J.v's engine_v35 label (item (b))", _s(r'''
            rec["engine_v35"] = "the Newton direction by the exact J.v (--nk-jvp exact; engine spec 40.4 (2) (b))"'''), _s(r'''
            rec["engine_v35"] = "the Newton direction by the exact J.v (--nk-jvp exact; engine spec 40.4 (2) (b))"
            if lagged_v8(a):                                                                            # v8 lagged (the ruling's v4): (b) the label
                rec["engine_v35"] = "the Newton direction by the exact J.v, exact in the covariance step and the field solve, the reference held (--nk-jvp exact with --reference-solve lagged: the references' derivative terms omitted; v8 lagged, the ruling's v4)"   # v8 lagged (the ruling's v4)''')))

# 9. run(): the lag line after the record line
R.append(("lag line", "run(): the lag line after the record line (item (d))", _s(r'''
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if chl:                                                                                         # v36: the light line'''), _s(r'''
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if lagged_v8(a) and lag_v8 is not None:                                                         # v8 lagged (the ruling's v4): the lag line
            print(lag_line_v8(T / rs, a, E, lag_v8), flush=True)                                        # v8 lagged (the ruling's v4)
        if chl:                                                                                         # v36: the light line''')))

# 9b. run(): the P2 readout's heavy current with the coupling on (the red team's R2: version 3's J - J_light_fed subtracts the stored reference's light source,
#     which in lagged mode is the transported one, while the J that sourced the fields carries the held one; the difference, the light lag, sat in the
#     heavy dust's current J_dust = J - J_pairs of s11_record: ball_source_min_outside_cell, J_ell_ball, ball_beyond.energy_density_route; the stop rule
#     and its case classification read info and were never affected; with the coupling off version 3's form stands)
R.append(("s11 heavy J", "run(): the P2 readout's heavy current with the coupling on: J minus the light source as it entered the solve (the held reference's)", _s(r'''
        elif chl and a.c_readouts:                                                                      # v36: the mass function's sign per record, the 3 cases (P2's stop rule)
            t_11 = time.time()
            s11, s11_arr = s11_record(a, info, R, rs, eps, J - J_light_fed, eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T)'''), _s(r'''
        elif chl and a.c_readouts:                                                                      # v36: the mass function's sign per record, the 3 cases (P2's stop rule)
            t_11 = time.time()
            if lagged_v8(a) and light_on and lag_v8 is not None:                                        # v8 lagged (the ruling's v4): with the coupling on the heavy current handed to the P2 readout is the solve's: J minus the light source as it entered the solve, the held reference's (J_light_fed is the transported reference's; their per-site difference is the lag's d_J_light); with the coupling off version 3's form stands
                s11, s11_arr = s11_record(a, info, R, rs, eps, J - (J_light_fed - LAG8_SITES["d_J_light"]), eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T)   # v8 lagged (the ruling's v4)
            else:                                                                                       # v8 lagged (the ruling's v4)
                s11, s11_arr = s11_record(a, info, R, rs, eps, J - J_light_fed, eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T)   # v8 lagged (the ruling's v4): version 3's call, under the else''')))

# 10. to 13. the json "script" names
R.append(("script periodic", "run(): the periodic json dump's script name", _s(r'''
                json.dump({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v3'''), _s(r'''
                json.dump({"script": "engine_twofield_v8.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v3; v8 lagged (the ruling's v4): the script's name''')))
R.append(("script s11 stop", "run(): the P2 stop's json dump's script name", _s(r'''
                json.dump({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v3'''), _s(r'''
                json.dump({"script": "engine_twofield_v8.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v3; v8 lagged (the ruling's v4): the script's name''')))
R.append(("script final", "run(): the final json dump's script name", _s(r'''
        json.dump(dict({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v3'''), _s(r'''
        json.dump(dict({"script": "engine_twofield_v8.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v3; v8 lagged (the ruling's v4): the script's name''')))
R.append(("script light-follow", "light_follow_v36(): the json's script name", _s(r'''
                json.dump({"script": "engine_twofield_v3.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v3'''), _s(r'''
                json.dump({"script": "engine_twofield_v8.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v3; v8 lagged (the ruling's v4): the script's name''')))

# 14. newton_record(): evalG (item (a))
R.append(("newton_record evalG", "newton_record(): evalG's heavy continue_ref_v34 call and its light_eval_v36 call (item (a))", _s(r'''
    nk = {"best": None, "n": 0, "E_prev": E0, "failed": None}

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if a.reference == "R2":
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36
        srcs_ = sources(cache_)'''), _s(r'''
    nk = {"best": None, "n": 0, "E_prev": E0, "failed": None}

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the heavy reference held through the evaluation, no continuation here
                ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                       # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
            elif a.reference == "R2":                                                                   # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        if lagged_v8(a):                                                                                 # v8 lagged (the ruling's v4): (a) the light eigendecompositions without light_continue_v36
            light_eval_held_v8(a, chl, st, cache_, N_a, v_a)                                             # v8 lagged (the ruling's v4)
        else:                                                                                            # v8 lagged (the ruling's v4)
            light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36; v8 lagged (the ruling's v4): version 3's call, under the else
        srcs_ = sources(cache_)''')))

# 15. nk_stage_v32(): evalG (item (a))
R.append(("nk_stage_v32 evalG", "nk_stage_v32(): evalG's heavy continue_ref_v34 call and its light_eval_v36 call (item (a))", _s(r'''
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if a.reference == "R2":
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36
        srcs_ = sources_fn(cache_)'''), _s(r'''
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the heavy reference held through the evaluation, no continuation here
                ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                       # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
            elif a.reference == "R2":                                                                   # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        if lagged_v8(a):                                                                                 # v8 lagged (the ruling's v4): (a) the light eigendecompositions without light_continue_v36
            light_eval_held_v8(a, chl, st, cache_, N_a, v_a)                                             # v8 lagged (the ruling's v4)
        else:                                                                                            # v8 lagged (the ruling's v4)
            light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36; v8 lagged (the ruling's v4): version 3's call, under the else
        srcs_ = sources_fn(cache_)''')))

# 16. nk_stage_v32(): jvp's heavy reference-derivative loop (item (b))
R.append(("jvp heavy", "nk_stage_v32(): jvp's heavy reference-derivative loop, the Uo, Uu, M, X terms (item (b))", _s(r'''
        deps = np.zeros(n); dJ = np.zeros(n)
        for ch in chs:
            _, e, U, ref, cinfo = cache_[ch.kappa]'''), _s(r'''
        deps = np.zeros(n); dJ = np.zeros(n)
        for ch in chs:
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (b) the heavy reference's derivative terms (Uo, Uu, M, X) omitted: the reference is constant within the record
                break                                                                                    # v8 lagged (the ruling's v4): deps = dJ = 0, so version 3's sc == 0 branch returns J.v = -dx and enters no field solve
            _, e, U, ref, cinfo = cache_[ch.kappa]''')))

# 17. nk_stage_v32(): jvp's light reference-derivative loop (item (b))
R.append(("jvp light", "nk_stage_v32(): jvp's light reference-derivative loop (item (b))", _s(r'''
        if light_on:                                                                                    # v36: the light sourcing channels' dP (cross-sector pairs dropped: [dh, S] = 0)
            for chL in chl:
                if not chL.sourcing:
                    continue
                _, e, U, ref, cinfo = cache_[chL.key]'''), _s(r'''
        if light_on:                                                                                    # v36: the light sourcing channels' dP (cross-sector pairs dropped: [dh, S] = 0)
            for chL in chl:
                if lagged_v8(a):                                                                         # v8 lagged (the ruling's v4): (b) the light references' derivative terms omitted (the reference held)
                    break                                                                                # v8 lagged (the ruling's v4)
                if not chL.sourcing:
                    continue
                _, e, U, ref, cinfo = cache_[chL.key]''')))

# 18. main(): the argument
R.append(("argument", "main(): the run argument --reference-solve {implicit, lagged}", _s(r'''
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    a = ap.parse_args()'''), _s(r'''
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    ap.add_argument("--reference-solve", dest="reference_solve", default="implicit", choices=["implicit", "lagged"], help="(v8 lagged, the ruling's engine_twofield_v4) implicit: version 3 bitwise; lagged: the references held through the record's solve at the previous accepted record's and transported once along the accepted path after acceptance, the lag residual printed (busi/notes/C64_fix_ruling_v4.md I.2)")   # v8 lagged (the ruling's v4)
    a = ap.parse_args()''')))

# 19. main(): the lagged mode's checks
R.append(("checks", "main(): the lagged mode's 2 checks (--reference R2 required; no --light-follow)", _s(r'''
    if a.reference != "R2" and (a.light_kappas or a.light_guard_kappas):
        ap.error("v36: the light field's continued reference needs --reference R2")'''), _s(r'''
    if a.reference != "R2" and (a.light_kappas or a.light_guard_kappas):
        ap.error("v36: the light field's continued reference needs --reference R2")
    if lagged_v8(a) and a.reference != "R2":                                                             # v8 lagged (the ruling's v4)
        ap.error("v8 lagged (the ruling's v4): --reference-solve lagged holds the continued sea R2 through the record's solve: --reference R2 is required")   # v8 lagged (the ruling's v4)
    if lagged_v8(a) and a.light_follow:                                                                  # v8 lagged (the ruling's v4)
        ap.error("v8 lagged (the ruling's v4): --light-follow has no record solve; --reference-solve lagged does not apply to it")   # v8 lagged (the ruling's v4)''')))


def build(src):
    """every OLD exactly once in version 3's text (the line numbers taken there), then the replacements in order, each exactly once at its turn."""
    pos = []
    for name, where, old, new in R:
        hits = list(pattern_of(old).finditer(src))
        if len(hits) != 1:
            raise SystemExit("build: %r (%s) occurs %d times in version 3, expected exactly once; the version 3 text differs from input 02 at that place; nothing written" % (name, where, len(hits)))
        pos.append(src[:hits[0].start()].count("\n") + 1)
    out = src
    for name, where, old, new in R:
        hits = list(pattern_of(old).finditer(out))
        if len(hits) != 1:
            raise SystemExit("build: %r (%s) matched %d times at its turn, expected exactly once; nothing written" % (name, where, len(hits)))
        out = out[:hits[0].start()] + new + out[hits[0].end():]
    log = []
    for (name, where, old, new), l3 in zip(R, pos):
        k = out.find(new)
        log.append((name, where, l3, (out[:k].count("\n") + 1) if k >= 0 else -1))
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True, help="engine_twofield_v3.py (input 02)")
    ap.add_argument("--out", required=True, help="engine_twofield_v8.py to write")
    a = ap.parse_args()
    src = open(a.v3, encoding="utf-8").read()
    if "engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied" not in src:
        raise SystemExit("build: %s is not engine_twofield_v3.py" % a.v3)
    if MARK in src:
        raise SystemExit("build: %s already carries the mark %r; the source must be version 3" % (a.v3, MARK))
    out, log = build(src)
    compile(out, a.out, "exec")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    n_marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    print("built %s from %s: %d replacements, each matched exactly once; %d lines carry the mark %r; %d -> %d lines; compiled" % (
        a.out, a.v3, len(log), n_marked, MARK, len(src.splitlines()), len(out.splitlines())))
    for name, where, l3, l8 in log:
        print("  %-20s version 3 line %5d -> version 8 line %5d: %s" % (name, l3, l8, where))


if __name__ == "__main__":
    main()
