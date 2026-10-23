#!/usr/bin/env python
"""build_engine_collapse_v36v.py (busi lane, W2 12.625; busi/notes/W2_pass12625_ruling_reviewa_v1.md sec. 3, reviewer A ruling w2_pass12625): writes
engine_collapse_v36v.py from engine_collapse_v36p_fixv4c.py (md5 b623d3b0, the W2 production engine) by the ruling's ten exact string replacements,
in build_engine_collapse_v36u.py's form. Each OLD is the ruling's, copied verbatim; each must occur exactly once in the source (else the build aborts);
replacing every NEW by its OLD (in reverse order) must restore the source byte for byte (checked here); every inserted or changed line carries the
marker "[v36v dg]" (checked here); the source's md5 must be fixv4c's (else the build aborts); the output file must not exist (never overwritten).

Adaptations to the real source: none in the OLD blocks. All ten OLD blocks of the ruling match engine_collapse_v36p_fixv4c.py (b623d3b0) exactly once as written
(OLD 1 line 908, OLD 2 line 2416, OLD 3 lines 2463-2477, OLD 4 line 2483, OLD 5 lines 4409-4410, OLD 6 line 4460, OLD 7 line 3039, OLD 8 line 2963,
OLD 9 line 2982, OLD 10 line 2944); the NEW blocks are the ruling's verbatim; the source's docstring is left unchanged (the ruling has no docstring
replacement). One adaptation of a NEW, marker only: NEW 4's first line (the prim entry line, which gains a trailing comma) carries the
ruling's text plus "; [v36v dg] the comma before the stall entry" at the end of its comment, so that every changed line is marked (no code change).

Two options, both off by default, so that the defaults are fixv4c bitwise in every record number, printed line, stored array and entry key (the json's
"args" dict alone gains dg_stall and dg_sub_counts):
  --dg-stall S (0 = off): the dg sub-step mixer's stall rule: after S iterates without the node-field residual falling to 0.7 of the sub-step's best,
        the Anderson history is purged and the trust region halved (floor --dg-max-change/64); at a tracked-set change after iterate S the damping
        is halved as well (floor --dg-beta/16); each event printed as "v36v dg T/rs ... sub-step j/n iteration i: <cause>: beta ..., trust region ...,
        the Anderson history purged (event k of this sub-step)"; the sub-step entry gains stall_events, beta_final, max_change_final.
  --dg-sub-counts LIST (absent = 2^L): the step's piece counts by level (e.g. 1,2,3,4,5,6,8); level L runs the step as LIST[L] equal pieces;
        --dg-sub-start and --dg-sub-max index the list (sub_max <= len - 1); the retry line reads "the next piece count (level L: n pieces of tau t)".
Both need --step dg --dg-chart frozen.
Usage: python build_engine_collapse_v36v.py [engine_collapse_v36p_fixv4c.py path] [engine_collapse_v36v.py path]   (defaults: beside this script)."""
import hashlib
import os
import py_compile
import sys

MARK = "[v36v dg]"
SRC_MD5 = "b623d3b010fa846cffcf914b4fa4900a"

REPLACEMENTS = [
("1 DGK defaults: stall off, sub_counts None",
r'''DGK.update({"chain_trigger": True, "attr": False})   # v36l attr: gate 1 as a trigger (version 36k) or a measurement; the chord-switch attribution's extra solves
''',
r'''DGK.update({"chain_trigger": True, "attr": False})   # v36l attr: gate 1 as a trigger (version 36k) or a measurement; the chord-switch attribution's extra solves
DGK.update({"stall": 0, "stall_fall": 0.7, "stall_beta_floor": 16.0, "stall_clip_floor": 64.0, "sub_counts": None})   # [v36v dg] W2 12.625: the sub-step mixer's stall rule (--dg-stall S, 0 off) and the step's piece counts by level (--dg-sub-counts, None: 2^L); the defaults are version 36p bitwise
'''),
("2 dgk_substep_v36k: beta_eff, clip_eff, the stall state",
r'''    mixer = DgkAnderson_v36k(DGK["m"], DGK["beta"]) if DGK["mix"] == "anderson" else None   # v36k chart
    tol = DG["tol"] * max(abs(E_rec), 1e-300)   # v36k chart
''',
r'''    mixer = DgkAnderson_v36k(DGK["m"], DGK["beta"]) if DGK["mix"] == "anderson" else None   # v36k chart
    beta_eff = float(DGK["beta"]); clip_eff = float(DGK["max_change"]); best_df = None; best_it = 0; shrinks = []   # [v36v dg] the damping and trust region in force (the nominal values, and with --dg-stall 0 always), the sub-step's smallest node-field residual, the iterate it last fell by the factor at, the stall events
    tol = DG["tol"] * max(abs(E_rec), 1e-300)   # v36k chart
'''),
("3 dgk_substep_v36k: the stall rule and the mixer in force",
r'''        if abs(dE) < tol and dfields < a.stop_fields and changed == 0:   # v36k chart
            conv = True   # v36k chart
            break   # v36k chart
        xf = flat(x); gf = flat(x_new)   # v36k chart
        if mixer is not None:   # v36k chart
            if changed:   # v36k chart
                mixer.reset()   # v36k chart
            xm, _used = mixer.step(xf, gf)   # v36k chart
        else:   # v36k chart
            xm = xf + DGK["beta"] * (gf - xf)   # v36k chart
        if DGK["max_change"] > 0.0:   # v36k chart: the trust region: the whole update scaled so that no node's fields move by more than --dg-max-change
            big = float(np.max(np.abs(xm - xf)))   # v36k chart
            if big > DGK["max_change"]:   # v36k chart
                xm = xf + (xm - xf) * (DGK["max_change"] / big); clipped += 1   # v36k chart
        x = unflat(xm)   # v36k chart
''',
r'''        if abs(dE) < tol and dfields < a.stop_fields and changed == 0:   # v36k chart
            conv = True   # v36k chart
            break   # v36k chart
        if DGK["stall"] > 0:   # [v36v dg] the stall rule (W2 12.6641 sub-step 6/16 at tau 0.5: node 1 inside the window of the avoided crossing 2042/2043, its residual on a 4.2e-3 floor from iterate 7 to 43 with the index fixed and the Anderson proposals clipped 20 times, the index flipping at 32-33 and 44-50, the fixed point found at iterate 60 at weight 0.58): the history is purged and the steps bounded instead of the iterate wandering
            if best_df is None:   # [v36v dg]
                best_df = dfields; best_it = it   # [v36v dg]
            else:   # [v36v dg]
                if dfields < DGK["stall_fall"] * best_df:   # [v36v dg] the residual fell by the factor: the stall clock restarts
                    best_it = it   # [v36v dg]
                if dfields < best_df:   # [v36v dg]
                    best_df = dfields   # [v36v dg]
            S_ = int(DGK["stall"]); cause = None; b_new = beta_eff   # [v36v dg]
            if it - best_it >= S_:   # [v36v dg] a stall: the secants are degenerate (the node's output pinned on a branch while the iterate moves): the history is purged and the trust region halved; the damping is kept, the Picard pull being what carries the iterate across the switching surface
                cause = "stalled: the node-field residual has not fallen to %.2g of its best %.2e in %d iterates" % (DGK["stall_fall"], best_df, it - best_it)   # [v36v dg]
            elif changed and it > S_:   # [v36v dg] a flip after the start's transient: the iterate crossed a switching surface; the damping and the trust region are halved so that it does not overshoot the fixed point next to the surface
                cause = "the tracked set changed at %d node-channel(s)" % changed; b_new = max(0.5 * beta_eff, float(DGK["beta"]) / float(DGK["stall_beta_floor"]))   # [v36v dg]
            if cause is not None:   # [v36v dg]
                c_new = (max(0.5 * clip_eff, float(DGK["max_change"]) / float(DGK["stall_clip_floor"])) if clip_eff > 0.0 else clip_eff)   # [v36v dg]
                beta_eff, clip_eff = b_new, c_new; best_it = it; shrinks.append([int(it), cause, beta_eff, clip_eff])   # [v36v dg]
                if mixer is not None:   # [v36v dg]
                    mixer.reset(); mixer.beta = beta_eff   # [v36v dg]
                print("  v36v dg T/rs %.4f sub-step %d/%d iteration %d: %s: beta %.4g, trust region %.2e, the Anderson history purged (event %d of this sub-step)" % (T0 / rs, j + 1, nsub, it, cause, beta_eff, clip_eff, len(shrinks)), flush=True)   # [v36v dg]
        xf = flat(x); gf = flat(x_new)   # v36k chart
        if mixer is not None:   # v36k chart
            if changed:   # v36k chart
                mixer.reset()   # v36k chart
            xm, _used = mixer.step(xf, gf)   # v36k chart
        else:   # v36k chart
            xm = xf + beta_eff * (gf - xf)   # v36k chart; [v36v dg] beta_eff is DGK["beta"] with --dg-stall 0
        if clip_eff > 0.0:   # v36k chart: the trust region: the whole update scaled so that no node's fields move by more than --dg-max-change; [v36v dg] clip_eff is DGK["max_change"] with --dg-stall 0
            big = float(np.max(np.abs(xm - xf)))   # v36k chart
            if big > clip_eff:   # v36k chart; [v36v dg]
                xm = xf + (xm - xf) * (clip_eff / big); clipped += 1   # v36k chart; [v36v dg]
        x = unflat(xm)   # v36k chart
'''),
("4 dgk_substep_v36k entry: stall_events (option on only)",
r'''           **({"prim": dgp_entry_v36p()} if DGP["on"] else {})}   # v36p prim: the primitive's decision, verification and printability of this sub-step (no key under the defaults)
''',
r'''           **({"prim": dgp_entry_v36p()} if DGP["on"] else {}),   # v36p prim: the primitive's decision, verification and printability of this sub-step (no key under the defaults); [v36v dg] the comma before the stall entry
           **({"stall_events": shrinks, "beta_final": beta_eff, "max_change_final": clip_eff} if DGK["stall"] > 0 else {})}   # [v36v dg] the stall events (iterate, cause, beta, trust region) and the values in force at the end (no key with --dg-stall 0)
'''),
("5 argparse: --dg-stall, --dg-sub-counts",
r'''    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:
''',
r'''    ap.add_argument("--dg-stall", dest="dg_stall", type=int, default=0, help="(v36v) the dg sub-step mixer's stall rule (W2 12.625: a Gauss node inside an avoided crossing's window): after this many iterates without the node-field residual falling to 0.7 of the sub-step's best, the Anderson history is purged and the trust region halved (floor --dg-max-change/64); at a tracked-set change after this many iterates the damping is halved as well (floor --dg-beta/16); 0: off (version 36p bitwise)")   # [v36v dg]
    ap.add_argument("--dg-sub-counts", dest="dg_sub_counts", default=None, help="(v36v) the step's piece counts by level as a comma list, e.g. 1,2,3,4,5,6,8: level L runs the step as the L-th entry's equal pieces (the attempt order on a failed sub-step; a count that puts the Gauss nodes outside a wide avoided crossing's window converges where the halving, which shrinks the pieces against the window's fixed clock width, cannot); --dg-sub-start and --dg-sub-max index the list; absent: 2^L (version 36k)")   # [v36v dg]
    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:
'''),
("6 main: DGK stall, sub_counts and their checks",
r'''    DGK["chain_trigger"] = (a.dg_chain_trigger == "on"); DGK["attr"] = (a.dg_attr == "on")   # v36l attr
''',
r'''    DGK["chain_trigger"] = (a.dg_chain_trigger == "on"); DGK["attr"] = (a.dg_attr == "on")   # v36l attr
    DGK["stall"] = int(a.dg_stall); DGK["sub_counts"] = ([int(q_) for q_ in str(a.dg_sub_counts).split(",") if q_.strip()] if a.dg_sub_counts else None)   # [v36v dg]
    if DGK["stall"] < 0 or ((DGK["stall"] > 0 or DGK["sub_counts"] is not None) and not DGK["on"]):   # [v36v dg]
        ap.error("version 36v: --dg-stall must be nonnegative; --dg-stall and --dg-sub-counts act under --step dg --dg-chart frozen")   # [v36v dg]
    if DGK["sub_counts"] is not None and (any(q_ < 1 for q_ in DGK["sub_counts"]) or DGK["sub_max"] > len(DGK["sub_counts"]) - 1):   # [v36v dg]
        ap.error("version 36v: --dg-sub-counts needs positive counts and --dg-sub-max at most len(counts) - 1 (the levels index the list)")   # [v36v dg]
'''),
("7 run label: v36v description",
r'''        label += "; v36l: " + dgk_desc_v36l(a)   # v36l attr
''',
r'''        label += "; v36l: " + dgk_desc_v36l(a)   # v36l attr
        if DGK["stall"] > 0 or DGK["sub_counts"] is not None:   # [v36v dg]
            label += "; v36v: " + dgk_desc_v36v(a)   # [v36v dg]
'''),
("8 dg_step_v36k: nsub from dgk_nsub_v36v",
r'''        nsub = 2 ** level; tau_s = float(a.tau) / nsub   # v36k chart
''',
r'''        nsub = dgk_nsub_v36v(level); tau_s = float(a.tau) / nsub   # v36k chart; [v36v dg] 2^level, or the --dg-sub-counts entry of the level
'''),
("9 dg_step_v36k: the retry line names the next count",
r'''        print("T/rs %.4f v36k chart: %s%s" % (T / rs, why, (("; the sub-step halved (level %d)" % (level + 1)) if level < DGK["sub_max"] else ("; the deepest level (%d) reached" % level))), flush=True)   # v36k chart
''',
r'''        print("T/rs %.4f v36k chart: %s%s" % (T / rs, why, (("; the sub-step halved (level %d)" % (level + 1)) if (level < DGK["sub_max"] and DGK["sub_counts"] is None) else ("; the next piece count (level %d: %d pieces of tau %.4g)" % (level + 1, dgk_nsub_v36v(level + 1), float(a.tau) / dgk_nsub_v36v(level + 1))) if level < DGK["sub_max"] else ("; the deepest level (%d) reached" % level))), flush=True)   # v36k chart; [v36v dg] the next count named under --dg-sub-counts
'''),
("10 helpers dgk_nsub_v36v, dgk_desc_v36v",
r'''def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart
''',
r'''def dgk_nsub_v36v(level):   # [v36v dg]
    """version 36v: the number of equal pieces a step is cut into at the given level: 2^level (version 36k), or under --dg-sub-counts the list's entry at that index. W2 12.625 (R1, level 4, sub-step 6/16): the avoided crossing of channel 1's levels 2042/2043 has gap_min 2.35e-4 and alpha_T 1.63e-3, so its window (w_mix 0.9: |delta| < 1.33 gap_min) spans 0.38 a of clock while the Gauss nodes are 0.387 tau_s apart (0.19 a at tau 0.5): a halving moves nodes into the window, another count moves them out."""   # [v36v dg]
    cl = DGK.get("sub_counts")   # [v36v dg]
    return int(cl[int(level)]) if cl else 2 ** int(level)   # [v36v dg]


def dgk_desc_v36v(a):   # [v36v dg]
    """version 36v: the options in words (the run label)."""   # [v36v dg]
    parts = []   # [v36v dg]
    if DGK["sub_counts"] is not None:   # [v36v dg]
        parts.append("the step's piece counts by level %s (the attempt order from level %d to at most %d; a quiet record lowers the level as in version 36k)" % (",".join(str(q_) for q_ in DGK["sub_counts"]), int(DGK["level_min"]), int(DGK["sub_max"])))   # [v36v dg]
    if DGK["stall"] > 0:   # [v36v dg]
        parts.append("the sub-step mixer's stall rule: after %d iterates without the node-field residual falling to %.2g of its best the Anderson history is purged and the trust region halved (floor max_change/%g); at a tracked-set change after iterate %d the damping is halved as well (floor beta/%g)" % (DGK["stall"], DGK["stall_fall"], DGK["stall_clip_floor"], DGK["stall"], DGK["stall_beta_floor"]))   # [v36v dg]
    return "; ".join(parts)   # [v36v dg]


def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart
'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36v: the OLD string of '%s' occurs %d times in the source (1 required): the source is not engine_collapse_v36p_fixv4c.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36v: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36p_fixv4c.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36v.py")
    if os.path.exists(dst):
        raise SystemExit("build_engine_collapse_v36v: %s exists; refusing to overwrite" % dst)
    raw = open(src, "rb").read()
    if md5(raw) != SRC_MD5:
        raise SystemExit("build_engine_collapse_v36v: the source md5 is %s, not fixv4c's %s" % (md5(raw), SRC_MD5))
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36v: the reverse replacements do not restore the source (a NEW text overlaps another)")
    srcset = set(text.splitlines())
    unmarked = [l_ for l_ in out.splitlines() if l_ not in srcset and MARK not in l_]
    if unmarked:
        raise SystemExit("build_engine_collapse_v36v: %d new lines without the marker, the first: %r" % (len(unmarked), unmarked[0][:200]))
    marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    with open(dst, "xb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines, every new line marked; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
