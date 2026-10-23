#!/usr/bin/env python
"""build_engine_collapse_v36p.py (reviewer A for the busi lane; PRIM_ruling_v1.md): writes engine_collapse_v36p.py from engine_collapse_v36l.py (the
file build_engine_collapse_v36l.py writes) by 36 exact string replacements in that builder's form (16 of PRIM_ruling_v1, the primitive's import chain v1 -> v1b -> v1c -> v1d; the v36p prim
hold re-decided inside the solve by W2_v36p_fix_v1, ended at the three-node rule's endpoint, settled, resolved and reverted by W2_v36p_fix_v2; the chord's end taken off the endpoint call by
W2_v36p_fix_v3: the endpoint's input fields as dgk_substep_v36k forms them, the node fields' quadratic extrapolation to s = 1 through the endpoint map's passes before its last, since the
engine's endpoint is a quasi-static solve and no cubic u(1); the flips and settle-max guard of W2_v36p_fix_v2 extended from a replacement's hold to every hold by W2_v36p_fix_v4, the
seed's hold reverting to the anchor's tracking (version 36l's) and its guard counting a tracked set off the hold from --dg-prim-settle iterates after the decision through the solve, only at
iterates whose node fields' change is at most 1000 times --dg-prim-redecide [v36p prim]). Each OLD is copied exactly from
engine_collapse_v36l.py (or from the text an earlier replacement inserted) and must occur exactly once (else the build aborts); each NEW contains inserted or changed text so that replacing every NEW
by its OLD restores the source byte for byte (checked here). Every inserted or changed code line carries the marker "v36p prim" (a trailing
comment; the docstring's lines "[v36p prim]"). The new option --dg-assign {anchor, prim} (default anchor) and the --dg-prim-* options are read
into DGP and used only under --dg-assign prim, so the defaults are version 36l bitwise in every record number, printed line, stored array and
entry key (the json's "args" dict alone gains the keys). Under --dg-assign prim version 36p differs from version 36l by: the chord's set at the
three nodes and the endpoint decided once per sub-step solve, at its first iterate, by the assignment primitive assign_primitive_v1 (the set
followed along the straight chord from the sub-step's start fields to the start's candidate endpoint, the declared rule at each near-crossing,
sub-stepped until stable under halving) and held through the solve (dgk_track_v36k tracks each node locally against its held set instead of the
anchor); the verification on the converged chord (the held sets must be the primitive's there); the sub-step entry's "prim" dict and the step
entry's "prim" summary; the printed decision, crossing, verification and printability lines; and the retry path of gate 1 taken by a sub-step whose
set is not stable under halving, not held or whose jumps do not fall (--dg-prim-halve on). The record solve is unchanged (the comparator).
Usage: python build_engine_collapse_v36p.py [engine_collapse_v36l.py path] [engine_collapse_v36p.py path]   (defaults: beside this script);
assign_primitive_v1.py must sit beside the built engine at run time."""
import hashlib
import os
import py_compile
import sys

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36l.py (reviewer A for the busi lane; W2 dg ruling v3): version 36l = version 36k with the review's edit B1 and two options, off by default, so that the [v36l attr]''',
r'''"""engine_collapse_v36p.py (reviewer A for the busi lane; PRIM_ruling_v1): version 36p = version 36l with the one assignment primitive, assign_primitive_v1, as the chord's [v36p prim]
set-following, under one option off by default, so that the defaults are version 36l bitwise in every record number, printed line, stored array and entry key (the json's [v36p prim]
"args" dict gains dg_assign and the dg_prim_* keys). Under --dg-assign prim: at each sub-step's first iterate the occupied set is followed along the straight chord in [v36p prim]
the fields from the sub-step's start fields f0 to the start's candidate endpoint (the quadratic Lagrange extrapolation of the start's node fields to s = 1), the declared [v36p prim]
rule (--dg-prim-rule lz: the record convention's LZ rule, tau_LZ = k gap_min^2/|alpha_T| against c_LZ; overlap; energy) applied at each near-crossing, the chord halved from [v36p prim]
level --dg-prim-level-min to at most --dg-prim-level-max until the decided set is stable under halving (assign_primitive_v1.follow_chord); the decided sets at the three [v36p prim]
nodes and at the endpoint, with their eigenvectors at the chord's points, are held through the solve as that sub-step's references (dgk_node_v36k receives the held set of [v36p prim]
its point in place of the anchor, so dgk_track_v36k tracks locally against it); at convergence the chord from f0 to the endpoint's input fields is followed again, and the [v36p prim]
held sets must be the primitive's there: the endpoint's mask equal, the nodes' masks equal as held and as tracked, the crossings equal (dgp_verify_v36p). The hold is a seed [v36p prim]
(W2_v36p_fix_v1, _v2): the chord's end inside the solve is the three-node rule's own endpoint, the cubic through f0 and the three nodes at s = 1 (the endpoint's input fields at the [v36p prim]
fixed point); whenever the node fields' relative change between iterates has fallen below --dg-prim-redecide, at least --dg-prim-settle iterates after the last decision, the chord is [v36p prim]
followed again and a hold differing from a resolved and stable decision (masks at the nodes and the endpoint, crossings) is replaced, the solve continuing under the new sets, at most [v36p prim]
--dg-prim-rounds times (an unstable or undecided decision replaces nothing); after a replacement a tracked node set leaving the hold --dg-prim-flips times, or a solve not settled [v36p prim]
--dg-prim-settle-max iterates later, reverts the hold to the one before (a crossing on a node: not held, the halving separates them); at a step's first sub-step the start's node [v36p prim]
fields are f0 at every node and the chord has no length, so the anchor seeds iterate 1 and the decision is made at iterate 2. The sub-step [v36p prim]
entry gains "prim" (the decision's summary, the verification, held, stable, jumps_fall, the resolved jumps' sum over E_rec, printable) and the step entry gains "prim" (the [v36p prim]
sub-steps' conjunction); the decision, crossing, verification and printability lines print as "v36p prim"; under --dg-prim-halve on a sub-step that is not stable, not held [v36p prim]
or whose jumps do not fall takes the retry path gate 1 took in version 36k (the sub-step halves). The record solve, its mixer and its continuation are unchanged (the [v36p prim]
comparator; its own hold is version 36q). Every changed code line of this file is marked "v36p prim". Version 36l's docstring follows. [v36p prim]
engine_collapse_v36l.py (reviewer A for the busi lane; W2 dg ruling v3): version 36l = version 36k with the review's edit B1 and two options, off by default, so that the [v36l attr]'''),
("module state",
r'''DGK.update({"chain_trigger": True, "attr": False})   # v36l attr: gate 1 as a trigger (version 36k) or a measurement; the chord-switch attribution's extra solves''',
r'''DGK.update({"chain_trigger": True, "attr": False})   # v36l attr: gate 1 as a trigger (version 36k) or a measurement; the chord-switch attribution's extra solves
DGP = {"on": False, "rule": "lz", "c_lz": 1.0, "lz_k": 1.0, "tie": "adiabatic", "level_min": 1, "level_max": 4, "w_mix": 0.9, "eig_tol": 1e-10, "jump_tol": 1e-9, "knife": 0.1, "mirror": "auto", "halve": True, "rounds": 3, "redecide": 1e-6, "settle": 3, "settle_max": 10, "flips": 2, "rule_obj": None, "hold": None, "decision": None, "decision_eigh": 0, "n_eigh": 0, "round": 0, "hold_hist": [], "dec_hist": [], "x_prev": None, "armed": True, "redecided_at": [], "seed_degenerate": False, "cycle": False, "exhausted": False, "decided_at": None, "off_hold": [], "revert": None, "unresolved_at": [], "f1_last": None}   # v36p prim: the assignment primitive's state (--dg-assign prim); W2_v36p_fix_v1: the hold's rounds inside the solve; W2_v36p_fix_v2: the settle count, the revert, the unresolved re-decisions, the last decision's chord end
try:   # v36p prim
    import assign_primitive_v1 as PRIM   # v36p prim: the one assignment primitive (busi/code/prim/assign_primitive_v1.py beside the engine)
except ImportError:   # v36p prim
    PRIM = None   # v36p prim'''),
("helpers before dg_step",
r'''def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart''',
r'''def dgp_desc_v36p(a):   # v36p prim
    """version 36p: the options in words (the run label)."""   # v36p prim
    return ("the chord's set by the assignment primitive %s: followed from the sub-step's start to the three-node rule's endpoint (the cubic through the start fields and the three nodes at s = 1) with the rule [%s] at each near-crossing, the chord halved from level %d to at most %d until the set is stable under halving (w_mix %.3g, eig_tol %.2g, mirror %s), decided at the sub-step solve's first iterate as the seed and re-decided whenever the node fields' relative change has fallen below %.1e at least %d iterates after the last decision (at most %d replacements of the hold, each by a resolved and stable decision; a tracked set leaving the hold %d times after a replacement, or a solve not settled %d iterates after it, reverts the hold: a crossing on a node, not held; W2_v36p_fix_v1, _v2), held at the three nodes and the endpoint, verified on the converged chord; a sub-step not stable, not held or with a jump that does not fall %s" % (   # v36p prim
        getattr(PRIM, "VERSION", "?"), DGP["rule_obj"].describe() if DGP["rule_obj"] is not None else DGP["rule"], DGP["level_min"], DGP["level_max"], DGP["w_mix"], DGP["eig_tol"], DGP["mirror"], DGP["redecide"], DGP["settle"], DGP["rounds"], DGP["flips"], DGP["settle_max"], "halves (gate 1's retry path)" if DGP["halve"] else "is printed not printable, no halving"))   # v36p prim


def dgp_f_v36p(x, f):   # v36p prim
    """version 36p: None-safe formatting."""   # v36p prim
    return (f % x) if x is not None else "-"   # v36p prim


def dgp_yn_v36p(v):   # v36p prim
    """version 36p: yes, NO or - (not measured)."""   # v36p prim
    return "-" if v is None else ("yes" if v else "NO")   # v36p prim


def dgp_tau_min_v36p(dec):   # v36p prim
    """version 36p (W2_v36p_fix_v1): the smallest tau_LZ over a decision's avoided crossings (gap_min^2 above its band; an exact crossing has tau_LZ 0 by construction and is left out), None when there is none."""   # v36p prim
    vals = [float(ev["A"]) for ev in dec.crossings if ev.get("A") is not None and ev.get("gap_min2") is not None and ev["gap_min2"] > (ev.get("gap_min2_err") or 0.0)]   # v36p prim
    return min(vals) if vals else None   # v36p prim


class DgpOracleV36p(object):   # v36p prim
    """version 36p: the primitive's oracle on the straight chord in the fields from f0 to f1: spectrum(s, channels) by eigh_all at N0 + s (N1 - N0), v0 + s (v1 - v0) (the eigenvalues as numpy, the eigenvectors on the device); the weights on the device; keep(s, kappa, e, U, mask) returns the held set in the anchor's format (ref, n, mask, info, e)."""   # v36p prim

    def __init__(self, a, chs, f0, f1):   # v36p prim
        self.a = a; self.chs = {ch.kappa: ch for ch in chs}; self.order = [ch.kappa for ch in chs]   # v36p prim
        self.N0 = np.asarray(f0[0], float); self.v0 = np.asarray(f0[1], float); self.N1 = np.asarray(f1[0], float); self.v1 = np.asarray(f1[1], float); self.n_eigh = 0   # v36p prim

    def spectrum(self, s, channels):   # v36p prim
        s = float(s); Nf = self.N0 + s * (self.N1 - self.N0); vf = self.v0 + s * (self.v1 - self.v0); want = set(channels)   # v36p prim
        sub = [self.chs[k_] for k_ in self.order if k_ in want]   # v36p prim
        eig = eigh_all(sub, Nf, vf, bool(self.a.streams)); self.n_eigh += len(sub)   # v36p prim
        out = {}   # v36p prim
        for ch in sub:   # v36p prim
            e, U = eig.pop(ch.kappa); out[ch.kappa] = (e.detach().cpu().numpy().astype(float), U)   # v36p prim
        del eig   # v36p prim
        return out   # v36p prim

    def columns(self, U, cols):   # v36p prim
        torch = BK["torch"]   # v36p prim
        return U[:, torch.as_tensor(np.asarray(list(cols), dtype=np.int64), device=U.device)]   # v36p prim

    def weights(self, A, B):   # v36p prim
        return ((A.conj().transpose(0, 1) @ B).abs() ** 2).detach().cpu().numpy().astype(float)   # v36p prim

    def project(self, A, B):   # v36p prim
        return ((A.conj().transpose(0, 1) @ B).abs() ** 2).sum(0).detach().cpu().numpy().astype(float)   # v36p prim

    def keep(self, s, kappa, e, U, mask):   # v36p prim
        m = np.asarray(mask, bool).copy(); ref = self.columns(U, np.flatnonzero(m))   # v36p prim
        return (ref, int(m.sum()), m, {"prim": True, "s": float(s)}, np.asarray(e, float).copy())   # v36p prim


def dgp_lagrange_v36p(nodes, s):   # v36p prim
    """version 36p: the Lagrange weights of the given nodes at s (the three nodes: quadratic; f0 and the three nodes: the cubic of W2_v36p_fix_v2)."""   # v36p prim
    L = []; n_ = len(nodes)   # v36p prim
    for k_ in range(n_):   # v36p prim
        w_ = 1.0   # v36p prim
        for m_ in range(n_):   # v36p prim
            if m_ != k_:   # v36p prim
                w_ *= (s - nodes[m_]) / (nodes[k_] - nodes[m_])   # v36p prim
        L.append(w_)   # v36p prim
    return L   # v36p prim


def dgp_point_anchor_v36p(anchor, k):   # v36p prim
    """version 36p: the held set at chord point k (0, 1, 2 the nodes, 3 the endpoint) under --dg-assign prim, else the anchor (version 36l)."""   # v36p prim
    h = DGP.get("hold")   # v36p prim
    if DGP["on"] and h is not None and h[k] is not None:   # v36p prim
        return h[k]   # v36p prim
    return anchor   # v36p prim


def dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, level_min, level_max):   # v36p prim
    """version 36p: the primitive on the straight chord f0 -> f1 with the anchor's masks as the start set and the nodes and s = 1 as the query points; returns (Decision, oracle)."""   # v36p prim
    orc = DgpOracleV36p(a, chs, f0, f1)   # v36p prim
    masks0 = {kap: np.asarray(anchor[kap][2], bool) for kap in anchor}   # v36p prim
    wch = {ch.kappa: float(ch.w) for ch in chs if ch.kappa in masks0}   # v36p prim
    dec = PRIM.follow_chord(orc, masks0, DGP["rule_obj"], queries=list(s_nodes) + [1.0], tau=float(tau_s), weights_ch=wch, level_min=int(level_min), level_max=int(level_max), w_mix=DGP["w_mix"], eig_tol=DGP["eig_tol"], jump_tol=DGP["jump_tol"], mirror=DGP["mirror"], keep=orc.keep, channels=[ch.kappa for ch in chs if ch.kappa in masks0])   # v36p prim
    return dec, orc   # v36p prim


def dgp_print_v36p(dec, tag, T_over_rs, j1, nsub, E_rec, n_eigh):   # v36p prim
    """version 36p: the primitive's levels and crossings (one line per level, one per crossing) and its verdict."""   # v36p prim
    for row in dec.level_table:   # v36p prim
        print("  v36p prim T/rs %.4f sub-step %d/%d %s level %d: channels %s at %d points; crossings %d (%d unresolved); stable at this level: %s" % (T_over_rs, j1, nsub, tag, row["level"], ",".join(row["channels"]), row["n_points"], row["n_crossings"], row["n_unresolved"], ",".join(row["stable"]) or "-"), flush=True)   # v36p prim
    for ev in sorted(dec.crossings, key=lambda q_: (str(q_["channel"]), q_["s_star"])):   # v36p prim
        kind = " with its mirror" if ev["mirrored"] else (" self-mirror" if ev["self_mirror"] else "")   # v36p prim
        print("  v36p prim T/rs %.4f sub-step %d/%d %s crossing: channel %s levels %d -> %d%s at s* %.4f (passages %s, crossed members %s): gap_min %s (grid %s at s %s), alpha_T %s, tau_LZ %s [%s, %s] against c %.3g: %s%s%s; jump at the grid %s of E (previous level %s), resolved %s of E%s%s" % (   # v36p prim
            T_over_rs, j1, nsub, tag, ev["channel"], ev["i"], ev["n"], kind, ev["s_star"], ev.get("passages", "-"), ev.get("crossed_members", "-"), dgp_f_v36p(ev["gap_min"], "%.3e"), dgp_f_v36p(ev["delta_grid"], "%.3e"), dgp_f_v36p(ev["s_grid"], "%.4f"),   # v36p prim
            dgp_f_v36p(ev["alpha_T"], "%.3e"), dgp_f_v36p(ev["A"], "%.3g"), dgp_f_v36p(ev["A_lo"], "%.3g"), dgp_f_v36p(ev["A_hi"], "%.3g"), DGP["rule_obj"].c, ev["decision"], "" if ev["decided"] else " (UNDECIDED)", "" if ev["resolved"] else " (UNRESOLVED)",   # v36p prim
            dgp_f_v36p((ev["jump_grid"] / E_rec) if (ev["jump_grid"] is not None and E_rec) else None, "%+.3e"), dgp_f_v36p((ev["jump_prev"] / E_rec) if (ev["jump_prev"] is not None and E_rec) else None, "%+.3e"),   # v36p prim
            dgp_f_v36p((ev["jump_fit"] / E_rec) if (ev["jump_fit"] is not None and E_rec) else None, "%+.3e"), ("" if ev["falls"] is None else (" falls" if ev["falls"] else " DOES NOT FALL")), ("; " + "; ".join(ev["flags"])) if ev["flags"] else ""), flush=True)   # v36p prim
    print("  v36p prim T/rs %.4f sub-step %d/%d %s: set stable under halving %s (levels %s); crossings %d: %d diabatic, %d adiabatic, %d unresolved; jumps fall %s; resolved jumps sum %s of E (at the grid %s); eigh %d; mirror %s%s" % (   # v36p prim
        T_over_rs, j1, nsub, tag, dgp_yn_v36p(dec.stable), ",".join("%s:%s" % (k_, v_ if v_ is not None else "-") for k_, v_ in sorted(dec.level_stable.items(), key=lambda t_: str(t_[0]))), len(dec.crossings), dec.n_diabatic, dec.n_adiabatic, dec.n_unresolved, dgp_yn_v36p(dec.jumps_fall),   # v36p prim
        dgp_f_v36p((dec.jump_fit_sum / E_rec) if E_rec else None, "%+.3e"), dgp_f_v36p((dec.jump_grid_sum / E_rec) if E_rec else None, "%+.3e"), n_eigh, ",".join("%s:%s" % (k_, "on" if v_ else "off") for k_, v_ in sorted(dec.mirror.items(), key=lambda t_: str(t_[0]))),   # v36p prim
        ("; flags: " + "; ".join("%s: %s" % (k_, " | ".join(v_)) for k_, v_ in dec.flags.items() if v_)) if any(dec.flags.values()) else ""), flush=True)   # v36p prim


def dgp_chord_end_v36p(f0, x, s_nodes):   # v36p prim
    """version 36p (W2_v36p_fix_v2): the chord's end from the node fields x: the three-node rule's own endpoint, the cubic Lagrange polynomial through (0, f0) and the three nodes at s = 1 (-f0 + (5 x1 - 4 x2 + 5 x3)/3 at the Gauss nodes), which is f0 + tau sum_j b_j F_j, the endpoint's input fields, at the fixed point (the cubic's derivative interpolates the chart's charge at the nodes); the quadratic through the nodes alone (W2_v36p_fix_v1) misses f0 and is short of the endpoint by q(0) - f0 (16 to 19 % of the chord at tau 8, 2 % at tau 4 on bench1), its crossings sitting at 0.84 and 0.98 of the verification's s*."""   # v36p prim
    L = dgp_lagrange_v36p([0.0] + list(s_nodes), 1.0)   # v36p prim
    pts = [(np.asarray(f0[0], float), np.asarray(f0[1], float))] + [(np.asarray(x[k_][0], float), np.asarray(x[k_][1], float)) for k_ in range(3)]   # v36p prim
    return (sum(L[i_] * pts[i_][0] for i_ in range(4)), sum(L[i_] * pts[i_][1] for i_ in range(4)))   # v36p prim


def dgp_copy_x_v36p(x):   # v36p prim
    """version 36p: the node fields as numpy copies (the previous iterate's, for the change)."""   # v36p prim
    return [(np.array(x[k_][0], dtype=float, copy=True), np.array(x[k_][1], dtype=float, copy=True)) for k_ in range(3)]   # v36p prim


def dgp_change_v36p(x, xp):   # v36p prim
    """version 36p: the node fields' relative change between iterates, max over nodes and fields of |x - xp|_inf over max |x|_inf; None without a previous iterate."""   # v36p prim
    if xp is None:   # v36p prim
        return None   # v36p prim
    num = 0.0; den = 0.0   # v36p prim
    for k_ in range(3):   # v36p prim
        for q_ in range(2):   # v36p prim
            cur = np.asarray(x[k_][q_], float); prv = np.asarray(xp[k_][q_], float)   # v36p prim
            if cur.size:   # v36p prim
                num = max(num, float(np.max(np.abs(cur - prv)))); den = max(den, float(np.max(np.abs(cur))))   # v36p prim
    return num / max(den, 1e-300)   # v36p prim


def dgp_hold_from_v36p(dec, anchor, s_nodes):   # v36p prim
    """version 36p: the held references from a decision: anchor-format tuples per channel at the three nodes and the endpoint (the anchor's where the primitive kept none)."""   # v36p prim
    hold = []   # v36p prim
    for q_ in list(s_nodes) + [1.0]:   # v36p prim
        kept = dec.kept.get(q_, {})   # v36p prim
        hold.append({kap: (kept[kap] if kap in kept else anchor[kap]) for kap in anchor})   # v36p prim
    return hold   # v36p prim


def dgp_hold_equal_v36p(h0, h1):   # v36p prim
    """version 36p: two holds are equal when their masks agree at every point and channel."""   # v36p prim
    if h0 is None or h1 is None or len(h0) != len(h1):   # v36p prim
        return False   # v36p prim
    return all(set(p0) == set(p1) and all(np.array_equal(np.asarray(p0[kap][2], bool), np.asarray(p1[kap][2], bool)) for kap in p0) for p0, p1 in zip(h0, h1))   # v36p prim


def dgp_resolved_v36p(dec):   # v36p prim
    """version 36p (W2_v36p_fix_v2): a decision fit to replace the hold: stable under halving, every crossing resolved and decided, the jumps falling (an unresolved or undecided crossing, decided by the tie, is the halving's matter, not a hold)."""   # v36p prim
    return bool(dec is not None and dec.stable and dec.jumps_fall and int(dec.n_unresolved) == 0)   # v36p prim


def dgp_off_hold_v36p(hold, masks_prev):   # v36p prim
    """version 36p (W2_v36p_fix_v2): the node-channels whose tracked mask of the previous iterate differs from the held mask: [(k, kappa)]."""   # v36p prim
    out = []   # v36p prim
    if hold is None or masks_prev is None:   # v36p prim
        return out   # v36p prim
    for k_ in range(3):   # v36p prim
        mk = masks_prev[k_] if (k_ < len(masks_prev) and masks_prev[k_] is not None) else None   # v36p prim
        if mk is None or hold[k_] is None:   # v36p prim
            continue   # v36p prim
        for kap in hold[k_]:   # v36p prim
            if (kap in mk) and not np.array_equal(np.asarray(mk[kap], bool), np.asarray(hold[k_][kap][2], bool)):   # v36p prim
                out.append((int(k_), kap))   # v36p prim
    return out   # v36p prim


def dgp_gap_v36p(f1, fe):   # v36p prim
    """version 36p (W2_v36p_fix_v2): the relative gap between two field pairs, max |f1 - fe|_inf over max |fe|_inf; None when either is missing."""   # v36p prim
    if f1 is None or fe is None:   # v36p prim
        return None   # v36p prim
    num = 0.0; den = 0.0   # v36p prim
    for q_ in range(2):   # v36p prim
        a_ = np.asarray(f1[q_], float); b_ = np.asarray(fe[q_], float)   # v36p prim
        if a_.size and a_.shape == b_.shape:   # v36p prim
            num = max(num, float(np.max(np.abs(a_ - b_)))); den = max(den, float(np.max(np.abs(b_))))   # v36p prim
    return num / max(den, 1e-300)   # v36p prim


def dgp_revert_v36p(it, ch_, why, T_over_rs, j1, nsub):   # v36p prim
    """version 36p (W2_v36p_fix_v2): the hold reverted to the one before the last replacement (the hold under which the fields were settled when the replacement was decided); the sub-step is flagged a cycle (a crossing on a node), re-decided no more, not held: the solve re-converges, the verification records it and the halving separates the crossing from the node."""   # v36p prim
    hh = DGP["hold_hist"]; dh = DGP["dec_hist"]   # v36p prim
    prev_h = hh[-2] if len(hh) >= 2 else None; prev_d = dh[-2] if len(dh) >= 2 else None; to_round = max(len(hh) - 2, 0)   # v36p prim
    DGP.update({"hold": prev_h, "decision": prev_d, "cycle": True, "armed": False, "revert": {"at": int(it), "to_round": int(to_round), "why": why, "off_hold": [[int(i_), [[int(k_), str(kap)] for k_, kap in o_]] for i_, o_ in DGP["off_hold"]]}})   # v36p prim
    print("  v36p prim T/rs %.4f sub-step %d/%d revert at iterate %d (node fields' change %s): %s; the hold reverts to that of round %d (the sets the fields had settled under)%s, is re-decided no more and is not held: the solve re-converges, the verification records it, the halving separates the crossing from the node" % (T_over_rs, j1, nsub, it, ch_, why, to_round, "" if prev_h is not None else " (none: the anchor's tracking, version 36l)"), flush=True)   # v36p prim


def dgp_iterate_v36p(a, chs, anchor, f0, x, it, masks_prev, s_nodes, tau_s, E_rec, T_over_rs, j1, nsub):   # v36p prim
    """version 36p (W2_v36p_fix_v1, _v2): the hold decided and re-decided inside the sub-step's solve, at the top of each iterate with the current node fields x and the previous iterate's tracked node masks masks_prev. The chord runs from f0 to the three-node rule's endpoint (dgp_chord_end_v36p: the cubic through f0 and the nodes at s = 1, the verification's chord at the fixed point). Iterate 1: the state is reset; the chord is followed as the seed when it has a length (at a step's first sub-step the node fields are f0 at every node, the chord has no length, the anchor seeds iterate 1 and the decision is made at iterate 2 from the first iterate's node fields); the seed is set whether or not it is resolved. Later iterates: after a replacement, a tracked node mask differing from the hold at DGP["flips"] iterates within DGP["settle_max"] of it (the set alternates with the fields: a crossing on a node), or a change still above 1000 times DGP["redecide"] DGP["settle_max"] iterates after it, reverts the hold to the one before (dgp_revert_v36p: a cycle, not held, the halving's matter). When the node fields' relative change since the previous iterate has fallen below DGP["redecide"], the check is armed and at least DGP["settle"] iterates have passed since the last decision, the chord is followed again; a hold whose masks at the nodes and the endpoint and whose crossings equal the new decision's is kept and the check disarmed (re-armed by a change above 100 times the threshold); a decision that is not resolved (unstable under halving, an unresolved or undecided crossing, a jump not falling) replaces nothing, the hold stays and the check is disarmed; a resolved one that differs replaces the hold (a round: the nodes are solved under the new sets from this iterate on, the fields move, convergence is deferred), at most DGP["rounds"] times per sub-step; a hold equal to an earlier replaced one is a cycle, flagged, the hold kept. The verification on the converged chord (dgp_verify_v36p) stays the arbiter of held."""   # v36p prim
    if anchor is None or PRIM is None:   # v36p prim
        return   # v36p prim
    f1 = dgp_chord_end_v36p(f0, x, s_nodes)   # v36p prim
    if it == 1:   # v36p prim
        DGP.update({"hold": None, "decision": None, "decision_eigh": 0, "round": 0, "hold_hist": [], "dec_hist": [], "x_prev": None, "armed": True, "redecided_at": [], "seed_degenerate": False, "cycle": False, "exhausted": False, "decided_at": None, "off_hold": [], "revert": None, "unresolved_at": [], "f1_last": None})   # v36p prim
        scale = max([float(np.max(np.abs(f_))) for f_ in f1 if f_.size] + [1e-300])   # v36p prim
        length = max([float(np.max(np.abs(f_ - np.asarray(g_, float)))) for f_, g_ in zip(f1, f0) if f_.size] + [0.0]) / scale   # v36p prim
        DGP["seed_degenerate"] = bool(length <= 1e-14)   # v36p prim
        if DGP["seed_degenerate"]:   # v36p prim
            print("  v36p prim T/rs %.4f sub-step %d/%d decision at iterate 1: the start's node fields are the start fields at every node and the chord to the three-node rule's endpoint has no length (%.1e of the fields): the anchor seeds iterate 1, the set is decided at iterate 2 from the first iterate's node fields" % (T_over_rs, j1, nsub, length), flush=True)   # v36p prim
        else:   # v36p prim
            dec, orc = dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, DGP["level_min"], DGP["level_max"])   # v36p prim
            hold = dgp_hold_from_v36p(dec, anchor, s_nodes)   # v36p prim
            DGP.update({"hold": hold, "decision": dec, "decision_eigh": orc.n_eigh, "hold_hist": [hold], "dec_hist": [dec], "decided_at": 1, "f1_last": f1}); DGP["n_eigh"] += orc.n_eigh   # v36p prim
            dgp_print_v36p(dec, "decision", T_over_rs, j1, nsub, E_rec, orc.n_eigh)   # v36p prim
            print("  v36p prim T/rs %.4f sub-step %d/%d decision at iterate 1: the hold set as the seed%s; checked again when the node fields have settled, %d iterates after this one at the earliest" % (T_over_rs, j1, nsub, "" if dgp_resolved_v36p(dec) else (" (not resolved: stable under halving %s, %d unresolved or undecided, jumps fall %s; a seed only)" % (dgp_yn_v36p(dec.stable), int(dec.n_unresolved), dgp_yn_v36p(dec.jumps_fall))), DGP["settle"]), flush=True)   # v36p prim
        DGP["x_prev"] = dgp_copy_x_v36p(x)   # v36p prim
        return   # v36p prim
    change = dgp_change_v36p(x, DGP.get("x_prev")); DGP["x_prev"] = dgp_copy_x_v36p(x)   # v36p prim
    ch_ = ("%.2e" % change) if change is not None else "-"   # v36p prim
    if DGP["hold"] is not None and DGP["round"] >= 1 and (not DGP["cycle"]) and DGP["decided_at"] is not None and it <= DGP["decided_at"] + DGP["settle_max"]:   # v36p prim: W2_v36p_fix_v2 (ii): after a replacement, the tracked sets against the hold and the settling
        off = dgp_off_hold_v36p(DGP["hold"], masks_prev)   # v36p prim
        if off:   # v36p prim
            DGP["off_hold"].append((int(it) - 1, off))   # v36p prim
            print("  v36p prim T/rs %.4f sub-step %d/%d iterate %d: the tracked set of iterate %d left the hold at node-channel(s) %s (%d of %d since the replacement at iterate %d; node fields' change %s)" % (T_over_rs, j1, nsub, it, it - 1, ",".join("%d:%s" % (k_, kap) for k_, kap in off), len(DGP["off_hold"]), DGP["flips"], DGP["decided_at"], ch_), flush=True)   # v36p prim
            if len(DGP["off_hold"]) >= DGP["flips"]:   # v36p prim
                dgp_revert_v36p(it, ch_, "the tracked set left the hold at %d iterate(s) after the replacement at iterate %d (node-channels %s): the set at the node alternates with the fields, a crossing on a node" % (len(DGP["off_hold"]), DGP["decided_at"], ",".join(sorted(set("%d:%s" % (k_, kap) for _i, o_ in DGP["off_hold"] for k_, kap in o_)))), T_over_rs, j1, nsub)   # v36p prim
                return   # v36p prim
        if it == DGP["decided_at"] + DGP["settle_max"] and change is not None and change > 1000.0 * DGP["redecide"]:   # v36p prim
            dgp_revert_v36p(it, ch_, "the solve has not settled %d iterates after the replacement at iterate %d (node fields' change %s against %.1e)" % (DGP["settle_max"], DGP["decided_at"], ch_, 1000.0 * DGP["redecide"]), T_over_rs, j1, nsub)   # v36p prim
            return   # v36p prim
    if change is not None and (not DGP["armed"]) and change > 100.0 * DGP["redecide"]:   # v36p prim: the fields moved again after a kept hold or an unresolved re-decision: the check re-armed
        DGP["armed"] = True   # v36p prim
    settled = bool(DGP["armed"] and change is not None and change <= DGP["redecide"] and (DGP["decided_at"] is None or it >= DGP["decided_at"] + DGP["settle"]))   # v36p prim: W2_v36p_fix_v2 (ii): at least DGP["settle"] iterates after the last decision, so the solve has answered it
    due = bool((it == 2 and DGP["seed_degenerate"]) or settled)   # v36p prim
    if (not due) or DGP["cycle"] or DGP["exhausted"]:   # v36p prim
        return   # v36p prim
    if DGP["hold"] is not None and DGP["round"] >= DGP["rounds"]:   # v36p prim
        DGP["exhausted"] = True   # v36p prim
        print("  v36p prim T/rs %.4f sub-step %d/%d re-decision at iterate %d (node fields' change %s): the %d replacements of the hold are used: the hold stays, the verification decides" % (T_over_rs, j1, nsub, it, ch_, DGP["rounds"]), flush=True)   # v36p prim
        return   # v36p prim
    tag = ("decision at iterate %d" % it) if DGP["hold"] is None else ("re-decision %d at iterate %d" % (DGP["round"] + 1, it))   # v36p prim
    dec, orc = dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, DGP["level_min"], DGP["level_max"])   # v36p prim
    hold = dgp_hold_from_v36p(dec, anchor, s_nodes); DGP["decision_eigh"] = int(DGP.get("decision_eigh", 0)) + orc.n_eigh; DGP["n_eigh"] += orc.n_eigh; DGP["f1_last"] = f1   # v36p prim
    if DGP["hold"] is not None and dgp_hold_equal_v36p(DGP["hold"], hold) and dgp_events_equal_v36p(DGP["decision"], dec):   # v36p prim
        DGP["armed"] = False   # v36p prim
        print("  v36p prim T/rs %.4f sub-step %d/%d %s (node fields' change %s): the hold is the chord's set (masks at the nodes and the endpoint, %d crossings): kept after %d replacement(s); eigh %d" % (T_over_rs, j1, nsub, tag, ch_, len(dec.crossings), DGP["round"], orc.n_eigh), flush=True)   # v36p prim
        return   # v36p prim
    if DGP["hold"] is not None and not dgp_resolved_v36p(dec):   # v36p prim: W2_v36p_fix_v2 (i): an unstable or undecided decision replaces nothing
        DGP["armed"] = False; DGP["unresolved_at"].append(int(it))   # v36p prim
        dgp_print_v36p(dec, tag, T_over_rs, j1, nsub, E_rec, orc.n_eigh)   # v36p prim
        print("  v36p prim T/rs %.4f sub-step %d/%d %s (node fields' change %s): the chord's set is not resolved (stable under halving %s, %d unresolved or undecided, jumps fall %s): it does not replace the hold, which stays; checked again when the fields move; the verification decides, the halving resolves" % (T_over_rs, j1, nsub, tag, ch_, dgp_yn_v36p(dec.stable), int(dec.n_unresolved), dgp_yn_v36p(dec.jumps_fall)), flush=True)   # v36p prim
        return   # v36p prim
    if any(dgp_hold_equal_v36p(h_, hold) for h_ in DGP["hold_hist"][:-1]):   # v36p prim
        DGP["cycle"] = True   # v36p prim
        dgp_print_v36p(dec, tag, T_over_rs, j1, nsub, E_rec, orc.n_eigh)   # v36p prim
        print("  v36p prim T/rs %.4f sub-step %d/%d %s (node fields' change %s): the chord's set returns to an earlier hold of this sub-step: a cycle (a crossing on a node; the sets alternate with the fields): the hold stays, not held, the halving separates them" % (T_over_rs, j1, nsub, tag, ch_), flush=True)   # v36p prim
        return   # v36p prim
    if DGP["hold"] is not None:   # v36p prim
        DGP["round"] += 1; DGP["redecided_at"].append(int(it))   # v36p prim
    DGP.update({"hold": hold, "decision": dec, "armed": True, "decided_at": int(it), "off_hold": []}); DGP["hold_hist"].append(hold); DGP["dec_hist"].append(dec)   # v36p prim
    dgp_print_v36p(dec, tag, T_over_rs, j1, nsub, E_rec, orc.n_eigh)   # v36p prim
    print("  v36p prim T/rs %.4f sub-step %d/%d %s (node fields' change %s): the hold %s (%d of at most %d replacements)%s; the nodes are solved under these sets from this iterate on; checked again when the fields have settled, %d iterates after this one at the earliest" % (T_over_rs, j1, nsub, tag, ch_, "set" if (DGP["round"] == 0 and len(DGP["hold_hist"]) == 1) else "replaced", DGP["round"], DGP["rounds"], "" if dgp_resolved_v36p(dec) else (" (the seed is not resolved: stable under halving %s, %d unresolved or undecided)" % (dgp_yn_v36p(dec.stable), int(dec.n_unresolved))), DGP["settle"]), flush=True)   # v36p prim


def dgp_events_equal_v36p(dec0, dec1):   # v36p prim
    """version 36p: the crossings of two decisions equal (channel, decision, direction, mirror flags, s* within a piece of the first level), one to one."""   # v36p prim
    key = lambda ev: (str(ev["channel"]), ev["decision"], int(np.sign(ev["n"] - ev["i"])), bool(ev["mirrored"]), bool(ev["self_mirror"]))   # v36p prim
    a_ = sorted(dec0.crossings, key=lambda ev: (str(ev["channel"]), ev["s_star"])); b_ = sorted(dec1.crossings, key=lambda ev: (str(ev["channel"]), ev["s_star"]))   # v36p prim
    if len(a_) != len(b_):   # v36p prim
        return False   # v36p prim
    tol = 2.0 ** (-int(DGP["level_min"])) + 1e-9   # v36p prim
    return all(key(x_) == key(y_) and abs(x_["s_star"] - y_["s_star"]) <= tol for x_, y_ in zip(a_, b_))   # v36p prim


def dgp_verify_v36p(a, chs, anchor, f0, f_in_e, s_nodes, masks, m_e, tau_s, E_rec, T_over_rs, j1, nsub, ent, conv):   # v36p prim
    """version 36p: the verification on the converged chord f0 -> the endpoint's input fields: the primitive's sets there must be the held ones (the endpoint's mask equal, the nodes' masks equal as held and as tracked) and its crossings the decision's; the sub-step's prim entry gains the verification, held, stable, jumps_fall and printability."""   # v36p prim
    dec0 = DGP.get("decision"); hold = DGP.get("hold"); pr = ent.get("prim")   # v36p prim
    if pr is None or PRIM is None:   # v36p prim
        return   # v36p prim
    l0 = (max([v_ for v_ in dec0.level_stable.values() if v_ is not None] or [DGP["level_max"]]) if dec0 is not None else DGP["level_min"] + 1)   # v36p prim: no decision (a seed without length and a solve ended before iterate 2): the verification from the first level, the set not held
    dec1, orc = dgp_run_v36p(a, chs, anchor, f0, f_in_e, s_nodes, tau_s, max(DGP["level_min"], l0 - 1), DGP["level_max"])   # v36p prim
    qs = list(s_nodes) + [1.0]; end_ok = True; node_ok = []; node_tracked_ok = []   # v36p prim
    for kap in anchor:   # v36p prim
        me = m_e.get(kap) if isinstance(m_e, dict) else None   # v36p prim
        if me is None or not np.array_equal(np.asarray(me, bool), dec1.masks_at[1.0][kap]):   # v36p prim
            end_ok = False   # v36p prim
    for k_ in range(3):   # v36p prim
        node_ok.append(bool(hold is not None and all(np.array_equal(dec1.masks_at[qs[k_]][kap], np.asarray(hold[k_][kap][2], bool)) for kap in anchor)))   # v36p prim
        node_tracked_ok.append(bool(hold is not None and all((kap in masks[k_]) and np.array_equal(np.asarray(masks[k_][kap], bool), np.asarray(hold[k_][kap][2], bool)) for kap in anchor)))   # v36p prim
    ev_ok = bool(dec0 is not None and dgp_events_equal_v36p(dec0, dec1))   # v36p prim
    held = bool(end_ok and all(node_ok) and all(node_tracked_ok) and ev_ok)   # v36p prim
    pr.update({"verified": True, "verify": dec1.summary(samples=False), "held": held, "end_held": bool(end_ok), "nodes_held": node_ok, "nodes_tracked_held": node_tracked_ok, "crossings_held": bool(ev_ok),   # v36p prim
               "rounds": int(DGP.get("round", 0)), "redecided_at": [int(v_) for v_ in DGP.get("redecided_at", [])], "seed_degenerate": bool(DGP.get("seed_degenerate")), "cycle": bool(DGP.get("cycle")), "rounds_exhausted": bool(DGP.get("exhausted")), "tau_lz_min": dgp_tau_min_v36p(dec1),   # v36p prim: W2_v36p_fix_v1: the hold's rounds inside the solve; the smallest tau_LZ over avoided crossings
               "revert": DGP.get("revert"), "reverted_at": (int(DGP["revert"]["at"]) if DGP.get("revert") else None), "unresolved_at": [int(v_) for v_ in DGP.get("unresolved_at", [])], "end_gap": dgp_gap_v36p(DGP.get("f1_last"), f_in_e),   # v36p prim: W2_v36p_fix_v2: the revert (a crossing on a node), the unresolved re-decisions that replaced nothing, the last decision's chord end against the endpoint's input fields
               "stable": bool(dec1.stable), "jumps_fall": bool(dec1.jumps_fall), "level_stable": dec1.level_stable_max, "n_crossings": len(dec1.crossings), "n_diabatic": int(dec1.n_diabatic), "n_adiabatic": int(dec1.n_adiabatic),   # v36p prim
               "n_unresolved": int(dec1.n_unresolved), "jump_fit_sum_over_E_rec": (dec1.jump_fit_sum / E_rec) if E_rec else None, "jump_grid_sum_over_E_rec": (dec1.jump_grid_sum / E_rec) if E_rec else None,   # v36p prim
               "eigh": int(DGP.get("decision_eigh", 0)) + int(orc.n_eigh), "converged": bool(conv), "printable_set": bool(dec1.stable and dec1.jumps_fall and held), "printable": bool(conv and dec1.stable and dec1.jumps_fall and held)})   # v36p prim
    DGP["n_eigh"] += orc.n_eigh   # v36p prim
    dgp_print_v36p(dec1, "verification", T_over_rs, j1, nsub, E_rec, orc.n_eigh)   # v36p prim
    print("  v36p prim T/rs %.4f sub-step %d/%d verification: the set held through the solve %s (endpoint %s, nodes by the chord %s, nodes as tracked %s, crossings %s; hold replaced %d time(s) at iterates %s%s%s%s%s; the last decision's chord end against the endpoint's input fields %s); stable under halving %s; jumps fall %s; tau_LZ min over avoided crossings %s; residual at the stop rule %s; printable %s" % (   # v36p prim
        T_over_rs, j1, nsub, dgp_yn_v36p(held), dgp_yn_v36p(end_ok), "/".join(dgp_yn_v36p(v_) for v_ in node_ok), "/".join(dgp_yn_v36p(v_) for v_ in node_tracked_ok), dgp_yn_v36p(ev_ok), int(DGP.get("round", 0)), ",".join(str(v_) for v_ in DGP.get("redecided_at", [])) or "-", ", a cycle" if DGP.get("cycle") else "", ", the rounds used" if DGP.get("exhausted") else "",   # v36p prim
        (", reverted at iterate %d to round %d (%s)" % (pr["revert"]["at"], pr["revert"]["to_round"], pr["revert"]["why"])) if pr.get("revert") else "", (", unresolved re-decisions at iterates %s replaced nothing" % ",".join(str(v_) for v_ in pr["unresolved_at"])) if pr.get("unresolved_at") else "", dgp_f_v36p(pr["end_gap"], "%.1e"),   # v36p prim: W2_v36p_fix_v2
        dgp_yn_v36p(dec1.stable), dgp_yn_v36p(dec1.jumps_fall), dgp_f_v36p(pr["tau_lz_min"], "%.3g"), dgp_yn_v36p(conv), dgp_yn_v36p(pr["printable"])), flush=True)   # v36p prim


def dgp_entry_v36p():   # v36p prim
    """version 36p: the sub-step entry's prim dict from the decision (the last hold's; the verification fills the rest)."""   # v36p prim
    dec = DGP.get("decision")   # v36p prim
    base = {"rounds": int(DGP.get("round", 0)), "redecided_at": [int(v_) for v_ in DGP.get("redecided_at", [])], "seed_degenerate": bool(DGP.get("seed_degenerate")), "cycle": bool(DGP.get("cycle")), "rounds_exhausted": bool(DGP.get("exhausted")), "tau_lz_min": None,   # v36p prim: W2_v36p_fix_v1
            "revert": DGP.get("revert"), "reverted_at": (int(DGP["revert"]["at"]) if DGP.get("revert") else None), "unresolved_at": [int(v_) for v_ in DGP.get("unresolved_at", [])], "end_gap": None}   # v36p prim: W2_v36p_fix_v2
    if dec is None:   # v36p prim
        return dict(base, decision=None, verified=False, held=None, stable=None, jumps_fall=None, printable_set=None, printable=None, level_stable=None, n_crossings=0, n_diabatic=0, n_adiabatic=0, n_unresolved=0, jump_fit_sum_over_E_rec=None, jump_grid_sum_over_E_rec=None, eigh=int(DGP.get("decision_eigh", 0)))   # v36p prim
    return dict(base, decision=dec.summary(samples=False), verified=False, held=None, stable=bool(dec.stable), jumps_fall=bool(dec.jumps_fall), printable_set=None, printable=None, level_stable=dec.level_stable_max,   # v36p prim
                n_crossings=len(dec.crossings), n_diabatic=int(dec.n_diabatic), n_adiabatic=int(dec.n_adiabatic), n_unresolved=int(dec.n_unresolved), jump_fit_sum_over_E_rec=None, jump_grid_sum_over_E_rec=None, eigh=int(DGP.get("decision_eigh", 0)))   # v36p prim


def dgp_step_summary_v36p(subs, conv, E_rec):   # v36p prim
    """version 36p: the step's prim summary (the sub-steps' conjunction): stable, held, jumps_fall, printable, the counts and the resolved jumps' sum over E_rec."""   # v36p prim
    prs = [q_.get("prim") for q_ in subs if q_.get("prim") is not None]   # v36p prim
    if not prs:   # v36p prim
        return None   # v36p prim
    ver = [p_ for p_ in prs if p_.get("verified")]; allv = bool(ver) and len(ver) == len(prs)   # v36p prim
    def all_(key):   # v36p prim
        return (all(bool(p_.get(key)) for p_ in ver) if allv else None)   # v36p prim
    return {"n_sub": len(prs), "n_verified": len(ver), "stable": all_("stable"), "held": all_("held"), "jumps_fall": all_("jumps_fall"), "converged": bool(conv), "printable_set": all_("printable_set"), "printable": bool(conv and allv and all(bool(p_.get("printable_set")) for p_ in ver)),   # v36p prim
            "level_stable": max([int(p_.get("level_stable") or 0) for p_ in prs] or [0]), "n_crossings": sum(int(p_.get("n_crossings") or 0) for p_ in prs), "n_diabatic": sum(int(p_.get("n_diabatic") or 0) for p_ in prs), "n_adiabatic": sum(int(p_.get("n_adiabatic") or 0) for p_ in prs),   # v36p prim
            "n_unresolved": sum(int(p_.get("n_unresolved") or 0) for p_ in prs), "jump_fit_sum_over_E_rec": sum(float(p_.get("jump_fit_sum_over_E_rec") or 0.0) for p_ in prs), "jump_grid_sum_over_E_rec": sum(float(p_.get("jump_grid_sum_over_E_rec") or 0.0) for p_ in prs), "eigh": sum(int(p_.get("eigh") or 0) for p_ in prs),   # v36p prim
            "rounds": sum(int(p_.get("rounds") or 0) for p_ in prs), "n_cycle": sum(1 for p_ in prs if p_.get("cycle")), "n_rounds_exhausted": sum(1 for p_ in prs if p_.get("rounds_exhausted")),   # v36p prim: W2_v36p_fix_v1
            "n_reverted": sum(1 for p_ in prs if p_.get("reverted_at") is not None), "n_unresolved_redecisions": sum(len(p_.get("unresolved_at") or []) for p_ in prs), "end_gap_max": (max([float(p_["end_gap"]) for p_ in prs if p_.get("end_gap") is not None]) if any(p_.get("end_gap") is not None for p_ in prs) else None),   # v36p prim: W2_v36p_fix_v2
            "tau_lz_min": (min([float(p_["tau_lz_min"]) for p_ in prs if p_.get("tau_lz_min") is not None]) if any(p_.get("tau_lz_min") is not None for p_ in prs) else None)}   # v36p prim


def dgp_unstable_v36p(ent):   # v36p prim
    """version 36p: a verified sub-step whose set is not stable under halving, not held or whose jumps do not fall (the halving trigger)."""   # v36p prim
    p_ = ent.get("prim")   # v36p prim
    return bool(p_ and p_.get("verified") and not p_.get("printable_set"))   # v36p prim


def dgp_line_v36p(p, kind, T_over_rs, j1, nsub):   # v36p prim
    """version 36p: one line with the printability of a sub-step (p = its prim entry) or of a step or record (p = the step's prim summary)."""   # v36p prim
    if p is None:   # v36p prim
        return "  v36p prim T/rs %.4f %s: no decision (no anchor)" % (T_over_rs, kind)   # v36p prim
    where = ("sub-step %d/%d" % (j1, nsub)) if j1 is not None else kind   # v36p prim
    return ("  v36p prim T/rs %.4f %s: set stable under halving %s (level %s), held through the solve %s (hold replaced %d time(s)%s%s), jumps fall %s; crossings %d (%d diabatic, %d adiabatic, %d unresolved); resolved jumps %s of E (at the grid %s); tau_LZ min over avoided crossings %s; the decisions' chord end against the endpoint's input fields %s; residual at the stop rule %s; printable %s" % (   # v36p prim
        T_over_rs, where, dgp_yn_v36p(p.get("stable")), p.get("level_stable", "-"), dgp_yn_v36p(p.get("held")), int(p.get("rounds") or 0), (", cycles %d" % int(p.get("n_cycle") or 0)) if p.get("n_cycle") else (", a cycle" if p.get("cycle") else ""), (", reverted %d" % int(p.get("n_reverted") or 0)) if p.get("n_reverted") else ((", reverted at iterate %d" % int(p["reverted_at"])) if p.get("reverted_at") is not None else ""), dgp_yn_v36p(p.get("jumps_fall")), int(p.get("n_crossings") or 0), int(p.get("n_diabatic") or 0), int(p.get("n_adiabatic") or 0), int(p.get("n_unresolved") or 0),   # v36p prim
        dgp_f_v36p(p.get("jump_fit_sum_over_E_rec"), "%+.3e"), dgp_f_v36p(p.get("jump_grid_sum_over_E_rec"), "%+.3e"), dgp_f_v36p(p.get("tau_lz_min"), "%.3g"), dgp_f_v36p(p.get("end_gap_max", p.get("end_gap")), "%.1e"), dgp_yn_v36p(p.get("converged")), dgp_yn_v36p(p.get("printable"))))   # v36p prim


def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart'''),
("substep decide hook",
r'''        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []; specs = []; bases = []   # v36k chart; v36l attr''',
r'''        if DGP["on"]:   # v36p prim: the hold decided at the first iterate (the seed) and re-decided on the chord to the three-node rule's endpoint when the node fields have settled, until it is the chord's set (W2_v36p_fix_v1, _v2); masks still holds the previous iterate's tracked node masks here (the line below resets it): a set that left the hold is read off them
            dgp_iterate_v36p(a, chs, anchor, f0, x, it, (masks if it > 1 else None), s_nodes, tau_s, E_rec, T0 / rs, j + 1, nsub)   # v36p prim
        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []; specs = []; bases = []   # v36k chart; v36l attr'''),
("substep node call",
r'''            Nk, vk, Ek, info_k, ti_k, mk, _r, sp_k = dgk_node_v36k(a, chs, st, R, base_s, x[k][0], x[k][1], anchor)   # v36k chart; v36l attr''',
r'''            Nk, vk, Ek, info_k, ti_k, mk, _r, sp_k = dgk_node_v36k(a, chs, st, R, base_s, x[k][0], x[k][1], dgp_point_anchor_v36p(anchor, k))   # v36k chart; v36l attr; v36p prim: the held set at the node under --dg-assign prim'''),
("substep endpoint call",
r'''            Ne, ve, Ee, info_e, ti_e, m_e, refs_e, sp_e = dgk_node_v36k(a, chs, st, R, base1, N1, v1, anchor, keep_ref=last, check=last)   # v36k chart; v36l attr''',
r'''            Ne, ve, Ee, info_e, ti_e, m_e, refs_e, sp_e = dgk_node_v36k(a, chs, st, R, base1, N1, v1, dgp_point_anchor_v36p(anchor, 3), keep_ref=last, check=last)   # v36k chart; v36l attr; v36p prim: the held set at the endpoint under --dg-assign prim'''),
("substep entry keys",
r'''           "vs_anchor_hist": vs_anchor_hist, "attribution": None, "switch_sum_over_E_rec": None, "n_switch": None, "rest_over_E_rec": None, "attribution_solves": 0}   # v36l attr: the chord's switches and the two bins of r_chain''',
r'''           "vs_anchor_hist": vs_anchor_hist, "attribution": None, "switch_sum_over_E_rec": None, "n_switch": None, "rest_over_E_rec": None, "attribution_solves": 0,   # v36l attr: the chord's switches and the two bins of r_chain
           **({"prim": dgp_entry_v36p()} if DGP["on"] else {})}   # v36p prim: the primitive's decision, verification and printability of this sub-step (no key under the defaults)'''),
("substep verification",
r'''                ent.update({"attribution": recs_, "switch_sum_over_E_rec": jsum_, "n_switch": len(recs_), "rest_over_E_rec": r_rel - jsum_, "attribution_solves": ns_})   # v36l attr''',
r'''                ent.update({"attribution": recs_, "switch_sum_over_E_rec": jsum_, "n_switch": len(recs_), "rest_over_E_rec": r_rel - jsum_, "attribution_solves": ns_})   # v36l attr
            if DGP["on"] and anchor is not None:   # v36p prim: the verification on the converged chord f0 -> the endpoint's input fields; the held sets must be the primitive's there
                dgp_verify_v36p(a, chs, anchor, f0, f_in_e, s_nodes, masks, m_e, tau_s, E_rec, T0 / rs, j + 1, nsub, ent, conv)   # v36p prim'''),
("substep print",
r'''    print("T/rs %.4f v36k chart sub-step %d/%d (tau %.4g) to T/rs %.4f: %s in %d iterations; energy defect %s of E (change between iterates %s), node fields %s; gate 1 r_chain %s of E (%s; in-chord switching %s over %s differences, remainder %s); the convention would reassign %s levels at the end; tracked-set changes %d, clipped updates %d; eigh %d, solves %d, %.1f s; start: %s%s" % (   # v36k chart; v36l attr''',
r'''    if DGP["on"]:   # v36p prim
        print(dgp_line_v36p(ent.get("prim"), "sub-step", T0 / rs, j + 1, nsub), flush=True)   # v36p prim
    print("T/rs %.4f v36k chart sub-step %d/%d (tau %.4g) to T/rs %.4f: %s in %d iterations; energy defect %s of E (change between iterates %s), node fields %s; gate 1 r_chain %s of E (%s; in-chord switching %s over %s differences, remainder %s); the convention would reassign %s levels at the end; tracked-set changes %d, clipped updates %d; eigh %d, solves %d, %.1f s; start: %s%s" % (   # v36k chart; v36l attr'''),
("step trigger",
r'''            if DGK["chain_trigger"] and ent["gate1"] == "exceeded" and level < DGK["sub_max"]:   # v36k chart; v36l attr: --dg-chain-trigger off makes gate 1 a measurement''',
r'''            if ((DGK["chain_trigger"] and ent["gate1"] == "exceeded") or (DGP["on"] and DGP["halve"] and dgp_unstable_v36p(ent))) and level < DGK["sub_max"]:   # v36k chart; v36l attr: --dg-chain-trigger off makes gate 1 a measurement; v36p prim: a sub-step whose set is not stable under halving, not held or whose jumps do not fall halves as gate 1 did'''),
("step entry",
r'''           "switch_sum_over_E_rec": ssum, "n_switch": nsw, "rest_over_E_rec": ((rsum - ssum) if (rsum is not None and ssum is not None) else None),   # v36l attr: the switching bin of r_chain and its remainder''',
r'''           "switch_sum_over_E_rec": ssum, "n_switch": nsw, "rest_over_E_rec": ((rsum - ssum) if (rsum is not None and ssum is not None) else None),   # v36l attr: the switching bin of r_chain and its remainder
           **({"prim": dgp_step_summary_v36p(subs, conv, E_rec)} if DGP["on"] else {}),   # v36p prim: the step's printability (the sub-steps' sets stable under halving and held, the jumps falling, the residual at the stop rule); no key under the defaults'''),
("step print",
r'''    print("T/rs %.4f v36k chart step to T/rs %.4f: %s; level %d (%d sub-steps of tau %.4g; %d attempts; next level %d); %d iterations; energy defect %s of E; E_start/E_rec - 1 = %s (%d passes); gate 1 r_chain %s of E (in-chord switching %s over %s differences, remainder %s); E_end/E_rec - 1 = %s; E_pred/E_rec - 1 = %s; the convention would reassign %s levels; anchor: %s; eigh %d, solves %d, %.1f s%s" % (   # v36k chart; v36l attr (review B1 and the switching bin)''',
r'''    if DGP["on"]:   # v36p prim
        print(dgp_line_v36p(ent.get("prim"), "step", T / rs, None, None), flush=True)   # v36p prim
    print("T/rs %.4f v36k chart step to T/rs %.4f: %s; level %d (%d sub-steps of tau %.4g; %d attempts; next level %d); %d iterations; energy defect %s of E; E_start/E_rec - 1 = %s (%d passes); gate 1 r_chain %s of E (in-chord switching %s over %s differences, remainder %s); E_end/E_rec - 1 = %s; E_pred/E_rec - 1 = %s; the convention would reassign %s levels; anchor: %s; eigh %d, solves %d, %.1f s%s" % (   # v36k chart; v36l attr (review B1 and the switching bin)'''),
("record line",
r'''                print("T/rs %.4f v36k chart: reassignment move (E - E_end)/E0 = %s; the step: level %d (%d sub-steps of tau %.4g, %d attempts), start E_start/E_rec - 1 %s, gate 1 r_chain %s of E (in-chord switching %s over %s differences, remainder %s), gate 2 defect %s of E, tracked-set changes %s, the convention would reassign %s levels at the step's end, anchor %s; the record solve converged: %s" % (   # v36k chart; v36l attr (review B1 and the switching bin)''',
r'''                if d_.get("prim") is not None:   # v36p prim: the record's printability (the comparator's convergence prints on the chart line below)
                    print(dgp_line_v36p(d_.get("prim"), "record", T / rs, None, None), flush=True)   # v36p prim
                print("T/rs %.4f v36k chart: reassignment move (E - E_end)/E0 = %s; the step: level %d (%d sub-steps of tau %.4g, %d attempts), start E_start/E_rec - 1 %s, gate 1 r_chain %s of E (in-chord switching %s over %s differences, remainder %s), gate 2 defect %s of E, tracked-set changes %s, the convention would reassign %s levels at the step's end, anchor %s; the record solve converged: %s" % (   # v36k chart; v36l attr (review B1 and the switching bin)'''),
("run label",
r'''        label += "; v36l: " + dgk_desc_v36l(a)   # v36l attr''',
r'''        label += "; v36l: " + dgk_desc_v36l(a)   # v36l attr
        if DGP["on"]:   # v36p prim
            label += "; v36p: " + dgp_desc_v36p(a)   # v36p prim'''),
("arguments",
r'''    ap.add_argument("--dg-attr", dest="dg_attr", default="off", choices=["off", "on"], help="(v36l) on: at each difference of the tracked set between consecutive chord points the chart's charge is also evaluated at both points with the other point's mask (two eigendecomposition sets and two solves per difference) as the measured jump beside the first-order -w gap_min; needs --step dg --dg-chart frozen")   # v36l attr''',
r'''    ap.add_argument("--dg-attr", dest="dg_attr", default="off", choices=["off", "on"], help="(v36l) on: at each difference of the tracked set between consecutive chord points the chart's charge is also evaluated at both points with the other point's mask (two eigendecomposition sets and two solves per difference) as the measured jump beside the first-order -w gap_min; needs --step dg --dg-chart frozen")   # v36l attr
    ap.add_argument("--dg-assign", dest="dg_assign", default="anchor", choices=["anchor", "prim"], help="(v36p) anchor: the chord's set by maximum overlap against the anchor at every chord point and iterate (version 36l); prim: the assignment primitive assign_primitive_v1 (the set followed along the chord with the declared rule at each near-crossing, the chord halved until the set is stable, decided once per sub-step solve and held); needs --step dg --dg-chart frozen")   # v36p prim
    ap.add_argument("--dg-prim-rule", dest="dg_prim_rule", default="lz", choices=["lz", "overlap", "energy"], help="(v36p) the declared rule at a near-crossing: lz (tau_LZ = k gap_min^2/|alpha_T| against c: the record convention's LZ rule), overlap (every crossing diabatic), energy (every crossing adiabatic)")   # v36p prim
    ap.add_argument("--dg-prim-c-lz", dest="dg_prim_c_lz", type=float, default=1.0, help="(v36p) the LZ threshold c (the record convention's c_LZ)")   # v36p prim
    ap.add_argument("--dg-prim-lz-k", dest="dg_prim_lz_k", type=float, default=1.0, help="(v36p) the factor k in tau_LZ = k gap_min^2/(hbar |alpha_T|) (1: the crossing's traversal time in precession times; pi/2: the LZ exponent)")   # v36p prim
    ap.add_argument("--dg-prim-tie", dest="dg_prim_tie", default="adiabatic", choices=["adiabatic", "diabatic"], help="(v36p) the decision when the band on tau_LZ still contains c at the last level (flagged; the sub-step is then not printable)")   # v36p prim
    ap.add_argument("--dg-prim-level-min", dest="dg_prim_level_min", type=int, default=1, help="(v36p) the first halving level of the chord (2^level pieces)")   # v36p prim
    ap.add_argument("--dg-prim-level-max", dest="dg_prim_level_max", type=int, default=4, help="(v36p) the last halving level (at least level-min + 1)")   # v36p prim
    ap.add_argument("--dg-prim-wmix", dest="dg_prim_wmix", type=float, default=0.9, help="(v36p) a chord point is clean when every set member's character has this weight and no other level more than 1 - it")   # v36p prim
    ap.add_argument("--dg-prim-eig-tol", dest="dg_prim_eig_tol", type=float, default=1e-10, help="(v36p) the eigenvalue tolerance that bands gap_min^2")   # v36p prim
    ap.add_argument("--dg-prim-jump-tol", dest="dg_prim_jump_tol", type=float, default=1e-9, help="(v36p) the relative tolerance of the jumps-fall test")   # v36p prim
    ap.add_argument("--dg-prim-knife", dest="dg_prim_knife", type=float, default=0.1, help="(v36p) a decided crossing with |ln(tau_LZ/c)| below this is flagged knife-edge")   # v36p prim
    ap.add_argument("--dg-prim-mirror", dest="dg_prim_mirror", default="auto", choices=["auto", "on", "off"], help="(v36p) the e <-> -e mirror of the spectrum (index i <-> N - 1 - i): a crossing and its mirror counted once; auto reads it off the spectrum at the chord's first point")   # v36p prim
    ap.add_argument("--dg-prim-halve", dest="dg_prim_halve", default="on", choices=["on", "off"], help="(v36p) on: a sub-step whose set is not stable under halving, not held or whose jumps do not fall halves the sub-step (the retry path gate 1 used in version 36k); off: printed only")   # v36p prim
    ap.add_argument("--dg-prim-rounds", dest="dg_prim_rounds", type=int, default=3, help="(v36p) W2_v36p_fix_v1: the most replacements of the hold inside one sub-step's solve (the chord re-decided when the node fields have settled; a hold still differing after them is not held)")   # v36p prim
    ap.add_argument("--dg-prim-redecide", dest="dg_prim_redecide", type=float, default=1e-6, help="(v36p) W2_v36p_fix_v1: the node fields' relative change between iterates (max |x - x_prev| over max |x|) below which the chord to the current nodes' extrapolation is followed again and the hold compared with it")   # v36p prim
    ap.add_argument("--dg-prim-settle", dest="dg_prim_settle", type=int, default=3, help="(v36p) W2_v36p_fix_v2: the fewest iterates between a decision of the hold and its next check (the solve answers a replacement over the mixer's history; one iterate later the fields have not moved)")   # v36p prim
    ap.add_argument("--dg-prim-settle-max", dest="dg_prim_settle_max", type=int, default=10, help="(v36p) W2_v36p_fix_v2: a solve whose node fields' change is still above 1000 times --dg-prim-redecide this many iterates after a replacement of the hold reverts to the hold before it (not held; the halving)")   # v36p prim
    ap.add_argument("--dg-prim-flips", dest="dg_prim_flips", type=int, default=2, help="(v36p) W2_v36p_fix_v2: a tracked node set differing from the hold at this many iterates after a replacement (within --dg-prim-settle-max of it) reverts the hold: a crossing on a node, the set alternating with the fields (not held; the halving)")   # v36p prim'''),
("validation",
r'''    DGK["chain_trigger"] = (a.dg_chain_trigger == "on"); DGK["attr"] = (a.dg_attr == "on")   # v36l attr''',
r'''    DGK["chain_trigger"] = (a.dg_chain_trigger == "on"); DGK["attr"] = (a.dg_attr == "on")   # v36l attr
    DGP.update({"on": (a.dg_assign == "prim"), "rule": a.dg_prim_rule, "c_lz": float(a.dg_prim_c_lz), "lz_k": float(a.dg_prim_lz_k), "tie": a.dg_prim_tie, "level_min": int(a.dg_prim_level_min), "level_max": int(a.dg_prim_level_max), "w_mix": float(a.dg_prim_wmix),   # v36p prim
                "eig_tol": float(a.dg_prim_eig_tol), "jump_tol": float(a.dg_prim_jump_tol), "knife": float(a.dg_prim_knife), "mirror": {"auto": "auto", "on": True, "off": None}[a.dg_prim_mirror], "halve": (a.dg_prim_halve == "on"), "rounds": int(a.dg_prim_rounds), "redecide": float(a.dg_prim_redecide),   # v36p prim
                "settle": int(a.dg_prim_settle), "settle_max": int(a.dg_prim_settle_max), "flips": int(a.dg_prim_flips)})   # v36p prim: W2_v36p_fix_v2
    if DGP["on"]:   # v36p prim
        if not DGK["on"]:   # v36p prim
            ap.error("version 36p: --dg-assign prim needs --step dg --dg-chart frozen")   # v36p prim
        if PRIM is None:   # v36p prim
            ap.error("version 36p: --dg-assign prim needs assign_primitive_v1.py beside the engine (the import failed)")   # v36p prim
        if DGP["level_max"] < DGP["level_min"] + 1:   # v36p prim
            ap.error("version 36p: --dg-prim-level-max must be at least --dg-prim-level-min + 1 (stability under halving needs two levels)")   # v36p prim
        if not (0.5 < DGP["w_mix"] < 1.0):   # v36p prim
            ap.error("version 36p: --dg-prim-wmix must lie in (0.5, 1)")   # v36p prim
        if DGP["rounds"] < 1 or not (DGP["redecide"] > 0.0):   # v36p prim
            ap.error("version 36p: --dg-prim-rounds must be at least 1 and --dg-prim-redecide positive")   # v36p prim
        if DGP["settle"] < 1 or DGP["settle_max"] < DGP["settle"] or DGP["flips"] < 1:   # v36p prim
            ap.error("version 36p: --dg-prim-settle must be at least 1, --dg-prim-settle-max at least --dg-prim-settle and --dg-prim-flips at least 1")   # v36p prim
        DGP["rule_obj"] = PRIM.Rule(kind=DGP["rule"], c=DGP["c_lz"], k=DGP["lz_k"], hbar=1.0, tie=DGP["tie"], knife=DGP["knife"])   # v36p prim'''),
("v36p prim: the primitive's import, assign_primitive_v1 -> assign_primitive_v1b (PRIM_fix_v1b section 5)",
r'''import assign_primitive_v1 as PRIM''',
r'''import assign_primitive_v1b as PRIM'''),
("v36p prim: the primitive's import, assign_primitive_v1b -> assign_primitive_v1c (reviewer A's fix of e/e2)",
r'''import assign_primitive_v1b as PRIM''',
r'''import assign_primitive_v1c as PRIM'''),
("v36p prim: the primitive's import, assign_primitive_v1c -> assign_primitive_v1d (W2_v36p_fix_v1: the third sample of a crossing inside the chord's last piece)",
r'''import assign_primitive_v1c as PRIM''',
r'''import assign_primitive_v1d as PRIM'''),
("v36p prim: W2_v36p_fix_v3, the engine's docstring (the chord's end is the endpoint's input as the endpoint call forms it, no cubic)",
r'''(W2_v36p_fix_v1, _v2): the chord's end inside the solve is the three-node rule's own endpoint, the cubic through f0 and the three nodes at s = 1 (the endpoint's input fields at the [v36p prim]
fixed point); whenever the node fields' relative change between iterates has fallen below --dg-prim-redecide, at least --dg-prim-settle iterates after the last decision, the chord is [v36p prim]''',
r'''(W2_v36p_fix_v1, _v2, _v3): the chord's end inside the solve is the endpoint's input fields as the endpoint evaluation forms them from the node fields, their quadratic Lagrange [v36p prim]
extrapolation to s = 1 passed through the first --dg-end-passes - 1 passes of the endpoint map (dgp_end_input_v36p; the node fields are quasi-static solves at the chord's densities, [v36p prim]
not a quadrature of a field equation, so the engine's endpoint is no cubic u(1) through f0 and the nodes: W2_v36p_fix_v3); whenever the node fields' relative change between iterates [v36p prim]
has fallen below --dg-prim-redecide, at least --dg-prim-settle iterates after the last decision, the chord is [v36p prim]'''),
("v36p prim: W2_v36p_fix_v3, the run label",
r'''followed from the sub-step's start to the three-node rule's endpoint (the cubic through the start fields and the three nodes at s = 1) with the rule [%s]''',
r'''followed from the sub-step's start to the endpoint's input fields as the endpoint call forms them (the node fields' quadratic extrapolation to s = 1 through the endpoint map's passes before its last; W2_v36p_fix_v3) with the rule [%s]'''),
("v36p prim: W2_v36p_fix_v3, the chord's end taken off the endpoint call (dgp_chord_end_v36p, the cubic, replaced by dgp_end_input_v36p)",
r'''def dgp_chord_end_v36p(f0, x, s_nodes):   # v36p prim
    """version 36p (W2_v36p_fix_v2): the chord's end from the node fields x: the three-node rule's own endpoint, the cubic Lagrange polynomial through (0, f0) and the three nodes at s = 1 (-f0 + (5 x1 - 4 x2 + 5 x3)/3 at the Gauss nodes), which is f0 + tau sum_j b_j F_j, the endpoint's input fields, at the fixed point (the cubic's derivative interpolates the chart's charge at the nodes); the quadratic through the nodes alone (W2_v36p_fix_v1) misses f0 and is short of the endpoint by q(0) - f0 (16 to 19 % of the chord at tau 8, 2 % at tau 4 on bench1), its crossings sitting at 0.84 and 0.98 of the verification's s*."""   # v36p prim
    L = dgp_lagrange_v36p([0.0] + list(s_nodes), 1.0)   # v36p prim
    pts = [(np.asarray(f0[0], float), np.asarray(f0[1], float))] + [(np.asarray(x[k_][0], float), np.asarray(x[k_][1], float)) for k_ in range(3)]   # v36p prim
    return (sum(L[i_] * pts[i_][0] for i_ in range(4)), sum(L[i_] * pts[i_][1] for i_ in range(4)))   # v36p prim''',
r'''def dgp_end_input_v36p(a, chs, st, R, base1, anchor, x, passes=True, T_over_rs=None, j1=None, nsub=None):   # v36p prim
    """version 36p (W2_v36p_fix_v3): the chord's end as the endpoint call receives it, formed from the node fields x exactly as dgk_substep_v36k forms the endpoint's input fields at convergence: the quadratic Lagrange extrapolation of the three nodes to s = 1 (dg_lagrange_v36f(1.0), version 36f's rule: N1 = L1 . x) and then, with passes, the first DGK["end_passes"] - 1 passes of the endpoint map (dgk_node_v36k at the iterate's propagated densities base1 with the held endpoint set: the endpoint call's passes before its last), whose last pass's input is the verification's f_in_e, where the endpoint's masks and the next anchor are taken. The engine's endpoint is no polynomial in the node fields: each node's fields are the quasi-static solve at the sources of C_0 + s (C_1 - C_0) with the reference tracked to them (dgk_node_v36k), not a quadrature of a field equation, so there is no u(1) = f0 + tau sum_j b_j F_j; the cubic through f0 and the nodes (W2_v36p_fix_v2) and the quadratic through the nodes (W2_v36p_fix_v1) both differ from f_in_e by the endpoint map's first pass (read on the quadratic: 16 to 19 % of the chord at tau 8 and 2 % at tau 4; on the cubic: 3.7e-4 to 7.5e-3 of the fields at tau 2; bench1). A pass whose field solve fails ends the passes at its input (printed). Called from dgp_iterate_v36p only (not under the defaults)."""   # v36p prim
    L1 = dg_lagrange_v36f(1.0)   # v36p prim
    N1 = np.asarray(L1[0] * x[0][0] + L1[1] * x[1][0] + L1[2] * x[2][0], float); v1 = np.asarray(L1[0] * x[0][1] + L1[1] * x[1][1] + L1[2] * x[2][1], float)   # v36p prim
    if passes:   # v36p prim
        for p_ in range(int(DGK["end_passes"]) - 1):   # v36p prim
            Ne, ve, _Ee, info_e, _ti, _m, _r, _sp = dgk_node_v36k(a, chs, st, R, base1, N1, v1, dgp_point_anchor_v36p(anchor, 3))   # v36p prim
            if Ne is None:   # v36p prim
                print("  v36p prim T/rs %s sub-step %s/%s: the endpoint map's pass %d of %d failed in the decision hook (%s): the chord's end is that pass's input" % (dgp_f_v36p(T_over_rs, "%.4f"), j1, nsub, p_ + 1, int(DGK["end_passes"]) - 1, info_e.get("failed")), flush=True)   # v36p prim
                break   # v36p prim
            N1, v1 = np.asarray(Ne, float), np.asarray(ve, float)   # v36p prim
    return (N1, v1)   # v36p prim'''),
("v36p prim: W2_v36p_fix_v3, the hook's signature (st, R and the iterate's propagated densities for the endpoint map)",
r'''def dgp_iterate_v36p(a, chs, anchor, f0, x, it, masks_prev, s_nodes, tau_s, E_rec, T_over_rs, j1, nsub):   # v36p prim''',
r'''def dgp_iterate_v36p(a, chs, anchor, f0, x, it, masks_prev, s_nodes, tau_s, E_rec, T_over_rs, j1, nsub, st=None, R=None, base1=None):   # v36p prim; W2_v36p_fix_v3: st, R and the iterate's propagated densities base1, the endpoint map's inputs'''),
("v36p prim: W2_v36p_fix_v3, the hook's docstring",
r'''The chord runs from f0 to the three-node rule's endpoint (dgp_chord_end_v36p: the cubic through f0 and the nodes at s = 1, the verification's chord at the fixed point).''',
r'''The chord runs from f0 to the endpoint's input fields as the endpoint call forms them (dgp_end_input_v36p, W2_v36p_fix_v3: the node fields' quadratic extrapolation to s = 1 through the endpoint map's passes before its last; the verification's chord up to the fields' motion since the decision).'''),
("v36p prim: W2_v36p_fix_v3, the hook's iterate 1 (the endpoint call's first input for the length test; the passes only where a decision is made)",
r'''    f1 = dgp_chord_end_v36p(f0, x, s_nodes)   # v36p prim''',
r'''    f1 = dgp_end_input_v36p(a, chs, st, R, base1, anchor, x, passes=False) if it == 1 else None   # v36p prim: W2_v36p_fix_v3: at iterate 1 the endpoint call's first input (the node fields' quadratic extrapolation to s = 1; f0 when the start's node fields are f0: the length test); the chord's end through the endpoint map's passes is formed below, only where a decision is made'''),
("v36p prim: W2_v36p_fix_v3, the no-length line",
r'''            print("  v36p prim T/rs %.4f sub-step %d/%d decision at iterate 1: the start's node fields are the start fields at every node and the chord to the three-node rule's endpoint has no length (%.1e of the fields): the anchor seeds iterate 1, the set is decided at iterate 2 from the first iterate's node fields" % (T_over_rs, j1, nsub, length), flush=True)   # v36p prim''',
r'''            print("  v36p prim T/rs %.4f sub-step %d/%d decision at iterate 1: the start's node fields are the start fields at every node, so the endpoint call's first input (their quadratic extrapolation to s = 1) is f0 and the chord has no length (%.1e of the fields): the anchor seeds iterate 1, the set is decided at iterate 2 from the first iterate's node fields" % (T_over_rs, j1, nsub, length), flush=True)   # v36p prim'''),
("v36p prim: W2_v36p_fix_v3, the seed's chord end",
r'''        else:   # v36p prim
            dec, orc = dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, DGP["level_min"], DGP["level_max"])   # v36p prim''',
r'''        else:   # v36p prim
            f1 = dgp_end_input_v36p(a, chs, st, R, base1, anchor, x, T_over_rs=T_over_rs, j1=j1, nsub=nsub)   # v36p prim: W2_v36p_fix_v3: the chord's end as the endpoint call receives it (the extrapolation through the endpoint map's passes before its last, at this iterate's propagated densities)
            dec, orc = dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, DGP["level_min"], DGP["level_max"])   # v36p prim'''),
("v36p prim: W2_v36p_fix_v3, the re-decision's chord end",
r'''    tag = ("decision at iterate %d" % it) if DGP["hold"] is None else ("re-decision %d at iterate %d" % (DGP["round"] + 1, it))   # v36p prim
    dec, orc = dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, DGP["level_min"], DGP["level_max"])   # v36p prim''',
r'''    tag = ("decision at iterate %d" % it) if DGP["hold"] is None else ("re-decision %d at iterate %d" % (DGP["round"] + 1, it))   # v36p prim
    f1 = dgp_end_input_v36p(a, chs, st, R, base1, anchor, x, T_over_rs=T_over_rs, j1=j1, nsub=nsub)   # v36p prim: W2_v36p_fix_v3: the chord's end as the endpoint call receives it, from this iterate's node fields and propagated densities under the current hold
    dec, orc = dgp_run_v36p(a, chs, anchor, f0, f1, s_nodes, tau_s, DGP["level_min"], DGP["level_max"])   # v36p prim'''),
("v36p prim: W2_v36p_fix_v3, the hook's call (st, R and this iterate's base1 passed through)",
r'''        if DGP["on"]:   # v36p prim: the hold decided at the first iterate (the seed) and re-decided on the chord to the three-node rule's endpoint when the node fields have settled, until it is the chord's set (W2_v36p_fix_v1, _v2); masks still holds the previous iterate's tracked node masks here (the line below resets it): a set that left the hold is read off them
            dgp_iterate_v36p(a, chs, anchor, f0, x, it, (masks if it > 1 else None), s_nodes, tau_s, E_rec, T0 / rs, j + 1, nsub)   # v36p prim''',
r'''        if DGP["on"]:   # v36p prim: the hold decided at the first iterate (the seed) and re-decided on the chord to the endpoint's input fields (as the endpoint call forms them: W2_v36p_fix_v3) when the node fields have settled, until it is the chord's set (W2_v36p_fix_v1, _v2); masks still holds the previous iterate's tracked node masks here (the line below resets it): a set that left the hold is read off them
            dgp_iterate_v36p(a, chs, anchor, f0, x, it, (masks if it > 1 else None), s_nodes, tau_s, E_rec, T0 / rs, j + 1, nsub, st=st, R=R, base1=base1)   # v36p prim; W2_v36p_fix_v3: st, R and this iterate's propagated densities base1 for the endpoint map'''),
("v36p prim: W2_v36p_fix_v4, the engine's docstring (the flips and settle-max guard on every hold: the seed's as well as a replacement's)",
r'''after a replacement a tracked node set leaving the hold --dg-prim-flips times, or a solve not settled [v36p prim]
--dg-prim-settle-max iterates later, reverts the hold to the one before (a crossing on a node: not held, the halving separates them); at a step's first sub-step the start's node [v36p prim]''',
r'''after every decision (W2_v36p_fix_v4: a replacement's hold watched from the next iterate for --dg-prim-settle-max iterates; the seed's from --dg-prim-settle iterates after it [v36p prim]
through the rest of the solve, only at iterates whose node fields' change is at most 1000 times --dg-prim-redecide, the sets of fields still forming being transients) a tracked [v36p prim]
node set leaving the hold --dg-prim-flips times, or a solve not settled --dg-prim-settle-max iterates after the decision, reverts the hold to the one before (the seed's to the [v36p prim]
anchor's tracking; a crossing on a node: not held, the halving separates them); at a step's first sub-step the start's node [v36p prim]'''),
("v36p prim: W2_v36p_fix_v4, the run label",
r'''a tracked set leaving the hold %d times after a replacement, or a solve not settled %d iterates after it, reverts the hold: a crossing on a node, not held; W2_v36p_fix_v1, _v2), held at the three nodes and the endpoint, verified on the converged chord; a sub-step not stable, not held or with a jump that does not fall %s" % (   # v36p prim
        getattr(PRIM, "VERSION", "?"), DGP["rule_obj"].describe() if DGP["rule_obj"] is not None else DGP["rule"], DGP["level_min"], DGP["level_max"], DGP["w_mix"], DGP["eig_tol"], DGP["mirror"], DGP["redecide"], DGP["settle"], DGP["rounds"], DGP["flips"], DGP["settle_max"], "halves (gate 1's retry path)" if DGP["halve"] else "is printed not printable, no halving"))   # v36p prim''',
r'''a tracked set leaving the hold %d times after a decision (a replacement's hold watched from the next iterate for %d iterates, the seed's from %d iterates after it through the solve at a node fields' change of at most %.1e), or a solve not settled %d iterates after the decision, reverts the hold (the seed's to the anchor's tracking): a crossing on a node, not held; W2_v36p_fix_v1, _v2, _v4), held at the three nodes and the endpoint, verified on the converged chord; a sub-step not stable, not held or with a jump that does not fall %s" % (   # v36p prim
        getattr(PRIM, "VERSION", "?"), DGP["rule_obj"].describe() if DGP["rule_obj"] is not None else DGP["rule"], DGP["level_min"], DGP["level_max"], DGP["w_mix"], DGP["eig_tol"], DGP["mirror"], DGP["redecide"], DGP["settle"], DGP["rounds"], DGP["flips"], DGP["settle_max"], DGP["settle"], 1000.0 * DGP["redecide"], DGP["settle_max"], "halves (gate 1's retry path)" if DGP["halve"] else "is printed not printable, no halving"))   # v36p prim'''),
("v36p prim: W2_v36p_fix_v4, dgp_revert_v36p (the seed's hold, with no hold before it, reverts to the anchor's tracking)",
r'''def dgp_revert_v36p(it, ch_, why, T_over_rs, j1, nsub):   # v36p prim
    """version 36p (W2_v36p_fix_v2): the hold reverted to the one before the last replacement (the hold under which the fields were settled when the replacement was decided); the sub-step is flagged a cycle (a crossing on a node), re-decided no more, not held: the solve re-converges, the verification records it and the halving separates the crossing from the node."""   # v36p prim
    hh = DGP["hold_hist"]; dh = DGP["dec_hist"]   # v36p prim
    prev_h = hh[-2] if len(hh) >= 2 else None; prev_d = dh[-2] if len(dh) >= 2 else None; to_round = max(len(hh) - 2, 0)   # v36p prim
    DGP.update({"hold": prev_h, "decision": prev_d, "cycle": True, "armed": False, "revert": {"at": int(it), "to_round": int(to_round), "why": why, "off_hold": [[int(i_), [[int(k_), str(kap)] for k_, kap in o_]] for i_, o_ in DGP["off_hold"]]}})   # v36p prim
    print("  v36p prim T/rs %.4f sub-step %d/%d revert at iterate %d (node fields' change %s): %s; the hold reverts to that of round %d (the sets the fields had settled under)%s, is re-decided no more and is not held: the solve re-converges, the verification records it, the halving separates the crossing from the node" % (T_over_rs, j1, nsub, it, ch_, why, to_round, "" if prev_h is not None else " (none: the anchor's tracking, version 36l)"), flush=True)   # v36p prim''',
r'''def dgp_revert_v36p(it, ch_, why, T_over_rs, j1, nsub):   # v36p prim
    """version 36p (W2_v36p_fix_v2, _v4): the hold reverted to the one before the last replacement (the hold under which the fields were settled when the replacement was decided) or, when the seed's hold is reverted (W2_v36p_fix_v4: no hold before it), to the anchor's tracking (version 36l's: hold None, to_round None); the sub-step is flagged a cycle (a crossing on a node), re-decided no more, not held: the solve re-converges, the verification records it and the halving separates the crossing from the node."""   # v36p prim
    hh = DGP["hold_hist"]; dh = DGP["dec_hist"]   # v36p prim
    prev_h = hh[-2] if len(hh) >= 2 else None; prev_d = dh[-2] if len(dh) >= 2 else None; to_round = (len(hh) - 2) if len(hh) >= 2 else None   # v36p prim; W2_v36p_fix_v4: None when the seed's hold reverts to the anchor's tracking
    DGP.update({"hold": prev_h, "decision": prev_d, "cycle": True, "armed": False, "revert": {"at": int(it), "to_round": (int(to_round) if to_round is not None else None), "why": why, "off_hold": [[int(i_), [[int(k_), str(kap)] for k_, kap in o_]] for i_, o_ in DGP["off_hold"]]}})   # v36p prim
    print("  v36p prim T/rs %.4f sub-step %d/%d revert at iterate %d (node fields' change %s): %s; the hold reverts to %s, is re-decided no more and is not held: the solve re-converges, the verification records it, the halving separates the crossing from the node" % (T_over_rs, j1, nsub, it, ch_, why, ("that of round %d (the sets the fields had settled under)" % to_round) if prev_h is not None else "the anchor's tracking (version 36l: no hold before the seed; W2_v36p_fix_v4)"), flush=True)   # v36p prim'''),
("v36p prim: W2_v36p_fix_v4, the hook's docstring",
r'''Later iterates: after a replacement, a tracked node mask differing from the hold at DGP["flips"] iterates within DGP["settle_max"] of it (the set alternates with the fields: a crossing on a node), or a change still above 1000 times DGP["redecide"] DGP["settle_max"] iterates after it, reverts the hold to the one before (dgp_revert_v36p: a cycle, not held, the halving's matter).''',
r'''Later iterates: after every decision (W2_v36p_fix_v4: the seed's as well as a replacement's), a tracked node mask differing from the hold at DGP["flips"] iterates (the set alternates with the fields, or stays off the hold: a crossing on a node; counted from the next iterate after a replacement, whose fields were settled, within DGP["settle_max"] iterates of it, and after the seed, which is decided on moving fields, from DGP["settle"] iterates after it through the rest of the solve but only at iterates whose node fields' change is at most 1000 times DGP["redecide"]: the sets of fields still forming are transients, read on bench1 at changes of 1e-2 near crossings 0.05 of the chord off a node, and above that change the settle-max rule is the guard; a seed's fields can form as late as DGP["settle_max"] iterates after it, so its readings carry no such window), or a change still above 1000 times DGP["redecide"] DGP["settle_max"] iterates after the decision, reverts the hold to the one before, the seed's to the anchor's tracking (dgp_revert_v36p: a cycle, not held, the halving's matter).'''),
("v36p prim: W2_v36p_fix_v4, the guard on every hold (a replacement's watched from the next iterate, the seed's from --dg-prim-settle iterates after its decision)",
r'''    if DGP["hold"] is not None and DGP["round"] >= 1 and (not DGP["cycle"]) and DGP["decided_at"] is not None and it <= DGP["decided_at"] + DGP["settle_max"]:   # v36p prim: W2_v36p_fix_v2 (ii): after a replacement, the tracked sets against the hold and the settling
        off = dgp_off_hold_v36p(DGP["hold"], masks_prev)   # v36p prim
        if off:   # v36p prim
            DGP["off_hold"].append((int(it) - 1, off))   # v36p prim
            print("  v36p prim T/rs %.4f sub-step %d/%d iterate %d: the tracked set of iterate %d left the hold at node-channel(s) %s (%d of %d since the replacement at iterate %d; node fields' change %s)" % (T_over_rs, j1, nsub, it, it - 1, ",".join("%d:%s" % (k_, kap) for k_, kap in off), len(DGP["off_hold"]), DGP["flips"], DGP["decided_at"], ch_), flush=True)   # v36p prim
            if len(DGP["off_hold"]) >= DGP["flips"]:   # v36p prim
                dgp_revert_v36p(it, ch_, "the tracked set left the hold at %d iterate(s) after the replacement at iterate %d (node-channels %s): the set at the node alternates with the fields, a crossing on a node" % (len(DGP["off_hold"]), DGP["decided_at"], ",".join(sorted(set("%d:%s" % (k_, kap) for _i, o_ in DGP["off_hold"] for k_, kap in o_)))), T_over_rs, j1, nsub)   # v36p prim
                return   # v36p prim
        if it == DGP["decided_at"] + DGP["settle_max"] and change is not None and change > 1000.0 * DGP["redecide"]:   # v36p prim
            dgp_revert_v36p(it, ch_, "the solve has not settled %d iterates after the replacement at iterate %d (node fields' change %s against %.1e)" % (DGP["settle_max"], DGP["decided_at"], ch_, 1000.0 * DGP["redecide"]), T_over_rs, j1, nsub)   # v36p prim
            return   # v36p prim''',
r'''    if DGP["hold"] is not None and (not DGP["cycle"]) and DGP["decided_at"] is not None and (DGP["round"] == 0 or it <= DGP["decided_at"] + DGP["settle_max"]):   # v36p prim: W2_v36p_fix_v2 (ii), _v4: after every decision (the seed's as well as a replacement's), the tracked sets against the hold and the settling; a replacement's hold within DGP["settle_max"] iterates of it, the seed's through the solve (its readings begin when the fields have formed, as late as DGP["settle_max"] iterates after it)
        what = "replacement" if DGP["round"] >= 1 else "decision"   # v36p prim: W2_v36p_fix_v4: the hold under guard is a replacement's (round >= 1) or the seed's
        off = dgp_off_hold_v36p(DGP["hold"], masks_prev) if (DGP["round"] >= 1 or (it >= DGP["decided_at"] + DGP["settle"] and change is not None and change <= 1000.0 * DGP["redecide"])) else []   # v36p prim: W2_v36p_fix_v4: a replacement's hold is watched from the next iterate (its fields were settled); the seed's, decided on moving fields, from DGP["settle"] iterates after its decision and only at iterates whose node fields' change is at most 1000 times DGP["redecide"] (the sets of fields still forming are transients, not flips; above that change the settle-max rule below is the guard)
        if off:   # v36p prim
            DGP["off_hold"].append((int(it) - 1, off))   # v36p prim
            print("  v36p prim T/rs %.4f sub-step %d/%d iterate %d: the tracked set of iterate %d left the hold at node-channel(s) %s (%d of %d since the %s at iterate %d; node fields' change %s)" % (T_over_rs, j1, nsub, it, it - 1, ",".join("%d:%s" % (k_, kap) for k_, kap in off), len(DGP["off_hold"]), DGP["flips"], what, DGP["decided_at"], ch_), flush=True)   # v36p prim
            if len(DGP["off_hold"]) >= DGP["flips"]:   # v36p prim
                dgp_revert_v36p(it, ch_, "the tracked set left the hold at %d iterate(s) after the %s at iterate %d (node-channels %s): the set at the node alternates with the fields, a crossing on a node" % (len(DGP["off_hold"]), what, DGP["decided_at"], ",".join(sorted(set("%d:%s" % (k_, kap) for _i, o_ in DGP["off_hold"] for k_, kap in o_)))), T_over_rs, j1, nsub)   # v36p prim
                return   # v36p prim
        if it == DGP["decided_at"] + DGP["settle_max"] and change is not None and change > 1000.0 * DGP["redecide"]:   # v36p prim
            dgp_revert_v36p(it, ch_, "the solve has not settled %d iterates after the %s at iterate %d (node fields' change %s against %.1e)" % (DGP["settle_max"], what, DGP["decided_at"], ch_, 1000.0 * DGP["redecide"]), T_over_rs, j1, nsub)   # v36p prim
            return   # v36p prim'''),
("v36p prim: W2_v36p_fix_v4, the verification line (a revert to the anchor's tracking has no round)",
r'''(", reverted at iterate %d to round %d (%s)" % (pr["revert"]["at"], pr["revert"]["to_round"], pr["revert"]["why"])) if pr.get("revert") else ""''',
r'''(", reverted at iterate %d to %s (%s)" % (pr["revert"]["at"], ("round %d" % pr["revert"]["to_round"]) if pr["revert"].get("to_round") is not None else "the anchor's tracking", pr["revert"]["why"])) if pr.get("revert") else ""'''),
("v36p prim: W2_v36p_fix_v4, the options' help",
r'''    ap.add_argument("--dg-prim-settle", dest="dg_prim_settle", type=int, default=3, help="(v36p) W2_v36p_fix_v2: the fewest iterates between a decision of the hold and its next check (the solve answers a replacement over the mixer's history; one iterate later the fields have not moved)")   # v36p prim
    ap.add_argument("--dg-prim-settle-max", dest="dg_prim_settle_max", type=int, default=10, help="(v36p) W2_v36p_fix_v2: a solve whose node fields' change is still above 1000 times --dg-prim-redecide this many iterates after a replacement of the hold reverts to the hold before it (not held; the halving)")   # v36p prim
    ap.add_argument("--dg-prim-flips", dest="dg_prim_flips", type=int, default=2, help="(v36p) W2_v36p_fix_v2: a tracked node set differing from the hold at this many iterates after a replacement (within --dg-prim-settle-max of it) reverts the hold: a crossing on a node, the set alternating with the fields (not held; the halving)")   # v36p prim''',
r'''    ap.add_argument("--dg-prim-settle", dest="dg_prim_settle", type=int, default=3, help="(v36p) W2_v36p_fix_v2: the fewest iterates between a decision of the hold and its next check (the solve answers a replacement over the mixer's history; one iterate later the fields have not moved); W2_v36p_fix_v4: also the iterates after the seed's decision before a tracked set off its hold is counted")   # v36p prim
    ap.add_argument("--dg-prim-settle-max", dest="dg_prim_settle_max", type=int, default=10, help="(v36p) W2_v36p_fix_v2, _v4: a solve whose node fields' change is still above 1000 times --dg-prim-redecide this many iterates after a decision of the hold (the seed's or a replacement's) reverts to the hold before it, the seed's to the anchor's tracking (not held; the halving); also the window of a replacement's flips guard")   # v36p prim
    ap.add_argument("--dg-prim-flips", dest="dg_prim_flips", type=int, default=2, help="(v36p) W2_v36p_fix_v2, _v4: a tracked node set differing from the hold at this many iterates after a decision of the hold (a replacement's watched from the next iterate for --dg-prim-settle-max iterates; the seed's from --dg-prim-settle iterates after it through the solve, only at iterates whose node fields' change is at most 1000 times --dg-prim-redecide) reverts the hold, the seed's to the anchor's tracking: a crossing on a node, the set alternating with the fields or staying off the hold (not held; the halving)")   # v36p prim'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36p: the OLD string of '%s' occurs %d times in the source (1 required): the source is not build_engine_collapse_v36l.py's engine_collapse_v36l.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36p: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36l.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36p.py")
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36p: the reverse replacements do not restore the source (a NEW text overlaps another)")
    marked = sum(1 for l_ in out.splitlines() if ("v36p prim" in l_))
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled; needs assign_primitive_v1.py beside it under --dg-assign prim" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
