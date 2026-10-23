#!/usr/bin/env python
"""build_engine_collapse_v36g.py (reviewer A for the busi lane; W3 (critical) and W4 (b) of WEB40I_TASKS/busi.md; requirements edit 26): writes
engine_collapse_v36g.py from engine_collapse_v36d.py by 19 exact string replacements in the form of build_engine_collapse_v36e.py. Each OLD is copied
exactly from engine_collapse_v36d.py and must occur exactly once; each NEW contains its OLD plus inserted text, so that replacing every NEW by its OLD
restores the source byte for byte (checked here); every line the replacements add or change carries the mark "v36g W3" (checked here by the unified
diff: every added line contains it; the pair-sources lines carry "v36g W3 (W4b)"). The new options are off by default: with --completion absent and
--pair-sources on (its default) nothing of the inserted code changes any number, printed line or stored array (the record loop runs through the same
operations: steps_v36g yields the engine's own indices, cpl_solve_v36g calls solve() unchanged, accumulate_sources_v36g evaluates the engine's own
expressions on the same arrays), so v36g with the options off is v36d bitwise and the json's "args" dict alone gains the new keys (the runtime gate:
engine_collapse_v36g_selftest.py --gate on 3 records of the r_s/a 16 fixture on both engines; RUN_W3_W4b_v1.md section 1).
Where the changed regions lie (so that engine_collapse_v36f, busi W2's --step dg over the same v36d, merges with this one by applying this list to
v36f's output, or v36f's to this one's): nothing here touches the propagation block at the record's end (the exponential of the record's fields:
"t_p = time.time()" to "T += a.tau") nor resume_slice_v34's propagation, the Newton loops or the continuation; this builder touches (1) the module
docstring's first line, (2) the module state after NSR, (3) the functions before run(), (4) resume_slice_v34's argument check and (5) its dtype line,
(6) the run label after v34's, (7) the head after cstr, (8) save_fields' first line, (9) the loop header, (10) the step's first 3 lines, (11) and (12)
the accumulation line of sources() and sources_with(), (13) do_solve's solve call, (14) the 3 lines after the record's accepted charge, (15) the
indicator's and r_0 check's calls, (16) the record entry after the v36 entry, (17) the record line, (18) the argument list before parse_args and
(19) the validation before C5.backend; (6), (18) and (19) are anchors v36f also extends: after the W2/W3 red team their OLDs are the lines that
follow both builders' insertions (rs = a.rs with the N_B line; a = ap.parse_args() with the m = 0 check; C5.backend with run), so this list chains
after v36f's output (build v36d -> v36f -> v36g, in that order only: v36f's docstring OLD carries the opening quotes and does not occur in this
builder's output); the docstring OLD here is the first line's text without its opening quotes so that it still occurs once after v36f's docstring
is prepended. A chained engine composes the options only after dg_node_v36f routes its solve through cpl_solve_v36g and its accumulation through
accumulate_sources_v36g (the review's section 8); until then --step dg is not to be run with --completion or --pair-sources off.
Usage: python build_engine_collapse_v36g.py [engine_collapse_v36d.py path] [engine_collapse_v36g.py path]   (defaults: beside this script)"""
import difflib
import hashlib
import os
import py_compile
import sys

REPLACEMENTS = [
("docstring",
r'''engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change''',
r'''engine_collapse_v36g.py (v36g W3; reviewer A for the busi lane: W3 (critical) and W4 (b) of WEB40I_TASKS/busi.md, requirements edit 26): version 36g =  [v36g W3]
version 36d with the options below, each off by default, so that the defaults are version 36d bitwise in every record number, every printed line and every  [v36g W3]
stored array (the json's "args" dict alone gains the new keys); written by build_engine_collapse_v36g.py as exact replacements over engine_collapse_v36d.py,  [v36g W3]
every inserted line marked "v36g W3".  [v36g W3]
--completion frozen-sources | paired-avg | point-core [--completion-from-T-over-rs T_g] [--completion-radius-over-rs X] [--completion-gate-value 0.1]: the  [v36g W3]
    completion control. The hydrodynamic gate on a record's dense fields is G(r) = sigma max(|v'/N|, |v/(r N)|) <= 0.1 with sigma = ell (the engine's Gaussian  [v36g W3]
    kernel's standard deviation; at a root v = -N it is sigma/R exactly): a record is gated when G <= 0.1 everywhere from ell/16 to the collar, and its gate  [v36g W3]
    radius is the outermost r with G > 0.1 (0 when gated). Every record is first solved as the record's own; while nothing is armed its gate profile is tested  [v36g W3]
    and printed, and a gated record's accepted sources and charge are kept as the latest gated record's; the first ungated record at or after T_g arms the  [v36g W3]
    completion with the gate radius R_* (the option's X r_s, else that record's own gate radius rounded up to the next multiple of ell) and is re-solved at  [v36g W3]
    the same clock from the same propagated state with the completion applied; every later record carries it. The blend at the gate radius: beta(R) = 1 for  [v36g W3]
    R <= R_* - ell, 0 for R >= R_* + ell, the quintic smoothstep between (C^2 at both ends), evaluated at the site's shell radius.  [v36g W3]
    (i) frozen-sources: the field solver sees eps_s = (1 - beta) eps + beta lambda eps_g and J_s likewise, eps_g and J_g the gated record's accepted sources  [v36g W3]
    (frozen exactly for R <= R_* - ell, the live sources exactly for R >= R_* + ell); lambda is the charge projection: one scalar per field solve, fixed by a  [v36g W3]
    secant iteration so that the solved collar charge equals its target to --completion-charge-tol. --completion-charge matched (default): the stored member's  [v36g W3]
    E at the same clock, read from --completion-run (default --resume-run), so that the exterior sees the record's own enclosed charge while the interior is  [v36g W3]
    the gated record's; gated: the gated record's E; none: lambda = 1. lambda - 1 measures the net charge the propagated state has carried across R_* plus  [v36g W3]
    the stored history's step drift at that clock.  [v36g W3]
    (ii) paired-avg: the paired averaging widened to --completion-ell-in (8 a) inside R_*, the run's ell (4 a) outside: a shell-to-shell Gaussian mixing of  [v36g W3]
    width ell_x = sqrt(ell_in^2 - ell_out^2) acts on the site sources with the blend at the source's shell, B = (1 - beta) 1 + M diag(beta), before the  [v36g W3]
    engine's ell_out solve (Gaussians compose: the interior shells are smoothed with the width ell_in), and B^T acts on the back-averaged site fields after it,  [v36g W3]
    the weighted adjoint of the composite source averaging (the boundary charge still generates the Hartree evolution); M_0's columns are normalized to unit  [v36g W3]
    sum (the charge, K_0 on constants, exact) and M_1's to the linear-flow identity (K_1 on v = c r, exact); the identities, the adjoint pairing and the  [v36g W3]
    composition check print at arming.  [v36g W3]
    (iii) point-core: the site fields entering h_m are (1 - beta) times the back-averaged fields plus beta times the dense fields point-sampled at the sites:  [v36g W3]
    the core's coupling point-sampled inside R_*, the source averaging unchanged (the adjoint pairing broken inside R_*: the engine's own point-sampling  [v36g W3]
    control, which differs from the paired form by (ell/L_geom)^2).  [v36g W3]
    The record loop's field solve goes through cpl_solve_v36g (solve() itself unchanged); the indicator and the r_0 check read the sources the solver saw;  [v36g W3]
    the json head and every record carry the completion's name and parameters (head["v36g"], rec["v36g"]); the sources the solver saw, lambda and the  [v36g W3]
    targets go to collapse_<tag>_v36g.npz beside the fields npz, whose eps_tot and J_tot stay the record's raw accepted sources.  [v36g W3]
--pair-sources off (W4 (b)): the dust-only sources: the pair terms w (dens(h, sea) - dens(h, R2)) and w (dens(Q, sea) - dens(Q, R2)) of every channel are  [v36g W3]
    dropped from eps and J and from nothing else (the sea is propagated, the reference R2 continued with its Weyl and LZ bookkeeping, the pair densities  [v36g W3]
    computed against it and stored as eps_pairs and J_pairs, the M1 ledger and the stresses unchanged); every record carries the dropped totals.  [v36g W3]
--resume-args-check relaxed: a key absent from the stored run's args matches when this call's value is the argument's default (printed); a slice without a  [v36g W3]
    dtype entry is read by its arrays' dtype. strict (the default) is version 34 (a)'s check.  [v36g W3]
Where version 36g touches version 36d (for the merge with engine_collapse_v36f, busi W2's --step dg over the same v36d): disjoint from the propagation block  [v36g W3]
    at the record's end and from resume_slice_v34's propagation; it changes the module state and the functions before run(); the run label, the head, the  [v36g W3]
    argument list and the validation before C5.backend (anchors v36f also extends); the loop header (steps_v36g), the step's first lines (the unconverged  [v36g W3]
    label's capture), the two source-accumulation lines (accumulate_sources_v36g), do_solve's call (cpl_solve_v36g), the 3 lines after the record's accepted  [v36g W3]
    charge (the gate test and the arming re-solve), the indicator's and r_0 check's calls, the record entry after the v36 entry, the record line, save_fields,  [v36g W3]
    and resume_slice_v34's argument check and dtype line. Version 36d's docstring follows.  [v36g W3]
[v36g W3: version 36d's docstring, verbatim, from here] engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change'''),
("module state",
r'''NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other''',
r'''NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other
CPL = {"requested": False, "mode": None, "armed": False, "redo": False, "last_gated": None, "gate_rows": [], "arming": None, "arming_own": None, "T_arm": None, "R_star": None, "R_star_auto": None, "beta": None, "mix": None, "frozen_eps": None, "frozen_J": None, "E_gated": None, "T_gated": None, "lambda": 1.0, "charge_mode": "matched", "charge_tol": 1e-9, "E_stored": {}, "store": {"T": [], "eps_solver": [], "J_solver": [], "lambda": [], "E_target": [], "E": []}}   # v36g W3 (--completion)
PSRC = {"on": True}   # v36g W3 (W4b) (--pair-sources off sets it False)'''),
("functions before run",
r'''def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)''',
r'''# --------------------------------------------------------------------------- v36g W3: the completion control (W3) and the dust-only sources (W4 (b))
def steps_v36g(step0, nsteps):  # v36g W3
    """v36g W3: the record loop's indices step0 .. nsteps as the engine's own range; the current index is yielded again when CPL["redo"] is set (the arming  [v36g W3]
    record's re-solve with the completion), so the run still ends at the same last record.  [v36g W3]"""  # v36g W3
    k = step0  # v36g W3
    while k <= nsteps:  # v36g W3
        CPL["redo"] = False  # v36g W3
        yield k  # v36g W3
        if not CPL["redo"]:  # v36g W3
            k += 1  # v36g W3


def accumulate_sources_v36g(w, ds, Js, dd, Jd, eps, J, eps_pairs, eps_dust, J_pairs):  # v36g W3 (W4b)
    """v36g W3 (W4b): one channel's contribution to the record's sources, in place and in the engine's own expressions (bitwise the default): eps += w (ds + dd),  [v36g W3]
    J += w (Js + Jd), eps_pairs += w ds, eps_dust += w dd, J_pairs += w Js; with --pair-sources off the pair terms w ds and w Js are dropped from eps and J  [v36g W3]
    alone (the dust-only sources), eps_pairs and J_pairs still accumulating them for the record and the fields npz.  [v36g W3]"""  # v36g W3 (W4b)
    if PSRC["on"]:  # v36g W3 (W4b)
        eps += w * (ds + dd); J += w * (Js + Jd); eps_pairs += w * ds; eps_dust += w * dd; J_pairs += w * Js  # v36g W3 (W4b)
    else:  # v36g W3 (W4b)
        eps += w * dd; J += w * Jd; eps_pairs += w * ds; eps_dust += w * dd; J_pairs += w * Js  # v36g W3 (W4b)


def _intervals_v36g(mask, r):  # v36g W3
    """v36g W3: the maximal runs of True in mask as [r_first, r_last] pairs.  [v36g W3]"""  # v36g W3
    out = []; i = 0; n = len(mask)  # v36g W3
    while i < n:  # v36g W3
        if mask[i]:  # v36g W3
            j = i  # v36g W3
            while j + 1 < n and mask[j + 1]:  # v36g W3
                j += 1  # v36g W3
            out.append([float(r[i]), float(r[j])]); i = j + 1  # v36g W3
        else:  # v36g W3
            i += 1  # v36g W3
    return out  # v36g W3


def gate_profile_v36g(info, ell, gate_value=0.1, h=None):  # v36g W3
    """v36g W3: the hydrodynamic gate on a solve's dense fields: G(r) = sigma max(|K_r|, |K_t|), K_r = v'/N, K_t = v/(r N), sigma = ell (the engine's Gaussian of  [v36g W3]
    standard deviation ell, no cell term), on the fine grid from ell/16 to the collar, v' by central differences with h = ell/64; gated where G <= gate_value  [v36g W3]
    everywhere; the gate radius is the outermost r with G > gate_value (0 when gated); at a root v = -N the value is sigma/R exactly.  [v36g W3]"""  # v36g W3
    rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float)  # v36g W3
    rf = rf[(rf >= ell / 16.0 - 1e-12) & (rf <= rc)]  # v36g W3
    h = ell / 64.0 if h is None else float(h)  # v36g W3
    Nf = np.asarray(info["N_dense"](rf), float).ravel(); vf = np.asarray(info["v_dense"](rf), float).ravel()  # v36g W3
    vp = (np.asarray(info["v_dense"](rf + h), float).ravel() - np.asarray(info["v_dense"](rf - h), float).ravel()) / (2.0 * h)  # v36g W3
    Ns = np.maximum(np.abs(Nf), 1e-300)  # v36g W3
    Kr = np.abs(vp) / Ns; Kt = np.abs(vf) / (rf * Ns)  # v36g W3
    G = float(ell) * np.maximum(Kr, Kt)  # v36g W3
    k = int(np.argmax(G)); over = np.nonzero(G > gate_value)[0]  # v36g W3
    return {"r": rf, "G": G, "G_max": float(G[k]), "r_at_max": float(rf[k]), "dominant_at_max": ("K_t" if Kt[k] >= Kr[k] else "K_r"),  # v36g W3
            "gated": bool(len(over) == 0), "gate_radius": (float(rf[over[-1]]) if len(over) else 0.0), "gate_value": float(gate_value), "sigma": float(ell),  # v36g W3
            "failing_intervals": _intervals_v36g(G > gate_value, rf)}  # v36g W3


def blend_v36g(R, R_star, w):  # v36g W3
    """v36g W3: the blend at the gate radius: beta(R) = 1 for R <= R_* - w, 0 for R >= R_* + w, the quintic smoothstep 1 - (10 s^3 - 15 s^4 + 6 s^5),  [v36g W3]
    s = (R - R_* + w)/(2 w), between (C^2 at both ends); w is the run's averaging scale ell, so the transition spans one averaging scale on each side.  [v36g W3]"""  # v36g W3
    R = np.asarray(R, float)  # v36g W3
    s = np.clip((R - (R_star - w)) / (2.0 * w), 0.0, 1.0)  # v36g W3
    return 1.0 - (10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5)  # v36g W3


def shell_kernels_v36g(r, Rn, ell):  # v36g W3
    """v36g W3: the shell-to-point kernels of the engine's 3-dimensional Gaussian of standard deviation ell (engine_fields_v12's analytic centre coefficients  [v36g W3]
    identify them): for a shell of unit total at radius R, K_0(r, R) = c sinh(x)/x and K_1(r, R) = c (cosh(x)/x - sinh(x)/x^2), c = (2 pi ell^2)^(-3/2)  [v36g W3]
    exp(-(r^2 + R^2)/(2 ell^2)), x = r R/ell^2, so that int 4 pi r^2 K_0 dr = 1 and int 4 pi r^2 K_1(r, R) c r dr = c R; the difference-of-Gaussians form  [v36g W3]
    for x >= 0.1 (no overflow), the series below it. r and R broadcast.  [v36g W3]"""  # v36g W3
    r = np.asarray(r, float); Rn = np.asarray(Rn, float)  # v36g W3
    ell = float(ell); x = r * Rn / ell ** 2  # v36g W3
    pref = (2.0 * np.pi * ell ** 2) ** (-1.5)  # v36g W3
    em = np.exp(-(r - Rn) ** 2 / (2.0 * ell ** 2)); ep = np.exp(-(r + Rn) ** 2 / (2.0 * ell ** 2))  # v36g W3
    xs = np.where(x > 0.0, x, 1.0)  # v36g W3
    K0d = pref * (em - ep) / (2.0 * xs); K1d = pref * ((em + ep) / (2.0 * xs) - (em - ep) / (2.0 * xs ** 2))  # v36g W3
    e0 = np.exp(-(r ** 2 + Rn ** 2) / (2.0 * ell ** 2))  # v36g W3
    K0s = pref * e0 * (1.0 + x ** 2 / 6.0 + x ** 4 / 120.0 + x ** 6 / 5040.0 + x ** 8 / 362880.0)  # v36g W3
    K1s = pref * e0 * (x / 3.0 + x ** 3 / 30.0 + x ** 5 / 840.0 + x ** 7 / 45360.0 + x ** 9 / 3991680.0)  # v36g W3
    small = x < 0.1  # v36g W3
    return np.where(small, K0s, K0d), np.where(small, K1s, K1d)  # v36g W3


def mixing_v36g(R, beta, ell_in, ell_out):  # v36g W3
    """v36g W3 (paired-avg): the shell-to-shell mixing M of width ell_x = sqrt(ell_in^2 - ell_out^2) over the sites with beta > 0 (Gaussians compose: the engine's  [v36g W3]
    ell_out kernel applied to the mixed shell totals is the ell_in kernel applied to the originals, to the accuracy of the unit-shell quadrature 4 pi R_n^2 a);  [v36g W3]
    B = (1 - beta) 1 + M diag(beta) acts on the sources and B^T on the back-averaged site fields, the weighted adjoint; M_0's columns normalized to unit sum  [v36g W3]
    (the charge, K_0 on constants, exact) and M_1's so that sum_n M_1[n, m] R_n = R_m (K_1 on v = c r, exact).  [v36g W3]"""  # v36g W3
    R = np.asarray(R, float); beta = np.asarray(beta, float)  # v36g W3
    ell_x = math.sqrt(float(ell_in) ** 2 - float(ell_out) ** 2)  # v36g W3
    cols = np.nonzero(beta > 0.0)[0]  # v36g W3
    K0, K1 = shell_kernels_v36g(R[:, None], R[None, cols], ell_x)  # v36g W3
    M0 = 4.0 * np.pi * R[:, None] ** 2 * K0; M1 = 4.0 * np.pi * R[:, None] ** 2 * K1  # v36g W3
    M0 = M0 / np.where(M0.sum(axis=0) > 0.0, M0.sum(axis=0), 1.0)[None, :]  # v36g W3
    s1 = (M1 * R[:, None]).sum(axis=0)  # v36g W3
    M1 = M1 * (R[cols] / np.where(s1 != 0.0, s1, 1.0))[None, :]  # v36g W3
    return {"cols": cols, "M0": M0, "M1": M1, "ell_x": ell_x, "ell_in": float(ell_in), "ell_out": float(ell_out), "beta": beta}  # v36g W3


def mix_forward_v36g(mx, eps, J):  # v36g W3
    """v36g W3: B eps and B J, the sources the ell_out solve sees.  [v36g W3]"""  # v36g W3
    c = mx["cols"]; b = mx["beta"][c]  # v36g W3
    eps = np.asarray(eps, float); J = np.asarray(J, float)  # v36g W3
    es = np.array(eps, dtype=float, copy=True); Js = np.array(J, dtype=float, copy=True)  # v36g W3
    es[c] -= b * eps[c]; es += mx["M0"] @ (b * eps[c])  # v36g W3
    Js[c] -= b * J[c]; Js += mx["M1"] @ (b * J[c])  # v36g W3
    return es, Js  # v36g W3


def mix_backward_v36g(mx, N, v):  # v36g W3
    """v36g W3: B^T N and B^T v on the back-averaged site fields, the adjoint of mix_forward_v36g.  [v36g W3]"""  # v36g W3
    c = mx["cols"]; b = mx["beta"][c]  # v36g W3
    N = np.asarray(N, float); v = np.asarray(v, float)  # v36g W3
    Nb = np.array(N, dtype=float, copy=True); vb = np.array(v, dtype=float, copy=True)  # v36g W3
    Nb[c] = (1.0 - b) * N[c] + b * (mx["M0"].T @ N); vb[c] = (1.0 - b) * v[c] + b * (mx["M1"].T @ v)  # v36g W3
    return Nb, vb  # v36g W3


def mix_identities_v36g(mx, R):  # v36g W3
    """v36g W3: the mixing's exact identities as the largest deviation: B on a unit source conserves the total (K_0 on constants), B^T on N = 1 gives 1, B^T on  [v36g W3]
    v = c R gives c R, and the adjoint pairing sum_n (B^T N)_n e_n = sum_n N_n (B e)_n on random fields (the same for the vector pair).  [v36g W3]"""  # v36g W3
    R = np.asarray(R, float); one = np.ones(len(R)); lin = 0.37 * R  # v36g W3
    es, Js = mix_forward_v36g(mx, one, lin)  # v36g W3
    Nb, vb = mix_backward_v36g(mx, one, lin)  # v36g W3
    rng = np.random.RandomState(36)  # v36g W3
    e = rng.standard_normal(len(R)); N_ = rng.standard_normal(len(R)); J_ = rng.standard_normal(len(R)); v_ = rng.standard_normal(len(R))  # v36g W3
    Be, BJ = mix_forward_v36g(mx, e, J_); BtN, Btv = mix_backward_v36g(mx, N_, v_)  # v36g W3
    sc0 = max(abs(float(N_ @ Be)), 1e-300); sc1 = max(abs(float(v_ @ BJ)), 1e-300)  # v36g W3
    return {"charge_sum_rel": float(np.sum(es) - np.sum(one)) / float(np.sum(one)), "K0_constant_max_abs": float(np.max(np.abs(Nb - 1.0))),  # v36g W3
            "K1_linear_max_rel": float(np.max(np.abs(vb - lin))) / float(np.max(np.abs(lin))),  # v36g W3
            "adjoint_K0_rel": abs(float(BtN @ e) - float(N_ @ Be)) / sc0, "adjoint_K1_rel": abs(float(Btv @ J_) - float(v_ @ BJ)) / sc1,  # v36g W3
            "mixed_sites": int(len(mx["cols"])), "ell_x": mx["ell_x"]}  # v36g W3


def composition_check_v36g(mx, R, R_star):  # v36g W3
    """v36g W3: a unit shell at the beta = 1 site nearest R_*/2, mixed by M then smoothed with the ell_out kernel, against the ell_in kernel of the unmixed shell on  [v36g W3]
    a fine radial grid: the largest deviation over the ell_in kernel's maximum, for K_0 (M_0) and K_1 (M_1); the unit-shell quadrature's accuracy.  [v36g W3]"""  # v36g W3
    R = np.asarray(R, float); c = mx["cols"]; inner = c[mx["beta"][c] >= 1.0]  # v36g W3
    if len(inner) == 0:  # v36g W3
        return {"note": "no site with beta = 1 (the gate radius lies below the blend's width)"}  # v36g W3
    m = int(inner[int(np.argmin(np.abs(R[inner] - 0.5 * R_star)))])  # v36g W3
    e = np.zeros(len(R)); e[m] = 1.0  # v36g W3
    es, Js = mix_forward_v36g(mx, e, e)  # v36g W3
    rg = np.linspace(0.05, float(R[m]) + 8.0 * mx["ell_in"], 2000)  # v36g W3
    K0o, K1o = shell_kernels_v36g(rg[:, None], R[None, :], mx["ell_out"])  # v36g W3
    K0i, K1i = shell_kernels_v36g(rg, float(R[m]), mx["ell_in"])  # v36g W3
    rho_n = K0o @ es; J_n = K1o @ Js  # v36g W3
    return {"site": m, "R_site": float(R[m]), "K0_rel_dev": float(np.max(np.abs(rho_n - K0i))) / max(float(np.max(np.abs(K0i))), 1e-300),  # v36g W3
            "K1_rel_dev": float(np.max(np.abs(J_n - K1i))) / max(float(np.max(np.abs(K1i))), 1e-300)}  # v36g W3


def load_stored_E_v36g(run, rs):  # v36g W3
    """v36g W3: the stored member's collar charge by clock, {round(T/rs, 6): E} from its collapse_<tag>.json (the matched charge projection's targets).  [v36g W3]"""  # v36g W3
    rdir, rtag = os.path.split(run)  # v36g W3
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag)))  # v36g W3
    return {round(float(r_["T"]) / rs, 6): float(r_["E"]) for r_ in J["records"] if r_.get("E") is not None}  # v36g W3


def frozen_target_v36g(T, rs):  # v36g W3
    """v36g W3 (frozen-sources): the charge projection's target at the clock T: matched, the stored member's E at the same clock (the gated record's E with a  [v36g W3]
    FALLBACK label where the stored history has no record there); gated, the gated record's E; none, no target (lambda = 1). Returns (target, label).  [v36g W3]"""  # v36g W3
    mode = CPL["charge_mode"]  # v36g W3
    if mode == "none":  # v36g W3
        return None, "none (lambda = 1)"  # v36g W3
    if mode == "matched":  # v36g W3
        E_t = CPL["E_stored"].get(round(T / rs, 6))  # v36g W3
        if E_t is not None:  # v36g W3
            return float(E_t), "matched: the stored member's E at T/rs %.6f" % (T / rs)  # v36g W3
        return float(CPL["E_gated"]), "gated (FALLBACK: the stored member has no record at T/rs %.6f)" % (T / rs)  # v36g W3
    return float(CPL["E_gated"]), "gated: the gated record's E"  # v36g W3


def frozen_solve_v36g(eps, J, g, R, ell, T, rs):  # v36g W3
    """v36g W3 (frozen-sources): the solve with eps_s = (1 - beta) eps + beta lambda eps_g and J_s likewise (eps_g, J_g the gated record's accepted sources),  [v36g W3]
    lambda fixed by the secant iteration on the solved collar charge E(lambda) = target to --completion-charge-tol (the first slope the frozen bare charge  [v36g W3]
    sum(beta eps_g), then secant steps, at most 8 further solves; lambda starts from the previous evaluation's; a failed solve keeps the best point).  [v36g W3]
    Returns the solve's tuple with info["v36g_sources"] the sources the solver saw and info["v36g"] the projection's entry.  [v36g W3]"""  # v36g W3
    beta = CPL["beta"]; eg = CPL["frozen_eps"]; Jg = CPL["frozen_J"]  # v36g W3
    eps = np.asarray(eps, float); J = np.asarray(J, float)  # v36g W3
    target, label = frozen_target_v36g(T, rs)  # v36g W3

    def at(lam):  # v36g W3
        es = (1.0 - beta) * eps + beta * (lam * eg); Js = (1.0 - beta) * J + beta * (lam * Jg)  # v36g W3
        out = solve(es, Js, g, R, ell, None)  # v36g W3
        return es, Js, out, (None if out[3].get("failed") else float(FLD.charge(out[3], g)))  # v36g W3
    tol = float(CPL["charge_tol"]); lam = float(CPL["lambda"])  # v36g W3
    es, Js, out, E_ = at(lam); hist = [[lam, E_]]  # v36g W3
    best = [lam, es, Js, out, E_]  # v36g W3
    if target is not None and E_ is not None and abs(E_ - target) > tol * abs(target):  # v36g W3
        slope = float(np.sum(beta * eg))  # v36g W3
        for it_ in range(8):  # v36g W3
            if not (np.isfinite(slope) and abs(slope) > 0.0):  # v36g W3
                break  # v36g W3
            lam_n = lam + (target - E_) / slope  # v36g W3
            es_n, Js_n, out_n, E_n = at(lam_n); hist.append([lam_n, E_n])  # v36g W3
            if E_n is None:  # v36g W3
                break  # v36g W3
            if abs(E_n - target) < abs(best[4] - target):  # v36g W3
                best = [lam_n, es_n, Js_n, out_n, E_n]  # v36g W3
            slope = (E_n - E_) / (lam_n - lam) if lam_n != lam else slope  # v36g W3
            lam, E_ = lam_n, E_n  # v36g W3
            if abs(E_n - target) <= tol * abs(target):  # v36g W3
                break  # v36g W3
    lam, es, Js, out, E_ = best  # v36g W3
    CPL["lambda"] = lam  # v36g W3
    Nn, vn, Wn, info = out  # v36g W3
    if not info.get("failed"):  # v36g W3
        info["v36g_sources"] = (es, Js)  # v36g W3
        info["v36g"] = {"lambda": lam, "E": E_, "E_target": target, "E_target_source": label, "charge_residual_rel": ((E_ - target) / target if (target is not None and E_ is not None) else None),  # v36g W3
                        "charge_solves": len(hist), "charge_history": hist, "charge_converged": bool(target is None or (E_ is not None and abs(E_ - target) <= tol * abs(target)))}  # v36g W3
    return Nn, vn, Wn, info  # v36g W3


def cpl_solve_v36g(eps, J, g, R, ell, T, rs):  # v36g W3
    """v36g W3: the record's field solve: the engine's solve() unchanged while no completion is armed (bitwise the default); armed, the completion's solve, with  [v36g W3]
    info["v36g_sources"] the site sources the solver saw and info["v36g"] the completion's entry for the record.  [v36g W3]"""  # v36g W3
    if not CPL["armed"]:  # v36g W3
        return solve(eps, J, g, R, ell, None)  # v36g W3
    if CPL["mode"] == "frozen-sources":  # v36g W3
        return frozen_solve_v36g(eps, J, g, R, ell, T, rs)  # v36g W3
    if CPL["mode"] == "paired-avg":  # v36g W3
        es, Js = mix_forward_v36g(CPL["mix"], eps, J)  # v36g W3
        Nn, vn, Wn, info = solve(es, Js, g, R, ell, None)  # v36g W3
        if info.get("failed"):  # v36g W3
            return Nn, vn, Wn, info  # v36g W3
        Nb, vb = mix_backward_v36g(CPL["mix"], Nn, vn)  # v36g W3
        info["v36g_sources"] = (es, Js)  # v36g W3
        info["v36g"] = {"ell_in": CPL["mix"]["ell_in"], "ell_out": CPL["mix"]["ell_out"], "ell_x": CPL["mix"]["ell_x"], "mixed_sites": int(len(CPL["mix"]["cols"]))}  # v36g W3
        return Nb, vb, Wn, info  # v36g W3
    Nn, vn, Wn, info = solve(eps, J, g, R, ell, None)  # point-core  # v36g W3
    if info.get("failed"):  # v36g W3
        return Nn, vn, Wn, info  # v36g W3
    beta = CPL["beta"]  # v36g W3
    Nb = (1.0 - beta) * np.asarray(Nn, float) + beta * np.asarray(info["N_point"], float); vb = (1.0 - beta) * np.asarray(vn, float) + beta * np.asarray(info["v_point"], float)  # v36g W3
    info["v36g_sources"] = (np.asarray(eps, float), np.asarray(J, float))  # v36g W3
    info["v36g"] = {"point_sampled_sites": int(np.sum(beta >= 1.0)), "blended_sites": int(np.sum((beta > 0.0) & (beta < 1.0)))}  # v36g W3
    return Nb, vb, Wn, info  # v36g W3


def completion_desc_v36g(a):  # v36g W3
    """v36g W3: the completion in words (the run label, the head).  [v36g W3]"""  # v36g W3
    base = ("completion %s from the first record failing the hydrodynamic gate sigma max(|v'/N|, |v/(r N)|) <= %g (sigma = ell = %d a) at or after T/rs %g, inside the gate radius "  # v36g W3
            "%s, the blend 1 -> 0 over [R_* - ell, R_* + ell]; ") % (a.completion, a.completion_gate_value, a.ell, a.completion_from_T_over_rs,  # v36g W3
                                                                      ("%g r_s (the option)" % a.completion_radius_over_rs) if a.completion_radius_over_rs is not None else "the arming record's own, rounded up to the next multiple of ell")  # v36g W3
    if a.completion == "frozen-sources":  # v36g W3
        return base + ("the sources frozen inside at the gated record's values, the charge projected by one scalar lambda per field solve so that the collar charge equals %s (tolerance %g)" % (  # v36g W3
            {"matched": "the stored member's E at the same clock", "gated": "the gated record's E", "none": "nothing (lambda = 1)"}[a.completion_charge], a.completion_charge_tol))  # v36g W3
    if a.completion == "paired-avg":  # v36g W3
        return base + ("the paired averaging widened to %g a inside (the shell mixing of width sqrt(%g^2 - %d^2) a on the sources before the ell = %d a solve, its transpose on the back-averaged fields: adjoint-paired)" % (  # v36g W3
            a.completion_ell_in, a.completion_ell_in, a.ell, a.ell))  # v36g W3
    return base + "the core coupling point-sampled inside (the dense fields at the sites in place of the back-averaged ones; the source averaging unchanged)"  # v36g W3


def v36g_desc(a):  # v36g W3
    """v36g W3: the active options in words.  [v36g W3]"""  # v36g W3
    parts = []  # v36g W3
    if CPL["requested"]:  # v36g W3
        parts.append(completion_desc_v36g(a))  # v36g W3
    if not PSRC["on"]:  # v36g W3
        parts.append("pair sources off: the dust-only sources (the pair terms w (dens(h, sea) - dens(h, R2)) and w (dens(Q, sea) - dens(Q, R2)) of every channel dropped from eps and J; the sea propagated, R2 continued, eps_pairs and J_pairs stored)")  # v36g W3
    return "; ".join(parts)  # v36g W3


def v36g_head(a):  # v36g W3
    """v36g W3: the json head's v36g entry: the options, the completion's parameters and, after arming, the arming record and the gate rows.  [v36g W3]"""  # v36g W3
    out = {"engine": "engine_collapse_v36g (exact replacements over engine_collapse_v36d; v36g W3)", "description": v36g_desc(a), "pair_sources": ("off" if not PSRC["on"] else "on")}  # v36g W3
    if CPL["requested"]:  # v36g W3
        out["completion"] = {"mode": CPL["mode"], "from_T_over_rs": a.completion_from_T_over_rs, "gate_value": a.completion_gate_value, "sigma": float(a.ell), "blend_width": float(a.ell),  # v36g W3
                             "radius_over_rs_option": a.completion_radius_over_rs, "ell_in": (a.completion_ell_in if CPL["mode"] == "paired-avg" else None),  # v36g W3
                             "charge": (a.completion_charge if CPL["mode"] == "frozen-sources" else None), "charge_tol": a.completion_charge_tol, "stored_member": (a.completion_run or a.resume_run),  # v36g W3
                             "armed": CPL["armed"], "arming": CPL["arming"], "gate_rows": CPL["gate_rows"]}  # v36g W3
    return out  # v36g W3


def cpl_gate_record_v36g(a, info, R, rs, T, eps, J, E, N, v, head):  # v36g W3
    """v36g W3: the gate test on a record solved as the record's own while no completion is armed: the record's gate profile is kept in CPL["gate_rows"]; a gated  [v36g W3]
    record stores its accepted sources and charge as the latest gated record's; the first ungated record at or after --completion-from-T-over-rs arms the  [v36g W3]
    completion (R_* the option's or the record's own gate radius rounded up to the next multiple of ell; the blend; the mixing for paired-avg; the gated  [v36g W3]
    record's sources, charge and clock for frozen-sources, with the stored member's E(T) for the matched projection) and returns True: the caller re-solves  [v36g W3]
    the record with it. Returns False otherwise.  [v36g W3]"""  # v36g W3
    if CPL["armed"]:  # v36g W3
        return False  # v36g W3
    prof = gate_profile_v36g(info, a.ell, a.completion_gate_value)  # v36g W3
    tz = FLD.trapping_radius(N, v, R, info)  # v36g W3
    row = {"T_over_rs": T / rs, "E": float(E), "G_max": prof["G_max"], "r_at_G_max": prof["r_at_max"], "dominant_at_max": prof["dominant_at_max"],  # v36g W3
           "gate_radius": prof["gate_radius"], "failing_intervals": prof["failing_intervals"], "gated": prof["gated"], "trapping_zeros": tz["zeros"], "min_N_minus_absv": tz["min"]}  # v36g W3
    CPL["gate_rows"].append(row)  # v36g W3
    print("T/rs %.4f v36g gate on the record's own solve: sigma max(|v'/N|, |v/(r N)|) max %.4f at r %.2f a (%s), failing intervals %s, gate radius %.2f a (%.4f r_s): %s" % (  # v36g W3
        T / rs, prof["G_max"], prof["r_at_max"], prof["dominant_at_max"], ["[%.2f, %.2f]" % (x_, y_) for x_, y_ in prof["failing_intervals"]], prof["gate_radius"], prof["gate_radius"] / rs,  # v36g W3
        "gated" if prof["gated"] else "NOT gated"), flush=True)  # v36g W3
    if prof["gated"]:  # v36g W3
        CPL["last_gated"] = {"T": float(T), "eps": np.array(eps, dtype=float, copy=True), "J": np.array(J, dtype=float, copy=True), "E": float(E)}  # v36g W3
        return False  # v36g W3
    if T / rs < a.completion_from_T_over_rs - 1e-9:  # v36g W3
        print("  v36g: before --completion-from-T-over-rs %g: the record's own description continues" % a.completion_from_T_over_rs, flush=True)  # v36g W3
        return False  # v36g W3
    lg = CPL["last_gated"]  # v36g W3
    if lg is None and CPL["mode"] == "frozen-sources":  # v36g W3
        raise SystemExit("v36g W3: the first record this run solved (T/rs %.4f) fails the gate: frozen-sources needs the gated record's sources from a record this run solved; resume from an earlier slice" % (T / rs))  # v36g W3
    ell = float(a.ell)  # v36g W3
    R_star = (float(a.completion_radius_over_rs) * rs) if a.completion_radius_over_rs is not None else (math.ceil(prof["gate_radius"] / ell - 1e-9) * ell)  # v36g W3
    beta = blend_v36g(R, R_star, ell)  # v36g W3
    CPL.update({"armed": True, "T_arm": float(T), "R_star": float(R_star), "R_star_auto": prof["gate_radius"], "beta": beta, "lambda": 1.0})  # v36g W3
    CPL["arming_own"] = {"eps": np.array(eps, dtype=float, copy=True), "J": np.array(J, dtype=float, copy=True), "N": np.array(N, dtype=float, copy=True), "v": np.array(v, dtype=float, copy=True), "E": float(E)}  # v36g W3
    if lg is not None:  # v36g W3
        CPL.update({"frozen_eps": lg["eps"], "frozen_J": lg["J"], "E_gated": lg["E"], "T_gated": lg["T"]})  # v36g W3
    arming = {"T_arm_over_rs": T / rs, "arming_record_own_solve": row, "R_star": float(R_star), "R_star_over_rs": float(R_star) / rs, "R_star_auto": prof["gate_radius"],  # v36g W3
              "R_star_rule": ("the option's %g r_s" % a.completion_radius_over_rs) if a.completion_radius_over_rs is not None else "the arming record's own gate radius rounded up to the next multiple of ell",  # v36g W3
              "T_gated_over_rs": (lg["T"] / rs if lg is not None else None), "E_gated": (lg["E"] if lg is not None else None),  # v36g W3
              "sites_beta_1": int(np.sum(beta >= 1.0)), "sites_blended": int(np.sum((beta > 0.0) & (beta < 1.0)))}  # v36g W3
    if CPL["mode"] == "paired-avg":  # v36g W3
        CPL["mix"] = mixing_v36g(R, beta, a.completion_ell_in, ell)  # v36g W3
        arming["mixing_identities"] = mix_identities_v36g(CPL["mix"], R); arming["composition_check"] = composition_check_v36g(CPL["mix"], R, R_star)  # v36g W3
    if CPL["mode"] == "frozen-sources" and CPL["charge_mode"] == "matched":  # v36g W3
        CPL["E_stored"] = load_stored_E_v36g(a.completion_run or a.resume_run, rs)  # v36g W3
        arming["stored_E_records"] = len(CPL["E_stored"])  # v36g W3
    CPL["arming"] = arming  # v36g W3
    head["v36g"] = v36g_head(a)  # v36g W3
    mi = arming.get("mixing_identities"); cc = arming.get("composition_check") or {}  # v36g W3
    print("T/rs %.4f v36g: ARMED %s: gate radius R_* %.2f a (%.4f r_s; the record's own %.2f a; %s); gated record T/rs %s (E %s); beta = 1 on %d sites, blended on %d%s%s; the record is re-solved with the completion" % (  # v36g W3
        T / rs, CPL["mode"], R_star, R_star / rs, prof["gate_radius"], arming["R_star_rule"], ("%.4f" % (lg["T"] / rs)) if lg is not None else "none", ("%.10f" % lg["E"]) if lg is not None else "none",  # v36g W3
        arming["sites_beta_1"], arming["sites_blended"],  # v36g W3
        ("; mixing: charge sum %.1e, K0 on 1 %.1e, K1 on c r %.1e, adjoint %.1e %.1e, composition K0 %.2e K1 %.2e over %d sites (ell_x %.3f a)" % (  # v36g W3
            mi["charge_sum_rel"], mi["K0_constant_max_abs"], mi["K1_linear_max_rel"], mi["adjoint_K0_rel"], mi["adjoint_K1_rel"], cc.get("K0_rel_dev", float("nan")),  # v36g W3
            cc.get("K1_rel_dev", float("nan")), mi["mixed_sites"], mi["ell_x"])) if mi is not None else "",  # v36g W3
        ("; the stored member's E(T) at %d records for the matched projection" % arming["stored_E_records"]) if "stored_E_records" in arming else ""), flush=True)  # v36g W3
    return True  # v36g W3


def record_entry_v36g(a, info, T, rs, eps_pairs, J_pairs):  # v36g W3
    """v36g W3: the record's v36g entry: the completion's name and parameters (armed: its solve's lambda, target and residual, the record's own gate profile; the  [v36g W3]
    sources the solver saw appended to the store) and, with the pair sources off, the dropped pair totals.  [v36g W3]"""  # v36g W3
    ent = {"engine": "engine_collapse_v36g (exact replacements over engine_collapse_v36d; v36g W3)"}  # v36g W3
    if not PSRC["on"]:  # v36g W3
        ent["pair_sources"] = {"mode": "off", "dropped_terms": "w (dens(h, sea) - dens(h, R2)) from eps and w (dens(Q, sea) - dens(Q, R2)) from J, every channel",  # v36g W3
                               "dropped_eps_pairs_sum": float(np.sum(eps_pairs)), "dropped_J_pairs_sum": float(np.sum(J_pairs)),  # v36g W3
                               "reference_subtraction": "kept: R2 continued, the pair densities computed against it and stored as eps_pairs, J_pairs"}  # v36g W3
    if CPL["requested"]:  # v36g W3
        c_ = {"mode": CPL["mode"], "armed": CPL["armed"], "T_arm_over_rs": (CPL["T_arm"] / rs if CPL["T_arm"] is not None else None),  # v36g W3
              "T_gated_over_rs": (CPL["T_gated"] / rs if CPL["T_gated"] is not None else None), "E_gated": CPL["E_gated"], "R_star": CPL["R_star"],  # v36g W3
              "R_star_over_rs": (CPL["R_star"] / rs if CPL["R_star"] is not None else None), "gate_value": a.completion_gate_value, "sigma": float(a.ell), "blend_width": float(a.ell),  # v36g W3
              "ell_in": (a.completion_ell_in if CPL["mode"] == "paired-avg" else None), "charge": (a.completion_charge if CPL["mode"] == "frozen-sources" else None)}  # v36g W3
        if CPL["armed"]:  # v36g W3
            c_["solve"] = dict(info.get("v36g") or {})  # v36g W3
            prof = gate_profile_v36g(info, a.ell, a.completion_gate_value)  # v36g W3
            c_["record_gate"] = {"G_max": prof["G_max"], "r_at_G_max": prof["r_at_max"], "dominant": prof["dominant_at_max"], "gate_radius": prof["gate_radius"], "gated": prof["gated"], "failing_intervals": prof["failing_intervals"]}  # v36g W3
            es, Js = info.get("v36g_sources", (None, None))  # v36g W3
            if es is not None:  # v36g W3
                sv = info.get("v36g") or {}; st_ = CPL["store"]  # v36g W3
                st_["T"].append(float(T)); st_["eps_solver"].append(np.asarray(es, float)); st_["J_solver"].append(np.asarray(Js, float))  # v36g W3
                st_["lambda"].append(float(sv.get("lambda", 1.0))); st_["E_target"].append(float(sv["E_target"]) if sv.get("E_target") is not None else float("nan")); st_["E"].append(float(FLD.charge(info, a.g)))  # v36g W3
        ent["completion"] = c_  # v36g W3
    return ent  # v36g W3


def record_line_v36g(ent):  # v36g W3
    """v36g W3: the printed line of a record's v36g entry (after the record line).  [v36g W3]"""  # v36g W3
    parts = []  # v36g W3
    if "pair_sources" in ent:  # v36g W3
        p_ = ent["pair_sources"]  # v36g W3
        parts.append("v36g pair sources OFF (the dust-only sources): the dropped pair source totals eps %+.6e J %+.6e (eps_pairs, J_pairs stored; R2 continued)" % (p_["dropped_eps_pairs_sum"], p_["dropped_J_pairs_sum"]))  # v36g W3
    if "completion" in ent:  # v36g W3
        c_ = ent["completion"]  # v36g W3
        if not c_["armed"]:  # v36g W3
            parts.append("v36g completion %s not yet armed (the record's own description)" % c_["mode"])  # v36g W3
        else:  # v36g W3
            sv = c_.get("solve") or {}; rg = c_.get("record_gate") or {}  # v36g W3
            s_ = "v36g completion %s armed at T/rs %.4f (gated record T/rs %s, E %s), gate radius %.2f a (%.4f r_s)" % (  # v36g W3
                c_["mode"], c_["T_arm_over_rs"], ("%.4f" % c_["T_gated_over_rs"]) if c_["T_gated_over_rs"] is not None else "-", ("%.6f" % c_["E_gated"]) if c_["E_gated"] is not None else "-", c_["R_star"], c_["R_star_over_rs"])  # v36g W3
            if c_["mode"] == "frozen-sources":  # v36g W3
                s_ += "; lambda %.8f, E %s against the target %s (%s), residual %s in %s solves%s" % (  # v36g W3
                    sv.get("lambda", float("nan")), fmt(sv.get("E"), "%.8f"), fmt(sv.get("E_target"), "%.8f"), sv.get("E_target_source"), fmt(sv.get("charge_residual_rel"), "%.1e"), sv.get("charge_solves"), "" if sv.get("charge_converged") else " NOT CONVERGED")  # v36g W3
            s_ += "; this record's gate: G max %.4f at r %.2f a (%s), gate radius %.2f a%s" % (rg.get("G_max", float("nan")), rg.get("r_at_G_max", float("nan")), rg.get("dominant", "-"), rg.get("gate_radius", float("nan")), " (gated)" if rg.get("gated") else "")  # v36g W3
            parts.append(s_)  # v36g W3
    return "; ".join(parts)  # v36g W3


def cpl_store_arrays_v36g(a, R):  # v36g W3
    """v36g W3: the arrays of collapse_<tag>_v36g.npz: per armed record the site sources the solver saw, lambda, the target and the solved charge; the blend, the  [v36g W3]
    gate radius, the gated record's frozen sources and the arming record's own (discarded) solve.  [v36g W3]"""  # v36g W3
    st_ = CPL["store"]; ao = CPL["arming_own"] or {}  # v36g W3
    nan = float("nan")  # v36g W3
    return {"T": np.array(st_["T"], float), "eps_solver": np.array(st_["eps_solver"]), "J_solver": np.array(st_["J_solver"]), "lambda": np.array(st_["lambda"], float),  # v36g W3
            "E_target": np.array(st_["E_target"], float), "E": np.array(st_["E"], float), "R": np.asarray(R, float), "beta": (CPL["beta"] if CPL["beta"] is not None else np.zeros(0)),  # v36g W3
            "R_star": float(CPL["R_star"] if CPL["R_star"] is not None else nan), "T_arm": float(CPL["T_arm"] if CPL["T_arm"] is not None else nan), "mode": str(CPL["mode"]),  # v36g W3
            "E_gated": float(CPL["E_gated"] if CPL["E_gated"] is not None else nan), "T_gated": float(CPL["T_gated"] if CPL["T_gated"] is not None else nan),  # v36g W3
            "frozen_eps": (CPL["frozen_eps"] if CPL["frozen_eps"] is not None else np.zeros(0)), "frozen_J": (CPL["frozen_J"] if CPL["frozen_J"] is not None else np.zeros(0)),  # v36g W3
            "arming_own_eps": ao.get("eps", np.zeros(0)), "arming_own_J": ao.get("J", np.zeros(0)), "arming_own_N": ao.get("N", np.zeros(0)), "arming_own_v": ao.get("v", np.zeros(0)), "arming_own_E": float(ao.get("E", nan)),  # v36g W3
            "note": "eps_solver, J_solver: the site sources the field solver saw at each armed record (the fields npz keeps the raw accepted sources); the arming record's own solve is the discarded pre-arming solve"}  # v36g W3


def resume_args_relaxed_v36g(a, stored_args, key):  # v36g W3
    """v36g W3: --resume-args-check relaxed: a key absent from the stored run's args counts as matching when this call's value is the argument's default (printed);  [v36g W3]
    every other difference still stops the run as version 34 (a) does. Returns True when the difference is accepted.  [v36g W3]"""  # v36g W3
    if getattr(a, "resume_args_check", "strict") != "relaxed" or key in stored_args:  # v36g W3
        return False  # v36g W3
    defaults = {"c_LZ": 1.0, "mix": "plain", "massless_ball": False, "fields_v17": False}  # v36g W3
    ok = key in defaults and getattr(a, key) == defaults[key]  # v36g W3
    print("v36g W3: --resume-args-check relaxed: the stored run's args lack %r; this call's %r %s" % (key, getattr(a, key), "is the default: accepted" if ok else "is not the default: refused"), flush=True)  # v36g W3
    return ok  # v36g W3


def slice_dtype_v36g(a, Z):  # v36g W3
    """v36g W3: the slice's stored dtype, or (--resume-args-check relaxed) the dtype of its first sea array when the slice has no dtype entry (written before version 34).  [v36g W3]"""  # v36g W3
    if "dtype" in Z.files:  # v36g W3
        return str(Z["dtype"])  # v36g W3
    if getattr(a, "resume_args_check", "strict") != "relaxed":  # v36g W3
        raise SystemExit("version 34 (a): the slice %s has no dtype entry (written before version 34); --resume-args-check relaxed reads the arrays' dtype" % a.resume_slice)  # v36g W3
    keys = [k_ for k_ in Z.files if k_.startswith("sea_")]  # v36g W3
    return str(Z[keys[0]].dtype) if keys else "unknown"  # v36g W3


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)'''),
("resume args check",
r'''    for k_ in ("m", "g", "rs", "nsites", "ell", "kappa_sea", "tau", "reference", "site_fields", "c_LZ", "mix", "massless_ball", "fields_v17"):
        if J["args"].get(k_) != getattr(a, k_):
            raise SystemExit("version 34 (a): the run's argument %s is %r, this call's %r" % (k_, J["args"].get(k_), getattr(a, k_)))''',
r'''    for k_ in ("m", "g", "rs", "nsites", "ell", "kappa_sea", "tau", "reference", "site_fields", "c_LZ", "mix", "massless_ball", "fields_v17"):
        if J["args"].get(k_) != getattr(a, k_):
            if resume_args_relaxed_v36g(a, J["args"], k_):  # v36g W3
                continue  # v36g W3
            raise SystemExit("version 34 (a): the run's argument %s is %r, this call's %r" % (k_, J["args"].get(k_), getattr(a, k_)))'''),
("slice dtype",
r'''    if str(Z["dtype"]) != "complex128":''',
r'''    if slice_dtype_v36g(a, Z) != "complex128":  # v36g W3'''),
("run label",
r'''    rs = a.rs
    NB = rs / (2.0 * a.g * a.m) if a.m > 0 else None''',
r'''    if CPL["requested"] or not PSRC["on"]:  # v36g W3
        label += "; v36g W3: " + v36g_desc(a)  # v36g W3
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m) if a.m > 0 else None'''),
("head entry",
r'''    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}''',
r'''    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    if CPL["requested"] or not PSRC["on"]:  # v36g W3
        head["v36g"] = v36g_head(a)  # v36g W3'''),
("save fields",
r'''    def save_fields():
        if s11_store:''',
r'''    def save_fields():
        if CPL["store"]["T"]:  # v36g W3
            np.savez_compressed(os.path.join(a.out, "collapse_%s_v36g.npz" % a.tag), **cpl_store_arrays_v36g(a, R))  # v36g W3
        if s11_store:'''),
("loop header",
r'''    for step in range(step0, nsteps + 1):''',
r'''    for step in steps_v36g(step0, nsteps):  # v36g W3'''),
("step top",
r'''        tr = time.time()
        tm_reset()
        TM["propagation_prev_s"] = last_prop_s''',
r'''        tr = time.time()
        tm_reset()
        TM["propagation_prev_s"] = last_prop_s
        unconv36g_prev = unconv_state["first"]  # v36g W3'''),
("sources accumulation",
r'''                hs_, Qs_, dd, Jd = base[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js''',
r'''                hs_, Qs_, dd, Jd = base[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                accumulate_sources_v36g(ch.w, ds, Js, dd, Jd, eps, J, eps_pairs, eps_dust, J_pairs)  # v36g W3 (W4b)'''),
("sources_with accumulation",
r'''                hs_, Qs_, dd, Jd = base_[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js''',
r'''                hs_, Qs_, dd, Jd = base_[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                accumulate_sources_v36g(ch.w, ds, Js, dd, Jd, eps, J, eps_pairs, eps_dust, J_pairs)  # v36g W3 (W4b)'''),
("do_solve",
r'''        def do_solve(srcs_):
            t_s = time.time()
            out = solve(srcs_[0], srcs_[1], a.g, R, a.ell, None)''',
r'''        def do_solve(srcs_):
            t_s = time.time()
            out = cpl_solve_v36g(srcs_[0], srcs_[1], a.g, R, a.ell, T, rs)  # v36g W3'''),
("gate test and arming",
r'''        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)''',
r'''        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)
        if CPL["requested"] and cpl_gate_record_v36g(a, info, R, rs, T, eps, J, E, N, v, head):  # v36g W3
            unconv_state["first"] = unconv36g_prev  # v36g W3
            CPL["redo"] = True  # v36g W3
            continue  # v36g W3'''),
("diagnostics on the solver's sources",
r'''        fresh = bool(step % max(1, a.delta_rtol_every) == 0 or drtol_last["value"] is None)
        ind = indicator(eps, J, a.g, R, a.ell, info, N, v, fresh, drtol_last)
        if fresh and ind["delta_rtol"] is not None:
            drtol_last.update({"value": ind["delta_rtol"], "T_over_rs": T / rs})
        r0c = FLD.r0_check(eps, J, a.g, R, a.ell, info, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"]) if fresh else None''',
r'''        fresh = bool(step % max(1, a.delta_rtol_every) == 0 or drtol_last["value"] is None)
        eps_s36g, J_s36g = info.get("v36g_sources", (eps, J))  # v36g W3
        ind = indicator(eps_s36g, J_s36g, a.g, R, a.ell, info, N, v, fresh, drtol_last)  # v36g W3
        if fresh and ind["delta_rtol"] is not None:
            drtol_last.update({"value": ind["delta_rtol"], "T_over_rs": T / rs})
        r0c = FLD.r0_check(eps_s36g, J_s36g, a.g, R, a.ell, info, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"]) if fresh else None  # v36g W3'''),
("record entry",
r'''        if V36["fixes"] or V36["kcache"]:
            rec["v36"] = {"fixes": V36["fixes"], "gap_none_substeps": V36["gap_none_substeps"], "kernel_cache": dict(V36["kcache"]) if V36["kcache"] else None}''',
r'''        if V36["fixes"] or V36["kcache"]:
            rec["v36"] = {"fixes": V36["fixes"], "gap_none_substeps": V36["gap_none_substeps"], "kernel_cache": dict(V36["kcache"]) if V36["kcache"] else None}
        if CPL["requested"] or not PSRC["on"]:  # v36g W3
            rec["v36g"] = record_entry_v36g(a, info, T, rs, eps_pairs, J_pairs)  # v36g W3'''),
("record line",
r'''        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)''',
r'''        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if "v36g" in rec:  # v36g W3
            print("T/rs %.4f %s" % (T / rs, record_line_v36g(rec["v36g"])), flush=True)  # v36g W3'''),
("arguments",
r'''    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:''',
r'''    ap.add_argument("--completion", dest="completion", default=None, choices=["frozen-sources", "paired-avg", "point-core"], help="(v36g W3) the interior completion inside the gate radius from the first record failing the hydrodynamic gate at or after --completion-from-T-over-rs")  # v36g W3
    ap.add_argument("--completion-from-T-over-rs", dest="completion_from_T_over_rs", type=float, default=0.0, help="(v36g W3) records below this clock stay the record's own even where the gate fails (the last gated record's clock, 11.625 on the (0.2, 64) sigma/a 4 tau 8 member if the gate arrays confirm it)")  # v36g W3
    ap.add_argument("--completion-radius-over-rs", dest="completion_radius_over_rs", type=float, default=None, help="(v36g W3) the gate radius in r_s; absent: the arming record's own outermost radius with sigma max(|v'/N|, |v/(r N)|) > the gate value, rounded up to the next multiple of ell")  # v36g W3
    ap.add_argument("--completion-gate-value", dest="completion_gate_value", type=float, default=0.1, help="(v36g W3) the hydrodynamic gate's working value")  # v36g W3
    ap.add_argument("--completion-ell-in", dest="completion_ell_in", type=float, default=8.0, help="(v36g W3, paired-avg) the averaging scale inside the gate radius in a (the run's --ell outside)")  # v36g W3
    ap.add_argument("--completion-charge", dest="completion_charge", default="matched", choices=["matched", "gated", "none"], help="(v36g W3, frozen-sources) the charge projection's target: the stored member's E at the same clock, the gated record's E, or no rescaling")  # v36g W3
    ap.add_argument("--completion-run", dest="completion_run", default=None, help="(v36g W3) DIR/TAG of the stored member whose E(T) the matched projection reads (default --resume-run)")  # v36g W3
    ap.add_argument("--completion-charge-tol", dest="completion_charge_tol", type=float, default=1e-9, help="(v36g W3) the projection's relative tolerance on the collar charge")  # v36g W3
    ap.add_argument("--pair-sources", dest="pair_sources", default="on", choices=["on", "off"], help="(v36g W3, W4 (b)) off: the pair terms w (dens(h, sea) - dens(h, R2)) and w (dens(Q, sea) - dens(Q, R2)) of every channel dropped from eps and J (computed and stored as eps_pairs, J_pairs); the sea propagated, R2 continued")  # v36g W3 (W4b)
    ap.add_argument("--resume-args-check", dest="resume_args_check", default="strict", choices=["strict", "relaxed"], help="(v36g W3) relaxed: a key absent from the stored run's args matches when this call's value is the default; a slice without a dtype entry read by its arrays' dtype")  # v36g W3
    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:'''),
("validation",
r'''    C5.backend(a.gpu)
    run(a)''',
r'''    PSRC["on"] = (a.pair_sources != "off")  # v36g W3 (W4b)
    CPL["requested"] = a.completion is not None  # v36g W3
    if CPL["requested"]:  # v36g W3
        CPL.update({"mode": a.completion, "charge_mode": a.completion_charge, "charge_tol": float(a.completion_charge_tol)})  # v36g W3
        if a.pair_workers:  # v36g W3
            ap.error("v36g W3: --completion acts inside the parent's field solves; --pair-workers runs solves in worker processes that would not see it")  # v36g W3
        if a.massless_ball:  # v36g W3
            ap.error("v36g W3: the completion control is defined for the dust collapse (--massless-ball excluded)")  # v36g W3
        if a.completion in ("paired-avg", "point-core") and a.site_fields != "averaged":  # v36g W3
            ap.error("v36g W3: --completion %s needs --site-fields averaged (the back-averaged fields it widens or replaces)" % a.completion)  # v36g W3
        if a.completion == "paired-avg" and not a.completion_ell_in > a.ell:  # v36g W3
            ap.error("v36g W3: --completion-ell-in must exceed --ell (the widening inside the gate radius)")  # v36g W3
        if a.completion == "frozen-sources" and a.completion_charge == "matched" and not (a.completion_run or a.resume_run):  # v36g W3
            ap.error("v36g W3: --completion-charge matched reads the stored member's E(T): give --completion-run DIR/TAG or --resume-run")  # v36g W3
        if a.completion_radius_over_rs is not None and not a.completion_radius_over_rs > 0.0:  # v36g W3
            ap.error("v36g W3: --completion-radius-over-rs must be positive")  # v36g W3
    C5.backend(a.gpu)
    run(a)'''),
]

MARK = "v36g W3"


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        if MARK in old:
            raise SystemExit("build_engine_collapse_v36g: the OLD string of '%s' carries the mark (it must be v36d's text)" % name)
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36g: the OLD string of '%s' occurs %d times in the source (1 required): the source is not input 05's engine_collapse_v36d.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36g: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def unmarked_added_lines(src_text, out_text):
    """every line the replacements add or change (the unified diff's '+' lines, no context) must carry the mark; returns the offenders."""
    bad = []
    for ln in difflib.unified_diff(src_text.splitlines(), out_text.splitlines(), lineterm="", n=0):
        if ln.startswith("+") and not ln.startswith("+++") and MARK not in ln and ln[1:].strip():   # busi 2026-10-18 (analyst): blank added lines skipped (they cannot carry a mark), as the C64 red team ruled for v8
            bad.append(ln)
    return bad


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36d.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36g.py")
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36g: the reverse replacements do not restore the source (a NEW text overlaps another)")
    bad = unmarked_added_lines(text, out)
    if bad:
        raise SystemExit("build_engine_collapse_v36g: %d added lines lack the mark '%s'; the first: %s" % (len(bad), MARK, bad[0][:160]))
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    n_added = sum(1 for ln in difflib.unified_diff(text.splitlines(), out.splitlines(), lineterm="", n=0) if ln.startswith("+") and not ln.startswith("+++"))
    print("built %s from %s: %d replacements, %d added or changed lines, every one marked '%s'; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), n_added, MARK, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
