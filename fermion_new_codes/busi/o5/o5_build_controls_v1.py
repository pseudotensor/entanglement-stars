#!/usr/bin/env python
"""o5_build_controls_v1.py -- RUN_O5_controls_v1: builds o5_lift_v3.py and o5_compare_v3.py from o5_lift_v2.py and o5_compare_v2.py by exact
textual replacements. Every anchor must occur exactly once in the source file (otherwise the source is not the attached one and the build stops);
the v2 files are never edited; each built file is compiled before it is written; the md5 of every source and built file is printed.

What the replacements add.
  o5_lift_v3.py: --lift_convention {default, isometric} (default: o5_lift_v2's lift, every path unchanged) and --iso_min_ratio. The isometric
    lift (control 4's second convention; the cube side of the native shell's check): (i) in every shell j >= 1, the chain's midpoint cell
    [j - 1/2, j + 1/2) on the cube (j = rint(r) of the site), the sampled angular pair (Oa, Ob) is replaced by (Oa, Ob) S_j^(-1/2) sqrt(tr S_j / 2),
    S_j the pair's raw 2 x 2 Gram over the shell's sites (plain site sums over all components), so that the pair is orthonormal within the shell up
    to one scalar; shells whose Gram eigenvalue ratio lies below --iso_min_ratio are left as sampled and listed; (ii) after the lift every shell
    j >= 1 is rescaled so that its weight equals the chain's cell weight p_chain[j - 1] (the midpoint rule), the origin shell is left as lifted, a
    shell with lifted weight where the chain has none is set to 0; then the global renormalization as before. The default lift's statistic and
    support maximum stay in the json beside the written vector's (gram_defect_default, gram_dev_max_support_default); the convention and the
    isometric record are written into the src and det npz meta and the json (lift_convention, isometric).
  o5_compare_v3.py: tolerates a cube json whose directional root r_h is null (constant fields, the cut given by --o5_rh) and adds --lift_term
    {gram, none}: 'none' sets the lift's Gram defect, amplitude lift error and detector Gram terms to 0 for a source built natively on both codes
    (no lift), so the error sum is the 2 propagation bands; the option is recorded in the json's options and the md's header.
Usage: python o5_build_controls_v1.py --lift_v2 o5_lift_v2.py --compare_v2 o5_compare_v2.py --out .
"""
import argparse
import hashlib
import os
import sys


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


# ------------------------------------------------------------------ o5_lift_v3.py
LIFT_HELPERS = r'''def _shell_index(r):
    """The chain's midpoint cells on the cube: shell j = rint(r) is the cell [j - 1/2, j + 1/2) of the chain site r = j a; the origin is shell 0."""
    return np.rint(np.asarray(r, dtype=float)).astype(int)


def _site_axis(x, n):
    ax = [k for k, d in enumerate(np.shape(x)) if d == n]
    if not ax:
        raise ValueError(f"no axis of length {n} (the sites) in an array of shape {np.shape(x)}")
    return ax[0]


def isometric_pair(r, Oa, Ob, min_ratio):
    """Control 4's isometric lift, step (i): per shell j >= 1 the sampled pair (Oa, Ob) becomes (Oa, Ob) M_j, M_j = S_j^(-1/2) sqrt(tr S_j / 2),
    S_j the pair's raw Gram over the shell's sites (plain site sums over all components); the corrected pair is orthonormal within the shell up
    to one scalar, the shell's raw size kept for step (ii). Shells with Gram eigenvalue ratio below min_ratio are left as sampled and listed."""
    n = np.asarray(r).size
    j = _shell_index(r)
    axa, axb = _site_axis(Oa, n), _site_axis(Ob, n)
    A0 = np.moveaxis(np.array(Oa, dtype=np.complex128), axa, 0)
    B0 = np.moveaxis(np.array(Ob, dtype=np.complex128), axb, 0)
    A1, B1 = A0.copy(), B0.copy()
    done, skipped, mix = 0, [], 0.0
    for jj in np.unique(j):
        if jj < 1:
            continue
        mk = j == jj
        A, B = A0[mk], B0[mk]
        saa = float(np.sum(np.abs(A) ** 2))
        sbb = float(np.sum(np.abs(B) ** 2))
        sab = complex(np.sum(np.conj(A) * B))
        S = np.array([[saa, sab], [np.conj(sab), sbb]], dtype=np.complex128)
        w, U = np.linalg.eigh(S)
        if w[1] <= 0.0 or w[0] <= 0.0 or w[0] / w[1] < min_ratio:
            skipped.append([int(jj), float(w[0] / w[1]) if w[1] > 0.0 else 0.0, int(np.count_nonzero(mk))])
            continue
        M = (U @ np.diag(w ** -0.5) @ U.conj().T) * math.sqrt(0.5 * float(np.trace(S).real))
        A1[mk] = M[0, 0] * A + M[1, 0] * B
        B1[mk] = M[0, 1] * A + M[1, 1] * B
        done += 1
        mix = max(mix, float(abs(M[1, 0])))
    return (np.moveaxis(A1, 0, axa), np.moveaxis(B1, 0, axb),
            dict(shells_corrected=done, shells_skipped=skipped, max_offdiagonal_of_M=mix, min_ratio=float(min_ratio),
                 shell_rule="j = rint(r): the chain's midpoint cell; S_j the plain site sum over the shell; M_j = S_j^(-1/2) sqrt(tr S_j / 2)"))


def isometric_weights(r, vals, p_chain, info):
    """Step (ii): every shell j >= 1 is rescaled so that its weight equals the chain's cell weight p_chain[j - 1]; the origin shell is left as
    lifted; a shell with lifted weight where the chain has none is set to 0 (the weight removed is recorded)."""
    n = np.asarray(r).size
    j = _shell_index(r)
    p = np.asarray(p_chain, dtype=float)
    out = np.array(vals, dtype=np.complex128)
    v = np.moveaxis(out, _site_axis(out, n), 0)
    dens = (np.abs(v) ** 2).reshape(v.shape[0], -1).sum(1)
    scale_max, removed, scaled = 0.0, 0.0, 0
    for jj in np.unique(j):
        if jj < 1:
            continue
        mk = j == jj
        W = float(dens[mk].sum())
        if W <= 0.0:
            continue
        tgt = float(p[jj - 1]) if jj - 1 < p.size else 0.0
        if tgt <= 0.0:
            removed += W
            v[mk] = 0.0
            continue
        s = math.sqrt(tgt / W)
        v[mk] = v[mk] * s
        scale_max = max(scale_max, abs(s - 1.0))
        scaled += 1
    info = dict(info or {}, shells_weight_matched=scaled, max_abs_scale_minus_1=scale_max, weight_removed_where_chain_has_none=removed,
                origin_weight_left=float(dens[j < 1].sum()) if np.any(j < 1) else 0.0)
    return out, info


'''

LIFT_DOC_OLD = "\"\"\"o5_lift_v2.py -- O5 (GAP10_T7_final_v1.1 section 9): P21's lift of the chain column to the cubic lattice, reading the column's declared"
LIFT_DOC_NEW = ("\"\"\"o5_lift_v3.py -- RUN_O5_controls_v1: o5_lift_v2.py (never edited) with --lift_convention {default, isometric}, built by o5_build_controls_v1.py by\n"
                "exact replacements; with the default convention every path is o5_lift_v2's. o5_lift_v2's docstring: P21's lift of the chain column to the cubic lattice, "
                "reading the column's declared")

LIFT_SIG_OLD = "def lift_profile_column(prof, u, H, kappa_H, mj, lower_sign, kappa_assign, p_chain=None, support_floor=1e-6, pair=None, support_eps=1e-10):"
LIFT_SIG_NEW = LIFT_HELPERS + ("def lift_profile_column(prof, u, H, kappa_H, mj, lower_sign, kappa_assign, p_chain=None, support_floor=1e-6, pair=None, support_eps=1e-10,\n"
                               "                        convention='default', min_ratio=0.25):")

LIFT_BODY_OLD = "\n".join([
    "    Oa, Ob = L1.angular_pair(coords, r, kappa_H, mj, kappa_assign) if pair is None else pair",
    "    vals = L1.lift_vector(r, f_r, u, Oa, Ob, lower_sign)",
    "    W = float(np.sum(np.abs(vals) ** 2))",
    "    gram = L1.shell_gram_dev(r, Oa, Ob)",
    "    out = dict(coords=coords, r=r, values=vals / math.sqrt(W), values_raw=vals, W_lift=W, gram=gram, R_lo=R_lo, R_hi=R_hi, support_half_width=half,",
    "               box_clipped=bool(R_hi > H), n_sites=int(coords.shape[0]), Oa=Oa, Ob=Ob, f_r=f_r, profile=prof)",
])
LIFT_BODY_NEW = "\n".join([
    "    Oa, Ob = L1.angular_pair(coords, r, kappa_H, mj, kappa_assign) if pair is None else pair",
    "    Oa0, Ob0 = Oa, Ob",
    "    gram0 = L1.shell_gram_dev(r, Oa0, Ob0)",
    "    iso = None",
    "    if convention == 'isometric':",
    "        if p_chain is None:",
    "            raise ValueError(\"the isometric lift needs the chain's cell weights (p_chain)\")",
    "        Oa, Ob, iso = isometric_pair(r, Oa0, Ob0, min_ratio)",
    "    elif convention != 'default':",
    "        raise ValueError(f\"unknown lift convention {convention}\")",
    "    vals = L1.lift_vector(r, f_r, u, Oa, Ob, lower_sign)",
    "    if convention == 'isometric':",
    "        vals, iso = isometric_weights(r, vals, p_chain, iso)",
    "    W = float(np.sum(np.abs(vals) ** 2))",
    "    gram = L1.shell_gram_dev(r, Oa, Ob)",
    "    out = dict(coords=coords, r=r, values=vals / math.sqrt(W), values_raw=vals, W_lift=W, gram=gram, R_lo=R_lo, R_hi=R_hi, support_half_width=half,",
    "               box_clipped=bool(R_hi > H), n_sites=int(coords.shape[0]), Oa=Oa, Ob=Ob, Oa_default=Oa0, Ob_default=Ob0, gram_default=gram0, isometric=iso,",
    "               convention=convention, f_r=f_r, profile=prof)",
])

LIFT_STAT_OLD = "\n".join([
    "        out['weight_defect_bound_check'] = float(abs(W - 1.0))",
    "    return out",
])
LIFT_STAT_NEW = "\n".join([
    "        out['weight_defect_bound_check'] = float(abs(W - 1.0))",
    "        dev0 = np.ones(p.size + 1)",
    "        k0 = min(gram0['dev'].size, dev0.size)",
    "        dev0[:k0] = gram0['dev'][:k0]",
    "        out['gram_defect_default'] = float(np.sum(p * dev0[1:p.size + 1]))",
    "        out['gram_dev_max_support_default'] = float(np.max(dev0[1:p.size + 1][sup])) if np.any(sup) else float('nan')",
    "    return out",
])

LIFT_ARGS_OLD = "    ap.add_argument('--out', default='o5_lift')"
LIFT_ARGS_NEW = "\n".join([
    "    ap.add_argument('--lift_convention', choices=['default', 'isometric'], default='default', help=\"default: o5_lift_v2's lift; isometric: per shell the sampled pair orthonormalized (S_j^(-1/2) sqrt(tr S_j / 2)) and the shell weight set to the chain's cell weight (RUN_O5_controls_v1 control 4)\")",
    "    ap.add_argument('--iso_min_ratio', type=float, default=0.25, help=\"isometric: shells whose Gram eigenvalue ratio lies below this are left as sampled (listed)\")",
    "    ap.add_argument('--out', default='o5_lift')",
])

LIFT_SRC_CALL_OLD = "            lifts[f\"lower{ls_:+g}_kappa{ka:+d}\"] = lift_profile_column(prof, u, H, kappa_H, args.mj, ls_, ka, p_chain=p_chain, support_floor=args.support_floor, pair=pair, support_eps=args.support_eps)"
LIFT_SRC_CALL_NEW = ("            lifts[f\"lower{ls_:+g}_kappa{ka:+d}\"] = lift_profile_column(prof, u, H, kappa_H, args.mj, ls_, ka, p_chain=p_chain, support_floor=args.support_floor, pair=pair, support_eps=args.support_eps,\n"
                     "                                                                convention=args.lift_convention, min_ratio=args.iso_min_ratio)")

LIFT_DET_CALL_OLD = "                                pair=None if coords_d is None else det_lifts[0]['pair'], support_eps=args.support_eps)"
LIFT_DET_CALL_NEW = "                                pair=None if coords_d is None else det_lifts[0]['pair'], support_eps=args.support_eps, convention=args.lift_convention, min_ratio=args.iso_min_ratio)"

LIFT_PAIR_OLD = "            D['pair'] = (D['Oa'], D['Ob'])"
LIFT_PAIR_NEW = "            D['pair'] = (D['Oa_default'], D['Ob_default'])"

LIFT_PRINT_OLD = "    for c_ in t0_check:"
LIFT_PRINT_NEW = "\n".join([
    "    if S.get('isometric'):",
    "        iso = S['isometric']",
    "        print(f\"[isometric] convention {args.lift_convention}: {iso['shells_corrected']} shells orthonormalized (eigenvalue-ratio floor {args.iso_min_ratio:g}; skipped {iso['shells_skipped']}), \"",
    "              f\"largest off-diagonal of M {iso['max_offdiagonal_of_M']:.1e}; {iso['shells_weight_matched']} shells set to the chain's cell weights (largest |scale - 1| {iso['max_abs_scale_minus_1']:.3e}; \"",
    "              f\"weight removed where the chain has none {iso['weight_removed_where_chain_has_none']:.1e}; origin weight left {iso['origin_weight_left']:.1e}); the default lift's statistic \"",
    "              f\"{S['gram_defect_default']:.3e} (support maximum {S['gram_dev_max_support_default']:.3e}) against the written vector's {S['gram_defect']:.3e} ({S['gram_dev_max_support']:.3e})\", flush=True)",
    "    for c_ in t0_check:",
])

LIFT_SRCINFO_OLD = "t0_check=t0_check, profile=prof, chain_grid_max_diff=grid_diff)"
LIFT_SRCINFO_NEW = ("t0_check=t0_check, profile=prof, chain_grid_max_diff=grid_diff,\n"
                    "                       lift_convention=args.lift_convention, isometric=S.get('isometric'), gram_defect_default=S.get('gram_defect_default'),\n"
                    "                       gram_dev_max_support_default=S.get('gram_dev_max_support_default'))")

LIFT_SRCMETA_OLD = "    src_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=L1.CONVENTION_NOTE, chain=chain_info, source=source_info)"
LIFT_SRCMETA_NEW = "    src_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=L1.CONVENTION_NOTE, chain=chain_info, source=source_info, lift_convention=args.lift_convention)"

LIFT_DETMETA_OLD = "                         gram_dev_max_support=D['gram_dev_max_support'], n_sites=D['n_sites'], profile=prof_d)"
LIFT_DETMETA_NEW = ("                         gram_dev_max_support=D['gram_dev_max_support'], n_sites=D['n_sites'], profile=prof_d, lift_convention=args.lift_convention,\n"
                    "                         gram_defect_default=D.get('gram_defect_default'), isometric=D.get('isometric'))")

LIFT_DETDICT_OLD = "    det_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, chain=chain_info, members=[D['meta'] for D in det_lifts],"
LIFT_DETDICT_NEW = "    det_meta = dict(lift_convention=args.lift_convention, H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, chain=chain_info, members=[D['meta'] for D in det_lifts],"

LIFT_OUT_OLD = "    out = dict(script='o5_lift_v2.py', args=vars(args), chain=chain_info, H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=L1.CONVENTION_NOTE,"
LIFT_OUT_NEW = "    out = dict(script='o5_lift_v3.py', lift_convention=args.lift_convention, args=vars(args), chain=chain_info, H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=L1.CONVENTION_NOTE,"

LIFT_REPLACEMENTS = [
    (LIFT_DOC_OLD, LIFT_DOC_NEW),
    (LIFT_SIG_OLD, LIFT_SIG_NEW),
    (LIFT_BODY_OLD, LIFT_BODY_NEW),
    (LIFT_STAT_OLD, LIFT_STAT_NEW),
    (LIFT_ARGS_OLD, LIFT_ARGS_NEW),
    (LIFT_SRC_CALL_OLD, LIFT_SRC_CALL_NEW),
    (LIFT_DET_CALL_OLD, LIFT_DET_CALL_NEW),
    (LIFT_PAIR_OLD, LIFT_PAIR_NEW),
    (LIFT_PRINT_OLD, LIFT_PRINT_NEW),
    (LIFT_SRCINFO_OLD, LIFT_SRCINFO_NEW),
    (LIFT_SRCMETA_OLD, LIFT_SRCMETA_NEW),
    (LIFT_DETMETA_OLD, LIFT_DETMETA_NEW),
    (LIFT_DETDICT_OLD, LIFT_DETDICT_NEW),
    (LIFT_OUT_OLD, LIFT_OUT_NEW),
]

# ------------------------------------------------------------------ o5_compare_v3.py
CMP_DOC_OLD = "\"\"\"o5_compare_v2.py -- O5 (GAP10_T7_final_v1.1 section 9): the comparison of the 2 propagations of the same lifted column and the acceptance rule."
CMP_DOC_NEW = ("\"\"\"o5_compare_v3.py -- RUN_O5_controls_v1: o5_compare_v2.py (never edited) plus --lift_term {gram, none} (none: no lift term, a source built natively\n"
               "on both codes; the error sum is then the 2 propagation bands) and tolerance of a cube json without a directional root (constant fields, the cut\n"
               "given by --o5_rh); built by o5_build_controls_v1.py by exact replacements. o5_compare_v2's docstring: the comparison of the 2 propagations of the same "
               "lifted column and the acceptance rule.")

CMP_CUT_OLD = "    cut_cube = float(K.get('o5_rh') or 0.0) or float(K.get('r_h'))"
CMP_CUT_NEW = "    cut_cube = float(K.get('o5_rh') or 0.0) or float(K.get('r_h') if K.get('r_h') is not None else float('nan'))"

CMP_RH_OLD = "r_h_direction=float(K['r_h'])"
CMP_RH_NEW = "r_h_direction=(float(K['r_h']) if K.get('r_h') is not None else float('nan'))"

CMP_DG_OLD = "    dG = pair['lift']['gram_defect']"
CMP_DG_NEW = "\n".join([
    "    if opts.get('lift_term') == 'none':",
    "        pair['lift'] = dict(pair['lift'], gram_defect=0.0, amp_lift_error=0.0, det_gram=[0.0 for _ in (pair['lift'].get('det_gram') or [])],",
    "                            note=\"--lift_term none: a source built natively on both codes; no lift term enters any error sum\")",
    "    dG = pair['lift']['gram_defect']",
])

CMP_HEAD_OLD = "         f\"{opts['clock_tol_rs']:g} r_s (the sweep). Every number is a reading on 1 held record at 1 averaging width; no digit is the object's.\", \"\"]"
CMP_HEAD_NEW = "\n".join([
    "         f\"{opts['clock_tol_rs']:g} r_s (the sweep). Every number is a reading on 1 held record at 1 averaging width; no digit is the object's.\"",
    "         + (\" The lift term is 0 by --lift_term none: a source built natively on both codes, no lift; the error sum is the 2 propagation bands.\" if opts.get('lift_term') == 'none' else \"\"), \"\"]",
])

CMP_ARG_OLD = "    ap.add_argument('--operator_check', default='', help=\"o5_operator_check_v1's json; with equality to rounding a v1 chain export (the reconstructed operator) is result eligible\")"
CMP_ARG_NEW = "\n".join([
    CMP_ARG_OLD,
    "    ap.add_argument('--lift_term', choices=['gram', 'none'], default='gram', help=\"none: the lift's Gram defect, amplitude lift error and detector Gram terms are set to 0 (a source built natively on both codes; RUN_O5_controls_v1's shell)\")",
])

CMP_OPTS_OLD = "                operator_check=(load_operator_check(args.operator_check) if args.operator_check else None))"
CMP_OPTS_NEW = "                operator_check=(load_operator_check(args.operator_check) if args.operator_check else None), lift_term=args.lift_term)"

CMP_JSON_OLD = "        json.dump(dict(script='o5_compare_v2.py', options=opts, results=results), fh, indent=1, default=_json_default)"
CMP_JSON_NEW = "        json.dump(dict(script='o5_compare_v3.py', options=opts, results=results), fh, indent=1, default=_json_default)"

CMP_TITLE_OLD = "    L = [\"# O5 witness exchange: the 2 propagations of the same lifted column (GAP10_T7_final_v1.1 section 9), o5_compare_v2.py\", \"\","
CMP_TITLE_NEW = "    L = [\"# O5 witness exchange: the 2 propagations of the same column (GAP10_T7_final_v1.1 section 9; RUN_O5_controls_v1), o5_compare_v3.py\", \"\","

COMPARE_REPLACEMENTS = [
    (CMP_DOC_OLD, CMP_DOC_NEW),
    (CMP_CUT_OLD, CMP_CUT_NEW),
    (CMP_RH_OLD, CMP_RH_NEW),
    (CMP_DG_OLD, CMP_DG_NEW),
    (CMP_HEAD_OLD, CMP_HEAD_NEW),
    (CMP_ARG_OLD, CMP_ARG_NEW),
    (CMP_OPTS_OLD, CMP_OPTS_NEW),
    (CMP_JSON_OLD, CMP_JSON_NEW),
    (CMP_TITLE_OLD, CMP_TITLE_NEW),
]


def build(src, dst, replacements):
    with open(src) as fh:
        text = fh.read()
    for i, (old, new) in enumerate(replacements):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{src}: replacement {i} anchor found {n} times (1 expected): the file is not the attached one; stop. Anchor:\n{old}")
        text = text.replace(old, new)
    compile(text, dst, 'exec')
    with open(dst, 'w') as fh:
        fh.write(text)
    print(f"[built] {dst} (md5 {md5(dst)}) from {src} (md5 {md5(src)}): {len(replacements)} replacements, compiled", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RUN_O5_controls_v1: build o5_lift_v3.py and o5_compare_v3.py from the v2 files by exact replacements")
    ap.add_argument('--lift_v2', default='o5_lift_v2.py')
    ap.add_argument('--compare_v2', default='o5_compare_v2.py')
    ap.add_argument('--out', default='.', help="the directory the built files are written to (beside the v2 files, which are never edited)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    build(args.lift_v2, os.path.join(args.out, 'o5_lift_v3.py'), LIFT_REPLACEMENTS)
    build(args.compare_v2, os.path.join(args.out, 'o5_compare_v3.py'), COMPARE_REPLACEMENTS)
    return 0


if __name__ == '__main__':
    sys.exit(main())
