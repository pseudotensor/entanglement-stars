#!/usr/bin/env python3
"""
gap06_addon_symmetry_refsector_v2.py  (busi lane, task 4; version 2; v1 kept; imports GAP06's task8_cluster_control.py
unchanged)

(1) The corrected symmetry test of GAP06_judgment_v1.md section 1.5: S commutes with every vertex and h0; Sigma
    commutes with h0 and maps e_x -> e_{-x}, p_x -> -p_{-x}; on every Hartree background the Hessian B and the pair
    kernel V are invariant under the signed reflection sigma of the vertex index, hbar and Y commute with S and Sigma,
    and the sector Hamiltonian conserves the 4 block numbers.
(2) The second-order connected reference energy of each case (S10 item c, native scheme), and the sea relaxation
    (singles, not connected); optionally the exact N0 sector of the 5-site parent (--refsector-exact).

Changes against version 1 (HF_judgment_v2.md sections 2 and 3):
  1. The off-group norm of the sector Hamiltonian was computed as sqrt(||H||^2 - sum_g ||H_gg||^2), a difference of two
     nearly equal sums whose roundoff (1e-15 relative) gives sqrt(1e-15) = 3e-8 or, when negative, the clipped value
     0.0 (the 24 printed values 0.0 / 1e-8 .. 1.3e-7). It is now the Frobenius norm of the off-group entries themselves,
     together with the commutator norms ||[H, N_b]||/||H|| for the 4 block-number operators; the v1 value is printed
     beside them for the record.
  2. Per case the Hartree background's convergence (iterations, residual, block gap), the chart margins (mass margin,
     sheet gap, lowest sheet, q*, sigma_min(I + P^A B), |w|max, compactness), the inner source residual and the final
     source defect ||(C_H - R(h[Dbar])) - Dbar|| recomputed on the returned background are printed and enter the
     acceptance (the hreduce review's and red team's requirement).
  3. --gap06-json audits GAP06's own results file: per case the selected HF trial's convergence, residual, block gap and
     aufbau gap, the Hartree convergence and the inner residual (the trials whose energies the printed ratios use).
"""
import os
import sys


def _set_threads():
    if '--threads' in sys.argv:
        t = sys.argv[sys.argv.index('--threads') + 1]
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[v] = str(t)


_set_threads()
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

VERSION = 'gap06_addon_symmetry_refsector_v2'


def log(*a):
    print(*a, flush=True)


def fnorm(X):
    return float(np.linalg.norm(X))


def offgroup_norm_v1(H, groups, dense):
    """Version-1 formula, kept for the record: sqrt(||H||^2 - sum_g ||H_gg||^2)/||H|| (cancellation-limited)."""
    tot2 = (fnorm(H) if dense else sp.linalg.norm(H)) ** 2
    diag2 = 0.0
    for idx in groups.values():
        Hb = H[np.ix_(idx, idx)] if dense else H[idx][:, idx]
        diag2 += (fnorm(Hb) if dense else sp.linalg.norm(Hb)) ** 2
    return math.sqrt(max(tot2 - diag2, 0.0) / tot2)


def offgroup_direct(H, groups, dense):
    """Frobenius norm of the off-group entries of H over ||H||_F, and max_b ||[H, N_b]||_F/||H||_F for the block-number
    operators N_b (diagonal in the sector basis: the block numbers of each basis state)."""
    n = H.shape[0]
    lab = np.full(n, -1, dtype=np.int64)
    pats = list(groups.keys())
    for gi, pat in enumerate(pats):
        lab[groups[pat]] = gi
    assert np.all(lab >= 0), 'every basis state must belong to a group'
    nb = np.array(pats, dtype=float)
    if dense:
        Hd = np.asarray(H)
        a2 = np.abs(Hd) ** 2
        tot2 = float(np.sum(a2))
        mask = lab[:, None] != lab[None, :]
        off2 = float(np.sum(a2[mask]))
        comm2 = 0.0
        for b in range(nb.shape[1]):
            nbi = nb[lab, b]
            comm2 = max(comm2, float(np.sum(a2 * (nbi[:, None] - nbi[None, :]) ** 2)))
    else:
        Hc = H.tocoo()
        a2 = np.abs(Hc.data) ** 2
        tot2 = float(np.sum(a2))
        m = lab[Hc.row] != lab[Hc.col]
        off2 = float(np.sum(a2[m]))
        comm2 = 0.0
        for b in range(nb.shape[1]):
            nbi = nb[lab, b]
            comm2 = max(comm2, float(np.sum(a2 * (nbi[Hc.row] - nbi[Hc.col]) ** 2)))
    tot2 = max(tot2, 1e-300)
    return math.sqrt(off2 / tot2), math.sqrt(comm2 / tot2)


def signed_reflection(parent):
    ix = {int(x): i for i, x in enumerate(parent.xs)}
    L = parent.L
    sigma = np.zeros((parent.nv, parent.nv))
    for x in parent.xs:
        sigma[ix[-x], ix[x]] = 1.0
        sigma[L + ix[-x], L + ix[x]] = -1.0
    cov = max(max(fnorm(parent.Sig_op @ parent.e[ix[x]] @ parent.Sig_op - parent.e[ix[-x]]),
                  fnorm(parent.Sig_op @ parent.pc[ix[x]] @ parent.Sig_op + parent.pc[ix[-x]])) for x in parent.xs)
    commS = max(fnorm(parent.S_op @ A - A @ parent.S_op) for A in parent.A + [parent.h0])
    commSig = fnorm(parent.Sig_op @ parent.h0 - parent.h0 @ parent.Sig_op)
    return sigma, dict(comm_S=commS, comm_Sigma_h0=commSig, cov_Sigma=cov, offblock_h0=parent.offblock_norm(parent.to_blk(parent.h0)))


def reference_second_order(parent, bgH, V, quad):
    U, eps, occ = bgH.U, bgH.eps, bgH.occ
    unocc = np.setdiff1d(np.arange(parent.d), occ)
    At = np.array([U.conj().T @ A @ U for A in parent.A])
    Avi = At[:, unocc][:, :, occ]                                   # (nv, nu, no)
    T = np.einsum('mn,mai,nbj->aibj', V, Avi, Avi)
    Tas = T - T.transpose(0, 3, 2, 1)
    ea, ei = eps[unocc], eps[occ]
    den = ea[:, None, None, None] - ei[None, :, None, None] + ea[None, None, :, None] - ei[None, None, None, :]
    E2d = -0.25 * float(np.sum(np.abs(Tas) ** 2 / den))
    B21t = U.conj().T @ quad.B21 @ U
    sing = B21t[np.ix_(unocc, occ)]
    E2s = -float(np.sum(np.abs(sing) ** 2 / (ea[:, None] - ei[None, :])))
    return E2d, E2s, float(np.max(np.abs(sing)))


def final_source_defect(T8, P, kt, G, C, bg, ref):
    """The source fixed point evaluated once more on the returned background: with s = src(Dbar), h2 = h[s],
    R2 = R(h2) (moving) or P0 (fixed), the defect ||(C - R2) - Dbar||, the change of h and of R."""
    L = P.L
    sol = T8.radial_solve(bg.s[:L], bg.s[L:], G, kt)
    h2 = P.h_of(sol['Phi'].real)
    R2 = P.P0 if ref == 'fixed' else T8.continued_projector(h2, bg.R, P.N0)[0]
    return fnorm((C - R2) - bg.D), fnorm(h2 - bg.h) / max(fnorm(bg.h), 1e-300), fnorm(R2 - bg.R)


def audit_gap06_json(path):
    """Per case of GAP06's results file: the selected HF trial's convergence data, the Hartree convergence and the inner
    residual (the hreduce review's requirement: a stopped damped iterate is not a Slater trial)."""
    J = json.load(open(path))
    rows = []
    ok_all = True
    log('\nAUDIT of %s (used trials):' % path)
    for c in J.get('cases', []):
        patF = str(c.get('pattern_F'))
        hfp = c.get('HF_patterns', {}).get(patF, {})
        hart = c.get('hartree', {})
        marg = c.get('margins', {})
        vinfo = c.get('V_info', {})
        inner = marg.get('inner_residual', None)
        ok = bool(hfp.get('converged', False)) and bool(hart.get('converged', False)) and (inner is not None and float(inner) < 1e-10)
        ok_all = ok_all and ok
        row = dict(tag=c.get('tag'), pattern_F=patF, HF_converged=hfp.get('converged'), HF_iterations=hfp.get('iterations'),
                   HF_residual=hfp.get('residual'), HF_block_gap=hfp.get('block_gap'), HF_aufbau_gap=hfp.get('aufbau_gap'),
                   hartree_converged=hart.get('converged'), hartree_residual=hart.get('residual'), inner_residual=inner,
                   mass_margin=marg.get('mass_margin'), sheet_gap=marg.get('sheet_gap'), sheet_is_lowest=marg.get('sheet_is_lowest'),
                   qstar=vinfo.get('qstar'), smin=vinfo.get('smin'), used_trials_ok=ok)
        rows.append(row)
        log('  %-32s HF(%s): converged %s it %s res %s block gap %s aufbau gap %s | Hartree converged %s res %s | inner res %s | mass margin %s gap %s q* %s -> %s'
            % (row['tag'], patF, row['HF_converged'], row['HF_iterations'], row['HF_residual'], row['HF_block_gap'], row['HF_aufbau_gap'],
               row['hartree_converged'], row['hartree_residual'], inner, row['mass_margin'], row['sheet_gap'], row['qstar'], 'PASS' if ok else 'FAIL'))
    log('  used trials converged in every case: %s' % ('PASS' if ok_all else 'FAIL'))
    return rows, ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gap06-dir', required=True, help='directory containing GAP06 task8_cluster_control.py')
    ap.add_argument('--gap06-json', default=None, help="GAP06's task8_results.json for the audit of the used HF/Hartree trials")
    ap.add_argument('--couplings', default='0.0009765625,0.00390625,0.015625')
    ap.add_argument('--dr', type=float, default=0.01)
    ap.add_argument('--nq', type=int, default=24)
    ap.add_argument('--out', default='gap06_addon_v2.json')
    ap.add_argument('--refsector-exact', action='store_true', help='exact N0 sector of the 5-site parent (sea pattern block)')
    ap.add_argument('--skip-large', action='store_true', help='skip the 7-site 4-hole sector')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.gap06_dir)
    import task8_cluster_control as T8  # noqa: E402
    couplings = [float(x) for x in args.couplings.split(',')]
    results = dict(version=VERSION, parents={}, cases=[])
    acc_static = True
    acc_cases = True
    acc_backgrounds = True
    for L in (5, 7):
        P = T8.Parent(L)
        sigma, st = signed_reflection(P)
        results['parents'][L] = st
        okp = all(v < 1e-12 for v in st.values())
        acc_static = acc_static and okp
        log('parent %d: comm S %.1e  comm Sigma-h0 %.1e  Sigma covariance e_x->e_-x, p_x->-p_-x %.1e  h0 off-block %.1e  -> %s'
            % (L, st['comm_S'], st['comm_Sigma_h0'], st['cov_Sigma'], st['offblock_h0'], 'PASS' if okp else 'FAIL'))
        kt = T8.KernelTables(P, 4.0, 8.0, args.dr, args.nq)
        sectors = {}
        for holes in ((2, 4) if (L == 5 or not args.skip_large) else (2,)):
            N = P.d - holes
            t0 = time.time()
            sectors[N] = T8.Sector(P.d, N, P.block_of_mode)
            log('  sector d=%d N=%d dim=%d (%.1f s)' % (P.d, N, sectors[N].dim, time.time() - t0))
        for G in couplings:
            for N in sorted(sectors, reverse=True):
                for ref in ('moving', 'fixed'):
                    if N == P.d - 4 and L == 7 and ref == 'fixed':
                        continue
                    pats, shell = T8.natural_patterns(P, N)
                    best = None
                    for pat in pats:
                        C, bg, info = T8.hartree_blocked(P, kt, G, pat, T8.free_state(P, pat), ref)
                        if best is None or bg.E < best[1].E:
                            best = (C, bg, info, pat)
                    C_H, bgH, infoH, patH = best
                    defect, dh, dR = final_source_defect(T8, P, kt, G, C_H, bgH, ref)
                    B, bM = T8.hessian_B(P, kt, G, bgH.s)
                    B = 0.5 * (B + B.T)
                    PA = T8.response_PA(P, bgH)[0] if ref == 'moving' else np.zeros_like(B)
                    V, vinfo = T8.build_V(B, PA)
                    quad = T8.Quad(P, bgH, V)
                    sector = sectors[N]
                    dense = sector.dim <= 5000
                    H = T8.build_sector_H(sector, quad.Es, P.to_blk(quad.Y), P.A_blk, V, dense)
                    off_direct, comm_Nb = offgroup_direct(H, sector.groups, dense)
                    sym = dict(V_odd=fnorm(V - sigma @ V @ sigma) / max(fnorm(V), 1e-300), B_odd=fnorm(B - sigma @ B @ sigma) / max(fnorm(B), 1e-300),
                               h_Sigma=fnorm(bgH.h @ P.Sig_op - P.Sig_op @ bgH.h) / fnorm(bgH.h),
                               Y_Sigma=fnorm(quad.Y @ P.Sig_op - P.Sig_op @ quad.Y) / fnorm(quad.Y),
                               Y_S=fnorm(quad.Y @ P.S_op - P.S_op @ quad.Y) / fnorm(quad.Y),
                               H_offgroup=off_direct, H_block_number_commutator=comm_Nb)
                    ok = max(sym.values()) < 1e-12
                    acc_cases = acc_cases and ok
                    background = dict(converged=bool(infoH['converged']), iterations=infoH['iterations'], residual=infoH['residual'],
                                      block_gap=infoH['block_gap'], aufbau_gap=infoH['aufbau_gap'], inner_iterations=bgH.inner_iterations,
                                      inner_residual=bgH.inner_residual, final_source_defect=defect, final_h_change_rel=dh, final_R_change=dR,
                                      mass_margin=bgH.mass_margin, sheet_gap=bgH.gap, sheet_is_lowest=bool(bgH.sheet_is_lowest),
                                      compactness=bgH.compactness, wmax=bgH.wmax, N_center=bgH.N_center,
                                      qstar=vinfo['qstar'], smin_I_plus_PAB=vinfo['smin'])
                    okb = (background['converged'] and defect < 1e-9 and bgH.mass_margin > 0 and bgH.gap > 0 and bool(bgH.sheet_is_lowest) and bgH.wmax < 1.0)
                    acc_backgrounds = acc_backgrounds and okb
                    E2d, E2s, smax = reference_second_order(P, bgH, V, quad)
                    case = dict(L=L, N=N, G=G, ref=ref, pattern_H=list(patH), E_H=bgH.E, dE_grav=bgH.E - P.E_free_pattern(patH), B20=quad.B20,
                                sym=sym, sym_pass=ok, H_offgroup_v1_formula=offgroup_norm_v1(H, sector.groups, dense),
                                background=background, background_pass=okb,
                                E2_ref_doubles=E2d, E2_ref_singles_relaxation=E2s, max_single_B21=smax,
                                V_eig=[vinfo['V_eig_min'], vinfo['V_eig_max']])
                    log('  L%d N%d G%.6g %s: sym V_odd %.1e B_odd %.1e h_Sigma %.1e Y_Sigma %.1e Y_S %.1e H off-group %.1e [H,N_b] %.1e (v1 formula %.1e) -> %s ; '
                        'background: Hartree converged %s (it %d res %.1e, block gap %.3e), inner res %.1e, final defect %.1e, mass margin %.3e, sheet gap %.3f, '
                        'lowest %s, |w|max %.3f, q* %.3e, smin %.3e -> %s ; reference 2p2h E2 %.4e (dE_grav %.4e, ratio %.2e); sea relaxation (singles, not connected) %.4e'
                        % (L, N, G, ref, sym['V_odd'], sym['B_odd'], sym['h_Sigma'], sym['Y_Sigma'], sym['Y_S'], sym['H_offgroup'],
                           sym['H_block_number_commutator'], case['H_offgroup_v1_formula'], 'PASS' if ok else 'FAIL',
                           background['converged'], background['iterations'], background['residual'], background['block_gap'],
                           background['inner_residual'], defect, bgH.mass_margin, bgH.gap, bool(bgH.sheet_is_lowest), bgH.wmax, vinfo['qstar'], vinfo['smin'],
                           'PASS' if okb else 'FAIL', E2d, case['dE_grav'], E2d / case['dE_grav'], E2s))
                    if args.refsector_exact and L == 5 and ref == 'moving' and N == P.d - 2:
                        t0 = time.time()
                        sec0 = T8.Sector(P.d, P.N0, P.block_of_mode)
                        pat0 = tuple(int(round(x)) for x in P.block_counts(bgH.R))
                        idx = sec0.groups[pat0]
                        lam, Uv = np.linalg.eigh(0.5 * (V + V.T))
                        lmax = max(float(np.max(np.abs(lam))), 1e-300)
                        Yg = sec0.one_body(P.to_blk(quad.Y))[idx][:, idx]
                        Oms = []
                        for k in range(len(lam)):
                            if abs(lam[k]) < 1e-15 * lmax:
                                continue
                            Ak = sum(Uv[mu, k] * P.A_blk[mu] for mu in range(P.nv))
                            Oms.append((0.5 * lam[k], sec0.one_body(Ak)[idx][:, idx]))

                        def mv(x):
                            y = quad.Es * x + Yg @ x
                            for lk, Om in Oms:
                                y = y + lk * (Om @ (Om @ x))
                            return y

                        op = spla.LinearOperator((len(idx), len(idx)), matvec=mv, dtype=complex)
                        w = spla.eigsh(op, k=2, which='SA', tol=1e-12, maxiter=20000, return_eigenvectors=False)
                        w = np.sort(w)
                        case['refsector_exact'] = dict(pattern=list(pat0), dim=len(idx), E_Q_N0=float(w[0]), E_Q_N0_second=float(w[1]),
                                                       E_Q_N0_minus_B20=float(w[0]) - quad.B20, seconds=time.time() - t0)
                        log('    exact N0 sector (pattern %s, dim %d): E_Q(N0) = %.12f, E_Q(N0) - B20 = %.4e  (doubles %.4e + singles %.4e = %.4e) (%.0f s)'
                            % (pat0, len(idx), w[0], w[0] - quad.B20, E2d, E2s, E2d + E2s, time.time() - t0))
                    results['cases'].append(case)
    audit = None
    acc_audit = None
    if args.gap06_json:
        try:
            rows, acc_audit = audit_gap06_json(args.gap06_json)
            audit = rows
        except Exception as ex:
            log('audit of %s failed: %s' % (args.gap06_json, ex))
    results['gap06_audit'] = audit
    results['acceptance'] = dict(symmetry_static=acc_static, symmetry_cases=acc_cases, backgrounds=acc_backgrounds, gap06_used_trials=acc_audit)
    log('\nACCEPTANCE (%s): static %s  per-case symmetry (direct off-group and block-number commutator) %s  backgrounds (converged, final defect, chart) %s  GAP06 used trials %s'
        % (VERSION, 'PASS' if acc_static else 'FAIL', 'PASS' if acc_cases else 'FAIL', 'PASS' if acc_backgrounds else 'FAIL',
           'n/a' if acc_audit is None else ('PASS' if acc_audit else 'FAIL')))
    with open(args.out, 'w') as fh:
        json.dump(T8.jsonable(results), fh, indent=1)
    log('written %s' % args.out)


if __name__ == '__main__':
    main()
