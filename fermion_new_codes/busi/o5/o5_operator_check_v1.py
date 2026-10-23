#!/usr/bin/env python
"""o5_operator_check_v1.py -- O5: the reconstructed chain operator of o5_chain_export_v1 (HORIZON v2's written definition) against the solve
code's own operator (page_ops_v6.clocked_csr(page_ops_v6.chain_sparse(r, m, kappa), N, v), imported read-only from --solve_path) on the T7
record's grid and applied fields, at m 0.2 and kappa 1 and 2: the max entry difference, the Frobenius difference (with the operator's Frobenius
norm for scale), the Hermiticity and particle-hole defects of each, the 5 lowest positive eigenvalues of each (and the largest difference over
the whole spectrum), the fixed-cut number-current operator at the table's outer root (v1's i[h, P_>] against page_ops_v6.number_current_csr).
The same for the launch column at a k_0 2 pi/64 at R_c 20 and 14 a: the particle spinor (v1's closed form against
engine_single_object_B_v1.particle_spinor's eigh vector, phase-aligned, both eigen-residuals on the symbol), the columns (v1's default: the
density-width-sigma Gaussian with the 5 sigma cut; HORIZON's own member at this carrier (2 pi/64, sigma 16 amplitude width, no cut); HORIZON's
form at the exchange width sigma 8 (o5_chain_export_v2's column, the KS override reported); v1's form at the density width sigma/sqrt 2 without a
cut, the width-equivalent of HORIZON's), their mutual overlaps and phase-aligned differences, every column's <h> and energy rms on both
operators, its weight at r <= 3 a (the shells where the lift is poorest) and its lattice-scale share. Writes <out>.json; prints a verdict line:
'the reconstructed operator equals the solve operator to rounding (max entry difference ...)' or 'DIFFERS'. The builder runs it before the
chain exports: v2 is used whatever it shows; if the operators differ, nothing run with v1's operator is a result. CPU, under a minute per kappa.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import o5_chain_export_v1 as V1  # noqa: E402
import o5_chain_export_v2 as V2  # noqa: E402


def phase_align(a, b):
    """(e^{i phi} b, phi, |<a, b>|) with the unit phase that makes <a, e^{i phi} b> real and positive."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    ov = np.vdot(a, b)
    phi = float(-np.angle(ov)) if abs(ov) > 0.0 else 0.0
    return b * np.exp(1.0j * phi), phi, float(abs(ov))


def lowest_positive(w, k=5):
    pos = np.sort(np.asarray(w)[np.asarray(w) > 0.0])
    return pos[:k]


def compare_operators(r, N, v, m, kappa, OPS, R_c=None, neig=5):
    """v1's reconstruction against the solve code's matrix on the same grid and fields; returns (results, v1 chain, solve chain)."""
    c1 = V1.RadialChain(r, N, v, m, kappa)
    c2 = V2.SolveChain(r, N, v, m, kappa, OPS)
    D = c1.H - c2.H
    out = dict(kappa=int(kappa), m=float(m), dim=int(c1.dim), max_entry_diff=float(np.max(np.abs(D))), frobenius_diff=float(np.linalg.norm(D)),
               frobenius_v1=float(np.linalg.norm(c1.H)), frobenius_solve=float(np.linalg.norm(c2.H)), max_entry_abs=float(np.max(np.abs(c2.H))),
               hermiticity_v1=c1.hermiticity_defect(), hermiticity_solve=c2.hermiticity_defect(), particle_hole_v1=c1.particle_hole_defect(),
               particle_hole_solve=c2.particle_hole_defect(), gershgorin_solve=c2.gershgorin, nnz_solve=int(c2.h_csr.nnz))
    w1 = np.linalg.eigvalsh(c1.H)
    w2 = np.linalg.eigvalsh(c2.H)
    l1, l2 = lowest_positive(w1, neig), lowest_positive(w2, neig)
    out.update(lowest_positive_v1=l1.tolist(), lowest_positive_solve=l2.tolist(), lowest_positive_diff_max=float(np.max(np.abs(l1 - l2))),
               eigenvalue_diff_max=float(np.max(np.abs(w1 - w2))), spectrum_abs_max_v1=float(np.max(np.abs(w1))), spectrum_abs_max_solve=float(np.max(np.abs(w2))))
    if R_c is not None:
        Qs, _ = c2.number_current(R_c)
        Qv = c1.cut_operator(np.asarray(r) > float(R_c))
        out.update(cut_radius=float(R_c), number_current_max_entry_diff=float(np.max(np.abs(Qs - Qv))), number_current_nonzeros_solve=int(np.count_nonzero(Qs)),
                   number_current_max_entry_abs=float(np.max(np.abs(Qs))))
    return out, c1, c2


def compare_columns(r, N, v, m, kappa, Rc, ak0, sigma, cut, ESB, c1, c2):
    """The spinors and the 4 columns at R_c (see the module docstring), with <h> and rms on both operators and their pairwise overlaps."""
    r = np.asarray(r, dtype=float)
    R_sp = max(float(Rc), 1.0)
    u1, E1 = V1.particle_spinor(ak0, Rc, m, kappa)
    u2 = np.asarray(ESB.particle_spinor(m, ak0, kappa, R_sp), dtype=np.complex128)
    u2a, phi_u, ov_u = phase_align(u1, u2)
    D = m * V1.SZ + math.sin(ak0) * V1.SX + (kappa / R_sp) * V1.SY
    spinor = dict(v1=[[u1[0].real, u1[0].imag], [u1[1].real, u1[1].imag]], engine=[[u2[0].real, u2[0].imag], [u2[1].real, u2[1].imag]], overlap_abs=ov_u, phase=phi_u,
                  phase_aligned_max_diff=float(np.max(np.abs(u1 - u2a))), eigen_residual_v1=float(np.linalg.norm(D @ u1 - E1 * u1)),
                  eigen_residual_engine=float(np.linalg.norm(D @ u2 - E1 * u2)), E_symbol=float(E1))
    cols = []
    psiA, _ = V1.gaussian_column(r, Rc, sigma, ak0, cut, u1)
    cols.append((f"v1 default: density width {sigma:g} a, cut {cut:g} sigma, closed-form spinor", psiA, None))
    ks_sig = [float(s) for k, s, _ in ESB.KS if abs(float(k) - abs(ak0)) < 1e-9]
    if ks_sig:
        hB = V2.horizon_column(ESB, r, m, kappa, Rc, ak0, ks_sig[0], N, v)
        cols.append((f"HORIZON's own member at this carrier: amplitude width {ks_sig[0]:g} a (density {ks_sig[0] / math.sqrt(2.0):.4g} a), no cut, eigh spinor (KS override {hB['ks_override']})", hB['psi0'], hB))
    hC = V2.horizon_column(ESB, r, m, kappa, Rc, ak0, sigma, N, v)
    cols.append((f"HORIZON's form at the exchange width: amplitude width {sigma:g} a (density {sigma / math.sqrt(2.0):.4g} a), no cut, eigh spinor (KS override {hC['ks_override']}): o5_chain_export_v2's column", hC['psi0'], hC))
    psiD, _ = V1.gaussian_column(r, Rc, sigma / math.sqrt(2.0), ak0, 1e6, u1)
    cols.append((f"v1's form at the density width {sigma / math.sqrt(2.0):.4g} a without a cut: the width-equivalent of HORIZON's sigma {sigma:g}", psiD, None))
    rows = []
    for name, psi, hinfo in cols:
        f = psi.reshape(-1)
        e1, e1sq = c1.expectation(f)
        e2, e2sq = c2.expectation(f)
        rows.append(dict(name=name, norm=float(np.vdot(f, f).real), E_mean_v1_operator=e1, E_rms_v1_operator=math.sqrt(max(e1sq - e1 * e1, 0.0)), E_mean_solve_operator=e2,
                         E_rms_solve_operator=math.sqrt(max(e2sq - e2 * e2, 0.0)), E_mean_operator_diff=e2 - e1, lattice_scale_share=float(V1.lattice_scale_weight(psi)['lattice_scale']),
                         weight_r_le_3=float(np.sum(np.abs(psi[r <= 3.0]) ** 2)), weight_below_half=float(V2.envelope_weights(hinfo['profile'], r)['gaussian_weight_below_grid']) if hinfo else None,
                         declared_profile_max_diff=(hinfo['declared_profile_max_diff'] if hinfo else None), ks_override=(hinfo['ks_override'] if hinfo else None)))
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i][1].reshape(-1), cols[j][1].reshape(-1)
            ba, phi, ov = phase_align(a, b)
            pairs.append(dict(a=cols[i][0], b=cols[j][0], overlap_abs=ov, phase=phi, phase_aligned_max_entry_diff=float(np.max(np.abs(a - ba))),
                              E_mean_diff_solve_operator=rows[j]['E_mean_solve_operator'] - rows[i]['E_mean_solve_operator']))
    return dict(kappa=int(kappa), Rc=float(Rc), ak0=float(ak0), sigma=float(sigma), cut=float(cut), spinor=spinor, columns=rows, pairs=pairs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="O5: v1's reconstructed chain operator and column against the solve code's (page_ops_v6, engine_single_object_B_v1) on the T7 record")
    ap.add_argument('--record', required=True, help="record_slice.npz (keys r, N, v)")
    ap.add_argument('--solve_path', default=os.environ.get('O5_SOLVE_PATH', ''), help="the directory holding page_ops_v6.py and engine_single_object_B_v1.py (imported read-only)")
    ap.add_argument('--ma', type=float, default=0.2)
    ap.add_argument('--kappas', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--Rc', type=float, nargs='+', default=[20.0, 14.0])
    ap.add_argument('--ak0', type=float, default=2.0 * math.pi / 64.0, help="the carrier (2 pi/64 exactly by default: HORIZON's own member is found only at its exact k_0)")
    ap.add_argument('--sigma', type=float, default=8.0, help="the exchange width (HORIZON's amplitude convention) and v1's default density width")
    ap.add_argument('--cut', type=float, default=5.0, help="v1's cut")
    ap.add_argument('--rounding_tol', type=float, default=1e-13, help="max entry difference below which the operators count as equal to rounding")
    ap.add_argument('--out', default='o5_operator_check_v1', help="writes <out>.json")
    args = ap.parse_args(argv)
    if not args.solve_path:
        raise SystemExit("--solve_path (or the environment's O5_SOLVE_PATH) is required")
    tw = time.time()
    OPS, ESB, solve_info = V2.load_solve_modules(args.solve_path)
    r, N, v = V1.load_record(args.record)
    roots = V1.table_roots(r, N, v)
    r_h = roots[-1][0] if roots else None
    print(f"[operator check] record {args.record} (md5 {V1.md5(args.record)}): n={r.size}, r_h (table outer root) {r_h}; solve code {solve_info['path']} "
          f"(page_ops_v6 md5 {solve_info['page_ops_v6']['md5']}, engine md5 {solve_info['engine_single_object_B_v1']['md5']}, page_history_v6b placeholder {solve_info['page_history_v6b_placeholder']}); "
          f"m={args.ma:g}, kappas {args.kappas}, R_c {args.Rc}, a k_0 {args.ak0:.10f}, sigma {args.sigma:g}", flush=True)
    out = dict(script='o5_operator_check_v1.py', args=vars(args), record=dict(path=args.record, md5=V1.md5(args.record), n=int(r.size), roots_on_table=[list(x) for x in roots], r_h=r_h),
               solve_code=solve_info, operators=[], columns=[])
    worst = 0.0
    for kappa in args.kappas:
        t0 = time.time()
        res, c1, c2 = compare_operators(r, N, v, args.ma, kappa, OPS, R_c=r_h)
        res['wall'] = time.time() - t0
        out['operators'].append(res)
        worst = max(worst, res['max_entry_diff'])
        print(f"[operators kappa {kappa}] dim {res['dim']}: max entry difference {res['max_entry_diff']:.2e} (largest entry {res['max_entry_abs']:.4f}); Frobenius difference {res['frobenius_diff']:.2e} "
              f"(||h||_F {res['frobenius_solve']:.4f}); Hermiticity defects v1 {res['hermiticity_v1']:.1e}, solve {res['hermiticity_solve']:.1e}; particle-hole defects v1 {res['particle_hole_v1']:.1e}, "
              f"solve {res['particle_hole_solve']:.1e}; 5 lowest positive eigenvalues v1 {[round(x, 10) for x in res['lowest_positive_v1']]}, solve {[round(x, 10) for x in res['lowest_positive_solve']]} "
              f"(max difference {res['lowest_positive_diff_max']:.1e}); whole spectrum max difference {res['eigenvalue_diff_max']:.1e}; number current at r_h {res.get('cut_radius')}: "
              f"max entry difference {res.get('number_current_max_entry_diff', float('nan')):.1e} ({res.get('number_current_nonzeros_solve')} nonzeros); {res['wall']:.1f}s", flush=True)
        for Rc in args.Rc:
            cres = compare_columns(r, N, v, args.ma, kappa, Rc, args.ak0, args.sigma, args.cut, ESB, c1, c2)
            out['columns'].append(cres)
            sp = cres['spinor']
            print(f"[column kappa {kappa} R_c {Rc:g}] spinor: |<u_v1, u_engine>| = {sp['overlap_abs']:.12f} (phase {sp['phase']:+.6f}), phase-aligned max difference {sp['phase_aligned_max_diff']:.1e}; "
                  f"eigen-residuals on the symbol v1 {sp['eigen_residual_v1']:.1e}, engine {sp['eigen_residual_engine']:.1e}; E of the symbol {sp['E_symbol']:.6f}", flush=True)
            for row in cres['columns']:
                print(f"[column kappa {kappa} R_c {Rc:g}] {row['name']}: norm {row['norm']:.12f}; <h> on v1's operator {row['E_mean_v1_operator']:.8f}, on the solve operator {row['E_mean_solve_operator']:.8f} "
                      f"(difference {row['E_mean_operator_diff']:+.1e}); rms {row['E_rms_solve_operator']:.6f}; weight at r <= 3 a {row['weight_r_le_3']:.2e}; lattice-scale share {row['lattice_scale_share']:.2e}"
                      + (f"; density below r = 1/2: {row['weight_below_half']:.2e}; declared profile vs engine {row['declared_profile_max_diff']:.1e}" if row['weight_below_half'] is not None else ""), flush=True)
            for pr in cres['pairs']:
                print(f"[pairs kappa {kappa} R_c {Rc:g}] ({pr['a'].split(':')[0]}) against ({pr['b'].split(':')[0]}): |overlap| {pr['overlap_abs']:.6f}, phase-aligned max entry difference "
                      f"{pr['phase_aligned_max_entry_diff']:.2e}, <h> difference on the solve operator {pr['E_mean_diff_solve_operator']:+.2e}", flush=True)
    equal = bool(worst <= args.rounding_tol)
    verdict = (f"the reconstructed operator equals the solve operator to rounding (max entry difference {worst:.2e} <= {args.rounding_tol:g}) at kappa {args.kappas}"
               if equal else f"DIFFERS: max entry difference {worst:.2e} > {args.rounding_tol:g}: nothing run with v1's operator is a result; v2 (the solve operator) is used")
    out['summary'] = dict(max_entry_diff=worst, rounding_tol=args.rounding_tol, equal_to_rounding=equal, verdict=verdict, wall=time.time() - tw)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=V1._json_default)
    print(f"[verdict] {verdict}; {args.out}.json written ({time.time() - tw:.0f}s)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
