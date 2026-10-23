"""Saved-artifact-only spectral comparison; no original records or evolution opened."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bprime_runtime import admit, identity, require_identity, verify_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('production', 'inputs'):
        parser.add_argument('--'+role, type=Path, required=True)
        parser.add_argument('--'+role+'-sha256', required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Saved-only Bprime full-coordinate spectral comparison', 1)
    if any((args.outdir/name).exists() for name in ('comparison.json', 'comparison_arrays.npz')):
        raise ValueError('Comparison outputs already exist; use a fresh directory')
    import numpy as np
    verify_threads(runtime, 'after_imports')
    source = identity(__file__)
    input_binding = require_identity(args.inputs, args.inputs_sha256)
    production_binding = require_identity(args.production, args.production_sha256)
    inputs, production = json.loads(args.inputs.read_text()), json.loads(args.production.read_text())
    if not inputs['all_passed']:
        raise ValueError('Native input admission failed')
    if production['input_summary_binding']['sha256'] != input_binding['sha256']:
        raise ValueError('Production used a different admitted input summary')
    ia, pa = inputs['artifact'], production['artifact']
    if production['input_artifact_binding']['sha256'] != ia['sha256']:
        raise ValueError('Production used a different frozen benchmark artifact')
    require_identity(ia['path'], ia['sha256'])
    require_identity(pa['path'], pa['sha256'])
    with np.load(ia['path'], allow_pickle=False) as archive:
        data = {name: archive[name] for name in ('benchmark_U', 'benchmark_lambda', 'benchmark_ranks', 'initial_Q0',
                'full_supply_Q0', 'supply_source_G', 'source_full_spectrum', 'source_T', 'L')}
    with np.load(pa['path'], allow_pickle=False) as archive:
        own = {name: archive[name] for name in ('coarse_loss_Gram', 'fine_loss_Gram', 'coarse_loss_action', 'fine_loss_action',
                                               'initial_Q0', 'benchmark_U', 'benchmark_lambda', 'benchmark_ranks')}
    for name in ('initial_Q0', 'benchmark_U', 'benchmark_lambda', 'benchmark_ranks'):
        if not np.array_equal(data[name], own[name]):
            raise ValueError('Saved producer/input association differs: '+name)
    U, lam, ranks, Q0 = [data[name] for name in ('benchmark_U', 'benchmark_lambda', 'benchmark_ranks', 'initial_Q0')]
    if U.shape != (128, 16) or lam.shape != (16,) or ranks.shape != (16,) or Q0.shape != (16, 16):
        raise ValueError('Wrong frozen benchmark dimensions')
    arrays, checks = {}, []

    def herm(value):
        return (value+value.conj().T)/2

    def norm2(value):
        return float(np.linalg.norm(value, 2))

    def maximum(value):
        return float(np.max(np.abs(value)))

    def check(name, measured, threshold):
        checks.append({'name': name, 'measured': float(measured), 'threshold': float(threshold), 'passed': bool(measured <= threshold)})

    qvalues, qvectors = np.linalg.eigh(herm(Q0))
    if np.any(qvalues <= 0):
        raise ValueError('Initial benchmark Gram not positive definite')
    inverse_half = (qvectors/np.sqrt(qvalues)[None])@qvectors.conj().T
    target = U.conj().T@herm(data['supply_source_G'][0])@U
    target_raw_spectrum = np.linalg.eigvalsh(herm(target))
    target_normalized_spectrum = np.linalg.eigvalsh(herm(inverse_half@target@inverse_half))
    arrays.update(source_selected_Gram=target, source_selected_raw_spectrum=target_raw_spectrum,
                  source_selected_normalized_spectrum=target_normalized_spectrum, initial_Q0=Q0,
                  Q0_eigenvalues=qvalues, common_Q0_inverse_half=inverse_half,
                  full_basis_source_norm_convention_difference=data['full_supply_Q0']-np.eye(128))
    metrics = {}
    for level in ('coarse', 'fine'):
        H, Y = own[level+'_loss_Gram'], own[level+'_loss_action']
        if H.shape != (16, 16) or Y.shape != (128, 16) or not np.all(np.isfinite(H)) or not np.all(np.isfinite(Y)):
            raise ValueError('Malformed independently produced loss arrays')
        Braw = U.conj().T@Y
        B = herm(Braw)
        R = Y-U@Braw
        residual = Y-U*lam[None]
        rayleigh_residual = Y-U*np.diag(Braw).real[None]
        raw_values = np.linalg.eigvalsh(herm(H))
        normalized_values = np.linalg.eigvalsh(herm(inverse_half@herm(H)@inverse_half))
        difference = H-target
        spectral_part, offspan_part = norm2(B-np.diag(lam)), norm2(R)
        arrays.update({level+'_B_raw': Braw, level+'_B_Hermitian_part': B, level+'_off_span': R,
                       level+'_full_eigenpair_residual': residual, level+'_Rayleigh_residual': rayleigh_residual,
                       level+'_source_Gram_difference': difference, level+'_projection_closure': Braw-H,
                       level+'_raw_spectrum': raw_values, level+'_normalized_spectrum': normalized_values,
                       level+'_raw_spectrum_difference': raw_values-target_raw_spectrum,
                       level+'_normalized_spectrum_difference': normalized_values-target_normalized_spectrum})
        metrics[level] = {'beta': spectral_part+offspan_part, 'within_selected_block_norm2': spectral_part,
                          'off_span_norm2': offspan_part, 'full_eigenpair_residual_norm2': norm2(residual),
                          'Gram_anti_Hermitian_maximum_entry': maximum(H-H.conj().T),
                          'B_anti_Hermitian_maximum_entry': maximum(Braw-Braw.conj().T),
                          'projection_closure_maximum_entry': maximum(Braw-H),
                          'Gram_source_difference_maximum_entry': maximum(difference),
                          'raw_spectrum_source_maximum_difference': maximum(raw_values-target_raw_spectrum),
                          'normalized_spectrum_source_maximum_difference': maximum(normalized_values-target_normalized_spectrum),
                          'own_trace': float(np.trace(H).real), 'source_trace': float(np.trace(target).real)}
        check(level+'_projection_closure', maximum(Braw-H), 1e-8)
        check(level+'_raw_Gram_spectrum_bounds', max(0., -float(raw_values[0]), float(raw_values[-1])-1), 1e-10)
        if level == 'fine':
            check('full_spectral_block_beta', spectral_part+offspan_part, 1e-4)
            check('source_compressed_Gram', maximum(difference), 1e-4)
            check('source_raw_compressed_spectrum', maximum(raw_values-target_raw_spectrum), 1e-4)
    check('loss_Gram_refinement', maximum(own['fine_loss_Gram']-own['coarse_loss_Gram']), 1e-8)
    check('full_action_refinement_norm2', norm2(own['fine_loss_action']-own['coarse_loss_action']), 1e-8)
    check('raw_spectrum_refinement', maximum(arrays['fine_raw_spectrum']-arrays['coarse_raw_spectrum']), 1e-8)
    check('normalized_spectrum_refinement', maximum(arrays['fine_normalized_spectrum']-arrays['coarse_normalized_spectrum']), 1e-8)
    residual = arrays['fine_full_eigenpair_residual']
    rayleigh = arrays['fine_Rayleigh_residual']
    rows = [{'source_rank_zero_based': int(ranks[j]), 'source_p': float(lam[j]),
             'own_delivered_weight': float(own['fine_loss_Gram'][j, j].real),
             'own_minus_source': float(own['fine_loss_Gram'][j, j].real-lam[j]),
             'initial_norm_squared': float(Q0[j, j].real),
             'normalized_delivered_fraction_diagnostic': float(own['fine_loss_Gram'][j, j].real/Q0[j, j].real),
             'full_eigenpair_residual_column_norm2': float(np.linalg.norm(residual[:, j])),
             'own_Rayleigh_residual_column_norm2': float(np.linalg.norm(rayleigh[:, j]))} for j in range(16)]
    artifact = args.outdir/'comparison_arrays.npz'
    np.savez(artifact, **arrays)
    result = {'runtime': runtime, 'source_binding': source, 'input_summary_binding': input_binding,
              'production_summary_binding': production_binding, 'input_artifact_binding': ia,
              'production_artifact_binding': pa, 'artifact': identity(artifact), 'checks': checks,
              'all_passed': bool(production['all_passed']) and all(row['passed'] for row in checks),
              'producer_numerical_gates_passed': bool(production['all_passed']), 'metrics': metrics, 'rows': rows,
              'T_over_L': float(data['source_T'][0]/data['L']), 'T': float(data['source_T'][0]),
              'source_full_spectrum_context': inputs['source_full_spectrum_summary'],
              'source_symmetry': inputs['source_symmetry'],
              'full_basis_Q_minus_I_maximum_entry': maximum(data['full_supply_Q0']-np.eye(128)),
              'benchmark_Q_minus_I_maximum_entry': maximum(Q0-np.eye(16)),
              'scope': 'Sixteen fixed source spectral directions, full128-coordinate residual; own global ranks, full median and extrema unassigned.',
              'custody': 'All scientific artifacts remain on the permitted box; renderer may emit a lean judged Markdown table.'}
    for binding in (source, input_binding, production_binding, ia, pa):
        require_identity(binding['path'], binding['sha256'])
    verify_threads(runtime, 'after_saved_comparison')
    result['all_source_and_input_identities_rechecked_unchanged'] = True
    (args.outdir/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'all_passed': result['all_passed'], 'beta': metrics['fine']['beta'], 'rows': len(rows)}), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
