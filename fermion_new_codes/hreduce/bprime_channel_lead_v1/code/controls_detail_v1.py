"""Saved fixture differences, nonzero current algebra and runtime identities."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from lead_runtime import admit, require_identity, identity, code_bindings, verify_threads, check_memory
from lead_prepare import save_record, recheck_artifacts, clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--controls-sha256', required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--comparison-sha256', required=True)
    p.add_argument('--outdir', type=Path, required=True)
    a = p.parse_args()
    runtime = admit(a.outdir, __file__, 'Requested saved manufactured evidence', 1, 0, mode='dense', blas_threads=1)
    if any(a.outdir.iterdir()):
        raise ValueError('Fresh evidence directory required')
    control_binding = require_identity(a.controls, a.controls_sha256)
    comparison_binding = require_identity(a.comparison, a.comparison_sha256)
    controls, comparison = json.loads(a.controls.read_text()), json.loads(a.comparison.read_text())
    if controls['bindings']['code'] != code_bindings() or comparison['code_bindings'] != code_bindings():
        raise ValueError('Saved evidence source identities differ')
    recheck_artifacts(controls)
    recheck_artifacts(comparison)
    import numpy as np
    from lead_read import residual_summary
    from lead_run import write_arrays, require_storage
    verify_threads(runtime, 'after_imports')
    check_memory(runtime, 'saved_fixture_views', 2*1024**3)
    cases = {c['name']: c for c in controls['cases']}
    raw = controls['output_bindings']['controls_arrays.npz']
    rows, arrays = [], {}
    with np.load(raw['path'], allow_pickle=False) as z:
        def value(case, side, name):
            return z[cases[case][side][name]['array_key']]
        pairs = (
            ('native_H', 'native_open_variable_bonds_and_eight_valleys', 'H_apply', 'H_dense'),
            ('hybrid_H', 'actual_four_shell_completed_hybrid_apply', 'H_apply', 'H_dense'),
            ('complete_exterior_state', 'complete_native_exterior_unitary_coordinates', 'final_restored', 'final'),
            ('angular_state', 'actual_angular_rank_whitening_PQ_and_valley_dynamics', 'final', 'dense_final'))
        for label, case, observed, expected in pairs:
            delta = value(case, 'observed', observed)-value(case, 'expected', expected)
            arrays[label] = delta
            rows.append({'difference': label, 'case': case, 'observed': observed, 'expected': expected,
                         'summary': residual_summary(delta), 'original_case_passed': cases[case]['passed']})
        E = value('actual_angular_rank_whitening_PQ_and_valley_dynamics', 'observed', 'E')
        arrays['angular_EdaggerE_minus_I'] = E.conj().T@E-np.eye(E.shape[1])
        rows.append({'difference': 'angular_EdaggerE_minus_I',
                     'summary': residual_summary(arrays['angular_EdaggerE_minus_I'])})
        fixed = 'fixed_degree_refinement_bound_guards_batching_and_end_return'
        H, F = value(fixed, 'inputs', 'H'), value(fixed, 'inputs', 'F')
        expected_H = np.array([[.25, .5+.125j], [.5-.125j, -.125]], dtype=complex)
        expected_F = np.array([[1., .5j], [.25j, 1.]], dtype=complex)
        if not np.array_equal(H, expected_H) or not np.array_equal(F, expected_F):
            raise ValueError('The scientist-fixed saved current fixture changed')
        from lead_native import DenseHamiltonian
        from lead_operator import projector_current
        from lead_run import projector_fluxes
        from lead_numerics import block_norm, matrix_norm
        operator = DenseHamiltonian(H)
        mask = np.array([True, False])
        C = np.array([[.5, 9/64-7j/16], [9/64+7j/16, -.25]])
        energy_C = C/16
        HF = operator.h_apply(F)
        H2F = operator.h_apply(HF)
        actual_fluxes = projector_fluxes(F, HF, H2F, {'fixed_cut': mask})['fixed_cut']
        actuals = {'operator_probability': projector_current(operator, F, mask),
                   'operator_energy': projector_current(operator, F, mask, energy=True),
                   'saved_clock_probability': actual_fluxes['probability_current'],
                   'saved_clock_energy': actual_fluxes['symmetric_energy_current']}
        target = 5e-12*max(1., operator.B*block_norm(F))
        supplement_checks = {'nonzero_exact_energy_current': matrix_norm(energy_C) > 1e-3}
        arrays.update(current_H=H, current_F=F, current_mask=mask,
                      exact_probability_current=C, exact_energy_current=energy_C)
        for label, actual in actuals.items():
            exact = energy_C if label.endswith('energy') else C
            delta = actual-exact
            arrays[label+'_actual'], arrays[label+'_difference'] = actual, delta
            supplement_checks[label] = matrix_norm(delta) <= target
            rows.append({'difference': label, 'summary': residual_summary(delta),
                         'exact': exact, 'actual': actual, 'target': target,
                         'operator_norm': matrix_norm(delta), 'passed': supplement_checks[label]})
    original_runtime = controls['runtime']
    memory = {k: v for k, v in original_runtime.items() if k.startswith('memory_')}
    resource = {'minimum_recorded_MemAvailable_bytes': min(v['available_bytes'] for v in memory.values()),
                'root_filesystem_available_bytes_at_admission': original_runtime['root_filesystem_available_bytes_at_admission'],
                'resident_thread_observations': {k: v for k, v in original_runtime.items() if k.startswith('resident_threads_')},
                'recorded_memory_guards': memory}
    result = {'self': identity(__file__), 'controls': control_binding, 'comparison': comparison_binding,
              'runtime': runtime, 'original_resources': resource, 'rows': rows,
              'current_supplement_checks': supplement_checks,
              'current_supplement_passed': all(supplement_checks.values()),
              'original_outputs': controls['output_bindings'], 'renderer_outputs': comparison['output_bindings'],
              'saved_fixtures_only': True, 'new_physical_evaluation': False,
              'new_evolution': False, 'code_bindings': code_bindings()}
    require_storage(a.outdir, 2*sum(x.nbytes for x in arrays.values())+64*1024**2, runtime, 'bounded_fixture_differences')
    result['output_bindings'] = {'arrays': write_arrays(a.outdir/'differences.npz', arrays)}
    recheck_artifacts(controls)
    recheck_artifacts(comparison)
    recheck_artifacts(result)
    verify_threads(runtime, 'before_save')
    save_record(a.outdir/'evidence.json', result)
    lines = ['# Requested saved manufactured evidence', '', 'No new evolution, field solve or physical input.', '',
             '| Difference | Exact saved-array summary |', '|---|---|']
    for row in rows:
        lines.append('| '+row['difference']+' | '+json.dumps(clean(row['summary']), sort_keys=True)+' |')
    lines += ['', 'Nonzero current supplement (existing saved H/F, no propagation):', '',
              '```json', json.dumps(clean({'checks': supplement_checks,
                  'target': target, 'H': H, 'F': F, 'mask': mask,
                  'exact_probability_current': C, 'exact_energy_current': energy_C,
                  'actual_matrices': actuals}), indent=2, sort_keys=True), '```']
    lines += ['', 'Original resource observations:', '', '```json', json.dumps(clean(resource), indent=2, sort_keys=True), '```',
              '', 'Bound control and renderer output identities:', '', '```json',
              json.dumps({'controls': controls['output_bindings'], 'renderer': comparison['output_bindings']}, indent=2, sort_keys=True), '```',
              '', 'This table source SHA256: '+result['self']['sha256']+'.',
              'Controls record SHA256: '+control_binding['sha256']+'.',
              'Saved comparison record SHA256: '+comparison_binding['sha256']+'.', '']
    with (a.outdir/'TABLE.md').open('x') as f:
        f.write('\n'.join(lines))
    return 0 if all(supplement_checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
