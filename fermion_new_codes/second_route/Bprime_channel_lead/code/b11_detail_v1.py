"""Bounded saved B11 spectra/association and radial-coefficient evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from lead_runtime import admit, identity, require_identity, code_bindings, verify_threads, check_memory
from lead_prepare import recheck_artifacts, clean, save_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('b11', 'fields'):
        p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--outdir', type=Path, required=True)
    a = p.parse_args()
    runtime = admit(a.outdir, __file__, 'Saved B11 input-spectrum diagnosis', 1, 0, mode='dense', blas_threads=1)
    if any(a.outdir.iterdir()):
        raise ValueError('Fresh evidence directory required')
    bindings, records = {}, {}
    for name in ('b11', 'fields'):
        path = getattr(a, name)
        bindings[name] = require_identity(path, getattr(a, name+'_sha256'))
        records[name] = json.loads(path.read_text())
        if records[name]['stage'] != name or records[name]['bindings']['code'] != code_bindings():
            raise ValueError('Exact original input stage/source identities required')
        recheck_artifacts(records[name])
    if (not records['fields']['all_passed'] or
            records['b11']['bindings']['fields']['sha256'] != bindings['fields']['sha256']):
        raise ValueError('Accepted exact field-stage association required')
    import numpy as np
    from lead_run import write_arrays, require_storage
    verify_threads(runtime, 'after_imports')
    check_memory(runtime, 'saved_array_views', 2*1024**3)
    rows, arrays, coefficients = {}, {}, {}
    b11_archive = records['b11']['output_bindings']['arrays']
    field_archive = records['fields']['output_bindings']['arrays']
    for item in (b11_archive, field_archive):
        require_identity(item['path'], item['sha256'])
    names = ('energies', 'selected_indices', 'source_input_energy', 'source_column_map',
             'energy_multiset_representative_permutation', 'source_energy_feasible_mask',
             'source_energy_feasible_permutations', 'selected_cluster_membership', 'energy_difference')
    with np.load(b11_archive['path'], allow_pickle=False) as z:
        for family in ('F1', 'F2'):
            for kappa in (1, 2, 3):
                prefix = family+'_k'+str(kappa)
                row = {name: z[prefix+'_'+name] for name in names}
                own = row['energies'][row['selected_indices']]
                row['selected_own_energies'] = own
                row['signed_selected_own_minus_source'] = own[:, None]-row['source_input_energy'][None, :]
                row['selected_cluster_member_indices'] = [np.flatnonzero(mask) for mask in row['selected_cluster_membership']]
                row['spectrum_indices'] = np.arange(len(row['energies']))
                row['saved_case_status'] = records['b11']['families'][family]['cases'][str(kappa)]
                rows[prefix] = row
                for name, value in row.items():
                    if isinstance(value, np.ndarray):
                        arrays[prefix+'_'+name] = value
    with np.load(field_archive['path'], allow_pickle=False) as z:
        coefficients['radial_sites'] = z['radial_sites'][:17]
        for family in ('F1', 'F2'):
            for variant in ('P', 'H'):
                key = family+'_'+variant+'_radial_Nv'
                coefficients[key] = z[key][:17]
    arrays.update(coefficients)
    report = {'self': identity(__file__), 'bindings': bindings, 'source_code': code_bindings(),
              'original_B11_input_passed': records['b11']['all_passed'],
              'energy_gate': 1e-8, 'cluster_threshold': 1e-10, 'cases': rows,
              'radial_coefficient_rows': coefficients, 'runtime': runtime,
              'new_eigensolve': False, 'new_field_evaluation': False, 'new_assignment_fitted': False,
              'source_loss_opened': False,
              'unavailability': 'Source maps of -1 and NaN differences remain unavailable; no negative source index is used.'}
    require_storage(a.outdir, 2*sum(x.nbytes for x in arrays.values())+64*1024**2, runtime, 'saved_input_diagnosis')
    report['output_bindings'] = {'arrays': write_arrays(a.outdir/'saved_views.npz', arrays)}
    for value in records.values():
        recheck_artifacts(value)
    verify_threads(runtime, 'before_save')
    save_record(a.outdir/'evidence.json', report)
    lines = ['# Saved B11 input diagnosis', '',
             'No field evaluation, eigensolve, reassignment, tolerance change or source-loss read.', '',
             'Original aggregate B11 input pass: '+str(report['original_B11_input_passed'])+'.',
             'Frozen energy gate1e-8; selected-cluster threshold1e-10.', report['unavailability'], '',
             'Input record identities:', '```json', json.dumps(bindings, indent=2, sort_keys=True), '```',
             'Reader source SHA256: '+report['self']['sha256']+'.', '']
    for label, row in rows.items():
        lines += ['## '+label, '', '```json', json.dumps(clean(row), indent=2, sort_keys=True), '```', '']
    lines += ['## Saved radial coefficients', '', '```json', json.dumps(clean(coefficients), indent=2, sort_keys=True), '```', '']
    with (a.outdir/'TABLE.md').open('x') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
