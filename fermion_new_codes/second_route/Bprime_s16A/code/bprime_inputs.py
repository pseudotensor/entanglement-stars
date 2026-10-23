"""Own pinned-input admission and frozen spectral benchmark construction, on box only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from bprime_runtime import admit, core_bindings, identity, require_identity, verify_threads


RANKS = [0, 9, 18, 27, 36, 45, 54, 63, 64, 73, 82, 91, 100, 109, 118, 127]
RATIOS = [6.931471805599452, 69.31471805599453, 69.32]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--metadata-sha256', required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--controls-sha256', required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Own Bprime native input admission before spectral benchmark freeze', 16, 15)
    if any((args.outdir/name).exists() for name in ('inputs.json', 'inputs.npz')):
        raise ValueError('Input-admission outputs already exist; use a fresh directory')
    import numpy as np
    import bprime_operator as op
    import bprime_numerics as num
    verify_threads(runtime, 'after_imports')
    bindings = {'self': identity(__file__), 'spec': require_identity(args.spec, args.spec_sha256),
                'metadata': require_identity(args.metadata, args.metadata_sha256),
                'controls': require_identity(args.controls, args.controls_sha256), 'core': core_bindings()}
    metadata = json.loads(args.metadata.read_text())
    controls = json.loads(args.controls.read_text())
    if not controls['all_passed'] or controls['core_bindings'] != bindings['core']:
        raise ValueError('Manufactured controls do not admit the current exact core sources')
    if controls['execution']['spec_binding']['sha256'] != args.spec_sha256:
        raise ValueError('Manufactured controls have a different scientific specification')
    control_outputs = []
    def bind_outputs(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                require_identity(value['path'], value['sha256'])
                control_outputs.append(value)
            else:
                for item in value.values():
                    bind_outputs(item)
        elif isinstance(value, list):
            for item in value:
                bind_outputs(item)
    bind_outputs(controls['output_bindings'])
    if not control_outputs:
        raise ValueError('No raw manufactured artifacts are bound')
    pinned = metadata['source_identities']
    for item in pinned.values():
        require_identity(item['path'], item['sha256'])
    arrays, checks = {}, []
    result = {'runtime': runtime, 'bindings': bindings, 'source_identities': pinned, 'checks': checks,
              'benchmark_ranks': RANKS, 'source_consequence_read_after_native_admission': False,
              'custody': 'Original records and every scientific output remain on the permitted box.'}

    def check(name, passed, maximum=None, threshold=None):
        row = {'name': name, 'passed': bool(passed)}
        if maximum is not None:
            row['maximum'] = float(maximum)
        if threshold is not None:
            row['threshold'] = float(threshold)
        checks.append(row)

    def gate():
        if not all(row['passed'] for row in checks):
            raise ValueError('Admission failed: '+', '.join(row['name'] for row in checks if not row['passed']))

    def load(role, names):
        with np.load(pinned[role]['path'], allow_pickle=False) as archive:
            return {name: archive[name] for name in names}

    try:
        names = ('N', 'vx', 'vy', 'vz', 'eta', 'U_sub', 'e', 'H', 'Hs', 'L', 'T0', 'R_I', 'n_supply')
        full = load('prep_v5_full_npz', names)
        supply = load('prep_v5_supply_npz', names)
        exit_ = load('prep_v5_exit_npz', names)
        H, Hs, L = int(full['H']), int(full['Hs']), float(full['L'])
        T0, R_I = float(full['T0']), float(full['R_I'])
        if (H, Hs, L, T0, R_I) != (40, 12, 16., .0049735919716217296, 9.246593046624408):
            raise ValueError('Configuration differs from the frozen actual-format contract')
        if full['U_sub'].shape != (62500, 256) or full['U_sub'].dtype != np.complex128:
            raise ValueError('Full basis dimensions/dtype differ')
        for label, block in (('supply', supply), ('exit', exit_)):
            for name in names:
                if name not in ('U_sub', 'e', 'n_supply'):
                    op.require_exact(full[name], block[name], label+' '+name)
            if block['U_sub'].shape != (62500, 128) or block['U_sub'].dtype != np.complex128:
                raise ValueError('Split basis dimensions/dtype differ: '+label)
        for name in ('N', 'vx', 'vy', 'vz', 'eta'):
            value = full[name]
            if value.dtype != np.float64 or value.shape != (81, 81, 81) or not np.all(np.isfinite(value)):
                raise ValueError('Invalid native field: '+name)
        if np.any(full['N'] <= 0) or np.any(full['eta'] < 0):
            raise ValueError('Nonpositive lapse or negative sink')
        for block in (full, supply, exit_):
            if block['e'].dtype != np.float64 or block['e'].shape != (block['U_sub'].shape[1],):
                raise ValueError('Malformed input energy array')
            if not np.all(np.isfinite(block['U_sub'])) or not np.all(np.isfinite(block['e'])):
                raise ValueError('Nonfinite basis or input energies')
        hashes = {}
        for index in range(256):
            key = op.canonical_identity(full['U_sub'][:, index])['sha256']
            if key in hashes:
                raise ValueError('Duplicate exact full-basis column')
            hashes[key] = index
        maps = []
        for label, block in (('supply', supply), ('exit', exit_)):
            mapping = []
            for index in range(128):
                key = op.canonical_identity(block['U_sub'][:, index])['sha256']
                if key not in hashes:
                    raise ValueError('Split basis column absent from full prep: '+label)
                target = hashes[key]
                op.require_exact(full['U_sub'][:, target], block['U_sub'][:, index], label+' column')
                if full['e'][target] != block['e'][index]:
                    raise ValueError('Split/full eigenvalue association differs')
                mapping.append(target)
            maps.append(mapping)
            arrays[label+'_full_column_map'] = np.asarray(mapping)
        check('split_maps_unique_disjoint_full256_cover', sorted(maps[0]+maps[1]) == list(range(256)))
        check('supply_open_energy_window', np.all((supply['e'] > 0) & (supply['e'] < T0)))
        check('exit_open_energy_window', np.all((exit_['e'] > -T0) & (exit_['e'] < 0)))
        result['canonical_basis_identities'] = {label: op.canonical_identity(block['U_sub']) for label, block in (('supply', supply), ('exit', exit_))}
        for label, prefix in (('supply', '37be42d2bf86'), ('exit', 'a1ed5f1c7e49')):
            check(label+'_reported_basis_MD5_prefix', result['canonical_basis_identities'][label]['md5'].startswith(prefix))
        x, y, z = np.meshgrid(np.arange(-H, H+1), np.arange(-H, H+1), np.arange(-H, H+1), indexing='ij')
        radius = np.sqrt(x*x+y*y+z*z)
        expected_eta = np.sin(np.pi/2*np.minimum(1., np.maximum(0., (radius-2*L)/(.5*L))))**2
        sink_difference = float(np.max(np.abs(full['eta']-expected_eta)))
        check('supplied_sink_matches_declared_ramp', sink_difference <= 1e-13, sink_difference, 1e-13)
        arrays['sink_profile_difference'] = full['eta']-expected_eta
        sl = slice(H-Hs, H+Hs+1)
        mask = radius[sl, sl, sl] < R_I
        support_margin = float(Hs-R_I)
        check('retained_mask_has_two_link_subcube_margin', support_margin > 2., support_margin)
        flow = np.asarray([full[name] for name in ('vx', 'vy', 'vz')])
        subop = op.NativeOperator(full['N'][sl, sl, sl], flow[:, sl, sl, sl], full['eta'][sl, sl, sl])
        for label, block in (('supply', supply), ('exit', exit_)):
            U = block['U_sub']
            states = np.ascontiguousarray(U.T.reshape(128, 4, 25, 25, 25))
            outside = np.linalg.norm((states*(~mask)[None, None]).reshape(128, -1), axis=1)
            residual = subop.h_apply(states*mask[None, None])*mask[None, None]-states*block['e'][:, None, None, None, None]
            norms = np.linalg.norm(residual.reshape(128, -1), axis=1)
            Q0 = U.conj().T@U
            gram_error = float(np.max(np.abs(Q0-np.eye(128))))
            arrays.update({label+'_Q0': Q0, label+'_e': block['e'], label+'_outside_norms': outside,
                           label+'_eigenpair_residuals': residual, label+'_eigenpair_residual_norms': norms,
                           label+'_scaled_eigenpair_residual_norms': norms/np.maximum(np.abs(block['e']), np.finfo(float).tiny)})
            check(label+'_initial_Gram', gram_error <= 1e-10, gram_error, 1e-10)
            check(label+'_support', np.max(outside) <= 1e-12, np.max(outside), 1e-12)
            check(label+'_retained_eigenpairs', np.max(norms) <= 1e-8, np.max(norms), 1e-8)
        Jsup = op.antiunitary(supply['U_sub'].T.reshape(128, 4, 25, 25, 25)).reshape(128, -1).T
        C = Jsup.conj().T@exit_['U_sub']
        span_residual = exit_['U_sub']-Jsup@C
        span_norms = np.linalg.norm(span_residual, axis=0)
        unitary_residual = C.conj().T@C-np.eye(128)
        arrays.update(pairing_C=C, pairing_span_residual=span_residual, pairing_span_column_norms=span_norms,
                      pairing_unitarity_residual=unitary_residual,
                      paired_energy_intertwiner=supply['e'][:, None]*C+C*exit_['e'][None])
        check('full_window_antiunitary_span_pairing', np.max(span_norms) <= 1e-9, np.max(span_norms), 1e-9)
        check('pairing_C_unitarity', np.max(np.abs(unitary_residual)) <= 1e-9, np.max(np.abs(unitary_residual)), 1e-9)
        native = op.NativeOperator(full['N'], flow, full['eta'])
        alpha, beta, uc = op.spin_matrices()
        algebra = np.asarray([uc@a.conj()@uc.conj().T-a for a in alpha]+[uc@beta.conj()@uc.conj().T+beta, uc@uc.conj()-np.eye(4)])
        arrays['canonical_symmetry_algebra'] = algebra
        check('all_spin_coefficient_symmetry_identities', np.max(np.abs(algebra)) <= 5e-12, np.max(np.abs(algebra)), 5e-12)
        # Deterministic full-box probes include faces; no foreign operator is used.
        probe = np.empty((2, 4, 81, 81, 81), dtype=np.complex128)
        for column in range(2):
            for spin in range(4):
                probe[column, spin] = np.sin((column+1)*x*.17+(spin+1)*y*.13+z*.11)+1j*np.cos(x*.07+(column+1)*y*.19+(spin+1)*z*.23)
        for label, difference in (('h', native.h_apply(op.antiunitary(probe))+op.antiunitary(native.h_apply(probe))),
                                  ('A', native.apply(op.antiunitary(probe))-op.antiunitary(native.apply(probe))),
                                  ('eta', full['eta'][None, None]*op.antiunitary(probe)-op.antiunitary(full['eta'][None, None]*probe))):
            maximum = float(np.max(np.abs(difference)))
            arrays['native_'+label+'_probe_residual'] = difference
            check('actual_native_'+label+'_antiunitary_probe', maximum <= 5e-12*max(1., native.B), maximum, 5e-12*max(1., native.B))
        result['native_operator_bound'] = {'B': native.B, 'norm_one': native.norm_one, 'norm_infinity': native.norm_infinity}
        result['unavailable_standing_quantities'] = ['Full-operator extrema, initial physical E_inv and collar flux have no declared targets in this window fixture; they are not passed.']
        verify_threads(runtime, 'after_native_input_checks')
        gate()
        result['native_admission_passed_before_source_Gram_read'] = True
        # Only now read saved source consequences to define transparent benchmark inputs.
        result['source_consequence_read_after_native_admission'] = True
        runs = [load(role, ('T', 'G', 'e')) for role in ('run_1_sup_npz', 'run_1_exit_npz')]
        T = op.exact_clock_association(runs[0]['T'], np.asarray(RATIOS), L)
        op.exact_clock_association(runs[1]['T'], np.asarray(RATIOS), L)
        for label, run, block in zip(('supply', 'exit'), runs, (supply, exit_)):
            op.require_exact(run['e'], block['e'], label+' run/prep energies')
            if run['G'].dtype != np.complex128 or run['G'].shape != (3, 128, 128) or not np.all(np.isfinite(run['G'])):
                raise ValueError('Malformed saved source bath Gram')
            anti = np.max(np.abs(run['G']-run['G'].conj().transpose(0, 2, 1)))
            result[label+'_source_Gram_anti_Hermitian_maximum_entry'] = float(anti)
            first_anti = float(np.max(np.abs(run['G'][0]-run['G'][0].conj().T)))
            check(label+'_source_first_Gram_Hermitian_defect', first_anti <= 1e-10, first_anti, 1e-10)
            arrays[label+'_source_G'] = run['G']
        symmetry_rows = []
        for index in range(3):
            gp, gm = runs[0]['G'][index], runs[1]['G'][index]
            difference = gm-C.conj().T@gp.conj()@C
            spectral = np.linalg.eigvalsh((gm+gm.conj().T)/2)-np.linalg.eigvalsh((gp+gp.conj().T)/2)
            arrays['source_Gram_pairing_difference_'+str(index)] = difference
            arrays['source_spectral_pairing_difference_'+str(index)] = spectral
            maximum = float(np.max(np.abs(difference)))
            check('source_full_Gram_pairing_clock_'+str(index), maximum <= 1e-9, maximum, 1e-9)
            symmetry_rows.append({'T_over_L': float(runs[0]['T'][index]/L), 'Gram_relation_maximum_entry': maximum,
                                  'spectrum_maximum_difference': float(np.max(np.abs(spectral)))})
        gate()
        G = runs[0]['G'][0]
        values, vectors = np.linalg.eigh((G+G.conj().T)/2)
        phase_pivots = np.argmax(np.abs(vectors), axis=0)
        for column, pivot in enumerate(phase_pivots):
            entry = vectors[pivot, column]
            vectors[:, column] *= entry.conjugate()/abs(entry)
        arrays['source_eigenvector_phase_pivots'] = phase_pivots
        arrays['source_full_eigenvectors'] = vectors
        arrays['source_full_eigendecomposition_residual'] = ((G+G.conj().T)/2)@vectors-vectors*values[None]
        arrays['source_full_eigenvector_unitarity_residual'] = vectors.conj().T@vectors-np.eye(128)
        U, lam = vectors[:, RANKS], values[RANKS]
        unitary_defect = float(np.max(np.abs(U.conj().T@U-np.eye(16))))
        source_residual = float(np.max(np.abs(((G+G.conj().T)/2)@U-U*lam[None])))
        check('frozen_source_spectral_vectors_unitarity', unitary_defect <= 5e-12, unitary_defect, 5e-12)
        check('frozen_source_spectral_eigenpairs', source_residual <= 5e-12, source_residual, 5e-12)
        gate()
        initial = op.embed_subcube(supply['U_sub']@U, H, Hs)
        Q0 = num.gram(initial)
        initial_rates = 2*num.gram(initial, full['eta'])
        check('sixteen_initial_sink_rates_exact_zero', np.max(np.abs(initial_rates)) == 0., np.max(np.abs(initial_rates)), 0.)
        check('benchmark_initial_Gram', np.max(np.abs(Q0-np.eye(16))) <= 1e-10, np.max(np.abs(Q0-np.eye(16))), 1e-10)
        arrays.update(lapse=full['N'], flow=flow, eta=full['eta'], initial_columns=initial,
                      supply_subcube=supply['U_sub'], full_supply_Q0=arrays['supply_Q0'],
                      benchmark_U=U, benchmark_lambda=lam, benchmark_ranks=np.asarray(RANKS),
                      source_T=runs[0]['T'], initial_Q0=Q0, initial_sink_rates=initial_rates,
                      source_norm_convention_correction=arrays['supply_Q0']-np.eye(128),
                      source_full_spectrum=values, T=np.asarray(T), L=np.asarray(L), H=np.asarray(H), Hs=np.asarray(Hs),
                      mass=np.asarray(0.), spacing=np.asarray(1.))
        result.update(source_consequence_read_after_native_admission=True, source_symmetry=symmetry_rows,
                      eigensolver={'call': 'numpy.linalg.eigh of explicitly labelled Hermitian part, ascending', 'numpy_version': np.__version__,
                                   'phase_rule': 'Largest-magnitude component real nonnegative; lowest index wins exact tie.'},
                      canonical_benchmark_identity=op.canonical_identity(initial),
                      source_full_spectrum_summary={'minimum': float(values[0]), 'median': float(np.median(values)), 'maximum': float(values[-1])},
                      benchmark_source_values=lam.tolist(),
                      benchmark_selection='Fixed ascending source spectral ranks; source eigenvectors are validation inputs, never a solver or propagated state.',
                      initial_sink_rates_maximum_entry=float(np.max(np.abs(initial_rates))))
        gate()
    except Exception as error:
        result.update(exception=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for item in pinned.values():
            require_identity(item['path'], item['sha256'])
        for key in ('self', 'spec', 'metadata', 'controls'):
            require_identity(bindings[key]['path'], bindings[key]['sha256'])
        for item in control_outputs:
            require_identity(item['path'], item['sha256'])
        if core_bindings() != bindings['core']:
            raise RuntimeError('Core sources changed during native input admission')
        verify_threads(runtime, 'after_input_admission')
        result['all_source_and_code_identities_rechecked_unchanged'] = True
        result['all_passed'] = bool(checks) and all(row['passed'] for row in checks) and 'exception' not in result
        result['failed_checks'] = [row['name'] for row in checks if not row['passed']]
        artifact = args.outdir/'inputs.npz'
        np.savez(artifact, **arrays)
        result['artifact'] = identity(artifact)
        (args.outdir/'inputs.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'all_passed': result['all_passed'], 'failed_checks': result['failed_checks'], 'error': result.get('error'), 'artifact': str(artifact)}), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
