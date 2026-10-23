"""Own Bprime four-action producer and separate short CPU cost benchmark.

All numerical imports follow runtime admission. Original scientific inputs are
identified by their admitted records; this program loads the own frozen input
artifact, never a foreign operator. No source loss comparison is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from bprime_runtime import (admit, check_memory, core_bindings, identity,
                            require_identity, verify_threads)


SPEC_SHA256 = 'ca1071223753f58cd8a1afe664e6369af69a0f803808fcf54e71f1351a24d755'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def bind_nested(value, bindings):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            actual = require_identity(value['path'], value['sha256'])
            bindings[actual['path']] = actual
        else:
            for child in value.values():
                bind_nested(child, bindings)
    elif isinstance(value, list):
        for child in value:
            bind_nested(child, bindings)


def scalar_record(value, arrays, prefix, np):
    """Keep all arrays in NPZ and leave precise keys in the small JSON record."""
    if isinstance(value, np.ndarray):
        arrays[prefix] = value
        return {'array_key': prefix, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.generic):
        return scalar_record(value.item(), arrays, prefix, np)
    if isinstance(value, dict):
        return {str(key): scalar_record(child, arrays, prefix+'__'+str(key), np) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_record(child, arrays, prefix+'__'+str(index), np) for index, child in enumerate(value)]
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    return value


def run(args, runtime):
    import numpy as np
    import numba
    import bprime_operator as op
    import bprime_numerics as num
    verify_threads(runtime, 'after_numerical_imports')
    # The fixed native artifact has sixteen 4*81^3 complex128 columns. Allow
    # every retained endpoint plus the eight quadrature accumulators and work.
    check_memory(runtime, 'before_native_artifact', workspace_bytes=24*16*4*81**3*16+1024**3)
    bindings = {}
    for path, expected in ((args.spec, SPEC_SHA256), (args.inputs, args.inputs_sha256)):
        actual = require_identity(path, expected)
        bindings[actual['path']] = actual
    own_source, core = identity(__file__), core_bindings()
    bindings[own_source['path']] = own_source
    bind_nested(core, bindings)
    admitted = json.loads(args.inputs.read_text())
    if admitted.get('all_passed') is not True:
        raise ValueError('Passing native input admission is required')
    if admitted['bindings']['spec']['sha256'] != SPEC_SHA256 or admitted['bindings']['core'] != core:
        raise ValueError('Native input admission does not bind the frozen spec and current core')
    bind_nested(admitted['bindings'], bindings)
    bind_nested(admitted['source_identities'], bindings)
    control = admitted['bindings']['controls']
    controls = json.loads(Path(control['path']).read_text())
    if controls.get('all_passed') is not True or controls.get('core_bindings') != core:
        raise ValueError('Passing controls for this exact immutable core are required')
    bind_nested(controls['output_bindings'], bindings)
    artifact = require_identity(admitted['artifact']['path'], admitted['artifact']['sha256'])
    bindings[artifact['path']] = artifact
    with np.load(artifact['path'], allow_pickle=False) as archive:
        keys = ('lapse', 'flow', 'eta', 'initial_columns', 'supply_subcube', 'full_supply_Q0',
                'benchmark_U', 'benchmark_lambda', 'benchmark_ranks', 'initial_Q0', 'T', 'L', 'H', 'Hs', 'mass', 'spacing')
        data = {key: archive[key] for key in keys}
    H, Hs, T = int(data['H']), int(data['Hs']), float(data['T'])
    if (H, Hs, float(data['L']), float(data['mass']), float(data['spacing'])) != (40, 12, 16., 0., 1.):
        raise ValueError('Native input configuration differs from fixed contract')
    if T != 6.931471805599452*16:
        raise ValueError('Native endpoint differs from exact admitted first clock')
    initial = data['initial_columns']
    if initial.dtype != np.complex128 or initial.shape != (16, 4, 81, 81, 81) or not np.all(np.isfinite(initial)):
        raise ValueError('Invalid sixteen-column physical input block')
    operator = op.NativeOperator(data['lapse'], data['flow'], data['eta'], float(data['mass']), float(data['spacing']))
    if operator.B != admitted['native_operator_bound']['B']:
        raise ValueError('Reassembled own generator bound differs from native admission')
    Q0 = num.gram(initial)
    op.require_exact(Q0, data['initial_Q0'], 'initial Gram artifact versus actual initial columns')
    reference_norm_bound = float(np.sqrt(np.trace(data['full_supply_Q0']).real))
    plans = num.protocol_plans(operator, initial, T, reference_norm_bound)
    output = {'runtime': runtime, 'spec_binding': bindings[str(args.spec)],
              'producer_binding': own_source, 'core_bindings': core,
              'input_summary_binding': bindings[str(args.inputs)], 'input_artifact_binding': artifact,
              'mode': 'benchmark' if args.benchmark else 'four_action_production',
              'native_operator_bound': {'B': operator.B, 'norm_one': operator.norm_one, 'norm_infinity': operator.norm_infinity},
              'benchmark_ranks': data['benchmark_ranks'].tolist(),
              'endpoint': T, 'L': float(data['L']),
              'reference_norm_bound': reference_norm_bound,
              'reference_norm_convention': 'Full supplied basis Frobenius norm, a conservative spectral-norm bound.',
              'numba': {'version': numba.__version__, 'workers': numba.get_num_threads()},
              'actions': {}, 'custody': 'Every numerical record, endpoint, array and log remains on the box.',
              'source_comparison': 'Not performed by this producer; separate saved-output comparator follows.'}
    output_arrays = {}
    output['preflight'] = scalar_record(plans, output_arrays, 'preflight', np)
    progress_file = args.outdir/'progress.jsonl'

    def progress(action, index, count, clock, elapsed):
        row = {'action': action, 'step': index, 'total_steps': count, 'clock': float(clock), 'elapsed_seconds': elapsed}
        with progress_file.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if index == 1 or index == count or index % 16 == 0:
            print(action+': '+str(index)+'/'+str(count)+' macro-steps saved.', flush=True)

    folder = args.outdir/'actions'
    folder.mkdir()

    def after_action(name, result):
        endpoint_path, array_path = folder/(name+'_columns.npy'), folder/(name+'_diagnostics.npz')
        np.save(endpoint_path, result['final_columns'], allow_pickle=False)
        arrays = {}
        record = scalar_record({key: value for key, value in result.items() if key != 'final_columns'}, arrays, name, np)
        np.savez(array_path, **arrays)
        output['actions'][name] = {'endpoint': identity(endpoint_path), 'arrays': identity(array_path), 'record': record}
        verify_threads(runtime, 'after_'+name)
        check_memory(runtime, 'after_'+name)
        write_json(args.outdir/'producer_partial.json', output)

    if args.benchmark:
        started = time.perf_counter()
        operator.apply(initial)  # Compile/warm the exact native block shape.
        compile_warm_seconds = time.perf_counter()-started
        kernel_times = []
        for _ in range(2):
            started = time.perf_counter()
            operator.apply(initial)
            kernel_times.append(time.perf_counter()-started)
        delta = float(plans['coarse']['intervals'][0])
        started = time.perf_counter()
        coarse_endpoint, increment, samples = num.taylor_step(operator, initial, delta, integrate=True)
        integrated_step_seconds = time.perf_counter()-started
        started = time.perf_counter()
        integrated_norm_Gram = num.gram(coarse_endpoint)
        integrated_Gram_seconds = time.perf_counter()-started
        started = time.perf_counter()
        plain_endpoint, _, _ = num.taylor_step(operator, initial, delta, integrate=False)
        plain_step_seconds = time.perf_counter()-started
        started = time.perf_counter()
        plain_norm_Gram = num.gram(plain_endpoint)
        plain_Gram_seconds = time.perf_counter()-started
        started = time.perf_counter()
        np.save(args.outdir/'benchmark_integrated_endpoint.npy', coarse_endpoint, allow_pickle=False)
        np.save(args.outdir/'benchmark_plain_endpoint.npy', plain_endpoint, allow_pickle=False)
        two_endpoint_write_seconds = time.perf_counter()-started
        output_arrays.update(initial_Q0=Q0, integrated_step_norm_Gram=integrated_norm_Gram,
                             plain_step_norm_Gram=plain_norm_Gram, step_sink_Gram_increment=increment,
                             gauss_sink_Gram_samples=samples, kernel_elapsed_seconds=np.asarray(kernel_times))
        output['benchmark'] = {'compile_warm_seconds': compile_warm_seconds,
                               'kernel_times_seconds': kernel_times, 'integrated_macro_step_seconds': integrated_step_seconds,
                               'plain_macro_step_seconds': plain_step_seconds, 'macro_step_interval': delta,
                               'integrated_remaining_norm_Gram_seconds': integrated_Gram_seconds,
                               'plain_remaining_norm_Gram_seconds': plain_Gram_seconds,
                               'integrated_full_macro_step_seconds': integrated_step_seconds+integrated_Gram_seconds,
                               'plain_full_macro_step_seconds': plain_step_seconds+plain_Gram_seconds,
                               'two_endpoint_write_seconds': two_endpoint_write_seconds,
                               'coarse_steps': plans['coarse']['steps'], 'fine_steps': plans['fine']['steps'],
                               'estimated_four_action_seconds': plans['coarse']['steps']*(integrated_step_seconds+integrated_Gram_seconds+5*(plain_step_seconds+plain_Gram_seconds))+2*two_endpoint_write_seconds,
                               'estimated_remaining_norm_Gram_contractions': 6*plans['coarse']['steps'],
                               'estimate_exclusions': 'Startup, input/output identity hashes, small-matrix reductions, per-step log and diagnostics-file I/O; measured endpoint write cost is scaled to four endpoints.',
                               'benchmark_generator_applications': 67,
                               'workspace_input_block_bytes': initial.nbytes,
                               'scope': 'Measured short CPU cost only; no target endpoint result or source agreement claim.'}
        output['all_passed'] = True
        artifact_name = 'benchmark_arrays.npz'
    else:
        actions = num.run_actions(operator, initial, T, reference_norm_bound, after_action=after_action, progress=progress)
        Ys, projections = {}, {}
        for label in ('coarse', 'fine'):
            returned = op.extract_subcube(actions[label+'_adjoint']['final_columns'], H, Hs)
            Ys[label] = num.project_action(data['full_supply_Q0'], data['benchmark_U'], data['supply_subcube'], returned)
            projections[label] = num.projection_diagnostics(Ys[label], data['benchmark_U'], actions[label+'_forward']['loss_Gram'])
            output_arrays[label+'_loss_Gram'] = actions[label+'_forward']['loss_Gram']
            output_arrays[label+'_loss_action'] = Ys[label]
            output_arrays[label+'_B_raw'] = projections[label]['B_raw']
            output_arrays[label+'_R'] = projections[label]['R']
        refinement = num.refinement_diagnostics(actions['coarse_forward']['loss_Gram'], actions['fine_forward']['loss_Gram'], Ys['coarse'], Ys['fine'], Q0)
        integral_closure = actions['coarse_forward']['integral_closure_maximum_entry']
        projection_pass = all(value['projection_closure_maximum_entry'] <= 1e-8 for value in projections.values())
        eigenvalue_failures = {}
        for label in ('coarse', 'fine'):
            current = refinement[label+'_spectrum']
            eigenvalue_failures[label] = {key: np.flatnonzero((current[key] < -1e-10) | (current[key] > 1+1e-10)).tolist()
                                         for key in ('raw_Hermitian_part_eigenvalues', 'orthonormal_representation_eigenvalues')}
        eigenvalue_pass = all(not indices for value in eigenvalue_failures.values() for indices in value.values())
        output_arrays.update(initial_Q0=Q0, benchmark_U=data['benchmark_U'], benchmark_lambda=data['benchmark_lambda'],
                             benchmark_ranks=data['benchmark_ranks'], initial_column_norms=np.sqrt(np.real(np.diag(Q0))),
                             initial_Q0_minus_identity=Q0-np.eye(len(initial)))
        output['convergence'] = {
            'refinement': scalar_record(refinement, output_arrays, 'refinement', np),
            'projection': scalar_record(projections, output_arrays, 'projection', np),
            'coarse_sink_integral_closure_maximum_entry': integral_closure,
            'coarse_sink_integral_closure_pass': integral_closure <= 1e-8,
            'forward_adjoint_projection_closure_pass': projection_pass,
            'eigenvalues_outside_declared_bounds': eigenvalue_failures, 'eigenvalue_bounds_pass': eigenvalue_pass}
        output['all_passed'] = bool(refinement['all_passed'] and integral_closure <= 1e-8 and projection_pass and eigenvalue_pass)
        artifact_name = 'own_arrays.npz'
    for path, expected in bindings.items():
        require_identity(path, expected['sha256'])
    if core_bindings() != core:
        raise RuntimeError('Frozen numerical core changed during producer')
    verify_threads(runtime, 'at_completion')
    check_memory(runtime, 'at_completion')
    output['numba']['threading_layer'] = numba.threading_layer()
    output['all_input_and_source_identities_rechecked_unchanged'] = True
    artifact_path = args.outdir/artifact_name
    np.savez(artifact_path, **output_arrays)
    output['artifact'] = identity(artifact_path)
    output['output_bindings'] = {str(path.relative_to(args.outdir)): identity(path)
                                 for path in sorted(args.outdir.rglob('*')) if path.is_file() and path != args.outdir/'summary.json'}
    write_json(args.outdir/'summary.json', output)
    print('Saved own '+output['mode']+'; all numerical gates passed='+str(output['all_passed']), flush=True)
    if not output['all_passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Own Bprime CPU cost benchmark' if args.benchmark else 'Own Bprime coupled Taylor four-action producer', 16, 15)
    if any(args.outdir.iterdir()):
        raise RuntimeError('Preserve existing producer outputs; use a fresh empty directory')
    try:
        run(args, runtime)
    except Exception as error:
        write_json(args.outdir/'producer_failure.json', {'runtime': runtime, 'status': 'incomplete',
                                                        'exception': type(error).__name__, 'message': str(error),
                                                        'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
