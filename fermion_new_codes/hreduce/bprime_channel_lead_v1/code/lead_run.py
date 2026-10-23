"""Box-only staged channel-lead setup, benchmark, propagation and references.

All scientific stages require separate actual-source/output review. CLI
availability is not admission to skip the specification's cost/result gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import sys
import time
import traceback
from lead_runtime import (admit, identity, require_identity, code_bindings,
                          check_memory, verify_threads, SPEC_SHA256, DERIVATION_SHA256)
from lead_prepare import (save_record, clean, admitted_record, artifact_bindings,
                          recheck_artifacts)

T_FINAL = 6.931471805599452*16
CASE_K_LENGTH = {'pilot-k10-200': (10, 3200), 'pilot-k12-200': (12, 3200),
                 'pilot-k10-400': (10, 6400), 'pilot-k10-200-fine': (10, 3200),
                 'flat-supply': (10, 3200), 'flat-manufactured': (10, 3200),
                 'variable': (10, 3200), 'variable-fine': (10, 3200)}


def array_identity(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': value.dtype.str,
            'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def write_arrays(path, arrays):
    import numpy as np
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)
    return identity(path)


def write_state(path, states):
    """Lossless full-state encoding; every complex128 component is retained."""
    import numpy as np
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, states=states)
    return identity(path)


def load_state(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return archive['states']


def require_storage(outdir, bytes_needed, runtime, phase):
    filesystem = os.statvfs(outdir)
    available = filesystem.f_bavail*filesystem.f_frsize
    root = os.statvfs('/')
    root_free = root.f_bavail*root.f_frsize
    runtime['storage_'+phase] = {'available_bytes': available,
                                'required_new_bytes': int(bytes_needed), 'root_free_bytes': root_free}
    if available < bytes_needed+1024**3 or root_free < 100*10**9:
        raise ValueError('Insufficient prepriced output/root disk space: '+phase)


def peak_memory(runtime):
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    runtime['process_peak_RSS_bytes'] = peak
    return peak


def flatten_arrays(value):
    """Retain ragged shell arrays as separate arrays, never object pickles."""
    import numpy as np
    arrays = {}
    def walk(item, key):
        if isinstance(item, np.ndarray):
            arrays[key] = item
            return {'array_key': key, 'shape': list(item.shape), 'dtype': item.dtype.str}
        if isinstance(item, dict):
            return {str(k): walk(v, key+'__'+str(k)) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [walk(v, key+'__'+str(k)) for k, v in enumerate(item)]
        return item
    return walk(value, 'data'), arrays


def binding_record(path, sha, bindings, role):
    value, bindings[role] = admitted_record(path, sha)
    bindings[role+'_artifacts'] = artifact_bindings(value)
    recheck_artifacts(value)
    return value


def load_array_record(record):
    import numpy as np
    item = record['output_bindings']['arrays']
    require_identity(item['path'], item['sha256'])
    return np.load(item['path'], allow_pickle=False)


def native_ball_arrays(native_archive, field):
    import numpy as np
    given_n, given_v = native_archive['native_N'], native_archive['native_flow']
    if field == 'flat':
        given_n, given_v = np.ones_like(given_n), np.zeros_like(given_v)
    coords = np.arange(-40, 41)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    mask = x*x+y*y+z*z <= 1600
    return given_n, given_v, mask


def association(n, v, mask, radial):
    return {'native_N': array_identity(n), 'native_v': array_identity(v),
            'native_mask': array_identity(mask), 'radial_Nv': array_identity(radial),
            'mass': 0., 'spacing': 1.}


def manufactured_flat_inputs():
    import itertools
    import numpy as np
    from lead_channels import gamma_spinor
    from lead_native import valley_matrix
    coordinates = np.arange(-40, 41)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    xyz = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    radius = np.sqrt(np.sum(xyz**2, axis=1))
    support = (radius > 4)&(radius < 12)
    points, r = xyz[support], radius[support]
    envelope = np.sin(np.pi*(r-4)/8)**4/r
    columns, labels = [], []
    for nu in itertools.product((0, 1), repeat=3):
        phase = 1-2*((points@np.asarray(nu)) % 2)
        for k in (1, 3):
            sign = (-1)**sum(nu)
            spinor = gamma_spinor(sign*k, 2*k-1, np.array([1., 1j])/np.sqrt(2), points)
            spinor = (spinor@valley_matrix(nu).T)*(phase*envelope)[:, None]
            state = np.zeros((4, 81**3), dtype=np.complex128)
            state[:, support] = spinor.T
            state /= np.linalg.norm(state)
            columns.append(state.ravel())
            labels.append([*nu, k, sign])
    return np.array(columns), np.array(labels, dtype=np.int64)


def hybrid_inventory(operator, states):
    import numpy as np
    from lead_operator import cross_gram
    lead = operator.layout.lead_view(states)
    cells = lead.shape[2]
    odd = 41+2*np.arange(cells)
    near = odd+1 <= 800
    bins, limits, ranges = [], [], []
    for lower in range(800, operator.lead.end, 160):
        selected = (odd > lower)&(odd+1 <= lower+160)
        bins.append(cross_gram(lead[:, :, selected, :].reshape(len(states), -1)))
        limits.append([lower, min(lower+160, operator.lead.end)])
        indices = np.flatnonzero(selected)
        ranges.append([int(indices[0]), int(indices[-1]+1)] if len(indices) else [0, 0])
    near_gram = cross_gram(lead[:, :, near, :].reshape(len(states), -1))
    all_lead = cross_gram(lead.reshape(len(states), -1))
    sum_gram = near_gram+sum(bins, np.zeros_like(near_gram))
    return {'near_lead_Gram': near_gram, 'far_bin_Grams': np.array(bins),
            'far_bin_radial_limits': np.array(limits), 'far_bin_cell_ranges': np.array(ranges),
            'lead_partition_remainder_Gram': all_lead-sum_gram,
            'lead_odd_radii': odd, 'lead_chain_labels': operator.labels,
            'lead_amplitude_storage': 'Complex folded amplitudes are the lead view of the states array in the bound full-state lossless NPZ; complete-cell ranges define disjoint spatial bins.'}


def hybrid_projectors(operator):
    import numpy as np
    coordinates = np.arange(-40, 41)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    r2 = x*x+y*y+z*z
    masks = {}
    for name, spatial in (('core_r_le_32', r2 <= 32**2), ('ball_r_le_40', r2 <= 40**2)):
        projector = np.zeros(operator.layout.total_dofs, dtype=bool)
        projector[:operator.layout.native_size] = np.tile(spatial.ravel(), 4)
        masks[name] = projector
    projector = np.zeros(operator.layout.total_dofs, dtype=bool)
    lead_mask = np.zeros((operator.layout.chains, operator.layout.cells, 4), dtype=bool)
    lead_mask[:, 41+2*np.arange(operator.layout.cells) > 800, :] = True
    projector[operator.layout.native_size:] = lead_mask.ravel()
    masks['far_lead_r_gt_800'] = projector
    return masks


def projector_fluxes(states, hstates, h2states, masks):
    from lead_operator import cross_gram
    result = {}
    for name, projector in masks.items():
        c = cross_gram(states[:, projector], hstates[:, projector])
        c2 = cross_gram(states[:, projector], h2states[:, projector])
        result[name] = {'probability_current': 1j*(c.conj().T-c),
                        'symmetric_energy_current': .5j*(c2.conj().T-c2)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('setup-benchmark', 'setup', 'audit', 'apply-benchmark',
                                         'hybrid', 'fourier', 'native-reference', 'b11'))
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--derivation', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--controls-sha256', required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--native-sha256', required=True)
    parser.add_argument('--fields', type=Path)
    parser.add_argument('--fields-sha256')
    parser.add_argument('--setup', type=Path)
    parser.add_argument('--setup-sha256')
    parser.add_argument('--frame-source', type=Path)
    parser.add_argument('--frame-source-sha256')
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--benchmark-sha256')
    parser.add_argument('--b11-inputs', type=Path)
    parser.add_argument('--b11-inputs-sha256')
    parser.add_argument('--coarse', type=Path)
    parser.add_argument('--coarse-sha256')
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--audit-sha256')
    parser.add_argument('--end-radius', type=int, choices=(3200, 6400), default=3200)
    parser.add_argument('--field', choices=('F1', 'flat'), default='F1')
    parser.add_argument('--K', type=int, choices=(10, 12), default=10)
    parser.add_argument('--case', choices=tuple(CASE_K_LENGTH), default='pilot-k10-200')
    parser.add_argument('--family', choices=('F1', 'F2'), default='F1')
    parser.add_argument('--kappa', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--arm', choices=('cap2', 'lead200', 'lead400', 'cap50'), default='lead200')
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--column', type=int, default=0)
    parser.add_argument('--fourier-family', choices=('supply', 'manufactured'), default='supply')
    args = parser.parse_args()
    dense = args.stage in ('setup', 'setup-benchmark', 'audit')
    fft = args.stage == 'fourier'
    runtime = admit(args.outdir, __file__, 'Channel-lead '+args.stage,
                    16 if dense or not fft else 1, 0 if dense or fft else 15,
                    mode='dense' if dense or fft else 'mixed', blas_threads=8 if dense else 1)
    if any(args.outdir.iterdir()):
        raise ValueError('Fresh empty stage directory required')
    bindings = {'spec': require_identity(args.spec, SPEC_SHA256),
                'derivation': require_identity(args.derivation, DERIVATION_SHA256),
                'self': identity(__file__), 'code': code_bindings()}
    result = {'stage': args.stage, 'runtime': runtime, 'bindings': bindings,
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_consequences_opened': False, 'all_passed': False, 'output_bindings': {}}
    try:
        controls = binding_record(args.controls, args.controls_sha256, bindings, 'controls')
        if controls['code_bindings'] != bindings['code'] or controls['spec_binding']['sha256'] != SPEC_SHA256:
            raise ValueError('Exact complete manufactured source/specification required')
        native_record = binding_record(args.native, args.native_sha256, bindings, 'native')
        if args.fields is not None:
            fields_record = binding_record(args.fields, args.fields_sha256, bindings, 'fields')
        elif args.stage not in ('setup-benchmark', 'fourier'):
            raise ValueError('Passed actual radial-field preparation is required')
        else:
            fields_record = None
        import numpy as np
        from lead_native import NativeHamiltonian
        from lead_channels import RadialHamiltonian
        from lead_inputs import opnorm
        verify_threads(runtime, 'after_imports')
        with load_array_record(native_record) as archive:
            n, v, mask = native_ball_arrays(archive, args.field)
            pilot = archive['pilot_native']
            probes = archive['variable_probe_native']
        native = NativeHamiltonian(n, v, mask=mask)
        radial = None
        if fields_record is not None:
            with load_array_record(fields_record) as archive:
                radial = archive['F1_P_radial_Nv']
            if args.field == 'flat':
                radial = np.array([np.ones(len(radial)), np.zeros(len(radial))]).T
        field_association = association(n, v, mask, radial) if radial is not None else None
        result['field_association'] = field_association
        if args.stage in ('setup-benchmark', 'setup'):
            from lead_setup import benchmark_setup, build_setup
            if args.stage == 'setup-benchmark':
                setup_started = time.perf_counter()
                measured = benchmark_setup(native, args.K, args.outdir, runtime)
                result['setup_compute_seconds'] = time.perf_counter()-setup_started
                result['setup_benchmark'] = measured
                result['all_passed'] = True
            else:
                if args.benchmark is None or args.benchmark_sha256 is None:
                    raise ValueError('Measured setup benchmark and subsequent scientific cost gate required')
                benchmark = binding_record(args.benchmark, args.benchmark_sha256, bindings, 'setup_benchmark')
                if benchmark['arguments']['K'] != args.K:
                    raise ValueError('Setup benchmark cutoff differs')
                frame_source = None
                if args.frame_source is not None:
                    original = binding_record(args.frame_source, args.frame_source_sha256, bindings, 'frame_source')
                    frame_source = original['output_bindings']['setup']['path']
                from lead_setup import shell_indices
                shells = shell_indices()
                width = 32*args.K*(args.K+1)
                frame_bytes = 0 if frame_source is not None else 2*sum(len(shell['dofs']) for shell in shells)*width*16
                # Sixteen full diagnostic matrices per frame plus all raw,
                # realized, ideal and correction collar blocks is conservative.
                setup_bytes = frame_bytes+100*width*width*16
                require_storage(args.outdir, setup_bytes, runtime, 'full_setup')
                setup_started = time.perf_counter()
                measured = build_setup(native, args.K, radial[:, 0], radial[:, 1], args.outdir,
                                       runtime, frame_source=frame_source)
                result['setup_compute_seconds'] = time.perf_counter()-setup_started
                result['setup'] = measured
                result['all_passed'] = measured['all_passed']
                result['output_bindings']['setup'] = identity(args.outdir/'setup.json')
            result['output_bindings']['setup_artifacts'] = measured.get('output_bindings', {})
        elif args.stage == 'fourier':
            if args.field != 'flat' or args.K != 10 or not 0 <= args.column < 16:
                raise ValueError('Frozen flat Fourier roster only')
            from lead_references import embed_periodic, native_ball_observer, fourier_reference, free_image_bound
            initial, labels = manufactured_flat_inputs() if args.fourier_family == 'manufactured' else (pilot, np.arange(0, 128, 8))
            column = initial[args.column]
            cube = column.reshape(4, 81, 81, 81)
            occupied = np.any(initial.reshape(len(initial), 4, 81, 81, 81) != 0, axis=(0, 1))
            coords = np.array(np.nonzero(occupied)).T-40
            rin = float(np.max(np.abs(coords)))
            side = 256
            block_norm = opnorm(initial)
            certificate = free_image_bound(side, rin, T_FINAL, block_norm)
            while not certificate['passed']:
                side *= 2
                check_memory(runtime, 'Fourier_candidate_'+str(side), 16*side**3*4*6)
                require_storage(args.outdir, side**3*24+4*81**3*16*4, runtime, 'Fourier_candidate_'+str(side))
                certificate = free_image_bound(side, rin, T_FINAL, block_norm)
            check_memory(runtime, 'Fourier_buffers', 16*side**3*4*6)
            require_storage(args.outdir, side**3*24+4*81**3*16*4, runtime, 'Fourier_outputs')
            observe, sites = native_ball_observer(side)
            output_bindings = result['output_bindings']
            output_bindings['initial'] = write_arrays(args.outdir/'initial.npz', {
                'initial_native': column, 'label': np.asarray(labels[args.column]), 'ball_sites': sites})
            def spectrum(arrays):
                output_bindings['spectral_measure'] = write_arrays(args.outdir/'spectral_measure.npz', arrays)
            clocks = T_FINAL*np.arange(1, 5)/4
            saved = []
            for index, (clock, state, diagnostics) in enumerate(fourier_reference(embed_periodic(column, side), clocks, observe, spectrum)):
                output_bindings['clock_'+str(index)] = write_arrays(args.outdir/('clock_'+str(index)+'.npz'),
                                                                 {'native_ball': state, 'clock': np.asarray(clock)})
                saved.append(diagnostics)
                verify_threads(runtime, 'Fourier_clock_'+str(index))
            norm_pass = all(abs(row['periodic_norm']-row['initial_norm']) <= 1e-8*max(1., row['initial_norm']) for row in saved)
            result.update(certificate=certificate, diagnostics=saved, periodic_norm_pass=norm_pass,
                          all_passed=bool(norm_pass))
        else:
            # Coupled, radial, expanded-native and benchmark producers share
            # the same binding and checkpoint discipline below.
            return _evolution_stage(args, runtime, bindings, result, native_record, fields_record,
                                    native, radial, pilot, probes, field_association)
        verify_threads(runtime, 'at_completion')
        peak_memory(runtime)
        recheck_artifacts(result)
        save_record(args.outdir/'run.json', result)
        return 0 if result['all_passed'] else 1
    except Exception:
        result['failure'] = traceback.format_exc()
        try:
            recheck_artifacts(result)
        except Exception:
            result['final_identity_recheck_failure'] = traceback.format_exc()
        save_record(args.outdir/'run_failure.json', result)
        raise


def _evolution_stage(args, runtime, bindings, result, native_record, fields_record,
                     native, radial, pilot, probes, field_association):
    import numpy as np
    from lead_inputs import opnorm
    from lead_numerics import propagate, make_plan, bisect_plan, taylor_step, truncation_bound
    from lead_operator import cross_gram
    from lead_references import endpoint_bounds

    def storage(bytes_needed, phase):
        require_storage(args.outdir, bytes_needed, runtime, phase)

    def load_setup_operator(end_radius, cutoff):
        from lead_operator import HybridOperator
        if args.setup is None or args.setup_sha256 is None:
            raise ValueError('Bound coupled setup report required')
        setup_record = binding_record(args.setup, args.setup_sha256, bindings, 'setup')
        if (setup_record['field_association'] != field_association or
                setup_record['arguments']['K'] != cutoff or
                setup_record['arguments']['field'] != args.field):
            raise ValueError('Setup native/radial fields, geometry or cutoff differ from apply inputs')
        setup_binding = setup_record['output_bindings']['setup']
        require_identity(setup_binding['path'], setup_binding['sha256'])
        return HybridOperator(native, setup_binding['path'], radial[:, 0], radial[:, 1], end_radius), setup_binding

    def finish():
        verify_threads(runtime, 'at_completion')
        peak_memory(runtime)
        recheck_artifacts(result)
        save_record(args.outdir/'run.json', result)
        return 0 if result['all_passed'] else 1

    if args.stage == 'audit':
        from lead_setup import audit_setup
        audit_operator, setup = load_setup_operator(args.end_radius, args.K)
        width = 32*args.K*(args.K+1)
        frame_rows = [int(frame.shape[0]) for frame in audit_operator.E]
        # Both raw-A and stored-E residuals use complete streamed column
        # reductions, bound to the immutable frames/maps. Neither writes
        # rows-by-width COO chunks. Retain all dense polar/collar/port/lead
        # residual arrays and price the small reduction arrays explicitly.
        residual_maps = 9
        reduction_vectors = 5
        residual_storage = len(frame_rows)*residual_maps*2*width*reduction_vectors*8+64*1024**2
        matrix_storage = 120*width*width*16
        result['audit_storage_budget'] = {'matrix_artifact_bytes': matrix_storage,
            'frame_rows_including_spin': frame_rows, 'maps': residual_maps,
            'representations_per_frame_map': 2, 'complete_column_reduction_vectors': reduction_vectors,
            'reduction_and_metadata_storage_upper_bound': residual_storage,
            'total_required_new_bytes': matrix_storage+residual_storage,
            'scope': 'Raw A and actual E use complete streamed column reductions/locators and exact frame/map/source bindings; no residual COO chunk files. All dense polar/correction/port/lead residual arrays remain.'}
        storage(matrix_storage+residual_storage, 'whole_setup_symmetry_artifacts')
        # Audit uses saved actual factors and the small radial blocks. It
        # never calls the matrix-free generator or imports a Numba pool.
        audit = audit_setup(native, setup['path'], radial[:, 0], radial[:, 1],
                            args.end_radius, args.outdir, runtime)
        result['symmetry'] = audit
        result['output_bindings']['symmetry'] = identity(args.outdir/'symmetry.json')
        result['output_bindings']['symmetry_artifacts'] = audit['output_bindings']
        result['all_passed'] = bool(audit['audit_completed'] and audit['valley_covariance_pass'])
        result['gate_scope'] = 'Completed frame/valley audit; antiunitary may remain explicitly unresolved'
        return finish()

    if args.stage in ('hybrid', 'apply-benchmark'):
        if args.stage == 'hybrid':
            cutoff, end_radius = CASE_K_LENGTH[args.case]
            wanted_field = 'flat' if args.case.startswith('flat-') else 'F1'
            if args.K != cutoff or args.field != wanted_field or args.end_radius != end_radius:
                raise ValueError('Case/cutoff/field/length differs from the frozen roster')
        else:
            cutoff, end_radius = args.K, args.end_radius
        operator, setup = load_setup_operator(end_radius, cutoff)
        result['layout'] = operator.layout.metadata()
        result['operator_bound'] = operator.B
        result['operator_bound_record'] = operator.bound_record
        if args.audit is None or args.audit_sha256 is None:
            raise ValueError('Complete actual setup symmetry audit record required')
        audit_record = binding_record(args.audit, args.audit_sha256, bindings, 'audit')
        if (audit_record['field_association'] != field_association or
                audit_record['arguments']['K'] != cutoff or
                audit_record['arguments']['end_radius'] != end_radius):
            raise ValueError('Setup audit has a different physical operator/domain')
        result['exit_symmetry_status'] = audit_record['symmetry']['exit_symmetry_status']
        if args.stage == 'apply-benchmark':
            timings = []
            benchmark_projectors = hybrid_projectors(operator)
            storage((101*25*(operator.layout.total_dofs+native.layout.native_size)*16+99)//100+64*1024**2, 'apply_benchmark')
            for count in (1, 8, 16):
                initial = operator.layout.pack(pilot[:count].reshape(count, 4, 81, 81, 81))
                check_memory(runtime, 'apply_benchmark_'+str(count), initial.nbytes*8)
                warmup_start = time.perf_counter()
                operator.apply(initial)
                warmup_seconds = time.perf_counter()-warmup_start
                components = {}
                profiled = []
                for repeat in range(3):
                    times = {}
                    answer = operator.apply(initial, timings=times)
                    profiled.append(times)
                    del answer
                    verify_threads(runtime, 'profiled_apply_'+str(count)+'_'+str(repeat))
                def complete_checkpoint_diagnostics():
                    h = operator.h_apply(initial)
                    h2 = operator.h_apply(h)
                    diagnostic = operator.diagnostics(initial, hstates=h)
                    diagnostic['projector_fluxes'] = projector_fluxes(initial, h, h2, benchmark_projectors)
                    diagnostic['radial_bins'] = hybrid_inventory(operator, initial)
                    return diagnostic
                for label, function in (
                    ('native_generator', lambda: native.apply(initial[:, :native.layout.native_size])),
                    ('lead_generator', lambda: operator.lead.apply(operator.layout.lead_view(initial))),
                    ('full_generator', lambda: operator.apply(initial)),
                    ('complete_diagnostics', complete_checkpoint_diagnostics)):
                    values = []
                    for repeat in range(3):
                        start = time.perf_counter()
                        answer = function()
                        values.append(time.perf_counter()-start)
                        del answer
                        verify_threads(runtime, 'benchmark_'+str(count)+'_'+label+'_'+str(repeat))
                    components[label] = values
                plan = make_plan(operator.B, [T_FINAL])
                full_bound = truncation_bound(plan, initial)
                if not full_bound['bound_pass']:
                    raise ValueError('Benchmark full planned Taylor bound failed before action')
                first_delta = float(plan['intervals'][0])
                # Compile the shape-generic accumulator on tiny buffers and
                # the native contiguous-input signature before either step
                # timing; JIT overhead is recorded separately.
                from lead_numerics import _accumulator
                jit_started = time.perf_counter()
                _accumulator()(np.zeros((1, 1), dtype=np.complex128), np.zeros((1, 1), dtype=np.complex128))
                native.apply(np.ascontiguousarray(initial[:, :native.layout.native_size]))
                extra_warmup_seconds = time.perf_counter()-jit_started
                start = time.perf_counter()
                one_step = taylor_step(operator, initial, first_delta)
                step_seconds = time.perf_counter()-start
                native_initial = initial[:, :native.layout.native_size]
                native_plan = make_plan(native.B, [T_FINAL])
                native_bound = truncation_bound(native_plan, native_initial)
                if not native_bound['bound_pass']:
                    raise ValueError('Benchmark native Taylor bound failed before action')
                native_delta = float(native_plan['intervals'][0])
                start = time.perf_counter()
                native_step = taylor_step(native, native_initial, native_delta)
                native_step_seconds = time.perf_counter()-start
                # Time exactly the diagnostic path used after every real
                # macrostep. In particular, do not supply Hstates: H,
                # complete norm/energy Grams, norm reductions and the end
                # collar Gram must all be inside the measured helper call.
                from lead_numerics import _step_record
                start = time.perf_counter()
                diagnostic_Q0 = cross_gram(initial)
                diagnostic_energy0 = cross_gram(initial, operator.h_apply(initial))
                full_diagnostic_baseline_seconds = time.perf_counter()-start
                start = time.perf_counter()
                native_diagnostic_initial = np.ascontiguousarray(native_initial)
                native_diagnostic_Q0 = cross_gram(native_diagnostic_initial)
                native_diagnostic_energy0 = cross_gram(native_diagnostic_initial, native.h_apply(native_diagnostic_initial))
                native_diagnostic_baseline_seconds = time.perf_counter()-start
                diagnostic_samples = {}
                for label, target, block, Q0, energy0 in (
                    ('full_step_record', operator, initial, diagnostic_Q0, diagnostic_energy0),
                    ('native_step_record', native, native_diagnostic_initial, native_diagnostic_Q0, native_diagnostic_energy0)):
                    values = []
                    for repeat in range(3):
                        start = time.perf_counter()
                        answer = _step_record(target, block, Q0, energy0)
                        values.append(time.perf_counter()-start)
                        diagnostic_samples[label] = answer
                        verify_threads(runtime, 'benchmark_'+str(count)+'_'+label+'_'+str(repeat))
                    components[label] = values
                start = time.perf_counter()
                gram = cross_gram(initial)
                gram_seconds = time.perf_counter()-start
                sample = args.outdir/('sample_'+str(count)+'.npz')
                start = time.perf_counter()
                sample_binding = write_state(sample, one_step)
                io_seconds = time.perf_counter()-start
                result['output_bindings']['sample_'+str(count)] = sample_binding
                native_sample = args.outdir/('native_sample_'+str(count)+'.npz')
                start = time.perf_counter()
                native_sample_binding = write_state(native_sample, native_step)
                native_io_seconds = time.perf_counter()-start
                result['output_bindings']['native_sample_'+str(count)] = native_sample_binding
                timings.append({'columns': count, 'initial_bytes': initial.nbytes,
                                'warmup_seconds': warmup_seconds, 'component_seconds': components,
                                'Taylor_accumulator_and_native_contiguous_warmup_seconds': extra_warmup_seconds,
                                'profiled_full_apply_phases': profiled,
                                'full_planned_truncation_bound': full_bound,
                                'native_planned_truncation_bound': native_bound,
                                'Taylor32_one_step_seconds': step_seconds, 'step_delta_a': first_delta,
                                'native_Taylor32_one_step_seconds': native_step_seconds,
                                'native_step_delta_a': native_delta,
                                'full_step_record_seconds': components['full_step_record'],
                                'native_step_record_seconds': components['native_step_record'],
                                'full_step_record_observed': diagnostic_samples['full_step_record'],
                                'native_step_record_observed': diagnostic_samples['native_step_record'],
                                'step_record_clock_a': 0.,
                                'full_initial_Gram_energy_seconds': full_diagnostic_baseline_seconds,
                                'native_initial_Gram_energy_seconds': native_diagnostic_baseline_seconds,
                                'step_record_timing_scope': 'Actual lead_numerics._step_record on the same unchanged initial block; includes H, complete norm/energy Grams, reductions and end-collar Gram.',
                                'Gram_seconds': gram_seconds, 'write_seconds': io_seconds,
                                'native_write_seconds': native_io_seconds,
                                'process_peak_RSS_bytes': peak_memory(runtime),
                                'lossless_encoded_bytes': sample_binding['bytes'],
                                'native_lossless_encoded_bytes': native_sample_binding['bytes'],
                                'encoding': 'np.savez_compressed; full complex128 state unchanged',
                                'state_norm': opnorm(initial), 'step_state_norm': opnorm(one_step),
                                'Gram': gram})
                del initial, one_step, native_step, native_diagnostic_initial
            tree, arrays = flatten_arrays(timings)
            result['output_bindings']['benchmark_arrays'] = write_arrays(args.outdir/'benchmark_arrays.npz', arrays)
            result['timings'] = tree
            result['work_pricing'] = {'Taylor_H_calls_per_step': 32,
                'internal_energy_H_calls_per_step': 1, 'initial_energy_H_calls': 1,
                'checkpoint_hybrid_H_calls': 2, 'checkpoint_extra_native_H_calls': 1,
                'H100_native_only_seconds_per_column_per_a': .0648,
                'unknown_GPU_transfer_factors': ['dense shell projections/corrections', 'lead stencil', 'saved diagnostics', 'I/O']}
            result['all_passed'] = True
            result['gate_scope'] = 'Measured own CPU benchmark; subsequent scientific feasibility review still required'
            return finish()
        if args.benchmark is None or args.benchmark_sha256 is None:
            raise ValueError('Measured complete-operator benchmark and scientific cost review required')
        benchmark = binding_record(args.benchmark, args.benchmark_sha256, bindings, 'apply_benchmark')
        if (benchmark['stage'] != 'apply-benchmark' or benchmark['field_association'] != field_association or
                benchmark['arguments']['K'] != cutoff or benchmark['arguments']['end_radius'] != end_radius):
            raise ValueError('Wrong actual-operator benchmark')
        if args.case == 'flat-manufactured':
            native_initial, labels = manufactured_flat_inputs()
        else:
            native_initial, labels = (probes, np.array([0, 1, 4, 7])) if args.case.startswith('variable') else (pilot, np.arange(0, 128, 8))
        initial = operator.layout.pack(native_initial.reshape(len(native_initial), 4, 81, 81, 81))
        result['physical_operator_association'] = {
            'type': 'coupled_native_channel', 'field_association': field_association,
            'K': cutoff, 'end_radius': end_radius, 'setup_sha256': setup['sha256'],
            'canonical_case': args.case.removesuffix('-fine')}
        duration = 32. if args.case.startswith('variable') else T_FINAL
        clocks = np.r_[0., duration*np.arange(1, 5)/4]
        upper = fields_record['field_details']['P']['exterior799'] if args.field == 'F1' else {'N_upper': 1., 'v_abs_upper': 0.}
        certificate = endpoint_bounds(end_radius, duration, upper['N_upper'], upper['v_abs_upper'], opnorm(initial))
        result['finite_end_certificate'] = certificate
        if not certificate['passed']:
            raise ValueError('Finite lead end bound failed before propagation')
        fine = args.case.endswith('-fine')
        return _propagate_and_save(args, runtime, bindings, result, operator, initial, clocks,
                                   {'labels': labels}, fine, storage, hybrid=True)

    if args.stage == 'native-reference':
        if args.field != 'F1' or args.K != 10:
            raise ValueError('Only the frozen four-probe held-F1 native reference is admitted')
        from lead_native import NativeHamiltonian
        h = fields_record['field_details']['expanded_reference_halfside']
        if h is None:
            raise ValueError('No expanded reference size was admitted')
        side = 2*h+1
        check_memory(runtime, 'expanded_reference_buffers', 4*4*side**3*16*8+side**3*8*32)
        with load_array_record(fields_record) as archive:
            squares = archive['reference_radius_squares']
            radial_values = archive['F1_P_reference_Nv']
        coords = np.arange(-h, h+1, dtype=np.int64)
        x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
        radius2 = x*x+y*y+z*z
        outside = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]) > 40
        indices = np.searchsorted(squares, radius2[outside])
        if np.any(indices >= len(squares)) or not np.array_equal(squares[indices], radius2[outside]):
            raise ValueError('Expanded reference radial geometry association failed')
        expanded_n = np.empty((side,)*3)
        expanded_v = np.empty((3, side, side, side))
        values = radial_values[indices]
        expanded_n[outside] = values[:, 0]
        r = np.sqrt(radius2[outside])
        for axis, coordinate in enumerate((x, y, z)):
            expanded_v[axis, outside] = values[:, 1]*coordinate[outside]/r
        sl = slice(h-40, h+41)
        expanded_n[sl, sl, sl] = native.lapse
        expanded_v[:, sl, sl, sl] = native.flow
        # native passed here is the ball compression, but its original fields
        # retain every given cube site; reference has no ball mask or port.
        reference = NativeHamiltonian(expanded_n, expanded_v)
        initial_cube = np.zeros((4, 4, side, side, side), dtype=np.complex128)
        initial_cube[:, :, sl, sl, sl] = probes.reshape(4, 4, 81, 81, 81)
        initial = initial_cube.reshape(4, -1)
        candidates = fields_record['field_details']['expanded_reference_candidates']
        certificate = candidates[-1]
        if not certificate['passed'] or certificate['end_radius'] != h:
            raise ValueError('Wrong expanded-reference bound/size association')
        changing = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]) >= 39
        actual_n = float(np.max(expanded_n[changing]))
        actual_v = float(np.max(np.abs(expanded_v[:, changing])))
        if actual_n > certificate['N_upper'] or actual_v > certificate['v_abs_upper']:
            raise ValueError('Actual reference coefficients exceed the preselected end certificate')
        result.update(expanded_halfside=h, finite_end_certificate=certificate,
                      reference_coefficients={'N_max': actual_n, 'v_abs_max': actual_v},
                      reference_field_association={'N': array_identity(expanded_n), 'v': array_identity(expanded_v)})
        result['physical_operator_association'] = {
            'type': 'expanded_native_reference', 'halfside': h,
            'N': array_identity(expanded_n), 'v': array_identity(expanded_v),
            'fields_report_sha256': bindings['fields']['sha256'], 'mass': 0., 'spacing': 1.}
        return _propagate_and_save(args, runtime, bindings, result, reference, initial,
                                   np.array([0., 8., 16., 24., 32.]), {'DFT_rows': np.array([0, 1, 4, 7])},
                                   args.fine, storage, native_reference_halfside=h)

    if args.stage == 'b11':
        from lead_channels import RadialHamiltonian
        if args.b11_inputs is None or args.b11_inputs_sha256 is None:
            raise ValueError('Passed independent retained-level input gate required')
        retained_record = binding_record(args.b11_inputs, args.b11_inputs_sha256, bindings, 'b11_inputs')
        if retained_record['bindings']['fields']['sha256'] != bindings['fields']['sha256']:
            raise ValueError('B11 retained preparation has a different actual radial field gate')
        with load_array_record(fields_record) as archive:
            radial_values = archive[args.family+'_P_radial_Nv']
        stem = args.family+'_k'+str(args.kappa)+'_'
        with load_array_record(retained_record) as archive:
            retained = archive[stem+'initial']
            source_map = archive[stem+'source_column_map']
            individual = archive[stem+'individual_comparison_available']
        endpoint = {'cap2': 40, 'lead200': 3200, 'lead400': 6400, 'cap50': 808}[args.arm]
        radius = np.arange(1., endpoint+1)
        eta = np.zeros(endpoint)
        if args.arm.startswith('cap'):
            onset = 32. if args.arm == 'cap2' else 800.
            eta = np.sin(np.pi/2*np.clip((radius-onset)/8., 0., 1.))**2
        operator = RadialHamiltonian(args.kappa, radial_values[:endpoint, 0], radial_values[:endpoint, 1], eta)
        result['physical_operator_association'] = {
            'type': 'radial_B11', 'family': args.family, 'kappa': args.kappa,
            'arm': args.arm, 'end_radius': endpoint,
            'N': array_identity(radial_values[:endpoint, 0]),
            'v': array_identity(radial_values[:endpoint, 1]), 'eta': array_identity(eta),
            'retained_initial': array_identity(retained),
            'retained_input_report_sha256': bindings['b11_inputs']['sha256'],
            'fields_report_sha256': bindings['fields']['sha256'], 'mass': 0., 'spacing': 1.}
        initial = np.zeros((6, 2*endpoint), dtype=np.complex128)
        initial[:, :retained.shape[1]] = retained
        if args.fine and (args.family, args.kappa, args.arm) not in (('F1', 1, 'lead200'), ('F2', 3, 'cap50')):
            raise ValueError('Only two frozen B11 temporal-refinement cases are admitted')
        if args.arm.startswith('lead'):
            upper = (fields_record['field_details']['P']['exterior799'] if args.family == 'F1'
                     else fields_record['field_details']['F2_exterior799'])
            actual_n = float(np.max(radial_values[798:endpoint+2, 0]))
            actual_v = float(np.max(np.abs(radial_values[798:endpoint+2, 1])))
            if actual_n > upper['N_upper'] or actual_v > upper['v_abs_upper']:
                raise ValueError('B11 changing-weight coefficients exceed endpoint certificate maxima')
            certificate = endpoint_bounds(endpoint, 1600., upper['N_upper'], upper['v_abs_upper'], opnorm(initial))
            result['finite_end_certificate'] = certificate
            if not certificate['passed']:
                raise ValueError('B11 finite endpoint certificate failed before propagation')
        result['retained_dofs'] = retained.shape[1]
        return _propagate_and_save(args, runtime, bindings, result, operator, initial, 4*np.arange(401),
                                   {'source_column_map': source_map, 'individual_comparison_available': individual},
                                   args.fine, storage, b11_retained=retained.shape[1])
    raise ValueError('Unimplemented scientific stage')


def _propagate_and_save(args, runtime, bindings, result, operator, initial, clocks,
                        input_metadata, fine, storage, hybrid=False,
                        native_reference_halfside=None, b11_retained=None):
    import numpy as np
    from lead_numerics import propagate, make_plan, bisect_plan
    from lead_operator import cross_gram
    from lead_inputs import opnorm
    from lead_references import native_ball_observer
    state_size = initial.nbytes
    storage(state_size*len(clocks)+state_size*2, 'propagation_states_and_diagnostics')
    check_memory(runtime, 'propagation_buffers', state_size*10)
    original = initial.copy()
    initial_path = args.outdir/'initial.npz'
    result['output_bindings']['initial'] = write_state(initial_path, initial)
    result['input_metadata'] = clean(input_metadata)
    projector_masks = {}
    if hybrid:
        projector_masks = hybrid_projectors(operator)
    if b11_retained is not None:
        projector_masks['retained'] = np.arange(initial.shape[1]) < b11_retained
        projector_masks['far_r_gt_800'] = np.repeat(operator.radii > 800, 2)
    if projector_masks:
        result['output_bindings']['projector_masks'] = write_arrays(args.outdir/'projector_masks.npz', projector_masks)
    if fine:
        if args.coarse is None or args.coarse_sha256 is None:
            raise ValueError('Exact saved coarse interval plan is required for bisection')
        coarse_binding = require_identity(args.coarse, args.coarse_sha256)
        coarse = json.loads(args.coarse.read_text())
        if not coarse.get('run_completed') or coarse['bindings']['code'] != code_bindings():
            raise ValueError('Only completed exact-source coarse output can bind the fixed refinement')
        recheck_artifacts(coarse)
        bindings['coarse'], bindings['coarse_artifacts'] = coarse_binding, artifact_bindings(coarse)
        if coarse['field_association'] != result['field_association']:
            raise ValueError('Coarse/fine field association differs')
        if (coarse['stage'] != result['stage'] or
                coarse['physical_operator_association'] != result['physical_operator_association']):
            raise ValueError('Coarse/fine actual physical operator, geometry, stage or case differs')
        coarse_initial = load_state(coarse['output_bindings']['initial']['path'])
        if not np.array_equal(coarse_initial, initial):
            raise ValueError('Coarse/fine physical initial states differ')
        plan_binding = coarse['output_bindings']['plan']
        with np.load(plan_binding['path'], allow_pickle=False) as archive:
            plan = {key: archive[key].item() if archive[key].ndim == 0 else archive[key] for key in archive.files}
        plan = bisect_plan(plan)
    else:
        plan = make_plan(operator.B, clocks)
    result['output_bindings']['plan'] = write_arrays(args.outdir/'plan.npz', {key: np.asarray(value) for key, value in plan.items()})
    per_clock_aux = 0
    if hybrid:
        per_clock_aux = 2*len(initial)*sum(len(dofs) for dofs in operator.shell_dofs)*16
        per_clock_aux += 4*len(initial)*operator.E[0].shape[1]*16
    elif native_reference_halfside is not None:
        coords = np.arange(-40, 41)
        x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
        per_clock_aux = len(initial)*4*int(np.sum(x*x+y*y+z*z <= 1600))*16
    matrix_bytes = len(initial)**2*16
    diagnostic_bytes = len(plan['intervals'])*(10*matrix_bytes+4096)
    diagnostic_bytes += len(clocks)*(20*matrix_bytes+4096+per_clock_aux)
    storage(int(state_size*max(0, len(clocks)-1)*1.01)+diagnostic_bytes,
            'remaining_full_states_and_all_diagnostic_payloads')
    readouts, engagement = [], np.zeros(len(initial))
    callback_H_calls = callback_native_H_calls = 0
    def checkpoint(clock, states, diagnostics):
        nonlocal callback_H_calls, callback_native_H_calls
        index = len(readouts)
        verify_threads(runtime, 'before_checkpoint_'+str(index))
        storage(states.nbytes, 'checkpoint_'+str(index))
        state_path = args.outdir/('state_'+str(index)+'.npz')
        if clock == 0:
            if not np.array_equal(states, original):
                raise ValueError('Clock-zero state differs from the bound initial state')
            os.link(initial_path, state_path)
        else:
            write_state(state_path, states)
        state_binding = identity(state_path)
        result['output_bindings']['state_'+str(index)] = state_binding
        payload = {'clock': np.asarray(clock), 'numerics': diagnostics,
                   'return_overlap': original.conj()@states.T}
        hstates = None
        if projector_masks:
            hstates = operator.h_apply(states)
            h2states = operator.h_apply(hstates)
            callback_H_calls += 2
            payload['projector_fluxes'] = projector_fluxes(states, hstates, h2states, projector_masks)
            payload['flux_scope'] = 'Signed Hamiltonian currents; CAP dissipation rates are separate from these commutators'
        if hybrid:
            inventory = operator.diagnostics(states, hstates=hstates)
            callback_native_H_calls += 1
            payload.update(hybrid=inventory, bins=hybrid_inventory(operator, states))
            shell_lead = inventory['lead_Gram']+np.sum(inventory['shell_Grams'], axis=0)
            norm0 = np.sum(np.abs(original)**2, axis=1)
            ratio = np.real(np.diag(shell_lead))/norm0
            engagement[:] = np.maximum(engagement, ratio)
        if native_reference_halfside is not None:
            h = native_reference_halfside
            side, sl = 2*h+1, slice(h-40, h+41)
            coords = np.arange(-40, 41)
            x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
            mask = x*x+y*y+z*z <= 1600
            cube = states.reshape(len(states), 4, side, side, side)
            payload['native_ball'] = cube[:, :, sl, sl, sl][:, :, mask].reshape(len(states), -1)
        if b11_retained is not None:
            retained = states[:, :b11_retained]
            payload['retained_Gram'] = cross_gram(retained)
            if operator.is_unitary:
                far = operator.view(states)[:, operator.radii > 800, :]
                payload['output_Gram'] = cross_gram(far.reshape(len(states), -1))
            else:
                payload['output_Gram'] = cross_gram(original)-cross_gram(states)
        tree, arrays = flatten_arrays(payload)
        array_binding = write_arrays(args.outdir/('readout_'+str(index)+'.npz'), arrays)
        result['output_bindings']['readout_'+str(index)] = array_binding
        record = {'clock': float(clock), 'state_binding': state_binding,
                  'array_binding': array_binding, 'array_tree': tree}
        readouts.append(record)
        # Completed checkpoint inventory survives an interruption. Every write
        # is a fresh scientific record, never a coordination monitor.
        save_record(args.outdir/('checkpoint_'+str(index)+'.json'), record)
        return {'state': state_binding, 'readout': array_binding}
    started = time.perf_counter()
    evolution = propagate(operator, initial, clocks, interval_plan=plan, checkpoint=checkpoint,
                          progress=lambda step, total, clock, elapsed: print(
                              json.dumps({'step': step, 'steps': total, 'clock_a': clock, 'seconds': elapsed}), flush=True))
    result['elapsed_seconds'] = time.perf_counter()-started
    numerical = {key: value for key, value in evolution.items() if key not in ('final_columns', 'readout_states')}
    tree, arrays = flatten_arrays(numerical)
    result['output_bindings']['numerics'] = write_arrays(args.outdir/'numerics.npz', arrays)
    result['numerics'] = tree
    result['readouts'] = readouts
    result['callback_operator_work'] = {'extra_full_H_applications': callback_H_calls,
                                       'extra_native_H_applications': callback_native_H_calls}
    result['run_completed'] = True
    result['gate_scope'] = 'Producer numerical/end-bound checks only; response/convergence require the saved-only reader and scientific reviews'
    if getattr(operator, 'is_unitary', True):
        result['all_passed'] = evolution['unitary_numerical_gates_pass']
    else:
        previous = evolution['Q0']
        maximum = 0.
        for step in evolution['step_diagnostics']:
            maximum = max(maximum, float(np.linalg.eigvalsh(step['Gram']-previous)[-1]))
            previous = step['Gram']
        result['maximum_CAP_norm_increase'] = maximum
        result['all_passed'] = bool(maximum <= 1e-8)
    if hybrid:
        result['engagement_inventory_ratios'] = engagement.tolist()
        result['per_column_transport_informative'] = (engagement > 1e-8).tolist()
        result['transport_gate_informative'] = bool(np.all(engagement > 1e-8))
        result['uninformative_meaning'] = 'Neither a matching failure nor a transport pass; no new direction or duration authorized'
    verify_threads(runtime, 'after_propagation')
    peak_memory(runtime)
    recheck_artifacts(result)
    save_record(args.outdir/'run.json', result)
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
