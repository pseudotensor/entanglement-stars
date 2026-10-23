"""Standalone dense angular-frame setup and its bounded preconstruction timing.

Call only after lead_runtime.admit(mode='dense').  Native restricted bonds use
SciPy sparse multiplication; no Numba module/pool is imported by this stage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from lead_channels import frame_labels, sampled_frame, folded_blocks, ideal_cell_block
from lead_native import ball_mask, coordinates, canonical_identity
from lead_runtime import check_memory, core_bindings, identity, verify_threads


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def shell_indices(H=40, core_radius=32, cells=4, cell_width=2):
    points = coordinates(H)
    squared = np.sum(points*points, axis=1)
    sites = len(points)
    result = []
    for cell in range(cells):
        lower, upper = core_radius+cell_width*cell, core_radius+cell_width*(cell+1)
        selected = np.flatnonzero((squared > lower*lower) & (squared <= upper*upper))
        dofs = (np.arange(4)[:, None]*sites+selected[None]).reshape(-1)
        result.append({'sites': selected, 'dofs': dofs, 'points': points[selected],
                       'lower_radius': lower, 'upper_radius': upper})
    return result


def blocked_gram(matrix, row_tile=4096):
    result = np.zeros((matrix.shape[1],)*2, dtype=np.complex128)
    for start in range(0, matrix.shape[0], row_tile):
        part = np.asarray(matrix[start:start+row_tile])
        result += part.conj().T@part
    return result


def hermitian_realization(raw, scale):
    """Reject a large raw defect before realizing an exact Hermitian block."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.dtype.hasobject or not np.all(np.isfinite(raw)) or not np.isfinite(scale) or scale < 0:
        raise ValueError('Hermitian realization needs a finite square matrix and scale')
    defect = raw-raw.conj().T
    upper = float(np.linalg.norm(defect))
    threshold = 5e-12*max(1., float(scale))
    passed = bool(upper <= threshold)
    value = (raw+raw.conj().T)/2 if passed else None
    return {'passed': passed, 'raw_defect_Frobenius_upper_bound': upper,
            'raw_defect_maximum_entry': float(np.max(np.abs(defect))),
            'norm_convention': 'Frobenius upper bound on induced defect; failure withholds realization',
            'scale': float(scale), 'threshold': threshold,
            'value': value, 'arithmetic_difference': value-raw if passed else None}


def _write_angular(path, points, K, point_tile=512):
    count, width = len(points), len(frame_labels(K))
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.complex128, shape=(4*count, width))
    for start in range(0, count, point_tile):
        stop = min(start+point_tile, count)
        part = sampled_frame(points[start:stop], K).reshape(4, stop-start, width)
        for spin in range(4):
            output[spin*count+start:spin*count+stop] = part[spin]
    output.flush()
    return output


def project_native_block(native, target_dofs, source_dofs, left, right, column_tile=128):
    bonds = native.sparse_block(target_dofs, source_dofs)
    result = np.empty((left.shape[1], right.shape[1]), dtype=np.complex128)
    for start in range(0, right.shape[1], column_tile):
        stop = min(start+column_tile, right.shape[1])
        applied = bonds@np.asarray(right[:, start:stop])
        result[:, start:stop] = (left.T@applied.conj()).conj()
    return result


def benchmark_setup(native, K, outdir, runtime):
    """Fixed first1024 points/all prescribed columns in each physical shell."""
    if K not in (10, 12):
        raise ValueError('Only the prescribed K10/K12 setup timings are admitted')
    outdir = Path(outdir)
    if any(outdir.iterdir()):
        raise RuntimeError('Setup benchmark requires a fresh empty directory')
    width = len(frame_labels(K))
    check_memory(runtime, 'before_setup_benchmark', workspace_bytes=8*16*(4096*width+width*width))
    reports, bindings = [], {}
    for cell, shell in enumerate(shell_indices()):
        points = shell['points'][:1024]
        indices = shell['sites'][:len(points)]
        dofs = (np.arange(4)[:, None]*native.lapse.size+indices[None]).reshape(-1)
        started = time.perf_counter()
        angular = sampled_frame(points, K)
        assembly = time.perf_counter()-started
        started = time.perf_counter()
        gram = angular.conj().T@angular
        gram_seconds = time.perf_counter()-started
        started = time.perf_counter()
        projected = project_native_block(native, dofs, dofs, angular, angular)
        projection_seconds = time.perf_counter()-started
        raw = outdir/('setup_tile_c'+str(cell)+'.npz')
        np.savez(raw, points=points, global_dofs=dofs, A=angular, G=gram, projected_native_tile=projected)
        bindings[raw.name] = identity(raw)
        ratio = len(shell['points'])/len(points)
        reports.append({'cell': cell, 'sample_points': len(points), 'full_points': len(shell['points']),
                        'row_count': len(angular), 'column_count': width, 'projection_column_tile': 128,
                        'angular_assembly_seconds': assembly, 'Gram_seconds': gram_seconds,
                        'native_projection_seconds': projection_seconds,
                        'row_linear_Gram_extrapolation_seconds': ratio*gram_seconds,
                        'row_linear_projection_extrapolation_seconds': ratio*projection_seconds,
                        'full_A_or_E_bytes': 4*len(shell['points'])*width*16,
                        'dense_Gram_bytes': width*width*16,
                        'qualification': 'Fixed unwhitened representative tile; no frame or physical accuracy claim. Eigensolve/whitening and full sparse row adjacency are not priced by this extrapolation.'})
        verify_threads(runtime, 'after_benchmark_cell_'+str(cell))
        check_memory(runtime, 'after_benchmark_cell_'+str(cell))
    output = {'mode': 'bounded_setup_benchmark', 'K': K, 'runtime': runtime,
              'code_bindings': core_bindings(), 'tiles': reports, 'output_bindings': bindings,
              'performed_propagation': False, 'constructed_complete_frame': False}
    _save_json(outdir/'setup_benchmark.json', output)
    return output


def build_setup(native, K, radial_N, radial_v, outdir, runtime, frame_source=None):
    """Construct exactly the physical four-shell frame, or withhold on any gate."""
    if K not in (10, 12) or native.shape != (81,)*3 or not np.array_equal(native.mask, ball_mask(40, 40)):
        raise ValueError('Full setup requires the fixed K10/K12 and native ball geometry')
    outdir = Path(outdir)
    if any(outdir.iterdir()):
        raise RuntimeError('Full setup requires a fresh empty directory')
    labels, shells, frames, reports = frame_labels(K), shell_indices(), [], []
    bindings = {}
    reused = None
    reuse_binding = None
    if frame_source is not None:
        frame_source = Path(frame_source)
        reused = json.loads(frame_source.read_text())
        if reused.get('all_passed') is not True or reused['K'] != K or reused['code_bindings'] != core_bindings():
            raise ValueError('Frame reuse requires accepted identical-K/current-source setup')
        from lead_runtime import require_identity
        for bound in reused['output_bindings'].values():
            require_identity(bound['path'], bound['sha256'])
        for cell, shell in enumerate(shells):
            record = reused['frames'][cell]
            if not (record['conditioning_pass'] and record['orthogonality_pass'] and record['Gram_raw_Hermiticity_pass'] and
                    record['lambda_minimum'] >= .25 and record['lambda_maximum'] <= 4 and record['orthogonality_maximum_entry'] <= 5e-12):
                raise ValueError('A reused frame lacks the exact frozen admission gates')
            with np.load(frame_source.parent/('frame_c'+str(cell)+'.npz'), allow_pickle=False) as archive:
                if not (np.array_equal(archive['labels'], labels) and np.array_equal(archive['shell_sites'], shell['sites']) and np.array_equal(archive['shell_dofs'], shell['dofs'])):
                    raise ValueError('Field-independent frame geometry/labels differ')
        reuse_binding = identity(frame_source)
    check_memory(runtime, 'before_full_setup', workspace_bytes=sum(8*len(shell['points'])*len(labels)*16 for shell in shells)+8*len(labels)**2*16)
    for cell, shell in enumerate(shells):
        if reused is not None:
            for name in ('A_c'+str(cell)+'.npy', 'E_c'+str(cell)+'.npy', 'frame_c'+str(cell)+'.npz'):
                source, destination = frame_source.parent/name, outdir/name
                os.link(source, destination)  # Same bulk filesystem; never silently duplicate huge files.
                bindings[name] = identity(destination)
            frames.append(np.load(outdir/('E_c'+str(cell)+'.npy'), mmap_mode='r', allow_pickle=False))
            reports.append({**reused['frames'][cell], 'reused_existing_frame': True,
                            'frame_source': str(frame_source), 'original_frame_timings_are_reused_provenance': True})
            continue
        started = time.perf_counter()
        path_A = outdir/('A_c'+str(cell)+'.npy')
        A = _write_angular(path_A, shell['points'], K)
        assembled = time.perf_counter()-started
        started = time.perf_counter()
        G = blocked_gram(A)
        gram_time = time.perf_counter()-started
        anti = float(np.max(np.abs(G-G.conj().T)))
        gram_scale = float(np.sqrt(np.linalg.norm(G, 1)*np.linalg.norm(G, np.inf)))
        gram_realization = hermitian_realization(G, gram_scale)
        if not gram_realization['passed']:
            diagnostic_path = outdir/('frame_c'+str(cell)+'.npz')
            np.savez(diagnostic_path, G=G, shell_sites=shell['sites'], shell_dofs=shell['dofs'], labels=labels)
            bindings[path_A.name], bindings[diagnostic_path.name] = identity(path_A), identity(diagnostic_path)
            reports.append({'cell': cell, 'Gram_defect_Frobenius_upper_bound': gram_realization['raw_defect_Frobenius_upper_bound'],
                            'Gram_defect_threshold': gram_realization['threshold'], 'Gram_raw_Hermiticity_pass': False})
            output = {'all_passed': False, 'withheld': 'raw angular Gram Hermiticity failed before eigensolve',
                      'K': K, 'runtime': runtime, 'frames': reports, 'output_bindings': bindings}
            _save_json(outdir/'setup.json', output)
            return output
        started = time.perf_counter()
        values, vectors = np.linalg.eigh(gram_realization['value'])
        eigen_time = time.perf_counter()-started
        conditioning = bool(np.all(np.isfinite(values)) and values[0] >= .25 and values[-1] <= 4.)
        record = {'cell': cell, 'A_assembly_seconds': assembled, 'Gram_seconds': gram_time,
                  'eigensolve_seconds': eigen_time, 'Gram_anti_Hermitian_maximum_entry': anti,
                  'Gram_defect_Frobenius_upper_bound': gram_realization['raw_defect_Frobenius_upper_bound'],
                  'Gram_defect_threshold': gram_realization['threshold'], 'Gram_raw_Hermiticity_pass': True,
                  'lambda_minimum': float(values[0]), 'lambda_maximum': float(values[-1]),
                  'conditioning_pass': conditioning}
        diagnostic = {'G': G, 'eigenvalues': values, 'eigenvectors': vectors,
                      'G_Hermitian_part': gram_realization['value'], 'G_arithmetic_symmetrization': gram_realization['arithmetic_difference'],
                      'shell_sites': shell['sites'], 'shell_dofs': shell['dofs'], 'labels': labels}
        if not conditioning:
            diagnostic_path = outdir/('frame_c'+str(cell)+'.npz')
            np.savez(diagnostic_path, **diagnostic)
            bindings[path_A.name], bindings[diagnostic_path.name] = identity(path_A), identity(diagnostic_path)
            reports.append(record)
            output = {'all_passed': False, 'withheld': 'prescribed frame conditioning failed; no rank or cutoff change',
                      'K': K, 'runtime': runtime, 'frames': reports, 'output_bindings': bindings}
            _save_json(outdir/'setup.json', output)
            return output
        inverse_half = (vectors/np.sqrt(values)[None])@vectors.conj().T
        path_E = outdir/('E_c'+str(cell)+'.npy')
        E = np.lib.format.open_memmap(path_E, mode='w+', dtype=np.complex128, shape=A.shape)
        started = time.perf_counter()
        for start in range(0, len(A), 4096):
            E[start:start+4096] = A[start:start+4096]@inverse_half
        E.flush()
        whitening_time = time.perf_counter()-started
        orthogonal = blocked_gram(E)
        orthogonal_error = float(np.max(np.abs(orthogonal-np.eye(len(labels)))))
        record.update(whitening_seconds=whitening_time,
                      orthogonality_maximum_entry=orthogonal_error,
                      orthogonality_pass=bool(np.isfinite(orthogonal_error) and orthogonal_error <= 5e-12),
                      E_norm_bound=float(np.sqrt(1+len(labels)*orthogonal_error)))
        diagnostic.update(inverse_half=inverse_half, E_Gram=orthogonal)
        diagnostic_path = outdir/('frame_c'+str(cell)+'.npz')
        np.savez(diagnostic_path, **diagnostic)
        for path in (path_A, path_E, diagnostic_path):
            bindings[path.name] = identity(path)
        frames.append(E); reports.append(record)
        verify_threads(runtime, 'after_frame_'+str(cell))
        check_memory(runtime, 'after_frame_'+str(cell))
        if not record['orthogonality_pass']:
            output = {'all_passed': False, 'withheld': 'prescribed frame orthogonality failed',
                      'K': K, 'runtime': runtime, 'frames': reports, 'output_bindings': bindings}
            _save_json(outdir/'setup.json', output)
            return output
    paper = folded_blocks(K, radial_N, radial_v, first_odd=33, end=42)
    s = (0., .25, .75, 1.)
    Dg, corrections, block_records = {}, {}, {}
    for c in range(4):
        for d in range(c, min(4, c+2)):
            started = time.perf_counter()
            value = project_native_block(native, shells[c]['dofs'], shells[d]['dofs'], frames[c], frames[d])
            raw = value.copy()
            realization = hermitian_realization(raw, native.B*reports[c]['E_norm_bound']**2) if c == d else None
            raw_anti = realization['raw_defect_maximum_entry'] if realization is not None else None
            if realization is not None and not realization['passed']:
                path = outdir/('blocks_'+str(c)+'_'+str(d)+'.npz')
                np.savez(path, Dg_raw=raw)
                bindings[path.name] = identity(path)
                output = {'all_passed': False, 'withheld': 'raw diagonal native projection Hermiticity failed; not averaged away',
                          'failed_block': [c, d], 'raw_defect_Frobenius_upper_bound': realization['raw_defect_Frobenius_upper_bound'],
                          'threshold': realization['threshold'], 'K': K, 'runtime': runtime,
                          'frames': reports, 'output_bindings': bindings}
                _save_json(outdir/'setup.json', output)
                return output
            if realization is not None:
                value = realization['value']
            ideal = ideal_cell_block(K, paper, c, d)
            correction = ((s[c]+s[d])/2)*(ideal-value)
            path = outdir/('blocks_'+str(c)+'_'+str(d)+'.npz')
            saved = {'Dg_raw': raw, 'Dg': value, 'Dp': ideal, 'correction': correction}
            if realization is not None:
                saved['Dg_arithmetic_symmetrization'] = realization['arithmetic_difference']
            np.savez(path, **saved)
            bindings[path.name] = identity(path)
            Dg[c, d], corrections[c, d] = value, correction
            block_records[str(c)+'_'+str(d)] = {'seconds': time.perf_counter()-started,
                                               'Dg_diagonal_anti_Hermitian_maximum_entry': raw_anti,
                                               'raw_defect_Frobenius_upper_bound': realization['raw_defect_Frobenius_upper_bound'] if realization is not None else None,
                                               'raw_defect_threshold': realization['threshold'] if realization is not None else None,
                                               'diagonal_realization_pass': realization['passed'] if realization is not None else None}
            verify_threads(runtime, 'after_projection_'+str(c)+'_'+str(d))
    port = ideal_cell_block(K, paper, 3, 4)
    port_path = outdir/'port.npz'
    np.savez(port_path, Dp34=port, chain_labels=labels[::4, :5])
    bindings[port_path.name] = identity(port_path)
    output = {'all_passed': True, 'K': K, 'runtime': runtime, 'frames': reports,
              'projected_blocks': block_records, 'code_bindings': core_bindings(),
              'operator_input_fingerprints': {'native_N': canonical_identity(native.lapse),
                  'native_v': canonical_identity(native.flow), 'native_mask': canonical_identity(native.mask),
                  'radial_collar_N': canonical_identity(np.asarray(radial_N)[:42]),
                  'radial_collar_v': canonical_identity(np.asarray(radial_v)[:42]),
                  'mass': native.mass, 'spacing': native.spacing},
              'field_independent_frame_source': reuse_binding,
              'frame_reuse_mode': 'hardlinks to accepted immutable frame artifacts; all field-dependent blocks recomputed' if reused is not None else None,
              'output_bindings': bindings, 'shell_geometry': [{'lower': x['lower_radius'], 'upper': x['upper_radius'], 'points': len(x['points'])} for x in shells]}
    _save_json(outdir/'setup.json', output)
    return output


def load_setup(path):
    """Load own reviewed setup arrays, preserving file identities and read-only E."""
    from lead_runtime import require_identity
    path = Path(path)
    report = json.loads(path.read_text())
    if report.get('all_passed') is not True:
        raise ValueError('Passing complete setup is required')
    if report['code_bindings'] != core_bindings():
        raise ValueError('Setup was constructed by a different numerical source version')
    for item in report['output_bindings'].values():
        require_identity(item['path'], item['sha256'])
    root = path.parent
    frames, dofs, norms = [], [], []
    for cell in range(4):
        frames.append(np.load(root/('E_c'+str(cell)+'.npy'), mmap_mode='r', allow_pickle=False))
        with np.load(root/('frame_c'+str(cell)+'.npz'), allow_pickle=False) as archive:
            dofs.append(archive['shell_dofs'])
        norms.append(report['frames'][cell]['E_norm_bound'])
    dg, corrections = {}, {}
    for c in range(4):
        for d in range(c, min(4, c+2)):
            with np.load(root/('blocks_'+str(c)+'_'+str(d)+'.npz'), allow_pickle=False) as archive:
                dg[c, d], corrections[c, d] = archive['Dg'], archive['correction']
    with np.load(root/'port.npz', allow_pickle=False) as archive:
        port = archive['Dp34']
    return {'K': report['K'], 'E': frames, 'shell_dofs': dofs, 'E_norm_bounds': norms,
            'Dg': dg, 'corrections': corrections, 'port': port, 'report': report}


def audit_setup(native, setup_json, radial_N, radial_v, end_radius, outdir, runtime):
    from lead_symmetry import audit_setup as audit
    return audit(native, setup_json, radial_N, radial_v, end_radius, outdir, runtime)
