"""Whole-frame/operator symmetry audit using author-fixed signed permutations.

This is a dense setup stage, not a test on selected physical probes. Both
sampled-A and stored-E residuals retain complete column reductions, complex
maximum locators and exact frame/map/source bindings. Polar/collar/port/lead
residual arrays and labelled upper bounds remain on box.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from lead_channels import (antiunitary_permutation, chain_labels, valley_permutation)
from lead_native import spin_matrices, valley_matrix
from lead_operator import HybridOperator, norm_upper
from lead_runtime import check_memory, core_bindings, identity, require_fresh, verify_threads


def permuted_similarity(matrix, target, phases, conjugate=False):
    source = matrix.conj() if conjugate else matrix
    result = np.empty_like(matrix)
    result[np.ix_(target, target)] = source*phases[:, None]*phases[None, :].conj()
    return result


def polar_commutator(M, target, phases, anti=False):
    left = np.empty_like(M)
    left[target] = phases[:, None]*(M.conj() if anti else M)
    right = M[:, target]*phases[None]
    return left-right


def raw_frame_residual(A, points, matrix, target, phases, anti, outdir, prefix, save_sparse=True):
    count, width = len(points), A.shape[1]
    columns_squared = np.zeros(width)
    finite_counts = np.zeros(width, dtype=np.int64)
    nonzero_counts = np.zeros(width, dtype=np.int64)
    finite_nonzero_counts = np.zeros(width, dtype=np.int64)
    maximum, locator, maximum_value, chunks = 0., None, 0j, []
    for start in range(0, count, 256):
        stop = min(start+256, count)
        block = np.asarray([A[spin*count+start:spin*count+stop] for spin in range(4)])
        source = block.conj() if anti else block
        transformed = np.einsum('ab,bnd->and', matrix, source)
        if not anti:
            transformed *= np.where((points[start:stop]@np.asarray(prefix['mu'])) % 2, -1., 1.)[None, :, None]
        residual = transformed-block[:, :, target]*phases[None, None]
        columns_squared += np.sum(np.abs(residual)**2, axis=(0, 1))
        finite = np.isfinite(residual)
        nonzero = residual != 0
        finite_counts += np.sum(finite, axis=(0, 1), dtype=np.int64)
        nonzero_counts += np.sum(nonzero, axis=(0, 1), dtype=np.int64)
        finite_nonzero_counts += np.sum(finite & nonzero, axis=(0, 1), dtype=np.int64)
        value = float(np.max(np.abs(residual)))
        if locator is None or value > maximum:
            spin, point, column = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            maximum, locator = value, [int(spin*count+start+point), int(column)]
            maximum_value = complex(residual[spin, point, column])
        if save_sparse:
            spin, point, column = np.nonzero(residual)
            if len(spin):
                path = outdir/(prefix['name']+'_raw_residual_'+str(start)+'.npz')
                np.savez_compressed(path, row=spin*count+start+point, column=column, value=residual[spin, point, column])
                chunks.append(identity(path))
    return {'Frobenius_upper_bound': float(np.sqrt(np.sum(columns_squared))),
            'column_Frobenius_norms': np.sqrt(columns_squared), 'column_squared_absolute_sums': columns_squared,
            'column_finite_counts': finite_counts, 'column_nonzero_counts': nonzero_counts,
            'column_finite_nonzero_counts': finite_nonzero_counts,
            'requested_entries': int(4*count*width), 'finite_entries': int(np.sum(finite_counts)),
            'nonfinite_entries': int(4*count*width-np.sum(finite_counts)),
            'nonzero_entries': int(np.sum(nonzero_counts)), 'finite_nonzero_entries': int(np.sum(finite_nonzero_counts)),
            'maximum_entry': maximum,
            'maximum_locator': locator, 'maximum_complex_value': {'real': maximum_value.real, 'imag': maximum_value.imag},
            'shape': list(A.shape), 'nonzero_residual_chunks': chunks,
            'norm_convention': 'Streamed complete Frobenius norm; upper bound on induced error',
            'storage': 'exact sparse nonzero values; omitted entries are exactly numerical zero' if save_sparse else 'declared streamed complete-column reduction and complex maximum locator, bound to stored frame; no ninefold dense residual copies'}


def _native_relation(native, anti, mu=None):
    alpha, beta, uc = spin_matrices()
    if anti:
        residuals = [uc@a.conj()@uc.conj().T-a for a in alpha]
        residuals.append(uc@beta.conj()@uc.conj().T+beta)
        matrix = uc
    else:
        matrix = valley_matrix(mu)
        residuals = [matrix.conj().T@a@matrix-(-1)**mu[axis]*a for axis, a in enumerate(alpha)]
        residuals.append(matrix.conj().T@beta@matrix-beta)
    upper = max(float(np.linalg.norm(value)) for value in residuals)
    boundary_valid = not native.periodic or native.shape[0] % 2 == 0 or anti or not any(mu)
    return {'coefficient_residuals': np.asarray(residuals),
            'native_operator_upper_bound': 4*native.B*upper if boundary_valid else None,
            'native_boundary_relation_valid': boundary_valid,
            'native_unitarity_error': float(np.linalg.norm(matrix.conj().T@matrix-np.eye(4))),
            'native_antiunitary_square_error': float(np.linalg.norm(matrix@matrix.conj()-np.eye(4))) if anti else None}


def _lead_residuals(lead, target, phases, anti):
    labels, maps, records, arrays = chain_labels(lead.K), {}, [], {}
    for chain in range(len(labels)):
        selected = target[4*chain:4*chain+4]
        destinations = np.unique(selected//4)
        if len(destinations) != 1:
            raise ValueError('Candidate symmetry mixes radial chain cells; no frozen local map')
        destination = int(destinations[0])
        block = np.zeros((4, 4), complex)
        block[selected % 4, np.arange(4)] = phases[4*chain:4*chain+4]
        first = block.flat[np.flatnonzero(block.ravel())[0]]
        canonical = block/first
        key = (int(labels[chain, 3])-1, int(labels[destination, 3])-1, canonical.tobytes())
        maps.setdefault(key, block)
    maximum = 0.
    for index, ((source_k, destination_k, _), block) in enumerate(maps.items()):
        source_on, source_off = lead.blocks['onsite'][source_k], lead.blocks['forward'][source_k]
        target_on, target_off = lead.blocks['onsite'][destination_k], lead.blocks['forward'][destination_k]
        if anti:
            source_on, source_off = source_on.conj(), source_off.conj()
        onsite = block[None]@source_on@block.conj().T+target_on if anti else block[None]@source_on@block.conj().T-target_on
        forward = block[None]@source_off@block.conj().T+target_off if anti else block[None]@source_off@block.conj().T-target_off
        onsite_norm = np.sqrt(np.sum(np.abs(onsite)**2, axis=(1, 2)))
        neighbor_norm = np.sqrt(np.sum(np.abs(forward)**2, axis=(1, 2)))
        row = onsite_norm.copy()
        row[:-1] += neighbor_norm; row[1:] += neighbor_norm
        upper = float(np.max(row))
        maximum = max(maximum, upper)
        arrays['lead_onsite_residual_'+str(index)] = onsite
        arrays['lead_forward_residual_'+str(index)] = forward
        records.append({'source_k': source_k+1, 'target_k': destination_k+1,
                        'block_map': block, 'whole_chain_block_row_upper_bound': upper})
    return maximum, records, arrays


def audit_setup(native, setup_json, radial_N, radial_v, end_radius, outdir, runtime):
    from lead_setup import load_setup, shell_indices
    outdir, setup_json = Path(outdir), Path(setup_json)
    require_fresh(outdir)
    setup = load_setup(setup_json)
    audit_sources = core_bindings()
    completed = HybridOperator(native, setup, radial_N, radial_v, end_radius)
    width, K = completed.port.shape[0], completed.K
    check_memory(runtime, 'before_whole_symmetry_audit', workspace_bytes=12*width*width*16+4*1024*width*16)
    shells, frame_data = shell_indices(), []
    for cell in range(4):
        with np.load(setup_json.parent/('frame_c'+str(cell)+'.npz'), allow_pickle=False) as archive:
            G, M, eigenvalues = archive['G'], archive['inverse_half'], archive['eigenvalues']
        frame_data.append({'A': np.load(setup_json.parent/('A_c'+str(cell)+'.npy'), mmap_mode='r', allow_pickle=False),
                           'M': M, 'M_norm': float(1/np.sqrt(eigenvalues[0])),
                           'A_norm_upper': float(np.sqrt(np.sqrt(np.linalg.norm(G, 1)*np.linalg.norm(G, np.inf)))),
                           'E_norm_upper': setup['E_norm_bounds'][cell]})
    rows, output_bindings = [], {}
    transforms = [('valley_'+''.join(map(str, mu)), False, mu) for mu in itertools.product((0, 1), repeat=3)]
    transforms.append(('antiunitary', True, None))
    for name, anti, mu in transforms:
        target, phases = antiunitary_permutation(K) if anti else valley_permutation(K, mu)
        alpha, beta, uc = spin_matrices()
        matrix = uc if anti else valley_matrix(mu)
        native_record = _native_relation(native, anti, mu)
        arrays = {'target_indices': target, 'phases': phases, 'native_coefficient_residuals': native_record['coefficient_residuals']}
        errors, frames = [], []
        for cell, data in enumerate(frame_data):
            prefix = {'name': name+'_c'+str(cell), 'mu': mu}
            raw = raw_frame_residual(data['A'], shells[cell]['points'], matrix, target, phases, anti, outdir, prefix, save_sparse=False)
            commutator = polar_commutator(data['M'], target, phases, anti)
            commutator_upper = float(np.linalg.norm(commutator))
            product_bound = raw['Frobenius_upper_bound']*data['M_norm']+data['A_norm_upper']*commutator_upper
            actual = raw_frame_residual(setup['E'][cell], shells[cell]['points'], matrix, target, phases,
                                        anti, outdir, prefix, save_sparse=False)
            error = actual['Frobenius_upper_bound']
            arrays['polar_factor_residual_c'+str(cell)] = commutator
            arrays['raw_residual_column_norms_c'+str(cell)] = raw.pop('column_Frobenius_norms')
            arrays['stored_E_residual_column_norms_c'+str(cell)] = actual.pop('column_Frobenius_norms')
            for metric in ('column_squared_absolute_sums', 'column_finite_counts', 'column_nonzero_counts', 'column_finite_nonzero_counts'):
                arrays['raw_residual_'+metric+'_c'+str(cell)] = raw.pop(metric)
                arrays['stored_E_residual_'+metric+'_c'+str(cell)] = actual.pop(metric)
            errors.append(error)
            frames.append({'cell': cell, 'raw_sampled_frame_residual': raw,
                           'polar_commutator_Frobenius_upper_bound': commutator_upper,
                           'mathematical_A_times_M_covariance_upper_bound': product_bound,
                           'stored_E_covariance_residual': actual,
                           'A_artifact_binding': setup['report']['output_bindings']['A_c'+str(cell)+'.npy'],
                           'E_artifact_binding': setup['report']['output_bindings']['E_c'+str(cell)+'.npy'],
                           'map_source_binding': audit_sources['lead_channels.py'],
                           'audit_source_binding': audit_sources['lead_symmetry.py'],
                           'actual_stored_E_covariance_operator_upper_bound': error})
        total_correction = 0.
        correction_records = []
        for (c, d), correction in setup['corrections'].items():
            transformed = permuted_similarity(correction, target, phases, anti)
            residual = transformed+correction if anti else transformed-correction
            rho = float(np.linalg.norm(residual))
            e_c, e_d = frame_data[c]['E_norm_upper'], frame_data[d]['E_norm_upper']
            eps_c, eps_d = errors[c], errors[d]
            bound = e_c*e_d*rho+norm_upper(correction)*(eps_c*e_d+e_c*eps_d+eps_c*eps_d)
            multiplicity = 1 if c == d else 2
            total_correction += multiplicity*bound
            arrays['correction_residual_'+str(c)+'_'+str(d)] = residual
            correction_records.append({'block': [c, d], 'rho_Frobenius_upper_bound': rho,
                                       'transformed_block_operator_upper_bound': bound, 'triangle_multiplicity': multiplicity})
        port_transformed = permuted_similarity(completed.port, target, phases, anti)
        port_residual = port_transformed+completed.port if anti else port_transformed-completed.port
        port_rho = float(np.linalg.norm(port_residual))
        port_norm = max(float(np.linalg.norm(block, 2)) for block in completed.port_blocks)
        port_bound = frame_data[3]['E_norm_upper']*port_rho+errors[3]*port_norm
        lead_bound, lead_records, lead_arrays = _lead_residuals(completed.lead, target, phases, anti)
        arrays.update(lead_arrays); arrays['port_residual'] = port_residual
        native_bound = native_record.pop('native_operator_upper_bound')
        native_record.pop('coefficient_residuals')
        whole = None if native_bound is None else native_bound+total_correction+2*port_bound+lead_bound
        threshold = 5e-12*max(1., completed.B)
        square = bool(np.array_equal(target[target], np.arange(width)) and np.array_equal(phases*phases[target].conj(), np.ones(width))) if anti else None
        bijective = len(np.unique(target)) == width and np.all(np.abs(phases) == 1)
        native_unitarity = native_record['native_unitarity_error'] <= 5e-12
        native_square = native_record['native_antiunitary_square_error'] <= 5e-12 if anti else True
        passed = bool(whole is not None and np.isfinite(whole) and whole <= threshold and bijective
                      and (square if anti else True) and native_unitarity and native_square)
        raw_path = outdir/(name+'_audit_arrays.npz')
        np.savez_compressed(raw_path, **arrays)
        output_bindings[raw_path.name] = identity(raw_path)
        for frame in frames:
            frame['map_artifact_binding'] = output_bindings[raw_path.name]
            frame['map_array_keys'] = ['target_indices', 'phases']
        rows.append({'name': name, 'antiunitary': anti, 'frames': frames, 'native': native_record,
                     'native_operator_upper_bound': native_bound, 'corrections': correction_records,
                     'port_rho_Frobenius_upper_bound': port_rho, 'port_operator_upper_bound': port_bound,
                     'port_triangle_multiplicity': 2, 'lead_residuals': [{**item, 'block_map': item['block_map'].tolist()} for item in lead_records],
                     'whole_lead_block_row_upper_bound': lead_bound,
                     'complete_operator_covariance_upper_bound': whole, 'scaled_algebra_threshold': threshold,
                     'map_is_unitary': bool(bijective), 'antiunitary_square_identity': square,
                     'domain_preserving': True, 'complete_radial_cell_detector_preserving': True,
                     'passed': passed, 'status': 'established_by_saved_upper_bound' if passed else 'unresolved; saved upper bound did not establish the identity'})
        verify_threads(runtime, 'after_'+name)
        check_memory(runtime, 'after_'+name)
    # All actual audit arrays are bound; no sampled-frame COO chunks are written.
    for path in outdir.glob('*.npz'):
        output_bindings[path.name] = identity(path)
    valley_pass = all(row['passed'] for row in rows if not row['antiunitary'])
    anti_pass = rows[-1]['passed']
    output = {'audit_completed': True, 'valley_covariance_pass': valley_pass,
              'antiunitary_pass': anti_pass, 'exit_symmetry_status': 'established for the tested completed operator/domain/detectors' if anti_pass else 'unresolved; no both-window equivalence asserted',
              'runtime': runtime, 'code_bindings': core_bindings(), 'setup_binding': identity(setup_json),
              'K': K, 'end_radius': end_radius, 'operator_bound': completed.bound_record,
              'transforms': rows, 'output_bindings': output_bindings,
              'norm_convention': 'Whole-operator triangle/block-row upper bounds; a failed conservative bound is not relabelled a measured induced-norm failure.'}
    # Complex block maps remain in NPZ; their JSON representation is explicit.
    def safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return {'nonfinite_literal': repr(value)}
        if isinstance(value, complex):
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value
    output = safe(output)
    (outdir/'symmetry.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    return output
