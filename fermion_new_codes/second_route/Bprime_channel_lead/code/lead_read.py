"""Saved-only channel-lead comparisons and conditional full-window costing.

No generator, propagator, field solve or new physical input is invoked.
Every scientific JSON/NPZ result stays on the admitted box.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
from lead_runtime import (admit, identity, require_identity, code_bindings,
                          check_memory, verify_threads, SPEC_SHA256)
from lead_prepare import save_record, recheck_artifacts, clean


def report(path, sha):
    binding = require_identity(path, sha)
    value = json.loads(Path(path).read_text())
    if value['bindings']['spec']['sha256'] != SPEC_SHA256 or value['bindings']['code'] != code_bindings():
        raise ValueError('Saved run has different science/source identities')
    recheck_artifacts(value)
    value['_reader_binding'] = binding
    return value, binding


def payload(readout, keys):
    import numpy as np
    item = readout['array_binding']
    require_identity(item['path'], item['sha256'])
    tree = readout['array_tree']
    for key in keys:
        tree = tree[key]
    if not isinstance(tree, dict) or 'array_key' not in tree:
        raise ValueError('Requested saved payload is not an array')
    with np.load(item['path'], allow_pickle=False) as archive:
        return archive[tree['array_key']]


def state(readout):
    from lead_run import load_state
    item = readout['state_binding']
    require_identity(item['path'], item['sha256'])
    return load_state(item['path'])


def row_gram(value):
    return value.conj()@value.T


def residual_summary(value):
    import numpy as np
    flat = np.asarray(value).ravel()
    if not flat.size:
        return {'shape': list(value.shape), 'maximum_entry': 0., 'locator': None, 'value': None}
    index = int(np.argmax(np.abs(flat)))
    result = {'shape': list(value.shape), 'maximum_entry': float(np.abs(flat[index])),
            'locator': list(np.unravel_index(index, value.shape)), 'value': complex(flat[index]),
            'Frobenius_norm': float(np.linalg.norm(value)), 'all_finite': bool(np.all(np.isfinite(value)))}
    if flat.size <= 64:
        result['literal_entries'] = np.asarray(value).tolist()
    return result


def controls_evidence(value):
    """Render saved evidence only; original recorder verdicts are not recomputed."""
    import numpy as np
    item = value['output_bindings'].get('controls_arrays.npz', value['output_bindings'].get('partial_arrays'))
    if item is None:
        raise ValueError('Saved manufactured array binding required')
    require_identity(item['path'], item['sha256'])
    rows, arrays = [], {}
    with np.load(item['path'], allow_pickle=False) as archive:
        def leaves(node, path):
            if isinstance(node, dict) and 'array_key' in node:
                key = node['array_key']
                arr = archive[key]
                arrays[key] = arr
                return [{'leaf': path, 'array_key': key, 'summary': residual_summary(arr)}]
            if isinstance(node, dict):
                return [leaf for key, child in node.items() for leaf in leaves(child, path+'.'+str(key))]
            if isinstance(node, list):
                return [leaf for index, child in enumerate(node) for leaf in leaves(child, path+'.'+str(index))]
            return [{'leaf': path, 'value': node}]
        for case in value['cases']:
            expected, observed = leaves(case['expected'], 'expected'), leaves(case['observed'], 'observed')
            pairs = []
            by_name = {leaf['leaf'].removeprefix('expected.'): leaf for leaf in expected if 'array_key' in leaf}
            for leaf in observed:
                old = by_name.get(leaf['leaf'].removeprefix('observed.'))
                if old and 'array_key' in leaf:
                    a, b = arrays[old['array_key']], arrays[leaf['array_key']]
                    if a.shape == b.shape:
                        delta = b-a
                        key = case['name']+'__reader_difference__'+leaf['array_key']
                        arrays[key] = delta
                        pairs.append({'expected': old['array_key'], 'observed': leaf['array_key'],
                                      'difference_array': key, 'residual': residual_summary(delta)})
            rows.append({'case': case['name'], 'checks': case['checks'], 'passed': case['passed'],
                         'expected': expected, 'observed': observed, 'matched_array_differences': pairs})
        inventory = []
        for key in archive.files:
            inventory.append({'array_key': key, 'summary': residual_summary(archive[key])})
    return {'rows': rows, 'array_inventory': inventory, 'all_passed': bool(value.get('all_passed', False)),
            'scope': 'Saved manufactured evidence; original control predicates and negative-control interpretation unchanged'}, arrays


def gram_norm(gram):
    import numpy as np
    return float(np.sqrt(max(0., float(np.linalg.eigvalsh(gram)[-1]))))


def hybrid_shapes(record):
    layout = record['layout']
    return layout['native_size'], layout['lead_shape']


def aligned_hybrid_difference(a, ar, b, br):
    """Complete state-difference Gram in common physical native/channel space."""
    import numpy as np
    from lead_channels import chain_labels
    na, shapea = hybrid_shapes(ar)
    nb, shapeb = hybrid_shapes(br)
    if na != nb or len(a) != len(b):
        raise ValueError('Hybrid native/input spaces differ')
    difference = b[:, :nb]-a[:, :na]
    gram = row_gram(difference)
    lead_a, lead_b = a[:, na:].reshape(len(a), *shapea), b[:, nb:].reshape(len(b), *shapeb)
    la, lb = chain_labels(ar['arguments']['K']), chain_labels(br['arguments']['K'])
    lookup_a = {tuple(label): index for index, label in enumerate(la)}
    lookup_b = {tuple(label): index for index, label in enumerate(lb)}
    maximum, locator = float(np.max(np.abs(difference))), {'region': 'native', 'flat_index': int(np.argmax(np.abs(difference)))}
    for label in sorted(set(lookup_a)|set(lookup_b)):
        ia, ib = lookup_a.get(label), lookup_b.get(label)
        if ia is not None and ib is not None:
            left, right = lead_a[:, ia], lead_b[:, ib]
            common = min(left.shape[1], right.shape[1])
            delta = right[:, :common]-left[:, :common]
            gram += row_gram(delta.reshape(len(a), -1))
            if delta.size and np.max(np.abs(delta)) > maximum:
                maximum = float(np.max(np.abs(delta)))
                locator = {'region': 'common_lead', 'chain_label': list(label), 'flat_index': int(np.argmax(np.abs(delta)))}
            for side, tail in (('left', left[:, common:]), ('right', right[:, common:])):
                if tail.size:
                    gram += row_gram(tail.reshape(len(a), -1))
                    if np.max(np.abs(tail)) > maximum:
                        maximum = float(np.max(np.abs(tail)))
                        locator = {'region': 'zero_extended_tail', 'side': side,
                                   'chain_label': list(label), 'first_cell': common,
                                   'flat_index': int(np.argmax(np.abs(tail)))}
        else:
            extra = lead_a[:, ia] if ia is not None else lead_b[:, ib]
            gram += row_gram(extra.reshape(len(a), -1))
            if extra.size and np.max(np.abs(extra)) > maximum:
                maximum = float(np.max(np.abs(extra)))
                locator = {'region': 'added_or_removed_chain', 'chain_label': list(label), 'flat_index': int(np.argmax(np.abs(extra)))}
    return gram, {'maximum_entry': maximum, 'locator': locator}


def pair_compare(left, right, kind):
    import numpy as np
    from lead_inputs import opnorm
    if not left.get('run_completed') or not right.get('run_completed'):
        raise ValueError('Two completed saved trajectories required')
    clocks_a = [x['clock'] for x in left['readouts']]
    clocks_b = [x['clock'] for x in right['readouts']]
    if clocks_a != clocks_b:
        raise ValueError('Comparison clocks differ')
    if kind == 'time' and left['physical_operator_association'] != right['physical_operator_association']:
        raise ValueError('Temporal refinement changes the physical operator')
    if kind == 'time':
        def fine_flag(value):
            return (value['arguments']['case'].endswith('-fine') if value['stage'] == 'hybrid'
                    else bool(value['arguments']['fine']))
        if left['stage'] != right['stage'] or fine_flag(left) or not fine_flag(right):
            raise ValueError('Temporal comparison requires the declared coarse then fine members')
        coarse_binding = right['bindings'].get('coarse')
        original_binding = left['_reader_binding']
        if (coarse_binding is None or coarse_binding['sha256'] != original_binding['sha256'] or
                Path(coarse_binding['path']).resolve() != Path(original_binding['path']).resolve()):
            raise ValueError('Fine producer is not bound to this exact coarse run')
        for value in (left, right):
            plan = value['output_bindings']['plan']
            require_identity(plan['path'], plan['sha256'])
        with np.load(left['output_bindings']['plan']['path'], allow_pickle=False) as aplan, np.load(right['output_bindings']['plan']['path'], allow_pickle=False) as bplan:
            at, bt = aplan['times'], bplan['times']
            if (len(bt) != 2*len(at)-1 or not np.array_equal(bt[::2], at) or
                    not np.array_equal(bt[1::2], at[:-1]+np.diff(at)/2) or
                    not np.array_equal(bplan['intervals'], np.diff(bt)) or
                    not np.array_equal(bplan['readout_times'], aplan['readout_times']) or
                    not np.array_equal(bplan['readout_indices'], 2*aplan['readout_indices']) or
                    float(aplan['B']) != float(bplan['B']) or int(aplan['degree']) != int(bplan['degree'])):
                raise ValueError('Saved fine plan is not the exact prescribed interval bisection')
    if kind in ('K', 'length'):
        if left['stage'] != right['stage'] or left['bindings']['native']['sha256'] != right['bindings']['native']['sha256']:
            raise ValueError('Convergence comparison changes its native input association')
        if left['bindings']['fields']['sha256'] != right['bindings']['fields']['sha256']:
            raise ValueError('Convergence comparison changes its radial fields')
        if kind == 'K':
            if (left['stage'] != 'hybrid' or {left['arguments']['case'], right['arguments']['case']} !=
                    {'pilot-k10-200', 'pilot-k12-200'}):
                raise ValueError('Only frozen K10/K12 actual-input comparison is admitted')
        elif left['stage'] == 'hybrid':
            if {left['arguments']['case'], right['arguments']['case']} != {'pilot-k10-200', 'pilot-k10-400'}:
                raise ValueError('Only frozen actual-input doubled-lead comparison is admitted')
        elif left['stage'] == 'b11':
            if ({left['arguments']['arm'], right['arguments']['arm']} != {'lead200', 'lead400'} or
                    left['arguments']['family'] != right['arguments']['family'] or
                    left['arguments']['kappa'] != right['arguments']['kappa']):
                raise ValueError('B11 doubled-lead channel association differs')
        else:
            raise ValueError('Unknown length-convergence roster')
    rows, arrays = [], {}
    for index, (ra, rb) in enumerate(zip(left['readouts'], right['readouts'])):
        a, b = state(ra), state(rb)
        if left['stage'] == right['stage'] == 'hybrid':
            difference_gram, locator = aligned_hybrid_difference(a, left, b, right)
        elif left['stage'] == right['stage'] == 'b11' and kind == 'length':
            common = min(a.shape[1], b.shape[1])
            common_delta = b[:, :common]-a[:, :common]
            difference_gram = row_gram(common_delta)
            common_gram = difference_gram.copy()
            if a.shape[1] > common:
                difference_gram += row_gram(a[:, common:])
            if b.shape[1] > common:
                difference_gram += row_gram(b[:, common:])
            locator = {'common_dofs': common, 'common_residual': residual_summary(common_delta)}
            arrays['zero_extended_full_state_difference_Gram_'+str(index)] = difference_gram.copy()
            locator['zero_extended_full_state_norm_diagnostic'] = gram_norm(difference_gram)
            difference_gram = common_gram
        else:
            if a.shape != b.shape:
                raise ValueError('Non-hybrid state spaces differ')
            delta = b-a
            difference_gram = row_gram(delta)
            locator = {'maximum_entry': float(np.max(np.abs(delta))), 'flat_index': int(np.argmax(np.abs(delta)))}
        gram_difference = row_gram(b)-row_gram(a)
        ea, eb = payload(ra, ['numerics', 'energy_Gram']), payload(rb, ['numerics', 'energy_Gram'])
        e0 = payload(left['readouts'][0], ['numerics', 'energy_Gram'])
        state_error, gram_error = gram_norm(difference_gram), opnorm(gram_difference)
        energy_error = opnorm(eb-ea)/max(1., opnorm(e0))
        state_target, gram_target = (5e-3, 1e-2) if kind == 'K' else (1e-8, 1e-8)
        checks = {'state': state_error <= state_target, 'Gram': gram_error <= gram_target}
        if kind == 'time':
            checks['energy'] = energy_error <= 1e-8
        output_difference = None
        if left['stage'] == 'b11':
            ga, gb = payload(ra, ['output_Gram']), payload(rb, ['output_Gram'])
            output_difference = opnorm(gb-ga)
            if kind == 'length':
                checks['complete_output_Gram'] = output_difference <= 1e-8
            arrays['output_Gram_difference_'+str(index)] = gb-ga
        if left['stage'] == 'hybrid':
            ga = payload(ra, ['hybrid', 'native_Gram'])
            gb = payload(rb, ['hybrid', 'native_Gram'])
            arrays['native_Gram_difference_'+str(index)] = gb-ga
            inventory_errors = {'native': opnorm(gb-ga)}
            for name in ('core_Gram', 'shell_Grams', 'lead_Gram'):
                ga, gb = payload(ra, ['hybrid', name]), payload(rb, ['hybrid', name])
                diff = gb-ga
                arrays[name+'_difference_'+str(index)] = diff
                inventory_errors[name] = max(opnorm(x) for x in diff) if diff.ndim == 3 else opnorm(diff)
            bins_a = payload(ra, ['bins', 'far_bin_Grams'])
            bins_b = payload(rb, ['bins', 'far_bin_Grams'])
            bin_diff = bins_b[:min(len(bins_a), len(bins_b))]-bins_a[:min(len(bins_a), len(bins_b))]
            arrays['common_bin_Gram_differences_'+str(index)] = bin_diff
            inventory_errors['common_bins'] = max((opnorm(x) for x in bin_diff), default=0.)
            output_diff = np.sum(bins_b, axis=0)-np.sum(bins_a, axis=0)
            arrays['far_complete_Gram_difference_'+str(index)] = output_diff
            inventory_errors['far_complete'] = opnorm(output_diff)
        else:
            inventory_errors = {}
        arrays.update({'state_difference_Gram_'+str(index): difference_gram,
                       'complete_Gram_difference_'+str(index): gram_difference,
                       'energy_Gram_difference_'+str(index): eb-ea})
        rows.append({'clock': ra['clock'], 'state_error': state_error, 'Gram_error': gram_error,
                     'state_space': 'common radial region' if left['stage'] == 'b11' and kind == 'length' else 'complete aligned physical state, added labels zero extended',
                     'energy_error': energy_error, 'output_Gram_error': output_difference,
                     'state_target': state_target, 'Gram_target': gram_target,
                     'checks': checks, 'all_passed': all(checks.values()), 'locator': locator})
        rows[-1]['inventory_Gram_errors'] = inventory_errors
        del a, b
    informative = (left.get('transport_gate_informative', True) and right.get('transport_gate_informative', True))
    producer_gates = all(value['all_passed'] and value.get('finite_end_certificate', {'passed': True})['passed'] for value in (left, right))
    return {'kind': kind, 'rows': rows, 'numerical_comparison_passed': all(x['all_passed'] for x in rows),
            'producer_numerical_and_end_gates_passed': producer_gates,
            'transport_informative': bool(informative),
            'all_passed': bool(producer_gates and informative and all(x['all_passed'] for x in rows))}, arrays


def common_ball(states):
    import numpy as np
    coords = np.arange(-40, 41)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    mask = x*x+y*y+z*z <= 1600
    native = states[:, :4*81**3].reshape(len(states), 4, 81, 81, 81)
    return native[:, :, mask].reshape(len(states), -1)


def native_reference_compare(hybrid, reference):
    import numpy as np
    from lead_inputs import opnorm
    if hybrid['arguments']['case'].removesuffix('-fine') != 'variable' or reference['stage'] != 'native-reference':
        raise ValueError('Frozen variable-flow four-probe control only')
    if (hybrid['arguments']['case'].endswith('-fine') != reference['arguments']['fine'] or
            hybrid['bindings']['native']['sha256'] != reference['bindings']['native']['sha256'] or
            hybrid['bindings']['fields']['sha256'] != reference['bindings']['fields']['sha256']):
        raise ValueError('Native response reference has different input/field/refinement associations')
    rows, arrays = [], {}
    if [x['clock'] for x in hybrid['readouts']] != [x['clock'] for x in reference['readouts']]:
        raise ValueError('Reference observation clocks differ')
    for index, (hr, rr) in enumerate(zip(hybrid['readouts'], reference['readouts'])):
        own = common_ball(state(hr))
        target = payload(rr, ['native_ball'])
        if hr['clock'] == 0 and not np.array_equal(own, target):
            raise ValueError('Native response initial common-domain amplitudes differ')
        difference = own-target
        error = opnorm(difference)
        arrays.update({'difference_'+str(index): difference,
                       'Gram_difference_'+str(index): row_gram(own)-row_gram(target)})
        rows.append({'clock': hr['clock'], 'state_error': error, 'target': 5e-3,
                     'residual': residual_summary(difference), 'passed': error <= 5e-3})
    informative = hybrid['transport_gate_informative']
    producer_gates = all(value['all_passed'] and value.get('finite_end_certificate', {'passed': True})['passed'] for value in (hybrid, reference))
    return {'rows': rows, 'informative': informative,
            'producer_numerical_and_end_gates_passed': producer_gates,
            'all_passed': bool(producer_gates and informative and all(x['passed'] for x in rows)),
            'scope': 'Four fixed DFT probes through2L; later common bias and full128 remain untested'}, arrays


def _ball_valley_x(columns):
    import numpy as np
    from lead_native import valley_matrix
    coords = np.arange(-40, 41)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    mask = x*x+y*y+z*z <= 1600
    phase = 1-2*(x[mask] % 2)
    matrix = valley_matrix((1, 0, 0))
    value = columns.reshape(len(columns), 4, -1)
    return (np.einsum('ij,cjp->cip', matrix, value)*phase[None, None]).reshape(len(columns), -1)


def _full_valley_x(columns, record):
    import numpy as np
    from lead_native import valley_matrix
    from lead_channels import valley_permutation
    size, lead_shape = hybrid_shapes(record)
    result = np.empty_like(columns)
    native = columns[:, :size].reshape(len(columns), 4, 81, 81, 81)
    phase = 1-2*((np.arange(81)-40) % 2)
    result[:, :size] = (np.einsum('ij,cjxyz->cixyz', valley_matrix((1, 0, 0)), native)*
                       phase[None, None, :, None, None]).reshape(len(columns), -1)
    target, phases = valley_permutation(record['arguments']['K'], (1, 0, 0))
    old = columns[:, size:].reshape(len(columns), *lead_shape)
    new = result[:, size:].reshape(len(columns), *lead_shape)
    for source, dest in enumerate(target):
        new[:, dest//4, :, dest % 4] = phases[source]*old[:, source//4, :, source % 4]
    return result


def _coherent64_transform():
    import numpy as np
    transform = np.zeros((64, 32), dtype=np.complex128)
    transform[:32] = np.eye(32)
    for column in range(16):
        original_positive = bin(column//2).count('1') % 2 == 0
        p, m = (column, column+16) if original_positive else (column+16, column)
        transform[32+column, p], transform[32+column, m] = 1/np.sqrt(2), 1j/np.sqrt(2)
        transform[48+column, p], transform[48+column, m] = 1/np.sqrt(2), -1j/np.sqrt(2)
    return transform


def _two_block_gram(original, opposite):
    import numpy as np
    return np.block([[row_gram(original), original.conj()@opposite.T],
                     [opposite.conj()@original.T, row_gram(opposite)]])


def flat_compare(hybrid, fourier_records):
    import numpy as np
    from lead_inputs import metric_normalizer, opnorm
    from lead_run import load_state, manufactured_flat_inputs
    from lead_references import free_image_bound
    if hybrid['arguments']['case'] not in ('flat-supply', 'flat-manufactured') or len(fourier_records) != 16:
        raise ValueError('Frozen flat16 plus all16 exact Fourier columns required')
    expected_family = 'manufactured' if hybrid['arguments']['case'] == 'flat-manufactured' else 'supply'
    by_column = {r['arguments']['column']: r for r in fourier_records}
    if sorted(by_column) != list(range(16)) or any(r['arguments']['fourier_family'] != expected_family for r in fourier_records):
        raise ValueError('Fourier reference identities/roster differ')
    initial = load_state(hybrid['output_bindings']['initial']['path'])
    native_initial = common_ball(initial)
    reference_gates = all(r['all_passed'] and r['certificate']['passed'] for r in fourier_records)
    reference_certificate = fourier_records[0]['certificate']
    if any(r['certificate'] != reference_certificate for r in fourier_records):
        raise ValueError('Fourier columns do not share the frozen uniform block certificate')
    for column in range(16):
        binding = by_column[column]['output_bindings']['initial']
        require_identity(binding['path'], binding['sha256'])
        with np.load(binding['path'], allow_pickle=False) as archive:
            if not np.array_equal(archive['initial_native'], initial[column, :4*81**3]):
                raise ValueError('Fourier/hybrid exact initial physical column differs')
    matrices, arrays, rows, current_rows = {}, {}, [], []
    coord = np.arange(-40, 41)
    xyz = np.meshgrid(coord, coord, coord, indexing='ij')
    radius2 = sum(component*component for component in xyz)
    selected_radius2 = radius2[radius2 <= 40**2]
    regions = {'core_r_le_32': np.tile(selected_radius2 <= 32**2, 4)}
    for low, high in ((32, 34), (34, 36), (36, 38), (38, 40)):
        regions['shell_'+str(low)+'_'+str(high)] = np.tile((selected_radius2 > low**2)&(selected_radius2 <= high**2), 4)
    for index, hr in enumerate(hybrid['readouts']):
        if hr['clock'] == 0:
            target = native_initial.copy()
        else:
            target_columns = []
            for column in range(16):
                reference = by_column[column]
                clocks = [float(x['clock']) for x in reference['diagnostics']]
                if hr['clock'] not in clocks:
                    raise ValueError('Fourier clock association failed')
                which = clocks.index(hr['clock'])
                binding = reference['output_bindings']['clock_'+str(which)]
                require_identity(binding['path'], binding['sha256'])
                with np.load(binding['path'], allow_pickle=False) as archive:
                    target_columns.append(archive['native_ball'])
            target = np.array(target_columns)
        actual_full = state(hr)
        actual = common_ball(actual_full)
        blocks = {'original': (native_initial, actual, target)}
        if expected_family == 'manufactured':
            audit = hybrid['bindings']['audit']
            audit_record = json.loads(Path(audit['path']).read_text())
            if not audit_record['symmetry']['valley_covariance_pass']:
                raise ValueError('Actual whole-operator valley audit is required before generating orbit outputs')
            # Original ordering is lexicographic nu, then k1,k3. Flipping
            # nu_x is index xor8, and Mx S_(nu xor x)=S_nu exactly.
            source = np.arange(16)^8
            opposite_i = _ball_valley_x(native_initial[source])
            opposite_a = _ball_valley_x(actual[source])
            opposite_r = _ball_valley_x(target[source])
            blocks['opposite'] = (opposite_i, opposite_a, opposite_r)
            plus_i = np.empty_like(native_initial)
            minus_i = np.empty_like(native_initial)
            plus_a, minus_a = np.empty_like(actual), np.empty_like(actual)
            plus_r, minus_r = np.empty_like(target), np.empty_like(target)
            for column in range(16):
                nu = column//2
                original_positive = (bin(nu).count('1') % 2 == 0)
                for original, opposite, plus, minus in ((native_initial, opposite_i, plus_i, minus_i),
                                                        (actual, opposite_a, plus_a, minus_a),
                                                        (target, opposite_r, plus_r, minus_r)):
                    p, m = (original[column], opposite[column]) if original_positive else (opposite[column], original[column])
                    plus[column] = (p+1j*m)/np.sqrt(2)
                    minus[column] = (p-1j*m)/np.sqrt(2)
            blocks['plus_i'] = (plus_i, plus_a, plus_r)
            blocks['minus_i'] = (minus_i, minus_a, minus_r)
            arrays['generated_raw64_common_Gram_'+str(index)] = row_gram(np.concatenate([block[1] for block in blocks.values()]))
            arrays['generated_raw64_reference_Gram_'+str(index)] = row_gram(np.concatenate([block[2] for block in blocks.values()]))
            transform = _coherent64_transform()
            opposite_full = _full_valley_x(actual_full[source], hybrid)
            base_gram = _two_block_gram(actual_full, opposite_full)
            arrays['generated64_from_two_signed_blocks'] = transform
            arrays['generated_flip_x_source_indices'] = source
            arrays['generated_raw64_complete_Gram_'+str(index)] = transform.conj()@base_gram@transform.T
            arrays['generated_raw64_initial_metric'] = row_gram(np.concatenate([block[0] for block in blocks.values()]))
            size, lead_shape = hybrid_shapes(hybrid)
            lead_a = actual_full[:, size:].reshape(len(actual_full), *lead_shape)
            lead_b = opposite_full[:, size:].reshape(len(actual_full), *lead_shape)
            bin_limits = payload(hr, ['bins', 'far_bin_cell_ranges'])
            generated_bins = []
            for low, high in bin_limits:
                ga = lead_a[:, :, low:high].reshape(16, -1)
                gb = lead_b[:, :, low:high].reshape(16, -1)
                generated_bins.append(transform.conj()@_two_block_gram(ga, gb)@transform.T)
            arrays['generated_raw64_far_bin_Grams_'+str(index)] = np.array(generated_bins)
            del opposite_full
        for name, (start, own, ref) in blocks.items():
            raw_metric = row_gram(start)
            difference = own-ref
            arrays[name+'_metric'] = raw_metric
            arrays[name+'_metric_eigenvalues'] = np.linalg.eigvalsh(raw_metric)
            arrays[name+'_difference_'+str(index)] = difference
            arrays[name+'_Gram_difference_'+str(index)] = row_gram(own)-row_gram(ref)
            raw_image = free_image_bound(reference_certificate['side'], reference_certificate['R_in'],
                                        reference_certificate['clock'], opnorm(start))
            regional = {}
            for region, mask in regions.items():
                raw_delta = difference[:, mask]
                raw_inventory_difference = row_gram(own[:, mask])-row_gram(ref[:, mask])
                regional[region] = {'raw_state_error': opnorm(raw_delta),
                                    'raw_inventory_Gram_error': opnorm(raw_inventory_difference)}
                arrays[name+'_'+region+'_raw_difference_Gram_'+str(index)] = row_gram(raw_delta)
                arrays[name+'_'+region+'_raw_inventory_Gram_difference_'+str(index)] = raw_inventory_difference
            currents = {}
            if name == 'original':
                # Always preserve the already saved raw16 currents, even
                # when common-metric normalization is unavailable.
                currents['signed_port_current_into_ball'] = payload(hr, ['hybrid', 'signed_port_current_into_ball'])
                for cut in ('core_r_le_32', 'ball_r_le_40', 'far_lead_r_gt_800'):
                    for quantity in ('probability_current', 'symmetric_energy_current'):
                        currents[cut+'_'+quantity] = payload(hr, ['projector_fluxes', cut, quantity])
                for quantity, current in currents.items():
                    arrays['original_'+quantity+'_raw_'+str(index)] = current
            try:
                _, metric = metric_normalizer(start)
                factor = metric['right_factor']
                normalized_difference = factor.T@(own-ref)
                error = opnorm(normalized_difference)
                admitted = True
                arrays[name+'_metric'] = metric['metric']
                arrays[name+'_metric_eigenvalues'] = metric['eigenvalues']
                arrays[name+'_metric_right_factor'] = factor
                arrays[name+'_difference_'+str(index)] = own-ref
                arrays[name+'_Gram_difference_'+str(index)] = row_gram(own)-row_gram(ref)
                normalized_image = free_image_bound(reference_certificate['side'], reference_certificate['R_in'],
                                                     reference_certificate['clock'], opnorm(factor.T@start))
                for region, mask in regions.items():
                    normalized_delta = normalized_difference[:, mask]
                    normalized_inventory_difference = factor.conj().T@(
                        row_gram(own[:, mask])-row_gram(ref[:, mask]))@factor
                    regional[region]['metric_normalized_state_error'] = opnorm(normalized_delta)
                    regional[region]['metric_normalized_inventory_Gram_error'] = opnorm(normalized_inventory_difference)
                    arrays[name+'_'+region+'_normalized_difference_Gram_'+str(index)] = row_gram(normalized_delta)
                    arrays[name+'_'+region+'_normalized_inventory_Gram_difference_'+str(index)] = normalized_inventory_difference
            except ValueError as exc:
                error, admitted = None, False
                normalized_image = None
                matrices[name+'_metric_refusal'] = str(exc)
            for quantity, current in currents.items():
                normalized_current = factor.conj().T@current@factor if admitted else None
                if normalized_current is not None:
                    arrays['original_'+quantity+'_metric_normalized_'+str(index)] = normalized_current
                for column in range(16):
                    current_rows.append({'clock': hr['clock'], 'scope': 'original16 block only',
                        'quantity': quantity, 'column': column,
                        'raw_diagonal': complex(current[column, column]),
                        'metric_normalized_diagonal': None if normalized_current is None else complex(normalized_current[column, column])})
            rows.append({'clock': hr['clock'], 'block': name, 'metric_admitted': admitted,
                         'metric_normalized_state_error': error, 'target': 5e-3,
                         'raw_residual': residual_summary(difference),
                         'normalized_residual': residual_summary(normalized_difference) if admitted else None,
                         'regional_diagnostics': regional,
                         'raw_image_certificate': raw_image, 'normalized_image_certificate': normalized_image,
                         'passed': bool(admitted and raw_image['passed'] and normalized_image['passed'] and error <= 5e-3)})
    return {'rows': rows, 'metric_refusals': matrices, 'signed_current_rows': current_rows,
            'current_scope': 'Original16 signed matrices and metric-transformed diagonals only; generated coherent blocks add response/Gram tests, not new current claims.',
            'reference_numerical_and_image_gates_passed': reference_gates,
            'all_passed': bool(reference_gates and hybrid['all_passed'] and all(x['passed'] for x in rows)),
            'scope': 'Separate rank-admitted16-column blocks; dependent raw64 union is never whitened'}, arrays


def b11_compare(records):
    import numpy as np
    source_root = Path('<internal>/b11')
    source_sha = {'F1': '17e0864ccec7a3807c1a8fe3bc08eff6a839cb7b496ee41211c6ea1f7e134ec8',
                  'F2': 'a8b55c35e82b110b5cc196b09103f32ad70787452e009c386387b4d0e6584505'}
    by_case = {}
    for value in records:
        if value['stage'] != 'b11' or not value.get('run_completed'):
            raise ValueError('Only completed B11 records are admitted')
        a = value['arguments']
        key = (a['family'], a['kappa'], a['arm'], a['fine'])
        if key in by_case:
            raise ValueError('Duplicate B11 roster member')
        by_case[key] = value
    expected = {(family, k, arm, False) for family in ('F1', 'F2') for k in (1, 2, 3)
                for arm in ('cap2', 'lead200', 'lead400', 'cap50')}
    expected |= {('F1', 1, 'lead200', True), ('F2', 3, 'cap50', True)}
    if set(by_case) != expected:
        raise ValueError('Complete frozen24-arm plus2-refinement B11 roster required')
    if not all(value['all_passed'] for value in by_case.values()):
        raise ValueError('Own producer numerical gates must pass before source consequences are opened')
    rows, arrays, bindings, comparisons = [], {}, {}, []
    arm_name = {'cap2': 'cap2', 'lead200': 'lead', 'lead400': 'lead2', 'cap50': 'cap50'}
    for family in ('F1', 'F2'):
        path = source_root/('b11_'+family+'_s16.npz')
        bindings[family] = require_identity(path, source_sha[family])
        # Every own record has passed the source-energy/field input stage.
        # Source numerical consequences are first opened here, after own runs.
        with np.load(path, allow_pickle=False) as source:
            if not np.array_equal(source['t'], 4*np.arange(401)):
                raise ValueError('B11 source clock association differs')
            for k in (1, 2, 3):
                prefix = family+'_k'+str(k)
                source_g = {arm: source['k'+str(k)+'_'+name+'_Gbins'][-1]
                            for arm, name in arm_name.items()}
                source_p = {arm: np.linalg.eigvalsh(g, UPLO='L') for arm, g in source_g.items()}
                population = source_p['lead200'] > 1e-2
                own_p = {}
                checks, retained_rows, spectral_rows, defects = {}, [], [], {}
                for arm in arm_name:
                    run = by_case[(family, k, arm, False)]
                    if run['readouts'][-1]['clock'] != 1600.:
                        raise ValueError('B11 own final clock is not100L')
                    own_g = payload(run['readouts'][-1], ['output_Gram'])
                    own_p[arm] = np.linalg.eigvalsh(own_g, UPLO='L')
                    defects[arm] = {'source_raw_Frobenius': float(np.linalg.norm(source_g[arm]-source_g[arm].conj().T)),
                                    'own_raw_Frobenius': float(np.linalg.norm(own_g-own_g.conj().T))}
                    difference = own_p[arm]-source_p[arm]
                    denominator = source_p[arm]
                    entry_available = population & (denominator > 0)
                    available = bool(np.any(population) and np.all(entry_available[population]))
                    relative = np.full(6, np.nan)
                    relative[entry_available] = np.abs(difference[entry_available])/denominator[entry_available]
                    checks[arm+'_source_absolute'] = bool(np.any(population) and np.max(np.abs(difference[population])) <= 1e-4)
                    checks[arm+'_source_relative'] = bool(available and np.max(relative[population]) <= .0025)
                    arrays.update({prefix+'_'+arm+'_source_final_Gram': source_g[arm],
                                   prefix+'_'+arm+'_own_final_Gram': own_g,
                                   prefix+'_'+arm+'_source_p': source_p[arm],
                                   prefix+'_'+arm+'_own_p': own_p[arm],
                                   prefix+'_'+arm+'_difference': difference,
                                   prefix+'_'+arm+'_relative_available': entry_available,
                                   prefix+'_'+arm+'_relative': relative})
                    for ordered in range(6):
                        spectral_rows.append({'arm': arm, 'ordered_index': ordered,
                            'fixed_population': bool(population[ordered]),
                            'source_p': float(source_p[arm][ordered]), 'own_p': float(own_p[arm][ordered]),
                            'signed_difference': float(difference[ordered]),
                            'relative_available': bool(entry_available[ordered]),
                            'relative_error': float(relative[ordered]) if entry_available[ordered] else None})
                    source_map = np.array(run['input_metadata']['source_column_map'])
                    individual = np.array(run['input_metadata']['individual_comparison_available'], dtype=bool)
                    source_retained = source['k'+str(k)+'_'+arm_name[arm]+'_ret']
                    for clock in (800., 1600.):
                        own_readout = next(r for r in run['readouts'] if r['clock'] == clock)
                        weights = np.real(np.diag(payload(own_readout, ['retained_Gram'])))
                        row_index = int(clock/4)
                        for column in range(6):
                            mapped = int(source_map[column])
                            source_value = float(source_retained[row_index, mapped]) if individual[column] and mapped >= 0 else None
                            ratio = float(weights[column]/source_value) if source_value is not None and source_value != 0 else None
                            retained_rows.append({'arm': arm, 'clock': clock, 'own_column': column,
                                'source_column': mapped if individual[column] else None,
                                'own_weight': float(weights[column]), 'source_weight': source_value,
                                'signed_difference': None if source_value is None else float(weights[column]-source_value),
                                'own_over_source': ratio,
                                'individual_available': bool(individual[column]),
                                'unavailable_reason': None if individual[column] else 'selected eigencluster/source association not individually bound'})
                dilation = np.full(6, np.nan)
                valid = bool(np.any(population) and np.all(own_p['lead200'][population] > 0))
                if valid:
                    dilation[population] = np.abs(own_p['lead200'][population]-own_p['cap50'][population])/own_p['lead200'][population]
                checks['own_far_dilation_0_25_percent'] = bool(valid and np.max(dilation[population]) <= .0025)
                arrays[prefix+'_source_lead_population'] = population
                arrays[prefix+'_own_dilation_relative'] = dilation
                near_denominator = own_p['cap50']
                near_ratio = np.divide(own_p['cap2'], near_denominator,
                                       out=np.full(6, np.nan), where=near_denominator != 0)
                arrays[prefix+'_near_over_far_population_ratio'] = near_ratio
                length, data = pair_compare(by_case[(family, k, 'lead200', False)],
                                            by_case[(family, k, 'lead400', False)], 'length')
                comparisons.append({'family': family, 'kappa': k, 'length': length})
                arrays.update({prefix+'_length_'+key: value for key, value in data.items()})
                checks['lead_length'] = length['all_passed']
                rows.append({'family': family, 'kappa': k, 'source_lead_population_count': int(np.sum(population)),
                             'maximum_own_dilation_relative': float(np.max(dilation[population])) if valid else None,
                             'checks': checks, 'all_passed': all(checks.values()), 'retained_weights': retained_rows,
                             'ordered_spectra': spectral_rows, 'raw_Hermitian_defects': defects,
                             'own_dilation_relative': dilation, 'near_over_far_ratio': near_ratio})
    for family, k, arm in (('F1', 1, 'lead200'), ('F2', 3, 'cap50')):
        temporal, data = pair_compare(by_case[(family, k, arm, False)], by_case[(family, k, arm, True)], 'time')
        comparisons.append({'family': family, 'kappa': k, 'arm': arm, 'temporal': temporal})
        arrays.update({family+'_k'+str(k)+'_'+arm+'_temporal_'+key: value for key, value in data.items()})
    all_producers = all(value['all_passed'] for value in by_case.values())
    all_comparisons = all(value.get('length', value.get('temporal'))['all_passed'] for value in comparisons)
    return {'rows': rows, 'comparisons': comparisons, 'source_bindings': bindings,
            'spectral_convention': "numpy.linalg.eigvalsh(UPLO='L'): Hermitian matrix represented by raw lower triangle; raw matrices and unmodified eigenvalues retained, no clipping or symmetrization",
            'all_producer_numerical_gates_passed': all_producers,
            'all_passed': bool(all_producers and all_comparisons and all(row['all_passed'] for row in rows)),
            'scope': 'Six out-of-thermal-window radial cases; no entropy or actual cubic-window certificate'}, arrays


def cost_estimate(benchmarks):
    import numpy as np
    from lead_numerics import make_plan
    rows = []
    for benchmark in benchmarks:
        if benchmark['stage'] != 'apply-benchmark' or not benchmark['all_passed']:
            raise ValueError('Completed exact own component benchmark required')
        samples = benchmark['timings']
        # Timing trees contain Gram/bound array references; scalar durations
        # and phase timings remain ordinary saved numeric metadata.
        sample = next(x for x in samples if x['columns'] == 16)
        full_h = float(np.median(sample['component_seconds']['full_generator']))
        native_h = float(np.median(sample['component_seconds']['native_generator']))
        diagnostics = float(np.median(sample['component_seconds']['complete_diagnostics']))
        full_step_record = float(np.median(sample['full_step_record_seconds']))
        native_step_record = float(np.median(sample['native_step_record_seconds']))
        native_rate = (sample['native_Taylor32_one_step_seconds']+native_step_record)/(16*sample['native_step_delta_a'])
        full_rate = (sample['Taylor32_one_step_seconds']+full_step_record)/(16*sample['step_delta_a'])
        ratio = full_rate/native_rate
        phases = {key: float(np.median([trial[key] for trial in sample['profiled_full_apply_phases']]))
                  for key in sample['profiled_full_apply_phases'][0]}
        dofs = benchmark['layout']['total_dofs']
        setup_binding = benchmark['bindings']['setup']
        require_identity(setup_binding['path'], setup_binding['sha256'])
        setup_outer = json.loads(Path(setup_binding['path']).read_text())
        recheck_artifacts(setup_outer)
        artifacts = setup_outer['output_bindings']['setup_artifacts']
        e_bytes = sum(item['bytes'] for name, item in artifacts.items() if name.startswith('E_c'))
        block_bytes = sum(item['bytes'] for name, item in artifacts.items() if name.startswith('blocks_') or name.startswith('port'))
        setup_disk_bytes = sum(item['bytes'] for item in artifacts.values())
        case_rows = []
        for duration in (6.931471805599452*16, 1600.):
            plan = make_plan(benchmark['operator_bound'], np.r_[0., duration*np.arange(1, 5)/4])
            nsteps = len(plan['intervals'])
            batches = 8
            baseline_seconds = sample['full_initial_Gram_energy_seconds']
            cpu_seconds = batches*(nsteps*sample['Taylor32_one_step_seconds']+
                                   (nsteps+1)*full_step_record+baseline_seconds+5*diagnostics)
            native_gpu_hours = .0648*128*duration/3600
            conditional_gpu_hours = native_gpu_hours*ratio
            checkpoint_raw_bytes = 128*dofs*16*5
            case_rows.append({'T_a': duration, 'T_over_L': duration/16, 'columns': 128,
                'batch_columns': 16, 'coherent_batches': batches, 'steps_per_batch': nsteps,
                'Taylor_H_actions': 32*nsteps*batches,
                'internal_diagnostic_H_actions': (nsteps+1)*batches,
                'saved_clock_extra_full_H_actions': 2*5*batches,
                'saved_clock_extra_native_H_actions': 5*batches,
                'CPU_compute_hours_extrapolated': cpu_seconds/3600,
                'H100_native_only_calibration_hours': native_gpu_hours,
                'conditional_H100_compute_hours_equal_relative_transfer': conditional_gpu_hours,
                'conditional_both_windows_H100_hours': 2*conditional_gpu_hours,
                'one_state_full128_bytes': 128*dofs*16,
                'frame_E_resident_file_bytes_upper': e_bytes,
                'saved_raw_and_realized_block_file_bytes_upper': block_bytes,
                'one_setup_artifact_disk_bytes': setup_disk_bytes,
                'measured_CPU_setup_seconds': setup_outer['setup_compute_seconds'],
                'measured_CPU_benchmark_peak_RSS_bytes': sample['process_peak_RSS_bytes'],
                'batch16_10_state_buffers_plus_saved_map_block_bytes_upper': 10*16*dofs*16+e_bytes+block_bytes,
                'full128_10_state_buffers_plus_saved_map_block_bytes_upper': 10*128*dofs*16+e_bytes+block_bytes,
                'five_clock_full128_state_bytes_uncompressed': checkpoint_raw_bytes,
                'five_clock_cross_batch_Gram_assembly_flop_scale': 5*128*128*dofs,
                'compression_sample_raw_bytes': sample['initial_bytes'],
                'compression_sample_encoded_bytes': sample['lossless_encoded_bytes'],
                'compression_sample_seconds': sample['write_seconds']})
        rows.append({'K': benchmark['arguments']['K'], 'end_radius': benchmark['arguments']['end_radius'],
                     'field': benchmark['arguments']['field'], 'operator_bound': benchmark['operator_bound'],
                     'native_CPU_seconds_per_column_a': native_rate,
                     'full_CPU_seconds_per_column_a': full_rate,
                     'full_step_diagnostic_median_seconds': full_step_record,
                     'native_step_diagnostic_median_seconds': native_step_record,
                     'full_initial_Gram_energy_seconds': sample['full_initial_Gram_energy_seconds'],
                     'native_initial_Gram_energy_seconds': sample['native_initial_Gram_energy_seconds'],
                     'full_H_only_median_seconds': full_h, 'native_H_only_median_seconds': native_h,
                     'measured_CPU_rate_ratio': ratio, 'measured_phase_medians': phases,
                     'durations': case_rows})
    return {'benchmarks': rows, 'all_passed': True,
            'measured_H100_job': False,
            'formula': 'H100_compute =0.0648*128*T_a/3600*(own_CPU_full_rate/own_CPU_native_rate)*unknown_relative_GPU_transfer_factor',
            'reported_numeric_scenario_transfer_factor': 1.,
            'CPU_initial_record_pricing': 'Conservative: baseline H and Grams plus one complete initial _step_record including H; actual propagation reuses the baseline H for its initial record, so one H per batch is overcounted.',
            'unmeasured_additions': ['GPU setup and frame transfers', 'GPU cross-batch assembly',
                                     'saved-clock current, bin and other checkpoint diagnostics on GPU',
                                     'checkpoint compression and I/O at populated later states',
                                     'backend/algorithm difference from the source native-sink kernel',
                                     'geometry/operator-work mismatch: own CPU denominator is the ball-compressed native operator, source H100 calibration is the full cube with a sink'],
            'memory_scope': 'Array/file-based conservative planning only; actual GPU workspace, copies and allocator peak unmeasured. CPU implementation has no authorized GPU run.',
            'required_H100_measurement': 'One admitted batch16 complete native/collar/port/lead Taylor+diagnostic step, peak memory and checkpoint write; no card launched',
            'scope': 'Fixed durations and128 columns,256 cost also shown. No arbitrary later tail duration; no measured H100 throughput of this build.'}, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('time', 'length', 'K', 'native-reference', 'flat', 'b11', 'cost', 'controls'))
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--run-sha256', action='append', required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len(args.run_sha256):
        raise ValueError('Every saved report needs one exact SHA256')
    runtime = admit(args.outdir, __file__, 'Saved-only channel-lead '+args.mode, 1, 0, mode='dense', blas_threads=1)
    if any(args.outdir.iterdir()):
        raise ValueError('Fresh empty reader directory required')
    import numpy as np
    from lead_run import write_arrays, require_storage
    verify_threads(runtime, 'after_imports')
    result = {'mode': args.mode, 'runtime': runtime, 'bindings': [], 'code_bindings': code_bindings(),
              'saved_only': True, 'new_physics_evaluated': False, 'all_passed': False}
    try:
        values = []
        for path, sha in zip(args.run, args.run_sha256):
            value, binding = report(path, sha)
            values.append(value)
            result['bindings'].append(binding)
        # Full auxiliary reference states can be large; this guard includes
        # both inputs, difference, generated blocks and full coherent Grams.
        check_memory(runtime, 'saved_state_comparisons', 64*1024**3)
        if args.mode in ('time', 'length', 'K'):
            if len(values) != 2:
                raise ValueError('Pair comparison needs exactly two saved runs')
            details, arrays = pair_compare(*values, args.mode)
        elif args.mode == 'native-reference':
            if len(values) != 2:
                raise ValueError('Two saved response runs required')
            details, arrays = native_reference_compare(*values)
        elif args.mode == 'flat':
            details, arrays = flat_compare(values[0], values[1:])
        elif args.mode == 'b11':
            details, arrays = b11_compare(values)
        elif args.mode == 'controls':
            if len(values) != 1:
                raise ValueError('One saved manufactured record required')
            details, arrays = controls_evidence(values[0])
        else:
            details, arrays = cost_estimate(values)
        result.update(details=details, all_passed=details['all_passed'])
        raw_bytes = sum(np.asarray(array).nbytes for array in arrays.values())
        require_storage(args.outdir, (101*raw_bytes+99)//100+4096*len(arrays)+64*1024**2,
                        runtime, 'saved_reader_arrays_and_tables')
        result['output_bindings'] = {'arrays': write_arrays(args.outdir/'comparison_arrays.npz', arrays)}
        for value in values:
            recheck_artifacts(value)
        recheck_artifacts(result)
        verify_threads(runtime, 'before_save')
        save_record(args.outdir/'comparison.json', result)
        lines = ['# Channel-lead saved comparison', '', 'Mode: '+args.mode,
                 '', ('Planning calculation complete: ' if args.mode == 'cost' else 'Fixed numerical gates: ')+('PASS' if result['all_passed'] else 'FAIL OR WITHHELD'),
                 '', 'All raw arrays and scientific records remain on the permitted box.', '']
        if 'rows' in details and args.mode not in ('b11', 'controls'):
            lines += ['| Case | Clock / a | State error | Passed |', '|---|---:|---:|---|']
            for row in details['rows']:
                label = row.get('block', str(row.get('family', 'pair'))+' '+str(row.get('kappa', '')))
                error = row.get('state_error', row.get('metric_normalized_state_error'))
                lines.append('| '+str(label)+' | '+str(row.get('clock', ''))+' | '+str(error)+' | '+str(row.get('passed', row.get('all_passed')))+' |')
        if args.mode == 'controls':
            lines += ['', '| Case | Check | Passed |', '|---|---|---|']
            for case in details['rows']:
                for key, passed in case['checks'].items():
                    lines.append('| %s | %s | %s |' % (case['case'], key, passed))
            lines += ['', '| Case | Saved evidence | Value or array summary |', '|---|---|---|']
            for case in details['rows']:
                for leaf in case['expected']+case['observed']:
                    rendered = json.dumps(clean(leaf.get('summary', leaf.get('value'))), sort_keys=True)
                    lines.append('| %s | %s | %s |' % (case['case'], leaf['leaf'], rendered.replace('|', '\\|')))
                for pair in case['matched_array_differences']:
                    lines.append('| %s | observed minus expected: %s | %s |' % (
                        case['case'], pair['observed'], json.dumps(clean(pair['residual']), sort_keys=True)))
            lines += ['', '| Saved array | Summary |', '|---|---|']
            for leaf in details['array_inventory']:
                lines.append('| %s | %s |' % (leaf['array_key'], json.dumps(clean(leaf['summary']), sort_keys=True)))
        if args.mode == 'flat':
            lines += ['', 'Core/shell errors below are diagnostics with no additional numerical threshold.', '',
                      '| Block | T / a | Region | Raw state error | Metric state error | Raw inventory Gram error | Metric inventory Gram error |',
                      '|---|---:|---|---:|---:|---:|---:|']
            for row in details['rows']:
                for region, diagnostic in row['regional_diagnostics'].items():
                    lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (row['block'], row['clock'], region,
                        diagnostic['raw_state_error'], diagnostic.get('metric_normalized_state_error'),
                        diagnostic['raw_inventory_Gram_error'], diagnostic.get('metric_normalized_inventory_Gram_error')))
            lines += ['', '| Block | T / a | Raw input norm | Raw image bound | Metric input norm | Metric image bound | Raw bound passed | Metric bound passed |',
                      '|---|---:|---:|---:|---:|---:|---|---|']
            for row in details['rows']:
                raw, normalized = row['raw_image_certificate'], row['normalized_image_certificate']
                lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (row['block'], row['clock'],
                    raw['input_block_norm'], raw['decimal_upper'],
                    None if normalized is None else normalized['input_block_norm'],
                    None if normalized is None else normalized['decimal_upper'],
                    raw['passed'],
                    None if normalized is None else normalized['passed']))
            lines += ['', details['current_scope'], '',
                      '| T / a | Current | Original column | Raw signed diagonal | Metric signed diagonal |',
                      '|---:|---|---:|---|---|']
            for row in details['signed_current_rows']:
                lines.append('| %s | %s | %s | %s | %s |' % (row['clock'], row['quantity'], row['column'],
                    json.dumps(clean(row['raw_diagonal'])), json.dumps(clean(row['metric_normalized_diagonal']))))
        if args.mode == 'b11':
            lines += ['', details['spectral_convention'], '',
                      '| Family | kappa | Ordered | Arm | Population | Source p | Own p | Signed error | Relative error / availability |',
                      '|---|---:|---:|---|---|---:|---:|---:|---|']
            for case in details['rows']:
                for row in case['ordered_spectra']:
                    lines.append('| %s | %s | %s | %s | %s | %.17g | %.17g | %.17g | %s / %s |' % (
                        case['family'], case['kappa'], row['ordered_index'], row['arm'], row['fixed_population'],
                        row['source_p'], row['own_p'], row['signed_difference'], row['relative_error'], row['relative_available']))
            lines += ['', '| Family | kappa | Ordered | Dilation relative | Near / far |', '|---|---:|---:|---:|---:|']
            for case in details['rows']:
                for column in range(6):
                    lines.append('| %s | %s | %s | %s | %s |' % (case['family'], case['kappa'], column,
                        case['own_dilation_relative'][column], case['near_over_far_ratio'][column]))
                lines += ['', 'Raw Hermitian defects '+case['family']+' k'+str(case['kappa'])+': '+json.dumps(case['raw_Hermitian_defects'], sort_keys=True),
                          'Required checks: '+json.dumps(case['checks'], sort_keys=True), '']
            lines += ['', '| Family | kappa | Arm | T / a | Own column | Source column | Own retained | Source retained | Signed difference | Own / source | Individual available |',
                      '|---|---:|---|---:|---:|---|---:|---|---|---|---|']
            for case in details['rows']:
                for row in case['retained_weights']:
                    lines.append('| %s | %s | %s | %s | %s | %s | %.17g | %s | %s | %s | %s |' % (
                        case['family'], case['kappa'], row['arm'], row['clock'], row['own_column'], row['source_column'],
                        row['own_weight'], row['source_weight'], row['signed_difference'], row['own_over_source'], row['individual_available']))
        if args.mode == 'cost':
            lines += ['', '| K | End / a | T / L | Conditional H100 h,128 | CPU h |',
                      '|---:|---:|---:|---:|---:|']
            for bench in details['benchmarks']:
                for row in bench['durations']:
                    lines.append('| %s | %s | %.8g | %.8g | %.8g |' % (
                        bench['K'], bench['end_radius'], row['T_over_L'],
                        row['conditional_H100_compute_hours_equal_relative_transfer'], row['CPU_compute_hours_extrapolated']))
            lines += ['', 'H100 values are a conditional planning scenario for steady stepping: the measured complete CPU step-cost ratio is assumed to transfer unchanged to the source H100 calibration. The CPU denominator compresses the native ball; the calibration uses the full cube sink. Geometry, GPU backend, setup, saved-clock current/bin diagnostics, cross-batch assembly and later-state I/O transfer factors remain unmeasured. No GPU run was made.',
                      '', details['CPU_initial_record_pricing']]
        with (args.outdir/'TABLE.md').open('x') as stream:
            stream.write('\n'.join(lines)+'\n')
        return 0 if result['all_passed'] else 1
    except Exception:
        result['failure'] = traceback.format_exc()
        try:
            for value in locals().get('values', []):
                recheck_artifacts(value)
            recheck_artifacts(result)
        except Exception:
            result['failure_identity_recheck'] = traceback.format_exc()
        save_record(args.outdir/'comparison_failure.json', result)
        raise


if __name__ == '__main__':
    sys.exit(main())
