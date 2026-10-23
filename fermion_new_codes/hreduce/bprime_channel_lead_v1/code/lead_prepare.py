"""Guarded native-input, radial-field and B11 input-only preparation stages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
from lead_runtime import (admit, code_bindings, core_bindings, require_identity,
                          identity, verify_threads, check_memory, SPEC_SHA256,
                          DERIVATION_SHA256)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, complex):
        return {'real': clean(value.real), 'imag': clean(value.imag)}
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return {'nonfinite_literal': repr(value)}
    return value


def save_record(path, record):
    path = Path(path)
    if path.exists():
        raise ValueError('Fresh immutable record required: '+str(path))
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('x') as stream:
        json.dump(clean(record), stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)
    return identity(path)


def admitted_record(path, sha):
    binding = require_identity(path, sha)
    record = json.loads(Path(path).read_text())
    if not record['all_passed']:
        raise ValueError('Prerequisite saved stage did not pass')
    if 'bindings' in record:
        if record['bindings']['spec']['sha256'] != SPEC_SHA256 or record['bindings']['code'] != code_bindings():
            raise ValueError('Prerequisite stage has a different specification/source version')
    return record, binding


def recheck_artifacts(value):
    """Rehash every recursive bound artifact without reading its arrays."""
    seen = {}
    def walk(item):
        if isinstance(item, dict):
            if 'path' in item and 'sha256' in item:
                key = str(Path(item['path']).resolve())
                if key in seen and seen[key] != item['sha256']:
                    raise ValueError('Conflicting recursive artifact identities')
                if key not in seen:
                    require_identity(item['path'], item['sha256'])
                    seen[key] = item['sha256']
            else:
                for child in item.values():
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)


def artifact_bindings(value):
    found = {}
    def walk(item):
        if isinstance(item, dict):
            if 'path' in item and 'sha256' in item:
                found[(item['path'], item['sha256'])] = item
            else:
                for child in item.values():
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return list(found.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('native', 'fields', 'b11'))
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--derivation', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--controls-sha256', required=True)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--native-sha256')
    parser.add_argument('--fields', type=Path)
    parser.add_argument('--fields-sha256')
    args = parser.parse_args()
    mixed = args.stage == 'native'
    runtime = admit(args.outdir, __file__, 'Channel-lead '+args.stage+' input preparation',
                    16 if mixed else 1, 15 if mixed else 0,
                    mode='mixed' if mixed else 'dense', blas_threads=1)
    if any(args.outdir.iterdir()):
        raise ValueError('A fresh empty stage directory is required')
    destination = args.outdir/(args.stage+'.json')
    arrays_path = args.outdir/(args.stage+'.npz')
    if (destination.exists() or arrays_path.exists() or
            (args.outdir/(args.stage+'_failure.json')).exists() or
            destination.with_suffix('.json.partial').exists()):
        raise ValueError('Refuse to replay into existing stage outputs')
    bindings = {'self': identity(__file__), 'spec': require_identity(args.spec, SPEC_SHA256),
                'derivation': require_identity(args.derivation, DERIVATION_SHA256),
                'code': code_bindings(), 'core': core_bindings()}
    controls, bindings['controls'] = admitted_record(args.controls, args.controls_sha256)
    if controls['code_bindings'] != bindings['code']:
        raise ValueError('Manufactured controls are not for the complete exact source set')
    if controls['spec_binding']['sha256'] != SPEC_SHA256:
        raise ValueError('Wrong manufactured specification')
    checked_outputs = []
    def recheck_outputs(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                require_identity(value['path'], value['sha256'])
                checked_outputs.append(value['sha256'])
            else:
                for child in value.values():
                    recheck_outputs(child)
        elif isinstance(value, list):
            for child in value:
                recheck_outputs(child)
    recheck_outputs(controls['output_bindings'])
    if not checked_outputs:
        raise ValueError('Manufactured control arrays must be explicitly bound')
    bindings['control_artifacts'] = artifact_bindings(controls)
    import numpy as np
    from lead_inputs import (PREP_PATHS, PREP_SHA256, load_native_inputs,
                             b11_declared_inputs, select_b11_inputs)
    verify_threads(runtime, 'after_imports')
    result = {'stage': args.stage, 'runtime': runtime, 'bindings': bindings,
              'source_consequences_opened': False, 'all_passed': False}
    arrays = {}
    try:
        if args.stage == 'native':
            from lead_native import NativeHamiltonian
            check_memory(runtime, 'native_inputs', 4*1024**3)
            pinned = {role: {'path': str(path), 'sha256': PREP_SHA256[role]}
                      for role, path in PREP_PATHS.items()}
            details, arrays = load_native_inputs(pinned, NativeHamiltonian)
            result['native'] = details
            result['all_passed'] = details['all_passed']
        elif args.stage == 'fields':
            if args.native is None or args.native_sha256 is None:
                raise ValueError('Passed native stage required before field preparation')
            native, bindings['native'] = admitted_record(args.native, args.native_sha256)
            bindings['native_artifacts'] = artifact_bindings(native)
            native_arrays = require_identity(native['output_bindings']['arrays']['path'],
                                             native['output_bindings']['arrays']['sha256'])
            with np.load(native_arrays['path'], allow_pickle=False) as archive:
                given_n, given_v = archive['native_N'], archive['native_flow']
                variable_probes = archive['variable_probe_native']
            from lead_fields import (reconstruct_source, RadialProfile, back_average,
                                     dense_artifact, SOURCE_PATH, SOURCE_MD5,
                                     SOURCE_JSON_MD5, averaged_exterior_majorants)
            source_binding = identity(SOURCE_PATH)
            json_binding = identity(SOURCE_PATH.replace('_fields.npz', '.json'))
            if source_binding['md5'] != SOURCE_MD5 or json_binding['md5'] != SOURCE_JSON_MD5:
                raise ValueError('Wrong raw original-collapse source identities')
            bindings['source_npz'], bindings['source_json'] = source_binding, json_binding
            check_memory(runtime, 'field_reconstruction', 4*1024**3)
            radial_sites = np.arange(1., 6403.)  # 400L plus both endpoint crossings
            coords = np.arange(-40, 41)
            x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
            maxcoord = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
            seam_mask = (maxcoord == 39)|(maxcoord == 40)
            seam_xyz = np.array([x[seam_mask], y[seam_mask], z[seam_mask]]).T
            seam_square, seam_inverse = np.unique(np.sum(seam_xyz**2, axis=1), return_inverse=True)
            seam_radius = np.sqrt(seam_square)
            arrays['radial_sites'] = radial_sites
            arrays['seam_xyz'] = seam_xyz
            checks, details, profiles = [], {}, {}
            def check(name, measured, threshold):
                checks.append({'name': name, 'measured': float(measured), 'threshold': threshold,
                               'passed': bool(measured <= threshold)})
            for variant, tight in (('P', False), ('H', True)):
                profile, reconstruction = reconstruct_source(variant=variant)
                profiles[variant] = profile
                radial, quadrature = back_average(profile, radial_sites, tight=tight)
                seam_fields, seam_quadrature = back_average(profile, seam_radius, tight=tight)
                seam_values = seam_fields[seam_inverse]
                seam_n = seam_values[:, 0]
                r_seam = np.sqrt(np.sum(seam_xyz**2, axis=1))
                seam_v = seam_values[:, 1, None]*seam_xyz/r_seam[:, None]
                n_difference = seam_n-given_n[seam_mask]
                v_difference = seam_v-given_v[:, seam_mask].T
                axis_n = given_n[41:81, 40, 40]
                axis_v = given_v[0, 41:81, 40, 40]
                axis_difference = radial[:40]-np.array([axis_n, axis_v]).T
                arrays['F1_'+variant+'_radial_Nv'] = radial
                arrays['F1_'+variant+'_axis_difference'] = axis_difference
                arrays['F1_'+variant+'_seam_N_difference'] = n_difference
                arrays['F1_'+variant+'_seam_v_difference'] = v_difference
                arrays['F1_'+variant+'_seam_Nv'] = seam_fields
                arrays.update({'F1_'+variant+'_'+key: value for key, value in dense_artifact(profile.field).items()})
                check(variant+'_charge_relative', abs(reconstruction['charge_relative_difference']), 1e-8)
                check(variant+'_chain_axis', np.max(np.abs(axis_difference)), 1e-8)
                check(variant+'_seam_components', max(np.max(np.abs(n_difference)), np.max(np.abs(v_difference))), 1e-8)
                details[variant] = {'reconstruction': reconstruction, 'radial_quadrature': quadrature,
                                    'seam_quadrature': seam_quadrature,
                                    'exterior39': averaged_exterior_majorants(profile, 39.),
                                    'exterior799': averaged_exterior_majorants(profile, 799.)}
            check('F1_radial_P_H_all_used_sites', np.max(np.abs(arrays['F1_P_radial_Nv']-arrays['F1_H_radial_Nv'])), 1e-10)
            check('F1_seam_P_H_all_used_radii', np.max(np.abs(arrays['F1_P_seam_Nv']-arrays['F1_H_seam_Nv'])), 1e-10)
            p24_radial, p24_radial_report = back_average(profiles['P'], radial_sites, tight=True)
            p24_seam, p24_seam_report = back_average(profiles['P'], seam_radius, tight=True)
            arrays['F1_P24_radial_Nv'], arrays['F1_P24_seam_Nv'] = p24_radial, p24_seam
            details['P24'] = {'radial_quadrature': p24_radial_report, 'seam_quadrature': p24_seam_report}
            for domain in ('radial', 'seam'):
                base = arrays['F1_P_'+domain+'_Nv']
                cross = arrays['F1_P24_'+domain+'_Nv']
                tight = arrays['F1_H_'+domain+'_Nv']
                arrays['F1_'+domain+'_P16_minus_P24'] = base-cross
                arrays['F1_'+domain+'_P24_minus_H24'] = cross-tight
                check('F1_'+domain+'_same_field_quadrature', np.max(np.abs(base-cross)), 1e-10)
                check('F1_'+domain+'_same_rule_ODE', np.max(np.abs(cross-tight)), 1e-10)
            from lead_references import endpoint_bounds
            from lead_inputs import opnorm
            probe_norm = opnorm(variable_probes)
            n_bound = max(float(np.max(given_n[seam_mask])),
                          *(details[v]['exterior39']['N_upper'] for v in ('P', 'H')))
            v_bound = max(float(np.max(np.abs(given_v[:, seam_mask]))),
                          *(details[v]['exterior39']['v_abs_upper'] for v in ('P', 'H')))
            reference_candidates, reference_halfside = [], None
            for halfside in (96, 112, 128):
                candidate = endpoint_bounds(halfside, 32., n_bound, v_bound, probe_norm, native=True)
                reference_candidates.append(candidate)
                if candidate['passed']:
                    reference_halfside = halfside
                    break
            details['expanded_reference_candidates'] = reference_candidates
            details['expanded_reference_halfside'] = reference_halfside
            check('expanded_reference_size_admitted', 0. if reference_halfside is not None else 1., 0.)
            if reference_halfside is not None:
                # All distinct radii actually used outside the original81³
                # cube, with exact integer squared-radius associations.
                h = reference_halfside
                c = np.arange(h+1, dtype=np.int64)
                yz_square = c[:, None]**2+c[None, :]**2
                yz_outside = (c[:, None] > 40)|(c[None, :] > 40)
                seen = np.zeros(3*h*h+1, dtype=bool)
                for a in c:
                    squares = yz_square+a*a
                    seen[squares.ravel() if a > 40 else squares[yz_outside]] = True
                reference_squares = np.flatnonzero(seen)
                reference_radii = np.sqrt(reference_squares)
                arrays['reference_radius_squares'] = reference_squares
                for variant, tight in (('P', False), ('H', True)):
                    reference_fields, quadrature = back_average(profiles[variant], reference_radii, tight=tight)
                    arrays['F1_'+variant+'_reference_Nv'] = reference_fields
                    details[variant]['reference_quadrature'] = quadrature
                check('F1_reference_P_H_all_used_radii',
                      np.max(np.abs(arrays['F1_P_reference_Nv']-arrays['F1_H_reference_Nv'])), 1e-10)
                reference_p24, report_p24 = back_average(profiles['P'], reference_radii, tight=True)
                arrays['F1_P24_reference_Nv'] = reference_p24
                details['P24']['reference_quadrature'] = report_p24
                base, tight = arrays['F1_P_reference_Nv'], arrays['F1_H_reference_Nv']
                arrays['F1_reference_P16_minus_P24'] = base-reference_p24
                arrays['F1_reference_P24_minus_H24'] = reference_p24-tight
                check('F1_reference_same_field_quadrature', np.max(np.abs(base-reference_p24)), 1e-10)
                check('F1_reference_same_rule_ODE', np.max(np.abs(reference_p24-tight)), 1e-10)
                actual_n = max(float(np.max(given_n[seam_mask])),
                               *(float(np.max(arrays['F1_'+variant+'_reference_Nv'][:, 0]))
                                 for variant in ('P', 'H', 'P24')))
                actual_v = max(float(np.max(np.abs(given_v[:, seam_mask]))),
                               *(float(np.max(np.abs(arrays['F1_'+variant+'_reference_Nv'][:, 1])))
                                 for variant in ('P', 'H', 'P24')))
                details['expanded_reference_actual_coefficient_maxima'] = {'N': actual_n, 'v_abs': actual_v}
                check('expanded_reference_N_bound_covers_used_coefficients', actual_n-n_bound, 0.)
                check('expanded_reference_v_bound_covers_used_coefficients', actual_v-v_bound, 0.)
            for variant in ('P', 'H'):
                far = details[variant]['exterior799']
                for accuracy in ('P', 'H', 'P24'):
                    actual = arrays['F1_'+accuracy+'_radial_Nv'][798:]
                    check(variant+'_'+accuracy+'_far_N_bound_covers_used_coefficients',
                          float(np.max(actual[:, 0]))-far['N_upper'], 0.)
                    check(variant+'_'+accuracy+'_far_v_bound_covers_used_coefficients',
                          float(np.max(np.abs(actual[:, 1])))-far['v_abs_upper'], 0.)
            for variant, tight in (('P', False), ('H', True)):
                f2, report = back_average(RadialProfile('F2'), radial_sites, tight=tight)
                arrays['F2_'+variant+'_radial_Nv'] = f2
                details['F2_'+variant] = report
            details['F2_exterior799'] = averaged_exterior_majorants(RadialProfile('F2'), 799.)
            for accuracy in ('P', 'H'):
                actual = arrays['F2_'+accuracy+'_radial_Nv'][798:]
                upper = details['F2_exterior799']
                check('F2_'+accuracy+'_far_N_bound_covers_used_coefficients',
                      float(np.max(actual[:, 0]))-upper['N_upper'], 0.)
                check('F2_'+accuracy+'_far_v_bound_covers_used_coefficients',
                      float(np.max(np.abs(actual[:, 1])))-upper['v_abs_upper'], 0.)
            check('F2_radial_P_H_all_used_sites', np.max(np.abs(arrays['F2_P_radial_Nv']-arrays['F2_H_radial_Nv'])), 1e-10)
            result.update(field_details=details, checks=checks, all_passed=all(row['passed'] for row in checks))
        else:
            if args.fields is None or args.fields_sha256 is None:
                raise ValueError('Passed radial field stage required before B11 levels')
            fields, bindings['fields'] = admitted_record(args.fields, args.fields_sha256)
            bindings['field_artifacts'] = artifact_bindings(fields)
            field_array = require_identity(fields['output_bindings']['arrays']['path'],
                                            fields['output_bindings']['arrays']['sha256'])
            from lead_channels import RadialHamiltonian
            results = {}
            with np.load(field_array['path'], allow_pickle=False) as archive:
                for family in ('F1', 'F2'):
                    declaration = b11_declared_inputs(family)
                    radial = archive[family+'_P_radial_Nv']
                    cases = {}
                    for kappa in (1, 2, 3):
                        case, data = select_b11_inputs(family, kappa, radial[:, 0], radial[:, 1],
                                                      RadialHamiltonian, declaration['input_cases'][kappa]['energies'])
                        cases[kappa] = case
                        arrays.update({family+'_k'+str(kappa)+'_'+key: value for key, value in data.items()})
                    results[family] = {'input_declaration': declaration, 'cases': cases}
            result['families'] = results
            result['all_passed'] = all(case['all_passed'] for family in results.values() for case in family['cases'].values())
        verify_threads(runtime, 'after_stage')
        recheck_artifacts(result)
        if arrays:
            check_memory(runtime, 'before_save', 64*1024**2)
            with arrays_path.open('xb') as stream:
                np.savez(stream, **arrays)
            result['output_bindings'] = {'arrays': identity(arrays_path)}
        else:
            result['output_bindings'] = {}
        recheck_artifacts(result)
        save_record(destination, result)
        return 0 if result['all_passed'] else 1
    except Exception:
        result['failure'] = traceback.format_exc()
        try:
            recheck_artifacts(result)
        except Exception:
            result['final_identity_recheck_failure'] = traceback.format_exc()
        # Preserve completed arrays and failure evidence, never overwrite it.
        if arrays and not arrays_path.exists():
            with arrays_path.open('xb') as stream:
                np.savez(stream, **arrays)
            result['output_bindings'] = {'arrays': identity(arrays_path)}
        save_record(args.outdir/(args.stage+'_failure.json'), result)
        raise


if __name__ == '__main__':
    sys.exit(main())
