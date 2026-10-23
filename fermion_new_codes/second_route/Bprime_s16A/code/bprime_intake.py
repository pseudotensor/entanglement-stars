"""Stdlib identities, JSON metadata/schema and NPY headers for inbox task 17.

No scientific numeric array or object payload is loaded. JSON numbers remain
lexical tokens. Version two additionally projects the admitted prep scalars and
timing lists, and reads only scalar Hs.npy as configuration if JSON lacks Hs.
Run result values remain excluded. All files and own outputs stay on the box.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import struct
import zipfile

from bprime_runtime import admit, code_bindings, identity, verify_threads


SOURCE_ROOT = Path('<internal>/bprime')
PREP_ROOT = SOURCE_ROOT / 'prep_v5'
KIT_ROOT = SOURCE_ROOT / 'kit_s16A_v2/prep'
RUN_ROOT = SOURCE_ROOT / 's16A_h100x8_v1'
INPUTS = {}
for label, stem in (('full', 'F1_s16_d0'), ('supply', 'F1_s16_d0_sup'), ('exit', 'F1_s16_d0_exit')):
    for suffix in ('json', 'npz'):
        INPUTS['prep_v5_' + label + '_' + suffix] = PREP_ROOT / (stem + '_prep.' + suffix)
for suffix in ('json', 'npz'):
    INPUTS['kit_full_' + suffix] = KIT_ROOT / ('F1_s16_d0_prep.' + suffix)
for label in ('sup', 'exit'):
    for suffix in ('json', 'npz'):
        INPUTS['run_1_' + label + '_' + suffix] = RUN_ROOT / ('s16A_1_F1_d0_dt0.5_T69.32_' + label + '.' + suffix)

HEADER_LIMIT = 1024 * 1024
JSON_LIMIT = 32 * 1024 * 1024
ARCHIVE_MEMBER_LIMIT = 4096
SCHEMA_PATH_LIMIT = 20000
METADATA_LEAF_LIMIT = 4096
METADATA_STRING_LIMIT = 4096
TIMING_LIST_LIMIT = 64
DECLARED_FULL_PREP_MD5 = 'b97c1f871a9a090efb898be3d7b57fe9'
REPORT_REFERENCE = {
    'path': '<internal>/PAGE_Bprime_s16A_v1.md',
    'sha256_of_read_document': '40b8c45f4bee0ad298e8210033d771eccc320aefb0fb40375f5633b9192832c0',
    'scope': 'Document read before implementation; no scientific output values are selected from it.'}
CONFIGURATION_ROOTS = frozenset((
    'args', 'cfg', 'config', 'configuration', 'settings', 'parameters', 'params',
    'options', 'setup', 'inputs', 'input', 'source', 'sources', 'provenance',
    'conventions', 'operator_convention', 'source_operator', 'lattice', 'grid',
    'embedding', 'field_embedding', 'window_spec', 'sink', 'sink_config',
    'prep_config', 'basis_metadata', 'field_metadata'))
TOP_STRING_KEYS = frozenset((
    'schema', 'schema_version', 'script', 'tag', 'name', 'case', 'kind', 'label',
    'window', 'window_name', 'port', 'port_tag', 'hypothesis', 'hypothesis_tag',
    'representation', 'convention', 'ordering', 'site_ordering', 'component_order',
    'axis_order', 'basis_file', 'basis_path', 'field_file', 'field_path',
    'prep_file', 'prep_path', 'parent', 'backend', 'device', 'dtype'))
TOP_CONFIGURATION_SCALARS = frozenset((
    's', 'L', 'a', 'd', 'nx', 'ny', 'nz', 'nsites', 'n_sites', 'ncols',
    'n_columns', 'ncomp', 'components', 'dt', 'dt_a', 'T_end', 'Tmax', 't_end',
    'eta_max', 'onset', 'ramp', 'sink_onset', 'sink_width', 'sigma', 'sigma_sites'))
EXCLUDED_SUBTREES = frozenset((
    'records', 'rows', 'samples', 'results', 'readout', 'diagnostics', 'stats',
    'metrics', 'spectrum', 'spectra', 'eigenvalues', 'eigvals', 'eigenvectors',
    'basis', 'vectors', 'columns', 'orbitals', 'states', 'N', 'v', 'N_dense',
    'v_dense', 'R', 'r', 'x', 'y', 'z', 'p', 'G', 'S', 'K', 'energy', 'energies',
    'E', 'E0', 'E_inv', 'flux', 'residuals', 'gram', 'grams', 'eta_profile', 'field_values'))
PREP_CONFIGURATION_KEYS = ('H', 'Hs', 'T0', 'R_I', 'L', 'box_L', 'onset_L',
                           'sigma_sites', 'family', 'profile')
DECLARATION_STRING_KEYS = frozenset((
    'source', 'source_file', 'source_path', 'source_record', 'field_source',
    'basis_source', 'declaration', 'declarations', 'convention', 'conventions',
    'operator_convention', 'ordering', 'site_order', 'site_ordering', 'axis_order',
    'spin_order', 'spin_axis_order', 'spin_basis', 'component_order', 'alpha',
    'beta', 'boundary', 'boundary_condition', 'subcube_embedding', 'embedding',
    'dictionary', 'site_fields', 'kernel', 'ramp', 'label', 'module', 'prep',
    'family', 'profile', 'port_tag', 'hypothesis_tag', 'npz_md5', 'of',
    'window', 'source_npz', 'source_json'))
RESULT_STRING_SUBTREES = EXCLUDED_SUBTREES | frozenset((
    'poles', 'supply', 'exit', 'joint_spectra', 'cin', 'reader', 'timing'))
CONVENTION_KEY_PATTERN = re.compile(r'alpha|beta|spin|order|subcube|embedding|boundary', re.I)


class JsonInteger(str):
    pass


class JsonFloat(str):
    pass


class JsonNonfinite(str):
    pass


def kind(value):
    if isinstance(value, JsonInteger):
        return 'int'
    if isinstance(value, JsonFloat):
        return 'float'
    if isinstance(value, JsonNonfinite):
        return 'nonfinite_json_constant'
    return type(value).__name__


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def schema_paths(value, path='.', result=None):
    """Only types, container lengths and occurrence counts; no leaf values."""
    if result is None:
        result = {}
    if path not in result and len(result) >= SCHEMA_PATH_LIMIT:
        raise ValueError('JSON schema exceeds its bounded path count')
    item = result.setdefault(path, {'types': set(), 'occurrences': 0, 'container_lengths': set()})
    item['types'].add(kind(value))
    item['occurrences'] += 1
    if isinstance(value, (dict, list)):
        item['container_lengths'].add(len(value))
    if isinstance(value, dict):
        for key, child in value.items():
            schema_paths(child, path + '.' + key, result)
    elif isinstance(value, list):
        for child in value:
            schema_paths(child, path + '[]', result)
    return result


def configuration_projection(value, budget, path):
    """Bounded scalar metadata; array/list contents and result trees stay out."""
    if isinstance(value, dict):
        return {key: ({'omitted': 'scientific_result_or_array_subtree', 'type': kind(child)}
                      if key in EXCLUDED_SUBTREES else configuration_projection(child, budget, path + '.' + key))
                for key, child in value.items()}
    if isinstance(value, list):
        return {'omitted': 'list_payload_not_projected', 'type': 'list', 'length': len(value)}
    budget[0] += 1
    if budget[0] > METADATA_LEAF_LIMIT:
        raise ValueError('Configuration metadata exceeds its scalar bound: ' + path)
    if isinstance(value, (JsonInteger, JsonFloat, JsonNonfinite)):
        return {'type': kind(value), 'literal': str(value), 'converted_to_number': False}
    if isinstance(value, str) and len(value) > METADATA_STRING_LIMIT:
        return {'omitted': 'string_exceeds_metadata_bound', 'type': 'str', 'characters': len(value)}
    return {'type': kind(value), 'value': value}


def path_value(raw, keys):
    value = raw
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def admitted_prep_metadata(raw):
    result = {}
    for parent in ((), ('args',), ('prep_manifest',), ('prep_manifest', 'args')):
        for key in PREP_CONFIGURATION_KEYS:
            keys = parent + (key,)
            present, value = path_value(raw, keys)
            if present:
                result['.'.join(keys)] = (configuration_projection(value, [0], '.'.join(keys))
                                          if not isinstance(value, (dict, list))
                                          else {'status': 'unresolved', 'reason': 'admitted_prep_value_is_not_scalar', 'type': kind(value)})
    return result


def admitted_timing_metadata(raw):
    result = {}
    for keys in (('clocks_over_L',), ('args', 'extra_T_over_L'), ('prep_manifest', 'args', 'extra_T_over_L')):
        present, value = path_value(raw, keys)
        if not present:
            continue
        if not isinstance(value, list) or len(value) > TIMING_LIST_LIMIT or any(not isinstance(x, (JsonInteger, JsonFloat)) for x in value):
            raise ValueError('Malformed or oversized admitted timing metadata: ' + '.'.join(keys))
        result['.'.join(keys)] = {'status': 'present', 'length': len(value),
                                  'values': [configuration_projection(x, [0], '.'.join(keys)) for x in value],
                                  'scope': 'Exact lexical timing metadata only; no clock arithmetic or p/Gram result values.'}
    return result


def declaration_strings(raw):
    result, budget = {}, [0]
    def walk(value, path=()):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if key in RESULT_STRING_SUBTREES:
                continue
            keys = path + (key,)
            if key in DECLARATION_STRING_KEYS and type(child) is str:
                result['.'.join(keys)] = configuration_projection(child, budget, '.'.join(keys))
            elif isinstance(child, dict):
                walk(child, keys)
    walk(raw)
    return result


def json_metadata(path):
    path = Path(path)
    if path.stat().st_size > JSON_LIMIT:
        raise ValueError('JSON exceeds its metadata-intake byte bound: ' + str(path))
    raw = json.loads(path.read_text(), parse_int=JsonInteger, parse_float=JsonFloat,
                     parse_constant=JsonNonfinite, object_pairs_hook=unique_json_object)
    if not isinstance(raw, dict):
        raise ValueError('Contracted JSON must have an object at the top level')
    schema = schema_paths(raw)
    budget = [0]
    config, strings, scalars = {}, {}, {}
    for key, value in raw.items():
        if key in CONFIGURATION_ROOTS:
            config[key] = configuration_projection(value, budget, key)
        elif key in TOP_STRING_KEYS and type(value) is str:
            strings[key] = configuration_projection(value, budget, key)
        elif key in TOP_CONFIGURATION_SCALARS and not isinstance(value, (dict, list)):
            scalars[key] = configuration_projection(value, budget, key)
    records = raw.get('records')
    return {'top_level_keys': list(raw),
            'top_level_key_types': {key: kind(value) for key, value in raw.items()},
            'schema_paths': {key: {'types': sorted(item['types']), 'occurrences': item['occurrences'],
                                   'container_lengths': sorted(item['container_lengths'])}
                             for key, item in sorted(schema.items())},
            'configuration_metadata': config, 'top_level_string_metadata': strings,
            'top_level_configuration_scalars': scalars,
            'admitted_prep_metadata': admitted_prep_metadata(raw),
            'admitted_timing_metadata': admitted_timing_metadata(raw),
            'source_declaration_strings': declaration_strings(raw),
            'native_convention_named_schema_paths': [key for key in sorted(schema) if CONVENTION_KEY_PATTERN.search(key)],
            'Hs_named_schema_paths': [key for key in sorted(schema) if key.rsplit('.', 1)[-1] == 'Hs'],
            'record_count_if_present': len(records) if isinstance(records, list) else None,
            'configuration_projection_policy': {
                'configuration_roots': sorted(CONFIGURATION_ROOTS), 'top_string_keys': sorted(TOP_STRING_KEYS),
                'top_configuration_scalars': sorted(TOP_CONFIGURATION_SCALARS),
                'excluded_subtrees': sorted(EXCLUDED_SUBTREES),
                'all_other_list_contents_omitted': True, 'admitted_timing_lists_exported_separately': True,
                'numeric_literals_converted': False},
            'scientific_result_values_exported': False,
            'scope': 'Full key/type schema and bounded declared input metadata only; no run-result reduction or array-value read.'}


def descriptor_has_object(descriptor):
    if isinstance(descriptor, str):
        return re.fullmatch(r'[<>=|]?O[0-9]*', descriptor) is not None
    if isinstance(descriptor, list):
        for field in descriptor:
            if not isinstance(field, tuple) or len(field) not in (2, 3):
                raise ValueError('Malformed structured NPY descriptor')
        return any(descriptor_has_object(field[1]) for field in descriptor)
    raise ValueError('Unsupported NPY dtype descriptor')


def read_npy_header(stream):
    magic = stream.read(6)
    version_bytes = stream.read(2)
    if magic != b'\x93NUMPY' or len(version_bytes) != 2:
        raise ValueError('Missing or truncated NPY signature')
    version = tuple(version_bytes)
    if version == (1, 0):
        length_size, length_format, encoding = 2, '<H', 'latin1'
    elif version in ((2, 0), (3, 0)):
        length_size, length_format, encoding = 4, '<I', 'utf8' if version == (3, 0) else 'latin1'
    else:
        raise ValueError('Unsupported NPY version')
    encoded_length = stream.read(length_size)
    if len(encoded_length) != length_size:
        raise ValueError('Truncated NPY header length')
    length = struct.unpack(length_format, encoded_length)[0]
    if not 0 < length <= HEADER_LIMIT:
        raise ValueError('NPY header exceeds the metadata-only bound')
    encoded_header = stream.read(length)
    if len(encoded_header) != length:
        raise ValueError('Truncated NPY header')
    header = ast.literal_eval(encoded_header.decode(encoding).strip())
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Unexpected NPY header keys')
    if type(header['fortran_order']) is not bool:
        raise ValueError('Malformed NPY storage order')
    shape = header['shape']
    if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('Malformed NPY shape')
    return {'version': list(version), 'shape': list(shape), 'descriptor': header['descr'],
            'fortran_order': header['fortran_order'], 'object_payload': descriptor_has_object(header['descr']),
            'header_bytes': 8 + length_size + length, 'header_only': True,
            'array_values_loaded': False, 'object_payload_decoded': False}


def archive_metadata(path):
    members = []
    with zipfile.ZipFile(path) as archive:
        directory = archive.infolist()
        if len(directory) > ARCHIVE_MEMBER_LIMIT:
            raise ValueError('NPZ directory exceeds the admitted member count')
        duplicates = sorted(name for name, count in Counter(item.filename for item in directory).items() if count > 1)
        if duplicates:
            raise ValueError('Duplicate NPZ member names: ' + repr(duplicates))
        for member in directory:
            if member.flag_bits & 1:
                raise ValueError('Encrypted NPZ members are not admitted')
            item = {'name': member.filename, 'compressed_bytes': member.compress_size,
                    'uncompressed_bytes': member.file_size, 'compression_method': member.compress_type,
                    'zip_crc32': format(member.CRC, '08x'), 'array_values_loaded': False}
            if member.filename.endswith('.npy'):
                with archive.open(member) as stream:
                    item.update(read_npy_header(stream))
                if item['header_bytes'] > member.file_size:
                    raise ValueError('NPY header exceeds its declared member size')
                item['declared_bytes_after_header'] = member.file_size-item['header_bytes']
            members.append(item)
    return {'members': members,
            'object_member_names': [item['name'] for item in members if item.get('object_payload')],
            'array_values_loaded': False, 'object_payload_decoded': False,
            'scalar_text_payloads_decoded': False,
            'scope': 'ZIP directory and bounded NPY headers only; no array values, object decoding or member-payload CRC validation.'}


def scalar_Hs_configuration(path, headers):
    """Explicitly admitted configuration exception, never a general array read."""
    expected = next((item for item in headers['members'] if item['name'] == 'Hs.npy'), None)
    if expected is None:
        return {'status': 'missing', 'reason': 'no_Hs_configuration_scalar_member'}
    if expected.get('descriptor') != '<i8' or expected.get('shape') != [] or expected.get('object_payload') is not False:
        raise ValueError('Hs configuration member must be one nonobject little-endian int64 scalar')
    if expected['uncompressed_bytes'] != expected['header_bytes'] + 8:
        raise ValueError('Hs configuration scalar byte count differs from exactly eight')
    with zipfile.ZipFile(path) as archive:
        with archive.open('Hs.npy') as stream:
            header = read_npy_header(stream)
            if any(header[key] != expected[key] for key in ('descriptor', 'shape', 'fortran_order', 'header_bytes', 'object_payload')):
                raise ValueError('Hs configuration scalar header changed')
            payload = stream.read(8)
            if len(payload) != 8:
                raise ValueError('Truncated Hs configuration scalar')
    value = struct.unpack('<q', payload)[0]
    return {'status': 'present', 'source': 'Hs.npy', 'shape': [], 'descriptor': '<i8',
            'type': 'int', 'literal': str(value), 'configuration_payload_bytes': 8,
            'configuration_payload_hex': payload.hex(), 'object_payload_decoded': False,
            'scope': 'Only explicitly admitted configuration scalar; no field, basis, Gram or scientific result array loaded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Bprime_s16A_v1 metadata/schema/header/hash intake only; no numerical readout', thread_cap=1)
    destination = args.outdir/'input_metadata.json'
    if destination.exists():
        raise RuntimeError('Preserve the existing intake; use a fresh output directory')
    code = code_bindings()
    before = {role: identity(path) for role, path in INPUTS.items()}
    print('Hashed all twelve explicit source files; reading metadata and headers only.', flush=True)
    full_checks = {role: {'expected_md5': DECLARED_FULL_PREP_MD5, 'actual_md5': before[role]['md5'],
                           'matched': before[role]['md5'] == DECLARED_FULL_PREP_MD5}
                   for role in ('prep_v5_full_npz', 'kit_full_npz')}
    result = {'intake_schema_version': 2, 'runtime': runtime, 'code_bindings': code,
              'documentary_report': REPORT_REFERENCE, 'source_identities': before,
              'declared_full_prep_npz_md5_checks': full_checks,
              'prep_v5_vs_kit_copy': {suffix: {
                  'prep_v5_role': 'prep_v5_full_'+suffix, 'kit_role': 'kit_full_'+suffix,
                  'exact_byte_identity': all(before['prep_v5_full_'+suffix][key] == before['kit_full_'+suffix][key]
                                             for key in ('bytes', 'sha256', 'md5'))}
                  for suffix in ('json', 'npz')},
              'documentary_basis_byte_md5_prefixes': {
                  'supply': '37be42d2bf86', 'exit': 'a1ed5f1c7e49',
                  'status': 'not_tested_basis_array_bytes_not_loaded; not compared with whole-file hashes'},
              'json_metadata': {}, 'archive_metadata': {},
              'scalar_configuration_metadata': {},
              'scope': 'Metadata intake with lexical prep/timing metadata and explicitly admitted Hs scalar fallback only; no original run results, fields, bases, spectra, Grams, sink arrays or object values interpreted.',
              'custody': 'Every original remains in place on the box; this own metadata product is written on the box. No copy or transfer is performed.'}
    for role, path in INPUTS.items():
        if path.suffix == '.json':
            result['json_metadata'][role] = json_metadata(path)
        elif path.suffix == '.npz':
            result['archive_metadata'][role] = archive_metadata(path)
        else:
            raise ValueError('Unadmitted source suffix')
        print('Saved metadata in memory for '+role+'.', flush=True)
    for role, path in INPUTS.items():
        if path.suffix == '.npz' and (role.startswith('prep_v5_') or role == 'kit_full_npz'):
            json_role = role[:-3]+'json'
            metadata = result['json_metadata'][json_role]
            json_Hs = {key: value for key, value in metadata['admitted_prep_metadata'].items()
                       if key.rsplit('.', 1)[-1] == 'Hs'}
            if json_Hs:
                result['scalar_configuration_metadata'][role] = {'status': 'JSON_source_preferred', 'JSON_role': json_role, 'Hs': json_Hs}
            elif metadata['Hs_named_schema_paths']:
                result['scalar_configuration_metadata'][role] = {'status': 'unresolved', 'reason': 'Hs_JSON_path_requires_explicit_mapping',
                                                                 'JSON_paths': metadata['Hs_named_schema_paths']}
            else:
                result['scalar_configuration_metadata'][role] = {
                    'JSON_role': json_role, 'Hs_absent_from_full_JSON_schema': True,
                    'Hs': scalar_Hs_configuration(path, result['archive_metadata'][role])}
    for role, path in INPUTS.items():
        if identity(path) != before[role]:
            raise RuntimeError('Source changed during intake: '+role)
    if code_bindings() != code:
        raise RuntimeError('Own source changed during intake')
    verify_threads(runtime, 'after_stdlib_intake')
    result['all_source_and_code_identities_rechecked_unchanged'] = True
    result['declared_full_prep_npz_pins_all_match'] = all(item['matched'] for item in full_checks.values())
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(destination), 'source_file_count': len(before),
                      'full_prep_pins_match': result['declared_full_prep_npz_pins_all_match'],
                      'duplicate_prep_identities': result['prep_v5_vs_kit_copy'],
                      'object_members': {role: item['object_member_names'] for role, item in result['archive_metadata'].items()}}), flush=True)
    if not result['declared_full_prep_npz_pins_all_match']:
        raise SystemExit('Declared full-prep MD5 pin mismatch; intake saved for inspection.')


if __name__ == '__main__':
    main()
