"""Stdlib-only, metadata-only intake of the fixed saved-output gate package.

Whole files are hashed opaquely. Only selected JSON input metadata is
materialized; outcome values are structurally skipped without numeric
conversion. NPZ members are header-only except bounded T/n_supply payloads.
No G/R/e/qF/C_in/U_sub or embedded-manifest payload is interpreted.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import zipfile

from gate_runtime import (SPEC_SHA256, admit, check_memory, code_bindings,
                          enforce_storage, identity, require_identity, verify_threads)


SOURCE_ROOT = Path('<internal>/bprime')
RUN_STEMS = {
    'baseline': 's16A_qh1_v1/s16A_1_F1_d0_dt0.5_T100_both',
    '2c': 's16A_h100f_v1/s16A_2c_F1_dthalf_dt0.25_T100_both',
    '2b': 's16A_qh1_v1/s16A_2b_F1_onset225_dt0.5_T100_both',
    'x_full': 's16A_h100x8_v2/g2/s16A_1_F1_d0_dt0.5_T6.932_both',
    'x_n32': 's16A_h100x8_v2/g2/s16A_1_F1_d0_dt0.5_T69.32_both_n32',
    'run1_sup': 's16A_h100x8_v1/s16A_1_F1_d0_dt0.5_T69.32_sup',
    'run1_exit': 's16A_h100x8_v1/s16A_1_F1_d0_dt0.5_T69.32_exit',
}
PREP_PATHS = {
    'kit_full': 'kit_s16A_v2/prep/F1_s16_d0_prep.npz',
    'prep_full': 'prep_v5/F1_s16_d0_prep.npz',
    'prep_sup': 'prep_v5/F1_s16_d0_sup_prep.npz',
    'prep_exit': 'prep_v5/F1_s16_d0_exit_prep.npz',
    'kit_onset225': 'kit_s16A_v2/prep/F1_s16_d0_onset225_bb_prep.npz',
    'prep_onset225': 'prep_v5/F1_s16_d0_onset225_bb_prep.npz',
}
INPUT_JSON_KEYS = frozenset(('args', 'prep_manifest', 'schema', 'record_schema',
    'script', 'prep', 'port_tag', 'hypothesis_tag', 'backend', 'device', 'K',
    'n_supply', 'clocks_over_L', 'cin', 'own_md5', 'npz_md5'))
MAX_HEADER_BYTES = 65536
MAX_JSON_BYTES = 32*1024**2
MAX_SELECTED_JSON_BYTES = 4*1024**2
MAX_MEMBERS = 1024
MAX_T_VALUES = 64


def _unchanged(path, binding):
    current = Path(path).stat()
    if any(int(getattr(current, key)) != value for key, value in binding['stable_stat'].items()):
        raise RuntimeError('Input changed during metadata access: '+str(path))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate input-metadata JSON key: '+key)
        result[key] = value
    return result


def _nonfinite_constant(value):
    raise ValueError('Nonfinite required input-metadata literal: '+value)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite required input-metadata number')
    return number


def _white(text, pos):
    while pos < len(text) and text[pos] in ' \t\r\n':
        pos += 1
    return pos


def _string_end(text, pos):
    if pos >= len(text) or text[pos] != '"':
        raise ValueError('Expected JSON string')
    pos += 1
    while pos < len(text):
        if text[pos] == '\\':
            pos += 2
        elif text[pos] == '"':
            return pos+1
        else:
            pos += 1
    raise ValueError('Unterminated JSON string')


def _skip_value(text, pos, depth=0):
    """Skip syntax only; never convert or store an outcome numeric value."""
    if depth > 128:
        raise ValueError('JSON nesting exceeds the bounded metadata scanner')
    pos = _white(text, pos)
    if pos >= len(text):
        raise ValueError('Truncated JSON value')
    if text[pos] == '"':
        return _string_end(text, pos)
    if text[pos] in '[{':
        opening = text[pos]
        closing = ']' if opening == '[' else '}'
        pos = _white(text, pos+1)
        if pos < len(text) and text[pos] == closing:
            return pos+1
        while True:
            if opening == '{':
                pos = _white(text, _string_end(text, pos))
                if pos >= len(text) or text[pos] != ':':
                    raise ValueError('Missing JSON object colon')
                pos += 1
            pos = _white(text, _skip_value(text, pos, depth+1))
            if pos >= len(text):
                raise ValueError('Truncated JSON container')
            if text[pos] == closing:
                return pos+1
            if text[pos] != ',':
                raise ValueError('Missing JSON value separator')
            pos = _white(text, pos+1)
    start = pos
    while pos < len(text) and text[pos] not in ' \t\r\n,]}':
        pos += 1
    if pos == start:
        raise ValueError('Empty JSON value')
    return pos


def _kind_at(text, pos):
    first = text[_white(text, pos)]
    return {'{': 'object', '[': 'array', '"': 'string', 't': 'boolean',
            'f': 'boolean', 'n': 'null'}.get(first, 'unmaterialized_atom')


def input_json_metadata(path, binding):
    if binding['bytes'] > MAX_JSON_BYTES:
        raise ValueError('JSON exceeds the bounded metadata scanner size')
    text = Path(path).read_text(encoding='utf-8')
    _unchanged(path, binding)
    pos = _white(text, 0)
    if pos >= len(text) or text[pos] != '{':
        raise ValueError('Source JSON must be a top-level object')
    pos = _white(text, pos+1)
    metadata, keys, kinds, skipped = {}, [], {}, []
    selected_bytes = 0
    if pos < len(text) and text[pos] != '}':
        while True:
            end = _string_end(text, pos)
            key = json.loads(text[pos:end])
            if key in kinds:
                raise ValueError('Duplicate top-level source JSON key: '+key)
            keys.append(key)
            pos = _white(text, end)
            if pos >= len(text) or text[pos] != ':':
                raise ValueError('Missing top-level JSON colon')
            start = _white(text, pos+1)
            kinds[key] = _kind_at(text, start)
            end = _skip_value(text, start)
            if key in INPUT_JSON_KEYS:
                selected_bytes += len(text[start:end].encode('utf-8'))
                if selected_bytes > MAX_SELECTED_JSON_BYTES:
                    raise ValueError('Selected input metadata exceeds its bounded budget')
                metadata[key] = json.loads(text[start:end], object_pairs_hook=_unique_object,
                                          parse_constant=_nonfinite_constant, parse_float=_finite_float)
            else:
                skipped.append(key)
            pos = _white(text, end)
            if pos >= len(text):
                raise ValueError('Truncated source JSON object')
            if text[pos] == '}':
                break
            if text[pos] != ',':
                raise ValueError('Missing top-level JSON separator')
            pos = _white(text, pos+1)
    if pos >= len(text) or text[pos] != '}' or _white(text, pos+1) != len(text):
        raise ValueError('Invalid source JSON closing structure')
    return {'input_metadata': metadata, 'top_level_keys': keys, 'top_level_value_kinds': kinds,
            'skipped_outcome_or_unadmitted_keys': skipped,
            'outcome_numeric_values_materialized': False,
            'npz_md5_label_note': 'Any npz_md5 source-record field is inherited profile/source metadata, not an independently established prep checksum.'}


def parse_md5sums(path, binding):
    if binding['bytes'] > 1024**2:
        raise ValueError('Checksum manifest exceeds the fixed intake budget')
    text = Path(path).read_text(encoding='utf-8')
    _unchanged(path, binding)
    entries = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        match = re.fullmatch(r'([0-9A-Fa-f]{32}) [ *](.+)', line)
        if match is None:
            raise ValueError('Unsupported complete MD5SUMS entry at line'+str(line_number))
        filename = match[2]
        if '\x00' in filename or '\\' in filename:
            raise ValueError('Unsupported escaped checksum filename')
        target = Path(filename)
        target = target if target.is_absolute() else Path(path).parent/target
        entries.append({'line': line_number, 'filename': filename,
                        'resolved_path': str(target.resolve()), 'md5': match[1].lower()})
    return entries


def _read_exact(stream, count):
    result = stream.read(count)
    if len(result) != count:
        raise ValueError('Truncated NPY header or admitted coordinate payload')
    return result


def _has_object(value):
    if isinstance(value, str):
        return re.fullmatch(r'[<>=|]?O[0-9]*', value) is not None
    if isinstance(value, (tuple, list)):
        return any(_has_object(item) for item in value)
    return False


def npy_header(stream):
    if _read_exact(stream, 6) != b'\x93NUMPY':
        raise ValueError('Missing NPY magic')
    version = tuple(_read_exact(stream, 2))
    if version == (1, 0):
        length = struct.unpack('<H', _read_exact(stream, 2))[0]
        offset = 10
    elif version in ((2, 0), (3, 0)):
        length = struct.unpack('<I', _read_exact(stream, 4))[0]
        offset = 12
    else:
        raise ValueError('Unsupported NPY format version')
    if not 1 <= length <= MAX_HEADER_BYTES:
        raise ValueError('NPY header exceeds bounded size')
    literal = _read_exact(stream, length).decode('utf-8' if version == (3, 0) else 'latin1')
    header = ast.literal_eval(literal)
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Malformed NPY header dictionary')
    shape = header['shape']
    if not isinstance(shape, tuple) or any(type(size) is not int or size < 0 for size in shape) or type(header['fortran_order']) is not bool:
        raise ValueError('Malformed NPY shape/order metadata')
    if not isinstance(header['descr'], (str, list)):
        raise ValueError('Unsupported NPY dtype descriptor metadata')
    return {'npy_version': list(version), 'descr': header['descr'], 'dtype_descr': header['descr'],
            'shape': list(shape), 'fortran_order': header['fortran_order'],
            'contains_object': _has_object(header['descr']), 'header_length': length,
            'data_offset': offset+length, 'header_literal': literal}


def _coordinate(stream, name, header, member_size):
    descriptor = header['descr']
    if not isinstance(descriptor, str) or header['contains_object']:
        raise ValueError('Admitted coordinate/count must have a primitive non-object dtype')
    shape = header['shape']
    if name == 'T':
        if len(shape) > 1:
            raise ValueError('T must be a scalar or bounded one-dimensional float64 array')
        count = 1 if not shape else shape[0]
        if not 1 <= count <= MAX_T_VALUES or re.fullmatch(r'[<>=]?f8', descriptor) is None:
            raise ValueError('T exceeds its bounded float64 coordinate contract')
        kind, width, format_code = 'float64', 8, 'd'
    else:
        if shape:
            raise ValueError('n_supply must be an explicit scalar NPY count')
        count = 1
        integer = re.fullmatch(r'([<>=|]?)([iu])([1248])', descriptor)
        if integer is not None:
            width = int(integer[3]); kind = 'signed_integer' if integer[2] == 'i' else 'unsigned_integer'
            if integer[1] == '|' and width != 1:
                raise ValueError('A multi-byte integer count needs an explicit or native byte order')
            format_code = {('i', 1): 'b', ('i', 2): 'h', ('i', 4): 'i', ('i', 8): 'q',
                           ('u', 1): 'B', ('u', 2): 'H', ('u', 4): 'I', ('u', 8): 'Q'}[integer[2], width]
        elif re.fullmatch(r'[<>=]?f8', descriptor):
            width, format_code, kind = 8, 'd', 'float64_count_label'
        else:
            raise ValueError('n_supply requires a bounded integer or float64 scalar')
    byte_count = count*width
    if member_size != header['data_offset']+byte_count:
        raise ValueError('Coordinate member byte count does not match its exact header')
    payload = _read_exact(stream, byte_count)
    if stream.read(1):
        raise ValueError('Trailing coordinate/count payload bytes')
    order = descriptor[0] if descriptor[0] in '<>=' else '='
    byte_order = '>' if order == '>' or order == '=' and sys.byteorder == 'big' else '<'
    values = struct.unpack(byte_order+format_code*count, payload)
    items, finite = [], True
    for index, value in enumerate(values):
        row = {'index': [] if not shape else [index], 'payload_hex': payload[index*width:(index+1)*width].hex()}
        if isinstance(value, float):
            row['float64_hex'] = value.hex(); row['finite'] = math.isfinite(value)
            finite = finite and row['finite']
            if row['finite']:
                row['value'] = value
            else:
                row['unavailable_reason'] = 'nonfinite_admitted_coordinate_or_count'
        else:
            row['value'] = value; row['finite'] = True
        items.append(row)
    return {'status': 'available' if finite else 'unavailable_nonfinite', 'dtype': descriptor,
            'shape': shape, 'kind': kind, 'byte_count': byte_count,
            'payload_hex': payload.hex(), 'values': items,
            'unit_conversion_or_clock_join_performed': False}


def archive_headers(path, binding, coordinate_payloads=False):
    headers, coordinates, opened = {}, {}, []
    with zipfile.ZipFile(path, 'r') as archive:
        members = archive.infolist()
        if len(members) > MAX_MEMBERS or len({item.filename for item in members}) != len(members):
            raise ValueError('NPZ member count/duplicate-name admission failed')
        for member in members:
            name = member.filename
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or member.is_dir() or not name.endswith('.npy'):
                raise ValueError('NPZ must contain unique safe NPY array members')
            key = name[:-4]
            with archive.open(member, 'r') as stream:
                header = npy_header(stream)
                header.update(member=name, zip_file_size=member.file_size,
                              zip_compressed_size=member.compress_size, zip_CRC32_hex=format(member.CRC, '08x'))
                headers[key] = header
                if coordinate_payloads and key in ('T', 'n_supply'):
                    coordinates[key] = _coordinate(stream, key, header, member.file_size)
                    opened.append(name)
    _unchanged(path, binding)
    if coordinate_payloads:
        for key in ('T', 'n_supply'):
            coordinates.setdefault(key, {'status': 'not_present', 'payload_read': False})
    return {'array_headers': headers, 'coordinate_payloads': coordinates,
            'admitted_payload_members_opened': opened, 'scientific_payloads_opened': False,
            'object_payloads_opened': False, 'embedded_manifest_payload_opened': False}


def inventory_run(role, relative_stem):
    stem = SOURCE_ROOT/relative_stem
    record = {'role': role, 'status': 'unavailable', 'input_metadata': {},
              'array_headers': {}, 'coordinate_payloads': {}, 'checks': {}}
    try:
        for key, extension in (('json', '.json'), ('npz', '.npz'), ('manifest', '.MD5SUMS')):
            record[key] = identity(str(stem)+extension)
        entries = parse_md5sums(record['manifest']['path'], record['manifest'])
        record['manifest_entries'] = entries
        for key in ('json', 'npz'):
            found = [item for item in entries if item['resolved_path'] == record[key]['resolved_path']]
            if len(found) != 1:
                raise ValueError('Required exact checksum entry is missing or duplicated: '+key)
            record['checks'][key+'_MD5SUMS'] = found[0]['md5'] == record[key]['md5']
        if not all(record['checks'].values()):
            raise ValueError('Required whole-file MD5 differs from complete source MD5SUMS')
        record.update(input_json_metadata(record['json']['path'], record['json']))
        record.update(archive_headers(record['npz']['path'], record['npz'], coordinate_payloads=True))
        if record['coordinate_payloads']['T']['status'] != 'available':
            raise ValueError('Required T coordinate is unavailable')
        if record['coordinate_payloads']['n_supply']['status'].startswith('unavailable'):
            raise ValueError('Present n_supply count is unavailable')
        record['status'] = 'available'
    except Exception as error:
        record['unavailable_reason'] = {'type': type(error).__name__, 'message': str(error)}
    return record


def inventory_prep(role, relative_path):
    record = {'role': role, 'status': 'unavailable', 'checksum_manifest_status': 'not_supplied; no prep checksum inferred'}
    try:
        record['npz'] = identity(SOURCE_ROOT/relative_path)
        record.update(archive_headers(record['npz']['path'], record['npz'], coordinate_payloads=False))
        record['status'] = 'available'
    except Exception as error:
        record['unavailable_reason'] = {'type': type(error).__name__, 'message': str(error)}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Own Bprime gate stdlib metadata-only intake')
    if args.spec_sha256 != SPEC_SHA256:
        raise ValueError('CLI specification differs from the final admitted contract')
    bindings = {'spec': require_identity(args.spec, SPEC_SHA256), 'self': identity(__file__),
                'runtime': identity(Path(__file__).with_name('gate_runtime.py')), 'code': code_bindings()}
    record = {'stage': 'metadata_only_intake', 'bindings': bindings, 'spec_binding': bindings['spec'],
              'code_bindings': bindings['code'], 'runtime': runtime,
              'runs': {role: inventory_run(role, stem) for role, stem in RUN_STEMS.items()},
              'preps': {role: inventory_prep(role, path) for role, path in PREP_PATHS.items()},
              'scientific_payloads_opened': False, 'foreign_code_imported_or_executed': False,
              'output_bindings': {}, 'custody': 'Raw metadata JSON and all records remain on the box.'}
    for group in ('runs', 'preps'):
        for item in record[group].values():
            for key in ('json', 'npz', 'manifest'):
                if key in item:
                    try:
                        _unchanged(item[key]['path'], item[key])
                    except Exception as error:
                        item['status'] = 'unavailable'
                        item['completion_identity_check'] = {'status': 'unavailable', 'file_role': key,
                            'type': type(error).__name__, 'message': str(error)}
    record['all_passed'] = all(item['status'] == 'available' for group in ('runs', 'preps') for item in record[group].values())
    if code_bindings() != bindings['code']:
        raise RuntimeError('Source changed during metadata-only intake')
    require_identity(args.spec, SPEC_SHA256)
    verify_threads(runtime, 'at_completion'); check_memory(runtime, 'at_completion')
    record['storage_before_save'] = enforce_storage()
    data = (json.dumps(record, indent=2, allow_nan=False)+'\n').encode('utf-8')
    enforce_storage(len(data)+8192)
    with (args.outdir/'input_metadata.json').open('xb') as stream:
        stream.write(data)
    enforce_storage()
    print('Saved metadata-only intake; all_passed='+str(record['all_passed']), flush=True)
    return 0 if record['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
