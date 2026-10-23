"""Check upstream archive bytes and saved input witnesses; no contractions."""
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path('<internal>/profiles_v1')
ROOT = BASE.parent/'profiles_missing_v2'/'relocated_inputs'
MIRROR = Path('<internal>/states')
ALIAS = Path('<internal>/states')
MANIFEST = Path('<internal>/pod_md5.txt')
TAG = 'c19_rs64_A_R2_tau8_ell4_avg_r11'
EXPECTED = {12: '8cf1aef596d405804acf250b8dd9611a',
            15: '5dec15a3d172ec233ae1f234c10f1b1a',
            16: 'de85355758d0c6f2eeb8b40f6c34430a'}


def main():
    witness_path = BASE.parent/'current_v1'/'core_well_current_v1.json'
    pin_paths = {BASE/'core_well_input_v1.py': '294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b',
        BASE/'core_well_profiles_v1.json': '96a9a3bfb67e4cc5759c8a50eb7cdf6bcf51e13c15ec10e5f6c3e42730a77aa8',
        witness_path: '5798c16fc31fb2fdf640341b15362cbeb9c3e7fe1c7b658c1b96f22e330ef99f'}
    for path, digest in pin_paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Historical witness/helper changed: '+str(path))
    import numpy as np
    sys.path.insert(0, str(BASE))
    from core_well_input_v1 import StoredNpz, file_identity
    target = ROOT/'relocated_input_check_v2.json'
    if target.exists() or target.with_suffix('.json.tmp').exists():
        raise FileExistsError('Existing source check must be assessed, never overwritten')
    manifest_bytes = MANIFEST.read_bytes()
    lines = manifest_bytes.decode().splitlines()
    previous = json.loads((BASE/'core_well_profiles_v1.json').read_text())
    current = json.loads(witness_path.read_text())
    result = dict(status='incomplete', scientific_evaluations=0, inputs={},
        upstream_record=dict(path=str(MANIFEST), sha256=hashlib.sha256(manifest_bytes).hexdigest()),
        limitation='Matches upstream production mirror and retained content witnesses; old missing inode whole-file and historical sea payload identity were not independently hashed',
        pinned_witnesses={str(k): v for k,v in pin_paths.items()}, raw_product_pins={},
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def descriptor_identity(state):
        return state.st_size, state.st_mtime_ns, state.st_dev, state.st_ino
    for clock in (12, 15, 16):
        name = 'orbitals_'+TAG+'_T%07.2f.npz' % clock
        path, alias = MIRROR/TAG/name, ALIAS/TAG/name
        suffix = 'workspace/engine/states/'+TAG+'/'+name
        rows = [x.split() for x in lines if x.endswith('  '+suffix)]
        if rows != [[EXPECTED[clock], suffix]]:
            raise ValueError('Missing or ambiguous upstream checksum')
        identity = file_identity(path)
        old = previous['slices'][str(clock)]['input']
        if path.name != Path(old['path']).name or any(identity[k] != old[k] for k in ('size','mtime_ns')):
            raise ValueError('Mirror differs from original size/clock-named metadata')
        md5, sha = hashlib.md5(), hashlib.sha256()
        expected_stat = tuple(identity[k] for k in ('size','mtime_ns','device','inode'))
        with path.open('rb') as stream:
            if descriptor_identity(os.fstat(stream.fileno())) != expected_stat:
                raise ValueError('Hashing descriptor differs from captured identity')
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                md5.update(block); sha.update(block)
            if descriptor_identity(os.fstat(stream.fileno())) != expected_stat:
                raise ValueError('Hashing descriptor changed')
        if md5.hexdigest() != EXPECTED[clock] or file_identity(path) != identity:
            raise ValueError('Mirror byte checksum or identity changed')
        raw_path = BASE/f'raw_T{clock:02d}.npz'
        raw_pin = next(x['sha256'] for x in previous['saved_arrays'] if x['path'] == str(raw_path))
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_pin:
            raise ValueError('Saved raw product changed')
        result['raw_product_pins'][str(raw_path)] = raw_pin
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {k: archive[k] for k in ('R','N_applied','v_applied','weights')}
        verified = []
        with StoredNpz(path) as source:
            source.verify_identity()
            if descriptor_identity(source.initial) != expected_stat or file_identity(path) != identity:
                raise ValueError('Mapped archive differs from the hashed identity')
            for key, value in raw.items():
                if not np.array_equal(source.array(key), value):
                    raise ValueError('Saved raw witness differs: '+key)
                verified.append(key)
            if (float(source.array('T')) != clock*64 or float(source.array('rs')) != 64
                    or not np.array_equal(source.array('kappas'), np.arange(1,11))
                    or not np.array_equal(source.array('weights'), 16*np.arange(1,11))):
                raise ValueError('Clock/channel convention changed')
            expected_dtype = np.dtype('complex64' if clock == 15 else 'complex128')
            for k in range(1,11):
                f = source.array('f_'+str(k))
                dust = source.array('dust_'+str(k))
                count = (5,4,3,2,2,0,0,0,0,0)[k-1]
                if dust.shape != (4096,count) or f.shape != (count,) or not np.array_equal(f,np.ones(count)):
                    raise ValueError('Dust occupation convention differs')
                if dust.dtype != expected_dtype:
                    raise ValueError('Dust storage precision differs')
                for prefix in ('sea_', 'R2_'):
                    header = source.member_header(prefix+str(k))
                    if tuple(header['shape']) != (4096,2048) or header['dtype'] != expected_dtype:
                        raise ValueError('Sea/reference header differs')
                if clock == 16:
                    for label, array in (('dust',dust),('occupations',f)):
                        witness = current['channels'][str(k)]['metadata'][label]
                        if (hashlib.sha256(array.tobytes(order='C')).hexdigest() != witness['sha256_c_bytes']
                                or list(array.shape) != witness['shape'] or array.dtype.str != witness['dtype']):
                            raise ValueError('Historical T16 dust/occupation byte witness differs')
                verified.extend(['dust_'+str(k),'f_'+str(k)])
            if clock == 16:
                for key, witness in current['metadata_arrays'].items():
                    array = source.array(key)
                    if (hashlib.sha256(array.tobytes(order='C')).hexdigest() != witness['sha256_c_bytes']
                            or list(array.shape) != witness['shape'] or array.dtype.str != witness['dtype']):
                        raise ValueError('Historical T16 metadata byte witness differs')
            source.verify_identity()
            if file_identity(path) != identity:
                raise ValueError('Mapped archive changed from the hashed identity')
        alias_identity = file_identity(alias)
        if any(identity[k] != alias_identity[k] for k in ('device','inode','size','mtime_ns')):
            raise ValueError('Mirror alias no longer names the same upstream inode')
        result['inputs'][str(clock)] = dict(original_identity=old, current_identity=identity,
            alias_identity=alias_identity, upstream_md5_expected=EXPECTED[clock],
            upstream_md5_observed=md5.hexdigest(), current_sha256=sha.hexdigest(),
            same_hardlink_alias=True, verified_saved_members=verified,
            historical_T16_C_byte_witnesses_verified=(clock == 16))
    if MANIFEST.read_bytes() != manifest_bytes:
        raise ValueError('Upstream checksum record changed during verification')
    result['status'] = 'passed'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
