#!/usr/bin/env python3
"""Missing-only B11 signed views and K12 extension; execute only on the CPU box."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import sys
import time
import traceback

ENTRY = time.monotonic()
BASE = Path('<internal>/Bprime_channel_lead_v1')
ROOT = Path('<internal>/bprime_channel_lead_v1')
PYTHON = Path('<internal>/python')
FROZEN = '4069e461cc25d44cccdae6b0d2f6f6c9d1b45816f4abe0b385913111b586ca5f'
SEEDS = {
 'controls': ('controls_952e9e6e_v1/controls.json', 'bc3f5093671243721c05eaa7f2f7013ff8a4f9405c9ae89a3c7bdd54f3a6f45c'),
 'native': ('native_952e9e6e_v1/native.json', '477e6dcee52372949dd4a6314cdacfc6c8e288611f5c1764d4f63989daeed4a3'),
 'fields': ('fields_952e9e6e_v1/fields.json', '1b4428fceb8f580c0b5bb3c409f6dbfdb0c738299500bd1b7496489e2368fb94'),
 'b11': ('b11_inputs_952e9e6e_v1/b11.json', '661d8c8df963aa758e448465dd2bbed0e6153046a383cc659b091153b5ab7aee'),
 'k10': ('setup_benchmark_K10_952e9e6e_v1/run.json', '0841a6e86400afa64c0022725b20902212ac82066b5cf949c786f6e0f9bfbef4'),
 'setup10': ('setup_benchmark_K10_952e9e6e_v1/setup_benchmark.json', '6181a094748fbc47555e478422d7adb3244a510beb1271854f41e2b215a4e07f'),
}
PREFIXES = tuple(f'{f}_k{k}' for f in ('F1', 'F2') for k in (1, 2, 3))
PRIMARY = ('energies selected_indices source_input_energy source_column_map '
 'energy_multiset_representative_permutation individual_comparison_available '
 'source_energy_feasible_mask source_energy_feasible_permutations '
 'selected_cluster_membership energy_difference').split()
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
STOP = []

class Stopped(Exception): pass
class Withheld(Exception): pass

def handler(sig, frame):
    STOP.append(int(sig))

def boundary():
    if STOP:
        raise Stopped('handled signals: '+repr(STOP))

def stamp():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (time.monotonic(), r.ru_utime, r.ru_stime)

def elapsed(before):
    now = stamp()
    return dict(zip(('wall_seconds', 'user_seconds', 'system_seconds'),
                    (b-a for a, b in zip(before, now))))

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, complex): return {'real': clean(x.real), 'imag': clean(x.imag)}
    if isinstance(x, float) and not math.isfinite(x):
        return {'inherited_nonfinite': repr(x)}
    if x is None or type(x) in (str, int, float, bool): return x
    raise TypeError('unsupported JSON value '+str(type(x)))

def atomic_json(path, value):
    temp = path.with_name(path.name+'.partial.'+str(time.monotonic_ns()))
    with temp.open('x') as f:
        json.dump(clean(value), f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def metadata(st):
    return {k: getattr(st, 'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}

def pin_read(path, sha, data=False, honor_stop=True):
    path = Path(path)
    observation = {'path': str(path), 'expected_sha256': sha, 'phase': 'resolve', 'bytes_read': 0}
    digest = None
    try:
        resolved = str(path.resolve(strict=True)); observation['resolved_path'] = resolved
        digest = hashlib.sha256(); chunks = [] if data else None
        observation['phase'] = 'open'
        with path.open('rb') as f:
            observation['phase']='initial_fstat'
            before = metadata(os.fstat(f.fileno())); observation['before'] = before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode): raise ValueError('regular input required')
            observation['phase'] = 'read'
            while True:
                block = f.read(8*1024**2)
                if not block: break
                digest.update(block); observation['bytes_read'] += len(block)
                if data: chunks.append(block)
                if honor_stop: boundary()
            observation['phase']='final_fstat'
            after = metadata(os.fstat(f.fileno())); observation['after'] = after
            observation['phase']='close'
        observation['phase']='path_stat'
        current = metadata(path.stat()); observation['path_now'] = current
        observation['phase']='final_resolve'
        resolved_now = str(path.resolve(strict=True)); observation['resolved_path_now'] = resolved_now
        actual = digest.hexdigest(); observation['sha256'] = actual
        observation['phase'] = 'compare'
        if before != after or before != current or resolved != resolved_now or (sha is not None and actual != sha):
            raise ValueError('changed or mismatched input: '+str(path))
        return ({'path': str(path), 'resolved_path': resolved, 'sha256': actual, **before},
                b''.join(chunks) if data else None)
    except BaseException as exc:
        if digest is not None: observation['partial_sha256']=digest.hexdigest()
        exc.observation = observation
        raise

def file_identity(path):
    # Output hashing preserves the same partial observations as input hashing.
    return pin_read(path, None, honor_stop=False)[0]


def plans(mode):
    if mode == 'b11-diagnosis':
        files = ['radial_rows.npz']+[p+'_'+s+'.npz' for p in PREFIXES for s in ('primary','delta')]
        checks = ['D.radial_schema']+[f'D.{p}.{s}' for p in PREFIXES for s in ('primary_schema','delta_finite')]
    else:
        files = ['label_context.npz']
        checks = ['K.label_join']
        for c in range(4):
            files += [f'c{c}_{s}.npz' for s in ('H','A_N','G_ON','G_NO','G_NN','P_ON','P_NO','P_NN')]
            files += [f'setup_tile_c{c}.npz']
            files += [f'c{c}_{s}_check.npz' for s in ('G_NO','G_NN','P_NO','P_NN')]
            checks += [f'K.c{c}.{s}' for s in ('geometry','H_Hermitian','G_cross','G_new','P_cross','P_new','copy_A','copy_G','copy_P')]
    return files, checks

class Records:
    def __init__(self, mode, out):
        self.mode, self.out = mode, out
        self.files, self.checks = plans(mode)
        self.saved, self.decisions, self.pins = {}, {}, {}
        self.results, self.error, self.withheld, self.terminal = {}, None, None, False
        self.errors=[];self.postcheck_plan=[];self.postchecks={}
        self.sequence = 0; self.inherited = {}; self.timings = []
        self.name = 'b11_diagnosis_result.json' if mode == 'b11-diagnosis' else 'setup_increment_K12_result.json'
        self.journal = (out/'admission.jsonl').open('x')
        self.event('plan', mode, {'NPZ_files': self.files, 'decisions': self.checks,
            'completion_is_not_physical_clearance': True})
        self.checkpoint()
    def event(self, kind, name, value, refs=()):
        row = {'sequence': self.sequence+1, 'kind': kind, 'name': name,
               'payload': clean(value), 'operand_references': list(refs)}
        self.journal.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
        self.journal.flush(); os.fsync(self.journal.fileno()); self.sequence += 1
        return {'journal':'admission.jsonl','sequence':self.sequence,'kind':kind,'name':name}
    def checkpoint(self):
        missing_files = [p for p in self.files if p not in self.saved]
        missing_checks = [p for p in self.checks if p not in self.decisions]
        complete = not missing_files and not missing_checks
        truth = complete and all(self.decisions[p] is True for p in self.checks)
        status = ('completed' if truth else 'completed_with_control_failures') if complete else ('incomplete' if self.terminal else 'running')
        atomic_json(self.out/self.name, {'mode': self.mode, 'status': status,
            'raw_plan': self.files, 'decision_plan': self.checks, 'saved_files': self.saved,
            'raw_complete': not missing_files, 'validation_complete': not missing_checks,
            'all_controls_passed': truth, 'target_complete': complete,
            'missing_files': missing_files, 'missing_decisions': missing_checks,
            'decisions': self.decisions, 'results': self.results, 'inherited': self.inherited,
            'input_pins': list(self.pins.values()), 'error': self.error, 'errors':self.errors, 'withheld': self.withheld,
            'final_postcheck_plan':self.postcheck_plan,'final_postchecks':self.postchecks,
            'missing_final_postchecks':[k for k in self.postcheck_plan if k not in self.postchecks],
            'all_final_inputs_unchanged':bool(self.postcheck_plan) and len(self.postchecks)==len(self.postcheck_plan) and all(x.get('matched') is True for x in self.postchecks.values()),
            'handled_signals': STOP, 'terminal': self.terminal, 'phase_timings': self.timings,
            'elapsed': elapsed((ENTRY,0.,0.)), 'sequence': self.sequence,
            'scope': 'saved-input diagnostic and incremental representative tiles only; no B11 gate repair or physical propagation'})
    def save(self, name, arrays, refs=(), compute=None):
        if name not in self.files or name in self.saved: raise ValueError('unplanned or duplicate output '+name)
        started = stamp(); path = self.out/name; temp = path.with_name(path.name+'.partial')
        with temp.open('xb') as f:
            np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
        if path.exists(): raise ValueError('immutable destination exists')
        os.replace(temp, path)
        self.saved[name] = file_identity(path)
        receipt = {'file': self.saved[name], 'keys': {k: {'shape': list(v.shape), 'dtype': v.dtype.str} for k,v in arrays.items()},
                   'serialization_and_hash': elapsed(started), 'calculation': compute}
        self.event('returned_raw', name, receipt, refs)
        self.checkpoint()
    def unit(self, name, fn, refs=()):
        boundary(); self.event('begin', name, {}, refs); started = stamp()
        arrays = fn(); receipt = elapsed(started)
        self.save(name, arrays, refs, receipt)
        return arrays
    def decide(self, name, passed, actual, expected, refs=()):
        if name not in self.checks or name in self.decisions: raise ValueError('unplanned or duplicate decision')
        passed = bool(passed)
        self.event('comparison_actual', name, actual, refs)
        self.event('comparison_expected', name, expected, refs)
        self.decisions[name] = passed
        self.event('comparison_decision', name, passed, refs)
        self.checkpoint()
        if not passed:
            self.withheld = name
            raise Withheld(name)
        # A complete final result survives a stop received during its last receipt.
        if len(self.saved) != len(self.files) or len(self.decisions) != len(self.checks): boundary()
    def admit(self, path, sha, data=False):
        key = str(path)
        if key in self.pins and self.pins[key]['sha256'] != sha:
            self.event('conflicting_requested_pin',key,{'existing':self.pins[key],'requested_sha256':sha})
            raise ValueError('conflicting identity')
        started = stamp()
        try: pin, payload = pin_read(path, sha, data)
        except BaseException as exc:
            self.event('failed_input_observation', key, getattr(exc, 'observation', {}))
            raise
        self.event('input_read_returned',key,{'pin':pin,'read_hash_time':elapsed(started)})
        if key in self.pins and pin != self.pins[key]:
            observation={'existing':self.pins[key],'returned':pin}
            self.event('conflicting_returned_pin',key,observation)
            exc=ValueError('input identity interval changed');exc.observation=observation;raise exc
        self.pins[key] = pin
        self.event('input_pin', key, {'pin': pin, 'read_hash_time': elapsed(started)})
        return payload

@contextlib.contextmanager
def archive(rec, item):
    path=Path(item['path']);pin=rec.pins[str(path)]
    observation={'path':str(path),'admitted_pin':pin,'phase':'resolve','cleanup_errors':[]}
    f=None;z=None;body_error=None
    try:
        observation['resolved_path']=str(path.resolve(strict=True))
        observation['phase']='open';f=path.open('rb')
        observation['phase']='initial_fstat';observation['before']=metadata(os.fstat(f.fileno()))
        if observation['before']!={k:pin[k] for k in observation['before']} or observation['resolved_path']!=pin['resolved_path']:
            raise ValueError('archive descriptor does not match admitted bytes')
        observation['phase']='np_load';z=np.load(f,allow_pickle=False)
        observation['phase']='yield';yield z
    except BaseException as exc:
        body_error=exc
        observation['body_failure']={'type':type(exc).__name__,'message':str(exc),'phase':observation['phase']}
        observation['original_body_observation']=getattr(exc,'observation',None)
    finally:
        for phase,fn in (
            ('close_archive',lambda:z.close() if z is not None else None),
            ('final_fstat',lambda:metadata(os.fstat(f.fileno())) if f is not None and not f.closed else None),
            ('path_stat',lambda:metadata(path.stat())),
            ('final_resolve',lambda:str(path.resolve(strict=True))),
            ('close_descriptor',lambda:f.close() if f is not None else None)):
            observation['phase']=phase
            try:
                value=fn()
                if phase=='final_fstat':observation['after']=value
                if phase=='path_stat':observation['path_now']=value
                if phase=='final_resolve':observation['resolved_path_now']=value
            except BaseException as exc:
                observation['cleanup_errors'].append({'phase':phase,'type':type(exc).__name__,'message':str(exc)})
        before=observation.get('before')
        matched=(before is not None and before==observation.get('after')==observation.get('path_now')
                 and observation.get('resolved_path')==pin['resolved_path']==observation.get('resolved_path_now')
                 and not observation['cleanup_errors'])
        observation['interval_verified']=matched
        ref=rec.event('decoded_archive_observation',str(path),observation,[item])
        if not matched:
            error={'phase':'archive_interval','type':'ArchiveObservationFailure','message':'archive interval is not verified','observation':observation,'receipt':ref}
            rec.errors.append(error)
            if body_error is None:
                body_error=ValueError('archive interval is not verified')
        if body_error is not None:
            body_error.observation=observation
            raise body_error


def inherited_admission(rec):
    roles = ('controls','native','fields','b11') if rec.mode == 'b11-diagnosis' else ('controls','native','fields','k10','setup10')
    docs = {}
    for role in roles:
        path, sha = SEEDS[role]
        docs[role] = json.loads(rec.admit(BASE/path, sha, True))
    for role in ('controls','native','fields'):
        if docs[role].get('all_passed') is not True: raise ValueError('accepted prerequisite required: '+role)
    old_codes = docs['controls']['code_bindings']
    if len(old_codes) != 17 or any(not k.startswith('lead_') or not k.endswith('.py') for k in old_codes):
        raise ValueError('fixed seventeen-file source map required')
    for role in ('native','fields'):
        if docs[role]['bindings']['code'] != old_codes: raise ValueError('historical source binding differs')
    if docs['fields']['bindings']['native']['sha256'] != SEEDS['native'][1]: raise ValueError('field/native linkage')
    if rec.mode == 'b11-diagnosis':
        d = docs['b11']
        if d.get('all_passed') is not False or d['bindings']['fields']['sha256'] != SEEDS['fields'][1]:
            raise ValueError('exact saved B11 refusal and field linkage required')
        if d['bindings']['code'] != old_codes or d.get('source_consequences_opened') is not False: raise ValueError('B11 scope/source')
    else:
        if docs['k10'].get('all_passed') is not True or docs['k10'].get('failure') is not None:
            raise ValueError('complete inherited K10 required')
        core_names = ('lead_runtime.py','lead_native.py','lead_channels.py','lead_setup.py','lead_symmetry.py','lead_operator.py','lead_numerics.py','lead_controls.py')
        expected_core = {k: old_codes[k] for k in core_names}
        if docs['k10']['bindings']['code'] != old_codes or docs['setup10']['code_bindings'] != expected_core:
            raise ValueError('K10 historical source map')
        if docs['k10']['arguments']['K'] != 10 or docs['k10']['arguments']['field'] != 'F1': raise ValueError('K10 target')
    collected = {}
    def walk(x):
        if isinstance(x, dict):
            if 'path' in x and 'sha256' in x:
                key, sha = x['path'], x['sha256']
                if key in collected and collected[key] != sha: raise ValueError('conflicting recursive hash')
                collected[key] = sha
            else:
                for value in x.values(): walk(value)
        elif isinstance(x, list):
            for value in x: walk(value)
    for d in docs.values(): walk(d)
    for path, sha in sorted(collected.items()):
        if path not in rec.pins: rec.admit(Path(path), sha)
        elif rec.pins[path]['sha256'] != sha: raise ValueError('seed/recursive identity mismatch')
    source_root = Path(__file__).resolve().parent
    lineage = {}
    for name, old in old_codes.items():
        own = source_root/name
        rec.admit(own, old['sha256'])
        lineage[name] = {'historical': old, 'owned': rec.pins[str(own)]}
    rec.inherited = {'historical_source_map': old_codes, 'owned_source_lineage': lineage,
        'K10_repeated': False, 'B11_preparation_repeated': False,
        'historical_B11_pass': False if rec.mode == 'b11-diagnosis' else 'not consumed'}
    rec.event('lineage', 'historical_to_owned', rec.inherited)
    rec.checkpoint()
    return docs

def schema(x, shape, dtype, finite=True):
    if not isinstance(x, np.ndarray) or x.shape != shape or x.dtype != np.dtype(dtype) or x.dtype.hasobject:
        raise ValueError('unexpected shape/dtype')
    if finite and not np.all(np.isfinite(x)): raise ValueError('nonfinite value')

def cache_ref(item, key):
    return {'path': item['path'], 'sha256': item['sha256'], 'array_key': key, 'origin': 'inherited'}

def bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))

def diagnosis(rec, docs):
    fields = docs['fields']['output_bindings']['arrays']; old = docs['b11']['output_bindings']['arrays']
    names = ('F1_P_radial_Nv','F1_H_radial_Nv','F2_P_radial_Nv','F2_H_radial_Nv')
    with archive(rec, fields) as z:
        rows = {'radial_sites': z['radial_sites'][:17]}
        rows.update({n: z[n][:17] for n in names})
    rec.save('radial_rows.npz', rows, [cache_ref(fields,n) for n in ('radial_sites',)+names])
    schema(rows['radial_sites'], (17,), '<f8')
    for name in names: schema(rows[name], (17,2), '<f8')
    ok = np.array_equal(rows['radial_sites'], np.arange(1.,18.))
    rec.results['radial_rows'] = clean(rows)
    rec.decide('D.radial_schema', ok, clean(rows['radial_sites']), list(range(1,18)),[rec.saved['radial_rows.npz']])
    with archive(rec, old) as z:
        for prefix in PREFIXES:
            boundary(); source = {n:z[prefix+'_'+n] for n in PRIMARY}
            rec.event('cached_operands',prefix,{'keys':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in source.items()}},[cache_ref(old,prefix+'_'+key) for key in PRIMARY])
            n = 18 if prefix.startswith('F1') else 30
            schema(source['energies'], (n,), '<f8'); schema(source['selected_indices'], (6,), '<i8')
            indices = source['selected_indices']
            if np.any(indices < 0) or np.any(indices >= n) or len(set(map(int,indices))) != 6: raise ValueError('invalid selected index')
            primary = {**source, 'selected_own_energies': source['energies'][indices]}
            rec.save(prefix+'_primary.npz', primary, [cache_ref(old,prefix+'_'+key) for key in PRIMARY])
            for name in ('source_input_energy','energy_difference'): schema(source[name], (6,), '<f8', name!='energy_difference')
            for name in ('source_column_map','energy_multiset_representative_permutation'): schema(source[name], (6,), '<i8')
            schema(source['individual_comparison_available'], (6,), '|b1')
            schema(source['source_energy_feasible_mask'], (6,6), '|b1')
            schema(source['selected_cluster_membership'], (6,n), '|b1')
            f1 = prefix.startswith('F1'); rep = source['energy_multiset_representative_permutation']
            schema(source['source_energy_feasible_permutations'], (1 if f1 else 0,6), '<i8')
            own_status = docs['b11']['families'][prefix[:2]]['cases'][prefix[-1]]
            case_ref={'path':str(BASE/SEEDS['b11'][0]),'sha256':SEEDS['b11'][1],'json_pointer':'/families/'+prefix[:2]+'/cases/'+prefix[-1]}
            rec.event('cached_case_status',prefix,own_status,[case_ref])
            if f1:
                ok = (np.array_equal(rep, source['source_column_map']) and sorted(map(int,rep)) == list(range(6))
                      and np.array_equal(source['source_energy_feasible_permutations'][0], rep)
                      and bool(np.all(source['individual_comparison_available'])) and bool(np.all(np.isfinite(source['energy_difference'])))
                      and bool(np.all(source['source_energy_feasible_mask'][np.arange(6),rep])))
            else:
                ok = (bool(np.all(rep == -1)) and bool(np.all(source['source_column_map'] == -1))
                      and not bool(np.any(source['individual_comparison_available']))
                      and not bool(np.any(source['source_energy_feasible_mask'])) and bool(np.all(np.isnan(source['energy_difference']))))
            ok = ok and all(np.array_equal(np.flatnonzero(row),[indices[i]]) for i,row in enumerate(source['selected_cluster_membership']))
            ok = ok and own_status['all_passed'] is f1
            rec.results[prefix] = {'primary': clean(primary), 'historical_case': own_status,
                                   'energy_gate': 1e-8, 'cluster_threshold': 1e-10}
            rec.decide('D.'+prefix+'.primary_schema', ok, {'relationships_preserved':ok}, {'relationships_preserved':True},[rec.saved[prefix+'_primary.npz'],case_ref])
            def difference():
                delta = np.empty((6,6),dtype=np.float64); copied = np.zeros((6,6),dtype=bool)
                for i in range(6):
                    for j in range(6):
                        if f1 and j == int(rep[i]):
                            delta[i,j] = source['energy_difference'][i]; copied[i,j] = True
                        else: delta[i,j] = primary['selected_own_energies'][i]-source['source_input_energy'][j]
                return {'signed_selected_own_minus_source':delta, 'inherited_difference_entries':copied}
            new = rec.unit(prefix+'_delta.npz', difference, [rec.saved[prefix+'_primary.npz']])
            mask = new['inherited_difference_entries']; delta = new['signed_selected_own_minus_source']
            finite = bool(np.all(np.isfinite(delta)))
            exact_copy = bitwise(delta[np.arange(6),rep],source['energy_difference']) if f1 else True
            ok = finite and int(mask.sum()) == (6 if f1 else 0) and exact_copy
            rec.results[prefix]['delta'] = clean(new)
            rec.decide('D.'+prefix+'.delta_finite',ok, {'finite':finite, 'inherited_entries':int(mask.sum()),'exact_copy':exact_copy},
                       {'finite':True,'inherited_entries':6 if f1 else 0,'exact_copy':True},[rec.saved[prefix+'_delta.npz'],rec.saved[prefix+'_primary.npz']])
    rec.results['new_signed_subtractions'] = 198; rec.results['reused_signed_values'] = 18
    rec.results['inherited_B11_gate_changed'] = False

def logical_identity(x):
    y = np.ascontiguousarray(x)
    return {'shape':list(y.shape),'dtype':y.dtype.str,'sha256':hashlib.sha256(memoryview(y).cast('B')).hexdigest()}

def restricted_h(points, lapse, flow):
    from scipy.sparse import coo_matrix
    from lead_native import spin_matrices
    alpha, _, _ = spin_matrices(); count = len(points)
    lookup = {tuple(map(int,p)):i for i,p in enumerate(points)}
    rows, cols, values = [], [], []
    for i,p in enumerate(points):
        here = tuple(map(int,p+40))
        for axis in range(3):
            for distance in (1,2):
                target = p.copy(); target[axis] += distance
                j = lookup.get(tuple(map(int,target)))
                if j is None: continue
                there = tuple(map(int,target+40))
                if distance == 1:
                    coefficient = -1j*(lapse[here]+lapse[there])/4
                    for a in range(4):
                        for b in range(4):
                            value = coefficient*alpha[axis,a,b]
                            if value != 0:
                                r,c = a*count+i,b*count+j
                                rows.extend((r,c)); cols.extend((c,r)); values.extend((value,value.conjugate()))
                else:
                    value = -1j*(flow[(axis,)+here]+flow[(axis,)+there])/8
                    if value != 0:
                        for a in range(4):
                            r,c = a*count+i,a*count+j
                            rows.extend((r,c)); cols.extend((c,r)); values.extend((value,value.conjugate()))
    h = coo_matrix((np.asarray(values,dtype=np.complex128),(rows,cols)),shape=(4*count,4*count)).tocsr()
    h.sum_duplicates(); h.sort_indices()
    return h

def projected(h, left, right):
    result = np.empty((left.shape[1],right.shape[1]),dtype=np.complex128)
    for start in range(0,right.shape[1],128):
        stop = min(start+128,right.shape[1])
        applied = h@right[:,start:stop]
        result[:,start:stop] = (left.T@applied.conj()).conj()
    return result

def block_control(rec, c, block, counterpart, kind, actual_block):
    raw_name = f'c{c}_{block}_check.npz'
    raw = rec.unit(raw_name, lambda:{'residual':actual_block-counterpart.conj().T},
                   [rec.saved[f'c{c}_{block}.npz'],rec.saved[f'c{c}_{kind}.npz']])['residual']
    schema(raw,actual_block.shape,'<c16')
    before = stamp(); norm = float(np.linalg.norm(raw,'fro')); scale = float(np.linalg.norm(actual_block,'fro'))
    maximum = float(np.max(np.abs(raw))) if raw.size else 0.
    threshold = 5e-12*max(1.,scale)
    norm_receipt=rec.event('raw_scalar_receipt',f'c{c}_{block}_norms',{'residual_Frobenius':norm,'block_Frobenius':scale,'maximum_entry':maximum,'threshold':threshold,'timing':elapsed(before)},[rec.saved[raw_name],rec.saved[f'c{c}_{block}.npz']])
    tag = {'G_NO':'G_cross','G_NN':'G_new','P_NO':'P_cross','P_NN':'P_new'}[block]
    rec.decide(f'K.c{c}.{tag}',norm<=threshold,{'residual_Frobenius':norm,'maximum_entry':maximum}, {'maximum_Frobenius':threshold},[norm_receipt,rec.saved[raw_name],rec.saved[f'c{c}_{block}.npz']])

def resources(rec, phase):
    observation={}
    def metric(name,fn):
        try:
            value=fn();observation[name]=value
            rec.event('resource_observation',phase+'.'+name,value)
            return value
        except BaseException as exc:
            failed={'phase':phase,'metric':name,'type':type(exc).__name__,'message':str(exc),'available_observations':observation}
            rec.event('failed_resource_observation',phase+'.'+name,failed)
            exc.observation=failed;raise
    task_count=metric('resident_threads',lambda:len(list(Path('/proc/self/task').iterdir())))
    cap=1 if rec.mode=='b11-diagnosis' else 16
    if not 1<=task_count<=cap:raise ValueError('resident thread ceiling')
    loaded_numba=metric('Numba_loaded',lambda:any(n=='numba' or n.startswith('numba.') for n in sys.modules))
    if loaded_numba:raise ValueError('Numba is not admitted')
    mem=metric('memory',lambda:{line.split()[0].rstrip(':'):int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()})
    workspace=64*1024**2 if rec.mode=='b11-diagnosis' else 8*16*(4096*4992+4992**2)
    def filesystem(path):
        st=os.statvfs(path)
        return {'bavail':st.f_bavail,'frsize':st.f_frsize}
    disk=metric('bulk_statvfs',lambda:filesystem(rec.out))
    root=metric('root_statvfs',lambda:filesystem('/'))
    nice=metric('niceness',lambda:os.getpriority(os.PRIO_PROCESS,0))
    affinity=metric('affinity',lambda:sorted(os.sched_getaffinity(0)))
    metric('AS',lambda:list(resource.getrlimit(resource.RLIMIT_AS)))
    metric('CPU',lambda:list(resource.getrlimit(resource.RLIMIT_CPU)))
    metric('virtual_memory',lambda:{line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmPeak:','VmSize:','VmRSS:'))})
    env=metric('thread_environment',lambda:{k:os.environ.get(k) for k in THREADS})
    expected_threads='1' if rec.mode=='b11-diagnosis' else '8'
    if any(env[k]!=expected_threads for k in THREADS) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('runtime environment changed')
    if len(affinity)!=(1 if rec.mode=='b11-diagnosis' else 8):raise ValueError('runtime affinity changed')
    free=disk['bavail']*disk['frsize'];root_free=root['bavail']*root['frsize']
    required=(1 if rec.mode=='b11-diagnosis' else 16)*1024**3
    used=metric('own_current_file_bytes',lambda:sum(p.stat().st_size for p in rec.out.iterdir() if p.is_file()))
    rec.event('resource_limits',phase,{'workspace':workspace,'RAM_reserve':64*1024**3,'output_allowance':required,'output_margin':1024**3})
    if free<max(0,required-used)+1024**3 or root_free<100*10**9 or mem['MemAvailable']-workspace<64*1024**3:
        raise ValueError('disk/RAM reserve guard')
    if nice<10:raise ValueError('niceness guard')


def increment(rec,docs):
    from lead_channels import frame_labels, sampled_frame
    for name in ('lead_channels','lead_native'):
        if Path(sys.modules[name].__file__).resolve() != Path(__file__).resolve().parent/(name+'.py'):
            raise ValueError('unexpected mathematical module path')
    labels10,labels12 = frame_labels(10),frame_labels(12)
    jo = np.concatenate([624*v+np.arange(440,dtype=np.int64) for v in range(8)])
    jn = np.concatenate([624*v+np.arange(440,624,dtype=np.int64) for v in range(8)])
    context = {'labels10':labels10,'labels12':labels12,'old_to_new':jo,'new_indices':jn}
    rec.save('label_context.npz',context)
    ok = (labels10.shape==(3520,7) and labels12.shape==(4992,7) and np.array_equal(labels12[jo],labels10)
          and len(set(map(tuple,labels12)))==4992 and np.array_equal(np.sort(np.r_[jo,jn]),np.arange(4992)))
    rec.decide('K.label_join',ok,{'valid_join':ok,'old':len(jo),'new':len(jn)}, {'valid_join':True,'old':3520,'new':1472},[rec.saved['label_context.npz']])
    native_item=docs['native']['output_bindings']['arrays']; fields_item=docs['fields']['output_bindings']['arrays']
    with archive(rec,native_item) as z: lapse,flow=z['native_N'],z['native_flow']
    schema(lapse,(81,81,81),'<f8');schema(flow,(3,81,81,81),'<f8')
    with archive(rec,fields_item) as z: radial=z['F1_P_radial_Nv']
    association=docs['k10']['field_association']
    for key,value in [('native_N',lapse),('native_v',flow),('radial_Nv',radial)]:
        actual=logical_identity(value)
        identity_refs=[cache_ref(native_item,'native_N' if key=='native_N' else 'native_flow')] if key!='radial_Nv' else [cache_ref(fields_item,'F1_P_radial_Nv')]
        identity_refs.append({'path':str(BASE/SEEDS['k10'][0]),'sha256':SEEDS['k10'][1],'json_pointer':'/field_association/'+key})
        rec.event('logical_input_identity',key,{'actual':actual,'expected':association[key]},identity_refs)
        if actual!=association[key]:raise ValueError('K10 field association mismatch '+key)
    if association['mass']!=0. or association['spacing']!=1.:raise ValueError('K10 units/mass')
    tile_bindings=docs['setup10']['output_bindings']
    for c in range(4):
        boundary();resources(rec,f'before_cell_{c}')
        item=tile_bindings[f'setup_tile_c{c}.npz']
        with archive(rec,item) as z:
            if set(z.files)!={'points','global_dofs','A','G','projected_native_tile'}:raise ValueError('K10 tile keys')
            old={k:z[k] for k in z.files}
        points,dofs,ao,goo,poo=(old[k] for k in ('points','global_dofs','A','G','projected_native_tile'))
        schema(points,(1024,3),'<i8');schema(dofs,(4096,),'<i8')
        schema(ao,(4096,3520),'<c16');schema(goo,(3520,3520),'<c16');schema(poo,(3520,3520),'<c16')
        coordinates=(np.indices((81,81,81),dtype=np.int64)-40).reshape(3,-1).T
        r2=np.sum(coordinates*coordinates,axis=1);lower,upper=32+2*c,34+2*c
        chosen=np.flatnonzero((r2>lower*lower)&(r2<=upper*upper))
        expected_dofs=(np.arange(4,dtype=np.int64)[:,None]*81**3+chosen[:1024][None]).reshape(-1)
        ok=np.array_equal(points,coordinates[chosen[:1024]]) and np.array_equal(dofs,expected_dofs) and len(chosen)==(27452,30752,34280,38212)[c]
        rec.decide(f'K.c{c}.geometry',ok,{'exact_saved_geometry':ok,'shell_points':len(chosen)}, {'exact_saved_geometry':True,'shell_points':(27452,30752,34280,38212)[c]},[cache_ref(item,'points'),cache_ref(item,'global_dofs')])
        del coordinates,r2,chosen,expected_dofs
        boundary();rec.event('begin',f'c{c}_H.npz',{},[cache_ref(native_item,'native_N'),cache_ref(native_item,'native_flow'),cache_ref(item,'points'),cache_ref(item,'global_dofs')]);started=stamp();h=restricted_h(points,lapse,flow)
        rec.results['versions']['scipy']=sys.modules['scipy'].__version__
        rec.save(f'c{c}_H.npz',{'data':h.data,'indices':h.indices.astype(np.int64),'indptr':h.indptr.astype(np.int64),'shape':np.array(h.shape,dtype=np.int64)},[cache_ref(native_item,'native_N'),cache_ref(native_item,'native_flow'),cache_ref(item,'points'),cache_ref(item,'global_dofs')],elapsed(started))
        if not np.all(np.isfinite(h.data)):raise ValueError('nonfinite restricted H')
        defect=(h-h.getH()).tocoo();maximum=float(np.max(np.abs(defect.data))) if defect.nnz else 0.
        h_receipt=rec.event('raw_scalar_receipt',f'c{c}_H_defect',{'rows':defect.row,'columns':defect.col,'values':defect.data,'shape':h.shape,'maximum_entry':maximum},[rec.saved[f'c{c}_H.npz']])
        rec.decide(f'K.c{c}.H_Hermitian',maximum==0.,{'maximum_entry':maximum},{'maximum_entry':0.},[h_receipt,rec.saved[f'c{c}_H.npz']])
        an=rec.unit(f'c{c}_A_N.npz',lambda:{'A_N':sampled_frame(points,12,columns=jn)},[rec.saved['label_context.npz'],cache_ref(item,'points')])['A_N']
        schema(an,(4096,1472),'<c16')
        blocks={}
        for tag,left,right in [('G_ON',ao,an),('G_NO',an,ao)]:
            blocks[tag]=rec.unit(f'c{c}_{tag}.npz',lambda left=left,right=right,tag=tag:{tag:left.conj().T@right},[rec.saved[f'c{c}_A_N.npz'],cache_ref(item,'A')])[tag]
            schema(blocks[tag],(left.shape[1],right.shape[1]),'<c16')
        block_control(rec,c,'G_NO',blocks['G_ON'],'G_ON',blocks['G_NO'])
        blocks['G_NN']=rec.unit(f'c{c}_G_NN.npz',lambda:{'G_NN':an.conj().T@an},[rec.saved[f'c{c}_A_N.npz']])['G_NN']
        schema(blocks['G_NN'],(1472,1472),'<c16');block_control(rec,c,'G_NN',blocks['G_NN'],'G_NN',blocks['G_NN'])
        for tag,left,right in [('P_ON',ao,an),('P_NO',an,ao)]:
            blocks[tag]=rec.unit(f'c{c}_{tag}.npz',lambda left=left,right=right,tag=tag:{tag:projected(h,left,right)},[rec.saved[f'c{c}_H.npz'],rec.saved[f'c{c}_A_N.npz'],cache_ref(item,'A')])[tag]
            schema(blocks[tag],(left.shape[1],right.shape[1]),'<c16')
        block_control(rec,c,'P_NO',blocks['P_ON'],'P_ON',blocks['P_NO'])
        blocks['P_NN']=rec.unit(f'c{c}_P_NN.npz',lambda:{'P_NN':projected(h,an,an)},[rec.saved[f'c{c}_H.npz'],rec.saved[f'c{c}_A_N.npz']])['P_NN']
        schema(blocks['P_NN'],(1472,1472),'<c16');block_control(rec,c,'P_NN',blocks['P_NN'],'P_NN',blocks['P_NN'])
        def assemble():
            a=np.empty((4096,4992),dtype=np.complex128);a[:,jo]=ao;a[:,jn]=an
            g=np.empty((4992,4992),dtype=np.complex128);p=np.empty_like(g)
            for target,oo,prefix in ((g,goo,'G'),(p,poo,'P')):
                target[np.ix_(jo,jo)]=oo
                for left,right,tag in ((jo,jn,'ON'),(jn,jo,'NO'),(jn,jn,'NN')):target[np.ix_(left,right)]=blocks[prefix+'_'+tag]
            return {'points':points,'global_dofs':dofs,'A':a,'G':g,'projected_native_tile':p}
        full=rec.unit(f'setup_tile_c{c}.npz',assemble,[rec.saved[f'c{c}_{tag}.npz'] for tag in ('A_N','G_ON','G_NO','G_NN','P_ON','P_NO','P_NN')]+[rec.saved['label_context.npz']]+[cache_ref(item,k) for k in ('points','global_dofs','A','G','projected_native_tile')])
        for key,source,view,decision in [('A',ao,full['A'][:,jo],'copy_A'),('G',goo,full['G'][np.ix_(jo,jo)],'copy_G'),('projected_native_tile',poo,full['projected_native_tile'][np.ix_(jo,jo)],'copy_P')]:
            schema(full[key],(4096,4992) if key=='A' else (4992,4992),'<c16')
            ok=bitwise(view,source)
            rec.decide(f'K.c{c}.{decision}',ok,{'bitwise_copy':ok},{'bitwise_copy':True},[cache_ref(item,key),rec.saved[f'setup_tile_c{c}.npz'],rec.saved['label_context.npz']])
        rec.results[f'cell_{c}']={'incremental_only':True,'old_columns_copied':3520,'new_columns':1472,'full_width':4992,'completed':True}
        rec.checkpoint();del old,ao,goo,poo,an,blocks,full,h
        resources(rec,f'after_cell_{c}')
    rec.results['fresh_full_K12_benchmark_performed']=False
    rec.results['full_frame_or_propagation_performed']=False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('b11-diagnosis','k12-increment'));p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();rec=None
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU):signal.signal(sig,handler)
    launch=float(os.environ['BPRIME_CHILD_LAUNCH']);handled=110 if args.mode=='b11-diagnosis' else 390
    signal.setitimer(signal.ITIMER_REAL,max(.001,launch+handled-time.monotonic()))
    try:
        if 'compute-host' not in __import__('subprocess').check_output(['hostname','-I'],text=True,timeout=2).split():raise ValueError('box only')
        if Path(sys.executable).resolve()!=PYTHON.resolve():raise ValueError('qi2 executable required')
        if args.outdir.resolve().parent!=ROOT.resolve():raise ValueError('owned output root only')
        if not args.outdir.is_dir() or any(q.name not in ('stdout.log','stderr.log','execution.json') for q in args.outdir.iterdir()):raise ValueError('wrapper must supply a fresh owned directory')
        ownership=json.loads((args.outdir/'execution.json').read_text())
        if ownership.get('child_pid')!=os.getpid() or ownership.get('wrapper_pid')!=os.getppid() or ownership.get('terminal_established') is not False:raise ValueError('wrapper ownership checkpoint required')
        rec=Records(args.mode,args.outdir)
        expected=json.loads(os.environ['BPRIME_REVIEWED_SOURCE_PINS'])
        if set(expected)!={'producer','wrapper','contract','SPEC','DERIVATION'}:raise ValueError('source pin roles')
        for role,item in expected.items():
            pin,_=pin_read(Path(item['path']),item['sha256'])
            observation={'role':role,'expected':item,'returned':pin}
            rec.event('owned_source_read_returned',role,observation,[item])
            if pin!=item:
                exc=ValueError('wrapper/child source interval differs');exc.observation=observation;raise exc
            rec.pins[item['path']]=pin;rec.event('owned_source_pin',role,pin)
        if expected['producer']['path']!=str(Path(__file__).resolve()) or expected['contract']['sha256']!=FROZEN:raise ValueError('source/contract identity')
        threads='1' if args.mode=='b11-diagnosis' else '8'
        if any(os.environ.get(k)!=threads for k in THREADS) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('thread/CUDA environment')
        limit=2 if args.mode=='b11-diagnosis' else 16
        cpulimit=[90,100] if args.mode=='b11-diagnosis' else [5900,6000]
        if list(resource.getrlimit(resource.RLIMIT_AS))!=[limit*1024**3]*2 or list(resource.getrlimit(resource.RLIMIT_CPU))!=cpulimit:raise ValueError('resource limits')
        if len(os.sched_getaffinity(0))!=(1 if args.mode=='b11-diagnosis' else 8):raise ValueError('CPU affinity')
        resources(rec,'before_numeric_import')
        docs=inherited_admission(rec)
        global np
        import numpy as np
        resources(rec,'after_numpy_import')
        rec.results['versions']={'python':sys.version,'numpy':np.__version__}
        if args.mode=='b11-diagnosis':diagnosis(rec,docs)
        else:increment(rec,docs)
    except Withheld as exc:
        if rec is not None:rec.withheld=str(exc)
    except BaseException as exc:
        if rec is None:raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'observation':getattr(exc,'observation',None)}
        rec.errors.append({'phase':'main',**rec.error})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            # The fixed postcheck plan survives any partial final metadata pass.
            rec.postcheck_plan=list(rec.pins)
            rec.checkpoint()
            for old in list(rec.pins.values()):
                started=stamp();receipt={'expected':old,'matched':False}
                try:
                    pin,_=pin_read(Path(old['path']),old['sha256'],honor_stop=False)
                    receipt['returned']=pin
                    rec.event('final_input_read_returned',old['path'],{'pin':pin,'timing':elapsed(started)})
                    if pin!=old:raise ValueError('final input/source interval changed')
                    receipt['matched']=True
                except BaseException as exc:
                    issue={'phase':'final_input_check','path':old['path'],'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                    receipt['failure']=issue
                    rec.errors.append(issue)
                    if rec.error is None:rec.error=issue
                receipt['timing']=elapsed(started)
                rec.postchecks[old['path']]=receipt
                rec.event('final_input_check',old['path'],receipt)
                rec.checkpoint()
            rec.terminal=True;rec.checkpoint();rec.journal.close()
    return 0 if rec is not None and rec.error is None and not rec.errors and len(rec.saved)==len(rec.files) and len(rec.decisions)==len(rec.checks) else 2

if __name__=='__main__':sys.exit(main())
