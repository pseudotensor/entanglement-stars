#!/usr/bin/env python3
"""Task 99 only. New finite filtered targets; old products are admitted read-only."""
import hashlib
import itertools
import json
import math
import os
import resource
from pathlib import Path
import signal
import sys
import time
import traceback
import sympy as sp

ROOT = Path('<internal>/vertex_gate_filtered_v1')
OUT = ROOT / 'run_v1'
OLD = Path('<internal>/vertex_gate_v1')
PREDECESSORS = [
    (OLD/'spinor_algebra.json', 'af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb'),
    (OLD/'continuation_v1/on_site_completion.json', '77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d'),
    (OLD/'continuation_v1/amplitudes.json', 'af9c99e96e10da536cdecf779e6f4f5e01a6754c1e8d3e1997aea1110a5604df'),
    (OLD/'continuation_v1/vertex_gate_missing_result.json', '52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d'),
    (OLD/'continuation_v1/execution.json', 'be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9'),
]
R = sp.Rational
BITS = list(itertools.product(range(2), repeat=3))
OFFSETS = list(itertools.product(range(-1, 2), repeat=3))
FRAW = ('one_dimensional_stencils valley_labels stencil_offsets filter_stencil corner_response '
        'partition_stencil inversion_residual rotation_z_residual rotation_cycle_residual '
        'factor_polynomial corner_quartets conserving_corner_counts opposite_cone_corner_action support').split()
ARAW = ('kinematics phase_incoming phase_same phase_cross kinematic_residuals threshold_weights threshold_same_tensor threshold_cross_tensor '
        'mixed_rest_incoming_products finite_filter_weights filter_factor_same filter_factor_cross heavy_spatial_current_1 heavy_spatial_current_2 '
        'heavy_spatial_current_3 wedge_H wedge_L swapped_wedge_H swapped_wedge_L jet_H_plus jet_H_minus '
        'jet_eigen_plus jet_eigen_minus jet_norm_plus jet_norm_minus jet_JH_0 jet_JH_1 jet_JH_2 jet_JH_3 '
        'jet_same_tensor jet_cross_tensor finite_spin_boost').split()
for b in ('000', '001'):
    ARAW += [f'filtered_JH_{b}_{m}' for m in range(4)]
    for case in ('same', 'cross'):
        ARAW += [f'filtered_JL_{case}_{b}_{m}' for m in range(4)]
    ARAW += [f'filtered_wedge_H_{b}']
    ARAW += [f'filtered_wedge_L_{case}_{b}' for case in ('same', 'cross')]
for case in ('same', 'cross'):
    ARAW += [f'{base}_{case}' for base in ('tensor_current', 'tensor_wedge', 'reverse', 'spin_average')]
CRAW = ('relative_indices one_axis_L0 one_axis_L1 one_axis_L2 L0 L1 L2 S0 G10 G01 S1 G20 G11 G02 S2 '
        'evaluation l0 l1 l2 r t u spin_factor a2_constant a2_h c1_constant c1_h c1tilde_constant c1tilde_h '
        'current_slope_gram current_slope_direct kernel_moment_scope product_state_scope').split()
DECISIONS = {
 'F': ['F.'+s for s in ('corner_identity partition_identity inversion rotation_z rotation_cycle factor_polynomial '
                       'corner_selector conserving_counts opposite_cone_removed finite_support').split()],
 'A': ['A.'+s for s in ('kinematics threshold_weights threshold_same threshold_cross mixed_rest_removed '
                       'filter_same filter_cross swap_H swap_L same_routes same_tensor same_reverse same_average '
                       'cross_routes cross_tensor cross_reverse cross_average jet_eigen_plus jet_eigen_minus '
                       'jet_norm_plus jet_norm_minus jet_same jet_cross finite_spin_boost').split()],
 'C': ['C.'+s for s in ('axis_L0 axis_L1 axis_L2 S0 S1 S2 l0 l1 current_routes current_slope '
                       'current_nonzero_coefficients').split()]}
RAW = {'F': FRAW, 'A': ARAW, 'C': CRAW}
JOURNALS = {'F': 'filters.jsonl', 'A': 'amplitudes.jsonl', 'C': 'current.jsonl', 'admission': 'admission.jsonl'}
STOP = []

def handle(sig, frame):
    STOP.append(int(sig))

class Stopped(Exception):
    pass

def boundary():
    if STOP:
        raise Stopped('handled signal '+str(STOP))

def integer(x):
    if type(x) is not int:
        raise ValueError('integer field is not an integer')
    return x

def checked_fraction(pair):
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError('rational pair required')
    n, d = map(integer, pair)
    if d <= 0 or math.gcd(n, d) != 1:
        raise ValueError('noncanonical rational')
    return n,d

def fraction(pair):
    return R(*checked_fraction(pair))

def finite_metadata_float(token):
    value=float(token)
    if not math.isfinite(value):
        raise ValueError("nonfinite metadata float")
    return value

def no_duplicates(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError('duplicate JSON key '+k)
        result[k] = v
    return result

def reject_constant(x):
    raise ValueError('nonfinite JSON token '+x)

def identity(st):
    return {k: getattr(st, 'st_'+k) for k in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}

def read_pinned(path, expected):
    resolved = str(path.resolve(strict=True))
    with path.open('rb') as f:
        before = identity(os.fstat(f.fileno()))
        data = f.read()
        digest = hashlib.sha256(data).hexdigest()
        after = identity(os.fstat(f.fileno()))
        path_now = identity(path.stat())
    if before != after or before != path_now or str(path.resolve()) != resolved or digest != expected:
        exc=ValueError('changed input '+str(path))
        exc.observation={'path':str(path),'resolved_path':resolved,'sha256':digest,'before':before,'after':after,'path_now':path_now,'expected_sha256':expected}
        raise exc
    return data, {'path': str(path), 'resolved_path': resolved, 'sha256': digest, 'bytes': len(data), **before}

def old_scalar(v):
    if not isinstance(v, dict):
        if type(v) is int:
            return sp.Integer(v)
        raise ValueError('old scalar dictionary required')
    if 'gaussian_rational' in v:
        g = v['gaussian_rational']
        return fraction(g['real']) + sp.I*fraction(g['imag'])
    white = {
      'Mul(Integer(-1), Rational(1, 2), Pow(Integer(2), Rational(1, 2)))': -sp.sqrt(2)/2,
      'Mul(Integer(-1), Rational(1, 5), Pow(Integer(5), Rational(1, 2)))': -sp.sqrt(5)/5,
      'Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2)))': sp.sqrt(2)/2,
      'Mul(Rational(1, 5), Pow(Integer(5), Rational(1, 2)))': sp.sqrt(5)/5,
      'Mul(Rational(2, 5), Pow(Integer(5), Rational(1, 2)))': 2*sp.sqrt(5)/5,
    }
    if v.get('srepr') not in white:
        raise ValueError('unapproved algebraic scalar')
    return white[v['srepr']]

def old_nonzero_structure(v):
    if not isinstance(v,dict):
        if type(v) is int:
            return v!=0
        raise ValueError('scalar schema')
    if 'gaussian_rational' in v:
        real=checked_fraction(v['gaussian_rational']['real'])
        imag=checked_fraction(v['gaussian_rational']['imag'])
        return real[0]!=0 or imag[0]!=0
    allowed={
      'Mul(Integer(-1), Rational(1, 2), Pow(Integer(2), Rational(1, 2)))',
      'Mul(Integer(-1), Rational(1, 5), Pow(Integer(5), Rational(1, 2)))',
      'Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2)))',
      'Mul(Rational(1, 5), Pow(Integer(5), Rational(1, 2)))',
      'Mul(Rational(2, 5), Pow(Integer(5), Rational(1, 2)))'}
    if v.get('srepr') not in allowed:
        raise ValueError('unapproved scalar representation')
    return True

def sparse(v, shape, select=None):
    if not isinstance(v, dict) or v.get('shape') != list(shape):
        raise ValueError('matrix dimension mismatch')
    if any(type(x) is not int for x in v['shape']) or type(v.get('entries')) is not list:
        raise ValueError('matrix schema')
    prev = (-1, -1)
    values = {}
    for entry in v['entries']:
        if type(entry) is not list or len(entry) != 3:
            raise ValueError('sparse entry schema')
        i, j = integer(entry[0]), integer(entry[1])
        if not (0 <= i < shape[0] and 0 <= j < shape[1]) or (i, j) <= prev:
            raise ValueError('sparse coordinate ordering/range')
        prev = (i, j)
        # Validate all scalar entries, but instantiate no unused full matrix.
        if not old_nonzero_structure(entry[2]):
            raise ValueError('explicit zero sparse entry')
        if select is None or (i, j) == select:
            values[i, j] = old_scalar(entry[2])
    if select is not None:
        if select not in values:
            raise ValueError('missing unique selected scalar')
        return values[select]
    return sp.SparseMatrix(*shape, values)

def scalar_tree(v):
    v = sp.sympify(v)
    if v.is_Rational:
        return {'rational': [int(v.p), int(v.q)]}
    if v == sp.I:
        return {'imaginary_unit': True}
    if v.is_Add:
        return {'add': [scalar_tree(x) for x in v.args]}
    if v.is_Mul:
        return {'multiply': [scalar_tree(x) for x in v.args]}
    if v.is_Pow and v.exp.is_Integer and v.exp >= 0:
        if v.exp == 0:
            return {'rational':[1,1]}
        return {'multiply':[scalar_tree(v.base) for _ in range(int(v.exp))]}
    if v.is_Pow and v.exp == R(1, 2) and v.base.is_Integer and v.base > 0:
        n = int(v.base)
        residue = n
        for p in (2, 3, 5, 7):
            while residue % p == 0:
                residue //= p
        if residue == 1:
            return {'positive_integer_square_root': n}
    raise ValueError('unsupported exact scalar '+str(v))

def encoded(v):
    if isinstance(v, sp.MatrixBase):
        return {'shape': list(v.shape), 'entries': [[i, j, scalar_tree(v[i,j])]
                for i in range(v.rows) for j in range(v.cols) if v[i,j] != 0]}
    if isinstance(v, sp.Basic):
        return scalar_tree(v)
    if isinstance(v, dict):
        return {str(k): encoded(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encoded(x) for x in v]
    if v is None or type(v) in (int, str, bool):
        return v
    raise ValueError('unsupported payload '+str(type(v)))

def shapeof(v):
    if isinstance(v, sp.MatrixBase):
        return list(v.shape)
    if isinstance(v, (list, tuple)):
        return ['list', len(v)]
    return type(v).__name__

def atomic(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

class Record:
    def __init__(self):
        self.seq = 0; self.ids = {}; self.values = {}; self.done = {}; self.pins = []
        self.streams = {k: (OUT/v).open('x') for k,v in JOURNALS.items()}
        self.error = None; self.terminal=False
        self.put('admission', 'stage_state', 'plan', {'raw': RAW, 'decisions': DECISIONS}, [])
        self.checkpoint()
    def put(self, stage, kind, name, value, operands):
        key = stage+':'+name
        if key in self.ids:
            raise ValueError('duplicate record '+key)
        # Preserve a returned object before honoring a handled stop.
        self.seq += 1
        refs = [self.ids[x] for x in operands]
        payload = {'sequence': self.seq, 'type': kind, 'name': name,
                   'shape_type': shapeof(value), 'payload': encoded(value), 'operand_ids': refs}
        f = self.streams[stage]
        f.write(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
        self.ids[key] = self.seq; self.values[key] = value
        if kind == 'comparison_decision':
            self.done[name] = value
        self.checkpoint()
        return value
    def checkpoint(self):
        stages = {}
        for s in RAW:
            missing_raw = [n for n in RAW[s] if s+':'+n not in self.ids]
            missing_dec = [n for n in DECISIONS[s] if n not in self.done]
            stages[s] = {'missing_raw_names': missing_raw, 'missing_decision_names': missing_dec,
                         'raw_complete': not missing_raw, 'validation_complete': not missing_dec,
                         'all_equal': not missing_dec and all(self.done[n] for n in DECISIONS[s])}
        complete=all(x['raw_complete'] and x['validation_complete'] for x in stages.values())
        equal=complete and all(x['all_equal'] for x in stages.values())
        status=('completed' if equal else 'completed_with_scientific_failures') if complete else ('incomplete' if self.terminal else 'running')
        atomic(OUT/'vertex_gate_filtered_result.json', {
          'target_complete':complete,
          'stages': stages, 'status':status,
          'error': self.error, 'handled_signals': STOP, 'raw_plan': RAW, 'decision_plan': DECISIONS,
          'saved_objects': self.ids, 'completed_decisions': self.done, 'input_source_pins': self.pins,
          'inherited_scope': {'A': '74 accepted decisions', 'B': '80 accepted decisions',
                              'D': 'incomplete;430 missing decisions preserved'},
          'python_version': sys.version, 'sympy_version': sp.__version__,
          'physical_verdicts': 'not inferred from validation completion; scoped by paired delivery'})
    def raw(self, stage, name, fn, deps=()):
        boundary()
        v = fn()
        self.put(stage, 'raw', name, v, list(deps))
        boundary()
        return v
    def compare(self, stage, name, actual_fn, expected_fn, deps=()):
        boundary()
        actual=actual_fn()
        self.put(stage, 'comparison_operands', name+'.actual', actual, list(deps)); boundary()
        expected=expected_fn()
        self.put(stage, 'comparison_operands', name+'.expected', expected, []); boundary()
        refs = [stage+':'+name+'.actual', stage+':'+name+'.expected']
        if isinstance(actual, sp.MatrixBase):
            residual = (actual-expected).applyfunc(sp.simplify)
        elif isinstance(actual, (list, tuple)):
            if len(actual) != len(expected):
                raise ValueError('comparison list dimension')
            residual = [bool(a) != bool(b) if type(a) is bool and type(b) is bool else sp.simplify(a-b)
                        for a,b in zip(actual, expected)]
        else:
            residual = sp.simplify(actual-expected)
        self.put(stage, 'comparison_residual', name+'.residual', residual, refs); boundary()
        if isinstance(residual, sp.MatrixBase):
            decision = all(x == 0 for x in residual)
        elif isinstance(residual, list):
            decision = all(x is False if type(x) is bool else x == 0 for x in residual)
        else:
            decision = residual == 0
        self.put(stage, 'comparison_decision', name, bool(decision), [stage+':'+name+'.residual'])
        # put/checkpoint installs final completion even if a late signal is pending.
        if not all(n in self.done for ds in DECISIONS.values() for n in ds):
            boundary()

def admit(rec):
    reviewed_list=json.loads(os.environ['VERTEX_FILTERED_REVIEWED_PINS'],object_pairs_hook=no_duplicates,parse_constant=reject_constant,parse_float=finite_metadata_float)
    reviewed={pin['path']:pin for pin in reviewed_list}
    fixed_inputs={
      'ReviewB_vgate_v1.md':'4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b',
      'ReviewA_pc14_v1.md':'918430260445b6829bc5ed237a21507f6c7aba754a0162c92d4785ea0cce82ac',
      'ReviewB_pc14_v1.md':'e4d950ac3443f4c867aa7f80128806a38ec8651dbc86c5b72812c1cb1fbd715a',
      '03_model.tex':'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb'}
    required=[ROOT/'inputs'/n for n in (*fixed_inputs,'frozen_spec.tex')]
    required += [ROOT/'code'/n for n in ('vertex_gate_filtered_exact_v1.py','run_vertex_gate_filtered_exact_v1.py')]
    all_paths=[p for p,_ in PREDECESSORS]+required
    if len(reviewed_list)!=12 or len(reviewed)!=12 or set(reviewed)!=set(map(str,all_paths)):
        raise ValueError('reviewed twelve-path set')
    for name,digest in fixed_inputs.items():
        if reviewed[str(ROOT/'inputs'/name)]['sha256']!=digest:
            raise ValueError('fixed proposal pin mismatch')
    def across(pin):
        initial=reviewed[pin['path']]
        if any(pin[k]!=initial[k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')):
            raise ValueError('wrapper-child identity mismatch '+pin['path'])
    docs = []
    for idx, (path, digest) in enumerate(PREDECESSORS):
        boundary()
        data, pin = read_pinned(path, digest)
        rec.put('admission', 'stage_state', 'predecessor_'+str(idx), pin, [])
        rec.pins.append(pin); across(pin); boundary()
        doc = json.loads(data, object_pairs_hook=no_duplicates, parse_constant=reject_constant,parse_float=finite_metadata_float)
        docs.append(doc); boundary()
    a,b,d,old_result,old_execution = docs
    for doc, count in ((a,74),(b,80)):
        if doc.get('validation_complete') is not True or len(doc['comparisons']) != count:
            raise ValueError('inherited validation count')
        if len({x['name'] for x in doc['comparisons']}) != count or any(x.get('equal') is not True for x in doc['comparisons']):
            raise ValueError('inherited validation flags')
    if old_result.get('status') != 'incomplete' or old_result.get('target_complete') is not False or len(old_result['missing_equality_decisions']['amplitudes']) != 430:
        raise ValueError('old partial scope changed')
    # Bind current source/spec bytes to the reviewed wrapper admission.
    for path in required:
        boundary()
        data,pin=read_pinned(path,reviewed[str(path)]['sha256'])
        rec.put('admission','stage_state','staged_'+path.name,pin,[])
        rec.pins.append(pin); across(pin); boundary()
    cache = {}
    def keep(alias, index, pointer, val):
        path,digest = PREDECESSORS[index]
        rec.put('admission', 'raw', alias, {'value': val, 'predecessor_sha256': digest,
                'path': str(path), 'json_pointer': pointer}, ['admission:predecessor_'+str(index)])
        cache[alias] = val; boundary()
    for m in range(4):
        keep('M'+str(m),0,f'/constructed/matrices/pair/{m}',sparse(a['constructed']['matrices']['pair'][m],(4,4)))
    keep('alpha_z',0,'/constructed/matrices/alpha/2',sparse(a['constructed']['matrices']['alpha'][2],(4,4)))
    bc=b['constructed']; dc=d['constructed']
    keep('K',1,'/constructed/ordered_pair_definition/tensor',sparse(bc['ordered_pair_definition']['tensor'],(6,6)))
    pairs=bc['ordered_pair_definition']['pair_indices']
    if pairs != [[i,j] for i in range(4) for j in range(i+1,4)] or any(type(x) is not int for p in pairs for x in p):
        raise ValueError('pair basis')
    keep('pair_indices',1,'/constructed/ordered_pair_definition/pair_indices',pairs)
    keep('mixed_boost',1,'/constructed/mixed_boost_action',sparse(bc['mixed_boost_action'],(6,1)))
    keep('common_boost_residual',1,'/constructed/common_boost_residual',sparse(bc['common_boost_residual'],(6,6)))
    if bc['basis'] != list(range(256)) or any(type(x) is not int for x in bc['basis']):
        raise ValueError('Fock basis identity')
    keep('spin_factor',1,'/constructed/W_squared/entries/(5,5)',sparse(bc['W_squared'],(256,256),(5,5)))
    keep('spin_basis',1,'/constructed/basis',{'basis':bc['basis'],'mask':5,'mode_order':['H0','H1','H2','H3','L0','L1','L2','L3']})
    for n in range(1,6):
        key=f'leg_{n}_spinors'; keep(key,2,'/constructed/'+key,sparse(dc[key],(4,2)))
        for suffix in ('inputs','cache_origin'):
            key=f'leg_{n}_{suffix}'; keep(key,2,'/constructed/'+key,dc[key])
        key=f'leg_{n}_energy'; keep(key,2,'/constructed/'+key,old_scalar(dc[key]))
    keep('leg_1_H',2,'/constructed/leg_1_H',sparse(dc['leg_1_H'],(4,4)))
    for key in ['collinear_JH_0']+[f'same_valley_threshold_J{a}_{m}' for a in ('H','L') for m in range(4)]:
        keep(key,2,'/constructed/'+key,sparse(dc[key],(2,2)))
    keep('threshold_tensor',2,'/constructed/same_valley_threshold_tensor',sparse(dc['same_valley_threshold_tensor'],(4,4)))
    rec.put('admission','stage_state','inherited_scope',{'A_count':74,'B_count':80,'old_D_missing_decisions':430,
              'old_D_residuals_are_not_new_decisions':True},[])
    return cache

def filters(rec, c):
    raw=lambda n,fn,deps=(): rec.raw('F',n,fn,deps)
    cmp=lambda n,a,e,deps=(): rec.compare('F','F.'+n,a,e,deps)
    t=raw('one_dimensional_stencils',lambda:sp.Matrix([[R(s,4),R(1,2),R(s,4)] for s in (1,-1)]))
    raw('valley_labels',lambda:sp.Matrix(BITS)); raw('stencil_offsets',lambda:sp.Matrix(OFFSETS))
    T=raw('filter_stencil',lambda:sp.Matrix([[sp.prod(t[b[i],n[i]+1] for i in range(3)) for n in OFFSETS] for b in BITS]),['F:one_dimensional_stencils','F:valley_labels','F:stencil_offsets'])
    response=raw('corner_response',lambda:sp.Matrix([[sum(T[j,k]*((-1)**(sum(b[i]*n[i] for i in range(3))%2)) for k,n in enumerate(OFFSETS)) for b in BITS] for j in range(8)]),['F:filter_stencil'])
    partition=raw('partition_stencil',lambda:sp.Matrix([sum(T[j,k] for j in range(8)) for k in range(27)]),['F:filter_stencil'])
    inv=raw('inversion_residual',lambda:sp.Matrix([[T[j,OFFSETS.index(tuple(-x for x in n))]-T[j,k] for k,n in enumerate(OFFSETS)] for j in range(8)]),['F:filter_stencil'])
    def rotation(rotate, perm):
        return sp.Matrix([[T[BITS.index(perm(b)),OFFSETS.index(rotate(n))]-T[j,k] for k,n in enumerate(OFFSETS)] for j,b in enumerate(BITS)])
    rz=raw('rotation_z_residual',lambda:rotation(lambda n:(-n[1],n[0],n[2]),lambda b:(b[1],b[0],b[2])),['F:filter_stencil'])
    rc=raw('rotation_cycle_residual',lambda:rotation(lambda n:(n[2],n[0],n[1]),lambda b:(b[2],b[0],b[1])),['F:filter_stencil'])
    # One-leg cosine constant and coefficient are t(0), 2*t(1).
    poly=raw('factor_polynomial',lambda:sp.Matrix([sum(sp.prod(2*t[s,2] if (mask>>j)&1 else t[s,1] for j in range(4)) for s in range(2)) for mask in range(16)]),['F:one_dimensional_stencils'])
    quartet=raw('corner_quartets',lambda:sp.Matrix([sum(poly[mask]*sp.prod((-1)**bits[j] for j in range(4) if (mask>>j)&1) for mask in range(16)) for bits in itertools.product(range(2),repeat=4)]),['F:factor_polynomial'])
    def corner_counts():
        even=[k for k,x in enumerate(itertools.product(range(2),repeat=4)) if sum(x)%2==0]
        total=len(even)**3
        surviving=sum(int(quartet[k]!=0) for k in even)**3
        return sp.Matrix([total,surviving,total-surviving])
    counts=raw('conserving_corner_counts',corner_counts,['F:corner_quartets'])
    action=raw('opposite_cone_corner_action',lambda:quartet[3]*c['mixed_boost'],['F:corner_quartets','admission:mixed_boost'])
    support=raw('support',lambda:sp.Matrix([[min(n[i] for k,n in enumerate(OFFSETS) if any(T[j,k]!=0 for j in range(8))),max(n[i] for k,n in enumerate(OFFSETS) if any(T[j,k]!=0 for j in range(8)))] for i in range(3)]),['F:filter_stencil'])
    cmp('corner_identity',lambda:(response),lambda:(sp.eye(8)),['F:corner_response'])
    cmp('partition_identity',lambda:(partition),lambda:(sp.Matrix([int(n==(0,0,0)) for n in OFFSETS])),['F:partition_stencil'])
    cmp('inversion',lambda:(inv),lambda:(sp.zeros(8,27)),['F:inversion_residual'])
    cmp('rotation_z',lambda:(rz),lambda:(sp.zeros(8,27)),['F:rotation_z_residual'])
    cmp('rotation_cycle',lambda:(rc),lambda:(sp.zeros(8,27)),['F:rotation_cycle_residual'])
    cmp('factor_polynomial',lambda:(poly),lambda:(sp.Matrix([R(1,8) if mask.bit_count()%2==0 else 0 for mask in range(16)])),['F:factor_polynomial'])
    cmp('corner_selector',lambda:(quartet),lambda:(sp.Matrix([int(k in (0,15)) for k in range(16)])),['F:corner_quartets'])
    cmp('conserving_counts',lambda:(counts),lambda:(sp.Matrix([512,8,504])),['F:conserving_corner_counts'])
    cmp('opposite_cone_removed',lambda:(action),lambda:(sp.zeros(6,1)),['F:opposite_cone_corner_action'])
    cmp('finite_support',lambda:(support),lambda:(sp.Matrix([[-1,1]]*3)),['F:support'])
    return t

def wedge(u,v,pairs):
    return sp.Matrix([[u[i,s]*v[j,t]-u[j,s]*v[i,t] for s in range(2) for t in range(2)] for i,j in pairs])

def amplitudes(rec,c):
    raw=lambda n,fn,deps=(): rec.raw('A',n,fn,deps)
    cmp=lambda n,a,e,deps=(): rec.compare('A','A.'+n,a,e,deps)
    kin=raw('kinematics',lambda:{'mass_H':R(3,10),'d_H':R(2,5),'E_H':R(1,2),'d_L':R(1,2),'E_L':R(1,2),
      'cos_H':sp.sqrt(21)/5,'cos_L':sp.sqrt(3)/2,'old_to_new_scale':R(1,2),'ordered_signed_legs':{
      'incoming':[[sp.sqrt(21)/5,R(2,5)],[sp.sqrt(21)/5,-R(2,5)]],
      'same':[[sp.sqrt(3)/2,R(1,2)],[sp.sqrt(3)/2,-R(1,2)]],
      'cross':[[-sp.sqrt(3)/2,R(1,2)],[-sp.sqrt(3)/2,-R(1,2)]]},
      'heavy_spinor_cache':[4,5],'light_spinor_cache':[2,3],'pair_momentum':'0 modulo reciprocal lattice'},[f'admission:leg_{n}_{suffix}' for n in (2,3,4,5) for suffix in ('inputs','energy','cache_origin')])
    phases={}
    for case in ('incoming','same','cross'):
        phases[case]=raw('phase_'+case,lambda case=case:sp.prod(cosine+sp.I*sine for cosine,sine in kin['ordered_signed_legs'][case]),['A:kinematics'])
    kr=raw('kinematic_residuals',lambda:sp.Matrix([kin['E_H']**2-kin['mass_H']**2-kin['d_H']**2,kin['E_L']**2-kin['d_L']**2,2*kin['E_H']-2*kin['E_L'],kin['d_H']**2+kin['cos_H']**2-1,kin['d_L']**2+kin['cos_L']**2-1,phases['same']-phases['incoming'],phases['cross']-phases['incoming']]),['A:kinematics','A:phase_incoming','A:phase_same','A:phase_cross'])
    tw=raw('threshold_weights',lambda:sp.Matrix([((1+sp.sqrt(1-c['leg_1_energy']**2))/2)**2,((1-sp.sqrt(1-c['leg_1_energy']**2))/2)**2]),['admission:leg_1_energy'])
    ths=raw('threshold_same_tensor',lambda:tw[0]*c['threshold_tensor'],['A:threshold_weights','admission:threshold_tensor'])
    thc=raw('threshold_cross_tensor',lambda:tw[1]*c['threshold_tensor'],['A:threshold_weights','admission:threshold_tensor'])
    def f(b,cosz):
        return sp.prod(R(1,2)*(1+(-1)**b[i]*(cosz if i==2 else 1)) for i in range(3))
    mixed=raw('mixed_rest_incoming_products',lambda:sp.Matrix([f(b,1)*f(b,-1) for b in BITS]))
    weights=raw('finite_filter_weights',lambda:sp.Matrix([[f(b,kin['cos_H'])**2,f(b,kin['cos_L'])**2,f(b,-kin['cos_L'])**2] for b in BITS]),['A:kinematics'])
    hp,hm,lp,lm=(c[f'leg_{n}_spinors'] for n in (4,5,2,3)); M=[c['M'+str(m)] for m in range(4)]
    jh=[c['collinear_JH_0']]
    for m in range(1,4):
        jh.append(raw(f'heavy_spatial_current_{m}',lambda m=m:hp.T*M[m]*hm,[f'admission:M{m}','admission:leg_4_spinors','admission:leg_5_spinors']))
    jl=[c[f'same_valley_threshold_JL_{m}'] for m in range(4)]
    wh=raw('wedge_H',lambda:wedge(hp,hm,c['pair_indices']),['admission:leg_4_spinors','admission:leg_5_spinors','admission:pair_indices'])
    wl=raw('wedge_L',lambda:wedge(lp,lm,c['pair_indices']),['admission:leg_2_spinors','admission:leg_3_spinors','admission:pair_indices'])
    swh=raw('swapped_wedge_H',lambda:wedge(hm,hp,c['pair_indices']),['admission:leg_4_spinors','admission:leg_5_spinors','admission:pair_indices'])
    swl=raw('swapped_wedge_L',lambda:wedge(lm,lp,c['pair_indices']),['admission:leg_2_spinors','admission:leg_3_spinors','admission:pair_indices'])
    fjh={}; fjl={}; fwh={}; fwl={}
    for j,b in enumerate(('000','001')):
        for m in range(4):
            source='admission:collinear_JH_0' if m==0 else f'A:heavy_spatial_current_{m}'
            fjh[b,m]=raw(f'filtered_JH_{b}_{m}',lambda j=j,m=m:weights[j,0]*jh[m],['A:finite_filter_weights',source])
        for case, col in (('same',1),('cross',2)):
            for m in range(4):
                fjl[case,b,m]=raw(f'filtered_JL_{case}_{b}_{m}',lambda j=j,col=col,m=m:weights[j,col]*jl[m],['A:finite_filter_weights',f'admission:same_valley_threshold_JL_{m}'])
        fwh[b]=raw(f'filtered_wedge_H_{b}',lambda j=j:weights[j,0]*wh,['A:finite_filter_weights','A:wedge_H'])
        for case,col in (('same',1),('cross',2)):
            fwl[case,b]=raw(f'filtered_wedge_L_{case}_{b}',lambda j=j,col=col:weights[j,col]*wl,['A:finite_filter_weights','A:wedge_L'])
    tensors={}; wedges={}; rev={}; averages={}
    for case in ('same','cross'):
        deps=[f'A:filtered_JH_{b}_{m}' for b in ('000','001') for m in range(4)]+[f'A:filtered_JL_{case}_{b}_{m}' for b in ('000','001') for m in range(4)]
        tensors[case]=raw('tensor_current_'+case,lambda case=case:sp.Matrix(4,4,lambda x,y:4*sum((1 if m==0 else -1)*sp.conjugate(fjl[case,b,m][x//2,x%2])*fjh[b,m][y//2,y%2] for b in ('000','001') for m in range(4))),deps)
        deps=[f'A:filtered_wedge_H_{b}' for b in ('000','001')]+[f'A:filtered_wedge_L_{case}_{b}' for b in ('000','001')]+['admission:K']
        wedges[case]=raw('tensor_wedge_'+case,lambda case=case:sum((fwl[case,b].H*c['K']*fwh[b] for b in ('000','001')),sp.zeros(4)),deps)
        rev[case]=raw('reverse_'+case,lambda case=case:sum((fwh[b].H*c['K']*fwl[case,b] for b in ('000','001')),sp.zeros(4)),deps)
        averages[case]=raw('spin_average_'+case,lambda case=case:sp.trace(tensors[case].H*tensors[case])/4,['A:tensor_current_'+case])
    u0=c['leg_1_spinors']; az=c['alpha_z']; h0=c['leg_1_H']; e0=c['leg_1_energy']
    jp=raw('jet_H_plus',lambda:az*u0,['admission:alpha_z','admission:leg_1_spinors'])
    jm=raw('jet_H_minus',lambda:-az*u0,['admission:alpha_z','admission:leg_1_spinors'])
    ep=raw('jet_eigen_plus',lambda:(h0-e0*sp.eye(4))*jp+R(6,5)*az*u0,['A:jet_H_plus','admission:leg_1_H','admission:leg_1_energy','admission:alpha_z','admission:leg_1_spinors'])
    em=raw('jet_eigen_minus',lambda:(h0-e0*sp.eye(4))*jm-R(6,5)*az*u0,['A:jet_H_minus','admission:leg_1_H','admission:leg_1_energy','admission:alpha_z','admission:leg_1_spinors'])
    np=raw('jet_norm_plus',lambda:jp.H*u0+u0.H*jp,['A:jet_H_plus','admission:leg_1_spinors'])
    nm=raw('jet_norm_minus',lambda:jm.H*u0+u0.H*jm,['A:jet_H_minus','admission:leg_1_spinors'])
    jets=[]
    for m in range(4):
        jets.append(raw(f'jet_JH_{m}',lambda m=m:jp.T*M[m]*u0+u0.T*M[m]*jm,['A:jet_H_plus','A:jet_H_minus','admission:leg_1_spinors',f'admission:M{m}']))
    jet_t={}
    for case,j in (('same',0),('cross',1)):
        jet_t[case]=raw('jet_'+case+'_tensor',lambda j=j:sp.Matrix(4,4,lambda x,y:4*tw[j]*sum((1 if m==0 else -1)*sp.conjugate(jl[m][x//2,x%2])*jets[m][y//2,y%2] for m in range(4))),['A:threshold_weights']+[f'A:jet_JH_{m}' for m in range(4)]+[f'admission:same_valley_threshold_JL_{m}' for m in range(4)])
    factors=[]
    for case,col in (('same',1),('cross',2)):
        factors.append(raw('filter_factor_'+case,lambda col=col:sum(weights[j,0]*weights[j,col] for j in range(8)),['A:finite_filter_weights']))
    boost=raw('finite_spin_boost',lambda:factors[1]*c['mixed_boost'],['A:filter_factor_cross','admission:mixed_boost'])
    cmp('kinematics',lambda:(kr),lambda:(sp.zeros(7,1)),['A:kinematic_residuals']); cmp('threshold_weights',lambda:(tw),lambda:(sp.Matrix([R(81,100),R(1,100)])),['A:threshold_weights'])
    cmp('threshold_same',lambda:(ths),lambda:(sp.zeros(4)),['A:threshold_same_tensor']); cmp('threshold_cross',lambda:(thc),lambda:(sp.zeros(4)),['A:threshold_cross_tensor'])
    cmp('mixed_rest_removed',lambda:(mixed),lambda:(sp.zeros(8,1)),['A:mixed_rest_incoming_products'])
    factor_target=lambda sign:(161+sign*60*sp.sqrt(7))/400
    cmp('filter_same',lambda:(factors[0]),lambda:(factor_target(1)),['A:filter_factor_same']); cmp('filter_cross',lambda:(factors[1]),lambda:(factor_target(-1)),['A:filter_factor_cross'])
    swap=lambda:sp.Matrix(4,4,lambda x,y:int(x==2*(y%2)+y//2))
    cmp('swap_H',lambda:(swh),lambda:(-wh*swap()),['A:swapped_wedge_H','A:wedge_H']); cmp('swap_L',lambda:(swl),lambda:(-wl*swap()),['A:swapped_wedge_L','A:wedge_L'])
    for case, sign in (('same',1),('cross',-1)):
        cmp(case+'_routes',lambda:(tensors[case]),lambda:(wedges[case]),['A:tensor_current_'+case,'A:tensor_wedge_'+case])
        cmp(case+'_tensor',lambda:(tensors[case]),lambda:(sp.diag(-R(32,5)*factor_target(sign),0,0,-R(32,5)*factor_target(sign))),['A:tensor_current_'+case])
        cmp(case+'_reverse',lambda:(rev[case]),lambda:(tensors[case].H),['A:reverse_'+case,'A:tensor_current_'+case])
        cmp(case+'_average',lambda:(averages[case]),lambda:(R(512,25)*factor_target(sign)**2),['A:spin_average_'+case])
    cmp('jet_eigen_plus',lambda:(ep),lambda:(sp.zeros(4,2)),['A:jet_eigen_plus'])
    cmp('jet_eigen_minus',lambda:(em),lambda:(sp.zeros(4,2)),['A:jet_eigen_minus'])
    cmp('jet_norm_plus',lambda:(np),lambda:(sp.zeros(2)),['A:jet_norm_plus'])
    cmp('jet_norm_minus',lambda:(nm),lambda:(sp.zeros(2)),['A:jet_norm_minus'])
    cmp('jet_same',lambda:(jet_t['same']),lambda:(sp.diag(-R(324,25),0,0,-R(324,25))),['A:jet_same_tensor'])
    cmp('jet_cross',lambda:(jet_t['cross']),lambda:(sp.diag(-R(4,25),0,0,-R(4,25))),['A:jet_cross_tensor'])
    cmp('finite_spin_boost',lambda:(boost),lambda:(sp.Matrix([0,0,32*factor_target(-1),32*factor_target(-1),0,0])),['A:finite_spin_boost'])

def current(rec,c,t):
    raw=lambda n,fn,deps=(): rec.raw('C',n,fn,deps)
    cmp=lambda n,a,e,deps=(): rec.compare('C','C.'+n,a,e,deps)
    relative=list(itertools.product(range(-2,3),repeat=3))
    raw('relative_indices',lambda:sp.Matrix(relative))
    axes=[]
    for jet in range(3):
        factors=[lambda n,m:sp.Integer(1),lambda n,m:sp.I*R(n+m,2),lambda n,m:-R((n+m)**2,4)]
        axes.append(raw('one_axis_L'+str(jet),lambda jet=jet:sp.Matrix([[sum(t[s,n+1]*t[s,m+1]*factors[jet](n,m) for n in range(-1,2) for m in range(-1,2) if n-m==r) for r in range(-2,3)] for s in range(2)]),['F:one_dimensional_stencils']))
    L=[]
    for jet in range(3):
        L.append(raw('L'+str(jet),lambda jet=jet:sp.Matrix([[axes[0][b[0],r[0]+2]*axes[0][b[1],r[1]+2]*axes[jet][b[2],r[2]+2] for r in relative] for b in BITS]),['C:relative_indices','C:one_axis_L0','C:one_axis_L'+str(jet)]))
    s0=raw('S0',lambda:L[0]*L[0].T,['C:L0'])
    g10=raw('G10',lambda:L[1]*L[0].T,['C:L1','C:L0']); g01=raw('G01',lambda:L[0]*L[1].T,['C:L0','C:L1'])
    s1=raw('S1',lambda:g10+g01,['C:G10','C:G01'])
    g20=raw('G20',lambda:L[2]*L[0].T,['C:L2','C:L0']); g11=raw('G11',lambda:L[1]*L[1].T,['C:L1']); g02=raw('G02',lambda:L[0]*L[2].T,['C:L0','C:L2'])
    s2=raw('S2',lambda:g20+2*g11+g02,['C:G20','C:G11','C:G02'])
    e=raw('evaluation',lambda:sp.ones(125,1)); ls=[]
    for j in range(3):
        ls.append(raw('l'+str(j),lambda j=j:L[j]*e,['C:L'+str(j),'C:evaluation']))
    r=raw('r',lambda:L[0].T*ls[0],['C:L0','C:l0']); tv=raw('t',lambda:L[2].T*ls[0],['C:L2','C:l0']); u=raw('u',lambda:L[0].T*ls[2],['C:L0','C:l2'])
    raw('spin_factor',lambda:{'cache_id':rec.ids['admission:spin_factor'],'inherited':True},['admission:spin_factor'])
    a0=raw('a2_constant',lambda:tv+u,['C:t','C:u']); ah=raw('a2_h',lambda:-r,['C:r'])
    c0=raw('c1_constant',lambda:u,['C:u']); ch=raw('c1_h',lambda:-r/2,['C:r'])
    ct0=raw('c1tilde_constant',lambda:tv,['C:t']); cth=raw('c1tilde_h',lambda:-r/2,['C:r'])
    z=c['spin_factor']
    gram=raw('current_slope_gram',lambda:sp.Matrix([z*(ls[0].T*s2*ls[0])[0],-z*(ls[0].T*s0*ls[0])[0]]),['admission:spin_factor','C:l0','C:S2','C:S0'])
    direct=raw('current_slope_direct',lambda:sp.Matrix([z*((r.T*a0)[0]-(r.T*c0)[0]+(ct0.T*r)[0]),z*((r.T*ah)[0]-(r.T*ch)[0]+(cth.T*r)[0])]),['admission:spin_factor','C:r','C:a2_constant','C:a2_h','C:c1_constant','C:c1_h','C:c1tilde_constant','C:c1tilde_h'])
    raw('kernel_moment_scope',lambda:{'h':'sum_r g_r*r_z^2 = -kappa_second_derivative_at_zero','domain':'h>=0; finite actual compact-kernel lattice moment','numeric_kernel_substitution':False})
    raw('product_state_scope',lambda:{'state':'heavy components0,2 at p=(0,0,theta/2a); light vacuum','normalization':'(V_s/a)*d< Q2_z >/dtheta at0','finite_box':'even L>=6, La>8ell+8sqrt(3)a; local lifted displacements','continuum_or_history_claim':False})
    axis0=lambda:sp.Matrix([[R(1,16),R(s,4),R(3,8),R(s,4),R(1,16)] for s in (1,-1)])
    axis2=lambda:sp.Matrix([[0,-R(s,16),-R(1,8),-R(s,16),0] for s in (1,-1)])
    base=lambda:sp.Matrix([[35,3],[3,35]])/128
    second=lambda:sp.Matrix([[-5,-1],[-1,-5]])/32
    cmp('axis_L0',lambda:axes[0],axis0,['C:one_axis_L0'])
    cmp('axis_L1',lambda:axes[1],lambda:sp.zeros(2,5),['C:one_axis_L1'])
    cmp('axis_L2',lambda:axes[2],axis2,['C:one_axis_L2'])
    cmp('S0',lambda:s0,lambda:sp.kronecker_product(base(),base(),base()),['C:S0'])
    cmp('S1',lambda:s1,lambda:sp.zeros(8),['C:S1'])
    cmp('S2',lambda:s2,lambda:sp.kronecker_product(base(),base(),second()),['C:S2'])
    cmp('l0',lambda:(ls[0]),lambda:(sp.Matrix([1]+[0]*7)),['C:l0']); cmp('l1',lambda:(ls[1]),lambda:(sp.zeros(8,1)),['C:l1'])
    cmp('current_routes',lambda:(gram),lambda:(direct),['C:current_slope_gram','C:current_slope_direct'])
    cmp('current_slope',lambda:(gram),lambda:(sp.Matrix([-R(6125,8192),-R(42875,32768)])),['C:current_slope_gram'])
    # Rational sign decision is distinct from equality of coefficient pairs.
    cmp('current_nonzero_coefficients',lambda:([bool(x.p<0 and x.q>0) for x in gram]),lambda:([True,True]),['C:current_slope_gram'])

def main():
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM,signal.SIGXCPU):
        signal.signal(sig,handle)
    signal.setitimer(signal.ITIMER_REAL,100)
    rec=None
    try:
        rec=Record()
        receipt={'executable':sys.executable,'python_version':sys.version,'sympy_version':sp.__version__,
          'affinity':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),
          'address_space_limit':list(resource.getrlimit(resource.RLIMIT_AS)),
          'cpu_limit':list(resource.getrlimit(resource.RLIMIT_CPU)),
          'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')},
          'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
        rec.put('admission','stage_state','actual_runtime_receipt',receipt,[])
        if receipt['executable']!='<internal>/python' or len(receipt['affinity'])!=1 or receipt['nice']<10 or receipt['address_space_limit']!=[2*1024**3]*2 or receipt['cpu_limit']!=[90,100] or any(v!='1' for v in receipt['thread_environment'].values()) or receipt['CUDA_VISIBLE_DEVICES']!='':
            raise ValueError('actual runtime settings violate contract')
        cache=admit(rec); t=filters(rec,cache)
        amplitudes(rec,cache); current(rec,cache,t)
    except BaseException as exc:
        if rec:
            rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)}
        else:
            traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec:
            rec.terminal=True
            rec.checkpoint()
            for f in rec.streams.values():
                f.flush(); os.fsync(f.fileno()); f.close()
    if rec is None or rec.error:
        return 2
    return 0 if all(s+':'+n in rec.ids for s in RAW for n in RAW[s]) and all(rec.done.values()) and len(rec.done)==45 else 3

if __name__=='__main__':
    sys.exit(main())
