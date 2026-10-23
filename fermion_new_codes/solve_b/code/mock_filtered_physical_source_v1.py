"""UNRUN exact source adapter from retained #145 n2/n3 static records.

Reuses the admitted U0/U1 operator maps by the declared linear Hamiltonian
coefficient. Q is rebuilt from the combined source, NEVER interpolated. No
static reference solver, propagation, midpoint evaluation or RDM is run here.
Complete original-pair static/source/control reading remains prerequisite.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import ast
import hashlib
import json
import run_mock_vertex_admission_v1 as e
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_source_context_v1 as context
import mock_unitary_lead_run_v1 as baseline

BASE=Path('<internal>/MOCK_filtered_clockwell_v1')
# These immutable fingerprints/payloads are minted only by byte-verified
# intake and successful binding, never by a candidate's self-declared seal.
_ORIGINAL_ROUTES={}
_ORIGINAL_LIVE_SOURCES={}

JOURNALS={2:'0f00e041d42160ac070a0fa0a8dbe597f6dddeb91eb2d13bc08634079430610e',
          3:'231c7e799d08c6577bfcdcdd577d50162caeb5454c30ced155997868414ad492'}


def complex_value(value):
    if not isinstance(value,dict) or set(value)!=set(('re','im')) or any(type(value[k]) is not str for k in value):
        raise ValueError('retained exact complex encoding')
    return e.C(F(value['re']),F(value['im']))


def sparse_map(value,rows,columns):
    if not isinstance(value,dict):raise ValueError('retained sparse dictionary')
    result={}
    for key,z in value.items():
        pair=ast.literal_eval(key)
        if type(pair) is not tuple or len(pair)!=2 or any(type(i) is not int for i in pair):raise ValueError('retained sparse key')
        result[pair]=complex_value(z)
    context.matrix_rows(result,rows,columns)
    return result


def combine(zero,one,g):
    return e.matrix_add(zero,joining._subtract(one,zero),g)


def load_admitted(n):
    if type(n) is not int or n not in (2,3):raise ValueError('physical source only n2/n3')
    path=BASE/('static_n'+str(n)+'_v1')/'records.ndjson'
    if path.is_symlink() or not path.is_file():raise ValueError('retained static journal path')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=JOURNALS[n]:raise ValueError('retained static journal identity')
    short=clock.StaticBasis.make(n);long=clock.StaticBasis.make(n+4,'joining_reference')
    routes={};references={};terminal=None;summaries=[];intake=None
    with path.open() as stream:
        for line in stream:
            record=json.loads(line);kind=record['kind'];data=record['data'];terminal=record
            if kind=='intake':intake=data
            elif kind=='weighted_static_admission_summary':summaries.append(data)
            elif kind=='patterned_joining_before_reference' and data['U'] in (0,1):
                U=data['U']
                if U in routes:raise ValueError('duplicate retained tested route')
                routes[U]=data['_exact']
            elif kind=='independent_joining_before_comparison' and data['U'] in (0,1):
                U=data['U']
                if U in references:raise ValueError('duplicate retained independent route')
                references[U]=data['_exact']
    if terminal['kind']!='termination' or terminal['data']['status']!='complete' or not terminal['data']['all_pass'] or not terminal['data']['sources_unchanged'] or terminal['data']['threads']!=1:
        raise ValueError('retained static admission incomplete')
    if len(summaries)!=1 or summaries[0]['n']!=n or set(routes)!={0,1} or set(references)!={0,1}:
        raise ValueError('retained static size/route association')
    if intake['sources']!=terminal['data']['sources']:raise ValueError('retained static sources changed')
    decoded={}
    for U in (0,1):
        tested=routes[U];reference=references[U];patterns={}
        for key,O in tested['patterns'].items():
            P=ast.literal_eval(key)
            if type(P) is not tuple or len(P) not in (1,2) or any(type(j) is not int for j in P):raise ValueError('retained outside pattern')
            if tuple(sorted(set(P)))!=P or any(not short.modes<=j<long.modes for j in P):raise ValueError('retained pattern physical labels')
            patterns[P]=sparse_map(O,1 if len(P)==2 else short.modes,short.dimension)
        S=sparse_map(tested['S'],long.dimension,short.dimension);Q=sparse_map(tested['Q'],short.dimension,short.dimension)
        Sr=sparse_map(reference['S'],long.dimension,short.dimension);Qr=sparse_map(reference['Q'],short.dimension,short.dimension)
        Hs=sparse_map(reference['H_short'],short.dimension,short.dimension);Hl=sparse_map(reference['H_long'],long.dimension,long.dimension)
        if S!=Sr or Q!=Qr:raise ValueError('retained exact tested/reference source mismatch')
        embedding=tuple(reference['embedding']);index={mask:i for i,mask in enumerate(long.states)}
        if embedding!=tuple(index[mask] for mask in short.states):raise ValueError('retained empty embedding')
        decoded[U]=dict(patterns=patterns,S=S,Q=Q,H_short=Hs,H_long=Hl,embedding=embedding)
    payload=dict(short=short,long=long,routes=decoded,path=str(path),journal_sha256=digest,
        source_identities=intake['sources'],header=clock.TABLE_HEADER)
    seal=context.fingerprint(payload)
    if n in _ORIGINAL_ROUTES and _ORIGINAL_ROUTES[n]!=seal:raise ValueError('pinned original routes changed')
    _ORIGINAL_ROUTES[n]=seal
    return dict(**payload,seal=seal)


@dataclass(frozen=True)
class PhysicalSource:
    n: int
    strength: F
    basis_states: tuple
    H_rows: tuple
    S_rows: tuple
    Q_rows: tuple
    patterns: tuple
    basis: str
    operator: str
    actual_operator: str
    target_error: F
    provenance: str
    seal: str

    def payload(self):
        return (self.n,self.strength,self.basis_states,self.H_rows,self.S_rows,self.Q_rows,self.patterns,
            self.basis,self.operator,self.actual_operator,self.target_error,self.provenance)

    def verify(self):
        baseline.domain(self.n,float(self.strength),2)
        if type(self.strength) is not F or self.strength not in (F(0),F(1,2),F(1),F(2)):raise ValueError('fixed physical source strength')
        if context.fingerprint(self.payload())!=self.seal:raise ValueError('physical source/context changed')
        key=(self.n,self.strength,self.actual_operator,self.target_error,self.provenance)
        if _ORIGINAL_LIVE_SOURCES.get(key)!=self.payload():raise ValueError('original live source authority differs from candidate')
        return self

    def norm_ratio(self,state,expected,save=None):
        # Reuse only the exact action formula, not the old constructor or its
        # signed-static-U gate. This new class retains its own physical roster,
        # actual-model and exact-target identities in its immutable seal.
        return context.JoiningSource.norm_ratio(self,state,expected,save=save)

    @classmethod
    def bind(cls,model,admitted,*,save,phase):
        model.verify()
        if not callable(save) or not callable(phase):raise ValueError('combined source/phase callbacks required')
        if model.n not in _ORIGINAL_ROUTES:
            phase('source_original_pinned_intake',lambda:load_admitted(model.n))
        if admitted.get('seal')!=context.fingerprint({key:value for key,value in admitted.items() if key!='seal'}):raise ValueError('retained admitted source content changed')
        n=model.n;short=admitted['short'];long=admitted['long'];g=F(model.strength)
        if n not in (2,3) or short!=clock.StaticBasis.make(n) or long!=clock.StaticBasis.make(n+4,'joining_reference'):
            raise ValueError('physical/static complete source basis')
        if admitted['journal_sha256']!=JOURNALS[n] or admitted['path']!=str(BASE/('static_n'+str(n)+'_v1')/'records.ndjson'):
            raise ValueError('physical source admitted record association')
        zero=admitted['routes'][0];one=admitted['routes'][1]
        Hs=combine(zero['H_short'],one['H_short'],g);Hl=combine(zero['H_long'],one['H_long'],g)
        save('physical_source_combined_H_before_checks',dict(header=clock.TABLE_HEADER,g=g,short_H=Hs,long_H=Hl,
            original_route_identity=_ORIGINAL_ROUTES[n],claimed_route_identity=admitted['seal'],claimed_journal=admitted['journal_sha256']))
        S=combine(zero['S'],one['S'],g);patterns={}
        for P in set(zero['patterns'])|set(one['patterns']):
            matrix=combine(zero['patterns'].get(P,{}),one['patterns'].get(P,{}),g)
            if matrix:patterns[P]=matrix
        lower={0:(0,),1:tuple(1<<j for j in range(short.modes))};index={mask:i for i,mask in enumerate(long.states)}
        reconstructed={};Q={}
        for P,O in patterns.items():
            outside=sum(1<<j for j in P)
            for (i,j),z in O.items():e.add_entry(reconstructed,(index[lower[2-len(P)][i]|outside],j),z)
            Q=e.matrix_add(Q,joining._gram(O,len(lower[2-len(P)])))
        direct={};embedding=zero['embedding']
        save('physical_source_combined_patterns_before_checks',dict(header=clock.TABLE_HEADER,g=g,S=S,patterns=patterns,Q=Q,
            reconstructed=reconstructed,embedding_zero=embedding,embedding_one=one['embedding']))
        inverse={j:i for i,j in enumerate(embedding)}
        for (i,j),z in Hl.items():
            if j in inverse:e.add_entry(direct,(i,inverse[j]),z)
        for (i,j),z in Hs.items():e.add_entry(direct,(embedding[i],j),-z)
        gram=phase('source_combined_direct_Gram',lambda:joining._gram(S,long.dimension))
        save('physical_source_combined_reference_before_checks',dict(header=clock.TABLE_HEADER,g=g,H_short=Hs,H_long=Hl,
            S=S,patterns=patterns,Q=Q,reconstructed=reconstructed,direct=direct,direct_Gram=gram,
            actual_model_target=model.independent_intended,original_route_identity=_ORIGINAL_ROUTES[n],claimed_route_identity=admitted['seal']))
        if admitted['seal']!=_ORIGINAL_ROUTES[n]:raise ValueError('candidate routes differ from original pinned journal contents')
        if Hs!=model.independent_intended:raise ValueError('physical source differs from actual model intended target')
        if embedding!=one['embedding']:raise ValueError('static source embedding changed with coupling')
        if reconstructed!=S or direct!=S or gram!=Q:raise ValueError('combined physical source/reference interference identity')
        context.validate_Q_axis(Q)
        Hrows=context.matrix_rows(Hs,short.dimension,short.dimension)
        Srows=context.matrix_rows(S,long.dimension,short.dimension);Qrows=context.matrix_rows(Q,short.dimension,short.dimension)
        prows=tuple((P,context.matrix_rows(O,len(lower[2-len(P)]),short.dimension)) for P,O in sorted(patterns.items()))
        basis=context.fingerprint((short.n,short.modes,short.states,short.pairs,short.role))
        target=context.fingerprint((clock.SPEC_SHA,clock.TABLE_HEADER,g,basis,Hrows))
        provenance=context.fingerprint((admitted['path'],admitted['journal_sha256'],admitted['source_identities']))
        payload=(n,g,short.states,Hrows,Srows,Qrows,prows,basis,target,model.bound_H_identity,F(model.error),provenance)
        key=(n,g,model.bound_H_identity,F(model.error),provenance)
        if key in _ORIGINAL_LIVE_SOURCES and _ORIGINAL_LIVE_SOURCES[key]!=payload:raise ValueError('original live source cannot be redefined')
        _ORIGINAL_LIVE_SOURCES[key]=payload
        result=cls(*payload,context.fingerprint(payload));result.verify()
        save('physical_source_original_live_authority',dict(header=clock.TABLE_HEADER,key=key,payload=payload,seal=result.seal))
        return result
