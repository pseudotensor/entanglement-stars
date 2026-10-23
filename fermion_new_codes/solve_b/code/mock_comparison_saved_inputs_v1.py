"""UNRUN draft: context-bound intake of the frozen baseline's saved actions.

Definition only. No new propagation, eigensolve, entropy or D_RK calculation.
This adapter still requires original-pair source and artificial-control reads.
It admits only the original size-two/three baseline; no filtered/n7 path.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import re

BASE=Path('<internal>/MOCK_unitary_lead_run_v1')
JOURNAL_SHA={
 'even_n2_g0p5_v1':'83a4a316c85428041b801c77d2a61804ca753359b7ba151eb614ee30de28239a',
 'even_n2_g0_v1':'8b966d79301320b7c7e6aa528baab07f6abd30880554da831c8b71f090247f9d',
 'even_n2_g1_v1':'f38284e1c250f6e3e4e7b279627aae668bf29bb9b0067f82beed2593a447d1fc',
 'even_n2_g2_v1':'e99ebf31c89f2c1fc934b56947ffd0883eaa80d4347afd2436611afde4fe448a',
 'even_n3_g0p5_v1':'c72fb126759a37728bf8f56e8ca6db127aa2ed3e082966da25574ef179421505',
 'even_n3_g0_v1':'b6430d97168838516eb8c91da03cf1cb659c209aa2ebdfb2e4a93057c0ffe036',
 'even_n3_g1_v1':'bf133c6985acd54ef85739f8eb9544cb8328bd8eee11faf9af15c97ba6e086b2',
 'even_n3_g2_v1':'b29013e6e9aa26fc2bb10c22cf987506d680cae793c87f5774a2c47d31e56bad',
 'odd_n2_v1':'90781badafa22ed6fa9ad34770c61236e12d73c2b18ad8f630a3dd3b13a55bad',
 'odd_n3_v1':'ce6fb40d7cfdddcf74de2b562bc6397f4ad80f53622e125b7d48d79e773e851b',
}
SOURCE_SHA={
 'run_mock_unitary_lead_run_v1.py':'756007de07105b74ad945884db6c91d58982833ff5dbf13131d3b61ac155e680',
 'mock_unitary_lead_run_v1.py':'898235fb289a2a3118861ee2db8d10ddaf6385b412eb762f0311292e9646e112',
 'mock_unitary_lead_protection_v1.py':'8636f63645e808d24c00bb147883db05fbe719b055375eca86777d9945cf04b2',
 'mock_unitary_lead_records_v1.py':'a18f1461ce08b2ccfbeedb57a15deddb8f430628c441bf308c5125edb3f04536',
 'mock_unitary_lead_v1.py':'6fe256e6940c35db8ebfcdce8e3e2228d8ed2361a2b37b0728d799b00133e21a',
 'mock_axial_vertex_adapter_v1.py':'482b1b7a3a7d4600d06cdd5cc6030d63cdc22d30876006ac0214a712da579e7b',
 'run_mock_vertex_admission_v1.py':'6f74f133d01070c25c32c96b5231e78f6840c216cb9d0ef94f8b925c6a148265',
 'cpu_runtime.py':'207c7a16432e722fa0b77c1e9d05e0d2ef712b540a95f25762bcb13cd10feb85',
 'MOCK_unitary_lead_run_spec_v1.md':'f2f6f1d1052f79a3c41a5a15849037566c6c05690bb543281e217bee5f1f6ab2',
 'redteam_MOCK_unitary_lead_run_v1.md':'3607224b6f1efc60b676c6d3c6d7b36afd76ad05e03469a8be3626ef5a922306',
}


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()


@contextmanager
def original_precision():
    from mpmath import iv
    previous=iv.dps
    try:
        iv.dps=55
        yield
    finally:iv.dps=previous


def _source_roster(sources):
    result={}
    for path,sha in sources.items():
        name=Path(path).name
        if name in result:raise ValueError('duplicate original source basename')
        result[name]=sha
    if result!=SOURCE_SHA:raise ValueError('original source/review/spec identity changed')


@dataclass(frozen=True)
class OriginalContext:
    """Immutable intake authority; mutable working maps are never authority."""
    path: str
    journal: str
    model: tuple
    roles: tuple
    outputs: tuple
    clocks: tuple
    sources: tuple
    actions: tuple



class SavedBaseline:
    """Read immutable source-bound primary records, then join by fixed role/clock.

    Large unrelated raw exact-density rows are streamed past without parsing.
    The complete journal hash and original row locators accompany every input.
    Reusing an old eigensystem audits its residual; it never diagonalizes H.
    """
    def __setattr__(self,name,value):
        if name=='_original_context' and hasattr(self,name):raise ValueError('immutable original context')
        object.__setattr__(self,name,value)

    def __init__(self,n,g,number):
        import numpy as np
        import mock_unitary_lead_v1 as q
        import mock_unitary_lead_run_v1 as m
        import mock_unitary_lead_protection_v1 as p
        import run_mock_unitary_lead_run_v1 as runner
        import mock_axial_vertex_adapter_v1 as vertex
        import run_mock_vertex_admission_v1 as exact
        for module in (q,m,p,runner,vertex,exact):
            if file_sha(module.__file__)!=SOURCE_SHA[Path(module.__file__).name]:
                raise ValueError('loaded baseline dependency differs from frozen source')
        m.domain(n,g,number)
        if number==1 and g!=0:raise ValueError('shared odd baseline is coupling zero only')
        label={0.:'0',1.:'1',.5:'0p5',2.:'2'}[g]
        self.path=BASE/(f'odd_n{n}_v1' if number==1 else f'even_n{n}_g{label}_v1')
        if self.path.resolve()!=self.path:raise ValueError('baseline path alias')
        self.locators=[];self.reused_readouts=[];single={};operators=[];actions=[]
        wanted={'intake','model','preparations','plans','physical_job_complete','termination'}
        reused={'entropy','register','conversion_status','purification','purification_status','wall_complete'}
        journal=self.path/'records.ndjson';digest=hashlib.sha256()
        with journal.open('rb') as stream:
            for line_number,line in enumerate(stream,1):
                digest.update(line)
                match=re.match(rb'\{"kind":"([^"]+)"',line)
                if not match:raise ValueError('noncanonical original journal')
                kind=match[1].decode('ascii')
                if kind in reused:
                    self.reused_readouts.append({'line':line_number,'kind':kind,'row_sha256':hashlib.sha256(line).hexdigest()})
                    continue
                if kind not in wanted|{'numerical_action','action'}:continue
                row=json.loads(line);data=row['data']
                locator={'line':line_number,'kind':kind,'row_sha256':hashlib.sha256(line).hexdigest()}
                if kind in wanted:
                    if kind in single:raise ValueError('duplicate singleton record')
                    single[kind]=(data,locator)
                elif kind=='numerical_action' and data.get('kind')=='operator':operators.append((data,locator))
                elif kind=='action' and data['metadata'].get('role')=='primary' and data['metadata'].get('end') in m.CLOCKS:
                    actions.append((data,locator))
        if set(single)!=wanted:raise ValueError('incomplete original baseline journal')
        self.journal_sha=digest.hexdigest()
        if self.journal_sha!=JOURNAL_SHA[self.path.name]:raise ValueError('frozen baseline journal identity changed')
        intake=single['intake'][0];terminal=single['termination'][0]
        _source_roster(intake['sources']);_source_roster(terminal['sources'])
        if intake['sources']!=terminal['sources']:raise ValueError('changed source paths')
        if (terminal['status']!='complete' or terminal['all_pass'] is not True or
            terminal['sources_unchanged'] is not True or terminal['threads']!=1 or terminal['runtime_admitted'] is not True):
            raise ValueError('original runtime/termination not admitted')
        complete=single['physical_job_complete'][0]
        if any(complete[k]!=v for k,v in {'n':n,'g':g,'number':number,'filtered':False,'status':'complete_fixed_readouts'}.items()):
            raise ValueError('original physical job role mismatch')
        model=single['model'][0];metadata=model['metadata'];basis=q.Basis.make(8*n,number)
        expected={'n':n,'g':g,'number':number,'basis':q.basis_identity(basis),'profile':[.5]+[1.]*(n-1),
                  'flow':0.,'a':1.,'mass_intended':'3/5'}
        if any(metadata[k]!=v for k,v in expected.items()):raise ValueError('original model context mismatch')
        h=self.array(model['actual_H'],(basis.dimension,basis.dimension),'complex128')
        if q.array_identity(h)!=metadata['operator']:raise ValueError('original H identity mismatch')
        terms=tuple(q.Term(complex(re,im),tuple((j,d) for j,d in ops)) for re,im,ops in model['terms'])
        self.model=SimpleNamespace(n=n,strength=float(g),number=number,basis=basis,terms=terms,h=h,error=metadata['target_error'])
        self.plans={};self.actions={};self.basis=basis
        with original_precision():
            prep=m.preparations(self.model);stored=single['preparations'][0]
            for key in ('pure','registered','words'):
                array=self.array(stored[key],prep[key].shape,'complex128')
                if not np.array_equal(array,prep[key]):raise ValueError('fixed original code/preparation mismatch')
            for key in ('pure_error','registered_error'):
                if prep[key]!=stored[key]:raise ValueError('original coefficient-error mismatch')
            plans=single['plans'][0]
            for role,prefix in (('pure','pure_'),('registered','register_')):
                plan=runner.make_plan(self.model,prep[role],prep[role+'_error'],prefix+str((n,float(g),number)))
                if plan.identity!=plans[role]:raise ValueError('original plan/code/collection mismatch')
                self.plans[role]=plan
            for key,value in {'collections':list(plan.collections),'blocks':[list(x) for x in plan.blocks],
                              'primary_clocks':list(m.CLOCKS),'outputs':{k:list(v) for k,v in m.outputs(self.model).items()},
                              'first_outputs':{k:list(v) for k,v in m.outputs(self.model,True).items()}}.items():
                if plans[key]!=value:raise ValueError('original schedule/output mismatch')
            candidates=[(data,loc) for data,loc in operators if data['hamiltonian']['shape']==list(h.shape)]
            candidates=[(data,loc) for data,loc in candidates if np.array_equal(self.array(data['hamiltonian'],h.shape,'complex128'),h)]
            if not candidates:raise ValueError('missing saved primary eigensystem')
            data,locator=candidates[0]
            # Array files can differ in their container encoding, so also bind H values.
            original_h=self.array(data['hamiltonian'],h.shape,'complex128')
            if not np.array_equal(original_h,h):raise ValueError('saved eigensystem/H mismatch')
            vectors=self.array(data['eigenvectors'],h.shape,'complex128')
            values=self.array(data['eigenvalues'],(basis.dimension,),'float64')
            self.solver=q.Spectral(h,self.model.error,eigenvectors=vectors,eigenvalues=values)
            if any(data[k]!=getattr(self.solver,k) for k in ('eta','residual','assembly_error')):
                raise ValueError('saved eigensystem residual/allowance mismatch')
            self.locators.append(locator)
            by_plan={value.identity:key for key,value in self.plans.items()}
            for data,locator in actions:
                payload=data['metadata'];role=by_plan.get(payload['plan'])
                if role is None:raise ValueError('foreign primary action plan')
                if role=='pure' and number==1:raise ValueError('unexpected odd pure trajectory')
                plan=self.plans[role];clock=payload['end'];key=(role,clock)
                if key in self.actions:raise ValueError('duplicate original primary clock')
                vector=self.array(data['raw_vector'],plan.initial.shape,'complex128')
                action=p.Action(vector,payload['error'],payload,self.solver,plan.initial)
                p.verify_primary(plan,action)
                self.actions[key]=action;self.locators.append(locator)
        required={(role,t) for role in (('pure','registered') if number==2 else ('registered',)) for t in m.CLOCKS}
        if set(self.actions)!=required:raise ValueError('incomplete fixed primary clock roster')
        self.locators.extend(loc for _,loc in single.values())
        if file_sha(journal)!=self.journal_sha:raise ValueError('input journal changed during intake')
        self.bind_original_context()

    def array(self,descriptor,shape,dtype):
        import numpy as np
        if set(descriptor)!={'array','shape','dtype','sha256'}:raise ValueError('invalid array descriptor')
        name=descriptor['array']
        if not re.fullmatch(r'array_[0-9]+\.npz',name):raise ValueError('foreign array path')
        path=self.path/name
        if path.resolve()!=path or descriptor['shape']!=list(shape) or descriptor['dtype']!=dtype:
            raise ValueError('array path/shape/dtype binding')
        if file_sha(path)!=descriptor['sha256']:raise ValueError('array container checksum mismatch')
        with np.load(path,allow_pickle=False) as archive:
            if archive.files!=['data']:raise ValueError('unexpected array archive members')
            array=archive['data']
        if array.shape!=tuple(shape) or str(array.dtype)!=dtype or not np.all(np.isfinite(array)):
            raise ValueError('actual array shape/dtype/finiteness mismatch')
        array.setflags(write=False)
        return array

    def bind_original_context(self):
        """Call once after verified intake (or an explicitly artificial fixture)."""
        import mock_unitary_lead_run_v1 as m
        if hasattr(self,'_original_context'):raise ValueError('original context already bound')
        self.output_maps={(first,key):tuple(value) for first in (False,True)
                          for key,value in m.outputs(self.model,first).items()}
        self._original_context=OriginalContext(str(self.path),self.journal_sha,
            self._model_identity(),self._role_identities(),
            tuple(sorted(self.output_maps.items())),tuple(sorted(self.actions)),tuple(sorted(SOURCE_SHA.items())),self._action_identities())

    def _model_identity(self):
        import mock_unitary_lead_v1 as q
        import mock_unitary_lead_protection_v1 as p
        model=self.model
        return (model.n,model.strength,model.number,q.basis_identity(self.basis),
            q.basis_identity(model.basis),q.array_identity(model.h),model.error,
            repr(p.term_record(model.terms)))

    def _role_identities(self):
        import mock_unitary_lead_v1 as q
        import mock_unitary_lead_protection_v1 as p
        return tuple(sorted((role,plan.identity,q.array_identity(plan.initial),
            plan.initial_error,plan.operator_error,plan.experiment,plan.terminal,q.basis_identity(plan.basis),repr(plan.collections),repr(plan.blocks),
            q.array_identity(plan.h),repr(p.term_record(plan.terms)))
            for role,plan in self.plans.items()))

    def _action_identities(self):
        return tuple(sorted((key,action.payload['digest'],action.payload['output'],
            action.error,action.payload['eigensystem']) for key,action in self.actions.items()))

    def verify_original_context(self):
        original=self._original_context
        if (str(self.path)!=original.path or self.journal_sha!=original.journal or
            self._model_identity()!=original.model or self._role_identities()!=original.roles or
            tuple(sorted(self.output_maps.items()))!=original.outputs or
            tuple(sorted(self.actions))!=original.clocks or tuple(sorted(SOURCE_SHA.items()))!=original.sources or
            self._action_identities()!=original.actions):
            raise ValueError('original source/role/code/operator/allowance/output roster changed')

    def snapshot(self,role,clock):
        import mock_unitary_lead_v1 as q
        import mock_unitary_lead_protection_v1 as p
        self.verify_original_context()
        if (role,clock) not in self.actions:raise ValueError('unregistered preparation/clock')
        plan=self.plans[role];action=self.actions[role,clock]
        if action.payload['end']!=clock:raise ValueError('Action end differs from requested dictionary clock')
        if (q.basis_identity(self.basis)!=q.basis_identity(plan.basis) or
            q.basis_identity(self.model.basis)!=q.basis_identity(plan.basis) or
            self.model.n*8!=self.basis.modes or self.model.number!=self.basis.number or
            q.array_identity(self.model.h)!=q.array_identity(plan.h) or
            self.model.error!=plan.operator_error or
            p.term_record(self.model.terms)!=p.term_record(plan.terms)):
            raise ValueError('saved context model/basis/operator/term substitution')
        with original_precision():p.verify_primary(plan,action)
        return q.Snapshot.make(self.basis,action.vector,plan.experiment,'primary',plan.initial,plan.identity,clock,action.error,plan.h)

    def require_snapshot(self,snapshot,role,clock):
        snapshot.verify();trusted=self.snapshot(role,clock)
        if snapshot.metadata()!=trusted.metadata():raise ValueError('detached state/code/clock/context')
        return self.actions[role,clock]

    def output(self,choice,first=False,clock=None):
        import mock_unitary_lead_run_v1 as m
        self.verify_original_context()
        if choice not in ('light','all') or type(first) is not bool:raise ValueError('fixed output choice')
        if first and (self.model.n!=3 or clock!=1.):raise ValueError('first-block output exists only at size three, clock one')
        actual=m.outputs(self.model,first)
        if set(actual)!={'light','all'}:raise ValueError('actual output roster missing or extra choice')
        output=tuple(actual[choice])
        if output!=dict(self._original_context.outputs)[first,choice]:raise ValueError('actual output map differs from original roster')
        return output
