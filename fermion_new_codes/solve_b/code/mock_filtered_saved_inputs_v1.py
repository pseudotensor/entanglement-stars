"""UNRUN filtered saved-state intake draft; no entry point or admission.

Only the already specified n2/n3 even cohorts, cuts, clocks and code are
admitted by this definition. The required journal SHA comes from the retained
own physical run, whose complete actual assessment is a prerequisite. No new
trajectory, eigensolve, conversion, entropy, D_RK or wall integral is evaluated.
Source/control review and retained-data artificial controls remain pending.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import re
import mock_filtered_clockwell_v1 as clock
from mock_comparison_saved_inputs_v1 import SavedBaseline, OriginalContext, SOURCE_SHA, file_sha, original_precision

import mock_filtered_intake_contract_v1 as contract

BASE=Path('<internal>/MOCK_filtered_physical_v1')


def _physical_sources(sources,accepted):
    import run_mock_filtered_physical_v1 as physical
    root=Path(physical.__file__).parent
    paths=physical.source_paths(root/'MOCK_filtered_clockwell_spec_v1.md',
        root/'MOCK_filtered_physical_controls_spec_v1.md',root/'MOCK_unitary_lead_run_spec_v1.md',
        root/'MOCK_filtered_readout_addendum_v1.md')
    expected={path.name:file_sha(path) for path in paths}
    review=Path(accepted['review'])
    if file_sha(review)!=accepted['review_sha256']:raise ValueError('retained actual-control review changed')
    if accepted['written_report_sha256'] not in review.read_text():raise ValueError('retained control report/review association')
    if review.name in expected:raise ValueError('duplicate scientific source basename')
    gate=physical.control_gate(accepted['controls'],review,accepted['review_sha256'],paths)
    if gate!=accepted:raise ValueError('actual control gate differs from original accepted evidence')
    expected[review.name]=accepted['review_sha256']
    if len(sources)!=len(expected) or contract.source_names(sources)!=expected:
        raise ValueError('filtered physical producer/source/control/spec identities differ')


class SavedFiltered(SavedBaseline):
    """Shared frozen association/array methods, separate strict physical intake."""
    def __init__(self,n,g,expected_journal_sha):
        number=2
        import mock_comparison_saved_inputs_v1 as saved
        if file_sha(saved.__file__)!='87e2abe63f71079514daaae627403f1e54f9b9f7d64e2c80fbedd071d6642c61':raise ValueError('shared original-Action/array association code changed')
        if not re.fullmatch(r'[0-9a-f]{64}',expected_journal_sha):raise ValueError('explicit retained physical journal identity required')
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
        label={0.:'0',1.:'1',.5:'0p5',2.:'2'}[g]
        self.path=BASE/f'even_n{n}_g{label}_v1'
        if self.path.resolve()!=self.path:raise ValueError('baseline path alias')
        self.locators=[];self.reused_readouts=[];single={};operators=[];actions=[];order=[];last_kind=None
        wanted={'intake','model','preparations','plans','physical_job_complete','termination','accepted_filtered_physical_controls'}
        reused={'entropy','register','conversion_status','purification','purification_status','wall_complete'}
        journal=self.path/'records.ndjson';digest=hashlib.sha256()
        with journal.open('rb') as stream:
            for line_number,line in enumerate(stream,1):
                digest.update(line)
                match=re.match(rb'\{"kind":"([^"]+)"',line)
                if not match:raise ValueError('noncanonical original journal')
                kind=match[1].decode('ascii')
                if last_kind=='termination':raise ValueError('record after termination')
                last_kind=kind
                if kind in reused:
                    self.reused_readouts.append({'line':line_number,'kind':kind,'row_sha256':hashlib.sha256(line).hexdigest()})
                    continue
                if kind not in wanted|{'numerical_action','action'}:continue
                row=json.loads(line);data=row['data']
                locator={'line':line_number,'kind':kind,'row_sha256':hashlib.sha256(line).hexdigest()}
                if kind in wanted:
                    if kind in single:raise ValueError('duplicate singleton record')
                    single[kind]=(data,locator);order.append(kind)
                elif kind=='numerical_action' and data.get('kind')=='operator':operators.append((data,locator))
                elif kind=='action' and data['metadata'].get('role')=='primary':
                    actions.append((data,locator))
        contract.journal_contract({key:value[0] for key,value in single.items()},order,n,g)
        self.journal_sha=digest.hexdigest()
        if self.journal_sha!=expected_journal_sha:raise ValueError('frozen filtered journal identity changed')
        intake=single['intake'][0];terminal=single['termination'][0]
        accepted=single['accepted_filtered_physical_controls'][0]
        _physical_sources(intake['sources'],accepted);_physical_sources(terminal['sources'],accepted)
        if intake['sources']!=terminal['sources']:raise ValueError('changed source paths')
        self.physical_sources=dict(intake['sources'])
        if (terminal['status']!='complete' or terminal['all_pass'] is not True or
            terminal['sources_unchanged'] is not True or terminal['threads']!=1 or terminal['runtime_admitted'] is not True):
            raise ValueError('physical runtime/termination not admitted')
        complete=single['physical_job_complete'][0]
        if any(complete[k]!=v for k,v in {'n':n,'g':g,'number':number,'filtered':True,'status':'complete_fixed_readouts'}.items()):
            raise ValueError('original physical job role mismatch')
        model=single['model'][0];metadata=model['metadata'];basis=q.Basis.make(8*n,number)
        self.filtered_metadata=contract.model_contract(model,n,g,q.basis_identity(basis),clock.TABLE_HEADER)
        self.exact_budget_locator=single['model'][1]
        self.original_model_record=model
        expected={'n':n,'g':g,'number':number,'basis':q.basis_identity(basis),'profile':[.5]+[1.]*(n-1),
                  'flow':0.,'a':1.,'mass_intended':'3/5','filtered':True,
                  'halo_completion':clock.TABLE_HEADER,'halo_values':['1/2','1']}
        if any(metadata[k]!=v for k,v in expected.items()):raise ValueError('filtered model context mismatch')
        h=self.array(model['actual_H'],(basis.dimension,basis.dimension),'complex128')
        contract.require_operator(h,metadata['operator'])
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
            for role,prefix in (('pure','filtered_pure_'),('registered','filtered_register_')):
                plan=runner.make_plan(self.model,prep[role],prep[role+'_error'],prefix+str((n,float(g),number)))
                contract.require_plan(plan,plans[role])
                self.plans[role]=plan
            for key,value in {'collections':list(plan.collections),'blocks':[list(x) for x in plan.blocks],
                              'primary_clocks':list(m.CLOCKS),'outputs':{k:list(v) for k,v in m.outputs(self.model).items()},
                              'first_outputs':{k:list(v) for k,v in m.outputs(self.model,True).items()}}.items():
                if plans[key]!=value:raise ValueError('original schedule/output mismatch')
            candidates=[(data,loc) for data,loc in operators if data['hamiltonian']['shape']==list(h.shape)]
            matching=[]
            for candidate,location in candidates:
                candidate_h=self.array(candidate['hamiltonian'],h.shape,'complex128')
                try:contract.require_components(candidate_h,h)
                except ValueError:continue
                matching.append((candidate,location))
            candidates=matching
            if not candidates:raise ValueError('missing saved primary eigensystem')
            data,locator=candidates[0]
            self.solver=contract.supplied_solver(self,data,h,self.model.error)
            self.locators.append(locator)
            by_plan={value.identity:key for key,value in self.plans.items()}
            selected,self.excluded_wall_actions=contract.select_primary(actions,{role:plan.identity for role,plan in self.plans.items()})
            self.locators.extend(row['locator'] for row in self.excluded_wall_actions)
            for data,locator in selected:
                payload=data['metadata'];role=by_plan.get(payload['plan'])
                if role is None:raise ValueError('foreign primary action plan')
                if role=='pure' and number==1:raise ValueError('unexpected odd pure trajectory')
                plan=self.plans[role];action_clock=payload['end'];key=(role,action_clock)
                if key in self.actions:raise ValueError('duplicate original primary clock')
                vector=self.array(data['raw_vector'],plan.initial.shape,'complex128')
                action=contract.supplied_action(plan,self.solver,payload,vector)
                self.actions[key]=action;self.locators.append(locator)
        required={(role,t) for role in (('pure','registered') if number==2 else ('registered',)) for t in m.CLOCKS}
        contract.action_roster(self.actions)
        self.locators.extend(loc for _,loc in single.values())
        if file_sha(journal)!=self.journal_sha:raise ValueError('input journal changed during intake')
        self.bind_original_context()


    def __setattr__(self,name,value):
        if name=='_filtered_authority' and hasattr(self,name):raise ValueError('immutable filtered target authority')
        super().__setattr__(name,value)

    def bind_original_context(self):
        """Bind the actual verified physical source roster, not baseline labels."""
        import mock_unitary_lead_run_v1 as m
        if hasattr(self,'_original_context'):raise ValueError('original filtered context already bound')
        self.output_maps={(first,key):tuple(value) for first in (False,True)
            for key,value in m.outputs(self.model,first).items()}
        self._filtered_authority=contract.canonical(dict(metadata=self.filtered_metadata,budget_locator=self.exact_budget_locator))
        self._original_context=OriginalContext(str(self.path),self.journal_sha,self._model_identity(),
            self._role_identities(),tuple(sorted(self.output_maps.items())),tuple(sorted(self.actions)),
            tuple(sorted(self.physical_sources.items())),self._action_identities())

    def verify_original_context(self):
        original=self._original_context
        if contract.canonical(dict(metadata=self.filtered_metadata,budget_locator=self.exact_budget_locator))!=self._filtered_authority:
            raise ValueError('original filtered target/metadata authority changed')
        if contract.canonical(dict(metadata=self.original_model_record['metadata'],intended=self.original_model_record['intended'],exact_budget=self.original_model_record['exact_budget']))!=self.filtered_metadata:
            raise ValueError('working filtered metadata/exact budget differs from original authority')
        if (str(self.path)!=original.path or self.journal_sha!=original.journal or
            self._model_identity()!=original.model or self._role_identities()!=original.roles or
            tuple(sorted(self.output_maps.items()))!=original.outputs or tuple(sorted(self.actions))!=original.clocks or
            tuple(sorted(self.physical_sources.items()))!=original.sources or self._action_identities()!=original.actions):
            raise ValueError('original filtered physical source/role/code/clock/output/Action authority changed')
