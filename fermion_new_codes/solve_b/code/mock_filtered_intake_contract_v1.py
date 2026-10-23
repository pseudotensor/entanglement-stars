"""UNRUN production-used filtered intake/evidence/roster association validators.

No Hamiltonian, eigensystem or observable is computed here. Artificial records
can exercise these pure validators but cannot enter the real file loader.
"""
from pathlib import Path
import hashlib
import json

SINGLETONS=frozenset(('intake','model','preparations','plans','physical_job_complete',
    'termination','accepted_filtered_physical_controls'))
COMMON=Path('<internal>/MOCK_comparison_readouts_v1')
COMMON_REPORT='effb902c8310ad3e10f0a364cfcf4e4820bf9fea37fa6f652fa0bcaf85d1bac7'
COMMON_JOURNAL='a3f8a79f52474e87496c5bb2be8aac73b06ceef8c43497be142e87f9af574a33'
COMMON_REVIEW='4ca13092de3f909d2fc427a4534cdebe3a6af89f9477969d0830030b79af0a90'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_names(sources):
    result={}
    for path,sha in sources.items():
        name=Path(path).name
        if name in result:raise ValueError('duplicate source basename')
        result[name]=sha
    return result


def terminal(data):
    expected=dict(status='complete',error=None,all_pass=True,sources_unchanged=True,
        threads=1,runtime_admitted=True)
    if (any(data.get(k)!=v for k,v in expected.items()) or
        any(data.get(k) is not True for k in ('all_pass','sources_unchanged','runtime_admitted')) or
        type(data.get('threads')) is not int):raise ValueError('incomplete/error/runtime terminal')


def journal_contract(single,order,n,g):
    if type(n) is not int or type(g) not in (int,float) or n not in (2,3) or g not in (0.,.5,1.,2.):raise ValueError('fixed n2/n3 even coupling domain')
    if set(single)!=SINGLETONS or len(order)!=len(set(order)) or set(order)!=SINGLETONS:
        raise ValueError('missing/duplicate singleton roster')
    if order[-1]!='termination' or order.index('physical_job_complete')>order.index('termination'):
        raise ValueError('physical completion/terminal order')
    terminal(single['termination'])
    complete=single['physical_job_complete']
    wanted=dict(n=n,g=g,number=2,filtered=True,status='complete_fixed_readouts',n7_propagation=False)
    if (any(complete.get(k)!=v for k,v in wanted.items()) or complete.get('filtered') is not True or
        complete.get('n7_propagation') is not False or type(complete.get('n')) is not int or
        type(complete.get('number')) is not int or type(complete.get('g')) not in (int,float)):
        raise ValueError('filtered physical completion role')
    if single['intake']['sources']!=single['termination']['sources']:raise ValueError('changed producer source paths')
    source_names(single['intake']['sources'])


def model_contract(model,n,g,basis_identity,header):
    metadata=model['metadata']
    expected=dict(n=n,g=g,number=2,basis=basis_identity,profile=[.5]+[1.]*(n-1),flow=0.,a=1.,
        mass_intended='3/5',filtered=True,halo_completion=header,halo_values=['1/2','1'],n7_propagation=False)
    if (model.get('header')!=header or any(metadata.get(k)!=v for k,v in expected.items()) or
        metadata.get('filtered') is not True or metadata.get('n7_propagation') is not False or
        type(metadata.get('n')) is not int or type(metadata.get('number')) is not int or
        any(type(metadata.get(k)) not in (int,float) for k in ('g','flow','a','target_error','assembly_error'))):
        raise ValueError('filtered endpoint-copy model metadata')
    for key in ('operator','exact_target','target_error','assembly_error'):
        if key not in metadata:raise ValueError('missing original target/error identity')
    if not isinstance(model.get('exact_budget'),dict) or not model.get('intended'):
        raise ValueError('missing original exact target/budget record')
    return canonical(dict(metadata=metadata,intended=model['intended'],exact_budget=model['exact_budget']))


def action_roster(keys):
    actual=list(keys);wanted={(role,j/4) for role in ('pure','registered') for j in range(17)}
    if (len(actual)!=len(wanted) or set(actual)!=wanted or
        any(type(role) is not str or type(t) not in (int,float) for role,t in actual)):raise ValueError('missing/duplicate/extra primary role-clock')


def readout_roster(n):
    if n not in (2,3):raise ValueError('fixed output size')
    receiver=[('receiver',t,c,False) for t in (0.,1.,2.,4.) for c in ('light','all')]
    if n==3:
        at=receiver.index(('receiver',1.,'all',False))+1
        receiver[at:at]=[('receiver',1.,c,True) for c in ('light','all')]
    purity=[('purity',j/4,cut) for j in range(17) for cut in
        (('core','lead','B1','B2','A2') if n==3 else ('core','lead','B1'))]
    return receiver+purity+[('history',j/4) for j in range(4,17)]


def check_readout_roster(n,actual):
    if list(actual)!=readout_roster(n):raise ValueError('missing/duplicate/reordered readout roster')


def evidence_contract(selected,last,sources,n,g,parent_sha,journal_sha,report_sha,review_text):
    if last['kind']!='termination' or set(selected)!= {'intake','filtered_saved_intake_complete'}:
        raise ValueError('incomplete intake evidence')
    end=last['data'];terminal(end);complete=selected['filtered_saved_intake_complete']
    if (any(complete.get(k)!=v for k,v in dict(n=n,g=g,parent_journal_sha256=parent_sha,
            new_observables_evaluated=False).items()) or complete.get('new_observables_evaluated') is not False or
        type(complete.get('n')) is not int or type(complete.get('g')) not in (int,float)):raise ValueError('intake role/parent binding')
    prior=selected['intake']['sources']
    if prior!=end['sources'] or source_names(prior)!=source_names(sources):raise ValueError('intake source authority changed')
    # Review names each complete actual journal/report, tuple and source roster.
    if journal_sha not in review_text or report_sha not in review_text:
        raise ValueError('same sole assessment does not bind intake journal and report')
    binding=identity(dict(n=n,g=g,parent=parent_sha,sources=source_names(prior)))
    if binding not in review_text:raise ValueError('same sole assessment does not bind intake tuple/sources')
    return dict(journal_sha256=journal_sha,report_sha256=report_sha,binding=binding,
        n=n,g=g,parent_journal_sha256=parent_sha,sources=prior)


def common_evidence():
    """Reuse already judged actual86 controls; never run its dispatcher."""
    from mock_comparison_saved_inputs_v1 import file_sha
    report=COMMON/'controls_v1/written_report.md';journal=COMMON/'controls_v1/records.ndjson'
    review=COMMON/'reviews_controls_v1/redteam_MOCK_unitary_lead_run_v1.md'
    if (file_sha(report),file_sha(journal),file_sha(review))!=(COMMON_REPORT,COMMON_JOURNAL,COMMON_REVIEW):
        raise ValueError('frozen common86 evidence changed')
    if COMMON_REPORT not in review.read_text():raise ValueError('common86 sole review/report association')
    return dict(scope='inherited86, not a new pass count',report=str(report),report_sha256=COMMON_REPORT,
        journal=str(journal),journal_sha256=COMMON_JOURNAL,review=str(review),review_sha256=COMMON_REVIEW)


def select_primary(records,plans):
    """Metadata only:34 quarter-clock outputs and256 excluded wall records."""
    by_plan={value:key for key,value in plans.items()}
    if set(plans)!={'pure','registered'} or len(by_plan)!=2:raise ValueError('two distinct original role Plans required')
    selected=[];wall=[];keys=[];wall_keys=[]
    quarters={j/4 for j in range(17)};midpoints={(2*j+1)/128 for j in range(256)}
    for data,locator in records:
        metadata=data['metadata'];role=by_plan.get(metadata.get('plan'));t=metadata.get('end')
        if metadata.get('role')!='primary' or role is None or type(t) not in (int,float):
            raise ValueError('foreign primary category/Plan/clock type')
        if t in quarters:
            keys.append((role,t));selected.append((data,locator))
        elif role=='pure' and t in midpoints:
            wall_keys.append(t);wall.append(dict(locator=locator,plan=metadata['plan'],clock=t,
                reason='original pure wall midpoint; excluded from new-readout Action map'))
        else:raise ValueError('unrecognized primary clock/category')
    action_roster(keys)
    if len(wall_keys)!=256 or set(wall_keys)!=midpoints:raise ValueError('missing/duplicate wall metadata roster')
    return selected,wall


def require_components(actual,expected):
    import numpy as np
    if not np.array_equal(actual,expected):raise ValueError('original saved H/component association')


def require_operator(actual,expected_identity):
    import mock_unitary_lead_v1 as q
    if q.array_identity(actual)!=expected_identity:raise ValueError('original model/operator identity')


def require_plan(plan,expected_identity):
    if plan.identity!=expected_identity:raise ValueError('original role Plan membership')


def supplied_solver(loader,record,h,error):
    """Only residual audit of supplied original eigensystem; never eigh."""
    import mock_unitary_lead_v1 as q
    actual=loader.array(record['hamiltonian'],h.shape,'complex128')
    require_components(actual,h)
    vectors=loader.array(record['eigenvectors'],h.shape,'complex128')
    values=loader.array(record['eigenvalues'],(len(h),),'float64')
    solver=q.Spectral(h,error,eigenvectors=vectors,eigenvalues=values)
    if any(record[k]!=getattr(solver,k) for k in ('eta','residual','assembly_error')):
        raise ValueError('saved eigensystem residual/allowance mismatch')
    return solver


def supplied_action(plan,solver,metadata,vector):
    import mock_unitary_lead_protection_v1 as p
    action=p.Action(vector,metadata['error'],metadata,solver,plan.initial)
    p.verify_primary(plan,action)
    return action


def read_evidence(directory,review,review_sha,save=None):
    """Bind the exact parsed journal and verified review text to stable bytes."""
    from mock_comparison_saved_inputs_v1 import file_sha
    directory=Path(directory);review=Path(review);journal=directory/'records.ndjson';report=directory/'written_report.md'
    before=file_sha(journal);report_before=file_sha(report);review_before=file_sha(review)
    rows=[];parsed=hashlib.sha256()
    with journal.open('rb') as stream:
        for line in stream:
            parsed.update(line);rows.append(json.loads(line))
    review_bytes=review.read_bytes();text=review_bytes.decode('utf-8')
    hashes=dict(journal_before=before,journal_parsed=parsed.hexdigest(),journal_after=file_sha(journal),
        report_before=report_before,report_after=file_sha(report),review_before=review_before,
        review_parsed=hashlib.sha256(review_bytes).hexdigest(),review_after=file_sha(review),review_expected=review_sha)
    if save:save('filtered_evidence_read_bytes',dict(directory=str(directory),review=str(review),**hashes))
    if not (hashes['journal_before']==hashes['journal_parsed']==hashes['journal_after'] and
        hashes['report_before']==hashes['report_after'] and
        hashes['review_before']==hashes['review_parsed']==hashes['review_after']==review_sha):
        raise ValueError('evidence bytes changed across parse: '+canonical(hashes))
    if not rows or rows[-1]['kind']!='termination' or any(r['kind']=='termination' for r in rows[:-1]):
        raise ValueError('evidence terminal must be unique EOF')
    return dict(rows=rows,journal_sha256=before,report_sha256=report_before,
        review_sha256=review_sha,review_text=text,byte_association=hashes)
