"""UNRUN filtered saved-state intake/new-readout programme draft.

No execution admission. The mathematical readouts remain the frozen common
receiver/purity/history primitives; only physical input intake is new. The
physical producer and this adapter each need their own complete source and
actual-input/control reading. No parent trajectory or old metric is replayed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import mock_filtered_intake_contract_v1 as contract

FROZEN={
 'mock_comparison_readouts_v1.py':'038def34491d873ea7d54c4a8ef837ab26a939ba7f515cb8d1f26a3028fb4a2d',
 'mock_comparison_saved_inputs_v1.py':'87e2abe63f71079514daaae627403f1e54f9b9f7d64e2c80fbedd071d6642c61',
 'mock_comparison_return_v1.py':'31421d64d08a4c9755d8ff428582a363c31dee3a0b75998f397ea35cd1e6ebd9',
 'mock_comparison_controls_v1.py':'73456cc8e2d242719351ebe4948a6189578556d8c2f065355b63caeedf2b8cb6',
 'run_mock_comparison_readouts_v1.py':'585093d4db46c214c4381e9f1ee86acbca5931a823f9ba5211eeae400fe9f53c',
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(args):
    import run_mock_filtered_physical_v1 as physical
    root=Path(__file__).parent
    paths=physical.source_paths(args.spec,args.control_spec,args.baseline_spec,args.readout_spec)
    for name,expected in FROZEN.items():
        path=root/name
        if digest(path)!=expected:raise ValueError('previously admitted common readout source changed')
        paths.append(path)
    paths.extend((Path(__file__),root/'mock_filtered_saved_inputs_v1.py',root/'mock_filtered_intake_contract_v1.py',root/'mock_filtered_comparison_controls_v1.py'))
    if len({p.name for p in paths})!=len(paths):raise ValueError('duplicate input source basename')
    return paths


def intake(n,g,parent_sha,bundle):
    from mock_filtered_saved_inputs_v1 import SavedFiltered
    bundle.save('reused_common86_evidence',contract.common_evidence())
    context=bundle.phase('filtered_retained_input_intake',lambda:SavedFiltered(n,g,parent_sha))
    bundle.save('filtered_saved_intake_complete',dict(n=n,g=g,parent_journal_sha256=parent_sha,
        parent_journal=str(context.path/'records.ndjson'),locators=context.locators,
        intake_binding=contract.identity(dict(n=n,g=g,parent=parent_sha,sources=contract.source_names(bundle.identities))),
        filtered_metadata=context.filtered_metadata,exact_budget_locator=context.exact_budget_locator,
        actual_sources=context.physical_sources,authority=context._filtered_authority,
        original_role_plans={role:plan.identity for role,plan in context.plans.items()},
        actual_action_roster=tuple(sorted(context.actions)),excluded_wall_actions=context.excluded_wall_actions,
        new_observables_evaluated=False,
        scope='retained own physical model/code/source/action/eigensystem association only'))
    return context


def intake_gate(directory,review,review_sha,sources,n,g,parent_sha):
    directory=Path(directory);review=Path(review)
    evidence=contract.read_evidence(directory,review,review_sha)
    selected={}
    for row in evidence['rows']:
        if row['kind'] in ('intake','filtered_saved_intake_complete'):
            if row['kind'] in selected:raise ValueError('duplicate intake identity/completion')
            selected[row['kind']]=row['data']
    current={str(p):digest(p) for p in sources}
    accepted=contract.evidence_contract(selected,evidence['rows'][-1],current,n,g,parent_sha,
        evidence['journal_sha256'],evidence['report_sha256'],evidence['review_text'])
    if digest(directory/'records.ndjson')!=evidence['journal_sha256']:raise ValueError('intake journal changed during gate')
    return dict(accepted,intake=str(directory),review=str(review),review_sha256=review_sha,
        byte_association=evidence['byte_association'])



def adapter_control_gate(directory,review,review_sha,sources):
    directory=Path(directory);review=Path(review)
    evidence=contract.read_evidence(directory,review,review_sha);rows=evidence['rows']
    if not rows or rows[-1]['kind']!='termination':raise ValueError('unfinished adapter controls')
    contract.terminal(rows[-1]['data'])
    intake=[r['data'] for r in rows if r['kind']=='intake']
    complete=[r['data'] for r in rows if r['kind']=='filtered_adapter_controls_complete']
    if len(intake)!=1 or len(complete)!=1 or complete[0]!=dict(groups=['contract','subclass','authority','evidence'],physical_run=False):
        raise ValueError('adapter control groups incomplete')
    if (intake[0]['sources']!=rows[-1]['data']['sources'] or contract.source_names(intake[0]['sources'])!=
            contract.source_names({str(p):digest(p) for p in sources})):
        raise ValueError('adapter source differs from complete controls')
    journal_sha=evidence['journal_sha256'];report_sha=evidence['report_sha256']
    text=evidence['review_text']
    if journal_sha not in text or report_sha not in text:raise PermissionError('same sole actual controls not bound')
    if digest(directory/'records.ndjson')!=journal_sha:raise ValueError('adapter controls changed during gate')
    return dict(byte_association=evidence['byte_association'],controls=str(directory),journal_sha256=journal_sha,report_sha256=report_sha,
        review=str(review),review_sha256=review_sha)


def new_readouts(n,g,parent_sha,bundle):
    number=2
    import mock_filtered_saved_inputs_v1 as saved
    import mock_comparison_readouts_v1 as readout
    import mock_comparison_return_v1 as history
    from mock_comparison_controls_v1 import record_value
    actual=[]
    context=bundle.phase('saved_filtered_intake',lambda:saved.SavedFiltered(n,g,parent_sha))
    bundle.save('saved_input_context',dict(original_journal=str(context.path/'records.ndjson'),
        original_journal_sha256=context.journal_sha,input_locators=context.locators,
        reused_readout_locators=context.reused_readouts,
        old_quantities='reuse by source-bound locator only; no propagation/conversion/entropy/D_RK/wall replay',
        model=dict(n=n,g=g,number=number,filtered=True),addendum=readout.ADDENDUM_SHA))
    for clock in (0.,1.,2.,4.):
        snapshot=context.snapshot('registered',clock)
        bundle.save('receiver_saved_input',dict(snapshot=snapshot,clock=clock))
        for choice in ('light','all'):
            result=bundle.phase('new_receiver_'+str((clock,choice)),
                lambda choice=choice:readout.receiver_from_saved(context,snapshot,clock,choice,save=bundle.save))
            bundle.save('new_fixed_receiver',record_value(result));actual.append(('receiver',clock,choice,False))
        if n==3 and clock==1.:
            for choice in ('light','all'):
                result=bundle.phase('new_first_receiver_'+choice,
                    lambda choice=choice:readout.receiver_from_saved(context,snapshot,clock,choice,True,bundle.save))
                bundle.save('new_fixed_receiver',record_value(result));actual.append(('receiver',clock,choice,True))
    if number==2:
        for clock in tuple(j/4 for j in range(17)):
            snapshot=context.snapshot('pure',clock)
            bundle.save('purity_saved_input',dict(snapshot=snapshot,clock=clock))
            for cut in (('core','lead','B1','B2','A2') if n==3 else ('core','lead','B1')):
                result=bundle.phase('new_purity_'+str((clock,cut)),lambda cut=cut:readout.purity_from_saved(context,snapshot,clock,cut,bundle.save))
                bundle.save('new_pure_cut_purity',result);actual.append(('purity',clock,cut))
        for clock in tuple(j/4 for j in range(4,17)):
            bundle.phase('new_return_'+str(clock),lambda clock=clock:history.return_from_saved(context,clock,bundle.save));actual.append(('history',clock))
    bundle.save('actual_new_readout_roster',dict(n=n,actual=actual,expected=contract.readout_roster(n)))
    contract.check_readout_roster(n,actual)
    # All original files must retain the admitted journal binding after use.
    if saved.file_sha(context.path/'records.ndjson')!=context.journal_sha:raise ValueError('original input journal changed during new readouts')
    bundle.save('comparison_job_complete',dict(n=n,g=g,number=number,filtered=True,
        status='new_filtered_fixed_readouts_complete',old_observables_recomputed=False,new_return_branch=number==2,
        scope='new named readouts of retained fixed filtered programme; no rate, chronology or source'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('controls','intake','readouts'))
    for name in ('spec','control-spec','baseline-spec','readout-spec','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--n',type=int);parser.add_argument('--g',type=float)
    parser.add_argument('--parent-journal-sha')
    parser.add_argument('--controls-output');parser.add_argument('--control-review');parser.add_argument('--control-review-sha')
    parser.add_argument('--intake-output');parser.add_argument('--intake-review');parser.add_argument('--intake-review-sha')
    args=parser.parse_args()
    import mock_unitary_lead_run_v1 as baseline
    if args.stage!='controls':baseline.domain(args.n,args.g,2)
    paths=source_paths(args);output=Path(args.output);base=Path('<internal>/MOCK_filtered_comparison_readouts_v1')
    if not output.is_absolute() or output.parent!=base or output.resolve()!=output:raise ValueError('fresh direct filtered-readout box child required')
    gate=None;control=None
    if args.stage!='controls':
        if not all((args.controls_output,args.control_review,args.control_review_sha,args.parent_journal_sha)):
            raise PermissionError('complete same-pair actual adapter controls and physical parent required')
        control=adapter_control_gate(args.controls_output,args.control_review,args.control_review_sha,paths)
    if args.stage=='readouts':
        if not all((args.intake_output,args.intake_review,args.intake_review_sha)):raise PermissionError('complete original-pair actual-intake assessment required')
        gate=intake_gate(args.intake_output,args.intake_review,args.intake_review_sha,paths+[Path(args.control_review)],args.n,args.g,args.parent_journal_sha)
    from mock_unitary_lead_records_v1 import Bundle
    import mock_unitary_lead_v1 as q
    bundle=Bundle(output,paths+([Path(args.intake_review)] if gate else [])+([Path(args.control_review)] if control else []),570)
    q.set_recorder(lambda row:bundle.save('numerical_action',row))
    try:
        if control:bundle.save('accepted_filtered_adapter_controls',control)
        if gate:bundle.save('accepted_filtered_saved_intake',gate)
        if args.stage=='controls':
            import mock_filtered_comparison_controls_v1 as tests
            bundle.save('reused_common86_evidence',contract.common_evidence())
            tests.controls(bundle)
        elif args.stage=='intake':intake(args.n,args.g,args.parent_journal_sha,bundle)
        else:new_readouts(args.n,args.g,args.parent_journal_sha,bundle)
        bundle.finish('complete')
    except BaseException as error:
        bundle.finish('incomplete',dict(exception=type(error).__name__,reason=str(error)))
        raise


if __name__=='__main__':main()
