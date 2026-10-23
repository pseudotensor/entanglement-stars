"""UNRUN source candidate: new size-two/three readouts of frozen baseline data.

Only new receiver/purity readouts and the defined return branch are evaluated.
No original trajectory, conversion, entropy, D_RK or wall calculation is run.
No filtered/n7 mechanism entry point exists in this source candidate.
Complete original-pair source/control assessment is required before use.
"""
from pathlib import Path
import argparse
import json


def source_paths(spec):
    import mock_comparison_saved_inputs_v1 as saved
    from mock_comparison_readouts_v1 import ADDENDUM_SHA
    spec=Path(spec)
    if saved.file_sha(spec)!=ADDENDUM_SHA:raise ValueError('fixed mathematical addendum identity')
    names=('run_mock_comparison_readouts_v1.py','mock_comparison_readouts_v1.py',
           'mock_comparison_saved_inputs_v1.py','mock_comparison_return_v1.py','mock_comparison_controls_v1.py',
           'run_mock_unitary_lead_run_v1.py','mock_unitary_lead_run_v1.py','mock_unitary_lead_protection_v1.py',
           'mock_unitary_lead_records_v1.py','mock_unitary_lead_v1.py','mock_axial_vertex_adapter_v1.py',
           'run_mock_vertex_admission_v1.py','cpu_runtime.py')
    paths=[Path(__file__).with_name(name) for name in names]+[spec]
    for path in paths:
        if path.name in saved.SOURCE_SHA and saved.file_sha(path)!=saved.SOURCE_SHA[path.name]:
            raise ValueError('original dependency differs from frozen baseline source')
    return paths


GROUPS=['fixed_decoder','pure_cut_purity','odd_decoder','context_joins','array_intake','two_time_return','production_wrappers','retained_input_intake']


def controls(bundle):
    import mock_comparison_controls_v1 as tests
    bundle.phase('fixed_decoder',lambda:tests.decoder_controls(bundle))
    bundle.phase('pure_cut_purity',lambda:tests.purity_controls(bundle))
    bundle.phase('odd_decoder',lambda:tests.odd_decoder_control(bundle))
    context=bundle.phase('context_joins',lambda:tests.context_controls(bundle))
    bundle.phase('array_intake',lambda:tests.array_intake_controls(bundle))
    bundle.phase('two_time_return',lambda:tests.return_controls(bundle,context))
    bundle.phase('production_wrappers',lambda:tests.production_wrapper_controls(bundle))
    bundle.phase('retained_input_intake',lambda:tests.retained_input_intake_controls(bundle))
    bundle.save('comparison_controls_complete',dict(groups=GROUPS,physical_run=False,
        scope='new baseline size-two/three readouts; no filtered or size-seven path'))


def control_gate(directory,review,review_sha,sources):
    import mock_comparison_saved_inputs_v1 as saved
    directory=Path(directory);review=Path(review)
    if not directory.is_dir() or not review.is_file():raise PermissionError('missing complete controls or same sole review')
    if saved.file_sha(review)!=review_sha:raise PermissionError('review identity changed')
    rows=[]
    with (directory/'records.ndjson').open() as stream:
        for line in stream:rows.append(json.loads(line))
    if not rows or rows[-1]['kind']!='termination':raise PermissionError('unfinished controls')
    terminal=rows[-1]['data']
    if (terminal['status']!='complete' or terminal['all_pass'] is not True or terminal['sources_unchanged'] is not True or
        terminal['threads']!=1 or terminal['runtime_admitted'] is not True):raise PermissionError('controls not admitted')
    complete=[row['data'] for row in rows if row['kind']=='comparison_controls_complete']
    if len(complete)!=1 or complete[0]['groups']!=GROUPS or complete[0]['physical_run'] is not False:
        raise PermissionError('missing fixed control groups')
    intake=[row['data'] for row in rows if row['kind']=='intake']
    if len(intake)!=1 or intake[0]['sources']!=terminal['sources']:raise PermissionError('control source identities changed')
    prior={Path(k).name:v for k,v in intake[0]['sources'].items()}
    current={Path(path).name:saved.file_sha(path) for path in sources}
    if len(prior)!=len(intake[0]['sources']) or prior!=current:raise PermissionError('new readout sources differ from controls')
    report_sha=saved.file_sha(directory/'written_report.md')
    if report_sha not in review.read_text():raise PermissionError('sole review does not bind complete actual controls')
    return dict(controls=str(directory),written_report_sha256=report_sha,review=str(review),review_sha256=review_sha)


def new_readouts(n,g,number,bundle):
    import mock_comparison_saved_inputs_v1 as saved
    import mock_comparison_readouts_v1 as readout
    import mock_comparison_return_v1 as history
    from mock_comparison_controls_v1 import record_value
    context=bundle.phase('saved_baseline_intake',lambda:saved.SavedBaseline(n,g,number))
    bundle.save('saved_input_context',dict(original_journal=str(context.path/'records.ndjson'),
        original_journal_sha256=context.journal_sha,input_locators=context.locators,
        reused_readout_locators=context.reused_readouts,
        old_quantities='reuse by source-bound locator only; no propagation/conversion/entropy/D_RK/wall replay',
        model=dict(n=n,g=g,number=number,filtered=False),addendum=readout.ADDENDUM_SHA))
    for clock in (0.,1.,2.,4.):
        snapshot=context.snapshot('registered',clock)
        bundle.save('receiver_saved_input',dict(snapshot=snapshot,clock=clock))
        for choice in ('light','all'):
            result=bundle.phase('new_receiver_'+str((clock,choice)),
                lambda choice=choice:readout.receiver_from_saved(context,snapshot,clock,choice,save=bundle.save))
            bundle.save('new_fixed_receiver',record_value(result))
        if n==3 and clock==1.:
            for choice in ('light','all'):
                result=bundle.phase('new_first_receiver_'+choice,
                    lambda choice=choice:readout.receiver_from_saved(context,snapshot,clock,choice,True,bundle.save))
                bundle.save('new_fixed_receiver',record_value(result))
    if number==2:
        for clock in tuple(j/4 for j in range(17)):
            snapshot=context.snapshot('pure',clock)
            bundle.save('purity_saved_input',dict(snapshot=snapshot,clock=clock))
            for cut in (('core','lead','B1','B2','A2') if n==3 else ('core','lead','B1')):
                result=bundle.phase('new_purity_'+str((clock,cut)),lambda cut=cut:readout.purity_from_saved(context,snapshot,clock,cut,bundle.save))
                bundle.save('new_pure_cut_purity',result)
        for clock in tuple(j/4 for j in range(4,17)):
            bundle.phase('new_return_'+str(clock),lambda clock=clock:history.return_from_saved(context,clock,bundle.save))
    # All original files must retain the admitted journal binding after use.
    if saved.file_sha(context.path/'records.ndjson')!=context.journal_sha:raise ValueError('original input journal changed during new readouts')
    bundle.save('comparison_job_complete',dict(n=n,g=g,number=number,filtered=False,
        status='new_fixed_readouts_complete',old_observables_recomputed=False,new_return_branch=number==2,
        scope='new named readouts of original fixed baseline; no rate, chronology or source'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('controls','readouts'))
    parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--n',type=int);parser.add_argument('--g',type=float);parser.add_argument('--number',type=int)
    parser.add_argument('--controls-output');parser.add_argument('--control-review');parser.add_argument('--control-review-sha')
    args=parser.parse_args();sources=source_paths(args.spec)
    output=Path(args.output)
    base=Path('<internal>/MOCK_comparison_readouts_v1')
    if not output.is_absolute() or output.parent!=base or output.resolve()!=output:raise ValueError('new readout output must be a direct fresh box-bulk child')
    gate=None
    if args.stage=='readouts':
        import mock_unitary_lead_run_v1 as m
        m.domain(args.n,args.g,args.number)
        if args.number==1 and args.g!=0:raise ValueError('shared odd state only')
        if not all((args.controls_output,args.control_review,args.control_review_sha)):raise PermissionError('same original pair actual-control assessment is required')
        gate=control_gate(args.controls_output,args.control_review,args.control_review_sha,sources)
    from mock_unitary_lead_records_v1 import Bundle
    import mock_unitary_lead_v1 as q
    bundle=Bundle(output,sources+([Path(args.control_review)] if gate else []),600)
    q.set_recorder(lambda row:bundle.save('numerical_action',row))
    try:
        if gate:bundle.save('accepted_comparison_controls',gate)
        if args.stage=='controls':controls(bundle)
        else:new_readouts(args.n,args.g,args.number,bundle)
        bundle.finish('complete')
    except BaseException as error:
        bundle.finish('incomplete',dict(exception=type(error).__name__,reason=str(error)))
        raise


if __name__=='__main__':main()
