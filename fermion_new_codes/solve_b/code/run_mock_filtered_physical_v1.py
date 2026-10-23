"""UNRUN gated n2/n3 endpoint-copy-halo physical programme candidate.

No closed static roster is replayed. Fresh controls and their complete actual
original-pair assessments precede the physical stage. No n7 propagation.
"""
from pathlib import Path
import argparse
import hashlib
import json

CONTROL_SPEC='c59de540d06d73cd095b3a312976f6d0e5a354d8a01f6bb81e71a6bb72909b21'
GROUPS=('physical_adapters','stationary_wall','tiny_certified_collection')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(spec,control_spec,baseline_spec,readout_spec):
    expected=('f3ea645a0835ce1cc23356e2ef336b47961ff1c2687ca072a93e4bf58f9bd92a',CONTROL_SPEC,
        'f2f6f1d1052f79a3c41a5a15849037566c6c05690bb543281e217bee5f1f6ab2',
        'f436ecbf81963df9c6695438c27e5ebf3ec4deb56e02b5917688c44b38b2c1ff')
    specs=tuple(Path(x) for x in (spec,control_spec,baseline_spec,readout_spec))
    if tuple(digest(path) for path in specs)!=expected:raise ValueError('fixed physical/control protocol identities')
    names=('run_mock_filtered_physical_v1','mock_filtered_physical_model_v1','mock_filtered_physical_source_v1',
        'mock_filtered_physical_protection_v1','mock_filtered_physical_wall_v1','mock_filtered_physical_producer_v1',
        'mock_filtered_physical_controls_v1','mock_filtered_wall_core_v1','mock_filtered_wall_controls_v1',
        'mock_filtered_collection_core_v1','mock_filtered_collection_controls_v1','mock_filtered_clockwell_v1',
        'mock_filtered_joining_v1','mock_filtered_source_context_v1','mock_filtered_static_associations_v1',
        'mock_filtered_kernel_v1','mock_filtered_vertex_v1','run_mock_unitary_lead_run_v1',
        'mock_unitary_lead_run_v1','mock_unitary_lead_protection_v1','mock_unitary_lead_records_v1',
        'mock_unitary_lead_v1','mock_axial_vertex_adapter_v1','run_mock_vertex_admission_v1','cpu_runtime')
    return [Path(__file__).with_name(name+'.py') for name in names]+list(specs)


def controls(bundle):
    import mock_filtered_physical_controls_v1 as adapters
    import mock_filtered_wall_controls_v1 as wall
    import mock_filtered_collection_controls_v1 as collection
    bundle.phase('physical_adapters',lambda:adapters.controls(bundle))
    bundle.phase('stationary_wall',lambda:wall.stationary_controls(bundle))
    bundle.phase('tiny_certified_collection',lambda:collection.collection_controls(bundle))
    bundle.save('filtered_physical_controls_complete',dict(groups=GROUPS,control_spec=CONTROL_SPEC,physical_propagation=False))


def control_gate(directory,review,review_sha,sources):
    directory=Path(directory);review=Path(review)
    if not directory.is_dir() or not review.is_file() or digest(review)!=review_sha:raise PermissionError('complete actual controls and same sole scientific review required')
    selected={};last=None
    with (directory/'records.ndjson').open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row['kind'] in ('intake','filtered_physical_controls_complete'):
                if row['kind'] in selected:raise PermissionError('duplicate control identity/completion')
                selected[row['kind']]=row['data']
    if not last or last['kind']!='termination':raise PermissionError('unfinished manufactured stage')
    end=last['data'];complete=selected.get('filtered_physical_controls_complete',{})
    if (end['status']!='complete' or end['error'] is not None or end['all_pass'] is not True or
        end['sources_unchanged'] is not True or end['threads']!=1 or end['runtime_admitted'] is not True or
        complete.get('groups')!=list(GROUPS) or complete.get('control_spec')!=CONTROL_SPEC or complete.get('physical_propagation') is not False):
        raise PermissionError('manufactured stage/context/runtime not admitted')
    prior=selected['intake']['sources'];current={str(path):digest(path) for path in sources}
    if prior!=end['sources'] or {Path(k).name:v for k,v in prior.items()}!={Path(k).name:v for k,v in current.items()}:
        raise PermissionError('physical source differs from complete controls')
    report_sha=digest(directory/'written_report.md')
    if report_sha not in review.read_text():raise PermissionError('scientific review does not bind complete actual controls')
    return dict(controls=str(directory),written_report_sha256=report_sha,review=str(review),review_sha256=review_sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('controls','physical'))
    for name in ('spec','control-spec','baseline-spec','readout-spec','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--n',type=int);parser.add_argument('--g',type=float)
    parser.add_argument('--controls-output');parser.add_argument('--control-review');parser.add_argument('--control-review-sha')
    args=parser.parse_args();sources=source_paths(args.spec,args.control_spec,args.baseline_spec,args.readout_spec)
    output=Path(args.output);base=Path('<internal>/MOCK_filtered_physical_v1')
    if not output.is_absolute() or output.parent!=base or output.resolve()!=output:raise ValueError('fresh direct box-bulk child required')
    gate=None
    if args.stage=='physical':
        import mock_unitary_lead_run_v1 as baseline
        baseline.domain(args.n,args.g,2)
        if not all((args.controls_output,args.control_review,args.control_review_sha)):raise PermissionError('same original-pair actual control assessment required')
        gate=control_gate(args.controls_output,args.control_review,args.control_review_sha,sources)
    from mock_unitary_lead_records_v1 import Bundle
    import mock_unitary_lead_v1 as q
    bundle=Bundle(output,sources+([Path(args.control_review)] if gate else []),570 if args.stage=='controls' else 3500)
    q.set_recorder(lambda row:bundle.save('numerical_action',row))
    try:
        if gate:bundle.save('accepted_filtered_physical_controls',gate)
        if args.stage=='controls':controls(bundle)
        else:
            import mock_filtered_physical_producer_v1 as producer
            producer.physical_job(args.n,args.g,bundle)
        bundle.finish('complete')
    except BaseException as error:
        bundle.finish('incomplete',dict(exception=type(error).__name__,reason=str(error)))
        raise


if __name__=='__main__':main()
