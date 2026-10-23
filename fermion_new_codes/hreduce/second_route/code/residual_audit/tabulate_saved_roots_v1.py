"""Aggregate owned root outputs; never read fields or reconstruct a root."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

ROOT=Path('<internal>/readout_v1')
SOURCE=ROOT.parent/'summary.json'
SOURCE_SHA='5795ce5eb5f70e8b9f2b7ff42ec009c49ba393e80a0015c865751e3e2266556b'
REPRESENTATIONS=('applied_sites','solve_output_sites','stored_nodes')


def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)


def location(row):return {k:row[k] for k in ('index','T','T_over_rs')}


def scalar_range(pairs):
    if not pairs:return dict(minimum=None,maximum=None)
    lo=min(x for x,row in pairs);hi=max(x for x,row in pairs)
    return dict(minimum=dict(value=lo,record=next(location(row) for x,row in pairs if x==lo),ties=sum(x==lo for x,row in pairs)),
                maximum=dict(value=hi,record=next(location(row) for x,row in pairs if x==hi),ties=sum(x==hi for x,row in pairs)))


def main():
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory only')
    targets=[ROOT/'root_readout_v1.json',ROOT/'root_records_v1.tsv']
    if any(x.exists() for x in targets):raise FileExistsError('Existing readout, never overwrite')
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[10] if len(cpus)>10 else cpus[0]])
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(45,50))
    def deadline(signum,frame):raise TimeoutError('Owned output aggregation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(55)
    start=time.monotonic();report=dict(status='incomplete',scope='Stored owned outputs only; no root or row recomputation',
        root_recomputations=0,source_sha256=SOURCE_SHA,members=[])
    try:
        raw=SOURCE.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Changed root summary')
        source=json.loads(raw)
        if source['status']!='completed_roots_and_raw_gates_only' or source['remaining_member_indices']:
            raise ValueError('Incomplete producer output')
        rows_written=0;expected_rows=0
        with targets[1].open('x',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t')
            writer.writerow(('member_index','member','record_index','T','T_over_rs','representation',
                'field_index','normalization_contract','available','isolated_roots','zero_intervals','outer_linear','outer_minus_record_R_t',
                'outer_over_record_2gE','record_root_count','root_count_mismatch','ordered_list_comparison_available','reused','missing_reason'))
            for item in source['members']:
                path=Path(item['path']);raw=path.read_bytes()
                if len(raw)!=item['size'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                    raise ValueError('Changed member output')
                data=json.loads(raw);expected_rows+=3*len(data['records'])
                member=dict(member_index=item['member_index'],member=item['member'],source_sha256=item['sha256'],
                    recorded_gates=item['summary'],missing=item['missing'],representations={})
                for representation in REPRESENTATIONS:
                    stats=dict(available_records=0,missing_records=0,no_isolated_root_records=0,no_root_or_interval_records=0,
                        isolated_root_records=0,zero_interval_records=0,zero_interval_metadata_unavailable_records=0,root_count_mismatches=0,
                        root_count_comparison_unavailable=0,missing_outer_record_comparisons=0,
                        first_root_record=None,last_root_record=None,first_count_mismatch=None,first_missing_record=None)
                    radii=[];differences=[]
                    for row in data['records']:
                        value=row['roots'].get(representation);available=value is not None
                        reason=row.get('root_unavailable') or row.get('stored_node_roots_unavailable') or 'Representation unavailable'
                        record_list=row.get('record_trapping_zeros')
                        numeric_list=isinstance(record_list,list) and all(finite(x) for x in record_list)
                        recorded_count=len(record_list) if numeric_list else None
                        if available:
                            n=len(value.get('isolated_roots',value.get('exact_saved_roots',[])));z=len(value['zero_intervals']) if 'zero_intervals' in value else None
                            outer=value.get('outer_isolated_root');delta=value.get('outer_linear_minus_record_R_t')
                            ratio=value.get('outer_R_over_record_2gE');reused=bool(value.get('reused',False))
                            stats['available_records']+=1;stats['no_isolated_root_records']+=n==0
                            stats['no_root_or_interval_records']+=n==0 and z==0
                            stats['isolated_root_records']+=n>0;stats['zero_interval_records']+=z is not None and z>0
                            stats['zero_interval_metadata_unavailable_records']+=z is None
                            if n>0:
                                if stats['first_root_record'] is None:stats['first_root_record']=location(row)
                                stats['last_root_record']=location(row)
                            if finite(outer):radii.append((outer,row))
                            if finite(delta):differences.append((delta,row))
                            else:stats['missing_outer_record_comparisons']+=1
                            mismatch=(n!=recorded_count) if numeric_list and z==0 else None
                            if mismatch is True:
                                stats['root_count_mismatches']+=1
                                if stats['first_count_mismatch'] is None:stats['first_count_mismatch']=dict(**location(row),isolated_count=n,record_count=recorded_count)
                            if mismatch is None:stats['root_count_comparison_unavailable']+=1
                            ordered=value.get('ordered_linear_minus_record_root_list') is not None and z==0
                            reason='Zero-interval metadata unavailable in reused object' if z is None else ''
                        else:
                            stats['missing_records']+=1;stats['root_count_comparison_unavailable']+=1
                            stats['missing_outer_record_comparisons']+=1
                            if stats['first_missing_record'] is None:stats['first_missing_record']=dict(**location(row),reason=reason)
                            n=z=outer=delta=ratio=mismatch=None;ordered=False;reused=False
                        def cell(x):return 'unavailable' if x is None else int(x) if type(x)==bool else x
                        writer.writerow([cell(x) for x in (item['member_index'],item['member'],row['index'],row['T'],row['T_over_rs'],
                            representation,row.get('field_index'),row.get('normalization_contract'),available,n,z,outer,delta,ratio,recorded_count,mismatch,ordered,reused,reason)])
                        rows_written+=1
                    stats['outer_radius_range']=scalar_range(radii)
                    if differences:
                        maximum=max(abs(x) for x,row in differences)
                        value,row=next((x,row) for x,row in differences if abs(x)==maximum)
                        stats['largest_absolute_outer_minus_record']=dict(absolute=maximum,signed=value,record=location(row),
                            ties=sum(abs(x)==maximum for x,row in differences))
                    else:stats['largest_absolute_outer_minus_record']=None
                    member['representations'][representation]=stats
                report['members'].append(member)
        if rows_written!=expected_rows or len(report['members'])!=32:raise ValueError('Aggregation bookkeeping failed')
        report.update(status='completed_owned_output_aggregation',per_record_rows=rows_written,
            table=dict(path=str(targets[1]),size=targets[1].stat().st_size,sha256=hashlib.sha256(targets[1].read_bytes()).hexdigest()))
    except Exception as error:report['error']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(50,50))
        report['elapsed_seconds']=time.monotonic()-start
        report['producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with targets[0].open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    if report['status']!='completed_owned_output_aggregation':raise SystemExit(1)


if __name__=='__main__':main()
