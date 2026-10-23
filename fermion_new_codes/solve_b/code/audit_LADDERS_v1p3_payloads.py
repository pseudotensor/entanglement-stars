"""Complete typed version inventory; independent arithmetic bindings stay separate."""
import argparse
import hashlib
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import mpmath as mp


ABSENT=object()
# Exact target files whose full definitions/model strings the original pair read.
APPROVED_TARGET_SHA256={'ladders_readout_v1_2.json': 'bf3ecf93d2f5fb5900173c519a2a091d644a6629f3beb29f69fac1f1484086d0', 'ladders_readout_v1_3.json': '1c1034d3bf165a57648b63f529b9207a0edb83fd78d248d3b57b634e77b8bb99'}
EVENT_FIELDS=['dT_f','dR_t','dR_t_rel','dR_t_over_rs','dR_t_over_2gE','classes','both_certified']
PRESENTATION_ORDER='M3b_s4 P64_s4 M3a_s4 M3c_s4_nk M3d_s4 S4_16_s4 RS32_s4 S4_128_s4 M3b_s8 M3a_s2 M3a_s4_tau4 M3c_s4_plain P64_s2 P64_s8 P64_s4_tau4 P64_s4_tau2 P64_s4_c19 S4_16_s2 S4_16_s4_tau1 RS32_s2 RS32_s4_tau1 S4_128_s4_nk_v36m S4_128_s4_nk_v36q S4_128_s8 S4_128_s16 S4_R2_m0p2 S4_R2_m0p1'.split()
NEWTON_NAMES=set('M3c_s4_nk P64_s2 RS32_s2 S4_128_s4_nk_v36m S4_128_s4_nk_v36q'.split())
CERTEXT_KEYS='name certext_extent followup_extent producer_extent agrees banded_in_extent_producer banded_in_extent_certext'.split()


def kind(value):
    if value is ABSENT:return 'absent'
    if value is None:return 'null'
    if isinstance(value,bool):return 'boolean'
    if isinstance(value,int):return 'integer'
    if isinstance(value,float):return 'float'
    if isinstance(value,str):return 'string'
    if isinstance(value,list):return 'list'
    if isinstance(value,dict):return 'object'
    raise TypeError(type(value))


def same(left,right):
    if kind(left)!=kind(right):return False
    if isinstance(left,dict):
        return set(left)==set(right) and all(same(left[k],right[k]) for k in left)
    if isinstance(left,list):
        return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right


def escaped(text):return str(text).replace('~','~0').replace('/','~1')


def inventory(value,path=''):
    """Keep every container's type/shape node and every scalar leaf."""
    if isinstance(value,dict) and value:
        answer={path:value}
        for key,item in value.items():answer.update(inventory(item,path+'/'+escaped(key)))
        return answer
    if isinstance(value,list) and value:
        answer={path:value}
        for i,item in enumerate(value):answer.update(inventory(item,path+'/'+str(i)))
        return answer
    return {path:value}


def encoded(value):
    if isinstance(value,dict):return dict(type='object',keys=sorted(value),length=len(value))
    if isinstance(value,list):return dict(type='list',length=len(value))
    return dict(type=kind(value),**({} if value is ABSENT else dict(value=value)))


def unique_object(pairs):
    answer={}
    for key,value in pairs:
        if key in answer:raise ValueError('duplicate JSON object key: '+key)
        answer[key]=value
    return answer


def decode(text):return json.loads(text,object_pairs_hook=unique_object)


def present_identity(left,right):
    return left is not ABSENT and right is not ABSENT and same(left,right)


def record_presence(target,available):
    good=(isinstance(target,dict) and bool(target)) if available else target is None
    return dict(status='record_presence_agrees' if good else 'record_presence_mismatch',own_row_available=available,target_type=kind(target),reason='Available root-free rows remain record objects; absent rows use null and are distinct from empty objects.')


def match_record(sequence,physical):
    allowance=16*math.ulp(max(1.,abs(physical)))
    if len(sequence)>1 and 2*allowance>=min(b['T']-a['T'] for a,b in zip(sequence,sequence[1:])):raise ValueError('clock allowance not below cadence')
    matches=[row for row in sequence if abs(row['T']-physical)<=allowance]
    if len(matches)>1:raise ValueError('nonunique exact-clock match')
    row=matches[0] if matches else None
    return row,dict(requested_physical_T=physical,tolerance_T=allowance,actual_physical_T=None if row is None else row['T'],actual_minus_requested=None if row is None else row['T']-physical)


def empty_encoding(target,empty,field):
    sentinel=kind(target)=='integer' and target==-1 if field=='cert_end_index' else isinstance(target,dict) and not target
    return dict(status=('legacy_empty_prefix_encoding' if field=='cert_end_index' else 'empty_minimum_encoding') if empty and sentinel else 'empty_encoding_mismatch',own_last_index_or_minimum=None if empty else 'nonempty',target=encoded(target),reason='Legacy target encoding is retained; no numerical minimum or last index exists for an empty prefix. Minus one is never dereferenced.')


def query_keys(member,formed):
    checks=[]
    for key,expected in [('matched',[str(i) for i in range(41)]),('matched_from_formation',['0','0.5','1','2','4'] if formed else [])]:
        actual=member.get(key,ABSENT)
        good=isinstance(actual,dict) and set(actual)==set(expected)
        checks.append(dict(field=key,status='query_key_set_agrees' if good else 'query_key_set_mismatch',expected_keys=expected,actual_keys=list(actual) if isinstance(actual,dict) else None))
    return checks


def object_gate(target,keys):
    good=isinstance(target,dict) and bool(target) and set(target)==set(keys)
    return dict(status='object_shape_agrees' if good else 'object_shape_mismatch',expected_keys=list(keys),target_type=kind(target),actual_keys=list(target) if isinstance(target,dict) else None)


def episode_gate(target,episodes):
    good=isinstance(target,list) and len(target)==len(episodes)
    boundaries=[]
    if isinstance(target,list):
        for actual,own in zip(target,episodes):
            passed=isinstance(actual,dict) and actual.get('first',ABSENT)==own['start'] and actual.get('last',ABSENT)==own['last']
            boundaries.append(dict(expected_first=own['start'],expected_last=own['last'],actual_first=actual.get('first') if isinstance(actual,dict) else None,actual_last=actual.get('last') if isinstance(actual,dict) else None,passed=passed))
            good=good and passed
    return dict(status='episode_sequence_agrees' if good else 'episode_sequence_mismatch',expected_count=len(episodes),actual_count=len(target) if isinstance(target,list) else None,ordered_boundaries=boundaries,target_type=kind(target))


def event_encoding(target,member_present,control_present):
    both=member_present and control_present
    present=[key for key in EVENT_FIELDS if isinstance(target,dict) and key in target]
    good=isinstance(target,dict) and len(present)==(len(EVENT_FIELDS) if both else 0)
    status=('required_event_fields_present' if both else 'absent_event_encoding') if good else 'event_encoding_mismatch'
    return dict(status=status,member_formation_present=member_present,control_formation_present=control_present,present_event_fields=present,
                reason='All seven event fields are required for two own formations; their joint absence is an explicit encoding only when an own formation is missing. Null is not absence.')


def changes(old,new):
    left,right=inventory(old),inventory(new);answer=[]
    for path in sorted(set(left)|set(right)):
        a,b=left.get(path,ABSENT),right.get(path,ABSENT)
        status='added' if a is ABSENT else 'removed' if b is ABSENT else 'exact_identity' if same(a,b) else 'changed'
        answer.append(dict(path=path,version_status=status,node_kind='container' if isinstance(a,(list,dict)) or isinstance(b,(list,dict)) else 'scalar',old=encoded(a),current=encoded(b)))
    return answer


def spread_join(data):
    answer={}
    for i,row in enumerate(data):
        key=(row['member'],row['control'],row['kind'])
        if key in answer:raise ValueError('duplicate semantic spread identity')
        answer[key]=dict(source_index=i,row=row)
    return answer


def controls():
    result=[]
    def c(name,passed):result.append(dict(name=name,passed=bool(passed)))
    c('absent differs from null',changes({},dict(a=None))[-1]['version_status']=='added')
    c('false differs from zero',not same(False,0))
    c('empty object differs from empty list',not same({},[]))
    c('list order preserved',not same(['a','b'],['b','a']))
    old=[dict(member='a',control='b',kind='step',value=1)]
    new=[dict(member='c',control='d',kind='width',value=2)]+old
    a,b=spread_join(old),spread_join(new)
    key=('a','b','step')
    c('semantic join survives changed list position',a[key]['source_index']==0 and b[key]['source_index']==1 and same(a[key]['row'],b[key]['row']))
    leaf=changes(dict(a=dict(x=1)),dict(a=dict(x=2)))
    c('changed descendant retained under ancestor',any(r['path']=='/a/x' and r['version_status']=='changed' for r in leaf))
    shapes=changes(dict(a=[1]),dict(a={'0':1}))
    c('nonempty list versus numeric-key object detected',any(r['path']=='/a' and r['version_status']=='changed' and r['old']['type']=='list' and r['current']['type']=='object' for r in shapes))
    rejected=False
    try:decode('{"a":1,"a":2}')
    except ValueError:rejected=True
    c('duplicate object keys rejected',rejected)
    c('both-absent subtree cannot pass',not present_identity(ABSENT,ABSENT))
    c('null target cannot hide available root-free row',record_presence(None,True)['status']=='record_presence_mismatch' and record_presence({'R_t':None},True)['status']=='record_presence_agrees')
    c('absent row null differs from empty object',record_presence(None,False)['status']=='record_presence_agrees' and record_presence({},False)['status']=='record_presence_mismatch')
    sequence=[dict(T=0.),dict(T=1.+4*math.ulp(1.)),dict(T=2.)]
    row,locator=match_record(sequence,1.)
    c('representation-only clock match retains difference',row is not None and locator['actual_minus_requested']==4*math.ulp(1.) and locator['tolerance_T']==16*math.ulp(1.))
    row,locator=match_record(sequence,1.125)
    c('different saved-step clock not substituted',row is None and locator['actual_physical_T'] is None)
    c('empty-prefix sentinel is never a final-row index',empty_encoding(-1,True,'cert_end_index')['status']=='legacy_empty_prefix_encoding' and empty_encoding(-1,False,'cert_end_index')['status']=='empty_encoding_mismatch')
    c('empty minimum cannot hide a nonempty prefix',empty_encoding({},True,'Rc_min')['status']=='empty_minimum_encoding' and empty_encoding({},False,'Rc_min')['status']=='empty_encoding_mismatch')
    complete=dict(matched={str(i):None for i in range(41)},matched_from_formation={})
    c('complete query sets independently checked',all(r['status']=='query_key_set_agrees' for r in query_keys(complete,False)))
    del complete['matched']['1']
    c('missing query cannot disappear during child traversal',query_keys(complete,False)[0]['status']=='query_key_set_mismatch')
    c('null cannot hide a present minimum or bracket',object_gate(None,['at'])['status']=='object_shape_mismatch' and object_gate({'at':1},['at'])['status']=='object_shape_agrees')
    expected_episodes=[dict(start=1.,last=2.),dict(start=3.,last=4.)]
    c('complete episode tail cannot disappear',episode_gate([dict(first=1.,last=2.)],expected_episodes)['status']=='episode_sequence_mismatch')
    c('episode row type and order checked',episode_gate([None,dict(first=3.,last=4.)],expected_episodes)['status']=='episode_sequence_mismatch' and episode_gate([dict(first=3.,last=4.),dict(first=1.,last=2.)],expected_episodes)['status']=='episode_sequence_mismatch')
    def fixture(kept,confirm,flag,contract,T=0.,banded=False):
        return dict(source_index=int(T),T=T,T_over_rs=T,kept=kept,confirming=confirm,converged=flag,contract=contract,banded=banded,kept_prefix=kept<=.001)
    plain=fixture(.0002,.01,False,'plain false')
    old,_=convergence_projection([plain],[False],[False],'old',1e-6,[])
    current,_=convergence_projection([plain],[False],[False],'current',1e-6,[])
    c('old plain confirmer inapplicable, current retained',old['excluded'][0]['confirming'] is None and old['excluded'][0]['gate']=='capped (converged False)' and current['excluded'][0]['confirming']==.01)
    newton=fixture(1e-7,2e-6,True,'Newton explicit')
    old,_=convergence_projection([newton],[False],[False],'old',1e-6,[])
    current,_=convergence_projection([newton],[True],[True],'current',1e-6,[])
    c('old-only Newton confirming failure',old['conf_fail_count']==1 and old['first_noncounting_gate']=='confirming residual above the stop rule' and current['conf_fail_count']==0)
    both=fixture(.002,.003,False,'plain false')
    projection,_=convergence_projection([both],[False],[False],'current',1e-6,[])
    c('all current gate reasons ordered',projection['first_noncounting_gate']=='kept residual above 1e-3; confirming residual above 1e-3')
    outside=fixture(.0002,None,True,'plain converged',T=1.,banded=True);outside['kept_prefix']=False
    projection,_=convergence_projection([both,outside],[False,True],[False,False],'current',1e-6,[])
    c('local band beyond prefix retained',projection['banded_all']==1 and projection['banded_in_extent']==0 and projection['banded_records'][0]['certified'] is False)
    true_plain=fixture(0.,None,True,'plain converged')
    plain_projection,_=convergence_projection([true_plain],[True],[True],'current',1e-6,[])
    old_projection,_=convergence_projection([true_plain],[True],[True],'old',1e-6,[])
    c('plain empty Newton null old absent stay distinct',plain_projection['unconverged_plain']==[] and plain_projection['plain_confirmed']==0 and current['unconverged_plain'] is None and current['plain_confirmed'] is None and 'unconverged_plain' not in old_projection)
    last=fixture(.0002,None,False,'plain last-kept',banded=True)
    projection,_=convergence_projection([last],[True],[True],'current',1e-6,[])
    c('last-kept exception retained with null following',len(projection['unconverged_plain'])==1 and projection['unconverged_plain'][0]['following'] is None and projection['plain_confirmed']==0)
    c('own missing formation permits joint event absence',event_encoding({},False,True)['status']=='absent_event_encoding')
    c('null or partial event fields are not absence',event_encoding({k:None for k in EVENT_FIELDS},False,True)['status']=='event_encoding_mismatch' and event_encoding({'dT_f':None},False,True)['status']=='event_encoding_mismatch')
    c('two formations require every event field',event_encoding({k:0 for k in EVENT_FIELDS},True,True)['status']=='required_event_fields_present' and event_encoding({k:0 for k in EVENT_FIELDS[:-1]},True,True)['status']=='event_encoding_mismatch')
    target_equal=same(dict(value=1),dict(value=1));own_equal=same(1,2)
    c('target identity does not imply own agreement',target_equal and not own_equal)
    if not all(row['passed'] for row in result):raise RuntimeError('typed payload controls failed')
    return result


def family(path):
    parts=path.split('/')[1:]
    if not parts:return 'root'
    if parts[0]=='members' and len(parts)>=3:return 'members/'+parts[2]
    if parts[0]=='fits' and len(parts)>=2:return 'fits/'+parts[1]
    return parts[0]


def source_bindings(targets,out):
    paths=[targets/'ladders_readout_v1_2.json',targets/'ladders_readout_v1_3.json']
    paths.extend(out/name for name in ['scalar_tables.json','residual_series.json','record_diagnostics.json','controls_catalogs.json','descriptive_arithmetic.json','historical_width_direct_control.json'])
    sources=[]
    for path in paths:
        data=path.read_bytes();sources.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),md5=hashlib.md5(data).hexdigest()))
    return dict(files=sources,raw_file_identity_audit=str(out/'source_audit.json'),complete_metadata_binding=str(out/'source_metadata_audit.json'),crosswalk='LADDERS_v1p3_manual_occurrence_map_v1.md section8.5')


def retrieve(value,parts):
    for key in parts:
        if isinstance(value,dict):
            if key not in value:return ABSENT
            value=value[key]
        elif isinstance(value,list):
            if not key.isdigit() or int(key)>=len(value):return ABSENT
            value=value[int(key)]
        else:return ABSENT
    return value


def own_comparison(target,own,locator,derived=False):
    result=dict(locator=locator,own=encoded(own))
    if own is ABSENT:
        result.update(status='binding_input_unavailable');return result
    if derived and kind(target) in ['integer','float'] and kind(own) in ['integer','float']:
        with localcontext() as context:
            context.prec=100
            delta=Decimal.from_float(float(target))-Decimal.from_float(float(own))
        result.update(status='derived_value_diagnostic',target_minus_own=str(delta),
                      reason='Complete unprinted scalar tail is diagnostic; the published precision uses its separate occurrence check.')
    else:
        equal=same(target,own)
        if kind(target) in ['integer','float'] and kind(own) in ['integer','float']:
            equal=Fraction(target)==Fraction(own)
        result.update(status='exact_primitive_or_discrete_agreement' if equal else 'own_value_mismatch')
    return result


def confirmer(row,version):
    return None if version=='old' and not row['contract'].startswith('Newton') else row['confirming']


def reasons(row,version,stop):
    result=[];newton=row['contract'].startswith('Newton');confirm=confirmer(row,version)
    if version=='old' and not row['converged']:result.append('capped (converged False)')
    if row['kept']>.001:result.append('residual above 1e-3' if version=='old' else 'kept residual above 1e-3')
    if confirm is not None:
        if version=='old' and newton and stop is ABSENT:raise ValueError('old Newton confirmer has no raw stop threshold')
        threshold=stop if version=='old' else .001
        if confirm>threshold:result.append('confirming residual above the stop rule' if version=='old' else 'confirming residual above 1e-3')
    return '; '.join(result)


def convergence_projection(rows,counts,prefix,version,stop,dropped):
    failures=[(i,row) for i,row in enumerate(rows) if not counts[i]]
    kept_failures=[row for row in rows if row['kept']>.001]
    is_newton=any(row['contract'].startswith('Newton') for row in rows)
    confirmed=[(i,confirmer(row,version)) for i,row in enumerate(rows) if confirmer(row,version) is not None]
    conf_failures=[rows[i] for i,value in confirmed if value>(stop if version=='old' else .001)]
    global_max=max(row['kept'] for row in rows);kept_ties=[row['source_index'] for row in rows if row['kept']==global_max]
    admitted=[row for row,flag in zip(rows,prefix) if flag]
    kept_prefix=[row for row in rows if row['kept_prefix']]
    end=admitted[-1]['T_over_rs'] if admitted else None
    newton_values=[row['confirming'] for row in rows if row['contract'].startswith('Newton') and row['confirming'] is not None]
    conf_max=max(newton_values) if newton_values else None
    conf_ties=[row['source_index'] for row in rows if row['contract'].startswith('Newton') and row['confirming']==conf_max] if conf_max is not None else []
    excluded=[]
    for i,row in failures:
        gate=reasons(row,version,stop)
        if not gate:raise ValueError('version mask failure lacks a complete mapped gate')
        excluded.append(dict(T=row['T_over_rs'],residual=row['kept'],converged=row['converged'],confirming=confirmer(row,version),gate=gate))
    result=dict(records=len(rows),records_dropped_without_E=len(dropped),first=rows[0]['T_over_rs'],last=rows[-1]['T_over_rs'],
                converged_true_stored=sum(row['converged'] for row in rows),capped=sum(not row['converged'] for row in rows),counting=sum(counts),
                conf_fail_count=len(conf_failures),conf_fail_clocks=[row['T_over_rs'] for row in conf_failures],conf_max=conf_max,
                conf_max_at=next((row['T_over_rs'] for row in rows if row['source_index'] in conf_ties),None),stop_fields=stop,
                last_converged=end,certified_prefix_end=end,first_noncounting=failures[0][1]['T_over_rs'] if failures else None,
                first_noncounting_gate=excluded[0]['gate'] if excluded else None,r303_prefix_end=kept_prefix[-1]['T_over_rs'] if kept_prefix else None,
                r303_last_counting=ABSENT,excluded=excluded,residual_max=global_max,
                residual_max_at=next(row['T_over_rs'] for row in rows if row['source_index'] in kept_ties),residual_max_certified=max(row['kept'] for row in admitted) if admitted else None)
    result.update({'first_above_1e-3':kept_failures[0]['T_over_rs'] if kept_failures else None,'above_1e-3_count':len(kept_failures),'below_1e-4':ABSENT,'band_1e-4_1e-3':ABSENT})
    if version=='current':
        bands=[(i,row) for i,row in enumerate(rows) if row['banded']]
        result.update(banded_all=len(bands),banded_in_extent=sum(prefix[i] for i,row in bands),banded_clocks=[row['T_over_rs'] for i,row in bands],
                      banded_records=[dict(T=row['T_over_rs'],kept=row['kept'],confirming=row['confirming'],certified=prefix[i]) for i,row in bands])
        plain=[(i,row) for i,row in enumerate(rows) if row['contract'].startswith('plain') and not row['converged']]
        result['unconverged_plain']=None if is_newton else [dict(T=row['T_over_rs'],kept=row['kept'],following=row['confirming'],counts=counts[i]) for i,row in plain]
        result['plain_confirmed']=None if is_newton else sum(row['confirming'] is not None for i,row in plain)
    return result,dict(kept_maximum_ties=kept_ties,Newton_confirming_maximum_ties=conf_ties,excluded_source_indices=[row['source_index'] for i,row in failures],is_newton=is_newton)


def load_own(out):
    a=decode((out/'scalar_tables.json').read_text());h=decode((out/'record_diagnostics.json').read_text());r=decode((out/'residual_series.json').read_text())
    resources=dict(A=a,H=h,R=r,C=decode((out/'controls_catalogs.json').read_text()),D=decode((out/'descriptive_arithmetic.json').read_text()),historical_width=decode((out/'historical_width_direct_control.json').read_text()),raw={},records={},convergence={},basis={},spreads={})
    for name,member in a.items():
        resources['raw'][name]=decode(Path(member['source']).read_text())
        resources['records'][name]={row['source_index']:row for row in decode(Path(member['record_table']).read_text())}
    return resources


ORDINATES=['T_f','R_t_over_rs','R_t_over_2gE']


def numeric_tree(value):
    if isinstance(value,list):return [numeric_tree(v) for v in value]
    return None if value is None else float(value)


def shape_equal(target,own):
    if kind(target) in ['integer','float'] and kind(own) in ['integer','float']:return True
    if kind(target)!=kind(own):return False
    if isinstance(own,dict):return set(target)==set(own) and all(shape_equal(target[k],v) for k,v in own.items())
    if isinstance(own,list):return len(target)==len(own) and all(shape_equal(a,b) for a,b in zip(target,own))
    return True


def fit_basis(kind_name,name,res):
    cache=(kind_name,name)
    if cache in res['basis']:return res['basis'][cache]
    scalar=res['A'][name]['scalars'];calculated={}
    for precision in [80,160]:
        with mp.workdps(precision):
            s=lambda key:mp.mpf(scalar[key]);rs=s('rs')
            if kind_name=='size':basis=[mp.mpf(1),rs**(-mp.mpf(1)/3)]
            elif kind_name=='mass_iso':basis=[mp.mpf(1),((9*mp.pi*s('NB')/32)**(mp.mpf(1)/3)/(s('m')*s('R0')))**2]
            elif kind_name=='mass_own':basis=[mp.mpf(1),s('f_kin')/mp.mpf('0.3')]
            else:
                z,h=mp.log(1/rs),mp.log(s('sigma'));basis=[mp.mpf(1),z,h,z*h]
            calculated[str(precision)]=[mp.nstr(v,precision) for v in basis]
    res['basis'][cache]=calculated
    return calculated


def version_member(version,name,res):
    return res['A'][name] if version=='current' else res['H']['historical']['old_report']['members'][name]


def first_class(version,name,res):
    m=version_member(version,name,res);f=m['formation']
    if f is None:return None
    i=f['row']['source_index']
    return 'certified' if f['row']['certified'] else '30.3 only' if res['R'][name]['rows'][i]['kept_prefix'] else 'uncertified'


def fit_projection(version,title,res):
    d=res['D']['160'];a=res['A'];historical=res['H']['historical'];details=dict(source='descriptive_arithmetic.json',precision_scope='cached own80/160; empirical precision, not an enclosure')
    if title in ['size_formed_certified','mass_formed_certified']:
        roster=['S4_16_s4','RS32_s4','P64_s4','S4_128_s4'] if title.startswith('size') else ['M3b_s4','P64_s4','M3a_s4','M3c_s4_nk','M3d_s4']
        return [n for n in roster if version_member(version,n,res)['formation'] is not None and version_member(version,n,res)['formation']['row']['certified']],details
    if title=='design_rank':
        result={}
        for subset in ['certified','every_formation_record']:
            own=historical['old_rank'] if version=='old' and subset=='certified' else d['rank_tables'][subset]
            names=own['members'];result[subset]=dict(rows=names,rows_rs_sigma_tau=[[a[n]['scalars'][k] for k in ['rs','sigma','tau']] for n in names],rank=own['rank'],columns=8,singular_values=numeric_tree(own['singular_values']),residual_dof=len(names)-own['rank'])
        details['old_certified_scope']='Historical actual eight-row rank/singular values retain own80-digit and exact rank-proof scope; no new paired singular-value precision.'
        return result,details
    if title=='width_per_doubling':
        result=[]
        for left,right in [('P64_s2','P64_s4'),('P64_s4','P64_s8'),('S4_128_s4','S4_128_s8'),('S4_128_s8','S4_128_s16')]:
            x,y=version_member(version,left,res),version_member(version,right,res);xr,yr=x['formation']['row'],y['formation']['row']
            def ordinate(m,r,key):
                if key=='T_f':return Fraction(r['T'])/Fraction(m['scalars']['rs'])
                if key=='R_t_over_rs':return Fraction(r['R_t'])/Fraction(m['scalars']['rs'])
                return Fraction(r['R_t'])/(2*Fraction(m['scalars']['g'])*Fraction(r['E']))
            item={'r_s/a':a[left]['scalars']['rs'],'from':left,'to':right,'both_certified':xr['certified'] and yr['certified']}
            for key in ORDINATES:item[{'T_f':'dT_f','R_t_over_rs':'dR_t_over_rs','R_t_over_2gE':'dR_t_over_2gE'}[key]]=float(ordinate(y,yr,key)-ordinate(x,xr,key))
            result.append(item)
        return result,details
    if title in ['size_power3_with_uncertified','mass_power3_iso']:
        family='size' if title.startswith('size') else 'mass_isotropic';names=['RS32_s4','P64_s4','S4_128_s4'] if family=='size' else ['M3b_s4','P64_s4','M3c_s4_nk']
        result={key:[{k:float(d['powers'][family][key][k]) for k in ['p','A','Y0']}] for key in ORDINATES};result['members']=names
        details['root_scope']='Only the independently bracketed prescribed nonzero three-point branch; no target exponent is used as input.'
        return result,details
    mapping={'mass_diag_iso':('mass_isotropic','mass_iso'),'mass_diag_own':('mass_own','mass_own'),'size_diag_third':('size_certified','size'),'size_diag_third_with_uncertified':('size_labelled','size'),'width_cross_certified':('width_certified','width'),'width_cross_with_uncertified':('width_labelled','width'),'width_cross_with_s2':('width_certified','width')}
    if title not in mapping:return None,dict(status='unmapped_format',title=title)
    fit_name,kind_name=mapping[title];old_width=version=='old' and title=='width_cross_certified'
    fit=historical['old_width_fit'] if old_width else d['fits'][fit_name];names=fit['members'];first=fit['ordinates']['T_f'];columns=len(first['coefficients'])
    result=dict(model='declared descriptive design',members=names,n=len(names),coefficients=columns,dof=first['dof'])
    for key in ORDINATES:
        own=fit['ordinates'][key]
        if old_width:
            direct=res['historical_width']['precisions']['160'][key]
            result[key]=dict(coef=numeric_tree(direct['coefficients']),pred=numeric_tree(direct['fitted']),residual=numeric_tree(direct['computed_residuals']),rms=float(direct['analytic_rms']),max_abs=0.,y=numeric_tree(direct['ordinates']))
        else:result[key]=dict(coef=numeric_tree(own['coefficients']),pred=numeric_tree(own['fitted']),residual=numeric_tree(own['residuals']),rms=float(own['rms']),max_abs=float(own['max_residual']),y=[float(row[0]) for row in own['ordinate']])
    designs=[[float(v) for v in row] for row in first['design']]
    if kind_name=='width':result.update(z=[row[1] for row in designs],h=[row[2] for row in designs])
    else:result['x']=[row[1] for row in designs]
    if title in ['size_diag_third_with_uncertified','width_cross_certified','width_cross_with_uncertified','width_cross_with_s2']:
        result['uncertified']=[n for n in names if not version_member(version,n,res)['formation']['row']['certified']]
    queries=['S4_128_s4','P64_s2'] if old_width else list(first.get('predictions',{}))
    if queries:
        predictions={}
        for name in queries:
            own=version_member(version,name,res);f=own['formation'];record={'none_to':own['mask_summary']['last']} if f is None else {**{k:f['row'][{'T_f':'T_over_rs','R_t_over_rs':'R_t_over_rs','R_t_over_2gE':'R_t_over_2gE'}[k]] for k in ORDINATES},'class':first_class(version,name,res)}
            item=dict(basis=numeric_tree(fit_basis(kind_name,name,res)['160']),record=record)
            for key in ORDINATES:
                item[key]=float(res['historical_width']['precisions']['160'][key]['predictions'][name] if old_width else fit['ordinates'][key]['predictions'][name])
            predictions[name]=item
        result['value_at_other_members']=predictions
    if old_width:details['historical_direct_control']='Own direct160 coefficients/queries, with separately accepted cached80 comparison; analytic training residual/RMS zero, dof0.'
    return result,details


def fit_binding(version,title,parts,target,res):
    projection,details=fit_projection(version,title,res)
    if projection is None:return details
    own=retrieve(projection,parts)
    locator=dict(file='descriptive_arithmetic.json/record_diagnostics.json/historical_width_direct_control.json',fit=title,version=version,path='/'.join(parts),**details)
    if own is ABSENT:return dict(status='unmapped_fit_field',locator=locator)
    if parts and parts[0]=='model':return dict(status='adopted_definition',reason='The model string names the fixed descriptive design in the frozen manual map; no physical law is fitted.',locator=locator)
    if isinstance(own,(dict,list)):
        return dict(status='fit_shape_agrees' if shape_equal(target,own) else 'fit_shape_mismatch',locator=locator,own_shape=encoded(own),target_shape=encoded(target))
    derived=(len(parts)>1 and parts[0] in ORDINATES and parts[1] in ['coef','pred','residual','rms','max_abs']) or (parts and parts[0] in ['x','z','h','value_at_other_members'])
    if title=='design_rank':derived='singular_values' in parts
    if title=='width_per_doubling':derived=bool(parts and any(k in parts for k in ['dT_f','dR_t_over_rs','dR_t_over_2gE']))
    if title in ['size_power3_with_uncertified','mass_power3_iso']:derived=bool(parts and parts[0] in ORDINATES)
    if len(parts)>1 and parts[0] in ['R_t_over_rs','R_t_over_2gE'] and parts[1]=='y':derived=True
    if len(parts)>3 and parts[0]=='value_at_other_members' and parts[2]=='record' and parts[3] in ['T_f','none_to','class']:derived=False
    return own_comparison(target,own,locator,derived)


def control_kind(left,right,res):
    if (left,right)==('S4_128_s8','S4_128_s16'):return 'control_sigma (the certified factor-2 pair at 128)'
    if right.startswith('S4_R2_'):return 'preparation'
    if right=='P64_s4_c19':return 'comparator'
    if right.startswith('S4_128_s4_nk_') or right=='M3c_s4_plain':return 'control_solver'
    a,b=res['A'][left]['scalars'],res['A'][right]['scalars']
    if a['sigma']!=b['sigma']:return 'control_sigma'
    if a['tau']!=b['tau']:return 'control_step'
    raise ValueError('control role absent from the fixed prescribed roster')


def spread_projection(version,left,right,res):
    cache=(version,left,right)
    if cache in res['spreads']:return res['spreads'][cache]
    key=left+'__'+right
    pair=res['C']['160']['pairs'][key] if version=='current' else res['H']['historical']['old_report']['control_pairs'][key]
    x,y=version_member(version,left,res),version_member(version,right,res);xf,yf=x['formation'],y['formation']
    formed=xf is not None and yf is not None;formation=pair['formation'] or {}
    result=dict(member=left,control=right,kind=control_kind(left,right,res),
                classes=(first_class(version,left,res)+' / '+first_class(version,right,res)) if formed else None,
                both_certified=(xf['row']['certified'] and yf['row']['certified']) if formed else None)
    for target,own_key in [('dT_f','clock'),('dR_t','radius'),('dR_t_rel','relative'),('dR_t_over_rs','normalized_radius'),('dR_t_over_2gE','root_fraction')]:
        value=formation.get(own_key);result[target]=None if value is None else float(value)
    if not formed:
        for event_key in EVENT_FIELDS:del result[event_key]
    def number(value):return None if value is None else float(value)
    def maximum(block,key):
        value=block['maxima'][key];return None if value is None else float(value['value'])
    primary,diagnostic=pair['primary'],pair['diagnostic']
    result.update(raw_common_clocks=diagnostic['count'],raw_Rt_common_count=diagnostic['both_zero_count'],
                  raw_max_abs_dRt=maximum(diagnostic,'R_t'),raw_max_rel_dRt=maximum(diagnostic,'relative'),
                  raw_max_rel_dRt_at=number(diagnostic['maxima']['relative']['T_over_rs']) if diagnostic['maxima']['relative'] is not None else None,
                  raw_rel_dRt_at_16=number(diagnostic['relative_at_16']),
                  common_clocks=primary['count'],common_from=number(primary['first']),common_to=number(primary['last']),
                  max_abs_dEE0=maximum(primary,'E'),max_abs_dminNv=maximum(primary,'F'),max_abs_dRc_over_rs=maximum(primary,'centroid'),
                  Rt_common_count=primary['both_zero_count'],max_abs_dRt=maximum(primary,'R_t'),max_rel_dRt=maximum(primary,'relative'),
                  max_rel_dRt_at=number(primary['maxima']['relative']['T_over_rs']) if primary['maxima']['relative'] is not None else None)
    eligible=[]
    for sample in res['C']['160']['pairs'][key]['samples']:
        i,j=sample['member_index'],sample['control_index'];ar,br=res['records'][left][i],res['records'][right][j]
        if ar['T']!=br['T']:raise ValueError('control sample has different physical clocks')
        if version=='current':admitted=res['R'][left]['rows'][i]['certified'] and res['R'][right]['rows'][j]['certified']
        else:admitted=res['H']['members'][left]['old_report']['prefix'][i] and res['H']['members'][right]['old_report']['prefix'][j]
        if admitted and ar['R_t'] is not None and br['R_t'] is not None:eligible.append(dict(T_over_rs=ar['T_over_rs'],member_index=i,control_index=j,T=ar['T']))
    if len(eligible)!=primary['both_zero_count']:raise ValueError('own eligible root-sample count differs from cached primary reduction')
    result.update(Rt_common_from=eligible[0]['T_over_rs'] if eligible else None,Rt_common_to=eligible[-1]['T_over_rs'] if eligible else None)
    details=dict(pair=key,version=version,source='controls_catalogs.json' if version=='current' else 'record_diagnostics.json historical.old_report.control_pairs',
                 primary_maximum_records=primary['maxima'],diagnostic_maximum_records=diagnostic['maxima'],eligible_root_samples=eligible,
                 denominator_scope='Signed first-event differences are control minus member; relative radii use the fixed member denominator; at16 is absolute. Raw diagnostic uses local masks, primary uses both prefixes.')
    res['spreads'][cache]=(result,details);return result,details


def spread_binding(version,left,right,parts,target,res):
    projection,details=spread_projection(version,left,right,res);own=retrieve(projection,parts)
    if own is ABSENT:return dict(status='unmapped_control_field',locator=details,path='/'.join(parts))
    locator=dict(file=details['source'],pair=details['pair'],version=version,path='/'.join(parts),maximum_locators=details['primary_maximum_records'],diagnostic_maximum_locators=details['diagnostic_maximum_records'],denominator_scope=details['denominator_scope'])
    if isinstance(own,(dict,list)):
        encoding=event_encoding(target,version_member(version,left,res)['formation'] is not None,version_member(version,right,res)['formation'] is not None) if not parts else None
        good=shape_equal(target,own) and (encoding is None or encoding['status']!='event_encoding_mismatch')
        return dict(status='control_shape_agrees' if good else 'control_shape_mismatch',event_encoding=encoding,locator=locator,own_shape=encoded(own),target_shape=encoded(target))
    key=parts[0] if parts else ''
    derived=key.startswith('d') or key in ['max_abs_dEE0','max_abs_dminNv','max_abs_dRc_over_rs','max_abs_dRt','max_rel_dRt','raw_max_abs_dRt','raw_max_rel_dRt','raw_rel_dRt_at_16']
    comparison=own_comparison(target,own,locator,derived)
    if (left,right)==('M3c_s4_nk','M3c_s4_plain') and key in ['common_to','max_rel_dRt_at','raw_max_rel_dRt_at','Rt_common_from','Rt_common_to'] and comparison['status']=='own_value_mismatch':
        diagnostic=own_comparison(target,own,locator,True)
        comparison.update(status='unprinted_clock_format_diagnostic',target_minus_own=diagnostic.get('target_minus_own'),reason='Exact own physical-sample clock is retained for the ten inspected M3c summary representations; no formatter or tolerance is inferred to force equality. Published occurrence precision is checked separately; formatting remains undefined.')
    return comparison


def member_binding(version,name,parts,target,res):
    """First own-binding stage: raw metadata, first-root and all clock queries."""
    if name not in res['A']:return dict(status='binding_input_unavailable',reason='Member absent from fixed own roster')
    a=res['A'][name];raw=res['raw'][name];h=res['H']['members'][name];rr=res['R'][name]['rows']
    summary=a if version=='current' else res['H']['historical']['old_report']['members'][name]
    if not parts:return dict(status='structural_container',reason='Child fields are bound individually; container identity is separate.')
    group=parts[0];tail=parts[1:];source=a['source']
    if group=='convergence':
        cache=(version,name)
        if cache not in res['convergence']:
            counts=[row['counts'] for row in rr] if version=='current' else h['old_report']['counts']
            prefix=[row['certified'] for row in rr] if version=='current' else h['old_report']['prefix']
            stop=raw.get('args',{}).get('stop_fields',ABSENT)
            res['convergence'][cache]=convergence_projection(rr,counts,prefix,version,stop,a['mask_summary']['dropped_without_E'])
        projection,details=res['convergence'][cache]
        if not tail:return object_gate(target,list(projection))
        key=tail[0]
        if key in ['r303_last_counting','below_1e-4','band_1e-4_1e-3']:
            return dict(status='legacy_format_unresolved',reason='Retained alias/bin lacks a separate non-code endpoint/selection contract; no convention is inferred from numerical agreement.',field=key)
        if version=='current' and not details['is_newton'] and key in ['conf_max','conf_max_at']:
            return dict(status='legacy_format_unresolved',reason='Current plain summary null has unresolved legacy meaning; actual following maximum remains independently checked from the complete unconverged-plain rows.',field=key)
        if key not in projection:return dict(status='unmapped_format',field=key)
        own=retrieve(projection,tail)
        locator=dict(file='residual_series.json/record_diagnostics.json/raw args',member=name,version=version,field='/'.join(tail),**details)
        return own_comparison(target,own,locator)
    if group=='catalog':
        catalog=res['C']['160']['catalogs'].get(name)
        if catalog is None or 'catalog_entry' not in catalog:return dict(status='input_unavailable',reason='No primitive catalog for this member is supplied; no catalog or valley census is manufactured.')
        if not tail:
            required=['columns','top','n_rad_max','onset_own_a','onset_own_rs']
            good=isinstance(target,dict) and all(k in target for k in required)
            return dict(status='catalog_required_shape_agrees' if good else 'catalog_required_shape_mismatch',required_keys=required,target_type=kind(target),actual_keys=list(target) if isinstance(target,dict) else None)
        key=tail[0];rest=tail[1:];locator=dict(file='controls_catalogs.json',member=name,catalog_entry=catalog['catalog_entry'],catalog_source=catalog['source_path'],field=key)
        if key=='columns':return own_comparison(target,retrieve(catalog['columns'],rest),locator)
        if key in ['top','n_rad_max']:return own_comparison(target,catalog['top' if key=='top' else 'max_columns'],locator)
        if key in ['onset_own_a','onset_own_rs']:
            return own_comparison(target,float(catalog['onset' if key=='onset_own_a' else 'onset_over_rs']),locator,True)
        if key=='head_comparison':
            if not rest:
                if not catalog['head_comparisons']:return dict(status='input_unavailable',reason='No primitive parent-head comparison is supplied.',locator=locator)
                return {**object_gate(target,list(catalog['head_comparisons'])),'locator':locator}
            field=rest[0];comparison=catalog['head_comparisons'].get(field)
            if comparison is None:return dict(status='input_unavailable',reason='Named primitive preparation/head comparison is absent.',locator=locator)
            if len(rest)==1:return object_gate(target,['prepared','stored_head','rel_diff','flag'])
            which=rest[1];mapping=dict(prepared='prepared',stored_head='stored',rel_diff='relative')
            if which=='flag':return dict(status='definition_unresolved',reason='Unprinted head flag lacks an explicit non-code threshold contract; parent values and relative arithmetic remain separate.',locator=locator)
            if which not in mapping:return dict(status='unmapped_format',field=which)
            own=comparison[mapping[which]]
            if which=='rel_diff':own=float(own)
            result=own_comparison(target,own,locator,which=='rel_diff')
            result['alias_scope']='Stored value is the named catalog parent; saved own alias-head difference is distinct.'
            result['own_alias_head_difference']=comparison['stored_head_difference']
            return result
        if key in ['from','source','path','same_preparation_as']:
            return dict(status='metadata_or_source_process',reason='Catalog parent/path and preparation attribution are supplied provenance, separately bound by source audit; no engine execution is inferred.',locator=locator)
        return dict(status='unmapped_format',reason='Additional catalog field lacks a specific frozen crosswalk.',field=key)
    if group=='meta' and tail:
        key=tail[0];rest=tail[1:]
        if key in ['args','head']:
            own=retrieve(raw.get(key,ABSENT),rest)
            if isinstance(target,(dict,list)):
                return dict(status='structural_container',reason='Each target metadata child binds to the raw key; unexported raw keys are not invented target fields.',locator=dict(file=source,pointer='/'+key+('/'+'/'.join(rest) if rest else '')))
            return own_comparison(target,own,dict(file=source,pointer='/'+key+('/'+'/'.join(rest) if rest else '')))
        field={'m':'m','rs':'rs','sigma':'sigma','tau':'tau','nsites':'sites'}
        if key in field:return own_comparison(target,a['scalars'][field[key]],dict(file='scalar_tables.json',pointer='/'+name+'/scalars/'+field[key]))
        if key in ['name','path','records','records_dropped_without_E','stop_fields','stop_E','newton']:
            values=dict(name=name,path=source,records=len(rr),records_dropped_without_E=len(a['mask_summary']['dropped_without_E']),stop_fields=raw.get('args',{}).get('stop_fields',ABSENT),stop_E=raw.get('args',{}).get('stop_E',ABSENT),newton=any(row['contract'].startswith('Newton') for row in rr))
            return own_comparison(target,values[key],dict(file='scalar_tables.json/residual_series.json/raw args',member=name,field=key))
        if key=='script':return own_comparison(target,raw.get('script',ABSENT),dict(file=source,pointer='/script'))
        if key=='R0_over_rs':return own_comparison(target,a['scalars']['R0']/a['scalars']['rs'],dict(file='scalar_tables.json',member=name,formula='R0/rs'),True)
        return dict(status='metadata_or_source_process',reason='Raw-byte/hash/roster and execution fields use separate source_metadata_audit and source_audit; no foreign program is inspected.',field=key)
    if group=='meta':
        return object_gate(target,['name','label','role','path','read_from','bytes','md5','script','engine_entries','newton','running','args','head','records','records_dropped_without_E','stop_fields','stop_E','load_seconds','m','rs','sigma','tau','R0_over_rs','nsites'])
    if group=='cert_end_index':
        prefix=([row['certified'] for row in rr] if version=='current' else h['old_report']['prefix'])
        admitted=[i for i,flag in enumerate(prefix) if flag]
        if not admitted:return empty_encoding(target,True,'cert_end_index')
        return own_comparison(target,admitted[-1] if admitted else None,dict(file='residual_series.json' if version=='current' else 'record_diagnostics.json',member=name,field='complete prefix'))
    if group=='counts':
        if not tail:return dict(status='structural_container',reason='Scalar children bind independently.')
        mapping=dict(N_B='NB',E0='E0',f_kin='f_kin',E_over_Emax='nominal_count_ratio',E0_over_Emax='charge_ratio',two_gE0='charge0',E0_over_NBm='rest_energy_ratio',tau_over_rs='tau_over_rs',sigma_over_rs='sigma_over_rs',p_F_over_m_iso='pF_over_m_isotropic',x_iso='isotropic_x',pressure_corr_iso='isotropic_coldness',x_own='own_x',p_F_over_m_own='own_pF_over_m',onset_iso_a='onset_isotropic',onset_iso_rs='onset_isotropic_over_rs',excl_floor_a='exclusion_floor_literal')
        key=tail[0];s=a['scalars']
        if key=='N_B_rule':
            own=float(Fraction(s['rs'])/(2*Fraction(s['g'])*Fraction(s['m'])))
            binding=own_comparison(target,s['NB'],dict(file='scalar_tables.json/raw head',member=name,field='NB preparation count'))
            binding['formula_tail_diagnostic']=own_comparison(target,own,dict(formula='rs/(2gm)'),True)
            binding['own_preparation_identity_difference']=s['NB_preparation_identity_difference']
            return binding
        elif key=='E_max':
            if Fraction(s['g'])!=Fraction(1,4):raise ValueError('fixed exact Emax reduction requires g1/4')
            own=float(Fraction('0.135858')*8/Fraction(s['m'])**2)
        elif key in mapping:own=s[mapping[key]]
        else:return dict(status='definition_unresolved',reason='Unprinted auxiliary scalar lacks an independently specified formula.',field=key)
        binding=own_comparison(target,own,dict(file='scalar_tables.json',member=name,field=mapping.get(key,key),formula='NB=rs/(2gm);Emax=.135858/(g^(3/2)m^2)' if key in ['N_B_rule','E_max'] else 'fixed scalar formula'),key not in ['N_B','E0','f_kin'])
        if key in ['onset_iso_a','onset_iso_rs','excl_floor_a']:
            binding['convention_scope']='Own literal1.222/.49 formula retained; a full JSON-tail difference does not supply an unprinted higher-precision coefficient. Published occurrence rounding is separate.'
        return binding
    if group=='Rc_min':
        own=summary['centroid_minimum']
        if own is None:return empty_encoding(target,True,'Rc_min')
        if not tail:return object_gate(target,['min_Rc','at','min_Rc_over_rs','over_end'])
        mapping=dict(min_Rc='value',at='T_over_rs',min_Rc_over_rs='over_rs',over_end='at_last_certified')
        key=tail[0]
        if key not in mapping:return dict(status='unmapped_format',field=key)
        return own_comparison(target,own.get(mapping[key],ABSENT),dict(file='scalar_tables.json/record_diagnostics.json',member=name,field='centroid_minimum/'+mapping[key],source_index=own['source_index'],ties=own['ties']),key=='min_Rc_over_rs')
    if group=='episodes':
        if not tail:return episode_gate(target,summary['episodes'])
        if not tail[0].isdigit() or int(tail[0])>=len(summary['episodes']):return dict(status='binding_input_unavailable',reason='Episode index absent in own complete sequence.')
        episode=summary['episodes'][int(tail[0])]
        if len(tail)==1:
            keys=['first','last','records','min_minNv','min_at','min_cert','min_cert_at','R_t_first','R_t_last','closes_at','noncounting_inside','uncertified_inside']
            if episode['closing_source_index'] is not None:keys.append('closes_certified')
            return {**object_gate(target,keys),'source_indices':episode['source_indices']}
        key=tail[1];mapping=dict(first='start',last='last',records='count',R_t_first='first_R_t',R_t_last='last_R_t',closes_at='closes_at',noncounting_inside='noncounting',uncertified_inside='uncertified')
        if key in mapping:own=episode.get(mapping[key],ABSENT)
        elif key in ['min_minNv','min_at']:own=episode['minimum']['value' if key=='min_minNv' else 'T_over_rs']
        elif key in ['min_cert','min_cert_at']:
            minimum=episode['certified_minimum'];own=None if minimum is None else minimum['value' if key=='min_cert' else 'T_over_rs']
        elif key=='closes_certified':
            i=episode['closing_source_index']
            own=ABSENT if i is None else rr[i]['certified'] if version=='current' else h['old_report']['prefix'][i]
        else:return dict(status='unmapped_format',field=key)
        return own_comparison(target,own,dict(file='scalar_tables.json/record_diagnostics.json',member=name,field='episodes/'+tail[0]+'/'+key,source_indices=episode['source_indices'],closing_source_index=episode['closing_source_index']))
    if group in ['formation','matched','matched_from_formation']:
        formation=summary['formation']
        query_locator={}
        if group=='formation':
            if formation is None:return record_presence(target,False)
            row=formation['row'];field_parts=tail
            if not tail:return {**record_presence(target,True), 'source_index':row['source_index']}
            if tail and tail[0] in ['bracket','prev_record_T','prev_minNv']:
                if tail[0]=='bracket':
                    if len(tail)==1:return object_gate(target,['prev_T','prev_minNv','width','T_lin','T_quad'])
                    key=tail[1];mapping=dict(prev_T='previous_clock',prev_minNv='previous_F',width='width',T_lin='linear_zero')
                    if key=='T_quad':return dict(status='definition_unresolved',reason='No non-code quadratic stencil or root-selection contract.')
                    own=formation['bracket'].get(mapping.get(key,''),ABSENT)
                    return own_comparison(target,own,dict(file='scalar_tables.json/record_diagnostics.json',member=name,field='formation/bracket/'+mapping.get(key,key)),key=='T_lin')
                own=formation['bracket'].get('previous_clock' if tail[0]=='prev_record_T' else 'previous_F',ABSENT)
                return own_comparison(target,own,dict(file='scalar_tables.json/record_diagnostics.json',member=name,field='formation/'+tail[0]))
        else:
            if not tail:return dict(status='structural_container',reason='Every query key and child is retained.')
            query=tail[0];field_parts=tail[1:]
            if group=='matched':clock=Fraction(query)
            else:
                if formation is None:return record_presence(target,False)
                clock=Fraction(formation['row']['T'])/Fraction(a['scalars']['rs'])+Fraction(query)
            physical=float(clock*Fraction(a['scalars']['rs']))
            sequence=list(res['records'][name].values())
            found,query_locator=match_record(sequence,physical)
            if found is None:return {**record_presence(target,False),'locator':query_locator}
            row=dict(found)
            i=row['source_index']
            if version=='old':
                row['counts']=h['old_report']['counts'][i];row['certified']=h['old_report']['prefix'][i]
                row['banded']=bool(row['counts'] and (row['kept']>.0001 or (rr[i]['contract'].startswith('Newton') and row['confirming'] is not None and row['confirming']>.0001)))
        if not field_parts:return {**record_presence(target,True),'source_index':row['source_index'],'locator':query_locator}
        key=field_parts[0];rest=field_parts[1:];i=row['source_index'];literal=rr[i]
        mapping=dict(T='T_over_rs',index='source_index',R_t='R_t',R_t_over_rs='R_t_over_rs',R_t_over_2gE='R_t_over_2gE',E_over_E0='E_over_E0',EE0='E_over_E0',minNv='F',R_c='R_c',R_c_over_rs='R_c_over_rs',zeros='zeros',slope='outer_slope',residual='kept',confirming='confirming',counts='counts',banded='banded',gate_sigma_over_Rt='gate',gate='gate',certified='certified')
        derived=key in ['R_t_over_rs','R_t_over_2gE','E_over_E0','EE0','R_c_over_rs','gate_sigma_over_Rt','gate']
        if key in mapping:own=retrieve(row.get(mapping[key],ABSENT),rest)
        elif key=='converged':own=literal['converged']
        elif key=='n_zeros':own=len(row['zeros'])
        elif key=='tangencies':own=len(row['tangencies'])
        elif key=='simple':own=bool(row['zeros'] and not row['tangencies'] and row['outer_slope'] is not None and row['outer_slope']!=0)
        elif key=='four_gE_kappa_T':
            record=raw['records'][i];kappa=record.get('kappa_T')
            own=None if kappa is None else 4*a['scalars']['g']*row['E']*kappa;derived=True
        elif key=='pockets':return dict(status='definition_unresolved',reason='No non-code aggregation meaning for pocket alias.')
        elif key in ['class','label']:
            kept_prefix=literal['kept_prefix'];classification='certified' if row['certified'] else '30.3 only' if kept_prefix else 'uncertified'
            failure_locator={}
            if key=='class':own=classification
            elif classification=='certified':own='certified (inside the certified prefix)'
            elif classification=='uncertified':
                failure=next((r for r in rr if r['kept']>.001),None)
                if failure is None or failure['source_index']>i:return dict(status='categorical_definition_unresolved',reason='Required first kept failure not found at/before formation.')
                own="uncertified: after the member's first record above 1e-3 ("+format(failure['T_over_rs'],'.4f')+")"
                failure_locator=dict(first_kept_failure_index=failure['source_index'],first_kept_failure_clock=failure['T_over_rs'])
            elif version=='old':
                failure_index=next((j for j,passed in enumerate(h['old_report']['counts']) if not passed),None)
                if failure_index is None:return dict(status='categorical_definition_unresolved',reason='Own old local mask has no first failure.')
                failure=rr[failure_index];reasons=[]
                if not failure['converged']:reasons.append('capped (converged False)')
                if failure['kept']>.001:reasons.append('residual above 1e-3')
                if failure['contract'].startswith('Newton'):
                    stop=raw.get('args',{}).get('stop_fields')
                    if stop is None:return dict(status='binding_input_unavailable',reason='Old Newton stop threshold absent; no invented default.')
                    if failure['confirming'] is not None and failure['confirming']>stop:reasons.append('confirming residual above the stop rule')
                if not reasons:return dict(status='categorical_definition_unresolved',reason='Old failure has no mapped complete gate reason.')
                own='certified under the 30.3 rule only; after a non-counting record at '+format(failure['T_over_rs'],'.4f')+' ('+'; '.join(reasons)+')'
                failure_locator=dict(first_failure_index=failure['source_index'],first_failure_clock=failure['T_over_rs'],all_reasons=reasons)
            else:return dict(status='categorical_definition_unresolved',reason='No current 30.3-only label grammar is supplied beyond the existing actual target set.')
            return own_comparison(target,own,dict(file='residual_series.json/record_diagnostics.json',member=name,source_index=i,mask_version=version,field=key,**failure_locator))
        else:return dict(status='unmapped_format',field=key)
        return own_comparison(target,own,dict(file='scalar_records/'+name+'.json',member=name,source_index=i,T=row['T'],T_over_rs=row['T_over_rs'],field=mapping.get(key,key),mask_version=version,**query_locator),derived)
    return dict(status='not_yet_bound',reason='Remaining family requires its fixed independent mapping.',family=group)


def named_rows(rows,expected,required_keys=None):
    good=isinstance(rows,list);joined={};errors=[]
    if good:
        for i,row in enumerate(rows):
            if not isinstance(row,dict) or not isinstance(row.get('name'),str):
                errors.append(dict(source_index=i,error='malformed named row'));continue
            name=row['name']
            if name in joined:errors.append(dict(source_index=i,error='duplicate name',name=name))
            else:joined[name]=dict(source_index=i,row=row)
            if required_keys is not None and set(row)!=set(required_keys):errors.append(dict(source_index=i,error='malformed complete key set',name=name))
    else:errors.append(dict(error='not a list'))
    missing=sorted(set(expected)-set(joined));unexpected=sorted(set(joined)-set(expected))
    return joined,dict(status='named_roster_agrees' if good and not errors and not missing and not unexpected else 'named_roster_mismatch',expected_names=sorted(expected),actual_names=list(joined),missing=missing,unexpected=unexpected,errors=errors)


def presentation_gate(value,own_names,target_names):
    good=isinstance(value,list) and all(isinstance(v,str) for v in value)
    good=good and len(value)==len(set(value)) and set(value)==set(own_names)==set(target_names)
    return dict(status='presentation_roster_agrees' if good else 'presentation_roster_mismatch',adopted_order=PRESENTATION_ORDER,
                inspected_order_identity=same(value,PRESENTATION_ORDER),reason='Ordering is documentary; complete scientific membership is independently bound to the own raw roster.')


def followup_binding(target,applicable,own_endpoint):
    if not applicable:
        return dict(status='inapplicable_followup_encoding' if target is None else 'followup_encoding_mismatch',reason='Newton member absent from the plain-only follow-up; null does not mean an empty prefix.')
    return own_comparison(target,own_endpoint,dict(file='residual_series.json',field='summary/last_certified',applicable_plain=True))


def top_controls():
    result=[]
    def c(name,passed):result.append(dict(name=name,passed=bool(passed)))
    roster=['a','b'];rows=[dict(name='a'),dict(name='b')]
    c('complete named roster accepted',named_rows(rows,roster,['name'])[1]['status']=='named_roster_agrees')
    c('duplicate named row rejected',named_rows(rows+[dict(name='a')],roster,['name'])[1]['status']=='named_roster_mismatch')
    c('missing named row rejected',named_rows(rows[:1],roster,['name'])[1]['status']=='named_roster_mismatch')
    c('malformed named shape rejected',named_rows([dict(name='a',extra=0),dict(name='b')],roster,['name'])[1]['status']=='named_roster_mismatch')
    c('wrong presentation membership rejected',presentation_gate(['a','a'],roster,roster)['status']=='presentation_roster_mismatch')
    c('applicable empty and inapplicable null remain distinct',followup_binding(None,True,None)['status']=='exact_primitive_or_discrete_agreement' and followup_binding(None,False,3.)['status']=='inapplicable_followup_encoding')
    c('wrong producer endpoint exposed independently of true annotation',own_comparison(4.,3.,dict(field='producer_extent'))['status']=='own_value_mismatch')
    if not all(r['passed'] for r in result):raise RuntimeError('top-level controls failed')
    return result


def bind_top(old,new,res,out,targets,known):
    followup_path=targets/'certext_followup_v1.json';followup=decode(followup_path.read_text())
    plain=set(res['A'])-NEWTON_NAMES
    following,follow_check=named_rows(followup['members'],plain)
    joined,cert_check=named_rows(new.get('certext_check'),set(res['A']),CERTEXT_KEYS)
    projections={};follow_checks={};endpoint_changes=[]
    for name in res['A']:
        h=res['H']['members'][name]['intermediate_rule'];rr=res['R'][name]['rows']
        prefix=h['prefix'];intermediate_end=h['end'];current_end=res['R'][name]['summary']['last_certified']
        # Intermediate bands use kept-only plain admission, explicit Newton C.
        intermediate_band_indices=[]
        for i,(row,admitted) in enumerate(zip(rr,prefix)):
            band=row['kept']>.0001 or (name in NEWTON_NAMES and row['confirming'] is not None and row['confirming']>.0001)
            if admitted and band:intermediate_band_indices.append(row['source_index'])
        current_bands=[r['source_index'] for r in rr if r['certified'] and r['banded']]
        projection=dict(name=name,certext_extent=intermediate_end,producer_extent=current_end,
                        banded_in_extent_certext=len(intermediate_band_indices),banded_in_extent_producer=len(current_bands))
        if name in plain:
            # Equality uses the applicable supplied follow-up endpoint, which is
            # independently bound below, against the own producer endpoint.
            follow_endpoint=following.get(name,{}).get('row',{}).get('shortened_extent',ABSENT)
            projection.update(followup_extent=current_end,agrees=same(follow_endpoint,current_end))
        projections[name]=dict(values=projection,intermediate_band_indices=intermediate_band_indices,current_band_indices=current_bands,
                               intermediate_current_endpoint_identity=same(intermediate_end,current_end),followup_applicable=name in plain)
        if not same(intermediate_end,current_end):endpoint_changes.append(name)
        if name in following:
            f=following[name]
            follow_checks[name]=dict(source_index=f['source_index'],extent=own_comparison(f['row'].get('extent',ABSENT),intermediate_end,dict(file='record_diagnostics.json',member=name,field='intermediate_rule/end')),
                                    shortened_extent=own_comparison(f['row'].get('shortened_extent',ABSENT),current_end,dict(file='residual_series.json',member=name,field='summary/last_certified')))
    entries={};counts={}
    for version,data in [('old',old),('current',new)]:
        rows=[]
        for path,target in inventory(data).items():
            parts=[p.replace('~1','/').replace('~0','~') for p in path.split('/')[1:]]
            if parts and parts[0] in ['members','fits','spreads']:continue
            if not parts:
                binding=object_gate(target,['members','fits','spreads','order','definitions']+(['certext_check'] if version=='current' else []))
            elif parts[0]=='order':
                binding=presentation_gate(target,res['A'],data['members']) if len(parts)==1 else own_comparison(target,PRESENTATION_ORDER[int(parts[1])],dict(field='documentary presentation order'))
            elif parts[0]=='definitions':
                binding=dict(status='adopted_definition',reason='Entire exact source string inspected by original author and sole reviewer; source SHA256 guard requires re-reading altered text. Prefix ancestry, applicability, plain qualification and lack of error theorem remain as manual8.7.',source_version=version)
            elif parts[0]=='certext_check':
                if len(parts)==1:binding=cert_check
                else:
                    index=int(parts[1]);row=data['certext_check'][index];name=row['name'];own=projections[name]
                    if len(parts)==2:binding={**object_gate(target,CERTEXT_KEYS), 'semantic_name':name,'source_index':index,'own_selection':own}
                    else:
                        field=parts[2];locator=dict(file='record_diagnostics.json/residual_series.json',member=name,field=field)
                        if field=='followup_extent':binding=followup_binding(target,name in plain,own['values']['producer_extent'])
                        elif field=='agrees' and name in NEWTON_NAMES:
                            binding=dict(status='source_summary_format_unresolved',independent_endpoint_identity=own['intermediate_current_endpoint_identity'],reason='Plain-only follow-up has no Newton row; exported true fallback is undocumented, not null/endpoint equality.')
                        else:binding=own_comparison(target,own['values'].get(field,ABSENT),locator)
            else:binding=dict(status='unmapped_format',field=parts[0])
            rows.append(dict(path=path,target=encoded(target),own_arithmetic=binding));status=binding['status'];counts[status]=counts.get(status,0)+1
        entries[version]=rows
    result=dict(scope='Complete top-level bindings under frozen manual8.7. Documentary order, adopted exact rule strings and source-dependent execution remain separate from own admission arithmetic.',input_sources=source_bindings(targets,out),
                followup_source=dict(path=str(followup_path),sha256=hashlib.sha256(followup_path.read_bytes()).hexdigest()),known_answers=known+top_controls(),
                plain_followup_roster=follow_check,current_certext_roster=cert_check,old_certext_absent='certext_check' not in old,order_version_identity=present_identity(old.get('order',ABSENT),new.get('order',ABSENT)),
                followup_endpoint_checks=follow_checks,endpoint_changes=endpoint_changes,unchanged_endpoint_names=sorted(set(res['A'])-set(endpoint_changes)),status_counts=counts,entries=entries)
    (out/'payload_top_bindings.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status_counts=counts,controls=len(result['known_answers']),shortened=len(endpoint_changes),plain_followup_roster=follow_check['status'],current_certext_roster=cert_check['status'])))


ASSIGNED_BINDING_STATUSES=set('structural_container object_shape_agrees exact_primitive_or_discrete_agreement metadata_or_source_process derived_value_diagnostic definition_unresolved legacy_format_unresolved record_presence_agrees episode_sequence_agrees empty_minimum_encoding catalog_required_shape_agrees input_unavailable legacy_empty_prefix_encoding binding_input_unavailable fit_shape_agrees adopted_definition control_shape_agrees unprinted_clock_format_diagnostic presentation_roster_agrees named_roster_agrees inapplicable_followup_encoding source_summary_format_unresolved control_identity_set_agrees'.split())


def integrated_binding(index,path,target):
    binding=index.get(path,dict(status='unmapped_binding'))
    return binding,binding['status'] not in ASSIGNED_BINDING_STATUSES,('target' in binding and not same(binding['target'],target))


def integration_controls():
    result=[]
    def c(name,passed):result.append(dict(name=name,passed=bool(passed)))
    _,blocked,_=integrated_binding({},'/fits/a/x',encoded(1))
    c('missing child pointer stays blocked',blocked)
    c('known parent cannot hide unknown child status',attribution('/fits/a/x','unmapped_format')=='unclassified' and attribution('/spreads/0/x','unmapped_control_field')=='unclassified')
    c('pending or mismatch stage stays blocked',all(integrated_binding({'p':dict(status=s,target=encoded(1))},'p',encoded(1))[1] for s in ['not_yet_bound','own_value_mismatch','future_status']))
    c('stale null versus absent target rejected',integrated_binding({'p':dict(status='definition_unresolved',target=encoded(None))},'p',encoded(ABSENT))[2])
    c('explicit unresolved disposition remains nonnumerical coverage',not integrated_binding({'p':dict(status='definition_unresolved',target=encoded(1))},'p',encoded(1))[1])
    if not all(r['passed'] for r in result):raise RuntimeError('integration controls failed')
    return result


def attribution(path,binding_status):
    if binding_status not in ASSIGNED_BINDING_STATUSES:return 'unclassified'
    p=[v.replace('~1','/').replace('~0','~') for v in path.split('/')[1:]]
    if not p:return 'complete_top_level_structure'
    if p[0] in ['order','definitions','certext_check']:return 'documentary_order_or_admission_bookkeeping'
    if p[0]=='fits':return 'prescribed_fit_rank_width_values_and_membership'
    if p[0]=='spreads':return 'versioned_semantic_control_sample_sets_and_reductions'
    if p[0]!='members':return 'unclassified'
    if len(p)<3:return 'complete_member_structure'
    group=p[2]
    if group in ['convergence','cert_end_index']:return 'versioned_admission_and_summaries'
    if group in ['Rc_min','episodes']:return 'versioned_extremum_or_episode_selection'
    if group in ['formation','matched','matched_from_formation']:
        if p[-1] in ['class','label','counts','certified','banded','confirming']:return 'versioned_record_admission_annotation'
        return 'physical_formation_or_query_value_and_presence'
    if group in ['counts','catalog']:return 'preparation_scalar_or_catalog_readout'
    if group=='meta':
        if binding_status=='metadata_or_source_process':return 'explicit_source_process_metadata'
        return 'raw_preparation_or_source_binding'
    return 'unclassified'


def integrate_payload(old,new,res,out,targets,known):
    indexed={'old':{},'current':{}};stage_sources=[]
    paths=[]
    for folder in ['payload_member_bindings','payload_fit_bindings','payload_control_bindings']:
        paths.extend(sorted((out/folder).glob('*.json')))
    paths.append(out/'payload_top_bindings.json')
    for file in paths:
        data=decode(file.read_text());stage_sources.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        versions=data['entries'] if file.name=='payload_top_bindings.json' else data
        for version in indexed:
            entries=versions.get(version,[])
            if isinstance(entries,dict):
                target_version=old if version=='old' else new
                if entries.get('status')!='fit_absent' or file.parent.name!='payload_fit_bindings' or file.stem in target_version['fits']:
                    raise ValueError('unexpected non-list binding stage '+str(file))
                continue
            for index,entry in enumerate(entries):
                path=entry['path']
                if path in indexed[version]:raise ValueError('duplicate own binding path '+version+path)
                indexed[version][path]=dict(status=entry['own_arithmetic']['status'],file=str(file),entry_index=index,target=entry['target'])
    fit_summary=decode((out/'payload_fit_binding_summary.json').read_text())
    control_summary=decode((out/'payload_control_binding_summary.json').read_text())
    for version,data in [('old',old),('current',new)]:
        for path,value,binding in [('/members',data['members'],object_gate(data['members'],list(res['A']))),
                                  ('/fits',data['fits'],fit_summary['complete_fit_key_checks'][version]),
                                  ('/spreads',data['spreads'],control_summary['complete_control_identity_checks'][version])]:
            indexed[version][path]=dict(status=binding['status'],file='integration_container_checks',entry_index=None,target=encoded(value),complete_check=binding)
    ledger=changes(old,new);groups={};counts={};changed_counts={};unmapped=[];unclassified=[];stale=[];blocked=[];binding_counts={}
    for row in ledger:
        path=row['path'];category=None;bindings={}
        for version in indexed:
            if row[version]['type']=='absent':
                bindings[version]=dict(status='absent_in_target_version');continue
            binding,is_blocked,is_stale=integrated_binding(indexed[version],path,row[version])
            if binding['status']=='unmapped_binding':unmapped.append(dict(version=version,path=path))
            if is_blocked:blocked.append(dict(version=version,path=path,status=binding['status']))
            if is_stale:stale.append(dict(version=version,path=path))
            binding_counts[binding['status']]=binding_counts.get(binding['status'],0)+1
            bindings[version]=binding
            selected=attribution(path,binding['status'])
            if category is None:category=selected
            elif category!=selected:category='version_specific_attribution'
        if category=='unclassified':unclassified.append(path)
        row['attribution']=category;row['own_bindings']=bindings
        counts[category]=counts.get(category,0)+1
        if row['version_status']!='exact_identity':changed_counts[category]=changed_counts.get(category,0)+1
        # Keep files well below the git limit while retaining every typed node.
        parts=path.split('/');key='members_'+parts[2] if len(parts)>2 and parts[1]=='members' else parts[1] if len(parts)>1 else 'root'
        groups.setdefault(key,[]).append(row)
    directory=out/'payload_attribution';directory.mkdir(exist_ok=True);files={}
    for key,rows in groups.items():
        file=directory/(key+'.json');file.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');files[key]=str(file)
    named=['size_diag_third','size_diag_third_with_uncertified','size_power3_with_uncertified','mass_formed_certified','mass_diag_iso','mass_diag_own','mass_power3_iso']
    unchanged_fits={name:present_identity(old['fits'].get(name,ABSENT),new['fits'].get(name,ABSENT)) for name in named}
    release={}
    physical_fields=['T','index','R_t','R_t_over_rs','R_t_over_2gE','E_over_E0','minNv','R_c','zeros','n_zeros','tangencies','slope','four_gE_kappa_T','simple','residual','gate_sigma_over_Rt','converged','bracket','prev_record_T','prev_minNv']
    meta_fields=['path','md5','bytes','args','head','m','rs','sigma','tau','R0_over_rs','nsites']
    for name in ['S4_R2_m0p2','S4_R2_m0p1']:
        left=old['members'][name];right=new['members'][name];checks={}
        for key in meta_fields:
            checks['meta/'+key]=dict(present_in_both=key in left['meta'] and key in right['meta'],exact_typed_identity=present_identity(left['meta'].get(key,ABSENT),right['meta'].get(key,ABSENT)))
        formations=[left.get('formation'),right.get('formation')]
        checks['formation_presence']=dict(present_in_both=all(isinstance(f,dict) for f in formations),exact_typed_identity=kind(formations[0])==kind(formations[1]))
        for key in physical_fields:
            values=[f.get(key,ABSENT) if isinstance(f,dict) else ABSENT for f in formations]
            checks['formation/'+key]=dict(present_in_both=all(v is not ABSENT for v in values),exact_typed_identity=present_identity(*values))
        release[name]=dict(checks=checks,scope='Only named preparation/source fields and represented physical first-root fields; excludes admission flags, release-control spreads, unexported profiles and parent histories.')
    old_five=old['fits']['width_cross_with_s2'];new_five=new['fits']['width_cross_certified']
    design_keys=['members','z','h']+ORDINATES
    five_identity={key:present_identity(old_five.get(key,ABSENT),new_five.get(key,ABSENT)) for key in design_keys}
    summary=dict(scope='Complete typed-node coverage with independent stage pointers or explicit unresolved/input/format/process dispositions. This is coverage, not blanket numerical agreement or foreign execution clearance.',
                 input_sources=source_bindings(targets,out),stage_sources=stage_sources,known_answers=known+integration_controls(),typed_nodes=len(ledger),attribution_counts=counts,changed_attribution_counts=changed_counts,binding_status_counts=binding_counts,blocking_stage_bindings=blocked,
                 unmapped_bindings=unmapped,unclassified_paths=unclassified,stale_target_bindings=stale,named_unchanged_fit_subtrees=unchanged_fits,
                 release_preparation_and_firstroot_checks=release,old_labelled_five_current_primary_design_identity=five_identity,
                 removed_old_labelled_five_key='width_cross_with_s2' in old['fits'] and 'width_cross_with_s2' not in new['fits'],
                 files=files,container_checks={version:{p:b['complete_check'] for p,b in index.items() if 'complete_check' in b} for version,index in indexed.items()})
    summary['changed_raw_preparation_binding_nodes']=[row for row in ledger if row['version_status']!='exact_identity' and row['attribution']=='raw_preparation_or_source_binding']
    summary['changed_physical_record_scalar_nodes']=[row for row in ledger if row['version_status']!='exact_identity' and row['attribution']=='physical_formation_or_query_value_and_presence' and row['node_kind']=='scalar']
    width_members=res['D']['160']['fits']['width_labelled']['members']
    uncertified={version:[name for name in width_members if not version_member(version,name,res)['formation']['row']['certified']] for version in ['old','current']}
    pair={version:[dict(member=name,source_index=version_member(version,name,res)['formation']['row']['source_index'],certified=version_member(version,name,res)['formation']['row']['certified']) for name in ['P64_s2','P64_s4']] for version in ['old','current']}
    rank_old=res['H']['historical']['old_rank']['members'];rank_current=res['D']['160']['rank_tables']['certified']['members']
    primary_order=res['D']['160']['fits']['width_certified']['members']
    checks=dict(primary_five_exact_order=primary_order==['P64_s2','P64_s4','P64_s8','S4_128_s8','S4_128_s16'],
                P64_s2_leaves_uncertified='P64_s2' in uncertified['old'] and 'P64_s2' not in uncertified['current'] and set(uncertified['old'])-set(uncertified['current'])=={'P64_s2'} and set(uncertified['current']).issubset(uncertified['old']),
                width_two_to_four_becomes_both_certified=not all(row['certified'] for row in pair['old']) and all(row['certified'] for row in pair['current']),
                rank_adds_exact_two=set(rank_old).issubset(rank_current) and set(rank_current)-set(rank_old)=={'RS32_s2','P64_s2'} and len(rank_old)==8 and len(rank_current)==10)
    summary['S32_named_membership_changes']=dict(status='agrees' if all(checks.values()) else 'membership_mismatch',checks=checks,primary_member_order=primary_order,old_current_uncertified=uncertified,width_pair_source_rows=pair,old_rank_members=rank_old,current_rank_members=rank_current,
                                                own_sources=['descriptive_arithmetic.json/160/fits/width_certified/members','descriptive_arithmetic.json/160/fits/width_labelled/members','record_diagnostics.json/historical/old_report/members/*/formation/row','scalar_tables.json/*/formation/row','record_diagnostics.json/historical/old_rank/members','descriptive_arithmetic.json/160/rank_tables/certified/members'])
    (out/'payload_attribution_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(typed_nodes=len(ledger),unmapped=len(unmapped),unclassified=len(unclassified),stale=len(stale),files=len(files),changed_attribution_counts=changed_counts)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--targets',required=True)
    parser.add_argument('--bind-members-only',action='store_true')
    parser.add_argument('--bind-fits-only',action='store_true')
    parser.add_argument('--bind-controls-only',action='store_true')
    parser.add_argument('--bind-top-only',action='store_true')
    parser.add_argument('--integrate-only',action='store_true')
    args=parser.parse_args();out=Path(args.output);targets=Path(args.targets)
    if os.environ.get('QI2_REMOTE_CPU')!='1' or sys.executable!='<internal>/python':raise RuntimeError('prescribed remote interpreter required')
    known=controls()
    for name,expected in APPROVED_TARGET_SHA256.items():
        if hashlib.sha256((targets/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed target requires the original pair to read its definitions/models: '+name)
    old=decode((targets/'ladders_readout_v1_2.json').read_text());new=decode((targets/'ladders_readout_v1_3.json').read_text())
    if args.integrate_only:
        integrate_payload(old,new,load_own(out),out,targets,known);return
    if args.bind_top_only:
        bind_top(old,new,load_own(out),out,targets,known);return
    if args.bind_controls_only:
        res=load_own(out);directory=out/'payload_control_bindings';directory.mkdir(exist_ok=True)
        expected={}
        for key in res['C']['160']['pairs']:
            left,right=key.split('__');expected[(left,right,control_kind(left,right,res))]=key
        joins={version:spread_join(data['spreads']) for version,data in [('old',old),('current',new)]}
        completeness={version:dict(status='control_identity_set_agrees' if set(join)==set(expected) else 'control_identity_set_mismatch',expected_identities=[list(k) for k in expected],actual_identities=[list(k) for k in join]) for version,join in joins.items()}
        counts={};files={};sample_bindings={}
        for identity,key in expected.items():
            left,right,_=identity;result={}
            for version,join in joins.items():
                if identity not in join:
                    result[version]=dict(status='control_identity_unavailable');continue
                item=join[identity];entries=[]
                for path,target in inventory(item['row']).items():
                    parts=[part.replace('~1','/').replace('~0','~') for part in path.split('/')[1:]]
                    binding=spread_binding(version,left,right,parts,target,res)
                    entries.append(dict(path='/spreads/'+str(item['source_index'])+path,semantic_identity=list(identity),target=encoded(target),own_arithmetic=binding))
                    status=binding['status'];counts[status]=counts.get(status,0)+1
                result[version]=entries;sample_bindings[version+'/'+key]=res['spreads'][(version,left,right)][1]
            filename=directory/(key+'.json');filename.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');files[key]=str(filename)
        summary=dict(scope='Full ordered-pair semantic joins and independent cached old/current control reductions; primary prefix and local diagnostic sets, fixed member denominator and absolute at16 remain distinct. Derived unprinted tails stay diagnostic.',input_sources=source_bindings(targets,out),known_answers=known,status_counts=counts,complete_control_identity_checks=completeness,sample_bindings=sample_bindings,files=files)
        (out/'payload_control_binding_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(pairs=len(files),status_counts=counts)));return
    if args.bind_fits_only:
        res=load_own(out);directory=out/'payload_fit_bindings';directory.mkdir(exist_ok=True)
        counts={};files={};shape_checks={}
        expected=['mass_formed_certified','mass_diag_iso','mass_diag_own','mass_power3_iso','size_formed_certified','size_diag_third','size_diag_third_with_uncertified','size_power3_with_uncertified','width_cross_certified','width_cross_with_uncertified','width_per_doubling','design_rank']
        for version,data in [('old',old),('current',new)]:shape_checks[version]=object_gate(data['fits'],expected+(['width_cross_with_s2'] if version=='old' else []))
        for title in sorted(set(old['fits'])|set(new['fits'])):
            result={}
            for version,data in [('old',old),('current',new)]:
                if title not in data['fits']:
                    result[version]=dict(status='fit_absent',reason='Absent key is preserved, not a null fit.');continue
                entries=[]
                for path,target in inventory(data['fits'][title]).items():
                    parts=[part.replace('~1','/').replace('~0','~') for part in path.split('/')[1:]]
                    binding=fit_binding(version,title,parts,target,res)
                    entries.append(dict(path='/fits/'+escaped(title)+path,target=encoded(target),own_arithmetic=binding))
                    status=binding['status'];counts[status]=counts.get(status,0)+1
                result[version]=entries
            filename=directory/(title+'.json');filename.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');files[title]=str(filename)
        summary=dict(scope='Independent binding of already prescribed cached fit/rank/power outputs, fixed raw member predicates and elementary query bases only; no refitting, new dynamics or physical-law calibration. Unprinted derived tails are diagnostic.',input_sources=source_bindings(targets,out),known_answers=known,status_counts=counts,complete_fit_key_checks=shape_checks,files=files,query_basis_precision=res['basis'])
        # Tuple cache keys become explicit model/member records in the durable output.
        summary['query_basis_precision']=[dict(kind=k[0],member=k[1],values=v) for k,v in res['basis'].items()]
        (out/'payload_fit_binding_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(fits=len(files),status_counts=counts)));return
    if args.bind_members_only:
        res=load_own(out);directory=out/'payload_member_bindings';directory.mkdir(exist_ok=True)
        counts={};members={};key_checks={}
        for name in sorted(set(old['members'])|set(new['members'])):
            result={}
            for version,data in [('old',old),('current',new)]:
                if name not in data['members']:
                    result[version]=dict(status='member_unavailable');continue
                independent=res['A'][name] if version=='current' else res['H']['historical']['old_report']['members'][name]
                key_checks[name+'/'+version]=query_keys(data['members'][name],independent['formation'] is not None)
                entries=[]
                for path,target in inventory(data['members'][name]).items():
                    parts=[part.replace('~1','/').replace('~0','~') for part in path.split('/')[1:]]
                    binding=member_binding(version,name,parts,target,res)
                    entries.append(dict(path='/members/'+escaped(name)+path,target=encoded(target),own_arithmetic=binding))
                    status=binding['status'];counts[status]=counts.get(status,0)+1
                result[version]=entries
            filename=directory/(name+'.json');filename.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');members[name]=str(filename)
        summary=dict(scope='Member families have independent bindings or explicit source/format/input dispositions, including convergence, metadata, scalar counts, first-root classes/labels, all clock queries, episodes, centroid and catalogs. Fit/control/top-level payload families remain separate; no full-payload clearance.',input_sources=source_bindings(targets,out),known_answers=known,status_counts=counts,query_key_set_checks=key_checks,members=members)
        (out/'payload_member_binding_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(members=len(members),status_counts=counts)));return
    ledger=changes(old,new);counts={};families={}
    for row in ledger:
        status=row['version_status'];counts[status]=counts.get(status,0)+1
        name=family(row['path']);families[name]=families.get(name,0)+1
        row['family']=name
        row['own_arithmetic']=dict(status='not_yet_bound',reason='Version identity is a target/target comparison; independent own binding is a separate stage under manual map8.5.')
    names=['size_diag_third','size_diag_third_with_uncertified','size_power3_with_uncertified','mass_formed_certified','mass_diag_iso','mass_diag_own','mass_power3_iso']
    comparisons=[]
    for name in names:
        a=old['fits'].get(name,ABSENT);b=new['fits'].get(name,ABSENT)
        comparisons.append(dict(path='/fits/'+name,exact_typed_subtree_identity=present_identity(a,b),old_type=kind(a),current_type=kind(b),own_arithmetic_status='separate prescribed fit output required'))
    a,b=spread_join(old['spreads']),spread_join(new['spreads'])
    semantic=[]
    for key in sorted(set(a)|set(b)):
        left,right=a.get(key),b.get(key)
        semantic.append(dict(member=key[0],control=key[1],kind=key[2],old_source_index=None if left is None else left['source_index'],current_source_index=None if right is None else right['source_index'],exact_typed_identity=left is not None and right is not None and same(left['row'],right['row'])))
    output=dict(scope='Complete typed target-version inventory and named target-subtree identity only; no independent own-arithmetic or physical-input clearance.',input_sources=source_bindings(targets,out),known_answers=known,version_status_counts=counts,path_family_counts=families,named_unchanged_fit_subtrees=comparisons,semantic_control_joins=semantic,leaves=ledger)
    (out/'payload_version_inventory.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(controls=len(known),version_counts=counts,named_fit_subtrees=len(comparisons),semantic_controls=len(semantic))))


if __name__=='__main__':main()
