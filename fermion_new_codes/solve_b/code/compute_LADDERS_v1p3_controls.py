"""Specified control joins, catalog arithmetic and structural diagnostics."""
import argparse
import json
from pathlib import Path
import mpmath as mp


PAIRS={
 'P64_s4':['P64_s2','P64_s8','P64_s4_tau4','P64_s4_tau2','P64_s4_c19','S4_R2_m0p2'],
 'S4_16_s4':['S4_16_s2','S4_16_s4_tau1'], 'RS32_s4':['RS32_s2','RS32_s4_tau1'],
 'S4_128_s4':['S4_128_s4_nk_v36m','S4_128_s4_nk_v36q','S4_128_s8','S4_128_s16'],
 'S4_128_s8':['S4_128_s16'],'M3b_s4':['M3b_s8'],
 'M3a_s4':['M3a_s2','M3a_s4_tau4','S4_R2_m0p1'],'M3c_s4_nk':['M3c_s4_plain']}
CATALOGS=[(0,'S4_16_s4',['S4_16_s4','S4_16_s2','S4_16_s4_tau1']),
 (1,'RS32_s4',['RS32_s4','RS32_s2','RS32_s4_tau1']),
 (2,'P64_s4',['P64_s4','P64_s2','P64_s8','P64_s4_tau4','P64_s4_tau2','P64_s4_c19']),
 (0,'S4_128_s4_nk',['S4_128_s4','S4_128_s4_nk_v36m','S4_128_s4_nk_v36q','S4_128_s16']),
 (0,'S4_128_s8',['S4_128_s8']),(0,'M3b_s4',['M3b_s4']),(0,'M3b_s8',['M3b_s8']),
 (0,'M3a_s4_tau4',['M3a_s4','M3a_s2','M3a_s4_tau4']),
 (0,'M3c_s4_nk',['M3c_s4_nk']),(0,'M3c_s4',['M3c_s4_plain']),
 (1,'S4_R2_m0p2',['S4_R2_m0p2']),(2,'S4_R2_m0p1',['S4_R2_m0p1'])]


def pack(value):
    if isinstance(value,mp.mpf): return mp.nstr(value,mp.mp.dps)
    if isinstance(value,dict): return {k:pack(v) for k,v in value.items()}
    if isinstance(value,list): return [pack(v) for v in value]
    return value


def maximum(samples,key):
    values=[r for r in samples if r.get(key) is not None]
    if not values: return None
    value=max(abs(r[key]) for r in values)
    ties=[r for r in values if abs(r[key])==value]
    return dict(value=value,T_over_rs=ties[0]['T_over_rs'],
                indices=[[r['member_index'],r['control_index']] for r in ties])


def channel_particles(channel,count,fraction):
    return mp.mpf(0) if count==0 else 16*channel*(count-1+mp.mpf(fraction)) if fraction is not None else None


def pair(m,c,mrows,crows):
    ms,cs=m['scalars'],c['scalars']
    if ms['rs']!=cs['rs']: raise ValueError('control nominal sizes differ')
    rs=mp.mpf(ms['rs']); other={r['T']:r for r in crows}
    samples=[]
    for a in mrows:
        b=other.get(a['T'])
        if b is None: continue
        rt=a['R_t']; ct=b['R_t']; both_roots=bool(a['zeros'] and b['zeros'])
        samples.append(dict(T=a['T'],T_over_rs=mp.mpf(a['T'])/rs,
                            member_index=a['source_index'],control_index=b['source_index'],
                            certified=bool(a['certified'] and b['certified']),counts=bool(a['counts'] and b['counts']),
                            E=mp.mpf(b['E'])/mp.mpf(cs['E0'])-mp.mpf(a['E'])/mp.mpf(ms['E0']),
                            F=mp.mpf(b['F'])-mp.mpf(a['F']),
                            centroid=(mp.mpf(b['R_c'])-mp.mpf(a['R_c']))/rs,
                            R_t=mp.mpf(ct)-mp.mpf(rt) if both_roots else None,
                            relative=(mp.mpf(ct)-mp.mpf(rt))/mp.mpf(rt) if both_roots else None))
    def summarize(which):
        selected=[r for r in samples if r[which]]
        zero=[r for r in selected if r['R_t'] is not None]
        at16=next((r['relative'] for r in zero if r['T_over_rs']==16),None)
        return dict(count=len(selected),first=selected[0]['T_over_rs'] if selected else None,
                    last=selected[-1]['T_over_rs'] if selected else None,both_zero_count=len(zero),
                    maxima={key:maximum(selected,key) for key in ['E','F','centroid','R_t','relative']},
                    relative_at_16=None if at16 is None else abs(at16),
                    signed_relative_at_16=at16)
    formation=None
    if m['formation'] is not None and c['formation'] is not None:
        a,b=m['formation']['row'],c['formation']['row']
        mr,cr=mp.mpf(a['R_t']),mp.mpf(b['R_t'])
        formation=dict(clock=(mp.mpf(b['T'])-mp.mpf(a['T']))/rs,radius=cr-mr,relative=(cr-mr)/mr,
                        normalized_radius=(cr-mr)/rs,
                        root_fraction=cr/(2*mp.mpf(cs['g'])*mp.mpf(b['E']))-
                                      mr/(2*mp.mpf(ms['g'])*mp.mpf(a['E'])),
                        certified=[a['certified'],b['certified']],
                        clock_accept=abs(mp.mpf(b['T'])-mp.mpf(a['T']))/rs<=mp.mpf('0.5'),
                        formation_radius_accept=abs(cr-mr)/mr<=mp.mpf('0.10'))
    primary=summarize('certified'); diagnostic=summarize('counts')
    primary['radius_accept']=None if primary['maxima']['relative'] is None else primary['maxima']['relative']['value']<=mp.mpf('0.10')
    return dict(formation=formation,common_count=len(samples),primary=primary,diagnostic=diagnostic,samples=samples)


def catalogs(raw_catalogs,members):
    answer={}
    for file_index,key,names in CATALOGS:
        cat=raw_catalogs[file_index][key]; columns=cat['columns']
        maximum_columns=max(columns.values()); top=max(int(k) for k,v in columns.items() if v>0)
        particles=mp.fsum(mp.mpf(v) for v in cat['particles'].values())
        occupation_checks={}
        for channel,count in columns.items():
            f=cat['lowest_occupation'].get(channel)
            expected=channel_particles(int(channel),count,f)
            occupation_checks[channel]=None if expected is None else expected-mp.mpf(cat['particles'][channel])
        for name in names:
            head=members[name]['scalars']; comparisons={}
            for quantity,data in cat.get('head_comparison',{}).items():
                stored,prepared=mp.mpf(data['stored_head']),mp.mpf(data['prepared'])
                head_key={'N_B':'NB','E0':'E0','f_kin':'f_kin'}[quantity]
                comparisons[quantity]=dict(prepared=data['prepared'],stored=data['stored_head'],
                    relative=abs(prepared-stored)/abs(stored) if stored else None,
                    stored_head_difference=stored-mp.mpf(head[head_key]),source_rel_diff=data['rel_diff'])
            answer[name]=dict(catalog_file_index=file_index,catalog_entry=key,source=cat.get('source'),
                              source_path=cat.get('path'),columns=columns,top=top,max_columns=maximum_columns,
                              onset=4*maximum_columns,onset_over_rs=mp.mpf(4*maximum_columns)/mp.mpf(head['rs']),
                              particle_total=particles,particle_total_minus_NB=particles-mp.mpf(head['NB']),
                              occupation_checks=occupation_checks,head_comparisons=comparisons,
                              source_targets={k:cat.get(k) for k in ['top','n_rad_max','onset_4a_nradmax','weighted_total']})
    answer['M3d_s4']=dict(status='input_unavailable',reason='no supplied catalog')
    return answer


def structural(rows):
    problems=[]; missing={}; maxima={}
    for r in rows:
        if bool(r['zeros']) != (r['R_t'] is not None):
            problems.append(dict(index=r['source_index'],reason='root presence mismatch'))
        if r['zeros']:
            inner=r['primitives'].get('r_in')
            if inner is None or inner!=min(r['zeros']):
                problems.append(dict(index=r['source_index'],reason='inner root mismatch or unavailable',own=inner))
        for key,value in r['internal_checks'].items():
            if value is None: missing[key]=missing.get(key,0)+1
            elif key not in maxima or abs(value)>maxima[key]['absolute']:
                maxima[key]=dict(absolute=abs(value),index=r['source_index'],T_over_rs=r['T_over_rs'])
    prefix=[r for r in rows if r['certified']]
    last_difference=None if len(prefix)<2 else mp.mpf(prefix[-1]['R_c'])-mp.mpf(prefix[-2]['R_c'])
    return dict(problems=problems,missing_internal_fields=missing,internal_maxima=maxima,
                last_certified_centroid_difference=last_difference,
                last_centroid_decreasing=None if last_difference is None else last_difference<0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--catalogs',nargs=3,required=True)
    p.add_argument('--repair-at16',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if args.repair_at16:
        data=json.loads((out/'controls_catalogs.json').read_text())
        for precision,block in data.items():
            mp.mp.dps=int(precision)
            for pair_data in block['pairs'].values():
                for which in ['primary','diagnostic']:
                    section=pair_data[which]
                    old=section.get('signed_relative_at_16',section['relative_at_16'])
                    section['signed_relative_at_16']=old
                    section['relative_at_16']=None if old is None else mp.nstr(abs(mp.mpf(old)),mp.mp.dps)
        data['80']['at16_negative_control']=dict(signed='-0.25',absolute='0.25',passed=abs(mp.mpf('-0.25'))==mp.mpf('0.25'))
        (out/'controls_catalogs.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        return
    members=json.loads((out/'scalar_tables.json').read_text())
    rows={name:json.loads(Path(data['record_table']).read_text()) for name,data in members.items()}
    raw_catalogs=[json.loads(Path(name).read_text()) for name in args.catalogs]
    mp.mp.dps=80
    def fixture_row(radius,energy,F,centroid):
        return dict(T=0,source_index=0,T_over_rs=0,R_t=radius,E=energy,F=F,R_c=centroid,
                    zeros=[1,radius],certified=True,counts=True)
    a,b=fixture_row(4,10,1,2),fixture_row(5,20,.5,3)
    m=dict(scalars=dict(rs=8,E0=12,g=.25),formation=dict(row=a))
    c=dict(scalars=dict(rs=8,E0=24,g=.25),formation=dict(row=b))
    check=pair(m,c,[a],[b])
    tests=[dict(name='control denominator and sign',passed=check['formation']['relative']==mp.mpf(1)/4
                and check['formation']['radius']==1 and check['formation']['normalized_radius']==mp.mpf(1)/8),
           dict(name='separate energy heads',passed=check['primary']['maxima']['E']['value']==0),
           dict(name='control primitive differences',passed=check['primary']['maxima']['F']['value']==mp.mpf(1)/2
                and check['primary']['maxima']['centroid']['value']==mp.mpf(1)/8),
           dict(name='weighted fractional channel catalog',passed=channel_particles(1,3,.5)==40
                and channel_particles(2,1,1)==32 and channel_particles(3,0,None)==0)]
    if not all(t['passed'] for t in tests):raise RuntimeError('control/catalog known-answer failure')
    (out/'controls_catalogs_known_answers.json').write_text(json.dumps(pack(dict(tests=tests,pair=check)),indent=2)+'\n')
    result={}
    for precision in [80,160]:
        mp.mp.dps=precision
        pairs={m+'__'+c:pair(members[m],members[c],rows[m],rows[c]) for m,controls in PAIRS.items() for c in controls}
        result[str(precision)]=pack(dict(pairs=pairs,catalogs=catalogs(raw_catalogs,members),
                                         structural={n:structural(r) for n,r in rows.items()}))
        (out/'controls_catalogs.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(precision=precision,pairs=len(pairs),completed=True)),flush=True)


if __name__=='__main__':main()
