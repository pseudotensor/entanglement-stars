"""T22--T24 exact saved-output audit; standard library, no kernel imports."""
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

BASE=Path('<internal>/eta_massive_scalar_v1')
OLD=BASE/'environment_continuation';ROOT=OLD/'positive_moment_v1'
RAW=ROOT/'positive_moment_result.json';GEN=OLD/'separate_images_v1/separate_images_result.json'
PRIOR=OLD/'fixed_mass_sharp_repair_result.json';AUDIT=OLD/'separate_images_v1/saved_t9_floor_attribution.json'
PRODUCER=ROOT/'repair_110_positive_moment_v1.py'
PINS={RAW:'15b643497774b2919107666a6dae21f8f004c7013ad35f0f6fce7562d36d98e0',
 GEN:'276090a9343faa76b34710c78f3a2d56ebe71357b42e1e96fd6c794d2972d09f',
 PRIOR:'0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e',
 AUDIT:'5d7c7b31f5b708bc105d7bd9c8c3c7225e2f85aa16c42895f4dad151aae772fd',
 PRODUCER:'4f7380dfb33352f4dc2e8fd2d349175b63dfdb164f1687e47c730ef68db4d63d'}
LIMIT=8*2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(c,m):
 if not c:raise ValueError(m)
def rat(x):return F(int(x['numerator']),int(x['denominator'])) if isinstance(x,dict) else F(x)
def ends(x):
 lo,hi=[rat(x[k]) for k in ('lower_exact','upper_exact')];require(lo<=hi,'Reversed interval');return lo,hi
def point(x):
 lo,hi=ends(x);require(lo==hi and lo>=0,'Nonnegative point required');return lo
def vec(x):return [*ends(x['real']),*ends(x['imaginary'])]
def width(x):lo,hi=ends(x['real']);return hi-lo
def total(rows):return [sum((vec(x)[i] for x in rows),F(0)) for i in range(4)]
def pack(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,dict):return {str(k):pack(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [pack(v) for v in x]
 return x

def main():
 require(Path(__file__).resolve().parent==ROOT,'Named box directory')
 target=ROOT/'saved_positive_moment_floor_attribution.json'
 require(not target.exists() and not target.with_suffix('.json.tmp').exists(),'No retry/overwrite')
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[allowed[0]])
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(55,60))
 def deadline(s,f):raise TimeoutError('Read-only attribution deadline')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(85)
 start=time.monotonic();out=dict(status='incomplete',scientific_evaluations=0,purpose='T22--T24 saved new floor; zero scientific evaluations',source_pins={str(p):h for p,h in PINS.items()})
 try:
  for p,h in PINS.items():require(sha(p)==h,'Changed source '+str(p))
  d=json.loads(RAW.read_text());gen=json.loads(GEN.read_text());prior=json.loads(PRIOR.read_text());audit=json.loads(AUDIT.read_text())
  require(d['y']=='1/25','Fixed mass identity')
  require(audit['status']=='pass' and d['status']=='incomplete_positive_moment_floor' and d['image_stage_complete'] is True,'Unexpected predecessor status')
  require(d['counts']['new_nodes']==d['counts']['new_node_attempts']==0 and not d['results'],'No nodes/coefficient')
  require(min(F(5),F(4)+F(3))==5 and min(F(5),F(3))==3 and min(F(5),F(4))==4 and min(F(5),F(0))==0,'Active cap fixture')
  require(sum((F(4) for i in {1}|{1}),F(0))==4,'Overlap fixture')
  fixed=[x for x in prior['active_panel']['leaves'] if x['method']=='tensor_gauss'];oldparents={x['id']:x for x in prior['active_panel']['leaves'] if x['method']=='nonnegative_real_bound'}
  oldchildren={c['id']:c for p in gen['parents'] for c in p['children']}
  oldpassed={i for i,c in oldchildren.items() if all(v['status']=='pass' for v in c['images'].values())}
  changed=set(oldchildren)-oldpassed
  require((len(fixed),len(oldparents),len(oldchildren),len(oldpassed),len(changed))==(3798,443,1772,1092,680),'Wrong old cover')
  require(d['unchanged_gauss_source_ids']==[x['id'] for x in fixed] and len(d['unchanged_passed_pair_ids'])==1092 and set(d['unchanged_passed_pair_ids'])==oldpassed,'Frozen IDs')
  G=sum((width(x) for x in fixed),F(0));A0=sum((rat(oldchildren[i]['minimum_possible_width']) for i in oldpassed),F(0))
  omission=ends(prior['omitted_integral_bounds']['eta110_nonnegative'])[1];Brun=min(F(1,10000),F(1,5000)-omission);Bprint=F(8,1000000)-omission
  require(G==rat(d['fixed_gauss_width']) and omission==rat(d['omitted_upper']) and Brun==rat(d['retained_run_budget']) and Bprint==rat(d['retained_print_budget']),'Budgets/G')
  geom={r['source_child_id']:r for r in d['geometry']};require(len(d['geometry'])==len(geom)==680 and set(geom)==changed,'Geometry source IDs')
  # Reuse the old pinned strict fallback replacement proof rather than reevaluating a kernel.
  for p in gen['parents']:
   require(p['frontier_committed'] is True and total(p['children'])[1]-total(p['children'])[0]<ends(p['old_real'])[1]-ends(p['old_real'])[0],'Old strict parent cap proof')
  norm=F(2**16*math_factorial(8)**4,17*math_factorial(16)**2)
  def poly(r):return (6435*r**8-12012*r**6+6930*r**4-1260*r**2+35)/6435
  def K(r):return r/(r-1)*norm/poly(r)**2
  require(K(F(2))==F(32768,17*984467**2),'K2 identity')
  require(K(F(2))==rat(d['known_answers']['K2']) and K(F(4))==rat(d['known_answers']['K4']),'Stored K constants')
  reason_counts=Counter();signatures=defaultdict(lambda:dict(count=0,r=F(0),lam=F(0)))
  leaves=[];source_rows=[];seen_ids=set();seen_source=set();seen_parents=set();newcalls=0;selected_parent_sums=[]
  for p in d['parents']:
   pid=p['source_parent_id'];require(pid in oldparents and pid not in seen_parents,'Parent identity');seen_parents.add(pid)
   expected=next(x for x in gen['parents'] if x['source_parent_id']==pid)['children'];expected_ids=[x['id'] for x in expected]
   require(p['expected_source_child_ids']==expected_ids and [c['id'] for c in p['children']]==expected_ids,'Four source-child IDs')
   for c in p['children']:
    cid=c['id'];require(cid not in seen_source,'Duplicate source child');seen_source.add(cid);old=oldchildren[cid]
    require(ends(c['old_source_real'])==ends(old['real']) and ends(c['old_source_imaginary'])==ends(old['imaginary']),'Original source endpoints')
    require(c['minimum_possible_width']==old['minimum_possible_width'] and [rat(x) for x in c['retained_source_sum']]==vec(old),'Inherited field meaning')
    if cid in oldpassed:
     require(not c['descendants'] and vec(c)==vec(old),'Frozen passed source');continue
    g=geom[cid];a,b,u,v=map(F,old['bounds_z_t']);nz=nt=1;reuse=old['images']['separate_t_image']['status']=='pass'
    while (b-a)/nz>F(1,2):nz*=2
    if not reuse:
     while (v-u)/nt>F(1,8):nt*=2
    bounds=[[a+(b-a)*i/nz,a+(b-a)*(i+1)/nz,u+(v-u)*j/nt,u+(v-u)*(j+1)/nt] for i in range(nz) for j in range(nt)]
    require(g['nz']==nz and g['nt']==nt and g['reuse_t']==reuse and [[rat(x) for x in row] for row in g['bounds']]==bounds,'Exact Cartesian preflight')
    require(len(c['descendants'])==len(bounds),'Complete descendant cover')
    local=[]
    for leaf,bd in zip(c['descendants'],bounds):
     lid=leaf['id'];require(lid not in seen_ids and lid>max(oldchildren),'New unique leaf ID');seen_ids.add(lid)
     require(leaf['source_child_id']==cid and list(map(F,leaf['bounds_z_t']))==bd,'Leaf geometry identity')
     r=width(leaf);require(r>=0,'Leaf width');images=leaf['images'];require(set(images)=={'z','t'},'Two ordered images')
     sig=[];reasons=set()
     for kind in ('z','t'):
      im=images[kind];require(im['status'] in ('pass','refused'),'Complete image outcome')
      if 'reused_source_child_id' in im:
       require(kind=='t' and reuse and im['status']=='pass' and im['rho']==4 and im['reused_source_child_id']==cid and im['M']==old['images']['separate_t_image']['M'],'Reused t modulus')
      else:
       newcalls+=1;require(im['rho']==2,'New image radius')
       if im['status']=='refused':reason_counts[im['guard_reason']]+=1
      reason=im.get('guard_reason');sig.append('pass' if reason is None else 'refused:'+reason)
      if reason:reasons.add(reason)
     passed=all(x['status']=='pass' for x in images.values());q=None
     if passed:
      e=point(leaf['quadrature_error']);Mz,Mt=point(images['z']['M']),point(images['t']['M']);area=(bd[1]-bd[0])*(bd[3]-bd[2])
      require(e>=area*(K(F(2))*Mz+K(F(images['t']['rho']))*Mt),'Exact positive-moment allowance')
      q=2*e;lam=min(r,q);classification='allowance' if q<r else 'fallback' if r<q else 'tie'
      require(leaf['node_pruned_no_width_gain']==(q>=r),'Safe prune')
     else:
      require('quadrature_error' not in leaf and leaf['node_pruned_no_width_gain'] is True,'No allowance after refusal')
      lam=r;classification='refused_fallback'
     require(lam==rat(leaf['minimum_possible_width']) and leaf['node_evaluations']==0,'Leaf minimum/no nodes')
     key='|'.join(sig);signatures[key]['count']+=1;signatures[key]['r']+=r;signatures[key]['lam']+=lam
     row=dict(id=lid,source_child_id=cid,bounds=leaf['bounds_z_t'],r=r,q=q,minimum=lam,signature=sig,image_radii=[images['z']['rho'],images['t']['rho']],reasons=sorted(reasons),passed=passed,classification=classification,reused_t_source=cid if reuse else None)
     local.append(row);leaves.append(row)
    rc=width(old);Rc=sum((r['r'] for r in local if not r['passed']),F(0));Ac=sum((r['minimum'] for r in local if r['passed']),F(0));Sc=Rc+Ac;Cc=min(rc,Sc)
    source_rows.append(dict(source_child_id=cid,r=rc,refused=Rc,passed=Ac,total=Sc,capped=Cc,cap_discount=Sc-Cc,leaf_ids=[r['id'] for r in local]))
    sums=total(c['descendants']);adopt=sums[1]-sums[0]<rc
    require(vec(c)==(sums if adopt else vec(old)),'Strict source adoption')
    require(width(c)<=rc,'Source cap adoption')
   candidate=total(p['children']);old=oldparents[pid];adopt=candidate[1]-candidate[0]<width(old)
   selected=candidate if adopt else vec(old)
   require([rat(x) for x in p['retained_sum']]==selected,'Parent selected four endpoints')
   if adopt:
    require(isinstance(p['retained_cover'],list) and len(p['retained_cover'])==4,'Complete retained cover')
    for record,c in zip(p['retained_cover'],p['children']):require(record['id']==c['id'] and vec(record)==vec(c),'Retained endpoint identity')
   else:require(p['retained_cover']=='old_parent','Retained original parent')
   selected_parent_sums.append(selected)
  require((len(seen_parents),len(seen_source),len(leaves),newcalls)==(443,1772,1088,1564),'Complete new coverage/counts')
  require(d['planned_new_leaves']==len(leaves) and d['planned_new_images']==newcalls and d['maximum_new_nodes']==64*(len(leaves)+1092),'Stored planned counts')
  require(reason_counts==Counter({'nonpositive_real_p':768,'nonpositive_real_chi':32,'epsilon_real_guard':14}) and sum(reason_counts.values())==814,'Reason call counts')
  require(newcalls==d['counts']['evaluations']==d['counts']['new_modulus'] and sum(reason_counts.values())==d['counts']['refused_images'] and dict(reason_counts)==d['counts']['refusals_by_reason'],'Stored counters')
  fixedsum=total(fixed);frontier=[fixedsum[i]+sum((x[i] for x in selected_parent_sums),F(0)) for i in range(4)]
  require(frontier==[rat(x) for x in d['frontier_real']+d['frontier_imaginary']] and frontier[2]<=0<=frontier[3],'Four frontier endpoints')
  require(frontier[1]-frontier[0]==rat(d['retained_width']),'Retained width')
  floor=G+A0+sum((c['capped'] for c in source_rows),F(0))
  require(floor==rat(d['pre_node_floor']) and d['pre_node_run_target_impossible']==(floor>Brun) and d['pre_node_print_target_impossible']==(floor>Bprint),'T20 floor and strict flags')
  require(floor<=frontier[1]-frontier[0],'Floor lower bound')
  lm={r['id']:r for r in leaves};require(lm[10255]['signature']==['refused:nonpositive_real_p','pass'] and lm[10638]['signature']==['refused:nonpositive_real_p','refused:nonpositive_real_chi'],'Classification witnesses')
  source_map={r['source_child_id']:r for r in source_rows}
  sets=dict(all_refused={r['id'] for r in leaves if not r['passed']},all_passed={r['id'] for r in leaves if r['passed']},
   z_refused={r['id'] for r in leaves if r['signature'][0].startswith('refused:')},t_refused={r['id'] for r in leaves if r['signature'][1].startswith('refused:')})
  for name,reason in [('p','nonpositive_real_p'),('chi','nonpositive_real_chi'),('epsilon','epsilon_real_guard')]:sets[name]={r['id'] for r in leaves if reason in r['reasons']}
  def flags(value):return dict(run_obstructed=value>Brun,print_obstructed=value>Bprint,run_equal=value==Brun,print_equal=value==Bprint)
  zero={}
  for name,removed in sets.items():
   value=G+A0+sum((min(c['r'],sum((lm[i]['minimum'] for i in c['leaf_ids'] if i not in removed),F(0))) for c in source_rows),F(0))
   zero[name]=dict(count=len(removed),ids=sorted(removed),floor=value,**flags(value))
  for row in leaves:
   cap=source_map[row['source_child_id']];row['single_leaf_potential_decrease']=cap['capped']-min(cap['r'],cap['total']-row['minimum'])
   row['necessary_if_all_other_new_minima_zero']={name:G+A0+min(cap['r'],row['minimum'])>B for name,B in [('run',Brun),('print',Bprint)]}
  unions={a+'__'+b:dict(union_count=len(sets[a]|sets[b]),intersection_count=len(sets[a]&sets[b]),intersection_ids=sorted(sets[a]&sets[b])) for a,b in [('p','chi'),('p','epsilon'),('chi','epsilon'),('z_refused','t_refused')]}
  out.update(status='pass',G=G,A0=A0,omitted_upper=omission,run_budget=Brun,print_budget=Bprint,floor=floor,floor_flags=flags(floor),
   frontier=frontier,retained_width=frontier[1]-frontier[0],counts=dict(old_parents=443,source_children=1772,frozen_pairs=1092,new_leaves=1088,new_calls=newcalls,refused_images=sum(reason_counts.values())),
   unchanged_passed_source_ids=sorted(oldpassed),K2=K(F(2)),K4=K(F(4)),image_reasons=dict(reason_counts),signatures=dict(signatures),
   total_source_cap_discount=sum((c['cap_discount'] for c in source_rows),F(0)),active_source_cap_count=sum(c['cap_discount']>0 for c in source_rows),all_source_caps_inactive=all(c['cap_discount']==0 for c in source_rows),source_rows=source_rows,leaf_rows=leaves,zero_set_floors=zero,overlaps=unions,
   top_sources=sorted(source_rows,key=lambda c:(-c['capped'],c['source_child_id']))[:32],
   top_leaves=sorted(leaves,key=lambda r:(-r['single_leaf_potential_decrease'],-r['minimum'],r['id']))[:64],
   controls=dict(active_cap=True,overlap=True,allowance_inequalities=True,partitions=True,strict_adoptions=True,four_endpoints=True,no_nodes=True))
 except Exception as error:out.update(status='incomplete',error=type(error).__name__+': '+str(error))
 finally:
  signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(60,60));out.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=sha(Path(__file__)))
  encoded=json.dumps(pack(out),indent=2,allow_nan=False)+'\n'
  if len(encoded.encode())>LIMIT:
   out=dict(status='incomplete',error='Complete pre-encoded output exceeds8MiB',source_pins={str(p):h for p,h in PINS.items()},scientific_evaluations=0)
   encoded=json.dumps(out,indent=2)+'\n'
  tmp=target.with_suffix('.json.tmp');tmp.write_text(encoded);tmp.replace(target)
 print(json.dumps(dict(status=out['status'],error=out.get('error'))))
 if out['status']!='pass':raise SystemExit(1)

def math_factorial(n):
 value=1
 for i in range(2,n+1):value*=i
 return value

if __name__=='__main__':main()
