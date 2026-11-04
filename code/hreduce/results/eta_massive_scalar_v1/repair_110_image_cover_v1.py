"""T27--T29 fixed image cover, no new real panels or repeated passed images."""
from fractions import Fraction as F
from itertools import product
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time

BASE=Path('<internal>/eta_massive_scalar_v1');OLD=BASE/'environment_continuation'
ROOT=OLD/'image_cover_v1';PRIOR=OLD/'fixed_mass_sharp_repair_result.json'
GEN=OLD/'separate_images_v1/separate_images_result.json';RAW=OLD/'positive_moment_v1/positive_moment_result.json'
AUDIT=OLD/'positive_moment_v1/saved_positive_moment_floor_attribution.json'
HELPER=OLD/'positive_moment_v1/repair_110_positive_moment_v1.py';ENGINE=BASE/'fixed_mass_interval_sharp_repair.py'
PINS={PRIOR:'0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e',
 GEN:'276090a9343faa76b34710c78f3a2d56ebe71357b42e1e96fd6c794d2972d09f',
 RAW:'15b643497774b2919107666a6dae21f8f004c7013ad35f0f6fce7562d36d98e0',
 AUDIT:'ccdae92f5b296efc28ca8e2c70c496ffee7e3d33b49e94a5788badbecc2a87b6',
 HELPER:'4f7380dfb33352f4dc2e8fd2d349175b63dfdb164f1687e47c730ef68db4d63d',
 ENGINE:'d962ea9b594c2495284069eb2814ba6d6263786c374bd1b51475d15b29f3e870'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(x):return F(int(x['numerator']),int(x['denominator'])) if isinstance(x,dict) else F(x)
def ends(x):return [number(x[k]) for k in ('lower_exact','upper_exact')]
def vec(x):return [*ends(x['real']),*ends(x['imaginary'])]
def width(x):a,b=ends(x['real']);return b-a
def sums(rows):return [sum((vec(x)[j] for x in rows),F(0)) for j in range(4)]
def pack(x):
 if isinstance(x,F):return dict(numerator=str(x.numerator),denominator=str(x.denominator))
 if isinstance(x,dict):return {str(k):pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x

def tiles(domain):
 halves=[[(a,(a+b)/2),((a+b)/2,b)] for a,b in domain]
 return [list(parts) for parts in product(*halves)]

def completed_max(outcomes):
 if len(outcomes)!=8 or any(x['status']!='pass' for x in outcomes):raise ValueError('Incomplete image cover has no modulus')
 return max(number(x['M']['upper_exact']) for x in outcomes)

def main():
 if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory')
 target=ROOT/'image_cover_result.json'
 if target.exists() or target.with_suffix('.json.tmp').exists():raise FileExistsError('Never retry/overwrite')
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[allowed[0]])
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(580,600))
 def deadline(s,f):raise TimeoutError('Fixed image-cover allocation deadline')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(690)
 start=time.monotonic();last_save=start;frontier=None;context=None;engine=None;original_frontier=None
 report=dict(status='incomplete',y='1/25',purpose='SecIII G5/I6a failed image covers only',source_pins={str(p):h for p,h in PINS.items()},
  leaves=[],old_passed_cells=[],sources=[],parents=[],results={},image_stage_complete=False,
  counts=dict(evaluations=0,refused_images=0,refusals_by_reason={},tile_attempted=0,tile_returned=0,tile_passed=0,tile_refused=0,tile_unvisited=6512,node_attempted=0,node_completed=0),
  counter_definitions='returned counts numerical modulus returns; exceptions count refused; attempted=returned+refused for completed handled calls',
  guard_samples={},tail_radius_maxima={},resources=dict(threads=1,nice=10,address_space_bytes=2*2**30,cpu_soft=580,cpu_hard=600,wall_internal=690,wall_external=720))
 def persist():
  nonlocal last_save
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  try:
   report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=sha(Path(__file__)))
   if frontier is not None:
    report['frontier_real'],report['frontier_imaginary']=frontier[:2],frontier[2:]
    report['retained_width']=frontier[1]-frontier[0]
   tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(pack(report),indent=2,allow_nan=False)+'\n');tmp.replace(target);last_save=time.monotonic()
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 def checkpoint():
  if time.monotonic()-last_save>=15:persist()
 try:
  for p,h in PINS.items():
   if sha(p)!=h:raise ValueError('Changed source '+str(p))
  prior=json.loads(PRIOR.read_text());gen=json.loads(GEN.read_text());raw=json.loads(RAW.read_text());audit=json.loads(AUDIT.read_text())
  if audit['status']!='pass' or raw['status']!='incomplete_positive_moment_floor' or raw['counts']['new_node_attempts']!=0:raise ValueError('Unexpected saved state')
  fixed=[x for x in prior['active_panel']['leaves'] if x['method']=='tensor_gauss']
  G,A0=number(audit['G']),number(audit['A0']);Brun,Bprint=number(audit['run_budget']),number(audit['print_budget']);omission=number(audit['omitted_upper'])
  K2,K4=number(audit['K2']),number(audit['K4'])
  report.update(fixed_G=G,fixed_A0=A0,retained_run_budget=Brun,retained_print_budget=Bprint,omitted_upper=omission,previous_floor=number(audit['floor']),
   fixed_gauss_source_ids=[x['id'] for x in fixed],unchanged_old_passed_ids=audit['unchanged_passed_source_ids'],K2=K2,K4=K4)
  frontier=[number(x) for x in raw['frontier_real']+raw['frontier_imaginary']];original_frontier=frontier.copy()
  report['original_frontier']=frontier.copy()
  oldchildren={c['id']:c for p in gen['parents'] for c in p['children']}
  rankings={r['id']:number(r['single_leaf_potential_decrease']) for r in audit['leaf_rows']}
  leafmap={};sourcemap={};parentmap={};planned=0;frozen_new=0;missing_images=[]
  for p in raw['parents']:
   pr=dict(id=p['source_parent_id'],source_ids=[c['id'] for c in p['children']],selected_sum=[number(x) for x in p['retained_sum']],selected_cover=copy.deepcopy(p['retained_cover']))
   report['parents'].append(pr);parentmap[pr['id']]=pr
   for c in p['children']:
    sr=dict(id=c['id'],parent_id=pr['id'],original_source_real=c['old_source_real'],original_source_imaginary=c['old_source_imaginary'],real=c['real'],imaginary=c['imaginary'],panel_ids=[])
    report['sources'].append(sr);sourcemap[sr['id']]=sr
    if not c['descendants']:
     if c['id'] not in set(audit['unchanged_passed_source_ids']):raise ValueError('Unknown unsplit source')
     panel={k:copy.deepcopy(c[k]) for k in ('id','bounds_z_t','real','imaginary','quadrature_error','minimum_possible_width','node_pruned_no_width_gain')}
     panel.update(source_child_id=c['id'],frozen_pair=True,method='nonnegative_real_bound',node_evaluations=0)
     report['old_passed_cells'].append(panel);sr['panel_ids']=[panel['id']];continue
    for old in c['descendants']:
     both=all(x['status']=='pass' for x in old['images'].values())
     panel={k:copy.deepcopy(old[k]) for k in ('id','bounds_z_t','real','imaginary','minimum_possible_width','node_pruned_no_width_gain')}
     panel.update(source_child_id=c['id'],frozen_pair=both,method='nonnegative_real_bound',images={},node_evaluations=0,classified=both)
     if both:panel['quadrature_error']=old['quadrature_error'];frozen_new+=1
     a,b,u,v=map(F,panel['bounds_z_t']);cz,ct,hz,ht=(a+b)/2,(u+v)/2,(b-a)/2,(v-u)/2
     for kind in ('z','t'):
      image=old['images'][kind]
      if image['status']=='pass':panel['images'][kind]=dict(status='pass',M=image['M'],rho=image['rho'],reused_from=dict(raw=str(RAW),leaf_id=old['id'],kind=kind))
      else:
       if image['rho']!=2:raise ValueError('Refused image radius')
       domain=[(cz-2*hz,cz+2*hz),(-2*hz,2*hz),(u,v)] if kind=='z' else [(a,b),(ct-2*ht,ct+2*ht),(-2*ht,2*ht)]
       boxes=tiles(domain)
       expected=[[(lo,(lo+hi)/2) if bit==0 else ((lo+hi)/2,hi) for (lo,hi),bit in zip(domain,bits)] for bits in product((0,1),repeat=3)]
       if boxes!=expected or len(boxes)!=8:raise ArithmeticError('Cartesian cover')
       im=dict(status='pending',rho=2,original_domain=domain,original_refusal=image['guard_reason'],tiles=[dict(index=i,bounds=box,status='unvisited') for i,box in enumerate(boxes)])
       panel['images'][kind]=im;planned+=8;missing_images.append((panel['id'],kind))
     report['leaves'].append(panel);leafmap[panel['id']]=panel;sr['panel_ids'].append(panel['id'])
  if (len(fixed),len(report['parents']),len(report['sources']),len(report['leaves']),len(report['old_passed_cells']),frozen_new,planned)!=(3798,443,1772,1088,1092,448,6512):raise ValueError('Wrong missing-only sets')
  if len([k for i,k in missing_images if k=='z'])!=640 or len([k for i,k in missing_images if k=='t'])!=174:raise ValueError('Wrong eligible images')
  reconstructed=[sums(fixed)[j]+sum((p['selected_sum'][j] for p in report['parents']),F(0)) for j in range(4)]
  if reconstructed!=frontier:raise ArithmeticError('Current coherent frontier')
  # Exact non-integrand cover/max/invalid-cover fixtures; rule and K controls are reused, not repeated.
  unit=[(F(-1),F(1)),(F(-2),F(2)),(F(0),F(3))];fixture=tiles(unit)
  if len(fixture)!=8 or fixture[0]!=[(F(-1),F(0)),(F(-2),F(0)),(F(0),F(3,2))] or fixture[-1]!=[(F(0),F(1)),(F(0),F(2)),(F(3,2),F(3))]:raise ArithmeticError('Cover fixture')
  def point_record(v):return dict(lower_exact=dict(numerator=str(v),denominator='1'),upper_exact=dict(numerator=str(v),denominator='1'))
  sample=[dict(status='pass',M=point_record(j)) for j in range(1,9)]
  if completed_max(sample)!=8:raise ArithmeticError('Maximum fixture')
  for invalid in (sample[:-1],sample[:-1]+[dict(status='refused')]):
   try:completed_max(invalid)
   except ValueError:pass
   else:raise ArithmeticError('Incomplete maximum admitted')
  if completed_max([dict(status='pass',M=point_record(3)) for _ in range(8)])!=3:raise ArithmeticError('Constant no-area-factor fixture')
  report['known_answers']=dict(cartesian_cover=True,maximum=True,incomplete_refused_cover_rejected=True,constant_no_factor_eight=True,old_rule_and_K='reused pinned accepted controls')
  report['planned_tile_calls']=planned;persist()
  import flint
  from flint import arb,acb,ctx
  ctx.prec,ctx.threads=192,1
  spec=importlib.util.spec_from_file_location('owned_fixed_mass_integrand',ENGINE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
  engine.arb,engine.acb=arb,acb;engine.REPORT=report;engine.START=start;engine.constants()
  def ball(record):a,b=ends(record);return engine.point(a).union(engine.point(b))
  def interval(a,b):return engine.point(a).union(engine.point(b))
  engine.NODES=[(ball(r['node']),ball(r['weight'])) for r in prior['gauss_nodes_and_weights']]
  if len(engine.NODES)!=8:raise ValueError('Pinned saved-rule length')
  report['ordered_node_weight_source']=prior['gauss_nodes_and_weights']
  original_positive,original_denominator=engine.positive_real,engine.denominator
  def positive(z,stage):
   if context is not None:context['last_guard']=dict(stage=stage,real=engine.interval(z.real),imaginary=engine.interval(z.imag)) if z.is_finite() else dict(stage=stage,nonfinite=True)
   value=original_positive(z,stage)
   if context is not None:
    margin=engine.rational(value.real.lower());previous=context['positive_real_margins'].get(stage)
    if previous is None or margin<previous:context['positive_real_margins'][stage]=margin
   return value
  def denominator(z):
   value=original_denominator(z)
   if context is not None:
    context['nonzero_denominator_checks']+=1;margin=engine.rational(abs(value).lower());old=context.get('least_denominator_absolute_lower')
    if old is None or margin<old:context['least_denominator_absolute_lower']=margin
   return value
  engine.positive_real,engine.denominator=positive,denominator
  report['guard_margin_scope']='Executed callback guards logged; unchanged manual inequalities are executed without separate exhaustive margin fields'
  def modulus(z,t):
   if report['counts']['evaluations']>=146032:raise ArithmeticError('Combined scientific-call ceiling')
   report['counts']['evaluations']+=1
   try:
    x=engine.finite((-z*engine.LOG2).exp());original=acb(1)/25+(engine.PI/2*t).sin()**2
    rt=abs(t.imag).upper();arg=arb.pi()*rt/2;sh=(arg.exp()-(-arg).exp())/2;ch=(arg.exp()+(-arg).exp())/2
    refined=acb((arb(1)/25-sh**2).lower().union((arb(1)/25+ch**2).upper()),arb(0,((2*arg).exp()-(-2*arg).exp())/4));Y=engine.intersect_images(original,refined)
    context.update(original_mass_image=dict(real=engine.interval(original.real),imaginary=engine.interval(original.imag)),refined_mass_rectangle=dict(real=engine.interval(refined.real),imaginary=engine.interval(refined.imag)),intersected_mass_image=dict(real=engine.interval(Y.real),imaginary=engine.interval(Y.imag)))
    value=engine.entropy(1+Y,(engine.PI/2*x).sin(),Y,(engine.PI/2*x).cos(),True,modulus=True)
    return engine.finite((value*abs(engine.LOG2*x/engine.SQRT2).upper()).upper())
   except engine.NonanalyticImage as error:
    report['counts']['refused_images']+=1;r=report['counts']['refusals_by_reason'];r[error.reason]=r.get(error.reason,0)+1;raise
  def floor(optimistic=False):
   value=G+A0
   for sr in report['sources']:
    ids=sr['panel_ids']
    if len(ids)==1 and ids[0] not in leafmap:continue
    minima=sum((number(leafmap[i]['minimum_possible_width']) if leafmap[i]['classified'] else F(0) if optimistic else width(leafmap[i]) for i in ids),F(0))
    rc=ends(sr['original_source_real'])[1]-ends(sr['original_source_real'])[0];value+=min(rc,minima)
   return value
  order=sorted((p for p in report['leaves'] if not p['frozen_pair']),key=lambda p:(-rankings[p['id']],p['id']))
  early=False
  for panel in order:
   for kind in ('z','t'):
    im=panel['images'][kind]
    if im['status']=='pass':continue
    for tile in im['tiles']:
     bounds=tile['bounds']
     if kind=='z':z,t=acb(interval(*bounds[0]),interval(*bounds[1])),acb(interval(*bounds[2]))
     else:z,t=acb(interval(*bounds[0])),acb(interval(*bounds[1]),interval(*bounds[2]))
     mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
     try:
      context=dict(positive_real_margins={},nonzero_denominator_checks=0);tile['witness']=context
      report['counts']['tile_attempted']+=1;report['counts']['tile_unvisited']-=1
      try:
       M=modulus(z,t);tile.update(status='pass',M=engine.interval(M));report['counts']['tile_returned']+=1;report['counts']['tile_passed']+=1
      except engine.NonanalyticImage as error:
       tile.update(status='refused',guard_reason=error.reason,image=error.image,input_image=error.input_image);report['counts']['tile_refused']+=1
      finally:context=None
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
     if tile['status']=='refused':break
     checkpoint()
    if all(t['status']=='pass' for t in im['tiles']):
     maximum=completed_max(im['tiles']);im.update(status='pass',M=engine.rational_interval(maximum,maximum),unvisited_indices=[])
    else:im.update(status='refused',unvisited_indices=[t['index'] for t in im['tiles'] if t['status']=='unvisited'])
   if all(im['status']=='pass' for im in panel['images'].values()):
    a,b,u,v=map(F,panel['bounds_z_t']);Mz,Mt=ball(panel['images']['z']['M']),ball(panel['images']['t']['M']);kt=K2 if panel['images']['t']['rho']==2 else K4
    e=(engine.point((b-a)*(v-u))*(engine.point(K2)*Mz+engine.point(kt)*Mt)).upper();q=2*engine.rational(e)
    panel['quadrature_error']=engine.interval(e);panel['minimum_possible_width']=min(width(panel),q);panel['node_pruned_no_width_gain']=q>=width(panel)
   else:panel['minimum_possible_width']=width(panel);panel['node_pruned_no_width_gain']=True
   panel['classified']=True
   optimistic=floor(True);report['partial_optimistic_floor']=optimistic
   if optimistic>Brun:
    early=True;report.update(status='incomplete_image_cover_early_floor',floor_scope='partial optimistic fixed-cover lower bound',image_stage_complete=False)
    break
   checkpoint()
  if early:
   for panel in report['leaves']:
    for im in panel['images'].values():
     if im['status']=='pending':im.update(status='unvisited',unvisited_indices=list(range(8)))
  else:
   Fnew=floor();report.update(image_stage_complete=True,pre_node_floor=Fnew,floor_scope='complete image classification',pre_node_run_target_impossible=Fnew>Brun,pre_node_print_target_impossible=Fnew>Bprint)
   if Fnew>number(audit['floor']):raise ArithmeticError('Nonincreasing floor invariant')
   if frontier!=original_frontier:raise ArithmeticError('Image work changed real cover')
   persist()
   if Fnew<=Brun:
    allpanels={p['id']:p for p in report['leaves']+report['old_passed_cells']}
    for sr in report['sources']:
     work=[{k:copy.deepcopy(v) for k,v in allpanels[i].items() if k!='images'} for i in sr['panel_ids']];sr['working_panels']=work
     for panel in work:panel['image_witness_reference']=dict(result=str(target),panel_id=panel['id'],collection='leaves' if panel['id'] in leafmap else 'old_passed_cells')
     for panel in work:
      if panel['node_pruned_no_width_gain']:continue
      a,b,u,v=map(F,panel['bounds_z_t']);cz,ct,hz,ht=(a+b)/2,(u+v)/2,(b-a)/2,(v-u)/2;acc=acb(0)
      panel['node_progress']=dict(completed_indices=[],next_index=0,partial_real=engine.interval(acc.real),partial_imaginary=engine.interval(acc.imag),ordered_source=str(PRIOR),quadrature_error=panel['quadrature_error'])
      try:
       for index,(x,wx,y,wy) in enumerate((x,wx,y,wy) for x,wx in engine.NODES for y,wy in engine.NODES):
        if report['counts']['evaluations']>=146032:raise ArithmeticError('Combined scientific-call ceiling')
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        try:
         report['counts']['node_attempted']+=1
         try:val=engine.evaluate(acb(engine.point(cz)+engine.point(hz)*x),acb(engine.point(ct)+engine.point(ht)*y),dict(direction='110'),'gauss_node')
         except engine.NonanalyticImage as error:
          panel['node_refusal']=dict(index=index,reason=error.reason,image=error.image,input_image=error.input_image);raise
         acc+=wx*wy*val;panel['node_evaluations']+=1;report['counts']['node_completed']+=1
         p=panel['node_progress'];p['completed_indices'].append(index);p['next_index']=index+1;p['partial_real'],p['partial_imaginary']=engine.interval(acc.real),engine.interval(acc.imag)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
       e=ball(panel['quadrature_error']);value=engine.point(hz)*engine.point(ht)*acc+acb(arb(0,e),arb(0,e))
       if not value.is_finite() or not value.imag.contains(0):raise engine.NonanalyticImage('nonreal_integral',value)
       lo,hi=engine.rational(value.real.lower()),engine.rational(value.real.upper())
       if hi-lo<2*number(panel['quadrature_error']['upper_exact']):raise ArithmeticError('Width below fixed allowance')
       panel['gauss_real'],panel['gauss_imaginary']=engine.interval(value.real),engine.interval(value.imag)
       if hi-lo<width(panel):panel.update(real=panel['gauss_real'],imaginary=panel['gauss_imaginary'],method='tensor_gauss_image_cover' if panel['id'] in leafmap else 'tensor_gauss_old_T8')
      except engine.NonanalyticImage as error:panel.setdefault('node_refusal',dict(reason=error.reason,image=error.image,input_image=error.input_image))
      checkpoint()
     proposal=sums(work);mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
     try:
      if proposal[1]-proposal[0]<width(sr):sr.update(real=engine.rational_interval(proposal[0],proposal[1]),imaginary=engine.rational_interval(proposal[2],proposal[3]),adopted_panel_ids=[p['id'] for p in work])
      parent=parentmap[sr['parent_id']];sources=[sourcemap[i] for i in parent['source_ids']];proposal=sums(sources);old=parent['selected_sum']
      if len(sources)!=4:raise ArithmeticError('Incomplete parent cover')
      if proposal[1]-proposal[0]<old[1]-old[0]:
       frontier=[a-b+c for a,b,c in zip(frontier,old,proposal)];parent['selected_sum']=proposal;parent['selected_cover']=[dict(id=c['id'],real=c['real'],imaginary=c['imaginary']) for c in sources]
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
     checkpoint()
   w=frontier[1]-frontier[0];report['retained_width']=w
   if not frontier[2]<=0<=frontier[3] or w<Fnew:raise ArithmeticError('Final frontier consistency')
   if w<=Brun:
    report['status']='printed_target_met' if w<=Bprint else 'original_target_only';report['results']['eta110_y1_25']=engine.rational_interval(frontier[0],frontier[1]+omission)
   else:report['status']='incomplete_image_cover_floor' if Fnew>Brun else 'incomplete_image_cover_width'
 except Exception as error:report.update(status='incomplete',error=type(error).__name__+': '+str(error))
 finally:
  signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(600,600));persist()
 print(json.dumps(dict(status=report['status'],counts=report['counts'],image_stage_complete=report['image_stage_complete'])))
 if report['status'] not in ('printed_target_met','original_target_only'):raise SystemExit(1)

if __name__=='__main__':main()
