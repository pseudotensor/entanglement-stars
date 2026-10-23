"""T17--T20 missing-only positive-moment allocation; run only on box."""
from fractions import Fraction as F
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

BASE=Path('<internal>/eta_massive_scalar_v1')
OLD=BASE/'environment_continuation'
ROOT=OLD/'positive_moment_v1'
SOURCE=OLD/'fixed_mass_sharp_repair_result.json'
GEN=OLD/'separate_images_v1/separate_images_result.json'
AUDIT=OLD/'separate_images_v1/saved_t9_floor_attribution.json'
ENGINE=BASE/'fixed_mass_interval_sharp_repair.py'
PINS={SOURCE:'0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e',
 GEN:'276090a9343faa76b34710c78f3a2d56ebe71357b42e1e96fd6c794d2972d09f',
 AUDIT:'5d7c7b31f5b708bc105d7bd9c8c3c7225e2f85aa16c42895f4dad151aae772fd',
 ENGINE:'d962ea9b594c2495284069eb2814ba6d6263786c374bd1b51475d15b29f3e870'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x):return F(int(x['numerator']),int(x['denominator']))
def ends(x):return tuple(num(x[k]) for k in ('lower_exact','upper_exact'))
def vec(x):return [*ends(x['real']),*ends(x['imaginary'])]
def width(x):return ends(x['real'])[1]-ends(x['real'])[0]
def pack(x):
 if isinstance(x,F):return dict(numerator=str(x.numerator),denominator=str(x.denominator))
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x

def main():
 if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory required')
 target=ROOT/'positive_moment_result.json'
 if target.exists() or target.with_suffix('.json.tmp').exists():raise FileExistsError('No retry/overwrite')
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[allowed[0]])
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
 resource.setrlimit(resource.RLIMIT_CPU,(290,300))
 def deadline(s,f):raise TimeoutError('Bounded positive-moment deadline')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(345)
 start=time.monotonic();frontier=None;engine=None;context=None
 report=dict(status='incomplete',y='1/25',purpose='SecIII G5/I6a missing110',method='T17--T20 positive-moment radius2',
  source_pins={str(p):h for p,h in PINS.items()},parents=[],geometry=[],results={},
  counts=dict(evaluations=0,refused_images=0,refusals_by_reason={},new_modulus=0,new_nodes=0,new_node_attempts=0),
  guard_samples={},tail_radius_maxima={},replaced_parent_ids=[],
  resources=dict(threads=1,nice=10,address_space_bytes=2*2**30,cpu_soft=290,cpu_hard=300,wall_internal=345,wall_external=360))
 def persist():
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  try:
   report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    producer_sha256=sha(Path(__file__)))
   if frontier is not None:report['frontier_real'],report['frontier_imaginary']=frontier[:2],frontier[2:]
   tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(pack(report),indent=2,allow_nan=False)+'\n');tmp.replace(target)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 def sums(records):return [sum((vec(x)[i] for x in records),F(0)) for i in range(4)]
 def commit(entry):
  nonlocal frontier
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  try:
   if sorted(x['id'] for x in entry['children'])!=sorted(entry['expected_source_child_ids']):raise ArithmeticError('Incomplete source-child cover')
   proposal=sums(entry['children']);previous=entry['retained_sum']
   if proposal[1]-proposal[0]<previous[1]-previous[0]:
    frontier=[a-b+c for a,b,c in zip(frontier,previous,proposal)]
    entry['retained_sum']=proposal;entry['retained_cover']=[dict(id=x['id'],real=copy.deepcopy(x['real']),imaginary=copy.deepcopy(x['imaginary']),method=x['method']) for x in entry['children']]
    if entry['source_parent_id'] not in report['replaced_parent_ids']:report['replaced_parent_ids'].append(entry['source_parent_id'])
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 last_checkpoint=start;last_checkpoint_unit=None
 def checkpoint(completed):
  nonlocal last_checkpoint,last_checkpoint_unit
  if (completed>0 and completed%16==0 and completed!=last_checkpoint_unit) or time.monotonic()-last_checkpoint>=10:
   persist();last_checkpoint=time.monotonic();last_checkpoint_unit=completed
 try:
  for p,h in PINS.items():
   if sha(p)!=h:raise ValueError('Changed input '+str(p))
  source=json.loads(SOURCE.read_text());gen=json.loads(GEN.read_text());audit=json.loads(AUDIT.read_text())
  if audit['status']!='pass' or gen['status']!='incomplete_one_generation_floor':raise ValueError('Unaccepted predecessor')
  fixed=[x for x in source['active_panel']['leaves'] if x['method']=='tensor_gauss']
  oldparents={x['id']:x for x in source['active_panel']['leaves'] if x['method']=='nonnegative_real_bound'}
  children=[c for p in gen['parents'] for c in p['children']]
  passed=[c for c in children if all(i['status']=='pass' for i in c['images'].values())]
  refused=[c for c in children if c not in passed]
  if (len(fixed),len(oldparents),len(children),len(passed),len(refused))!=(3798,443,1772,1092,680):raise ValueError('Wrong missing-only sets')
  report['unchanged_gauss_source_ids']=[x['id'] for x in fixed]
  report['unchanged_passed_pair_ids']=[x['id'] for x in passed]
  report['reused_t_image_ids']=[x['id'] for x in refused if x['images']['separate_t_image']['status']=='pass']
  G=sum((width(x) for x in fixed),F(0));omission=ends(source['omitted_integral_bounds']['eta110_nonnegative'])[1]
  Brun=min(F(1,10000),F(1,5000)-omission);Bprint=F(8,1000000)-omission
  report.update(fixed_gauss_width=G,omitted_upper=omission,retained_run_budget=Brun,retained_print_budget=Bprint)
  frontier=sums(source['active_panel']['leaves'])
  if tuple(frontier[:2])!=ends(source['active_panel']['real']) or tuple(frontier[2:])!=ends(source['active_panel']['imaginary']):raise ValueError('Frontier sum mismatch')
  total=0;newimages=0
  for c in refused:
   a,b,u,v=map(F,c['bounds_z_t']);nz=nt=1;tp=c['images']['separate_t_image']['status']=='pass'
   while (b-a)/nz>F(1,2):nz*=2
   if not tp:
    while (v-u)/nt>F(1,8):nt*=2
   bounds=[[a+(b-a)*i/nz,a+(b-a)*(i+1)/nz,u+(v-u)*j/nt,u+(v-u)*(j+1)/nt] for i in range(nz) for j in range(nt)]
   if len(bounds)!=nz*nt or sum(((z1-z0)*(t1-t0) for z0,z1,t0,t1 in bounds),F(0))!=(b-a)*(v-u):raise ValueError('Cover mismatch')
   report['geometry'].append(dict(source_child_id=c['id'],nz=nz,nt=nt,reuse_t=tp,bounds=bounds))
   total+=nz*nt;newimages+=nz*nt*(1 if tp else 2)
  report.update(planned_new_leaves=total,planned_new_images=newimages,maximum_new_nodes=64*(total+1092))
  persist()
  if total>4096:raise RuntimeError('Incomplete geometry preflight: N cap')
  import flint
  from flint import arb,acb,ctx
  ctx.prec,ctx.threads=192,1
  spec=importlib.util.spec_from_file_location('owned_fixed_mass_integrand',ENGINE)
  engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
  engine.arb,engine.acb=arb,acb;engine.REPORT=report;engine.START=start;engine.constants()
  def ball(r):
   a,b=ends(r);return engine.point(a).union(engine.point(b))
  engine.NODES=[(ball(x['node']),ball(x['weight'])) for x in source['gauss_nodes_and_weights']]
  if len(engine.NODES)!=8 or not all(x>-1 and x<1 and w>0 for x,w in engine.NODES):raise ValueError('Saved Gauss rule')
  report['ordered_node_weight_source']=source['gauss_nodes_and_weights']
  P={0:F(35,6435),2:F(-1260,6435),4:F(6930,6435),6:F(-12012,6435),8:F(1)}
  I=lambda j:F(0) if j%2 else F(1,j+1)
  if any(sum((p*I(k+j) for k,p in P.items()),F(0)) for j in range(8)):raise ArithmeticError('Orthogonality')
  d16=F(2**16*math.factorial(8)**4,17*math.factorial(16)**2)
  if sum((p*q*I(k+j) for k,p in P.items() for j,q in P.items()),F(0))!=d16:raise ArithmeticError('Norm')
  def poly(r):return sum((p*r**k for k,p in P.items()),F(0))
  def K(r):return r/(r-1)*d16/poly(r)**2
  if poly(F(2))!=F(984467,6435) or K(F(2))!=F(32768,17*984467**2):raise ArithmeticError('K2')
  controls=[]
  for j in range(16):
   q=sum((w*x**j for x,w in engine.NODES),arb(0))
   if not q.contains(engine.point(2*I(j))):raise ArithmeticError('Gauss moment')
  for area in (F(1),F(6)):
   raw=[sum((wx*wy*x**j*y**k for x,wx in engine.NODES for y,wy in engine.NODES),arb(0))*engine.point(area/4) for j,k in ((0,0),(14,14),(16,0))]
   deficit=engine.point(area/17)-raw[2]
   if not raw[0].contains(engine.point(area)) or not raw[1].contains(engine.point(area/225)) or not deficit.contains(engine.point(area*d16)) or not deficit.lower()>0:raise ArithmeticError('Tensor controls')
   for rt in (F(2),F(4)):
    # The enclosing square has |u|<=sqrt(2)*rho, so Mz=(sqrt(2)*2)^16=2^24.
    e=area*(K(F(2))*2**24+K(rt))
    if not deficit.upper()<=engine.point(e):raise ArithmeticError('Positive moment allowance')
    controls.append(dict(area=area,rho_t=rt,raw=[engine.interval(x) for x in raw],deficit=engine.interval(deficit),exact_deficit=area*d16,allowance=e))
  report['known_answers']=dict(orthogonality='pass',norm=d16,P2=poly(F(2)),K2=K(F(2)),K4=K(F(4)),tensor=controls)
  def evaluate(z,t,kind):
   nonlocal context
   if report['counts']['evaluations']>=340224:raise ArithmeticError('Evaluation cap')
   report['counts']['evaluations']+=1;report['counts']['new_modulus']+=1
   try:
    x=engine.finite((-z*engine.LOG2).exp());phi=engine.PI/2*t
    original=acb(1)/25+phi.sin()**2
    rt=abs(t.imag).upper();arg=arb.pi()*rt/2
    sh=(arg.exp()-(-arg).exp())/2;ch=(arg.exp()+(-arg).exp())/2
    refined=acb((arb(1)/25-sh**2).lower().union((arb(1)/25+ch**2).upper()),arb(0,((2*arg).exp()-(-2*arg).exp())/4))
    Y=engine.intersect_images(original,refined)
    context.update(original_mass_image=dict(real=engine.interval(original.real),imaginary=engine.interval(original.imag)),
     refined_mass_rectangle=dict(real=engine.interval(refined.real),imaginary=engine.interval(refined.imag)),
     intersected_mass_image=dict(real=engine.interval(Y.real),imaginary=engine.interval(Y.imag)))
    s=engine.entropy(1+Y,(engine.PI/2*x).sin(),Y,(engine.PI/2*x).cos(),True,modulus=True)
    return engine.finite((s*abs(engine.LOG2*x/engine.SQRT2).upper()).upper())
   except engine.NonanalyticImage as e:
    report['counts']['refused_images']+=1
    reasons=report['counts']['refusals_by_reason'];reasons[e.reason]=reasons.get(e.reason,0)+1
    raise
  original_positive,original_denominator=engine.positive_real,engine.denominator
  def positive(z,stage):
   if context is not None:
    context['last_guard']=dict(stage=stage,real=engine.interval(z.real),imaginary=engine.interval(z.imag)) if z.is_finite() else dict(stage=stage,nonfinite=True)
   value=original_positive(z,stage)
   if context is not None:
    margin=engine.rational(value.real.lower());prior=context['positive_real_margins'].get(stage)
    if prior is None or margin<prior:context['positive_real_margins'][stage]=margin
   return value
  def denominator(z):
   value=original_denominator(z)
   if context is not None:
    context['nonzero_denominator_checks']+=1;margin=engine.rational(abs(value).lower())
    prior=context.get('least_denominator_absolute_lower')
    if prior is None or margin<prior:context['least_denominator_absolute_lower']=margin
   return value
  engine.positive_real,engine.denominator=positive,denominator
  report['guard_margin_scope']='Callbacks record executed positive-real and denominator guards; unchanged manual branch inequalities are not claimed as separate logged margins'
  geometry={r['source_child_id']:r for r in report['geometry']}
  nextid=max(c['id'] for c in children)
  for parent_index,p in enumerate(gen['parents'],1):
   old=oldparents[p['source_parent_id']]
   entry=dict(source_parent_id=old['id'],old_real=old['real'],old_imaginary=old['imaginary'],retained_sum=vec(old),retained_cover='old_parent',expected_source_child_ids=[c['id'] for c in p['children']],children=[])
   report['parents'].append(entry)
   for c in p['children']:
    cell=copy.deepcopy(c);cell['old_source_real']=copy.deepcopy(c['real']);cell['old_source_imaginary']=copy.deepcopy(c['imaginary']);cell['descendants']=[];cell['retained_source_sum']=vec(c)
    entry['children'].append(cell)
    if c['id'] in geometry:
     for bounds in geometry[c['id']]['bounds']:
      nextid+=1;a,b,u,v=bounds;cz,ct,hz,ht=(a+b)/2,(u+v)/2,(b-a)/2,(v-u)/2
      value,radius=engine.real_bound(bounds)
      leaf=dict(id=nextid,source_child_id=c['id'],bounds_z_t=list(map(str,bounds)),real=engine.interval(value.real),imaginary=engine.interval(value.imag),method='nonnegative_real_bound',images={},node_evaluations=0)
      cell['descendants'].append(leaf)
      cases=[('z',acb(arb(engine.point(cz),2*engine.point(hz)),arb(0,2*engine.point(hz))),acb(arb(engine.point(ct),engine.point(ht))))]
      if geometry[c['id']]['reuse_t']:
       leaf['images']['t']=dict(status='pass',M=c['images']['separate_t_image']['M'],reused_source_child_id=c['id'],rho=4)
      else:cases.append(('t',acb(arb(engine.point(cz),engine.point(hz))),acb(arb(engine.point(ct),2*engine.point(ht)),arb(0,2*engine.point(ht)))))
      for kind,z,t in cases:
       context=dict(rho=2,positive_real_margins={},nonzero_denominator_checks=0);leaf['images'][kind]=context
       mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
       try:
        try:context.update(status='pass',M=engine.interval(evaluate(z,t,kind)))
        except engine.NonanalyticImage as e:context.update(status='refused',guard_reason=e.reason,image=e.image,input_image=e.input_image)
       finally:
        context=None;signal.pthread_sigmask(signal.SIG_SETMASK,mask)
      if all(x['status']=='pass' for x in leaf['images'].values()):
       mz,mt=ball(leaf['images']['z']['M']),ball(leaf['images']['t']['M'])
       e=(engine.point((b-a)*(v-u))*(engine.point(K(F(2)))*mz+engine.point(K(F(leaf['images']['t']['rho'])))*mt)).upper()
       leaf['quadrature_error']=engine.interval(e);q=2*engine.rational(e)
       leaf['minimum_possible_width']=min(q,width(leaf));leaf['node_pruned_no_width_gain']=q>=width(leaf)
      else:leaf.update(minimum_possible_width=width(leaf),node_pruned_no_width_gain=True)
     ds=sums(cell['descendants'])
     if ds[1]-ds[0]<width(cell):cell['real']=engine.rational_interval(ds[0],ds[1]);cell['imaginary']=engine.rational_interval(ds[2],ds[3]);cell['method']='complete_fallback_descendants'
   commit(entry)
   checkpoint(parent_index)
  def cellfloor(c):
   if not c['descendants']:return num(c['minimum_possible_width'])
   return min(ends(c['old_source_real'])[1]-ends(c['old_source_real'])[0],sum((x['minimum_possible_width'] for x in c['descendants']),F(0)))
  # Reused minima were serialized as rational records; new minima remain Fractions.
  floor=G+sum((cellfloor(c) for p in report['parents'] for c in p['children']),F(0))
  report.update(pre_node_floor=floor,pre_node_run_target_impossible=floor>Brun,pre_node_print_target_impossible=floor>Bprint,image_stage_complete=True)
  persist()
  if floor<=Brun:
   completed_node_cells=0
   for p in report['parents']:
    for c in p['children']:
     work=copy.deepcopy(c['descendants'] or [c]);c['node_working_cover']=work
     for leaf in work:
      if leaf['node_pruned_no_width_gain']:continue
      a,b,u,v=map(F,leaf['bounds_z_t']);cz,ct,hz,ht=(a+b)/2,(u+v)/2,(b-a)/2,(v-u)/2
      acc=acb(0);leaf['node_progress']=dict(completed_indices=[],next_index=0,partial_real=engine.interval(acc.real),partial_imaginary=engine.interval(acc.imag),quadrature_error=leaf['quadrature_error'],source=str(SOURCE),method='saved8 tensor T19 or original T8')
      try:
       for index,(x,wx,y,wy) in enumerate((x,wx,y,wy) for x,wx in engine.NODES for y,wy in engine.NODES):
        if report['counts']['evaluations']>=340224:raise ArithmeticError('Evaluation cap')
        # Mask the complete evaluation AND returned-value publication, including weighted accumulation.
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        try:
         report['counts']['new_node_attempts']+=1
         try:
          val=engine.evaluate(acb(engine.point(cz)+engine.point(hz)*x),acb(engine.point(ct)+engine.point(ht)*y),dict(direction='110'),'gauss_node')
         except engine.NonanalyticImage as e:
          leaf['node_refusal']=dict(reason=e.reason,image=e.image,input_image=e.input_image,node_index=index)
          raise
         acc+=wx*wy*val;leaf['node_evaluations']+=1;report['counts']['new_nodes']+=1
         progress=leaf['node_progress'];progress['completed_indices'].append(index);progress['next_index']=index+1
         progress['partial_real'],progress['partial_imaginary']=engine.interval(acc.real),engine.interval(acc.imag)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
       e=ball(leaf['quadrature_error']);val=engine.point(hz)*engine.point(ht)*acc+acb(arb(0,e),arb(0,e))
       if not val.is_finite() or not val.imag.contains(0):raise engine.NonanalyticImage('nonreal_integral',val)
       lo,hi=engine.rational(val.real.lower()),engine.rational(val.real.upper())
       if hi-lo<2*num(leaf['quadrature_error']['upper_exact']):raise ArithmeticError('Width below allowance')
       leaf['gauss_real'],leaf['gauss_imaginary']=engine.interval(val.real),engine.interval(val.imag)
       if hi-lo<width(leaf):leaf.update(real=leaf['gauss_real'],imaginary=leaf['gauss_imaginary'],method='tensor_gauss_positive_moment' if c['descendants'] else 'tensor_gauss_original_T8')
      except engine.NonanalyticImage as e:leaf.setdefault('node_refusal',dict(reason=e.reason,image=e.image,input_image=e.input_image))
      completed_node_cells+=1;checkpoint(completed_node_cells)
     proposal=sums(work)
     mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
     try:
      if proposal[1]-proposal[0]<width(c):
       c['real']=engine.rational_interval(proposal[0],proposal[1]);c['imaginary']=engine.rational_interval(proposal[2],proposal[3]);c['method']='complete_node_cover';c['adopted_node_cover_ids']=[x['id'] for x in work]
      commit(p)
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
     checkpoint(completed_node_cells)
  w=frontier[1]-frontier[0]
  if w<floor or not frontier[2]<=0<=frontier[3]:raise ArithmeticError('Frontier consistency')
  report['retained_width']=w
  if w<=Brun:
   report['status']='printed_target_met' if w<=Bprint else 'original_target_only';report['results']['eta110_y1_25']=engine.rational_interval(frontier[0],frontier[1]+omission)
  else:report['status']='incomplete_positive_moment_floor' if floor>Brun else 'incomplete_positive_moment_width'
 except Exception as e:report.update(status='incomplete',error=type(e).__name__+': '+str(e))
 finally:
  signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(300,300));persist()
 print(json.dumps(dict(status=report['status'],counts=report['counts'],planned_new_leaves=report.get('planned_new_leaves'))))
 if report['status'] not in ('printed_target_met','original_target_only'):raise SystemExit(1)

if __name__=='__main__':main()
