"""Exact T38 replay of immutable saved b110 evidence; no kernel calls."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

ROOT=Path('<internal>/b110_saved_inventory_v1')
INPUT=ROOT.parent/'b110_subtracted_v1'
PINS={'p110_result.json':'5a272465b133303b29220a33c7f68437963741259e43f3ea7468356dccd8d938','b110_execution.json':'fe9fa6b763d5e6ed356594cf2a87f24705d836c0fbcf343ed88fc19a61ff3c70','p110_attempts.jsonl':'09c06f02730006e6a1a37e13b3b3d1414866d8b915e943f96ebb0af0448dff47'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
LIMIT=8*2**20
BUDGET=F(1,50000)
GUARDS={'nonpositive_real_sqrt_argument','nonpositive_real_p','nonpositive_real_m','denominator_contains_zero'}
EVALUATIONS={'nonfinite_RF','nonfinite_elliptic_value','nonfinite_complete_RD_p'}


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def check(condition,message):
 if not condition:raise ValueError(message)

def endpoint(record):return F(int(record['numerator']),int(record['denominator']))
def ends(record):return endpoint(record['lower_exact']),endpoint(record['upper_exact'])
def identity(path):
 s=path.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def packed(value):
 if isinstance(value,F):return str(value)
 if isinstance(value,dict):return {str(k):packed(v) for k,v in value.items()}
 if isinstance(value,(tuple,list,set)):return [packed(v) for v in value]
 return value

def cell(record):
 real=ends(record['real']);imag=ends(record['imaginary'])
 accumulator=record.get('raw_weighted_accumulator')
 check(accumulator is not None,'Missing saved accumulator')
 raw_acc=(*ends(accumulator['real']),*ends(accumulator['imaginary']))
 check(all(x==0 for x in raw_acc),'Unexpected nonzero accumulator')
 check(not record.get('completed_node_indices') and not record.get('node_refs'),'Unexpected saved node progress')
 check(not any(k in record for k in ('analytic_M','quadrature_error','joint_ref')),'Unexpected returned analytic rule')
 check(record['method']=='signed_real_bound','Unexpected selected method')
 return dict(id=record['id'],bounds=tuple(map(F,record['bounds_z_t'])),depths=tuple(record['depths']),method=record['method'],real=real,imaginary=imag,width=real[1]-real[0],partial_rule_status=record['partial_rule_status'],completed_node_indices=record['completed_node_indices'],node_refs=record['node_refs'],accumulator=raw_acc,guard_reason=record.get('guard_reason'),probe_refs=record.get('probe_refs',[]))

def halves(parent,axis):
 a,b,c,d=parent['bounds']
 return ((a,(a+b)/2,c,d),((a+b)/2,b,c,d)) if axis==0 else ((a,b,c,(c+d)/2),(a,b,(c+d)/2,d))

def adopt(selected,cells,parent,children):
 check(parent in selected and len(children)==2 and len(set(children))==2,'Invalid adoption membership')
 check(all(i in cells and cells[i]['parent']==parent and cells[i]['finished'] for i in children),'Unfinished or wrong-parent children')
 p=cells[parent]['created']
 for axis in (0,1):
  pair=halves(p,axis);depths=tuple(v+(i==axis) for i,v in enumerate(p['depths']))
  if tuple(cells[i]['created']['bounds'] for i in children)==pair and all(cells[i]['created']['depths']==depths for i in children):break
 else:raise ValueError('Adoption is not a complete exact midpoint pair')
 selected.remove(parent);selected.update(children)

def width_classes(rows):
 groups={}
 for row in rows:
  group=groups.setdefault(row['joint_reason'],dict(count=0,width=F(0)))
  group['count']+=1;group['width']+=row['width']
 total=sum((g['width'] for g in groups.values()),F(0))
 return total,groups

def four_sums(rows):
 return tuple(sum((r[field][side] for r in rows),F(0)) for field in ('real','imaginary') for side in (0,1))

def fixtures():
 cells={1:dict(parent=None,finished=True,created=dict(bounds=(F(0),F(1),F(0),F(1)),depths=(0,0)))};selected={1}
 def child(i,parent,bounds,depths):cells[i]=dict(parent=parent,finished=True,created=dict(bounds=bounds,depths=depths))
 left,right=halves(cells[1]['created'],0);child(2,1,left,(1,0));child(3,1,right,(1,0));adopt(selected,cells,1,[2,3])
 low,high=halves(cells[3]['created'],1);child(4,3,low,(1,1));child(5,3,high,(1,1));adopt(selected,cells,3,[4,5]);check(selected=={2,4,5},'Three-rectangle replay fixture')
 def toy_sums():
  rows=[]
  for i in selected:
   a,b,c,d=cells[i]['created']['bounds'];area=(b-a)*(d-c)
   rows.append(dict(real=(-area,2*area),imaginary=(-3*area,4*area)))
  return four_sums(rows)
 before=selected.copy();before_sums=toy_sums();check(before_sums==(F(-1),F(2),F(-3),F(4)),'Toy selected endpoint known answers')
 child(6,2,halves(cells[2]['created'],0)[0],(2,0));check(selected==before and toy_sums()==before_sums,'Pending child changed selected cover or endpoint sums')
 total,groups=width_classes([dict(joint_reason='a',width=F(1)),dict(joint_reason='a',width=F(2)),dict(joint_reason='b',width=F(3))])
 check(total==6 and groups['a']['width']==groups['b']['width']==3 and total-groups['a']['width']==3 and total-sum(g['width'] for g in groups.values())==0,'Exact attribution fixture')
 return dict(three_rectangle_replay=True,pending_child_excluded=True,pending_child_endpoint_sums=before_sums,class_totals=dict(a=F(3),b=F(3)),total=total,residual_a=F(3),residual_all=F(0))

def main():
 check(Path(__file__).resolve().parent==ROOT,'Named box directory only')
 target=ROOT/'inventory_result.json'
 if target.exists() or target.with_suffix('.json.tmp').exists():raise FileExistsError('Never overwrite or retry a bounded audit')
 resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(55,60))
 cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[0]])
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 def deadline(s,f):raise TimeoutError('Bounded exact saved-data audit')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(85)
 start=time.monotonic();result=dict(status='incomplete',scope='T38 exact saved first-panel cover and width attribution; no b110 coefficient or kernel call',phase='fixtures',scientific_evaluations=0,sources=[],cover_verified=False,width_decomposition_verified=False,counts={},checks={},selected_rows=[],pending_rows=[])
 selected=set();cells={};pending=set();events=Counter();refusals=Counter();kinds=Counter();fixture_rows=[];attempts=0;sequences=0;coherent_sequences=0
 def save():
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  try:
   result['counts']=dict(events=dict(events),attempts=attempts,refusals=dict(refusals),kinds=dict(kinds),selected=len(selected),pending=len(pending),returned_probes=sum(r['outcome']=='returned' for c in cells.values() for r in c['probes']))
   result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=digest(Path(__file__)),replay_prefix=dict(completed_records=coherent_sequences,attempted_record=sequences,current_event=result.get('current_event'),cells=len(cells),selected_ids=sorted(selected),pending_ids=sorted(pending),samples=[dict(id=i,parent=cells[i]['parent'],saved=cells[i]['finished'] or cells[i]['created'],probes=cells[i]['probes'],registry=cells[i]['registry']) for i in sorted(selected|pending)[:32]]))
   blob=json.dumps(packed(result),separators=(',',':'),allow_nan=False).encode()
   if len(blob)+1>LIMIT:
    result.update(status='incomplete',error='Preencoded result exceeds fixed eight-MiB cap',cover_verified=False,width_decomposition_verified=False,selected_rows=result['selected_rows'][:32],checks=dict(partial_only=True,full_inventory_complete=False))
    for key in ('groups','class_removal_residuals','union_removal_residuals','endpoint_sums','total_width'):result.pop(key,None)
    blob=json.dumps(packed(result),separators=(',',':'),allow_nan=False).encode();check(len(blob)+1<=LIMIT,'Bounded incomplete result too large')
   tmp=target.with_suffix('.json.tmp');tmp.write_bytes(blob+b'\n');tmp.replace(target)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 try:
  result['known_answers']=fixtures();loaded={};snapshots={}
  for name in ('p110_result.json','b110_execution.json'):
   path=INPUT/name;snapshots[name]=identity(path);raw=path.read_bytes();check(hashlib.sha256(raw).hexdigest()==PINS[name] and identity(path)==snapshots[name],'Changed pinned saved input')
   loaded[name]=json.loads(raw);result['sources'].append(dict(path=str(path),sha256=PINS[name],**snapshots[name]))
  native=loaded['p110_result.json'];execution=loaded['b110_execution.json'];check(native['status']=='incomplete' and not native['panels'] and not native['results'] and execution['exit_code']==1,'Changed native terminal scope')
  check(execution['p110_result.json']['sha256']==PINS['p110_result.json'] and execution['p110_attempts.jsonl']['sha256']==PINS['p110_attempts.jsonl'],'Execution does not bind saved data')
  check(execution['producer_sha256']=='3a3ed16ffce277b8109f9ff16abe70b7231c73e9ed6ad5ebd386aa8bea0b4992' and execution['wrapper_sha256']=='a9cfa04ee08cb0f9dd941a725b79fe1186107742fb6340f846bb3bb499bd2b92','Changed native source pins')
  result['phase']='journal_replay';save();path=INPUT/'p110_attempts.jsonl';before=identity(path);h=hashlib.sha256()
  def process_line(line):
   nonlocal attempts,sequences
   check(line.endswith(b'\n'),'Incomplete terminal journal line');h.update(line);row=json.loads(line);sequences+=1;check(row['sequence']==sequences,'Nonconsecutive journal sequence');event=row['event'];result['current_event']=dict(sequence=sequences,event=event);events[event]+=1
   if event=='evaluation':
    attempts+=1;check(row['attempt']==attempts,'Nonconsecutive or repeated attempt identity');context=row['context'];kind=context['kind'];kinds[kind]+=1
    if row['outcome']=='refused':refusals[row['reason']]+=1
    if kind=='new_derivative_known_answer':
     check(row['outcome']=='returned' and row['raw']['pass_fixture'] is True,'Saved fixture failed');fixture_rows.append(dict(journal_ref=sequences,**row['raw']));return
    check(context['panel']==0 and context['cell'] in cells,'Unbound evaluation context');c=cells[context['cell']]
    if kind=='joint_analytic_box':
     check(c.get('joint') is None and row['outcome']=='refused','Repeated or returned joint image');c['joint']=dict(ref=sequences,reason=row['reason'],context_ref=sequences)
    elif kind in ('single_axis_0','single_axis_1'):
     check(row['outcome'] in ('returned','refused'),'Unexpected probe outcome');c['probes'].append(dict(ref=sequences,kind=kind,outcome=row['outcome'],reason=row.get('reason')))
    else:raise ValueError('Unexpected node or evaluation kind')
   elif event=='cell_created':
    rec=cell(row['cell']);i=rec['id'];check(i not in cells and i==len(cells)+1 and row['panel']==0,'Duplicate or invalid cell creation')
    parent=row['parent'];check(parent is None or parent in selected,'Child of unselected or absent parent')
    if parent is not None:
     p=cells[parent]['created'];check(any(rec['bounds'] in halves(p,a) and rec['depths']==tuple(v+(j==a) for j,v in enumerate(p['depths'])) for a in (0,1)),'Invalid attempted half child')
    cells[i]=dict(parent=parent,created=rec,finished=None,joint=None,probes=[],registry=dict(panel=0,parent=parent,status='created',creation_ref=sequences))
    if row['selected']:
     check(i==1 and parent is None and rec['bounds']==(F(2),F(20),F(0),F(1)) and not selected,'Invalid initial complete rectangle');selected.add(i)
    else:pending.add(i)
   elif event=='cell_finished':
    check(row['panel']==0,'Other panel finish');rec=cell(row['cell']);i=rec['id'];check(i in cells and cells[i]['finished'] is None,'Repeated/unknown cell finish');c=cells[i];check(rec['bounds']==c['created']['bounds'] and rec['depths']==c['created']['depths'],'Finished geometry changed');check(c['joint'] is not None and rec['guard_reason']==c['joint']['reason'] and rec['partial_rule_status']=='refused','Finished outcome mismatch');c['finished']=rec;c['registry'].update(status='finished',completion_ref=sequences)
   elif event=='complete_cover_adoption':
    parent=row['parent'];children=row['children'];check(row['panel']==0,'Other panel adoption');adopt(selected,cells,parent,children);cells[parent]['registry'].update(status='replaced',adoption_ref=sequences)
    for i in children:check(i in pending,'Adopted child absent from pending');pending.remove(i);cells[i]['registry'].update(status='selected',adoption_ref=sequences)
   else:raise ValueError('Unknown journal event')
  with path.open('rb') as stream:
   for line in stream:
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     process_line(line);coherent_sequences=sequences;result.pop('current_event',None)
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    if sequences%5000==0:save()
  check(h.hexdigest()==PINS['p110_attempts.jsonl'] and identity(path)==before,'Changed journal bytes/identity');result['sources'].append(dict(path=str(path),sha256=h.hexdigest(),**before))
  check(sequences==45001 and attempts==20002 and events==Counter(evaluation=20002,cell_created=10000,cell_finished=10000,complete_cover_adoption=4999),'Journal totals mismatch')
  check(kinds==Counter(new_derivative_known_answer=2,joint_analytic_box=10000,single_axis_0=5000,single_axis_1=5000),'Operation decomposition mismatch')
  check(dict(refusals)==native['counts']['refusals_by_reason'] and sum(refusals.values())==native['counts']['refused_images'],'Refusal histogram mismatch')
  check(attempts==native['counts']['evaluations'] and len(cells)==native['counts']['created_cells'],'Native count mismatch');check(fixture_rows==native['new_derivative_known_answers'],'Saved fixture rows differ from journal')
  check(native['terminal_journal']['records']==sequences and native['terminal_journal']['size']==before['size'] and native['terminal_journal']['sha256']==h.hexdigest(),'Terminal journal declaration mismatch')
  check({str(i):c['registry'] for i,c in cells.items()}==native['attempted_children_registry'],'Native registry mismatch')
  active=native['active_panel'];check(active['index']==0 and active['cover_complete'] and active['leaf_count']==5000 and len(selected)==5000,'Active selected scope mismatch')
  saved={r['id']:r for r in active['leaves']};saved_pending={r['id']:r for r in native['unfinished_children']};check(len(saved)==len(active['leaves']) and set(saved)==selected and len(saved_pending)==len(native['unfinished_children']) and set(saved_pending)==pending=={10000} and 4805 in selected,'Terminal membership mismatch')
  result['phase']='terminal_rows';save()
  def terminal(i,raw):
   c=cells[i];actual=cell(raw);expected=dict(c['finished']);expected['probe_refs']=[r['ref'] for r in c['probes'] if r['outcome']=='returned'];check(actual==expected,'Terminal cell values/progress mismatch '+str(i))
   return dict(**actual,parent=c['parent'],creation_ref=c['registry']['creation_ref'],completion_ref=c['registry']['completion_ref'],joint_ref=c['joint']['ref'],joint_reason=c['joint']['reason'],probes=c['probes'])
  for i in sorted(selected):result['selected_rows'].append(terminal(i,saved[i]))
  for i in sorted(pending):result['pending_rows'].append(terminal(i,saved_pending[i]))
  rows=result['selected_rows'];real_lo,real_hi,imag_lo,imag_hi=four_sums(rows)
  check((real_lo,real_hi)==ends(active['real']) and (imag_lo,imag_hi)==ends(active['imaginary']),'Four endpoint sums mismatch')
  total,groups=width_classes(rows);check(total==real_hi-real_lo,'Width decomposition mismatch');check(set(groups)<=GUARDS|EVALUATIONS,'Unclassified selected refusal')
  result.update(counts=dict(events=events,attempts=attempts,kinds=kinds,refusals=refusals,selected=5000,pending=1,returned_probes=sum(r['outcome']=='returned' for c in cells.values() for r in c['probes'])),checks=dict(complete_midpoint_replay=True,terminal_values_and_progress=True,four_endpoint_sums=True,zero_nodes_and_returned_joint_images=True,fixture_journal_identity=True,registry_identity=True),endpoint_sums=dict(real=[real_lo,real_hi],imaginary=[imag_lo,imag_hi]),total_width=total,groups=groups,top32=sorted(rows,key=lambda r:(-r['width'],r['id']))[:32],panel_full_width_budget=BUDGET)
  def residual(names):
   value=total-sum((groups[g]['width'] for g in names if g in groups),F(0));return dict(removed_classes=sorted(names),residual=value,relation_to_budget='above' if value>BUDGET else 'equal' if value==BUDGET else 'below',scope='ideal zero widths only in listed current selected classes; fixed selected complement')
  result['class_removal_residuals']={g:residual({g}) for g in groups};result['union_removal_residuals']=dict(square_root_p_m_denominator_guards=residual(GUARDS),RF_complete_elliptic_complete_RD_refusals=residual(EVALUATIONS))
  for name,snapshot in snapshots.items():check(identity(INPUT/name)==snapshot,'Pinned input identity changed after replay')
  result.update(status='pass',phase='complete',cover_verified=True,width_decomposition_verified=True)
 except Exception as error:result.update(status='incomplete',error=type(error).__name__+': '+str(error),cover_verified=False,width_decomposition_verified=False)
 finally:
  signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.alarm(0);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN);resource.setrlimit(resource.RLIMIT_CPU,(60,60));save()
 print(json.dumps(dict(status=result['status'],phase=result['phase'],selected_rows=len(result['selected_rows']),error=result.get('error'))))
 if result['status']!='pass':raise SystemExit(1)

if __name__=='__main__':main()
