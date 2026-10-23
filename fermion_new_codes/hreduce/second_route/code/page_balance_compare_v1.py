"""Saved-row Page balance agreement audit; no solver, spectrum or new physical readout."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

ROOT=Path('<internal>/page_balance_compare_v1')
LEDGER=Path('<internal>/ledger.json')
BALANCE=LEDGER.parent/'balance_readout/balance_readout.json'
PINS={LEDGER:'e966a71813ace1d69395b43b284724b831fd6918c008f1a41695a73d1033357a',BALANCE:'abb2a8697be7e8a24c87cdbee6df00bc0de4a897ba31104a4be11f9a39e91bd6'}
FIRST_MD5={'<internal>/v6o_B2_k1_d13.json': '50fe59a03edc93837980127cbbcec51b', '<internal>/v6o_B2_k2_d13.json': 'd19ccb67b023255cf99afbac22d0eb4a', '<internal>/v6o_B2_k3_d13.json': '78f26e42db7d7b1e6443c25e66a4ba90', '<internal>/v6o_B2_k4_d13.json': '2909f8466f226c711f937dad59e03dd8', '<internal>/v6o_B2_k5_d13.json': '25ade5a6b7c217f5b8868fac7f00ccde', '<internal>/v6o_B2_k6_d13.json': '86bb53519c02ac98b479332a5a48f275', '<internal>/v6o_B2_k7to9_k7_d13.json': 'a74a4341dcba6995b27097d58dbcc055', '<internal>/v6o_B2_k7to9_k8_d13.json': 'da624eeabffa35698e4a6da2cde7efd4', '<internal>/v6o_B2_k7to9_k9_d13.json': 'b4e9825e91f995e10b87f6494cfd7acd', '<internal>/v6o_B2_k10to12_k10_d13.json': '8b67b3072305545bdede1752466e5bd0', '<internal>/v6o_B2_k10to12_k11_d13.json': '27e5b89d1d07de8ddd55c4ec19f5ec5f', '<internal>/v6o_B2_k10to12_k12_d13.json': 'c961ce48a701ec467dbdefbed6bf79a4', '<internal>/v6o_B2_k13to16_k13_d13.json': '4593725f9c75dd3635333782df5450bd', '<internal>/v6o_B2_k13to16_k14_d13.json': 'cb34915964f65e2392b8f4b43c4d3f33', '<internal>/v6o_B2_k13to16_k15_d13.json': 'b393dcf077afe320eaa04775cf215a45', '<internal>/v6o_B2_k13to16_k16_d13.json': 'f520414e587eea81a4f94079c6cf4be5', '<internal>/v6o_B2_k17to18_k17_d13.json': '03bfaba1b42a2afb10207cf68e3fc958', '<internal>/v6o_B2_k17to18_k18_d13.json': '37951baa8c72c66e6612759b007a5400'}
SETS=((1,),(1,2),(1,2,3),tuple(range(1,19)))
PRIMARY=((26,40),(36,50),(46,60),(56,70),(66,80),(76,90),(86,100))
UNIT=F(64,2**53)
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
FIELDS=('D','C','F','W','P','B','t','r','c')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(x):
 if isinstance(x,bool) or x is None or isinstance(x,Decimal) and not x.is_finite():raise ValueError('Nonfinite or invalid required number')
 return F(x)
def integer(x):
 x=number(x)
 if x.denominator!=1:raise ValueError('Noninteger identity')
 return x.numerator
def pack(x):
 if isinstance(x,F):
  with localcontext() as ctx:
   ctx.prec=50;d=str(Decimal(x.numerator)/Decimal(x.denominator))
  return dict(numerator=str(x.numerator),denominator=str(x.denominator),decimal_diagnostic=d)
 if isinstance(x,Decimal):return str(x)
 if isinstance(x,dict):return {str(k):pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x

def key(row,channel=None,producer=False,aggregate=False):
 start,end=(row['t0_over_rs'],row['t1_over_rs']) if producer else (row['clock_start_rs'],row['clock_end_rs'])
 cut=integer(row['cut'] if producer else row['cut_rs']);a,b=number(start),number(end)
 if cut not in (16,30) or b<=a:raise ValueError('Invalid cut/window identity')
 head=tuple(integer(k) for k in row['channels']) if aggregate else integer(channel if producer else row['kappa'])
 return head,cut,a,b

def insert(rows,k,value):
 if k in rows:raise ValueError('Duplicate complete key '+str(k))
 rows[k]=value

def require_channels(rows,channels,window):
 keys=[(k,*window) for k in channels]
 if any(k not in rows for k in keys):raise ValueError('Missing aggregate channel')
 return keys

def derived(D,C,current,jumps,vacuum):
 B=current+jumps-vacuum;t=C-current-jumps
 return dict(D=D,C=C,F=current,W=jumps,P=vacuum,B=B,t=t,r=D-B,c=D-B-t,c_independent=D-C+vacuum)

def first_values(row):
 return derived(number(row['inventory_D_end'])-number(row['inventory_D_start']),number(row['change_C']),number(row['crossing_current']),number(row['crossing_jumps']),number(row['vacuum_change']))
def new_values(row):return {k:number(row[n]) for k,n in dict(D='endpoint_change_D',C='delta_C',F='current',W='jumps',P='vacuum_change',B='balance_second_term_D',t='epsilon_tel',r='endpoint_minus_balance',c='endpoint_inventory_closure').items()}
def scale(a,b):return sum((abs(v[k]) for v in (a,b) for k in ('D','C','F','W','P')),F(0))


def weighted_values(channels, rows):
 return {f:sum((16*k*rows[k][f] for k in channels),F(0)) for f in FIELDS}

def weighted_absolute(channels, rows, field):
 return sum((16*k*abs(rows[k][field]) for k in channels),F(0))


def main():
 if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory only')
 names=('comparison_result.json','comparison_result.json.tmp','row_comparisons.jsonl','aggregate_comparisons.jsonl')
 if any((ROOT/n).exists() for n in names):raise FileExistsError('Existing allocation requires assessment, never retry')
 cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[0]])
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 resource.setrlimit(resource.RLIMIT_AS,(2**30,2**30));resource.setrlimit(resource.RLIMIT_CPU,(25,30))
 def deadline(s,f):raise TimeoutError('Fixed saved-row comparison allocation')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(55)
 start=time.monotonic();report=dict(status='incomplete',purpose='Inbox86 all-row first-ledger versus icloud balance comparison',scope='saved scalar arithmetic only; no covariance, spectrum, beta or independent primitives',sources=[],rows_completed=0,aggregates_completed=0,failures=[],missing_keys=[],duplicate_keys=[],nonfinite_or_missing=[],maxima={},maximum_normalized={},primary_rows=[])
 def save():
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  try:
   report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=digest(Path(__file__)))
   tmp=ROOT/'comparison_result.json.tmp';tmp.write_text(json.dumps(pack(report),indent=2,allow_nan=False)+'\n');tmp.replace(ROOT/'comparison_result.json')
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 def load(p):
  before=p.stat();raw=p.read_bytes();after=p.stat();h=hashlib.sha256(raw).hexdigest()
  if (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)!=(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev):raise ValueError('Input changed during read')
  if p in PINS and h!=PINS[p]:raise ValueError('Changed pinned saved route')
  md5=hashlib.md5(raw).hexdigest()
  if str(p) in FIRST_MD5 and md5!=FIRST_MD5[str(p)]:raise ValueError('Changed first producer manifest identity')
  report['sources'].append(dict(path=str(p),size=after.st_size,sha256=h,md5=md5,first_manifest_match=md5==FIRST_MD5.get(str(p)) if str(p) in FIRST_MD5 else None,mtime_ns=after.st_mtime_ns,inode=after.st_ino,device=after.st_dev))
  return json.loads(raw,parse_float=Decimal,parse_constant=Decimal)
 def compare(name,a,b,tolerance,identity,rows):
  delta=a-b;passed=abs(delta)<=tolerance
  row=dict(name=name,first=a,second=b,difference=delta,threshold=tolerance,normalized_absolute_difference=abs(delta)/tolerance,pass_comparison=passed)
  rows.append(row)
  old=report['maxima'].get(name)
  if old is None or abs(delta)>abs(old['difference']):report['maxima'][name]=dict(key=identity,**row)
  normalized=report['maximum_normalized'].get(name)
  if normalized is None or row['normalized_absolute_difference']>normalized['normalized_absolute_difference']:report['maximum_normalized'][name]=dict(key=identity,**row)
  if not passed:report['failures'].append(dict(key=identity,**row))
 def journal(stream,value):stream.write(json.dumps(pack(value),allow_nan=False)+'\n');stream.flush()
 def fixtures():
  positive=derived(F(7,8),F(1),F(1,2),F(1,4),F(1,8));bad=derived(F(1),F(1),F(1,2),F(1,4),F(1,8))
  if positive['B']!=F(5,8) or positive['t']!=F(1,4) or bad['c']!=F(1,8) or bad['c_independent']!=F(1,8):raise AssertionError('Known scalar answers')
  if abs(positive['D']-positive['B'])<=UNIT*max(F(1),scale(positive,positive)):raise AssertionError('Endpoint/balance alias trap')
  # Frozen cancellation control uses half of the negative-telescope fixture.
  negative=derived(F(3,16),F(1,4),F(1,4),F(1,8),F(1,16))
  fixture_rows={1:positive,2:negative};group=weighted_values((1,2),fixture_rows)
  expected=dict(F=F(16),W=F(8),P=F(4),C=F(24),D=F(20),B=F(20),t=F(0),r=F(0),c=F(0))
  if group!=expected or weighted_absolute((1,2),fixture_rows,'t')!=8:raise AssertionError('Actual weighted aggregation fixture')
  a=dict(kappa=1,cut_rs=16,clock_start_rs=2,clock_end_rs=15);b=dict(a,clock_start_rs=1)
  if key(a)==key(b) or key(a)[2:]!=(F(2),F(15)):raise AssertionError('Full key/window fixture')
  checks=[]
  for operation in (lambda:insert({key(a):a},key(a),a),lambda:require_channels({},(1,2),(16,F(2),F(15))),lambda:number(Decimal('NaN')),lambda:require_channels({key(a):a},(1,),(16,F(1),F(15)))):
   try:operation()
   except ValueError:checks.append(True)
   else:raise AssertionError('Required rejection fixture')
  report['known_answers']=dict(inconsistent_inventory_closure=bad['c'],signed_cancel=F(0),weighted_absolute_telescope=F(8),endpoint_balance_alias_rejected=True,weighted_fixture=group,weighted_fixture_jump=group['W']-group['P'],duplicate_rejected=checks[0],missing_channel_rejected=checks[1],nonfinite_rejected=checks[2],same_end_different_start_distinguished=checks[3],thirteen_unit_window_preserved=True)
 try:
  fixtures();ledger=load(LEDGER);balance=load(BALANCE);first={};old={};new={};aggregates={}
  if set(ledger['sources'])!=set(FIRST_MD5) or len(ledger['sources'])!=18:raise ValueError('Expected eighteen literal producer paths')
  for path in ledger['sources']:
   p=Path(path)
   if not str(p).startswith('<internal>/runs') or not p.name.endswith('_d13.json'):raise ValueError('Unexpected producer source')
   data=load(p)
   for channel,rows in data['windows'].items():
    k=integer(channel)
    if k not in range(1,19):raise ValueError('Unknown producer channel')
    for row in rows:
     if str(row.get('cut')) not in ('16','30'):continue
     ident=key(row,k,producer=True)
     try:insert(first,ident,dict(row=row,path=str(p)))
     except ValueError:report['duplicate_keys'].append(ident);raise
  for label,data,destination in [('old',ledger['rows'],old),('new',balance['rows'],new)]:
   for row in data:
    ident=key(row)
    try:insert(destination,ident,row)
    except ValueError:report['duplicate_keys'].append(dict(route=label,key=ident));raise
  for row in balance['aggregates']:
   ident=key(row,aggregate=True)
   if row['status']!='formed':report['missing_keys'].append(dict(aggregate=ident,status=row['status']));continue
   try:insert(aggregates,ident,row)
   except ValueError:report['duplicate_keys'].append(ident);raise
  for label,rows in [('first',first),('old',old),('new',new)]:
   if set(rows)!=set(new):report['missing_keys'].append(dict(route=label,missing=list(set(new)-set(rows)),extra=list(set(rows)-set(new))))
  if report['missing_keys']:raise ValueError('Different complete row key domains')
  windows={(a,b) for k,c,a,b in new}
  if len(first)!=1620 or len(new)!=1620 or len(windows)!=45:raise ValueError('Wrong row/window cardinality')
  if not {(F(2),F(15)),(F(14),F(28)),(F(15),F(28))}<=windows or (F(2),F(16)) in windows:raise ValueError('Actual early-window domain lost')
  expected={(k,c,a,b) for k in range(1,19) for c in (16,30) for a,b in windows}
  if set(new)!=expected:raise ValueError('Channel/cut/window coverage')
  expected_agg={(channels,c,a,b) for channels in SETS for c in (16,30) for a,b in windows}
  if set(aggregates)!=expected_agg:report['missing_keys'].append(dict(aggregate_missing=list(expected_agg-set(aggregates)),aggregate_extra=list(set(aggregates)-expected_agg)));raise ValueError('Aggregate coverage')
  report['coverage']=dict(first_rows=len(first),old_rows=len(old),new_rows=len(new),aggregate_rows=len(aggregates),channels=list(range(1,19)),cuts=[16,30],windows=sorted(windows),sets=SETS)
  save();cache={}
  with (ROOT/'row_comparisons.jsonl').open('x') as out:
   for ident,row in sorted(new.items()):
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     source=first[ident];raw=source['row'];before=first_values(raw);after=new_values(row);s=scale(before,after);tol=UNIT*max(F(1),s);checks=[]
     endpoint_size=sum((abs(number(route['inventory_'+f+'_'+end])) for route in (raw,row) for f in ('D',) for end in ('start','end')),F(0));endpoint_scale=s+endpoint_size;endpoint_tol=UNIT*max(F(1),endpoint_scale)
     if (integer(row['weight'])!=16*ident[0] or integer(old[ident]['weight'])!=16*ident[0]) or row['source_input']!=source['path'] or old[ident]['input']!=source['path']:raise ValueError('Source/weight association')
     for field in FIELDS:compare('channel_'+field,before[field],after[field],endpoint_tol if field in ('D','r','c') else tol,ident,checks)
     compare('producer_saved_B',number(raw['second_term_D']),after['B'],tol,ident,checks)
     compare('producer_saved_t',number(raw['telescoping_residual']),after['t'],tol,ident,checks)
     compare('legacy_endpoint_label',number(old[ident]['second_term_D']),after['D'],tol,ident,checks)
     own=derived(after['D'],after['C'],after['F'],after['W'],after['P'])
     for field in ('B','t','r','c'):compare('new_definition_'+field,own[field],after[field],tol,ident,checks)
     compare('independent_closure_identity',own['c_independent'],after['c'],tol,ident,checks)
     for field,delta_field in [('C','C'),('D','D')]:
      endpoints=[number(raw['inventory_'+field+'_'+end]) for end in ('start','end')]+[number(row['inventory_'+field+'_'+end]) for end in ('start','end')]
      subtraction_tol=UNIT*max(F(1),s+sum(map(abs,endpoints)))
      for end in ('start','end'):
       saved=number(raw['inventory_'+field+'_'+end])
       for route_name,route in (('new',row),('legacy',old[ident])):
        compare(route_name+'_metadata_'+field+'_'+end,saved,number(route['inventory_'+field+'_'+end]),UNIT*max(F(1),abs(saved)+abs(number(route['inventory_'+field+'_'+end]))),ident,checks)
      compare('endpoint_subtraction_'+field,endpoints[1]-endpoints[0],after[delta_field],subtraction_tol,ident,checks)
     cache[ident]=dict(values=before,scale=s,endpoint_scale=endpoint_scale,saved_B=number(raw['second_term_D']),saved_t=number(raw['telescoping_residual']))
     journal(out,dict(key=ident,producer_source=source['path'],producer=raw,new=row,legacy=old[ident],comparisons=checks));report['rows_completed']+=1
    except (KeyError,ValueError,TypeError) as error:report['nonfinite_or_missing'].append(dict(key=ident,error=str(error)));raise
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    if report['rows_completed']%100==0:save()
  with (ROOT/'aggregate_comparisons.jsonl').open('x') as out:
   for ident,row in sorted(aggregates.items()):
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     channels,cut,a,b=ident;keys=require_channels(cache,channels,(cut,a,b));group_rows={k:cache[(k,cut,a,b)]['values'] for k in channels};values=weighted_values(channels,group_rows);tol=UNIT*max(F(1),sum((16*k*cache[(k,cut,a,b)]['scale'] for k in channels),F(0)));endpoint_tol=UNIT*max(F(1),sum((16*k*cache[(k,cut,a,b)]['endpoint_scale'] for k in channels),F(0)));after=new_values(row);checks=[]
     for field in ('B','t'):
      saved=sum((16*k*cache[(k,cut,a,b)]['saved_'+field] for k in channels),F(0))
      compare('aggregate_producer_saved_'+field,saved,after[field],tol,ident,checks)
     for field in FIELDS:compare('aggregate_'+field,values[field],after[field],endpoint_tol if field in ('D','r','c') else tol,ident,checks)
     for field,name in [('t','epsilon_tel'),('r','endpoint_minus_balance'),('c','endpoint_inventory_closure')]:
      absolute=weighted_absolute(channels,group_rows,field)
      compare('aggregate_absolute_'+field,absolute,number(row['weighted_absolute_diagnostics'][name]),endpoint_tol if field in ('r','c') else tol,ident,checks)
     output=dict(key=ident,independent_weighted_first=values,new=row,comparisons=checks);journal(out,output)
     if (a,b) in PRIMARY:report['primary_rows'].append(output)
     report['aggregates_completed']+=1
    except (KeyError,ValueError,TypeError) as error:report['nonfinite_or_missing'].append(dict(aggregate=ident,error=str(error)));raise
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    if report['aggregates_completed']%50==0:save()
  if len(report['primary_rows'])!=56:raise ValueError('Expected fifty-six primary rows')
  report['status']='pass' if not report['failures'] else 'failed_comparison'
 except Exception as error:report.update(status='incomplete',error=type(error).__name__+': '+str(error))
 finally:
  signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.alarm(0);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN);resource.setrlimit(resource.RLIMIT_CPU,(30,30))
  for name in ('row_comparisons.jsonl','aggregate_comparisons.jsonl'):
   p=ROOT/name
   if p.exists():report[name]=dict(path=str(p),size=p.stat().st_size,sha256=digest(p))
  save()
 print(json.dumps(dict(status=report['status'],rows=report['rows_completed'],aggregates=report['aggregates_completed'],failures=len(report['failures']))))
 if report['status']!='pass':raise SystemExit(1)

if __name__=='__main__':main()
