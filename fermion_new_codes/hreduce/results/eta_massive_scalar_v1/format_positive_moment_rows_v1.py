"""Presentation only: copy stored JSON fields/strings; no scientific arithmetic."""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource

ROOT=Path('<internal>/positive_moment_v1')
RAW=ROOT/'positive_moment_result.json'
PIN='15b643497774b2919107666a6dae21f8f004c7013ad35f0f6fce7562d36d98e0'

def main():
 if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Box directory only')
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 resource.setrlimit(resource.RLIMIT_AS,(2**30,2**30));resource.setrlimit(resource.RLIMIT_CPU,(15,15))
 names=['positive_moment_summary.json','positive_moment_parents.json','positive_moment_leaf_rows.tsv']
 if any((ROOT/n).exists() for n in names):raise FileExistsError('Preserve presentation')
 raw=RAW.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PIN:raise ValueError('Raw changed')
 data=json.loads(raw)
 summary={k:v for k,v in data.items() if k!='parents'}
 summary.update(raw_path=str(RAW),raw_sha256=PIN,formatter_scope='literal stored fields; no scientific arithmetic')
 (ROOT/names[0]).write_text(json.dumps(summary,indent=2)+'\n')
 parents=[]
 with (ROOT/names[2]).open('x',newline='') as stream:
  writer=csv.writer(stream,delimiter='\t');writer.writerow(['source_parent_id','source_child_id','leaf_record_json'])
  for p in data['parents']:
   record={k:v for k,v in p.items() if k!='children'}
   record['children']=[]
   for c in p['children']:
    record['children'].append({k:v for k,v in c.items() if k not in ('images','descendants','node_working_cover')})
    for leaf in c['descendants']:
     writer.writerow([p['source_parent_id'],c['id'],json.dumps(leaf,separators=(',',':'))])
   parents.append(record)
 (ROOT/names[1]).write_text(json.dumps(parents,indent=2)+'\n')
 for n in names:
  p=ROOT/n;print(json.dumps(dict(path=str(p),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))

if __name__=='__main__':main()
