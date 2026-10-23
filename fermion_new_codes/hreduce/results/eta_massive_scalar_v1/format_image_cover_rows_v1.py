"""Copy saved image-cover fields into bounded review files; no science or arithmetic."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path('<internal>/image_cover_v1')

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 execution=json.loads((ROOT/'image_cover_execution.json').read_text())
 if execution['status']!='finished':raise ValueError('Terminal execution required')
 raw=ROOT/'image_cover_result.json';identity=execution[raw.name]
 if raw.stat().st_size!=identity['size'] or digest(raw)!=identity['sha256']:raise ValueError('Changed saved result')
 report=json.loads(raw.read_text());groups=('leaves','old_passed_cells','sources','parents')
 summary={k:v for k,v in report.items() if k not in groups}
 summary['raw_reference']=dict(path=str(raw),**identity)
 leaves=copy.deepcopy(report['leaves'])
 for leaf in leaves:
  for kind,im in leaf['images'].items():
   for tile in im.get('tiles',[]):
    if 'witness' in tile:
     del tile['witness']
     tile['witness_reference']=dict(raw=str(raw),leaf_id=leaf['id'],kind=kind,index=tile['index'])
 products={'image_cover_summary.json':summary,'image_cover_leaves.json':leaves,'image_cover_covers.json':{k:report[k] for k in groups if k!='leaves'}}
 for name,body in products.items():
  p=ROOT/name
  with p.open('x') as stream:json.dump(body,stream,indent=2,allow_nan=False);stream.write('\n')
  print(json.dumps(dict(path=str(p),size=p.stat().st_size,sha256=digest(p))))

if __name__=='__main__':main()
