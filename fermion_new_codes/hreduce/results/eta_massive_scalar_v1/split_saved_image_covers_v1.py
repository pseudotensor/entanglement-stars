"""Split already formatted saved-cover fields for the repository size ceiling."""
import hashlib
import json
from pathlib import Path
ROOT=Path('<internal>/image_cover_v1')
def main():
 source=ROOT/'image_cover_covers.json'
 if hashlib.sha256(source.read_bytes()).hexdigest()!='356a487936aca7aa6ccad324432241e1b0d8cb64d5dc5fc926d5294354ce0a7c':raise ValueError('Changed literal cover export')
 data=json.loads(source.read_text())
 products={'image_cover_old_cells.json':data['old_passed_cells'],'image_cover_parents.json':data['parents'],'image_cover_sources_0.json':data['sources'][:886],'image_cover_sources_1.json':data['sources'][886:]}
 for name,value in products.items():
  p=ROOT/name
  with p.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
  print(json.dumps(dict(path=str(p),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
if __name__=='__main__':main()
