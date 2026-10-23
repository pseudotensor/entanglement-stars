"""Literal requested stored tile witness exports; no arithmetic or science."""
import hashlib
import json
from pathlib import Path
ROOT=Path('<internal>/image_cover_v1')
def main():
 raw=ROOT/'image_cover_result.json'
 if hashlib.sha256(raw.read_bytes()).hexdigest()!='1773c9756663208c84b4fb5bc59ffed2a05c1fdfaf17a63e35f2552a1b7bd9a4':raise ValueError('Changed terminal output')
 data=json.loads(raw.read_text());leaves={x['id']:x for x in data['leaves']}
 for identity,kind,index in [(10574,'z',0),(10638,'t',0)]:
  leaf=leaves[identity];image=leaf['images'][kind];tile=image['tiles'][index]
  body=dict(raw=str(raw),leaf_id=identity,source_child_id=leaf['source_child_id'],kind=kind,leaf_bounds=leaf['bounds_z_t'],image_status=image['status'],original_domain=image['original_domain'],tile=tile)
  target=ROOT/('image_cover_witness_%s_%s_%s.json'%(identity,kind,index))
  with target.open('x') as stream:json.dump(body,stream,indent=2,allow_nan=False);stream.write('\n')
  print(target.name,target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest())
if __name__=='__main__':main()
