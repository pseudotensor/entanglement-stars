"""Metadata-only source binding; foreign implementation bytes are never decoded."""
import argparse
import hashlib
import json
from pathlib import Path


def hashes(path):
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            md5.update(block);sha.update(block)
    return dict(md5=md5.hexdigest(),sha256=sha.hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--bindings',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert hashlib.sha256(b'abc').hexdigest()=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    rows=[]
    for binding in json.loads(Path(a.bindings).read_text()):
        row=dict(binding);path=Path(binding['read_path'])
        try:
            row['hashes']=hashes(path);row['exists']=True
            if binding.get('printed_md5'):row['printed_md5_agrees']=row['hashes']['md5'].startswith(binding['printed_md5'])
        except OSError as e:row.update(exists=False,error=str(e))
        rows.append(row)
    result=dict(scope='Metadata-only exact source binding. No foreign implementation decoded, imported or executed; no physical payload validation inferred.',
                known_answers=dict(sha256_abc=True),rows=rows)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),unavailable=sum(not r['exists'] for r in rows),mismatches=sum(r.get('printed_md5_agrees') is False for r in rows))))


if __name__=='__main__':main()
