#!/usr/bin/env python3
"""Lossless literal factoring of a completed written report; no science imports.

Every CHECK retains its full name, pass bit and detail. Repeated JSON subtrees
are defined once, and all other rows remain complete JSON. Exact original
byte reconstruction must succeed before the exclusive output is created.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json


def encoded(obj):
    return json.dumps(obj,separators=(',',':'),allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('output');args=parser.parse_args()
    source=Path(args.input);output=Path(args.output)
    if output.exists():raise FileExistsError('never replace a prior view')
    raw=source.read_bytes();lines=raw.decode().splitlines();rows=[json.loads(line) for line in lines]
    if raw!=(''.join(encoded(row)+'\n' for row in rows)).encode():
        raise ValueError('input is not the complete canonical newline written format')
    if not rows or rows[-1]['kind']!='termination':raise ValueError('unfinished input')
    counts=Counter();objects={}
    def inventory(obj):
        if isinstance(obj,dict):
            if '@literal_ref' in obj:raise ValueError('reserved reference key in original')
            for child in obj.values():inventory(child)
        elif isinstance(obj,list):
            for child in obj:inventory(child)
        else:return
        key=encoded(obj)
        if len(key)>=48:counts[key]+=1;objects[key]=obj
    for row in rows:inventory(row)
    keys=sorted((k for k,n in counts.items() if n>=2 and (n-1)*(len(k)-24)>len(k)+32),key=lambda k:(len(k),k))
    identifiers={k:i for i,k in enumerate(keys)}
    def pack(obj,skip=None):
        if not isinstance(obj,(dict,list)):return obj
        key=encoded(obj)
        if key!=skip and key in identifiers:return {'@literal_ref':identifiers[key]}
        if isinstance(obj,dict):return {k:pack(v) for k,v in obj.items()}
        return [pack(v) for v in obj]
    definitions=[pack(objects[k],k) for k in keys]
    header={'original_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw),'original_rows':len(rows),
        'definitions':len(definitions),'rule':'D defines literal JSON; C is [check name,pass,detail]; R is a complete row; references are literal subtree reuse only'}
    view=['# Lossless written controls view\n','H '+encoded(header)+'\n']
    for i,obj in enumerate(definitions):view.append('D '+encoded([i,obj])+'\n')
    for row in rows:
        if list(row)==['kind','data'] and row['kind']=='check' and list(row['data'])==['name','pass','detail']:
            data=row['data'];view.append('C '+encoded([data['name'],data['pass'],pack(data['detail'])])+'\n')
        else:view.append('R '+encoded(pack(row))+'\n')
    result=''.join(view).encode()
    # Parse the actual candidate bytes, rather than merely trusting pack().
    restored=[];parsed_definitions={}
    def expand(obj):
        if isinstance(obj,dict):
            if list(obj)==['@literal_ref']:return expand(parsed_definitions[obj['@literal_ref']])
            return {k:expand(v) for k,v in obj.items()}
        if isinstance(obj,list):return [expand(v) for v in obj]
        return obj
    for line in result.decode().splitlines()[2:]:
        tag,payload=line.split(' ',1);obj=json.loads(payload)
        if tag=='D':parsed_definitions[obj[0]]=obj[1]
        elif tag=='R':restored.append(expand(obj))
        elif tag=='C':restored.append({'kind':'check','data':dict(zip(('name','pass','detail'),expand(obj)))})
        else:raise ValueError('unexpected view record')
    recovered=(''.join(encoded(row)+'\n' for row in restored)).encode()
    if recovered!=raw:raise ValueError('literal byte reconstruction failed')
    with output.open('xb') as stream:stream.write(result)
    print(encoded({'status':'literal_byte_reconstruction_PASS','input_sha256':header['original_sha256'],
        'output_sha256':hashlib.sha256(result).hexdigest(),'input_bytes':len(raw),'output_bytes':len(result),
        'rows':len(rows),'checks':sum(row['kind']=='check' for row in rows),'definitions':len(definitions)}))

if __name__=='__main__':main()
