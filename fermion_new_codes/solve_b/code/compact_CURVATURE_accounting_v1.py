"""Lossless run encoding of long integer lists in saved output accounting."""
import argparse
import hashlib
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu
from run_CURVATURE_secondroute_v1 import write


def encode(value):
    if isinstance(value,list):
        if len(value)>4 and all(type(x) is int for x in value):
            runs=[]
            for x in value:
                if runs and x==runs[-1][1]+1:runs[-1][1]=x
                else:runs.append([x,x])
            rebuilt=[i for start,end in runs for i in range(start,end+1)]
            if rebuilt!=value:raise ValueError('integer list encoding changed values or order')
            return {'encoding':'integer_index_runs','count':len(value),'inclusive_runs':runs}
        return [encode(x) for x in value]
    if isinstance(value,dict):return {key:encode(item) for key,item in value.items()}
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    runtime=require_remote_cpu(args.input.parent)
    fixture=[0,1,2,4,4,7,8]
    expected={'encoding':'integer_index_runs','count':7,'inclusive_runs':[[0,2],[4,4],[4,4],[7,8]]}
    if encode(fixture)!=expected:raise RuntimeError('run encoding fixture failed')
    raw=args.input.read_bytes();answer=encode(json.loads(raw))
    answer['index_list_encoding']='Objects marked integer_index_runs replace long integer lists losslessly, inclusive endpoints, original order and duplicates retained; short lists and all numeric values otherwise unchanged.'
    answer['formatting_runtime']=runtime
    answer['full_accounting_source']=str(args.input);answer['full_accounting_sha256']=hashlib.sha256(raw).hexdigest()
    answer['formatter_control_passed']=True
    write(args.input.with_name('output_accounting_compact.json'),answer)
    print('saved lossless compact accounting',flush=True)


if __name__=='__main__':main()
