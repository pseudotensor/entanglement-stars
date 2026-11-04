"""Literal JSON display of saved arrays, with no numerical reevaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from iso128_rebuild_core_v1 import require_source_cpu,file_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runtime=require_source_cpu(a.output.parent)
    if a.output.exists():raise ValueError('Refusing to replace a saved literal view')
    result={'scope':'literal saved-array display only; no numerical reevaluation','runtime':runtime,
            'input':file_identity(a.input),'implementation':file_identity(__file__),'arrays':{}}
    with np.load(a.input,allow_pickle=False) as archive:
        for key in archive.files:
            value=archive[key]
            if value.dtype.hasobject or value.dtype.kind=='c':raise ValueError('Only plain real saved arrays in this view')
            result['arrays'][key]={'dtype':str(value.dtype),'shape':list(value.shape),'values':value.tolist()}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(file_identity(a.output)))


if __name__=='__main__':main()
