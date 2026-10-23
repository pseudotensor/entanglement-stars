"""Extract structural metadata only; never print embedded readout values."""
import argparse,json
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); runtime=require_remote_cpu(args.output)
    import torch
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    allowed=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),
             (np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
             np.dtype,np.ndarray]
    allowed.extend(type(np.dtype(name)) for name in ('float64','float32','complex128','complex64','int64','int32','uint8','bool'))
    with torch.serialization.safe_globals(allowed):
        data=torch.load(args.input,map_location='cpu',weights_only=True,mmap=True)
    def structure(value):
        if value is None:return None
        if hasattr(value,'shape') and hasattr(value,'dtype'):
            return {'shape':list(value.shape),'dtype':str(value.dtype)}
        return {'type':type(value).__name__}
    result={'source':str(args.input),'runtime':runtime,'top_keys':list(data.keys())}
    for key in ('k','kappa','T','t','t_over_rs','T_over_rs','rs','rs_nominal','nsites','n2','n_occ','n_ck','last_ck','sector_on','sector_prop','m','g','tau','sigma','layout','what','history_label','record_json','fields_npz'):
        value=data.get(key)
        if isinstance(value,(int,float,bool,str)):result[key]=value
    if isinstance(data.get('rows'),list) and data['rows']:
        result['last_row_clock']={key:data['rows'][-1].get(key) for key in ('t','t_over_rs')}
    for key in ('Phi','R','N_applied','v_applied'):
        if key in data:result[key]=structure(data[key])
    result['channels']={}
    for label,channel in data.get('channels',{}).items():
        if not isinstance(channel,dict):continue
        entry={'keys':list(channel.keys())}
        for key in ('Phi','F','F_t','F_g'): 
            if key in channel:entry[key]=structure(channel[key])
        for key in ('sector_on','sector_prop','sector_report'):
            if key in channel:entry[key]=structure(channel[key]) if isinstance(channel[key],dict) else channel[key]
        result['channels'][str(label)]=entry
    vac=data.get('vac_meta')
    if isinstance(vac,dict):
        result['vac_meta']={str(key):({'keys':list(val)} if isinstance(val,dict) else structure(val)) for key,val in vac.items()}
    for label in ('v6c','v6g','w1'):
        meta=data.get(label)
        if not isinstance(meta,dict):continue
        entry={'keys':list(meta)}
        if 'sector_on' in meta:entry['sector_on']=meta['sector_on']
        if isinstance(meta.get('vac_meta'),dict):
            entry['vac_meta']={str(k):({'keys':list(v),'e_plus':structure(v.get('e_plus')),
                'V_plus':structure(v.get('V_plus'))} if isinstance(v,dict) else structure(v)) for k,v in meta['vac_meta'].items()}
        result[label]=entry
    target=args.output/(args.input.stem+'_structure.json')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
