"""Metadata-only reconstruction of loaded thread pools; no historical activity claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy
import scipy.linalg
import scipy.sparse
import scipy.special
from threadpoolctl import threadpool_info
from cpu_runtime import require_remote_cpu
from packet_core_v1 import file_hash

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
runtime=require_remote_cpu(args.output)
historical={}
for channel in (1,5):
    artifact=json.loads((args.output/f'k{channel}'/'producer_complete.json').read_text())
    historical[str(channel)]=artifact['runtime']
record={'metadata_runtime':runtime,'loaded_library_pools':threadpool_info(),
    'environment':{key:os.environ[key] for key in ('QI2_REMOTE_CPU','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
    'historical_producer_startup_runtime':historical,
    'historical_aggregate_active_thread_measurement':None,
    'scope':'metadata-only new process with same imports and environment; not a measurement of completed producer activity',
    'correction':'environment cap eight per backend is not an observation of eight resident process threads; both completed producer runtimes record fifteen',
    'future_allocation':'use one environment thread for any subsequent numerical job unless aggregate resident and active pool accounting is established before launch',
    'documentation':'https://github.com/joblib/threadpoolctl/blob/master/README.md',
    'metadata_code_sha256':file_hash(__file__),'command':sys.argv}
(args.output/'thread_pool_metadata.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'resident_threads':runtime['threads'],'library_pool_caps':[x['num_threads'] for x in record['loaded_library_pools']]}))
