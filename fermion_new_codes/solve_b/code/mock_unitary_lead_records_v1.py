"""Box-only incremental artifacts and written reports for the #138 jobs."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time

class JobStop(Exception):pass

class Bundle:
    def __init__(self,path,sources,limit):
        from cpu_runtime import require_remote_cpu
        path=Path(path)
        if path.exists():raise FileExistsError('Never overwrite any preceding run')
        self.identities={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
        self.path=path;self.runtime=require_remote_cpu(path);self.counter=0;self.checks=[];self.timing=[]
        self.serialization_wall=0.;self.serialization_cpu=0.
        self.started=time.perf_counter();self.cpu=time.process_time();self.closed=False
        names=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        if any(os.environ.get(k)!='1' for k in names):raise RuntimeError('one-thread settings required')
        resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
        self.journal=(path/'records.ndjson').open('x');self.report=(path/'written_report.md').open('x')
        import numpy as np
        import scipy
        import mpmath
        import ctypes
        import contextlib
        import io
        if len(list(Path('/proc/self/task').iterdir()))!=1:raise RuntimeError('one resident thread required')
        if ctypes.CDLL(None).fegetround()!=0:raise RuntimeError('FE_TONEAREST required')
        if np.float64(np.finfo(float).tiny)*np.float64(.5)==0:raise RuntimeError('gradual underflow required')
        config=io.StringIO()
        with contextlib.redirect_stdout(config):np.show_config()
        self.save('intake',{'runtime':self.runtime,'sources':self.identities,'command':sys.argv,'thread_environment':{k:os.environ[k] for k in names},
            'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mpmath.__version__,
            'rounding':'FE_TONEAREST','gradual_underflow':True,'backend':config.getvalue(),'handled_limit':limit})
        def stop(sig,frame):raise JobStop('handled job safety ceiling')
        signal.signal(signal.SIGALRM,stop);signal.alarm(limit)

    def convert(self,x):
        import numpy as np
        import mock_unitary_lead_v1 as q
        import run_mock_vertex_admission_v1 as exact
        if isinstance(x,np.ndarray):
            name='array_'+str(self.counter)+'.npz';self.counter+=1
            with (self.path/name).open('xb') as f:np.savez_compressed(f,data=x)
            return {'array':name,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256((self.path/name).read_bytes()).hexdigest()}
        if isinstance(x,q.Snapshot):return {'metadata':self.convert(x.metadata()),'raw_vector':self.convert(x.vector)}
        if isinstance(x,exact.C):return {'re':str(x.r),'im':str(x.i)}
        if isinstance(x,Fraction):return str(x)
        if isinstance(x,np.generic):return self.convert(x.item())
        if isinstance(x,complex):return {'re':self.convert(x.real),'im':self.convert(x.imag)}
        if isinstance(x,dict):return {str(k):self.convert(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [self.convert(v) for v in x]
        if type(x) in (str,int,bool) or x is None:return x
        if type(x) is float:
            if math.isfinite(x):return x
            return {'nonfinite':str(x)}
        raise TypeError('Unsupported artifact type '+str(type(x)))

    def save(self,kind,obj):
        wall=time.perf_counter();cpu=time.process_time()
        record={'kind':str(kind),'data':self.convert(obj)}
        encoded=json.dumps(record,separators=(',',':'),allow_nan=False)
        self.journal.write(encoded+'\n');self.journal.flush()
        def written(value,key=''):
            if isinstance(value,dict):
                if key in ('_exact','exact_density','exact_cross_block','intended','actual_literal','independent_intended'):
                    raw=json.dumps(value,separators=(',',':'),sort_keys=True).encode()
                    return {'raw_record_field':key,'entries':len(value),'sha256':hashlib.sha256(raw).hexdigest(),'storage':'complete entries in same on-box records.ndjson row'}
                return {k:written(v,k) for k,v in value.items()}
            if isinstance(value,list):return [written(v) for v in value]
            return value
        self.report.write(json.dumps(written(record),separators=(',',':'),allow_nan=False)+'\n');self.report.flush()
        self.serialization_wall+=time.perf_counter()-wall;self.serialization_cpu+=time.process_time()-cpu
        return record

    def check(self,name,condition,detail):
        row={'name':str(name),'pass':bool(condition),'detail':detail};self.checks.append(bool(condition));self.save('check',row)
        if not condition:raise AssertionError(str(name))

    def reject(self,name,fn,reason=None):
        try:fn()
        except (ValueError,TypeError,PermissionError,FileNotFoundError) as exc:
            if reason is not None and reason not in str(exc):raise AssertionError('wrong rejection reason for '+name+': '+str(exc))
            self.check(name,True,{'exception':type(exc).__name__,'reason':str(exc)});return
        self.check(name,False,{'reason':'required rejection did not occur'})

    def phase(self,name,fn):
        start=time.perf_counter();cpu=time.process_time();sw=self.serialization_wall;sc=self.serialization_cpu;status='incomplete'
        try:
            result=fn();status='complete';return result
        finally:
            row={'phase':str(name),'status':status,'wall':time.perf_counter()-start,'cpu':time.process_time()-cpu,
                'serialization_wall':self.serialization_wall-sw,'serialization_cpu':self.serialization_cpu-sc,
                'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            self.timing.append(row);self.save('cost',row)

    def finish(self,status,error=None):
        if self.closed:return
        signal.alarm(0)
        current={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.identities}
        import ctypes
        threads=len(list(Path('/proc/self/task').iterdir()))
        runtime_ok=threads==1 and ctypes.CDLL(None).fegetround()==0
        if not runtime_ok:status='runtime_failure'
        self.save('termination',{'status':status,'error':error,'checks':len(self.checks),'all_pass':all(self.checks),
            'sources_unchanged':current==self.identities,'sources':current,
            'threads':threads,'runtime_admitted':runtime_ok,'rounding_final':ctypes.CDLL(None).fegetround(),'wall':time.perf_counter()-self.started,
            'cpu':time.process_time()-self.cpu,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'phases':self.timing,'serialization_wall':self.serialization_wall,'serialization_cpu':self.serialization_cpu})
        self.journal.close();self.report.close();self.closed=True
        if current!=self.identities:raise RuntimeError('job source changed')
        if not runtime_ok:raise RuntimeError('final runtime admission failed')
