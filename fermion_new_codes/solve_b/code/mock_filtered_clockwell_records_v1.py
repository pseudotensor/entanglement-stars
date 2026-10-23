"""UNRUN #145 box-only stage-specific bounded records; old bundles unchanged."""
from pathlib import Path
import ctypes
import hashlib
import os
import resource
import signal
import sys
import time
from mock_unitary_lead_records_v1 import Bundle,JobStop


class ClockwellBundle(Bundle):
    def __init__(self,path,sources,stage,n):
        from cpu_runtime import require_remote_cpu
        path=Path(path)
        if path.exists():raise FileExistsError('Never overwrite an earlier run')
        if stage not in ('controls','static') or type(n) is not int or n not in (2,3,7) or (stage=='controls' and n!=2):raise ValueError('fixed static stage/size')
        self.identities={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
        self.path=path;self.runtime=require_remote_cpu(path);self.counter=0;self.checks=[];self.timing=[]
        self.serialization_wall=0.;self.serialization_cpu=0.;self.started=time.perf_counter();self.cpu=time.process_time();self.closed=False
        names=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        if any(os.environ.get(k)!='1' for k in names):raise RuntimeError('one numerical thread required')
        limit=2*1024**3 if stage=='controls' else 16000000000
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        self.journal=(path/'records.ndjson').open('x');self.report=(path/'written_report.md').open('x')
        import numpy as np
        import scipy
        import mpmath
        if len(list(Path('/proc/self/task').iterdir()))!=1:raise RuntimeError('one resident thread required')
        if ctypes.CDLL(None).fegetround()!=0:raise RuntimeError('FE_TONEAREST required')
        if np.float64(np.finfo(float).tiny)*np.float64(.5)==0 or np.float64(2)*np.float64(2**-1074)!=np.float64(2**-1073):raise RuntimeError('gradual underflow required')
        self.save('intake',dict(runtime=self.runtime,sources=self.identities,command=sys.argv,thread_environment={k:os.environ[k] for k in names},
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mpmath=mpmath.__version__,
            allocation_limit_bytes=limit,rounding='FE_TONEAREST',gradual_underflow=True,n=n,stage=stage,
            physical_propagation=False,filtered_midpoint_evaluation=False))
        def stop(sig,frame):raise JobStop('handled weighted static safety ceiling')
        signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGPROF,stop)
        signal.setitimer(signal.ITIMER_PROF,570 if stage=='controls' else 3500)
        signal.alarm(570 if stage=='controls' else 3550)

    def finish(self,status,error=None):
        signal.setitimer(signal.ITIMER_PROF,0)
        return super().finish(status,error)
