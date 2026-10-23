"""Record exact admitted-control invocation and same-environment software metadata."""
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from cpu_runtime import require_remote_cpu
from packet_core_v1 import file_hash

output=Path('<internal>/PACKET_secondroute_v1')
runtime=require_remote_cpu(output)
config=io.StringIO()
with contextlib.redirect_stdout(config):
    np.show_config()
control=output/'known_answers.json'
artifact=json.loads(control.read_text())
if not artifact['passed'] or artifact['count']!=435:
    raise ValueError('actual controls mismatch')
record={'metadata_runtime':runtime, 'control_runtime':artifact['runtime'],
    'controls_sha256':file_hash(control), 'controls_exit_code':0,
    'control_command':['nice','-n','10','<internal>/python',
        str(output/'code/control_PACKET_v1.py'),'--output',str(output),'--spec',str(output/'PACKET_secondroute_spec_v1.md')],
    'control_environment':{key:os.environ[key] for key in ('QI2_REMOTE_CPU','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
    'transport':'passwordless ssh compute-host; physical execution on that host only',
    'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,
    'platform':platform.platform(),'numpy_backend_configuration':config.getvalue(),
    'metadata_code_sha256':file_hash(__file__)}
(output/'execution_environment.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'environment_saved':True,'numpy':np.__version__,'scipy':scipy.__version__}))
