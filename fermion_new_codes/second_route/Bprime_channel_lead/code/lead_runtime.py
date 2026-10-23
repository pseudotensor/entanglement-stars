"""Stdlib CPU/path admission for own Bprime channel-lead work on the permitted box."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


OUTPUT_ROOT = Path('<internal>/Bprime_channel_lead_v1')
PYTHON_DIRECTORY = Path('<internal>/bin')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
SPEC_SHA256 = '24c0e2ce1d9cc08005b8813a6f8e83c3c6bce875e9e6f16dc0f5e0e38f553ff4'
DERIVATION_SHA256 = '9a633c86b5895cfb73d4d8fede127057afc2c66942d4dfc91da969dd619852ec'


def identity(path):
    """Hash bytes without interpreting any array, object, log or record value."""
    path = Path(path)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    before = path.stat()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            md5.update(chunk)
    after = path.stat()
    for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns'):
        if getattr(before, key) != getattr(after, key):
            raise RuntimeError('Input changed while hashing: ' + str(path))
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'bytes': after.st_size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def code_bindings():
    directory = Path(__file__).resolve().parent
    return {path.name: identity(path) for path in sorted(directory.glob('lead_*.py'))}


def core_bindings():
    directory = Path(__file__).resolve().parent
    names = ('lead_runtime.py', 'lead_native.py', 'lead_channels.py', 'lead_setup.py', 'lead_symmetry.py',
             'lead_operator.py', 'lead_numerics.py', 'lead_controls.py')
    return {name: identity(directory/name) for name in names}


def require_identity(path, expected_sha256):
    value = identity(path)
    if value['sha256'] != expected_sha256:
        raise ValueError('Input identity differs from admitted SHA256: ' + str(path))
    return value


def require_fresh(outdir):
    if any(Path(outdir).iterdir()):
        raise RuntimeError('Preserve reviewed outputs; use a fresh empty directory')


def check_memory(runtime, phase, workspace_bytes=0):
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts = line.split()
        values[parts[0].rstrip(':')] = int(parts[1])*1024
    available, reserve = values['MemAvailable'], 64*1024**3
    runtime['memory_'+phase] = {'available_bytes': available, 'estimated_new_workspace_bytes': int(workspace_bytes),
                               'required_remaining_reserve_bytes': reserve}
    if available-int(workspace_bytes) < reserve:
        raise RuntimeError('Available memory must preserve 64 GiB after workspace: '+phase)
    return runtime


def verify_threads(runtime, phase):
    observed = len(list(Path('/proc/self/task').iterdir()))
    cap = runtime['thread_cap']
    runtime['resident_threads_' + phase] = observed
    runtime['actual_niceness'] = os.getpriority(os.PRIO_PROCESS, 0)
    if not 1 <= observed <= cap:
        raise RuntimeError('Resident thread count exceeds the admitted cap: ' + phase)
    if runtime['actual_niceness'] < 10:
        raise RuntimeError('Niceness must remain at least ten')
    for name in THREAD_VARIABLES:
        if os.environ.get(name) != str(runtime['library_thread_cap']):
            raise RuntimeError('Thread environment changed: ' + name)
    if runtime.get('numba_workers'):
        if os.environ.get('NUMBA_NUM_THREADS') != str(runtime['numba_workers']) or os.environ.get('NUMBA_THREADING_LAYER') != 'workqueue':
            raise RuntimeError('Numba worker environment changed')
    if runtime.get('mode') == 'dense' and any(name == 'numba' or name.startswith('numba.') for name in sys.modules):
        raise RuntimeError('Standalone dense setup must not import Numba or create its pool')
    return runtime


def admit(outdir, script, purpose, thread_cap=1, numba_workers=0, mode=None, blas_threads=None):
    """Admit metadata, standalone dense setup, or mixed BLAS1/Numba CPU work."""
    mode = ('mixed' if numba_workers else 'dense' if thread_cap > 1 else 'metadata') if mode is None else mode
    if mode not in ('metadata', 'dense', 'mixed'):
        raise RuntimeError('Unknown CPU admission mode')
    if type(thread_cap) is not int or not 1 <= thread_cap <= 16:
        raise RuntimeError('Thread cap must be an integer from one through sixteen')
    if type(numba_workers) is not int or not 0 <= numba_workers <= 15 or numba_workers >= thread_cap:
        raise RuntimeError('Numba workers plus the caller must fit the total resident-thread cap')
    library_cap = (1 if mode in ('metadata', 'mixed') else min(8, thread_cap)) if blas_threads is None else blas_threads
    if type(library_cap) is not int or library_cap < 1 or library_cap > (8 if mode == 'dense' else 1):
        raise RuntimeError('Dense setup permits at most eight BLAS threads per pool; other modes require one')
    if mode == 'mixed' and not numba_workers or mode != 'mixed' and numba_workers:
        raise RuntimeError('Only mixed mode may admit a Numba pool')
    if mode == 'metadata' and thread_cap != 1:
        raise RuntimeError('Metadata requires one total thread')
    if os.environ.get('QI2_REMOTE_CPU') != '1':
        raise RuntimeError('Remote CPU admission flag required')
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    if 'compute-host' not in addresses:
        raise RuntimeError('Only the permitted box may execute Bprime channel-lead work')
    if Path(sys.executable).parent != PYTHON_DIRECTORY:
        raise RuntimeError('The qi2 Python environment is required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CUDA devices must be hidden')
    destination = Path(outdir).resolve()
    if not destination.is_relative_to(OUTPUT_ROOT.resolve()):
        raise RuntimeError('Outputs must remain under the Bprime channel-lead bulk root on the box')
    stats = os.statvfs('/')
    free = stats.f_bavail * stats.f_frsize
    if free < 100 * 10**9:
        raise RuntimeError('Root filesystem must retain at least 100 GB free')
    runtime = {'purpose': purpose, 'actual_script': str(Path(script).resolve()),
               'actual_executable': sys.executable, 'argv': sys.argv,
               'host': 'compute-host', 'pid': os.getpid(), 'device': 'CPU',
               'thread_cap': thread_cap, 'library_thread_cap': library_cap, 'mode': mode,
               'numba_workers': numba_workers,
               'thread_environment': {name: os.environ.get(name) for name in THREAD_VARIABLES},
               'output_directory': str(destination), 'root_filesystem_available_bytes_at_admission': free,
               'custody': 'Original records and all run products remain on the box; this guard does not authorize transfer.'}
    verify_threads(runtime, 'at_admission')
    check_memory(runtime, 'at_admission')
    destination.mkdir(parents=True, exist_ok=True)
    return runtime
