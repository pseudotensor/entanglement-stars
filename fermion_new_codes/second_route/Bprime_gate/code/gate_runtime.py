"""Stdlib CPU/path admission for own Bprime saved-output gate work on the permitted box."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys


OUTPUT_ROOT = Path('<internal>/Bprime_gate_v1')
PYTHON_DIRECTORY = Path('<internal>/bin')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
SPEC_SHA256 = 'abe82986096f9e8d4c7260591b34f4a0c4e04d167e3753c421b72e47d8a6b31b'
MAX_TASK_OUTPUT_BYTES = 64*1024**2
MINIMUM_ROOT_FREE_BYTES = 100*10**9


def _stable_stat(value):
    return {key: int(getattr(value, key)) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def identity(path):
    """Hash bytes without interpreting any array, object, log or record value."""
    path = Path(path)
    resolved_before = str(path.resolve(strict=True))
    sha, md5 = hashlib.sha256(), hashlib.md5()
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError('Only regular input files may be hashed: '+str(path))
    with path.open('rb') as stream:
        if _stable_stat(os.fstat(stream.fileno())) != _stable_stat(before):
            raise RuntimeError('Input changed before hashing: '+str(path))
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            md5.update(chunk)
        opened_after = os.fstat(stream.fileno())
    after = path.stat()
    if _stable_stat(before) != _stable_stat(after) or _stable_stat(after) != _stable_stat(opened_after):
        raise RuntimeError('Input changed while hashing: '+str(path))
    if str(path.resolve(strict=True)) != resolved_before:
        raise RuntimeError('Input path target changed while hashing: '+str(path))
    return {'path': str(path), 'resolved_path': resolved_before,
            'bytes': after.st_size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest(),
            'stable_stat': _stable_stat(after)}


def code_bindings():
    directory = Path(__file__).resolve().parent
    return {path.name: identity(path) for path in sorted(directory.glob('gate_*.py'))}


def require_identity(path, expected_sha256):
    value = identity(path)
    if value['sha256'] != expected_sha256:
        raise ValueError('Input identity differs from admitted SHA256: ' + str(path))
    return value


def enforce_storage(required_bytes=0):
    """Check total task bytes, including logs, before and after every save.

    Existing hardlinked content is counted once, plus link text for internal
    symlinks. Symlinks may not escape the task root. No file is removed.
    """
    if type(required_bytes) is not int or required_bytes < 0:
        raise ValueError('Planned output size must be a nonnegative integer')
    root = OUTPUT_ROOT.resolve()
    used, files, symlinks = 0, 0, 0
    seen_files, seen_directories = set(), set()
    pending = [root] if root.exists() else []
    while pending:
        directory = pending.pop()
        directory_stat = directory.stat()
        directory_key = (directory_stat.st_dev, directory_stat.st_ino)
        if directory_key in seen_directories:
            continue
        seen_directories.add(directory_key)
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                current = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(current.st_mode):
                    resolved = path.resolve(strict=True)
                    if not resolved.is_relative_to(root):
                        raise RuntimeError('Task output symlink escapes the admitted root: '+str(path))
                    symlinks += 1
                    used += current.st_size
                    path, current = resolved, resolved.stat()
                if stat.S_ISDIR(current.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(current.st_mode):
                    key = (current.st_dev, current.st_ino)
                    if key not in seen_files:
                        seen_files.add(key)
                        used += current.st_size
                        files += 1
                else:
                    raise RuntimeError('Unsupported task output file type: '+str(path))
    if used+required_bytes > MAX_TASK_OUTPUT_BYTES:
        raise RuntimeError('Resource hold: total task output including logs would exceed64MiB')
    root_stats = os.statvfs('/')
    root_free = root_stats.f_bavail*root_stats.f_frsize
    if root_free < MINIMUM_ROOT_FREE_BYTES:
        raise RuntimeError('Root filesystem must retain at least100GB free')
    existing = root if root.exists() else root.parent
    while not existing.exists():
        existing = existing.parent
    output_stats = os.statvfs(existing)
    output_free = output_stats.f_bavail*output_stats.f_frsize
    if output_free < required_bytes:
        raise RuntimeError('Resource hold: insufficient free space for planned task output')
    return {'task_root': str(root), 'used_bytes': int(used), 'required_new_bytes': required_bytes,
            'cap_bytes': MAX_TASK_OUTPUT_BYTES, 'regular_files': files, 'internal_symlinks': symlinks,
            'root_free_bytes': int(root_free), 'output_filesystem_free_bytes': int(output_free),
            'scope': 'Entire task root including logs and staged files; no deletion or evidence omission.'}


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
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CUDA devices must remain hidden')
    for name in THREAD_VARIABLES:
        if os.environ.get(name) != str(runtime['library_thread_cap']):
            raise RuntimeError('Thread environment changed: ' + name)
    if runtime.get('numba_workers'):
        if os.environ.get('NUMBA_NUM_THREADS') != str(runtime['numba_workers']) or os.environ.get('NUMBA_THREADING_LAYER') != 'workqueue':
            raise RuntimeError('Numba worker environment changed')
    return runtime


def admit(outdir, script, purpose, thread_cap=1, numba_workers=0):
    """Saved-output gate work is limited to one total resident CPU thread."""
    if type(thread_cap) is not int or thread_cap != 1 or type(numba_workers) is not int or numba_workers != 0:
        raise RuntimeError('This task permits exactly one total resident thread and no Numba pool')
    if os.environ.get('QI2_REMOTE_CPU') != '1':
        raise RuntimeError('Remote CPU admission flag required')
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    if 'compute-host' not in addresses:
        raise RuntimeError('Only the permitted box may execute Bprime work')
    if Path(sys.executable).parent != PYTHON_DIRECTORY:
        raise RuntimeError('The qi2 Python environment is required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CUDA devices must be hidden')
    destination = Path(outdir).resolve()
    if not destination.is_relative_to(OUTPUT_ROOT.resolve()):
        raise RuntimeError('Outputs must remain under the Bprime bulk root on the box')
    stats = os.statvfs('/')
    free = stats.f_bavail * stats.f_frsize
    if free < MINIMUM_ROOT_FREE_BYTES:
        raise RuntimeError('Root filesystem must retain at least 100 GB free')
    runtime = {'purpose': purpose, 'actual_script': str(Path(script).resolve()),
               'actual_executable': sys.executable, 'argv': sys.argv,
               'host': 'compute-host', 'pid': os.getpid(), 'device': 'CPU',
               'thread_cap': thread_cap, 'library_thread_cap': 1 if numba_workers else thread_cap,
               'numba_workers': numba_workers,
               'thread_environment': {name: os.environ.get(name) for name in THREAD_VARIABLES},
               'output_directory': str(destination), 'root_filesystem_available_bytes_at_admission': free,
               'custody': 'Original records and all run products remain on the box; this guard does not authorize transfer.'}
    verify_threads(runtime, 'at_admission')
    check_memory(runtime, 'at_admission')
    runtime['storage_at_admission'] = enforce_storage()
    destination.mkdir(parents=True, exist_ok=False)
    return runtime
