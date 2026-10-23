"""One bounded L execution; no retries or replacement of saved products."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


ROOT = Path('<internal>/profiles_v1')
PRODUCER = ROOT / 'core_well_profiles_v1.py'
HELPER = ROOT / 'core_well_input_v1.py'
RESULT = ROOT / 'core_well_profiles_v1.json'
EXIT = ROOT / 'execution.json'
STDOUT = ROOT / 'profiles.stdout.log'
STDERR = ROOT / 'profiles.stderr.log'
PRODUCER_SHA = '8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54'
HELPER_SHA = '294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(record, first=False):
    if first:
        with EXIT.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    else:
        temporary = EXIT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(EXIT)


def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (1610612736, 1610612736))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('This execution wrapper runs only in its named box folder')
    controls = (RESULT, EXIT, STDOUT, STDERR, EXIT.with_suffix('.json.tmp'),
                RESULT.with_suffix('.json.tmp'))
    patterns = ('raw_T*.npz*', 'joint_T*.npz*', 'spectral_T*.npz*',
                'matched_T*.npz*', 'eigensystem_T*.npz*', 'spectral_basis_T*.npy*')
    if any(p.exists() for p in controls) or any(any(ROOT.glob(p)) for p in patterns):
        raise FileExistsError('Existing or uncertain work requires assessment, never repetition')
    if sha(PRODUCER) != PRODUCER_SHA or sha(HELPER) != HELPER_SHA:
        raise ValueError('Producer or numeric-member helper binding changed')
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    root_free, data_free = shutil.disk_usage('/').free, shutil.disk_usage('/data').free
    if available < 64 * 2**30 or root_free < 100_000_000_000 or data_free < 8 * 2**30:
        raise RuntimeError('The specified memory or disk reserve is unavailable')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four allowed CPUs are unavailable')
    command = ['timeout', '-s', 'KILL', '900s', 'nice', '-n', '10',
               'taskset', '-c', ','.join(map(str, cpus)),
               '<internal>/python', str(PRODUCER)]
    environment = os.environ.copy()
    settings = {key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS')}
    settings.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    environment.update(settings)
    record = dict(status='running', purpose='owner66 L fixed three-slice second route',
        command=command, producer_sha256=PRODUCER_SHA, helper_sha256=HELPER_SHA,
        wrapper_sha256=sha(Path(__file__)), wrapper_pid=os.getpid(),
        started_unix=time.time(), selected_cpus=cpus,
        resources=dict(cpu_threads=4, wall_seconds=900, cpu_seconds=3600,
            internal_deadline_seconds=880, address_space_bytes=1610612736,
            nice=10, cuda='disabled'), environment_settings=settings,
        preflight=dict(mem_available_bytes=available, root_free_bytes=root_free,
            data_free_bytes=data_free, load_average=os.getloadavg()),
        stdout_path=str(STDOUT), stderr_path=str(STDERR))
    save(record, first=True)
    start = time.monotonic()
    try:
        with STDOUT.open('xb') as out, STDERR.open('xb') as err:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True, preexec_fn=child_limits)
            record['child_pid'] = child.pid
            save(record)
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain', error=type(error).__name__ + ': ' + str(error))
    record['elapsed_seconds'] = time.monotonic() - start
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record['rss_scope'] = 'fresh wrapper; maximum over this sole command and descendants'
    for name, path in (('stdout', STDOUT), ('stderr', STDERR)):
        if path.exists():
            record[name + '_bytes'] = path.stat().st_size
            record[name + '_sha256'] = sha(path)
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                record[name + '_tail'] = stream.read().decode(errors='replace')
    if RESULT.exists():
        record['result_identity'] = dict(path=str(RESULT), size=RESULT.stat().st_size,
                                         sha256=sha(RESULT))
        try:
            result = json.loads(RESULT.read_text())
            record['producer_status'] = result.get('status')
        except Exception as error:
            record['result_read_error'] = type(error).__name__ + ': ' + str(error)
    save(record)


if __name__ == '__main__':
    main()
