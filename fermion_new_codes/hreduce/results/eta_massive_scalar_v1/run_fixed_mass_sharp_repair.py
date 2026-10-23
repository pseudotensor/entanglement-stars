"""Run the one prescribed fixed-mass repair; never replace a prior attempt."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


ROOT = Path('<internal>/eta_massive_scalar_v1')
PRODUCER = ROOT / 'fixed_mass_interval_sharp_repair.py'
AXIAL = ROOT / 'stable_result.json'
RESULT = ROOT / 'fixed_mass_sharp_repair_result.json'
EXIT = ROOT / 'fixed_mass_sharp_repair_exit.json'
STDOUT = ROOT / 'fixed_mass_sharp_repair.stdout.log'
STDERR = ROOT / 'fixed_mass_sharp_repair.stderr.log'
PRODUCER_SHA = 'd962ea9b594c2495284069eb2814ba6d6263786c374bd1b51475d15b29f3e870'
AXIAL_SHA = '71c5df353b17ecde07d69872c611ce8c998fa419243d5151f71ca1713213bf52'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(record, first=False):
    if first:
        with EXIT.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    else:
        temporary = EXIT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(EXIT)


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('This numerical launcher runs only in its named box folder')
    if any(path.exists() for path in (RESULT, EXIT, STDOUT, STDERR,
                                      EXIT.with_suffix('.json.tmp'),
                                      RESULT.with_suffix('.json.tmp'))):
        raise FileExistsError('Existing attempt requires assessment, never repetition')
    if sha(PRODUCER) != PRODUCER_SHA or sha(AXIAL) != AXIAL_SHA:
        raise ValueError('The reviewed producer or accepted axial input changed')
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('The existing memory or root-disk reserve is unavailable')
    if shutil.disk_usage('/data').free < 2**30:
        raise RuntimeError('Insufficient space for the bounded scalar result')
    cpu = min(os.sched_getaffinity(0))
    command = ['timeout', '-s', 'KILL', '300s', 'nice', '-n', '10',
               'taskset', '-c', str(cpu), '<internal>/python',
               str(PRODUCER)]
    environment = os.environ.copy()
    settings = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}
    settings['CUDA_VISIBLE_DEVICES'] = ''
    environment.update(settings)
    record = dict(status='running', purpose='insert I6a; item7 fixed-mass light-cut ratio',
        command=command, producer_sha256=PRODUCER_SHA, axial_input_sha256=AXIAL_SHA,
        launcher_sha256=sha(Path(__file__)),
        wrapper_pid=os.getpid(), started_unix=time.time(), selected_cpu=cpu,
        resources=dict(cpu_threads=1, wall_seconds=300, address_space_bytes=805306368,
                       nice=10, cuda='disabled'),
        environment_settings=settings, stdout_path=str(STDOUT), stderr_path=str(STDERR))
    save(record, first=True)
    start = time.monotonic()
    try:
        with STDOUT.open('xb') as out, STDERR.open('xb') as err:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err)
            record['child_pid'] = child.pid
            save(record)
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain', error=type(error).__name__ + ': ' + str(error))
    record['elapsed_seconds'] = time.monotonic() - start
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for label, path in (('stdout', STDOUT), ('stderr', STDERR)):
        if path.exists():
            record[label + '_bytes'] = path.stat().st_size
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                record[label + '_tail'] = stream.read().decode(errors='replace')
    if RESULT.exists():
        record['result_identity'] = dict(path=str(RESULT), size=RESULT.stat().st_size,
                                         sha256=sha(RESULT))
    save(record)


if __name__ == '__main__':
    main()
