"""Run only the saved L continuation within its remaining original budget."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


ROOT = Path('<internal>/memory_continuation')
PRODUCER = ROOT / 'continue_core_well_profiles_v1.py'
RESULT = ROOT / 'core_well_profiles_v1.json'
EXIT = ROOT / 'execution.json'
STDOUT = ROOT / 'profiles.stdout.log'
STDERR = ROOT / 'profiles.stderr.log'
PRODUCER_SHA = 'd2b4540bd16e3e95304d57b2a4641597923021b70b94402a749e93783cb1f3b7'
PRIOR_EXIT_SHA = '3aa2a0082652c47d8d214da71ea9b66d6d7ba5d898a424a4f94d9473798b029d'
PRIOR_WALL = 9.88800534978509
WALL_CAP = 890
CPU_CAP = 3560


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
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_CAP, CPU_CAP))


def main():
    started = time.monotonic()
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box continuation folder')
    controls = (RESULT, EXIT, STDOUT, STDERR, EXIT.with_suffix('.json.tmp'),
                RESULT.with_suffix('.json.tmp'))
    patterns = ('raw_T*.npz*', 'joint_T*.npz*', 'spectral_T*.npz*',
                'matched_T*.npz*', 'eigensystem_T*.npz*', 'spectral_basis_T*.npy*')
    if any(p.exists() for p in controls) or any(any(ROOT.glob(p)) for p in patterns):
        raise FileExistsError('Existing continuation requires assessment, never repetition')
    if sha(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Saved continuation producer binding changed')
    prior_exit = ROOT.parent / 'execution.json'
    if sha(prior_exit) != PRIOR_EXIT_SHA:
        raise ValueError('Original execution record changed')
    prior = json.loads(prior_exit.read_text())
    if prior['exit_code'] != 1 or prior['elapsed_seconds'] != PRIOR_WALL:
        raise ValueError('Original spent budget differs')
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    root_free, data_free = shutil.disk_usage('/').free, shutil.disk_usage('/data').free
    if available < 64 * 2**30 or root_free < 100_000_000_000 or data_free < 8 * 2**30:
        raise RuntimeError('The specified memory or disk reserve is unavailable')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four allowed CPUs are unavailable')
    os.sched_setaffinity(0, cpus)
    settings = {key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS')}
    settings.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    environment = os.environ.copy()
    environment.update(settings)
    # Charge wrapper preflight and setup against the remaining wall allocation.
    remaining = WALL_CAP - (time.monotonic() - started) - 1
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError('Continuation budget expired before launch')
    command = ['timeout', '--foreground', '-s', 'KILL', str(remaining) + 's', 'nice', '-n', '10',
               'taskset', '-c', ','.join(map(str, cpus)),
               '<internal>/python', str(PRODUCER)]
    record = dict(status='running', purpose='owner66 L missing-work continuation; owner88 takeover',
        command=command, producer_sha256=PRODUCER_SHA,
        prior_execution_path=str(prior_exit), prior_execution_sha256=PRIOR_EXIT_SHA,
        wrapper_sha256=sha(Path(__file__)), wrapper_pid=os.getpid(),
        started_unix=time.time(), selected_cpus=cpus,
        resources=dict(cpu_threads=4, wall_seconds=WALL_CAP, cpu_seconds=CPU_CAP,
            internal_deadline_seconds=870, address_space_bytes=1610612736,
            original_wall_seconds=900, original_cpu_seconds=3600,
            prior_wall_seconds=PRIOR_WALL, prior_cpu_seconds_upper_bound=4 * PRIOR_WALL,
            final_save_reserve_seconds=1,
            accounting_scope='outer whole-wrapper hard 890s guard required; four CPUs bound total CPU',
            nice=10, cuda='disabled'), environment_settings=settings,
        preflight=dict(mem_available_bytes=available, root_free_bytes=root_free,
            data_free_bytes=data_free, load_average=os.getloadavg()),
        stdout_path=str(STDOUT), stderr_path=str(STDERR))
    save(record, first=True)
    try:
        with STDOUT.open('xb') as out, STDERR.open('xb') as err:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, preexec_fn=child_limits)
            record['child_pid'] = child.pid
            save(record)
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain', error=type(error).__name__ + ': ' + str(error))
    for name, path in (('stdout', STDOUT), ('stderr', STDERR)):
        if path.exists():
            record[name + '_bytes'] = path.stat().st_size
            record[name + '_sha256'] = sha(path)
            # Bulk logs remain on the box; only a bounded final tail enters JSON.
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                record[name + '_tail'] = stream.read().decode(errors='replace')
    if RESULT.exists():
        record['result_identity'] = dict(path=str(RESULT), size=RESULT.stat().st_size,
                                         sha256=sha(RESULT))
        try:
            record['producer_status'] = json.loads(RESULT.read_text()).get('status')
        except Exception as error:
            record['result_read_error'] = type(error).__name__ + ': ' + str(error)
    # Sample after result/log bookkeeping, then charge the final atomic-save reserve.
    # The outer whole-wrapper guard is authoritative if finalization is interrupted.
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    own_usage = resource.getrusage(resource.RUSAGE_SELF)
    record['elapsed_seconds_before_final_save'] = time.monotonic() - started
    record['elapsed_seconds_charged'] = record['elapsed_seconds_before_final_save'] + 1
    record['aggregate_elapsed_seconds_charged'] = PRIOR_WALL + record['elapsed_seconds_charged']
    record['child_cpu_seconds'] = usage.ru_utime + usage.ru_stime
    record['wrapper_cpu_seconds_before_final_save'] = own_usage.ru_utime + own_usage.ru_stime
    record['aggregate_cpu_seconds_charged_upper_bound'] = (
        4 * PRIOR_WALL + record['child_cpu_seconds'] +
        record['wrapper_cpu_seconds_before_final_save'] + 4)
    record['whole_wrapper_cpu_cap_by_affinity'] = 4 * WALL_CAP
    record['maximum_child_rss_kib'] = usage.ru_maxrss
    record['rss_scope'] = 'fresh wrapper; maximum over this sole command and descendants'
    record['within_original_budget_with_final_save_reserve'] = (
        record['elapsed_seconds_charged'] <= WALL_CAP and
        record['aggregate_elapsed_seconds_charged'] <= 900 and
        record['aggregate_cpu_seconds_charged_upper_bound'] <= 3600)
    record['termination_rule'] = 'missing final record or outer timeout means incomplete or uncertain; no retry'
    save(record)


if __name__ == '__main__':
    main()
