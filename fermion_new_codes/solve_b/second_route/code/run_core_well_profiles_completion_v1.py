"""Execute only the separately bounded unfinished L target, with no retries."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


ROOT = Path('<internal>/missing_work')
PRODUCER = ROOT / 'finish_core_well_profiles_v1.py'
RESULT = ROOT / 'core_well_profiles_v1.json'
EXIT = ROOT / 'execution.json'
STDOUT = ROOT / 'profiles.stdout.log'
STDERR = ROOT / 'profiles.stderr.log'
PRIOR_EXIT = Path('<internal>/execution.json')
PRIOR_EXIT_SHA = '1f49c90f8b83093e44b92632ebafbeb61c65bc4fd2f3ad99ae730916e1af16bc'
PRODUCER_SHA = 'a6a13f3935d7cd937eaf59e42e242285341a5210fe5b0f60467f6a232f8ccacc'
WALL_CAP = 6000
CPU_CAP = 24000
FINALIZATION_WINDOW = 60
MEMORY_CAP = 1610612736


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
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP, MEMORY_CAP))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_CAP, CPU_CAP))


def main():
    started = time.monotonic()
    child_deadline = started + WALL_CAP - FINALIZATION_WINDOW
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named own-lane completion folder')
    controls = (RESULT, EXIT, STDOUT, STDERR, EXIT.with_suffix('.json.tmp'),
                RESULT.with_suffix('.json.tmp'))
    patterns = ('raw_T*.npz*', 'joint_T*.npz*', 'spectral_T*.npz*',
                'matched_T*.npz*', 'eigensystem_T*.npz*', 'spectral_basis_T*.npy*')
    if any(p.exists() for p in controls) or any(any(ROOT.glob(p)) for p in patterns):
        raise FileExistsError('Existing completion requires assessment, never repetition')
    if sha(PRODUCER) != PRODUCER_SHA or sha(PRIOR_EXIT) != PRIOR_EXIT_SHA:
        raise ValueError('Completion producer or closed execution binding changed')
    prior = json.loads(PRIOR_EXIT.read_text())
    if prior['status'] != 'finished' or prior['exit_code'] != 124:
        raise ValueError('The prescribed closed attempt is not the saved timeout')
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    root_free, data_free = shutil.disk_usage('/').free, shutil.disk_usage('/data').free
    if available < 64 * 2**30 or root_free < 100_000_000_000 or data_free < 8 * 2**30:
        raise RuntimeError('The specified memory or disk reserve is unavailable')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 4:
        raise RuntimeError('The outer whole-wrapper guard must assign exactly four CPUs')
    settings = {key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS')}
    settings.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                    QI2_CORE_WELL_CHILD_DEADLINE=repr(child_deadline))
    environment = os.environ.copy()
    environment.update(settings)
    record = dict(status='preparing', purpose='same owner66 L target; exact missing work only',
        producer_sha256=PRODUCER_SHA, wrapper_sha256=sha(Path(__file__)),
        wrapper_pid=os.getpid(), started_unix=time.time(), selected_cpus=cpus,
        prior_execution_path=str(PRIOR_EXIT), prior_execution_sha256=PRIOR_EXIT_SHA,
        closed_prior_charged_wall_seconds=prior['aggregate_elapsed_seconds_charged'],
        closed_prior_cpu_upper_bound=prior['aggregate_cpu_seconds_charged_upper_bound'],
        closed_prior_local_reserve_flag=prior['within_original_budget_with_final_save_reserve'],
        resources=dict(separate_allocation=True, wall_seconds=WALL_CAP, cpu_seconds=CPU_CAP,
            cpu_threads=4, address_space_bytes=MEMORY_CAP, finalization_window_seconds=FINALIZATION_WINDOW,
            child_scheduling_deadline_monotonic=child_deadline, new_spectrum_minimum_seconds=420,
            child_reserve_scope='nominal 60s reserve; relative-timeout launch latency may reduce it',
            last_atomic_save_reserve_seconds=1,
            accounting_scope='external default whole-wrapper 6000s guard; shared four-CPU affinity',
            nice=10, cuda='disabled'), environment_settings=settings,
        preflight=dict(mem_available_bytes=available, root_free_bytes=root_free,
            data_free_bytes=data_free, load_average=os.getloadavg()),
        stdout_path=str(STDOUT), stderr_path=str(STDERR))
    save(record, first=True)
    try:
        with STDOUT.open('xb') as out, STDERR.open('xb') as err:
            remaining = child_deadline - time.monotonic()
            if remaining < 420:
                raise TimeoutError('The child deadline was exhausted during preparation')
            command = ['timeout', '--foreground', '-s', 'KILL', str(remaining) + 's',
                'nice', '-n', '10', 'taskset', '-c', ','.join(map(str, cpus)),
                '<internal>/python', str(PRODUCER)]
            record['command'] = command
            record['child_timeout_seconds'] = remaining
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, preexec_fn=child_limits)
            record.update(status='running', child_pid=child.pid)
            save(record)
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain', error=type(error).__name__ + ': ' + str(error))
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
            record['producer_status'] = json.loads(RESULT.read_text()).get('status')
        except Exception as error:
            record['result_read_error'] = type(error).__name__ + ': ' + str(error)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    own_usage = resource.getrusage(resource.RUSAGE_SELF)
    record['elapsed_seconds_before_final_save'] = time.monotonic() - started
    record['new_allocation_charged_wall_seconds'] = record['elapsed_seconds_before_final_save'] + 1
    record['child_cpu_seconds'] = usage.ru_utime + usage.ru_stime
    record['wrapper_cpu_seconds_before_final_save'] = own_usage.ru_utime + own_usage.ru_stime
    record['new_allocation_charged_cpu_upper_bound'] = (
        record['child_cpu_seconds'] + record['wrapper_cpu_seconds_before_final_save'] + 4)
    record['all_attempts_charged_wall_upper_bound'] = (
        record['closed_prior_charged_wall_seconds'] + record['new_allocation_charged_wall_seconds'])
    record['all_attempts_charged_cpu_upper_bound'] = (
        record['closed_prior_cpu_upper_bound'] + record['new_allocation_charged_cpu_upper_bound'])
    record['maximum_child_rss_kib'] = usage.ru_maxrss
    record['rss_scope'] = 'fresh wrapper; maximum over this sole command and descendants'
    record['new_allocation_with_final_save_reserve_respected'] = (
        record['new_allocation_charged_wall_seconds'] <= WALL_CAP and
        record['new_allocation_charged_cpu_upper_bound'] <= CPU_CAP)
    record['termination_rule'] = 'missing final record or outer timeout is uncertain; never retry automatically'
    save(record)
    if record.get('exit_code') != 0 or record.get('producer_status') != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
