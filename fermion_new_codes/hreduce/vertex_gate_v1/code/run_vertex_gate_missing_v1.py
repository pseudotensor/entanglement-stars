"""One exclusive box CPU allocation; all records remain in its /data directory."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import time

ROOT = Path('<internal>/continuation_v1')
OLD = ROOT.parent
PREDECESSOR_PINS = {'03_model.tex': 'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb', 'ReviewB_pcq5_v1.md': '2bacd47b093cdef79448dd5e4a6c3c4923c7829cfc5855d2ea117bb43630db9f', 'ReviewB_pcq6_v1.md': '9279dacb90acc5bd804629e6058e52765469181e3b823e6dadcdc02be6c149af', 'ReviewA_pcq4_v1.md': '1a4818713fb720ce41e3a90e0aa6d9a7e16c92b351d4bd718e9434371848dcb2', 'ReviewA_pcq5_v1.md': 'eafb1c6487282f65bf3edd5e3416643d2b2451a3e2a6ea39e267a5578c1167b2', 'JOINT_VISION_page_mock_signed_v1.md': 'fe8648e986ce8382b22c13a3ba6b27289d49229ae32f034182e1098428253279', 'VERTEX_gate_author_v1.tex': '7b32632b79d26b771192a86fb721eaeb85326938ba589fb953c502a594943d7c', 'execution.json': '4f48ecac00cb4591c85fb5911ddbf456246e4aff70148aaf8abf29e6675e8497', 'on_site.json': '37155ac55d9ba1ce59d456afcb3bee2060c8784d42ba6060a83a1005cbc5c5f8', 'run_vertex_gate_exact_v1.py': '61b8b82354a9fcbac56e641239c44301c888f25c23055d553235fd3e17704a41', 'spinor_algebra.json': 'af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb', 'stderr.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'stdout.log': '3300b3f59deb0433fad1338d9c67953b4d666b491e5b3ba3bb4fbbcce1c65654', 'vertex_gate_exact_v1.py': '50fa4a6c44b91295a59ac3d91181902aa963166bc328fca6c616ecc6a901f26b', 'vertex_gate_result.json': 'e8898fbdda68834f507b55ee94232deab6791139e225c25ef1d35ccec4c59547'}
SOURCE_SHA = '34c21167be2acd8cea0662d380aa8fc47ded65862176e272d3b0ef2646d167fe'
SPEC_SHA = '1c002616f0711c78687163bf3dbebb9b31e22c600773c7447321d2cf8ce357eb'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()



def predecessor_snapshot(name, pin):
    path = OLD/name
    def identity(s):
        return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                    mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    before_path = identity(path.lstat())
    descriptor = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    h = hashlib.sha256()
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Regular predecessor required: '+name)
        before = identity(info)
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
        after = identity(os.fstat(stream.fileno()))
    after_path = identity(path.lstat())
    if not before_path == before == after == after_path or h.hexdigest() != pin:
        raise ValueError('Predecessor binding failure: '+name)
    return dict(path=str(path), identity=before, sha256=h.hexdigest())


def save(record):
    temporary = ROOT/'execution.json.tmp'
    with temporary.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(ROOT/'execution.json')


class MetadataDeadline(BaseException):
    pass


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Named box directory required')
    if SOURCE_SHA.startswith('PENDING') or SPEC_SHA.startswith('PENDING'):
        raise RuntimeError('Final source and mathematical review are pending')
    pins = {'vertex_gate_missing_v1.py': SOURCE_SHA, 'VERTEX_gate_missing_contract_v1.tex': SPEC_SHA}
    for name, sha in pins.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Reviewed pin mismatch: '+name)
    if any(p.suffix in ('.json', '.jsonl', '.log', '.tmp', '.npz') for p in ROOT.iterdir()):
        raise FileExistsError('Existing allocation; no retry')
    if shutil.disk_usage('/').free < 100*2**30 or shutil.disk_usage('/data').free < 2*2**30:
        raise RuntimeError('Standing disk reserve unavailable')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
    cpu = min(os.sched_getaffinity(0))
    settings = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment = os.environ.copy()
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '104s', 'taskset', '-c', str(cpu),
               '<internal>/python', str(ROOT/'vertex_gate_missing_v1.py')]
    predecessor_before = {name: predecessor_snapshot(name, pin) for name, pin in PREDECESSOR_PINS.items()}
    record = dict(predecessor_before=predecessor_before, status='running', wrapper_pid=os.getpid(), command=command,
                  source_sha256=pins, wrapper_sha256=digest(Path(__file__)), environment=settings,
                  nice=os.getpriority(os.PRIO_PROCESS, 0), cpu=cpu,
                  resources=dict(threads=1, address_space_bytes=2*2**30, wall_seconds=104,
                                 handled_wall_seconds=94, cpu_soft_seconds=84, cpu_hard_seconds=94))
    # The exclusive create prevents an overlapping wrapper from claiming the directory.
    with (ROOT/'execution.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    child = None
    try:
        with (ROOT/'stdout.log').open('xb') as stdout, (ROOT/'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            record['child_pid'] = child.pid
            save(record)
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except BaseException as exc:
        record.update(status='failed_or_uncertain', error=type(exc).__name__+': '+str(exc))
        if child is not None:
            record['cleanup'] = dict(owned_process_group=child.pid)
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                        record['cleanup']['SIGTERM_sent'] = True
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                            record['cleanup']['SIGKILL_sent'] = True
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=5)
                record['cleanup'].update(reaped=True, return_code=child.returncode)
            except BaseException as cleanup:
                record['cleanup'].update(reaped=False, error=type(cleanup).__name__+': '+str(cleanup))
    record['elapsed_seconds'] = time.monotonic()-started
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record['surviving_files'] = {}
    terminal = child is None or child.poll() is not None
    record['child_terminal_before_inventory'] = terminal
    metadata_complete = False
    final_write_complete = False
    if terminal:
        tail_started = time.monotonic()
        predecessor_names = list(PREDECESSOR_PINS)
        inventory_names = []
        record.update(predecessor_after={}, custody_errors=[])
        record['metadata_tail'] = dict(status='running', started_monotonic=tail_started,
            elapsed_seconds=0, ceiling_seconds=10, completed_predecessors=[],
            unvisited_predecessors=predecessor_names.copy(), completed_inventory=[],
            unvisited_inventory=None)
        previous_handler = signal.getsignal(signal.SIGALRM)
        def metadata_stop(sig, frame):
            raise MetadataDeadline('Terminal metadata deadline')
        signal.signal(signal.SIGALRM, metadata_stop)
        signal.setitimer(signal.ITIMER_REAL, 10)
        try:
            save(record)
            for name, pin in PREDECESSOR_PINS.items():
                try:
                    value = predecessor_snapshot(name, pin)
                    record['predecessor_after'][name] = value
                    if value != predecessor_before[name]:
                        record['custody_errors'].append(name+': identity changed')
                except MetadataDeadline:
                    raise
                except Exception as exc:
                    record['custody_errors'].append(name+': '+type(exc).__name__+': '+str(exc))
                record['metadata_tail']['completed_predecessors'].append(name)
                record['metadata_tail']['unvisited_predecessors'].remove(name)
                record['metadata_tail']['elapsed_seconds'] = time.monotonic()-tail_started
                save(record)
            inventory_names = [p.name for p in sorted(ROOT.iterdir())
                               if p.name not in ('execution.json','execution.json.tmp')]
            record['metadata_tail']['unvisited_inventory'] = inventory_names.copy()
            save(record)
            for name in inventory_names:
                path = ROOT/name
                entry = dict(size=path.stat().st_size, sha256=digest(path),
                             unfinished_temporary=path.suffix == '.tmp')
                record['surviving_files'][name] = entry
                record['metadata_tail']['completed_inventory'].append(name)
                record['metadata_tail']['unvisited_inventory'].remove(name)
                record['metadata_tail']['elapsed_seconds'] = time.monotonic()-tail_started
                save(record)
            record['metadata_tail']['status'] = 'complete'
            metadata_complete = True
        except MetadataDeadline as exc:
            record['metadata_tail'].update(status='timeout', error=str(exc))
        except BaseException as exc:
            record['metadata_tail'].update(status='incomplete_error', error=type(exc).__name__+': '+str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            record['metadata_tail']['elapsed_seconds'] = time.monotonic()-tail_started
        record['final_write_attempt'] = dict(ceiling_seconds=2, status='bounded_attempt',
                                             started_monotonic=time.monotonic())
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            save(record)
            final_write_complete = True
        except BaseException:
            # Retain the last atomic checkpoint and any unfinished temporary bytes.
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    else:
        record['inventory_omitted_reason'] = 'Owned child termination uncertain; files may still change'
        previous_handler = signal.getsignal(signal.SIGALRM)
        def final_stop(sig, frame):
            raise MetadataDeadline('Final execution-record deadline')
        signal.signal(signal.SIGALRM, final_stop)
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            record['final_write_attempt'] = dict(ceiling_seconds=2, status='bounded_attempt',
                                                 started_monotonic=time.monotonic())
            save(record)
            final_write_complete = True
        except BaseException:
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    if (record.get('exit_code') != 0 or not metadata_complete or not final_write_complete
            or record.get('custody_errors')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
