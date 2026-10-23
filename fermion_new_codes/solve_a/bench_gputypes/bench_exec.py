#!/usr/bin/env python3
"""bench_exec.py (solve bench_gputypes v1): run one kit command on one GPU with telemetry; stdlib only (any python3).
Usage: bench_exec.py --gpu IDX --out DIR --log FILE -- CMD ...
Before: refuses if the GPU (nvidia-smi index IDX) holds any compute process. During: nvidia-smi --query-gpu sampling every 1 s of that
GPU into DIR/smi.csv (utilization.gpu, memory.used, power, SM clock). After: DIR/bench.json with wall, child CPU time (getrusage of the
children: user + sys, max RSS), the cgroup cpu.stat delta where readable, GPU-busy fraction (mean utilization.gpu / 100 and the share
of samples with utilization > 0), peak memory.used and the delta over the pre-run baseline, the host and GPU identity, the thread env.
The command's stdout and stderr go to FILE. Exit code: the command's."""
import argparse, json, os, platform, resource, socket, subprocess, sys, threading, time


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return 'ERR %s' % e


def cgroup_stat():
    for p in ('/sys/fs/cgroup/cpu.stat', '/sys/fs/cgroup/cpu,cpuacct/cpu.stat', '/sys/fs/cgroup/cpu/cpu.stat'):
        try:
            return {l.split()[0]: int(l.split()[1]) for l in open(p) if len(l.split()) == 2}
        except Exception:
            pass
    return None


def cpu_quota():
    try:
        q, p = open('/sys/fs/cgroup/cpu.max').read().split()
        return None if q == 'max' else float(q) / float(p)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--allow-busy', action='store_true', help='skip the empty-GPU guard (never used for a benchmark number)')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    os.makedirs(a.out, exist_ok=True)
    g = a.gpu
    uuid = sh('nvidia-smi -i %s --query-gpu=uuid --format=csv,noheader' % g)
    busy = [l for l in sh('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader').splitlines() if uuid and l.startswith(uuid)]
    if busy and not a.allow_busy:
        print('bench_exec: refusing: GPU %s (%s) holds compute processes: %s' % (g, uuid, busy), file=sys.stderr)
        return 3
    ident = {
        'host': socket.gethostname(), 'cpu_model': sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(), 'nproc': os.cpu_count(),
        'cgroup_cpu_quota': cpu_quota(), 'mem_total_kB': sh("grep MemTotal /proc/meminfo | awk '{print $2}'"),
        'gpu_index': g, 'gpu_uuid': uuid,
        'gpu': sh('nvidia-smi -i %s --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,power.limit --format=csv,noheader' % g),
        'python_exec': sys.executable, 'platform': platform.platform(),
        'env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTORCH_CUDA_ALLOC_CONF', 'NUMPY_MADVISE_HUGEPAGE')},
        'cmd': cmd,
    }
    base_mem = sh('nvidia-smi -i %s --query-gpu=memory.used --format=csv,noheader,nounits' % g)
    smi_path = os.path.join(a.out, 'smi.csv')
    smi = subprocess.Popen(['nvidia-smi', '-i', str(g), '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm', '--format=csv,noheader,nounits', '-lms', '1000'],
                           stdout=open(smi_path, 'w'), stderr=subprocess.DEVNULL)
    cg0 = cgroup_stat()
    t0 = time.time(); t0u = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))
    seen, stop = set(), threading.Event()
    def apps():   # the compute processes on this GPU every 20 s (another job starting on the card would spoil the number)
        while not stop.is_set():
            for l in sh('nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader').splitlines():
                if uuid and l.startswith(uuid):
                    seen.add(','.join(l.split(',')[1:3]).strip())
            stop.wait(20)
    th = threading.Thread(target=apps, daemon=True); th.start()
    with open(a.log, 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        rc = p.wait()
    stop.set()
    t1 = time.time()
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    cg1 = cgroup_stat()
    time.sleep(1.5); smi.terminate(); smi.wait()
    util, mem = [], []
    for l in open(smi_path):
        f = [x.strip() for x in l.split(',')]
        try:
            util.append(float(f[1])); mem.append(float(f[2]))
        except Exception:
            pass
    try:
        b = float(base_mem)
    except Exception:
        b = None
    res = dict(ident)
    res.update({
        'rc': rc, 'start_utc': t0u, 'wall_s': round(t1 - t0, 2), 'cpu_user_s': round(ru.ru_utime, 1), 'cpu_sys_s': round(ru.ru_stime, 1),
        'cpu_time_s': round(ru.ru_utime + ru.ru_stime, 1), 'cpu_cores_mean': round((ru.ru_utime + ru.ru_stime) / max(t1 - t0, 1e-9), 2),
        'child_maxrss_GiB': round(ru.ru_maxrss / 1048576.0, 2),
        'gpu_samples': len(util), 'gpu_busy_mean_util': round(sum(util) / len(util) / 100.0, 3) if util else None,
        'gpu_busy_share_nonzero': round(sum(1 for u in util if u > 0) / len(util), 3) if util else None,
        'gpu_mem_baseline_MiB': b, 'gpu_mem_peak_MiB': max(mem) if mem else None,
        'gpu_mem_peak_delta_MiB': (max(mem) - b) if (mem and b is not None) else None,
        'compute_apps_seen': sorted(seen), 'compute_apps_count': len(seen),
    })
    if cg0 and cg1:
        res['cgroup_delta'] = {k: cg1.get(k, 0) - cg0.get(k, 0) for k in ('usage_usec', 'nr_periods', 'nr_throttled', 'throttled_usec')}
    json.dump(res, open(os.path.join(a.out, 'bench.json'), 'w'), indent=1)
    print('bench_exec: rc %s wall_s %.1f cpu_time_s %.1f gpu_busy %s peak_MiB %s' % (rc, t1 - t0, ru.ru_utime + ru.ru_stime, res['gpu_busy_mean_util'], res['gpu_mem_peak_MiB']))
    return rc


if __name__ == '__main__':
    sys.exit(main())
