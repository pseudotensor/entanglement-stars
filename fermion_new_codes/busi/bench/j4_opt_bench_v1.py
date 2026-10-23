#!/usr/bin/env python
"""j4_opt_bench_v1.py -- the only cheap J4 probe (busi/notes/GPU_opt_judgment_v1.md, section 5; optional, low priority):
the FP64 symmetric eigensolver behind j4lib.block_pair_entropy's torch route, cuSOLVER against MAGMA through
torch.backends.cuda.preferred_linalg_library, on Gram matrices of the production block sizes, with the per-block
composition the design note estimates (the host-to-device copy of K, the Gram product K K^T, eigvalsh) and the roofline
numbers of the judgment (the FP64 rate reached and the ratio to the bandwidth floor of a 1-stage tridiagonalization,
n^3/6 doubles over 3.35 TB/s). Gate: the two libraries' entropies agree to 1e-9 nats and each eigenvalue sum equals the
trace to 1e-10 relative. Expected gain at most 1.3x and possibly none; it would matter only for a J4 rerun or further radii.
Idle GPU only (n 32768 needs 52 GB); never beside J4's stage E or a GAP10 run.

  python j4_opt_bench_v1.py --sizes 8192,16384 [--sizes 8192,16384,32768] [--libs cusolver,magma] [--reps 2]
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--sizes', default='8192,16384')
ap.add_argument('--libs', default='cusolver,magma')
ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--code_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'j4'))
ap.add_argument('--json', default='j4_opt_bench_v1.json')
ap.add_argument('--bw_tbs', type=float, default=3.35, help="HBM bandwidth of the device in TB/s (H100 SXM 3.35)")
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.code_dir))
import j4lib as jl  # noqa: E402

dev = torch.device('cuda')
rec = dict(device=torch.cuda.get_device_name(dev), torch=torch.__version__, rows=[], failures=[])
print(f"[j4bench] {rec['device']} torch {torch.__version__} free {torch.cuda.mem_get_info(dev)[0] / 1e9:.1f} GB", flush=True)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t0


for n in [int(s) for s in args.sizes.split(',') if s]:
    rng = np.random.default_rng(n)
    K = rng.normal(size=(n, n + 16)) / (2.0 * np.sqrt(n))          # singular values in [0, about 1]: a Gram spectrum in [0, 1]
    row = dict(n=n, libs={})
    Kd, row['h2d_s'] = timed(lambda: torch.from_numpy(K).to(dev))
    Gm, row['gram_s'] = timed(lambda: Kd @ Kd.T)
    del Kd
    trace = float(torch.diagonal(Gm).sum().item())
    S = {}
    for lib in [s for s in args.libs.split(',') if s]:
        try:
            torch.backends.cuda.preferred_linalg_library(lib)
        except Exception as e:
            row['libs'][lib] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f"[j4bench n={n}] {lib}: unavailable ({row['libs'][lib]['error']})", flush=True)
            continue
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            ts = []
            for r in range(args.reps):
                ev, t = timed(lambda: torch.linalg.eigvalsh(Gm))
                ts.append(t)
            evn = np.clip(ev.cpu().numpy().astype(float), 0.0, 1.0)
            S[lib] = float(np.sum(jl.pair_entropy_from_gram(evn)))
            tr_err = abs(float(ev.sum().item()) - trace) / trace
            t_best = min(ts)
            flops = (4.0 / 3.0) * n ** 3
            floor_s = (n ** 3 / 6.0) * 8.0 / (args.bw_tbs * 1e12)
            row['libs'][lib] = dict(eigvalsh_s=ts, best_s=t_best, tflops=flops / t_best / 1e12, floor_1stage_s=floor_s,
                                    floor_ratio=t_best / floor_s, peak_gb=torch.cuda.max_memory_allocated(dev) / 1e9,
                                    S=S[lib], trace_err=tr_err, ev_min=float(evn.min()), ev_max=float(ev.max().item()))
            ok = tr_err <= 1e-10
            print(f"[j4bench n={n}] {lib}: eigvalsh {t_best:.2f} s (reps {[round(t, 2) for t in ts]}), {flops / t_best / 1e12:.2f} TFLOPS FP64, "
                  f"{t_best / floor_s:.2f} x the 1-stage bandwidth floor {floor_s:.1f} s, peak {row['libs'][lib]['peak_gb']:.1f} GB, "
                  f"S {S[lib]:.9f}, trace check {tr_err:.1e} [{'PASS' if ok else 'FAIL'}]", flush=True)
            if not ok:
                rec['failures'].append(f"trace n={n} {lib}")
            del ev
        except Exception as e:
            row['libs'][lib] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f"[j4bench n={n}] {lib}: failed ({row['libs'][lib]['error']})", flush=True)
    if len(S) == 2:
        a, b = list(S.values())
        ok = abs(a - b) <= 1e-9
        print(f"[j4bench n={n}] entropy agreement between libraries {abs(a - b):.2e} nats [{'PASS' if ok else 'FAIL'}]", flush=True)
        row['S_diff'] = abs(a - b)
        if not ok:
            rec['failures'].append(f"S n={n}")
    print(f"[j4bench n={n}] host-to-device {row['h2d_s']:.2f} s ({8 * n * (n + 16) / 1e9 / row['h2d_s']:.1f} GB/s pageable), Gram product {row['gram_s']:.2f} s "
          f"({2.0 * n * n * (n + 16) / row['gram_s'] / 1e12:.1f} TFLOPS)", flush=True)
    rec['rows'].append(row)
    del Gm, K
    torch.cuda.empty_cache()
    with open(args.json, 'w') as f:
        json.dump(rec, f, indent=1)
try:
    torch.backends.cuda.preferred_linalg_library('default')
except Exception:
    pass
print(f"[j4bench] {'GATE PASS' if not rec['failures'] else 'GATE FAIL: ' + '; '.join(rec['failures'])}; report {args.json}", flush=True)
sys.exit(0 if not rec['failures'] else 1)
