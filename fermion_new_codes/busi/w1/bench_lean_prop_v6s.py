#!/usr/bin/env python
"""bench_lean_prop_v6s.py (busi lane, #309 item 3): the time of one production sub-step (dt = 1, cheb tol 1e-16) on an n x n/2 column block on
the GPU, page_history's propagate (OPS.Banded) against the lean mode's propagate_lean (BandedLean, in place), on B2's h_+ at T 0; and one held
step of 64 units with the lean propagator. Peak device memory printed. Usage: python bench_lean_prop_v6s.py REC FIELDS n1,n2,..."""
import sys, os, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import page_history_v6s as V6S, page_interior_v6s as IR, page_w1_sector_v1d as W1S, page_ops_v6 as OPS
dev = torch.device("cuda:0")
for n in [int(x) for x in sys.argv[3].split(",")]:
    H = V6S.History(sys.argv[1], sys.argv[2], n, None, "applied")
    ops = V6S.ChannelOps(1, H.R, dev)
    hp = W1S.sector_plus_csr(ops.h_csr(*H.fields(0)))
    torch.manual_seed(0)
    X = torch.randn(n, n // 2, dtype=torch.float64, device=dev).to(OPS.CD); X /= n ** 0.5
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    res = {}
    if n <= 16384:
        Y = X.clone(); torch.cuda.synchronize(); t = time.time()
        Y, K, _ = V6S.propagate(OPS.Banded(hp, dev), Y, 1.0, 1e-16, 2048); torch.cuda.synchronize(); res["prod_dt1"] = (time.time() - t, K)
        del Y
    Y = X.clone(); torch.cuda.synchronize(); t = time.time()
    Y, K, _ = IR.propagate_lean(IR.BandedLean(hp, dev), Y, 1.0, 1e-16, 2048); torch.cuda.synchronize(); res["lean_dt1"] = (time.time() - t, K)
    t = time.time()
    Y, K, _ = IR.propagate_lean(IR.BandedLean(hp, dev), Y, 64.0, 1e-16, 2048); torch.cuda.synchronize(); res["lean_dt64"] = (time.time() - t, K)
    pk = torch.cuda.max_memory_allocated() / 2 ** 30
    print("n %d (block %d x %d, %.2f GiB): %s; device peak %.2f GiB" % (n, n, n // 2, X.numel() * 16 / 2 ** 30,
          ", ".join("%s %.2f s (K %d)" % (k_, v_[0], v_[1]) for k_, v_ in res.items()), pk), flush=True)
    del X, Y; torch.cuda.empty_cache()
