#!/usr/bin/env python
"""make_timing_init_v6s.py (busi lane, #309 item 3): a TIMING-ONLY stand-in for the lean mode's --init-cache: an orthonormal n x n/2 block (QR of a
seeded Gaussian block on the device), keyed to h_+(T_0)'s md5 so that page_history_v6s.py --lean-interior loads it. Its readouts are meaningless;
only the run's timings and device memory are read. Usage: python make_timing_init_v6s.py REC FIELDS NSITES OUT.pt"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import page_history_v6s as V6S, page_interior_v6s as IR, page_w1_sector_v1d as W1S, page_ops_v6 as OPS
n = int(sys.argv[3]); dev = torch.device("cuda:0")
H = V6S.History(sys.argv[1], sys.argv[2], n, None, "applied")
ops = V6S.ChannelOps(1, H.R, torch.device("cpu"))
hp = W1S.sector_plus_csr(ops.h_csr(*H.fields(0)))
torch.manual_seed(12345)
X = torch.randn(n, n // 2, dtype=torch.float64, device=dev).to(OPS.CD)
Q, _ = torch.linalg.qr(X); del X
torch.save({"Phi_plus": Q.cpu(), "md5_h_plus": IR._hp_md5(hp), "n": n, "info": {"TIMING_ONLY": "an orthonormal stand-in, not the vacuum"}, "kappa": 1, "code": "make_timing_init_v6s.py"}, sys.argv[4])
print("written", sys.argv[4], Q.shape)
