#!/usr/bin/env python
"""engine_eigh_bench_v1.py (builder analyst; the owner's request of 15 October: the GPU jobs optimized for speed and memory): the collapse
engine's dominant operation, the full eigendecomposition of one channel's clocked chain hm (complex Hermitian, 2 nsites square, banded:
on-site, the nearest-neighbour sine bond and the two-step flow term), timed per backend on the stored fields of a collapse record:
torch.linalg.eigh on the GPU with the cuSOLVER and the MAGMA backends (complex128), the same in complex64 (for scale only), scipy's
eigh (evd and evr drivers) and eig_banded on the host (--threads). Per backend: the median wall time of --reps calls (after one warm-up),
the eigenvalues' largest difference from the cuSOLVER complex128 reference, the residual max |hm U - U e| and the orthonormality
max |U^H U - 1| on 64 sampled columns, and the GPU's peak memory. Also timed: the GEMMs of one continuation (ref^H U) and of one exact
propagation (U (ph * (U^H X))). Args: --fields --record --kappa --nsites --m --gpu --threads --reps --md."""
import os
import argparse
import json
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--fields", required=True)
ap.add_argument("--record", type=int, default=-1)
ap.add_argument("--kappa", type=int, default=1)
ap.add_argument("--m", type=float, default=0.2)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--md", required=True)
a = ap.parse_args()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(a.threads)

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v5 as C5  # noqa: E402

C5.backend(a.gpu)
dev = C5.BK["dev"]
Z = np.load(a.fields, allow_pickle=True)
R = np.asarray(Z["R"], float); nsites = len(R)
N = np.asarray(Z["N"][a.record], float); v = np.asarray(Z["v"][a.record], float)
ch = C5.Chan(a.kappa, a.m, nsites, False)
hmT = ch.clocked_dense(N, v)
hm = hmT.cpu().numpy()
n = hm.shape[0]
rows = []


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def checks(e, U, H):
    idx = np.linspace(0, n - 1, 64).astype(int)
    Us = U[:, idx]
    res = float(np.max(np.abs(H @ Us - Us * e[idx][None, :])))
    orth = float(np.max(np.abs(U.conj().T @ Us - np.eye(n)[:, idx])))
    return res, orth


def gpu_case(label, lib, dtype):
    torch.backends.cuda.preferred_linalg_library(lib)
    H = hmT.to(dtype)
    torch.cuda.reset_peak_memory_stats(dev)
    e, U = torch.linalg.eigh(H); sync()
    ts = []
    for _ in range(a.reps):
        t0 = time.time(); e, U = torch.linalg.eigh(H); sync(); ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    en = e.real.double().cpu().numpy(); Un = U.to(torch.complex128).cpu().numpy()
    return label, float(np.median(ts)), en, Un, peak


ref = None
for label, lib, dt in (("torch cuSOLVER complex128", "cusolver", torch.complex128), ("torch MAGMA complex128", "magma", torch.complex128),
                       ("torch cuSOLVER complex64 (scale only)", "cusolver", torch.complex64), ("torch MAGMA complex64 (scale only)", "magma", torch.complex64)):
    try:
        lab, t, en, Un, peak = gpu_case(label, lib, dt)
    except Exception as ex:  # a backend unavailable is recorded, not fatal
        rows.append({"backend": label, "error": repr(ex)[:200]})
        print(label, "failed:", repr(ex)[:200], flush=True)
        continue
    if ref is None:
        ref = en
    res, orth = checks(en, Un, hm)
    rows.append({"backend": lab, "seconds": t, "max_de_vs_ref": float(np.max(np.abs(en - ref))), "residual": res, "orthonormality": orth, "gpu_peak_GiB": peak})
    print(rows[-1], flush=True)
torch.backends.cuda.preferred_linalg_library("default")

# the host: scipy's drivers and the banded solver (lower band: the two-step term reaches 4 rows down)
for label, fn in (("scipy eigh evd (%d threads)" % a.threads, lambda: sla.eigh(hm, driver="evd", check_finite=False)),
                  ("scipy eigh evr (%d threads)" % a.threads, lambda: sla.eigh(hm, driver="evr", check_finite=False))):
    ts = []
    for _ in range(max(1, a.reps - 1)):
        t0 = time.time(); e, U = fn(); ts.append(time.time() - t0)
    res, orth = checks(e, U, hm)
    rows.append({"backend": label, "seconds": float(np.median(ts)), "max_de_vs_ref": float(np.max(np.abs(e - ref))), "residual": res, "orthonormality": orth})
    print(rows[-1], flush=True)
bw = 4
ab = np.zeros((bw + 1, n), complex)
for k in range(bw + 1):
    ab[k, :n - k] = np.diagonal(hm, -k)
ts = []
for _ in range(max(1, a.reps - 1)):
    t0 = time.time(); e, U = sla.eig_banded(ab, lower=True, check_finite=False); ts.append(time.time() - t0)
res, orth = checks(e, U, hm)
rows.append({"backend": "scipy eig_banded (lower band 4, %d threads)" % a.threads, "seconds": float(np.median(ts)), "max_de_vs_ref": float(np.max(np.abs(e - ref))),
             "residual": res, "orthonormality": orth, "band_check": float(np.max(np.abs(np.triu(np.tril(hm, -bw - 1))))) if False else float(np.max(np.abs(np.tril(hm, -bw - 1))))})
print(rows[-1], flush=True)

# the GEMMs of one continuation and one exact propagation (complex128 on the GPU)
torch.backends.cuda.preferred_linalg_library("cusolver")
e, U = torch.linalg.eigh(hmT); sync()
neg = e.real < 0
X = U[:, neg].clone()
t0 = time.time(); W = X.conj().T @ U; sync(); t_ov = time.time() - t0
t0 = time.time(); W = X.conj().T @ U; sync(); t_ov = min(t_ov, time.time() - t0)
ph = torch.exp(-1j * e.real.to(torch.complex128) * 8.0)
t0 = time.time(); Y = U @ (ph[:, None] * (U.conj().T @ X)); sync(); t_pr = time.time() - t0
t0 = time.time(); Y = U @ (ph[:, None] * (U.conj().T @ X)); sync(); t_pr = min(t_pr, time.time() - t0)
t0 = time.time(); hm2 = ch.clocked_dense(N, v); sync(); t_cd = time.time() - t0
extra = {"n": n, "n_sea": int(neg.sum()), "continuation_overlap_gemm_s": t_ov, "propagation_two_gemms_s": t_pr, "clocked_dense_build_and_transfer_s": t_cd}
print(extra, flush=True)
L = ["## The channel eigendecomposition per backend (engine_eigh_bench_v1.py; %s, record %d, kappa %d, n = %d, m %g)" % (os.path.basename(a.fields), a.record, a.kappa, n, a.m), "",
     "| backend | seconds (median of %d) | max abs(e - e_ref) | residual max abs(hm U - U e) | orthonormality | GPU peak GiB |" % a.reps, "|---|---|---|---|---|---|"]
for r in rows:
    if "error" in r:
        L.append("| %s | failed: %s | | | | |" % (r["backend"], r["error"]))
    else:
        L.append("| %s | %.3f | %.1e | %.1e | %.1e | %s |" % (r["backend"], r["seconds"], r["max_de_vs_ref"], r["residual"], r["orthonormality"],
                 ("%.2f" % r["gpu_peak_GiB"]) if "gpu_peak_GiB" in r else "-"))
L += ["", "Other operations (complex128 on the GPU): the continuation's overlap ref^H U %.3f s; one exact propagation U (ph (U^H X)) %.3f s; building hm on the host and moving it %.3f s (n_sea %d)." % (
    t_ov, t_pr, t_cd, extra["n_sea"]), "", "- every number above is printed by the script; e_ref is the cuSOLVER complex128 result."]
open(a.md, "w").write("\n".join(L) + "\n")
json.dump({"rows": rows, "extra": extra}, open(a.md.replace(".md", ".json"), "w"), indent=1, default=float)
