#!/usr/bin/env python
"""eigh_cpu_timing_v1.py (solve builder 5; REVIEWA_gpu_economics_v1 sec. 1, the measurement that replaces assumption 1): numpy.linalg.eigh on
one stored n = 4096 Hamiltonian, on the CPU, at the thread count set by the caller's environment (OMP/MKL/OPENBLAS_NUM_THREADS). The spectra
npz files hold eigenvalues and masks only, so the matrix is the engine's own clocked operator h_m = (1/2){N, h} + (1/2){v, Q}
(engine_collapse_v5.Chan, kappa 1, m 0.2, 2048 sites) at the stored applied fields of B's record T/r_s 40 (C3ext19 fields npz); its
eigenvalues are checked against the stored spectra npz's e_1 at that record. Repeats 3 times; prints each wall, the BLAS, the check.
Usage: OMP_NUM_THREADS=k ... python eigh_cpu_timing_v1.py <fields.npz> <spectra.npz> <T/r_s> <out.json>"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "<internal>/code")
import engine_collapse_v5 as C5

fz, sz, Tq, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
C5.backend(None)
F = np.load(fz, allow_pickle=True); S = np.load(sz, allow_pickle=True)
rs = 64.0
k = int(np.argmin(np.abs(F["T"] / rs - Tq))); ks = int(np.argmin(np.abs(S["T"] / rs - Tq)))
N = F["N_applied"][k] if "N_applied" in F.files else F["N"][k]; v = F["v_applied"][k] if "v_applied" in F.files else F["v"][k]
ch = C5.Chan(1, 0.2, 2048, False)
H = np.asarray(ch.clocked_dense(N, v)); H = H.cpu().numpy() if hasattr(H, "cpu") else H
H = np.asarray(H, dtype=np.complex128); H = 0.5 * (H + H.conj().T)
walls = []
for i in range(3):
    t0 = time.time(); e, U = np.linalg.eigh(H); walls.append(time.time() - t0)
es = np.sort(np.asarray(S["e_1"][ks], float)) if "e_1" in S.files else None
chk = float(np.max(np.abs(np.sort(e) - es))) if es is not None and es.shape == e.shape else None
try:
    import threadpoolctl; blas = [(d.get("internal_api"), d.get("num_threads")) for d in threadpoolctl.threadpool_info()]
except Exception:
    blas = None
r = {"n": int(H.shape[0]), "T_over_rs_fields": float(F["T"][k] / rs), "T_over_rs_spectra": float(S["T"][ks] / rs), "threads_env": os.environ.get("OMP_NUM_THREADS"),
     "walls_s": walls, "max_abs_eig_diff_vs_stored": chk, "blas": blas}
print(json.dumps(r)); json.dump(r, open(out, "w"), indent=1)
