#!/usr/bin/env python
"""lowdin_init_cache_v6s.py (busi lane, #309 item 3): an --init-cache of page_interior_v6s re-orthonormalised by Loewdin, Phi <- Phi (Phi^H Phi)^{-1/2}
(the closest orthonormal basis of the same span; the projector Phi Phi^H becomes the exact projector onto that span). MRRR's eigenvectors at n 32768
are orthonormal to about 1e-12 per level, which puts the trace-norm impurity at about 1.6e-8 (16384 levels); this brings it to the roundoff floor.
The keys (md5_h_plus, n, ...) are kept, so the engine loads it for the same h_+. Usage: lowdin_init_cache_v6s.py IN OUT [threads]"""
import sys
import time
import torch

src, dst = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 16)
t0 = time.time()
st = torch.load(src, map_location="cpu", weights_only=False)
P = st["Phi_plus"]
S = P.conj().T @ P
d0 = float((S - torch.eye(S.shape[0], dtype=S.dtype)).abs().max())
w, U = torch.linalg.eigh(S); del S
print("Gram eigenvalues in [%.17f, %.17f]; max|Phi^H Phi - I| before %.3e (%.0f s)" % (float(w.min()), float(w.max()), d0, time.time() - t0), flush=True)
Q = P @ ((U * w.rsqrt().to(U.dtype)) @ U.conj().T); del P, U
S2 = Q.conj().T @ Q
d1 = float((S2 - torch.eye(S2.shape[0], dtype=S2.dtype)).abs().max()); del S2
print("max|Phi^H Phi - I| after %.3e (%.0f s)" % (d1, time.time() - t0), flush=True)
st["Phi_plus"] = Q.contiguous()
st["info"] = dict(st.get("info") or {}, lowdin={"source": src, "max_dev_before": d0, "max_dev_after": d1, "gram_min": float(w.min()), "gram_max": float(w.max())})
torch.save(st, dst + ".tmp")
import os
os.replace(dst + ".tmp", dst)
print("written %s (%.0f s)" % (dst, time.time() - t0))
