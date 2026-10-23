#!/usr/bin/env python
"""unit_h3_v9c4.py (busi lane, PERF H3): byte checks of the v9c4 options A1 (band gather) and C1 (selected Gram gather) against v9c3c's own
expressions inside engine_twofield_v9c4, on the box CPU. A1: band_upper_v36 off against on for banded Hermitian squares Ks @ Ks of the walk's form
(sorted and unsorted CSR indices, explicit +0/-0 entries, repeated structures for the plan cache, a duplicate-entry CSR and a real-dtype matrix for
the fallbacks), ab bytes, u, dtype, shape, C order, and the pd_banded_v36 verdicts at a d ladder. C1: _h3_gram_columns called directly on CPU torch
tensors (the engine routes to it only for a device tensor) against _v9c_gram(A, U)[:, order]: bytes, strides (of the axes longer than 1:
numpy keeps an arbitrary stride on a length-1 axis, which no traversal reads), the C and F flags, and the bytes of the
downstream sum(axis=0) (V9cOracle.project).

Usage. python unit_h3_v9c4.py ENGINE_DIR
"""
import importlib
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

E = importlib.import_module("engine_twofield_v9c4")
fails = []; npass = 0


def check(ok, what):
    global npass
    if ok:
        npass += 1
    else:
        fails.append(what)
    print("%s %s" % ("PASS" if ok else "FAIL", what))


def herm_band(n, bw, rng, zeros=False):
    rows, cols, vals = [], [], []
    for i in range(n):
        for k in range(0, bw + 1):
            j = i + k
            if j >= n:
                break
            v = complex(rng.normal(), rng.normal()) if k else complex(rng.normal() + 4.0 * bw, 0.0)
            if zeros and k == bw and i % 7 == 0:
                v = complex(-0.0, 0.0) if i % 2 else 0j
            rows.append(i); cols.append(j); vals.append(v)
            if k:
                rows.append(j); cols.append(i); vals.append(np.conj(v))
    return sp.csr_matrix((np.array(vals, complex), (rows, cols)), shape=(n, n))


def band_off_on(A):
    E._H3["band_gather"] = False
    ab0, u0 = E.band_upper_v36(A)
    E._H3["band_gather"] = True
    ab1, u1 = E.band_upper_v36(A)
    E._H3["band_gather"] = False
    same = (u0 == u1 and ab0.dtype == ab1.dtype and ab0.shape == ab1.shape and ab0.tobytes() == ab1.tobytes()
            and ab0.flags["C_CONTIGUOUS"] == ab1.flags["C_CONTIGUOUS"])
    pd_same = True
    if same and u0 >= 0:
        for d in (2.0, 1.0, 0.5, 0.25, 0.125, 1e-3):
            pd_same = pd_same and (E.pd_banded_v36(ab0, d) == E.pd_banded_v36(ab1, d))
    return same, pd_same, u0


rng = np.random.default_rng(20261020)
for n, bw in ((64, 2), (256, 4), (512, 8), (2048, 3)):
    for zeros in (False, True):
        K0 = herm_band(n, bw, rng, zeros); K1 = herm_band(n, bw, rng, zeros); dK = (K1 - K0).tocsr()
        for t1 in (0.0, 0.25, 0.71):
            Ks = (K0 + t1 * dK).tocsr(); A = Ks @ Ks
            same, pd_same, u = band_off_on(A)
            check(same and pd_same, "A1 n %d band %d zeros %s t1 %.2f (u %d, nnz %d): ab bytes equal %s, pd verdicts equal %s" % (n, bw, zeros, t1, u, A.nnz, same, pd_same))
# unsorted indices
A = (herm_band(300, 5, rng) @ herm_band(300, 5, rng).conj().T).tocsr()
perm_rows = []
indices = A.indices.copy(); data = A.data.copy()
for i in range(A.shape[0]):
    a, b = A.indptr[i], A.indptr[i + 1]
    p = rng.permutation(b - a)
    indices[a:b] = A.indices[a:b][p]; data[a:b] = A.data[a:b][p]
Au = sp.csr_matrix((data, indices, A.indptr.copy()), shape=A.shape)
Au.has_sorted_indices = False
same, pd_same, u = band_off_on(Au)
check(same, "A1 unsorted CSR indices (u %d): ab bytes equal %s" % (u, same))
# duplicates: a non-canonical CSR with a repeated upper entry (the fallback)
Ad = sp.csr_matrix((np.array([1 + 0j, 2 + 1j, 3 - 1j, 2 - 1j, 5 + 0j]), np.array([0, 1, 1, 0, 1]), np.array([0, 3, 5])), shape=(2, 2))
E._H3_STATS.clear()
same, pd_same, u = band_off_on(Ad)
check(same and E._H3_STATS.get("band_fallback", 0) == 1, "A1 duplicate upper entries: the fallback taken (%s), ab bytes equal %s" % (E._H3_STATS.get("band_fallback"), same))
# real dtype: the fallback
Ar = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]]))
E._H3_STATS.clear()
same, pd_same, u = band_off_on(Ar)
check(same and E._H3_STATS.get("band_fallback", 0) == 1, "A1 float64 matrix: the fallback taken, ab bytes equal %s" % same)
# plan cache: the same structure twice, then another structure, then the first again
E._H3_STATS.clear()
K = herm_band(400, 4, rng); A1m = (K @ K).tocsr(); A2m = (A1m * (1.0 + 1e-3j)).tocsr(); K2 = herm_band(400, 6, rng); B1m = (K2 @ K2).tocsr()
ok = True
for M in (A1m, A2m, B1m, A1m):
    s_, p_, _ = band_off_on(M); ok = ok and s_ and p_
check(ok and E._H3_STATS.get("band_plan_hit") == 1 and E._H3_STATS.get("band_plan_rebuild") == 3,
      "A1 plan cache: bytes equal over 4 calls, hits %s rebuilds %s (expected 1, 3)" % (E._H3_STATS.get("band_plan_hit"), E._H3_STATS.get("band_plan_rebuild")))

# C1
import torch  # noqa: E402
for m, n, k in ((64, 128, 40), (512, 1024, 300), (2048, 512, 256), (100, 100, 1)):
    for a_np in (False, True):
        A = torch.randn(m, n // 4 + 1, dtype=torch.complex128)
        U = torch.randn(m, n, dtype=torch.complex128)
        order = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
        rng.shuffle(order)
        Ain = A.numpy() if a_np else A
        W0 = E._v9c_gram(Ain, U)[:, order]
        W1 = E._h3_gram_columns(Ain, U, order)
        same = (W0.dtype == W1.dtype and W0.shape == W1.shape and [s for s, d in zip(W0.strides, W0.shape) if d > 1] == [s for s, d in zip(W1.strides, W1.shape) if d > 1] and W0.flags["F_CONTIGUOUS"] == W1.flags["F_CONTIGUOUS"]
                and W0.flags["C_CONTIGUOUS"] == W1.flags["C_CONTIGUOUS"] and np.asarray(W0).tobytes(order="A") == np.asarray(W1).tobytes(order="A"))
        s0 = W0.sum(axis=0); s1 = W1.sum(axis=0)
        check(same and s0.tobytes() == s1.tobytes(), "C1 m %d n %d k %d A numpy %s: bytes/strides equal %s (strides %s vs %s), sum(axis=0) bytes equal %s" % (
            m, n, k, a_np, same, W0.strides, W1.strides, s0.tobytes() == s1.tobytes()))
# C1 routing: a CPU tensor never takes the gather path
E._H3["weight_gather"] = True; E._H3_STATS.clear()
P = E.V9cPlus(torch.randn(50, 20, dtype=torch.complex128), np.arange(0, 20, 2))
_ = E._v9c_weights(torch.randn(50, 5, dtype=torch.complex128), P)
check(E._H3_STATS.get("gram_gathers", 0) == 0, "C1 routing: a CPU tensor keeps v9c3c's path (gathers %s)" % E._H3_STATS.get("gram_gathers", 0))
E._H3["weight_gather"] = False
print("unit_h3_v9c4: %d PASS, %d FAIL%s" % (npass, len(fails), (": " + "; ".join(fails)) if fails else ""))
sys.exit(1 if fails else 0)
