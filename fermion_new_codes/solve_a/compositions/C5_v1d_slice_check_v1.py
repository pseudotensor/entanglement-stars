"""C5_v1d_slice_check_v1.py: the W2 T = 0 slice the delivered comparator read
(ladders/r3/slices/..._v36f_T0000.00.npz) is no longer on the box; this checks candidate T = 0 slices
of the same preparation against the delivered run's saved per-site count (profile_*.npz of
w4a/out/default) and prints each candidate's per-column barrier L2_c (item 1's T operator, open ends).
Read-only on every input; prints only."""
import sys, numpy as np

PROF = "<internal>/profile_orbitals_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_T0000.00.npz"


def L2_columns(R, kappa, d):
    n = len(R)
    A = kappa / R
    out = []
    for c in range(d.shape[1]):
        u = d[0::2, c]; l = d[1::2, c]
        dAp = np.zeros(n); dAm = np.zeros(n)
        dAp[:-1] = A[1:] - A[:-1]          # A_{j+1} - A_j
        dAm[1:] = A[:-1] - A[1:]           # A_{j-1} - A_j
        def br(x):
            xp = np.zeros(n, complex); xm = np.zeros(n, complex)
            xp[:-1] = x[1:]; xm[1:] = x[:-1]
            return 0.5 * (dAp * xp - dAm * xm)
        Tu = (kappa / R) ** 2 * u + br(u)
        Tl = (kappa / R) ** 2 * l - br(l)
        t = np.real(np.conj(u) * Tu + np.conj(l) * Tl)
        w = np.abs(u) ** 2 + np.abs(l) ** 2
        out.append(float(np.sum(R ** 2 * t) / np.sum(w)))
    return out


P = np.load(PROF)
n_ref = P["n_site"]; R_ref = P["R"]
for path in sys.argv[1:]:
    Z = np.load(path)
    R = np.asarray(Z["R"], float)
    n_site = np.zeros(len(R)); Ls = []
    for k in range(1, 6):
        if "dust_%d" % k not in Z.files:
            continue
        d = np.asarray(Z["dust_%d" % k], complex); f = np.asarray(Z["f_%d" % k], float)
        p = np.abs(d) ** 2
        n_site += float(np.atleast_1d(Z["weights"])[list(np.atleast_1d(Z["kappas"])).index(k)]) * ((p[0::2] + p[1::2]) @ f)
        Ls.append((k, d.shape[1], [round(x, 3) for x in L2_columns(R, k, d)]))
    print(path)
    print("  R same: %s; count %.6f; max |n - n_ref| %.3e" % (np.array_equal(R, R_ref), n_site.sum(), np.max(np.abs(n_site - n_ref))))
    for k, nc, L in Ls:
        print("  kappa %d: %d columns, L2 %s" % (k, nc, L))
