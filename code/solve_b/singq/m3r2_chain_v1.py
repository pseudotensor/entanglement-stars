# m3r2_chain_v1.py -- mark 3 root-layer test, route 2 (route2 lane).
# Chain operators of manifest v2 Sec. 1 (v3 governing where it differs), in band storage.
#   heavy / generic 2-spinor chain: basis i = 2(n-1)+s, sites n = 1..ns, s = 0 (upper), 1 (lower)
#   light (m = 0) chain: exact unitary symmetry Xi = (-1)^n sigma_2 and antiunitary K (h purely
#   imaginary); sector + basis |n, chi_n>, chi_n = |+y> (n even), |-y> (n odd); the sector -
#   is the K-image, Delta_- = -(Delta_+)^*, so every purely imaginary Xi-even one-body trace
#   is 2 x the sector-+ trace (verified numerically in m3r2_test_v1.py).
# Band storage: B[d, i] = h[i, i+d], d = 0..bw (upper), lower part by Hermiticity.
import numpy as np
import numba as nb


def heavy_h0_Q_bands(ns, kch, m, a=1.0):
    """Bands of h_{0,k} = m s3 + s1 p_r + s2 k/r_n and of Q_r (x) 1 on the 2-spinor chain."""
    dim = 2 * ns
    r = a * np.arange(1, ns + 1, dtype=float)
    H0 = np.zeros((5, dim), complex)
    Q = np.zeros((5, dim), complex)
    H0[0, 0::2] = m
    H0[0, 1::2] = -m
    H0[1, 0::2] = -1j * kch / r                      # s2 (0,1) element = -i, times k/r_n
    H0[1, 1:dim - 2:2] = -1j / (2 * a)               # (2n+1, 2(n+1)): s1(1,0)=1 times p_{n,n+1} = -i/2a
    H0[3, 0:dim - 3:2] = -1j / (2 * a)               # (2n, 2(n+1)+1): s1(0,1)=1 times -i/2a
    Q[4, 0:dim - 4:2] = -1j / (4 * a)                # Q_{n,n+2} = 1/(4ia) = -i/4a, both components
    Q[4, 1:dim - 4:2] = -1j / (4 * a)
    site_of = (np.arange(dim) // 2).astype(np.int64)
    return H0, Q, site_of, r


def light_sector_h0_Q_bands(ns, kch, a=1.0):
    """Sector-+ scalar chain (m = 0): on-site (-1)^n k/r_n, NN -(-1)^n/(2a), NNN Q = -i/(4a)."""
    n = np.arange(1, ns + 1)
    r = a * n.astype(float)
    sgn = np.where(n % 2 == 0, 1.0, -1.0)
    H0 = np.zeros((3, ns), complex)
    Q = np.zeros((3, ns), complex)
    H0[0, :] = sgn * kch / r
    H0[1, :ns - 1] = -sgn[:ns - 1] / (2 * a)
    Q[2, :ns - 2] = -1j / (4 * a)
    site_of = np.arange(ns).astype(np.int64)
    return H0, Q, site_of, r


@nb.njit(cache=True)
def weighted_band(H0, Q, site_of, Nbar, vbar):
    """h[N,v] = 1/2{N,h0} + 1/2{v,Q}: element (i,j) gets 1/2(N_si+N_sj) h0_ij + 1/2(v_si+v_sj) Q_ij.
    Also used for B_{f,Q} = sum_n (beta^rho_n e_n + beta^J_n p_n) with (Nbar, vbar) -> (beta^rho, beta^J)."""
    bw1, dim = H0.shape
    out = np.zeros((bw1, dim), np.complex128)
    for d in range(bw1):
        for i in range(dim - d):
            si = site_of[i]
            sj = site_of[i + d]
            out[d, i] = 0.5 * (Nbar[si] + Nbar[sj]) * H0[d, i] + 0.5 * (vbar[si] + vbar[sj]) * Q[d, i]
    return out


def band_to_dense(B):
    bw1, dim = B.shape
    A = np.zeros((dim, dim), complex)
    for d in range(bw1):
        idx = np.arange(dim - d)
        A[idx, idx + d] = B[d, :dim - d]
        if d > 0:
            A[idx + d, idx] = np.conj(B[d, :dim - d])
    return A


def dense_to_band(A, bw):
    dim = A.shape[0]
    B = np.zeros((bw + 1, dim), complex)
    for d in range(bw + 1):
        idx = np.arange(dim - d)
        B[d, :dim - d] = A[idx, idx + d]
    return B


@nb.njit(cache=True)
def alloc_trace(DB, AB, site_of, nsites):
    """out[n] = Tr(X a_n), a_n = 1/2{Pi_n, A}, X and A Hermitian given by bands (X band may be wider).
    Element pair (i,i+d): c = X_ii A_ii (d=0) or 2 Re(conj(X_ij) A_ij); half to site(i), half to site(j)."""
    bwA = AB.shape[0] - 1
    dim = AB.shape[1]
    out = np.zeros(nsites)
    for d in range(bwA + 1):
        for i in range(dim - d):
            Aij = AB[d, i]
            if Aij == 0:
                continue
            Xij = DB[d, i]
            if d == 0:
                c = (Xij * Aij).real
            else:
                c = 2.0 * (Xij.real * Aij.real + Xij.imag * Aij.imag)
            out[site_of[i]] += 0.5 * c
            out[site_of[i + d]] += 0.5 * c
    return out


@nb.njit(parallel=True, cache=True)
def band_from_orbitals(Psi, bw):
    """CB[d,i] = C[i,i+d] = sum_k Psi[i,k] conj(Psi[i+d,k]) (C = Psi Psi^dagger)."""
    dim, norb = Psi.shape
    CB = np.zeros((bw + 1, dim), np.complex128)
    for i in nb.prange(dim):
        for d in range(bw + 1):
            if i + d < dim:
                s = 0j
                for k in range(norb):
                    s += Psi[i, k] * np.conj(Psi[i + d, k])
                CB[d, i] = s
    return CB


@nb.njit(cache=True)
def band_matmul(A, B, bwout):
    """Band (offsets 0..bwout, upper) of the product of two Hermitian banded matrices A B
    (the product is not Hermitian; we return the full band of offsets -bwout..bwout as
    out[bwout + d, i] = (AB)[i, i+d])."""
    bwa = A.shape[0] - 1
    bwb = B.shape[0] - 1
    dim = A.shape[1]
    out = np.zeros((2 * bwout + 1, dim), np.complex128)
    for i in range(dim):
        for k in range(max(0, i - bwa), min(dim, i + bwa + 1)):
            if k >= i:
                aik = A[k - i, i]
            else:
                aik = np.conj(A[i - k, k])
            if aik == 0:
                continue
            for j in range(max(0, k - bwb, i - bwout), min(dim, k + bwb + 1, i + bwout + 1)):
                if j >= k:
                    bkj = B[j - k, k]
                else:
                    bkj = np.conj(B[k - j, j])
                out[bwout + j - i, i] += aik * bkj
    return out


def comm_rate_band(HB, CB, bwout):
    """Upper band (0..bwout) of -i[h, C] (Hermitian) from bands of h and C (C band must reach
    bwout + bw_h)."""
    hc = band_matmul(HB, CB, bwout)       # (hC)[i,i+d]
    ch = band_matmul(CB, HB, bwout)       # (Ch)[i,i+d]
    full = -1j * (hc - ch)
    return np.ascontiguousarray(full[bwout:, :])


def gershgorin(HB):
    bw1, dim = HB.shape
    rad = np.zeros(dim)
    for d in range(1, bw1):
        a = np.abs(HB[d, :dim - d])
        rad[:dim - d] += a
        rad[d:] += a
    diag = HB[0].real
    return float(np.min(diag - rad)), float(np.max(diag + rad))


def band_opnorm_bound(HB):
    lo, hi = gershgorin(HB)
    return max(abs(lo), abs(hi))
