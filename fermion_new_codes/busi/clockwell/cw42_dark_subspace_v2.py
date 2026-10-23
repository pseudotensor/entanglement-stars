"""cw42 (v2 of cw41: the rank threshold is ABSOLUTE, 1e-10 x the operator scale (||T||_2 for ker T, ||H_H||_2 for the
invariance steps); cw41 used 1e-10 x the largest singular value of each step matrix, which wrongly discarded an
already-invariant subspace whose step matrix is ~1e-16 and printed dim W = 0). cw41: reviewer A A static readout (#492): the exactly dark subspace W of the heavy N = 2 sector, per converter form at
g = 1: W = the unobservable subspace of (H_H, T), T = P_LL V P_HH (heavy-to-light converter block), H_H = P_HH H0 P_HH,
i.e. the largest H_H-invariant subspace of ker T (= intersection over j of ker(T H_H^j)). Computed by the subspace
recursion W_0 = ker T, W_{k+1} = {w in W_k : H_H w in W_k} with orthonormal bases and an SVD rank threshold
(singular values below 1e-10 x the operator scale count as zero; the smallest kept and largest dropped singular values are
printed as the gap). Readouts: dim W beside n(2n-1) (the symmetry-protected heavy |Q| = 2, R-odd sector), ||P_W B||,
||P_W B'|| (B' = (|03>_H + |12>_H)/sqrt 2), ||P_W D||, and ||[Gamma(r), H]||_max on the full H at g = 1.
Inputs: n = 3 from icloud's SAVED g = 1 Hamiltonians (both forms; read-only) and from busi's builder; n = 5 from busi's
builder (cw32). Output: <internal>/cw42_dark_subspace.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
ROOT = '<internal>/solve_b_bulk'; EXT = '<internal>/cw05_extract.json'
TOL = 1e-10

def nullspace(A, scale, tol=TOL):
    if A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=complex), None, None
    U, s, Vh = np.linalg.svd(A)
    r = int(np.sum(s > tol * scale))
    kept_min = float(s[r - 1]) if r > 0 else None
    dropped_max = float(s[r]) if r < len(s) else 0.0
    return Vh[r:].conj().T, kept_min, dropped_max

def dark_subspace(H1, H0, B):
    sp = np.array([(a % 8) // 4 + (b % 8) // 4 for a, b in B])
    HH = np.where(sp == 0)[0]; LL = np.where(sp == 2)[0]
    V = H1 - H0
    T = V[np.ix_(LL, HH)]; HHm = H0[np.ix_(HH, HH)]
    sT = np.linalg.norm(T, 2); sH = np.linalg.norm(HHm, 2)
    Z, kmin, dmax = nullspace(T, sT)
    gaps = [(kmin, dmax)]
    for _ in range(len(HH) + 1):
        if Z.shape[1] == 0:
            break
        Pperp = np.eye(len(HH)) - Z @ Z.conj().T
        C, kmin, dmax = nullspace(Pperp @ HHm @ Z, sH)
        gaps.append((kmin, dmax))
        Znew = Z @ C
        if Znew.shape[1] == Z.shape[1]:
            Z = Znew; break
        Z, _ = np.linalg.qr(Znew) if Znew.shape[1] else (Znew, None)
    return Z, HH, gaps

def readout(H1, H0, B, n, label):
    idx = {p: i for i, p in enumerate(B)}
    Z, HH, gaps = dark_subspace(H1, H0, B)
    pos = {i: k for k, i in enumerate(HH)}
    def vec(comps):
        v = np.zeros(len(HH), dtype=complex)
        for p, c in comps.items():
            v[pos[idx[p]]] = c
        return v / np.linalg.norm(v)
    Bv = vec({(0, 1): 1}); Bp = vec({(0, 3): 1, (1, 2): 1}); Dv = vec({(0, 3): 1, (1, 2): -1})
    proj = lambda v: float(np.linalg.norm(Z.conj().T @ v)) if Z.shape[1] else 0.0
    rperm = {0: 1, 1: 0, 2: 3, 3: 2}
    Rm = np.zeros((len(B), len(B)))
    for (a, b), i in idx.items():
        a2 = (a // 4) * 4 + rperm[a % 4]; b2 = (b // 4) * 4 + rperm[b % 4]
        Rm[idx[(min(a2, b2), max(a2, b2))], i] = 1.0 if a2 < b2 else -1.0
    comm = float(np.abs(Rm @ H1 - H1 @ Rm).max()); comm0 = float(np.abs(Rm @ H0 - H0 @ Rm).max())
    out = dict(label=label, n=n, heavy_dim=len(HH), dim_W=int(Z.shape[1]), protected_dim=n * (2 * n - 1), P_W_B=proj(Bv), P_W_Bprime=proj(Bp), P_W_D=proj(Dv),
               comm_R_H_max=comm, comm_R_H0_max=comm0, rank_gaps=[[a, b] for a, b in gaps])
    print(f"{label}: dim W {out['dim_W']} (n(2n-1) = {out['protected_dim']}, heavy {out['heavy_dim']}); |P_W B| {out['P_W_B']:.3e} |P_W B'| {out['P_W_Bprime']:.3e} |P_W D| {out['P_W_D']:.6f}; max|[R,H]| {comm:.3e} max|[R,H0]| {comm0:.3e}; gaps {gaps[:3]}", flush=True)
    return out

def main():
    ext = json.load(open(EXT)); res = []
    for arm, form in (('baseline', 'onsite'), ('filtered', 'filtered')):
        P = ext['physical'][f'{arm}/even_n3_g1_v1']; d = f"{ROOT}/{P['top']}/{P['name']}"
        Hs = None
        for line in open(P['report']):
            if line.startswith('{"kind":"numerical_action"'):
                r = json.loads(line)['data']
                if r.get('kind') == 'operator':
                    z = np.load(os.path.join(d, r['hamiltonian']['array'])); Hs = z[z.files[0]]; break
        P0 = ext['physical'][f'{arm}/even_n3_g0_v1']; d0 = f"{ROOT}/{P0['top']}/{P0['name']}"; H0s = None
        for line in open(P0['report']):
            if line.startswith('{"kind":"numerical_action"'):
                r = json.loads(line)['data']
                if r.get('kind') == 'operator':
                    z = np.load(os.path.join(d0, r['hamiltonian']['array'])); H0s = z[z.files[0]]; break
        B3 = M.basis(3)
        res.append(dict(source='icloud saved n = 3 H (g = 1 and g = 0 arrays)', form=form, **readout(Hs, H0s, B3, 3, f'saved n=3 {form}')))
        for n in (3, 5):
            H1, B = M.assemble(n, 1.0, form); H0, _ = M.assemble(n, 0.0, form)
            res.append(dict(source='busi builder cw32', form=form, **readout(H1, H0, B, n, f'busi n={n} {form}')))
    json.dump(res, open('<internal>/cw42_dark_subspace.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
