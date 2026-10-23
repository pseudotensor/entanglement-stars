"""cw52: #495 (3): reviewer A's check of the extra dark directions, printed as a readout (not assumed).
Pi = (-1)^x beta per mode (x = site, beta = diag(1, 1, -1, -1) on the components); Pi_total of a two-particle basis state
is the sum of the two single-particle values, in {0, +-2}. Prints max |[Pi, h]| (one-body, both species) and
max |[Pi_total, H0]| (two-particle), then dim(W intersect {Pi_total = +-2}) where W is the exactly dark heavy N = 2 subspace
of cw42 (largest H_H-invariant subspace of ker T at g = 1), counted as the eigenvalues of Z^dagger P Z within 1e-9 of 1
(Z an orthonormal basis of W, P the projector on Pi_total = +-2), also dim(W intersect {Pi_total = 0}) and the overlap with
the symmetry-protected sector (R-odd, |Q| = 2). Expected (reviewer A): 60 on-site and 20 filtered at n = 5; 18 and 6 at n = 3.
Sources: icloud's saved n = 3 g = 1 and g = 0 Hamiltonians (read-only) and busi's builder at n = 3, 5.
Output: <internal>/cw52_pi.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import cw42_dark_subspace_v2 as DS
ROOT = '<internal>/solve_b_bulk'; EXT = '<internal>/cw05_extract.json'
OUT = '<internal>/pi_v1'; os.makedirs(OUT, exist_ok=True)
betac = (1, 1, -1, -1); qc = (1, -1, -1, 1); rperm = {0: 1, 1: 0, 2: 3, 3: 2}

def pi_mode(m):
    return (-1) ** (m // 8) * betac[m % 4]

def count_in(Z, mask):
    if Z.shape[1] == 0:
        return 0
    Zs = Z[mask, :]
    ev = np.linalg.eigvalsh(Zs.conj().T @ Zs)
    return int(np.sum(ev > 1 - 1e-9))

def analyse(H1, H0, B, n, label):
    h = M.one_body(n)
    Pi1 = np.diag([pi_mode(m) for m in range(8 * n)])
    comm1 = float(np.abs(Pi1 @ h - h @ Pi1).max())
    pit = np.array([pi_mode(a) + pi_mode(b) for a, b in B])
    comm2 = float(np.abs(np.diag(pit) @ H0 - H0 @ np.diag(pit)).max())
    comm2V = float(np.abs(np.diag(pit) @ H1 - H1 @ np.diag(pit)).max())
    Z, HH, gaps = DS.dark_subspace(H1, H0, B)
    Bh = [B[i] for i in HH]; hidx = {p: k for k, p in enumerate(Bh)}
    piH = np.array([pi_mode(a) + pi_mode(b) for a, b in Bh])
    QH = np.array([qc[a % 4] + qc[b % 4] for a, b in Bh])
    Rm = np.zeros((len(Bh), len(Bh)))
    for (a, b), i in hidx.items():
        a2 = (a // 4) * 4 + rperm[a % 4]; b2 = (b // 4) * 4 + rperm[b % 4]
        Rm[hidx[(min(a2, b2), max(a2, b2))], i] = 1.0 if a2 < b2 else -1.0
    sel = np.abs(QH) == 2
    wR, UR = np.linalg.eigh(Rm[np.ix_(sel, sel)])
    Sprot = np.zeros((len(Bh), int((wR < 0).sum())), dtype=complex); Sprot[sel, :] = UR[:, wR < 0]
    ov = np.linalg.eigvalsh((Z.conj().T @ Sprot) @ (Sprot.conj().T @ Z)) if Z.shape[1] else np.array([])
    out = dict(label=label, n=n, max_comm_Pi_h=comm1, max_comm_Pitotal_H0=comm2, max_comm_Pitotal_H_g1=comm2V, dim_W=int(Z.shape[1]),
               dim_W_Pi_pm2=count_in(Z, np.abs(piH) == 2), dim_W_Pi_0=count_in(Z, piH == 0), dim_W_cap_protected=int(np.sum(ov > 1 - 1e-9)),
               protected_dim=int(Sprot.shape[1]), heavy_dim=len(HH))
    out['dim_W_minus_protected_minus_Pi_pm2'] = out['dim_W'] - out['dim_W_cap_protected'] - out['dim_W_Pi_pm2']
    print(label, {k: v for k, v in out.items() if k != 'label'}, flush=True)
    return out

def main():
    ext = json.load(open(EXT)); res = []
    for arm, form in (('baseline', 'onsite'), ('filtered', 'filtered')):
        Hs = {}
        for gs in ('1', '0'):
            P = ext['physical'][f'{arm}/even_n3_g{gs}_v1']; d = f"{ROOT}/{P['top']}/{P['name']}"
            for line in open(P['report']):
                if line.startswith('{"kind":"numerical_action"'):
                    r = json.loads(line)['data']
                    if r.get('kind') == 'operator':
                        z = np.load(os.path.join(d, r['hamiltonian']['array'])); Hs[gs] = z[z.files[0]]; break
        res.append(dict(source='icloud saved n = 3', form=form, **analyse(Hs['1'], Hs['0'], M.basis(3), 3, f'saved n=3 {form}')))
        for n in (3, 5):
            H1, B = M.assemble(n, 1.0, form); H0, _ = M.assemble(n, 0.0, form)
            res.append(dict(source='busi builder', form=form, **analyse(H1, H0, B, n, f'busi n={n} {form}')))
    json.dump(res, open(f'{OUT}/cw52_pi.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
