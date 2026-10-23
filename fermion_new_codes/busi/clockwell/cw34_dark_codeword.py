"""cw34: is the dark codeword a symmetry of the vertex? (#491 A(c)), busi, own builder cw32.
Test: D = (|03>_H - |12>_H)/sqrt 2 at site 0. If V H0^j D = 0 for every j (Krylov space of D under the free H0
is annihilated by the converter V), then exp(-iHt) D = exp(-iH0 t) D exactly: D never acquires light, at every
clock, for that n, profile and form. Checked with an orthonormal Krylov basis (full Arnoldi with
reorthogonalization) for n = 2, 3, 5, 7, both forms, the clock-well profile and random positive profiles, and
the heavy mass 3/5 and a random mass. Also the companion kernel direction (|01>_H + |23>_H)/sqrt 2 and B.
Prints max_j ||V q_j|| / ||V||. This is a finite-size numerical test, not a proof.
Output: <internal>/cw34_dark.json"""
import json, sys, time
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M

def krylov_leak(H0, V, v, tol=1e-12):
    Q = []; w = v / np.linalg.norm(v); worst = 0.0
    nV = np.linalg.norm(V, 2)
    for _ in range(H0.shape[0]):
        for q in Q:
            w = w - q * np.vdot(q, w)
        for q in Q:
            w = w - q * np.vdot(q, w)
        nw = np.linalg.norm(w)
        if nw < tol:
            break
        w = w / nw; Q.append(w)
        worst = max(worst, np.linalg.norm(V @ w) / nV)
        w = H0 @ w
    return worst, len(Q)

rng = np.random.default_rng(11)
res = []
for n in (2, 3, 5, 7):
    for form in ('onsite', 'filtered'):
        for prof_name in ('well', 'random'):
            N = M.profile(n) if prof_name == 'well' else list(rng.uniform(0.2, 1.5, n))
            mass_save = M.MH
            for mass in (3 / 5, float(rng.uniform(0.1, 2.0))):
                M.MH = mass
                t0 = time.time()
                H0, B = M.assemble(n, 0.0, form, N=N)
                H1, _ = M.assemble(n, 1.0, form, N=N)
                V = H1 - H0
                idx = {p: i for i, p in enumerate(B)}
                vecs = {}
                for name, comps in (('D', {(0, 3): 1, (1, 2): -1}), ('E', {(0, 1): 0, (2, 3): 0}), ('Kprime', {(0, 1): 1, (2, 3): 1}), ('B', {(0, 1): 1})):
                    if name == 'E':
                        continue
                    v = np.zeros(len(B), dtype=complex)
                    for p, c in comps.items():
                        v[idx[p]] = c
                    vecs[name] = v
                out = {}
                for name, v in vecs.items():
                    leak, dim = krylov_leak(H0, V, v)
                    out[name] = dict(max_rel_leak=float(leak), krylov_dim=dim)
                M.MH = mass_save
                row = dict(n=n, form=form, profile=[float(x) for x in N], mass=mass, results=out, wall_s=time.time() - t0)
                res.append(row)
                print(n, form, prof_name, f'm={mass:.3f}', {k: (f"{v['max_rel_leak']:.1e}", v['krylov_dim']) for k, v in out.items()}, flush=True)
json.dump(res, open('<internal>/cw34_dark.json', 'w'), indent=1)
