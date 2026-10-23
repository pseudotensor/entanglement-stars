"""cw58: #497 STEP 2, the n = 7 COSTING and ADMISSION pilot (busi). HARD CAP 1000 CPU s (ulimit -t 1000 set by the
launcher; ulimit -v 8000000). NO state readout is computed for printing: no f_L, N_L, N_H, p_LL; propagated vectors are
used only to time the midpoint step and are discarded.
Measures: active dimension and nnz; setup (assembly) CPU of H0 (common to both forms) and H1 per form; diagonalization
CPU; the certified eigen-residual ||H V - V Lambda||_F in ball arithmetic (128 bit, the existing accuracy rule) for H0 and
each H1; the joining-source build at n + 4 = 11 sites; midpoint throughput (propagate B + one S product, 256 midpoints);
peak RSS. Output: <internal>/cw58_pilot.json (costs and dimensions only)."""
import json, os, sys, time, resource
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
OUT = '<internal>/n7pilot'; os.makedirs(OUT, exist_ok=True)
log = []

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def stage(name, t0, w0, **kw):
    log.append(dict(stage=name, cpu_s=time.process_time() - t0, wall_s=time.time() - w0, peak_rss_kib=rss(), **kw))
    json.dump(dict(stages=log), open(f'{OUT}/cw58_pilot.json', 'w'), indent=1)
    print(f"{name:45s} cpu {log[-1]['cpu_s']:8.2f} wall {log[-1]['wall_s']:7.2f} rss {log[-1]['peak_rss_kib']}", flush=True)

def acb_from(Mx):
    return acb_mat([[acb(float(x.real), float(x.imag)) for x in row] for row in Mx])

def frob_upper(A):
    s = arb(0)
    for row in A.tolist():
        for e in row:
            s += e.real * e.real + e.imag * e.imag
    return float(s.sqrt().upper())

def main():
    n = 7
    t0, w0 = time.process_time(), time.time()
    H0, B = M.assemble(n, 0.0, 'onsite')
    stage('assembly_H0', t0, w0, dim=len(B), nnz=int(np.count_nonzero(H0)))
    t0, w0 = time.process_time(), time.time()
    w0v, V0 = np.linalg.eigh(H0)
    stage('diagonalization_H0', t0, w0)
    t0, w0 = time.process_time(), time.time()
    r0 = frob_upper(acb_from(H0) * acb_from(V0) - acb_from(V0 * w0v))
    stage('certified_residual_H0_ball', t0, w0, residual_frob_upper=r0)
    psi0, _ = M.initial_states(B)
    for form in ('onsite', 'filtered'):
        t0, w0 = time.process_time(), time.time()
        H1, _ = M.assemble(n, 1.0, form)
        stage(f'assembly_H1_{form}', t0, w0, nnz=int(np.count_nonzero(H1)))
        t0, w0 = time.process_time(), time.time()
        w, V = np.linalg.eigh(H1)
        stage(f'diagonalization_H1_{form}', t0, w0)
        t0, w0 = time.process_time(), time.time()
        r1 = frob_upper(acb_from(H1) * acb_from(V) - acb_from(V * w))
        stage(f'certified_residual_H1_{form}_ball', t0, w0, residual_frob_upper=r1)
        t0, w0 = time.process_time(), time.time()
        Hext, Bext = M.assemble(n + 4, 1.0, form, N=M.profile(n) + [1.0] * 4)
        iext = {p: i for i, p in enumerate(Bext)}; emb = np.array([iext[p] for p in B])
        S = Hext[:, emb].copy(); S[emb, :] -= H1
        stage(f'joining_source_n_plus_4_{form}', t0, w0, ext_dim=len(Bext), S_shape=list(S.shape))
        del Hext
        t0, w0 = time.process_time(), time.time()
        c = V.conj().T @ psi0
        for k in range(256):
            tm = (k + 0.5) / 64
            phi = V @ (np.exp(-1j * w * tm) * c)
            _ = np.linalg.norm(S @ phi)
        stage(f'midpoints_256_{form}', t0, w0, per_midpoint_cpu_s=(time.process_time() - t0) / 256)
        del S, V, H1
    json.dump(dict(stages=log, n=n, dim=len(B), peak_rss_kib=rss(), cpu_s_total=time.process_time()), open(f'{OUT}/cw58_pilot.json', 'w'), indent=1)
    print('done cpu', time.process_time(), 'rss', rss())

if __name__ == '__main__':
    main()
