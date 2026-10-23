"""cw36: MEASURED cost pilot for the n = 5 mock (#491 B(i)); busi; NOT the run.
Times every stage of one g = 1 arm per converter form with busi's own pipeline (cw32 builder) and records
dimensions, nnz, CPU, wall and peak RSS. No physical readout value is printed or saved: every computed
quantity is discarded after timing (only shapes, counts and costs are written).
Stages: assembly; eigendecomposition; certified residual bound ||H V - V Lambda||_F in ball arithmetic (acb, 128 bit);
actions at the 5 listed clocks (pure and registered); joining-source assembly at n + 4 = 9 sites and the 320
wall-midpoint source norms on [0, 5]; readouts at one clock (p_LL, cut entropies, D_RK light/all and first-block,
protected-segment assembly and action, LL-band spectrum).
Output: <internal>/cw36_pilot.json"""
import json, sys, time, resource, os
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
from flint import acb_mat, acb, arb
import flint
flint.ctx.prec = 128

def stamp():
    return time.process_time(), time.time()

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def stage(log, name, t0w0):
    t0, w0 = t0w0
    log.append(dict(stage=name, cpu_s=time.process_time() - t0, wall_s=time.time() - w0, peak_rss_kib=rss()))
    print(f'{name:40s} cpu {log[-1]["cpu_s"]:8.2f} s wall {log[-1]["wall_s"]:8.2f} s rss {log[-1]["peak_rss_kib"]} KiB', flush=True)

def acb_from(Mx):
    return acb_mat([[acb(float(x.real), float(x.imag)) for x in row] for row in Mx])

def main():
    n = 5; out = {'n': n, 'arms': {}}
    B = M.basis(n); D = len(B)
    out['dims'] = dict(modes=8 * n, two_particle=D, registered=2 * D, odd_one_particle=8 * n,
                       whole_lead_support=1 + 32 + 32 * 31 // 2, core_support=1 + 8 + 28,
                       RxK_light_whole_lead=2 * (1 + 24 + 24 * 23 // 2), RxK_all_whole_lead=2 * (1 + 8 + 28),
                       RxK_light_first_block_site4=2 * (1 + 36 + 36 * 35 // 2), LL_sector=20 * 19 // 2,
                       extension_sites=n + 4, extension_two_particle=(8 * (n + 4)) * (8 * (n + 4) - 1) // 2)
    for form in ('onsite', 'filtered'):
        log = []
        s = stamp(); H, _ = M.assemble(n, 1.0, form); stage(log, 'assembly', s)
        nnz = int(np.count_nonzero(H)); nq = len(M.onsite_quartic(n) if form == 'onsite' else M.filtered_quartic(n))
        s = stamp(); w, V = np.linalg.eigh(H); stage(log, 'eigendecomposition', s)
        s = stamp()
        Ha = acb_from(H); Va = acb_from(V)
        Rres = Ha * Va - acb_from(V * w)
        fro = arb(0)
        for row in Rres.tolist():
            for e in row:
                fro += e.real * e.real + e.imag * e.imag
        del Ha, Va, Rres
        stage(log, 'certified_residual_ball_128bit', s)
        psi0, reg0 = M.initial_states(B)
        s = stamp()
        for t in (1.0, 2.0, 3.0, 4.0, 5.0):
            _ = V @ (np.exp(-1j * w * t) * (V.conj().T @ psi0))
            _ = (V @ (np.exp(-1j * w * t)[:, None] * (V.conj().T @ reg0.T))).T
        stage(log, 'actions_5_clocks_pure_and_registered', s)
        s = stamp()
        n2 = n + 4
        Nshort = M.profile(n); Next = Nshort + [1.0] * 4
        Hext, Bext = M.assemble(n2, 1.0, form, N=Next)
        stage(log, 'extension_assembly_n_plus_4', s)
        s = stamp()
        iext = {p: i for i, p in enumerate(Bext)}
        emb = np.array([iext[p] for p in B])
        S = Hext[:, emb].copy(); S[emb, :] -= H
        Sschur = float(np.abs(S.conj().T @ S).sum(axis=1).max())
        mids = [(k + 0.5) / 64 for k in range(64 * 5)]
        for t in mids:
            psi = V @ (np.exp(-1j * w * t) * (V.conj().T @ psi0))
            _ = np.linalg.norm(S @ psi)
        stage(log, 'joining_source_and_320_midpoints', s)
        del Hext, S
        s = stamp()
        psi = V @ (np.exp(-1j * w * 4.0) * (V.conj().T @ psi0)); reg = (V @ (np.exp(-1j * w * 4.0)[:, None] * (V.conj().T @ reg0.T))).T
        _ = M.p_species(psi, B)
        for cut in (list(range(8, 40)), list(range(32, 40)), list(range(0, 8)), list(range(0, 32))):
            rho, _ = M.reduce_rho(psi.reshape(1, -1), B, cut); _ = M.vn(rho)
        for outm in ([m for x in range(1, 5) for m in range(8 * x + 4, 8 * x + 8)], list(range(8, 40)), list(range(36, 40)), list(range(32, 40))):
            _ = M.D_RK(reg, B, outm, n)
        stage(log, 'readouts_one_clock', s)
        s = stamp()
        site4 = set(range(32, 40))
        keep = lambda modes: (set(modes) <= site4) or not (set(modes) & site4)
        Hq, _ = M.assemble(n, 1.0, form, keep=keep)
        _ = np.linalg.eigh(Hq)
        stage(log, 'protected_assembly_and_eigendecomposition', s)
        s = stamp()
        LL = np.array([(a % 8) // 4 + (b % 8) // 4 == 2 for a, b in B])
        _ = np.linalg.eigvalsh(H[np.ix_(LL, LL)])
        stage(log, 'LL_band_spectrum', s)
        out['arms'][form] = dict(nnz_H=nnz, quartic_monomials=nq, stages=log, total_cpu_s=sum(x['cpu_s'] for x in log))
    out['peak_rss_kib'] = rss(); out['cpu_s_total'] = time.process_time()
    json.dump(out, open('<internal>/cw36_pilot.json', 'w'), indent=1)
    print('pilot done; total cpu', out['cpu_s_total'], 'peak rss', out['peak_rss_kib'])

if __name__ == '__main__':
    main()
