#!/usr/bin/env python3
"""route2_exchange.py (busi lane, ROUTE2 v1, stage S1 of REVIEWA_write_ROUTE2_v1.md): the exchange on the second route.

Per member: route 2's chains from the member's N, v; eigh of h_m; the zero-temperature filling (asserted against the
record's channels and E_top, every multiplet closed); the matter's occupied columns Phi_kappa. With
hcal_n = (1/2){Pi_n, h_kappa} (the unclocked chain h), the transition densities rho_ij(n) = <u_i|hcal_n|u_j> and the
monopole kernel V_nn' = -g/max(R_n, R_n'):
 (a) the orbital pair sums: E_H = (1/2) sum_{(k i),(k' j)} w_k w_k' D_ij, D_ij = sum V rho_ii(n) rho_jj(n');
     E_F = -(1/2) sum_k w_k sum_ij X_ij, X_ij = sum V rho_ij(n) rho_ji(n') (within each copy);
 (b) the enclosure form: 1/max(R, R') = sum_{s >= max} 1/(s(s+1)) + 1/(S+1) (S = nsites; the last enclosure carries
     the weight 1/S, the exterior's tail included), E_H = -(g/2) sum_s w_s M_<(s)^2, E_F = (g/2) sum_s w_s
     sum_k w_k ||Phi_k^dag E_<(s) Phi_k||_F^2;
 (c) the self part E_F^self = (g/2) sum_s w_s sum_k w_k sum_i E_i(s)^2 and the remainder;
 (d) the bound |E_H|/N_B, the ratio N_B |E_F/E_H|; the one-particle control; y against the engine's y^W.
Usage: route2_exchange.py --member K --out DIR [--star FAMILY] [--mf MASSFLUCT3_JSON] --threads 3
"""
import os
import sys

if '--threads' in sys.argv:
    _t = sys.argv[sys.argv.index('--threads') + 1]
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_v] = _t
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC  # noqa: E402


def transition_densities(ch, cols):
    """rho_ij(n) = (1/2) sum_s [conj(u_i) (h u_j) + conj(h u_i) u_j] at site n: shape (nsites, n, n), Hermitian in ij."""
    Phi = ch['U'][:, cols]
    hP = ch['h'] @ Phi
    a = np.einsum('xi,xj->xij', Phi.conj(), hP)
    a = 0.5 * (a + np.conj(np.transpose(a, (0, 2, 1))))
    return a.reshape(-1, 2, len(cols), len(cols)).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', type=int, required=True)
    ap.add_argument('--star', default=RC.FAMILY)
    ap.add_argument('--mf', default=None, help='engine massfluct3 json (default: MF_DIR/massfluct3_sfam5_m<k>.json when the star is the family)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--threads', default='3')
    a = ap.parse_args()
    t0 = time.time()
    mem, D = RC.load_member(a.star, a.member)
    tag = a.tag or 'm%d' % a.member
    g, ns, R = mem['g'], mem['nsites'], mem['R']
    RC.log('S1 member %d (%s): N_B %.3f E %.6f, %d sites, g %.3f, ell %s' % (a.member, os.path.basename(a.star), mem['N_B'], mem['E'], ns, g, mem['ell']))
    chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], ns, mem['N_B'], mem['half'])
    asserts = dict(channels=(info['channels'] == mem['channels']), E_top=(abs(info['E_top'] - mem['E_top']) < 1e-10),
                   closed=info['closed'], N_B=(abs(info['N_B_filled'] - mem['N_B']) < 1e-9), sea_is_half=info['sea_is_half'])
    RC.log('filling: channels %d (record %s), E_top %.12f (record %.12f), N_B %.6f, closed %s, kappa_max_occ %d, levels per channel %s; asserts %s'
           % (info['channels'], mem['channels'], info['E_top'], mem['E_top'], info['N_B_filled'], info['closed'], info['kappa_max_occupied'],
              {k: len(v) for k, v in info['levels'].items()}, asserts))
    if not all(asserts.values()):
        RC.log('FILLING ASSERTION FAILED: %s' % asserts)
    # transition densities of the matter, per channel
    rho = {}
    for ch in chans:
        if ch['n_mat']:
            rho[ch['kappa']] = transition_densities(ch, ch['mat'])
    occ_ch = [ch for ch in chans if ch['n_mat']]
    # (a) the orbital pair sums with the dense kernel
    Vk = -g / np.maximum(R[:, None], R[None, :])
    diag = np.concatenate([np.real(np.einsum('nii->ni', rho[ch['kappa']])) for ch in occ_ch], axis=1)    # (nsites, n_orb)
    wts = np.concatenate([np.full(ch['n_mat'], ch['w']) for ch in occ_ch])
    Dm = diag.T @ Vk @ diag
    E_H_a = 0.5 * float(wts @ Dm @ wts)
    E_F_a = 0.0
    X_per = {}
    for ch in occ_ch:
        Rm = rho[ch['kappa']].reshape(ns, -1)
        X = np.real(np.sum(Rm * (Vk @ Rm.conj()), axis=0)).reshape(ch['n_mat'], ch['n_mat'])
        X_per[ch['kappa']] = X
        E_F_a += -0.5 * ch['w'] * float(np.sum(X))
    # (b) the enclosure form
    s = np.arange(1, ns + 1, dtype=float)
    ws = 1.0 / (s * (s + 1.0))
    ws[-1] = 1.0 / s[-1]
    assert np.allclose(R, np.arange(1, ns + 1)), 'the enclosure weights assume R_n = n + 1'
    e_site = diag @ wts
    M_lt = np.cumsum(e_site)
    E_H_b = -0.5 * g * float(np.sum(ws * M_lt ** 2))
    E_F_b = 0.0
    E_F_self = 0.0
    per = []
    E_i = {}
    for ch in occ_ch:
        cum = np.cumsum(rho[ch['kappa']], axis=0)
        F2 = np.sum(np.abs(cum) ** 2, axis=(1, 2))
        Ei = np.real(np.einsum('sii->si', cum))
        E_i[ch['kappa']] = Ei
        ef = 0.5 * g * ch['w'] * float(np.sum(ws * F2))
        es = 0.5 * g * ch['w'] * float(np.sum(ws * np.sum(Ei ** 2, axis=1)))
        E_F_b += ef
        E_F_self += es
        per.append(dict(kappa=ch['kappa'], w=ch['w'], n_matter=ch['n_mat'], E_F=ef, E_F_self=es, E_F_pairsum=-0.5 * ch['w'] * float(np.sum(X_per[ch['kappa']]))))
    NB = info['N_B_filled']
    bound = abs(E_H_b) / NB
    y = E_F_b / E_H_b
    # (d) the one-particle control: channel 1's lowest positive mode alone, one copy
    ch1 = chans[0]
    j1 = ch1['nneg']
    r1 = transition_densities(ch1, [j1])[:, 0, 0].real
    EH1_pair = 0.5 * float(r1 @ Vk @ r1)
    EF1_encl = 0.5 * g * float(np.sum(ws * np.cumsum(r1) ** 2))
    ctrl = dict(mode_energy=float(ch1['e'][j1]), E_H_pairsum=EH1_pair, E_F_enclosure=EF1_encl, rel=(EF1_encl + EH1_pair) / abs(EH1_pair))
    # the engine's values
    mf_path = a.mf or (os.path.join(RC.MF_DIR, 'massfluct3_sfam5_m%d.json' % a.member) if os.path.abspath(a.star) == os.path.abspath(RC.FAMILY) else None)
    eng = None
    if mf_path and os.path.exists(mf_path):
        J = json.load(open(mf_path))
        eng = dict(path=mf_path, y_W=J['y_W'], E_x_W=J['E_x_W'], E_H=J['E_H_same_kernel'])
    acc = dict(a_vs_b_EH=abs(E_H_a - E_H_b) / abs(E_H_b), a_vs_b_EF=abs(E_F_a - E_F_b) / abs(E_F_b))
    acc['a_b_ok'] = bool(max(acc['a_vs_b_EH'], acc['a_vs_b_EF']) <= 1e-10)
    if eng:
        acc['y_vs_engine_rel'] = abs(y - eng['y_W']) / abs(eng['y_W'])
        acc['EF_vs_engine_rel'] = abs(E_F_b - eng['E_x_W']) / abs(eng['E_x_W'])
        acc['EH_vs_engine_rel'] = abs(E_H_b - eng['E_H']) / abs(eng['E_H'])
        acc['y_ok'] = bool(acc['y_vs_engine_rel'] <= 1e-8)
    acc['bound_ok'] = bool(bound <= E_F_self * (1 + 1e-12) and E_F_self <= E_F_b * (1 + 1e-12))
    acc['one_particle_ok'] = bool(abs(ctrl['rel']) <= 1e-14)
    acc['filling_ok'] = bool(all(asserts.values()))
    RC.log('(a) pair sums: E_H %.12e E_F %.12e ; (b) enclosure: E_H %.12e E_F %.12e ; rel diffs %.1e %.1e' % (E_H_a, E_F_a, E_H_b, E_F_b, acc['a_vs_b_EH'], acc['a_vs_b_EF']))
    RC.log('(c) E_F^self %.12e, remainder E_F - E_F^self %.6e ; (d) bound |E_H|/N_B %.12e ; N_B|y| %.6f ; E_F^self/E_F %.6f ; |E_H|/N_B <= E_F^self <= E_F: %s'
           % (E_F_self, E_F_b - E_F_self, bound, NB * abs(y), E_F_self / E_F_b, acc['bound_ok']))
    RC.log('y = E_F/E_H = %.12e%s' % (y, ('  engine y^W %.12e  rel %.2e  (E_F rel %.1e, E_H rel %.1e)' % (eng['y_W'], acc['y_vs_engine_rel'], acc['EF_vs_engine_rel'], acc['EH_vs_engine_rel'])) if eng else ''))
    RC.log('one-particle control: E_F (enclosure) + E_H (pair sum) relative %.1e (mode %.10f)' % (ctrl['rel'], ctrl['mode_energy']))
    out = dict(script='route2_exchange.py', star=a.star, member=a.member, tag=tag, N_B=NB, N_B_record=mem['N_B'], E=mem['E'], ell=mem['ell'],
               filling=info, filling_asserts=asserts, E_H_pairsum=E_H_a, E_F_pairsum=E_F_a, E_H=E_H_b, E_F=E_F_b, E_F_self=E_F_self,
               E_F_remainder=E_F_b - E_F_self, y=y, bound=bound, NB_abs_y=NB * abs(y), self_over_EF=E_F_self / E_F_b, per_channel=per,
               one_particle_control=ctrl, engine=eng, acceptance=acc, seconds=time.time() - t0,
               enclosure_weights='w_s = 1/(s(s+1)) for s < nsites, 1/nsites on the last enclosure (exterior tail 1/(nsites+1) included)')
    os.makedirs(a.out, exist_ok=True)
    RC.dump_json(os.path.join(a.out, 'route2_exchange_%s.json' % tag), out)
    np.savez_compressed(os.path.join(a.out, 'route2_exchange_%s.npz' % tag), M_lt=M_lt, e_site=e_site, ws=ws, R=R,
                        **{'rho_k%d' % k: v for k, v in rho.items()}, **{'Ei_k%d' % k: v for k, v in E_i.items()},
                        **{'X_k%d' % k: v for k, v in X_per.items()}, D_pairs=Dm, weights=wts,
                        per_channel_EF=np.array([p['E_F'] for p in per]), per_channel_EF_self=np.array([p['E_F_self'] for p in per]),
                        per_channel_kappa=np.array([p['kappa'] for p in per]))
    RC.log('S1 member %d done (%.0f s): %s' % (a.member, time.time() - t0, acc))


if __name__ == '__main__':
    main()
