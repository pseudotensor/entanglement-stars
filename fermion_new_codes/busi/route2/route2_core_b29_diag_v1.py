#!/usr/bin/env python3
"""route2_core_b29_diag_v1.py (busi lane, inbox #349): the diagnosis of the two prod T15 R_in sea-part differences of the B29
comparison (route 2 against route 1 at 2.7e-10 relative, beyond the declared contraction budget 4.8e-11 taken from route 1's Q6').

For the vertices at R_in (noref and full; route 1's stored vectors, equal to route 2's to 3e-13) on the slice's channels, the sea part
Var_P = Tr[P A (1 - P) A], P = R2 R2^+ (Loewdin), is evaluated three ways per channel:
  residual   ||A R2 - R2 (R2^+ A R2)||_F^2          (route 2's headline form; roundoff ~ 2 eps sqrt(s) relative, s = ||A R2||^2/Var_P)
  unreduced  ||A R2||_F^2 - ||R2^+ A R2||_F^2        (the same cancellation as a Gram quadratic form; roundoff ~ eps s relative)
  gram       beta^T Re[Tr(A_mu (1 - P) A_nu P)] beta (route 1's form, rebuilt with route 2's bilinear code)
and the cancellation scale s is printed. Output: <out>/diag_<hist>_T<TTTT.TT>.json. One GPU.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_core_b29_v1 as M  # noqa: E402
import route2_common as RC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', nargs=2, default=['prod', '15'])
    ap.add_argument('--route1-arrays', default='<internal>/core_hf_prod_T0015.00_arrays.npz')
    ap.add_argument('--route1-json', default='<internal>/core_hf_prod_T0015.00.json')
    ap.add_argument('--labels', default='R_in,R_50')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='<internal>/route2_core')
    a = ap.parse_args()
    os.nice(10)
    dev = torch.device(a.device)
    hist, t = a.slice[0], float(a.slice[1])
    Z = np.load(M.find_slice(hist, t))
    rec = M.load_record(hist, t)
    ns = len(Z['R'])
    N, v = np.asarray(Z['N_applied'], float), np.asarray(Z['v_applied'], float)
    Z1 = np.load(a.route1_arrays)
    J1 = json.load(open(a.route1_json))
    labels = a.labels.split(',')
    out = dict(script='route2_core_b29_diag_v1', code_md5=M.md5_8(os.path.abspath(__file__)), slice=[hist, t], labels={})
    acc = {(l, nm): dict(residual=0.0, unreduced=0.0, gram=0.0, per_channel=[]) for l in labels for nm in ('noref', 'full')}
    for k in [int(x) for x in np.atleast_1d(Z['kappas'])]:
        h, Q, R = RC.chain_ops(ns, rec['m'], float(k), False)
        hb, qb = M.Band.from_scipy(h, dev), M.Band.from_scipy(Q, dev)
        R2, _, _ = M.loewdin(torch.as_tensor(np.asarray(Z['R2_%d' % k]), device=dev).to(M.CT))
        P = M.herm(R2 @ M.adj(R2))
        I = torch.eye(2 * ns, dtype=M.CT, device=dev)
        G = torch.real(M.bilinear((hb, qb), I - P, P, ns)).cpu().numpy()
        G = 0.5 * (G + G.T)
        del P, I
        w = 16.0 * k
        for l in labels:
            for nm in ('noref', 'full'):
                x = Z1['beta_%s_%s' % (l, nm)]
                A = M.vertex_band(hb, qb, x, ns)
                AP = A.left(R2)
                Mm = M.adj(R2) @ AP
                res = M.fro2(AP - R2 @ Mm)
                big = M.fro2(AP)
                unr = big - M.fro2(Mm)
                gr = float(x @ (G @ x))
                e = acc[(l, nm)]
                e['residual'] += w * res
                e['unreduced'] += w * unr
                e['gram'] += w * gr
                e['per_channel'].append(dict(kappa=k, residual=res, unreduced=unr, gram=gr, scale=big / max(res, 1e-300),
                                             unreduced_minus_residual_rel=(unr - res) / res, gram_minus_residual_rel=(gr - res) / res))
        torch.cuda.empty_cache()
        print('kappa %d done' % k, flush=True)
    for (l, nm), e in acc.items():
        r1 = J1['variance'][l]['variants'][nm]['Var_C0']
        e['route1_json_Var_C0'] = r1
        e['route1_minus_residual_rel'] = (r1 - e['residual']) / e['residual']
        e['gram_minus_residual_rel'] = (e['gram'] - e['residual']) / e['residual']
        e['unreduced_minus_residual_rel'] = (e['unreduced'] - e['residual']) / e['residual']
        e['route1_minus_gram_rel'] = (r1 - e['gram']) / e['residual']
        e['max_scale'] = max(p['scale'] for p in e['per_channel'])
        out['labels']['%s %s' % (l, nm)] = e
        print('%s %s: residual %.15e unreduced %.15e gram %.15e route 1 %.15e | rel to residual: unreduced %.2e gram %.2e route 1 %.2e; route 1 - gram %.2e; '
              'max scale %.2e' % (l, nm, e['residual'], e['unreduced'], e['gram'], r1, e['unreduced_minus_residual_rel'], e['gram_minus_residual_rel'],
                                  e['route1_minus_residual_rel'], e['route1_minus_gram_rel'], e['max_scale']), flush=True)
    RC.dump_json(os.path.join(a.out, 'diag_%s_T%07.2f.json' % (hist, t)), out)


if __name__ == '__main__':
    main()
