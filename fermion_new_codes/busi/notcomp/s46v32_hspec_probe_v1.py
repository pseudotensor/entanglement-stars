#!/usr/bin/env python3
"""s46v32_hspec_probe_v1.py (busi #472, read-only probe): the eigenvalue spectra of the saved H_j and V (prod 15): effective ranks
(the number of |lambda| above 1e-2, 1e-4, 1e-6 of the largest), sum |lambda| / sqrt(sum lambda^2)."""
import numpy as np
O = np.load('<internal>/prod_T0015.00_ops.npz')
for l in [str(x) for x in O['order']]:
    H = O['H_' + l]
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    a = np.sort(np.abs(ev))[::-1]
    print(l, 'max %.3e' % a[0], 'ranks', [int((a > t * a[0]).sum()) for t in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8)],
          'sum|l|/sqrt(sum l^2) %.2f' % (a.sum() / np.sqrt((a ** 2).sum())), 'neg/pos max %.3e %.3e' % (ev.min(), ev.max()))
