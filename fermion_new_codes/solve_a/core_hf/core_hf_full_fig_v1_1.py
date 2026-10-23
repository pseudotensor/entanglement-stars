#!/usr/bin/env python3
"""
core_hf_full_fig_v1_1.py: the CORE_HF_full_v1.1 figure from core_hf_full_v1_1.npz (written by core_hf_full_data_v1_1.py); no computation.
    python3 core_hf_full_fig_v1_1.py --npz <internal>/core_hf_full_v1_1.npz --out .../CORE_HF_full_v1_1_fig
Panels: (a-c) sqrt(Var_C)/M_H (full, dressed) beside N_<^{-1/2} at R_in, R_50, R_t; (d) E_F/|E_pair| dressed and bare beside 1/N_B and the
comparator 5/(4 N_B); (e, f) the tt and rr evolution rows' residual over the stepping floor (both relative to the row's largest term,
so the ratio is |res| / |second difference over tau|) at R_50 and R_t.
prod (sigma/a 4) blue, s8 (sigma/a 8) red; filled markers complex128, open markers complex64 (stored floor).
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {'prod': 'tab:blue', 's8': 'tab:red'}
LAB = {'prod': 'prod (sigma/a 4)', 's8': 's8 (sigma/a 8)'}


def pts(ax, T, y, c64, color, marker, label=None, ls='-'):
    ax.plot(T, y, ls=ls, color=color, lw=1.0, alpha=0.8)
    m = ~np.isnan(y)
    f = m & ~c64
    o = m & c64
    ax.plot(T[f], y[f], marker, color=color, ms=5, ls='none', label=label if f.any() else None)
    ax.plot(T[o], y[o], marker, mfc='none', color=color, ms=5, ls='none', label=None if f.any() else label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='<internal>/core_hf_full_v1_1.npz')
    ap.add_argument('--out', default='<internal>/CORE_HF_full_v1_1_fig')
    a = ap.parse_args()
    Z = np.load(a.npz)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for i, lab in enumerate(('R_in', 'R_50', 'R_t')):
        ax = axs[0, i] if i < 3 else None
        for h in ('prod', 's8'):
            T, c64 = Z['%s_T' % h], Z['%s_c64' % h].astype(bool)
            pts(ax, T, Z['%s_%s_ratio' % (h, lab)], c64, COL[h], 'o', '%s sqrt(Var_C)/M_H' % LAB[h])
            pts(ax, T, Z['%s_%s_Ninvsqrt' % (h, lab)], c64, COL[h], 's', '%s N_<^-1/2' % LAB[h], ls='--')
        ax.set_title('(%s) %s: sqrt(Var_C)/M_H (circles), N_<^-1/2 (squares, dashed)' % ('abc'[i], lab), fontsize=9)
        ax.set_xlabel('T / r_s')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    ax = axs[1, 0]
    for h in ('prod', 's8'):
        T, c64 = Z['%s_T' % h], Z['%s_c64' % h].astype(bool)
        pts(ax, T, Z['%s_y_dressed' % h], c64, COL[h], 'o', '%s dressed' % LAB[h])
        pts(ax, T, Z['%s_y_bare' % h], c64, COL[h], '^', '%s bare' % LAB[h], ls=':')
    ax.axhline(Z['prod_invNB'][0], color='black', ls='--', lw=1, label='1/N_B (comparator)')
    ax.axhline(Z['prod_bound'][0], color='black', ls=':', lw=1, label='5/(4 N_B) (comparator, not a bound here)')
    ax.set_title('(d) whole-slice E_F/|E_pair|, dressed and bare native kernels', fontsize=9)
    ax.set_xlabel('T / r_s')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.93), ncol=2)
    for j, row in enumerate(('tt', 'rr')):
        ax = axs[1, 1 + j]
        for h in ('prod', 's8'):
            T, c64 = Z['%s_T' % h], Z['%s_c64' % h].astype(bool)
            for lab, mk, ls in (('R_50', 'o', '-'), ('R_t', '^', '--')):
                q = Z['%s_%s_rel_%s' % (h, lab, row)] / Z['%s_%s_floor_%s' % (h, lab, row)]
                pts(ax, T, q, c64, COL[h], mk, '%s at %s' % (LAB[h], lab), ls=ls)
        ax.axhline(1.0, color='black', lw=1, ls=':', label='residual = stepping floor')
        ax.set_yscale('log')
        ax.set_title('(%s) %s row: residual / stepping floor at R_50 (circles) and R_t (triangles)' % ('ef'[j], row), fontsize=9)
        ax.set_xlabel('T / r_s')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('CORE_HF full run: prod (10 slices, card bench1b) and s8 (8 slices, card R2); filled complex128, open complex64. '
                 'T 40: no T + tau record, rows not computed. s8 T 12, 20, 25, 30 and prod T 12 have no trapping roots (R_50 only).', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('png', 'pdf'):
        fig.savefig('%s.%s' % (a.out, ext), dpi=130)
    print('wrote %s.png/.pdf' % a.out)


if __name__ == '__main__':
    main()
