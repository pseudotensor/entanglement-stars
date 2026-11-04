# ref707_spread_v1 (#707 stage 1, ADDENDUM_707_ref_v1.md 87ef1164 "spread" item; 2026-10-31): READ-ONLY. The tau4-vs-tau8 step-control spread
# of the r11 member at any T/r_s (the frozen rule: c27 uses r11's spread at the same T/r_s), and exact per-slice field/row numbers from s2 JSONs.
#   python -I ref707_spread_v1.py <tau8 fields.npz> <tau4 fields.npz> <rs> <T/rs,...> <s2 json> [<s2 json> ...]
import sys, json
import numpy as np

Z8, Z4 = np.load(sys.argv[1]), np.load(sys.argv[2]); rs = float(sys.argv[3]); Ts = [float(t) for t in sys.argv[4].split(',')]
T8, T4 = np.asarray(Z8['T'], float), np.asarray(Z4['T'], float)
for t in Ts:
    i, j = int(np.argmin(np.abs(T8 - t * rs))), int(np.argmin(np.abs(T4 - t * rs)))
    d = {nm: float(np.abs(np.asarray(Z4[nm][j]) - np.asarray(Z8[nm][i])).max()) for nm in ('eps_tot', 'J_tot', 'N', 'v')}
    print('spread T/rs %.3f (tau8 T %.4f, tau4 T %.4f): %s' % (t, T8[i] / rs, T4[j] / rs, json.dumps(d)))
for p in sys.argv[5:]:
    o = json.load(open(p))
    for lab, r in o['slices'].items():
        rw = r['rows_dm']
        print('%s T/rs %s: V1 %s V2 %s dS_eps %r dS_J %r dN %r dv %r dE %r E %r res_W %r res_invN %r failed %r' % (o['member'], lab, r['V1'], r['V2'], r['dS_eps']['max'], r['dS_J']['max'],
              rw['dN_max'], rw['dv_max'], rw['E_dm'] - rw['E_held'], rw['E_held'], rw['residual_W'], rw['residual_invN'], rw['failed']))
