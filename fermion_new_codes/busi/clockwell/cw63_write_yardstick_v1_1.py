"""cw63: write MOCK_n5_yardstick_v1.1.md (#498) from cw55 (executed yardstick) and cw62 (certified HH -> LL norm, side-by-side
bounds). v1 (MOCK_n5_yardstick_v1.md) is kept. Every number is emitted here; the #498 sentence is printed only if the
certified numbers support it. Output: <internal>/MOCK_n5_yardstick_v1.1.md"""
import json, re, hashlib
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
Y = '<internal>/yardstick_v1'; B = '<internal>/clockwell'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def out(x, s, mode):
    if x == 0:
        return '0'
    d = Decimal(repr(float(x))); e = d.adjusted() - s + 1
    q = d.quantize(Decimal(1).scaleb(e), rounding=mode)
    return format(q.normalize(), 'g') if abs(q.adjusted()) < 6 else format(q, 'E')

up = lambda x, s=6: out(x, s, ROUND_CEILING)
lo = lambda x, s=6: out(x, s, ROUND_FLOOR)
I = lambda v, s=10: f'[{lo(v[0], s)}, {up(v[1], s)}]'

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    return o

a = json.load(open(f'{Y}/cw55_yardstick.json')); c = json.load(open(f'{Y}/cw62_tnorm_dw.json'))
lab = lambda fo: 'on-site' if fo == 'onsite' else 'filtered (endpoint-copy halo)'
v1 = f'{Y}/MOCK_n5_yardstick_v1.md'
L = ['# MOCK n = 5 free-phase yardstick v1.1 (#498): certified HH -> LL norm and side-by-side wall bounds', '',
     f'Author: busi (analyst engineer), busi inbox #498 after ReviewA_mock6_v1.md and ReviewB_mock6_v1.md. v1 is kept unchanged: `{v1}` md5 `{md5(v1)}`. **The verdict of record stays VOID for both forms** (v1, executed one-shot rule); nothing below changes it. No n = 7 state was started. No paper text.', '',
     '## 1. Ball-certified upper bound on the HH -> LL block norm of V_ext', '',
     'T_ext = the block of V_ext = H_ext(g = 1) - H_ext(g = 0) (empty 4-site extension, n + 4 = 9 sites) from the heavy-pair sector to the light-pair sector of the extended chain. Sector restriction: the zero-order wall error e0 = psi0_ext - J psi0 lies in the heavy-pair sector (B is a heavy pair and H0 preserves species), so V_ext e0 = T_ext e0 and only ||T_ext||_2 enters the V_ext e0 term; the other blocks of V_ext acting on the heavy-pair sector are zero (column below). Certificate: G = T^dag T in ball arithmetic (128 bit); s = lambda_max(G)_float + 2 eta, eta = ' + str(c['eta']) + '; s I - G = B B^dag + eta I + E with ||E||_F < eta (Weyl) proves lambda_max(G) <= s; ||T||_2 <= sqrt(s) + e_T (e_T: target error of T from the exact kernel deltas plus an a priori allowance).', '',
     '| form | T shape | other blocks from HH (max abs) | ||T||_2 (floating) | certified upper bound | PSD certificate holds | ||E||_F (upper) | e_T | Schur norm of V_ext (v1) |', '|---|---|---|---|---|---|---|---|---|']
for fo, x in c['forms'].items():
    L.append(f"| {lab(fo)} | {x['T_shape'][0]} x {x['T_shape'][1]} | {up(x['T_other_blocks_from_HH_max_abs'], 3)} | {up(x['T_norm_float'], 12)} | {up(x['T_norm_certified_upper'], 12)} | {x['psd_certificate']} | {up(x['E_frob_upper'], 3)} | {up(x['e_T'], 3)} | {up(x['Schur_V_ext'], 6)} |")
L += ['', '## 2. Side-by-side first-order wall bounds on psi1 (same P as executed)', '',
      "(A) = the executed bound, normalization 2E/l; (D on A) = Dunkl-Williams on A, 2E/(2l - E); (D on C) = Dunkl-Williams on C, where C uses the combined source ||S0 psi1 + S_V psi0|| and the certified ||T_ext||_2 of sec. 1 in the V_ext e0 term, all other constants and error terms as executed. l = ball-certified lower ||psi1(t)||; l - E > 0 is certified at every clock where 2E/(2l - E) is used.", '',
      '| form | P | t | l (lower ||psi1||) | E1 (A) | E1 (C) | l - E1 (A) | l - E1 (C) | (A) | (D on A) | (D on C) |', '|---|---|---|---|---|---|---|---|---|---|---|']
allpos = True
for fo, x in c['forms'].items():
    for r in x['rows']:
        allpos = allpos and r['l_minus_E_A'] > 0 and r['l_minus_E_C'] > 0
        L.append(f"| {lab(fo)} | {x['P']} | {r['t']:g} | {lo(r['l'], 10)} | {up(r['E1_A'], 6)} | {up(r['E1_C'], 6)} | {lo(r['l_minus_E_A'], 6)} | {lo(r['l_minus_E_C'], 6)} | {up(r['A'], 6)} | {up(r['DW_A'], 6) if r['DW_A'] is not None else 'n/a'} | {up(r['DW_C'], 6) if r['DW_C'] is not None else 'n/a'} |")
# sentence check
dc3 = {fo: [r for r in x['rows'] if r['t'] == 3.0][0]['DW_C'] for fo, x in c['forms'].items()}
Is = {fo: a['forms'][fo]['Delta_interval'] for fo in a['forms']}
other_ok = {fo: (a['forms'][fo]['VB_norm2_lower'] > 0 and a['forms'][fo]['linearity_recheck_max_abs_H2_minus_2H1_plus_H0'] < 1e-12 and
                 all(cc['N_L_over_norm2'][0] >= 0.05 and cc['psi1_norm_lower'] > 0 for cc in a['forms'][fo]['clocks'])) for fo in a['forms']}
supported = (allpos and all(x['psd_certificate'] for x in c['forms'].values()) and all(v is not None and v <= 0.1 for v in dc3.values())
             and all(v[1] < -0.1 for v in Is.values()) and all(other_ok.values()))
L += ['', 'Check of the #498 sentence (computed): ' + '; '.join(f"{lab(fo)}: (D on C) at t = 3 = {up(dc3[fo], 6)} (<= 0.1: {dc3[fo] is not None and dc3[fo] <= 0.1}), I = {I(Is[fo])} (negative, outside [-0.1, 0.1]: {Is[fo][1] < -0.1}), other prerequisites hold: {other_ok[fo]}" for fo in c['forms']) + f"; l - E > 0 certified everywhere used: {allpos}; norm certificates hold: {all(x['psd_certificate'] for x in c['forms'].values())}.", '']
if supported:
    L += ['under the exact normalization both forms would be judgeable, both I negative outside [-0.1, 0.1]; not adopted after the fact.', '']
else:
    L += ['The certified numbers do NOT support the #498 sentence, so it is not printed; the values above are what the certified numbers show.', '']
t62 = tparse(f'{Y}/cw62_time.txt')
L += ['## 3. Cost', '', '| item | CPU s |', '|---|---|'] + [f"| {lab(fo)}: norm certificate | {up(x['cert_cpu_s'], 4)} |" for fo, x in c['forms'].items()] + [f"| {lab(fo)}: recompute of C at P = {x['P']} | {up(x['recompute_cpu_s'], 4)} |" for fo, x in c['forms'].items()] + \
     [f"| cw62 process | {t62['user']} user, wall {t62['wall']}, peak RSS {t62['rss']} KiB |", '', 'Caps: nice 10, 4 threads, ulimit -v 8000000, ulimit -t 3600, box CPU.', '',
      '## 4. For the second reader', '',
      'Inputs: n5run/<form>_g0, _g1 hamiltonian.npz, eigenbasis.npz (g0), joining_source_S.npz, result.json; cw55_yardstick.json (executed E1, l, I). (1) Rebuild H_ext at n + 4 = 9 sites (cw32 builder, profile (1/2, 1, ..., 1), added clocks 1, filtered halo copy) for g = 1 and 0; T_ext = rows of the light-pair sector, columns of the heavy-pair sector of their difference; check the other blocks from the heavy-pair sector are zero; certify an upper bound of ||T_ext||_2 (e.g. an interval Gram matrix with a certified eigenvalue enclosure). (2) For (D on A): 2 E1/(2 l - E1) with E1 and l from cw55. (3) For (D on C): recompute E1 at the executed P with the combined source and the certified norm. Criterion: the table of sec. 2 and the sentence check; the verdict of record is not re-judged.', '',
      '## 5. Files (box <internal>/yardstick_v1, MD5SUMS there)', '', '| file | md5 |', '|---|---|']
for p_ in (f'{Y}/cw55_yardstick.json', f'{Y}/cw62_tnorm_dw.json', v1, f'{B}/code/cw32_mockmodel.py', f'{B}/code/cw55_yardstick.py', f'{B}/code/cw62_tnorm_dw.py', f'{B}/code/cw63_write_yardstick_v1_1.py'):
    L.append(f'| `{p_}` | `{md5(p_)}` |')
L.append('')
p = f'{Y}/MOCK_n5_yardstick_v1.1.md'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p), 'sentence printed:', supported)
