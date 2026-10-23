"""cw18: ADMISSION for the #490 channel-optimal recovery audit (busi). Read-only on icloud's saved outputs.

States: n = 3, g in {0, 1}, clocks t in {1, 2, 4}, both converter forms (on-site baseline and filtered
endpoint-copy halo, N_-1 = 1/2, N_n = 1): the saved registered primary raw vector (shape 2 x 276, R first)
of each physical job. For every state, in exact rational arithmetic (binary64 entries are dyadic rationals):
  * squared norm N2 = sum |psi|^2 exactly, a certified positive lower bound (outward 30-digit decimal),
    |N2 - 1|, and the tolerance it must meet: the readout addendum's normalized-state rule with the saved
    state error eps of that action, |N2 - 1| <= (1 + eps)^2 - 1 (a computed vector within eps of a unit
    target), also printed against the run spec's fixed norm-drift threshold 2e-12;
  * the trace divisor(s) used downstream: N2 only (rho = C C^dagger / N2); no block or branch divisor;
  * the code Gram matrix G_rs = 2 <v_r, v_s> / N2 of the two evolved codeword branches (v_r = row r of the raw
    array; the preparation is (|0>|B> + |1>|D>)/sqrt 2), its certified Frobenius distance from I_2, and the
    tolerance 2 * (2 sqrt 2 eps + 2 eps^2) (each branch within sqrt 2 eps of an orthonormal target).
STOP flag set if any check fails. Output: <internal>/cw18_admission.json
"""
import json, os, hashlib, math
from fractions import Fraction as Fr
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING
import numpy as np

getcontext().prec = 60
ROOT = '<internal>/solve_b_bulk'
EXT = '<internal>/cw05_extract.json'
OUTD = '<internal>/recovery'
os.makedirs(OUTD, exist_ok=True)

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def dec(x, rounding):
    getcontext().rounding = rounding
    return str(Decimal(x.numerator) / Decimal(x.denominator))[:34]

def sqrt_upper(x):
    """certified upper bound of sqrt of a nonnegative Fraction, as Fraction"""
    if x == 0:
        return Fr(0)
    r = Fr(math.sqrt(float(x)) * (1 + 1e-12) + 1e-300)
    while r * r < x:
        r *= Fr(1000001, 1000000)
    return r

def states():
    ext = json.load(open(EXT))
    out = []
    for arm in ('baseline', 'filtered'):
        for gs in ('0', '1'):
            k = f'{arm}/even_n3_g{gs}_v1'; P = ext['physical'][k]; d = f"{ROOT}/{P['top']}/{P['name']}"
            reg = P['plans']['registered']; found = {}
            with open(P['report']) as f:
                for line in f:
                    if not line.startswith('{"kind":"action"'):
                        continue
                    r = json.loads(line)['data']; m = r['metadata']
                    if m.get('plan') == reg and m.get('role') == 'primary' and m.get('start') == 0.0 and m.get('end') in (1.0, 2.0, 4.0):
                        found[m['end']] = (r['raw_vector']['array'], m['error'], m['output'])
            for t in (1.0, 2.0, 4.0):
                arr, err, outh = found[t]
                out.append(dict(arm=arm, g=gs, t=t, cohort=k, report=P['report'], report_md5=P['report_md5'], dir=d,
                                array=f'{d}/{arr}', state_error=err, state_digest=outh, row_keys=P['conversion'][0]['row_keys']))
    return out

def main():
    res = {'states': [], 'stop': False}
    for s in states():
        z = np.load(s['array']); v = z[z.files[0]]
        assert v.shape == (2, 276), v.shape
        re = [[Fr(float(x)) for x in row.real] for row in v]; im = [[Fr(float(x)) for x in row.imag] for row in v]
        def inner(a, b):  # <v_a, v_b> exact (complex as pair)
            rr = sum(re[a][i] * re[b][i] + im[a][i] * im[b][i] for i in range(276))
            ii = sum(re[a][i] * im[b][i] - im[a][i] * re[b][i] for i in range(276))
            return rr, ii
        n00 = inner(0, 0)[0]; n11 = inner(1, 1)[0]; n01 = inner(0, 1)
        N2 = n00 + n11
        eps = Fr(s['state_error'])
        tolN = (1 + eps) ** 2 - 1
        devN = abs(N2 - 1)
        G = [[2 * n00 / N2, (2 * n01[0] / N2, 2 * n01[1] / N2)], [None, 2 * n11 / N2]]
        fro2 = (G[0][0] - 1) ** 2 + (G[1][1] - 1) ** 2 + 2 * (G[0][1][0] ** 2 + G[0][1][1] ** 2)
        froU = sqrt_upper(fro2)
        sq2 = Fr(1414213562373095049, 10 ** 18) + Fr(1, 10 ** 18)  # > sqrt 2
        tolG = 2 * (2 * sq2 * eps + 2 * eps ** 2)
        ok = (N2 > 0) and devN <= tolN and froU <= tolG
        row = dict(arm=s['arm'], g=s['g'], t=s['t'], cohort=s['cohort'], array=s['array'], array_md5=md5(s['array']),
                   report=s['report'], report_md5=s['report_md5'], state_digest=s['state_digest'],
                   N2_exact=f'{N2.numerator}/{N2.denominator}', N2_lower=dec(N2, ROUND_FLOOR), N2_upper=dec(N2, ROUND_CEILING),
                   N2_positive=N2 > 0, abs_N2_minus_1=float(devN), abs_N2_minus_1_upper=dec(devN, ROUND_CEILING),
                   state_error=s['state_error'], tolerance_addendum=float(tolN), within_addendum=devN <= tolN,
                   within_runspec_2e_12=devN <= Fr(2, 10 ** 12),
                   trace_divisors=['N2 (rho_RO = C C^dagger / N2); no block, branch or postselection divisor'],
                   gram=dict(G00=float(G[0][0]), G11=float(G[1][1]), G01_re=float(G[0][1][0]), G01_im=float(G[0][1][1]),
                             frobenius_distance_upper=float(froU), tolerance=float(tolG), within=froU <= tolG),
                   admitted=ok)
        res['states'].append(row)
        res['stop'] = res['stop'] or not ok
        print(f"{s['arm']:8s} g={s['g']} t={s['t']}: N2 in [{row['N2_lower'][:22]}, {row['N2_upper'][:22]}] |N2-1|={row['abs_N2_minus_1']:.3e} tol={row['tolerance_addendum']:.3e} "
              f"Gram dist<={row['gram']['frobenius_distance_upper']:.3e} tol={row['gram']['tolerance']:.3e} -> {'ADMITTED' if ok else 'FAIL'}", flush=True)
    json.dump(res, open(f'{OUTD}/cw18_admission.json', 'w'), indent=1)
    print('STOP' if res['stop'] else 'ALL ADMITTED')

if __name__ == '__main__':
    main()
