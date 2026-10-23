"""Second reader #495: exact dim(W cap {Pi_total = +-2}) and dim(W cap {Pi_total = 0}) by rank over F_p (two primes = 1 mod 4).
Pi = (-1)^x beta is diagonal in the basis and commutes with H0, so each Pi_total sector is H_H-invariant and
W cap sector = unobservable subspace of the restricted pair (H_H|sector, T|sector). Exact rational target (m_H = 3/5, exact G).
Builder r2_50 (exact path). argv: out_json"""
import json, sys, os
import numpy as np
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
OUT = sys.argv[1]
K = json.load(open("<internal>/r2_51_validate_n3.json"))["K"]
def primes():
    f = []; c = (1 << 26) - 3
    while len(f) < 2:
        if c % 4 == 1 and all(c % q for q in range(3, int(c ** 0.5) + 1, 2)): f.append(c); c -= 1000004
        c -= 2
    return f
def sqrtm1(p):
    for a in range(2, 10000):
        r = pow(a, (p - 1) // 4, p)
        if r * r % p == p - 1: return r
def modq(x, p): x = Fr(x); return (x.numerator % p) * pow(x.denominator % p, p - 2, p) % p
def pi_mode(m): return (-1) ** (m // 8) * (1 if (m % 4) < 2 else -1)
def unobs_dim(n, form, p, sector):
    ip = sqrtm1(p); Bs, idx = M.basis(n); sp = lambda m: (m % 8) // 4
    HH = [i for i, (a, b) in enumerate(Bs) if sp(a) == 0 and sp(b) == 0 and sector(pi_mode(a) + pi_mode(b))]
    LL = [i for i, (a, b) in enumerate(Bs) if sp(a) == 1 and sp(b) == 1]
    hp = {v: k for k, v in enumerate(HH)}; lp = {v: k for k, v in enumerate(LL)}; d = len(HH)
    Hm = np.zeros((d, d), dtype=np.int64); leak = 0
    for (i, j), (re, im) in M.dgamma(n, M.one_body(n, exact=True), exact=True).items():
        if j in hp:
            if i in hp: Hm[hp[i], hp[j]] = (Hm[hp[i], hp[j]] + modq(re, p) + modq(im, p) * ip) % p
            elif sp(Bs[i][0]) == 0 and sp(Bs[i][1]) == 0 and (re != 0 or im != 0): leak += 1
    T = np.zeros((len(LL), d), dtype=np.int64)
    for coef, Lv, Hv in M.converter_terms(n, form, K, exact=True):
        cq = modq(coef, p)
        for li, la in Lv.items():
            for hi, ha in Hv.items():
                if hi in hp: T[lp[li], hp[hi]] = (T[lp[li], hp[hi]] + cq * modq(la, p) % p * modq(ha, p)) % p
    rows = []; piv = []
    def add(Bk):
        a = 0
        for row in Bk:
            r = row % p
            for b, pc in zip(rows, piv):
                if r[pc]: r = (r - r[pc] * b) % p
            nz = np.nonzero(r)[0]
            if len(nz):
                pc = nz[0]; r = r * pow(int(r[pc]), p - 2, p) % p
                for k in range(len(rows)):
                    if rows[k][pc]: rows[k] = (rows[k] - rows[k][pc] * r) % p
                rows.append(r); piv.append(pc); a += 1
        return a
    def mm(A, B):
        lo = B & 0x1FFF; hi = B >> 13
        return ((A @ lo) % p + ((A @ hi) % p) * 8192) % p
    blk = T.copy()
    for j in range(d + 1):
        if add(blk) == 0 and j > 0: break
        blk = mm(blk, Hm)
    return d, d - len(rows), leak
out = {"primes": primes(), "rows": []}
for n in (3, 5):
    for form in ("onsite", "filtered"):
        r = {"n": n, "form": form}
        for lab, sec in (("pm2", lambda q: abs(q) == 2), ("zero", lambda q: q == 0), ("all", lambda q: True)):
            res = [unobs_dim(n, form, p, sec) for p in out["primes"]]
            r[lab] = {"sector_dim": res[0][0], "dimW_cap": [x[1] for x in res], "H0_leak_out_of_sector": res[0][2]}
        out["rows"].append(r); print(r, flush=True)
json.dump(out, open(OUT, "w"), indent=1)
