#!/usr/bin/env python3
"""b3x_kept_diff_v1.py (solve, page_verify B3 readout): where the continuation's kept records (T <= the restart) differ from the reference
run's (record json, key by key; fields npz, array by array per record). Reads only.
    python3 b3x_kept_diff_v1.py CONT_PREFIX REF_PREFIX T_RES OUT.json"""
import json, sys
import numpy as np

def load(p):
    J = json.load(open(p + '.json')); Z = np.load(p + '_fields.npz', allow_pickle=True)
    return J, Z

def main():
    pc, pr, tres, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    JC, ZC = load(pc); JR, ZR = load(pr)
    rs = float(JC['args']['rs'])
    rc = {round(float(r['T']) / rs, 6): r for r in JC['records'] if isinstance(r, dict) and 'T' in r}
    rr = {round(float(r['T']) / rs, 6): r for r in JR['records'] if isinstance(r, dict) and 'T' in r}
    common = sorted(t for t in rc if t <= tres + 1e-9 and t in rr)
    diffs = []
    keys_count = {}
    for t in common:
        a, b = rc[t], rr[t]
        ks = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True, default=str) != json.dumps(b.get(k), sort_keys=True, default=str))
        if ks:
            ent = {'T_over_rs': t, 'keys': {}}
            for k in ks:
                keys_count[k] = keys_count.get(k, 0) + 1
                va, vb = a.get(k), b.get(k)
                ent['keys'][k] = {'cont': va if not isinstance(va, (dict, list)) or len(json.dumps(va, default=str)) < 300 else 'len %d' % len(json.dumps(va, default=str)),
                                  'ref': vb if not isinstance(vb, (dict, list)) or len(json.dumps(vb, default=str)) < 300 else 'len %d' % len(json.dumps(vb, default=str))}
            diffs.append(ent)
    # fields npz per array over the common record clocks
    TC = np.asarray(ZC['T'], float); TR = np.asarray(ZR['T'], float)
    fd = {}
    for key in ZR.files:
        if key not in ZC.files:
            fd[key] = 'missing in cont'; continue
        A, B = np.asarray(ZC[key]), np.asarray(ZR[key])
        if A.ndim >= 1 and B.ndim >= 1 and A.shape[0] == len(TC) and B.shape[0] == len(TR) and A.dtype.kind in 'fc':
            n = int(np.sum(TR <= tres * rs + 1e-9))
            ia = [int(np.argmin(np.abs(TC - t))) for t in TR[:n]]
            d = np.abs(A[ia] - B[:n]) if A.shape[1:] == B.shape[1:] else None
            fd[key] = None if d is None else {'records': n, 'max_abs': float(np.max(d)) if d.size else 0.0, 'n_records_differing': int(np.sum(np.any(d.reshape(n, -1) != 0, axis=1)))}
        else:
            fd[key] = {'equal': bool(A.shape == B.shape and np.array_equal(A, B))} if A.dtype.kind in 'fciub' and B.dtype.kind in 'fciub' else {'equal': str(A) == str(B)}
    res = {'cont': pc, 'ref': pr, 'T_res': tres, 'records_compared': len(common), 'records_differing': len(diffs), 'keys_count': keys_count, 'diffs': diffs, 'fields': fd}
    json.dump(res, open(out, 'w'), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ('records_compared', 'records_differing', 'keys_count')}, default=str))
    print('T of differing records:', [d['T_over_rs'] for d in diffs])
    print('fields:', json.dumps(fd, default=str)[:3000])

if __name__ == '__main__':
    main()
