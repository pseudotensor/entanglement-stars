"""Second reader #492: extract K from icloud's saved n = 3 on-site Hamiltonians and validate the own builder (r2_50)
against every saved n = 3 Hamiltonian (both forms, g = 0, 1/2, 1, 2). Read-only on inputs. argv: out_json"""
import json, sys, os, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
OUT = sys.argv[1]
B = "<internal>/solve_b_bulk"
DIRS = {("onsite", g): "%s/MOCK_unitary_lead_run_v1/even_n3_g%s_v1" % (B, gs) for g, gs in ((0, "0"), (0.5, "0p5"), (1, "1"), (2, "2"))}
DIRS.update({("filtered", g): "%s/MOCK_filtered_physical_v1/even_n3_g%s_v1" % (B, gs) for g, gs in ((0, "0"), (0.5, "0p5"), (1, "1"), (2, "2"))})
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
man = {}
for l in open("<internal>/INPUT_MD5SUMS.txt"):
    p = l.split()
    if len(p) == 2: man[p[1]] = p[0]
res = {"md5": {}}
saved = {}; keys = None
for k, d in DIRS.items():
    rp = os.path.join(d, "written_report.md"); res["md5"][rp] = [md5(rp), man.get(rp)]
    H = None
    for l in open(rp):
        if '"kind":"numerical_action"' in l[:40]:
            r = json.loads(l)["data"]
            if r.get("kind") == "operator":
                ap = os.path.join(d, r["hamiltonian"]["array"]); res["md5"][ap] = [md5(ap), man.get(ap)]
                z = np.load(ap); H = z[z.files[0]]; break
    for l in open(rp):
        if '"kind":"coherent_conversion"' in l[:40]:
            keys = [tuple(x) for x in json.loads(l)["data"]["density"]["row_keys"]]; break
    saved[k] = H
Bs, idx = M.basis(3)
res["basis_order_equal_saved_row_keys"] = [m for _, m in keys] == [(1 << a) | (1 << b) for a, b in Bs]
# K from the site-1 (N = 1) block of g = 1 minus g = 0 (on-site)
dH = saved[("onsite", 1)] - saved[("onsite", 0)]
Lv = [M.pair_vec(3, 1, s, {1: 1}) for s in M.PAIRS]; Hv = [M.pair_vec(3, 0, s, {1: 1}) for s in M.PAIRS]
def vec(d):
    v = np.zeros(len(Bs))
    for i, a in d.items(): v[i] = a
    return v
K = np.array([[vec(Lv[a]) @ dH @ vec(Hv[b]) for b in range(6)] for a in range(6)])
Kr = np.round(K.real).astype(int)
res["K"] = Kr.tolist(); res["K_integer_dev"] = float(np.abs(K - Kr).max())
Kl = Kr.tolist()
res["K_facts"] = {"col03_eq_col12": [r[2] for r in Kl] == [r[3] for r in Kl], "symmetric": bool((Kr == Kr.T).all()), "nonzero": int((Kr != 0).sum())}
res["compare"] = {}
for (form, g), Hs in saved.items():
    Hm = M.hamiltonian(3, form, g, Kl)
    res["compare"]["%s g%s" % (form, g)] = float(np.abs(Hm - Hs).max())
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "md5"}, indent=0))
print("md5 mismatches:", [p for p, (a, b) in res["md5"].items() if b is not None and a != b], "unmanifested:", sum(1 for a, b in res["md5"].values() if b is None))
