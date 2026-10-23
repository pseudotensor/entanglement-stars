"""Second reader #492 (reviewer A A): exactly dark subspace W of the heavy N = 2 sector = largest H_H-invariant subspace of ker T,
T = P_LL V P_HH, H_H = P_HH H0 P_HH (g = 1).  Two independent routes, no relative rank threshold:
 (1) exact dimension: rank over F_p (two primes p = 1 mod 4, i -> sqrt(-1) mod p) of the block Krylov row space
     span{T H_H^j}, from the exact rational target (m_H = 3/5, exact G); dim W = d_H - rank. (Rank mod p <= rank over Q(i);
     agreement of two unrelated primes is the robustness check.)
 (2) float spectral route: W = sum over eigenspaces E of H_H of ker(T|E); eigenvalue clusters and per-cluster null spaces with
     their gaps printed. Projections ||P_W B||, ||P_W B'||, ||P_W D|| and ||[Gamma(r), H]|| (full H, g = 1).
Sources: own builder r2_50 at n = 3, 5; and icloud's saved n = 3 Hamiltonians (float route only). argv: out_json"""
import json, sys, os, time, math
import numpy as np
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
OUT = sys.argv[1]
K = json.load(open("<internal>/r2_51_validate_n3.json"))["K"]
def _primes():
    found = []; c = (1 << 26) - 3
    while len(found) < 2:
        if c % 4 == 1 and all(c % q for q in range(3, int(c ** 0.5) + 1, 2)): found.append(c); c -= 1000004
        c -= 2
    return found
PRIMES = _primes()  # two primes = 1 mod 4 below 2^26 (checked below)
def is_prime(p):
    if p < 2: return False
    i = 2
    while i * i <= p:
        if p % i == 0: return False
        i += 1
    return True
def sqrtm1(p):
    for a in range(2, 1000):
        r = pow(a, (p - 1) // 4, p)
        if r * r % p == p - 1: return r
def modq(x, p): return (x.numerator % p) * pow(x.denominator % p, p - 2, p) % p

def sectors(n):
    Bs, idx = M.basis(n)
    bits = [[(1 << a) | (1 << b), a, b] for a, b in Bs]
    sp = lambda m: (m % 8) // 4
    HH = [i for i, (a, b) in enumerate(Bs) if sp(a) == 0 and sp(b) == 0]
    LL = [i for i, (a, b) in enumerate(Bs) if sp(a) == 1 and sp(b) == 1]
    return Bs, idx, HH, LL

def exact_rank(n, form, p):
    i_p = sqrtm1(p)
    Bs, idx, HH, LL = sectors(n); hpos = {v: k for k, v in enumerate(HH)}; lpos = {v: k for k, v in enumerate(LL)}
    dH = len(HH)
    Hm = np.zeros((dH, dH), dtype=np.int64)
    for (i, j), (re, im) in M.dgamma(n, M.one_body(n, exact=True), exact=True).items():
        if i in hpos and j in hpos:
            Hm[hpos[i], hpos[j]] = (Hm[hpos[i], hpos[j]] + modq(re, p) + modq(im, p) * i_p) % p
    T = np.zeros((len(LL), dH), dtype=np.int64)
    for coef, Lv, Hv in M.converter_terms(n, form, K, exact=True):
        cq = modq(Fr(coef), p)
        for li, la in Lv.items():
            for hi, ha in Hv.items():
                T[lpos[li], hpos[hi]] = (T[lpos[li], hpos[hi]] + cq * modq(Fr(la), p) % p * modq(Fr(ha), p)) % p
    basis_rows = []; piv = []
    def add_rows(Bk):
        added = 0
        for row in Bk:
            r = row.copy() % p
            for b, pc in zip(basis_rows, piv):
                if r[pc]: r = (r - r[pc] * b) % p
            nz = np.nonzero(r)[0]
            if len(nz):
                pc = nz[0]; r = r * pow(int(r[pc]), p - 2, p) % p
                for k in range(len(basis_rows)):           # keep reduced form
                    if basis_rows[k][pc]: basis_rows[k] = (basis_rows[k] - basis_rows[k][pc] * r) % p
                basis_rows.append(r); piv.append(pc); added += 1
        return added
    def matmul_mod(A, Bm):  # entries < 2^26: split to avoid int64 overflow
        lo = Bm & 0x1FFF; hi = Bm >> 13
        return ((A @ lo) % p + ((A @ hi) % p) * 8192) % p
    blk = T.copy(); hist = []
    for j in range(dH + 1):
        a = add_rows(blk); hist.append(a)
        if a == 0 and j > 0: break
        blk = matmul_mod(blk, Hm)
    return dH, len(basis_rows), hist

def float_route(H0, V, n_or_None, HH, LL, vecs):
    Hh = H0[np.ix_(HH, HH)]; T = V[np.ix_(LL, HH)]
    w, U = np.linalg.eigh(Hh)
    clusters = []; start = 0
    for k in range(1, len(w) + 1):
        if k == len(w) or w[k] - w[k - 1] > 1e-9:
            clusters.append((start, k)); start = k
    Wb = []; kept_min = np.inf; dropped_max = 0.0; gap_eig = min((w[b] - w[b - 1]) for (_, b), (a2, _) in zip(clusters[:-1], clusters[1:])) if len(clusters) > 1 else None
    min_cluster_gap = min(w[c2[0]] - w[c1[1] - 1] for c1, c2 in zip(clusters[:-1], clusters[1:]))
    max_cluster_width = max(w[b - 1] - w[a] for a, b in clusters)
    scale = np.linalg.norm(T, 2)
    for a, b in clusters:
        Ue = U[:, a:b]; s_full = np.linalg.svd(T @ Ue, compute_uv=False)
        _, s, Vh = np.linalg.svd(T @ Ue)
        sv = np.zeros(b - a); sv[:len(s)] = s
        null = [k for k in range(b - a) if sv[k] <= 1e-10 * scale]
        for k in range(b - a):
            if sv[k] <= 1e-10 * scale: dropped_max = max(dropped_max, sv[k] / scale)
            else: kept_min = min(kept_min, sv[k] / scale)
        if null: Wb.append(Ue @ Vh.conj().T[:, null])
    Wm = np.hstack(Wb) if Wb else np.zeros((len(HH), 0))
    Qw, _ = np.linalg.qr(Wm) if Wm.shape[1] else (Wm, None)
    proj = {k: float(np.linalg.norm(Qw.conj().T @ v)) for k, v in vecs.items()}
    return {"dimW": int(Wm.shape[1]), "clusters": len(clusters), "max_cluster_width": float(max_cluster_width), "min_cluster_gap": float(min_cluster_gap),
            "kept_min_rel_sv": float(kept_min), "dropped_max_rel_sv": float(dropped_max), "proj": proj,
            "TW_residual": float(np.linalg.norm(T @ Qw)) if Wm.shape[1] else 0.0, "HW_invariance": float(np.linalg.norm(Hh @ Qw - Qw @ (Qw.conj().T @ Hh @ Qw))) if Wm.shape[1] else 0.0}

def gamma_r(n):
    Bs, idx = M.basis(n); D = len(Bs); G = np.zeros((D, D))
    r = lambda m: m ^ 1  # c = 2u + s, sigma_x on s: 0<->1, 2<->3
    for j, (a, b) in enumerate(Bs):
        ra, rb = r(a), r(b)
        s = 1 if ra < rb else -1
        G[idx[(min(ra, rb), max(ra, rb))], j] = s
    return G

def heavy_vecs(n, HH):
    Bs, idx = M.basis(n); hpos = {v: k for k, v in enumerate(HH)}
    def e(a, b):
        v = np.zeros(len(HH), complex); v[hpos[idx[(a, b)]]] = 1; return v
    s = 1 / math.sqrt(2)
    return {"B": e(0, 1), "Bprime": s * (e(0, 3) + e(1, 2)), "D": s * (e(0, 3) - e(1, 2))}

out = {"primes": [[p, is_prime(p), p % 4] for p in PRIMES], "rows": []}
t0 = time.process_time()
for n in (3, 5):
    for form in ("onsite", "filtered"):
        Bs, idx, HH, LL = sectors(n)
        H0 = M.hamiltonian(n, form, 0, K); H1 = M.hamiltonian(n, form, 1, K); V = H1 - H0
        ex = [exact_rank(n, form, p) for p in PRIMES]
        fr = float_route(H0, V, n, HH, LL, heavy_vecs(n, HH))
        Gm = gamma_r(n)
        comm = {"H": float(np.abs(Gm @ H1 - H1 @ Gm).max()), "H0": float(np.abs(Gm @ H0 - H0 @ Gm).max()), "V": float(np.abs(Gm @ V - V @ Gm).max())}
        out["rows"].append({"source": "own n=%d %s" % (n, form), "n": n, "form": form, "dH": len(HH), "n2n1": n * (2 * n - 1),
                            "exact": [{"p": p, "rank": r, "dimW": dH - r, "krylov_new_rows": h} for p, (dH, r, h) in zip(PRIMES, ex)],
                            "float": fr, "comm_Gamma_r": comm})
        print(out["rows"][-1]["source"], [x["dimW"] for x in out["rows"][-1]["exact"]], fr["dimW"], fr["proj"], comm, flush=True)
# saved icloud n = 3 Hamiltonians, float route
B = "<internal>/solve_b_bulk"
for form, d0, d1 in (("onsite", B + "/MOCK_unitary_lead_run_v1/even_n3_g0_v1", B + "/MOCK_unitary_lead_run_v1/even_n3_g1_v1"),
                     ("filtered", B + "/MOCK_filtered_physical_v1/even_n3_g0_v1", B + "/MOCK_filtered_physical_v1/even_n3_g1_v1")):
    Hs = []
    for d in (d0, d1):
        for l in open(os.path.join(d, "written_report.md")):
            if '"kind":"numerical_action"' in l[:40]:
                r = json.loads(l)["data"]
                if r.get("kind") == "operator":
                    z = np.load(os.path.join(d, r["hamiltonian"]["array"])); Hs.append(z[z.files[0]]); break
    Bs, idx, HH, LL = sectors(3)
    fr = float_route(Hs[0], Hs[1] - Hs[0], 3, HH, LL, heavy_vecs(3, HH))
    Gm = gamma_r(3)
    out["rows"].append({"source": "saved icloud n=3 %s" % form, "n": 3, "form": form, "dH": len(HH), "n2n1": 15, "exact": None, "float": fr,
                        "comm_Gamma_r": {"H": float(np.abs(Gm @ Hs[1] - Hs[1] @ Gm).max())}})
    print(out["rows"][-1]["source"], fr["dimW"], fr["proj"], flush=True)
out["cpu_s"] = time.process_time() - t0
json.dump(out, open(OUT, "w"), indent=1, default=str)
