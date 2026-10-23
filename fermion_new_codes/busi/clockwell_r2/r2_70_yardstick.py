"""Second reader #497 Step 1: free-phase yardstick, own route. Own builder r2_50 (equal to the saved n = 5 matrices) and own
eigendecomposition of H0. Per form:
  V = H1 - H0 (linearity H2 - H1 = H1 - H0 checked on own and saved matrices);
  psi1(t) = -i int_0^t U0(t-s) V U0(s) B ds  (closed form in the H0 eigenbasis; checked by 64-node Gauss-Legendre quadrature);
  f_L1(t) = <N_L on sites 1-4>[psi1] / <N_L>[psi1], with <N_L>[psi1] = 2 ||psi1||^2 (psi1 is LL only);
  Delta = f_L(3) - f_L1(3) with f_L(3) at g = 1 from the own n5 run (r2_52);
  first-order wall bound: e1 = psi1_ext - J psi1 solves i e1' = H0_ext e1 + S0 psi1 + S_V psi0 + V_ext e0,  e0 = psi0_ext - J psi0,
  so ||e1(t)|| <= int_0^t ||S0 psi1 + S_V psi0|| ds + ||V_ext e0|| integrated, ||e0(s)|| <= E0(s) = int_0^s ||S0 psi0|| (+ quadrature);
  variants (A) the first reader's construction (separate source norms, Schur ||V_ext||, factor 2/||psi1||),
           (B) + combined source norm, (C) + spectral norm of the HH->LL block of V_ext (e0 is HH, V_ext e0 is LL),
           (D) + Dunkl-Williams normalization ||a/|a| - b/|b||| <= 2||a - b||/(|a| + |b|) <= 2E/(2|b| - E).
argv: form P out_json"""
import json, sys, os, time, math, resource, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
form, P, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t0 = time.process_time(); w0 = time.time()
K = json.load(open("<internal>/r2_51_validate_n3.json"))["K"]
n = 5; Bs, idx = M.basis(n); D = len(Bs)
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
out = {"form": form, "P": P}
H0 = M.hamiltonian(n, form, 0.0, K); H1 = M.hamiltonian(n, form, 1.0, K); H2 = M.hamiltonian(n, form, 2.0, K)
out["own_linearity"] = float(np.abs(H2 - 2 * H1 + H0).max())
WIP = {"onsite": ("2e2c477687fc864b53361d73841982a2", "412e0e86f98fc3936f377e982b6204b3", "3523508b409f6eb8ea95f7e62c153da0"),
       "filtered": ("2e2c477687fc864b53361d73841982a2", "647b8f6ae05dd6710d31d2e8408548b5", "71fe5814e9884fb5b04d2c9b7f45ed20")}
sv = []
for g, m_ in zip((0, 1, 2), WIP[form]):
    p = "<internal>/hamiltonian.npz" % (form, g); sv.append(np.load(p)["H"]); out.setdefault("saved_md5_ok", []).append(md5(p) == m_)
out["saved_linearity"] = float(np.abs(sv[2] - 2 * sv[1] + sv[0]).max()); out["own_vs_saved"] = float(max(np.abs(a - b).max() for a, b in zip((H0, H1, H2), sv)))
V = H1 - H0
w, V0 = np.linalg.eigh(H0)
B = np.zeros(D, complex); B[idx[(0, 1)]] = 1
c = V0.conj().T @ B; Vt = V0.conj().T @ V @ V0
out["eig_res"] = float(np.linalg.norm(H0 @ V0 - V0 * w)); out["VB2"] = float(np.linalg.norm(V @ B) ** 2)
dw = w[:, None] - w[None, :]
def psi1(t):
    with np.errstate(divide="ignore", invalid="ignore"):
        Km = np.where(np.abs(dw) > 1e-12, (np.exp(1j * dw * t) - 1) / (1j * dw), t)
    return -1j * V0 @ (np.exp(-1j * w * t) * ((Vt * Km) @ c))
def psi0(s): return V0 @ (np.exp(-1j * w * s) * c)
def psi1_quad(t, nn=64):
    x, wt = np.polynomial.legendre.leggauss(nn); s = (x + 1) * t / 2; acc = np.zeros(D, complex)
    for si, wi in zip(s, wt):
        acc += wi * (V0 @ (np.exp(-1j * w * (t - si)) * (V0.conj().T @ (V @ psi0(si)))))
    return -1j * acc * t / 2
bits = np.array([[(m >> b) & 1 for b in range(40)] for m in [(1 << a) | (1 << b) for a, b in Bs]])
spc = np.array([(b % 8) // 4 for b in range(40)]); ste = np.array([b // 8 for b in range(40)])
nL = bits[:, spc == 1].sum(1); nLlead = bits[:, (spc == 1) & (ste >= 1)].sum(1); nH = bits[:, spc == 0].sum(1)
rows = {}
for t in (1, 2, 3):
    p1 = psi1(float(t)); q = psi1_quad(float(t)); a2 = np.abs(p1) ** 2; nrm2 = float(a2.sum())
    rows[t] = {"norm2": nrm2, "quad_check": float(np.linalg.norm(p1 - q) / np.linalg.norm(p1)), "NL_over_norm2": float(a2 @ nL) / nrm2,
               "NH": float(a2 @ nH), "fL1": float(a2 @ nLlead) / float(a2 @ nL), "LL_only": float(a2[nL != 2].sum())}
out["rows"] = rows
# ---- wall bound on psi1 ----
He0 = M.hamiltonian(9, "onsite" if form == "onsite" else "filtered", 0.0, K); He1 = M.hamiltonian(9, form, 1.0, K)
Be, idxe = M.basis(9); emb = np.array([idxe[(a, b)] for a, b in Bs])
S0 = He0[:, emb].copy(); S0[emb, :] -= H0
S1 = He1[:, emb].copy(); S1[emb, :] -= H1
SV = S1 - S0; Vext = He1 - He0; del He1
schur = lambda A: float(np.abs(A).sum(1).max())
C0 = math.sqrt(schur(S0.conj().T @ S0)); nSV = math.sqrt(schur(SV.conj().T @ SV)); nVext = schur(Vext); nV = schur(V); nH0 = schur(H0)
spe = np.array([(b % 8) // 4 for b in range(72)])
HHe = [i for i, (a, b) in enumerate(Be) if spe[a] == 0 and spe[b] == 0]; LLe = [i for i, (a, b) in enumerate(Be) if spe[a] == 1 and spe[b] == 1]
Text = Vext[np.ix_(LLe, HHe)]; nT2 = float(np.linalg.norm(Text, 2))
out["norms"] = {"C0": C0, "S_V": nSV, "Vext_schur": nVext, "Vext_HH_to_LL_spectral": nT2, "V_schur": nV, "H0_schur": nH0}
L0 = C0 * schur(H0)        # Lipschitz constant of s -> ||S0 psi0(s)|| (normalized psi0)
del He0, Vext
d = 1.0 / P; T = 3.0; N = int(T * P)
mids = (np.arange(N) + 0.5) * d
fa = np.empty(N); fc = np.empty(N); fcomb = np.empty(N); f0 = np.empty(N)
for k, s in enumerate(mids):
    p0 = psi0(s); p1 = psi1(s)
    a = S0 @ p1; b = SV @ p0
    fa[k] = np.linalg.norm(a); fc[k] = np.linalg.norm(b); fcomb[k] = np.linalg.norm(a + b); f0[k] = np.linalg.norm(S0 @ p0)
# Lipschitz constants on [0, 3] (derivative bounds): ||psi1'|| <= ||H0|| ||psi1|| + ||V|| <= ||V|| (||H0|| s + 1)
La = C0 * nV * (nH0 * T + 1); Lc = nSV * nH0; Lcomb = La + Lc
def prefix(f, L, upto):
    k = int(round(upto * P)); return float(d * f[:k].sum()) + upto * L * d / 4
# E0(s) <= int_0^s ||S0 psi0|| (zero-order Duhamel, normalized psi0; C_join eps ~1e-11 omitted). For s in panel k the integrand is
# nonnegative, so E0(s) <= E0bar_k = (sum of panels 0..k) + (k+1) L0 d^2/4; hence int_0^t E0 <= sum_k d E0bar_k (no further quadrature).
cum0 = np.cumsum(d * f0)
E0bar = cum0 + (np.arange(N) + 1) * L0 * d * d / 4
res = {}
for t in (1, 2, 3):
    k = int(round(t * P)); n1 = math.sqrt(rows[t]["norm2"])
    partA = float(d * fa[:k].sum()) + t * La * d / 4; partC = float(d * fc[:k].sum()) + t * Lc * d / 4
    partComb = float(d * fcomb[:k].sum()) + t * Lcomb * d / 4
    intE0 = float(d * E0bar[:k].sum())
    E1_A = partA + partC + nVext * intE0
    E1_B = partComb + nVext * intE0
    E1_C = partComb + nT2 * intE0
    res[t] = {"E1_A": E1_A, "E1_B": E1_B, "E1_C": E1_C, "norm_psi1": n1,
              "Bwall1_A": 2 * E1_A / n1, "Bwall1_B": 2 * E1_B / n1, "Bwall1_C": 2 * E1_C / n1,
              "Bwall1_D_from_A": 2 * E1_A / (2 * n1 - E1_A), "Bwall1_D_from_C": 2 * E1_C / (2 * n1 - E1_C),
              "parts_A_normalized": {"S0psi1": 2 * partA / n1, "SVpsi0": 2 * partC / n1, "Vext_E0": 2 * nVext * intE0 / n1,
                                     "quadrature": 2 * (t * La * d / 4 + t * Lc * d / 4) / n1},
              "intE0": intE0}
out["wall"] = res; out["La"] = La; out["Lc"] = Lc; out["L0"] = L0
out["cpu_s"] = time.process_time() - t0; out["wall_s"] = time.time() - w0; out["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
json.dump(out, open(OUT, "w"), indent=1, default=str)
print(form, P, {t: (round(r["fL1"], 10), r["NL_over_norm2"], r["norm2"]) for t, r in rows.items()})
print({t: {k: round(v, 6) for k, v in r.items() if k.startswith("Bwall")} for t, r in res.items()}, out["norms"], round(out["cpu_s"], 1))
