"""Second reader #492: one n = 5 arm (form, g) with the own builder r2_50 and own propagation.
Readouts per MOCK_n5_plan_v2 sec. 4-5 for the normalized PURE input B = |01>_H at site 0: admission (exact N2), p_LL, <N_L>, <N_H>,
f_L, f_H at t = 1..5; Bwall by the frozen 64-panels-per-unit midpoint rule with the exact joining source S = H_ext J - J H
(4 empty added sites, added clocks 1, filtered halo copied at both ends), C_join^2 = R(S^dag S), L = C_join R(H);
panel-term split; t*; gate; LL-band. Also compares the own H and states with the first reader's saved n5run files (md5-checked).
argv: form g out_json"""
import json, sys, os, time, hashlib, math, resource
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
form, g, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
t0 = time.process_time(); w0 = time.time()
n = 5; NE = 9
K = json.load(open("<internal>/r2_51_validate_n3.json"))["K"]
WIP_MD5 = {  # first reader's sec. 10 listing for the files compared here
    ("onsite", 0): ("2e2c477687fc864b53361d73841982a2", "315f1363b42a7264972ca4aa260ac56d"), ("onsite", 1): ("412e0e86f98fc3936f377e982b6204b3", "64b3337e846168d055755f2a7e867946"),
    ("onsite", 2): ("3523508b409f6eb8ea95f7e62c153da0", "22e100dc2be5bcfdc9a0d5d28a1518cd"), ("filtered", 0): ("2e2c477687fc864b53361d73841982a2", "315f1363b42a7264972ca4aa260ac56d"),
    ("filtered", 1): ("647b8f6ae05dd6710d31d2e8408548b5", "9b4ce2d24479d1ec14618ec7cb52daef"), ("filtered", 2): ("71fe5814e9884fb5b04d2c9b7f45ed20", "013a22c7d4e709f841b9d9c60ec3b4db")}
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
out = {"form": form, "g": g}
Bs, idx = M.basis(n); D = len(Bs)
H = M.hamiltonian(n, form, g, K)
out["herm_defect"] = float(np.abs(H - H.conj().T).max()); H = (H + H.conj().T) / 2
# compare with the first reader's saved Hamiltonian (named saved input)
d = "<internal>/%s_g%d" % (form, int(g))
hp, sp = os.path.join(d, "hamiltonian.npz"), os.path.join(d, "states_pure.npz")
mh, ms = md5(hp), md5(sp)
out["saved_md5_ok"] = [mh == WIP_MD5[(form, int(g))][0], ms == WIP_MD5[(form, int(g))][1]]
Hs = np.load(hp)["H"]; out["max_abs_H_minus_saved"] = float(np.abs(H - Hs).max())
# free part only (g = 0 Hamiltonian) for e_H bookkeeping and LL-band
w, V = np.linalg.eigh(H)
psi0 = np.zeros(D, complex); psi0[idx[(0, 1)]] = 1.0
c = V.conj().T @ psi0
res_norm = float(np.linalg.norm(H @ V - V * w, 2)); recon = float(np.linalg.norm(V @ c - psi0))
# e_H: non-rigorous estimate of ||H_binary64 - H_target||_2 (mass 3/5 and G rounding, assembly): bounded generously
e_H = 1e-13
def prop(t): return V @ (np.exp(-1j * w * t) * c)
def eps(t): return recon + t * res_norm * float(np.linalg.norm(c)) + t * e_H
bits = np.array([[(m >> b) & 1 for b in range(8 * n)] for m in [(1 << a) | (1 << b) for a, b in Bs]])
spec = np.array([(b % 8) // 4 for b in range(8 * n)]); site = np.array([b // 8 for b in range(8 * n)])
nL = bits[:, spec == 1].sum(1); nH = bits[:, spec == 0].sum(1)
nL_lead = bits[:, (spec == 1) & (site >= 1)].sum(1); nH_lead = bits[:, (spec == 0) & (site >= 1)].sum(1)
LL = np.where(nL == 2)[0]
saved_states = np.load(sp)
rows = []
for t in (1, 2, 3, 4, 5):
    psi = prop(float(t))
    N2 = sum((Fr(float(x.real)) ** 2 + Fr(float(x.imag)) ** 2 for x in psi), Fr(0))
    e = eps(t); en = 2 * e / math.sqrt(float(N2))
    p = np.abs(psi) ** 2 / float(N2)
    vals = {"pLL": float(p[LL].sum()), "NL": float(p @ nL), "NH": float(p @ nH), "NL_lead": float(p @ nL_lead), "NH_lead": float(p @ nH_lead)}
    iv_ = {"pLL": [vals["pLL"] - 2 * en, vals["pLL"] + 2 * en]}
    for k in ("NL", "NH", "NL_lead", "NH_lead"): iv_[k] = [vals[k] - 4 * en, vals[k] + 4 * en]
    def qdiv(a, b):
        if b[0] <= 0: return None
        cands = [a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1]]; return [min(cands), max(cands)]
    fL = qdiv(iv_["NL_lead"], iv_["NL"]); fH = qdiv(iv_["NH_lead"], iv_["NH"])
    dsv = float(np.linalg.norm(psi - saved_states["t%d" % t]))
    rows.append({"t": t, "N2": float(N2), "N2_dev": float(abs(N2 - 1)), "eps": e, "admitted": bool(N2 > 0 and abs(N2 - 1) <= (1 + Fr(e)) ** 2 - 1),
                 "vals": vals, "iv": iv_, "fL": fL, "fH": fH, "dist_to_saved_state": dsv})
out["rows"] = rows; out["eig_residual_2norm"] = res_norm; out["recon"] = recon
# ---- Bwall ----
Be, idxe = M.basis(NE)
He = M.hamiltonian(NE, form, g, K)
emb = np.array([idxe[(a, b)] for a, b in Bs])
S = He[:, emb].copy(); S[emb, :] -= H
out["S_rows_nonzero_outside_embedding"] = int(np.count_nonzero(np.abs(np.delete(S, emb, axis=0)).sum(1) > 0))
out["S_norm_on_embedded_rows"] = float(np.abs(S[emb]).max())  # should be ~0 (V_short is the restriction of V_long)
del He
Q = S.conj().T @ S
Cj2 = float(np.abs(Q).sum(1).max()); Cj = math.sqrt(Cj2)
Hb = float(np.abs(H).sum(1).max()); L = Cj * Hb
dpan = 1 / 64
fm = []
for kk in range(320):
    tm = (kk + 0.5) / 64; phi = prop(tm)
    fm.append(float(np.linalg.norm(S @ phi) / np.linalg.norm(phi)) + Cj * 2 * eps(tm))
fm = np.array(fm)
Bw = {}; split = {}
for t in (1, 2, 3, 4, 5):
    smp = float(dpan * fm[:64 * t].sum()); pan = 64 * t * L * dpan ** 2 / 4
    Bw[t] = smp + pan; split[t] = {"sampled": smp, "panel_term": pan}
out.update({"C_join": Cj, "C_join2": Cj2, "H_bound": Hb, "L": L, "Bwall": Bw, "split": split})
ts = [t for t in (1, 2, 3, 4, 5) if Bw[t] <= 0.1]
# last clock of the initial run with Bwall <= 0.1 (Bwall is nondecreasing)
tstar = None
for t in (1, 2, 3, 4, 5):
    if Bw[t] <= 0.1: tstar = t
    else: break
out["tstar"] = tstar
# ---- gate (frozen v2) ----
def gate():
    if tstar is None or tstar < 3: return "VOID", "t* < 3" if tstar else "no qualifying t*"
    r = {x["t"]: x for x in rows}; prev = r[tstar - 1]; cur = r[tstar]
    pl = prev["iv"]["pLL"]; nl = cur["iv"]["NL"]; nh = cur["iv"]["NH"]
    if pl[1] < 0.1 or nl[1] < 0.05 or nh[1] <= 0: return "VOID", "prerequisite certified below"
    if pl[0] < 0.1 or nl[0] < 0.05 or nh[0] <= 0: return "UNRESOLVED", "prerequisite straddles"
    fL, fH = cur["fL"], cur["fH"]
    if fL[0] >= 0.5 and fL[0] > fH[1]: return "PASS", "transport witness"
    if fL[1] < 0.5: return "FAIL", "upper f_L < 1/2"
    return "UNRESOLVED", "f_L straddles"
out["gate"] = list(gate()) if g == 1 else ["ungated"] + list(gate())
# ---- LL band ----
ell = np.linalg.eigvalsh(H[np.ix_(LL, LL)])
out["LL"] = {"dim": int(len(LL)), "min": float(ell.min()), "max": float(ell.max()), "BHB": float((psi0.conj() @ H @ psi0).real)}
out["cpu_s"] = time.process_time() - t0; out["wall_s"] = time.time() - w0
out["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
json.dump(out, open(OUT, "w"), indent=1, default=str)
print(form, g, "t*", tstar, out["gate"], {t: round(Bw[t], 9) for t in Bw}, "dH", out["max_abs_H_minus_saved"], "cpu", round(out["cpu_s"], 1))
