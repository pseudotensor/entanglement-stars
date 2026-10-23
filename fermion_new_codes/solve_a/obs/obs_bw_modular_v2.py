#!/usr/bin/env python3
"""obs_bw_modular_v2.py (builder analyst; SPEC_final4b_observables.md sec. 8.2, J2's decisive test; the chain, the profiles and the chart
functions are obs_bw_modular_v1's, imported unchanged).

One normalization: the boost operators B (unscaled: obs_bw_modular_v1's b divided by 2 pi), beta their eigenvalues; one test subspace for
every comparator: P_test = the eigenvectors of the PREDICTED B restricted to A with |beta| < --beta-max (default 3). Per operator B
(prediction, uniform boost, naive flowing boost): R_in = ||P (k_A - 2 pi B) P||/||P 2 pi B P||, R_out = ||(k_A - 2 pi B) P||/||2 pi B P||, the
Fermi-Dirac deviation of the occupations on B's own eigenvectors with |beta| < beta_max (max and rms of |<phi|C_A|phi> - 1/(e^(2 pi beta) + 1)|),
and the commutator ||[C_A, B] P||/||C_A B P||. The separation (R_in(uniform) - R_in(prediction))/R_in(v = 0) at the same L and m_c is
printed; it is decisive when its modulus exceeds 1 with the same sign at the two largest L (the reading is the lead's; this script prints).
Sizes and profiles per the spec: L in {4096, 8192}, W in {128, 256, 512}, the profiles (v0, v1) = (0, 0.6) and (-0.5, 0.3), the cut at the
centre and at +W, m_c = 0 and 0.05 (the massive rows are deviation measurements, labelled), plus the v = 0 reference per (L, m_c).
--gpu runs the dense eigendecompositions in torch on that GPU (complex128); without it scipy on the host threads.
Args: --L --W --mc --profiles --beta-max --workers --gpu --out-json --md."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_bw_modular_v1 as B1  # noqa: E402

GPU = {"dev": None}


def heigh(M):
    if GPU["dev"] is not None:
        import torch
        a = torch.as_tensor(M, dtype=torch.complex128, device="cuda:%d" % GPU["dev"])
        e, U = torch.linalg.eigh(a)
        return e.real.cpu().numpy(), U.cpu().numpy()
    return eigh(M, driver="evr", check_finite=False)


def fro(M):
    return float(np.linalg.norm(M))


def analyse(kA, CA, B, P, beta_max):
    twopiB = 2 * math.pi * B
    D = kA - twopiB
    num_in = fro(P.conj().T @ D @ P); den_in = fro(P.conj().T @ twopiB @ P)
    num_out = fro(D @ P); den_out = fro(twopiB @ P)
    beta, phi = heigh(B)
    sel = np.abs(beta) < beta_max
    Q = phi[:, sel]
    occ = np.real(np.einsum("in,ij,jn->n", Q.conj(), CA, Q))
    fd = 1.0 / (np.exp(np.clip(2 * math.pi * beta[sel], -700, 700)) + 1.0)
    dev = np.abs(occ - fd)
    comm = fro((CA @ B - B @ CA) @ P) / max(fro(CA @ B @ P), 1e-300)
    return {"R_in": num_in / max(den_in, 1e-300), "R_out": num_out / max(den_out, 1e-300), "FD_max": float(dev.max()) if sel.any() else None,
            "FD_rms": float(np.sqrt(np.mean(dev ** 2))) if sel.any() else None, "commutator": comm, "n_own": int(sel.sum())}


def run_case(L, mc, spec, label, beta_max):
    t0 = time.time()
    vfun = B1.make_vfun(spec)
    x = np.arange(L, dtype=float)
    xc = L / 2 - 0.5
    v = np.asarray(vfun(x), float)
    h, h0, b_pred, b_unif, b_naive = B1.boost_operators(L, mc, v, xc, vfun=vfun)
    e, U = heigh(h)
    V_ = U[:, e < 0]
    C = V_ @ V_.conj().T
    a = L
    CA = C[a:, a:]
    lam, W = heigh(CA)
    lam = np.clip(lam, 1e-15, 1 - 1e-15)
    kA = (W * np.log((1 - lam) / lam)) @ W.conj().T
    Bs = {"pred": b_pred[a:, a:] / (2 * math.pi), "uniform": b_unif[a:, a:] / (2 * math.pi), "naive": b_naive[a:, a:] / (2 * math.pi)}
    bp, phip = heigh(Bs["pred"])
    P = phip[:, np.abs(bp) < beta_max]
    out = {"label": label, "L": L, "mc": mc, "n_test": int(P.shape[1]), "kA_eig_clipped": int(np.sum((lam <= 1e-15) | (lam >= 1 - 1e-15)))}
    for name, Bm in Bs.items():
        out[name] = analyse(kA, CA, Bm, P, beta_max)
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--W", type=float, nargs="+", default=[128.0, 256.0, 512.0])
    ap.add_argument("--mc", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--profiles", default="0:0.6,-0.5:0.3")
    ap.add_argument("--beta-max", type=float, default=3.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    GPU["dev"] = a.gpu
    profs = [tuple(float(y) for y in p.split(":")) for p in a.profiles.split(",")]
    res = []
    for L in a.L:
        for mc in a.mc:
            ref = run_case(L, mc, ("uniform", 0.0), "v = 0 (reference)", a.beta_max)
            res.append(ref)
            print(json.dumps({k: ref[k] for k in ("label", "L", "mc", "n_test")}), {n: "%.3e" % ref[n]["R_in"] for n in ("pred", "uniform", "naive")}, "%.0f s" % ref["seconds"], flush=True)
            R0 = ref["uniform"]["R_in"]
            for (v0, v1) in profs:
                for W in a.W:
                    for shift in (0.0, W):
                        xp = L / 2 - 0.5 - shift
                        r = run_case(L, mc, ("tanh", v0, v1, W, xp), "tanh v0 %g v1 %g W %g, cut at centre + %g" % (v0, v1, W, shift), a.beta_max)
                        r["separation_over_R0"] = (r["uniform"]["R_in"] - r["pred"]["R_in"]) / max(R0, 1e-300)
                        r["R0"] = R0
                        res.append(r)
                        print(r["label"], L, mc, {n: "%.3e" % r[n]["R_in"] for n in ("pred", "uniform", "naive")}, "sep/R0 %.3f" % r["separation_over_R0"], "%.0f s" % r["seconds"], flush=True)
                        json.dump({"script": "obs_bw_modular_v2.py", "args": vars(a), "cases": res}, open(a.out_json, "w"), indent=1, default=float)
    L_ = ["## J2 version 2: the decisive Bisognano-Wichmann test with the flow (obs_bw_modular_v2.py; one normalization, one test subspace)", "",
          "| case | L | m_c | n_test | R_in pred | R_in uniform | R_in naive | R_out pred | R_out uniform | commutator pred | commutator uniform | FD max pred | (R_unif - R_pred)/R_0 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L_.append("| %s | %d | %g | %d | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %s | %s |" % (
            r["label"], r["L"], r["mc"], r["n_test"], r["pred"]["R_in"], r["uniform"]["R_in"], r["naive"]["R_in"], r["pred"]["R_out"], r["uniform"]["R_out"],
            r["pred"]["commutator"], r["uniform"]["commutator"], ("%.2e" % r["pred"]["FD_max"]) if r["pred"]["FD_max"] is not None else "-",
            ("%.3f" % r["separation_over_R0"]) if "separation_over_R0" in r else "-"))
    L_ += ["", "- every number above is printed by the script; the massive rows are deviation measurements; decisive: |separation| > 1 with the same sign at the two largest L."]
    open(a.md, "w").write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
