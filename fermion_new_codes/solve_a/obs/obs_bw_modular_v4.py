#!/usr/bin/env python3
"""obs_bw_modular_v4.py (builder analyst; J2 version 4 per SPEC_final4b_observables.md sec. 13.3, after red team R24e; the chain, the profiles
and the boost operators are obs_bw_modular_v1's, imported unchanged).

The deciding measurement on common probes: smooth wavepackets phi on the half-line A (x > the cut at L/2 - 1/2), a Gaussian envelope of width
sigma = W/4 centred at the distance d in {W/2, W, 2W} from the cut, the carrier at either valley of the sine chain (k = q and k = pi + q,
q = 2 pi/W) and either spinor direction (the positive- and the negative-energy eigenspinor of h_0 at the carrier), normalized; the sizes
W in {64, 128, 256} with L/W in {16, 32} (the packets far from the far end). Per probe and operator B (the prediction, the uniform boost, and
the flat control's boost on the v = 0 chain with the same probe), on one common normalization (||phi|| = 1):
  the bounded residual  r_B = ||(C_A - F(B)) phi||,  F(B) = (1 + exp(2 pi B))^(-1)  (no logarithm, nothing clipped);
  the modular residual including leakage, clip stable: m_B = ||(k_A - 2 pi B) phi_u|| / ||2 pi B phi_u|| with phi_u the part of phi on the
  eigenvectors of C_A with lambda in [1e-12, 1 - 1e-12] (|k_A| < 27.6), the clipped weight 1 - ||phi_u||^2 printed;
  the probe's weight outside |beta| < 3 of the predicted B (the leakage of the old test subspace).
Controls restored: the uniform flows v = +0.3 and -0.3 (the prediction reduces to the uniform boost; R24e's lattice theorem gives the flat
sea's projector), the sign-reversed profile (v0, v1) -> (-v0, -v1). Diagnostics per chain: the smallest |e| of h (the gap), the filling
(the number of negative modes against L), the eigensolver residual max |h U - U e| on 64 sampled columns, the clipped eigenvalues of C_A.
Cuts: the profile's zero at the cut ("centre") and the profile shifted by W ("+W"). --gpu runs the dense eigendecompositions in torch.
Args: --W --LW --profiles --mc --d-over-W --gpu --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
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
SZ = np.array([[1, 0], [0, -1]], complex); SX = np.array([[0, 1], [1, 0]], complex)


def heigh(M):
    if GPU["dev"] is not None:
        import torch
        a = torch.as_tensor(M, dtype=torch.complex128, device="cuda:%d" % GPU["dev"])
        e, U = torch.linalg.eigh(a)
        return e.real.cpu().numpy(), U.cpu().numpy()
    return eigh(M, driver="evr", check_finite=False)


def fermi_op(B):
    beta, U = heigh(B)
    f = 1.0 / (np.exp(np.clip(2 * math.pi * beta, -700, 700)) + 1.0)
    return (U * f) @ U.conj().T, beta, U


def packets(nA, W, d_list, mc):
    """the probes on A's sites 0 .. nA - 1 (2 components each): Gaussian envelopes at distance d from the cut, carriers at both valleys,
    the two eigenspinors of h_0(k) = sin(k) sigma_x + m_c sigma_z at the carrier."""
    y = np.arange(nA, dtype=float) + 0.5                 # the distance of A's site j from the cut at -1/2
    sig = W / 4.0
    q = 2 * math.pi / W
    out = []
    for d in d_list:
        env = np.exp(-((y - d) ** 2) / (2 * sig * sig))
        for valley, k0 in (("k ~ 0", q), ("k ~ pi", math.pi + q)):
            hk = math.sin(k0) * SX + mc * SZ
            ev, vec = np.linalg.eigh(hk)
            for s_lab, s_idx in (("negative energy", 0), ("positive energy", 1)):
                spin = vec[:, s_idx]
                phi = (env * np.exp(1j * k0 * y))[:, None] * spin[None, :]
                phi = phi.reshape(-1)
                phi = phi / np.linalg.norm(phi)
                out.append({"d": d, "valley": valley, "spinor": s_lab, "phi": phi})
    return out


def chain_state(L, mc, vfun, xc):
    v = np.asarray(vfun(np.arange(L, dtype=float)), float)
    h, h0, b_pred, b_unif, b_naive = B1.boost_operators(L, mc, v, xc, vfun=vfun)
    e, U = heigh(h)
    idx = np.linspace(0, U.shape[1] - 1, 64).astype(int)
    eres = float(np.max(np.abs(h @ U[:, idx] - U[:, idx] * e[idx][None, :])))
    V_ = U[:, e < 0]
    C = V_ @ V_.conj().T
    a = L
    CA = C[a:, a:]
    lam, Wv = heigh(CA)
    clipped = int(np.sum((lam < 1e-12) | (lam > 1 - 1e-12)))
    lamc = np.clip(lam, 1e-300, 1 - 1e-16)
    kA = (Wv * np.log((1 - lamc) / lamc)) @ Wv.conj().T
    unclip = (lam >= 1e-12) & (lam <= 1 - 1e-12)
    diag = {"gap": float(np.min(np.abs(e))), "filling": int(np.sum(e < 0)), "L": L, "eigensolver_residual": eres, "clipped": clipped}
    return {"CA": CA, "kA": kA, "Wv": Wv, "unclip": unclip, "B_pred": b_pred[a:, a:] / (2 * math.pi), "B_unif": b_unif[a:, a:] / (2 * math.pi), "diag": diag}


def probe_rows(st, probes, B_names, beta_max=3.0):
    Fs = {}
    for name in B_names:
        Fs[name] = fermi_op(st[name])
    out = []
    for p in probes:
        phi = p["phi"]
        cu = st["Wv"][:, st["unclip"]].conj().T @ phi
        phi_u = st["Wv"][:, st["unclip"]] @ cu
        row = {k: p[k] for k in ("d", "valley", "spinor")}
        row["clipped_weight"] = float(1.0 - np.vdot(phi_u, phi_u).real)
        for name in B_names:
            F, beta, U = Fs[name]
            row["r_" + name] = float(np.linalg.norm(st["CA"] @ phi - F @ phi))
            twopiB = 2 * math.pi * st[name]
            row["m_" + name] = float(np.linalg.norm(st["kA"] @ phi_u - twopiB @ phi_u) / max(np.linalg.norm(twopiB @ phi_u), 1e-300))
            sel = np.abs(beta) < beta_max
            row["leak_" + name] = float(1.0 - np.sum(np.abs(U[:, sel].conj().T @ phi) ** 2))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, nargs="+", default=[64.0, 128.0, 256.0])
    ap.add_argument("--LW", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--profiles", default="0:0.6,-0.5:0.3")
    ap.add_argument("--mc", type=float, default=0.0)
    ap.add_argument("--d-over-W", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--Lmax", type=int, default=8192)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    GPU["dev"] = a.gpu
    profs = [tuple(float(y) for y in p.split(":")) for p in a.profiles.split(",")]
    cases = []
    t0 = time.time()

    def save():
        json.dump({"script": "obs_bw_modular_v4.py", "args": vars(a), "cases": cases}, open(a.out_json, "w"), indent=1, default=float)
    for W in a.W:
        for LW in a.LW:
            L = int(W * LW)
            if L > a.Lmax:
                continue
            xc = L / 2 - 0.5
            probes = packets(L // 2, W, [f * W for f in a.d_over_W], a.mc)
            flat = chain_state(L, a.mc, B1.make_vfun(("uniform", 0.0)), xc)
            flat_rows = probe_rows(flat, probes, ["B_unif"])
            cases.append({"W": W, "L": L, "case": "flat control (v = 0)", "diag": flat["diag"], "rows": flat_rows})
            print("W %g L %d flat: %s; r_flat %s (%.0f s)" % (W, L, flat["diag"], ["%.2e" % r["r_B_unif"] for r in flat_rows[:4]], time.time() - t0), flush=True)
            specs = [("uniform +0.3", ("uniform", 0.3)), ("uniform -0.3", ("uniform", -0.3))]
            for (v0, v1) in profs:
                for shift, cut_lab in ((0.0, "centre"), (W, "+W")):
                    specs.append(("tanh v0 %g v1 %g, cut at %s" % (v0, v1, cut_lab), ("tanh", v0, v1, W, xc - shift)))
                specs.append(("sign-reversed tanh v0 %g v1 %g, cut at centre" % (-v0, -v1), ("tanh", -v0, -v1, W, xc)))
            for lab, spec in specs:
                st = chain_state(L, a.mc, B1.make_vfun(spec), xc)
                rows = probe_rows(st, probes, ["B_pred", "B_unif"])
                for r, fr in zip(rows, flat_rows):
                    r["r_flat"] = fr["r_B_unif"]; r["m_flat"] = fr["m_B_unif"]
                cases.append({"W": W, "L": L, "case": lab, "diag": st["diag"], "rows": rows})
                print("W %g L %d %s: %s; r_pred %s r_unif %s (%.0f s)" % (W, L, lab, st["diag"], ["%.2e" % r["r_B_pred"] for r in rows[:4]],
                      ["%.2e" % r["r_B_unif"] for r in rows[:4]], time.time() - t0), flush=True)
                save()
    L_ = ["## J2 version 4: the deciding measurement on common wavepacket probes (obs_bw_modular_v4.py, observables 13.3)", "",
          "Per probe (d from the cut, valley, spinor): r = ||(C_A - F(B)) phi|| with F(B) = (1 + e^(2 pi B))^-1 for the prediction, the uniform boost and the flat control; m = the modular residual on the unclipped part; the clipped weight; the leakage out of |beta| < 3 of the predicted B.", "",
          "| W | L | case | gap; filling; eigensolver; clipped | d/W | valley | spinor | r pred | r uniform | r flat | m pred | m uniform | m flat | clipped weight | leakage (pred) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        dg = c["diag"]
        for r in c["rows"]:
            L_.append("| %g | %d | %s | %.1e; %d; %.1e; %d | %g | %s | %s | %s | %s | %s | %s | %s | %s | %.1e | %s |" % (
                c["W"], c["L"], c["case"], dg["gap"], dg["filling"], dg["eigensolver_residual"], dg["clipped"], r["d"] / c["W"], r["valley"], r["spinor"],
                ("%.2e" % r["r_B_pred"]) if "r_B_pred" in r else "-", "%.2e" % r["r_B_unif"], ("%.2e" % r["r_flat"]) if "r_flat" in r else "-",
                ("%.2e" % r["m_B_pred"]) if "m_B_pred" in r else "-", "%.2e" % r["m_B_unif"], ("%.2e" % r["m_flat"]) if "m_flat" in r else "-",
                r["clipped_weight"], ("%.1e" % r["leak_B_pred"]) if "leak_B_pred" in r else "-"))
    L_ += ["", "- every number above is printed by the script; one normalization (||phi|| = 1); the uniform-flow rows test R24e's lattice theorem (the prediction equals",
           "  the uniform boost and the projector is the flat sea's: their r must equal the flat control's); the sign-reversed rows mirror the profile."]
    open(a.md, "w").write("\n".join(L_) + "\n")
    save()


if __name__ == "__main__":
    main()
