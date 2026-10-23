#!/usr/bin/env python
"""obs_J6_gpu_v1.py (builder analyst; J6 per SPEC_final4b_observables.md sec. 13.4: the planar formed cut on one GPU).

J6's setting (observables sec. 1.2's J6 line): flow_chart_v4's cut protocol at kappa a = 1/64 (the planar tanh profile v = -1 + delta tanh(x/W),
delta 0.4, W 25.6, N = 1024 sites, the ramp v(s) = sin^2(pi s/2) v_final over 40/kappa, holds 0 and 12/kappa, the cut at N/2), the single
massless fermion's S_A per area from the quadrant quadrature with the factor 2, at nk = 24 and nk = 12, against the record's 0.786658 (formed)
and 0.7865834 (flat) (C6: within 2e-4). The 4-packet assembly of flow_chart_v4 is H_*'s retired machinery and is not computed.
The route of 13.4: per channel mass m_c the flat sea V0 (the negative modes of the flat chain, flow_chart_v4.chain_hamiltonian), the ramp by a
Chebyshev series of the banded chain Hamiltonian applied to the occupied block only (torch sparse CSR, complex128, the midpoint configuration
per step, terms until the Bessel coefficient falls below 1e-15), the hold by the final chain's eigendecomposition, the cut entropies from the
restricted correlation matrix (flow_chart_v4.cut_entropies' definition); the channels on a grid of --ngrid masses log-spaced over the
quadrant's range, the per-channel entropies interpolated (cubic spline in log m_c) to the exact massless masses of both quadratures, the
interpolation checked at --ncheck exact masses (acceptance 1e-5 per channel); the ramp's step ladder dt = 2, 1, 0.5 on two channels.
Modes: --benchmark (one channel, timing), production. Args: --inv-kappa --delta --nsites --ramp-time --dt --holds --ngrid --ncheck --nk
--ladder-masses --gpu --out-json --md --benchmark --tag."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "flow", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import flow_chart_v4 as FC  # noqa: E402


def channel(mc, a, torch, dev, dt):
    """S_A flat and formed (at the holds) for one channel mass, the ramp by Chebyshev on the device."""
    N = a.nsites
    kappa = 1.0 / a.inv_kappa
    width = a.delta * a.inv_kappa
    mz, yy, x = FC.planar_fields(N, mc)
    vfinal = FC.tanh_profile(x, a.delta, width)
    h0 = FC.chain_hamiltonian(mz, yy, np.zeros(N))
    V0, occ0 = FC.flat_sea(h0)
    flat = FC.cut_entropies(V0, occ0, N // 2)
    import scipy.sparse as sp
    steps = max(1, int(math.ceil(a.ramp_time / dt)))
    step = a.ramp_time / steps
    Phi = torch.as_tensor(V0, dtype=torch.complex128, device=dev)
    # the chain's parts as sparse matrices: h(s) = h_static + env(s) * h_flow (the flow enters linearly through v)
    hs = sp.csr_matrix(FC.chain_hamiltonian(mz, yy, np.zeros(N)))
    hf = sp.csr_matrix(FC.chain_hamiltonian(mz, yy, vfinal)) - hs
    to_t = lambda A: torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64, device=dev), torch.as_tensor(A.indices, dtype=torch.int64, device=dev),
                                             torch.as_tensor(A.data, dtype=torch.complex128, device=dev), size=A.shape)
    HS, HF = to_t(hs), to_t(hf)
    bound = float(abs(hs).sum(axis=1).max() + abs(hf).sum(axis=1).max())
    z = bound * step
    K = int(z + 10)
    while abs(jv(K, z)) > 1e-15:
        K += 1
    coef = [complex(jv(0, z))] + [complex(2 * ((-1j) ** k) * jv(k, z)) for k in range(1, K + 1)]
    for i in range(steps):
        s = min((i + 0.5) * step / a.ramp_time, 1.0)
        env = math.sin(math.pi * s / 2) ** 2
        Ht = lambda Y: (HS @ Y + env * (HF @ Y)) / bound
        T0 = Phi; T1 = Ht(Phi)
        out = coef[0] * T0 + coef[1] * T1
        for k in range(2, K + 1):
            T2 = 2 * Ht(T1) - T0
            out = out + coef[k] * T2
            T0, T1 = T1, T2
        Phi = out
    Phi = Phi.cpu().numpy()
    ef, Qf = FC.spectrum(FC.chain_hamiltonian(mz, yy, vfinal))
    formed = {str(th): FC.cut_entropies(FC.propagate(Qf, ef, th / kappa, Phi), occ0, N // 2) for th in a.holds}
    unit = float(np.max(np.abs(Phi.conj().T @ Phi - np.eye(Phi.shape[1]))))
    return {"mc": float(mc), "flat": [float(x) for x in flat], "formed": {k: [float(x) for x in v] for k, v in formed.items()}, "steps": steps, "terms": K,
            "orthonormality_after_ramp": unit}


def massless_masses(nk):
    return [(float(np.sqrt(np.sin(ky) ** 2 + np.sin(kz) ** 2)), w) for ky, kz, w in FC.quadrant_quadrature(nk)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv-kappa", type=float, default=64.0)
    ap.add_argument("--delta", type=float, default=0.4)
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--ramp-time", type=float, default=2560.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--holds", type=float, nargs="+", default=[0.0, 12.0])
    ap.add_argument("--ngrid", type=int, default=48)
    ap.add_argument("--ncheck", type=int, default=8)
    ap.add_argument("--nk", type=int, nargs="+", default=[24, 12])
    ap.add_argument("--ladder-masses", type=float, nargs="+", default=[0.1, 0.7])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--tag", default="J6")
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda:%d" % a.gpu) if a.gpu is not None else torch.device("cpu")
    out_json = a.out_json or "J6GPU/%s.json" % a.tag
    md = a.md or "J6GPU/%s.md" % a.tag
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    t0 = time.time()
    if a.benchmark:
        r = channel(0.3, a, torch, dev, a.dt)
        print("benchmark m_c 0.3: %d steps, %d terms, %.1f s; S_A flat %.10f, formed %s; orthonormality %.1e" % (r["steps"], r["terms"], time.time() - t0, r["flat"][0],
              {k: round(v[0], 10) for k, v in r["formed"].items()}, r["orthonormality_after_ramp"]), flush=True)
        return
    allm = sorted({m for nk in a.nk for m, _ in massless_masses(nk)})
    lo, hi = min(allm) * 0.98, max(allm) * 1.02
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), a.ngrid))
    rec = {"script": "obs_J6_gpu_v1.py", "args": vars(a), "grid": [], "checks": [], "ladder": [], "assembled": {}}

    def save():
        json.dump(rec, open(out_json, "w"), indent=1)
    for mc in grid:
        r = channel(mc, a, torch, dev, a.dt)
        rec["grid"].append(r)
        print("grid m_c %.6f: S_A flat %.10f formed %s (%.0f s)" % (mc, r["flat"][0], {k: round(v[0], 10) for k, v in r["formed"].items()}, time.time() - t0), flush=True)
        save()
    lg = np.log([r["mc"] for r in rec["grid"]])
    spl = {"flat": CubicSpline(lg, [r["flat"][0] for r in rec["grid"]])}
    for th in a.holds:
        spl[str(th)] = CubicSpline(lg, [r["formed"][str(th)][0] for r in rec["grid"]])
    # the interpolation check at exact masses spread over the range
    pick = [allm[int(i)] for i in np.linspace(1, len(allm) - 2, a.ncheck)]
    for mc in pick:
        r = channel(mc, a, torch, dev, a.dt)
        d = {"flat": abs(float(spl["flat"](math.log(mc))) - r["flat"][0])}
        for th in a.holds:
            d[str(th)] = abs(float(spl[str(th)](math.log(mc))) - r["formed"][str(th)][0])
        rec["checks"].append({"mc": mc, "exact": r, "interp_abs_error": d, "pass": all(x < 1e-5 for x in d.values())})
        print("check m_c %.6f: interpolation errors %s" % (mc, {k: "%.1e" % v for k, v in d.items()}), flush=True)
        save()
    # the ramp's step ladder on two channels
    for mc in a.ladder_masses:
        row = {"mc": mc}
        for dt in (2.0, 1.0, 0.5):
            r = channel(mc, a, torch, dev, dt)
            row[str(dt)] = {"flat": r["flat"][0], "formed": {k: v[0] for k, v in r["formed"].items()}}
        rec["ladder"].append(row)
        print("ladder m_c %g: %s" % (mc, {dt: row[dt]["formed"] for dt in ("2.0", "1.0", "0.5")}), flush=True)
        save()
    # the massless sea's S_A per area at each quadrature
    for nk in a.nk:
        mm = massless_masses(nk)
        Sf = sum(2 * w * float(spl["flat"](math.log(m))) for m, w in mm)
        Sh = {str(th): sum(2 * w * float(spl[str(th)](math.log(m))) for m, w in mm) for th in a.holds}
        rec["assembled"][str(nk)] = {"flat": Sf, "formed": Sh, "excess": {k: v / Sf - 1 for k, v in Sh.items()}}
    save()
    L = ["## J6 on the GPU (obs_J6_gpu_v1.py): the planar formed cut at kappa a = 1/%g, N = %d, ramp %g at dt %g, %d grid masses" % (a.inv_kappa, a.nsites, a.ramp_time, a.dt, a.ngrid), "",
         "| nk | S_A per area flat | formed at the holds %s | excess formed/flat - 1 | C6: |formed(12) - 0.786658| | flat - 0.7865834 |" % ", ".join("%g" % h for h in a.holds),
         "|---|---|---|---|---|---|"]
    for nk in a.nk:
        A_ = rec["assembled"][str(nk)]
        last = str(a.holds[-1])
        L.append("| %d | %.8f | %s | %s | %.2e | %.2e |" % (nk, A_["flat"], ", ".join("%.8f" % A_["formed"][str(h)] for h in a.holds),
                 ", ".join("%.3e" % A_["excess"][str(h)] for h in a.holds), abs(A_["formed"][last] - 0.786658), A_["flat"] - 0.7865834))
    L += ["", "Interpolation checks (acceptance 1e-5 per channel): " + "; ".join("m_c %.4f: %s" % (c["mc"], ", ".join("%s %.1e" % (k, v) for k, v in c["interp_abs_error"].items()))
                                                                            for c in rec["checks"]),
          "", "The ramp's step ladder (the formed S_A per channel at dt 2, 1, 0.5): " + "; ".join("m_c %g: %s" % (r["mc"], ", ".join(
              "dt %s: %s" % (dt, ", ".join("%.10f" % v for v in r[dt]["formed"].values())) for dt in ("2.0", "1.0", "0.5"))) for r in rec["ladder"]),
          "", "- every number above is printed by the script; the 4-packet assembly (H_*'s retired machinery) is not computed."]
    open(md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
