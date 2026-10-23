#!/usr/bin/env python
"""Strand 2d, the (3,1)' term of J^prep and the fixed-vacuum cost (spec SPEC_strand2d_jet.md sec. 7(2); builder analyst).
Stage 1: the free evaluation (exact at g = 0; at (0.2, 1) the estimate the spec allows on B4, with the cube's dressed value to come).

For the ruler perturbation psi = 1 + lam c_q (N = 1; the grading depends on psi only, so lam_N gives z = 0):
  z_km = <u_k| dP_+/dlam |v_m> at lam = 0, P_+(lam) = the projector on the light packet's particle orbitals at psi(lam)
        (gauge-invariant form of d<u'_k|v_m>/dlam), by central differences at h in {1e-3, 1e-4};
  Phi_0 = the flat (2,0) multiplet (d_0 vectors, strand2d_lib_v1.PairBlock at the coupling), phi_ij antisymmetric;
  Z_1^dag Phi_0 has the (3,1) amplitudes A_{bcd,m} = z_bm phi_cd + z_cm phi_db + z_dm phi_bc (b < c < d), orthonormal basis;
  norm2 = avg_a sum |A|^2, checked against ||z||_F^2 - sum_kk' (z z^dag)_kk' <Phi_0|d_k^dag d_k'|Phi_0>;
  E31_free = avg_a sum (eps_b + eps_c + eps_d + eps_m) |A|^2 (flat one-body quasiparticle energies), J^prep_(3,1)' = 2 E31_free;
  J^prep,vac (free) = 2 sum_km (eps_k + eps_m) |z_km|^2;
  the norm-defect check: 1 - Tr(C'^dag C')/d_0 at lam = +-h (C' = O C_0, Schur overlap) against lam^2 norm2.
These enter only the psi psi entry of the 2 x 2 blocks (the (3,1)' amplitude is first order in lam_psi alone).
Writes <out>/prep31_<lattice>_gM<gM>_gB<gB>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import socket
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2d_lib_v1 as L2D  # noqa: E402

CT = torch.complex128


def cq(lat, q):
    x1 = lat["coord"][:, 0].astype(float)
    return np.ones(lat["nsite"]) if q == 0 else np.cos(q * x1)


def particle_projector(lat, m, psi):
    lt = S2.Light(lat, m, psi, np.ones(lat["nsite"]))
    return lt.vp @ lt.vp.conj().T, lt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lattice", required=True, choices=["C8", "B4"])
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--gM", type=float, required=True)
    ap.add_argument("--gB", type=float, required=True)
    ap.add_argument("--Lam", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    dev = torch.device(a.device)
    lat = L2D.lattice(a.lattice)
    n = lat["nsite"]
    nW = n * lat["ncomp"]
    one = np.ones(n)
    qs = [0.0, np.pi] if a.lattice == "C8" else [0.0, np.pi / 2, np.pi]
    hs = (1e-3, 1e-4)
    pb = L2D.PairBlock(lat, a.m, a.gM, a.gB, a.Lam, a.device)
    b0 = pb.build(one)
    w0, V0 = L2D.eigh(pb.H(b0, one))
    d0 = int(np.sum(w0 - w0[0] < a.tol))
    C0 = V0[:, :d0]                                   # (dim, d0) torch
    P0, lt0 = particle_projector(lat, a.m, one)
    vp0, vm0, eps = lt0.vp, lt0.vm, lt0.eps
    i1, i2 = pb.pairs[:, 0], pb.pairs[:, 1]
    # phi^a as antisymmetric nW x nW matrices, all members: (d0, nW, nW)
    Ct = C0.T.contiguous()
    phi = torch.zeros((d0, nW, nW), dtype=CT, device=dev)
    I1 = torch.as_tensor(i1, device=dev)
    I2 = torch.as_tensor(i2, device=dev)
    phi[:, I1, I2] = Ct
    phi[:, I2, I1] = -Ct
    # one-body density of Phi_0 in the particle modes: <Phi|d_k^dag d_k'|Phi> = sum_j conj(phi_kj) phi_k'j  (for the norm check)
    rho1 = torch.einsum("akj,alj->akl", phi.conj(), phi).mean(0)    # avg over members, (nW, nW): [k, k'] = <d_k^dag d_k'>
    epst = torch.as_tensor(eps, dtype=torch.float64, device=dev)
    res = {"meta": {"host": socket.gethostname(), "lattice": a.lattice, "m": a.m, "gM": a.gM, "gB": a.gB, "Lambda": a.Lam,
                    "device": a.device, "d0": d0, "dim": pb.dim, "flat_levels": [float(x) for x in w0[:d0 + 4]],
                    "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, "rows": []}
    for q in qs:
        c = cq(lat, q)
        rowq = {"q": q, "steps": {}}
        zs = {}
        for h in hs:
            Pp, _ = particle_projector(lat, a.m, one + h * c)
            Pm, _ = particle_projector(lat, a.m, one - h * c)
            dP = (Pp - Pm) / (2 * h)
            z = vp0.conj().T @ dP @ vm0               # (nW particle k, nW hole m)
            zpp = vp0.conj().T @ dP @ vp0             # particle-particle block of dP (zero for a projector derivative)
            zs[h] = z
            rowq["steps"][str(h)] = {"z_frob": float(np.linalg.norm(z)), "pp_block_norm": float(np.linalg.norm(zpp))}
        rowq["z_step_agreement_rel"] = float(np.max(np.abs(zs[1e-3] - zs[1e-4])) / max(np.max(np.abs(zs[1e-4])), 1e-300))
        z = torch.as_tensor(zs[1e-4], dtype=CT, device=dev)
        # fixed-vacuum cost (free) and the Pauli-blocking norm formula
        Evac = float(torch.sum((epst[:, None] + epst[None, :]) * torch.abs(z) ** 2))
        normF2 = float(torch.sum(torch.abs(z) ** 2))
        zz = z @ z.conj().T
        pauli = float(torch.real(torch.sum(zz * rho1)))   # sum_kk' (z z^dag)_kk' <d_k^dag d_k'>, rho1[k, k'] = <d_k^dag d_k'>
        # (3,1) amplitudes per member, looping over the hole index m
        norm2, E31 = 0.0, 0.0
        for aa in range(d0):
            ph = phi[aa]
            for mm in range(nW):
                Zm = z[:, mm]
                # T_bcd = Z_b phi_cd + Z_c phi_db + Z_d phi_bc
                T = Zm[:, None, None] * ph[None, :, :] + Zm[None, :, None] * ph.T[:, None, :] + Zm[None, None, :] * ph[:, :, None]
                a2 = torch.abs(T) ** 2
                s = float(a2.sum()) / 6.0
                es = float(torch.einsum("b,bcd->", epst, a2)) * 3.0 / 6.0 + float(eps[mm]) * s
                norm2 += s
                E31 += es
        norm2 /= d0
        E31 /= d0
        rowq.update({"norm2_ZPhi": norm2, "norm2_pauli_formula": normF2 - pauli, "z_frob2": normF2,
                     "E31_free": E31, "J_prep_31_free": 2 * E31, "J_prep_vac_free": 2 * Evac})
        # norm-defect check from the Schur overlap at lam = +-h
        nd = {}
        for h in hs:
            vals = []
            for sgn in (1, -1):
                bq = pb.build(one + sgn * h * c)
                O = pb.overlap(bq, b0)
                Cp = O @ C0
                vals.append(1 - float(torch.real(torch.trace(Cp.conj().T @ Cp))) / d0)
            nd[str(h)] = {"defect_plus": vals[0], "defect_minus": vals[1], "second_diff": (vals[0] + vals[1]) / h ** 2}
        rowq["norm_defect"] = nd
        rowq["norm_defect_check_rel"] = abs(nd["0.0001"]["second_diff"] - 2 * norm2) / max(abs(2 * norm2), 1e-300)
        res["rows"].append(rowq)
        print("%s (%g, %g) q %.4f: J31_free %.6e J_vac_free %.6e norm2 %.6e pauli %.6e defect'' %.6e (%.0f s)" % (
            a.lattice, a.gM, a.gB, q, 2 * E31, 2 * Evac, norm2, normF2 - pauli, nd["0.0001"]["second_diff"], time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "prep31_%s_gM%g_gB%g.json" % (a.lattice, a.gM, a.gB)), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
