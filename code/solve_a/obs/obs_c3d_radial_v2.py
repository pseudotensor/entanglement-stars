#!/usr/bin/env python3
"""obs_c3d_radial_v2.py (version 2 of obs_c3d_radial_v1.py, whose records are c3d_radial_v1: the profile 'pg_taper' with the box's compact
C^2 quintic taper (obs_box3d_v3 B11: R_2 = taper_L/2 - 2, R_1 = R_2 - taper_width), the packet's centre --R0 in lattice units, the holds as an
argument, obs_box3d_v3's routines; for sec. 8.3's matched comparison of one detector mode, the replacement of C-3D).

obs_c3d_radial_v1.py: the radial channel's side of the control C-3D (SPEC_final4b_observables.md sec. 4.4; J5 part 2).  Builder analyst.

The occupation history n_w(t) = ||P_0 U(t)^dagger w||^2 of the radial channel's outgoing packet at the same (r_s, omega, j) as the
3-dimensional partial-wave packet of obs_box3d_v2.py (mode c3d), through the formation transient: the chain of flow_chart_v4 (on-site
m SZ + (kappa_j/R) SY, sine hop, two-step flow hop; R_n = n + 1), the formation ramp v(R, t) = sin^2(pi t/(2 t_r)) v(R) with t_r = 2 r_s by
flow_page_v2's CF4 route (Banded, ramp_backward) at the box's dt, the hold exactly in h_f's eigenbasis, P_0 the flat chain's sea.  The
packet is flow_chart_v4.gaussian_carrier(R, R_0, sigma_x, k_0, m, kappa_j/R_0) with k_0 the carrier of Killing frequency omega on the local
outgoing branch at R_0 (obs_box3d_v2.outgoing_k), i.e. the radial function the box's packet is built from (B9 of obs_box3d_v2).
Profiles: 'pg_trunc' (the box's: the exact profile times (1 - tanh((R - R_cut)/4))/2, R_cut = 2.5 r_s; the like-for-like comparison) and
'pg' (the exact profile untruncated; the channel the escaping-sector computation uses).  Chain lengths: a long chain (the open exterior;
no echo within the holds) and short chains whose wall sits at the box's half-width (the box's boundary has no exact radial analogue: its
faces wrap, the chain's wall reflects).  The map check: the flat chain's packet energy at small R_0 against the box's two maps.
Every record: the label of the exact profile (control) and the script name.
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import flow_chart_v4 as V4  # noqa: E402
import flow_page_v2 as FP  # noqa: E402
import obs_box3d_v3 as BOX  # noqa: E402

T0 = time.time()


def log(msg):
    print("[%7.0f s] %s" % (time.time() - T0, msg), flush=True)


TAPER = {"L": None, "width": 8.0}


def profile(name, rs, Rm, Rcut):
    if name == "pg":
        return lambda R: V4.pg_profile(R, rs, Rm)
    if name == "pg_taper":
        R2 = TAPER["L"] / 2.0 - 2.0
        R1 = R2 - TAPER["width"]
        return lambda R: V4.pg_profile(R, rs, Rm) * BOX.taper_quintic(R, R1, R2)
    if name == "pg_trunc":
        return lambda R: V4.pg_profile(R, rs, Rm) * BOX.truncation(R, Rcut)
    raise ValueError(name)


def channel(a, prof, N, kj, dev, holds):
    rs, Rm, m = a.rs, a.Rm, a.mass
    kappa = 1.0 / (2 * rs)
    vfun = profile(prof, rs, Rm, a.rcut_over_rs * rs)
    mz, yy, R = V4.radial_fields(N, m, kj)
    v = vfun(R)
    h0 = V4.chain_hamiltonian(mz, yy, np.zeros(N))
    hf = V4.chain_hamiltonian(mz, yy, v)
    B = FP.Banded(h0, hf - h0, torch, dev)
    e0, Q0 = torch.linalg.eigh(torch.as_tensor(h0, device=dev))
    V0 = Q0[:, e0 < 0].contiguous()
    del Q0
    ef, Qf = torch.linalg.eigh(torch.as_tensor(hf, device=dev))
    R0 = a.R0 if a.R0 else a.R0_over_rs * rs
    omega = a.omega_over_kappa * kappa
    v0 = float(vfun(np.array([R0]))[0])
    k0, vg0 = BOX.outgoing_k(omega, m, kj / R0, v0)
    w = V4.gaussian_carrier(R, R0, a.sigma_x, k0, m, kj / R0)
    wt = torch.as_tensor(w, device=dev)
    hw = torch.as_tensor(hf, device=dev) @ wt
    E = float(torch.vdot(wt, hw).real)
    dE = math.sqrt(max(float(torch.vdot(hw, hw).real) - E * E, 0.0))
    c = Qf.conj().T @ wt
    cols = torch.stack([Qf @ (torch.exp(1j * ef * float(h / kappa)) * c) for h in holds], dim=1)
    back = FP.ramp_backward(B, cols, a.ramp_over_rs * rs, a.dt)
    nw = ((V0.conj().T @ back).abs() ** 2).sum(0).cpu().numpy()
    n_flat = float(((V0.conj().T @ wt).abs() ** 2).sum())
    return {"profile": prof, "N": N, "R_max": float(R[-1]), "kappa_j": kj, "omega": omega, "R0": R0, "sigma_x": a.sigma_x, "k0": k0, "vg0": vg0,
            "v_R0": v0, "E_killing": E, "E_spread": dE, "flat_sea_weight": n_flat, "n_w": nw.tolist()}


def map_check(a, dev):
    rows = []
    for kj in (-1.0, 1.0):
        N = 64
        mz, yy, R = V4.radial_fields(N, a.mass, kj)
        h0 = V4.chain_hamiltonian(mz, yy, np.zeros(N))
        k0, _ = BOX.outgoing_k(a.map_omega, a.mass, kj / a.map_R0, 0.0)
        w = V4.gaussian_carrier(R, a.map_R0, a.map_sigma, k0, a.mass, kj / a.map_R0)
        E = float(np.real(np.vdot(w, h0 @ w)))
        dE = math.sqrt(max(float(np.real(np.vdot(h0 @ w, h0 @ w))) - E * E, 0.0))
        rows.append({"kappa_j": kj, "E_chain": E, "E_spread_chain": dE, "k0": k0, "R0": a.map_R0, "sigma_x": a.map_sigma})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rs", type=float, default=8.0)
    p.add_argument("--Rm", type=float, default=2.0)
    p.add_argument("--rcut-over-rs", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--ramp-over-rs", type=float, default=2.0)
    p.add_argument("--nsites", type=int, nargs="+", default=[4096, 24, 32])
    p.add_argument("--profiles", nargs="+", default=["pg_trunc", "pg"])
    p.add_argument("--omega-over-kappa", type=float, default=4.0)
    p.add_argument("--R0-over-rs", type=float, default=2.2)
    p.add_argument("--sigma-x", type=float, default=4.0)
    p.add_argument("--map-omega", type=float, default=0.5)
    p.add_argument("--map-R0", type=float, default=6.0)
    p.add_argument("--map-sigma", type=float, default=1.5)
    p.add_argument("--box-json", nargs="*", default=[])
    p.add_argument("--R0", type=float, default=None, help="the packet's centre in lattice units (overrides --R0-over-rs)")
    p.add_argument("--holds", type=float, nargs="+", default=BOX.C3D_HOLDS)
    p.add_argument("--taper-L", type=float, default=None, help="pg_taper: the box size whose taper is copied")
    p.add_argument("--taper-width", type=float, default=8.0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    dev = torch.device("cuda:0")
    holds = list(a.holds)
    TAPER["L"], TAPER["width"] = a.taper_L, a.taper_width
    out = {"label": BOX.LABEL_PG + "; radial channels of flow_chart_v4 (the record's radial code)", "script": os.path.basename(__file__),
           "args": vars(a), "holds_over_kappa": holds, "channels": [], "started_utc": BOX.utc()}
    path = os.path.join(a.outdir, "c3d_radial2_%s.json" % a.tag)
    for prof in a.profiles:
        for N in a.nsites:
            if prof in ("pg", "pg_taper") and N < 100:
                continue
            for kj in (-1.0, 1.0):
                t0 = time.time()
                r = channel(a, prof, N, kj, dev, holds)
                r["wall_s"] = time.time() - t0
                out["channels"].append(r)
                log("%s N %d kappa_j %g: E %.5f k0 %.4f n_w %s" % (prof, N, kj, r["E_killing"], r["k0"], ["%.3e" % x for x in r["n_w"]]))
                BOX.jdump(out, path)
    out["map_check_chain"] = map_check(a, dev)
    comp = []
    for bj in a.box_json:
        with open(bj) as f:
            box = json.load(f)
        Lb = box["box"]["L"]
        for pk in box["packets"]:
            for ch in out["channels"]:
                if ch["kappa_j"] == pk["kappa_j"]:
                    nb, nr = np.array(pk["n_w"]), np.array(ch["n_w"])
                    comp.append({"L": Lb, "kappa_j": pk["kappa_j"], "m_j": pk["m_j"], "profile": ch["profile"], "N": ch["N"],
                                 "E_box": pk["E_killing"], "E_chain": ch["E_killing"], "k0_box": pk["k0"], "k0_chain": ch["k0"],
                                 "n_box": nb.tolist(), "n_radial": nr.tolist(), "ratio_box_over_radial": (nb / nr).tolist(),
                                 "rel_diff": ((nb - nr) / nr).tolist()})
        out.setdefault("map_check_box", {})[str(Lb)] = box.get("map_check_flat_box")
    out["comparison"] = comp
    out["wall_s"] = time.time() - T0
    out["finished_utc"] = BOX.utc()
    BOX.jdump(out, path)
    write_md(out, path.replace(".json", ".md"))


def write_md(o, path):
    H = o["holds_over_kappa"]
    L = ["# The matched detector-mode comparison of sec. 8.3 (the replacement of C-3D): the box against the radial chain", "", "Label: " + o["label"] + ".", "",
         "Scripts obs_box3d_v3.py (mode detector, the box) and obs_c3d_radial_v2.py (the chain); finished %s.  Packet: j = 1/2, omega/kappa = %g, "
         "R_0 = %s, sigma_x = %g, r_s = %g, R_m = %g; ramp t_r = %g r_s, dt = %g; taper of the L = %s box." % (
             o.get("finished_utc"), o["args"]["omega_over_kappa"], o["args"]["R0"] or ("%g r_s" % o["args"]["R0_over_rs"]), o["args"]["sigma_x"],
             o["args"]["rs"], o["args"]["Rm"], o["args"]["ramp_over_rs"], o["args"]["dt"], o["args"]["taper_L"]), ""]
    L += ["## Radial channels: n_w(t) (holds in units of 1/kappa)", ""]
    head = ["profile", "N", "kappa_j", "k0", "E"] + ["%g" % h for h in H]
    L += [BOX.md_table(head, [(c["profile"], c["N"], c["kappa_j"], c["k0"], c["E_killing"]) + tuple(c["n_w"]) for c in o["channels"]]), ""]
    if o.get("comparison"):
        L += ["## Box over radial channel: n_box/n_radial at each hold", ""]
        head = ["L", "kappa_j", "m_j", "profile", "N", "E box", "E chain"] + ["%g" % h for h in H]
        L += [BOX.md_table(head, [(c["L"], c["kappa_j"], c["m_j"], c["profile"], c["N"], c["E_box"], c["E_chain"]) + tuple(c["ratio_box_over_radial"])
                                  for c in o["comparison"]]), ""]
    L += ["## The map check (flat background, R_0 = %g, sigma_x = %g, omega = %g)" % (o["args"]["map_R0"], o["args"]["map_sigma"], o["args"]["map_omega"]), ""]
    rows = [("chain", r["kappa_j"], "-", r["E_chain"], r["E_spread_chain"]) for r in o["map_check_chain"]]
    for Lb, mc in (o.get("map_check_box") or {}).items():
        rows += [("box L=%s" % Lb, r["kappa_j"], r["map"], r["E"], r["E_spread"]) for r in (mc or [])]
    L += [BOX.md_table(["route", "kappa_j", "map", "E", "E spread"], rows), ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
