#!/usr/bin/env python
"""cv_common.py (busi lane, correctness reading of CURVATURE_readout_v1): own loader, the engine's parity spline (its definition,
engine_fields_v4.parity_spline: quintic through the nodes with the data extended through the centre by parity), the engine's source kernels
(engine_fields_v4.kernels docstring: K_l(r, R) = (2 pi ell^2)^(-3/2) exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2), i_0 = sinh z/z,
i_1 = cosh z/z - sinh z/z^2), md5. Read-only on every input."""
import hashlib
import json
import os

import numpy as np
from scipy.interpolate import make_interp_spline

# the delivery's named inputs (CURVATURE_readout_v1 sec. 2 and curvature_v1_tables.md), certified extents per the delivery's table
MEMBERS = {
    "B2": ("<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096", None),
    "P64_s4_c19": ("<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11", 40.0),
    "P64_s4": ("<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40", 40.0),
    "P64_s2": ("<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d", 40.0),
    "P64_s8": ("<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40", 40.0),
    "P64_s4_tau4": ("<internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg", 40.0),
    "P64_s4_tau2": ("<internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg", 12.75),
}
GATE = 0.1


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parity_spline(rn, y, parity):
    """engine_fields_v4.parity_spline (the engine's definition of the dense fields between the nodes)."""
    xx = np.concatenate([-rn[:0:-1], rn])
    yy = np.concatenate([parity * y[:0:-1], y])
    return make_interp_spline(xx, yy, k=5)


def kernels(r, R, ell):
    """K_0(r, R), K_1(r, R) as (n_r, n_R), from the docstring formula, in the exponentially scaled form with a series below z = 1e-2."""
    r = np.asarray(r, float)[:, None]
    R = np.asarray(R, float)[None, :]
    z = r * R / ell ** 2
    Em = np.exp(-(r - R) ** 2 / (2.0 * ell ** 2))
    Ep = np.exp(-(r + R) ** 2 / (2.0 * ell ** 2))
    EA = np.exp(-(r ** 2 + R ** 2) / (2.0 * ell ** 2))
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    # i_0 e^{-(r^2+R^2)/2l^2} = (Em - Ep)/(2 z);  i_1 e^{...} = ((z - 1) Em + (z + 1) Ep)/(2 z^2)
    i0 = np.where(small, EA * (1.0 + z ** 2 / 6.0), (Em - Ep) / (2.0 * zs))
    i1 = np.where(small, EA * (z / 3.0) * (1.0 + z ** 2 / 10.0), ((zs - 1.0) * Em + (zs + 1.0) * Ep) / (2.0 * zs ** 2))
    pref = (2.0 * np.pi) ** (-1.5) / ell ** 3
    return pref * i0, pref * i1


def load(name, with_handoff=True):
    base, cert_end = MEMBERS[name]
    files = {"json": base + ".json", "fields": base + "_fields.npz"}
    if with_handoff and os.path.exists(base + "_handoff.npz"):
        files["handoff"] = base + "_handoff.npz"
    J = json.load(open(files["json"]))
    Z = np.load(files["fields"], allow_pickle=True)
    args = J["args"]; recs = J["records"]
    T = np.asarray(Z["T"], float)
    Tj = np.array([float(x["T"]) for x in recs])
    assert len(Tj) == len(T) and np.allclose(Tj, T)
    rn_all = [np.asarray(x, float) for x in Z["r_nodes"]]
    rn = rn_all[0]
    assert all(np.array_equal(x, rn) for x in rn_all), "node grid varies between records"
    M = dict(name=name, files=files, cert_end=cert_end, args=args, rs=float(args["rs"]), g=float(args["g"]), ell=float(args["ell"]),
             tau=float(args["tau"]), m=float(args["m"]), T=T, Trs=T / float(args["rs"]), R=np.asarray(Z["R"], float),
             Nbar=np.asarray(Z["N"], float), vbar=np.asarray(Z["v"], float), Napp=np.asarray(Z["N_applied"], float), vapp=np.asarray(Z["v_applied"], float),
             rn=rn, Nn=np.array([np.asarray(x, float) for x in Z["N_nodes"]]), vn=np.array([np.asarray(x, float) for x in Z["v_nodes"]]),
             eps=np.asarray(Z["eps_tot"], float), Jt=np.asarray(Z["J_tot"], float),
             zeros=[list(x.get("trapping_zeros") or []) for x in recs], converged=np.array([bool(x.get("converged")) for x in recs]),
             gauge_intact=np.array([((x.get("gauge") or {}).get("label", "intact") == "intact") for x in recs]),
             gauge_label=[(x.get("gauge") or {}).get("label", "intact") for x in recs],
             interp_floor_N=np.array([float(x.get("interp_floor_N") or np.nan) for x in recs]),
             interp_floor_v=np.array([float(x.get("interp_floor_v") or np.nan) for x in recs]),
             N_min=np.array([float(x.get("N_min") or np.nan) for x in recs]))
    if "handoff" in files:
        Hh = np.load(files["handoff"], allow_pickle=False)
        M["hT"] = np.asarray(Hh["T"], float); M["hrn"] = np.asarray(Hh["r_nodes"], float)
        M["Sr"] = np.asarray(Hh["S_r_ell"], float); M["St"] = np.asarray(Hh["S_t_ell"], float)
    return M


def first_zero_record(M):
    for k, z in enumerate(M["zeros"]):
        if len(z) > 0:
            return k
    return None
