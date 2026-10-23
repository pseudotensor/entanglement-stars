#!/usr/bin/env python
"""Strand 5 J4, spec sec. 10(r)(b) (builder analyst): stage-1 members of an existing family at chosen z_c, for the refined maximum and the
second crossing.  With strand5_lattice_star_v3's Star (the same kernel source, couplings, R_c as the family's), each new member starts from
the existing member nearest in z_c (its profile interpolated onto the new grid, J = ceil(r_s estimate) + R_c as stage 1, the exterior form
beyond the old grid; its A, B, mu_inf) and is solved by Newton at the new z_c, with stage 1's substep fallback (5 steps from the old z_c).
Writes <out>/lstar_<tag>_<suffix>.json and _profiles.npz in stage 1's format (the members, sorted by z_c), so that stage 2 version 9 runs on
them unchanged."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_lattice_star_v3 as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--zc", required=True, help="comma-separated z_c values")
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = json.load(open(a.family))
    M = R["meta"]
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, kinfo = S.setup_kernel(args)
    if abs(kop / M["k_op"] - 1) > 1e-12:
        raise SystemExit("k_op mismatch: %r against %r" % (kop, M["k_op"]))
    gM, gB = M["couplings"]
    coup = (gM, gB, 1.0, 1.0)
    e1tab = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    mem = [d for d in R["members"] if d.get("newton_ok")]
    zs = sorted(float(z) for z in a.zc.split(","))
    res = {"meta": dict(M, z_c=zs, refine_of=a.family, suffix=a.suffix, builder_driver="strand5_J4_members_v1.py"), "members": []}
    profs = {}
    for zc in zs:
        base = min(mem, key=lambda d: abs(d["z_c"] - zc))
        key = "z%.6f" % base["z_c"]
        rp, psi_p, N_p = prof[key + "_r"], prof[key + "_psi"], prof[key + "_N"]
        chi_p = N_p * psi_p
        A_p, B_p, mu_p = base["A"], base["B"], base["mu_inf"]
        _, rs_est = S.initial_guess(zc, kop, M["mlight"], 10)
        J = int(np.ceil(rs_est)) + int(M["Rc"])
        r = np.arange(1, J + 1, dtype=float)
        ps = np.where(r <= rp[-1], np.interp(r, rp, psi_p), 1 + A_p / r)
        ch = np.where(r <= rp[-1], np.interp(r, rp, chi_p), 1 - B_p / r)
        x0 = np.concatenate([ps, ch, [r[-1] * (ps[-1] - 1), -r[-1] * (ch[-1] - 1), mu_p]])
        star = S.Star(J, stencil, kop, M["mlight"], coup, e1tab)
        x, info = star.newton(x0, zc)
        sub = False
        if not info["ok"]:
            xx, ok = x0, True
            for zz in np.linspace(base["z_c"], zc, 6)[1:]:
                xx, inf2 = star.newton(xx, float(zz))
                if not inf2["ok"]:
                    ok = False
                    break
            if ok:
                x, info, sub = xx, inf2, True
        d = star.diagnostics(x, zc)
        d.update({"newton_ok": info["ok"], "newton_it": info["it"], "residual": info["res"], "started_from_z_c": base["z_c"], "substeps": sub})
        if info["ok"]:
            d["det_sign_fixed_zc"] = star.det_sign(x, zc)
            d["det_sign_fixed_NB"] = star.det_sign(x, zc, NB_target=d["N_B"])
            rr, ps_, nn_, qf_, n_ = star._profile
            k2 = "z%.6f" % zc
            profs[k2 + "_r"], profs[k2 + "_psi"], profs[k2 + "_N"], profs[k2 + "_qF"], profs[k2 + "_n"] = rr, ps_, nn_, qf_, n_
        res["members"].append(d)
        print("z_c %.6f (from %.6f) J %d ok %s it %d: M %.12g A/B-1 %.3e r_s %.3f N_B %.10g (%.0f s)" % (
            zc, base["z_c"], J, info["ok"], info["it"], d["M"], d["A_over_B_minus_1"], d["r_s"], d["N_B"], time.time() - t0), flush=True)
    tag = os.path.basename(a.family).replace("lstar_", "").replace(".json", "")
    with open(os.path.join(a.out, "lstar_%s_%s.json" % (tag, a.suffix)), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(a.out, "lstar_%s_%s_profiles.npz" % (tag, a.suffix)), **profs)


if __name__ == "__main__":
    main()
