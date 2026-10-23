#!/usr/bin/env python
"""engine_massless_ramp_v1.py (builder analyst; engine spec 25.5, R28a's diagnostic: "propagates the clean (0.45, 18) member's prepared
state to T = 2 under prescribed fields interpolated by s(t) = sin^2(pi t/4) between the tau 2 record's initial and final fields in 64
sub-steps (no re-solve), printing at T = 2 the pairs' shell energy, the same with the sea covariance frozen at its initial value under the
final fields and reference, the dust's and the pairs' radial sources apart (the pairs' interior integral to R_0 - 3w), and one constraint
solve on engine_fields_v16 (no fallback; a zero of U reported by its bracket or the failure point); the switch's values are the tau 2
record's"). The member is re-prepared by engine_collapse_v23.prepare_massless_shell with the stored record's arguments (the C-init fields
compared with the stored ones); the initial sea per channel is the negative-energy set of h_m at the record's T = 0 applied fields F_0
(= the C-init fields), the dust the prepared packets with their occupations. Three states at T = 2, each read against the same reference
R2(F_1) (R2 continued from the T = 0 reference to h_m at the record's T = 2 applied fields F_1 by engine_ops_v2.continue_sea_v2 with the
record's c_LZ and window, as the run did): (switch) the record's quasi-static step, the sea and the dust propagated for T = 2 under the
frozen h_m(F_0) (the record's own values beside, from its fields npz); (ramp) the sea and the dust propagated through 64 sub-steps of
dt = 2/64, each under h_m(F(t_mid)), F(t) = F_0 + sin^2(pi t/4)(F_1 - F_0), by the exact exponential of the sub-step's eigendecomposition;
(frozen) the initial sea unpropagated and the dust as in the ramp. Per state: the pairs' shell energy (sum over the shells of
dens(h, sea) - dens(h, R2), the record's E_pairs_shells), the pairs' and the dust's shell energies inside R_0 - 3w and outside, the dust's
total, and the sources' constraint solve on engine_fields_v16 (collar policy flag): failed or not with its message, the zeros of U with
their kinds, the continued intervals, min U/U_ref, the collar charge. Writes --md and --out-json (the per-shell profiles in --out-npz).
CPU (torch on the host). Args: --record dir/tag --substeps --md --out-json --out-npz."""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v23 as E23  # noqa: E402
import engine_ops_v2 as OPS2  # noqa: E402
import engine_fields_v16 as F16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--substeps", type=int, default=64)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--tag", default=None, help="accepted for the GPU runner, unused")
    x = ap.parse_args()
    t0 = time.time()
    d, tag = os.path.split(x.record)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    a = SimpleNamespace(**J["args"])
    E23.OPT["collar_policy"] = a.collar_policy; E23.OPT["support_rel"] = a.support_rel; E23.OPT["site_fields"] = a.site_fields
    E23.C5.backend(x.gpu)
    torch = E23.BK["torch"]
    rs = float(a.rs); R0 = a.shell_R0_over_rs * rs; w = a.shell_width_over_rs * rs
    chs = [E23.Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    Nc, vc, dust, info0, head = E23.prepare_massless_shell(chs, a.g, R, a.ell, rs, a)
    Tz = np.asarray(Z["T"], float)
    k0 = int(np.argmin(np.abs(Tz - 0.0))); k1 = int(np.argmin(np.abs(Tz - 2.0)))
    F0 = (np.asarray(Z["N_applied"][k0], float), np.asarray(Z["v_applied"][k0], float))
    F1 = (np.asarray(Z["N_applied"][k1], float), np.asarray(Z["v_applied"][k1], float))
    dC = (float(np.max(np.abs(Nc - np.asarray(Z["N_cinit"], float)))), float(np.max(np.abs(vc - np.asarray(Z["v_cinit"], float)))))
    print("re-prepared: max |N - N_cinit| %.2e, |v - v_cinit| %.2e; F_1 - F_0: max |dN| %.4e, |dv| %.4e (%.0f s)" % (
        dC[0], dC[1], float(np.max(np.abs(F1[0] - F0[0]))), float(np.max(np.abs(F1[1] - F0[1]))), time.time() - t0), flush=True)
    T_end = float(Tz[k1]); nsub = int(x.substeps); dt = T_end / nsub
    st = {}
    for ch in chs:
        e, U = E23.eigT(E23.clocked_gpu(ch, F0[0], F0[1]))
        neg = e < 0
        en = e.real.cpu().numpy(); kk = int(neg.sum())
        gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=E23.BK["dev"]), np.zeros(0)))
        st[ch.kappa] = {"sea0": U[:, neg].clone(), "dust0": Phi_d, "f": np.asarray(f_d, float), "n_sea": kk, "e0": e, "U0": U,
                        "window": [{"ref": U[:, neg].clone(), "f": (F0[0].copy(), F0[1].copy()), "gap": gap}]}
    # R2 at F_1 by the run's own continuation; the switch: propagation under h_m(F_0) for T_end
    for ch in chs:
        s_ = st[ch.kappa]
        e1, U1 = E23.eigT(E23.clocked_gpu(ch, F1[0], F1[1]))
        ref, ci, mask = OPS2.continue_sea_v2(ch, s_["window"], F1, e1, U1, s_["n_sea"], a.tau, a.c_LZ, E23.eig_at, E23.clocked_host_sparse, a.window_max)
        s_["R2"] = ref; s_["mask_neg"] = bool(np.array_equal(np.asarray(mask, bool), (e1 < 0).cpu().numpy()))
        ph = torch.exp(-1j * s_["e0"].to(torch.complex128) * T_end)
        prop0 = lambda X, U=s_["U0"], ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
        s_["sea_switch"] = prop0(s_["sea0"]); s_["dust_switch"] = prop0(s_["dust0"]) if s_["dust0"].shape[1] else s_["dust0"]
        s_["sea_ramp"] = s_["sea0"].clone(); s_["dust_ramp"] = s_["dust0"].clone()
    # the ramp
    t_r = time.time()
    for j in range(nsub):
        tm = (j + 0.5) * dt
        sv = math.sin(math.pi * tm / (2.0 * T_end)) ** 2
        Fm = (F0[0] + sv * (F1[0] - F0[0]), F0[1] + sv * (F1[1] - F0[1]))
        for ch in chs:
            s_ = st[ch.kappa]
            e, U = E23.eigT(E23.clocked_gpu(ch, Fm[0], Fm[1]))
            ph = torch.exp(-1j * e.to(torch.complex128) * dt)
            pr = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea_ramp"] = pr(s_["sea_ramp"])
            if s_["dust_ramp"].shape[1]:
                s_["dust_ramp"] = pr(s_["dust_ramp"])
        if j % 16 == 15:
            print("ramp sub-step %d of %d (s %.4f; %.0f s)" % (j + 1, nsub, sv, time.time() - t_r), flush=True)
    cut = R0 - 3.0 * w
    inside = R < cut
    out = {"record": x.record, "re_prepared": {"dN": dC[0], "dv": dC[1]}, "T_end": T_end, "substeps": nsub, "cut": cut, "mask_equals_negative_set": all(st[c.kappa]["mask_neg"] for c in chs)}
    prof = {"R": R}
    L = ["## The massless shell's prescribed-field ramp replay (engine_massless_ramp_v1.py; %s)" % tag, "",
         "Member: r_s %g, R_0 %g, w %g (k_0 %g), %d sites, m %g, %d channels; re-prepared to |N - N_cinit| %.2e, |v - v_cinit| %.2e; F_1 - F_0: max |dN| %.4e, |dv| %.4e; T = %g in %d sub-steps of the ramp s(t) = sin^2(pi t/(2 T)); every state read against R2(F_1) (R2 continued by the run's own rule; its mask equals the negative-energy set: %s); the interior cut R_0 - 3w = %g." % (
             rs, R0, w, a.shell_k0, a.nsites, a.m, len(chs), dC[0], dC[1], float(np.max(np.abs(F1[0] - F0[0]))), float(np.max(np.abs(F1[1] - F0[1]))), T_end, nsub,
             out["mask_equals_negative_set"], cut), "",
         "| state | pairs' shell energy | pairs inside the cut | pairs outside | dust total | dust inside the cut | constraint solve on fields v16 | zeros of U (kind at r) | continued intervals | min U/U_ref | collar charge |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, seak, dustk in (("switch (the record's step, recomputed)", "sea_switch", "dust_switch"), ("ramp", "sea_ramp", "dust_ramp"), ("frozen sea covariance (dust as in the ramp)", "sea0", "dust_ramp")):
        eps_p = np.zeros(a.nsites); J_p = np.zeros(a.nsites); eps_d = np.zeros(a.nsites); J_d = np.zeros(a.nsites)
        for ch in chs:
            s_ = st[ch.kappa]
            eps_p += ch.w * (E23.dens(ch.h, s_[seak]) - E23.dens(ch.h, s_["R2"])); J_p += ch.w * (E23.dens(ch.Q, s_[seak]) - E23.dens(ch.Q, s_["R2"]))
            if s_[dustk].shape[1]:
                eps_d += ch.w * E23.dens(ch.h, s_[dustk], s_["f"]); J_d += ch.w * E23.dens(ch.Q, s_[dustk], s_["f"])
        eps = eps_p + eps_d; Jt = J_p + J_d
        _, _, _, inf = F16.solve_constraints_ivp(eps, Jt, a.g, R, a.ell, collar_policy="flag")
        g_ = inf.get("gauge") or {}
        zeros = inf.get("zeros") or g_.get("zeros") or []
        ztxt = "; ".join("%s at %.3f" % (z.get("kind"), z.get("r")) for z in zeros) if zeros else "none"
        fail = inf.get("failed")
        E = F16.charge(inf, a.g) if not fail else None
        row = {"pairs": float(eps_p.sum()), "pairs_inside": float(eps_p[inside].sum()), "pairs_outside": float(eps_p[~inside].sum()), "dust": float(eps_d.sum()),
               "dust_inside": float(eps_d[inside].sum()), "solve_failed": fail, "zeros": zeros, "intervals": g_.get("intervals"), "min_U_over_U_ref": g_.get("min_U_over_U_ref"),
               "collar_charge": E}
        out[lab] = row
        prof["eps_pairs_" + seak] = eps_p; prof["eps_dust_" + dustk + "_" + seak] = eps_d
        L.append("| %s | %+.4e | %+.4e | %+.4e | %.6f | %+.4e | %s | %s | %s | %s | %s |" % (
            lab, row["pairs"], row["pairs_inside"], row["pairs_outside"], row["dust"], row["dust_inside"], ("failed: " + str(fail)[:120]) if fail else "solved",
            ztxt, row["intervals"] if row["intervals"] else "none", ("%.3e" % row["min_U_over_U_ref"]) if row["min_U_over_U_ref"] is not None else "-",
            ("%.6f" % E) if E is not None else "-"))
        print(L[-1], flush=True)
    rec = next((r for r in J["records"] if abs(r["T"] - T_end) < 1e-9), {})
    zp = float(np.asarray(Z["eps_pairs"][k1], float).sum())
    L += ["", "- the record's own values at T = %g: the pairs' shell energy %+.4e (its fields npz: %+.4e), E/E0 %s, the gauge %s." % (
        T_end, rec.get("E_pairs_shells", float("nan")), zp, ("%.7f" % rec["E_over_E0"]) if rec.get("E_over_E0") is not None else "-", (rec.get("gauge") or {}).get("label", "-")),
          "- the switch row recomputes the record's step (the sea under the frozen h_m(F_0) keeps its projector; only the reference moves); the frozen row reads the initial sea against R2(F_1); the ramp row propagates through the interpolated fields.",
          "- every number above is printed by the script (%.0f s)." % (time.time() - t0)]
    np.savez_compressed(x.out_npz, **prof)
    json.dump(out, open(x.out_json, "w"), indent=1, default=float)
    open(x.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
