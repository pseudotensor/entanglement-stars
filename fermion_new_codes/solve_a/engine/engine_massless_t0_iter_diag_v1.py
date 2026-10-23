"""engine_massless_t0_iter_diag_v1.py (builder analyst; a builder's diagnostic behind engine_collapse_v26's choice of the empty-interior
tolerance, docstring (a)): the massless r_s 16 member (512 sites, R_0 96, (k_0, w) = (0.45, 18)) prepared by engine_collapse_v26 on fields v16
(ops v3), its T = 0 record iterated plainly for four iterations, printing per iteration the charge, the pairs' energy and its largest site
value, the largest unmixed residual of N and v and where, N's residual at R 1, 20, 60, 96, 200, 400, and per channel 1, 5, 10 the gap at
zero, the field increment's Gershgorin bound, the continuation's mode and the reference's distance from the sea. Every number printed by the
script. CPU. Run from the box's engine directory: python code/engine_massless_t0_iter_diag_v1.py"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, "code")
import engine_collapse_v26 as E
from types import SimpleNamespace
E.C5.backend(None)
torch = E.BK["torch"]
E.OPT.update({"collar_policy": "flag", "support_rel": 1e-14, "site_fields": "averaged", "ops": "v3", "fields": "v16"})
E.FLD.OPT17["empty_rel"] = None
a = SimpleNamespace(m=0.0, g=0.25, rs=16.0, nsites=512, ell=4, kappa_sea=10, tau=2.0, shell_R0_over_rs=6.0, shell_width_over_rs=1.125, shell_k0=0.45, shell_dk=0.0, shell_E_target=0.0, half_offset=0)
chs = [E.Chan(k, 0.0, 512, False) for k in range(1, 11)]
R = chs[0].R
N, v, dust, info, head = E.prepare_massless_shell_v26(chs, a.g, R, a.ell, a.rs, a)
st = {}
for ch in chs:
    e, U = E.eigT(E.clocked_gpu(ch, N, v)); neg = e < 0
    Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((1024, 0), dtype=torch.complex128), np.zeros(0)))
    _en = e.real.cpu().numpy(); _k = int(neg.sum()); gap = float(_en[_k] - _en[_k - 1])
    ref0 = U[:, neg].clone()
    st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": _k, "window": [{"ref": ref0, "f": (N.copy(), v.copy()), "gap": gap}], "e0": _en}
base = {ch.kappa: (E.dens(ch.h, st[ch.kappa]["sea"]), E.dens(ch.Q, st[ch.kappa]["sea"]), E.dens(ch.h, st[ch.kappa]["dust"], st[ch.kappa]["f"]), E.dens(ch.Q, st[ch.kappa]["dust"], st[ch.kappa]["f"])) for ch in chs}
def sources(refs):
    eps = np.zeros(512); J = np.zeros(512); ep = np.zeros(512)
    for ch in chs:
        hs_, Qs_, dd, Jd = base[ch.kappa]
        ds = hs_ - E.dens(ch.h, refs[ch.kappa]); Js = Qs_ - E.dens(ch.Q, refs[ch.kappa])
        eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); ep += ch.w * ds
    return eps, J, ep
x = (N.copy(), v.copy())
refs = {ch.kappa: st[ch.kappa]["ref"] for ch in chs}
for it in range(4):
    eps, J, ep = sources(refs)
    Nn, vn, Wn, inf = E.solve(eps, J, a.g, R, a.ell, None)
    if inf.get("failed"):
        print("it", it, "solve failed:", inf["failed"]); break
    dN = Nn - x[0]; dv = vn - x[1]
    kN = int(np.argmax(np.abs(dN))); kv = int(np.argmax(np.abs(dv)))
    print("it %d: E %.15f pairs' energy %.3e max |pairs eps| %.3e at R %.0f; max|dN| %.3e at R %.0f (N there %.6f); max|dv| %.3e at R %.0f; dN at R 1, 20, 60, 96, 200, 400: %s" % (
        it, E.FLD.charge(inf, a.g), ep.sum(), np.abs(ep).max(), R[int(np.argmax(np.abs(ep)))], abs(dN[kN]), R[kN], Nn[kN], abs(dv[kv]), R[kv],
        ", ".join("%.2e" % dN[int(r_ - 1)] for r_ in (1, 20, 60, 96, 200, 400))), flush=True)
    # the per-channel change of the reference at the new fields
    x = (Nn.copy(), vn.copy())
    for ch in chs:
        e, U = E.eigT(E.clocked_gpu(ch, x[0], x[1]))
        ref, ci, mask = E.OPS3.continue_sea_v3(ch, st[ch.kappa]["window"], x, e, U, st[ch.kappa]["n_sea"], a.tau, 1.0, E.eig_at, E.clocked_host_sparse)
        en = e.cpu().numpy(); k_ = st[ch.kappa]["n_sea"]
        ov = float(((ref.conj().T @ st[ch.kappa]["sea"]).abs() ** 2).sum())
        if ch.kappa in (1, 5, 10):
            print("   kappa %d: gap at zero %.4e; ||dh||_G %.3e; mode %s; n_sea - |<ref|sea>|^2 = %.3e; min occupied weight %.12f; smallest |e| %.4e" % (
                ch.kappa, en[k_] - en[k_ - 1], ci.get("norm_dH_G"), ci.get("mode"), k_ - ov, ci.get("min_occupied_weight"), np.abs(en).min()), flush=True)
        refs[ch.kappa] = ref
