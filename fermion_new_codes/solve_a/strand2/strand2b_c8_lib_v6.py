"""Strand 2b on C8, library version 6 (spec SPEC_strand2b_row.md sec. 15 item 6 (i), (iv) and item 7; builder analyst).

Everything of strand2b_c8_lib_v5 unchanged (imported), with:
- SystemC8v6.state: the cluster gap gap_cluster = e[d_0] - e[d_0 - 1] beside the block gap e[d_0] - e[0] (key "gap", kept, and
  "gap_block"), as the R4 library from its version 7;
- newton_fd: the line search fails the step when no acceptable step is found and the routine stops there with the best iterate
  (version 5 executed the last rejected trial step); an exception raised while evaluating F or the Jacobian stops the routine with the
  best iterate and the exception recorded (why = "exception"); the info carries the stop reason, the best residual and the last
  computed Newton correction ||dz||;
- returned_vector_record: at a returned vector, its own residual ||F||, the Newton correction ||J^-1 F|| there (FD Jacobian), the
  smallest component and, when given, the distance to a reference solution (exceptions recorded);
- continuation_fd: version 5's continuation in lambda with this newton_fd, recording at every accepted step the cluster gap, the block
  gap and sigma_min of the FD Jacobian;
- dF_dk: the exact k-derivative of F (F is affine in k: C_x = lam src_x - k psi_x [K(psi - b)]_x, G_y = lam sum_x N_x t_yx +
  k [(K 1)_y - N_y [K(psi - b)]_y - (K (N psi))_y]), for item 7's continuation in k."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand2b_c8_lib_v5 as V5  # noqa: E402
from strand2b_c8_lib_v5 import BlockModel, herm, jacobian_fd, jacobian_c8, proj_response_block  # noqa: E402,F401


class SystemC8v6(V5.SystemC8):
    def state(self, psi, N):
        """Version 5's state (the same keys and values) with gap_block, gap_cluster and the lowest d_0 + 2 levels added."""
        b = self.bm.build(psi)
        H = np.einsum("y,yab->ab", N, b["M"])
        e, V = np.linalg.eigh(herm(H))
        d0_tol = int(np.sum(e - e[0] < self.tol_deg))
        d0 = self.d0_fixed if self.d0_fixed is not None else d0_tol
        gap = float(e[d0] - e[0]) if len(e) > d0 else np.inf
        return {"b": b, "H": H, "e": e, "V": V, "E": float(e[0]), "d0": d0, "C": V[:, :d0], "d0_at_tolerance": d0_tol,
                "gap": gap, "multiplet_splitting": float(e[d0 - 1] - e[0]), "E_multiplet_mean": float(np.mean(e[:d0])),
                "gap_block": gap, "gap_cluster": float(e[d0] - e[d0 - 1]) if len(e) > d0 else np.inf,
                "levels_low": [float(x) for x in e[:min(len(e), d0 + 2)]]}

    def dF_dk(self, z, lam=1.0, b=1.0, st_src_t=None):
        n, K = self.n, self.K
        psi, N = z[:n], z[n:]
        Kpb = K @ (psi - b * self.one)
        dC = -psi * Kpb
        dG = K @ self.one - N * Kpb - K @ (N * psi)
        return np.concatenate([dC, dG])


def newton_fd(sys_, z0, lam=1.0, b=1.0, tol=1e-12, maxit=12, accept=1e-11):
    """Version 5's Newton (FD Jacobian, best iterate, noise-floor stop) with the strict line search and exceptions caught."""
    z = np.array(z0, float)
    res = []
    best_z, best_r = z.copy(), np.inf
    nj = 0
    last_corr = None

    def out(zr, ok, **kw):
        d = {"ok": bool(ok), "residuals": res, "jacobians": nj, "best": best_r, "last_correction": last_corr}
        d.update(kw)
        return zr, d
    for it in range(maxit):
        try:
            Fz = sys_.F(z, lam, b)[0]
        except Exception as e:  # noqa: BLE001
            return out(best_z, best_r < accept, it=it, why="exception", error=repr(e))
        r = float(np.linalg.norm(Fz))
        res.append(r)
        if r < best_r:
            best_z, best_r = z.copy(), r
        if r < tol:
            return out(z, True, it=it, why="converged")
        if len(res) >= 3 and best_r < 1e-10 and res[-1] > 0.5 * res[-3]:
            return out(best_z, best_r < accept, it=it, why="noise floor")
        try:
            Jm = jacobian_fd(sys_, z, lam, b)
            nj += 1
            dz = np.linalg.solve(Jm, -Fz)
        except np.linalg.LinAlgError:
            return out(best_z, best_r < accept, it=it, why="singular")
        except Exception as e:  # noqa: BLE001
            return out(best_z, best_r < accept, it=it, why="exception", error=repr(e))
        last_corr = float(np.linalg.norm(dz))
        step = 1.0
        accepted = False
        while step > 1e-4:
            zn = z + step * dz
            if np.all(zn > 0):
                try:
                    rn = float(np.linalg.norm(sys_.F(zn, lam, b)[0]))
                except Exception:  # noqa: BLE001
                    rn = np.inf
                if rn < r or r < 1e-9:
                    accepted = True
                    break
            step /= 2
        if not accepted:
            return out(best_z, best_r < accept, it=it, why="line search")
        z = zn
    try:
        Fz = sys_.F(z, lam, b)[0]
        r = float(np.linalg.norm(Fz))
        res.append(r)
        if r < best_r:
            best_z, best_r = z.copy(), r
    except Exception as e:  # noqa: BLE001
        return out(best_z, best_r < accept, it=maxit, why="exception", error=repr(e))
    return out(best_z, best_r < accept, it=maxit, why="iterations")


def returned_vector_record(sys_, zz, zref=None, lam=1.0, b=1.0):
    rec = {"min_component": float(np.min(zz))}
    if zref is not None:
        rec["distance_to_reference"] = float(np.linalg.norm(zz - zref))
    try:
        Fz = sys_.F(zz, lam, b)[0]
        rec["own_residual"] = float(np.linalg.norm(Fz))
        Jm = jacobian_fd(sys_, zz, lam, b)
        rec["correction_at_returned"] = float(np.linalg.norm(np.linalg.solve(Jm, Fz)))
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    return rec


def continuation_fd(sys_, b=1.0, dlam0=0.25, min_dlam=1e-4, accept_path=1e-9, accept_final=1e-11):
    n = sys_.n
    z = np.concatenate([b * sys_.one, sys_.one])
    if sys_.d0_fixed is None:
        sys_.d0_fixed = sys_.state(z[:n], z[n:])["d0_at_tolerance"]
    lam, dlam = 0.0, dlam0
    path = [{"lam": 0.0}]
    while lam < 1.0 - 1e-15:
        lt = min(1.0, lam + dlam)
        zn, info = newton_fd(sys_, z, lt, b, accept=accept_path)
        if info["ok"] and np.all(zn > 0):
            z, lam = zn, lt
            st = sys_.state(z[:n], z[n:])
            sv = np.linalg.svd(jacobian_fd(sys_, z, lam, b), compute_uv=False)
            path.append({"lam": lam, "min_psi": float(z[:n].min()), "min_N": float(z[n:].min()),
                         "residual": info["residuals"][-1], "newton": len(info["residuals"]) - 1, "jacobians": info["jacobians"],
                         "gap_cluster": st["gap_cluster"], "gap_block": st["gap_block"], "sigma_min_J": float(sv[-1])})
            if len(info["residuals"]) <= 5:
                dlam = min(2 * dlam, 0.5)
        else:
            dlam /= 2
            if dlam < min_dlam:
                return z, {"ok": False, "lost_at_lam": lam, "path": path}
    zf, info = newton_fd(sys_, z, 1.0, b, maxit=20, accept=accept_final)
    if np.all(zf > 0):
        z = zf
    return z, {"ok": bool(info["ok"]), "path": path, "final_newton": {"residuals": info["residuals"], "best": info.get("best"),
                                                                     "why": info.get("why")},
               "final_residual_best": float(min(info["residuals"]))}
