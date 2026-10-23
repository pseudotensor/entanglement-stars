"""Strand 2b on R4, library version 8 (spec SPEC_strand2b_row.md sec. 15 item 6 (ii), (iv); builder analyst).

Everything of strand2b_lib_v7 unchanged (imported), with System's Newton and continuation replaced:
- newton: version 7's (the strict line search: no rejected step is executed) with an exception raised while evaluating F or the
  Jacobian caught inside the routine, which then stops with the best iterate and the exception recorded (why = "exception"); every
  stop carries why (converged, noise floor, line search, singular, exception, iterations), the best residual and the last computed
  Newton correction ||dz|| (version 7 left it out of the early returns);
- returned_vector_record: at a returned vector, its own residual ||F||, the Newton correction ||J^-1 F|| there, the smallest
  component and the distance to a reference solution (an exception recorded);
- continuation: version 7's, recording at every accepted step the cluster gap e[d_0] - e[d_0 - 1], the block gap e[d_0] - e[0] and
  sigma_min of the Jacobian."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand2b_lib_v7 as V7  # noqa: E402
from strand2b_lib_v7 import *  # noqa: E402,F401,F403


class System(V7.System):
    def newton(self, z0, lam=1.0, b=1.0, tol=1e-12, maxit=12, accept=1e-11):
        z = np.array(z0, float)
        res = []
        best_z, best_r = z.copy(), np.inf
        last_corr = None

        def out(zr, ok, **kw):
            d = {"ok": bool(ok), "residuals": res, "best": best_r, "last_correction": last_corr}
            d.update(kw)
            return zr, d
        for it in range(maxit):
            try:
                Fz, st, _, _ = self.F(z, lam, b)
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
                Jm, _, _ = self.J(z, lam, b)
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
                        rn = float(np.linalg.norm(self.F(zn, lam, b)[0]))
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
            Fz = self.F(z, lam, b)[0]
            r = float(np.linalg.norm(Fz))
            res.append(r)
            if r < best_r:
                best_z, best_r = z.copy(), r
        except Exception as e:  # noqa: BLE001
            return out(best_z, best_r < accept, it=maxit, why="exception", error=repr(e))
        return out(best_z, best_r < accept, it=maxit, why="iterations")

    def returned_vector_record(self, zz, zref=None, lam=1.0, b=1.0):
        rec = {"min_component": float(np.min(zz))}
        if zref is not None:
            rec["distance_to_reference"] = float(np.linalg.norm(zz - zref))
        try:
            Fz = self.F(zz, lam, b)[0]
            rec["own_residual"] = float(np.linalg.norm(Fz))
            Jm, _, _ = self.J(zz, lam, b)
            rec["correction_at_returned"] = float(np.linalg.norm(np.linalg.solve(Jm, Fz)))
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
        return rec

    def continuation(self, b=1.0, dlam0=0.25, min_dlam=1e-4, accept_path=1e-9, accept_final=1e-11):
        n = self.n
        z = np.concatenate([b * self.one, self.one])
        if self.d0_fixed is None:
            self.d0_fixed = self.state(z[:n], z[n:])["d0_at_tolerance"]
        lam, dlam = 0.0, dlam0
        path = [{"lam": 0.0, "min_psi": float(z[:n].min()), "min_N": float(z[n:].min())}]
        while lam < 1.0 - 1e-15:
            lt = min(1.0, lam + dlam)
            zn, info = self.newton(z, lt, b, accept=accept_path)
            if info["ok"] and np.all(zn > 0):
                z, lam = zn, lt
                st = self.state(z[:n], z[n:])
                try:
                    sv = np.linalg.svd(self.J(z, lam, b)[0], compute_uv=False)
                    smin = float(sv[-1])
                except Exception:  # noqa: BLE001
                    smin = None
                path.append({"lam": lam, "min_psi": float(z[:n].min()), "min_N": float(z[n:].min()),
                             "residual": info["residuals"][-1], "newton": len(info["residuals"]) - 1,
                             "gap_cluster": st["gap_cluster"], "gap_block": st["gap_block"], "sigma_min_J": smin})
                if len(info["residuals"]) <= 5:
                    dlam = min(2 * dlam, 0.5)
            else:
                dlam /= 2
                if dlam < min_dlam:
                    return z, {"ok": False, "lost_at_lam": lam, "path": path}
        zf, info = self.newton(z, 1.0, b, maxit=20, accept=accept_final)
        if np.all(zf > 0):
            z = zf
        return z, {"ok": bool(info["ok"]), "path": path, "final_newton": {"residuals": info["residuals"], "best": info.get("best"),
                                                                         "why": info.get("why"), "stopped": info.get("why")},
                   "final_residual_best": float(min(info["residuals"]))}
