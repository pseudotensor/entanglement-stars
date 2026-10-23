#!/usr/bin/env python
"""pt3_ledger.py (busi, PAGE design v3 sec. 5; A4, F7): the finite-radius discrete ledger identity, one per reference, on the actual generator.

For a reference P(t) and the subtracted state Delta = C - P, with h_R = sum_{x <= R} eps_x (the back-averaged generator's energy density summed
inside the radius R; sum_x eps_x = h) and the shared propagation i dC/dt = [h, C]:

  d/dt M_R = Tr(Delta i[h, h_R]) + Tr(P i[h, h_R]) + Tr(Delta dh_R/dt) - Tr(dP/dt h_R) + (moving partition),   M_R = Tr(Delta h_R)
             '-- inflow of Delta  '-- interface      '-- imposed work     '-- reference response

  (i[h, h_R] is the energy current into the region; Tr(P i[h, h_R]) = i Tr(h_R [P, h]) vanishes for a commuting reference and is R-star's
  [h, P_star] term; -Tr(dP h_R) is the -Tr(Pdot e_x)-type response, zero for R-star; at an R-loc crossing it holds the jump.)
  The exterior storage X_R = Tr(Delta (h - h_R)) obeys the same with h - h_R, so d/dt (M_R + X_R) = Tr(Delta dh/dt) - Tr(dP/dt h)
  (+ Tr(P i[h, h]) = 0), and for R-star reviewer B pc9's driven identity d/dt Tr[(C - P_star) h] = Tr[(C - P_star) dh/dt].
  Each term appears once; C is the complete covariance (no block truncation).

Ledger samples the terms at the ledger cadence (<= 5 r_s, F7) and closes the identity by the trapezoid rule; the reference response is the
exact difference -[Tr(P h_R)] + int Tr(P dh_R/dt) dt (jumps included), so the residual measures the propagation and the cadence; the held control's
residual is the floor of all residuals (F7). Tolerances relative to the emitted energy (one percent, A4) are applied by pt3_readers.
C and P may be dense matrices, occupied columns (C = Phi Phi^dag), or WindowP (a sea trace provider plus a window correction)."""
import math

import numpy as np
import scipy.sparse as sp

import pt3_core as K


class Cols:
    def __init__(self, Phi, f=None):
        self.Phi = np.asarray(Phi)
        self.f = None if f is None else np.asarray(f, float)


class WindowP:
    """P = P_-(h) + sum_win (n_alpha - theta(-e_alpha)) |alpha><alpha|; sea(O) returns Tr(P_-(h) O)."""

    def __init__(self, sea, U, occ, e):
        self.sea, self.U, self.dn = sea, np.asarray(U), np.asarray(occ, float) - (np.asarray(e) < 0).astype(float)


def tr(Rep, O):
    """Tr(Rep O) for a Hermitian sparse or dense O."""
    if isinstance(Rep, Cols):
        OX = O @ Rep.Phi
        v = np.real(np.einsum("ij,ij->j", Rep.Phi.conj(), OX))
        return float(v.sum() if Rep.f is None else v @ Rep.f)
    if isinstance(Rep, WindowP):
        corr = np.real(np.einsum("ij,ij->j", Rep.U.conj(), O @ Rep.U)) @ Rep.dn
        return float(Rep.sea(O) + corr)
    A = np.asarray(Rep)
    if sp.issparse(O):
        return float(np.real((O.multiply(A.T)).sum()))
    return float(np.real(np.trace(A @ O)))


def kpm_local_sea_trace(h, ws, w=None, tol=1e-14):
    """the production sea provider: Tr(P_-(h) O) ~ Tr(f_ws(h) O) by exact local probes on O's support (one Chebyshev series per support vector);
    exact to the series tolerance when the window correction covers every eigenvalue with f_ws(e) != theta(-e) to that tolerance. Returns a
    function of O (host sparse). The window correction then uses n - f_ws(e) in place of n - theta(-e): use WindowP with the returned dn_fix."""
    import pt3_prep as PR
    def sea(O):
        Oc = sp.csr_matrix(O)
        rows = np.unique(np.concatenate([Oc.indices, np.repeat(np.arange(Oc.shape[0]), np.diff(Oc.indptr))]))
        E = np.zeros((h.shape[0], len(rows)), complex)
        E[rows, np.arange(len(rows))] = 1.0
        Y, _ = PR.sharp_step_apply(h, E, ws, w, tol)        # columns f(h) e_j
        Osub = Oc[rows][:, rows].toarray()
        return float(np.real(np.trace(Osub @ Y[rows, :])))   # sum_{ij} O_ij f_ji
    return sea


class WindowPKPM(WindowP):
    """WindowP for the KPM sea: the correction is n - f_ws(e) (the smoothed sea's window occupations replaced by the arm's)."""

    def __init__(self, sea, U, occ, e, ws):
        import pt3_prep as PR
        self.sea, self.U = sea, np.asarray(U)
        self.dn = np.asarray(occ, float) - PR.f_ws(np.asarray(e), ws)


class Ledger:
    """samples per arm at the cadence; ops(t) -> dict(h, h_R, hdot, hdot_R) host sparse; R may move (moving-partition term booked as a jump)."""

    def __init__(self, arms):
        self.arms = list(arms)
        self.S = {a: {k: [] for k in ("t", "M", "X", "Fin_D", "Int", "W_in", "W_out", "MP", "XP", "WP_in", "WP_out", "part", "part_P", "R")} for a in self.arms}

    def sample(self, t, ops, C, Ps, R=None, R_prev_ops=None):
        h, hR, hd, hdR = ops["h"], ops["h_R"], ops["hdot"], ops["hdot_R"]
        J = K.commutator_i(h, hR)
        hX = (h - hR).tocsr(); hdX = (hd - hdR).tocsr()
        cJ = tr(C, J); cM = tr(C, hR); cX = tr(C, hX); cW = tr(C, hdR); cWx = tr(C, hdX)
        for a in self.arms:
            P = Ps[a]
            pJ = tr(P, J); pM = tr(P, hR); pX = tr(P, hX); pW = tr(P, hdR); pWx = tr(P, hdX)
            s = self.S[a]
            part = part_P = 0.0
            if R_prev_ops is not None:                      # the moving partition: Tr(Delta (h_R(new) - h_R(old))) at fixed state and time
                dh = (hR - R_prev_ops).tocsr()
                part_P = tr(P, dh)
                part = tr(C, dh) - part_P
            s["t"].append(t); s["M"].append(cM - pM); s["X"].append(cX - pX); s["Fin_D"].append(cJ - pJ); s["Int"].append(pJ)
            s["W_in"].append(cW - pW); s["W_out"].append(cWx - pWx); s["MP"].append(pM); s["XP"].append(pX); s["WP_in"].append(pW)
            s["WP_out"].append(pWx); s["part"].append(part); s["part_P"].append(part_P); s["R"].append(R)

    @staticmethod
    def _trap(t, y):
        t = np.asarray(t); y = np.asarray(y)
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t))) if len(t) > 1 else 0.0

    def close(self, E_scale=None):
        out = {}
        for a, s in self.S.items():
            t = np.asarray(s["t"])
            I = lambda k: self._trap(t, s[k])
            dM = s["M"][-1] - s["M"][0]
            dX = s["X"][-1] - s["X"][0]
            part = float(np.sum(s["part"]))
            pP = float(np.sum(s["part_P"]))                 # the reference's own share of the partition moves is not a response
            ref_in = -(s["MP"][-1] - s["MP"][0] - pP) + I("WP_in")
            ref_out = -(s["XP"][-1] - s["XP"][0] + pP) + I("WP_out")
            rhs_M = I("Fin_D") + I("Int") + I("W_in") + ref_in + part
            # exterior storage: its inflow is minus the region's (the generator's total current vanishes)
            rhs_X = -I("Fin_D") - I("Int") + I("W_out") + ref_out - part
            tot_rhs = I("W_in") + I("W_out") + ref_in + ref_out
            r = {"dM_retained": dM, "inflow_Delta": I("Fin_D"), "outflow_Delta_physical_current": -I("Fin_D"), "interface_Int": I("Int"),
                 "imposed_work_in": I("W_in"), "imposed_work_out": I("W_out"), "reference_response_in": ref_in, "reference_response_out": ref_out,
                 "moving_partition": part, "dX_exterior_storage": dX, "residual_retained": dM - rhs_M, "residual_exterior": dX - rhs_X,
                 "residual_total": (dM + dX) - tot_rhs, "n_samples": len(t), "cadence_max": float(np.max(np.diff(t))) if len(t) > 1 else None,
                 "max_abs_Int_rate": float(np.max(np.abs(s["Int"]))) if s["Int"] else None}
            if E_scale:
                r["residual_over_E"] = abs(r["residual_retained"]) / abs(E_scale)
            out[a] = r
        return out
