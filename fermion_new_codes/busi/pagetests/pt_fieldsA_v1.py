#!/usr/bin/env python
"""pt_fieldsA_v1.py (busi, inbox #428, Test A): the chain fields of reviewer A's prescribed-recession runs (ReviewA_pc6_v1.md sec. 4).

The record: a rooted r_s/a 64 history from the REDSHIFT inventory (read_history_v1's registry, md5-checked), by default P64_s4_B2n4096
(c19 on 4096 sites, the reference collapse in a doubled box, roots at every record from 18.375 to 100) at its last record T/r_s 100,
the clock from which the paper's held-field control runs (07_page_curve.tex: "held from 100 to 500").

Formation (0 <= t <= T_rec): the engine's applied (back-averaged) site fields N_applied, v_applied on its sites 1..n_e, interpolated
linearly between the record and the next (the paper's production light-field propagation, page_history_v6q --interp linear), with the
vacuum Painleve-Gullstrand exterior N = 1, v = -(2 g E_k/R)^(1/2) appended beyond n_e (v interpolated with the records).

After T_rec (the hold or the recession): the enclosed charge lowered by hand at dE/dt = -L_eff, the fields re-solved from the rows:
the record's shell sources (eps_tot, J_tot) scaled by lambda, re-solved by the field module of record (engine_fields_v16
solve_constraints_ivp, collar flag, support 1e-14, md5 7a69faad: the inventory's dense route) and averaged back onto the sites by its
site_fields_kernel_average (the engine's site_fields = averaged), E(lambda) the re-solve's collar charge; lambda(t) from
E(lambda(t)) = E(1) - L_eff (t - T_rec). BUILDER CHOICES (stated in the delivery): (i) the charge is lowered by a uniform scaling of
the record's sources, which keeps the profile's shape and a regular centre and makes the exterior exactly the vacuum at E(t); (ii) the
fields are re-solved on a lambda grid (33 nodes spanning the largest run's charge loss) and interpolated by cubic splines in lambda
(checked against direct re-solves at the midpoints), not re-solved at every propagation step; (iii) the change is applied on top of the
record's applied fields, F(t) = F_applied(T_rec) + [F_resolved(lambda(t)) - F_resolved(1)], so the held control (L_eff = 0) is the
paper's held convention exactly and the recession has no jump at T_rec (the re-solve's lambda = 1 fields against the stored site rows
and the applied rows are printed). The exterior beyond n_e follows the same convention with -(2 g E/R)^(1/2)."""
import json
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_history_v1 as RH  # noqa: E402

RD = torch.float64


def load_record(label, T_rec_rs):
    reg = RH.registry()
    e = reg[label]
    for k in ("npz", "json"):
        got = RH.md5(e[k])
        if got != e[k + "_md5"]:
            raise ValueError("%s %s md5 %s != registry %s" % (label, k, got, e[k + "_md5"]))
    J = json.load(open(e["json"]))
    Z = np.load(e["npz"], allow_pickle=True)
    rs_nom = float(J["args"]["rs"])
    T = np.asarray(Z["T"], float)
    recs = {round(float(r["T"]), 6): r for r in J["records"] if isinstance(r, dict) and r.get("T") is not None}
    E = np.array([float(recs[round(float(t), 6)]["E"]) for t in T])
    k_rec = int(np.argmin(np.abs(T - T_rec_rs * rs_nom)))
    if abs(T[k_rec] - T_rec_rs * rs_nom) > 1e-6:
        raise ValueError("no record at T/r_s %g" % T_rec_rs)
    rec = recs[round(float(T[k_rec]), 6)]
    out = {"label": label, "npz": e["npz"], "json": e["json"], "npz_md5": e["npz_md5"], "json_md5": e["json_md5"], "rs_nom": rs_nom,
           "T": T, "E": E, "k_rec": k_rec, "T_rec": float(T[k_rec]), "E_rec": float(E[k_rec]), "g": float(Z["g"]), "ell": float(Z["ell"]),
           "tau": float(Z["tau"]), "R_sites": np.asarray(Z["R"], float), "N_app": np.asarray(Z["N_applied"], float),
           "v_app": np.asarray(Z["v_applied"], float), "N_site": np.asarray(Z["N"][k_rec], float), "v_site": np.asarray(Z["v"][k_rec], float),
           "eps_tot": np.asarray(Z["eps_tot"][k_rec], float), "J_tot": np.asarray(Z["J_tot"][k_rec], float),
           "rec": {k: rec.get(k) for k in ("T", "E", "E_over_E0", "R_t", "kappa_T", "four_gE_kappa_T", "trapping_zeros", "min_N_minus_absv",
                                           "r_min_N_minus_absv", "R_c", "converged")}}
    return out


def resolve_table(H, lam_grid, log=print):
    """the fields re-solved from the record's sources scaled by lambda: per lambda the averaged site fields, E, the dense zeros."""
    FM = RH.field_module()
    Ns, vs, Es, zs, mins = [], [], [], [], []
    for lam in lam_grid:
        _, _, _, info = FM.solve_constraints_ivp(lam * H["eps_tot"], lam * H["J_tot"], H["g"], H["R_sites"], H["ell"], collar_policy="flag",
                                                 support_rel=1e-14)
        if info.get("failed"):
            raise RuntimeError("re-solve failed at lambda %.6f: %s" % (lam, info["failed"]))
        Na, va, _ = FM.site_fields_kernel_average(info, H["R_sites"], H["ell"])
        tz = FM.trapping_radius(None, None, H["R_sites"], info)
        Ns.append(Na); vs.append(va); Es.append(float(info["E_collar"])); zs.append([float(z) for z in tz["zeros"]]); mins.append(tz.get("min"))
        log("  lambda %.6f: E %.6f, dense zeros %s, min(N - |v|) %s" % (lam, Es[-1], ["%.3f" % z for z in zs[-1]], mins[-1]))
    return {"lam": np.asarray(lam_grid, float), "N": np.array(Ns), "v": np.array(vs), "E": np.array(Es), "zeros": zs, "min": mins}


class FieldsA:
    """the chain's site fields (N, v) on n chain sites at any clock t (lattice units), formation or post-record."""

    def __init__(self, H, n_chain, device, L_eff=0.0, table=None):
        self.H, self.n, self.device = H, int(n_chain), torch.device(device)
        self.L = float(L_eff)
        self.ne = len(H["R_sites"])
        self.g = H["g"]
        R = np.arange(1, self.n + 1, dtype=float)
        self.ext = torch.as_tensor(np.sqrt(2.0 * self.g / R[self.ne:]), dtype=RD, device=self.device)
        self.T, self.E, self.k_rec, self.T_rec = H["T"], H["E"], H["k_rec"], H["T_rec"]
        self.sqE = np.sqrt(self.E)
        self.table = table
        if self.L > 0.0:
            if table is None:
                raise ValueError("a recession needs the re-solved table")
            lam, E = table["lam"], table["E"]
            o = np.argsort(E)
            self.lam_of_E = CubicSpline(E[o], lam[o])
            self.Nspl = CubicSpline(lam, table["N"], axis=0)
            self.vspl = CubicSpline(lam, table["v"], axis=0)
            self.E1 = float(CubicSpline(lam, E)(1.0))
            self.N1 = self.Nspl(1.0); self.v1 = self.vspl(1.0)
            self.E_lo = float(E.min())
        self.N_rec = H["N_app"][self.k_rec]; self.v_rec = H["v_app"][self.k_rec]

    def charge(self, t):
        if t <= self.T_rec or self.L == 0.0:
            return float(np.interp(min(t, self.T_rec), self.T, self.E))
        return self.E1 - self.L * (t - self.T_rec)

    def lam(self, t):
        if t <= self.T_rec or self.L == 0.0:
            return 1.0
        E = self.charge(t)
        if E < self.E_lo:
            raise ValueError("the charge %.6f left the re-solved table (E_min %.6f)" % (E, self.E_lo))
        return float(self.lam_of_E(E))

    def host(self, t):
        """(N_engine (n_e,), v_engine (n_e,), c) with the exterior v = -c (2 g/R)^(1/2) beyond n_e."""
        if t < self.T_rec - 1e-9:
            tau = self.H["tau"]
            k = int(math.floor(t / tau + 1e-12))
            k = min(max(k, 0), self.k_rec - 1)
            s = (t - self.T[k]) / (self.T[k + 1] - self.T[k])
            Ne = (1 - s) * self.H["N_app"][k] + s * self.H["N_app"][k + 1]
            ve = (1 - s) * self.H["v_app"][k] + s * self.H["v_app"][k + 1]
            c = (1 - s) * self.sqE[k] + s * self.sqE[k + 1]
            return Ne, ve, c
        if self.L == 0.0:
            return self.N_rec, self.v_rec, self.sqE[self.k_rec]
        lam = self.lam(t)
        Ne = self.N_rec + (self.Nspl(lam) - self.N1)
        ve = self.v_rec + (self.vspl(lam) - self.v1)
        c = self.sqE[self.k_rec] + (math.sqrt(self.E1 - self.L * (t - self.T_rec)) - math.sqrt(self.E1))
        return Ne, ve, c

    def device_fields(self, t):
        Ne, ve, c = self.host(t)
        N = torch.ones(self.n, dtype=RD, device=self.device)
        v = torch.empty(self.n, dtype=RD, device=self.device)
        N[: self.ne] = torch.as_tensor(Ne, dtype=RD, device=self.device)
        v[: self.ne] = torch.as_tensor(ve, dtype=RD, device=self.device)
        v[self.ne:] = -c * self.ext
        return N, v

    def numpy_fields(self, t):
        Ne, ve, c = self.host(t)
        R = np.arange(1, self.n + 1, dtype=float)
        N = np.ones(self.n); v = np.empty(self.n)
        m = min(self.n, self.ne)
        N[:m] = Ne[:m]; v[:m] = ve[:m]
        if self.n > self.ne:
            v[self.ne:] = -c * np.sqrt(2.0 * self.g / R[self.ne:])
        return N, v

    def exterior_callables(self, t):
        """N(R), v(R) on the vacuum exterior (R beyond n_e) at clock t, for the packets."""
        _, _, c = self.host(t)
        g = self.g
        return (lambda R: np.ones_like(np.asarray(R, float))), (lambda R: -c * np.sqrt(2.0 * g / np.asarray(R, float)))
