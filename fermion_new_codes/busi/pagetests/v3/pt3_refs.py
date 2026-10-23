#!/usr/bin/env python
"""pt3_refs.py (busi, PAGE design v3 sec. 2): the five reference arms, all post-processing ONE shared propagation C(t), h(t) on prescribed fields
(A11; reviewer B pc9 Test B arm 5). Changing the reporting reference never changes C: the arms only read spectra of h(t) and, at their start, C.

Every arm keeps occupations n_alpha on the eigenvectors |alpha> of h(t) (a full spectrum on the dense fixtures and the paper's 4096-site parent, or
a window spectrum lo < e < hi on long chains, outside which the occupations are the Killing ones, theta(-e)); P(t) = sum_alpha n_alpha |alpha><alpha|.

R-an   (v3 sec. 2, A1, F1): occupations attached to the rank-ordered branches of h(t) within each symmetry sector (the staggered sectors of
       page_w1_sector_v1d; label 0 where no symmetry); start occupations given (the stored reference's occupations in the eigenbasis of h(T_hold),
       or round <n|C(T_hold)|n>, the source printed). Printed per step: the minimum same-sector gap overall and between an occupied and an empty
       branch; an exact crossing (gap below tol_exact) is FLAGGED, never assumed absent. mode="overlap": the overlap lift (Hungarian maximum
       |<old|new>|^2 within each sector), the formation audit's second lift (F1a).
R-code (F2): the paper's production continuation reproduced from the engine code: engine_ops_v2.continue_sea_v2 (lines 107-152, unchanged copy)
       with its assignment assign_max_overlap (lines 55-99): per record, dH the change of h_m from the previous accepted record, s_T =
       ||dH||_G/tau, tau_LZ = 1/(c_LZ sqrt(s_T)); tau_LZ <= tau: nsub = max(1, round(c_LZ sqrt(tau ||dH||_G))) sub-steps along the linear field
       path; else the window of the last m = min(window_max, round(tau_LZ/tau)) accepted records; the occupied count exact. The schedule of
       engine_collapse_v19.py: the reference initialized as P_-(h_0) with gap e[k] - e[k-1] (lines 1115-1121), continued once per record of
       spacing tau = 8 a (--tau, line 1594) by continue_sea_v2 (line 1237), the window appended after acceptance and cut to --window-max = 4
       (lines 1358-1362, 1614); the three crossing settings are the c_LZ ladder 1/2, 1, 2 (engine spec 21.2, engine_clz_ladder_v1.py line 2),
       c_LZ = 1 the production (--c-LZ, line 1613; the record's json label "Landau-Zener matched maximum overlap (c_LZ = 1, window 4)").
R-loc  (reviewer A pc9 secs. 1.3, 1.4): analytic following (max-overlap within sector) except at an avoided crossing between an eigenvector of interior
       weight w_in = <alpha|Q|alpha> > 1/2 and one with w_in < 1/2 whose weights straddle 1/2 within the step and whose gap there is below the
       declared minigap delta_c: the occupations are transported by localization (the eigenvector of larger w_in keeps the inner level's occupation).
       Every jump Tr(h dP) at a transported crossing is recorded (-+2|V| for a two-level crossing); crossings with a gap above delta_c are kept
       analytic and listed. R_Q (the partition radius of Q) and delta_c are scanned by running one arm per value.
R-star (reviewer B pc9 sec. 1.1): P_star = 1_(-inf,0)(h_0), zero modes empty, fixed; prints ||[h, P_star]||_F and the polarization charge
       Tr h (P_-(h(t)) - P_star).
R-Q    (F3; diagnostic, CONFLICT C3): Q the sites R <= R_Q; P = (occupations on eigenvectors of QhQ, rank-lifted within the block and sector from
       round <n|C(T_hold)|n>, printed unrounded) (+) theta(-omega) on eigenvectors of (1-Q)h(1-Q); no cross block. Printed at t_0: the cut-bond
       energy Tr h_0 (P_0 - P_wall).
The absorber run hosts R-Q and R-star only: R-an, R-loc and R-code refuse a non-Hermitian h (require_hermitian)."""
import math

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

import pt3_core as K


def require_hermitian(h, tol=1e-12):
    H = h if not sp.issparse(h) else h
    d = abs(H - H.conj().T)
    m = float(d.max()) if sp.issparse(d) else float(np.max(d))
    if m > tol:
        raise ValueError("non-Hermitian h (defect %.2e): the eigenstate-defined references R-an, R-loc and R-code are undefined (F3)" % m)
    return m


def projector(U, occ):
    occ = np.asarray(occ, float)
    return (U * occ[None, :]) @ U.conj().T


def w_in(U, qmask):
    return np.real((np.abs(U[qmask, :]) ** 2).sum(axis=0))


class Spec:
    """one step's spectrum: e (m,) ascending, U (d, m), sec (m,) sector labels, t, f (the fields, for R-code), h (sparse, optional)."""

    def __init__(self, e, U, sec=None, t=None, f=None, h=None):
        self.e = np.asarray(e, float)
        self.U = np.asarray(U)
        self.sec = np.zeros(len(self.e), int) if sec is None else np.asarray(sec, int)
        self.t, self.f, self.h = t, f, h


def _sector_ranks(spec):
    return {s: np.where(spec.sec == s)[0][np.argsort(spec.e[spec.sec == s], kind="stable")] for s in np.unique(spec.sec)}


def _gaps(spec, occ):
    """the minimum same-sector adjacent gap overall and between an occupied and an empty branch (occupations differing by > 1/2)."""
    g_all, g_oe = math.inf, math.inf
    for s, idx in _sector_ranks(spec).items():
        if len(idx) < 2:
            continue
        d = np.diff(spec.e[idx])
        g_all = min(g_all, float(d.min()))
        diff = np.abs(np.diff(occ[idx])) > 0.5
        if diff.any():
            g_oe = min(g_oe, float(d[diff].min()))
    return g_all, g_oe


class ArmBase:
    name = "arm"

    def __init__(self):
        self.log = []

    def P(self):
        return projector(self.spec.U, self.occ)

    def record(self, **kw):
        kw.setdefault("t", getattr(getattr(self, "spec", None), "t", None))
        self.log.append(kw)


class RAn(ArmBase):
    """R-an: rank lift within sector (mode 'rank') or the overlap lift (mode 'overlap')."""

    def __init__(self, mode="rank", tol_exact=1e-10, source="given"):
        super().__init__()
        self.mode, self.tol_exact, self.source = mode, float(tol_exact), source
        self.name = "R-an" if mode == "rank" else "R-an(overlap)"
        self.flags = []

    def start(self, spec, occ):
        self.spec = spec
        self.occ = np.asarray(occ, float).copy()
        self.rank_occ = {s: self.occ[idx].copy() for s, idx in _sector_ranks(spec).items()}
        g_all, g_oe = _gaps(spec, self.occ)
        self.record(min_gap=g_all, min_gap_occ_empty=g_oe, init_source=self.source)

    def step(self, spec):
        old = self.spec
        occ = np.zeros(len(spec.e))
        if self.mode == "rank":
            for s, idx in _sector_ranks(spec).items():
                ro = self.rank_occ.get(s)
                if ro is None or len(ro) != len(idx):
                    raise ValueError("%s: sector %d holds %s levels, the start held %s (window edge or sector change)" % (self.name, s, len(idx), None if ro is None else len(ro)))
                occ[idx] = ro
        else:
            for s in np.unique(spec.sec):
                io = np.where(old.sec == s)[0]; jn = np.where(spec.sec == s)[0]
                O = np.abs(old.U[:, io].conj().T @ spec.U[:, jn]) ** 2
                ri, ci = linear_sum_assignment(-O)
                occ[jn[ci]] = self.occ[io[ri]]
        g_all, g_oe = _gaps(spec, occ)
        flag = bool(g_oe < self.tol_exact * max(1.0, float(np.max(np.abs(spec.e)))))
        if flag:
            self.flags.append({"t": spec.t, "gap_occ_empty": g_oe})
        dE = _tr_h_dP(old, self.occ, spec, occ)
        self.spec, self.occ = spec, occ
        self.record(min_gap=g_all, min_gap_occ_empty=g_oe, exact_crossing_flag=flag, tr_h_dP=dE)
        return occ


def _tr_h_dP(old, occ_old, new, occ_new):
    """Tr(h_mid (P_new - P_old)) with h_mid = (h_old + h_new)/2 (sparse h on the specs), on the dense or window representation; None when no h."""
    if old.h is None or new.h is None:
        return None
    hm = 0.5 * (old.h + new.h)
    a = np.real(np.einsum("ij,ij->j", new.U.conj(), hm @ new.U)) @ occ_new
    b = np.real(np.einsum("ij,ij->j", old.U.conj(), hm @ old.U)) @ occ_old
    return float(a - b)


class RLoc(ArmBase):
    """R-loc with partition mask qmask (sites R <= R_Q, both components) and the declared minigap delta_c."""

    def __init__(self, qmask, delta_c, R_Q=None):
        super().__init__()
        self.qmask = np.asarray(qmask, bool)
        self.delta_c = float(delta_c)
        self.R_Q = R_Q
        self.name = "R-loc(R_Q=%s, delta_c=%.3g)" % (R_Q, delta_c)
        self.jumps = []
        self.kept_analytic = []
        self.deferred = []

    def start(self, spec, occ):
        self.spec = spec
        self.occ = np.asarray(occ, float).copy()
        self.w = w_in(spec.U, self.qmask)
        self.char = self.w > 0.5
        self.record(n_inner=int(self.char.sum()))

    def step(self, spec):
        """v3.1 rule (reviewer B pdes3 item 8): (1) the analytic map old -> new by maximum overlap within each sector; (2) for every pair (j, k) of
        new levels adjacent in energy within a sector, with gap < delta_c, whose analytic sources (i_j, i_k) EXCHANGE character over the step
        (the more interior of the old pair maps to the less interior of the new pair) and whose inner member is localized inside (w_in > 1/4
        at both ends): transport by localization (the new more-interior vector takes the old more-interior one's occupation); (3) the event is
        located where the pair's weights are equal, by linear interpolation of the weight difference over the step, and the jump is the trace
        at the event, (p_a - p_b)(e_b - e_a) with the event energies interpolated likewise (step-convergent; the +-2|V| of a two-level crossing
        only for unequal unit occupations; zero for equal occupations, e.g. empty-empty); (4) tie rule: equal new weights (to 1e-12) defer
        the decision to the next step. The as-applied trace Tr(h_k dP) is recorded beside it (the ledger uses the as-applied P)."""
        old = self.spec
        w_new = w_in(spec.U, self.qmask)
        src = np.full(len(spec.e), -1)
        for s in np.unique(spec.sec):
            io = np.where(old.sec == s)[0]; jn = np.where(spec.sec == s)[0]
            O = np.abs(old.U[:, io].conj().T @ spec.U[:, jn]) ** 2
            ri, ci = linear_sum_assignment(-O)
            src[jn[ci]] = io[ri]
        occ_an = self.occ[src]
        occ = occ_an.copy()
        used = set()
        cands = []
        for s in np.unique(spec.sec):
            idx = np.where(spec.sec == s)[0]
            idx = idx[np.argsort(spec.e[idx], kind="stable")]
            for j, k in zip(idx[:-1], idx[1:]):
                gap = float(spec.e[k] - spec.e[j])
                d_old = self.w[src[j]] - self.w[src[k]]
                d_new = w_new[j] - w_new[k]
                if abs(d_new) < 1e-12:
                    if gap < self.delta_c and abs(d_old) > 1e-12:
                        self.deferred.append({"t": spec.t, "gap": gap})
                    continue
                if d_old * d_new < 0 and max(self.w[src[j]], self.w[src[k]]) > 0.25 and max(w_new[j], w_new[k]) > 0.25:
                    cands.append((gap, j, k, d_old, d_new))
        for gap, j, k, d_old, d_new in sorted(cands):
            if j in used or k in used:
                continue
            used.add(j); used.add(k)
            x = d_old / (d_old - d_new)                               # the event's fraction of the step
            ej = old.e[src[j]] + x * (spec.e[j] - old.e[src[j]]); ek = old.e[src[k]] + x * (spec.e[k] - old.e[src[k]])
            if gap < self.delta_c:
                occ[j], occ[k] = occ_an[k], occ_an[j]
                dE_event = float((occ[j] - occ_an[j]) * ej + (occ[k] - occ_an[k]) * ek)
                dE_applied = float((occ[j] - occ_an[j]) * spec.e[j] + (occ[k] - occ_an[k]) * spec.e[k])
                self.jumps.append({"t": spec.t, "t_event_fraction": float(x), "gap_event": float(ek - ej), "gap_step": gap, "tr_h_dP": dE_event,
                                   "tr_h_dP_applied": dE_applied, "e_event": [float(ej), float(ek)], "w_in_new": [float(w_new[j]), float(w_new[k])],
                                   "transported": bool(occ[j] != occ_an[j])})
            else:
                self.kept_analytic.append({"t": spec.t, "gap": gap})
        dE_total = _tr_h_dP(old, self.occ, spec, occ)
        self.spec, self.occ, self.w = spec, occ, w_new
        self.record(tr_h_dP=dE_total, n_jumps=len(self.jumps))
        return occ

    def jump_sum(self):
        return float(sum(j["tr_h_dP"] for j in self.jumps))

    def jump_sum_applied(self):
        return float(sum(j["tr_h_dP_applied"] for j in self.jumps))


class RCode(ArmBase):
    """R-code: engine_ops_v2.continue_sea_v2 at the production schedule. family: h_lin(dN, dv) (host sparse, linear in the fields) and
    eig_at(N, v) -> (e, U) numpy ascending (full or window). Torch on the CPU only (the engine functions take torch tensors)."""

    def __init__(self, family, tau, c_LZ=1.0, window_max=4):
        super().__init__()
        import torch
        import engine_ops_v2 as OPS2
        self.torch, self.OPS2 = torch, OPS2
        self.family, self.tau, self.c_LZ, self.window_max = family, float(tau), float(c_LZ), int(window_max)
        self.name = "R-code(c_LZ=%g, tau=%g, window %d)" % (c_LZ, tau, window_max)

    def _T(self, x):
        return self.torch.as_tensor(np.asarray(x))

    def start(self, spec, occ):
        """the engine's initialization (engine_collapse_v19 lines 1115-1121): ref0 = the occupied columns, the gap from the assignment's
        occupied-to-empty gap between blocks (= e[k] - e[k-1] for the Killing vacuum)."""
        occ = np.asarray(occ) > 0.5
        self.n_sea = int(occ.sum())
        U = self._T(spec.U); e = self._T(spec.e)
        ref0 = U[:, self.torch.as_tensor(np.where(occ)[0])]
        _, info, _ = self.OPS2.assign_max_overlap(ref0, e, U, self.n_sea)
        self.window = [{"ref": ref0, "f": spec.f, "gap": info["gap_between_blocks"]}]
        self.spec, self.occ, self.ref = spec, occ.astype(float), ref0
        self.record(gap=info["gap_between_blocks"])

    def step(self, spec):
        fam = self.family
        h_of = lambda ch, dN, dv: fam.h_lin(dN, dv)
        def eig_at(ch, N, v):
            e_, U_ = fam.eig_at(N, v)
            return self._T(e_), self._T(U_)
        ref, info, mask = self.OPS2.continue_sea_v2(fam, self.window, spec.f, self._T(spec.e), self._T(spec.U), self.n_sea, self.tau, self.c_LZ,
                                                    eig_at, h_of, self.window_max)
        self.window.append({"ref": ref, "f": spec.f, "gap": info.get("gap_between_blocks")})
        self.window = self.window[-self.window_max:]
        old = self.spec
        Rn = ref.numpy()
        # occupations on spec's eigenbasis: the weights of each eigenvector on ran(ref) (0/1 away from degenerate blocks)
        occ = np.real((np.abs(spec.U.conj().T @ Rn) ** 2).sum(axis=1))
        dE = _tr_h_dP(old, self.occ, spec, occ)
        self.spec, self.occ, self.ref = spec, occ, ref
        self.record(mode=info.get("mode"), nsub=info.get("nsub"), m=info.get("m_eff"), verified=info.get("verified"), tau_LZ=info.get("tau_LZ"),
                    min_occ_weight=info.get("min_occupied_weight"), max_empty_weight=info.get("max_empty_weight"), tr_h_dP=dE)
        return occ

    def P(self):
        Rn = self.ref.numpy()
        return Rn @ Rn.conj().T


class RStar(ArmBase):
    """R-star: P_star fixed (dense matrix on the fixtures, or columns)."""

    def __init__(self, P_star):
        super().__init__()
        self.Ps = np.asarray(P_star)
        self.name = "R-star"

    def start(self, spec, occ=None):
        self.spec = spec
        self.record(**self.diag(spec))

    def diag(self, spec):
        out = {}
        if spec.h is not None:
            H = spec.h.toarray() if sp.issparse(spec.h) else spec.h
            Cm = H @ self.Ps - self.Ps @ H
            out["comm_F"] = float(np.linalg.norm(Cm))
            Pm = projector(spec.U, (spec.e < 0).astype(float))
            out["polarization_charge"] = float(np.real(np.trace(H @ (Pm - self.Ps))))
        return out

    def step(self, spec):
        self.spec = spec
        self.record(tr_h_dP=0.0, **self.diag(spec))
        return None

    def P(self):
        return self.Ps


class RQ(ArmBase):
    """R-Q: the site partition at R_Q (qmask over the full dimension); dense fixtures and the engine parent."""

    def __init__(self, qmask, R_Q=None, sector_op=None):
        super().__init__()
        self.qmask = np.asarray(qmask, bool)
        self.R_Q = R_Q
        self.S = sector_op
        self.name = "R-Q(R_Q=%s)" % R_Q

    def _blocks(self, h):
        H = h.toarray() if sp.issparse(h) else np.asarray(h)
        qi = np.where(self.qmask)[0]; qo = np.where(~self.qmask)[0]
        Hin = H[np.ix_(qi, qi)]; Hout = H[np.ix_(qo, qo)]
        Sin = None if self.S is None else (self.S.toarray() if sp.issparse(self.S) else self.S)[np.ix_(qi, qi)]
        ein, Uin, sin = K.dense_spectrum(Hin, Sin)
        eo, Uo = np.linalg.eigh(Hout)
        return qi, qo, Spec(ein, Uin, sin), eo, Uo

    def start(self, h, C):
        qi, qo, sin, eo, Uo = self._blocks(h)
        Cin = np.asarray(C)[np.ix_(qi, qi)]
        nu = np.real(np.einsum("ij,ik,kj->j", sin.U.conj(), Cin, sin.U))
        self.unrounded = nu.copy()
        self.inner = RAn("rank", source="round <n|C(T_hold)|n> inside QhQ")
        self.inner.start(sin, np.rint(np.clip(nu, 0, 1)))
        self.qi, self.qo = qi, qo
        self._build(sin, eo, Uo)
        self.record(unrounded_occupations=[float(x) for x in nu], min_gap_inner=self.inner.log[-1]["min_gap"])

    def _build(self, sin, eo, Uo):
        d = len(self.qmask)
        P = np.zeros((d, d), complex)
        P[np.ix_(self.qi, self.qi)] = projector(sin.U, self.inner.occ)
        P[np.ix_(self.qo, self.qo)] = projector(Uo, (eo < 0).astype(float))
        self._P = P

    def step(self, h):
        _, _, sin, eo, Uo = self._blocks(h)
        self.inner.step(sin)
        self._build(sin, eo, Uo)
        self.record(min_gap_inner=self.inner.log[-1]["min_gap"], min_gap_inner_occ_empty=self.inner.log[-1]["min_gap_occ_empty"],
                    exact_crossing_flag=self.inner.log[-1]["exact_crossing_flag"])

    def P(self):
        return self._P

    @staticmethod
    def cut_bond_energy(h0, qmask):
        """Tr h_0 (P_0 - P_wall), P_0 = theta(-h_0), P_wall = theta(-Q h_0 Q) (+) theta(-(1-Q) h_0 (1-Q)); both routes (projectors, eigenvalue
        sums) returned."""
        H = h0.toarray() if sp.issparse(h0) else np.asarray(h0)
        e, U = np.linalg.eigh(H)
        P0 = projector(U, (e < 0).astype(float))
        qi = np.where(qmask)[0]; qo = np.where(~np.asarray(qmask, bool))[0]
        ei, Ui = np.linalg.eigh(H[np.ix_(qi, qi)]); eo, Uo = np.linalg.eigh(H[np.ix_(qo, qo)])
        Pw = np.zeros_like(P0)
        Pw[np.ix_(qi, qi)] = projector(Ui, (ei < 0).astype(float)); Pw[np.ix_(qo, qo)] = projector(Uo, (eo < 0).astype(float))
        by_proj = float(np.real(np.trace(H @ (P0 - Pw))))
        by_eigs = float(e[e < 0].sum() - ei[ei < 0].sum() - eo[eo < 0].sum())
        return by_proj, by_eigs


class LinearFamily:
    """h(f) = sum_k f_k A_k for generic fixtures (f = (N, v) with N the coefficient vector and v unused zeros), the R-code family interface."""

    def __init__(self, mats):
        self.A = [sp.csr_matrix(a) for a in mats]

    def h(self, f):
        N = np.asarray(f[0], float)
        out = self.A[0] * N[0]
        for k in range(1, len(self.A)):
            out = out + self.A[k] * N[k]
        return out.tocsr()

    def h_lin(self, dN, dv):
        return self.h((dN, dv))

    def eig_at(self, N, v):
        H = self.h((N, v)).toarray()
        return np.linalg.eigh(H)


class ChainFamily:
    """the chain h[N, v] (pt3_core.ChainOps) as the R-code family; eig_at full (sector route) or a window (lo, hi)."""

    def __init__(self, ops, window=None):
        self.ops, self.window = ops, window

    def h_lin(self, dN, dv):
        return self.ops.h(dN, dv)

    def eig_at(self, N, v):
        h = self.ops.h(N, v)
        if self.window is None:
            e, U, _ = K.full_spectrum_from_sector(h)
        else:
            e, U, _ = K.window_spectrum_banded(h, *self.window)
        return e, U
