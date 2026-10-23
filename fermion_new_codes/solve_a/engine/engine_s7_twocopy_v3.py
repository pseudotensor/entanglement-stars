#!/usr/bin/env python
"""engine_s7_twocopy_v3.py (builder analyst; engine spec 36.3 (2), turn 44: "engine_s7_twocopy_v3 (a new file; the CPU; the tau 8 member to
T/rs 20.0 only): the exact bound beyond R_t(t) evaluated after projecting the difference onto the half-zones |ap| below and above pi/2
separately (the projector by the chain's discrete Fourier transform on the sites beyond R_t), both printed per record from T0 to the closing,
beside the unsplit bound; a reading, not a criterion; read in turn 45 against the held clause"; the brief lead/BRIEF_s7_v3_builder.md:
"engine_s7_twocopy_v3.py (a new file; never edit v2) with 36.3 (2)'s half-zone split (the branch-resolved bound beyond R_t(t)) and everything
else as v2; its gate against v2 on the records both compute"; "both members as v2 did"). Version 2 is the code extended: its text is copied
here verbatim and every quantity it computed keeps its code path (the split reads G before version 2's deletion of it and writes nothing
back), so the gate against version 2 is bitwise.

What changes against version 2 (everything else is version 2's):
(1) The half-zone split (36.3 (2)), per record from T0 while the surface persists, per configuration: on the region beyond R_t(t) (the sites
i0 .. n-1 with R > R_t(t), M = n - i0 of them, two spinor components per site), the chain's discrete Fourier transform per spinor component,
F[q, j] = exp(-2 pi i q j / M)/sqrt(M), the momentum a p_q = 2 pi q_s / M (q_s the signed index in numpy's fftfreq order, (-pi, pi]);
G_hat = F G_D F^dagger (G_D the copy-off-diagonal difference block G restricted to the region; Delta C_D = 2 G_D between the copies);
the half-zone projectors Pi_lo (|a p| < pi/2) and Pi_hi (|a p| >= pi/2); printed: the exact bounds 2 ||Pi_lo G_D Pi_lo||_op and
2 ||Pi_hi G_D Pi_hi||_op (the largest |eigenvalue| of the Hermitian i G_hat on each half-zone's block) beside version 2's unsplit
2 ||G_D||_op. The momenta at |a p| = pi/2 exactly (present when 4 divides M) join the upper half-zone (the builder's convention; their
count printed). The chain is the sine chain (h(p) = m sz + sin(a p) sx + (kappa/R) sy, the two-step Q giving sin(2 a p)/2): |a p| = pi/2
separates the valley at p = 0 from the doubler's valley at a p = pi.
(2) Beside it, labelled contrasts that source nothing: the cross bound 2 ||Pi_lo G_D Pi_hi||_op; the one-sided momentum weight
w(q) = sum |G_hat[q, .]|^2 (rows; equal to the columns' by anti-Hermiticity) as shares of ||G_D||_F^2 over the half-zones and over the
quarter-zones [0, pi/4), [pi/4, pi/2), [pi/2, 3 pi/4], (3 pi/4, pi] (vf10's band boundaries pi/4 and 3 pi/4 joining the middle bands, so the
middle share is vf10's cos 2ap < 0 bands); the local moments E[cos a p] = Re sum_j <g_(j+1), g_j> / ||G_D||_F^2 and E[cos 2 a p] (g_j the
rows of site j; nearest and next-nearest site overlaps, no wrap-around: free of the transform's periodic edge); Parseval's identity and
the compression check (each split bound at most the unsplit one, the unsplit one at most max(lo, hi) + cross) printed; the same split
beyond R_t(T0) (version 2's fixed-site contrast; every record, the closing slice included) and over the whole chain (the preparation's
own momentum content from T0 on: the 4-site ramp's seeding).
(3) --T-end-over-rs: the run ends at the last stored slice at or before it (36.3 (2): T/rs 20.0); --records stays version 2's.
(4) Compose: version 2's md and json for the records computed, the split's tables, and the gate against version 2 (--gate-v2 the
version 2 member jsons in the members' order): every numeric field version 2 stored, on the records both compute, compared bitwise
(the head with the labels and the law, the replay rows, the S7 rows, the charge series, the crossing clocks); the gate's verdict is the
unsplit bound beyond R_t(t) bitwise equal on every common record, the other fields' counts printed beside it.

Args (new): --T-end-over-rs; compose: --gate-v2 v2_primary.json[,v2_control.json]. Every number in the outputs is printed by the script.

Version 2's docstring:
engine_s7_twocopy_v2.py (builder analyst; engine spec 33.4 (2), S7 part 3, binding, turn 41: "Part 3 (engine_s7_twocopy_v2, a new file;
the box CPU, both members, about 2.5 hours): T0 the stored tau 8 record in [17.5, 18.5] with the largest R_t - r_in (the episode 16.5 to 20.0;
tau 4's 16.125 to 19.5), the same T0 on the tau 4 member; theta from the quadratic law so that the unrotated collar charge difference stays
below 1e-6 of E through the run (about 0.017), printed; the window at the dust density's interior minimum within [r_in, R_t] if one exists,
else centred in the annulus, the profile printed, w 4; each preparation's reference the record's continued P(t) (+) P(t) from the first record
(the model's rule), the rotated one at T0 a labelled contrast; per record the region bound at r = R_t(t) while the surface persists, at r 32,
64, 128, 256, 512, the fronts, the collar detector; the crossing clocks (the first records where the bound beyond R_t(t) exceeds 1e-3 and
1e-6) against the closing clock"; the brief lead/BRIEF_s7_part3_builder.md: "Everything else as v1 (the replays, the audit, the step
control)"). Version 1 (engine_s7_twocopy_v1.py, part 2) is the code extended; its helpers are copied here verbatim.

What changes against version 1 (everything else is version 1's):
(1) T0: the stored record inside --T0-range (17.5, 18.5 in r_s) holding both trapping zeros with the largest R_t - r_in (the candidates
printed); --T0-over-rs or --match-json (the primary's T0) override. The episode about T0 printed: the first and last records of the run of
records holding R_t that contains T0, and the closing clock, the first stored slice after T0 holding no R_t (the surface persists while
R_t is held; a later reappearance is printed, never joined).
(2) The window: chi_X as version 1 (the C^2 ramp of 28.5 (b), w --width 4); X at the dust shell density's interior minimum within
[r_in, R_t] of T0's record (a site strictly inside the interval whose density lies below both neighbours; the deepest if several), else
X = (r_in + R_t)/2, the annulus' centre; the profile and the rule taken printed; --windows or --match-json override.
(3) Theta: per kappa the labels at the --law-thetas (pi/8, pi/16, pi/32, pi/64) give the quadratic law of the unrotated collar charge
difference, dE/E = a theta^2 + b theta^4 (least squares; the pure quadratic coefficient, the log-log slopes and the residuals printed;
the rotated contrast fitted beside it); theta is the law's root at --law-margin of --charge-tol (0.8 of 1e-6: the builder's margin for
the growth through the run, labelled), the smallest over kappa, common to both channels; --theta or --match-json override. The label at
that theta is the full version 1 label (+ and - solved). Through the run the unrotated collar charge difference is solved by the run's
own solver (engine_collapse_v17.solve, the member's total sources with the two copies replaced by the + preparation's) at T0, every
--solve-every-rs r_s after it and at the end, printed against the bar; the law's theta for the bar over the run (theta (1e-6/max)^(1/2))
printed beside it.
(4) The reference: each preparation's reference is P(t) (+) P(t), the record's continued reference in both copies, from the first record
(T0) on (the model's rule: part 2 found the rotated reference continued by the model returns to P(t) (+) P(t) at the first or second record);
the rotated reference R (P_0 (+) P_0) R^dagger at T0 is a labelled contrast (its source difference and its solved charge difference), and
its continuation by the model's rule on the two-copy block (version 1's route) is followed as a contrast for at most --rule-check-max records
after T0 or until its projector distance to P(t) (+) P(t) is below 1e-8 (the distances printed; it sources nothing). The audit's source
kinds are therefore P (the run's sources) and P (+) P (the preparations').
(5) Per record from T0, per configuration: the region bound 2 ||G_{R > R_t(t)}||_op (exact: the largest |eigenvalue| of i G restricted to the
sites beyond the record's R_t; = the largest ||Delta C_D||_op of any detector beyond the surface) while the surface persists, beside it its
Frobenius upper bound; the same exact bound beyond R_t(T0) (the site fixed at T0's surface: a contrast that separates the difference's
outward motion from the surface's recession) at every record; the exact bound at r 32, 64, 128, 256, 512 (--ladder-radii) at every record
(version 1: every 1 r_s); the fronts (version 1's: the outermost r whose Frobenius bound exceeds 1e-12, 1e-9, 1e-6, 1e-3); D_det and the
collar's bound (version 1's detector).
(6) The crossing clocks: per configuration the first record from T0 at which the exact bound beyond R_t(t) exceeds 1e-3 and 1e-6 while the
surface persists, against the closing clock (T_close - T0 and the last record holding the surface); beside them, as contrasts, the same
clocks from the Frobenius bound and beyond R_t(T0) (also after the closing).
(7) Compose: the md and json of the deliverable from both members' jsons; the inputs of 33.4 (3)'s sentence printed (the charge
difference, the time before the closing, the crossing time and its bound, the step control's crossing difference), or its "otherwise"
branch.

Args (new or changed): --T0-range --law-thetas --law-margin --theta --solve-every-rs --rule-check-max; version 1's --member --T0-over-rs
--windows --match-json --kappas --charge-tol --width --ladder-radii --omega-over-kappa --mp-over-kappa --accept --gpu --threads --records
--out-json --out-npz; --compose a.json,b.json --md --out-json --md-every-rs. Every number in the outputs is printed by the script."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import math
import re
import struct
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "flow"))
import engine_collapse_v5 as C5  # noqa: E402
import engine_collapse_v17 as E17  # noqa: E402
import flow_chart_v4 as V4  # noqa: E402

EPS_FRONT = (1e-12, 1e-9, 1e-6, 1e-3)
EPS_CROSS = (1e-3, 1e-6)
K_P = "P (the record's continued reference)"
K_PBAR = "P_bar (the propagated initial reference)"
K_PP = "P (+) P (the record's continued reference in both copies: each preparation's reference from the first record, the model's rule)"
K_ROT = "R (P_0 (+) P_0) R^dagger (the reference rotated at T0)"
K_HELD = "R (P(t) (+) P(t)) R^dagger (the reference held rotated)"
SOURCE_KINDS = (K_P, K_PP)


class Audit:
    """every source the script computes passes here: the reference's columns are an explicit argument, and the call is recorded with its
    label, its role ("source", or "contrast" which sources nothing) and which registered object the reference was (version 1's class;
    the source kinds are version 2's)."""

    def __init__(self):
        self.calls = {}

    def subtract(self, ch, Cpart, Pcols, label, role, known, mult=1.0, twocopy=False):
        """eps_x, J_x of C - mult * P_ref: Cpart the densities of C's columns, the reference's (single-copy, or two-copy and copy-summed)
        subtracted here; known = [(object, kind), ...], the first object identical to Pcols names the kind."""
        k_ = next((kind for obj, kind in known if obj is Pcols), "other")
        key = (label, role, k_)
        self.calls[key] = self.calls.get(key, 0) + 1
        pe, pj = cdens2(ch, Pcols, None) if twocopy else (C5.dens(ch.h, Pcols), C5.dens(ch.Q, Pcols))
        return Cpart[0] - mult * pe, Cpart[1] - mult * pj

    def table(self):
        rows, ok = [], True
        for (label, role, k_), c in sorted(self.calls.items()):
            good = (k_ in SOURCE_KINDS) if role == "source" else True
            ok = ok and good
            rows.append({"label": label, "role": role, "reference": k_, "calls": c,
                         "verdict": (("C - P" if k_ == K_P else "C_+- - P (+) P") if good else "FAIL") if role == "source" else "contrast (sources nothing)"})
        return rows, ok

def cdens(ch, cols):
    """the densities of C's columns: sum over (columns, f or None) of the engine's site densities of h and Q."""
    e = np.zeros(ch.hs.shape[0] // 2); j = np.zeros_like(e)
    for X, f in cols:
        e = e + C5.dens(ch.h, X, f); j = j + C5.dens(ch.Q, X, f)
    return e, j


def cdens2(ch, Y, f2):
    """the copy-summed densities of the two-copy columns Y (4n rows, copy-major): h and Q acting within each copy."""
    n2 = Y.shape[0] // 2
    e1, j1 = cdens(ch, [(Y[:n2], f2)]); e2, j2 = cdens(ch, [(Y[n2:], f2)])
    return e1 + e2, j1 + j2


def rel(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.max(np.abs(x - y))) / max(float(np.max(np.abs(y))), 1e-300)


def ramp(s):
    """28.5 (b)'s ramp s^3 (10 - 15 s + 6 s^2) on [0, 1], 0 below and 1 above (C^2)."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)


def window(R, X, w):
    """chi_X(r) = 1 - S((r - X)/w + 1/2): 1 for r <= X - w/2, 0 for r >= X + w/2, the transition C^2 and compact."""
    return 1.0 - ramp((np.asarray(R, float) - X) / w + 0.5)


def twocopy_cols(Phi, c, s, sign):
    """R_+- (Phi (+) Phi) = [[c Phi, +-s Phi], [-+s Phi, c Phi]] (rows copy-major; c, s per component as column vectors)."""
    torch = C5.BK["torch"]
    cP = c * Phi; sP = s * Phi
    if sign > 0:
        return torch.cat([torch.cat([cP, sP], 1), torch.cat([-sP, cP], 1)], 0)
    return torch.cat([torch.cat([cP, -sP], 1), torch.cat([sP, cP], 1)], 0)


def doubled(ref):
    """P (+) P's columns: [[ref, 0], [0, ref]]."""
    return C5.BK["torch"].block_diag(ref, ref)


def twocopy_eig(e, U):
    """the eigendecomposition of h_m (x) 1 from h_m's: each eigenvalue twice (adjacent, ascending), its vectors in copy 1 and copy 2."""
    torch = C5.BK["torch"]
    n2 = U.shape[0]
    e2 = torch.repeat_interleave(e, 2)
    U2 = torch.zeros((2 * n2, 2 * n2), dtype=U.dtype, device=U.device)
    U2[:n2, 0::2] = U; U2[n2:, 1::2] = U
    return e2, U2


def projdist(A, B):
    """||A A^dagger - B B^dagger||_F for orthonormal column sets A, B."""
    ov = float((A.conj().T @ B).abs().pow(2).sum())
    return math.sqrt(max(A.shape[1] + B.shape[1] - 2.0 * ov, 0.0))


def continue_twocopy(Pi_prev, e, U, n2sea):
    """the model's continuation (engine_collapse_v17's endpoint routes) of a two-copy reference to the eigenbasis of h_m (x) 1."""
    e2, U2 = twocopy_eig(e, U)
    ref, inf_ = E17.continue_fast_info(Pi_prev, e2, U2, n2sea)
    mode = "endpoint (fast)"
    if ref is None:
        ref, inf_ = E17.continue_reference_resolved_v17(Pi_prev, e2, U2, n2sea)
        mode = "endpoint (blocks)"
    info = {k_: v_ for k_, v_ in (inf_ or {}).items() if k_ not in ("occ_e", "emp_e")} if isinstance(inf_, dict) else {"why": inf_}
    del e2, U2
    return ref, mode, info


def build_detector(n, kappa_ch, rs_nom, okap, mpkap, lo):
    """J3''s probe packets (obs_raymap_solved_v5's construction) centred at the collar (site indices lo .. n - 1), restricted to its sites
    and renormalized; the orthonormal span F (2n x nd), the restricted packets W (2n x np) and the record."""
    kp = 1.0 / (2.0 * rs_nom)
    xD = 0.5 * ((lo + 1) + n)                      # the collar's centre radius (R = index + 1)
    xi = int(round(xD - 1))
    vext = -math.sqrt(rs_nom / xD)
    a_ = float(kappa_ch) / xD
    mask = np.zeros(2 * n, bool); mask[2 * lo:] = True
    rows, W = [], []
    for mk in mpkap:
        mp = mk * kp
        for r_ in okap:
            omega = r_ * kp; sg = omega / 6.0
            if mp > 0 and omega - 3 * sg < mp * (1 - 1e-9):
                rows.append({"mp_over_kappa": mk, "omega_over_kappa": r_, "skipped": "the packet's 3 sigma window reaches below the threshold m_p"})
                continue
            me = math.sqrt(mp * mp + a_ * a_)
            k, om, a2 = V4.branch_momenta(omega, sg, me, vext, outgoing=True)
            w = V4.build_packet(n, xi, k, a2, me, m=mp, a=a_)
            dk = np.gradient(k)
            kc = float(np.sum(a2 * k * dk) / np.sum(a2 * dk))
            vg = float(np.sum(a2 * V4.group_velocity(k, me, vext) * dk) / np.sum(a2 * dk))
            wr = np.where(mask, w, 0.0)
            ret = float(np.real(np.vdot(wr, wr)))
            wr = wr / math.sqrt(ret)
            rows.append({"mp_over_kappa": mk, "omega_over_kappa": r_, "omega": omega, "m_p": mp, "sigma_omega": sg, "k_c": kc, "v_g": vg,
                         "s_x_density": abs(vg) / (2 * sg), "retained_norm_in_collar": ret})
            W.append(wr)
    Wm = np.array(W).T
    ev = np.linalg.eigvalsh(Wm.conj().T @ Wm)
    keep = ev > 1e-10 * ev.max()
    # the orthonormal basis by column-pivoted Householder QR of the collar's rows alone, embedded with exact zeros elsewhere (a QR of the
    # full column would leave rounding-level weight on the chain's first rows, where Delta C is of order theta)
    Qc, Rc, piv = sla.qr(Wm[2 * lo:], mode="economic", pivoting=True)
    F = np.zeros((2 * n, int(keep.sum())), complex)
    F[2 * lo:] = Qc[:, :int(keep.sum())]
    info = {"kappa_p": kp, "x_D": xD, "x_index": xi, "v_ext": vext, "a": a_, "collar_sites": [lo + 1, n], "packets": rows,
            "gram_eigenvalues": [float(x) for x in ev], "n_detector": int(keep.sum()), "orthonormality": float(np.max(np.abs(F.conj().T @ F - np.eye(int(keep.sum())))))}
    return F, Wm, info


# ------------------------------------------------------------------------------------------------------------------ version 2's helpers
def fit_law(th, y):
    """the quadratic law of a charge difference in theta: y = a theta^2 + b theta^4 by least squares on the points given (sign-folded: the
    fit is of s y with s the sign of the largest |y|, mixed signs flagged), the pure quadratic coefficient a2 = sum(y theta^2)/sum(theta^4),
    the per-point coefficients y/theta^2, the log-log slopes between neighbouring thetas and the fits' largest relative residuals."""
    th = np.asarray(th, float); y = np.asarray(y, float)
    order = np.argsort(th)[::-1]
    th = th[order]; y = y[order]
    s = 1.0 if y[np.argmax(np.abs(y))] >= 0 else -1.0
    ys = s * y
    A = np.stack([th ** 2, th ** 4], axis=1)
    (a_, b_), *_ = np.linalg.lstsq(A, ys, rcond=None)
    a2 = float(np.sum(ys * th ** 2) / np.sum(th ** 4))
    fit = A @ np.array([a_, b_])
    slopes = [float(math.log(ys[i] / ys[i + 1]) / math.log(th[i] / th[i + 1])) if ys[i] > 0 and ys[i + 1] > 0 else None for i in range(len(th) - 1)]
    return {"thetas": [float(x) for x in th], "values": [float(x) for x in y], "sign": s, "mixed_signs": bool(np.any(ys < 0)),
            "a": float(a_), "b": float(b_), "a_pure": a2, "per_point_a": [float(v_ / t_ ** 2) for v_, t_ in zip(ys, th)], "loglog_slopes": slopes,
            "resid_ab": float(np.max(np.abs(ys - fit) / np.maximum(np.abs(ys), 1e-300))),
            "resid_pure": float(np.max(np.abs(ys - a2 * th ** 2) / np.maximum(np.abs(ys), 1e-300)))}


def theta_for(law, target):
    """the smallest positive root of a x + b x^2 = target, x = theta^2 (the stable form x = 2 target / (a + sqrt(a^2 + 4 b target))); the pure
    quadratic root where the discriminant fails or a is not positive."""
    a_, b_ = law["a"], law["b"]
    disc = a_ * a_ + 4.0 * b_ * target
    if a_ > 0 and disc > 0:
        return math.sqrt(2.0 * target / (a_ + math.sqrt(disc)))
    return math.sqrt(target / law["a_pure"])


def site_beyond(R, r):
    """the index of the first site whose radius exceeds r (the region R > r begins there; len(R) if none)."""
    return int(np.searchsorted(np.asarray(R, float), float(r), side="right"))


def opnorm_region(G, i0):
    """2 ||G restricted to the sites i0 .. n-1||_op, exact (the largest |eigenvalue| of the Hermitian i G there); 0 for an empty region."""
    torch = C5.BK["torch"]
    if 2 * i0 >= G.shape[0]:
        return 0.0
    blk = G[2 * i0:, 2 * i0:]
    return 2.0 * float(torch.linalg.eigvalsh(1j * blk).abs().max())


# ------------------------------------------------------------------------------------------------------------------ version 3's helpers
def zone_masks(M):
    """the signed index q_s of each output of an M-point DFT (numpy's fftfreq order; a p = 2 pi q_s / M in (-pi, pi]) and the zones by integer
    arithmetic on |q_s|: the half-zones |a p| < pi/2 (4 |q_s| < M) and |a p| >= pi/2 (the boundary |a p| = pi/2, 4 |q_s| = M, joins the upper
    half-zone: the builder's convention); the quarter-zones [0, pi/4), [pi/4, pi/2), [pi/2, 3 pi/4], (3 pi/4, pi] (vf10's boundaries pi/4 and
    3 pi/4 joining the middle bands); the count of boundary momenta at |a p| = pi/2."""
    qs = np.rint(np.fft.fftfreq(M) * M).astype(np.int64)
    aq = np.abs(qs)
    lo = 4 * aq < M
    quarters = [8 * aq < M, (8 * aq >= M) & (4 * aq < M), (4 * aq >= M) & (8 * aq <= 3 * M), 8 * aq > 3 * M]
    return qs, lo, ~lo, quarters, int(np.sum(4 * aq == M))


def halfzone(G, i0, cross=True, unsplit=None):
    """36.3 (2)'s split of the copy-off-diagonal difference G on the sites i0 .. n-1 (M sites, two spinor components per site, site-major):
    G_hat = F G_D F^dagger with the chain's DFT on the M sites per spinor component (F[q, j] = exp(-2 pi i q j/M)/sqrt(M): torch's fft along
    the row sites, ifft along the column sites, both orthonormal); the exact bounds 2 ||Pi G_D Pi||_op on the half-zones |a p| < pi/2 and
    >= pi/2 (the largest |eigenvalue| of the Hermitian i G_hat block); the contrasts: the cross bound 2 ||Pi_lo G_D Pi_hi||_op (cross),
    the one-sided momentum weight's shares (half- and quarter-zones), the local moments E[cos a p], E[cos 2 a p] (site overlaps without
    wrap-around), Parseval's identity, and against the unsplit bound (unsplit, if given) the compression check. G is read, never written."""
    torch = C5.BK["torch"]
    n = G.shape[0] // 2
    M = n - int(i0)
    if M <= 0:
        return None
    dev = G.device
    blk = G[2 * i0:, 2 * i0:].reshape(M, 2, M, 2)
    Gk = torch.fft.ifft(torch.fft.fft(blk, dim=0, norm="ortho"), dim=2, norm="ortho")
    qs, lo, hi, quarters, nb = zone_masks(M)
    il = torch.as_tensor(np.nonzero(lo)[0], dtype=torch.int64, device=dev)
    ih = torch.as_tensor(np.nonzero(hi)[0], dtype=torch.int64, device=dev)

    def sub(r, c):
        return Gk.index_select(0, r).index_select(2, c).reshape(2 * len(r), 2 * len(c))

    def herm_bound(idx):
        if len(idx) == 0:
            return 0.0
        return 2.0 * float(torch.linalg.eigvalsh(1j * sub(idx, idx)).abs().max())
    b_lo = herm_bound(il); b_hi = herm_bound(ih)
    b_x = (2.0 * float(torch.linalg.svdvals(sub(il, ih)).max()) if (len(il) and len(ih)) else 0.0) if cross else None
    w = (Gk.abs() ** 2).sum(dim=(1, 2, 3)).cpu().numpy()
    tot_k = float(w.sum()); tot_x = float((blk.abs() ** 2).sum())
    res = {"i0": int(i0), "M": M, "boundary_momenta": nb, "n_lo": int(lo.sum()), "n_hi": int(hi.sum()), "bound_lo": b_lo, "bound_hi": b_hi,
           "bound_cross": b_x, "fro2": tot_x, "parseval": (abs(tot_k - tot_x) / tot_x) if tot_x > 0 else None,
           "share_lo": None, "share_hi": None, "quarter_shares": None, "share_middle": None, "cos_ap": None, "cos_2ap": None}
    if tot_x > 0:
        res["share_lo"] = float(w[lo].sum()) / tot_k; res["share_hi"] = float(w[hi].sum()) / tot_k
        res["quarter_shares"] = [float(w[qm].sum()) / tot_k for qm in quarters]
        res["share_middle"] = res["quarter_shares"][1] + res["quarter_shares"][2]
        res["cos_ap"] = float((blk[1:].conj() * blk[:-1]).real.sum()) / tot_x
        res["cos_2ap"] = float((blk[2:].conj() * blk[:-2]).real.sum()) / tot_x if M > 2 else None
    if unsplit is not None:
        res["unsplit"] = unsplit
        res["check_lower"] = max(b_lo, b_hi, b_x or 0.0) - unsplit
        res["check_upper"] = (unsplit - (max(b_lo, b_hi) + b_x)) if b_x is not None else None
    return res


def member(a):
    C5.backend(a.gpu)
    torch = C5.BK["torch"]; dev = C5.BK["dev"]
    if a.gpu is None:
        torch.set_num_threads(a.threads)
    t_start = time.time()
    d, tag = os.path.split(a.member)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    if J.get("script") != "engine_collapse_v17.py":
        raise RuntimeError("the member %s was written by %s; this replay carries engine_collapse_v17's rule only" % (tag, J.get("script")))
    ra = J["args"]
    for key in ("collar_policy", "support_rel", "site_fields"):
        E17.OPT[key] = ra.get(key, E17.OPT[key])
    rs = float(ra["rs"]); tau = float(ra["tau"]); n = int(ra["nsites"]); m = float(ra["m"]); g = float(ra["g"]); ell = ra["ell"]
    chs = [C5.Chan(k, m, n, bool(ra.get("half_offset", 0))) for k in range(1, int(ra["kappa_sea"]) + 1)]
    byk = {ch.kappa: ch for ch in chs}
    R = chs[0].R
    shell = [float(q) * rs for q in ra["shell"].split(",")] if ra.get("shell") else None
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    Tz = np.asarray(Z["T"], float)
    K = len(Tz) if a.records is None else min(len(Tz), a.records)
    if a.T_end_over_rs is not None:                                     # version 3: the run ends at the last stored slice at or before it
        K = min(K, int(np.sum(Tz / rs <= a.T_end_over_rs + 1e-9)))
    match = json.load(open(a.match_json)) if a.match_json else None
    # T0 (33.4 (2)): the stored record in --T0-range holding both trapping zeros with the largest R_t - r_in
    lo_T, hi_T = [float(x) for x in a.T0_range.split(",")]
    cand = sorted([r for r in recs.values() if r.get("R_t") is not None and r.get("r_in") is not None and lo_T - 1e-9 <= float(r["T"]) / rs <= hi_T + 1e-9],
                  key=lambda r: float(r["T"]))
    cand_rows = [{"T_over_rs": float(r["T"]) / rs, "r_in": float(r["r_in"]), "R_t": float(r["R_t"]), "R_t_minus_r_in": float(r["R_t"]) - float(r["r_in"])} for r in cand]
    best = max(cand, key=lambda r: float(r["R_t"]) - float(r["r_in"])) if cand else None
    if a.T0_over_rs is not None:
        T0req = a.T0_over_rs; t0_rule = "--T0-over-rs %g" % a.T0_over_rs
    elif match:
        T0req = float(match["head"]["T0_over_rs"]); t0_rule = "--match-json %s (the primary's T0)" % os.path.basename(a.match_json)
    else:
        if best is None:
            raise RuntimeError("no record in T/rs [%g, %g] holds both trapping zeros" % (lo_T, hi_T))
        T0req = float(best["T"]) / rs
        t0_rule = "the stored record in T/rs [%g, %g] holding both trapping zeros with the largest R_t - r_in (33.4 (2))" % (lo_T, hi_T)
    hit = [r for r in recs.values() if abs(float(r["T"]) / rs - T0req) < 1e-9]
    if not hit:
        raise RuntimeError("no record at T/rs %g" % T0req)
    rec0 = hit[0]
    T0 = float(rec0["T"])
    k0 = int(np.argmin(np.abs(Tz - T0)))
    if abs(Tz[k0] - T0) > 1e-9 or k0 >= K:
        raise RuntimeError("T0 %.3f is not a stored slice within the records replayed" % (T0 / rs))
    # the surface per slice (R_t held), the episode about T0 and the closing clock: the first slice after T0 holding no R_t
    Rt_of = [recs.get(round(float(t), 6), {}).get("R_t") for t in Tz]
    rin_of = [recs.get(round(float(t), 6), {}).get("r_in") for t in Tz]
    no_record = [float(Tz[k]) / rs for k in range(len(Tz)) if round(float(Tz[k]), 6) not in recs]
    k_first = k0
    while k_first - 1 >= 0 and Rt_of[k_first - 1] is not None and Rt_of[k0] is not None:
        k_first -= 1
    k_close = next((k for k in range(k0, len(Tz)) if Rt_of[k] is None), None)
    k_last = (k_close - 1) if k_close is not None else len(Tz) - 1
    if Rt_of[k0] is None:
        k_last = None
    reappear = [float(Tz[k]) / rs for k in range(k_close + 1, len(Tz)) if Rt_of[k] is not None] if k_close is not None else []
    episode = {"surface_at_T0": Rt_of[k0] is not None, "first_T_over_rs": float(Tz[k_first]) / rs if Rt_of[k0] is not None else None,
               "last_T_over_rs": float(Tz[k_last]) / rs if k_last is not None else None,
               "closing_T_over_rs": float(Tz[k_close]) / rs if k_close is not None else None, "k_close": k_close, "k_last": k_last,
               "closing_minus_T0_rs": (float(Tz[k_close]) - T0) / rs if k_close is not None else None,
               "last_minus_T0_rs": (float(Tz[k_last]) - T0) / rs if k_last is not None else None, "reappearances_T_over_rs": reappear,
               "slices_without_record": no_record,
               "track": [{"T_over_rs": float(Tz[k]) / rs, "R_t": Rt_of[k], "r_in": rin_of[k]} for k in range(k0, (k_close + 1) if k_close is not None else len(Tz))]}
    i_Rt0 = site_beyond(R, Rt_of[k0]) if Rt_of[k0] is not None else None
    kappas = [int(x) for x in a.kappas.split(",")]
    law_thetas = [float(x) for x in a.law_thetas.split(",")]
    ladder = [int(x) for x in a.ladder_radii.split(",")]
    i_lad = {r: int(np.searchsorted(R, float(r), side="left")) for r in ladder}            # the region R >= r begins at this site (version 1's)
    width = float(match["head"]["window"]["width"]) if match else a.width
    cuts = sorted({int(r["M1_parts"]["cut"]) for r in recs.values() if r.get("M1_parts") and r["M1_parts"].get("cut") is not None})
    cut = int(rec0["M1_parts"]["cut"])
    collars = sorted({float(r["r_collar"]) for r in recs.values() if r.get("r_collar") is not None})
    det = {}
    for kap in kappas:
        F, Wm, dinfo = build_detector(n, kap, rs, [float(x) for x in a.omega_over_kappa.split(",")], [float(x) for x in a.mp_over_kappa.split(",")], cut)
        det[kap] = {"F": C5.T_(F), "W": C5.T_(Wm), "info": dinfo}
    ic = torch.arange(2 * cut, 2 * n, device=dev)                       # the collar's components (sites R = cut + 1 .. n)
    head = {"script": "engine_s7_twocopy_v3.py", "member": a.member, "tag": tag, "member_script": J.get("script"), "rs": rs, "tau": tau, "nsites": n, "m": m, "g": g,
            "ell": ell, "kappa_sea": len(chs), "T0": T0, "T0_over_rs": T0 / rs, "T0_rule": t0_rule, "T0_range": [lo_T, hi_T], "T0_candidates": cand_rows,
            "T0_best_in_range": (float(best["T"]) / rs) if best is not None else None, "k0": k0, "records": K, "T_end_over_rs": float(Tz[K - 1]) / rs,
            "trapping_zeros_at_T0": {"r_in": rec0.get("r_in"), "R_t": rec0.get("R_t")}, "E_stored_at_T0": rec0.get("E"), "episode": episode,
            "R_t0_site": (float(R[i_Rt0]) if (i_Rt0 is not None and i_Rt0 < n) else None), "kappas": kappas, "law_thetas": law_thetas, "law_margin": a.law_margin,
            "theta_arg": a.theta, "charge_tol": a.charge_tol, "solve_every_rs": a.solve_every_rs, "rule_check_max": a.rule_check_max, "ladder_radii": ladder,
            "ladder_first_sites": {str(r): float(R[i_lad[r]]) for r in ladder}, "m1_cuts": cuts, "r_collars": collars,
            "detector": {str(k): det[k]["info"] for k in kappas}, "accept": a.accept, "threads": a.threads, "gpu": a.gpu, "match_json": a.match_json,
            "T_end_arg": a.T_end_over_rs, "records_arg": a.records,
            "halfzone": {"spec": "36.3 (2)", "transform": "the chain's DFT on the region's sites per spinor component, F[q, j] = exp(-2 pi i q j/M)/sqrt(M), G_hat = F G_D F^dagger",
                         "zones": "|a p| < pi/2 and |a p| >= pi/2 (a p = 2 pi q_s/M; the boundary |a p| = pi/2 joins the upper half-zone, the builder's convention)",
                         "quarters": "[0, pi/4), [pi/4, pi/2), [pi/2, 3 pi/4], (3 pi/4, pi] (vf10's boundaries joining the middle bands)",
                         "regions": "beyond R_t(t) while the surface persists (36.3 (2)); beyond R_t(T0) and the whole chain (contrasts)"}}
    print("member %s (%s): r_s %g tau %g n %d; T0/rs %.3f (%s, k0 %d), %d records to T/rs %.3f; M1 cuts %s; collars %s" % (
        tag, J.get("script"), rs, tau, n, T0 / rs, t0_rule, k0, K, Tz[K - 1] / rs, cuts, collars), flush=True)
    print("T0 candidates in T/rs [%g, %g]: %s" % (lo_T, hi_T, "; ".join("%.3f: r_in %.4f, R_t %.4f, R_t - r_in %.4f" % (
        c_["T_over_rs"], c_["r_in"], c_["R_t"], c_["R_t_minus_r_in"]) for c_ in cand_rows)), flush=True)
    print("the surface: held from T/rs %s to %s about T0, closing (the first slice without R_t) at %s, %s r_s after T0; reappearances %s; slices without a record %s; R_t(T0) %s, its first site beyond R %s" % (
        fmt(episode["first_T_over_rs"], "%.3f"), fmt(episode["last_T_over_rs"], "%.3f"), fmt(episode["closing_T_over_rs"], "%.3f"),
        fmt(episode["closing_minus_T0_rs"], "%.3f"), reappear, no_record, fmt(Rt_of[k0], "%.4f"), fmt(head["R_t0_site"], "%.0f")), flush=True)
    for kap in kappas:
        di = det[kap]["info"]
        print("detector kappa %d: x_D %.2f, collar sites %s, %d packets, %d orthonormal modes (orthonormality %.1e), Gram eigenvalues %s, retained norms %s" % (
            kap, di["x_D"], di["collar_sites"], len(di["packets"]), di["n_detector"], di["orthonormality"], ", ".join("%.3e" % x for x in di["gram_eigenvalues"]),
            ", ".join("%.3e" % p_.get("retained_norm_in_collar", float("nan")) for p_ in di["packets"])), flush=True)
    AUD = Audit()
    E17.tm_reset()
    t_p = time.time()
    Nc, vc, dust, info0, head0 = E17.prepare_dust(chs, g, R, ell, rs / (2.0 * g * m), float(ra["R0_over_rs"]) * rs, m, ra["reading"], shell)
    Ncs = np.asarray(Z["N_cinit"], float); vcs = np.asarray(Z["v_cinit"], float)
    head["cinit"] = {"dN": float(np.max(np.abs(Nc - Ncs))), "dv": float(np.max(np.abs(vc - vcs))), "passes": head0.get("passes"), "stored_passes": J["head"].get("passes"),
                     "N_B": head0.get("N_B"), "E0": head0.get("E0"), "stored_E0": J["head"].get("E0"), "seconds": time.time() - t_p}
    print("C-init re-prepared in %.0f s: %s passes (stored %s), max |N - N_cinit| %.2e, |v - v_cinit| %.2e, E0 %.10f (stored %.10f)" % (
        head["cinit"]["seconds"], head0.get("passes"), J["head"].get("passes"), head["cinit"]["dN"], head["cinit"]["dv"], head0.get("E0"), J["head"].get("E0")), flush=True)
    st = {}
    for ch in chs:
        e, U = E17.eigT(E17.clocked_gpu(ch, Ncs, vcs))
        neg = e < 0
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * n, 0), dtype=torch.complex128, device=dev), np.zeros(0)))
        en = e.real.cpu().numpy(); kk = int(neg.sum())
        gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
        st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": np.asarray(f_d, float), "ref": U[:, neg].clone(), "pbar": U[:, neg].clone(), "n_sea": kk,
                        "gap_between_prev": gap, "levels_prev": (np.asarray(en[:kk], float), np.asarray(en[kk:], float), 0)}
        del e, U
    f_ref = (Ncs, vcs)
    replay_rows, s7_rows, cfgs, states = [], [], [], {}
    leak_sum = 0.0
    iw = None
    X = None; chi = None
    law = {}
    out = {"head": head, "replay": replay_rows, "s7": s7_rows}

    def dump():
        rows_a, ok_a = AUD.table()
        out["audit"] = {"rows": rows_a, "pass": ok_a}
        out["seconds"] = time.time() - t_start
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)

    def solve_total(eps, Jt):
        Na, va, W_, inf_ = E17.solve(eps, Jt, g, R, ell, None)
        return Na, va, (E17.FLD.charge(inf_, g) if not inf_.get("failed") else float("nan")), inf_.get("failed")

    def snapshot(label, cache_):
        snap = {}
        for kap in kappas:
            s_ = st[kap]
            for key in ("sea", "dust", "pbar"):
                snap["%s_k%d_%s" % (label, kap, key)] = s_[key].cpu().numpy()
            snap["%s_k%d_ref" % (label, kap)] = cache_[kap][2].cpu().numpy()
            snap["%s_k%d_f" % (label, kap)] = np.asarray(s_["f"], float)
        return snap

    def cs_of(chi_, th):
        c = torch.as_tensor(np.repeat(np.cos(th * chi_), 2), dtype=torch.complex128, device=dev)[:, None]
        s = torch.as_tensor(np.repeat(np.sin(th * chi_), 2), dtype=torch.complex128, device=dev)[:, None]
        return c, s

    def src_stats(es, js, e0, j0, X_, raw):
        """the copy-summed source (es, js) against 2 (C_0 - P) = (2 e0, 2 j0): relative to the comparator's largest value; the shells
        counted by part 1's convention, above 1e-15 of the raw (not normal-ordered) density's largest value, 2 max |raw| of C_0's columns
        (version 1's)."""
        sc_e = max(float(np.max(np.abs(2 * e0))), 1e-300); sc_j = max(float(np.max(np.abs(2 * j0))), 1e-300)
        rw_e = max(float(np.max(np.abs(2 * raw[0]))), 1e-300); rw_j = max(float(np.max(np.abs(2 * raw[1]))), 1e-300)
        de = np.abs(es - 2 * e0); dj = np.abs(js - 2 * j0)
        bad = np.nonzero((de > 1e-15 * rw_e) | (dj > 1e-15 * rw_j))[0]
        far = np.abs(R - X_) > width
        return {"eps": float(np.max(de)) / sc_e, "J": float(np.max(dj)) / sc_j, "eps_abs": float(np.max(de)), "J_abs": float(np.max(dj)),
                "far_eps": float(np.max(de[far])) / sc_e, "far_J": float(np.max(dj[far])) / sc_j, "shells_above_1e-15": int(len(bad)),
                "shells_first_last": [float(R[bad[0]]), float(R[bad[-1]])] if len(bad) else None, "scale_eps": sc_e, "scale_J": sc_j,
                "raw_scale_eps": rw_e, "raw_scale_J": rw_j, "net_eps": float(np.sum(es - 2 * e0)), "net_J": float(np.sum(js - 2 * j0))}

    for k in range(K):
        tr = time.time()
        T = float(Tz[k]); rec = recs.get(round(T, 6), {})
        N_app = np.asarray(Z["N_applied"][k], float); v_app = np.asarray(Z["v_applied"][k], float)
        force = bool(rec.get("verification_differs"))
        t_e = time.time()
        eig = E17.eigh_all(chs, N_app, v_app)
        t_eig = time.time() - t_e
        cache, modes = {}, {}
        smodes = rec.get("continuation_modes") or {}
        mode_differs = []
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            ref, ci = E17.continue_v17(ch, s_["ref"], f_ref, (N_app, v_app), e, U, s_["n_sea"], s_.get("gap_between_prev"), s_.get("levels_prev"), force_subdivide=force)
            if ref is None:
                raise RuntimeError("the replay's continuation failed at T/rs %.3f, kappa %d (%s)" % (T / rs, ch.kappa, ci.get("why")))
            cache[ch.kappa] = (e, U, ref, ci)
            md_ = ci.get("mode", "?")
            modes[md_] = modes.get(md_, 0) + 1
            if str(ch.kappa) in smodes and str(smodes[str(ch.kappa)]).split(";")[0] != md_:
                mode_differs.append(ch.kappa)
        del eig
        # the audit's sources at this record, every channel: C - P (the run's sources) and the contrast C - P_bar (version 1's)
        eP = np.zeros(n); jP = np.zeros(n); eB = np.zeros(n); jB = np.zeros(n)
        Cpart = {}
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            known = [(ref, K_P), (s_["pbar"], K_PBAR)]
            Cpart[ch.kappa] = cdens(ch, [(s_["sea"], None), (s_["dust"], s_["f"])])
            ea, ja = AUD.subtract(ch, Cpart[ch.kappa], ref, "replay: the member's total source, every channel (C - P)", "source", known)
            eb, jb = AUD.subtract(ch, Cpart[ch.kappa], s_["pbar"], "replay: the contrast, every channel (C - P_bar)", "contrast", known)
            eP += ch.w * ea; jP += ch.w * ja; eB += ch.w * eb; jB += ch.w * jb
        es = np.asarray(Z["eps_tot"][k], float); Js = np.asarray(Z["J_tot"][k], float)
        esd = es - np.asarray(Z["eps_pairs"][k], float); Jsd = Js - np.asarray(Z["J_pairs"][k], float)
        rr = {"k": k, "T_over_rs": T / rs, "CP_eps": rel(eP, es), "CP_J": rel(jP, Js), "CPbar_eps": rel(eB, es), "CPbar_J": rel(jB, Js),
              "CPbar_vs_dust_eps": rel(eB, esd), "CPbar_vs_dust_J": rel(jB, Jsd), "modes": modes, "modes_differ_from_stored": mode_differs, "force_subdivide": force,
              "pbar_minus_sea": max(float((st[c.kappa]["pbar"] - st[c.kappa]["sea"]).abs().max()) for c in chs), "eig_s": t_eig}
        rr["accepted"] = bool(max(rr["CP_eps"], rr["CP_J"]) <= a.accept)
        replay_rows.append(rr)
        if not rr["accepted"] and k <= k0:
            dump()
            raise RuntimeError("the replayed sources differ from the stored ones at T/rs %.3f (eps %.2e, J %.2e; acceptance %g)" % (T / rs, rr["CP_eps"], rr["CP_J"], a.accept))
        # the window, the law, theta and the preparations at T0 (33.4 (2))
        if k == k0:
            nd = np.zeros(n); ed = np.zeros(n); per = {}
            for ch in chs:
                s_ = st[ch.kappa]
                if s_["dust"].shape[1] == 0:
                    continue
                ft_ = torch.as_tensor(s_["f"], dtype=torch.float64, device=dev)
                occ = ((s_["dust"].abs() ** 2) * ft_[None, :]).sum(dim=1).reshape(n, 2).sum(dim=1).cpu().numpy()
                nd += ch.w * occ; ed += ch.w * C5.dens(ch.h, s_["dust"], s_["f"]); per[ch.kappa] = occ
            rin = rec0.get("r_in"); Rt = rec0.get("R_t")
            loc_min, glob = [], None
            if rin is not None and Rt is not None:
                inside = np.nonzero((R >= float(rin)) & (R <= float(Rt)))[0]
                loc_min = [int(i) for i in inside[1:-1] if nd[i] < nd[i - 1] and nd[i] < nd[i + 1]]
                glob = int(inside[int(np.argmin(nd[inside]))]) if len(inside) else None
            if a.windows:
                X = float(a.windows.split(",")[0]); wrule = "--windows"
            elif match:
                X = float(match["head"]["window"]["X"]); wrule = "--match-json %s (the primary's X)" % os.path.basename(a.match_json)
            else:
                if rin is None or Rt is None:
                    raise RuntimeError("the record at T0 holds no trapping zeros; give --windows or --match-json")
                if loc_min:
                    i_min = min(loc_min, key=lambda i: nd[i])
                    X = float(R[i_min])
                    wrule = "the dust shell density's interior minimum within [r_in, R_t] = [%.4f, %.4f] of T0's record (the deepest of %d)" % (rin, Rt, len(loc_min))
                else:
                    X = 0.5 * (float(rin) + float(Rt))
                    wrule = "centred in the annulus [r_in, R_t] = [%.4f, %.4f] of T0's record: the dust shell density has no interior minimum there" % (rin, Rt)
            chi = window(R, X, width)
            iw = torch.arange(0, 2 * min(n, int(math.ceil(X + 0.5 * width))), device=dev)     # chi > 0 only below X + w/2
            if rin is not None and Rt is not None:
                lo_p = int(max(1, math.floor(min(X, float(rin))) - 4)); hi_p = int(min(n, math.ceil(max(X, float(Rt))) + 4))
            else:
                lo_p = int(max(1, math.floor(X) - 8)); hi_p = int(min(n, math.ceil(X) + 8))
            prof = [{"R": float(R[i]), "dust_per_shell": float(nd[i]), "dust_energy_per_shell": float(ed[i]), "stored_dust_energy": float(esd[i]), "chi": float(chi[i]),
                     **{"dust_k%d" % kq: float(per[kq][i]) for kq in kappas if kq in per}} for i in range(lo_p - 1, hi_p)]
            head["window"] = {"X": X, "width": width, "rule": wrule, "form": "chi = 1 - S((r - X)/w + 1/2), S(s) = s^3 (10 - 15 s + 6 s^2) on [0, 1]",
                              "support_start": X - 0.5 * width, "support_end": X + 0.5 * width, "R_t": Rt, "r_in": rin,
                              "interior_local_minima": [{"R": float(R[i]), "dust_per_shell": float(nd[i])} for i in loc_min],
                              "interval_minimum": ({"R": float(R[glob]), "dust_per_shell": float(nd[glob])} if glob is not None else None),
                              "inside_annulus": (bool(float(rin) < X - 0.5 * width and X + 0.5 * width < float(Rt)) if (rin is not None and Rt is not None) else None),
                              "profile": prof, "dust_total": float(np.sum(nd)), "N_B": head["cinit"]["N_B"]}
            print("window: X %.4f (%s), w %g, the ramp over R %.2f to %.2f (r_in %s, R_t %s; inside the annulus: %s); interior local minima %s; the interval's minimum %s" % (
                X, wrule, width, X - 0.5 * width, X + 0.5 * width, fmt(rin, "%.4f"), fmt(Rt, "%.4f"), head["window"]["inside_annulus"],
                [(q["R"], "%.4e" % q["dust_per_shell"]) for q in head["window"]["interior_local_minima"]], head["window"]["interval_minimum"]), flush=True)
            N_run, v_run, E_run, fail_run = solve_total(eP, jP)
            head["solve_at_T0"] = {"E_total": E_run, "E_stored": rec0.get("E"), "failed": fail_run}
            labels = []

            def make_label(kap, th, full):
                """the label at T0 (version 1's, the roles exchanged by 33.4 (2)): the source C_+- - P_0 (+) P_0, the contrast with the reference
                rotated at T0; the collar charge differences solved (the unrotated one is the preparation's); full: the - preparation too, and
                the columns returned for the run."""
                ch = byk[kap]; s_ = st[kap]; ref = cache[kap][2]
                Phi = torch.cat([s_["sea"], s_["dust"]], dim=1)
                f = np.concatenate([np.ones(s_["n_sea"]), np.asarray(s_["f"], float)]); f2 = np.concatenate([f, f])
                c, s = cs_of(chi, th)
                e0, j0 = AUD.subtract(ch, Cpart[kap], ref, "S7: the comparator, the channel's state (C_0(t) - P(t)), twice", "source", [(ref, K_P), (s_["pbar"], K_PBAR)])
                D = doubled(ref)
                res = {}
                for sign in ((+1, -1) if full else (+1,)):
                    Y = twocopy_cols(Phi, c, s, sign); Pi = twocopy_cols(ref, c, s, sign)
                    known = [(D, K_PP), (Pi, K_ROT)]
                    C2 = cdens2(ch, Y, f2)
                    e_u, j_u = AUD.subtract(ch, C2, D, "S7 label at T0: C_+- - P_0 (+) P_0 (each preparation's reference)", "source", known, twocopy=True)
                    e_r, j_r = AUD.subtract(ch, C2, Pi, "S7 label's contrast at T0: C_+- - R (P_0 (+) P_0) R^dagger (the reference rotated at T0)", "contrast", known, twocopy=True)
                    if full:
                        res[sign] = (Y, Pi, e_u, j_u, e_r, j_r)
                    else:
                        res[sign] = (None, None, e_u, j_u, e_r, j_r)
                        del Y, Pi
                lab = {"kappa": kap, "theta": th, "full": full, "unrotated": src_stats(res[1][2], res[1][3], e0, j0, X, Cpart[kap]),
                       "rotated": src_stats(res[1][4], res[1][5], e0, j0, X, Cpart[kap])}
                if full:
                    lab["pm_eps"] = float(np.max(np.abs(res[1][2] - res[-1][2]))) / max(float(np.max(np.abs(2 * e0))), 1e-300)
                    lab["pm_J"] = float(np.max(np.abs(res[1][3] - res[-1][3]))) / max(float(np.max(np.abs(2 * j0))), 1e-300)
                Nu, vu, Eu, fu = solve_total(eP + res[1][2] - 2 * e0, jP + res[1][3] - 2 * j0)
                Nr, vr, Er, fr = solve_total(eP + res[1][4] - 2 * e0, jP + res[1][5] - 2 * j0)
                sv = {"E_plus": Eu, "E_total": E_run, "dE_over_E": (Eu - E_run) / abs(E_run), "dE_over_E_rotated": (Er - E_run) / abs(E_run),
                      "N_vs_total": float(np.max(np.abs(Nu - N_run))), "v_vs_total": float(np.max(np.abs(vu - v_run))),
                      "N_vs_total_rotated": float(np.max(np.abs(Nr - N_run))), "v_vs_total_rotated": float(np.max(np.abs(vr - v_run))), "failed": [fu, fr, fail_run]}
                if full:
                    Nm, vm, Em, fm = solve_total(eP + res[-1][2] - 2 * e0, jP + res[-1][3] - 2 * j0)
                    sv.update({"E_minus": Em, "dE_pm_over_E": (Eu - Em) / abs(E_run), "N_pm": float(np.max(np.abs(Nu - Nm))), "v_pm": float(np.max(np.abs(vu - vm)))})
                    sv["failed"].append(fm)
                lab["solved"] = sv
                lab["charge_ok"] = bool(abs(sv["dE_over_E"]) <= a.charge_tol)
                cols = {"Yp": res[1][0], "Ym": res[-1][0], "Pi_rot_p": res[1][1], "f": f, "f2": f2, "nc": int(Phi.shape[1])} if full else None
                return lab, cols

            # the quadratic law per kappa at the law's thetas
            for kap in kappas:
                rows_ = []
                for th in law_thetas:
                    lab, _ = make_label(kap, th, False)
                    lab["role"] = "law"; labels.append(lab); rows_.append(lab)
                    print("law kappa %d theta %.6f: the unrotated collar charge difference %+.4e of E (the rotated contrast %+.4e); source difference with P_0 (+) P_0 %.2e (eps), %.2e (J); rotated %.2e, %.2e" % (
                        kap, th, lab["solved"]["dE_over_E"], lab["solved"]["dE_over_E_rotated"], lab["unrotated"]["eps"], lab["unrotated"]["J"],
                        lab["rotated"]["eps"], lab["rotated"]["J"]), flush=True)
                th_ = [l_["theta"] for l_ in rows_]
                law[str(kap)] = {"unrotated": fit_law(th_, [l_["solved"]["dE_over_E"] for l_ in rows_]),
                                 "rotated": fit_law(th_, [l_["solved"]["dE_over_E_rotated"] for l_ in rows_])}
                lu = law[str(kap)]["unrotated"]
                law[str(kap)]["theta_at_target"] = theta_for(lu, a.charge_tol * a.law_margin)
                law[str(kap)]["theta_at_bar"] = theta_for(lu, a.charge_tol)
                print("law kappa %d (unrotated): dE/E = %.6e theta^2 %+.6e theta^4 (largest relative residual %.2e); pure quadratic %.6e (residual %.2e); log-log slopes %s; theta at %g of the bar %.6f, at the bar %.6f" % (
                    kap, lu["a"], lu["b"], lu["resid_ab"], lu["a_pure"], lu["resid_pure"], ", ".join(fmt(x_, "%.4f") for x_ in lu["loglog_slopes"]),
                    a.law_margin, law[str(kap)]["theta_at_target"], law[str(kap)]["theta_at_bar"]), flush=True)
            head["law"] = law
            if match:
                th_main = float(match["head"]["theta_main"]); th_rule = "--match-json %s (the primary's theta)" % os.path.basename(a.match_json)
            elif a.theta is not None:
                th_main = float(a.theta); th_rule = "--theta"
            else:
                per_k = {kap: law[str(kap)]["theta_at_target"] for kap in kappas}
                k_min = min(per_k, key=lambda q: per_k[q])
                th_main = per_k[k_min]
                th_rule = "the T0 law's root at %g of the bar %g (the builder's margin for the growth through the run), the smallest over kappa (kappa %d)" % (a.law_margin, a.charge_tol, k_min)
            head["theta_main"] = th_main; head["theta_rule"] = th_rule
            for kap in kappas:
                lab, cols = make_label(kap, th_main, True)
                lab["role"] = "run"; labels.append(lab)
                print("run label kappa %d theta %.6f (%s): the unrotated collar charge difference %+.4e of E (the bar %g: %s; the rotated contrast %+.4e); source difference %.2e (eps), %.2e (J), %d shells above 1e-15; + against - %.1e, solved %.1e" % (
                    kap, th_main, th_rule, lab["solved"]["dE_over_E"], a.charge_tol, "below" if lab["charge_ok"] else "ABOVE", lab["solved"]["dE_over_E_rotated"],
                    lab["unrotated"]["eps"], lab["unrotated"]["J"], lab["unrotated"]["shells_above_1e-15"], lab["pm_eps"], lab["solved"]["dE_pm_over_E"]), flush=True)
                cfgs.append({"kappa": kap, "theta": th_main, "X": X, "label": lab, "rule_done": False, "rule_trace": [], "Pi_rule": cols.pop("Pi_rot_p"), "rule_n": 0,
                             "charge": [], **cols})
            head["labels"] = labels
            head["thetas_final"] = {str(kap): [th_main] for kap in kappas}
            print("T/rs %.3f: %d preparations (kappa, theta): %s" % (T / rs, len(cfgs), ", ".join("(%d, %.6f)" % (cf["kappa"], cf["theta"]) for cf in cfgs)), flush=True)
        # each preparation's reference is P(t) (+) P(t) from T0 on (33.4 (2)); the reference rotated at T0, continued by the model's rule on the
        # two-copy block (version 1's route), is followed as a contrast (its distance to P(t) (+) P(t); it sources nothing)
        Dk = {}
        if k >= k0:
            for kap in kappas:
                Dk[kap] = doubled(cache[kap][2])
        if k > k0:
            for cf in cfgs:
                if cf["rule_done"]:
                    continue
                kap = cf["kappa"]
                e, U, ref, ci = cache[kap]
                new, mode, info = continue_twocopy(cf["Pi_rule"], e, U, 2 * st[kap]["n_sea"])
                cf["rule_n"] += 1
                if new is None:
                    row_ = {"T_over_rs": T / rs, "mode": "unresolved (%s)" % {k_: info.get(k_) for k_ in ("ambiguous_blocks", "n_occ", "n_sea")}, "distance_to_doubled": None}
                    cf["rule_done"] = True; cf["Pi_rule"] = None
                else:
                    dd = projdist(new, Dk[kap])
                    row_ = {"T_over_rs": T / rs, "mode": mode, "distance_to_doubled": dd, "min_occupied_weight": info.get("min_occupied_weight"),
                            "ambiguous_blocks": info.get("ambiguous_blocks"), "partly_occupied_blocks": info.get("partly_occupied_blocks")}
                    cf["Pi_rule"] = new
                    if dd < 1e-8 or cf["rule_n"] >= a.rule_check_max:
                        cf["rule_done"] = True; cf["Pi_rule"] = None
                cf["rule_trace"].append(row_)
                print("T/rs %.3f: kappa %d theta %.6f: the rotated reference continued by the model's rule (contrast): %s, distance to P (+) P %s; %s" % (
                    T / rs, kap, cf["theta"], row_["mode"], fmt(row_["distance_to_doubled"], "%.2e"),
                    ("coincides" if (row_["distance_to_doubled"] is not None and row_["distance_to_doubled"] < 1e-8) else ("followed no further" if cf["rule_done"] else "followed"))), flush=True)
        # per record from T0 on
        if k >= k0:
            Rt_k = Rt_of[k] if (k_close is None or k < k_close) else None        # the surface persists from T0 until the closing slice
            i_Rt = site_beyond(R, Rt_k) if Rt_k is not None else None
            xs_ = (T - T0) / (a.solve_every_rs * rs)
            do_solve = (k == k0) or (k == K - 1) or abs(xs_ - round(xs_)) < 1e-9
            if do_solve and k != k0:
                N_run, v_run, E_run, fail_run = solve_total(eP, jP)
            e1, U1, ref1, _ = cache[kappas[0]]
            ph1 = torch.exp(-1j * e1.to(torch.complex128) * tau)
            leak = float(((U1[ic] * ph1[None, :]) @ U1[iw].conj().T).abs().max())
            leak_sum += leak
            srow = {"k": k, "T_over_rs": T / rs, "R_t": Rt_k, "R_t_site": (float(R[i_Rt]) if (i_Rt is not None and i_Rt < n) else None), "persists": Rt_k is not None,
                    "R_t_stored": Rt_of[k], "leakage": leak, "leakage_sum": leak_sum, "solved_here": bool(do_solve), "E_total": (E_run if do_solve else None), "configs": []}
            for kap in kappas:
                s_ = st[kap]; ref = cache[kap][2]
                srow["P_minus_Pbar_F_k%d" % kap] = projdist(ref, s_["pbar"])
                srow["trP_k%d" % kap] = float((ref.abs() ** 2).sum()); srow["trPbar_k%d" % kap] = float((s_["pbar"].abs() ** 2).sum())
            for cf in cfgs:
                kap = cf["kappa"]; ch = byk[kap]; s_ = st[kap]; ref = cache[kap][2]
                Yp, Ym, nc = cf["Yp"], cf["Ym"], cf["nc"]
                n2 = 2 * n
                ft = torch.as_tensor(cf["f"], dtype=torch.complex128, device=dev)
                f2t = torch.as_tensor(cf["f2"], dtype=torch.complex128, device=dev)
                A = Yp[:n2, :nc]; B = Yp[:n2, nc:]
                G = (B * ft[None, :]) @ A.conj().T - (A * ft[None, :]) @ B.conj().T
                struct = max(float((Yp[n2:, :nc] + B).abs().max()), float((Yp[n2:, nc:] - A).abs().max()), float((Ym[:n2, :nc] - A).abs().max()),
                             float((Ym[:n2, nc:] + B).abs().max()), float((Ym[n2:, :nc] - B).abs().max()), float((Ym[n2:, nc:] - A).abs().max()))
                F = det[kap]["F"]; Wd = det[kap]["W"]
                Zp = torch.cat([F.conj().T @ Yp[:n2], F.conj().T @ Yp[n2:]], 0); Zm = torch.cat([F.conj().T @ Ym[:n2], F.conj().T @ Ym[n2:]], 0)
                DCD = (Zp * f2t[None, :]) @ Zp.conj().T - (Zm * f2t[None, :]) @ Zm.conj().T
                DCD = 0.5 * (DCD + DCD.conj().T)
                Ddet = float(torch.linalg.eigvalsh(DCD).abs().max())
                Ddet1 = 2.0 * float(torch.linalg.svdvals(F.conj().T @ G @ F).max())
                pk = (2.0 * ((Wd.conj() * (G @ Wd)).sum(dim=0)).abs()).cpu().numpy()
                Yc_p = torch.cat([Yp[ic], Yp[n2 + ic]], 0); Yc_m = torch.cat([Ym[ic], Ym[n2 + ic]], 0)
                DCc = (Yc_p * f2t[None, :]) @ Yc_p.conj().T - (Yc_m * f2t[None, :]) @ Yc_m.conj().T
                DCc = 0.5 * (DCc + DCc.conj().T)
                Dcol = float(torch.linalg.eigvalsh(DCc).abs().max())
                Wsp = (G.abs() ** 2).reshape(n, 2, n, 2).sum(dim=(1, 3))
                S2 = torch.flip(torch.cumsum(torch.cumsum(torch.flip(Wsp, [0, 1]), 0), 1), [0, 1])
                ub = (2.0 * torch.diagonal(S2).clamp_min(0.0).sqrt()).cpu().numpy()
                front = {}
                for ep_ in EPS_FRONT:
                    idx = np.nonzero(ub > ep_)[0]
                    front["%g" % ep_] = float(R[idx[-1]]) if len(idx) else 0.0
                # the exact region bounds (33.4 (2)): beyond R_t(t) while the surface persists, beyond R_t(T0) (the contrast) and the ladder, every record
                b_Rt = opnorm_region(G, i_Rt) if i_Rt is not None else None
                b_Rt0 = opnorm_region(G, i_Rt0) if i_Rt0 is not None else None
                opn = {str(r): opnorm_region(G, i_lad[r]) for r in ladder}
                fro_Rt = (float(ub[i_Rt]) if i_Rt < n else 0.0) if i_Rt is not None else None
                fro_Rt0 = (float(ub[i_Rt0]) if i_Rt0 < n else 0.0) if i_Rt0 is not None else None
                fro = {str(r): float(ub[i_lad[r]]) for r in ladder}
                # version 3 (36.3 (2)): the half-zone split beyond R_t(t) while the surface persists; the contrasts beyond R_t(T0) (reused when
                # its site is R_t(t)'s) and over the whole chain; G is read only, so every version 2 quantity above and below is unchanged
                t_hz = time.time()
                hz_Rt = halfzone(G, i_Rt, cross=True, unsplit=b_Rt) if i_Rt is not None else None
                if i_Rt0 is None:
                    hz_Rt0 = None
                elif i_Rt is not None and i_Rt == i_Rt0:
                    hz_Rt0 = dict(hz_Rt); hz_Rt0["unsplit"] = b_Rt0; hz_Rt0["same_as_Rt"] = True
                else:
                    hz_Rt0 = halfzone(G, i_Rt0, cross=False, unsplit=b_Rt0)
                b_all = opnorm_region(G, 0)
                hz_all = halfzone(G, 0, cross=False, unsplit=b_all)
                t_hz = time.time() - t_hz
                del G, Wsp, S2
                # the preparations' sources: each preparation's reference P(t) (+) P(t); contrasts: held rotated, P_bar (+) P_bar
                chi_c = window(R, cf["X"], width)
                c, s = cs_of(chi_c, cf["theta"])
                Pheld = twocopy_cols(ref, c, s, +1)
                known = [(Dk[kap], K_PP), (Pheld, K_HELD), (ref, K_P), (s_["pbar"], K_PBAR)]
                Cp = cdens2(ch, Yp, cf["f2"]); Cm = cdens2(ch, Ym, cf["f2"])
                ep_s, jp_s = AUD.subtract(ch, Cp, Dk[kap], "S7: the preparation C_+, copy-summed (C_+ - P(t) (+) P(t))", "source", known, twocopy=True)
                em_s, jm_s = AUD.subtract(ch, Cm, Dk[kap], "S7: the preparation C_-, copy-summed (C_- - P(t) (+) P(t))", "source", known, twocopy=True)
                e0, j0 = AUD.subtract(ch, Cpart[kap], ref, "S7: the comparator, the channel's state (C_0(t) - P(t)), twice", "source", known)
                eh_s, jh_s = AUD.subtract(ch, Cp, Pheld, "S7: the contrast C_+ - R (P(t) (+) P(t)) R^dagger (the reference held rotated)", "contrast", known, twocopy=True)
                eb_s, jb_s = AUD.subtract(ch, Cp, s_["pbar"], "S7: the contrast C_+ - P_bar (+) P_bar", "contrast", known, mult=2.0)
                del Pheld
                sp = src_stats(ep_s, jp_s, e0, j0, cf["X"], Cpart[kap]); sh = src_stats(eh_s, jh_s, e0, j0, cf["X"], Cpart[kap])
                crow = {"kappa": kap, "theta": cf["theta"], "X": cf["X"], "D_det": Ddet, "D_det_single_copy": Ddet1,
                        "D_det_rel_check": abs(Ddet - Ddet1) / max(Ddet, Ddet1, 1e-300), "copy_structure": struct, "per_packet": [float(x) for x in pk],
                        "collar_bound": Dcol, "bound_Rt": b_Rt, "bound_Rt_fro": fro_Rt, "bound_Rt0": b_Rt0, "bound_Rt0_fro": fro_Rt0,
                        "ladder_op": opn, "ladder_fro": fro, "front": front,
                        "src_pm_eps": float(np.max(np.abs(ep_s - em_s))) / sp["scale_eps"], "src_pm_J": float(np.max(np.abs(jp_s - jm_s))) / sp["scale_J"],
                        "src_own": sp, "src_held": sh, "contrast_Pbar_eps": float(np.max(np.abs(eb_s - ep_s))) / sp["scale_eps"],
                        "trC_plus": float(((Yp.abs() ** 2).sum(dim=0) * f2t.real).sum()), "trC_minus": float(((Ym.abs() ** 2).sum(dim=0) * f2t.real).sum()),
                        "two_trC0": 2.0 * float(np.sum(cf["f"])), "trPP": float((Dk[kap].abs() ** 2).sum()),
                        "halfzone_Rt": hz_Rt, "halfzone_Rt0": hz_Rt0, "halfzone_chain": hz_all, "bound_chain": b_all, "halfzone_seconds": t_hz}
                if do_solve:
                    if k == k0:
                        lsv = cf["label"]["solved"]
                        crow["charge"] = {"dE_over_E": lsv["dE_over_E"], "E_plus": lsv["E_plus"], "E_total": lsv["E_total"], "from": "the label at T0"}
                    else:
                        Nq, vq, Eq, fq = solve_total(eP + ep_s - 2 * e0, jP + jp_s - 2 * j0)
                        crow["charge"] = {"dE_over_E": (Eq - E_run) / abs(E_run), "E_plus": Eq, "E_total": E_run, "N_vs_total": float(np.max(np.abs(Nq - N_run))),
                                          "v_vs_total": float(np.max(np.abs(vq - v_run))), "failed": [fq, fail_run], "from": "solved at this record"}
                        if k == K - 1:
                            Nm, vm, Em, fm = solve_total(eP + em_s - 2 * e0, jP + jm_s - 2 * j0)
                            crow["solved"] = {"N_pm": float(np.max(np.abs(Nq - Nm))), "v_pm": float(np.max(np.abs(vq - vm))), "E_pm_rel": abs(Eq - Em) / abs(E_run),
                                              "N_vs_total": float(np.max(np.abs(Nq - N_run))), "v_vs_total": float(np.max(np.abs(vq - v_run))),
                                              "E_vs_total_rel": (Eq - E_run) / abs(E_run), "E_total": E_run, "failed": [fq, fm, fail_run]}
                    cf["charge"].append([T / rs, crow["charge"]["dE_over_E"]])
                srow["configs"].append(crow)
            srow["seconds"] = time.time() - tr
            s7_rows.append(srow)
            print("T/rs %.3f: R_t %s (beyond: R >= %s); bound beyond R_t(t) %s (Frobenius %s); beyond R_t(T0) %s; r 32 %s; front(1e-6) %s; front(1e-3) %s; D_det %s; collar %s; dE/E %s; check %.1e; leakage %.1e; replay eps %.1e J %.1e; %.1f s" % (
                T / rs, fmt(Rt_k, "%.4f"), fmt(srow["R_t_site"], "%.0f"), ", ".join(fmt(q["bound_Rt"]) for q in srow["configs"]),
                ", ".join(fmt(q["bound_Rt_fro"]) for q in srow["configs"]), ", ".join(fmt(q["bound_Rt0"]) for q in srow["configs"]),
                ", ".join(fmt(q["ladder_op"].get("32")) for q in srow["configs"]), ", ".join("%.0f" % q["front"]["1e-06"] for q in srow["configs"]),
                ", ".join("%.0f" % q["front"]["0.001"] for q in srow["configs"]), ", ".join("%.3e" % q["D_det"] for q in srow["configs"]),
                ", ".join("%.2e" % q["collar_bound"] for q in srow["configs"]), ", ".join(fmt(q.get("charge", {}).get("dE_over_E"), "%+.4e") for q in srow["configs"]),
                max(q["D_det_rel_check"] for q in srow["configs"]), leak, rr["CP_eps"], rr["CP_J"], srow["seconds"]), flush=True)
            print("T/rs %.3f: half-zones (36.3 (2)) beyond R_t(t): %s; beyond R_t(T0): %s; the whole chain: %s; split %.1f s" % (
                T / rs, " | ".join(hz_str(q["halfzone_Rt"], True) for q in srow["configs"]), " | ".join(hz_str(q["halfzone_Rt0"], False) for q in srow["configs"]),
                " | ".join(hz_str(q["halfzone_chain"], False) for q in srow["configs"]), sum(q["halfzone_seconds"] for q in srow["configs"])), flush=True)
        else:
            if k % 10 == 0 or k == k0 - 1:
                print("T/rs %.3f: replay eps %.2e J %.2e (C - P_bar against the stored %.2e, against the stored dust part %.2e); modes %s; %.1f s (eigh %.1f)" % (
                    T / rs, rr["CP_eps"], rr["CP_J"], rr["CPbar_eps"], rr["CPbar_vs_dust_eps"], modes, time.time() - tr, t_eig), flush=True)
        # the reference's state after the record (resume_state's order; version 1's)
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            if "occ_e" in ci:
                s_["levels_prev"] = (ci.pop("occ_e"), ci.pop("emp_e"), ci.get("partial_pairs", 0))
            if ci.get("gap_between_blocks") is not None or "weyl" in ci:
                s_["gap_between_prev"] = ci.get("gap_between_blocks")
        if a.out_npz and k == k0:
            states.update(snapshot("T0", cache))
            for i_, cf in enumerate(cfgs):
                if cf["Pi_rule"] is not None:                                  # at T0 the rule contrast still holds the rotated reference
                    states["T0_cfg%d_Pi_rot_plus" % i_] = cf["Pi_rule"].cpu().numpy()
        if a.out_npz and k == K - 1:
            states.update(snapshot("end", cache))
            for i_, cf in enumerate(cfgs):
                states["end_cfg%d_A" % i_] = cf["Yp"][:2 * n, :cf["nc"]].cpu().numpy(); states["end_cfg%d_B" % i_] = cf["Yp"][:2 * n, cf["nc"]:].cpu().numpy()
                states["end_cfg%d_f" % i_] = cf["f"]
                states["cfg%d_params" % i_] = np.array([cf["kappa"], cf["X"], cf["theta"], width])
        # the propagation by the record's step (version 1's)
        if k < K - 1:
            for ch in chs:
                e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
                ph = torch.exp(-1j * e.to(torch.complex128) * tau)
                prop = lambda X_, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X_))
                s_["sea"] = prop(s_["sea"]); s_["pbar"] = prop(s_["pbar"])
                if s_["dust"].shape[1]:
                    s_["dust"] = prop(s_["dust"])
            for cf in cfgs:
                e, U, ref, ci = cache[cf["kappa"]]
                ph = torch.exp(-1j * e.to(torch.complex128) * tau)
                prop = lambda X_, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X_))
                n2 = 2 * n
                for key in ("Yp", "Ym"):
                    Y = cf[key]
                    Y[:n2] = prop(Y[:n2]); Y[n2:] = prop(Y[n2:])
        f_ref = (N_app, v_app)
        del cache, Dk
        if k % 5 == 0 or k == K - 1:
            out["references"] = [{"kappa": cf["kappa"], "theta": cf["theta"], "rule_trace": cf["rule_trace"]} for cf in cfgs]
            out["charge"] = [{"kappa": cf["kappa"], "theta": cf["theta"], "series": cf["charge"]} for cf in cfgs]
            dump()
    out["references"] = [{"kappa": cf["kappa"], "theta": cf["theta"], "rule_trace": cf["rule_trace"]} for cf in cfgs]
    out["charge"] = [{"kappa": cf["kappa"], "theta": cf["theta"], "series": cf["charge"]} for cf in cfgs]
    out["crossing"] = crossing_clocks(out)
    dump()
    for cr in out["crossing"]:
        print("crossing kappa %d theta %.6f: closing at T/rs %s (%s r_s after T0; the last record holding the surface %s); %s" % (
            cr["kappa"], cr["theta"], fmt(cr["closing_T_over_rs"], "%.3f"), fmt(cr["closing_minus_T0_rs"], "%.3f"), fmt(cr["last_T_over_rs"], "%.3f"),
            "; ".join("eps %s: exact beyond R_t(t) %s, Frobenius %s, beyond R_t(T0) while held %s, over the run %s" % (
                ek, clock_str(cv["exact_Rt"]), clock_str(cv["fro_Rt"]), clock_str(cv["exact_Rt0_persisting"]), clock_str(cv["exact_Rt0_run"])) for ek, cv in cr["clocks"].items())), flush=True)
    if a.out_npz:
        np.savez(a.out_npz, **states)
    print("done: %d records in %.0f s" % (K, time.time() - t_start), flush=True)


# ------------------------------------------------------------------------------------------------------------------ compose
def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and x is not None and x == x else "-"


def clock_str(c):
    if c is None:
        return "none"
    return "T/rs %.3f (%.3f r_s after T0, %s r_s before the closing; bound %.3e; R_t %s)" % (
        c["T_over_rs"], c["after_T0_rs"], fmt(c["before_closing_rs"], "%.3f"), c["bound"], fmt(c["R_t"], "%.4f"))


def hz_str(h, full):
    """one configuration's half-zone split for the log (version 3)."""
    if h is None:
        return "-"
    s = "lo %s hi %s (unsplit %s" % (fmt(h["bound_lo"]), fmt(h["bound_hi"]), fmt(h.get("unsplit")))
    if full:
        s += ", cross %s" % fmt(h["bound_cross"])
    s += "; weight lo %s hi %s" % (fmt(h["share_lo"], "%.4f"), fmt(h["share_hi"], "%.4f"))
    if full:
        s += ", quarters %s, middle %s; E[cos ap] %s E[cos 2ap] %s; checks %s, %s; Parseval %s" % (
            ", ".join(fmt(x, "%.4f") for x in h["quarter_shares"]) if h["quarter_shares"] else "-", fmt(h["share_middle"], "%.4f"),
            fmt(h["cos_ap"], "%+.4f"), fmt(h["cos_2ap"], "%+.4f"), fmt(h.get("check_lower"), "%+.1e"), fmt(h.get("check_upper"), "%+.1e"), fmt(h["parseval"], "%.1e"))
    else:
        s += "; E[cos ap] %s" % fmt(h["cos_ap"], "%+.4f")
    return s + "; M %d, at pi/2 %d)" % (h["M"], h["boundary_momenta"])


def crossing_clocks(Mx):
    """33.4 (2)'s crossing clocks per configuration: the first record from T0 at which the exact bound beyond R_t(t) exceeds each of EPS_CROSS
    while the surface persists; the contrasts: the same from the Frobenius upper bound, and from the exact bound beyond R_t(T0) while the
    surface persists and over the whole run; the closing clock beside them."""
    h = Mx["head"]; s7 = Mx["s7"]; ep = h["episode"]; T0r = h["T0_over_rs"]
    res = []
    if not s7:
        return res
    for i_, c0 in enumerate(s7[0]["configs"]):
        row = {"kappa": c0["kappa"], "theta": c0["theta"], "closing_T_over_rs": ep["closing_T_over_rs"], "last_T_over_rs": ep["last_T_over_rs"],
               "closing_minus_T0_rs": ep["closing_minus_T0_rs"], "last_minus_T0_rs": ep["last_minus_T0_rs"], "clocks": {}}

        def first(key, persist_only, eps):
            for r in s7:
                if persist_only and not r["persists"]:
                    continue
                v = r["configs"][i_].get(key)
                if v is not None and v > eps:
                    return {"T_over_rs": r["T_over_rs"], "after_T0_rs": r["T_over_rs"] - T0r, "bound": v, "R_t": r["R_t"], "R_t_site": r["R_t_site"],
                            "before_closing_rs": (ep["closing_T_over_rs"] - r["T_over_rs"]) if ep["closing_T_over_rs"] is not None else None}
            return None
        for eps in EPS_CROSS:
            row["clocks"]["%g" % eps] = {"exact_Rt": first("bound_Rt", True, eps), "fro_Rt": first("bound_Rt_fro", True, eps),
                                        "exact_Rt0_persisting": first("bound_Rt0", True, eps), "exact_Rt0_run": first("bound_Rt0", False, eps)}
        pers = [(r["T_over_rs"], r["configs"][i_]["bound_Rt"]) for r in s7 if r["persists"] and r["configs"][i_]["bound_Rt"] is not None]
        row["max_bound_Rt_persisting"] = list(max(pers, key=lambda q: q[1])) if pers else None
        row["last_persisting"] = list(pers[-1]) if pers else None
        row["n_persisting_records"] = len(pers)
        res.append(row)
    return res


def charge_summary(Mx, bar):
    """the unrotated collar charge difference through the run per configuration: its largest |value| and where, against the bar; the law's
    coefficient through the run dE/(E theta^2); the law's theta for the bar over the run, theta (bar / max)^(1/2)."""
    res = []
    for c_ in Mx.get("charge", []):
        ser = c_["series"]
        if not ser:
            continue
        th = c_["theta"]
        mx = max(ser, key=lambda q: abs(q[1]))
        co = [q[1] / th ** 2 for q in ser]
        res.append({"kappa": c_["kappa"], "theta": th, "n_solved": len(ser), "at_T0": ser[0][1], "at_end": ser[-1][1], "max_abs": abs(mx[1]), "max_at_T_over_rs": mx[0],
                    "below_bar": bool(abs(mx[1]) <= bar), "coef_min": min(co), "coef_max": max(co), "theta_for_bar": th * math.sqrt(bar / abs(mx[1])) if mx[1] != 0 else None,
                    "min_value": min(q[1] for q in ser), "max_value": max(q[1] for q in ser)})
    return res


def halfzone_md(Ms):
    """version 3's md: the half-zone split beyond R_t(t) per record from T0 to the closing (36.3 (2)) beside the unsplit bound, a summary per
    configuration, and the contrasts' table (beyond R_t(T0), the whole chain)."""
    L = ["", "### The half-zone split beyond R_t(t) (36.3 (2)): the exact bound after projecting the difference onto the half-zones |ap| < pi/2 and |ap| >= pi/2 separately, beside the unsplit bound, per record from T0 to the closing; a reading, not a criterion", "",
         "- the projector: the chain's discrete Fourier transform on the M sites beyond R_t(t) (R = index + 1), per spinor component: G_hat = F G_D F^dagger, F[q, j] = exp(-2 pi i q j/M)/sqrt(M), a p = 2 pi q_s/M in (-pi, pi]; Pi_lo keeps |a p| < pi/2 and Pi_hi |a p| >= pi/2 (the momenta at |a p| = pi/2 exactly, present when 4 divides M, join the upper half-zone; their count in the column 'at pi/2'). Each bound is 2 ||Pi G_D Pi||_op, twice the largest |eigenvalue| of i G_hat on the half-zone's block: the largest ||Delta C_D||_op of any detector beyond the surface whose modes lie in that half-zone (Delta C_D = 2 G_D between the copies). The chain is the sine chain (h(p) = m sz + sin(a p) sx + (kappa/R) sy; the two-step Q gives sin(2 a p)/2), so |a p| = pi/2 separates the valley at p = 0 from the doubler's valley at a p = pi.",
         "- beside it (contrasts, sourcing nothing): the cross bound 2 ||Pi_lo G_D Pi_hi||_op; the one-sided momentum weight w(q) = sum |G_hat[q, .]|^2 as shares of ||G_D||_F^2 in the half-zones and in the quarter-zones [0, pi/4), [pi/4, pi/2), [pi/2, 3 pi/4], (3 pi/4, pi] (the middle two are vf10's cos 2ap < 0 bands); E[cos a p] = Re sum_j <g_(j+1), g_j>/||G_D||_F^2 and E[cos 2 a p] from the site overlaps without wrap-around (E[cos a p] is +1 on the p = 0 valley and -1 on the doubler's; E[cos 2 a p] is +1 on both valleys and -1 at |a p| = pi/2); the compression check max(lo, hi, cross) - unsplit and unsplit - (max(lo, hi) + cross), each at most 0 to rounding; Parseval's identity |sum w - ||G_D||_F^2| / ||G_D||_F^2.", ""]
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            L += ["- %s (%s, tau %g), kappa %d, theta %.6f (the closing clock T/rs %s):" % (role, h["tag"], h["tau"], c0["kappa"], c0["theta"], fmt(h["episode"]["closing_T_over_rs"], "%.3f")), "",
                  "| T/rs | R_t(t) | beyond: R >= | sites M (at pi/2) | unsplit bound (version 2's) | |ap| < pi/2 | |ap| >= pi/2 | upper / lower | cross | weight: lower; upper | weight by quarter-zone | E[cos ap]; E[cos 2ap] | checks (lower; upper) | Parseval |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            closed = False
            for r in s7:
                q = r["configs"][i_]; hz = q.get("halfzone_Rt")
                if hz is None:
                    if not closed and not r["persists"]:
                        L.append("| %.3f | - | - | - | - | - | - | - | - | - | - | - | - | - |" % r["T_over_rs"])
                        closed = True
                    continue
                ratio = (hz["bound_hi"] / hz["bound_lo"]) if hz["bound_lo"] > 0 else None
                L.append("| %.3f | %s | %s | %d (%d) | %s | %s | %s | %s | %s | %s; %s | %s | %s; %s | %s; %s | %s |" % (
                    r["T_over_rs"], fmt(r["R_t"], "%.4f"), fmt(r["R_t_site"], "%.0f"), hz["M"], hz["boundary_momenta"], fmt(q["bound_Rt"]), fmt(hz["bound_lo"]),
                    fmt(hz["bound_hi"]), fmt(ratio, "%.4f"), fmt(hz["bound_cross"]), fmt(hz["share_lo"], "%.4f"), fmt(hz["share_hi"], "%.4f"),
                    ", ".join(fmt(x, "%.4f") for x in hz["quarter_shares"]) if hz["quarter_shares"] else "-",
                    fmt(hz["cos_ap"], "%+.4f"), fmt(hz["cos_2ap"], "%+.4f"), fmt(hz.get("check_lower"), "%+.1e"), fmt(hz.get("check_upper"), "%+.1e"), fmt(hz["parseval"], "%.1e")))
            rows = [(r["T_over_rs"], r["configs"][i_]["halfzone_Rt"]) for r in s7
                    if r["configs"][i_].get("halfzone_Rt") is not None and r["configs"][i_]["halfzone_Rt"]["fro2"] > 0]
            if rows:
                rat = [x for x in ((z["bound_hi"] / z["bound_lo"]) if z["bound_lo"] > 0 else None for _, z in rows) if x is not None]
                up = [t for t, z in rows if z["bound_hi"] > z["bound_lo"]]
                L += ["", "  Over the %d records after T0 holding the surface with a nonzero difference beyond it (T/rs %.3f to %.3f): upper / lower %s to %s; the upper half-zone's weight share %s to %s; the middle bands' (vf10's cos 2ap < 0) %s to %s; E[cos ap] %s to %s; E[cos 2ap] %s to %s; the upper half-zone's bound above the lower's at %s; the compression check at most %s (lower) and %s (upper); Parseval at most %s." % (
                    len(rows), rows[0][0], rows[-1][0], fmt(min(rat) if rat else None, "%.4f"), fmt(max(rat) if rat else None, "%.4f"),
                    fmt(min(z["share_hi"] for _, z in rows), "%.4f"), fmt(max(z["share_hi"] for _, z in rows), "%.4f"),
                    fmt(min(z["share_middle"] for _, z in rows), "%.4f"), fmt(max(z["share_middle"] for _, z in rows), "%.4f"),
                    fmt(min(z["cos_ap"] for _, z in rows), "%+.4f"), fmt(max(z["cos_ap"] for _, z in rows), "%+.4f"),
                    fmt(min(z["cos_2ap"] for _, z in rows), "%+.4f"), fmt(max(z["cos_2ap"] for _, z in rows), "%+.4f"),
                    ("T/rs " + ", ".join("%.3f" % t for t in up)) if up else "no record",
                    fmt(max(z["check_lower"] for _, z in rows), "%+.1e"), fmt(max(z["check_upper"] for _, z in rows), "%+.1e"), fmt(max(z["parseval"] for _, z in rows), "%.1e"))]
            L.append("")
    L += ["### The half-zone split's contrasts: beyond R_t(T0) (version 2's fixed site; every record, the closing slice included) and over the whole chain (the preparation's own momentum content: the 4-site ramp's seeding at T0 and its evolution)", ""]
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            L += ["- %s (%s, tau %g), kappa %d, theta %.6f; beyond R_t(T0): R >= %s:" % (role, h["tag"], h["tau"], c0["kappa"], c0["theta"], fmt(h["R_t0_site"], "%.0f")), "",
                  "| T/rs | surface | beyond R_t(T0): unsplit (version 2's) | |ap| < pi/2 | |ap| >= pi/2 | weight: lower; upper | E[cos ap] | whole chain: 2 ||G||_op | |ap| < pi/2 | |ap| >= pi/2 | weight: lower; upper | weight by quarter-zone | E[cos ap]; E[cos 2ap] | sites (at pi/2) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in s7:
                q = r["configs"][i_]; z0 = q.get("halfzone_Rt0"); za = q.get("halfzone_chain")
                g0 = lambda key, f="%.3e": fmt(z0.get(key) if z0 else None, f)
                ga = lambda key, f="%.3e": fmt(za.get(key) if za else None, f)
                L.append("| %.3f | %s | %s | %s | %s | %s; %s | %s | %s | %s | %s | %s; %s | %s | %s; %s | %s |" % (
                    r["T_over_rs"], "held" if r["persists"] else "closed", fmt(q["bound_Rt0"]), g0("bound_lo"), g0("bound_hi"), g0("share_lo", "%.4f"), g0("share_hi", "%.4f"),
                    g0("cos_ap", "%+.4f"), fmt(q.get("bound_chain")), ga("bound_lo"), ga("bound_hi"), ga("share_lo", "%.4f"), ga("share_hi", "%.4f"),
                    ", ".join(fmt(x, "%.4f") for x in za["quarter_shares"]) if (za and za["quarter_shares"]) else "-",
                    ga("cos_ap", "%+.4f"), ga("cos_2ap", "%+.4f"), ("%d (%d)" % (za["M"], za["boundary_momenta"])) if za else "-"))
            L.append("")
    return L


def leaves(x, path=""):
    """a json object's leaves by path."""
    out = {}
    if isinstance(x, dict):
        for k_, v_ in x.items():
            out.update(leaves(v_, "%s/%s" % (path, k_)))
    elif isinstance(x, list):
        for i_, v_ in enumerate(x):
            out.update(leaves(v_, "%s[%d]" % (path, i_)))
    else:
        out[path] = x
    return out


def bits_equal(va, vb):
    """floats by their bits (the json's round-trip doubles), other leaves by type and value."""
    if isinstance(va, float) and isinstance(vb, float):
        return struct.pack("<d", va) == struct.pack("<d", vb)
    return type(va) == type(vb) and va == vb


GATE_SKIP = ("seconds", "eig_s", "records", "T_end_over_rs")


def compare_v2(o2, o3, skip=GATE_SKIP):
    """every leaf of version 2's object against version 3's at the same path: numbers, booleans and None bitwise; strings literally (listed
    when they differ); keys version 3 adds are not compared; the timing keys and the run length (records, T_end_over_rs) skipped."""
    A = leaves(o2); B = leaves(o3)
    res = {"numeric": 0, "equal": 0, "worst_rel": 0.0, "worst_path": None, "unequal_paths": [], "missing_in_v3": [], "strings": 0, "string_diffs": []}
    for p_, va in A.items():
        if set(re.split(r"[/\[\]]", p_)) & set(skip):
            continue
        if p_ not in B:
            res["missing_in_v3"].append(p_)
            continue
        vb = B[p_]
        if isinstance(va, str) or isinstance(vb, str):
            res["strings"] += 1
            if va != vb:
                res["string_diffs"].append(p_)
            continue
        res["numeric"] += 1
        if bits_equal(va, vb):
            res["equal"] += 1
            continue
        res["unequal_paths"].append(p_)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
            d = abs(float(va) - float(vb)) / max(abs(float(va)), 1e-300)
        else:
            d = float("inf")
        if d > res["worst_rel"] or res["worst_path"] is None:
            res["worst_rel"] = d; res["worst_path"] = p_
    return res


def gate_all(Ms, paths):
    """the gate against version 2 (the brief): per member, version 2's member json against version 3's on the records both compute."""
    out = []
    for (Mx, role), p2 in zip(Ms, paths):
        M2 = json.load(open(p2))
        g = {"member": Mx["head"]["tag"], "role": role, "v2_json": p2, "v2_records": M2["head"]["records"], "v3_records": Mx["head"]["records"],
             "v2_script": M2["head"].get("script")}
        g["head"] = compare_v2(M2["head"], Mx["head"])
        rp3 = {r["k"]: r for r in Mx["replay"]}
        c2 = [r for r in M2["replay"] if r["k"] in rp3]
        g["replay"] = compare_v2(c2, [rp3[r["k"]] for r in c2]); g["replay_n"] = len(c2); g["replay_k"] = [c2[0]["k"], c2[-1]["k"]] if c2 else None
        s3 = {r["k"]: r for r in Mx["s7"]}
        c7 = [r for r in M2["s7"] if r["k"] in s3]
        g["s7"] = compare_v2(c7, [s3[r["k"]] for r in c7]); g["s7_n"] = len(c7); g["s7_T"] = [c7[0]["T_over_rs"], c7[-1]["T_over_rs"]] if c7 else None
        nb = ne = nnone = 0; worst = 0.0
        for r2 in c7:
            r3 = s3[r2["k"]]
            for q2, q3 in zip(r2["configs"], r3["configs"]):
                b2, b3 = q2.get("bound_Rt"), q3.get("bound_Rt")
                if b2 is None and b3 is None:
                    nnone += 1
                    continue
                nb += 1
                if b2 is not None and b3 is not None and bits_equal(float(b2), float(b3)):
                    ne += 1
                elif b2 is not None and b3 is not None:
                    worst = max(worst, abs(b2 - b3) / max(abs(b2), 1e-300))
                else:
                    worst = float("inf")
        g["gate"] = {"n": nb, "equal": ne, "no_surface_in_both": nnone, "worst_rel": worst, "pass": bool(nb > 0 and ne == nb)}
        ch2 = {(c_["kappa"], c_["theta"]): c_["series"] for c_ in M2.get("charge", [])}
        pairs = [(ch2.get((c_["kappa"], c_["theta"]), [])[:len(c_["series"])], c_["series"]) for c_ in Mx.get("charge", [])]
        g["charge"] = compare_v2([x for x, _ in pairs], [y for _, y in pairs])
        g["crossing"] = compare_v2(M2.get("crossing", []), Mx.get("crossing", []))
        g["references"] = compare_v2(M2.get("references", []), Mx.get("references", []))
        g["audit_pass"] = [M2["audit"]["pass"], Mx["audit"]["pass"]]
        g["all_numeric_equal"] = all(g[k_]["numeric"] == g[k_]["equal"] for k_ in ("head", "replay", "s7", "charge", "crossing", "references"))
        out.append(g)
    return out


def gate_md(gates):
    if not gates:
        return ["", "### The gate against version 2: not run (no --gate-v2 given)", ""]
    L = ["", "### The gate against version 2 on the records both compute (the brief: the unsplit bound beyond R_t(t) bitwise; every numeric field version 2 stored compared beside it by the doubles' bits)", "",
         "| member | version 2's json | records both compute: replay (k); S7 (T/rs) | the unsplit bound beyond R_t(t): bitwise equal | largest relative difference | head (T0, window, law, labels, theta, episode, C-init, detector) | replay rows | S7 rows (every field version 2 stored) | charge series | crossing clocks | rule traces | audit pass (v2; v3) | the gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(c):
        return "%d of %d equal%s" % (c["equal"], c["numeric"], ("; worst %.2e at %s" % (c["worst_rel"], c["worst_path"])) if c["equal"] < c["numeric"] else "")
    for g in gates:
        L.append("| %s (%s) | %s (%s records) | %d (k %s); %d (T/rs %s) | %d of %d record-configurations (%d without a surface in both) | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s |" % (
            g["member"], g["role"], os.path.basename(g["v2_json"]), g["v2_records"], g["replay_n"], ("%d to %d" % tuple(g["replay_k"])) if g["replay_k"] else "-", g["s7_n"],
            ("%.3f to %.3f" % tuple(g["s7_T"])) if g["s7_T"] else "-", g["gate"]["equal"], g["gate"]["n"], g["gate"]["no_surface_in_both"],
            ("%.2e" % g["gate"]["worst_rel"]) if g["gate"]["equal"] < g["gate"]["n"] else "0 (bitwise)",
            cell(g["head"]), cell(g["replay"]), cell(g["s7"]), cell(g["charge"]), cell(g["crossing"]), cell(g["references"]),
            g["audit_pass"][0], g["audit_pass"][1], "PASS" if g["gate"]["pass"] else "FAIL"))
    L.append("")
    for g in gates:
        L.append("- %s: the gate %s (the unsplit bound beyond R_t(t) bitwise equal on %d of %d record-configurations from T/rs %s); every numeric field version 2 stored equal on the common records: %s." % (
            g["member"], "PASS" if g["gate"]["pass"] else "FAIL", g["gate"]["equal"], g["gate"]["n"], ("%.3f to %.3f" % tuple(g["s7_T"])) if g["s7_T"] else "-",
            "yes" if g["all_numeric_equal"] else "no"))
        for part in ("head", "replay", "s7", "charge", "crossing", "references"):
            c = g[part]
            if c["unequal_paths"]:
                L.append("  %s: %d unequal numeric leaves, the first %s." % (part, len(c["unequal_paths"]), ", ".join(c["unequal_paths"][:8])))
            if c["missing_in_v3"]:
                L.append("  %s: %d leaves of version 2 absent in version 3, the first %s." % (part, len(c["missing_in_v3"]), ", ".join(c["missing_in_v3"][:8])))
            if c["string_diffs"]:
                L.append("  %s: strings differing (%d of %d): %s." % (part, len(c["string_diffs"]), c["strings"], ", ".join(c["string_diffs"][:8])))
    L.append("")
    return L


def compose(a):
    paths = a.compose.split(",")
    M = [json.load(open(p)) for p in paths]
    P_ = M[0]; Cn = M[1] if len(M) > 1 else None
    hP = P_["head"]
    Ms = [(P_, "primary")] + ([(Cn, "step control")] if Cn is not None else [])
    for Mx, role in Ms:
        if "crossing" not in Mx:
            Mx["crossing"] = crossing_clocks(Mx)
    L = ["## S7 part 3, the carrying branch: two-copy preparations inside the trapping episode, propagated on the stored history to T/rs %s; the region bound beyond the trapping surface split into the half-zones |ap| < pi/2 and >= pi/2, beside the unsplit bound, and the gate against version 2 (engine_s7_twocopy_v3.py; engine spec 36.3 (2), with 33.4 (2), 30.5, 29.5 and 28.5 S7)" % (
        ", ".join("%.3f" % Mx["head"]["T_end_over_rs"] for Mx, role in Ms)), ""]
    for Mx, role in Ms:
        h = Mx["head"]
        L.append("- %s: %s (%s; r_s %g, tau %g, %d sites, m %g, g %g, ell %s, %d channels), the preparation record T/rs %.3f (%s; trapping zeros there: r_in %s, R_t %s, R_t - r_in %s; the stored E %s, the replayed total's solve %s), %d records to T/rs %.3f; the C-init re-prepared in %s passes (stored %s): max |N - N_cinit| %s, |v - v_cinit| %s, E0 %s (stored %s); %.0f s in all." % (
            role, h["tag"], h["member_script"], h["rs"], h["tau"], h["nsites"], h["m"], h["g"], h["ell"], h["kappa_sea"], h["T0_over_rs"], h["T0_rule"],
            fmt(h["trapping_zeros_at_T0"]["r_in"], "%.4f"), fmt(h["trapping_zeros_at_T0"]["R_t"], "%.4f"),
            fmt((h["trapping_zeros_at_T0"]["R_t"] - h["trapping_zeros_at_T0"]["r_in"]) if (h["trapping_zeros_at_T0"]["R_t"] is not None and h["trapping_zeros_at_T0"]["r_in"] is not None) else None, "%.4f"),
            fmt(h["E_stored_at_T0"], "%.10f"), fmt(h["solve_at_T0"]["E_total"], "%.10f"),
            h["records"], h["T_end_over_rs"], h["cinit"]["passes"], h["cinit"]["stored_passes"], fmt(h["cinit"]["dN"], "%.2e"), fmt(h["cinit"]["dv"], "%.2e"),
            fmt(h["cinit"]["E0"], "%.10f"), fmt(h["cinit"]["stored_E0"], "%.10f"), Mx.get("seconds", float("nan"))))
    L += ["- the ell control: no r_s 32 ell 8 member on this lane's v17 set exists (C3set25's c25_rs32_A_R2_tau8_ell8_avg is an engine_collapse_v25 run stopped at T/rs 3.75, before T0; version 1's finding); none runs.", ""]
    # T0
    L += ["### T0 (33.4 (2)): the stored records of the primary in T/rs [%g, %g] holding both trapping zeros" % tuple(hP["T0_range"]), "",
          "| T/rs | r_in | R_t | R_t - r_in | T0 |", "|---|---|---|---|---|"]
    for c_ in hP["T0_candidates"]:
        L.append("| %.3f | %.4f | %.4f | %.4f | %s |" % (c_["T_over_rs"], c_["r_in"], c_["R_t"], c_["R_t_minus_r_in"], "yes" if abs(c_["T_over_rs"] - hP["T0_over_rs"]) < 1e-9 else ""))
    # the surface and the closing clock
    L += ["", "### The trapping surface from T0 to the closing clock (the stored records' R_t; the surface persists while R_t is held)", ""]
    for Mx, role in Ms:
        h = Mx["head"]; ep = h["episode"]
        L.append("- %s (%s, tau %g): R_t held from T/rs %s to %s about T0 (the run of records holding it); the closing clock, the first slice after T0 without R_t, T/rs %s: %s r_s after T0 (the last record holding the surface %s r_s after T0); reappearances after the closing: %s; slices without a stored record: %s; R_t(T0) %s, the first site beyond it R %s." % (
            role, h["tag"], h["tau"], fmt(ep["first_T_over_rs"], "%.3f"), fmt(ep["last_T_over_rs"], "%.3f"), fmt(ep["closing_T_over_rs"], "%.3f"),
            fmt(ep["closing_minus_T0_rs"], "%.3f"), fmt(ep["last_minus_T0_rs"], "%.3f"), ep["reappearances_T_over_rs"] or "none", ep["slices_without_record"] or "none",
            fmt(h["trapping_zeros_at_T0"]["R_t"], "%.4f"), fmt(h["R_t0_site"], "%.0f")))
    L.append("")
    trk = {round(r_["T_over_rs"], 6): r_ for r_ in (Cn["head"]["episode"]["track"] if Cn else [])}
    L += ["| T/rs | %s r_in | %s R_t | %s |" % (hP["tag"], hP["tag"], ("%s r_in | %s R_t" % (Cn["head"]["tag"], Cn["head"]["tag"])) if Cn else "-"), "|---|---|---|---|%s" % ("---|" if Cn else "")]
    tset = sorted(set([round(r_["T_over_rs"], 6) for r_ in hP["episode"]["track"]] + list(trk.keys())))
    trP = {round(r_["T_over_rs"], 6): r_ for r_ in hP["episode"]["track"]}
    for t_ in tset:
        p_ = trP.get(t_, {}); q_ = trk.get(t_, {})
        L.append("| %.3f | %s | %s | %s |%s" % (t_, fmt(p_.get("r_in"), "%.4f") if p_ else "", fmt(p_.get("R_t"), "%.4f") if p_ else "",
                                                ("%s | %s" % (fmt(q_.get("r_in"), "%.4f") if q_ else "", fmt(q_.get("R_t"), "%.4f") if q_ else "")) if Cn else "-", ""))
    # the window
    wP = hP["window"]
    L += ["", "### The window (33.4 (2)) and the dust shell density at T0 (%s, T/rs %.3f)" % (hP["tag"], hP["T0_over_rs"]), "",
          "- %s, w %g sites, X %.4f (%s); the ramp over R %.2f to %.2f (chi = 1 below, 0 above), r_in %s, R_t %s: inside the annulus %s; the dust shell density's interior local minima in [r_in, R_t]: %s; its minimum over the interval's sites: %s; the dust's total occupation %.6f (N_B %s)." % (
              wP["form"], wP["width"], wP["X"], wP["rule"], wP["support_start"], wP["support_end"], fmt(wP["r_in"], "%.4f"), fmt(wP["R_t"], "%.4f"), wP["inside_annulus"],
              ", ".join("R %.0f (%.6e)" % (q["R"], q["dust_per_shell"]) for q in wP["interior_local_minima"]) or "none",
              ("R %.0f (%.6e)" % (wP["interval_minimum"]["R"], wP["interval_minimum"]["dust_per_shell"])) if wP["interval_minimum"] else "-", wP["dust_total"], fmt(wP["N_B"], "%.6f"))]
    if Cn is not None:
        wC = Cn["head"]["window"]
        L.append("- the step control: X %.4f (%s); its own trapping zeros at T0 r_in %s, R_t %s: the ramp inside its annulus %s; its interior local minima %s." % (
            wC["X"], wC["rule"], fmt(wC["r_in"], "%.4f"), fmt(wC["R_t"], "%.4f"), wC["inside_annulus"],
            ", ".join("R %.0f" % q["R"] for q in wC["interior_local_minima"]) or "none"))
    L += ["", "| R | dust per shell (sum_kappa w f |phi|^2) | dust energy per shell (replayed) | stored dust energy per shell (eps_tot - eps_pairs) | %s | chi_X |" % " | ".join("kappa %d's dust (unit weight)" % kq for kq in hP["kappas"]),
          "|---|---|---|---|%s---|" % ("---|" * len(hP["kappas"]))]
    for p_ in wP["profile"]:
        L.append("| %.0f | %.6e | %.6e | %.6e | %s | %.6f |" % (p_["R"], p_["dust_per_shell"], p_["dust_energy_per_shell"], p_["stored_dust_energy"],
                 " | ".join(fmt(p_.get("dust_k%d" % kq), "%.4e") for kq in hP["kappas"]), p_["chi"]))
    # the law and theta
    L += ["", "### Theta from the quadratic law (33.4 (2)): the unrotated collar charge difference at T0 per kappa (the member's total sources with two copies of the channel replaced by the + preparation's, solved by the run's own solver, against the unreplaced total's solve)", "",
          "| member | kappa | theta | dE/E with P_0 (+) P_0 (the preparation's reference) | dE/E with R (P_0 (+) P_0) R^dagger (the rotated contrast) | dE/(E theta^2) | source difference with P_0 (+) P_0 (eps; J) |", "|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        h = Mx["head"]
        for lab in h["labels"]:
            L.append("| %s | %d | %.6f%s | %+.6e | %+.6e | %.6e | %.3e; %.3e |" % (h["tag"], lab["kappa"], lab["theta"], " (the run's)" if lab.get("role") == "run" else "",
                     lab["solved"]["dE_over_E"], lab["solved"]["dE_over_E_rotated"], lab["solved"]["dE_over_E"] / lab["theta"] ** 2, lab["unrotated"]["eps"], lab["unrotated"]["J"]))
    L.append("")
    for Mx, role in Ms:
        h = Mx["head"]
        for kap in h["kappas"]:
            lw = h["law"][str(kap)]; lu = lw["unrotated"]; lr = lw["rotated"]
            L.append("- %s, kappa %d, the law of the unrotated difference over theta %s: dE/E = %.6e theta^2 %+.6e theta^4 (largest relative residual %.2e); the pure quadratic %.6e theta^2 (largest relative residual %.2e); the log-log slopes between neighbouring thetas %s; the root at %g of the bar %.6f, at the bar %.6f. The rotated contrast: %.6e theta^2 %+.6e theta^4 (residual %.2e), slopes %s." % (
                h["tag"], kap, ", ".join("%.6f" % t for t in lu["thetas"]), lu["a"], lu["b"], lu["resid_ab"], lu["a_pure"], lu["resid_pure"],
                ", ".join(fmt(x_, "%.4f") for x_ in lu["loglog_slopes"]), h["law_margin"], lw["theta_at_target"], lw["theta_at_bar"],
                lr["a"], lr["b"], lr["resid_ab"], ", ".join(fmt(x_, "%.4f") for x_ in lr["loglog_slopes"])))
        L.append("- %s: the run's theta %.6f (%s)." % (h["tag"], h["theta_main"], h["theta_rule"]))
    # the run's label
    L += ["", "### The preparations' label at T0 (the run's theta): the remaining normal-ordered source difference and the collar charge difference", "",
          "| member | kappa | theta | source difference with P_0 (+) P_0 (the preparation's reference): largest (eps; J) relative to max |2 (C_0 - P_0)| | largest absolute (eps; J) | net sum (eps; J) | shells above 1e-15 of the raw density's largest value (first, last) | away from the transition (> w) | with the rotated R (P_0 (+) P_0) R^dagger, the contrast (eps; J) | + against - (eps; J) | collar charge difference dE/E (the preparation's; the rotated contrast) | solved max |dN|, |dv| against the unreplaced total | + against -: |dN|, |dv|, dE/E |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        h = Mx["head"]
        for lab in h["labels"]:
            if lab.get("role") != "run":
                continue
            un, ro, sv = lab["unrotated"], lab["rotated"], lab["solved"]
            sh = ("%d (%s)" % (un["shells_above_1e-15"], ", ".join("%.0f" % x for x in un["shells_first_last"]))) if un["shells_first_last"] else "0"
            L.append("| %s | %d | %.6f | %.3e; %.3e | %.3e; %.3e | %+.3e; %+.3e | %s | %.1e; %.1e | %.3e; %.3e | %.1e; %.1e | %+.4e; %+.4e | %.2e, %.2e | %.1e, %.1e, %.1e |" % (
                h["tag"], lab["kappa"], lab["theta"], un["eps"], un["J"], un["eps_abs"], un["J_abs"], un["net_eps"], un["net_J"], sh, un["far_eps"], un["far_J"],
                ro["eps"], ro["J"], lab["pm_eps"], lab["pm_J"], sv["dE_over_E"], sv["dE_over_E_rotated"], sv["N_vs_total"], sv["v_vs_total"], sv["N_pm"], sv["v_pm"], sv["dE_pm_over_E"]))
    # the rotated reference under the model's rule (contrast)
    L += ["", "### The reference rotated at T0, continued by the model's rule on the two-copy block (a contrast; each preparation's reference is P(t) (+) P(t) from T0)", ""]
    for Mx, role in Ms:
        for rf in Mx.get("references", []):
            tr_ = rf["rule_trace"]
            L.append("- %s, kappa %d, theta %.6f: %s." % (Mx["head"]["tag"], rf["kappa"], rf["theta"],
                     "; ".join("T/rs %.3f %s, distance to P(t) (+) P(t) %s" % (q["T_over_rs"], q["mode"], fmt(q["distance_to_doubled"], "%.2e")) for q in tr_) or "not followed"))
    # the audit
    L += ["", "### The audit: every source the script computes, and the reference it subtracted", "",
          "| member | source computed | role | reference received | calls | verdict |", "|---|---|---|---|---|---|"]
    ok_all = True
    for Mx, role in Ms:
        for r in Mx["audit"]["rows"]:
            L.append("| %s | %s | %s | %s | %d | %s |" % (Mx["head"]["tag"], r["label"], r["role"], r["reference"], r["calls"], r["verdict"]))
        ok_all = ok_all and Mx["audit"]["pass"]
    L.append("")
    L.append("- the audit's verdict: %s (every call with the role \"source\" subtracted the record's continued reference P, or for a preparation P(t) (+) P(t); the contrasts, which source nothing, subtracted P_bar, the reference rotated at T0 or the reference held rotated)." % ("PASS" if ok_all else "FAIL"))
    for Mx, role in Ms:
        rp = Mx["replay"]
        nacc = sum(1 for r in rp if r["accepted"])
        mx = lambda key: max(r[key] for r in rp)
        mn = lambda key: min(r[key] for r in rp)
        L.append("- %s, the numeric audit over its %d records (T/rs %.3f to %.3f): the replayed C - P against the stored eps_tot, J_tot: at most %.2e, %.2e relative (acceptance %g; %d of %d records within it); the contrast C - P_bar against the stored totals: %.2e to %.2e (eps), %.2e to %.2e (J), and against the stored dust part eps_tot - eps_pairs at most %.2e, %.2e; the propagated P_bar's columns against the sea's: at most %.1e." % (
            Mx["head"]["tag"], len(rp), rp[0]["T_over_rs"], rp[-1]["T_over_rs"], mx("CP_eps"), mx("CP_J"), Mx["head"]["accept"], nacc, len(rp),
            mn("CPbar_eps"), mx("CPbar_eps"), mn("CPbar_J"), mx("CPbar_J"), mx("CPbar_vs_dust_eps"), mx("CPbar_vs_dust_J"), mx("pbar_minus_sea")))
        nmd = sum(len(r["modes_differ_from_stored"]) for r in rp)
        allm = {}
        for r in rp:
            for k_, v_ in r["modes"].items():
                allm[k_] = allm.get(k_, 0) + v_
        L.append("  The replay's continuation modes (channel-records): %s; %d channel-records differ from the run's stored mode." % ("; ".join("%s %d" % kv for kv in sorted(allm.items())), nmd))
    # D_det per configuration
    L += ["", "### D_det per configuration (||Delta C_D||_op on the collar's detector modes; the first and last records, the largest)", "",
          "| member | kappa | theta | D_det at T0 | D_det at the end | largest D_det (T/rs) | two-copy against single-copy: largest abs. diff.; largest rel. diff. where D_det > 1e-12 | copy structure (largest) | collar bound at T0 | collar bound at the end | largest collar bound | largest single packet |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            ser = [(r["T_over_rs"], r["configs"][i_]) for r in s7]
            dmax = max(ser, key=lambda q: q[1]["D_det"])
            big = [q[1]["D_det_rel_check"] for q in ser if q[1]["D_det"] > 1e-12]
            L.append("| %s | %d | %.6f | %.3e | %.3e | %.3e (%.3f) | %.1e; %s | %.1e | %.3e | %.3e | %.3e | %.3e |" % (
                Mx["head"]["tag"], c0["kappa"], c0["theta"], ser[0][1]["D_det"], ser[-1][1]["D_det"], dmax[1]["D_det"], dmax[0],
                max(abs(q[1]["D_det"] - q[1]["D_det_single_copy"]) for q in ser), ("%.1e" % max(big)) if big else "none above 1e-12",
                max(q[1]["copy_structure"] for q in ser), ser[0][1]["collar_bound"], ser[-1][1]["collar_bound"],
                max(q[1]["collar_bound"] for q in ser), max(max(q[1]["per_packet"]) for q in ser)))
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if s7:
            L.append("")
            L.append("- %s: the propagator's rounding leakage max |(U_k)_{collar, window}| (kappa %d) per record %.1e to %.1e, summed over the %d steps from T0 %.1e." % (
                Mx["head"]["tag"], Mx["head"]["kappas"][0], min(r["leakage"] for r in s7), max(r["leakage"] for r in s7), len(s7) - 1, s7[-1]["leakage_sum"]))
    # the region bounds per record
    lad = [str(r) for r in hP["ladder_radii"]]
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        Tp = set(round(r["T_over_rs"], 6) for r in P_["s7"])
        L += ["", "### The region bounds per record, %s (%s, tau %g): 2 ||G_{R > R_t(t)}||_op exact while the surface persists (the largest ||Delta C_D|| of any detector beyond the surface), its Frobenius upper bound, the exact bound beyond R_t(T0) (site R >= %s fixed), the exact ladder (R >= r), the fronts (Frobenius), D_det and the collar's bound, the unrotated collar charge difference where solved; %s" % (
            role, h["tag"], h["tau"], fmt(h["R_t0_site"], "%.0f"), "every record" if Mx is P_ else "at the primary's records and every record while the surface persists"), ""]
        for i_, c0 in enumerate(s7[0]["configs"]):
            L += ["- kappa %d, theta %.6f:" % (c0["kappa"], c0["theta"]), "",
                  "| T/rs | R_t(t) | beyond: R >= | bound beyond R_t(t) | its Frobenius bound | beyond R_t(T0) | %s | front 1e-6 | front 1e-3 | D_det | collar bound | dE/E |" % " | ".join("r %s" % r for r in lad),
                  "|---|---|---|---|---|---|%s---|---|---|---|---|" % ("---|" * len(lad))]
            for r in s7:
                if not (Mx is P_ or r["persists"] or round(r["T_over_rs"], 6) in Tp):
                    continue
                q = r["configs"][i_]
                L.append("| %.3f | %s | %s | %s | %s | %s | %s | %.0f | %.0f | %.3e | %.2e | %s |" % (
                    r["T_over_rs"], fmt(r["R_t"], "%.4f"), fmt(r["R_t_site"], "%.0f"), fmt(q["bound_Rt"]), fmt(q["bound_Rt_fro"]), fmt(q["bound_Rt0"]),
                    " | ".join(fmt(q["ladder_op"][x]) for x in lad), q["front"]["1e-06"], q["front"]["0.001"], q["D_det"], q["collar_bound"],
                    fmt(q.get("charge", {}).get("dE_over_E"), "%+.4e")))
            L.append("")
    # the crossing clocks
    L += ["", "### The crossing clocks against the closing clock (33.4 (2)): the first record from T0 at which the exact bound beyond R_t(t) exceeds 1e-3 and 1e-6 while the surface persists; the contrasts beside it", "",
          "| member | kappa | theta | eps | the crossing (exact, beyond R_t(t)) | Frobenius bound beyond R_t(t) (contrast) | exact beyond R_t(T0) while the surface persists (contrast) | exact beyond R_t(T0) over the run (contrast) | the closing clock (r_s after T0) | the last record holding the surface |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        for cr in Mx["crossing"]:
            for ek, cv in cr["clocks"].items():
                L.append("| %s | %d | %.6f | %s | %s | %s | %s | %s | T/rs %s (%s) | T/rs %s (%s) |" % (
                    Mx["head"]["tag"], cr["kappa"], cr["theta"], ek, clock_str(cv["exact_Rt"]), clock_str(cv["fro_Rt"]), clock_str(cv["exact_Rt0_persisting"]),
                    clock_str(cv["exact_Rt0_run"]), fmt(cr["closing_T_over_rs"], "%.3f"), fmt(cr["closing_minus_T0_rs"], "%.3f"), fmt(cr["last_T_over_rs"], "%.3f"), fmt(cr["last_minus_T0_rs"], "%.3f")))
    L.append("")
    for Mx, role in Ms:
        for cr in Mx["crossing"]:
            L.append("- %s, kappa %d: the exact bound beyond R_t(t) over the %d records holding the surface from T0: largest %s at T/rs %s, at the last record holding it %s (T/rs %s)." % (
                Mx["head"]["tag"], cr["kappa"], cr["n_persisting_records"], fmt(cr["max_bound_Rt_persisting"][1] if cr["max_bound_Rt_persisting"] else None),
                fmt(cr["max_bound_Rt_persisting"][0] if cr["max_bound_Rt_persisting"] else None, "%.3f"), fmt(cr["last_persisting"][1] if cr["last_persisting"] else None),
                fmt(cr["last_persisting"][0] if cr["last_persisting"] else None, "%.3f")))
    # version 3: the half-zone split (36.3 (2)) and the gate against version 2
    L += halfzone_md(Ms)
    gates = gate_all(Ms, a.gate_v2.split(",")) if a.gate_v2 else None
    L += gate_md(gates)
    # the charge through the run
    L += ["", "### The unrotated collar charge difference through the run (the bar %g of E, 33.4 (2))" % hP["charge_tol"], "",
          "| member | kappa | theta | records solved | at T0 | at the end | largest |dE/E| (T/rs) | below the bar | dE/(E theta^2) over the run | the law's theta for the bar over the run |", "|---|---|---|---|---|---|---|---|---|---|"]
    csum = {}
    for Mx, role in Ms:
        csum[role] = charge_summary(Mx, Mx["head"]["charge_tol"])
        for c_ in csum[role]:
            L.append("| %s | %d | %.6f | %d | %+.4e | %+.4e | %.4e (%.3f) | %s | %.6e to %.6e | %s |" % (
                Mx["head"]["tag"], c_["kappa"], c_["theta"], c_["n_solved"], c_["at_T0"], c_["at_end"], c_["max_abs"], c_["max_at_T_over_rs"],
                "yes" if c_["below_bar"] else "NO", c_["coef_min"], c_["coef_max"], fmt(c_["theta_for_bar"], "%.6f")))
    # the step control
    if Cn is not None and Cn["s7"] and P_["s7"]:
        hC = Cn["head"]
        byT = {round(r["T_over_rs"], 6): r for r in Cn["s7"]}
        L += ["", "### The step control: %s (tau %g) against %s (tau %g) at the times both hold" % (hC["tag"], hC["tau"], hP["tag"], hP["tau"]), "",
              "| kappa | T/rs | R_t tau %g, tau %g | bound beyond the own R_t(t): tau %g, tau %g | exact r 32: tau %g, tau %g (rel. diff.) | exact r 128: tau %g, tau %g | front 1e-6 tau %g, tau %g | D_det tau %g, tau %g |" % (
                  hP["tau"], hC["tau"], hP["tau"], hC["tau"], hP["tau"], hC["tau"], hP["tau"], hC["tau"], hP["tau"], hC["tau"], hP["tau"], hC["tau"]),
              "|---|---|---|---|---|---|---|---|"]
        worst = {}
        for r in P_["s7"]:
            rc = byT.get(round(r["T_over_rs"], 6))
            if rc is None:
                continue
            for i_, q in enumerate(r["configs"]):
                qc = next((x for x in rc["configs"] if x["kappa"] == q["kappa"] and abs(x["theta"] - q["theta"]) < 1e-12), None)
                if qc is None:
                    continue
                b8, b4 = q["ladder_op"]["32"], qc["ladder_op"]["32"]
                fr = abs(b8 - b4) / max(b8, 1e-300)
                if b8 > 1e-12:
                    worst[q["kappa"]] = max(worst.get(q["kappa"], 0.0), fr)
                L.append("| %d | %.3f | %s, %s | %s, %s | %s, %s (%.2e) | %s, %s | %.0f, %.0f | %.3e, %.3e |" % (
                    q["kappa"], r["T_over_rs"], fmt(r["R_t"], "%.4f"), fmt(rc["R_t"], "%.4f"), fmt(q["bound_Rt"]), fmt(qc["bound_Rt"]), fmt(b8), fmt(b4), fr,
                    fmt(q["ladder_op"]["128"]), fmt(qc["ladder_op"]["128"]), q["front"]["1e-06"], qc["front"]["1e-06"], q["D_det"], qc["D_det"]))
        L.append("")
        L.append("- the largest relative difference of the exact r 32 bound between the step sizes over the matched times where it exceeds 1e-12, per kappa: %s." % (
            "; ".join("kappa %d: %.2e" % kv for kv in sorted(worst.items())) or "-"))
        for crP in P_["crossing"]:
            crC = next((x for x in Cn["crossing"] if x["kappa"] == crP["kappa"]), None)
            if crC is None:
                continue
            for ek in crP["clocks"]:
                a8 = crP["clocks"][ek]["exact_Rt"]; a4 = crC["clocks"][ek]["exact_Rt"]
                L.append("- kappa %d, eps %s: the crossing at tau %g %s, at tau %g %s%s." % (crP["kappa"], ek, hP["tau"], clock_str(a8), hC["tau"], clock_str(a4),
                         ("; the difference %.3f r_s" % abs(a8["T_over_rs"] - a4["T_over_rs"])) if (a8 and a4) else ""))
    # the sources after T0
    L += ["", "### The preparations' sources after T0 (copy-summed, with P(t) (+) P(t), against 2 (C_0(t) - P(t)) of the unrotated state; every %g r_s)" % a.md_every_rs, "",
          "| member | kappa | theta | T/rs | with P(t) (+) P(t) (eps; J) | its shells above 1e-15 of the raw density | held rotated R (P (+) P) R^dagger, the contrast (eps; J) | + against - (eps; J) | the C - P_bar contrast | Tr C_+ - 2 Tr C_0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    def shown(r, s7):
        x = (r["T_over_rs"] - s7[0]["T_over_rs"]) / a.md_every_rs
        return r is s7[0] or r is s7[-1] or abs(x - round(x)) < 1e-9
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            for r in s7:
                if not shown(r, s7):
                    continue
                q = r["configs"][i_]
                L.append("| %s | %d | %.6f | %.3f | %.3e; %.3e | %d | %.3e; %.3e | %.1e; %.1e | %.3e | %+.1e |" % (Mx["head"]["tag"], q["kappa"], q["theta"], r["T_over_rs"],
                         q["src_own"]["eps"], q["src_own"]["J"], q["src_own"]["shells_above_1e-15"], q["src_held"]["eps"], q["src_held"]["J"], q["src_pm_eps"], q["src_pm_J"],
                         q["contrast_Pbar_eps"], q["trC_plus"] - q["two_trC0"]))
    L += ["", "| member | kappa | theta | T/rs | solved: max |N_+ - N_-|, |v_+ - v_-|, |E_+ - E_-|/E | against the unreplaced total: max |dN|, |dv|, dE/E | E (the replayed total's solve) |", "|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        if not Mx["s7"]:
            continue
        r = Mx["s7"][-1]
        for q in r["configs"]:
            sv = q.get("solved")
            if sv:
                L.append("| %s | %d | %.6f | %.3f | %.1e, %.1e, %.1e | %.2e, %.2e, %+.3e | %.8f |" % (Mx["head"]["tag"], q["kappa"], q["theta"], r["T_over_rs"],
                         sv["N_pm"], sv["v_pm"], sv["E_pm_rel"], sv["N_vs_total"], sv["v_vs_total"], sv["E_vs_total_rel"], sv["E_total"]))
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        for kap in Mx["head"]["kappas"]:
            L.append("- %s, kappa %d: ||P - P_bar||_F %.4f at T0 and %.4f at the end; Tr P %.6f, Tr P_bar %.6f at the end." % (
                Mx["head"]["tag"], kap, s7[0]["P_minus_Pbar_F_k%d" % kap], s7[-1]["P_minus_Pbar_F_k%d" % kap], s7[-1]["trP_k%d" % kap], s7[-1]["trPbar_k%d" % kap]))
    # the inputs of 33.4 (3)'s sentence
    L += ["", "### The inputs of 33.4 (3)'s sentence (the lead's to print; the primary, kappa %d)" % hP["kappas"][0], ""]
    crP = P_["crossing"][0] if P_["crossing"] else None
    cs0 = csum["primary"][0] if csum.get("primary") else None
    if crP is not None:
        for ek in ("0.001", "1e-06"):
            cv = crP["clocks"][ek]["exact_Rt"]
            cc = None
            if Cn is not None and Cn["crossing"]:
                crC = next((x for x in Cn["crossing"] if x["kappa"] == crP["kappa"]), None)
                cc = crC["clocks"][ek]["exact_Rt"] if crC else None
            if cv is not None:
                L.append("- eps %s: a crossing while the surface persists: [x] the unrotated collar charge difference, at most %s of E over the run (%s at T0); [y] the closing clock %s r_s after T0 (the last record holding the surface %s r_s after T0); [t] %.3f r_s after T0 (T/rs %.3f); [D] %.3e (the exact bound beyond R_t(t) = %.4f, R >= %.0f); [z] the step control's crossing %s." % (
                    ek, fmt(cs0["max_abs"] if cs0 else None, "%.3e"), fmt(cs0["at_T0"] if cs0 else None, "%+.3e"), fmt(crP["closing_minus_T0_rs"], "%.3f"),
                    fmt(crP["last_minus_T0_rs"], "%.3f"), cv["after_T0_rs"], cv["T_over_rs"], cv["bound"], cv["R_t"], cv["R_t_site"],
                    ("at T/rs %.3f, %.3f r_s from the primary's" % (cc["T_over_rs"], abs(cc["T_over_rs"] - cv["T_over_rs"]))) if cc else "none while its surface persists"))
            else:
                L.append("- eps %s: no crossing while the surface persists (\"the preparations' difference did not cross the surface while it persisted\"): the largest exact bound beyond R_t(t) over the %d records holding it %s (T/rs %s); the closing clock %s r_s after T0; [x] %s of E at most over the run." % (
                    ek, crP["n_persisting_records"], fmt(crP["max_bound_Rt_persisting"][1] if crP["max_bound_Rt_persisting"] else None),
                    fmt(crP["max_bound_Rt_persisting"][0] if crP["max_bound_Rt_persisting"] else None, "%.3f"), fmt(crP["closing_minus_T0_rs"], "%.3f"),
                    fmt(cs0["max_abs"] if cs0 else None, "%.3e")))
    # what the specification left open
    pk0 = [p_ for p_ in hP["detector"][str(hP["kappas"][0])]["packets"] if "m_p" in p_]
    L += ["", "### Left open by the specification (the builder's readings, labelled)", "",
          "- theta: the law is fitted at T0 on %s; the run's theta is its root at %g of the bar (the builder's margin for the growth through the run, part 2's unrotated difference having moved between its T0 and its end), common to both kappas (the smaller root); the charge difference through the run is solved every %g r_s and its law's theta for the bar is printed beside the run's." % (
              ", ".join("%.6f" % t for t in hP["law_thetas"]), hP["law_margin"], hP["solve_every_rs"]),
          "- the surface's site: the region beyond R_t(t) is the sites with R > R_t(t) (R = index + 1); the bound beyond R_t(T0) keeps T0's site through the run, so it separates the difference's outward motion from the surface's own recession.",
          "- the step control: the tau %s member takes the primary's T0, X, w and theta (--match-json, version 1's rule); its own surface, its own closing clock and its own law are printed." % (fmt(Cn["head"]["tau"], "%g") if Cn else "-"),
          "- the detector's mass: J3''s probe set is light (m_p at most %.4f, omega %.4f to %.4f) against the model's m %g (version 1's label: an occupation measurement of the collar's modes)." % (
              max(p_["m_p"] for p_ in pk0), min(p_["omega"] for p_ in pk0), max(p_["omega"] for p_ in pk0), hP["m"]),
          "- P_bar's P_in: the run's initial reference (the C-init negative-energy set at T = 0), never a source here (version 1's).",
          "- the massless shell and the ell control: neither exists as a stored v17 history (version 1's finding); not run.",
          "- the half-zone split (version 3, 36.3 (2)): the transform is the periodic DFT on the region beyond R_t(t), whose first site is the surface's (the difference is cut there, not tapered), so a difference concentrated at that site spreads over both half-zones in the weight shares; E[cos a p] and E[cos 2 a p] are the site-overlap moments without wrap-around, beside them as the edge-free contrast. The boundary momenta |a p| = pi/2 (present when 4 divides M) join the upper half-zone, counted per record. The split is a reading, not a criterion (36.3 (2)).",
          "", "- every number above is printed by the script (engine_s7_twocopy_v3.py --compose); the per-record values of every column are in the json."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_s7_twocopy_v3.py", "members": M, "charge": csum, "gate_v2": gates, "md": os.path.basename(a.md)}, open(a.out_json, "w"), separators=(",", ":"), default=float)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default=None, help="a stored engine_collapse_v17 run: dir/tag")
    ap.add_argument("--T0-over-rs", dest="T0_over_rs", type=float, default=None)
    ap.add_argument("--T0-range", dest="T0_range", default="17.5,18.5", help="33.4 (2): T0 the record in this T/rs range with the largest R_t - r_in")
    ap.add_argument("--windows", default=None, help="the window's centre X (sites), overriding the rule")
    ap.add_argument("--match-json", default=None, help="the primary member's json: its T0, X, w and theta (the step control)")
    ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--law-thetas", default="0.39269908169872414,0.19634954084936207,0.09817477042468103,0.04908738521234052", help="pi/8, pi/16, pi/32, pi/64")
    ap.add_argument("--law-margin", type=float, default=0.8, help="the law's target as a fraction of the bar (the builder's margin for the growth through the run)")
    ap.add_argument("--theta", type=float, default=None, help="override the law's theta")
    ap.add_argument("--charge-tol", type=float, default=1e-6, help="33.4 (2): the bar on the unrotated collar charge difference, relative to E")
    ap.add_argument("--solve-every-rs", type=float, default=0.25, help="the charge difference solved every this many r_s from T0 (and at the end)")
    ap.add_argument("--rule-check-max", type=int, default=4, help="the rotated reference followed by the model's rule for at most this many records (contrast)")
    ap.add_argument("--width", type=float, default=4.0, help="the C^2 ramp's full width w (sites)")
    ap.add_argument("--ladder-radii", default="32,64,128,256,512")
    ap.add_argument("--omega-over-kappa", default="0.5,1,2")
    ap.add_argument("--mp-over-kappa", default="0,0.25")
    ap.add_argument("--accept", type=float, default=1e-8)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--T-end-over-rs", dest="T_end_over_rs", type=float, default=None, help="version 3: the run ends at the last stored slice at or before this T/rs (36.3 (2): 20.0)")
    ap.add_argument("--gate-v2", dest="gate_v2", default=None, help="compose (version 3): version 2's member jsons in the members' order, the gate's reference")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--compose", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--md-every-rs", type=float, default=2.0, help="compose: the sources table's time stride (T0 and the end always shown)")
    a = ap.parse_args()
    if a.compose:
        compose(a)
    else:
        member(a)


if __name__ == "__main__":
    main()
