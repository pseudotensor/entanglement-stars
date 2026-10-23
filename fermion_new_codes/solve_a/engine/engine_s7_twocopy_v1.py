#!/usr/bin/env python
"""engine_s7_twocopy_v1.py (builder analyst; engine spec 29.5, S7: "Two copies of one channel (kappa 1; kappa 2 the check) form C_0 (+) C_0;
C_+- = R_+- (C_0 (+) C_0) R_+-^dagger with R_+- = sum_x |x><x| (x) exp(+-i theta chi_X(x) sigma_y) in copy space ... both preparations then
propagate on that member's stored history by engine_replay_v4's propagator with the two-copy block, D_det as 28.5 specifies"; 28.5, S7: "D_det
at least ||Delta C_D||_op on the detector packets (J3''s probe set at the collar) per record, the tau 4 member the step control and the r_s 32
ell 8 member the ell control where it exists; C, P and P_bar stored apart, every source audited to use C - P"; 30.5, S7 (binding, turn 38):
"part 2 uses the rotated reference for each preparation, the compact C^2 window (the ramp s^3 (10 - 15 s + 6 s^2) of 28.5 (b)), the transition
at the dust shell density's minimum inside [r_in, R_t] (the profile printed), and prints the remaining normal-ordered source difference and
the collar charge difference as the preparations' label (theta reduced to pi/8 if the charge difference exceeds 1e-6 of E); D_det as 28.5";
REVIEWA_REVIEW_web40e S7; the goal's task 5, P36's level (iii)): S7 part 2 (part 1: engine_s7_prep_check_v1.py).

Per member (a stored engine_collapse_v17 run: its json and fields npz), the script computes:
(1) The replay by the member's own rule (engine_collapse_v19.resume_state's loop, here over the whole history): the C-init state re-prepared
(engine_collapse_v17.prepare_dust; its fields against the stored N_cinit, v_cinit), the sea the negative-energy eigenvectors of h_m at the
stored C-init fields; at every record the eigendecompositions of h_m at the stored applied fields (N_applied, v_applied), the reference P (R2)
continued by engine_collapse_v17.continue_v17 from the previous record's (verification_differs forcing the subdivisions, as the run did; the
mode compared with the stored one), then every column propagated by the record's step U_k = V_k exp(-i e_k tau) V_k^dagger (engine_replay_v4's
step). All the member's channels are carried through the whole history, so the sources are audited against the stored totals at every record.
(2) C, P and P_bar apart, per channel: C = sea sea^dagger + dust f dust^dagger (the propagated state), P the continued reference's columns,
P_bar = U P_in U^dagger with P_in the run's initial reference (the C-init negative-energy set), its columns propagated as a separate object
(equal to the sea's columns, the run's initial sea being its initial reference: printed). The audit: every source the script computes passes
through one function that takes the reference's columns explicitly and records which object it received with the call's role (a source, or
a contrast that sources nothing); the printed table passes when every source subtracted the record's continued P or, for a preparation, its
own reference P_+-. The numeric audit per record: sum_kappa w Tr[a_x(h)(C - P)] and the momentum density against the stored eps_tot, J_tot
(acceptance --accept, 1e-8, relative to each stored array's largest value), the contrast C - P_bar against the stored totals and against the
stored dust part eps_tot - eps_pairs.
(3) The window (30.5): chi_X(r) = 1 - S((r - X)/w + 1/2), S(s) = s^3 (10 - 15 s + 6 s^2) on [0, 1] (0 below, 1 above; C^2, compact): chi = 1 for
r <= X - w/2 and 0 for r >= X + w/2, w = --width sites (4, BUILDER); X at the minimum of the dust shell density over the sites inside
[r_in, R_t] of the record T0 (the dust's occupation per shell, sum_kappa w sum_d f_d |phi_d|^2 on the site's components, from the replayed
dust; its energy per shell beside), the profile printed; --windows X or --match-json (the primary's X, w and thetas) override.
(4) The preparations at T0 (default: the record nearest T/rs 20 holding both trapping zeros, part 1's record): for kappa in --kappas and theta
the two-copy columns Y_+- = R_+- (Phi (+) Phi) (Phi = [sea, dust] the channel's propagated state at T0, f = 1 on the sea and the dust's
occupations) and the preparation's reference rotated with it, Pi_+- = R_+- (ref (+) ref) (P_+-(T0) = R_+- (P_0 (+) P_0) R_+-^dagger, 30.5),
R_+- = [[c, +-s], [-+s, c]] per site on both components (c = cos(theta chi_X), s = sin(theta chi_X); part 1's copy-major layout and sign).
The label (30.5): the copy-summed normal-ordered sources of C_+- - P_+- against 2 (C_0 - P_0) (their remaining difference: largest relative
to max |2 (C_0 - P_0)|, the shells above 1e-15 of the raw density's largest value (part 1's convention), the largest away from the transition), the same with the unrotated P_0 (+) P_0 as the
contrast, + against -, and the collar charge difference: the member's total sources with the two copies replaced by the preparation's,
solved by the run's own solver (engine_collapse_v17.solve, the run's site-field rule), E, N, v against the unreplaced total's solve. A theta
in --thetas whose charge difference exceeds 1e-6 of E is replaced by pi/8 (30.5; the label at pi/8 printed, kept even if it also exceeds).
(5) From T0 to the run's end: Y_+ and Y_- (4n rows each) propagated apart by the record's step acting within each copy (the two-copy block
h_m (x) 1: U_k applied to each copy's rows); each preparation's reference continued at every record by the model's rule on the two-copy
block (engine_collapse_v17.continue_fast_info, else continue_reference_resolved_v17, on the eigendecomposition of h_m (x) 1, the copy pairs
degenerate; the Weyl test and the subdivisions are not applied to the two-copy continuation, whose gap is undefined at a rotated reference:
labelled) until it coincides with P(t) (+) P(t) (projector distance below 1e-8, printed), after which it is P(t) (+) P(t) exactly (the
two-copy continuation of a doubled reference is the doubled continuation).
(6) Per record from T0 on, per configuration: D_det >= ||Delta C_D||_op, Delta C = C_+ - C_-, on the detector packets in both copies (F_D (x) 1,
F_D the orthonormal span of the restricted packets below), from the two-copy columns; beside it the single-copy identity (A = U c Phi, B =
U s Phi, Y_+ = [[A, B], [-B, A]], Y_- = [[A, -B], [B, A]]: Delta C = [[0, 2G], [-2G, 0]], G = B f A^dagger - A f B^dagger, ||Delta C_D||_op =
2 ||F_D^dagger G F_D||_op; the relative difference and the copy structure of Y_+- printed); per packet 2 |w^dagger G w|; the collar's bound
||Pi_c Delta C Pi_c||_op over every mode of the collar's sites in both copies; the escape ladder 2 ||G_{R >= r}||_op (exact) for r in
--ladder-radii every --ladder-every-rs r_s, at T0 and at the end, its Frobenius upper bound 2 ||G_{R >= r}||_F at every record and the front
(the outermost r whose bound exceeds eps, eps in 1e-12, 1e-9, 1e-6, 1e-3); the propagator's rounding leakage max |(U_k)_{collar, window}|
per record and its sum since T0 (the floor D_det at the collar is read against). The preparations' copy-summed sources from C_+- - P_+-(t):
+ against -, against 2 (C_0(t) - P(t)) of the channel's unrotated state; as contrasts (sourcing nothing) the reference held rotated,
R (P(t) (+) P(t)) R^dagger, and P_bar (+) P_bar; the traces; ||P - P_bar||_F per channel; at the end the solved table again.

The detector (BUILDER, labelled; the brief: "the outgoing packets of J3''s probe set restricted to the collar"): J3''s construction
(obs_raymap_solved_v5: flow_chart_v4.branch_momenta on the outgoing branch of the exterior v = -sqrt(r_s/x) at the packet's centre with
sigma_omega = omega/6, flow_chart_v4.build_packet with the local positive spinor of m_p SZ + (kappa/x) SY + sin k SX, kappa the S7 channel's),
J3''s probe set omega/kappa_p in --omega-over-kappa (1/2, 1, 2) and m_p/kappa_p in --mp-over-kappa (0, 0.25) at kappa_p = 1/(2 r_s) with the
member's nominal r_s (version 5's rule when the late slice holds no surface, the nominal r_s in place of 2 g E_late so that both step sizes
carry the same packets: version 6's fixed-packet rule, 29.4), centred at the collar (the sites beyond the record's M1 cut, R = cut + 1 .. n;
the engine's collar is the chain's last node, r_collar = n), restricted to the collar's sites and renormalized (the retained norm printed),
orthonormalized as a set by column-pivoted QR (the Gram eigenvalues printed; those below 1e-10 of the largest dropped).

Compose mode (--compose primary.json[,control.json]): the md and json of the deliverable from the members' jsons; the step control at the
times both members hold. Every number in the outputs is printed by the script. Args: --member dir/tag --T0-over-rs --windows --match-json
--kappas --thetas --theta-reduced --charge-tol --width --ladder-radii --ladder-every-rs --omega-over-kappa --mp-over-kappa --accept --gpu
--threads --records --out-json --out-npz; --compose a.json,b.json --md --out-json --md-every-rs."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import math
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
K_P = "P (the record's continued reference)"
K_PBAR = "P_bar (the propagated initial reference)"
K_PPM = "P_+- (the preparation's reference: R (P_0 (+) P_0) R^dagger at T0, continued by the model's rule)"
K_PP = "P (+) P (the unrotated continued reference in both copies)"
K_HELD = "R (P(t) (+) P(t)) R^dagger (the reference held rotated)"
SOURCE_KINDS = (K_P, K_PPM)


class Audit:
    """every source the script computes passes here: the reference's columns are an explicit argument, and the call is recorded with its
    label, its role ("source", or "contrast" which sources nothing) and which registered object the reference was."""

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
                         "verdict": (("C - P" if k_ == K_P else "C - P_+-") if good else "FAIL") if role == "source" else "contrast (sources nothing)"})
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
    if a.T0_over_rs is None:
        cand = [r for r in recs.values() if r.get("R_t") is not None and r.get("r_in") is not None]
        rec0 = min(cand, key=lambda r: abs(float(r["T"]) / rs - 20.0))
        t0_rule = "the record nearest T/rs 20 holding both trapping zeros (part 1's record)"
    else:
        hit = [r for r in recs.values() if abs(float(r["T"]) / rs - a.T0_over_rs) < 1e-9]
        if not hit:
            raise RuntimeError("no record at T/rs %g" % a.T0_over_rs)
        rec0 = hit[0]
        t0_rule = "--T0-over-rs %g" % a.T0_over_rs
    T0 = float(rec0["T"])
    k0 = int(np.argmin(np.abs(Tz - T0)))
    if abs(Tz[k0] - T0) > 1e-9 or k0 >= K:
        raise RuntimeError("T0 %.3f is not a stored slice within the records replayed" % (T0 / rs))
    match = json.load(open(a.match_json)) if a.match_json else None
    kappas = [int(x) for x in a.kappas.split(",")]
    thetas_req = [float(x) for x in a.thetas.split(",")]
    ladder = [int(x) for x in a.ladder_radii.split(",")]
    width = float(match["head"]["window"]["width"]) if match else a.width
    cuts = sorted({int(r["M1_parts"]["cut"]) for r in recs.values() if r.get("M1_parts") and r["M1_parts"].get("cut") is not None})
    cut = int(rec0["M1_parts"]["cut"])
    collars = sorted({float(r["r_collar"]) for r in recs.values() if r.get("r_collar") is not None})
    det = {}
    for kap in kappas:
        F, Wm, dinfo = build_detector(n, kap, rs, [float(x) for x in a.omega_over_kappa.split(",")], [float(x) for x in a.mp_over_kappa.split(",")], cut)
        det[kap] = {"F": C5.T_(F), "W": C5.T_(Wm), "info": dinfo}
    ic = torch.arange(2 * cut, 2 * n, device=dev)                       # the collar's components (sites R = cut + 1 .. n)
    head = {"script": "engine_s7_twocopy_v1.py", "member": a.member, "tag": tag, "member_script": J.get("script"), "rs": rs, "tau": tau, "nsites": n, "m": m, "g": g,
            "ell": ell, "kappa_sea": len(chs), "T0": T0, "T0_over_rs": T0 / rs, "T0_rule": t0_rule, "k0": k0, "records": K, "T_end_over_rs": float(Tz[K - 1]) / rs,
            "trapping_zeros_at_T0": {"r_in": rec0.get("r_in"), "R_t": rec0.get("R_t")}, "E_stored_at_T0": rec0.get("E"), "kappas": kappas, "thetas_requested": thetas_req,
            "theta_reduced": a.theta_reduced, "charge_tol": a.charge_tol, "ladder_radii": ladder, "ladder_every_rs": a.ladder_every_rs, "m1_cuts": cuts, "r_collars": collars,
            "detector": {str(k): det[k]["info"] for k in kappas}, "accept": a.accept, "threads": a.threads, "gpu": a.gpu, "match_json": a.match_json}
    print("member %s (%s): r_s %g tau %g n %d; T0/rs %.3f (%s, k0 %d), %d records to T/rs %.3f; M1 cuts %s; collars %s" % (
        tag, J.get("script"), rs, tau, n, T0 / rs, t0_rule, k0, K, Tz[K - 1] / rs, cuts, collars), flush=True)
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

    def cs_of(chi, th):
        c = torch.as_tensor(np.repeat(np.cos(th * chi), 2), dtype=torch.complex128, device=dev)[:, None]
        s = torch.as_tensor(np.repeat(np.sin(th * chi), 2), dtype=torch.complex128, device=dev)[:, None]
        return c, s

    def src_stats(es, js, e0, j0, X, raw):
        """the copy-summed source (es, js) against 2 (C_0 - P) = (2 e0, 2 j0): relative to the comparator's largest value; the shells
        counted by part 1's convention, above 1e-15 of the raw (not normal-ordered) density's largest value, 2 max |raw| of C_0's columns
        (the normal-ordered scale lies below the raw densities' rounding, so a count against it measures the rounding)."""
        sc_e = max(float(np.max(np.abs(2 * e0))), 1e-300); sc_j = max(float(np.max(np.abs(2 * j0))), 1e-300)
        rw_e = max(float(np.max(np.abs(2 * raw[0]))), 1e-300); rw_j = max(float(np.max(np.abs(2 * raw[1]))), 1e-300)
        de = np.abs(es - 2 * e0); dj = np.abs(js - 2 * j0)
        bad = np.nonzero((de > 1e-15 * rw_e) | (dj > 1e-15 * rw_j))[0]
        far = np.abs(R - X) > width
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
        # the audit's sources at this record, every channel: C - P (the run's sources) and the contrast C - P_bar
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
        # the window, the labels and the preparations at T0 (30.5)
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
            if a.windows:
                X = float(a.windows.split(",")[0]); wrule = "--windows"
            elif match:
                X = float(match["head"]["window"]["X"]); wrule = "--match-json %s (the primary's X)" % os.path.basename(a.match_json)
            else:
                if rin is None or Rt is None:
                    raise RuntimeError("the record at T0 holds no trapping zeros; give --windows or --match-json")
                inside = np.nonzero((R >= float(rin)) & (R <= float(Rt)))[0]
                i_min = inside[int(np.argmin(nd[inside]))]
                X = float(R[i_min])
                wrule = "the minimum of the dust shell density over the sites inside [r_in, R_t] = [%.4f, %.4f] of T0's record" % (rin, Rt)
            chi = window(R, X, width)
            iw = torch.arange(0, 2 * min(n, int(math.ceil(X + 0.5 * width))), device=dev)     # chi > 0 only below X + w/2
            if rin is not None and Rt is not None:
                ins = np.nonzero((R >= float(rin)) & (R <= float(Rt)))[0]
                interior_min = bool(R[ins[0]] < X < R[ins[-1]]) if len(ins) else None    # strictly between the interval's end sites
                lo_p = int(max(1, math.floor(min(X, float(rin))) - 4)); hi_p = int(min(n, math.ceil(max(X, float(Rt))) + 4))
            else:
                interior_min = None
                lo_p = int(max(1, math.floor(X) - 8)); hi_p = int(min(n, math.ceil(X) + 8))
            prof = [{"R": float(R[i]), "dust_per_shell": float(nd[i]), "dust_energy_per_shell": float(ed[i]), "stored_dust_energy": float(esd[i]), "chi": float(chi[i]),
                     **{"dust_k%d" % kq: float(per[kq][i]) for kq in kappas if kq in per}} for i in range(lo_p - 1, hi_p)]
            head["window"] = {"X": X, "width": width, "rule": wrule, "form": "chi = 1 - S((r - X)/w + 1/2), S(s) = s^3 (10 - 15 s + 6 s^2) on [0, 1]",
                              "support_end": X + 0.5 * width, "R_t": Rt, "r_in": rin, "interior_minimum": interior_min, "profile": prof,
                              "dust_total": float(np.sum(nd)), "N_B": head["cinit"]["N_B"]}
            print("window: X %.4f (%s), w %g, chi > 0 for R < %.2f (R_t %s); the minimum %s the interval's ends" % (X, wrule, width, X + 0.5 * width, Rt,
                  "lies between" if interior_min else "lies at one of"), flush=True)
            N_run, v_run, E_run, fail_run = solve_total(eP, jP)
            head["solve_at_T0"] = {"E_total": E_run, "E_stored": rec0.get("E"), "failed": fail_run}
            labels = []

            def make_label(kap, th):
                ch = byk[kap]; s_ = st[kap]; ref = cache[kap][2]
                Phi = torch.cat([s_["sea"], s_["dust"]], dim=1)
                f = np.concatenate([np.ones(s_["n_sea"]), np.asarray(s_["f"], float)]); f2 = np.concatenate([f, f])
                c, s = cs_of(chi, th)
                e0, j0 = AUD.subtract(ch, Cpart[kap], ref, "S7: the comparator, the channel's state (C_0(t) - P(t)), twice", "source", [(ref, K_P), (s_["pbar"], K_PBAR)])
                D = doubled(ref)
                res = {}
                for sign in (+1, -1):
                    Y = twocopy_cols(Phi, c, s, sign); Pi = twocopy_cols(ref, c, s, sign)
                    known = [(Pi, K_PPM), (D, K_PP)]
                    C2 = cdens2(ch, Y, f2)
                    e_r, j_r = AUD.subtract(ch, C2, Pi, "S7 label at T0: C_+- - R (P_0 (+) P_0) R^dagger (the preparation's reference)", "source", known, twocopy=True)
                    e_u, j_u = AUD.subtract(ch, C2, D, "S7 label's contrast at T0: C_+- - P_0 (+) P_0 (the unrotated reference)", "contrast", known, twocopy=True)
                    res[sign] = (Y, Pi, e_r, j_r, e_u, j_u)
                lab = {"kappa": kap, "theta": th, "rotated": src_stats(res[1][2], res[1][3], e0, j0, X, Cpart[kap]), "unrotated": src_stats(res[1][4], res[1][5], e0, j0, X, Cpart[kap]),
                       "pm_eps": float(np.max(np.abs(res[1][2] - res[-1][2]))) / max(float(np.max(np.abs(2 * e0))), 1e-300),
                       "pm_J": float(np.max(np.abs(res[1][3] - res[-1][3]))) / max(float(np.max(np.abs(2 * j0))), 1e-300)}
                sol = {}
                for sign in (+1, -1):
                    Nq, vq, Eq, fq = solve_total(eP + res[sign][2] - 2 * e0, jP + res[sign][3] - 2 * j0)
                    sol[sign] = (Nq, vq, Eq, fq)
                Nu, vu, Eu, fu = solve_total(eP + res[1][4] - 2 * e0, jP + res[1][5] - 2 * j0)
                lab["solved"] = {"E_plus": sol[1][2], "E_minus": sol[-1][2], "E_total": E_run, "dE_over_E": (sol[1][2] - E_run) / abs(E_run),
                                 "dE_pm_over_E": (sol[1][2] - sol[-1][2]) / abs(E_run), "N_vs_total": float(np.max(np.abs(sol[1][0] - N_run))),
                                 "v_vs_total": float(np.max(np.abs(sol[1][1] - v_run))), "N_pm": float(np.max(np.abs(sol[1][0] - sol[-1][0]))),
                                 "v_pm": float(np.max(np.abs(sol[1][1] - sol[-1][1]))), "dE_over_E_unrotated": (Eu - E_run) / abs(E_run),
                                 "failed": [sol[1][3], sol[-1][3], fail_run, fu]}
                lab["charge_ok"] = bool(abs(lab["solved"]["dE_over_E"]) <= a.charge_tol)
                return lab, {"Yp": res[1][0], "Pi_p": res[1][1], "Ym": res[-1][0], "Pi_m": res[-1][1], "f": f, "f2": f2, "nc": int(Phi.shape[1])}

            for kap in kappas:
                if match:
                    th_list = [float(x) for x in match["head"]["thetas_final"][str(kap)]]
                    cand_list = th_list
                else:
                    cand_list = list(thetas_req)
                built, failed = {}, False
                for th in cand_list:
                    lab, cols = make_label(kap, th)
                    lab["requested"] = True; labels.append(lab)
                    print("label kappa %d theta %.4f: rotated reference: source difference %.2e (eps), %.2e (J), %d shells above 1e-15; unrotated %.2e, %.2e; E %+.3e of E (unrotated %+.3e); + against - %.1e" % (
                        kap, th, lab["rotated"]["eps"], lab["rotated"]["J"], lab["rotated"]["shells_above_1e-15"], lab["unrotated"]["eps"], lab["unrotated"]["J"],
                        lab["solved"]["dE_over_E"], lab["solved"]["dE_over_E_unrotated"], lab["pm_eps"]), flush=True)
                    if match or lab["charge_ok"]:
                        built[th] = (lab, cols)
                    else:
                        failed = True
                        del cols
                if failed and a.theta_reduced not in built:
                    lab, cols = make_label(kap, a.theta_reduced)
                    lab["requested"] = False; lab["reduced_from"] = [l_["theta"] for l_ in labels if l_["kappa"] == kap and not l_["charge_ok"] and l_["requested"]]
                    labels.append(lab)
                    print("label kappa %d theta %.4f (reduced): rotated reference: source difference %.2e (eps), %.2e (J); E %+.3e of E" % (
                        kap, a.theta_reduced, lab["rotated"]["eps"], lab["rotated"]["J"], lab["solved"]["dE_over_E"]), flush=True)
                    built[a.theta_reduced] = (lab, cols)
                for th in sorted(built):
                    lab, cols = built[th]
                    cfgs.append({"kappa": kap, "theta": th, "X": X, "label": lab, "snapped": False, "Pi_trace": [], **cols})
            head["labels"] = labels
            head["thetas_final"] = {str(kap): sorted(cf["theta"] for cf in cfgs if cf["kappa"] == kap) for kap in kappas}
            print("T/rs %.3f: %d preparations (kappa, theta): %s" % (T / rs, len(cfgs), ", ".join("(%d, %.4f)" % (cf["kappa"], cf["theta"]) for cf in cfgs)), flush=True)
        # each preparation's reference continued by the model's rule (two-copy block) until it coincides with P (+) P
        Dk = {}
        if k >= k0:
            for kap in kappas:
                Dk[kap] = doubled(cache[kap][2])
        if k > k0:
            for cf in cfgs:
                kap = cf["kappa"]
                if cf["snapped"]:
                    cf["Pi_p"] = cf["Pi_m"] = Dk[kap]
                    continue
                e, U, ref, ci = cache[kap]
                rowp = {"T_over_rs": T / rs}
                for key in ("Pi_p", "Pi_m"):
                    new, mode, info = continue_twocopy(cf[key], e, U, 2 * st[kap]["n_sea"])
                    if new is None:
                        rowp[key] = {"mode": "unresolved (%s)" % {k_: info.get(k_) for k_ in ("ambiguous_blocks", "n_occ", "n_sea")}, "distance_to_doubled": None}
                        new = Dk[kap]
                    else:
                        rowp[key] = {"mode": mode, "distance_to_doubled": projdist(new, Dk[kap]), "min_occupied_weight": info.get("min_occupied_weight"),
                                     "ambiguous_blocks": info.get("ambiguous_blocks"), "partly_occupied_blocks": info.get("partly_occupied_blocks")}
                    cf[key] = new
                cf["Pi_trace"].append(rowp)
                dd = [rowp[key]["distance_to_doubled"] for key in ("Pi_p", "Pi_m")]
                if all(x is not None and x < 1e-8 for x in dd):
                    cf["snapped"] = True; cf["snapped_at"] = T / rs
                    cf["Pi_p"] = cf["Pi_m"] = Dk[kap]
                print("T/rs %.3f: kappa %d theta %.4f: the preparations' references continued (%s, %s); distance to P (+) P %s; %s" % (
                    T / rs, kap, cf["theta"], rowp["Pi_p"]["mode"], rowp["Pi_m"]["mode"], ", ".join(fmt(x, "%.2e") for x in dd),
                    "coincides from here on" if cf["snapped"] else "kept"), flush=True)
        # per record from T0 on
        if k >= k0:
            exact = (k == k0) or (k == K - 1) or (abs((T - T0) / (a.ladder_every_rs * rs) - round((T - T0) / (a.ladder_every_rs * rs))) < 1e-9)
            do_solve = (k == K - 1)
            if do_solve:
                N_run, v_run, E_run, fail_run = solve_total(eP, jP)
            e1, U1, ref1, _ = cache[kappas[0]]
            ph1 = torch.exp(-1j * e1.to(torch.complex128) * tau)
            leak = float(((U1[ic] * ph1[None, :]) @ U1[iw].conj().T).abs().max())
            leak_sum += leak
            srow = {"k": k, "T_over_rs": T / rs, "leakage": leak, "leakage_sum": leak_sum, "exact_ladder": exact, "configs": []}
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
                fro = {str(r): float(ub[r - 1]) for r in ladder}
                front = {}
                for ep_ in EPS_FRONT:
                    idx = np.nonzero(ub > ep_)[0]
                    front["%g" % ep_] = float(R[idx[-1]]) if len(idx) else 0.0
                opn = None
                if exact:
                    opn = {}
                    for r in ladder:
                        blk = G[2 * (r - 1):, 2 * (r - 1):]
                        opn[str(r)] = 2.0 * float(torch.linalg.eigvalsh(1j * blk).abs().max())
                del G, Wsp, S2
                # the preparations' sources: the preparation's own reference P_+-(t); contrasts: held rotated, P_bar (+) P_bar
                chi = window(R, cf["X"], width)
                c, s = cs_of(chi, cf["theta"])
                Pheld = twocopy_cols(ref, c, s, +1)
                known = [(cf["Pi_p"], K_PPM), (cf["Pi_m"], K_PPM), (Dk[kap], K_PP), (Pheld, K_HELD), (ref, K_P), (s_["pbar"], K_PBAR)]
                Cp = cdens2(ch, Yp, cf["f2"]); Cm = cdens2(ch, Ym, cf["f2"])
                ep_s, jp_s = AUD.subtract(ch, Cp, cf["Pi_p"], "S7: the preparation C_+, copy-summed (C_+ - P_+(t))", "source", known, twocopy=True)
                em_s, jm_s = AUD.subtract(ch, Cm, cf["Pi_m"], "S7: the preparation C_-, copy-summed (C_- - P_-(t))", "source", known, twocopy=True)
                e0, j0 = AUD.subtract(ch, Cpart[kap], ref, "S7: the comparator, the channel's state (C_0(t) - P(t)), twice", "source", known)
                eh_s, jh_s = AUD.subtract(ch, Cp, Pheld, "S7: the contrast C_+ - R (P(t) (+) P(t)) R^dagger (the reference held rotated)", "contrast", known, twocopy=True)
                eb_s, jb_s = AUD.subtract(ch, Cp, s_["pbar"], "S7: the contrast C_+ - P_bar (+) P_bar", "contrast", known, mult=2.0)
                del Pheld
                sp = src_stats(ep_s, jp_s, e0, j0, cf["X"], Cpart[kap]); sh = src_stats(eh_s, jh_s, e0, j0, cf["X"], Cpart[kap])
                crow = {"kappa": kap, "theta": cf["theta"], "X": cf["X"], "D_det": Ddet, "D_det_single_copy": Ddet1,
                        "D_det_rel_check": abs(Ddet - Ddet1) / max(Ddet, Ddet1, 1e-300), "copy_structure": struct, "per_packet": [float(x) for x in pk],
                        "collar_bound": Dcol, "ladder_fro": fro, "ladder_op": opn, "front": front, "reference_snapped": cf["snapped"],
                        "src_pm_eps": float(np.max(np.abs(ep_s - em_s))) / sp["scale_eps"], "src_pm_J": float(np.max(np.abs(jp_s - jm_s))) / sp["scale_J"],
                        "src_own": sp, "src_held": sh, "contrast_Pbar_eps": float(np.max(np.abs(eb_s - ep_s))) / sp["scale_eps"],
                        "trC_plus": float(((Yp.abs() ** 2).sum(dim=0) * f2t.real).sum()), "trC_minus": float(((Ym.abs() ** 2).sum(dim=0) * f2t.real).sum()),
                        "two_trC0": 2.0 * float(np.sum(cf["f"])), "trPi_plus": float((cf["Pi_p"].abs() ** 2).sum())}
                if do_solve:
                    sol = {}
                    for lab, (es_, js_) in (("+", (ep_s, jp_s)), ("-", (em_s, jm_s))):
                        sol[lab] = solve_total(eP + es_ - 2 * e0, jP + js_ - 2 * j0)
                    crow["solved"] = {"N_pm": float(np.max(np.abs(sol["+"][0] - sol["-"][0]))), "v_pm": float(np.max(np.abs(sol["+"][1] - sol["-"][1]))),
                                      "E_pm_rel": abs(sol["+"][2] - sol["-"][2]) / abs(E_run), "N_vs_total": float(np.max(np.abs(sol["+"][0] - N_run))),
                                      "v_vs_total": float(np.max(np.abs(sol["+"][1] - v_run))), "E_vs_total_rel": (sol["+"][2] - E_run) / abs(E_run), "E_total": E_run,
                                      "failed": [sol["+"][3], sol["-"][3], fail_run]}
                srow["configs"].append(crow)
            srow["seconds"] = time.time() - tr
            s7_rows.append(srow)
            print("T/rs %.3f: D_det %s; collar bound %s; front(1e-6) %s; check %.1e; source vs 2 (C_0 - P): own %s, held %s; leakage %.1e (sum %.1e); replay eps %.1e J %.1e (C - P_bar %.1e); %.1f s" % (
                T / rs, ", ".join("%.3e" % q["D_det"] for q in srow["configs"]), ", ".join("%.2e" % q["collar_bound"] for q in srow["configs"]),
                ", ".join("%.0f" % q["front"]["1e-06"] for q in srow["configs"]), max(q["D_det_rel_check"] for q in srow["configs"]),
                ", ".join("%.1e" % q["src_own"]["eps"] for q in srow["configs"]), ", ".join("%.1e" % q["src_held"]["eps"] for q in srow["configs"]),
                leak, leak_sum, rr["CP_eps"], rr["CP_J"], rr["CPbar_eps"], srow["seconds"]), flush=True)
        else:
            if k % 10 == 0 or k == k0 - 1:
                print("T/rs %.3f: replay eps %.2e J %.2e (C - P_bar against the stored %.2e, against the stored dust part %.2e); modes %s; %.1f s (eigh %.1f)" % (
                    T / rs, rr["CP_eps"], rr["CP_J"], rr["CPbar_eps"], rr["CPbar_vs_dust_eps"], modes, time.time() - tr, t_eig), flush=True)
        # the reference's state after the record (resume_state's order)
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
                states["T0_cfg%d_Pi_plus" % i_] = cf["Pi_p"].cpu().numpy()
        if a.out_npz and k == K - 1:
            states.update(snapshot("end", cache))
            for i_, cf in enumerate(cfgs):
                states["end_cfg%d_A" % i_] = cf["Yp"][:2 * n, :cf["nc"]].cpu().numpy(); states["end_cfg%d_B" % i_] = cf["Yp"][:2 * n, cf["nc"]:].cpu().numpy()
                states["end_cfg%d_f" % i_] = cf["f"]
                states["cfg%d_params" % i_] = np.array([cf["kappa"], cf["X"], cf["theta"], width])
        # the propagation by the record's step
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
            out["references"] = [{"kappa": cf["kappa"], "theta": cf["theta"], "snapped_at_T_over_rs": cf.get("snapped_at"), "trace": cf["Pi_trace"]} for cf in cfgs]
            dump()
    out["references"] = [{"kappa": cf["kappa"], "theta": cf["theta"], "snapped_at_T_over_rs": cf.get("snapped_at"), "trace": cf["Pi_trace"]} for cf in cfgs]
    dump()
    if a.out_npz:
        np.savez(a.out_npz, **states)
    print("done: %d records in %.0f s" % (K, time.time() - t_start), flush=True)


# ------------------------------------------------------------------------------------------------------------------ compose
def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and x is not None and x == x else "-"


def compose(a):
    paths = a.compose.split(",")
    M = [json.load(open(p)) for p in paths]
    P_ = M[0]; Cn = M[1] if len(M) > 1 else None
    hP = P_["head"]
    Ms = [(P_, "primary")] + ([(Cn, "step control")] if Cn is not None else [])
    L = ["## S7 part 2: the copy-mixing preparations with their rotated references, propagated on the stored history; D_det at the collar (engine_s7_twocopy_v1.py; engine spec 30.5, 29.5 and 28.5 S7)", ""]
    for Mx, role in Ms:
        h = Mx["head"]
        L.append("- %s: %s (%s; r_s %g, tau %g, %d sites, m %g, g %g, ell %s, %d channels), the preparation record T/rs %.3f (%s; trapping zeros there: r_in %s, R_t %s; the stored E %s, the replayed total's solve %s), %d records to T/rs %.3f; the C-init re-prepared in %s passes (stored %s): max |N - N_cinit| %s, |v - v_cinit| %s, E0 %s (stored %s)." % (
            role, h["tag"], h["member_script"], h["rs"], h["tau"], h["nsites"], h["m"], h["g"], h["ell"], h["kappa_sea"], h["T0_over_rs"], h["T0_rule"],
            fmt(h["trapping_zeros_at_T0"]["r_in"], "%.4f"), fmt(h["trapping_zeros_at_T0"]["R_t"], "%.4f"), fmt(h["E_stored_at_T0"], "%.10f"), fmt(h["solve_at_T0"]["E_total"], "%.10f"),
            h["records"], h["T_end_over_rs"], h["cinit"]["passes"], h["cinit"]["stored_passes"], fmt(h["cinit"]["dN"], "%.2e"), fmt(h["cinit"]["dv"], "%.2e"),
            fmt(h["cinit"]["E0"], "%.10f"), fmt(h["cinit"]["stored_E0"], "%.10f")))
    wP = hP["window"]
    L += ["- the ell control: no r_s 32 ell 8 member on this lane's v17 set exists (C3set25's c25_rs32_A_R2_tau8_ell8_avg is an engine_collapse_v25 run stopped at T/rs 3.75, before T0); none runs (28.5's \"where it exists\"; 29.0 (1) reads ell histories re-solved per preparation as a record).",
          "- the window (30.5): %s, w %g sites, X %.4f (%s); chi > 0 below R %.2f, R_t %s; the dust's total occupation %.6f (N_B %s); the minimum %s." % (
              wP["form"], wP["width"], wP["X"], wP["rule"], wP["support_end"], fmt(wP["R_t"], "%.4f"), wP["dust_total"], fmt(wP["N_B"], "%.6f"),
              "lies inside the interval" if wP["interior_minimum"] else "lies at an end of the interval: the dust shell density has no interior minimum in [r_in, R_t] at this record (the profile below)"),
          "- the preparations: per channel kappa in %s the two copies Phi (+) Phi of the channel's propagated state at T0 (Phi = [sea, dust], f = 1 on the sea and the dust's occupations), Y_+- = R_+- (Phi (+) Phi), R_+- = [[c, +-s], [-+s, c]] per site on both components (c = cos(theta chi_X), s = sin(theta chi_X); part 1's layout and sign), C_+- = Y_+- f2 Y_+-^dagger; each preparation's reference rotated with it, P_+-(T0) = R_+- (P_0 (+) P_0) R_+-^dagger (30.5), P_0 the record's continued reference; theta requested %s, reduced to %.4f where the collar charge difference exceeds %g of E; the thetas run: %s." % (
              hP["kappas"], ", ".join("%.4f" % t for t in hP["thetas_requested"]), hP["theta_reduced"], hP["charge_tol"],
              "; ".join("kappa %s: %s" % (k_, ", ".join("%.4f" % t for t in v_)) for k_, v_ in hP["thetas_final"].items())),
          "- the propagator: the member's own record-to-record step, U_k = V_k exp(-i e_k tau) V_k^dagger from the eigendecomposition of h_m at the record's stored applied fields (engine_replay_v4's step; engine_collapse_v17's propagation), applied within each copy to Y_+ and Y_- apart (the two-copy block h_m (x) 1), from T0 to the run's end; every channel of the member replayed through the whole history by its rule (prepare_dust, continue_v17), the S7 channels' C_0(t), P(t) and P_bar(t) carried beside the preparations; each preparation's reference continued at every record by the model's rule on the two-copy block (continue_fast_info, else the block step, on h_m (x) 1's eigenbasis; the Weyl test and the subdivisions not applied to it, its gap undefined at a rotated reference) until it coincides with P(t) (+) P(t).",
          "- D_det: ||Delta C_D||_op, Delta C = C_+ - C_-, on the detector modes in both copies (F_D (x) 1), the lower bound on D_det = (1/2) ||rho_D^+ - rho_D^-||_1 (the number of the extremal mode); the single-copy identity Delta C = [[0, 2G], [-2G, 0]], G = B f A^dagger - A f B^dagger (A = U c Phi, B = U s Phi), gives 2 ||F_D^dagger G F_D||_op as the check; the collar's bound is ||Pi_c Delta C Pi_c||_op over every mode of the collar's sites in both copies; the escape ladder 2 ||G_{R >= r}||_op is the same bound for the region R >= r (its Frobenius form an upper bound at every record; the front at eps is the outermost r with the Frobenius bound above eps). Delta C involves no reference: the rotated references change the sources, not D_det.",
          "- the detector (BUILDER, labelled): J3''s probe packets by obs_raymap_solved_v5's construction (flow_chart_v4.branch_momenta on the outgoing branch of the exterior v = -sqrt(r_s/x) at the centre, sigma_omega = omega/6; flow_chart_v4.build_packet with the local positive spinor of m_p SZ + (kappa/x) SY + sin k SX, kappa the S7 channel's), omega/kappa_p 1/2, 1, 2 and m_p/kappa_p 0, 0.25 at kappa_p = 1/(2 r_s) with the nominal r_s (the same packets on both step sizes, version 6's fixed-packet rule of 29.4), centred at the collar (the engine's collar is the chain's last node, r_collar %s; the sites beyond the record's M1 cut, R %d to %d), restricted to the collar's sites, renormalized and orthonormalized as a set (column-pivoted QR):" % (
              hP["r_collars"], hP["detector"][str(hP["kappas"][0])]["collar_sites"][0], hP["detector"][str(hP["kappas"][0])]["collar_sites"][1]), ""]
    L += ["| kappa | m_p/kappa_p | omega/kappa_p | omega | k_c | v_g | s_x (density) | norm retained in the collar |", "|---|---|---|---|---|---|---|---|"]
    for kap in hP["kappas"]:
        di = hP["detector"][str(kap)]
        for p_ in di["packets"]:
            if "skipped" in p_:
                L.append("| %d | %g | %g | %s | | | | |" % (kap, p_["mp_over_kappa"], p_["omega_over_kappa"], p_["skipped"]))
            else:
                L.append("| %d | %g | %g | %.6f | %.5f | %.4f | %.1f | %.3e |" % (kap, p_["mp_over_kappa"], p_["omega_over_kappa"], p_["omega"], p_["k_c"], p_["v_g"], p_["s_x_density"], p_["retained_norm_in_collar"]))
    for kap in hP["kappas"]:
        di = hP["detector"][str(kap)]
        L.append("")
        L.append("- kappa %d: x_D %.2f (site index %d), v_ext %.5f, kappa/x_D %.3e; the restricted packets' Gram eigenvalues %s; %d orthonormal detector modes per copy (orthonormality %.1e)." % (
            kap, di["x_D"], di["x_index"], di["v_ext"], di["a"], ", ".join("%.3e" % x for x in di["gram_eigenvalues"]), di["n_detector"], di["orthonormality"]))
    # the dust profile
    L += ["", "### The dust shell density at T0 (%s, T/rs %.3f): the window's placement (30.5)" % (hP["tag"], hP["T0_over_rs"]), "",
          "| R | dust per shell (sum_kappa w f |phi|^2) | dust energy per shell (replayed) | stored dust energy per shell (eps_tot - eps_pairs) | %s | chi_X |" % " | ".join("kappa %d's dust (unit weight)" % kq for kq in hP["kappas"]),
          "|---|---|---|---|%s---|" % ("---|" * len(hP["kappas"]))]
    for p_ in wP["profile"]:
        L.append("| %.0f | %.6e | %.6e | %.6e | %s | %.6f |" % (p_["R"], p_["dust_per_shell"], p_["dust_energy_per_shell"], p_["stored_dust_energy"],
                 " | ".join(fmt(p_.get("dust_k%d" % kq), "%.4e") for kq in hP["kappas"]), p_["chi"]))
    # the labels
    L += ["", "### The preparations' label at T0 (30.5): the remaining normal-ordered source difference and the collar charge difference", "",
          "| member | kappa | theta | run | source difference with P_+- = R (P_0 (+) P_0) R^dagger: largest (eps; J) relative to max |2 (C_0 - P_0)| | largest absolute (eps; J) | net sum (eps; J) | shells above 1e-15 of the raw density's largest value, part 1's convention (first, last) | away from the transition (> w) | with the unrotated P_0 (+) P_0 (eps; J) | + against - (eps; J) | collar charge difference dE/E (rotated; unrotated) | solved max |dN|, |dv| against the unreplaced total | + against -: |dN|, |dv|, dE/E |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        h = Mx["head"]
        for lab in h["labels"]:
            ro, un, sv = lab["rotated"], lab["unrotated"], lab["solved"]
            run_ = ("yes" if (lab["theta"] in [float(x) for x in h["thetas_final"][str(lab["kappa"])]]) else "no (dE/E above %g: reduced to %.4f)" % (h["charge_tol"], h["theta_reduced"]))
            if not lab.get("requested", True):
                run_ = "yes (reduced from %s)" % ", ".join("%.4f" % t for t in lab.get("reduced_from", []))
            sh = ("%d (%s)" % (ro["shells_above_1e-15"], ", ".join("%.0f" % x for x in ro["shells_first_last"]))) if ro["shells_first_last"] else "0"
            L.append("| %s | %d | %.4f | %s | %.3e; %.3e | %.3e; %.3e | %+.3e; %+.3e | %s | %.1e; %.1e | %.3e; %.3e | %.1e; %.1e | %+.3e; %+.3e | %.2e, %.2e | %.1e, %.1e, %.1e |" % (
                h["tag"], lab["kappa"], lab["theta"], run_, ro["eps"], ro["J"], ro["eps_abs"], ro["J_abs"], ro["net_eps"], ro["net_J"], sh, ro["far_eps"], ro["far_J"],
                un["eps"], un["J"], lab["pm_eps"], lab["pm_J"], sv["dE_over_E"], sv["dE_over_E_unrotated"], sv["N_vs_total"], sv["v_vs_total"], sv["N_pm"], sv["v_pm"], sv["dE_pm_over_E"]))
    # the references after T0
    L += ["", "### Each preparation's reference after T0 (the model's continuation on the two-copy block)", ""]
    for Mx, role in Ms:
        for rf in Mx.get("references", []):
            tr_ = rf["trace"]
            first = tr_[0] if tr_ else None
            L.append("- %s, kappa %d, theta %.4f: %s; at the first record after T0 (T/rs %s) the modes %s, %s and the projector distances to P (+) P %s, %s." % (
                Mx["head"]["tag"], rf["kappa"], rf["theta"],
                ("coincides with P(t) (+) P(t) from T/rs %.3f on" % rf["snapped_at_T_over_rs"]) if rf["snapped_at_T_over_rs"] is not None else "never coincides with P(t) (+) P(t) within the run",
                fmt(first["T_over_rs"], "%.3f") if first else "-", first["Pi_p"]["mode"] if first else "-", first["Pi_m"]["mode"] if first else "-",
                fmt(first["Pi_p"]["distance_to_doubled"], "%.2e") if first else "-", fmt(first["Pi_m"]["distance_to_doubled"], "%.2e") if first else "-"))
    # the audit
    L += ["", "### The audit: every source the script computes, and the reference it subtracted", "",
          "| member | source computed | role | reference received | calls | verdict |", "|---|---|---|---|---|---|"]
    ok_all = True
    for Mx, role in Ms:
        for r in Mx["audit"]["rows"]:
            L.append("| %s | %s | %s | %s | %d | %s |" % (Mx["head"]["tag"], r["label"], r["role"], r["reference"], r["calls"], r["verdict"]))
        ok_all = ok_all and Mx["audit"]["pass"]
    L.append("")
    L.append("- the audit's verdict: %s (every call with the role \"source\" subtracted the record's continued reference P, or for a preparation its own reference P_+-; the contrasts, which source nothing, subtracted P_bar, the unrotated P (+) P at T0 or the reference held rotated)." % ("PASS" if ok_all else "FAIL"))
    for Mx, role in Ms:
        rp = Mx["replay"]
        nacc = sum(1 for r in rp if r["accepted"])
        mx = lambda key: max(r[key] for r in rp)
        mn = lambda key: min(r[key] for r in rp)
        L.append("- %s, the numeric audit over its %d records (T/rs %.3f to %.3f): the replayed C - P against the stored eps_tot, J_tot: at most %.2e, %.2e relative (acceptance %g; %d of %d records within it); the contrast C - P_bar against the stored totals: %.2e to %.2e (eps), %.2e to %.2e (J), and against the stored dust part eps_tot - eps_pairs at most %.2e, %.2e (C - P_bar is the dust's source alone: the pairs' part is what P_bar would drop); the propagated P_bar's columns against the sea's: at most %.1e." % (
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
            L.append("| %s | %d | %.4f | %.3e | %.3e | %.3e (%.3f) | %.1e; %s | %.1e | %.3e | %.3e | %.3e | %.3e |" % (
                Mx["head"]["tag"], c0["kappa"], c0["theta"], ser[0][1]["D_det"], ser[-1][1]["D_det"], dmax[1]["D_det"], dmax[0],
                max(abs(q[1]["D_det"] - q[1]["D_det_single_copy"]) for q in ser), ("%.1e" % max(big)) if big else "none above 1e-12",
                max(q[1]["copy_structure"] for q in ser), ser[0][1]["collar_bound"], ser[-1][1]["collar_bound"],
                max(q[1]["collar_bound"] for q in ser), max(max(q[1]["per_packet"]) for q in ser)))
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        L.append("")
        L.append("- %s: the propagator's rounding leakage max |(U_k)_{collar, window}| (kappa %d) per record %.1e to %.1e, summed over the %d steps from T0 %.1e: the floor D_det at the collar is read against (the exact element is below 1e-300)." % (
            Mx["head"]["tag"], Mx["head"]["kappas"][0], min(r["leakage"] for r in s7), max(r["leakage"] for r in s7), len(s7) - 1, s7[-1]["leakage_sum"]))

    def shown(r, s7):
        """the md's stride: T0, the end, and every --md-every-rs r_s from T0 (the json holds every record)."""
        x = (r["T_over_rs"] - s7[0]["T_over_rs"]) / a.md_every_rs
        return r is s7[0] or r is s7[-1] or abs(x - round(x)) < 1e-9

    # the escape ladder
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        lad = [str(r) for r in h["ladder_radii"]]
        L += ["", "### The escape ladder, %s: 2 ||G_{R >= r}||_op (exact; = ||Delta C restricted to R >= r||_op, the largest ||Delta C_D|| of any detector beyond r) and the front (the outermost r whose Frobenius bound exceeds eps); every %g r_s for kappa %d, T0 and the end for the others" % (
            h["tag"], a.md_every_rs, h["kappas"][0]), "",
              "| kappa | theta | T/rs | %s | front 1e-12 | front 1e-9 | front 1e-6 | front 1e-3 |" % " | ".join("r = %s" % r for r in lad),
              "|---|---|---|%s---|---|---|---|" % ("---|" * len(lad))]
        for i_, c0 in enumerate(s7[0]["configs"]):
            for r in s7:
                q = r["configs"][i_]
                if q["ladder_op"] is None:
                    continue
                if not shown(r, s7) or (q["kappa"] != h["kappas"][0] and r is not s7[0] and r is not s7[-1]):
                    continue
                L.append("| %d | %.4f | %.3f | %s | %.0f | %.0f | %.0f | %.0f |" % (q["kappa"], q["theta"], r["T_over_rs"], " | ".join(fmt(q["ladder_op"][x]) for x in lad),
                         q["front"]["1e-12"], q["front"]["1e-09"], q["front"]["1e-06"], q["front"]["0.001"]))
    # the sources per record
    L += ["", "### The preparations' sources after T0 (copy-summed, against 2 (C_0(t) - P(t)) of the unrotated state)", "",
          "| member | kappa | theta | T/rs | own reference P_+-(t) (eps; J) | its shells above 1e-15 of the raw density | held rotated R (P (+) P) R^dagger (eps; J) | + against - (eps; J) | the C - P_bar contrast | Tr C_+ - 2 Tr C_0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            for r in s7:
                if not shown(r, s7):
                    continue
                q = r["configs"][i_]
                L.append("| %s | %d | %.4f | %.3f | %.3e; %.3e | %d | %.3e; %.3e | %.1e; %.1e | %.3e | %+.1e |" % (Mx["head"]["tag"], q["kappa"], q["theta"], r["T_over_rs"],
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
                L.append("| %s | %d | %.4f | %.3f | %.1e, %.1e, %.1e | %.2e, %.2e, %+.2e | %.8f |" % (Mx["head"]["tag"], q["kappa"], q["theta"], r["T_over_rs"],
                         sv["N_pm"], sv["v_pm"], sv["E_pm_rel"], sv["N_vs_total"], sv["v_vs_total"], sv["E_vs_total_rel"], sv["E_total"]))
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        for kap in Mx["head"]["kappas"]:
            L.append("- %s, kappa %d: ||P - P_bar||_F %.4f at T0 and %.4f at the end; Tr P %.6f, Tr P_bar %.6f at the end." % (
                Mx["head"]["tag"], kap, s7[0]["P_minus_Pbar_F_k%d" % kap], s7[-1]["P_minus_Pbar_F_k%d" % kap], s7[-1]["trP_k%d" % kap], s7[-1]["trPbar_k%d" % kap]))
    # the per-record table (the primary)
    if P_["s7"]:
        r0 = str(hP["ladder_radii"][0])
        L += ["", "### Per record, the primary member, every configuration: D_det; the collar bound; the ladder's Frobenius bound at r = %s; the front at 1e-6 (every %g r_s; every record in the json)" % (r0, a.md_every_rs / 2), "",
              "| T/rs | %s |" % " | ".join("D_det, collar, ladder, front: k%d th %.4f" % (q["kappa"], q["theta"]) for q in P_["s7"][0]["configs"]), "|---|%s" % ("---|" * len(P_["s7"][0]["configs"]))]
        for r in P_["s7"]:
            x_ = (r["T_over_rs"] - P_["s7"][0]["T_over_rs"]) / (a.md_every_rs / 2)
            if not (r is P_["s7"][0] or r is P_["s7"][-1] or abs(x_ - round(x_)) < 1e-9):
                continue
            L.append("| %.3f | %s |" % (r["T_over_rs"], " | ".join("%.2e, %.2e, %.2e, %.0f" % (q["D_det"], q["collar_bound"], q["ladder_fro"][r0], q["front"]["1e-06"]) for q in r["configs"])))
    # the step control
    if Cn is not None and Cn["s7"] and P_["s7"]:
        hC = Cn["head"]
        byT = {round(r["T_over_rs"], 6): r for r in Cn["s7"]}
        lad = [str(r) for r in hP["ladder_radii"]]
        L += ["", "### The step control: %s (tau %g) against %s (tau %g) at the times both hold" % (hC["tag"], hC["tau"], hP["tag"], hP["tau"]), "",
              "| kappa | theta | T/rs | D_det tau %g | D_det tau %g | collar bound tau %g, tau %g | 2 ||G_{R >= %s}||_F tau %g, tau %g (rel. diff.) | front 1e-6 tau %g, tau %g |" % (
                  hP["tau"], hC["tau"], hP["tau"], hC["tau"], lad[0], hP["tau"], hC["tau"], hP["tau"], hC["tau"]), "|---|---|---|---|---|---|---|---|"]
        worst = {}
        for r in P_["s7"]:
            rc = byT.get(round(r["T_over_rs"], 6))
            if rc is None:
                continue
            for i_, q in enumerate(r["configs"]):
                qc = next((x for x in rc["configs"] if x["kappa"] == q["kappa"] and abs(x["theta"] - q["theta"]) < 1e-12), None)
                if qc is None:
                    continue
                fr = abs(q["ladder_fro"][lad[0]] - qc["ladder_fro"][lad[0]]) / max(q["ladder_fro"][lad[0]], 1e-300)
                worst[i_] = max(worst.get(i_, 0.0), fr)
                if shown(r, P_["s7"]):
                    L.append("| %d | %.4f | %.3f | %.3e | %.3e | %.2e, %.2e | %.3e, %.3e (%.2e) | %.0f, %.0f |" % (q["kappa"], q["theta"], r["T_over_rs"], q["D_det"], qc["D_det"],
                             q["collar_bound"], qc["collar_bound"], q["ladder_fro"][lad[0]], qc["ladder_fro"][lad[0]], fr, q["front"]["1e-06"], qc["front"]["1e-06"]))
        L.append("")
        L.append("- the largest relative difference of the r = %s Frobenius bound between the step sizes over the matched times, per configuration: %s." % (
            lad[0], "; ".join("k%d th %.4f: %.2e" % (P_["s7"][0]["configs"][i_]["kappa"], P_["s7"][0]["configs"][i_]["theta"], v_) for i_, v_ in sorted(worst.items()))))
    # what the specification left open
    pk0 = [p_ for p_ in hP["detector"][str(hP["kappas"][0])]["packets"] if "m_p" in p_]
    L += ["", "### Left open by the specification (the builder's readings, labelled)", "",
          "- the detector's mass: J3''s probe set is light (m_p at most %.4f, omega %.4f to %.4f), the model's fermion has m %g, so the packets are detector modes in the S7 channel's space (an occupation measurement of each mode), not outgoing modes of the propagating field, whose threshold at the collar is about m; a detector of the model's own outgoing quanta needs omega above m (C-capture's omega/m 1.1 to 2, 28.3) and is not in this version." % (
              max(p_["m_p"] for p_ in pk0), min(p_["omega"] for p_ in pk0), max(p_["omega"] for p_ in pk0), hP["m"]),
          "- the collar's extent: the engine's collar is the chain's last node (r_collar %s) and M1's cut sits %d sites inside it, so the collar's sites are R %d to %d; J3''s packets (density widths %.0f to %.0f sites) restricted there span %d dimensions per copy (the Gram eigenvalues above); the escape ladder gives the same bound for every region R >= r, so a wider collar reads off its row." % (
              hP["r_collars"], hP["nsites"] - hP["m1_cuts"][0], hP["m1_cuts"][0] + 1, hP["nsites"], min(p_["s_x_density"] for p_ in pk0), max(p_["s_x_density"] for p_ in pk0),
              hP["detector"][str(hP["kappas"][0])]["n_detector"]),
          "- the window's placement: %s; the transition's width w %g (BUILDER: the two-step operator's hop range is 2 sites, the ramp spans two hops) is not fixed by 30.5." % (
              ("the dust shell density's minimum over [r_in, R_t] lies at the interval's end, so the ramp's support ends at R %.2f, beyond R_t %.4f" % (wP["support_end"], wP["R_t"])) if (wP["interior_minimum"] is False and wP["R_t"] is not None and wP["support_end"] > wP["R_t"])
              else "the transition sits at the dust's minimum inside [r_in, R_t]", wP["width"]),
          "- the references after T0: the model's continuation of each preparation's rotated reference on h_m (x) 1 is spanned by that operator's eigenvectors, which carry no site-dependent copy rotation, so it returns to P(t) (+) P(t) (the references' section above gives where); from there the preparations' normal-ordered sources differ from C_0 (+) C_0's by the rotated state's bond terms across the transition (the sources' table: own reference), the reference held rotated the contrast; the two preparations equal each other throughout, which is what the common history needs, and the stored history is the run's own (C_0's), not each preparation's self-consistent one (the solved table gives the fields' difference).",
          "- P_bar's P_in: the run's initial reference (the C-init negative-energy set at T = 0), so P_bar(t) = U(t, 0) P_in U(t, 0)^dagger is the propagated initial sea (R1's reference); it is never a source here.",
          ("- the step control: the tau %g member holds %s at T/rs %.3f, so its window and thetas are the tau %g member's (--match-json), its own C_0, P_0 and label at that time." % (
              Cn["head"]["tau"], ("no trapping zeros" if Cn["head"]["trapping_zeros_at_T0"]["R_t"] is None else "trapping zeros"), hP["T0_over_rs"], hP["tau"])) if Cn else "- no step control member was composed.",
          "- the massless shell: S4's member does not exist as a stored history yet, so the same computation on it (29.5's \"the same on the massless shell when it exists\") is not here; the r_s 32 ell 8 control does not exist (above)."]
    L += ["", "- every number above is printed by the script (engine_s7_twocopy_v1.py --compose); the per-record values of every column are in the json."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_s7_twocopy_v1.py", "members": M, "md": os.path.basename(a.md)}, open(a.out_json, "w"), separators=(",", ":"), default=float)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default=None, help="a stored engine_collapse_v17 run: dir/tag")
    ap.add_argument("--T0-over-rs", dest="T0_over_rs", type=float, default=None)
    ap.add_argument("--windows", default=None, help="the window's centre X (sites), overriding the dust-minimum rule")
    ap.add_argument("--match-json", default=None, help="the primary member's json: its X, w and final thetas (the step control)")
    ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--thetas", default="0.7853981633974483,1.5707963267948966")
    ap.add_argument("--theta-reduced", type=float, default=0.39269908169872414, help="30.5: pi/8")
    ap.add_argument("--charge-tol", type=float, default=1e-6, help="30.5: the collar charge difference's bar, relative to E")
    ap.add_argument("--width", type=float, default=4.0, help="the C^2 ramp's full width w (sites)")
    ap.add_argument("--ladder-radii", default="32,64,128,256,512")
    ap.add_argument("--ladder-every-rs", type=float, default=1.0)
    ap.add_argument("--omega-over-kappa", default="0.5,1,2")
    ap.add_argument("--mp-over-kappa", default="0,0.25")
    ap.add_argument("--accept", type=float, default=1e-8)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--compose", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--md-every-rs", type=float, default=2.0, help="compose: the md tables' time stride (T0 and the end always shown)")
    a = ap.parse_args()
    if a.compose:
        compose(a)
    else:
        member(a)


if __name__ == "__main__":
    main()
