#!/usr/bin/env python
"""gate_dense_oracle_v3.py (busi lane, Computation C): gate_dense_oracle_v2b.py with criterion (5)'s archive replaced by a defined one and the
readouts of the fixture document added (busi/notes/C_oracle_archive_ruling_v1.md; C_fixture_definition_v1.md sections 7 and 10). Everything of
version 2b that is not named below is unchanged: the fixture, the states, the propagation, the references and criteria (1) to (4).
The archive. Version 2b admitted an exterior mode u, an eigenvector of the Hermitian exterior block h_ext, when Re(u^H i[h_ext, X] u) > 0; for an
eigenvector that number is e u^H X u - e u^H X u = 0 (the hard-wall block's modes are standing waves), so the 96 positive levels were split by
rounding of order 1e-14 (51, 52, 47, 52 columns in the lane's run; scipy 54/46/51/46 and torch's CPU eigh 44/48/46/49 on the same matrices) and no
second route could match S_esc. Version 3's archive is the positive spectral half of the exterior block, 1(h_ext > 0): the candidates at every
checkpoint are all eigenvectors of h_ext with eigenvalue > 0, embedded, projected against the archive (classical Gram-Schmidt twice), the block's SVD
and page_ops's drop rule as in 2b, no velocity or direction test; only span(A) enters, a spectral projector with the gap min |ee| (1.3e-2 to 1.7e-1
here), which every eigensolver returns to an angle of about 1e-14. It is the fixture's instance of the solve lane's corrected collector's column form
(every column inside one spectral half of the exterior block, no velocity admission; the velocity admission is web40g's second defect); on this
fixture nothing reaches R > 160 by t = 80, so S_esc is the vacuum's entanglement in the positive half, page_core's S_vac(archive), and the increment
above the reference's entropy is the propagation's reach.
Recorded, new in v3: per checkpoint and channel the archive's mode indices (positions of the positive levels in h_ext's ascending spectrum), their
eigenvalues, the half gap, the spectrum of A^H C A (a second route matches mode by mode), S_vac(archive) = sum H(s^2) over the singular values of
A^H ref, S_dense - S_vac, the negative half's entropy by the retained route (the T-image of the archive; equal to S_dense on a pure sea, so a free
check on the light channels; information), the compressed velocity V_+ = U_+^H i[h_ext, X] U_+ on the half (its bounds, gap at 0 and positive count:
the fixture document's candidate (B); information) and version 2b's test |Re diag V_+| at its maximum (the rounding it is); per step and channel
E_inv = Re Tr[h_m (C - P_ref)] by the dense trace (the coefficient route beside it, criterion (4) as 2b), the outward Killing-energy current
Phi_E(R_c) = Re Tr[(C - P_ref) J_E], J_E = (i/2)[h_m^2, Pi_{R > R_c}], and the outward number current Phi_N(R_c) = Re Tr[(C - P_ref) J_N],
J_N = i[h_m, Pi_{R > R_c}], at the collars --collars (160 first: the archive's cut and the ruled collar; 80 = R_b(0): information), the operators built
from the formula and differenced against page_ops_v5.current_out_csr and number_current_csr at step 0; the weighted sums 16 (heavy), 16 kappa 1 +
32 kappa 2 (sourcing light) and 48 kappa 3 (guard) apart; the environment (hostname, device and its name, torch, numpy, scipy, python and platform
versions, threads, niceness, torch's build configuration).
Criteria: (1) to (4) as 2b; (5) on the new archive as 2b (routes 1e-10, Z/floor <= 1, Gram 1e-12, columns <= 192) plus the half gap >= 1e-6 and
exactly 96 positive levels (the T symmetry e -> -e of every channel).
Outputs in --out: gate_dense_oracle_v3_result.json; gate_dense_oracle_v3_table.txt (one line per checkpoint and channel, tab-separated, every float at
17 significant digits); the table is also printed at the end of the log.
Usage: python gate_dense_oracle_v3.py [--gpu 0] [--threads 16] [--collars 160,80] [--steps 40] [--out gateD_v3]; prints PASS or the failing items."""
import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_twofield_v3 as E36  # noqa: E402
import engine_collapse_v5 as C5  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402
import engine_ops_v34_cert as CERT  # noqa: E402
try:
    import page_ops_v5 as PO  # noqa: E402  (the collar operators' second form, differenced at step 0; information)
except Exception:  # pragma: no cover
    PO = None

BK = E36.BK
EPS = float(np.finfo(float).eps)
G17 = "%.17g"


def fields_at(R, t, g2E=48.0, Rb0=80.0, Rb1=32.0, t_end=80.0):
    Rb = Rb0 + (Rb1 - Rb0) * min(max(t / t_end, 0.0), 1.0)
    N = 1.0 - 0.25 * np.clip(1.0 - (R / Rb) ** 2, 0.0, None)
    v = -np.sqrt(g2E / R)
    inside = R < Rb
    v[inside] = -np.sqrt(g2E / Rb ** 3) * R[inside]
    return N, v


def binary_entropy(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    q = x[(x > 1e-300) & (x < 1.0 - 1e-16)]
    return float(np.sum(-q * np.log(q) - (1.0 - q) * np.log1p(-q)))


def f_(x, f="%.2e"):
    """a number in format f; inf, None and nan printed as they are."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return (f % xf) if np.isfinite(xf) else str(x)


def jsonable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)


def environment(torch, device):
    """v3: the device and the library versions of the run (section 10 item 3 of the fixture document)."""
    env = {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split("\n")[0], "numpy": np.__version__,
           "scipy": scipy.__version__, "torch": torch.__version__, "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
           "device": str(device), "torch_threads": int(torch.get_num_threads()), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}
    try:
        env["nice"] = os.nice(0)
    except Exception:
        env["nice"] = None
    try:
        if getattr(device, "type", "cpu") == "cuda":
            env["device_name"] = torch.cuda.get_device_name(device)
        else:
            name = platform.processor() or ""
            try:
                with open("/proc/cpuinfo") as fh:
                    for line in fh:
                        if line.lower().startswith("model name"):
                            name = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
            env["device_name"] = name or "cpu"
    except Exception:
        env["device_name"] = None
    try:
        env["torch_mkl"] = bool(torch.backends.mkl.is_available()); env["torch_openmp"] = bool(torch.backends.openmp.is_available())
    except Exception:
        pass
    try:
        env["torch_config"] = torch.__config__.show()
    except Exception:
        env["torch_config"] = None
    try:
        env["numpy_config"] = jsonable(np.show_config(mode="dicts"))
    except Exception:
        env["numpy_config"] = None
    return env


def collar_operators(h_csr, R, R_c):
    """v3: the outward Killing-energy current J_E = (i/2)[h^2, Pi_{R > R_c}] and the outward number current J_N = i[h, Pi_{R > R_c}] through the
    cut at R_c as host CSR matrices, from the formula (page_ops_v5.current_out_csr and number_current_csr re-derived; both Hermitian):
    (J_E)_ab = (i/2) (h^2)_ab (pi_b - pi_a), (J_N)_ab = i h_ab (pi_b - pi_a), pi the indicator of R > R_c on both components. h has range 2 sites,
    h^2 range 4, so both live on the sites about the cut. d<H_out>/dt = Tr[C J_E] with H_out = (1/2){Pi_out, h}: positive outward."""
    pi = np.repeat((np.asarray(R, float) > R_c).astype(float), 2)
    h = sp.csr_matrix(h_csr)
    h2 = (h @ h).tocoo()
    dE = 0.5j * h2.data * (pi[h2.col] - pi[h2.row]); kE = np.abs(dE) > 0
    JE = sp.csr_matrix((dE[kE], (h2.row[kE], h2.col[kE])), shape=h2.shape)
    co = h.tocoo()
    dN = 1j * co.data * (pi[co.col] - pi[co.row]); kN = np.abs(dN) > 0
    JN = sp.csr_matrix((dN[kN], (co.row[kN], co.col[kN])), shape=co.shape)
    return JE, JN


def tr_real(D, J):
    """Re Tr[D J] for dense device tensors: sum_ab D_ab J_ba."""
    return float((D * J.transpose(0, 1)).sum().real)


def weighted_sums(rows_step, collars, ck):
    """v3: the 16 kappa weighted sums of a step: the heavy channel; the sourcing light channels (kappa 1 and 2); the guard kappa 3 apart."""
    groups = ("heavy_w", "light_sourcing_w", "light_guard_w")
    out = {"E_inv": {g: 0.0 for g in groups}}
    for Rc_ in collars:
        out["Phi_E_" + ck(Rc_)] = {g: 0.0 for g in groups}
        out["Phi_N_" + ck(Rc_)] = {g: 0.0 for g in groups}
    for r_ in rows_step:
        grp = "heavy_w" if r_["label"] == "heavy" else ("light_guard_w" if r_["label"] == "light3" else "light_sourcing_w")
        out["E_inv"][grp] += r_["w"] * r_["E_inv"]
        for Rc_ in collars:
            out["Phi_E_" + ck(Rc_)][grp] += r_["w"] * r_["collar"][ck(Rc_)]["Phi_E"]
            out["Phi_N_" + ck(Rc_)][grp] += r_["w"] * r_["collar"][ck(Rc_)]["Phi_N"]
    for k_ in list(out.keys()):
        d_ = out[k_]
        d_["guard_over_sourcing"] = (d_["light_guard_w"] / d_["light_sourcing_w"]) if d_["light_sourcing_w"] != 0.0 else None
    return out


def fine_continuation(ch, ref_prev, f_prev, f_cur, e_cur, U_cur, n_sea, prev_mask, nsub=64):
    """version 3's maximum-overlap assignment at nsub uniform sub-steps of the panel (v1b's "truth"). Information: at an avoided crossing
    it is the diabatic branch when the sub-steps' level motion exceeds the avoided gap and the adiabatic branch when it does not."""
    r = ref_prev; pm = prev_mask
    for k in range(1, nsub + 1):
        if k == nsub:
            e_s, U_s = e_cur, U_cur
        else:
            s = k / nsub
            e_s, U_s = E36.eigT(E36.clocked_gpu(ch, f_prev[0] + s * (f_cur[0] - f_prev[0]), f_prev[1] + s * (f_cur[1] - f_prev[1])))
        r, inf_, pm = OPS3.assign_max_overlap_v3(r, e_s, U_s, n_sea, prev_mask=pm)
    return r, np.asarray(pm, bool)


def resid(V, W):
    """||(1 - V V^dagger) W||_F for V with orthonormal columns."""
    return float((W - V @ (V.conj().T @ W)).norm().item())


def cert_oracle(ch, f_prev, f_cur, cert, S_host, n_sea, ns=65):
    """criterion (2): the certificate's claims against dense eigenvalues of K(s) = (1 - s) K0 + s K1 (as 2b)."""
    K0 = (E36.clocked_host_sparse(ch, f_prev[0], f_prev[1]) - cert["mu0"] * S_host).tocsr()
    K1 = (E36.clocked_host_sparse(ch, f_cur[0], f_cur[1]) - cert["mu1"] * S_host).tocsr()
    GK = max(OPS3.gershgorin(K0), OPS3.gershgorin(K1))
    K0d = K0.toarray(); dKd = K1.toarray() - K0d
    nG = float(cert["norm_dK_G"])
    F = float(cert["rounding_F"]) if cert.get("rounding_F") is not None else 64.0 * EPS * GK * GK
    out = {"pieces": 0, "norm_dK_2_minus_G": float(np.max(np.abs(np.linalg.eigvalsh(dKd)))) - nG, "rounding_F": F, "norm_K_G": GK,
           "start_margin_over_tol": np.inf, "cholesky_margin_d2_over_F": np.inf, "cholesky_margin_d": np.inf, "soundness_min_d2_over_F": np.inf,
           "tiling_defect": 0.0, "grid_weyl_margin": np.inf, "grid_count_defect": 0, "grid_gap_min": np.inf}
    pieces = [tuple(float(x) for x in p) for p in (cert.get("piece_list") or [])]
    out["pieces"] = len(pieces)
    if not pieces or nG <= 0.0:
        return out
    rig = lambda s0, d: d if s0 == 0.0 else math.sqrt(max(d * d - F, 0.0))
    td = max(abs(pieces[0][0]), abs(pieces[-1][1] - 1.0))
    for i, (s0, s1, d) in enumerate(pieces):
        td = max(td, (s1 - s0) - rig(s0, d) / nG)
        if i:
            td = max(td, abs(s0 - pieces[i - 1][1]))
        g = float(np.min(np.abs(np.linalg.eigvalsh(K0d + s0 * dKd))))
        if s0 == 0.0:
            out["start_margin_over_tol"] = min(out["start_margin_over_tol"], (g - d) / (16.0 * EPS * GK))
        else:
            out["cholesky_margin_d2_over_F"] = min(out["cholesky_margin_d2_over_F"], (g * g - d * d) / F)
            out["cholesky_margin_d"] = min(out["cholesky_margin_d"], g - d)
            out["soundness_min_d2_over_F"] = min(out["soundness_min_d2_over_F"], d * d / F)
    out["tiling_defect"] = float(td)
    starts = np.array([p[0] for p in pieces], float)
    for s in np.linspace(0.0, 1.0, ns):
        ev = np.linalg.eigvalsh(K0d + s * dKd); g = float(np.min(np.abs(ev)))
        out["grid_gap_min"] = min(out["grid_gap_min"], g)
        if int(np.sum(ev < 0.0)) != n_sea:
            out["grid_count_defect"] += 1
        k = max(int(np.searchsorted(starts, s, side="right")) - 1, 0)
        s0, s1, d = pieces[k]
        out["grid_weyl_margin"] = min(out["grid_weyl_margin"], g - max(0.0, rig(s0, d) - (s - s0) * nG))
    return out


ARCHIVE_DEFINITION = ("1(h_ext > 0): h_ext the 192 x 192 principal submatrix of h_m(t_k) on the rows of the sites with R > R_cut (both components); "
                      "the candidates at every checkpoint are all eigenvectors of h_ext with eigenvalue > 0, embedded (zero outside the exterior rows), "
                      "projected against the archive by classical Gram-Schmidt twice, the block's SVD taken and its Gram eigenvalues s^2 passed through "
                      "the drop rule g > max(1e-12, 1e-14 g_max), the kept left singular vectors appended; no velocity or direction test. Only span(A) "
                      "enters; the half gap min |eigenvalue| (criterion >= 1e-6) makes the span independent of rounding. "
                      "busi/notes/C_oracle_archive_ruling_v1.md.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--nsites", type=int, default=256); ap.add_argument("--out", default="gateD_v3"); ap.add_argument("--R-cut", type=float, default=160.0)
    ap.add_argument("--every", type=int, default=4); ap.add_argument("--nsub-fine", type=int, default=64)
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads; set OMP_NUM_THREADS and MKL_NUM_THREADS in the shell for numpy's LAPACK")
    ap.add_argument("--collars", default="160,80", help="the cuts R_c of the collar currents; the first is the ruled collar (the archive's cut)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C5.backend(a.gpu)
    E36.tm_reset(); E36.tml_reset_v36()
    torch = BK["torch"]
    if a.threads:
        torch.set_num_threads(int(a.threads))
    collars = [float(x) for x in a.collars.split(",") if x.strip()]
    ck = lambda Rc_: "R%g" % Rc_
    n = a.nsites
    args = SimpleNamespace(light_transport="banded", light_tol_block=1e-8, light_sigma_tol=1e-6, light_d_min=1e-7, light_max_pieces=4096, tau=a.tau, c_LZ=1.0,
                           weyl_refine_max=10, nsites=n)
    heavy = C5.Chan(1, 0.2, n)
    lights = [E36.LightChan(k, n) for k in (1, 2, 3)]
    R = heavy.R
    sgn_dev, S_host = E36.light_S_v36(n)
    chans = [("heavy", heavy)] + [("light", c) for c in lights]
    fails = []; notes = {}
    t_gate = time.time()
    # ---- (1) the symmetry at the fixture's fields and the operator against vacflow's (to rounding), as 2b
    N0, v0 = fields_at(R, 0.0)
    for c in lights:
        Hm = E36.clocked_host_sparse(c, N0, v0)
        comm = (Hm @ S_host - S_host @ Hm).tocsr(); re = Hm.real
        cm = float(abs(comm).max()) if comm.nnz else 0.0; rm = float(abs(re).max()) if re.nnz else 0.0
        oc = c.operator_check
        tol_op = 4.0 * EPS * c.kappa
        notes["symmetry_k%d" % c.kappa] = {"commutator_max": cm, "real_part_max": rm, "operator_check": oc, "operator_tolerance": tol_op}
        if cm > 1e-14 or rm > 1e-14 or max(oc["h_max_abs_diff_vs_vacflow"], oc["Q_max_abs_diff_vs_vacflow"]) > tol_op:
            fails.append("(1) symmetry or operator check failed on kappa %d: %s" % (c.kappa, notes["symmetry_k%d" % c.kappa]))
    print("(1) [h_m, S] and Re h_m at m = 0, the operator against vacflow's within 4 eps kappa: %s" % json.dumps(notes, default=float), flush=True)
    # ---- the states at t = 0, as 2b
    st = {}
    e, U = E36.eigT(E36.clocked_gpu(heavy, N0, v0))
    env = environment(torch, U.device)
    print("environment: %s" % json.dumps({k: v for k, v in env.items() if k not in ("torch_config", "numpy_config")}, default=str), flush=True)
    neg = e < 0; kneg = int(neg.sum()); en = e.cpu().numpy()
    pos_idx = torch.nonzero(~neg).ravel()[:6]
    dust = U[:, pos_idx].clone()
    st["heavy"] = {"sea": U[:, neg].clone(), "dust": dust, "f": np.ones(6), "ref": U[:, neg].clone(), "n_sea": kneg,
                   "window": [{"ref": U[:, neg].clone(), "f": (N0.copy(), v0.copy()), "gap": float(en[kneg] - en[kneg - 1]), "weyl_gap": float(en[kneg] - en[kneg - 1]), "mask": neg.cpu().numpy()}]}
    stL, headL = E36.light_init_v36(args, lights, N0, v0, sgn_dev, S_host)
    for c in lights:
        st[("light", c.kappa)] = stL[c.key]
        st[("light3", c.kappa)] = {"ref": stL[c.key]["ref"].clone(), "window": [dict(stL[c.key]["window"][0])]}
        st[("light34", c.kappa)] = {"ref": stL[c.key]["ref"].clone(), "window": [dict(stL[c.key]["window"][0])]}
    print("the light seas at t = 0: %s" % json.dumps(headL, default=float), flush=True)
    archive = {}
    worst = {"cert_dK_2_minus_G": -np.inf, "cert_start_margin_over_tol": np.inf, "cert_cholesky_margin_d2_over_F": np.inf, "cert_cholesky_margin_d": np.inf,
             "cert_soundness_min_d2_over_F": np.inf, "cert_tiling_defect": 0.0, "cert_grid_weyl_margin": np.inf, "cert_grid_count_defect": 0, "cert_panels": 0,
             "cert_pieces": 0, "conv_mask_diff": 0, "conv_resid": 0.0, "fine_vs_conv_diff": 0, "fine_vs_conv_panels": 0, "src": 0.0, "einv": 0.0, "sesc": 0.0,
             "z": 0.0, "z_over_floor": 0.0, "purity_1": 0.0, "purity_2": 0.0, "z_purified": 0.0, "sesc_purity_shift": 0.0, "gram": 0.0, "archive_over_dim": 0,
             "heavy_fine_vs_v3_diff": 0,
             # v3
             "half_gap_min": np.inf, "vexp_abs_max": 0.0, "increment_abs": 0.0, "t_symmetry_light": 0.0, "collar_vs_page_ops": (0.0 if PO is not None else None)}
    panels = []; sesc_rows = []; einv_rows = []; weighted_rows = []; first_trapped = None
    first_pocket_conv = {}; first_fallback = {}; certified_panels = {c.kappa: 0 for c in lights}; cut_crossings = {c.kappa: 0 for c in lights}
    Rc = a.R_cut
    ext = np.asarray(R, float) > Rc
    ext_rows = np.repeat(ext, 2); n_ext2 = int(2 * ext.sum())
    for step in range(a.steps + 1):
        t = step * a.tau
        N, v = fields_at(R, t)
        trapped = bool(np.any(N - np.abs(v) < 0.0))
        if trapped and first_trapped is None:
            first_trapped = step
        f_cur = (N, v)
        eigs = {}; step_rows = []
        for name, c in chans:
            key = "heavy" if name == "heavy" else ("light", c.kappa)
            lab = "heavy" if name == "heavy" else "light%d" % c.kappa
            e, U = E36.eigT(E36.clocked_gpu(c, N, v))
            s_ = st[key]
            if step > 0:
                prev = s_["window"][-1]
                if name == "heavy":
                    ref, info, mask = OPS3.continue_sea_v3(c, s_["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse)
                    mask = np.asarray(mask, bool)
                    r_f, m_f = fine_continuation(c, prev["ref"], prev["f"], f_cur, e, U, s_["n_sea"], prev.get("mask"), a.nsub_fine)
                    worst["heavy_fine_vs_v3_diff"] = max(worst["heavy_fine_vs_v3_diff"], int(np.sum(mask != m_f)))
                    s_["window"] = [{"ref": ref, "f": (N.copy(), v.copy()), "gap": info.get("gap_between_blocks"), "weyl_gap": info.get("weyl_gap"), "mask": mask}]
                else:
                    ref, info, mask = E36.light_continue_v36(args, c, s_["window"], f_cur, e, U, s_["n_sea"], sgn_dev, S_host)
                    mask = np.asarray(mask, bool)
                    cert = info.get("certified")
                    # ---- (3) the declared convention on its own chain, as 2b
                    s3 = st[("light3", c.kappa)]
                    ref3, inf3, mask3 = OPS3.continue_sea_v3(c, s3["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse)
                    mask3 = np.asarray(mask3, bool)
                    md3 = int(np.sum(mask != mask3)); rr3 = max(resid(ref, ref3), resid(ref3, ref))
                    worst["conv_mask_diff"] = max(worst["conv_mask_diff"], md3); worst["conv_resid"] = max(worst["conv_resid"], rr3)
                    if (inf3.get("n_pocket") or 0) > 0 and c.kappa not in first_pocket_conv:
                        first_pocket_conv[c.kappa] = step
                    r_f, m_f = fine_continuation(c, prev["ref"], prev["f"], f_cur, e, U, s_["n_sea"], prev.get("mask"), a.nsub_fine)
                    mdf = int(np.sum(m_f != mask3))
                    worst["fine_vs_conv_diff"] = max(worst["fine_vs_conv_diff"], mdf); worst["fine_vs_conv_panels"] += int(mdf > 0)
                    s34 = st[("light34", c.kappa)]
                    ref34, info34, mask34 = CERT.continue_sea_v34(c, s34["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse, "eigvals", 10)
                    md34 = int(np.sum(np.asarray(mask34, bool) != mask3))
                    # ---- (2) the certificate's claims, piece by piece, as 2b
                    co = None
                    if isinstance(cert, dict) and cert.get("certified"):
                        co = cert_oracle(c, prev["f"], f_cur, cert, S_host, s_["n_sea"])
                        worst["cert_panels"] += 1; worst["cert_pieces"] += int(co["pieces"])
                        worst["cert_dK_2_minus_G"] = max(worst["cert_dK_2_minus_G"], co["norm_dK_2_minus_G"])
                        if co["pieces"]:
                            worst["cert_start_margin_over_tol"] = min(worst["cert_start_margin_over_tol"], co["start_margin_over_tol"])
                            worst["cert_cholesky_margin_d2_over_F"] = min(worst["cert_cholesky_margin_d2_over_F"], co["cholesky_margin_d2_over_F"])
                            worst["cert_cholesky_margin_d"] = min(worst["cert_cholesky_margin_d"], co["cholesky_margin_d"])
                            worst["cert_soundness_min_d2_over_F"] = min(worst["cert_soundness_min_d2_over_F"], co["soundness_min_d2_over_F"])
                            worst["cert_tiling_defect"] = max(worst["cert_tiling_defect"], co["tiling_defect"])
                            worst["cert_grid_weyl_margin"] = min(worst["cert_grid_weyl_margin"], co["grid_weyl_margin"])
                            worst["cert_grid_count_defect"] += int(co["grid_count_defect"])
                    if info.get("verified"):
                        certified_panels[c.kappa] += 1
                    elif c.kappa not in first_fallback:
                        first_fallback[c.kappa] = {"step": step, "why": info.get("why")}
                    if info.get("cut_crossing"):
                        cut_crossings[c.kappa] += 1
                    panels.append({"step": step, "kappa": c.kappa, "trapped": trapped, "walk_verified": bool(info.get("verified")), "walk_fallback": bool(info.get("fallback")),
                                   "walk_why": info.get("why"), "cut_crossing": bool(info.get("cut_crossing")), "cut_check": info.get("cut_check"), "walk_nsub": info.get("nsub"),
                                   "walk_pieces": (cert.get("pieces") if isinstance(cert, dict) else None), "walk_d_min": (cert.get("d_min_certified") if isinstance(cert, dict) else None),
                                   "walk_d_floor": (cert.get("d_floor") if isinstance(cert, dict) else None), "walk_certificate_valid": (bool(cert.get("certified")) if isinstance(cert, dict) else None),
                                   "conv_mask_diff": md3, "conv_resid": rr3, "conv_n_pocket": inf3.get("n_pocket"), "conv_inversions": inf3.get("inversions"),
                                   "fine_vs_conv_diff": mdf, "v34_verified": bool(info34.get("certified", {}).get("verified")), "v34_pieces": info34.get("nsub"),
                                   "v34_mask_vs_conv": md34, "cert_oracle": co})
                    if md3 or info.get("cut_crossing") or mdf or md34:
                        print("  step %d kappa %d: walk %s%s; convention mask differences %d (residual %.1e); fine %d-sub-step vs convention %d; v34 %s vs convention %d; convention pockets %s; %s" % (
                            step, c.kappa, "certified" if info.get("verified") else "FALLBACK", " at a cut crossing" if info.get("cut_crossing") else "", md3, rr3, a.nsub_fine, mdf,
                            "verified" if panels[-1]["v34_verified"] else "not verified", md34, inf3.get("n_pocket"), (info.get("why") or "")[:160]), flush=True)
                    s34["ref"] = ref34
                    s34["window"] = [{"ref": ref34, "f": (N.copy(), v.copy()), "gap": info34.get("gap_between_blocks"), "weyl_gap": info34.get("weyl_gap"), "mask": np.asarray(mask34, bool)}]
                    s3["ref"] = ref3
                    s3["window"] = [{"ref": ref3, "f": (N.copy(), v.copy()), "gap": inf3.get("gap_between_blocks"), "weyl_gap": inf3.get("weyl_gap"), "mask": mask3}]
                    s_["window"] = [{"ref": ref, "f": (N.copy(), v.copy()), "gap": info.get("gap"), "weyl_gap": info.get("weyl_gap", info.get("gap")), "mask": mask, "mu": info.get("mu"), "n_plus": info.get("n_plus")}]
                s_["ref"] = ref
            else:
                mask = np.asarray(s_["window"][0]["mask"], bool); ref = s_["ref"]
            eigs[key] = (e, U, ref, mask)
            # ---- (4) the sources and E_inv against dense arithmetic (as 2b); v3 records E_inv and the collar currents at every step
            sea = s_["sea"]
            if name == "heavy":
                C = sea @ sea.conj().T + (s_["dust"] * torch.as_tensor(s_["f"], dtype=torch.complex128, device=sea.device)[None, :]) @ s_["dust"].conj().T
            else:
                C = sea @ sea.conj().T
            P = ref @ ref.conj().T
            D = C - P
            hd = E36.clocked_gpu(c, N, v)
            eps_sparse = E36.dens(c.h, sea) - E36.dens(c.h, ref) + (E36.dens(c.h, s_["dust"], s_["f"]) if name == "heavy" else 0.0)
            eps_dense = torch.diagonal(E36.V35.dev_csr(c.hs) @ D).real.reshape(-1, 2).sum(dim=1).cpu().numpy()
            worst["src"] = max(worst["src"], float(np.max(np.abs(eps_sparse - eps_dense))))
            A = U.conj().T @ sea
            nocc = (A.abs() ** 2).sum(dim=1)
            e64 = e.to(torch.float64)
            E_coef = float((e64 * nocc).sum()) - float(e64[torch.as_tensor(mask, device=U.device)].sum())
            if name == "heavy":
                Ad = U.conj().T @ s_["dust"]
                E_coef += float((e64 * ((Ad.abs() ** 2) * torch.as_tensor(s_["f"], dtype=torch.float64, device=U.device)[None, :]).sum(dim=1)).sum())
            E_dense = float(torch.trace(hd @ D).real)
            worst["einv"] = max(worst["einv"], abs(E_coef - E_dense))
            # v3: the collar currents at the step's fields, from the host operator
            h_csr = sp.csr_matrix(E36.clocked_host_sparse(c, N, v))
            cur = {}
            for Rc_ in collars:
                JE, JN = collar_operators(h_csr, R, Rc_)
                JEd = torch.as_tensor(JE.toarray(), dtype=torch.complex128, device=D.device)
                JNd = torch.as_tensor(JN.toarray(), dtype=torch.complex128, device=D.device)
                cur[ck(Rc_)] = {"Phi_E": tr_real(D, JEd), "Phi_N": tr_real(D, JNd)}
                if step == 0 and PO is not None:
                    dE_ = (JE - PO.current_out_csr(h_csr, R, Rc_)).tocsr(); dN_ = (JN - PO.number_current_csr(h_csr, R, Rc_)).tocsr()
                    dd = max(float(abs(dE_).max()) if dE_.nnz else 0.0, float(abs(dN_).max()) if dN_.nnz else 0.0)
                    notes.setdefault("collar_operators_vs_page_ops_v5", {})["%s_%s" % (lab, ck(Rc_))] = dd
                    worst["collar_vs_page_ops"] = max(worst["collar_vs_page_ops"], dd)
            erow = {"step": step, "t": t, "key": str(key), "label": lab, "w": c.w, "sourcing": lab != "light3", "E_inv": E_dense, "E_inv_coef": E_coef,
                    "E_inv_diff": abs(E_coef - E_dense), "collar": cur}
            einv_rows.append(erow); step_rows.append(erow)
            # ---- (5) S_esc at the checkpoints; v3: the archive 1(h_ext > 0)
            if step % a.every == 0:
                hext = hd[ext_rows][:, ext_rows]
                ee, Ue = torch.linalg.eigh(hext)
                ee_np = ee.cpu().numpy().astype(float)
                half_gap = float(np.min(np.abs(ee_np)))
                sel_np = ee_np > 0.0
                idx_sel = np.where(sel_np)[0]
                n_pos = int(idx_sel.size)
                sel = torch.as_tensor(sel_np, device=hd.device)
                Up = Ue[:, sel]
                worst["half_gap_min"] = min(worst["half_gap_min"], half_gap)
                # information: version 2b's test and the fixture document's candidate (B) on the half: V_+ = U_+^H i[h_ext, X] U_+
                Xr = torch.as_tensor(np.repeat(np.asarray(R, float)[ext], 2), dtype=torch.complex128, device=hd.device)
                vel = (1j * (hext @ torch.diag(Xr) - torch.diag(Xr) @ hext))
                Vp = Up.conj().T @ (vel @ Up)
                vexp_abs_max = float(torch.diagonal(Vp).real.abs().max()) if n_pos else 0.0
                vp_ev = torch.linalg.eigvalsh(0.5 * (Vp + Vp.conj().T)).cpu().numpy() if n_pos else np.zeros(0)
                worst["vexp_abs_max"] = max(worst["vexp_abs_max"], vexp_abs_max)
                rows_dev = torch.as_tensor(ext_rows, device=hd.device)
                new = torch.zeros((2 * n, n_pos), dtype=torch.complex128, device=hd.device)
                new[rows_dev] = Up
                Aold = archive.get(key)
                if Aold is not None:
                    new = new - Aold @ (Aold.conj().T @ new); new = new - Aold @ (Aold.conj().T @ new)
                n_cand = int(new.shape[1]); n_kept = 0
                if n_cand:
                    Q_, sv, _ = torch.linalg.svd(new, full_matrices=False)
                    gsv = (sv * sv).cpu().numpy()
                    keep = gsv > max(1e-12, 1e-14 * float(gsv.max()))        # page_ops's rule on unit columns; a block already in the archive adds nothing
                    n_kept = int(keep.sum())
                    if n_kept:
                        newQ = Q_[:, torch.as_tensor(np.where(keep)[0], dtype=torch.int64, device=Q_.device)]
                        archive[key] = newQ if Aold is None else torch.cat([Aold, newQ], dim=1)
                    elif Aold is None:
                        archive[key] = new[:, :0]
                Ak = archive[key]
                gram = float((Ak.conj().T @ Ak - torch.eye(Ak.shape[1], dtype=torch.complex128, device=Ak.device)).abs().max()) if Ak.shape[1] else 0.0
                worst["gram"] = max(worst["gram"], gram)
                if Ak.shape[1] > n_ext2:
                    worst["archive_over_dim"] += 1
                M = Ak.conj().T @ C @ Ak
                lamM = torch.linalg.eigvalsh(0.5 * (M + M.conj().T)).cpu().numpy()
                S_dense = binary_entropy(lamM)
                Pi = Ak @ Ak.conj().T
                lam, V = torch.linalg.eigh(0.5 * (Pi + Pi.conj().T))
                Qc = V[:, lam < 0.5]
                Mc = Qc.conj().T @ C @ Qc
                S_comp = binary_entropy(torch.linalg.eigvalsh(0.5 * (Mc + Mc.conj().T)).cpu().numpy())
                cols = torch.cat([sea, s_["dust"]], dim=1) if name == "heavy" else sea
                B = Ak.conj().T @ cols
                svB = torch.linalg.svdvals(B).cpu().numpy()
                S_mem = binary_entropy(svB * svB)
                # v3: the reference's entropy in the archive's modes (page_core's S_vac(archive)) and the increment above it
                svR = torch.linalg.svdvals(Ak.conj().T @ ref).cpu().numpy()
                S_vac = binary_entropy(svR * svR)
                worst["increment_abs"] = max(worst["increment_abs"], abs(S_dense - S_vac))
                # v3 (information): the negative half, the T-image of the archive, by the retained-channel route; equal to S_dense on a pure sea
                Um = Ue[:, ~sel]
                newm = torch.zeros((2 * n, int(Um.shape[1])), dtype=torch.complex128, device=hd.device)
                newm[rows_dev] = Um
                svN = torch.linalg.svdvals(newm.conj().T @ cols).cpu().numpy()
                S_neg = binary_entropy(svN * svN)
                if name != "heavy":
                    worst["t_symmetry_light"] = max(worst["t_symmetry_light"], abs(S_dense - S_neg))
                # the Z identity against its purity floor and the purified control, as 2b
                cC, VC = torch.linalg.eigh(0.5 * (C + C.conj().T))
                cc = cC.cpu().numpy(); pur = np.abs(cc * (cc - 1.0))
                purity_1 = float(pur.sum()); purity_2 = float(pur.max())
                hz = (purity_1 + 2.0 * n * 8.0 * EPS) / (2.0 * n)
                z_floor = 2.0 * n * (-hz * math.log(hz) - (1.0 - hz) * math.log1p(-hz))
                V1 = VC[:, cC >= 0.5]; C0 = V1 @ V1.conj().T
                M0 = Ak.conj().T @ C0 @ Ak; Mc0 = Qc.conj().T @ C0 @ Qc
                S0_dense = binary_entropy(torch.linalg.eigvalsh(0.5 * (M0 + M0.conj().T)).cpu().numpy())
                S0_comp = binary_entropy(torch.linalg.eigvalsh(0.5 * (Mc0 + Mc0.conj().T)).cpu().numpy())
                z_ = abs(S_dense - S_comp); z0_ = abs(S0_dense - S0_comp)
                worst["sesc"] = max(worst["sesc"], abs(S_dense - S_mem)); worst["z"] = max(worst["z"], z_); worst["z_over_floor"] = max(worst["z_over_floor"], z_ / z_floor)
                worst["purity_1"] = max(worst["purity_1"], purity_1); worst["purity_2"] = max(worst["purity_2"], purity_2)
                worst["z_purified"] = max(worst["z_purified"], z0_); worst["sesc_purity_shift"] = max(worst["sesc_purity_shift"], abs(S_dense - S0_dense))
                row = {"step": step, "t": t, "key": str(key), "label": lab, "w": c.w, "archive_columns": int(Ak.shape[1]), "exterior_dimension": n_ext2,
                       "block_candidates": n_cand, "block_kept": n_kept, "gram_defect": gram, "positive_levels": n_pos, "half_gap": half_gap,
                       "archive_mode_indices": idx_sel.tolist(), "archive_eigenvalues": ee_np[idx_sel].tolist(), "spectrum_AHCA": lamM.tolist(),
                       "S_dense": S_dense, "S_vac_archive": S_vac, "S_increment": S_dense - S_vac, "S_complement": S_comp, "S_retained_channel": S_mem,
                       "S_negative_half": S_neg, "weighted": c.w * S_dense, "purity_defect_1": purity_1, "purity_defect_2": purity_2, "Z_identity": z_,
                       "Z_floor": z_floor, "Z_over_floor": z_ / z_floor, "Z_identity_purified": z0_, "S_dense_purified": S0_dense,
                       "vexp_abs_max_v2b_test": vexp_abs_max,
                       "V_plus": {"min": (float(vp_ev.min()) if n_pos else None), "max": (float(vp_ev.max()) if n_pos else None),
                                  "gap_at_zero": (float(np.min(np.abs(vp_ev))) if n_pos else None), "n_positive": int(np.sum(vp_ev > 0))},
                       "E_inv": E_dense, "E_inv_coef": E_coef, "collar": cur}
                for Rc_ in collars:
                    row["Phi_E_" + ck(Rc_)] = cur[ck(Rc_)]["Phi_E"]; row["Phi_N_" + ck(Rc_)] = cur[ck(Rc_)]["Phi_N"]
                sesc_rows.append(row)
        # v3: the weighted sums of the step (16 kappa; the guard kappa 3 apart)
        wr = weighted_sums(step_rows, collars, ck); wr.update({"step": step, "t": t})
        weighted_rows.append(wr)
        # ---- the propagation, as 2b
        for name, c in chans:
            key = "heavy" if name == "heavy" else ("light", c.kappa)
            e, U, ref, mask = eigs[key]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[key]
            s_["sea"] = U @ (ph[:, None] * (U.conj().T @ s_["sea"]))
            if name == "heavy":
                s_["dust"] = U @ (ph[:, None] * (U.conj().T @ s_["dust"]))
        if step % a.every == 0:
            tot = sum(r_["weighted"] for r_ in sesc_rows if r_["step"] == step)
            guard = sum(r_["weighted"] for r_ in sesc_rows if r_["step"] == step and r_["key"] == str(("light", 3)))
            arch = {r_["key"]: r_["archive_columns"] for r_ in sesc_rows if r_["step"] == step}
            print("step %d t %.1f trapped region %s: S_esc weighted total %.15f (guard kappa 3 share %.3e; archive columns %s, Gram defect %.1e); worst so far: certificate start %s tol, Cholesky %s F (in d %s), tiling %s, grid Weyl %s, counts %d; convention mask differences %d residual %.1e; fine vs convention %d; sources %.2e, E_inv %.2e, S_esc routes %.2e, Z identity %.2e at %.2f of its purity floor (||C^2 - C||_1 %.1e, purified %.1e); heavy fine vs v3 %d (%.0f s)" % (
                step, t, trapped, tot, guard / tot if tot else 0.0, arch, worst["gram"], f_(worst["cert_start_margin_over_tol"], "%.2f"), f_(worst["cert_cholesky_margin_d2_over_F"], "%.2f"),
                f_(worst["cert_cholesky_margin_d"]), f_(worst["cert_tiling_defect"], "%.1e"), f_(worst["cert_grid_weyl_margin"]), worst["cert_grid_count_defect"], worst["conv_mask_diff"],
                worst["conv_resid"], worst["fine_vs_conv_diff"], worst["src"], worst["einv"], worst["sesc"], worst["z"], worst["z_over_floor"], worst["purity_1"], worst["z_purified"],
                worst["heavy_fine_vs_v3_diff"], time.time() - t_gate), flush=True)
            rk = [r_ for r_ in sesc_rows if r_["step"] == step]
            print("  v3 step %d: archive columns %s, positive levels %s, half gap %s; S_vac(archive) %s; S_esc - S_vac %s; negative half - S_dense %s; weighted E_inv %s; weighted Phi_E %s; weighted Phi_N %s" % (
                step, json.dumps({r_["label"]: r_["archive_columns"] for r_ in rk}), json.dumps({r_["label"]: r_["positive_levels"] for r_ in rk}),
                json.dumps({r_["label"]: f_(r_["half_gap"], "%.4e") for r_ in rk}), json.dumps({r_["label"]: G17 % r_["S_vac_archive"] for r_ in rk}),
                json.dumps({r_["label"]: f_(r_["S_increment"], "%.3e") for r_ in rk}), json.dumps({r_["label"]: f_(r_["S_negative_half"] - r_["S_dense"], "%.2e") for r_ in rk}),
                json.dumps(wr["E_inv"], default=float), json.dumps({ck(Rc_): wr["Phi_E_" + ck(Rc_)] for Rc_ in collars}, default=float),
                json.dumps({ck(Rc_): wr["Phi_N_" + ck(Rc_)] for Rc_ in collars}, default=float)), flush=True)
    # ---- the verdicts
    if worst["cert_dK_2_minus_G"] > 1e-12:
        fails.append("(2) ||dK||_2 exceeds the certificate's Gershgorin bound by %.2e" % worst["cert_dK_2_minus_G"])
    if worst["cert_start_margin_over_tol"] < -1.0:
        fails.append("(2) a walk's start d exceeds the true gap of K(0) by %.2f times the eigensolver's tolerance 16 eps ||K||_G" % (-worst["cert_start_margin_over_tol"]))
    if worst["cert_cholesky_margin_d2_over_F"] < -1.0:
        fails.append("(2) a Cholesky-certified piece start has gap^2 - d^2 = %.2f F, below the stated floor -F (in d: %.2e)" % (worst["cert_cholesky_margin_d2_over_F"], worst["cert_cholesky_margin_d"]))
    if worst["cert_tiling_defect"] > 1e-12:
        fails.append("(2) the pieces do not tile [0, 1] within their Weyl lengths (defect %.2e)" % worst["cert_tiling_defect"])
    if worst["cert_grid_weyl_margin"] < -1e-12:
        fails.append("(2) the true gap falls below the Weyl bound on the grid by %.2e" % (-worst["cert_grid_weyl_margin"]))
    if worst["cert_grid_count_defect"] > 0:
        fails.append("(2) the count of negative eigenvalues of K(s) differs from n_sea at %d grid points" % worst["cert_grid_count_defect"])
    print("(2) certificate: %d certified walks (%d pieces) oracled; ||dK||_2 - ||dK||_G at most %s; start margin %s tolerances (criterion >= -1); Cholesky margin (gap^2 - d^2)/F at least %s (criterion >= -1; in d %s); tiling defect %s; grid Weyl margin %s (criterion >= -1e-12); grid count defects %d; the smallest certified d^2/F %s (4 by construction)" % (
        worst["cert_panels"], worst["cert_pieces"], f_(worst["cert_dK_2_minus_G"]), f_(worst["cert_start_margin_over_tol"], "%.2f"), f_(worst["cert_cholesky_margin_d2_over_F"], "%.3f"),
        f_(worst["cert_cholesky_margin_d"]), f_(worst["cert_tiling_defect"], "%.1e"), f_(worst["cert_grid_weyl_margin"]), worst["cert_grid_count_defect"], f_(worst["cert_soundness_min_d2_over_F"], "%.1f")), flush=True)
    if worst["conv_mask_diff"] > 0 or worst["conv_resid"] > 1e-10:
        fails.append("(3) the light reference differs from the declared convention (engine_ops_v3's continuation): mask differences %d, residual %.2e" % (worst["conv_mask_diff"], worst["conv_resid"]))
    n_light_panels = a.steps * len(lights)
    print("(3) convention: mask differences at most %d, residual %.2e over %d panels; the walk certified %s panels per channel, first fallback %s, cut crossings %s; the convention's first pocket per channel %s; the trapped region from step %s; the %d-sub-step continuation differs from the convention on %d panels (at most %d levels)" % (
        worst["conv_mask_diff"], worst["conv_resid"], n_light_panels, json.dumps(certified_panels), json.dumps(first_fallback, default=str), json.dumps(cut_crossings), json.dumps(first_pocket_conv),
        first_trapped, a.nsub_fine, worst["fine_vs_conv_panels"], worst["fine_vs_conv_diff"]), flush=True)
    if worst["src"] > 1e-12:
        fails.append("(4) the sources differ from the dense route by %.2e" % worst["src"])
    if worst["einv"] > 1e-10:
        fails.append("(4) E_inv differs from the dense trace by %.2e" % worst["einv"])
    if worst["sesc"] > 1e-10:
        fails.append("(5) S_esc's dense and retained-channel routes differ by %.2e" % worst["sesc"])
    if worst["z_over_floor"] > 1.0:
        fails.append("(5) the Z identity's defect %.2e is %.2f times its purity floor 2n H((||C^2 - C||_1 + 16 n eps)/2n) (||C^2 - C||_1 at most %.2e; on the purified C0 the identity holds to %.2e)" % (
            worst["z"], worst["z_over_floor"], worst["purity_1"], worst["z_purified"]))
    if worst["gram"] > 1e-12:
        fails.append("(5) the archive's Gram defect reached %.2e" % worst["gram"])
    if worst["archive_over_dim"] > 0:
        fails.append("(5) the archive exceeded the exterior dimension %d at %d checkpoints" % (n_ext2, worst["archive_over_dim"]))
    if worst["half_gap_min"] < 1e-6:
        fails.append("(5) the exterior block has a level within 1e-6 of 0 (half gap %.2e): the archive 1(h_ext > 0) is not rounding-independent there" % worst["half_gap_min"])
    if any(r_["positive_levels"] != n_ext2 // 2 for r_ in sesc_rows):
        fails.append("(5) the positive half's count is not %d at some checkpoint: %s" % (n_ext2 // 2, sorted({r_["positive_levels"] for r_ in sesc_rows})))
    print("(5) archive 1(h_ext > 0): columns %s against the exterior dimension %d (positive levels %s; criterion exactly %d), half gap at least %.3e (criterion 1e-6), Gram defect at most %.1e (criterion 1e-12); S_esc routes %.2e (criterion 1e-10); Z identity %.2e at %.2f of its purity floor 2n H((||C^2 - C||_1 + 16 n eps)/2n) (criterion <= 1; ||C^2 - C||_1 at most %.2e, ||C^2 - C||_2 at most %.2e); on the purified C0 the identity holds to %.2e and S_esc shifts by at most %.2e; version 2b's test |<u|i[h_ext, X]|u>| at most %.1e (the rounding it was); S_esc - S_vac(archive) at most %.2e (the propagation's reach); the light channels' negative half differs from S_dense by at most %.2e (T symmetry); collar operators against page_ops_v5 at most %s (all information)" % (
        sorted({r_["archive_columns"] for r_ in sesc_rows}), n_ext2, sorted({r_["positive_levels"] for r_ in sesc_rows}), n_ext2 // 2, worst["half_gap_min"], worst["gram"], worst["sesc"],
        worst["z"], worst["z_over_floor"], worst["purity_1"], worst["purity_2"], worst["z_purified"], worst["sesc_purity_shift"], worst["vexp_abs_max"], worst["increment_abs"],
        worst["t_symmetry_light"], f_(worst["collar_vs_page_ops"])), flush=True)
    n_cert_before = sum(1 for q in panels if not q["trapped"] and q["v34_verified"]); n_before = sum(1 for q in panels if not q["trapped"])
    n_cert_after = sum(1 for q in panels if q["trapped"] and q["v34_verified"]); n_after = sum(1 for q in panels if q["trapped"])
    n_walk_after = sum(1 for q in panels if q["trapped"] and q["walk_verified"])
    print("version 34's eigenvalue route (information): before the trapped region certified %d of %d panels, after %d of %d; the walk certified %d of %d after; v34 mask differences from the convention: %s; heavy channel: the %d-sub-step continuation differs from version 3's one step on at most %d levels" % (
        n_cert_before, n_before, n_cert_after, n_after, n_walk_after, n_after, sorted({q["v34_mask_vs_conv"] for q in panels}), a.nsub_fine, worst["heavy_fine_vs_v3_diff"]), flush=True)
    # ---- v3: the table every route can match (17 significant digits)
    tab_cols = ["step", "t", "key", "archive_columns", "half_gap", "S_dense", "S_vac_archive", "S_increment", "S_complement", "S_retained_channel", "Z_identity", "Z_floor",
                "Z_over_floor", "purity_defect_1", "purity_defect_2", "S_negative_half", "E_inv"]
    for Rc_ in collars:
        tab_cols += ["Phi_E_" + ck(Rc_), "Phi_N_" + ck(Rc_)]
    head = ["# gate_dense_oracle_v3 S_esc table: one line per checkpoint and channel (key heavy, light1, light2, light3); tab-separated; floats at 17 significant digits.",
            "# archive: " + ARCHIVE_DEFINITION,
            "# S_dense = sum H(lambda(A^H C A)) (nats); S_vac_archive = sum H(s^2), s the singular values of A^H ref; S_increment = S_dense - S_vac_archive; S_complement on the orthogonal complement of A; S_retained_channel = sum H(s^2), s the singular values of A^H [sea, dust]; Z_identity = |S_dense - S_complement| against Z_floor = 2n H((||C^2 - C||_1 + 16 n eps)/2n); S_negative_half by the retained route on 1(h_ext < 0) (information); E_inv = Re Tr[h_m (C - P_ref)]; Phi_E_Rc = Re Tr[(C - P_ref) (i/2)[h_m^2, Pi_{R > Rc}]], Phi_N_Rc = Re Tr[(C - P_ref) i[h_m, Pi_{R > Rc}]] (outward positive; the first collar is the ruled one).",
            "# columns: " + "\t".join(tab_cols)]
    lines = []
    for r_ in sesc_rows:
        vals = []
        for cn in tab_cols:
            x = r_["label"] if cn == "key" else r_[cn]
            vals.append(str(x) if cn in ("step", "key", "archive_columns") else (G17 % float(x)))
        lines.append("\t".join(vals))
    wlines = []
    for step_ in sorted({r_["step"] for r_ in sesc_rows}):
        tot = sum(r_["weighted"] for r_ in sesc_rows if r_["step"] == step_)
        guard = sum(r_["weighted"] for r_ in sesc_rows if r_["step"] == step_ and r_["label"] == "light3")
        wr = next(w_ for w_ in weighted_rows if w_["step"] == step_)
        wlines.append("# weighted step %d: S_esc total %s guard share %s; E_inv heavy %s light sourcing %s guard %s; %s" % (
            step_, G17 % tot, G17 % (guard / tot if tot else 0.0), G17 % wr["E_inv"]["heavy_w"], G17 % wr["E_inv"]["light_sourcing_w"], G17 % wr["E_inv"]["light_guard_w"],
            "; ".join("Phi_E %s heavy %s light sourcing %s guard %s" % (ck(Rc_), G17 % wr["Phi_E_" + ck(Rc_)]["heavy_w"], G17 % wr["Phi_E_" + ck(Rc_)]["light_sourcing_w"], G17 % wr["Phi_E_" + ck(Rc_)]["light_guard_w"]) for Rc_ in collars)))
    with open(os.path.join(a.out, "gate_dense_oracle_v3_table.txt"), "w") as fh:
        fh.write("\n".join(head + lines + wlines) + "\n")
    print("S_esc table (17 significant digits; also %s):" % os.path.join(a.out, "gate_dense_oracle_v3_table.txt"), flush=True)
    for ln in head + lines + wlines:
        print(ln, flush=True)
    with open(os.path.join(a.out, "gate_dense_oracle_v3_result.json"), "w") as fh:
        json.dump({"script": "gate_dense_oracle_v3", "argv": sys.argv, "args": vars(a), "archive_definition": ARCHIVE_DEFINITION, "environment": env,
                   "fixture": {"nsites": n, "tau": a.tau, "steps": a.steps, "every": a.every, "R_cut": a.R_cut, "collars": collars, "ruled_collar": collars[0]},
                   "worst": worst, "symmetry": notes, "panels": panels, "S_esc": sesc_rows, "E_inv_rows": einv_rows, "weighted_rows": weighted_rows,
                   "first_trapped_step": first_trapped, "first_pocket_convention": first_pocket_conv, "first_fallback": first_fallback, "certified_panels": certified_panels,
                   "cut_crossings": cut_crossings, "exterior_dimension": n_ext2, "fails": fails, "pass": not fails, "seconds": time.time() - t_gate}, fh, default=float, indent=1)
    print("GATE D: %s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails)), flush=True)


if __name__ == "__main__":
    main()
