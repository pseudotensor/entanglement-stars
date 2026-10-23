"""curvature_v1_data.py (solve lane, inbox #250, the singularity question; CURVATURE_readout_v1). Data script; the figure is
curvature_v1_fig.py reading the npz this writes. Read-only on every input. Box CPU only (nice, <= 8 threads).

Definitions (followed exactly; source sections named):
  metric (engine dictionary, reviewer B singularity_v1 sec. (2)):  ds^2 = -N^2 dT^2 + (dr - v dT)^2 + r^2 dOmega^2, T and r in units of a.
  The solved, UNBARRED fields (reviewer B sec. (2): "Use the solved, unbarred fields") are the engine's dense constraint solution, stored at
  the coarse nodes r_i = i*ell (r_nodes, N_nodes, v_nodes; v_nodes = -sqrt(W/r) exactly); they are reconstructed here by the engine's own
  quintic parity spline (engine_fields_v4.parity_spline: N even, v odd through the centre), the engine's interp_floor being its floor.
  The stored site fields N, v (= N_applied, v_applied to 1e-6) are the kernel BACK-AVERAGED (barred) fields fed to the matter
  (engine_collapse_v36d.solve, site_fields averaged); they are carried as a labelled variant ("barred").
  (a) reviewer A singularity_v1 sec. (1)(ii) and (2)(a): the spheres' curvature R^{th ph}_{th ph} = 2GM/r^3 = K_t^2 = (v/(rN))^2, against the
      cap (8/3) pi G rho_bar, rho_bar = (4/a^4) sqrt((ma)^2 + 3/2) (= 10.4/a^2 at ma = 0.2, G/a^2 = 1/4) and the gate scale (0.1/sigma)^2.
      reviewer B sec. (2): the slice-normal orthonormal frame norm  script-R = (sum_{abcd} |R_abcd|^2)^{1/2}  from N, v, their radial
      derivatives through second order and first time derivatives (time differences between records); Kretschmann alongside.
      Components (derived symbolically by curvature_v1_riemann_sympy.py, frame e0 = (d_T + v d_r)/N, e1 = d_r):
        R0101 = (N N'' - v v'' - v'^2 - dT v')/N^2 + (N' v v' + dT N v')/N^3
        R0202 = R0303 = (N^2 N' - N v v' - N dT v + N' v^2 + v dT N)/(N^3 r)
        R0212 = R0313 = v N'/(N^2 r);  R1212 = R1313 = v v'/(N^2 r) = K_r K_t;  R2323 = (v/(rN))^2 = K_t^2
        script-R^2 = 4 (R0101^2 + 2 R0202^2 + 2 R1212^2 + R2323^2) + 16 R0212^2;  Kretschmann = same with -16 R0212^2.
      reviewer A sec. (2)(a) second route: the full Riemann from (M, rho, p_r, p_t, J) by the Einstein identities (R2323 from M, the Ricci
      tensor from the kernel-averaged sources rho_ell, J_ell (engine kernels at the nodes) and stresses S_r_ell, S_t_ell (handoff npz)),
      against the (N, v) route; the difference is the evolution residual.
  Gate (both sources point to the manuscript's test, fermion_new/05_star.tex: "the one-term test sigma|H|, H = -d_r v ... against the
      gate value 0.1", first exceeded at T/r_s = 11.75 on the reference history; reviewer A's "curvature <= (0.1/sigma)^2"): the first record
      with sigma * max_r |d_r v| > 0.1. Computed on the barred site v and on the unbarred v; reviewer A's strain form sigma * max(|K_r|,|K_t|)
      also reported.
  (b) reviewer A sec. (2)(b): radial null rays dr/dT = v + s N (s = +1 outgoing, -1 ingoing) from the formation record (first record holding a
      trapping zero), slice-frame energy d ln eps/dT = -(v' + s N') (checked against the geodesic equation by the sympy script), affine
      parameter d lambda = N dT / eps (eps = 1 at launch), reflected at the regular centre (integrated through r < 0 on the parity-extended
      fields, which is the reflection), through the stored records.
"""
import argparse
import hashlib
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np
from scipy.interpolate import make_interp_spline

ENGINE_DIR = "<internal>/engine"   # read-only import of the source kernels (engine_fields_v12.kernels)

MEMBERS = {
    # name: (base path without suffix, certified extent per LADDER_certification_list_v1.2 or None when not listed, role)
    "B2": ("<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096", None,
           "reference: production c19 history B2 (n 4096, to T/r_s 100); not in the v1.2 list"),
    "P64_s4_c19": ("<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11", 40.0,
                   "sigma/a 4 tau 8 production record (engine v19)"),
    "P64_s4": ("<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40", 40.0,
               "sigma/a 4 tau 8 ladder member (v36d from 0)"),
    "P64_s2": ("<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d", 40.0,
               "sigma/a 2 tau 8 composite (Newton)"),
    "P64_s8": ("<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40", 40.0,
               "sigma/a 8 tau 8 (v36d from 0)"),
    "P64_s4_tau4": ("<internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg", 40.0,
                    "sigma/a 4 tau 4 (engine v19)"),
    "P64_s4_tau2": ("<internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg", 12.75,
                    "sigma/a 4 tau 2 (engine v25, to 21.5; certified to 12.75)"),
}

H_CORE = 0.25      # fine radial grid spacing (a) for the curvature profiles
R_CORE = 256.0     # curvature profiles out to 4 r_s (the max over r is also checked on all nodes)
H_RAY = 0.5        # radial grid for the ray fields
GATE = 0.1


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parity_spline(r_nodes, y_nodes, parity):
    """engine_fields_v4.parity_spline, verbatim: quintic spline through the nodes, extended through the centre by parity."""
    rr = np.concatenate([-r_nodes[:0:-1], r_nodes])
    yy = np.concatenate([parity * y_nodes[:0:-1], y_nodes])
    return make_interp_spline(rr, yy, k=5)


def tdiff(F, T, c):
    """time derivative at every record from records k - c and k + c (central), one-sided at the ends; NaN where neither exists."""
    n = F.shape[0]
    D = np.full(F.shape, np.nan)
    for k in range(n):
        lo, hi = k - c, k + c
        if lo >= 0 and hi < n:
            D[k] = (F[hi] - F[lo]) / (T[hi] - T[lo])
        elif hi < n:
            D[k] = (F[hi] - F[k]) / (T[hi] - T[k])
        elif lo >= 0:
            D[k] = (F[k] - F[lo]) / (T[k] - T[lo])
    return D


def comps(r, N, Nr, Nrr, v, vr, vrr, Nt, vt, vtr):
    R0101 = (N * Nrr - v * vrr - vr ** 2 - vtr) / N ** 2 + (Nr * v * vr + Nt * vr) / N ** 3
    R0202 = (N ** 2 * Nr - N * v * vr - N * vt + Nr * v ** 2 + v * Nt) / (N ** 3 * r)
    R0212 = v * Nr / (N ** 2 * r)
    R1212 = v * vr / (N ** 2 * r)
    R2323 = (v / (r * N)) ** 2
    return R0101, R0202, R0212, R1212, R2323


def norms(R0101, R0202, R0212, R1212, R2323):
    base = 4.0 * (R0101 ** 2 + 2 * R0202 ** 2 + 2 * R1212 ** 2 + R2323 ** 2)
    return np.sqrt(base + 16.0 * R0212 ** 2), base - 16.0 * R0212 ** 2


def fd_nodes(rn, y, parity, order):
    """first and second radial derivatives on the node grid by central differences on the parity-extended data (order 2 or 4)."""
    h = rn[1] - rn[0]
    ye = np.concatenate([parity * y[2:0:-1], y, [np.nan, np.nan]])  # two ghost points at each end
    i = np.arange(len(y)) + 2
    if order == 2:
        d1 = (ye[i + 1] - ye[i - 1]) / (2 * h)
        d2 = (ye[i + 1] - 2 * ye[i] + ye[i - 1]) / h ** 2
    else:
        d1 = (-ye[i + 2] + 8 * ye[i + 1] - 8 * ye[i - 1] + ye[i - 2]) / (12 * h)
        d2 = (-ye[i + 2] + 16 * ye[i + 1] - 30 * ye[i] + 16 * ye[i - 1] - ye[i - 2]) / (12 * h ** 2)
    return d1, d2


def fd_sites(R, y, parity):
    """central differences on the site grid R_j = j a (j >= 1) with the centre value from parity (v(0) = 0; N(0) by the even extension)."""
    h = R[1] - R[0]
    if parity < 0:
        ye = np.concatenate([[-y[0]], [0.0], y, [np.nan]])
    else:
        ye = np.concatenate([[y[0]], [y[0]], y, [np.nan]])  # N(0) ~ N(1) to O(h^2) for the even field; only used at the first site
    i = np.arange(len(y)) + 2
    d1 = (ye[i + 1] - ye[i - 1]) / (2 * h)
    d2 = (ye[i + 1] - 2 * ye[i] + ye[i - 1]) / h ** 2
    return d1, d2


def load_member(name):
    base, cert_end, role = MEMBERS[name]
    files = {"json": base + ".json", "fields": base + "_fields.npz"}
    if os.path.exists(base + "_handoff.npz"):
        files["handoff"] = base + "_handoff.npz"
    J = json.load(open(files["json"]))
    Z = np.load(files["fields"], allow_pickle=True)   # object arrays (r_nodes per record) written by the engine itself
    args = J["args"]
    recs = J["records"]
    T = np.asarray(Z["T"], float)
    Tj = np.array([float(r_["T"]) for r_ in recs])
    assert len(Tj) == len(T) and np.allclose(Tj, T), "json/fields T mismatch"
    rs = float(args["rs"])
    out = dict(name=name, role=role, cert_end=cert_end, files=files, args=args, rs=rs, g=float(args["g"]), ell=float(args["ell"]),
               tau=float(args["tau"]), m=float(args["m"]), T=T, Trs=T / rs, R=np.asarray(Z["R"], float),
               Nbar=np.asarray(Z["N"], float), vbar=np.asarray(Z["v"], float),
               Napp=np.asarray(Z["N_applied"], float), vapp=np.asarray(Z["v_applied"], float),
               rn=[np.asarray(x, float) for x in Z["r_nodes"]], Nn=[np.asarray(x, float) for x in Z["N_nodes"]],
               vn=[np.asarray(x, float) for x in Z["v_nodes"]], eps=np.asarray(Z["eps_tot"], float), Jt=np.asarray(Z["J_tot"], float),
               zeros=[r_.get("trapping_zeros") or [] for r_ in recs], converged=[bool(r_.get("converged")) for r_ in recs],
               gauge_intact=[((r_.get("gauge") or {}).get("label", "intact") == "intact") for r_ in recs],
               interp_floor=[(r_.get("interp_floor_N"), r_.get("interp_floor_v")) for r_ in recs])
    if "handoff" in files:
        Hh = np.load(files["handoff"], allow_pickle=False)
        out["hT"] = np.asarray(Hh["T"], float)
        out["hrn"] = np.asarray(Hh["r_nodes"], float)
        out["Sr"] = np.asarray(Hh["S_r_ell"], float)
        out["St"] = np.asarray(Hh["S_t_ell"], float)
    return out


def analyse(M, cadences=(1, 2, 4), do_rays=True, ray_dt_frac=1.0 / 8.0):
    t0 = time.time()
    name, T, rs, g, ell, mA = M["name"], M["T"], M["rs"], M["g"], M["ell"], M["m"]
    nrec = len(T)
    rho_bar = 4.0 * np.sqrt(mA ** 2 + 1.5)          # a = 1 (reviewer A (1)(i))
    cap = 8.0 / 3.0 * np.pi * g * rho_bar             # reviewer A (1)(ii)
    gate_scale = (GATE / ell) ** 2
    rn0 = M["rn"][0]
    for x in M["rn"]:
        assert np.array_equal(x, rn0), "nodes vary between records"
    rc = np.arange(H_CORE, R_CORE + 1e-9, H_CORE)
    rmax_nodes = float(rn0[-1])
    rr = np.arange(-rmax_nodes, rmax_nodes + 1e-9, H_RAY)   # parity-extended ray grid
    sel_core_nodes = (rn0 > 0) & (rn0 <= R_CORE)
    rnc = rn0[sel_core_nodes]
    # --- unbarred fields per record on the core grid, the nodes and the ray grid
    F = {k: np.zeros((nrec, len(rc))) for k in ("N", "Nr", "Nrr", "v", "vr", "vrr")}
    Fn = {k: np.zeros((nrec, len(rnc))) for k in ("N", "Nr", "Nrr", "v", "vr", "vrr")}
    G = {k: np.zeros((nrec, len(rr)), dtype=np.float32) for k in ("N", "v", "Nr", "vr")} if do_rays else None
    Kt2_nodes_max = np.zeros(nrec); Kt2_nodes_rmax = np.zeros(nrec)
    for k in range(nrec):
        sN = parity_spline(rn0, M["Nn"][k], +1); sv = parity_spline(rn0, M["vn"][k], -1)
        for (key, sp_, d) in (("N", sN, 0), ("Nr", sN, 1), ("Nrr", sN, 2), ("v", sv, 0), ("vr", sv, 1), ("vrr", sv, 2)):
            F[key][k] = sp_(rc, d)
            Fn[key][k] = sp_(rnc, d)
        if do_rays:
            G["N"][k] = sN(rr); G["v"][k] = sv(rr); G["Nr"][k] = sN(rr, 1); G["vr"][k] = sv(rr, 1)
        rpos = rn0[1:]
        kt2n = (M["vn"][k][1:] / (rpos * M["Nn"][k][1:])) ** 2   # exact at the nodes, all radii
        j = int(np.argmax(kt2n)); Kt2_nodes_max[k] = kt2n[j]; Kt2_nodes_rmax[k] = rpos[j]
    # --- (a) spheres' curvature, unbarred, fine grid
    Kt2 = (F["v"] / (rc * F["N"])) ** 2
    jmax = np.argmax(Kt2, axis=1)
    Kt2_max = Kt2[np.arange(nrec), jmax]; Kt2_rmax = rc[jmax]
    Kt2_centre = Kt2[:, 0]
    Kr = F["vr"] / F["N"]; Kt = F["v"] / (rc * F["N"])
    strain = np.maximum(np.abs(Kr), np.abs(Kt)).max(axis=1)
    # barred variant: site fields (and applied)
    R = M["R"]; selR = R <= R_CORE
    Kt2_bar = (M["vbar"][:, selR] / (R[selR] * M["Nbar"][:, selR])) ** 2
    Kt2_app = (M["vapp"][:, selR] / (R[selR] * M["Napp"][:, selR])) ** 2
    # gate: one-term test sigma max|d_r v|
    dvbar = np.array([fd_sites(R, M["vbar"][k], -1)[0] for k in range(nrec)])
    onet_bar = ell * np.nanmax(np.abs(dvbar), axis=1)
    onet_unbar = ell * np.abs(F["vr"]).max(axis=1)
    strain_gate = ell * strain

    def first_above(x):
        i = np.where(x > GATE)[0]
        return int(i[0]) if len(i) else None
    gate_k = {"onet_barred": first_above(onet_bar), "onet_unbarred": first_above(onet_unbar), "strain_unbarred": first_above(strain_gate)}
    kgate = gate_k["onet_barred"]
    # --- reviewer B's frame norm, unbarred spline, cadences
    res_norm = {}
    for c in cadences:
        Nt = tdiff(F["N"], T, c); vt = tdiff(F["v"], T, c); vtr = tdiff(F["vr"], T, c)
        C = comps(rc, F["N"], F["Nr"], F["Nrr"], F["v"], F["vr"], F["vrr"], Nt, vt, vtr)
        nr_, kr_ = norms(*C)
        jn = np.nanargmax(nr_, axis=1)
        res_norm[c] = dict(norm_max=nr_[np.arange(nrec), jn], norm_rmax=rc[jn], kret_absmax=np.nanmax(np.abs(kr_), axis=1),
                           comp_absmax=np.array([np.nanmax(np.abs(x), axis=1) for x in C]), norm_centre=nr_[:, 0])
        if c == 1:
            norm_prof = nr_.astype(np.float32); comp_prof = [x.astype(np.float32) for x in C]
    # stencil variants on the nodes (unbarred): spline at nodes, FD2, FD4; and barred site FD2
    sten = {}
    Ntn = tdiff(Fn["N"], T, 1); vtn = tdiff(Fn["v"], T, 1); vtrn = tdiff(Fn["vr"], T, 1)
    nr_s, _ = norms(*comps(rnc, Fn["N"], Fn["Nr"], Fn["Nrr"], Fn["v"], Fn["vr"], Fn["vrr"], Ntn, vtn, vtrn))
    sten["spline@nodes"] = np.nanmax(nr_s, axis=1)
    Nn_all = np.array(M["Nn"]); vn_all = np.array(M["vn"])
    for order in (2, 4):
        d = [fd_nodes(rn0, Nn_all[k], +1, order) for k in range(nrec)]
        e = [fd_nodes(rn0, vn_all[k], -1, order) for k in range(nrec)]
        Nr_ = np.array([x[0] for x in d])[:, sel_core_nodes]; Nrr_ = np.array([x[1] for x in d])[:, sel_core_nodes]
        vr_ = np.array([x[0] for x in e])[:, sel_core_nodes]; vrr_ = np.array([x[1] for x in e])[:, sel_core_nodes]
        N_ = Nn_all[:, sel_core_nodes]; v_ = vn_all[:, sel_core_nodes]
        nr_f, _ = norms(*comps(rnc, N_, Nr_, Nrr_, v_, vr_, vrr_, tdiff(N_, T, 1), tdiff(v_, T, 1), tdiff(vr_, T, 1)))
        sten["FD%d@nodes" % order] = np.nanmax(nr_f, axis=1)
    # barred site fields (FD2 on sites, time differences between records)
    Nb = M["Nbar"][:, selR]; vb = M["vbar"][:, selR]
    db = [fd_sites(R, M["Nbar"][k], +1) for k in range(nrec)]; eb = [fd_sites(R, M["vbar"][k], -1) for k in range(nrec)]
    Nbr = np.array([x[0] for x in db])[:, selR]; Nbrr = np.array([x[1] for x in db])[:, selR]
    vbr = np.array([x[0] for x in eb])[:, selR]; vbrr = np.array([x[1] for x in eb])[:, selR]
    Cb = comps(R[selR], Nb, Nbr, Nbrr, vb, vbr, vbrr, tdiff(Nb, T, 1), tdiff(vb, T, 1), tdiff(vbr, T, 1))
    nr_b, _ = norms(*Cb)
    # the first sites carry the centre ghost; report the barred max from R >= 2
    sten["barred_sites_FD2"] = np.nanmax(nr_b[:, 1:], axis=1)
    # --- reviewer A's source route at the nodes (where the handoff stores the stresses)
    reviewa = None
    if "Sr" in M:
        sys.path.insert(0, ENGINE_DIR)
        import engine_fields_v12 as F12   # noqa: E402  (read-only: kernels only)
        hT = M["hT"]; idx_h = {round(float(t_), 6): i for i, t_ in enumerate(hT)}
        assert np.array_equal(M["hrn"], rn0)
        sel_f = (rnc >= ell) & (rnc <= 160.0)
        rf_ = rnc[sel_f]
        Ricci_cmp = []
        resid = np.full((nrec, 5), np.nan); scale = np.full(nrec, np.nan); con = np.full((nrec, 2), np.nan)
        norm_src = np.full(nrec, np.nan); sgnJ = None
        C1 = comps(rnc, Fn["N"], Fn["Nr"], Fn["Nrr"], Fn["v"], Fn["vr"], Fn["vrr"], Ntn, vtn, vtrn)
        for k in range(nrec):
            ih = idx_h.get(round(float(T[k]), 6))
            if ih is None:
                continue
            sel = (np.abs(M["eps"][k]) > 0) | (np.abs(M["Jt"][k]) > 0)
            K0, K1 = F12.kernels(rf_, R[sel], ell)
            rho = K0 @ M["eps"][k][sel]; Jl = K1 @ M["Jt"][k][sel]
            Sr = M["Sr"][ih][sel_core_nodes][sel_f]; St = M["St"][ih][sel_core_nodes][sel_f]
            Cn = [x[k][sel_f] for x in C1]
            Gnn = Cn[4] + 2 * Cn[3]           # K_t^2 + 2 K_r K_t
            Gnr = 2 * Cn[2]
            if sgnJ is None:
                sgnJ = -1.0 if np.abs(Gnr + 8 * np.pi * g * Jl).max() < np.abs(Gnr - 8 * np.pi * g * Jl).max() else 1.0
            T01 = sgnJ * Jl
            con[k] = [np.abs(Gnn - 8 * np.pi * g * rho).max() / max(np.abs(8 * np.pi * g * rho).max(), 1e-300),
                      np.abs(Gnr - 8 * np.pi * g * T01).max() / max(np.abs(8 * np.pi * g * T01).max(), 1e-300)]
            tr = -rho + Sr + 2 * St
            Ric00 = 8 * np.pi * g * (rho + tr / 2); Ric11 = 8 * np.pi * g * (Sr - tr / 2); Ric22 = 8 * np.pi * g * (St - tr / 2)
            Ric01 = 8 * np.pi * g * T01
            R2323s = Cn[4]
            R0212s = Ric01 / 2
            A_ = (Ric00 + Ric11) / 2; B_ = Ric22 - R2323s
            R1212s = (A_ + B_) / 2; R0202s = (A_ - B_) / 2; R0101s = Ric00 - 2 * R0202s
            Cs = (R0101s, R0202s, R0212s, R1212s, R2323s)
            nsrc, _ = norms(*Cs)
            nnv, _ = norms(*Cn)
            norm_src[k] = np.nanmax(nsrc)
            scale[k] = np.nanmax(nnv)
            resid[k] = [np.nanmax(np.abs(Cs[i] - Cn[i])) for i in range(5)]
        reviewa = dict(resid=resid, scale=scale, constraint=con, norm_src=norm_src, sign_T01_vs_Jell=sgnJ, r_range=[float(rf_[0]), float(rf_[-1])])
    # --- reviewer A (1)(ii) centre identity: K_t^2(0) = (8/3) pi g rho_ell(0), rho_ell from the engine kernels at r = 0
    sys.path.insert(0, ENGINE_DIR)
    import engine_fields_v12 as F12c   # noqa: E402  (read-only: kernels only)
    Kt2_centre_series = np.zeros(nrec)
    for k in range(nrec):
        sel = (np.abs(M["eps"][k]) > 0) | (np.abs(M["Jt"][k]) > 0)
        K0, _ = F12c.kernels(np.array([0.0]), R[sel], ell)
        Kt2_centre_series[k] = 8.0 / 3.0 * np.pi * g * float((K0 @ M["eps"][k][sel])[0])
    # --- formation record and roots
    kform = next((k for k in range(nrec) if len(M["zeros"][k]) > 0), None)
    out = dict(name=name, T=T, Trs=M["Trs"], cap=cap, rho_bar=rho_bar, gate_scale=gate_scale, gate_k=gate_k, kform=kform,
               Kt2_max=Kt2_max, Kt2_rmax=Kt2_rmax, Kt2_centre=Kt2_centre, Kt2_centre_series=Kt2_centre_series, Kt2_nodes_max=Kt2_nodes_max, Kt2_nodes_rmax=Kt2_nodes_rmax,
               Kt2_bar_max=Kt2_bar.max(axis=1), Kt2_bar_rmax=R[selR][np.argmax(Kt2_bar, axis=1)], Kt2_app_max=Kt2_app.max(axis=1),
               strain=strain, onet_bar=onet_bar, onet_unbar=onet_unbar, res_norm=res_norm, sten=sten, reviewa=reviewa, rc=rc,
               Kt2_prof=Kt2.astype(np.float32), norm_prof=norm_prof, comp_prof=comp_prof,
               certified=np.array([(M["cert_end"] is not None and t_ <= M["cert_end"] + 1e-9) for t_ in M["Trs"]]),
               converged=np.array(M["converged"]), zeros=M["zeros"])
    # --- (b) rays
    if do_rays and kform is not None:
        out["rays"] = {}
        for c in cadences:
            out["rays"][c] = rays(M, G, rr, kform, c, ray_dt_frac)
        out["rays_dt_half"] = rays(M, G, rr, kform, 1, ray_dt_frac / 2)
    out["seconds"] = time.time() - t0
    return out


def rays(M, G, rr, kform, c, dt_frac):
    """all launches integrated together: RK4 in T, fields linear in r on the H_RAY grid and linear in T between the records kept at
    cadence c (records kform, kform + c, ...)."""
    T = M["T"]; nrec = len(T)
    keep = np.arange(kform, nrec, c)
    Tk = T[keep]
    z = M["zeros"][kform]
    r_in, r_out = float(min(z)), float(max(z))
    rn0 = M["rn"][0]
    base = [0.25, 0.5 * r_in, r_in, 0.5 * (r_in + r_out), r_out, 1.5 * r_out, 2.0 * r_out, 4.0 * r_out]
    nodes = [float(x) for x in rn0 if 0 < x <= 3.0 * r_out]
    r0s = np.array(sorted(set([round(x, 6) for x in base + nodes])))
    labels = []
    for x in r0s:
        lab = ""
        for nm, val in (("centre", 0.25), ("r_in/2", 0.5 * r_in), ("r_in", r_in), ("mid", 0.5 * (r_in + r_out)), ("r_out", r_out),
                        ("1.5 r_out", 1.5 * r_out), ("2 r_out", 2.0 * r_out), ("4 r_out", 4.0 * r_out)):
            if abs(x - round(val, 6)) < 1e-9:
                lab = nm
        labels.append(lab)
    n0 = len(r0s)
    r = np.concatenate([r0s, r0s]); s = np.concatenate([np.ones(n0), -np.ones(n0)])
    lneps = np.zeros(2 * n0); lam = np.zeros(2 * n0)
    eps_max = np.ones(2 * n0); eps_min = np.ones(2 * n0)
    alive = np.ones(2 * n0, bool)
    t_centre = np.full(2 * n0, np.nan); lam_centre = np.full(2 * n0, np.nan)
    dt = (T[kform + 1] - T[kform]) * dt_frac if kform + 1 < nrec else 1.0
    rlim = rr[-1] - 2.0
    h = rr[1] - rr[0]
    Gk = {key: G[key][keep].astype(np.float64) for key in G}

    def fields(t, x):
        j = np.searchsorted(Tk, t, side="right") - 1
        j = np.clip(j, 0, len(Tk) - 2)
        w = (t - Tk[j]) / (Tk[j + 1] - Tk[j])
        xi = (x - rr[0]) / h
        i = np.clip(np.floor(xi).astype(int), 0, len(rr) - 2)
        u = xi - i
        out = []
        for key in ("N", "v", "Nr", "vr"):
            A = Gk[key]
            f0 = A[j, i] * (1 - u) + A[j, i + 1] * u
            f1 = A[j + 1, i] * (1 - u) + A[j + 1, i + 1] * u
            out.append(f0 * (1 - w) + f1 * w)
        return out

    def rhs(t, x, ss, le):
        N, v, Nr, vr = fields(t, x)
        return v + ss * N, -(vr + ss * Nr), N / np.exp(le)

    t = T[kform]; tend = Tk[-1]
    traj_t = [t]; traj_r = [r.copy()]; traj_lam = [lam.copy()]; traj_le = [lneps.copy()]
    nstep = 0
    while t < tend - 1e-12:
        h_ = min(dt, tend - t)
        a = alive
        x0, l0, m0 = r[a], lneps[a], lam[a]
        ss = s[a]
        k1 = rhs(t, x0, ss, l0)
        k2 = rhs(t + h_ / 2, x0 + h_ / 2 * k1[0], ss, l0 + h_ / 2 * k1[1])
        k3 = rhs(t + h_ / 2, x0 + h_ / 2 * k2[0], ss, l0 + h_ / 2 * k2[1])
        k4 = rhs(t + h_, x0 + h_ * k3[0], ss, l0 + h_ * k3[1])
        x1 = x0 + h_ / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        l1 = l0 + h_ / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        m1 = m0 + h_ / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        # centre crossing (sign change of r), linear in the step
        cross = (x0 > 0) & (x1 <= 0) | (x0 < 0) & (x1 >= 0)
        idx = np.where(a)[0]
        for ii in np.where(cross)[0]:
            g_ = idx[ii]
            if np.isnan(t_centre[g_]):
                f = x0[ii] / (x0[ii] - x1[ii])
                t_centre[g_] = t + f * h_; lam_centre[g_] = m0[ii] + f * (m1[ii] - m0[ii])
        r[a] = x1; lneps[a] = l1; lam[a] = m1
        e_ = np.exp(l1)
        eps_max[a] = np.maximum(eps_max[a], e_); eps_min[a] = np.minimum(eps_min[a], e_)
        gone = a & (np.abs(r) > rlim)
        alive = alive & ~gone
        t += h_; nstep += 1
        if nstep % 8 == 0 or t >= tend - 1e-12:
            traj_t.append(t); traj_r.append(r.copy()); traj_lam.append(lam.copy()); traj_le.append(lneps.copy())
    # end-state classification against the last kept record's trapping zeros
    zl = M["zeros"][keep[-1]]
    Ne, ve, _, _ = fields(t, np.where(alive, r, 0.0))
    vpN_end = np.where(alive, (ve + s * Ne) * np.sign(np.where(r == 0, 1.0, r)), np.nan)   # d|r|/dT at the end (reflected rays have r < 0)
    reff = np.abs(r)
    seff = s * np.sign(np.where(r == 0, 1.0, r))
    status = []
    for i in range(2 * n0):
        if not alive[i]:
            status.append("left the grid outward (r > %.0f a): escaped" % rlim)
            continue
        x = reff[i]
        if len(zl) >= 2:
            lo, hi = min(zl), max(zl)
            if x > hi:
                status.append("outside the outer root, outgoing: escaping" if seff[i] > 0 else "outside the outer root, ingoing")
            elif x >= lo:
                status.append("between the roots (trapped shell)")
            else:
                status.append("inside the inner root (untrapped core)")
        elif len(zl) == 1:
            status.append("beyond the single zero" if x > zl[0] else "inside the single zero")
        else:
            status.append("no trapping zero at the window's end")
    return dict(r0=np.concatenate([r0s, r0s]), s=s, labels=labels + labels, r_end=reff, s_eff_end=seff, lam_end=lam, eps_end=np.exp(lneps),
                eps_max=eps_max, eps_min=eps_min, t_centre=t_centre, lam_centre=lam_centre, alive=alive, status=status,
                T_start=float(T[kform]), T_end=float(tend), zeros_end=zl, r_in0=r_in, r_out0=r_out, cadence=c, dt=dt,
                traj_t=np.array(traj_t), traj_r=np.array(traj_r, dtype=np.float32), traj_lam=np.array(traj_lam), traj_le=np.array(traj_le),
                vpN_end=vpN_end, nsteps=nstep)


def summarise(name, M, A):
    """per member: the numbers for the readout (json-able)."""
    T, Trs = A["T"], A["Trs"]
    kg = A["gate_k"]["onet_barred"]
    pre = np.arange(len(T)) < (kg if kg is not None else len(T))
    post = ~pre

    def mx(x, m):
        if not m.any():
            return None
        xx = np.where(m, x, -np.inf)
        j = int(np.nanargmax(xx))
        return {"value": float(x[j]), "T_over_rs": float(Trs[j]), "k": j}
    cert = A["certified"]
    S = {"name": name, "role": M["role"], "cert_end": M["cert_end"], "rs": M["rs"], "g": M["g"], "sigma": M["ell"], "tau": M["tau"], "m": M["m"],
         "records": int(len(T)), "T_over_rs_range": [float(Trs[0]), float(Trs[-1])], "n_converged": int(A["converged"].sum()),
         "cap": A["cap"], "rho_bar": A["rho_bar"], "gate_scale": A["gate_scale"],
         "gate": {k_: (None if v_ is None else {"k": v_, "T_over_rs": float(Trs[v_])}) for k_, v_ in A["gate_k"].items()},
         "formation": None if A["kform"] is None else {"k": A["kform"], "T_over_rs": float(Trs[A["kform"]]), "zeros": A["zeros"][A["kform"]]},
         "Kt2_max_all": mx(A["Kt2_max"], np.ones(len(T), bool)), "Kt2_max_pre": mx(A["Kt2_max"], pre), "Kt2_max_post": mx(A["Kt2_max"], post),
         "Kt2_max_certified": mx(A["Kt2_max"], cert), "Kt2_nodes_max_all": mx(A["Kt2_nodes_max"], np.ones(len(T), bool)),
         "Kt2_bar_max_all": mx(A["Kt2_bar_max"], np.ones(len(T), bool)), "Kt2_app_max_all": mx(A["Kt2_app_max"], np.ones(len(T), bool)),
         "Kt2_centre_max": mx(A["Kt2_centre"], np.ones(len(T), bool)),
         "Kt2_centre_series_max": mx(A["Kt2_centre_series"], np.ones(len(T), bool)),
         "centre_spline_vs_series_rel_max": float(np.max(np.abs(A["Kt2_centre"] - A["Kt2_centre_series"]) / np.maximum(np.abs(A["Kt2_centre_series"]), 1e-300))),
         "frac_cap_max": float(np.max(A["Kt2_max"]) / A["cap"]), "over_gate_scale_max": float(np.max(A["Kt2_max"]) / A["gate_scale"]),
         "records_Kt2_above_cap": int((A["Kt2_max"] > A["cap"]).sum()), "records_Kt2_above_gate_scale": int((A["Kt2_max"] > A["gate_scale"]).sum()),
         "norm": {}, "stencil": {}, "seconds": A["seconds"]}
    if A["kform"] is not None:
        kf = A["kform"]
        S["at_formation"] = {"Kt2_max": float(A["Kt2_max"][kf]), "r": float(A["Kt2_rmax"][kf]), "Kt2_centre": float(A["Kt2_centre"][kf])}
    if kg is not None:
        S["at_gate"] = {"Kt2_max": float(A["Kt2_max"][kg]), "r": float(A["Kt2_rmax"][kg])}
    for c, R_ in A["res_norm"].items():
        S["norm"][str(c)] = {"pre": mx(R_["norm_max"], pre), "post": mx(R_["norm_max"], post), "certified": mx(R_["norm_max"], cert),
                             "kret_pre": mx(R_["kret_absmax"], pre), "kret_post": mx(R_["kret_absmax"], post),
                             "r_at_post_max": None}
        p = S["norm"][str(c)]["post"]
        if p is not None:
            S["norm"][str(c)]["r_at_post_max"] = float(R_["norm_rmax"][p["k"]])
            S["norm"][str(c)]["components_at_post_max"] = dict(zip(["R0101", "R0202", "R0212", "R1212", "R2323"], [float(x) for x in R_["comp_absmax"][:, p["k"]]]))
        q = S["norm"][str(c)]["pre"]
        if q is not None:
            S["norm"][str(c)]["r_at_pre_max"] = float(R_["norm_rmax"][q["k"]])
    for key, arr in A["sten"].items():
        S["stencil"][key] = {"pre": mx(arr, pre), "post": mx(arr, post)}
    if A["reviewa"] is not None:
        Fb = A["reviewa"]
        rel = Fb["resid"] / Fb["scale"][:, None]
        ok = ~np.isnan(Fb["scale"])

        def mxv(x, m):
            m = m & ok
            return None if not m.any() else float(np.nanmax(x[m]))
        S["reviewa_route"] = {"r_range": Fb["r_range"], "sign_T01_vs_Jell": Fb["sign_T01_vs_Jell"], "records": int(ok.sum()),
                            "constraint_rel_max_pre": [mxv(Fb["constraint"][:, i], pre) for i in range(2)],
                            "constraint_rel_max_post": [mxv(Fb["constraint"][:, i], post) for i in range(2)],
                            "resid_over_norm_pre": {nm: mxv(rel[:, i], pre) for i, nm in enumerate(["R0101", "R0202", "R0212", "R1212", "R2323"])},
                            "resid_over_norm_post": {nm: mxv(rel[:, i], post) for i, nm in enumerate(["R0101", "R0202", "R0212", "R1212", "R2323"])},
                            "norm_src_pre": mxv(Fb["norm_src"], pre), "norm_src_post": mxv(Fb["norm_src"], post),
                            "norm_Nv_nodes_pre": mxv(Fb["scale"], pre), "norm_Nv_nodes_post": mxv(Fb["scale"], post)}
    if "rays" in A:
        S["rays"] = {}
        for c, Ry in list(A["rays"].items()) + [("dt_half", A["rays_dt_half"])]:
            rows = []
            tt = Ry["traj_t"]; j5 = int(np.searchsorted(tt, tt[-1] - 5.0 * M["rs"]))
            j5 = min(max(j5, 0), len(tt) - 1)
            dlam5 = Ry["traj_lam"][-1] - Ry["traj_lam"][j5]
            rate5 = (Ry["traj_le"][-1] - Ry["traj_le"][j5]) / max(tt[-1] - tt[j5], 1e-300)
            cc_ = 1 if c == "dt_half" else int(c)
            kl = A["kform"] + ((len(A["T"]) - 1 - A["kform"]) // cc_) * cc_   # the last record kept at this cadence
            rc_ = A["rc"]
            for i in range(len(Ry["r0"])):
                rows.append({"label": Ry["labels"][i], "r0": float(Ry["r0"][i]), "dir": "out" if Ry["s"][i] > 0 else "in",
                             "r_end": float(Ry["r_end"][i]), "eps_end": float(Ry["eps_end"][i]), "eps_max": float(Ry["eps_max"][i]),
                             "eps_min": float(Ry["eps_min"][i]), "lam_end": float(Ry["lam_end"][i]),
                             "T_centre_over_rs": (None if np.isnan(Ry["t_centre"][i]) else float(Ry["t_centre"][i] / M["rs"])),
                             "lam_centre": (None if np.isnan(Ry["lam_centre"][i]) else float(Ry["lam_centre"][i])), "status": Ry["status"][i],
                             "dlam_last_5rs": float(dlam5[i]), "dlneps_dT_last_5rs": float(rate5[i]),
                             "drdT_end": (None if np.isnan(Ry["vpN_end"][i]) else float(Ry["vpN_end"][i])),
                             "Kt2_at_end": (float(np.interp(Ry["r_end"][i], rc_, A["Kt2_prof"][kl])) if Ry["r_end"][i] <= rc_[-1] else None),
                             "norm_at_end": (float(np.interp(Ry["r_end"][i], rc_, A["norm_prof"][kl])) if Ry["r_end"][i] <= rc_[-1] else None)})
            S["rays"][str(c)] = {"T_start_over_rs": Ry["T_start"] / M["rs"], "T_end_over_rs": Ry["T_end"] / M["rs"], "zeros_end": Ry["zeros_end"],
                                 "r_in0": Ry["r_in0"], "r_out0": Ry["r_out0"], "dt": Ry["dt"], "nsteps": Ry["nsteps"], "rows": rows}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=",".join(MEMBERS))
    ap.add_argument("--out", default="<internal>/curvature_v1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inputs = {}
    summ = {}
    for name in a.members.split(","):
        M = load_member(name)
        inputs[name] = {k: {"path": p, "md5": md5(p)} for k, p in M["files"].items()}
        A = analyse(M)
        S = summarise(name, M, A)
        summ[name] = S
        # npz for the figure script
        arrs = dict(T=A["T"], Trs=A["Trs"], cap=A["cap"], gate_scale=A["gate_scale"], Kt2_max=A["Kt2_max"], Kt2_rmax=A["Kt2_rmax"],
                    Kt2_centre=A["Kt2_centre"], Kt2_centre_series=A["Kt2_centre_series"], Kt2_bar_max=A["Kt2_bar_max"], Kt2_nodes_max=A["Kt2_nodes_max"], strain=A["strain"],
                    onet_bar=A["onet_bar"], onet_unbar=A["onet_unbar"], rc=A["rc"], Kt2_prof=A["Kt2_prof"], norm_prof=A["norm_prof"],
                    certified=A["certified"], gauge_intact=np.array(M["gauge_intact"]),
                    nzeros=np.array([len(z_) for z_ in M["zeros"]]), r_in_rec=np.array([min(z_) if z_ else np.nan for z_ in M["zeros"]]),
                    r_out_rec=np.array([max(z_) if z_ else np.nan for z_ in M["zeros"]]), kform=-1 if A["kform"] is None else A["kform"],
                    kgate=-1 if A["gate_k"]["onet_barred"] is None else A["gate_k"]["onet_barred"])
        for c, R_ in A["res_norm"].items():
            arrs["norm_max_c%d" % c] = R_["norm_max"]; arrs["norm_rmax_c%d" % c] = R_["norm_rmax"]; arrs["kret_absmax_c%d" % c] = R_["kret_absmax"]
            arrs["comp_absmax_c%d" % c] = R_["comp_absmax"]
        for key, arr in A["sten"].items():
            arrs["sten_" + key.replace("@", "_at_")] = arr
        if A["reviewa"] is not None:
            arrs["reviewa_resid"] = A["reviewa"]["resid"]; arrs["reviewa_scale"] = A["reviewa"]["scale"]; arrs["reviewa_constraint"] = A["reviewa"]["constraint"]
            arrs["reviewa_norm_src"] = A["reviewa"]["norm_src"]
        if "rays" in A:
            for c, Ry in A["rays"].items():
                arrs["ray_c%d_t" % c] = Ry["traj_t"]; arrs["ray_c%d_r" % c] = Ry["traj_r"]; arrs["ray_c%d_r0" % c] = Ry["r0"]; arrs["ray_c%d_s" % c] = Ry["s"]
                arrs["ray_c%d_lam" % c] = Ry["traj_lam"]; arrs["ray_c%d_le" % c] = Ry["traj_le"]
        np.savez_compressed(os.path.join(a.out, "curvature_v1_%s.npz" % name), **arrs)
        json.dump(S, open(os.path.join(a.out, "curvature_v1_%s.json" % name), "w"), indent=1, default=float)
        print("%s: %d records, %.0f s; max Kt2 %.4g (cap %.3g, frac %.3g) at T/rs %.3f r %.2f; gate %s; formation %s" % (
            name, S["records"], S["seconds"], S["Kt2_max_all"]["value"], S["cap"], S["frac_cap_max"], S["Kt2_max_all"]["T_over_rs"],
            float(A["Kt2_rmax"][S["Kt2_max_all"]["k"]]), S["gate"], S["formation"]), flush=True)
    json.dump({"inputs": inputs, "summary": summ, "script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__))},
              open(os.path.join(a.out, "curvature_v1_all_%s.json" % a.members.replace(",", "+")[:80]), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
