#!/usr/bin/env python
"""Strand 0 part A, data script (spec solve/specs/SPEC_strand0_source_allocation.md, secs. 1-6; builder analyst).

ring4, 8 modes, sectors N = 4, 5, 6.  The law's clock derivative rho^P_x (dressed and bare), the law's midpoint density
rho^M_x, the native midpoint density e_x^nat(psi) (frozen code, mapped by Gamma(W)), the native re-solves at k = 1
(B1 backgrounds), and controls C1 to C11.  Writes <out>/partA.json (all scalars and tables) and <out>/mats/*.npz
(matrices per case).  Every number in the result file comes from partA.json through strand0_partA_report.py.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import gzip
import json
import socket
import sys
import tarfile
import time

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand0_lib as L  # noqa: E402
import native_core as nc  # noqa: E402
import native_sources as ns  # noqa: E402
import native_run as nr  # noqa: E402

MASSES = (0.1, 0.2)
UVALS = (0.0, 0.01, 0.1)
SECTORS = (4, 5, 6)
NSITE, NMODE = 4, 8
QOF = {4: 0, 5: 1, 6: 2}
# n_exc block of interest per sector: (n_p, n_h) -> n_exc value
BLOCKS = {4: {(0, 0): 0, (1, 1): 2, (2, 2): 4, (3, 3): 6, (4, 4): 8},
          5: {(1, 0): 1, (2, 1): 3, (3, 2): 5, (4, 3): 7},
          6: {(2, 0): 2, (3, 1): 4, (4, 2): 6}}


def fl(z):
    return float(np.real(z))


def nrm(M):
    return float(np.linalg.norm(M))


def snrm(M):
    return float(np.linalg.norm(M, 2)) if M.size else 0.0


def eigs(M):
    return [float(v) for v in np.linalg.eigvalsh(L.herm(M))]


def compress(M, P):
    return P.conj().T @ M @ P


# ------------------------------------------------------------------ backgrounds
def fixed_backgrounds():
    x = np.arange(NSITE)
    bgs = {"B0": np.ones(NSITE)}
    for e in (0.05, 0.2):
        bgs["B2_e%g" % e] = 1 + e * np.cos(np.pi * x / 2)
        bgs["B3_e%g" % e] = 1 + e * (-1.0) ** x
    bgs["B4"] = 1 + 0.2 * (x == 0)
    return bgs


REPORT_SITES = {"B0": [0], "B2_e0.05": [0, 2], "B2_e0.2": [0, 2], "B3_e0.05": [0, 1], "B3_e0.2": [0, 1], "B4": [0, 2]}


# ------------------------------------------------------------------ stored native energies (control C9)
def stored_energies(jobs_tar, kscan_gz):
    out = {}
    with tarfile.open(jobs_tar) as tf:
        for mem in tf.getmembers():
            if not (mem.isfile() and mem.name.endswith(".json")):
                continue
            rec = json.load(tf.extractfile(mem))
            sp = rec.get("spec", {})
            if sp.get("kind") == "main" and sp.get("lat") == "ring4" and sp.get("sub") == "wick":
                E = rec.get("dispersion", {}).get("0.0", {}).get("E")
                if E is not None:
                    out[(sp["prom"], int(sp["q"]), float(sp["m"]), float(sp["U"]))] = float(E)
    D = json.load(gzip.open(kscan_gz, "rt"))
    for rec in D["records"]:
        prom, sub, m, q, grid, k = rec["spec"]
        if sub == "wick" and grid == "wide" and float(k) == 1.0 and "0.0" in rec["P"] and "E" in rec["P"]["0.0"]:
            out[(prom, int(q), float(m), 0.0)] = float(rec["P"]["0.0"]["E"])
    return out


# ------------------------------------------------------------------ native re-solves (sec. 2.1)
def native_resolves(stored, matdir):
    recs = []
    for prom in ("J_W6", "END"):
        for q in (0, 1, 2):
            for m in MASSES:
                for U in (0.0, 0.1):
                    t0 = time.time()
                    prob = nr.build("ring4", prom, "wick", q, m, U, 0.0, 1.0)
                    br, Ntop = prob["br"], prob["Ntop"]
                    S, info = nr.solve_at(prob)
                    r = {"prom": prom, "q": q, "m": m, "U": U, "Ntop": Ntop, "ok": bool(info.get("ok", False))}
                    if not r["ok"]:
                        r["info"] = nr.summarize_info(info)
                        recs.append(r)
                        continue
                    r["final_residual"] = float(info.get("final_residual", np.nan))
                    H = br.Q(S)
                    w, V, blocks, E0, deg = nr.lowest_block(H)
                    bl = blocks[0]
                    rho = V[:, bl] @ V[:, bl].conj().T / deg
                    Sx = [S[br.index[(x, Ntop)]] for x in range(NSITE)]
                    sbar = [fl(np.trace(rho @ Sx[x])) for x in range(NSITE)]
                    ex = br.source_values(S)
                    r.update({"E0": E0, "deg": deg, "sbar": sbar,
                              "S_min_eig": [eigs(Sx[x])[0] for x in range(NSITE)],
                              "S_max_eig": [eigs(Sx[x])[-1] for x in range(NSITE)],
                              "eS_min": [eigs(ex[x])[0] for x in range(NSITE)],
                              "eS_max": [eigs(ex[x])[-1] for x in range(NSITE)],
                              "sbar_spread": float(max(sbar) - min(sbar)),
                              "seconds": time.time() - t0})
                    st = stored.get((prom, q, m, U))
                    r["E_stored"] = st
                    r["E_minus_stored"] = None if st is None else E0 - st
                    np.savez_compressed(os.path.join(matdir, "native_%s_q%d_m%g_U%g.npz" % (prom, q, m, U)),
                                        S=np.array(Sx), H=H, eS=np.array([ex[x] for x in range(NSITE)]), rho=rho)
                    recs.append(r)
    return recs


# ------------------------------------------------------------------ the spinor map (sec. 2.3, control C1)
def choose_W(m):
    lat = nc.lattice("ring4", (0.0, 0.0, 0.0))
    law = L.Law(m, np.ones(NSITE))
    target = law.h_star()
    res = {}
    for name, W in L.W_candidates().items():
        Wf = np.kron(np.eye(NSITE), W)
        res[name] = nrm(Wf @ L.native_h(lat, m, np.ones(NSITE)) @ Wf.conj().T - target)
    best = min(res, key=res.get)
    return best, L.W_candidates()[best], res


def gamma_controls(Wf, fock, lat, m):
    out = {}
    h = L.native_h(lat, m, np.ones(NSITE))
    gams = {}
    for N in range(1, NMODE):
        sec = fock.sector(N)
        Gm = L.gamma_of(Wf, sec)
        gams[N] = Gm
        if N in SECTORS:
            out["C1b_unitary_N%d" % N] = nrm(Gm.conj().T @ Gm - np.eye(sec.d))
            out["C1b_intertwine_N%d" % N] = nrm(Gm @ sec.bilinear(h) @ Gm.conj().T - sec.bilinear(Wf @ h @ Wf.conj().T))
            worst = 0.0
            for x in range(NSITE):
                n1 = sec.number(2 * x)
                n2 = sec.number(2 * x + 1)
                npl = sec.bilinear(L.site_pm(NSITE, x, +1))
                nmi = sec.bilinear(L.site_pm(NSITE, x, -1))
                worst = max(worst, nrm(Gm @ (n1 @ n2) @ Gm.conj().T - npl @ nmi))
            out["C1c_quartic_N%d" % N] = worst
    return gams, out


# ------------------------------------------------------------------ one law case
def law_case(name, psi, m, U, fock, gams, lat, Gfrozen, Nfield=None, sectors=SECTORS, sites=None, eS=None,
             matdir=None, save=True):
    t0 = time.time()
    law = L.Law(m, psi, N=Nfield)
    comp = L.Completion(law, fock, U, sectors=range(2, 8), dress=sectors)
    qf = comp.qf
    Om, N0 = comp.Omega, comp.N0
    sites = list(range(NSITE)) if sites is None else sites
    rec = {"bg": name, "m": m, "U": U, "psi": list(map(float, psi)),
           "N": list(map(float, law.N)), "sectors": {}, "controls": {}}
    c = rec["controls"]
    # C2 polar decomposition
    c["C2_A_minus_UE0"] = nrm(law.A - law.U @ law.E0)
    c["C2_UdagU_minus_1"] = nrm(law.U.conj().T @ law.U - np.eye(NSITE))
    c["C2_E1_minus_UE0Udag"] = nrm(law.E1 - law.U @ law.E0 @ law.U.conj().T)
    c["C2_min_eig_AdagA"] = law.minAA
    # C3 spectrum at the background N
    hs = law.h_star()
    ev_h = np.sort(np.linalg.eigvalsh(L.herm(hs)))
    epsN = np.linalg.eigvalsh(law.B0)
    c["C3_eig_h_vs_pm_eps"] = float(np.max(np.abs(ev_h - np.sort(np.concatenate([-epsN, epsN])))))
    if np.allclose(psi, psi[0]) and np.allclose(law.N, 1.0):
        s = float(psi[0])
        p = 2 * np.pi * np.arange(NSITE) / NSITE
        c["C3_eps_closed_form"] = float(np.max(np.abs(np.sort(law.eps) - np.sort(np.sqrt(m * m + s ** -4 * np.sin(p) ** 2)))))
        c["C3b_eig_h_vs_native"] = float(np.max(np.abs(ev_h - np.sort(np.linalg.eigvalsh(L.native_h(lat, m, psi))))))
    # C4b vacuum of the fermion bilinear in N = 4
    sec4 = fock.sector(N0)
    Hc = sec4.bilinear(hs)
    w4, V4 = np.linalg.eigh(L.herm(Hc))
    c["C4b_Evac_minus_minusTrB0"] = float(w4[0] + np.trace(law.B0).real)
    c["C4b_overlap_minus_1"] = float(abs(abs(np.vdot(V4[:, 0], Om)) - 1.0))
    rec["Evac_fermion"] = float(-np.trace(law.B0).real)
    rec["Evac_bilinear_lowest"] = float(w4[0])
    # C5a, C5b
    c["C5a_vacuum"] = float(abs(Om.conj() @ comp.Vno[N0] @ Om))
    worst = 0.0
    for i in range(2 * NSITE):
        for j in range(2 * NSITE):
            (vi, Ni), (vj, Nj) = comp.one[i], comp.one[j]
            if Ni == Nj:
                worst = max(worst, abs(vi.conj() @ comp.Vno[Ni] @ vj))
    c["C5a_one_excitation"] = float(worst)
    c["C5b_mixed_class"] = "classes live in different N sectors (pp: 6, hh: 2, ph: 4); no mixed matrix element exists"
    # T5 law reference derivatives (per site)
    t5 = []
    for x in range(NSITE):
        t5.append({"x": x, "sea": fl(-psi[x] * law.E0[x, x]), "boson": fl(psi[x] * law.E1[x, x]),
                   "w": fl(psi[x] * (law.E1[x, x] - law.E0[x, x]))})
    rec["T5_law"] = t5
    rec["sum_Nw"] = float(sum(law.N[x] * t5[x]["w"] for x in range(NSITE)))
    rec["TrL_E1_minus_E0"] = fl(np.trace(np.diag(law.N * law.psi) @ (law.E1 - law.E0)))
    # one-body (single-particle) objects
    X1 = {x: law.dh_dN(x) for x in range(NSITE)}
    M1 = {x: law.midpoint(x, law.h_star(B0=law.B0_unit)) for x in range(NSITE)}
    Vq = law.Vq
    ob = {}
    for x in range(NSITE):
        Dq = Vq.conj().T @ (X1[x] - M1[x]) @ Vq
        Xq = Vq.conj().T @ X1[x] @ Vq
        Mq = Vq.conj().T @ M1[x] @ Vq
        ob[x] = {"pp": nrm(Dq[:NSITE, :NSITE]) / nrm(X1[x]), "hh": nrm(Dq[NSITE:, NSITE:]) / nrm(X1[x]),
                 "anom": (nrm(Dq[:NSITE, NSITE:]) + nrm(Dq[NSITE:, :NSITE])) / nrm(X1[x]),
                 "X_anom": nrm(Xq[:NSITE, NSITE:]) + nrm(Xq[NSITE:, :NSITE]),
                 "X1_pp_eigs": eigs(Xq[:NSITE, :NSITE]), "M1_pp_eigs": eigs(Mq[:NSITE, :NSITE])}
        # native one-particle block of M1_x[h_nat(psi)] on the positive eigenvectors of h_nat(psi)
        hn = L.native_h(lat, m, psi)
        wn, Vn = np.linalg.eigh(L.herm(hn))
        Vpos = Vn[:, wn > 0]
        Mn = L.native_midpoint(lat, m, psi, x)
        Mnb = L.herm(Vpos.conj().T @ Mn @ Vpos)
        wl, vl = np.linalg.eigh(Mnb)
        ob[x]["nat_pp_eigs"] = [float(v) for v in wl]
        ob[x]["nat_witness_energy"] = float(np.real(vl[:, 0].conj() @ np.diag(wn[wn > 0]) @ vl[:, 0]))
        ob[x]["nat_pos_spectrum"] = [float(v) for v in wn[wn > 0]]
    rec["onebody"] = {str(x): ob[x] for x in range(NSITE)}
    rec["eps"] = [float(v) for v in law.eps]
    # band-bottom pair (uniform backgrounds): the two lowest eps
    if np.allclose(psi, psi[0]):
        idx = np.argsort(law.eps)[:2]
        bb = {}
        for x in range(NSITE):
            Xr = law.rbasis(law.X(x))[np.ix_(idx, idx)]
            Mr = (law.vp[:, idx].conj().T @ M1[x] @ law.vp[:, idx])
            bb[str(x)] = {"diff_norm": nrm(Xr - Mr), "rhoP_norm": nrm(Xr), "rhoP": [[complex(v) for v in row] for row in Xr],
                          "rhoM": [[complex(v) for v in row] for row in Mr]}
        rec["band_bottom"] = {k: {"diff_norm": v["diff_norm"], "rhoP_norm": v["rhoP_norm"]} for k, v in bb.items()}
    mats = {}
    for N in sectors:
        sec = fock.sector(N)
        q = QOF[N]
        Gm = gams[N]
        co = ns.Coefficients(lat, fock, N, m, U, Gfrozen, "wick")
        co0 = ns.Coefficients(lat, fock, N, m, 0.0, Gfrozen, "wick")
        Hexc = comp.H_exc(N)
        Hfree = comp.H_free(N)
        Vstar = comp.Vstar[N]
        S = comp.S[N]
        proj = comp.proj[N]
        rhoP = {x: comp.rhoP(x, N) for x in range(NSITE)}
        rhoP0 = {x: comp.rhoP(x, N, dressed=False) for x in range(NSITE)}
        rhoM = {}
        for x in range(NSITE):
            rhoM[x], _ = L.normal_ordered_bilinear(sec, M1[x], Om, N0, fock)
        enat = {x: Gm @ co.scalar_source(x, psi) @ Gm.conj().T for x in range(NSITE)}
        enat0 = {x: Gm @ co0.scalar_source(x, psi) @ Gm.conj().T for x in range(NSITE)}
        cnat = {x: psi[x] ** -6 * (Gm @ co.V[x] @ Gm.conj().T) for x in range(NSITE)}
        Hnat_m = sum(enat[x] for x in range(NSITE))
        sr = {"q": q, "N": N, "dim": sec.d, "sites": {}}
        # controls per sector
        cs = {}
        cs["C5c_S_nexc_commutator"] = nrm(S @ comp.nex[N] - comp.nex[N] @ S)
        P01 = np.hstack([proj[N][k] for k in proj[N] if k <= 1]) if any(k <= 1 for k in proj[N]) else None
        cs["C5c_S_on_nexc_le1"] = 0.0 if P01 is None else nrm(S @ P01)
        cs["C5c_S_hermitian"] = nrm(S - S.conj().T)
        cs["C5d_monomials_minus_Vstar"] = nrm(comp.monomial_sum(N) - Vstar)
        cs["C5e_Hexc_minus_Hfree_on_nexc_le1"] = 0.0 if P01 is None else nrm(compress(Hexc - Hfree, P01))
        cs["C5e_min_eig_Hexc"] = eigs(Hexc)[0]
        cs["C5e_norm_Hexc_minus_Hfree_minus_Vstar"] = nrm(Hexc - Hfree - Vstar)
        cs["C5e_spec_Hexc_minus_Hfree_minus_Vstar"] = snrm(Hexc - Hfree - Vstar)
        # C4: c-basis form of the free clock derivative
        worst4 = 0.0
        for x in range(NSITE):
            cb = sec.bilinear(X1[x]) + np.trace(law.X(x)).real * np.eye(sec.d)
            worst4 = max(worst4, nrm(cb - rhoP0[x]))
        cs["C4_cbasis_vs_qp"] = worst4
        # C8a: native scalar source at U = 0 equals the frozen-sea bilinear of the native midpoint
        worst8 = 0.0
        for s_test in (np.ones(NSITE), np.array([1.1, 0.9, 1.05, 0.95])):
            for x in range(NSITE):
                Mx = L.native_midpoint(lat, m, s_test, x)
                const = np.sum(Mx * Gfrozen)
                bil = sec.bilinear(Mx) - const * np.eye(sec.d)
                worst8 = max(worst8, nrm(co0.scalar_source(x, s_test) - bil))
        cs["C8a_native_identification"] = worst8
        # C8b at B0, U = 0
        if name == "B0":
            cs["C8b_rhoM_vs_mapped_native"] = max(nrm(rhoM[x] - enat0[x]) for x in range(NSITE))
        # T6 sum rules
        cs["T6_sum_rhoP_minus_Hexc_rel"] = nrm(sum(law.N[x] * rhoP[x] for x in range(NSITE)) - Hexc) / max(nrm(Hexc), 1e-300)
        if np.allclose(law.N, 1.0):
            cs["T6_sum_rhoM_minus_Hfree_rel"] = nrm(sum(rhoM.values()) - Hfree) / max(nrm(Hfree), 1e-300)
        cs["T6_sum_enat_minus_Hnatm_rel"] = nrm(sum(enat.values()) - Hnat_m) / max(nrm(Hnat_m), 1e-300)
        sr["controls"] = cs
        # n_exc blocks
        blocks = {}
        for (npp, nh), k in BLOCKS[N].items():
            if k in proj[N]:
                blocks["%d,%d" % (npp, nh)] = proj[N][k]
        sr["block_dims"] = {b: int(P.shape[1]) for b, P in blocks.items()}
        # energies (T7)
        sr["E_Hexc_lowest"] = eigs(Hexc)[0]
        sr["E_Hexc_block_lowest"] = {b: eigs(compress(Hexc, P))[0] for b, P in blocks.items()}
        sr["E_Hnatm_lowest"] = eigs(Hnat_m)[0]
        # lowest state of H_exc and its block
        Elow, deglow, rholow = L.lowest_state_dm(Hexc)
        lowblock = None
        for b, P in blocks.items():
            if np.trace(compress(rholow, P)).real > 0.5:
                lowblock = b
        sr["lowest_state"] = {"E": Elow, "deg": deglow, "block": lowblock}
        # C10 at B0, U = 0.1
        if name == "B0" and U == 0.1:
            Dn = Hnat_m - Hfree
            Pc = np.zeros_like(Dn)
            for k, P in proj[N].items():
                Pc += P @ (P.conj().T @ Dn @ P) @ P.conj().T
            cs["C10_cons_part_minus_Vstar"] = nrm(Pc - Vstar)
            cs["C10_number_changing_norm"] = nrm(Dn - Pc)
            cs["C10_Vstar_norm"] = nrm(Vstar)
        if name == "B0":
            sr["lower_face"] = {"min_eig_enat_sector": min(eigs(enat[x])[0] for x in range(NSITE)),
                                "min_eig_rhoP_sector": min(eigs(rhoP[x])[0] for x in range(NSITE))}
        # per site
        for x in range(NSITE):
            st = {}
            st["rhoP_eig"] = [eigs(rhoP[x])[0], eigs(rhoP[x])[-1]]
            st["contact_eig"] = [eigs(rhoP[x] - rhoP0[x])[0], eigs(rhoP[x] - rhoP0[x])[-1]]
            st["rhoM_eig"] = [eigs(rhoM[x])[0], eigs(rhoM[x])[-1]]
            st["enat_eig"] = [eigs(enat[x])[0], eigs(enat[x])[-1]]
            st["rhoP_block_min"] = {b: (eigs(compress(rhoP[x], P))[0] if P.shape[1] else None) for b, P in blocks.items()}
            st["rhoM_block_min"] = {}
            for b, P in blocks.items():
                Mb = L.herm(compress(rhoM[x], P))
                wb, vb = np.linalg.eigh(Mb)
                v = P @ vb[:, 0]
                st["rhoM_block_min"][b] = {"min": float(wb[0]), "energy_Hfree": fl(v.conj() @ Hfree @ v),
                                           "energy_Hexc": fl(v.conj() @ Hexc @ v),
                                           "rhoP_in_state": fl(v.conj() @ rhoP[x] @ v),
                                           "rhoP0_in_state": fl(v.conj() @ rhoP0[x] @ v)}
            st["vac_rhoP"] = fl(Om.conj() @ rhoP[x] @ Om) if N == N0 else None
            # T3 operator differences
            Dpm, Dmn, Dpn = rhoP0[x] - rhoM[x], rhoM[x] - enat0[x], rhoP[x] - enat[x]
            st["Delta_PM"] = L.rel_norms(Dpm, rhoP[x])
            st["Delta_MN"] = L.rel_norms(Dmn, rhoP[x])
            st["Delta_PN"] = L.rel_norms(Dpn, rhoP[x])
            cdiff = (rhoP[x] - rhoP0[x]) - cnat[x]
            st["contact_diff"] = L.rel_norms(cdiff, rhoP[x])
            if lowblock is not None:
                P = blocks[lowblock]
                ref = compress(rhoP[x], P)
                st["lowblock"] = {"Delta_PM": snrm(compress(Dpm, P)) / max(snrm(ref), 1e-300),
                                  "Delta_MN": snrm(compress(Dmn, P)) / max(snrm(ref), 1e-300),
                                  "contact_diff": snrm(compress(cdiff, P)) / max(snrm(ref), 1e-300),
                                  "Delta_PN": snrm(compress(Dpn, P)) / max(snrm(ref), 1e-300)}
            # T4 contacts in the 2-excitation blocks
            t4 = {}
            for b in ("2,0", "2,1", "1,1"):
                if b in blocks:
                    P = blocks[b]
                    Elb, dgb, rhob = L.lowest_state_dm(Hexc, P=P)
                    cl = rhoP[x] - rhoP0[x]
                    cb = compress(cl, P)
                    nb = compress(cnat[x], P)
                    t4[b] = {"E": Elb, "deg": dgb, "law_contact": fl(np.trace(rhob @ cl)),
                             "native_contact": fl(np.trace(rhob @ cnat[x])),
                             "rel_diff_spec": snrm(cb - nb) / max(snrm(cb), 1e-300),
                             "rel_diff_fro": nrm(cb - nb) / max(nrm(cb), 1e-300),
                             "law_contact_block_norm": nrm(cb)}
            st["T4"] = t4
            # one-particle block in q = +1 (N = 5): eigenvalues and the witness (T1, T2)
            if N == 5 and "1,0" in blocks:
                P = blocks["1,0"]
                st["onepart_rhoP_eigs"] = eigs(compress(rhoP[x], P))
                st["onepart_rhoM_eigs"] = eigs(compress(rhoM[x], P))
                Mb = L.herm(compress(rhoM[x], P))
                wb, vb = np.linalg.eigh(Mb)
                v = P @ vb[:, 0]
                st["witness"] = {"rhoM": float(wb[0]), "energy": fl(v.conj() @ Hfree @ v),
                                 "rhoP": fl(v.conj() @ rhoP[x] @ v)}
            sr["sites"][str(x)] = st
        if eS is not None and N == eS["N"]:
            sr["eS_eig"] = eS["eig"]
        rec["sectors"][str(N)] = sr
        if save:
            mats.update({"N%d_Hexc" % N: Hexc, "N%d_Hfree" % N: Hfree, "N%d_Vstar" % N: Vstar, "N%d_Sstar" % N: S})
            for x in range(NSITE):
                mats.update({"N%d_rhoP_x%d" % (N, x): rhoP[x], "N%d_rhoP0_x%d" % (N, x): rhoP0[x],
                             "N%d_rhoM_x%d" % (N, x): rhoM[x], "N%d_enat_x%d" % (N, x): enat[x],
                             "N%d_enat0_x%d" % (N, x): enat0[x], "N%d_cnat_x%d" % (N, x): cnat[x]})
    if save and matdir is not None:
        np.savez_compressed(os.path.join(matdir, "law_%s_m%g_U%g.npz" % (name, m, U)), **mats)
    rec["seconds"] = time.time() - t0
    return rec, comp


# ------------------------------------------------------------------ C6 and the Euler control
def euler_and_fd(fock, m=0.1, U=0.1):
    out = []
    x = np.arange(NSITE)
    Neul = 1 + 0.3 * np.cos(np.pi * x / 2)
    for name, psi in (("B0", np.ones(NSITE)), ("B2_e0.2", 1 + 0.2 * np.cos(np.pi * x / 2))):
        lawE = L.Law(m, psi, N=Neul)
        comp = L.Completion(lawE, fock, U, sectors=range(2, 8), dress=SECTORS)
        rec = {"bg": name, "m": m, "U": U, "N": list(map(float, Neul))}
        for N in SECTORS:
            H = comp.H_exc(N)
            rho = [comp.rhoP(xx, N) for xx in range(NSITE)]
            rec["euler_rel_N%d" % N] = nrm(sum(Neul[xx] * rho[xx] for xx in range(NSITE)) - H) / nrm(H)
            fd = {}
            for h in (1e-3, 1e-4):
                worst = 0.0
                fds = []
                for xx in range(NSITE):
                    Np, Nm = Neul.copy(), Neul.copy()
                    Np[xx] += h
                    Nm[xx] -= h
                    Bp = L.herm(lawE.E0h @ np.diag(Np * psi) @ lawE.E0h)
                    Bm = L.herm(lawE.E0h @ np.diag(Nm * psi) @ lawE.E0h)
                    d = (comp.H_exc(N, B0=Bp) - comp.H_exc(N, B0=Bm)) / (2 * h)
                    fds.append(d)
                    worst = max(worst, nrm(d - rho[xx]))
                fd[h] = (worst, fds)
            rec["C6_fd_vs_rhoP_h1e-3_N%d" % N] = fd[1e-3][0]
            rec["C6_fd_vs_rhoP_h1e-4_N%d" % N] = fd[1e-4][0]
            rec["C6_fd_h1e-3_vs_h1e-4_N%d" % N] = max(nrm(a - b) for a, b in zip(fd[1e-3][1], fd[1e-4][1]))
        # C4b at the Euler lapse
        sec4 = fock.sector(comp.N0)
        w4, V4 = np.linalg.eigh(L.herm(sec4.bilinear(lawE.h_star())))
        rec["C4b_Evac_minus_minusTrB0"] = float(w4[0] + np.trace(lawE.B0).real)
        rec["C4b_overlap_minus_1"] = float(abs(abs(np.vdot(V4[:, 0], comp.Omega)) - 1.0))
        rec["sum_Nw"] = float(sum(Neul[xx] * psi[xx] * (lawE.E1[xx, xx] - lawE.E0[xx, xx]).real for xx in range(NSITE)))
        rec["TrL_E1_minus_E0"] = float(np.trace(np.diag(Neul * psi) @ (lawE.E1 - lawE.E0)).real)
        out.append(rec)
    return out


# ------------------------------------------------------------------ C11 (first-order contact formula)
def c11(fock, m=0.1, Usmall=1e-4):
    out = []
    x = np.arange(NSITE)
    for name, psi in (("B0", np.ones(NSITE)), ("B2_e0.2", 1 + 0.2 * np.cos(np.pi * x / 2))):
        law = L.Law(m, psi)
        comp = L.Completion(law, fock, Usmall, sectors=range(2, 8), dress=SECTORS)
        for N in SECTORS:
            for xx in range(NSITE):
                num = (comp.rhoP(xx, N) - comp.rhoP(xx, N, dressed=False)) / Usmall
                form = comp.contact1(xx, N) / Usmall
                out.append({"bg": name, "N": N, "x": xx, "rel_diff": nrm(num - form) / max(nrm(form), 1e-300),
                            "form_norm": nrm(form)})
    return out


# ------------------------------------------------------------------ T5 native c-numbers and vacuum residuals
def native_cnumbers(lat, fock, m, U, s, Gfrozen):
    """c_x^nat(s) by the spec formula, checked against <sea_1| unsubtracted source |sea_1> (at s given)."""
    sec4 = fock.sector(4)
    h1 = m * lat["beta"] + lat["D"]
    w, V = np.linalg.eigh(L.herm(sec4.bilinear(h1)))
    sea = V[:, 0]
    out = []
    for x in range(NSITE):
        a, b = 2 * x, 2 * x + 1
        cm = m * sum(nc.SZ[i, j] * Gfrozen[2 * x + i, 2 * x + j] for i in range(2) for j in range(2))
        co = ns.Coefficients(lat, fock, 4, m, U, Gfrozen, "wick")
        cb = 0.0
        for (i, j) in co.site_bonds(x):
            cb += np.real(co.Bsea[(i, j)]) / (s[i] * s[j])
        pair = Gfrozen[a, a] * Gfrozen[b, b] - Gfrozen[a, b] * Gfrozen[b, a]
        cformula = cm + cb + s[x] ** -6 * U * pair
        Mx = L.native_midpoint(lat, m, s, x)
        unsub = sec4.bilinear(Mx) + s[x] ** -6 * U * (sec4.number(a) @ sec4.number(b))
        cexp = sea.conj() @ unsub @ sea
        out.append({"x": x, "c_formula": fl(cformula), "c_expect": fl(cexp), "diff": float(abs(cformula - cexp))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stored-jobs", required=True)
    ap.add_argument("--stored-kscan", required=True)
    ap.add_argument("--quick", action="store_true", help="B0 and B2_e0.2 at m = 0.1 only (smoke test)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "mats"), exist_ok=True)
    t0 = time.time()
    meta = {"host": socket.gethostname(), "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "threads": os.environ.get("OMP_NUM_THREADS"),
            "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    print("strand0 part A on %s" % meta["host"], flush=True)
    lat = nc.lattice("ring4", (0.0, 0.0, 0.0))
    fock = ns.Fock(NMODE)
    out = {"meta": meta, "controls": {}}
    # C1
    Wname, W, c1 = choose_W(0.1)
    out["controls"]["C1_W_choice"] = Wname
    out["controls"]["C1_residuals"] = c1
    Wf = np.kron(np.eye(NSITE), W)
    gams, gc = gamma_controls(Wf, fock, lat, 0.1)
    out["controls"].update(gc)
    print("C1 done: %s %s" % (Wname, c1), flush=True)
    # native re-solves
    stored = stored_energies(args.stored_jobs, args.stored_kscan)
    natrecs = native_resolves(stored, os.path.join(args.out, "mats"))
    out["native"] = natrecs
    print("native re-solves done: %d, ok %d (%.0f s)" % (len(natrecs), sum(r["ok"] for r in natrecs), time.time() - t0),
          flush=True)
    # T5 native c-numbers (s = 1 and the B1 sbar) and vacuum residuals
    t5n = []
    for m in MASSES:
        Gfrozen, nsea = ns.sea_contractions(lat, m, NMODE)
        for U in (0.0, 0.1):
            t5n.append({"m": m, "U": U, "s": "1", "rows": native_cnumbers(lat, fock, m, U, np.ones(NSITE), Gfrozen)})
    out["T5_native_cnumbers"] = t5n
    vac = []
    for r in natrecs:
        if not r["ok"] or r["q"] != 0:
            continue
        Gfrozen, nsea = ns.sea_contractions(lat, r["m"], NMODE)
        sbar = np.array(r["sbar"])
        co = ns.Coefficients(lat, fock, 4, r["m"], r["U"], Gfrozen, "wick")
        Hm = sum(co.scalar_source(x, sbar) for x in range(NSITE))
        co0 = ns.Coefficients(lat, fock, 4, r["m"], 0.0, Gfrozen, "wick")
        Hm0 = sum(co0.scalar_source(x, sbar) for x in range(NSITE))
        w0, V0 = np.linalg.eigh(L.herm(Hm0))
        Om_nat = V0[:, 0]
        vac.append({"prom": r["prom"], "m": r["m"], "U": r["U"], "E_HnatS_q0": r["E0"],
                    "E_Hnatm_sbar_N4": eigs(Hm)[0], "E_Hnatm0_sbar_N4": float(w0[0]),
                    "vac_enat_per_site": [fl(Om_nat.conj() @ co0.scalar_source(x, sbar) @ Om_nat) for x in range(NSITE)],
                    "sbar": r["sbar"]})
    out["T5_native_vacuum"] = vac
    # law cases
    cases = []
    bgs = fixed_backgrounds()
    if args.quick:
        bgs = {k: bgs[k] for k in ("B0", "B2_e0.2")}
    masses = (0.1,) if args.quick else MASSES
    for name, psi in bgs.items():
        for m in masses:
            Gfrozen, nsea = ns.sea_contractions(lat, m, NMODE)
            for U in UVALS:
                rec, _ = law_case(name, psi, m, U, fock, gams, lat, Gfrozen, matdir=os.path.join(args.out, "mats"))
                rec["report_sites"] = REPORT_SITES[name]
                cases.append(rec)
                print("  %s m=%g U=%g done (%.1f s)" % (name, m, U, rec["seconds"]), flush=True)
    # B1 backgrounds
    for r in natrecs:
        if not r["ok"]:
            continue
        if not ((r["prom"] == "J_W6" and r["U"] in (0.0, 0.1)) or (r["prom"] == "END" and r["U"] == 0.1)):
            continue
        if args.quick and r["m"] != 0.1:
            continue
        Gfrozen, nsea = ns.sea_contractions(lat, r["m"], NMODE)
        name = "B1_%s_q%d" % (r["prom"], r["q"])
        N = 4 + r["q"]
        eS = {"N": N, "eig": {"min": r["eS_min"], "max": r["eS_max"]}}
        rec, _ = law_case(name, np.array(r["sbar"]), r["m"], r["U"], fock, gams, lat, Gfrozen, sectors=SECTORS,
                          eS=eS, matdir=os.path.join(args.out, "mats"))
        rec["report_sites"] = [0]
        rec["native_sector"] = N
        cases.append(rec)
        print("  %s m=%g U=%g done (%.1f s)" % (name, r["m"], r["U"], rec["seconds"]), flush=True)
    out["cases"] = cases
    # contact scaling with U (T4): needs U = 0.01 and 0.1 on the same fixed background
    scal = []
    for name in bgs:
        for m in masses:
            fA = os.path.join(args.out, "mats", "law_%s_m%g_U%g.npz" % (name, m, 0.01))
            fB = os.path.join(args.out, "mats", "law_%s_m%g_U%g.npz" % (name, m, 0.1))
            A, B = np.load(fA), np.load(fB)
            for N in SECTORS:
                for x in range(NSITE):
                    cA = A["N%d_rhoP_x%d" % (N, x)] - A["N%d_rhoP0_x%d" % (N, x)]
                    cB = B["N%d_rhoP_x%d" % (N, x)] - B["N%d_rhoP0_x%d" % (N, x)]
                    scal.append({"bg": name, "m": m, "N": N, "x": x, "ratio": nrm(cB) / max(nrm(cA), 1e-300),
                                 "second_order": nrm(cB - 10 * cA) / 0.01})
                RA = A["N%d_Hexc" % N] - A["N%d_Hfree" % N] - A["N%d_Vstar" % N]
                RB = B["N%d_Hexc" % N] - B["N%d_Hfree" % N] - B["N%d_Vstar" % N]
                scal.append({"bg": name, "m": m, "N": N, "x": "total", "C5e_ratio_U0.01_over_U0.1": nrm(RA) / max(nrm(RB), 1e-300)})
    out["contact_scaling"] = scal
    print("contact scaling done", flush=True)
    out["euler_fd"] = euler_and_fd(fock)
    print("Euler and C6 done", flush=True)
    out["C11"] = c11(fock)
    print("C11 done", flush=True)
    out["meta"]["seconds"] = time.time() - t0
    out["meta"]["end_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(os.path.join(args.out, "partA.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o))
    print("strand0 part A done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
