#!/usr/bin/env python
"""dust_tail_v1.py (busi lane, reviewer A; inbox #48 item 2: the shed dust's second route, read in place from the solve lane's stored
continuation slices with this lane's own code; the first route is solve/code/engine/dust_tail_v1.py with engine_massive_channel_D_v2's
stored jsons, whose numbers are quoted beside and never averaged).

Per stored slice (an orbitals npz: T, rs, R, N_applied, v_applied, kappas, weights, dtype, sea_k, dust_k, f_k, R2_k, e_k, mask_k) and per
channel kappa (weight w = 16 kappa), with the Killing Hamiltonian h_m = (1/2){N, h_0} + (1/2){v, Q} at the slice's applied fields
(dust_tail_ops_v1 builds it from the definitions):
- the dust (the stored dust columns with their occupations f; the tagged quantities of the first route): the site density, the bare energy and
  flow densities (h_0, Q), the symmetrized Killing density (h_m) and the number current through every cut; beyond each cut R_c (sites with
  R > R_c): the number N, the Killing energy E_sym (the symmetrized density summed over the sites beyond; the first route's E_out_dust), the
  density route K = sum (N eps + v J) (the engine's own source convention; checked against the run's stored shell totals when --run is given),
  the exterior block's Killing spectrum (particles omega > 0, holes omega < 0, their energies; sum n = N and sum omega n = the block energy
  as identities), the energy per particle in units of m (E_particles / particles, the first route's form, and E_sym / N), the occupation by
  omega/m band, and the number current F through the cut (outward positive);
- the sea (the stored propagated sea, f = 1): its site density against 1, its excess number beyond each cut and its current through each cut
  (the structural fact: all three vanish, so the number beyond a cut is the dust's alone);
- the spectrum of this construction's h_m against the slice's stored eigenvalues e_k (the operator check), and with --pairs the Killing
  vacuum P_- (the negative-energy eigenvectors of h_m) with the sea-subtracted energy beyond each cut (the pairs; optional, one dense
  eigendecomposition with vectors per channel).
Outputs (COMMS rule 10): per slice <out>/<tag>_T<T/rs>.json (every number) and .npz (the per-site and per-level arrays); <tag>_all.json;
<tag>_table.md (dust_tail_table_v1: the tables, the first route's numbers beside where its jsons are given, the checks, the provenance).
Usage: --slices a.npz,b.npz --m 0.2 --out-dir DIR --tag TAG [--cuts 16,24] [--profile 4,8,12,16,20,24,28] [--run DIR/TAG of the collapse
json and fields npz] [--first-route-tail-json D_dust_tail_v1.json] [--first-route-dv2-jsons Dv2_a.json,...] [--spectrum dust|all|none]
[--pairs] [--threads 4] [--overwrite]; --selftest (dust_tail_selftest_v1); --check SLICE.npz."""
import os
import sys

_T = "4"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _T = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_tail_ops_v1 as OPS  # noqa: E402
import dust_tail_table_v1 as TBL  # noqa: E402

M_EDGES = (-math.inf, -3.0, -2.0, -1.5, -1.1, -1.0, -0.9, 0.0, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, math.inf)   # omega/m bands (the first route's edges)
fmt = TBL.fmt
utc = TBL.utc


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def npz_structure(path):
    """the members of an npz (name, shape, dtype, bytes) from the zip headers, without loading any data."""
    out = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            with z.open(info) as fp:
                ver = np.lib.format.read_magic(fp)
                rd = getattr(np.lib.format, "read_array_header_%d_%d" % ver, None)
                shape, _fo, dt = rd(fp) if rd is not None else np.lib.format._read_array_header(fp, ver)
            out.append((info.filename, tuple(shape), str(dt), info.file_size))
    return out


def load_run(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Zf = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    return J, Zf


def load_first_route(tail_json, dv2_jsons):
    first = {"tail": {}, "dv2": {}}
    if tail_json:
        for r in json.load(open(tail_json)).get("rows", []):
            first["tail"][round(float(r["T_over_rs"]), 6)] = r
    for p in [x for x in (dv2_jsons or "").split(",") if x]:
        for sl in json.load(open(p)).get("slices", []):
            first["dv2"][round(float(sl["T_over_rs"]), 6)] = {int(c["kappa"]): c for c in sl["channels"]}
    return first


def bands(om, nl, m):
    out = []
    x = om / m
    for lo, hi in zip(M_EDGES[:-1], M_EDGES[1:]):
        s = (x >= lo) & (x < hi)
        out.append({"lo": lo, "hi": hi, "n": float(nl[s].sum()), "E": float((om[s] * nl[s]).sum())})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ one slice
def process_slice(path, m, cuts_rs, prof_rs, spectrum, pairs, log=print):
    t_s = time.time()
    Z = np.load(path)
    T = float(Z["T"]); rs = float(Z["rs"]); R = np.asarray(Z["R"], float); n = len(R)
    Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
    kappas = [int(k) for k in Z["kappas"]]; wts = {int(k): float(w) for k, w in zip(Z["kappas"], Z["weights"])}
    cuts = [(c, OPS.cut_index(R, c * rs)) for c in cuts_rs]
    prof = [(c, OPS.cut_index(R, c * rs)) for c in prof_rs]
    row = {"slice": path, "T": T, "T_over_rs": T / rs, "rs": rs, "m": m, "nsites": n, "dtype": str(Z["dtype"]) if "dtype" in Z.files else None,
           "cuts": [{"R_c_over_rs": c, "R_c": c * rs, "first_site_index": p, "R_first": float(R[p]), "sites_beyond": int(n - p)} for c, p in cuts],
           "kappas": kappas, "weights": wts, "channels": []}
    arrays = {"R": R, "N_applied": Napp, "v_applied": vapp}
    log("%s: T/rs %.3f (%s), %d sites, cuts (R_c/r_s, first site index beyond) %s" % (os.path.basename(path), T / rs, row["dtype"], n, [(c, p) for c, p in cuts]))
    for k in kappas:
        t0 = time.time()
        hm = OPS.killing_h(R, m, k, Napp, vapp)
        h0, Q = OPS.chain_operators(R, m, k)
        dust = np.asarray(Z["dust_%d" % k], np.complex128); f = np.asarray(Z["f_%d" % k], float); nd = int(dust.shape[1])
        sea = np.asarray(Z["sea_%d" % k], np.complex128)
        ch = {"kappa": k, "w": wts[k], "n_sea": int(sea.shape[1]), "n_dust": nd, "f": f.tolist()}
        rho_sea = OPS.site_density(sea)
        ch["sea_density_max_dev"] = float(np.max(np.abs(rho_sea - 1.0)))
        ch["sea_norm_max_dev"] = float(np.max(np.abs(np.sum(np.abs(sea) ** 2, axis=0) - 1.0)))
        ch["sea_number_total"] = float(rho_sea.sum())
        ch["sea_excess_beyond"] = {str(c): float(np.sum(rho_sea[p:] - 1.0)) for c, p in cuts}
        ch["sea_current"] = {str(c): OPS.cut_current_at(hm, sea, None, p) for c, p in cuts}
        arrays["rho_sea_minus_1_%d" % k] = rho_sea - 1.0
        if nd:
            G = dust.conj().T @ dust
            ch["dust_gram_max_dev"] = float(np.max(np.abs(G - np.eye(nd))))
            ch["dust_sea_overlap_max"] = float(np.max(np.abs(sea.conj().T @ dust)))
        ksym_sea = OPS.op_density(hm, sea) if pairs else None
        del sea
        if spectrum == "all" or (spectrum == "dust" and nd) or pairs:
            if pairs:
                e_mine, V = OPS.full_spectrum(hm, vectors=True)
                Vn = V[:, e_mine < 0]; del V
                rho_vac = OPS.site_density(Vn); ksym_vac = OPS.op_density(hm, Vn); del Vn
                ch["vacuum_density_max_dev"] = float(np.max(np.abs(rho_vac - 1.0)))
                ch["pairs_energy_beyond"] = {str(c): float(np.sum(ksym_sea[p:] - ksym_vac[p:])) for c, p in cuts}
                ch["pairs_energy_total"] = float(np.sum(ksym_sea - ksym_vac))
                arrays["ksym_pairs_%d" % k] = ksym_sea - ksym_vac
            else:
                e_mine = OPS.full_spectrum(hm)
            arrays["e_%d" % k] = e_mine
            ch["n_negative"] = int(np.sum(e_mine < 0)); ch["spectrum_symmetry_max_dev"] = float(np.max(np.abs(e_mine + e_mine[::-1])))
            if ("e_%d" % k) in Z.files:
                ch["spectrum_vs_stored_max_dev"] = float(np.max(np.abs(np.sort(e_mine) - np.sort(np.asarray(Z["e_%d" % k], float)))))
        ch["beyond"] = {}
        if nd:
            rho = OPS.site_density(dust, f); eps0 = OPS.op_density(h0, dust, f); J = OPS.op_density(Q, dust, f); ksym = OPS.op_density(hm, dust, f)
            Fall = OPS.cut_currents_all(Napp, vapp, dust, f)
            arrays.update({"rho_dust_%d" % k: rho, "eps0_dust_%d" % k: eps0, "J_dust_%d" % k: J, "ksym_dust_%d" % k: ksym, "F_dust_%d" % k: Fall})
            ch["dust_number_total"] = float(rho.sum()); ch["dust_bare_energy_total"] = float(eps0.sum()); ch["dust_flow_total"] = float(J.sum())
            ch["dust_killing_energy_total"] = float(ksym.sum()); ch["dust_density_route_total"] = float(np.sum(Napp * eps0 + vapp * J))
            for c, p in cuts:
                reg = {"N": float(rho[p:].sum()), "E_sym": float(ksym[p:].sum()), "K_density": float(np.sum(Napp[p:] * eps0[p:] + vapp[p:] * J[p:])),
                       "eps0": float(eps0[p:].sum()), "J": float(J[p:].sum()), "F": float(Fall[p - 1]), "F_sparse": OPS.cut_current_at(hm, dust, f, p)}
                om, phi = OPS.exterior_spectrum(hm, p)
                nl = OPS.occupations_in_basis(phi, dust[2 * p:], f); del phi
                pos = om > 0
                reg.update({"particles": float(nl[pos].sum()), "holes": float(nl[~pos].sum()), "E_particles": float((om[pos] * nl[pos]).sum()),
                            "E_holes": float((om[~pos] * nl[~pos]).sum()), "E_block": float((om * nl).sum()), "E_block_direct": OPS.block_energy(hm, dust[2 * p:], p, f),
                            "omega_range": [float(om.min()), float(om.max())], "omega_min_positive": float(om[pos].min()) if pos.any() else None,
                            "omega_max_negative": float(om[~pos].max()) if (~pos).any() else None, "levels": int(len(om)), "bands_m": bands(om, nl, m)})
                reg["sum_n_minus_N"] = reg["particles"] + reg["holes"] - reg["N"]; reg["block_identity"] = reg["E_block"] - reg["E_block_direct"]
                reg["E_per_particle_over_m"] = (reg["E_particles"] / reg["particles"] / m) if reg["particles"] > 0 else None
                reg["E_sym_per_N_over_m"] = (reg["E_sym"] / reg["N"] / m) if reg["N"] > 0 else None
                arrays["omega_%g_%d" % (c, k)] = om; arrays["n_%g_%d" % (c, k)] = nl
                ch["beyond"][str(c)] = reg
            ch["profile_N"] = {str(c): float(rho[p:].sum()) for c, p in prof}
            ch["profile_F"] = {str(c): float(Fall[p - 1]) for c, p in prof}
        ch["seconds"] = time.time() - t0
        row["channels"].append(ch)
        b = ch["beyond"].get(str(cuts_rs[0])) or {}
        log("  kappa %d (w %g, %d dust): sea density dev %.1e, excess beyond %g r_s %+.1e, current %+.1e; spectrum vs stored %s; dust beyond %g r_s: N %s, E_sym %s, K %s, "
            "particles %s (E %s; %s m each), holes %s; F %s; %.1f s" % (k, wts[k], nd, ch["sea_density_max_dev"], cuts_rs[0], ch["sea_excess_beyond"][str(cuts_rs[0])],
                                                                        ch["sea_current"][str(cuts_rs[0])], fmt(ch.get("spectrum_vs_stored_max_dev"), "%.1e"), cuts_rs[0],
                                                                        fmt(b.get("N")), fmt(b.get("E_sym")), fmt(b.get("K_density")), fmt(b.get("particles")), fmt(b.get("E_particles")),
                                                                        fmt(b.get("E_per_particle_over_m"), "%.4f"), fmt(b.get("holes")), fmt(b.get("F"), "%+.6e"), ch["seconds"]))
    Z.close()
    W = {}
    dch = [c for c in row["channels"] if c["n_dust"]]
    for c, p in cuts:
        key = str(c)
        regs = [(ch["w"], ch["beyond"][key]) for ch in dch]
        S = lambda q: float(sum(w * r[q] for w, r in regs))  # noqa: E731
        q = {x: S(x) for x in ("N", "E_sym", "K_density", "eps0", "J", "F", "particles", "holes", "E_particles", "E_holes", "E_block")}
        q["E_per_particle_over_m"] = (q["E_particles"] / q["particles"] / m) if q["particles"] > 0 else None
        q["E_sym_per_N_over_m"] = (q["E_sym"] / q["N"] / m) if q["N"] > 0 else None
        q["bands_m"] = [{"lo": b_["lo"], "hi": b_["hi"], "n": float(sum(w * r["bands_m"][i]["n"] for w, r in regs)), "E": float(sum(w * r["bands_m"][i]["E"] for w, r in regs))}
                        for i, b_ in enumerate(regs[0][1]["bands_m"])] if regs else []
        q["max_sum_n_minus_N"] = max((abs(r["sum_n_minus_N"]) for _, r in regs), default=0.0)
        q["max_block_identity"] = max((abs(r["block_identity"]) for _, r in regs), default=0.0)
        q["max_F_minus_F_sparse"] = max((abs(r["F"] - r["F_sparse"]) for _, r in regs), default=0.0)
        q["sea_excess_weighted"] = float(sum(ch["w"] * ch["sea_excess_beyond"][key] for ch in row["channels"]))
        q["sea_current_weighted"] = float(sum(ch["w"] * ch["sea_current"][key] for ch in row["channels"]))
        q["max_sea_excess"] = max(abs(ch["sea_excess_beyond"][key]) for ch in row["channels"])
        q["max_sea_current"] = max(abs(ch["sea_current"][key]) for ch in row["channels"])
        if pairs:
            q["pairs_energy"] = float(sum(ch["w"] * ch["pairs_energy_beyond"][key] for ch in row["channels"]))
        W[key] = q
    row["weighted"] = W
    row["profile_N_weighted"] = {str(c): float(sum(ch["w"] * ch["profile_N"][str(c)] for ch in dch)) for c, _ in prof}
    row["profile_F_weighted"] = {str(c): float(sum(ch["w"] * ch["profile_F"][str(c)] for ch in dch)) for c, _ in prof}
    row["dust_number_total_weighted"] = float(sum(ch["w"] * ch["dust_number_total"] for ch in dch))
    row["dust_number_nominal"] = float(sum(ch["w"] * sum(ch["f"]) for ch in dch))
    row["dust_bare_energy_total_weighted"] = float(sum(ch["w"] * ch["dust_bare_energy_total"] for ch in dch))
    row["dust_killing_energy_total_weighted"] = float(sum(ch["w"] * ch["dust_killing_energy_total"] for ch in dch))
    row["dust_density_route_total_weighted"] = float(sum(ch["w"] * ch["dust_density_route_total"] for ch in dch))
    row["checks"] = {"max_sea_density_dev": max(ch["sea_density_max_dev"] for ch in row["channels"]), "max_sea_norm_dev": max(ch["sea_norm_max_dev"] for ch in row["channels"]),
                     "max_dust_gram_dev": max((ch["dust_gram_max_dev"] for ch in dch), default=0.0), "max_dust_sea_overlap": max((ch["dust_sea_overlap_max"] for ch in dch), default=0.0),
                     "max_spectrum_vs_stored": max((ch["spectrum_vs_stored_max_dev"] for ch in row["channels"] if "spectrum_vs_stored_max_dev" in ch), default=None),
                     "max_spectrum_symmetry": max((ch["spectrum_symmetry_max_dev"] for ch in row["channels"] if "spectrum_symmetry_max_dev" in ch), default=None),
                     "n_negative": sorted({ch.get("n_negative") for ch in row["channels"] if "n_negative" in ch}),
                     "total_number_minus_nominal": row["dust_number_total_weighted"] - row["dust_number_nominal"],
                     "density_route_minus_killing_total": row["dust_density_route_total_weighted"] - row["dust_killing_energy_total_weighted"],
                     "max_vacuum_density_dev": max((ch["vacuum_density_max_dev"] for ch in row["channels"] if "vacuum_density_max_dev" in ch), default=None)}
    row["seconds"] = time.time() - t_s
    w0 = W[str(cuts_rs[0])]
    log("  weighted beyond %g r_s: N %.6e, E_sym %.6e, K %.6e, particles %.6e (E %.6e, %.4f m each; E_sym/N %.4f m), holes %.4e, F %+.6e; total dust %.6f (nominal %g); %.0f s" % (
        cuts_rs[0], w0["N"], w0["E_sym"], w0["K_density"], w0["particles"], w0["E_particles"], w0["E_per_particle_over_m"] or 0.0, w0["E_sym_per_N_over_m"] or 0.0, w0["holes"], w0["F"],
        row["dust_number_total_weighted"], row["dust_number_nominal"], row["seconds"]))
    return row, arrays


def run_checks(row, Napp, vapp, J, Zf, cuts_rs):
    """the slice against the run's records: E, T_H, the applied fields, the stored shell totals' dust energy beyond the cuts (the density
    route, weighted, as the engine stored it) against this readout's K_density, and the dust's total bare energy against E_dust_shells."""
    a = J["args"]; g = float(a["g"]); rs = float(a["rs"]); m_run = float(a["m"])
    T = row["T"]; Tz = np.asarray(Zf["T"], float); k = int(np.argmin(np.abs(Tz - T)))
    out = {"m_run": m_run, "m_used": row["m"], "g": g, "fields_record_index": k, "fields_T_mismatch": float(abs(Tz[k] - T))}
    if abs(Tz[k] - T) > 1e-6:
        out["note"] = "no fields record at the slice's T"
        return out
    rec = next((r for r in J["records"] if r.get("E") is not None and abs(float(r["T"]) - T) < 1e-6), None)
    out["applied_fields_max_dev"] = [float(np.max(np.abs(np.asarray(Zf["N_applied"][k], float) - Napp))), float(np.max(np.abs(np.asarray(Zf["v_applied"][k], float) - vapp)))]
    N = np.asarray(Zf["N_applied"][k], float); v = np.asarray(Zf["v_applied"][k], float); R = np.asarray(Zf["R"], float)
    ed = np.asarray(Zf["eps_tot"][k], float) - np.asarray(Zf["eps_pairs"][k], float); jd = np.asarray(Zf["J_tot"][k], float) - np.asarray(Zf["J_pairs"][k], float)
    out["stored_dust_bare_energy_total"] = float(ed.sum()); out["stored_dust_density_route_total"] = float(np.sum(N * ed + v * jd))
    out["beyond"] = {}
    for c in cuts_rs:
        s = R > c * rs
        Ks = float(np.sum(N[s] * ed[s] + v[s] * jd[s])); Km = row["weighted"][str(c)]["K_density"]
        out["beyond"][str(c)] = {"stored_K_dust": Ks, "stored_eps_dust": float(ed[s].sum()), "stored_J_dust": float(jd[s].sum()), "this_K_density": Km,
                                 "diff": Km - Ks, "rel": (Km - Ks) / abs(Ks) if Ks else None, "this_eps0": row["weighted"][str(c)]["eps0"], "this_J": row["weighted"][str(c)]["J"]}
    out["bare_energy_diff"] = row["dust_bare_energy_total_weighted"] - out["stored_dust_bare_energy_total"]
    if rec is not None:
        E = float(rec["E"]); TH = 1.0 / (8.0 * math.pi * g * E)
        out.update({"E_G": E, "two_gE": 2 * g * E, "T_H": TH, "m_over_TH": m_run / TH, "E_dust_shells": rec.get("E_dust_shells"), "E_pairs_shells": rec.get("E_pairs_shells"),
                    "R_t": rec.get("R_t"), "min_N_minus_absv": rec.get("min_N_minus_absv"), "continuation_label": rec.get("continuation_label"), "record_label": rec.get("record_label")})
        if rec.get("E_dust_shells") is not None:
            out["E_dust_shells_diff"] = row["dust_bare_energy_total_weighted"] - float(rec["E_dust_shells"])
    return out


# ------------------------------------------------------------------------------------------------------------------------------ main
def check(path):
    print("structure of %s (name, shape, dtype, bytes):" % path)
    for name, shape, dt, nb in npz_structure(path):
        print("  %-14s %-16s %-12s %d" % (name, shape, dt, nb))
    Z = np.load(path)
    T = float(Z["T"]); rs = float(Z["rs"]); R = np.asarray(Z["R"], float)
    print("T %g, T/rs %.4f, rs %g, dtype %s, sites %d (R %g .. %g), kappas %s, weights %s" % (T, T / rs, rs, str(Z["dtype"]) if "dtype" in Z.files else "?", len(R), R[0], R[-1],
                                                                                              [int(k) for k in Z["kappas"]], [float(w) for w in Z["weights"]]))
    tot = 0.0
    for k, w in zip(Z["kappas"], Z["weights"]):
        dust = np.asarray(Z["dust_%d" % k]); f = np.asarray(Z["f_%d" % k], float)
        if dust.shape[1] == 0:
            continue
        rho = OPS.site_density(dust, f); tot += float(w) * rho.sum()
        print("  kappa %d: %d dust columns, f %s, number %.12f; beyond 16 r_s %.6e, beyond 24 r_s %.6e (the dust columns alone; the sea is not read here)" % (
            int(k), dust.shape[1], f.tolist(), rho.sum(), rho[OPS.cut_index(R, 16 * rs):].sum(), rho[OPS.cut_index(R, 24 * rs):].sum()))
    print("weighted dust number %.9f" % tot)


def default_args(**kw):
    a = argparse.Namespace(slices="", m=0.2, out_dir="", tag="dust_tail_v1", cuts="16,24", profile="4,8,12,16,20,24,28", run="", first_route_tail_json="", first_route_dv2_jsons="",
                           spectrum="dust", pairs=False, threads=4, overwrite=False, selftest=False, check="")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def run_main(a, log=print):
    slices = [p for p in a.slices.split(",") if p]
    if not slices or not a.out_dir:
        sys.exit("--slices and --out-dir are required")
    os.makedirs(a.out_dir, exist_ok=True)
    cuts_rs = [float(x) for x in a.cuts.split(",")]; prof_rs = [float(x) for x in a.profile.split(",")]
    for name in ("%s_table.md" % a.tag, "%s_all.json" % a.tag):
        if os.path.exists(os.path.join(a.out_dir, name)) and not a.overwrite:
            sys.exit("refusing to overwrite %s (use a new --tag)" % os.path.join(a.out_dir, name))
    try:
        load = open("/proc/loadavg").read().split()[:3]
    except OSError:
        load = None
    prov = {"script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)), "modules": ", ".join("%s %s" % (x, md5(os.path.join(HERE, x))) for x in ("dust_tail_ops_v1.py", "dust_tail_table_v1.py")),
            "args": " ".join(sys.argv[1:]), "host": socket.gethostname(), "utc_start": utc(), "load_at_start": load, "threads": a.threads, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": sys.version.split()[0], "m": a.m, "spectrum": a.spectrum, "pairs": a.pairs}
    run = load_run(a.run) if a.run else None
    if run is not None and abs(float(run[0]["args"]["m"]) - a.m) > 1e-12:
        sys.exit("--m %g differs from the run's m %g" % (a.m, float(run[0]["args"]["m"])))
    first = load_first_route(a.first_route_tail_json, a.first_route_dv2_jsons)
    rows = []
    for p in slices:
        row, arrays = process_slice(p, a.m, cuts_rs, prof_rs, a.spectrum, a.pairs, log=log)
        if run is not None:
            row["run_checks"] = run_checks(row, arrays["N_applied"], arrays["v_applied"], run[0], run[1], cuts_rs)
            rb = row["run_checks"].get("beyond") or {}
            log("  run checks: applied fields dev %s; K_density - stored K_dust %s; bare energy - E_dust_shells %s; E_G %s, m/T_H %s" % (
                row["run_checks"].get("applied_fields_max_dev"), {c: "%+.3e (rel %s)" % (q["diff"], fmt(q["rel"], "%.2e")) for c, q in rb.items()}, fmt(row["run_checks"].get("E_dust_shells_diff"), "%+.3e"),
                fmt(row["run_checks"].get("E_G"), "%.6f"), fmt(row["run_checks"].get("m_over_TH"), "%.3f")))
        stem = os.path.join(a.out_dir, "%s_T%07.2f" % (a.tag, row["T_over_rs"]))
        if (os.path.exists(stem + ".json") or os.path.exists(stem + ".npz")) and not a.overwrite:
            sys.exit("refusing to overwrite %s.*" % stem)
        json.dump(row, open(stem + ".json", "w"), indent=1, default=float)
        np.savez(stem + ".npz", **arrays)
        rows.append(row)
        log("  wrote %s.json and .npz" % stem)
    rows.sort(key=lambda r: r["T_over_rs"])
    prov["utc_end"] = utc()
    md = TBL.compose_table(rows, first, cuts_rs, prof_rs, a.m, prov)
    open(os.path.join(a.out_dir, "%s_table.md" % a.tag), "w").write(md)
    json.dump({"script": "dust_tail_v1.py", "provenance": prov, "cuts_rs": cuts_rs, "profile_rs": prof_rs, "rows": rows}, open(os.path.join(a.out_dir, "%s_all.json" % a.tag), "w"), indent=1, default=float)
    log("wrote %s and %s" % (os.path.join(a.out_dir, "%s_table.md" % a.tag), os.path.join(a.out_dir, "%s_all.json" % a.tag)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=""); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--out-dir", default=""); ap.add_argument("--tag", default="dust_tail_v1")
    ap.add_argument("--cuts", default="16,24"); ap.add_argument("--profile", default="4,8,12,16,20,24,28"); ap.add_argument("--run", default="")
    ap.add_argument("--first-route-tail-json", default=""); ap.add_argument("--first-route-dv2-jsons", default=""); ap.add_argument("--spectrum", default="dust", choices=["dust", "all", "none"])
    ap.add_argument("--pairs", action="store_true"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--check", default="")
    a = ap.parse_args()
    if a.selftest:
        import dust_tail_selftest_v1 as ST
        sys.exit(0 if ST.selftest(sys.modules[__name__]) else 1)
    if a.check:
        check(a.check); return
    run_main(a)


if __name__ == "__main__":
    main()
