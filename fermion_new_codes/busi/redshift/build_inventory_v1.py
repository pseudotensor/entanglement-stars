"""busi REDSHIFT inputs (inbox #416), step 2: the inventory of the stored formation histories, and the registry the reader uses.

DATA script, box CPU only, read-only on every input; no redshift computed. Inputs (all in <internal>/redshift):
  census_v1.jsonl (census_v1.py over every *_fields.npz on the box), all_slices.txt (every orbitals_*.npz on the box),
  manifests.txt (md5 manifest files on the box), local_overflow_slices.txt + local_overflow_md5.txt (the local machine's
  <internal>/busi_bulk_overflow, where some slices were moved, listed and copied here by the lane; read only).
Per history in HISTORIES: md5 of every copy of the json and the fields npz (copies = same run tag), the run args, the npz rows
(clocks, cadence, gaps, equality with the json records), unconverged records, formation (first record with min(N-|v|) < 0, first
root clock and R_t), the radial grid (sites and nodes), the finiteness of every N, v row, and the slices (clock, size, location,
md5 from a manifest where one exists, else marked). Output: inventory_v1.json, registry_v1.json (for read_history_v1.py)."""
import glob, hashlib, json, os, re, sys
import numpy as np

D = "<internal>/redshift"
CASB = "<internal>/solve_a_bulk"
BB = "<internal>/busi_bulk"

# label: (canonical fields npz, short description, group)
HISTORIES = {
    # --- the lead's named set
    "P64_s4_prod": (CASB + "/v36d_from0/runs_pod7_ell4/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40_fields.npz",
                    "r_s/a 64 production pair, sigma/a 4 (v36d plain, from T=0 to 40)", "named"),
    "P64_s8_prod": (CASB + "/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40_fields.npz",
                    "r_s/a 64 production pair, sigma/a 8 (v36d plain, from T=0 to 40)", "named"),
    "P64_s4_prod56": (BB + "/marks325/out/runs/own56/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to56m325_fields.npz",
                      "P64_s4 production continued to 56 (marks325 own56)", "named"),
    "P64_ref_c19r11": (CASB + "/core_hf_v1/stage_prod_rec/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz",
                       "the reference collapse c19 r11 (r_s/a 64, sigma/a 4, plain, 0 to 40)", "named"),
    "P64_s4_newton": (CASB + "/p64_newton_h100_1/final/run/collapse_lad_P64_s4_newton_v36nq_h100_1_fields.npz",
                      "P64_s4 by Newton from T=0 (FORMATION_newton_route_v1)", "named"),
    "ISO_128_s4": (CASB + "/reopen2/iso128/run/collapse_lad_ISO_128_s4_rd_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_rd2_fields.npz",
                   "ISO_128_s4, job (b), to 12.0625 (ended by its engine rule)", "named"),
    "W256_s4": (D + "/inputs/W256_s4/collapse_lad_W256_s4_m0p2_rs256_n4096_R04_k14_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mf_fields.npz",
                "W256_s4 (h100-f), ended by its engine rule at 2.125 (box copy of the card's final run files)", "named"),
    "S4_128_s16": (CASB + "/ladders/p8/runs/lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d_fields.npz",
                   "the r_s/a 128 sigma/a 16 member (plain, 0 to 40)", "named"),
    # --- the 64 triple of GATECOMP (iii)
    "P64_s2_nk": (CASB + "/pod_sync/pod2_engine/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_fields.npz",
                  "64 triple sigma/a 2 (Newton nk40r2b, 0 to 18.125)", "triple"),
    "P64_s4_c27": (CASB + "/pod_sync/pod2_engine/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg_fields.npz",
                   "64 triple sigma/a 4 (c27 plain, 0 to 40)", "triple"),
    "P64_s8_c27": (CASB + "/pod_sync/pod2_engine/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz",
                   "64 triple sigma/a 8 (c27 plain, 0 to 40)", "triple"),
    "P64_s2_to40": (CASB + "/v36d_relaunch/records/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d_fields.npz",
                    "sigma/a 2 Newton lineage continued to 40 (v36cc_v36d)", "triple"),
    # --- W2
    "W2_v36v": (CASB + "/w2_final_R3b/runs/lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc_fields.npz",
                "W2 (dg step) v36v anchored history, 0 to 40", "other"),
    # --- other formation histories with N, v on every record from T = 0 (r_s/a >= 64)
    "S4_128_s8": (CASB + "/ladders/r1/runs/lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d_fields.npz",
                  "r_s/a 128 sigma/a 8 (plain, 0 to 40)", "other"),
    "S4_128_s4": (CASB + "/ladders/r2/runs/lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d_fields.npz",
                  "r_s/a 128 sigma/a 4 well (plain; no certified formation)", "other"),
    "ISO_128_s4_plain": (CASB + "/ladders/p6/runs/lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n/collapse_lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n_fields.npz",
                         "ISO_128_s4 plain mix (v36n, 0 to 14.375)", "other"),
    "ISO_64_s4": (CASB + "/ladders/r1/runs/lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n/collapse_lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n_fields.npz",
                  "ISO_64_s4 (isotropic, plain, 0 to 40)", "other"),
    "M3c_s4_96": (CASB + "/old_volumes/R3/runs/lad_M3c_s4_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d/collapse_lad_M3c_s4_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d_fields.npz",
                  "r_s/a 96 (m 0.1) sigma/a 4 plain", "other"),
    "M3c_s4_96_nk": (CASB + "/old_volumes/bench1/runs/lad_M3c_s4_nk_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36j/collapse_lad_M3c_s4_nk_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36j_fields.npz",
                     "r_s/a 96 (m 0.1) sigma/a 4 Newton", "other"),
    "P64_s4_B2n4096": (CASB + "/pod_sync/pod2_engine/B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz",
                       "c19 at r_s/a 64 on 4096 sites (doubled box), 0 to 100", "other"),
    "P64_s4_ext200": (CASB + "/B3_ext200/run/out/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext200_r100_fields.npz",
                      "c19 extended to 200 (2048 sites)", "other"),
    "P64_s4_tau4": (CASB + "/pod_sync/pod1_engine/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg_fields.npz",
                    "c19 at tau 4 (0 to 40)", "other"),
    "P64_s4_tau2": (CASB + "/pod_sync/pod2_engine/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg_fields.npz",
                    "c25 at tau 2 (0 to 21.5)", "other"),
    "C64_s4_tau1_v4L": (BB + "/C64/c64s4Lt1/results/collapse_C64_s4_ell4_tau1_twofield_v4L_fields.npz",
                        "Computation C two-field v4L, sigma/a 4, tau 1 (0 to 16)", "other"),
    "C64_s8_v4L": (BB + "/C64/G4s8_c64s8L_sector/segment_c64s8L_lagged_sector_20261019T080907Z/collapse_C64_s8_ell8_tau2_twofield_v4L_fields.npz",
                   "Computation C two-field v4L, sigma/a 8, tau 2 (0 to 16)", "other"),
}

MD5_RE = re.compile(r"^([0-9a-f]{32})\s+\*?(\S.*)$")
MD5_REV = re.compile(r"^(\S+\.npz)\s+([0-9a-f]{32})$")   # md5_done.txt: '<name> <md5>'

# slices written under a parent run's tag before a continuation (the history's records include the parent's)
PARENTS = {
    "P64_s4_prod": ["c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0"],
    "P64_s8_prod": ["c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0"],
    "P64_s4_prod56": ["c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40", "c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0"],
    "P64_s2_to40": ["c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc", "c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b"],
    "ISO_128_s4": ["lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq"],
    "W2_v36v": ["lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_r2b",
                "lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f"],
    "P64_s4_ext200": ["c19_rs64_A_R2_tau8_ell4_avg_ext100_r98", "c19_rs64_A_R2_tau8_ell4_avg_ext100", "c19_rs64_A_R2_tau8_ell4_avg_r11"],
    "P64_s4_B2n4096": [],
}


def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def manifest_map():
    mm = {}
    for src in [l.strip() for l in open(D + "/manifests.txt")] + [D + "/local_overflow_md5.txt"]:
        try:
            for line in open(src, errors="replace"):
                m = MD5_RE.match(line.strip())
                if m:
                    mm.setdefault(os.path.basename(m.group(2).strip()), set()).add(m.group(1))
                    continue
                m = MD5_REV.match(line.strip())
                if m:
                    mm.setdefault(os.path.basename(m.group(1)), set()).add(m.group(2))
        except Exception:
            pass
    return mm


def tag_of(npz):
    return os.path.basename(npz)[:-len("_fields.npz")][len("collapse_"):]


def main():
    census = [json.loads(l) for l in open(D + "/census_v1.jsonl")]
    slices = [l.split(None, 1) for l in open(D + "/all_slices.txt")]
    slices += [l.split(None, 1) for l in open(D + "/local_overflow_slices.txt")]
    slices = [(int(s), p.strip()) for s, p in slices]
    mm = manifest_map()
    inv, reg = {}, {}
    for lab, (npz, desc, grp) in HISTORIES.items():
        tag = tag_of(npz)
        js = npz[:-len("_fields.npz")] + ".json"
        out = {"label": lab, "desc": desc, "group": grp, "tag": tag, "npz": npz, "json": js}
        out["npz_md5"], out["json_md5"] = md5(npz), md5(js)
        out["npz_size"], out["json_size"] = os.path.getsize(npz), os.path.getsize(js)
        copies = []
        for r in census:
            if tag_of(r["npz"]) == tag and r["npz"] != npz:
                c = {"npz": r["npz"], "npz_n": r.get("npz_n"), "npz_T1": r.get("npz_T1")}
                try:
                    c["npz_md5"] = md5(r["npz"])
                    c["same_npz"] = c["npz_md5"] == out["npz_md5"]
                    if r.get("json"):
                        c["json_md5"] = md5(r["json"]); c["same_json"] = c["json_md5"] == out["json_md5"]
                except Exception as e:  # noqa
                    c["error"] = repr(e)[:100]
                copies.append(c)
        out["copies"] = copies
        J = json.load(open(js)); a = J["args"]; rs = float(a["rs"])
        out["args"] = {k: a.get(k) for k in ("rs", "nsites", "tau", "ell", "prep_ell", "kappa_sea", "m", "g", "R0_over_rs", "mix",
                                              "dust_set", "dust_kappa_max", "site_fields", "reading", "reference", "shell", "end_above",
                                              "source_continuation", "resume_from", "resume_T_over_rs") if k in a}
        out["script"] = J.get("script")
        E0 = J.get("head", {}).get("E0"); out["E0"] = E0
        out["two_gE0"] = 2.0 * float(a.get("g", 0.25)) * float(E0) if E0 else None
        recs = [r for r in J["records"] if isinstance(r, dict) and r.get("T") is not None]
        Tr = np.array([float(r["T"]) / rs for r in recs])
        out["engines"] = sorted(set(str(r.get("engine")) for r in recs if r.get("engine")))
        out["n_rec"] = len(recs)
        unc = [(round(float(r["T"]) / rs, 6), (r.get("unconverged") or {}).get("residual")) for r in recs if r.get("converged") is False]
        out["unconverged"] = unc
        form = next((r for r in recs if r.get("min_N_minus_absv") is not None and r["min_N_minus_absv"] < 0), None)
        out["first_negative_m_T"] = float(form["T"]) / rs if form else None
        root = next((r for r in recs if r.get("R_t") is not None), None)
        out["first_root_T"] = float(root["T"]) / rs if root else None
        out["first_root_Rt"] = float(root["R_t"]) if root else None
        out["first_root_Rt_over_2gE0"] = (float(root["R_t"]) / out["two_gE0"]) if (root and out["two_gE0"]) else None
        last = recs[-1]
        out["last_rec"] = {"T": float(last["T"]) / rs, "R_t": last.get("R_t"), "min_m": last.get("min_N_minus_absv"),
                           "E_over_E0": last.get("E_over_E0"), "converged": last.get("converged")}
        out["n_rec_min_m"] = sum(1 for r in recs if r.get("min_N_minus_absv") is not None)
        out["n_rec_zeros"] = sum(1 for r in recs if "trapping_zeros" in r)
        Z = np.load(npz, allow_pickle=True)
        T = np.asarray(Z["T"], float) / rs; R = np.asarray(Z["R"], float)
        N = np.asarray(Z["N"], float); v = np.asarray(Z["v"], float)
        out["npz_members"] = list(Z.files)
        out["npz_n"], out["npz_T0"], out["npz_T1"] = int(len(T)), float(T[0]), float(T[-1])
        d = np.diff(T); dmin = float(d.min()) if len(d) else None
        out["npz_cadences"] = sorted(set(np.round(d, 6).tolist()))
        out["npz_gaps"] = [[float(T[i]), float(T[i + 1])] for i in range(len(d)) if d[i] > 1.5 * dmin] if dmin else []
        out["npz_monotone"] = bool(np.all(d > 0))
        out["npz_eq_records"] = bool(len(T) == len(Tr) and np.allclose(T, Tr, atol=1e-9))
        out["npz_missing_record_T"] = sorted(set(np.round(Tr, 6)) - set(np.round(T, 6)))[:20]
        out["NV_shape"] = list(N.shape)
        out["NV_finite"] = bool(np.isfinite(N).all() and np.isfinite(v).all())
        out["N_min"], out["N_max"] = float(N.min()), float(N.max())
        out["R_grid"] = {"R0": float(R[0]), "dR": float(R[1] - R[0]), "R_last": float(R[-1]), "n": int(len(R)),
                         "uniform": bool(np.allclose(np.diff(R), R[1] - R[0]))}
        if "r_nodes" in Z.files:
            rn = Z["r_nodes"]; Nn = Z["N_nodes"]
            lens = sorted(set(len(x) for x in rn if x is not None))
            none = int(sum(1 for x in Nn if x is None))
            r0 = np.asarray(rn[0], float)
            same = bool(all(x is not None and len(x) == len(r0) and np.allclose(np.asarray(x, float), r0) for x in rn))
            out["nodes"] = {"lengths": lens, "rows_None": none, "r0": float(r0[0]), "dr": float(r0[1] - r0[0]),
                            "r_last": float(r0[-1]), "fixed_grid": same}
        out["applied"] = "N_applied" in Z.files
        # slices
        sl = []
        for t_ in [tag] + PARENTS.get(lab, []):
            pref = "orbitals_" + t_ + "_T"
            for size, p in slices:
                b = os.path.basename(p)
                if b.startswith(pref) and b.endswith(".npz") and b[len(pref):len(pref) + 1].isdigit():
                    clk = b[len(pref):-4]
                    mds = sorted(mm.get(b, set()))
                    sl.append({"T": clk, "size_GB": round(size / 1e9, 2), "path": p, "manifest_md5": mds,
                               "tag": "own" if t_ == tag else "parent " + t_})
        sl.sort(key=lambda s: (float(s["T"][:7]), s["T"], s["path"]))
        out["slices"] = sl
        inv[lab] = out
        reg[lab] = {"npz": npz, "json": js, "npz_md5": out["npz_md5"], "json_md5": out["json_md5"], "rs": rs, "desc": desc,
                    "site_fields": a.get("site_fields"), "copies_same_npz": [c["npz"] for c in copies if c.get("same_npz")]}
        print(lab, out["npz_n"], out["npz_T0"], out["npz_T1"], out["first_root_T"], len(unc), len(sl), flush=True)
    json.dump(inv, open(D + "/inventory_v1.json", "w"), indent=1, default=float)
    json.dump(reg, open(D + "/registry_v1.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
