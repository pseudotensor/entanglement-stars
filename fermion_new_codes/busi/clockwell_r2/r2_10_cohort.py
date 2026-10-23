"""Second reader (clockwell_r2): independent re-derivation for ONE physical cohort report.

Written from the definitions in MOCK_unitary_lead_run_spec_v1.md, MOCK_filtered_clockwell_spec_v1.md,
MOCK_filtered_readout_addendum_v1.md and the formulas in sec. 9 of MOCK_filtered_clockwell_v1.md.wip.
Does not import or read any busi/code/clockwell/cw*.py or <internal>/clockwell products
(the INPUT_MD5SUMS manifest excepted, used only to verify input md5s).

Usage: python r2_10_cohort.py <physical_dir> <arm: filtered|onsite> <readout_dir or -> <out_json>
Reads <physical_dir>/written_report.md and the npz arrays it names; read-only on inputs.
"""
import json, sys, os, hashlib, math, time
import numpy as np

t0 = time.time()
phys_dir, arm, ro_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MANIFEST = "<internal>/INPUT_MD5SUMS.txt"
HALO = "endpoint-copy halo; N_-1=N_0=1/2; N_n=N_(n-1)=1; declared convention, not a measurement"

manifest = {}
with open(MANIFEST) as f:
    for line in f:
        p = line.split()
        if len(p) == 2: manifest[p[1]] = p[0]
md5_log = {"checked": 0, "mismatch": [], "not_in_manifest": 0, "not_in_manifest_paths": [], "digests": {}}

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    d = h.hexdigest()
    md5_log["checked"] += 1
    if path in manifest:
        if manifest[path] != d: md5_log["mismatch"].append(path)
    else:
        md5_log["not_in_manifest"] += 1; md5_log["not_in_manifest_paths"].append(path)
    md5_log["digests"][path] = d
    return d

def load_report(d):
    p = os.path.join(d, "written_report.md")
    recs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line: recs.append(json.loads(line))
    return recs, md5(p)

_arr_cache = {}
def arr(d, ref):
    p = os.path.join(d, ref["array"])
    if p not in _arr_cache:
        md5(p)
        z = np.load(p)
        _arr_cache[p] = z[z.files[0]]
    return _arr_cache[p]

recs, rep_md5 = load_report(phys_dir)
by = {}
for r in recs: by.setdefault(r["kind"], []).append(r["data"])
out = {"phys_dir": phys_dir, "arm": arm, "report_md5": rep_md5}

# ---------- (a) termination ----------
term = by["termination"][0]
checks = by.get("check", [])
out["termination"] = {"status": term["status"], "all_pass": term["all_pass"],
                      "sources_unchanged": term["sources_unchanged"], "checks_field": term["checks"],
                      "check_rows": len(checks), "check_rows_all_pass": all(c["pass"] for c in checks)}
out["termination"]["ok"] = (term["status"] == "complete" and term["all_pass"] is True and
                            term["sources_unchanged"] is True and term["checks"] == len(checks) and
                            out["termination"]["check_rows_all_pass"])

# ---------- halo header audit (filtered only) ----------
if arm == "filtered":
    hk = ["coherent_conversion", "conversion_status", "wall_complete", "purification",
          "later_collected_B2_at_first_collection"]
    rows = sum(len(by.get(k, [])) for k in hk)
    good = sum(1 for k in hk for x in by.get(k, []) if x.get("header") == HALO)
    model_hv = by["model"][0]["metadata"].get("halo_values")
    wall_hv = by["wall_complete"][0]["context"]["model"].get("halo_values")
    out["halo_audit"] = {"rows": rows, "rows_with_header": good, "model_halo_values": model_hv,
                         "wall_halo_values": wall_hv,
                         "ok": good == rows and model_hv == ["1/2", "1"] and wall_hv == ["1/2", "1"]}

# ---------- basis ----------
conv = sorted([c for c in by["coherent_conversion"] if not c["dephased"]], key=lambda c: c["clock"])
keys = conv[0]["density"]["row_keys"]
assert all(k[0] == 0 for k in keys)
masks = np.array([k[1] for k in keys], dtype=np.int64)
D = len(masks)
n = len(by["model"][0]["metadata"]["profile"])
nmodes = 8 * n
idx_of = {int(m): i for i, m in enumerate(masks)}
bits = np.array([[(m >> b) & 1 for b in range(nmodes)] for m in masks], dtype=np.int64)
species = np.array([(b % 8) // 4 for b in range(nmodes)])
nL = bits[:, species == 1].sum(1); nH = bits[:, species == 0].sum(1)
HHrows = np.where(nH == 2)[0]; LLrows = np.where(nL == 2)[0]
out["basis"] = {"D": D, "n": n, "HH": len(HHrows), "LL": len(LLrows)}

# ---------- actions index ----------
act_by_out = {}
for a in by["action"]:
    act_by_out[a["metadata"]["output"]] = a
def vec_of_digest(dg):
    a = act_by_out[dg]
    return arr(phys_dir, a["raw_vector"]), a["metadata"]

# ---------- graded reduction ----------
def kept_traced(keep):
    keep = sorted(keep); tr = [m for m in range(nmodes) if m not in set(keep)]
    kmask = sum(1 << m for m in keep); tmask = sum(1 << m for m in tr)
    sgn = np.ones(D)
    for i, m in enumerate(masks):
        cnt = 0
        for t in tr:
            if (m >> t) & 1:
                for k in keep:
                    if k > t and (m >> k) & 1: cnt += 1
        sgn[i] = -1.0 if cnt % 2 else 1.0
    kpart = masks & kmask; tpart = masks & tmask
    ku = {v: i for i, v in enumerate(sorted(set(kpart.tolist())))}
    tu = {v: i for i, v in enumerate(sorted(set(tpart.tolist())))}
    ki = np.array([ku[v] for v in kpart.tolist()]); ti = np.array([tu[v] for v in tpart.tolist()])
    return sgn, ki, ti, len(ku), len(tu)

_kt_cache = {}
def KT(keep):
    key = tuple(sorted(keep))
    if key not in _kt_cache: _kt_cache[key] = kept_traced(keep)
    return _kt_cache[key]

def rdm_pure(psi, keep):
    psi = psi / np.linalg.norm(psi)
    sgn, ki, ti, nk, nt = KT(keep)
    M = np.zeros((nk, nt), complex); M[ki, ti] = sgn * psi
    return M @ M.conj().T

def vn(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2); w = w[w > 1e-300]
    return float(-(w * np.log(w)).sum())

def drk(C, out_modes):
    """C: (2, D) registered vector. K = all fermion modes not in out_modes. D_RK = 1/2||rho_RK - rho_R x rho_K||_1"""
    C = C / np.linalg.norm(C)
    keepK = [m for m in range(nmodes) if m not in set(out_modes)]
    sgn, ki, ti, nk, nt = KT(keepK)
    M = np.zeros((2, nk, nt), complex)
    for r in range(2): M[r, ki, ti] = sgn * C[r]
    M2 = M.reshape(2 * nk, nt)
    rhoRK = M2 @ M2.conj().T
    R4 = rhoRK.reshape(2, nk, 2, nk)
    rhoR = np.einsum("aibi->ab", R4); rhoK = np.einsum("aiaj->ij", R4)
    X = rhoRK - np.kron(rhoR, rhoK)
    w = np.linalg.eigvalsh((X + X.conj().T) / 2)
    return 0.5 * float(np.abs(w).sum())

def fe(C, out_modes):
    """fixed local-pair receiver (addendum sec. 2): F_e = sum_b |<Phi|v_b>|^2 + p_fail/4"""
    C = C / np.linalg.norm(C)
    O = set(out_modes); s = 0.0; ps = 0.0; nb = 0
    for x in range(1, n):
        for A in (0, 1):
            ms = [8 * x + 4 * A + c for c in range(4)]
            if not all(m in O for m in ms): continue
            nb += 1
            def amp(a, b): return C[:, idx_of[(1 << ms[a]) | (1 << ms[b])]]
            v = np.zeros((2, 2), complex)
            v[:, 0] = amp(0, 1); v[:, 1] = (amp(0, 3) - amp(1, 2)) / math.sqrt(2)
            s += abs((v[0, 0] + v[1, 1]) / math.sqrt(2)) ** 2
            ps += float((abs(v) ** 2).sum())
    return s + (1 - ps) / 4, nb

def purity(psi, keep):
    r = rdm_pure(psi, keep); return float(np.real(np.trace(r @ r)))

# ---------- (b) conversion ----------
conv_rows = []
dev = {"pLL_density": 0.0, "CHL_density": 0.0, "pLL_vector": 0.0, "CHL_vector": 0.0, "rho_vs_vector": 0.0}
for c in conv:
    rho = arr(phys_dir, c["density"]["matrix"])
    pLL = float(np.real(np.trace(rho[np.ix_(LLrows, LLrows)])))
    CHL = float(np.linalg.norm(rho[np.ix_(HHrows, LLrows)]))
    psi, _ = vec_of_digest(c["state"])
    psi = psi / np.linalg.norm(psi)
    pLLv = float((abs(psi[LLrows]) ** 2).sum()); CHLv = math.sqrt(float((abs(psi[HHrows]) ** 2).sum()) * pLLv)
    dev["rho_vs_vector"] = max(dev["rho_vs_vector"], float(abs(rho - np.outer(psi, psi.conj())).max()))
    sv = c["populations"]["LL"]["value"]; sc = c["C_HL"]["value"]
    dev["pLL_density"] = max(dev["pLL_density"], abs(pLL - sv)); dev["CHL_density"] = max(dev["CHL_density"], abs(CHL - sc))
    dev["pLL_vector"] = max(dev["pLL_vector"], abs(pLLv - sv)); dev["CHL_vector"] = max(dev["CHL_vector"], abs(CHLv - sc))
    sp = c["site_populations"]
    conv_rows.append({"clock": c["clock"], "pLL": pLL, "pLL_saved": sv, "pLL_int": [c["populations"]["LL"]["lower"], c["populations"]["LL"]["upper"]],
                      "CHL": CHL, "CHL_int": [c["C_HL"]["lower"], c["C_HL"]["upper"]],
                      "core_H": sp["core_H"]["value"], "core_L": sp["core_L"]["value"], "lead_H": sp["lead_H"]["value"],
                      "lead_L": sp["lead_L"]["value"], "state_error": c["state_error"],
                      "site1": [c["per_site_species_populations"].get("1_H", {}).get("value"),
                                c["per_site_species_populations"].get("1_L", {}).get("value"),
                                c["per_site_species_populations"].get("1_total", {}).get("value")]})
st = {s["clock"]: s for s in by["conversion_status"]}
for r in conv_rows:
    s = st[r["clock"]]
    r["flag"] = s["coherent_conversion"]
    r["flag_from_bounds"] = (r["pLL_int"][0] > 0 and r["CHL_int"][0] > 0)
out["conversion"] = {"rows": conv_rows, "dev": dev,
                     "count_flag": sum(r["flag"] for r in conv_rows),
                     "flag_eq_bounds": all(r["flag"] == r["flag_from_bounds"] for r in conv_rows)}
mx = max(conv_rows, key=lambda r: r["pLL_saved"])
out["conversion"]["max"] = {"clock": mx["clock"], "pLL": mx["pLL_saved"], "pLL_int": mx["pLL_int"], "CHL_int": mx["CHL_int"]}

# later-collected B2 record
if "later_collected_B2_at_first_collection" in by:
    b2 = by["later_collected_B2_at_first_collection"][0]
    c1 = [r for r in conv_rows if r["clock"] == b2["clock"]][0]
    out["B2"] = {"record_clock": b2["clock"], "preceding_primary_clock": b2.get("preceding_primary_clock"),
                 "H": b2["site_H"]["value"], "L": b2["site_L"]["value"], "T": b2["site_total"]["value"],
                 "err": b2["site_H"]["error"], "equals_conversion_site1_at_record_clock": c1["site1"] == [b2["site_H"]["value"], b2["site_L"]["value"], b2["site_total"]["value"]]}

# ---------- (c) entropies ----------
ent_dev = 0.0; ent_n = 0; ent_rows = []
for e in by["entropy"]:
    if "_snapshot" not in e or e.get("register"): continue
    psi = arr(phys_dir, e["_snapshot"]["raw_vector"])
    if psi.ndim != 1: continue
    S = vn(rdm_pure(psi, e["modes"]))
    ent_dev = max(ent_dev, abs(S - e["value"])); ent_n += 1
out["entropy"] = {"rows": ent_n, "max_dev": ent_dev}

# ---------- (d) register rows: D_RK and F_e ----------
reg_rows = []; d_dev = 0.0
for g in by["register"]:
    C, meta = vec_of_digest(g["state"]["output"])
    assert C.shape == (2, D), C.shape
    dval = drk(C, g["output"])
    d_dev = max(d_dev, abs(dval - g["decoupling"]["value"]))
    f, nb = fe(C, g["output"])
    eps = min(1.0, g["state"]["error"])
    reg_rows.append({"clock": g["state"]["end"], "role": g["role"], "first": g.get("first_collected", False),
                     "out_len": len(g["output"]), "D": dval, "D_saved": g["decoupling"]["value"],
                     "D_int": [g["decoupling"]["lower"], g["decoupling"]["upper"]], "status": g["status"],
                     "status_mine": "not_decoupled_to_1e-4" if g["decoupling"]["upper"] > 1e-4 else ("decoupled" if g["decoupling"]["upper"] <= 1e-4 else "?"),
                     "Fe": f, "Fe_int": [f - eps, f + eps], "branches": nb})
out["register"] = {"rows": reg_rows, "max_dev_D": d_dev,
                   "all_not_decoupled": all(r["D_int"][1] > 1e-4 for r in reg_rows),
                   "max_Fe_upper": max(r["Fe_int"][1] for r in reg_rows)}

# ---------- (e) purification ----------
def h(T): return 0.0 if T <= 0 or T >= 1 else -T * math.log(T) - (1 - T) * math.log(1 - T)
def F(d, T):
    T = min(1.0, T)
    return T * math.log(d - 1) + h(T) if T < 1 - 1 / d else math.log(d)
if "purification" in by:
    p = by["purification"][0]
    d2, d1 = p["dimensions"]
    cq = p["protected_conditional"]["upper"]; B1 = p["B_first"]; B2 = p["B_second"]
    W1 = p["Bwall_first"]; W2 = p["Bwall_second"]
    mine = {"Ucond_fin": cq + F(d2, B2) + F(d1, B2), "Ucum_fin": cq + F(d2, B2) + F(d1, B1),
            "Ucond_ext": cq + F(d2, B2 + W2) + F(d1, B2 + W2), "Ucum_ext": cq + F(d2, B2 + W2) + F(d1, B1 + W1)}
    saved = {"Ucond_fin": p["finite_chain"]["conditional_upper"], "Ucum_fin": p["finite_chain"]["cumulative_upper"],
             "Ucond_ext": p["empty_uniform_extension"]["conditional_upper"], "Ucum_ext": p["empty_uniform_extension"]["cumulative_upper"]}
    out["purification"] = {"mine": mine, "saved": saved, "max_dev": max(abs(mine[k] - saved[k]) for k in mine),
                           "dims": [d2, d1], "B_first": B1, "B_second": B2, "Bwall_first": W1, "Bwall_second": W2,
                           "c2": [p["primary_conditional"]["lower"], p["primary_conditional"]["upper"]],
                           "D2": [p["primary_cumulative"]["lower"], p["primary_cumulative"]["upper"]],
                           "Cq": [p["protected_conditional"]["lower"], p["protected_conditional"]["upper"]],
                           "status": [p["finite_chain"]["status"], p["empty_uniform_extension"]["status"]],
                           "exclusion_fin": mine["Ucond_fin"] < 0 and mine["Ucum_fin"] < 0,
                           "exclusion_ext": mine["Ucond_ext"] < 0 and mine["Ucum_ext"] < 0}
wc = by["wall_complete"][0]
if "bounds" in wc:
    out["wall"] = {"prefix_clocks": wc.get("prefix_clocks"), "bounds": wc["bounds"]}
else:
    out["wall"] = {"bounds": None}

# ---------- trajectory check: exp(-iH) psi(0) vs psi(1) ----------
na = [x for x in by["numerical_action"] if x.get("kind") == "operator"][0]
H = arr(phys_dir, na["hamiltonian"])
w, V = np.linalg.eigh((H + H.conj().T) / 2)
def U(t, v): return V @ (np.exp(-1j * w * t) * (V.conj().T @ v))
pure_plan = by["plans"][0]["pure"]
prim = {}
for a in by["action"]:
    m = a["metadata"]
    if m["role"] == "primary" and m["plan"] == pure_plan and m["start"] == 0.0 and (m["end"] * 4) == int(m["end"] * 4):
        prim[m["end"]] = a  # the 17 registered primary clocks j/4 only (wall-midpoint actions at odd/128 excluded)
psi0 = arr(phys_dir, prim[0.0]["raw_vector"]); psi1 = arr(phys_dir, prim[1.0]["raw_vector"])
out["traj_check_t1"] = float(np.linalg.norm(U(1.0, psi0) - psi1))
# own per-site populations from the saved primary vectors (all 17 clocks), and fixed-cut purities at t = 1, 2, 4
def site_pops(psi):
    p = abs(psi) ** 2 / float((abs(psi) ** 2).sum()); res = {}
    for x in range(n):
        for A, nm in ((0, "H"), (1, "L")):
            cols = [8 * x + 4 * A + c for c in range(4)]
            res["%d_%s" % (x, nm)] = float(p @ bits[:, cols].sum(1))
    res["core_H"] = res["0_H"]; res["core_L"] = res["0_L"]
    res["lead_H"] = sum(res["%d_H" % x] for x in range(1, n)); res["lead_L"] = sum(res["%d_L" % x] for x in range(1, n))
    return res
out["own_pops"] = {str(c): site_pops(arr(phys_dir, prim[c]["raw_vector"])) for c in sorted(prim)}
cuts = {"core": list(range(8)), "lead": list(range(8, nmodes))}
if n == 3: cuts["B1"] = list(range(16, 24))
out["own_purity"] = {k: {str(c): purity(arr(phys_dir, prim[c]["raw_vector"]), m) for c in (1.0, 2.0, 4.0)} for k, m in cuts.items()}
out["own_pLL_by_clock"] = {}
for c in sorted(prim):
    v = arr(phys_dir, prim[c]["raw_vector"]); v = v / np.linalg.norm(v)
    pl = float((abs(v[LLrows]) ** 2).sum()); ph = float((abs(v[HHrows]) ** 2).sum())
    out["own_pLL_by_clock"][str(c)] = [pl, math.sqrt(ph * pl)]
src = (masks & 0xFF) != 0
eta = np.where(src, 0, psi1)
ret = {"p_escape": float(np.vdot(eta, eta).real), "p_return": {}}
for j in range(4, 17):
    t = j / 4
    ph = U(t - 1.0, eta)
    ret["p_return"][t] = float((abs(ph[src]) ** 2).sum())
out["return_mine"] = ret
out["g0_vectors"] = None

# ---------- readout report ----------
if ro_dir != "-":
    rrec, rmd5 = load_report(ro_dir)
    rb = {}
    for r in rrec: rb.setdefault(r["kind"], []).append(r["data"])
    rt = rb["termination"][0]
    rchecks = rb.get("check", [])
    ro = {"report_md5": rmd5, "termination": {k: rt.get(k) for k in ("status", "all_pass", "sources_unchanged", "checks")}}
    ro["termination"]["ok"] = rt.get("status") == "complete" and rt.get("all_pass") is True
    # receivers
    fdev = 0.0; frows = []
    for x in rb.get("new_fixed_receiver", []):
        dg = x["context"]["action"]["output"]
        C, _ = vec_of_digest(dg)
        f, nb = fe(C, x["output"])
        lo, hi = x["fidelity_computed_interval"]
        fdev = max(fdev, abs(f - (lo + hi) / 2))
        frows.append({"clock": x["context"]["clock"], "choice": x["context"]["choice"], "first": x["context"]["first"],
                      "Fe": f, "Fe_int_saved": x["fidelity_interval"]})
    pdev = 0.0; prow = 0
    for x in rb.get("new_pure_cut_purity", []):
        dg = x["context"]["action"]["output"]
        psi, _ = vec_of_digest(dg)
        pv = purity(psi, x["modes"])
        lo, hi = x["unclipped"]
        pdev = max(pdev, abs(pv - (lo + hi) / 2)); prow += 1
    rrows = []; inside = True; maxout = 0.0
    for x in rb.get("return_probability", []):
        t = x["context"]["readout_clock"]
        assert x["context"]["escape_clock"] == 1.0
        assert x["context"]["root_action"]["output"] == prim[1.0]["metadata"]["output"]
        e_int = x["escape"]["interval"]; j_int = x["joint_return"]["interval"]
        pe = ret["p_escape"]; pj = 0.0 if t == 1.0 else ret["p_return"][t]
        ok = (e_int[0] - 1e-12 <= pe <= e_int[1] + 1e-12) and (j_int[0] - 1e-12 <= pj <= j_int[1] + 1e-12)
        inside = inside and ok
        maxout = max(maxout, max(0, e_int[0] - pe, pe - e_int[1], j_int[0] - pj, pj - j_int[1]))
        rrows.append({"t": t, "pe": pe, "pe_int": e_int, "pj": pj, "pj_int": j_int, "inside": ok,
                      "cond_int": x.get("conditional_return_interval"), "cond_status": x.get("conditional_status")})
    ro.update({"receiver_rows": len(frows), "receiver_max_dev": fdev, "receivers": frows,
               "purity_rows": prow, "purity_max_dev": pdev, "return_rows": len(rrows),
               "return_all_inside": inside, "return_max_outside": maxout, "returns": rrows})
    out["readout"] = ro

out["md5_log"] = md5_log
out["wall_seconds"] = time.time() - t0
with open(out_path, "w") as f: json.dump(out, f, indent=1, default=str)
print(json.dumps({"term": out["termination"]["ok"], "dev": dev, "ent": out["entropy"], "dD": d_dev,
                  "pur": out.get("purification", {}).get("max_dev"), "traj": out["traj_check_t1"],
                  "ro": {k: out.get("readout", {}).get(k) for k in ("receiver_max_dev", "purity_max_dev", "return_all_inside", "return_max_outside")},
                  "md5": {"checked": md5_log["checked"], "mismatch": len(md5_log["mismatch"]), "not_in_manifest": md5_log["not_in_manifest"]},
                  "secs": out["wall_seconds"]}))
