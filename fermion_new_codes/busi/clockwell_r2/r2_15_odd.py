"""Second reader: odd (N=1) on-site cohorts: termination, D_RK re-derivation, fixed-receiver F_e. Read-only. argv: n out_json"""
import json, sys, os, hashlib
import numpy as np
n = int(sys.argv[1]); out_path = sys.argv[2]
d = "<internal>/odd_n%d_v1" % n
rd = "<internal>/odd_n%d_v1" % n
def recs(p): return [json.loads(l) for l in open(p) if l.strip()]
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
R = recs(os.path.join(d, "written_report.md")); by = {}
for r in R: by.setdefault(r["kind"], []).append(r["data"])
t = by["termination"][0]; ch = by.get("check", [])
M = 8 * n; masks = [1 << m for m in range(M)]  # assumed canonical one-particle order; verified below on the preparation
def load(ref):
    z = np.load(os.path.join(d, ref["array"])); return z[z.files[0]]
acts = {a["metadata"]["output"]: a for a in by["action"]}
prep = load(by["preparations"][0]["registered"])
prep_ok = bool(abs(abs(prep[0, 0]) - 2 ** -0.5) < 1e-15 and abs(abs(prep[1, 1]) - 2 ** -0.5) < 1e-15 and abs(prep).sum() - 2 * 2 ** -0.5 < 1e-14)
def drk(C, out):
    C = C / np.linalg.norm(C); K = [m for m in range(M) if m not in out]
    # one particle: K basis = vacuum + one particle in each K mode; O traced. No sign issue with one particle.
    nk = 1 + len(K); psi = np.zeros((2, nk, 1 + len(out)), complex)
    for r in range(2):
        for i, m in enumerate(range(M)):
            if m in K: psi[r, 1 + K.index(m), 0] = C[r, i]
            else: psi[r, 0, 1 + out.index(m)] = C[r, i]
    X = psi.reshape(2 * nk, -1); rho = X @ X.conj().T; R4 = rho.reshape(2, nk, 2, nk)
    rR = np.einsum("aibi->ab", R4); rK = np.einsum("aiaj->ij", R4)
    w = np.linalg.eigvalsh(rho - np.kron(rR, rK)); return 0.5 * float(abs(w).sum())
rows = []; dev = 0.0
for g in by["register"]:
    C = load(acts[g["state"]["output"]]["raw_vector"])
    v = drk(C, list(g["output"])); dev = max(dev, abs(v - g["decoupling"]["value"]))
    rows.append({"clock": g["state"]["end"], "role": g["role"], "first": g.get("first_collected"), "D": v, "D_int": [g["decoupling"]["lower"], g["decoupling"]["upper"]], "status": g["status"]})
RR = recs(os.path.join(rd, "written_report.md")); rb = {}
for r in RR: rb.setdefault(r["kind"], []).append(r["data"])
fe = [x["fidelity_interval"] for x in rb["new_fixed_receiver"]]
out = {"report_md5": md5(os.path.join(d, "written_report.md")), "readout_md5": md5(os.path.join(rd, "written_report.md")),
       "term_ok": t["status"] == "complete" and t["all_pass"] is True and t["sources_unchanged"] is True and t["checks"] == len(ch) and all(c["pass"] for c in ch),
       "checks": len(ch), "prep_basis_ok": prep_ok, "max_dev_D": dev, "rows": rows,
       "all_not_decoupled": all(r["D_int"][1] > 1e-4 for r in rows), "light_rows_three_quarters": all(abs(r["D"] - 0.75) < 1e-12 for r in rows if r["role"] == "light"),
       "readout_term": {k: rb["termination"][0].get(k) for k in ("status", "all_pass", "sources_unchanged")},
       "receiver_intervals": fe, "receivers_all_quarter": all(abs(a - 0.25) < 1e-9 and abs(b - 0.25) < 1e-9 for a, b in fe)}
json.dump(out, open(out_path, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
