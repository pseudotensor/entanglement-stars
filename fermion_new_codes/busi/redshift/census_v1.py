"""busi REDSHIFT inputs (inbox #416), step 1: census of every stored collapse history (_fields.npz + its json) on the box.

DATA script, box CPU only, read-only on every input. No redshift is computed. For each _fields.npz in a list file it reads the
npz member names and the small members (R, T, g, ell, tau, the shapes of N and v), and the sibling json's args and records (no N, v
arrays loaded), and writes one JSON line per file:
  clocks (T/r_s) of the npz rows and the json records, cadence, gaps, unconverged records, first trapping root (clock, R_t),
  min over records of min(N - |v|), whether N, v, N_nodes, v_nodes, N_applied, v_applied are present, run args.
Usage: python census_v1.py LIST OUT.jsonl     (LIST: lines '<size> <date> <path>', as find -printf "%s %TY-%Tm-%Td %p")."""
import json, os, sys, zipfile
import numpy as np

SMALL = ("R", "T", "g", "ell", "tau")
ARGS = ("rs", "nsites", "tau", "ell", "prep_ell", "m", "g", "R0_over_rs", "kappa_sea", "dust_set", "dust_kappa_max", "mix",
        "site_fields", "reading", "reference", "resume_from", "resume_T_over_rs", "T_over_rs", "segment_end_T_over_rs", "tag")


def npz_shapes(path):
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            with zf.open(info) as fh:
                try:
                    ver = np.lib.format.read_magic(fh)
                    shape, _, dtype = (np.lib.format.read_array_header_1_0(fh) if ver == (1, 0)
                                       else np.lib.format.read_array_header_2_0(fh))
                    out[name] = [list(shape), str(dtype)]
                except Exception as e:  # noqa
                    out[name] = ["?", repr(e)[:60]]
    return out


def main(lst, out):
    rows = []
    for line in open(lst):
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        p = parts[2].strip()
        row = {"npz": p, "size": int(parts[0]), "date": parts[1]}
        try:
            row["members"] = npz_shapes(p)
            Z = np.load(p, allow_pickle=False)
            for k in SMALL:
                if k in Z.files:
                    a = Z[k]
                    row[k] = a.tolist() if a.ndim == 0 else None
            T = np.asarray(Z["T"], float) if "T" in Z.files else None
            R = np.asarray(Z["R"], float) if "R" in Z.files else None
            if R is not None:
                row["R0"], row["R1"], row["Rlast"], row["nR"] = float(R[0]), float(R[1]), float(R[-1]), int(len(R))
        except Exception as e:  # noqa
            row["npz_error"] = repr(e)[:200]
            T = None
        js = p[:-len("_fields.npz")] + ".json"
        row["json"] = js if os.path.exists(js) else None
        rs = None
        if row["json"]:
            try:
                J = json.load(open(js))
                a = J.get("args", {})
                row["args"] = {k: a.get(k) for k in ARGS if k in a}
                row["script"] = J.get("script")
                rs = float(a["rs"]) if a.get("rs") else None
                recs = [r for r in J.get("records", []) if isinstance(r, dict)]
                Tr = [float(r["T"]) for r in recs if r.get("T") is not None]
                row["n_rec"] = len(recs)
                row["engine"] = sorted(set(str(r.get("engine")) for r in recs if r.get("engine")))[:4]
                if rs and Tr:
                    row["rec_T0"], row["rec_T1"] = Tr[0] / rs, Tr[-1] / rs
                    row["rec_dT"] = sorted(set(round(x / rs, 6) for x in np.diff(Tr)))[:6] if len(Tr) > 1 else []
                unc = [round(float(r["T"]) / rs, 6) for r in recs if r.get("converged") is False and rs and r.get("T") is not None]
                row["n_unconv"], row["unconv"] = len(unc), unc[:40]
                first = next((r for r in recs if r.get("R_t") is not None), None)
                if first is not None and rs:
                    row["root_T"], row["root_Rt"] = float(first["T"]) / rs, float(first["R_t"])
                mins = [(float(r["min_N_minus_absv"]), float(r["T"]) / rs) for r in recs
                        if r.get("min_N_minus_absv") is not None and rs]
                if mins:
                    row["min_m"], row["min_m_T"] = min(mins)
                    row["last_m"] = mins[-1][0]
                row["has_min_m"] = len(mins)
                row["has_zeros"] = sum(1 for r in recs if "trapping_zeros" in r)
                row["E0"] = J.get("head", {}).get("E0")
            except Exception as e:  # noqa
                row["json_error"] = repr(e)[:200]
        if T is not None and len(T):
            rs = rs or row.get("args", {}).get("rs")
            if rs:
                t = T / float(rs)
                row["npz_T0"], row["npz_T1"], row["npz_n"] = float(t[0]), float(t[-1]), int(len(t))
                d = np.diff(t)
                row["npz_dT"] = sorted(set(np.round(d, 6).tolist()))[:6]
                row["npz_monotone"] = bool(np.all(d > 0))
        rows.append(row)
        print(p, row.get("npz_n"), row.get("root_T"), flush=True)
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
