#!/usr/bin/env python
"""run_collapse_v19_from_slice_v1.py (builder analyst; inbox #36 B, the engine continuation: "continue the sigma/a 4 member from its stored
T/r_s 40 state ... to T/r_s 100 with the member's own settings (tau 8, sigma/a 4, c_LZ, the stop rule and reading rule 30.3), storing the
fields every record and the full state every 2 r_s; if the T/r_s 40 state is not restartable, the production's own last state with its
engine").

Why this runner: neither stored T/rs 40 state restarts the run. The sigma/a 4 member's slices are all complex64, and the production's T/rs
30, 35 and 40 slices are complex64 (its complex128 slices are T/rs 12, 16, 20, 24 and 28): engine_slice_restart_check_v1 forms the sources
from a slice exactly as the engine's replay gate does and finds the T/rs 40 slices 6.2e-06 (the member) and 8.7e-06 (the production) away
from the stored shell totals against the gate's acceptance 1e-8, the production's complex128 T/rs 28 slice 3.9e-14. So the continuation
starts from the production's last restartable state, T/rs 28 (complex128), with its own engine, engine_collapse_v19, and its own arguments
(read from its json and required equal), and the records T/rs 28.125 to 40 recomputed are the restart's check against the stored ones.

engine_collapse_v19 run unchanged: this runner imports it, replaces its resume_state (the replay of a stored v17 parent) by the load of a
stored complex128 slice, and calls its own main(): the slice's sea, dust (with its occupations) and R2 columns per channel at the record
T_res; the gate (the sources from those columns against the stored shell totals at T_res, acceptance 1e-8, else the run does not start);
the eigendecomposition at the slice's applied fields (its eigenvalues printed against the slice's stored ones) and the propagation of the
sea and the dust past T_res by exp(-i h tau), as the engine's own loop does after a record; the continuation's window restarts with the one
reference R2(T_res) and its stored gap between blocks (the uninterrupted run held up to 4; ops v2 uses min(m, entries), so the first three
records after T_res may take a nearer base: the overlap with the stored records shows whether that changes anything); the stored records,
fields (N_cinit, v_cinit included) and head up to T_res are kept in the new run's files, as v19's own resume keeps them. Two runner-level
guards, no change to the computation: a slice is not written when the disk's free space is below --min-free-gb (the line printed and the
record's slice_saved set to null in the json); the json carries a top-level "runner" entry. Usage: as engine_collapse_v19.py with
--resume-from <dir/tag of the stored run> --resume-T-over-rs <T_res> plus the runner's --slice <path> [--min-free-gb 60]."""
import json as _json
import os
import shutil
import sys
import time

import numpy as np


def _pop(argv, key, default=None):
    if key in argv:
        i = argv.index(key)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


SLICE = _pop(sys.argv, "--slice")
MIN_FREE_GB = float(_pop(sys.argv, "--min-free-gb", "60"))
RUNNER = {"script": "run_collapse_v19_from_slice_v1.py", "engine": "engine_collapse_v19.py (imported and run unchanged; resume_state replaced by the slice load)",
          "slice": SLICE, "min_free_gb": MIN_FREE_GB, "skipped_slices": []}


def free_gb(path):
    return shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".").free / 1e9


def main():
    import engine_collapse_v19 as E
    real_json = E.json

    class _JsonShim:
        def __getattr__(self, k):
            return getattr(real_json, k)

        def dump(self, obj, fh, **kw):
            if isinstance(obj, dict) and "records" in obj:
                obj = dict(obj)
                recs = []
                for r in obj["records"]:
                    if isinstance(r, dict) and r.get("slice_saved") and not os.path.exists(r["slice_saved"]):
                        r = dict(r); r["slice_saved"] = None; r["slice_skipped_low_disk"] = True
                    recs.append(r)
                obj["records"] = recs
                obj["runner"] = RUNNER
            return real_json.dump(obj, fh, **kw)
    E.json = _JsonShim()
    real_savez = np.savez

    def savez_guarded(file, *args, **kw):
        if isinstance(file, str) and os.path.basename(file).startswith("orbitals_"):
            fg = free_gb(file)
            if fg < MIN_FREE_GB:
                RUNNER["skipped_slices"].append({"file": file, "free_gb": fg})
                print("runner: slice %s NOT written: the disk's free space %.1f GB is below %.1f GB" % (file, fg, MIN_FREE_GB), flush=True)
                return None
        return real_savez(file, *args, **kw)
    np.savez = savez_guarded

    def resume_from_slice(a, chs, R, shell):
        torch = E.BK["torch"]
        t0 = time.time()
        d, tag = os.path.split(a.resume_from)
        J = real_json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        ra = J["args"]
        for key in ("m", "g", "rs", "R0_over_rs", "reading", "nsites", "ell", "kappa_sea", "tau", "half_offset", "site_fields", "shell", "reference",
                    "c_LZ", "window_max", "stop_E", "stop_fields", "maxit", "mix", "collar_policy", "support_rel"):
            if not E._num_equal(ra.get(key), getattr(a, key)):
                raise RuntimeError("slice resume: the stored run's %s = %r differs from this run's %r" % (key, ra.get(key), getattr(a, key)))
        S = np.load(SLICE, allow_pickle=True)
        rs = a.rs
        T_res = float(S["T"])
        if abs(T_res / rs - a.resume_T_over_rs) > 1e-9:
            raise RuntimeError("slice resume: the slice holds T/rs %.6f, not the requested %g" % (T_res / rs, a.resume_T_over_rs))
        if str(S["dtype"]) != "complex128":
            raise RuntimeError("slice resume: the slice is %s, not complex128 (not restartable: engine_slice_restart_check_v1)" % S["dtype"])
        Tz = np.asarray(Z["T"], float)
        hits = np.where(np.abs(Tz - T_res) < 1e-9)[0]
        if not len(hits):
            raise RuntimeError("slice resume: the stored fields hold no record at T/rs %g" % (T_res / rs))
        k_res = int(hits[0])
        rec = next(r for r in J["records"] if r.get("E") is not None and abs(r["T"] - T_res) < 1e-9)
        N_app = np.asarray(S["N_applied"], float); v_app = np.asarray(S["v_applied"], float)
        dNa = float(np.max(np.abs(N_app - np.asarray(Z["N_applied"][k_res], float)))); dva = float(np.max(np.abs(v_app - np.asarray(Z["v_applied"][k_res], float))))
        dev = E.BK["dev"]
        cols = {}
        eps = np.zeros(a.nsites); Jt = np.zeros(a.nsites)
        for ch in chs:
            sea = torch.as_tensor(np.asarray(S["sea_%d" % ch.kappa]), dtype=torch.complex128, device=dev)
            du = torch.as_tensor(np.asarray(S["dust_%d" % ch.kappa]), dtype=torch.complex128, device=dev)
            f = np.asarray(S["f_%d" % ch.kappa], float)
            ref = torch.as_tensor(np.asarray(S["R2_%d" % ch.kappa]), dtype=torch.complex128, device=dev)
            eps += ch.w * (E.dens(ch.h, sea) - E.dens(ch.h, ref) + E.dens(ch.h, du, f))
            Jt += ch.w * (E.dens(ch.Q, sea) - E.dens(ch.Q, ref) + E.dens(ch.Q, du, f))
            cols[ch.kappa] = (sea, du, f, ref)
        es = np.asarray(Z["eps_tot"][k_res], float); Js = np.asarray(Z["J_tot"][k_res], float)
        de = float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300)
        dj = float(np.max(np.abs(Jt - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
        print("slice resume: %s at T/rs %.3f (%s); the slice's applied fields against the stored ones: N %.2e, v %.2e; its sources against the stored shell totals: eps %.2e, J %.2e (acceptance 1e-8)" % (
            os.path.basename(SLICE), T_res / rs, S["dtype"], dNa, dva, de, dj), flush=True)
        if max(de, dj) > 1e-8 or max(dNa, dva) > 0.0:
            raise RuntimeError("slice resume: the gate fails (sources eps %.2e, J %.2e; applied fields %.2e, %.2e)" % (de, dj, dNa, dva))
        st = {}
        de_max = 0.0
        cont = rec.get("continuation") or {}
        for ch in chs:
            sea, du, f, ref = cols.pop(ch.kappa)
            e, U = E.eigT(E.clocked_gpu(ch, N_app, v_app))
            de_max = max(de_max, float(np.max(np.abs(e.real.cpu().numpy() - np.asarray(S["e_%d" % ch.kappa], float)))))
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            prop = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            gap = (cont.get(str(ch.kappa)) or {}).get("gap_between_blocks")
            st[ch.kappa] = {"sea": prop(sea), "dust": (prop(du) if du.shape[1] else du), "f": f, "ref": ref, "n_sea": int(sea.shape[1]),
                            "window": [{"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": gap}]}
            del e, U, sea
        fields = {}
        for key in ("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied"):
            fields[key] = [np.asarray(x, float) if key != "T" else float(x) for x in Z[key][:k_res + 1]]
        for key in ("r_nodes", "N_nodes", "v_nodes"):
            fields[key] = [np.asarray(x, float) for x in Z[key][:k_res + 1]]
        recs = [r for r in J["records"] if r.get("E") is not None and r["T"] <= T_res + 1e-9]
        head = dict(J["head"])
        resume = {"from": a.resume_from, "from_slice": SLICE, "slice_dtype": str(S["dtype"]), "T_over_rs": T_res / rs, "script": J.get("script"),
                  "gate_eps_rel": de, "gate_J_rel": dj, "applied_fields_max_diff": [dNa, dva], "eigenvalues_max_diff": de_max, "window_restarted_with": 1,
                  "stored_records_kept": len(recs), "seconds": time.time() - t0,
                  "superseded_records_after": [r["T"] / rs for r in J["records"] if r.get("E") is not None and r["T"] > T_res + 1e-9]}
        RUNNER["resume"] = {k: v for k, v in resume.items() if k != "superseded_records_after"}
        print("slice resume: the eigenvalues at the slice's applied fields against its stored ones: max %.2e; the sea and dust propagated past T/rs %.3f; window restarted with 1 entry; %d stored records kept (%.0f s)" % (
            de_max, T_res / rs, len(recs), resume["seconds"]), flush=True)
        return st, (N_app.copy(), v_app.copy()), {"records": recs, "fields": fields, "N_cinit": np.asarray(Z["N_cinit"], float), "v_cinit": np.asarray(Z["v_cinit"], float),
                                                   "k_res": k_res, "T_next": T_res + a.tau}, head, resume
    E.resume_state = resume_from_slice
    print("runner: %s; slice %s; slices not written below %.1f GB free" % (RUNNER["engine"], SLICE, MIN_FREE_GB), flush=True)
    E.main()


if __name__ == "__main__":
    main()
