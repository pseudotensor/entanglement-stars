"""slice_c64_to_c128_v1.py (solve builder 8, analyst; engine spec 49.3 (1), route (a)): the v32 member's complex64 slice made readable
by engine_collapse_v36b's --resume-slice (version 34 (a), which takes complex128 slices only), with no engine change.

Per channel: the stored sea and ball orbitals and the reference R2's columns are cast to complex128 and re-orthonormalized by the symmetric
(Lowdin) map V -> V (V^dag V)^(-1/2); the sea and ball columns jointly (one occupied family, mutually orthogonal by construction), R2 on its own.
Printed per channel: the Gram deviation max |V^dag V - 1| before (the complex64 rounding) and after, the sea-ball cross block, and the largest
entry moved. The applied fields, eigenvalues, occupations, masks and clock are copied unchanged; 'dtype' becomes complex128 and the source
slice's name and md5 are kept in 'converted_from'. Also writes a copy of the run's 4 files whose json args carry v36b's two store_true flags
(massless_ball, fields_v17) as False where the v32 json lacks them (the v32 engine had neither option); nothing else in the copy changes.

Usage: python slice_c64_to_c128_v1.py --slice S --run DIR/TAG --out-slice S128 --out-run DIR2 [--json-out gram.json]
"""
import argparse, hashlib, json, os, shutil, sys
import numpy as np


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def lowdin(V):
    S = V.conj().T @ V
    dev = float(np.max(np.abs(S - np.eye(S.shape[0])))) if S.size else 0.0
    if not S.size:
        return V, dev, S
    lam, W = np.linalg.eigh(S)
    if lam.min() <= 0.5:
        raise SystemExit("Gram matrix not near the identity (smallest eigenvalue %.3e): not a rounding" % lam.min())
    Vn = V @ ((W * lam ** -0.5) @ W.conj().T)
    return Vn, dev, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-slice", required=True)
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    Z = np.load(a.slice)
    if str(Z["dtype"]) != "complex64":
        raise SystemExit("expected a complex64 slice, found %s" % Z["dtype"])
    out = {k: Z[k] for k in Z.files}
    rep = {"slice": a.slice, "slice_md5": md5(a.slice), "T_over_rs": float(Z["T"]) / float(Z["rs"]), "channels": {}}
    for k in [int(x) for x in Z["kappas"]]:
        sea = Z["sea_%d" % k].astype(np.complex128)
        dust = Z["dust_%d" % k].astype(np.complex128)
        R2 = Z["R2_%d" % k].astype(np.complex128)
        ns = sea.shape[1]
        X = np.concatenate([sea, dust], axis=1)
        Xn, dev_x, S = lowdin(X)
        cross = float(np.max(np.abs(S[:ns, ns:]))) if dust.shape[1] else 0.0
        dev_dust = float(np.max(np.abs(S[ns:, ns:] - np.eye(dust.shape[1])))) if dust.shape[1] else 0.0
        R2n, dev_r, _ = lowdin(R2)
        after_x = float(np.max(np.abs(Xn.conj().T @ Xn - np.eye(Xn.shape[1]))))
        after_r = float(np.max(np.abs(R2n.conj().T @ R2n - np.eye(R2n.shape[1]))))
        moved = max(float(np.max(np.abs(Xn - X))), float(np.max(np.abs(R2n - R2))))
        out["sea_%d" % k] = Xn[:, :ns]
        out["dust_%d" % k] = Xn[:, ns:]
        out["R2_%d" % k] = R2n
        c = {"gram_dev_occupied_before": dev_x, "gram_dev_ball_before": dev_dust, "sea_ball_cross_before": cross,
             "gram_dev_R2_before": dev_r, "gram_dev_occupied_after": after_x, "gram_dev_R2_after": after_r, "max_entry_moved": moved,
             "n_sea": ns, "n_ball": int(dust.shape[1])}
        rep["channels"][str(k)] = c
        print("channel %d: Gram deviation (sea+ball) %.3e -> %.3e; ball %.3e; sea-ball cross %.3e; R2 %.3e -> %.3e; largest entry moved %.3e"
              % (k, dev_x, after_x, dev_dust, cross, dev_r, after_r, moved), flush=True)
    out["dtype"] = np.array("complex128")
    out["converted_from"] = np.array("%s md5 %s by slice_c64_to_c128_v1 (Lowdin re-orthonormalization)" % (a.slice, rep["slice_md5"]))
    tmp = a.out_slice + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a.out_slice)
    rep["out_slice"] = a.out_slice
    rep["out_slice_md5"] = md5(a.out_slice)
    # the run copy
    rdir, rtag = os.path.split(a.run)
    os.makedirs(a.out_run, exist_ok=True)
    for suf in ("_fields.npz", "_handoff.npz", "_spectra.npz"):
        shutil.copy2(os.path.join(rdir, "collapse_%s%s" % (rtag, suf)), os.path.join(a.out_run, "collapse_%s%s" % (rtag, suf)))
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag)))
    added = []
    for k_ in ("massless_ball", "fields_v17"):
        if k_ not in J["args"]:
            J["args"][k_] = False
            added.append(k_)
    J.setdefault("head", {})["copied_for_slice_restart"] = "args %s added as False by slice_c64_to_c128_v1 (absent in the v32 engine)" % added
    json.dump(J, open(os.path.join(a.out_run, "collapse_%s.json" % rtag), "w"))
    rep["json_args_added"] = added
    rep["worst"] = {"gram_dev_before": max(max(c["gram_dev_occupied_before"], c["gram_dev_R2_before"]) for c in rep["channels"].values()),
                    "gram_dev_after": max(max(c["gram_dev_occupied_after"], c["gram_dev_R2_after"]) for c in rep["channels"].values()),
                    "sea_ball_cross": max(c["sea_ball_cross_before"] for c in rep["channels"].values())}
    print("worst: Gram deviation before %.3e, after %.3e; sea-ball cross %.3e; json args added %s" % (
        rep["worst"]["gram_dev_before"], rep["worst"]["gram_dev_after"], rep["worst"]["sea_ball_cross"], added), flush=True)
    if a.json_out:
        json.dump(rep, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
