"""b4_args_v1.py (busi f52b1152, 20 Oct; REVIEWA_B4_rs16_v1.md): B4's engine_collapse_v19 argument lists, rebuilt from B2's record json
(collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json, md5 417a4018) with only the ruled substitutions.
  h16   : --rs 16 --nsites 1024 --tau 2 --T-over-rs 100           (the history, tag c19_rs16_A_R2_tau2_ell4_avg_B4n1024)
  twin8 : the same with --tau 8                                    (the tau 8 twin, tag c19_rs16_A_R2_tau8_ell4_avg_B4n1024)
  f24   : --rs 24 --nsites 1536 --tau 3 --T-over-rs 100           (the fallback, tag c19_rs24_A_R2_tau3_ell4_avg_B4n1536)
Besides the physics substitutions, only the bookkeeping changes: --out B4, --tag, --slice-dir states/<tag> (B2's own pattern), and B2's
crash-restart pair (resume_from, resume_T_over_rs) is dropped because every B4 line starts at T = 0. --gpu stays 0 (the card's device under
CUDA_VISIBLE_DEVICES); --gpu none gives the CPU route (the dry check only).
Usage: python -I b4_args_v1.py <B2 json> h16|twin8|f24 [--gpu none] [--T-over-rs x] [--out d]    -> prints the argument line
       python -I b4_args_v1.py <B2 json> diff h16|twin8|f24                                      -> prints the key-by-key diff against B2"""
import json, sys

ORDER = [("m", "--m"), ("g", "--g"), ("rs", "--rs"), ("R0_over_rs", "--R0-over-rs"), ("reading", "--reading"), ("shell", "--shell"),
         ("nsites", "--nsites"), ("ell", "--ell"), ("kappa_sea", "--kappa-sea"), ("tau", "--tau"), ("T_over_rs", "--T-over-rs"),
         ("reference", "--reference"), ("half_offset", "--half-offset"), ("benchmark", "--benchmark"), ("collar_policy", "--collar-policy"),
         ("support_rel", "--support-rel"), ("gpu", "--gpu"), ("out", "--out"), ("tag", "--tag"), ("site_fields", "--site-fields"),
         ("stop_E", "--stop-E"), ("stop_fields", "--stop-fields"), ("maxit", "--maxit"), ("mix", "--mix"), ("delta_rtol_every", "--delta-rtol-every"),
         ("commutator_every", "--commutator-every"), ("streams", "--streams"), ("m1_simpson_steps", "--m1-simpson-steps"), ("c_LZ", "--c-LZ"),
         ("window_max", "--window-max"), ("resume_from", "--resume-from"), ("resume_T_over_rs", "--resume-T-over-rs"),
         ("slices_c128", "--slices-c128"), ("slices_c64", "--slices-c64"), ("slice_dir", "--slice-dir")]
MODES = {"h16": dict(rs=16.0, nsites=1024, tau=2.0, T_over_rs=100.0, tag="c19_rs16_A_R2_tau2_ell4_avg_B4n1024"),
         "twin8": dict(rs=16.0, nsites=1024, tau=8.0, T_over_rs=100.0, tag="c19_rs16_A_R2_tau8_ell4_avg_B4n1024"),
         "f24": dict(rs=24.0, nsites=1536, tau=3.0, T_over_rs=100.0, tag="c19_rs24_A_R2_tau3_ell4_avg_B4n1536")}


def b4_args(b2, mode, gpu="keep", T_over_rs=None, out=None):
    a = dict(b2)
    if set(a) != {k for k, _ in ORDER}:
        raise SystemExit("B2's args keys differ from the engine's parser: %s" % sorted(set(a) ^ {k for k, _ in ORDER}))
    sub = MODES[mode]
    for k in ("rs", "nsites", "tau", "T_over_rs"):
        a[k] = sub[k]
    a["tag"] = sub["tag"]; a["out"] = out or "B4"; a["slice_dir"] = "states/%s" % sub["tag"]
    a["resume_from"] = None; a["resume_T_over_rs"] = None
    if gpu == "none":
        a["gpu"] = None
    if T_over_rs is not None:
        a["T_over_rs"] = float(T_over_rs)
    return a


def cli(a):
    out = []
    for k, f in ORDER:
        v = a[k]
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                out.append(f)
            continue
        if isinstance(v, float) and v == int(v) and abs(v) >= 1:
            s = "%g" % v
        else:
            s = repr(v) if isinstance(v, float) else str(v)
        if s == "":
            continue
        out += [f, s]
    return out


if __name__ == "__main__":
    b2 = json.load(open(sys.argv[1]))["args"]
    if sys.argv[2] == "diff":
        a = b4_args(b2, sys.argv[3])
        for k, _ in ORDER:
            if a[k] != b2[k]:
                print("%-18s B2 %-60s -> %s" % (k, json.dumps(b2[k]), json.dumps(a[k])))
        sys.exit(0)
    kw = {}
    rest = sys.argv[3:]
    for i in range(0, len(rest), 2):
        kw[rest[i].lstrip("-").replace("-", "_")] = rest[i + 1]
    a = b4_args(b2, sys.argv[2], gpu=kw.get("gpu", "keep"), T_over_rs=kw.get("T_over_rs"), out=kw.get("out"))
    print(" ".join(cli(a)))
