#!/usr/bin/env python
"""engine_ops_v4_vf_unit_v1.py (builder analyst; the brief BRIEF_v26_builder_S4_S9, the unit tests: "S4 (d) ... on vf9's channels 1 to 4
(the vacuum-flow chain of engine_vacflow_v9/v10 where the overlap rule was ambiguous): ops v4 against ops v3 and against the propagated
state").

vf9's (and vf10's) set-up: r_s 32, the exact Painleve-Gullstrand profile, m 0, the full chain current, the ramp s(t) = sin^2(pi t/(2 t_r)) over
t_r = 32 r_s and the hold of 2 r_s, readouts every 0.5 r_s, the propagation by engine_vacflow_v7's commutator-free fourth-order scheme
(dt 0.5); per channel kappa (--kappas 1,2,3,4) the propagated sea X(t) from the flat sea and, at every readout from the same
eigendecomposition of h(s) = h_0 + s V, three continued references each from its own window as the engine keeps it: engine_ops_v3
(continue_sea_v3, vf10's), engine_ops_v4 with the Weyl isolation (the default) and ops v4 with the angle variant (all at c_LZ 1, the
readout interval as the step). Printed per channel: the real-skew defect of h at s = 1 (max |h + h^*| over max |h|), d(X, P) = n_sea -
||X^dag P||_F^2 (= ||X X^dag - P||_F^2/2) for each reference at the ramp's end and the hold's end and its largest, d(P_v4, P_v3) likewise,
the readouts with a transported pair, the transported pair-steps, the largest transport residual, the readouts failing the certificate
(2 eta_n >= g_n), the largest pocket count, and the first readout where v4 and v3 differ (d above 1e-10). Every number is printed by the
script. Args: --rs --kappas --tr --hold --dt --every --gpu --md --json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v3 as OPS3  # noqa: E402
import engine_ops_v4 as OPS4  # noqa: E402
import engine_vacflow_v7 as V7  # noqa: E402


def run_channel(kappa, Rfull, a, dev):
    import torch
    rs = a.rs
    n0 = int(np.searchsorted(Rfull, kappa / a.Ecut))
    R = Rfull[n0:]
    h0, Q = V7.chain_sparse(R, 0.0, kappa)
    V = V7.flow_term(Q, V7.profile(R, rs, "exact"))
    H1 = (h0 + V).tocsr()
    skew = float(abs(H1 + H1.conj()).max()) / float(abs(H1).max())
    eig = lambda s: torch.linalg.eigh(torch.as_tensor((h0 + s * V).toarray(), device=dev))
    e0, U0 = eig(0.0)
    e0r = e0.real
    nsea = int((e0r < 0).sum())
    X = U0[:, e0r < 0].clone()
    P = V7.Prop(h0, V, dev)
    t_r = a.tr * rs
    env = lambda t: math.sin(math.pi * t / (2 * t_r)) ** 2
    nst = max(1, int(math.ceil(t_r / a.dt - 1e-9)))
    h = t_r / nst
    every = a.every * rs
    per = max(1, int(round(every / h)))
    Vh = V.tocsr()
    h_of = lambda ch_, ds_, dv_: Vh * float(ds_)
    eig_at = lambda ch_, s_, v_: (lambda ee: (ee[0].real, ee[1]))(eig(float(s_)))
    e0n = e0r.cpu().numpy()
    gap0 = float(e0n[nsea] - e0n[nsea - 1]) if 0 < nsea < len(e0n) else None
    rules = {"v3": None, "v4_weyl": "weyl", "v4_angle": "angle"}
    W = {}
    for lab, iso in rules.items():
        w0 = {"ref": X.clone(), "f": (0.0, 0.0), "gap": gap0, "weyl_gap": gap0, "mask": (e0n < 0)}
        if iso is not None:
            opts = dict(OPS4.OPTS); opts["isolation"] = iso
            w0["lines"] = OPS4.initial_lines(e0r, U0, opts["n_track"])
            W[lab] = {"window": [w0], "opts": opts}
        else:
            W[lab] = {"window": [w0], "opts": None}
    st = {lab: {"unverified": 0, "transport_readouts": 0, "transported": 0, "max_transport_residual": 0.0, "max_pockets": 0, "d_end_ramp": None,
                "d_end": None, "d_max": 0.0, "first_unverified_t": None} for lab in rules}
    cross = {"v4_weyl": {"d_end_ramp": None, "d_end": None, "d_max": 0.0, "first_differ_t": None}, "v4_angle": {"d_end_ramp": None, "d_end": None, "d_max": 0.0, "first_differ_t": None}}
    pairs_log = []
    reads = 0
    t0 = time.time()

    def readout(t, s, phase):
        nonlocal reads
        e, U = eig(s)
        er = e.real
        refs = {}
        for lab, iso in rules.items():
            w = W[lab]
            if iso is None:
                ref, ci, mask = OPS3.continue_sea_v3(None, w["window"], (s, 0.0), er, U, nsea, every, a.c_LZ, eig_at, h_of)
                w["window"] = [{"ref": ref, "f": (s, 0.0), "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap"), "mask": mask}]
            else:
                ref, ci, mask = OPS4.continue_sea_v4(None, w["window"], (s, 0.0), er, U, nsea, every, a.c_LZ, eig_at, h_of, opts=w["opts"])
                w["window"] = [{"ref": ref, "f": (s, 0.0), "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap"), "mask": mask, "lines": ci.get("lines")}]
                pl = (ci.get("oriented") or {}).get("pairs_all_substeps") or []
                if pl:
                    st[lab]["transport_readouts"] += 1; st[lab]["transported"] += len(pl)
                    st[lab]["max_transport_residual"] = max(st[lab]["max_transport_residual"], max(q["transport_residual"] for q in pl))
                    if len(pairs_log) < 300:
                        pairs_log.extend([dict(q, rule=lab, t_over_rs=t / rs) for q in pl[:4]])
            refs[lab] = ref
            if not ci.get("verified"):
                st[lab]["unverified"] += 1
                if st[lab]["first_unverified_t"] is None:
                    st[lab]["first_unverified_t"] = t / rs
            st[lab]["max_pockets"] = max(st[lab]["max_pockets"], int(ci.get("n_pocket") or 0))
            d = float(nsea - float((X.conj().T @ ref).abs().pow(2).sum()))
            st[lab]["d_max"] = max(st[lab]["d_max"], d)
            if phase == "ramp":
                st[lab]["d_end_ramp"] = d
            st[lab]["d_end"] = d
        for lab in ("v4_weyl", "v4_angle"):
            dc = float(nsea - float((refs[lab].conj().T @ refs["v3"]).abs().pow(2).sum()))
            cross[lab]["d_max"] = max(cross[lab]["d_max"], dc)
            if phase == "ramp":
                cross[lab]["d_end_ramp"] = dc
            cross[lab]["d_end"] = dc
            if dc > 1e-10 and cross[lab]["first_differ_t"] is None:
                cross[lab]["first_differ_t"] = t / rs
        reads += 1
        del e, U, refs
    for i in range(nst):
        t = i * h
        s1, s2 = env(t + V7.CF_C[0] * h), env(t + V7.CF_C[1] * h)
        X = P.expm(X, h / 2, 2 * (V7.CF_A[0] * s1 + V7.CF_A[1] * s2))
        X = P.expm(X, h / 2, 2 * (V7.CF_A[1] * s1 + V7.CF_A[0] * s2))
        if (i + 1) % per == 0 or i + 1 == nst:
            readout((i + 1) * h, env((i + 1) * h), "ramp")
    nh = max(1, int(round(a.hold * rs / every)))
    for j in range(nh):
        X = P.expm(X, a.hold * rs / nh, 1.0)
        readout(t_r + (j + 1) * a.hold * rs / nh, 1.0, "hold")
    orth = float(torch.linalg.norm(X.conj().T @ X - torch.eye(nsea, dtype=X.dtype, device=X.device)))
    return {"kappa": kappa, "n0": n0, "dim": int(2 * len(R)), "nsea": nsea, "realskew_defect": skew, "readouts": reads, "stats": st, "v4_vs_v3": cross,
            "pairs_log": pairs_log, "orthonormality_error": orth, "seconds": time.time() - t0, "propagator_bound": P.bound}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--kappas", default="1,2,3,4"); ap.add_argument("--Ecut", type=float, default=6.0)
    ap.add_argument("--tr", type=float, default=32.0); ap.add_argument("--hold", type=float, default=2.0); ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--every", type=float, default=0.5); ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    V7.CC["full"] = True
    import torch
    dev = torch.device("cuda:%d" % a.gpu) if a.gpu is not None else torch.device("cpu")
    Rfull = np.arange(int(64 * a.rs), dtype=float) + 1.0
    out = {"script": "engine_ops_v4_vf_unit_v1.py", "args": vars(a), "channels": []}
    for kappa in [int(x) for x in a.kappas.split(",")]:
        c = run_channel(kappa, Rfull, a, dev)
        out["channels"].append(c)
        json.dump(out, open(a.json, "w"), indent=1, default=float)
        print("kappa %d done: %d readouts, %.0f s; d(X, P) at the end v3 %.6e v4 weyl %.6e v4 angle %.6e; d(v4 weyl, v3) %.3e d(v4 angle, v3) %.3e; transported (weyl, angle) %d, %d" % (
            kappa, c["readouts"], c["seconds"], c["stats"]["v3"]["d_end"], c["stats"]["v4_weyl"]["d_end"], c["stats"]["v4_angle"]["d_end"], c["v4_vs_v3"]["v4_weyl"]["d_end"],
            c["v4_vs_v3"]["v4_angle"]["d_end"], c["stats"]["v4_weyl"]["transported"], c["stats"]["v4_angle"]["transported"]), flush=True)
    L = ["## S4 (d) on vf9's channels: engine_ops_v4 against engine_ops_v3 and against the propagated sea (engine_ops_v4_vf_unit_v1.py)", "",
         "- vf9's set-up: r_s %g, the exact profile, m 0, the full chain current, t_r %g r_s, hold %g r_s, readouts every %g r_s, dt %g; the continuation step is the readout interval, c_LZ %g; d(X, P) = n_sea - ||X^dag P||_F^2 with X the propagated sea." % (
             a.rs, a.tr, a.hold, a.every, a.dt, a.c_LZ), "",
         "| kappa | dimension | n_sea | real-skew defect | readouts | d(X, v3) ramp end; hold end; largest | d(X, v4 weyl) ramp end; hold end; largest | d(X, v4 angle) ramp end; hold end; largest | d(v4 weyl, v3) hold end; largest; first t/r_s above 1e-10 | d(v4 angle, v3) hold end; largest; first t/r_s | transported pair-steps (readouts) weyl; angle | largest transport residual weyl; angle | unverified readouts v3; weyl; angle | largest pockets v3; weyl; angle | orthonormality of X |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in out["channels"]:
        S = c["stats"]; C = c["v4_vs_v3"]
        f = lambda x: ("%.6e" % x) if x is not None else "-"
        L.append("| %d | %d | %d | %.1e | %d | %s; %s; %s | %s; %s; %s | %s; %s; %s | %s; %s; %s | %s; %s; %s | %d (%d); %d (%d) | %.2e; %.2e | %d; %d; %d | %d; %d; %d | %.1e |" % (
            c["kappa"], c["dim"], c["nsea"], c["realskew_defect"], c["readouts"], f(S["v3"]["d_end_ramp"]), f(S["v3"]["d_end"]), f(S["v3"]["d_max"]),
            f(S["v4_weyl"]["d_end_ramp"]), f(S["v4_weyl"]["d_end"]), f(S["v4_weyl"]["d_max"]), f(S["v4_angle"]["d_end_ramp"]), f(S["v4_angle"]["d_end"]), f(S["v4_angle"]["d_max"]),
            f(C["v4_weyl"]["d_end"]), f(C["v4_weyl"]["d_max"]), ("%.2f" % C["v4_weyl"]["first_differ_t"]) if C["v4_weyl"]["first_differ_t"] is not None else "never",
            f(C["v4_angle"]["d_end"]), f(C["v4_angle"]["d_max"]), ("%.2f" % C["v4_angle"]["first_differ_t"]) if C["v4_angle"]["first_differ_t"] is not None else "never",
            S["v4_weyl"]["transported"], S["v4_weyl"]["transport_readouts"], S["v4_angle"]["transported"], S["v4_angle"]["transport_readouts"],
            S["v4_weyl"]["max_transport_residual"], S["v4_angle"]["max_transport_residual"], S["v3"]["unverified"], S["v4_weyl"]["unverified"], S["v4_angle"]["unverified"],
            S["v3"]["max_pockets"], S["v4_weyl"]["max_pockets"], S["v4_angle"]["max_pockets"], c["orthonormality_error"]))
    pl = [q for c in out["channels"] for q in c["pairs_log"]][:40]
    if pl:
        L += ["", "### Transported pair-steps (the first, over the channels in order)", "", "| rule | t/r_s | eps | sep | eta | smallest principal cosine | det O | signed energy | occupied | transport residual |", "|---|---|---|---|---|---|---|---|---|---|"]
        for q in pl:
            L.append("| %s | %.2f | %.3e | %.3e | %.3e | %.6f | %+.0f | %+.4e | %s | %.2e |" % (q["rule"], q["t_over_rs"], q["eps"], q["sep"], q["eta"], q["cos_min"], q["det_O"], q["signed_energy"], q["occupied"], q["transport_residual"]))
    L += ["", "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
