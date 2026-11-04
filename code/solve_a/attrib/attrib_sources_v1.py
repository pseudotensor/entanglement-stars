#!/usr/bin/env python
"""attrib_sources_v1.py: the SOURCES lines and notes of W5_W4_formation_attribution_v1 and W3_completions_v1, generated from the
producers' json and the files' md5. Usage: python attrib_sources_v1.py --attrib-dir DIR --w3-dir DIR --code-dir DIR --out FILE"""
import argparse
import hashlib
import json
import os

LC = "<internal>/attrib"
LR = "<internal>/attrib"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib-dir", required=True)
    ap.add_argument("--w3-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(os.path.join(a.attrib_dir, "attrib_readout_v1.json")))
    W = json.load(open(os.path.join(a.w3_dir, "w3_readout_v1.json")))
    H = D["histories"]
    code = {k: md5(os.path.join(a.code_dir, k)) for k in ("attrib_readout_v1.py", "attrib_compose_v1.py", "attrib_figs_v1.py", "attrib_sources_v1.py",
                                                          "w3_readout_v1.py", "w5_readout_v1.py")}
    w2 = H["W2_dg"]; w4 = H["W4b_pairsoff"]
    notes = [
        "W4 (b) ran on engine v36g with --pair-sources off (its launch line in <internal>/STATUS); 113 records to 14; its formation "
        "record's residual prints as stored (%s)." % (w4["formation"]["residual"],),
        "W2's history carries the conserving (dg) step on v36f through 12.375, a restart at 12.5 and the v36l overlap rule from 12.625 (W2_v36l_readout_v1); "
        "its formation %.4f lies in the dg part; its non-counting records: %s." % (
            w2["formation"]["T"], ", ".join("%.4f" % x["T"] for x in w2["convergence"]["excluded"]) or "none"),
        "The sigma/a 2 production point is the composite history (Newton across its seams); the plain rung 2 is not on the box.",
        "The tau 4 step control appears twice: the production's tau 4 record (engine v19) and W5's kappa_sea 10 tau 4 rung (v36d); both formation rows "
        "print in section 2 (%.4f at %.4f a and %.4f at %.4f a)." % (H["P64_s4_tau4"]["formation"]["T"], H["P64_s4_tau4"]["formation"]["R_t"],
                                                                       H["W5_k10_tau4"]["formation"]["T"], H["W5_k10_tau4"]["formation"]["R_t"]),
        "W4 (a) has run (busi's dust_comparator_v1c, 15 runs on the stored T = 0 profile of the W2 member, whose preparation's E0 equals the production's); "
        "the lapse-off runs end before any root (their end column).",
    ]
    srcs = ["W5 slice readout: busi's w5_readout_v1.py (md5 %s, copied unchanged to %s) run on the box CPU over the 6 rungs' T0014.00 slices; "
            "its outputs %s (md5 %s) and w5_readout_v1.md beside it; engine_vacflow_v1.py md5 5225e959b9e0bc730062041d60d800c8" % (
                code["w5_readout_v1.py"], LC, D["w5_json"], D["w5_json_md5"]),
            "W4 (a) outputs: <internal>/default (DUST_comparator_v1c.md md5 %s; the run jsons' md5 in attrib_readout_v1.json; "
            "manifest <internal>/MANIFEST_w4a.md5); the copy <internal>/default_s4" % D["w4a_files"]["md_md5"],
            "producer %s/attrib_readout_v1.py (md5 %s) with the ladder reader <internal>/ladders_readout_v1.py; "
            "composer attrib_compose_v1.py (md5 %s); figures attrib_figs_v1.py (md5 %s); this list attrib_sources_v1.py (md5 %s)" % (
                LC, code["attrib_readout_v1.py"], code["attrib_compose_v1.py"], code["attrib_figs_v1.py"], code["attrib_sources_v1.py"]),
            "outputs %s (attrib_readout_v1.json, attrib_readout_v1.npz, the figure W5_W4_formation_attribution_v1_fig.png), box mirror %s with "
            "MANIFEST_attrib_readout_v1.md5" % (LR, a.attrib_dir)]
    pc = W["histories"]["point-core"]
    w3n = ["The completions start at T/r_s 11.125 (the last gated record by busi's gate arrays), resumed from rung 4's T0011.00 slice, not at 11.625.",
           "point-core: %d records, last %.4f (its run ended rc 1, chain.log); it enters the spreads only where it holds a record." % (pc["records"], pc["convergence"]["last"]),
           "The completions' files were recovered from the Drive mirror gdrive:qi2-fermion-bulk/solve_a_bulk/w3/pod4 (the box copy was moved) "
           "into <internal>/w3_from_gdrive; every json and fields npz md5 equals <internal>/MANIFEST_w3.md5.",
           "The far dust is printed as the bare shell energy of the dust (the fields rows' eps_tot - eps_pairs) beyond 16 and 24 r_s, not the count: "
           "the count and its Killing split need the slices, which are on the Drive only; not composed.",
           "Not composed (they need the slices or the light channel's runs): E_rad^(0)/E, the near-cut and 16 r_s excess, the fixed-cut transfer of the W3 item.",
           "The late root: the own history holds a second episode from 18.375 open to 40; the completions' episodes are in section 1; the R_t spread "
           "is printed only where at least 2 histories hold a root."]
    w3s = ["producer %s/w3_readout_v1.py (md5 %s); composer attrib_compose_v1.py; figure W3_completions_v1_fig.png; outputs %s (w3_readout_v1.json, .npz), "
           "box mirror %s with MANIFEST_w3_readout_v1.md5; the chain's STATUS and chain.log recovered beside the runs" % (LC, code["w3_readout_v1.py"], LR, a.w3_dir)]
    json.dump({"attrib_notes": notes, "attrib_sources": srcs, "w3_notes": w3n, "w3_sources": w3s, "code_md5": code}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
