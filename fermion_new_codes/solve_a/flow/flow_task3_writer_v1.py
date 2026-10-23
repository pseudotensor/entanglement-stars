#!/usr/bin/env python
"""Task 3 result writer, revision 1 (builder analyst): assembles solve/results/RESULT_task3_prescribed_flow.md from the emitted reports, the
radial parts (a) and (d) (flow_task3_report_v2.py over flow_chart_v4.py's records) and the conversion part (b) (flow_B_conversion_v1.py's
report over the record's conversion package), with the header, the builder's notes (their numbers read from the JSONs) and the labels; no
number is retyped.  Args: --ad-md --ad-json --b-md --b-json --checks-json --stamp --out."""
import argparse
import json


def demote(md):
    return ["#" + x if x.startswith("##") else ("## " + x[2:] if x.startswith("# ") else x) for x in open(md).read().split("\n")]


def f(v, d=4):
    return "-" if v is None else "%.*g" % (d, v)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--ad-md", "--ad-json", "--b-md", "--b-json", "--checks-json", "--stamp", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    AD = json.load(open(a.ad_json))
    B = json.load(open(a.b_json))
    brows = [r for r in B["rows"] if "plateau" in r and r["branch"] == 0]
    b0 = [r for r in brows if r["m_over_kappa"] == 0 and r["record_theta"] is not None]
    worst_b0 = max(abs(r["theta_minus_record"]) for r in b0) if b0 else None
    b1 = [r for r in brows if r["m_over_kappa"] > 0]
    closed = [r for r in B["rows"] if "closed" in r]
    d3 = AD.get("D3", {})
    d3_ret = [v["final_beyond_Rref"] for v in d3.values()]
    L = ["# Task 3 result, revision 1 (%s): the prescribed-flow reproduction with H_*'s operator, parts (a), (b) and (d)" % a.stamp, "",
         "Builder: analyst. Specs: solve/specs/SPEC_task3_prescribed_flow.md (lead, final4 turn 9 F). Every number below is labelled \"prescribed "
         "flow (simpler version); v declared\" (the PG profile v = -sqrt(r_s/R) continued by the ball v = -sqrt(r_s/R_m^3) R inside R_m, lapse 1, "
         "ruler 1; the planar jet of the record's conversion package); the tables are emitted by the scripts named; the builder's notes are labelled.", "",
         "## Header",
         "- Code: solve/code/flow/flow_chart_v4.py (the lead's draft flow_chart_v1.py completed: versions 2 to 4; version 4 carries the code fixes of "
         "the reviewer B red team of the derivations, reviewb/review_output/reviewb_azure_redteam_flow_derivations_20261014_205632/result.md items 1, 9.1, 9.4, "
         "9.5), flow_task3_report_v2.py; part (b): the record's conversion package (review_output/layer_response_2026-10-02/spectro_work: lattice.py, "
         "campaign.py, tscan_report.py, unchanged; copies in <internal>/record_spectro) driven by flow_B_conversion_v1.py; "
         "launchers launch_flow_task3_v4_v1.sh, launch_flow_task3_A2rm8_v1.sh, launch_flow_B_v1.sh. Box: <internal>/flow (task3v4, B).",
         "- Records: solve/results/task3/ (the JSONs and the emitted reports).", "",
         "## Builder's notes (labelled; for the lead)",
         "- (1) The draft's chain_hamiltonian completed the Hermitian matrix as h + triu(h, 1)^H, which counts the on-site y SY entry twice; eigh (lower "
         "triangle) then diagonalized a centrifugal term 2 kappa_j/R in every radial channel (found by the reviewer B red team, item 1). Fixed in version 4 "
         "(h = triu(h) + triu(h, 1)^H) with the radial Hermiticity and on-site block checks; every radial record of versions 2 and 3 is superseded; "
         "the planar runs (y = 0) are unchanged.",
         "- (2) Part (b) runs on the record's conversion package (the goal names the earlier flow-term computations as the template): its planar "
         "jet profile, ramp, packets in the frozen lead and plateau window are the record's, so B0 is the record's computation repeated with H_*'s "
         "operator (the same sine and range-two hops), and B1 is that computation with the light packet's mass; the draft's convert mode (a tanh "
         "profile, Killing-frequency packets) is not used for (b) (BUILDER, labelled). B0 against the record: the largest |theta_r - record| %s "
         "over the rows with a record value. The package reports a ratio below the port's threshold omega_th = m as closed: %d closed rows." % (
             f(worst_b0, 3), len(closed)),
         "- (3) A2 at the spec's r_s = 64 cannot hold the outgoing carriers (their sigma_x = 16 to 32 overlap the centre at their launch radii); the "
         "outgoing scan runs at r_s = 256 (R_m = 64) and r_s = 1024 (R_m = 256 and 128: at R_m = r_s/4 the 3 pi/16 carrier's R_turn = 0.21 r_s lies "
         "inside the ball, where the profile is linear); the k0 = pi/4 carrier has R_turn = 0 and is launched at 0.25, 0.5, 0.75 r_s.",
         "- (4) D3: the final norm beyond 2 r_s ranges over %s to %s of the packet over the frequencies run; the returning part leaves at finite k "
         "after reflection at the centre; the declared ball's profile has |v| = 1 again at R = sqrt(R_m^3/r_s) (an inner surface); the momentum split "
         "at [pi/4, 3pi/4] inside r_s is in the D3 table. (Builder's reading, labelled: the return is the turning law's finite-wavelength escape fed "
         "through the declared regular centre.)" % (f(min(d3_ret), 3) if d3_ret else "-", f(max(d3_ret), 3) if d3_ret else "-"),
         "- (5) D3 at m = 0.1 uses omega r_s in {4, 8} with sigma_omega = min(omega/4, (omega - m)/4): the spec's frequencies are below the threshold "
         "there, and the draft's (omega - m)/3 placed the window's edge exactly at the threshold.",
         "- (6) Records carry the script name, not a code hash (the goal's guard against hashes).", ""]
    L += ["## Parts (a) and (d): the radial runs"] + demote(a.ad_md)[1:] + ["", "## Part (b): the conversion temperature"] + demote(a.b_md)[1:]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
