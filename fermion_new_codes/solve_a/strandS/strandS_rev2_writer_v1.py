#!/usr/bin/env python
"""Strand S revision 2 writer (builder analyst): appends to solve/results/RESULT_strandS_graded_entropy.md a section "Revision 2" holding the
emitted reports (no number retyped): the sphere (strandS_sphere_report_v2.py), the static chart's density of states D-a1
(strandS_dos_report_v1.py over strandS_dos_v3.py's sizes), the flow chart's D-a2 (strandS_dos_flow_report_v1.py), the detailed-balance table
(strandS_tdb_v1.py), and the pointer to task 3's B0 rows (the ray-map temperature on the record's package).  Refuses to append twice.
Args: --sphere-md --da1-md --da2-md --tdb-md --b-md --stamp --result."""
import argparse


def demote(path):
    return ["##" + x if x.startswith("#") else x for x in open(path).read().split("\n")]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--sphere-md", "--da1-md", "--da2-md", "--tdb-md", "--b-md", "--stamp", "--result"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    t = open(a.result).read()
    if "## Revision 2 (" in t:
        raise SystemExit("revision 2 already appended")
    L = ["", "## Revision 2 (%s): the spherical cut, the ray-map temperature, the densities of states and the detailed-balance table" % a.stamp, "",
         "Builder analyst. Every table below is emitted by the script named in its title; this section is assembled by strandS_rev2_writer_v1.py. "
         "Records: box sS_entropy/S3v3 (strandS_sphere_v3.py, the six normals), S3v5 (strandS_sphere_v5.py: the spacings 0.025 and 0.0125, the "
         "controls of the judgment's sec. 8.5(6)), S3v3ctl (strandS_sphere_v4.py, the cap-1024 control), Da1v3 (strandS_dos_v3.py), Da2 "
         "(strandS_dos_flow_v1.py); task 3's B rows from flow_B_conversion_v1.py's report (the record's package, the planar jet).", ""]
    L += ["### R2.1 The ray-map temperature (task 3's B0 and B1 on the record's package; the temperature first)", ""]
    L += [x for x in demote(a.b_md)]
    L += ["", "### R2.2 The spherical cut", ""] + demote(a.sphere_md)
    L += ["", "### R2.3 Diagnostic D-a1: the static chart's density of states", ""] + demote(a.da1_md)
    L += ["", "### R2.4 Diagnostic D-a2: the flow chart's density of states", ""] + demote(a.da2_md)
    L += ["", "### R2.5 The detailed-balance temperature against T_H and the ray map", ""] + demote(a.tdb_md)
    open(a.result, "a").write("\n".join(L) + "\n")
    print("appended %d lines" % len(L))


if __name__ == "__main__":
    main()
