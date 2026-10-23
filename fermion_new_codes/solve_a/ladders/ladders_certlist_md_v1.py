#!/usr/bin/env python3
"""ladders_certlist_md_v1.py: markdown tables of ladders_certlist_v1.json (no recomputation). Usage: ladders_certlist_md_v1.py IN.json OUT.md"""
import json, os, sys


def f(x, fmt="%.4f"):
    return "-" if x is None else (fmt % x if isinstance(x, (int, float)) else str(x))


def main():
    d = json.load(open(sys.argv[1]))
    L = ["| member | role | (m a, r_s/a, sigma/a, tau, sites, solver) | engine (from) | engine md5 (box copies) | T requested; reached | completion | certified prefix to | first non-counting (gate) | formation (class) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        md5s = sorted(set(x["md5"] for x in r["engine_box_copies"]))
        comp = "finished (reached its requested end)" if r["reached_request"] else "stopped at %s of %g" % (f(r["T_last"]), r["T_requested"])
        L.append("| %s | %s | (%g, %g, %g, %g, %d, %s) | %s (%s) | %s | %g; %s | %s | %s | %s | %s |" % (
            r["name"], r["role"], r["m"], r["rs_a"], r["sigma_a"], r["tau"], r["nsites"], r["solver"], r["engine"], r["engine_from"],
            ", ".join(md5s) if md5s else "no box copy found", r["T_requested"], f(r["T_last"]), comp,
            f(r["certified_prefix_end"]) if r["certified_prefix_end"] is not None else "none (the first record does not count)",
            ("%s (%s)" % (f(r["first_noncounting"]), r["first_noncounting_gate"])) if r["first_noncounting"] is not None else "none",
            ("%s (%s)" % (f(r["formation_T"]), r["formation_class"])) if r["formation_T"] is not None else "no trapping zero"))
    L += ["", "### Final run files on the box and the status files (last line, verbatim)", ""]
    for r in d["rows"]:
        L.append("- **%s**: run json %s (md5 %s)." % (r["name"], r["run_json"], r["run_json_md5"]))
        for x in r["run_files"]:
            if x["path"] != r["run_json"]:
                L.append("  - %s (%d B, md5 %s)" % (os.path.basename(x["path"]), x["bytes"], x["md5"]))
        if r["status"]:
            L.append("  - status %s (md5 %s): \"%s\"" % (r["status"]["path"], r["status"]["md5"], r["status"]["last_lines"][-1].replace("|", "/")))
        else:
            L.append("  - status: no ladder status file (not a ladder-queue run; its provenance is in its own folder)")
    open(sys.argv[2], "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
