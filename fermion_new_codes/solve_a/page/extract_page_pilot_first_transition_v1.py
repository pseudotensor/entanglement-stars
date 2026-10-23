#!/usr/bin/env python
"""extract_page_pilot_first_transition_v1.py (builder analyst; SPEC_final4b_engine.md 35.3 (1), (2), (4)): from a snapshot of pod 1's
<internal>/pilot32_v5.log (NUL bytes, left where /workspace dropped writes at 16:43:35 UTC, removed first) it writes
(a) the extraction file of 35.3 (1), the lines verbatim: stage 0's end line, every "transition check (8.3 (8))" line, the transition's
    "(33.1 (6)): device peak" line (the code numbers it "transition 0 -> 1": k_from 0, the stage index of the code), both channels'
    memory-phase lines on every attempt, the restores' lines, the first stage-1 checkpoint lines (the first run's and the resumed run's,
    with the resume line), and beside them, not in 35.3 (1)'s list, every per-channel transition line (the per-channel S-continuity
    defects the check lines take the maximum of) and the formation's device-memory lines;
(b) to stdout, 35.3 (2)'s rule evaluated on the check lines: the x 2 attempt's defect against the x 1 attempt's (within a factor 2, or
    below half), the kept attempt's defect against its accumulated D8 bound, and the branch;
(c) to stdout, for 35.3 (4), every memory phase at which the peak since the transition's start rose: the rise over the allocation
    before it, in GB and in n2 x n2 complex128 arrays (n2 = 2 nsites; nsites read from the log's settings line).
Usage: python extract_page_pilot_first_transition_v1.py <log snapshot> <out .md>"""
import re
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    raw = open(src, "rb").read()
    n_nul = raw.count(b"\x00")
    lines = raw.replace(b"\x00", b"").decode("utf-8", "replace").splitlines()
    nsites = None
    for ln in lines:
        m = re.search(r"nsites (\d+)", ln)
        if m and "settings:" in ln:
            nsites = int(m.group(1)); break
    unit = (2 * nsites) ** 2 * 16 / 1e9 if nsites else None
    # the first run is the part before the second "settings:" line (the resume)
    starts = [i for i, ln in enumerate(lines) if "] settings:" in ln]
    first = lines[: starts[1]] if len(starts) > 1 else lines
    resumed = lines[starts[1]:] if len(starts) > 1 else []
    pick = lambda pat, L: [ln for ln in L if re.search(pat, ln)]
    groups = [
        ("stage 0's end line", pick(r"stage 0 ends", first)),
        ("every transition check (8.3 (8)) line", pick(r"transition check \(8\.3 \(8\)\)", first)),
        ("the transition's device peak line (the code's \"transition 0 -> 1\", k_from 0)", pick(r"transition \d+ -> \d+ \(33\.1 \(6\)\): device peak", first)),
        ("both channels' memory-phase lines on every attempt", pick(r"transition memory phases", first)),
        ("the restores' lines", pick(r"restored for tau_tr", first)),
        ("the first stage-1 checkpoint lines (the first run, which ended at 16:43:35 UTC writing its block-30 checkpoint)", pick(r"stage 1 block \d+ ", first)),
        ("the resumed run (page_v5_pilot32_r1, from the checkpoint written right after the transition): its resume line and first stage-1 checkpoint lines",
         pick(r"resumed at|stage 1 block \d+ ", resumed)),
        ("beside (not in 35.3 (1)'s list): every per-channel transition line", pick(r"kappa \d transition \(tau_tr x", first)),
        ("beside: the formation's device-memory lines", pick(r"device memory after the formation", first)),
    ]
    L = ["# The Page pilot's first transition: the log lines (engine spec 35.3 (1))", "",
         "Extracted verbatim by code/page/extract_page_pilot_first_transition_v1.py from %s (a snapshot of pod 1's <internal>/pilot32_v5.log; %d NUL bytes removed, left where /workspace dropped writes at 16:43:35 UTC). The pilot: page_v5_pilot32 (start 14:12:44 UTC, PID 51178, ended 16:43:35 UTC while writing its block-30 checkpoint), resumed as page_v5_pilot32_r1 (start 17:00:58 UTC) from its checkpoint of 16:43:14 UTC." % (src.split("/")[-1], n_nul), ""]
    for title, g in groups:
        L.append("## " + title)
        L.append("")
        L.append("```")
        L.extend(g if g else ["(none in the log)"])
        L.append("```")
        L.append("")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote %s (%d groups, %d lines; %d NUL bytes removed from the source)" % (out, len(groups), sum(len(g) for _, g in groups), n_nul))
    # (b) 35.3 (2)
    chk = pick(r"transition check \(8\.3 \(8\)\)", first)
    att = []
    for ln in chk:
        m = re.search(r"tau_tr x (\d+): accumulated D8 bound ([0-9.e+-]+) nats against 1 percent of S_esc ([0-9.e+-]+) \((\w+)\); leak max ([0-9.e+-]+).*S-continuity defect ([0-9.e+-]+) against 1e-6 \((\w+)\)", ln)
        if m:
            att.append({"x": int(m.group(1)), "D8": float(m.group(2)), "S1pct": float(m.group(3)), "leak": float(m.group(5)), "S": float(m.group(6)), "S_ok": m.group(7)})
    print("35.3 (2): the attempts' S-continuity defects:", ", ".join("x %d %.3e" % (a_["x"], a_["S"]) for a_ in att))
    a1 = [a_ for a_ in att if a_["x"] == 1]; a2 = [a_ for a_ in att if a_["x"] == 2]
    if a1 and a2:
        r = a2[0]["S"] / a1[0]["S"]
        kept = att[-1]
        within2 = (0.5 <= r <= 2.0)
        below_half = r < 0.5
        print("x 2 over x 1: %.4f (within a factor 2: %s; below half: %s); the kept attempt (x %d): defect %.3e against its accumulated D8 bound %.3e (at or below: %s)" % (
            r, within2, below_half, kept["x"], kept["S"], kept["D8"], kept["S"] <= kept["D8"]))
        if within2 and kept["S"] <= kept["D8"]:
            print("branch: the first (tau-independent defect): stop at the next checkpoint by exact PID and resume with --tau-tr-doublings 0")
        elif below_half:
            print("branch: the second (the defect falls with tau_tr, the ramp's): the retries stand and nothing changes")
        else:
            print("branch: neither condition of 35.3 (2) holds (the ratio lies between 2 and 0.5 outside both branches, or the bound fails): for the lead")
    # (c) 35.3 (4): the phases at which the peak rose
    if unit:
        print("35.3 (4): one n2 x n2 complex128 array (n2 = %d) = %.4f GB" % (2 * nsites, unit))
    prev_peak = None; before = None                                            # carried across lines (a rise can come between attempts)
    for ln in pick(r"transition memory phases", first):
        ch = re.search(r"kappa (\d)", ln).group(1)
        body = ln.split("GB): ", 1)[1]
        rows = re.findall(r"(.+?) ([0-9.]+) / ([0-9.]+)(?:; |$)", body)
        for name, alloc, peak in rows:
            alloc = float(alloc); peak = float(peak)
            if prev_peak is not None and peak > prev_peak + 1e-9 and before is not None:
                rise = peak - before
                label = name.strip()
                if label == "transition start":                                 # the rise came before this attempt began
                    label = "before this attempt began (the restores between attempts: the old stage's rebuild), against the previous line's last allocation"
                print("  kappa %s, %-95s peak %.2f GB (rose from %.2f); over the allocation before it (%.2f GB): %.2f GB%s" % (
                    ch, label, peak, prev_peak, before, rise, (" = %.2f arrays" % (rise / unit)) if unit else ""))
            prev_peak = peak if prev_peak is None else max(prev_peak, peak)
            before = alloc
    print("the transition's device peak line:", " | ".join(pick(r"\(33\.1 \(6\)\): device peak", first)))


if __name__ == "__main__":
    main()
