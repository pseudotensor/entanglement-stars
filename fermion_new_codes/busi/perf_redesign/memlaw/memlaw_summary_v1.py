#!/usr/bin/env python
"""memlaw_summary_v1.py (busi, inbox #384): the tables of the memlaw notes from the probes' and the per-call runs' jsons (box only).
For every probe: the run's device peak (torch max_memory_allocated) in GiB and in units of nsites^2 bytes, the label it occurred under,
the ten largest phase peaks, the census at each mark (unique CUDA storages by class, in nsites^2 bytes), the host RSS high-water mark.
For every per-call size: each phase's peak and kept bytes in nsites^2 units. Then the law's coefficients read off the censuses and the
projection to 8192 sites.   python memlaw_summary_v1.py ROOT > summary.txt"""
import glob
import json
import os
import sys

root = sys.argv[1]
G = 2.0 ** 30


def gib(x):
    return x / G


for pj in sorted(glob.glob(os.path.join(root, "runs", "*", "probe.json"))):
    d = json.load(open(pj))
    ns = d["nsites"]; n2 = float(ns) ** 2
    name = os.path.basename(os.path.dirname(pj))
    print("=" * 100)
    print("%s: %s, %d sites; stop: %s after %.0f s; device %s" % (name, os.path.basename(d["engine"]), ns, d["why"], d["wall_s"], d.get("device_name")))
    print("  run peak allocated %.2f GiB = %.1f n^2 B under %s (t %.0f s); reserved peak %.2f GiB; host RSS high-water %.1f GB" % (
        gib(d["global_peak_allocated"]), d["global_peak_allocated"] / n2, d["global_peak_label"], d["global_peak_t"] or -1,
        gib(d.get("reserved_peak") or 0), d["host_rss_hwm_kB"] / 1e6))
    segs = sorted(d["segments"].items(), key=lambda x: -x[1]["peak"])
    print("  phase peaks (label: peak, n^2 units; segments):")
    for k, s in segs[:12]:
        print("    %-80s %7.2f GiB %7.1f  (%d)" % (k.replace("run/newton_record/nk_stage/", "..")[-80:], gib(s["peak"]), s["peak"] / n2, s["n"]))
    print("  census (unique CUDA storages; n^2 units):")
    for c in d["census"]:
        gs = ", ".join("%s %d x = %.0f" % (g.split(" (")[0], v["n"], v["bytes"] / n2) for g, v in sorted(c["groups"].items(), key=lambda x: -x[1]["bytes"]) if v["bytes"] / n2 >= 1)
        print("    %-22s alloc %7.1f  [%s]" % (c["tag"], c["allocated"] / n2, gs))
    if d.get("rd13_stats"):
        s = d["rd13_stats"]
        print("  rd13: calls %s banded %s (all %s) dense fallback %s" % (s.get("calls"), s.get("banded"), s.get("banded_all"), s.get("dense_fallback")))
for pj in sorted(glob.glob(os.path.join(root, "percall", "*", "memlaw_percall_v1.json"))):
    d = json.load(open(pj))
    print("=" * 100)
    print("per call (%s): %s" % (os.path.basename(os.path.dirname(pj)), d["device"]))
    for ns, rec in d["sizes"].items():
        n2 = float(ns) ** 2
        print("  %s sites: " % ns + "; ".join("%s %.2f GiB (%.0f; kept %.0f)" % (k, gib(r["peak"]), r["peak"] / n2, r["kept"] / n2) for k, r in rec.items() if "peak" in r))
        print("    host<->device one U: " + ", ".join("%s %.1f GB/s" % (k, v) for k, v in rec["xfer"]["GBps"].items()))
