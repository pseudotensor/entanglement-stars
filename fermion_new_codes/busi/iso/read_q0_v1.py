#!/usr/bin/env python
"""read_q0_v1.py (busi; ISO_128_nk_route_v2 sec. 4.1): Q0's read. The resume line, the "v36c round"/"direct round" lines, the fallback lines and the
record lines ("T/rs ... newton" and "T/rs ... E/E0") of the Q0 log against the v36r log (e4806d99) through its 11.938 record line, with the wall
clocks and the version label masked: (1) the per-record timing parenthetical "(N s; eigh ... s)", (2) " in N s", (3) every version token v36<letters>
(v36r/v36q/v36nq/v36m ...) and the v32 label's "exceeds 0.002" / "exceeds 0.001" (the ruled --end-above change; v36r ran END 0.002). The release's
own "  v36m offload:" lines are new by design: printed, not compared. PASS: the masked line lists are equal, and the Q0 log holds no line for
T/rs 12.000 and no round line after its 11.938 record. Any other difference prints and is a FAIL (the route stops at the builder).
Usage: python read_q0_v1.py V36R_LOG Q0_LOG"""
import re, sys
KEEP = re.compile(r"^(version 34 \(a\): restarted|  v36[a-z]+ (round|direct|fallback)|T/rs [0-9.]+ (newton|E/E0))")
MASKS = [(re.compile(r"\(\d+ s; [^)]*\)"), "(WALL)"), (re.compile(r" in \d+(\.\d+)? s\b"), " in W s"), (re.compile(r"\bv36[a-z]+\b"), "v36*"),
         (re.compile(r"exceeds 0\.00[12]\b"), "exceeds END")]
def lines(p, stop_after=None):
    out, extra, after = [], [], False
    for ln in open(p, errors="replace"):
        ln = ln.rstrip("\n")
        if "v36m offload" in ln:
            extra.append(ln); continue
        if not KEEP.match(ln):
            continue
        if after:
            extra.append("AFTER-END " + ln); continue
        m = ln
        for rx, rep in MASKS:
            m = rx.sub(rep, m)
        out.append(m)
        if stop_after and ln.startswith(stop_after):
            after = True
    return out, extra
A, _ = lines(sys.argv[1], "T/rs 11.938 E/E0")
B, extra = lines(sys.argv[2], "T/rs 11.938 E/E0")
bad = 0
for ln in extra:
    print("Q0 extra:", ln[:300])
    if ln.startswith("AFTER-END"):
        bad += 1
if any("T/rs 12.000" in x for x in B) or any("T/rs 12.000" in x for x in extra):
    print("FAIL: Q0 holds a T/rs 12.000 line"); bad += 1
n = max(len(A), len(B))
for i in range(n):
    a = A[i] if i < len(A) else "<none>"; b = B[i] if i < len(B) else "<none>"
    if a != b:
        bad += 1
        j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        print("DIFF line %d at char %d\n  v36r: ...%s\n  Q0  : ...%s" % (i + 1, j, a[max(0, j - 80): j + 160], b[max(0, j - 80): j + 160]))
print("compared %d v36r lines (through its 11.938 record) against %d Q0 lines; records v36r %d, Q0 %d" % (len(A), len(B), sum(x.startswith("T/rs") and " E/E0 " in x for x in A), sum(x.startswith("T/rs") and " E/E0 " in x for x in B)))
print("Q0 READ: %s (%d differences)" % ("PASS" if bad == 0 and len(A) == len(B) else "FAIL", bad))
sys.exit(0 if bad == 0 and len(A) == len(B) else 1)
