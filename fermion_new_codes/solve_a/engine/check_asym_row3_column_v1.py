"""check: in ASYMPTOTICS row (3)'s table (the header line 476), the column FILE 4 item 2 renames, recomputed from the row's sigma^2 and
r_s range as 1 - (9/4) sigma^2/r_s^2 and as -(9/4) sigma^2/r_s^2 at both ends (r_s's ends map to the column's ends in reverse order of
the term's size)."""
import sys
t = open(sys.argv[1]).read().split("\n")
hdr_i = [i for i, ln in enumerate(t) if ln.startswith("| history | sigma^2 | records | r_s = 2gE (range) |")][0]
print("header line %d: %s" % (hdr_i + 1, t[hdr_i][:160]))
for ln in t[hdr_i + 2:]:
    if not ln.startswith("| "):
        break
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    s2 = float(c[1]); r0, r1 = [float(x) for x in c[3].split(" to ")]; v0, v1 = [float(x) for x in c[4].split(" to ")]
    one = sorted([1 - 2.25 * s2 / r0 ** 2, 1 - 2.25 * s2 / r1 ** 2]); neg = sorted([-2.25 * s2 / r0 ** 2, -2.25 * s2 / r1 ** 2])
    print("%s: sigma^2 %s, r_s %s; the column %s; 1 - (9/4) sigma^2/r_s^2 %.6f to %.6f (equal at six decimals: %s); -(9/4) sigma^2/r_s^2 %.6f to %.6f" % (
        c[0], c[1], c[3], c[4], one[0], one[1], "%.6f" % one[0] == "%.6f" % v0 and "%.6f" % one[1] == "%.6f" % v1, neg[0], neg[1]))
