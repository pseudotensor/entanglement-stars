"""pace_capture_v6_min_v1.py (builder analyst): when the version 6 minimum run on pod 1 lane 3 is expected to end, from the pace of the
version 6 last-record run (same code, same arguments but the slice, same lane): the minimum run's elapsed log seconds at its latest
(background, probe, |kappa|) line against the last-record run's at the same line, scaled to the last-record run's length (status-file start
and exit). A projection from the pace, printed as such."""
import re
import sys
from datetime import datetime, timedelta

def parse(path):
    out = []
    for line in open(path):
        m = re.match(r"^(.{12}) \| (.*?) omega ([0-9.]+): \|kappa\| (\d+): .*\((\d+) s\)\s*$", line)
        if m:
            out.append(((m.group(1), m.group(2), int(m.group(4))), float(m.group(5))))
    return out

last_log, min_log, status = sys.argv[1:4]
L = dict(parse(last_log)); M = parse(min_log)
st = open(status).read()
def hms(tag, what):
    m = re.search(r"lane3 %s %s(?: 0)? (\d\d:\d\d:\d\d) UTC" % (re.escape(tag), what), st)
    return datetime.strptime("2026-10-16 " + m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
s_last, e_last, s_min = hms("ccap6_prod19_last_l3", "start"), hms("ccap6_prod19_last_l3", "exit"), hms("ccap6_prod19_min", "start")
total_last = (e_last - s_last).total_seconds()
key, t_min = M[-1]
t_last = L.get(key)
print("the last-record run: start %s, exit %s UTC, %.0f s; its log's last line at %.0f s" % (s_last.strftime("%H:%M:%S"), e_last.strftime("%H:%M:%S"), total_last, max(L.values())))
print("the minimum run: start %s UTC; its latest line %s | %s |kappa| %d at %.0f s; the last-record run reached that line at %s s"
      % (s_min.strftime("%H:%M:%S"), key[0], key[1], key[2], t_min, ("%.0f" % t_last) if t_last is not None else "-"))
if t_last:
    pace = t_min / t_last
    end = s_min + timedelta(seconds=total_last * pace)
    print("pace %.4f of the last-record run; projected end of the minimum run %s UTC (a projection from the pace, not a measurement)" % (pace, end.strftime("%H:%M")))
