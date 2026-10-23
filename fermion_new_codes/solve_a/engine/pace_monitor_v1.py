"""pace_monitor_v1.py (the lane's analyst builder; engine spec 43.1 (4)): every 4 r_s of a running member's records, the running mean wall
per record (the records' "seconds") and the projected H100-hours to T/r_s 40; a mean above the threshold flagged OVER. Usage: JSON RS EVERY_RS THRESHOLD_S"""
import json, sys, time
p, rs, every, thr = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
done = 0.0
while True:
    try:
        r = [x for x in json.load(open(p))["records"] if x.get("E") is not None and x.get("seconds") is not None]
    except Exception:
        time.sleep(120); continue
    if r and r[-1]["T"] / rs >= done + every:
        done = r[-1]["T"] / rs
        rec = [x for x in r if x["T"] > 0]
        mean = sum(x["seconds"] for x in rec) / max(len(rec), 1)
        print("%s T/r_s %.3f: %d records, mean wall %.1f s per record, about %.1f H100-hours for 40 r_s at this mean%s" % (
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), done, len(rec), mean, mean * 640 / 3600.0, "; OVER %g s" % thr if mean > thr else ""), flush=True)
    if r and r[-1]["T"] / rs >= 40 - 1e-9:
        break
    time.sleep(300)
