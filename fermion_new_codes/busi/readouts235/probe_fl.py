# Probe: FIRSTLAW v3b/v4b json structure (top keys, first record/pair keys and values)
import sys, json
for p in sys.argv[1:]:
    j = json.load(open(p)); print("==", p, type(j).__name__)
    if isinstance(j, dict):
        for k, v in j.items():
            ln = len(v) if isinstance(v, (list, dict)) else ""
            print(" ", k, type(v).__name__, ln, (repr(v)[:200] if not isinstance(v, (list, dict)) else ""))
            if isinstance(v, list) and v and isinstance(v[0], dict): print("    [0]", json.dumps(v[0])[:2500])
            if isinstance(v, dict): print("    keys", list(v.keys())[:40])
