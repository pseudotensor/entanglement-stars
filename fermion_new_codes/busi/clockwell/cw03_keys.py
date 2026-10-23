"""cw03: list top-level keys (and small scalar values) of chosen record kinds. Read-only."""
import json, sys
p, kinds = sys.argv[1], sys.argv[2].split(',')
done = set()
with open(p) as f:
    for line in f:
        r = json.loads(line); k = r.get('kind')
        if k in kinds and k not in done:
            done.add(k); d = r['data']
            print('=====', k)
            def show(d, pre=''):
                for kk, v in d.items():
                    if isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and len(v) < 150):
                        print(pre + kk, '=', v)
                    elif isinstance(v, dict) and len(pre) < 8:
                        print(pre + kk, ': dict'); show(v, pre + '  ')
                    else:
                        print(pre + kk, ':', type(v).__name__, len(v) if hasattr(v, '__len__') else '')
            show(d) if isinstance(d, dict) else print(str(d)[:2000])
