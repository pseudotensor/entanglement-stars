#!/usr/bin/env python3
"""s46v31_coreprobe_v1.py: structure of the CORE_HF route-1 jsons (read-only probe)."""
import json, glob, os
def walk(x, p='', d=0, maxd=4):
    if d > maxd:
        return
    if isinstance(x, dict):
        for k, v in list(x.items())[:40]:
            if isinstance(v, (dict, list)):
                print('  ' * d + p + str(k), type(v).__name__, len(v))
                walk(v, '', d + 1, maxd)
            else:
                print('  ' * d + str(k), repr(v)[:100])
    elif isinstance(x, list) and x:
        walk(x[0], '[0].', d + 1, maxd)
for f in ['<internal>/core_hf_full_v1.json']:
    print('=====', f, os.path.getsize(f))
    walk(json.load(open(f)), maxd=3)
print(sorted(glob.glob('<internal>/*'))[:60])
