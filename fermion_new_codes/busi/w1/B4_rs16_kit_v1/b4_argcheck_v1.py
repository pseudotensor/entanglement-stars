"""b4_argcheck_v1.py (busi f52b1152, 20 Oct): the engine's own argparse on a B4 argument line. engine_collapse_v19.main() is called with
run() and the backend replaced by a capture, so nothing is computed; the parsed vars are compared key by key with B2's record args
(417a4018) and the expected substitutions (b4_args_v1.b4_args).
Usage (PYTHONPATH = the kit's engine/): python -s b4_argcheck_v1.py <B2 json> <mode> <ARGS file>"""
import json, os, shlex, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_args_v1 as BA
import engine_collapse_v19 as E

b2 = json.load(open(sys.argv[1]))["args"]; mode = sys.argv[2]; line = open(sys.argv[3]).read().strip()
cap = {}
E.run = lambda a: cap.update(vars(a))
E.C5.backend = lambda gpu: None
sys.argv = ["engine_collapse_v19.py"] + shlex.split(line)
E.main()
want = BA.b4_args(b2, mode)
bad = [(k, cap.get(k), want.get(k)) for k in sorted(set(cap) | set(want)) if cap.get(k) != want.get(k)]
diff_b2 = [(k, b2.get(k), cap.get(k)) for k in sorted(set(cap) | set(b2)) if cap.get(k) != b2.get(k)]
print("engine file:", E.__file__)
print("%s: parsed %d keys; against the expected list: %s" % (mode, len(cap), "IDENTICAL" if not bad else "DIFFER %s" % bad))
print("%s: parsed against B2's record args, the changed keys:" % mode)
for k, x, y in diff_b2:
    print("   %-18s %s -> %s" % (k, json.dumps(x), json.dumps(y)))
sys.exit(1 if bad else 0)
