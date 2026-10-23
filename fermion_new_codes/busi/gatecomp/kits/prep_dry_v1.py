#!/usr/bin/env python
"""prep_dry_v1.py (busi lane, GATECOMP kits, inbox #400): the CPU dry run of an engine launch line, no evolution, no GPU.

Modes (argv: MODE CODE_DIR ENGINE_FILE ARGS_FILE OUT_DIR):
  parse  the engine's own argparse on the line exactly as written; vars(a) to OUT_DIR/dry_args.json; exit 0 before any compute
         (solve's dryparse_v1.py, the same patch).
  check  the line with --gpu removed (CPU) and --out/--slice-dir moved into OUT_DIR: the engine's post-parse checks run (every ap.error of
         main(), iso_configure, backend, rd13_configure, sch_configure), then the run stops at its first statement that builds the
         channels (engine_collapse_v5.Chan is replaced by a stub that raises). Proves the option set is accepted; touches no input file.
  prep   as check, but the run stops at the first engine_stress_v2.ChanStress (right after the preparation of a fresh line, or after the
         version-34 restart's state is rebuilt from its slice): the initial data (or the restart) is computed on the CPU and its printed
         lines are the evidence. For the box's small fixtures only (a 4096-site K = 14 preparation is hours on 8 CPU threads).
Exit: 0 when the mode's stop is reached (PARSED / CHECK OK / PREP OK printed); the engine's own exit code otherwise.
"""
import argparse
import json
import os
import runpy
import sys


class _Stop(BaseException):
    pass


def main():
    mode, code, eng, af, out = sys.argv[1:6]
    assert mode in ("parse", "check", "prep"), mode
    args = open(af).read().split()
    os.makedirs(out, exist_ok=True)
    if mode != "parse":
        x = []
        i = 0
        while i < len(args):
            if args[i] == "--gpu":
                i += 2; continue
            if args[i] in ("--out", "--slice-dir"):
                x += [args[i], os.path.join(out, args[i].lstrip("-").replace("-", "_"))]; i += 2; continue
            x.append(args[i]); i += 1
        args = x
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    orig = argparse.ArgumentParser.parse_args

    def pa(self, a=None, ns=None):
        r = orig(self, a, ns)
        if any(q.dest == "tag" for q in self._actions):
            json.dump(vars(r), open(os.path.join(out, "dry_args_%s.json" % mode), "w"), indent=0, default=str)
            print("PARSED %d args (%s)" % (len(vars(r)), mode), flush=True)
            if mode == "parse":
                sys.exit(0)
        return r
    argparse.ArgumentParser.parse_args = pa
    sys.path.insert(0, code)
    if mode == "check":
        import engine_collapse_v5 as C5

        def stub(*a_, **k_):
            raise _Stop("check")
        C5.Chan = stub
    if mode == "prep":
        import engine_stress_v2 as STR

        def stub2(*a_, **k_):
            raise _Stop("prep")
        STR.ChanStress = stub2
    sys.argv = [os.path.join(code, eng)] + args
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except _Stop as e:
        print({"check": "CHECK OK: the engine accepted the option set (post-parse checks and backend on the CPU); stopped before the channels",
               "prep": "PREP OK: the preparation (or the restart's state) is complete on the CPU; stopped before the record loop (no evolution)"}[str(e)], flush=True)
        sys.exit(0)
    print("the engine returned without reaching the dry stop", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
