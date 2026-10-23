#!/usr/bin/env python
"""run_collapse_v27_float_ell_v1.py (builder analyst; engine spec 36.4 (1): "two short members at sigma/a 3.6 and 4.4 from the stored
preparation on engine_collapse_v27 (the plain iteration, the sigma/a 4 and 8 members' engine), tau 8, to T/rs 17.5"; 36.0: "no new engine
version for the collapse").

engine_collapse_v27 run unchanged: this runner imports it and calls its own main() (its argument list, its defaults, its run()), with one
change at the argument parser: --ell is parsed as a float (v27 declares it type=int, so sigma/a 3.6 and 4.4 cannot be given to it). --prep-ell
stays an integer (the preparation at ell 4, as the stored members prepared it). Nothing in v27's computation needs an integer ell: the
Gaussian kernels K_0, K_1, K_2, the coarse nodes r_i = i ell, the fine grids of spacing ell/16 and ell/64, the collar test R_last + 8 ell and
the averaging back onto the sites all take a float. Two consequences, labelled here and in the json's "runner" entry, not changed:
(1) the coarse nodes are not sites when ell is not an integer (engine_fields_v4.coarse_grid's docstring: "ell integer so that
    r_i = R_{i ell - 1}"); the last node, where the collar sits once the sources reach the chain's end, is floor(nsites/ell) ell, printed at
    the start against the chain's last site;
(2) v27 formats ell with %d in three labels (the run label "ell = %d;", the head's kernel "... evolved at ell %d", the record label
    "evolved at ell %d, ..."), which would print the integer part; this runner rewrites those strings to the float value (%g) in the json
    (head, records), in the fields npz's label and on stdout, and adds the top-level json key "runner". An integer ell is passed through with
    no rewrite (the json then equals v27's but for the "runner" key).
Usage: as engine_collapse_v27.py, e.g. python run_collapse_v27_float_ell_v1.py --ell 3.6 --prep-ell 4 ... --out <dir> --tag <tag> --gpu 0."""
import argparse
import io
import json as _json
import math
import re
import sys

import numpy as np

_ADD = argparse.ArgumentParser.add_argument


def _add_argument_float_ell(self, *names, **kw):
    """the one change: v27's --ell declared type=int is parsed as a float."""
    if names and names[0] == "--ell" and kw.get("type") is int:
        kw["type"] = float
    return _ADD(self, *names, **kw)


def _pre(argv, key, default=None):
    """the value given for key (before v27's parser reads it), as a string, or default."""
    out = default
    for i, x in enumerate(argv):
        if x == key and i + 1 < len(argv):
            out = argv[i + 1]
        elif x.startswith(key + "="):
            out = x.split("=", 1)[1]
    return out


_e = _pre(sys.argv[1:], "--ell")
ELL = float(_e) if _e is not None else None
PREP = _pre(sys.argv[1:], "--prep-ell")
NSITES = int(_pre(sys.argv[1:], "--nsites", "2048"))
SUBS = []
if ELL is not None and ELL != int(ELL):
    _i = int(ELL)
    SUBS = [(re.compile(r"ell = %d;" % _i), "ell = %g;" % ELL),
            (re.compile(r"evolved at ell %d(?![0-9.])" % _i), "evolved at ell %g" % ELL)]
LAST_NODE = (math.floor(NSITES / ELL) * ELL) if ELL else None
RUNNER = ("run_collapse_v27_float_ell_v1.py: engine_collapse_v27.py imported and run unchanged (its main(), arguments and run()), with --ell "
          "parsed as a float (v27 declares type=int)%s; --prep-ell %s as given; the coarse nodes r_i = i ell (sites only for an integer ell): the "
          "last node floor(nsites/ell) ell = %s against the chain's last site %d, the collar's node once the sources reach the chain's end" % (
              (" and v27's three '%%d' labels of ell rewritten to %g" % ELL) if SUBS else "", PREP, ("%.6g" % LAST_NODE) if LAST_NODE is not None else "-", NSITES))


def fix(s):
    if not isinstance(s, str):
        return s
    for rx, rep in SUBS:
        s = rx.sub(rep, s)
    return s


class _JsonShim:
    """v27's module-level json: dump rewrites the ell labels (head, records) and adds "runner"; everything else is json's."""

    def __getattr__(self, k):
        return getattr(_json, k)

    def dump(self, obj, fh, **kw):
        if isinstance(obj, dict) and "records" in obj:
            obj = dict(obj)
            head = dict(obj.get("head") or {})
            for k in ("label", "kernel"):
                if k in head:
                    head[k] = fix(head[k])
            obj["head"] = head
            recs = []
            for r in obj["records"]:
                if isinstance(r, dict) and ("label" in r or "record_label" in r):
                    r = dict(r)
                    for k in ("label", "record_label"):
                        if k in r:
                            r[k] = fix(r[k])
                recs.append(r)
            obj["records"] = recs
            obj["runner"] = RUNNER
        return _json.dump(obj, fh, **kw)


_SAVEZ_C = np.savez_compressed


def _savez_compressed_fixed(file, *args, **kw):
    if isinstance(kw.get("label"), str):
        kw["label"] = fix(kw["label"])
    return _SAVEZ_C(file, *args, **kw)


class _Out(io.TextIOBase):
    """stdout with the ell labels rewritten (v27 prints its run label and every record label)."""

    def __init__(self, s):
        self.s = s

    def write(self, x):
        return self.s.write(fix(x))

    def flush(self):
        return self.s.flush()

    def fileno(self):
        return self.s.fileno()

    def isatty(self):
        return self.s.isatty()

    @property
    def encoding(self):
        return self.s.encoding


def main():
    argparse.ArgumentParser.add_argument = _add_argument_float_ell
    import engine_collapse_v27 as E
    E.json = _JsonShim()
    if SUBS:
        np.savez_compressed = _savez_compressed_fixed
        sys.stdout = _Out(sys.stdout)
    print("runner: %s" % RUNNER, flush=True)
    E.main()


if __name__ == "__main__":
    main()
