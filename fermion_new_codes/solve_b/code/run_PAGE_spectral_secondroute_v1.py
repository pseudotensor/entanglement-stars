"""Independent entropy contractions of primitive saved Page eigenspectra.

This consumes eigenvalues, never another lane's entropy readout or solver.
It does not independently generate the spectra from occupied columns.
Formula and route specification: solve/PAGE_secondroute_spec_v1.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpu_runtime import require_remote_cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=18)
    parser.add_argument("--channel-list", help="Comma-separated labels; default 1 through --channels")
    parser.add_argument("--cut-list", default="2,16,24,30,48")
    parser.add_argument("--filename-template", default="v6o_B2_k{k}_d13.npz")
    parser.add_argument("--full-chain",action="store_true",help="Unreduced full-chain exported spectra")
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output)
    import numpy as np

    extrema = [1.0, 0.0]
    clipped = 0
    context = ""

    def entropy(values):
        nonlocal clipped
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite covariance eigenvalue")
        extrema[0] = min(extrema[0], float(values.min(initial=1.0)))
        extrema[1] = max(extrema[1], float(values.max(initial=0.0)))
        bounded = np.clip(values, 0.0, 1.0)
        delta = float(np.sum(np.abs(values - bounded)))
        count = int(((values < 0) | (values > 1)).sum())
        invalid = bool((values < -1e-8).any() or (values > 1 + 1e-8).any())
        diagnostics = {"input_key": context, "length": len(values),
                       "available":True,
                       "minimum": float(values.min(initial=1.0)),
                       "maximum": float(values.max(initial=0.0)),
                       "outside_count": count, "clipping_displacement": delta,
                       "valid_within_tolerance": not invalid}
        if invalid:
            return None, diagnostics
        clipped += count
        values = bounded
        interior = values[(values > 0) & (values < 1)]
        result = float(np.sum(-interior * np.log(interior)
                              - (1 - interior) * np.log1p(-interior)))
        ratio = min(delta / max(1, len(values)), 0.5)
        diagnostics["clipping_sensitivity"] = (0.0 if ratio == 0 else
            len(values) * (-ratio * np.log(ratio) - (1 - ratio) * np.log1p(-ratio)))
        return result, diagnostics

    def combine(function, *values):
        return None if any(value is None for value in values) else function(*values)

    rows = []
    progress = (args.output / "spectral_entropy_rows.jsonl").open("w")
    channels = [int(k) for k in args.channel_list.split(",")] if args.channel_list else list(range(1,args.channels+1))
    cuts = [int(cut) for cut in args.cut_list.split(",")]
    if len(channels)!=len(set(channels)) or not channels or not cuts:
        raise ValueError("Distinct nonempty channel and cut lists required")
    for kappa in channels:
        path = args.inputs / args.filename_template.format(k=kappa)
        with np.load(path, allow_pickle=False) as archive:
            suffix = f"_k{kappa}_eig_X_cut{cuts[0]}" if args.full_chain else f"_k{kappa}_eig_Xc_plus_cut{cuts[0]}"
            clocks = sorted(float(key[1:-len(suffix)])
                            for key in archive.files if key.endswith(suffix))
            for clock in clocks:
                prefix = f"T{clock:010.4f}_k{kappa}_"
                for cut in cuts:
                    def read(key):
                        nonlocal context
                        context = str(path) + ":" + prefix + key + f"_cut{cut}"
                        return archive[prefix + key + f"_cut{cut}"]

                    diagnostics = {}

                    def evaluate(key):
                        full_key = prefix + key + f"_cut{cut}"
                        if full_key not in archive:
                            diagnostics[key] = {"input_key":str(path)+":"+full_key,
                                                "available":False,"valid_within_tolerance":False,
                                                "reason":"saved spectrum missing"}
                            return None
                        value, diagnostic = entropy(read(key))
                        diagnostics[key] = diagnostic
                        return value

                    cp = evaluate("eig_Xc" if args.full_chain else "eig_Xc_plus")
                    pp = evaluate("eig_Xc_vac" if args.full_chain else "eig_Xc_vac_plus")
                    factor = 1 if args.full_chain else 2
                    row = {"kappa": kappa, "weight": 16 * kappa,
                           "clock_rs": clock, "cut_rs": cut,
                           "complement_state": combine(lambda x: factor*x, cp),
                           "complement_vacuum": combine(lambda x: factor*x, pp),
                           "complement_excess": combine(lambda x,y: factor*(x-y), cp, pp),
                           "sector_convention":"full chain" if args.full_chain else "two paired sectors",
                           "input": str(path.resolve()), "spectra": diagnostics}
                    if args.full_chain:
                        sx = evaluate("eig_X")
                        sp = evaluate("eig_X_vac")
                        row.update({"direct_state":sx,"direct_vacuum":sp,
                                    "direct_excess":combine(lambda x,y:x-y,sx,sp),
                                    "state_complement_gap":combine(lambda x,y:x-y,sx,cp),
                                    "vacuum_complement_gap":combine(lambda x,y:x-y,sp,pp)})
                    elif prefix + f"eig_X_minus_cut{cut}" in archive:
                        cm = evaluate("eig_Xc_minus")
                        pm = evaluate("eig_Xc_vac_minus")
                        sx = combine(lambda x,y:x+y, evaluate("eig_X_plus"),
                                     evaluate("eig_X_minus"))
                        sp = combine(lambda x,y:x+y, pp, pm)
                        row.update({"direct_state": sx, "direct_vacuum": sp,
                                    "direct_excess": combine(lambda x,y:x-y, sx, sp),
                                    "state_complement_gap": combine(lambda x,y,z:x-y-z,sx,cp,cm),
                                    "state_sector_gap": combine(lambda x,y:x-y, cp, cm),
                                    "vacuum_sector_gap": combine(lambda x,y:x-y, pp, pm)})
                    row["all_saved_spectra_valid"] = all(
                        diagnostic["valid_within_tolerance"] for diagnostic in diagnostics.values())
                    rows.append(row)
                    progress.write(json.dumps(row) + "\n")
                    progress.flush()
    progress.close()
    checkpoints = {}
    invalid_spectra = []
    for row in rows:
        key = (row["kappa"],row["clock_rs"])
        checkpoint = checkpoints.setdefault(key, {"kappa":key[0],"clock_rs":key[1],
            "state_spectra_valid":True,"vacuum_spectra_valid":True,"invalid_keys":[],"missing_keys":[]})
        for name, diagnostic in row["spectra"].items():
            if not diagnostic["available"]:
                checkpoint["missing_keys"].append(diagnostic["input_key"])
                continue
            if not diagnostic["valid_within_tolerance"]:
                checkpoint["invalid_keys"].append(diagnostic["input_key"])
                checkpoint["vacuum_spectra_valid" if "vac" in name else "state_spectra_valid"] = False
                invalid_spectra.append(diagnostic)
    for row in rows:
        checkpoint = checkpoints[row["kappa"],row["clock_rs"]]
        row["checkpoint_state_spectra_valid"] = checkpoint["state_spectra_valid"]
        row["checkpoint_vacuum_spectra_valid"] = checkpoint["vacuum_spectra_valid"]
        row["checkpoint_saved_spectra_complete"] = not checkpoint["missing_keys"]
    (args.output/"invalid_spectra.json").write_text(json.dumps(
        {"invalid_spectra":invalid_spectra,"checkpoints":list(checkpoints.values())},indent=2)+"\n")
    output = {"scope": "Independent entropy arithmetic on saved eigenspectra; "
                        "covariance contraction and eigenspectrum generation are not repeated.",
              "runtime": runtime, "eigenvalue_extrema": extrema,
              "eigenvalues_clipped_within_tolerance": clipped,
              "rows": rows, "checkpoint_validation":list(checkpoints.values())}
    target = args.output / "spectral_entropy.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(target), "rows": len(rows),
                      "channels": len(channels), "status": "saved"}))


if __name__ == "__main__":
    main()
