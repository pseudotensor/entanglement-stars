"""Independent primitive-column ledger arithmetic, PAGE specification section 4."""

import argparse
import json
import math
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=18)
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output)
    rows, source_paths = [], []
    progress = (args.output/"ledger_rows.jsonl").open("w")
    for kappa in range(1, args.channels+1):
        path = args.inputs/f"v6o_B2_k{kappa}_d13.json"
        source_paths.append(str(path.resolve()))
        source = json.loads(path.read_text())
        seen = set()
        for original in source["windows"][str(kappa)]:
            a, b = (float(original[key]) for key in ("t0_over_rs", "t1_over_rs"))
            raw_cut = str(original["cut"])
            if raw_cut not in ("16", "30"):
                continue
            cut = float(raw_cut)
            key = (a, b, cut)
            if key in seen:
                raise ValueError(f"Duplicate window in {path}: {key}")
            seen.add(key)
            dc = original["inventory_C_end"]-original["inventory_C_start"]
            dd = original["inventory_D_end"]-original["inventory_D_start"]
            current = original["crossing_current"]
            jumps = original["crossing_jumps"]
            vacuum = original["vacuum_change"]
            row = {"kappa": kappa, "weight": 16*kappa,
                   "clock_start_rs": a, "clock_end_rs": b, "cut_rs": cut,
                   "inventory_D_start": original["inventory_D_start"],
                   "inventory_D_end": original["inventory_D_end"],
                   "inventory_C_start": original["inventory_C_start"],
                   "inventory_C_end": original["inventory_C_end"],
                   "delta_C": dc, "second_term_D": dd,
                   "current": current, "jumps": jumps, "vacuum_change": vacuum,
                   "jumps_minus_vacuum": jumps-vacuum,
                   "telescoping_residual": dc-current-jumps,
                   "balance_residual": dd-current-jumps+vacuum,
                   "saved_second_term_difference": dd-original["second_term_D"],
                   "saved_crossing_sum_difference": current+jumps-original["crossing_total"],
                   "saved_telescoping_difference": dc-current-jumps-original["telescoping_residual"],
                   "current_inventory_relative_difference": abs(current-dd)/abs(dd) if dd else None,
                   "input": str(path.resolve())}
            rows.append(row)
            progress.write(json.dumps(row)+"\n")
            progress.flush()
    progress.close()
    by_key = {(row["kappa"],row["clock_start_rs"],row["clock_end_rs"],row["cut_rs"]):row
              for row in rows}
    aggregates = []
    keys = sorted({key[1:] for key in by_key}, key=lambda item:(item[0],item[1],str(item[2])))
    fields = ("inventory_D_start", "inventory_D_end", "delta_C", "second_term_D",
              "current", "jumps", "vacuum_change", "jumps_minus_vacuum",
              "telescoping_residual", "balance_residual")
    for channels in ((1,), (1,2), (1,2,3), tuple(range(1,args.channels+1))):
        for a,b,cut in keys:
            subset = [by_key.get((k,a,b,cut)) for k in channels]
            if any(row is None for row in subset):
                aggregates.append({"channels": list(channels), "clock_start_rs": a,
                                   "clock_end_rs": b, "cut_rs": cut,
                                   "status": "missing channel; no aggregate"})
                continue
            result = {"channels": list(channels), "clock_start_rs": a,
                      "clock_end_rs": b, "cut_rs": cut, "status": "formed"}
            for field in fields:
                result[field] = math.fsum(row["weight"]*row[field] for row in subset)
            aggregates.append(result)
    shell_rows = []
    for kappa,a,b,cut in by_key:
        if cut != 16 or (kappa,a,b,30) not in by_key:
            continue
        inner, outer = by_key[kappa,a,b,16], by_key[kappa,a,b,30]
        shell_rows.append({"kappa": kappa, "clock_start_rs": a, "clock_end_rs": b,
                           "inventory_D_shell_start": inner["inventory_D_start"]-outer["inventory_D_start"],
                           "inventory_D_shell_end": inner["inventory_D_end"]-outer["inventory_D_end"]})
    output = {"scope": "Independent arithmetic of exported endpoint/current/jump columns; "
                       "endpoint traces and integrated currents are not regenerated.",
              "runtime": runtime, "sources": source_paths,
              "rows": rows, "aggregates": aggregates, "shell_rows": shell_rows}
    (args.output/"ledger.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status": "saved", "rows": len(rows),
                      "output": str(args.output/"ledger.json")}))


if __name__ == "__main__":
    main()
