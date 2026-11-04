#!/usr/bin/env python3
"""Read saved X4 JSON only; emit small tables/summary and the raw-file list.

No model equation, quadrature, root finder or solver is called here.
"""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    try:
        return format(Decimal(str(value)), ".12g")
    except InvalidOperation:
        return str(value)


def table(rows, columns):
    return ("| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns)
            + " |\n" + "".join("| " + " | ".join(display(row.get(key)) for key in columns)
                                     + " |\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path("<internal>/solve_b_bulk").resolve()
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    if not run_dir.is_relative_to(root) or not output.is_relative_to(root):
        parser.error("saved-data readout stays under the lane's box directory")
    if run_dir == output:
        parser.error("readout directory must differ from the raw run directory")
    raw_path = run_dir / "scalar.json"
    receipt_path = run_dir / "execution.json"
    raw = json.loads(raw_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    raw_hash = digest(raw_path)
    if receipt.get("result", {}).get("sha256") != raw_hash:
        parser.error("raw result differs from the execution receipt")
    bound = raw.get("bound_precision_results", [])
    high = bound[-1] if bound else {}
    keep = ("decimal_working_precision", "onsets", "endpoints", "segments", "analytic_limits", "consistency",
            "function_evaluation_counts", "identity_rows", "quadrature_function_evaluations")
    summary = {key: raw.get(key) for key in
               ("schema", "status", "case", "code", "bound_module", "runtime",
                "metadata_preflight", "production_lineage_verified", "profile_stage",
                "point_scalar_status", "full_X4_task_complete", "numerical_error_certificate",
                "precision_results", "precision_comparison", "release_crossing_precision_comparison",
                "inherited_marginal_comparison", "error_type", "error")}
    comparison = raw.get("bound_precision_comparison", {})
    summary["bound_high_precision"] = {key: high.get(key) for key in keep}
    summary["bound_cross_precision"] = {key: value for key, value in comparison.items()
                                        if key != "differences"}
    summary["raw_result"] = {"box_path": str(raw_path), "bytes": raw_path.stat().st_size,
                             "sha256": raw_hash, "raw_tables_remain_on_box": True}
    summary["formatter"] = {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__)),
                            "saved_data_only": True}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    shutil.copyfile(receipt_path, output / "execution.json")
    text = ["# X4 saved scalar tables\n", "status: "+str(raw.get("status"))+"\n",
            "full_X4_task_complete: "+str(raw.get("full_X4_task_complete"))+"\n",
            "profile_stage: "+str(raw.get("profile_stage", {}).get("status"))+"\n",
            "All displayed entries are read from saved producer fields.\n"]
    point = raw.get("precision_results", [])
    if point:
        rows = [{"clock": key, **value} for key, value in point[-1]["clocks"].items()]
        text += ["\nPoint/sine clocks (supplied metadata case)\n", table(rows, ["clock", "over_a", "over_r0", "over_rg"])]
    marginal = raw.get("inherited_marginal_comparison", {})
    inherited = {Decimal(row["criterion"]): row["constant"] for row in marginal.get("onsets", [])}
    rows = [{**row, "bound_constant": row.get("constant"),
             "inherited_marginal_constant": inherited.get(Decimal(row["criterion"]))}
            for row in high.get("onsets", [])]
    text += ["\nBound-rest onsets; marginal constants are inherited comparison columns\n",
             table(rows, ["criterion", "x", "gap", "lapse", "bound_constant", "inherited_marginal_constant"]),
             "\nBound-rest endpoints\n",
             table(high.get("endpoints", []), ["frequency", "convention", "q", "b", "x", "K"]),
             "\nBound-rest segments\n",
             table(high.get("segments", []), ["frequency", "convention", "criterion", "status",
                                                "D", "N_H", "N_peel", "mode_cycles"]),
             "\nRaw tables and quadrature nodes: "+str(raw_path)+"\n",
             "Raw SHA256: "+raw_hash+"\n"]
    (output / "scalar.md").write_text("\n".join(text))
    omitted = [raw_path]
    partial = raw_path.with_name(raw_path.name+".partial")
    if partial.exists():
        omitted.append(partial)
    lines = ["path\tsize_bytes\tsha256\tbox_location\n"]
    for path in omitted:
        lines.append(f"{path}\t{path.stat().st_size}\t{digest(path)}\tcompute-host:{path}\n")
    (output / "LARGE_FILES.tsv").write_text("".join(lines))
    print(json.dumps({"summary": str(output / "summary.json"), "saved_data_only": True}))


if __name__ == "__main__":
    main()
