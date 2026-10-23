#!/usr/bin/env python3
"""Render the saved small own summary. No scientific imports or solver calls."""
import argparse
import hashlib
import json
from pathlib import Path


def render(value):
    if value is None:
        return "null"
    if isinstance(value, (str, int)):
        return str(value)
    return f"{value:.17g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.summary.read_bytes()
    result = json.loads(data)
    lines = ["# Computation C: own fixed-fixture readout", "",
        "Saved-data rendering; the original N/A output review and foreign comparison remain separate.", "",
        f"Status: {result['status']}; complete: {result['complete']}.",
        f"Own summary SHA256: {hashlib.sha256(data).hexdigest()}.",
        f"Wall through final summary: {result.get('elapsed_seconds')} seconds; peak RSS: {result.get('peak_RSS_KiB')} KiB.",
        "", "| Channel | k | t | S dense | S retained | S complement | S vacuum | Increment | Z / own floor | C0 Z |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, information in result["channels"].items():
        for row in information["checkpoints"]:
            values = (name, row["k"], row["t"], row["S_dense"], row["S_retained"],
                row["S_complement"], row["S_vac"], row["increment"], row["purity_floor_ratio"], row["projected_Z"])
            lines.append("| " + " | ".join(render(v) for v in values) + " |")
    lines += ["", "| Channel | k | t | E inv | Phi E 80 | Phi N 80 | Phi E 160 | Phi N 160 | Trapping |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, information in result["channels"].items():
        for row in information["rows"]:
            values = (name, row["k"], row["t"], row["E_inv"], row["collars"]["80"]["Phi_E"],
                row["collars"]["80"]["Phi_N"], row["collars"]["160"]["Phi_E"],
                row["collars"]["160"]["Phi_N"], str(row["strict_trapping"]))
            lines.append("| " + " | ".join(render(v) for v in values) + " |")
    lines += ["", "All unrounded channel controls and weighted combinations remain in the saved summary.",
              "Raw states, spectra, reference rotations and checkpoint matrices remain on the box at the LARGE_FILES.tsv locations.",
              "This formatter makes no physical-history, radiation-supply, escape or numerical-error certification."]
    args.output.write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
