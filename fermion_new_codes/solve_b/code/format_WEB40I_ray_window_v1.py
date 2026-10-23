#!/usr/bin/env python3
"""Format existing X1 JSON fields only; no solver, refit or new quadrature."""
from pathlib import Path
import argparse
from decimal import Decimal
import json


def number(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x).lower()
    try:
        return format(Decimal(str(x)), ".12g")
    except Exception:
        return str(x)


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(number(x) for x in row) + " |" for row in rows)
    lines.append("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--execution", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior formatted output")
    d = json.loads(args.input.read_text())
    execution = json.loads(args.execution.read_text())
    s, c = d["scalar_80"], d["content"]
    lines = ["# X1 scalar and inherited-row readback", "",
        "Stage-one output only. This formatter loads saved JSON fields and does not solve, fit, interpolate or integrate. "
        "The inherited finite-domain stationary-rate comparator is not an actual finite-time radiation entropy; "
        "the new infrared interval and angular tail remain unresolved.", "",
        f"Producer: `{d['source_sha256']}`. Input: `{d['inherited']['sha256']}`.",
        f"Actual outer exit: {execution['actual_outer_returncode']}; started {execution['started_utc']}; finished {execution['finished_utc']}.",
        f"Outer wall: {number(execution['outer_wall_seconds'])} s. Producer resources: `{json.dumps(d['resources'], sort_keys=True)}`.",
        "The producer measures its phase after mpmath import; the outer receipt includes interpreter/import time. "
        "The registered ceilings are 60 s phase wall, 512 MiB peak RSS and 768 MiB address space. New ODE solves: 0.", "",
        "## Onsets and constants", ""]
    table(lines, ["criterion", "y_c", "gap", "lapse", "C_c", "2 ln gap", "u_c-u(2) over rs"],
          [[o[k] for k in ("criterion", "y", "gap", "lapse", "constant", "two_log_gap", "u_from_y2_over_rs")] for o in s["onsets"]])
    lines += ["## One-e-fold thresholds at q=1, criterion 0.95", ""]
    table(lines, ["definition", "y_star", "size gap", "size lapse", "leading-log size"],
          [[o[k] for k in ("efold_definition", "y", "size_gap", "size_lapse", "leading_log_size")] for o in s["thresholds"]])
    lines += ["## Main segment rows", "", "Both accepted e-fold counts are zero when empty. "
              "The raw JSON also retains the exact support endpoints, root brackets/residuals, "
              "clock integrals, scalar precision changes and exact-symbol diagnostics.", ""]
    table(lines, ["criterion", "endpoint", "size", "q", "D", "N_H", "N_peel", "delta_T/rs", "empty"],
          [[o[k] for k in ("criterion", "convention", "size", "q", "D", "N_H", "N_peel", "delta_T_over_rs", "empty")]
           for o in s["segments"] if not o["is_support_endpoint"]])
    lines += ["## Period maxima", "", "These are scalar qD diagnostics, not resolved-packet certificates.", ""]
    table(lines, ["criterion", "endpoint", "size", "q", "D", "qD", "cycles"],
          [[o[k] for k in ("criterion", "convention", "size", "q", "D", "qD", "cycles")] for o in s["period_maxima"]])
    lines += ["## Inherited finite-domain content at size 64", "",
              "S, N and rs E use the exact positive-part duration inside each saved-frequency sum. "
              "The 64/96 comparison changes quadrature and endpoint/tolerance settings together; "
              "it is an empirical indicator, not an absolute error enclosure.", ""]
    rows = []
    for rb in c["readbacks"]:
        for a in rb["aggregates"]:
            for cum in a["cumulative"]:
                rows.append([rb["settings"]["name"], a["criterion"], a["convention"], cum["k_cap"], *cum["S_N_rsE"]])
    table(lines, ["readback", "criterion", "endpoint", "k cap", "S", "N", "rs E"], rows)
    lines += ["## Observed refined-minus-first changes", ""]
    table(lines, ["criterion", "endpoint", "delta S", "delta N", "delta rs E", "support refinement trigger"],
          [[o["criterion"], o["convention"], *o["refined_minus_first_S_N_rsE"], o["support_refinement_requested"]] for o in c["comparisons"]])
    lines += ["## Conditional finite-k<=8 frequency-tail allowances", "",
              "The bounds require exact 0<=Gamma<=1. They do not certify the computed transmissions "
              "or the omitted angular channels. The target bound below 1e-8 does not replace the "
              "missing integral from 1e-8 to 1e-3.", ""]
    rows = []
    for o in c["frequency_tail_allowances"]:
        for key, label in [("inherited_omitted_IR_S_N_rsE_bound", "below 1e-3"),
                           ("target_omitted_IR_S_N_rsE_bound", "below 1e-8"),
                           ("UV_above_3_S_N_rsE_bound", "above 3")]:
            rows.append([o["criterion"], o["convention"], label, *o[key]])
    table(lines, ["criterion", "endpoint", "interval", "S allowance", "N allowance", "rs E allowance"], rows)
    lines += ["## Unresolved", ""] + ["- " + x for x in c["unresolved"]]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
