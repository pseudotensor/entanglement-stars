#!/usr/bin/env python3
"""Run the unchanged inherited numerical stage with boolean-safe reporting."""
import argparse
from pathlib import Path

import horizon_exact
from hreduce_horizon_json import write_stage_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('spectrum', 'checks'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    horizon_exact.write_json = write_stage_json
    getattr(horizon_exact, args.stage)(Path(args.out))
