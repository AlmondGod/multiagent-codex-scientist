#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from commsci.evaluation import aggregate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate peer/self critique ablation artifacts.")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    result = aggregate_run(Path(args.run_dir).expanduser().resolve())
    print(f"Aggregated {len(result['rows'])} agent rows from {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
