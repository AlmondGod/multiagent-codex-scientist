#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from commsci.codex_scientist.paper import write_domain_paper_from_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a domain paper from a Codex-Scientist TinyWorlds run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", default="world_model_paper.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = write_domain_paper_from_run(run_dir, run_dir / args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
