"""Command-line interface for tearsheet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tearsheet",
        description="Generate a trade tear sheet from a Sierra Chart TradeActivityLog file.",
    )
    p.add_argument("--input", required=True, metavar="FILE",
                   help="Path to TradeActivityLog_*.txt")
    p.add_argument("--output", default="report.html", metavar="FILE",
                   help="Output HTML path (default: report.html)")
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    from tearsheet.app.main import run
    run(input_path, args.output)


if __name__ == "__main__":
    main()
