#!/usr/bin/env python3
"""
Post-hoc CDI analyzer.

Reads a calibration CSV produced by `scripts/calibrate_critic.py` and emits a
Markdown table + a LaTeX table + a JSON summary suitable for the paper. Run
this whenever you re-export a CSV, without re-spending OpenRouter credits.

Usage:
    python scripts/compute_cdi.py output/_critic_validation/math_gpt-4o-2024-11-20.csv
    python scripts/compute_cdi.py output/_critic_validation/math_ensemble_*.csv --latex paper/cdi.tex

The script intentionally takes only a CSV in and emits files out — no LLM
calls, no network, no surprises.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cdi import (
    compute_cdi,
    format_cdi_markdown,
    format_cdi_latex,
    cdi_to_jsonable,
    DIMENSIONS,
)


# Columns in the input CSV that must be coerced from string → float|bool|None.
_FLOAT_COLS = set(DIMENSIONS) | {"overall", "figure_relevance"}
_BOOL_COLS = {"pass_fail", "critic_matches_gt", "ensemble_unanimous"}
_INT_COLS = {"idx"}


def _coerce(val: str, col: str):
    if val == "" or val is None:
        return None
    if col in _INT_COLS:
        try:
            return int(val)
        except ValueError:
            return None
    if col in _FLOAT_COLS:
        try:
            return float(val)
        except ValueError:
            return None
    if col in _BOOL_COLS:
        return val.strip().lower() in ("true", "1", "yes")
    return val


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({k: _coerce(v, k) for k, v in r.items()})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=Path, help="Calibration CSV from calibrate_critic.py")
    p.add_argument("--markdown", type=Path, default=None, help="Write Markdown table here")
    p.add_argument("--latex", type=Path, default=None, help="Write LaTeX table here")
    p.add_argument("--json", type=Path, default=None, help="Write CDI JSON here")
    p.add_argument(
        "--caption",
        default="Critic Discrimination Index across degraded variants and dimensions.",
        help="LaTeX table caption",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_path.is_file():
        print(f"error: {args.csv_path} does not exist", file=sys.stderr)
        sys.exit(2)

    rows = load_rows(args.csv_path)
    if not rows:
        print(f"error: {args.csv_path} has no rows", file=sys.stderr)
        sys.exit(2)

    cdi = compute_cdi(rows)

    # Always print Markdown to stdout for quick eyeballing.
    md = format_cdi_markdown(cdi)
    print(md)

    # Optional writes.
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md + "\n", encoding="utf-8")
        print(f"\nWrote Markdown → {args.markdown}", file=sys.stderr)

    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(format_cdi_latex(cdi, caption=args.caption) + "\n", encoding="utf-8")
        print(f"Wrote LaTeX    → {args.latex}", file=sys.stderr)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(cdi_to_jsonable(cdi), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON     → {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
