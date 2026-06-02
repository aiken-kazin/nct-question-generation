#!/usr/bin/env python3
"""
Targeted repair: re-run ONLY the failed (idx, variant) rows in an existing
calibration CSV and splice the results back in. Avoids re-spending credits on
the whole subject when only a few critic calls failed (e.g. transient JSON or
402-credit errors).

Reuses the exact question-construction and degradation logic from
calibrate_critic.py so patched rows are consistent with the original run.
Degraded variants use a fresh seeded RNG — for CDI purposes any valid
degradation is equivalent, so bit-exact reproduction is unnecessary.

Usage:
    python scripts/patch_failed.py --subject math --api gemini-3.1-pro
    python scripts/patch_failed.py --subject kazakh --api gpt-5.5 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.agents import CriticAgent
from src.config import Config
from src.output import _model_slug

from calibrate_critic import (
    API_CHOICES,
    DATASET_FILES,
    feedback_row,
    make_weak_distractors_variant,
    make_wrong_key_variant,
    real_to_generated_question,
)

# Same column order calibrate_critic.py writes, so the CSV stays uniform.
FIELDNAMES = [
    "idx", "variant", "topic", "level",
    "overall", "correctness", "distractor_quality", "difficulty_alignment",
    "kazakh_language_quality", "latex_validity", "figure_relevance",
    "pass_fail", "critic_answer", "ground_truth_answer", "critic_matches_gt", "error",
    "ensemble_agreement", "ensemble_unanimous",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, choices=["math", "kazakh"])
    p.add_argument("--api", default=None, choices=list(API_CHOICES.keys()))
    p.add_argument("--model", default=None, help="Raw OpenRouter model ID (overrides --api)")
    p.add_argument("--output-dir", default="output/_critic_validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="List failed rows, do not call the API")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if not args.model and not args.api:
        print("error: pass --api or --model", file=sys.stderr)
        sys.exit(2)
    config.model = args.model if args.model else API_CHOICES[args.api]
    config.critic_model = config.model
    slug = _model_slug(config.model)

    csv_path = Path(args.output_dir) / f"{args.subject}_{slug}.csv"
    if not csv_path.is_file():
        print(f"error: {csv_path} not found", file=sys.stderr)
        sys.exit(2)

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    failed = [(i, r) for i, r in enumerate(rows) if r.get("error")]
    if not failed:
        print(f"{csv_path.name}: no failed rows, nothing to do.")
        return

    print(f"{csv_path.name}: {len(failed)} failed rows:")
    for _, r in failed:
        print(f"  idx={r['idx']} variant={r['variant']}")
    if args.dry_run:
        return

    if not config.api_key:
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    dataset = json.load(open(_REPO_ROOT / DATASET_FILES[args.subject], encoding="utf-8"))
    critic = CriticAgent(config)
    rng = random.Random(args.seed)

    patched = 0
    for pos, r in failed:
        idx = int(r["idx"])
        variant = r["variant"]
        q, level, gt, figure_path = real_to_generated_question(dataset[idx], config, args.subject)
        if variant == "wrong_key":
            qx = make_wrong_key_variant(q, rng)
        elif variant == "weak_distractors":
            qx = make_weak_distractors_variant(q, rng)
        else:
            qx = q  # real
        try:
            fb = critic.evaluate(question=qx, level=level, subject=args.subject, figure_path=figure_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  idx={idx} {variant}: STILL FAILED ({exc})")
            continue
        new_row = feedback_row(idx=idx, variant=variant, q=qx, level=level, gt=gt, fb=fb)
        if new_row.get("error"):
            print(f"  idx={idx} {variant}: STILL FAILED (critic returned no score)")
            continue
        rows[pos] = {k: new_row.get(k, r.get(k, "")) for k in FIELDNAMES}
        patched += 1
        print(f"  idx={idx} {variant}: OK overall={new_row['overall']} correct={new_row['correctness']}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    still = sum(1 for r in rows if r.get("error"))
    print(f"\nPatched {patched}/{len(failed)}. Remaining failed rows: {still}.")
    print(f"Wrote {csv_path}")
    print(f"Now refresh tables:\n  python scripts/compute_cdi.py {csv_path} "
          f"--markdown paper/critic_validation/{args.subject}_{slug}.md "
          f"--latex paper/critic_validation/{args.subject}_{slug}.tex "
          f"--json paper/critic_validation/{args.subject}_{slug}.json")


if __name__ == "__main__":
    main()
