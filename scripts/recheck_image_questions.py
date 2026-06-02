#!/usr/bin/env python3
"""
Diagnostic: re-evaluate the math image-anchored questions with vision EXPLICITLY
enabled, per critic model, and report whether each model — seeing the figure —
solves the item correctly.

Why this exists: the main calibration logs never showed the `[vision]` marker
for image items, leaving doubt whether the critic actually saw the figure. Here
we force each model to act as its own vision critic (monkeypatching the
capability set at runtime — the source pipeline is NOT modified) and print the
routing decision plus the resulting score, so the doubt is settled empirically.

Image items are auto-detected from the dataset (non-empty metadata.context that
resolves to a real file). Only the `real` variant is evaluated (the central
question is: can the critic solve the figure-dependent item correctly?).

Usage:
    python scripts/recheck_image_questions.py
    python scripts/recheck_image_questions.py --variant real --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src import agents as _agents
from src.agents import CriticAgent, VISION_CAPABLE_MODELS, _encode_image_data_url
from src.config import Config

from calibrate_critic import DATASET_FILES, real_to_generated_question

MODELS = {
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "gpt-5.5": "openai/gpt-5.5",
}


def find_image_items(dataset: list[dict], config: Config) -> list[int]:
    out = []
    for i, entry in enumerate(dataset):
        _, _, _, fp = real_to_generated_question(entry, config, "math")
        if fp:
            out.append(i)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Detect image items + routing, no API calls")
    p.add_argument("--out", default="output/_critic_validation/image_recheck.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    # Let gemini & gpt-5.5 accept images for THIS process only. Source untouched.
    _agents.VISION_CAPABLE_MODELS |= set(MODELS.values())

    dataset = json.load(open(_REPO_ROOT / DATASET_FILES["math"], encoding="utf-8"))
    image_idx = find_image_items(dataset, config)
    print(f"Image items detected: {image_idx}")

    results = []
    for name, model_id in MODELS.items():
        for idx in image_idx:
            q, level, gt, fp = real_to_generated_question(dataset[idx], config, "math")
            enc = _encode_image_data_url(fp) if fp else None
            use_vision = enc is not None and (model_id in _agents.VISION_CAPABLE_MODELS)
            print(f"\n[{name}] idx={idx} figure={Path(fp).name if fp else None} "
                  f"vision_will_fire={use_vision}")
            if args.dry_run:
                continue
            critic = CriticAgent(config, text_model_override=model_id, vision_model_override=model_id)
            try:
                fb = critic.evaluate(question=q, level=level, subject="math", figure_path=fp)
            except Exception as exc:  # noqa: BLE001
                print(f"   ERROR: {exc}")
                results.append({"model": model_id, "idx": idx, "error": str(exc)})
                continue
            ok = fb.critic_answer.strip().upper() == gt.strip().upper()
            print(f"   critic_answer={fb.critic_answer} gt={gt} correct={ok} "
                  f"overall={fb.overall_score} figure_relevance={fb.dimensions.figure_relevance}")
            results.append({
                "model": model_id, "idx": idx, "vision_used": use_vision,
                "critic_answer": fb.critic_answer, "ground_truth": gt, "correct": ok,
                "overall": fb.overall_score,
                "correctness": fb.dimensions.correctness,
                "figure_relevance": fb.dimensions.figure_relevance,
            })

    if args.dry_run:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Per-model accuracy on image items.
    print("\n=== image-item accuracy (with vision) ===")
    for name, model_id in MODELS.items():
        rs = [r for r in results if r.get("model") == model_id and "correct" in r]
        if rs:
            acc = sum(r["correct"] for r in rs) / len(rs)
            print(f"  {name:18s} {sum(r['correct'] for r in rs)}/{len(rs)} correct  (acc={acc:.2f})")


if __name__ == "__main__":
    main()
