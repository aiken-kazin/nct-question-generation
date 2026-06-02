#!/usr/bin/env python3
"""
Patch image-anchored math items with vision-enabled critic scores and emit a
blind-vs-vision ablation table.

The main calibration evaluated the 3 figure-dependent math items without the
image (critics answered blind and failed the figure-dependent ones). This script
re-evaluates the `real` variant of those items per critic model WITH vision
(each model acts as its own vision critic — consistent with the 3-model ensemble
design), splices the corrected rows back into each math CSV, and records the
old(blind)→new(vision) answers for an ablation table.

Source pipeline is NOT modified (vision capability is monkeypatched at runtime).

Usage:
    python scripts/patch_image_questions.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src import agents as _agents
from src.agents import CriticAgent
from src.config import Config
from src.output import _model_slug

from calibrate_critic import DATASET_FILES, feedback_row, real_to_generated_question

MODELS = {
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "gpt-5.5": "openai/gpt-5.5",
}
OUTDIR = Path("output/_critic_validation")

FIELDNAMES = [
    "idx", "variant", "topic", "level",
    "overall", "correctness", "distractor_quality", "difficulty_alignment",
    "kazakh_language_quality", "latex_validity", "figure_relevance",
    "pass_fail", "critic_answer", "ground_truth_answer", "critic_matches_gt", "error",
    "ensemble_agreement", "ensemble_unanimous",
]


def image_indices(dataset, config) -> list[int]:
    out = []
    for i, e in enumerate(dataset):
        if real_to_generated_question(e, config, "math")[3]:
            out.append(i)
    return out


def main() -> None:
    config = Config()
    if not config.api_key:
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    # Let each model accept images for this process only (source untouched).
    _agents.VISION_CAPABLE_MODELS |= set(MODELS.values())

    dataset = json.load(open(_REPO_ROOT / DATASET_FILES["math"], encoding="utf-8"))
    idxs = image_indices(dataset, config)
    print(f"Image items: {idxs}")

    ablation = []  # blind vs vision rows
    for name, model_id in MODELS.items():
        slug = _model_slug(model_id)
        csv_path = OUTDIR / f"math_{slug}.csv"
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        critic = CriticAgent(config, text_model_override=model_id, vision_model_override=model_id)

        for idx in idxs:
            q, level, gt, fp = real_to_generated_question(dataset[idx], config, "math")
            # capture OLD (blind) answer for ablation
            old = next((r for r in rows if r["variant"] == "real" and r["idx"] == str(idx)), None)
            old_ans = old["critic_answer"] if old else ""
            try:
                fb = critic.evaluate(question=q, level=level, subject="math", figure_path=fp)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] idx={idx} ERROR {exc}")
                continue
            new = feedback_row(idx=idx, variant="real", q=q, level=level, gt=gt, fb=fb)
            for r in rows:
                if r["variant"] == "real" and r["idx"] == str(idx):
                    r.update({k: new.get(k, r.get(k, "")) for k in FIELDNAMES})
                    break
            ablation.append({
                "model": model_id, "idx": idx, "gt": gt,
                "blind_answer": old_ans, "blind_correct": (old_ans.strip().upper() == gt.upper()),
                "vision_answer": fb.critic_answer,
                "vision_correct": (fb.critic_answer.strip().upper() == gt.upper()),
            })
            print(f"  [{name}] idx={idx}: blind={old_ans} -> vision={fb.critic_answer} (gt={gt})")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  patched {csv_path.name}")

    # Ablation artifacts
    (OUTDIR / "image_ablation.json").write_text(
        json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")

    def acc(key):
        per = {}
        for m in MODELS.values():
            rs = [a for a in ablation if a["model"] == m]
            per[m] = (sum(a[key] for a in rs), len(rs))
        return per

    blind, vision = acc("blind_correct"), acc("vision_correct")
    lines = ["# Vision ablation on figure-dependent math items (n=3 per model)\n",
             "| Critic | Blind (no image) | With image |", "|---|---:|---:|"]
    for name, m in MODELS.items():
        b, bn = blind[m]; v, vn = vision[m]
        lines.append(f"| {name} | {b}/{bn} | {v}/{vn} |")
    tb = sum(b for b, _ in blind.values()); tn = sum(n for _, n in blind.values())
    tv = sum(v for v, _ in vision.values())
    lines += ["", f"**Total: blind {tb}/{tn} → with image {tv}/{tn}.** "
              "Figure-dependent items the critic cannot solve without the image "
              "are recovered once the figure is supplied."]
    md = "\n".join(lines) + "\n"
    Path("paper/critic_validation").mkdir(parents=True, exist_ok=True)
    (Path("paper/critic_validation") / "image_ablation.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {OUTDIR/'image_ablation.json'} and paper/critic_validation/image_ablation.md")


if __name__ == "__main__":
    main()
