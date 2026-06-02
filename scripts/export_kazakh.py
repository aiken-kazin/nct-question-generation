#!/usr/bin/env python3
"""
Consolidate all generated Kazakh questions (standalone + context blocks) under a
run directory into ONE JSON file that is easy to parse and hand to others.

Walks:
  <root>/kazakh/<model>/level_*/*.json        -> standalone (no-context) items
  <root>/kazakh/<model>/context/*.json        -> context blocks (passage + 5 Q)

Usage:
    python scripts/export_kazakh.py --root output/kazakh_final
    python scripts/export_kazakh.py --root output/kazakh_final --out output/kazakh_final/all_kazakh_questions.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="output/kazakh_final", help="Run output directory")
    p.add_argument("--subject", default="kazakh", help="Subject subfolder under <root> (kazakh/math)")
    p.add_argument("--out", default=None, help="Output JSON path (default <root>/all_<subject>_questions.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    kz = root / args.subject
    out_path = Path(args.out) if args.out else root / f"all_{args.subject}_questions.json"

    no_context: list[dict] = []
    with_context: list[dict] = []

    if kz.is_dir():
        for model_dir in sorted(p for p in kz.iterdir() if p.is_dir()):
            model = model_dir.name
            # standalone items under level_* dirs
            for lvl_dir in sorted(model_dir.glob("level_*")):
                for jf in sorted(lvl_dir.glob("*.json")):
                    if jf.name.startswith("_"):
                        continue
                    d = json.loads(jf.read_text(encoding="utf-8"))
                    no_context.append({
                        "type": "no_context",
                        "model": d.get("model_id", model),
                        "level": d.get("level"),
                        "topic": d.get("topic"),
                        "question_text": d.get("question_text"),
                        "options": d.get("options"),
                        "correct_answer": d.get("correct_answer"),
                        "explanation": d.get("explanation"),
                        "critic_score": d.get("critic_score"),
                        "critic_feedback": d.get("critic_feedback"),
                        "ensemble": d.get("ensemble"),
                        "source_file": str(jf),
                    })
            # context blocks
            for jf in sorted((model_dir / "context").glob("*.json")):
                d = json.loads(jf.read_text(encoding="utf-8"))
                with_context.append({
                    "type": "with_context",
                    "model": d.get("model_id", model),
                    "level": d.get("level"),
                    "topic": d.get("topic"),
                    "passage": d.get("passage"),
                    "questions": d.get("questions"),
                    "ensemble_critic": d.get("ensemble_critic"),
                    "source_file": str(jf),
                })

    n_ctx_q = sum(len(b.get("questions", [])) for b in with_context)
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "counts": {
            "no_context_questions": len(no_context),
            "context_blocks": len(with_context),
            "context_questions": n_ctx_q,
            "total_questions": len(no_context) + n_ctx_q,
        },
        "no_context": no_context,
        "with_context": with_context,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"no_context items : {len(no_context)}")
    print(f"context blocks   : {len(with_context)}  ({n_ctx_q} questions)")
    print(f"total questions  : {bundle['counts']['total_questions']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
