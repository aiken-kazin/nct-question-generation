#!/usr/bin/env python3
"""
Context-based (мәнмәтіндік) question generation: produce ONE shared passage +
N linked MCQs, then validate each question with a single critic or the
3-model ensemble (the passage is shown to the critic so it can solve in
context).

Implements the NTC "context block" the standalone pipeline lacks. Output is a
single bundle JSON + a readable Markdown, saved under
  output/<subject>/<model_slug>/context/

Usage:
    python scripts/generate_context.py --subject kazakh --level B --api claude-sonnet-4.6 --ensemble
    python scripts/generate_context.py --subject kazakh --level B --api gpt-5.5 --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from string import Template

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from rich.console import Console

from src.agents import CriticAgent, _BASE_URL, _parse_json
from src.config import Config
from src.ensemble import EnsembleCriticAgent
from src.models import GeneratedQuestion, QuestionOption
from src.output import _model_slug

API_CHOICES = {
    "gpt-5.5": "openai/gpt-5.5",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
}
DEFAULT_ENSEMBLE = ["openai/gpt-5.5", "anthropic/claude-sonnet-4.6", "google/gemini-3.1-pro-preview"]
console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="kazakh", choices=["kazakh", "math"])
    p.add_argument("--level", default="B", choices=["A", "B", "C"])
    p.add_argument("--api", default="claude-sonnet-4.6", choices=list(API_CHOICES.keys()))
    p.add_argument("--model", default=None, help="Raw OpenRouter model ID (overrides --api)")
    p.add_argument("--n", type=int, default=5, help="Questions per passage (NTC uses 5)")
    p.add_argument("--ensemble", action="store_true", help="Validate with 3-model ensemble")
    p.add_argument("--topic", default="style_and_text_analysis")
    p.add_argument("--output-dir", default="output")
    return p.parse_args()


def _real_example_block() -> str:
    """Format the single real NTC context block as a few-shot example so the
    generator matches the real exam's style and structure. Returns '' if the
    file is absent."""
    path = _REPO_ROOT / "files" / "kazakh_questions_with_context.json"
    if not path.is_file():
        return ""
    try:
        items = json.load(open(path, encoding="utf-8"))
    except Exception:
        return ""
    if not items:
        return ""
    passage = items[0]["metadata"].get("context", "").strip()
    lines = [
        "## Reference example — a REAL NTC context block (match this style/structure)",
        "",
        "PASSAGE:", passage, "",
        "QUESTIONS (one shared passage, each single-correct):",
    ]
    for i, e in enumerate(items, 1):
        opts = e.get("options", {})
        opt_str = "  ".join(f"{k}) {v}" for k, v in opts.items())
        key = e["metadata"].get("correct_answer", "")
        lines.append(f"{i}. {e['question_text']}  [{opt_str}]  (correct: {key})")
    lines += ["", "Produce a NEW block in this style on a DIFFERENT text. Do not copy the example."]
    return "\n".join(lines)


def build_prompt(config: Config, subject: str, level: str, n: int) -> str:
    if subject != "kazakh":
        raise SystemExit("context generation prompt is implemented for kazakh only")
    tpl = (_REPO_ROOT / "prompts" / "generator_kazakh_context.md").read_text(encoding="utf-8")
    diff = config.difficulty_info(level, subject)
    return Template(tpl).safe_substitute(
        n_questions=n,
        level=level,
        difficulty_description=diff["description"].strip(),
        distractor_guidance=diff["distractor_guidance"].strip(),
        example_block=_real_example_block(),
    )


def main() -> None:
    args = parse_args()
    config = Config()
    if not config.api_key:
        console.print("[red]OPENROUTER_API_KEY not set[/red]")
        sys.exit(1)
    config.model = args.model or API_CHOICES[args.api]
    config.critic_model = config.model

    console.rule(f"[bold]Context-block generation[/bold] — {args.subject}/{args.level} "
                 f"gen=[cyan]{config.model}[/cyan] n={args.n} ensemble={args.ensemble}")

    # 1) Generate the bundle (passage + N questions) in one call.
    # Big passages occasionally yield malformed JSON (unescaped quotes/newlines),
    # so retry a few times before giving up.
    client = OpenAI(api_key=config.api_key, base_url=_BASE_URL)
    system_prompt = build_prompt(config, args.subject, args.level, args.n)
    data = None
    last_err: Exception | None = None
    for attempt in range(1, config.max_retries + 2):
        resp = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate the passage and exactly {args.n} linked questions. Return only the JSON object."},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        try:
            data = _parse_json(resp.choices[0].message.content or "")
            if data.get("passage") and len(data.get("questions", [])) >= 1:
                break
            raise ValueError("missing passage or questions")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            console.print(f"  [yellow]attempt {attempt}: bad JSON ({exc}); retrying[/yellow]")
            data = None
    if data is None:
        console.print(f"[red]Generation failed after retries: {last_err}[/red]")
        sys.exit(1)
    passage = data["passage"]
    questions = data["questions"]
    console.print(f"[green]Passage generated[/green] ({len(passage)} chars), {len(questions)} questions\n")
    console.print(passage + "\n")

    # 2) Critic setup.
    if args.ensemble:
        critic = EnsembleCriticAgent(config, model_ids=DEFAULT_ENSEMBLE)
    else:
        critic = CriticAgent(config)

    # 3) Evaluate each question with the passage shown as context.
    results = []
    for i, q in enumerate(questions, 1):
        options = [QuestionOption(label=o["label"], text=str(o["text"])) for o in q["options"]]
        gq = GeneratedQuestion(
            topic=args.topic,
            question_text=f"Мәтінді оқып, сұраққа жауап беріңіз.\n\nМӘТІН:\n{passage}\n\nСҰРАҚ: {q['question_text']}",
            options=options,
            correct_answer=q["correct_answer"],
            explanation=q.get("explanation", ""),
        )
        ret = critic.evaluate(question=gq, level=args.level, subject=args.subject, figure_path=None)
        fb = ret.aggregated if args.ensemble else ret
        agreement = getattr(ret, "answer_agreement", None)
        ok = fb.critic_answer.strip().upper() == q["correct_answer"].strip().upper()
        console.print(f"  Q{i}: key={q['correct_answer']} critic={fb.critic_answer} match={ok} "
                      f"score={fb.overall_score:.1f} pass={fb.pass_fail}"
                      + (f" agreement={agreement:.0%}" if agreement is not None else ""))
        results.append({
            "question_text": q["question_text"],
            "options": [o.model_dump() for o in options],
            "correct_answer": q["correct_answer"],
            "explanation": q.get("explanation", ""),
            "critic_score": fb.overall_score,
            "critic_answer": fb.critic_answer,
            "critic_matches_key": ok,
            "pass_fail": bool(fb.pass_fail),
            "dimensions": fb.dimensions.model_dump(),
            "ensemble": ret.to_dict() if args.ensemble else None,
        })

    # 4) Save bundle.
    bundle = {
        "id": uuid.uuid4().hex,
        "subject": args.subject,
        "level": args.level,
        "format": "context",
        "topic": args.topic,
        "model_id": config.model,
        "ensemble_critic": args.ensemble,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passage": passage,
        "questions": results,
    }
    out_dir = Path(args.output_dir) / args.subject / _model_slug(config.model) / "context"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{bundle['timestamp'][:19].replace(':', '-').replace('T', '_')}_{bundle['id'][:8]}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md = [f"# Context block — {args.subject.upper()} / Level {args.level}", "",
          f"**Model:** {config.model}  **Ensemble critic:** {args.ensemble}", "",
          "## Мәтін (passage)", "", passage, ""]
    for i, r in enumerate(results, 1):
        md += [f"## Question {i}", "", r["question_text"], ""]
        for o in r["options"]:
            mark = " ✓" if o["label"] == r["correct_answer"] else ""
            md.append(f"- **{o['label']})** {o['text']}{mark}")
        md += ["", f"**Key:** {r['correct_answer']}  |  **Critic:** {r['critic_answer']} "
               f"(match={r['critic_matches_key']}, score={r['critic_score']:.1f}, pass={r['pass_fail']})",
               "", f"_Explanation:_ {r['explanation']}", ""]
    (out_dir / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")

    n_match = sum(1 for r in results if r["critic_matches_key"])
    n_pass = sum(1 for r in results if r["pass_fail"])
    console.print(f"\n[bold]Summary:[/bold] {len(results)} questions, "
                  f"critic-key match {n_match}/{len(results)}, passed {n_pass}/{len(results)}")
    console.print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
