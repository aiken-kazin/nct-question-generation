#!/usr/bin/env python3
"""
Batch question-generation driver for the NTC research corpus.

Generates `count` Kazakh-language MCQs for a given subject and model, distributing
across difficulty levels per the NTC mix in prompts/difficulty.yaml and across
topics round-robin. Only critic-passing questions are saved; everything attempted
(saved + rejected + errored) is recorded in a per-run manifest.

Usage:
    python scripts/run_batch.py --subject math   --api gpt-5.5 --count 50
    python scripts/run_batch.py --subject kazakh --api claude-sonnet-4.6 --count 50
    python scripts/run_batch.py --subject math   --model anthropic/claude-sonnet-4.6 --count 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python scripts/run_batch.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from src.agents import CriticAgent, GeneratorAgent
from src.ensemble import EnsembleCriticAgent
from src.config import Config
from src.figure_gen import FigureGenerator
from src.models import Question
from src.output import save_question, _model_slug

# Same mapping the CLI uses — kept in sync with generate.py
API_CHOICES = {
    "gpt-5.5": "openai/gpt-5.5",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
}
DEFAULT_API = "gemini-3.1-pro"

DEFAULT_ENSEMBLE = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
]


def _resolve_ensemble_list(raw: str | None) -> list[str]:
    """Parse --ensemble-critics value. Mirrors generate.py."""
    if not raw:
        return list(DEFAULT_ENSEMBLE)
    out: list[str] = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        out.append(API_CHOICES.get(s, s))
    return out

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", required=True, choices=["math", "kazakh", "both"])
    parser.add_argument("--api", default=DEFAULT_API, choices=list(API_CHOICES.keys()),
                        help="API model for the GENERATOR")
    parser.add_argument("--model", default=None,
                        help="Raw OpenRouter model ID for the GENERATOR (overrides --api)")
    parser.add_argument("--critic-api", default=None, choices=list(API_CHOICES.keys()),
                        help="API model for the CRITIC on text-format questions (default: same as --api)")
    parser.add_argument("--critic-model", default=None,
                        help="Raw OpenRouter model ID for the CRITIC on text-format questions (overrides --critic-api)")
    parser.add_argument("--vision-critic-api", default=None, choices=list(API_CHOICES.keys()),
                        help="API model for the CRITIC on IMAGE-format questions (default: gpt-4o)")
    parser.add_argument("--vision-critic-model", default=None,
                        help="Raw OpenRouter model ID for the CRITIC on IMAGE-format questions (overrides --vision-critic-api)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use multi-critic ensemble instead of a single critic")
    parser.add_argument("--ensemble-critics", default=None,
                        help="Comma-separated critic model IDs/shortcuts (default: GPT-5.5, Claude Sonnet 4.6, Gemini 3.1 Pro)")
    parser.add_argument("--ensemble-strict", action="store_true",
                        help="Require ALL critics to pass (default: majority)")
    parser.add_argument("--count", type=int, default=50, help="Questions to produce per subject (default 50)")
    parser.add_argument("--format", dest="fmt", default="text", choices=["text", "image"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--resume", action="store_true",
                        help="Skip if subject/model/level already has enough saved questions")
    return parser.parse_args()


def distribute_across_levels(total: int, difficulty_yaml: dict) -> dict[str, int]:
    """Allocate `total` questions to levels A/B/C using percentages from difficulty.yaml.

    Uses largest-remainder rounding so the per-level counts sum to exactly `total`.
    """
    pcts = {lvl: float(difficulty_yaml[lvl]["percentage"]) for lvl in ("A", "B", "C")}
    raw = {lvl: total * p / 100.0 for lvl, p in pcts.items()}
    floored = {lvl: int(v) for lvl, v in raw.items()}
    short = total - sum(floored.values())
    by_remainder = sorted(((v - floored[lvl], lvl) for lvl, v in raw.items()), reverse=True)
    for _, lvl in by_remainder[:short]:
        floored[lvl] += 1
    return floored


def existing_saved_count(output_dir: Path, subject: str, model_slug: str, level: str) -> int:
    level_dir = output_dir / subject / model_slug / f"level_{level}"
    if not level_dir.exists():
        return 0
    return sum(1 for p in level_dir.glob("*.json") if not p.name.startswith("_"))


def generate_one(
    *,
    config: Config,
    generator: GeneratorAgent,
    critic,  # CriticAgent | EnsembleCriticAgent — both expose .evaluate(...)
    fig_gen: FigureGenerator,
    subject: str,
    level: str,
    fmt: str,
    topic: str,
    output_dir: Path,
    ensemble_mode: bool = False,
) -> tuple[Question | None, dict]:
    """Run the generator+critic loop once, mirroring generate.py:main()."""
    last_raw = None
    last_critique = None
    last_ensemble = None
    last_figure_path = None
    feedback: str | None = None
    attempt = 0
    last_error: str | None = None

    for attempt in range(config.max_retries + 1):
        try:
            last_raw = generator.generate(
                subject=subject, level=level, fmt=fmt, topic=topic, feedback=feedback,
            )
        except Exception as exc:
            last_error = f"generator: {exc}"
            break

        if fmt == "image" and last_raw.figure_spec is None:
            feedback = (
                "CRITICAL ERROR: figure_spec was null. Image format REQUIRES a populated figure_spec. "
                "Return a fully-specified figure_spec object."
            )
            if attempt < config.max_retries:
                continue

        last_figure_path = None
        if fmt == "image" and last_raw.figure_spec is not None:
            try:
                last_figure_path = fig_gen.generate(last_raw.figure_spec, output_dir=output_dir / "figures")
            except Exception as exc:
                last_error = f"figure: {exc}"

        try:
            raw_critique = critic.evaluate(
                question=last_raw,
                level=level,
                subject=subject,
                figure_path=str(last_figure_path) if last_figure_path else None,
            )
        except Exception as exc:
            last_error = f"critic: {exc}"
            break

        if ensemble_mode:
            last_ensemble = raw_critique
            last_critique = raw_critique.aggregated
        else:
            last_ensemble = None
            last_critique = raw_critique

        if last_critique.pass_fail:
            break
        feedback = last_critique.improvement_suggestions or last_critique.comments

    attempts_used = attempt + 1
    rejected_reason: str | None = None
    if last_raw is None:
        rejected_reason = last_error or "generator returned nothing"
    elif last_critique is None:
        rejected_reason = last_error or "critic returned nothing"
    elif not last_critique.pass_fail or last_critique.dimensions.correctness < 7:
        rejected_reason = (
            f"failed critic: overall={last_critique.overall_score:.1f} "
            f"correctness={last_critique.dimensions.correctness:.1f}"
        )

    if rejected_reason:
        return None, {
            "topic": topic, "level": level, "format": fmt,
            "attempts": attempts_used, "saved": False, "reason": rejected_reason,
            "overall_score": last_critique.overall_score if last_critique else None,
        }

    question = Question(
        id=str(uuid.uuid4()),
        subject=subject,
        level=level,
        format=fmt,
        topic=topic,
        model_id=config.model,
        question_text=last_raw.question_text,
        options=last_raw.options,
        correct_answer=last_raw.correct_answer,
        explanation=last_raw.explanation,
        latex_formulas=last_raw.latex_formulas,
        figure_spec=last_raw.figure_spec,
        figure_path=str(last_figure_path) if last_figure_path else None,
        critic_score=last_critique.overall_score,
        critic_feedback=last_critique,
        ensemble=last_ensemble.to_dict() if last_ensemble is not None else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        generation_attempts=attempts_used,
    )
    return question, {
        "topic": topic, "level": level, "format": fmt,
        "attempts": attempts_used, "saved": True,
        "id": question.id, "overall_score": last_critique.overall_score,
        "ensemble_agreement": (last_ensemble.answer_agreement if last_ensemble else None),
        "ensemble_unanimous": (last_ensemble.unanimous if last_ensemble else None),
    }


def run_subject(
    *, config: Config, subject: str, count: int, fmt: str, seed: int,
    output_dir: Path, resume: bool,
    ensemble: bool = False, ensemble_models: list[str] | None = None,
    ensemble_strict: bool = False,
) -> dict:
    rng = random.Random(seed if seed else None)

    diff_yaml = config._difficulty  # already loaded; safe internal access
    per_level = distribute_across_levels(count, diff_yaml)
    topics = config.get_topics(subject)
    if not topics:
        raise RuntimeError(f"No topics defined for subject={subject}")

    generator = GeneratorAgent(config)
    if ensemble:
        critic = EnsembleCriticAgent(
            config, model_ids=ensemble_models or DEFAULT_ENSEMBLE,
            strict_pass=ensemble_strict,
        )
    else:
        critic = CriticAgent(config)
    fig_gen = FigureGenerator()

    model_slug = _model_slug(config.model)
    console.rule(f"[bold]{subject}[/bold] / model=[cyan]{model_slug}[/cyan] / total={count}")
    console.print(f"Level distribution: {per_level}")

    saved_count = 0
    attempt_log: list[dict] = []

    for level, n in per_level.items():
        if resume:
            already = existing_saved_count(output_dir, subject, model_slug, level)
            if already >= n:
                console.print(f"  [dim]Level {level}: already have {already}/{n}. Skipping.[/dim]")
                saved_count += n
                continue
            else:
                n_remaining = n - already
                console.print(f"  Level {level}: resuming, {already} present, generating {n_remaining} more.")
        else:
            n_remaining = n

        # Round-robin topics, shuffled so we don't always start at the same one
        topic_order = list(topics)
        rng.shuffle(topic_order)

        produced = 0
        topic_idx = 0
        safety_cap = n_remaining * 4  # don't loop forever if the critic keeps failing
        attempts_total = 0
        while produced < n_remaining and attempts_total < safety_cap:
            topic = topic_order[topic_idx % len(topic_order)]
            topic_idx += 1
            attempts_total += 1
            console.print(
                f"  [dim]level={level} {produced + 1}/{n_remaining} topic={topic}[/dim]"
            )
            question, log = generate_one(
                config=config, generator=generator, critic=critic, fig_gen=fig_gen,
                subject=subject, level=level, fmt=fmt, topic=topic, output_dir=output_dir,
                ensemble_mode=ensemble,
            )
            attempt_log.append(log)
            if question is not None:
                paths = save_question(question, output_dir)
                produced += 1
                saved_count += 1
                console.print(
                    f"    [green]saved[/green] {paths['json'].name} "
                    f"(score {question.critic_score:.1f}, attempts {question.generation_attempts})"
                )
            else:
                console.print(f"    [yellow]rejected[/yellow] — {log['reason']}")

        if produced < n_remaining:
            console.print(
                f"  [red]Level {level} fell short: produced {produced}/{n_remaining} "
                f"after {attempts_total} attempts. Moving on.[/red]"
            )

    # Manifest
    manifest_dir = output_dir / subject / model_slug
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    manifest_path = manifest_dir / f"_manifest_{ts}.json"
    manifest = {
        "subject": subject,
        "model_id": config.model,
        "model_slug": model_slug,
        "intended_count": count,
        "saved_count": saved_count,
        "rejected_count": sum(1 for a in attempt_log if not a["saved"]),
        "per_level_intended": per_level,
        "format": fmt,
        "seed": seed,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempt_log,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold]Manifest:[/bold] {manifest_path}")
    return manifest


def main() -> None:
    args = parse_args()
    config = Config()
    if not config.api_key:
        console.print("[red]OPENROUTER_API_KEY not set in environment[/red]")
        sys.exit(1)
    config.model = args.model if args.model else API_CHOICES[args.api]

    # Resolve critic model: --critic-model > --critic-api > mirror generator.
    if args.critic_model:
        config.critic_model = args.critic_model
    elif args.critic_api:
        config.critic_model = API_CHOICES[args.critic_api]
    else:
        config.critic_model = config.model

    # Vision critic. Only matters for image-format runs. Default (GPT-4o)
    # already set in Config.
    if args.vision_critic_model:
        config.vision_critic_model = args.vision_critic_model
    elif args.vision_critic_api:
        config.vision_critic_model = API_CHOICES[args.vision_critic_api]

    subjects = ["math", "kazakh"] if args.subject == "both" else [args.subject]

    ensemble_models = _resolve_ensemble_list(args.ensemble_critics) if args.ensemble else None

    summaries = []
    for subj in subjects:
        m = run_subject(
            config=config,
            subject=subj,
            count=args.count,
            fmt=args.fmt,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            resume=args.resume,
            ensemble=args.ensemble,
            ensemble_models=ensemble_models,
            ensemble_strict=args.ensemble_strict,
        )
        summaries.append(m)

    # Final summary table
    table = Table(title="Batch run summary", header_style="bold")
    table.add_column("Subject")
    table.add_column("Model slug")
    table.add_column("Saved", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Intended", justify="right")
    for m in summaries:
        table.add_row(m["subject"], m["model_slug"], str(m["saved_count"]),
                      str(m["rejected_count"]), str(m["intended_count"]))
    console.print(table)


if __name__ == "__main__":
    main()
