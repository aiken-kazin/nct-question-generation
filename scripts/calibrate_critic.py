#!/usr/bin/env python3
"""
Critic self-validation: run CriticAgent against real NTC questions and
degraded variants. No experts required.

Two tests run back-to-back per subject:

  Test A — Real-question pass rate
    Real questions should score high. We log per-dimension scores and whether
    the critic's independent answer matches the ground-truth correct_answer.

  Test B — Degraded-variant discrimination
    For each real question we synthesise two degraded variants:
      * wrong_key:        relabel the correct option to a random distractor
      * weak_distractors: replace 2 distractors with obviously implausible strings
    A well-calibrated critic should score these materially lower than the real one.

Outputs (under output/_critic_validation/):
  <subject>_<model_slug>.csv             — one row per (question, variant)
  <subject>_<model_slug>_summary.json    — aggregate stats
  <subject>_<model_slug>_details.json    — full critic feedback per row

Usage:
    python scripts/calibrate_critic.py --subject math --limit 5 --api gpt-4o-2024-11-20
    python scripts/calibrate_critic.py --subject both --api claude-sonnet-4.6 --no-degrade
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from src.agents import CriticAgent
from src.cdi import compute_cdi, format_cdi_markdown, cdi_to_jsonable, pairwise_kappa_matrix
from src.config import Config
from src.ensemble import EnsembleCriticAgent
from src.models import CriticFeedback, GeneratedQuestion, QuestionOption
from src.output import _model_slug

API_CHOICES = {
    "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
    "Qwen/Qwen2.5-72B-Instruct": "qwen/qwen-2.5-72b-instruct",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
}
DEFAULT_API = "gpt-4o-2024-11-20"

DEFAULT_ENSEMBLE = [
    "openai/gpt-4o-2024-11-20",
    "anthropic/claude-sonnet-4.6",
    "qwen/qwen-2.5-72b-instruct",
]


def _resolve_ensemble_list(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_ENSEMBLE)
    out: list[str] = []
    for item in raw.split(","):
        s = item.strip()
        if s:
            out.append(API_CHOICES.get(s, s))
    return out

console = Console()

# Map subject → dataset path. Skip a subject silently if the file is missing.
DATASET_FILES = {
    "math": "files/mathematics_questions_kz.json",
    "kazakh": "files/kazakh_language_questions_kz.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="math", choices=["math", "kazakh", "both"])
    parser.add_argument("--api", default=DEFAULT_API, choices=list(API_CHOICES.keys()))
    parser.add_argument("--model", default=None, help="Raw OpenRouter model ID (overrides --api)")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of real questions tested")
    parser.add_argument("--no-degrade", action="store_true", help="Skip degraded-variant Test B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/_critic_validation")
    # Ensemble: each "row" represents the ENSEMBLE verdict. We also write
    # per-critic rows so we can compute pairwise Cohen's κ across critics.
    parser.add_argument("--ensemble", action="store_true",
                        help="Use multi-critic ensemble for calibration")
    parser.add_argument("--ensemble-critics", default=None,
                        help="Comma-separated critic models (default: GPT-4o, Claude Sonnet 4.6, Qwen-2.5-72B)")
    return parser.parse_args()


# ── Real-question normalization ──────────────────────────────────────────────


def normalize_level(raw: str) -> str:
    """Real dataset uses both Cyrillic 'А' and Latin 'A'. Normalize to A/B/C."""
    c = (raw or "").strip().upper()
    if c in ("A", "А"):  # Latin A, Cyrillic А
        return "A"
    if c == "B":
        return "B"
    if c == "C":
        return "C"
    return "B"  # safe default — Medium is the most common level


def _build_topic_index(config: Config, subject: str) -> dict[str, str]:
    """Map name_kz (lowercased, stripped) → topic id, for nearest-match lookups."""
    topics = config._topics_math if subject == "math" else config._topics_kazakh
    return {t["name_kz"].strip().lower(): t["id"] for t in topics}


def resolve_topic(config: Config, subject: str, raw_topic_kz: str) -> str:
    """Best-effort map a Kazakh topic name from the dataset to an internal topic id."""
    idx = _build_topic_index(config, subject)
    key = (raw_topic_kz or "").strip().rstrip(",").lower()
    if key in idx:
        return idx[key]
    # Substring match either direction
    for name, tid in idx.items():
        if key and (key in name or name in key):
            return tid
    # Unknown topic — return as-is; Config.get_topic_info falls back gracefully.
    return raw_topic_kz or "unknown"


_REAL_IMAGES_DIR = _REPO_ROOT / "files" / "mathematics_images"


def resolve_real_image_path(context_field: str) -> str | None:
    """The real-question dataset stores image references in `metadata.context`
    as paths like
        ./data/mathematics/parsed/mathematics_questions_kz/mathematics_images/image_222562.jpeg

    The actual file is in this repo at files/mathematics_images/<basename>.
    Map one to the other; return None if we can't find the file on disk.
    """
    if not context_field or not context_field.strip():
        return None
    basename = Path(context_field).name
    candidate = _REAL_IMAGES_DIR / basename
    return str(candidate) if candidate.is_file() else None


def real_to_generated_question(
    entry: dict, config: Config, subject: str
) -> tuple[GeneratedQuestion, str, str, str | None]:
    """Convert one raw dataset entry → (GeneratedQuestion, level, ground_truth_answer, figure_path).

    The fourth tuple element is the resolved local image path for
    image-anchored items, or None for text-only items. Critic evaluation
    routes to the vision model when this is non-None.
    """
    meta = entry["metadata"]
    options_dict = entry["options"]
    options = [QuestionOption(label=lbl, text=str(text)) for lbl, text in options_dict.items()]
    # Pad/truncate to exactly 4 — the model requires it.
    if len(options) < 4:
        for missing in "ABCD":
            if missing not in {o.label for o in options}:
                options.append(QuestionOption(label=missing, text=""))
                if len(options) == 4:
                    break
    options = options[:4]

    topic_id = resolve_topic(config, subject, meta.get("topic", ""))
    level = normalize_level(meta.get("difficulty", "B"))
    gt = (meta.get("correct_answer") or "A").strip().upper()
    figure_path = resolve_real_image_path(meta.get("context", ""))

    q = GeneratedQuestion(
        topic=topic_id,
        question_text=entry["question_text"],
        options=options,
        correct_answer=gt,
        explanation="",  # real dataset has no per-question explanation
        latex_formulas=[],
        figure_spec=None,
    )
    return q, level, gt, figure_path


# ── Degraded variants ───────────────────────────────────────────────────────


_WEAK_DISTRACTOR_POOL_KZ = ["Жоқ", "0", "100", "Дұрыс жауап жоқ", "Анықталмаған"]


def make_wrong_key_variant(q: GeneratedQuestion, rng: random.Random) -> GeneratedQuestion:
    """Relabel the correct answer to a random wrong option."""
    labels = [o.label for o in q.options]
    wrong = [lbl for lbl in labels if lbl != q.correct_answer] or labels
    new_correct = rng.choice(wrong)
    return q.model_copy(update={"correct_answer": new_correct})


def make_weak_distractors_variant(q: GeneratedQuestion, rng: random.Random) -> GeneratedQuestion:
    """Replace 2 of the wrong options with implausible filler strings."""
    distractors = [i for i, o in enumerate(q.options) if o.label != q.correct_answer]
    rng.shuffle(distractors)
    to_replace = distractors[:2]
    new_options = []
    pool = _WEAK_DISTRACTOR_POOL_KZ[:]
    rng.shuffle(pool)
    for i, o in enumerate(q.options):
        if i in to_replace and pool:
            new_options.append(QuestionOption(label=o.label, text=pool.pop()))
        else:
            new_options.append(o)
    return q.model_copy(update={"options": new_options})


# ── Critic evaluation wrapper ───────────────────────────────────────────────


def evaluate_one(
    *, critic: CriticAgent, q: GeneratedQuestion, level: str, subject: str,
    figure_path: str | None = None,
) -> CriticFeedback | None:
    """Single critic call. `figure_path` triggers vision-model routing inside
    the agent for image-anchored real items.
    """
    try:
        return critic.evaluate(question=q, level=level, subject=subject, figure_path=figure_path)
    except Exception as exc:
        console.print(f"  [red]critic error:[/red] {exc}")
        return None


def feedback_row(
    *, idx: int, variant: str, q: GeneratedQuestion, level: str, gt: str,
    fb: CriticFeedback | None,
) -> dict:
    if fb is None:
        return {
            "idx": idx, "variant": variant, "topic": q.topic, "level": level,
            "overall": None, "correctness": None, "distractor_quality": None,
            "difficulty_alignment": None, "kazakh_language_quality": None,
            "latex_validity": None, "figure_relevance": None,
            "pass_fail": None, "critic_answer": None,
            "ground_truth_answer": gt, "critic_matches_gt": None, "error": "critic_failed",
        }
    d = fb.dimensions
    return {
        "idx": idx, "variant": variant, "topic": q.topic, "level": level,
        "overall": fb.overall_score,
        "correctness": d.correctness,
        "distractor_quality": d.distractor_quality,
        "difficulty_alignment": d.difficulty_alignment,
        "kazakh_language_quality": d.kazakh_language_quality,
        "latex_validity": d.latex_validity,
        "figure_relevance": d.figure_relevance,
        "pass_fail": bool(fb.pass_fail),
        "critic_answer": fb.critic_answer,
        "ground_truth_answer": gt,
        "critic_matches_gt": fb.critic_answer.strip().upper() == gt.strip().upper(),
        "error": None,
    }


# ── Aggregation ─────────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def summarize(rows: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = {}
    for variant, group in by_variant.items():
        scored = [r for r in group if r["overall"] is not None]
        summary[variant] = {
            "n": len(group),
            "n_scored": len(scored),
            "pass_rate": (
                round(sum(1 for r in scored if r["pass_fail"]) / len(scored), 3)
                if scored else None
            ),
            "critic_answer_accuracy": (
                round(sum(1 for r in scored if r["critic_matches_gt"]) / len(scored), 3)
                if scored else None
            ),
            "mean_overall": _mean([r["overall"] for r in scored]),
            "mean_correctness": _mean([r["correctness"] for r in scored]),
            "mean_distractor_quality": _mean([r["distractor_quality"] for r in scored]),
            "mean_difficulty_alignment": _mean([r["difficulty_alignment"] for r in scored]),
            "mean_kazakh_language_quality": _mean([r["kazakh_language_quality"] for r in scored]),
            "mean_latex_validity": _mean([r["latex_validity"] for r in scored]),
        }
    return summary


def print_summary_table(summary: dict) -> None:
    table = Table(title="Critic self-validation summary", header_style="bold")
    table.add_column("Variant")
    table.add_column("n", justify="right")
    table.add_column("pass rate", justify="right")
    table.add_column("critic-vs-gt", justify="right")
    table.add_column("mean overall", justify="right")
    table.add_column("mean correctness", justify="right")
    for variant in ("real", "wrong_key", "weak_distractors"):
        if variant not in summary:
            continue
        s = summary[variant]
        table.add_row(
            variant, str(s["n_scored"]),
            f"{s['pass_rate']:.2f}" if s["pass_rate"] is not None else "—",
            f"{s['critic_answer_accuracy']:.2f}" if s["critic_answer_accuracy"] is not None else "—",
            f"{s['mean_overall']:.2f}" if s["mean_overall"] is not None else "—",
            f"{s['mean_correctness']:.2f}" if s["mean_correctness"] is not None else "—",
        )
    console.print(table)


# ── Per-subject driver ──────────────────────────────────────────────────────


def run_subject(
    *, subject: str, config: Config, limit: int | None, do_degrade: bool, seed: int,
    output_dir: Path,
    ensemble: bool = False, ensemble_models: list[str] | None = None,
) -> dict | None:
    rel_path = DATASET_FILES[subject]
    full_path = _REPO_ROOT / rel_path
    if not full_path.exists():
        console.print(f"[yellow]No dataset for subject={subject} at {rel_path}. Skipping.[/yellow]")
        return None

    with open(full_path, encoding="utf-8") as f:
        raw = json.load(f)
    if limit:
        raw = raw[:limit]

    rng = random.Random(seed)
    if ensemble:
        critic = EnsembleCriticAgent(config, model_ids=ensemble_models or DEFAULT_ENSEMBLE)
        model_slug = "ensemble_" + "_".join(_model_slug(m) for m in critic.model_ids)
        ensemble_model_ids = list(critic.model_ids)
    else:
        critic = CriticAgent(config)
        model_slug = _model_slug(config.model)
        ensemble_model_ids = []

    console.rule(f"[bold]{subject}[/bold] / critic=[cyan]{model_slug}[/cyan] / n={len(raw)}")

    rows: list[dict] = []
    details: list[dict] = []
    # Per-critic answer matrix for pairwise Cohen's κ (ensemble mode only).
    # Shape: {model_id: {idx_str: critic_answer}}. We use a dict-of-dict so
    # critic failures (no row) don't shift alignment.
    per_critic_answers: dict[str, dict[str, str]] = {m: {} for m in ensemble_model_ids}

    def _run(variant: str, qx, level, gt, figure_path, idx):
        """Inner: dispatch to single critic or ensemble and produce a row."""
        try:
            ret = critic.evaluate(
                question=qx, level=level, subject=subject, figure_path=figure_path,
            )
        except Exception as exc:
            console.print(f"  [red]critic error:[/red] {exc}")
            return feedback_row(idx=idx, variant=variant, q=qx, level=level, gt=gt, fb=None)

        if ensemble:
            # `ret` is an EnsembleCriticFeedback. The "row" is the aggregated
            # verdict; per-critic verdicts are captured separately for κ.
            row = feedback_row(idx=idx, variant=variant, q=qx, level=level, gt=gt, fb=ret.aggregated)
            row["ensemble_agreement"] = round(ret.answer_agreement, 3)
            row["ensemble_unanimous"] = ret.unanimous
            if variant == "real":
                for p in ret.per_critic:
                    if p.feedback is not None:
                        per_critic_answers[p.model_id][str(idx)] = p.feedback.critic_answer
            details.append({"row": row, "ensemble": ret.to_dict()})
            return row
        else:
            row = feedback_row(idx=idx, variant=variant, q=qx, level=level, gt=gt, fb=ret)
            details.append({"row": row, "feedback": ret.model_dump() if ret else None})
            return row

    for idx, entry in enumerate(raw):
        q, level, gt, figure_path = real_to_generated_question(entry, config, subject)
        img_tag = " [vision]" if figure_path else ""
        console.print(f"  [{idx + 1}/{len(raw)}] level={level} topic={q.topic[:30]}{img_tag}")

        # Test A — real
        row = _run("real", q, level, gt, figure_path, idx)
        rows.append(row)
        if row.get("overall") is not None:
            extra = ""
            if "ensemble_agreement" in row:
                extra = f"  agreement={row['ensemble_agreement']:.0%}"
            console.print(
                f"    real     → overall={row['overall']:.1f} "
                f"correct={row['correctness']:.1f} "
                f"critic_ans={row['critic_answer']} gt={gt}{extra}"
            )

        if not do_degrade:
            continue

        # Test B — wrong_key
        q_wk = make_wrong_key_variant(q, rng)
        row_wk = _run("wrong_key", q_wk, level, gt, figure_path, idx)
        rows.append(row_wk)
        if row_wk.get("overall") is not None:
            console.print(
                f"    wrongkey → overall={row_wk['overall']:.1f} "
                f"correct={row_wk['correctness']:.1f}"
            )

        # Test B — weak_distractors
        q_wd = make_weak_distractors_variant(q, rng)
        row_wd = _run("weak_distractors", q_wd, level, gt, figure_path, idx)
        rows.append(row_wd)
        if row_wd.get("overall") is not None:
            console.print(
                f"    weakdis  → overall={row_wd['overall']:.1f} "
                f"distractor_q={row_wd['distractor_quality']:.1f}"
            )

    summary = summarize(rows)
    print_summary_table(summary)

    # CDI — the paper's main critic-quality metric. Computed from the same
    # rows the CSV is built from, so the reader can reproduce it exactly.
    cdi = compute_cdi(rows) if do_degrade else None
    if cdi:
        console.print()
        console.print(format_cdi_markdown(cdi))

    # Inter-critic Cohen's κ (ensemble only): how often do the 3 critics
    # agree on the answer to a real item?
    kappa_matrix = None
    if ensemble and per_critic_answers:
        # Align labels by item index (only items where all 3 responded).
        common_idx = sorted(
            set.intersection(*(set(d.keys()) for d in per_critic_answers.values()))
            if per_critic_answers and all(per_critic_answers.values()) else set()
        )
        if common_idx:
            aligned = {
                m: [per_critic_answers[m][i] for i in common_idx]
                for m in per_critic_answers
            }
            raw_matrix = pairwise_kappa_matrix(aligned)
            kappa_matrix = [
                {"critic_a": a, "critic_b": b, "n_items": len(common_idx), "kappa": k}
                for (a, b), k in raw_matrix.items()
            ]
            console.print("\n[bold]Pairwise Cohen's κ (critic_answer on real items)[/bold]")
            for row_k in kappa_matrix:
                console.print(
                    f"  {row_k['critic_a']} ↔ {row_k['critic_b']}: "
                    f"κ={row_k['kappa']}  (n={row_k['n_items']})"
                )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{subject}_{model_slug}.csv"
    summary_path = output_dir / f"{subject}_{model_slug}_summary.json"
    details_path = output_dir / f"{subject}_{model_slug}_details.json"

    fieldnames = [
        "idx", "variant", "topic", "level",
        "overall", "correctness", "distractor_quality", "difficulty_alignment",
        "kazakh_language_quality", "latex_validity", "figure_relevance",
        "pass_fail", "critic_answer", "ground_truth_answer", "critic_matches_gt", "error",
        "ensemble_agreement", "ensemble_unanimous",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "subject": subject,
        "model_id": config.model,
        "model_slug": model_slug,
        "ensemble": ensemble,
        "ensemble_model_ids": ensemble_model_ids,
        "n_real_questions": len(raw),
        "degraded_enabled": do_degrade,
        "seed": seed,
        "summary": summary,
        "cdi": cdi_to_jsonable(cdi) if cdi else None,
        "pairwise_kappa": kappa_matrix,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"  Wrote {csv_path}")
    console.print(f"  Wrote {summary_path}")
    return summary_payload


def main() -> None:
    args = parse_args()
    config = Config()
    if not config.api_key:
        console.print("[red]OPENROUTER_API_KEY not set in environment[/red]")
        sys.exit(1)
    config.model = args.model if args.model else API_CHOICES[args.api]

    subjects = ["math", "kazakh"] if args.subject == "both" else [args.subject]
    out_dir = Path(args.output_dir)
    ensemble_models = _resolve_ensemble_list(args.ensemble_critics) if args.ensemble else None

    for subj in subjects:
        run_subject(
            subject=subj,
            config=config,
            limit=args.limit,
            do_degrade=not args.no_degrade,
            seed=args.seed,
            output_dir=out_dir,
            ensemble=args.ensemble,
            ensemble_models=ensemble_models,
        )


if __name__ == "__main__":
    main()
