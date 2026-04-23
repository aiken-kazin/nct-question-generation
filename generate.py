#!/usr/bin/env python3
"""
Kazakhstan NTC — Kazakh-language teacher certification exam question generator.

Usage:
    python generate.py --subject math --level B --format text
    python generate.py --subject kazakh --level A --format image --count 3
"""
import argparse
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import Config
from src.agents import GeneratorAgent, CriticAgent
from src.figure_gen import FigureGenerator
from src.models import Question
from src.output import save_question

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Kazakh-language teacher certification MCQ questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True, choices=["math", "kazakh"],
                        help="Subject area")
    parser.add_argument("--level", required=True, choices=["A", "B", "C"],
                        help="Difficulty level: A=Basic(26%%), B=Medium(60%%), C=Hard(14%%)")
    parser.add_argument("--format", required=True, choices=["text", "image"],
                        dest="fmt", help="Question format")
    parser.add_argument("--topic", default=None,
                        help="Topic ID (from topics_math/kazakh.yaml). Random if omitted.")
    parser.add_argument("--model", default=None,
                        help="OpenRouter model ID (overrides OPENROUTER_MODEL env var)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of questions to generate (default: 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config()
    if not config.api_key:
        console.print("[red]Error:[/red] OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    if args.model:
        config.model = args.model

    topics_available = config.get_topics(args.subject)

    generator = GeneratorAgent(config)
    critic = CriticAgent(config)
    fig_gen = FigureGenerator()
    output_dir = Path(args.output_dir)

    console.print(Panel(
        f"[bold]NTC Question Generator[/bold]\n"
        f"Model: [cyan]{config.model}[/cyan]  "
        f"Subject: [yellow]{args.subject}[/yellow]  "
        f"Level: [green]{args.level}[/green]  "
        f"Format: [blue]{args.fmt}[/blue]  "
        f"Count: [magenta]{args.count}[/magenta]",
        border_style="blue",
    ))

    questions: list[Question] = []

    for i in range(args.count):
        topic = args.topic or random.choice(topics_available)
        console.rule(f"Question {i+1}/{args.count} — topic: [magenta]{topic}[/magenta]")

        last_raw = None
        last_critique = None
        last_figure_path = None
        feedback: str | None = None

        for attempt in range(config.max_retries + 1):
            attempt_label = f"[dim]attempt {attempt+1}/{config.max_retries+1}[/dim]"

            with console.status(f"Generator Agent {attempt_label}..."):
                try:
                    last_raw = generator.generate(
                        subject=args.subject,
                        level=args.level,
                        fmt=args.fmt,
                        topic=topic,
                        feedback=feedback,
                    )
                except Exception as exc:
                    console.print(f"[red]Generator error:[/red] {exc}")
                    break

            # Validate: image format MUST produce a figure_spec
            if args.fmt == "image" and last_raw.figure_spec is None:
                console.print(
                    f"[yellow]figure_spec is null (attempt {attempt+1}). "
                    "Retrying with explicit instruction...[/yellow]"
                )
                feedback = (
                    "CRITICAL ERROR: You set figure_spec to null. This is NOT allowed for "
                    "image-format questions. You MUST provide a figure_spec with a valid "
                    "figure_type and parameters. The figure must contain the specific "
                    "measurements (side lengths, angles, coordinates) needed to solve the question. "
                    "Do NOT set figure_spec to null under any circumstances."
                )
                if attempt < config.max_retries:
                    continue
                else:
                    console.print("[red]Max retries reached with no figure_spec. Saving without figure.[/red]")

            last_figure_path = None
            if args.fmt == "image" and last_raw.figure_spec:
                with console.status("Generating figure..."):
                    try:
                        last_figure_path = fig_gen.generate(
                            last_raw.figure_spec,
                            output_dir=output_dir / "figures",
                        )
                        console.print(f"  [green]Figure saved:[/green] {last_figure_path.name}")
                    except Exception as exc:
                        console.print(f"[red]Figure error:[/red] {exc}")

            with console.status(f"Critic Agent {attempt_label}..."):
                try:
                    last_critique = critic.evaluate(
                        question=last_raw,
                        level=args.level,
                        subject=args.subject,
                        figure_path=str(last_figure_path) if last_figure_path else None,
                    )
                except Exception as exc:
                    console.print(f"[red]Critic error:[/red] {exc}")
                    break

            _display_critic_table(last_critique)

            if last_critique.pass_fail:
                console.print(f"[green]PASS[/green] — score {last_critique.overall_score:.1f}/10")
                break
            else:
                console.print(
                    f"[yellow]FAIL[/yellow] — score {last_critique.overall_score:.1f}/10"
                    + (f". Retrying..." if attempt < config.max_retries else ". Max retries reached.")
                )
                feedback = last_critique.improvement_suggestions or last_critique.comments

        if last_raw is None:
            console.print("[red]Skipping — no output from generator.[/red]")
            continue

        question = Question(
            id=str(uuid.uuid4()),
            subject=args.subject,
            level=args.level,
            format=args.fmt,
            topic=topic,
            question_text=last_raw.question_text,
            options=last_raw.options,
            correct_answer=last_raw.correct_answer,
            explanation=last_raw.explanation,
            latex_formulas=last_raw.latex_formulas,
            figure_spec=last_raw.figure_spec,
            figure_path=str(last_figure_path) if last_figure_path else None,
            critic_score=last_critique.overall_score if last_critique else None,
            critic_feedback=last_critique,
            timestamp=datetime.now(timezone.utc).isoformat(),
            generation_attempts=attempt + 1,
        )

        paths = save_question(question, output_dir)
        questions.append(question)
        console.print(
            f"Saved [green]{paths['json'].name}[/green] + "
            f"[green]{paths['markdown'].name}[/green]"
        )
        _display_question_preview(question)

    if questions:
        _display_summary(questions)


def _display_critic_table(critique) -> None:
    if critique is None:
        return
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Dimension", style="cyan", min_width=26)
    table.add_column("Score", justify="right", min_width=8)

    d = critique.dimensions
    rows = [
        ("Correctness", d.correctness),
        ("Distractor quality", d.distractor_quality),
        ("Difficulty alignment", d.difficulty_alignment),
        ("Kazakh language quality", d.kazakh_language_quality),
        ("LaTeX validity", d.latex_validity),
    ]
    if d.figure_relevance is not None:
        rows.append(("Figure relevance", d.figure_relevance))

    for name, score in rows:
        color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
        table.add_row(name, f"[{color}]{score:.1f}[/{color}]")

    table.add_row("─" * 26, "─" * 8)
    oc = "green" if critique.overall_score >= 7 else "yellow" if critique.overall_score >= 5 else "red"
    table.add_row("[bold]Overall[/bold]", f"[{oc}][bold]{critique.overall_score:.1f}[/bold][/{oc}]")

    console.print(table)
    if critique.comments:
        console.print(f"  [dim]{critique.comments[:200]}[/dim]")


def _display_question_preview(q: Question) -> None:
    console.print(Panel(
        f"[bold]{q.question_text[:200]}{'...' if len(q.question_text) > 200 else ''}[/bold]\n\n"
        + "\n".join(
            f"  [{'green' if o.label == q.correct_answer else 'white'}]{o.label}) {o.text[:80]}[/]"
            for o in q.options
        ),
        title=f"Preview — {q.topic} / Level {q.level}",
        border_style="dim",
        expand=False,
    ))


def _display_summary(questions: list[Question]) -> None:
    if len(questions) < 2:
        return
    table = Table(title=f"Summary — {len(questions)} questions generated", header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Topic", min_width=20)
    table.add_column("Level")
    table.add_column("Score", justify="right")
    table.add_column("Attempts", justify="right")
    table.add_column("Pass")

    for i, q in enumerate(questions, 1):
        score_str = f"{q.critic_score:.1f}" if q.critic_score is not None else "—"
        passed = q.critic_feedback.pass_fail if q.critic_feedback else False
        pass_str = "[green]✓[/green]" if passed else "[red]✗[/red]"
        table.add_row(str(i), q.topic[:25], q.level, score_str, str(q.generation_attempts), pass_str)

    console.print(table)


if __name__ == "__main__":
    main()
