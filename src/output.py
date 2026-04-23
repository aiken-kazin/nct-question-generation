from __future__ import annotations
import json
import os
from pathlib import Path

from .models import Question


def save_question(question: Question, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    q_dir = output_dir / question.subject / "questions" / f"level_{question.level}"
    q_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{question.timestamp[:19].replace(':', '-').replace('T', '_')}_{question.id[:8]}"
    json_path = q_dir / f"{stem}.json"
    md_path = q_dir / f"{stem}.md"

    # JSON keeps the absolute figure path for programmatic use
    json_path.write_text(
        json.dumps(question.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown uses a relative path so preview works regardless of machine
    fig_rel: str | None = None
    if question.figure_path:
        try:
            fig_rel = os.path.relpath(
                Path(question.figure_path).resolve(),
                md_path.parent,
            )
        except ValueError:
            # Windows cross-drive: fall back to absolute
            fig_rel = question.figure_path

    md_path.write_text(_to_markdown(question, fig_rel), encoding="utf-8")

    return {"json": json_path, "markdown": md_path}


def _to_markdown(q: Question, fig_rel_path: str | None = None) -> str:
    lines: list[str] = []

    lines += [
        f"# Exam Question — {q.subject.upper()} / Level {q.level}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **ID** | `{q.id}` |",
        f"| **Subject** | {q.subject} |",
        f"| **Level** | {q.level} |",
        f"| **Format** | {q.format} |",
        f"| **Topic** | {q.topic} |",
        f"| **Generated** | {q.timestamp} |",
        f"| **Attempts** | {q.generation_attempts} |",
        "",
    ]

    if fig_rel_path:
        # Use forward slashes so the path works in all markdown viewers
        fig_display = fig_rel_path.replace("\\", "/")
        lines += [
            "## Figure",
            "",
            f"![{q.topic}]({fig_display})",
            "",
        ]

    lines += [
        "## Question",
        "",
        q.question_text,
        "",
        "## Options",
        "",
    ]
    for opt in q.options:
        marker = " ✓" if opt.label == q.correct_answer else ""
        lines.append(f"- **{opt.label})** {opt.text}{marker}")
    lines.append("")

    lines += [
        "## Correct Answer",
        "",
        f"**{q.correct_answer}**",
        "",
        "## Explanation",
        "",
        q.explanation,
        "",
    ]

    if q.latex_formulas:
        lines += ["## Key Formulas", ""]
        for formula in q.latex_formulas:
            lines.append(f"$$\n{formula}\n$$")
            lines.append("")

    if q.critic_feedback:
        fb = q.critic_feedback
        lines += [
            "## Critic Evaluation",
            "",
            f"**Overall score:** {q.critic_score:.1f}/10 — {'PASS' if fb.pass_fail else 'FAIL'}",
            "",
            "| Dimension | Score |",
            "|---|---|",
            f"| Correctness | {fb.dimensions.correctness}/10 |",
            f"| Distractor quality | {fb.dimensions.distractor_quality}/10 |",
            f"| Difficulty alignment | {fb.dimensions.difficulty_alignment}/10 |",
            f"| Kazakh language | {fb.dimensions.kazakh_language_quality}/10 |",
            f"| LaTeX validity | {fb.dimensions.latex_validity}/10 |",
        ]
        if fb.dimensions.figure_relevance is not None:
            lines.append(f"| Figure relevance | {fb.dimensions.figure_relevance}/10 |")
        lines += [
            "",
            f"**Comments:** {fb.comments}",
            "",
        ]
        if fb.improvement_suggestions:
            lines += [f"**Suggestions:** {fb.improvement_suggestions}", ""]

        lines += [
            "### Critic's Independent Solution",
            "",
            fb.critic_solution,
            f"\n**Critic's answer:** {fb.critic_answer}",
            "",
        ]

    return "\n".join(lines)
