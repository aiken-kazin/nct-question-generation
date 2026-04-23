from __future__ import annotations
import json
import re
import textwrap
from typing import Any

from openai import OpenAI

from .config import Config
from .models import CriticFeedback, DimensionScores, GeneratedQuestion, QuestionOption, FigureSpec

_BASE_URL = "https://openrouter.ai/api/v1"


def _extract_json(text: str) -> str:
    """Strip markdown code fences if the model wraps JSON in them."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def _parse_json(text: str) -> dict:
    raw = _extract_json(text)
    return json.loads(raw)


class GeneratorAgent:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=_BASE_URL,
        )

    def generate(
        self,
        subject: str,
        level: str,
        fmt: str,
        topic: str,
        feedback: str | None = None,
    ) -> GeneratedQuestion:
        system_prompt = self._cfg.render_generator(
            subject=subject,
            level=level,
            topic_id=topic,
            fmt=fmt,
            feedback=feedback,
        )

        if feedback and fmt == "image":
            user_msg = (
                f"Generate a level-{level} {subject} MCQ question on the topic '{topic}'. "
                "Format: image. "
                "MANDATORY: figure_spec MUST be a fully populated object — NOT null. "
                "Copy the structure from the example in the schema and fill it with real values. "
                "Return only the JSON object as specified."
            )
        else:
            user_msg = (
                f"Generate a level-{level} {subject} MCQ question "
                f"on the topic '{topic}'. "
                f"Format: {fmt}. "
                "Return only the JSON object as specified."
            )

        response = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )

        content = response.choices[0].message.content or ""
        data = _parse_json(content)
        return _build_generated_question(data)


class CriticAgent:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=_BASE_URL,
        )

    def evaluate(
        self,
        question: GeneratedQuestion,
        level: str,
        subject: str,
        figure_path: str | None = None,
    ) -> CriticFeedback:
        question_block = _format_question_block(question, include_answer=False, figure_path=figure_path)

        # ── Step 1: Independent solve ─────────────────────────────────────
        solve_prompt = self._cfg.render_critic_solve(
            subject=subject,
            topic_id=question.topic,
            question_block=question_block,
        )
        solve_resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=[
                {"role": "system", "content": solve_prompt},
                {"role": "user", "content": "Solve the question independently and return your JSON response."},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        solve_content = solve_resp.choices[0].message.content or "{}"
        solve_data = _parse_json(solve_content)
        critic_solution = solve_data.get("critic_solution", "")
        critic_answer = solve_data.get("critic_answer", "")

        # ── Step 2: Full evaluation ───────────────────────────────────────
        question_block_full = _format_question_block(question, include_answer=True, figure_path=figure_path)
        eval_prompt = self._cfg.render_critic_eval(
            subject=subject,
            level=level,
            topic_id=question.topic,
            question_block=question_block_full,
            correct_answer=question.correct_answer,
            explanation=question.explanation,
            critic_solution=critic_solution,
            critic_answer=critic_answer,
        )
        eval_resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=[
                {"role": "system", "content": eval_prompt},
                {"role": "user", "content": "Evaluate the question and return your JSON scoring."},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        eval_content = eval_resp.choices[0].message.content or "{}"
        eval_data = _parse_json(eval_content)

        return _build_critic_feedback(eval_data, critic_solution, critic_answer)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_question_block(
    q: GeneratedQuestion,
    include_answer: bool,
    figure_path: str | None,
) -> str:
    lines = [
        f"Topic: {q.topic}",
        "",
        "Question:",
        q.question_text,
        "",
        "Options:",
    ]
    for opt in q.options:
        lines.append(f"  {opt.label}) {opt.text}")

    if figure_path:
        lines += ["", f"[Figure: {figure_path}]"]

    if include_answer:
        lines += ["", f"Provided correct answer: {q.correct_answer}"]

    return "\n".join(lines)


def _build_generated_question(data: dict[str, Any]) -> GeneratedQuestion:
    raw_opts = data.get("options", [])
    options = []
    for o in raw_opts:
        if isinstance(o, dict):
            options.append(QuestionOption(label=str(o.get("label", "")), text=str(o.get("text", ""))))

    fig_data = data.get("figure_spec")
    figure_spec = None
    if isinstance(fig_data, dict) and fig_data.get("figure_type"):
        figure_spec = FigureSpec(
            figure_type=str(fig_data.get("figure_type", "")),
            parameters=fig_data.get("parameters", {}),
            caption=str(fig_data.get("caption", "")),
        )

    return GeneratedQuestion(
        topic=str(data.get("topic", "unknown")),
        question_text=str(data.get("question_text", "")),
        options=options,
        correct_answer=str(data.get("correct_answer", "A")).upper(),
        explanation=str(data.get("explanation", "")),
        latex_formulas=[str(f) for f in data.get("latex_formulas", [])],
        figure_spec=figure_spec,
    )


def _build_critic_feedback(
    data: dict[str, Any],
    critic_solution: str,
    critic_answer: str,
) -> CriticFeedback:
    dims_raw = data.get("dimensions", {})
    dims = DimensionScores(
        correctness=float(dims_raw.get("correctness", 5)),
        distractor_quality=float(dims_raw.get("distractor_quality", 5)),
        difficulty_alignment=float(dims_raw.get("difficulty_alignment", 5)),
        kazakh_language_quality=float(dims_raw.get("kazakh_language_quality", 5)),
        latex_validity=float(dims_raw.get("latex_validity", 5)),
        figure_relevance=(
            float(dims_raw["figure_relevance"])
            if dims_raw.get("figure_relevance") is not None
            else None
        ),
    )

    overall = float(data.get("overall_score", _compute_weighted(dims)))
    pass_fail = bool(data.get("pass_fail", overall >= 6.0))

    return CriticFeedback(
        critic_solution=critic_solution,
        critic_answer=critic_answer,
        dimensions=dims,
        overall_score=overall,
        pass_fail=pass_fail,
        comments=str(data.get("comments", "")),
        improvement_suggestions=data.get("improvement_suggestions"),
    )


def _compute_weighted(d: DimensionScores) -> float:
    weights = {
        "correctness": 3,
        "distractor_quality": 2,
        "difficulty_alignment": 2,
        "kazakh_language_quality": 2,
        "latex_validity": 1,
    }
    total_w = sum(weights.values())
    score = sum(getattr(d, k) * w for k, w in weights.items())
    if d.figure_relevance is not None:
        score += d.figure_relevance * 1
        total_w += 1
    return round(score / total_w, 2)
