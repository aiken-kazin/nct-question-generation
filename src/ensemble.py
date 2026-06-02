"""
EnsembleCriticAgent — run N critic models in parallel on the same question
and aggregate their verdicts.

Why this matters (paper-claim level)
------------------------------------
The literature (Yao et al. 2025; Byun & Choi 2025) repeatedly flags LLM-as-judge
as unreliable for fine-grained quality dimensions. The standard answer is human
experts, which are expensive. Our middle ground:

  * Run K critics from DIFFERENT vendors (e.g. OpenAI, Anthropic, Alibaba).
  * Their AGREEMENT on `critic_answer` is a free uncertainty signal — when all
    three pick the same option, confidence is high; when they split, the item
    needs human eyes.
  * Per-dimension scores are averaged across critics, smoothing out single-model
    biases (a strict critic + a lenient critic average to something closer to
    "real").
  * Pairwise Cohen's κ on the K-critic answer matrix gives us *inter-rater
    reliability among models* — a quantity that, to our knowledge, has not been
    reported as a calibration metric for MCQ critic agents in the literature.

What this module owns
---------------------
  * Construction: takes a list of model IDs, builds K CriticAgent instances
    (each pinned to one model — see CriticAgent's *_model_override args).
  * Sequential or threaded dispatch (threaded by default).
  * Aggregation policy: majority-vote answer, mean per-dimension scores,
    strict/majority pass rule.
  * Per-critic results preserved on the returned EnsembleCriticFeedback so the
    pipeline can persist them and downstream stats can compute κ.

What this module does NOT own
-----------------------------
  * CDI math — that lives in src/cdi.py.
  * Cost accounting — caller's responsibility.
  * Retry / single-critic mode — caller uses CriticAgent directly when ensemble
    is off.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Sequence

from .agents import CriticAgent
from .config import Config
from .models import CriticFeedback, DimensionScores, GeneratedQuestion


@dataclass
class PerCriticResult:
    """One critic's verdict, with model identity attached."""

    model_id: str
    feedback: CriticFeedback | None  # None if the critic call failed
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "error": self.error,
            "feedback": self.feedback.model_dump() if self.feedback else None,
        }


@dataclass
class EnsembleCriticFeedback:
    """Aggregated result from N critics.

    Shaped to be a drop-in replacement for CriticFeedback wherever the pipeline
    expects one — but with extra fields for downstream analysis.
    """

    aggregated: CriticFeedback   # the synthesized verdict the pipeline uses
    per_critic: list[PerCriticResult] = field(default_factory=list)
    answer_agreement: float = 0.0   # fraction of critics that picked the majority answer
    unanimous: bool = False
    n_critics_responded: int = 0
    n_critics_failed: int = 0

    def to_dict(self) -> dict:
        return {
            "aggregated": self.aggregated.model_dump(),
            "per_critic": [p.to_dict() for p in self.per_critic],
            "answer_agreement": round(self.answer_agreement, 3),
            "unanimous": self.unanimous,
            "n_critics_responded": self.n_critics_responded,
            "n_critics_failed": self.n_critics_failed,
        }


# ── Aggregation helpers ─────────────────────────────────────────────────────


def _majority_vote(answers: list[str]) -> tuple[str, float]:
    """Pick the most common A/B/C/D answer. Ties broken by alphabetic order.
    Returns (winning_label, fraction_agreeing).
    """
    answers = [a.strip().upper() for a in answers if a]
    if not answers:
        return ("A", 0.0)
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    max_n = max(counts.values())
    winners = sorted(lbl for lbl, n in counts.items() if n == max_n)
    return (winners[0], max_n / len(answers))


def _mean_or_none(xs: list[float | None]) -> float | None:
    xs2 = [x for x in xs if x is not None]
    return round(statistics.fmean(xs2), 3) if xs2 else None


def _aggregate_dimensions(per_critic: list[PerCriticResult]) -> DimensionScores:
    """Mean per dimension, ignoring failed critics."""
    fbs = [p.feedback for p in per_critic if p.feedback is not None]
    if not fbs:
        # Fall back to neutral 5s so the pipeline can continue with a clear fail.
        return DimensionScores(
            correctness=0, distractor_quality=0, difficulty_alignment=0,
            kazakh_language_quality=0, latex_validity=0, figure_relevance=None,
        )
    return DimensionScores(
        correctness=_mean_or_none([f.dimensions.correctness for f in fbs]) or 0.0,
        distractor_quality=_mean_or_none([f.dimensions.distractor_quality for f in fbs]) or 0.0,
        difficulty_alignment=_mean_or_none([f.dimensions.difficulty_alignment for f in fbs]) or 0.0,
        kazakh_language_quality=_mean_or_none([f.dimensions.kazakh_language_quality for f in fbs]) or 0.0,
        latex_validity=_mean_or_none([f.dimensions.latex_validity for f in fbs]) or 0.0,
        figure_relevance=_mean_or_none([f.dimensions.figure_relevance for f in fbs]),
    )


def _pass_decision(per_critic: list[PerCriticResult], strict: bool, threshold: float) -> bool:
    """Pass rule.

    strict=True  → all responding critics must pass individually
    strict=False → at least a majority of responding critics must pass
    """
    fbs = [p.feedback for p in per_critic if p.feedback is not None]
    if not fbs:
        return False
    passes = sum(1 for f in fbs if f.overall_score >= threshold and f.pass_fail)
    if strict:
        return passes == len(fbs)
    return passes > len(fbs) / 2


# ── Main agent ──────────────────────────────────────────────────────────────


class EnsembleCriticAgent:
    """Run K CriticAgent instances and aggregate."""

    def __init__(
        self,
        config: Config,
        model_ids: Sequence[str],
        strict_pass: bool = False,
        pass_threshold: float | None = None,
        parallel: bool = True,
    ) -> None:
        if len(model_ids) < 2:
            raise ValueError("EnsembleCriticAgent requires at least 2 model IDs")
        # De-duplicate while preserving order. Repeating a model in the list is
        # almost always a user mistake and would inflate apparent agreement.
        seen: set[str] = set()
        deduped: list[str] = []
        for m in model_ids:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)
        if len(deduped) < 2:
            raise ValueError("EnsembleCriticAgent requires ≥ 2 distinct models")

        self._cfg = config
        self._strict = strict_pass
        self._threshold = pass_threshold if pass_threshold is not None else config.pass_threshold
        self._parallel = parallel

        # Each agent is pinned to a single model — same vision model for all
        # (we don't have per-critic vision overrides yet; defer if needed).
        self._agents: list[tuple[str, CriticAgent]] = [
            (mid, CriticAgent(config, text_model_override=mid, vision_model_override=mid))
            for mid in deduped
        ]

    @property
    def model_ids(self) -> list[str]:
        return [mid for mid, _ in self._agents]

    def evaluate(
        self,
        question: GeneratedQuestion,
        level: str,
        subject: str,
        figure_path: str | None = None,
    ) -> EnsembleCriticFeedback:
        per_critic: list[PerCriticResult] = self._dispatch(question, level, subject, figure_path)

        # Aggregate
        answers = [p.feedback.critic_answer for p in per_critic if p.feedback is not None]
        majority_answer, agreement = _majority_vote(answers)
        unanimous = bool(answers) and all(a.strip().upper() == majority_answer for a in answers)

        dims = _aggregate_dimensions(per_critic)
        responded = [p for p in per_critic if p.feedback is not None]
        overall = _mean_or_none([p.feedback.overall_score for p in responded]) or 0.0

        # Build the aggregated CriticFeedback that the rest of the pipeline
        # consumes. critic_solution + comments are merged across critics for
        # transparency in the saved JSON.
        aggregated = CriticFeedback(
            critic_solution=_merge_solutions(per_critic),
            critic_answer=majority_answer,
            dimensions=dims,
            overall_score=overall,
            pass_fail=_pass_decision(per_critic, self._strict, self._threshold),
            comments=_merge_comments(per_critic),
            improvement_suggestions=_merge_suggestions(per_critic),
        )

        # Symbolic verification (math) is deterministic and identical across
        # critics; surface it on the aggregated verdict so it is not lost when
        # the pipeline saves only the aggregated feedback.
        for p in per_critic:
            if p.feedback is not None and p.feedback.verification is not None:
                aggregated.verification = p.feedback.verification
                break

        return EnsembleCriticFeedback(
            aggregated=aggregated,
            per_critic=per_critic,
            answer_agreement=agreement,
            unanimous=unanimous,
            n_critics_responded=len(responded),
            n_critics_failed=len(per_critic) - len(responded),
        )

    # ── Dispatch ────────────────────────────────────────────────────────────

    def _dispatch(
        self, question: GeneratedQuestion, level: str, subject: str,
        figure_path: str | None,
    ) -> list[PerCriticResult]:
        def _one(model_id: str, agent: CriticAgent) -> PerCriticResult:
            try:
                fb = agent.evaluate(
                    question=question, level=level, subject=subject,
                    figure_path=figure_path,
                )
                return PerCriticResult(model_id=model_id, feedback=fb)
            except Exception as exc:  # one critic failing should not abort
                return PerCriticResult(model_id=model_id, feedback=None, error=str(exc))

        if not self._parallel:
            return [_one(mid, ag) for mid, ag in self._agents]

        # Thread-pool dispatch. OpenAI client is thread-safe; each request
        # is independent. Order of results follows the input model list.
        results_by_model: dict[str, PerCriticResult] = {}
        with ThreadPoolExecutor(max_workers=len(self._agents)) as ex:
            futures = {ex.submit(_one, mid, ag): mid for mid, ag in self._agents}
            for fut in as_completed(futures):
                res = fut.result()
                results_by_model[res.model_id] = res
        return [results_by_model[mid] for mid, _ in self._agents]


# ── Cosmetics: merge text from multiple critics for the audit trail ─────────


def _merge_solutions(per_critic: list[PerCriticResult]) -> str:
    chunks = []
    for p in per_critic:
        if p.feedback is None:
            continue
        chunks.append(f"[{p.model_id}] answered {p.feedback.critic_answer}: {p.feedback.critic_solution[:400]}")
    return "\n\n".join(chunks)


def _merge_comments(per_critic: list[PerCriticResult]) -> str:
    chunks = []
    for p in per_critic:
        if p.feedback and p.feedback.comments:
            chunks.append(f"[{p.model_id}] {p.feedback.comments}")
    return " | ".join(chunks)


def _merge_suggestions(per_critic: list[PerCriticResult]) -> str | None:
    chunks = []
    for p in per_critic:
        if p.feedback and p.feedback.improvement_suggestions:
            chunks.append(f"[{p.model_id}] {p.feedback.improvement_suggestions}")
    return " | ".join(chunks) if chunks else None
