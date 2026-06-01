"""
Few-shot exemplar bank built from real NTC questions.

Purpose
-------
Inject 1-3 *real* NTC questions into the generator prompt so the LLM mimics the
authentic style, tone, length, and difficulty of published items. This is the
"few-shot retrieval" pattern from MCQG-SRefine (Yao et al. 2025) and QUEST-AI
(Bedi et al. 2025).

Data sources (under files/)
---------------------------
- mathematics_questions_kz.json       — 40 real NTC math items in Kazakh
- kazakh_questions_no_context.json    — 20 standalone Kazakh-language items
- kazakh_questions_with_context.json  —  5 items sharing one reading passage

Math items that reference an image (metadata.context contains a file path) are
excluded from the text-format pool — a text-only LLM cannot reason about them
without vision, so they would mislead the generator.

The "kazakh_with_context" file is loaded but currently NOT served as a few-shot
exemplar: those items only make sense as part of a 5-question block tied to a
shared passage, which the single-question pipeline does not yet support.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Default location relative to repo root.
_DEFAULT_FILES_DIR = Path(__file__).resolve().parent.parent / "files"


def _normalize_level(raw: str) -> str:
    """Real datasets mix Cyrillic 'А' and Latin 'A'. Always return Latin A/B/C."""
    c = (raw or "").strip().upper()
    if c in ("A", "А"):  # Latin A + Cyrillic А (U+0410)
        return "A"
    if c in ("B", "В"):  # Cyrillic В (U+0412) looks like Latin B
        return "B"
    if c in ("C", "С"):  # Cyrillic С (U+0421) looks like Latin C
        return "C"
    return ""


def _load_json_safe(path: Path) -> list[dict]:
    """Load a JSON file; return [] if missing or unreadable rather than crashing.

    Generation should never fail just because a reference dataset isn't on disk.
    """
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


class ExemplarBank:
    """In-memory bank of real NTC items, queryable by subject + topic + level.

    Loaded once at Config-construction time; reads are cheap thereafter.
    """

    def __init__(self, files_dir: Path | None = None) -> None:
        files_dir = files_dir or _DEFAULT_FILES_DIR

        raw_math = _load_json_safe(files_dir / "mathematics_questions_kz.json")
        raw_kz = _load_json_safe(files_dir / "kazakh_questions_no_context.json")
        raw_kz_ctx = _load_json_safe(files_dir / "kazakh_questions_with_context.json")

        # Math: drop items that reference an external image. The few-shot
        # pool is consumed by text-only generation; image-anchored examples
        # would mislead the model into describing a picture it cannot see.
        self._math: list[dict] = [
            q for q in raw_math
            if not (q.get("metadata", {}).get("context") or "").strip()
        ]

        # Kazakh language/literature: take only standalone items for now.
        # Context-block items need a different orchestration layer (5 questions
        # sharing one passage) which isn't implemented yet — see README.
        self._kazakh: list[dict] = list(raw_kz)
        self._kazakh_with_context: list[dict] = list(raw_kz_ctx)

    # ── Public API ──────────────────────────────────────────────────────────

    def has_data(self, subject: str) -> bool:
        return bool(self._pool_for(subject))

    def select(
        self,
        subject: str,
        topic_kz: str,
        level: str,
        k: int = 2,
        seed: int | None = None,
    ) -> list[dict]:
        """Pick up to k exemplars best matching (topic_kz, level).

        Ranking is deliberately simple:
          * +3 if either string contains the other (case-insensitive)
          * +1 if normalised difficulty matches the requested level
          * tie-break: shuffled by seed for variety across runs

        We avoid embedding/cosine because the bank is tiny (40 + 20 items)
        and we don't want a heavy dependency just for ranking few-shot picks.
        """
        pool = self._pool_for(subject)
        if not pool:
            return []

        rng = random.Random(seed)
        target_topic = (topic_kz or "").strip().lower()
        target_level = (level or "").strip().upper()

        scored: list[tuple[float, dict]] = []
        for q in pool:
            meta = q.get("metadata", {})
            q_topic = (meta.get("topic") or "").lower()
            q_level = _normalize_level(meta.get("difficulty", ""))

            score = 0.0
            if target_topic and q_topic:
                if target_topic in q_topic or q_topic in target_topic:
                    score += 3.0
            if target_level and q_level == target_level:
                score += 1.0
            # tiny jitter so identical scores rotate across runs
            score += rng.random() * 0.01
            scored.append((score, q))

        scored.sort(key=lambda x: -x[0])
        return [q for _, q in scored[:k]]

    def format_for_prompt(self, exemplars: list[dict]) -> str:
        """Render a markdown block suitable for direct prompt injection.

        Real NTC items don't carry explanations, so we only show stem + options
        + correct-answer letter. The model is told this is the *target style*,
        not the target content — it should never copy these questions verbatim.
        """
        if not exemplars:
            return ""

        lines: list[str] = [
            "## Reference Examples — Real Published NTC Items",
            "",
            "Below are authentic NTC test items. Use them as a *style* and *quality* reference:",
            "match the tone, sentence length, option style, and difficulty calibration.",
            "Do NOT copy any of these questions verbatim — generate a NEW question on the assigned topic.",
            "",
        ]
        for i, q in enumerate(exemplars, 1):
            meta = q.get("metadata", {})
            lines.append(
                f"### Example {i} — Topic: {meta.get('topic', 'N/A')} | "
                f"Level: {meta.get('difficulty', 'N/A')}"
            )
            lines.append("")
            lines.append(f"**Stem:** {q.get('question_text', '').strip()}")
            lines.append("")
            opts = q.get("options", {})
            if isinstance(opts, dict):
                for label, text in opts.items():
                    lines.append(f"- **{label})** {text}")
            elif isinstance(opts, list):
                for o in opts:
                    if isinstance(o, dict):
                        lines.append(f"- **{o.get('label','?')})** {o.get('text','')}")
            lines.append("")
            lines.append(f"**Correct answer:** {meta.get('correct_answer', 'N/A')}")
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    # ── Internals ───────────────────────────────────────────────────────────

    def _pool_for(self, subject: str) -> list[dict]:
        if subject == "math":
            return self._math
        if subject == "kazakh":
            return self._kazakh
        return []

    # Convenience accessors for callers that want raw items (e.g. critic
    # calibration script).
    @property
    def math_items(self) -> list[dict]:
        return list(self._math)

    @property
    def kazakh_items(self) -> list[dict]:
        return list(self._kazakh)

    @property
    def kazakh_with_context_items(self) -> list[dict]:
        return list(self._kazakh_with_context)
