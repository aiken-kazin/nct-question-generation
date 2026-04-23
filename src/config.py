from __future__ import annotations
import json
import os
from pathlib import Path
from string import Template
import yaml

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_text(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_yaml(name: str) -> dict:
    with open(_PROMPTS_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self) -> None:
        self.model: str = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
        self.api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
        self.pass_threshold: float = 6.0
        self.max_retries: int = 3

        self._difficulty: dict = _load_yaml("difficulty.yaml")
        self._topics_math: list[dict] = _load_yaml("topics_math.yaml")["topics"]
        self._topics_kazakh: list[dict] = _load_yaml("topics_kazakh.yaml")["topics"]
        self._figures: dict = _load_yaml("figures.yaml")

        self._gen_math_tpl: str = _load_text("generator_math.md")
        self._gen_kazakh_tpl: str = _load_text("generator_kazakh.md")
        self._critic_solve_tpl: str = _load_text("critic_solve.md")
        self._critic_eval_tpl: str = _load_text("critic_eval.md")

    # ── Topics ──────────────────────────────────────────────────────────────

    def get_topics(self, subject: str) -> list[str]:
        topics = self._topics_math if subject == "math" else self._topics_kazakh
        return [t["id"] for t in topics]

    def get_topic_info(self, subject: str, topic_id: str) -> dict:
        topics = self._topics_math if subject == "math" else self._topics_kazakh
        for t in topics:
            if t["id"] == topic_id:
                return t
        # fallback: return a stub so generation can still proceed
        return {"id": topic_id, "name_en": topic_id, "name_kz": topic_id, "subtopics": [], "keywords": []}

    # ── Difficulty ───────────────────────────────────────────────────────────

    def difficulty_info(self, level: str) -> dict:
        return self._difficulty[level]

    # ── Figure schemas ───────────────────────────────────────────────────────

    def _figure_spec_example(self, topic_id: str, fmt: str) -> str:
        """Return a concrete figure_spec JSON example for the prompt schema."""
        if fmt != "image":
            return "null"
        # Find the first figure type relevant to this topic
        for fig_type, spec in self._figures.items():
            if topic_id in spec.get("use_for", []):
                params = {k: v.get("example") for k, v in spec.get("parameters", {}).items()}
                return json.dumps(
                    {"figure_type": fig_type, "parameters": params, "caption": params.get("caption", "Сурет")},
                    ensure_ascii=False,
                )
        # fallback: triangle with specific measurements
        return json.dumps({
            "figure_type": "triangle",
            "parameters": {
                "vertices": [[0, 0], [6, 0], [0, 8]],
                "labels": ["A", "B", "C"],
                "side_lengths": {"AB": 6, "BC": None, "AC": 8},
                "angles": {"A": 90, "B": None, "C": None},
                "right_angle_vertex": "A",
                "show_height": None,
                "height_foot_label": None,
                "show_incircle": False,
                "incircle_radius_label": None,
                "show_circumcircle": False,
                "caption": "ABC үшбұрышы",
            },
        }, ensure_ascii=False)

    def figure_schema_for_topic(self, topic_id: str) -> str:
        """Return YAML snippet of figure types relevant to this topic."""
        relevant = []
        for fig_type, spec in self._figures.items():
            use_for = spec.get("use_for", [])
            if topic_id in use_for:
                relevant.append((fig_type, spec))
        if not relevant:
            # fallback: return function_graph + coordinate_plane
            relevant = [
                ("function_graph", self._figures.get("function_graph", {})),
                ("coordinate_plane", self._figures.get("coordinate_plane", {})),
            ]
        lines = []
        for fig_type, spec in relevant:
            lines.append(f"### {fig_type}")
            lines.append(f"Description: {spec.get('description', '')}")
            lines.append("Parameters:")
            for param, info in spec.get("parameters", {}).items():
                lines.append(f"  {param}: {info.get('description', '')} (example: {info.get('example', 'null')})")
            lines.append("")
        return "\n".join(lines)

    # ── Prompt rendering ─────────────────────────────────────────────────────

    def render_generator(
        self,
        subject: str,
        level: str,
        topic_id: str,
        fmt: str,
        feedback: str | None = None,
    ) -> str:
        topic = self.get_topic_info(subject, topic_id)
        diff = self.difficulty_info(level)

        subtopics = "\n".join(f"  - {s}" for s in topic.get("subtopics", []))
        keywords = ", ".join(topic.get("keywords", []))
        topic_details = f"Subtopics:\n{subtopics}\nKeywords: {keywords}"

        if fmt == "image":
            fig_schema = self.figure_schema_for_topic(topic_id)
            format_instructions = (
                "A figure IS required. Populate figure_spec — do NOT set it to null.\n\n"
                "## CRITICAL — The question must be FIGURE-DEPENDENT\n\n"
                "The figure is not decoration. The student must look at the figure to solve the question.\n"
                "The figure contains the specific numerical data (side lengths, angles, coordinates, "
                "function values) needed to compute the answer.\n\n"
                "Rules that MUST be followed:\n"
                "1. Question text MUST reference the figure: start with 'Суреттегі...' "
                "('In the figure...') or 'Берілген суреттегі...' ('In the given figure...').\n"
                "2. The figure_spec.parameters MUST include all specific measurement values "
                "shown on the figure (e.g., side_lengths: {AB: 6, BC: 8}, angles: {A: 30, B: 90}).\n"
                "3. Do NOT restate the measurements in the question text — let the figure carry them.\n"
                "4. The question asks to FIND one specific unknown value that is computable "
                "from the labeled data in the figure.\n"
                "5. Good examples of figure-dependent questions:\n"
                "   - 'Find the radius of the inscribed circle in triangle ABC shown in the figure.'\n"
                "   - 'Find the area of the shaded region in the figure.'\n"
                "   - 'Find the angle marked α in the figure.'\n"
                "   - 'Find the x-coordinate of the minimum of the function graphed in the figure.'\n\n"
                "## Available figure types\n\n"
                + fig_schema
            )
        else:
            format_instructions = (
                "Text-only question. Set figure_spec to null. "
                "No image reference needed."
            )

        feedback_block = ""
        if feedback:
            feedback_block = (
                "## Previous Attempt Failed — Critic Feedback\n\n"
                "The previous version of this question was rejected. Address ALL of the following:\n\n"
                f"{feedback}\n\n"
                "Regenerate the question, fixing every issue the critic raised."
            )

        examples_block = ""

        tpl = self._gen_math_tpl if subject == "math" else self._gen_kazakh_tpl
        return Template(tpl).safe_substitute(
            level=level,
            topic_name=topic["name_en"],
            topic_name_kz=topic.get("name_kz", topic["name_en"]),
            topic_details=topic_details,
            difficulty_description=diff["description"].strip(),
            distractor_guidance=diff["distractor_guidance"].strip(),
            format_instructions=format_instructions,
            figure_spec_example=self._figure_spec_example(topic_id, fmt),
            feedback_block=feedback_block,
            examples_block=examples_block,
        )

    def render_critic_solve(self, subject: str, topic_id: str, question_block: str) -> str:
        topic = self.get_topic_info(subject, topic_id)
        return Template(self._critic_solve_tpl).safe_substitute(
            subject=subject,
            topic_name=topic["name_en"],
            topic_name_kz=topic.get("name_kz", topic["name_en"]),
            question_block=question_block,
        )

    def render_critic_eval(
        self,
        subject: str,
        level: str,
        topic_id: str,
        question_block: str,
        correct_answer: str,
        explanation: str,
        critic_solution: str,
        critic_answer: str,
    ) -> str:
        topic = self.get_topic_info(subject, topic_id)
        diff = self.difficulty_info(level)
        return Template(self._critic_eval_tpl).safe_substitute(
            subject=subject,
            level=level,
            level_name=diff["name"],
            topic_name=topic["name_en"],
            topic_name_kz=topic.get("name_kz", topic["name_en"]),
            difficulty_description=diff["description"].strip(),
            question_block=question_block,
            correct_answer=correct_answer,
            explanation=explanation,
            critic_solution=critic_solution,
            critic_answer=critic_answer,
        )
