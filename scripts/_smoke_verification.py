"""One-off live smoke: tampered question → critic catches contradiction.

Run from repo root: python scripts/_smoke_verification.py
Safe to delete after the demo.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

from src.config import Config
from src.agents import CriticAgent
from src.models import GeneratedQuestion, QuestionOption, VerificationSpec


def main() -> None:
    cfg = Config()
    critic = CriticAgent(cfg)

    # A limit question with the WRONG answer key, but a CORRECT verification.
    # verification.matches_option = "A" but correct_answer = "B".
    # The critic must catch this and clamp correctness to 0.
    q = GeneratedQuestion(
        topic="functions_limits",
        question_text=r"$\lim_{x \to 2}(x^2 - 3x + 2)$ шегінің мәнін табыңыз:",
        options=[
            QuestionOption(label="A", text="0"),
            QuestionOption(label="B", text="2"),
            QuestionOption(label="C", text="4"),
            QuestionOption(label="D", text="1"),
        ],
        correct_answer="B",  # WRONG on purpose
        explanation="(deliberately invalid for the smoke test)",
        latex_formulas=[],
        figure_spec=None,
        verification=VerificationSpec(
            applicable=True,
            code=(
                "from sympy import Symbol, limit\n"
                "x = Symbol('x')\n"
                "print(limit(x**2 - 3*x + 2, x, 2))"
            ),
            expected_output="0",
            matches_option="A",  # honest about the verified answer
        ),
    )

    print("Tampered question: correct_answer=B but verification.matches_option=A")
    print("Expected behavior: critic catches the contradiction, correctness=0.")
    print()
    fb = critic.evaluate(question=q, level="A", subject="math")
    print(f"  correctness:                {fb.dimensions.correctness}")
    print(f"  overall_score:              {fb.overall_score}")
    print(f"  pass_fail:                  {fb.pass_fail}")
    print(f"  verification.passed:        {fb.verification.get('passed')}")
    print(f"  verification.contradicted:  {fb.verification.get('contradicted')}")
    print(f"  self_inconsistency:         {fb.verification.get('self_inconsistency')}")
    print(f"  comments (first 200):       {fb.comments[:200]}")


if __name__ == "__main__":
    main()
