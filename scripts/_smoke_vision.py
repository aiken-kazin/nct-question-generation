"""One-off live smoke test: real image-anchored question through the critic.
Verifies vision routing and that scores look sensible. Safe to delete after.
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

from src.config import Config
from src.agents import CriticAgent
from scripts.calibrate_critic import real_to_generated_question


def main() -> None:
    cfg = Config()
    print("Generator model:    ", cfg.model)
    print("Text critic model:  ", cfg.critic_model)
    print("Vision critic model:", cfg.vision_critic_model)
    print()

    data = json.load(open(_REPO_ROOT / "files/mathematics_questions_kz.json", encoding="utf-8"))
    entry = data[26]  # the trapezoid-area image item
    q, level, gt, figure_path = real_to_generated_question(entry, cfg, "math")

    print(f'Item id={entry["id"]}  level={level}  gt={gt}')
    print(f"figure_path resolved → {figure_path}")
    print(f"question_text: {q.question_text[:120]}")
    print()

    critic = CriticAgent(cfg)
    print("Calling critic.evaluate with image attached...")
    fb = critic.evaluate(question=q, level=level, subject="math", figure_path=figure_path)
    print()
    print("=== Critic result ===")
    print(f"  critic_answer:  {fb.critic_answer}   (ground truth: {gt})")
    print(f"  overall_score:  {fb.overall_score}")
    print(f"  pass_fail:      {fb.pass_fail}")
    print(f"  correctness:    {fb.dimensions.correctness}")
    print(f"  comments:       {fb.comments[:300]}")
    print()
    print("Critic solution excerpt:")
    print(fb.critic_solution[:500])


if __name__ == "__main__":
    main()
