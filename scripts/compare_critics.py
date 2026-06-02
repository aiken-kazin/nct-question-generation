#!/usr/bin/env python3
"""
Critic model selection: aggregate calibration summaries and rank critics.

Reads every `*_summary.json` produced by scripts/calibrate_critic.py under a
directory, aggregates per-model metrics across subjects, computes a transparent
selection score, and writes a leaderboard (Markdown + JSON + LaTeX) for the
paper. No network, no LLM calls — pure post-hoc analysis.

Selection rationale (all components reported so the paper can justify the pick):
  * Competence   — critic_answer_accuracy on REAL items (does the critic
                   independently arrive at the ground-truth answer?). Primary.
  * Recognition  — pass_rate on REAL items (does it accept genuine questions?).
  * Discrimination — targeted CDI gaps:
                   wrong_key → correctness, weak_distractors → distractor_quality.
                   A useful critic scores degraded items materially lower.
  * Stability    — untargeted dimensions should NOT move under degradation;
                   large unexpected swings are penalized.

Usage:
    python scripts/compare_critics.py
    python scripts/compare_critics.py --dir output/_critic_validation --latex paper/critic_selection.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# (variant, dimension) pairs the degradation is DESIGNED to depress.
TARGETED = [("wrong_key", "correctness"), ("weak_distractors", "distractor_quality")]
# Dimensions a degradation should leave roughly unchanged (stability check).
UNTARGETED = {
    "wrong_key": ["distractor_quality", "difficulty_alignment", "kazakh_language_quality", "latex_validity"],
    "weak_distractors": ["difficulty_alignment", "latex_validity"],
}

# Composite weights. Competence dominates: a critic that can't solve the item
# can't be trusted to grade it. Discrimination is the paper's headline metric.
W_COMPETENCE = 0.40
W_RECOGNITION = 0.15
W_DISCRIMINATION = 0.35
W_STABILITY = 0.10


def _wmean(pairs: list[tuple[float, int]]) -> float | None:
    """Weighted mean of (value, weight); ignores None values."""
    num = sum(v * w for v, w in pairs if v is not None and w)
    den = sum(w for v, w in pairs if v is not None and w)
    return num / den if den else None


def load_summaries(directory: Path) -> list[dict]:
    out = []
    for p in sorted(directory.glob("*_summary.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"warn: skipping {p.name}: {exc}", file=sys.stderr)
    return out


def aggregate(summaries: list[dict]) -> dict[str, dict]:
    """Group by model_id; aggregate metrics across subjects (n-weighted)."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_model[s["model_id"]].append(s)

    agg: dict[str, dict] = {}
    for model_id, runs in by_model.items():
        subjects = sorted({r["subject"] for r in runs})
        real_acc, real_pass, real_overall = [], [], []
        n_real_total = 0
        n_failures = 0
        targeted_gaps: dict[str, list[tuple[float, int]]] = {f"{v}:{d}": [] for v, d in TARGETED}
        untargeted_abs: list[tuple[float, int]] = []
        wilcoxon_sig = {f"{v}:{d}": [] for v, d in TARGETED}

        for r in runs:
            real = r["summary"].get("real", {})
            n = real.get("n_scored") or 0
            n_real_total += real.get("n", 0)
            if real.get("critic_answer_accuracy") is not None:
                real_acc.append((real["critic_answer_accuracy"], n))
            if real.get("pass_rate") is not None:
                real_pass.append((real["pass_rate"], n))
            if real.get("mean_overall") is not None:
                real_overall.append((real["mean_overall"], n))

            cdi = r.get("cdi") or {}
            n_failures += (cdi.get("summary", {}) or {}).get("n_critic_failures", 0)
            for variant, dim in TARGETED:
                cell = (cdi.get(variant) or {}).get(dim)
                if cell:
                    targeted_gaps[f"{variant}:{dim}"].append((cell["cdi_gap"], cell["n_pairs"]))
                    p = cell.get("wilcoxon_p")
                    if p is not None:
                        wilcoxon_sig[f"{variant}:{dim}"].append(p < 0.05)
            for variant, dims in UNTARGETED.items():
                for dim in dims:
                    cell = (cdi.get(variant) or {}).get(dim)
                    if cell:
                        untargeted_abs.append((abs(cell["cdi_gap"]), cell["n_pairs"]))

        competence = _wmean(real_acc) or 0.0
        recognition = _wmean(real_pass) or 0.0
        gap_means = {k: (_wmean(v) or 0.0) for k, v in targeted_gaps.items()}
        discrimination = sum(gap_means.values()) / len(gap_means) if gap_means else 0.0
        instability = _wmean(untargeted_abs) or 0.0

        # Composite on a 0-10 scale. Discrimination/stability are already on a
        # 0-10 point scale; competence/recognition are rates → scale to 10.
        score = (
            W_COMPETENCE * (competence * 10)
            + W_RECOGNITION * (recognition * 10)
            + W_DISCRIMINATION * min(discrimination, 10.0)
            - W_STABILITY * min(instability, 10.0)
        )

        agg[model_id] = {
            "model_id": model_id,
            "model_slug": runs[0]["model_slug"],
            "subjects": subjects,
            "n_real_total": n_real_total,
            "n_critic_failures": n_failures,
            "competence_real_accuracy": round(competence, 3),
            "recognition_real_pass_rate": round(recognition, 3),
            "real_mean_overall": round(_wmean(real_overall) or 0.0, 2),
            "gap_wrong_key_correctness": round(gap_means.get("wrong_key:correctness", 0.0), 2),
            "gap_weak_distractors_distractor_quality": round(
                gap_means.get("weak_distractors:distractor_quality", 0.0), 2
            ),
            "discrimination_mean_gap": round(discrimination, 2),
            "instability_mean_abs_gap": round(instability, 2),
            "wilcoxon_significant": {
                k: (all(v) if v else None) for k, v in wilcoxon_sig.items()
            },
            "selection_score": round(score, 3),
        }
    return agg


def format_markdown(agg: dict[str, dict]) -> str:
    ranked = sorted(agg.values(), key=lambda d: d["selection_score"], reverse=True)
    lines = ["# Critic model selection\n"]
    if not ranked:
        return "_(no summaries found)_\n"
    winner = ranked[0]
    lines.append(f"**Selected critic: `{winner['model_id']}`** "
                 f"(score {winner['selection_score']:.2f})\n")
    lines.append("| Rank | Model | Score | Real acc. | Real pass | "
                 "Gap wrong_key→correct | Gap weak_distr→distr_q | Instability | Failures |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, d in enumerate(ranked, 1):
        lines.append(
            f"| {i} | `{d['model_id']}` | {d['selection_score']:.2f} | "
            f"{d['competence_real_accuracy']:.2f} | {d['recognition_real_pass_rate']:.2f} | "
            f"{d['gap_wrong_key_correctness']:+.2f} | "
            f"{d['gap_weak_distractors_distractor_quality']:+.2f} | "
            f"{d['instability_mean_abs_gap']:.2f} | {d['n_critic_failures']} |"
        )
    lines.append("")
    lines.append("Score = "
                 f"{W_COMPETENCE}·(real accuracy) + {W_RECOGNITION}·(real pass rate) + "
                 f"{W_DISCRIMINATION}·(mean targeted CDI gap) − {W_STABILITY}·(instability), "
                 "rates scaled ×10.")
    return "\n".join(lines) + "\n"


def format_latex(agg: dict[str, dict]) -> str:
    ranked = sorted(agg.values(), key=lambda d: d["selection_score"], reverse=True)
    parts = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Critic model selection. Competence = independent answer "
        "accuracy on real items; gaps are Critic Discrimination Index values "
        "on targeted dimensions.}",
        "\\label{tab:critic-selection}",
        "\\begin{tabular}{lrrrrr}", "\\hline",
        "Critic & Score & Real acc. & Real pass & "
        "CDI$_\\text{wk}$ & CDI$_\\text{wd}$ \\\\", "\\hline",
    ]
    for d in ranked:
        slug = d["model_slug"].replace("_", r"\_")
        parts.append(
            f"{slug} & {d['selection_score']:.2f} & "
            f"{d['competence_real_accuracy']:.2f} & {d['recognition_real_pass_rate']:.2f} & "
            f"{d['gap_wrong_key_correctness']:+.2f} & "
            f"{d['gap_weak_distractors_distractor_quality']:+.2f} \\\\"
        )
    parts += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(parts) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("output/_critic_validation"))
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument("--latex", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.dir)
    if not summaries:
        print(f"error: no *_summary.json under {args.dir}", file=sys.stderr)
        sys.exit(2)
    agg = aggregate(summaries)
    md = format_markdown(agg)
    print(md)

    ranked = sorted(agg.values(), key=lambda d: d["selection_score"], reverse=True)
    winner = ranked[0]["model_id"] if ranked else None

    md_path = args.markdown or (args.dir / "comparison.md")
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {md_path}", file=sys.stderr)

    json_path = args.json or (args.dir / "comparison.json")
    json_path.write_text(
        json.dumps({"winner": winner, "models": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}", file=sys.stderr)

    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(format_latex(agg), encoding="utf-8")
        print(f"Wrote {args.latex}", file=sys.stderr)


if __name__ == "__main__":
    main()
