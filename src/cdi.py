"""
Critic Discrimination Index (CDI) — a metric for validating an automated
question critic *without* expert grading.

Definition
----------
Given a fixed critic and a set of N real (human-authored, published) MCQ items,
plus a *degraded* version of each item (e.g. wrong-answer-key relabel, weak
distractor substitution), the Critic Discrimination Index for dimension d is:

    CDI_d  =  mean( critic_score_d on real items )
              −  mean( critic_score_d on degraded items )

A well-calibrated critic produces:
  • large positive CDI on dimensions that *should* respond to the degradation
    (e.g. wrong-key degradation → big drop in correctness)
  • near-zero CDI on dimensions the degradation does not target
    (e.g. wrong-key degradation should NOT change kazakh_language_quality)

Reported alongside:
  • Wilcoxon signed-rank p-value (paired non-parametric test of "scores differ")
  • Cohen's d for paired samples (effect size; >0.8 is large)

Why a new metric
----------------
The MCQ-generation literature evaluates critics by (a) inter-rater reliability
against humans, which is expensive, or (b) LLM-as-judge on a fixed dataset,
which is reflexive. We can't find a published metric that validates a critic
agent by injecting *synthetic* quality faults and measuring detection. This
file formalizes that pattern so it can be cited.

Pairing
-------
Items are paired by `idx` (the position in the source dataset). The Wilcoxon
test requires same-N paired samples; we filter to indexes present in BOTH
groups before computing.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from scipy import stats  # type: ignore[import-untyped]

# Dimensions we compute CDI for. Matches DimensionScores in src/models.py.
DIMENSIONS = (
    "correctness",
    "distractor_quality",
    "difficulty_alignment",
    "kazakh_language_quality",
    "latex_validity",
)

# The variant labels emitted by scripts/calibrate_critic.py
VARIANT_REAL = "real"
VARIANT_WRONG_KEY = "wrong_key"
VARIANT_WEAK_DISTRACTORS = "weak_distractors"


@dataclass
class DimensionCDI:
    """CDI for a single dimension comparing one variant against `real`."""

    dimension: str
    variant: str
    n_pairs: int
    mean_real: float
    mean_degraded: float
    sd_real: float
    sd_degraded: float
    gap: float                  # mean_real − mean_degraded   (this is the CDI)
    wilcoxon_p: float | None    # None when n < 5 or scores are identical
    cohens_d: float | None      # None when sd is zero
    interpretation: str         # short qualitative label

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "variant": self.variant,
            "n_pairs": self.n_pairs,
            "mean_real": round(self.mean_real, 3),
            "mean_degraded": round(self.mean_degraded, 3),
            "sd_real": round(self.sd_real, 3),
            "sd_degraded": round(self.sd_degraded, 3),
            "cdi_gap": round(self.gap, 3),
            "wilcoxon_p": (round(self.wilcoxon_p, 4) if self.wilcoxon_p is not None else None),
            "cohens_d": (round(self.cohens_d, 3) if self.cohens_d is not None else None),
            "interpretation": self.interpretation,
        }


# ── Statistics ──────────────────────────────────────────────────────────────


def _safe_stdev(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _cohens_d_paired(real: list[float], degraded: list[float]) -> float | None:
    """Cohen's d for paired samples = mean(diff) / sd(diff). None if sd=0."""
    if len(real) != len(degraded) or len(real) < 2:
        return None
    diffs = [r - d for r, d in zip(real, degraded)]
    sd = _safe_stdev(diffs)
    if sd == 0.0:
        return None
    return statistics.fmean(diffs) / sd


def _wilcoxon(real: list[float], degraded: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value. Returns None on degenerate input."""
    if len(real) != len(degraded) or len(real) < 5:
        return None
    diffs = [r - d for r, d in zip(real, degraded)]
    if all(d == 0 for d in diffs):
        return None
    try:
        # zero_method="zsplit" handles tied pairs the standard way.
        res = stats.wilcoxon(real, degraded, zero_method="zsplit", alternative="two-sided")
        p = float(res.pvalue)
        if math.isnan(p):
            return None
        return p
    except ValueError:
        return None


def _interpret(gap: float, p: float | None, expected_drop: bool) -> str:
    """Pithy verdict for the table cell.

    `expected_drop=True` means: for THIS combination of variant×dimension,
    we expect the critic to score the degraded item lower than the real one.
    (e.g. wrong_key×correctness → expected_drop=True)
    """
    sig = (p is not None and p < 0.05)
    if expected_drop:
        if gap > 1.0 and sig:
            return "strong discrimination"
        if gap > 0.3 and sig:
            return "moderate discrimination"
        if gap > 0.3:
            return "weak (not significant)"
        return "FAILED to discriminate"
    # Not expected to drop — we want gap ≈ 0
    if abs(gap) < 0.5:
        return "stable (as expected)"
    if gap > 0:
        return "unexpected drop"
    return "unexpected gain"


# What we EXPECT each degradation to affect.
# Used only for `interpretation` labels — never gates the computation itself.
_EXPECTED_AFFECTED: dict[str, set[str]] = {
    VARIANT_WRONG_KEY: {"correctness"},
    VARIANT_WEAK_DISTRACTORS: {"distractor_quality", "correctness"},
}


# ── Top-level entrypoint ────────────────────────────────────────────────────


def compute_cdi(
    rows: list[dict],
    dimensions: Sequence[str] = DIMENSIONS,
    variants: Sequence[str] = (VARIANT_WRONG_KEY, VARIANT_WEAK_DISTRACTORS),
) -> dict:
    """Compute CDI for every (variant, dimension) pair from a calibration CSV.

    `rows` is the list of records produced by scripts/calibrate_critic.py
    (each has `idx`, `variant`, and per-dimension columns). Items where a
    given dimension is missing/None are excluded from that dimension only;
    pairing is by `idx`.

    Returns a dict shaped:
      {
        "wrong_key":  {"correctness": DimensionCDI(...), ...},
        "weak_distractors": {...},
        "summary": {
            "n_real_items": int,
            "n_critic_failures": int,
        }
      }
    """
    by_variant: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], {})[int(r["idx"])] = r

    real_by_idx = by_variant.get(VARIANT_REAL, {})
    if not real_by_idx:
        return {"summary": {"n_real_items": 0, "n_critic_failures": 0}}

    n_failures = sum(1 for r in real_by_idx.values() if r.get("overall") is None)

    out: dict = {
        "summary": {
            "n_real_items": len(real_by_idx),
            "n_critic_failures": n_failures,
        }
    }

    for variant in variants:
        deg_by_idx = by_variant.get(variant, {})
        if not deg_by_idx:
            continue
        expected = _EXPECTED_AFFECTED.get(variant, set())
        per_dim: dict[str, DimensionCDI] = {}
        for dim in dimensions:
            real_scores: list[float] = []
            deg_scores: list[float] = []
            for idx, real_row in real_by_idx.items():
                deg_row = deg_by_idx.get(idx)
                if not deg_row:
                    continue
                r_val = real_row.get(dim)
                d_val = deg_row.get(dim)
                if r_val is None or d_val is None:
                    continue
                real_scores.append(float(r_val))
                deg_scores.append(float(d_val))

            n = len(real_scores)
            if n == 0:
                continue
            mr = statistics.fmean(real_scores)
            md = statistics.fmean(deg_scores)
            gap = mr - md
            p = _wilcoxon(real_scores, deg_scores)
            d = _cohens_d_paired(real_scores, deg_scores)
            per_dim[dim] = DimensionCDI(
                dimension=dim,
                variant=variant,
                n_pairs=n,
                mean_real=mr,
                mean_degraded=md,
                sd_real=_safe_stdev(real_scores),
                sd_degraded=_safe_stdev(deg_scores),
                gap=gap,
                wilcoxon_p=p,
                cohens_d=d,
                interpretation=_interpret(gap, p, dim in expected),
            )
        out[variant] = per_dim
    return out


# ── Formatters ──────────────────────────────────────────────────────────────


def format_cdi_markdown(cdi: dict) -> str:
    """One Markdown table per variant. Good for README / preview."""
    if not cdi or cdi.get("summary", {}).get("n_real_items", 0) == 0:
        return "_(no CDI data — calibration produced no rows)_"

    lines: list[str] = []
    s = cdi["summary"]
    lines.append(f"**Critic Discrimination Index (CDI)** — n_real={s['n_real_items']}, critic_failures={s['n_critic_failures']}\n")
    for variant, per_dim in cdi.items():
        if variant == "summary":
            continue
        lines.append(f"### Variant: `{variant}`\n")
        lines.append("| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for dim in DIMENSIONS:
            row = per_dim.get(dim)
            if row is None:
                continue
            p_str = f"{row.wilcoxon_p:.4f}" if row.wilcoxon_p is not None else "—"
            d_str = f"{row.cohens_d:.2f}" if row.cohens_d is not None else "—"
            lines.append(
                f"| {dim} | {row.n_pairs} | {row.mean_real:.2f} | {row.mean_degraded:.2f} | "
                f"{row.gap:+.2f} | {p_str} | {d_str} | {row.interpretation} |"
            )
        lines.append("")
    return "\n".join(lines)


def format_cdi_latex(cdi: dict, caption: str = "Critic Discrimination Index by variant and dimension.") -> str:
    """One \\subtable per variant — drop straight into the paper's methods section."""
    if not cdi or cdi.get("summary", {}).get("n_real_items", 0) == 0:
        return "% no CDI data\n"

    parts: list[str] = []
    parts.append("\\begin{table*}[t]")
    parts.append("\\centering")
    parts.append(f"\\caption{{{caption}}}")
    parts.append("\\label{tab:cdi}")
    for variant, per_dim in cdi.items():
        if variant == "summary":
            continue
        parts.append("\\vspace{0.5em}")
        parts.append(f"\\textbf{{Variant: \\texttt{{{variant}}}}}\\\\")
        parts.append("\\begin{tabular}{lrrrrrrl}")
        parts.append("\\hline")
        parts.append("Dimension & $n$ & $\\mu_\\text{real}$ & $\\mu_\\text{degraded}$ & CDI gap & Wilcoxon $p$ & Cohen's $d$ & Verdict \\\\")
        parts.append("\\hline")
        for dim in DIMENSIONS:
            row = per_dim.get(dim)
            if row is None:
                continue
            p_str = f"{row.wilcoxon_p:.4f}" if row.wilcoxon_p is not None else "--"
            d_str = f"{row.cohens_d:.2f}" if row.cohens_d is not None else "--"
            dim_label = dim.replace("_", r"\_")
            parts.append(
                f"{dim_label} & {row.n_pairs} & "
                f"{row.mean_real:.2f} & {row.mean_degraded:.2f} & "
                f"{row.gap:+.2f} & {p_str} & {d_str} & "
                f"{row.interpretation} \\\\"
            )
        parts.append("\\hline")
        parts.append("\\end{tabular}")
    parts.append("\\end{table*}")
    return "\n".join(parts)


def cdi_to_jsonable(cdi: dict) -> dict:
    """Convert dataclasses to plain dicts so json.dump works."""
    out: dict = {"summary": cdi.get("summary", {})}
    for variant, per_dim in cdi.items():
        if variant == "summary":
            continue
        out[variant] = {dim: row.to_dict() for dim, row in per_dim.items()}
    return out


# ── Inter-critic agreement (used by ensemble) ──────────────────────────────


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    """Cohen's kappa for two raters on a categorical task (A/B/C/D answers).

    Returns None when fewer than 2 paired observations or only one category.
    """
    if len(labels_a) != len(labels_b) or len(labels_a) < 2:
        return None
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    if len(pairs) < 2:
        return None
    categories = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    if len(categories) < 2:
        return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum(
        (sum(1 for a, _ in pairs if a == c) / n)
        * (sum(1 for _, b in pairs if b == c) / n)
        for c in categories
    )
    if pe == 1.0:
        return None
    return round((po - pe) / (1 - pe), 3)


def pairwise_kappa_matrix(per_critic_labels: dict[str, list[str]]) -> dict[tuple[str, str], float | None]:
    """For each pair of critics, compute Cohen's κ on their chosen answers.

    `per_critic_labels` is {model_id: [answer_for_item_1, answer_for_item_2, ...]}.
    """
    critics = sorted(per_critic_labels.keys())
    out: dict[tuple[str, str], float | None] = {}
    for i, a in enumerate(critics):
        for b in critics[i + 1:]:
            out[(a, b)] = cohens_kappa(per_critic_labels[a], per_critic_labels[b])
    return out
