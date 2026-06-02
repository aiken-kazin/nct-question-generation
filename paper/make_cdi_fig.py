#!/usr/bin/env python3
"""Render a CDI bar chart (real vs degraded mean scores) for the paper."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "output" / "_critic_validation"
MODELS = [("claude-sonnet-4.6", "Claude-Sonnet-4.6"),
          ("gemini-3.1-pro-preview", "Gemini-3.1-Pro"),
          ("gpt-5.5", "GPT-5.5")]


def gap(subj, slug, variant, dim):
    d = json.load(open(VAL / f"{subj}_{slug}_summary.json"))
    cell = ((d.get("cdi") or {}).get(variant) or {}).get(dim) or {}
    return cell.get("cdi_gap", 0.0)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, subj, title in [(axes[0], "math", "Mathematics"), (axes[1], "kazakh", "Kazakh language")]:
    labels = [m[1] for m in MODELS]
    wk = [gap(subj, m[0], "wrong_key", "correctness") for m in MODELS]
    wd = [gap(subj, m[0], "weak_distractors", "distractor_quality") for m in MODELS]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, wk, w, label=r"wrong-key $\rightarrow$ correctness", color="#2b6cb0")
    ax.bar(x + w/2, wd, w, label=r"weak-distractors $\rightarrow$ distractor quality", color="#dd6b20")
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 10); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("CDI gap (points / 10)")
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
out = Path(__file__).resolve().parent / "fig_cdi.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
