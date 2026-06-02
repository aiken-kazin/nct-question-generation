**Critic Discrimination Index (CDI)** — n_real=40, critic_failures=0

### Variant: `wrong_key`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 40 | 9.80 | 0.82 | +8.98 | 0.0000 | 7.18 | strong discrimination |
| distractor_quality | 40 | 6.88 | 5.60 | +1.28 | 0.0000 | 1.24 | unexpected drop |
| difficulty_alignment | 40 | 7.45 | 7.28 | +0.17 | 0.0173 | 0.12 | stable (as expected) |
| kazakh_language_quality | 40 | 7.62 | 7.59 | +0.04 | 0.1278 | 0.04 | stable (as expected) |
| latex_validity | 40 | 8.57 | 8.57 | +0.00 | 0.1678 | 0.00 | stable (as expected) |

### Variant: `weak_distractors`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 40 | 9.80 | 9.30 | +0.50 | 0.0208 | 0.43 | moderate discrimination |
| distractor_quality | 40 | 6.88 | 3.92 | +2.95 | 0.0000 | 1.39 | strong discrimination |
| difficulty_alignment | 40 | 7.45 | 7.39 | +0.06 | 0.0211 | 0.04 | stable (as expected) |
| kazakh_language_quality | 40 | 7.62 | 7.31 | +0.31 | 0.0157 | 0.26 | stable (as expected) |
| latex_validity | 40 | 8.57 | 8.50 | +0.07 | 0.7774 | 0.05 | stable (as expected) |

