**Critic Discrimination Index (CDI)** — n_real=20, critic_failures=0

### Variant: `wrong_key`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 20 | 8.85 | 0.70 | +8.15 | 0.0001 | 1.87 | strong discrimination |
| distractor_quality | 20 | 7.80 | 5.70 | +2.10 | 0.0062 | 0.72 | unexpected drop |
| difficulty_alignment | 20 | 6.40 | 5.60 | +0.80 | 0.1224 | 0.32 | unexpected drop |
| kazakh_language_quality | 20 | 7.70 | 8.30 | -0.60 | 0.1683 | -0.39 | unexpected gain |
| latex_validity | 20 | 10.00 | 9.75 | +0.25 | 0.6767 | 0.22 | stable (as expected) |

### Variant: `weak_distractors`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 19 | 9.11 | 7.84 | +1.26 | 0.0151 | 0.55 | strong discrimination |
| distractor_quality | 19 | 8.05 | 0.16 | +7.89 | 0.0001 | 4.07 | strong discrimination |
| difficulty_alignment | 19 | 6.74 | 2.00 | +4.74 | 0.0001 | 1.88 | unexpected drop |
| kazakh_language_quality | 19 | 7.95 | 6.05 | +1.89 | 0.0082 | 0.75 | unexpected drop |
| latex_validity | 19 | 10.00 | 10.00 | +0.00 | — | — | stable (as expected) |

