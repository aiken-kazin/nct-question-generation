**Critic Discrimination Index (CDI)** — n_real=40, critic_failures=0

### Variant: `wrong_key`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 40 | 9.28 | 0.12 | +9.15 | 0.0000 | 4.50 | strong discrimination |
| distractor_quality | 40 | 7.97 | 6.40 | +1.57 | 0.0000 | 0.81 | unexpected drop |
| difficulty_alignment | 40 | 8.05 | 7.85 | +0.20 | 0.0744 | 0.11 | stable (as expected) |
| kazakh_language_quality | 40 | 8.55 | 8.18 | +0.38 | 0.0776 | 0.28 | stable (as expected) |
| latex_validity | 40 | 9.03 | 8.95 | +0.08 | 0.8488 | 0.07 | stable (as expected) |

### Variant: `weak_distractors`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 40 | 9.28 | 9.07 | +0.20 | 0.8106 | 0.08 | FAILED to discriminate |
| distractor_quality | 40 | 7.97 | 2.58 | +5.40 | 0.0000 | 1.80 | strong discrimination |
| difficulty_alignment | 40 | 8.05 | 7.12 | +0.93 | 0.0026 | 0.34 | unexpected drop |
| kazakh_language_quality | 40 | 8.55 | 7.83 | +0.73 | 0.0027 | 0.48 | unexpected drop |
| latex_validity | 40 | 9.03 | 8.43 | +0.60 | 0.0371 | 0.40 | unexpected drop |

