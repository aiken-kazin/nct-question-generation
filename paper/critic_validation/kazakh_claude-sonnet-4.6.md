**Critic Discrimination Index (CDI)** — n_real=20, critic_failures=0

### Variant: `wrong_key`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 20 | 7.15 | 2.00 | +5.15 | 0.0001 | 1.73 | strong discrimination |
| distractor_quality | 20 | 5.95 | 5.00 | +0.95 | 0.0009 | 1.07 | unexpected drop |
| difficulty_alignment | 20 | 5.75 | 5.35 | +0.40 | 0.1605 | 0.34 | stable (as expected) |
| kazakh_language_quality | 20 | 7.62 | 7.45 | +0.17 | 0.2419 | 0.21 | stable (as expected) |
| latex_validity | 20 | 10.00 | 10.00 | +0.00 | — | — | stable (as expected) |

### Variant: `weak_distractors`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 20 | 7.15 | 6.70 | +0.45 | 0.1734 | 0.29 | weak (not significant) |
| distractor_quality | 20 | 5.95 | 1.35 | +4.60 | 0.0001 | 3.87 | strong discrimination |
| difficulty_alignment | 20 | 5.75 | 4.10 | +1.65 | 0.0001 | 2.46 | unexpected drop |
| kazakh_language_quality | 20 | 7.62 | 6.10 | +1.53 | 0.0001 | 1.62 | unexpected drop |
| latex_validity | 20 | 10.00 | 10.00 | +0.00 | — | — | stable (as expected) |

