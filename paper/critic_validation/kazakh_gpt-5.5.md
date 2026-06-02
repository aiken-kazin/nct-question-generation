**Critic Discrimination Index (CDI)** — n_real=20, critic_failures=0

### Variant: `wrong_key`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 20 | 8.00 | 1.30 | +6.70 | 0.0002 | 1.69 | strong discrimination |
| distractor_quality | 20 | 5.65 | 5.12 | +0.53 | 0.1601 | 0.28 | unexpected drop |
| difficulty_alignment | 20 | 5.00 | 4.78 | +0.22 | 0.3842 | 0.24 | stable (as expected) |
| kazakh_language_quality | 20 | 6.35 | 6.15 | +0.20 | 0.3162 | 0.27 | stable (as expected) |
| latex_validity | 20 | 10.00 | 10.00 | +0.00 | — | — | stable (as expected) |

### Variant: `weak_distractors`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 20 | 8.00 | 8.53 | -0.53 | 0.7068 | -0.26 | FAILED to discriminate |
| distractor_quality | 20 | 5.65 | 1.75 | +3.90 | 0.0001 | 2.99 | strong discrimination |
| difficulty_alignment | 20 | 5.00 | 4.42 | +0.58 | 0.0253 | 0.54 | unexpected drop |
| kazakh_language_quality | 20 | 6.35 | 5.75 | +0.60 | 0.0033 | 0.73 | unexpected drop |
| latex_validity | 20 | 10.00 | 10.00 | +0.00 | — | — | stable (as expected) |

