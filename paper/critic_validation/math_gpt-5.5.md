**Critic Discrimination Index (CDI)** — n_real=40, critic_failures=0

### Variant: `wrong_key`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 40 | 9.03 | 0.15 | +8.88 | 0.0000 | 3.30 | strong discrimination |
| distractor_quality | 40 | 6.40 | 5.86 | +0.54 | 0.0075 | 0.41 | unexpected drop |
| difficulty_alignment | 40 | 6.56 | 7.16 | -0.60 | 0.0122 | -0.34 | unexpected gain |
| kazakh_language_quality | 40 | 7.67 | 7.91 | -0.24 | 0.6032 | -0.25 | stable (as expected) |
| latex_validity | 40 | 9.18 | 9.16 | +0.01 | 0.7627 | 0.02 | stable (as expected) |

### Variant: `weak_distractors`

| Dimension | n | mean(real) | mean(degraded) | CDI gap | Wilcoxon p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| correctness | 40 | 9.03 | 8.94 | +0.09 | 0.7070 | 0.05 | FAILED to discriminate |
| distractor_quality | 40 | 6.40 | 3.48 | +2.93 | 0.0000 | 1.53 | strong discrimination |
| difficulty_alignment | 40 | 6.56 | 6.96 | -0.40 | 0.7527 | -0.21 | stable (as expected) |
| kazakh_language_quality | 40 | 7.67 | 7.58 | +0.10 | 0.0615 | 0.10 | stable (as expected) |
| latex_validity | 40 | 9.18 | 8.97 | +0.20 | 0.1407 | 0.27 | stable (as expected) |

