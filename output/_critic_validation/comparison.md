# Critic model selection

**Selected critic: `google/gemini-3.1-pro-preview`** (score 7.51)

| Rank | Model | Score | Real acc. | Real pass | Gap wrong_key→correct | Gap weak_distr→distr_q | Instability | Failures |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `google/gemini-3.1-pro-preview` | 7.51 | 0.93 | 0.90 | +8.43 | +6.02 | 0.91 | 1 |
| 2 | `openai/gpt-5.5` | 6.97 | 0.93 | 0.85 | +8.05 | +3.29 | 0.23 | 0 |
| 3 | `anthropic/claude-sonnet-4.6` | 6.83 | 0.90 | 0.88 | +7.62 | +3.55 | 0.51 | 0 |

Score = 0.4·(real accuracy) + 0.15·(real pass rate) + 0.35·(mean targeted CDI gap) − 0.1·(instability), rates scaled ×10.
