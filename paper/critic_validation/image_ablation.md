# Vision ablation on figure-dependent math items (n=3 per model)

| Critic | Blind (no image) | With image |
|---|---:|---:|
| claude-sonnet-4.6 | 2/3 | 3/3 |
| gemini-3.1-pro | 1/3 | 3/3 |
| gpt-5.5 | 3/3 | 3/3 |

**Total: blind 6/9 → with image 9/9.** Figure-dependent items the critic cannot solve without the image are recovered once the figure is supplied.
