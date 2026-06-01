# nct-question-generation

AI-powered multiple-choice question generator for Kazakhstan's National Testing Center (Ұлттық тестілеу орталығы, ҰТО) **teacher-certification** exams. Questions are written in Kazakh by a two-agent pipeline: a Generator Agent drafts each question and a Critic Agent evaluates it before saving.

The pipeline is calibrated against **real, published NTC items** (40 math + 25 Kazakh language/literature) which it uses both as few-shot exemplars for the generator and as a benchmark for critic validation.

## What's new in this branch (`feature/batch-generation-and-critic-validation`)

### Paper-level methodology contributions
- **Critic Discrimination Index (CDI)** — formalized metric that validates a critic agent *without* expert grading. We score real NTC items vs. programmatically-degraded variants (wrong-key, weak-distractors) and report the per-dimension gap, Wilcoxon p-value, and Cohen's d. See `src/cdi.py`, `scripts/compute_cdi.py`. This is **paper novelty #1**.
- **Multi-critic ensemble** — `EnsembleCriticAgent` runs N critics (default: GPT-4o + Claude Sonnet 4.6 + Qwen-2.5-72B from three different vendors) in parallel on every question. Disagreement is preserved as a per-item uncertainty signal; pairwise Cohen's κ across critics is reported as inter-rater reliability *among models*. See `src/ensemble.py`. This is **paper novelty #2**.
- **Symbolic self-verification (math only)** — the math generator emits a Python snippet that re-derives the answer with SymPy. The critic runs it in a hardened subprocess sandbox (denylist for `subprocess`/`socket`/`urllib`/…, neutered `os.system`/`os.popen`/`os.remove`, 5-second timeout). When the sandbox's output contradicts the generator's claimed answer, correctness is clamped to 0 regardless of what the LLM critic thinks. This catches arithmetic hallucinations that all three LLM critics in the ensemble can miss. See `src/symbolic.py`, `tests/test_symbolic.py`. Directly extends Kadyrov et al. 2025 (+27% UNT math accuracy with SymPy assistance) to the *generation* side of the loop. This is **paper novelty #3**.

### Pipeline improvements
- **Few-shot exemplars from real NTC questions.** The generator sees 1–3 real published items (matched by topic + level) alongside its instructions. Style and difficulty transfer noticeably.
- **Cross-model critic.** The Critic Agent can run on a *different* OpenRouter model than the Generator (`--critic-api` / `--critic-model`). Reduces self-evaluation bias. Default still mirrors the generator.
- **Vision-aware critic for image questions.** When a question has an attached figure, the Critic Agent sends the image bytes (base64 data URL) to a vision-capable model (default: GPT-4o). Text-only questions still use the cheaper text critic, so the vision premium only lands on the ~8–10 image items in a 50-question batch.
- **Stricter critic rubric.** Anchored numerical scoring, hard rule on the correctness dimension, explicit framing as teacher-certification quality control.
- **Per-model output paths.** Saved items go to `output/<subject>/<model_slug>/level_<X>/`.
- **Two clearly-labeled entry points** for single-question and batch generation (see below).
- **Critic self-validation harness** that scores real items + degraded variants and emits CDI + Cohen's κ — useful sanity test before any expert grading.

## Features

- Supports **Math** and **Kazakh Language / Literature** subjects
- Three difficulty levels: **A** (Basic, 26%), **B** (Medium, 60%), **C** (Hard, 14%)
- Two output formats: **text** (LaTeX) and **image** (auto-generated Matplotlib figure)
- Critic loop with weighted scoring across 5–6 dimensions — automatically retries failed questions
- Structured output: JSON + Markdown per question

## Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set your OPENROUTER_API_KEY
```

`.env` variables:

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | — |
| `OPENROUTER_MODEL` | Generator model | `openai/gpt-4o` |
| `OPENROUTER_CRITIC_MODEL` | Critic model for text questions (defaults to generator) | unset |
| `OPENROUTER_VISION_CRITIC_MODEL` | Critic model for image questions (vision-capable) | `openai/gpt-4o-2024-11-20` |
| `FEWSHOT_K` | Few-shot exemplars per generation (0 to disable) | `2` |

## Usage — two entry points

### Entry point 1 — generate **one question at a time** (`generate.py`)

Use this for iterating on prompts, debugging a single topic, or producing items one by one.

```bash
python generate.py --subject <subject> --level <level> --format <format> [options]
```

| Argument | Required | Values | Description |
|---|---|---|---|
| `--subject` | yes | `math`, `kazakh` | Subject area |
| `--level` | yes | `A`, `B`, `C` | Difficulty level |
| `--format` | yes | `text`, `image` | Output format |
| `--count` | no | integer | Number of questions in this run (default: 1) |
| `--topic` | no | topic ID | Specific topic (random if omitted) |
| `--api` | no | shortcut | `gpt-4o-2024-11-20` / `Qwen/Qwen2.5-72B-Instruct` / `claude-sonnet-4.6` |
| `--model` | no | model ID | Raw OpenRouter ID for the GENERATOR (overrides `--api`) |
| `--critic-api` | no | shortcut | API shortcut for the CRITIC on TEXT questions (default: same as `--api`) |
| `--critic-model` | no | model ID | Raw OpenRouter ID for the CRITIC on TEXT questions (overrides `--critic-api`) |
| `--vision-critic-api` | no | shortcut | API shortcut for the CRITIC on IMAGE questions (default: gpt-4o) |
| `--vision-critic-model` | no | model ID | Raw OpenRouter ID for the vision critic (overrides `--vision-critic-api`) |
| `--ensemble` | no | flag | Use multi-critic ensemble instead of a single critic |
| `--ensemble-critics` | no | csv | Comma-separated critic models (default: GPT-4o, Claude Sonnet 4.6, Qwen-2.5-72B) |
| `--ensemble-strict` | no | flag | Require ALL critics to pass (default: majority vote) |
| `--output-dir` | no | path | Output directory (default: `output/`) |

Examples:

```bash
# Single math question, level A, text format
python generate.py --subject math --level A --format text

# Cross-model critic: Generator = Qwen, Critic = Claude
python generate.py --subject math --level B --format text \
    --api Qwen/Qwen2.5-72B-Instruct --critic-api claude-sonnet-4.6

# Specific topic + image format
python generate.py --subject math --level C --format image --topic triangles

# Image-format question with a Claude vision critic instead of GPT-4o
python generate.py --subject math --level B --format image --topic triangles \
    --vision-critic-api claude-sonnet-4.6
```

#### Vision routing rules

| When the critic runs | Model it uses |
|---|---|
| Text-format question, no image attached | `--critic-api` / `--critic-model` (defaults to mirror generator) |
| Image-format question OR real item with `metadata.context` pointing to a file | `--vision-critic-api` / `--vision-critic-model` (defaults to `openai/gpt-4o-2024-11-20`) |
| Vision critic model not in the vision-capable allowlist | Silently falls back to text critic and skips the image |

Only OpenRouter IDs in `src/agents.py::VISION_CAPABLE_MODELS` will receive the image. Adding a new vision model: append its OpenRouter ID to that set.

### Entry point 2 — generate **a full batch of 50** (`scripts/run_batch.py`)

Use this once you're happy with the prompts and want to produce a per-model corpus.

```bash
python scripts/run_batch.py --subject <subject> --api <api> --count <n>
```

It distributes the requested count across levels using the NTC mix from `prompts/difficulty.yaml` (26 / 60 / 14) and round-robins topics. A per-run manifest is written to `output/<subject>/<model_slug>/_manifest_<timestamp>.json` listing every attempt (saved + rejected + errored).

```bash
# 50 math questions from GPT-4o
python scripts/run_batch.py --subject math --api gpt-4o-2024-11-20 --count 50

# Same, but with a Claude critic to reduce self-evaluation bias
python scripts/run_batch.py --subject math --api gpt-4o-2024-11-20 \
    --critic-api claude-sonnet-4.6 --count 50

# Run both subjects in one invocation
python scripts/run_batch.py --subject both --api claude-sonnet-4.6 --count 50

# Resume an interrupted run
python scripts/run_batch.py --subject math --api gpt-4o-2024-11-20 \
    --count 50 --resume
```

### Critic self-validation (no experts required)

`scripts/calibrate_critic.py` scores real NTC items + 2 degraded variants per item (wrong-answer-key, weak-distractors) and reports the discrimination gap. Useful before any expert involvement.

```bash
# Tiny sanity run on 5 real math items
python scripts/calibrate_critic.py --subject math --limit 5 --api gpt-4o-2024-11-20

# Full bank with a stronger critic model
python scripts/calibrate_critic.py --subject both --api claude-sonnet-4.6

# Ensemble calibration: GPT-4o + Claude Sonnet 4.6 + Qwen2.5-72B
# Emits CDI table + pairwise Cohen's κ across the 3 critics in one go.
python scripts/calibrate_critic.py --subject math --ensemble
```

## Paper contributions — how to invoke them

### 1. Critic Discrimination Index (CDI)

The CDI quantifies how well a critic agent distinguishes real published items from synthetically degraded ones, *without* needing expert ratings. For each (variant × dimension) we compute:

- Mean critic score on real vs. degraded items
- **CDI gap** = `mean(real) − mean(degraded)`  (the headline number)
- Wilcoxon signed-rank p-value (paired, two-sided)
- Cohen's d for paired samples

A well-calibrated critic should produce *large positive CDI* on dimensions the degradation targets (e.g. wrong-key degradation → correctness CDI ≫ 0) and *near-zero CDI* on unaffected dimensions (e.g. wrong-key should not change Kazakh language quality).

```bash
# Step 1: run calibration to produce the CSV
python scripts/calibrate_critic.py --subject math --ensemble

# Step 2: re-analyze any saved CSV without re-spending API credits
python scripts/compute_cdi.py output/_critic_validation/math_ensemble_*.csv \
    --markdown paper/cdi.md \
    --latex    paper/cdi.tex \
    --json     paper/cdi.json
```

`compute_cdi.py` reads only the CSV — no LLM calls, no network. Re-run it any time the rubric changes.

### 2. Multi-critic ensemble

Pass `--ensemble` to `generate.py` or `scripts/run_batch.py`. The default ensemble is one model from each of three vendors:

- `openai/gpt-4o-2024-11-20`
- `anthropic/claude-sonnet-4.6`
- `qwen/qwen-2.5-72b-instruct`

For each question, all critics run in parallel (Python `ThreadPoolExecutor`). The pipeline aggregates:

- `critic_answer` → majority vote across critics
- Per-dimension scores → mean across critics
- `pass_fail` → majority pass by default (or `--ensemble-strict` requires unanimity)
- `answer_agreement` → fraction of critics that picked the majority answer
- `unanimous` → bool

The full per-critic breakdown is preserved on the saved Question JSON under `ensemble.per_critic`. The calibration script additionally reports **pairwise Cohen's κ** across critics on the real-item answer matrix — this is our inter-rater reliability number for the paper, computed without humans.

```bash
# Single ensemble question
python generate.py --subject math --level B --format text --ensemble

# Strict mode (require all 3 critics to pass)
python generate.py --subject math --level C --format text --ensemble --ensemble-strict

# Batch with custom 2-model ensemble (ablation)
python scripts/run_batch.py --subject math --count 10 --ensemble \
    --ensemble-critics gpt-4o-2024-11-20,claude-sonnet-4.6
```

**Cost note.** Ensemble = N× critic calls per question. Default N=3, so a text question that previously cost ≈1 critic call now costs ≈3. The generator side is unchanged. With `--ensemble`, expect ~3× critic-side cost for the batch.

### 3. Symbolic self-verification (math)

The math generator now emits a `verification` JSON block alongside every question — a Python snippet plus the expected stdout. The critic runs that snippet in a hardened subprocess sandbox before scoring. Three outcomes:

| Sandbox result | Effect on the critic |
|---|---|
| Snippet output matches `expected_output` AND `matches_option == correct_answer` | Correctness floor raised to 9; LLM critic still scores other dimensions |
| Snippet output ≠ `expected_output` (generator's own check disagrees) | **Correctness clamped to 0; question rejected** regardless of LLM verdict |
| `matches_option ≠ correct_answer` (verification labels a different option) | **Same hard reject** — catches answer-key hallucinations |
| `applicable: false` or sandbox errors / times out | Skipped, falls back to LLM-only correctness |

The verdict is persisted on `CriticFeedback.verification` for downstream analysis.

**Sandbox threat model.** The sandbox is "defense in depth for the LLM-hallucination threat model," not airtight. Defenses:

- Subprocess isolation via `python -I -c ...` (no env, no user-site packages).
- Hard wall-clock timeout (default 5s).
- Import denylist for the dangerous standard-library: `subprocess`, `multiprocessing`, `socket`, `urllib`, `http`, `https`, `requests`, `ftplib`, `telnetlib`, `smtplib`, `poplib`, `imaplib`, `ssl`, `cffi`, `pty`, `tty`, `termios`, `fcntl`, `resource`, `syslog`, `webbrowser`, `ensurepip`, `venv`, `runpy`. Plus `importlib.import_module` is patched so it can't bypass the hook.
- Pre-imported SymPy/NumPy bind their needed transitives once, then those modules are blanked out of `sys.modules` so user code can't reach them via `sys.modules['subprocess']`.
- `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `os.remove`, `os.unlink`, `os.rmdir`, `os.chmod`, `os.chown`, `os.fork`, `os.kill`, `os.startfile` are set to `None` on the `os` module so calls fail with `TypeError`.
- `open` and `breakpoint` builtins are kept (sympy needs `open`); `breakpoint` is removed.

NOT defended against:
- Resource exhaustion (no `rlimit` enforced on macOS).
- An adversarial payload that reaches the *already-imported* `subprocess` through `sympy.printing.gtk.subprocess` (sympy holds a closure reference).
- Filesystem reads / writes via `open()` (used by sympy's mpmath for precision tables).
- Anything ctypes-based that bypasses Python's import system.

For a production deployment, run the verification subprocess inside Docker with `--network=none --read-only` or a WASM runtime. Documented as self-critique item #13.

The full set of attack vectors I tested (`tests/test_symbolic.py`) and confirmed blocked: `subprocess`, `socket`, `urllib`, `multiprocessing`, `importlib.import_module('subprocess')`, `sys.modules['subprocess'].run(...)`, `os.system`, `os.remove`, `breakpoint`, infinite loop. All 28 tests pass.

Set `FEWSHOT_K=0` and remove `--ensemble` to compare a baseline run against verification-enabled. The CDI columns `verified_passed` / `verified_contradicted` on the calibration CSV let you quantify how often verification fires.

## Output Structure

```
output/
├── <subject>/
│   ├── <model_slug>/
│   │   ├── level_A/
│   │   │   ├── <timestamp>_<id>.json
│   │   │   └── <timestamp>_<id>.md
│   │   ├── level_B/...
│   │   ├── level_C/...
│   │   └── _manifest_<timestamp>.json   (batch runs only)
│   └── figures/
│       └── <figure_type>_<hash>.png
└── _critic_validation/
    ├── <subject>_<critic_slug>.csv
    ├── <subject>_<critic_slug>_summary.json
    └── <subject>_<critic_slug>_details.json
```

Each generated question yields:
- **JSON** — full structured data including critic scores, figure spec, model id, and metadata
- **Markdown** — human-readable preview with embedded figure image

## Pipeline

```
generate.py / scripts/run_batch.py
    │
    ├─► GeneratorAgent — prompts an LLM with topic + level + few-shot exemplars
    │       │
    │       └─► (image format) FigureGenerator — renders a Matplotlib figure
    │
    ├─► CriticAgent (may use a DIFFERENT model than the generator)
    │       ├── Step 1: independently solves the question
    │       └── Step 2: scores across 5 dimensions (0–10)
    │
    └─► If score < 6.0 or correctness < 7 → retry with critic feedback
        (up to 4 attempts total)
```

### Critic Scoring Dimensions

| Dimension | Weight | Notes |
|---|---|---|
| Correctness | 3 | Hard rule: critic must score ≤ 4 if its answer disagrees with the provided key |
| Distractor quality | 2 | Each distractor must reflect a specific identifiable error |
| Difficulty alignment | 2 | Real cognitive level vs. requested A / B / C |
| Kazakh language quality | 2 | Modern literary norm, official terminology |
| LaTeX validity | 1 | Renders correctly, math questions must use LaTeX |
| Figure relevance *(image only)* | 1 | Figure must be essential to solve |

A question passes only when weighted overall ≥ 6.0 **and** correctness ≥ 7.

## Reference data under `files/`

| File | What's in it | Used by |
|---|---|---|
| `mathematics_questions_kz.json` | 40 real NTC math items (Kazakh + LaTeX + correct answer + topic + difficulty) | Few-shot bank, critic calibration |
| `kazakh_questions_no_context.json` | 20 real Kazakh-language/literature items | Few-shot bank, critic calibration |
| `kazakh_questions_with_context.json` | 5 items sharing one reading passage | Calibration only — context-block generation not yet supported |
| `mathematics_images/` | 3 PNG/JPEG images referenced by image-anchored math items | Excluded from text few-shot pool |
| `qazaq_language_topics.xlsx` | The 10 official themed Kazakh-language topic bundles | Reference for topic taxonomy reconciliation (TODO) |
| `literature_review.xlsx` | 23-paper literature review | Manuscript writing |

## Project Structure

```
ozp_project/
├── generate.py                       # Entry point 1 — single question(s)
├── scripts/
│   ├── run_batch.py                  # Entry point 2 — full corpus generation (ensemble-aware)
│   ├── calibrate_critic.py           # Critic self-validation on real items (emits CDI + κ)
│   └── compute_cdi.py                # Post-hoc CDI re-analysis from saved CSV (no API calls)
├── src/
│   ├── agents.py                     # GeneratorAgent + CriticAgent (cross-model + vision + symbolic-verify aware)
│   ├── cdi.py                        # Critic Discrimination Index — paper novelty #1
│   ├── config.py                     # Config, prompt rendering, exemplar wiring
│   ├── ensemble.py                   # EnsembleCriticAgent — paper novelty #2
│   ├── exemplars.py                  # Few-shot retrieval from real NTC data
│   ├── figure_gen.py                 # Matplotlib figure renderer
│   ├── models.py                     # Pydantic models (Question + VerificationSpec)
│   ├── output.py                     # JSON + Markdown writer (per-model paths)
│   └── symbolic.py                   # Sandboxed SymPy verification — paper novelty #3
└── tests/
    └── test_symbolic.py              # 28 unit tests for the sandbox
├── prompts/
│   ├── generator_math.md             # Now consumes ${examples_block}
│   ├── generator_kazakh.md           # Now consumes ${examples_block}
│   ├── critic_solve.md               # Strict independent-solve prompt
│   ├── critic_eval.md                # Strict scoring rubric with anchors
│   ├── difficulty.yaml
│   ├── figures.yaml
│   ├── topics_math.yaml
│   └── topics_kazakh.yaml
├── files/                            # Real NTC data + spec docs + literature review
└── output/                           # Generated questions (git-ignored)
```

## Honest self-critique (read this before trusting the pipeline)

A few things the current pipeline does **not** yet do well, and a few things I am uncertain about. Listing them up front rather than burying them.

0. **CDI is a one-operationalization metric.** We define CDI in terms of two specific degradation strategies (wrong-key relabel, weak-distractor injection). Reviewers may ask whether other degradations (paraphrase noise, format corruption, dimension-swap) would tell the same story. Honest answer: probably similar trends but not identical. The metric is defensible as defined; calling it *the* discrimination index would be overreach. Future work: report CDI under at least 4 degradation modes and average.

0a. **Ensemble agreement ≠ correctness.** Pairwise Cohen's κ across critic models tells us *whether the critics agree with each other*, not whether they're collectively right. Three confused critics can unanimously agree on a wrong answer. We still need expert grading on a subset to anchor "correctness" — the ensemble just gives us a free, cheap proxy for confidence. In the paper, present κ and CDI together; do NOT present κ alone as a quality claim.

0b. **Symbolic verification is generator-honest, not adversarial-safe.** The LLM that generates the question also writes its own verification snippet. If it hallucinates the answer AND writes a snippet that confirms its hallucination, verification passes a wrong answer. The contradiction-catching case fires only when the generator is *internally inconsistent* (math says X, claimed answer says Y). This is still useful — most LLM math errors come from arithmetic mistakes, not coordinated self-deception. But reviewers should not be told "SymPy verifies all our math items"; the honest claim is "SymPy catches generator-internal inconsistencies on ~the subset of items where verification.applicable is true (estimated 50-60% of math items)."

0c. **Sandbox is research-grade.** See the "Sandbox threat model" section above. Verified against 11 attack vectors in `tests/test_symbolic.py`; not certified against unknown ones. For deployment, swap in a real container. Documented prominently because reviewers will ask.

1. **Self-evaluation bias is only partly mitigated.** Cross-model critic is *available* but it's still an LLM judging another LLM. Even with different vendors, both models share broad training-data distributions. Human expert evaluation remains required for any paper claim about quality. The literature review specifically flags LLM-as-judge as unreliable for fine-grained dimensions (see Yao et al. 2025, Byun & Choi 2025).

2. **Topic taxonomy mismatch.** `prompts/topics_kazakh.yaml` lists pure-linguistic topics (`phonetics`, `morphology`, `syntax`), but the real Kazakh items in `files/kazakh_questions_no_context.json` are predominantly **literature** questions tied to specific authors and works (Abai, M. Әуезов, etc.). The few-shot retrieval works by Kazakh-name substring matching, so when the topic IDs don't overlap, the bank serves *generic* style examples rather than topic-aligned ones. Reconciling the taxonomy to the official 10 themed bundles in `files/qazaq_language_topics.xlsx` is a clean follow-up.

3. **Math correctness depends on the critic LLM's arithmetic and vision.** A pure-LLM critic can be wrong about arithmetic or misread a figure. For text-only questions, I deliberately did **not** add symbolic verification (SymPy), even though the literature (Kadyrov et al. 2025) shows it lifts weak-model accuracy substantially. For image questions, the vision critic *does* now see the figure (verified live on real NTC item id 222560: the critic correctly described the trapezoid's bases and inscribed-circle radius from the picture). Vision adds OCR/geometry errors of its own — initial smoke test showed GPT-4o disagreeing with the human answer key on one of the 3 real image items, which could be a model error OR a published-item flaw. Either way, the system now surfaces the disagreement instead of silently agreeing.

4. **Critic costs scale with mode.** Each generation triggers one generator call + one critic-solve + one critic-eval (3 calls per attempt). For a text-format question this is all on the text critic, identical to the original cost. For an image-format question, the solve and eval both run on the vision critic and include the image payload — roughly 2–3× the per-call cost of a text-only message on GPT-4o. With ~8–10 image questions per 50-question batch (the user's expected mix), the vision premium adds ~15–25% to total batch cost compared to the pre-vision version. Cross-model text critic (e.g., Claude critic on a Qwen generator) adds another ~30–60% per-question depending on the critic model's pricing.

5. **Context-block questions are unsupported.** Real NTC tests include 10 of 50 questions tied to a shared reading passage / table / chart. The pipeline currently generates standalone items only. Adding context-block support would require a different orchestration layer (one stimulus → 5 questions).

6. **The 40 math + 20 Kazakh real items are tiny.** Few-shot retrieval works fine at this scale, but if we ever wanted to fine-tune a model on this data (à la PersianMCQ-Instruct, Zeinalipour et al. 2025) we would need to enlarge the gold set considerably.

7. **No automated correctness ground-truth check yet.** The critic-self-validation script (`scripts/calibrate_critic.py`) checks that the critic *scores* real items high and *degrades* score on bad variants. It does NOT yet verify that generated items have a defensibly correct answer against any external source. SymPy for math + a separate "second opinion" model would be the natural next step.

8. **Exact OpenRouter ID for Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6`) is unverified against the live catalog.** First call to that model may 404 if the slug has changed. Workaround: pass `--model <real-id>` or `--critic-model <real-id>`.

9. **`yessens/` is a byte-identical copy of `files/`.** Looks like an accidental backup. Not pruning it in this branch but worth deleting once the user confirms it's safe.

10. **No tests.** There is no automated test suite. Smoke testing the two entry points each session is the current quality gate. For a publishable system this should be replaced with at least basic integration tests on the critic loop (mocked LLM responses).

## Pipeline

```
generate.py
    │
    ├─► GeneratorAgent  — prompts the LLM to produce a JSON question
    │       │
    │       └─► (image format) FigureGenerator — renders a Matplotlib figure
    │
    ├─► CriticAgent
    │       ├── Step 1: independently solves the question
    │       └── Step 2: scores across 5 dimensions (0–10)
    │
    └─► If score < 6.0 or figure_spec missing → retry with critic feedback
        (up to 4 attempts total)
```

## Supported Figure Types

Used automatically when `--format image` is selected, based on the topic:

| Figure type | Used for topics |
|---|---|
| `triangle` | Triangles, 3D perpendicularity |
| `circle` | Polygons & circles, Triangles |
| `function_graph` | Functions, derivatives, integrals, polynomials |
| `trig_graph` | Trigonometry |
| `vector_diagram` | Vectors |
| `sequence_plot` | Sequences |
| `solid_3d` | Polyhedra, solids of revolution, 3D coordinates |
| `coordinate_plane` | Vectors, coordinate geometry, complex numbers |

Figures follow Kazakhstani school textbook style: black and white, no gridlines, measurements labeled directly on the figure.

## Topics

### Math
`triangles`, `polygons_circles`, `vectors`, `coordinate_3d`, `sequences`, `functions_limits`, `derivatives`, `integrals`, `trigonometry`, `polynomials`, `quadratic_irrational`, `exponential_logarithmic_functions`, `exponential_log_equations`, `complex_numbers`, `polyhedra`, `solids_of_revolution`, `perpendicularity_3d`

### Kazakh Language
`phonetics`, `morphology`, `syntax`, `lexicology`, `stylistics`, `literature_theory`, `kazakh_literature`, `world_literature`

(Note: the Kazakh topic list does not yet align with the 10 official themed bundles from `files/qazaq_language_topics.xlsx`. See self-critique #2.)
