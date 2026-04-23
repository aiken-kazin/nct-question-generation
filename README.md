# nct-question-generation

AI-powered multiple-choice question generator for Kazakhstan's National Testing Center (Ұлттық тестілеу орталығы, ҰТО) teacher certification exams. Questions are written in Kazakh using a two-agent pipeline: a Generator Agent drafts each question and a Critic Agent evaluates it before saving.

## Features

- Supports **Math** and **Kazakh Language** subjects
- Three difficulty levels: **A** (Basic, 26%), **B** (Medium, 60%), **C** (Hard, 14%)
- Two output formats: **text** (LaTeX) and **image** (auto-generated Matplotlib figure)
- Critic loop with scoring across 5 dimensions — automatically retries failed questions
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
| `OPENROUTER_MODEL` | Model to use | `openai/gpt-4o` |

## Usage

```bash
python generate.py --subject <subject> --level <level> --format <format> [options]
```

### Arguments

| Argument | Required | Values | Description |
|---|---|---|---|
| `--subject` | yes | `math`, `kazakh` | Subject area |
| `--level` | yes | `A`, `B`, `C` | Difficulty level |
| `--format` | yes | `text`, `image` | Output format |
| `--count` | no | integer | Number of questions to generate (default: 1) |
| `--topic` | no | topic ID | Specific topic (random if omitted) |
| `--model` | no | model ID | Override the OpenRouter model |
| `--output-dir` | no | path | Output directory (default: `output/`) |

### Examples

```bash
# Single math question, level A, text format
python generate.py --subject math --level A --format text

# 3 Kazakh language questions, level B, image format
python generate.py --subject kazakh --level B --format image --count 3

# Specific topic, hard difficulty
python generate.py --subject math --level C --format image --topic triangles

# Use a different model
python generate.py --subject math --level B --format text --model anthropic/claude-3.5-sonnet
```

## Output Structure

Generated files are saved under:

```
output/
└── <subject>/
    ├── questions/
    │   └── level_<A|B|C>/
    │       ├── <timestamp>_<id>.json
    │       └── <timestamp>_<id>.md
    └── figures/
        └── <figure_type>_<hash>.png
```

Each question produces:
- **JSON** — full structured data including critic scores, figure spec, and metadata
- **Markdown** — human-readable preview with embedded figure image

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

### Critic Scoring Dimensions

| Dimension | Weight |
|---|---|
| Correctness | 3 |
| Distractor quality | 2 |
| Difficulty alignment | 2 |
| Kazakh language quality | 2 |
| LaTeX validity | 1 |
| Figure relevance *(image only)* | 1 |

A question passes with an overall weighted score ≥ 6.0.

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

## Project Structure

```
ozp_project/
├── generate.py          # CLI entry point
├── src/
│   ├── agents.py        # GeneratorAgent and CriticAgent
│   ├── config.py        # Config, prompt rendering, figure schema lookup
│   ├── figure_gen.py    # Matplotlib figure renderer
│   ├── models.py        # Pydantic data models
│   └── output.py        # JSON and Markdown file writer
├── prompts/
│   ├── generator_math.md
│   ├── generator_kazakh.md
│   ├── critic_solve.md
│   ├── critic_eval.md
│   ├── difficulty.yaml
│   ├── figures.yaml
│   ├── topics_math.yaml
│   └── topics_kazakh.yaml
├── output/              # Generated questions (git-ignored)
├── .env.example
└── requirements.txt
```
