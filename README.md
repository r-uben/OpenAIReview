# OpenAIReview

AI-powered academic paper reviewer that detects technical and logical errors using LLMs.

## Installation

```bash
pip install .
# or for development:
pip install -e .
```

## Quick Start

First, set your OpenRouter API key (get one at [openrouter.ai/keys](https://openrouter.ai/keys)):

```bash
export OPENROUTER_API_KEY=your_key_here
```

Or create a `.env` file in your working directory:

```
OPENROUTER_API_KEY=your_key_here
```

Then review a paper and visualize results:

```bash
openaireview review paper.pdf
openaireview serve
# Open http://localhost:8080
```

## CLI Reference

### `openaireview review <file>`

Review an academic paper for technical and logical issues.

| Option | Default | Description |
|---|---|---|
| `--method` | `incremental` | Review method: `zero_shot`, `local`, `incremental`, `incremental_full` |
| `--model` | `anthropic/claude-opus-4-5` | Model to use |
| `--output-dir` | `./review_results` | Directory for output JSON files |
| `--name` | (from filename) | Paper slug name |

### `openaireview serve`

Start a local visualization server to browse review results.

| Option | Default | Description |
|---|---|---|
| `--results-dir` | `./review_results` | Directory containing result JSON files |
| `--port` | `8080` | Server port |

## Supported File Formats

- **PDF** (`.pdf`) — text extraction via PyMuPDF
- **DOCX** (`.docx`) — via python-docx
- **LaTeX** (`.tex`) — plain text with title extraction from `\title{}`
- **Text/Markdown** (`.txt`, `.md`) — plain text

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `MODEL` | `anthropic/claude-opus-4-5` | Default model |

These can be set as environment variables or in a `.env` file. See `.env.example` for a template.

## Review Methods

- **zero_shot** — single prompt asking the model to find all issues
- **local** — deep-checks each chunk with surrounding window context (no filtering)
- **incremental** — sequential processing with running summary, then consolidation
- **incremental_full** — same as incremental but returns all comments before consolidation

## Benchmarks

Benchmark data and experiment scripts are in `benchmarks/`. See `benchmarks/REPORT.md` for results.
