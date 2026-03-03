# RAG Overhaul and Benchmarking Plan

## Summary of Changes Implemented

All code changes below have been implemented and verified locally. Benchmark data has been regenerated.

### 1. Fixed prices in `evaluate.py`

Corrected OpenRouter prices (per 1M tokens):

| Model | Prompt | Completion |
|---|---|---|
| anthropic/claude-opus-4-6 | $5 | $25 |
| anthropic/claude-opus-4-5 | $5 | $25 |
| anthropic/claude-haiku-4-5 | $1 | $5 |
| openai/gpt-4o | $2.5 | $10 |
| openai/gpt-4o-mini | $0.15 | $0.60 |
| google/gemini-2.0-flash-001 | $0.10 | $0.40 |

Also removed precision from `print_report` summary; only recall + cost are shown.

### 2. Added `overall_feedback` to `ReviewResult` in `models.py`

New field `overall_feedback: str = ""` and included in `to_dict()`.

### 3. Added `parse_review_response` to `utils.py`

New function returns `tuple[str, list[Comment]]` (overall_feedback, comments).

Handles two formats:
- `{"overall_feedback": "...", "comments": [...]}` (preferred)
- `[...]` bare array (fallback)

Uses `json.JSONDecoder().raw_decode()` for robust JSON extraction. The old `parse_comments_from_response` is kept as a thin wrapper for backward compatibility.

Inner list parsing extracted into `parse_comments_from_list(items)`.

### 4. Fixed `parse_examples.py` — extracts `overallFeedback`

Changed from trying `summary`/`overview`/`verdict` keys to directly extracting `overallFeedback`. Benchmark regenerated — all 4 papers now have an `overall_feedback` field.

### 5. Updated `method_zero_shot.py` and `method_few_shot.py`

Prompts updated to return JSON object format:
```json
{
  "overall_feedback": "One paragraph high-level assessment",
  "comments": [{"title": ..., "quote": ..., "explanation": ..., "type": ...}]
}
```

Both methods now use `parse_review_response` and populate `result.overall_feedback`.

### 6. Redesigned `method_rag.py` — four variants

**Removed** `temperature=0.1` from all `chat()` calls.

**Four named variants** (`RAG_VARIANTS` dict):

| Variant | Context strategy | Filter | CoT |
|---|---|---|---|
| `rag_local` | ±3 window (baseline) | filter (top-15) | no |
| `rag_retrieved` | small-model retrieval | filter (top-15) | no |
| `rag_retrieved_cot` | small-model retrieval | filter (top-15) | yes |
| `rag_top_k_filter` | small-model retrieval | score-and-rank top-10 | no |

**Retrieval context**: small model receives all paragraph summaries (300 chars each) + full target paragraph and returns up to 8 relevant paragraph indices. Local ±1 neighbors always included.

**Top-k filter**: `SCORING_PROMPT` asks small model to score each paragraph 1-10; top 10 selected.

**CoT**: `DEEP_CHECK_COT_PROMPT` prepends "Think step by step: restate what the passage claims, check each claim..."

**Overall feedback**: after the comment-finding loop, one call to the large model with the first 8000 chars of the paper.

### 7. Updated `run_benchmark.py`

- Added `--rag-variant` argument (choices: all four variant names, default: `rag_retrieved`)
- Removed precision from progress print; now shows `Recall: X  Cost: $Y`

---

## Running on the Server

### Step 0: Verify LLM judge works before full runs

The LLM judge is slow (one API call per predicted×GT pair). Before running it at scale, test it on a small slice using the existing results file to confirm it produces non-zero recall and isn't stuck:

```bash
uv run python -c "
import json, sys
sys.path.insert(0, 'src')
from reviewer.models import Comment, ReviewResult
from reviewer.evaluate import evaluate

# Load one paper's predictions from the existing results file
results_file = 'results/results_20260223_131914.jsonl'
with open(results_file) as f:
    for line in f:
        r = json.loads(line)
        if r['method'] == 'zero_shot' and r['paper_slug'] == 'chaotic-balanced-state':
            break

# Load ground truth
with open('data/benchmark.jsonl') as f:
    for line in f:
        p = json.loads(line)
        if p['slug'] == 'chaotic-balanced-state':
            gt = p['comments']
            break

# Reconstruct ReviewResult from saved data
from reviewer.models import Comment, ReviewResult
result = ReviewResult(
    method=r['result']['method'],
    paper_slug=r['result']['paper_slug'],
    model=r['result']['model'],
    total_prompt_tokens=r['result']['total_prompt_tokens'],
    total_completion_tokens=r['result']['total_completion_tokens'],
    comments=[Comment(**c) for c in r['result']['comments']],
)

# Test judge on just first 2 GT comments vs first 3 predictions
from reviewer.evaluate import llm_judge_is_match
for gt in gt[:2]:
    for pred in result.comments[:3]:
        match = llm_judge_is_match(pred, gt)
        print(f'GT: {gt[\"title\"][:40]} | Pred: {pred.title[:40]} | match={match}')
print('LLM judge smoke test complete')
"
```

If the above prints results quickly and some `match=True`, the judge is working. Then run full LLM judge eval on one paper:

```bash
uv run python run_benchmark.py \
  --methods zero_shot \
  --papers chaotic-balanced-state \
  --llm-judge \
  --judge-model anthropic/claude-haiku-4-5 \
  --large-model anthropic/claude-opus-4-6
```

Check that `recall_llm` is higher than `recall` (similarity). If both are near 0, something is wrong with the prompts before scaling up.

### Run all methods on all papers

```bash
uv run python run_benchmark.py --methods zero_shot few_shot rag --large-model anthropic/claude-opus-4-6
```

### Benchmark all four RAG variants

```bash
for variant in rag_local rag_retrieved rag_retrieved_cot rag_top_k_filter; do
  uv run python run_benchmark.py --methods rag --rag-variant $variant --large-model anthropic/claude-opus-4-6
done
```

### Test cheaper small models for retrieval/filter step

```bash
# Default: Haiku ($1/$5 per M)
uv run python run_benchmark.py --methods rag --rag-variant rag_retrieved

# Gemini Flash (10x cheaper at $0.10/$0.40 per M)
uv run python run_benchmark.py --methods rag --rag-variant rag_retrieved --small-model google/gemini-2.0-flash-001

# GPT-4o-mini ($0.15/$0.60 per M)
uv run python run_benchmark.py --methods rag --rag-variant rag_retrieved --small-model openai/gpt-4o-mini
```

### Run with LLM judge for better recall estimation

```bash
uv run python run_benchmark.py --methods rag --rag-variant rag_retrieved --llm-judge --judge-model anthropic/claude-haiku-4-5
```

### Single paper quick test

```bash
uv run python run_benchmark.py --methods rag --rag-variant rag_local --papers targeting-interventions-networks
```

### 8. Added `paragraph_index` (comment location tracking)

Every comment — both ground-truth and model-generated — now carries a `paragraph_index` field (0-based index into the list of paragraphs obtained by splitting `document_content` on `\n\n`, keeping only paragraphs ≥ 100 chars).

**Changes:**

| File | What changed |
|---|---|
| `models.py` | `Comment` has `paragraph_index: int \| None = None`; included in `to_dict()` |
| `utils.py` | New helpers: `split_into_paragraphs`, `locate_comment_in_document` (fuzzy match), `assign_paragraph_indices` |
| `parse_examples.py` | Computes `paragraph_index` for each GT comment by matching `paragraph` text against the document's paragraph list |
| `method_zero_shot.py` | Calls `assign_paragraph_indices` on generated comments before returning |
| `method_few_shot.py` | Same as zero-shot |
| `method_rag.py` | Sets `paragraph_index = idx` directly since RAG iterates over paragraphs |
| `evaluate.py` | New metric: **location recall** — a GT comment is "location-recalled" if any predicted comment falls within ±3 paragraphs of it. Also computes `technical_location_recall` and `logical_location_recall`. Shown in `print_report`. |

**Location matching logic:**
- `locate_comment_in_document(quote, paragraphs)` first tries an exact substring match (first 80 chars), then falls back to `SequenceMatcher` with threshold 0.3.
- Location recall uses a ±3 paragraph window (`LOCATION_WINDOW = 3` in `evaluate.py`).

**Why this matters:**
- Location recall is a *complementary* metric to similarity/LLM-judge recall: it measures whether the model is looking at the right part of the paper, even if it describes the issue differently.
- Enables future side-by-side HTML rendering of papers with GT and predicted comments anchored to specific paragraphs.
- For RAG, `paragraph_index` comes for free since the method iterates over paragraphs.

---

## Evaluation Metrics

- **Recall (similarity)**: fuzzy quote matching with threshold 0.35
- **Recall (LLM judge)**: LLM decides if predicted and GT comments refer to the same issue (more accurate)
- **Recall (location)**: predicted comment is within ±3 paragraphs of a GT comment (measures whether the model focuses on the right region)
- **Cost**: estimated from token counts + OpenRouter prices above

Key comparison axes:
1. `rag_local` vs `rag_retrieved` — does semantic retrieval beat window context?
2. `rag_retrieved` vs `rag_retrieved_cot` — does CoT improve recall?
3. `rag_retrieved` vs `rag_top_k_filter` — does score-and-rank beat binary filter?
4. Small model choice — Haiku vs Gemini Flash vs GPT-4o-mini for retrieval quality vs cost
5. Location recall vs similarity/LLM recall — are models finding the right passages but describing issues differently?

---

## Known Issues / Next Steps

- Recall is low (4-6%) with similarity matching because models paraphrase rather than quote verbatim. LLM judge recall will likely be much higher.
- If LLM judge recall is also low, consider improving prompts to explicitly instruct verbatim quoting.
- RAG context window for retrieval prompt grows O(n_paragraphs). For very long papers (>200 paragraphs), may need to batch the retrieval step.
- Build side-by-side HTML visualization: render paper paragraphs with GT and predicted comments anchored at their `paragraph_index` positions.
