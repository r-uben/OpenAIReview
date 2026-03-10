# RFC 002: Structured reviewer pipeline

**Status**: proposal
**Issue**: #27 (improve overall feedback), #31 (concrete suggestions)

## Problem

Current review methods treat the paper as flat text. The progressive method
carries a prose summary across chunks, but doesn't build structured state.
This means the model can't reliably:

- Track notation across sections (symbol defined in S2, misused in S5)
- Verify claim-evidence links (theorem stated in S3, proof has gap in S4)
- Catch cross-reference errors (equation numbers, theorem references)

Benchmark recall is 4-6%. Models find correct passages but the evaluation
metric (string similarity) undercounts. Still, structured state should
improve both real and measured recall.

## Approach

### Phase 1: Structured extraction (pre-review)

Before reviewing, parse the paper into a structured representation:

```json
{
  "definitions": [{"term": "...", "definition": "...", "location": "S2.1"}],
  "notation": [{"symbol": "\\alpha", "meaning": "...", "first_use": "S1"}],
  "equations": [{"id": "eq3", "content": "...", "location": "S3"}],
  "theorems": [{"id": "thm1", "statement": "...", "proof_location": "S4"}],
  "claims": [{"claim": "...", "evidence": "...", "location": "S5"}]
}
```

This can be done with a single LLM pass over the full paper.

### Phase 2: Review against structure

Feed sections to the reviewer along with the relevant structured context
(not the whole paper). For each section, include:

- Notation table (symbols used in this section)
- Relevant definitions and theorems
- Claims that reference this section

### Phase 3: Structured output

Change reviewer output from free-form comments to atomic JSON:

```json
{
  "issue_type": "math_error",
  "severity": "major",
  "location": "S3, eq. 7",
  "claim": "The gradient is \\nabla_x f = 2x",
  "evidence": "By chain rule, should be \\nabla_x f = 2Ax",
  "suggestion": "Apply chain rule correctly to the quadratic form"
}
```

## Dependencies

- RFC 001 (Mistral OCR) — structured extraction needs clean math input
- Better evaluation metric (see RFC 003)

## Open questions

- How much does extraction cost? One extra LLM call per paper.
- Can we use a cheaper model (e.g. GPT-4o-mini) for extraction?
- How to handle papers without clear section structure?
