"""Method 3: RAG-based review.

A small model retrieves relevant context for each paragraph, then a large
model checks correctness of each paragraph.

Original variants:
  rag_local          - ±3 window context (baseline)
  rag_retrieved      - small-model retrieval of relevant context
  rag_retrieved_cot  - retrieved context + chain-of-thought deep-check
  rag_top_k_filter   - retrieved context + score-and-rank paragraph filter

Chunked variants (merge small paragraphs into ~500-token chunks):
  rag_chunked        - chunks, filter, window=3
  rag_chunked_w5     - chunks, filter, window=5
  rag_chunked_defs   - chunks, filter, window=3, definitions prefix
  rag_chunked_2pass  - chunks, two-pass section filter, window=3
  rag_chunked_full   - chunks, two-pass, window=5, definitions prefix
"""

import json
import re

from .client import chat
from .models import ReviewResult
from .utils import count_tokens, locate_comment_in_document, parse_comments_from_list

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FILTER_PROMPT = """\
You are screening paragraphs of an academic paper. \
Identify which of the following paragraphs MOST LIKELY contain a technical or logical mistake.

A paragraph is worth checking if it:
- Contains mathematical notation, equations, or formulas
- Makes a specific quantitative or logical claim
- Describes a proof, theorem, or algorithm
- Makes a comparative or causal claim
- Uses notation that could be inconsistent with earlier definitions

Return ONLY a JSON array of paragraph indices (0-based) that should be checked. \
Include at most 15 paragraphs. Example: [0, 3, 5, 7]

PARAGRAPHS:
{paragraphs}
"""

SCORING_PROMPT = """\
You are screening paragraphs of an academic paper for likely mistakes.

Score each paragraph from 1 (very unlikely to contain a mistake) to 10 \
(very likely to contain a technical or logical mistake).

Focus on paragraphs with:
- Mathematical notation, equations, or formulas
- Specific quantitative or logical claims
- Proofs, theorems, or algorithms
- Comparative or causal claims
- Notation that may be inconsistently defined

Return ONLY a JSON array of objects, one per paragraph, in order:
[{{"index": 0, "score": 3}}, {{"index": 1, "score": 8}}, ...]

PARAGRAPHS:
{paragraphs}
"""

RETRIEVAL_PROMPT = """\
Given this TARGET PARAGRAPH (index {target_idx}), identify which other paragraphs \
from the paper provide the most relevant context for checking its correctness.

Include paragraphs that contain: definitions of symbols used, theorems it relies on, \
prior derivations whose results it uses, assumptions that govern its claims.

TARGET PARAGRAPH:
{target_paragraph}

ALL PARAGRAPHS (truncated to 300 chars each):
{all_paragraphs_summary}

Return ONLY a JSON array of paragraph indices (at most 8). Do not include {target_idx}. \
Example: [2, 5, 11]
"""

DEEP_CHECK_PROMPT = """\
You are a thoughtful reviewer checking a passage from an academic paper. \
Engage deeply with the material. For each potential issue, first try to understand the authors' \
intent and check whether your concern is resolved by context before flagging it.

FULL PAPER CONTEXT (relevant sections):
{context}

---

PASSAGE TO CHECK:
{passage}

---

Check for:
1. Mathematical / formula errors: wrong formulas, sign errors, missing factors, incorrect derivations, subscript or index errors
2. Notation inconsistencies: symbols used in a way that contradicts their earlier definition
3. Inconsistency between text and formal definitions: prose says one thing but the equation says another
4. Parameter / numerical inconsistencies: stated values contradict what can be derived from definitions or tables elsewhere
5. Insufficient justification: a key derivation step is skipped where the result is non-trivial
6. Questionable claims: statements that overstate what has actually been shown
7. Ambiguity that could mislead: flag only if a careful reader could reasonably reach an incorrect conclusion
8. Underspecified methods: an algorithm, procedure, or modification is described too vaguely for a reader to reproduce — key choices, boundary conditions, or parameter settings are left implicit

For each issue, write like a careful reader thinking aloud. Describe what initially confused or \
concerned you, what you checked to resolve it, and what specifically remains problematic. \
Acknowledge what the authors got right before noting the issue. Reference standard results \
or conventions in the field when relevant.

Be lenient with:
- Introductory and overview sections, which intentionally simplify or gloss over details
- Forward references — symbols or claims that may be defined or justified later in the paper
- Informal prose that paraphrases a formal result without repeating every qualifier

Do NOT flag:
- Formatting, typesetting, or capitalization issues
- References to equations or sections not shown in the context (they exist elsewhere)
- Incomplete text at passage boundaries
- Trivial observations that any reader in the field would immediately resolve

Return ONLY a JSON array (can be []). Each item:
- "title": concise title of the issue
- "quote": the exact verbatim text (preserving LaTeX)
- "explanation": deep reasoning — what you initially thought, whether context resolves it, and what specifically remains problematic
- "type": "technical" or "logical"
"""

OVERALL_FEEDBACK_PROMPT = """\
You are an expert academic reviewer. Based on the beginning of the paper below, \
write one paragraph of high-level feedback on the paper's quality, clarity, \
and most significant issues.

PAPER (first 8000 characters):
{paper_start}
"""

SECTION_SELECT_PROMPT = """\
You are screening an academic paper for sections most likely to contain technical or logical mistakes.

Below is the paper outline (section headers and first line of each section). \
Select 5-8 sections that are most likely to contain mistakes worth checking.

Focus on sections with proofs, derivations, algorithms, or complex technical claims. \
Skip introductions, related work, and conclusions unless they make novel claims.

OUTLINE:
{outline}

Return ONLY a JSON array of section indices (0-based). Example: [1, 3, 5, 7]
"""

# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

RAG_VARIANTS = {
    "rag_local": {
        "context_strategy": "window",
        "filter_strategy": "filter",
        "cot": False,
        "use_chunks": False,
        "window_size": 3,
        "defs_prefix": False,
    },
    "rag_local_defs": {
        "context_strategy": "window",
        "filter_strategy": "filter",
        "cot": False,
        "use_chunks": False,
        "window_size": 3,
        "defs_prefix": True,
    },
    "rag_retrieved": {
        "context_strategy": "retrieved",
        "filter_strategy": "filter",
        "cot": False,
        "use_chunks": False,
        "window_size": 3,
        "defs_prefix": False,
    },
    "rag_retrieved_cot": {
        "context_strategy": "retrieved",
        "filter_strategy": "filter",
        "cot": True,
        "use_chunks": False,
        "window_size": 3,
        "defs_prefix": False,
    },
    "rag_top_k_filter": {
        "context_strategy": "retrieved",
        "filter_strategy": "top_k",
        "cot": False,
        "use_chunks": False,
        "window_size": 3,
        "defs_prefix": False,
    },
    # --- Chunked variants ---
    "rag_chunked": {
        "context_strategy": "window",
        "filter_strategy": "filter",
        "cot": False,
        "use_chunks": True,
        "window_size": 3,
        "defs_prefix": False,
    },
    "rag_chunked_w5": {
        "context_strategy": "window",
        "filter_strategy": "filter",
        "cot": False,
        "use_chunks": True,
        "window_size": 5,
        "defs_prefix": False,
    },
    "rag_chunked_defs": {
        "context_strategy": "window",
        "filter_strategy": "filter",
        "cot": False,
        "use_chunks": True,
        "window_size": 3,
        "defs_prefix": True,
    },
    "rag_chunked_2pass": {
        "context_strategy": "window",
        "filter_strategy": "two_pass",
        "cot": False,
        "use_chunks": True,
        "window_size": 3,
        "defs_prefix": False,
    },
    "rag_chunked_full": {
        "context_strategy": "window",
        "filter_strategy": "two_pass",
        "cot": False,
        "use_chunks": True,
        "window_size": 5,
        "defs_prefix": True,
    },
    # --- No-filter variants (check every chunk) ---
    "rag_nofilter": {
        "context_strategy": "window",
        "filter_strategy": "none",
        "cot": False,
        "use_chunks": True,
        "window_size": 3,
        "defs_prefix": False,
    },
    "rag_nofilter_w5": {
        "context_strategy": "window",
        "filter_strategy": "none",
        "cot": False,
        "use_chunks": True,
        "window_size": 5,
        "defs_prefix": False,
    },
    "rag_nofilter_defs": {
        "context_strategy": "window",
        "filter_strategy": "none",
        "cot": False,
        "use_chunks": True,
        "window_size": 3,
        "defs_prefix": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str, min_chars: int = 100) -> list[str]:
    """Split document into paragraphs, merging short ones with the next."""
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs: list[str] = []
    carry = ""
    for p in raw:
        if carry:
            p = carry + "\n\n" + p
            carry = ""
        if len(p) < min_chars:
            carry = p
        else:
            paragraphs.append(p)
    if carry:
        if paragraphs:
            paragraphs[-1] = paragraphs[-1] + "\n\n" + carry
        else:
            paragraphs.append(carry)
    return paragraphs


def _parse_index_array(response: str) -> list[int]:
    match = re.search(r"\[[\d,\s]*\]", response)
    if match:
        try:
            indices = json.loads(match.group(0))
            return [i for i in indices if isinstance(i, int)]
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Chunking infrastructure
# ---------------------------------------------------------------------------

def merge_into_chunks(
    paragraphs: list[str],
    target_chars: int = 4000,
) -> list[tuple[list[int], str]]:
    """Merge adjacent paragraphs into chunks of ~target_chars (~1000 tokens).

    Returns list of (paragraph_indices, merged_text) tuples.
    """
    chunks: list[tuple[list[int], str]] = []
    current_indices: list[int] = []
    current_text = ""

    for i, para in enumerate(paragraphs):
        if current_text and len(current_text) + len(para) > target_chars:
            chunks.append((current_indices, current_text))
            current_indices = []
            current_text = ""
        current_indices.append(i)
        current_text = (current_text + "\n\n" + para).strip()

    if current_text:
        chunks.append((current_indices, current_text))

    return chunks


def get_definitions_context(
    paragraphs: list[str],
    up_to_para: int | None = None,
    max_tokens: int = 4000,
) -> str:
    """Extract paragraphs containing definitions, notation, assumptions, and equations.

    If *up_to_para* is given, only include content from paragraphs
    0..up_to_para-1 (i.e. those encountered before the current passage).
    """
    keywords = [
        "definition", "denote", "let ", "theorem", "assume",
        "notation", "defined as", "we define", "suppose",
    ]
    # Patterns that indicate equations / formal expressions
    equation_patterns = [
        "\\begin{equation", "\\begin{align", "\\begin{eqnarray",
        "\\[", "$$",
    ]
    limit = up_to_para if up_to_para is not None else len(paragraphs)
    def_paragraphs = []
    for i in range(min(limit, len(paragraphs))):
        para_lower = paragraphs[i].lower()
        has_keyword = any(kw in para_lower for kw in keywords)
        has_equation = any(pat in paragraphs[i] for pat in equation_patterns)
        if has_keyword or has_equation:
            def_paragraphs.append(f"[para {i}] {paragraphs[i]}")

    context = "\n\n".join(def_paragraphs)
    if count_tokens(context) > max_tokens:
        context = context[:max_tokens * 4]
    return context


def _extract_sections(
    chunks: list[tuple[list[int], str]],
) -> list[tuple[int, str, str]]:
    """Extract (chunk_index, header, first_line) for outline generation.

    Looks for markdown headers (# ...) or all-caps lines as section headers.
    """
    sections: list[tuple[int, str, str]] = []
    for ci, (_, text) in enumerate(chunks):
        lines = text.strip().split("\n")
        header = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or (stripped.isupper() and len(stripped) > 5):
                header = stripped
                break
        if header is None:
            header = lines[0].strip()[:80] if lines else f"Chunk {ci}"
        first_content = ""
        for line in lines[1:]:
            if line.strip():
                first_content = line.strip()[:120]
                break
        sections.append((ci, header, first_content))
    return sections


# ---------------------------------------------------------------------------
# Filter strategies
# ---------------------------------------------------------------------------

def filter_paragraphs(
    paragraphs: list[str],
    small_model: str,
    result: ReviewResult,
    defs_context: str = "",
) -> list[int]:
    """Use small model to identify paragraphs worth deep-checking (filter strategy)."""
    batch_size = 50
    selected = []

    for batch_start in range(0, len(paragraphs), batch_size):
        batch = paragraphs[batch_start : batch_start + batch_size]
        formatted_batch = "\n\n".join(
            f"[{batch_start + i}] {p[:500]}" for i, p in enumerate(batch)
        )
        prompt = FILTER_PROMPT.format(paragraphs=formatted_batch)
        if defs_context:
            prompt = f"DEFINITIONS AND NOTATION:\n{defs_context}\n\n---\n\n{prompt}"
        response, usage = chat(
            messages=[{"role": "user", "content": prompt}],
            model=small_model,
            max_tokens=512,
        )
        result.total_prompt_tokens += usage["prompt_tokens"]
        result.total_completion_tokens += usage["completion_tokens"]
        selected.extend(_parse_index_array(response))

    return sorted(set(selected))


def filter_chunks(
    chunks: list[tuple[list[int], str]],
    small_model: str,
    result: ReviewResult,
    defs_context: str = "",
) -> list[int]:
    """Use small model to filter chunks (same logic as filter_paragraphs but on chunks)."""
    texts = [text for _, text in chunks]
    batch_size = 30  # chunks are bigger than paragraphs

    selected = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        formatted_batch = "\n\n".join(
            f"[{batch_start + i}] {t[:800]}" for i, t in enumerate(batch)
        )
        prompt = FILTER_PROMPT.format(paragraphs=formatted_batch)
        if defs_context:
            prompt = f"DEFINITIONS AND NOTATION:\n{defs_context}\n\n---\n\n{prompt}"
        response, usage = chat(
            messages=[{"role": "user", "content": prompt}],
            model=small_model,
            max_tokens=512,
        )
        result.total_prompt_tokens += usage["prompt_tokens"]
        result.total_completion_tokens += usage["completion_tokens"]
        selected.extend(_parse_index_array(response))

    return sorted(set(i for i in selected if 0 <= i < len(chunks)))


def section_filter(
    chunks: list[tuple[list[int], str]],
    small_model: str,
    result: ReviewResult,
) -> list[int]:
    """Two-pass filter: select sections first, then filter chunks within those sections."""
    # Pass 1: build outline and select relevant sections
    sections = _extract_sections(chunks)
    outline = "\n".join(
        f"[{ci}] {header} — {first_line}" for ci, header, first_line in sections
    )
    prompt = SECTION_SELECT_PROMPT.format(outline=outline)
    response, usage = chat(
        messages=[{"role": "user", "content": prompt}],
        model=small_model,
        max_tokens=256,
    )
    result.total_prompt_tokens += usage["prompt_tokens"]
    result.total_completion_tokens += usage["completion_tokens"]

    selected_sections = set(_parse_index_array(response))
    if not selected_sections:
        # Fallback: use all chunks
        selected_sections = set(range(len(chunks)))

    # Pass 2: filter only within selected sections
    section_chunks = [(i, chunks[i]) for i in sorted(selected_sections) if i < len(chunks)]
    texts = [text for _, (_, text) in section_chunks]
    indices = [i for i, _ in section_chunks]

    if not texts:
        return []

    batch_size = 30
    selected = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        batch_indices = indices[batch_start : batch_start + batch_size]
        formatted_batch = "\n\n".join(
            f"[{batch_indices[j]}] {t[:800]}" for j, t in enumerate(batch)
        )
        prompt = FILTER_PROMPT.format(paragraphs=formatted_batch)
        response, usage = chat(
            messages=[{"role": "user", "content": prompt}],
            model=small_model,
            max_tokens=512,
        )
        result.total_prompt_tokens += usage["prompt_tokens"]
        result.total_completion_tokens += usage["completion_tokens"]
        selected.extend(_parse_index_array(response))

    return sorted(set(i for i in selected if 0 <= i < len(chunks)))


def top_k_filter_paragraphs(
    paragraphs: list[str],
    small_model: str,
    result: ReviewResult,
    top_k: int = 10,
) -> list[int]:
    """Use small model to score paragraphs and return top-k indices."""
    batch_size = 50
    scores: dict[int, float] = {}

    for batch_start in range(0, len(paragraphs), batch_size):
        batch = paragraphs[batch_start : batch_start + batch_size]
        formatted_batch = "\n\n".join(
            f"[{batch_start + i}] {p[:500]}" for i, p in enumerate(batch)
        )
        prompt = SCORING_PROMPT.format(paragraphs=formatted_batch)
        response, usage = chat(
            messages=[{"role": "user", "content": prompt}],
            model=small_model,
            max_tokens=1024,
        )
        result.total_prompt_tokens += usage["prompt_tokens"]
        result.total_completion_tokens += usage["completion_tokens"]

        # Parse [{index: N, score: N}, ...]
        try:
            arr_match = re.search(r"\[.*\]", response, re.DOTALL)
            if arr_match:
                items = json.loads(arr_match.group(0))
                for item in items:
                    if isinstance(item, dict) and "index" in item and "score" in item:
                        scores[int(item["index"])] = float(item["score"])
        except (json.JSONDecodeError, ValueError):
            pass

    sorted_indices = sorted(scores, key=lambda i: scores[i], reverse=True)
    return sorted(sorted_indices[:top_k])


# ---------------------------------------------------------------------------
# Context strategies
# ---------------------------------------------------------------------------

def get_window_context(
    paragraphs: list[str],
    idx: int,
    window: int = 3,
    max_tokens: int = 4000,
) -> str:
    """Get surrounding paragraphs as context (asymmetric window: more before, less after)."""
    before = window + 2  # e.g. window=3 → 5 before
    after = max(1, window - 1)  # e.g. window=3 → 2 after
    start = max(0, idx - before)
    end = min(len(paragraphs), idx + after + 1)
    context_parts = []
    for i in range(start, end):
        marker = ">>> " if i == idx else "    "
        context_parts.append(f"{marker}[para {i}] {paragraphs[i]}")
    context = "\n\n".join(context_parts)
    if count_tokens(context) > max_tokens:
        context = context[: max_tokens * 4]
    return context


def get_chunk_window_context(
    chunks: list[tuple[list[int], str]],
    chunk_idx: int,
    window: int = 3,
    max_tokens: int = 6000,
) -> str:
    """Get surrounding passages as context (asymmetric window: more before, less after)."""
    before = window + 2  # e.g. window=3 → 5 before
    after = max(1, window - 1)  # e.g. window=3 → 2 after
    start = max(0, chunk_idx - before)
    end = min(len(chunks), chunk_idx + after + 1)
    context_parts = []
    for i in range(start, end):
        _, text = chunks[i]
        marker = ">>> " if i == chunk_idx else "    "
        context_parts.append(f"{marker}[section {i}] {text}")
    context = "\n\n".join(context_parts)
    if count_tokens(context) > max_tokens:
        context = context[: max_tokens * 4]
    return context


def retrieve_context_for_paragraph(
    paragraphs: list[str],
    target_idx: int,
    small_model: str,
    result: ReviewResult,
    max_tokens: int = 6000,
) -> str:
    """Use small model to retrieve relevant paragraphs for a target paragraph."""
    all_summary = "\n".join(
        f"[{i}] {p[:300]}" for i, p in enumerate(paragraphs)
    )
    prompt = RETRIEVAL_PROMPT.format(
        target_idx=target_idx,
        target_paragraph=paragraphs[target_idx],
        all_paragraphs_summary=all_summary,
    )
    response, usage = chat(
        messages=[{"role": "user", "content": prompt}],
        model=small_model,
        max_tokens=256,
    )
    result.total_prompt_tokens += usage["prompt_tokens"]
    result.total_completion_tokens += usage["completion_tokens"]

    retrieved_indices = _parse_index_array(response)
    # Always include ±1 local neighbors as fallback
    neighbors = [i for i in (target_idx - 1, target_idx + 1) if 0 <= i < len(paragraphs)]
    context_indices = sorted(set(retrieved_indices + neighbors) - {target_idx})

    context_parts = []
    for i in context_indices:
        context_parts.append(f"[para {i}] {paragraphs[i]}")
    context_parts.append(f">>> [para {target_idx}] {paragraphs[target_idx]}")
    context = "\n\n".join(context_parts)
    if count_tokens(context) > max_tokens:
        context = context[: max_tokens * 4]
    return context


# ---------------------------------------------------------------------------
# Main review function
# ---------------------------------------------------------------------------

def review_rag(
    paper_slug: str,
    document_content: str,
    small_model: str = "anthropic/claude-haiku-4-5",
    large_model: str = "anthropic/claude-opus-4-5",
    variant: str = "rag_retrieved",
) -> ReviewResult:
    if variant not in RAG_VARIANTS:
        raise ValueError(f"Unknown RAG variant: {variant}. Choose from {list(RAG_VARIANTS)}")

    config = RAG_VARIANTS[variant]
    result = ReviewResult(
        method=variant,
        paper_slug=paper_slug,
        model=f"{small_model}+{large_model}",
    )

    paragraphs = split_into_paragraphs(document_content)
    use_chunks = config.get("use_chunks", False)
    window_size = config.get("window_size", 3)
    defs_prefix = config.get("defs_prefix", False)

    print(f"  RAG [{variant}]: {len(paragraphs)} paragraphs total")

    # Defs prefix is now computed per-passage (scoped to preceding paragraphs)
    defs_prefix_enabled = defs_prefix

    if use_chunks:
        # --- Chunked path ---
        chunks = merge_into_chunks(paragraphs)
        print(f"  RAG [{variant}]: {len(chunks)} chunks (from {len(paragraphs)} paragraphs)")

        # Step 1: filter chunks (or skip filtering)
        if config["filter_strategy"] == "none":
            candidate_chunk_indices = list(range(len(chunks)))
            print(f"  RAG [{variant}]: no filter, checking all {len(candidate_chunk_indices)} chunks")
        elif config["filter_strategy"] == "two_pass":
            candidate_chunk_indices = section_filter(chunks, small_model, result)
            print(f"  RAG [{variant}]: {len(candidate_chunk_indices)} candidate chunks selected")
        else:
            # Pass defs for filtering (use all paragraphs since filter sees the whole paper)
            filter_defs = get_definitions_context(paragraphs) if defs_prefix_enabled else ""
            candidate_chunk_indices = filter_chunks(chunks, small_model, result, defs_context=filter_defs)
            print(f"  RAG [{variant}]: {len(candidate_chunk_indices)} candidate chunks selected")

        # Step 2: deep-check each candidate chunk
        deep_check_prompt_template = DEEP_CHECK_PROMPT
        all_comments = []

        for chunk_idx in candidate_chunk_indices:
            para_indices, chunk_text = chunks[chunk_idx]

            context = get_chunk_window_context(chunks, chunk_idx, window=window_size)
            if defs_prefix_enabled:
                # Only include definitions from before this chunk
                first_para = para_indices[0]
                defs_context = get_definitions_context(paragraphs, up_to_para=first_para)
                if defs_context:
                    context = f"DEFINITIONS AND NOTATION:\n{defs_context}\n\n---\n\n{context}"

            prompt = deep_check_prompt_template.format(context=context, passage=chunk_text)
            response, usage = chat(
                messages=[{"role": "user", "content": prompt}],
                model=large_model,
                max_tokens=4096,
            )
            result.raw_responses.append(response)
            result.total_prompt_tokens += usage["prompt_tokens"]
            result.total_completion_tokens += usage["completion_tokens"]

            arr_match = re.search(r"\[.*\]", response, re.DOTALL)
            if arr_match:
                try:
                    items = json.loads(arr_match.group(0))
                    new_comments = parse_comments_from_list(items)
                    # Locate each comment's quote within the chunk's paragraphs
                    chunk_paras = [paragraphs[i] for i in para_indices]
                    for c in new_comments:
                        located = locate_comment_in_document(c.quote, chunk_paras)
                        if located is not None and located < len(para_indices):
                            c.paragraph_index = para_indices[located]
                        else:
                            c.paragraph_index = para_indices[0]
                    all_comments.extend(new_comments)
                except json.JSONDecodeError:
                    pass

        result.comments = all_comments

    else:
        # --- Original paragraph path ---
        # Step 1: filter/select candidate paragraphs
        if config["filter_strategy"] == "top_k":
            candidate_indices = top_k_filter_paragraphs(paragraphs, small_model, result)
        else:
            filter_defs = get_definitions_context(paragraphs) if defs_prefix_enabled else ""
            candidate_indices = filter_paragraphs(paragraphs, small_model, result, defs_context=filter_defs)
        print(f"  RAG [{variant}]: {len(candidate_indices)} candidates selected")

        # Step 2: deep-check each candidate with context from large model
        deep_check_prompt_template = DEEP_CHECK_PROMPT
        all_comments = []

        for idx in candidate_indices:
            if config["context_strategy"] == "retrieved":
                context = retrieve_context_for_paragraph(paragraphs, idx, small_model, result)
            else:
                context = get_window_context(paragraphs, idx, window=window_size)

            if defs_prefix_enabled:
                defs_context = get_definitions_context(paragraphs, up_to_para=idx)
                if defs_context:
                    context = f"DEFINITIONS AND NOTATION:\n{defs_context}\n\n---\n\n{context}"

            passage = paragraphs[idx]
            prompt = deep_check_prompt_template.format(context=context, passage=passage)
            response, usage = chat(
                messages=[{"role": "user", "content": prompt}],
                model=large_model,
                max_tokens=4096,
            )
            result.raw_responses.append(response)
            result.total_prompt_tokens += usage["prompt_tokens"]
            result.total_completion_tokens += usage["completion_tokens"]

            # Parse JSON array from response (may follow CoT reasoning)
            arr_match = re.search(r"\[.*\]", response, re.DOTALL)
            if arr_match:
                try:
                    items = json.loads(arr_match.group(0))
                    new_comments = parse_comments_from_list(items)
                    for c in new_comments:
                        c.paragraph_index = idx
                    all_comments.extend(new_comments)
                except json.JSONDecodeError:
                    pass

        result.comments = all_comments

    # Step 3: generate overall feedback using large model
    paper_start = document_content[:8000]
    feedback_response, usage = chat(
        messages=[{"role": "user", "content": OVERALL_FEEDBACK_PROMPT.format(paper_start=paper_start)}],
        model=large_model,
        max_tokens=512,
    )
    result.overall_feedback = feedback_response.strip()
    result.total_prompt_tokens += usage["prompt_tokens"]
    result.total_completion_tokens += usage["completion_tokens"]

    return result
