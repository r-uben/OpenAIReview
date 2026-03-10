# RFC 001: Mistral OCR for PDF parsing

**Status**: implemented (draft)
**Issue**: improves PDF parsing for all review methods

## Problem

Current PDF pipeline: Marker (heavy, needs GPU) or PyMuPDF (loses all math).
Most users will submit PDFs. If math is garbled, the reviewer LLM cannot catch
math errors, notation issues, or formula inconsistencies — which are ~30% of
ground-truth issues in the benchmark.

## Approach

Add Mistral OCR as the primary PDF parser. Sends the full PDF to Mistral's API,
gets back clean markdown with preserved LaTeX, tables, and structure.

- Cost: ~$0.001/page (~$0.03 for a 30-page paper)
- No local dependencies beyond `mistralai` SDK
- Fallback chain: Mistral OCR → Marker → PyMuPDF

## What's done

- `_parse_pdf_mistral()` in `parsers.py`
- `--ocr {mistral,marker,pymupdf}` CLI flag
- Auto-detection via `MISTRAL_API_KEY`

## Open questions

- Run ablation: same paper through all 3 parsers, compare review quality
- Should we add other cloud OCR options (e.g. Gemini vision)?
- File size limits: Mistral caps at 50MB — handle gracefully?
