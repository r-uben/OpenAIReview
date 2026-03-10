# RFC 003: Evaluation metric redesign

**Status**: proposal
**Issue**: benchmark recall is 4-6% but likely undercounting real hits

## Problem

Current evaluation uses fuzzy string similarity between generated comments
and ground-truth quotes. Models find the right passages but paraphrase them,
so the metric fails to credit correct issue identification.

With 52 gold comments across 4 papers, one extra match changes recall by ~2%.
The benchmark is both noisy and biased downward.

## Approach

### LLM-judge matching

Replace string similarity with an LLM judge that evaluates:

1. **Issue identity**: does the generated comment identify the same issue
   as the gold comment? (yes/no/partial)
2. **Location accuracy**: does it point to the same passage/equation?
3. **Correctness**: is the criticism actually valid?

### Oracle ablations

Run controlled experiments to isolate bottlenecks:

- **Gold passage → reviewer**: if recall jumps, localization is the bottleneck
- **Same reviewer, different parsers**: isolates OCR impact
- **Same reviewer, different models**: isolates model capability

### Expand benchmark

- More papers (target: 15-20 papers, 200+ comments)
- Diverse fields (math-heavy, empirical, theoretical)
- Multiple annotators for inter-rater reliability

## Open questions

- Which LLM for judging? Needs to be different from the reviewer to avoid bias.
- Cost of LLM-judge evaluation per run?
- Can we get annotations from the Refine platform or do we need manual labeling?
