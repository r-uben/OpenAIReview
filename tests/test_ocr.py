"""Tests for OCR postprocessing and parse_document OCR-related behavior."""

import tempfile
from pathlib import Path

from reviewer.ocr_postprocess import fix_ocr_notation


class TestFixOcrNotation:
    """Test fix_ocr_notation detects and fixes OCR misreads."""

    def test_singleton_hat_t_corrected_to_hat_i(self):
        """If \\hat{t} appears once but \\hat{i} appears multiple times, fix it."""
        text = r"We define $\hat{i}_t$ and $\hat{i}_s$ and $\hat{t}_k$."
        fixed, corrections = fix_ocr_notation(text)
        assert r"\hat{i}_k" in fixed
        assert len(corrections) == 1
        assert corrections[0]["old"] == r"\hat{t}"

    def test_singleton_hat_with_space(self):
        """OCR sometimes inserts a space: \\hat {t} instead of \\hat{t}."""
        text = r"$\hat{i}_t$ and $\hat{i}_s$ and $\hat {t}_k$."
        fixed, corrections = fix_ocr_notation(text)
        assert r"\hat{i}_k" in fixed
        assert len(corrections) == 1

    def test_no_correction_when_both_frequent(self):
        """Don't fix if both symbols appear equally often."""
        text = r"$\hat{i}_t$ and $\hat{t}_s$."
        fixed, corrections = fix_ocr_notation(text)
        assert fixed == text
        assert len(corrections) == 0

    def test_no_correction_for_non_confusable(self):
        """Don't fix symbols that aren't visually confusable."""
        text = r"$\hat{a}_t$ and $\hat{z}_s$ and $\hat{z}_k$."
        fixed, corrections = fix_ocr_notation(text)
        assert fixed == text
        assert len(corrections) == 0

    def test_bar_accent_also_works(self):
        """Fix works for \\bar as well, not just \\hat."""
        text = r"$\bar{i}_t$ and $\bar{i}_s$ and $\bar{t}_k$."
        fixed, corrections = fix_ocr_notation(text)
        assert r"\bar{i}_k" in fixed

    def test_empty_text(self):
        text = ""
        fixed, corrections = fix_ocr_notation(text)
        assert fixed == ""
        assert corrections == []


class TestParseTextFrontmatter:
    """Test that _parse_text detects OCR frontmatter."""

    def test_ocr_frontmatter_detected(self):
        from reviewer.parsers import _parse_text

        content = '---\ntitle: "Test Paper"\nocr_engine: "mistral"\n---\n\n# Hello\n\nBody text.'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            title, text, was_ocr = _parse_text(Path(f.name))
        assert title == "Test Paper"
        assert was_ocr is True
        assert "# Hello" in text

    def test_no_frontmatter(self):
        from reviewer.parsers import _parse_text

        content = "# Regular Paper\n\nNo frontmatter here."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            title, text, was_ocr = _parse_text(Path(f.name))
        assert title == "Regular Paper"
        assert was_ocr is False
