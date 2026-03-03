"""Document parsers for PDF, DOCX, TEX, TXT, and MD files."""

import re
from pathlib import Path


def parse_document(file_path: str | Path) -> tuple[str, str]:
    """Parse a document file and return (title, full_text).

    Supported formats: .pdf, .docx, .tex, .txt, .md
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix == ".tex":
        return _parse_tex(path)
    elif suffix in (".txt", ".md", ".markdown"):
        return _parse_text(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _parse_pdf(path: Path) -> tuple[str, str]:
    """Extract text from PDF using pymupdf."""
    import pymupdf

    doc = pymupdf.open(str(path))
    pages = []
    title = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append(text)

        if page_num == 0 and not title:
            # Try to find title from largest font text on first page
            blocks = page.get_text("dict")["blocks"]
            best_size = 0
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > best_size and span["text"].strip():
                            best_size = span["size"]
                            title = span["text"].strip()

    doc.close()
    full_text = "\n\n".join(pages)

    if not title:
        # Fallback: first non-empty line
        for line in full_text.split("\n"):
            if line.strip():
                title = line.strip()[:200]
                break

    return title, full_text


def _parse_docx(path: Path) -> tuple[str, str]:
    """Extract text from DOCX using python-docx."""
    import docx

    doc = docx.Document(str(path))
    paragraphs = []
    title = ""

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        paragraphs.append(text)
        if not title and para.style and para.style.name.startswith("Heading"):
            title = text

    full_text = "\n\n".join(paragraphs)

    if not title and paragraphs:
        title = paragraphs[0][:200]

    return title, full_text


def _parse_tex(path: Path) -> tuple[str, str]:
    """Extract text from LaTeX source."""
    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""

    # Extract title from \title{...}
    title_match = re.search(r"\\title\{([^}]+)\}", text)
    if title_match:
        title = title_match.group(1).strip()

    if not title:
        for line in text.split("\n"):
            if line.strip():
                title = line.strip()[:200]
                break

    return title, text


def _parse_text(path: Path) -> tuple[str, str]:
    """Extract text from plain text or markdown."""
    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Markdown heading
        if stripped.startswith("#"):
            title = stripped.lstrip("# ").strip()
        else:
            title = stripped[:200]
        break

    return title, text
