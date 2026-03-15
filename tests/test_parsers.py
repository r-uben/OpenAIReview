"""Unit tests for arXiv HTML parsing."""

from reviewer.parsers import parse_arxiv_html


class _FakeResponse:
    def __init__(self, html: str, url: str):
        self._html = html
        self.url = url

    def read(self):
        return self._html.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_parse_arxiv_html_keeps_figure_caption_but_drops_image_link(monkeypatch):
    html = """
    <html>
      <head><title>Fallback title</title></head>
      <body>
        <article class="ltx_document">
          <h1 class="ltx_title_document">Test Paper</h1>
          <div class="ltx_para">Intro text.</div>
          <figure class="ltx_figure">
            <img class="ltx_graphics" src="fig1.png" alt="Figure 1 alt" width="400" />
            <figcaption class="ltx_caption">Figure 1 caption</figcaption>
          </figure>
          <div class="ltx_para">After figure text.</div>
        </article>
      </body>
    </html>
    """

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResponse(html, "https://arxiv.org/html/1234.5678"),
    )

    title, parsed = parse_arxiv_html("https://arxiv.org/html/1234.5678")

    assert title == "Test Paper"
    assert "*Figure 1 caption*" in parsed
    assert "![Figure 1 alt]" not in parsed
    assert "After figure text." in parsed


def test_parse_arxiv_html_converts_tables_to_markdown(monkeypatch):
    html = """
    <html>
      <body>
        <article class="ltx_document">
          <h1 class="ltx_title_document">Test Paper</h1>
          <figure class="ltx_table">
            <figcaption class="ltx_caption">Table 1 caption</figcaption>
            <table class="ltx_tabular">
              <tr><th>A</th><th>B</th></tr>
              <tr><td>1</td><td>2</td></tr>
            </table>
          </figure>
        </article>
      </body>
    </html>
    """

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResponse(html, "https://arxiv.org/html/1234.5678"),
    )

    _, parsed = parse_arxiv_html("https://arxiv.org/html/1234.5678")

    assert "**Table 1 caption**" in parsed
    assert "| A | B |" in parsed
    assert "| 1 | 2 |" in parsed
