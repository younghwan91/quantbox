"""Convert Markdown strategy report to PDF using WeasyPrint."""
import sys
from pathlib import Path
import markdown
from weasyprint import HTML

MD_PATH = Path(__file__).resolve().parent.parent / "docs" / "backtest_results" / "quantbox_v14_strategy_report.md"
PDF_PATH = MD_PATH.with_suffix(".pdf")

CSS = """
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center {
        content: "QuantBox v14.0.0 Strategy Report — Page " counter(page) " / " counter(pages);
        font-size: 8pt;
        color: #888;
    }
}
body {
    font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #1a1a1a;
}
h1 {
    font-size: 20pt;
    color: #0d47a1;
    border-bottom: 3px solid #0d47a1;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-before: avoid;
}
h2 {
    font-size: 14pt;
    color: #1565c0;
    border-bottom: 1.5px solid #90caf9;
    padding-bottom: 4px;
    margin-top: 24px;
    page-break-after: avoid;
}
h3 {
    font-size: 12pt;
    color: #1976d2;
    margin-top: 16px;
    page-break-after: avoid;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}
th {
    background-color: #e3f2fd;
    color: #0d47a1;
    font-weight: bold;
    text-align: left;
    padding: 6px 8px;
    border: 1px solid #bbdefb;
}
td {
    padding: 5px 8px;
    border: 1px solid #e0e0e0;
}
tr:nth-child(even) {
    background-color: #fafafa;
}
tr:hover {
    background-color: #f5f5f5;
}
code {
    background-color: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}
pre {
    background-color: #263238;
    color: #eeffff;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    page-break-inside: avoid;
}
pre code {
    background: none;
    color: inherit;
    padding: 0;
}
blockquote {
    border-left: 4px solid #1565c0;
    margin: 12px 0;
    padding: 8px 16px;
    background-color: #e8f5e9;
    color: #1b5e20;
    font-style: normal;
    font-size: 9.5pt;
}
blockquote p {
    margin: 4px 0;
}
strong {
    color: #0d47a1;
}
hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 20px 0;
}
p {
    margin: 6px 0;
}
ul, ol {
    margin: 6px 0;
    padding-left: 24px;
}
li {
    margin: 3px 0;
}
/* TL;DR section highlight */
h2 + p:first-of-type {
    font-weight: normal;
}
"""


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")

    extensions = ["tables", "fenced_code", "codehilite", "toc"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(str(PDF_PATH))
    print(f"✅ PDF generated: {PDF_PATH}")
    print(f"   Size: {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
