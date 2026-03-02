"""Generate PDF from the R40 deep stats markdown report using fpdf2."""
from __future__ import annotations
import re
from pathlib import Path
from fpdf import FPDF, XPos, YPos

REPORT_MD = Path(__file__).resolve().parent.parent / "docs" / "backtest_results" / "748630b_20260302_r40_deep_stats.md"
OUTPUT_PDF = REPORT_MD.with_suffix(".pdf")

REG_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

MARGIN = 10
PAGE_W = 210  # A4 width mm


class ReportPDF(FPDF):
    def header(self):
        self.set_font("NotoSans", "B", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, "QuantBox R40 Deep Statistical Analysis", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("NotoSans", "", 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(re.match(r'^[-:]+$', c) for c in cells):
            continue
        rows.append(cells)
    return rows


def safe_write(pdf: FPDF, h: float, txt: str):
    """Write text with safety check for remaining horizontal space."""
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    if avail < 5:
        pdf.ln(h)
        return
    try:
        pdf.multi_cell(0, h, txt)
    except Exception:
        pdf.cell(0, h, txt[:80], new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_table(pdf: FPDF, rows: list[list[str]]):
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    usable = PAGE_W - 2 * MARGIN
    col_w = usable / n_cols

    for row_idx, row in enumerate(rows):
        while len(row) < n_cols:
            row.append("")

        if row_idx == 0:
            pdf.set_font("NotoSans", "B", 6.5)
            pdf.set_fill_color(235, 235, 245)
        else:
            pdf.set_font("NotoSans", "", 6.5)
            if row_idx % 2 == 0:
                pdf.set_fill_color(248, 248, 252)
            else:
                pdf.set_fill_color(255, 255, 255)

        for ci, cell in enumerate(row):
            text = cell[:35]
            pdf.cell(col_w, 4.5, text, border=1, fill=True)
        pdf.ln()
    pdf.ln(2)


def main():
    text = REPORT_MD.read_text(encoding="utf-8")
    lines = text.split("\n")

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_left_margin(MARGIN)
    pdf.set_right_margin(MARGIN)

    pdf.add_font("NotoSans", "", REG_FONT)
    pdf.add_font("NotoSans", "B", BOLD_FONT)

    pdf.add_page()

    i = 0
    in_table = False
    table_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # === Headings ===
        if stripped.startswith("# ") and not stripped.startswith("##"):
            pdf.set_font("NotoSans", "B", 16)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)
            safe_write(pdf, 9, stripped.lstrip("# "))
            pdf.ln(3)
            i += 1
            continue

        if stripped.startswith("## "):
            pdf.set_font("NotoSans", "B", 12)
            pdf.set_text_color(25, 25, 90)
            pdf.ln(4)
            safe_write(pdf, 7, stripped.lstrip("# "))
            pdf.ln(2)
            i += 1
            continue

        if stripped.startswith("### "):
            pdf.set_font("NotoSans", "B", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.ln(2)
            safe_write(pdf, 6, stripped.lstrip("# "))
            pdf.ln(1)
            i += 1
            continue

        # === Horizontal rule ===
        if stripped == "---":
            y = pdf.get_y()
            pdf.set_draw_color(200, 200, 200)
            pdf.line(MARGIN, y + 2, PAGE_W - MARGIN, y + 2)
            pdf.ln(5)
            i += 1
            continue

        # === Table ===
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(stripped)
            i += 1
            if i >= len(lines) or not lines[i].strip().startswith("|"):
                in_table = False
                rows = parse_table(table_lines)
                render_table(pdf, rows)
                table_lines = []
            continue

        # === Blockquote ===
        if stripped.startswith("> "):
            pdf.set_font("NotoSans", "", 8)
            pdf.set_text_color(90, 90, 90)
            content = stripped.lstrip("> ").replace("**", "")
            safe_write(pdf, 4.5, f"  \u25B6 {content}")
            pdf.set_text_color(0, 0, 0)
            i += 1
            continue

        # === Bullet points ===
        if stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("NotoSans", "", 8)
            pdf.set_text_color(0, 0, 0)
            content = stripped[2:].replace("**", "").replace("`", "")
            safe_write(pdf, 4.5, f"  \u2022 {content}")
            i += 1
            continue

        # === Bold metadata ===
        if stripped.startswith("**") and ":**" in stripped:
            pdf.set_font("NotoSans", "", 8)
            pdf.set_text_color(0, 0, 0)
            clean = stripped.replace("**", "").replace("`", "")
            pdf.cell(0, 4.5, clean, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            i += 1
            continue

        # === Empty line ===
        if not stripped:
            pdf.ln(1.5)
            i += 1
            continue

        # === Regular text ===
        pdf.set_font("NotoSans", "", 8)
        pdf.set_text_color(0, 0, 0)
        clean = stripped.replace("**", "").replace("`", "").replace("*", "")
        safe_write(pdf, 4.5, clean)
        i += 1

    pdf.output(str(OUTPUT_PDF))
    print(f"PDF saved: {OUTPUT_PDF}")
    print(f"Pages: {pdf.pages_count}")
    print(f"Size: {OUTPUT_PDF.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
