#!/usr/bin/env python3
"""
Script to convert the documentation markdown to PDF using ReportLab.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted, ListFlowable, ListItem
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re
import os

def parse_markdown_to_elements(md_content):
    """Convert markdown content to ReportLab flowables."""
    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        name='Title1',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        spaceBefore=20,
        textColor=colors.darkblue
    ))

    styles.add(ParagraphStyle(
        name='Title2',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=15,
        spaceBefore=15,
        textColor=colors.darkblue
    ))

    styles.add(ParagraphStyle(
        name='Title3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=10,
        textColor=colors.darkblue
    ))

    styles.add(ParagraphStyle(
        name='Title4',
        parent=styles['Heading4'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=8,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        backColor=colors.lightgrey,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=5,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name='BodyJustified',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        fontSize=10,
        spaceAfter=8,
        leading=14
    ))

    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.white
    ))

    elements = []
    lines = md_content.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                code_text = '\n'.join(code_lines)
                code_text = code_text.replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Preformatted(code_text, styles['CodeBlock']))
                elements.append(Spacer(1, 10))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Handle tables
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []

            # Parse table row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]

            # Skip separator row (|---|---|)
            if all(set(cell.replace('-', '').replace(':', '')) == set() for cell in cells):
                i += 1
                continue

            table_rows.append(cells)
            i += 1
            continue
        elif in_table and table_rows:
            # End of table
            if table_rows:
                # Determine column widths
                num_cols = len(table_rows[0])
                col_width = (7.0 * inch) / num_cols
                col_widths = [col_width] * num_cols

                # Create table
                t = Table(table_rows, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(t)
                elements.append(Spacer(1, 12))
            table_rows = []
            in_table = False

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Handle headers
        if line.startswith('# '):
            text = escape_html(line[2:])
            elements.append(Paragraph(text, styles['Title1']))
            elements.append(Spacer(1, 10))
        elif line.startswith('## '):
            text = escape_html(line[3:])
            elements.append(Paragraph(text, styles['Title2']))
        elif line.startswith('### '):
            text = escape_html(line[4:])
            elements.append(Paragraph(text, styles['Title3']))
        elif line.startswith('#### '):
            text = escape_html(line[5:])
            elements.append(Paragraph(text, styles['Title4']))
        elif line.startswith('---'):
            elements.append(Spacer(1, 20))
        elif line.startswith('- ') or line.startswith('* '):
            # List item
            text = process_inline_formatting(line[2:])
            elements.append(Paragraph('&bull; ' + text, styles['BodyJustified']))
        elif re.match(r'^\d+\.', line):
            # Numbered list
            text = process_inline_formatting(re.sub(r'^\d+\.\s*', '', line))
            elements.append(Paragraph(line[:line.index('.')+1] + ' ' + text, styles['BodyJustified']))
        else:
            # Regular paragraph
            text = process_inline_formatting(line)
            if text.strip():
                elements.append(Paragraph(text, styles['BodyJustified']))

        i += 1

    return elements


def escape_html(text):
    """Escape HTML special characters."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def process_inline_formatting(text):
    """Process inline markdown formatting."""
    text = escape_html(text)

    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)

    # Inline code: `code`
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="9" color="darkred">\1</font>', text)

    # Links: [text](url) - just keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'<u>\1</u>', text)

    return text


def create_pdf(md_file, pdf_file):
    """Create PDF from markdown file."""
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Parse markdown and get elements
    elements = parse_markdown_to_elements(md_content)

    # Build PDF
    doc.build(elements)
    print(f"PDF generated successfully: {pdf_file}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, 'DOCUMENTATION_3D.md')
    pdf_file = os.path.join(script_dir, 'DOCUMENTATION_3D.pdf')

    create_pdf(md_file, pdf_file)
