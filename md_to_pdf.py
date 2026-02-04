# md_to_pdf.py — with page breaks per ticket
import asyncio
from pathlib import Path
import markdown
from playwright.async_api import async_playwright

MD_INPUT = Path("output/ticket_archive_minimal.md")
HTML_OUT = Path("output/ticket_archive_minimal.html")
PDF_OUT = Path("output/ticket_archive_minimal.pdf")

def build_html(md_text: str) -> str:
    """Convert Markdown → HTML and automatically add page breaks at each ticket header."""

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables"]
    )

    # Inject CSS for page breaks + styling
    css = """
    <style>
        /* Base typography */
        body {
            font-family: Arial, sans-serif;
            line-height: 1.5;
            padding: 20px;
        }

        h1 {
            font-size: 26px;
            margin-top: 40px;
            margin-bottom: 10px;
            page-break-before: always;   /* Force a NEW PAGE on each ticket */
        }

        /* Do NOT page-break on first H1 (before TOC) */
        h1:first-of-type {
            page-break-before: avoid;
        }

        h2, h3 {
            margin-top: 20px;
            margin-bottom: 10px;
        }

        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        code {
            font-size: 14px;
        }

        /* Page breaks inside HTML blocks */
        .page-break {
            page-break-before: always;
        }
    </style>
    """

    # Wrap inside a full HTML doc
    return f"""
    <html>
    <head>{css}</head>
    <body>
    {html_body}
    </body>
    </html>
    """

async def render_pdf():
    # Read markdown and build HTML
    md_text = MD_INPUT.read_text(encoding="utf-8")
    full_html = build_html(md_text)

    # Save intermediary HTML for debugging
    HTML_OUT.write_text(full_html, encoding="utf-8")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Load local HTML
        await page.goto(HTML_OUT.resolve().as_uri(), wait_until="load")

        # Generate A4 PDF with page numbers
        await page.pdf(
            path=str(PDF_OUT),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template="""
                <div style='font-size:10px; width:100%; text-align:center;'>
                    <span class='pageNumber'></span> / <span class='totalPages'></span>
                </div>
            """,
            margin={
                "top": "1cm",
                "bottom": "1.2cm",
                "left": "1cm",
                "right": "1cm",
            },
        )

        await browser.close()

    print("PDF generated:", PDF_OUT)


if __name__ == "__main__":
    asyncio.run(render_pdf())
