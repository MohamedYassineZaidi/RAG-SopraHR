import re
import os
from pathlib import Path
from datetime import datetime

INPUT_FILE = "data/raw/FR_5000.txt"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MD_FILE = OUTPUT_DIR / "ticket_archive_minimal.md"

# Improved header detection
HEADER_RE = re.compile(
    r"""
    ^\s*(\d+)\s+                # index number
    ([^\t]+?)\s+                # site_env
    ([A-Z0-9.\-]+)\s+           # offer
    ([A-Z0-9]+)\s+              # system
    ([A-Z]{2}\sW\d+)\s+         # reference (FR Wxxxxx)
    (.+)$                       # title
    """,
    re.VERBOSE
)

# Cleaning function for content lines
def clean_line(line: str) -> str:
    # Remove Lotus artifacts
    line = line.replace("::", " ")
    line = line.replace("\\t", " ")
    line = line.replace("\t", " ")
    line = line.replace("\\", "")

    # Normalize spaces
    line = re.sub(r"\s+", " ", line).strip()

    # Fix encoding issues automatically (best effort)
    line = (
        line.replace("Ã©", "é")
            .replace("Ã¨", "è")
            .replace("Ã ", "à")
            .replace("Ã§", "ç")
            .replace("â€™", "'")
            .replace("â€“", "-")
            .replace("â€œ", '"')
            .replace("â€", '"')
    )

    return line


def parse_tickets():
    tickets = []
    current = None

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            match = HEADER_RE.match(line)
            if match:
                # Save previous ticket if exists
                if current:
                    tickets.append(current)

                idx, site_env, offer, system, ref, title = match.groups()

                current = {
                    "reference": ref.strip(),
                    "title": clean_line(title),
                    "site_env": clean_line(site_env),
                    "offer": clean_line(offer),
                    "system": clean_line(system),
                    "content_lines": []
                }

            else:
                if current:
                    cleaned = clean_line(line)
                    if cleaned:
                        current["content_lines"].append(cleaned)

        # add last ticket
        if current:
            tickets.append(current)

    return tickets


def generate_markdown(tickets):
    md = []

    md.append("# Sopra HR — Ticket Archive (Clean Edition)")
    md.append(f"Generated on: **{datetime.now().strftime('%Y-%m-%d')}**\n")

    # TOC
    md.append("## Table of Contents\n")
    for t in tickets:
        anchor = t['reference'].replace(" ", "-")
        md.append(f"- [{t['reference']} — {t['title']}](#{anchor})")
    md.append("\n---\n")

    # Ticket sections
    for t in tickets:
        anchor = t['reference'].replace(" ", "-")

        md.append(f"# {t['reference']} — {t['title']}")
        md.append(f"<a name='{anchor}'></a>\n")

        md.append("### Metadata")
        md.append(f"- **Reference:** {t['reference']}")
        md.append(f"- **Site/Env:** {t['site_env']}")
        md.append(f"- **Offer:** {t['offer']}")
        md.append(f"- **System:** {t['system']}\n")

        md.append("### Ticket Content")
        md.append("```")
        md.extend(t["content_lines"])
        md.append("```")
        md.append("\n---\n")

    return "\n".join(md)


def main():
    tickets = parse_tickets()

    if not tickets:
        print("❌ ERROR: No tickets detected. Regex may need adjustment.")
        return

    print(f"✔ Detected {len(tickets)} tickets.")

    # Sort by reference for consistency
    tickets = sorted(tickets, key=lambda t: t["reference"])

    md_text = generate_markdown(tickets)
    MD_FILE.write_text(md_text, encoding="utf-8")

    print(f"✔ Clean Markdown saved to: {MD_FILE}")


if __name__ == "__main__":
    main()