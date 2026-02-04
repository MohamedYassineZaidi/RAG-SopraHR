#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

# Input directory with your per-ticket files
TICKETS_DIR = Path("data/output")

# Output CSVs
COUNTS_CSV = Path("data/closing_counts.csv")
DETAIL_CSV = Path("data/closing_occurrences.csv")  # optional, can be commented out

# Same pattern used in your pipeline for explicit closing lines:
# Example line: "\t\tCP\t3\tPAPFRINT"
CLOSING_RE = re.compile(r'^\t\t([A-Z]{2})\t(\d+)\t([A-Z0-9]+)$')

def iter_ticket_files(root: Path):
    # All .txt ticket files
    for p in root.rglob("*.txt"):
        if p.is_file():
            yield p

def main():
    counts = Counter()  # counts per closing_code (3rd group: e.g. PAPFRINT)
    detail_rows = []    # detailed occurrences (optional)

    total_files = 0
    total_hits  = 0

    for f in iter_ticket_files(TICKETS_DIR):
        total_files += 1
        # We read line by line to avoid loading big files entirely
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    line = raw.rstrip("\n")
                    m = CLOSING_RE.match(line)
                    if m:
                        status_code = m.group(1)  # e.g. CP
                        number      = m.group(2)  # e.g. 3
                        closing_code= m.group(3)  # e.g. PAPFRINT (we aggregate on this)
                        counts[closing_code] += 1
                        total_hits += 1

                        # store optional detail line
                        detail_rows.append({
                            "file_name": f.name,
                            "path": str(f),
                            "status": status_code,
                            "number": number,
                            "closing_code": closing_code,
                            "raw_line": line
                        })
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")

    # Write aggregated counts
    COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COUNTS_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["closing_code", "count"])
        for code, cnt in counts.most_common():
            w.writerow([code, cnt])

    # Write detailed occurrences (optional)
    if detail_rows:
        with DETAIL_CSV.open("w", newline="", encoding="utf-8") as out:
            fieldnames = ["file_name", "path", "status", "number", "closing_code", "raw_line"]
            w = csv.DictWriter(out, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(detail_rows)

    print(f"[done] scanned_files={total_files}  explicit_closing_hits={total_hits}")
    print(f"[done] counts -> {COUNTS_CSV}")
    if detail_rows:
        print(f"[done] occurrences -> {DETAIL_CSV}")

if __name__ == "__main__":
    main()

