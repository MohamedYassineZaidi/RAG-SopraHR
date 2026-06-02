#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_raw.py
============
Splits a large Sopra HR raw export into smaller files of N tickets each.

Usage:
    python scripts/split_raw.py --input "data/raw/Archive_All_NoIN_NoV1 (2).txt" --chunk-size 5000
    python scripts/split_raw.py --input "data/raw/Archive_All_NoIN_NoV1 (2).txt" --chunk-size 5000 --out-dir data/raw/chunks
"""

import argparse
import re
from pathlib import Path

# Same header detection as tickets_pipeline.py
START_RE = re.compile(r'^\d+\t')
REF_RE = re.compile(r'^(FR|AF|SP|DE|UK|BE|IT|NL)\s*W\d{5,}$')


def is_valid_header(line: str) -> bool:
    if not START_RE.match(line):
        return False
    cols = line.split('\t')
    if len(cols) < 6:
        return False
    reference = cols[4].strip()
    title = cols[5].strip()
    version = cols[2].strip()
    system = cols[3].strip()
    if not REF_RE.match(reference):
        return False
    if not title or title.strip('*').strip() == "":
        return False
    if not re.match(r'^(HRA|HRV|HRC|HRE|HRON|HRN|HRP)[0-9A-Z\.]+$', version):
        return False
    if not re.match(r'^[A-Z0-9]{2,4}$', system):
        return False
    return True


def split_file(input_path: Path, out_dir: Path, chunk_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"Reading {input_path} ...")
    text = input_path.read_text(encoding='latin-1')
    lines = text.splitlines(keepends=True)
    print(f"Total lines: {len(lines)}")

    # Find ticket start positions
    starts = [i for i, l in enumerate(lines) if is_valid_header(l)]
    total_tickets = len(starts)
    print(f"Detected {total_tickets} tickets")

    # Split into chunks
    file_num = 0
    for chunk_start in range(0, total_tickets, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_tickets)
        file_num += 1

        line_start = starts[chunk_start]
        line_end = starts[chunk_end] if chunk_end < total_tickets else len(lines)

        chunk_lines = lines[line_start:line_end]
        out_path = out_dir / f"{stem}_part{file_num:03d}.txt"
        out_path.write_text(''.join(chunk_lines), encoding='latin-1')

        print(f"  {out_path.name}  tickets {chunk_start+1}-{chunk_end}  ({chunk_end - chunk_start} tickets)")

    print(f"\nDone: {file_num} files written to {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Split a large raw ticket export into smaller files")
    ap.add_argument("--input", type=Path, required=True, help="Path to the large raw .txt file")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same dir as input, in a 'chunks' subfolder)")
    ap.add_argument("--chunk-size", type=int, default=5000, help="Number of tickets per file (default: 5000)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.input.parent / "chunks"
    split_file(args.input, out_dir, args.chunk_size)
