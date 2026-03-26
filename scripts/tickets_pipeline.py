#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tickets_pipeline.py — Corrected & Complete Version
====================================================
Splits a raw Sopra HR IBM Lotus export (.txt) into one .txt file per ticket.

FIXES over original:
  1. Encoding: reads as latin-1 (not utf-8) — preserves all French accents
  2. Description: extracted from col[6], :::: separators cleaned
  3. Resolution: extracted from the last substantive Reply block
  4. Team codes: updated to match the real codes found in the data
  5. Offer field: now populated from col[2] (was always empty)
  6. Excel fallback: hard warning instead of silent degradation

Each output .txt file contains:
  - YAML frontmatter (metadata for RAG)
  - Original raw block (conversation history)

Usage:
    python tickets_pipeline.py --raw-file FR_5000.txt --out-dir data/output
    python tickets_pipeline.py --raw-file FR_5000.txt --out-dir data/output --watch
"""

import argparse
import csv
import hashlib
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


###############################################################################
# 1. UTILITIES
###############################################################################

FORBIDDEN_WIN_CHARS_RE = re.compile(r'[<>:"/\\|?*]')

def sanitize_filename(text: str, max_len: int = 150) -> str:
    text = (text or "").strip()
    text = FORBIDDEN_WIN_CHARS_RE.sub('_', text)
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text[:max_len] or "no_name"

def block_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def normalize_key(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())


###############################################################################
# 2. TICKET BOUNDARY DETECTION
###############################################################################

# Closing line: \t\tCP\t3\tPAPFRINT
END_STATUS_RE = re.compile(r'^\t\t([A-Z]{2,3})\t(\d+)\t([A-Z0-9_]+)\s*$')

# Header line: N\tSITE\tVERSION\tSYSTEM\tREFERENCE\tTITLE\tDESCRIPTION\t...
START_RE = re.compile(r'^\d+\t')
REF_RE   = re.compile(r'^(FR|AF|SP|DE|UK|BE|IT|NL)\s*W\d{5,}$')


def is_valid_header(line: str) -> bool:
    if not START_RE.match(line):
        return False
    cols = line.split('\t')
    if len(cols) < 6:
        return False

    reference = cols[4].strip()
    title     = cols[5].strip()
    version   = cols[2].strip()
    system    = cols[3].strip()

    if not REF_RE.match(reference):
        return False
    if not title or title.strip('*').strip() == "":
        return False
    if not re.match(r'^(HRA|HRV|HRC|HRE|HRON|HRN|HRP)[0-9A-Z\.]+$', version):
        return False
    if not re.match(r'^[A-Z0-9]{2,4}$', system):
        return False

    return True


def detect_segments(lines: List[str]) -> List[Tuple[int, int]]:
    starts = [i for i, l in enumerate(lines) if is_valid_header(l)]
    return [
        (s, starts[i + 1] if i + 1 < len(starts) else len(lines))
        for i, s in enumerate(starts)
    ]


def parse_header(line: str) -> Dict[str, str]:
    cols = line.split('\t')
    while len(cols) < 8:
        cols.append("")
    return {
        "ticket_no":   cols[0].strip(),
        "site_env":    cols[1].strip(),
        "offer":       cols[2].strip(),   # FIX: was never populated
        "system":      cols[3].strip(),
        "reference":   cols[4].strip(),
        "title":       cols[5].strip(),
        "description": cols[6].strip(),   # FIX: was never extracted
    }


###############################################################################
# 3. DESCRIPTION & RESOLUTION EXTRACTION
###############################################################################

# :::: and :: are Lotus line separators inside the tab-separated description field
LOTUS_SEP_RE = re.compile(r':{2,}')

# Timestamp line patterns
TIMESTAMP_RE = re.compile(
    r'^(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})'
    r'(?:\s+(?:CEDT|CET|UTC|GMT))?'
    r'\s*(.*)'
)

COURTESY_PATTERNS = [
    r'nous proc[eé]dons.{0,50}fermeture',
    r'nous vous remercions de votre retour',
    r'^cordialement[\.\s]*$',
    r'remettons l.{0,15}v[eé]nement en attente',
    r'^bien [àa] vous[\.\s]*$',
]

TECHNICAL_SIGNALS = [
    r'patch\s+\d+',
    r'kit\s+\w+',
    r'int[eé]grer',
    r'erreur',
    r'param[eé]trage',
    r'rubrique\s+\w+',
    r'veuillez\s+\w+',
    r'proc[eé]dure',
    r'commande\s+\w+',
    r'table\s+\w+',
    r'champ\s+\w+',
    r'ORA-\d+',
    r'[A-Z]{2,}\.[A-Z]{3,}',   # field references like ZD7R.NATETA
]


def clean_description(raw: str) -> str:
    """Cleans the raw description from col[6] — removes :::: separators."""
    if not raw:
        return ""
    cleaned = LOTUS_SEP_RE.sub('\n', raw)
    # Remove leading greeting lines that add no information
    lines = [l.strip() for l in cleaned.split('\n')]
    lines = [l for l in lines if l and l.lower() not in ('bonjour,', 'bonjour', 'cordialement,', 'cordialement')]
    return '\n'.join(lines).strip()


def is_courtesy_only(text: str) -> bool:
    t = text.lower().strip()
    # If ALL lines match courtesy patterns, it's a courtesy message
    meaningful_lines = [l.strip() for l in t.split('\n') if l.strip()]
    if not meaningful_lines:
        return True
    courtesy_count = sum(
        1 for line in meaningful_lines
        if any(re.search(p, line) for p in COURTESY_PATTERNS)
    )
    return courtesy_count == len(meaningful_lines)


def technical_score(text: str) -> int:
    t = text.lower()
    return sum(1 for sig in TECHNICAL_SIGNALS if re.search(sig, t))


def extract_reply_blocks(block_lines: List[str]) -> List[str]:
    """
    Extracts all 'Reply :' blocks from the conversation.
    Returns list of reply texts.
    """
    replies = []
    in_reply = False
    buffer = []

    for line in block_lines:
        stripped = line.strip()

        if stripped == 'Reply :':
            if buffer:
                text = '\n'.join(buffer).strip()
                if text:
                    replies.append(text)
            buffer = []
            in_reply = True
            continue

        if in_reply:
            ts = TIMESTAMP_RE.match(line)
            if ts and ('Archived' in line or 'Status' in line):
                if buffer:
                    text = '\n'.join(buffer).strip()
                    if text:
                        replies.append(text)
                buffer = []
                in_reply = False
            elif stripped not in ('Hot Line :', 'Follow-up :', 'Client :'):
                buffer.append(line)

    if buffer and in_reply:
        text = '\n'.join(buffer).strip()
        if text:
            replies.append(text)

    return replies


def extract_resolution(block_lines: List[str]) -> str:
    """
    Extracts the most technically meaningful support reply as the resolution.
    This is the KEY FIELD for RAG — it's the answer to retrieve.

    Strategy:
    - Collect all Reply blocks
    - Score each by technical content
    - Skip courtesy-only messages
    - Return highest-scoring reply (prefer later ones on tie)
    """
    replies = extract_reply_blocks(block_lines)

    candidates = []
    for i, reply in enumerate(replies):
        if len(reply) < 30:
            continue
        if is_courtesy_only(reply):
            continue
        score = technical_score(reply)
        candidates.append((score, i, reply))  # i used for stable sort by position

    if candidates:
        # Sort by score desc, then by position desc (prefer later replies)
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    # Fallback: last non-empty reply regardless
    for reply in reversed(replies):
        if len(reply) > 30:
            return reply

    return ""


def extract_patches(block: str) -> List[str]:
    """
    Extracts patch numbers explicitly labelled as patches.
    Uses >= 150000 threshold to exclude reference numbers and IDs.
    """
    patches = []

    # Pattern 1: "Patches : 178236;178237" (same line or multiline list)
    for m in re.finditer(r'Patches?\s*[:\uff1a]\s*\n?([\d][\d\s;,\n]*)', block, re.IGNORECASE):
        raw = m.group(1)
        raw = re.split(r'\n[^\d\s;,]', raw)[0]
        patches.extend(re.findall(r'\b(\d{5,6})\b', raw))

    # Pattern 2: "Patch 178236" alone on its own line
    for m in re.finditer(r'^Patch\s+(\d{5,6})\s*$', block, re.IGNORECASE | re.MULTILINE):
        patches.append(m.group(1))

    # Extract ticket reference to exclude it from patches
    ref_match = re.search(r'^reference:\s*\S+\s*W(\d+)', block, re.MULTILINE)
    ref_suffix = ref_match.group(1)[-5:] if ref_match else None

    # Filter: real HRA patches are >= 150000; exclude ticket reference numbers
    filtered = []
    for p in patches:
        if int(p) < 150000:
            continue
        if ref_suffix and p.endswith(ref_suffix):
            continue
        filtered.append(p)

    return list(dict.fromkeys(filtered))


###############################################################################
# 4. TEAM CLASSIFICATION
###############################################################################

# FIXED: Updated to match real team codes found in the actual data
DSN_CODES    = {'DSN', 'DSNEXT', 'REGDSN', 'ESPDSN', 'DSN_J2E_DISPATCH'}
APPLI_CODES  = {'PAPFRINT', 'PAYFR', 'PAYFG', 'PAYFC', 'PADFR', 'PADFG',
                'PADFE', 'PAPIN', 'COLTER', 'IPS', 'PAYFRINT', 'PAPFR',
                'SPAIN', 'MAROC', 'SHRFRINT', 'TM'}
OUTILS_CODES = {'DESGN', 'CSTC', 'HRCT', 'HRASPACE', 'HRCTV2', 'QRYBT',
                'PROCREP', 'GTA', 'WIM', 'PACKAGING', 'PLT_TOOLS',
                'HRAANA', 'ORASUPP'}

# Keywords for fallback classification
KW_DSN   = {'DSN', 'N4DS', 'URSSAF', 'SIREN', 'DECLARATION', 'DADS-U',
             'REGDSN', 'DSNEXT', 'DSN_J2E'}
KW_APPLI = {'PAIE', 'PAIEMENT', 'CET', 'ABSENCE', 'CONGE', 'MALADIE',
             'RUBRIQUE', 'NRB', 'NRA', 'BULLETIN', 'MANDATEMENT',
             'BORDEREAU', 'TRAITEMENT', 'AFFECTATION'}
KW_OUTIL = {'DESIGN CENTER', 'COMPILATION', 'GENERATION', 'BNA', 'BNK',
             'DBA', 'DBI', 'HRCT', 'HRQUERY', 'QUERY', 'HRASPACE',
             'HRANALYTICS', 'STUDIO', 'PACKAGING'}

WEIGHTS = {
    'explicit_closing': 5,
    'h2_route':         4,
    'module':           2,
    'keyword':          1,
}

H2_RE     = re.compile(r'\bH[123]\s+([A-Z][A-Z0-9_]+)\b')
MODULE_RE = re.compile(r'\b([A-Z]{3,10})(?::[A-Z0-9]+)?\b')

# Status codes to skip in module detection
SKIP_CODES = {"H2", "H1", "H3", "TS", "CP", "CI", "CN", "CQ", "CU",
              "CO", "CH", "CX", "CY", "AA", "AK", "AI", "AW"}


def classify_team(block: str, closing_teamcode: str) -> Tuple[str, str, float]:
    """
    Classifies a ticket into DSN / Appli / Outils / Unknown.
    Returns (team, source, confidence).

    Priority:
    1. Closing team code (most reliable — it's who actually handled it)
    2. H2 routing codes in conversation
    3. Module codes
    4. Keywords
    """
    # 1. Closing team code — highest confidence
    if closing_teamcode:
        code = closing_teamcode.upper()
        if code in DSN_CODES:
            return ("DSN", "closing_teamcode", 0.97)
        if code in APPLI_CODES:
            return ("Appli", "closing_teamcode", 0.97)
        if code in OUTILS_CODES:
            return ("Outils", "closing_teamcode", 0.97)

    # 2. Score-based classification from text signals
    signals = []

    # H2/H1/H3 routing lines
    for m in H2_RE.finditer(block):
        mod = m.group(1).upper()
        if mod in DSN_CODES:
            signals.append(("h2_route", "DSN"))
        elif mod in APPLI_CODES:
            signals.append(("h2_route", "Appli"))
        elif mod in OUTILS_CODES:
            signals.append(("h2_route", "Outils"))

    # Module codes
    for m in MODULE_RE.finditer(block):
        mod = m.group(1).upper()
        if mod in SKIP_CODES:
            continue
        if mod in DSN_CODES:
            signals.append(("module", "DSN"))
        elif mod in APPLI_CODES:
            signals.append(("module", "Appli"))
        elif mod in OUTILS_CODES:
            signals.append(("module", "Outils"))

    # Keywords
    block_upper = block.upper()
    if any(k in block_upper for k in KW_DSN):
        signals.append(("keyword", "DSN"))
    if any(k in block_upper for k in KW_APPLI):
        signals.append(("keyword", "Appli"))
    if any(k in block_upper for k in KW_OUTIL):
        signals.append(("keyword", "Outils"))

    if not signals:
        return ("Unknown", "no_signal", 0.0)

    score: Counter = Counter()
    for kind, team in signals:
        score[team] += WEIGHTS.get(kind, 0)

    best_team, best_score = score.most_common(1)[0]
    total = sum(score.values()) or 1
    conf  = round(best_score / total, 3)

    # Determine source label
    src = "keyword"
    for preferred in ("h2_route", "module", "keyword"):
        if any(k == preferred and t == best_team for k, t in signals):
            src = preferred
            break

    return (best_team, src, conf)


###############################################################################
# 5. STATUS CODES
###############################################################################

CLOSING_STATUS_EXPLANATIONS = {
    "AA": "To be processed",
    "AF": "Preparation Hotfix delivery",
    "AI": "Information Received",
    "AK": "In Progress",
    "AW": "Waiting information",
    "CA": "New patch 4YOU (closed)",
    "CH": "Sent to Services (closed)",
    "CI": "Waiting for customer information",
    "CN": "New patch delivered (closed)",
    "CO": "Old patch delivered (closed)",
    "CP": "Request processed (closed)",
    "CQ": "Corrected in future release",
    "CR": "Corrected in existing release",
    "CS": "Bypass solution (closed)",
    "CT": "Transferred (closed)",
    "CU": "Incorrect use (closed)",
    "CV": "Validated (closed)",
    "CX": "Cancelled (closed)",
    "CY": "Duplicate (closed)",
    "CZ": "Obsolete (closed)",
    "C2": "Cancelled/service (closed)",
    "MC": "Closed without solution",
}

STATUS_CLOSED = {"CA", "CN", "CP", "CH", "CU", "C2", "MC", "CO", "CS",
                 "CT", "CX", "CY", "CZ", "CQ", "CV", "CR"}


def explain_status(code: str) -> str:
    return CLOSING_STATUS_EXPLANATIONS.get(code.upper(), "")


###############################################################################
# 6. EXCEL PRODUCT CODE MAPPING (optional enhancement)
###############################################################################

def load_excel_mapping(xlsx_path: Optional[Path]) -> Dict:
    """
    Loads Code_Produit.xlsx and builds a lookup table:
      PREFIX:VERSION  → team   (exact match, e.g. PAYFR:900 → Appli)
      PREFIX:*DIGIT   → team   (wildcard, e.g. REGDSN:3XX → DSN)
      PREFIX          → team   (bare code, e.g. ABSENC → Appli)

    Domain → Team mapping:
      DSN / PAS                          → DSN
      Core / CODPROWEB / Interop / GTA   → Outils
      France Privé / France Publique
        / International Application
        / 4YOU                           → Appli
    """
    empty = {"_EXACT": {}, "_WILD": []}
    if not xlsx_path:
        return empty

    try:
        from openpyxl import load_workbook
    except ImportError:
        print("[WARN] openpyxl not installed — Excel mapping disabled.")
        print("       Install with: pip install openpyxl")
        return empty

    if not xlsx_path.exists():
        print(f"[WARN] Excel file not found: {xlsx_path} — Excel mapping disabled.")
        return empty

    def _domain_to_team(domain: str) -> Optional[str]:
        d = domain.upper()
        if 'DSN' in d or d == 'PAS':
            return 'DSN'
        if any(k in d for k in ('CORE', 'CODPROWEB', 'INTEROP', 'GTA', 'SUPPORT')):
            return 'Outils'
        if any(k in d for k in ('FRANCE', 'INTERNATIONAL', '4YOU')):
            return 'Appli'
        return None

    try:
        wb = load_workbook(str(xlsx_path), read_only=True, data_only=True)
        ws = wb.active

        exact: Dict[str, str] = {}   # "PAYFR:900" → "Appli"
        wild:  List           = []   # ("REGDSN", "3", "DSN")

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or row[0] is None:
                continue
            code   = str(row[0]).strip()
            domain = str(row[1] or '').strip()

            if not code or code in ('-', '?', ''):
                continue

            team = _domain_to_team(domain)
            if not team:
                continue

            # Parse code formats:
            # "PAYFR - 900"   → prefix=PAYFR, version=900  (exact)
            # "REGDSN - 3XX"  → prefix=REGDSN, digit=3     (wildcard)
            # "CPWXX - 7XX"   → prefix=CPW, digit=7         (wildcard)
            # "ABSENC"        → bare code                   (exact prefix)
            m = re.match(
                r'^([A-Z0-9]+?)(?:XX)?\s*-\s*([0-9]+|[0-9]XX|[0-9]000|XXX)\s*$',
                code, re.IGNORECASE
            )
            if m:
                prefix  = m.group(1).upper()
                version = m.group(2).upper()
                if 'X' in version:
                    # Wildcard: REGDSN:3XX → any version starting with "3"
                    wild.append((prefix, version[0], team))
                else:
                    exact[f"{prefix}:{version}"] = team
            else:
                # Bare code like "ABSENC", "JAUGES", "AFI"
                bare = re.sub(r'XX$', '', code.upper().strip())
                if re.match(r'^[A-Z][A-Z0-9]+$', bare):
                    exact[bare] = team

        print(f"[INFO] Excel mapping loaded: {len(exact)} exact + {len(wild)} wildcard codes")
        return {"_EXACT": exact, "_WILD": wild}

    except Exception as e:
        print(f"[WARN] Excel mapping failed: {e} — continuing without it.")
        return empty


def domain_to_team(domain: str) -> str:
    d = domain.upper()
    if "DSN" in d:
        return "DSN"
    if any(k in d for k in ("CORE", "WEBTOOLS", "HRCT", "API", "RUNTOOLS",
                              "STUDIO", "DELIVERY", "IPS", "DESIGN")):
        return "Outils"
    return "Appli"


def classify_via_excel(block: str, excel_maps: Dict) -> Optional[str]:
    """
    Classifies ticket using product codes from H2 routing lines.
    Matches against Code_Produit.xlsx mapping.

    Looks for: "H2 TEAMCODE PRODUCTCODE:VERSION"
    e.g.  "H2 DSN REGDSN:3XX"      → DSN
          "H1 PAPFRINT PAYFR:900"  → Appli
          "H2 CSTC DSGXX:710"      → Outils

    Returns team string if match found, else None.
    """
    exact = excel_maps.get("_EXACT", {})
    wild  = excel_maps.get("_WILD", [])

    if not exact and not wild:
        return None

    H2_PROD_RE = re.compile(r'\bH[123]\s+\w+\s+([A-Z][A-Z0-9]+):([0-9A-Z]+)\b')

    for m in H2_PROD_RE.finditer(block):
        prefix  = m.group(1).upper()
        version = m.group(2).upper()

        # 1. Exact match: "PAYFR:900" → Appli
        key = f"{prefix}:{version}"
        if key in exact:
            return exact[key]

        # 2. Wildcard match: "REGDSN:3XX" → first digit=3 matches wild entry
        if version and version[0].isdigit():
            for w_prefix, w_digit, w_team in wild:
                if prefix == w_prefix and version[0] == w_digit:
                    return w_team

        # 3. Bare prefix fallback: "ABSENC" with no version
        if prefix in exact:
            return exact[prefix]

    return None


###############################################################################
# 7. CORE PROCESSING
###############################################################################

INDEX_FIELDS = [
    'ordinal', 'ticket_no', 'site_env', 'offer',
    'reference', 'title', 'version', 'system',
    'start_line', 'end_line',
    'closing_status_code', 'closing_status_explanation',
    'closing_level', 'closing_teamcode',
    'file_name', 'support_team', 'team_source',
    'confidence', 'patches', 'content_hash',
]


def process_raw_file(
    raw_file: Path,
    out_dir: Path,
    index_csv: Path,
    excel_maps: Dict,
) -> Tuple[int, int, int]:

    out_dir.mkdir(parents=True, exist_ok=True)

    # FIX 1: Read as latin-1 to preserve all French accented characters
    text  = raw_file.read_text(encoding='latin-1')
    lines = text.splitlines()

    segments = detect_segments(lines)
    print(f"  Detected {len(segments)} ticket segments")

    existing_index = _load_existing_index(index_csv)

    rows    = []
    created = 0
    updated = 0
    unknown = 0

    for ordinal, (s, e) in enumerate(segments, start=1):
        header_line  = lines[s]
        meta         = parse_header(header_line)
        block_lines  = lines[s:e]
        block        = "\n".join(block_lines)
        content_md5  = block_hash(block)

        reference = meta["reference"]
        title     = meta["title"]
        version   = meta["offer"]    # col[2] = version/offer
        system    = meta["system"]

        # FIX 2: Extract description
        description = clean_description(meta["description"])

        # FIX 3: Extract resolution
        resolution = extract_resolution(block_lines)

        # Extract patches
        patches = extract_patches(block)

        # Extract closing metadata (use LAST closing line found)
        closing_code = closing_level = closing_teamcode = closing_expl = ""
        for line in block_lines:
            m = END_STATUS_RE.match(line)
            if m:
                closing_code     = m.group(1).upper()
                closing_level    = m.group(2)
                closing_teamcode = m.group(3).upper()
                closing_expl     = explain_status(closing_code)

        # FIX 4: Team classification with correct codes
        # Try Excel first, then text-based classification
        excel_team = classify_via_excel(block, excel_maps)
        if excel_team:
            team, src, conf = excel_team, "excel_product", 0.92
        else:
            team, src, conf = classify_team(block, closing_teamcode)

        if team == "Unknown":
            unknown += 1

        # Build output filename
        ref_clean = sanitize_filename(reference or f"noRef_{ordinal:04d}")
        filename  = f"{ordinal:04d}_{ref_clean}.txt"
        file_path = out_dir / filename

        # Check if we need to write
        prev       = existing_index.get(reference)
        need_write = (
            prev is None
            or prev.get("content_hash") != content_md5
            or prev.get("file_name") != filename
            or not file_path.exists()
        )

        if prev is None:
            created += 1
        elif need_write:
            updated += 1

        if need_write:
            # Build YAML frontmatter
            frontmatter = (
                "---\n"
                f"reference: {reference}\n"
                f"title: {title}\n"
                f"version: {version}\n"
                f"system: {system}\n"
                f"support_team: {team}\n"
                f"team_source: {src}\n"
                f"confidence: {conf}\n"
                f"content_hash: {content_md5}\n"
                f"closing_status_code: {closing_code}\n"
                f"closing_status_explanation: {closing_expl}\n"
                f"closing_level: {closing_level}\n"
                f"closing_teamcode: {closing_teamcode}\n"
                f"patches: {';'.join(patches)}\n"
                "---\n"
                f"[DESCRIPTION]\n{description}\n\n"
                f"[RESOLUTION]\n{resolution}\n\n"
                f"[CONVERSATION]\n"
            )
            file_path.write_text(
                frontmatter + block.rstrip() + "\n",
                encoding='utf-8'
            )

        rows.append({
            'ordinal':                   str(ordinal),
            'ticket_no':                 meta['ticket_no'],
            'site_env':                  meta['site_env'],
            'offer':                     version,
            'reference':                 reference,
            'title':                     title,
            'version':                   version,
            'system':                    system,
            'start_line':                str(s + 1),
            'end_line':                  str(e),
            'closing_status_code':       closing_code,
            'closing_status_explanation': closing_expl,
            'closing_level':             closing_level,
            'closing_teamcode':          closing_teamcode,
            'file_name':                 filename,
            'support_team':              team,
            'team_source':               src,
            'confidence':                f"{conf:.3f}",
            'patches':                   ';'.join(patches),
            'content_hash':              content_md5,
        })

        if ordinal % 500 == 0:
            print(f"  ... {ordinal}/{len(segments)} processed")

    _write_index(index_csv, rows)

    print(f"\n  Summary:")
    print(f"    Total segments : {len(segments)}")
    print(f"    Created        : {created}")
    print(f"    Updated        : {updated}")
    print(f"    Unknown team   : {unknown} ({unknown/len(segments)*100:.1f}%)")

    # Team distribution
    team_dist: Counter = Counter(r['support_team'] for r in rows)
    print(f"\n  Team distribution:")
    for t, c in team_dist.most_common():
        print(f"    {t:12s} {c:5d}  ({c/len(rows)*100:.1f}%)")

    return len(segments), created, updated


###############################################################################
# 8. CSV INDEX
###############################################################################

def _load_existing_index(path: Path) -> Dict:
    if not path.exists():
        return {}
    out = {}
    with path.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ref = row.get("reference", "").strip()
            if ref:
                out[ref] = row
    return out


def _write_index(path: Path, new_rows: List[Dict]):
    existing = _load_existing_index(path)

    for r in new_rows:
        ref = r.get("reference", "").strip() or f"__NOREF__:{r.get('ordinal','')}"
        existing[ref] = r

    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()

        def sort_key(item):
            try:
                return (int(item[1].get('ordinal', '0')), item[1].get('reference', ''))
            except ValueError:
                return (0, item[1].get('reference', ''))

        for _, row in sorted(existing.items(), key=sort_key):
            w.writerow({k: row.get(k, '') for k in INDEX_FIELDS})

    tmp.replace(path)
    print(f"  Index saved → {path}  ({len(existing)} total entries)")


###############################################################################
# 9. MAIN / WATCH LOOP
###############################################################################

def main():
    ap = argparse.ArgumentParser(
        description="Split Sopra HR IBM Lotus tickets into individual .txt files"
    )
    ap.add_argument("--raw-file",      type=Path, required=True,
                    help="Path to the raw .txt export (latin-1 encoded)")
    ap.add_argument("--out-dir",       type=Path, default=Path("data/output"),
                    help="Output folder for individual ticket files")
    ap.add_argument("--index-csv",     type=Path, default=Path("data/tickets_index.csv"),
                    help="CSV index file path")
    ap.add_argument("--product-excel", type=Path, default=None,
                    help="Optional: Code Produit.xlsx for product-code classification")
    ap.add_argument("--watch",         action="store_true",
                    help="Watch for file changes and reprocess automatically")
    ap.add_argument("--interval",      type=int, default=30,
                    help="Watch interval in seconds (default: 30)")
    args = ap.parse_args()

    excel_maps = load_excel_mapping(args.product_excel)

    last_raw   = None
    last_excel = args.product_excel.stat().st_mtime if (args.product_excel and args.product_excel.exists()) else None

    while True:
        try:
            raw_mtime   = args.raw_file.stat().st_mtime
            excel_mtime = args.product_excel.stat().st_mtime if (args.product_excel and args.product_excel.exists()) else None

            excel_changed = (excel_mtime != last_excel)
            raw_changed   = (raw_mtime != last_raw)

            if excel_changed and args.product_excel:
                excel_maps = load_excel_mapping(args.product_excel)
                print("[watch] Excel updated — mapping reloaded")
                last_excel = excel_mtime

            if raw_changed or (args.watch and excel_changed):
                print(f"\n[pipeline] Processing {args.raw_file} ...")
                t0 = time.time()
                total, created, updated = process_raw_file(
                    args.raw_file, args.out_dir, args.index_csv, excel_maps
                )
                elapsed = time.time() - t0
                print(f"[pipeline] Done in {elapsed:.1f}s — tickets={total} created={created} updated={updated}")
                last_raw = raw_mtime

            if not args.watch:
                break

            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            if not args.watch:
                raise
            time.sleep(args.interval)


if __name__ == "__main__":
    main()