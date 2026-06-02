#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
txt_to_json.py
==============
Converts the .txt ticket files produced by tickets_pipeline.py
into clean JSON files — one per ticket.

Each JSON contains:
  - All metadata from the YAML frontmatter
  - description   (cleaned client problem statement)
  - resolution    (best support reply — KEY field for RAG)
  - conversation  (parsed list of exchanges with actor/timestamp/text)
  - patches       (list of patch numbers)

Usage:
    python txt_to_json.py --input data/output --output data/json
    python txt_to_json.py --input data/output --output data/json --team DSN
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# PATTERNS
# ─────────────────────────────────────────────

# Timestamp lines: "02/10/2015 10:29:05 - Madame Patricia Pignon - Création..."
# or              "02/10/2015 12:40:27 CEDT Claire Bisiau H2 DSN REGDSN:3XX"
CLIENT_EVENT_RE = re.compile(
    r'^(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s*-\s*(.+?)\s*-\s*(.+)$'
)
SUPPORT_EVENT_RE = re.compile(
    r'^(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+(?:CEDT|CET|UTC|GMT)?\s*(.+)$'
)
STATUS_LINE_RE = re.compile(
    r'^\d{2}/\d{2}/\d{4}.+- Status ([A-Z]{2}) - (.+)$'
)
SEPARATOR_RE = re.compile(r'^_{10,}$')
SECTION_RE   = re.compile(r'^\[(DESCRIPTION|RESOLUTION|CONVERSATION)\]$')

_PATCH_RE = re.compile(
    r'(?:patch|correctif|livraison)\s*(?:n[°o]?\s*)?(\d{5,6})',
    re.IGNORECASE
)
_PATCH_CODE_RE = re.compile(r'\b(ZY\w{4,}|ZX\w{4,})\b', re.IGNORECASE)

_STATUS_CLOSED = {"CA","CN","CP","CH","CU","C2","MC","CO","CS","CT","CX","CY","CZ","CQ","CV","CR"}


# ─────────────────────────────────────────────
# FRONTMATTER PARSER
# ─────────────────────────────────────────────

def parse_frontmatter(text: str) -> dict:
    """
    Parses the YAML-like frontmatter between the two '---' lines.
    Returns a dict of all key: value pairs.
    """
    meta = {}
    # Extract between first and second ---
    m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if not m:
        return meta

    for line in m.group(1).splitlines():
        if ':' in line:
            key, _, value = line.partition(':')
            meta[key.strip()] = value.strip()

    return meta


# ─────────────────────────────────────────────
# SECTION SPLITTER
# ─────────────────────────────────────────────

def split_sections(text: str) -> dict:
    """
    Splits the file body (after frontmatter) into
    DESCRIPTION, RESOLUTION, and CONVERSATION sections.
    """
    # Remove frontmatter
    body = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

    sections = {'DESCRIPTION': '', 'RESOLUTION': '', 'CONVERSATION': ''}
    current  = None
    buffer   = []

    for line in body.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            if current and buffer:
                sections[current] = '\n'.join(buffer).strip()
            current = m.group(1)
            buffer  = []
        else:
            if current:
                buffer.append(line)

    if current and buffer:
        sections[current] = '\n'.join(buffer).strip()

    return sections


# ─────────────────────────────────────────────
# CONVERSATION PARSER
# ─────────────────────────────────────────────

def parse_conversation(conv_text: str) -> list:
    """
    Parses the raw conversation block into a structured list of exchanges.

    Each exchange is:
    {
        "timestamp": "2015-10-02T10:29:05",
        "actor":     "client" | "support" | "system",
        "author":    "Patricia Pignon",
        "text":      "...",
        "type":      "creation" | "message" | "reply" | "status" | "archive"
    }
    """
    lines    = conv_text.splitlines()
    exchanges = []
    current   = None
    buffer    = []

    def flush():
        if current and buffer:
            text = '\n'.join(buffer).strip()
            # Remove filler lines
            text = re.sub(r'^(Hot Line|Follow-up|Reply)\s*:\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if text:
                current['text'] = text
            exchanges.append({k: v for k, v in current.items() if v})
        buffer.clear()

    for line in lines:
        # Skip separators and blank section markers
        if SEPARATOR_RE.match(line.strip()):
            flush()
            current = None
            continue

        if line.strip() in ('Hot Line :', 'Follow-up :', 'Reply :', ' Client :', 'Client :'):
            continue

        # Client event: "02/10/2015 10:29:05 - Madame Patricia Pignon - Création de l'événement."
        cm = CLIENT_EVENT_RE.match(line)
        if cm:
            flush()
            ts     = _parse_ts(cm.group(1))
            author = cm.group(2).strip()
            action = cm.group(3).strip()
            etype  = _classify_client_action(action)
            current = {
                'timestamp': ts,
                'actor':     'client',
                'author':    author,
                'type':      etype,
                'action':    action,
                'text':      '',
            }
            buffer = []
            continue

        # Support/status line: "02/10/2015 12:40:27 CEDT Claire Bisiau H2 DSN REGDSN:3XX"
        sm = SUPPORT_EVENT_RE.match(line)
        if sm and not line.startswith('\t'):
            raw_rest = sm.group(2).strip()

            # Status change line: "Claire Bisiau - Status CP - Request processed (closed)"
            if ' - Status ' in raw_rest:
                flush()
                parts  = raw_rest.split(' - Status ')
                author = parts[0].strip()
                code_desc = parts[1].strip() if len(parts) > 1 else ''
                current = {
                    'timestamp': _parse_ts(sm.group(1)),
                    'actor':     'system',
                    'author':    author,
                    'type':      'status_change',
                    'text':      code_desc,
                }
                flush()
                current = None
                continue

            # Archive line
            if '- Archived' in raw_rest:
                flush()
                current = None
                continue

            # Regular support message
            flush()
            author = _extract_author(raw_rest)
            current = {
                'timestamp': _parse_ts(sm.group(1)),
                'actor':     'support',
                'author':    author,
                'type':      'message',
                'text':      '',
            }
            buffer = []
            continue

        # Continuation line — append to current exchange
        if current is not None:
            # Skip closing metadata line \t\tCP\t3\tDSN
            if re.match(r'^\t\t[A-Z]{2}\t', line):
                continue
            buffer.append(line)

    flush()
    return exchanges


def _parse_ts(raw: str) -> str:
    """Converts "02/10/2015 10:29:05" to ISO format "2015-10-02T10:29:05"."""
    try:
        dt = datetime.strptime(raw.strip(), '%d/%m/%Y %H:%M:%S')
        return dt.isoformat()
    except ValueError:
        return raw.strip()


def _classify_client_action(action: str) -> str:
    a = action.lower()
    if 'création' in a or 'creation' in a:
        return 'creation'
    if 'fermer' in a or 'clôture' in a or 'cloture' in a:
        return 'close_request'
    if 'complément' in a or 'complement' in a or 'information' in a:
        return 'info_provided'
    if 'rejet' in a or 'convient pas' in a:
        return 'rejection'
    return 'message'


def _extract_author(text: str) -> str:
    """Extracts the agent name from a support line."""
    # Pattern: "Claire Bisiau H2 DSN REGDSN:3XX" → "Claire Bisiau"
    # Stop at known role/routing tokens
    m = re.match(r'^([A-ZÀÂÄÉÈÊËÎÏÔÙÛÜ][a-zàâäéèêëîïôùûü]+(?:\s+[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜ][a-zàâäéèêëîïôùûü]+)+)', text)
    if m:
        return m.group(1).strip()
    # Fallback: first two words
    parts = text.split()
    return ' '.join(parts[:2]) if len(parts) >= 2 else text


# ─────────────────────────────────────────────
# ESPDSN VERSION EXTRACTION
# ─────────────────────────────────────────────

# Matches "ESPDSN:3HR", "ESPDSN - 10PL", "ESPDSN:12HR", "ESPDSN - 3PL" etc.
_ESPDSN_RE = re.compile(r'ESPDSN[\s:*\-]+(\d{1,2}(?:HR|PL))', re.IGNORECASE)


def _extract_espdsn_version(conv_text: str) -> str:
    """Extract the latest (highest) ESPDSN version from conversation text."""
    matches = _ESPDSN_RE.findall(conv_text)
    if not matches:
        return ""
    # Normalize and deduplicate
    versions = sorted(set(m.upper() for m in matches), key=lambda v: (int(re.match(r'\d+', v).group()), v), reverse=True)
    return versions[0]  # highest version number


# ─────────────────────────────────────────────
# MAIN CONVERTER
# ─────────────────────────────────────────────

def txt_to_json(txt_path: Path) -> dict:
    """
    Converts a single .txt ticket file into a structured dict.
    """
    text = txt_path.read_text(encoding='utf-8')

    # 1. Parse frontmatter
    meta = parse_frontmatter(text)

    # 2. Split sections
    sections = split_sections(text)

    # 3. Parse conversation
    conversation = parse_conversation(sections['CONVERSATION'])

    # 4. Build patches list (filter empty strings)
    patches_raw = meta.get('patches', '')
    patches = [p.strip() for p in patches_raw.split(';') if p.strip()]

    # 4b. Extract patch numbers mentioned in resolution/conversation text
    all_text = sections['RESOLUTION'] + '\n' + sections['CONVERSATION']
    found_patches = _PATCH_RE.findall(all_text)
    found_codes = _PATCH_CODE_RE.findall(all_text)
    # Merge with frontmatter patches, deduplicate
    all_patches = list(dict.fromkeys(
        patches + found_patches + [c.upper() for c in found_codes]
    ))

    # 5. Derived fields
    site_env_parts = meta.get('site_env', '').split('*')
    client_company = site_env_parts[1] if len(site_env_parts) >= 2 else ''

    embed_text = f"{meta.get('title', '')}\n{sections['DESCRIPTION']}\n{sections['RESOLUTION']}"

    # 6. Assemble final JSON
    ticket = {
        # — Identity —
        'reference':    meta.get('reference', ''),
        'title':        meta.get('title', ''),
        'source_file':  txt_path.name,

        # — Environment —
        'version':      meta.get('version', ''),
        'system':       meta.get('system', ''),
        'site_env':     meta.get('site_env', ''),
        'client_company': client_company,

        # — Routing —
        'support_team':    meta.get('support_team', 'Unknown'),
        'team_source':     meta.get('team_source', ''),
        'closing_teamcode': meta.get('closing_teamcode', ''),

        # — Status —
        'closing_status_code':        meta.get('closing_status_code', ''),
        'closing_status_explanation': meta.get('closing_status_explanation', ''),
        'closing_level':              meta.get('closing_level', ''),
        'is_closed': meta.get('closing_status_code', '') in _STATUS_CLOSED,

        # — Content (KEY FIELDS FOR RAG) —
        'description':  sections['DESCRIPTION'],
        'resolution':   sections['RESOLUTION'],
        'embed_text':   embed_text,
        'patches':      all_patches,

        # — Conversation —
        'conversation': conversation,
        'message_count': len(conversation),

        # — ESPDSN Version (DSN team only) —
        'espdsn_version': _extract_espdsn_version(sections['CONVERSATION']) if meta.get('support_team') == 'DSN' else '',

        # — Meta —
        'confidence':    float(meta.get('confidence', 0) or 0),
        'content_hash':  meta.get('content_hash', ''),
    }

    return ticket


# ─────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────

def convert_all(input_dir: Path, output_dir: Path, team_filter: Optional[str] = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob('*.txt'))
    print(f"\n📂 Found {len(txt_files)} .txt files in {input_dir}")
    if team_filter:
        print(f"🔍 Filtering for team: {team_filter}")

    converted = 0
    skipped   = 0
    failed    = 0
    team_dist = {}

    for txt_path in txt_files:
        try:
            # Skip if JSON already exists and is newer than the .txt
            out_path = output_dir / (txt_path.stem + '.json')
            if out_path.exists() and out_path.stat().st_mtime >= txt_path.stat().st_mtime:
                skipped += 1
                continue

            ticket = txt_to_json(txt_path)

            # Team filter
            if team_filter and ticket['support_team'].lower() != team_filter.lower():
                skipped += 1
                continue

            # Track team distribution
            team = ticket['support_team']
            team_dist[team] = team_dist.get(team, 0) + 1

            # Output filename: same stem, .json extension
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(ticket, f, ensure_ascii=False, indent=2)

            converted += 1

        except Exception as e:
            print(f"  ❌ Failed: {txt_path.name} — {e}")
            failed += 1

    print(f"\n✅ Converted : {converted}")
    print(f"⏭️  Skipped   : {skipped}")
    print(f"❌ Failed    : {failed}")
    print(f"\n📊 Team distribution:")
    for team, count in sorted(team_dist.items(), key=lambda x: -x[1]):
        print(f"   {team:10s} {count:5d}  ({count / max(converted, 1) * 100:.1f}%)")
    print(f"\n💾 Output → {output_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Sopra HR .txt ticket files to JSON'
    )
    parser.add_argument('--input',  type=Path, required=True,
                        help='Folder containing .txt ticket files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output folder for .json files')
    parser.add_argument('--team',   type=str,  default=None,
                        help='Optional: only convert tickets for this team (DSN / Appli / Outils)')
    args = parser.parse_args()

    convert_all(args.input, args.output, args.team)