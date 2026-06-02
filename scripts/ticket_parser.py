"""
ticket_parser.py
----------------
Parses raw Lotus-exported .txt ticket files into clean JSON.

Ticket structure detected:
  1. YAML-like frontmatter block (between --- delimiters)
  2. Tab-separated data line (raw Lotus row)
  3. Conversation thread (Client / Hot Line / Reply blocks with timestamps)

Usage:
    python ticket_parser.py --input ./raw_tickets --output ./json_tickets
    python ticket_parser.py --input ./raw_tickets/0001_FRW210000.txt --output ./json_tickets
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 1. FRONTMATTER PARSER
# ─────────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Extracts YAML-like frontmatter between --- delimiters.
    Returns (metadata_dict, remaining_body).
    """
    metadata = {}
    body = text

    fm_pattern = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n', re.DOTALL)
    match = fm_pattern.match(text)

    if match:
        fm_text = match.group(1)
        body = text[match.end():]

        for line in fm_text.splitlines():
            line = line.strip()
            if ':' in line:
                key, _, value = line.partition(':')
                metadata[key.strip()] = value.strip()

    return metadata, body


# ─────────────────────────────────────────────
# 2a. SECTION-BASED BODY PARSER
# ─────────────────────────────────────────────

def parse_structured_body(body: str) -> tuple[str, str]:
    """
    Parses bodies that use [DESCRIPTION] / [RESOLUTION] / [CONVERSATION] section markers.
    Returns (description_text, thread_text).
    The [CONVERSATION] section starts with a tab-separated data line which is skipped.
    """
    desc_match = re.search(
        r'\[DESCRIPTION\]\s*\n(.*?)(?=\[RESOLUTION\]|\[CONVERSATION\]|\Z)',
        body, re.DOTALL
    )
    description = desc_match.group(1).strip() if desc_match else ''

    conv_match = re.search(r'\[CONVERSATION\]\s*\n(.*)', body, re.DOTALL)
    if conv_match:
        thread_raw = conv_match.group(1)
        lines = thread_raw.split('\n')
        # First line is a tab-separated data row — skip it
        thread_text = '\n'.join(lines[1:]) if lines and '\t' in lines[0] else thread_raw
    else:
        thread_text = body

    return description, thread_text


# ─────────────────────────────────────────────
# 2b. TAB-SEPARATED DATA LINE PARSER
# ─────────────────────────────────────────────

def parse_data_line(body: str) -> tuple[dict, str]:
    """
    Parses the first tab-separated line (raw Lotus export row).
    Columns: index | client_id | version | system | reference | title | description_raw | thread_start
    Returns (data_dict, remaining_thread_text).
    """
    data = {}
    lines = body.split('\n')
    first_line = lines[0].strip()

    parts = first_line.split('\t')

    column_map = {
        0: 'row_index',
        1: 'client_id',
        2: 'version',
        3: 'system',
        4: 'reference',
        5: 'title',
        6: 'description_raw',
        7: 'thread_raw_start',
    }

    for idx, col_name in column_map.items():
        if idx < len(parts):
            data[col_name] = parts[idx].strip()

    # Clean description: :::: is a line separator in Lotus export
    if 'description_raw' in data:
        desc = data['description_raw']
        desc = re.sub(r'::{2,}', '\n', desc)
        data['description_clean'] = desc.strip()
        del data['description_raw']

    # Remaining body = everything after the first line
    remaining = '\n'.join(lines[1:])
    return data, remaining


# ─────────────────────────────────────────────
# 3. CONVERSATION THREAD PARSER
# ─────────────────────────────────────────────

TIMESTAMP_PATTERN = re.compile(
    r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})'  # datetime
    r'(?:\s+\w+)?'                                 # optional timezone (CEDT, etc.)
    r'\s*[-–]?\s*'
    r'(.*)'                                         # rest of line
)

def parse_thread(thread_text: str) -> list[dict]:
    """
    Parses the full conversation thread into structured events.
    Each event has: timestamp, actor, action, content, patches.
    """
    # Split on the separator line
    segments = re.split(r'_{10,}', thread_text)
    events = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        event_lines = segment.splitlines()
        current_event = None
        buffer = []

        for line in event_lines:
            line = line.strip()
            ts_match = TIMESTAMP_PATTERN.match(line)

            if ts_match:
                # Save previous event
                if current_event is not None:
                    current_event['content'] = '\n'.join(buffer).strip()
                    events.append(current_event)

                timestamp_str = ts_match.group(1)
                rest = ts_match.group(2).strip()

                # Determine actor and action
                actor = 'unknown'
                action = rest
                rest_lower = rest.lower()

                SUPPORT_SIGNALS = [
                    'support a', 'support b', 'hot line', 'hotline',
                    'archived', 'archiv', 'papfrint', 'colter', 'payfr',
                    'h2 pap', 'h1 pap', 'h3 pap', 'status co', 'status cp',
                    'status ci', 'status cu', 'status cx', 'status ch',
                    'status cq', 'status cy', 'status cn', 'status ak',
                ]
                CLIENT_SIGNALS = [
                    'monsieur', 'madame', 'mme ', 'mr ', 'client :',
                    "j'ai pris connaissance", 'votre réponse ne me convient',
                    "complément d'information", "création de l'évènement",
                ]

                if any(s in rest_lower for s in SUPPORT_SIGNALS):
                    actor = 'support'
                elif any(s in rest_lower for s in CLIENT_SIGNALS):
                    actor = 'client'
                elif re.search(r'Status\s+C[A-Z]', rest):
                    actor = 'support'

                # Extract status code if present
                status_match = re.search(r'Status\s+(\w+)\s*[-–]?\s*(.*)', rest)
                status_code = status_match.group(1) if status_match else None
                status_desc = status_match.group(2).strip() if status_match else None

                current_event = {
                    'timestamp': timestamp_str,
                    'actor': actor,
                    'action': action,
                    'status_code': status_code,
                    'status_description': status_desc,
                    'patches': [],
                    'content': ''
                }
                buffer = []

            else:
                # Collect patches
                if current_event is not None:
                    patch_match = re.match(r'^Patches?\s*[:：]?\s*(\d+)', line, re.IGNORECASE)
                    inline_patch = re.match(r'^Patch\s+(\d+)$', line, re.IGNORECASE)

                    if patch_match:
                        current_event['patches'].append(patch_match.group(1))
                    elif inline_patch:
                        current_event['patches'].append(inline_patch.group(1))
                    elif line and line not in ('Reply :', 'Hot Line :', 'Follow-up :'):
                        buffer.append(line)

        # Don't forget the last event
        if current_event is not None:
            current_event['content'] = '\n'.join(buffer).strip()
            events.append(current_event)

    return events


# ─────────────────────────────────────────────
# 4. RESOLUTION EXTRACTOR
# ─────────────────────────────────────────────

def extract_resolution(events: list[dict]) -> str:
    """
    Finds the most technically meaningful support reply as the resolution.
    This is the KEY FIELD for RAG — it's the answer we want to retrieve.

    Strategy:
    - Prefer replies that mention patches, technical terms, or instructions
    - Skip courtesy-only closings ("Nous procédons à la fermeture...")
    - Fall back to the last substantive support message
    """
    COURTESY_PATTERNS = [
        r'nous proc[eé]dons.{0,40}fermeture',
        r'nous vous remercions de votre retour',
        r'bien [àa] vous\.',
        r'^cordialement\.$',
        r'remettons l.{0,10}v[eé]nement en attente',
    ]

    TECHNICAL_SIGNALS = [
        r'patch\s+\d+',
        r'int[eé]grer',
        r'erreur',
        r'rubrique',
        r'param[eè]tre',
        r'configuration',
        r'version',
        r'veuillez',
        r'proc[eé]dure',
    ]

    def is_courtesy_only(text: str) -> bool:
        t = text.lower()
        for pattern in COURTESY_PATTERNS:
            if re.search(pattern, t):
                # Check if there's also technical content
                for sig in TECHNICAL_SIGNALS:
                    if re.search(sig, t):
                        return False
                return True
        return False

    def technical_score(text: str) -> int:
        t = text.lower()
        return sum(1 for sig in TECHNICAL_SIGNALS if re.search(sig, t))

    candidates = []
    # First pass: look for support actor events
    for event in events:
        if event['actor'] in ('support', 'unknown') and event['content']:
            content = event['content']
            if len(content) > 30 and not is_courtesy_only(content):
                score = technical_score(content)
                # Boost score if actor is explicitly support
                if event['actor'] == 'support':
                    score += 5
                candidates.append((score, event['timestamp'], content))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    # Fallback: last event with any content
    for event in reversed(events):
        if event['content'] and len(event['content']) > 30:
            return event['content']

    return ''


# ─────────────────────────────────────────────
# 5. MAIN PARSER
# ─────────────────────────────────────────────

def parse_ticket(filepath: str) -> dict:
    """
    Full pipeline: raw .txt → structured JSON dict.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()

    # Step 1: frontmatter
    metadata, body = parse_frontmatter(raw_text)

    # Step 2: parse body — section-based format or raw tab-separated format
    if '[DESCRIPTION]' in body:
        description, thread_text = parse_structured_body(body)
        data_fields = {}
    else:
        data_fields, thread_text = parse_data_line(body)
        description = data_fields.get('description_clean', '')

    # Step 3: conversation thread
    events = parse_thread(thread_text)

    # Step 4: extract resolution
    resolution = extract_resolution(events)

    # Step 5: collect all patches mentioned
    all_patches = []
    for event in events:
        all_patches.extend(event.get('patches', []))
    all_patches = list(dict.fromkeys(all_patches))  # deduplicate, preserve order

    # Step 6: assemble final JSON
    ticket = {
        # Identity
        'reference': metadata.get('reference', data_fields.get('reference', '')),
        'title': metadata.get('title', data_fields.get('title', '')),
        'client_id': data_fields.get('client_id', ''),

        # Technical context
        'version': metadata.get('version', data_fields.get('version', '')),
        'system': metadata.get('system', data_fields.get('system', '')),
        'support_team': metadata.get('support_team', ''),
        'team_source': metadata.get('team_source', ''),

        # Problem
        'description': description,

        # Resolution (KEY FIELD for RAG)
        'resolution': resolution,

        # Closure
        'closing_status_code': metadata.get('closing_status_code', ''),
        'closing_status_explanation': metadata.get('closing_status_explanation', ''),
        'closing_level': metadata.get('closing_level', ''),
        'closing_teamcode': metadata.get('closing_teamcode', ''),

        # Patches applied
        'patches': all_patches,

        # Full conversation (for context retrieval)
        'conversation': events,

        # Meta
        'confidence': float(metadata.get('confidence', 0)),
        'content_hash': metadata.get('content_hash', ''),
        'source_file': os.path.basename(filepath),
        'parsed_at': datetime.utcnow().isoformat() + 'Z',
    }

    return ticket


# ─────────────────────────────────────────────
# 6. BATCH RUNNER
# ─────────────────────────────────────────────

def run_batch(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob('*.txt'))

    print(f"\n📂 Found {len(files)} ticket(s) to process\n")

    success, failed = [], []

    for filepath in sorted(files):
        try:
            ticket = parse_ticket(str(filepath))
            out_file = output_dir / (filepath.stem + '.json')
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(ticket, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {filepath.name} → {out_file.name}")
            success.append(filepath.name)
        except Exception as e:
            print(f"  ❌ {filepath.name} → FAILED: {e}")
            failed.append({'file': filepath.name, 'error': str(e)})

    # Write failure report
    if failed:
        report_path = output_dir / '_parse_failures.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {len(failed)} failure(s) logged → {report_path}")

    print(f"\n✅ Done: {len(success)} parsed | ❌ {len(failed)} failed")
    return success, failed


# ─────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Lotus ticket .txt files to JSON')
    parser.add_argument('--input', required=True, help='Path to a .txt file or folder of .txt files')
    parser.add_argument('--output', required=True, help='Output folder for JSON files')
    args = parser.parse_args()

    run_batch(args.input, args.output)