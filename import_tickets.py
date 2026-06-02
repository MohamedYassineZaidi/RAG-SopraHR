#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import_tickets.py
=================
Bulk import JSON tickets from data/json/ into MongoDB.

Usage:
  python import_tickets.py
  python import_tickets.py --limit 5  (import only 5 files for testing)
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from scripts.models import TicketCreate, ConversationEntry
from scripts.database import init_db, bulk_insert_tickets, close_db

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "json"
CLEANED_DIR = ROOT / "data" / "cleaned"


# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def parse_ticket_from_json(json_data: dict) -> TicketCreate:
    """Convert raw JSON to Pydantic TicketCreate model."""
    
    # Parse conversation entries
    conversation = []
    if "conversation" in json_data:
        for conv in json_data["conversation"]:
            try:
                # Convert ISO string to datetime if needed
                timestamp = conv.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                entry = ConversationEntry(
                    timestamp=timestamp,
                    actor=conv.get("actor", "support"),
                    author=conv.get("author", "Unknown"),
                    type=conv.get("type", "message"),
                    action=conv.get("action"),
                    text=conv.get("text")
                )
                conversation.append(entry)
            except Exception as e:
                print(f"⚠ Warning: Failed to parse conversation entry: {e}")
    
    # Create ticket
    ticket = TicketCreate(
        reference=json_data.get("reference", "UNKNOWN"),
        title=json_data.get("title", "Untitled"),
        source_file=json_data.get("source_file"),
        version=json_data.get("version"),
        system=json_data.get("system"),
        site_env=json_data.get("site_env"),
        support_team=json_data.get("support_team"),
        team_source=json_data.get("team_source"),
        closing_teamcode=json_data.get("closing_teamcode"),
        closing_status_code=json_data.get("closing_status_code"),
        closing_status_explanation=json_data.get("closing_status_explanation"),
        closing_level=json_data.get("closing_level"),
        is_closed=json_data.get("is_closed", False),
        description=json_data.get("description", ""),
        resolution=json_data.get("resolution"),
        espdsn_version=json_data.get("espdsn_version"),
        patches=json_data.get("patches", []),
        conversation=conversation,
        message_count=json_data.get("message_count", 0),
        confidence=json_data.get("confidence"),
        content_hash=json_data.get("content_hash")
    )
    
    return ticket


async def import_tickets(limit: int = None) -> None:
    """Bulk import all JSON tickets from data/json/ or data/cleaned/ into MongoDB."""
    
    await init_db()
    
    # Try data/cleaned first, fall back to data/json
    source_dir = CLEANED_DIR if CLEANED_DIR.exists() else DATA_DIR
    
    if not source_dir.exists():
        print(f"[ERROR] Directory not found: {source_dir}")
        await close_db()
        return
    
    # Find all JSON files
    json_files = sorted(source_dir.glob("*.json"))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {source_dir}")
        await close_db()
        return
    
    if limit:
        json_files = json_files[:limit]
    
    print(f"[INFO] Found {len(json_files)} JSON files to import from {source_dir.name}/")
    
    tickets_to_insert = []
    failed_count = 0
    duplicate_count = 0
    
    # Parse all tickets
    for i, json_file in enumerate(json_files, 1):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            ticket = parse_ticket_from_json(json_data)
            tickets_to_insert.append(ticket)
            
            if i % 10 == 0 or i == len(json_files):
                print(f"[OK] Parsed {i}/{len(json_files)} tickets")
        
        except Exception as e:
            print(f"[ERROR] Failed to parse {json_file.name}: {e}")
            failed_count += 1
    
    print(f"\n[INFO] Successfully parsed {len(tickets_to_insert)} tickets")
    print(f"[INFO] Inserting into MongoDB...")
    
    # Bulk insert
    try:
        inserted_count = await bulk_insert_tickets(tickets_to_insert)
        print(f"[OK] Successfully inserted {inserted_count} tickets into MongoDB")
    except Exception as e:
        print(f"[ERROR] Bulk insert failed: {e}")
    
    print(f"[INFO] Failed to parse: {failed_count} files")
    
    await close_db()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bulk import JSON tickets into MongoDB")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to import (for testing)")
    args = parser.parse_args()
    
    await import_tickets(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
