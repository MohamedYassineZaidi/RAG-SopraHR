#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import_tickets.py
================
Script to bulk import JSON tickets from data/json/ into MongoDB.

Usage:
  python scripts/import_tickets.py [--limit N]  # Import first N files
  python scripts/import_tickets.py              # Import all files
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TicketCreate, ConversationEntry
from database import init_db, close_db, bulk_insert_tickets


async def load_json_tickets(json_dir: Path, limit: int = None) -> List[TicketCreate]:
    """Load and parse JSON ticket files."""
    tickets = []
    json_files = sorted(json_dir.glob("*.json"))
    
    if limit:
        json_files = json_files[:limit]
    
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    
    for i, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse conversation entries
            conversation = []
            if "conversation" in data and data["conversation"]:
                for entry in data["conversation"]:
                    try:
                        conv = ConversationEntry(**entry)
                        conversation.append(conv)
                    except Exception as e:
                        print(f"  [WARNING] Skipped conversation entry in {file_path.name}: {e}")
                        continue
            
            # Create ticket object with parsed conversation
            ticket = TicketCreate(
                reference=data.get("reference", "UNKNOWN"),
                title=data.get("title", ""),
                source_file=data.get("source_file"),
                version=data.get("version"),
                system=data.get("system"),
                site_env=data.get("site_env"),
                support_team=data.get("support_team"),
                team_source=data.get("team_source"),
                closing_teamcode=data.get("closing_teamcode"),
                closing_status_code=data.get("closing_status_code"),
                closing_status_explanation=data.get("closing_status_explanation"),
                closing_level=data.get("closing_level"),
                is_closed=data.get("is_closed", False),
                description=data.get("description", ""),
                resolution=data.get("resolution"),
                patches=data.get("patches", []),
                conversation=conversation,
                message_count=data.get("message_count", 0),
                confidence=data.get("confidence"),
                content_hash=data.get("content_hash"),
            )
            tickets.append(ticket)
            print(f"  [{i:3d}] {ticket.reference:20s} - Loaded")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load {file_path.name}: {e}")
            continue
    
    print(f"\nSuccessfully parsed {len(tickets)} tickets")
    return tickets


async def main():
    """Main import function."""
    # Parse command line args
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            if sys.argv[1] == "--limit" and len(sys.argv) > 2:
                limit = int(sys.argv[2])
    
    print("=" * 70)
    print("TICKET IMPORT SCRIPT")
    print("=" * 70)
    
    # Initialize database
    print("\n[1] Connecting to MongoDB...")
    try:
        await init_db()
        print("    [OK] Connected")
    except Exception as e:
        print(f"    [ERROR] Connection failed: {e}")
        return 1
    
    # Load tickets
    print("\n[2] Loading JSON files...")
    json_dir = Path(__file__).parent.parent / "data" / "json"
    if not json_dir.exists():
        print(f"    [ERROR] Directory not found: {json_dir}")
        await close_db()
        return 1
    
    try:
        tickets = await load_json_tickets(json_dir, limit=limit)
    except Exception as e:
        print(f"    [ERROR] Failed to load tickets: {e}")
        await close_db()
        return 1
    
    if not tickets:
        print("    [WARNING] No tickets loaded")
        await close_db()
        return 0
    
    # Import to MongoDB
    print(f"\n[3] Importing {len(tickets)} tickets to MongoDB...")
    try:
        count = await bulk_insert_tickets(tickets)
        print(f"    [OK] Imported {count} tickets")
    except Exception as e:
        print(f"    [ERROR] Import failed: {e}")
        await close_db()
        return 1
    
    # Close connection
    print("\n[4] Closing database connection...")
    await close_db()
    print("    [OK] Closed")
    
    print("\n" + "=" * 70)
    print(f"SUCCESS: {count} tickets imported to 'soprahr_rag' database")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
