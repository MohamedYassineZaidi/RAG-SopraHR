#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clear_db.py
===========
Clear all tickets from MongoDB (for testing/reset).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import init_db, close_db, clear_all_tickets


async def main():
    print("=" * 70)
    print("CLEAR MONGODB TICKETS")
    print("=" * 70)
    
    print("\n[1] Connecting to MongoDB...")
    try:
        await init_db()
        print("    [OK] Connected")
    except Exception as e:
        print(f"    [ERROR] {e}")
        return 1
    
    print("\n[2] Clearing tickets collection...")
    try:
        count = await clear_all_tickets()
        print(f"    [OK] Deleted {count} tickets")
    except Exception as e:
        print(f"    [ERROR] {e}")
        await close_db()
        return 1
    
    print("\n[3] Closing database...")
    await close_db()
    print("    [OK] Closed")
    
    print("\n" + "=" * 70)
    print("Database cleared! Ready for fresh import.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
