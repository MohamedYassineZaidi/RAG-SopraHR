#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_pipeline.py
================
Automates ticket ingestion and RAG index refresh.

It supports two input types:
1) Raw Lotus export files (.txt) -> parsed by tickets_pipeline.py
2) Simple ticket files (.json or .txt) -> converted to canonical .txt ticket format

Then it runs:
- txt_to_json.convert_all
- rag_assistant.build_indexes
- bm25_rag.build_bm25_index
- vectorless_rag.build_pageindex

Examples
--------
One-shot for a single raw export:
  python scripts/auto_pipeline.py --raw-file data/raw/new_export.txt

One-shot for a simple ticket JSON:
  python scripts/auto_pipeline.py --simple-file data/inbox/simple/new_ticket.json

Watch mode (polling):
  python scripts/auto_pipeline.py --watch --interval 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from tickets_pipeline import load_excel_mapping, process_raw_file
from txt_to_json import convert_all
from vector_rag import build_indexes
from bm25_rag import build_bm25_index
from vectorless_rag import build_pageindex


ROOT = _SCRIPTS.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_SIMPLE_DIR = ROOT / "data" / "inbox" / "simple"
DEFAULT_OUT_DIR = ROOT / "data" / "output"
DEFAULT_JSON_DIR = ROOT / "data" / "json"
DEFAULT_INDEX_CSV = ROOT / "data" / "tickets_index.csv"
DEFAULT_DB_DIR = ROOT / "data" / "indexes"
DEFAULT_BM25_DIR = ROOT / "data" / "bm25"
DEFAULT_PAGEINDEX_DIR = ROOT / "data" / "pageindex"
DEFAULT_STATE_FILE = ROOT / "data" / ".auto_pipeline_state.json"
DATA_DIR = ROOT / "data"
DEFAULT_BATCH_SIZE = 256


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _safe_ref(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]", "", ref).strip() or "FR WAUTO"


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _resolve_under_data(path: Path, arg_name: str) -> Path:
    """Resolve a CLI-provided path and ensure it stays under project data/."""
    resolved = path.expanduser().resolve(strict=False)
    base = DATA_DIR.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must stay under {base}: {resolved}") from exc
    return resolved


def _load_state(path: Path) -> dict[str, float]:
    path = _resolve_under_data(path, "state_file")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict[str, float]) -> None:
    path = _resolve_under_data(path, "state_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _ticket_txt_from_simple_payload(payload: dict[str, Any], source: Path, out_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    reference = _safe_ref(str(payload.get("reference") or f"FR WAUTO {ts}"))
    title = str(payload.get("title") or payload.get("subject") or "Ticket imported")
    version = str(payload.get("version") or "Unknown")
    system = str(payload.get("system") or "Unknown")
    team = str(payload.get("support_team") or payload.get("team") or "DSN")
    description = str(payload.get("description") or payload.get("problem") or "")
    resolution = str(payload.get("resolution") or payload.get("solution") or "")

    patches = payload.get("patches") or []
    if isinstance(patches, list):
        patches_str = ";".join(str(p).strip() for p in patches if str(p).strip())
    else:
        patches_str = str(patches)

    content_hash = datetime.now().strftime("simple_%Y%m%d%H%M%S")

    frontmatter = (
        "---\n"
        f"reference: {reference}\n"
        f"title: {title}\n"
        f"version: {version}\n"
        f"system: {system}\n"
        f"support_team: {team}\n"
        "team_source: manual_simple\n"
        "confidence: 1.0\n"
        f"content_hash: {content_hash}\n"
        "closing_status_code: C\n"
        "closing_status_explanation: Closed\n"
        "closing_level: 3\n"
        "closing_teamcode: H2\n"
        f"patches: {patches_str}\n"
        "---\n\n"
    )

    body = (
        "[DESCRIPTION]\n"
        f"{description}\n\n"
        "[RESOLUTION]\n"
        f"{resolution}\n\n"
        "[CONVERSATION]\n"
        f"Imported from simple source: {source.name}\n"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _sanitize_filename(f"simple_{ts}_{reference.replace(' ', '_')}.txt")
    out_path = out_dir / fname
    out_path.write_text(frontmatter + body, encoding="utf-8")
    return out_path


def _ingest_simple_file(simple_file: Path, out_dir: Path) -> Path:
    if simple_file.suffix.lower() == ".json":
        payload = json.loads(simple_file.read_text(encoding="utf-8"))
        return _ticket_txt_from_simple_payload(payload, simple_file, out_dir)

    # Plain text fallback: treat content as description and create minimal ticket.
    payload = {
        "title": simple_file.stem,
        "description": simple_file.read_text(encoding="utf-8", errors="ignore"),
        "resolution": "To be analyzed",
        "support_team": "DSN",
    }
    return _ticket_txt_from_simple_payload(payload, simple_file, out_dir)


def _rebuild_all_indexes(
) -> None:
    # Use fixed managed directories to avoid untrusted path influence.
    out_dir = DEFAULT_OUT_DIR
    json_dir = DEFAULT_JSON_DIR
    db_dir = DEFAULT_DB_DIR
    bm25_dir = DEFAULT_BM25_DIR
    pageindex_dir = DEFAULT_PAGEINDEX_DIR

    _log("[rebuild] Converting txt -> json")
    convert_all(out_dir, json_dir)

    _log("[rebuild] Building vector index")
    build_indexes(json_dir, db_dir, batch_size=DEFAULT_BATCH_SIZE)

    _log("[rebuild] Building BM25 index")
    build_bm25_index(json_dir, bm25_dir)

    _log("[rebuild] Building PageIndex")
    build_pageindex(json_dir, pageindex_dir)
    _log("[rebuild] All indexes updated successfully")


def _process_once(
    raw_files: list[Path],
    simple_files: list[Path],
    out_dir: Path,
    index_csv: Path,
    product_excel: Path | None,
    json_dir: Path,
    db_dir: Path,
    bm25_dir: Path,
    pageindex_dir: Path,
) -> bool:
    t0 = time.time()
    changed = False

    _log(
        "[pipeline] Starting run with "
        f"raw_files={len(raw_files)} simple_files={len(simple_files)}"
    )

    excel_maps = load_excel_mapping(product_excel)

    for raw_file in raw_files:
        _log(f"[raw] Processing {raw_file}")
        total, created, updated = process_raw_file(raw_file, out_dir, index_csv, excel_maps)
        _log(f"[raw] Done tickets={total} created={created} updated={updated}")
        changed = True

    for simple_file in simple_files:
        _log(f"[simple] Ingesting {simple_file}")
        created_txt = _ingest_simple_file(simple_file, out_dir)
        _log(f"[simple] Created {created_txt.name}")
        changed = True

    if changed:
        _rebuild_all_indexes()
    else:
        _log("[pipeline] No new files detected")

    elapsed = time.time() - t0
    _log(f"[pipeline] Run finished in {elapsed:.2f}s")

    return changed


def _iter_candidate_files(raw_dir: Path, simple_dir: Path) -> tuple[list[Path], list[Path]]:
    raw_files = sorted(raw_dir.glob("*.txt")) if raw_dir.exists() else []
    simple_files = []
    if simple_dir.exists():
        simple_files.extend(sorted(simple_dir.glob("*.json")))
        simple_files.extend(sorted(simple_dir.glob("*.txt")))
    return raw_files, simple_files


def main() -> None:
    ap = argparse.ArgumentParser(description="Automate ingestion + RAG index refresh")
    ap.add_argument("--raw-file", type=Path, default=None, help="Single raw export .txt file")
    ap.add_argument("--simple-file", type=Path, default=None, help="Single simple ticket (.json or .txt)")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw ticket folder for watch mode")
    ap.add_argument("--simple-dir", type=Path, default=DEFAULT_SIMPLE_DIR, help="Simple ticket folder for watch mode")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Canonical txt output dir")
    ap.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR, help="JSON output dir")
    ap.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV, help="CSV index path")
    ap.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Vector index dir")
    ap.add_argument("--bm25-dir", type=Path, default=DEFAULT_BM25_DIR, help="BM25 index dir")
    ap.add_argument("--pageindex-dir", type=Path, default=DEFAULT_PAGEINDEX_DIR, help="PageIndex dir")
    ap.add_argument("--product-excel", type=Path, default=None, help="Optional product code mapping Excel")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size for vector index")
    ap.add_argument("--watch", action="store_true", help="Poll folders for new files")
    ap.add_argument("--interval", type=int, default=20, help="Polling interval in seconds")
    ap.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Watch state file")
    args = ap.parse_args()

    # Security hardening: keep all write/index/state paths managed by the project.
    # This prevents untrusted CLI path overrides from reaching serialization/index sinks.
    args.raw_dir = DEFAULT_RAW_DIR
    args.simple_dir = DEFAULT_SIMPLE_DIR
    args.out_dir = DEFAULT_OUT_DIR
    args.json_dir = DEFAULT_JSON_DIR
    args.index_csv = DEFAULT_INDEX_CSV
    args.db_dir = DEFAULT_DB_DIR
    args.bm25_dir = DEFAULT_BM25_DIR
    args.pageindex_dir = DEFAULT_PAGEINDEX_DIR
    args.state_file = DEFAULT_STATE_FILE
    args.product_excel = None
    args.batch = DEFAULT_BATCH_SIZE

    try:
        # Constrain all writable/index paths under project data/.
        args.raw_dir = _resolve_under_data(args.raw_dir, "--raw-dir")
        args.simple_dir = _resolve_under_data(args.simple_dir, "--simple-dir")
        args.out_dir = _resolve_under_data(args.out_dir, "--out-dir")
        args.json_dir = _resolve_under_data(args.json_dir, "--json-dir")
        args.index_csv = _resolve_under_data(args.index_csv, "--index-csv")
        args.db_dir = _resolve_under_data(args.db_dir, "--db-dir")
        args.bm25_dir = _resolve_under_data(args.bm25_dir, "--bm25-dir")
        args.pageindex_dir = _resolve_under_data(args.pageindex_dir, "--pageindex-dir")
        args.state_file = _resolve_under_data(args.state_file, "--state-file")
        if args.product_excel is not None:
            args.product_excel = _resolve_under_data(args.product_excel, "--product-excel")

        # Single-file mode: ensure file args remain under their inbox roots.
        if args.raw_file is not None:
            args.raw_file = _resolve_under_data(args.raw_file, "--raw-file")
            args.raw_file.relative_to(args.raw_dir)
            if args.raw_file.suffix.lower() != ".txt":
                raise ValueError("--raw-file must be a .txt file")
        if args.simple_file is not None:
            args.simple_file = _resolve_under_data(args.simple_file, "--simple-file")
            args.simple_file.relative_to(args.simple_dir)
            if args.simple_file.suffix.lower() not in {".json", ".txt"}:
                raise ValueError("--simple-file must be .json or .txt")
    except (ValueError, OSError) as exc:
        ap.error(str(exc))

    if args.raw_file or args.simple_file:
        raw_files = [args.raw_file] if args.raw_file else []
        simple_files = [args.simple_file] if args.simple_file else []
        _process_once(
            raw_files=raw_files,
            simple_files=simple_files,
            out_dir=args.out_dir,
            index_csv=args.index_csv,
            product_excel=args.product_excel,
            json_dir=args.json_dir,
            db_dir=args.db_dir,
            bm25_dir=args.bm25_dir,
            pageindex_dir=args.pageindex_dir,
        )
        return

    if not args.watch:
        _log("Nothing to do. Use --watch or provide --raw-file / --simple-file.")
        return

    state = _load_state(args.state_file)
    _log(f"[watch] watching raw={args.raw_dir} simple={args.simple_dir} interval={args.interval}s")

    while True:
        try:
            raw_files, simple_files = _iter_candidate_files(args.raw_dir, args.simple_dir)
            new_raw: list[Path] = []
            new_simple: list[Path] = []

            for fp in raw_files + simple_files:
                key = str(fp.resolve())
                mtime = fp.stat().st_mtime
                if state.get(key) != mtime:
                    if fp in raw_files:
                        new_raw.append(fp)
                    else:
                        new_simple.append(fp)
                    state[key] = mtime

            changed = _process_once(
                raw_files=new_raw,
                simple_files=new_simple,
                out_dir=args.out_dir,
                index_csv=args.index_csv,
                product_excel=args.product_excel,
                json_dir=args.json_dir,
                db_dir=args.db_dir,
                bm25_dir=args.bm25_dir,
                pageindex_dir=args.pageindex_dir,
            )

            if changed:
                _save_state(args.state_file, state)
                _log(f"[watch] State saved to {args.state_file}")

            time.sleep(args.interval)
        except KeyboardInterrupt:
            _log("Stopped.")
            break


if __name__ == "__main__":
    main()
