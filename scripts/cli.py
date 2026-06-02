#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cli.py
======
Command-line interface for the Sopra HR ReAct support agent.

Usage
-----
  # Interactive session
  python scripts/cli.py --team DSN

  # Single question (non-interactive)
  python scripts/cli.py --team DSN --question "ORA-00942 lors REGDSN"

  # Custom index paths
  python scripts/cli.py --team Appli \\
      --db data/indexes \\
      --bm25 data/bm25 \\
      --index data/pageindex \\
      --verbose

  # Pipe output to file
  python scripts/cli.py --team DSN -q "erreur REGDSN" --no-pretty > result.json
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add scripts/ to path so sibling imports work when cli.py is run
# from the project root (e.g. python scripts/cli.py).
_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from agent import create_rag_agent, run_query
from utils.validators import validate_agent_output
from utils.formatting import pretty_print_result


# ─────────────────────────────────────────────
# DEFAULTS  (relative to project root)
# ─────────────────────────────────────────────

_ROOT      = _SCRIPTS.parent
_DB_DIR    = _ROOT / "data" / "indexes"
_BM25_DIR  = _ROOT / "data" / "bm25"
_INDEX_DIR = _ROOT / "data" / "pageindex"

VALID_TEAMS = ["DSN", "Appli", "Outils"]


# ─────────────────────────────────────────────
# PATH VALIDATION
# ─────────────────────────────────────────────

def _resolve_under_data(path: str, name: str) -> Path:
    """
    Validates that a CLI-provided path stays within the project's data/ directory.
    Prevents path traversal attacks via CLI arguments.

    Parameters
    ----------
    path : str
        The path provided by the user (from CLI argument)
    name : str
        The argument name (for error messages)

    Returns
    -------
    Path
        The canonicalized, validated path

    Raises
    ------
    ValueError
        If the path escapes the data/ directory
    """
    user_path = Path(path).resolve()
    data_dir = (_ROOT / "data").resolve()
    
    try:
        # Check if user_path is under data_dir
        user_path.relative_to(data_dir)
    except ValueError:
        raise ValueError(
            f"[ERROR] {name} must be under {data_dir}\n"
            f"       Provided: {user_path}\n"
            f"       This prevents path traversal attacks."
        )
    
    return user_path


# ─────────────────────────────────────────────
# CORE RUN FUNCTION
# ─────────────────────────────────────────────

def run_agent(
    question: str,
    team: str,
    db_dir: Path,
    bm25_dir: Path,
    index_dir: Path,
    executor,
    pretty: bool = True,
) -> dict:
    """
    Runs a single question through the agent and returns the output dict.

    Parameters
    ----------
    question   : free-text support question
    team       : "DSN" | "Appli" | "Outils"
    db_dir     : FAISS index directory
    bm25_dir   : BM25 index directory
    index_dir  : PageIndex directory
    executor   : pre-built AgentExecutor (avoids reloading indexes each call)
    pretty     : if True, pretty-print the result to stdout
    """
    result = run_query(executor, question)
    warnings = validate_agent_output(result)

    if pretty:
        pretty_print_result(question, team, result, warnings)

    return result


# ─────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────

def interactive_loop(executor, team: str) -> None:
    """Runs an interactive REPL until the user types 'quit' or 'exit'."""
    print(f"\n{'═'*65}")
    print(f"  Sopra HR Support Agent  |  Équipe: {team}")
    print(f"  Tapez votre question. 'quit' pour quitter.")
    print(f"{'═'*65}\n")

    while True:
        try:
            question = input("❓  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir.")
            break

        run_agent(
            question=question,
            team=team,
            db_dir=Path("."),   # unused — executor already loaded
            bm25_dir=Path("."),
            index_dir=Path("."),
            executor=executor,
            pretty=True,
        )


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sopra HR ReAct Support Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/cli.py --team DSN
              python scripts/cli.py --team DSN -q "ORA-00942 table inexistante"
              python scripts/cli.py --team Appli --verbose -q "rubrique absente"
        """),
    )

    parser.add_argument(
        "--team", "-t",
        required=True,
        choices=VALID_TEAMS,
        help="Support team: DSN | Appli | Outils",
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Single question (non-interactive mode)",
    )
    parser.add_argument(
        "--db",
        default=str(_DB_DIR),
        help=f"FAISS indexes directory (default: {_DB_DIR})",
    )
    parser.add_argument(
        "--bm25",
        default=str(_BM25_DIR),
        help=f"BM25 indexes directory (default: {_BM25_DIR})",
    )
    parser.add_argument(
        "--index",
        default=str(_INDEX_DIR),
        help=f"PageIndex directory (default: {_INDEX_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show ReAct chain (Thought/Action/Observation)",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Output raw JSON only (useful for piping)",
    )

    args = parser.parse_args()

    # Validate and canonicalize paths to prevent path traversal attacks
    try:
        db_dir    = _resolve_under_data(args.db, "--db")
        bm25_dir  = _resolve_under_data(args.bm25, "--bm25")
        index_dir = _resolve_under_data(args.index, "--index")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # Validate paths exist
    for name, p in [("--db", db_dir), ("--bm25", bm25_dir), ("--index", index_dir)]:
        if not p.exists():
            print(f"[ERROR] Path {name} does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"🔧 Chargement des indexes ({args.team})…")
    try:
        executor = create_rag_agent(
            team=args.team,
            db_dir=db_dir,
            bm25_dir=bm25_dir,
            index_dir=index_dir,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] Index manquant: {exc}", file=sys.stderr)
        print(
            "Construisez d'abord les indexes:\n"
            "  python scripts/rag_assistant.py index --json data/json --db data/indexes\n"
            "  python scripts/bm25_rag.py build --json data/json --db data/bm25\n"
            "  python scripts/vectorless_rag.py build --json data/json --index data/pageindex",
            file=sys.stderr,
        )
        sys.exit(1)

    pretty = not args.no_pretty

    if args.question:
        result = run_agent(
            question=args.question,
            team=args.team,
            db_dir=db_dir,
            bm25_dir=bm25_dir,
            index_dir=index_dir,
            executor=executor,
            pretty=pretty,
        )
        if not pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        interactive_loop(executor, args.team)


if __name__ == "__main__":
    main()
