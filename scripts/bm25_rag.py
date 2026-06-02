#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bm25_rag.py
===========
BM25 keyword-based retrieval for Sopra HR support tickets.

Why BM25 alongside vector + PageIndex?
  Vector RAG uses semantic similarity — good for paraphrased queries.
  PageIndex uses LLM reasoning — good for structured navigation.
  BM25 uses exact term frequency — best for:
    - Error codes:    ORA-00942, R_SYSTEM, ZDAG-COMMO
    - Patch numbers:  ZY1234, ZYHRCT01
    - Product names:  REGDSN, HRCT, EspDSN, HRQuery
    - Version refs:   HRA 9.1, V01X10
  These are exactly the tokens consultants type when they know
  what they're looking for.

Architecture:
  Index  — BM25Okapi built from title + description + resolution
           of every ticket in a team. Saved as JSON per team.
  Query  — tokenizes the query, scores all tickets, returns top-K.
  Fusion — designed to be merged into hybrid_rag.py via RRF.

Commands:
  # Build BM25 indexes (run once after txt_to_json.py)
  python scripts/bm25_rag.py build --json data/json --db data/bm25

  # Test a single query
  python scripts/bm25_rag.py query --db data/bm25 --team DSN
  python scripts/bm25_rag.py query --db data/bm25 --team DSN \\
      --question "ORA-00942 REGDSN table inexistante"

  # Evaluate BM25 alone (self-retrieval)
  python scripts/bm25_rag.py evaluate --db data/bm25 --json data/json --team DSN

Requirements:
  pip install rank-bm25 python-dotenv
"""

import re
import json
import pickle
import time
import argparse
import random
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from rag_utils import TEAMS, TOP_K


_SCRIPTS = Path(__file__).parent
ROOT = _SCRIPTS.parent
DATA_DIR = ROOT / "data"
DEFAULT_JSON_DIR = DATA_DIR / "json"
DEFAULT_FAISS_DIR = DATA_DIR / "indexes"
DEFAULT_BM25_DIR = DATA_DIR / "bm25"


def _resolve_under_data(path: Path, arg_name: str) -> Path:
    """Resolve a path and ensure it stays inside project data/."""
    resolved = path.expanduser().resolve(strict=False)
    base = DATA_DIR.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must stay under {base}: {resolved}") from exc
    return resolved


def _extract_last_date(ticket: dict) -> str:
    """Extract the most recent timestamp from the ticket's conversation."""
    conversation = ticket.get("conversation", [])
    dates = [entry.get("timestamp", "") for entry in conversation if entry.get("timestamp")]
    if dates:
        return max(dates)
    return ""


# ─────────────────────────────────────────────
# FRENCH STOPWORDS
# BM25 works better when stopwords are removed — they add noise
# without discriminating between tickets.
# ─────────────────────────────────────────────

STOPWORDS = {
    # French
    "le", "la", "les", "de", "du", "des", "un", "une", "en", "et", "ou",
    "à", "au", "aux", "ce", "il", "elle", "nous", "vous", "ils", "elles",
    "est", "sont", "a", "ont", "être", "avoir", "non", "pas", "ne", "se",
    "si", "sur", "par", "pour", "avec", "dans", "que", "qui", "quoi", "dont",
    "je", "tu", "me", "te", "lui", "leur", "y", "en", "tout", "bien", "même",
    "lors", "suite", "après", "avant", "depuis", "sous", "entre", "vers",
    "bonjour", "cordialement", "merci", "madame", "monsieur", "cher", "chère",
    # English (tickets sometimes mix)
    "the", "and", "or", "is", "are", "to", "of", "in", "for", "on", "with",
    "this", "that", "it", "be", "was", "has", "have", "from", "at", "by",
    # Generic support phrases
    "ticket", "incident", "problème", "problem", "issue", "boite", "merci",
    "pouvez", "pouvons", "faire", "faut", "faut-il", "peut", "peuvent",
    "avons", "avez", "avait", "était", "serait", "pourrait",
}


# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Tokenizes text for BM25.

    Key decisions:
    - Lowercase everything
    - Keep error codes intact:  ORA-00942 stays as "ora-00942"
    - Keep alphanumeric tokens ≥ 2 chars
    - Remove stopwords
    - Split on whitespace AND common separators

    This means "ORA-00942" is one token, not "ORA" and "00942",
    which is what you want — "00942" alone is meaningless.
    """
    text = text.lower()

    # Normalise common separators to space (but keep hyphens inside tokens)
    text = re.sub(r'[/\\|;:,\(\)\[\]\{\}]', ' ', text)

    # Extract tokens: allow hyphens inside (for ORA-00942, R_SYSTEM etc.)
    # Pattern: alphanumeric + hyphen + underscore sequences of length >= 2
    tokens = re.findall(r'[a-z0-9][a-z0-9_\-]*[a-z0-9]|[a-z0-9]{2,}', text)

    # Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

    return tokens


# ─────────────────────────────────────────────
# INDEX — BUILD, SAVE, LOAD
# ─────────────────────────────────────────────

def ticket_to_bm25_text(ticket: dict) -> str:
    """
    Builds the text corpus entry for a ticket.

    Field weighting strategy:
    - Title is repeated 3× — it's the most reliable signal
    - Description once — full symptom text
    - Resolution once — action taken
    - Patches repeated 2× — exact patch numbers are high-value
    - Error codes extracted and repeated — ORA-*, R_SYSTEM, etc.
    - Conversation included once — captures error codes / module names
      that appear in client follow-ups (E-312, FSWDRE01, ZDAG-COMMO, etc.)

    Repetition simulates field boosting without changing BM25 internals.
    """
    title       = ticket.get("title", "")
    description = ticket.get("description", "")
    resolution  = ticket.get("resolution", "")
    patches     = " ".join(ticket.get("patches", []))
    version     = ticket.get("version", "")

    # NEW: Concatenate conversation text for technical-term coverage
    conv_parts = []
    for entry in (ticket.get("conversation") or []):
        text = entry.get("text") or ""
        if text and entry.get("type") not in ("status_change", "archive") and len(text) >= 20:
            conv_parts.append(text)
    conversation = "\n".join(conv_parts)

    # Extract error/product codes for extra boost — search BOTH the
    # title/description/resolution AND the conversation
    code_pattern = re.compile(
        r'\b(ORA-\d+|R_SYSTEM|[A-Z]{2,}[-_]\d{3,}|ZY\w+|ZX\w+|FSW\w+|'
        r'BAY\w+|NRB\w*|REGDSN|HRCT|ESPDSN|HRQUERY|HRASPACE|DADSN?U?|'
        r'E-\d+|W-\d+|ZDAG[-\.]?\w+)\b',
        re.IGNORECASE
    )
    codes = " ".join(code_pattern.findall(
        f"{title} {description} {resolution} {conversation}"
    ))

    # Field weights via repetition
    parts = [
        title, title, title,          # 3×
        description,                   # 1×
        resolution,                    # 1×
        patches, patches,              # 2×
        version,                       # 1×
        codes, codes, codes,           # 3× (extracted codes — extra boost)
        conversation,                  # 1× (full conversation text)
    ]
    return " ".join(p for p in parts if p)


def build_bm25_index(json_dir: Path, db_dir: Path, faiss_dir: Path | None = None):
    """
    Reads all tickets, groups by team, builds one BM25 index per team.
    Saves tokenized corpus + docstore to db_dir/{team}/bm25.json

    Source priority:
    1. faiss_dir  — if given, reads directly from FAISS docstores
                      (the authoritative 189k-ticket dataset, uses 'team' field)
      2. json_dir   — legacy: loads *.json files (uses 'support_team' field)

    Incremental: skips teams where all tickets are already indexed.
    """
    # Security hardening: keep paths managed by project defaults.
    # Caller-provided paths are intentionally ignored.
    _ = (json_dir, db_dir, faiss_dir)
    db_dir = _resolve_under_data(DEFAULT_BM25_DIR, "db_dir")
    json_dir = _resolve_under_data(DEFAULT_JSON_DIR, "json_dir")
    faiss_dir = _resolve_under_data(DEFAULT_FAISS_DIR, "faiss_dir") if DEFAULT_FAISS_DIR.exists() else None

    db_dir.mkdir(parents=True, exist_ok=True)

    tickets: list[dict] = []

    if faiss_dir is not None and faiss_dir.exists():
        # Load directly from FAISS docstores — same data the vector retriever uses
        print(f"\n📂 Loading tickets from FAISS docstores in {faiss_dir} ...")
        for team in TEAMS:
            pkl_f = _resolve_under_data(faiss_dir / team.lower() / "docstore.pkl", "faiss_docstore")
            if not pkl_f.exists():
                print(f"  ⚠️  {team}: {pkl_f} not found — skipping")
                continue
            with open(pkl_f, "rb") as fh:
                docs = pickle.load(fh)
            # FAISS docs already have 'team' field set correctly
            tickets.extend(docs)
            print(f"  {team}: loaded {len(docs):,} docs from FAISS")
        # team field in FAISS docs is 'team' (not 'support_team')
        team_field = "team"
    else:
        files = sorted(json_dir.glob("*.json"))
        print(f"\n📂 Loading {len(files)} JSON tickets from {json_dir} ...")
        for f in files:
            try:
                tickets.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"  ❌ {f.name}: {e}")
        # legacy JSON files use 'support_team'
        team_field = "support_team"

    print(f"  Total tickets loaded: {len(tickets):,}")

    # Group by team
    by_team = {t: [] for t in TEAMS}
    by_team["Unknown"] = []
    for ticket in tickets:
        team  = ticket.get(team_field) or ticket.get("support_team") or ticket.get("team") or "Unknown"
        group = team if team in TEAMS else "Unknown"
        by_team[group].append(ticket)

    print(f"\n  Team breakdown:")
    for team, ts in by_team.items():
        if ts:
            print(f"    {team:10s}  {len(ts):5d} tickets")

    for team in TEAMS:
        ts = by_team[team]
        if not ts:
            print(f"\n⚠️  No tickets for {team} — skipping")
            continue

        new_count = len(ts)

        print(f"\n{'─'*50}")
        print(f"🔍 Building BM25 index: {team} ({len(ts)} tickets, {new_count} new)")

        # Build corpus
        corpus    = [ticket_to_bm25_text(t) for t in ts]
        tokenized = [tokenize(text) for text in corpus]

        # Build BM25 model
        bm25 = BM25Okapi(tokenized)

        # Docstore — same compact format as vector_retrieve output
        docstore = []
        for t in ts:
            # FAISS docs have last_date directly; legacy JSON has conversation[]
            last_date = t.get("last_date") or _extract_last_date(t)
            docstore.append({
                "reference":   t.get("reference", ""),
                "title":       t.get("title", ""),
                "version":     t.get("version", ""),
                "description": t.get("description", ""),
                "resolution":  t.get("resolution", ""),
                "patches":     t.get("patches", []),
                "team":        t.get("team") or t.get("support_team", ""),
                "last_date":   last_date,
                "espdsn_version": t.get("espdsn_version", ""),
            })

        # Save
        out_dir = db_dir / team.lower()
        out_dir = _resolve_under_data(out_dir, "bm25_team_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _resolve_under_data(out_dir / "bm25.pkl", "bm25_team_index")
        out_payload = {
            "team": team,
            "tokenized": tokenized,
            "docstore": docstore,
        }
        with open(out_path, "wb") as f:
            pickle.dump(out_payload, f)

        # Stats
        avg_len = sum(len(t) for t in tokenized) / len(tokenized)
        print(f"  ✅ Saved → {out_path}")
        print(f"     Avg tokens/ticket: {avg_len:.0f}")
        print(f"     Vocabulary size:   {len(bm25.idf)}")

    print(f"\n🎉 BM25 indexes built in {db_dir}")
    print(f"   Run queries with:")
    print(f"   python scripts/bm25_rag.py query --db {db_dir} --team DSN")


def load_bm25_index(db_dir: Path, team: str):
    """Loads BM25 index and docstore for a team."""
    # Security hardening: use managed BM25 directory only.
    _ = db_dir
    db_dir = _resolve_under_data(DEFAULT_BM25_DIR, "db_dir")
    if team not in TEAMS:
        raise ValueError(f"Invalid team '{team}'. Expected one of: {', '.join(TEAMS)}")
    team_dir = _resolve_under_data(db_dir / team.lower(), f"bm25_index_{team.lower()}")
    pkl_path  = _resolve_under_data(team_dir / "bm25.pkl",  f"bm25_index_pkl_{team.lower()}")
    json_path = _resolve_under_data(team_dir / "bm25.json", f"bm25_index_json_{team.lower()}")

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        bm25 = BM25Okapi(data["tokenized"])
        return bm25, data["docstore"]

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bm25 = BM25Okapi(data["tokenized"])
        return bm25, data["docstore"]

    raise FileNotFoundError(
        f"No BM25 index for team '{team}'.\n"
        "Run: python scripts/bm25_rag.py build"
    )


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def bm25_retrieve(
    query: str,
    bm25: BM25Okapi,
    docstore: list,
    top_k: int = TOP_K,
) -> list:
    """
    BM25 retrieval — tokenizes query, scores all tickets, returns top-k.

    Returns results in the same dict format as vector_retrieve and
    pageindex_retrieve so they can all be merged by RRF without changes.
    """
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # Get top-k indices by score (descending)
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        score = float(scores[idx])
        if score <= 0:
            # BM25 score of 0 means zero term overlap — not useful
            continue
        doc = docstore[idx]
        results.append({
            "rank":        rank + 1,
            "score":       round(score, 4),
            "bm25_score":  round(score, 4),
            "reference":   doc.get("reference", ""),
            "title":       doc.get("title", ""),
            "version":     doc.get("version", ""),
            "description": doc.get("description", ""),
            "resolution":  doc.get("resolution", ""),
            "patches":     doc.get("patches", []),
            "last_date":   doc.get("last_date", ""),
            "source":      "bm25",
            "cluster":     "",
        })

    return results


# ─────────────────────────────────────────────
# INTERACTIVE QUERY
# ─────────────────────────────────────────────

def interactive_query(db_dir: Path, team: str, question: Optional[str] = None):
    print(f"\n📦 Loading BM25 index ({team})...")
    bm25, docstore = load_bm25_index(db_dir, team)
    print(f"   {len(docstore)} tickets indexed")

    if question:
        _show_results(question, team, bm25, docstore)
        return

    print(f"\n{'─'*70}")
    print(f"  Sopra HR — BM25 RAG — Team {team}")
    print(f"  Entrez des mots-clés ou codes d'erreur (ou 'quit' pour quitter)")
    print(f"  Exemples: 'ORA-00942 REGDSN', 'kit DSN phase 3', 'HRCT compilation'")
    print(f"{'─'*70}\n")

    while True:
        try:
            query = input("🔍  Recherche: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break
        if query.lower() in ("quit", "exit", "q", ""):
            print("Au revoir.")
            break
        _show_results(query, team, bm25, docstore)


def _show_results(query: str, team: str, bm25: BM25Okapi, docstore: list):
    tokens = tokenize(query)
    hits   = bm25_retrieve(query, bm25, docstore)

    width = 70
    print(f"\n{'═'*width}")
    print(f"  BM25 | Team {team} | Query: {query[:55]}")
    print(f"  Tokens: {tokens}")
    print(f"{'═'*width}\n")

    if not hits:
        print("  ⚠️  Aucun résultat — aucun token commun avec l'index.")
        print(f"{'═'*width}\n")
        return

    for h in hits[:5]:
        print(f"  #{h['rank']}  {h['reference']}  [BM25: {h['bm25_score']:.3f}]")
        print(f"      {h['title'][:65]}")
        if h.get("version"):
            print(f"      Version: {h['version']}")
        if h.get("resolution"):
            res = h["resolution"].replace("\n", " ")[:130]
            print(f"      ↳ {res}{'...' if len(h['resolution']) > 130 else ''}")
        if h.get("patches"):
            print(f"      🔧 {', '.join(h['patches'][:5])}")
        print()
    print(f"{'═'*width}\n")


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(db_dir: Path, json_dir: Path, team: str, n_samples: int = 50):
    """
    Self-retrieval evaluation for BM25.
    Uses each ticket's title + description as the query.
    """
    random.seed(42)
    print(f"\n🎯 Evaluating BM25 — Team: {team} | Samples: {n_samples}\n")

    db_dir = _resolve_under_data(db_dir, "db_dir")
    json_dir = _resolve_under_data(json_dir, "json_dir")

    all_json = []
    for f in sorted(json_dir.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("support_team") == team and t.get("description") and t.get("is_closed"):
            all_json.append(t)

    if not all_json:
        print(f"❌ No closed tickets found for team {team}")
        return

    sample = random.sample(all_json, min(n_samples, len(all_json)))

    print(f"📦 Loading BM25 index ({team})...")
    bm25, docstore = load_bm25_index(db_dir, team)
    print(f"   {len(docstore)} tickets indexed\n")

    recall    = {1: 0, 3: 0, 5: 0}
    zero_hits = 0

    for i, ticket in enumerate(sample, 1):
        # Use title + description as query (what a consultant would describe)
        query    = f"{ticket.get('title', '')} {ticket.get('description', '')}"
        expected = ticket["reference"]

        hits = bm25_retrieve(query, bm25, docstore, top_k=5)
        refs = [h["reference"] for h in hits]

        if not refs:
            zero_hits += 1

        for k in [1, 3, 5]:
            if expected in refs[:k]:
                recall[k] += 1

        hit    = "✅" if expected in refs[:3] else "❌"
        top    = refs[0] if refs else "no result"
        tokens = tokenize(query)
        print(f"  [{i:3d}/{n_samples}] {hit} | {expected} | top: {top} | {len(tokens)} tokens")

    n = len(sample)
    print(f"\n{'═'*50}")
    print(f"  RÉSULTATS — BM25 — Team {team}")
    print(f"{'═'*50}")
    print(f"  Recall@1 : {recall[1]/n:.1%}  ({recall[1]}/{n})")
    print(f"  Recall@3 : {recall[3]/n:.1%}  ({recall[3]}/{n})")
    print(f"  Recall@5 : {recall[5]/n:.1%}  ({recall[5]}/{n})")
    print(f"  Zero-hit : {zero_hits}  (queries with no term overlap)")
    print(f"{'═'*50}\n")
    print("  Note: BM25 shines on queries with exact error codes/product names.")
    print("  For vague queries ('problème de paie'), vector RAG is stronger.")

    out = _resolve_under_data(db_dir / f"eval_bm25_{team.lower()}.json", "eval_output")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "team": team, "mode": "bm25", "n_samples": n,
            "recall": {str(k): round(v/n, 3) for k, v in recall.items()},
            "zero_hits": zero_hits,
        }, f, indent=2)
    print(f"✅ Results saved → {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sopra HR BM25 keyword RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build indexes (once)
  python scripts/bm25_rag.py build --json data/json --db data/bm25

  # Interactive query
  python scripts/bm25_rag.py query --db data/bm25 --team DSN
  python scripts/bm25_rag.py query --db data/bm25 --team Appli \\
      --question "ORA-00942 REGDSN table inexistante"

  # Evaluate
  python scripts/bm25_rag.py evaluate --db data/bm25 --json data/json --team DSN
  python scripts/bm25_rag.py evaluate --db data/bm25 --json data/json --team Appli
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bld = sub.add_parser("build", help="Build BM25 indexes using managed project data paths")
    bld.add_argument("--json", type=str, default=None, help="(ignored — uses managed project path)")
    bld.add_argument("--db",   type=str, default=None, help="(ignored — uses managed project path)")

    qry = sub.add_parser("query", help="Query BM25 index")
    qry.add_argument("--team",     type=str,  required=True, choices=TEAMS)
    qry.add_argument("--question", type=str,  default=None)

    evl = sub.add_parser("evaluate", help="Evaluate BM25 retrieval quality")
    evl.add_argument("--team",    type=str,  required=True, choices=TEAMS)
    evl.add_argument("--samples", type=int,  default=50)

    args = parser.parse_args()

    if args.command == "build":
        faiss_dir = DEFAULT_FAISS_DIR if DEFAULT_FAISS_DIR.exists() else None
        build_bm25_index(DEFAULT_JSON_DIR, DEFAULT_BM25_DIR, faiss_dir=faiss_dir)
    elif args.command == "query":
        interactive_query(DEFAULT_BM25_DIR, args.team, args.question)
    elif args.command == "evaluate":
        evaluate(DEFAULT_BM25_DIR, DEFAULT_JSON_DIR, args.team, args.samples)
