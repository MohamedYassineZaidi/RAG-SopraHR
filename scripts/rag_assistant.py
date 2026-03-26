#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_assistant.py
================
Sopra HR support assistant — RAG over resolved tickets.

Three commands:

  1. Index — embed all JSON tickets into 3 FAISS indexes (one per team)
     python rag_assistant.py index --json data/json --db data/indexes

  2. Query — find similar past tickets and suggest a resolution
     python rag_assistant.py query --db data/indexes --team DSN
     python rag_assistant.py query --db data/indexes --team Appli --question "Erreur ORA-00942 REGDSN"

  3. Evaluate — measure Recall@1/3/5 against a sample of known tickets
     python rag_assistant.py evaluate --db data/indexes --json data/json --team DSN --samples 50

Requirements:
    pip install faiss-cpu sentence-transformers boto3 python-dotenv numpy
"""

import os
import re
import json
import time
import pickle
import argparse
import textwrap
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import faiss
import boto3
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM   = 768

BEDROCK_MODEL  = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

TOP_K  = 5   # retrieve top 5, show top 3
TEAMS  = ["DSN", "Appli", "Outils"]

# Status codes worth showing to consultants
CLOSED_CODES = {"CP", "CN", "CO", "CS", "CQ", "CH", "CU", "CX", "CY"}


# ─────────────────────────────────────────────
# 1. BEDROCK CLIENT
# ─────────────────────────────────────────────

def get_bedrock_client():
    kwargs = {"service_name": "bedrock-runtime", "region_name": BEDROCK_REGION}
    key    = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    token  = os.getenv("AWS_SESSION_TOKEN")
    if key and secret:
        kwargs["aws_access_key_id"]     = key
        kwargs["aws_secret_access_key"] = secret
        if token:
            kwargs["aws_session_token"] = token
    return boto3.client(**kwargs)


def call_bedrock(client, prompt: str, max_tokens: int = 1024) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    })
    response = client.invoke_model(
        modelId=BEDROCK_MODEL, body=body,
        contentType="application/json", accept="application/json"
    )
    return json.loads(response["body"].read())["content"][0]["text"].strip()


# ─────────────────────────────────────────────
# 2. TICKET → EMBEDDABLE TEXT
# ─────────────────────────────────────────────

def ticket_to_embed_text(ticket: dict) -> str:
    """
    Builds the text we embed for each ticket.
    Combines title + description + resolution so the vector
    captures both the problem AND the solution.
    """
    parts = []

    if ticket.get("title"):
        parts.append(f"Titre: {ticket['title']}")
    if ticket.get("version"):
        parts.append(f"Version: {ticket['version']}")
    if ticket.get("description"):
        parts.append(f"Problème: {ticket['description'][:500]}")
    if ticket.get("resolution"):
        parts.append(f"Résolution: {ticket['resolution'][:400]}")
    if ticket.get("patches"):
        parts.append(f"Patches: {', '.join(ticket['patches'])}")

    return "\n".join(parts)


def ticket_to_display(ticket: dict) -> dict:
    """Extracts the fields we want to show consultants."""
    return {
        "reference":   ticket.get("reference", ""),
        "title":       ticket.get("title", ""),
        "version":     ticket.get("version", ""),
        "system":      ticket.get("system", ""),
        "description": ticket.get("description", ""),
        "resolution":  ticket.get("resolution", ""),
        "patches":     ticket.get("patches", []),
        "status":      ticket.get("closing_status_code", ""),
        "status_desc": ticket.get("closing_status_explanation", ""),
        "team":        ticket.get("support_team", ""),
    }


# ─────────────────────────────────────────────
# 3. FAISS INDEX — SAVE / LOAD
# ─────────────────────────────────────────────

def index_path(db_dir: Path, team: str) -> Path:
    return db_dir / team.lower()


def save_team_index(db_dir: Path, team: str, index, docstore: list):
    p = index_path(db_dir, team)
    p.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p / "index.faiss"))
    with open(p / "docstore.pkl", "wb") as f:
        pickle.dump(docstore, f)
    with open(p / "config.json", "w") as f:
        json.dump({"team": team, "model": EMBEDDING_MODEL, "count": len(docstore)}, f)


def load_team_index(db_dir: Path, team: str):
    p = index_path(db_dir, team)
    if not (p / "index.faiss").exists():
        raise FileNotFoundError(
            f"No index for team '{team}' at {p}.\n"
            f"Run: python rag_assistant.py index --json <json_dir> --db {db_dir}"
        )
    index = faiss.read_index(str(p / "index.faiss"))
    with open(p / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open(p / "config.json") as f:
        config = json.load(f)
    return index, docstore, config


# ─────────────────────────────────────────────
# 4. INDEXER
# ─────────────────────────────────────────────

def build_indexes(json_dir: Path, db_dir: Path, batch_size: int = 256):
    """
    Reads all JSON tickets, groups by team, embeds each group,
    builds one FAISS index per team.
    """
    json_files = sorted(json_dir.glob("*.json"))
    print(f"\n📂 Found {len(json_files)} JSON tickets")

    # Group tickets by team
    by_team: dict = {t: [] for t in TEAMS}
    by_team["Unknown"] = []

    failed = 0
    for f in json_files:
        try:
            ticket = json.loads(f.read_text(encoding="utf-8"))
            team   = ticket.get("support_team", "Unknown")
            group  = team if team in TEAMS else "Unknown"
            by_team[group].append(ticket)
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
            failed += 1

    print(f"\n  Team breakdown:")
    for team, tickets in by_team.items():
        if tickets:
            print(f"    {team:10s} {len(tickets):5d} tickets")

    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~400MB)\n")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build one index per team (skip Unknown)
    for team in TEAMS:
        tickets = by_team[team]
        if not tickets:
            print(f"⚠️  No tickets for team {team} — skipping")
            continue

        print(f"\n{'─'*50}")
        print(f"📌 Indexing team: {team} ({len(tickets)} tickets)")

        # Build docstore and embed texts
        docstore = [ticket_to_display(t) for t in tickets]
        texts    = [ticket_to_embed_text(t) for t in tickets]

        # Embed in batches
        all_embeddings = []
        total_batches  = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch      = texts[i:i + batch_size]
            embeddings = model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False
            )
            all_embeddings.append(embeddings)
            batch_num = i // batch_size + 1
            print(f"  Batch {batch_num:3d}/{total_batches} — {min(i+batch_size, len(texts))}/{len(texts)}")

        matrix = np.vstack(all_embeddings).astype("float32")

        # IndexFlatIP = exact cosine similarity (vectors are L2-normalized)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(matrix)

        save_team_index(db_dir, team, index, docstore)
        print(f"  ✅ Saved {team} index → {index_path(db_dir, team)}")

    print(f"\n🎉 All indexes built. Run queries with:")
    print(f"   python rag_assistant.py query --db {db_dir} --team DSN")


# ─────────────────────────────────────────────
# 5. RETRIEVER
# ─────────────────────────────────────────────

def retrieve(
    query: str,
    index,
    docstore: list,
    model: SentenceTransformer,
    top_k: int = TOP_K
) -> list:
    """Embeds query and returns top-k most similar tickets."""
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    similarities, indices = index.search(query_vec, top_k)

    hits = []
    for rank, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
        if idx < 0:
            continue
        doc = docstore[idx]
        hits.append({
            "rank":       rank + 1,
            "similarity": round(float(sim), 4),
            **doc,
        })
    return hits


# ─────────────────────────────────────────────
# 6. GENERATOR
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant technique pour l'équipe support de Sopra HR.
Un consultant vient de recevoir un nouveau ticket d'incident.
En te basant sur les tickets similaires résolus ci-dessous, propose une action concrète.

Règles:
- Réponds dans la même langue que la question du consultant (français/anglais)
- Sois précis: cite les numéros de patch, les noms de champs, les tables si pertinent
- Si les tickets similaires ne correspondent pas exactement, dis-le clairement
- Structure ta réponse: (1) Diagnostic probable, (2) Action recommandée, (3) Patches si applicable
- Sois concis — le consultant est occupé"""


def generate_suggestion(query: str, hits: list, bedrock_client) -> str:
    """Generates a consultant-facing suggestion using top retrieved tickets."""
    context_parts = []
    for h in hits[:3]:
        part = f"--- Ticket {h['reference']} (similarité: {h['similarity']}) ---\n"
        part += f"Titre: {h['title']}\n"
        if h.get('description'):
            part += f"Problème: {h['description'][:300]}\n"
        if h.get('resolution'):
            part += f"Résolution: {h['resolution'][:400]}\n"
        if h.get('patches'):
            part += f"Patches: {', '.join(h['patches'])}\n"
        context_parts.append(part)

    context = "\n\n".join(context_parts)
    prompt  = f"{SYSTEM_PROMPT}\n\nNouveau ticket:\n{query}\n\nTickets similaires résolus:\n{context}\n\nSuggestion:"

    try:
        return call_bedrock(bedrock_client, prompt)
    except Exception as e:
        return f"[Erreur génération: {e}]"


# ─────────────────────────────────────────────
# 7. DISPLAY
# ─────────────────────────────────────────────

def print_results(query: str, team: str, hits: list, suggestion: str):
    width = 70

    print(f"\n{'═'*width}")
    print(f"  ÉQUIPE: {team}  |  QUESTION: {query[:50]}{'...' if len(query)>50 else ''}")
    print(f"{'═'*width}")

    print(f"\n📋  TICKETS SIMILAIRES TROUVÉS:\n")
    for h in hits[:3]:
        sim_bar = "█" * int(h['similarity'] * 10) + "░" * (10 - int(h['similarity'] * 10))
        print(f"  #{h['rank']}  {h['reference']}  [{sim_bar}] {h['similarity']:.2f}")
        print(f"      {h['title'][:65]}")
        if h.get('version'):
            print(f"      Version: {h['version']}")
        if h.get('resolution'):
            # Show first 150 chars of resolution
            res = h['resolution'].replace('\n', ' ')[:150]
            print(f"      ↳ {res}{'...' if len(h['resolution'])>150 else ''}")
        if h.get('patches'):
            print(f"      🔧 Patches: {', '.join(h['patches'])}")
        print()

    print(f"{'─'*width}")
    print(f"💬  SUGGESTION:\n")
    # Word-wrap suggestion at 68 chars
    for line in suggestion.splitlines():
        if line.strip():
            for wrapped in textwrap.wrap(line, width=66):
                print(f"  {wrapped}")
        else:
            print()
    print(f"{'═'*width}\n")


# ─────────────────────────────────────────────
# 8. INTERACTIVE QUERY MODE
# ─────────────────────────────────────────────

def interactive_query(db_dir: Path, team: str, question: Optional[str] = None):
    """
    Interactive query loop for a specific team.
    Can also run a single question if --question is provided.
    """
    print(f"\n🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"📦 Loading {team} index...")
    index, docstore, config = load_team_index(db_dir, team)
    print(f"✅ {config['count']} tickets loaded")

    print(f"☁️  Connecting to Bedrock...")
    bedrock = get_bedrock_client()

    if question:
        # Single question mode
        hits       = retrieve(question, index, docstore, model)
        suggestion = generate_suggestion(question, hits, bedrock)
        print_results(question, team, hits, suggestion)
        return

    # Interactive loop
    print(f"\n{'─'*70}")
    print(f"  Sopra HR Support Assistant — Team {team}")
    print(f"  Décrivez le problème du nouveau ticket (ou 'quit' pour quitter)")
    print(f"{'─'*70}\n")

    while True:
        try:
            query = input("❓  Problème: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if query.lower() in ("quit", "exit", "q", ""):
            print("Au revoir.")
            break

        hits       = retrieve(query, index, docstore, model)
        suggestion = generate_suggestion(query, hits, bedrock)
        print_results(query, team, hits, suggestion)


# ─────────────────────────────────────────────
# 9. EVALUATOR
# ─────────────────────────────────────────────

def evaluate(db_dir: Path, json_dir: Path, team: str, n_samples: int = 50):
    """
    Evaluates retrieval quality for a team by:
    1. Taking N random closed tickets from the team
    2. Using their description as the query
    3. Checking if the correct ticket appears in top-1/3/5 results

    This is a self-retrieval test — a perfect system scores 100% on Recall@1.
    Real queries from new consultants will score lower, but this gives a baseline.
    """
    import random

    print(f"\n🎯 Evaluating team: {team} | samples: {n_samples}")

    # Load tickets for this team
    all_tickets = []
    for f in sorted(json_dir.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("support_team") == team and t.get("description") and t.get("is_closed"):
            all_tickets.append(t)

    if len(all_tickets) < n_samples:
        print(f"  ⚠️  Only {len(all_tickets)} closed tickets with descriptions for {team}")
        n_samples = len(all_tickets)

    sample = random.sample(all_tickets, n_samples)

    print(f"🧠 Loading model and index...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index, docstore, config = load_team_index(db_dir, team)

    recall = {1: 0, 3: 0, 5: 0}
    results = []

    for i, ticket in enumerate(sample, 1):
        query    = ticket["description"]
        expected = ticket["reference"]

        hits  = retrieve(query, index, docstore, model, top_k=5)
        refs  = [h["reference"] for h in hits]

        for k in [1, 3, 5]:
            if expected in refs[:k]:
                recall[k] += 1

        hit = "✅" if expected in refs[:3] else "❌"
        results.append({
            "expected":  expected,
            "retrieved": refs,
            "hit_at_1":  expected in refs[:1],
            "hit_at_3":  expected in refs[:3],
            "hit_at_5":  expected in refs[:5],
        })
        print(f"  [{i:3d}/{n_samples}] {hit} | {expected} | top: {refs[0]}")

    n = len(sample)
    print(f"\n{'═'*50}")
    print(f"  RÉSULTATS — Team {team}")
    print(f"{'═'*50}")
    print(f"  Recall@1 : {recall[1]/n:.1%}  ({recall[1]}/{n})")
    print(f"  Recall@3 : {recall[3]/n:.1%}  ({recall[3]}/{n})")
    print(f"  Recall@5 : {recall[5]/n:.1%}  ({recall[5]}/{n})")
    print(f"{'═'*50}")
    print()
    print("  Recall@3 tells you: for 3 in 10 tickets does the right")
    print("  past ticket appear in the top 3 results?")
    print("  A score above 60% is good for this dataset size.")


# ─────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sopra HR RAG Support Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Build indexes (run once after txt_to_json.py)
          python rag_assistant.py index --json data/json --db data/indexes

          # Interactive query for DSN team
          python rag_assistant.py query --db data/indexes --team DSN

          # Single question
          python rag_assistant.py query --db data/indexes --team Appli \\
              --question "Erreur de validation page FSWDRE01"

          # Evaluate retrieval quality
          python rag_assistant.py evaluate --db data/indexes --json data/json --team DSN
        """)
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    idx = sub.add_parser("index", help="Build FAISS indexes from JSON tickets")
    idx.add_argument("--json",  type=Path, required=True, help="Folder of JSON tickets")
    idx.add_argument("--db",    type=Path, required=True, help="Output folder for indexes")
    idx.add_argument("--batch", type=int,  default=256,   help="Embedding batch size")

    # query
    qry = sub.add_parser("query", help="Query the assistant")
    qry.add_argument("--db",       type=Path, required=True,
                     help="Folder containing indexes")
    qry.add_argument("--team",     type=str,  required=True,
                     choices=TEAMS, help="Team to search: DSN / Appli / Outils")
    qry.add_argument("--question", type=str,  default=None,
                     help="Single question (omit for interactive mode)")

    # evaluate
    evl = sub.add_parser("evaluate", help="Evaluate retrieval quality")
    evl.add_argument("--db",      type=Path, required=True)
    evl.add_argument("--json",    type=Path, required=True)
    evl.add_argument("--team",    type=str,  required=True, choices=TEAMS)
    evl.add_argument("--samples", type=int,  default=50)

    args = parser.parse_args()

    if args.command == "index":
        build_indexes(args.json, args.db, args.batch)

    elif args.command == "query":
        interactive_query(args.db, args.team, args.question)

    elif args.command == "evaluate":
        evaluate(args.db, args.json, args.team, args.samples)