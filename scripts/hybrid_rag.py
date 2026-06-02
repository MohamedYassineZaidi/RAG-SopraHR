#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hybrid_rag.py
=============
Hybrid RAG combining:
  - Vector RAG  : FAISS cosine similarity (from rag_assistant.py indexes)
  - Vectorless  : PageIndex Claude navigation (from vectorless_rag.py)
  - Merger      : Reciprocal Rank Fusion (RRF)

RRF formula: score(ticket) = Σ 1 / (RRF_K + rank_i)
A ticket appearing in both result lists is boosted above tickets
found by only one system. This corrects the DSN weakness where
vector search confuses identical kit-delivery descriptions.

Requires:
  pip install faiss-cpu sentence-transformers boto3 python-dotenv numpy
  — rag_assistant.py indexes must already be built
  — vectorless_rag.py PageIndex must already be built

Usage:
  # Query (interactive)
  python hybrid_rag.py query --index data/pageindex --db data/indexes --team DSN

  # Single question
  python hybrid_rag.py query --index data/pageindex --db data/indexes --team DSN \\
      --question "Erreur ORA-00942 lors du lancement REGDSN"

  # Side-by-side evaluation: vector vs pageindex vs hybrid
  python hybrid_rag.py evaluate --index data/pageindex --db data/indexes --json data/json --team DSN --samples 50
"""

import os
import json
import pickle
import time
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import faiss
from sentence_transformers import SentenceTransformer

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        import ssl
        import urllib.request
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    except Exception:
        pass

from rag_utils import (
    TEAMS, TOP_K, RRF_K, EMBEDDING_MODEL,
    get_bedrock_client,
    generate_suggestion, print_results,
)
from vectorless_rag import (
    load_pageindex,
    retrieve as pageindex_retrieve,
)
from bm25_rag import (
    load_bm25_index,
    bm25_retrieve,
)


# ─────────────────────────────────────────────
# 1. VECTOR RAG — load index + retrieve
# ─────────────────────────────────────────────

def load_vector_index(db_dir: Path, team: str):
    p = db_dir / team.lower()
    if not (p / "index.faiss").exists():
        raise FileNotFoundError(
            f"No vector index for '{team}' at {p}.\n"
            f"Run: python vector_rag.py build --json <json_dir> --db {db_dir}"
        )
    index = faiss.read_index(str(p / "index.faiss"))
    with open(p / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    return index, docstore


def vector_retrieve(
    query: str,
    index,
    docstore: list,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list:
    """FAISS cosine similarity search. Vectors are L2-normalised so
    inner product == cosine similarity."""
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    similarities, indices = index.search(query_vec, top_k)

    results = []
    for rank, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
        if idx < 0:
            continue
        doc = docstore[idx]
        ref = doc.get("reference", "")
        # Filter out non-French-language tickets (German, English, Italian, Dutch)
        # but keep FR (France), AF (French-speaking Africa), SP (Maghreb/francophone)
        _NON_FRENCH = ("DE ", "UK ", "IT ", "NL ")
        if ref and ref.startswith(_NON_FRENCH):
            continue
        results.append({
            "rank":        len(results) + 1,
            "score":       round(float(sim), 4),
            "similarity":  round(float(sim), 4),
            "reference":   ref,
            "title":       doc.get("title", ""),
            "version":     doc.get("version", ""),
            "description": doc.get("description", ""),
            "resolution":  doc.get("resolution", ""),
            "patches":     doc.get("patches", []),
            "last_date":   doc.get("last_date", ""),
            "espdsn_version": doc.get("espdsn_version", ""),
            "source":      "vector",
            "cluster":     "",
        })
    return results


# ─────────────────────────────────────────────
# 2. RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: list,
    pageindex_results: list,
    bm25_results: list,
    k: int = RRF_K,
    top_k: int | None = None,
) -> list:
    """
    3-way RRF: Vector + PageIndex + BM25, with recency boost.

    Score for each ticket = Σ weight_i / (k + rank_i) across all systems.
    BM25 gets a 1.5× weight boost because it excels at matching exact
    technical terms (error codes, patch numbers, rubrique names) that
    vector search misses due to semantic dilution.

    A ticket appearing in 2 or 3 systems gets a strong boost.

    Recency boost: tickets with more recent conversation dates get a
    multiplicative bonus (up to 1.3x for tickets from the current year,
    down to 1.0x for tickets older than 5 years).
    """
    scores: dict = {}

    for result in vector_results:
        ref = result["reference"]
        scores[ref] = scores.get(ref, 0.0) + 1.0 / (k + result["rank"])

    for result in pageindex_results:
        ref = result["reference"]
        scores[ref] = scores.get(ref, 0.0) + 1.0 / (k + result["rank"])

    # BM25 gets 1.5× weight — exact keyword matches are more reliable
    # for technical support queries (error codes, patch numbers, etc.)
    for result in bm25_results:
        ref = result["reference"]
        scores[ref] = scores.get(ref, 0.0) + 1.5 / (k + result["rank"])

    # Merge metadata — priority: vector > bm25 > pageindex
    all_meta = {r["reference"]: r for r in pageindex_results}
    all_meta.update({r["reference"]: r for r in bm25_results})
    all_meta.update({r["reference"]: r for r in vector_results})

    # Apply recency boost based on last_date
    # Recent tickets are far more likely to be relevant (same software version,
    # same config). Boost decays exponentially so tickets from the last 2 years
    # are strongly preferred over 10-year-old ones.
    now = datetime.now()
    for ref in scores:
        last_date = all_meta[ref].get("last_date", "")
        if last_date:
            try:
                ticket_dt = datetime.fromisoformat(last_date)
                age_days = max(0, (now - ticket_dt).days)
                # Exponential decay: 1.8× for today → ~1.4× at 1 year → ~1.0× at 3+ years
                boost = 1.0 + 0.8 * math.exp(-age_days / 730)  # half-life ~2 years
                scores[ref] *= boost
            except (ValueError, TypeError):
                pass

    # Build source sets for labelling
    vector_refs    = {r["reference"] for r in vector_results}
    pageindex_refs = {r["reference"] for r in pageindex_results}
    bm25_refs      = {r["reference"] for r in bm25_results}

    merged = []
    for ref, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
        result = dict(all_meta[ref])
        result["rrf_score"] = round(rrf_score, 6)

        sources = []
        if ref in vector_refs:    sources.append("vec")
        if ref in bm25_refs:      sources.append("bm25")
        if ref in pageindex_refs: sources.append("pi")

        result["source"] = "+".join(sources) + (" ✓" if len(sources) > 1 else "")
        merged.append(result)

    for i, r in enumerate(merged):
        r["rank"] = i + 1

    return merged[: (top_k if top_k is not None else TOP_K)]


# ─────────────────────────────────────────────
# 3. INTERACTIVE QUERY
# ─────────────────────────────────────────────

def interactive_query(
    index_dir: Path,
    db_dir: Path,
    bm25_dir: Path,
    team: str,
    question: Optional[str] = None,
):
    print(f"🧠 Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"📦 Loading vector index ({team})...")
    vector_index, vector_docstore = load_vector_index(db_dir, team)
    print(f"   {vector_index.ntotal} tickets in vector index")

    print(f"📄 Loading PageIndex ({team})...")
    pageindex   = load_pageindex(index_dir)
    all_tickets = {doc["reference"]: doc for doc in vector_docstore}
    print(f"   {pageindex.get(team, {}).get('count', 0)} tickets in PageIndex")

    print(f"🔍 Loading BM25 index ({team})...")
    bm25_model, bm25_docstore = load_bm25_index(bm25_dir, team)
    print(f"   {len(bm25_docstore)} tickets in BM25 index")

    print(f"☁️  Connecting to Bedrock...")
    bedrock = get_bedrock_client()

    if question:
        _run_query(question, team, pageindex, all_tickets, bedrock,
                   embed_model, vector_index, vector_docstore,
                   bm25_model, bm25_docstore)
        return

    print(f"\n{'─'*70}")
    print(f"  Sopra HR — Hybrid RAG (Vector + BM25 + PageIndex) — Team {team}")
    print(f"  Décrivez le problème (ou 'quit' pour quitter)")
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
        _run_query(query, team, pageindex, all_tickets, bedrock,
                   embed_model, vector_index, vector_docstore,
                   bm25_model, bm25_docstore)


def _run_query(
    query, team, pageindex, all_tickets, bedrock,
    embed_model, vector_index, vector_docstore,
    bm25_model, bm25_docstore,
):
    v_hits   = vector_retrieve(query, vector_index, vector_docstore, embed_model)
    pi_hits  = pageindex_retrieve(query, team, pageindex, bedrock, all_tickets)
    bm25_hits = bm25_retrieve(query, bm25_model, bm25_docstore)
    hits     = reciprocal_rank_fusion(v_hits, pi_hits, bm25_hits)

    suggestion = generate_suggestion(query, hits, bedrock)
    print_results(query, team, hits, suggestion, "hybrid")


# ─────────────────────────────────────────────
# 4. EVALUATE — side-by-side: vector / pageindex / hybrid
# ─────────────────────────────────────────────

def evaluate(
    index_dir: Path,
    db_dir: Path,
    bm25_dir: Path,
    json_dir: Path,
    team: str,
    n_samples: int = 50,
):
    """
    Compares all four retrieval modes side by side on the same sample:
    vector | bm25 | pageindex | hybrid (3-way)
    """
    import random
    random.seed(42)

    print(f"\n🎯 Side-by-side evaluation — Team: {team} | Samples: {n_samples}")
    print(f"   vector | bm25 | pageindex | hybrid\n")

    all_json = []
    for f in sorted(json_dir.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("support_team") == team and t.get("description") and t.get("is_closed"):
            all_json.append(t)

    sample = random.sample(all_json, min(n_samples, len(all_json)))

    print(f"🧠 Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    vector_index, vector_docstore = load_vector_index(db_dir, team)

    pageindex   = load_pageindex(index_dir)
    all_tickets = {doc["reference"]: doc for doc in vector_docstore}
    for t in all_json:
        ref = t.get("reference", "")
        if ref in all_tickets:
            all_tickets[ref].update(t)

    print(f"🔍 Loading BM25 index ({team})...")
    bm25_model, bm25_docstore = load_bm25_index(bm25_dir, team)

    bedrock = get_bedrock_client()

    recall = {
        "vector":    {1: 0, 3: 0, 5: 0},
        "bm25":      {1: 0, 3: 0, 5: 0},
        "pageindex": {1: 0, 3: 0, 5: 0},
        "hybrid":    {1: 0, 3: 0, 5: 0},
    }

    for i, ticket in enumerate(sample, 1):
        query    = ticket["description"]
        expected = ticket["reference"]

        v_hits    = vector_retrieve(query, vector_index, vector_docstore, embed_model)
        bm25_hits = bm25_retrieve(query, bm25_model, bm25_docstore)
        pi_hits   = pageindex_retrieve(query, team, pageindex, bedrock, all_tickets)
        h_hits    = reciprocal_rank_fusion(v_hits, pi_hits, bm25_hits)

        for mode, hits in [
            ("vector", v_hits), ("bm25", bm25_hits),
            ("pageindex", pi_hits), ("hybrid", h_hits)
        ]:
            refs = [h["reference"] for h in hits]
            for k in [1, 3, 5]:
                if expected in refs[:k]:
                    recall[mode][k] += 1

        v_hit  = "✅" if expected in [h["reference"] for h in v_hits[:3]]  else "❌"
        b_hit  = "✅" if expected in [h["reference"] for h in bm25_hits[:3]] else "❌"
        pi_hit = "✅" if expected in [h["reference"] for h in pi_hits[:3]] else "❌"
        h_hit  = "✅" if expected in [h["reference"] for h in h_hits[:3]]  else "❌"

        print(f"  [{i:3d}/{n_samples}] V:{v_hit} B:{b_hit} PI:{pi_hit} H:{h_hit} | {expected}")
        time.sleep(0.3)

    n = len(sample)
    print(f"\n{'═'*60}")
    print(f"  RÉSULTATS — Team {team} ({n} samples)")
    print(f"{'═'*60}")
    print(f"  {'Mode':12s} | R@1    | R@3    | R@5")
    print(f"  {'─'*12}-+--------+--------+--------")
    for mode in ["vector", "bm25", "pageindex", "hybrid"]:
        r = recall[mode]
        marker = " ←" if mode == "hybrid" else ""
        print(f"  {mode:12s} | {r[1]/n:5.1%}  | {r[3]/n:5.1%}  | {r[5]/n:5.1%}{marker}")
    print(f"{'═'*60}\n")

    out = index_dir / f"eval_hybrid_{team.lower()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "team": team, "n_samples": n,
            "recall": {
                mode: {str(k): round(v/n, 3) for k, v in r.items()}
                for mode, r in recall.items()
            }
        }, f, indent=2)
    print(f"✅ Results saved → {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sopra HR Hybrid RAG (Vector + PageIndex + RRF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_rag.py query --index data/pageindex --db data/indexes --team DSN
  python hybrid_rag.py query --index data/pageindex --db data/indexes --team DSN \\
      --question "Erreur ORA-00942 lors du lancement REGDSN"
  python hybrid_rag.py evaluate --index data/pageindex --db data/indexes \\
      --json data/json --team DSN --samples 50
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    qry = sub.add_parser("query", help="Interactive hybrid query")
    qry.add_argument("--index",    type=Path, required=True, help="PageIndex folder")
    qry.add_argument("--db",       type=Path, required=True, help="Vector indexes folder")
    qry.add_argument("--bm25",     type=Path, required=True, help="BM25 indexes folder")
    qry.add_argument("--team",     type=str,  required=True, choices=TEAMS)
    qry.add_argument("--question", type=str,  default=None)

    evl = sub.add_parser("evaluate", help="Side-by-side evaluation: vector vs bm25 vs pageindex vs hybrid")
    evl.add_argument("--index",   type=Path, required=True)
    evl.add_argument("--db",      type=Path, required=True)
    evl.add_argument("--bm25",    type=Path, required=True)
    evl.add_argument("--json",    type=Path, required=True)
    evl.add_argument("--team",    type=str,  required=True, choices=TEAMS)
    evl.add_argument("--samples", type=int,  default=50)

    args = parser.parse_args()

    if args.command == "query":
        interactive_query(args.index, args.db, args.bm25, args.team, args.question)
    elif args.command == "evaluate":
        evaluate(args.index, args.db, args.bm25, args.json, args.team, args.samples)