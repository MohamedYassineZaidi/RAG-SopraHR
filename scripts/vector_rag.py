"""
vector_rag.py
─────────────
Vector RAG for Sopra HR ticket resolution.
Vector store: FAISS (local, no server)
Embeddings:   sentence-transformers (multilingual-e5-large, local)
Generator:    Amazon Bedrock (Claude Haiku)

Commands:
    # Build per-team indexes from JSON tickets (main pipeline entry point)
    python vector_rag.py build --json data/json --db data/indexes

    # Index a flat folder of tickets into a single index
    python vector_rag.py index --tickets ./data/cleaned --db ./faiss_db

    # Query a single-index db
    python vector_rag.py query --db ./faiss_db --question "La rubrique ZDAG-COMMO n'existe pas"

    # Evaluate against an eval set
    python vector_rag.py evaluate --db ./faiss_db --eval ./output/eval_set.json --output ./output/rag_eval_results.json

Requirements:
    pip install faiss-cpu sentence-transformers boto3 python-dotenv numpy
"""

import os
import json
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

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

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM   = 1024

BEDROCK_MODEL  = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

TOP_K = 5

_SCRIPTS = Path(__file__).parent
_ROOT    = _SCRIPTS.parent
_DATA    = _ROOT / "data"


def _resolve_under_data(path: Path, arg_name: str) -> Path:
    """Resolve path and ensure it stays inside the project data/ directory."""
    resolved = Path(path).expanduser().resolve(strict=False)
    base = _DATA.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be under {base}: {resolved}") from exc
    return resolved


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
# 2. FAISS STORAGE
#    db_path folder contains:
#      index.faiss   — the vector index
#      docstore.pkl  — list of dicts (text + metadata)
#      config.json   — model info
# ─────────────────────────────────────────────

def save_index(db_path: str, index, docstore: list):
    db = _resolve_under_data(Path(db_path), "db_path")
    db.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(db / "index.faiss"))
    with open(db / "docstore.pkl", "wb") as f:
        pickle.dump(docstore, f)
    with open(db / "config.json", "w") as f:
        json.dump({"model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "count": len(docstore)}, f)
    print(f"💾 Saved: {len(docstore)} vectors → {db_path}")


def load_index(db_path: str):
    db = _resolve_under_data(Path(db_path), "db_path")
    if not (db / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index at '{db_path}'. Run: python vector_rag.py index --tickets ... --db {db_path}")
    index    = faiss.read_index(str(db / "index.faiss"))
    with open(db / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open(db / "config.json") as f:
        config = json.load(f)
    print(f"✅ Loaded index: {config['count']} tickets | model: {config['model']}")
    return index, docstore, config


# ─────────────────────────────────────────────
# 3. TICKET → DOCUMENT
# ─────────────────────────────────────────────

TEAMS = ["DSN", "Appli", "Outils"]


def _extract_conversation_text(ticket: dict, max_chars: int = 1500) -> str:
    """
    Extracts searchable text from the conversation field.
    Error codes, page names, and field references (e.g. ZDAG-COMMO)
    often appear only in client follow-ups and support replies, not in
    the description or resolution. This makes them visible to retrieval.
    """
    conv = ticket.get("conversation") or []
    if not conv:
        return ""
    pieces = []
    for entry in conv:
        text = entry.get("content") or ""  # field is 'content', not 'text'
        if not text or len(text) < 20:
            continue
        # Skip pure status-change / archive messages
        if entry.get("type") in ("status_change", "archive"):
            continue
        pieces.append(text)
    joined = "\n".join(pieces)
    return joined[:max_chars] if len(joined) > max_chars else joined


def _extract_last_date(ticket: dict) -> str:
    """Return the most recent timestamp from the ticket's conversation."""
    dates = [
        entry.get("timestamp", "")
        for entry in ticket.get("conversation", [])
        if entry.get("timestamp")
    ]
    return max(dates) if dates else ""


def ticket_to_embed_text(ticket: dict) -> str:
    """
    Builds the text passed to the embedding model.
    Combines title + description + resolution + conversation so the vector
    captures the problem, the solution, AND technical terms that only
    appear in mid-conversation (error codes, page names, rubrique names).
    """
    parts = []
    if ticket.get("title"):
        parts.append(f"Titre: {ticket['title']}")
    if ticket.get("version"):
        parts.append(f"Version: {ticket['version']}")
    if ticket.get("system"):
        parts.append(f"Système: {ticket['system']}")
    if ticket.get("description"):
        parts.append(f"Problème: {ticket['description'][:600]}")
    if ticket.get("resolution"):
        parts.append(f"Résolution: {ticket['resolution'][:800]}")
    if ticket.get("patches"):
        parts.append(f"Patches: {', '.join(ticket['patches'])}")
    if ticket.get("espdsn_version"):
        parts.append(f"ESPDSN: {ticket['espdsn_version']}")
    conv_text = _extract_conversation_text(ticket, max_chars=1500)
    if conv_text:
        parts.append(f"Échanges: {conv_text}")
    return "\n".join(parts)


def ticket_to_display(ticket: dict) -> dict:
    """Extracts the fields stored in the docstore (shown to consultants)."""
    return {
        "reference":      ticket.get("reference", ""),
        "title":          ticket.get("title", ""),
        "version":        ticket.get("version", ""),
        "system":         ticket.get("system", ""),
        "description":    ticket.get("description", ""),
        "resolution":     ticket.get("resolution", ""),
        "patches":        ticket.get("patches", []),
        "status":         ticket.get("closing_status_code", ""),
        "status_desc":    ticket.get("closing_status_explanation", ""),
        "team":           ticket.get("support_team", ""),
        "last_date":      _extract_last_date(ticket),
        "espdsn_version": ticket.get("espdsn_version", ""),
        "source_file":    ticket.get("source_file", ""),
    }


# kept for backward-compat with the single-index pipeline
def ticket_to_document(ticket: dict) -> dict:
    text = ticket_to_embed_text(ticket)
    doc  = ticket_to_display(ticket)
    doc["text"] = text
    return doc


# ─────────────────────────────────────────────
# 4. INDEXER — single flat index (legacy / flat mode)
# ─────────────────────────────────────────────

def _embed_texts(texts: list, model: SentenceTransformer, batch_size: int = 256) -> np.ndarray:
    """Embed a list of texts in batches, returns float32 matrix."""
    all_embeddings = []
    total_batches  = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch      = [f"passage: {t}" for t in texts[i:i + batch_size]]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embeddings)
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num:3d}/{total_batches} — {min(i+batch_size, len(texts))}/{len(texts)} done")
    return np.vstack(all_embeddings).astype("float32")


def index_tickets(tickets_dir: str, db_path: str, batch_size: int = 256):
    """Build a single flat FAISS index from all tickets in a directory."""
    files = sorted(Path(tickets_dir).glob("*.json"))
    print(f"\n📂 Found {len(files)} JSON tickets")
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~1GB — please wait)\n")

    model = SentenceTransformer(EMBEDDING_MODEL)

    docs, failed = [], []
    for f in files:
        try:
            ticket = json.loads(f.read_text(encoding="utf-8"))
            doc    = ticket_to_document(ticket)
            if doc["text"].strip():
                docs.append(doc)
        except Exception as e:
            failed.append(f.name)

    print(f"✅ Loaded: {len(docs)} | ❌ Failed: {len(failed)}")
    if not docs:
        print("❌ No documents to index. Check the --tickets path.")
        return

    print(f"\n⚡ Embedding {len(docs)} tickets...")
    matrix = _embed_texts([d["text"] for d in docs], model, batch_size)
    print(f"\n📐 Matrix shape: {matrix.shape}")

    print("🔨 Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)
    print(f"✅ {index.ntotal} vectors indexed")
    save_index(db_path, index, docs)
    print(f"\n🎉 Done!")


# ─────────────────────────────────────────────
# 4b. PER-TEAM INDEXES (main pipeline entry point)
# ─────────────────────────────────────────────

def _team_index_path(db_dir: Path, team: str) -> Path:
    return db_dir / team.lower()


def save_team_index(db_dir: Path, team: str, index, docstore: list):
    p = _team_index_path(db_dir, team)
    p.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p / "index.faiss"))
    with open(p / "docstore.pkl", "wb") as f:
        pickle.dump(docstore, f)
    with open(p / "config.json", "w") as f:
        json.dump({"team": team, "model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "count": len(docstore)}, f)
    print(f"💾 Saved {team} index → {p}  ({len(docstore)} tickets)")


def load_team_index(db_dir: Path, team: str):
    p = _team_index_path(db_dir, team)
    if not (p / "index.faiss").exists():
        raise FileNotFoundError(
            f"No vector index for '{team}' at {p}.\n"
            f"Run: python vector_rag.py build --json <json_dir> --db {db_dir}"
        )
    index = faiss.read_index(str(p / "index.faiss"))
    with open(p / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open(p / "config.json") as f:
        config = json.load(f)
    print(f"✅ Loaded {team} index: {config['count']} tickets | model: {config['model']}")
    return index, docstore, config


def build_indexes(json_dir: Path, db_dir: Path, batch_size: int = 256):
    """
    Reads all JSON tickets, groups by support_team, builds one FAISS index
    per team (DSN, Appli, Outils). Incremental: only embeds new tickets.
    Called by auto_pipeline.py after new tickets are ingested.
    """
    json_dir = Path(json_dir)
    db_dir   = Path(db_dir)

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
            by_team[team if team in TEAMS else "Unknown"].append(ticket)
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
            failed += 1

    print(f"\n  Team breakdown:")
    for t, tickets in by_team.items():
        if tickets:
            print(f"    {t:10s} {len(tickets):5d} tickets")
    if failed:
        print(f"  ❌ Failed to parse: {failed}")

    model = None  # lazy-load only when needed

    for team in TEAMS:
        tickets = by_team[team]
        if not tickets:
            print(f"⚠️  No tickets for team {team} — skipping")
            continue

        # Incremental: detect already-indexed tickets
        existing_index, existing_docstore = None, []
        existing_refs: set = set()
        p = _team_index_path(db_dir, team)
        if (p / "index.faiss").exists():
            try:
                existing_index, existing_docstore, _ = load_team_index(db_dir, team)
                existing_refs = {d.get("reference", "") for d in existing_docstore}
                print(f"\n📌 {team}: existing index has {len(existing_docstore)} tickets")
            except Exception as e:
                print(f"  ⚠️  Could not load existing {team} index ({e}), rebuilding fully")

        new_tickets = [t for t in tickets if t.get("reference", "") not in existing_refs]
        if not new_tickets:
            print(f"  ✅ {team}: no new tickets, index is up-to-date")
            continue

        print(f"\n{'─'*50}")
        print(f"📌 {team}: {len(new_tickets)} new tickets to embed")

        if model is None:
            print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
            print(f"   (First run downloads ~1GB)\n")
            model = SentenceTransformer(EMBEDDING_MODEL)

        new_docstore = [ticket_to_display(t) for t in new_tickets]
        texts        = [ticket_to_embed_text(t) for t in new_tickets]

        print(f"⚡ Embedding {len(texts)} tickets...")
        new_matrix = _embed_texts(texts, model, batch_size)

        if existing_index is not None:
            existing_index.add(new_matrix)
            merged_index    = existing_index
            merged_docstore = existing_docstore + new_docstore
        else:
            merged_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            merged_index.add(new_matrix)
            merged_docstore = new_docstore

        save_team_index(db_dir, team, merged_index, merged_docstore)
        print(f"  ✅ {team}: {len(merged_docstore)} total ({len(new_docstore)} added)")

    print(f"\n🎉 All indexes built.")


# ─────────────────────────────────────────────
# 5. RETRIEVER
# ─────────────────────────────────────────────

def retrieve(query: str, index, docstore: list, model: SentenceTransformer, top_k: int = TOP_K) -> list:
    query_vec = model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
    similarities, indices = index.search(query_vec, top_k)

    hits = []
    for rank, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
        if idx < 0:
            continue
        doc = docstore[idx]
        hits.append({
            "rank":       rank + 1,
            "similarity": round(float(sim), 4),
            "reference":  doc["reference"],
            "title":      doc["title"],
            "version":    doc["version"],
            "patches":    doc["patches"],
            "resolution": doc["resolution"],
            "document":   doc["text"],
        })
    return hits


# ─────────────────────────────────────────────
# 6. GENERATOR
# ─────────────────────────────────────────────

PROMPT = """You are a Sopra HR technical support assistant.
Answer the user's question using the retrieved tickets below.

Rules:
- Answer in the SAME language as the question (French / English / Spanish)
- Be specific: mention patch numbers, field names, steps if relevant
- If tickets don't fully answer the question, say so clearly

User question:
{question}

Retrieved tickets (by similarity):
{context}

Answer:"""


def generate_answer(question: str, hits: list, bedrock_client) -> str:
    context = "\n\n".join(
        f"--- Ticket {h['reference']} (sim: {h['similarity']}) ---\n{h['document']}"
        for h in hits[:3]
    )
    try:
        return call_bedrock(bedrock_client, PROMPT.format(question=question, context=context))
    except Exception as e:
        return f"[Generation error: {e}]"


# ─────────────────────────────────────────────
# 7. EVALUATOR
# ─────────────────────────────────────────────

def evaluate(eval_path: str, index, docstore, embed_model, bedrock_client, output_path: str):
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    items = eval_data["eval_items"]
    print(f"\n🎯 Evaluating {len(items)} questions...\n")

    results, recall_at, correct_sims = [], {1: 0, 3: 0, 5: 0}, []

    for i, item in enumerate(items, 1):
        question     = item["question"]
        expected_ref = item["expected_ticket_ref"]

        hits           = retrieve(question, index, docstore, embed_model, top_k=TOP_K)
        retrieved_refs = [h["reference"] for h in hits]

        for k in [1, 3, 5]:
            if expected_ref in retrieved_refs[:k]:
                recall_at[k] += 1

        correct_hit = next((h for h in hits if h["reference"] == expected_ref), None)
        if correct_hit:
            correct_sims.append(correct_hit["similarity"])

        answer = generate_answer(question, hits, bedrock_client)

        results.append({
            "id":               item["id"],
            "question":         question,
            "expected_ref":     expected_ref,
            "retrieved_refs":   retrieved_refs,
            "hit_at_1":         expected_ref in retrieved_refs[:1],
            "hit_at_3":         expected_ref in retrieved_refs[:3],
            "hit_at_5":         expected_ref in retrieved_refs[:5],
            "top_similarity":   hits[0]["similarity"] if hits else 0,
            "generated_answer": answer,
            "expected_answer":  item["expected_answer"],
        })

        hit_str = "✅" if results[-1]["hit_at_3"] else "❌"
        print(f"  [{i:3d}/{len(items)}] {hit_str} R@3 | sim={results[-1]['top_similarity']:.3f} | {question[:60]}")

        if i % 10 == 0:
            time.sleep(0.5)

    n       = len(items)
    metrics = {
        "total_questions":         n,
        "recall_at_1":             round(recall_at[1] / n, 4),
        "recall_at_3":             round(recall_at[3] / n, 4),
        "recall_at_5":             round(recall_at[5] / n, 4),
        "mean_correct_similarity": round(sum(correct_sims) / len(correct_sims), 4) if correct_sims else 0,
    }

    print(f"\n{'═'*50}")
    print(f"  Recall@1 : {metrics['recall_at_1']:.1%}")
    print(f"  Recall@3 : {metrics['recall_at_3']:.1%}")
    print(f"  Recall@5 : {metrics['recall_at_5']:.1%}")
    print(f"  Mean sim : {metrics['mean_correct_similarity']:.4f}")
    print(f"{'═'*50}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "evaluated_at": datetime.utcnow().isoformat() + "Z",
                "embedding_model": EMBEDDING_MODEL,
                "bedrock_model": BEDROCK_MODEL,
            },
            "metrics": metrics,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved → {output_path}")
    return metrics


# ─────────────────────────────────────────────
# 8. INTERACTIVE QUERY
# ─────────────────────────────────────────────

def interactive_query(db_path: str):
    print(f"🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index, docstore, _ = load_index(db_path)
    bedrock = get_bedrock_client()
    print("Type your question (or 'quit')\n")

    while True:
        question = input("❓ ").strip()
        if question.lower() in ("quit", "exit", "q"): break
        if not question: continue

        hits   = retrieve(question, index, docstore, model)
        answer = generate_answer(question, hits, bedrock)

        print(f"\n📋 TOP MATCHES:")
        for h in hits[:3]:
            print(f"  #{h['rank']} [{h['similarity']:.3f}] {h['reference']} — {h['title'][:60]}")
            if h["patches"]: print(f"       Patches: {', '.join(h['patches'])}")
        print(f"\n💬 ANSWER:\n{answer}\n{'─'*60}\n")


# ─────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap
    parser = argparse.ArgumentParser(
        description="Vector RAG — Sopra HR (FAISS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Build per-team indexes (main pipeline)
          python vector_rag.py build --json data/json --db data/indexes

          # Build single flat index from a cleaned folder
          python vector_rag.py index --tickets data/cleaned --db faiss_db

          # Query a flat index interactively
          python vector_rag.py query --db faiss_db

          # Evaluate retrieval quality
          python vector_rag.py evaluate --db faiss_db --eval output/eval.json --output output/results.json
        """)
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build — per-team indexes (replaces rag_assistant.py index)
    bld = sub.add_parser("build", help="Build per-team FAISS indexes from JSON tickets")
    bld.add_argument("--json",  type=Path, required=True, help="Folder of cleaned JSON tickets")
    bld.add_argument("--db",   type=Path, required=True, help="Output folder for indexes")
    bld.add_argument("--batch", type=int,  default=256,   help="Embedding batch size")

    # index — single flat index (legacy)
    idx = sub.add_parser("index", help="Build a single flat FAISS index")
    idx.add_argument("--tickets", required=True)
    idx.add_argument("--db",      required=True)
    idx.add_argument("--batch",   type=int, default=256)

    qry = sub.add_parser("query", help="Query a flat index")
    qry.add_argument("--db",       required=True)
    qry.add_argument("--question", default=None)

    evl = sub.add_parser("evaluate", help="Evaluate retrieval against an eval set")
    evl.add_argument("--db",     required=True)
    evl.add_argument("--eval",   required=True)
    evl.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "build":
        build_indexes(args.json, args.db, args.batch)

    elif args.command == "index":
        index_tickets(args.tickets, args.db, args.batch)

    elif args.command == "query":
        if args.question:
            model = SentenceTransformer(EMBEDDING_MODEL)
            index, docstore, _ = load_index(args.db)
            bedrock = get_bedrock_client()
            hits    = retrieve(args.question, index, docstore, model)
            answer  = generate_answer(args.question, hits, bedrock)
            print("\n📋 TOP MATCHES:")
            for h in hits[:3]:
                print(f"  #{h['rank']} [{h['similarity']:.3f}] {h['reference']} — {h['title'][:60]}")
                if h["patches"]: print(f"       Patches: {', '.join(h['patches'])}")
            print(f"\n💬 ANSWER:\n{answer}")
        else:
            interactive_query(args.db)

    elif args.command == "evaluate":
        model = SentenceTransformer(EMBEDDING_MODEL)
        index, docstore, _ = load_index(args.db)
        bedrock = get_bedrock_client()
        evaluate(args.eval, index, docstore, model, bedrock, args.output)