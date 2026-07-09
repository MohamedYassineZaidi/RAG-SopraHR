#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_hybrid.py
==================
Complete evaluation of the Hybrid RAG pipeline (Vector + BM25 + PageIndex).

Measures ALL metrics in a single run:

  Part 1 — Recall@1 / @3 / @5
    Standard self-retrieval: is the exact ticket in the top-k results?
    Measured for each index individually AND for the hybrid fusion.

  Part 2 — Cluster Routing Accuracy
    Does the LLM route the query to the correct cluster? (PageIndex step 1)
    A routing miss means the ticket is invisible to the PageIndex component.

  Part 3 — Effective Recall
    For every miss, checks if the returned ticket has a similar resolution
    (keyword overlap ≥ 25%). A "useful miss" still helps the consultant even
    if it's not the exact ticket.

    Effective recall = (exact hits + useful misses) / total

  Part 4 — Per-index comparison table
    Side-by-side view: vector | bm25 | pageindex | hybrid

Usage:
    python scripts/evaluate_hybrid.py \\
        --index  data/pageindex \\
        --db     data/indexes \\
        --bm25   data/bm25 \\
        --json   data/json \\
        --team   Appli \\
        --samples 50

    # All three teams
    for team in Appli DSN Outils; do
        python scripts/evaluate_hybrid.py \\
            --index data/pageindex --db data/indexes \\
            --bm25 data/bm25 --json data/json \\
            --team $team --samples 50
    done
"""

import re
import json
import time
import random
import argparse
from pathlib import Path

import faiss
import pickle
from sentence_transformers import SentenceTransformer

from rag_utils import (
    TEAMS, TOP_K, RRF_K, EMBEDDING_MODEL,
    CLUSTER_DESCRIPTIONS,
    get_bedrock_client, call_bedrock,
)
from vectorless_rag import (
    load_pageindex,
    retrieve as pageindex_retrieve,
    CLUSTER_SELECTION_PROMPT,
    TICKET_SELECTION_PROMPT,
)
from bm25_rag import load_bm25_index, bm25_retrieve, _resolve_under_data
from hybrid_rag import (
    load_vector_index,
    vector_retrieve,
    reciprocal_rank_fusion,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_cluster_response(response: str, clusters: dict) -> list:
    """Parse LLM cluster selection response."""
    selected = []
    for line in response.strip().splitlines():
        line = line.strip().lstrip("-•* ").strip()
        for name in clusters:
            if name.lower() in line.lower() or line.lower() in name.lower():
                if name not in selected:
                    selected.append(name)
                break
    return selected


def resolution_overlap(res_a: str, res_b: str) -> float:
    """
    Keyword overlap between two resolution texts.
    Returns a score from 0.0 to 1.0.
    """
    STOPWORDS = {
        "le", "la", "les", "de", "du", "des", "en", "et", "ou", "un", "une",
        "à", "au", "aux", "ce", "il", "elle", "nous", "vous", "ils", "elles",
        "est", "sont", "a", "ont", "the", "and", "or", "is", "are", "to",
        "of", "in", "for", "on", "with", "this", "that", "it", "be",
        "non", "pas", "ne", "se", "si", "sur", "par", "pour",
    }

    def keywords(text: str) -> set:
        tokens = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    kw_a = keywords(res_a)
    kw_b = keywords(res_b)
    if not kw_a or not kw_b:
        return 0.0
    return round(len(kw_a & kw_b) / min(len(kw_a), len(kw_b)), 3)


# ─────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────

def evaluate_hybrid(
    index_dir: Path,
    db_dir: Path,
    bm25_dir: Path,
    json_dir: Path,
    team: str,
    n_samples: int = 50,
    relevance_threshold: float = 0.25,
):
    random.seed(42)

    print(f"\n{'═'*65}")
    print(f"  Hybrid RAG — Full Evaluation — Team: {team} | Samples: {n_samples}")
    print(f"{'═'*65}\n")

    # ── Load all data ──────────────────────────────────────────────
    all_json = []
    for f in sorted(json_dir.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("support_team") == team and t.get("description") and t.get("is_closed"):
            all_json.append(t)

    if not all_json:
        print(f"❌ No closed tickets with descriptions found for team {team}")
        return

    sample      = random.sample(all_json, min(n_samples, len(all_json)))
    all_tickets = {t["reference"]: t for t in all_json}

    print(f"🧠 Loading embedding model...")
    embed_model  = SentenceTransformer(EMBEDDING_MODEL)

    print(f"📦 Loading vector index ({team})...")
    vector_index, vector_docstore = load_vector_index(db_dir, team)
    for doc in vector_docstore:
        ref = doc.get("reference", "")
        if ref in all_tickets:
            all_tickets[ref].update(doc)

    print(f"📄 Loading PageIndex ({team})...")
    pageindex = load_pageindex(index_dir)
    clusters  = pageindex.get(team, {}).get("clusters", {})

    # Build ground-truth cluster map from pageindex
    ref_to_cluster = {}
    for cluster_name, cluster_info in clusters.items():
        for ref in cluster_info.get("refs", []):
            ref_to_cluster[ref] = cluster_name

    print(f"🔍 Loading BM25 index ({team})...")
    bm25_model, bm25_docstore = load_bm25_index(bm25_dir, team)

    print(f"☁️  Connecting to Bedrock...")
    bedrock = get_bedrock_client()

    # ── Metrics ────────────────────────────────────────────────────
    recall = {
        "vector":    {1: 0, 3: 0, 5: 0},
        "bm25":      {1: 0, 3: 0, 5: 0},
        "pageindex": {1: 0, 3: 0, 5: 0},
        "hybrid":    {1: 0, 3: 0, 5: 0},
    }

    # Routing (PageIndex step 1)
    cluster_hits      = 0
    cluster_misses    = 0
    cluster_miss_list = []

    # Effective recall (hybrid)
    useful_misses  = 0
    total_misses   = 0
    exact_hits     = 0

    # ── Per-ticket loop ────────────────────────────────────────────
    print(f"\n  {'#':>4}  {'Expected':12}  {'V':2}  {'B':2}  {'PI':2}  {'H':2}  "
          f"{'Routing':6}  {'Relevance':9}  Notes")
    print(f"  {'─'*4}  {'─'*12}  {'─'*2}  {'─'*2}  {'─'*2}  {'─'*2}  "
          f"{'─'*6}  {'─'*9}  {'─'*30}")

    for i, ticket in enumerate(sample, 1):
        query        = ticket["description"]
        expected_ref = ticket["reference"]
        true_cluster = ref_to_cluster.get(expected_ref, "Unknown")

        # ── Retrieve from all 3 indexes ────────────────────────────
        v_hits    = vector_retrieve(query, vector_index, vector_docstore, embed_model)
        bm25_hits = bm25_retrieve(query, bm25_model, bm25_docstore)
        pi_hits   = pageindex_retrieve(query, team, pageindex, bedrock, all_tickets)
        h_hits    = reciprocal_rank_fusion(v_hits, pi_hits, bm25_hits)

        # ── Recall metrics ─────────────────────────────────────────
        for mode, hits in [
            ("vector", v_hits), ("bm25", bm25_hits),
            ("pageindex", pi_hits), ("hybrid", h_hits)
        ]:
            refs = [h["reference"] for h in hits]
            for k in [1, 3, 5]:
                if expected_ref in refs[:k]:
                    recall[mode][k] += 1

        v_hit  = "✅" if expected_ref in [h["reference"] for h in v_hits[:3]]    else "❌"
        b_hit  = "✅" if expected_ref in [h["reference"] for h in bm25_hits[:3]] else "❌"
        pi_hit = "✅" if expected_ref in [h["reference"] for h in pi_hits[:3]]   else "❌"
        h_hit  = "✅" if expected_ref in [h["reference"] for h in h_hits[:3]]    else "❌"

        # ── Routing accuracy (PageIndex step 1) ────────────────────
        team_descs       = CLUSTER_DESCRIPTIONS.get(team, {})
        cluster_list_str = "\n".join(
            f"- {name}: {team_descs.get(name, '')} ({info['count']} tickets)"
            for name, info in clusters.items()
        )
        cluster_response = call_bedrock(
            bedrock,
            CLUSTER_SELECTION_PROMPT.format(
                team=team,
                cluster_list=cluster_list_str,
                query=query,
            ),
            max_tokens=200,
        )
        selected_clusters = parse_cluster_response(cluster_response, clusters)
        if not selected_clusters:
            selected_clusters = sorted(
                clusters, key=lambda k: clusters[k]["count"], reverse=True
            )[:2]

        cluster_correct = true_cluster in selected_clusters
        if cluster_correct:
            cluster_hits += 1
            routing_icon = "✅"
        else:
            cluster_misses += 1
            routing_icon = "❌"
            cluster_miss_list.append({
                "ref":          expected_ref,
                "true_cluster": true_cluster,
                "got_clusters": selected_clusters,
            })

        # ── Effective recall (hybrid) ──────────────────────────────
        hybrid_refs = [h["reference"] for h in h_hits]
        hit_exact   = expected_ref in hybrid_refs[:5]

        relevance_score = 0.0
        relevance_label = ""
        notes           = ""

        if hit_exact:
            exact_hits += 1
        else:
            total_misses += 1
            # Check if returned ticket has similar resolution
            returned_ref = hybrid_refs[0] if hybrid_refs else None
            if returned_ref and returned_ref in all_tickets:
                expected_res = all_tickets.get(expected_ref, {}).get("resolution", "")
                returned_res = all_tickets[returned_ref].get("resolution", "")
                relevance_score = resolution_overlap(expected_res, returned_res)
                if relevance_score >= relevance_threshold:
                    useful_misses += 1
                    relevance_label = f"{relevance_score:.2f} ✓"
                else:
                    relevance_label = f"{relevance_score:.2f}"

            if not cluster_correct:
                notes = f"routed to {selected_clusters[0] if selected_clusters else '?'} (true: {true_cluster[:20]})"
            elif returned_ref:
                notes = f"returned {returned_ref}"

        print(f"  {i:4d}  {expected_ref:12}  {v_hit}  {b_hit}  {pi_hit}  {h_hit}  "
              f"{routing_icon:6}  {relevance_label:9}  {notes[:40]}")

        time.sleep(0.3)

    # ── Summary ────────────────────────────────────────────────────
    n = len(sample)

    effective_recall = round((exact_hits + useful_misses) / n * 100, 1)
    routing_accuracy = round(cluster_hits / n * 100, 1)

    print(f"\n{'═'*65}")
    print(f"  RESULTS — Hybrid RAG Full Evaluation — Team {team} ({n} samples)")
    print(f"{'═'*65}")

    print(f"\n  PART 1 — Recall per index")
    print(f"  {'─'*50}")
    print(f"  {'Mode':12s} | R@1    | R@3    | R@5")
    print(f"  {'─'*12}-+--------+--------+--------")
    for mode in ["vector", "bm25", "pageindex", "hybrid"]:
        r      = recall[mode]
        marker = " ←" if mode == "hybrid" else ""
        print(f"  {mode:12s} | {r[1]/n:5.1%}  | {r[3]/n:5.1%}  | {r[5]/n:5.1%}{marker}")

    print(f"\n  PART 2 — Cluster Routing (PageIndex step 1)")
    print(f"  {'─'*50}")
    print(f"  Routing accuracy : {routing_accuracy}%  ({cluster_hits}/{n})")
    print(f"  Routing misses   : {cluster_misses}  (ticket invisible to PageIndex)")

    print(f"\n  PART 3 — Effective Recall (Hybrid)")
    print(f"  {'─'*50}")
    print(f"  Exact hits       : {exact_hits}  (hybrid R@5)")
    print(f"  Total misses     : {total_misses}")
    print(f"  Useful misses    : {useful_misses}  (similar resolution ≥{relevance_threshold*100:.0f}% overlap)")
    print(f"  Effective recall : {effective_recall}%  ← real-world usefulness")

    if cluster_miss_list:
        from collections import Counter
        miss_by_cluster = Counter(m["true_cluster"] for m in cluster_miss_list)
        print(f"\n  ROUTING FAILURES ({cluster_misses})")
        print(f"  {'─'*50}")
        for cluster, count in miss_by_cluster.most_common():
            print(f"  {cluster:30s}: {count} misses")

    print(f"\n{'═'*65}\n")

    # ── Save results ───────────────────────────────────────────────
    out = index_dir / f"eval_hybrid_full_{team.lower()}.json"
    results = {
        "team":     team,
        "n_samples": n,
        "recall": {
            mode: {str(k): round(v / n, 3) for k, v in r.items()}
            for mode, r in recall.items()
        },
        "routing_accuracy":  routing_accuracy,
        "routing_hits":      cluster_hits,
        "routing_misses":    cluster_misses,
        "exact_hits":        exact_hits,
        "useful_misses":     useful_misses,
        "total_misses":      total_misses,
        "effective_recall":  effective_recall,
        "relevance_threshold": relevance_threshold,
        "routing_failures":  cluster_miss_list,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved → {out}")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sopra HR — Hybrid RAG Full Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_hybrid.py \\
      --index data/pageindex --db data/indexes \\
      --bm25 data/bm25 --json data/json --team Appli

  python scripts/evaluate_hybrid.py \\
      --index data/pageindex --db data/indexes \\
      --bm25 data/bm25 --json data/json --team DSN --samples 50

  python scripts/evaluate_hybrid.py \\
      --index data/pageindex --db data/indexes \\
      --bm25 data/bm25 --json data/json --team Outils --samples 50
        """
    )
    parser.add_argument("--index",     type=Path, required=True, help="PageIndex folder")
    parser.add_argument("--db",        type=Path, required=True, help="Vector indexes folder")
    parser.add_argument("--bm25",      type=Path, required=True, help="BM25 indexes folder")
    parser.add_argument("--json",      type=Path, required=True, help="JSON tickets folder")
    parser.add_argument("--team",      type=str,  required=True, choices=TEAMS)
    parser.add_argument("--samples",   type=int,  default=50)
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Min resolution overlap to count as useful miss (default: 0.25)")

    args = parser.parse_args()

    evaluate_hybrid(
        index_dir=_resolve_under_data(args.index, "--index"),
        db_dir=_resolve_under_data(args.db, "--db"),
        bm25_dir=_resolve_under_data(args.bm25, "--bm25"),
        json_dir=_resolve_under_data(args.json, "--json"),
        team=args.team,
        n_samples=args.samples,
        relevance_threshold=args.threshold,
    )
    