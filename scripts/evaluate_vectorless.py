#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_vectorless.py
======================
Fair evaluation of vectorless_rag.py that measures each step of the
two-step PageIndex pipeline independently.

WHY the standard self-retrieval test gives misleading results:
  The standard eval asks "did it find the exact same ticket?" — but
  vectorless searches at most 150 summaries from the selected clusters.
  If the ticket's cluster is missed in step 1, it becomes invisible.
  This penalises step 1 failures twice (once for routing, once for retrieval).

THIS EVAL measures three things separately:

  Part 1 — Cluster routing accuracy
    Does Claude route the query to the ticket's actual cluster?
    This is the step vectorless controls directly.
    A cluster hit means the ticket was at least visible to step 2.

  Part 2 — Ticket selection accuracy (given correct cluster)
    Among tickets where step 1 succeeded, did step 2 find the right ticket?
    This measures Claude's ability to rank summaries correctly.
    Filters out step 1 failures so you see each step's true accuracy.

  Part 3 — Resolution relevance (real usefulness)
    For every miss, checks if the returned ticket has a similar resolution
    to the expected one (keyword overlap). Because returning a "twin" ticket
    is still useful — a consultant doesn't care about the reference number,
    only whether the resolution applies.

Usage:
    python scripts/evaluate_vectorless.py \\
        --index data/pageindex \\
        --json  data/json \\
        --team  Appli \\
        --samples 50

    # All three teams
    python scripts/evaluate_vectorless.py \\
        --index data/pageindex --json data/json --team DSN   --samples 50
    python scripts/evaluate_vectorless.py \\
        --index data/pageindex --json data/json --team Appli --samples 50
    python scripts/evaluate_vectorless.py \\
        --index data/pageindex --json data/json --team Outils --samples 50
"""

import re
import json
import time
import random
import argparse
from pathlib import Path

from rag_utils import (
    TEAMS, CLUSTER_RULES, CLUSTER_DESCRIPTIONS, assign_cluster,
    get_bedrock_client, call_bedrock,
)
from vectorless_rag import (
    load_pageindex,
    CLUSTER_SELECTION_PROMPT,
    TICKET_SELECTION_PROMPT,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_cluster_response(response: str, clusters: dict) -> list:
    """Same parser as vectorless_rag.retrieve — keeps results comparable."""
    selected = []
    for line in response.strip().splitlines():
        line = line.strip().lstrip("-•* ").strip()
        for name in clusters:
            if name.lower() in line.lower() or line.lower() in name.lower():
                if name not in selected:
                    selected.append(name)
                break
    return selected


def parse_ref_response(response: str) -> list:
    """Same ref parser as vectorless_rag.retrieve — handles THINKING:/REFS: format."""
    ref_pattern = re.compile(r'\b(FR|AF|SP)\s*W\d{5,}\b')
    # Only parse refs from after "REFS:" section if present
    ref_section = response.split("REFS:")[-1] if "REFS:" in response else response
    selected = []
    for line in ref_section.strip().splitlines():
        m = ref_pattern.search(line)
        if m:
            ref = re.sub(r'\s+', ' ', m.group(0).replace("W", " W").strip())
            if ref not in selected:
                selected.append(ref)
    return selected


def resolution_overlap(res_a: str, res_b: str, min_words: int = 3) -> float:
    """
    Measures keyword overlap between two resolution texts.
    Returns a score from 0.0 to 1.0.
    Filters stopwords and short tokens.
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
    intersection = kw_a & kw_b
    return round(len(intersection) / min(len(kw_a), len(kw_b)), 3)


# ─────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────

def evaluate_fair(
    index_dir: Path,
    json_dir: Path,
    team: str,
    n_samples: int = 50,
    relevance_threshold: float = 0.25,
):
    random.seed(42)

    print(f"\n{'═'*65}")
    print(f"  Fair Vectorless Evaluation — Team: {team} | Samples: {n_samples}")
    print(f"{'═'*65}\n")

    # ── Load data ──────────────────────────────────────────────────
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
    pageindex   = load_pageindex(index_dir)
    clusters    = pageindex.get(team, {}).get("clusters", {})
    bedrock     = get_bedrock_client()

    # ── Build ground-truth cluster map from pageindex ──────────────
    # The pageindex already stores which refs belong to each cluster
    ref_to_cluster = {}
    for cluster_name, cluster_info in clusters.items():
        for ref in cluster_info.get("refs", []):
            ref_to_cluster[ref] = cluster_name

    # ── Metrics ────────────────────────────────────────────────────
    # Part 1
    cluster_hits       = 0   # step 1 routed to correct cluster
    cluster_misses     = 0
    cluster_miss_list  = []

    # Part 2 (only tickets where cluster was found)
    ticket_hits_at1    = 0
    ticket_hits_at3    = 0
    ticket_hits_at5    = 0
    ticket_eligible    = 0  # tickets where cluster routing succeeded

    # Part 3 (resolution relevance for misses)
    relevant_misses    = 0  # misses where returned ticket has similar resolution
    total_misses       = 0
    miss_details       = []

    # ── Per-ticket loop ────────────────────────────────────────────
    print(f"  {'#':>4}  {'Expected':12}  {'Cluster':6}  {'Ticket':6}  {'Relevance':9}  Notes")
    print(f"  {'─'*4}  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*30}")

    for i, ticket in enumerate(sample, 1):
        query        = ticket["description"]
        expected_ref = ticket["reference"]
        true_cluster = ref_to_cluster.get(expected_ref, "Unknown")

        # ── Step 1: cluster routing ────────────────────────────────
        team_descs = CLUSTER_DESCRIPTIONS.get(team, {})
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

        # Fallback (same as vectorless_rag)
        if not selected_clusters:
            selected_clusters = sorted(
                clusters, key=lambda k: clusters[k]["count"], reverse=True
            )[:2]

        cluster_correct = true_cluster in selected_clusters
        if cluster_correct:
            cluster_hits += 1
        else:
            cluster_misses += 1
            cluster_miss_list.append({
                "ref":            expected_ref,
                "true_cluster":   true_cluster,
                "got_clusters":   selected_clusters,
                "query_preview":  query[:80],
            })

        # ── Step 2: ticket selection (always run, even on cluster miss) ──
        # We run it regardless so we can also measure resolution relevance
        candidate_summaries, candidate_refs = [], []
        for cname in selected_clusters:
            cluster_data = clusters.get(cname, {})
            for s, r in zip(cluster_data.get("tickets", []), cluster_data.get("refs", [])):
                candidate_summaries.append(s)
                candidate_refs.append(r)
            if len(candidate_summaries) >= 150:
                break
        candidate_summaries = candidate_summaries[:150]

        ticket_response = call_bedrock(
            bedrock,
            TICKET_SELECTION_PROMPT.format(
                query=query,
                ticket_list="\n".join(candidate_summaries),
            ),
            max_tokens=300,
        )
        selected_refs = parse_ref_response(ticket_response)

        hit_at1 = expected_ref in selected_refs[:1]
        hit_at3 = expected_ref in selected_refs[:3]
        hit_at5 = expected_ref in selected_refs[:5]

        # Part 2 metrics — only count when cluster routing was correct
        if cluster_correct:
            ticket_eligible += 1
            if hit_at1: ticket_hits_at1 += 1
            if hit_at3: ticket_hits_at3 += 1
            if hit_at5: ticket_hits_at5 += 1

        # Part 3 — resolution relevance for all misses
        relevance_score = 0.0
        relevance_label = ""
        if not hit_at3:
            total_misses += 1
            top_ref    = selected_refs[0] if selected_refs else None
            top_ticket = all_tickets.get(top_ref) if top_ref else None
            if top_ticket:
                res_expected = ticket.get("resolution", "")
                res_returned = top_ticket.get("resolution", "")
                relevance_score = resolution_overlap(res_expected, res_returned)
                if relevance_score >= relevance_threshold:
                    relevant_misses += 1
                    relevance_label = f"{relevance_score:.2f} ✓"
                else:
                    relevance_label = f"{relevance_score:.2f}"
            else:
                relevance_label = "no result"
            miss_details.append({
                "ref":            expected_ref,
                "true_cluster":   true_cluster,
                "cluster_hit":    cluster_correct,
                "top_returned":   top_ref,
                "relevance":      relevance_score,
            })

        # ── Console row ────────────────────────────────────────────
        c_mark = "✅" if cluster_correct else "❌"
        t_mark = "✅" if hit_at3        else "❌"
        note   = ""
        if not cluster_correct:
            note = f"routed to {selected_clusters[0] if selected_clusters else '?'} (true: {true_cluster})"
        elif not hit_at3:
            note = f"returned {selected_refs[0] if selected_refs else 'none'}"

        print(
            f"  {i:4d}  {expected_ref:12}  {c_mark}      {t_mark}      "
            f"{relevance_label:9}  {note[:50]}"
        )

        time.sleep(0.3)

    # ── Summary ────────────────────────────────────────────────────
    n = len(sample)
    e = max(ticket_eligible, 1)

    print(f"\n{'═'*65}")
    print(f"  RESULTS — Vectorless RAG (Fair) — Team {team} ({n} samples)")
    print(f"{'═'*65}\n")

    print(f"  PART 1 — Cluster Routing")
    print(f"  {'─'*40}")
    print(f"  Routing accuracy : {cluster_hits/n:.1%}  ({cluster_hits}/{n})")
    print(f"  Routing misses   : {cluster_misses}  (ticket invisible to step 2)")
    print()

    print(f"  PART 2 — Ticket Selection (given correct cluster routing)")
    print(f"  {'─'*40}")
    if ticket_eligible > 0:
        print(f"  Eligible tickets : {ticket_eligible}  (cluster was correctly found)")
        print(f"  Recall@1         : {ticket_hits_at1/e:.1%}  ({ticket_hits_at1}/{ticket_eligible})")
        print(f"  Recall@3         : {ticket_hits_at3/e:.1%}  ({ticket_hits_at3}/{ticket_eligible})")
        print(f"  Recall@5         : {ticket_hits_at5/e:.1%}  ({ticket_hits_at5}/{ticket_eligible})")
    else:
        print(f"  No eligible tickets (all cluster routings failed)")
    print()

    print(f"  PART 3 — Resolution Relevance for Misses")
    print(f"  {'─'*40}")
    if total_misses > 0:
        print(f"  Total misses     : {total_misses}")
        print(f"  Useful misses    : {relevant_misses}  (returned ticket had similar resolution, ≥{relevance_threshold:.0%} overlap)")
        print(f"  Useful miss rate : {relevant_misses/total_misses:.1%}")
        total_useful = (n - total_misses) + relevant_misses
        print(f"  Effective recall : {total_useful/n:.1%}  (exact hits + useful misses) ← real-world usefulness")
    else:
        print(f"  No misses — perfect retrieval")
    print()

    # ── Cluster routing breakdown ──────────────────────────────────
    if cluster_miss_list:
        print(f"  CLUSTER ROUTING FAILURES ({len(cluster_miss_list)})")
        print(f"  {'─'*40}")
        from collections import Counter
        missed_clusters = Counter(m["true_cluster"] for m in cluster_miss_list)
        for cluster, count in missed_clusters.most_common():
            print(f"  {cluster:30s}: {count} misses")
    print()

    # ── Save results ───────────────────────────────────────────────
    out = index_dir / f"eval_vectorless_fair_{team.lower()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "team":     team,
            "mode":     "vectorless_fair",
            "n_samples": n,
            "part1_cluster_routing": {
                "accuracy":       round(cluster_hits / n, 3),
                "hits":           cluster_hits,
                "misses":         cluster_misses,
                "miss_details":   cluster_miss_list,
            },
            "part2_ticket_selection": {
                "eligible":       ticket_eligible,
                "recall_at_1":    round(ticket_hits_at1 / e, 3),
                "recall_at_3":    round(ticket_hits_at3 / e, 3),
                "recall_at_5":    round(ticket_hits_at5 / e, 3),
            },
            "part3_resolution_relevance": {
                "total_misses":   total_misses,
                "useful_misses":  relevant_misses,
                "useful_rate":    round(relevant_misses / max(total_misses, 1), 3),
                "effective_recall": round(((n - total_misses) + relevant_misses) / n, 3),
                "threshold":      relevance_threshold,
                "miss_details":   miss_details,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved → {out}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fair evaluation of vectorless RAG — measures each step independently.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_vectorless.py --index data/pageindex --json data/json --team Appli
  python scripts/evaluate_vectorless.py --index data/pageindex --json data/json --team DSN --samples 30
        """
    )
    ap.add_argument("--index",     type=Path, required=True,
                    help="PageIndex folder (contains pageindex.json)")
    ap.add_argument("--json",      type=Path, required=True,
                    help="Folder of structured JSON ticket files")
    ap.add_argument("--team",      type=str,  required=True, choices=TEAMS,
                    help="Team to evaluate: DSN / Appli / Outils")
    ap.add_argument("--samples",   type=int,  default=50,
                    help="Number of tickets to sample (default: 50)")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="Min resolution overlap to count a miss as useful (default: 0.25)")
    args = ap.parse_args()

    evaluate_fair(args.index, args.json, args.team, args.samples, args.threshold)
