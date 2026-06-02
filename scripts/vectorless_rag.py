#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vectorless_rag.py
=================
Vectorless RAG using a 3-level PageIndex hierarchy.
No embeddings, no FAISS — Claude navigates a structured index
of all tickets using language understanding alone.

Architecture:
  Level 1 — Team node (DSN / Appli / Outils)
  Level 2 — Topic clusters (7-8 per team, keyword-assigned)
  Level 3 — Ticket summaries (one compact line per ticket)

At query time Claude runs in two steps:
  Step 1 — reads L2 cluster descriptions, picks the 1-3 most relevant
  Step 2 — scans L3 summaries from those clusters, picks top-5 tickets

Requires:
  pip install boto3 python-dotenv

Usage:
  # Build the PageIndex from JSON tickets (run once)
  python vectorless_rag.py build --json data/json --index data/pageindex

  # Interactive query
  python vectorless_rag.py query --index data/pageindex --team DSN

  # Single question
  python vectorless_rag.py query --index data/pageindex --team Appli \\
      --question "Erreur ORA-00942 lors du lancement REGDSN"

  # Evaluate retrieval quality
  python vectorless_rag.py evaluate --index data/pageindex --json data/json --team DSN
"""

import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional

from rag_utils import (
    TEAMS, TOP_K,
    CLUSTER_RULES, CLUSTER_DESCRIPTIONS, assign_cluster, ticket_to_summary,
    get_bedrock_client, call_bedrock,
    generate_suggestion, print_results,
)


# ─────────────────────────────────────────────
# 1. BUILD PAGEINDEX
# ─────────────────────────────────────────────

def build_pageindex(json_dir: Path, index_dir: Path):
    """
    Builds the 3-level PageIndex from JSON tickets and saves it to disk.
    Incremental: skips rebuild if no new tickets detected.

    Output files:
      index_dir/pageindex.json   — full hierarchical index (all teams)
      index_dir/{team}_l3.txt    — flat L3 ticket list per team
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(json_dir.glob("*.json"))
    print(f"\n📂 Loading {len(files)} tickets...")

    tickets = []
    for f in files:
        try:
            tickets.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")

    # --- Incremental: detect new tickets vs existing pageindex ---
    existing_refs = set()
    pi_path = index_dir / "pageindex.json"
    if pi_path.exists():
        try:
            with open(pi_path, encoding="utf-8") as f:
                old_pi = json.load(f)
            for team_data in old_pi.values():
                for cluster in team_data.get("clusters", {}).values():
                    existing_refs.update(cluster.get("refs", []))
        except Exception:
            pass

    all_refs = {t.get("reference", "") for t in tickets}
    new_refs = all_refs - existing_refs
    if not new_refs:
        print(f"  ✅ PageIndex is up-to-date, no new tickets")
        return old_pi if pi_path.exists() else {}

    print(f"  {len(new_refs)} new tickets detected, rebuilding PageIndex...")

    pageindex = {}

    for team in TEAMS:
        team_tickets = [t for t in tickets if t.get("support_team") == team]
        print(f"\n📌 Building {team} index ({len(team_tickets)} tickets)...")

        # Assign each ticket to a cluster
        clusters: dict = {}
        for ticket in team_tickets:
            cluster = assign_cluster(ticket, team)
            clusters.setdefault(cluster, []).append(ticket)

        # Build cluster nodes
        cluster_nodes = {}
        for cluster_name, cluster_tickets in sorted(clusters.items()):
            # Fix 3: sort tickets so those with resolutions + patches come first.
            # When truncated at 150, Claude sees the most actionable tickets,
            # not 150 arbitrary ones from file-system order.
            cluster_tickets_sorted = sorted(
                cluster_tickets,
                key=lambda t: (bool(t.get("resolution")), bool(t.get("patches"))),
                reverse=True,
            )
            summaries   = [ticket_to_summary(t) for t in cluster_tickets_sorted]
            has_res     = sum(1 for t in cluster_tickets if t.get("resolution"))
            has_patches = sum(1 for t in cluster_tickets if t.get("patches"))
            versions    = list({t.get("version", "") for t in cluster_tickets if t.get("version")})[:5]

            cluster_nodes[cluster_name] = {
                "name":               cluster_name,
                "count":              len(cluster_tickets),
                "versions":           versions,
                "has_resolution_pct": round(has_res / len(cluster_tickets) * 100),
                "has_patches_pct":    round(has_patches / len(cluster_tickets) * 100),
                "tickets":            summaries,
                "refs":               [t.get("reference", "") for t in cluster_tickets_sorted],
            }
            print(f"    {cluster_name:30s}: {len(cluster_tickets):4d} tickets  "
                  f"(res: {has_res}, patches: {has_patches})")

        pageindex[team] = {
            "team":     team,
            "count":    len(team_tickets),
            "clusters": cluster_nodes,
        }

    # Save full index
    out_path = index_dir / "pageindex.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pageindex, f, ensure_ascii=False, indent=2)

    # Save flat L3 text files for debugging / inspection
    for team in TEAMS:
        lines = []
        for cluster_name, cluster in pageindex[team]["clusters"].items():
            lines.append(f"\n## {cluster_name} ({cluster['count']} tickets)")
            lines.extend(cluster["tickets"])
        out = index_dir / f"{team.lower()}_l3.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ {team} L3 saved → {out}  ({len(lines)} lines)")

    print(f"\n🎉 PageIndex built → {out_path}")
    return pageindex


def load_pageindex(index_dir: Path) -> dict:
    path = index_dir / "pageindex.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No PageIndex at {path}.\n"
            f"Run: python vectorless_rag.py build --json <json_dir> --index {index_dir}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 2. RETRIEVAL — two Bedrock calls per query
# ─────────────────────────────────────────────

CLUSTER_SELECTION_PROMPT = """You are searching a ticket database for a Sopra HR support team.

Available topic clusters for team {team}:
{cluster_list}

New ticket query:
{query}

Which clusters are most likely to contain similar resolved tickets?
Reply with ONLY the cluster names, one per line, most relevant first. Maximum 3 clusters."""

TICKET_SELECTION_PROMPT = """You are searching resolved support tickets for Sopra HR.

New ticket query:
{query}

Candidate tickets (format: REF | Title | Version / Symptôme / Action / Codes & Patches):
{ticket_list}

First think about which tickets best match the query's specific symptom and resolution context.
Then reply in this exact format:
THINKING: <one sentence explaining your selection criteria>
REFS:
FR WXXXXX
FR WYYYYY
FR WZZZZZ

Maximum 5 ticket references. Most relevant first."""


def retrieve(
    query: str,
    team: str,
    pageindex: dict,
    bedrock_client,
    all_tickets: dict,
) -> list:
    """
    Two-step PageIndex retrieval with iterative fallback.

    Step 1 — Claude selects relevant clusters using richer descriptions.
    Step 2 — Claude scans L3 summaries with a thinking prompt and picks refs.
    Fallback — if step 2 returns < 2 results, tries next-best clusters.

    Returns a ranked list of ticket dicts.
    """
    clusters = pageindex.get(team, {}).get("clusters", {})
    ref_pattern = re.compile(r'\b(FR|AF|SP)\s*W\d{5,}\b')

    # ── Step 1: cluster selection with richer descriptions ──────────
    team_descs = CLUSTER_DESCRIPTIONS.get(team, {})
    cluster_list = "\n".join(
        f"- {name}: {team_descs.get(name, '')} ({info['count']} tickets)"
        for name, info in clusters.items()
    )
    cluster_response = call_bedrock(
        bedrock_client,
        CLUSTER_SELECTION_PROMPT.format(team=team, cluster_list=cluster_list, query=query),
        max_tokens=200,
    )

    selected_clusters = []
    for line in cluster_response.strip().splitlines():
        line = line.strip().lstrip("-•* ").strip()
        for name in clusters:
            if name.lower() in line.lower() or line.lower() in name.lower():
                if name not in selected_clusters:
                    selected_clusters.append(name)
                break

    # Fallback: top 2 clusters by size
    if not selected_clusters:
        selected_clusters = sorted(
            clusters, key=lambda k: clusters[k]["count"], reverse=True
        )[:2]

    # ── Step 2: ticket selection from chosen clusters ────────────────
    def _select_from_clusters(cluster_names: list, limit: int = 150) -> list:
        """Collects candidate summaries from given clusters and asks Claude."""
        candidate_summaries, candidate_refs = [], []
        for cluster_name in cluster_names:
            cluster = clusters.get(cluster_name, {})
            for s, r in zip(cluster.get("tickets", []), cluster.get("refs", [])):
                candidate_summaries.append(s)
                candidate_refs.append(r)
            if len(candidate_summaries) >= limit:
                break
        candidate_summaries = candidate_summaries[:limit]

        if not candidate_summaries:
            return []

        ticket_response = call_bedrock(
            bedrock_client,
            TICKET_SELECTION_PROMPT.format(
                query=query,
                ticket_list="\n".join(candidate_summaries),
            ),
            max_tokens=300,
        )

        # Parse refs from after "REFS:" section (Fix 5)
        ref_section = ticket_response.split("REFS:")[-1] if "REFS:" in ticket_response else ticket_response
        refs = []
        for line in ref_section.strip().splitlines():
            m = ref_pattern.search(line)
            if m:
                ref = re.sub(r'\s+', ' ', m.group(0).replace("W", " W").strip())
                if ref not in refs:
                    refs.append(ref)
        return refs

    selected_refs = _select_from_clusters(selected_clusters)

    # ── Fix 4: iterative fallback if < 2 results ────────────────────
    if len(selected_refs) < 2:
        remaining = [c for c in clusters if c not in selected_clusters]
        remaining = sorted(
            remaining, key=lambda c: clusters[c]["count"], reverse=True
        )[:2]
        if remaining:
            fallback_refs = _select_from_clusters(remaining, limit=80)
            for ref in fallback_refs:
                if ref not in selected_refs:
                    selected_refs.append(ref)

    # ── Build result list ────────────────────────────────────────────
    results = []
    for rank, ref in enumerate(selected_refs[:TOP_K]):
        ticket = all_tickets.get(ref)
        if ticket:
            results.append({
                "rank":        rank + 1,
                "score":       round(1.0 / (rank + 1), 4),
                "reference":   ticket.get("reference", ""),
                "title":       ticket.get("title", ""),
                "version":     ticket.get("version", ""),
                "description": ticket.get("description", ""),
                "resolution":  ticket.get("resolution", ""),
                "patches":     ticket.get("patches", []),
                "source":      "pageindex",
                "cluster":     selected_clusters[0] if selected_clusters else "",
            })

    return results


# ─────────────────────────────────────────────
# 3. INTERACTIVE QUERY
# ─────────────────────────────────────────────

def interactive_query(
    index_dir: Path,
    team: str,
    question: Optional[str] = None,
):
    pageindex = load_pageindex(index_dir)

    # Load full ticket data — try JSON folder first, fall back to pageindex summaries
    all_tickets: dict = {}

    # Strategy 1: load from data/json/ folder (same parent as data/pageindex/)
    json_dir = index_dir.parent / "json"
    if json_dir.exists():
        for f in sorted(json_dir.glob("*.json")):
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
                if t.get("support_team") == team:
                    all_tickets[t["reference"]] = t
            except Exception:
                pass

    # Strategy 2: parse summaries stored in pageindex (title/version/resolution preview)
    if not all_tickets:
        for cluster in pageindex.get(team, {}).get("clusters", {}).values():
            for ref, summary in zip(cluster.get("refs", []), cluster.get("tickets", [])):
                parts = [p.strip() for p in summary.split(" | ")]
                title      = parts[1] if len(parts) > 1 else ""
                version    = parts[2] if len(parts) > 2 and parts[2].startswith("HR") else ""
                resolution = ""
                patches    = []
                for p in parts:
                    if p.startswith("→ "):
                        resolution = p[2:].strip()
                    if p.startswith("[P:"):
                        patches = [x.strip() for x in p[3:-1].split(",") if x.strip()]
                all_tickets[ref] = {
                    "reference":  ref,
                    "title":      title,
                    "version":    version,
                    "resolution": resolution,
                    "patches":    patches,
                }

    print(f"☁️  Connecting to Bedrock...")
    bedrock = get_bedrock_client()

    team_count = pageindex.get(team, {}).get("count", 0)
    print(f"✅ PageIndex loaded: {team_count} tickets across "
          f"{len(pageindex.get(team, {}).get('clusters', {}))} clusters")

    if question:
        hits       = retrieve(question, team, pageindex, bedrock, all_tickets)
        suggestion = generate_suggestion(question, hits, bedrock)
        print_results(question, team, hits, suggestion, "pageindex")
        return

    print(f"\n{'─'*70}")
    print(f"  Sopra HR — Vectorless RAG — Team {team}")
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

        hits       = retrieve(query, team, pageindex, bedrock, all_tickets)
        suggestion = generate_suggestion(query, hits, bedrock)
        print_results(query, team, hits, suggestion, "pageindex")


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────

def evaluate(
    index_dir: Path,
    json_dir: Path,
    team: str,
    n_samples: int = 50,
):
    """
    Self-retrieval evaluation: uses each ticket's description as a query
    and checks whether the correct ticket appears in the top-1/3/5 results.
    """
    import random
    random.seed(42)

    print(f"\n🎯 Evaluating Vectorless RAG — Team: {team} | Samples: {n_samples}\n")

    # Load tickets
    all_json = []
    for f in sorted(json_dir.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("support_team") == team and t.get("description") and t.get("is_closed"):
            all_json.append(t)

    sample    = random.sample(all_json, min(n_samples, len(all_json)))
    pageindex = load_pageindex(index_dir)
    all_tickets = {t["reference"]: t for t in all_json}
    bedrock   = get_bedrock_client()

    recall = {1: 0, 3: 0, 5: 0}

    for i, ticket in enumerate(sample, 1):
        query    = ticket["description"]
        expected = ticket["reference"]

        hits = retrieve(query, team, pageindex, bedrock, all_tickets)
        refs = [h["reference"] for h in hits]

        for k in [1, 3, 5]:
            if expected in refs[:k]:
                recall[k] += 1

        hit = "✅" if expected in refs[:3] else "❌"
        print(f"  [{i:3d}/{n_samples}] {hit} | {expected} | top: {refs[0] if refs else 'none'}")

        time.sleep(0.3)  # avoid Bedrock rate limits

    n = len(sample)
    print(f"\n{'═'*50}")
    print(f"  RÉSULTATS — Vectorless RAG — Team {team}")
    print(f"{'═'*50}")
    print(f"  Recall@1 : {recall[1]/n:.1%}  ({recall[1]}/{n})")
    print(f"  Recall@3 : {recall[3]/n:.1%}  ({recall[3]}/{n})")
    print(f"  Recall@5 : {recall[5]/n:.1%}  ({recall[5]}/{n})")
    print(f"{'═'*50}\n")

    out = index_dir / f"eval_vectorless_{team.lower()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "team": team, "mode": "vectorless", "n_samples": n,
            "recall": {str(k): round(v/n, 3) for k, v in recall.items()}
        }, f, indent=2)
    print(f"✅ Results saved → {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sopra HR Vectorless RAG (PageIndex)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vectorless_rag.py build --json data/json --index data/pageindex
  python vectorless_rag.py query --index data/pageindex --team DSN
  python vectorless_rag.py query --index data/pageindex --team DSN --question "Erreur ORA-00942"
  python vectorless_rag.py evaluate --index data/pageindex --json data/json --team DSN
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bld = sub.add_parser("build", help="Build PageIndex from JSON tickets")
    bld.add_argument("--json",  type=Path, required=True, help="Folder of JSON tickets")
    bld.add_argument("--index", type=Path, required=True, help="Output folder for PageIndex")

    qry = sub.add_parser("query", help="Interactive query")
    qry.add_argument("--index",    type=Path, required=True)
    qry.add_argument("--team",     type=str,  required=True, choices=TEAMS)
    qry.add_argument("--question", type=str,  default=None)

    evl = sub.add_parser("evaluate", help="Evaluate retrieval quality")
    evl.add_argument("--index",   type=Path, required=True)
    evl.add_argument("--json",    type=Path, required=True)
    evl.add_argument("--team",    type=str,  required=True, choices=TEAMS)
    evl.add_argument("--samples", type=int,  default=50)

    args = parser.parse_args()

    if args.command == "build":
        build_pageindex(args.json, args.index)
    elif args.command == "query":
        interactive_query(args.index, args.team, args.question)
    elif args.command == "evaluate":
        evaluate(args.index, args.json, args.team, args.samples)