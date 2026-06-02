#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_tools.py
==============
RAG retriever functions and tool registry for the manual ReAct agent.

No LangChain dependency — tools are plain Python callables registered
in a dict so the ReAct loop can dispatch them by name.

Exports:
  - build_tool_registry()  → dict[str, callable]
  - TOOL_DESCRIPTIONS      — human-readable list injected into the prompt

Retriever functions:
  - search_vector(query)      — FAISS cosine similarity
  - search_bm25(query)        — BM25 keyword matching
  - search_pageindex(query)   — PageIndex cluster navigation
  - search_hybrid(query)      — 3-way RRF fusion
"""

import sys
import pickle
from pathlib import Path
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

# Add scripts/ to path so relative imports work when called from outside
_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rag_utils import EMBEDDING_MODEL, TOP_K, RRF_K, RETRIEVAL_K
from bm25_rag import load_bm25_index, bm25_retrieve
from vectorless_rag import load_pageindex, retrieve as pageindex_retrieve
from hybrid_rag import (
    load_vector_index,
    vector_retrieve,
    reciprocal_rank_fusion,
)


# ─────────────────────────────────────────────
# FORMATTING HELPER
# ─────────────────────────────────────────────

def format_results(hits: list[dict], max_hits: int = 7) -> str:
    """
    Converts a list of retrieval hit dicts into a plain-text block
    suitable for injecting into a LangChain ReAct Observation.

    Top 2 hits get the FULL resolution text so the LLM can synthesize
    a proper answer. Remaining hits are truncated to save context.
    """
    if not hits:
        return "Aucun ticket pertinent trouvé."

    lines: list[str] = ["Tickets trouvés (utilise ces références EXACTES dans tickets_utilises) :"]
    for h in hits[:max_hits]:
        rank   = h.get("rank", "?")
        ref    = h.get("reference", "")
        title  = h.get("title", "")
        full_res = (h.get("resolution") or "").strip()
        patches = ", ".join(h.get("patches") or [])
        last_date = h.get("last_date", "")
        score  = (
            h.get("rrf_score")
            or h.get("similarity")
            or h.get("score")
            or 0.0
        )

        # Top 2 hits: show full resolution; rest: truncated snippet
        if rank in (1, 2, "1", "2"):
            res = full_res.replace("\n", " ")
        else:
            res = full_res[:350].replace("\n", " ")
            if len(full_res) > 350:
                res += "…"

        line = f"#{rank} [ref: {ref}] {title}"
        if score:
            line += f"  (score={score:.4f})"
        if last_date:
            line += f"  [date: {last_date[:10]}]"
        espdsn = h.get("espdsn_version", "")
        if espdsn:
            line += f"  [ESPDSN: {espdsn}]"
        lines.append(line)
        if res:
            lines.append(f"   Résolution: {res}")
        if patches:
            lines.append(f"   Patches: {patches}")
        lines.append("")

    # Summary line listing all refs for easy copying
    all_refs = [h.get("reference", "") for h in hits[:max_hits] if h.get("reference")]
    if all_refs:
        lines.append(f"Références disponibles: {', '.join(all_refs)}")

    return "\n".join(lines).rstrip()


# ─────────────────────────────────────────────
# INDEX HOLDER
# Populated once by build_tools(); referenced by closures below.
# ─────────────────────────────────────────────

class _IndexStore:
    """Container for all loaded index objects."""
    embed_model: SentenceTransformer | None = None
    vector_index: Any = None
    vector_docstore: list = []
    bm25_index: Any = None
    bm25_docs: list = []
    pageindex: dict = {}
    all_tickets: dict = {}
    team: str = ""
    bedrock_client: Any = None
    # Buffer holding the last search's full hit list (most recent first).
    # Used by the agent to deterministically extract rank-1 resolution/patches.
    last_hits: list = []


_store = _IndexStore()


def get_last_top_hit() -> dict | None:
    """Returns the rank-1 ticket from the most recent hybrid search, or None."""
    return _store.last_hits[0] if _store.last_hits else None


# ─────────────────────────────────────────────
# RETRIEVER FUNCTIONS
# ─────────────────────────────────────────────

def search_vector(query: str) -> str:
    """Vector RAG: semantic FAISS cosine-similarity search."""
    hits = vector_retrieve(
        query,
        _store.vector_index,
        _store.vector_docstore,
        _store.embed_model,
        top_k=TOP_K,
    )
    return format_results(hits)


def search_bm25(query: str) -> str:
    """BM25 RAG: exact keyword / error-code / patch-number search."""
    hits = bm25_retrieve(query, _store.bm25_index, _store.bm25_docs, top_k=TOP_K)
    return format_results(hits)


def search_pageindex(query: str) -> str:
    """PageIndex RAG: LLM-guided 3-level cluster navigation (no vectors)."""
    hits = pageindex_retrieve(
        query,
        _store.team,
        _store.pageindex,
        _store.bedrock_client,
        _store.all_tickets,
    )
    return format_results(hits)


def search_hybrid(query: str) -> str:
    """Hybrid RAG: deep retrieval per source → RRF fusion.

    Pipeline
    --------
    1. Vector + BM25 + PageIndex each return RETRIEVAL_K candidates.
    2. Secondary BM25 pass on extracted technical codes (ORA-*, ZY*, FSW*, …)
       to ensure exact-match recall.
    3. RRF fuses and top TOP_K returned to the agent.
    """
    import re as _re

    expanded_k = RETRIEVAL_K
    vec_hits  = vector_retrieve(
        query,
        _store.vector_index,
        _store.vector_docstore,
        _store.embed_model,
        top_k=expanded_k,
    )
    bm25_hits = bm25_retrieve(query, _store.bm25_index, _store.bm25_docs, top_k=expanded_k)

    # Secondary BM25: extract technical codes from user query and search them separately
    # This catches exact error codes / patches that get diluted in the full-text query
    _TECH_RE = _re.compile(
        r'\b(ORA-\d+|[A-Z]{2,5}[-_][A-Z0-9]{3,}|Z[XY]\w{3,}|FSW\w{3,}|'
        r'REGDSN|HRCT|ESPDSN|HRQUERY|HRASPACE|DADSU?|N4DS|'
        r'patch\s*\d{5,6}|\d{5,6})\b',
        _re.IGNORECASE
    )
    tech_terms = _TECH_RE.findall(query)
    if tech_terms:
        tech_query = " ".join(tech_terms)
        tech_bm25 = bm25_retrieve(tech_query, _store.bm25_index, _store.bm25_docs, top_k=expanded_k)
        # Merge: existing BM25 hits take priority, then tech hits fill remaining slots
        existing_refs = {h['reference'] for h in bm25_hits}
        for h in tech_bm25:
            if h['reference'] not in existing_refs:
                bm25_hits.append(h)
                existing_refs.add(h['reference'])

    pi_hits   = pageindex_retrieve(
        query,
        _store.team,
        _store.pageindex,
        _store.bedrock_client,
        _store.all_tickets,
    )
    fused = reciprocal_rank_fusion(
        vec_hits, pi_hits, bm25_hits, k=RRF_K, top_k=TOP_K
    )

    fused = fused[:TOP_K]
    # Stash for the agent to access rank-1 deterministically
    _store.last_hits = fused
    return format_results(fused)


def get_ticket_details(ref_input: str) -> str:
    """
    DétailsTicket tool: returns the full resolution, description,
    patches, and conversation excerpts for one ticket by reference.

    The agent calls this after RechercheHybride to get the complete
    resolution text before composing its answer.
    """
    import re as _re

    # Extract a clean reference like "FR W210000" from the input
    m = _re.search(r'((?:FR|AF|SP|DE|UK|BE|IT|NL)\s*W\d{5,})', ref_input.strip(), _re.IGNORECASE)
    if not m:
        return f"Référence invalide: '{ref_input}'. Format attendu: FR WXXXXXX"
    ref = m.group(1).upper()
    # Normalise spacing: "FRW210000" → "FR W210000"
    if " " not in ref:
        ref = ref[:2] + " " + ref[2:]

    # Look up in docstore (already loaded in memory)
    ticket = _store.all_tickets.get(ref)
    if not ticket:
        # Try last_hits as fallback
        for h in _store.last_hits:
            if h.get("reference", "").upper() == ref:
                ticket = h
                break
    if not ticket:
        return f"Ticket {ref} non trouvé dans l'index."

    title       = ticket.get("title", "")
    description = (ticket.get("description") or "").strip()
    resolution  = (ticket.get("resolution") or "").strip()
    patches     = ticket.get("patches") or []
    version     = ticket.get("version", "")

    lines = [
        f"=== Détails du ticket {ref} ===",
        f"Titre: {title}",
    ]
    if version:
        lines.append(f"Version: {version}")
    lines.append("")

    if description:
        lines.append("── Description complète ──")
        lines.append(description[:1000])
        lines.append("")

    if resolution:
        lines.append("── Résolution complète ──")
        lines.append(resolution)
        lines.append("")

    if patches:
        lines.append(f"── Patches: {', '.join(patches)} ──")
        lines.append("")

    return "\n".join(lines).rstrip()


# ─────────────────────────────────────────────
# PUBLIC FACTORY
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# TOOL DESCRIPTIONS  (injected into ReAct prompt)
# ─────────────────────────────────────────────

TOOL_DESCRIPTIONS = """\
- RechercheHybride     : Recherche combinée (vectorielle + BM25 + PageIndex). \
Utilise cet outil pour TOUTES les recherches. Il combine recherche sémantique, \
recherche par mots-clés exacts et navigation par clusters pour un maximum de rappel.
- DétailsTicket        : Récupère le détail complet d'un ticket (description, résolution \
complète, patches, conversation). Utilise cet outil APRÈS RechercheHybride pour obtenir \
la résolution complète du ticket #1 avant de rédiger ta réponse. Entrée: la référence \
exacte du ticket (ex: FR W210000)."""


def build_tool_registry(
    team: str,
    db_dir: Path,
    bm25_dir: Path,
    index_dir: Path,
    bedrock_client: Any,
    embed_model: SentenceTransformer | None = None,
) -> dict[str, Any]:
    """
    Loads all indexes into the shared _store and returns a dict
    mapping tool names to callables.

    Parameters
    ----------
    team          : "DSN" | "Appli" | "Outils"
    db_dir        : directory containing FAISS sub-folders per team
    bm25_dir      : directory containing BM25 pickle files per team
    index_dir     : directory containing the PageIndex JSON
    bedrock_client: boto3 Bedrock runtime client (for PageIndex calls)
    embed_model   : pre-loaded SentenceTransformer (loaded here if None)
    """
    _store.team = team
    _store.bedrock_client = bedrock_client

    if embed_model is None:
        # Use the model recorded in config.json at build time to avoid
        # dimension mismatches when EMBEDDING_MODEL was changed after indexing.
        import json as _json
        config_path = db_dir / team.lower() / "config.json"
        if config_path.exists():
            index_model = _json.loads(config_path.read_text()).get("model", EMBEDDING_MODEL)
        else:
            index_model = EMBEDDING_MODEL
        if index_model != EMBEDDING_MODEL:
            print(f"[INFO] Index built with '{index_model}', loading that model (not '{EMBEDDING_MODEL}')")
        embed_model = SentenceTransformer(index_model)
    _store.embed_model = embed_model

    vi, vd = load_vector_index(db_dir, team)
    _store.vector_index    = vi
    _store.vector_docstore = vd

    bi, bd = load_bm25_index(bm25_dir, team)
    _store.bm25_index = bi
    _store.bm25_docs  = bd

    _store.pageindex = load_pageindex(index_dir)
    _store.all_tickets = {doc["reference"]: doc for doc in vd}

    return {
        "RechercheHybride":     search_hybrid,
        "DétailsTicket":        get_ticket_details,
    }
