#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/formatting.py
===================
Console output formatting for the Sopra HR support agent results.

Produces a structured, readable terminal display that mirrors the
existing print_results() style in rag_utils.py but operates on the
structured JSON dict produced by the ReAct agent.
"""

import textwrap
from typing import Any


# Terminal width used for separators
_WIDTH = 70


def _to_str(value: Any) -> str:
    """Safely coerces any field value to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # LLM sometimes returns {"fr": "...", "en": "..."} — pick first value
        return str(next(iter(value.values()), ""))
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _to_list(value: Any) -> list[str]:
    """Safely coerces any field value to a list of strings."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _wrap(text: str, indent: int = 4, width: int = _WIDTH) -> str:
    """Wraps a text block with a leading indent."""
    prefix = " " * indent
    lines  = []
    for paragraph in text.splitlines():
        if paragraph.strip():
            for wrapped in textwrap.wrap(paragraph, width=width - indent):
                lines.append(prefix + wrapped)
        else:
            lines.append("")
    return "\n".join(lines)


def pretty_print_result(
    question: str,
    team: str,
    result: dict[str, Any],
    warnings: list[str] | None = None,
) -> None:
    """
    Pretty-prints the agent JSON output to stdout.

    Parameters
    ----------
    question : the original user question
    team     : support team name (DSN / Appli / Outils)
    result   : dict from parse_agent_output() / run_query()
    warnings : list of validation warning strings (may be empty)
    """
    print(f"\n{'═' * _WIDTH}")
    print(f"  AGENT SOPRA HR  |  Équipe: {team}")
    print(f"  Question: {question[:60]}{'…' if len(question) > 60 else ''}")
    print(f"{'═' * _WIDTH}")

    if result.get("error"):
        print(f"\n  ⚠️  Erreur agent: {result['error']}")
        print(f"{'═' * _WIDTH}\n")
        return

    # ── Analyse ───────────────────────────────────────────────────────
    analyse = _to_str(result.get("analyse")).strip()
    if analyse:
        print(f"\n📋  ANALYSE:")
        print(_wrap(analyse))

    # ── Cause probable ────────────────────────────────────────────────
    cause = _to_str(result.get("cause_probable")).strip()
    if cause:
        print(f"\n🔍  CAUSE PROBABLE:")
        print(_wrap(cause))

    # ── Tickets utilisés ─────────────────────────────────────────────
    tickets = _to_list(result.get("tickets_utilises"))
    if tickets:
        print(f"\n🗂️   TICKETS UTILISÉS:")
        for ref in tickets:
            print(f"    • {ref}")

    # ── Résolution ────────────────────────────────────────────────────
    resolution = _to_str(result.get("resolution")).strip()
    if resolution:
        print(f"\n✅  RÉSOLUTION:")
        print(_wrap(resolution))

    # ── Patches ───────────────────────────────────────────────────────
    patches = _to_list(result.get("patches"))
    if patches:
        print(f"\n🔧  PATCHES:")
        print(f"    {', '.join(patches)}")

    # ── Réponse Lotus ─────────────────────────────────────────────────
    lotus = _to_str(result.get("reponse_lotus")).strip()
    if lotus:
        print(f"\n💬  RÉPONSE SUPPORT (style Lotus):")
        print(f"  {'─' * (_WIDTH - 4)}")
        print(_wrap(lotus))
        print(f"  {'─' * (_WIDTH - 4)}")

    # ── Warnings ──────────────────────────────────────────────────────
    if warnings:
        print(f"\n⚠️   AVERTISSEMENTS:")
        for w in warnings:
            print(f"    ! {w}")

    print(f"\n{'═' * _WIDTH}\n")


def format_json_output(result: dict[str, Any], indent: int = 2) -> str:
    """Returns a JSON string of the result dict with consistent formatting."""
    import json
    return json.dumps(result, ensure_ascii=False, indent=indent)
