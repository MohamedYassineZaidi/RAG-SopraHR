#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/validators.py
===================
Semantic sanity checks for the Sopra HR agent output.

These checks go beyond JSON Schema validation and catch common
hallucination patterns:

  - Ticket references in "tickets_utilises" that look fabricated
    (wrong format, numeric-only, etc.)
  - Resolution that is suspiciously short or generic
  - Patches listed in "patches" but not mentioned in "resolution"
  - "analyse" or "resolution" that echo the question verbatim
    without adding information (copy-paste hallucination)

None of these is a hard error — they are returned as warning strings
so the CLI can display them to the user for manual review.
"""

import re
from typing import Any

from utils.json_schema import validate_schema


# ─────────────────────────────────────────────
# REFERENCE FORMAT RULES
# ─────────────────────────────────────────────

# Sopra HR ticket references follow patterns like:
#   FR_W210001   FRW210001   0001_FR_W210001
_REF_PATTERN = re.compile(
    r"^(?:\d{4}_)?FR[_\s]?W\d{5,}$",
    re.IGNORECASE,
)

# Patch references: ZYxxx, ZXxxx, ZDAGxxx, HRCTxxx …
_PATCH_PATTERN = re.compile(
    r"^(ZY|ZX|ZDAG|HRCT|NRB|BAY|HAD|KFN)[A-Z0-9_\-]+$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# INDIVIDUAL CHECKS
# ─────────────────────────────────────────────

def _check_ticket_refs(data: dict[str, Any]) -> list[str]:
    """Warns when a ticket reference doesn't match the expected pattern."""
    warnings: list[str] = []
    for ref in data.get("tickets_utilises") or []:
        if not isinstance(ref, str):
            warnings.append(f"tickets_utilises: non-string entry '{ref}'")
        elif not _REF_PATTERN.match(ref.strip()):
            warnings.append(
                f"tickets_utilises: '{ref}' has unexpected format "
                f"(expected e.g. FR_W210001 or FRW210001)"
            )
    return warnings


def _check_patch_refs(data: dict[str, Any]) -> list[str]:
    """Warns when a patch identifier looks fabricated."""
    warnings: list[str] = []
    for patch in data.get("patches") or []:
        if not isinstance(patch, str):
            warnings.append(f"patches: non-string entry '{patch}'")
        elif not _PATCH_PATTERN.match(patch.strip()):
            warnings.append(
                f"patches: '{patch}' has unexpected format "
                f"(expected e.g. ZY1234 or HRCT01)"
            )
    return warnings


def _check_resolution_quality(data: dict[str, Any]) -> list[str]:
    """Warns when the resolution is suspiciously short or generic."""
    warnings: list[str] = []
    resolution = (data.get("resolution") or "").strip()

    if not resolution:
        warnings.append("resolution: field is empty — agent may have found no relevant tickets")
        return warnings

    if len(resolution) < 30:
        warnings.append(
            f"resolution: very short ({len(resolution)} chars) — "
            f"may be incomplete"
        )

    generic_phrases = [
        "contactez le support",
        "veuillez vérifier",
        "je ne sais pas",
        "aucune information",
        "non disponible",
    ]
    lower = resolution.lower()
    for phrase in generic_phrases:
        if phrase in lower:
            warnings.append(
                f"resolution: contains generic phrase '{phrase}' — "
                f"agent may not have found a specific answer"
            )
    return warnings


def _check_patches_in_resolution(data: dict[str, Any]) -> list[str]:
    """Warns when patches are listed but not referenced in the resolution text."""
    warnings: list[str] = []
    patches    = data.get("patches") or []
    resolution = (data.get("resolution") or "").lower()

    for patch in patches:
        if isinstance(patch, str) and patch.lower() not in resolution:
            warnings.append(
                f"patches: '{patch}' listed but not mentioned in resolution — "
                f"verify this patch actually applies"
            )
    return warnings


def _check_no_tickets_used(data: dict[str, Any]) -> list[str]:
    """Warns when the agent produced an answer without citing any ticket."""
    tickets = data.get("tickets_utilises") or []
    if not tickets and not data.get("error"):
        return [
            "tickets_utilises: empty — agent answered without citing any ticket. "
            "This may indicate hallucination."
        ]
    return []


# ─────────────────────────────────────────────
# PUBLIC VALIDATOR
# ─────────────────────────────────────────────

def validate_agent_output(data: dict[str, Any]) -> list[str]:
    """
    Runs all schema and semantic checks on the agent output dict.

    Parameters
    ----------
    data : dict returned by parse_agent_output() / run_query()

    Returns
    -------
    list[str]
        All warnings and errors found.  Empty list = clean output.
    """
    issues: list[str] = []

    # 1. JSON Schema structural check
    issues.extend(validate_schema(data))

    # 2. Skip semantic checks if there was a parse error
    if data.get("error"):
        issues.append(f"parse_error: {data['error']}")
        return issues

    # 3. Semantic / hallucination checks
    issues.extend(_check_no_tickets_used(data))
    issues.extend(_check_ticket_refs(data))
    issues.extend(_check_patch_refs(data))
    issues.extend(_check_resolution_quality(data))
    issues.extend(_check_patches_in_resolution(data))

    return issues
