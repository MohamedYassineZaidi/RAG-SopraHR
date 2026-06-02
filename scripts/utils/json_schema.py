#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/json_schema.py
====================
JSON Schema definition and validator for the Sopra HR agent output.

Schema
------
{
  "analyse":          string  — problem summary (required, non-empty)
  "tickets_utilises": array   — list of ticket reference strings
  "cause_probable":   string  — technical root cause
  "resolution":       string  — resolution steps
  "patches":          array   — list of patch identifiers (ZYxxx)
  "reponse_lotus":    string  — professional client-facing reply
}

All fields are required. Empty strings / empty arrays are allowed
when the agent has insufficient information, but the keys must be
present.
"""

from typing import Any

# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

SOPRA_HR_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SopraHROutput",
    "type": "object",
    "required": [
        "analyse",
        "tickets_utilises",
        "cause_probable",
        "resolution",
        "patches",
        "reponse_lotus",
    ],
    "additionalProperties": True,   # allow "error" field from parse failures
    "properties": {
        "analyse": {
            "type": "string",
            "description": "One-to-two sentence problem summary.",
        },
        "tickets_utilises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ticket references used to build the answer.",
        },
        "cause_probable": {
            "type": "string",
            "description": "Technical root cause of the problem.",
        },
        "resolution": {
            "type": "string",
            "description": "Step-by-step resolution instructions.",
        },
        "patches": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of patch identifiers (ZYxxx, ZXxxx …).",
        },
        "reponse_lotus": {
            "type": "string",
            "description": "Professional Lotus-Notes-style reply to the client.",
        },
    },
}


# ─────────────────────────────────────────────
# RUNTIME VALIDATOR
# ─────────────────────────────────────────────

def validate_schema(data: dict[str, Any]) -> list[str]:
    """
    Validates *data* against SOPRA_HR_OUTPUT_SCHEMA.

    Uses jsonschema if available; falls back to a manual field check
    so the module works even when jsonschema is not installed.

    Returns
    -------
    list[str]
        List of validation error messages.  Empty list = valid.
    """
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(SOPRA_HR_OUTPUT_SCHEMA)
        errors = [e.message for e in validator.iter_errors(data)]
        return errors
    except ImportError:
        pass

    # Fallback manual validation
    errors: list[str] = []
    required = SOPRA_HR_OUTPUT_SCHEMA["required"]
    props    = SOPRA_HR_OUTPUT_SCHEMA["properties"]

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue
        expected_type = props[field]["type"]
        value         = data[field]
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field}' must be a string, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field '{field}' must be a list, got {type(value).__name__}")

    return errors
