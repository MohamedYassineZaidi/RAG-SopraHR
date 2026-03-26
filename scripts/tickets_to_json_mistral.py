#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Config
# =========================
MODEL = "mistralai/mixtral-8x7b-instruct"
TEMP_JSON = 0.1
TOK_JSON = 1500
RATE_SLEEP = 0.3


# =========================
# Helpers — Team extraction
# =========================

TEAM_KEYWORDS = {
    "APPLI": r"\b(appli|application|pap[a-z0-9]+)\b",
    "DSN": r"\b(dsn)\b",
    "OUTILS": r"\b(outils?)\b",
}

def extract_team(text: str) -> str:
    """
    Extract responsible team from header:
    Example: 'support_team: Appli'
    Fallback to keyword detection.
    """
    lower = text.lower()

    # 1) Metadata block detection
    m = re.search(r"support_team:\s*([A-Za-z0-9]+)", lower)
    if m:
        return m.group(1).upper()

    # 2) Keyword fallback
    for team, pattern in TEAM_KEYWORDS.items():
        if re.search(pattern, lower):
            return team

    return ""


# =========================
# Prompts
# =========================

SYSTEM_PROMPT = """
You are a senior Sopra HR incident analyst producing enterprise post-mortem knowledge.
Return ONLY valid JSON. No commentary.
"""

JSON_USER_PROMPT = """
Analyze this Sopra HR support ticket carefully and return ONLY valid JSON.

Ticket:
{TICKET_TEXT}

Reference: {REF}

Return structure:
{{
  "reference": "",
  "facts": {{
    "client": "",
    "version": "",
    "systeme": "",
    "ouverture": "",
    "fermeture": "",
    "duree": ""
  }},
  "equipe_responsable": "",
  "probleme": {{
    "titre": "",
    "description": "",
    "gravite": "",
    "symptomes": []
  }},
  "cause_racine": {{
    "explication": "",
    "type_incident": "Configuration|Données|Utilisation incorrecte|Bug produit",
    "composants_affectes": []
  }},
 
  "chronologie": [
    {{
      "date": "",
      "action": "",
      "description": "",
      "hypothese": "",
      "resultat": ""
    }}
  ],
  "resolution": {{
    "statut": "",
    "code": "",
    "verdict": "",
    "responsabilite": "support_team",
    "actions_realisees": ""
  }},
  "impact_metier": "",
  "mots_cles": []
}}

Extraction rules:
- If the header contains "support_team: XXX", ALWAYS set "equipe_responsable" = XXX.
- If not found, infer the responsible team from the ticket content (Appli, DSN, Outils).
- Chronologie:
  - hypothese = why the action was taken / expected behavior
  - resultat = what happened (customer response, errors, test results)
  
- Return ONLY JSON. Never return markdown.
"""


# =========================
# Client
# =========================

def load_client() -> OpenAI:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def call_llm(client: OpenAI, system: str, user: str, temp: float, max_tokens: int) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


# =========================
# JSON extraction
# =========================

def generate_json(client: OpenAI, ticket_text: str, ref: str) -> Dict[str, Any]:
    user = JSON_USER_PROMPT.format(TICKET_TEXT=ticket_text, REF=ref)
    raw = call_llm(client, SYSTEM_PROMPT, user, TEMP_JSON, TOK_JSON)

    # Clean fences
    raw_clean = raw.strip()
    raw_clean = re.sub(r"^```json\s*|\s*```$", "", raw_clean, flags=re.MULTILINE).strip()

    # Try direct load
    try:
        return json.loads(raw_clean)
    except Exception as e:
        # Attempt repair
        raw_clean = raw_clean.replace("\u00a0", " ")
        raw_clean = raw_clean.encode("utf-8", errors="ignore").decode("utf-8")
        raw_clean = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_clean)
        raw_clean = raw_clean.replace("\n", " ").replace("\r", " ")

        for i in range(len(raw_clean), max(len(raw_clean)//2, 100), -50):
            try:
                test = raw_clean[:i]
                oc = test.count("{")
                cc = test.count("}")
                if oc > cc:
                    test += "}" * (oc - cc)
                ob = test.count("[")
                cb = test.count("]")
                if ob > cb:
                    test += "]" * (ob - cb)
                return json.loads(test)
            except:
                pass

        # Minimal fallback
        return {
            "reference": ref,
            "facts": {},
            "equipe_responsable": "",
            "probleme": {"titre": "", "description": "", "gravite": "", "symptomes": []},
            "cause_racine": {"explication": "", "type_incident": "Configuration", "composants_affectes": []},
            "chronologie": [],
            "pourquoi_probleme": "",
            "resolution": {"statut": "", "code": "", "verdict": "", "responsabilite": "", "actions_realisees": ""},
            "impact_metier": "",
            "mots_cles": [],
            "_error": f"JSON parse failed: {str(e)[:100]}"
        }


# =========================
# Formatting helpers
# =========================

def normalize_keywords(keywords: List[str]) -> List[str]:
    out = []
    seen = set()
    for k in keywords:
        k2 = re.sub(r"\s+", " ", k).strip()
        if k2 and k2.lower() not in seen:
            seen.add(k2.lower())
            out.append(k2)
    return out

def normalize_chronology_to_bullets(chron: List[Dict[str,str]]) -> List[str]:
    out=[]
    for step in chron:
        d = step.get("date","")
        a = step.get("action","")
        h = step.get("hypothese","")
        r = step.get("resultat","")
        out.append(f"- {d} – {a} – {h} – {r}")
    return out



# =========================
# Pipeline
# =========================

def process_ticket(client: OpenAI, src: Path, out_dir: Path):
    ref = re.sub(r"[^A-Za-z0-9_-]", "_", src.stem)
    txt = src.read_text(encoding="utf-8", errors="ignore")

    # Extract team before LLM
    detected_team = extract_team(txt)

    logging.info(f"[JSON] {src.name}")
    data = generate_json(client, txt, ref)

    # Guarantee team presence
    if not data.get("equipe_responsable"):
        data["equipe_responsable"] = detected_team

    # Metadata
    data["_metadata"] = {
        "generated_at": dt.datetime.now().isoformat(),
        "model": MODEL,
        "source_file": src.name
    }

    out = out_dir / f"{ref}_analysis.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"[OK] {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )

        
    client = load_client()
    args.output.mkdir(parents=True, exist_ok=True)

    for src in sorted(args.input.rglob("*.txt")):
        try:
            process_ticket(client, src, args.output)
        except Exception as e:
            logging.error(f"[ERROR] {src.name}: {e}", exc_info=True)
        time.sleep(RATE_SLEEP)

    logging.info("Done.")


if __name__ == "__main__":
    main()