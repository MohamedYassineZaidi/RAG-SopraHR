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

import boto3
import tiktoken
from dotenv import load_dotenv

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
REGION   = os.getenv("AWS_REGION", "eu-central-1")

TEMP       = 0.0          # 0.0 for maximum determinism on extraction tasks
MAX_TOKENS = 25000
SLEEP      = 0.2

MODEL_PRICES = {
    MODEL_ID: {
        "input_per_1k":  0.015,
        "output_per_1k": 0.075,
    }
}

# ==========================================================
# TOKEN COUNTER
# ==========================================================

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("gpt2")
    return len(enc.encode(text))

# ==========================================================
# TEAM EXTRACTION (pre-flight — Claude overrides this)
# ==========================================================

def extract_team(text: str) -> str:
    low = text.lower()
    m = re.search(r"support_team:\s*([A-Za-z0-9]+)", low)
    if m:
        return m.group(1).upper()
    for k, pat in [("DSN", r"\bdsn\b"), ("Outils", r"\boutils?\b"), ("Appli", r"\bappli\b")]:
        if re.search(pat, low):
            return k
    return ""

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """You are a Senior Sopra HR Incident Analyst and a rigorous information extractor.
Your only job is to extract a complete, accurate JSON analysis from a Sopra HR support ticket.
You return ONLY valid JSON — no preamble, no commentary, no markdown fences."""

# ==========================================================
# USER PROMPT
# All {{PLACEHOLDERS}} are replaced before sending.
# ==========================================================

JSON_USER_PROMPT = """Analyze the following Sopra HR support ticket and return a single JSON object.

========================================
TICKET
========================================
{{TICKET_TEXT}}

Reference: {{REF}}

========================================
PHASE 1 — SILENT INTERNAL ANALYSIS (DO NOT OUTPUT THIS PHASE)
========================================
Before writing any JSON, perform the following analysis entirely in your head:

TIMELINE RECONSTRUCTION
- List every timestamp in the ticket in strict chronological order.
- For each timestamp, identify: who acted (client/support/system), what they did, what the result was.
- NEVER merge two separate timestamps into one chronology entry.
- NEVER skip a timestamp.

TICKET STRUCTURE (specific to Sopra HR tickets)
- The ticket has a HEADER (reference, version, system, support_team).
- The CONVERSATION block contains the full exchange.
- Client lines start with "Client :" and a timestamp.
- Support lines start with a timestamp, agent name, and optionally "H2 TEAMCODE MODULE:CODE".
- "Reply :" sections contain the actual support message text.
- "Status XX" lines indicate status changes (AK=In Progress, CI=Waiting, CP=Closed, AA=To process).
- "Archived" lines indicate the end of a message block.

RESOLUTION IDENTIFICATION (CRITICAL)
- The RESOLUTION is the FINAL technical answer that actually resolves the problem.
- If the ticket has MULTIPLE Reply blocks, the LAST one before final closure is usually the real resolution.
- A "kit delivery" reply (sending download links) is NOT the resolution if a later reply corrects a bug.
- Look for the message that identifies the root cause AND provides a concrete fix.
- If the last substantive support message corrects a documentation error or delivers a corrective patch,
  THAT is the resolution — not an earlier kit delivery.

PATCHES (CRITICAL — COLLECT ALL)
- Patches appear in multiple forms:
  1) Explicit: "patch 181660", "Patches : 181660"
  2) In kit lists: "X09P\n178105; 173491; ..." — each number after X09P/X09V/X10P/X10V is a patch number.
  3) In patch delivery messages with job numbers (JOB 546, JOB 547...).
- You MUST collect ALL patch numbers from ALL reply blocks, not just the final one.
- List them all in the "patches" array.

PREREQUISITES
- A prerequisite is any kit or version that must be installed BEFORE the main kit.
- Phrases: "prérequis à l'installation de", "à installer avant", "doit être installé en premier".
- In this ticket type: "V01X09 est prérequis à V01X10" → prerequisite = V01X09 kit.

DOCUMENTATION ERRORS
- A documentation error is any mistake found in an installation guide, technical guide, or documentation.
- Phrases: "le guide demande de créer X, or X existe déjà", "remplacer X par Y dans le guide",
  "nous avons mis à jour le guide", "erreur dans le guide d'installation".
- Extract the exact error: wrong value → correct value, and the document name if given.

CLIENT COMPANY
- The client line in the conversation header is: "FR*COMPANY_NAME*00:CDXXXXX"
- Extract the company name from between the first and second asterisk.
- Example: "FR*SECHE-ENVIRT*00:CD10433" → company = "Séché Environnement" (or "SECHE-ENVIRT" if unsure).
- Store this in facts.societe.

CLOSURE DETECTION
- "facts.fermeture" = the LAST timestamp where the client explicitly says they want to close.
  Phrases: "je souhaite fermer le dossier", "je souhaite clôturer", "vous pouvez fermer",
  "merci de clôturer", "je confirme la fermeture", "ticket à fermer", or equivalent.
- If the client closes TWICE (ticket was re-opened), use the LAST client closure timestamp.
- If no client closure: use the last support Status CP/CU/CO/Closed line.

CAUSE RACINE
- Some tickets have TWO distinct root causes (e.g. kit delivery + documentation bug).
- If so, list both in "cause_racine.explication" and set "cause_racine.type_incident" to the
  most significant one (the one that required a patch or technical fix).

SEVERITY
- Nice to have (low) / Serious / Very Important / Blocking bug
- Infer from: does it block production? Does the client say it's urgent? Is it a minor question?

========================================
PHASE 2 — OUTPUT JSON (MANDATORY)
========================================
Return ONE valid JSON object with this EXACT structure.
Fill every field with maximum detail from the ticket.

STRICT RULES:
- Return ONLY the JSON object. No text before or after.
- Arrays MUST NEVER be empty — use ["non_specifie"] if truly nothing found.
- Strings MUST NEVER be empty — use "non_specifie".
- "equipe_responsable" MUST match the header "support_team:" value exactly.
- Every timestamp in the ticket MUST appear as a separate chronology entry.
- "patches" MUST contain ALL patch numbers from ALL reply blocks combined.
- "prerequis" MUST list any kit/version that must be installed before the main deliverable.
- "erreurs_documentation" MUST list any mistake found in guides or documentation.
- The JSON must be parseable — no trailing commas, no comments.

{
  "reference": "",
  "facts": {
    "societe": "",
    "client": "",
    "version": "",
    "systeme": "",
    "ouverture": "",
    "fermeture": "",
    "duree": ""
  },
  "equipe_responsable": "",
  "probleme": {
    "titre": "",
    "description": "",
    "gravite": "",
    "symptomes": []
  },
  "cause_racine": {
    "explication": "",
    "type_incident": "",
    "composants_affectes": []
  },
  "chronologie": [
    {
      "date": "",
      "acteur": "",
      "action": "",
      "description": "",
      "resultat": ""
    }
  ],
  "resolution": {
    "statut": "",
    "code": "",
    "verdict": "",
    "responsabilite": "",
    "actions_realisees": []
  },
  "patches": [],
  "prerequis": [],
  "erreurs_documentation": [
    {
      "document": "",
      "erreur": "",
      "correction": ""
    }
  ],
  "impact_metier": "",
  "mots_cles": []
}

CHRONOLOGIE RULES — read carefully:
- "acteur": "client" / "support" / "système"
- "action": What was actually done — name the technical action specifically.
  BAD: "analyse du problème"
  GOOD: "Livraison des kits DADS-U V01X09 (prérequis) et V01X10 (production + visualisation) avec liens de téléchargement"
  BAD: "demande d'informations"
  GOOD: "Demande au client de fournir les logs REGDSN depuis edsn-home/logs et résultat de dsn:list-ud"
- "description": Full technical detail of the message — include all patch numbers, error codes,
  field names, table names, kit names, URLs, job numbers, prerequisite instructions.
- "resultat": What changed as a direct consequence of this action.

RESOLUTION RULES:
- "resolution.actions_realisees": list each distinct action taken to resolve (array of strings).
- "resolution.verdict": one-line human summary of how the ticket was resolved.
"""

# ==========================================================
# BEDROCK CLIENT
# ==========================================================

def load_client():
    load_dotenv()
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

# ==========================================================
# CLAUDE INVOCATION
# ==========================================================

def call_claude(client, system_prompt: str, user_prompt: str):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
    }

    input_tokens = count_tokens(system_prompt + "\n" + user_prompt)

    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    data      = json.loads(response["body"].read().decode("utf-8"))
    completion = data["content"][0]["text"]

    output_tokens = count_tokens(completion)
    price         = MODEL_PRICES[MODEL_ID]
    cost_in       = (input_tokens  / 1000) * price["input_per_1k"]
    cost_out      = (output_tokens / 1000) * price["output_per_1k"]

    usage = {
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "cost_input_usd":  round(cost_in,  6),
        "cost_output_usd": round(cost_out, 6),
        "total_cost_usd":  round(cost_in + cost_out, 6),
    }

    return completion, usage

# ==========================================================
# JSON GENERATION
# ==========================================================

def generate_json(client, ticket_text: str, ref: str) -> dict:
    user_prompt = (
        JSON_USER_PROMPT
        .replace("{{TICKET_TEXT}}", ticket_text)
        .replace("{{REF}}", ref)
    )

    raw, usage = call_claude(client, SYSTEM_PROMPT, user_prompt)

    # Strip markdown fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$",     "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        data["_usage"] = usage
        return data
    except json.JSONDecodeError as e:
        logging.warning(f"  JSON parse failure for {ref}: {e}")
        # Attempt to recover by finding the outermost { } block
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                data["_usage"] = usage
                logging.warning(f"  Recovered via regex extraction")
                return data
            except Exception:
                pass
        return {
            "reference": ref,
            "error":     "JSON parse failure",
            "raw":       raw,
            "_usage":    usage,
        }

# ==========================================================
# POST-PROCESSING — patch up common Claude omissions
# ==========================================================

def postprocess(data: dict, ticket_text: str, fallback_team: str) -> dict:
    """
    Applies deterministic fixes on top of Claude's output:
    - Ensures equipe_responsable is set from header if Claude missed it.
    - Ensures patches extracted by our regex are merged into Claude's list.
    - Normalises the patches array (deduplicate, sort numerically).
    """

    # 1. Team fallback
    if not data.get("equipe_responsable") or data["equipe_responsable"] == "non_specifie":
        data["equipe_responsable"] = fallback_team

    # 2. Patch extraction — regex sweep over raw ticket text
    #    Catches: "patch 181660", "Patches : 181660", and kit list numbers
    patch_re = re.compile(
        r'(?:patch(?:es)?\s*:?\s*|(?<=\n))(\d{6})',
        re.IGNORECASE
    )
    found_patches = set(patch_re.findall(ticket_text))

    # Also catch numbers in kit list lines (X09P\n178105; 173491; ...)
    kit_line_re = re.compile(r'X\d{2}[PV]\s*\n([\d\s;]+)', re.IGNORECASE)
    for match in kit_line_re.finditer(ticket_text):
        for num in re.findall(r'\d{6}', match.group(1)):
            found_patches.add(num)

    # Merge with Claude's extracted patches
    claude_patches = data.get("patches", [])
    if isinstance(claude_patches, list):
        all_patches = found_patches | set(str(p) for p in claude_patches if str(p) != "non_specifie")
    else:
        all_patches = found_patches

    # Sort: numeric patches first (ascending), then any alphanumeric ones
    numeric  = sorted([p for p in all_patches if p.isdigit()], key=int)
    alphanum = sorted([p for p in all_patches if not p.isdigit()])
    data["patches"] = numeric + alphanum

    # 3. Ensure resolution.actions_realisees is a list
    res = data.get("resolution", {})
    if isinstance(res.get("actions_realisees"), str):
        data["resolution"]["actions_realisees"] = [res["actions_realisees"]]

    # 4. Ensure erreurs_documentation is a list of dicts (not strings)
    errs = data.get("erreurs_documentation", [])
    if errs and isinstance(errs[0], str):
        data["erreurs_documentation"] = [
            {"document": "non_specifie", "erreur": e, "correction": "non_specifie"}
            for e in errs
        ]

    return data

# ==========================================================
# PER-TICKET PROCESSING
# ==========================================================

def process_ticket(client, path: Path, out_dir: Path) -> dict:
    ref  = re.sub(r"[^A-Za-z0-9_\-]", "_", path.stem)
    text = path.read_text(encoding="utf-8", errors="ignore")

    fallback_team = extract_team(text)

    logging.info(f"[JSON] {path.name}")
    data = generate_json(client, text, ref)
    data = postprocess(data, text, fallback_team)

    data["_metadata"] = {
        "generated_at": dt.datetime.now().isoformat(),
        "model":        MODEL_ID,
        "source_file":  path.name,
        **data["_usage"],
    }

    out_path = out_dir / f"{ref}_analysis.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    cost = data["_usage"]["total_cost_usd"]
    logging.info(f"[COST] {path.name} → {cost} USD")
    logging.info(f"[OK]   {out_path.name}")

    return data

# ==========================================================
# MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser(description="Sopra HR ticket → rich JSON analysis via Claude")
    ap.add_argument("-i", "--input",   required=True, type=Path, help="Folder of .txt tickets")
    ap.add_argument("-o", "--output",  required=True, type=Path, help="Output folder for JSON analyses")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip tickets that already have an output file")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    client = load_client()
    args.output.mkdir(parents=True, exist_ok=True)

    sources = sorted(args.input.rglob("*.txt"))
    logging.info(f"Found {len(sources)} tickets")

    total = 0.0

    for src in sources:
        ref      = re.sub(r"[^A-Za-z0-9_\-]", "_", src.stem)
        out_path = args.output / f"{ref}_analysis.json"

        if args.skip_existing and out_path.exists():
            logging.info(f"[SKIP] {src.name}")
            continue

        try:
            data   = process_ticket(client, src, args.output)
            total += data["_usage"]["total_cost_usd"]
            time.sleep(SLEEP)
        except Exception as e:
            logging.error(f"[ERROR] {src.name}: {e}", exc_info=True)
            time.sleep(SLEEP)

    logging.info("=" * 50)
    logging.info(f"TOTAL COST: {round(total, 6)} USD")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()