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
from typing import Any, Dict

import boto3
import tiktoken
from dotenv import load_dotenv

# =========================
# Config
# =========================
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"
TEMP_JSON = 0.1  
TOK_JSON = 1500
RATE_SLEEP = 0.3

# Pricing (adjust according to your region)
MODEL_PRICES = {
    "meta.llama3-70b-instruct-v1:0": {
        "input_per_1k": 0.00799,   # example
        "output_per_1k": 0.00799
    }
}

# =========================
# Token counter
# =========================
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("gpt2")
    return len(enc.encode(text))

# =========================
# Helpers — Team extraction
# =========================
TEAM_KEYWORDS = {
    "APPLI": r"\b(appli|application|app)\b",
    "DSN": r"\b(dsn)\b",
    "OUTILS": r"\b(outils?)\b",
}

def extract_team(text: str) -> str:
    lower = text.lower()
    m = re.search(r"support_team:\s*([A-Za-z0-9]+)", lower)
    if m:
        return m.group(1).upper()
    for team, pat in TEAM_KEYWORDS.items():
        if re.search(pat, lower):
            return team
    return ""

# =========================
# Prompts
# =========================
SYSTEM_PROMPT = """You are a Senior Sopra HR Incident Analyst and a rigorous information extractor.

MISSION
- Read noisy Lotus-style support tickets and output a single, valid JSON object that matches the exact schema provided by the user prompt.
- You must be conservative, avoid assumptions, and never invent facts.

GENERAL NORMS
- Language of values: French (keep technical tokens as-is).
- Dates: return literal dates from the ticket; prefer ISO "YYYY-MM-DD HH:MM:SS" when exact time is present, otherwise "YYYY-MM-DD".
- Whitespace: trim all fields; collapse repeated spaces.
- Arrays: non-empty (use ["non_specifie"] if nothing is found).
- Strings: never empty (use "non_specifie" when unknown).
- No hallucinations: if the ticket does not state the information, return "non_specifie".
- Output: a SINGLE JSON object — no code fences, no markdown, no commentary.

TEAM & RESPONSIBILITY
- "equipe_responsable": prioritize header key `support_team: X` (normalize to "APPLI", "DSN", "OUTILS"); otherwise infer cautiously or "non_specifie".
- "resolution.responsabilite": one of ["client","soprahr","tierce_partie","partagee","non_specifie"]. Infer only when clearly indicated (e.g., "Incorrect use (Closed)" → "client").

SEVERITY & INCIDENT TYPE
- "probleme.gravite": keep ticket’s wording when present (e.g., "Blocking bug", "Very Important", "Serious", "Nice to have"); else "non_specifie".
- "cause_racine.type_incident": choose EXACTLY one from:
  ["defaut_logiciel","mauvaise_configuration","donnees","infrastructure","utilisation","non_specifie"].
  Map near-synonyms conservatively (e.g., "Problème de configuration" → "mauvaise_configuration").

CHRONOLOGY EXTRACTION
- Split the conversation by timestamps in the ticket (e.g., "DD/MM/YYYY HH:MM:SS [TZ]" or similar).
- Create one chronology entry per dated event.
- For each entry, fill:
  - "date": normalize to ISO if time present; else "YYYY-MM-DD".
  - "action": short label summarizing the step (e.g., "Demande de logs", "Livraison patch", "Analyse SQL").
  - "description": concise 1–3 sentences capturing what happened (keep essential technical details such as error codes/ORA errors).
  - "hypothese": the working assumption/diagnosis at that moment (or "non_specifie").
  - "resultat": outcome or next step ("logs fournis", "patch livré", "en attente client", etc. or "non_specifie").

FACTS & CLOSURE RULES
- "facts.client", "facts.version", "facts.systeme": extract from header or first message block.
- "facts.ouverture": first ticket creation datetime found.
- "facts.fermeture": the TRUE final closure:
  1) prefer latest **client-confirmed** closure (e.g., "je souhaite fermer le dossier"),
  2) else latest **support** closure (e.g., Status CP/CU Closed),
  3) else "non_specifie".
- "facts.duree": integer days between ouverture and fermeture (if both known) formatted as "<N> jours"; else "non_specifie".

PATCHES / PREREQUIS / DOC ERRORS
- "patches": collect explicit patch/kit numbers listed (e.g., "178236", "KTFRCPF15Q2"). Return as array of strings (deduplicate).
- "prerequis": list prerequisite kits/patches when explicitly stated; else ["non_specifie"].
- "erreurs_documentation": list any documentation errata indicated by support; else ["non_specifie"].

KEYWORDS & IMPACT
- "impact_metier": short sentence describing operational effect (e.g., "Blocage NRB en production", "Impossible de saisir APET (NAF)").
- "mots_cles": concise tokens (e.g., page/process IDs, program names, error codes).

VALIDATION GATE (do this silently before finalizing)
- Ensure every key from the schema exists.
- Ensure no string is empty; substitute "non_specifie" where missing.
- Ensure all arrays are non-empty.
- Ensure enums match the allowed sets.
- Ensure the output is a SINGLE JSON object (no markdown, no comments, no trailing text).
- If anything is uncertain → use "non_specifie".
"""

JSON_USER_PROMPT = """Vous devez suivre un processus de raisonnement en 2 phases :

========================================
PHASE 1 — COMPRÉHENSION INTERNE (NE RIEN AFFICHER)
========================================
- Lire TOUT le ticket du début à la fin (format Lotus hétérogène).
- Reconstituer la chronologie, le contexte, les symptômes, les actions client/support, l’analyse, la cause et la clôture finale.
- Identifier : client / version / système, ouverture, vraie fermeture (fermeture client prioritaire), responsabilité, gravité, patchs/kits, impacts métier.
- Ne rien afficher pendant cette phase.

========================================
PHASE 2 — SORTIE (UN SEUL JSON VALIDE, RIEN D’AUTRE)
========================================
Après compréhension complète, renvoyer UN SEUL OBJET JSON qui respecte STRICTEMENT ce schéma :

{
 "reference": "",
 "facts": {
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
     "action": "",
     "description": "",
     "hypothese": "",
     "resultat": ""
   }
 ],
 "resolution": {
   "statut": "",
   "code": "",
   "verdict": "",
   "responsabilite": "",
   "actions_realisees": ""
 },
 "patches": [],
 "prerequis": [],
 "erreurs_documentation": [],
 "impact_metier": "",
 "mots_cles": []
}

RÈGLES STRICTES (OBLIGATOIRES)
1) Aucune explication/markdown/texte libre autour du JSON. Un seul objet JSON.
2) Toutes les clés doivent exister. Aucune chaîne vide : si inconnu → "non_specifie".
3) Les tableaux ne doivent jamais être vides : si rien → ["non_specifie"].
4) "cause_racine.type_incident" ∈ ["defaut_logiciel","mauvaise_configuration","donnees","infrastructure","utilisation","non_specifie"].
5) "equipe_responsable" : utiliser le header "support_team: X" (normaliser "APPLI","DSN","OUTILS") ; sinon inférer prudemment ou "non_specifie".
6) "resolution.responsabilite" ∈ ["client","soprahr","tierce_partie","partagee","non_specifie"].
7) "resolution.statut" ∈ ["resolu","non_resolu","contournement","en_cours","Closed","non_specifie"].
8) Chaque entrée "chronologie" doit avoir date/action/description/hypothese/resultat (remplir "non_specifie" si manquant).
9) Avant d’afficher : valider les clés, les enums, l’absence de chaînes vides, et l’absence de tableaux vides.

TICKET À ANALYSER (brut) :
{{TICKET_TEXT}}

RÉFÉRENCE : {{REF}}

INSTRUCTION FINALE :
Renvoyer UNIQUEMENT l’objet JSON final — sans balises, ni commentaires, ni texte additionnel."""

# =========================
# Bedrock client
# =========================
def load_bedrock_client():
    load_dotenv()
    key = os.getenv("AWS_ACCESS_KEY_ID")
    sec = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "eu-west-2")
    if not key or not sec:
        raise RuntimeError("Missing AWS credentials")
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=sec
    )

# =========================
# Llama 3 Invocation
# =========================
def call_llm_bedrock(client, system_prompt, user_prompt):

    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    input_tokens = count_tokens(prompt)

    request_body = {
        "prompt": prompt,
        "max_gen_len": TOK_JSON,
        "temperature": TEMP_JSON
    }

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(request_body)
    )

    result = json.loads(response["body"].read().decode("utf-8"))
    completion = result["generation"].strip()

    output_tokens = count_tokens(completion)

    rates = MODEL_PRICES[BEDROCK_MODEL_ID]
    cost_input = (input_tokens / 1000) * rates["input_per_1k"]
    cost_output = (output_tokens / 1000) * rates["output_per_1k"]

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_input_usd": round(cost_input, 6),
        "cost_output_usd": round(cost_output, 6),
        "total_cost_usd": round(cost_input + cost_output, 6)
    }

    return completion, usage

# =========================
# JSON Extraction
# =========================
def generate_json(bedrock_client, ticket_text, ref):
    user = JSON_USER_PROMPT.format(TICKET_TEXT=ticket_text, REF=ref)
    raw, usage = call_llm_bedrock(bedrock_client, SYSTEM_PROMPT, user)

    raw_clean = raw.strip()
    raw_clean = re.sub(r"^```json\s*|\s*```$", "", raw_clean).strip()

    try:
        data = json.loads(raw_clean)
        data["_usage"] = usage
        return data
    except:
        return {
            "reference": ref,
            "_error": "Bad JSON from model",
            "_raw": raw,
            "_usage": usage
        }

# =========================
# Ticket processing
# =========================
def process_ticket(bedrock_client, src: Path, out_dir: Path):

    ref = re.sub(r"[^A-Za-z0-9_\-]", "_", src.stem)
    txt = src.read_text(encoding="utf-8", errors="ignore")

    team = extract_team(txt)
    logging.info(f"[JSON] {src.name}")

    data = generate_json(bedrock_client, txt, ref)

    if not data.get("equipe_responsable"):
        data["equipe_responsable"] = team

    data["_metadata"] = {
        "generated_at": dt.datetime.now().isoformat(),
        "model": BEDROCK_MODEL_ID,
        "source_file": src.name,
        **data["_usage"],
    }

    out = out_dir / f"{ref}_analysis.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"[COST] {src.name} → {data['_usage']['total_cost_usd']}$")
    logging.info(f"[OK] {out.name}")

    return data

# =========================
# Main
# =========================
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

    client = load_bedrock_client()
    args.output.mkdir(parents=True, exist_ok=True)

    total = 0.0
    for src in sorted(args.input.rglob("*.txt")):
        try:
            data = process_ticket(client, src, args.output)
            total += data["_usage"]["total_cost_usd"]
        except Exception as e:
            logging.error(f"[ERROR] {src.name}: {e}", exc_info=True)

    logging.info("=" * 50)
    logging.info(f"TOTAL PIPELINE COST: {round(total, 6)} USD")
    logging.info("=" * 50)
    logging.info("Done.")

if __name__ == "__main__":
    main()