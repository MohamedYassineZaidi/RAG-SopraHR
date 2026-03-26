#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_utils.py
============
Shared utilities for vectorless_rag.py and hybrid_rag.py.

Contains:
  - Configuration constants
  - Cluster taxonomy + assignment
  - Ticket summary generation
  - Bedrock client + call wrapper
  - Answer generation
  - Result display
"""

import os
import re
import json
import textwrap
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import boto3


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM   = 768
BEDROCK_MODEL   = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION  = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
TOP_K           = 5
RRF_K           = 60   # Reciprocal Rank Fusion constant
TEAMS           = ["DSN", "Appli", "Outils"]


# ─────────────────────────────────────────────
# CLUSTER TAXONOMY
# Keyword rules that assign each ticket to a topic cluster.
# Order matters — first match wins.
# ─────────────────────────────────────────────

CLUSTER_RULES = {
    "DSN": [
        ("Kit & Livraison",      [r"kit\s+dsn", r"demande de fourniture", r"livraison", r"espdsn", r"mise à disposition"]),
        ("REGDSN & Erreurs",     [r"regdsn", r"ora-\d+", r"erreur.*établissement", r"bad sql", r"exception"]),
        ("DSN Déclaration",      [r"bloc\s+\d+", r"n.ud", r"signalement", r"déclaration", r"phase\s+[23]"]),
        ("URSSAF & SIREN",       [r"urssaf", r"siren", r"télérèglement", r"cotisation"]),
        ("Espace DSN",           [r"espace dsn", r"edsn", r"redémarr", r"service.*dsn"]),
        ("Paramétrage DSN",      [r"paramétrage", r"configuration.*dsn", r"mapping", r"structure zz"]),
        ("Autre DSN",            []),
    ],
    "Appli": [
        # DADS-U must come BEFORE Congés to prevent "Absence J41010" in DADS-U titles
        # from being routed to Congés & Absences
        ("DADS-U & N4DS",        [r"dads[-\s]?u", r"\bn4ds\b", r"dadsu", r"\bkfn\b", r"v01x\d+"]),
        ("Paie & Calcul",        [r"\bpaie\b", r"calcul.*salaire", r"bulletin.*paie", r"acompte", r"\brubrique\b", r"zy\w{3,}", r"zx\w{3,}"]),
        ("Congés & Absences",    [r"cong[eé]", r"\bcp\b", r"\bdif\b", r"\bcpf\b", r"maladie",
                                  r"absence(?!.*j4\d{4})",  # absence NOT followed by J4xxxx code (those are DADS)
                                  r"temps partiel", r"attestation.*salaire", r"interruption.*travail",
                                  r"cdd.*cdi|cdi.*cdd", r"bloc\s+41", r"module.*formation"]),
        ("Kit & Patches",        [r"demande.*kit", r"demande.*patch", r"livraison", r"mise.*niveau"]),
        ("Pages Web & Design",   [r"page web", r"fswdre", r"fsw\w+", r"validation.*extr", r"design center",
                                  r"épuration", r"\bvs20\b", r"\bsubg\w*",
                                  r"noeud\s+[a-z]{2}\d", r"arbre fonctionnel", r"\bc3p\b"]),
        ("Saisie & Formulaires", [r"saisie", r"formulaire", r"[eé]cran", r"champ", r"zy2", r"\bedsn\b"]),
        ("Traitement & Batch",   [r"traitement", r"batch", r"\bnrb\b", r"bordereau", r"bay\w+",
                                  r"\bzygs\b", r"\bzygr\b"]),
        ("Autre Appli",          []),
    ],
    "Outils": [
        ("HRCT & Génération",    [r"hrct", r"g[eé]n[eé]ration", r"compilation", r"objet.*paie"]),
        ("Design Center",        [r"design center", r"dsgn", r"page web", r"validation", r"subg"]),
        ("HRQuery & Requêtes",   [r"hrquery", r"query", r"requ[eê]te", r"exploration"]),
        ("Erreurs Système",      [r"erreur syst[eè]me", r"r_system", r"fin anormale", r"ora-\d+", r"exception"]),
        ("Kit & Livraison",      [r"demande.*kit", r"fourniture", r"livraison", r"patch"]),
        ("Connexion & Services", [r"connexion", r"\bservice\b", r"red[eé]marr", r"acc[eè]s"]),
        ("HRAnalytics & Space",  [r"hraspace", r"analytics", r"space", r"smart"]),
        ("Autre Outils",         []),
    ],
}


def assign_cluster(ticket: dict, team: str) -> str:
    """Assigns a ticket to its topic cluster using keyword rules."""
    text = f"{ticket.get('title', '')} {ticket.get('description', '')}".lower()
    for cluster_name, patterns in CLUSTER_RULES.get(team, []):
        if not patterns:
            continue
        if any(re.search(p, text) for p in patterns):
            return cluster_name
    return f"Autre {team}"


# ─────────────────────────────────────────────
# CLUSTER DESCRIPTIONS
# Human-readable descriptions used in the step-1 cluster prompt.
# Includes common error codes so Claude can route correctly even
# when the query contains only a code like "ORA-00942".
# ─────────────────────────────────────────────

CLUSTER_DESCRIPTIONS = {
    "DSN": {
        "Kit & Livraison":   "Demandes de fourniture et livraison de kits DSN, mise à disposition EspDSN, téléchargement kits signalement",
        "REGDSN & Erreurs":  "Erreurs ORA-00942 ORA-01403 Bad SQL exception Java lors exécution REGDSN établissement table manquante",
        "DSN Déclaration":   "Blocs DSN manquants nœuds S21.G00 signalement phase 2 phase 3 déclaration incorrecte codes CRM",
        "URSSAF & SIREN":    "Erreurs URSSAF SIREN invalide télérèglement cotisation rejet bordereau DUCS",
        "Espace DSN":        "Service EspDSN inaccessible redémarrage connexion droits accès portail authentification",
        "Paramétrage DSN":   "Configuration paramétrage mapping structure ZZ établissement DSN rubriques DSN",
        "Autre DSN":         "Problèmes DSN divers non classifiés questions générales DSN",
    },
    "Appli": {
        "Paie & Calcul":        "Erreur calcul bulletin paie rubriques ZY ZX acompte moteur calcul exception Java salaire brut net",
        "Congés & Absences":    "Congés payés CP DIF CPF absences maladie temps partiels attestation salaire interruption travail "
                                "CDD CDI module formation codes absence J4xxxx bloc 41 KFN planning",
        "DADS-U & N4DS":        "Génération DADS-U N4DS V01X10 données manquantes salariés déclaration annuelle chaine KFN DADS "
                                "retraite position administrative S40 mise à disposition période",
        "Kit & Patches":        "Demandes kit applicatif patches mise à niveau version HRA livraison correctifs",
        "Pages Web & Design":   "Pages FSW FSWDRE01 validation écran Design Center affichage web épuration VS20 tables collections SUBG "
                                "noeuds arbre fonctionnel C3P pénibilité AU2 AU6 création noeud",
        "Saisie & Formulaires": "Saisie formulaire champs verrouillés écrans ZY2 contrôle saisie APET NAF code NAF "
                                "EDSN espace DSN signalement disparu accès formulaires",
        "Traitement & Batch":   "Batch NRB bordereaux traitements BAY fin anormale génération chaîne de traitement "
                                "ZYGS ZYGR carrière secondaire alimentation tables",
        "Autre Appli":          "Problèmes applicatifs divers non classifiés carrière échelon avancement modules spécifiques "
                                "paramétrage établissement questions diverses hors paie hors congés",
    },
    "Outils": {
        "HRCT & Génération":    "HRCT compilation objets paie erreurs génération fin anormale régénération",
        "Design Center":        "Design Center pages SUBG validation affichage mise en page épuration collections",
        "HRQuery & Requêtes":   "HRQuery Smart requêtes exploration paie résultats incorrects extraction données",
        "Erreurs Système":      "Erreurs R_SYSTEM ORA-00942 fin anormale exceptions système infrastructure serveur",
        "Kit & Livraison":      "Demandes kit outils patches livraison téléchargement mise à niveau",
        "Connexion & Services": "Connexion services HRxx redémarrage accès authentification droits",
        "HRAnalytics & Space":  "HRAnalytics HRSpace Smart données manquantes reporting tableaux de bord",
        "Autre Outils":         "Problèmes outils divers non classifiés questions générales hors catégories",
    },
}


# ─────────────────────────────────────────────
# TICKET SUMMARY
# Richer multi-line representation for Level 3 index.
# Gives Claude enough signal to distinguish between similar tickets.
# ─────────────────────────────────────────────

def ticket_to_summary(ticket: dict) -> str:
    """
    Produces a richer multi-line L3 entry per ticket.

    Format:
      REF | Title | Version
        Symptôme: <first meaningful sentence from description>
        Action:   <first action sentence from resolution>
        Codes: ORA-XXXXX ZY1234 | Patches: ZY1234,ZY5678

    The symptom + action lines give Claude real signal to distinguish
    between tickets that have similar titles but different root causes.
    Error codes and patch numbers are extracted explicitly so they
    are visible even if buried in long text.
    """
    ref     = ticket.get("reference", "")
    title   = ticket.get("title", "")[:70]
    version = ticket.get("version", "")

    # First meaningful sentence from description (the symptom)
    desc_phrase = ""
    for s in re.split(r'[.!?\n]\s*', ticket.get("description", "")):
        s = s.strip()
        if len(s) > 20 and not s.lower().startswith("bonjour"):
            desc_phrase = s[:130]
            break

    # First action sentence from resolution
    res_phrase = ""
    for s in re.split(r'[.!?\n]\s*', ticket.get("resolution", "")):
        s = s.strip()
        if len(s) > 20 and not s.lower().startswith("bonjour"):
            res_phrase = s[:130]
            break

    # Extract error/product codes for explicit visibility
    codes = re.findall(
        r'\b(ORA-\d+|R_SYSTEM|ZY\w+|ZX\w+|FSW\w+|BAY\w+|REGDSN|HRCT|NRB\w*)\b',
        f"{title} {desc_phrase} {res_phrase}",
        re.IGNORECASE,
    )
    code_str = " ".join(sorted({c.upper() for c in codes}))[:60]
    patches  = ",".join(ticket.get("patches", [])[:4])

    lines = [f"{ref} | {title}" + (f" | {version}" if version else "")]
    if desc_phrase: lines.append(f"  Symptôme: {desc_phrase}")
    if res_phrase:  lines.append(f"  Action:   {res_phrase}")
    extras = " | ".join(filter(None, [
        f"Codes: {code_str}" if code_str else "",
        f"Patches: {patches}" if patches else "",
    ]))
    if extras:
        lines.append(f"  {extras}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# BEDROCK
# ─────────────────────────────────────────────

def get_bedrock_client():
    kwargs = {"service_name": "bedrock-runtime", "region_name": BEDROCK_REGION}
    key    = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    token  = os.getenv("AWS_SESSION_TOKEN")
    if key and secret:
        kwargs["aws_access_key_id"]     = key
        kwargs["aws_secret_access_key"] = secret
        if token:
            kwargs["aws_session_token"] = token
    return boto3.client(**kwargs)


def call_bedrock(client, prompt: str, max_tokens: int = 1024) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    })
    response = client.invoke_model(
        modelId=BEDROCK_MODEL, body=body,
        contentType="application/json", accept="application/json"
    )
    return json.loads(response["body"].read())["content"][0]["text"].strip().replace("\r\n", "\n").replace("\r", "\n")


# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────

SUGGESTION_PROMPT = """Tu es un assistant technique pour l'équipe support de Sopra HR.
Un consultant vient de recevoir un nouveau ticket.
En te basant sur les tickets similaires résolus ci-dessous, propose une action concrète.

Règles:
- Réponds dans la même langue que la question (français/anglais)
- Sois précis: cite les numéros de patch, les noms de champs, les tables si pertinent
- Structure ta réponse: (1) Diagnostic probable, (2) Action recommandée, (3) Patches si applicable
- Sois concis — le consultant est occupé

Nouveau ticket:
{query}

Tickets similaires résolus:
{context}

Suggestion:"""


def generate_suggestion(query: str, hits: list, bedrock_client) -> str:
    context_parts = []
    for h in hits[:3]:
        part = f"--- {h['reference']} (source: {h.get('source', '')}) ---\n"
        part += f"Titre: {h['title']}\n"
        if h.get("resolution"):
            part += f"Résolution: {h['resolution'][:350]}\n"
        if h.get("patches"):
            part += f"Patches: {', '.join(h['patches'])}\n"
        context_parts.append(part)

    context = "\n\n".join(context_parts)
    prompt  = SUGGESTION_PROMPT.format(query=query, context=context)
    try:
        return call_bedrock(bedrock_client, prompt)
    except Exception as e:
        return f"[Erreur génération: {e}]"


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_results(query: str, team: str, hits: list, suggestion: str, mode: str):
    width = 70
    print(f"\n{'═'*width}")
    print(f"  MODE: {mode.upper()} | ÉQUIPE: {team}")
    print(f"  QUERY: {query[:60]}{'...' if len(query) > 60 else ''}")
    print(f"{'═'*width}")
    print(f"\n📋  TICKETS SIMILAIRES:\n")

    for h in hits[:3]:
        score_val = h.get("rrf_score", h.get("similarity", h.get("score", 0))) or 0
        score_str = f"{score_val:.4f}"
        source    = h.get("source", "")
        print(f"  #{h['rank']}  {h['reference']}  [score: {score_str}]  [{source}]")
        print(f"      {h['title'][:65]}")
        if h.get("version"):
            print(f"      Version: {h['version']}")
        if h.get("resolution"):
            res = h["resolution"].replace("\n", " ")[:150]
            print(f"      ↳ {res}{'...' if len(h['resolution']) > 150 else ''}")
        if h.get("patches"):
            print(f"      🔧 {', '.join(h['patches'])}")
        print()

    print(f"{'─'*width}")
    print(f"💬  SUGGESTION:\n")

    # Deduplicate repeated paragraphs (Bedrock occasionally echoes content)
    seen_lines = set()
    deduped    = []
    for line in suggestion.splitlines():
        key = line.strip().lower()
        if key and key in seen_lines:
            continue
        seen_lines.add(key)
        deduped.append(line)
    suggestion = "\n".join(deduped)

    for line in suggestion.splitlines():
        if line.strip():
            for wrapped in textwrap.wrap(line, width=66):
                print(f"  {wrapped}")
        else:
            print()
            
    print(f"{'═'*width}\n")