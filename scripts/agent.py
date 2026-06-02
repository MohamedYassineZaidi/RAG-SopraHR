#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent.py
========
Manual ReAct agent for Sopra HR ticket resolution.

No LangChain dependency — drives the Thought → Action → Observation
loop directly using call_bedrock(), fully compatible with Python 3.14.

Architecture
------------
  LLM    : Amazon Bedrock (Claude 3.7 Sonnet) via rag_utils.call_bedrock,
            or OpenAI-compatible via openai SDK (set OPENAI_API_KEY).
  Loop   : Manual ReAct — parse Action/Action Input from LLM text,
            execute the matching tool, append Observation, repeat.
  Memory : Conversation turns accumulated in a list (single session).
  Tools  : Four retriever callables from agent_tools.py.
  Output : Validated JSON matching SopraHROutput schema.

Environment variables (in .env)
--------------------------------
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
  AWS_DEFAULT_REGION   (default: eu-west-1)
  BEDROCK_MODEL_ID     (default: anthropic.claude-3-haiku-20240307-v1:0)
  OPENAI_API_KEY       (optional — use instead of Bedrock)
  OPENAI_BASE_URL      (optional — for Mistral or other OpenAI-compat API)
  OPENAI_MODEL         (optional — default: gpt-4o-mini)
"""

import os
import re
import sys
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rag_utils import get_bedrock_client, call_bedrock
from agent_tools import build_tool_registry, TOOL_DESCRIPTIONS, _store as _tool_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REACT SYSTEM PROMPT
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Tu es un agent expert en support Sopra HR. Tu aides les consultants à \
résoudre des tickets techniques.

## RÈGLES STRICTES
1. Utilise TOUJOURS l'outil RechercheHybride pour chercher des tickets.
2. Utilise TOUJOURS au moins un outil avant de répondre.
3. Ne cite JAMAIS un ticket que tu n'as pas trouvé dans les Observations. \
Les références de tickets sont au format "FR WXXXXXX" (ex: "FR W210XXX"). \
Copie-les EXACTEMENT telles qu'elles apparaissent dans les Observations — \
ne génère AUCUNE référence de ton propre chef.
4. Si les résultats de RechercheHybride sont insuffisants, relance avec une requête reformulée \
en utilisant des termes techniques différents (codes erreur, noms de rubriques, modules).
5. Ta réponse FINALE doit être UNIQUEMENT un objet JSON valide.

## STRATÉGIE DE RECHERCHE (OBLIGATOIRE)
- Étape 1: RechercheHybride avec les termes EXACTS du problème (codes erreur, noms de pages, rubriques, numéros de patch).
- Étape 2: RechercheHybride une SECONDE fois avec une reformulation DIFFÉRENTE — utilise un angle complémentaire :
  * si la 1ʳᵉ requête contenait des codes techniques, reformule en termes métier ;
  * si la 1ʳᵉ requête était en langage métier, ajoute les codes/modules concernés ;
  * varie au moins 3 mots-clés par rapport à la 1ʳᵉ requête.
- Étape 3 (OBLIGATOIRE): DétailsTicket sur la référence du ticket #1 (rang 1) de la MEILLEURE \
des deux recherches pour obtenir sa résolution complète. NE SAUTE PAS cette étape.
- Étape 4: rédige ta Final Answer. La "resolution" est ta PROPRE recommandation d'expert \
basée sur ta connaissance technique de Sopra HR, PAS une copie des tickets. \
Les tickets servent de RÉFÉRENCE pour le consultant, pas de source pour la résolution.
- Inclus TOUJOURS les codes techniques disponibles dans tes requêtes : ORA-XXXX, ZY/ZX, noms de pages (FSW*), numéros de patch, etc.

Tu DOIS effectuer AU MOINS 2 appels RechercheHybride avec des reformulations distinctes AVANT toute Final Answer.

## OUTILS DISPONIBLES
{tool_descriptions}

## FORMAT OBLIGATOIRE — respecte-le EXACTEMENT à chaque étape
Thought: <ta réflexion>
Action: <nom exact d'un outil>
Action Input: <requête pour l'outil>

Quand tu as assez d'informations:
Thought: j'ai maintenant assez d'informations pour répondre
Final Answer: {{
  "analyse": "<résumé du problème>",
  "tickets_utilises": ["<ref exacte copiée depuis Observation>"],
  "cause_probable": "<cause technique identifiée grâce à ton expertise>",
  "reponse_lotus": "<message professionnel et naturel au client. Inclure: contexte du problème, explication de la cause, tes recommandations de résolution, patches à intégrer si pertinents. Ton: courtois et technique.>",
  "resolution": "<ta recommandation d'expert : étapes de résolution basées sur tes connaissances techniques de Sopra HR, Oracle, et du contexte métier. Utilise les tickets trouvés comme contexte informatif mais rédige ta PROPRE résolution claire, structurée en étapes, adaptée au problème spécifique du consultant. Si les tickets trouvés ne correspondent pas exactement, base-toi sur ton expertise technique.>",
  "tickets_references": [{{"ref": "<ref exacte copiée depuis Observation>", "titre": "<titre du ticket>", "resolution_ticket": "<résumé court de la résolution du ticket pour référence>", "patches": ["<patches du ticket>"]}}],
  "patches": [{{"patch": "<numéro de patch>", "ref": "<ref exacte du ticket source, copiée depuis Observation>"}}]
}}

IMPORTANT: Dans "tickets_utilises", copie mot pour mot les références \
du champ [ref] dans les Observations (ex: si l'Observation dit "#1 [ref: FR W214117]", \
mets "FR W214117"). N'invente AUCUNE référence, n'utilise pas les exemples du prompt.
IMPORTANT: "resolution" est ta PROPRE recommandation technique d'expert. \
NE COPIE PAS les résolutions des tickets. Utilise les tickets comme contexte et inspiration, \
mais rédige une résolution originale, claire et structurée en étapes, adaptée au problème posé. \
Si les tickets trouvés traitent d'un problème légèrement différent, adapte ta résolution au problème réel.
IMPORTANT: "tickets_references" fournit les tickets trouvés comme RÉFÉRENCE pour le consultant. \
Résume brièvement la résolution de chaque ticket pour que le consultant puisse les consulter s'il le souhaite.
IMPORTANT: Dans "patches", liste TOUS les patches trouvés dans les Observations avec leur ref source. \
Pour chaque patch: {{"patch": "127043", "ref": "FR W214117"}} (ref = ticket où ce patch est mentionné).
N'inclus que des patches réels trouvés dans les Observations, pas des inventions.
""".format(tool_descriptions=TOOL_DESCRIPTIONS)


# ─────────────────────────────────────────────
# LLM ABSTRACTION
# ─────────────────────────────────────────────

class _LLM:
    """Thin wrapper that unifies Bedrock and OpenAI-compatible calls."""

    def __init__(self) -> None:
        self._openai_key = os.getenv("OPENAI_API_KEY")
        if self._openai_key:
            import openai
            self._oa_client = openai.OpenAI(
                api_key=self._openai_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
            self._model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self._backend = "openai"
        else:
            self._bedrock = get_bedrock_client()
            self._backend = "bedrock"

    def call(self, messages: list[dict], max_tokens: int = 4096) -> str:
        if self._backend == "openai":
            resp = self._oa_client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()

        # Bedrock: flatten message list into a single prompt string
        parts: list[str] = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"\nQuestion: {content}")
            else:
                parts.append(content)
        prompt = "\n".join(parts)

        # Bedrock Claude Haiku context limit ~200k chars; keep a safe margin.
        # If the prompt is too long, truncate the middle (scratchpad observations)
        # while preserving the system prompt (start) and the latest user question (end).
        _MAX_PROMPT_CHARS = 140_000
        if len(prompt) > _MAX_PROMPT_CHARS:
            # Find the boundary between system+question and the scratchpad
            # The system prompt ends before the first "\nQuestion:" occurrence.
            q_idx = prompt.find("\nQuestion:")
            if q_idx == -1:
                # Fallback: hard-truncate from the middle
                keep = _MAX_PROMPT_CHARS
                prompt = prompt[:keep]
            else:
                # Keep system prompt + last part of scratchpad + question
                system_part = prompt[:q_idx]
                rest = prompt[q_idx:]
                if len(system_part) + len(rest) > _MAX_PROMPT_CHARS:
                    allowed_rest = _MAX_PROMPT_CHARS - len(system_part) - 200
                    # Keep the tail of rest (most recent observations)
                    rest = "\n[... observations tronqu\u00e9es pour limite contexte ...]\n" + rest[-allowed_rest:]
                prompt = system_part + rest

        return call_bedrock(self._bedrock, prompt, max_tokens=max_tokens)


# ─────────────────────────────────────────────
# REACT PARSERS
# ─────────────────────────────────────────────

_ACTION_RE       = re.compile(r"Action\s*:\s*(.+)",              re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input\s*:\s*(.+)",        re.IGNORECASE | re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer\s*:\s*(.+)",        re.IGNORECASE | re.DOTALL)


def _parse_step(text: str) -> tuple[str | None, str | None]:
    """
    Returns (action_name, final_answer_text) from one LLM response.
    Both may be None if the LLM produced neither pattern.

    If both Action: and Final Answer: appear in the same response,
    the Action is prioritised — the tool must run before we accept
    a Final Answer (otherwise the LLM skips the tool and hallucinates).
    """
    action = _ACTION_RE.search(text)
    final  = _FINAL_RE.search(text)

    if action and final:
        # Both present — prioritise whichever comes first in the text,
        # but if Action comes first, always execute the tool.
        if action.start() < final.start():
            return action.group(1).strip(), None
        # Final Answer comes first (unusual) — trust it
        return None, final.group(1).strip()

    if action:
        return action.group(1).strip(), None
    if final:
        return None, final.group(1).strip()

    return None, None


def _extract_action_input(text: str, fallback: str) -> str:
    m = _ACTION_INPUT_RE.search(text)
    if not m:
        return fallback
    raw = m.group(1).strip()
    # Trim at any following Observation or Thought line
    raw = raw.split("\nObservation")[0].split("\nThought")[0].strip()
    return raw


# ─────────────────────────────────────────────
# AGENT CLASS
# ─────────────────────────────────────────────

_EMPTY_OUTPUT: dict = {
    "analyse": "",
    "tickets_utilises": [],
    "cause_probable": "",
    "resolution": "",
    "tickets_references": [],
    "patches": [],
    "reponse_lotus": "",
}


def _coerce_string_fields(result: dict) -> dict:
    """Always coerce string-typed output fields — even when there are no RAG hits."""
    for str_field in ("analyse", "cause_probable", "resolution", "reponse_lotus"):
        val = result.get(str_field)
        if val is None:
            result[str_field] = ""
        elif not isinstance(val, str):
            if isinstance(val, list):
                result[str_field] = " ".join(str(v) for v in val if v)
            elif isinstance(val, dict):
                result[str_field] = " ".join(str(v) for v in val.values() if v)
            else:
                result[str_field] = str(val)
    return result


def _enforce_rank1_resolution(result: dict) -> dict:
    """
    Enforces factual accuracy from retrieved tickets:
      - tickets_utilises    : rank-1 ref forced to first position
      - patches             : built as [{"patch": p, "ref": ref}] from ALL hits
      - tickets_references  : built from ALL hits as consultant reference material

    Kept as LLM-generated (natural language):
      - analyse, cause_probable, resolution, reponse_lotus
    """
    # Always coerce string fields regardless of whether there are hits.
    result = _coerce_string_fields(result)

    from agent_tools import _store as _tool_store
    hits = _tool_store.last_hits
    logger.debug("[enforce] last_hits count=%d", len(hits))
    if not hits:
        # No RAG hits — still try to populate tickets_utilises from LLM-provided
        # tickets_references so the front-end sources panel is not empty.
        llm_refs = result.get("tickets_references") or []
        if llm_refs and isinstance(llm_refs, list):
            utilises = result.get("tickets_utilises") or []
            if not utilises:
                result["tickets_utilises"] = [
                    r["ref"] for r in llm_refs
                    if isinstance(r, dict) and r.get("ref")
                ]
                logger.debug(
                    "[enforce] no hits — populated tickets_utilises from LLM refs: %s",
                    result["tickets_utilises"],
                )
        return result

    top_ref = hits[0].get("reference", "") if hits else ""

    # Build patch objects from ALL hits that have patches
    patch_objects: list[dict] = []
    seen_patches: set = set()
    for hit in hits:
        ref = hit.get("reference", "")
        for p in (hit.get("patches") or []):
            if p and p not in seen_patches:
                seen_patches.add(p)
                patch_objects.append({"patch": p, "ref": ref})
    if patch_objects:
        result["patches"] = patch_objects
    else:
        # Coerce LLM-generated patches to the expected [{patch, ref}] schema
        raw_patches = result.get("patches") or []
        if isinstance(raw_patches, list):
            coerced = []
            for item in raw_patches:
                if isinstance(item, dict) and "patch" in item:
                    coerced.append({"patch": str(item["patch"]), "ref": str(item.get("ref", ""))})
                elif isinstance(item, str) and item.strip():
                    coerced.append({"patch": item.strip(), "ref": top_ref})
            result["patches"] = coerced
        else:
            result["patches"] = []

    # Build tickets_references from retrieved hits for consultant reference
    ticket_refs: list[dict] = []
    for hit in hits:
        ref = hit.get("reference", "")
        if not ref:
            continue
        resolution_text = (hit.get("resolution") or "").strip()
        # Truncate long resolutions to a readable summary
        if len(resolution_text) > 300:
            resolution_text = resolution_text[:300] + "..."
        ticket_refs.append({
            "ref": ref,
            "titre": hit.get("title", "")[:200],
            "resolution_ticket": resolution_text,
            "patches": hit.get("patches") or [],
        })
    if ticket_refs:
        result["tickets_references"] = ticket_refs

    # Ensure the top ticket is listed first in tickets_utilises
    if top_ref:
        cited = result.get("tickets_utilises") or []
        if isinstance(cited, str):
            cited = [cited]
        cited = [r for r in cited if r != top_ref]
        result["tickets_utilises"] = [top_ref] + cited

    return result


class SopraHRAgent:
    """
    Stateful ReAct agent. One instance per CLI session.
    Conversation memory persists across questions within the session.
    """

    def __init__(
        self,
        team: str,
        db_dir: Path,
        bm25_dir: Path,
        index_dir: Path,
        verbose: bool = False,
        max_iterations: int = 7,
    ) -> None:
        self.verbose        = verbose
        self.max_iterations = max_iterations
        self._llm           = _LLM()
        self._bedrock       = get_bedrock_client()
        self._tools         = build_tool_registry(
            team=team,
            db_dir=db_dir,
            bm25_dir=bm25_dir,
            index_dir=index_dir,
            bedrock_client=self._bedrock,
        )
        self._history: list[dict] = []   # bounded conversation memory

    def reset(self) -> None:
        """Clears conversation history. Call between independent questions."""
        self._history.clear()

    def run(self, question: str) -> dict:
        """Runs the ReAct loop for one question and returns the JSON output dict."""
        scratchpad = ""
        tool_calls_made = 0  # track how many tool calls have been executed
        details_called = False  # track if DétailsTicket was called

        # Clear last-hits buffer so we always use this question's results
        _tool_store.last_hits = []

        # Force-conclude prompt injected when the agent stalls after searching
        _FORCE_CONCLUDE = (
            "\n[INSTRUCTION SYSTÈME] Tu as déjà effectué des recherches. "
            "Tu DOIS maintenant rédiger ta Final Answer JSON sans appeler d'autres outils. "
            "Si les tickets trouvés ne correspondent pas parfaitement, utilise les meilleurs disponibles.\n"
            "Thought: j'ai assez d'informations pour répondre\n"
            "Final Answer: "
        )

        # Base message list: system + memory
        base_messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        base_messages.extend(self._history)

        for iteration in range(self.max_iterations):

            # ── Force conclusion on last iteration ────────────────────
            is_last = (iteration == self.max_iterations - 1)
            if is_last and tool_calls_made > 0:
                # Inject a forced conclusion — ask LLM to produce only the JSON
                scratchpad += _FORCE_CONCLUDE
                messages = base_messages.copy()
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": scratchpad})
                response = self._llm.call(messages, max_tokens=4096)
                if self.verbose:
                    print(f"\n[iter {iteration + 1} FORCED]\n{response}\n{'─'*50}")
                result = _parse_json(response)
                result = _enforce_rank1_resolution(result)
                self._history.append({"role": "user",      "content": question})
                self._history.append({"role": "assistant",  "content": scratchpad + response})
                if len(self._history) > 8:
                    self._history = self._history[-8:]
                return result

            # Build messages for this iteration
            messages = base_messages.copy()
            messages.append({"role": "user", "content": question})
            if scratchpad:
                messages.append({"role": "assistant", "content": scratchpad})

            response = self._llm.call(messages)

            if self.verbose:
                print(f"\n[iter {iteration + 1}]\n{response}\n{'─'*50}")

            scratchpad += ("\n" if scratchpad else "") + response

            action_name, final_answer = _parse_step(response)

            # ── Final Answer ──────────────────────────────────────────
            if final_answer is not None:
                # Enforce at least 2 distinct hybrid searches before concluding —
                # single-shot retrieval was a major source of low ref_hit_rate.
                if tool_calls_made < 2 and not is_last:
                    scratchpad += (
                        "\n[INSTRUCTION SYSTÈME] Tu dois effectuer une SECONDE "
                        "RechercheHybride avec une reformulation différente AVANT "
                        "ta Final Answer. Reformule en variant au moins 3 mots-clés "
                        "(ajoute/retire des codes techniques, change l'angle métier).\n"
                        "Thought: je dois reformuler et chercher à nouveau\n"
                        "Action: RechercheHybride\nAction Input: "
                    )
                    continue
                # Enforce DétailsTicket before Final Answer — the resolution
                # quality depends on having the FULL resolution text, not the
                # truncated snippet from search results.
                if not details_called and not is_last and _tool_store.last_hits:
                    top_ref = _tool_store.last_hits[0].get("reference", "")
                    scratchpad += (
                        f"\n[INSTRUCTION SYSTÈME] Tu DOIS appeler DétailsTicket "
                        f"sur le ticket de rang 1 ({top_ref}) AVANT de rédiger ta "
                        f"Final Answer. La résolution doit se baser sur le texte "
                        f"complet du ticket, pas sur l'extrait de la recherche.\n"
                        f"Thought: je dois d'abord obtenir les détails complets du ticket {top_ref}\n"
                        f"Action: DétailsTicket\nAction Input: {top_ref}"
                    )
                    continue
                result = _parse_json(final_answer)
                result = _enforce_rank1_resolution(result)
                self._history.append({"role": "user",      "content": question})
                self._history.append({"role": "assistant",  "content": scratchpad})
                if len(self._history) > 8:
                    self._history = self._history[-8:]
                return result

            # ── Tool call ─────────────────────────────────────────────
            if action_name:
                # Truncate response: keep only up to Action Input line
                # to discard any hallucinated Observation / Final Answer
                action_input = _extract_action_input(response, question)
                ai_match = _ACTION_INPUT_RE.search(response)
                if ai_match:
                    truncated = response[:ai_match.end()].split("\n")
                    # Keep up to and including Action Input line
                    clean_response = "\n".join(
                        l for l in truncated
                        if not l.strip().startswith("Thought:") or l == truncated[0]
                    ).rstrip()
                    # Only take up to end of action input value
                    clean_response = response[:ai_match.start()] + "Action Input: " + action_input
                else:
                    clean_response = response

                # Replace raw response in scratchpad with truncated version
                scratchpad = scratchpad[:-(len(response))] + clean_response

                tool_fn = self._tools.get(action_name)
                if tool_fn is None:
                    observation = (
                        f"Outil inconnu: '{action_name}'. "
                        f"Disponibles: {', '.join(self._tools)}"
                    )
                else:
                    try:
                        observation = tool_fn(action_input)
                        tool_calls_made += 1
                        if action_name == "DétailsTicket":
                            details_called = True
                    except Exception as exc:
                        observation = f"Erreur outil: {exc}"

                if self.verbose:
                    print(f"[tool: {action_name}] input={action_input!r:.80}\n"
                          f"obs={observation[:200]}")

                # After 3 successful searches, nudge toward conclusion
                if tool_calls_made >= 3:
                    scratchpad += (
                        f"\nObservation: {observation}\n"
                        "Thought: j'ai effectué plusieurs recherches, je dois maintenant "
                        "formuler ma réponse finale avec les tickets trouvés.\n"
                    )
                else:
                    scratchpad += f"\nObservation: {observation}\nThought:"
                continue

            # LLM produced neither pattern — nudge toward conclusion if we have results
            if tool_calls_made > 0:
                scratchpad += (
                    "\n[INSTRUCTION] Tu dois maintenant écrire ta Final Answer JSON. "
                    "N'appelle plus d'outils.\nThought: je rédige ma Final Answer\nFinal Answer: "
                )
            else:
                scratchpad += "\nThought:"

        logger.warning("ReAct max_iterations reached without Final Answer")
        return {**_EMPTY_OUTPUT, "error": "Max iterations reached without Final Answer"}


# ─────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────

_KNOWN_FIELDS = ["analyse", "tickets_utilises", "cause_probable",
                 "reponse_lotus", "resolution", "patches"]


def _sanitize_json_string(s: str) -> str:
    """Escapes bare control characters (U+0000–U+001F) inside JSON string literals."""
    out: list[str] = []
    in_string   = False
    escape_next = False
    for ch in s:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            out.append(ch)
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ord(ch) < 0x20 and ch not in ("\t", "\n", "\r"):
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _repair_unescaped_quotes(s: str) -> str:
    """
    Escapes double-quote characters that appear *inside* JSON string values
    without a preceding backslash.

    Heuristic: if we are inside a string and encounter `"` but the very next
    non-whitespace character is NOT a JSON structural delimiter (`,` `}` `]`),
    the quote is content — escape it.  Genuine closing quotes ARE followed by
    one of those delimiters.
    """
    out: list[str] = []
    in_string   = False
    escape_next = False
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            if in_string:
                escape_next = True
            i += 1
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                # Peek ahead past whitespace
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                next_ch = s[j] if j < n else ""
                if next_ch in (",", "}", "]", ""):
                    in_string = False   # genuine closing quote
                    out.append(ch)
                else:
                    out.append('\\"')   # content quote — escape it
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _regex_extract(text: str) -> dict:
    """
    Last-resort field extraction without relying on json.loads.

    Matches each known field by name and captures its value using
    a non-greedy pattern anchored by a lookahead for the next field key
    or the closing brace.  Works even when the JSON contains structural
    errors such as unescaped quotes or stray commas.
    """
    result = dict(_EMPTY_OUTPUT)
    next_pat = "|".join(re.escape(f) for f in _KNOWN_FIELDS)

    for field in ("analyse", "cause_probable", "resolution", "reponse_lotus"):
        pat = (
            rf'"{re.escape(field)}"\s*:\s*"'
            rf'(.*?)'
            rf'"(?=\s*(?:,\s*"(?:{next_pat})"|[\]}}]))'
        )
        m = re.search(pat, text, re.DOTALL)
        if m:
            result[field] = m.group(1).replace('\\"', '"')

    for field in ("tickets_utilises",):
        pat = rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]'
        m = re.search(pat, text, re.DOTALL)
        if m:
            result[field] = re.findall(r'"([^"]*)"', m.group(1))

    # patches: may be string array or object array — try objects first
    patches_pat = r'"patches"\s*:\s*\[(.*?)\]'
    pm = re.search(patches_pat, text, re.DOTALL)
    if pm:
        inner = pm.group(1)
        obj_matches = re.findall(r'\{[^}]*"patch"\s*:\s*"([^"]*)"[^}]*"ref"\s*:\s*"([^"]*)"[^}]*\}', inner)
        if obj_matches:
            result["patches"] = [{"patch": p, "ref": r} for p, r in obj_matches]
        else:
            # fallback: plain string array (old format)
            strs = re.findall(r'"([^"]*)"', inner)
            result["patches"] = strs

    return result


def _parse_json(text: str) -> dict:
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return {**_EMPTY_OUTPUT, "error": f"No JSON object found in: {text[:200]}"}

    json_str = cleaned[start : end + 1]

    # Pass 1: as-is
    try:
        return {**_EMPTY_OUTPUT, **json.loads(json_str)}
    except json.JSONDecodeError:
        pass

    # Pass 2: sanitize illegal control characters inside strings
    try:
        return {**_EMPTY_OUTPUT, **json.loads(_sanitize_json_string(json_str))}
    except json.JSONDecodeError:
        pass

    # Pass 3: repair unescaped double-quote characters inside strings
    try:
        return {**_EMPTY_OUTPUT, **json.loads(_repair_unescaped_quotes(json_str))}
    except json.JSONDecodeError:
        pass

    # Pass 4: combine both repairs
    try:
        fixed = _repair_unescaped_quotes(_sanitize_json_string(json_str))
        return {**_EMPTY_OUTPUT, **json.loads(fixed)}
    except json.JSONDecodeError:
        pass

    # Pass 5: nuclear — strip ALL control characters
    try:
        nuclear = re.sub(r'(?<!\\)[\x00-\x1f]', " ", json_str)
        return {**_EMPTY_OUTPUT, **json.loads(nuclear)}
    except json.JSONDecodeError:
        pass

    # Pass 6: regex field-by-field extraction (no JSON parsing at all)
    extracted = _regex_extract(json_str)
    if any(extracted.get(k) for k in _KNOWN_FIELDS):
        return extracted

    return {**_EMPTY_OUTPUT, "error": "JSON parse error: all repair attempts failed"}


# ─────────────────────────────────────────────
# PUBLIC FACTORY + RUNNER  (same signatures as before)
# ─────────────────────────────────────────────

def create_rag_agent(
    team: str,
    db_dir: Path,
    bm25_dir: Path,
    index_dir: Path,
    verbose: bool = False,
) -> SopraHRAgent:
    """Creates and returns a ready-to-use SopraHRAgent."""
    return SopraHRAgent(
        team=team,
        db_dir=db_dir,
        bm25_dir=bm25_dir,
        index_dir=index_dir,
        verbose=verbose,
    )


def run_query(agent: SopraHRAgent, question: str) -> dict:
    """Runs a single question through the ReAct agent."""
    agent.reset()
    return agent.run(question)
