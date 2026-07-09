#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_agent.py
=================
End-to-end evaluation of the ReAct agent.

For each question in Test/test_queries.json this script:
  1. Runs the full agent (tools + LLM loop).
  2. Checks whether the expected ticket ref appears in tickets_utilises.
  3. Measures keyword overlap between the agent's resolution and the
     expected_answer title (same approach as evaluate_hybrid.py).
  4. Prints a per-item row and a final summary table.
  5. Saves all results to output/eval_agent_<timestamp>.json.

Metrics
-------
  ref_hit_rate      : % of questions where expected_ticket_ref was cited
  resolution_rate   : % of questions where resolution field is non-empty
  mean_overlap      : average keyword overlap score (0-1)
  effective_rate    : ref_hit OR overlap >= threshold
  mean_latency_s    : average seconds per question

Usage
-----
  python scripts/evaluate_agent.py
  python scripts/evaluate_agent.py --team DSN
  python scripts/evaluate_agent.py --team DSN --samples 5
  python scripts/evaluate_agent.py --team all --verbose
"""

import re
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from agent import create_rag_agent, run_query, _LLM

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

_ROOT      = _SCRIPTS.parent
_TEST_FILE = _ROOT / "Test" / "test_queries.json"
_OUT_DIR   = _ROOT / "output"
_DB_DIR    = _ROOT / "data" / "indexes"
_BM25_DIR  = _ROOT / "data" / "bm25"
_IDX_DIR   = _ROOT / "data" / "pageindex"

OVERLAP_THRESHOLD = 0.25   # same as evaluate_hybrid.py


# ─────────────────────────────────────────────
# LLM-AS-JUDGE
# ─────────────────────────────────────────────

_JUDGE_PROMPT = """\
Tu es un évaluateur expert pour un système de support technique Sopra HR.

On te donne :
- La QUESTION posée par un utilisateur
- La RÉPONSE produite par l'agent RAG
- Le TITRE du ticket attendu (indice de ce que la bonne réponse devrait couvrir)

Évalue la RÉPONSE sur ces 3 critères (note de 1 à 5 chacun) :

1. **Pertinence** : La réponse traite-t-elle le bon problème décrit dans la question ?
   1=hors-sujet, 2=partiellement lié, 3=bon sujet mais vague, 4=correct et ciblé, 5=parfaitement ciblé

2. **Complétude** : La réponse contient-elle les éléments clés (patches, étapes, codes erreur, tables) ?
   1=aucun détail, 2=très peu, 3=quelques éléments, 4=la plupart, 5=tous les éléments nécessaires

3. **Actionabilité** : Un technicien pourrait-il résoudre le problème en suivant cette réponse ?
   1=inutilisable, 2=indices vagues, 3=direction correcte mais incomplète, 4=applicable avec hypothèses, 5=directement applicable

Réponds UNIQUEMENT au format JSON (pas de markdown, pas de commentaires) :
{{"pertinence": <1-5>, "completude": <1-5>, "actionabilite": <1-5>, "justification": "<1 phrase>"}}"""


def _judge_answer(llm: "_LLM", question: str, agent_output: dict, expected_title: str) -> dict:
    """Use the LLM to score the agent's answer on 3 dimensions."""
    resolution = str(agent_output.get("resolution") or "")
    analyse    = str(agent_output.get("analyse") or "")
    patches    = agent_output.get("patches") or []
    answer_text = f"Analyse: {analyse}\nRésolution: {resolution}"
    if patches:
        answer_text += f"\nPatches: {', '.join(str(p) for p in patches)}"

    messages = [
        {"role": "system", "content": _JUDGE_PROMPT},
        {"role": "user",   "content": (
            f"QUESTION:\n{question}\n\n"
            f"RÉPONSE DE L'AGENT:\n{answer_text}\n\n"
            f"TITRE DU TICKET ATTENDU:\n{expected_title}"
        )},
    ]

    try:
        raw = llm.call(messages, max_tokens=300)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        # Robust JSON extraction: locate the first {...} block.
        # The judge LLM sometimes prefixes "Voici l'évaluation:" or similar.
        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            import re as _re
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if not m:
                raise
            block = m.group(0)
            try:
                scores = json.loads(block)
            except json.JSONDecodeError:
                # Last-resort key/value salvage for unquoted keys
                fixed = _re.sub(
                    r'([{,]\s*)([A-Za-z_][A-Za-z_0-9]*)\s*:',
                    r'\1"\2":',
                    block,
                )
                scores = json.loads(fixed)
        return {
            "pertinence":    int(scores.get("pertinence", 0)),
            "completude":    int(scores.get("completude", 0)),
            "actionabilite": int(scores.get("actionabilite", 0)),
            "justification": str(scores.get("justification", "")),
            "judge_mean":    round(
                (int(scores.get("pertinence", 0))
                 + int(scores.get("completude", 0))
                 + int(scores.get("actionabilite", 0))) / 3, 2
            ),
        }
    except Exception as e:
        return {
            "pertinence": 0, "completude": 0, "actionabilite": 0,
            "justification": f"Judge error: {e}", "judge_mean": 0.0,
        }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en",
    "est", "au", "aux", "ce", "se", "sa", "son", "sur", "par", "pour",
    "dans", "avec", "ne", "pas", "que", "qui", "ou", "a", "à",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _overlap(agent_text: str, expected_title: str) -> float:
    """Keyword overlap between agent resolution text and expected_answer title."""
    exp_kw = _keywords(expected_title)
    if not exp_kw:
        return 0.0
    agent_kw = _keywords(agent_text)
    return len(exp_kw & agent_kw) / len(exp_kw)


def _normalize_ref(ref: str) -> str:
    """Normalize ticket ref for comparison: 'FR W210006' → 'FRW210006'."""
    return re.sub(r"[\s_\-]", "", ref).upper()


def _ref_hit(result: dict, expected_ref: str) -> bool:
    """True if expected_ref appears in tickets_utilises (fuzzy-normalized)."""
    cited = result.get("tickets_utilises") or []
    if isinstance(cited, str):
        cited = [cited]
    norm_expected = _normalize_ref(expected_ref)
    return any(_normalize_ref(str(r)) == norm_expected for r in cited)


def _load_test_items(path: Path, team: str | None, samples: int | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("eval_items", data) if isinstance(data, dict) else data
    if team and team.lower() != "all":
        items = [i for i in items if i.get("team", "").lower() == team.lower()]
    if samples:
        items = items[:samples]
    return items


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate(team_filter: str | None, samples: int | None, verbose: bool) -> dict:
    items = _load_test_items(_TEST_FILE, team_filter, samples)
    if not items:
        print(f"[!] No items found for team='{team_filter}'")
        return {}

    print(f"\n{'─'*70}")
    print(f"  Agent evaluation  |  {len(items)} questions"
          + (f"  |  team: {team_filter}" if team_filter else ""))
    print(f"{'─'*70}\n")

    # LLM judge — reuses the same backend (Bedrock or OpenAI) as the agent
    judge_llm = _LLM()

    # Group by team so we load indexes once per team
    teams_needed: dict[str, list[dict]] = {}
    for item in items:
        t = item.get("team", "DSN")
        teams_needed.setdefault(t, []).append(item)

    results = []

    for team, team_items in teams_needed.items():
        print(f"[+] Loading agent for team: {team} …")
        try:
            agent = create_rag_agent(
                team=team,
                db_dir=_DB_DIR,
                bm25_dir=_BM25_DIR,
                index_dir=_IDX_DIR,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[!] Failed to load agent for team {team}: {e}")
            for item in team_items:
                results.append(_error_row(item, str(e)))
            continue

        for item in team_items:
            qid      = item["id"]
            question = item["question"]
            exp_ref  = item.get("expected_ticket_ref", "")
            exp_ans  = item.get("expected_answer", "")

            print(f"  [{qid}] …", end="", flush=True)
            agent.reset()   # clear history between questions
            t0 = time.time()

            try:
                result   = run_query(agent, question)
                latency  = round(time.time() - t0, 2)

                hit      = _ref_hit(result, exp_ref)
                resolution = str(result.get("resolution") or "")
                overlap  = _overlap(
                    resolution + " " + str(result.get("analyse") or ""),
                    exp_ans,
                )
                has_res  = bool(resolution.strip())
                effective = hit or overlap >= OVERLAP_THRESHOLD

                row = {
                    "id":             qid,
                    "team":           team,
                    "question":       question,
                    "expected_ref":   exp_ref,
                    "expected_answer": exp_ans,
                    "cited_refs":     result.get("tickets_utilises") or [],
                    "ref_hit":        hit,
                    "has_resolution": has_res,
                    "overlap":        round(overlap, 4),
                    "effective":      effective,
                    "latency_s":      latency,
                    "agent_output":   result,
                    "error":          None,
                }

                # LLM-as-judge scoring
                judge = _judge_answer(judge_llm, question, result, exp_ans)
                row["judge"] = judge

                status = "✅" if hit else ("~" if effective else "✗")
                jm = judge["judge_mean"]
                print(f" {status}  ref_hit={hit}  overlap={overlap:.2f}"
                      f"  judge={jm:.1f}/5  ({latency}s)")

            except Exception as e:
                latency = round(time.time() - t0, 2)
                print(f" ERROR: {e}")
                if verbose:
                    traceback.print_exc()
                row = _error_row(item, str(e), latency)

            results.append(row)

    return _summarize(results)


def _error_row(item: dict, error: str, latency: float = 0.0) -> dict:
    return {
        "id":             item["id"],
        "team":           item.get("team", ""),
        "question":       item["question"],
        "expected_ref":   item.get("expected_ticket_ref", ""),
        "expected_answer": item.get("expected_answer", ""),
        "cited_refs":     [],
        "ref_hit":        False,
        "has_resolution": False,
        "overlap":        0.0,
        "effective":      False,
        "latency_s":      latency,
        "agent_output":   {},
        "error":          error,
        "judge":          {
            "pertinence": 0, "completude": 0, "actionabilite": 0,
            "justification": "Skipped (error)", "judge_mean": 0.0,
        },
    }


def _summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    ref_hits    = sum(r["ref_hit"]        for r in results)
    has_res     = sum(r["has_resolution"] for r in results)
    effectives  = sum(r["effective"]      for r in results)
    mean_ov     = sum(r["overlap"]        for r in results) / n
    mean_lat    = sum(r["latency_s"]      for r in results) / n
    errors      = sum(1 for r in results if r["error"])

    # Judge aggregates
    judged = [r for r in results if r.get("judge", {}).get("judge_mean", 0) > 0]
    n_judged = len(judged) or 1  # avoid division by zero
    mean_pertinence   = sum(r["judge"]["pertinence"]    for r in judged) / n_judged
    mean_completude   = sum(r["judge"]["completude"]    for r in judged) / n_judged
    mean_actionabilite = sum(r["judge"]["actionabilite"] for r in judged) / n_judged
    mean_judge        = sum(r["judge"]["judge_mean"]    for r in judged) / n_judged

    # Per-team breakdown
    teams: dict[str, dict] = {}
    for r in results:
        t = r["team"]
        if t not in teams:
            teams[t] = {"n": 0, "hits": 0, "eff": 0, "overlap": 0.0, "judge_sum": 0.0, "judge_n": 0}
        teams[t]["n"]       += 1
        teams[t]["hits"]    += r["ref_hit"]
        teams[t]["eff"]     += r["effective"]
        teams[t]["overlap"] += r["overlap"]
        jm = r.get("judge", {}).get("judge_mean", 0)
        if jm > 0:
            teams[t]["judge_sum"] += jm
            teams[t]["judge_n"]  += 1

    team_summary = {
        t: {
            "questions":      v["n"],
            "ref_hit_rate":   round(v["hits"]    / v["n"], 4),
            "effective_rate": round(v["eff"]     / v["n"], 4),
            "mean_overlap":   round(v["overlap"] / v["n"], 4),
            "mean_judge":     round(v["judge_sum"] / max(v["judge_n"], 1), 2),
        }
        for t, v in teams.items()
    }

    metrics = {
        "total_questions":  n,
        "ref_hit_rate":     round(ref_hits   / n, 4),
        "resolution_rate":  round(has_res    / n, 4),
        "effective_rate":   round(effectives / n, 4),
        "mean_overlap":     round(mean_ov,       4),
        "mean_latency_s":   round(mean_lat,      2),
        "errors":           errors,
        "judge_pertinence":   round(mean_pertinence,   2),
        "judge_completude":   round(mean_completude,   2),
        "judge_actionabilite": round(mean_actionabilite, 2),
        "judge_mean":         round(mean_judge,         2),
    }

    _print_summary(metrics, team_summary, results)

    return {
        "metadata": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "test_file":    str(_TEST_FILE),
            "overlap_threshold": OVERLAP_THRESHOLD,
        },
        "metrics":      metrics,
        "by_team":      team_summary,
        "results":      results,
    }


def _print_summary(metrics: dict, team_summary: dict, results: list[dict]) -> None:
    W = 70
    print(f"\n{'═'*W}")
    print("  EVALUATION SUMMARY")
    print(f"{'─'*W}")
    print(f"  Questions       : {metrics['total_questions']}")
    print(f"  Ref hit rate    : {metrics['ref_hit_rate']:.1%}  "
          f"(expected ticket cited in agent output)")
    print(f"  Effective rate  : {metrics['effective_rate']:.1%}  "
          f"(ref hit OR overlap >= {OVERLAP_THRESHOLD})")
    print(f"  Resolution rate : {metrics['resolution_rate']:.1%}  "
          f"(non-empty resolution field)")
    print(f"  Mean overlap    : {metrics['mean_overlap']:.2f}")
    print(f"  Mean latency    : {metrics['mean_latency_s']:.1f}s / question")
    if metrics["errors"]:
        print(f"  Errors          : {metrics['errors']} ⚠️")
    print(f"{'─'*W}")
    print(f"  LLM JUDGE (1-5 scale):")
    print(f"    Pertinence    : {metrics.get('judge_pertinence', 0):.2f}")
    print(f"    Complétude    : {metrics.get('judge_completude', 0):.2f}")
    print(f"    Actionabilité : {metrics.get('judge_actionabilite', 0):.2f}")
    print(f"    Moyenne       : {metrics.get('judge_mean', 0):.2f}")

    if len(team_summary) > 1:
        print(f"\n{'─'*W}")
        print(f"  {'Team':<10} {'Questions':>10} {'Ref Hit':>10} "
              f"{'Effective':>10} {'Overlap':>10} {'Judge':>8}")
        print(f"  {'─'*8:<10} {'─'*8:>10} {'─'*8:>10} {'─'*8:>10} {'─'*8:>10} {'─'*6:>8}")
        for team, v in team_summary.items():
            print(f"  {team:<10} {v['questions']:>10} "
                  f"{v['ref_hit_rate']:>9.1%} "
                  f"{v['effective_rate']:>9.1%} "
                  f"{v['mean_overlap']:>10.2f}"
                  f"{v.get('mean_judge', 0):>8.2f}")

    # Misses
    misses = [r for r in results if not r["ref_hit"] and not r["error"]]
    if misses:
        print(f"\n{'─'*W}")
        print("  MISSES (ref not cited):")
        for r in misses:
            ov_flag = " (overlap ok)" if r["overlap"] >= OVERLAP_THRESHOLD else ""
            print(f"    [{r['id']}] expected {r['expected_ref']}"
                  f"  cited={r['cited_refs']}{ov_flag}")

    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def _save(report: dict, team_filter: str | None) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{team_filter.lower()}" if team_filter and team_filter != "all" else ""
    path = _OUT_DIR / f"eval_agent{suffix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Results saved → {path.relative_to(_ROOT)}\n")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end agent evaluation")
    parser.add_argument("--team",    default="all",
                        help="Team to evaluate: DSN | Appli | Outils | all")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max questions per team (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show agent reasoning steps")
    args = parser.parse_args()

    _VALID_TEAMS = {"DSN", "Appli", "Outils", "all"}
    if args.team not in _VALID_TEAMS:
        parser.error(f"--team must be one of {sorted(_VALID_TEAMS)}, got: {args.team!r}")

    team = None if args.team.lower() == "all" else args.team

    report = evaluate(team, args.samples, args.verbose)
    if report:
        _save(report, args.team)


if __name__ == "__main__":
    main()
