# RAG Support Agent for Sopra HR — Complete Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Setup and Installation](#3-setup-and-installation)
4. [Data Pipeline](#4-data-pipeline)
   - 4.1 [tickets_pipeline.py — Raw Export Parser](#41-tickets_pipelinepy--raw-export-parser)
   - 4.2 [txt_to_json.py — TXT to JSON Converter](#42-txt_to_jsonpy--txt-to-json-converter)
   - 4.3 [auto_pipeline.py — Automated Ingestion & Index Refresh](#43-auto_pipelinepy--automated-ingestion--index-refresh)
5. [Index Building](#5-index-building)
   - 5.1 [Vector Index (FAISS)](#51-vector-index-faiss)
   - 5.2 [BM25 Keyword Index](#52-bm25-keyword-index)
   - 5.3 [PageIndex (Vectorless, Claude-driven)](#53-pageindex-vectorless-claude-driven)
6. [Retrieval Layer](#6-retrieval-layer)
   - 6.1 [Hybrid Retrieval — hybrid_rag.py](#61-hybrid-retrieval--hybrid_ragpy)
   - 6.2 [agent_tools.py — Tool Registry](#62-agent_toolspy--tool-registry)
7. [ReAct Agent](#7-react-agent)
   - 7.1 [agent.py — Agent Core](#71-agentpy--agent-core)
   - 7.2 [cli.py — Command-Line Interface](#72-clipy--command-line-interface)
8. [API Server](#8-api-server)
   - 8.1 [api.py — FastAPI Server](#81-apipy--fastapi-server)
   - 8.2 [database.py — MongoDB Layer](#82-databasepy--mongodb-layer)
   - 8.3 [models.py — Pydantic Schemas](#83-modelspy--pydantic-schemas)
   - 8.4 [routes.py — MongoDB REST Router](#84-routespy--mongodb-rest-router)
9. [Authentication & RBAC](#9-authentication--rbac)
10. [Evaluation](#10-evaluation)
    - 10.1 [evaluate_agent.py](#101-evaluate_agentpy)
    - 10.2 [Other Evaluation Scripts](#102-other-evaluation-scripts)
11. [Audit Logging & Notifications](#11-audit-logging--notifications)
12. [Data Structures and Schemas](#12-data-structures-and-schemas)
13. [Environment Variables and Configuration](#13-environment-variables-and-configuration)
14. [Full End-to-End Workflow](#14-full-end-to-end-workflow)
15. [Directory Reference](#15-directory-reference)
16. [All Scripts Reference](#16-all-scripts-reference)

---

## 1. Project Overview

This project is a full-stack support search and answer-generation system built from historical Sopra HR technical support tickets originally stored in IBM Lotus Notes. It converts raw ticket exports into structured data, builds multiple retrieval indexes over resolved tickets, and exposes a ReAct-style agent that answers consultant questions by combining retrieval with LLM-generated expert recommendations. A React frontend provides a complete UI for querying, administration, pipeline management, evaluations, and audit logging.

### Support Teams

| Team | Scope | Ticket Prefix | Approx Count |
|------|-------|---------------|--------------|
| `DSN` | DSN declarations and social reporting issues | `FR W21xxxx` | ~4,000 |
| `Appli` | Functional and application-layer issues | `FR W21xxxx` | ~2,000 |
| `Outils` | Tools, generation, query, and system-side tooling | `FR W21xxxx` | ~2,000 |
| `AF` | Africa/MENA region — same domains for African clients | `AF W19xxxx`–`AF W25xxxx` | ~4,625 |

### RBAC — Role-Based Access Control

| Role | Permissions |
|------|-------------|
| `ADMIN` | Full access — pipeline ingestion, evaluations, audit logs, user management |
| `MANAGER` | Dashboard (all users), user list, team analytics |
| `TEAM_LEAD` | Dashboard (team-scoped), own queries, team members view |
| `CONSULTANT` | Dashboard (personal), own queries and history (default role) |

### LLM Backends

| Backend | Activation |
|---------|-----------|
| **Amazon Bedrock** (Claude 3 Haiku) | Default — uses AWS credentials |
| **OpenAI-compatible API** (Mistral, GPT, etc.) | Activated when `OPENAI_API_KEY` is set |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                      │
│                                                                     │
│  Auth ─── Navigation ─── MainContent ─── Dashboard ─── Pipeline    │
│                              │               │            │         │
│                         POST /query    GET /dashboard  POST /ingest │
└──────────────────────────────┼───────────────┼────────────┼─────────┘
                               │               │            │
┌──────────────────────────────┼───────────────┼────────────┼─────────┐
│                        API (FastAPI :8080)                           │
│                                                                     │
│  Auth (JWT) ── routes.py (/api/*) ── Dashboard Stats ── Pipeline   │
│                      │                                    │         │
│                  database.py                        auto_pipeline   │
│                  (MongoDB Motor)                         │          │
│                                                          ▼          │
│                              ┌─────────────────────────────────┐   │
│                              │         AGENT (ReAct Loop)       │   │
│                              │                                  │   │
│                              │  Question ──► LLM (Claude Haiku) │   │
│                              │       ▲              │           │   │
│                              │       │         Action:          │   │
│                              │  Observation   RechercheHybride  │   │
│                              │       │              │           │   │
│                              │       ◄──────────────┘           │   │
│                              │                                  │   │
│                              │  ┌─────────┐ ┌──────┐ ┌───────┐ │   │
│                              │  │  FAISS  │ │ BM25 │ │PageIdx│ │   │
│                              │  │ Vector  │ │Keywrd│ │ LLM   │ │   │
│                              │  └────┬────┘ └──┬───┘ └───┬───┘ │   │
│                              │       └─────────┼─────────┘     │   │
│                              │          Reciprocal Rank        │   │
│                              │             Fusion (RRF)        │   │
│                              └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Flow

```
Raw Lotus .txt export  ──or──  Simple .json/.txt upload (via Pipeline UI)
        │                                │
        └──► tickets_pipeline.py         │
                │                        │
                ▼                        ▼
        data/output/ (.txt)    ◄── auto_pipeline.py orchestrates
                │
        txt_to_json.py
                │
                ▼
        data/json/  (one JSON per ticket)
                │
        ┌───────┼───────────────────┐
        │       │                   │
        ▼       ▼                   ▼
   vector_rag  bm25_rag.py    vectorless_rag.py
   (FAISS)      (BM25)         (PageIndex)
        │       │                   │
        └───────┴─────────┬─────────┘
                          │
                   hybrid_rag.py (RRF)
                          │
                   agent_tools.py → agent.py (ReAct)
                          │
              ┌───────────┼───────────┐
              │           │           │
           cli.py      api.py   evaluate_agent.py
```

---

## 3. Setup and Installation

### Prerequisites

- Python 3.10+ (tested with 3.14)
- AWS credentials configured (for Bedrock), or `OPENAI_API_KEY` set
- MongoDB 6+ (local or Atlas) — required for the API server
- Node.js 18+ — required for the Frontend

### Install dependencies

```bash
cd RAG-SopraHR
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---------|---------|
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | `paraphrase-multilingual-mpnet-base-v2` embeddings (768-dim) |
| `rank-bm25` | BM25 keyword retrieval |
| `boto3` | AWS Bedrock access |
| `openai` | OpenAI-compatible API access (fallback) |
| `tiktoken` | Token counting |
| `python-dotenv` | `.env` file support |
| `fastapi` | API server framework |
| `uvicorn` | ASGI server |
| `motor` / `pymongo` | Async MongoDB driver |
| `python-jose[cryptography]` | JWT token encoding/decoding |
| `passlib[bcrypt]` | Password hashing (PBKDF2-SHA256) |
| `python-multipart` | Form data parsing |
| `apscheduler` | Scheduled tasks (weekly digest) |
| `reportlab` | PDF generation (optional) |

---

## 4. Data Pipeline

### 4.1 `tickets_pipeline.py` — Raw Export Parser

Splits a raw tab-separated IBM Lotus Notes export into individual per-ticket `.txt` files.

**Input:** A single `.txt` file exported from Lotus Notes (latin-1 encoding, tab-separated columns).
**Output:** One `.txt` file per ticket under `data/output/`, with YAML frontmatter and structured content blocks.

#### Output file format

```
---
reference: FR W210001
title: Problème de calcul DSN
version: HRA 9.1
support_team: DSN
closing_status_code: CP
patches: 178236;178237
---

[DESCRIPTION]
<client problem statement>

[RESOLUTION]
<best support reply extracted from conversation>

[CONVERSATION]
<chronological parsed exchanges>
```

#### Key functions

| Function | Description |
|----------|-------------|
| `detect_segments(lines)` | Finds ticket boundaries in the export |
| `parse_header(line)` | Extracts ticket metadata from a tab-separated header line |
| `extract_reply_blocks(block_lines)` | Collects all `Reply :` sections from a ticket block |
| `extract_resolution(block_lines)` | Selects the best technical reply (scores blocks for technical signal words) |
| `extract_patches(block)` | Extracts patch numbers ≥ 150000, excludes ticket reference numbers |
| `classify_team(block, closing_teamcode)` | Classifies ticket by team with a confidence score |

#### Team classification priority

1. **Closing team code** — highest confidence (0.97)
2. **H2/H1/H3 routing codes** in conversation body (weight 4)
3. **Module codes** (REGDSN, HRCT, etc.) (weight 2)
4. **Keywords** (weight 1)

---

### 4.2 `txt_to_json.py` — TXT to JSON Converter

Converts the per-ticket `.txt` files into structured JSON.

```bash
python scripts/txt_to_json.py --input data/output --output data/json [--team DSN]
```

#### Key functions

| Function | Description |
|----------|-------------|
| `parse_frontmatter(text)` | Extracts key:value pairs between `---` markers |
| `split_sections(text)` | Splits into DESCRIPTION / RESOLUTION / CONVERSATION |
| `parse_conversation(conv_text)` | Parses exchanges into structured ConversationEntry objects |
| `convert_all(input_dir, output_dir, team_filter)` | Batch converts an entire folder |

---

### 4.3 `auto_pipeline.py` — Automated Ingestion & Index Refresh

Orchestrates the full ingestion pipeline. Called by `/pipeline/ingest` API endpoint or CLI.

```bash
python scripts/auto_pipeline.py --raw-file data/raw/export.txt
python scripts/auto_pipeline.py --watch --interval 30   # watch mode
```

Steps: `txt_to_json` → `vector_rag.build_indexes` → `bm25_rag.build_bm25_index` → `vectorless_rag.build_pageindex`

---

## 5. Index Building

### 5.1 Vector Index (FAISS)

- **Model:** `paraphrase-multilingual-mpnet-base-v2` (768-dim, multilingual)
- **Index type:** FAISS `IndexFlatIP` on L2-normalized vectors (= cosine similarity)
- **Output:** `data/indexes/{team}/index.faiss`, `docstore.pkl`, `config.json`
- **Embeds:** Title + Description + Resolution + Patches (concatenated)

```bash
python scripts/vector_rag.py index --tickets data/cleaned --db data/indexes
```

### 5.2 BM25 Keyword Index

- **Algorithm:** BM25Okapi with French tokenizer
- **Output:** `data/bm25/{team}/bm25.json`

```bash
python scripts/bm25_rag.py build --json data/json --db data/bm25
```

#### Field boosting (via token repetition)

| Field | Repetitions | Rationale |
|-------|-------------|-----------|
| Title | 3× | Most reliable signal |
| Description | 2× | Full symptom text, key for matching |
| Resolution | 1× | Action taken |
| Patches | 1× | Exact patch numbers |
| Version | 1× | Product version |
| Error/product codes (extracted) | 3× | ORA-*, ZY*, FSW*, REGDSN, HRCT, etc. |
| Conversation | 2× | Technical codes in follow-up exchanges |

#### Tokenizer

- Lowercase, keep error codes intact (`ORA-00942` = one token)
- Keep alphanumeric tokens ≥ 2 chars, remove French + English stopwords

### 5.3 PageIndex (Vectorless, Claude-driven)

- **Hierarchy:** L1 (Team) → L2 (7–8 topic clusters) → L3 (one-line ticket summaries)
- **Output:** `data/pageindex/pageindex.json`

```bash
python scripts/vectorless_rag.py build --json data/json --index data/pageindex
```

#### Cluster taxonomy

| Team | Clusters |
|------|----------|
| **DSN** | Kit & Livraison, REGDSN & Erreurs, DSN Déclaration, URSSAF & SIREN, Espace DSN, Paramétrage DSN, Autre DSN |
| **Appli** | Paie & Calcul, Congés & Absences, DADS-U & N4DS, Kit & Patches, Pages Web & Design, Saisie & Formulaires, Traitement & Batch, Autre Appli |
| **Outils** | HRCT & Génération, Design Center, HRQuery & Requêtes, Erreurs Système, Kit & Livraison, Connexion & Services, HRAnalytics & Space, Autre Outils |

#### Two-step retrieval

1. Claude selects 1–3 relevant clusters from metadata
2. Claude scans L3 summaries (≤150 lines) → returns top-5 refs
3. Fallback: tries next-best clusters if < 2 results

---

## 6. Retrieval Layer

### 6.1 Hybrid Retrieval — `hybrid_rag.py`

Combines all three retrievers using **Reciprocal Rank Fusion (RRF)**.

#### RRF formula

$$\text{score}(d) = \sum_{i \in \{vec, bm25, pi\}} \frac{w_i}{k + \text{rank}_i(d)}$$

- **k** = 30 (RRF smoothing constant)
- **Weights:** Vector = 1.0, **BM25 = 1.5**, PageIndex = 1.0

#### Recency boost

$$\text{boost} = 1.0 + 0.8 \times e^{-\text{age\_days} / 730}$$

Recent tickets get up to +80% boost (half-life ~2 years).

#### Configuration constants

| Constant | Value |
|----------|-------|
| `TOP_K` | 7 (final results) |
| `RETRIEVAL_K` | 25 (candidates per retriever) |
| `RRF_K` | 30 |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` |
| `EMBEDDING_DIM` | 768 |

### 6.2 `agent_tools.py` — Tool Registry

| Tool | Description |
|------|-------------|
| `RechercheHybride` | 3-way RRF fusion + secondary BM25 on tech codes |
| `DétailsTicket` | Full ticket details: description (≤800 chars), resolution (≤1500 chars), patches |

#### Result formatting

- Top 2 hits: full resolution shown
- Remaining hits: resolution truncated to 300 chars
- All indexes loaded once into `_IndexStore` singleton

---

## 7. ReAct Agent

### 7.1 `agent.py` — Agent Core

Manual ReAct loop (no LangChain). Default LLM: Claude 3 Haiku via Bedrock.

#### Mandatory strategy (enforced by code)

1. `RechercheHybride` with exact terms from the problem
2. `RechercheHybride` with reformulated query (≥3 different keywords)
3. `DétailsTicket` on rank-1 ticket (code-enforced, not just prompted)
4. Final Answer with expert-generated resolution

#### Code-level enforcements

| Enforcement | Mechanism |
|-------------|-----------|
| Min 2 hybrid searches | Blocks Final Answer if `tool_calls_made < 2` |
| DétailsTicket mandatory | Blocks Final Answer if `details_called == False` |
| Force conclusion on iteration 7 | Injects `_FORCE_CONCLUDE` |
| Context overflow protection | Hard-cap at 140k chars, truncates oldest observations |
| String field coercion | All text fields forced to `str` type in post-processing |

#### Output schema

```json
{
  "analyse": "Problem summary",
  "tickets_utilises": ["FR W210001"],
  "cause_probable": "Root cause (agent's expertise)",
  "resolution": "Agent's OWN expert recommendation (not copied from tickets)",
  "tickets_references": [
    { "ref": "FR W210001", "titre": "...", "resolution_ticket": "...", "patches": ["178236"] }
  ],
  "patches": [{ "patch": "178236", "ref": "FR W210001" }],
  "reponse_lotus": "Professional client message"
}
```

#### Key design: Resolution vs Tickets References

- **`resolution`**: The LLM's own expert recommendation based on its technical knowledge. Not copied from tickets.
- **`tickets_references`**: Retrieved tickets provided as reference material for consultants. Auto-built from all hits by `_enforce_rank1_resolution()`.

### 7.2 `cli.py` — Command-Line Interface

```bash
python scripts/cli.py --team DSN                          # interactive
python scripts/cli.py --team Appli --question "..."       # single shot
python scripts/cli.py --team DSN --verbose                # show ReAct chain
```

---

## 8. API Server

### 8.1 `api.py` — FastAPI Server

```bash
python -m uvicorn scripts.api:app --port 8080 --reload
```

On startup: connects to MongoDB, preloads all 3 RAG agents, starts APScheduler.

#### Complete endpoint reference

##### Authentication

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/signup` | — | Create user → `{token, name, email, team, role}` |
| POST | `/auth/login` | — | Authenticate → `{token, name, email, team, role}` |
| POST | `/auth/logout` | user | End session |
| GET | `/auth/me` | user | Current user profile |
| POST | `/auth/change-password` | user | Change password (min 8 chars) |
| GET | `/auth/export-data` | user | Export all user data |
| POST | `/auth/delete-account` | user | Delete account |

##### RAG Query

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/query` | user | `{question, team}` → full agent output JSON |

##### Tickets

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/tickets/search` | user | Full-text search `?q=&team=&closed=&limit=&skip=` |
| GET | `/api/tickets/recent` | user | Recent tickets `?limit=20` |
| GET | `/api/tickets/{reference}` | user | By reference (e.g., `FR W210000`) |
| GET | `/api/tickets/id/{ticket_id}` | user | By MongoDB `_id` |
| POST | `/api/tickets` | ADMIN | Create ticket |
| POST | `/api/tickets/bulk` | ADMIN | Bulk create |
| PUT | `/api/tickets/{ticket_id}` | ADMIN | Update ticket |

##### Users

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/users/{user_id}` | user | User profile |
| GET | `/api/users` | role-scoped | List users |
| PUT | `/api/users/{user_id}` | user | Update profile |
| POST | `/api/users/{id}/favorites/{ticket_id}` | user | Add favorite |
| DELETE | `/api/users/{id}/favorites/{ticket_id}` | user | Remove favorite |

##### Analysis History

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/analysis` | user | Save analysis |
| GET | `/api/analysis` | user | User's history `?limit=&skip=` |
| PUT | `/api/analysis/{entry_id}` | user | Rate `{user_rating: 1-5, tags: []}` |
| DELETE | `/api/analysis/{entry_id}` | user | Delete entry |

##### Notifications

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/notifications` | user | List `?limit=&skip=&read=` |
| GET | `/notifications/unread-count` | user | Unread count |
| PUT | `/notifications/{id}/read` | user | Mark read |
| PUT | `/notifications/read-all` | user | Mark all read |

##### Admin

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/admin/users` | ADMIN/MANAGER | List all users |
| POST | `/admin/users` | ADMIN | Create user with role |
| PUT | `/admin/users/{email}` | ADMIN | Update user |
| DELETE | `/admin/users/{email}` | ADMIN | Soft-delete user |
| GET | `/admin/audit-logs` | ADMIN | Paginated audit `?limit=&skip=` |
| GET | `/admin/stats` | ADMIN | System statistics |

##### Pipeline

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/pipeline/ingest` | ADMIN | Upload & ingest `{file, mode}` |
| GET | `/pipeline/jobs/{job_id}` | ADMIN | Live job status |
| GET | `/pipeline/jobs` | ADMIN | List live jobs |
| GET | `/pipeline/history` | ADMIN | Past runs `?limit=50` |
| GET | `/pipeline/stats` | ADMIN | Statistics |
| POST | `/pipeline/reload-conversations` | ADMIN | Reload agent indexes |

##### Evaluation

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/eval/start` | ADMIN | Start eval `{team, samples, eval_type}` |
| GET | `/eval/jobs/{run_id}` | ADMIN | Live eval status |
| GET | `/eval/jobs` | ADMIN | List live jobs |
| GET | `/eval/history` | ADMIN | Past runs `?limit=50` |
| GET | `/eval/history/{run_id}` | ADMIN | Full results |

##### Dashboard & Health

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/dashboard/stats` | role-scoped | Metrics (scope varies by role) |
| GET | `/health` | — | `{status: "ok"}` |

### 8.2 `database.py` — MongoDB Layer

Async MongoDB via Motor. 7 collections, 38+ functions.

| Collection | Purpose | Key Indexes |
|------------|---------|-------------|
| `tickets` | Ticket records | `reference` (unique), `support_team`, `is_closed`, full-text |
| `users` | User accounts | `username` (unique), `email` (unique), `role`, `team` |
| `analysis_history` | Query results & ratings | `user_id`, `created_at` |
| `pipeline_jobs` | Ingestion runs | `job_id` (unique), `status` |
| `eval_runs` | Evaluation runs | `run_id` (unique), `eval_type` |
| `audit_logs` | System actions | `actor_email`, `action`, `created_at` |
| `notifications` | User notifications | `user_id`, `read`, `created_at` |

### 8.3 `models.py` — Pydantic Schemas

28 model classes, 26 audit action types, role/team/status enums. See [Section 12](#12-data-structures-and-schemas).

### 8.4 `routes.py` — MongoDB REST Router

Mounted at `/api`. 18 endpoints for tickets, users, analysis, favorites, team analytics.

---

## 9. Authentication & RBAC

| Property | Value |
|----------|-------|
| Algorithm | JWT HS256 |
| Expiry | 8 hours (configurable) |
| Password hashing | PBKDF2-SHA256 (passlib) |
| Token storage | `localStorage` (frontend) → `Authorization: Bearer` header |
| User storage | MongoDB `users` collection (primary) / `data/users.json` (fallback) |

### Dashboard scoping

| Role | Visible Data |
|------|-------------|
| CONSULTANT | Own queries and history |
| TEAM_LEAD | Team members' data |
| ADMIN / MANAGER | All users, all teams |

---

## 10. Evaluation

### 10.1 `evaluate_agent.py`

```bash
python scripts/evaluate_agent.py --team all --samples 30 --verbose
```

#### Metrics

| Metric | Description |
|--------|-------------|
| `ref_hit_rate` | % expected ticket ref cited |
| `resolution_rate` | % non-empty resolution |
| `effective_rate` | ref_hit OR overlap ≥ 0.25 |
| `mean_overlap` | Keyword overlap with expected answer |
| `mean_latency_s` | Seconds per question |
| `judge_pertinence` | LLM score (1–5): right problem? |
| `judge_completude` | LLM score (1–5): key details? |
| `judge_actionabilite` | LLM score (1–5): consultant can apply? |

Output: `output/eval_agent_<timestamp>.json`

### 10.2 Other Evaluation Scripts

| Script | Description |
|--------|-------------|
| `evaluate_hybrid.py` | Side-by-side: vector vs BM25 vs PageIndex vs hybrid |
| `evaluate_vectorless.py` | PageIndex-only evaluation |

---

## 11. Audit Logging & Notifications

### Audit logging

All actions logged to `audit_logs` via `_log_audit_sync()` (thread-safe with `asyncio.run_coroutine_threadsafe`). 26 action types tracked.

### Notifications

| Trigger | Recipients | Type |
|---------|------------|------|
| Query executed | User | info (prompt to rate) |
| Query failed | Team leads + managers | alert |
| Pipeline completed/failed | Admins | info/alert |
| Evaluation completed | Admins | info |
| Weekly digest | All active users | info |

---

## 12. Data Structures and Schemas

### Ticket (JSON / MongoDB)

```json
{
  "reference": "FR W210000",
  "title": "Souci avec la page FSWDRE01",
  "description": "...",
  "resolution": "...",
  "patches": ["178236"],
  "support_team": "Appli",
  "version": "HRA9.00",
  "system": "UNO",
  "is_closed": true,
  "closing_status_code": "CP",
  "closing_status_explanation": "Request processed (closed)",
  "conversation": [
    { "timestamp": "02/10/2015 09:57:21", "actor": "client", "action": "...", "content": "" }
  ]
}
```

### Agent output

```json
{
  "analyse": "Problem summary",
  "tickets_utilises": ["FR W210000"],
  "cause_probable": "Root cause",
  "resolution": "Agent's expert recommendation (not copied from tickets)",
  "tickets_references": [
    { "ref": "FR W210000", "titre": "...", "resolution_ticket": "...", "patches": ["178236"] }
  ],
  "patches": [{ "patch": "178236", "ref": "FR W210000" }],
  "reponse_lotus": "Professional client message"
}
```

### Users (data/users.json)

```json
{
  "admin@sopra.com": {
    "name": "Admin",
    "hashed_password": "$pbkdf2-sha256$...",
    "team": "DSN",
    "role": "ADMIN",
    "is_active": true
  }
}
```

---

## 13. Environment Variables and Configuration

```bash
# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=soprahr_rag

# JWT
JWT_SECRET=<random 64-char hex>
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=8

# API
API_HOST=0.0.0.0
API_PORT=8080

# AWS Bedrock (default LLM)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_DEFAULT_REGION=eu-west-1
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# OpenAI Fallback (replaces Bedrock if set)
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=gpt-4o-mini

# HuggingFace (optional)
HF_TOKEN=...
```

---

## 14. Full End-to-End Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # edit with credentials

# 3. Process raw tickets
python scripts/tickets_pipeline.py --input data/raw/export.txt --output data/output
python scripts/txt_to_json.py --input data/output --output data/json

# 4. Build indexes
python scripts/vector_rag.py index --tickets data/cleaned --db data/indexes
python scripts/bm25_rag.py build --json data/json --db data/bm25
python scripts/vectorless_rag.py build --json data/json --index data/pageindex

# 5. Import to MongoDB & start server
python import_tickets.py
python -m uvicorn scripts.api:app --port 8080 --reload

# 6. Start frontend
cd Frontend && npm install && npm run dev
```

---

## 15. Directory Reference

```
RAG-SopraHR/
├── scripts/
│   ├── agent.py                 # ReAct agent (7-iteration max)
│   ├── agent_tools.py           # Tool registry
│   ├── api.py                   # FastAPI server (:8080)
│   ├── database.py              # MongoDB CRUD (Motor)
│   ├── models.py                # Pydantic schemas (28 models)
│   ├── routes.py                # REST routes (/api/*)
│   ├── rag_utils.py             # Shared config & Bedrock client
│   ├── vector_rag.py            # FAISS indexing
│   ├── bm25_rag.py              # BM25 indexing
│   ├── hybrid_rag.py            # RRF fusion
│   ├── vectorless_rag.py        # PageIndex
│   ├── cli.py                   # CLI interface
│   ├── tickets_pipeline.py      # Raw → TXT
│   ├── txt_to_json.py           # TXT → JSON
│   ├── auto_pipeline.py         # Automated ingestion
│   ├── evaluate_agent.py        # Agent evaluation
│   ├── evaluate_hybrid.py       # Retrieval comparison
│   ├── evaluate_vectorless.py   # PageIndex evaluation
│   └── utils/                   # Validators, formatting
├── data/
│   ├── json/                    # Cleaned JSON tickets
│   ├── cleaned/                 # Pre-processed tickets
│   ├── indexes/{team}/          # FAISS indexes
│   ├── bm25/{team}/             # BM25 indexes
│   ├── pageindex/               # PageIndex hierarchy
│   ├── raw/                     # Raw Lotus exports
│   ├── output/                  # Canonical TXT files
│   └── users.json               # Flat user store
├── output/                      # Evaluation results
├── Test/test_queries.json       # 30-question test set
├── check_db.py                  # MongoDB health check
├── import_tickets.py            # Bulk import JSON → MongoDB
├── test_retrieval.py            # Quick retrieval diagnostic
└── requirements.txt
```

---

## 16. All Scripts Reference

| Script | Purpose |
|--------|---------|
| `agent.py` | ReAct agent — 7 iterations, dual LLM backend |
| `agent_tools.py` | Tool registry — RechercheHybride, DétailsTicket |
| `api.py` | FastAPI server — all endpoints, JWT auth, agent preloading |
| `database.py` | MongoDB async CRUD — 7 collections |
| `models.py` | Pydantic schemas — 28 models |
| `routes.py` | REST router — tickets, users, analysis |
| `rag_utils.py` | Shared config — constants, Bedrock client |
| `vector_rag.py` | FAISS indexing & retrieval |
| `bm25_rag.py` | BM25 indexing & retrieval |
| `hybrid_rag.py` | 3-way weighted RRF fusion |
| `vectorless_rag.py` | PageIndex — Claude-driven retrieval |
| `cli.py` | CLI — interactive REPL or single-shot |
| `tickets_pipeline.py` | Raw Lotus → canonical TXT |
| `txt_to_json.py` | TXT → structured JSON |
| `auto_pipeline.py` | Full ingestion orchestration |
| `evaluate_agent.py` | Agent evaluation with LLM judge |
| `evaluate_hybrid.py` | Retrieval method comparison |
| `evaluate_vectorless.py` | PageIndex-only evaluation |
| `check_db.py` | MongoDB health check |
| `import_tickets.py` | Bulk JSON → MongoDB import |
| `test_retrieval.py` | Quick retrieval diagnostic |
| `json_to_pdf.py` | PDF export |
| `clear_db.py` | Delete all MongoDB data |
