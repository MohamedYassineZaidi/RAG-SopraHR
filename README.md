# RAG-Based Support Assistant for Sopra HR Tickets

## Overview

AI-powered support assistant for **Sopra HR** built on historical **IBM Lotus support tickets**. The system uses **Retrieval-Augmented Generation (RAG)** to help support agents quickly find similar past issues and reuse validated resolutions.

Three retrieval strategies are benchmarked and combined:

| Strategy | Description |
|---|---|
| **Vector RAG** | FAISS + `paraphrase-multilingual-mpnet-base-v2` embeddings |
| **BM25 RAG** | Keyword-frequency retrieval, best for error codes and patch numbers |
| **Vectorless / PageIndex** | LLM-navigated 3-level index (no embeddings) via AWS Bedrock |
| **Hybrid RAG** | Reciprocal Rank Fusion (RRF) of all three systems |

---

## Project Structure

```
.
├── scripts/
│   ├── tickets_pipeline.py        # Split raw .txt export → one .txt per ticket
│   ├── tickets_to_json_claude.py  # Structured JSON extraction via Claude (Bedrock)
│   ├── tickets_to_json_aws.py     # Alternative extraction via AWS Titan
│   ├── tickets_to_json_mistral.py # Alternative extraction via Mistral
│   ├── ticket_parser.py           # Shared parsing utilities
│   ├── txt_to_json.py             # Rule-based .txt → JSON converter
│   ├── rag_assistant.py           # Vector RAG: index + query + evaluate
│   ├── vectorless_rag.py          # PageIndex RAG: build + query + evaluate
│   ├── bm25_rag.py                # BM25 RAG: build + query + evaluate
│   ├── hybrid_rag.py              # Hybrid RRF fusion: query + evaluate
│   ├── evaluate_hybrid.py         # Full evaluation suite (all metrics, all teams)
│   ├── evaluate_vectorless.py     # Standalone vectorless evaluation
│   ├── rag_utils.py               # Shared constants, cluster taxonomy, Bedrock client
│   ├── json_to_pdf.py             # Export ticket JSON to PDF
│   └── check_imports.py           # Environment dependency check
│
├── data/
│   ├── json/                      # Structured ticket JSON files (source of truth)
│   ├── analysis_json/             # Per-ticket deep analysis JSON
│   ├── pageindex/                 # PageIndex files + eval results
│   │   ├── pageindex.json
│   │   ├── {team}_l3.txt
│   │   └── eval_*.json
│   ├── bm25/                      # BM25 eval results (pickle indexes are gitignored)
│   │   └── eval_bm25_{team}.json
│   ├── tickets_index.csv          # Master ticket index
│   └── eval_results.json          # Vector RAG evaluation results
│
├── output/
│   └── eval_set.json              # Evaluation question set
│
├── Test/
│   └── test_queries.json          # Manual test queries
│
├── .env                           # AWS credentials (gitignored)
├── .gitignore
└── README.md
```

---

## Setup

### Requirements

```bash
pip install faiss-cpu sentence-transformers boto3 python-dotenv numpy rank-bm25 tiktoken
```

### AWS credentials

Create a `.env` file at the project root:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-1
```

---

## Data Pipeline

```
Raw .txt export (IBM Lotus)
        │
        ▼
tickets_pipeline.py        ← splits into one .txt per ticket
        │
        ▼
tickets_to_json_claude.py  ← extracts structured JSON via LLM
        │
        ▼
  data/json/*.json          ← source of truth for all RAG systems
        │
   ┌────┴────────────┬──────────────────┐
   ▼                 ▼                  ▼
rag_assistant.py  bm25_rag.py    vectorless_rag.py
(FAISS index)    (BM25 index)    (PageIndex)
   └────┬────────────┴──────────────────┘
        ▼
   hybrid_rag.py   ← RRF fusion of all three
```

---

## Usage

### Vector RAG

```bash
# Build FAISS indexes
python scripts/rag_assistant.py index --json data/json --db data/indexes

# Query
python scripts/rag_assistant.py query --db data/indexes --team DSN
python scripts/rag_assistant.py query --db data/indexes --team Appli --question "Erreur ORA-00942 REGDSN"

# Evaluate
python scripts/rag_assistant.py evaluate --db data/indexes --json data/json --team DSN --samples 50
```

### BM25 RAG

```bash
# Build
python scripts/bm25_rag.py build --json data/json --db data/bm25

# Query
python scripts/bm25_rag.py query --db data/bm25 --team DSN --question "ORA-00942 REGDSN table inexistante"

# Evaluate
python scripts/bm25_rag.py evaluate --db data/bm25 --json data/json --team DSN
```

### PageIndex (Vectorless) RAG

```bash
# Build
python scripts/vectorless_rag.py build --json data/json --index data/pageindex

# Query
python scripts/vectorless_rag.py query --index data/pageindex --team DSN --question "Erreur ORA-00942"

# Evaluate
python scripts/vectorless_rag.py evaluate --index data/pageindex --json data/json --team DSN
```

### Hybrid RAG

```bash
# Query
python scripts/hybrid_rag.py query --index data/pageindex --db data/indexes --team DSN

# Single question
python scripts/hybrid_rag.py query --index data/pageindex --db data/indexes --team DSN \
    --question "Erreur ORA-00942 lors du lancement REGDSN"

# Full evaluation (all metrics, all teams)
python scripts/evaluate_hybrid.py \
    --index data/pageindex \
    --db    data/indexes \
    --bm25  data/bm25 \
    --json  data/json \
    --team  Appli \
    --samples 50
```

---

## Evaluation Results

Results are stored as JSON in `data/pageindex/`, `data/bm25/`, and `data/eval_results.json`.

The `evaluate_hybrid.py` script reports:
- **Recall@1 / @3 / @5** — exact ticket self-retrieval per system
- **Cluster Routing Accuracy** — PageIndex cluster assignment correctness
- **Effective Recall** — useful misses counted via resolution keyword overlap
- **Per-system comparison table** — vector vs BM25 vs PageIndex vs Hybrid

---

## Teams

Tickets are segmented into three support teams:

| Team | Domain |
|------|--------|
| **DSN** | Déclaration Sociale Nominative |
| **Appli** | Application / functional issues |
| **Outils** | Tools and integrations |

---
