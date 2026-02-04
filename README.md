# RAG-Based Support Assistant for Sopra HR Tickets

## 📌 Overview

This project focuses on building an **AI-powered support assistant** for **Sopra HR** by leveraging historical **IBM Lotus support tickets**.

The system uses **Retrieval-Augmented Generation (RAG)** to analyze archived tickets and assist support agents by:
- Finding similar past issues
- Reusing validated resolutions
- Reducing resolution time and repetitive work

This project is designed for **internal support optimization** and **knowledge reuse**.

---

## 🎯 Objectives

- Centralize historical ticket knowledge
- Improve accuracy and speed of ticket resolution
- Reduce dependency on individual expertise
- Build a reusable and extensible RAG architecture

---

## 🧠 RAG Strategy

The project explores and compares multiple retrieval approaches:

### 1. Vector-Based RAG
- Embedding generation
- Semantic similarity search
- Vector databases (e.g. FAISS)

### 2. Vectorless / Structured RAG
- Page- or document-level indexing
- Metadata-driven retrieval
- Deterministic filtering

### 3. Hybrid RAG (Target)
- Combines semantic search and structured retrieval
- Improves recall and precision for support use cases

---

## 📁 Project Structure
RAG/
│
├── .venv/                  # Python virtual environment (ignored)
│
├── data/
│   ├── raw/                # Original ticket exports (ignored)
│   ├── output/             # Cleaned and processed data (ignored)
│   └── tests/              # Small test datasets
│
├── output/                 # Generated artifacts (ignored)
│   ├── ticket_archive_.pdf
│   ├── ticket_archive_.html
│
├── scripts/
│   ├── check_imports.py
│   ├── extract_closing_counts.py
│   ├── generate_minimal.py
│   ├── md_to_pdf.py
│
├── tickets_pipeline.py     # Main processing pipeline
├── template.css            # Styling for generated documents
├── README.md
└── .gitignore
## 🔄 Data Pipeline

1. **Ingestion**
   - Raw tickets exported from IBM Lotus

2. **Cleaning & Normalization**
   - Text cleanup
   - Field standardization
   - Minimal, LLM-friendly formatting

3. **Document Generation**
   - Markdown → HTML → PDF
   - One document per ticket

4. **Indexing**
   - Embeddings
   - Metadata extraction (product, module, issue type, etc.)

5. **Retrieval**
   - Semantic search
   - Structured filtering
   - Hybrid ranking logic

---
