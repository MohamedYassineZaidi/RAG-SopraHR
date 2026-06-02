#!/usr/bin/env python3
"""Quick diagnostic: test retrieval for FR W151124."""
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
from bm25_rag import load_bm25_index, bm25_retrieve
from sentence_transformers import SentenceTransformer
from hybrid_rag import load_vector_index, vector_retrieve, reciprocal_rank_fusion
from rag_utils import RRF_K

team = 'Appli'
target = 'FR W151124'

# User query similar to what the eval would send
user_query = (
    "cotisation saisie en element variable en base genere un paiement "
    "retenue annulant les montants sur le bulletin du mois suivant"
)

print(f"Query: {user_query}")
print(f"Target: {target}")
print()

# BM25
idx, docs = load_bm25_index(Path('data/bm25'), team)
bm25_hits = bm25_retrieve(user_query, idx, docs, top_k=10)
print("=== BM25 top 5 ===")
for h in bm25_hits[:5]:
    print(f"  #{h['rank']} {h['reference']:15s} {h.get('title','')[:60]}")
bm25_found = target in [h['reference'] for h in bm25_hits]
print(f"  → {target} in top 10: {bm25_found}")
print()

# Vector
print("Loading vector index...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
vi, vd = load_vector_index(Path('data/indexes'), team)
vec_hits = vector_retrieve(user_query, vi, vd, model, top_k=10)
print("=== VECTOR top 5 ===")
for h in vec_hits[:5]:
    print(f"  #{h['rank']} {h['reference']:15s} sim={h['similarity']:.4f} {h.get('title','')[:50]}")
vec_found = target in [h['reference'] for h in vec_hits]
print(f"  → {target} in top 10: {vec_found}")
print()

# RRF fusion
fused = reciprocal_rank_fusion(vec_hits, [], bm25_hits, k=RRF_K)
print("=== HYBRID (vec+bm25) top 7 ===")
for h in fused[:7]:
    print(f"  #{h['rank']} {h['reference']:15s} rrf={h['rrf_score']:.4f} src={h.get('source','')} {h.get('title','')[:40]}")
hybrid_found = target in [h['reference'] for h in fused[:7]]
print(f"  → {target} in top 7: {hybrid_found}")
