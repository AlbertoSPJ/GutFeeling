"""
02_prepare_rag_jsonl.py — JSON to JSONL Conversion for LlamaIndex
==================================================================
Converts the raw PubMed JSON corpus into a JSONL (JSON Lines) format
optimized for ingestion by LlamaIndex to build the vector index.

Input:
    pubmed_microbiome_metadata_only.json  — output of 01_pubmed_fetch.py

Output:
    pubmed_microbiome_rag.jsonl  — one JSON object per line, each containing:
        - content  : title + abstract + MeSH terms (the text to be embedded)
        - metadata : pmid, doi, pmc_id, journal, year, authors
                     (returned alongside retrieved chunks for citation display)

Why JSONL instead of JSON?
    LlamaIndex reads documents sequentially. JSONL (one record per line)
    allows streaming line by line without loading the entire corpus into
    memory at once — critical for a 56,000-article dataset.

Why combine title + abstract + MeSH terms as content?
    - Title: concise topic signal, highly weighted by the embedding model.
    - Abstract: the main semantic content for retrieval.
    - MeSH terms: controlled vocabulary that improves retrieval of articles
      that use different terminology for the same concept (e.g. "gut-brain
      axis" vs "enteric nervous system").

This script is the second step in the GutFeeling data pipeline:
    01_pubmed_fetch.py        → downloads raw metadata from PubMed
    02_prepare_rag_jsonl.py   ← this script
    GutFeeling.py             → builds the vector index and runs the app

Usage:
    $ python 02_prepare_rag_jsonl.py

Author: Alberto Sánchez-Pascuala
WBS Coding School, 2026
"""

import json


# ---------------------------------------------------------------------------
# 1. LOAD RAW CORPUS
# ---------------------------------------------------------------------------

input_file = "pubmed_microbiome_metadata_only.json"

with open(input_file, "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles from {input_file}")


# ---------------------------------------------------------------------------
# 2. CONVERT AND WRITE JSONL
# ---------------------------------------------------------------------------
# Each article is transformed into two components:
#   - content  : the text that will be chunked and embedded by LlamaIndex
#   - metadata : structured fields stored alongside each chunk for retrieval

output_file = "pubmed_microbiome_rag.jsonl"

with open(output_file, "w", encoding="utf-8") as f_out:
    for art in articles:

        # --- Build content string ---
        # Title and abstract are the primary semantic content.
        # MeSH terms are appended as a comma-separated list to enrich
        # the embedding with controlled vocabulary signal.
        title      = art.get("title", "")
        abstract   = art.get("abstract", "")
        mesh_terms = ", ".join(art.get("mesh_terms", []))
        content    = f"{title}. {abstract} MeSH: {mesh_terms}"

        # --- Build metadata dict ---
        # Authors are serialized as "Last, Fore; Last, Fore; ..." for compact storage.
        # PMID, DOI, and journal are used to generate citation links in the Streamlit UI.
        authors_list = art.get("authors", [])
        authors_str  = "; ".join([
            f"{a.get('last_name', '')}, {a.get('fore_name', '')}"
            for a in authors_list
        ])

        metadata = {
            "pmid":    art.get("pmid",    ""),
            "doi":     art.get("doi",     ""),
            "pmc_id":  art.get("pmc_id",  ""),
            "journal": art.get("journal", ""),
            "year":    art.get("year",    ""),
            "authors": authors_str
        }

        # --- Assemble JSONL record ---
        # Each line is a self-contained JSON object with a unique id,
        # the embeddable content, and the citation metadata.
        obj = {
            "id":       art.get("pmid", ""),
            "content":  content,
            "metadata": metadata
        }

        # Write one JSON object per line (JSONL format)
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Done. RAG-ready JSONL saved to: {output_file}")
