"""
01_pubmed_fetch.py — PubMed Data Retrieval
==========================================
Downloads metadata for all microbiota-related articles from PubMed
via the NCBI Entrez API (Biopython), covering the period 1980–2025.

Output:
    pubmed_microbiome_metadata_only.json — one JSON object per article,
    containing: pmid, doi, pmc_id, title, abstract, journal, year,
    mesh_terms, and authors.

This script is the first step in the GutFeeling data pipeline:
    01_pubmed_fetch.py        ← this script
    02_prepare_rag_jsonl.py   → converts JSON to JSONL for LlamaIndex
    GutFeeling.py             → builds the vector index and runs the app

Usage:
    Set NCBI_EMAIL and NCBI_API_KEY in your .env file, then run:
    $ python 01_pubmed_fetch.py

Notes:
    - NCBI requires an email address for all API requests.
    - An API key increases the rate limit from 3 to 10 requests/second.
    - Free API keys can be obtained at: https://www.ncbi.nlm.nih.gov/account/
    - The script fetches ~56,000 articles and takes several hours to complete.
      Progress is printed to the console for monitoring.

Author: Alberto Sánchez-Pascuala
WBS Coding School, 2026
"""

import os
import json
import time
from dotenv import load_dotenv
from Bio import Entrez                                  # Biopython's NCBI Entrez interface
from http.client import IncompleteRead, RemoteDisconnected  # Network errors to catch and retry


# ---------------------------------------------------------------------------
# 1. AUTHENTICATION
# ---------------------------------------------------------------------------
# NCBI requires identification for all API requests.
# Credentials are loaded from .env to avoid hardcoding sensitive data in code.
# Required .env variables:
#   NCBI_EMAIL   — your email address (used by NCBI to contact you if needed)
#   NCBI_API_KEY — obtained from your NCBI account settings

load_dotenv()
email   = os.getenv("NCBI_EMAIL")
api_key = os.getenv("NCBI_API_KEY")

Entrez.email   = email
Entrez.api_key = api_key

if not email or not api_key:
    raise ValueError(
        "NCBI_EMAIL and NCBI_API_KEY must be set in your .env file.\n"
        "Get a free API key at: https://www.ncbi.nlm.nih.gov/account/"
    )


# ---------------------------------------------------------------------------
# 2. SEARCH QUERY
# ---------------------------------------------------------------------------
# The query targets articles about microbiome/microbiota in a health/disease
# context, restricted to human studies and English-language publications.
#
# Key design decisions:
# - Title/Abstract search (not MeSH): catches articles before MeSH indexing
#   is complete (recent publications) and increases recall.
# - The health/disease filter avoids purely methodological papers (e.g.
#   "16S rRNA sequencing protocol") that are not relevant to the chatbot.
# - humans[MeSH Terms] and english[lang] narrow the corpus to the most
#   clinically relevant and accessible literature.

query_base = (
    "("
        "microbiome[Title/Abstract] OR microbiota[Title/Abstract] OR "
        "gut flora[Title/Abstract] OR gut microbiota[Title/Abstract] OR "
        "intestinal flora[Title/Abstract]"
    ") AND ("
        "disease[Title/Abstract] OR disorder[Title/Abstract] OR "
        "syndrome[Title/Abstract] OR condition[Title/Abstract] OR "
        "benefit[Title/Abstract] OR probiotic[Title/Abstract] OR "
        "health[Title/Abstract]"
    ") AND humans[MeSH Terms] AND english[lang]"
)


# ---------------------------------------------------------------------------
# 3. RETRY HELPER FUNCTIONS
# ---------------------------------------------------------------------------
# The NCBI API occasionally drops connections or returns incomplete responses,
# especially for large batch requests. These wrappers add automatic retry logic
# to make the download robust to transient network errors.

def safe_efetch(db, id_list, retmode="xml", max_retries=3, sleep_time=1):
    """
    Fetch article records from NCBI with automatic retry on network errors.

    Args:
        db         : NCBI database (e.g. "pubmed")
        id_list    : List of PMIDs to fetch
        retmode    : Return format ("xml" required for full metadata)
        max_retries: Number of retry attempts before raising an error
        sleep_time : Seconds to wait between retries

    Returns:
        Parsed NCBI records dict
    """
    for attempt in range(max_retries):
        try:
            handle  = Entrez.efetch(db=db, id=id_list, retmode=retmode)
            records = Entrez.read(handle)
            handle.close()
            return records
        except IncompleteRead:
            print(f"IncompleteRead detected, retry {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Other efetch error: {e}, retry {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
    raise ValueError(f"Failed to fetch after {max_retries} retries")


def safe_esearch(db, term, retstart=0, retmax=50, max_retries=3, sleep_time=1):
    """
    Search PubMed for article IDs (PMIDs) with automatic retry.

    Args:
        db        : NCBI database (e.g. "pubmed")
        term      : Search query string
        retstart  : Starting position in results (for pagination)
        retmax    : Maximum number of results to return per request
        max_retries / sleep_time: same as safe_efetch

    Returns:
        NCBI search record dict containing "IdList" and "Count"
    """
    for attempt in range(max_retries):
        try:
            handle = Entrez.esearch(db=db, term=term, retstart=retstart, retmax=retmax)
            record = Entrez.read(handle)
            handle.close()
            return record
        except RemoteDisconnected:
            print(f"RemoteDisconnected in esearch, retry {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Other esearch error: {e}, retry {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
    raise ValueError(f"Failed esearch after {max_retries} retries")


# ---------------------------------------------------------------------------
# 4. PMID RETRIEVAL — YEARLY BATCHES
# ---------------------------------------------------------------------------
# PubMed's API returns a maximum of 9,999 results per query.
# With 56,000+ articles in our corpus, a single query would truncate results.
# Solution: split the query by year (1980–2025) and retrieve PMIDs year by year.
# Each yearly batch is well under the 9,999 limit.
#
# batch_size=20 uses small batches to minimize data loss on connection drops.

start_year = 1980
end_year   = 2025
batch_size = 20
pmids      = []

for year in range(start_year, end_year + 1):
    yearly_query = query_base + f" AND {year}[PDAT]"

    # Get total article count for this year
    record            = safe_esearch(db="pubmed", term=yearly_query, retstart=0, retmax=1)
    total_count_year  = int(record["Count"])

    if total_count_year == 0:
        continue

    print(f"Year {year}: {total_count_year} articles found")

    # Paginate through results in batches
    for start in range(0, total_count_year, batch_size):
        print(f"  Fetching PMIDs: records {start}–{start + batch_size}")
        record = safe_esearch(db="pubmed", term=yearly_query,
                              retstart=start, retmax=batch_size)
        pmids.extend(record["IdList"])
        time.sleep(0.5)  # Respect NCBI rate limits (max 10 req/sec with API key)

print(f"\nTotal PMIDs retrieved: {len(pmids)}")


# ---------------------------------------------------------------------------
# 5. METADATA FETCHING
# ---------------------------------------------------------------------------
# For each PMID, fetch the full article metadata in XML format and extract:
# title, abstract, journal, year, MeSH terms, authors, DOI, PMC ID.
#
# Full text is NOT retrieved — abstracts are sufficient for RAG retrieval
# and dramatically reduce download time and storage requirements.

articles = []

for start in range(0, len(pmids), batch_size):
    batch_pmids = pmids[start:start + batch_size]
    print(f"Fetching article metadata: {start}–{start + len(batch_pmids)}")

    records = safe_efetch(db="pubmed", id_list=batch_pmids)

    for article in records["PubmedArticle"]:
        medline      = article["MedlineCitation"]
        article_data = medline["Article"]
        pmid         = medline["PMID"]

        # Title
        title = article_data.get("ArticleTitle", "")

        # Abstract — some articles have structured abstracts (multiple sections)
        # joined into a single string with spaces
        abstract_text = ""
        if "Abstract" in article_data:
            abstract_text = " ".join(
                [str(x) for x in article_data["Abstract"]["AbstractText"]]
            )

        # Journal name and publication year
        journal  = article_data["Journal"]["Title"]
        year_pub = article_data["Journal"]["JournalIssue"]["PubDate"].get("Year", "")

        # MeSH terms — controlled vocabulary assigned by PubMed indexers
        # Used for topic analysis in the Literature Landscape tab
        mesh_terms = []
        if "MeshHeadingList" in medline:
            mesh_terms = [
                str(mh["DescriptorName"]) for mh in medline["MeshHeadingList"]
            ]

        # DOI and PMC ID — used to generate direct links to source articles
        doi    = ""
        pmc_id = ""
        if "ArticleIdList" in article["PubmedData"]:
            for aid in article["PubmedData"]["ArticleIdList"]:
                if aid.attributes.get("IdType") == "doi":
                    doi = str(aid)
                if aid.attributes.get("IdType") == "pmc":
                    pmc_id = str(aid)

        # Authors — stored as structured dicts for co-authorship network analysis
        authors = []
        if "AuthorList" in article_data:
            for a in article_data["AuthorList"]:
                authors.append({
                    "last_name": a.get("LastName",  ""),
                    "fore_name": a.get("ForeName",  ""),
                    "initials":  a.get("Initials",  "")
                })

        articles.append({
            "pmid":       str(pmid),
            "doi":        doi,
            "pmc_id":     pmc_id,
            "title":      title,
            "abstract":   abstract_text,
            "journal":    journal,
            "year":       year_pub,
            "mesh_terms": mesh_terms,
            "authors":    authors
        })

    time.sleep(0.5)  # Respect NCBI rate limits


# ---------------------------------------------------------------------------
# 6. SAVE TO JSON
# ---------------------------------------------------------------------------
# Saved as a single JSON file (list of article dicts).
# This file is the input for 02_prepare_rag_jsonl.py.

output_file = "pubmed_microbiome_metadata_only.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(articles)} articles saved to {output_file}")
