"""
analysis.py — Literature Landscape Analysis Functions
======================================================
All visualization functions for the GutFeeling Literature Landscape tab.
Each function receives a preprocessed DataFrame and returns a Matplotlib
figure, which is then rendered in Streamlit via the show_figure() helper
in GutFeeling.py.

Functions:
    load_data()            : Load and preprocess the PubMed JSON corpus
    plot_temporal()        : Analysis 1 — Publications per year (1980–2025)
    plot_journals()        : Analysis 2 — Top 20 most prolific journals
    plot_mesh_terms()      : Analysis 3 — Top 30 MeSH terms
    plot_disease_heatmap() : Analysis 4 — Global health topics by decade
    plot_network()         : Analysis 5 — Co-authorship network

Author: Alberto Sánchez-Pascuala
WBS Coding School, 2026
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from itertools import combinations   # Generates all author pairs per article for edge building
from collections import Counter      # Efficiently counts co-authorship frequencies
from networkx.algorithms import community  # Greedy modularity for community detection

# Global plot styling — applied to all figures in this module
sns.set_theme(style="darkgrid")
plt.rcParams["font.size"] = 11


# =============================================================
# DATA LOADING
# =============================================================

def load_data(json_path: str):
    """
    Load the PubMed JSON corpus and return two DataFrames.

    Returns:
        df      : One row per article. Contains year, journal, authors, mesh_terms, pmid.
        df_mesh : One row per MeSH term (exploded from df), with generic terms removed.

    The MeSH explosion is necessary for topic-level analyses: each article
    can have multiple MeSH terms, so we need one row per term to count frequencies.

    Generic terms (Humans, Animals, Male, Female, etc.) are removed because
    they appear in virtually every article and add no topic-specific information.
    Microbiota-specific generic terms (Microbiota, Bacteria, Feces, etc.) are
    also removed for the same reason — they would dominate all topic charts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    df = pd.DataFrame(articles)

    # Convert year to numeric and filter valid range
    # errors="coerce" turns unparseable values into NaN instead of raising an error
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(1980, 2025)].copy()

    # Build df_mesh: keep only articles with at least one MeSH term, then explode
    df_mesh = df[df["mesh_terms"].apply(lambda x: len(x) > 0)].copy()
    df_mesh = df_mesh.explode("mesh_terms")
    df_mesh["mesh_terms"] = df_mesh["mesh_terms"].str.strip()

    # Terms to exclude: demographic, methodological, and field-generic terms
    # that appear so frequently they obscure meaningful topic patterns
    generic_terms = {
        "Humans", "Animals", "Male", "Female", "Adult",
        "Middle Aged", "Aged", "Young Adult", "Mice",
        "Child", "Adolescent", "Infant", "Aged, 80 and over",
        "Child, Preschool",
        "Gastrointestinal Microbiome", "Microbiota", "Bacteria",
        "Feces", "RNA, Ribosomal, 16S", "Dysbiosis",
        "Gastrointestinal Tract", "Intestines", "Intestinal Mucosa",
        "Biomarkers", "Risk Factors", "Disease Models, Animal",
        "Case-Control Studies", "Prospective Studies",
        "Treatment Outcome",
        "Mice, Inbred C57BL"
    }
    df_mesh = df_mesh[~df_mesh["mesh_terms"].isin(generic_terms)]

    print(f"Loaded {len(df)} articles | {df_mesh['mesh_terms'].nunique()} unique MeSH terms")
    return df, df_mesh


# =============================================================
# ANALYSIS 1 — TEMPORAL EVOLUTION
# =============================================================

def plot_temporal(df: pd.DataFrame) -> plt.Figure:
    """
    Line chart showing the number of publications per year (1980–2025).

    A filled area chart was chosen over a simple bar chart to better
    convey the exponential growth trend. The fill (alpha=0.15) adds
    visual weight without obscuring the line.

    Three milestone annotations mark key events that drove the explosive
    growth of the field:
    - 2008: Launch of the NIH Human Microbiome Project
    - 2012: Publication of the first HMP results (Science, Nature)
    - 2020: COVID-19 pandemic (surge in microbiota-immunity research)

    The COVID-19 label is offset to the left of the dashed line to avoid
    overlapping with the sharp 2020-2021 peak in the data.
    """
    publications_per_year = (
        df.groupby("year")
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.lineplot(
        data=publications_per_year,
        x="year", y="count",
        color="#4caf50", linewidth=2.5, ax=ax
    )

    # Filled area under the curve for visual emphasis
    ax.fill_between(
        publications_per_year["year"],
        publications_per_year["count"],
        alpha=0.15, color="#4caf50"
    )

    # Milestone annotations: (label, y-position as fraction of y-max, text alignment)
    milestones = {
        2008: ("Human Microbiome\nProject launched", 0.75, "left"),
        2012: ("HMP first\nresults published",        0.55, "right"),
        2020: ("COVID-19\npandemic",                  0.35, "right"),
    }
    for year, (label, ypos, side) in milestones.items():
        ax.axvline(x=year, color="#888", linestyle="--", linewidth=1)
        offset = -0.5 if side == "left" else 0.3
        ax.text(year + offset, ax.get_ylim()[1] * ypos, label,
                fontsize=8.5, color="#555",
                ha="right" if side == "left" else "left")

    ax.set_title("Microbiota Research — Publications per Year (1980–2025)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Publications", fontsize=11)
    ax.set_xlim(1980, 2025)
    sns.despine()
    plt.tight_layout()

    return fig


# =============================================================
# ANALYSIS 2 — JOURNAL LANDSCAPE
# =============================================================

def plot_journals(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of the top 20 most prolific journals.

    A horizontal layout was chosen over vertical bars because journal
    names are long strings that would be unreadable on a vertical axis.

    The x-axis is extended to 115% of the maximum count to leave space
    for the count labels placed to the right of each bar.

    The Greens_r palette is consistent with the app's green theme and
    uses darker shades for higher-ranked journals, drawing the eye to
    the most prolific ones.
    """
    journal_counts = (
        df.groupby("journal")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top20 = journal_counts.head(20)

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.barplot(
        data=top20,
        x="count", y="journal",
        palette="Greens_r", ax=ax
    )

    # Count labels to the right of each bar for readability
    for i, count in enumerate(top20["count"]):
        ax.text(count + 10, i, str(count), va="center",
                fontsize=9, color="#444")

    ax.set_title("Top 20 Most Prolific Journals in Microbiota Research",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Publications", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, top20["count"].max() * 1.15)  # Extra space for count labels
    sns.despine()
    plt.tight_layout()

    return fig


# =============================================================
# ANALYSIS 3 — MESH TERMS LANDSCAPE
# =============================================================

def plot_mesh_terms(df_mesh: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of the top 30 MeSH terms.

    MeSH (Medical Subject Headings) terms are controlled vocabulary
    assigned by PubMed indexers to each article. They provide a
    standardized, curated view of the topics covered in the corpus —
    more reliable than keyword extraction from free text.

    Generic terms were removed in load_data() so that this chart
    reveals meaningful topic clusters (gut-brain axis, probiotics,
    metabolomics, etc.) rather than obvious field-wide terms.
    """
    top_mesh = (
        df_mesh.groupby("mesh_terms")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.barplot(
        data=top_mesh,
        x="count", y="mesh_terms",
        palette="Greens_r", ax=ax
    )

    for i, count in enumerate(top_mesh["count"]):
        ax.text(count + 10, i, str(count), va="center",
                fontsize=8.5, color="#444")

    ax.set_title("Top 30 MeSH Terms in Microbiota Research",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Articles", fontsize=11)
    ax.set_ylabel("")
    sns.despine()
    plt.tight_layout()

    return fig


# =============================================================
# ANALYSIS 4 — GLOBAL HEALTH TOPICS BY DECADE (NORMALISED)
# =============================================================

def plot_disease_heatmap(df: pd.DataFrame, df_mesh: pd.DataFrame) -> plt.Figure:
    """
    Heatmap showing the % of total articles per decade that mention
    each major disease group, correcting for the overall growth of the field.

    Why normalize by decade totals?
    Without normalization, all disease groups would appear to grow over time
    simply because the total number of publications grows. Dividing by the
    total articles per decade converts raw counts into relative prevalence,
    revealing genuine shifts in research focus (e.g. the rise of neurological
    and cancer research in the 2010s–2020s).

    Disease groups are defined by keyword patterns matched against MeSH terms.
    Each article is counted at most once per group per decade (deduplication
    by pmid + decade) to avoid inflation from multi-term articles.
    """
    # Keyword patterns for each disease group (matched case-insensitively against MeSH terms)
    disease_groups = {
        "Neurological / Psychiatric": (
            "depression|anxiety|autism|alzheimer|parkinson|"
            "multiple sclerosis|schizophrenia|bipolar|stress|"
            "cognitive|dementia|neurodegeneration"
        ),
        "Cardiovascular": (
            "cardiovascular|atherosclerosis|hypertension|"
            "heart failure|coronary|stroke|myocardial|"
            "cardiac|blood pressure|cholesterol"
        ),
        "Metabolic": (
            "diabetes|obesity|fatty liver|metabolic syndrome|"
            "insulin resistance|NAFLD|NASH|dyslipidemia|"
            "adipose|adiposity"
        ),
        "Autoimmune": (
            "rheumatoid|lupus|autoimmun|celiac|psoriasis|thyroid"
        ),
        "IBD": (
            "inflammatory bowel|crohn|ulcerative colitis|colitis"
        ),
        "Maternal / Infant": (
            "pregnancy|newborn|infant|asthma|allerg|"
            "breastfeed|breast milk|caesarean|birth|neonatal"
        ),
        "Infectious": (
            "covid|sars-cov|coronavirus|"
            "clostridium|clostridioides|"
            "salmonella|campylobacter|helicobacter|"
            "sepsis|bacteremia|"
            "antibiotic resistance|antimicrobial resistance"
        ),
        "Cancer": (
            "cancer|tumor|tumour|neoplasm|carcinoma|"
            "oncol|malignant|metastasis"
        )
    }

    df_topics = df_mesh.copy()
    df_topics["decade"] = (
        df_topics["year"] // 10 * 10
    ).astype(int).astype(str) + "s"

    # Count unique articles per group and decade
    results = []
    for group_name, keywords in disease_groups.items():
        matches = df_topics[
            df_topics["mesh_terms"].str.contains(keywords, case=False, na=False)
        ]
        # drop_duplicates ensures each article is counted once per decade per group
        decade_counts = (
            matches.drop_duplicates(subset=["pmid", "decade"])
            .groupby("decade")
            .size()
            .reset_index(name="count")
        )
        decade_counts["group"] = group_name
        results.append(decade_counts)

    df_results = pd.concat(results, ignore_index=True)

    pivot_topics = df_results.pivot(
        index="group", columns="decade", values="count"
    ).fillna(0)

    # Sort rows by total volume across all decades
    pivot_topics["total"] = pivot_topics.sum(axis=1)
    pivot_topics = pivot_topics.sort_values(
        "total", ascending=False
    ).drop(columns="total")

    # Normalize: divide each cell by the total articles in that decade × 100
    total_per_decade = (
        df[df["year"].notna()]
        .assign(decade=(df["year"] // 10 * 10).astype(int).astype(str) + "s")
        .groupby("decade")
        .size()
        .rename("total")
    )
    pivot_normalized = pivot_topics.div(total_per_decade) * 100

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.heatmap(
        pivot_normalized,
        annot=True, fmt=".1f",  # Show % values with 1 decimal place
        cmap="Greens",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Global Health Topics in Microbiota Research by Decade (% of total articles)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_ylabel("")
    plt.tight_layout()

    return fig


# =============================================================
# ANALYSIS 5 — CO-AUTHORSHIP NETWORK
# =============================================================

def build_author_data(df: pd.DataFrame):
    """
    Extract author publication counts and co-authorship edges from the corpus.

    For each article with 2+ authors, all possible author pairs are generated
    using itertools.combinations. Each pair represents one co-authored article.
    Repeated pairs across articles accumulate as edge weights in plot_network().

    Returns:
        edges        : List of (author1, author2) tuples (one per co-authored article)
        author_counts: Dict mapping author name → total publications
    """
    edges = []
    author_counts = {}

    for _, row in df.iterrows():
        authors = row["authors"]
        if not isinstance(authors, list) or len(authors) < 2:
            continue

        names = []
        for a in authors:
            last = a.get("last_name", "").strip()
            fore = a.get("fore_name", "").strip()
            if last:
                full = f"{fore} {last}" if fore else last
                names.append(full)
                author_counts[full] = author_counts.get(full, 0) + 1

        # Generate all pairs: for 3 authors A, B, C → (A,B), (A,C), (B,C)
        for pair in combinations(names, 2):
            edges.append(pair)

    return edges, author_counts


def plot_network(df: pd.DataFrame,
                 min_publications: int = 30,
                 min_coauthorships: int = 3) -> plt.Figure:
    """
    Co-authorship network of the most prolific microbiota researchers.

    Design decisions:
    - min_publications=30: filters to authors with significant output,
      reducing noise from occasional contributors.
    - min_coauthorships=3: keeps only strong, repeated collaborations,
      which reflect stable research groups rather than one-off co-authorships.
    - Only the largest connected component is shown, which contains the
      main international research community.
    - Community detection uses greedy modularity maximization (NetworkX),
      which identifies research clusters automatically without prior knowledge.
    - Node size encodes publication count; edge thickness encodes shared articles.
    - The spring layout (Fruchterman-Reingold) naturally places heavily
      connected nodes (research hubs) at the center.

    Note: this function takes ~2 minutes on first run due to the size of
    the corpus. Results are cached by Streamlit after the first execution.
    """
    edges, author_counts = build_author_data(df)

    # Keep only prolific authors
    prolific = {a for a, c in author_counts.items() if c >= min_publications}

    # Count co-authorships between prolific authors only
    edge_weights = Counter()
    for a1, a2 in edges:
        if a1 in prolific and a2 in prolific:
            edge_weights[tuple(sorted([a1, a2]))] += 1

    # Filter to strong edges (repeated collaborations)
    strong_edges = {p: w for p, w in edge_weights.items()
                    if w >= min_coauthorships}

    # Build undirected weighted graph
    G = nx.Graph()
    for (a1, a2), weight in strong_edges.items():
        G.add_edge(a1, a2, weight=weight)
    for node in G.nodes():
        G.nodes[node]["publications"] = author_counts.get(node, 0)

    # Extract the largest connected component (main research community)
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()

    # Detect research communities using greedy modularity maximization
    communities_list = list(
        community.greedy_modularity_communities(G_main)
    )
    community_map = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            community_map[node] = i

    # Spring layout: k controls node spacing, seed ensures reproducibility
    pos = nx.spring_layout(G_main, k=1.2, seed=42, iterations=100)

    node_sizes  = [G_main.nodes[n]["publications"] * 4 for n in G_main.nodes()]
    palette     = ["#2e7d32", "#1565c0", "#c62828", "#f57f17", "#6a1b9a",
                   "#00838f", "#4e342e", "#558b2f", "#d84315", "#37474f"]
    node_colors = [palette[community_map[n] % len(palette)] for n in G_main.nodes()]
    edge_widths = [G_main[u][v]["weight"] * 0.4 for u, v in G_main.edges()]

    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor("#f7f5f0")
    ax.set_facecolor("#f7f5f0")

    nx.draw_networkx_edges(G_main, pos, width=edge_widths,
                           alpha=0.25, edge_color="#aaaaaa", ax=ax)
    nx.draw_networkx_nodes(G_main, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, ax=ax)

    # Label only the top 10 most prolific authors to avoid label overlap
    top_nodes = sorted(G_main.nodes(),
                       key=lambda n: G_main.nodes[n]["publications"],
                       reverse=True)[:10]
    nx.draw_networkx_labels(G_main, pos, labels={n: n for n in top_nodes},
                            font_size=13, font_weight="bold",
                            font_color="#111111", ax=ax)

    ax.set_title(
        "Co-authorship Network in Microbiota Research\n"
        "Node size = publications  ·  Color = research community  ·  "
        "Edge thickness = shared articles",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.axis("off")
    plt.tight_layout()

    return fig
