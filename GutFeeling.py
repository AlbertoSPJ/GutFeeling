"""
GutFeeling.py — Main Streamlit Application
===========================================
RAG chatbot for gut microbiota research, grounded in 56,000+ PubMed abstracts (1980-2025).

Architecture overview:
- The user sends a question via the chat interface.
- The question is embedded using a HuggingFace sentence-transformer model (all-MiniLM-L6-v2).
- The embedding is used to retrieve the 5 most semantically similar PubMed abstracts
  from a pre-built vector index (LlamaIndex / FAISS).
- The retrieved abstracts + the question are passed to Llama 3.3 70B (via Groq API),
  which generates a grounded, citation-backed answer.
- The app also includes a Literature Landscape tab with exploratory visualizations
  of the full PubMed corpus.

Author: Alberto Sánchez-Pascuala
WBS Coding School, 2026
"""

import os       # For reading environment variables (API keys, file paths)
import io       # For rendering matplotlib figures as in-memory PNG buffers (avoids temp files)
import json     # For parsing the JSONL corpus line by line
import streamlit as st
from dotenv import load_dotenv  # Loads GROQ_API_KEY from .env file (keeps secrets out of code)

# LlamaIndex: core RAG framework components
from llama_index.core import (
    VectorStoreIndex,        # Builds and queries the vector index
    StorageContext,          # Handles loading a persisted index from disk
    load_index_from_storage, # Loads a pre-built index (avoids rebuilding on every run)
    Settings,                # Global config for LLM, embeddings, and chunking strategy
    Document                 # LlamaIndex document object wrapping each PubMed abstract
)
from llama_index.llms.groq import Groq                          # Groq API integration for Llama 3.3 70B
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Local sentence-transformer embeddings
from llama_index.core.node_parser import SentenceSplitter        # Splits documents into overlapping chunks
from llama_index.core.chat_engine import ContextChatEngine       # Chat engine that injects retrieved context into each turn
from llama_index.core.memory import ChatMemoryBuffer             # Keeps recent conversation history within token budget
from llama_index.core.base.llms.types import ChatMessage, MessageRole  # For building structured system prompts

# Analysis functions for the Literature Landscape tab (defined in analysis.py)
from analysis import (
    load_data,
    plot_temporal,
    plot_journals,
    plot_mesh_terms,
    plot_disease_heatmap,
    plot_network
)


# ---------------------------------------------------------------------------
# CACHED DATA LOADERS
# ---------------------------------------------------------------------------
# @st.cache_data tells Streamlit to run these functions only once and reuse
# the result across reruns. This is essential for performance: loading 56k
# articles and generating network graphs would otherwise block the UI every time.

@st.cache_data
def load_literature_data():
    """Load and preprocess the full PubMed metadata for the Literature Landscape tab."""
    return load_data("./pubmed_microbiome_metadata_only.json")

@st.cache_data
def get_temporal(_df):
    """Generate the temporal evolution plot (publications per year)."""
    return plot_temporal(_df)

@st.cache_data
def get_journals(_df):
    """Generate the top 20 journals bar chart."""
    return plot_journals(_df)

@st.cache_data
def get_mesh(_df_mesh):
    """Generate the top 30 MeSH terms bar chart."""
    return plot_mesh_terms(_df_mesh)

@st.cache_data
def get_disease_heatmap(_df, _df_mesh):
    """Generate the global health topics by decade heatmap."""
    return plot_disease_heatmap(_df, _df_mesh)

@st.cache_data
def get_network(_df):
    """Generate the co-authorship network graph (~2 min on first run, then cached)."""
    return plot_network(_df)


# ---------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM CSS THEME
# ---------------------------------------------------------------------------
# Must be the first Streamlit call in the script.

st.set_page_config(
    page_title="GutFeeling — Microbiota RAG Assistant",
    page_icon="🦠",
    layout="wide"   # Wide layout gives more horizontal space for figures and chat
)

# Custom CSS injected directly into the Streamlit app.
# Streamlit uses React under the hood, so some selectors need !important to override defaults.
# Theme: muted green (#2e7d32, #4caf50) on a neutral grey-white background (#f0f2f0).
# Typography: Bitter (serif) for headings, Source Sans 3 (sans-serif) for body text.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bitter:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

    header[data-testid="stHeader"] {
        background-color: #f0f2f0 !important;
        border-bottom: none !important;
    }
    header[data-testid="stHeader"]::before {
        background-color: #f0f2f0 !important;
    }

    html, body, .stApp {
        font-family: 'Source Sans 3', sans-serif;
        background-color: #f0f2f0 !important;
        color: #1a2e1a !important;
    }

    .block-container {
        background-color: #f0f2f0 !important;
        padding-top: 3.5rem !important;
    }

    h1, h2, h3 {
        font-family: 'Bitter', serif !important;
        color: #1a2e1a !important;
    }

    p, li, span, label, div, h4, h5, h6, small {
        color: #1a2e1a !important;
    }

    .hero-title {
        font-family: 'Bitter', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #2e7d32 !important;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #2e7d32 !important;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Tab bar styling */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #e8f5e9 !important;
        border-bottom: 2px solid #a5d6a7 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.2rem 0.5rem !important;
    }

    /* Tab text — targeting multiple selectors for Streamlit version compatibility */
    [data-testid="stTabs"] [data-baseweb="tab"] > div,
    [data-testid="stTabs"] [data-baseweb="tab"] p,
    [data-testid="stTabs"] [data-baseweb="tab"] span,
    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        padding: 0.6rem 1.2rem !important;
        letter-spacing: 0.02em !important;
        color: #4a7a4a !important;
        background-color: transparent !important;
    }

    /* Active tab: underline indicator instead of background fill */
    [data-testid="stTabs"] [aria-selected="true"],
    [data-testid="stTabs"] [aria-selected="true"] > div,
    [data-testid="stTabs"] [aria-selected="true"] p,
    [data-testid="stTabs"] [aria-selected="true"] span {
        color: #1b5e20 !important;
        border-bottom: 3px solid #2e7d32 !important;
        background-color: transparent !important;
        border-radius: 0 !important;
    }

    [data-testid="stTabsContent"] {
        background-color: #f0f2f0 !important;
        padding-top: 1.5rem !important;
    }

    /* Mode selection buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.2s ease !important;
        border: 2px solid #4caf50 !important;
        background-color: #ffffff !important;
        color: #2e7d32 !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        border-color: #2e7d32 !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #4caf50 !important;
        color: #ffffff !important;
        border-color: #4caf50 !important;
    }

    /* Send button (form submit) */
    .stFormSubmitButton > button {
        border-radius: 12px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        background-color: #4caf50 !important;
        color: #ffffff !important;
        border: 2px solid #4caf50 !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    .stFormSubmitButton > button:hover {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
    }

    /* Active mode badge displayed below the mode buttons */
    .active-mode-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        letter-spacing: 0.03em;
    }

    .badge-public {
        background-color: #e8f5e9;
        color: #2e7d32 !important;
        border: 1px solid #a5d6a7;
    }

    .badge-scientist {
        background-color: #e3f2fd;
        color: #1565c0 !important;
        border: 1px solid #90caf9;
    }

    /* Chat bubbles: user messages aligned right, AI messages aligned left */
    .chat-message-user {
        background-color: #dcedc8 !important;
        color: #1a2e1a !important;
        border-radius: 16px 16px 4px 16px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        border: 1px solid #aed581;
    }

    .chat-message-ai {
        background-color: #ffffff !important;
        color: #1a2e1a !important;
        border-radius: 16px 16px 16px 4px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 85%;
        border: 1px solid #c8e6c9;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Source citation chips (clickable PubMed links) */
    .source-chip {
        display: inline-block;
        background-color: #dcedc8 !important;
        border: 1px solid #8bc34a !important;
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        color: #33691e !important;
        margin: 0.2rem 0.2rem 0 0;
        text-decoration: none;
        font-weight: 600;
    }

    .source-chip:hover {
        background-color: #c5e1a5 !important;
        color: #1b5e20 !important;
    }

    .sources-label {
        font-size: 0.78rem;
        color: #33691e !important;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }

    /* Chat input field */
    input[type="text"],
    [data-testid="stTextInput"] input,
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a2e1a !important;
        border: 2px solid #a5d6a7 !important;
        border-radius: 24px !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 0.95rem !important;
        caret-color: #4caf50 !important;
    }

    input[type="text"]:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.15) !important;
        outline: none !important;
    }

    /* Selectbox (Literature Landscape analysis picker) */
    [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        border: 2px solid #a5d6a7 !important;
        border-radius: 8px !important;
        color: #1a2e1a !important;
    }

    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #1a2e1a !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: #e8f5e9 !important;
        color: #2e7d32 !important;
    }

    [role="option"] {
        background-color: #ffffff !important;
        color: #1a2e1a !important;
    }

    [role="option"]:hover {
        background-color: #e8f5e9 !important;
    }

    .divider {
        border: none;
        border-top: 1px solid #c8e6c9;
        margin: 1.5rem 0;
    }

    [data-testid="stAlert"] {
        background-color: #f1f8e9 !important;
        border-color: #a5d6a7 !important;
        color: #2e7d32 !important;
    }

    footer {
        font-size: 0.78rem;
        color: #6a9a6a !important;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv()  # Load GROQ_API_KEY from .env file into environment

# Paths to the JSONL corpus and pre-built vector indexes.
# Two embedding models were evaluated; all-MiniLM-L6-v2 was selected for its
# balance between retrieval quality and inference speed on CPU.
JSONL_PATH = "./pubmed_microbiome_rag.jsonl"
PERSIST_DIRS = {
    "minilm": "./vector_index_minilm",
    "mpnet":  "./vector_index_mpnet"
}
EMBED_CACHE_DIRS = {
    "minilm": "./embed_minilm",
    "mpnet":  "./embed_mpnet"
}

MODEL_NAME = "llama-3.3-70b-versatile"  # Groq-hosted Llama 3.3 70B

EMBED_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # Chosen model: fast, lightweight, strong on biomedical text
    "mpnet":  "sentence-transformers/all-mpnet-base-v2"    # Alternative tested during development
}


# ---------------------------------------------------------------------------
# LLM & INDEX INITIALIZATION
# ---------------------------------------------------------------------------
# @st.cache_resource is used for objects that are expensive to create and
# should be shared across all user sessions (LLM connections, vector indexes).
# Unlike @st.cache_data, it does not serialize/deserialize the output.

@st.cache_resource
def init_llm():
    """Initialize the Groq LLM client. Cached to avoid redundant API connections."""
    return Groq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY")
    )


@st.cache_resource
def load_or_create_index(embedding_name: str):
    """
    Load a pre-built vector index from disk, or download it from Hugging Face Hub
    if not found locally (e.g. on Streamlit Cloud deployment).

    Building the index involves:
    1. Reading all 55,990 PubMed abstracts from the JSONL corpus.
    2. Splitting each abstract into overlapping chunks (chunk_size=1024, overlap=50 tokens).
       Overlap ensures that sentences at chunk boundaries are not lost during retrieval.
    3. Embedding each chunk with the selected sentence-transformer model.
    4. Storing the resulting FAISS index to disk for reuse on subsequent runs.

    On subsequent runs, the index is loaded directly from disk (~seconds vs ~hours).
    On Streamlit Cloud, the index is downloaded from Hugging Face Hub on first run.
    """
    llm = init_llm()
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODELS[embedding_name],
        cache_folder=EMBED_CACHE_DIRS[embedding_name]  # Cache model weights locally
    )

    # Set global LlamaIndex settings (applies to all index operations in this session)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

    persist_dir = PERSIST_DIRS[embedding_name]

    if not os.path.exists(persist_dir):
        # Check if index is available on Hugging Face Hub (Streamlit Cloud deployment)
        hf_token = os.getenv("HF_TOKEN_READ")
        if hf_token:
            from huggingface_hub import snapshot_download
            with st.spinner("Downloading index from Hugging Face Hub..."):
                snapshot_download(
                    repo_id="AlbertoSPJ/gutfeeling-index",
                    repo_type="dataset",
                    local_dir=persist_dir,
                    token=hf_token
                )
        else:
            # Local fallback: build index from corpus
            documents = []
            with open(JSONL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    article = json.loads(line)
                    doc_metadata = {
                        "pmid":    article["metadata"].get("pmid", "N/A"),
                        "journal": article["metadata"].get("journal", "Unknown"),
                        "year":    article["metadata"].get("year", "N/A")
                    }
                    documents.append(Document(text=article["content"], metadata=doc_metadata))

            vector_index = VectorStoreIndex.from_documents(documents, show_progress=True)
            vector_index.storage_context.persist(persist_dir=persist_dir)

    # Load index from disk (whether just downloaded or already present)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_index = load_index_from_storage(storage_context)

    return vector_index


# ---------------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------------
# Two modes with different system prompts allow the same RAG engine to serve
# two distinct audiences: the general public and professional scientists.
# The key constraint in both is "Answer ONLY based on the provided context"
# — this is what makes it a grounded RAG system rather than a generic chatbot.

CONSUMER_PROMPT = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are a friendly and approachable health assistant specialized in gut microbiota. "
            "Your audience is the general public — people who may have received a commercial microbiota analysis "
            "and want to understand what it means for their health. "
            "Always explain concepts in simple, clear, non-technical language. Avoid jargon. "
            "Answer ONLY based on the provided scientific context. "
            "Give practical, actionable advice when possible (e.g. dietary changes, lifestyle habits). "
            "If the context only partially answers the question, summarize what is found and indicate what is not covered. "
            "Only say 'I cannot find this information' if the context is completely unrelated to the question."
        )
    )
]

SCIENTIST_PROMPT = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are a strict biomedical research assistant specialized in microbiota science. "
            "Your audience is professional scientists and researchers. "
            "Use precise scientific terminology. Reference methodologies, study designs, and findings accurately. "
            "Answer ONLY based on the provided scientific context. "
            "If the context only partially answers the question, summarize what is found and clearly indicate the gaps. "
            "Only say 'I cannot find this information in the documents' if the context is completely unrelated to the question. "
            "Keep answers concise and scientifically rigorous."
        )
    )
]


# ---------------------------------------------------------------------------
# CHAT ENGINE BUILDER
# ---------------------------------------------------------------------------

def build_chat_engine(retriever, prompt):
    """
    Build a ContextChatEngine instance for a given mode.

    ContextChatEngine works by:
    1. Retrieving the top-k most relevant chunks from the vector index for each user query.
    2. Injecting those chunks as context into the LLM prompt.
    3. Maintaining a rolling conversation history via ChatMemoryBuffer.

    token_limit=1500 for the memory buffer keeps recent turns within the LLM's
    context window without exceeding token limits or inflating API costs.
    """
    return ContextChatEngine(
        llm=init_llm(),
        retriever=retriever,
        memory=ChatMemoryBuffer.from_defaults(token_limit=1500),
        prefix_messages=prompt
    )


# ---------------------------------------------------------------------------
# FIGURE RENDERING HELPER
# ---------------------------------------------------------------------------

def show_figure(fig, dpi=200):
    """
    Render a matplotlib figure centered in the Streamlit layout.

    st.pyplot() renders figures edge-to-edge. To center them, we instead:
    1. Save the figure to an in-memory PNG buffer (io.BytesIO) at the desired DPI.
    2. Display it with st.image() inside the middle column of a 3-column layout.

    dpi=200 is used for most figures; dpi=300 for the co-authorship network
    due to its high node density and label overlap.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    col = st.columns([2, 6, 2])   # Left padding | Figure | Right padding
    with col[1]:
        st.image(buf, use_container_width=True)


# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------
# Streamlit reruns the entire script on every user interaction.
# st.session_state persists variables across reruns within the same session.

if "mode" not in st.session_state:
    st.session_state.mode = None           # No mode selected initially
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # Stores all chat messages for display
if "rag_bot" not in st.session_state:
    st.session_state.rag_bot = None        # Chat engine instance (built on mode selection)


# ---------------------------------------------------------------------------
# INDEX LOADING
# ---------------------------------------------------------------------------
# Load or build the vector index at startup (cached after first run).
# all-MiniLM-L6-v2 was selected over all-mpnet-base-v2 after comparing
# retrieval quality on a set of test queries: it was faster with comparable results.

with st.spinner("Loading index..."):
    vector_index = load_or_create_index("minilm")
    retriever = vector_index.as_retriever(similarity_top_k=5)  # Retrieve top 5 most relevant abstracts per query


# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">🦠 GutFeeling — Microbiota Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Answers grounded exclusively in peer-reviewed PubMed literature · 56,000+ abstracts · 1980–2025</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB LAYOUT
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Literature Landscape"])


# ===========================================================================
# TAB 1 — CHATBOT
# ===========================================================================

with tab1:

    st.markdown("#### Choose your mode")
    col1, col2 = st.columns(2)

    # Mode selection buttons: clicking a button sets the mode in session_state
    # and rebuilds the chat engine with the appropriate system prompt.
    with col1:
        if st.button(
            "🌿  General Public\n— I want to understand my microbiota analysis",
            use_container_width=True,
            type="primary" if st.session_state.mode == "public" else "secondary"
        ):
            if st.session_state.mode != "public":
                st.session_state.mode = "public"
                st.session_state.rag_bot = build_chat_engine(retriever, CONSUMER_PROMPT)
                st.rerun()

    with col2:
        if st.button(
            "🔬  Scientist\n— I want to explore the scientific literature",
            use_container_width=True,
            type="primary" if st.session_state.mode == "scientist" else "secondary"
        ):
            if st.session_state.mode != "scientist":
                st.session_state.mode = "scientist"
                st.session_state.rag_bot = build_chat_engine(retriever, SCIENTIST_PROMPT)
                st.rerun()

    # Display active mode badge
    if st.session_state.mode == "public":
        st.markdown('<span class="active-mode-badge badge-public">🌿 General Public mode active</span>', unsafe_allow_html=True)
    elif st.session_state.mode == "scientist":
        st.markdown('<span class="active-mode-badge badge-scientist">🔬 Scientist mode active</span>', unsafe_allow_html=True)
    else:
        st.info("Please select a mode above to start chatting.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Render full chat history from session_state
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">{msg["content"]}', unsafe_allow_html=True)
            if msg.get("sources"):
                st.markdown('<div class="sources-label">Sources</div>', unsafe_allow_html=True)
                sources_html = ""
                for src in msg["sources"]:
                    pmid = src.get("pmid", "")
                    journal = src.get("journal", "Unknown")
                    year = src.get("year", "N/A")
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    sources_html += f'<a class="source-chip" href="{pubmed_url}" target="_blank">PMID {pmid} · {journal} · {year}</a>'
                st.markdown(sources_html + "</div>", unsafe_allow_html=True)

    # Chat input form
    # st.form with clear_on_submit=True clears the text field after each submission,
    # which is the expected UX behavior for a chat interface.
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="Ask a question",
                placeholder="e.g. What is the relationship between gut microbiota and depression?",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        # Append user message to history before querying the LLM
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Searching literature..."):
            response = st.session_state.rag_bot.chat(user_input)

        # Extract source metadata from retrieved nodes for citation display
        # A set is used to deduplicate PMIDs (multiple chunks may come from the same article)
        sources = []
        seen = set()
        if response.source_nodes:
            for node in response.source_nodes:
                pmid = node.metadata.get("pmid", "N/A")
                if pmid not in seen:
                    seen.add(pmid)
                    sources.append({
                        "pmid":    pmid,
                        "journal": node.metadata.get("journal", "Unknown"),
                        "year":    node.metadata.get("year", "N/A")
                    })

        st.session_state.chat_history.append({
            "role":    "ai",
            "content": response.response,
            "sources": sources
        })
        st.rerun()  # Rerun to display the new messages immediately


# ===========================================================================
# TAB 2 — LITERATURE LANDSCAPE
# ===========================================================================

with tab2:

    st.markdown("### 📊 Literature Landscape")
    st.markdown("Exploratory analysis of 55,990 peer-reviewed microbiota articles · PubMed 1980–2025")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("Loading corpus data..."):
        df_lit, df_mesh_lit = load_literature_data()

    # Selectbox for choosing the analysis to display
    analysis = st.selectbox(
        "Select analysis",
        [
            "📈 Temporal Evolution",
            "📰 Journal Landscape",
            "🏷️ MeSH Terms / Topics",
            "🌍 Global Health Topics by Decade",
            "🔗 Co-authorship Network"
        ]
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if analysis == "📈 Temporal Evolution":
        st.markdown("#### Publications per year (1980–2025)")
        st.markdown("Growth of microbiota research over time, annotated with key field milestones.")
        with st.spinner("Generating plot..."):
            fig = get_temporal(df_lit)
        show_figure(fig)

    elif analysis == "📰 Journal Landscape":
        st.markdown("#### Top 20 most prolific journals")
        st.markdown("Journals ranked by total number of microbiota publications.")
        with st.spinner("Generating plot..."):
            fig = get_journals(df_lit)
        show_figure(fig)

    elif analysis == "🏷️ MeSH Terms / Topics":
        st.markdown("#### Top 30 MeSH terms")
        st.markdown("Most frequent Medical Subject Headings across the corpus, excluding generic biomedical terms.")
        with st.spinner("Generating plot..."):
            fig = get_mesh(df_mesh_lit)
        show_figure(fig)

    elif analysis == "🌍 Global Health Topics by Decade":
        st.markdown("#### Global health topics by decade (% of total articles)")
        st.markdown("Normalized by total articles per decade to correct for the overall growth of the field.")
        with st.spinner("Generating plot..."):
            fig = get_disease_heatmap(df_lit, df_mesh_lit)
        show_figure(fig)

    elif analysis == "🔗 Co-authorship Network":
        st.markdown("#### Co-authorship network")
        st.markdown("Authors with 30+ publications connected by 3+ shared articles. Node size = publications · Color = research community.")
        st.warning("⚠️ This analysis takes ~2 minutes to generate. It will be cached after the first run.")
        with st.spinner("Building network... this may take a moment"):
            fig = get_network(df_lit)
        show_figure(fig, dpi=300)  # Higher DPI for dense network with many labels
        st.markdown("""
        > **How to read this network:** Each dot is a researcher. Larger dots = more publications.
        > Connected dots have co-authored at least 3 papers together. Colors represent distinct research communities detected automatically.
        > Central figures like **Rob Knight** (microbiome diversity) and **Curtis Huttenhower** (computational metagenomics)
        > are the most connected hubs of the field.
        """)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown("""
<footer>
    Built with LlamaIndex · Groq · HuggingFace · Streamlit &nbsp;|&nbsp;
    Data source: PubMed / NCBI &nbsp;|&nbsp;
    AlbertoSPJ — WBS Coding School (2026)
</footer>
""", unsafe_allow_html=True)
