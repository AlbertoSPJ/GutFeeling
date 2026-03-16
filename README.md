# 🦠 GutFeeling — Microbiota RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about gut microbiota research, grounded exclusively in peer-reviewed PubMed literature. Built as a final project for the WBS Coding School Data Science Bootcamp (Berlin, 2026).

---

## 🌿 Live Demo

👉 **[Launch GutFeeling](https://albertospj-gutfeeling.hf.space)**

---

## 📌 Features

- **RAG Chatbot** — answers grounded in 56,000+ PubMed abstracts (1980–2025)
- **Two modes** — General Public (plain language) and Scientist (technical)
- **Source citations** — every answer links to the original PubMed articles
- **Literature Landscape** — exploratory analysis of the microbiota corpus:
  - Publications per year with milestone annotations
  - Top 20 most prolific journals
  - Top 30 MeSH terms
  - Global health topics by decade (heatmap)
  - Co-authorship network of leading researchers

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| RAG Framework | LlamaIndex |
| Frontend | Streamlit |
| Data | PubMed / NCBI Entrez API |
| Visualization | Matplotlib, Seaborn, NetworkX |

---

## 🗂️ Project Structure
```
Final_Project/
├── GutFeeling.py                   # Main Streamlit app
├── analysis.py                     # Literature landscape analysis functions
├── 01_pubmed_fetch.py              # Retrieves articles from PubMed via NCBI Entrez API
├── 02_prepare_rag_jsonl.py         # Converts PubMed JSON to JSONL format for LlamaIndex
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files excluded from version control
└── .env                            # API keys (not included in repo)
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/AlbertoSPJ/gutfeeling.git
cd gutfeeling
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API keys**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_ncbi_api_key_here
```

**4. Download the data**

The PubMed corpus is not included in this repository due to file size. Run the fetch scripts to download it:
```bash
python 01_pubmed_fetch.py
python 02_prepare_rag_jsonl.py
```

**5. Run the app**
```bash
streamlit run GutFeeling.py
```

The vector index will be built automatically on first run (~5 minutes).

---

## 📊 Data

- **Source:** PubMed / NCBI Entrez API
- **Query:** "gut microbiota" OR "gut microbiome"
- **Coverage:** 55,990 articles · 1980–2025
- **Fields:** title, abstract, authors, journal, year, MeSH terms

---

## 🧠 Architecture
```
User query
    │
    ▼
HuggingFace Embeddings (MiniLM-L6-v2)
    │
    ▼
Vector Index (LlamaIndex) ──► Top 5 relevant abstracts
    │
    ▼
Llama 3.3 70B (Groq) + System Prompt (Public / Scientist)
    │
    ▼
Answer + PubMed source links
```

---

## 👤 Author

**Alberto Sánchez-Pascuala**  
PhD in Microbiology · Data Scientist

[LinkedIn](https://linkedin.com/in/alberto-sanchez-pascuala-jerez) · [GitHub](https://github.com/AlbertoSPJ)

---

## 📄 License

© 2026 Alberto Sánchez-Pascuala. All rights reserved.
Contact the author for permissions.