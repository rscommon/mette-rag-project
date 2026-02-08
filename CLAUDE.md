# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A semantic search engine for Danish politicians' Facebook posts. Embeds posts with a multilingual model and stores them in a local vector database, enabling search by meaning — topic, tone, sentiment are all captured in the embeddings without needing separate models.

## Data

- **`data/all_politics.csv`** — ~77K Facebook posts from ~286 Danish politicians (semicolon-delimited)
- Columns: `ccpost_id`, `ccpageid`, `total_interactions`, `date`, `country`, `profile`, `facebook_url`, `category`, `post_url`, `post_text`
- ~37K rows have empty/None profiles — recovered via `facebook_url` slug where possible, rest discarded
- Final indexed dataset: ~56K posts
- Posts are in Danish, spanning approximately the past year
- The `data/` directory is gitignored

## Language

The dataset is in Danish. When building prompts, retrieval queries, or UI text that interacts with this data, default to Danish unless otherwise specified.

## Architecture

### Stack (all free, local)

| Component | Tool | Model / Details |
|---|---|---|
| Embeddings | `sentence-transformers` | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| Vector store | ChromaDB | Persistent, cosine similarity, stored in `./chroma_db/` |
| Web UI | Streamlit | Social-media-style feed at `localhost:8501` |

No separate sentiment model — the embeddings capture topic, tone, and sentiment naturally. Searching "vrede opslag om kriminalitet" finds angry posts about crime.

### Scripts

- **`build_index.py`** — Loads `data/all_politics.csv`, recovers missing profiles from URL slugs, embeds all posts, stores in ChromaDB. Run once (or re-run to rebuild).
- **`search.py`** — CLI search with profile filtering and interaction boosting.
- **`app.py`** — Streamlit web UI with social-media-style post cards.

### How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Build the index (~10 min on CPU for ~56K posts)
python build_index.py

# Web UI
streamlit run app.py
# Opens at http://localhost:8501

# CLI search
python search.py "Grønland"
python search.py "sundhed" --profile "Mette Frederiksen"
python search.py "klima" --min-interactions 5000 --boost-interactions
```

### Key directories

- `./data/` — Raw CSV data (gitignored)
- `./chroma_db/` — Persistent ChromaDB index (gitignored, rebuild with `build_index.py`)
