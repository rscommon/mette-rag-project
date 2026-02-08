"""
Build the ChromaDB index from the all_politics.csv dataset.

- Loads all politician Facebook posts (semicolon-delimited CSV)
- Recovers missing profiles from ccpageid mapping or facebook_url slugs
- Embeds post_text with a multilingual sentence-transformer
- Stores in persistent ChromaDB with metadata (profile, date, interactions)
"""

import csv
import re
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/all_politics.csv"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "politik_posts"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256


def is_meaningful(text: str) -> bool:
    """Return False for emoji-only or very short posts with no real text content."""
    stripped = re.sub(r'[\U0001F000-\U0001FFFF\U00002600-\U000027BF\U0000FE00-\U0000FEFF\u200d\u20e3\ufe0f]', '', text)
    stripped = re.sub(r'[\s\.\,\!\?\:\;\-\—\–\'\"\(\)\[\]\{\}]+', '', stripped)
    return len(stripped) >= 20


def load_data(csv_path: str) -> list[dict]:
    """Load CSV, recover profiles, discard rows without profile or post_text."""

    # First pass: build ccpageid -> profile mapping from rows that have profiles
    pageid_to_profile = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f, delimiter=";"):
            profile = (row.get("profile") or "").strip()
            pid = (row.get("ccpageid") or "").strip()
            if profile and profile != "None" and pid:
                pageid_to_profile[pid] = profile

    # Second pass: load all rows, recover missing profiles
    posts = []
    skipped = 0
    recovered_pageid = 0
    recovered_url = 0

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f, delimiter=";"):
            text = (row.get("post_text") or "").strip()
            if not text or not is_meaningful(text):
                skipped += 1
                continue

            profile = (row.get("profile") or "").strip()
            if not profile or profile == "None":
                # Try recovery from ccpageid
                pid = (row.get("ccpageid") or "").strip()
                if pid in pageid_to_profile:
                    profile = pageid_to_profile[pid]
                    recovered_pageid += 1
                else:
                    # Try recovery from facebook_url slug
                    url = (row.get("facebook_url") or "").strip()
                    if url:
                        slug = url.rstrip("/").split("/")[-1]
                        if not slug.isdigit() and slug:
                            profile = slug
                            recovered_url += 1
                        else:
                            skipped += 1
                            continue
                    else:
                        skipped += 1
                        continue

            posts.append({
                "id": row.get("ccpost_id", ""),
                "profile": profile,
                "date": row.get("date", ""),
                "total_interactions": int(row.get("total_interactions", 0) or 0),
                "post_url": row.get("post_url", ""),
                "facebook_url": row.get("facebook_url", ""),
                "text": text,
            })

    print(f"Loaded {len(posts)} posts ({skipped} skipped)")
    print(f"  Recovered via ccpageid: {recovered_pageid}")
    print(f"  Recovered via URL slug: {recovered_url}")

    profiles = defaultdict(int)
    for p in posts:
        profiles[p["profile"]] += 1
    print(f"  Unique profiles: {len(profiles)}")
    return posts


def build_index(posts: list[dict]):
    # Compute embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [p["text"] for p in posts]
    print(f"Embedding {len(texts)} posts (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)

    # Build ChromaDB
    print("Building ChromaDB index...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches (ChromaDB has limits on single add)
    chroma_batch = 5000
    for i in range(0, len(posts), chroma_batch):
        batch_end = min(i + chroma_batch, len(posts))
        batch_posts = posts[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]

        collection.add(
            ids=[p["id"] for p in batch_posts],
            embeddings=batch_embeddings.tolist(),
            documents=[p["text"] for p in batch_posts],
            metadatas=[
                {
                    "profile": p["profile"],
                    "date": p["date"],
                    "total_interactions": p["total_interactions"],
                    "post_url": p["post_url"],
                    "facebook_url": p["facebook_url"],
                }
                for p in batch_posts
            ],
        )
        print(f"  Indexed {batch_end}/{len(posts)}")

    print(f"Done! {collection.count()} posts indexed in {CHROMA_DIR}")


if __name__ == "__main__":
    posts = load_data(CSV_PATH)
    build_index(posts)
