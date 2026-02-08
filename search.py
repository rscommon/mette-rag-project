"""
Search the politician posts index.

Supports:
- Semantic search (query by meaning — topic, sentiment, tone all captured in embeddings)
- Profile filtering (search within one politician)
- Interaction filtering / boosting
"""

import argparse

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "politik_posts"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def search(
    query: str,
    top_k: int = 5,
    profile: str | None = None,
    min_interactions: int | None = None,
    boost_interactions: bool = False,
):
    collection = get_collection()
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    # Build metadata filter
    conditions = []
    if profile:
        conditions.append({"profile": {"$eq": profile}})
    if min_interactions is not None:
        conditions.append({"total_interactions": {"$gte": min_interactions}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    fetch_k = top_k * 3 if boost_interactions else top_k

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    items = []
    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1 - (dist / 2)
        items.append({"text": doc, "metadata": meta, "similarity": similarity})

    if boost_interactions and items:
        max_inter = max(it["metadata"]["total_interactions"] for it in items)
        if max_inter > 0:
            for it in items:
                norm = it["metadata"]["total_interactions"] / max_inter
                it["hybrid_score"] = 0.7 * it["similarity"] + 0.3 * norm
            items.sort(key=lambda x: x.get("hybrid_score", x["similarity"]), reverse=True)

    return items[:top_k]


def print_results(items: list[dict], query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Results: {len(items)}")
    print(f"{'='*60}\n")

    for i, item in enumerate(items, 1):
        meta = item["metadata"]
        score_str = f"sim={item['similarity']:.3f}"
        if "hybrid_score" in item:
            score_str += f"  hybrid={item['hybrid_score']:.3f}"

        print(f"--- #{i} [{score_str}] ---")
        print(f"  {meta['profile']}  |  {meta['date']}  |  {meta['total_interactions']:,} interactions")
        print(f"  {meta['post_url']}")
        text = item["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"  {text}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Search politician posts")
    parser.add_argument("query", help="Search query (Danish or English)")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("-p", "--profile", help="Filter by politician/profile name")
    parser.add_argument("-m", "--min-interactions", type=int, help="Minimum interactions")
    parser.add_argument("-b", "--boost-interactions", action="store_true", help="Boost popular posts")

    args = parser.parse_args()

    items = search(
        query=args.query,
        top_k=args.top_k,
        profile=args.profile,
        min_interactions=args.min_interactions,
        boost_interactions=args.boost_interactions,
    )
    print_results(items, args.query)


if __name__ == "__main__":
    main()
