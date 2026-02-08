"""
Streamlit search UI for Danish politician Facebook posts.
Social-media-style feed with semantic search.

Run with: streamlit run app.py
"""

import html

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "politik_posts"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Custom CSS for social media look ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.post-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.post-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.post-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}
.profile-avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1877f2, #42b72a);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 18px;
    margin-right: 12px;
    flex-shrink: 0;
}
.profile-info {
    flex: 1;
}
.profile-name {
    font-weight: 600;
    font-size: 15px;
    color: #1a1a1a;
    text-decoration: none;
}
.profile-name:hover {
    text-decoration: underline;
}
.post-date {
    font-size: 13px;
    color: #8a8a8a;
}

.post-text {
    font-size: 15px;
    line-height: 1.55;
    color: #1a1a1a;
    margin-bottom: 14px;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.post-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 12px;
    border-top: 1px solid #f0f0f0;
}
.interactions {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    color: #606770;
    font-weight: 500;
}
.score-badge {
    background: #f0f2f5;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 12px;
    color: #606770;
    font-weight: 500;
}
.fb-link {
    font-size: 13px;
    color: #1877f2;
    text-decoration: none;
    font-weight: 500;
}
.fb-link:hover {
    text-decoration: underline;
}

.result-count {
    font-size: 14px;
    color: #8a8a8a;
    margin-bottom: 16px;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 10px 0 20px 0;
}
.app-title {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 4px;
}
.app-subtitle {
    font-size: 15px;
    color: #8a8a8a;
}

/* Search box */
div[data-testid="stTextInput"] input {
    border-radius: 24px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    border: 2px solid #e0e0e0 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #1877f2 !important;
    box-shadow: 0 0 0 2px rgba(24,119,242,0.2) !important;
}
</style>
"""


def format_interactions(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


@st.cache_data
def get_profiles() -> list[str]:
    """Get all unique profile names from the collection."""
    collection = load_collection()
    # Fetch a sample to get profiles — ChromaDB doesn't have a distinct query
    # so we get all metadata
    results = collection.get(include=["metadatas"], limit=collection.count())
    profiles = sorted(set(m["profile"] for m in results["metadatas"] if m.get("profile")))
    return profiles


def search(query, top_k, profile, min_interactions, boost_interactions):
    collection = load_collection()
    model = load_model()
    query_embedding = model.encode([query])[0].tolist()

    conditions = []
    if profile and profile != "Alle":
        conditions.append({"profile": {"$eq": profile}})
    if min_interactions > 0:
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


def render_post(item: dict, rank: int):
    meta = item["metadata"]
    profile_raw = meta.get("profile", "Ukendt")
    initials = "".join(w[0] for w in profile_raw.split()[:2] if w).upper() or "?"
    profile = html.escape(profile_raw)
    date = meta.get("date", "")
    interactions = meta.get("total_interactions", 0)
    post_url = meta.get("post_url", "")
    text = html.escape(item["text"])

    sim = item["similarity"]
    hybrid = item.get("hybrid_score")
    score_label = f"{sim:.0%} match"
    if hybrid:
        score_label = f"{hybrid:.0%} hybrid"

    st.markdown(f"""
    <div class="post-card">
        <div class="post-header">
            <div class="profile-avatar">{initials}</div>
            <div class="profile-info">
                <div class="profile-name">{profile}</div>
                <div class="post-date">{date}</div>
            </div>
            <span class="score-badge">#{rank} &middot; {score_label}</span>
        </div>
        <div class="post-text">{text}</div>
        <div class="post-footer">
            <span class="interactions">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="#1877f2"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>
                {format_interactions(interactions)}
            </span>
            <a class="fb-link" href="{post_url}" target="_blank">Se på Facebook &rarr;</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Page config ---
st.set_page_config(
    page_title="Dansk Politik Søgning",
    page_icon="🏛️",
    layout="centered",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="app-header">
    <div class="app-title">🏛️ Dansk Politik Søgning</div>
    <div class="app-subtitle">Semantisk søgning i danske politikeres Facebook-opslag</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Filtre")

    profiles = ["Alle"] + get_profiles()
    profile = st.selectbox("Politiker / profil", profiles, index=0)

    min_interactions = st.slider("Minimum interaktioner", 0, 50000, 0, step=500)
    top_k = st.slider("Antal resultater", 1, 20, 5)
    boost_interactions = st.checkbox(
        "Boost populære opslag",
        value=False,
        help="Blander semantisk lighed med interaktionstal"
    )

    st.markdown("---")
    collection = load_collection()
    st.markdown(f"**{collection.count():,}** opslag indekseret")
    st.markdown(f"**{len(profiles) - 1}** profiler")

# --- Search ---
query = st.text_input("", placeholder="Søg efter emne, holdning, stemning...")

if query:
    with st.spinner("Søger..."):
        results = search(query, top_k, profile, min_interactions, boost_interactions)

    if not results:
        st.warning("Ingen resultater. Prøv at justere filtrene eller søgeordet.")
    else:
        st.markdown(f'<div class="result-count">{len(results)} resultater for "{query}"</div>', unsafe_allow_html=True)
        for i, item in enumerate(results, 1):
            render_post(item, i)
else:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #8a8a8a;">
        <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
        <div style="font-size: 16px;">Skriv et søgeord for at finde relevante opslag</div>
        <div style="font-size: 14px; margin-top: 8px;">Prøv f.eks. "Grønland", "sundhed", "vrede opslag om kriminalitet"</div>
    </div>
    """, unsafe_allow_html=True)
