import streamlit as st
import faiss
import json
import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer
from rocchio import rocchio

st.set_page_config(page_title="Cross-Lingual Retrieval", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; max-width: 900px; }
        .doc-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem; margin-bottom: 0.75rem; }
        .doc-id { font-size: 0.75rem; color: #888; margin-bottom: 0.4rem; }
        .score { font-size: 0.75rem; color: #555; }
        h1 { font-size: 1.6rem; font-weight: 600; }
        .stButton button { border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    index = faiss.read_index("index/corpus.index")
    with open("index/doc_ids.json") as f:
        doc_ids = json.load(f)
    dataset = ir_datasets.load("mr-tydi/ja/dev")
    docs_lookup = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
    return model, index, doc_ids, docs_lookup

model, index, doc_ids, docs_lookup = load_resources()

def retrieve(query_vec, top_k=10):
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc_id = doc_ids[idx]
        text = docs_lookup.get(doc_id, "")
        results.append({"doc_id": doc_id, "score": float(score), "text": text, "idx": int(idx)})
    return results

# --- UI ---

st.title("Cross-Lingual Retrieval")
st.caption("English query → Japanese document retrieval using LaBSE embeddings")

query = st.text_input("Enter a query in English", placeholder="e.g. Where was the founder of Nintendo born?")

if st.button("Search") and query:
    query_vec = model.encode([query], normalize_embeddings=True, device="cuda")
    results = retrieve(query_vec)
    st.session_state.results = results
    st.session_state.query_vec = query_vec
    st.session_state.feedback_done = False
    st.session_state.reranked = None

if "results" in st.session_state:
    st.divider()

    if not st.session_state.get("feedback_done"):
        st.subheader("Results — mark relevant documents")
        st.caption("Select documents that answer your query, then click Apply Feedback.")

        feedback = {}
        for r in st.session_state.results:
            with st.container():
                st.markdown(f'<div class="doc-card"><div class="doc-id">ID: {r["doc_id"]} &nbsp;|&nbsp; <span class="score">Score: {r["score"]:.4f}</span></div>{r["text"][:300]}...</div>', unsafe_allow_html=True)
                feedback[r["doc_id"]] = st.checkbox("Relevant", key=r["doc_id"])

        if st.button("Apply Feedback"):
            relevant_vecs, non_relevant_vecs = [], []
            for r in st.session_state.results:
                vec = index.reconstruct(r["idx"])
                if feedback[r["doc_id"]]:
                    relevant_vecs.append(vec)
                else:
                    non_relevant_vecs.append(vec)

            if not relevant_vecs:
                st.warning("Mark at least one document as relevant.")
            else:
                new_vec = rocchio(
                    st.session_state.query_vec[0],
                    np.array(relevant_vecs),
                    np.array(non_relevant_vecs)
                )
                reranked = retrieve(new_vec.reshape(1, -1), top_k=10)
                st.session_state.reranked = reranked
                st.session_state.feedback_done = True
                st.rerun()

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before feedback")
            for r in st.session_state.results:
                st.markdown(f'<div class="doc-card"><div class="doc-id">ID: {r["doc_id"]} &nbsp;|&nbsp; <span class="score">Score: {r["score"]:.4f}</span></div>{r["text"][:250]}...</div>', unsafe_allow_html=True)

        with col2:
            st.subheader("After feedback")
            for r in st.session_state.reranked:
                st.markdown(f'<div class="doc-card"><div class="doc-id">ID: {r["doc_id"]} &nbsp;|&nbsp; <span class="score">Score: {r["score"]:.4f}</span></div>{r["text"][:250]}...</div>', unsafe_allow_html=True)

        if st.button("New search"):
            for key in ["results", "query_vec", "feedback_done", "reranked"]:
                del st.session_state[key]
            st.rerun()