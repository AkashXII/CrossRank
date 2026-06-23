import streamlit as st
import faiss
import json
import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from rocchio import rocchio

st.set_page_config(page_title="Cross-Lingual Retrieval", layout="centered")

@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    index = faiss.read_index("index/corpus.index")
    with open("index/doc_ids.json") as f:
        doc_ids = json.load(f)
    dataset = ir_datasets.load("mr-tydi/ja/dev")
    docs_lookup = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
    return model, index, doc_ids, docs_lookup

@st.cache_data
def translate_text(text):
    try:
        return GoogleTranslator(source="ja", target="en").translate(text)
    except Exception:
        return "Translation unavailable."

model, index, doc_ids, docs_lookup = load_resources()

def retrieve(query_vec, top_k=10):
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc_id = doc_ids[idx]
        text = docs_lookup.get(doc_id, "")
        results.append({
            "doc_id": doc_id,
            "score": float(score),
            "text": text,
            "idx": int(idx)
        })
    return results

def render_card(r, rank=None, show_checkbox=False):
    snippet = r["text"][:300]
    translation = translate_text(snippet)

    rank_str = f"Rank #{rank} · " if rank else ""
    st.caption(f"{rank_str}ID: {r['doc_id']} · Score: {r['score']:.4f}")
    st.write(translation)

    with st.expander("Show Original Japanese"):
        st.write(snippet)

    relevant = False
    if show_checkbox:
        relevant = st.checkbox("Relevant", key=r["doc_id"])

    st.divider()
    return relevant

# --- UI ---

st.title("Cross-Lingual Retrieval")
st.caption("English query → Japanese document retrieval using LaBSE embeddings")

query = st.text_input(
    "Enter a query in English",
    placeholder="e.g. Where was the founder of Nintendo born?"
)

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
            feedback[r["doc_id"]] = render_card(r, show_checkbox=True)

        if st.button("Apply Feedback"):
            relevant_vecs = []
            non_relevant_vecs = []

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
        st.subheader("Updated Results After Feedback")
        st.caption("Results re-ranked using Rocchio relevance feedback.")

        for rank, r in enumerate(st.session_state.reranked, start=1):
            render_card(r, rank=rank)

        if st.button("New search"):
            for key in ["results", "query_vec", "feedback_done", "reranked"]:
                del st.session_state[key]
            st.rerun()