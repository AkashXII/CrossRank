import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ir_datasets
#testing file
model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
index = faiss.read_index("index/corpus.index")
with open("index/doc_ids.json") as f:
    doc_ids = json.load(f)
# Load docs into memory for display
dataset = ir_datasets.load("mr-tydi/ja/dev")
docs_lookup = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
def retrieve(query, top_k=10):
    query_vec = model.encode([query], normalize_embeddings=True, device="cuda")
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc_id = doc_ids[idx]
        text = docs_lookup.get(doc_id, "")
        results.append((doc_id, score, text))
    return results
#testing
query = "Where is mount fuji?"
print(f"Query: {query}\n")
results = retrieve(query)
for i, (doc_id, score, text) in enumerate(results):
    print(f"Rank {i+1} | Score: {score:.4f} | ID: {doc_id}")
    print(f"{text[:200]}")
    print()

   
#notes:    
#a query comes then fiass stores the position and its corresponding embedding, wehn a query comes it retreives the topk positions and matches to doc_id to get the actual id and get the document.