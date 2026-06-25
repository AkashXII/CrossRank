import faiss
import json
import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
#japanese to japanese
#the comments are for my reference to learn as i build :))
model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
index = faiss.read_index("index/corpus.index")

with open("index/doc_ids.json") as f:
    doc_ids = json.load(f)

dataset = ir_datasets.load("mr-tydi/ja/dev")

#creating a dict for the qrels for easier access cus qrels in the dataset have multiple entries per query. now we can easily get the relevant doc ids for each query by looking up the query id in this dict.
qrels={}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels:
        qrels[qrel.query_id] = set()
    qrels[qrel.query_id].add(qrel.doc_id)

#all the queries from the dataset
queries = []
for q in dataset.queries_iter():
    queries.append((q.query_id, q.text))

#we jus adding rel benchmark queries which are there in our 50000 subset
doc_ids_set = set(doc_ids)
filtered_queries = []
for qid, qtext in queries:
    relevant_docs = qrels.get(qid, set())
    keep_query = False
    for doc_id in relevant_docs:
        if doc_id in doc_ids_set:
            keep_query = True
            break
    if keep_query:
        filtered_queries.append((qid, qtext))
queries = filtered_queries
print(f"Evaluable queries: {len(queries)}")


query_texts = []
for q in queries:
    query_texts.append(q[1])
query_vecs = model.encode(query_texts, normalize_embeddings=True,device="cuda", show_progress_bar=True)
# Retrieve top 100
scores, indices = index.search(query_vecs.astype(np.float32), 100)


mrr_scores = []
recall_scores = []
for i, (qid, _) in enumerate(queries):
    relevant = qrels.get(qid, set())
    top100 = [doc_ids[idx] for idx in indices[i]]
    #Recall@100
    hits = sum(1 for d in top100 if d in relevant)
    recall_scores.append(hits / len(relevant))
    # MRR@10
    mrr = 0.0
    for rank, doc_id in enumerate(top100[:10], start=1):
        if doc_id in relevant:
            mrr = 1.0 / rank
            break
    mrr_scores.append(mrr)
print(f"MRR@10:      {np.mean(mrr_scores):.4f}")
print(f"Recall@100:  {np.mean(recall_scores):.4f}")


from rocchio import rocchio
# Evaluate with simulated relevance feedback
mrr_rf_scores = []
recall_rf_scores = []
for i, (qid, _) in enumerate(queries):
    relevant = qrels.get(qid, set())
    top100_ids = [doc_ids[idx] for idx in indices[i]]
    top100_vecs = np.array([index.reconstruct(int(idx)) for idx in indices[i]])
    # Simulate feedback: mark top 10 as relevant/non-relevant based on qrels
    relevant_vecs = []
    non_relevant_vecs = []
    for j, doc_id in enumerate(top100_ids[:10]):
        if doc_id in relevant:
            relevant_vecs.append(top100_vecs[j])
        else:
            non_relevant_vecs.append(top100_vecs[j])
    # If no relevant docs in top 10, skip feedback
    if not relevant_vecs:
        mrr_rf_scores.append(mrr_scores[i])
        recall_rf_scores.append(recall_scores[i])
        continue
    # Apply Rocchio
    new_query_vec = rocchio(
        query_vecs[i],
        np.array(relevant_vecs),
        np.array(non_relevant_vecs)
    )
    # Re-retrieve
    new_scores, new_indices = index.search(
        new_query_vec.reshape(1, -1).astype(np.float32), 100
    )
    new_top100 = [doc_ids[idx] for idx in new_indices[0]]
    # Recall@100
    hits = sum(1 for d in new_top100 if d in relevant)
    recall_rf_scores.append(hits / len(relevant))
    # MRR@10
    mrr = 0.0
    for rank, doc_id in enumerate(new_top100[:10], start=1):
        if doc_id in relevant:
            mrr = 1.0 / rank
            break
    mrr_rf_scores.append(mrr)

print(f"\nAfter Relevance Feedback ")
print(f"MRR@10:      {np.mean(mrr_rf_scores):.4f}")
print(f"Recall@100:  {np.mean(recall_rf_scores):.4f}")