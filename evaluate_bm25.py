import ir_datasets
import numpy as np
import MeCab
from rank_bm25 import BM25Okapi
from tqdm import tqdm
#this is just for my curiosity to see how bm25 performs on japanese to japanese
mecab = MeCab.Tagger("-Owakati")

def tokenize(text):
    return mecab.parse(text).strip().split()

dataset = ir_datasets.load("mr-tydi/ja/dev")

# Build qrels
qrels = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels:
        qrels[qrel.query_id] = set()
    qrels[qrel.query_id].add(qrel.doc_id)

# Load same 50k subset
print("Loading and tokenizing corpus...")
doc_ids = []
tokenized_docs = []

for i, doc in enumerate(tqdm(dataset.docs_iter(), total=50000)):
    doc_ids.append(doc.doc_id)
    tokenized_docs.append(tokenize(doc.text))
    if i >= 49999:
        break

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_docs)

# Load evaluable queries
queries = [(q.query_id, q.text) for q in dataset.queries_iter()]
doc_ids_set = set(doc_ids)
queries = [(qid, qtext) for qid, qtext in queries
           if any(d in doc_ids_set for d in qrels.get(qid, set()))]

print(f"Evaluable queries: {len(queries)}")

mrr_scores = []
recall_scores = []

for qid, qtext in tqdm(queries):
    relevant = qrels.get(qid, set())
    tokenized_query = tokenize(qtext)

    scores = bm25.get_scores(tokenized_query)
    top100_indices = np.argsort(scores)[::-1][:100]
    top100_ids = [doc_ids[i] for i in top100_indices]

    hits = sum(1 for d in top100_ids if d in relevant)
    recall_scores.append(hits / len(relevant))

    mrr = 0.0
    for rank, doc_id in enumerate(top100_ids[:10], start=1):
        if doc_id in relevant:
            mrr = 1.0 / rank
            break
    mrr_scores.append(mrr)

print(f"BM25 MRR@10:     {np.mean(mrr_scores):.4f}")
print(f"BM25 Recall@100: {np.mean(recall_scores):.4f}")
