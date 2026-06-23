import ir_datasets
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

# Config
SUBSET_SIZE = 50000
BATCH_SIZE = 256
OUTPUT_DIR = "index"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

#we load the model first
print("Loading LaBSE...")
model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")

#loading datasetr
print("Loading corpus...")
dataset = ir_datasets.load("mr-tydi/ja/dev")
docs = []
doc_ids = []
for i, doc in enumerate(dataset.docs_iter()):
    doc_ids.append(doc.doc_id)
    docs.append(doc.text)
    if i >= SUBSET_SIZE - 1:
        break
print(f"Loaded {len(docs)} documents")

# Embedding
print("Embedding documents...")
embeddings = model.encode(
    docs,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True, #needed for cosine similarity ok
    device="cuda"
)
print(f"Embeddings shape: {embeddings.shape}")
#FAISSSSindex
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product vs IRF hmm..since small dataset, we can use flat index, for larger datasets, we cld consider IVF 
index.add(embeddings.astype(np.float32))
print(f"Index total vectors: {index.ntotal}")
#Saving 
faiss.write_index(index, f"{OUTPUT_DIR}/corpus.index")

with open(f"{OUTPUT_DIR}/doc_ids.json", "w") as f:
    json.dump(doc_ids, f)

print("Done. Saved to index/")