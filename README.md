## CrossRank (かけ橋): Cross-Lingual Japanese Information Retrieval with Rocchio Feedback

CrossRank is a cross-lingual information retrieval system that lets English queries retrieve relevant Japanese documents using multilingual embeddings. The project combines dense retrieval, classical BM25 baseline experimentation, and Rocchio relevance feedback to study how ranking quality changes before and after feedback.

---

## Overview

This project explores:

- **Cross-lingual retrieval**: English queries over a Japanese corpus
- **Dense semantic search** using **LaBSE**
- **Classical retrieval** using **BM25**
- **Interactive relevance feedback** using **Rocchio**
- **Retrieval evaluation** using **MRR@10** and **Recall@100**

The system includes an interactive Streamlit demo where users can enter an English query, inspect retrieved Japanese documents, mark results as relevant/not relevant, and re-rank results using Rocchio feedback.

---

## Features

- English query → Japanese document retrieval
- Dense embedding search with **LaBSE**
- Vector indexing with **FAISS**
- Interactive relevance feedback with **Rocchio**
- Japanese snippet display with English translation for usability
- Retrieval evaluation with **MRR@10** and **Recall@100**
- Parameter sensitivity analysis for Rocchio
- Exploratory BM25 baseline comparison on Japanese queries

---
## Architecture

CrossRank follows a two-stage retrieval pipeline with relevance feedback. Japanese documents are embedded offline using LaBSE and indexed in FAISS. During retrieval, an English query is encoded into the same embedding space, used to retrieve the most similar documents, and then refined using Rocchio relevance feedback before a second retrieval pass.

![CrossRank Architecture](assets/CrossRank4.jpg)

### Pipeline Overview

1. **Offline Indexing** – Encode 50k Japanese documents using LaBSE and store them in a FAISS IndexFlatIP index.
2. **Query Encoding** – Convert the English query into the shared multilingual embedding space.
3. **Initial Retrieval** – Retrieve the top-K most similar documents from FAISS.
4. **User Feedback** – Mark retrieved documents as relevant or non-relevant.
5. **Rocchio Feedback** – Update the query vector using relevance feedback.
6. **Second Retrieval Pass** – Search the same FAISS index using the updated query vector.
7. **Re-ranked Results** – Present the updated ranking to the user.

The architecture separates offline indexing from online retrieval and demonstrates how classical relevance feedback can improve cross-lingual retrieval effectiveness without modifying the underlying document index.

## Dataset

The project uses the Japanese split of the **Mr. TyDi** retrieval benchmark.

### Corpus
- 50,000 Japanese documents were indexed for experimentation
- Documents are stored as dense embeddings in FAISS

### Queries
- The benchmark provides Japanese queries with relevance judgments
- For cross-lingual evaluation, the 33 evaluable queries were translated into English and evaluated against the same relevance judgments

### Qrels
- Qrels provide query-to-relevant-document mappings
- These were used for retrieval evaluation and simulated feedback experiments

---

## Methodology

### 1. Corpus Indexing
- Japanese documents were embedded using **LaBSE**
- Embeddings were normalized
- A **FAISS IndexFlatIP** index was built for similarity search

### 2. Retrieval
- A query is embedded using the same multilingual encoder
- FAISS retrieves the top-k nearest documents by similarity

### 3. Relevance Feedback
- Rocchio feedback updates the query vector using:
  - the original query vector
  - relevant document vectors
  - non-relevant document vectors
- The updated query is re-run through the retriever to obtain a new ranking

### 4. Evaluation
Retrieval quality was measured using:

- **MRR@10**: how early the first relevant result appears
- **Recall@100**: how many relevant documents appear in the top 100

---

## Key Findings

- LaBSE successfully supported cross-lingual retrieval, allowing English queries to retrieve relevant Japanese documents with only a small drop in performance compared to Japanese queries.
- Rocchio relevance feedback substantially improved ranking quality, increasing MRR@10 from 0.1781 to 0.4848 in the English→Japanese setting while maintaining Recall@100.
- Retrieval improvements remained stable across multiple Rocchio coefficient configurations, suggesting that the relevance feedback signal itself contributed more to performance gains than specific parameter choices.
- An exploratory BM25 baseline showed stronger lexical precision on some Japanese queries but lower recall than dense retrieval, highlighting the tradeoff between lexical and semantic search approaches.

Detailed experimental results are available in results_50k.json.
---

## Demo Flow

1. User enters an English query
2. The system retrieves top Japanese documents using LaBSE + FAISS
3. Retrieved snippets are shown with English translation for readability
4. User marks documents as relevant or not relevant
5. Rocchio updates the query vector
6. The system re-ranks and displays updated results

---

## Project Structure

```text
.
├── embed_corpus.py
├── evaluate.py
├── evaluate_bm25.py
├── rocchio.py
├── app.py
├── results.json
├── index/
│   ├── corpus.index
│   └── doc_ids.json
└── README.md
