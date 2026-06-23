import ir_datasets

dataset = ir_datasets.load("mr-tydi/ja/dev")

print("=== QUERIES ===")
for i, query in enumerate(dataset.queries_iter()):
    print(query)
    if i == 2:
        break

print("\n=== DOCS ===")
for i, doc in enumerate(dataset.docs_iter()):
    print(doc)
    if i == 2:
        break

print("\n=== QRELS ===")
for i, qrel in enumerate(dataset.qrels_iter()):
    print(qrel)
    if i == 2:
        break