import json

from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore


QUERY_FILE = "data/evaluation/retrieval_queries.json"
INDEX_PATH = "data/vector_store/index.faiss"
METADATA_PATH = "data/vector_store/metadata.json"


def load_queries():
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_retrieved_documents(results, k):
    return [
        result["document_id"]
        for result in results[:k]
    ]


def recall_at_k(results, relevant_documents, k):
    retrieved = set(
        get_retrieved_documents(results, k)
    )

    relevant = set(relevant_documents)

    return int(bool(retrieved & relevant))


def precision_at_k(results, relevant_documents, k):
    retrieved = get_retrieved_documents(results, k)

    relevant = set(relevant_documents)

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for doc_id in retrieved
        if doc_id in relevant
    )

    return relevant_count / k


def reciprocal_rank(results, relevant_documents):
    relevant = set(relevant_documents)

    for rank, result in enumerate(results, start=1):
        if result["document_id"] in relevant:
            return 1.0 / rank

    return 0.0


def main():

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    queries = load_queries()

    model = EmbeddingModel()

    store = FAISSVectorStore.load(
        INDEX_PATH,
        METADATA_PATH,
    )

    recall_1 = []
    recall_3 = []
    recall_5 = []

    precision_5 = []
    reciprocal_ranks = []

    for item in queries:

        query = item["query"]

        relevant_documents = item[
            "relevant_document_ids"
        ]

        query_embedding = model.encode_query(query)

        results = store.search(
            query_embedding,
            top_k=5,
        )

        r1 = recall_at_k(
            results,
            relevant_documents,
            1,
        )

        r3 = recall_at_k(
            results,
            relevant_documents,
            3,
        )

        r5 = recall_at_k(
            results,
            relevant_documents,
            5,
        )

        p5 = precision_at_k(
            results,
            relevant_documents,
            5,
        )

        rr = reciprocal_rank(
            results,
            relevant_documents,
        )

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)

        precision_5.append(p5)
        reciprocal_ranks.append(rr)

        print("\nQuery:")
        print(query)

        print(f"Recall@1:    {r1:.4f}")
        print(f"Recall@3:    {r3:.4f}")
        print(f"Recall@5:    {r5:.4f}")
        print(f"Precision@5: {p5:.4f}")
        print(f"RR:          {rr:.4f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Recall@1:    "
        f"{sum(recall_1) / len(recall_1):.4f}"
    )

    print(
        f"Recall@3:    "
        f"{sum(recall_3) / len(recall_3):.4f}"
    )

    print(
        f"Recall@5:    "
        f"{sum(recall_5) / len(recall_5):.4f}"
    )

    print(
        f"Precision@5: "
        f"{sum(precision_5) / len(precision_5):.4f}"
    )

    print(
        f"MRR:         "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()