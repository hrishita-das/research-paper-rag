import json
from pathlib import Path

from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever


def load_chunks():
    chunks = []

    for path in Path("data/processed").glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            chunks.extend(data)
        else:
            chunks.extend(data["chunks"])

    return chunks


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    embedding_model = EmbeddingModel()

    vector_store = FAISSVectorStore.load(
        "data/vector_store/index.faiss",
        "data/vector_store/metadata.json",
    )

    bm25 = BM25Retriever(chunks)

    retriever = HybridRetriever(
        vector_store,
        embedding_model,
        bm25,
    )

    query = input("\nQuery: ").strip()

    results = retriever.search(query, top_k=5)

    print("\n" + "=" * 70)

    for rank, result in enumerate(results, start=1):
        print(
            f"\n[{rank}] Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )
        print(f"Document: {result['document_id']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_id']}")
        print()
        print(result["text"][:1000])
        print("-" * 70)


if __name__ == "__main__":
    main()
