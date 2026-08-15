import numpy as np

from app.retrieval.vector_store import FAISSVectorStore


def main():

    print("=" * 70)
    print("FAISS VECTOR STORE TEST")
    print("=" * 70)

    # Three-dimensional toy embeddings.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    # Normalize embeddings.
    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    metadata = [
        {
            "chunk_id": "chunk_0",
            "document_id": "attention.pdf",
            "page": 1,
            "text": "Self-attention allows tokens to interact.",
        },
        {
            "chunk_id": "chunk_1",
            "document_id": "transformer.pdf",
            "page": 2,
            "text": "Transformers rely on attention mechanisms.",
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "ddpm.pdf",
            "page": 3,
            "text": "Diffusion models iteratively denoise samples.",
        },
    ]

    store = FAISSVectorStore(
        dimension=3
    )

    store.add(
        embeddings,
        metadata,
    )

    print(f"\nVectors indexed: {store.index.ntotal}")

    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype="float32",
    )

    results = store.search(
        query,
        top_k=3,
    )

    print("\nSearch results:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{rank}. "
            f"Score: {result['score']:.4f}"
        )

        print(
            f"   Document: "
            f"{result['document_id']}"
        )

        print(
            f"   Page: "
            f"{result['page']}"
        )

        print(
            f"   Text: "
            f"{result['text']}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()