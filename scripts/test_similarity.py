import numpy as np

from app.retrieval.embedder import EmbeddingModel


def main():
    model = EmbeddingModel()

    documents = [
        "Self-attention allows each token to interact with other tokens.",
        "The Transformer architecture relies heavily on self-attention.",
        "Diffusion models generate images through iterative denoising.",
        "MRI scanners acquire medical images using magnetic fields.",
    ]

    query = "How does self-attention work in Transformers?"

    document_embeddings = model.encode_documents(documents)
    query_embedding = model.encode_query(query)

    scores = document_embeddings @ query_embedding.T
    scores = scores[:, 0]

    ranked_indices = np.argsort(scores)[::-1]

    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY TEST")
    print("=" * 70)

    print(f"\nQuery: {query}\n")

    for rank, idx in enumerate(ranked_indices, start=1):
        print(f"{rank}. Score: {scores[idx]:.4f}")
        print(f"   {documents[idx]}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()