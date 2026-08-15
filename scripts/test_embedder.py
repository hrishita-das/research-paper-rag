from app.retrieval.embedder import EmbeddingModel


def main():
    model = EmbeddingModel()

    texts = [
        "The Transformer architecture uses self-attention.",
        "Diffusion models generate samples through an iterative denoising process.",
        "Convolutional neural networks use convolution operations for feature extraction.",
    ]

    embeddings = model.encode_documents(texts)

    print("\n" + "=" * 70)
    print("EMBEDDING TEST")
    print("=" * 70)

    print("Number of texts :", len(texts))
    print("Embedding shape :", embeddings.shape)
    print("Embedding dtype :", embeddings.dtype)

    query = "How does self-attention work?"

    query_embedding = model.encode_query(query)

    print("Query shape     :", query_embedding.shape)

    print("=" * 70)


if __name__ == "__main__":
    main()