from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    Wrapper around the BGE embedding model.

    Converts text into normalized dense vectors that can
    be searched using cosine similarity / inner product.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device

        print(f"Loading embedding model: {model_name}")
        print(f"Device: {device}")

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        print(
            f"Embedding dimension: "
            f"{self.model.get_sentence_embedding_dimension()}"
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode document chunks.

        Returns:
            numpy array with shape:
            (number_of_chunks, embedding_dimension)
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single user query.

        Returns:
            numpy array with shape:
            (1, embedding_dimension)
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype("float32").reshape(1, -1)