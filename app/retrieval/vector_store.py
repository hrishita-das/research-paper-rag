from pathlib import Path
import json

import faiss
import numpy as np


class FAISSVectorStore:
    """
    FAISS-based vector store using inner product similarity.

    Embeddings must be L2-normalized before insertion.
    For normalized vectors, inner product is equivalent to
    cosine similarity.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Add embeddings and their metadata to the FAISS index.
        """

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match number of metadata entries."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got shape {embeddings.shape}"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search the FAISS index and return top-k results.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            result = dict(self.metadata[index])
            result["score"] = float(score)
            result["faiss_index"] = int(index)

            results.append(result)

        return results

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """
        Persist the FAISS index and metadata.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"FAISS index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ):
        """
        Load a previously saved FAISS index and metadata.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        store = cls(index.d)

        store.index = index
        store.metadata = metadata

        return store