class HybridRetriever:
    """
    Combines dense FAISS retrieval and sparse BM25 retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        bm25_retriever,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query,
        top_k=20,
        candidate_k=20,
    ):
        """
        Retrieve candidate chunks using:
        1. Dense FAISS retrieval
        2. Sparse BM25 retrieval
        3. Reciprocal Rank Fusion

        top_k:
            Number of final hybrid candidates returned.

        candidate_k:
            Number of candidates retrieved from each retriever.
        """

        # ----------------------------------------------------
        # Dense retrieval
        # ----------------------------------------------------

        query_embedding = self.embedding_model.encode_query(
            query
        )

        dense_results = self.vector_store.search(
            query_embedding,
            top_k=candidate_k,
        )

        # ----------------------------------------------------
        # Sparse BM25 retrieval
        # ----------------------------------------------------

        sparse_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k,
        )

        # ----------------------------------------------------
        # Reciprocal Rank Fusion
        # ----------------------------------------------------

        fused_scores = {}
        result_lookup = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (60 + rank)
            )

            result_lookup[chunk_id] = result

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (60 + rank)
            )

            if chunk_id not in result_lookup:
                result_lookup[chunk_id] = result

        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        ranked_chunk_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        results = []

        for rank, chunk_id in enumerate(
            ranked_chunk_ids[:top_k],
            start=1,
        ):
            result = dict(result_lookup[chunk_id])

            result["hybrid_score"] = fused_scores[chunk_id]
            result["hybrid_rank"] = rank

            results.append(result)

        return results