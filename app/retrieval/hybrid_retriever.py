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
        top_k=5,
        candidate_k=20,
    ):
        # Dense retrieval
        query_embedding = self.embedding_model.encode_query(query)

        dense_results = self.vector_store.search(
            query_embedding,
            top_k=candidate_k,
        )

        # Sparse retrieval
        sparse_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k,
        )

        # Reciprocal Rank Fusion
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

        ranked_chunk_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        results = []

        for chunk_id in ranked_chunk_ids[:top_k]:
            result = dict(result_lookup[chunk_id])
            result["hybrid_score"] = fused_scores[chunk_id]
            results.append(result)

        return results
