import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval configuration
DENSE_TOP_K = 20
BM25_TOP_K = 20

# Number of candidates passed from RRF to reranker
RRF_TOP_K = 20

# Final number of chunks sent toward generation
RERANK_TOP_K = 5

# RRF constant
RRF_K = 60


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():

    print("Loading FAISS index...")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    print(f"FAISS vectors: {index.ntotal}")
    print(f"Metadata entries: {len(metadata)}")

    return index, metadata


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(metadata):

    print("Building BM25 index...")

    documents = []

    for item in metadata:

        text = item.get("text", "")

        documents.append(text)

    tokenized_documents = [
        text.lower().split()
        for text in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
    query,
    model,
    index,
    metadata,
    top_k=DENSE_TOP_K
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        result = dict(metadata[idx])

        result["dense_score"] = float(score)
        result["dense_rank"] = rank

        results.append(result)

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    bm25,
    metadata,
    top_k=BM25_TOP_K
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    rank = 1

    for idx in ranked_indices:

        if idx < 0 or idx >= len(metadata):
            continue

        result = dict(metadata[idx])

        result["bm25_score"] = float(
            scores[idx]
        )

        result["bm25_rank"] = rank

        results.append(result)

        rank += 1

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=RRF_K,
    top_k=RRF_TOP_K
):
    """
    Combine dense and BM25 rankings using
    Reciprocal Rank Fusion.

    RRF(d) =
        1 / (k + rank_dense)
        +
        1 / (k + rank_bm25)
    """

    fused = {}

    # --------------------------------------------------------
    # Dense ranking
    # --------------------------------------------------------

    for result in dense_results:

        chunk_id = result.get("chunk_id")

        if chunk_id is None:

            chunk_id = (
                result.get("document_id", "")
                + "_"
                + str(result.get("page", ""))
                + "_"
                + str(result.get("chunk_index", ""))
            )

        if chunk_id not in fused:

            fused[chunk_id] = {
                "result": result,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            }

        rank = result["dense_rank"]

        fused[chunk_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

        fused[chunk_id]["dense_rank"] = rank

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for result in bm25_results:

        chunk_id = result.get("chunk_id")

        if chunk_id is None:

            chunk_id = (
                result.get("document_id", "")
                + "_"
                + str(result.get("page", ""))
                + "_"
                + str(result.get("chunk_index", ""))
            )

        if chunk_id not in fused:

            fused[chunk_id] = {
                "result": result,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            }

        rank = result["bm25_rank"]

        fused[chunk_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

        fused[chunk_id]["bm25_rank"] = rank

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    final_results = []

    for rank, item in enumerate(
        ranked[:top_k],
        start=1
    ):

        result = dict(
            item["result"]
        )

        result["rrf_score"] = (
            item["rrf_score"]
        )

        result["rrf_rank"] = rank

        result["dense_rank"] = (
            item["dense_rank"]
        )

        result["bm25_rank"] = (
            item["bm25_rank"]
        )

        final_results.append(result)

    return final_results


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(
    query,
    results,
    reranker,
    top_k=RERANK_TOP_K
):
    """
    Rerank RRF candidates using a cross-encoder.

    The cross-encoder receives:

        [query, document_chunk]

    and predicts a relevance score.
    """

    if not results:
        return []

    pairs = []

    for result in results:

        text = result.get(
            "text",
            ""
        )

        pairs.append(
            [query, text]
        )

    print()
    print("Running cross-encoder reranking...")

    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    reranked = []

    for result, score in zip(
        results,
        scores
    ):

        item = dict(result)

        item["reranker_score"] = float(
            score
        )

        reranked.append(item)

    # Highest relevance first
    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    final_results = []

    for rank, result in enumerate(
        reranked[:top_k],
        start=1
    ):

        result["rerank"] = rank

        final_results.append(
            result
        )

    return final_results


# ============================================================
# PRINT HYBRID RESULTS
# ============================================================

def print_hybrid_results(
    query,
    results
):

    print()
    print("=" * 70)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 70)

    print()
    print("Query:")
    print(query)

    print()
    print("-" * 70)

    for result in results:

        print(
            f"\n[{result['rrf_rank']}] "
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Document: "
            f"{result.get('document_id', 'N/A')}"
        )

        print(
            f"Page: "
            f"{result.get('page', 'N/A')}"
        )

        print(
            f"Chunk: "
            f"{result.get('chunk_id', 'N/A')}"
        )

        print(
            f"Dense Rank: "
            f"{result.get('dense_rank', 'N/A')}"
        )

        print(
            f"BM25 Rank: "
            f"{result.get('bm25_rank', 'N/A')}"
        )

        text = result.get(
            "text",
            ""
        )

        if len(text) > 500:

            text = (
                text[:500]
                + "..."
            )

        print()
        print(text)

        print("-" * 70)


# ============================================================
# PRINT RERANKED RESULTS
# ============================================================

def print_reranked_results(
    query,
    results
):

    print()
    print("=" * 70)
    print("CROSS-ENCODER RERANKED RESULTS")
    print("=" * 70)

    print()
    print("Query:")
    print(query)

    print()
    print("-" * 70)

    for result in results:

        print(
            f"\n[{result['rerank']}] "
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Original RRF Rank: "
            f"{result.get('rrf_rank', 'N/A')}"
        )

        print(
            f"Document: "
            f"{result.get('document_id', 'N/A')}"
        )

        print(
            f"Page: "
            f"{result.get('page', 'N/A')}"
        )

        print(
            f"Chunk: "
            f"{result.get('chunk_id', 'N/A')}"
        )

        print(
            f"Dense Rank: "
            f"{result.get('dense_rank', 'N/A')}"
        )

        print(
            f"BM25 Rank: "
            f"{result.get('bm25_rank', 'N/A')}"
        )

        print(
            f"RRF Score: "
            f"{result.get('rrf_score', 0):.6f}"
        )

        text = result.get(
            "text",
            ""
        )

        if len(text) > 700:

            text = (
                text[:700]
                + "..."
            )

        print()
        print(text)

        print("-" * 70)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    model,
    index,
    metadata,
    bm25,
    dense_top_k=DENSE_TOP_K,
    bm25_top_k=BM25_TOP_K,
    final_top_k=RRF_TOP_K
):

    dense_results = dense_search(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
        top_k=dense_top_k
    )

    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        metadata=metadata,
        top_k=bm25_top_k
    )

    results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=RRF_K,
        top_k=final_top_k
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PRODUCTION RAG RETRIEVAL PIPELINE"
    )
    print(
        "FAISS + BM25 + RRF + CROSS-ENCODER"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load FAISS + metadata
    # --------------------------------------------------------

    index, metadata = load_vector_store()

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25 = build_bm25(
        metadata
    )

    print(
        "BM25 index ready."
    )

    # --------------------------------------------------------
    # Load reranker
    # --------------------------------------------------------

    print()
    print(
        "Loading cross-encoder reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512
    )

    print(
        "Cross-encoder loaded."
    )

    # --------------------------------------------------------
    # Query loop
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Enter a question.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:

        try:

            query = input(
                "\nQuery: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nExiting..."
            )

            break

        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "Exiting..."
            )

            break

        if not query:

            continue

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        results = hybrid_search(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            bm25=bm25,
            dense_top_k=DENSE_TOP_K,
            bm25_top_k=BM25_TOP_K,
            final_top_k=RRF_TOP_K
        )

        # ----------------------------------------------------
        # Print RRF results
        # ----------------------------------------------------

        print_hybrid_results(
            query,
            results
        )

        # ----------------------------------------------------
        # Cross-encoder reranking
        # ----------------------------------------------------

        reranked_results = rerank_results(
            query=query,
            results=results,
            reranker=reranker,
            top_k=RERANK_TOP_K
        )

        # ----------------------------------------------------
        # Print final results
        # ----------------------------------------------------

        print_reranked_results(
            query,
            reranked_results
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()