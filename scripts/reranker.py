from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self, model_name=MODEL_NAME):

        print("Loading cross-encoder reranker...")

        self.model = CrossEncoder(
            model_name,
            max_length=512
        )

        print("Reranker loaded.")

    def rerank(
        self,
        query,
        results,
        top_k=5
    ):
        """
        Rerank retrieved chunks using a cross-encoder.
        """

        if not results:
            return []

        pairs = []

        for result in results:

            text = result.get("text", "")

            pairs.append(
                [query, text]
            )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores
        ):

            item = dict(result)

            item["reranker_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        for rank, result in enumerate(
            reranked[:top_k],
            start=1
        ):

            result["rerank"] = rank

        return reranked[:top_k]


def print_reranked_results(
    query,
    results
):

    print()
    print("=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    print()
    print("Query:")
    print(query)

    print()

    for result in results:

        print(
            f"[{result['rerank']}] "
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
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

        print()

        text = result.get(
            "text",
            ""
        )

        if len(text) > 700:
            text = text[:700] + "..."

        print(text)

        print("-" * 70)