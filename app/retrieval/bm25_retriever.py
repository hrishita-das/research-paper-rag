import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based sparse retriever using BM25.
    """

    def __init__(self, chunks):
        self.chunks = chunks

        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def tokenize(text):
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def search(self, query, top_k=20):
        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1]

        results = []

        for idx in ranked_indices[:top_k]:
            result = dict(self.chunks[idx])
            result["bm25_score"] = float(scores[idx])
            results.append(result)

        return results
