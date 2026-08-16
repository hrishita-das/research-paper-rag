import re

from pathlib import Path

from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from scripts.reranker import Reranker

from app.generation.prompt import (
    SYSTEM_PROMPT,
    build_prompt,
)

from app.generation.llm import LocalLLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_INDEX = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "index.faiss"
)

METADATA = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "metadata.json"
)

# Retrieval relevance threshold.
# If the best reranker score falls below this, the retrieved
# evidence is treated as insufficient and the LLM is never called.
RETRIEVAL_SCORE_THRESHOLD = -2.0


class RAGPipeline:

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 70)

        # --------------------------------------------------
        # Embedding model
        # --------------------------------------------------

        self.embedding_model = EmbeddingModel()

        # --------------------------------------------------
        # FAISS vector store
        # --------------------------------------------------

        print("Loading FAISS vector store...")

        self.vector_store = FAISSVectorStore.load(
            VECTOR_INDEX,
            METADATA,
        )

        # --------------------------------------------------
        # BM25 retriever
        # --------------------------------------------------

        print("Building BM25 retriever...")

        self.bm25_retriever = BM25Retriever(
            self.vector_store.metadata
        )

        # --------------------------------------------------
        # Hybrid retriever
        # --------------------------------------------------

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            bm25_retriever=self.bm25_retriever,
        )

        # --------------------------------------------------
        # Cross-encoder reranker
        # --------------------------------------------------

        self.reranker = Reranker()

        # --------------------------------------------------
        # Local LLM
        # --------------------------------------------------

        self.llm = LocalLLM()

        print("=" * 70)
        print("RAG PIPELINE READY")
        print("=" * 70)

    # ======================================================
    # RETRIEVAL
    # ======================================================

    def retrieve(
        self,
        query,
        candidate_k=20,
        rerank_k=5,
    ):
        """
        Retrieve relevant chunks using:

        Dense FAISS
             +
        Sparse BM25
             ↓
        Reciprocal Rank Fusion
             ↓
        Cross-encoder reranking
             ↓
        Top-k evidence
        """

        # --------------------------------------------------
        # Hybrid retrieval
        # --------------------------------------------------

        hybrid_results = self.hybrid_retriever.search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        # --------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=rerank_k,
        )

        return reranked_results

    # ======================================================
    # CITATION EXTRACTION
    # ======================================================

    def extract_cited_sources(self, answer, results):
        """
        Extract [Source N] citations from the generated answer
        and map them to the corresponding retrieved sources.
        """

        # Find citations such as:
        # [Source 1]
        # [Source 2]
        # [Source 5]
        cited_numbers = re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )

        # Remove duplicates while preserving order
        cited_numbers = list(
            dict.fromkeys(
                int(number)
                for number in cited_numbers
            )
        )

        cited_sources = []

        for number in cited_numbers:

            # Source numbering is 1-based
            index = number - 1

            if 0 <= index < len(results):
                source = dict(results[index])
                source["source_number"] = number
                cited_sources.append(source)

        return cited_sources

    def has_valid_citation(self, answer, results):
        """
        Validate that the generated answer contains at least one
        citation referring to an actual retrieved source.
        """

        # Accept:
        # [Source 1]
        # [source 1]
        # [Source 2]
        citations = re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )

        if not citations:
            return False

        # Valid source numbers
        valid_numbers = {
            str(i)
            for i in range(1, len(results) + 1)
        }

        # Every citation must refer to an existing source
        for citation in citations:

            if citation not in valid_numbers:
                return False

        # Reject source metadata dumping
        metadata_patterns = [
            r"Document\s*:",
            r"Page\s*:",
            r"Content\s*:",
            r"Chunk\s*:",
        ]

        for pattern in metadata_patterns:

            if re.search(
                pattern,
                answer,
                flags=re.IGNORECASE,
            ):
                return False

        # Citation cannot be the first thing in the answer
        stripped = answer.strip()

        if re.match(
            r"^\[Source\s+\d+\]",
            stripped,
            flags=re.IGNORECASE,
        ):
            return False

        return True

    def validate_citations(self, answer, results):
        """
        TEMPORARY DEBUG VALIDATOR.

        Same core checks as has_valid_citation(), but prints
        exactly what was detected and why validation failed.
        Use this in place of has_valid_citation() while debugging
        why the retry is failing.
        """

        citations = re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )

        print("\n========== CITATION DEBUG ==========")
        print("Answer:", answer)
        print("Detected citations:", citations)

        if not citations:
            print("FAIL: No citations found")
            return False

        valid_numbers = {
            str(i)
            for i in range(1, len(results) + 1)
        }

        print("Valid source numbers:", valid_numbers)

        for citation in citations:

            if citation not in valid_numbers:
                print("FAIL: Invalid citation:", citation)
                return False

        print("Citation validation passed")
        return True

    def answer_is_relevant(self, query, answer):
        """
        Lightweight relevance check.

        This is intentionally permissive because the LLM may
        paraphrase the retrieved content. Only requires that at
        least one meaningful query term appears in the answer.
        """

        stopwords = {
            "what", "are", "is", "the", "a", "an",
            "how", "does", "do", "did",
            "of", "to", "in", "on", "for", "from",
            "and", "or", "with", "by",
            "this", "that", "these", "those",
            "describe", "describes",
            "explain", "explain",
            "work", "works",
            "main", "primary",
            "difference", "differ",
        }

        query_words = [
            word.lower()
            for word in re.findall(r"\b[a-zA-Z0-9]+\b", query)
            if word.lower() not in stopwords
        ]

        answer_words = {
            word.lower()
            for word in re.findall(
                r"\b[a-zA-Z0-9]+\b",
                answer
            )
        }

        if not query_words:
            return True

        overlap = set(query_words) & answer_words

        # Only require ONE meaningful query term.
        return len(overlap) >= 1

    # ======================================================
    # ANSWER GENERATION
    # ======================================================

    def answer(
        self,
        query,
        candidate_k=20,
        rerank_k=5,
    ):
        """
        Complete RAG pipeline:

        Query
          ↓
        Hybrid retrieval
          ↓
        Cross-encoder reranking
          ↓
        Relevance check (retrieval)
          ↓
        Grounded prompt
          ↓
        Local Qwen LLM
          ↓
        Citation validation → retry → safe error
          ↓
        Cited sources
        """

        # --------------------------------------------------
        # Retrieve evidence
        # --------------------------------------------------

        results = self.retrieve(
            query=query,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )

        # --------------------------------------------------
        # TEMPORARY DEBUG: inspect retrieved results
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("RETRIEVED RESULTS")
        print("=" * 70)

        for i, result in enumerate(results, start=1):

            print(f"\n[Source {i}]")
            print("Document:", result.get("document_id"))
            print("Page:", result.get("page"))
            print(
                "Score:",
                result.get("reranker_score", result.get("score")),
            )
            print("Text:")
            print(result.get("text", "")[:1500])

        print("=" * 70)

        # --------------------------------------------------
        # Retrieval relevance check
        # --------------------------------------------------

        if not results:

            return (
                "The retrieved documents do not contain enough "
                "information to answer this question.",
                [],
                [],
            )

        # Highest reranker score
        best_score = results[0].get(
            "reranker_score",
            results[0].get("score", -999),
        )

        # If even the best retrieved chunk is not relevant,
        # do not allow the LLM to answer from its own knowledge.
        if best_score < RETRIEVAL_SCORE_THRESHOLD:

            return (
                "The retrieved documents do not contain enough "
                "information to answer this question.",
                results,
                [],
            )

        # --------------------------------------------------
        # Build grounded prompt
        # --------------------------------------------------

        prompt = build_prompt(
            query=query,
            results=results,
        )

        # --------------------------------------------------
        # Generate answer
        # --------------------------------------------------

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_new_tokens=350,
            temperature=0.0,
        )

        print("\n" + "=" * 70)
        print("RAW GENERATED ANSWER")
        print("=" * 70)
        print(repr(answer))
        print("=" * 70)

        # --------------------------------------------------
        # Validate citation
        # --------------------------------------------------

        if not self.validate_citations(answer, results):

            print("WARNING: Generated answer failed validation.")
            print("Retrying generation with citation enforcement...")

            # Build a CLEAN retry prompt.
            # Do NOT embed the previous build_prompt() inside it.

            evidence_parts = []

            for i, result in enumerate(results, start=1):

                text = result.get(
                    "text",
                    ""
                )

                evidence_parts.append(
                    f"""[Source {i}]
{text}
"""
                )

            evidence = "\n".join(evidence_parts)

            valid_source_list = "\n".join(
                f"   [Source {i}]"
                for i in range(1, len(results) + 1)
            )

            citation_prompt = f"""You are answering a research-paper question.

You MUST answer using ONLY the evidence below.

EVIDENCE
============================================================

{evidence}

============================================================
QUESTION
============================================================

{query}

============================================================
STRICT OUTPUT RULES
============================================================

1. Every factual sentence MUST end with [Source N].

2. N MUST be one of:
{valid_source_list}

3. Put the citation at the END of the sentence.

4. NEVER write a factual sentence without a citation.

5. NEVER write citations before a sentence.

6. NEVER create a Sources or References section.

7. NEVER output Document, Page, Content, or Chunk metadata.

8. Use ONLY information explicitly present in the evidence.

9. For equations, copy the equation exactly from the evidence
   as closely as possible. Do not simplify, modify, or derive it.

10. If the evidence contains the answer, answer the question.

11. If the evidence does not contain the answer, return exactly:

The retrieved documents do not contain enough information to answer this question.

============================================================
EXAMPLE
============================================================

Question:
What is DDPM?

Answer:
DDPM is a diffusion model that consists of forward and reverse
processes. [Source 1]

Question:
What equation describes the forward diffusion process?

Answer:
The forward diffusion process is described by
q(x_t|x_{{t-1}}) = N(x_t; sqrt(1-beta_t)x_{{t-1}}, beta_t I). [Source 1]

============================================================

Return ONLY the answer.
"""

            answer = self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=citation_prompt,
                max_new_tokens=300,
                temperature=0.0,
            )

            print("\n" + "=" * 70)
            print("RAW RETRY ANSWER")
            print("=" * 70)
            print(repr(answer))
            print("=" * 70)

        # --------------------------------------------------
        # Extract cited sources
        # --------------------------------------------------

        cited_sources = self.extract_cited_sources(
            answer=answer,
            results=results,
        )

        # --------------------------------------------------
        # Final validation
        # --------------------------------------------------

        if not self.validate_citations(answer, results):

            print("WARNING: Citation enforcement failed after retry.")

            # Do NOT attach an irrelevant citation.
            # Do NOT return an uncited hallucinated answer.

            answer = (
                "The retrieved documents do not contain enough "
                "information to generate a properly grounded answer."
            )

            cited_sources = []

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return answer, results, cited_sources