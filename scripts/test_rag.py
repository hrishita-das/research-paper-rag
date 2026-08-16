from app.generation.rag_pipeline import RAGPipeline


def main():

    print("=" * 70)
    print("RESEARCH PAPER RAG")
    print("=" * 70)

    rag = RAGPipeline()

    while True:

        try:
            query = input("\nQuery: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting...")
            break

        if not query:
            continue

        print()
        print("=" * 70)
        print("RUNNING RAG PIPELINE")
        print("=" * 70)

        try:

            answer, results, cited_sources = rag.answer(
                query=query,
                candidate_k=20,
                rerank_k=5,
            )

        except Exception as e:

            print()
            print("ERROR:")
            print(e)
            continue

        print()
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print()

        print(answer)

        print()
        print("=" * 70)
        print("CITED SOURCES")
        print("=" * 70)

        if not cited_sources:
            print("No sources were cited by the generated answer.")
        else:
            for source in cited_sources:

                print(
                    f"[Source {source['source_number']}] "
                    f"Document: {source.get('document_id', 'unknown')}, "
                    f"Page: {source.get('page', 'unknown')}"
                )

                print(
                    f"Chunk: {source.get('chunk_id', 'unknown')}"
                )

                print()

        print("=" * 70)


if __name__ == "__main__":
    main()