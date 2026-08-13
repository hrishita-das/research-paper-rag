from pathlib import Path

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.metadata import generate_document_id
from app.ingestion.chunker import chunk_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {RAW_DATA_DIR}")
        return

    print("=" * 80)
    print("DOCUMENT INGESTION + CHUNKING TEST")
    print("=" * 80)

    total_pages = 0
    total_chunks = 0

    for pdf_path in pdf_files:

        print("\n" + "=" * 80)
        print(f"FILE: {pdf_path.name}")
        print("=" * 80)

        try:

            # --------------------------------------------------
            # 1. Generate document ID
            # --------------------------------------------------

            document_id = generate_document_id(
                str(pdf_path)
            )

            print(f"Document ID: {document_id}")

            # --------------------------------------------------
            # 2. Extract PDF text
            # --------------------------------------------------

            pages = extract_text_from_pdf(
                str(pdf_path)
            )

            print(f"Pages extracted: {len(pages)}")

            # --------------------------------------------------
            # 3. Create chunks
            # --------------------------------------------------

            chunks = chunk_pages(
                pages=pages,
                document_id=document_id,
            )

            print(f"Chunks created: {len(chunks)}")

            total_pages += len(pages)
            total_chunks += len(chunks)

            # --------------------------------------------------
            # 4. Show first chunk
            # --------------------------------------------------

            if chunks:

                first_chunk = chunks[0]

                print("\n--- FIRST CHUNK ---")

                print(
                    f"Chunk ID      : "
                    f"{first_chunk['chunk_id']}"
                )

                print(
                    f"Document ID   : "
                    f"{first_chunk['document_id']}"
                )

                print(
                    f"Page          : "
                    f"{first_chunk['page']}"
                )

                print(
                    f"Chunk index   : "
                    f"{first_chunk['chunk_index']}"
                )

                print("\nText preview:")
                print(first_chunk["text"][:1000])

            else:

                print(
                    "[WARNING] No chunks generated."
                )

        except Exception as e:

            print(
                f"[ERROR] Failed to process "
                f"{pdf_path.name}"
            )

            print(f"Reason: {e}")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    print("\n" + "=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)

    print(
        f"PDFs processed : {len(pdf_files)}"
    )

    print(
        f"Total pages    : {total_pages}"
    )

    print(
        f"Total chunks   : {total_chunks}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()