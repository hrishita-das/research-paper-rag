from pathlib import Path

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.metadata import generate_document_id
from app.ingestion.chunker import chunk_pages
from app.ingestion.storage import save_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


def process_pdf(pdf_path: Path):

    print(f"\nProcessing: {pdf_path.name}")

    # Generate document ID
    document_id = generate_document_id(
        str(pdf_path)
    )

    # Extract pages
    pages = extract_text_from_pdf(
        str(pdf_path)
    )

    # Create chunks
    chunks = chunk_pages(
        pages=pages,
        document_id=document_id,
    )

    # Output file
    output_path = (
        PROCESSED_DATA_DIR
        / f"{document_id}.json"
    )

    # Save
    save_chunks(
        chunks=chunks,
        output_path=str(output_path),
    )

    print(
        f"Pages : {len(pages)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Saved : {output_path}"
    )


def main():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        RAW_DATA_DIR.glob("*.pdf")
    )

    print("=" * 80)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 80)

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    for pdf_path in pdf_files:

        process_pdf(pdf_path)

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()