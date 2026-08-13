from pathlib import Path

from app.ingestion.pdf_parser import extract_text_from_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    print("=" * 80)
    print("CHECKING FOR EMPTY PDF PAGES")
    print("=" * 80)

    total_empty_pages = 0

    for pdf_path in pdf_files:

        pages = extract_text_from_pdf(
            str(pdf_path)
        )

        empty_pages = [
            page["page"]
            for page in pages
            if not page["text"].strip()
        ]

        if empty_pages:

            print(
                f"\n{pdf_path.name}"
            )

            print(
                f"Empty pages: {empty_pages}"
            )

            total_empty_pages += len(
                empty_pages
            )

        else:

            print(
                f"{pdf_path.name}: OK"
            )

    print("\n" + "=" * 80)

    print(
        f"Total empty pages: "
        f"{total_empty_pages}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()