from pathlib import Path
from typing import List, Dict

import fitz


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text page-by-page from a PDF.

    Returns:
        List of dictionaries:
        {
            "page": page_number,
            "text": extracted_text
        }
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages