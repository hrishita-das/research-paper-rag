from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    document_id: str,
    chunk_size: int = 2000,
    overlap: int = 300,
) -> List[Dict]:
    """
    Split extracted PDF pages into overlapping chunks.

    Args:
        pages:
            List of page dictionaries returned by the PDF parser.

        document_id:
            Unique identifier for the source document.

        chunk_size:
            Approximate chunk size in characters.

        overlap:
            Number of overlapping characters between consecutive chunks.

    Returns:
        List of chunks containing text and metadata.
    """

    chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunk = {
                    "chunk_id": (
                        f"{document_id}_p{page_number}_c{chunk_index}"
                    ),
                    "document_id": document_id,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }

                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - overlap
            chunk_index += 1

    return chunks