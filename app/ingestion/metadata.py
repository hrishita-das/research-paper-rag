from pathlib import Path
import hashlib


def generate_document_id(pdf_path: str) -> str:
    """
    Generate a deterministic document ID from the PDF filename.
    """

    filename = Path(pdf_path).stem

    normalized = filename.lower().replace(" ", "_")

    document_id = hashlib.md5(
        normalized.encode("utf-8")
    ).hexdigest()[:12]

    return document_id