import json
from pathlib import Path

from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore


PROCESSED_DIR = Path("data/processed")

INDEX_PATH = Path(
    "data/vector_store/index.faiss"
)

METADATA_PATH = Path(
    "data/vector_store/metadata.json"
)


def load_chunks():
    """
    Load all chunks from processed JSON files.
    """

    json_files = sorted(
        PROCESSED_DIR.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {PROCESSED_DIR}"
        )

    all_chunks = []

    print(
        f"Found {len(json_files)} processed documents."
    )

    for json_file in json_files:

        print(
            f"Loading: {json_file.name}"
        )

        with open(
            json_file,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        # Your Phase 1 output is expected to
        # contain a list of chunk dictionaries.
        if isinstance(data, list):
            chunks = data

        elif isinstance(data, dict):
            if "chunks" in data:
                chunks = data["chunks"]
            else:
                raise ValueError(
                    f"Could not find chunks in {json_file}"
                )

        else:
            raise ValueError(
                f"Unexpected JSON structure in {json_file}"
            )

        all_chunks.extend(chunks)

    return all_chunks


def main():

    print("=" * 70)
    print("BUILDING FAISS VECTOR INDEX")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load chunks
    # --------------------------------------------------

    chunks = load_chunks()

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    # --------------------------------------------------
    # 2. Extract text
    # --------------------------------------------------

    texts = []

    valid_chunks = []

    for chunk in chunks:

        text = chunk.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        texts.append(text)
        valid_chunks.append(chunk)

    print(
        f"Chunks with text: "
        f"{len(valid_chunks)}"
    )

    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    model = EmbeddingModel()

    # --------------------------------------------------
    # 4. Generate embeddings
    # --------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode_documents(
        texts,
        batch_size=32,
    )

    print(
        f"\nEmbedding shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------
    # 5. Build FAISS index
    # --------------------------------------------------

    dimension = embeddings.shape[1]

    store = FAISSVectorStore(
        dimension=dimension
    )

    store.add(
        embeddings,
        valid_chunks,
    )

    print(
        f"\nVectors indexed: "
        f"{store.index.ntotal}"
    )

    # --------------------------------------------------
    # 6. Save
    # --------------------------------------------------

    store.save(
        INDEX_PATH,
        METADATA_PATH,
    )

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()