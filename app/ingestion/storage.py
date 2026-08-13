import json
from pathlib import Path
from typing import List, Dict


def save_chunks(
    chunks: List[Dict],
    output_path: str,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_chunks(
    input_path: str,
) -> List[Dict]:

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)