import json
from pathlib import Path

from app.generation.rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"


def load_dataset():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def main():

    dataset = load_dataset()

    pipeline = RAGPipeline()

    results = []

    for item in dataset:

        question = item["question"]

        print("=" * 70)
        print(f"Question {item['id']}: {question}")
        print("=" * 70)

        answer, retrieved, cited = pipeline.answer(question)

        result = {
            "id": item["id"],
            "question": question,
            "category": item["category"],
            "expected_answer": item["expected_answer"],
            "answerable": item["answerable"],
            "generated_answer": answer,
            "retrieved_sources": retrieved,
            "cited_sources": cited,
        }

        results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()