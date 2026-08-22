
import json
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# Project Root / Import Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import citation validator
from app.generation.validator import CitationValidator


# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results.json"
METRICS_FILE = PROJECT_ROOT / "evaluation" / "metrics.json"


# ============================================================
# Helper Functions
# ============================================================

def is_refusal(answer):
    """
    Detect retrieval-grounded refusal.

    Used primarily for out-of-domain questions.
    """

    if not answer:
        return False

    answer = answer.lower()

    refusal_phrases = [
        "do not contain enough information",
        "cannot answer",
        "insufficient information",
        "not enough information",
        "unable to answer",
        "don't contain enough information",
    ]

    return any(
        phrase in answer
        for phrase in refusal_phrases
    )


def get_answer(sample):
    """
    Get the generated answer from an evaluation sample.

    Supports the current results.json format where the
    generated answer is stored under 'generated_answer'.

    Also supports 'answer' as a fallback.
    """

    answer = sample.get("generated_answer")

    if answer is None:
        answer = sample.get("answer", "")

    return answer or ""


# ============================================================
# Main Evaluation
# ============================================================

def evaluate(results):

    total = len(results)

    # --------------------------------------------------------
    # Citation statistics
    # --------------------------------------------------------

    citation_valid = 0
    citation_total = 0

    # --------------------------------------------------------
    # Answerable statistics
    # --------------------------------------------------------

    answerable_total = 0
    answerable_correct = 0

    # --------------------------------------------------------
    # Out-of-domain statistics
    # --------------------------------------------------------

    ood_total = 0
    ood_correct = 0

    # --------------------------------------------------------
    # Category statistics
    # --------------------------------------------------------

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
        }
    )

    # ========================================================
    # Evaluate every sample
    # ========================================================

    for sample in results:

        sample_id = sample.get("id", "unknown")

        category = sample.get(
            "category",
            "unknown",
        )

        answerable = sample.get(
            "answerable",
            category != "out_of_domain",
        )

        answer = get_answer(sample)

        retrieved_sources = sample.get(
            "retrieved_sources",
            [],
        )

        # ----------------------------------------------------
        # Category count
        # ----------------------------------------------------

        category_stats[category]["total"] += 1

        # ----------------------------------------------------
        # Citation validity
        #
        # Only answerable / in-domain questions are expected
        # to contain citations.
        #
        # OOD refusals should NOT reduce citation validity.
        # ----------------------------------------------------

        if answerable:

            citation_total += 1

            citation_is_valid = CitationValidator.has_valid_citation(
                answer,
                retrieved_sources,
            )

            if citation_is_valid:
                citation_valid += 1

        # ----------------------------------------------------
        # Answerable questions
        # ----------------------------------------------------

        if answerable:

            answerable_total += 1

            # A non-refusal answer is considered successfully
            # answered for this basic generation metric.
            if not is_refusal(answer):

                answerable_correct += 1
                category_stats[category]["success"] += 1

        # ----------------------------------------------------
        # Out-of-domain questions
        # ----------------------------------------------------

        else:

            ood_total += 1

            # Correct OOD behavior = grounded refusal.
            if is_refusal(answer):

                ood_correct += 1
                category_stats[category]["success"] += 1

    # ========================================================
    # Calculate Metrics
    # ========================================================

    metrics = {

        "total_questions": total,

        # Citation validity is calculated only over
        # answerable/in-domain questions.
        "citation_validity": round(
            citation_valid / max(citation_total, 1),
            4,
        ),

        "answerable_success_rate": round(
            answerable_correct / max(answerable_total, 1),
            4,
        ),

        "ood_refusal_rate": round(
            ood_correct / max(ood_total, 1),
            4,
        ),

        # Additional useful counts
        "citation_valid_count": citation_valid,

        "citation_evaluated_count": citation_total,

        "answerable_correct_count": answerable_correct,

        "answerable_total": answerable_total,

        "ood_correct_count": ood_correct,

        "ood_total": ood_total,
    }

    # ========================================================
    # Print Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("RAG EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Total Questions: {total}"
    )

    print("\nGENERATION")
    print("-" * 70)

    print(
        f"Citation Validity      : "
        f"{metrics['citation_validity']:.2%}"
        f" ({citation_valid}/{citation_total})"
    )

    print(
        f"Answerable Success     : "
        f"{metrics['answerable_success_rate']:.2%}"
        f" ({answerable_correct}/{answerable_total})"
    )

    print(
        f"Out-of-domain Refusal  : "
        f"{metrics['ood_refusal_rate']:.2%}"
        f" ({ood_correct}/{ood_total})"
    )

    # ========================================================
    # Category Results
    # ========================================================

    print("\nCATEGORY RESULTS")
    print("-" * 70)

    for category, stats in category_stats.items():

        accuracy = (
            stats["success"]
            / max(stats["total"], 1)
        )

        print(
            f"{category:<20}"
            f"{accuracy:.2%}"
            f" ({stats['success']}/{stats['total']})"
        )

    print("=" * 70)

    # ========================================================
    # Save Metrics
    # ========================================================

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(
        f"\nMetrics saved to: {METRICS_FILE}"
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check results file
    # --------------------------------------------------------

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation results not found:\n"
            f"{RESULTS_FILE}"
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    # --------------------------------------------------------
    # Validate results format
    # --------------------------------------------------------

    if not isinstance(results, list):

        raise ValueError(
            "evaluation/results.json must contain "
            "a JSON list of evaluation samples."
        )

    if len(results) == 0:

        raise ValueError(
            "evaluation/results.json contains no samples."
        )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    evaluate(results)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()

