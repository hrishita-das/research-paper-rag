import re


class CitationValidator:

    @staticmethod
    def has_valid_citation(answer, results):
        """
        Validate that the generated answer contains at least one
        citation referring to an actual retrieved source.
        """

        citations = re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )

        # No citation
        if not citations:
            return False

        # Valid source numbers
        valid_numbers = {
            str(i)
            for i in range(1, len(results) + 1)
        }

        # Check that every citation refers to a retrieved source
        for citation in citations:
            if citation not in valid_numbers:
                return False

        # Reject source metadata dumping
        metadata_patterns = [
            r"Document\s*:",
            r"Page\s*:",
            r"Content\s*:",
            r"Chunk\s*:",
        ]

        for pattern in metadata_patterns:
            if re.search(
                pattern,
                answer,
                flags=re.IGNORECASE,
            ):
                return False

        # Citation cannot be the first thing in the answer
        stripped = answer.strip()

        if re.match(
            r"^\[Source\s+\d+\]",
            stripped,
            flags=re.IGNORECASE,
        ):
            return False

        return True