from typing import List, Dict, Optional


class ConversationHistory:
    """
    Stores conversation history for a RAG session.

    Each conversation turn contains:
        - user query
        - generated answer

    Retrieval results and source metadata are intentionally NOT stored
    here because they can become large and are not required for
    reconstructing conversational context.
    """

    def __init__(self, max_turns: int = 5):
        """
        Args:
            max_turns:
                Maximum number of previous conversation turns to retain.
        """

        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    # ==========================================================
    # ADD TURN
    # ==========================================================

    def add_turn(
        self,
        user_query: str,
        assistant_answer: str,
    ):
        """
        Add one completed conversation turn.
        """

        self.history.append(
            {
                "user": user_query,
                "assistant": assistant_answer,
            }
        )

        # Keep only the most recent turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    # ==========================================================
    # GET HISTORY
    # ==========================================================

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return conversation history as structured data.
        """

        return list(self.history)

    # ==========================================================
    # GET RECENT HISTORY
    # ==========================================================

    def get_recent_history(
        self,
        turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Return the most recent N conversation turns.

        If turns is None, return the complete stored history.
        """

        if turns is None:
            turns = self.max_turns

        return list(self.history[-turns:])

    # ==========================================================
    # FORMAT HISTORY
    # ==========================================================

    def format_history(
        self,
        turns: Optional[int] = None,
    ) -> str:
        """
        Convert conversation history into text suitable for
        inclusion in an LLM prompt.
        """

        history = self.get_recent_history(turns)

        if not history:
            return "No previous conversation."

        formatted = []

        for i, turn in enumerate(history, start=1):

            formatted.append(
                f"""Turn {i}

User:
{turn["user"]}

Assistant:
{turn["assistant"]}
"""
            )

        return "\n".join(formatted)

    # ==========================================================
    # LAST USER QUERY
    # ==========================================================

    def get_last_user_query(self) -> Optional[str]:
        """
        Return the previous user query.
        """

        if not self.history:
            return None

        return self.history[-1]["user"]

    # ==========================================================
    # LAST ASSISTANT ANSWER
    # ==========================================================

    def get_last_answer(self) -> Optional[str]:
        """
        Return the previous assistant answer.
        """

        if not self.history:
            return None

        return self.history[-1]["assistant"]

    # ==========================================================
    # EMPTY CHECK
    # ==========================================================

    def is_empty(self) -> bool:
        """
        Check whether conversation history is empty.
        """

        return len(self.history) == 0

    # ==========================================================
    # CLEAR
    # ==========================================================

    def clear(self):
        """
        Clear the complete conversation history.
        """

        self.history.clear()

    # ==========================================================
    # LENGTH
    # ==========================================================

    def __len__(self):
        """
        Number of stored conversation turns.
        """

        return len(self.history)

    # ==========================================================
    # DEBUG
    # ==========================================================

    def print_history(self):
        """
        Print conversation history for debugging.
        """

        print("\n" + "=" * 70)
        print("CONVERSATION HISTORY")
        print("=" * 70)

        if not self.history:
            print("No conversation history.")
            return

        for i, turn in enumerate(self.history, start=1):

            print(f"\n--- TURN {i} ---")

            print("\nUser:")
            print(turn["user"])

            print("\nAssistant:")
            print(turn["assistant"])

        print("=" * 70)