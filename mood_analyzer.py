# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Matches a single emoji / pictographic character so we can treat each one
# as its own token (emojis are strong mood signals in the sample posts).
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0000FE00-\U0000FE0F\U0000200D]"
)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Improvements implemented here:
          - Lowercase and strip surrounding whitespace.
          - Split each emoji out as its own token (so "day 😂🔥" -> "day", "😂", "🔥"),
            since emojis are strong mood signals in the sample posts.
          - Strip leading/trailing punctuation off word tokens
            (so "fine." matches "fine" and "exam!!" matches "exam").
          - Drop tokens that become empty after cleaning.
        """
        cleaned = text.strip().lower()

        # Surround each emoji with spaces so it splits into its own token.
        cleaned = _EMOJI_RE.sub(lambda m: f" {m.group(0)} ", cleaned)

        tokens: List[str] = []
        for raw in cleaned.split():
            if _EMOJI_RE.match(raw):
                # Keep emoji tokens exactly as-is.
                tokens.append(raw)
                continue
            # Strip punctuation from the edges of word tokens, keeping
            # internal characters like the apostrophe in "can't".
            word = raw.strip(".,!?;:\"'()[]{}…-")
            if word:
                tokens.append(word)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Enhancement implemented here: NEGATION HANDLING.
        A negation word ("not", "never", "no", "don't", "can't") flips the
        polarity of the next mood word. So "not happy" scores -1 instead of +1,
        and "not bad" scores +1 instead of -1.
        """
        tokens = self.preprocess(text)

        negators = {"not", "never", "no", "don't", "dont", "can't", "cant"}

        score = 0
        negate_next = False

        for token in tokens:
            if token in negators:
                # Flip the polarity of the next mood word we encounter.
                negate_next = True
                continue

            # +1 for positive, -1 for negative; a pending negation flips the sign.
            if token in self.positive_words:
                score += -1 if negate_next else 1
                negate_next = False
            elif token in self.negative_words:
                score += 1 if negate_next else -1
                negate_next = False

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
