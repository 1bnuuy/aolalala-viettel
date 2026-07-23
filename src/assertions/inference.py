# src/assertions/inference.py

from __future__ import annotations

import re


class AssertionModel:

    NEGATION_PATTERNS = [
        r"\bkhông\b",
        r"\bkhông có\b",
        r"\bchưa\b",
        r"\bchưa từng\b",
        r"\bphủ nhận\b",
        r"\bkhông ghi nhận\b",
        r"\bkhông đáng kể\b",
        r"\bkhông có gì bất thường\b",
        r"\bkhông thấy\b",
        r"\bkhông phát hiện\b",
    ]

    HISTORICAL_PATTERNS = [
        r"\btiền sử\b",
        r"\btrước khi nhập viện\b",
        r"\bthuốc trước khi nhập viện\b",
        r"\btiền sử bệnh\b",
        r"\btiền sử dùng thuốc\b",
        r"\bđã từng\b",
        r"\btừng\b",
    ]

    def predict(
        self,
        text: str,
        start: int,
        end: int,
    ) -> list[str]:

        sentence = self._get_sentence_context(
            text=text,
            start=start,
            end=end,
        )

        entity_text = text[start:end]

        before_entity = sentence[
            : max(
                0,
                start
                - self._sentence_start(
                    text,
                    start,
                ),
            )
        ]

        assertions: list[str] = []

        # -----------------------------------------------------
        # Negation
        # -----------------------------------------------------

        if self._matches(
            before_entity,
            self.NEGATION_PATTERNS,
        ):

            assertions.append("isNegated")

        # -----------------------------------------------------
        # Historical
        # -----------------------------------------------------

        if self._matches(
            sentence,
            self.HISTORICAL_PATTERNS,
        ):

            # Don't automatically mark all symptoms
            # in a historical section as historical.
            #
            # This is especially useful for medications.
            #
            # The competition example indicates that
            # "Thuốc trước nhập viện" causes medication
            # assertions to be historical.

            medication_context = self._matches(
                sentence,
                [
                    r"\bthuốc\b",
                    r"\bdùng thuốc\b",
                    r"\bsử dụng\b",
                    r"\bđơn thuốc\b",
                ],
            )

            if medication_context:

                assertions.append("isHistorical")

        return assertions

    # ---------------------------------------------------------
    # SENTENCE EXTRACTION
    # ---------------------------------------------------------

    @staticmethod
    def _get_sentence_context(
        text: str,
        start: int,
        end: int,
    ) -> str:

        sentence_start = AssertionModel._sentence_start(
            text,
            start,
        )

        sentence_end = AssertionModel._sentence_end(
            text,
            end,
        )

        return text[sentence_start:sentence_end].lower()

    @staticmethod
    def _sentence_start(
        text: str,
        position: int,
    ) -> int:

        boundaries = [
            text.rfind(
                ".",
                0,
                position,
            ),
            text.rfind(
                "\n",
                0,
                position,
            ),
            text.rfind(
                "!",
                0,
                position,
            ),
            text.rfind(
                "?",
                0,
                position,
            ),
            text.rfind(
                ":",
                0,
                position,
            ),
        ]

        return max(boundaries) + 1

    @staticmethod
    def _sentence_end(
        text: str,
        position: int,
    ) -> int:

        boundaries = []

        for character in (
            ".",
            "\n",
            "!",
            "?",
        ):

            index = text.find(
                character,
                position,
            )

            if index != -1:

                boundaries.append(index)

        if not boundaries:

            return len(text)

        return min(boundaries) + 1

    @staticmethod
    def _matches(
        text: str,
        patterns: list[str],
    ) -> bool:

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )
