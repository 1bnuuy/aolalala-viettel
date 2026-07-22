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
    ]

    HISTORICAL_PATTERNS = [
        r"tiền sử",
        r"trước khi nhập viện",
        r"thuốc trước khi nhập viện",
        r"tiền sử bệnh",
        r"tiền sử dùng thuốc",
    ]

    def predict(
        self,
        text: str,
        start: int,
        end: int,
    ) -> list[str]:

        context_start = max(
            0,
            start - 200,
        )

        context = text[context_start:end].lower()

        assertions = []

        if self._matches(
            context,
            self.NEGATION_PATTERNS,
        ):
            assertions.append("isNegated")

        if self._matches(
            context,
            self.HISTORICAL_PATTERNS,
        ):
            assertions.append("isHistorical")

        return assertions

    @staticmethod
    def _matches(
        text: str,
        patterns: list[str],
    ) -> bool:

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        )
