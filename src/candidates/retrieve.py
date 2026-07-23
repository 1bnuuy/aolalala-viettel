# src/candidates/retrieve.py

from __future__ import annotations

import re

from pathlib import Path

from rapidfuzz.fuzz import (
    ratio,
    token_set_ratio,
)

from src.candidates.index import (
    CandidateIndex,
)


class CandidateRetriever:

    def __init__(
        self,
        index_path: str | Path,
        top_k: int = 10,
        fuzzy_threshold: float = 75.0,
    ):

        self.index = CandidateIndex.load(index_path)

        self.top_k = top_k

        self.fuzzy_threshold = fuzzy_threshold

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def normalize(
        text: str,
    ) -> str:

        text = text.lower()

        # Normalize decimal separators.
        text = text.replace(
            ",",
            ".",
        )

        # Remove medication dosage.
        text = re.sub(
            r"\b\d+(?:\.\d+)?\s*" r"(?:mg|g|mcg|µg|ml|mL|%|iu)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Remove dosage ranges.
        text = re.sub(
            r"\b\d+(?:\.\d+)?\s*-\s*" r"\d+(?:\.\d+)?\s*" r"(?:mg|g|mcg|µg|ml|mL|%)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Route.
        text = re.sub(
            r"\b(?:po|iv|im|sc|sl|pr)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Frequency.
        text = re.sub(
            r"\b(?:qd|daily|bid|tid|qid|qhs|qam|prn)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # q6h, q8h, etc.
        text = re.sub(
            r"\bq\d+h\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def search(
        self,
        entity_text: str,
        entity_type: str | None = None,
    ) -> list[str]:

        normalized_entity = self.normalize(entity_text)

        if not normalized_entity:

            return []

        # -----------------------------------------------------
        # Exact normalized match
        # -----------------------------------------------------

        exact_matches: list[str] = []

        for concept in self.index.ontology:

            concept_type = concept.get("type")

            if entity_type is not None and concept_type != entity_type:
                continue

            concept_name = self.normalize(
                str(
                    concept.get(
                        "name",
                        "",
                    )
                )
            )

            if concept_name == normalized_entity:

                exact_matches.append(str(concept["id"]))

        if exact_matches:

            return exact_matches[: self.top_k]

        # -----------------------------------------------------
        # BM25
        # -----------------------------------------------------

        query_tokens = self.index.tokenize(normalized_entity)

        if not query_tokens:

            return []

        scores = self.index.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[str] = []

        for index in ranked_indices:

            concept = self.index.ontology[index]

            concept_type = concept.get("type")

            if entity_type is not None and concept_type != entity_type:
                continue

            concept_name = str(
                concept.get(
                    "name",
                    "",
                )
            )

            normalized_name = self.normalize(concept_name)

            char_similarity = ratio(
                normalized_entity,
                normalized_name,
            )

            token_similarity = token_set_ratio(
                normalized_entity,
                normalized_name,
            )

            similarity = max(
                char_similarity,
                token_similarity,
            )

            if similarity < self.fuzzy_threshold:
                continue

            results.append(str(concept["id"]))

            if len(results) >= self.top_k:

                break

        return results
