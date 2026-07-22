import re

from pathlib import Path

from rapidfuzz.fuzz import ratio

from src.candidates.index import (
    CandidateIndex,
)


class CandidateRetriever:

    def __init__(
        self,
        index_path: str | Path,
        top_k: int = 10,
        fuzzy_threshold: float = 65.0,
    ):

        self.index = CandidateIndex.load(index_path)

        self.top_k = top_k

        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def normalize(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg|ml|mL|%)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b(?:po|iv|im|sc|sl|pr)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b(?:qd|daily|bid|tid|qid|qhs|qam|prn)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def search(
        self,
        entity_text: str,
        entity_type: str | None = None,
    ) -> list[str]:

        normalized_entity = self.normalize(entity_text)

        # --------------------------------------------------------------
        # 1. Exact normalized match
        # --------------------------------------------------------------

        exact_matches = []

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

        # --------------------------------------------------------------
        # 2. BM25 fallback
        # --------------------------------------------------------------

        query_tokens = self.index.tokenize(normalized_entity)

        if not query_tokens:
            return []

        scores = self.index.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

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

            # ----------------------------------------------------------
            # Require some lexical similarity.
            # ----------------------------------------------------------

            similarity = ratio(
                normalized_entity,
                self.normalize(concept_name),
            )

            if similarity < self.fuzzy_threshold:
                continue

            results.append(str(concept["id"]))

            if len(results) >= self.top_k:
                break

        return results
