# src/ner/inference.py

from __future__ import annotations

import re

from dataclasses import dataclass

from src.ner.lexicon import (
    get_seed_lexicon,
)


@dataclass
class Entity:
    text: str
    type: str
    start: int
    end: int


class NERModel:

    MEDICATION_INSTRUCTION = re.compile(
        r"""
        (?:
            \s+
            |
            [,:;]
            \s*
        )
        (
            \d+(?:[.,]\d+)?
            \s*
            (?:mg|g|mcg|µg|ml|mL|%|iu)
            (?:\s*-\s*\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|µg|ml|mL|%|iu))?
            |
            \b(?:po|iv|im|sc|sl|pr)\b
            |
            \b(?:qd|daily|bid|tid|qid|qhs|qam)\b
            |
            \bq\d+h\b
            |
            \bprn\b
            |
            \bx\b
        )
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # Generic medical phrases.
    #
    # These are intentionally conservative.
    # They should not match every noun in a medical document.
    MEDICAL_PATTERNS = [
        (
            re.compile(
                r"\bthiếu\s+(?:men|máu|hồng cầu)\b" r"(?:\s+[A-Za-zÀ-ỹ0-9-]+){0,4}",
                re.IGNORECASE,
            ),
            "BỆNH",
        ),
        (
            re.compile(
                r"\b(?:suy|viêm|nhiễm|rối loạn)\s+"
                r"[A-Za-zÀ-ỹ0-9-]+"
                r"(?:\s+[A-Za-zÀ-ỹ0-9-]+){0,4}",
                re.IGNORECASE,
            ),
            "BỆNH",
        ),
    ]

    def __init__(
        self,
        ontology: list[dict] | None = None,
    ):
        self.ontology = ontology or []

        self.ontology_terms = self._build_ontology_terms()

        self.lexicon = get_seed_lexicon()

    # ---------------------------------------------------------
    # ONTOLOGY
    # ---------------------------------------------------------

    def _build_ontology_terms(
        self,
    ) -> list[tuple[str, str]]:

        terms: list[tuple[str, str]] = []

        for concept in self.ontology:

            name = concept.get("name")
            entity_type = concept.get("type")

            if not name or not entity_type:
                continue

            terms.append(
                (
                    str(name),
                    str(entity_type),
                )
            )

        terms.sort(
            key=lambda item: len(item[0]),
            reverse=True,
        )

        return terms

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    def predict(
        self,
        text: str,
    ) -> list[Entity]:

        entities: list[Entity] = []

        # -----------------------------------------------------
        # 1. Ontology terms
        # -----------------------------------------------------

        for term, entity_type in self.ontology_terms:

            entities.extend(
                self._find_term(
                    text=text,
                    term=term,
                    entity_type=entity_type,
                )
            )

        # -----------------------------------------------------
        # 2. Seed lexicon
        # -----------------------------------------------------

        for term, entity_type in self.lexicon:

            entities.extend(
                self._find_term(
                    text=text,
                    term=term,
                    entity_type=entity_type,
                )
            )

        # -----------------------------------------------------
        # 3. Conservative medical patterns
        # -----------------------------------------------------

        for pattern, entity_type in self.MEDICAL_PATTERNS:

            for match in pattern.finditer(text):

                entity = Entity(
                    text=match.group(0).strip(),
                    type=entity_type,
                    start=match.start(),
                    end=match.end(),
                )

                entities.append(entity)

        # -----------------------------------------------------
        # 4. Remove overlaps
        # -----------------------------------------------------

        entities = self._deduplicate(entities)

        # -----------------------------------------------------
        # 5. Expand medications
        # -----------------------------------------------------

        expanded: list[Entity] = []

        for entity in entities:

            if entity.type == "THUỐC":

                entity = self._expand_medication(
                    text=text,
                    entity=entity,
                )

            expanded.append(entity)

        # -----------------------------------------------------
        # 6. Final deduplication
        # -----------------------------------------------------

        expanded = self._deduplicate(expanded)

        expanded.sort(
            key=lambda entity: (
                entity.start,
                entity.end,
            )
        )

        return expanded

    # ---------------------------------------------------------
    # TERM MATCHING
    # ---------------------------------------------------------

    @staticmethod
    def _find_term(
        text: str,
        term: str,
        entity_type: str,
    ) -> list[Entity]:

        pattern = re.compile(
            rf"(?<!\w)" rf"{re.escape(term)}" rf"(?!\w)",
            flags=re.IGNORECASE,
        )

        results: list[Entity] = []

        for match in pattern.finditer(text):

            results.append(
                Entity(
                    text=match.group(0),
                    type=entity_type,
                    start=match.start(),
                    end=match.end(),
                )
            )

        return results

    # ---------------------------------------------------------
    # OVERLAP RESOLUTION
    # ---------------------------------------------------------

    @staticmethod
    def _deduplicate(
        entities: list[Entity],
    ) -> list[Entity]:

        # Longest entities first.
        ordered = sorted(
            entities,
            key=lambda entity: (
                -(entity.end - entity.start),
                entity.start,
            ),
        )

        selected: list[Entity] = []

        for entity in ordered:

            overlaps = False

            for existing in selected:

                if entity.start < existing.end and entity.end > existing.start:
                    overlaps = True
                    break

            if not overlaps:

                selected.append(entity)

        selected.sort(
            key=lambda entity: (
                entity.start,
                entity.end,
            )
        )

        return selected

    # ---------------------------------------------------------
    # MEDICATION EXPANSION
    # ---------------------------------------------------------

    def _expand_medication(
        self,
        text: str,
        entity: Entity,
    ) -> Entity:

        line_end = text.find(
            "\n",
            entity.end,
        )

        if line_end == -1:

            line_end = len(text)

        current_end = entity.end

        while current_end < line_end:

            remaining = text[current_end:line_end]

            match = self.MEDICATION_INSTRUCTION.match(remaining)

            if not match:
                break

            current_end += match.end()

        return Entity(
            text=text[entity.start : current_end].strip(),
            type=entity.type,
            start=entity.start,
            end=current_end,
        )
