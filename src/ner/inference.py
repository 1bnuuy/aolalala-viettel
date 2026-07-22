import re

from dataclasses import dataclass


@dataclass
class Entity:
    text: str
    type: str
    start: int
    end: int


class NERModel:

    # ==========================================================
    # MEDICATIONS
    # ==========================================================

    MEDICATIONS = {
        "aspirin",
        "amlodipine",
        "metoprolol",
        "metoprolol succinate",
        "guaifenesin",
        "nystatin",
        "acetaminophen",
        "pravastatin",
        "docusate sodium",
        "senna",
        "clonazepam",
        "doxycycline",
        "atenolol",
        "amoxicillin",
        "ibuprofen",
        "paracetamol",
        "warfarin",
        "heparin",
        "insulin",
        "atorvastatin",
        "simvastatin",
        "losartan",
        "lisinopril",
        "furosemide",
        "omeprazole",
        "pantoprazole",
    }

    # ==========================================================
    # SYMPTOMS
    # ==========================================================

    SYMPTOMS = {
        "đánh trống ngực",
        "khó thở",
        "mệt mỏi",
        "đau ngực",
        "thắt chặt ngực",
        "cảm giác thắt chặt ngực",
        "buồn nôn",
        "nôn",
        "đổ mồ hôi",
        "sốt",
        "đau",
        "đau nhức",
        "lo âu",
        "mất ngủ",
        "táo bón",
        "giảm dung nạp gắng sức",
        "ho",
    }

    # ==========================================================
    # DISEASES
    # ==========================================================

    DISEASES = {
        "viêm tuyến mồ hôi",
        "viêm gan cấp tính do virus B",
    }

    # ==========================================================
    # MEDICATION INSTRUCTION
    # ==========================================================

    MEDICATION_INSTRUCTION = re.compile(
        r"""
        (?:
            \s+
            |
            [,:;]
            \s*
        )
        (
            \d+(?:\.\d+)?
            \s*
            (?:mg|g|mcg|µg|ml|mL|%)

            |

            \b(?:po|iv|im|sc|sl|pr)\b

            |

            \b(?:qd|daily|bid|tid|qid|qhs|qam)\b

            |

            \bq\d+h\b

            |

            \bprn\b
        )
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self):

        self.lexicon = self._build_lexicon()

    # ==========================================================
    # BUILD LEXICON
    # ==========================================================

    def _build_lexicon(self):

        terms = []

        for term in self.MEDICATIONS:

            terms.append(
                (
                    term,
                    "THUỐC",
                )
            )

        for term in self.SYMPTOMS:

            terms.append(
                (
                    term,
                    "TRIỆU_CHỨNG",
                )
            )

        for term in self.DISEASES:

            terms.append(
                (
                    term,
                    "BỆNH",
                )
            )

        # Longest first.
        #
        # Example:
        #
        # "cảm giác thắt chặt ngực"
        #
        # should be matched before:
        #
        # "thắt chặt ngực"

        terms.sort(
            key=lambda x: len(x[0]),
            reverse=True,
        )

        return terms

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(
        self,
        text: str,
    ):

        entities = []

        # ------------------------------------------------------
        # Detect every lexicon term
        # ------------------------------------------------------

        for term, entity_type in self.lexicon:

            pattern = re.compile(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                flags=re.IGNORECASE,
            )

            for match in pattern.finditer(text):

                entities.append(
                    Entity(
                        text=match.group(0),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # ------------------------------------------------------
        # Remove overlapping entities
        # ------------------------------------------------------

        entities = self._deduplicate(entities)

        # ------------------------------------------------------
        # Expand medication mentions
        # ------------------------------------------------------

        expanded = []

        for entity in entities:

            if entity.type == "THUỐC":

                entity = self._expand_medication(
                    text,
                    entity,
                )

            expanded.append(entity)

        # ------------------------------------------------------
        # Final deduplication
        # ------------------------------------------------------

        expanded = self._deduplicate(expanded)

        expanded.sort(
            key=lambda x: (
                x.start,
                x.end,
            )
        )

        return expanded

    # ==========================================================
    # DEDUPLICATION
    # ==========================================================

    @staticmethod
    def _deduplicate(
        entities,
    ):

        # Longer entities first.

        ordered = sorted(
            entities,
            key=lambda x: (
                -(x.end - x.start),
                x.start,
            ),
        )

        selected = []

        for entity in ordered:

            overlap = False

            for existing in selected:

                if entity.start < existing.end and entity.end > existing.start:

                    overlap = True
                    break

            if not overlap:

                selected.append(entity)

        selected.sort(key=lambda x: x.start)

        return selected

    # ==========================================================
    # MEDICATION EXPANSION
    # ==========================================================

    def _expand_medication(
        self,
        text,
        entity,
    ):

        # Only expand within the same line.

        newline = text.find(
            "\n",
            entity.end,
        )

        if newline == -1:

            line_end = len(text)

        else:

            line_end = newline

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
