# src/pipeline.py

from __future__ import annotations


class Pipeline:

    def __init__(
        self,
        ner_model,
        candidate_retriever,
        assertion_model,
    ):

        self.ner_model = ner_model

        self.candidate_retriever = candidate_retriever

        self.assertion_model = assertion_model

    def predict(
        self,
        text: str,
    ) -> list[dict]:

        entities = self.ner_model.predict(text)

        predictions: list[dict] = []

        for entity in entities:

            # -------------------------------------------------
            # Candidate retrieval
            # -------------------------------------------------

            candidates = self.candidate_retriever.search(
                entity_text=entity.text,
                entity_type=entity.type,
            )

            # -------------------------------------------------
            # Assertion detection
            # -------------------------------------------------

            assertions = self.assertion_model.predict(
                text=text,
                start=entity.start,
                end=entity.end,
            )

            # -------------------------------------------------
            # Output
            # -------------------------------------------------

            predictions.append(
                {
                    "text": entity.text,
                    "type": entity.type,
                    "candidates": candidates,
                    "assertions": assertions,
                    "position": [
                        entity.start,
                        entity.end,
                    ],
                }
            )

        return predictions
