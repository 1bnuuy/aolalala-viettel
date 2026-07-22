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

        predictions = []

        for entity in entities:

            candidates = self.candidate_retriever.search(
                entity_text=entity.text,
                entity_type=entity.type,
            )

            assertions = self.assertion_model.predict(
                text=text,
                start=entity.start,
                end=entity.end,
            )

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
