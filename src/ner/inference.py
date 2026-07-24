from dataclasses import dataclass
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


@dataclass
class Entity:

    text: str

    type: str

    start: int

    end: int


class NERModel:

    def __init__(
        self,
        model_path: str = "model/ner",
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(f"NER model not found: " f"{self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

        self.model.eval()

    def predict(
        self,
        text: str,
    ) -> list[Entity]:

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )

        offsets = encoded.pop("offset_mapping")[0]

        with torch.no_grad():

            outputs = self.model(**encoded)

        predictions = outputs.logits.argmax(dim=-1)[0]

        entities = []

        current_type = None

        current_start = None

        current_end = None

        for index, label_id in enumerate(predictions.tolist()):

            label = self.model.config.id2label[label_id]

            token_start = offsets[index][0].item()

            token_end = offsets[index][1].item()

            if token_start == token_end:

                continue

            if label == "O":

                if current_type is not None:

                    entities.append(
                        Entity(
                            text=text[current_start:current_end],
                            type=current_type,
                            start=current_start,
                            end=current_end,
                        )
                    )

                    current_type = None

                    current_start = None

                    current_end = None

                continue

            prefix, entity_type = label.split(
                "-",
                1,
            )

            if prefix == "B":

                if current_type is not None:

                    entities.append(
                        Entity(
                            text=text[current_start:current_end],
                            type=current_type,
                            start=current_start,
                            end=current_end,
                        )
                    )

                current_type = entity_type

                current_start = token_start

                current_end = token_end

            elif prefix == "I" and current_type == entity_type:

                current_end = token_end

            else:

                if current_type is not None:

                    entities.append(
                        Entity(
                            text=text[current_start:current_end],
                            type=current_type,
                            start=current_start,
                            end=current_end,
                        )
                    )

                current_type = entity_type

                current_start = token_start

                current_end = token_end

        if current_type is not None:

            entities.append(
                Entity(
                    text=text[current_start:current_end],
                    type=current_type,
                    start=current_start,
                    end=current_end,
                )
            )

        return entities
