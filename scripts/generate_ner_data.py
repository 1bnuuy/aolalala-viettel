import json
import random
from pathlib import Path

ONTOLOGY_PATH = Path("data/ontology.json")
OUTPUT_DIR = Path("data/generated")

TRAIN_PATH = OUTPUT_DIR / "ner_train.jsonl"
VALID_PATH = OUTPUT_DIR / "ner_valid.jsonl"


DRUG_TEMPLATES = [
    "Bệnh nhân đang dùng {concept}.",
    "Bệnh nhân được kê {concept}.",
    "Người bệnh sử dụng {concept}.",
    "Đang điều trị bằng {concept}.",
    "Thuốc hiện tại gồm {concept}.",
    "Bệnh nhân uống {concept}.",
    "Tiền sử dùng thuốc {concept}.",
    "Trước khi nhập viện bệnh nhân dùng {concept}.",
    "Bệnh nhân được chỉ định {concept}.",
    "Tiếp tục sử dụng {concept}.",
]


SYMPTOM_TERMS = [
    "đánh trống ngực",
    "khó thở",
    "đau ngực",
    "đau đầu",
    "mệt mỏi",
    "buồn nôn",
    "nôn",
    "sốt",
    "ho",
    "đau nhức",
    "lo âu",
    "mất ngủ",
    "táo bón",
    "đau bụng",
    "chóng mặt",
]


SYMPTOM_TEMPLATES = [
    "Bệnh nhân xuất hiện {concept}.",
    "Người bệnh có triệu chứng {concept}.",
    "Bệnh nhân than phiền {concept}.",
    "Ghi nhận {concept}.",
    "Hiện tại bệnh nhân có {concept}.",
    "Người bệnh cảm thấy {concept}.",
]


DISEASE_TERMS = [
    "tăng huyết áp",
    "đái tháo đường",
    "viêm phổi",
    "suy tim",
    "thiếu máu",
    "viêm gan",
    "thiếu men G6PD",
]


DISEASE_TEMPLATES = [
    "Bệnh nhân có tiền sử {concept}.",
    "Bệnh nhân được chẩn đoán {concept}.",
    "Người bệnh mắc {concept}.",
    "Tiền sử bệnh lý ghi nhận {concept}.",
    "Chẩn đoán hiện tại là {concept}.",
]


def load_ontology():
    with ONTOLOGY_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def make_example(
    concept: str,
    entity_type: str,
    template: str,
):
    text = template.format(
        concept=concept,
    )

    start = text.index(concept)
    end = start + len(concept)

    return {
        "text": text,
        "entities": [
            {
                "start": start,
                "end": end,
                "type": entity_type,
            }
        ],
    }


def generate_examples():

    ontology = load_ontology()

    examples = []

    # --------------------------------------------------
    # Drugs from ontology
    # --------------------------------------------------

    for concept in ontology:

        name = concept.get("name")
        entity_type = concept.get("type")

        if not name or not entity_type:
            continue

        if entity_type != "THUỐC":
            continue

        for template in DRUG_TEMPLATES:

            examples.append(
                make_example(
                    concept=name,
                    entity_type="THUỐC",
                    template=template,
                )
            )

    # --------------------------------------------------
    # Symptoms
    # --------------------------------------------------

    for concept in SYMPTOM_TERMS:

        for template in SYMPTOM_TEMPLATES:

            examples.append(
                make_example(
                    concept=concept,
                    entity_type="TRIỆU_CHỨNG",
                    template=template,
                )
            )

    # --------------------------------------------------
    # Diseases
    # --------------------------------------------------

    for concept in DISEASE_TERMS:

        for template in DISEASE_TEMPLATES:

            examples.append(
                make_example(
                    concept=concept,
                    entity_type="BỆNH",
                    template=template,
                )
            )

    random.shuffle(examples)

    return examples


def save_jsonl(
    path: Path,
    examples,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples = generate_examples()

    split = int(len(examples) * 0.9)

    train_examples = examples[:split]
    valid_examples = examples[split:]

    save_jsonl(
        TRAIN_PATH,
        train_examples,
    )

    save_jsonl(
        VALID_PATH,
        valid_examples,
    )

    print(f"Generated {len(examples)} examples.")

    print(f"Train: {len(train_examples)}")

    print(f"Valid: {len(valid_examples)}")


if __name__ == "__main__":
    main()
