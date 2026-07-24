import json

from pathlib import Path

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

TRAIN_PATH = Path("data/generated/ner_train.jsonl")

VALID_PATH = Path("data/generated/ner_valid.jsonl")

OUTPUT_DIR = Path("model/ner")


MODEL_NAME = "xlm-roberta-base"


LABELS = [
    "O",
    "B-THUỐC",
    "I-THUỐC",
    "B-TRIỆU_CHỨNG",
    "I-TRIỆU_CHỨNG",
    "B-BỆNH",
    "I-BỆNH",
]


LABEL2ID = {label: index for index, label in enumerate(LABELS)}


ID2LABEL = {index: label for index, label in enumerate(LABELS)}


def load_jsonl(path):

    examples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            examples.append(json.loads(line))

    return examples


def tokenize_and_align(
    example,
    tokenizer,
):

    text = example["text"]

    entities = example["entities"]

    tokenized = tokenizer(
        text,
        truncation=True,
        return_offsets_mapping=True,
    )

    offsets = tokenized["offset_mapping"]

    labels = [LABEL2ID["O"] for _ in offsets]

    for entity in entities:

        start = entity["start"]

        end = entity["end"]

        entity_type = entity["type"]

        first = True

        for i, (
            token_start,
            token_end,
        ) in enumerate(offsets):

            if token_start == token_end:
                continue

            overlap = token_start < end and token_end > start

            if not overlap:
                continue

            if first:

                labels[i] = LABEL2ID[f"B-{entity_type}"]

                first = False

            else:

                labels[i] = LABEL2ID[f"I-{entity_type}"]

    tokenized.pop("offset_mapping")

    tokenized["labels"] = labels

    return tokenized


def main():

    train_examples = load_jsonl(TRAIN_PATH)

    valid_examples = load_jsonl(VALID_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = Dataset.from_list(train_examples)

    valid_dataset = Dataset.from_list(valid_examples)

    train_dataset = train_dataset.map(
        lambda example: tokenize_and_align(
            example,
            tokenizer,
        ),
        remove_columns=[
            "text",
            "entities",
        ],
    )

    valid_dataset = valid_dataset.map(
        lambda example: tokenize_and_align(
            example,
            tokenizer,
        ),
        remove_columns=[
            "text",
            "entities",
        ],
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=(DataCollatorForTokenClassification(tokenizer=tokenizer)),
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print(
        "NER model saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
