# scripts/predict.py

from __future__ import annotations

import json

from pathlib import Path

from src.data.loader import (
    load_input_directory,
)

from src.ner.inference import (
    NERModel,
)

from src.candidates.retrieve import (
    CandidateRetriever,
)

from src.assertions.inference import (
    AssertionModel,
)

from src.pipeline import (
    Pipeline,
)

from src.output import (
    save_prediction,
)

INPUT_DIR = Path("data/input")

ONTOLOGY_PATH = Path("data/ontology.json")

INDEX_PATH = Path("model/candidates/bm25.pkl")

OUTPUT_DIR = Path("output")


def load_ontology() -> list[dict]:

    if not ONTOLOGY_PATH.exists():

        raise FileNotFoundError(f"Ontology not found: " f"{ONTOLOGY_PATH}")

    with open(
        ONTOLOGY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        ontology = json.load(file)

    if not isinstance(
        ontology,
        list,
    ):

        raise ValueError("ontology.json must contain " "a JSON array.")

    return ontology


def main():

    # ---------------------------------------------------------
    # Load ontology
    # ---------------------------------------------------------

    print("Loading ontology...")

    ontology = load_ontology()

    print(f"Loaded {len(ontology)} concepts.")

    # ---------------------------------------------------------
    # Initialize NER
    # ---------------------------------------------------------

    print("Initializing NER...")

    ner_model = NERModel(
        ontology=ontology,
    )

    # ---------------------------------------------------------
    # Candidate index
    # ---------------------------------------------------------

    print("Loading candidate index...")

    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            f"Candidate index not found: "
            f"{INDEX_PATH}\n\n"
            "Run:\n"
            "python -m scripts.build_index"
        )

    candidate_retriever = CandidateRetriever(
        index_path=INDEX_PATH,
        top_k=10,
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    print("Initializing assertion engine...")

    assertion_model = AssertionModel()

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    pipeline = Pipeline(
        ner_model=ner_model,
        candidate_retriever=(candidate_retriever),
        assertion_model=(assertion_model),
    )

    # ---------------------------------------------------------
    # Load TXT files
    # ---------------------------------------------------------

    documents = load_input_directory(INPUT_DIR)

    if not documents:

        raise RuntimeError(f"No .txt files found in " f"{INPUT_DIR}")

    print(f"Found {len(documents)} " f"input files.")

    # ---------------------------------------------------------
    # Clear output
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in OUTPUT_DIR.glob("*.json"):

        old_file.unlink()

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------

    print("\nStarting inference...\n")

    for index, (
        filename,
        text,
    ) in enumerate(
        documents,
        start=1,
    ):

        print(f"[{index}/{len(documents)}] " f"Processing {filename}")

        result = pipeline.predict(text)

        output_path = save_prediction(
            result=result,
            index=index,
            output_dir=OUTPUT_DIR,
        )

        print(f"  Found " f"{len(result)} entities")

        print(f"  Saved: " f"{output_path}")

    print("\nInference completed.")

    print(f"Predictions saved to: " f"{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
