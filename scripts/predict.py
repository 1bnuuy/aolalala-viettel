from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_input_directory
from src.ner.inference import NERModel
from src.candidates.retrieve import CandidateRetriever
from src.assertions.inference import AssertionModel
from src.pipeline import Pipeline
from src.output import save_prediction

# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_INPUT_DIR = Path("data/input")

DEFAULT_OUTPUT_DIR = Path("output")

DEFAULT_NER_MODEL = Path("model/ner")

DEFAULT_INDEX = Path("model/candidates/bm25.pkl")


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=("Run medical concept extraction " "on TXT files.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=("Directory containing input .txt files. " "Default: data/input"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=("Directory for generated JSON files. " "Default: output"),
    )

    parser.add_argument(
        "--ner-model",
        type=Path,
        default=DEFAULT_NER_MODEL,
        help=("Path to trained NER model. " "Default: model/ner"),
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=("Path to BM25 candidate index. " "Default: model/candidates/bm25.pkl"),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():

    args = parse_args()

    input_dir = args.input

    output_dir = args.output

    ner_model_path = args.ner_model

    index_path = args.index

    # ========================================================
    # CHECK INPUT DIRECTORY
    # ========================================================

    if not input_dir.exists():

        raise FileNotFoundError(f"Input directory not found:\n" f"{input_dir}")

    # ========================================================
    # CHECK NER MODEL
    # ========================================================

    if not ner_model_path.exists():

        raise FileNotFoundError(
            f"NER model not found:\n"
            f"{ner_model_path}\n\n"
            "Train the NER model first:\n"
            "python -m scripts.train_ner"
        )

    # ========================================================
    # CHECK CANDIDATE INDEX
    # ========================================================

    if not index_path.exists():

        raise FileNotFoundError(
            f"Candidate index not found:\n"
            f"{index_path}\n\n"
            "Build the candidate index first:\n"
            "python -m scripts.build_index"
        )

    # ========================================================
    # INITIALIZE NER
    # ========================================================

    print("Loading NER model...")

    ner_model = NERModel(model_path=str(ner_model_path))

    # ========================================================
    # INITIALIZE CANDIDATE RETRIEVER
    # ========================================================

    print("Loading candidate index...")

    candidate_retriever = CandidateRetriever(
        index_path=index_path,
        top_k=10,
    )

    # ========================================================
    # INITIALIZE ASSERTION MODEL
    # ========================================================

    print("Initializing assertion engine...")

    assertion_model = AssertionModel()

    # ========================================================
    # CREATE PIPELINE
    # ========================================================

    pipeline = Pipeline(
        ner_model=ner_model,
        candidate_retriever=candidate_retriever,
        assertion_model=assertion_model,
    )

    # ========================================================
    # LOAD INPUT TXT FILES
    # ========================================================

    print(f"\nLoading input files from:\n" f"{input_dir.resolve()}")

    documents = load_input_directory(input_dir)

    if not documents:

        raise RuntimeError(f"No .txt files found in:\n" f"{input_dir}")

    print(f"Found {len(documents)} " f"input files.")

    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old predictions.

    for old_file in output_dir.glob("*.json"):

        old_file.unlink()

    # ========================================================
    # RUN INFERENCE
    # ========================================================

    print("\nStarting inference...\n")

    for index, (
        filename,
        text,
    ) in enumerate(
        documents,
        start=1,
    ):

        print(f"[{index}/{len(documents)}] " f"Processing {filename}")

        # Run NER
        # Candidate mapping
        # Assertion detection

        result = pipeline.predict(text)

        # Save:
        #
        # output/1.json
        # output/2.json
        # ...
        #
        output_path = save_prediction(
            result=result,
            index=index,
            output_dir=output_dir,
        )

        print(f"  Found " f"{len(result)} entities")

        print(f"  Saved: " f"{output_path}")

    # ========================================================
    # DONE
    # ========================================================

    print("\nInference completed.")

    print("Predictions saved to:")

    print(output_dir.resolve())


if __name__ == "__main__":
    main()
