from pathlib import Path

from src.data.loader import load_input_directory
from src.ner.inference import NERModel
from src.candidates.retrieve import CandidateRetriever
from src.assertions.inference import AssertionModel
from src.pipeline import Pipeline
from src.output import save_prediction

# ==============================================================
# PATHS
# ==============================================================

INPUT_DIR = Path("data/input")

INDEX_PATH = Path("model/candidates/bm25.pkl")

OUTPUT_DIR = Path("output")


# ==============================================================
# MAIN
# ==============================================================


def main():

    # ----------------------------------------------------------
    # 1. Initialize NER
    #
    # NER does NOT use ontology.json.
    #
    # It detects:
    #
    # - THUỐC
    # - TRIỆU_CHỨNG
    # - BỆNH
    #
    # The ontology is only used later by CandidateRetriever.
    # ----------------------------------------------------------

    print("Initializing NER...")

    ner_model = NERModel()

    # ----------------------------------------------------------
    # 2. Load candidate retrieval index
    #
    # This index was built from:
    #
    # data/ontology.json
    #
    # It is responsible for mapping detected concepts to
    # candidate IDs.
    # ----------------------------------------------------------

    print("Loading candidate index...")

    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            f"Candidate index not found: {INDEX_PATH}\n\n"
            "Build it first with:\n"
            "python -m scripts.build_index"
        )

    candidate_retriever = CandidateRetriever(
        index_path=INDEX_PATH,
        top_k=10,
    )

    # ----------------------------------------------------------
    # 3. Initialize assertion detection
    # ----------------------------------------------------------

    print("Initializing assertion engine...")

    assertion_model = AssertionModel()

    # ----------------------------------------------------------
    # 4. Build complete pipeline
    # ----------------------------------------------------------

    pipeline = Pipeline(
        ner_model=ner_model,
        candidate_retriever=candidate_retriever,
        assertion_model=assertion_model,
    )

    # ----------------------------------------------------------
    # 5. Load input TXT files
    #
    # Expected:
    #
    # data/
    # └── input/
    #     ├── 1.txt
    #     ├── 2.txt
    #     ├── 3.txt
    #     └── ...
    #
    # Each TXT file becomes one output JSON.
    # ----------------------------------------------------------

    print(f"Loading input files from: {INPUT_DIR}")

    documents = load_input_directory(INPUT_DIR)

    if not documents:

        raise RuntimeError(f"No .txt files found in {INPUT_DIR}")

    print(f"Found {len(documents)} input files.")

    # ----------------------------------------------------------
    # 6. Prepare output directory
    # ----------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove previous predictions.
    # This prevents stale JSON files from remaining
    # in the output directory.
    # ----------------------------------------------------------

    for old_file in OUTPUT_DIR.glob("*.json"):

        old_file.unlink()

    # ----------------------------------------------------------
    # 7. Run inference
    # ----------------------------------------------------------

    print("\nStarting inference...\n")

    for index, (
        filename,
        text,
    ) in enumerate(
        documents,
        start=1,
    ):

        print(f"[{index}/{len(documents)}] " f"Processing {filename}")

        # ------------------------------------------------------
        # Run:
        #
        # TXT
        #   ↓
        # NER
        #   ↓
        # Candidate Retrieval
        #   ↓
        # Assertion Detection
        #   ↓
        # Output objects
        # ------------------------------------------------------

        result = pipeline.predict(text)

        # ------------------------------------------------------
        # Save:
        #
        # output/1.json
        # output/2.json
        # ...
        # ------------------------------------------------------

        save_prediction(
            result=result,
            index=index,
            output_dir=OUTPUT_DIR,
        )

        print(f"  Found {len(result)} entities")

    # ----------------------------------------------------------
    # 8. Done
    # ----------------------------------------------------------

    print("\nInference completed.")

    print(f"Predictions saved to: " f"{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
