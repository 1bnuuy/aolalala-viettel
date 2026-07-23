from pathlib import Path
import json

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

INPUT_DIR = Path("data/input")

ONTOLOGY_PATH = Path("data/ontology.json")

INDEX_PATH = Path("model/candidates/bm25.pkl")


def main():

    with open(
        ONTOLOGY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        ontology = json.load(file)

    ner = NERModel(ontology=ontology)

    candidates = CandidateRetriever(
        index_path=INDEX_PATH,
        top_k=10,
    )

    assertions = AssertionModel()

    pipeline = Pipeline(
        ner_model=ner,
        candidate_retriever=candidates,
        assertion_model=assertions,
    )

    documents = load_input_directory(INPUT_DIR)

    for filename, text in documents:

        print()
        print("=" * 80)
        print(filename)
        print("=" * 80)

        result = pipeline.predict(text)

        for item in result:

            print(f"\nTEXT: " f"{item['text']}")

            print(f"TYPE: " f"{item['type']}")

            print(f"CANDIDATES: " f"{item['candidates']}")

            print(f"ASSERTIONS: " f"{item['assertions']}")

            print(f"POSITION: " f"{item['position']}")


if __name__ == "__main__":
    main()
