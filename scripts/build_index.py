import json

from pathlib import Path

from src.candidates.index import (
    CandidateIndex,
)

ONTOLOGY_PATH = Path("data/ontology.json")

INDEX_PATH = Path("model/candidates/bm25.pkl")


def main():

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

    print(f"Loaded {len(ontology)} concepts.")

    index = CandidateIndex(ontology)

    index.build()

    index.save(INDEX_PATH)

    print(f"Candidate index saved to: " f"{INDEX_PATH}")


if __name__ == "__main__":
    main()
