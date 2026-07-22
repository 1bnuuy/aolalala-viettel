from pathlib import Path

from src.candidates.retrieve import (
    CandidateRetriever,
)

INDEX_PATH = Path("model/candidates/bm25.pkl")


def main():

    retriever = CandidateRetriever(
        index_path=INDEX_PATH,
        top_k=10,
    )

    test_entities = [
        (
            "metoprolol 25mg po bid",
            "THUỐC",
        ),
        (
            "aspirin 81 mg po daily",
            "THUỐC",
        ),
        (
            "đánh trống ngực",
            "TRIỆU_CHỨNG",
        ),
    ]

    for text, entity_type in test_entities:

        candidates = retriever.search(
            entity_text=text,
            entity_type=entity_type,
        )

        print(f"Entity: {text}")

        print(f"Type: {entity_type}")

        print(f"Candidates: {candidates}")

        print("-" * 60)


if __name__ == "__main__":
    main()
