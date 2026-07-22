import pickle
import re

from pathlib import Path

from rank_bm25 import BM25Okapi


class CandidateIndex:

    def __init__(
        self,
        ontology: list[dict],
    ):
        self.ontology = ontology

        self.documents: list[str] = []

        self.tokenized_documents = []

        self.bm25 = None

    @staticmethod
    def tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE,
        )

    def build(self):

        self.documents = []

        for concept in self.ontology:

            name = str(
                concept.get(
                    "name",
                    "",
                )
            )

            self.documents.append(name)

        self.tokenized_documents = [
            self.tokenize(document) for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    def save(
        self,
        path: str | Path,
    ):

        if self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "wb",
        ) as file:

            pickle.dump(
                {
                    "ontology": self.ontology,
                    "documents": self.documents,
                    "tokenized_documents": (self.tokenized_documents),
                    "bm25": self.bm25,
                },
                file,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Candidate index not found: {path}")

        with open(
            path,
            "rb",
        ) as file:

            data = pickle.load(file)

        instance = cls(data["ontology"])

        instance.documents = data["documents"]

        instance.tokenized_documents = data["tokenized_documents"]

        instance.bm25 = data["bm25"]

        return instance
