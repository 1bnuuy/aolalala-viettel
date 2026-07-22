import json

from collections import Counter

from pathlib import Path

ONTOLOGY_PATH = Path("data/ontology.json")


def main():

    with open(
        ONTOLOGY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        ontology = json.load(file)

    print(f"Total concepts: {len(ontology)}")

    types = Counter(
        item.get(
            "type",
            "UNKNOWN",
        )
        for item in ontology
    )

    print("\nEntity types:")

    for entity_type, count in types.most_common():

        print(f"  {entity_type}: {count}")

    print("\nFirst 20 concepts:")

    for item in ontology[:20]:

        print(f"  {item.get('id')} | " f"{item.get('type')} | " f"{item.get('name')}")


if __name__ == "__main__":
    main()
