from pathlib import Path

from src.ner.inference import NERModel
from src.data.loader import load_input_directory

INPUT_DIR = Path("data/input")


def main():

    model = NERModel()

    documents = load_input_directory(INPUT_DIR)

    for filename, text in documents:

        print()
        print("=" * 80)
        print(filename)
        print("=" * 80)

        entities = model.predict(text)

        for entity in entities:

            print(
                f"{entity.type:15} "
                f"{entity.start:5} "
                f"{entity.end:5} "
                f"{entity.text!r}"
            )


if __name__ == "__main__":
    main()
