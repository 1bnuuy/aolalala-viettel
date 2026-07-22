from pathlib import Path

from src.data.loader import load_input_directory

INPUT_DIR = Path("data/input")


def main():

    documents = load_input_directory(INPUT_DIR)

    print(f"Loaded {len(documents)} documents")

    for filename, text in documents:

        print()
        print("=" * 80)
        print(f"FILE: {filename}")
        print("=" * 80)

        print(f"Character count: {len(text)}")

        print()
        print("First 1000 characters:")
        print("-" * 80)

        print(repr(text[:1000]))

        print()
        print("Medication checks:")
        print("-" * 80)

        for medication in [
            "metoprolol",
            "doxycycline",
            "atenolol",
            "aspirin",
            "amlodipine",
        ]:

            print(f"{medication}: " f"{medication.lower() in text.lower()}")


if __name__ == "__main__":
    main()
