import shutil

from pathlib import Path

OUTPUT_DIR = Path("output")

ZIP_PATH = Path("output.zip")


def main():

    files = sorted(
        OUTPUT_DIR.glob("*.json"),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else path.stem),
    )

    if not files:
        raise RuntimeError("No JSON files found in output/. " "Run prediction first.")

    print(f"Found {len(files)} output files.")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    shutil.make_archive(
        base_name="output",
        format="zip",
        root_dir=".",
        base_dir="output",
    )

    print(f"Created: {ZIP_PATH}")


if __name__ == "__main__":
    main()
