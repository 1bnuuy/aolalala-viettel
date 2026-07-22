from pathlib import Path


def load_input_directory(
    input_dir: str | Path,
) -> list[tuple[str, str]]:

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        input_dir.glob("*.txt"),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else path.stem),
    )

    documents: list[tuple[str, str]] = []

    for path in files:
        text = path.read_text(encoding="utf-8")

        documents.append(
            (
                path.name,
                text,
            )
        )

    return documents
